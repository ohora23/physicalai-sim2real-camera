"""Virtual sensor (Phase 4 R5): GL-only sim -> dma-buf -> CUDA effects process -> Axon -> policy.

Three spawned processes share a perf_counter (CLOCK_MONOTONIC) epoch ``t0``:

- SIM (GL/EGL only; never imports torch or cuda, checked in its own ``sys.modules``): board scene,
  N envs (``MjData``) sharing one ``mujoco.Renderer`` (offsamples=0, render size ``H_r x W_r`` and
  ``fovy_r`` from a CPU ``SimCamera``), ``CameraStreamBatch(cfg, N, render=hook)`` stepped at 500 Hz
  wall-clock (sim time == wall time since t0, as examples/axon_bridge.py). The hook renders env i into
  dma-buf slot ``flat_slot(i, r_i, Ks)`` of a ``DmabufExporter`` with ``N * Ks`` slots (``--sim-slots``
  Ks = 2: the virtual sensor copies a slot out when its note arrives and holds the frame in its own
  ``ring_depth(cfg)``-deep output ring; there is no release fence back, so a note older than
  ``(Ks - 1)/fps`` is an error in the virtual sensor); after the tick every camera that fired sends
  ``NOTE = (cam, slot, n, capture_t, ready_t)`` + its native fence fd over a SOCK_SEQPACKET Unix
  socket (the slot fds went over once at start, SCM_RIGHTS). ``ready_t`` comes from
  ``CameraStream.last_render``. Logs its own CameraStreamBatch ages -> ``pub_age.csv`` (the truth).
- VIRTUAL SENSOR (CUDA): ``DmabufImporter`` + ``SimCameraBatch(renderer=False)`` (no GL context).
  On each batch of notifications: wait fences, then — if a publish is due within ``--publish-guard-ms`` —
  publish first (``publish_first``), then ``read`` the raw slot, ``ingest`` it, one ``process()`` for all of
  them (flushed early if a camera repeats after a stall, ``pending(i)``);
  then hold each processed frame until ``t0 + ready_t`` and publish it
  to Axon topic ``realsensor/head_cam/<i>`` (one topic per camera: ``latest_view`` returns only the
  newest message of a topic) as ``frame | capture_t f64 | n i64`` (the 16-byte trailer of
  axon_bridge.py). The sensor latency is modelled here, not in transport: the raw frame arrives at
  capture time, the effects run inside the latency budget. A frame older than one already published
  for that camera is dropped (same "never go back in time" rule as ``CameraStream``). Default pool:
  Axon ``Accelerator`` (CUDA VMM, ``publish_device``: the D2D copy + trailer go straight into the pool
  buffer, no host copy); ``--pool custom|udmabuf`` copies to host instead. Logs
  ``(cam, n, capture_t, ready_t, t_publish)`` -> ``vs_publish.csv``.
- POLICY (Axon only; no EGL/GL/torch/mujoco, checked): one subscriber per camera polled at 500 Hz
  wall-clock, ``POLL_LAG`` after the tick; reads the trailer (device frames: 16-byte
  ``cuMemcpyDtoH`` via cuda-python, host frames: the view) -> ``sub_age.csv``.

Window mode (``--exposure-ms``/``--readout-ms`` > 0): the SIM keeps a ``StateRing`` per env
(``CameraStreamBatch(cams=rings)`` pushes it every step) and renders each capture as a group of ``n_sub``
colour sub-renders at ``SimCamera``'s plan times into consecutive exporter slots ``g*n_sub + k`` (group
``g = flat_slot(i, r_i, Ks)``, the NOTE slot), plus with ``--depth`` one depth render at ``capture_t`` in slot
``g`` of a second, depth-only exporter; only the group's last render carries a fence. The VIRTUAL SENSOR
reads the group into an ``(n_sub, H_r, W_r, 3)`` staging tensor and ``ingest``s it (the batch accumulates
the sub-frames). Descriptors go over the socket in chunks of at most 250 fds (``send_fds``/``recv_descs``).
With ``--stagger`` camera i uses ``phase_s = cfg.phase_s + i/(N*fps)`` (``stagger_phases``), so at most
``ceil(DT*N*fps)`` cameras fire on one tick (1 for N <= 16 at 30 fps, DT 2 ms); the virtual sensor and policy are
unchanged (NOTE carries capture_t/ready_t).

Prism scene (``--scene prism``): the SIM loads ``scenes.prism_scene`` (prism yawed -60 deg, as the O6
closed loop) and drives every env's mocap head with the shared ``HeadSweep``: after ``mj_step`` it sets the pose of
``d.time`` and runs ``mj_kinematics`` + ``mj_camlight``, so a capture renders the pose of its own time (the order of
``observer_replay.sim_records``). The VIRTUAL SENSOR loads the same scene (same camera, same depth range).

Live observer (``--observer``, prism only): a 4th process, OBSERVER, subscribes to one camera's topic
(``--observer-cams``, default 0; one camera only) and runs ``observer_replay.Observer`` on it. The SIM
writes the head state the robot would report (env 0: the true pose 4 ms ago plus noise, ``write_head``) into a
``HeadPoseRing`` the parent owns. The observer runs at ``nice 10`` with one OpenCV thread; its main thread polls
``latest_view`` on the DT grid and copies each new frame + trailer to host (Accelerator pools: one ``cuMemcpyDtoH``
of the payload, ~0.9 MB per frame, in a cuda-python context: the observer uses the GPU only for that copy), stamps
``t_arrival`` and replaces the true capture time by a sensor timestamp (``sensor_clock``); a worker thread pushes the
records into the ``Observer`` (a full queue drops the record: ``worker_drops``) and at most once per second of arrival
time estimates (``calibrateCamera`` runs there) and writes ``<out>/camera_config_estimate.json`` atomically
(``estimate_summary``).

The parent reads ``nvidia-smi`` while all three (four with the observer) wait at the end, computes KS (tick-timed
age vs sim age, per camera and pooled), seqno coverage and memory, and writes ``out/virtual_sensor/summary.json``.

Run: MUJOCO_GL=egl .venv/bin/python examples/virtual_sensor.py [--seconds 5] [--n 4] [--depth] [--pool accelerator]
     [--exposure-ms 8 --readout-ms 16] [--scene prism [--observer]]
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco` (children inherit it)

import argparse
import csv
import gc
import heapq
import json
import math
import multiprocessing as mp
import multiprocessing.connection as mpc
import re
import select
import socket
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.append(os.path.join(ROOT, "third_party", "axon", "build", "python"))  # fallback if PYTHONPATH unset

TOPIC = "realsensor/head_cam/{}"
W, H = 640, 480
FRAME = H * W * 3
TRAILER = struct.Struct("<dq")    # capture_t float64, n int64 (== axon_bridge.py's 16-byte trailer)
NOTE = struct.Struct("<iiqdd")    # cam (-1 = end), slot, n, capture_t, ready_t
DT = 0.002                        # 500 Hz control / poll rate
POLL_LAG = 0.0005                 # policy poll phase after the tick (see axon_bridge.py)
POOL_BUFFERS = 4                  # per camera; frames are ~33 ms apart, a reader holds one for < 2 ms
VMM_GRAN = 2 << 20                # CUDA VMM allocation granularity (Accelerator pool buffers round up to it)
OUT = os.path.join(ROOT, "out", "virtual_sensor")  # one sub-directory per --run-name
PUBLISH_GUARD_MS = 1.5             # publish-first lookahead, ms: ~ the note segment p99 (V1 baseline 1.3-1.5 ms)
PRISM_XML = os.path.join(ROOT, "assets", "prism", "prism.xml")
PRISM_YAW_DEG = -60.0              # wedge face readable on one side of the orbit (as the O6 closed loop)
OBS_FOVY_DEG = 45.0                # the observer's nominal FOV (spec sheet / MJCF) for its initial K
OBS_QUEUE = 32                     # receive -> worker records (~0.9 MB each); a full queue drops one (worker_drops)
OBS_EVERY_S = 1.0                  # estimate (calibrateCamera) at most once per second of arrival time


def flat_slot(i, r, K):
    """Exporter slot of camera i's render r: the ring key ``(i, r % K)`` flattened to ``i * K + r % K``."""
    return i * K + r % K


def colour_slot(g, k, n_sub):
    """Colour exporter slot of sub-frame ``k`` of capture group ``g`` (window mode; ``g * n_sub + k``)."""
    return g * n_sub + k


def set_scene(r, model, cam, state):
    """``update_scene`` for ``state`` at the camera's render fovy (as ``SimCamera._raw``); model restored after."""
    cid = cam.cam
    fovy0, ss0 = float(model.cam_fovy[cid]), model.cam_sensorsize[cid].copy()
    try:
        model.cam_sensorsize[cid] = 0  # sensorsize cameras ignore cam_fovy
        model.cam_fovy[cid] = cam.fovy_r
        r.update_scene(state, camera=cid)  # fovy is read here
    finally:
        model.cam_fovy[cid], model.cam_sensorsize[cid] = fovy0, ss0


def render_group(ex, exz, r, model, cam, ring, scratch, t, g):
    """Render capture group ``g`` of a window-mode camera whose window ends at sim time ``t`` (the sim hook).

    Colour sub-frame k (``cam._plan``, times ``t - half + dt_k``) goes to ``ex`` slot ``colour_slot(g, k, n_sub)``;
    with a depth exporter ``exz`` one depth render at ``t - half`` goes to its slot ``g``. Only the group's last
    render creates a fence (one GL context: the earlier renders are ordered before it).

    Args:
        ex: Colour ``DmabufExporter`` with at least ``(g + 1) * n_sub`` slots.
        exz: Depth ``DmabufExporter`` or None.
        r: The shared ``mujoco.Renderer``.
        model: Its ``MjModel``.
        cam: CPU window-mode ``SimCamera`` (plan, render fovy).
        ring: The env's ``StateRing``.
        scratch: ``MjData`` the ring loads into.
        t: Sim time the window ends at (``data.time`` when the capture fires).
        g: Capture group (the NOTE slot).

    Returns:
        ``(fd, n_renders)``: the group's fence fd (None after ``glFinish``) and the number of GL renders.
    """
    tc, n_sub, fd, renders = t - cam.half_window, len(cam._plan), None, 0
    for k, (dt, _) in enumerate(cam._plan):
        ring.load(tc + dt, scratch)
        set_scene(r, model, cam, scratch)
        fd = ex.render(colour_slot(g, k, n_sub), depth=False, fence=k == n_sub - 1 and exz is None)
        renders += 1
    if exz is not None:
        ring.load(tc, scratch)
        set_scene(r, model, cam, scratch)
        fd = exz.render(g, depth=True, fence=True)
        renders += 1
    return fd, renders


def publish_time(ready_t):
    """When the virtual sensor publishes a frame that becomes ready at ``ready_t`` (s since t0).

    ``CameraStream`` delivers it at the first control tick ``T = k*DT >= ready_t`` (1 ns tolerance), and the
    policy polls ``POLL_LAG`` after each tick, so any publish in ``(T - DT + POLL_LAG, T + POLL_LAG]`` shows
    the policy exactly the sim's frame. Publishing at ``ready_t`` itself lands in the first ``POLL_LAG`` of
    that window a quarter of the time (then the previous tick's poll sees it one tick early); instead publish
    at ``max(ready_t, T - DT/2)``: never before ``ready_t``, never before the previous poll, and at least
    ``POLL_LAG`` of slack before the tick's poll.

    Args:
        ready_t: Frame-available time (``capture_t + latency``), seconds since t0.

    Returns:
        Publish time, seconds since t0.
    """
    tick = math.ceil((ready_t - 1e-9) / DT) * DT
    return max(ready_t, tick - DT / 2)


def publish_first(due_t, now, guard_s):
    """True when the earliest due publish (``due_t``, seconds since t0) should run before a new note segment.

    A note segment (dma-buf reads, ingest, process) takes ~1 ms; starting it when a publish is due within
    ``guard_s`` would push that publish past its tick. Waiting at most ``guard_s`` costs the note nothing
    against its 50 ms latency budget.

    Args:
        due_t: Earliest due time in the heap, seconds since t0.
        now: Current time, seconds since t0.
        guard_s: Lookahead window, seconds (``--publish-guard-ms`` / 1e3).

    Returns:
        bool: Handle the publish first.
    """
    return due_t <= now + guard_s


def payload_size(depth):
    """Axon payload bytes: RGB frame (+ float32 depth) + 16-byte trailer."""
    return FRAME + (H * W * 4 if depth else 0) + TRAILER.size


def make_cfg(a=None):
    """Camera config shared by SIM (timing, render geometry) and VIRTUAL SENSOR (effects).

    Args:
        a: Parsed arguments; ``a.exposure_ms`` / ``a.readout_ms`` > 0 turn on window mode (motion blur /
            rolling shutter, ``CameraConfig`` defaults of 4 blur samples x 8 bands). None: no window.

    Returns:
        The ``CameraConfig`` all three processes use.
    """
    from physicalai_sim2real_camera import CameraConfig
    e, r = (a.exposure_ms / 1e3, a.readout_ms / 1e3) if a is not None else (0.0, 0.0)
    return CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0,
                        sigma_shot=0.02, sigma_read=0.005, black_level=0.02,
                        fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.05,
                        exposure_s=e, readout_s=r)


def scene_model(a):
    """Model of ``--scene`` (camera ``cam`` in both) and the head sweep that drives it.

    Args:
        a: Parsed arguments (``a.scene``).

    Returns:
        ``(model, sweep)``: the board scene with ``sweep=None``, or ``scenes.prism_scene`` with its ``HeadSweep``.
    """
    if a.scene == "board":
        from test_camera import board_scene
        return board_scene()[0], None
    from physicalai_sim2real_camera.scenes import prism_scene
    model, _, sweep = prism_scene(PRISM_XML, width=W, height=H, prism_yaw_deg=PRISM_YAW_DEG)
    # the sim's prism step runs mj_kinematics + mj_camlight only: right for fixed-mode cameras and lights
    assert (model.cam_mode == 0).all() and (model.light_mode == 0).all(), "tracking camera / light needs mj_forward"
    return model, sweep


def _num(x):
    """JSON-safe float: None for missing or non-finite values."""
    return float(x) if x is not None and np.isfinite(x) else None


def estimate_summary(est, counters=None):
    """Flat, JSON-safe summary of an ``Observer.estimate`` dict (``camera_config_estimate.json``).

    Same spirit as ``calib_uncertainty.json``: per estimated field its ``mean`` and ``std`` (``std`` None when the
    observer gives none), the sample counts behind each group, the geometric coverage and the observer's health.

    Args:
        est: ``observer_replay.Observer.estimate`` result (``geom`` / ``photo`` may be None).
        counters: Extra health counters (receive / worker side), merged into ``health``.

    Returns:
        dict with ``fx fy cx cy k1 k2 p1 p2 k3`` (when the geometric fit exists), ``sigma_shot sigma_read`` (when
        the photometric fit has them), ``latency_s jitter_theta camera_drop_rate`` (when timing has them), each
        ``{"mean", "std"}``; ``n`` = ``{"geom": n_views, "photo": n_intervals, "timing": n_frames}``; ``coverage`` =
        fraction of non-empty coverage cells; ``geom_rms`` = the fit's RMS reprojection error, px (None without a
        fit); ``photo_insufficient_levels`` = the photometric fit's flag (True: its sigmas, if any, are not to be
        used; None without a fit); ``health`` = the estimate's health dict plus ``counters``.
    """
    g, p, tm = est.get("geom"), est.get("photo") or {}, est.get("timing") or {}
    out = {}
    if g is not None:
        K, d = np.asarray(g["K"]), np.asarray(g["dist"]).ravel()
        vals = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4])
        out.update({f: dict(mean=_num(v), std=_num(g["std"].get(f))) for f, v in vals.items()})
    for f, src, sk in (("sigma_shot", p, "sigma_shot_std"), ("sigma_read", p, "sigma_read_std"),
                       ("latency_s", tm, "latency_std"), ("jitter_theta", tm, "jitter_theta_std"),
                       ("camera_drop_rate", tm, "camera_drop_std")):
        if src.get(f) is not None:
            out[f] = dict(mean=_num(src[f]), std=_num(src.get(sk)))
    out["n"] = dict(geom=int(g["n_views"]) if g else 0, photo=int(p.get("n_intervals", 0)),
                    timing=int(tm.get("n_frames", 0)))
    out["coverage"] = float(np.mean(np.asarray(g["coverage"]) > 0)) if g else 0.0
    out["geom_rms"] = _num(g["rms"]) if g else None
    il = p.get("insufficient_levels")
    out["photo_insufficient_levels"] = None if il is None else bool(il)
    out["health"] = {k: int(v) for k, v in {**est.get("health", {}), **(counters or {})}.items()}
    return out


def _loadavg():
    """``/proc/loadavg``'s 1, 5 and 15 min load (None where it does not exist)."""
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:
        return None


SLOW_LAP_MS = 2.0  # a note phase slower than one control tick is logged with its time


def _mem_marks(dev):
    """(reserved MiB, pinned host allocations) of this process: a before/after pair detects growth in the loop."""
    import torch
    return torch.cuda.memory_reserved(dev) / 2**20, torch.cuda.host_memory_stats().get("allocations.allocated", -1)


def _gc_arm():
    """Freeze the setup-time heap and time every collection in the paced loop.

    A gen2 collection that has to scan the setup objects (torch, Axon, dma-buf imports) takes 30-45 ms and lands
    once per ~10 s run, wherever the interpreter happens to be. ``gc.freeze()`` after a full
    ``gc.collect()`` moves those objects out of the collector's view, so later collections only scan the small
    per-note churn. GC stays enabled (``gc.disable()`` would hide cycle leaks). Trade-off: frozen objects that
    later become cyclic garbage are never collected, which is fine for a bounded run; a long-lived service
    built from this would need ``gc.unfreeze()`` at a quiet moment, and the callback stays registered.

    Returns:
        ``{"gen0": {"count", "max_ms"}, "gen1": ..., "gen2": ...}``, filled in by the callback; goes into the summary.
    """
    gc.collect()
    gc.freeze()
    stats, t_start = {f"gen{g}": dict(count=0, max_ms=0.0) for g in (0, 1, 2)}, {}

    def cb(phase, info):
        g = info["generation"]
        if phase == "start":
            t_start[g] = time.perf_counter()
        elif g in t_start:
            d = stats[f"gen{g}"]
            d["count"] += 1
            d["max_ms"] = max(d["max_ms"], 1e3 * (time.perf_counter() - t_start.pop(g)))
    gc.callbacks.append(cb)
    return stats


def _wait_until(target):
    d = target - time.perf_counter()
    if d > 0:
        time.sleep(d)


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        csv.writer(f).writerows(rows)


def _stats(x):
    if not len(x):
        return None
    return dict(mean=float(np.mean(x)), p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                max=float(np.max(x)))


def _gl_free():
    return not any(m.split(".")[0] in ("OpenGL", "mujoco") for m in sys.modules)


def _torch_cuda_free():
    return "torch" not in sys.modules, not any(m == "cuda" or m.startswith("cuda.") for m in sys.modules)


# ── SIM: GL only ──────────────────────────────────────────────────────
def sim(a, addr, ready, skip, barrier, t0v, exit_ok, q, ring_name=None):
    import mujoco

    from physicalai_sim2real_camera import CameraStreamBatch, SimCamera, ring_depth, stagger_phases
    from physicalai_sim2real_camera.camera import StateRing
    from physicalai_sim2real_camera.dmabuf import DmabufExporter

    cfg, N = make_cfg(a), a.n
    model, sweep = scene_model(a)
    model.opt.timestep = DT
    model.vis.quality.offsamples = 0  # the exporter's blit needs a single-sample offscreen FBO
    cam = SimCamera(model, "cam", cfg)  # CPU: render geometry (H_r x W_r, fovy_r) + the shared renderer
    r = cam.renderer
    datas = [mujoco.MjData(model) for _ in range(N)]
    for d in datas:
        if sweep is not None:  # prism: every env's head follows the one sweep, pose of d.time
            d.mocap_pos[0], d.mocap_quat[0] = sweep.pose(d.time)
        mujoco.mj_forward(model, d)
    head = None
    if ring_name is not None:  # live observer: the head state the robot reports (env 0), once per tick
        from physicalai_sim2real_camera.observer_replay import write_head
        from physicalai_sim2real_camera.scenes import HeadPoseRing
        head, head_rng = HeadPoseRing(name=ring_name), np.random.default_rng([0, 8])
    K, Ks = ring_depth(cfg), a.sim_slots  # VS output ring depth / exporter slots (groups) per camera
    # window mode: a capture is a group of n_sub colour sub-renders (SimCamera's plan, times capture_t + t) in
    # consecutive colour slots g*n_sub + k, plus (with --depth) one depth render at capture_t in slot g of a
    # separate depth exporter (so depth renderbuffers exist only where depth is read)
    window = cam.half_window > 0
    n_sub = len(cam._plan) if window else 1
    rings = [StateRing(model, cam.half_window) for _ in range(N)] if window else None
    scratch = mujoco.MjData(model) if window else None
    exz = None
    try:
        ex = DmabufExporter(r, N * Ks * n_sub, depth=a.depth and not window)
        ex.export()
        if window and a.depth:
            exz = DmabufExporter(r, N * Ks, depth=True)
            exz.export()
    except RuntimeError as e:
        q.send(dict(who="sim", skip=str(e)))
        skip.value = 1
        ready.set()
        return
    lst = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    lst.bind(addr)
    lst.listen(1)
    ready.set()
    conn, _ = lst.accept()
    ex.send_fds(conn, K=K, Ks=Ks, n=N, n_sub=n_sub, depth_exporter=exz is not None)
    if exz is not None:
        exz.send_fds(conn)

    renders, fired_at = np.zeros(N, np.int64), {}
    waits, group_ms, gl_renders = [], [], 0

    def hook(i, d, n):
        nonlocal gl_renders
        slot = flat_slot(i, int(renders[i]), Ks)  # the NOTE slot: the capture's group g
        renders[i] += 1
        if not window:
            set_scene(r, model, cam, d)
            t = time.perf_counter()
            fired_at[i] = (slot, ex.render(slot))  # fence fd (fence mode) or None after glFinish
            gl_renders += 1
        else:
            t = time.perf_counter()
            fd, nr = render_group(ex, exz, r, model, cam, rings[i], scratch, d.time, slot)
            fired_at[i] = (slot, fd)
            gl_renders += nr
            group_ms.append(1e3 * (time.perf_counter() - t))
        waits.append(time.perf_counter() - t)
        return slot

    streams = CameraStreamBatch(cfg, N, render=hook, cams=rings,
                                phase_s=stagger_phases(cfg, N) if a.stagger else None)
    fd = ex.render(0)  # warm-up outside the paced loop (slot 0 is rewritten by the first real render)
    if fd is not None:
        os.close(fd)
    waits.clear()
    gc_stats = _gc_arm()

    barrier.wait(60)
    t0v.value = time.perf_counter() + 0.2
    barrier.wait(60)
    t0 = t0v.value
    rows, late, n_steps, max_fired = [], 0, int(round(a.seconds / DT)), 0
    for k in range(1, n_steps + 1):
        _wait_until(t0 + k * DT)
        late += time.perf_counter() - t0 - k * DT > DT  # this tick started over one tick late
        for d in datas:
            mujoco.mj_step(model, d)  # d.time == k*DT == wall time since t0
            if sweep is not None:  # pose(t) -> kinematics -> stream.step, as observer_replay.sim_records
                d.mocap_pos[0], d.mocap_quat[0] = sweep.pose(d.time)
                # rendering needs only body / camera / light poses: no full mj_forward in the 500 Hz loop
                mujoco.mj_kinematics(model, d)
                mujoco.mj_camlight(model, d)
        if head is not None:
            write_head(head, sweep, datas[0].time, head_rng)
        ready_, cap, seq, _, _, fired = streams.step(datas)
        max_fired = max(max_fired, int(fired.sum()))
        for i in np.flatnonzero(fired):
            slot, fd = fired_at.pop(int(i))
            ct, rt, n = streams.streams[i].last_render
            msg = NOTE.pack(int(i), slot, n, ct, rt)
            if fd is None:
                conn.send(msg)
            else:
                socket.send_fds(conn, [msg], [fd])
                os.close(fd)
        t = datas[0].time
        rows += [(k, t, i, seq[i], cap[i], t - cap[i]) for i in np.flatnonzero(ready_)]
    conn.send(NOTE.pack(-1, 0, 0, 0.0, 0.0))
    _write(os.path.join(a.out, "pub_age.csv"), "k,t,cam,n,capture_t,age", rows)
    torch_free, cuda_free = _torch_cuda_free()
    q.send(dict(who="sim", pid=os.getpid(), K=K, Ks=Ks, slots=len(ex._rb), sync=ex.sync, gc=gc_stats,
               render_ms=_stats(1e3 * np.array(waits)),
               renders=int(renders.sum()), late_ticks=int(late), ticks=n_steps,
               n_sub=n_sub, group_renders=n_sub + (exz is not None), gl_renders=gl_renders,
               group_ms=_stats(np.array(group_ms)) if group_ms else None,
               sim_gl_mib=sum(len(e._rb) * len(e._rb[0]) for e in (ex, exz) if e is not None)
               * r.width * r.height * 4 / 2**20,
               stagger=bool(a.stagger), max_fired_per_tick=max_fired,
               torch_absent=torch_free, cuda_absent=cuda_free))
    exit_ok.wait(120)
    ex.close()
    if exz is not None:
        exz.close()
    cam.close()
    conn.close()
    if head is not None:
        head.close()


# ── VIRTUAL SENSOR: CUDA effects + Axon publish ───────────────────────
def vsensor(a, addr, ready_vs, barrier, t0v, exit_ok, q):
    # fewer half-empty 20 MiB allocator segments: the per-process footprint is gated in tests/test_virtual_sensor.py
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import axon
    import torch
    from test_batch import chunk_bytes

    from physicalai_sim2real_camera.batch import SimCameraBatch
    from physicalai_sim2real_camera.dmabuf import DmabufImporter, recv_descs, wait_fence

    cfg, N, dev = make_cfg(a), a.n, torch.device("cuda")
    torch.zeros(1, device=dev)  # CUDA context alone: the bound's baseline (plan: 270 MiB, docs/gpu_rl_design.md)
    ctx_mib = _own_smi_mib()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.connect(addr)
    descs, meta = recv_descs(s)  # blocking socket until every chunk is in (non-blocking below)
    im = DmabufImporter(descs)
    zdescs = recv_descs(s)[0] if meta["depth_exporter"] else []  # window mode + depth
    imz = DmabufImporter(zdescs) if zdescs else im
    model, _ = scene_model(a)  # the sim's camera and depth range
    b = SimCameraBatch(model, "cam", cfg, N, renderer=False, ring=meta["K"])
    assert (descs[0]["height"], descs[0]["width"]) == (b.H_r, b.W_r), "sim/vsensor render size mismatch"
    n_sub = meta["n_sub"]
    assert b.n_sub == n_sub, f"sim renders {n_sub} sub-frames per capture, the batch expects {b.n_sub}"
    # staging: one frame, or the capture's n_sub sub-frames in window mode (the batch's ingest shape)
    raw = torch.zeros((n_sub, b.H_r, b.W_r, 3) if b.window else (b.H_r, b.W_r, 3), dtype=torch.uint8, device=dev)
    dz = torch.zeros((b.H_r, b.W_r), dtype=torch.float32, device=dev) if a.depth else None
    # warm-up = exactly what the loop does, once, outside the timed loop: the dma-buf reads (their first launch
    # loads kernels lazily, ~7 ms; read_depth's float64 temporaries), ingest (advances each camera's noise/AE state
    # once) and process. Slot 0's content is undefined here (the sim may still be writing it) and is never used:
    # only the kernel loads and allocations matter. If AE/AWB is ever enabled in make_cfg, ingest zeros instead.
    if b.window:
        for k in range(n_sub):
            im.read(colour_slot(0, k, n_sub), raw[k])
    else:
        im.read(0, raw)
    if a.depth:
        imz.read_depth(0, dz)
    for i in range(N):
        b.ingest(i, raw, dz)
    b.process(depth=a.depth)

    accel = a.pool == "accelerator"
    backend = {"accelerator": axon.PoolBackend.Accelerator, "custom": axon.PoolBackend.Custom,
               "udmabuf": axon.PoolBackend.UDMABUF}[a.pool]
    PAY = payload_size(a.depth)
    pools = [axon.TensorPool.create(n_buffers=POOL_BUFFERS, buffer_size=PAY, backend=backend) for _ in range(N)]
    pubs = [axon.TensorPublisher.create(TOPIC.format(i), p) for i, p in enumerate(pools)]
    if accel:  # torch views of every pool buffer (CUDA array interface; same primary context)
        dsts = [[torch.as_tensor(p.device_array(k, [PAY], axon.DType.U8), device=dev) for k in range(POOL_BUFFERS)]
                for p in pools]
    else:
        host = np.empty(PAY, np.uint8)
    s.setblocking(False)  # select() waits; socket.recv_fds ignores its flags argument (CPython 3.12)
    ready_vs.set()
    # handshake_pool accepts only the connections pending once the first arrives: the observer's camera has two
    # subscribers (policy + observer) that connect independently, so its publisher repeats until both are in (a
    # missing one fails its own wait_handshake)
    got = []
    for p in pubs:
        n = p.handshake_pool()
        if n < 0:
            raise RuntimeError("vsensor: handshake_pool failed")
        got.append(n)
    if a.observer:
        i, deadline = a.observer_cams, time.perf_counter() + 20
        while got[i] < 2 and time.perf_counter() < deadline:
            n = pubs[i].handshake_pool()
            if n < 0:
                raise RuntimeError("vsensor: handshake_pool failed (observer camera)")
            got[i] += n
    # No release fence goes back to the sim: it reuses exporter slot (i, r % Ks) Ks renders later, i.e. no
    # earlier than capture_t + Ks/fps. A note older than (Ks - 1)/fps could race that overwrite, so its frame is
    # DROPPED (an "overrun drop", as a real camera drops frames when the host stalls), never read as possibly
    # wrong pixels. Beyond 4x that window the virtual sensor is hopelessly behind: hard error.
    max_note_age = (meta["Ks"] - 1) / cfg.fps

    def publish(i, n, cap, sl):
        # ponytail: copies + synchronize on the current stream. A dedicated stream waiting on a per-batch event
        # was measured too: same p99 (1.33 vs 1.35 ms) but +60-80 MiB of driver memory, so it
        # was dropped; with publish-first the current stream is idle when a publish is due.
        tr = torch.frombuffer(bytearray(TRAILER.pack(cap, n)), dtype=torch.uint8)
        t = time.perf_counter()
        if accel:
            acq = pubs[i].acquire()
            dst = dsts[i][acq.buffer_index]
            dst[:FRAME].copy_(b.rgb[i, sl].view(-1))
            if a.depth:
                dst[FRAME:PAY - TRAILER.size].copy_(b.depth[i, sl].reshape(-1).view(torch.uint8))
            dst[PAY - TRAILER.size:].copy_(tr)
            t = lap("publish_copy", t)
            torch.cuda.current_stream(dev).synchronize()  # pool buffer complete before the descriptor goes out
            lap("publish_sync", t)
            pubs[i].publish_device(acq, [PAY], axon.DType.U8)
        else:  # ponytail: synchronous D2H into one host buffer; pinned double buffer if it shows up in lateness
            host[:FRAME] = b.rgb[i, sl].cpu().numpy().reshape(-1)
            if a.depth:
                host[FRAME:PAY - TRAILER.size] = b.depth[i, sl].cpu().numpy().view(np.uint8).reshape(-1)
            host[PAY - TRAILER.size:] = tr.numpy()
            t = lap("publish_copy", t)
            lap("publish_sync", t)
            pubs[i].publish(host, axon.DType.U8)

    gc_stats = _gc_arm()
    barrier.wait(60)
    barrier.wait(60)
    t0 = t0v.value
    # read once here and once after the loop (never inside it: memory_stats builds a dict per call)
    reserved_t0, pinned_t0 = _mem_marks(dev)
    heap, last_n, logs, proc_ms, stale_drops, ended = [], [-1] * N, [], [], 0, False
    batch, flushes, overrun_drops, stall = [], 0, 0, a.stall_ms / 1e3
    note_ms = {k: [] for k in ("fence", "read", "ingest", "process", "publish_copy", "publish_sync")}
    batch_sizes = []
    publish_first_count = 0

    slow_laps = []  # (t - t0, key, ms) of any lap > SLOW_LAP_MS: where a stall sat, not just that one happened

    def lap(key, t):  # append elapsed ms since t and return a fresh timestamp
        now = time.perf_counter()
        ms = 1e3 * (now - t)
        note_ms[key].append(ms)
        if ms > SLOW_LAP_MS and len(slow_laps) < 50:
            slow_laps.append((now - t0, key, ms))
        return now

    def flush():
        t = time.perf_counter()
        slots = b.process(depth=a.depth)
        lap("process", t)
        batch_sizes.append(len(batch))
        for i, n, ct, rt in batch:
            heapq.heappush(heap, (publish_time(rt), rt, i, n, ct, int(slots[i])))
        batch.clear()

    def publish_due():
        nonlocal stale_drops
        now = time.perf_counter() - t0
        while heap and heap[0][0] <= now:
            due, rt, i, n, ct, sl = heapq.heappop(heap)
            if n <= last_n[i]:  # a newer frame of this camera is already out: never go back in time
                stale_drops += 1
                continue
            last_n[i] = n
            publish(i, n, ct, sl)
            logs.append((i, n, ct, rt, due, time.perf_counter() - t0))
            now = time.perf_counter() - t0

    while not ended or heap:
        timeout = None if not heap else max(0.0, t0 + heap[0][0] - time.perf_counter())
        if not ended and select.select([s], [], [], timeout)[0]:
            if stall and time.perf_counter() - t0 > 1.0:  # test hook: one stall, notes pile up meanwhile
                time.sleep(stall)
                stall = 0
            notes = []
            while True:  # drain everything already queued (non-blocking socket): one process() per batch
                try:
                    msg, fds, _, _ = socket.recv_fds(s, NOTE.size, 1)
                except BlockingIOError:
                    break
                notes.append((NOTE.unpack(msg), fds))
            ready = []
            t = time.perf_counter()  # process_ms: note received -> flush done (fence waits and the deferral included)
            for (i, slot, n, ct, rt), fds in notes:
                age = time.perf_counter() - t0 - ct
                if i >= 0 and age > 4 * max_note_age:
                    raise RuntimeError(f"note cam {i} n {n} is {1e3 * age:.1f} ms old > 4 x "
                                       f"{1e3 * max_note_age:.1f} ms slot window: virtual sensor hopelessly behind")
                if i >= 0 and age > max_note_age:  # its slot may already hold a newer render: drop the frame
                    for fd in fds:
                        os.close(fd)
                    overrun_drops += 1
                    continue
                if i < 0:
                    ended = True
                    continue
                tf = time.perf_counter()
                if fds:
                    wait_fence(fds[0])
                    lap("fence", tf)
                ready.append((i, slot, n, ct, rt))
            # publish-first: a publish due within the guard would otherwise wait behind this note segment
            if ready and heap and publish_first(heap[0][0], time.perf_counter() - t0, a.publish_guard_ms / 1e3):
                _wait_until(t0 + heap[0][0])
                publish_due()
                publish_first_count += 1
            for i, slot, n, ct, rt in ready:
                if b.pending(i):  # after a stall the same camera shows up twice in one drain
                    flush()
                    flushes += 1
                tr_ = time.perf_counter()
                if b.window:  # the NOTE slot is the group: colour slots slot*n_sub + k, depth slot `slot`
                    for k in range(n_sub):
                        im.read(colour_slot(slot, k, n_sub), raw[k])
                else:
                    im.read(slot, raw)
                if a.depth:
                    imz.read_depth(slot, dz)
                tr_ = lap("read", tr_)
                b.ingest(i, raw, dz)
                lap("ingest", tr_)
                batch.append((i, n, ct, rt))
            if batch:
                flush()
                proc_ms.append(1e3 * (time.perf_counter() - t))
        elif ended and heap:
            _wait_until(t0 + heap[0][0])
        publish_due()
    reserved_end, pinned_end = _mem_marks(dev)
    _write(os.path.join(a.out, "vs_publish.csv"), "cam,n,capture_t,ready_t,t_due,t_publish", logs)
    lg = np.array(logs)
    torch.cuda.synchronize()
    # bound terms from the formula (plan R5 Accept 3), checked against the live tensors so neither drifts alone
    Hr, Wr, K, C = b.H_r, b.W_r, b.ring, b.chunk
    fpn_per_cam = b._mul.shape[0] > 1  # (N,H,W) with per-camera FPN, one shared (1,H,W) map pair without
    # window: the per-camera output-space linear frame _wlin (12 B/px, output size) replaces the raw colour slot
    per_cam = (H * W * 12 if b.window else Hr * Wr * 3) + K * H * W * 3 + (2 * H * W * 4 if fpn_per_cam else 0)
    per_cam += K * H * W * 4 + Hr * Wr * 4 if a.depth else 0                         # depth ring + raw depth
    shared = 0 if fpn_per_cam else 2 * H * W * 4
    if not b.window:
        # idx(int32) + lin + LUT + grid (1,H,W,2) + nearest-map idx (int64) + work + noise + shot sigma
        # + grid_sample's (C,3,H,W) output, which process() allocates per chunk before copying it into _work
        chunk = 2 * C * 3 * Hr * Wr * 4 + 256 * 4 + H * W * 2 * 4 + H * W * 8 + 4 * C * 3 * H * W * 4
    else:
        # LUT + nearest-map idx + noise + shot sigma + sub-frame u8/idx/lin + band accumulators and grids (rows
        # [s0, s1) each band's remap reads); no chunk idx/lin/work/grid and no chunk grid_sample output
        bands = sum(s1 - s0 for _, _, s0, s1 in b.base._rows)
        # (make_cfg always sets `dist`, so every band has maps: an accumulator (12 B/px) and a grid (8 B/px at
        # output size))
        chunk = 256 * 4 + H * W * 8 + 2 * C * 3 * H * W * 4 + Hr * Wr * (3 + 12 + 12) + bands * Wr * 12 + H * W * 8
    chunk += C * Hr * Wr * 4 + C * H * W * 4 if a.depth else 0                     # depth chunk buffers
    stage = Hr * Wr * 4 + n_sub * Hr * Wr * 3 + (Hr * Wr * 4 if a.depth else 0)   # importer RGBA + raw (+ dz)
    stage += Hr * Wr * 4 if imz is not im else 0                                   # depth importer's RGBA
    live_cam = ((b._wlin if b.window else b._u8s)[0].nbytes + b.rgb[0].nbytes
                + ((b._mul[0].nbytes + b._add[0].nbytes) if fpn_per_cam else 0))
    live_cam += b.depth[0].nbytes + b._zs[0].nbytes if a.depth else 0
    live = (chunk_bytes(b) + (0 if b.window else b._work.nbytes), live_cam,
            0 if fpn_per_cam else b._mul.nbytes + b._add.nbytes,
            im._buf.nbytes + raw.nbytes + (dz.nbytes if a.depth else 0) + (imz._buf.nbytes if imz is not im else 0))
    assert live == (chunk, per_cam, shared, stage), f"bound formula {(chunk, per_cam, shared, stage)} != live {live}"
    pool_bytes = N * POOL_BUFFERS * -(-PAY // VMM_GRAN) * VMM_GRAN if accel else 0
    # imported dma-buf slots are the sim's memory; reported only (they do not show in this process's row)
    imports = sum(-(-d["stride"] * d["height"] // VMM_GRAN) * VMM_GRAN for d in descs)
    q.send(dict(who="vsensor", pid=os.getpid(), pool=a.pool, published=len(logs), stale_drops=stale_drops,
               descs_received=len(descs) + len(zdescs),
               late_ms=_stats(1e3 * (lg[:, 5] - lg[:, 3])), overdue_ms=_stats(1e3 * (lg[:, 5] - lg[:, 4])),
               process_ms=_stats(proc_ms), note_ms={k: _stats(v) for k, v in note_ms.items()},
               flush_batch=_stats(batch_sizes), publish_first=publish_first_count,
               slow_laps=slow_laps, reserved_mib_t0=reserved_t0, reserved_mib_end=reserved_end,
               gc=gc_stats,
               pinned_allocs_t0=pinned_t0, pinned_allocs_end=pinned_end,
               publish_guard_ms=a.publish_guard_ms, chunk_mib=chunk / 2**20, per_cam_mib=per_cam / 2**20,
               pool_mib=pool_bytes / 2**20, ctx_mib=ctx_mib, staging_mib=stage / 2**20, shared_mib=shared / 2**20,
               flushes=flushes, overrun_drops=overrun_drops, import_mib=imports / 2**20,
               torch_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
               torch_allocated_mib=torch.cuda.memory_allocated() / 2**20,
               torch_reserved_mib=torch.cuda.memory_reserved() / 2**20))
    exit_ok.wait(120)
    im.close()
    if imz is not im:
        imz.close()
    b.close()


# ── POLICY: Axon only ─────────────────────────────────────────────────
def policy(a, ready_vs, barrier, t0v, exit_ok, q):
    import axon
    accel = a.pool == "accelerator"
    if accel:  # device frames: read the 16-byte trailer with one DtoH copy (no torch, no GL)
        from cuda.bindings import driver as cu
        cu.cuInit(0)
        _, dev0 = cu.cuDeviceGet(0)
        _, ctx = cu.cuDevicePrimaryCtxRetain(dev0)  # the context Axon maps the pool into
        cu.cuCtxSetCurrent(ctx)
    PAY, N = payload_size(a.depth), a.n
    if not ready_vs.wait(60):
        raise RuntimeError("policy: virtual sensor did not come up")
    subs = [axon.TensorSubscriber.create(TOPIC.format(i)) for i in range(N)]
    for sub in subs:
        if sub.wait_handshake(10000) != 0:
            raise RuntimeError("policy: handshake failed")
        sub.set_fallback_policy(axon.FallbackPolicy.LastKnownGood)
    tr = np.empty(TRAILER.size, np.uint8)
    gc_stats = _gc_arm()

    barrier.wait(60)
    barrier.wait(60)
    t0 = t0v.value
    rows, n_steps, polls = [], int(round(a.seconds / DT)), []
    for k in range(1, n_steps + 1):
        _wait_until(t0 + k * DT + POLL_LAG)
        tp = time.perf_counter()
        for i, sub in enumerate(subs):
            v = sub.latest_view()
            now = time.perf_counter() - t0
            if v is None:
                continue
            if accel:
                err, = cu.cuMemcpyDtoH(tr.ctypes.data, v.device_ptr + PAY - TRAILER.size, TRAILER.size)
                if err != cu.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"policy: trailer read {err.name}")
                cap, n = TRAILER.unpack(tr.tobytes())
            else:
                cap, n = TRAILER.unpack(v.data.reshape(-1)[PAY - TRAILER.size:].tobytes())
            rows.append((k, now, i, n, cap, now - cap, k * DT - cap, v.staleness_ns, v.seqno))
        polls.append(time.perf_counter() - tp)
    _write(os.path.join(a.out, "sub_age.csv"), "k,t,cam,n,capture_t,age_sub,age_tick,staleness_ns,seqno", rows)
    torch_free, cuda_free = _torch_cuda_free()
    q.send(dict(who="policy", pid=os.getpid(), samples=len(rows), gl_absent=_gl_free(), torch_absent=torch_free,
               cuda_absent=cuda_free, poll_ms=_stats(1e3 * np.array(polls)), gc=gc_stats))
    exit_ok.wait(120)


# ── OBSERVER: one camera's frames -> camera parameter estimates ───────
def observer(a, ring_name, ready_vs, barrier, t0v, exit_ok, q):
    """Live observer: receive one camera's frames, estimate its parameters, write them at <= 1 Hz.

    The offline inputs of ``Observer`` (gamma, black level, exposure / readout, nominal FOV) are the sim's own
    ``make_cfg`` values here: in this live demo the offline calibration is exact, so what the estimates show is the
    observer itself (the O6 closed loop perturbs the prism; this process does not).

    Args:
        a: Parsed arguments (``pool``, ``depth``, ``seconds``, ``observer_cams``, ``out`` and the window flags).
        ring_name: Shared-memory name of the parent's ``HeadPoseRing`` (the sim writes the reported head state).
        ready_vs: Event set once the virtual sensor's publishers are up.
        barrier: Start barrier shared with the sim, virtual sensor and policy (two waits, as theirs).
        t0v: Shared ``perf_counter`` epoch, valid after the second barrier wait.
        exit_ok: Event the parent sets once it has read every result.
        q: Result pipe (``Connection``): receives the ``who="observer"`` dict.

    Raises:
        RuntimeError: The virtual sensor did not come up, the handshake or a frame copy failed, or the worker
            thread died (its exception is chained).
    """
    os.nice(10)  # background job: the sim / virtual sensor / policy keep the CPU
    sys.setswitchinterval(0.001)  # the receive loop gets the GIL back within 1 ms of the worker's Python work
    import cv2
    cv2.setNumThreads(1)  # one core: one marker pass + six ChArUco interpolations, ~15 ms/frame
    import queue
    import threading

    import axon

    from physicalai_sim2real_camera.observer_replay import FrameRecord, Observer, sensor_clock
    from physicalai_sim2real_camera.scenes import HeadPoseRing
    from physicalai_sim2real_camera.target import PrismTarget

    accel = a.pool == "accelerator"
    if accel:  # device frames: two DtoH copies (frame, trailer) per new frame: the observer's only GPU use
        from cuda.bindings import driver as cu
        cu.cuInit(0)
        _, dev0 = cu.cuDeviceGet(0)
        _, ctx = cu.cuDevicePrimaryCtxRetain(dev0)
        cu.cuCtxSetCurrent(ctx)  # per thread: the receive loop runs in this (main) thread
    cfg = make_cfg(a)
    gamma, black, exposure_s, readout_s = cfg.gamma, cfg.black_level, cfg.exposure_s, cfg.readout_s
    del cfg  # the observer gets offline values only, never the simulator's config object
    PAY, cam = payload_size(a.depth), a.observer_cams
    ring = HeadPoseRing(name=ring_name)
    obs = Observer(PrismTarget.from_json(os.path.join(os.path.dirname(PRISM_XML), "prism.json")), (W, H),
                   OBS_FOVY_DEG, gamma, black, exposure_s=exposure_s, readout_s=readout_s)
    if not ready_vs.wait(60):
        raise RuntimeError("observer: virtual sensor did not come up")
    sub = axon.TensorSubscriber.create(TOPIC.format(cam))
    if sub.wait_handshake(10000) != 0:
        raise RuntimeError("observer: handshake failed")
    sub.set_fallback_policy(axon.FallbackPolicy.LastKnownGood)
    buf, rng = np.empty(FRAME + TRAILER.size, np.uint8), np.random.default_rng([0, 9])  # frame | trailer, no depth
    recs = queue.Queue(maxsize=OBS_QUEUE)
    path = os.path.join(a.out, "camera_config_estimate.json")
    st = dict(received=0, worker_drops=0, pushed=0, estimates=0)
    worker_ms, estimate_ms, last, failed = [], [], {}, []

    def write_estimate():
        t = time.perf_counter()
        est = obs.estimate(ring.omega_fn())
        estimate_ms.append(1e3 * (time.perf_counter() - t))
        st["estimates"] += 1
        last.update(est=est, summary=estimate_summary(est, st))
        with open(path + ".tmp", "w") as f:
            json.dump(last["summary"], f, indent=1)
        os.replace(path + ".tmp", path)  # a reader never sees a half-written file

    def work():  # replay()'s schedule: an estimate once t_arrival passes the next whole period
        t_next = None
        try:
            while (rec := recs.get()) is not None:
                t = time.perf_counter()
                obs.push(rec, ring.omega_fn())
                worker_ms.append(1e3 * (time.perf_counter() - t))
                st["pushed"] += 1
                t_next = rec.t_arrival + OBS_EVERY_S if t_next is None else t_next
                if rec.t_arrival >= t_next:
                    write_estimate()
                    while t_next <= rec.t_arrival:
                        t_next += OBS_EVERY_S
        except BaseException as e:  # the receive loop re-raises it (a dead worker would look like 100 % drops)
            failed.append(e)

    def check_worker():
        if not worker.is_alive():
            raise RuntimeError("observer: worker thread died") from (failed[0] if failed else None)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    barrier.wait(60)
    barrier.wait(60)
    t0, seq = t0v.value, None
    for k in range(1, int(round(a.seconds / DT)) + 1):
        _wait_until(t0 + k * DT + POLL_LAG)
        v = sub.latest_view()
        t_arrival = time.perf_counter() - t0  # receipt, before the copy (sim time == wall time since t0)
        if v is None or v.seqno == seq:
            continue
        check_worker()
        seq = v.seqno
        if accel:  # frame and trailer only (with --depth the depth block sits between them)
            for dst, src, nb in ((0, 0, FRAME), (FRAME, PAY - TRAILER.size, TRAILER.size)):
                err, = cu.cuMemcpyDtoH(buf.ctypes.data + dst, v.device_ptr + src, nb)
                if err != cu.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"observer: frame read {err.name}")
            frame, tr = buf[:FRAME], buf[FRAME:]
        else:
            d = v.data.reshape(-1)
            frame, tr = d[:FRAME], d[PAY - TRAILER.size:]
        capture_t, n = TRAILER.unpack(tr.tobytes())
        st["received"] += 1
        # the true capture time never reaches the observer: only the camera's own timestamp does
        rec = FrameRecord(frame.reshape(H, W, 3).copy(), t_arrival, sensor_clock(capture_t, rng),
                          seq=int(seq), frame_number=int(n), exposure=exposure_s, gain=1.0)
        try:
            recs.put_nowait(rec)
        except queue.Full:
            st["worker_drops"] += 1
    check_worker()
    try:
        recs.put(None, timeout=60)  # the worker drains at most OBS_QUEUE frames (~0.5 s) first
    except queue.Full:
        raise RuntimeError("observer: worker did not drain its queue in 60 s") from None
    worker.join()
    if failed:
        raise RuntimeError("observer: worker thread died") from failed[0]
    obs.flush()
    write_estimate()
    timing = last["est"]["timing"]
    q.send(dict(who="observer", pid=os.getpid(), cam=cam, **st,
                observer_miss_rate=_num(timing.get("observer_miss_rate")),
                worker_ms=_stats(worker_ms), estimate_ms=_stats(estimate_ms), final=last["summary"]))
    exit_ok.wait(120)
    ring.close()


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


def _compute_apps():
    """{pid: used MiB} of CUDA (compute) processes."""
    rows = _smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").strip().splitlines()
    return dict((int(p), int(m)) for p, m in (ln.split(",") for ln in rows))


def _gpu_condition(own=()):
    """Compute processes and utilisation right now; contended if a FOREIGN process holds > 4 GB or util > 50 %.

    Args:
        own: PIDs of this run's processes. They are listed but never count as contention, and while they are
            alive the utilisation figure is theirs (an N=16 run alone reads ~50 %), so util is ignored then.
    """
    rows = _smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits").strip()
    apps = []
    for ln in filter(None, rows.splitlines()):
        pid, *name, mib = ln.split(",")
        apps.append(dict(pid=int(pid), name=",".join(name).strip(), mib=int(mib)))
    util = int(_smi("--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").split()[0])
    own = set(own)
    foreign = [x for x in apps if x["pid"] not in own]
    return dict(apps=apps, util_pct=util,
                contended=(util > 50 and not own) or any(x["mib"] > 4096 for x in foreign))


def _own_smi_mib():
    time.sleep(0.3)  # nvidia-smi lags allocations slightly
    return _compute_apps().get(os.getpid())


# ── parent ────────────────────────────────────────────────────────────

def analyse(N, out_dir):
    """KS(tick-timed policy age vs sim age) per camera and pooled, seqno coverage, wall-age mean diff."""
    from test_axon_bridge import ks
    pa = np.genfromtxt(os.path.join(out_dir, "pub_age.csv"), delimiter=",", names=True)
    sa = np.genfromtxt(os.path.join(out_dir, "sub_age.csv"), delimiter=",", names=True)
    # KS on age_tick rounded to 1 us, as tests/test_axon_bridge.py (pub ages sit on the 2 ms grid)
    vp = np.genfromtxt(os.path.join(out_dir, "vs_publish.csv"), delimiter=",", names=True)
    out = dict(ks={}, coverage={}, dmean_wall_ms={}, n_pub={}, n_sub={})
    # overrun gates: the policy sees the sim's frame on (almost) every tick, and publishes are not a tick late
    sim_n = {(int(k), int(c)): int(n) for k, c, n in zip(pa["k"], pa["cam"], pa["n"])}
    out["tick_match"] = float(np.mean([sim_n.get((int(k), int(c))) == int(n)
                                       for k, c, n in zip(sa["k"], sa["cam"], sa["n"])]))
    out["publish_late_p99_ms"] = float(1e3 * np.percentile(vp["t_publish"] - vp["ready_t"], 99))
    out["publish_overdue_p99_ms"] = float(1e3 * np.percentile(vp["t_publish"] - vp["t_due"], 99))
    for i in list(range(N)) + ["pooled"]:
        p = pa if i == "pooled" else pa[pa["cam"] == i]
        s_ = sa if i == "pooled" else sa[sa["cam"] == i]
        out["ks"][str(i)] = ks(np.round(s_["age_tick"], 6), np.round(p["age"], 6))
        out["dmean_wall_ms"][str(i)] = 1e3 * abs(s_["age_sub"].mean() - p["age"].mean())
        out["n_pub"][str(i)], out["n_sub"][str(i)] = len(p), len(s_)
        if i != "pooled":
            out["coverage"][str(i)] = len(np.unique(s_["n"])) / len(np.unique(p["n"]))
    out["coverage"]["pooled"] = min(out["coverage"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=4, help="envs / cameras")
    ap.add_argument("--depth", action="store_true", help="also export, process and publish depth")
    ap.add_argument("--pool", choices=["accelerator", "custom", "udmabuf"], default="accelerator")
    ap.add_argument("--sim-slots", type=int, default=4,
                    help="exporter slots per camera; the virtual sensor copies a slot out on arrival and drops "
                         "(overrun) notes older than (sim_slots - 1)/fps = 100 ms at 4 slots / 30 fps")
    ap.add_argument("--run-name", default="main", help="output sub-directory under out/virtual_sensor/")
    ap.add_argument("--stall-ms", type=float, default=0.0, help="test hook: stall the virtual sensor once at 1 s")
    ap.add_argument("--exposure-ms", type=float, default=0.0, help="exposure time (motion blur, window mode)")
    ap.add_argument("--readout-ms", type=float, default=0.0, help="rolling-shutter readout time (window mode)")
    ap.add_argument("--stagger", action="store_true",
                    help="stagger phase_s per camera (cfg.phase_s + i/(N*fps)) so captures spread over the frame")
    ap.add_argument("--publish-guard-ms", type=float, default=PUBLISH_GUARD_MS,
                    help="publish-first lookahead: a publish due within this window runs before a new note segment")
    ap.add_argument("--scene", choices=["board", "prism"], default="board",
                    help="board: static chessboard; prism: ChArUco prism under the scripted head sweep")
    ap.add_argument("--observer", action="store_true",
                    help="run the live observer process on one camera (needs --scene prism)")
    ap.add_argument("--observer-cams", default="0",
                    help="camera the observer subscribes to (one camera only)")
    a = ap.parse_args()
    if a.observer and a.scene != "prism":
        ap.error("--observer needs --scene prism")
    if not re.fullmatch(r"\d+", a.observer_cams) or int(a.observer_cams) >= a.n:
        ap.error(f"--observer-cams: one camera index in [0, {a.n}) (only one camera is supported)")
    a.observer_cams = int(a.observer_cams)
    a.out = os.path.join(OUT, a.run_name)
    load_start = _loadavg()
    try:
        import axon
        from cuda.bindings import driver  # noqa: F401
        from OpenGL.EGL.MESA import image_dma_buf_export  # noqa: F401
    except ImportError as e:
        print(f"SKIP: {e} (axon: scripts/build_axon.sh; cuda-python; PyOpenGL EGL extensions)")
        return
    if a.pool == "accelerator" and not hasattr(axon.PoolBackend, "Accelerator"):
        print("SKIP: axon built without AXON_WITH_CUDA (no PoolBackend.Accelerator); use --pool custom")
        return
    os.makedirs(a.out, exist_ok=True)
    gpu = _gpu_condition()
    if gpu["contended"]:
        print(f"NOTE: GPU contended at start (foreign compute > 4 GB or util > 50 %): {gpu}", file=sys.stderr)
    ring = None
    if a.observer:  # head-state ring: the parent owns (and unlinks) it, the sim writes, the observer reads
        from physicalai_sim2real_camera.scenes import HeadPoseRing
        ring = HeadPoseRing()
    try:
        _run(a, gpu, load_start, ring)
    finally:  # also on sys.exit from a failed child: the ring is never left in /dev/shm
        if ring is not None:
            ring.close()


def _run(a, gpu, load_start, ring):
    """Start the processes, collect their results and write ``summary.json`` (``main`` owns ``ring``)."""
    addr = f"\0realsensor-vsensor-{os.getpid()}"
    ctx = mp.get_context("spawn")
    n_procs = 3 + a.observer
    # one result pipe per child (a shared Queue would need its private _reader for connection.wait)
    pipes = [ctx.Pipe(duplex=False) for _ in range(n_procs)]
    ready, ready_vs, exit_ok = ctx.Event(), ctx.Event(), ctx.Event()
    skip, barrier, t0v = ctx.Value("b", 0), ctx.Barrier(n_procs), ctx.Value("d", 0.0)
    t_start = time.perf_counter()
    ps = ctx.Process(target=sim, args=(a, addr, ready, skip, barrier, t0v, exit_ok, pipes[0][1]),
                     kwargs=dict(ring_name=ring and ring.name))
    ps.start()
    if not ready.wait(60):
        ps.kill()
        sys.exit("sim did not come up")
    if skip.value:
        print(f"SKIP: {pipes[0][0].recv()['skip']}")
        ps.join(10)
        return
    procs = [ps, ctx.Process(target=vsensor, args=(a, addr, ready_vs, barrier, t0v, exit_ok, pipes[1][1])),
             ctx.Process(target=policy, args=(a, ready_vs, barrier, t0v, exit_ok, pipes[2][1]))]
    if a.observer:
        procs.append(ctx.Process(target=observer, args=(a, ring.name, ready_vs, barrier, t0v, exit_ok, pipes[3][1])))
    for p in procs[1:]:
        p.start()
    res, deadline = {}, time.perf_counter() + a.seconds + 90
    readers = [r for r, _ in pipes]
    while len(res) < n_procs:  # children exit only after exit_ok: a sentinel firing first means one died
        rdy = mpc.wait(readers + [p.sentinel for p in procs], max(0.0, deadline - time.perf_counter()))
        got = [r for r in rdy if r in readers]
        for r in got:
            m = r.recv()
            res[m["who"]] = m
            readers.remove(r)
        if not got and (rdy or time.perf_counter() > deadline):
            for p in procs:
                if p.is_alive():
                    p.kill()
            sys.exit(f"virtual sensor failed: got {sorted(res)}, exit codes {[p.exitcode for p in procs]}; "
                     f"GPU at start: {gpu}")
    apps = _compute_apps()
    table = _smi()
    gpu_end = _gpu_condition(own={m["pid"] for m in res.values()})  # contention can start mid-run
    if gpu_end["contended"]:
        print(f"NOTE: GPU contended at end of run: {gpu_end}", file=sys.stderr)
    exit_ok.set()
    for p in procs:
        p.join(30)
    if any(p.exitcode != 0 for p in procs):
        sys.exit(f"virtual sensor failed: exit codes {[p.exitcode for p in procs]}; GPU at start: {gpu}")
    runtime = time.perf_counter() - t_start

    sm, vs, po = res["sim"], res["vsensor"], res["policy"]
    row = re.search(rf"\b{sm['pid']}\b.*?(\d+)MiB", table)  # GL ("G") processes: table only
    # CUDA context (measured in the process; plan: 270) + chunk work buffers + per-camera raw slot/output ring
    # (+ FPN when per-camera; shared (1,H,W) maps once) + staging + Axon pool, +10 %. Imported dma-buf slots
    # are the sim's memory and do not show in this process's row (8 vs 24 slots: both 506 MiB).
    bound = 1.1 * (vs["ctx_mib"] + vs["chunk_mib"] + a.n * vs["per_cam_mib"] + vs["shared_mib"]
                   + vs["staging_mib"] + vs["pool_mib"])
    summary = dict(n=a.n, seconds=a.seconds, depth=a.depth, pool=a.pool, sim_slots=a.sim_slots,
                   stall_ms=a.stall_ms, exposure_ms=a.exposure_ms, readout_ms=a.readout_ms, run_name=a.run_name,
                   scene=a.scene, gpu_at_start=gpu, gpu_at_end=gpu_end, loadavg=dict(start=load_start, end=_loadavg()),
                   latency_ms=1e3 * make_cfg(a).latency_s, runtime_s=runtime,
                   **analyse(a.n, a.out), sim=sm, vsensor=vs, policy=po,
                   mem_mib=dict(sim_table=int(row.group(1)) if row else None, sim_in_compute_apps=sm["pid"] in apps,
                                vsensor=apps.get(vs["pid"]), policy=apps.get(po["pid"]),
                                policy_in_compute_apps=po["pid"] in apps, vsensor_bound=bound))
    if "observer" in res:
        summary["observer"] = res["observer"]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
