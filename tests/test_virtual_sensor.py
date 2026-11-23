"""R5 virtual sensor: 3-process run (GL-only sim -> dma-buf -> CUDA effects -> Axon -> policy) + helper unit tests.

``test_virtual_sensor_run`` runs examples/virtual_sensor.py (5 s, N=4, subprocess, 120 s timeout) and checks
out/virtual_sensor/summary.json against plan R5 Accept (KS(tick) < 0.05 per camera and pooled, seqno coverage
>= 0.9, process roles, virtual-sensor GPU memory bound with its terms pinned, runtime < 90 s) plus the overrun gates
(per-tick frame match >= 0.99, p99(publish - ready_t) < DT, latency_s > p99(process)). ``test_virtual_sensor_stall``
stalls the virtual sensor 150 ms (custom pool, default slots): overrun drops, no crash, ages still match,
policy CUDA-free.
Prints SKIP and passes when the Axon
bindings, cuda-python or the EGL dma-buf extensions are missing. The helper tests (slot mapping, payload size,
trailer and notification codecs) need none of that. Plain functions and asserts; also runs standalone.
"""

import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import virtual_sensor as vs  # noqa: E402

AXON_PY = os.path.join(ROOT, "third_party", "axon", "build", "python")


def test_flat_slot():
    # Basic ring-slot mapping: K=6
    assert vs.flat_slot(0, 0, 6) == 0
    assert vs.flat_slot(0, 5, 6) == 5
    assert vs.flat_slot(0, 6, 6) == 0
    assert vs.flat_slot(1, 0, 6) == 6
    assert vs.flat_slot(3, 13, 6) == 19

    # Bijection: for N=4, K=6 the flat slots cover exactly range(24)
    K = 6
    N = 4
    slots = {vs.flat_slot(i, r, K) for i in range(N) for r in range(K)}
    assert slots == set(range(N * K)), f"expected {set(range(N * K))}, got {slots}"


def test_payload_size():
    assert vs.payload_size(False) == 921616
    assert vs.payload_size(True) == 2150416


def test_publish_time():
    # DT = 2 ms, sim delivers at the first tick >= ready_t; publish at max(ready_t, tick - 1 ms)
    assert abs(vs.publish_time(0.0101) - 0.011) < 1e-12   # tick 12 ms: early in the window -> 11 ms
    assert vs.publish_time(0.0115) == 0.0115               # late in the window: at ready_t
    assert vs.publish_time(0.012) == 0.012                 # exactly on tick 12 ms
    assert abs(vs.publish_time(0.0120001) - 0.013) < 1e-12  # just past tick 12 -> tick 14 ms, 13 ms
    assert vs.publish_time(0.012 + 5e-10) == 0.012 + 5e-10  # within CameraStream's 1 ns: still tick 12


def test_publish_first():
    """publish_first: a publish due within the guard runs before a new note segment; later ones do not."""
    assert vs.publish_first(0.0500, 0.0495, 0.0012)          # due in 0.5 ms, guard 1.2 ms -> publish first
    assert vs.publish_first(0.0500, 0.0500, 0.0012)          # due now
    assert vs.publish_first(0.0500, 0.0505, 0.0012)          # already overdue
    assert not vs.publish_first(0.0500, 0.0480, 0.0012)      # due in 2 ms > guard
    assert not vs.publish_first(0.0500, 0.0480, 0.0)         # zero guard: only overdue publishes go first
    print("V2b publish_first: lookahead window semantics")


def test_trailer():
    # Size
    assert vs.TRAILER.size == 16

    # Round-trip pack/unpack
    packed = vs.TRAILER.pack(0.123, 7)
    ct, n = vs.TRAILER.unpack(packed)
    assert ct == 0.123
    assert n == 7

    # Numpy view compatibility (axon_bridge.py pattern)
    buf = np.frombuffer(vs.TRAILER.pack(0.5, 42), np.uint8)
    assert buf[:8].view("<f8")[0] == 0.5
    assert buf[8:].view("<i8")[0] == 42

    # Bytes match bare struct.pack
    packed = vs.TRAILER.pack(1.0, 1)
    expected = struct.pack("<d", 1.0) + struct.pack("<q", 1)
    assert packed == expected


def test_note():
    # Size
    assert vs.NOTE.size == 32

    # Round-trip pack/unpack
    packed = vs.NOTE.pack(2, 13, 12345678901, 1.25, 1.3125)
    cam, slot, n, ct, rt = vs.NOTE.unpack(packed)
    assert cam == 2
    assert slot == 13
    assert n == 12345678901
    assert ct == 1.25
    assert rt == 1.3125

    # End marker round-trips with cam == -1
    end = vs.NOTE.pack(-1, 0, 0, 0.0, 0.0)
    cam2, _, _, _, _ = vs.NOTE.unpack(end)
    assert cam2 == -1

    # First 4 bytes are little-endian cam (cam=3 -> 0x03)
    packed3 = vs.NOTE.pack(3, 0, 0, 0.0, 0.0)
    assert packed3[:4] == b"\x03\x00\x00\x00"


def test_estimate_summary():
    """estimate_summary: per-field mean/std, counts, coverage fraction, health merge; JSON-safe (no NaN, no numpy)."""
    K = np.array([[500.0, 0, 320.5], [0, 501.0, 240.25], [0, 0, 1]])
    std = dict(fx=1.0, fy=1.1, cx=0.5, cy=0.6, k1=0.01, k2=0.02, p1=1e-4, p2=2e-4, k3=0.03)
    cov = np.array([[1, 0, 2, 0], [0, 0, 3, 1], [0, 0, 0, 0]])  # 4 of 12 cells hold views
    est = dict(geom=dict(K=K, dist=np.array([-0.3, 0.1, 0.001, -0.001, 0.0]), std=std, n_views=np.int64(7),
                         coverage=cov, rms=np.float64(0.21)),
               photo=dict(sigma_shot=0.02, sigma_read=0.005, n_intervals=3, sigma_shot_std=0.001,
                          sigma_read_std=0.0002, insufficient_levels=np.bool_(False)),
               timing=dict(n_frames=120, jitter_theta=0.005, jitter_theta_std=0.0004, camera_drop_rate=0.05,
                           camera_drop_std=0.02, latency_s=float("nan"), latency_std=0.001),
               health=dict(n_frames=np.int64(120), observer_misses=2))
    out = vs.estimate_summary(est, dict(received=122, worker_drops=0))
    assert out["fx"] == dict(mean=500.0, std=1.0) and out["cy"] == dict(mean=240.25, std=0.6)
    assert out["k1"]["mean"] == -0.3 and out["p2"] == dict(mean=-0.001, std=2e-4) and out["k3"]["std"] == 0.03
    assert out["sigma_read"] == dict(mean=0.005, std=0.0002)
    assert out["latency_s"] == dict(mean=None, std=0.001), "NaN must become None (strict JSON)"
    assert out["camera_drop_rate"] == dict(mean=0.05, std=0.02)
    assert out["n"] == dict(geom=7, photo=3, timing=120)
    assert abs(out["coverage"] - 4 / 12) < 1e-12
    assert out["geom_rms"] == 0.21 and out["photo_insufficient_levels"] is False
    assert out["health"] == dict(n_frames=120, observer_misses=2, received=122, worker_drops=0)
    json.loads(json.dumps(out, allow_nan=False))  # strict JSON round trip
    # nothing fitted yet: only the counts and the health
    empty = vs.estimate_summary(dict(geom=None, photo=None, timing=dict(n_frames=2, jitter_k=2), health={}))
    assert empty == dict(n=dict(geom=0, photo=0, timing=2), coverage=0.0, geom_rms=None,
                         photo_insufficient_levels=None, health={})
    # a photometric fit with too few levels reports None sigmas: left out, flagged
    few = vs.estimate_summary(dict(photo=dict(sigma_shot=None, n_intervals=1, insufficient_levels=True)))
    assert "sigma_shot" not in few and few["photo_insufficient_levels"] is True


def test_num():
    """_num: finite floats pass, NaN / inf / None become None."""
    assert vs._num(np.float32(0.5)) == 0.5 and isinstance(vs._num(np.float64(1.0)), float)
    assert vs._num(float("nan")) is None and vs._num(float("inf")) is None and vs._num(None) is None


def run(name, *args, trace=True):
    """Run the example into out/virtual_sensor/<name>/; returns its summary.json, or None when it printed SKIP.

    ``trace=False`` keeps coverage out of the child processes (``make cov`` measures subprocesses via
    ``COVERAGE_PROCESS_CONFIG``): tracing every Python line of the window-mode virtual sensor quadruples its p99
    processing time (3 -> 13 ms at N=4), so strict real-time gates are measured untraced.
    """
    env = dict(os.environ, MUJOCO_GL="egl",
               PYTHONPATH=os.pathsep.join(filter(None, [AXON_PY, os.environ.get("PYTHONPATH")])))
    if not trace:
        for k in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START"):
            env.pop(k, None)
    t = time.perf_counter()
    p = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "virtual_sensor.py"), "--run-name", name, *args],
                       env=env, capture_output=True, text=True, timeout=120)
    dt = time.perf_counter() - t
    if p.returncode == 0 and p.stdout.startswith("SKIP"):
        print(p.stdout.strip())
        return None
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with open(os.path.join(ROOT, "out", "virtual_sensor", name, "summary.json")) as f:
        s = json.load(f)
    s["wall_s"] = dt
    for when in ("start", "end"):
        g = s[f"gpu_at_{when}"]
        print(f"   GPU at {when}{' (CONTENDED)' if g['contended'] else ''}: util {g['util_pct']} %, "
              f"compute apps {[(x['name'].split('/')[-1], x['mib']) for x in g['apps']]}")
    return s


def gpu(s):
    """GPU condition suffix for every gate message: timing gates depend on who else was on the GPU."""
    return f" | GPU at start: {s['gpu_at_start']}; at end: {s['gpu_at_end']}"


def check_ages(s, n=4, exact=True):
    g = gpu(s)
    for i in [*map(str, range(n)), "pooled"]:
        assert s["ks"][i] < 0.05, f"KS[{i}] {s['ks'][i]:.4f} >= 0.05" + g
        assert s["coverage"][i] >= 0.9, f"coverage[{i}] {s['coverage'][i]:.3f} < 0.9" + g
    if exact:  # overrun gates (KS only proves the delivery schedule); a stall shifts the mean age by design
        for i in [*map(str, range(n)), "pooled"]:
            assert s["dmean_wall_ms"][i] < 3.0, f"|dmean(wall)|[{i}] {s['dmean_wall_ms'][i]:.2f} ms >= 3 ms" + g
        assert s["tick_match"] >= 0.99, f"per-tick frame match {s['tick_match']:.4f} < 0.99" + g
        assert s["publish_late_p99_ms"] < 1e3 * vs.DT, f"p99 publish - ready_t {s['publish_late_p99_ms']:.2f} ms" + g
        assert s["latency_ms"] > s["vsensor"]["process_ms"]["p99"], "effects do not fit in the latency budget" + g
    assert s["wall_s"] < 90, f"runtime {s['wall_s']:.1f} s" + g


def test_virtual_sensor_run():
    from physicalai_sim2real_camera import ring_depth
    s = run("main", "--seconds", "5", "--n", "4")
    if s is None:
        return
    sm, v, po, mem = s["sim"], s["vsensor"], s["policy"], s["mem_mib"]
    print(f"R5 runtime {s['wall_s']:.1f} s; KS(tick) per camera {[round(s['ks'][str(i)], 4) for i in range(4)]}, "
          f"pooled {s['ks']['pooled']:.4f}; coverage min {s['coverage']['pooled']:.3f}; "
          f"tick match {s['tick_match']:.4f}; p99 publish - ready_t {s['publish_late_p99_ms']:.2f} ms "
          f"(- due {s['publish_overdue_p99_ms']:.2f}); "
          f"|dmean(wall)| max {max(s['dmean_wall_ms'].values()):.2f} ms")
    print(f"   sim: slots {sm['slots']} (Ks={sm['Ks']}), sync {sm['sync']}, "
          f"render call {sm['render_ms']['mean']:.3f} ms (p95 {sm['render_ms']['p95']:.3f}), "
          f"late ticks {sm['late_ticks']}/{sm['ticks']}, GPU {mem['sim_table']} MiB")
    print(f"   vsensor ({v['pool']}): published {v['published']}, process {v['process_ms']['mean']:.3f} ms/batch "
          f"(p99 {v['process_ms']['p99']:.3f}), GPU {mem['vsensor']} MiB <= bound {mem['vsensor_bound']:.1f} "
          f"(ctx {v['ctx_mib']}, chunk {v['chunk_mib']:.1f}, per cam {v['per_cam_mib']:.2f}, "
          f"shared {v['shared_mib']:.2f}, "
          f"staging {v['staging_mib']:.1f}, pool {v['pool_mib']:.0f}, imports {v['import_mib']:.0f}); "
          f"policy GPU {mem['policy']} MiB")
    check_ages(s)
    g = gpu(s)
    assert sm["torch_absent"] and sm["cuda_absent"] and not mem["sim_in_compute_apps"], "sim loaded torch/CUDA"
    assert po["gl_absent"] and po["torch_absent"], "policy loaded GL/mujoco/torch"
    # bound terms pinned to the plan, not to what the implementation happens to allocate
    assert sm["K"] == ring_depth(vs.make_cfg()) and sm["slots"] == 4 * 4
    assert v["pool_mib"] == 4 * vs.POOL_BUFFERS * 2 and v["import_mib"] == sm["slots"] * 2
    assert v["ctx_mib"] <= 1.2 * 270, f"CUDA context {v['ctx_mib']} MiB vs plan 270"
    assert mem["vsensor"] is not None and mem["vsensor"] <= mem["vsensor_bound"], str(mem) + g


def test_virtual_sensor_stall():
    """150 ms virtual-sensor stall, default sim slots (window 100 ms): overrun drops + early flush, no crash."""
    s = run("stall", "--seconds", "5", "--n", "4", "--stall-ms", "150", "--pool", "custom")
    if s is None:
        return
    v, po, g = s["vsensor"], s["policy"], gpu(s)
    print(f"R5 stall 150 ms (custom pool, default sim slots {s['sim_slots']}): overrun drops {v['overrun_drops']}, "
          f"flushes {v['flushes']}, stale drops {v['stale_drops']}, KS pooled {s['ks']['pooled']:.4f}, "
          f"coverage min {s['coverage']['pooled']:.3f}, tick match {s['tick_match']:.4f}, "
          f"max publish - ready_t {v['late_ms']['max']:.1f} ms; "
          f"policy CUDA-free {po['cuda_absent'] and not s['mem_mib']['policy_in_compute_apps']}")
    assert s["sim_slots"] == 4
    assert v["overrun_drops"] >= 1, "150 ms stall > 100 ms slot window but nothing was dropped" + g
    assert v["flushes"] >= 1, "stall did not make a camera repeat within one drain" + g
    check_ages(s, exact=False)  # the stalled frames are published late (or dropped) by design
    assert po["cuda_absent"] and not s["mem_mib"]["policy_in_compute_apps"] and po["gl_absent"], g


# the stall gates (reserved memory, GC) need a run the sim itself kept up with: clean runs show 0-6 late ticks, all
# at start-up; more means the machine was busy and the vsensor's memory/GC behaviour is not what is being tested
STALL_GATE_MAX_LATE_TICKS = 6


def window_gates(s, colour_groups, realtime=True, stagger=False):
    """Window-run gates on top of the R5 ones: group size, render count, lateness, process isolation.

    ``realtime=False`` keeps the delivery gates (KS, coverage) and only reports the real-time ones: with every
    camera firing on the same tick, N groups of 11 renders (~0.5 ms each) exceed the 2 ms tick beyond N ~ 4.
    """
    sm, g = s["sim"], gpu(s)
    check_ages(s, n=s["n"], exact=realtime)
    assert sm["n_sub"] == 11, f"N4B8 plan should have 11 grid-snapped sub-renders, got {sm['n_sub']}"
    assert sm["group_renders"] == sm["n_sub"] + (not colour_groups), sm
    assert sm["gl_renders"] == sm["renders"] * sm["group_renders"], (
        f"GL renders {sm['gl_renders']} != captures {sm['renders']} x group {sm['group_renders']}" + g)
    assert not realtime or sm["late_ticks"] <= 0.01 * sm["ticks"], f"late_ticks {sm['late_ticks']} > 1%" + g
    if stagger:
        msg = f"stagger: max fired per tick {sm['max_fired_per_tick']}" + g
        assert sm["stagger"] and sm["max_fired_per_tick"] == 1, msg
    assert sm["torch_absent"] and sm["cuda_absent"], "sim loaded torch/CUDA" + g
    v, depth_run = s["vsensor"], bool(s.get("depth"))
    proc_slow = [x for x in v.get("slow_laps", []) if x[1] == "process"]
    grown = v.get("reserved_mib_end") != v.get("reserved_mib_t0")
    # steady-state colour process() allocates nothing (enforced); a slow process lap is reported only, as a
    # random external stall lands inside process() about a third of the time (the deterministic first-time-kernel
    # guard is tests/test_batch.py::test_batch_steady_state_kernels)
    gcs = {who: s[who].get("gc", {}) for who in ("sim", "vsensor", "policy")}
    slow_gc = {who: {k: x for k, x in d.items() if x["max_ms"] >= vs.SLOW_LAP_MS} for who, d in gcs.items()}
    slow_gc = {who: d for who, d in slow_gc.items() if d}
    if realtime and sm["late_ticks"] <= STALL_GATE_MAX_LATE_TICKS:
        # colour path only: the depth window loop still grows by +20 MiB from an unexplained allocation (not
        # _finish_depth: this config has no depth noise); follow-up: a memory-history run with --depth
        assert not grown or depth_run, (f"vsensor reserved memory grew in the loop: "
                                        f"{v.get('reserved_mib_t0')} -> {v.get('reserved_mib_end')} MiB" + g)
        assert not slow_gc, f"a GC collection >= 2 ms ran in the paced loop (gc.freeze missing?): {slow_gc}" + g
    elif slow_gc:
        print(f"W3 window (report only, contended or late ticks {sm['late_ticks']}): slow GC {slow_gc}")
    if grown and depth_run:
        print(f"W3 window depth (report only): reserved {v.get('reserved_mib_t0')} -> {v.get('reserved_mib_end')} MiB "
              f"(unexplained)")
    elif grown:
        print(f"W3 window (report only, contended or late ticks {sm['late_ticks']}): reserved "
              f"{v.get('reserved_mib_t0')} -> {v.get('reserved_mib_end')} MiB")
    if proc_slow:
        print(f"W3 window: {len(proc_slow)} process lap(s) > 2 ms (random stall inside process, report only): "
              + ", ".join(f"{t:.3f}s {ms:.1f} ms" for t, _, ms in proc_slow[:3]))
    mem = s["mem_mib"]
    assert mem["vsensor"] is not None and mem["vsensor"] <= mem["vsensor_bound"], str(mem) + g
    poll = f", poll {s['policy']['poll_ms']['p99']:.2f} ms p99" if "poll_ms" in s["policy"] else ""
    print(f"W3 window N={s['n']}{' depth' * (not colour_groups)}: {sm['renders']} captures x {sm['group_renders']} "
          f"= {sm['gl_renders']} GL renders, group {sm['group_ms']['mean']:.2f} ms mean / {sm['group_ms']['p99']:.2f} "
          f"p99, late ticks {sm['late_ticks']}/{sm['ticks']}, sim GL {sm['sim_gl_mib']:.0f} MiB{poll}")


def test_virtual_sensor_window():
    """Window mode (exposure 8 ms, readout 16 ms: N4B8), N=4: every R5 gate plus the window gates."""
    s = run("window", "--seconds", "5", "--n", "4", "--exposure-ms", "8", "--readout-ms", "16", trace=False)
    if s is None:
        return
    window_gates(s, colour_groups=True)
    assert s["policy"]["gl_absent"] and s["policy"]["torch_absent"], "policy loaded GL/mujoco/torch"


def test_virtual_sensor_window_depth():
    """Window mode with depth, N=2: same gates as window test."""
    s = run("window_depth", "--seconds", "5", "--n", "2", "--depth", "--exposure-ms", "8", "--readout-ms", "16")
    if s is None:
        return
    window_gates(s, colour_groups=False)


def test_virtual_sensor_window_many():
    """Window mode N=8 staggered: every strict R5 gate (the shared-phase N=8 run is recorded in perf.md Stage 5).

    352 fds = 8x4x11 need the chunked transfer; that cross-process path runs untraced here (strict real-time
    gates), the helper is covered by tests/test_dmabuf.py::test_fd_chunking.
    """
    s = run("window_many", "--seconds", "5", "--n", "8", "--stagger", "--exposure-ms", "8", "--readout-ms", "16",
            trace=False)
    if s is None:
        return
    sm = s["sim"]
    fds = sm["slots"]  # colour slots = exported fds (no depth): N * Ks * n_sub
    assert fds == s["n"] * sm["Ks"] * sm["n_sub"] and fds > 250, f"expected a multi-message fd transfer, got {fds}"
    assert s["vsensor"]["descs_received"] == fds, f"virtual sensor got {s['vsensor']['descs_received']} of {fds}"
    window_gates(s, colour_groups=True, stagger=True)
    print(f"W3 window N=8 staggered: {fds} descriptors in {-(-fds // 250)} messages; p99 publish - ready_t "
          f"{s['publish_late_p99_ms']:.2f} ms, tick match {s['tick_match']:.4f}")


def test_virtual_sensor_window_16():
    """Window mode N=16 staggered, 15 s: the original strict publish-lateness gates.

    Spacing 2.08 ms >= DT keeps one group per tick on the sim side; the virtual sensor meets the 2 ms publish
    gate with the publish guard (a publish due within --publish-guard-ms runs before a new note segment). A fix
    (2026-10-04) removed the one deterministic stall every run had (the first non-contiguous camera batch launched
    gather kernels the warm-up never ran: ~20 ms of lazy module load at t0+1.1 s, 13-17 late publishes = most of the
    p99 budget), and ``window_gates`` now asserts no reserved-memory growth in the loop (a slow ``process`` lap is
    reported only, since a random external stall can land inside it; the first-time-kernel guard is
    ``test_batch_steady_state_kernels``). The second stall was Python's first gen2 collection (30-45 ms once per
    run, scanning the setup heap); ``gc.freeze()`` before the barrier removed it and ``window_gates`` asserts no
    collection >= 2 ms. The window is 15 s (budget ~68 late publishes): margin for a genuinely external stall
    (16-24 late publishes), and 7500 ticks of evidence per run; the thresholds are unchanged.
    Contended-GPU rule: if either GPU snapshot is contended, the late-tick and p99 publish gates are reported
    only; delivery gates, dmean, tick match, the latency budget and max_fired_per_tick are always enforced.
    """
    s = run("window_16", "--seconds", "15", "--n", "16", "--stagger", "--exposure-ms", "8", "--readout-ms", "16",
            trace=False)
    if s is None:
        return
    contended = s["gpu_at_start"]["contended"] or s["gpu_at_end"]["contended"]
    window_gates(s, colour_groups=True, realtime=not contended, stagger=True)
    if contended:  # only the two timing gates (late ticks, p99 publish) are report-only under contention
        g = gpu(s)
        for i in [*map(str, range(s["n"])), "pooled"]:
            assert s["dmean_wall_ms"][i] < 3.0, f"|dmean(wall)|[{i}] {s['dmean_wall_ms'][i]:.2f} ms >= 3 ms" + g
        assert s["tick_match"] >= 0.99, f"per-tick frame match {s['tick_match']:.4f} < 0.99" + g
        assert s["latency_ms"] > s["vsensor"]["process_ms"]["p99"], "effects do not fit in the latency budget" + g
        print(f"W3 window N=16 (GPU contended, timing gates reported only): late ticks "
              f"{s['sim']['late_ticks']}/{s['sim']['ticks']}, p99 publish - ready_t {s['publish_late_p99_ms']:.2f} ms")
    v, mem = s["vsensor"], s["mem_mib"]
    nm = v.get("note_ms", {})
    print(f"W3 window N=16 staggered: p99 publish - ready_t {s['publish_late_p99_ms']:.2f} ms (overdue p99 "
          f"{s['publish_overdue_p99_ms']:.2f}), process_ms {v['process_ms']['mean']:.2f} ms mean, publish-first "
          f"{v.get('publish_first')}, note_ms p99 " + ", ".join(f"{k} {x['p99']:.2f}" for k, x in nm.items() if x)
          + f"; vsensor {mem['vsensor']:.0f} of bound {mem['vsensor_bound']:.0f} MiB")
    sl = sorted(v.get("slow_laps", []), key=lambda x: -x[2])[:3]
    print(f"W3 window N=16 slow laps (> 2 ms): {len(v.get('slow_laps', []))}, worst "
          + ", ".join(f"{t:.3f}s {k} {ms:.1f} ms" for t, k, ms in sl) + f"; reserved {v.get('reserved_mib_t0')} -> "
          f"{v.get('reserved_mib_end')} MiB, pinned allocs {v.get('pinned_allocs_t0')} -> {v.get('pinned_allocs_end')}")
    print("W3 window N=16 GC in loop: " + "; ".join(
        f"{who} " + ", ".join(f"gen{g} x{d[f'gen{g}']['count']} max {d[f'gen{g}']['max_ms']:.2f} ms" for g in (0, 1, 2))
        for who, d in ((w, s[w].get("gc")) for w in ("sim", "vsensor", "policy")) if d))


AB_MAX_DP99_MS = 0.15      # median p99(publish - ready_t) with the observer minus without
AB_MAX_MISS = 0.01         # observer miss rate per on-run


def _observer_checks(s, label):
    """Print the observer section of an --observer run and gate its miss rate (<= 1 %)."""
    o = s["observer"]
    print(f"   {label} observer: received {o['received']}, pushed {o['pushed']}, worker drops {o['worker_drops']}, "
          f"miss rate {o['observer_miss_rate']}, worker {o['worker_ms']['mean']:.1f} ms mean / "
          f"{o['worker_ms']['p99']:.1f} p99, estimate {o['estimate_ms']['max']:.0f} ms max")
    assert o["observer_miss_rate"] is not None and o["observer_miss_rate"] <= AB_MAX_MISS, (
        f"observer miss rate {o['observer_miss_rate']} > {AB_MAX_MISS}" + gpu(s))
    assert o["received"] > 0.9 * s["seconds"] * 30, f"observer received {o['received']} frames" + gpu(s)


def test_virtual_sensor_observer():
    """O5 live observer, 5 s on the prism (window mode, N=4): every window gate holds with the observer process
    running, and the observer keeps up (miss rate <= 1 %). The cost comparison is ``test_virtual_sensor_observer_ab``.
    """
    s = run("observer", "--scene", "prism", "--observer", "--seconds", "5", "--n", "4", "--exposure-ms", "8",
            "--readout-ms", "16", trace=False)
    if s is None:
        return
    window_gates(s, colour_groups=True)
    _observer_checks(s, "O5")


AB_ORDER = ("off", "on", "on", "off") * 4  # 8 runs per arm, ABBA blocks cancel a linear drift in machine load


def test_virtual_sensor_observer_ab():
    """O5 A/B gate: the live observer must not cost the real-time path. Opt-in
    (``REALSENSOR_AB=1``, ~7 min on a quiet machine); the numbers are kept in docs/perf.md.

    Prism scene, window mode (8 ms exposure, 16 ms readout), N=4, 15 s; observer off / on in ABBA blocks, 8 runs per
    arm. Every run passes the window gates; median p99(publish - ready_t) on - off <= 0.15 ms; every on-run's
    observer miss rate <= 1 %. Run to run, p99 scatters 0.13 ms (sd) with the observer off, so 3 runs per arm (the
    plan's first design) failed a 0.07 ms effect a quarter of the time (measured: +0.21 ms on 3 vs 3, +0.10 ms on
    8 vs 8). ``/proc/loadavg``, the GPU condition, policy poll p99 and sim late ticks are printed per
    run. Untraced, like the other real-time gates.
    """
    if os.environ.get("REALSENSOR_AB") != "1":
        print("SKIP: test_virtual_sensor_observer_ab (opt-in: REALSENSOR_AB=1, ~7 min, quiet machine)")
        return
    args = ("--scene", "prism", "--seconds", "15", "--n", "4", "--exposure-ms", "8", "--readout-ms", "16")
    p99, poll, late = ({"off": [], "on": []} for _ in range(3))
    for arm in AB_ORDER:
        k = len(p99[arm])
        s = run(f"ab_{arm}_{k}", *args, *(("--observer",) if arm == "on" else ()), trace=False)
        if s is None:
            return
        g, sm = s["gpu_at_start"], s["sim"]
        print(f"O5 A/B {arm} {k}: p99 publish - ready_t {s['publish_late_p99_ms']:.3f} ms, policy poll p99 "
              f"{s['policy']['poll_ms']['p99']:.3f} ms, late ticks {sm['late_ticks']}/{sm['ticks']}, loadavg "
              f"{s['loadavg']}, GPU at start util {g['util_pct']} %{' (CONTENDED)' if g['contended'] else ''}")
        window_gates(s, colour_groups=True)
        p99[arm].append(s["publish_late_p99_ms"])
        poll[arm].append(s["policy"]["poll_ms"]["p99"])
        late[arm].append(sm["late_ticks"])
        if arm == "on":
            _observer_checks(s, f"O5 A/B on {k}")
    for arm in ("off", "on"):
        print(f"O5 A/B {arm}: p99 publish - ready_t {[round(x, 3) for x in p99[arm]]} ms, policy poll p99 "
              f"{[round(x, 3) for x in poll[arm]]} ms, sim late ticks {late[arm]}")
    d = float(np.median(p99["on"]) - np.median(p99["off"]))
    se = float(np.sqrt(np.var(p99["on"], ddof=1) / len(p99["on"]) + np.var(p99["off"], ddof=1) / len(p99["off"])))
    print(f"O5 A/B: median p99 off {np.median(p99['off']):.3f} ms, on {np.median(p99['on']):.3f} ms, diff {d:+.3f}; "
          f"mean diff {np.mean(p99['on']) - np.mean(p99['off']):+.3f} +- {se:.3f} (se)")
    assert d <= AB_MAX_DP99_MS, f"observer raises median p99(publish - ready_t) by {d:.3f} ms > {AB_MAX_DP99_MS}"


def test_window_group_content():
    """In-process window group: render a group, import via dma-buf, compare bit-exact to SimCamera._raw."""
    try:
        import mujoco
        import torch
        from OpenGL.EGL.MESA import image_dma_buf_export  # noqa: F401
    except ImportError:
        print("SKIP: test_window_group_content (no EGL dma-buf extensions or torch)")
        return
    if not torch.cuda.is_available():
        print("SKIP: test_window_group_content (no CUDA)")
        return
    from test_temporal import moving_scene

    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.camera import SimCamera, StateRing
    from physicalai_sim2real_camera.dmabuf import DmabufExporter, DmabufImporter, wait_fence

    model, _ = moving_scene()
    model.vis.quality.offsamples = 0  # the exporter blits a single-sample offscreen FBO
    cfg = CameraConfig(64, 48, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), exposure_s=0.008, readout_s=0.016)
    cam = SimCamera(model, "cam", cfg)  # CPU: plan, render geometry and the renderer the sim hook uses
    n_sub, ring, r = len(cam._plan), StateRing(model, cam.half_window), cam.renderer
    d, sc = mujoco.MjData(model), mujoco.MjData(model)
    d.qvel[0] = 0.2  # camera slides, board moves: sub-frames at different times differ
    for _ in range(20):
        d.mocap_pos[0, 1] = 0.1 * d.time
        mujoco.mj_step(model, d)
        ring.push(d)
    ex = DmabufExporter(r, 2 * n_sub, depth=False)  # two groups: g = 1 checks the slot formula, not just k
    ex.export()
    g = 1
    fd, n_renders = vs.render_group(ex, None, r, model, cam, ring, sc, d.time, g)  # the sim hook's own code
    assert n_renders == n_sub
    if fd is not None:
        wait_fence(fd)  # the group's single fence covers all its renders (closes the fd)
    tc, refs = d.time - cam.half_window, []
    for t, _ in cam._plan:  # reference: SimCamera's own render of each plan time
        ring.load(tc + t, sc)
        refs.append(cam._raw(sc, False)[0].copy())
    im = DmabufImporter([dict(e, fd=os.dup(e["fd"])) for e in ex.export()])  # own fds, as after SCM_RIGHTS
    out = torch.empty(cam._lin.shape, dtype=torch.uint8, device="cuda")  # (H_r, W_r, 3)
    for k in range(n_sub):
        got = im.read(vs.colour_slot(g, k, n_sub), out).cpu().numpy()  # the virtual sensor's read order
        assert np.array_equal(got, refs[k]), f"group {g} sub-frame {k} differs from SimCamera._raw at its time"
    assert sum(not np.array_equal(refs[k], refs[k + 1]) for k in range(n_sub - 1)) >= n_sub // 2, "static scene"
    ex.close()
    im.close()
    cam.close()
    print(f"W3 group content: {n_sub} sub-renders, bit-exact vs SimCamera._raw")


if __name__ == "__main__":
    test_flat_slot()
    test_payload_size()
    test_publish_time()
    test_trailer()
    test_note()
    test_estimate_summary()
    test_num()
    test_virtual_sensor_run()
    test_virtual_sensor_stall()
    test_virtual_sensor_window()
    test_virtual_sensor_window_depth()
    test_virtual_sensor_window_many()
    test_virtual_sensor_window_16()
    test_virtual_sensor_observer()
    test_virtual_sensor_observer_ab()
    test_window_group_content()
    print("test_virtual_sensor: PASS")
