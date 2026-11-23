"""Closed loop in the simulator and the replay API that a recording also goes through.

``sim_records`` drives the prism scene through a ``CameraStream`` (distortion, noise, exposure / readout window,
latency, jitter, drops) and yields ``FrameRecord``s in arrival order. A record carries only what a real camera
delivers: the frame, the observer's receipt time ``t_arrival`` (robot-state clock; the virtual sensor delivers at the
frame's ready time), the **sensor** timestamp ``ts_sensor = (1 + rho) capture_t + c0 + N(0, 50 us)`` (``sensor_clock``;
the true capture time never reaches the observer), the transport ``seq``, the camera ``frame_number`` and
the exposure / gain metadata. Observer misses are published frames that never reach the observer (``seq`` advances).
The head trajectory goes to a ``HeadPoseRing`` the way the robot reports it: the true pose ``proprio_delay_s`` ago
plus noise (leakage item (d)).

``Observer`` runs the three estimators on one stream with one prism detection per frame: the geometric observer
gets the detections; the timing observer gets each frame one frame late with its |omega| from the face rotations
(solvePnP with the current K) of its received neighbours (``omega_triplet``, which drops pose flips); the photometric
observer gets the wedge patch boxes and the reported head speed around the estimated capture time. ``replay`` feeds
records and estimates at 1 Hz. O7 replays a real recording through the same ``replay``: ``record`` writes a stream
to disk while passing it on, ``load_recording`` reads it back as records plus a ``HeadLog`` that shows, at each frame,
the head samples the ring held when that frame arrived (so a replay reproduces the in-memory run: bit-identical
inputs, estimates equal up to multi-threaded summation order).

Recording layout (``docs/capture_protocol.md`` D7)::

    <dir>/info.json      {"ring_n": 16384, "head_base": int, "frame_number_bits": null | int, "seq_bits": null | int}
    <dir>/frames.jsonl   one line per received frame: t_arrival, ts_sensor, seq, frame_number, exposure, gain,
                         head_count (head samples known at arrival)
    <dir>/frames/000000.png ...  lossless frames (RGB stored as PNG's BGR)
    <dir>/head.npy       (m, 11) float64 head samples: t (robot clock), pos (3), quat wxyz (4), omega (3)
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings

import cv2
import mujoco
import numpy as np

from .observer_geom import MIN_CORNERS, GeomObserver
from .observer_photo import PhotoObserver
from .observer_timing import DELTA_MAX_S, TimingObserver, omega_triplet, unwrap_counter
from .target import PrismTarget
from .timing import CameraStream

RHO, C0, TS_NOISE_S = 50e-6, 1.234, 50e-6  # sensor clock: drift, offset (unknown to the observer), stamp noise
PROPRIO_DELAY_S = 0.004                    # reported head state lags the true pose (leakage item (d))
HEAD_NOISE = (2e-4, 1e-3, 5e-3)            # reported head pos (m), orientation (rad), angular velocity (rad/s) σ
CAPTURE_HALF_S = 0.02                      # half-width of the photometric speed window around the capture time
EDGE_MARGIN_PX = 3                         # photometric boxes keep this far from the frame edge (> BOX_TOL_PX)
DARK_CROP_PX = 16                          # side of the centred crop taken from each black square


@dataclasses.dataclass
class FrameRecord:
    """One frame as the observer receives it (no true capture time, on purpose)."""

    frame: np.ndarray
    t_arrival: float
    ts_sensor: float
    seq: int
    frame_number: int
    exposure: float | None = None
    gain: float | None = None


def sensor_clock(capture_t, rng, rho=RHO, c0=C0, noise_s=TS_NOISE_S):
    """Camera timestamp of a capture: ``(1 + rho) capture_t + c0 + N(0, noise_s)`` (offset and drift unknown)."""
    return (1.0 + rho) * capture_t + c0 + rng.normal(0.0, noise_s)


def perturb_prism(target, rng, trans_m=0.5e-3, rot_deg=0.3):
    """The observer's copy of the prism: each face pose moved by ~``trans_m`` / ``rot_deg`` (leakage item (b)).

    The fit uses face-frame corners only (per-face planar views), so this must not change the estimate.

    Returns:
        A new ``PrismTarget``.
    """
    faces = []
    for f in target.faces:
        T = f.T_prism_face.copy()
        T[:3, :3] = cv2.Rodrigues(rng.normal(0.0, np.radians(rot_deg), 3))[0] @ T[:3, :3]
        T[:3, 3] += rng.normal(0.0, trans_m, 3)
        faces.append(dataclasses.replace(f, T_prism_face=T))
    return PrismTarget(faces, target.apothem_m, target.height_m, target.working_distance_m, target.focal_px)


def write_head(ring, sweep, t, rng, delay_s=PROPRIO_DELAY_S, noise=HEAD_NOISE):
    """Write the head state the robot reports at ``t``: the true pose ``delay_s`` ago, plus noise."""
    pos, quat = sweep.pose(t - delay_s)
    dq = np.zeros(4)
    ax = rng.normal(0.0, 1.0, 3)
    mujoco.mju_axisAngle2Quat(dq, ax / np.linalg.norm(ax), abs(rng.normal(0.0, noise[1])))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, quat, dq)
    ring.write(t, pos + rng.normal(0.0, noise[0], 3), q, sweep.omega(t - delay_s) + rng.normal(0.0, noise[2], 3))


def sim_records(model, data, sweep, cam, cfg, duration_s, ring=None, miss_prob=0.01, seed=0, gain=1.0, stats=None):
    """Drive the sweep for ``duration_s`` and yield the frames the observer receives, in arrival order.

    Args:
        model: Prism scene model (``scenes.prism_scene``).
        data: Its data.
        sweep: ``HeadSweep`` driving the mocap head.
        cam: ``SimCamera`` on the head camera (its ``cfg`` carries the true optics, noise and timing).
        cfg: The same ``CameraConfig`` (fps, latency, jitter, drops, exposure / readout for the stream).
        duration_s: Simulated seconds.
        ring: Optional ``HeadPoseRing`` that receives the reported head state every tick.
        miss_prob: Probability that the observer misses a published frame.
        seed: Seed of the clock noise, misses and head noise.
        gain: Gain metadata reported with every frame.
        stats: Optional dict that receives the realised truth the records hide: ``frame_numbers`` (every rendered
            capture) and ``dropped`` (captures the camera dropped between the first and last rendered one).

    Yields:
        ``FrameRecord`` per received frame, once the simulation reaches its ready time.
    """
    rng = np.random.default_rng([seed, 7])
    rendered = []
    stream = CameraStream(cam, cfg, render=lambda d, n: rendered.append(cam.render(d)) or len(rendered) - 1)
    inflight, seq, last = [], 0, None
    while data.time < duration_s:
        t = data.time
        data.mocap_pos[0], data.mocap_quat[0] = sweep.pose(t)
        mujoco.mj_forward(model, data)  # the render and the state ring see the pose of data.time (no 1-step lag)
        if ring is not None:
            write_head(ring, sweep, t, rng)
        stream.step(data)
        if stream.last_render is not None and stream.last_render is not last:
            last = stream.last_render
            capture_t, ready_t, n = last
            inflight.append((ready_t, n, sensor_clock(capture_t, rng), rendered.pop()))
            if stats is not None:
                fns = stats.setdefault("frame_numbers", [])
                fns.append(n)
                stats["dropped"] = fns[-1] - fns[0] + 1 - len(fns)
        inflight.sort(key=lambda r: r[0])
        while inflight and inflight[0][0] <= t + 1e-9:
            ready_t, n, ts, frame = inflight.pop(0)
            seq += 1
            if rng.random() >= miss_prob:
                yield FrameRecord(frame, ready_t, ts, seq, n, cfg.exposure_s, gain)
        mujoco.mj_step(model, data)


class Observer:
    """The geometric, photometric and timing observers on one frame stream; see the module docstring."""

    def __init__(self, target, size, fovy_deg, gamma, black_level_lin, exposure_s=0.0, readout_s=0.0,
                 proprio_delay_s=PROPRIO_DELAY_S, eps_speed=0.03):
        """Create the three estimators from offline inputs only (never the simulator's own config).

        Args:
            target: The observer's ``PrismTarget`` (``perturb_prism`` in the closed loop).
            size: ``(W, H)`` px.
            fovy_deg: Nominal vertical FOV (spec sheet / MJCF) for the initial K.
            gamma: Offline gamma (D4, ``scripts/calibrate.py``).
            black_level_lin: Offline black level (D3).
            exposure_s: Exposure metadata (motion gate).
            readout_s: Offline rolling-shutter readout (D5b; 0 = global shutter).
            proprio_delay_s: Known delay of the reported head state.
            eps_speed: Max reported head |omega| (rad/s) for a still frame (head noise: 5e-3 per axis, ~9e-3 in norm).
        """
        self.target = target
        self.wedge_face = next((f.index for f in target.faces if f.wedge), None)
        self.geom = GeomObserver(target, size, fovy_deg, exposure_s=exposure_s, readout_s=readout_s)
        self.photo = PhotoObserver(gamma, black_level_lin, eps_speed=eps_speed)
        self.timing = TimingObserver(readout_s=readout_s, height_px=size[1], proprio_delay_s=proprio_delay_s)
        self._hist = []    # the two previous received frames: (record, {face: (R, row_mean_px, n_corners)})
        self._delta = None  # camera-vs-head lag and clock line from the last estimate (photometric capture time)
        self._line = None

    def _poses(self, detections):
        out = {}
        for face, ids, px, obj in detections:
            if len(ids) < MIN_CORNERS:
                continue
            ok, rvec, tvec = cv2.solvePnP(obj, px, self.geom.K, np.asarray(self.geom.dist), flags=cv2.SOLVEPNP_IPPE)
            if ok:
                out[face] = (cv2.Rodrigues(rvec)[0], float(px[:, 1].mean()), len(ids), rvec, tvec)
        return out

    def _push_timing(self, nxt=None):
        """Hand the middle frame of (previous, middle, next) to the timing observer with its |omega|."""
        rec, poses = self._hist[-1]
        omega, span = None, None
        if nxt is not None and len(self._hist) == 2:
            (r0, p0), (r2, p2) = self._hist[0], nxt
            common = [f for f in poses if f in p0 and f in p2]
            if common:
                f = max(common, key=lambda f: poses[f][2])
                # each pose at the time its face's rows were read (rolling shutter): the sample spans those two times
                k = self.timing.readout_s / self.timing.H
                t0, t1, t2 = (r.ts_sensor + (p[f][1] - self.timing.H / 2) * k
                              for r, p in ((r0, p0), (rec, poses), (r2, p2)))
                omega = omega_triplet(p0[f][0], poses[f][0], p2[f][0], t0, t1, t2)
                span = (t0, t2)
        self.timing.push(rec.t_arrival, rec.ts_sensor, rec.seq, rec.frame_number, omega, None, span)

    def push(self, rec, omega_head_fn):
        """Feed one received frame.

        Args:
            rec: ``FrameRecord``.
            omega_head_fn: ``t -> |omega_head|`` (vectorised, ``nan`` outside the record) on the robot-state clock.
        """
        dets = self.target.detect(rec.frame)
        self.geom.push_detections(dets, rec.t_arrival, rec.ts_sensor)
        poses = self._poses(dets)
        if self._hist:
            self._push_timing((rec, poses))
        self._hist = (self._hist + [(rec, poses)])[-2:]
        # photometric stillness at the capture time (arrival - lag): before the first latency estimate, the whole
        # search range; reported-state noise needs eps_speed above it
        a = rec.t_arrival
        if self._delta is None:
            lo, hi = a - DELTA_MAX_S, a
        else:  # frame time from the sensor stamp (no per-frame jitter) minus the lag = capture time on the robot clock
            c = self._line[0] * rec.ts_sensor + self._line[1] - self._delta
            lo, hi = c - CAPTURE_HALF_S, c + CAPTURE_HALF_S
        w = omega_head_fn(np.linspace(lo, hi, 9))
        speed = float("inf") if np.isnan(w).any() else float(w.max())
        boxes, dark = [], []
        if self.wedge_face in poses:
            _, _, _, rvec, tvec = poses[self.wedge_face]
            W, H = self.geom.size
            K, d = self.geom.K, np.asarray(self.geom.dist)
            m = EDGE_MARGIN_PX  # a box within a few px of the frame edge flips in and out with pose noise

            def inside(bs):
                return [b for b in bs if b[0] >= m and b[1] >= m and b[2] <= W - m and b[3] <= H - m]
            boxes = inside(self.target.wedge_patches(self.wedge_face, rvec, tvec, K, d))
            # black squares: the level at the black level pins sigma_read; a centred DARK_CROP_PX crop
            # is plenty for the read noise and keeps the interval memory small
            dark = [((x0 + x1) // 2 - DARK_CROP_PX // 2, (y0 + y1) // 2 - DARK_CROP_PX // 2,
                     (x0 + x1) // 2 + DARK_CROP_PX // 2, (y0 + y1) // 2 + DARK_CROP_PX // 2)
                    for x0, y0, x1, y1 in self.target.black_squares(self.wedge_face, rvec, tvec, K, d)
                    if min(x1 - x0, y1 - y0) >= DARK_CROP_PX]
            dark = inside(dark)
        self.photo.push(rec.frame, speed, boxes, rec.exposure, rec.gain, dark_boxes=dark)

    def flush(self):
        """End of stream: hand over the last frame and close the photometric interval."""
        if self._hist:
            self._push_timing()
            self._hist = []
        self.photo.flush()

    def estimate(self, omega_head_fn, rng=None):
        """Current estimates of all three observers.

        Returns:
            dict ``geom`` (``GeomObserver.estimate``), ``photo`` (``PhotoObserver.estimate``), ``timing``
            (``TimingObserver.estimate``) and ``health`` (counters: frames, gate rejections, misses).
        """
        timing = self.timing.estimate(omega_head_fn, rng=rng)
        if np.isfinite(timing.get("delta", np.nan)):
            self._delta, self._line = timing["delta"], timing["clock_line"]
        g, p = self.geom, self.photo
        health = dict(n_frames=g.n_pushed, geom_accepted=g.n_accepted, geom_rejected_motion=g.n_rejected_motion,
                      geom_rejected_gap=g.n_rejected_gap, geom_evicted=g.n_evicted, photo_still=p.n_still,
                      photo_rejected_motion=p.n_rejected_motion, photo_rejected_drift=p.n_rejected_drift,
                      observer_misses=timing.get("n_observer_misses", 0))
        return dict(geom=self.geom.estimate(), photo=self.photo.estimate(), timing=timing, health=health)


def replay(records, observer, head, every_s=1.0, rng=None):
    """Feed ``records`` to ``observer`` and estimate every ``every_s`` of arrival time (and once at the end).

    Args:
        records: Iterable of ``FrameRecord`` (``sim_records`` or a recording).
        observer: ``Observer``.
        head: Object with ``omega_fn()`` (``HeadPoseRing``); read after every frame, so only head data up to the
            frame's arrival is used.
        every_s: Estimate period, s.
        rng: Bootstrap generator for the latency std.

    Returns:
        List of ``(t_arrival, estimate)``; the last entry is after ``flush``.
    """
    series, t_next, a = [], None, None
    for rec in records:
        a = rec.t_arrival
        observer.push(rec, head.omega_fn())
        t_next = a + every_s if t_next is None else t_next
        if a >= t_next:
            series.append((a, observer.estimate(head.omega_fn(), rng=rng)))
            while t_next <= a:  # after a pause in the recording: one estimate, not one per missed period
                t_next += every_s
    observer.flush()
    series.append((a, observer.estimate(head.omega_fn(), rng=rng)))
    return series


class HeadLog:
    """Recorded head samples replayed the way a ``HeadPoseRing`` showed them.

    ``omega_fn()`` covers the first ``count`` rows (``load_recording`` sets ``count`` and ``base`` before each frame,
    so use its generator lazily: materialising it first leaves them at the last frame), limited like
    ``HeadPoseRing.read``:
    the ring held ``count + base`` samples and shows the last ``ring_n - 1`` of them once that exceeds ``ring_n``
    (``base`` = samples the ring had dropped before the recording started).
    """

    def __init__(self, rows, ring_n, base=0):
        """Hold ``rows`` ``(m, 11)``, the ring size of the live run and the ring samples before row 0."""
        self.rows, self.ring_n, self.base = np.asarray(rows, np.float64), int(ring_n), int(base)
        self.count = len(self.rows)

    def omega_fn(self):
        """``t -> |omega|`` interpolated over the visible samples, ``nan`` outside them (as ``HeadPoseRing``)."""
        k = self.count
        shown = self.ring_n - 1 if k + self.base > self.ring_n else k + self.base
        r = self.rows[max(k - shown, 0):k]
        t, w = r[:, 0], np.linalg.norm(r[:, 8:11], axis=1)
        return lambda q: np.interp(np.asarray(q, float), t, w, left=np.nan, right=np.nan)


def _num_or_none(x):
    return None if x is None else float(x)


def record(records, ring, path, frame_number_bits=None, seq_bits=None):
    """Write a stream to ``path`` while yielding it on (a tee in front of ``replay``); see the module docstring.

    Head samples are taken from ``ring`` by time (rows newer than the last saved one), so a ring that has already
    wrapped when recording starts, or a writer in another process that adds samples while the ring is read, still
    gives rows in sample order; ``head_count`` is the number of saved rows at each frame. The ring's tail after the
    last frame is saved too (the final estimate after ``flush`` sees it), and the files are written in a ``finally``
    so an interrupted recording keeps its head samples.

    Args:
        records: ``FrameRecord`` iterable (``sim_records`` or a live receiver); frames uint8 / uint16, ``(H, W)`` or
            ``(H, W, 3)`` RGB.
        ring: ``HeadPoseRing`` the head state is written to.
        path: Output directory (created).
        frame_number_bits: Width of the camera's frame counter when it wraps (``None`` = never).
        seq_bits: Width of the transport sequence counter when it wraps.

    Yields:
        The same records.
    """
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    head, t_saved, saved, base = [], -np.inf, 0, None

    def take():
        nonlocal t_saved, saved, base
        k, r = ring.count, ring.read()
        new = r[r[:, 0] > t_saved]
        if base is None:  # samples the ring dropped before the recording: HeadLog needs them for its window
            base = max(k - len(r), 0)  # a writer adding samples during the read must not make it negative
        elif k - base - saved > len(new):
            warnings.warn(f"record: {k - base - saved - len(new)} head samples lost (more than the ring holds "
                          "arrived between two frames)")
            base = k - saved - len(new)
        if len(new):
            head.append(new)
            t_saved, saved = new[-1, 0], saved + len(new)

    try:
        with open(os.path.join(path, "frames.jsonl"), "w") as f:
            for i, rec in enumerate(records):
                take()
                fr = rec.frame
                if fr.dtype not in (np.uint8, np.uint16) or not (fr.ndim == 2 or fr.shape[-1] == 3):
                    raise ValueError(f"record: frames must be uint8/uint16 (H, W) or (H, W, 3), got {fr.dtype} "
                                     f"{fr.shape}")
                cv2.imwrite(os.path.join(path, "frames", f"{i:06d}.png"), fr[..., ::-1] if fr.ndim == 3 else fr)
                f.write(json.dumps(dict(t_arrival=float(rec.t_arrival), ts_sensor=float(rec.ts_sensor),
                                        seq=int(rec.seq), frame_number=int(rec.frame_number),
                                        exposure=_num_or_none(rec.exposure), gain=_num_or_none(rec.gain),
                                        head_count=saved, head_base=base)) + "\n")
                f.flush()
                yield rec
        take()  # the tail after the last frame (not in finally: an interrupted caller may have closed the ring)
    finally:
        np.save(os.path.join(path, "head.npy"), np.concatenate(head) if head else np.zeros((0, 11)))
        with open(os.path.join(path, "info.json"), "w") as f:
            json.dump(dict(ring_n=ring.n, head_base=base or 0, frame_number_bits=frame_number_bits,
                           seq_bits=seq_bits), f)


def load_recording(path):
    """Read a recording written by ``record`` (or by a real-camera recorder in the same layout).

    Counters are unwrapped with the widths in ``info.json``.

    Args:
        path: Recording directory.

    Returns:
        ``(records, head)``: a generator of ``FrameRecord`` in arrival order and the ``HeadLog`` to pass to
        ``replay`` (the generator advances it to each frame's ``head_count``, and to every row after the last frame,
        as the live ring at ``flush``).
    """
    with open(os.path.join(path, "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(path, "frames.jsonl")) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    head = HeadLog(np.load(os.path.join(path, "head.npy")), info["ring_n"], info.get("head_base", 0))
    fn, sq = [x["frame_number"] for x in lines], [x["seq"] for x in lines]
    if info.get("frame_number_bits"):
        fn = unwrap_counter(fn, info["frame_number_bits"])
    if info.get("seq_bits"):
        sq = unwrap_counter(sq, info["seq_bits"])

    def gen():
        for i, x in enumerate(lines):
            head.count, head.base = x["head_count"], x.get("head_base", head.base)
            img = cv2.imread(os.path.join(path, "frames", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            img = np.ascontiguousarray(img[..., ::-1]) if img.ndim == 3 else img
            yield FrameRecord(img, x["t_arrival"], x["ts_sensor"], int(sq[i]), int(fn[i]), x["exposure"], x["gain"])
        head.count = len(head.rows)

    return gen(), head
