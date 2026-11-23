"""Camera timing: capture schedule, latency/jitter/drop, sample-and-hold delivery.

Latency reference point (same as real-robot measurement C5/D5):
``latency_s`` = mid-exposure -> frame available to the policy process.
With an exposure/readout window (``half = (exposure_s + readout_s)/2``) the
capture fires ``half`` late, when the whole window is in the past, and
``capture_t = data.time - half`` stays mid-exposure; with half=0 this is
``capture_t = data.time``.
The policy sees age = t_ctrl - capture_t, uniform on [latency, latency + 1/fps]
(+ jitter) under sample-and-hold; that age distribution is the sim<->real metric.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

_EPS = 1e-9


def schedule(cfg, n_frames, rng):
    """Offline/test-only schedule -> (t_c, latency, dropped) arrays.

    Its random sequence need not match ``CameraStream`` (which draws per capture).
    """
    t_c = cfg.phase_s + np.arange(n_frames) / cfg.fps
    latency = np.full(n_frames, float(cfg.latency_s))
    if cfg.jitter_k > 0:
        latency += rng.gamma(cfg.jitter_k, cfg.jitter_theta, n_frames)
    dropped = rng.random(n_frames) < cfg.drop_prob
    return t_c, latency, dropped


def latest_ready(capture_t, ready_t, t):
    """Index of max ``capture_t`` among entries with ``ready_t <= t``; None if none."""
    ready = np.flatnonzero(np.asarray(ready_t) <= t + _EPS)
    if ready.size == 0:
        return None
    return int(ready[np.argmax(np.asarray(capture_t)[ready])])


class CameraStream:
    """Sample-and-hold camera stream around any ``cam`` with ``.render(data)``.

    - Capture index n fires when ``data.time + 1e-9 - half >= phase_s + n/fps``;
      ``capture_t = data.time - half``, ``ready_t = capture_t + latency``.
    - If ``cam`` has ``push(data)`` it is called every step (state ring for
      windowed renders).
    - Several capture times passing in one step (ctrl dt > 1/fps): one render,
      assigned to the latest index; earlier ones are missed (no rng draws).
    - Per capture: one Gamma(k, theta) latency draw if jitter_k > 0 (latency =
      latency_s + draw) and one drop draw from ``default_rng([seed + cam_id, 1])``. Dropped frames are not
      rendered, and no earlier missed frame is rendered instead.
    - ``step`` returns the ready frame with the latest capture_t and holds it
      until a newer one is ready; never goes back in time.
    - ``render(data, n)``, if given, replaces ``cam.render(data)``; whatever it
      returns (slot index, tensor, array) is the handle held and returned.
    - Handle lifetime: a capture is rendered when it fires but delivered
      ``latency`` later, so up to ``ceil(latency * fps) + 1`` renders are in
      flight. A returned handle must stay valid and unmodified until a newer
      frame is delivered (``n`` has moved past it); a single reused buffer
      delivers pixels newer than ``capture_t``. Use a ring of
      ``K = ring_depth(cfg)`` slots keyed ``r % K``, r = number of earlier renders
      (0 for the first render; not ``n``: dropped captures leave gaps in n, so ``n % K`` wraps early).
    - Readiness is keyed on ``capture_t``; a handle of None still counts as a
      ready frame (the returned ``capture_t``/``n`` are then not None).
    - ``last_render``: ``(capture_t, ready_t, n)`` of the latest rendered (non-dropped)
      capture, None before the first; lets a caller that ships the frame elsewhere at
      render time (``examples/virtual_sensor.py``) deliver it at ``ready_t`` itself.
    """

    def __init__(self, cam, cfg, cam_id=None, render=None):
        """Create the stream.

        Args:
            cam: Camera with ``render(data)`` (unused when ``render`` is given; may
                then be None) and optional ``push(data)`` called every step.
            cfg: ``CameraConfig``; uses fps, phase_s, latency_s, jitter_k,
                jitter_theta, drop_prob, exposure_s, readout_s and seed.
            cam_id: Seed offset, rng = ``default_rng([cfg.seed + cam_id, 1])``;
                defaults to ``cam.cam`` (or 0), the same rule as SimCamera's FPN.
            render: Optional hook ``render(data, n) -> handle`` called instead of
                ``cam.render(data)`` for each non-dropped capture with index n; the
                returned handle (slot int, tensor, array, ...) is held and returned
                by ``step``. The stream holds it by reference and never copies it,
                so it must stay valid until superseded (see class docstring). The
                hook must not raise; stream state is undefined if it does.

        Raises:
            ValueError: If ``latency_s`` is shorter than the half exposure window,
                or if there is neither a ``render`` hook nor a callable ``cam.render``.
        """
        if render is None and not callable(getattr(cam, "render", None)):
            raise ValueError("need a render hook or a cam with render(data)")
        self.cam = cam
        self._render = render
        if cam_id is None:  # same s = seed + cam_id as SimCamera's FPN
            cam_id = getattr(cam, "cam", 0)
        self._rng = np.random.default_rng([cfg.seed + cam_id, 1])
        self._fps = float(cfg.fps)
        self._phase = float(cfg.phase_s)
        self._lat = float(cfg.latency_s)
        self._k = float(cfg.jitter_k)
        self._theta = float(cfg.jitter_theta)
        self._drop = float(cfg.drop_prob)
        self._half = (cfg.exposure_s + cfg.readout_s) / 2
        if self._lat < self._half:  # a frame can't exist before its window ends
            raise ValueError(f"latency_s={self._lat} < half exposure window={self._half}")
        self._push = getattr(cam, "push", None)
        if not callable(self._push):
            self._push = None
        self._next = 0  # next capture index not yet passed
        # ponytail: plain list, no maxlen -- Gamma jitter is unbounded, maxlen would drop silently
        self._buf = []  # [capture_t, ready_t, n, frame], in capture order
        self._last = None  # held entry
        self._none = (None, None, None)
        self.last_render = None  # (capture_t, ready_t, n) of the latest rendered capture

    def step(self, data):
        """Advance to ``data.time``: maybe capture, then deliver the newest ready frame.

        Args:
            data: Object with ``.time`` (``mujoco.MjData``); passed to ``push`` and
                to the renderer.

        Returns:
            ``(frame, capture_t, n)`` of the held frame (the handle, its
            mid-exposure time and capture index), or ``(None, None, None)``
            until the first frame is ready.
        """
        if self._push is not None:
            self._push(data)
        t = data.time - self._half
        n = math.floor((t + _EPS - self._phase) * self._fps)
        if n >= self._next:
            self._next = n + 1
            lat = self._lat
            if self._k > 0:
                lat += self._rng.gamma(self._k, self._theta)
            if not self._rng.random() < self._drop:
                f = self._render(data, n) if self._render is not None else self.cam.render(data)
                self._buf.append((t, t + lat, n, f))
                self.last_render = (t, t + lat, n)

        # latest capture_t among ready entries (buffer is capture-ordered)
        tr = data.time + _EPS
        best = None
        for e in self._buf:
            if e[1] <= tr:
                best = e
        last = self._last
        if best is not None and (last is None or best[0] > last[0]):
            self._last = last = best
            c = best[0]
            self._buf = [e for e in self._buf if e[0] >= c]  # strict < pruned
        if last is None:
            return self._none
        return last[3], last[0], last[2]


def stagger_phases(cfg, n):
    """Evenly staggered ``phase_s`` for n cameras: ``cfg.phase_s + i / (n * fps)``.

    Spreads the n capture instants over one frame period so at most
    ``ceil(dt * n * fps)`` cameras fire on the same control tick of length dt.

    Args:
        cfg: ``CameraConfig`` (uses ``phase_s`` and ``fps``).
        n: Number of cameras.

    Returns:
        ``(n,)`` float64 array of per-camera phases for ``CameraStreamBatch(phase_s=)``.
    """
    return float(cfg.phase_s) + np.arange(n) / (n * float(cfg.fps))


def ring_depth(cfg):
    """Slots per camera for a render ring that no in-flight capture overwrites.

    ``K = ceil((latency_s + jitter_hi) * fps) + 2``: renders after the held frame
    are all not yet ready, so they span less than ``latency + 1/fps``, i.e. at most
    ``ceil(latency * fps) + 1`` renders, +1 for the held frame. Valid for a ring
    keyed on the render count (``r % K``), not on the capture index.
    ``jitter_hi = (ke + 6*sqrt(ke)) * theta`` with ``ke = max(k, 1)``: mean + 6 sigma
    of Gamma(ke, theta), at or above the 0.999 quantile for ``k >= 1``
    (exponential: 7 theta vs 6.92 theta). The Gamma tail grows with k, so clamping
    the shape to 1 also covers ``k < 1`` (k=0.5: 7 theta vs q999 5.41 theta),
    where the unclamped mean + 6 sigma falls short.

    Args:
        cfg: ``CameraConfig`` (fps, latency_s, jitter_k, jitter_theta).

    Returns:
        Ring depth K >= 2.
    """
    k, th = float(cfg.jitter_k), float(cfg.jitter_theta)
    ke = max(k, 1.0)
    hi = (ke + 6 * math.sqrt(ke)) * th if k > 0 else 0.0
    return math.ceil((cfg.latency_s + hi) * cfg.fps - 1e-9) + 2


class CameraStreamBatch:
    """n independent camera timings; stream i == ``CameraStream(cfg_i, cam_id=i)`` sample for sample.

    ``cfg_i`` is ``cfg`` with ``phase_s[i]`` substituted when ``phase_s`` is given (see
    ``stagger_phases``), else ``cfg`` itself.

    Each camera has its own rng ``default_rng([seed + i, 1])``, schedule, drop and
    sample-and-hold state; the batch only loops over n ``CameraStream`` objects.
    Streams are seeded with ``cam_id=i`` (the batch index), not ``cams[i].cam``:
    batch cameras usually share one MuJoCo camera id, which would give them
    identical draws. Argument order is ``(cfg, n)`` (the plan sketch had
    ``(n, cfg)``); pass ``n=`` by keyword if in doubt.

    Handle lifetime is the caller's job, as in ``CameraStream``: render into a
    ring of ``K = ring_depth(cfg)`` slots per camera, ``slot = (i, r_i % K)``
    with ``r_i`` = number of earlier hook calls for camera i (0 for the first
    render; not ``n_i``: drops leave gaps in n),
    never a single slot per camera.
    """

    def __init__(self, cfg, n, seed=None, render=None, cams=None, phase_s=None):
        """Create n streams sharing one config.

        Args:
            cfg: ``CameraConfig`` shared by all cameras.
            n: Number of cameras.
            seed: Overrides ``cfg.seed`` (via ``dataclasses.replace``); camera i
                then draws from ``default_rng([seed + i, 1])``.
            render: Hook ``render(i, data_i, n_i) -> handle`` called only for camera
                i's non-dropped captures; its return is the handle camera i holds.
                The handle must stay valid and unmodified until ``seq[i]`` has
                moved past ``n_i`` (use ``ring_depth``). The hook must not raise;
                stream state is undefined if it does. Required unless every
                ``cams[i]`` has a callable ``render(data)``.
            cams: Optional per-camera objects; ``cams[i].push(data_i)`` (if
                present) runs every step, ``cams[i].render`` is used when
                ``render`` is None.
            phase_s: Optional sequence of n per-camera phases (seconds); camera i
                then uses ``dataclasses.replace(cfg, phase_s=phase_s[i])``.
                None: every camera uses ``cfg.phase_s``.
                See ``stagger_phases``.

        Raises:
            ValueError: If there is no ``render`` hook and some camera lacks a
                callable ``render``, or ``phase_s`` does not have length n.
        """
        if seed is not None:
            cfg = dataclasses.replace(cfg, seed=seed)
        if cams is None:
            cams = [None] * n
        if render is None and not all(callable(getattr(c, "render", None)) for c in cams):
            raise ValueError("need a render hook or cams that all have render(data)")
        if phase_s is None:
            cfgs = [cfg] * n
        else:
            if len(phase_s) != n:
                raise ValueError(f"phase_s has {len(phase_s)} entries for {n} cameras")
            cfgs = [dataclasses.replace(cfg, phase_s=float(p)) for p in phase_s]
        self._n = n
        self._fired = np.zeros(n, dtype=bool)
        self._prev_seq = np.full(n, -1, dtype=np.int64)

        def fire(i, d, k):
            self._fired[i] = True
            return render(i, d, k) if render is not None else cams[i].render(d)

        # ponytail: Python loop over n streams; vectorise if n >> 64 shows up in profiles
        self.streams = [CameraStream(cams[i], cfgs[i], cam_id=i, render=lambda d, k, i=i: fire(i, d, k))
                        for i in range(n)]

    def step(self, datas):
        """Advance every camera by one control step.

        Args:
            datas: Sequence of exactly n objects with ``.time`` (one ``MjData``
                per env); a length mismatch raises ValueError.

        Returns:
            ``(ready, capture_t, seq, handles, new, fired)``:

            - ``ready`` (n,) bool: camera i holds a frame (keyed on capture_t).
            - ``capture_t`` (n,) float64: mid-exposure time of the held frame,
              nan if not ready.
            - ``seq`` (n,) int64: capture index of the held frame, -1 if not ready.
            - ``handles``: list of the held handles, None if not ready.
            - ``new`` (n,) bool: ``seq`` changed this step (a newer frame was
              delivered).
            - ``fired`` (n,) bool: camera i's render was called this step.

            Held values persist until a newer frame is ready.
        """
        if len(datas) != self._n:  # check before stepping: a partial step would desync cameras
            raise ValueError(f"got {len(datas)} datas for {self._n} cameras")
        self._fired[:] = False
        out = [s.step(d) for s, d in zip(self.streams, datas, strict=True)]
        ready = np.array([o[1] is not None for o in out], dtype=bool)
        capture_t = np.array([np.nan if o[1] is None else o[1] for o in out], dtype=np.float64)
        seq = np.array([-1 if o[2] is None else o[2] for o in out], dtype=np.int64)
        new = seq != self._prev_seq
        self._prev_seq = seq
        return ready, capture_t, seq, [o[0] for o in out], new, self._fired.copy()
