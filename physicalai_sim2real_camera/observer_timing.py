"""Online timing observer: fps, jitter, drops and latency from frame timestamps and head motion.

Two independent channels, on purpose:

- **Stream statistics** use only ``t_arrival - ts_sensor``: the sensor clock is ``ts_sensor = (1 + rho) * t_capture +
  c0 + noise`` with an unknown offset ``c0`` and drift ``rho``, so a line fit of that difference against ``ts_sensor``
  removes both, and the residual's variance is the delivery jitter. With the Gamma(k = 2, theta) jitter model of
  ``CameraConfig`` (``calib_timing.fit_timing`` fixes k = 2 too), ``theta = sqrt(var / 2)``. The period comes from
  ``ts_sensor`` differences per captured frame. Camera drops are frames that were never published (the camera's
  ``frame_number`` advances, the transport ``seq`` does not); observer misses are published frames the observer
  skipped (both counters advance), so per gap ``misses = seq gap - 1`` and ``drops = frame_number gap - seq gap``;
  they are reported separately so a slow observer does not look like a dropping camera.
- **Mean latency** cannot come from timestamps (the offset ``c0`` hides it). It comes from motion: the camera's
  angular speed ``|omega_cam|`` from the poses of one prism face in the frames before and after (``omega_central``;
  no hand-eye rotation or prism pose needed: the norm is invariant) is matched against the robot's reported head
  speed ``|omega_head|(t)`` by a 1-D
  least-squares fit over the delay ``delta``:
  ``delta = argmin_delta min_{s,b} sum_i (|omega_cam_i| - s |omega_head|(a_i - delta + rs_i) - b)^2`` (the scale
  and offset absorb a K error in the solvePnP speeds and the noise floor at rest) on the frame times
  ``a_i = alpha ts_i + beta``, the line of the arrival times against the sensor stamps (capture + L + mean jitter
  on the robot clock without the per-frame jitter, which met the step of |omega| at every stop and biased the fit;
  no uniform grid and no cross-correlation peak), where
  ``rs_i = (y_i / H - 0.5) * readout_s`` moves a rolling-shutter face to the time its rows were read. Reported head
  state lags the true pose by the proprioception delay ``p`` (``omega_head_reported(t) = omega_true(t - p)``), so
  the fit returns ``delta = L + mean_jitter - p`` and ``latency_s = delta - 2 theta + p``: the ``CameraConfig`` meaning
  (mid-exposure -> frame available, jitter mean excluded), relative to the robot-state clock. ``omega_cam`` must
  belong to its frame's capture time: a backward difference of poses i-1 -> i belongs to their midpoint and would
  bias ``delta`` by half a frame period (16.7 ms at 30 fps), hence the central difference over i-1 -> i+1.
"""

from __future__ import annotations

import numpy as np

JITTER_K = 2            # Gamma shape fixed, as in calib_timing.fit_timing
TS_NOISE_S = 50e-6      # sensor timestamp quantisation / noise removed from the jitter variance
DELTA_MAX_S = 0.25      # latency search range
DELTA_STEP_S = 1e-4     # grid step before the parabolic refinement
_U = np.linspace(-1.0, 1.0, 7)  # window-average nodes and trapezoid weights (fit_delay half_window_s)
_U_W = np.r_[0.5, np.ones(5), 0.5] / 6.0
WINDOW_S = 30.0         # latency fit over the last 30 s of samples (HeadPoseRing: 32 s)
HEAD_PERIOD_S = 0.002   # reported head state at 500 Hz (the control loop)
FLIP_RAD_S = 2.0        # max |omega| difference between the two half-steps of a triplet (a pose flip is >> this)


def omega_central(R_prev, R_next, ts_prev, ts_next):
    """|omega| (rad/s) at a frame from the face rotations of its neighbours.

    A central difference: it belongs to the midpoint of the two neighbours, which is the frame itself when they are
    evenly spaced; after a drop or miss on one side it is up to half a period off (symmetric drops mostly cancel:
    the O4 test with 5 % drops and 1 % misses stays within 1.3 ms).

    Args:
        R_prev: ``(3, 3)`` camera-from-face rotation in the previous received frame (``cv2.Rodrigues`` of solvePnP).
        R_next: The same in the next received frame.
        ts_prev: Sensor timestamp of the previous frame, s (no delivery jitter, unlike the arrival time).
        ts_next: Sensor timestamp of the next frame, s.

    Returns:
        Rotation angle between the two poses divided by their time difference; ``nan`` when ``ts_next <= ts_prev``.
    """
    dt = float(ts_next) - float(ts_prev)
    if dt <= 0:
        return float("nan")
    c = (np.trace(np.asarray(R_next) @ np.asarray(R_prev).T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)) / dt)


def omega_triplet(R0, R1, R2, ts0, ts1, ts2, max_dw=FLIP_RAD_S):
    """``omega_central`` at the middle frame, or ``nan`` when the triplet holds a pose flip.

    Planar PnP (IPPE) on a small or tilted face sometimes returns the mirrored pose: the angle to that frame jumps by
    tens of degrees and one such sample pulled the latency fit 11 ms off (O6, prism turned towards the wedge face).
    The head's angular speed changes by at most ~1 rad/s between neighbouring frames (a stop), a flip by far more, so
    the triplet is refused when the speeds over its two half-steps differ by more than ``max_dw`` (a flip at either
    end), or when they do not add up to the speed over the whole step (``w01 + w12 = 2 w02`` for a rotation about a
    steady axis at an even frame spacing; a flipped middle frame makes both half-steps fast and the whole step slow).

    Args:
        R0: Face rotation in the previous received frame.
        R1: The same in the middle frame.
        R2: The same in the next received frame.
        ts0: Sensor timestamp of the previous frame, s.
        ts1: Of the middle frame.
        ts2: Of the next frame.
        max_dw: Largest accepted half-step speed difference, rad/s.

    Returns:
        |omega| at the middle frame (rad/s) or ``nan``.
    """
    w01, w12, w02 = omega_central(R0, R1, ts0, ts1), omega_central(R1, R2, ts1, ts2), omega_central(R0, R2, ts0, ts2)
    if not max(abs(w01 - w12), abs(w01 + w12 - 2 * w02)) <= max_dw:  # also catches nan
        return float("nan")
    return w02


def detrended_delay(t_arrival, ts_sensor):
    """Residual of ``t_arrival - ts_sensor`` after removing offset and linear drift against ``ts_sensor``."""
    d = np.asarray(t_arrival, float) - np.asarray(ts_sensor, float)
    if len(d) < 3:
        return d - d.mean() if len(d) else d
    a, b = np.polyfit(np.asarray(ts_sensor, float), d, 1)
    return d - (a * np.asarray(ts_sensor, float) + b)


def jitter_theta(residual):
    """Gamma(k = 2) scale from the residual variance: ``theta = sqrt(max(var - noise^2, 0) / 2)``."""
    var = float(np.var(residual)) - TS_NOISE_S**2
    return float(np.sqrt(max(var, 0.0) / JITTER_K))


def unwrap_counter(values, bits):
    """Undo the wraps of a ``bits``-bit counter (frame number, transport seq) in arrival order.

    Jitter may deliver a frame after its successor; ``np.unwrap`` takes the step closest to zero modulo
    ``2**bits``, so reordering by less than half the range is kept as reordering, not taken for a wrap. A gap or
    reorder of ``2**(bits-1)`` or more is read as a wrap (128 lost frames on an 8-bit counter); values go through
    float64, exact up to ``bits`` = 52.

    Args:
        values: Raw counter values in arrival order.
        bits: Counter width.

    Returns:
        ``int64`` array, monotonic apart from the reordering.
    """
    if bits > 52:
        raise ValueError(f"unwrap_counter: {bits}-bit counters are not exact in float64; they do not wrap anyway")
    v = np.asarray(values, np.float64)
    return np.round(np.unwrap(v, period=float(2**bits))).astype(np.int64)


def drop_accounting(frame_number, seq):
    """Camera drops (``frame_number`` gaps) and observer misses (``seq`` gaps with continuous ``frame_number``).

    Returns:
        ``(camera_drop_rate, observer_miss_rate, n_camera_drops, n_observer_misses)``; rates are over the span of the
        respective counters.
    """
    fn = np.asarray(frame_number, np.int64)
    sq = np.asarray(seq, np.int64)
    if len(fn) < 2:
        return 0.0, 0.0, 0, 0
    # every published frame carries both counters; a dropped frame was never published (frame_number advances,
    # seq does not), a frame the observer missed was published (both advance). Counted over the ranges, not per gap:
    # a frame delivered after its successor (jitter) made the per-gap sums count a drop that never happened (O6)
    n = len(np.unique(fn))
    # counters must be monotonic (no camera restart); a recording's wraps are undone by unwrap_counter (O7)
    misses = max(int(sq.max() - sq.min() + 1 - n), 0)
    cam_drops = max(int(fn.max() - fn.min() + 1 - n) - misses, 0)
    span_fn = int(fn.max() - fn.min()) or 1
    span_sq = int(sq.max() - sq.min()) or 1
    return cam_drops / span_fn, misses / span_sq, cam_drops, misses


def _parabolic_min(grid, c):
    """Grid argmin of ``c`` refined by the parabola through it and its neighbours (uniform ``grid``)."""
    j = int(np.nanargmin(c))
    if 0 < j < len(grid) - 1:
        denom = c[j - 1] - 2 * c[j] + c[j + 1]
        if denom > 0:  # also False for nan neighbours
            return float(grid[j] + 0.5 * (grid[1] - grid[0]) * (c[j - 1] - c[j + 1]) / denom)
    return float(grid[j])


def fit_delay(omega_cam, t_arrival, omega_head_fn, row_shift_s=None, delta_max=DELTA_MAX_S, step=DELTA_STEP_S,
              n_boot=20, rng=None, affine=True, half_window_s=None):
    """1-D least squares of the camera-vs-head speed lag ``delta`` with a parabolic refinement and a bootstrap std.

    Args:
        omega_cam: ``(n,)`` camera angular speed per frame (rad/s), referenced to that frame (``omega_central``).
        t_arrival: ``(n,)`` arrival time of each frame on the observer clock (robot-state clock), s.
        omega_head_fn: Callable ``t -> |omega_head|`` (rad/s) on the robot-state clock (interpolated head trajectory).
        row_shift_s: Optional ``(n,)`` rolling-shutter row-time shift per frame ``(y/H - 0.5) * readout_s``; 0 for
            global shutter.
        delta_max: Search range ``[0, delta_max]`` s.
        step: Grid step, s.
        n_boot: Block-bootstrap resamples for the std (0 disables; then ``std`` is from the cost curvature).
        rng: ``numpy.random.Generator`` for the bootstrap.
        affine: Match ``|omega_cam| = s |omega_head| + b`` with the best ``s, b`` at every lag (default) instead of
            ``s = 1, b = 0``. The camera speed comes from solvePnP with the observer's current K: at the start
            (nominal K x 1.05, no distortion) it read 5-20 % low, which moved the lag by up to 17 ms (O6).
        half_window_s: Optional ``(n,)`` half-width of the interval each ``omega_cam`` sample averages over (half
            the span of a central difference); the head speed is then averaged over the same interval instead of
            taken at its centre. At the step of |omega| at a stop, a window average against a point value biased
            the lag by 3 ms once detection kept more samples on the still side (O6, rolling shutter).

    Returns:
        ``(delta, std, cost_min)`` in seconds (std also in seconds); all ``nan`` when no sample can be compared
        (``omega_head_fn`` is ``nan`` outside its record).
    """
    w = np.asarray(omega_cam, float)
    a = np.asarray(t_arrival, float)
    rs = np.zeros_like(a) if row_shift_s is None else np.asarray(row_shift_s, float)
    hw = None if half_window_s is None else np.asarray(half_window_s, float)

    def costs(deltas, idx=slice(None)):
        """Cost at every lag in ``deltas`` at once: ``(len(deltas),)``, nan where fewer than 3 samples compare."""
        wv = w[idx][None, :]
        tt = a[idx][None, :] - np.asarray(deltas)[:, None] + rs[idx][None, :]
        if hw is None:
            h = np.asarray(omega_head_fn(tt), float)
        else:  # trapezoid mean over the sample's interval
            h = np.asarray(omega_head_fn(tt[..., None] + hw[idx][None, :, None] * _U), float) @ _U_W
        ok = np.isfinite(h)  # nan = outside the recorded head trajectory
        n = ok.sum(axis=1)
        h0 = np.where(ok, h, 0.0)
        w0 = np.where(ok, wv, 0.0)
        with np.errstate(all="ignore"):
            if not affine:
                c = (np.where(ok, (wv - h0) ** 2, 0.0)).sum(axis=1) / n
            else:  # residual variance of the best line w = s h + b: a scale error in |omega_cam| (solvePnP with a K
                #    that is still off) or a noise floor at rest must not move the lag
                mw, mh = w0.sum(axis=1) / n, h0.sum(axis=1) / n
                dw, dh = np.where(ok, wv - mw[:, None], 0.0), np.where(ok, h0 - mh[:, None], 0.0)
                var_w, var_h, cov = (dw * dw).sum(1) / n, (dh * dh).sum(1) / n, (dw * dh).sum(1) / n
                #    the slope must be positive: an anti-correlated lag (periodic motion) would otherwise fit as well
                c = var_w - np.where((var_h > 0) & (cov > 0), cov * cov / np.where(var_h > 0, var_h, 1.0), 0.0)
        return np.where(n >= (1 if not affine else 3), c, np.nan)

    grid = np.arange(0.0, delta_max + step, step)
    cost_grid = costs(grid)
    if np.all(np.isnan(cost_grid)):  # no camera sample falls inside the recorded head trajectory at any lag
        return float("nan"), float("nan"), float("nan")
    j = int(np.nanargmin(cost_grid))
    delta = _parabolic_min(grid, cost_grid)
    n_eff = int(np.isfinite(omega_head_fn(a - grid[j] + rs)).sum())  # samples that entered the cost
    if n_boot and n_eff >= 20:
        rng = np.random.default_rng(0) if rng is None else rng
        blk = max(5, len(w) // 20)
        starts = np.arange(0, len(w) - blk + 1)
        est = []
        # the whole search range on a 5x coarser grid: a window around the full-data minimum capped the std at a few
        # ms when the cost was flat (O6: std 1.1 ms for an error of 11 ms); the refinement is not needed for a spread
        coarse = grid[::5]
        for _ in range(n_boot):
            idx = np.concatenate([np.arange(s, s + blk) for s in rng.choice(starts, len(w) // blk, replace=True)])
            cb = costs(coarse, idx)
            if not np.all(np.isnan(cb)):
                est.append(_parabolic_min(coarse, cb))
        std = float(np.std(est)) if est else float("nan")
    else:  # curvature of the cost around the minimum
        c = cost_grid
        d2 = (c[min(j + 1, len(grid) - 1)] - 2 * c[j] + c[max(j - 1, 0)]) / step**2
        std = float(np.sqrt(2 * c[j] / max(d2, 1e-12) / max(n_eff, 1)))
    return float(delta), std, float(cost_grid[j])


class TimingObserver:
    """Streaming timing estimator; feed one record per received frame, call ``estimate()`` at any time."""

    def __init__(self, readout_s=0.0, height_px=480, proprio_delay_s=0.0, head_period_s=HEAD_PERIOD_S,
                 window_s=WINDOW_S):
        """Set the rolling-shutter and proprioception constants (both offline inputs).

        Args:
            readout_s: Rolling-shutter readout time (0 for global shutter), from offline D5b.
            height_px: Image height, for the row-time shift.
            proprio_delay_s: Known delay of the reported robot state behind the true pose.
            head_period_s: Sample period of the reported head state; ``latency_std`` has a floor of
                ``head_period_s / sqrt(12)`` (the interpolated |omega| places a stop anywhere inside one period).
            window_s: The latency fit uses the |omega| samples of the last ``window_s`` (the head ring holds 32 s;
                older samples compare with nothing and made the fit's cost grow with the run length).
        """
        self.readout_s, self.H, self.p, self.head_period_s = readout_s, height_px, proprio_delay_s, head_period_s
        self.window_s = window_s
        self.t_arrival, self.ts_sensor, self.seq, self.frame_number = [], [], [], []
        self.omega_cam, self.omega_ts, self.omega_hw, self.row_shift = [], [], [], []

    def push(self, t_arrival, ts_sensor, seq, frame_number, omega_cam=None, row_mean_px=None, omega_span=None):
        """Record one received frame.

        Args:
            t_arrival: Receipt time on the observer (robot-state) clock, s.
            ts_sensor: The camera's own timestamp for this frame, s (any offset / drift).
            seq: Transport sequence number (Axon seq); gaps here are observer misses.
            frame_number: The camera's capture counter; gaps here are camera drops.
            omega_cam: Camera angular speed at this frame (rad/s), or ``None``; it must belong to this frame's
                capture time, e.g. ``omega_central`` over the previous and next received frames (pass it when the
                next frame arrives, with this frame's other arguments).
            row_mean_px: Mean image row of the tracked face (for the rolling-shutter time shift); ``None`` = centre.
            omega_span: ``(ts_prev, ts_next)`` sensor stamps (row-time corrected) of the two poses ``omega_cam`` was
                differenced over. The sample then belongs to the span's midpoint and is matched against the head
                speed averaged over the span; ``None`` = a point sample at ``ts_sensor``. With a span,
                ``row_mean_px`` is ignored.
        """
        self.t_arrival.append(float(t_arrival))
        self.ts_sensor.append(float(ts_sensor))
        self.seq.append(int(seq))
        self.frame_number.append(int(frame_number))
        if omega_cam is not None and np.isfinite(omega_cam):
            self.omega_cam.append(float(omega_cam))
            t0, t1 = (ts_sensor, ts_sensor) if omega_span is None else omega_span
            self.omega_ts.append(0.5 * (t0 + t1))
            self.omega_hw.append(0.5 * (t1 - t0))
            # a span is already row-time corrected: adding the row shift as well would apply it twice
            y = self.H / 2 if row_mean_px is None or omega_span is not None else float(row_mean_px)
            self.row_shift.append((y / self.H - 0.5) * self.readout_s)

    def estimate(self, omega_head_fn=None, rng=None):
        """Current estimates.

        Args:
            omega_head_fn: ``t -> |omega_head|`` on the robot-state clock; when ``None`` no latency is estimated.
            rng: Bootstrap generator.

        Returns:
            dict with ``fps``, ``period_s``, ``jitter_k`` (2), ``jitter_theta`` and ``jitter_theta_std``,
            ``camera_drop_rate`` and ``camera_drop_std``, ``observer_miss_rate``, ``n_frames``, and when motion data
            exist: ``delta`` (raw lag),
            ``latency_s`` (= delta − 2θ + p), ``latency_std`` and ``clock_line`` ``(alpha, beta)``: robot-clock
            frame time ``alpha * ts_sensor + beta`` (capture + L + mean jitter, no per-frame jitter).
        """
        n = len(self.t_arrival)
        out = dict(n_frames=n, jitter_k=JITTER_K)
        if n < 3:
            return out
        ts, fn = np.asarray(self.ts_sensor), np.asarray(self.frame_number)
        dfn = np.diff(fn)
        ok = dfn > 0
        if not ok.any():  # frame numbers never advanced: no period to measure
            return out
        period = float(np.median(np.diff(ts)[ok] / dfn[ok]))
        resid = detrended_delay(self.t_arrival, self.ts_sensor)
        theta = jitter_theta(resid)
        cam, miss, ncd, nom = drop_accounting(fn, self.seq)
        # theta^2 = (var - noise^2) / k: the sample variance of Gamma(k) jitter has std sqrt((2 + 6 / k) / n) of the
        # variance (excess kurtosis 6 / k), so std(theta) = that * var / (2 k theta) (inf at theta 0: not
        # identifiable). Drops: binomial over the frame-number span, rate floored at one drop per span; bursty
        # real drops make it read low (the O8 gate then holds the drop rate back more often)
        span = max(int(fn.max() - fn.min()), 1)
        var = float(np.var(resid))
        theta_std = np.sqrt((2 + 6 / JITTER_K) / n) * var / (2 * JITTER_K * theta) if theta > 0 else np.inf
        out.update(period_s=period, fps=1.0 / period, jitter_theta=theta, camera_drop_rate=cam,
                   observer_miss_rate=miss, n_camera_drops=ncd, n_observer_misses=nom,
                   jitter_theta_std=float(theta_std),
                   camera_drop_std=float(np.sqrt(max(cam, 1.0 / span) * (1 - cam) / span)))
        if omega_head_fn is not None and len(self.omega_cam) >= 20:
            # time axis: the line of t_arrival against ts_sensor, i.e. capture + L + mean jitter on the robot clock
            # without the per-frame jitter. On the arrival times themselves the jitter (std 7 ms) met the step of
            # |omega| at every stop and biased the fit +13 ms even with the true |omega|
            a, b = np.polyfit(ts, np.asarray(self.t_arrival), 1)
            out.update(clock_line=(float(a), float(b)))
            t_frame = a * np.asarray(self.omega_ts) + b
            m = t_frame >= t_frame[-1] - self.window_s
            hw = a * np.asarray(self.omega_hw)[m]
            delta, std, cmin = fit_delay(np.asarray(self.omega_cam)[m], t_frame[m], omega_head_fn,
                                         np.asarray(self.row_shift)[m], rng=rng, half_window_s=hw if hw.any() else None)
            std = float(np.hypot(std, self.head_period_s / np.sqrt(12)))  # bootstrap misses the shared sampling error
            out.update(delta=delta, latency_s=delta - JITTER_K * theta + self.p, latency_std=std, fit_cost=cmin)
        return out
