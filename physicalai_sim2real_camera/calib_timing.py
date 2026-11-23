"""Timing calibration: fps, latency, jitter, and drop rate from event/availability pairs."""

from __future__ import annotations

import numpy as np

MIN_SAMPLES = 50
GRID_TOL = 1.05  # max event-residual spread, in frame periods, for a regular capture grid


def fit_timing(event_t, avail_t, frame_seq, event_seen, fps_hint=None, quantized=True):
    """Fit fps, latency, jitter, and drop probability from timed events (C5).

    Latency reference: mid-exposure -> frame available to the policy. Events and
    availability are on the same host clock (no offset estimation).

    Inputs: ``event_t`` (M,) sorted event times (LED toggles; D5b: exposure
    times read per frame); per surviving frame i: ``avail_t[i]`` policy-available
    time, ``frame_seq[i]`` capture index (increasing; gaps = drops),
    ``event_seen[i]`` number of events at or before its capture (non-decreasing).

    - period P = slope of a line fit of avail_t on frame_seq; fps = ``fps_hint`` or 1/P.
    - Sample for event j: first frame i with ``event_seen[i] >= j+1``,
      ``s_j = avail_t[i] - event_t[j]``. Excluded when i is the first frame or
      index ``frame_seq[i]-1`` is missing (the true first frame may have been
      dropped -> sample inflated by a period), and events after the last frame.
    - ``quantized=True`` (D5a): the capture lands U(0, 1/fps) after the event.
      Captures sit on a regular grid c + n*P (P from a line fit of avail_t on
      frame_seq), and each sample brackets c: event_t - n*P in (c - P, c]. If
      that spread is <= GRID_TOL periods, c is its midpoint and the per-sample
      capture offset is removed exactly (halves the theta scatter vs. moments).
      Otherwise (irregular grid) fall back to moments with a warning:
      mean_c = mean - 1/(2 fps), var_c = var - 1/(12 fps^2).
      ``quantized=False``: no correction.
    - jitter_k fixed at 2 (moment fit of k is unstable at ~300 samples);
      jitter_theta = sqrt(max(var_c, 0)/2); latency_s = mean_c - 2 theta.
      With < 50 samples: theta = 0, latency_s = mean_c.
    - drop_prob = missing indices / span of indices.

    Returns dict(fps, latency_s, jitter_k, jitter_theta, drop_prob, n_samples,
    samples (raw s_j), period, grid_offset (capture_t = grid_offset + frame_seq*period;
    None unless the regular-grid correction was used), warnings).
    """
    event_t = np.asarray(event_t, float)
    avail_t = np.asarray(avail_t, float)
    seq = np.asarray(frame_seq, np.int64)
    seen = np.asarray(event_seen, np.int64)
    warnings = []

    consec = np.diff(seq) == 1
    period = float(np.polyfit(seq, avail_t, 1)[0])
    fps = float(fps_hint) if fps_hint else 1.0 / period

    i = np.searchsorted(seen, np.arange(1, len(event_t) + 1), side="left")
    ok = (i >= 1) & (i < len(seq))
    ok[ok] = consec[i[ok] - 1]
    i = i[ok]
    samples = avail_t[i] - event_t[ok]
    n = int(samples.size)

    corr, moments, c = samples, False, None
    if quantized and n >= 2:
        r = event_t[ok] - seq[i] * period  # each in (c - period, c]
        spread = float(r.max() - r.min()) / period
        if spread <= GRID_TOL:
            c = float(r.max() + r.min() + period) / 2
            corr = samples - (c - r)  # avail - capture
        else:
            warnings.append(f"capture grid irregular (event spread {spread:.2f} periods): "
                            "moment quantisation correction used")
            moments = True
    mean = float(corr.mean()) if n else float("nan")
    var = float(corr.var(ddof=1)) if n > 1 else 0.0
    if moments:
        mean -= 0.5 / fps
        var -= 1.0 / (12.0 * fps * fps)
    if n < MIN_SAMPLES:
        warnings.append(f"only {n} samples (< {MIN_SAMPLES}): jitter set to 0")
        theta = 0.0
    else:
        if var <= 0:
            warnings.append("corrected variance <= 0: quantisation dominates, jitter ~ 0")
        theta = float(np.sqrt(max(var, 0.0) / 2.0))
    latency = mean - 2.0 * theta
    if latency < 0:
        warnings.append(f"negative latency_s {latency:.4g}: check clocks/event detection")

    span = int(seq[-1] - seq[0] + 1)
    return dict(fps=fps, latency_s=latency, jitter_k=2.0, jitter_theta=theta,
                drop_prob=(span - len(seq)) / span, n_samples=n, samples=samples,
                period=period, grid_offset=c, warnings=warnings)
