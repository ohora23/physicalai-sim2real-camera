"""Mains flicker detection from a short exposure ladder, and the frame rates / exposures that avoid it.

Lighting on an AC grid is modulated at twice the mains frequency (100 Hz on 50 Hz grids, 120 Hz on 60 Hz grids). A
camera integrating over ``e`` seconds sees the modulation scaled by ``sinc(hz * e)`` (``lens.flicker_rows``), so the
frame-to-frame brightness of a static scene varies with relative standard deviation

    index(e) = amp * |sinc(hz * e)| / sqrt(2)

when the frame phase is spread over the flicker cycle (sinusoidal lamp). Real lamps are often not sinusoidal
(rectified / PWM LED drivers add harmonics), so the grid is decided by cancellation, which holds for any waveform of
period ``1 / hz``: an exposure of a whole number of periods integrates it out. 8.33 ms cancels 120 Hz but not 100 Hz,
10 ms the other way round. The sinc model is only used to estimate ``amp``. Capture at a frame rate that divides
neither 100 nor 120 (``MEASURE_FPS``) so the phase walks through the cycle within a few seconds; at a divisor
(e.g. 30 fps under 120 Hz) the phase barely moves and the flicker hides.

Typical use at camera setup (any backend: collect the per-frame mean brightness of a static view per exposure)::

    res = detect_mains({0.001: means_1ms, 0.00833: means_8ms, 0.010: means_10ms, ...})
    rec = recommend(res["flicker_hz"], fps_options=(15, 20, 24, 25, 30, 50, 60))
    cfg = dataclasses.replace(cfg, **res["config"])   # flicker_hz / flicker_amp for the simulator
"""

from __future__ import annotations

import math

import numpy as np

MAINS_HZ = (50.0, 60.0)
MEASURE_FPS = 27.0  # divides neither 100 nor 120: aliases 8 / 12 Hz, a full phase sweep in < 0.2 s
DEFAULT_FPS_OPTIONS = (10, 15, 20, 24, 25, 30, 40, 50, 60, 90, 100, 120)


def flicker_index(frame_means) -> float:
    """Relative standard deviation (std / mean) of per-frame mean brightness of a static scene."""
    m = np.asarray(frame_means, np.float64)
    if m.size < 2 or m.mean() <= 0:
        raise ValueError("need >= 2 frame means with a positive mean")
    return float(m.std(ddof=1) / m.mean())


def flicker_index_model(exposure_s, flicker_hz: float, amp: float):
    """``amp * |sinc(flicker_hz * exposure_s)| / sqrt(2)``: expected ``flicker_index`` under random frame phase."""
    return amp * np.abs(np.sinc(flicker_hz * np.asarray(exposure_s, np.float64))) / math.sqrt(2)


def detect_mains(measurements: dict, min_index: float = 0.005, gamma: float = 1.0) -> dict:
    """Fit the 50 Hz and 60 Hz grid models to ``{exposure_s: index or frame means}`` and pick the better one.

    When the ladder holds both grid-cancelling exposures (10 ms and 8.33 ms, or multiples, within 0.2 ms) the grid
    whose exposure leaves the smaller index wins (``method = "cancel"``, waveform independent); otherwise the
    sinusoidal model with the smaller rms wins (``method = "fit"``).

    Args:
        measurements: Exposure (s) -> ``flicker_index`` (float) or the per-frame mean brightness series (array).
            Use >= 3 exposures including one below 2 ms and the two grid-cancelling ones (10 ms, 8.33 ms).
        min_index: Below this largest index the lighting is treated as flicker-free (DC / high-frequency LED).
        gamma: Encoding gamma of the brightness values (1 for linear data). Gamma-encoded 8-bit means shrink small
            relative changes by about ``1 / gamma``, so the indices are multiplied by ``gamma`` before fitting
            ``amp`` (the simulator modulates linear light). The grid decision does not depend on it. A rolling
            shutter averages the frame mean over its readout, which lowers ``amp`` further (not corrected).

    Returns:
        Dict with ``mains_hz`` (50, 60 or None), ``flicker_hz`` (100, 120 or None), ``amp`` (illumination
        modulation depth from the sinusoidal fit; approximate for non-sinusoidal lamps), ``method`` (``"cancel"``,
        ``"fit"`` or ``"none"`` when no flicker is seen), ``ratio``
        (residual index, or rms, of the losing grid over the winning one; > 2 is a clear call), ``confident``,
        ``rms`` per candidate and ``config`` (``CameraConfig`` fields: ``flicker_amp`` and ``flicker_hz``).
    """
    if len(measurements) < 3:
        raise ValueError("need >= 3 exposures")
    e = np.array(sorted(measurements), np.float64)
    idx = gamma * np.array([measurements[k] if np.ndim(measurements[k]) == 0 else flicker_index(measurements[k])
                            for k in sorted(measurements)], np.float64)
    if idx.max() < min_index:
        return dict(mains_hz=None, flicker_hz=None, amp=0.0, method="none", rms={}, ratio=float("inf"),
                    confident=True, config=dict(flicker_amp=0.0))
    fits = {}
    for mains in MAINS_HZ:
        basis = flicker_index_model(e, 2 * mains, 1.0)
        amp = float(np.clip(basis @ idx / max(basis @ basis, 1e-12), 0.0, 1.0))  # least squares through the origin
        fits[mains] = (amp, float(np.sqrt(np.mean((idx - amp * basis) ** 2))))
    on = {}
    for mains in MAINS_HZ:  # exposures within 0.2 ms of a whole number of this grid's flicker periods
        k = e * 2 * mains
        on[mains] = (k >= 0.98) & (np.abs(k - np.round(k)) / (2 * mains) < 2e-4)
    both = on[50.0] & on[60.0]  # common multiples (50 ms, 100 ms) cancel both grids and decide nothing
    cancel = {m: float(idx[on[m] & ~both].min()) for m in MAINS_HZ if (on[m] & ~both).any()}
    if len(cancel) == 2:
        method, score = "cancel", cancel
    else:
        method, score = "fit", {m: f[1] for m, f in fits.items()}
    best, other = sorted(score, key=score.get)
    ratio = score[other] / max(score[best], 1e-12)
    return dict(mains_hz=best, flicker_hz=2 * best, amp=fits[best][0], method=method, ratio=float(ratio),
                confident=bool(ratio > 2.0), rms={m: f[1] for m, f in fits.items()},
                config=dict(flicker_amp=fits[best][0], flicker_hz=2 * best))


def beat_hz(flicker_hz: float, fps: float) -> float:
    """Apparent (aliased) flicker frequency at ``fps``: distance from ``flicker_hz`` to the nearest multiple of fps."""
    r = flicker_hz % fps
    return float(min(r, fps - r))


def recommend(flicker_hz, fps_options=DEFAULT_FPS_OPTIONS, max_exposure_s: float | None = None) -> dict:
    """Frame rates and exposures that avoid visible flicker under ``flicker_hz`` lighting.

    - Exposures that are whole multiples of ``1 / flicker_hz`` cancel the flicker at any frame rate (global shutter)
      and also remove rolling-shutter banding.
    - When a scene is too bright for that (exposure < ``1 / flicker_hz``), a frame rate dividing ``flicker_hz``
      (beat 0 Hz) keeps the flicker phase fixed from frame to frame: a constant brightness offset instead of
      pulsing. Real camera clocks drift slightly, so the phase still walks slowly (seconds to minutes).

    Args:
        flicker_hz: 100, 120, or None (no flicker detected: every option is fine).
        fps_options: Frame rates the camera supports.
        max_exposure_s: Longest exposure the application tolerates (motion blur); default ``1 / min(fps)``.

    Returns:
        Dict with ``safe_exposures_s`` (multiples of the flicker period up to the limit), ``no_beat_fps`` (beat 0),
        ``beat_hz`` per option, ``min_safe_exposure_s`` and ``fps_for_safe_exposure`` (options whose frame period is
        at least one flicker period, so a flicker-cancelling exposure fits in a frame).
    """
    fps_options = sorted(float(f) for f in fps_options)
    if not fps_options or fps_options[0] <= 0:
        raise ValueError(f"fps_options must be non-empty and positive, got {fps_options}")
    if flicker_hz is None:
        return dict(safe_exposures_s=[], no_beat_fps=fps_options, beat_hz={f: 0.0 for f in fps_options},
                    min_safe_exposure_s=0.0, fps_for_safe_exposure=fps_options)
    period = 1.0 / flicker_hz
    limit = max_exposure_s if max_exposure_s is not None else 1.0 / fps_options[0]
    n = int(math.floor(limit / period + 1e-9))
    beats = {f: beat_hz(flicker_hz, f) for f in fps_options}
    return dict(safe_exposures_s=[k * period for k in range(1, n + 1)],
                no_beat_fps=[f for f, b in beats.items() if b < 1e-9], beat_hz=beats, min_safe_exposure_s=period,
                fps_for_safe_exposure=[f for f in fps_options if 1.0 / f >= period - 1e-12])
