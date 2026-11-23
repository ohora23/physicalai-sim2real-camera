"""Mains flicker detection and frame-rate / exposure recommendation (physicalai_sim2real_camera.flicker)."""

import math

import numpy as np
import pytest

from physicalai_sim2real_camera.flicker import (
    MEASURE_FPS,
    beat_hz,
    detect_mains,
    flicker_index,
    flicker_index_model,
    recommend,
)

EXPOSURES = (0.001, 0.002, 0.003, 0.005, 1 / 120, 0.010)


def ladder(flicker_hz, amp, fps=MEASURE_FPS, n=90, noise=0.002, seed=0, wave="sine", exposures=EXPOSURES):
    """Per-frame mean brightness of a static scene: lamp L(t) integrated over each exposure (1000 samples).

    ``wave="sine"`` is the simulator's 1 + amp cos(2 pi hz t); ``"rectified"`` a full-wave rectified mains sine
    (fundamental ``flicker_hz``, strong harmonics) scaled to the same depth.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for e in exposures:
        t0 = np.arange(n) / fps + rng.uniform(0, 1)
        t = t0[:, None] + np.linspace(0, e, 1000)[None]
        if wave == "sine":
            lamp = 1 + amp * np.cos(2 * math.pi * flicker_hz * t)
        else:
            r = np.abs(np.sin(math.pi * flicker_hz * t))  # period 1 / flicker_hz, mean 2 / pi
            lamp = 1 + amp * (r - 2 / math.pi) / (1 - 2 / math.pi)
        out[e] = 100 * lamp.mean(1) * (1 + noise * rng.standard_normal(n))
    return out


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_detect_mains(mains):
    """Both grids are told apart from a 90-frame ladder at 27 fps; the fitted depth recovers amp within 15 %."""
    res = detect_mains(ladder(2 * mains, 0.2))
    assert res["mains_hz"] == mains and res["flicker_hz"] == 2 * mains and res["confident"] and res["ratio"] > 2
    assert res["method"] == "cancel"
    assert abs(res["amp"] - 0.2) < 0.03 and res["config"] == dict(flicker_amp=res["amp"], flicker_hz=2 * mains)
    # a rectified (harmonic-rich) lamp: cancellation still decides the grid
    assert detect_mains(ladder(2 * mains, 0.4, wave="rectified"))["mains_hz"] == mains
    # without the grid-cancelling pair the sinusoidal fit decides
    fit = detect_mains(ladder(2 * mains, 0.2, exposures=(0.001, 0.002, 0.003, 0.004, 0.006)))
    assert fit["method"] == "fit" and fit["mains_hz"] == mains


def test_no_flicker_and_inputs():
    """Flat lighting -> no grid; indices or raw series are both accepted; too few exposures / frames refused."""
    res = detect_mains(ladder(120.0, 0.0))
    assert res["mains_hz"] is None and res["config"] == dict(flicker_amp=0.0)
    idx = {e: flicker_index(m) for e, m in ladder(120.0, 0.3).items()}
    assert detect_mains(idx)["mains_hz"] == 60.0
    assert abs(flicker_index([1.0, 3.0]) - math.sqrt(2) / 2) < 1e-12
    with pytest.raises(ValueError):
        detect_mains({0.001: 0.1, 0.01: 0.1})
    with pytest.raises(ValueError):
        flicker_index([5.0])


def test_index_model():
    """Exposures of one flicker period cancel it; a very short exposure keeps amp / sqrt(2)."""
    assert flicker_index_model(1 / 120, 120.0, 0.5) < 1e-12
    assert abs(flicker_index_model(1e-6, 100.0, 0.5) - 0.5 / math.sqrt(2)) < 1e-6


def test_recommend():
    """120 Hz: exposures k/120 s, beat-free rates divide 120 (25 fps beats at 5 Hz); no flicker allows everything."""
    rec = recommend(120.0, fps_options=(15, 24, 25, 30, 50, 60), max_exposure_s=0.034)
    assert np.allclose(rec["safe_exposures_s"], [1 / 120, 2 / 120, 3 / 120, 4 / 120])
    assert rec["no_beat_fps"] == [15.0, 24.0, 30.0, 60.0] and rec["beat_hz"][25.0] == 5.0
    assert rec["min_safe_exposure_s"] == 1 / 120 and rec["fps_for_safe_exposure"] == [15.0, 24.0, 25.0, 30.0, 50.0,
                                                                                         60.0]
    assert recommend(100.0, fps_options=(30, 150))["fps_for_safe_exposure"] == [30.0]
    assert beat_hz(100.0, 30.0) == 10.0 and beat_hz(100.0, 27.0) == 8.0 and beat_hz(120.0, 27.0) == 12.0
    free = recommend(None, fps_options=(30, 25))
    assert free["no_beat_fps"] == [25.0, 30.0] and free["safe_exposures_s"] == []


def test_review_edges():
    """Common multiples of both grids (50 ms) are ignored; gamma scales amp only; bad fps options are refused."""
    lad = ladder(120.0, 0.2, exposures=EXPOSURES + (0.050,))
    assert detect_mains(lad)["mains_hz"] == 60.0 and detect_mains(lad)["confident"]
    only_common = {e: v for e, v in ladder(100.0, 0.2, exposures=(0.001, 0.002, 0.050)).items()}
    assert detect_mains(only_common)["method"] == "fit"
    g = detect_mains(ladder(100.0, 0.1), gamma=2.2)
    assert g["mains_hz"] == 50.0 and abs(g["amp"] - 0.22) < 0.03
    assert detect_mains(ladder(100.0, 0.0))["method"] == "none"
    for bad in ((), (0, 30)):
        with pytest.raises(ValueError):
            recommend(120.0, fps_options=bad)
