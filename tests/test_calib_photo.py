"""C3/C4 loop-closure tests for calib_photo — plain script, no pytest."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from synth import synth_chart, synth_dark, synth_flat

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.calib_photo import fit_ptc, fit_response, match_exposure, ptc_points
from physicalai_sim2real_camera.sensor import linearize

REFL = [0.03, 0.09, 0.19, 0.36, 0.59, 0.9]
LEVELS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]


def test_response():
    g, res = fit_response(synth_chart(CameraConfig(64, 64, gamma=2.2), REFL), REFL)
    print(f"C3 black=0: gamma={g:.4f} residual={res:.2e}")
    assert abs(g / 2.2 - 1) < 0.03

    dn = synth_chart(CameraConfig(64, 64, gamma=2.2, black_level=0.02), REFL)
    g_biased, _ = fit_response(dn, REFL)
    g, res = fit_response(dn, REFL, black_level_lin=0.02)
    print(f"C3 black=0.02: gamma={g:.4f} residual={res:.2e} (uncorrected gamma={g_biased:.4f})")
    assert abs(g / 2.2 - 1) < 0.03


def _ptc_data():
    cfg = CameraConfig(64, 64, sigma_shot=0.01, sigma_read=0.002, gamma=2.2, black_level=0.02)
    rng = np.random.default_rng(0)
    stacks = [synth_flat(cfg, rng, lv, 200)[0] for lv in LEVELS]
    dark_mean_lin = float(linearize(synth_dark(cfg, rng, 200)[0], cfg.gamma).mean())
    return stacks, dark_mean_lin


def test_ptc(stacks, dark_mean_lin):
    m, v = ptc_points(stacks, 2.2, dark_mean_lin)
    for a, b in zip(m, v):
        print(f"  m_lin={a:.5f} v_lin={b:.3e}")
    s_shot, s_read, r2 = fit_ptc(stacks, 2.2, dark_mean_lin)
    print(f"C4: slope={s_shot**2:.3e} intercept={s_read**2 + s_shot**2 * dark_mean_lin:.3e}")
    print(f"C4: dark_mean_lin={dark_mean_lin:.5f} sigma_shot={s_shot:.5f} "
          f"sigma_read={s_read:.5f} r2={r2:.5f}")
    assert abs(s_shot / 0.01 - 1) < 0.10
    assert abs(s_read / 0.002 - 1) < 0.10  # without the sigma_shot^2*dark term: +22%
    assert r2 > 0.99


def test_ptc_wrong_gamma(stacks, dark_mean_lin):
    s_shot, s_read, r2 = fit_ptc(stacks, 1.0, dark_mean_lin)
    print(f"C4 wrong gamma=1.0: sigma_shot={s_shot:.5f} sigma_read={s_read:.5f} r2={r2:.5f}")
    assert abs(s_shot / 0.01 - 1) > 0.10


def test_response_black_level_iteration():
    """Lines 33->41: fit_response with black_level_lin > 0 enters iteration loop."""
    # With black_level > 0, the for-loop (lines 33-40) runs until fixed-point
    cfg = CameraConfig(64, 64, gamma=2.2, black_level=0.02)
    dn = synth_chart(cfg, REFL)
    # Without black level correction
    g0, _ = fit_response(dn, REFL, black_level_lin=0.0)
    # With black level correction -> enters iteration, converges to correct gamma
    g1, res1 = fit_response(dn, REFL, black_level_lin=0.02)
    print(f"response black_level_iter: g(no-bl)={g0:.4f} g(bl=0.02)={g1:.4f} res={res1:.2e}")
    assert abs(g1 / 2.2 - 1) < 0.03


def test_ptc_skip_invalid_level():
    """Line 54: ptc_points skips levels where mean DN is outside (5, 250)."""
    # Build a stack with out-of-range mean to trigger the skip
    cfg = CameraConfig(64, 64, gamma=2.2, black_level=0.0)
    rng = np.random.default_rng(5)
    # Very bright stack -> mean > 250, should be skipped
    bright = rng.integers(250, 256, (5, 64, 64), dtype=np.uint8)
    dark = synth_dark(cfg, rng, 10)[0]
    dark_mean_lin = float(linearize(dark, cfg.gamma).mean())
    m_pts, v_pts = ptc_points([bright], 2.2, dark_mean_lin)
    print(f"ptc_skip: m_pts={m_pts}, v_pts={v_pts}")
    assert len(m_pts) == 0  # skipped because mean > 250
    # Very dark stack -> mean < 5, should also be skipped
    very_dark = rng.integers(0, 3, (5, 64, 64), dtype=np.uint8)
    m_pts2, v_pts2 = ptc_points([very_dark], 2.2, dark_mean_lin)
    assert len(m_pts2) == 0


def test_response_black_level_oscillation():
    """Branch 33->41: for loop exhausts 100 iterations without break.

    Uses a black_level_lin that clamps some u**gamma values to 1e-12 on some
    iterations, causing the slope to oscillate and never converge within 100 steps.
    """
    # u1=10/255~0.039, u2=50/255~0.196. black=0.005 causes u1^gamma < black for
    # high gamma (clamped to 1e-12) and u1^gamma > black for low gamma.
    # This makes the slope jump between large positive and small positive values,
    # causing gamma to oscillate between ~0.2 and ~4.1 forever.
    patch_means = [10.0, 50.0]
    reflectances = [0.03, 0.5]
    g, r = fit_response(patch_means, reflectances, black_level_lin=0.005)
    u = np.array(patch_means, np.float64) / 255.0
    x = np.log(np.array(reflectances, np.float64))
    y = np.log(np.maximum(u**g - 0.005, 1e-12)) / g
    slope, _ = np.polyfit(x, y, 1)
    print(f"black_level_osc: gamma={g:.6f} residual={r:.2e}")
    assert abs(1.0 / slope - g) > 1e-6, "expected oscillation, got a fixed point"


def test_match_exposure_invalid():
    """Lines 95-99: match_exposure with non-positive/non-finite inputs returns 0.0 with warning."""
    import warnings as _w
    # non-finite real
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        result = match_exposure(float('nan'), 0.5)
        assert result == 0.0
        assert any("invalid means" in str(x.message) for x in w)
    # zero sim
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        result = match_exposure(0.5, 0.0)
        assert result == 0.0
        assert any("invalid means" in str(x.message) for x in w)
    # negative real
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        result = match_exposure(-0.5, 0.5)
        assert result == 0.0
        assert any("invalid means" in str(x.message) for x in w)


def test_match_exposure_valid():
    """Line 99: match_exposure with valid inputs -> return log2(real/sim)."""
    import math
    result = match_exposure(0.5, 0.25)
    expected = math.log2(0.5 / 0.25)  # = 1.0
    assert abs(result - expected) < 0.001
    # log2(1.0) = 0
    assert match_exposure(0.5, 0.5) == 0.0


if __name__ == "__main__":
    test_response()
    test_response_black_level_iteration()
    test_response_black_level_oscillation()
    test_ptc_skip_invalid_level()
    test_match_exposure_invalid()
    test_match_exposure_valid()
    stacks, dark_mean_lin = _ptc_data()
    test_ptc(stacks, dark_mean_lin)
    test_ptc_wrong_gamma(stacks, dark_mean_lin)
    print("all calib_photo tests passed")
