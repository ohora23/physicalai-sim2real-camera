"""Tests for physicalai_sim2real_camera/lens.py (camera-side DR pure functions)."""

import numpy as np
import pytest

from physicalai_sim2real_camera.lens import dirt_maps, flicker_rows, jpeg_roundtrip, veil

# ---------------------------------------------------------------------------
# flicker_rows
# ---------------------------------------------------------------------------

def test_flicker_rows_global():
    H = 8
    t_c = 0.0123
    readout_s = 0.0
    exposure_s = 0.0
    amp = 0.3
    hz = 100.0
    phase = 0.4

    g = flicker_rows(H, t_c, exposure_s, readout_s, amp, hz, phase)
    expected = 1.0 + amp * np.cos(2.0 * np.pi * hz * t_c + phase)
    np.testing.assert_array_equal(g, np.full(H, expected, dtype=np.float32),
                                err_msg="flicker_rows global value mismatch")


def test_flicker_rows_integrates_out():
    amp = 0.5
    hz = 100.0
    readout_s = 0.01

    # one full period exposure -> sinc(1) = 0 -> all rows == 1
    g = flicker_rows(480, 0.0, 1.0 / 100, readout_s, amp, hz, 0.0)
    np.testing.assert_array_equal(g, np.ones(480, dtype=np.float32),
                                err_msg="flicker_rows should integrate to 1 over full period")

    # half-period exposure -> sinc(0.5) = 2/pi
    e_half = 1.0 / 200
    g = flicker_rows(480, 0.0, e_half, readout_s, amp, hz, 0.0)
    expected_amp = amp * 2.0 / np.pi
    assert np.max(g) - 1 <= expected_amp + 1e-6, "sinc(0.5) amplitude too high"

    # With readout, a full period is sampled: the swing is exactly amp * sinc(0.5) = amp * 2/pi
    g2 = flicker_rows(400, 0.0, e_half, 1.0 / 100, amp, hz, 0.0)
    assert abs(np.ptp(g2) / 2 - expected_amp) < 1e-3
    # one row against the formula written out: row 100 of 400, t_y = 0.01 * (100.5/400 - 0.5)
    t_y = 0.01 * (100.5 / 400 - 0.5)
    assert abs(g2[100] - (1 + amp * (2 / np.pi) * np.cos(2 * np.pi * hz * t_y))) < 1e-6


def test_flicker_rows_rolling_period():
    H = 480
    readout_s = 0.03
    exposure_s = 0.0
    hz = 100.0
    amp = 0.2
    phase = 0.1  # offset so max doesn't sit exactly on an edge

    g = flicker_rows(H, 0.0, exposure_s, readout_s, amp, hz, phase)
    diffs = np.diff(g)
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
    maxima = sign_changes[np.array(diffs[sign_changes]) < 0]
    assert len(maxima) == 3, f"Expected 3 maxima, got {len(maxima)}"


# ---------------------------------------------------------------------------
# dirt_maps
# ---------------------------------------------------------------------------

def test_dirt_maps_off():
    H, W = 64, 80
    T, D = dirt_maps(H, W, 0.0, np.random.default_rng(0))
    assert T.shape == (H, W, 1)
    assert D.shape == (H, W, 1)
    assert T.dtype == np.float32
    assert D.dtype == np.float32
    np.testing.assert_array_equal(T, np.ones((H, W, 1), dtype=np.float32))
    np.testing.assert_array_equal(D, np.zeros((H, W, 1), dtype=np.float32))


def test_dirt_maps_range_and_seed():
    H, W = 120, 160
    rng = np.random.default_rng(1)
    T, D = dirt_maps(H, W, 0.5, rng)
    assert 0.3 <= float(T.min())
    assert float(T.max()) <= 1.0
    np.testing.assert_array_equal(D, 1.0 - T)
    assert float(T.min()) < 0.95, "T should have visible dirt"

    # Same seed -> identical
    rng2 = np.random.default_rng(1)
    T2, D2 = dirt_maps(H, W, 0.5, rng2)
    np.testing.assert_array_equal(T, T2)
    np.testing.assert_array_equal(D, D2)

    # Different seed -> different
    rng3 = np.random.default_rng(42)
    T3, _ = dirt_maps(H, W, 0.5, rng3)
    assert not np.allclose(T, T3), "Different seeds should give different maps"

    # Density ordering
    rng_lo = np.random.default_rng(1)
    T_lo, D_lo = dirt_maps(H, W, 0.2, rng_lo)
    rng_hi = np.random.default_rng(1)
    T_hi, D_hi = dirt_maps(H, W, 1.0, rng_hi)
    assert float(np.mean(D_hi)) > float(np.mean(D_lo))


# ---------------------------------------------------------------------------
# veil
# ---------------------------------------------------------------------------

def test_veil_uniform():
    H, W = 64, 80
    rng = np.random.default_rng(0)
    D = dirt_maps(H, W, 0.5, rng)[1]
    w = np.full((H, W, 3), 0.5, dtype=np.float32)
    scatter = 0.2

    v = veil(w, D, scatter)
    expected = scatter * D * 0.5
    expected3 = np.broadcast_to(expected, (H, W, 3))
    np.testing.assert_allclose(v, expected3, atol=1e-5)

    # w unchanged
    assert np.allclose(w, 0.5)

    # out= path returns the same object
    out = np.empty((H, W, 3), dtype=np.float32)
    result = veil(w, D, scatter, out=out)
    assert result is out


def test_veil_spreads():
    H = W = 128
    rng = np.random.default_rng(0)
    D = dirt_maps(H, W, 1.0, rng)[1]

    w = np.zeros((H, W, 3), dtype=np.float32)
    cy, cx = H // 2, W // 2
    w[cy - 4:cy + 5, cx - 4:cx + 5, :] = 1.0  # 8x8 bright square

    v = veil(w, D, 1.0)
    assert float(v[cy, cx, 0]) > 0, "veil at centre should be > 0"
    far = v[cy, cx + 20, 0]
    assert far > 0, "veil 20 px away should be > 0 (spread)"
    assert float(v.max()) < 1.0, "veil.max() should be < 1.0"


# ---------------------------------------------------------------------------
# jpeg_roundtrip
# ---------------------------------------------------------------------------

def test_jpeg_roundtrip():
    H, W = 64, 96

    # smooth gradient: R=x ramp, G=y ramp, B=128
    x = np.arange(W, dtype=np.uint8)
    y = np.arange(H, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grad = np.stack([xx, yy, np.full((H, W), 128, dtype=np.uint8)], axis=2)

    # quality 95 -> PSNR > 35 dB
    rt95 = jpeg_roundtrip(grad, 95)
    assert rt95.shape == grad.shape
    assert rt95.dtype == np.uint8
    mse = np.mean((grad.astype(np.float64) - rt95.astype(np.float64)) ** 2)
    psnr = 10.0 * np.log10(255.0 ** 2 / (mse + 1e-30))
    assert psnr > 35, f"PSNR too low: {psnr:.1f} dB"

    # quality 10 on random noise -> shape/dtype kept, larger error
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    rt10 = jpeg_roundtrip(noisy, 10)
    assert rt10.shape == noisy.shape
    assert rt10.dtype == np.uint8
    err10 = np.mean(np.abs(noisy.astype(np.int16) - rt10.astype(np.int16)))
    err95 = np.mean(np.abs(grad.astype(np.int16) - rt95.astype(np.int16)))
    assert err10 > err95, "quality 10 should have larger error than quality 95"

    # blockiness: horizontal diff at 8-px boundaries > interior
    rt10_grad = jpeg_roundtrip(grad, 10)
    boundary_cols = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95]
    boundary_diffs = []
    interior_diffs = []
    for c in boundary_cols:
        if c + 1 < W:
            boundary_diffs.append(np.mean(np.abs(
                rt10_grad[:, c, :].astype(np.int16) - rt10_grad[:, c + 1, :].astype(np.int16)
            )))
    for c in range(1, W - 1):
        if c not in boundary_cols:
            interior_diffs.append(np.mean(np.abs(
                rt10_grad[:, c, :].astype(np.int16) - rt10_grad[:, c + 1, :].astype(np.int16)
            )))
    assert np.mean(boundary_diffs) > np.mean(interior_diffs), (
        "Blockiness: boundary diff should exceed interior"
    )

    # ValueError on bad quality
    with pytest.raises(ValueError):
        jpeg_roundtrip(grad, 0)
    with pytest.raises(ValueError):
        jpeg_roundtrip(grad, 101)


def test_dirt_maps_oracle():
    """Both blob groups add up: full-image sum of every blob (no +-3 sigma box) with the documented draw order."""
    H, W, density = 90, 120, 0.4
    T, D = dirt_maps(H, W, density, np.random.default_rng(5))
    rng, s, A = np.random.default_rng(5), min(H, W), np.zeros((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    groups = ((int(round(200 * density)), (0.005, 0.02), (0.1, 0.3)),
              (1 + int(round(5 * density)), (0.08, 0.25), (0.05, 0.15)))
    for n, sr, ar in groups:
        cx, cy = rng.uniform(0, W, n), rng.uniform(0, H, n)
        sig, amp = rng.uniform(*sr, n) * s, rng.uniform(*ar, n)
        for x0, y0, sg, am in zip(cx, cy, sig, amp):
            A += am * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sg * sg))
    want = np.clip(1 - A, 0.3, 1)
    # box truncation at 3 sigma drops <= amp * exp(-4.5) = 0.0033 per speck; a few overlap
    assert np.abs(T[..., 0] - want).max() < 0.01
    assert (want < 0.9).sum() > 20  # specks are present (the map is not smudges only)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(Path(tempfile.mkdtemp())) if f.__code__.co_argcount else f()
    print("ALL TESTS PASSED")
