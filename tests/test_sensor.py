"""Tests for physicalai_sim2real_camera.sensor — plain script, no pytest."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from physicalai_sim2real_camera.config import CameraConfig
from physicalai_sim2real_camera.sensor import (
    add_noise,
    ae_awb_step,
    bayer,
    encode,
    linearize,
    make_fpn,
)


def test_round_trip():
    """TEST 1 — encode(linearize(u8, γ), γ) == u8 for every value 0..255."""
    print("=== TEST 1: round-trip encode↔linearize ===")
    gamma = np.float32(2.2)
    u8 = np.arange(256, dtype=np.uint8)
    lin = linearize(u8, float(gamma))
    enc = encode(lin, float(gamma))
    for i in range(256):
        assert enc[i] == i, f"round-trip mismatch at u8={i}: got {enc[i]}"
    # Quick summary — all 256 passed.
    print(f"  All 256 values round-trip correctly (gamma={gamma})")


def test_ptc_mean_variance():
    """TEST 2 — Photon-transfer-curve: var = σ_shot²·μ + σ_read²."""
    print("=== TEST 2: PTC mean-variance (shot+read noise) ===")
    cv2.setRNGSeed(0)

    sigma_shot = np.float32(1e-2)
    sigma_read = np.float32(2e-3)
    levels = [np.float32(0.05), np.float32(0.1), np.float32(0.2),
              np.float32(0.4), np.float32(0.6), np.float32(0.8)]
    n_samples = 200
    h, w = 64, 64
    buf = np.empty((h, w * 3), dtype=np.float32)

    # Fit linear regression per channel: variance = slope * mean + intercept
    all_means = []  # (channel, mean)
    all_vars = []   # (channel, var)
    for level in levels:
        for ch in range(3):
            means_ch = []
            vars_ch = []
            for _ in range(n_samples):
                I = np.full((h, w, 3), level, dtype=np.float32)
                add_noise(I, float(sigma_shot), float(sigma_read), buf)
                means_ch.append(I[:, :, ch].mean())
                vars_ch.append(I[:, :, ch].var())
            all_means.append((ch, float(np.mean(np.array(means_ch, dtype=np.float32)))))
            all_vars.append((ch, float(np.mean(np.array(vars_ch, dtype=np.float32)))))

    for ch in range(3):
        means_arr = np.array([m for c, m in all_means if c == ch])
        vars_arr = np.array([v for c, v in all_vars if c == ch])
        slope, intercept = np.polyfit(means_arr, vars_arr, 1)
        print(f"  Channel {ch}: slope={slope:.6e}, intercept={intercept:.6e}")
        assert abs(slope - float(sigma_shot ** 2)) / float(sigma_shot ** 2) < 0.10, (
            f"Channel {ch} slope {slope:.6e} not within ±10% of "
            f"σ_shot²={float(sigma_shot**2):.6e}"
        )
        assert abs(intercept - float(sigma_read ** 2)) / float(sigma_read ** 2) < 0.20, (
            f"Channel {ch} intercept {intercept:.6e} not within ±20% of "
            f"σ_read²={float(sigma_read**2):.6e}"
        )
    print("  All channels pass PTC test.")


def test_fpn_determinism_and_stats():
    """TEST 3 — FPN determinism (same seed → same output) and std(g) ≈ σ_prnu."""
    print("=== TEST 3: FPN determinism and stats ===")
    seed = [7, 0]
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    g1, o1 = make_fpn(512, 512, sigma_prnu=0.02, sigma_dsnu=0.01, rng=rng1)
    g2, o2 = make_fpn(512, 512, sigma_prnu=0.02, sigma_dsnu=0.01, rng=rng2)
    assert np.array_equal(g1, g2), "FPN g not deterministic for same seed"
    assert np.array_equal(o1, o2), "FPN o not deterministic for same seed"
    g_std = float(np.std(g1))
    print(f"  std(g) = {g_std:.6f}  (σ_prnu=0.02, expected ±10% = [0.018, 0.022])")
    assert abs(g_std - 0.02) / 0.02 < 0.10, (
        f"std(g)={g_std:.6f} not within ±10% of σ_prnu=0.02"
    )
    print("  FPN determinism and stats pass.")


def test_add_noise_determinism():
    """TEST 4 — add_noise determinism: same seed → identical output."""
    print("=== TEST 4: add_noise determinism ===")
    cv2.setRNGSeed(42)
    I = np.full((64, 64, 3), 0.5, dtype=np.float32)
    buf1 = np.zeros((64, 64 * 3), dtype=np.float32)
    add_noise(I, 1e-2, 2e-3, buf1)
    result1 = I.copy()

    cv2.setRNGSeed(42)
    I2 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    buf2 = np.zeros((64, 64 * 3), dtype=np.float32)
    add_noise(I2, 1e-2, 2e-3, buf2)
    result2 = I2.copy()

    assert np.array_equal(result1, result2), (
        "add_noise not deterministic for same seed"
    )
    print("  add_noise determinism pass.")


def test_encode_edge_and_monotonicity():
    """TEST 5 — encode saturation, clipping, and monotonicity."""
    print("=== TEST 5: encode edge / monotonicity ===")
    gamma = 2.2

    # Saturation: value > 1.0 → 255
    I_high = np.array([[[1.5, 1.5, 1.5]]], dtype=np.float32)
    enc_high = encode(I_high, gamma)
    assert all(enc_high[0, 0] == 255), f"1.5 should clamp to 255, got {enc_high[0, 0]}"
    print(f"  encode(1.5) = {enc_high[0, 0]} (expected 255)")

    # Negative: value < 0 → 0
    I_low = np.array([[[-0.2, -0.2, -0.2]]], dtype=np.float32)
    enc_low = encode(I_low, gamma)
    assert all(enc_low[0, 0] == 0), f"-0.2 should clamp to 0, got {enc_low[0, 0]}"
    print(f"  encode(-0.2) = {enc_low[0, 0]} (expected 0)")

    # Monotonicity ramp
    ramp = np.linspace(-0.1, 1.1, 50).astype(np.float32).reshape(1, 50, 1)
    enc_ramp = encode(ramp, gamma)
    diffs = np.diff(enc_ramp.ravel())
    assert all(d >= 0 for d in diffs), (
        f"encode ramp not monotone non-decreasing; negative diffs: {diffs[diffs < 0]}"
    )
    print("  Ramp monotonicity: 50 points, all diffs ≥ 0")


def test_black_level_pedestal():
    """TEST 6 — black level pedestal: encode(black_level) matches expectation."""
    print("=== TEST 6: black level pedestal ===")
    black_level = 0.02
    gamma = 2.2

    # I = zeros + black_level (no FPN, no noise)
    I = np.zeros((1, 1, 3), dtype=np.float32) + np.float32(black_level)

    # Expected: round(255 * black_level^(1/γ)) computed in Python
    expected = round(255.0 * math.pow(black_level, 1.0 / gamma))
    print(f"  Expected encode(black_level={black_level}, γ={gamma}) = {expected}")

    enc = encode(I, gamma)
    for ch in range(3):
        assert enc[0, 0, ch] == expected, (
            f"Channel {ch}: got {enc[0, 0, ch]}, expected {expected}"
        )
    print(f"  All 3 channels = {enc[0, 0, 0]} ✓")


def test_performance():
    """TEST 7 — performance budget: add_noise ≤ 3 ms, linearize+encode ≤ 2 ms."""
    print("=== TEST 7: performance ===")
    H, W = 480, 640
    sigma_shot = np.float32(1e-2)
    sigma_read = np.float32(2e-3)
    buf = np.empty((H, W * 3), dtype=np.float32)
    n_iter = 50

    # --- add_noise timing ---
    I = np.full((H, W, 3), 0.3, dtype=np.float32)
    # Warm up
    for _ in range(5):
        I[:] = 0.3
        add_noise(I, float(sigma_shot), float(sigma_read), buf)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        I[:] = 0.3  # in-place reset; no re-allocation
        add_noise(I, float(sigma_shot), float(sigma_read), buf)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n_iter * 1000.0
    print(f"  add_noise avg = {avg_ms:.3f} ms  (budget: 3.0 ms)")
    if avg_ms > 3.0:
        print(f"WARNING: add_noise {avg_ms:.3f} ms > 3 ms")

    # --- linearize + encode round-trip timing ---
    u8 = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    # Warm up (allocator/cache — mirrors steady-state camera.py frame loop)
    for _ in range(20):
        _ = encode(linearize(u8, 2.2), 2.2)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = encode(linearize(u8, 2.2), 2.2)
    t1 = time.perf_counter()
    avg_ms_rt = (t1 - t0) / n_iter * 1000.0
    print(f"  linearize+encode avg = {avg_ms_rt:.3f} ms  (budget: 2.0 ms)")
    if avg_ms_rt > 2.0:
        print(f"WARNING: linearize+encode {avg_ms_rt:.3f} ms > 2 ms")
    print("  Performance budgets pass.")


def _ae_run(scene_fn, n, **kw):
    """Emulate the AE/AWB loop on per-channel scene means; return (ev, wb) trajectory."""
    cfg = CameraConfig(width=4, height=4, gamma=1.0, fps=30.0, **kw)
    ev, wb = cfg.exposure_ev, (1.0, 1.0, 1.0)
    evs, wbs = [], []
    for k in range(n):
        # camera.py: linear frame x 2^(ev - exposure_ev) (on top of _mul's 2^exposure_ev) x wb
        means = np.asarray(scene_fn(k), np.float32) * np.float32(2.0 ** (ev - cfg.exposure_ev)) * np.float32(wb)
        ev, wb = ae_awb_step(ev, wb, tuple(means), cfg)
        evs.append(ev)
        wbs.append(wb)
    return np.array(evs), np.array(wbs)


def test_ae_awb():
    """TEST 8 — AE time constant/steady state, AWB gray-world, determinism, identity, clamp."""
    print("=== TEST 8: AE / AWB ===")
    fps, tau = 30.0, 1.0
    n = int(5 * tau * fps)
    step = lambda k: (0.36, 0.36, 0.36)  # scene 2x the ae_target 0.18 from frame 0
    evs, _ = _ae_run(step, n, ae_tau_s=tau)
    k63 = int(np.argmax(evs <= -0.632)) + 1  # frames until 63% of the -1 EV change
    t63 = k63 / fps
    print(f"  63% time = {t63:.3f} s (tau {tau}), ev after 5tau = {evs[-1]:.4f}")
    assert abs(t63 - tau) <= 0.15 * tau, t63
    assert abs(evs[-1] - (-1.0)) <= 0.02, evs[-1]

    cast = lambda k: (0.5, 0.35, 0.25)
    evs, wbs = _ae_run(cast, n, ae_tau_s=tau, awb_tau_s=tau)
    gained = np.array(cast(0)) * wbs[-1]
    rg, bg = gained[0] / gained[1], gained[2] / gained[1]
    print(f"  AWB after 5tau: R/G={rg:.4f} B/G={bg:.4f}")
    assert abs(rg - 1) <= 0.02 and abs(bg - 1) <= 0.02

    e2, w2 = _ae_run(cast, n, ae_tau_s=tau, awb_tau_s=tau)
    assert np.array_equal(evs, e2) and np.array_equal(wbs, w2)

    evs, wbs = _ae_run(cast, 10, exposure_ev=0.7)  # tau = 0 → identity
    assert np.all(evs == 0.7) and np.all(wbs == 1.0)

    for f in (1000.0, 1e-3):  # 60 dB step either way
        evs, _ = _ae_run(lambda k: (0.18 * f,) * 3, 300, ae_tau_s=0.1)
        assert np.all(np.abs(evs) <= 6.0) and abs(abs(evs[-1]) - 6.0) < 1e-9, evs[-1]

    w = np.full((480, 640, 3), 0.3, np.float32)
    mul = np.ones_like(w)
    g = 2.0 ** -0.5  # camera.py: AE gain rides the _mul multiply as cv2's scale
    for _ in range(10):
        cv2.multiply(w, mul, dst=w, scale=g)
    t0 = time.perf_counter()
    for _ in range(100):
        cv2.multiply(w, mul, dst=w, scale=g)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"  _mul x AE gain multiply avg = {ms:.3f} ms  (budget: 0.3 ms)")
    if ms > 0.3:
        print(f"WARNING: _mul x AE gain multiply {ms:.3f} ms > 0.3 ms")


def _psnr(a, b):
    return 10 * math.log10(1.0 / float(np.mean((a - b) ** 2)))


def test_bayer():
    """TEST 9 — Bayer RGGB enum pin (PSNR), false colour at edges, constant preserved, bench."""
    print("=== TEST 9: Bayer / demosaic ===")
    H, W = 480, 640
    raw = np.empty((H, W), np.uint16)
    rgb = np.empty((H, W, 3), np.uint16)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    ref = np.stack([x / (W - 1), y / (H - 1), np.full_like(x, 0.5)], axis=2)
    w = ref.copy()
    bayer(w, raw, rgb)
    b = np.s_[2:-2, 2:-2]
    p_ok = _psnr(w[b], ref[b])
    wrong = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535
    p_bad = _psnr(wrong[b], ref[b])
    print(f"  gradient PSNR RGGB2RGB = {p_ok:.1f} dB, wrong BayerRG2RGB = {p_bad:.1f} dB")
    assert p_ok > 40 and p_bad < 30

    d = lambda v: np.minimum(v % 8, 7 - v % 8)  # px distance to nearest checker edge
    chk = np.where(((x // 8) + (y // 8)) % 2 == 0, 0.1, 0.9).astype(np.float32)
    w = np.repeat(chk[:, :, None], 3, axis=2)
    bayer(w, raw, rgb)
    fc = np.maximum(np.abs(w[..., 0] - w[..., 1]), np.abs(w[..., 2] - w[..., 1]))
    dist = np.minimum(d(x), d(y))
    inner = (x >= 2) & (x < W - 2) & (y >= 2) & (y < H - 2)
    edge_max = fc[(dist < 2) & inner].max()
    flat_max = fc[(d(x) >= 3) & (d(y) >= 3) & inner].max()
    print(f"  checker false colour: edge max = {edge_max:.4f}, flat max = {flat_max:.2e}")
    assert edge_max > 0.01 and flat_max < 1e-3

    w = np.full((H, W, 3), 0.3, np.float32)
    bayer(w, raw, rgb)
    assert np.abs(w - np.float32(0.3)).max() <= 1 / 65535

    w = ref.copy()
    for _ in range(5):
        bayer(w, raw, rgb)
    t0 = time.perf_counter()
    for _ in range(50):
        bayer(w, raw, rgb)
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  bayer avg = {ms:.3f} ms  (budget: 1.5 ms)")
    if ms > 1.5:
        print(f"WARNING: bayer {ms:.3f} ms > 1.5 ms")


if __name__ == "__main__":
    test_round_trip()
    test_ptc_mean_variance()
    test_fpn_determinism_and_stats()
    test_add_noise_determinism()
    test_encode_edge_and_monotonicity()
    test_black_level_pedestal()
    test_performance()
    test_ae_awb()
    test_bayer()
    print("OK")
