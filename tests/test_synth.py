"""C0b tests for tests/synth.py — plain script, no pytest (MUJOCO_GL=egl)."""

import time

import cv2
import numpy as np
from synth import synth_chart, synth_dark, synth_flat, synth_geometry, synth_timing

from physicalai_sim2real_camera import CameraConfig

DIST = (-0.3, 0.1, 0.001, -0.001, 0.0)
PHOTO = CameraConfig(640, 480, sigma_shot=0.01, sigma_read=0.002, sigma_prnu=0.02,
                     sigma_dsnu=0.005, black_level=0.02, vignette=1.0, seed=3)


def test_geometry():
    cfg = CameraConfig(640, 480, fx=500, fy=500, dist=DIST)
    imgs, pattern, sq, K, dist = synth_geometry(cfg, np.random.default_rng(0))
    assert len(imgs) == 20 and pattern == (7, 5) and sq == 0.05 and dist == DIST
    assert K[0, 0] == K[1, 1] == 500 and K[0, 2] == 319.5 and K[1, 2] == 239.5
    for im in imgs:
        assert im.shape == (480, 640, 3) and im.dtype == np.uint8
        ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), pattern)
        assert ok and len(c) == 35
    imgs2 = synth_geometry(cfg, np.random.default_rng(0))[0]
    assert all(np.array_equal(a, b) for a, b in zip(imgs, imgs2))
    print("geometry: 20 views, 35 corners each, deterministic")


def test_flat_dark():
    cfg, g = PHOTO, PHOTO.gamma
    s, t = synth_flat(cfg, np.random.default_rng(1), 0.4, 50)
    assert s.shape == (50, 480, 640, 3) and s.dtype == np.uint8
    c = s[:, 230:250, 310:330].astype(np.float64)
    V_c = float(t["vignette_map"][240, 320, 0])
    want = 255 * (0.4 * 2 ** cfg.exposure_ev * V_c + cfg.black_level) ** (1 / g)
    print(f"flat centre mean DN {c.mean():.2f} (want {want:.2f}, V_c={V_c:.4f})")
    assert abs(c.mean() - want) <= 3
    tstd = c.std(axis=0).mean(axis=(0, 1))  # temporal std per channel
    print(f"flat temporal std per channel {tstd}")
    assert tstd.min() > 0 and tstd.max() / tstd.min() < 1.1
    s2, _ = synth_flat(cfg, np.random.default_rng(1), 0.4, 50)
    assert np.array_equal(s, s2)

    d, td = synth_dark(cfg, np.random.default_rng(2), 20)
    assert td["vignette_map"] is None and d.shape == (20, 480, 640, 3)
    want = 255 * cfg.black_level ** (1 / g)
    print(f"dark mean DN {d.mean():.2f} (want {want:.2f})")
    assert abs(d.mean() - want) <= 2
    assert np.array_equal(td["g"], t["g"]) and np.array_equal(td["o"], t["o"])  # same FPN


def test_chart():
    dn = synth_chart(PHOTO, [0.03, 0.09, 0.19, 0.36, 0.59, 0.9])
    print(f"chart DN {dn}")
    assert dn.dtype.kind == "f" and np.all(np.diff(dn) > 0) and dn[-1] <= 255


def test_timing():
    cfg = CameraConfig(640, 480, fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005,
                       drop_prob=0.1, seed=0)
    r = synth_timing(cfg, np.random.default_rng(4))
    assert len(r["event_t"]) == 300 and np.all(np.diff(r["event_t"]) >= 0.3)
    n = len(r["avail_t"])
    assert n == len(r["frame_seq"]) == len(r["event_seen"]) == len(r["t_c"])
    assert np.all(r["avail_t"] >= r["t_c"] + cfg.latency_s)
    assert np.all(np.diff(r["frame_seq"]) >= 1) and np.all(np.diff(r["event_seen"]) >= 0)
    assert r["event_seen"][-1] == 300
    drops = r["n_frames"] - n
    print(f"timing: {r['n_frames']} frames, {drops} dropped (expect {0.1 * r['n_frames']:.0f})")
    assert abs(drops - 0.1 * r["n_frames"]) <= 0.3 * 0.1 * r["n_frames"]
    r2 = synth_timing(cfg, np.random.default_rng(4))
    assert all(np.array_equal(r[k], r2[k]) for k in r)


if __name__ == "__main__":
    t0 = time.time()
    test_geometry()
    test_flat_dark()
    test_chart()
    test_timing()
    dt = time.time() - t0
    print(f"all synth tests passed in {dt:.1f} s")
    assert dt < 90
