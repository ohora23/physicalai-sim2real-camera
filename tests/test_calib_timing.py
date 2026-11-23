"""C5 tests for fit_timing — plain script, no pytest (MUJOCO_GL=egl)."""

import time

import numpy as np
from synth import synth_timing

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.calib_timing import fit_timing

CFG = CameraConfig(640, 480, fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.1)


def _fit(d, **kw):
    return fit_timing(d["event_t"], d["avail_t"], d["frame_seq"], d["event_seen"], **kw)


def _check_c1(r, cfg=CFG):
    assert abs(r["fps"] / cfg.fps - 1) < 0.01, r["fps"]
    assert abs(r["latency_s"] / cfg.latency_s - 1) < 0.05, r["latency_s"]
    assert abs(r["jitter_theta"] / cfg.jitter_theta - 1) < 0.20, r["jitter_theta"]
    assert abs(r["drop_prob"] - cfg.drop_prob) < 0.01, r["drop_prob"]
    assert r["jitter_k"] == 2 and r["n_samples"] == len(r["samples"])


def test_loop_closure():
    d = synth_timing(CFG, np.random.default_rng(0), 300)
    r = _fit(d)
    print(f"C1: fps={r['fps']:.3f} latency_s={r['latency_s']:.5f} theta={r['jitter_theta']:.5f} "
          f"drop={r['drop_prob']:.4f} n_samples={r['n_samples']} excluded={300 - r['n_samples']} "
          f"warnings={r['warnings']}")
    _check_c1(r)
    assert not r["warnings"]  # regular grid must not fall back to moments


def test_no_jitter():
    cfg = CameraConfig(640, 480, fps=30, latency_s=0.05)
    r = _fit(synth_timing(cfg, np.random.default_rng(1), 300))
    print(f"C2: latency_s={r['latency_s']:.5f} theta={r['jitter_theta']:.5f} n={r['n_samples']} "
          f"warnings={r['warnings']}")
    assert abs(r["latency_s"] / 0.05 - 1) < 0.03 and r["jitter_theta"] < 0.002
    assert r["drop_prob"] == 0


def test_exact_exposure():
    # D5b: exposure time read directly per frame -> samples avail_t - t_c, no quantisation
    d = synth_timing(CFG, np.random.default_rng(2), 300)
    n = len(d["t_c"])
    r = fit_timing(d["t_c"], d["avail_t"], d["frame_seq"], np.arange(1, n + 1), fps_hint=30,
                   quantized=False)
    print(f"C3: latency_s={r['latency_s']:.5f} theta={r['jitter_theta']:.5f} n={r['n_samples']}")
    assert r["n_samples"] > 0.8 * n
    assert abs(r["latency_s"] / 0.05 - 1) < 0.03, r["latency_s"]
    assert abs(r["jitter_theta"] / 0.005 - 1) < 0.15, r["jitter_theta"]


def test_irregular_grid():
    # capture times wobble by up to 0.4 frame (non-uniform host clock) -> moment fallback
    d = synth_timing(CFG, np.random.default_rng(3), 300)
    t_c = d["t_c"] + 0.4 / 30 * np.sin(2 * np.pi * d["t_c"] / 10)
    r = fit_timing(d["event_t"], t_c + d["latency"], d["frame_seq"],
                   np.searchsorted(d["event_t"], t_c, side="right"))
    print(f"C3b: latency_s={r['latency_s']:.5f} theta={r['jitter_theta']:.5f} warnings={r['warnings']}")
    assert any("irregular" in w for w in r["warnings"])
    assert abs(r["latency_s"] / 0.05 - 1) < 0.10 and abs(r["jitter_theta"] / 0.005 - 1) < 0.4


def test_seeds():
    rs = [_fit(synth_timing(CFG, np.random.default_rng(s), 300)) for s in range(10)]
    lat = np.array([r["latency_s"] for r in rs])
    th = np.array([r["jitter_theta"] for r in rs])
    print(f"C4: latency_s {lat.min():.5f}..{lat.max():.5f} (mean {lat.mean():.5f}, sd {lat.std():.5f}); "
          f"theta {th.min():.5f}..{th.max():.5f} (mean {th.mean():.5f}, sd {th.std():.5f})")
    for r in rs:
        _check_c1(r)


def test_low_sample_count():
    """Lines 78-79: n < MIN_SAMPLES (50) -> warning + theta=0."""
    cfg = CameraConfig(640, 480, fps=30, latency_s=0.05)
    d = synth_timing(cfg, np.random.default_rng(100), 10)  # only 10 samples
    r = fit_timing(d["event_t"], d["avail_t"], d["frame_seq"], d["event_seen"])
    print(f"low_samples: n={r['n_samples']} theta={r['jitter_theta']} warnings={r['warnings']}")
    assert r["n_samples"] < 50
    assert r["jitter_theta"] == 0.0
    assert any("only" in w and "< 50" in w for w in r["warnings"])


def test_negative_latency():
    """Line 86: latency < 0 -> warning."""
    # Create timing data where avail_t is very close to event_t -> negative latency
    event_t = np.arange(0, 10, 0.03333, dtype=float)  # ~30 Hz
    avail_t = event_t + 0.001  # tiny latency, almost zero
    frame_seq = np.arange(len(event_t), dtype=np.int64)
    event_seen = np.arange(1, len(event_t) + 1, dtype=np.int64)
    r = fit_timing(event_t, avail_t, frame_seq, event_seen, fps_hint=30)
    print(f"neg_latency: latency_s={r['latency_s']:.5f} warnings={r['warnings']}")
    assert r["latency_s"] < 0
    assert any("negative latency" in w for w in r["warnings"])


def test_variance_le_zero():
    """Line 82: corrected variance <= 0 -> warning with perfectly regular grid."""
    # Use integer-like values where floating-point arithmetic is exact
    n_frames = 300
    # period=1.0 is exactly representable
    event_t = np.arange(n_frames, dtype=float)          # [0,1,2,...,299]
    avail_t = 0.5 + np.arange(n_frames, dtype=float)   # [0.5,1.5,2.5,...,299.5]
    frame_seq = np.arange(n_frames, dtype=np.int64)
    event_seen = np.arange(1, n_frames + 1, dtype=np.int64)
    r = fit_timing(event_t, avail_t, frame_seq, event_seen, fps_hint=1.0, quantized=True)
    print(f"var_le_zero: n={r['n_samples']} theta={r['jitter_theta']} warnings={r['warnings']}")
    assert r["n_samples"] >= 50
    assert r["jitter_theta"] < 1e-10
    assert any("variance <= 0" in w or "quantisation dominates" in w for w in r["warnings"])


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_loop_closure()
    test_no_jitter()
    test_exact_exposure()
    test_irregular_grid()
    test_seeds()
    test_low_sample_count()
    test_negative_latency()
    test_variance_le_zero()
    dt = time.perf_counter() - t0
    print(f"C5: runtime {dt:.2f} s")
    assert dt < 20
    print("all calib_timing tests passed")
