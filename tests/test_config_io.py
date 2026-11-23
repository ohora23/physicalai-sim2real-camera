"""Acceptance tests for CameraConfig save/load and calib.randomize (C0a)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.calib import randomize


def non_default_cfg():
    return CameraConfig(
        width=640,
        height=480,
        fx=500.1,
        fy=499.7,
        cx=319.5,
        cy=239.5,
        dist=(-0.3, 0.1, 0.001, -0.001, 0.002),
        vignette=1.2,
        gamma=2.1,
        exposure_ev=0.5,
        sigma_shot=0.01,
        sigma_read=0.002,
        sigma_prnu=0.02,
        sigma_dsnu=0.005,
        black_level=0.02,
        fps=60,
        phase_s=0.01,
        latency_s=0.05,
        jitter_k=2,
        jitter_theta=0.005,
        drop_prob=0.1,
        seed=7,
    )


def test_roundtrip():
    cfg = non_default_cfg()
    with tempfile.TemporaryDirectory() as d:
        path = cfg.save(os.path.join(d, "cfg.json"))
        loaded = CameraConfig.load(path)
    assert cfg == loaded
    assert isinstance(loaded.dist, tuple)
    assert all(isinstance(x, float) for x in loaded.dist)


def test_flat_field_roundtrip():
    import dataclasses
    ff = np.random.default_rng(0).standard_normal((480, 640, 1)).astype(np.float32)
    cfg = dataclasses.replace(non_default_cfg(), flat_field=ff)
    with tempfile.TemporaryDirectory() as d:
        path = cfg.save(os.path.join(d, "cfg.json"))
        assert os.path.exists(path + ".flat_field.npy")
        loaded = CameraConfig.load(path)
    assert np.array_equal(loaded.flat_field, ff) and loaded.flat_field.dtype == np.float32
    assert cfg == loaded  # flat_field is compare=False; other fields must still match


def test_flat_field_bad_shape():
    bad = np.zeros((480, 640), np.float32)  # missing channel dim
    try:
        CameraConfig(640, 480, flat_field=bad)
    except ValueError as e:
        assert "480" in str(e) and "640" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong flat_field shape")


def test_randomize():
    cfg = non_default_cfg()

    def in_range(new, base, s, absolute, lo=None, hi=None):
        if absolute:
            ok = abs(new - base) <= s + 1e-12
        else:
            ok = abs(new - base) <= abs(base) * s + 1e-12
        if lo is not None:
            ok = ok and new >= lo - 1e-12
        if hi is not None:
            ok = ok and new <= hi + 1e-12
        return ok

    r = randomize(cfg, rng=np.random.default_rng(0))

    for x, x0 in zip(r.dist, cfg.dist):
        assert in_range(x, x0, 0.02, absolute=True)
    assert in_range(r.sigma_shot, cfg.sigma_shot, 0.2, absolute=False, lo=0.0)
    assert in_range(r.sigma_read, cfg.sigma_read, 0.2, absolute=False, lo=0.0)
    assert in_range(r.sigma_prnu, cfg.sigma_prnu, 0.2, absolute=False, lo=0.0)
    assert in_range(r.sigma_dsnu, cfg.sigma_dsnu, 0.2, absolute=False, lo=0.0)
    assert in_range(r.vignette, cfg.vignette, 0.2, absolute=False, lo=0.0)
    assert in_range(r.latency_s, cfg.latency_s, 0.2, absolute=False)
    assert in_range(r.jitter_theta, cfg.jitter_theta, 0.2, absolute=False, lo=0.0)
    assert in_range(r.gamma, cfg.gamma, 0.05, absolute=False)
    assert in_range(r.exposure_ev, cfg.exposure_ev, 0.3, absolute=True)
    assert in_range(r.drop_prob, cfg.drop_prob, 0.02, absolute=True, lo=0.0, hi=1.0)

    r2 = randomize(cfg, rng=np.random.default_rng(0))
    assert r == r2, "same seed must reproduce identical randomize() output"


def test_calib_exports():
    import physicalai_sim2real_camera.calib as calib

    for name in (
        "fit_intrinsics",
        "fit_flat_field",
        "fit_response",
        "fit_ptc",
        "match_exposure",
        "fit_timing",
        "randomize",
    ):
        assert hasattr(calib, name), name


if __name__ == "__main__":
    test_roundtrip()
    test_flat_field_roundtrip()
    test_flat_field_bad_shape()
    test_randomize()
    test_calib_exports()
    print("all config_io tests passed")


def test_rational_dist():
    """8-term (rational) dist: kept by preset and save/load; 5-term form kept when k4-k6 are zero; past k6 refused."""
    rational = (0.2299, -0.0713, -0.0003, 0.0004, -0.0026, 0.5551, -0.0717, -0.0152)
    cfg = CameraConfig.preset("calibrated", 640, 400, dist=rational)
    assert cfg.dist == rational
    assert len(CameraConfig.preset("calibrated", 640, 400, dist=rational[:5] + (0.0,) * 9).dist) == 5
    assert CameraConfig.preset("calibrated", 640, 400, dist=rational[:4]).dist == rational[:4] + (0.0,)
    for bad in (rational + (0.01,), rational[:3]):
        try:
            CameraConfig.preset("calibrated", 640, 400, dist=bad)
            raise AssertionError(f"dist {bad} must be refused")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as d:
        p = CameraConfig(640, 400, dist=rational).save(os.path.join(d, "c.json"))
        assert CameraConfig.load(p).dist == rational
    from physicalai_sim2real_camera.optics import outside_domain_fraction

    K = np.array([[286.3, 0.0, 320.1], [0.0, 286.3, 202.2], [0.0, 0.0, 1.0]])
    base = outside_domain_fraction(K, rational, 640, 400, step=8)
    rng = np.random.default_rng(0)
    # a scalar spread leaves a rational (calibrated wide-angle) model as is: the fold fraction stays the base's
    d = randomize(CameraConfig(640, 400, dist=rational), {"dist": 0.02}, rng).dist
    assert d == rational and outside_domain_fraction(K, d, 640, 400, step=8) == base > 0
    assert randomize(CameraConfig(640, 400, dist=(0.1, 0, 0, 0, 0)), {"dist": 0.02}, rng).dist != (0.1, 0, 0, 0, 0)
    # a 5-term spread (estimate_to_config) on an 8-term dist pads with zeros; a longer spread is refused
    d = randomize(CameraConfig(640, 400, dist=rational), {"dist": (0.01,) * 5}, rng).dist
    assert d[5:] == rational[5:]
    try:
        randomize(CameraConfig(640, 400, dist=(0.1, 0, 0, 0, 0)), {"dist": (0.01,) * 8}, rng)
        raise AssertionError("longer spread must be refused")
    except ValueError:
        pass
    for n in (3, 6, 7, 9):  # CameraConfig itself checks the length
        try:
            CameraConfig(640, 400, dist=(0.0,) * n)
            raise AssertionError(f"{n}-term dist must be refused")
        except ValueError:
            pass
