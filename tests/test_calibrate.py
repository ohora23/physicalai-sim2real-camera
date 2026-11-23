"""C6 end-to-end loop closure: truth cfg -> write_calib_data -> scripts/calibrate.py -> load -> compare.

Plain script, no pytest (MUJOCO_GL=egl). D2/D3 are written at 64x48 (flat_size) with the K scaled
accordingly; the CLI rescales its D1 K to the D2 resolution. D1 is full 640x480.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import glob
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from synth import write_calib_data

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.calib import _DEFAULT_SPREAD, randomize

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "scripts", "calibrate.py")
TRUTH = CameraConfig(640, 480, fx=500, fy=500, cx=319.5, cy=239.5, dist=(-0.3, 0.1, 0.001, -0.001, 0.0),
                     vignette=1.0, gamma=2.2, exposure_ev=-0.5, black_level=0.02, sigma_shot=0.01,
                     sigma_read=0.002, sigma_prnu=0.02, sigma_dsnu=0.005, fps=30, latency_s=0.05,
                     jitter_k=2, jitter_theta=0.005, drop_prob=0.1)
REL = dict(fx=0.005, fy=0.005, gamma=0.03, sigma_shot=0.10, sigma_read=0.20, sigma_prnu=0.10,
           sigma_dsnu=0.15, black_level=0.10, vignette=0.10, fps=0.01, latency_s=0.05, jitter_theta=0.20)
ABS = dict(cx=1.0, cy=1.0, drop_prob=0.01, exposure_ev=0.1)
REPORTS = ("ptc", "radial_profile", "latency_samples", "age", "reprojection_grid")


def run_cli(root, out):
    return subprocess.run([sys.executable, CLI, root, "--out", f"{out}/camera_config.json",
                           "--report", f"{out}/calib_report"],
                          capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})


def test_end_to_end(tmp):
    root, out = f"{tmp}/calib_data", f"{tmp}/out"
    write_calib_data(TRUTH, root, np.random.default_rng(0), flat_size=(64, 48))
    p = run_cli(root, out)
    print(p.stdout, p.stderr, sep="")
    assert p.returncode == 0, p.stderr
    cfg = CameraConfig.load(f"{out}/camera_config.json")
    assert (cfg.width, cfg.height, cfg.jitter_k) == (640, 480, 2)
    for f, tol in REL.items():
        got, want = getattr(cfg, f), getattr(TRUTH, f)
        print(f"  {f:13s} fit={got:.6g} truth={want:.6g} rel_err={got / want - 1:+.4f} (tol {tol})")
        assert abs(got / want - 1) <= tol, f
    for f, tol in ABS.items():
        got, want = getattr(cfg, f), getattr(TRUTH, f)
        print(f"  {f:13s} fit={got:.6g} truth={want:.6g} abs_err={got - want:+.4f} (tol {tol})")
        assert abs(got - want) <= tol, f
    assert os.path.exists(f"{out}/calib_uncertainty.json")
    for name in REPORTS:
        with open(f"{out}/calib_report/{name}.csv") as fh:
            assert len(fh.read().splitlines()) > 2, name
    with open(f"{out}/calib_report/age.csv") as fh:
        ks = float(fh.readline().split(",")[1])
    print(f"  age KS = {ks:.4f}")
    assert ks < 0.08  # grid-based real age: +3 ms latency error -> KS ~0.10

    a, b = randomize(cfg, rng=np.random.default_rng(0)), randomize(cfg, rng=np.random.default_rng(0))
    assert a == b
    for f, s in _DEFAULT_SPREAD.items():
        v, r = getattr(cfg, f), getattr(a, f)
        if f == "dist":
            assert all(abs(x - y) <= s for x, y in zip(r, v))
        elif f in ("exposure_ev", "drop_prob"):
            assert abs(r - v) <= s + 1e-12, f
        else:
            assert abs(r - v) <= s * abs(v) + 1e-12, f


def test_partial(tmp):
    part = f"{tmp}/partial"  # real sessions may lack D3/D4/D5: skip those fits with warnings, keep defaults
    shutil.copytree(f"{tmp}/calib_data", part, ignore=shutil.ignore_patterns("D3", "D4", "D5"))
    p = run_cli(part, f"{tmp}/out_partial")
    print(f"  partial run: exit={p.returncode} warnings={p.stderr.count('WARNING')}")
    assert p.returncode == 0 and all(f"no {d}" in p.stderr for d in ("D3", "D4", "D5")), p.stderr
    cfg = CameraConfig.load(f"{tmp}/out_partial/camera_config.json")
    assert (cfg.gamma, cfg.black_level, cfg.exposure_ev, cfg.latency_s) == (2.2, 0.0, 0.0, 0.0)


def test_ptc_with_one_level(tmp):
    """A D2 session with a single usable level (the rest clipped / saturated): C4 warns and keeps the defaults."""
    one = f"{tmp}/one_level"
    shutil.copytree(f"{tmp}/calib_data", one,
                    ignore=lambda d, names: [n for n in names if n.startswith("level_") and n != "level_3"])
    p = run_cli(one, f"{tmp}/out_one")
    print(f"  one-level run: exit={p.returncode} stderr={p.stderr.strip()}")
    assert p.returncode == 0 and "usable D2 level" in p.stderr, p.stderr
    cfg = CameraConfig.load(f"{tmp}/out_one/camera_config.json")
    assert (cfg.sigma_shot, cfg.sigma_read) == (0.0, 0.0)


def test_ae_drift(tmp):
    root = f"{tmp}/calib_data"  # reuse the e2e tree; ramp level_3 by 0..3%
    files = sorted(glob.glob(f"{root}/D2/level_3/*.png"))
    for i, fn in enumerate(files):
        img = cv2.imread(fn).astype(np.float64) * (1 + 0.03 * i / (len(files) - 1))
        cv2.imwrite(fn, np.clip(np.rint(img), 0, 255).astype(np.uint8))
    p = run_cli(root, f"{tmp}/out_ae")
    print(f"  AE drift run: exit={p.returncode} stderr={p.stderr.strip()}")
    assert p.returncode != 0 and "AE" in p.stderr


def test_randomize_default_spread_rng():
    """Lines 56->58, 59: spread=None -> _DEFAULT_SPREAD; rng=None -> default_rng()."""
    cfg = CameraConfig(640, 480)
    a = randomize(cfg, spread=None)
    b = randomize(cfg, spread=None)
    assert a != b  # different rng seeds each call
    c = randomize(cfg, spread=None, rng=np.random.default_rng(42))
    d = randomize(cfg, spread=None, rng=np.random.default_rng(42))
    assert c == d


def test_randomize_explicit_spread():
    """Branch 56->58: spread is not None -> skip default assignment."""
    cfg = CameraConfig(640, 480, sigma_shot=0.1, sigma_read=0.05)
    custom_spread = {"sigma_shot": 0.5, "sigma_read": 0.3}
    rng_check = np.random.default_rng(10)
    u_shot = rng_check.uniform(-1, 1)
    expected_shot = max(cfg.sigma_shot * (1 + 0.5 * u_shot), 0.0)
    result = randomize(cfg, spread=custom_spread, rng=np.random.default_rng(10))
    assert abs(result.sigma_shot - expected_shot) < 1e-12
    assert result.gamma == cfg.gamma


if __name__ == "__main__":
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        test_end_to_end(tmp)
        test_partial(tmp)
        test_ae_drift(tmp)
        test_randomize_default_spread_rng()
        test_randomize_explicit_spread()
    dt = time.perf_counter() - t0
    print(f"runtime {dt:.1f} s")
    assert dt < 180
    print("OK")
