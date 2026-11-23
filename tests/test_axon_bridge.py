"""Axon bridge age distribution == CameraStream age distribution (pytest-collectable; also runs standalone).

Runs examples/axon_bridge.py for 5 s (subprocess) and compares out/axon/sub_age.csv vs pub_age.csv.
Skips (exit 0) when the axon bindings are not built.
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXON_PY = os.path.join(ROOT, "third_party", "axon", "build", "python")
OUT = os.path.join(ROOT, "out", "axon")


def ks(a, b):
    """Two-sample KS statistic via numpy ECDFs (same as scripts/calibrate.py)."""
    grid = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a)
                               - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


def _run_checks(backend):
    env = dict(os.environ, MUJOCO_GL="egl",
               PYTHONPATH=os.pathsep.join(filter(None, [AXON_PY, os.environ.get("PYTHONPATH")])))
    t = time.perf_counter()
    subprocess.run([sys.executable, os.path.join(ROOT, "examples", "axon_bridge.py"),
                    "--seconds", "5", "--backend", backend], env=env, check=True, timeout=90)
    dt = time.perf_counter() - t
    pa = np.genfromtxt(os.path.join(OUT, "pub_age.csv"), delimiter=",", names=True)
    sa = np.genfromtxt(os.path.join(OUT, "sub_age.csv"), delimiter=",", names=True)
    # pub ages sit on the 2 ms control grid, so ANY continuous shift (poll lag + transport) of the wall-clock
    # age_sub puts KS at one atom's mass (~0.06). KS is therefore gated on age_tick (same read, logical tick
    # time); the wall-clock age is gated by its mean and its KS printed.
    # Round to 1 us: pub age uses data.time (summed mj_step dt), age_tick uses k*DT; 1e-15 s float
    # differences would otherwise split tied atoms and inflate KS.
    k = ks(np.round(sa["age_tick"], 6), np.round(pa["age"], 6))
    k_wall = ks(sa["age_sub"], pa["age"])
    dmean = abs(sa["age_sub"].mean() - pa["age"].mean())
    cover = len(np.unique(sa["seqno"])) / len(np.unique(pa["n"]))
    # staleness_ns only counts time since publish; age_sub - staleness must recover avail - capture,
    # whose mean is latency_s + k*theta (0.05 + 2*0.005 in examples/axon_bridge.py).
    d_stale = abs(np.mean(sa["age_sub"] - sa["staleness_ns"] * 1e-9) - 0.060)
    print(f"[{backend}] runtime {dt:.1f} s, KS(tick) {k:.4f}, KS(wall) {k_wall:.4f}, "
          f"|dmean(wall)| {1e3 * dmean:.2f} ms, seqno coverage {cover:.3f}, "
          f"n_pub {len(pa)}, n_sub {len(sa)}")
    assert k < 0.05, f"KS {k:.4f} >= 0.05"
    assert dmean < 0.003, f"|mean diff| {1e3 * dmean:.2f} ms >= 3 ms"
    assert cover >= 0.9, f"seqno coverage {cover:.3f} < 0.9"
    assert d_stale < 0.003, f"mean(age_sub - staleness) off by {1e3 * d_stale:.2f} ms from latency + k*theta"
    assert dt < 90, f"runtime {dt:.1f} s"


def _skip(msg):
    """Skip under pytest; print and return False when run as a script."""
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    print(f"SKIP: {msg}")
    return False


def _run_all():
    sys.path.append(AXON_PY)
    try:
        import axon  # noqa: F401
    except ImportError:
        return _skip("axon bindings unavailable (run scripts/build_axon.sh; PYTHONPATH=third_party/axon/build/python)")
    _run_checks("custom")
    if os.access("/dev/udmabuf", os.W_OK):
        _run_checks("udmabuf")
    else:
        print("NOTE: /dev/udmabuf not writable; udmabuf backend skipped")
    return True


def test_axon_bridge():
    """pytest entry point (skips via ``pytest.skip`` when prerequisites are missing)."""
    _run_all()


def main():
    if _run_all():
        print("test_axon_bridge: PASS")


if __name__ == "__main__":
    main()
