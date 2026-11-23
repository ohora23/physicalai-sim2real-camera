"""G1 head-camera demo smoke tests — plain script, no pytest.  MUJOCO_GL=egl.

Skips (exit 0) when the menagerie asset is missing: bash scripts/fetch_menagerie.sh
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "third_party", "mujoco_menagerie", "unitree_g1")):
    print("SKIP test_g1: third_party/mujoco_menagerie/unitree_g1 missing (bash scripts/fetch_menagerie.sh)")
    sys.exit(0)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "examples"))
import g1_walk  # noqa: E402


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def floats(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] != ""])


def main():
    t0 = time.perf_counter()
    model = g1_walk.build_model()
    assert model.camera("head_cam").id >= 0 and model.nmocap == 1
    print("ok scene loads")

    with tempfile.TemporaryDirectory() as d:
        s = g1_walk.run(seconds=1.0, preset="d435i_like", out=d, fast=True)
        pf, ps = read(os.path.join(d, "per_frame.csv")), read(os.path.join(d, "per_step.csv"))
        for key in ("speed_v", "speed_w", "Z", "blur_px", "rs_px", "lap_ratio", "mean_real"):
            assert np.all(np.isfinite(floats(pf, key))), key
        ages = floats(ps, "age")
        assert ages.size and np.all(np.isfinite(ages)) and ages.min() >= 0
        assert abs(s["frames_delivered"] - 30) <= 3, s["frames_delivered"]
        pz = floats(ps, "pelvis_z")
        assert np.abs(pz - g1_walk.KNEES_BENT[2]).max() < 0.1, "weld does not hold the pelvis"
        assert s["blur_max_px"] > 0.5, s["blur_max_px"]
        print(f"ok moving run: {s['frames_delivered']} frames, blur max {s['blur_max_px']:.2f} px")

        s0 = g1_walk.run(seconds=1.0, preset="d435i_like", out=d, fast=True, v_fwd=0.0, a_z=0.0, osc=0.0)
        # static base + static joints: residual servo settling only (numerically ~0)
        assert s0["blur_max_px"] < 0.01, s0["blur_max_px"]
        print(f"ok static run: blur max {s0['blur_max_px']:.2e} px")

    dt = time.perf_counter() - t0
    assert dt < 60, dt
    print(f"ALL G1 TESTS PASSED ({dt:.1f} s)")


if __name__ == "__main__":
    main()
