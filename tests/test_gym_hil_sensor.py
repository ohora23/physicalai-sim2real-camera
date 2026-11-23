"""examples/gym_hil_sensor.py smoke test; runs only where gym-hil is installed (.venv-hil, MuJoCo < 3.9)."""

import json
import os
import sys

import pytest

pytest.importorskip("gym_hil")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import gym_hil_sensor as ghs  # noqa: E402


def test_scripted_pick_with_realistic_cameras(tmp_path):
    """The scripted pick succeeds through the wrapper and both outputs are written. Frame age = time - capture_t
    of the newest delivered frame: >= latency, and < the next capture's arrival = latency + capture spacing
    (2/fps with one drop) + the 2 ms capture grid + the Gamma(2, theta) jitter's 99.9 % quantile (9.23 theta)."""
    s = ghs.run(steps=80, out=str(tmp_path))
    assert s["success"] and (tmp_path / "compare.png").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["steps"] == s["steps"]
    r = ghs.REALISTIC
    lo, hi = r["latency_s"], r["latency_s"] + 2 / r["fps"] + 0.002 + 9.23 * r["jitter_theta"]
    for a in s["age_s"].values():
        assert lo - 1e-9 <= a["min"] and a["max"] < hi
