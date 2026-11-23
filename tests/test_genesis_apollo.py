"""examples/genesis_apollo.py: head-camera mount, hidden collision geoms, motion limits, MuJoCo capture (EGL).

Skips when the menagerie asset is missing: bash scripts/fetch_menagerie.sh
"""

import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, "third_party", "mujoco_menagerie", "apptronik_apollo")),
    reason="third_party/mujoco_menagerie/apptronik_apollo missing (bash scripts/fetch_menagerie.sh)")


@pytest.fixture(scope="module")
def ex():
    import genesis_apollo

    return genesis_apollo


def test_apollo_scene(ex):
    """The head camera rides on the neck, sits where declared at the stand pose and looks forward, pitched down.

    Position and axes are compared through the MJCF text round trip (6 significant digits: up to ~5e-7 m, ~2e-6 rad).
    """
    m = mujoco.MjModel.from_xml_path(str(ex.write_xml()))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    c = m.camera(ex.CAM).id
    p = math.radians(ex.CAM_PITCH_DEG)
    assert m.body(m.cam_bodyid[c]).name == "neck_pitch_link"
    R = d.cam_xmat[c].reshape(3, 3)
    np.testing.assert_allclose(d.cam_xpos[c], ex.CAM_WORLD, atol=2e-6)
    np.testing.assert_allclose(-R[:, 2], [math.cos(p), 0, -math.sin(p)], atol=1e-5)  # view axis
    np.testing.assert_allclose(R[:, 1], [math.sin(p), 0, math.cos(p)], atol=1e-5)  # image up: not upside down
    # the camera turns with the neck: its view axis rotates by the joint angle about the joint axis (tilted ~15 deg
    # from vertical at the stand pose, so the azimuth alone would not show 0.3)
    j, f0 = m.joint("neck_yaw").id, -R[:, 2]
    d.qpos[m.jnt_qposadr[j]] += 0.3
    mujoco.mj_forward(m, d)
    f1, a = -d.cam_xmat[c].reshape(3, 3)[:, 2], d.xaxis[j]
    assert abs(math.atan2(np.cross(f0, f1) @ a, f0 @ f1 - (f0 @ a) * (f1 @ a)) - 0.3) < 1e-4
    hidden = m.geom_group == 3
    assert hidden.any() and np.all(m.geom_rgba[hidden, 3] == 0)  # Genesis would draw them
    assert all(m.geom(name).id >= 0 for name, *_ in ex.BOXES)


def test_apollo_motion_in_limits(ex):
    """The scripted pose stays inside every hinge range over the whole 34 s and actually moves the neck fast."""
    m = mujoco.MjModel.from_xml_path(str(ex.write_xml()))
    motion = ex.Motion(m)
    t = np.arange(0.0, 34.0, 0.01)
    q = np.stack([motion(x) for x in t])
    for j in range(1, m.njnt):
        lo, hi = m.jnt_range[j]
        qj = q[:, m.jnt_qposadr[j]]
        assert lo - 1e-9 <= qj.min() and qj.max() <= hi + 1e-9, m.joint(j).name
    yaw_rate = np.abs(np.diff(q[:, motion.adr["neck_yaw"]])) / 0.01
    assert yaw_rate.max() > 1.0  # fast turns: blur and rolling shutter become visible
    assert np.array_equal(q[0, :7], motion.q0[:7]) and np.array_equal(q[-1, :7], motion.q0[:7])  # base stays put


def test_apollo_mujoco_capture(ex):
    """Half a second through the calibrated preset in MuJoCo: frames arrive, late by about the preset latency."""
    cfg = ex.make_cfg("calibrated")
    rows = ex.capture("mujoco", cfg, 0.5)
    age = np.array(rows["age"])
    print(f"{len(age)} frames, age {1e3 * age.mean():.1f} ms")
    assert len(age) >= 8 and rows["real"][0].shape == (ex.H, ex.W, 3) and rows["depth"][0].shape == (ex.H, ex.W)
    # age = latency + Gamma(jitter_k, jitter_theta) jitter (mean 6 ms) + up to one physics step
    assert age.mean() > cfg.latency_s and age.max() < cfg.latency_s + 0.03 + ex.DT
    cfgs = {name: ex.make_cfg(name) for name in ex.CFGS}
    assert cfgs["calibrated"].dist == (0.0,) * 5 and cfgs["calibrated"].depth_sigma_b > 0
    assert all(cfgs[n].dist == ex.DIST and cfgs[n].latency_s == cfg.latency_s for n in ("geom", "static", "wingeom"))
    assert cfgs["static"].sigma_shot > 0 and cfgs["static"].ae_tau_s > 0 and cfgs["static"].exposure_s == 0
    assert cfgs["wingeom"].exposure_s > 0 and cfgs["wingeom"].sigma_shot == 0 and cfgs["geom"].sigma_shot == 0
