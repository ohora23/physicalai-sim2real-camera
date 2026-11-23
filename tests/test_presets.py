"""CameraConfig.preset (calibrated / ideal) and set_extrinsics (OpenCV optical frame -> MJCF camera)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig, SimCamera, perturb_extrinsics, set_extrinsics
from physicalai_sim2real_camera.config import TYPICAL_UNCALIBRATED

K = np.array([[300.0, 0, 161.0], [0, 302.0, 118.0], [0, 0, 1]])

# flat (ambient-only) shading, so the red mask is the silhouette (a top light biases it 1.2 px); the camera
# parent body is rotated 90 deg about z and offset, so the extrinsic must be read in the body frame (its hinge
# only keeps the body a Genesis link with a qpos; it stays at 0)
SCENE = """<mujoco><visual><headlight ambient='1 1 1' diffuse='0 0 0' specular='0 0 0'/></visual><worldbody>
<geom name='red' type='sphere' pos='1.0 0.15 0.6' size='0.02' rgba='1 0 0 1'/>
<body name='mount' pos='0.5 0.2 1.0' quat='0.70710678 0 0 0.70710678'><joint type='hinge' axis='0 0 1'/>
  <inertial pos='0 0 0' mass='0.1' diaginertia='1e-3 1e-3 1e-3'/><camera name='c' fovy='45'/></body>
</worldbody></mujoco>"""


def test_preset_fields():
    """ideal is the identity; calibrated = measured geometry + typical values for what calibration misses."""
    assert CameraConfig.preset("ideal", 320, 240) == CameraConfig(320, 240)
    rect = CameraConfig.preset("calibrated", 320, 240, K=K)
    assert (rect.fx, rect.fy, rect.cx, rect.cy) == (300.0, 302.0, 161.0, 118.0)
    assert rect.dist == (0.0,) * 5  # no dist: rectified frames
    assert all(getattr(rect, k) == v for k, v in TYPICAL_UNCALIBRATED.items())
    raw = CameraConfig.preset("calibrated", 320, 240, dist=[-0.1, 0.02, 0.001, -0.001], latency_s=0.03)
    assert raw.fx is None and raw.dist == (-0.1, 0.02, 0.001, -0.001, 0.0) and raw.latency_s == 0.03
    assert CameraConfig.preset("calibrated", 320, 240, dist=[0.1, 0, 0, 0, 0.01, 0, 0, 0]).dist[4] == 0.01
    for bad in (dict(name="nope"), dict(K=np.eye(2)), dict(dist=[0.1, 0.0]), dict(dist=[0, 0, 0, 0, 0, 0, 0, 0, 0.3])):
        with pytest.raises(ValueError):
            CameraConfig.preset(**{"name": "calibrated", "width": 320, "height": 240, **bad})


def look_at(c, target):
    """OpenCV optical-frame camera pose (x right, y down, z forward) at ``c`` looking at ``target``, world +z up."""
    z = np.subtract(target, c) / np.linalg.norm(np.subtract(target, c))
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, np.cross(z, x), z]), c
    return T


def calibrated_pose(model, c=(0.0, 0.0, 1.0), perturb=None):
    """A camera-to-parent-body extrinsic (OpenCV axes, optionally ``perturb``-ed) and where K puts the red sphere."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T_world_body = np.eye(4)
    T_world_body[:3, :3], T_world_body[:3, 3] = data.xmat[1].reshape(3, 3), data.xpos[1]
    T_body_cam = np.linalg.inv(T_world_body) @ look_at(c, [1.0, 0.0, 0.5])
    if perturb is not None:
        T_body_cam = perturb(T_body_cam)
    T_world_cam = T_world_body @ T_body_cam
    p = T_world_cam[:3, :3].T @ (np.array([1.0, 0.15, 0.6]) - T_world_cam[:3, 3])
    return T_body_cam, (K @ (p / p[2]))[:2]


def check_sphere(name, rgb, want, tol):
    red = (rgb[..., 0] > 120) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    got = np.argwhere(red).mean(0)[::-1]
    print(f"{name}: sphere centre at {got}, cv2 projection {want}")
    assert red.sum() > 20 and np.abs(got - want).max() < tol


def test_set_extrinsics_projects_like_opencv():
    """A calibrated optical-frame extrinsic + K put a world point where cv2-style projection says (MuJoCo, EGL)."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    T_body_cam, want = calibrated_pose(model)
    set_extrinsics(model, "c", T_body_cam)
    mujoco.mj_forward(model, data)
    cam = SimCamera(model, "c", CameraConfig.preset("ideal", 320, 240, K=K))
    check_sphere("mujoco", cam.render(data), want, 0.5)  # measured 0.08 px

    # the calibrated preset (noise, AE, depth artefacts on) renders through the same path
    cam = SimCamera(model, "c", CameraConfig.preset("calibrated", 320, 240, K=K, dist=(-0.05, 0.01, 0, 0)))
    rgb, z = cam.render(data, depth=True)
    assert rgb.shape == (240, 320, 3) and z.shape == (240, 320)

    with pytest.raises(ValueError):
        set_extrinsics(model, "c", np.eye(3))
    with pytest.raises(ValueError):
        set_extrinsics(model, "nope", np.eye(4))
    model.cam_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
    with pytest.raises(ValueError):
        set_extrinsics(model, "c", np.eye(4))


def test_perturb_extrinsics():
    """Mount draws stay inside the bounds, use them, leave the input alone, and render where they say (MuJoCo, EGL)."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    T = calibrated_pose(model)[0]
    T0, rng = T.copy(), np.random.default_rng(0)
    D = [np.linalg.inv(T) @ perturb_extrinsics(T, rng, rot_deg=0.5, trans_m=0.003) for _ in range(300)]
    angle = np.array([np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)) for d in D])
    offset = np.array([np.abs(d[:3, 3]).max() for d in D])
    assert angle.max() <= np.radians(0.5) + 1e-9 and offset.max() <= 0.003 + 1e-12
    assert angle.max() > np.radians(0.4) and offset.max() > 0.0027  # the spread is used
    assert np.array_equal(T, T0) and np.allclose(perturb_extrinsics(T, rng, 0, 0), T, atol=1e-12)
    for bad in (dict(T_body_cam=np.eye(3)), dict(rot_deg=-1.0), dict(trans_m=-1.0)):
        with pytest.raises(ValueError):
            perturb_extrinsics(**{"T_body_cam": T, "rng": rng, **bad})

    # a large draw (2 deg, 2 cm) moves the sphere by pixels, and the render follows the drawn pose
    P, want = calibrated_pose(model, perturb=lambda T: perturb_extrinsics(T, rng, rot_deg=2.0, trans_m=0.02))
    print(f"perturbed projection {want} vs calibrated {calibrated_pose(model)[1]}")
    set_extrinsics(model, "c", P)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = SimCamera(model, "c", CameraConfig.preset("ideal", 320, 240, K=K))
    check_sphere("perturbed", cam.render(data), want, 0.5)


def test_genesis_extrinsics():
    """GenesisCamera(extrinsics=...) and set_extrinsics render from the given mount pose (needs genesis, GPU)."""
    import types

    gs = pytest.importorskip("genesis")
    from test_genesis_bridge import ambient_only, genesis_xml

    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    T_body_cam, want = calibrated_pose(mujoco.MjModel.from_xml_string(SCENE))
    cam = GenesisCamera(genesis_xml(gs, SCENE), "c", CameraConfig.preset("ideal", 320, 240, K=K), 0.002,
                        vis_options=ambient_only(gs, 1.0), extrinsics=T_body_cam)
    data = types.SimpleNamespace(time=0.0, qpos=cam.entity.get_qpos())
    check_sphere("genesis", cam.render(data), want, 1.0)
    T2, want2 = calibrated_pose(mujoco.MjModel.from_xml_string(SCENE), c=(0.0, 0.05, 1.0))  # remount 5 cm aside
    cam.set_extrinsics(T2)
    check_sphere("genesis remounted", cam.render(data), want2, 1.0)
