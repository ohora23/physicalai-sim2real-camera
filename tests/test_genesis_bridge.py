"""Unit tests for physicalai_sim2real_camera.genesis_bridge (pure helpers: camera_mount, interp_qpos)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np

from physicalai_sim2real_camera.genesis_bridge import camera_mount, interp_qpos, quat_starts

MJCF = """<mujoco><worldbody>
<body name='base' pos='0 0 1'>
  <body name='head' pos='0.1 0 0.2' euler='0 0.3 0.2'>
    <joint name='j' type='hinge' axis='0 1 0'/>
    <geom type='sphere' size='0.05'/>
    <body name='cam_link' pos='0.02 0.01 0.05' euler='0.1 0 0'>
      <geom type='box' size='0.01 0.01 0.01'/>
      <camera name='c' pos='0.01 0 0.02' euler='0 -1.2 0.4'/>
    </body>
  </body>
</body>
</worldbody></mujoco>"""


def test_camera_mount_pose():
    model = mujoco.MjModel.from_xml_string(MJCF)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)

    for link_names, expected_body in [(['base', 'head'], 'head'),
                                       (['base', 'head', 'cam_link'], 'cam_link')]:
        name, offset = camera_mount(model, 'c', link_names)
        assert name == expected_body, (name, expected_body)

        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, expected_body))
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'c')
        T_link = np.eye(4)
        T_link[:3, :3], T_link[:3, 3] = d.xmat[body_id].reshape(3, 3), d.xpos[body_id]
        T_cam = np.eye(4)
        T_cam[:3, :3], T_cam[:3, 3] = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
        result = T_link @ offset
        got = result[:3, :3], result[:3, 3]
        want = T_cam[:3, :3], T_cam[:3, 3]
        np.testing.assert_array_almost_equal(got[0], want[0], decimal=9)
        np.testing.assert_array_almost_equal(got[1], want[1], decimal=9)


def test_camera_mount_errors():
    model = mujoco.MjModel.from_xml_string(MJCF)

    try:
        camera_mount(model, 'nope', ['base', 'head'])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown camera must raise")

    try:
        camera_mount(model, 'c', ['other'])
    except ValueError:
        pass
    else:
        raise AssertionError("no kept ancestor must raise")


def test_interp_qpos_numpy():
    ring = [(0.0, np.array([0.0, 0.0])),
            (0.01, np.array([1.0, 2.0])),
            (0.02, np.array([3.0, 2.0]))]

    assert np.allclose(interp_qpos(ring, 0.005), [0.5, 1.0], atol=1e-12)
    assert np.allclose(interp_qpos(ring, 0.015), [2.0, 2.0], atol=1e-12)
    assert np.allclose(interp_qpos(ring, 0.01), [1.0, 2.0], atol=1e-12)
    assert np.allclose(interp_qpos(ring, -1.0), [0.0, 0.0], atol=1e-12)
    assert np.allclose(interp_qpos(ring, 5.0), [3.0, 2.0], atol=1e-12)


def test_interp_qpos_torch():
    try:
        import torch
    except ImportError:
        return

    ring = [(0.0, torch.tensor([0.0, 0.0], dtype=torch.float64)),
            (0.01, torch.tensor([1.0, 2.0], dtype=torch.float64)),
            (0.02, torch.tensor([3.0, 2.0], dtype=torch.float64))]

    result = interp_qpos(ring, 0.005)
    assert torch.allclose(result, torch.tensor([0.5, 1.0], dtype=torch.float64), atol=1e-12)


TREE = """<mujoco><worldbody>
<body name="a"><freejoint/><geom size=".1"/><body name="b" pos="0 0 .3"><joint type="ball"/><geom size=".05"/>
<body name="c" pos="0 0 .2"><joint type="hinge" axis="0 1 0"/><geom size=".05"/></body></body></body>
</worldbody></mujoco>"""


def test_quat_starts():
    assert quat_starts(mujoco.MjModel.from_xml_string(TREE)) == (3, 7)  # free: after its 3 positions; ball: at start
    assert quat_starts(mujoco.MjModel.from_xml_string(MJCF)) == ()  # hinge only


def test_interp_qpos_matches_stateRing_free_ball():
    """interp_qpos == StateRing.load's mj_differentiatePos + mj_integratePos on random free/ball/hinge pairs."""
    m = mujoco.MjModel.from_xml_string(TREE)
    rng = np.random.default_rng(0)
    quats = quat_starts(m)
    worst = 0.0
    for _ in range(200):
        q0, q1 = np.zeros(m.nq), np.zeros(m.nq)
        for q in (q0, q1):
            q[:3], q[11] = rng.normal(size=3), rng.normal()
            for i in quats:
                v = rng.normal(size=4)
                q[i:i + 4] = v / np.linalg.norm(v)
        a = rng.uniform()
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, q0, q1)
        want = q0.copy()
        mujoco.mj_integratePos(m, want, dq, a)
        got = interp_qpos([(0.0, q0), (1.0, q1)], a, quats)
        for i in quats:  # q and -q are the same rotation
            if np.dot(got[i:i + 4], want[i:i + 4]) < 0:
                got[i:i + 4] *= -1
        worst = max(worst, np.abs(got - want).max())
    print(f"interp_qpos vs mj_integratePos, 200 random free/ball/hinge pairs: max |diff| {worst:.1e}")
    assert worst < 1e-9


def test_interp_qpos_torch_quaternion():
    try:
        import torch
    except ImportError:
        return
    q0 = torch.tensor([0.0, 0, 0, 1, 0, 0, 0], dtype=torch.float64)
    q1 = torch.tensor([1.0, 0, 0, 0, 0, 0, 1], dtype=torch.float64)  # 180 deg about z
    got = interp_qpos([(0.0, q0), (1.0, q1)], 0.5, (3,))
    h = math.sqrt(0.5)
    assert torch.allclose(got, torch.tensor([0.5, 0, 0, h, 0, 0, h], dtype=torch.float64), atol=1e-12)  # 90 deg


RIG = """<mujoco><worldbody>
<geom name='red' type='box' pos='1.0 0.1 0.05' size='0.05 0.05 0.05' rgba='1 0 0 1'/>
<body name='rig'><joint type='hinge' axis='0 0 1'/><geom type='sphere' size='0.01' pos='0 0 -0.5'/>
  <camera name='c' xyaxes='0 -1 0 0 0 1' fovy='45'/></body>
</worldbody></mujoco>"""


DIST = (-0.05, 0.01, 0.0005, -0.0005, 0.0)


def red_mask(rgb):
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 150)


def box_silhouette():
    """Oracle: the RIG box's 8 corners projected with K and DIST, convex hull filled at 1/16 px (320 x 240).

    OpenCV camera coordinates (x right, y down, z forward) for the rig camera looking along world +x.
    """
    import cv2

    from physicalai_sim2real_camera.config import intrinsics_from_model

    K = intrinsics_from_model(mujoco.MjModel.from_xml_string(RIG), 0, 320, 240)
    corners = np.array([[-(0.1 + sy), -(0.05 + sz), 1.0 + sx] for sx in (-0.05, 0.05) for sy in (-0.05, 0.05)
                        for sz in (-0.05, 0.05)])
    px = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), K, np.array(DIST))[0].reshape(-1, 2)
    hull = np.zeros((240, 320), np.uint8)
    cv2.fillPoly(hull, [np.round(cv2.convexHull(px.astype(np.float32)).reshape(-1, 2) * 16).astype(np.int32)], 1,
                 shift=4)
    return hull > 0


def check_silhouette(name, rgb, tol):
    want, got = box_silhouette(), red_mask(rgb)
    c, w = np.argwhere(got).mean(0)[::-1], np.argwhere(want).mean(0)[::-1]
    err = np.linalg.norm(c - w)
    print(f"{name}: red centroid {c.round(2)} vs projected silhouette {w.round(2)}: {err:.2f} px")
    assert err < tol, (name, c, w)
    assert (got & want).sum() / (got | want).sum() > 0.9
    return got


def genesis_xml(gs, text):
    """Write an MJCF to a temp file and ``gs.init`` once per process (Genesis tests only)."""
    import tempfile

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    path = os.path.join(tempfile.mkdtemp(), "model.xml")
    with open(path, "w") as f:
        f.write(text)
    return path


def ambient_only(gs, level):
    """Rasterizer lighting with ambient only. A zero-intensity light, not lights=[]: Genesis' rasterizer corrupts the
    heap with an empty light list on these rigs ("malloc(): corrupted top size")."""
    return gs.options.VisOptions(
        lights=[gs.options.vis.DirectionalLight(dir=(0, 0, -1), color=(1, 1, 1), intensity=0.0)],
        ambient_light=(level,) * 3)


def test_simcamera_projection_oracle():
    """The oracle itself, on MuJoCo SimCamera: the distorted box lands on the projected silhouette."""
    from physicalai_sim2real_camera import CameraConfig, SimCamera

    m = mujoco.MjModel.from_xml_string(RIG)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    check_silhouette("mujoco", SimCamera(m, "c", CameraConfig(320, 240, dist=DIST)).render(d), 0.5)  # measured 0.17 px


def test_genesis_camera_projection():
    """GenesisCamera output (distortion + window) puts the box where K and dist project it (needs genesis, GPU)."""
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.depth import zfar_of
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, RIG)
    masks = {}
    for name, extra in (("gs", {}), ("window", dict(exposure_s=0.008, readout_s=0.02, blur_samples=4, rs_bands=4))):
        cam = GenesisCamera(xml, "c", CameraConfig(320, 240, dist=DIST, latency_s=0.05, **extra), 0.002)
        q = cam.entity.get_qpos()
        for k in range(12):  # static rig: the window ring holds the same pose
            cam.push(types.SimpleNamespace(time=k * 0.002, qpos=q))
        rgb, z = cam.render(types.SimpleNamespace(time=11 * 0.002, qpos=q), depth=True)
        # measured: Genesis 0.70 px up-left of the oracle here (MuJoCo 0.17 px, test above). On ALLEX's checkerboard
        # (shading-independent corners) Genesis sits (-0.21, -0.12) px from MuJoCo, uniformly: a small rasteriser
        # offset plus mask-edge shading under Genesis' default light. Tolerance 1.0 px.
        masks[name] = check_silhouette(name, rgb, 1.0)
        bg = z[~masks[name]]
        zfar = zfar_of(mujoco.MjModel.from_xml_string(RIG))
        print(f"{name}: background depth {np.median(bg):.6f} m, zfar_of {zfar:.6f} m")
        assert np.median(bg) == np.float32(zfar)  # snapped: the pipeline's background test (z >= zfar(1-1e-6)) holds
    a, b = masks.values()
    assert (a & b).sum() / (a | b).sum() > 0.98  # static scene: window output == global-shutter geometry


def red_centroid(rgb):
    return np.argwhere(red_mask(rgb)).mean(0)[::-1]


def test_genesis_window_time_base():
    """A panning rig: the motion-blurred frame is centred on the pose at capture_t = time - half (needs genesis, GPU).

    Exposure 20 ms, pan 1 rad/s: the blur is symmetric about mid-exposure, so its centroid equals a global-shutter
    render at q(capture_t); the pose at the frame's end time (10 ms later, ~3 px away) must not match.
    """
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, RIG)
    dt, w, t_end = 0.002, 1.0, 0.04
    win = GenesisCamera(xml, "c", CameraConfig(320, 240, exposure_s=0.02, blur_samples=10, latency_s=0.05), dt)
    gsc = GenesisCamera(xml, "c", CameraConfig(320, 240, latency_s=0.05), dt)
    q0 = win.entity.get_qpos()
    for k in range(int(round(t_end / dt)) + 1):
        win.push(types.SimpleNamespace(time=k * dt, qpos=q0 + w * k * dt))
    blurred = red_centroid(win.render(types.SimpleNamespace(time=t_end, qpos=q0 + w * t_end)))
    tc = t_end - win.half
    sharp = {}
    for t in (tc, t_end):
        gsc.push(types.SimpleNamespace(time=t, qpos=q0 + w * t))  # earlier time after a later one: ring resets
        sharp[t] = red_centroid(gsc.render(types.SimpleNamespace(time=t, qpos=q0 + w * t)))
        gsc._ring.clear()
    print(f"window centroid {blurred.round(2)}, q(capture_t) {sharp[tc].round(2)}, q(end) {sharp[t_end].round(2)}")
    assert np.linalg.norm(sharp[t_end] - sharp[tc]) > 2.0  # the check can tell the two time bases apart
    assert np.linalg.norm(blurred - sharp[tc]) < 0.5


FREE_RIG = """<mujoco><worldbody>
<geom name='red' type='box' pos='1.0 0.1 0.05' size='0.05 0.05 0.05' rgba='1 0 0 1'/>
<body name='rig'><freejoint/><geom type='sphere' size='0.01' pos='0 0 -0.5'/>
  <camera name='c' xyaxes='0 -1 0 0 0 1' fovy='45'/></body>
</worldbody></mujoco>"""


def test_genesis_free_joint_time_base():
    """Floating base: yawing free-joint rig, exposure only -> blurred centroid == pose at capture_t (genesis, GPU)."""
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, FREE_RIG)
    dt, w, t_end = 0.002, 1.0, 0.04

    def q(t):  # free joint: position 0, yaw w*t about z (w, x, y, z)
        return np.array([0, 0, 0, math.cos(w * t / 2), 0, 0, math.sin(w * t / 2)])

    win = GenesisCamera(xml, "c", CameraConfig(320, 240, exposure_s=0.02, blur_samples=10, latency_s=0.05), dt)
    gsc = GenesisCamera(xml, "c", CameraConfig(320, 240, latency_s=0.05), dt)
    assert win.quats == (3,)
    for k in range(int(round(t_end / dt)) + 1):
        win.push(types.SimpleNamespace(time=k * dt, qpos=q(k * dt)))
    blurred = red_centroid(win.render(types.SimpleNamespace(time=t_end, qpos=q(t_end))))
    tc = t_end - win.half
    sharp = {t: red_centroid(gsc.render(types.SimpleNamespace(time=t, qpos=q(t)))) for t in (tc, t_end)}
    print(f"free rig: window centroid {blurred.round(2)}, q(capture_t) {sharp[tc].round(2)}, "
          f"q(end) {sharp[t_end].round(2)}")
    assert np.linalg.norm(sharp[t_end] - sharp[tc]) > 2.0
    assert np.linalg.norm(blurred - sharp[tc]) < 0.5


def test_genesis_batched_equals_single():
    """n_envs = 4: env i of one batched GenesisCamera == a single GenesisCamera with cfg.seed + i (genesis, GPU).

    Window mode, depth and noise on (the noise makes the per-env seed rule visible); each env yaws at its own rate.
    """
    import dataclasses
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, FREE_RIG)
    n, dt, t_end = 4, 0.002, 0.04
    cfg = CameraConfig(320, 240, dist=DIST, exposure_s=0.008, readout_s=0.02, blur_samples=4, rs_bands=4,
                       sigma_shot=0.02, sigma_read=0.004, sigma_prnu=0.01, sigma_dsnu=0.003, latency_s=0.05)

    def q(t):  # (n, 7): env i yaws at 0.5 * (i + 1) rad/s
        w = 0.5 * (np.arange(n) + 1) * t
        return np.stack([np.zeros(n), np.zeros(n), np.zeros(n), np.cos(w / 2), np.zeros(n), np.zeros(n),
                         np.sin(w / 2)], axis=1).astype(np.float32)

    batched = GenesisCamera(xml, "c", cfg, dt, n_envs=n)
    singles = [GenesisCamera(xml, "c", dataclasses.replace(cfg, seed=cfg.seed + i), dt) for i in range(n)]
    for k in range(int(round(t_end / dt)) + 1):
        batched.push(types.SimpleNamespace(time=k * dt, qpos=q(k * dt)))
        for i, c in enumerate(singles):
            c.push(types.SimpleNamespace(time=k * dt, qpos=q(k * dt)[i]))
    rgb, z = batched.render(types.SimpleNamespace(time=t_end, qpos=q(t_end)), depth=True)
    assert rgb.shape == (n, 240, 320, 3) and z.shape == (n, 240, 320)
    for i, c in enumerate(singles):
        rgb_i, z_i = c.render(types.SimpleNamespace(time=t_end, qpos=q(t_end)[i]), depth=True)
        d = np.abs(rgb[i].astype(int) - rgb_i.astype(int))
        print(f"env {i}: |batched - single| rgb max {d.max()} DN mean {d.mean():.4f}, depth max "
              f"{np.abs(z[i] - z_i).max():.2e} m")
        assert np.array_equal(rgb[i], rgb_i) and np.array_equal(z[i], z_i)
    assert not np.array_equal(rgb[0], rgb[1])  # the envs really differ (pose and noise)


def test_genesis_headlight_follows_camera():
    """The headlight brightens what the camera faces, from any heading; batched scenes refuse it (genesis, GPU)."""
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, FREE_RIG.replace("<geom name='red'", "<geom name='red2' type='box' pos='0 -1.0 0.05' "
                                           "size='0.05 0.05 0.05' rgba='1 0 0 1'/><geom name='red'"))  # +90 deg box
    dark = ambient_only(gs, 0.1)  # only the headlight lights the box
    cfg = CameraConfig(320, 240, latency_s=0.05)
    lit = GenesisCamera(xml, "c", cfg, 0.002, vis_options=dark, headlight=2.0)
    off = GenesisCamera(xml, "c", cfg, 0.002, vis_options=dark)
    fixed = GenesisCamera(xml, "c", cfg, 0.002, vis_options=gs.options.VisOptions(  # the same light, fixed along +x
        lights=[gs.options.vis.DirectionalLight(dir=(1, 0, 0), color=(1, 1, 1), intensity=2.0)],
        ambient_light=(0.1, 0.1, 0.1)))
    level = {}
    for heading, yaw in (("+x", 0.0), ("-y", -math.pi / 2)):  # face the first box, then the second
        q = np.array([0, 0, 0, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], np.float32)
        for name, cam in (("headlight", lit), ("none", off), ("fixed", fixed)):
            rgb = cam.render(types.SimpleNamespace(time=0.0, qpos=q))
            m = red_mask(rgb) | (rgb[..., 0] > 20)  # the box, however dim
            level[heading, name] = float(rgb[..., 0][m].mean()) if m.any() else 0.0
    print("box red level:", {f"{h}/{n}": round(v, 1) for (h, n), v in level.items()})
    for heading in ("+x", "-y"):
        assert level[heading, "headlight"] > level[heading, "none"] + 40  # the headlight lights the faced box
    assert abs(level["+x", "headlight"] / level["-y", "headlight"] - 1) < 0.05  # it follows the camera round
    assert abs(level["+x", "fixed"] / level["+x", "headlight"] - 1) < 0.05  # facing +x, fixed == headlight
    assert level["-y", "fixed"] < level["-y", "headlight"] - 40  # facing -y, the fixed light misses the faced face
    with pytest.raises(ValueError):
        GenesisCamera(xml, "c", cfg, 0.002, n_envs=2, headlight=1.0)


def test_genesis_madrona_path():
    """renderer="madrona" (BatchRenderer, frames stay on the GPU): the box lands on the analytic silhouette, and
    batched envs == single cameras (genesis + gs_madrona, GPU)."""
    import dataclasses
    import types

    import pytest

    gs = pytest.importorskip("genesis")
    pytest.importorskip("gs_madrona")
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = genesis_xml(gs, FREE_RIG)  # at identity the free rig is RIG's camera: box_silhouette() applies
    # Madrona has no ambient light: six shadowless directional lights light every face about evenly
    lights = [dict(pos=tuple(-10.0 * float(v) * sg for v in ax), dir=tuple(float(v) * sg for v in ax),
                   directional=True, castshadow=False, intensity=0.5) for ax in np.eye(3) for sg in (1, -1)]
    dt, t_end = 0.002, 0.04
    cam = GenesisCamera(xml, "c", CameraConfig(320, 240, dist=DIST, latency_s=0.05), dt, renderer="madrona",
                        madrona_lights=lights)
    rgb = cam.render(types.SimpleNamespace(time=0.0, qpos=np.array([0, 0, 0, 1, 0, 0, 0], np.float32)))
    # measured 0.44 px (rasterizer path 0.70 px, MuJoCo 0.17 px); same 1 px tolerance as the rasterizer test
    check_silhouette("madrona", rgb, 1.0)

    def q(t, i=0):  # free joint yawing at 0.5 * (i + 1) rad/s
        w = 0.5 * (i + 1) * t
        return np.array([0, 0, 0, math.cos(w / 2), 0, 0, math.sin(w / 2)], np.float32)

    def run(c, qq):
        for k in range(int(round(t_end / dt)) + 1):
            c.push(types.SimpleNamespace(time=k * dt, qpos=qq(k * dt)))
        return c.render(types.SimpleNamespace(time=t_end, qpos=qq(t_end)))

    cfg = CameraConfig(320, 240, dist=DIST, exposure_s=0.008, readout_s=0.02, blur_samples=4, rs_bands=4,
                       sigma_shot=0.02, sigma_read=0.004, latency_s=0.05)
    n = 2
    batched = run(GenesisCamera(xml, "c", cfg, dt, n_envs=n, renderer="madrona", madrona_lights=lights),
                  lambda t: np.stack([q(t, i) for i in range(n)]))
    for i in range(n):
        single = run(GenesisCamera(xml, "c", dataclasses.replace(cfg, seed=cfg.seed + i), dt, renderer="madrona",
                                   madrona_lights=lights), lambda t, i=i: q(t, i))
        diff = np.abs(batched[i].astype(int) - single.astype(int))
        print(f"madrona env {i}: |batched - single| max {diff.max()} DN mean {diff.mean():.4f}")
        assert diff.max() == 0
    with pytest.raises(ValueError):
        GenesisCamera(xml, "c", cfg, dt, renderer="madrona", headlight=1.0)

if __name__ == "__main__":
    test_camera_mount_pose()
    test_camera_mount_errors()
    test_interp_qpos_numpy()
    test_interp_qpos_torch()
    test_quat_starts()
    test_interp_qpos_matches_stateRing_free_ball()
    test_interp_qpos_torch_quaternion()
    test_simcamera_projection_oracle()
    import importlib.util

    if importlib.util.find_spec("genesis"):  # GPU + Genesis venv only (pytest skips them elsewhere)
        test_genesis_camera_projection()
        test_genesis_window_time_base()
        test_genesis_free_joint_time_base()
        test_genesis_batched_equals_single()
        test_genesis_headlight_follows_camera()
        test_genesis_madrona_path()
    print("all genesis_bridge tests passed")
