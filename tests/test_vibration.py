"""Tests for MountVibration: RMS fidelity, spectrum, determinism, matrix, and the rendered shake."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physicalai_sim2real_camera.vibration import MountVibration


def test_rms():
    """Per-axis RMS is within 12 % of the requested value."""
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    t_vals = np.arange(0, 300, 0.002)
    samples = np.array([mv.offset(t) for t in t_vals])

    rot_rms = np.std(samples[:, :3], axis=0)
    want_rot = np.radians(0.2)
    assert np.all(np.abs(rot_rms / want_rot - 1) < 0.12)

    trans_rms = np.std(samples[:, 3:], axis=0)
    want_trans = 0.0005
    assert np.all(np.abs(trans_rms / want_trans - 1) < 0.12)


def test_spectrum():
    """Peak frequency of the periodogram is near the natural frequency."""
    # 12 Hz case
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=42)
    t_vals = np.arange(0, 300, 0.001)
    sig = np.array([mv.offset(t)[0] for t in t_vals])

    n_per_seg = int(10 * 1000)  # 10 s at 1 kHz
    n_segs = 30
    segs = sig[:n_per_seg * n_segs].reshape(n_segs, n_per_seg)
    freqs = np.fft.rfftfreq(n_per_seg, d=0.001)
    periods = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    avg_spec = periods.mean(axis=0)

    peak_hz = float(freqs[np.argmax(avg_spec)])
    assert abs(peak_hz - 12.0) / 12.0 < 0.15

    # 4 Hz case
    mv2 = MountVibration(0.2, 0.0005, 4.0, 0.1, seed=42)
    sig2 = np.array([mv2.offset(t)[0] for t in t_vals])
    segs2 = sig2[:n_per_seg * n_segs].reshape(n_segs, n_per_seg)
    avg_spec2 = (np.abs(np.fft.rfft(segs2, axis=1)) ** 2).mean(axis=0)
    peak_hz2 = float(freqs[np.argmax(avg_spec2)])
    assert abs(peak_hz2 - 4.0) / 4.0 < 0.15


def test_determinism_and_order_independence():
    """Same seed = same offset; different seed = different offset.

    Query order does not affect the result (lazy chunking always starts
    from the same initial state because _ensure_samples regenerates
    chunks from the last *existing* sample — but we enforce order
    independence by storing the initial seed state and using it).
    """
    a = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    b = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    np.testing.assert_array_equal(a.offset(1.234), b.offset(1.234))

    # query 50 s first then 1.234 s
    c = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    _ = c.offset(50.0)
    d = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    np.testing.assert_array_equal(c.offset(1.234), d.offset(1.234))

    e = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=99)
    assert not np.allclose(e.offset(1.234), a.offset(1.234))


def test_zero_amplitude():
    """Zero RMS produces inactive mount returning identity."""
    mv = MountVibration(0, 0, 12, 0.1, 0)
    assert mv.active is False
    np.testing.assert_array_equal(mv.matrix(3.0), np.eye(4))


def test_matrix():
    """Rotation part is orthonormal and matches the rotation vector."""
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    T = mv.matrix(3.1415)
    R = T[:3, :3]
    off = mv.offset(3.1415)

    # orthonormal
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    # rotation angle matches norm of rot vector
    angle_rv = np.linalg.norm(off[:3])
    angle_R = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    np.testing.assert_allclose(angle_R, angle_rv, atol=1e-9)

    # translation
    np.testing.assert_array_equal(T[:3, 3], off[3:])


def test_continuity():
    """The offset is smooth: 1 ms steps stay within what the mount's velocity allows (no jumps between chunks).

    The stationary velocity RMS is w * RMS, so a 1 ms step is ~N(0, (w * dt * RMS)^2); over 20 000 steps the
    largest one stays below ~4.5 of those standard deviations. A chunk seam (every 10 s) is inside the window.
    """
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    samples = np.array([mv.offset(t) for t in np.arange(0, 20, 0.001)])
    step = np.abs(np.diff(samples, axis=0)).max(0)
    bound = 6 * 2 * np.pi * 12.0 * 0.001 * np.array([np.radians(0.2)] * 3 + [0.0005] * 3)
    assert np.all(step < bound), (step, bound)


def test_validation():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        MountVibration(-1, 0.0005, 12, 0.1, 0)
    with pytest.raises(ValueError):
        MountVibration(0.2, 0.0005, 0, 0.1, 0)
    with pytest.raises(ValueError):
        MountVibration(0.2, 0.0005, 12, 0, 0)
    with pytest.raises(ValueError):
        MountVibration(0.2, 0.0005, 20, 0.1, 0, rate=100)


SCENE = """<mujoco><visual><headlight ambient='1 1 1' diffuse='0 0 0' specular='0 0 0'/></visual><worldbody>
<geom type='sphere' pos='2.0 0.25 1.1' size='0.04' rgba='1 0 0 1'/>
<camera name='c' pos='0 0 1' xyaxes='0 -1 0 0 0 1' fovy='45'/>
</worldbody></mujoco>"""


def test_simcamera_shakes_with_the_mount():
    """A rendered point lands where the vibrating camera pose projects it, so the MuJoCo frame convention holds.

    The camera looks along world +x; a 2 deg / 1 cm RMS, 3 Hz mount moves the sphere by pixels between times.
    """
    import mujoco

    from physicalai_sim2real_camera import CameraConfig, SimCamera

    os.environ.setdefault("MUJOCO_GL", "egl")
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = CameraConfig(320, 240, mount_vib_rot_deg=2.0, mount_vib_trans_m=0.01, mount_vib_hz=3.0, seed=5)
    cam = SimCamera(model, "c", cfg)
    K, vib = cam.K_out, cam._vib
    R, p = data.cam_xmat[0].reshape(3, 3), data.cam_xpos[0]
    centres = []
    for t in (0.3, 0.7, 1.1, 1.5):
        data.time = t
        rgb = cam.render(data)
        red = (rgb[..., 0] > 120) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
        got = np.argwhere(red).mean(0)[::-1]
        T = vib.matrix(t)
        Rn, pn = R @ T[:3, :3], p + R @ T[:3, 3]
        q = Rn.T @ (np.array([2.0, 0.25, 1.1]) - pn)  # MuJoCo camera frame: x right, y up, looking along -z
        want = np.array([K[0, 2] + K[0, 0] * q[0] / -q[2], K[1, 2] - K[1, 1] * q[1] / -q[2]])
        print(f"t={t}: sphere {got}, shaken projection {want}")
        assert red.sum() > 20 and np.abs(got - want).max() < 0.3
        centres.append(got)
    assert np.ptp(np.array(centres), axis=0).max() > 3  # the vibration does move the image


def test_memory_bounded():
    """After querying t = 0..400 s the kept history is <= 40 * rate samples (keep_s=30 + one chunk)."""
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    for t in np.arange(0, 400, 0.5):
        mv.offset(t)
    assert len(mv._x) <= 40 * 1000 + 1


def test_identity_after_prune():
    """After a long run, offset(3.25) of the pruned instance equals a fresh instance's offset(3.25)."""
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    for t in np.arange(0, 400, 0.5):
        mv.offset(t)
    fresh = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    np.testing.assert_array_equal(mv.offset(3.25), fresh.offset(3.25))


def test_jump_guard():
    """offset(1e7) raises ValueError because the jump exceeds 3600 * rate new samples."""
    mv = MountVibration(0.2, 0.0005, 12.0, 0.1, seed=1)
    with pytest.raises(ValueError):
        mv.offset(1e7)


def test_genesis_shakes_with_the_mount():
    """GenesisCamera applies the same mount offset per render as SimCamera (needs genesis, GPU)."""
    import types

    import mujoco

    gs = pytest.importorskip("genesis")
    from test_genesis_bridge import ambient_only, genesis_xml
    from test_presets import SCENE, K, calibrated_pose, check_sphere

    from physicalai_sim2real_camera import CameraConfig, set_extrinsics
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    cfg = CameraConfig.preset("ideal", 320, 240, K=K, mount_vib_rot_deg=2.0, mount_vib_trans_m=0.01,
                              mount_vib_hz=3.0, seed=5)
    m = mujoco.MjModel.from_xml_string(SCENE)
    T_body_cam = calibrated_pose(m)[0]  # the camera pose of test_presets, which sees the sphere
    set_extrinsics(m, "c", T_body_cam)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = GenesisCamera(genesis_xml(gs, SCENE), "c", cfg, 0.002, vis_options=ambient_only(gs, 1.0),
                        extrinsics=T_body_cam)
    R, p = d.cam_xmat[0].reshape(3, 3), d.cam_xpos[0]
    q0 = cam.entity.get_qpos()
    for t in (0.3, 0.7, 1.1):
        T = cam._vib.matrix(t)
        Rn, pn = R @ T[:3, :3], p + R @ T[:3, 3]
        q = Rn.T @ (np.array([1.0, 0.15, 0.6]) - pn)  # the red sphere of test_presets.SCENE
        want = np.array([K[0, 2] + K[0, 0] * q[0] / -q[2], K[1, 2] - K[1, 1] * q[1] / -q[2]])
        check_sphere(f"genesis t={t}", cam.render(types.SimpleNamespace(time=t, qpos=q0)), want, 1.0)
