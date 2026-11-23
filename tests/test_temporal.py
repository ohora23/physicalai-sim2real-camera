"""Acceptance tests for Phase 2 P0: StateRing, render_at, window-delayed capture."""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np
from test_camera import BX, BY, CAM_Z, NX, NY, SQ, TOP, H, W, board_scene

from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera, intrinsics_from_model
from physicalai_sim2real_camera.camera import StateRing

DT = 0.002


def test_interp():
    rng = np.random.default_rng(0)
    # hinge: qpos = sin(2*pi*2*t)
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>')
    d, sc = mujoco.MjData(m), mujoco.MjData(m)
    ring = StateRing(m, 0.5)  # holds >= 1 s at 500 Hz
    f = lambda t: np.sin(2 * np.pi * 2 * t)
    for k in range(501):
        d.time = k * DT
        d.qpos[0] = f(d.time)
        ring.push(d)
    ts = rng.uniform(0, 1, 1000)
    err = 0.0
    for t in ts:
        ring.load(t, sc)
        err = max(err, abs(sc.qpos[0] - f(t)))
    print(f"hinge max interp err {err:.2e} rad (theory ~8e-5)")
    assert err < 1e-3
    ring.load(0.1, sc)  # exactly on a sample -> verbatim
    assert sc.qpos[0] == f(50 * DT) and sc.time == 0.1

    # free joint rotating at 2 rad/s about z
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body></worldbody></mujoco>')
    d, sc = mujoco.MjData(m), mujoco.MjData(m)
    ring = StateRing(m, 0.5)
    quat = lambda t: np.array([np.cos(t), 0, 0, np.sin(t)])  # angle 2t
    for k in range(501):
        d.time = k * DT
        d.qpos[3:7] = quat(d.time)
        ring.push(d)
    aerr = nerr = 0.0
    for t in ts:
        ring.load(t, sc)
        q = sc.qpos[3:7]
        nerr = max(nerr, abs(np.linalg.norm(q) - 1))
        aerr = max(aerr, 2 * np.arccos(min(1.0, abs(q @ quat(t)))))
    print(f"free max angle err {aerr:.2e} rad, |quat|-1 {nerr:.1e}")
    assert aerr < 1e-3 and nerr < 1e-9


def moving_scene():
    """board_scene with the board on a mocap body and the camera on a slide joint."""
    boxes = "".join(
        f'<geom type="box" size="{SQ/2} {SQ/2} {TOP/2}" rgba="0 0 0 1" '
        f'pos="{(i - NX/2 + 0.5)*SQ} {(j - NY/2 + 0.5)*SQ} {TOP/2}"/>'
        for i in range(NX) for j in range(NY) if (i + j) % 2 == 0)
    xml = f"""<mujoco><option gravity="0 0 0" timestep="{DT}"/>
<visual><global offwidth="900" offheight="700"/></visual><worldbody>
  <light pos="0 0 2" directional="true" diffuse="1 1 1"/>
  <geom type="plane" size="1 1 0.01" rgba="1 1 1 1"/>
  <body name="board" mocap="true" pos="{BX} {BY} 0">{boxes}</body>
  <body pos="0.05 0 {CAM_Z}"><joint name="slide" type="slide" axis="1 0 0" damping="0"/>
    <geom size="0.001" mass="1" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    <camera name="cam" fovy="45" xyaxes="1 0 0 0 1 0"/></body>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[0] = 0.2  # camera: 0.2 m/s along x, no damping/gravity
    mujoco.mj_forward(model, data)
    return model, data


def phase1_cfg(**kw):
    return CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=0.3,
                        sigma_prnu=0.02, sigma_dsnu=0.005, black_level=0.02,
                        exposure_s=0.004, readout_s=0.004, **kw)


def advance(model, data, cam, n):
    for _ in range(n):
        data.mocap_pos[0, 1] = BY + 0.1 * data.time  # board: 0.1 m/s along y
        mujoco.mj_step(model, data)
        cam.push(data)


def test_render_at():
    model, data = moving_scene()
    cam = SimCamera(model, "cam", phase1_cfg())
    assert cam.half_window == 0.004 and cam._ring.cap == 7
    cam.render(data)  # consume the one-time cv2 reseed
    advance(model, data, cam, 20)

    # 2. live data untouched
    before = [data.time, data.qpos.copy(), data.qvel.copy(), data.xpos.copy()]
    cam.render_at(data.time - 0.003)
    assert data.time == before[0]
    for a, b in zip(before[1:], [data.qpos, data.qvel, data.xpos]):
        assert np.array_equal(a, b)

    # 3. xpos lag: without mj_forward, render(data) shows the pre-step pose
    lag = not np.array_equal(cam._render_data(data, False), cam.render_at(data.time))
    print(f"render(data) != render_at(data.time) without mj_forward: {lag}")
    if not lag:
        print("NOTE: xpos lag not visible in this frame")
    mujoco.mj_forward(model, data)
    a = cam._render_data(data, False)  # single instant (render() now integrates the window)
    b = cam.render_at(data.time)
    assert np.array_equal(a, b), np.abs(a.astype(int) - b).max()
    print("render(data) == render_at(data.time) after mj_forward: bit-identical")
    # interpolated mid-step differs from both neighbours
    c = cam.render_at(data.time - DT / 2)
    assert not np.array_equal(a, c)
    # window depth = one depth-only render at capture_t = data.time - half
    _, d = cam.render(data, depth=True)
    _, d2 = cam.render_at(data.time - cam.half_window, depth=True)
    assert np.array_equal(d, d2) and not np.array_equal(d, cam.render_at(data.time, depth=True)[1])

    # push cost (no allocation, just copies)
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        cam.push(data)
    us = (time.perf_counter() - t0) / N * 1e6
    t0 = time.perf_counter()
    for _ in range(2000):
        cam._ring.load(data.time - 0.0031, cam._scratch)
    lus = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"ring push {us:.2f} us, load(interp + mj_forward) {lus:.2f} us")
    if us > 5:
        print("WARNING: ring push > 5 us")
    # episode reset clears the ring
    data.time = 0.0
    cam.push(data)
    assert cam._ring.n == 1
    cam.close()


class Cam:
    def __init__(self):
        self.calls = self.pushes = 0

    def push(self, data):
        self.pushes += 1

    def render(self, data):
        self.calls += 1
        return self.calls


class Data:
    time = 0.0


def test_window_delay():
    c = CameraConfig(8, 8, fps=30, latency_s=0.05, exposure_s=0.012, readout_s=0.02)
    cam, d = Cam(), Data()
    s = CameraStream(cam, c)
    N = 10000
    fired, ages = {}, []
    for k in range(N):
        d.time = k / 500
        calls = cam.calls
        f, ct, n = s.step(d)
        if cam.calls != calls:
            e = s._buf[-1]
            fired[e[2]] = (k, e[0])
        if ct is not None:
            ages.append(d.time - ct)
    assert cam.pushes == N
    kc = lambda n: (50 * n + 2) // 3
    expect = [n for n in range(N) if kc(n) + 8 < N]
    assert sorted(fired) == expect, (len(fired), len(expect))
    for n, (k, ct) in fired.items():
        assert k == kc(n) + 8, (n, k)
        assert abs(ct - kc(n) / 500) < 1e-12, (n, ct)
    m, exp = float(np.mean(ages)), 0.05 + 1 / 60 - 1 / 1000
    print(f"window delay: {len(fired)} frames all at k_c+8; age mean {m:.5f} exp {exp:.5f}")
    assert abs(m - exp) <= 0.02 * exp
    try:
        CameraStream(Cam(), CameraConfig(8, 8, latency_s=0.01, exposure_s=0.012, readout_s=0.02))
    except ValueError as e:
        print(f"ValueError ok: {e}")
    else:
        raise AssertionError("latency < half must raise")


BAR_Z = 0.5  # camera height above the bar


def bar_scene(u):
    """Thin emissive white vertical bar (≈2 px wide) sliding along x at exactly u m/s, black
    background, no lights; camera looks straight down from BAR_Z (image x = world x)."""
    hw = BAR_Z / (240 / np.tan(np.radians(22.5)))  # 1 px at BAR_Z -> box half-width = 2 px wide
    xml = f"""<mujoco><option gravity="0 0 0" timestep="{DT}"/>
<visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
<asset><material name="glow" emission="1" rgba="1 1 1 1"/></asset><worldbody>
  <body><joint type="slide" axis="1 0 0" damping="0"/>
    <geom type="box" size="{hw} 0.15 0.001" material="glow" contype="0" conaffinity="0"/></body>
  <camera name="cam" pos="0 0 {BAR_Z}" fovy="45" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0] = -0.06, u
    mujoco.mj_forward(model, data)
    return model, data


def bar_cfg(**kw):
    return CameraConfig(W, H, gamma=1.0, **kw)  # dist=0, vignette/noise 0 by default


def run_bar(u, cfg, steps=40):
    """mj_step at 500 Hz pushing every step, then render(data) = window ending at data.time."""
    model, data = bar_scene(u)
    cam = SimCamera(model, "cam", cfg)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        cam.push(data)
    mujoco.mj_forward(model, data)
    img = cam.render(data).astype(np.float64).sum(axis=2)  # gamma 1 -> linear DN
    v = intrinsics_from_model(model, cam.cam, W, H)[0, 0] * u / BAR_Z  # px/s, render-independent
    cam.close()
    return img, v


def moments(row):
    x = np.arange(row.size)
    m = (row * x).sum() / row.sum()
    return m, (row * (x - m) ** 2).sum() / row.sum()


def test_blur():
    u, T = 1.0, 0.034  # v ≈ 1159 px/s -> v·T ≈ 39.4 px
    blur, v = run_bar(u, bar_cfg(exposure_s=T, blur_samples=8))
    sharp, _ = run_bar(u, bar_cfg())
    s2, s02 = moments(blur[H // 2])[1], moments(sharp[H // 2])[1]
    # N equally spaced copies (spacing L/N) of the sharp profile: variance adds that of a discrete
    # uniform over N points, (N²-1)/12·(L/N)², not the continuous box's L²/12 -> L = sqrt(12N²/(N²-1)·Δσ²)
    N = 8
    L = np.sqrt(12 * N**2 / (N**2 - 1) * (s2 - s02))
    print(f"blur: L = {L:.2f} px vs v·exposure = {v * T:.2f} px ({L / (v * T) - 1:+.1%})")
    assert abs(L - v * T) <= 0.1 * v * T


def test_rolling_shutter():
    u, R = 1.0, 0.034
    img, v = run_bar(u, bar_cfg(readout_s=R, rs_bands=8))
    rows = np.flatnonzero(img.sum(axis=1) > 0.5 * img.sum(axis=1).max())
    cx = np.array([moments(img[r])[0] for r in rows])
    slope = np.polyfit(rows, cx, 1)[0]
    # least squares on an 8-step staircase gives (B²-1)/B² = 0.984 of the continuous slope
    print(f"rolling shutter: slope·H = {slope * H:.2f} px vs v·readout = {v * R:.2f} px "
          f"({slope * H / (v * R) - 1:+.1%}); rows {rows[0]}..{rows[-1]}")
    assert slope > 0  # u > 0 and top rows earlier -> bar further left at the top
    assert abs(slope * H - v * R) <= 0.1 * v * R


def test_zero_window_identical():
    model, data = moving_scene()
    cfg = dataclasses.replace(phase1_cfg(), exposure_s=0.0, readout_s=0.0)
    cam, ref = SimCamera(model, "cam", cfg), SimCamera(model, "cam", cfg)
    assert cam._ring is None
    ref.render(data)  # consume the one-time cv2 reseed
    advance(model, data, cam, 10)  # push is a no-op without a window
    mujoco.mj_forward(model, data)
    assert np.array_equal(cam.render(data), ref.render(data))
    print("exposure_s=readout_s=0: bit-identical to Phase 1")
    cam.close(), ref.close()


def test_static_blur():
    model, data = board_scene()
    base = dataclasses.replace(phase1_cfg(), exposure_s=0.0, readout_s=0.0)
    ref = SimCamera(model, "cam", base)
    cam = SimCamera(model, "cam", dataclasses.replace(base, exposure_s=0.008, blur_samples=4))
    for k in range(10):  # static state, advancing time
        data.time = k * DT
        cam.push(data)
    a, b = ref.render(data), cam.render(data)
    dmax = np.abs(a.astype(int) - b).max()
    print(f"static scene, N=4: max |Δ| = {dmax} DN")
    assert dmax <= 1
    cam.close(), ref.close()


def test_grid_snap():
    """Grid-snap render cache: shared renders, time error <= delta/2."""
    model, _ = board_scene()
    base = phase1_cfg()
    for e, N, r, B, n_exp in [(0.008, 4, 0.016, 8, 11), (0.0, 1, 0.016, 8, 8), (0.008, 4, 0.0, 1, 4),
                              (0.006, 4, 0.016, 8, None)]:  # last: 1.5 ms vs 2 ms steps, not commensurate
        cam = SimCamera(model, "cam", dataclasses.replace(base, exposure_s=e, blur_samples=N,
                                                          readout_s=r, rs_bands=B))
        delta = min(x for x in (e / N if e else 1, r / B if r else 1))
        n = len(cam._plan)
        print(f"e={e} N={N} r={r} B={B}: {n} renders (of {N * B}), snap err "
              f"{cam._snap_err * 1e3:.3g} ms (delta/2 = {delta / 2 * 1e3:.3g} ms)")
        assert cam._snap_err <= delta / 2 + 1e-12 and n <= N * B
        assert n_exp is None or (n == n_exp and cam._snap_err < 1e-12)
        assert sum(len(u) for _, u in cam._plan) == N * B  # every (band, sample) fed exactly once
        cam.close()


def bench():
    model, data = board_scene()
    base = CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=-0.3,
                        sigma_shot=0.02, sigma_read=0.005, sigma_prnu=0.02, sigma_dsnu=0.005,
                        black_level=0.02, gamma=2.2)
    for N, B in [(1, 1), (4, 1), (1, 8), (4, 8)]:
        cfg = dataclasses.replace(base, exposure_s=0.008 if N > 1 else 0.0, blur_samples=N,
                                  readout_s=0.016 if B > 1 else 0.0, rs_bands=B)
        cam = SimCamera(model, "cam", cfg)
        for k in range(20):
            data.time = k * DT
            cam.push(data)
        for _ in range(5):
            cam.render(data)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            cam.render(data)
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"N={N} B={B}: {ms:6.2f} ms/frame, {len(getattr(cam, '_plan', [0]))} renders (N*B = {N * B}), "
              f"grid-snap time error {getattr(cam, '_snap_err', 0) * 1e3:.2g} ms")
        if ms > 17:
            print(f"WARNING: {ms:.1f} ms/frame > 17 ms budget")
        cam.close()


def test_empty_ring_single_instant_render():
    """A windowed camera with no remap renders ``data`` directly before its first push (the wrapper's reset
    frame): same frame as the zero-window camera within 1 DN (it used to finish an unwritten buffer)."""
    model, data = board_scene()
    base = dataclasses.replace(phase1_cfg(), dist=(0.0,) * 5, exposure_s=0.0, readout_s=0.0)
    ref = SimCamera(model, "cam", base)
    cam = SimCamera(model, "cam", dataclasses.replace(base, exposure_s=0.008, blur_samples=4))
    try:
        assert cam.maps is None and cam._work is not cam._lin and cam._ring.n == 0
        a, b = ref.render(data), cam.render(data)
        assert np.abs(a.astype(int) - b).max() <= 1
    finally:
        cam.close(), ref.close()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
        sys.exit()
    test_interp()
    test_render_at()
    test_window_delay()
    test_blur()
    test_rolling_shutter()
    test_zero_window_identical()
    test_static_blur()
    test_grid_snap()
    test_empty_ring_single_instant_render()
    print("all temporal tests passed")

