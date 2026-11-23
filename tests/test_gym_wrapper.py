"""RealSensorWrapper on a minimal gymnasium env: camera identity, capture timing per physics step, reset, hooks."""

import gc
import math
import sys
import weakref

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera import camera as camera_mod
from physicalai_sim2real_camera.gym_wrapper import RealSensorWrapper, _StepHook

XML = """<mujoco><option timestep="0.002" gravity="0 0 0"/><worldbody><light pos="0 0 2"/>
<geom type="plane" size="1 1 .1" rgba=".4 .5 .6 1"/>
<body name="box" pos="0 0 .1"><joint name="x" type="slide" axis="1 0 0"/>
<geom type="box" size=".05 .05 .05" rgba="1 .2 .2 1"/></body>
<camera name="cam" pos="0 -.6 .4" xyaxes="1 0 0 0 .55 .83" fovy="60"/>
<camera name="cam2" pos=".5 .6 .3" xyaxes="-1 0 0 0 -.4 .9" fovy="40"/></worldbody></mujoco>"""
CTRL_DT, N_SUB, W, H = 0.02, 10, 64, 48
THIS = sys.modules[__name__]


class SlideEnv(gym.Env):
    """A box sliding at 0.5 m/s past two fixed cameras; ``step`` runs N_SUB physics steps via this module's mujoco."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, H, W)

    def _obs(self):
        pix = {}
        for c in ("cam", "cam2"):
            self.renderer.update_scene(self.data, camera=c)
            pix[c] = self.renderer.render()
        return {"pixels": pix}

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[0] = 0.5
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        for _ in range(N_SUB):
            mujoco.mj_step(self.model, self.data)  # resolved through this module's global (patched by the hook)
        return self._obs(), 0.0, False, False, {}

    def close(self):
        self.renderer.close()


def wrap(cfg, hook, size=(W, H), keys=("cam",)):
    """SlideEnv wrapped with one ``CameraConfig(*size, **cfg)`` per key (key = MuJoCo camera name)."""
    c = CameraConfig(*size, **cfg)
    return RealSensorWrapper(SlideEnv(), {k: (k, c) for k in keys}, substep_module=THIS if hook else None)


def record(env, n, key="cam"):
    """Reset then n steps; per observation (time, age, ideal, real) for ``key``."""
    obs, info = env.reset()
    out = []
    for i in range(n + 1):
        if i:
            obs, _, _, _, info = env.step(None)
        out.append((env.unwrapped.data.time, info["camera_age_s"][key], info["ideal_pixels"][key],
                    obs["pixels"][key]))
    return out


def run(cfg, hook, n=12, size=(W, H)):
    """``record`` on a fresh wrapper, closed even when the run fails (the module patch must not leak)."""
    env = wrap(cfg, hook, size)
    try:
        return record(env, n)
    finally:
        env.close()


def test_identity_cameras_at_control_rate_equal_env_renders():
    """Identity config, fps = control rate, no latency: every delivered frame of each key is the env's own render
    of that camera (the two cameras see different images, so a swapped camera would fail)."""
    env = wrap(dict(fps=1 / CTRL_DT), hook=False, keys=("cam", "cam2"))
    try:
        obs, info = env.reset()
        first = obs["pixels"]["cam"]
        for i in range(13):
            if i:
                obs, _, _, _, info = env.step(None)
            for k in ("cam", "cam2"):
                assert info["camera_age_s"][k] == pytest.approx(0, abs=1e-9)
                np.testing.assert_array_equal(obs["pixels"][k], info["ideal_pixels"][k])
            assert not np.array_equal(obs["pixels"]["cam"], obs["pixels"]["cam2"])
        assert not np.array_equal(first, obs["pixels"]["cam"])  # the box moved
    finally:
        env.close()


@pytest.mark.parametrize("hook", [True, False])
def test_capture_times_follow_the_physics_step_only_with_the_hook(hook):
    """30 fps, 40 ms latency: with the hook, captures fire on the 2 ms physics grid (n/30 rounded up to it);
    without it they snap to the 20 ms control grid. Age = time - capture_t of the newest delivered frame."""
    lat = 0.04
    out = run(dict(fps=30, latency_s=lat), hook=hook, n=20)
    grid = 0.002 if hook else CTRL_DT
    fire = [math.ceil(k / 30 / grid - 1e-6) * grid for k in range(100)]  # fire time of capture k on the grid
    for t, age, _, _ in out[1:]:
        ready = [f for f in fire if f + lat <= t + 1e-9]
        assert t - age == pytest.approx(ready[-1] if ready else 0.0, abs=1e-9), t


def test_reset_holds_a_render_and_resizes_to_the_observation():
    """Before the first delivery the held frame is a render of the reset state (age 0); a 2x config is
    resized to the observation's 64 x 48 with INTER_AREA."""
    out = run(dict(fps=30, latency_s=0.1), hook=True, n=2, size=(2 * W, 2 * H))
    _, age0, ideal0, real0 = out[0]
    assert age0 == 0 and real0.shape == ideal0.shape == (H, W, 3)
    assert out[2][1] == pytest.approx(out[2][0])  # still the reset frame at t = 0.04 < latency
    assert np.abs(real0.astype(int) - ideal0).mean() < 8  # same scene, area-downsampled from 2x


def test_second_reset_shows_the_reset_state():
    """Windowed camera (8 ms blur, no distortion): after 30 steps (box moved 0.3 m) a reset's held frame is the
    reset state, not the last episode's (the ring is emptied before the render), within 1 DN of the ideal."""
    env = wrap(dict(fps=30, latency_s=0.05, exposure_s=0.008, blur_samples=2), hook=True)
    try:
        record(env, 30)
        _, age, ideal, real = record(env, 0)[0]
        assert age == 0 and np.abs(real.astype(int) - ideal).max() <= 1
    finally:
        env.close()


def test_two_wrappers_share_the_module():
    """Stepping env A advances only A's streams; B keeps its reset frame. Closing B keeps A's hook; closing A
    gives the module the real mujoco back."""
    b = wrap(dict(fps=30, latency_s=0.04), hook=True)  # registered first: a dispatch to "any hook" would hit B
    a = wrap(dict(fps=30, latency_s=0.04), hook=True)
    try:
        tb, ta = record(b, 0), record(a, 5)
        # t = 0.1: capture 1 fired at 0.034 (1/30 on the 2 ms grid), ready 0.074; capture 2 (0.068) not yet
        assert ta[-1][0] - ta[-1][1] == pytest.approx(0.034, abs=1e-9)
        assert b._frames["cam"][1] == 0 and tb[0][1] == 0 and b.unwrapped.data.time == 0
        b.close()
        assert isinstance(THIS.mujoco, _StepHook)
        _, _, _, _, info = a.step(None)
        assert 0.12 - info["camera_age_s"]["cam"] == pytest.approx(0.068, abs=1e-9)  # A still on its own grid
    finally:
        b.close(), a.close()
    assert THIS.mujoco is sys.modules["mujoco"]


def test_rebound_and_nested_wrappers_of_one_env_both_advance():
    """Rebinding a new wrapper on the same base env and dropping the old one keeps the new hook alive; a wrapper
    nested in another on the same env keeps advancing too (the hooks of one data are a list)."""
    cfg = CameraConfig(W, H, fps=30, latency_s=0.04)
    base = SlideEnv()
    old = RealSensorWrapper(base, {"cam": ("cam", cfg)}, substep_module=THIS)
    new = RealSensorWrapper(base, {"cam": ("cam", cfg)}, substep_module=THIS)
    del old
    gc.collect()
    outer = None
    try:
        t, age, _, _ = record(new, 5)[-1]
        assert t - age == pytest.approx(0.034, abs=1e-9)
        outer = RealSensorWrapper(new, {"cam2": ("cam2", cfg)}, substep_module=THIS)
        outer.reset()
        for _ in range(5):
            _, _, _, _, info = outer.step(None)
        assert 0.1 - info["camera_age_s"]["cam"] == pytest.approx(0.034, abs=1e-9)  # inner, via obs passthrough
        assert 0.1 - info["camera_age_s"]["cam2"] == pytest.approx(0.034, abs=1e-9)
        assert new._frames["cam"][1] == pytest.approx(0.034, abs=1e-9)
    finally:
        (outer or new).close()
    assert THIS.mujoco is sys.modules["mujoco"]

def test_dropped_wrapper_unpatches_and_proxy_delegates():
    """A wrapper garbage-collected without close() restores the module; the proxy forwards attributes and runs
    only the hook registered for the stepped data."""
    env = wrap(dict(fps=30), hook=True)
    env.reset()
    del env
    gc.collect()
    assert THIS.mujoco is sys.modules["mujoco"]
    seen = []

    class Sink:
        def hit(self, d):
            seen.append(d.time)

    s, p = Sink(), _StepHook()
    m = p.MjModel.from_xml_string(XML)
    d, other = p.MjData(m), p.MjData(m)
    p.hooks[id(d)] = [weakref.WeakMethod(s.hit)]
    p.mj_step(m, other)
    p.mj_step(m, d)
    assert seen == [pytest.approx(0.002)]


def test_scene_reaches_every_camera():
    """``scene=`` is applied by every camera: identity cameras with a dimmed scene differ from the env's frames."""
    from physicalai_sim2real_camera.scene_dr import SceneConfig

    c = CameraConfig(W, H)
    env = RealSensorWrapper(SlideEnv(), {k: (k, c) for k in ("cam", "cam2")}, scene=SceneConfig(light_gain=0.5))
    try:
        obs, info = env.reset()
        for k in ("cam", "cam2"):
            assert env.cams[k].scene is not None
            assert obs["pixels"][k].mean() < 0.8 * info["ideal_pixels"][k].mean()
    finally:
        env.close()


def test_depth_identity_equals_the_env_depth_render():
    """depth=True, identity config at the control rate: obs["depth"][k] is MuJoCo's own depth render of that
    camera at the current state (float32 metres), and colour still equals the env's render."""
    c = CameraConfig(W, H, fps=1 / CTRL_DT)
    env = RealSensorWrapper(SlideEnv(), {k: (k, c) for k in ("cam", "cam2")}, depth=True)
    r = mujoco.Renderer(env.unwrapped.model, H, W)
    r.enable_depth_rendering()
    try:
        obs, info = env.reset()
        for i in range(6):
            if i:
                obs, _, _, _, info = env.step(None)
            for k in ("cam", "cam2"):
                r.update_scene(env.unwrapped.data, camera=k)
                np.testing.assert_allclose(obs["depth"][k], r.render(), rtol=1e-6)
                assert obs["depth"][k].dtype == np.float32
                np.testing.assert_array_equal(obs["pixels"][k], info["ideal_pixels"][k])
    finally:
        r.close()
        env.close()


def test_depth_leaves_colour_unchanged_and_resizes_nearest(monkeypatch):
    """Noisy, blurred, jittered 2x camera: colour frames with depth on match depth off (measured: one pixel of one
    frame off by 1 DN, rasterisation rounding after the depth sub-render), and depth is resized to the observation
    with INTER_NEAREST from the delivered capture's 2x depth."""
    cfg = dict(fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.003, drop_prob=0.2, exposure_s=0.008,
               blur_samples=2, sigma_shot=0.05, sigma_read=0.01, seed=3)
    c = CameraConfig(2 * W, 2 * H, **cfg)
    outs = []
    monkeypatch.setattr(camera_mod, "_RNG_SEEDED", True)  # temporal noise: cv2's process-global RNG, seeded here
    for depth in (False, True):
        cv2.setRNGSeed(0)
        env = RealSensorWrapper(SlideEnv(), {"cam": ("cam", c)}, substep_module=THIS, depth=depth)
        try:
            obs, _ = env.reset()
            seq = [(obs["pixels"]["cam"].copy(), obs.get("depth", {}).get("cam"))]
            for _ in range(15):
                obs, *_ = env.step(None)
                seq.append((obs["pixels"]["cam"].copy(), obs.get("depth", {}).get("cam")))
            outs.append(seq)
            held = env._frames["cam"][0][1]  # 2x depth of the delivered capture
        finally:
            env.close()
    for (p0, d0), (p1, d1) in zip(*outs):
        diff = np.abs(p0.astype(int) - p1)
        assert diff.max() <= 1 and np.count_nonzero(diff) <= 3
        assert d0 is None and d1.shape == (H, W)
    assert held.shape == (2 * H, 2 * W)
    np.testing.assert_array_equal(d1, cv2.resize(held, (W, H), interpolation=cv2.INTER_NEAREST))


def test_depth_follows_the_capture_time():
    """60 ms latency, box sliding at 0.5 m/s: the delivered depth is a render of the state at the frame's capture
    time (``time - camera_age_s``), not of the current state."""
    c = CameraConfig(W, H, fps=30, latency_s=0.06)
    env = RealSensorWrapper(SlideEnv(), {"cam": ("cam", c)}, substep_module=THIS, depth=True)
    m = env.unwrapped.model
    d, r = mujoco.MjData(m), mujoco.Renderer(m, H, W)
    r.enable_depth_rendering()

    def render_at(x):
        d.qpos[0] = x
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="cam")
        return r.render()

    try:
        env.reset()
        for _ in range(10):
            obs, _, _, _, info = env.step(None)
        t, age = env.unwrapped.data.time, info["camera_age_s"]["cam"]
        assert age > 0.05
        off = lambda x: np.count_nonzero(~np.isclose(obs["depth"]["cam"], render_at(x), rtol=1e-5))  # noqa: E731
        assert off(0.5 * (t - age)) <= 3  # edge pixels may flip on float rounding of the box position
        assert off(0.5 * t) > 10  # the box moved 3 cm since the capture (measured 34 px vs 1)
    finally:
        r.close()
        env.close()
