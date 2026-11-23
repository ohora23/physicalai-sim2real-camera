"""Scene DR wired into SimCamera: light gain / tint, walker shadows that move in time, stains."""

import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig, SimCamera
from physicalai_sim2real_camera.scene_dr import Lamp, SceneConfig, Walker
from physicalai_sim2real_camera.sensor import linearize

W, H = 160, 160
LUT = linearize(np.arange(256, dtype=np.uint8), 2.2)
BASE = dict(vignette=0.3)  # non-identity, noise off: every difference is the scene


def floor_scene(light=True, box=False):
    """Camera 2 m above a matte grey floor looking straight down (view ~ +-0.73 m); optional top light and box."""
    lt = '<light pos="0 0 3" dir="0 0 -1" directional="true" diffuse=".6 .6 .6" specular="0 0 0" castshadow="false"/>'
    bx = '<body pos="0.3 0 0.05"><geom name="box" type="box" size=".15 .15 .05" rgba=".9 .9 .9 1"/></body>'
    xml = f"""<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
  <quality shadowsize="4096"/></visual><worldbody>
  {lt if light else ""}
  <geom name="floor" type="plane" size="3 3 .1" rgba=".6 .6 .6 1"/>{bx if box else ""}
  <camera name="cam" fovy="40" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def shot(model, data, scene=None, **kw):
    cam = SimCamera(model, "cam", CameraConfig(W, H, **BASE, **kw), scene=scene)
    try:
        return LUT[cam.render(data)].astype(np.float64)
    finally:
        cam.close()


def px(x, y):
    """Floor point (x, y) -> (row, col) for the straight-down camera (image x = world x, image down = world -y)."""
    f = H / 2 / np.tan(np.radians(20))
    return int(round(H / 2 - 0.5 - f * y / 2)), int(round(W / 2 - 0.5 + f * x / 2))


def test_light_gain_and_tint():
    model, data = floor_scene()
    off = shot(model, data)
    on = shot(model, data, SceneConfig(light_gain=0.5, light_tint=(1.0, 0.8, 0.6)))
    m = off > 0.05
    for c, g in enumerate((0.5, 0.4, 0.3)):  # MuJoCo shades in render values; the camera decodes them with gamma
        assert abs(np.median(on[..., c][m[..., c]] / off[..., c][m[..., c]]) - g ** 2.2) < 0.01
    assert np.array_equal(shot(model, data, SceneConfig()), off)  # defaults: bit-identical


def test_walker_shadow_moves_with_time():
    """45 deg sun from -x: a 1.7 m person standing at x = -1.2 (out of view) shades the floor up to x = 0.5."""
    model, data = floor_scene(light=False)
    sun = Lamp(pos=(-2.0, 0.0, 2.0), dir=(1.0, 0.0, -1.0), directional=True, rgb=(0.8, 0.8, 0.8))
    walker = Walker(p0=(-1.2, -1.5), p1=(-1.2, 1.5), period_s=4.0, radius=0.18, height=1.7)
    ref = shot(model, data, SceneConfig(lamps=(sun,), shadow_clip=3.0)).mean(-1)  # same light, nobody walking
    scene = SceneConfig(lamps=(sun,), walkers=(walker,), shadow_clip=3.0)
    data.time = 1.0  # quarter period: walker at y = 0
    ratio = shot(model, data, scene).mean(-1) / np.maximum(ref, 1e-6)
    assert ref[px(0.0, 0.0)] > 0.05
    for x in (-0.5, 0.0, 0.3):  # inside the shadow
        assert ratio[px(x, 0.0)] < 0.5, x
    assert ratio[px(0.65, 0.0)] > 0.95  # past the tip of the shadow (x = -1.2 + 1.7 = 0.5)
    assert ratio[px(0.0, 0.5)] > 0.95  # beside the shadow (width 2 r = 0.36 m)
    data.time = 0.0  # walker at y = -1.5: the strip at y = 0 is lit again
    assert shot(model, data, scene).mean(-1)[px(0.0, 0.0)] > 0.95 * ref[px(0.0, 0.0)]


def test_stains_only_on_their_geom():
    model, data = floor_scene(box=True)
    off = shot(model, data)
    on = shot(model, data, SceneConfig(marks=("box",), marks_per_geom=20, mark_size=0.4))
    r0, c0 = px(0.3 - 0.12, 0.12)
    r1, c1 = px(0.3 + 0.12, -0.12)
    top = (slice(r0, r1), slice(c0, c1))  # inside the box's top face (seen from above)
    assert on[top].mean() < 0.97 * off[top].mean()  # stains darken the face
    r0, c0 = px(-0.6, 0.6)
    r1, c1 = px(-0.2, -0.6)
    assert np.array_equal(on[r0:r1, c0:c1], off[r0:r1, c0:c1])  # bare floor away from the box: untouched


def test_shadow_clip_on_own_renderer():
    """shadow_clip sets this camera's MjrContext only; the model's vis.map is untouched (gym-hil needs 3 m)."""
    model, data = floor_scene()
    clip0 = model.vis.map.shadowclip
    cam = SimCamera(model, "cam", CameraConfig(W, H), scene=SceneConfig(shadow_clip=2.5))
    plain = SimCamera(model, "cam", CameraConfig(W, H), scene=SceneConfig())
    try:
        assert np.isclose(cam.renderer._mjr_context.shadowClip, 2.5)
        assert np.isclose(plain.renderer._mjr_context.shadowClip, clip0 * model.stat.extent)
        assert model.vis.map.shadowclip == clip0
    finally:
        cam.close()
        plain.close()


FAST = Walker(p0=(-0.5, 0.0), p1=(0.5, 0.0), period_s=0.4, radius=0.05, height=0.2)  # 5 m/s, in view


def test_walker_in_motion_blur_window():
    """A window camera averages its sub-renders: the frame equals the mean of single-instant frames at the sample
    times t_i = (time - e/2) + e ((i + 0.5)/N - 0.5), so a fast walker is smeared, not frozen."""
    model, data = floor_scene()
    scene, e, n = SceneConfig(walkers=(FAST,)), 0.02, 4
    cam = SimCamera(model, "cam", CameraConfig(W, H, **BASE, exposure_s=e, blur_samples=n), scene=scene)
    try:
        for i in range(30):
            data.time = 0.002 * i
            cam.push(data)
        win = LUT[cam.render(data)].astype(np.float64)
    finally:
        cam.close()
    t_end = data.time
    subs = []
    for i in range(n):
        data.time = t_end - e / 2 + e * ((i + 0.5) / n - 0.5)
        subs.append(shot(model, data, scene))
    data.time = 0.0
    assert np.abs(win - np.mean(subs, axis=0)).max() < 0.02
    assert np.abs(subs[0] - subs[-1]).max() > 0.05  # the walker really moved within the exposure


def test_walker_in_depth():
    model, data = floor_scene()
    data.time = 0.1  # quarter period: walker centre at x = 0, top at z = 0.2 m
    cam = SimCamera(model, "cam", CameraConfig(W, H, **BASE), scene=SceneConfig(walkers=(FAST,)))
    try:
        _, depth = cam.render(data, depth=True)
    finally:
        cam.close()
        data.time = 0.0
    r, c = px(0.0, 0.0)
    assert abs(depth[r, c] - 1.8) < 0.01 and abs(depth[px(0.5, 0.5)] - 2.0) < 0.01


def test_shared_renderer_does_not_leak_scene():
    """A camera sharing the renderer of a scene camera renders the plain scene (update_scene resets the MjvScene)."""
    model, data = floor_scene()
    data.time = 0.1
    plain = shot(model, data)
    a = SimCamera(model, "cam", CameraConfig(W, H, **BASE), scene=SceneConfig(walkers=(FAST,), lamps=(Lamp(),)))
    b = SimCamera(model, "cam", CameraConfig(W, H, **BASE), renderer=a.renderer)
    try:
        with_scene = LUT[a.render(data)].astype(np.float64)
        after = LUT[b.render(data)].astype(np.float64)
    finally:
        b.close()
        a.close()
        data.time = 0.0
    assert np.abs(with_scene - plain).max() > 0.2 and np.array_equal(after, plain)


def textured_floor_scene():
    """The straight-down camera over a floor with a 2D checker texture (64 x 64)."""
    xml = """<mujoco><visual><headlight ambient=".4 .4 .4" diffuse=".6 .6 .6" specular="0 0 0"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".2 .3 .4" rgb2=".1 .2 .3"/>
  <material name="grid" texture="grid" texrepeat="2 2"/></asset><worldbody>
  <geom name="floor" type="plane" size="3 3 .1" material="grid"/>
  <camera name="cam" fovy="40" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_floor_texture_changes_floor_not_model():
    """A random floor pattern changes the image, leaves model.tex_data untouched and differs between seeds."""
    model, data = textured_floor_scene()
    tex0 = model.tex_data.copy()
    off = shot(model, data)
    a = shot(model, data, SceneConfig(floor_texture=True, seed=1))
    b = shot(model, data, SceneConfig(floor_texture=True, seed=2))
    assert np.abs(a - off).mean() > 0.01 and np.abs(a - b).mean() > 0.01
    assert np.array_equal(model.tex_data, tex0)
    assert np.array_equal(shot(model, data), off)


def test_floor_texture_errors():
    model, data = floor_scene()
    with pytest.raises(ValueError):
        SimCamera(model, "cam", CameraConfig(W, H, **BASE), scene=SceneConfig(floor_texture=True))
    model, data = textured_floor_scene()
    owner = SimCamera(model, "cam", CameraConfig(W, H, **BASE))
    try:
        with pytest.raises(ValueError):
            SimCamera(model, "cam", CameraConfig(W, H, **BASE), renderer=owner.renderer,
                      scene=SceneConfig(floor_texture=True))
    finally:
        owner.close()
    model, data = textured_floor_scene()
    floor_owner = SimCamera(model, "cam", CameraConfig(W, H, **BASE), scene=SceneConfig(floor_texture=True))
    try:
        with pytest.raises(ValueError):
            SimCamera(model, "cam", CameraConfig(W, H, **BASE), renderer=floor_owner.renderer)
    finally:
        floor_owner.close()


def test_distractors_render():
    """Distractors inside the view change the image."""
    model, data = floor_scene()
    cfg = SceneConfig(distractors=6, distractor_region=(-0.5, -0.5, 0.5, 0.5), distractor_size=(0.08, 0.12), seed=3)
    off, on = shot(model, data), shot(model, data, cfg)
    assert np.abs(on - off).max(-1).mean() > 0.005
