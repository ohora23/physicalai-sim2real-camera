"""scene_dr without GL: pure functions, and SceneDR.apply on an MjvScene from mjv_updateScene."""

import cv2
import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera.scene_dr import (
    MAX_RENDER_LIGHTS,
    Lamp,
    SceneConfig,
    SceneDR,
    Walker,
    color_factors,
    distractor_geoms,
    floor_photo,
    floor_texture_image,
    mark_frames,
    walker_pos,
)

XML = """<mujoco><worldbody>
  <light pos="0 0 3" directional="true" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
  <light pos="1 1 2" diffuse="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
  <geom name="floor" type="plane" size="2 2 .1"/>
  <body pos="0.3 0.2 0.5" euler="10 20 30"><freejoint/><geom name="box" type="box" size=".1 .05 .02"/></body>
</worldbody></mujoco>"""
H_BOX = np.array([0.1, 0.05, 0.02])


def setup(maxgeom=200):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, update(m, d, maxgeom)


def update(m, d, maxgeom=200):
    scn = mujoco.MjvScene(m, maxgeom=maxgeom)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    return scn


def gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_walker_pos():
    w = Walker(p0=(0.0, 0.0), p1=(2.0, 0.0), period_s=4.0, height=1.7, z0=0.1)
    for t, x in ((0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (5, 1)):
        assert np.allclose(walker_pos(w, t), (x, 0, 0.1 + 0.85)), t
    shifted = Walker(p0=(0.0, 0.0), p1=(2.0, 0.0), period_s=4.0, phase=0.25)
    assert np.isclose(walker_pos(shifted, 0)[0], 1) and np.isclose(walker_pos(shifted, 1)[0], 2)


def test_mark_frames_on_surface_box():
    frames = mark_frames(mujoco.mjtGeom.mjGEOM_BOX, H_BOX, 400, 0.25, np.random.default_rng(0))
    on_z = 0
    for pos, R, size, rgba in frames:
        q = np.abs(pos) / H_BOX
        k = int(np.argmax(q))
        assert abs(q[k] - 1) < 1e-9  # on the surface
        n = np.zeros(3)
        n[k] = np.sign(pos[k])
        assert np.allclose(R[:, 2], n, atol=1e-9)  # z axis = outward normal of that face
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
        face = np.delete(H_BOX, k)
        assert all(0.5 * 0.25 * face.min() - 1e-12 <= a <= 1.5 * 0.25 * face.min() + 1e-12 for a in size[:2])
        assert size[2] == 0.0004 and rgba.dtype == np.float32 and 0.5 <= rgba[3] <= 0.9
        assert np.all(np.abs(np.delete(pos, k)) + max(size[:2]) <= face + 1e-12)  # stain inside the face
        on_z += k == 2
    assert abs(on_z / len(frames) - 0.02 / (0.02 + 0.008 + 0.004)) < 0.1  # face chosen by area


def test_mark_frames_plane_and_errors():
    frames = mark_frames(mujoco.mjtGeom.mjGEOM_PLANE, np.array([0.0, 0.5, 0.1]), 100, 0.25, np.random.default_rng(1))
    for pos, R, _, _ in frames:
        assert pos[2] == 0 and np.allclose(R[:, 2], (0, 0, 1)) and abs(pos[0]) <= 1.0 and abs(pos[1]) <= 0.5
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert len({round(float(R[0, 0]), 6) for _, R, _, _ in frames}) > 50  # stains are rotated in-plane
    assert SceneConfig(light_tint=np.array([1.0, 0.5, 0.5])).light_tint == (1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        mark_frames(mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), 1, 0.25, np.random.default_rng(0))


def test_color_factors():
    rng = np.random.default_rng(0)
    assert np.array_equal(color_factors(4, 0.0, rng), np.ones((4, 3), np.float32))
    assert rng.uniform() == np.random.default_rng(0).uniform()  # jitter 0 draws nothing
    f = color_factors(1000, 0.3, rng)
    assert f.shape == (1000, 3) and f.dtype == np.float32 and 0.7 <= f.min() and f.max() <= 1.3


def test_apply_lights_and_defaults():
    m, d, ref = setup()
    scn = update(m, d)
    SceneDR(m, SceneConfig(light_gain=0.5, light_tint=(1, 0.8, 0.6), ambient=0.1)).apply(scn, d)
    assert scn.nlight == ref.nlight >= 2
    for i in range(ref.nlight):
        for f in ("diffuse", "specular"):
            assert np.allclose(getattr(scn.lights[i], f), getattr(ref.lights[i], f) * 0.5 * np.array([1, 0.8, 0.6]))
        assert np.allclose(scn.lights[i].ambient, ref.lights[i].ambient + 0.1)
    scn = update(m, d)
    SceneDR(m, SceneConfig()).apply(scn, d)
    assert (scn.ngeom, scn.nlight) == (ref.ngeom, ref.nlight)
    for i in range(ref.nlight):
        assert all(np.array_equal(getattr(scn.lights[i], f), getattr(ref.lights[i], f))
                   for f in ("diffuse", "specular", "ambient", "pos", "dir"))
    for i in range(ref.ngeom):
        assert np.array_equal(scn.geoms[i].rgba, ref.geoms[i].rgba)


def test_apply_lamps_walkers_marks():
    m, d, scn = setup()
    d.time = 1.5
    n0, l0 = scn.ngeom, scn.nlight
    w = Walker(p0=(1.0, -1.0), p1=(1.0, 1.0), period_s=6.0)
    lamps = (Lamp(dir=(1.0, 0.0, -1.0), directional=True), Lamp(pos=(1.0, 0.0, 1.0), dir=(0.0, 0.0, -2.0)))
    SceneDR(m, SceneConfig(lamps=lamps, walkers=(w,), marks=("box",), marks_per_geom=5)).apply(scn, d)
    assert (scn.ngeom, scn.nlight) == (n0 + 6, l0 + 2)
    assert scn.lights[l0].type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    assert scn.lights[l0 + 1].type == mujoco.mjtLightType.mjLIGHT_SPOT
    assert np.allclose(scn.lights[l0].dir, np.array([1, 0, -1]) / np.sqrt(2), atol=1e-6)
    assert np.allclose(scn.lights[l0 + 1].dir, (0, 0, -1)) and scn.lights[l0 + 1].castshadow
    assert [scn.geoms[n0 + i].segid for i in range(6)] == list(range(n0, n0 + 6))  # unique, as updateScene
    for i in range(6):  # no stale model-geom identity in segmentation renders
        g = scn.geoms[n0 + i]
        assert (g.objtype, g.objid, g.category) == (mujoco.mjtObj.mjOBJ_UNKNOWN, -1, mujoco.mjtCatBit.mjCAT_DYNAMIC)
    walker = scn.geoms[n0 + 5]
    assert walker.type == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert np.allclose(walker.pos, (1.0, -1.0 + 0.5 * 2, 0.85))  # t = 1.5 s of 6: half way to p1
    b = gid(m, "box")
    X, c = d.geom_xmat[b].reshape(3, 3), d.geom_xpos[b]
    for i in range(5):  # on the ROTATED box surface (body tilted 10/20/30 deg)
        local = X.T @ (scn.geoms[n0 + i].pos - c)
        assert abs(np.max(np.abs(local) / H_BOX) - 1) < 1e-5  # mjvGeom.pos is float32


def test_apply_color():
    m, d, ref = setup()
    scn = update(m, d)
    SceneDR(m, SceneConfig(color_jitter=0.3, color_geoms=("box",))).apply(scn, d)
    changed = 0
    for i in range(ref.ngeom):
        a, b = ref.geoms[i].rgba, scn.geoms[i].rgba
        if ref.geoms[i].objid == gid(m, "box"):
            ratio = b[:3] / a[:3]
            assert np.all((0.7 - 1e-6 <= ratio) & (ratio <= 1.3 + 1e-6)) and not np.allclose(ratio, 1)
            assert b[3] == a[3]
            changed += 1
        else:
            assert np.array_equal(a, b)
    assert changed == 1


def stains(cfg):
    m, d, scn = setup()
    n0 = scn.ngeom
    SceneDR(m, cfg).apply(scn, d)
    return np.array([np.r_[scn.geoms[i].pos, scn.geoms[i].size, scn.geoms[i].rgba] for i in range(n0, scn.ngeom)])


def test_seed_and_draw_order():
    a = stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, seed=42))
    assert len(a) == 10 and np.array_equal(a, stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, seed=42)))
    assert not np.allclose(a, stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, seed=43)))
    jit = stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, seed=42, color_jitter=0.3))
    assert np.array_equal(a[:, :6], jit[:, :6])  # colour drawn after the marks: same stains


def test_model_and_data_untouched():
    m, d, scn = setup()
    before = [x.copy() for x in (m.light_diffuse, m.light_specular, m.light_ambient, m.geom_rgba, m.geom_size,
                                  d.qpos, d.geom_xpos, d.geom_xmat)]
    cfg = SceneConfig(light_gain=0.5, light_tint=(1, 0.8, 0.6), ambient=0.1, lamps=(Lamp(),), walkers=(Walker(),),
                      marks=("box", "floor"), color_jitter=0.3)
    SceneDR(m, cfg).apply(scn, d)
    after = (m.light_diffuse, m.light_specular, m.light_ambient, m.geom_rgba, m.geom_size, d.qpos, d.geom_xpos,
             d.geom_xmat)
    assert all(np.array_equal(x, y) for x, y in zip(before, after))


def test_full_scene_raises():
    m, d, ref = setup()
    full = update(m, d, maxgeom=ref.ngeom)
    with pytest.raises(RuntimeError, match="full"):
        SceneDR(m, SceneConfig(walkers=(Walker(),))).apply(full, d)
    scn = update(m, d)
    free = MAX_RENDER_LIGHTS - scn.nlight  # headlight + 2 model lights
    SceneDR(m, SceneConfig(lamps=(Lamp(),) * free)).apply(scn, d)
    assert scn.nlight == MAX_RENDER_LIGHTS
    with pytest.raises(RuntimeError, match="full"):
        SceneDR(m, SceneConfig(lamps=(Lamp(),) * (free + 1))).apply(update(m, d), d)


def test_floor_texture_image():
    """Deterministic per seed, uint8 (h, w, 3), varied across seeds; checker / stripes tile without a seam."""
    a = floor_texture_image(64, 96, np.random.default_rng(3))
    assert a.shape == (64, 96, 3) and a.dtype == np.uint8
    assert np.array_equal(a, floor_texture_image(64, 96, np.random.default_rng(3)))
    imgs = [floor_texture_image(64, 64, np.random.default_rng(s)) for s in range(40)]
    assert len({im.tobytes() for im in imgs}) == 40
    assert len({int(im.mean()) for im in imgs}) > 10
    for s in range(40):  # tiled 2 x 2, sharp cell / stripe edges stay evenly spaced across the seam
        t = np.tile(floor_texture_image(64, 64, np.random.default_rng(s)).astype(float).sum(-1), (2, 2))
        for prof in (np.abs(np.diff(t, axis=1)).mean(0), np.abs(np.diff(t, axis=0)).mean(1)):
            if prof.max() < 3 * np.median(prof) + 1:  # no sharp edges along this axis (blotches, flat, stripes)
                continue
            gaps = np.diff(np.flatnonzero(prof > 0.5 * prof.max()))
            assert gaps.max() - gaps.min() <= 1, s  # an odd cell count would double one gap at the seam


def test_floor_photo(tmp_path):
    """A photo is chosen by the rng, area-resized to the texture, turned only when square; unreadable -> error."""
    a = np.zeros((40, 60, 3), np.uint8)
    a[:, :30] = (0, 0, 255)  # BGR red left half
    b = np.full((40, 60, 3), 200, np.uint8)
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), a)
    cv2.imwrite(str(pb), b)
    seen = {floor_photo((pa, pb), 20, 30, np.random.default_rng(s))[0, 0].tobytes() for s in range(20)}
    assert len(seen) == 2  # both files get chosen
    img = floor_photo((pa,), 20, 30, np.random.default_rng(0))
    assert img.shape == (20, 30, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (255, 0, 0) and tuple(img[0, -1]) == (0, 0, 0)  # RGB, not turned (non-square)
    base = floor_photo((pa,), 32, 32, np.random.default_rng(1))
    turns = {np.rot90(base, k).tobytes() for k in range(4)}
    sq = {floor_photo((pa,), 32, 32, np.random.default_rng(s)).tobytes() for s in range(20)}
    assert len(sq) > 1 and sq <= turns  # square textures get turned, by multiples of 90 degrees only
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    floor_photo((pa,), 32, 32, r1), floor_photo((pa,), 20, 30, r2)
    assert r1.random() == r2.random()  # the turn is drawn for every shape: the stream does not depend on it
    assert np.array_equal(floor_photo((pa, pb), 32, 32, np.random.default_rng(5)),
                          floor_photo((pa, pb), 32, 32, np.random.default_rng(5)))
    with pytest.raises(ValueError):
        floor_photo((tmp_path / "missing.png",), 8, 8, np.random.default_rng(0))


def test_distractor_geoms():
    """n objects, inside the region, clear of the keep-out, resting on z = 0, opaque; deterministic per seed."""
    region, keep = (-0.5, -0.5, 0.5, 0.5), (-0.1, -0.1, 0.1, 0.1)
    g = distractor_geoms(12, region, keep, (0.02, 0.05), np.random.default_rng(1))
    assert len(g) == 12
    for gtype, size, pos, mat, rgba in g:
        geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(geom, gtype, size, pos, mat.ravel(), rgba)
        half_h = geom.size[2] + (geom.size[0] if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
        assert abs(pos[2] - half_h) < 2e-9  # resting on z = 0 (float32 precision)
        r = geom.size[0] * (np.sqrt(2) * max(1.0, geom.size[1] / geom.size[0]) if gtype == mujoco.mjtGeom.mjGEOM_BOX
                            else 1.0)
        assert region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]
        assert not (keep[0] - r < pos[0] < keep[2] + r and keep[1] - r < pos[1] < keep[3] + r)
        assert rgba[3] == 1.0 and mat.shape == (3, 3)
    again = distractor_geoms(12, region, keep, (0.02, 0.05), np.random.default_rng(1))
    assert all(np.array_equal(p[2], q[2]) for p, q in zip(g, again))
    assert distractor_geoms(0, region, keep, (0.02, 0.05), np.random.default_rng(1)) == []


def test_distractors_add_geoms():
    """SceneDR with distractors=5 adds exactly 5 geoms after the existing ones."""
    m, d, scn = setup()
    n0 = scn.ngeom
    SceneDR(m, SceneConfig(distractors=5)).apply(scn, d)
    assert scn.ngeom == n0 + 5


def test_draw_order_preserves_stains():
    """Turning distractors on leaves the stains of the same seed unchanged (their draws come later)."""
    a = stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, seed=4))
    b = stains(SceneConfig(marks=("box", "floor"), marks_per_geom=5, distractors=5, seed=4))
    assert np.array_equal(a, b[:len(a)])  # stains() returns every added geom: the distractors follow the stains


def test_config_validation():
    m, _, _ = setup()
    for bad in (dict(light_gain=-0.1), dict(ambient=-0.1), dict(color_jitter=-0.1), dict(color_jitter=1.0),
                dict(marks_per_geom=-1), dict(mark_size=0), dict(shadow_clip=-1), dict(light_tint=(1.0, 1.0)),
                dict(light_tint=(1.0, -0.1, 1.0)),
                dict(walkers=(Walker(period_s=0),)), dict(walkers=(Walker(radius=0),)),
                dict(walkers=(Walker(radius=0.2, height=0.3),)),
                dict(distractors=-1), dict(distractor_region=(1, 0, 0, 1)),
                dict(distractor_keepout=(0, 0, 0)), dict(distractor_size=(0.05, 0.01))):
        with pytest.raises(ValueError):
            SceneConfig(**bad)
    for bad in (dict(marks=("nope",)), dict(color_jitter=0.1, color_geoms=("nope",))):
        with pytest.raises(ValueError):
            SceneDR(m, SceneConfig(**bad))


def test_floor_and_distractor_streams():
    """B3's distractors equal B2's for the same seed: floor_texture draws from its own stream."""
    xml = """<mujoco><visual><headlight ambient=".4 .4 .4" diffuse=".6 .6 .6" specular="0 0 0"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".2 .3 .4" rgb2=".1 .2 .3"/>
  <material name="grid" texture="grid" texrepeat="2 2"/></asset><worldbody>
  <geom name="floor" type="plane" size="3 3 .1" material="grid"/>
  <camera name="cam" fovy="40" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    mujoco.mj_forward(m, mujoco.MjData(m))
    cfg_b2 = SceneConfig(distractors=4, seed=9)
    cfg_b3 = SceneConfig(distractors=4, floor_texture=True, seed=9)
    dr2 = SceneDR(m, cfg_b2)
    dr3 = SceneDR(m, cfg_b3)
    for a, b in zip(dr2._distractors, dr3._distractors):
        assert np.array_equal(a[2], b[2])  # positions identical
    assert np.array_equal(
        SceneDR(m, SceneConfig(floor_texture=True, seed=9))._floor[1],
        dr3._floor[1])


def test_scene_dr_uses_floor_photo(tmp_path):
    """SceneDR takes the photo from the floor stream; the distractors stay on their own stream."""
    xml = """<mujoco><asset><texture name="grid" type="2d" builtin="checker" width="64" height="64"/>
  <material name="grid" texture="grid"/></asset><worldbody>
  <geom name="floor" type="plane" size="3 3 .1" material="grid"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    img = np.zeros((48, 48, 3), np.uint8)
    img[:24] = (10, 200, 30)
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), img)
    cfg = SceneConfig(floor_texture=True, floor_images=(p,), distractors=3, seed=9)
    dr = SceneDR(m, cfg)
    assert np.array_equal(dr._floor[1], floor_photo((str(p),), 64, 64, np.random.default_rng([9, 12])))
    plain = SceneDR(m, SceneConfig(distractors=3, seed=9))
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(dr._distractors, plain._distractors))
    assert cfg.floor_images == (str(p),) and hash(cfg)
    with pytest.raises(ValueError):
        SceneConfig(floor_images=(p,))  # floor_images without floor_texture


if __name__ == "__main__":
    import inspect
    import pathlib
    import tempfile

    for name, f in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in inspect.signature(f).parameters:
                with tempfile.TemporaryDirectory() as d:
                    f(pathlib.Path(d))
            else:
                f()
    print("ALL TESTS PASSED")
