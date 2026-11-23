"""render_scale: supersampled render, pixel-area mean in linear light; depth = block centre."""

import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig, SimCamera
from physicalai_sim2real_camera.sensor import linearize

W, H = 160, 120
LUT = linearize(np.arange(256, dtype=np.uint8), 2.2)


def tilted_edge(offsamples):
    """Camera 0.5 m above a dark floor; a white slab rotated 7 deg gives staircase-prone, slightly tilted edges."""
    xml = f"""<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0"/><quality offsamples="{offsamples}"/></visual>
  <worldbody><light pos="0 0 3" directional="true" diffuse="0.8 0.8 0.8" specular="0 0 0" castshadow="false"/>
  <geom type="plane" size="5 5 0.01" rgba="0.1 0.1 0.1 1"/>
  <geom type="box" pos="0.3 0 0.001" euler="0 0 7" size="0.3 0.6 0.001" rgba="0.9 0.9 0.9 1"/>
  <camera name="cam" fovy="45" pos="0 0 0.5" xyaxes="1 0 0 0 1 0"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def render(model, data, depth=False, **kw):
    cam = SimCamera(model, "cam", CameraConfig(W, H, gamma=2.2, **kw))
    try:
        return cam.render(data, depth=depth)
    finally:
        cam.close()


@pytest.mark.parametrize("offsamples, factor", [(0, 0.3), (4, 0.4)])
def test_supersampling_approaches_reference(offsamples, factor):
    """Linear-light error vs an 8x reference: 4x cuts it below `factor` of 1x (measured 0.20 without MSAA, 0.24 with
    MSAA 4, which alone halves the 1x error); 2x lies in between."""
    model, data = tilted_edge(offsamples)
    ref = LUT[render(model, data, render_scale=8)]
    err = {s: np.abs(LUT[render(model, data, render_scale=s)] - ref).mean() for s in (1, 2, 4)}
    assert err[4] < factor * err[1] and err[4] < err[2] < err[1], err


def test_flat_field_and_size():
    model, data = tilted_edge(4)
    a, b = render(model, data), render(model, data, render_scale=2)
    assert a.shape == b.shape == (H, W, 3)
    flat = (slice(5, 25), slice(5, 25))  # bare floor corner, far from the slab
    assert np.abs(a[flat].astype(int) - b[flat]).max() <= 1  # area mean of a constant (8-bit re-encode: 1 LSB)


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([ys.mean(), xs.mean()]), mask.sum()


def small_box():
    """A 4 cm box 1 cm high near the image centre (whole outline in view), camera 0.5 m above."""
    xml = """<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0"/></visual><worldbody>
  <light pos="0 0 3" directional="true" diffuse="0.8 0.8 0.8" specular="0 0 0" castshadow="false"/>
  <geom type="plane" size="5 5 0.01" rgba="0.1 0.1 0.1 1"/>
  <geom type="box" pos="0.013 -0.007 0.01" euler="0 0 20" size="0.02 0.02 0.01" rgba="0.9 0.9 0.9 1"/>
  <camera name="cam" fovy="45" pos="0 0 0.5" xyaxes="1 0 0 0 1 0"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_geometry_aligned_across_scales():
    """A small box's centroid (RGB and depth) stays put for s = 1..4, with and without distortion; depth at even s
    takes the nearest of the 4 centre samples, so it is not shifted by 1/(2s) px (the plain centre-index sample
    moves it by 0.25 px at s = 2)."""
    for dist in ((0, 0, 0, 0, 0), (-0.2, 0.05, 0.001, -0.001, 0)):
        model, data = small_box()
        ref = None
        for s in (1, 2, 3, 4):
            rgb, d = render(model, data, depth=True, render_scale=s, dist=dist)
            c_rgb, _ = centroid(LUT[rgb].mean(-1) > 0.3)
            c_d, area = centroid(d < 0.495)
            ref = (c_rgb, c_d, area) if ref is None else ref
            assert np.abs(c_rgb - ref[0]).max() < 0.1 and np.abs(c_d - ref[1]).max() < 0.12, (dist, s, c_d, ref[1])


def test_window_and_distortion_run():
    model, data = tilted_edge(4)
    cam = SimCamera(model, "cam", CameraConfig(W, H, gamma=2.2, render_scale=2, dist=(-0.2, 0.05, 0, 0, 0),
                                               exposure_s=0.01, readout_s=0.02, rs_bands=4))
    try:
        for i in range(20):
            data.time = 0.002 * i
            cam.push(data)
        img = cam.render(data)
    finally:
        cam.close()
        data.time = 0.0
    assert np.abs(img.astype(int) - render(model, data, render_scale=2, dist=(-0.2, 0.05, 0, 0, 0))).max() <= 1


def test_depth_is_a_centre_sample():
    model, data = tilted_edge(4)
    _, d1 = render(model, data, depth=True, render_scale=1, vignette=0.1)
    _, d4 = render(model, data, depth=True, render_scale=4, vignette=0.1)
    assert d4.shape == d1.shape == (H, W)
    gap = (d4 > 0.4985) & (d4 < 0.4995)  # an averaged depth between slab top (0.498) and floor (0.5)
    assert not gap.any() and (d4 < 0.4985).any() and (d4 > 0.4995).any()
    assert np.mean(np.abs(d4 - d1) < 1e-3) > 0.98


def test_rejections_and_validation():
    model, data = tilted_edge(0)
    for bad in (0, 1.5, -2, True, 9):
        with pytest.raises(ValueError):
            CameraConfig(W, H, render_scale=bad)
    owner = SimCamera(model, "cam", CameraConfig(W, H, gamma=2.2, render_scale=2))
    try:
        with pytest.raises(ValueError, match="render_scale"):  # a sharer must match the supersampled size
            SimCamera(model, "cam", CameraConfig(W, H, gamma=2.2), renderer=owner.renderer)
    finally:
        owner.close()
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="interop=False"):
            SimCamera(model, "cam", CameraConfig(W, H, vignette=0.1, render_scale=2), device="cuda", interop=True)
    from physicalai_sim2real_camera.batch import SimCameraBatch

    with pytest.raises(ValueError, match="batch"):
        SimCameraBatch(model, "cam", CameraConfig(W, H, render_scale=2), n=2, interop=False)
