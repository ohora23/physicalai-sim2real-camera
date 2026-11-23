"""Camera-side DR effects wired into SimCamera: ccm, mains flicker, lens dirt, MJPEG; config + randomize."""

import dataclasses

import mujoco
import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig, SimCamera
from physicalai_sim2real_camera.calib import randomize
from physicalai_sim2real_camera.lens import dirt_maps
from physicalai_sim2real_camera.sensor import linearize

W, H = 160, 120
LUT = linearize(np.arange(256, dtype=np.uint8), 2.2)


def grey_scene(rgba="0.45 0.45 0.45 1"):
    """Camera looking straight down at a uniformly lit matte plane that fills the view."""
    xml = f"""<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0"/></visual><worldbody>
  <light pos="0 0 3" directional="true" diffuse="0.8 0.8 0.8" specular="0 0 0"/>
  <geom type="plane" size="5 5 0.01" rgba="{rgba}"/>
  <camera name="cam" fovy="45" pos="0 0 0.5" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def shot(model, data, **kw):
    cam = SimCamera(model, "cam", CameraConfig(W, H, **kw))
    try:
        return cam.render(data)
    finally:
        cam.close()


def lin(img):
    return LUT[img].astype(np.float64)


BASE = dict(vignette=0.3)  # non-identity, noise off: every effect below is deterministic


def test_defaults_are_identity():
    model, data = grey_scene()
    cam = SimCamera(model, "cam", CameraConfig(W, H))
    assert cam.identity and not cam.lens_on
    cam.close()


def test_ccm_identity_and_swap():
    model, data = grey_scene("0.6 0.4 0.2 1")
    off = shot(model, data, **BASE)
    assert np.array_equal(shot(model, data, **BASE, ccm=(1, 0, 0, 0, 1, 0, 0, 0, 1)), off)
    swap = shot(model, data, **BASE, ccm=(0, 0, 1, 0, 1, 0, 1, 0, 0))  # R <-> B on linear light
    assert np.array_equal(swap, off[..., ::-1])


def test_flicker_global_shutter():
    model, data = grey_scene()
    off = lin(shot(model, data, **BASE))
    for t in (0.0, 0.0031, 0.0047):
        data.time = t
        on = lin(shot(model, data, **BASE, flicker_amp=0.3, flicker_hz=100, flicker_phase=0.2))
        g = 1 + 0.3 * np.cos(2 * np.pi * 100 * t + 0.2)
        m = off > 0.05
        assert abs(np.median(on[m] / off[m]) - g) < 0.01, t  # uint8 quantisation ~0.5 % at this level
    data.time = 0.0


def test_flicker_rolling_shutter_bands():
    """Rows exposed at different times see different light: the on/off row ratio follows flicker_rows."""
    model, data = grey_scene()
    kw = dict(BASE, readout_s=0.03, rs_bands=8)
    cams = [SimCamera(model, "cam", CameraConfig(W, H, **kw, **f)) for f in ({}, dict(flicker_amp=0.4, flicker_hz=100))]
    try:
        for i in range(40):  # static scene: fill the state ring over > one window
            data.time = 0.002 * i
            for c in cams:
                c.push(data)
        off, on = (lin(c.render(data)) for c in cams)
    finally:
        for c in cams:
            c.close()
    tc = 0.078 - 0.015  # last push at 39 * 2 ms; mid-exposure of the centre row = time - readout / 2
    g = 1 + 0.4 * np.cos(2 * np.pi * 100 * (tc + 0.03 * ((np.arange(H) + 0.5) / H - 0.5)))
    ratio = np.median(on / np.maximum(off, 1e-6), axis=(1, 2))
    assert np.abs(ratio - g).max() < 0.02
    assert g.max() - g.min() > 0.5  # 3 periods over the frame: the bands are really there
    data.time = 0.0


def test_flicker_first_frame_same_time_reference():
    """Empty ring (episode start): the single-instant frame uses the window's mid-exposure time, data.time - half."""
    model, data = grey_scene()
    data.time = 0.05
    kw = dict(BASE, readout_s=0.03)
    off = lin(shot(model, data, **kw))
    on = lin(shot(model, data, **kw, flicker_amp=0.4, flicker_hz=100))
    t_c = 0.05 - 0.015  # 1.5 periods earlier than data.time at 100 Hz: antiphase if the reference were data.time
    g = 1 + 0.4 * np.cos(2 * np.pi * 100 * (t_c + 0.03 * ((np.arange(H) + 0.5) / H - 0.5)))
    assert np.abs(np.median(on / np.maximum(off, 1e-6), axis=(1, 2)) - g).max() < 0.02
    data.time = 0.0


def test_dirt_transmission_and_veil():
    model, data = grey_scene()
    off = lin(shot(model, data, **BASE))
    dirty = lin(shot(model, data, **BASE, dirt_density=0.5))
    T, _ = dirt_maps(H, W, 0.5, np.random.default_rng([0, 3]))  # cfg.seed 0 + cam id 0, stream 3
    m = off > 0.05
    assert np.abs(dirty[m] / off[m] - np.repeat(T, 3, 2)[m]).max() < 0.03
    glare = lin(shot(model, data, **BASE, dirt_density=0.5, dirt_scatter=0.5))
    assert (glare >= dirty - 1e-6).all() and glare.mean() > dirty.mean()  # scattered light only adds


def test_jpeg_quality():
    model, data = grey_scene("0.6 0.4 0.2 1")
    off = shot(model, data, **BASE).astype(int)
    q90, q10 = (shot(model, data, **BASE, jpeg_quality=q).astype(int) for q in (90, 10))
    assert q90.shape == off.shape and np.abs(q90 - off).mean() < 2.0
    assert np.abs(q10 - off).mean() > np.abs(q90 - off).mean()


def edge_scene():
    """Camera 0.5 m above a dark floor; a white slab covers x > 0, so the image has one vertical edge."""
    xml = """<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0"/></visual><worldbody>
  <light pos="0 0 3" directional="true" diffuse="0.8 0.8 0.8" specular="0 0 0" castshadow="false"/>
  <geom type="plane" size="5 5 0.01" rgba="0.1 0.1 0.1 1"/>
  <geom type="box" pos="1 0 0.001" size="1 2 0.001" rgba="0.9 0.9 0.9 1"/>
  <camera name="cam" fovy="45" pos="0 0 0.5" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def lsf_var(img):
    """Variance (px^2) of the line spread function: derivative of the row-averaged linear edge profile."""
    prof = lin(img).mean(axis=(0, 2))
    d = np.diff(prof)
    c = int(np.argmax(d))
    lsf = np.clip(d[c - 15:c + 16], 0, None)  # window around the edge
    x = np.arange(lsf.size) + 0.5
    m = (lsf * x).sum() / lsf.sum()
    return (lsf * (x - m) ** 2).sum() / lsf.sum()


def test_psf_widens_edge_by_sigma_squared():
    """Gaussian PSF: LSF variances add, so var(on) - var(off) = sigma^2 (uint8 + MSAA edge: within 15 %)."""
    model, data = edge_scene()
    off = lsf_var(shot(model, data))  # identity camera: the bare render (no vignette slope under the edge)
    for s in (1.0, 2.0, 3.0):
        on = lsf_var(shot(model, data, psf_sigma_px=s))
        assert abs((on - off) - s * s) < 0.15 * s * s, (s, on - off)
    flat = lin(shot(*grey_scene(), **BASE))
    assert np.abs(lin(shot(*grey_scene(), **BASE, psf_sigma_px=2.0)) - flat).max() < 0.01  # a flat field stays flat


def test_psf_once_on_window_path():
    """Static scene: a motion-blur window with the PSF equals the single-instant PSF render (one blur, not per band)."""
    model, data = edge_scene()
    single = shot(model, data, psf_sigma_px=2.0)
    cam = SimCamera(model, "cam", CameraConfig(W, H, psf_sigma_px=2.0, exposure_s=0.01, readout_s=0.02, rs_bands=4))
    try:
        for i in range(40):
            data.time = 0.002 * i
            cam.push(data)
        win = cam.render(data)
    finally:
        cam.close()
        data.time = 0.0
    assert np.abs(win.astype(int) - single).max() <= 1


def test_psf_before_noise():
    """Sensor noise is added after the lens: pixel-to-pixel noise on a flat field is not smoothed by the PSF."""
    model, data = grey_scene()
    sd = []
    for s in (0.0, 2.0):
        img = lin(shot(model, data, **BASE, sigma_read=0.02, psf_sigma_px=s))[40:80, 50:110, 1]
        sd.append(np.diff(img, axis=1).std())
    assert abs(sd[1] / sd[0] - 1) < 0.1, sd


def test_lens_fields():
    assert CameraConfig(W, H).lens_fields() == []
    c = CameraConfig(W, H, ccm=np.eye(3), dirt_density=0.1, psf_sigma_px=1.0)
    assert c.lens_fields() == ["ccm", "dirt_density", "psf_sigma_px"]


def test_cpu_only():
    model, data = grey_scene()
    with pytest.raises(ValueError, match="CPU"):
        SimCamera(model, "cam", CameraConfig(W, H, jpeg_quality=80), device="cuda")  # MJPEG needs host pixels


def test_batch_rejects():
    pytest.importorskip("torch")
    from physicalai_sim2real_camera.batch import SimCameraBatch

    model, data = grey_scene()
    with pytest.raises(ValueError, match="CPU"):  # MJPEG only; the other camera-side fields run in a batch
        SimCameraBatch(model, "cam", CameraConfig(W, H, jpeg_quality=80), n=2, interop=False)


def test_config_validation_and_roundtrip(tmp_path):
    for bad in (dict(ccm=(1, 0, 0)), dict(flicker_amp=1.5), dict(jpeg_quality=101), dict(dirt_density=-0.1),
                dict(dirt_scatter=1.5), dict(psf_sigma_px=-0.5)):
        with pytest.raises(ValueError):
            CameraConfig(W, H, **bad)
    c = CameraConfig(W, H, ccm=(1.1, 0, 0, 0, 1, 0, 0, 0, 0.9), flicker_amp=0.2, dirt_density=0.3, jpeg_quality=70)
    assert CameraConfig.load(c.save(tmp_path / "c.json")) == c


def test_randomize_new_fields():
    off = randomize(CameraConfig(W, H), dict(jpeg_quality=0.5), np.random.default_rng(0))
    assert off.jpeg_quality == 0  # off stays off (relative spread of 0)
    assert CameraConfig(W, H, ccm=np.eye(3)).ccm == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    c = CameraConfig(W, H, flicker_amp=0.9, dirt_density=0.4, dirt_scatter=0.9, jpeg_quality=95, psf_sigma_px=0.8)
    spread = dict(ccm=0.05, flicker_amp=0.5, flicker_phase=np.pi, dirt_density=0.5, dirt_scatter=0.5, jpeg_quality=0.5,
                  psf_sigma_px=1.5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = randomize(c, spread, rng)
        assert len(r.ccm) == 9 and np.abs(np.array(r.ccm) - np.eye(3).ravel()).max() <= 0.05
        assert 0 <= r.flicker_amp <= 1 and abs(r.flicker_phase) <= np.pi
        assert isinstance(r.jpeg_quality, int) and 1 <= r.jpeg_quality <= 100
        assert r.dirt_density >= 0 and 0 <= r.dirt_scatter <= 1 and r.seed != c.seed and r.psf_sigma_px >= 0
    assert dataclasses.replace(c).jpeg_quality == 95  # randomize did not mutate the fit


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        if f is test_config_validation_and_roundtrip:
            f(Path(tempfile.mkdtemp()))
        else:
            f()
    print("ALL TESTS PASSED")
