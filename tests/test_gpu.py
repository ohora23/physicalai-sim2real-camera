"""GPU backend (device='cuda') parity/statistics tests — plain script; ``--bench`` for timings."""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    print("SKIP: torch/cuda unavailable")
    sys.exit(0)

import cv2
import mujoco
import numpy as np
from test_camera import H, W, _plain, board_scene
from test_temporal import DT, moving_scene

from physicalai_sim2real_camera import CameraConfig, SimCamera

P1 = dict(dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=-0.3,
          sigma_prnu=0.02, sigma_dsnu=0.005, black_level=0.02, gamma=2.2)
NOISE = dict(sigma_shot=0.02, sigma_read=0.005)


# OpenCV 4.x cv2.remap quantises bilinear weights to 1/32 px (up to 0.024 of full scale = 6.2 DN vs exact bilinear,
# which 5.x and the GPU path compute): CPU-vs-GPU max-error bounds of remapped frames get this slack (means stay).
# Keyed on the version, not on undistortPointsIter: the quantisation is a remap property. Measured on 4.13: 5-7 DN.
REMAP_SLACK = 0 if int(cv2.__version__.split(".")[0]) >= 5 else 7

CAMS = []  # closed at exit: unclosed EGL renderers spam errors during interpreter teardown


def mk(*a, **kw):
    CAMS.append(SimCamera(*a, **kw))
    return CAMS[-1]


def pair(model, cfg):
    return mk(model, "cam", cfg), mk(model, "cam", cfg, device="cuda")


def diff(a, b):
    d = np.abs(a.astype(int) - b.astype(int))
    return int(d.max()), float(d.mean())


def two_depth_scene():
    """Checker planes at 0.5 m (left) and 1.0 m (right) seen straight down."""
    xml = """<mujoco><visual><global offwidth="640" offheight="480"/></visual>
<asset><texture name="chk" type="2d" builtin="checker" width="512" height="512"
  rgb1="1 1 1" rgb2="0.1 0.1 0.1"/>
  <material name="chk" texture="chk" texrepeat="10 10" texuniform="true"/></asset>
<worldbody><light pos="0 0 2" directional="true" diffuse="1 1 1"/>
  <geom type="plane" pos="0.5 0 0" size="0.5 1 0.01" material="chk"/>
  <geom type="plane" pos="-0.5 0 0.5" size="0.5 1 0.01" material="chk"/>
  <camera name="cam" pos="0 0 1" fovy="45"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_parity():
    model, data = board_scene()
    cpu, gpu = pair(model, CameraConfig(W, H, **P1))
    mx, mean = diff(cpu.render(data), gpu.render(data))
    print(f"G1 parity (noise off): max |Δ| {mx} DN, mean {mean:.4f} DN")
    assert mx <= 1 + REMAP_SLACK and mean <= 0.05
    cpu, gpu = pair(model, CameraConfig(W, H, **P1, bayer=True))
    a, b = cpu.render(data), gpu.render(data)
    mx, mean = diff(a[2:-2, 2:-2], b[2:-2, 2:-2])  # 2-px border: cv2 demosaic border rule differs
    print(f"G1 Bayer parity (interior): max |Δ| {mx} DN, mean {mean:.4f} DN; border max {diff(a, b)[0]} DN")
    assert mx <= 1 + REMAP_SLACK
    ident = mk(model, "cam", CameraConfig(W, H), device="cuda")
    assert np.array_equal(ident.render(data), _plain(model, data))
    t = ident.render(data, as_tensor=True)
    assert t.is_cuda and np.array_equal(t.cpu().numpy(), _plain(model, data))
    print("G1 identity cfg on cuda == plain render (bit)")


def test_dof_parity():
    model, data = two_depth_scene()
    cpu, gpu = pair(model, CameraConfig(W, H, **P1, aperture_m=0.016, focus_m=0.5))
    out = gpu.render(data)
    mx, mean = diff(cpu.render(data), out)
    sharp = mk(model, "cam", CameraConfig(W, H, **P1), device="cuda").render(data)
    print(f"G2 DoF parity: max |Δ| {mx} DN, mean {mean:.4f} DN (DoF itself moves {diff(out, sharp)[1]:.2f} DN mean)")
    assert mx <= 3 + REMAP_SLACK and mean <= 0.2 and diff(out, sharp)[1] > 1


def test_noise():
    model, _ = board_scene()
    cfg = CameraConfig(64, 64, **NOISE)
    g = mk(model, "cam", cfg, device="cuda")._gpu
    levels = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8], np.float32)
    var = np.empty((len(levels), 3))
    for k, lv in enumerate(levels):
        x = torch.empty((200, 3, 64, 64), device="cuda")
        for f in range(200):
            x[f] = g.add_noise(torch.full((3, 64, 64), float(lv), device="cuda"))
        var[k] = x.var((0, 2, 3)).cpu().numpy()
    for c in range(3):
        slope, icpt = np.polyfit(levels, var[:, c], 1)
        print(f"G3 PTC ch{c}: slope {slope:.3e} (σ_shot² {cfg.sigma_shot**2:.3e}), "
              f"intercept {icpt:.3e} (σ_read² {cfg.sigma_read**2:.3e})")
        assert abs(slope / cfg.sigma_shot ** 2 - 1) < 0.10
        assert abs(icpt / cfg.sigma_read ** 2 - 1) < 0.20
    model, data = board_scene()
    outs = [mk(model, "cam", CameraConfig(W, H, **P1, **NOISE, seed=s), device="cuda").render(data)
            for s in (7, 7, 8)]
    assert np.array_equal(outs[0], outs[1]) and not np.array_equal(outs[0], outs[2])
    print("G3 same seed -> identical, different seed -> different")


LENS = [("psf 1.5", dict(psf_sigma_px=1.5)), ("ccm", dict(ccm=(1.1, 0.05, 0, 0.02, 1, 0, 0, -0.05, 0.85))),
        # 120 Hz: half_window 0.01 s is 1.2 periods, so a frame time off by half_window changes every row gain
        ("flicker rolling", dict(flicker_amp=0.4, flicker_hz=120, readout_s=0.02, rs_bands=4)),
        ("dirt + veil", dict(dirt_density=0.5, dirt_scatter=0.4)),
        ("all", dict(psf_sigma_px=1.5, ccm=(1.1, 0.05, 0, 0.02, 1, 0, 0, -0.05, 0.85), flicker_amp=0.4,
                     dirt_density=0.5, dirt_scatter=0.4))]


def test_lens_parity():
    """Camera-side DR on the device == CPU SimCamera (noise off), single-instant and rolling-shutter window."""
    model, data = board_scene()
    for label, kw in LENS:
        cpu, gpu = pair(model, CameraConfig(W, H, **P1, **kw))
        data.time = 0.0123
        if cpu.half_window > 0:  # empty ring first: single-instant frame at data.time - half_window
            mx, mean = diff(cpu.render(data), gpu.render(data))
            assert mx <= 2 + REMAP_SLACK and mean <= 0.1, (label, "empty ring")
            for i in range(30):
                data.time = 0.002 * i
                cpu.push(data)
                gpu.push(data)
        mx, mean = diff(cpu.render(data), gpu.render(data))
        print(f"G6 lens parity {label}: max |Δ| {mx} DN, mean {mean:.4f} DN")
        assert mx <= 2 + REMAP_SLACK and mean <= 0.1, label
    data.time = 0.0


def test_scene_on_gpu():
    from physicalai_sim2real_camera.scene_dr import Lamp, SceneConfig

    model, data = board_scene()
    scene = SceneConfig(light_gain=0.6, lamps=(Lamp(pos=(0.3, 0.3, 1.0), dir=(-0.3, -0.3, -1.0)),))
    cpu = mk(model, "cam", CameraConfig(W, H, **P1), scene=scene)
    gpu = mk(model, "cam", CameraConfig(W, H, **P1), device="cuda", scene=scene)
    mx, mean = diff(cpu.render(data), gpu.render(data))
    print(f"G6 scene parity: max |Δ| {mx} DN, mean {mean:.4f} DN")
    plain = mk(model, "cam", CameraConfig(W, H, **P1)).render(data)
    assert mx <= 1 + REMAP_SLACK and diff(cpu.render(data), plain)[1] > 1  # the scene really changed the frame


def test_window():
    model, data = moving_scene()
    cfg = CameraConfig(W, H, **P1, exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)
    cpu, gpu = pair(model, cfg)
    worst = (0, 0.0)
    for k in range(3):
        for _ in range(10):
            data.mocap_pos[0, 1] = 0.1 * data.time
            mujoco.mj_step(model, data)
            cpu.push(data)
            gpu.push(data)
        worst = max(worst, diff(cpu.render(data), gpu.render(data)))
    print(f"G4 window N4B8 parity: max |Δ| {worst[0]} DN, mean {worst[1]:.4f} DN")
    assert worst[0] <= 1 + REMAP_SLACK


def test_ae():
    model, data = board_scene()
    cpu, gpu = pair(model, CameraConfig(W, H, **P1, ae_tau_s=1.0))
    err = 0.0
    for _ in range(30):
        cpu.render(data)
        gpu.render(data)
        err = max(err, abs(cpu._ev - gpu._ev))
    print(f"G5 AE: ev {cpu._ev:+.4f} (cpu) vs {gpu._ev:+.4f} (gpu), max |Δev| {err:.2e}")
    assert err < 1e-3


def plane_scene():
    xml = """<mujoco><statistic extent="0.4" center="0 0 0"/>
<visual><global offwidth="640" offheight="480"/></visual><worldbody>
  <light pos="0 0 3" directional="true"/>
  <geom type="box" size="0.5 0.5 0.01" pos="0 0 -0.01"/>
  <camera name="cam" fovy="45" pos="0 0 1"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth():
    model, data = board_scene()
    cpu, gpu = pair(model, CameraConfig(W, H, **P1))
    _, dc = cpu.render(data, depth=True)
    _, dg = gpu.render(data, depth=True)
    assert dg.dtype == np.float32 and dg.shape == (H, W)
    valid = dc > 0
    assert np.allclose(dg[valid], dc[valid], atol=1e-6)
    print(f"G6 depth remap parity: max |Δ| {float(np.abs(dg - dc)[valid].max()):.1e} m")
    model, data = plane_scene()
    a, b = 0.001, 0.002
    _, clean = mk(model, "cam", CameraConfig(W, H, vignette=0.3), device="cuda").render(data, depth=True)
    _, noisy = mk(model, "cam", CameraConfig(W, H, vignette=0.3, depth_sigma_a=a, depth_sigma_b=b),
                         device="cuda").render(data, depth=True)
    on = (clean > 0) & (clean < 5)  # plane only (background sits at the 20 m far clip -> 0)
    z = float(clean[on].mean())
    std, want = float(np.std(noisy[on] - clean[on])), a + b * z * z
    print(f"G6 depth noise at z={z:.3f} m: std {std:.2e} vs a+b·z² {want:.2e}")
    assert abs(std / want - 1) < 0.15


def test_tensor():
    model, data = board_scene()
    g = mk(model, "cam", CameraConfig(W, H, **P1, **NOISE), device="cuda")
    rgb, d = g.render(data, depth=True, as_tensor=True)
    torch.cuda.synchronize()
    assert rgb.is_cuda and rgb.dtype == torch.uint8 and tuple(rgb.shape) == (H, W, 3)
    assert d.is_cuda and d.dtype == torch.float32 and tuple(d.shape) == (H, W)
    rgb2 = g.render(data, as_tensor=True)
    assert rgb2.data_ptr() != rgb.data_ptr()  # fresh tensors
    print("G7 as_tensor: CUDA uint8 (H,W,3) + float32 (H,W)")


try:
    from cuda.bindings import driver as _cu  # noqa: F401  (interop tests need cuda-python)
    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


def _shift_board(model, dx):
    boxes = model.geom_type == mujoco.mjtGeom.mjGEOM_BOX
    model.geom_pos[boxes, 0] += dx


def test_interop_spike():
    """GlInterop copy == Renderer.render() bit-exactly (both offsamples=0), 20 moved frames."""
    if not HAVE_CUDA_PYTHON:
        print("SKIP interop spike: cuda-python missing")
        return
    from physicalai_sim2real_camera.interop import GlInterop
    model, data = board_scene()
    model.vis.quality.offsamples = 0
    r, ref = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
    iop = GlInterop(r, "cuda")
    out = torch.empty((H, W, 3), dtype=torch.uint8, device="cuda")
    dout = torch.empty((H, W), dtype=torch.float32, device="cuda")
    for _ in range(20):
        _shift_board(model, 0.003)
        mujoco.mj_forward(model, data)
        ref.update_scene(data, "cam")
        r.update_scene(data, "cam")
        assert np.array_equal(iop.render_into(out).cpu().numpy(), ref.render())
        ref.enable_depth_rendering()
        assert np.array_equal(iop.depth_into(dout).cpu().numpy(), ref.render())
        ref.disable_depth_rendering()
    iop.close()
    r.close()
    ref.close()
    print("I1 interop copy == Renderer.render() (offsamples=0), rgb + depth, 20 moved frames: bit-exact")


def test_interop():
    if not HAVE_CUDA_PYTHON:
        print("SKIP interop: cuda-python missing")
        return
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    a = mk(model, "cam", cfg, device="cuda", interop=True)  # first: sets offsamples=0 for the rest
    b, c = mk(model, "cam", cfg, device="cuda"), mk(model, "cam", CameraConfig(W, H, **P1))
    assert a._gpu.interop is not None and model.vis.quality.offsamples == 0
    for _ in range(3):
        assert np.array_equal(a.render(data), b.render(data))
    ai = mk(model, "cam", CameraConfig(W, H, **P1), device="cuda", interop=True)
    mx, mean = diff(c.render(data), ai.render(data))
    print(f"I2 interop == non-interop cuda (noise on, bit); vs CPU (noise off): max |Δ| {mx} DN, mean {mean:.4f}")
    assert mx <= 1 + REMAP_SLACK
    (ra, da), (rb, db) = a.render(data, depth=True), b.render(data, depth=True)
    assert np.array_equal(ra, rb) and np.array_equal(da, db)
    print("I2 depth path with interop (PBO) == without (readPixels + upload)")
    dn = CameraConfig(W, H, **P1, **NOISE, depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0,
                      focus_m=1.0, aperture_m=0.004)
    a, b = mk(model, "cam", dn, device="cuda", interop=True), mk(model, "cam", dn, device="cuda")
    (ra, da), (rb, db) = a.render(data, depth=True), b.render(data, depth=True)
    assert np.array_equal(ra, rb) and np.array_equal(da, db)
    print("I2 depth noise + DoF with interop == without (bit)")
    ident = mk(model, "cam", CameraConfig(W, H), device="cuda", interop=True)
    assert np.array_equal(ident.render(data), _plain(model, data))
    print("I2 identity cfg with interop == plain render (same offsamples)")

    model, data = moving_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE, exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)
    a = mk(model, "cam", cfg, device="cuda", interop=True)
    b = mk(model, "cam", cfg, device="cuda")
    for _ in range(3):
        for _ in range(10):
            data.mocap_pos[0, 1] = 0.1 * data.time
            mujoco.mj_step(model, data)
            a.push(data)
            b.push(data)
        assert np.array_equal(a.render(data), b.render(data))
    print("I3 window N4B8 interop == non-interop cuda (noise on, bit)")


def timeit(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench():
    model, data = board_scene()
    base = CameraConfig(W, H, **P1, **NOISE)
    c, g = pair(model, base)
    pin, dev = g._gpu._pin, g._gpu._u8
    out = torch.empty((H, W, 3), dtype=torch.uint8, device="cuda")
    up_ms = timeit(lambda: dev.copy_(pin, non_blocking=True), 500)
    print(f"upload {pin.numel() / 1e6:.2f} MB pinned->GPU: {up_ms:.3f} ms, "
          f"download (H,W,3) uint8 .cpu(): {timeit(lambda: out.cpu(), 500):.3f} ms")
    r = mujoco.Renderer(model, *pin.shape[:2])
    print(f"plain render (margin size): {timeit(lambda: (r.update_scene(data, 'cam'), r.render())):.2f} ms")
    r.close()
    cases = [("single instant (phase 1)", {}, False), ("+DoF", dict(aperture_m=0.016, focus_m=0.5), False),
             ("+Bayer", dict(bayer=True), False), ("+depth noise", dict(depth_sigma_b=0.002), True)]
    print(f"{'case':28s} {'cpu ms':>7s} {'gpu ms':>7s} {'gpu tensor':>10s}")
    for name, kw, dep in cases:
        cfg = dataclasses.replace(base, **kw)
        c, g = pair(model, cfg)
        t = [timeit(lambda: c.render(data, depth=dep)), timeit(lambda: g.render(data, depth=dep)),
             timeit(lambda: g.render(data, depth=dep, as_tensor=True))]
        print(f"{name:28s} {t[0]:7.2f} {t[1]:7.2f} {t[2]:10.2f}")
        if t[1] > t[0]:
            print(f"WARNING: GPU slower than CPU for {name}")
    cfg = dataclasses.replace(base, exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)
    c, g = pair(model, cfg)
    for k in range(20):
        data.time = k * DT
        c.push(data)
        g.push(data)
    t = [timeit(lambda: c.render(data), 30), timeit(lambda: g.render(data), 30),
         timeit(lambda: g.render(data, as_tensor=True), 30)]
    print(f"{'window N4B8 (8/16 ms)':28s} {t[0]:7.2f} {t[1]:7.2f} {t[2]:10.2f}")


def bench_interop():
    """cuda vs cuda+interop (offsamples=0 for both). Run on an idle GPU."""
    if not HAVE_CUDA_PYTHON:
        print("SKIP interop bench: cuda-python missing")
        return
    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv"],
                         capture_output=True, text=True).stdout.strip())
    from physicalai_sim2real_camera.interop import GlInterop
    model, data = board_scene()
    base = CameraConfig(W, H, **P1, **NOISE)
    a = mk(model, "cam", base, device="cuda", interop=True)
    b = mk(model, "cam", base, device="cuda")
    g, r = b._gpu, b.renderer
    iop = GlInterop(r, "cuda")
    r.update_scene(data, "cam")
    raw = [timeit(lambda: (r.render(out=b._rgb), g._u8.copy_(g._pin, non_blocking=True)), 300),
           timeit(lambda: iop.render_into(g._u8), 300)]
    iop.close()
    print(f"render {g._u8.shape[1]}x{g._u8.shape[0]} + readPixels + upload: {raw[0]:.3f} ms, "
          f"render + interop copy: {raw[1]:.3f} ms")
    rows = [("single instant", a, b, data)]
    t = [timeit(lambda: b.render(data, depth=True, as_tensor=True), 50),
         timeit(lambda: a.render(data, depth=True, as_tensor=True), 50)]
    print(f"single + depth: cuda {t[0]:.2f} ms, +interop {t[1]:.2f} ms (as_tensor)")
    model, data = moving_scene()
    cfg = dataclasses.replace(base, exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)
    a = mk(model, "cam", cfg, device="cuda", interop=True)
    b = mk(model, "cam", cfg, device="cuda")
    for k in range(20):
        data.time = k * DT
        a.push(data)
        b.push(data)
    rows.append(("window N4B8", a, b, data))
    print(f"{'case':16s} {'cuda ms':>8s} {'+interop':>8s} (as_tensor)")
    for name, a, b, d in rows:
        t = [timeit(lambda: b.render(d, as_tensor=True), 50),
             timeit(lambda: a.render(d, as_tensor=True), 50)]
        print(f"{name:16s} {t[0]:8.2f} {t[1]:8.2f}")
        if t[1] > t[0]:
            print(f"WARNING: interop slower than readPixels+upload for {name}")


def edge_stats(out, clean, band, mn, mx):
    """(hole ratio, flying count, flying in [mn, mx]) of ``out`` inside ``band`` over the valid pixels of ``clean``."""
    valid = band & (clean > 0) & (clean < 5)
    holes = valid & (out == 0)
    fly = valid & (out != 0) & (out != clean)
    inside = bool(np.all((out[fly] >= mn[fly]) & (out[fly] <= mx[fly])))
    return holes.sum() / valid.sum(), fly.sum(), inside, valid.sum(), holes.sum()


def test_edge_band_torch():
    """edge_band_torch == edge_band (mx, mn, band) on synthetic arrays and on the clean two-depth render."""
    from physicalai_sim2real_camera.depth import depth_buffers, edge_band
    from physicalai_sim2real_camera.gpu import edge_band_torch
    for k in (1, 2, 3):
        rng = np.random.default_rng(k + 42)
        h, w = 37 + k * 5, 53 + k * 3
        z = rng.uniform(0.3, 5.0, (h, w)).astype(np.float32)
        z[10:20, 5:30] = 2.0  # a flat patch inside random values: its border is a step
        bufs = depth_buffers(h, w, edge=True)
        band = edge_band(z, k, bufs).copy()
        mx, mn, bt = (t.cpu().numpy() for t in edge_band_torch(torch.from_numpy(z).cuda(), k))
        assert np.array_equal(mx, bufs["mx"]) and np.array_equal(mn, bufs["mn"]), f"k={k}: dilate/erode differ"
        assert np.array_equal(bt, band), f"k={k}: band differs"
        const = np.full((20, 20), 1.5, np.float32)
        assert not edge_band_torch(torch.from_numpy(const).cuda(), k)[2].any()
    model, data = two_depth_scene()
    cfg = CameraConfig(W, H)  # clean depth, no distortion
    _, dc = mk(model, "cam", cfg).render(data, depth=True)
    _, dg = mk(model, "cam", cfg, device="cuda").render(data, depth=True)
    bufs = depth_buffers(H, W, edge=True)
    for k in (1, 2):
        bc = edge_band(dc, k, bufs).copy()
        bg = edge_band_torch(torch.from_numpy(dg).cuda(), k)[2].cpu().numpy()
        assert np.array_equal(bc, bg), f"k={k}: band of the CPU and GPU clean depth differ"
    print(f"G7 edge_band_torch == edge_band: synthetic k=1,2,3 (mx, mn, band bit-equal); two-depth clean render "
          f"band {bc.sum()} px (k=2) identical CPU/GPU")


def test_edge_holes_gpu():
    """Edge holes / flying pixels on two_depth_scene: CPU and GPU pipelines, per-path stats and CPU-vs-GPU agreement."""
    from physicalai_sim2real_camera.depth import depth_buffers, edge_band
    model, data = two_depth_scene()
    clean_cfg, cfg = CameraConfig(W, H), CameraConfig(W, H, depth_edge_hole_px=2)  # no dist, no axial noise
    _, clean = mk(model, "cam", clean_cfg).render(data, depth=True)
    bufs = depth_buffers(H, W, edge=True)
    band = edge_band(clean, 2, bufs).copy()
    mx, mn = bufs["mx"].copy(), bufs["mn"].copy()
    stats = {}
    for dev in ("cpu", "cuda"):
        _, out = mk(model, "cam", cfg, device=dev).render(data, depth=True)
        ratio, nfly, inside, nband, nholes = edge_stats(out, clean, band, mn, mx)
        outside = ~band & (clean > 0) & (clean < 5)
        assert np.array_equal(out[outside], clean[outside]), f"{dev}: pixels outside the band changed"
        assert 0.45 <= ratio <= 0.55, f"{dev}: hole ratio {ratio:.3f}"
        assert nfly >= 0.9 * (nband - nholes) and inside, f"{dev}: flying {nfly} of {nband - nholes}, in range {inside}"
        assert not np.array_equal(out, clean)
        stats[dev] = (ratio, nfly)
        print(f"G8 edge holes {dev}: band {nband} px, holes {ratio:.3f}, flying {nfly} (all within [mn, mx])")
    (rc, fc), (rg, fg) = stats["cpu"], stats["cuda"]
    assert abs(rg / rc - 1) < 0.2 and abs(fg / fc - 1) < 0.2, "CPU vs GPU edge statistics differ by > 20 %"

if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
        bench_interop()
    else:
        test_parity()
        test_dof_parity()
        test_noise()
        test_window()
        test_ae()
        test_depth()
        test_tensor()
        test_interop_spike()
        test_interop()
        test_edge_band_torch()
        test_edge_holes_gpu()
        print("OK")
    for c in CAMS:
        c.close()
