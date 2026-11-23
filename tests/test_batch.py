"""R0 + R1: shared Renderer and SimCameraBatch; ``--mem`` GPU memory per camera, ``--bench`` batch timing."""

import dataclasses
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import cv2
import mujoco
import numpy as np
from bench_batch import chunk_bytes, r1_mem_bound
from test_camera import BY, H, W, board_scene

from physicalai_sim2real_camera import CameraConfig, SimCamera, camera
from physicalai_sim2real_camera.camera import StateRing

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False
try:
    import cuda.bindings.driver  # noqa: F401
    HAVE_INTEROP = HAVE_CUDA
except ImportError:
    HAVE_INTEROP = False

P1 = dict(dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=-0.3,
          sigma_prnu=0.02, sigma_dsnu=0.005, black_level=0.02, gamma=2.2)
NOISE = dict(sigma_shot=0.02, sigma_read=0.005)
N = 8
MODES = [("cpu", False)] + [("cuda", False)] * HAVE_CUDA + [("cuda", True)] * HAVE_INTEROP


def cams(model, cfg, device, interop, shared, n=N):
    """n cameras of ``cfg`` with seeds seed+i (same seeds shared or not); shared: all use cams[0]'s renderer."""
    out = []
    for i in range(n):
        r = out[0].renderer if shared and out else None
        out.append(SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i),
                             device=device, interop=interop, renderer=r))
    return out


def close(cs):
    for c in reversed(cs):  # sharers before the owner
        c.close()


def test_equal():
    camera._RNG_SEEDED = True  # else the first render reseeds cv2 with cfg.seed; we seed per set below
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE, depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0)
    for dev, io in MODES:
        def run(cs):  # same render sequence for both sets: 2 rounds rgb, 2 rounds rgb+depth
            out = []
            for depth in (False, True):
                for _ in range(2):  # second round: generators/RNG advanced identically
                    cv2.setRNGSeed(123)  # CPU noise: cv2's global RNG, same render order in both sets
                    out.append([c.render(data, depth=depth) for c in cs])
            return out
        own = cams(model, cfg, dev, io, False)
        assert len({id(c.renderer) for c in own}) == N
        a = run(own)
        close(own)  # one set alive at a time: halves peak GPU memory
        sh = cams(model, cfg, dev, io, True)
        assert sh[0].owns_renderer and not any(c.owns_renderer for c in sh[1:])
        assert len({id(c.renderer) for c in sh}) == 1
        if io:
            assert sh[0]._gpu.interop is not None and len({id(c._gpu.interop) for c in sh}) == 1
        b = run(sh)
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                for u, v in (zip(x, y) if isinstance(x, tuple) else [(x, y)]):
                    assert np.array_equal(u, v)
        assert not np.array_equal(a[0][0], a[0][1])  # noise on, seeds differ per camera
        name = dev + "+interop" * io
        print(f"B1 {name}: {N} cams on 1 shared renderer == {N} own renderers (rgb and rgb+depth, noise on, bit)")
        if io:
            from physicalai_sim2real_camera.gpu import _INTEROP
            r = sh[0].renderer
            close(sh[1:])
            assert _INTEROP[r][1] == 1
            close(sh[:1])
            assert r not in _INTEROP
            print("B1 shared GlInterop refcount -> released with the last camera")
        else:
            close(sh)

def test_errors():
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1)
    owner = SimCamera(model, "cam", cfg)
    vis = (model.vis.global_.offwidth, model.vis.global_.offheight, model.vis.quality.offsamples)
    for bad in (dict(model=board_scene()[0], cfg=cfg), dict(model=model, cfg=CameraConfig(320, 240, **P1))):
        try:
            SimCamera(bad["model"], "cam", bad["cfg"], renderer=owner.renderer)
        except ValueError as e:
            print(f"B2 ValueError: {e}")
        else:
            raise AssertionError("mismatched shared renderer accepted")
    if HAVE_INTEROP and owner.renderer._mjr_context.offSamples != 0:
        try:
            SimCamera(model, "cam", cfg, device="cuda", interop=True, renderer=owner.renderer)
        except ValueError as e:
            print(f"B2 ValueError: {e}")
        else:
            raise AssertionError("multisampled renderer accepted for interop")
    sharer = SimCamera(model, "cam", cfg, renderer=owner.renderer)
    assert vis == (model.vis.global_.offwidth, model.vis.global_.offheight, model.vis.quality.offsamples)
    sharer.close()
    owner.render(data)  # sharer's close left the renderer open
    owner.close()
    print("B2 sharer leaves model.vis and the renderer alone")


def smi():
    time.sleep(0.5)
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.split()[0])


def per_cam(model, data, cfg, dev, io, shared, n=N):
    """(nvidia-smi MiB, torch.cuda.memory_allocated MiB) per extra camera beyond the first."""
    cuda = dev != "cpu"
    first = cams(model, cfg, dev, io, False, 1)
    first[0].render(data)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    m0, a0 = smi(), torch.cuda.memory_allocated() if cuda else 0
    rest = []
    for i in range(n):
        rest.append(SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + 1 + i), device=dev,
                              interop=io, renderer=first[0].renderer if shared else None))
        rest[-1].render(data)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    m1, a1 = smi(), torch.cuda.memory_allocated() if cuda else 0
    close(rest)
    close(first)
    gc.collect()
    return (m1 - m0) / n, (a1 - a0) / n / 2**20


def mem():
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    if HAVE_CUDA:
        torch.zeros(1, device="cuda")  # CUDA context before any measurement
    print(f"{'mode':<14}{'shared MiB/cam':>16}{'own MiB/cam':>14}{'torch MiB/cam':>15}")
    for dev, io in MODES:
        s, t = per_cam(model, data, cfg, dev, io, True)
        o, _ = per_cam(model, data, cfg, dev, io, False)
        print(f"{dev + '+interop' * io:<14}{s:>16.1f}{o:>14.1f}{t:>15.1f}")
        assert s <= t + 5, f"{dev}: shared renderer costs {s:.1f} MiB/camera > torch {t:.1f} + 5"


# ── R1 unit tests: pure helpers of batch.py (no rendering) ─────────────
def test_contiguous_runs():
    """``contiguous_runs``: maximal consecutive ranges, in order, covering the input; empty and singleton cases."""
    from physicalai_sim2real_camera.batch import contiguous_runs
    assert contiguous_runs(np.array([], np.int64)) == []
    assert contiguous_runs(np.array([5])) == [(5, 6)]
    assert contiguous_runs(np.arange(8)) == [(0, 8)]
    assert contiguous_runs(np.array([0, 1, 15])) == [(0, 2), (15, 16)]
    assert contiguous_runs(np.array([3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])) == [(3, 6), (7, 15)]
    assert contiguous_runs(np.array([0, 2, 4])) == [(0, 1), (2, 3), (4, 5)]
    sel = np.array([0, 1, 15])
    assert np.concatenate([np.arange(lo, hi) for lo, hi in contiguous_runs(sel)]).tolist() == sel.tolist()
    print("contiguous_runs OK")


def test_chunk_ranges():
    from physicalai_sim2real_camera.batch import chunk_ranges
    assert chunk_ranges(64, 16) == [(0, 16), (16, 32), (32, 48), (48, 64)]
    assert chunk_ranges(8, 16) == [(0, 8)]
    assert chunk_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert chunk_ranges(1, 1) == [(0, 1)]
    assert chunk_ranges(0, 16) == []
    print("U1 chunk_ranges: full, partial, single, empty")


def test_camera_seeds():
    from physicalai_sim2real_camera.batch import camera_seeds
    assert camera_seeds(100, 3, 2) == [102, 103, 104]
    assert camera_seeds(0, 1, 0) == [0]
    # determinism rule: batch camera i == SimCamera(cfg.seed = seed_base + i), which adds cam_id itself
    seed_base, cam_id = 7, 1
    assert camera_seeds(seed_base, 5, cam_id) == [8, 9, 10, 11, 12]  # (7 + i) + 1, by hand
    print("U2 camera_seeds: seed_base + i + cam_id")


def test_generator_seed():
    from physicalai_sim2real_camera.batch import generator_seed
    a, b, c = generator_seed(5, 1), generator_seed(5, 2), generator_seed(6, 1)
    assert len({a, b, c}) == 3  # RGB / depth streams and neighbouring cameras are distinct
    assert all(0 <= x < 2**32 for x in (a, b, c))
    assert generator_seed(5, 1) == a  # pure: same input, same seed
    # same derivation as GpuPipeline: SeedSequence([s, stream]).generate_state(1)[0]
    assert a == int(np.random.SeedSequence([5, 1]).generate_state(1)[0])
    print("U3 generator_seed: distinct per stream/camera, 32-bit, deterministic")


def test_dof_groups():
    from physicalai_sim2real_camera.batch import dof_groups
    g = dof_groups([0.3, 5.0, 40.0, 5.0, 33.0, 0.5, 0.49])
    assert g == {5.0: [1, 3], 32.0: [2, 4], 0.5: [5]}  # < 0.5 skipped, > 32 clamped and merged
    assert list(g) == [5.0, 32.0, 0.5]  # first-seen order
    assert dof_groups([0.1, 0.2]) == {}
    assert dof_groups([2.0, 2.0, 2.0]) == {2.0: [0, 1, 2]}  # identical scenes: one batched cascade
    assert dof_groups([10.0], cap=8.0, min_c=1.0) == {8.0: [0]}
    print("U4 dof_groups: skip < 0.5 px, clamp at 32 px, one group per blur size")


def test_to_device():
    from physicalai_sim2real_camera.batch import to_device
    a = np.arange(12, dtype=np.int64).reshape(3, 4)[:, ::2]  # non-contiguous view
    torch.cuda.set_sync_debug_mode("error")  # a pageable .to(dev) raises here
    try:
        t = to_device(a, "cuda")
        f = to_device(np.array([0.5, 2.0], np.float32), "cuda")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert t.device.type == "cuda" and t.dtype == torch.int64 and f.dtype == torch.float32
    assert np.array_equal(t.cpu().numpy(), a) and f.tolist() == [0.5, 2.0]
    print("U5 to_device: values/dtype kept, non-contiguous ok, no stream sync (the point of the helper)")


# ── R1: SimCameraBatch ────────────────────────────────────────────────
def shifted_datas(model, n, step=0.01):
    """n MjData of one model whose camera poses differ (cam_pos x + i*step): per-env scenes."""
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    p0, out = model.cam_pos[cam].copy(), []
    for i in range(n):
        model.cam_pos[cam, 0] = p0[0] + i * step
        out.append(mujoco.MjData(model))
        mujoco.mj_forward(model, out[-1])
    model.cam_pos[cam] = p0
    return out


def test_tone_fused():
    """Contract test: ``tone_fused`` (eager, CPU) == the ``_finish`` chain re-typed here, shapes/dtype, noise on/off."""
    from physicalai_sim2real_camera.batch import tone_fused
    g = torch.Generator().manual_seed(0)
    w = torch.rand((2, 3, 5, 7), generator=g) * 1.4 - 0.1  # some < 0 and > 1
    add = torch.rand((2, 1, 5, 7), generator=g) * 0.05
    nz = torch.randn((2, 3, 5, 7), generator=g)
    ss, sr, gamma = 0.02, 0.005, 2.2
    for noise in (None, nz):
        x = w + add
        if noise is not None:
            s = torch.clamp(x, min=0).mul(ss * ss).add(sr * sr).sqrt_().mul(noise)
            x = x + s
        ref = x.clamp(0, 1).pow(1 / gamma).mul(255).round().permute(0, 2, 3, 1).to(torch.uint8)
        out = tone_fused(w, add, noise, ss * ss, sr * sr, 1 / gamma)
        assert out.shape == (2, 5, 7, 3) and out.dtype == torch.uint8
        assert torch.equal(out, ref)
    print("tone_fused == eager chain (CPU, noise on/off)")


def test_batch_compile():
    """``compile=True`` vs eager: colour <= 1 DN on <= 1e-4 of pixels, depth bit-equal; AE/Bayer stay eager, bit-equal.

    Cases: plain (P1+NOISE, depth), window mode (blur + rolling shutter), AE fallback, Bayer fallback;
    chunk=3 over N=8 exercises k = 3 and k = 2 (and the k = 1 variant is compiled in ``__init__``).
    """
    from test_gpu import two_depth_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    plain = dict(**P1, **NOISE, depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0)
    cases = [("plain+depth", two_depth_scene, plain, True, False),
             ("window N4B8", board_scene, dict(**P1, **NOISE, exposure_s=0.008, readout_s=0.016), False, False),
             ("AE fallback", board_scene, dict(**P1, **NOISE, ae_tau_s=0.1), False, True),
             ("Bayer fallback", board_scene, dict(**P1, **NOISE, bayer=True), False, True)]
    for name, scene, kw, depth, exact in cases:
        model, _ = scene()
        cfg = CameraConfig(W, H, **kw)
        datas = shifted_datas(model, N)
        outs = []
        for comp in (False, True):
            b = SimCameraBatch(model, "cam", cfg, N, interop=HAVE_INTEROP, chunk=3, compile=comp)
            assert b._fused == (comp and not exact)
            for _ in range(2):
                if b.window:
                    for i in range(N):
                        b.rings[i].push(datas[i])
                for i in range(N):
                    b.render_into(i, datas[i], depth=depth)
                slots = torch.from_numpy(b.process(depth=depth)).cuda()
            cams_ = torch.arange(N, device="cuda")
            outs.append((b.rgb[cams_, slots].int().cpu(), b.depth[cams_, slots].cpu() if depth else None))
            b.close()
        d = (outs[0][0] - outs[1][0]).abs()
        frac = (d > 0).float().mean().item()
        if exact:
            assert d.max().item() == 0, f"{name}: must stay eager (bit-equal)"
        else:
            assert d.max().item() <= 1 and frac <= 1e-4, f"{name}: max {d.max().item()} DN, frac {frac:.2e}"
        if depth:
            assert torch.equal(outs[0][1], outs[1][1]), f"{name}: depth must be bit-equal (eager depth tail)"
        print(f"compile=True vs eager, {name}: max {d.max().item()} DN, frac != {frac:.1e}"
              f"{', depth bit-equal' if depth else ''}")


def test_batch_steady_state_kernels():
    """After a vsensor-style warm-up, non-contiguous mixed-slot ``process`` calls launch no new kernel and
    grow no memory (a kernel the warm-up never ran costs ~20 ms of lazy module load on its first launch).

    Three modes: single-instant, window (N4B8) and window + depth noise. Warm-up = ``ingest`` all cameras + one
    ``process`` (same slot).
    Then cameras [0, 1, 15] and [3..12] with mixed slots inside the contiguous runs (some cameras advanced one
    frame ahead first, outside the profile) and [2, 5, 9] (k = 1 runs): kernel
    names from
    ``torch.profiler`` must be a subset of the warm-up's, ``memory_reserved`` must not change (the caching
    allocator serves any temporary from blocks the warm-up already freed), and the allocator peak must not exceed
    the warm-up's. The window colour path (the virtual sensor's) allocates nothing; the instant path keeps
    ``grid_sample``'s output temporary (no ``out=``) and the depth-noise path its masks and noise draws.
    """
    from torch.profiler import ProfilerActivity, profile

    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = board_scene()
    n = 16

    def profiled(fn):  # -> (fn's result, CUDA kernel names launched, allocator peak above the pre-call baseline)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = fn()
            torch.cuda.synchronize()
        names = {e.key for e in prof.key_averages() if e.device_type.name == "CUDA"}
        return out, names, torch.cuda.max_memory_allocated() - base

    depth_kw = dict(depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0, depth_edge_hole_px=2)
    modes = (("instant", dict(**P1, **NOISE), False),
             ("window", dict(**P1, **NOISE, exposure_s=0.008, readout_s=0.016), False),
             ("window+depth", dict(**P1, **NOISE, exposure_s=0.008, readout_s=0.016, **depth_kw), True))
    for name, kw, depth in modes:
        cfg = CameraConfig(W, H, **kw)
        b = SimCameraBatch(model, "cam", cfg, n, renderer=False, interop=False, chunk=16, ring=4)
        raw = torch.zeros((b.n_sub, b.H_r, b.W_r, 3), dtype=torch.uint8, device="cuda")
        raw = raw if b.window else raw[0]
        dz = torch.full((b.H_r, b.W_r), 2.0, dtype=torch.float32, device="cuda") if depth else None

        def feed(cams):
            for i in cams:
                b.ingest(i, raw, dz)

        feed(range(n))
        _, warm, warm_peak = profiled(lambda: b.process(depth=depth))  # warm-up: all cameras, slot 0
        reserved = torch.cuda.memory_reserved()
        # (advance-alone cameras, then the batch): the advanced cameras sit one slot ahead, so the contiguous runs
        # [0, 1] and [3..12] mix slots and take _store's per-camera branch; [2, 5, 9] are k = 1 runs
        for ahead, cams in (([0], [0, 1, 15]), ([4, 7, 8], list(range(3, 13))), ([], [2, 5, 9])):
            if ahead:  # set-up step, outside the profile
                feed(ahead)
                b.process(depth=depth)
            feed(cams)
            slots, names, pk = profiled(lambda: b.process(depth=depth))
            new = names - warm
            if ahead:  # the contiguous run starting at the lowest camera must really have mixed slots
                run = slots[min(cams):min(cams) + 2]
                assert run[0] != run[1], f"test setup: run slots {run.tolist()} should differ"
            assert not new, f"{name}: cameras {cams} launched kernels the warm-up never ran: {sorted(new)[:4]}"
            assert torch.cuda.memory_reserved() == reserved, f"{name}: cameras {cams}: reserved memory grew"
            assert pk <= warm_peak, (f"{name}: cameras {cams}: peak {pk / 2**20:.1f} MiB > warm-up "
                                     f"{warm_peak / 2**20:.1f}")
            if b.window and not depth:
                assert pk <= 1 << 20, f"window: cameras {cams}: process() allocated {pk / 2**20:.1f} MiB of temporaries"
        b.close()
        print(f"steady-state kernels ({name}): {len(warm)} kernel names in warm-up (peak {warm_peak / 2**20:.1f} MiB), "
              f"non-contiguous mixed-slot runs: no new kernel, reserved unchanged, peak <= warm-up")


def test_batch_equal():
    from test_gpu import two_depth_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    dof = dict(**P1, **NOISE, aperture_m=0.016, focus_m=0.5, depth_sigma_a=0.002, depth_sigma_b=0.01,
               depth_max_m=10.0, ae_tau_s=0.1, awb_tau_s=0.1)
    # (name, scene, cfg kwargs, depth, steps, chunk, interop); chunk=3 over n=8: chunks 3+3+2
    variants = [("P1+NOISE", board_scene, dict(**P1, **NOISE), False, 2, 16, HAVE_INTEROP),
                ("P1+NOISE chunk=3", board_scene, dict(**P1, **NOISE), False, 2, 3, HAVE_INTEROP),
                ("P1+NOISE+DoF+depth noise+AE/AWB", two_depth_scene, dof, True, 3, 16, HAVE_INTEROP),
                 ("P1+NOISE+DoF+depth noise+AE/AWB chunk=3", two_depth_scene, dof, True, 3, 3, HAVE_INTEROP),
                 ("P1+NOISE+DoF+depth noise+AE/AWB readPixels", two_depth_scene, dof, True, 2, 3, False),
                 ("P1+NOISE+mount vibration", board_scene,
                  dict(**P1, **NOISE, mount_vib_rot_deg=1.0, mount_vib_trans_m=0.003), False, 2, 3, HAVE_INTEROP),
                 ("P1+NOISE+edge holes", two_depth_scene,
                  dict(**P1, **NOISE, aperture_m=0.016, focus_m=0.5, depth_sigma_a=0.002, depth_sigma_b=0.01,
                       depth_max_m=10.0, depth_edge_hole_px=2), True, 2, 3, HAVE_INTEROP)]
    for name, scene, kw, depth, steps, chunk, io in variants:
        model, _ = scene()
        cfg = CameraConfig(W, H, **kw)
        datas = shifted_datas(model, N)
        b = SimCameraBatch(model, "cam", cfg, N, interop=io, chunk=chunk)
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                          interop=io, renderer=b.renderer) for i in range(N)]
        for _ in range(steps):  # later steps: generators and AE state advanced identically
            for i in range(N):
                b.render_into(i, datas[i], depth=depth)
            slots = torch.from_numpy(b.process(depth=depth)).cuda()
            cams_ = torch.arange(N, device="cuda")
            rgb = b.rgb[cams_, slots].cpu().numpy()
            dep = b.depth[cams_, slots].cpu().numpy() if depth else None
            for i, c in enumerate(refs):
                r = c.render(datas[i], depth=depth)
                r, d = r if depth else (r, None)
                assert np.array_equal(rgb[i], r), f"{name}: camera {i} rgb differs"
                if depth:
                    assert np.array_equal(dep[i], d), f"{name}: camera {i} depth differs"
            if cfg.ae_tau_s:
                assert all(b.ev[i] == c._ev and tuple(b.wb[i]) == c._wb for i, c in enumerate(refs))
        if depth and cfg.depth_edge_hole_px:
            ratio = edge_hole_ratio(model, datas[0], dep[0], cfg.depth_edge_hole_px, cfg.dist)
            assert ratio > 0.35, f"{name}: hole ratio {ratio:.2f} in the edge band"
        assert not np.array_equal(rgb[0], rgb[1])
        close(refs)
        b.close()
        print(f"R1-1 {name}: batch n={N} camera i == SimCamera(seed+i, cuda{'+interop' * io}), "
              f"{steps} steps{', rgb+depth' if depth else ''}: bit-equal")


def test_batch_lens_and_scene():
    """Camera-side DR + scene in a batch: camera i == SimCamera(seed + i, cuda) within 1 DN (conv algorithms may
    differ with the batch size), single-instant and rolling-shutter window; lens dirt differs per camera."""
    from test_gpu import LENS

    from physicalai_sim2real_camera.batch import SimCameraBatch
    from physicalai_sim2real_camera.scene_dr import Lamp, SceneConfig

    scene = SceneConfig(light_gain=0.7, lamps=(Lamp(pos=(0.3, 0.3, 1.0), dir=(-0.3, -0.3, -1.0)),))
    for label, kw in LENS + [("scene", {})]:
        model, _ = board_scene()
        cfg = CameraConfig(W, H, **P1, **kw)
        sc = scene if label == "scene" else None
        datas = shifted_datas(model, N)
        b = SimCameraBatch(model, "cam", cfg, N, interop=False, chunk=3, scene=sc)
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda", scene=sc,
                          renderer=b.renderer) for i in range(N)]
        for step in range(2):
            for i in range(N):
                datas[i].time = 0.0123 + 0.007 * i + 0.033 * step  # different flicker phase per camera
                if b.window:
                    b.rings[i].push(datas[i])
                    refs[i].push(datas[i])
                b.render_into(i, datas[i])
            slots = torch.from_numpy(b.process()).cuda()
            rgb = b.rgb[torch.arange(N, device="cuda"), slots].cpu().numpy()
            for i, c in enumerate(refs):
                d = np.abs(rgb[i].astype(int) - c.render(datas[i]).astype(int))
                assert d.max() <= 1 and d.mean() < 0.01, (label, i, d.max(), d.mean())
        if b._T is not None:  # per-camera dirt maps, as SimCamera(seed + i)
            assert not torch.equal(b._T[0], b._T[1])
        close(refs)
        b.close()
        print(f"R1-lens {label}: batch n={N} == SimCamera(seed+i, cuda) within 1 DN")
    for make, want in ((lambda: SimCameraBatch(model, "cam", CameraConfig(W, H, jpeg_quality=80), 2, interop=False),
                        "CPU"), (None, "capture_t")):
        try:
            if make is not None:
                make()
            else:
                with SimCameraBatch(model, "cam", CameraConfig(W, H, flicker_amp=0.3), 1, renderer=False) as ing:
                    ing.ingest(0, torch.zeros((ing.H_r, ing.W_r, 3), dtype=torch.uint8, device="cuda"))
        except ValueError as e:
            assert want in str(e), e
        else:
            raise AssertionError(f"no ValueError ({want})")


def test_batch_ingest(n=4):
    """R5: ingest-only batch (renderer=False, no GL) fed the raw renders == render_into batch, bit for bit."""
    from test_gpu import two_depth_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    dof = dict(**P1, **NOISE, aperture_m=0.016, focus_m=0.5, depth_sigma_a=0.002, depth_sigma_b=0.01,
               depth_max_m=10.0, ae_tau_s=0.1, awb_tau_s=0.1)
    for name, scene, kw, depth in [("P1+NOISE", board_scene, dict(**P1, **NOISE), False),
                                   ("P1+NOISE+DoF+depth noise+AE/AWB", two_depth_scene, dof, True)]:
        model, _ = scene()
        cfg = CameraConfig(W, H, **kw)
        datas = shifted_datas(model, n)
        ref = SimCameraBatch(model, "cam", cfg, n, interop=False)
        ing = SimCameraBatch(model, "cam", cfg, n, renderer=False)
        assert ing.renderer is None and ing.interop is None and ing.base.renderer is None
        cams_ = torch.arange(n, device="cuda")
        for _ in range(3):
            for i in range(n):
                ref.render_into(i, datas[i], depth=depth)
                rgb, d = ref.base._raw(datas[i], depth)  # what the GL-only sim process ships
                ing.ingest(i, torch.from_numpy(rgb).cuda(), None if d is None else torch.from_numpy(d).cuda())
            s1, s2 = ref.process(depth=depth), ing.process(depth=depth)
            assert np.array_equal(s1, s2)
            sl = torch.from_numpy(s1).cuda()
            assert torch.equal(ref.rgb[cams_, sl], ing.rgb[cams_, sl]), name
            if depth:
                assert torch.equal(ref.depth[cams_, sl], ing.depth[cams_, sl]), name
        assert np.array_equal(ref.ev, ing.ev) and np.array_equal(ref.wb, ing.wb)
        raw = torch.from_numpy(ref.base._raw(datas[0], depth)[0]).cuda()
        errs = [lambda: ing.render_into(0, datas[0]),
                lambda: ing.ingest(0, raw[:-1]),
                lambda: ing.ingest(0, raw.cpu()),
                lambda: ing.ingest(0, raw.float()),
                lambda: ing.ingest(0, raw, torch.zeros(raw.shape[:2], dtype=torch.float64, device="cuda")),
                lambda: ing.base.render(datas[0])]
        if depth:
            errs.append(lambda: ing.ingest(0, raw))  # DoF without depth
        for f in errs:
            try:
                f()
            except ValueError as e:
                print(f"R5-ingest err: {e}")
            else:
                raise AssertionError("bad ingest/render_into accepted")
            assert not ing.pending(0)  # a rejected ingest writes nothing
        if not depth:
            ing.ingest(0, raw)
            assert ing.pending(0) and not ing.pending(1)
            try:
                ing.ingest(0, raw)
            except ValueError as e:
                print(f"R5-ingest err twice: {e}")
            else:
                raise AssertionError("second ingest before process accepted")
        ref.close()
        ing.close()
        print(f"R5 ingest {name}: renderer=False batch n={n} fed raw frames == render_into batch, 3 steps"
              f"{', rgb+depth' if depth else ''}: bit-equal")
    if HAVE_INTEROP:
        model, data = board_scene()
        with SimCameraBatch(model, "cam", CameraConfig(W, H, **P1), 1, interop=True) as b:
            try:
                b.ingest(0, torch.zeros((b.H_r, b.W_r, 3), dtype=torch.uint8, device="cuda"))
            except ValueError as e:
                print(f"R5-ingest err interop: {e}")
            else:
                raise AssertionError("ingest into an interop batch accepted")


def ring_run(K, n=4, steps=30, lat=2, p_drop=0.3, depth=False):
    """Hook-driven stream: camera i fires with prob 1-p_drop, delivers a capture `lat` steps later and
    holds it. Count held frames whose slot content != the per-camera reference render of that capture."""
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    datas = shifted_datas(model, n)
    rng = np.random.default_rng(0)
    bad = total = 0
    with SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, ring=K) as b:
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                          interop=HAVE_INTEROP, renderer=b.renderer) for i in range(n)]
        flight, held = [[] for _ in range(n)], [None] * n  # (deliver step, handle, expected rgb[, depth])
        for t in range(steps):
            fired = rng.random(n) >= p_drop
            for i in np.flatnonzero(fired):  # dropped cameras neither render nor advance (ref too)
                h = b.render_into(i, datas[i], depth=depth)
                ref = refs[i].render(datas[i], depth=depth)
                flight[i].append((t + lat, h, ref if depth else (ref,)))
            b.process(depth=depth, mask=fired if t % 2 else None)  # explicit mask == default
            for i in range(n):
                while flight[i] and flight[i][0][0] <= t:
                    held[i] = flight[i].pop(0)
                if held[i] is not None:
                    _, h, ref = held[i]
                    total += 1
                    ok = np.array_equal(b.rgb[h].cpu().numpy(), ref[0])
                    if depth:
                        ok &= np.array_equal(b.depth[h].cpu().numpy(), ref[1])
                    bad += not ok
        close(refs)
    return bad, total


def test_batch_ring():
    # delivery lat=2 steps -> at most 3 frames live (held + 2 in flight): ring 3 is the smallest that works
    runs = {K: ring_run(K, depth=K == 4) for K in (4, 3, 2, 1)}
    print("R1-ring held slot content != reference render of that capture: " + "; ".join(
        f"ring={K}{' (rgb+depth)' if K == 4 else ''} {bad}/{total}" for K, (bad, total) in runs.items()))
    assert runs[4][1] > 0 and runs[4][0] == 0 and runs[3][0] == 0 and runs[2][0] > 0 and runs[1][0] > 0


def stream_run(ring, n=4, steps=300):
    """CameraStreamBatch drives the batch: hook = render_into (+ reference render of capture k)."""
    from physicalai_sim2real_camera import CameraStreamBatch
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.1)
    datas = shifted_datas(model, n)
    expected, bad, total = {}, 0, 0
    with SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, ring=ring) as b:
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                          interop=HAVE_INTEROP, renderer=b.renderer) for i in range(n)]

        def hook(i, data, k):
            expected[(i, k)] = refs[i].render(data)
            return b.render_into(i, data)
        sb = CameraStreamBatch(cfg, n, render=hook)
        for t in range(steps):
            for d in datas:
                d.time = t / 500
            ready, _, seq, hs, new, fired = sb.step(datas)
            b.process(mask=fired)
            for i in np.flatnonzero(ready & (new | (t % 25 == 0))):  # each delivery + held re-checks
                total += 1
                bad += not np.array_equal(b.rgb[hs[i]].cpu().numpy(), expected[(i, int(seq[i]))])
        close(refs)
        K = b.ring
    return bad, total, K


def test_batch_stream_ring():
    from physicalai_sim2real_camera import ring_depth
    bad, total, K = stream_run(None)  # default ring = ring_depth(cfg)
    bad1, total1, _ = stream_run(1)
    print(f"R1-stream CameraStreamBatch(latency 50 ms, jitter, 10 % drops) -> SimCameraBatch ring=ring_depth={K}: "
          f"delivered slot != reference render of that capture {bad}/{total}; ring=1 {bad1}/{total1}")
    assert K == ring_depth(CameraConfig(W, H, latency_s=0.05, jitter_k=2, jitter_theta=0.005))
    assert total > 0 and bad == 0 and bad1 > 0


def test_batch_errors():
    from physicalai_sim2real_camera.batch import SimCameraBatch
    from physicalai_sim2real_camera.gpu import _INTEROP
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    with SimCameraBatch(model, "cam", cfg, 3, interop=HAVE_INTEROP) as b:
        assert b.render_into(0, data, depth=True) == (0, 0) and b.render_into(1, data) == (1, 0)
        try:
            b.process(depth=True)
        except ValueError as e:
            assert "[1]" in str(e)
            print(f"R1-err depth: {e}")
        else:
            raise AssertionError("process(depth=True) without depth render accepted")
        try:
            b.process(mask=np.array([True, False, True]))
        except ValueError as e:
            assert "[2]" in str(e)
            print(f"R1-err mask: {e}")
        else:
            raise AssertionError("mask over an unrendered camera accepted")
        for bad_mask in ([0, 2], np.array([1, 0, 1]), np.array([True, False])):  # ints are not a mask
            try:
                b.process(mask=bad_mask)
            except ValueError as e:
                print(f"R1-err mask type: {e}")
            else:
                raise AssertionError(f"mask {bad_mask!r} accepted")
        try:
            b.render_into(0, data)
        except ValueError as e:
            print(f"R1-err twice: {e}")
        else:
            raise AssertionError("second render_into before process accepted")
        assert b.process().tolist() == [0, 0, -1]  # camera 2 not rendered: untouched
        assert b.ring == 2 and b.render_into(0, data) == (0, 1)  # default ring_depth: 0 latency -> 2
    assert b._closed
    if HAVE_INTEROP:
        owner = SimCamera(model, "cam", cfg, device="cuda", interop=True)
        r = owner.renderer
        b = SimCameraBatch(model, "cam", cfg, 2, interop=True, renderer=r)
        assert _INTEROP[r][1] == 2 and b.interop is owner._gpu.interop
        b.close()
        b.close()  # idempotent
        assert _INTEROP[r][1] == 1
        owner.render(data)  # batch close left the owner's renderer and interop alive
        owner.close()
        assert r not in _INTEROP
        print("R1-err close(): shared GlInterop refcount 2 -> 1 (twice: idempotent) -> released by owner")


def kernels(fn):
    from torch.profiler import DeviceType, ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == DeviceType.CUDA
               and not any(k in e.name for k in ("Memcpy", "Memset", "memcpy", "memset")))


def test_batch_launches():
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    counts = {}
    for n in (8, 16, 64):
        b = SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, chunk=16)
        render = lambda: [b.render_into(i, data) for i in range(n)]  # noqa: E731
        render()
        b.process()  # warm-up (cuDNN/allocator)
        kr, kp = kernels(render), kernels(b.process)
        counts[n] = kp
        print(f"R1-2 n={n:<3} chunk=16: render_into {kr} kernels, process {kp} kernels "
              f"({n} per-camera noise normal_ + {kp - n} in {-(-n // 16)} chunk launch set(s))")
        if HAVE_INTEROP:
            assert kr == 0
        b.close()
    ratio = counts[64] / counts[16]
    print(f"R1-2 launches n=64 / n=16 (4 chunks / 1 chunk) = {ratio:.2f} (<= 4.4); n=64 / n=8 = "
          f"{counts[64] / counts[8]:.2f}")
    assert ratio <= 4.4


def test_batch_no_sync():
    """process() without AE/DoF enqueues only: no host-device stream sync (contiguous and gathered chunks)."""
    # the AE scale / AWB gain uploads only run with AE on, which syncs by design (.tolist() of the means);
    # their values are covered by test_batch_equal
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, data = board_scene()
    n = 8
    b = SimCameraBatch(model, "cam", CameraConfig(W, H, **P1, **NOISE), n, interop=HAVE_INTEROP, chunk=3)
    every_other = np.arange(n) % 2 == 1
    try:
        for mask in (np.ones(n, bool), every_other):  # views vs rows gathered by an index tensor
            for i in np.flatnonzero(mask):
                b.render_into(i, data)
            torch.cuda.set_sync_debug_mode("error")  # the first process() of a fresh batch included
            try:
                b.process(mask=mask)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    finally:
        b.close()
    print("R1-4 process(): no stream sync (sync_debug_mode=error) for contiguous and gathered chunks")


def test_batch_mem(n=64):
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, data = board_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE)
    gc.collect()
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    b = SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, chunk=16)
    for i in range(n):
        b.render_into(i, data)
    b.process()
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - a0
    Hr, Wr = b.H_r, b.W_r
    cb = chunk_bytes(b)
    bound = r1_mem_bound(cb, n, Hr, Wr, H, W, b.ring)
    per = (bound / 1.1 - cb) / n
    print(f"R1-3 n={n} {W}x{H} ring={b.ring} (render {Wr}x{Hr}): allocated {used / 2**20:.1f} MiB <= bound "
          f"{bound / 2**20:.1f} MiB = 1.1 x (chunk work+maps {cb / 2**20:.1f} + {n} x {per / 2**20:.2f} MiB)")
    print(f"      per camera actually: _u8s {b._u8s[0].nbytes / 2**20:.2f} + rgb {b.rgb[0].nbytes / 2**20:.2f}"
          f" + FPN _mul/_add {(b._mul[0].nbytes + b._add[0].nbytes) / 2**20:.2f} MiB")
    assert used <= bound
    b.close()


# ── W1/W2: window mode in SimCameraBatch ──────────────────────────────
WIN = dict(exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)  # N4B8: 11 grid-snapped renders


def window_datas(model, n):
    """n MjData of the moving scene with different start states (camera x offset, board phase below)."""
    out = []
    for i in range(n):
        d = mujoco.MjData(model)
        d.qvel[0] = 0.2
        d.qpos[0] = 0.01 * i
        mujoco.mj_forward(model, d)
        out.append(d)
    return out


def window_step(model, datas, *rings):
    """One sim step of every env (board moves along y, phase per env) and a push into each ring list."""
    for i, d in enumerate(datas):
        d.mocap_pos[0, 1] = BY + 0.1 * d.time + 0.02 * i
        mujoco.mj_step(model, d)
        for rs in rings:
            rs[i].push(d)


def test_batch_window_equal():
    from test_temporal import moving_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    variants = [  # (name, cfg kwargs, depth, chunk, interop)
        ("blur N4", dict(**P1, **NOISE, exposure_s=0.008, blur_samples=4), False, 3, HAVE_INTEROP),
        ("RS B8", dict(**P1, **NOISE, readout_s=0.016, rs_bands=8), True, 3, HAVE_INTEROP),
        ("N4B8 +AE", dict(**P1, **NOISE, **WIN, ae_tau_s=0.1, awb_tau_s=0.1), True, 3, HAVE_INTEROP),
        ("N4B8 no maps", dict(**NOISE, **WIN, sigma_prnu=0.02, black_level=0.02), True, 16, HAVE_INTEROP),
        ("blur N4 no maps", dict(**NOISE, exposure_s=0.008, blur_samples=4, sigma_prnu=0.02), False, 3, HAVE_INTEROP),
        ("RS B8 no maps", dict(**NOISE, readout_s=0.016, rs_bands=8, sigma_prnu=0.02), True, 3, HAVE_INTEROP),
        ("N4B8 +DoF", dict(**P1, **NOISE, **WIN, aperture_m=0.016, focus_m=0.5), True, 3, HAVE_INTEROP),
        ("N4B8 readPixels", dict(**P1, **NOISE, **WIN, ae_tau_s=0.1, awb_tau_s=0.1), True, 3, False),
        ("N4B8 no noise", dict(**P1, **WIN), False, 3, HAVE_INTEROP),
    ]
    for name, kw, depth, chunk, io in variants:
        model, _ = moving_scene()
        cfg = CameraConfig(W, H, **kw)
        datas = window_datas(model, N)
        b = SimCameraBatch(model, "cam", cfg, N, interop=io, chunk=chunk)
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                          interop=io, renderer=b.renderer) for i in range(N)]
        assert b.window and b.n_sub == len(b.base._plan)
        if "N4B8" in name:
            assert b.n_sub == 11
        cams_t = torch.arange(N, device="cuda")
        for frame in range(3):
            for _ in range(12):  # 24 ms between frames: more than one ring window
                window_step(model, datas, b.rings, refs)
            for i in range(N):
                b.render_into(i, datas[i], depth=depth)
            if frame < 2:
                slots = b.process(depth=depth)
            else:  # masked: even cameras first, then the rest; per-camera state must not care
                even = np.arange(N) % 2 == 0
                s_even, s_odd = b.process(depth=depth, mask=even), b.process(depth=depth, mask=~even)
                slots = np.where(even, s_even, s_odd)
            sl = torch.from_numpy(slots).cuda()
            rgb = b.rgb[cams_t, sl].cpu().numpy()
            dep = b.depth[cams_t, sl].cpu().numpy() if depth else None
            for i, c in enumerate(refs):
                r = c.render(datas[i], depth=depth)
                r, d = r if depth else (r, None)
                assert np.array_equal(rgb[i], r), f"{name} frame {frame}: camera {i} rgb differs"
                if depth:
                    assert np.array_equal(dep[i], d), f"{name} frame {frame}: camera {i} depth differs"
                if cfg.ae_tau_s:
                    assert b.ev[i] == c._ev and tuple(b.wb[i]) == c._wb, f"{name} frame {frame}: AE camera {i}"
        if "no noise" in name:  # envs differ, so a camera rendering another env's ring would show here
            assert not np.array_equal(rgb[0], rgb[1]), f"{name}: cameras 0 and 1 identical"
        close(refs)
        b.close()
        print(f"W1-1 {name}: batch n={N} chunk={chunk} window == SimCamera(seed+i, cuda{'+interop' * io}), "
              f"3 frames (last masked){', rgb+depth' if depth else ''}: bit-equal")


def test_batch_window_ingest(n=4):
    from test_temporal import moving_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    kw = dict(**P1, **NOISE, **WIN, ae_tau_s=0.1, awb_tau_s=0.1, aperture_m=0.016, focus_m=0.5,
              depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0)
    model, _ = moving_scene()
    cfg = CameraConfig(W, H, **kw)
    datas = window_datas(model, n)
    bA = SimCameraBatch(model, "cam", cfg, n, interop=False)
    bB = SimCameraBatch(model, "cam", cfg, n, renderer=False)
    ref = SimCamera(model, "cam", cfg, renderer=bA.renderer)  # CPU: its _raw renders what the batch renders
    rings = [StateRing(model, ref.half_window) for _ in range(n)]
    sc = mujoco.MjData(model)
    cams_t = torch.arange(n, device="cuda")
    for frame in range(2):
        for _ in range(12):
            window_step(model, datas, bA.rings, rings)
        for i, d in enumerate(datas):
            bA.render_into(i, d, depth=True)
            tc, frames = d.time - ref.half_window, []
            for t, _ in ref._plan:
                rings[i].load(tc + t, sc)
                frames.append(ref._raw(sc, False)[0].copy())
            rings[i].load(tc, sc)
            z = ref._raw(sc, True, rgb=False)[1].copy()
            bB.ingest(i, torch.from_numpy(np.stack(frames)).cuda(), torch.from_numpy(z).cuda())
        sA, sB = bA.process(depth=True), bB.process(depth=True)
        assert np.array_equal(sA, sB)
        sl = torch.from_numpy(sA).cuda()
        assert torch.equal(bA.rgb[cams_t, sl], bB.rgb[cams_t, sl]), f"frame {frame}: ingest rgb differs"
        assert torch.equal(bA.depth[cams_t, sl], bB.depth[cams_t, sl]), f"frame {frame}: ingest depth differs"
        assert np.array_equal(bA.ev, bB.ev) and np.array_equal(bA.wb, bB.wb)
    ref.close()
    bA.close()
    bB.close()
    print(f"W1-2 ingest: renderer=False window batch fed stacked sub-renders (n_sub={bB.n_sub}) + depth == "
          f"render_into window batch, n={n}, 2 frames, DoF+depth noise+AE: bit-equal")


def test_batch_window_errors():
    from test_temporal import moving_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = moving_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE, **WIN)
    with SimCameraBatch(model, "cam", cfg, 2) as b:
        try:
            b.render_into(0, mujoco.MjData(model))
        except ValueError as e:
            print(f"W1-3 render_into before push: {e}")
        else:
            raise AssertionError("render_into before push accepted")
        assert not b.pending(0)
    with SimCameraBatch(model, "cam", cfg, 2, renderer=False) as b:
        for shape, what in [((b.H_r, b.W_r, 3), "unstacked"), ((b.n_sub - 1, b.H_r, b.W_r, 3), "n_sub - 1")]:
            try:
                b.ingest(0, torch.zeros(shape, dtype=torch.uint8, device="cuda"))
            except ValueError as e:
                print(f"W1-3 window ingest {what}: {e}")
            else:
                raise AssertionError(f"window ingest {what} accepted")
            assert not b.pending(0)
    model_p, _ = board_scene()
    with SimCameraBatch(model_p, "cam", CameraConfig(W, H, **P1, **NOISE), 2, renderer=False) as b:
        try:
            b.ingest(0, torch.zeros((2, b.H_r, b.W_r, 3), dtype=torch.uint8, device="cuda"))
        except ValueError as e:
            print(f"W1-3 non-window ingest stacked: {e}")
        else:
            raise AssertionError("non-window ingest stacked accepted")
        assert not b.pending(0)


def test_batch_window_no_sync(n=8):
    from test_temporal import moving_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = moving_scene()
    datas = window_datas(model, n)
    every_other = np.arange(n) % 2 == 1
    with SimCameraBatch(model, "cam", CameraConfig(W, H, **P1, **NOISE, **WIN), n, interop=HAVE_INTEROP,
                        chunk=3) as b:
        for mask in (np.ones(n, bool), every_other):  # views vs rows gathered by an index tensor
            for _ in range(2):
                window_step(model, datas, b.rings)
            for i in np.flatnonzero(mask):
                b.render_into(i, datas[i])
            torch.cuda.set_sync_debug_mode("error")
            try:
                b.process(mask=mask)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    print("W1-4 window process(): no stream sync (sync_debug_mode=error) for contiguous and gathered chunks")


def test_batch_window_mem(n=16):
    from test_temporal import moving_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = moving_scene()
    datas = window_datas(model, n)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a0 = torch.cuda.memory_allocated()
    b = SimCameraBatch(model, "cam", CameraConfig(W, H, **P1, **NOISE, **WIN), n, interop=HAVE_INTEROP, chunk=16)
    window_step(model, datas, b.rings)
    for i, d in enumerate(datas):
        b.render_into(i, d)
    b.process()
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - a0
    # window mode allocates neither raw colour slots nor the chunk LUT/remap buffers (_wlin replaces them)
    assert b._u8s is None and b._idx is None and b._lin is None and b._work is None and b._grid is None
    # per camera: _wlin (12 B/px) + output ring (3 B/px per slot) + per-camera FPN _mul/_add (8 B/px), output size
    per_cam = H * W * (12 + 3 * b.ring + 8)
    # 3 % slack (measured ~1.002 x the sum): a re-allocated raw slot stack (16 x 1.75 MiB) would not fit
    bound = 1.03 * (chunk_bytes(b) + n * per_cam)
    print(f"W1-5 window n={n} {W}x{H} ring={b.ring}: allocated {used / 2**20:.1f} MiB <= bound {bound / 2**20:.1f} "
          f"MiB = 1.03 x (shared {chunk_bytes(b) / 2**20:.1f} + {n} x {per_cam / 2**20:.2f} MiB)")
    assert used <= bound
    b.close()


def window_stream_run(ring, n=4, steps=300):
    """CameraStreamBatch drives a window batch; the hook also renders the single-camera reference."""
    import types

    from test_temporal import moving_scene

    from physicalai_sim2real_camera import CameraStreamBatch
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = moving_scene()
    cfg = CameraConfig(W, H, **P1, **NOISE, **WIN, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.1)
    datas = window_datas(model, n)
    expected, bad, total = {}, 0, 0
    with SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, ring=ring) as b:
        refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                          interop=HAVE_INTEROP, renderer=b.renderer) for i in range(n)]

        def hook(i, data, k):
            expected[(i, k)] = refs[i].render(data)
            return b.render_into(i, data)
        both = [types.SimpleNamespace(push=lambda d, i=i: (b.rings[i].push(d), refs[i].push(d))) for i in range(n)]
        sb = CameraStreamBatch(cfg, n, render=hook, cams=both)
        for t in range(steps):
            window_step(model, datas)  # CameraStream pushes (via `both`) before it fires
            ready, _, seq, hs, new, fired = sb.step(datas)
            b.process(mask=fired)
            for i in np.flatnonzero(ready & (new | (t % 25 == 0))):  # each delivery + held re-checks
                total += 1
                bad += not np.array_equal(b.rgb[hs[i]].cpu().numpy(), expected[(i, int(seq[i]))])
        close(refs)
    return bad, total


def test_batch_window_stream():
    bad, total = window_stream_run(None)
    bad1, total1 = window_stream_run(1)
    print(f"W2-1 CameraStreamBatch + window batch (N4B8, drops, jitter): ring_depth {bad}/{total} bad; "
          f"ring=1 {bad1}/{total1} bad (leak detected)")
    assert total > 0 and bad == 0 and bad1 > 0


def edge_hole_ratio(model, data, dep, k, dist):
    """Share of zeros in ``dep`` inside the edge band of the clean depth (CPU render at ``data``, same distortion)."""
    from physicalai_sim2real_camera.depth import depth_buffers, edge_band
    cam = SimCamera(model, "cam", CameraConfig(W, H, dist=dist))
    _, clean = cam.render(data, depth=True)
    cam.close()
    band = edge_band(clean, k, depth_buffers(H, W, edge=True)) & (clean > 0) & (clean < 5)
    return float(np.mean(dep[band] == 0))


def test_batch_window_edge_holes():
    from test_gpu import two_depth_scene

    from physicalai_sim2real_camera.batch import SimCameraBatch
    kw = dict(**P1, **NOISE, **WIN, depth_edge_hole_px=2)
    model, _ = two_depth_scene()
    cfg = CameraConfig(W, H, **kw)
    n = N
    datas = []
    for i in range(n):
        dd = mujoco.MjData(model)
        mujoco.mj_forward(model, dd)
        datas.append(dd)
    b = SimCameraBatch(model, "cam", cfg, n, interop=HAVE_INTEROP, chunk=3)
    refs = [SimCamera(model, "cam", dataclasses.replace(cfg, seed=cfg.seed + i), device="cuda",
                      interop=HAVE_INTEROP, renderer=b.renderer) for i in range(n)]
    cams_t = torch.arange(n, device="cuda")
    for frame in range(2):
        for _ in range(12):
            for i, d in enumerate(datas):
                mujoco.mj_step(model, d)
                b.rings[i].push(d)
                refs[i].push(d)
        for i in range(n):
            b.render_into(i, datas[i], depth=True)
        slots = b.process(depth=True)
        sl = torch.from_numpy(slots).cuda()
        dep = b.depth[cams_t, sl].cpu().numpy()
        for i, c in enumerate(refs):
            r = c.render(datas[i], depth=True)
            _, dref = r
            assert np.array_equal(dep[i], dref)
    ratio = edge_hole_ratio(model, datas[0], dep[0], cfg.depth_edge_hole_px, cfg.dist)
    assert ratio > 0.35, f"window edge holes: hole ratio {ratio:.2f} in the band"
    close(refs)
    b.close()
    print(f"W1-6 N4B8 +edge holes (two-depth scene): batch window depth == SimCamera(seed+i, cuda); "
          f"hole ratio in the edge band {ratio:.2f}")


def test_batch_ingest_all(n=5):
    """accumulate_all / ingest_all + claim_all (all cameras per call) == ingest per camera, bit for bit."""
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, _ = board_scene()
    base = dict(**P1, **NOISE, depth_sigma_a=0.002, depth_sigma_b=0.01, depth_max_m=10.0, ae_tau_s=0.1,
                flicker_amp=0.2, flicker_hz=100.0)
    g = torch.Generator(device="cuda").manual_seed(0)
    for name, kw in (("window", dict(base, **WIN)), ("global shutter", base)):
        cfg = CameraConfig(W, H, **kw)
        ref = SimCameraBatch(model, "cam", cfg, n, renderer=False, chunk=2)  # chunk < n: the chunked path too
        # compile=True: the fused LUT gather (exact); the tone tail is only compiled without AE (ae_tau_s here)
        new = SimCameraBatch(model, "cam", cfg, n, renderer=False, chunk=2, compile=True)
        shape = (n, ref.n_sub if ref.window else 1, ref.H_r, ref.W_r, 3)
        for step in range(3):  # AE state and noise generators advance across frames
            u8 = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=g)
            z = 0.5 + 3 * torch.rand((n, ref.H_r, ref.W_r), device="cuda", generator=g)
            tc = 0.1 * step
            for i in range(n):
                ref.ingest(i, u8[i] if ref.window else u8[i, 0], z[i], capture_t=tc)
            if new.window:
                for k in range(new.n_sub):
                    new.accumulate_all(k, u8[:, k].contiguous())
                new.claim_all(z, capture_t=tc)
            else:
                new.ingest_all(u8[:, 0].contiguous(), z, capture_t=tc)
            sr, sn = ref.process(depth=True), new.process(depth=True)
            for i in range(n):
                assert torch.equal(ref.rgb[i, sr[i]], new.rgb[i, sn[i]]), (name, step, i)
                assert torch.equal(ref.depth[i, sr[i]], new.depth[i, sn[i]]), (name, step, i)
        assert np.allclose(ref.ev, new.ev) and np.allclose(ref.wb, new.wb)
        print(f"ingest_all {name}: {n} cameras x 3 frames bit-identical to per-camera ingest (chunk 2)")
        try:  # wrong mode / double claim are refused
            (new.ingest_all if new.window else new.accumulate_all)(*((u8[:, 0],) if new.window else (0, u8[:, 0])))
        except ValueError:
            pass
        else:
            raise AssertionError("mode check")
        ref.close()
        new.close()


if __name__ == "__main__":
    if "--mem" in sys.argv:
        mem()
    elif "--bench" in sys.argv:  # R3 bench lives in scripts/bench_batch.py
        from bench_batch import main
        main([])
    else:
        test_equal()
        test_errors()
        test_chunk_ranges()
        test_contiguous_runs()
        test_camera_seeds()
        test_generator_seed()
        test_dof_groups()
        test_tone_fused()
        if HAVE_CUDA:
            test_to_device()
            test_batch_compile()
            test_batch_steady_state_kernels()
            test_batch_no_sync()
            test_batch_equal()
            test_batch_ring()
            test_batch_stream_ring()
            test_batch_errors()
            test_batch_ingest()
            test_batch_launches()
            test_batch_mem()
            test_batch_window_equal()
            test_batch_window_ingest()
            test_batch_window_errors()
            test_batch_window_no_sync()
            test_batch_window_mem()
            test_batch_window_stream()
            test_batch_window_edge_holes()
            test_batch_ingest_all()
    print("OK")
