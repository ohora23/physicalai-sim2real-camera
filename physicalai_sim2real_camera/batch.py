"""SimCameraBatch: n cameras of one config on one shared ``mujoco.Renderer``, post-processed in chunks.

``render_into(i, data_i)`` renders env i and copies the raw frame into camera i's slot
(``_u8s[i]``, depth ``_zs[i]``) with no CUDA kernel (outside window mode): with interop the RGBA
renderbuffer and the raw reversed-Z pixel-pack buffer are ``cuMemcpy``'d as-is (GL rows,
bottom-up); the flip, alpha drop and Z->metres conversion run batched in ``process``.
``process(depth=False)`` runs the ``GpuPipeline`` stages once per chunk of ``chunk`` cameras on
``(C,3,H,W)`` work buffers shared by all chunks and writes camera i's result into its output ring
slot ``rgb[i, r_i % ring]`` (``r_i`` = camera i's render count), so frames still in flight to a
``CameraStream`` consumer are not overwritten (``ring = ring_depth(cfg)``). Steady-state ``process`` on
the window colour path (the virtual sensor's) is allocation-free and launches only kernels the first call
already ran: cameras are processed per contiguous run (slice views, no gather) and results go into the
rings by ``copy_`` (no index tensors, no uint8 temporary). Still allocating, served from the cache after
first use: DoF, AE/AWB uploads, depth noise (``_finish_depth``), ``compile=True``'s output and the instant
path's ``grid_sample`` (a first-time kernel costs ~20 ms of lazy module load, which was the
N=16 virtual-sensor stall). ``compile=True`` runs the
plain tone tail (``tone_fused``: FPN offset, noise, gamma, uint8) as one ``torch.compile``d kernel
set, ``process`` 1.6-2.1x faster at n >= 16 (``docs/perf.md`` Stage 6); off by default, AE/AWB/Bayer
stay eager. ``ingest(i, rgb[, depth])``
is the no-render entry point: a frame rendered elsewhere (e.g. imported from a GL-only sim process
over dma-buf, R5) fills the raw slot instead of ``render_into``; with ``renderer=False`` the batch
then owns no GL context at all.

Determinism rule (non-identity configs): camera i == ``SimCamera(model, cam_name, replace(cfg, seed=seed_base + i),
device="cuda")`` rendering the same data, bit for bit (rgb, depth, AE/AWB state), where
``seed_base`` defaults to ``cfg.seed``. So camera i's FPN is ``default_rng([s, 0])`` and its
noise comes from its own CUDA generator seeded ``SeedSequence([s, 1])`` (depth ``[s, 2]``),
``s = seed_base + i + cam_id``. Bit-equality needs per-camera generators (Philox streams of
different seeds cannot share one launch), so noise is one ``normal_`` launch per camera into its
slice of the chunk noise buffer; everything else is one launch set per chunk, except:

- AE/AWB metering: one ``mean`` per camera (a batched reduction splits differently, and the
  last-bit drift feeds back through ``ev``); all means come back in one sync per chunk.
- DoF: the blur cascade's kernel sizes depend on each camera's max CoC, so cameras are grouped
  by that value (one sync per chunk) and each group is blurred as one batch; identical scenes
  -> one group, fully different scenes -> one cascade per camera.

Per-camera state: ``ev (n,)``, ``wb (n,3)`` (float64, CPU: the AE math is scalar), FPN folded
into ``_mul/_add (n,H,W)`` (``(1,H,W)`` when FPN is off), generators. One shared config;
per-camera ``randomize`` results (length-n parameter tensors) are a follow-up. The identity config
still runs the pipeline, whereas an identity ``SimCamera`` returns the plain render, so the rule does not
hold there.

Window mode (``exposure_s``/``readout_s`` > 0: motion blur, rolling shutter; ``SimCamera``'s window):
``rings[i]`` is camera i's ``StateRing``, filled by ``rings[i].push(data_i)`` every sim step (or
``CameraStreamBatch(..., cams=b.rings)``). ``render_into`` renders the ``base._plan`` sub-frames
(``n_sub`` of them, grid-snapped) from the ring and accumulates them at once into the per-camera
output-space linear frame ``_wlin[i]`` (LUT, band accumulate, band remap: ``GpuPipeline.window``'s ops,
so it launches CUDA kernels); ``ingest`` takes the ``(n_sub, H_r, W_r, 3)`` stack of those sub-frames;
``process`` starts from ``_wlin`` instead of the raw slots. Depth is one render at ``capture_t``. The
determinism rule holds from the first push on (``render_into`` on an empty ring raises, where
``SimCamera.render`` would fall back to a single-instant render).
"""

from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from . import lens_torch as lt
from .camera import SimCamera, StateRing
from .gpu import _INTEROP, _grid, _no_tf32, _remap, edge_holes_
from .lens import dirt_maps, flicker_rows
from .sensor import ae_awb_step, make_fpn
from .timing import ring_depth
from .vibration import mount_vibration, shake_glcameras


def chunk_ranges(n, chunk):
    """Split cameras ``0..n-1`` into consecutive chunks.

    Args:
        n: Number of cameras.
        chunk: Maximum cameras per chunk (the work-buffer batch size).

    Returns:
        List of ``(c0, c1)`` half-open index ranges; the last one may be shorter.
    """
    return [(c0, min(c0 + chunk, n)) for c0 in range(0, n, chunk)]


def contiguous_runs(sel):
    """Split a sorted camera-index array into maximal runs of consecutive indices.

    ``process`` works per run so every row access is a slice view: no gather kernel (a kernel the
    warm-up never ran costs ~20 ms of lazy module load the first time) and no temporary.

    Args:
        sel: Sorted 1-D int array of camera indices (e.g. ``np.flatnonzero(mask)``).

    Returns:
        List of ``(lo, hi)`` half-open camera-index ranges covering ``sel`` in order.
    """
    runs, n = [], len(sel)
    j = 0
    while j < n:
        lo = int(sel[j])
        while j + 1 < n and sel[j + 1] == sel[j] + 1:
            j += 1
        runs.append((lo, int(sel[j]) + 1))
        j += 1
    return runs


def camera_seeds(seed_base, n, cam_id):
    """Per-camera base seeds ``s_i = seed_base + i + cam_id``.

    Camera i then matches ``SimCamera`` with ``cfg.seed = seed_base + i`` (which adds ``cam_id``).

    Args:
        seed_base: Seed of camera 0 before the camera-id offset.
        n: Number of cameras.
        cam_id: MuJoCo camera index of the rendered camera.

    Returns:
        List of n integer seeds.
    """
    return [seed_base + i + cam_id for i in range(n)]


def generator_seed(s, stream):
    """Torch generator seed for camera seed ``s``: ``SeedSequence([s, stream])``, as ``GpuPipeline``.

    Args:
        s: Camera base seed (``camera_seeds``).
        stream: 1 for RGB noise, 2 for depth noise (0 is the CPU FPN stream).

    Returns:
        A 32-bit integer seed for ``torch.Generator.manual_seed``.
    """
    return int(np.random.SeedSequence([s, stream]).generate_state(1)[0])


def to_device(a, dev):
    """Upload a host array to ``dev`` without a host-device stream sync.

    ``torch.from_numpy(a).to(dev)`` copies from pageable memory, which PyTorch completes with a
    ``cudaStreamSynchronize``; here the array is staged in pinned memory from PyTorch's caching host
    allocator (which keeps the block until the copy's stream event completes) and copied with
    ``non_blocking=True``. The staging copy is synchronous, so ``a`` may be modified right after the call.

    Args:
        a: Numpy array (any layout; made contiguous).
        dev: Target CUDA device.

    Returns:
        A tensor on ``dev`` with ``a``'s values and dtype, ordered on the current stream.
    """
    return torch.from_numpy(np.ascontiguousarray(a)).pin_memory().to(dev, non_blocking=True)


def dof_groups(coc_max, cap=32.0, min_c=0.5):
    """Group cameras by DoF blur size so each group runs one batched blur cascade.

    ``GpuPipeline._dof`` derives the cascade's kernel sizes from ``c = min(max CoC, cap)`` and skips
    the blur when ``c < min_c``; cameras with the same ``c`` can therefore share one launch set.

    Args:
        coc_max: Per-camera maximum circle of confusion (px), in chunk order.
        cap: Upper clamp of the blur size (px).
        min_c: Blur sizes below this (px) mean no blur.

    Returns:
        Dict ``c -> [chunk-local camera indices]`` in first-seen order; skipped cameras are absent.
    """
    groups = {}
    for j, cm in enumerate(coc_max):
        c = min(cm, cap)
        if c >= min_c:
            groups.setdefault(c, []).append(j)
    return groups


_TONE = None


def tone_fused(w, add, nz, ss2, sr2, inv_gamma):
    """Vignette/FPN-multiplied linear frame -> uint8 NHWC: FPN offset, shot/read noise, gamma, round.

    Same arithmetic as the eager tail of ``SimCameraBatch._finish`` (``w`` already multiplied by
    ``_mul``); written as one pure function so ``torch.compile`` fuses it into one or two kernels.

    Args:
        w: ``(k, 3, H, W)`` float32 linear frame, already multiplied by the vignette/PRNU gain.
        add: ``(k, 1, H, W)`` or ``(1, H, W)`` float32 DSNU + black-level offset.
        nz: ``(k, 3, H, W)`` standard-normal draw, or ``None`` for no noise.
        ss2: ``sigma_shot**2``.
        sr2: ``sigma_read**2``.
        inv_gamma: ``1 / gamma``.

    Returns:
        ``(k, H, W, 3)`` uint8 frame.
    """
    x = w + add
    if nz is not None:
        x = x + (x.clamp(min=0) * ss2 + sr2).sqrt() * nz
    x = x.clamp(0, 1).pow(inv_gamma).mul(255).round()
    return x.permute(0, 2, 3, 1).to(torch.uint8)


def _tone():
    """Compile ``tone_fused`` once (lazily, so importing this module never triggers inductor)."""
    global _TONE
    if _TONE is None:
        _TONE = torch.compile(tone_fused, dynamic=True)
    return _TONE


class SimCameraBatch:
    """n cameras of one config on one shared renderer, post-processed in chunks on the GPU.

    Determinism rule (non-identity configs): camera i is bit-equal to
    ``SimCamera(model, cam_name, replace(cfg, seed=seed_base + i), device="cuda")`` on the same data,
    i.e. its seeds derive from ``s = seed_base + i + cam_id``. Seeds overlap across batches (camera i
    of one == camera i+1 of another with ``seed_base`` one higher): space ``seed_base`` by >= n.
    See the module docstring for the per-camera exceptions to batching.

    Usage per step: ``h = render_into(i, data_i)`` (or ``ingest(i, raw)``) for the cameras that fire
    (at most once per camera per ``process``), then ``process(mask=...)``; ``h = (i, slot)`` indexes
    ``rgb[i, slot]``. In window mode, also call ``b.rings[i].push(data_i)`` each sim step before
    ``render_into``, or pass ``cams=b.rings`` to ``CameraStreamBatch``.

    Slot lifetime: ``rgb[i, slot]`` for camera i's render r is readable only after the ``process`` that
    consumed that render, and stays valid until the ``process`` that consumes render ``r + ring`` of
    camera i. Its data is ordered on ``torch.cuda.current_stream(dev)`` as of that ``process``: a
    consumer on another stream (or via ``__cuda_array_interface__``) must record an event on that
    stream after ``process`` and wait on it first. Usable as a context manager (``close`` on exit).

    Attributes:
        rgb: ``(n, ring, H, W, 3)`` uint8 CUDA output ring (slot lifetime above).
        depth: ``(n, ring, H, W)`` float32 CUDA depth ring (m, 0 = invalid) once ``process(depth=True)``
            ran, else None. A slot processed with ``depth=False`` keeps stale depth.
        ev: ``(n,)`` float64 AE exposure (EV) per camera, used by the next ``process``.
        wb: ``(n, 3)`` float64 AWB RGB gains per camera.
        renderer: The shared ``mujoco.Renderer``.
        window: True when the config has ``exposure_s`` or ``readout_s`` > 0 (window mode active).
        n_sub: Number of sub-renders per frame in window mode; 1 in non-window mode.
        rings: ``[StateRing]`` per camera in window mode (call ``rings[i].push(data)`` each step);
            ``[None] * n`` in non-window mode.
    """

    def __init__(self, model, cam_name, cfg, n, device="cuda", interop=True, chunk=16, renderer=None,
                 seed_base=None, ring=None, compile=False, scene=None):
        """Build the renderer, per-camera slots and chunk work buffers.

        Args:
            model: ``mujoco.MjModel``; ``offwidth/offheight`` (and ``offsamples`` with interop) may be
                raised/zeroed before the renderer is created, as ``SimCamera`` does.
            cam_name: Name of the MuJoCo camera all n cameras render.
            cfg: Shared ``CameraConfig`` (output ``cfg.width x cfg.height`` px; window mode supported).
            n: Number of cameras (envs).
            device: Torch CUDA device.
            interop: Copy renders GPU->GPU via CUDA-GL interop; falls back to readPixels on failure.
            chunk: Cameras per post-processing batch; work buffers are ``(chunk, 3, H_r, W_r)`` etc.
            renderer: Existing ``mujoco.Renderer`` of this model to share (must be ``H_r x W_r``), or
                ``False`` for an ingest-only batch: no renderer, no GL context, interop off,
                ``render_into`` raises; frames come in through ``ingest``.
            seed_base: Camera 0's ``cfg.seed`` equivalent; defaults to ``cfg.seed``. Space the
                ``seed_base`` of different batches by >= n, or their cameras share seeds.
            ring: Output slots per camera; default ``timing.ring_depth(cfg)``, enough that no frame still
                in flight to a ``CameraStream``/``CameraStreamBatch`` consumer is overwritten. Pass 1
                only when every frame is consumed before the camera renders again.
            compile: Run the per-chunk tone chain (vignette/FPN, noise, gamma, uint8) as one
                ``torch.compile``d function instead of ~12 eager kernels. Only the plain path
                (no AE/AWB/Bayer) is fused; the rest stays eager. Values may differ from eager by
                1 DN (fused rounding). Both chunk-size variants (k == 1, k > 1) are compiled in
                ``__init__`` (a few seconds, once per process), so no step stalls on a compile.

            scene: Optional ``scene_dr.SceneConfig`` applied after every ``update_scene`` (as ``SimCamera``).

        Raises:
            ValueError: A mismatched shared renderer, a multisampled shared renderer with interop, or
                ``jpeg_quality`` (MJPEG needs host pixels: CPU ``SimCamera`` only).
        """
        self.cfg, self.n = cfg, n
        interop = interop and renderer is not False  # ingest-only: nothing to interop with
        self.ring = ring_depth(cfg) if ring is None else ring
        self._closed = False
        self.chunk = C = min(chunk, n)
        self.compile = bool(compile)
        # accumulate_all's u8 -> linear LUT lookup as one fused kernel (exact: a gather, bit-identical to the eager
        # permute + int32 cast + index_select, ~4x faster at 64 x 656x494); compiles on first use per chunk shape
        self._lut_gather = (torch.compile(lambda lut, u8: lut[u8.permute(0, 3, 1, 2).long()], dynamic=False)
                            if self.compile else None)
        dev = self.dev = torch.device(device)
        if interop and renderer is None and model.vis.quality.offsamples != 0:
            print("WARNING: interop=True disables MSAA (model.vis.quality.offsamples = 0)")
            model.vis.quality.offsamples = 0
        if cfg.render_scale > 1:
            raise ValueError("render_scale > 1 is not supported in a batch (SimCamera only)")
        if cfg.jpeg_quality > 0:
            raise ValueError("jpeg_quality runs on the CPU SimCamera only, not in a batch")
        # geometry, maps, LUT, renderer; its CPU buffers are a few MB, its GPU side is unused
        base = self.base = SimCamera(model, cam_name, cfg, renderer=renderer, scene=scene)
        self.window = base.half_window > 0
        self.n_sub = len(base._plan) if self.window else 1
        self.rings = [StateRing(model, base.half_window) for _ in range(n)] if self.window else [None] * n
        self._scratch = base._scratch  # window: one scratch MjData the rings load into (None otherwise)
        self.renderer, self.cam = base.renderer, base.cam
        # read once: another camera sharing the renderer may update_scene between render and process
        self._ortho = model.cam_projection[base.cam] == mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
        H, W = cfg.height, cfg.width
        H_r, W_r = self.H_r, self.W_r = base._lin.shape[:2]
        f32 = dict(dtype=torch.float32, device=dev)
        # window mode: per-camera output-space linear frame + shared sub-render / band buffers (else None)
        self._wlin = self._u8w = self._lin1 = self._idx1 = self._bacc = self._bgrid = None
        self._baccn = None  # accumulate_all's per-camera band accumulators (allocated on first use)
        if self.window:
            self._wlin = torch.empty((n, 3, H, W), **f32)
            self._u8w = torch.empty((H_r, W_r, 3), dtype=torch.uint8, device=dev)
            self._lin1 = torch.empty((3, H_r, W_r), **f32)
            self._idx1 = torch.empty((3, H_r, W_r), dtype=torch.int32, device=dev)
            self._bacc, self._bgrid = [], []
            for (r0, r1, s0, s1), bm in zip(base._rows, base._bmaps):
                if bm is None:
                    self._bacc.append(None)
                    self._bgrid.append(None)
                else:
                    self._bacc.append(torch.empty((3, s1 - s0, W_r), **f32))
                    self._bgrid.append(_grid(*bm, s1 - s0, W_r, dev))
        self.seed_base = cfg.seed if seed_base is None else seed_base
        seeds = camera_seeds(self.seed_base, n, base.cam)
        # as SimCamera(cfg.seed = seed_base + i)
        self._vibs = [mount_vibration(cfg, self.seed_base + i, base.cam) for i in range(n)]

        self.interop = None
        if interop:
            if self.renderer._mjr_context.offSamples != 0:
                raise ValueError("interop needs a single-sample renderer (offsamples=0 before it is created)")
            try:
                ent = _INTEROP.get(self.renderer)
                if ent is None:
                    from .interop import GlInterop
                    ent = _INTEROP[self.renderer] = [GlInterop(self.renderer, dev), 0]
                ent[1] += 1
                self.interop = ent[0]
            except Exception as e:
                print(f"WARNING: CUDA-GL interop unavailable ({e}); falling back to readPixels + upload")
        # per-camera raw slots: interop keeps GL's layout (RGBA/RGB rows bottom-up, raw reversed Z)
        self._ch = self.interop.channels if self.interop is not None else 3
        if not self.window:
            self._u8s = torch.empty((n, H_r, W_r, self._ch), dtype=torch.uint8, device=dev)
        else:
            self._u8s = None
        self._zs = None  # (n,H_r,W_r) f32, allocated on the first depth render

        # FPN folded exactly like SimCamera: _mul = v*gain*g, _add = o + black (float32, per camera)
        v = np.ones((H, W, 1), np.float32) if base.vignette is None else base.vignette
        if base.fpn is None:
            mul = [(v * base.gain)[..., 0]]
            add = [np.full((H, W), np.float32(cfg.black_level), np.float32)]
        else:
            mul, add = [], []
            for s in seeds:
                g, o = make_fpn(H, W, cfg.sigma_prnu, cfg.sigma_dsnu, np.random.default_rng([s, 0]))
                mul.append((v * base.gain * g)[..., 0])
                add.append((o + np.float32(cfg.black_level))[..., 0])
        self._mul = torch.from_numpy(np.stack(mul)).to(dev)
        self._add = torch.from_numpy(np.stack(add)).to(dev)
        # camera-side DR (lens_torch): colour matrix shared; lens dirt per camera like SimCamera(seed + i)
        self._ccm = None if base._ccm is None else torch.from_numpy(base._ccm).to(dev)
        self._T = self._sD = None
        if cfg.dirt_density > 0:
            T, D = zip(*(dirt_maps(H, W, cfg.dirt_density, np.random.default_rng([s, 3])) for s in seeds))
            self._T = torch.from_numpy(np.stack(T)).permute(0, 3, 1, 2).contiguous().to(dev)  # (n, 1, H, W)
            if cfg.dirt_scatter > 0:
                sD = np.stack([cfg.dirt_scatter * d for d in D])  # float32, as SimCamera's pre-scaled weight
                self._sD = torch.from_numpy(sD).permute(0, 3, 1, 2).contiguous().to(dev)
        self._tc = np.zeros(n)  # per-camera mid-exposure time of the pending frame (flicker phase)
        self._gens = [torch.Generator(device=dev).manual_seed(generator_seed(s, 1)) for s in seeds]
        self._dgens = [torch.Generator(device=dev).manual_seed(generator_seed(s, 2)) for s in seeds]
        self.ev = np.full(n, cfg.exposure_ev)
        self.wb = np.ones((n, 3))

        self._lut = torch.from_numpy(base.lut.ravel().copy()).to(dev)
        # chunk LUT/remap buffers: not in window mode (_wlin is already linear, output space); the depth
        # nearest-remap index is needed in both modes
        self._idx = self._lin = self._grid = self._didx = self._work = None
        if base.maps is not None:
            mx, my = (np.rint(m).astype(np.int64) for m in base.maps)
            self._didx = torch.from_numpy(
                (np.clip(my, 0, H_r - 1) * W_r + np.clip(mx, 0, W_r - 1)).ravel()).to(dev)
        if not self.window:
            self._idx = torch.empty((C, 3, H_r, W_r), dtype=torch.int32, device=dev)
            self._lin = torch.empty((C, 3, H_r, W_r), **f32)
            if base.maps is None:
                self._work = self._lin
            else:
                self._grid = _grid(*base.maps, H_r, W_r, dev)
                self._work = torch.empty((C, 3, H, W), **f32)
        self._noise = cfg.sigma_shot != 0 or cfg.sigma_read != 0
        self._nz = torch.empty((C, 3, H, W), **f32) if self._noise else None
        # fused tone path applies whenever nothing per-camera (AE/AWB) or spatial (Bayer) sits inside the
        # chain; the eager noise scratch `_s` is then allocated only if the eager path ever runs
        self._fused = self.compile and not base._ae and not cfg.bayer
        self._s = torch.empty((C, 3, H, W), **f32) if self._noise and not self._fused else None
        self._dbuf = None  # chunk depth buffers, allocated on first use
        if cfg.bayer:
            cfa = torch.zeros((3, H, W), **f32)
            cfa[0, 0::2, 0::2] = cfa[1, 0::2, 1::2] = cfa[1, 1::2, 0::2] = cfa[2, 1::2, 1::2] = 1
            rb = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], **f32)
            g = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], **f32)
            self._cfa, self._bk = cfa, torch.stack([rb, g, rb])[:, None]
            with _no_tf32():
                self._bnorm = F.conv2d(cfa[None], self._bk, padding=1, groups=3)[0]
        if self._fused:  # compile both guard variants now (k == 1 and k > 1 trace separately), not mid-run
            f = _tone()
            for kk in (1, C):
                add = self._add[:kk][:, None] if self._add.shape[0] > 1 else self._add
                f(torch.zeros((kk, 3, H, W), **f32), add, self._nz[:kk] if self._noise else None,
                  cfg.sigma_shot * cfg.sigma_shot, cfg.sigma_read * cfg.sigma_read, 1.0 / cfg.gamma)
        self.rgb = torch.empty((n, self.ring, H, W, 3), dtype=torch.uint8, device=dev)
        self.depth = None  # (n,ring,H,W) f32 on the first process(depth=True)
        self._r = np.zeros(n, np.int64)  # renders per camera: slot = r_i % ring
        self._pending = np.full(n, -1, np.int64)  # slot of the render awaiting process, -1 = none
        self._dok = np.zeros(n, bool)  # the pending render includes depth
        self._depth_noise = bool(cfg.depth_sigma_a or cfg.depth_sigma_b or cfg.depth_min_m or cfg.depth_max_m
                                 or cfg.depth_edge_hole_px)

    def _accumulate(self, i, k, u8):
        """Accumulate sub-render k of camera i into the band accumulators.

        ``u8`` is a ``(H_r, W_r, 3)`` uint8 CUDA tensor, top row first, of sub-render k
        (index into ``base._plan``). Follows the same LUT + band accumulate + remap order as
        ``GpuPipeline.window`` for bit equality.

        Args:
            i: Camera index in ``[0, n)``.
            k: Sub-render index into ``base._plan``.
            u8: Sub-render frame tensor.
        """
        base = self.base
        self._idx1.copy_(u8.permute(2, 0, 1))
        torch.index_select(self._lut, 0, self._idx1.view(-1), out=self._lin1.view(-1))
        t, uses = base._plan[k]
        for b, first, last in uses:
            r0, r1, s0, s1 = base._rows[b]
            acc = self._bacc[b] if self._bacc[b] is not None else self._wlin[i][:, r0:r1]
            if first:
                acc.copy_(self._lin1[:, s0:s1])
            else:
                acc.add_(self._lin1[:, s0:s1])
            if last:
                if base._N > 1:
                    acc.mul_(1.0 / base._N)
                gr = self._bgrid[b]
                if gr is not None:
                    _remap(acc, gr, self._wlin[i][:, r0:r1])

    def close(self):
        """Release the shared interop registration (last user unregisters) and the renderer if owned.

        Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        ent = _INTEROP.get(self.renderer) if self.interop is not None else None
        self.interop = None
        if ent is not None:
            ent[1] -= 1
            if ent[1] <= 0 or self.base.owns_renderer:
                del _INTEROP[self.renderer]
                ent[0].close()
        self.base.close()

    def __enter__(self):
        """Return self; ``close`` runs on exit."""
        return self

    def __exit__(self, *exc):
        """Close the batch."""
        self.close()

    # ── render: GL + memcpy only (non-window) / CUDA kernels (window) ─
    def render_into(self, i, data, depth=False):
        """Render camera i from ``data`` into its slot.

        Outside window mode: GL + memcpy, no CUDA kernel. Window mode: the ``n_sub`` sub-frames of the
        window ending at ``data.time`` come from ``rings[i]`` and are accumulated at once into ``_wlin[i]``
        (CUDA kernels); unlike ``SimCamera.render``, an empty ring is an error, not a single-instant render.

        Args:
            i: Camera index in ``[0, n)``.
            data: That env's ``mujoco.MjData`` (after ``mj_forward``).
            depth: Also render depth; needed before ``process(depth=True)``. DoF configs always do.

        Returns:
            Handle ``(i, slot)``: the next ``process`` writes this render's result to ``rgb[i, slot]``
            (``slot = r_i % ring`` with ``r_i`` camera i's render count before this call); readable
            only after that ``process`` (see the class docstring for lifetime and stream ordering).

        Raises:
            ValueError: Camera i already has a render awaiting ``process``, the batch has no
                renderer (``renderer=False``; use ``ingest``), or in window mode the ring is empty
                (call ``b.rings[i].push(data)`` before this method).
        """
        if self.renderer is None:
            raise ValueError("render_into: this batch has no renderer (renderer=False); use ingest()")
        self._check_free(i)
        self._tc[i] = data.time - self.base.half_window
        if self.window and self.rings[i].n == 0:
            raise ValueError("b.rings[i].push(data) before render_into in window mode")
        depth = depth or self.base._dof is not None
        if depth and self._zs is None:
            self._zs = torch.empty((self.n, self.H_r, self.W_r), dtype=torch.float32, device=self.dev)
        m, cam, r = self.base.model, self.cam, self.renderer
        fovy0, ss0 = float(m.cam_fovy[cam]), m.cam_sensorsize[cam].copy()
        try:  # render at the margin fovy (as SimCamera._raw); update_scene reads it
            m.cam_sensorsize[cam, :] = 0
            m.cam_fovy[cam] = self.base.fovy_r
            (self._render_window if self.window else self._render_nonwindow)(i, data, depth)
        finally:
            r.disable_depth_rendering()
            m.cam_fovy[cam] = fovy0
            m.cam_sensorsize[cam] = ss0
        return self._claim(i, depth)

    def _render_window(self, i, data, depth):
        """Window mode: render the plan's sub-frames from ``rings[i]``, accumulate into ``_wlin[i]`` (+ depth)."""
        r, sc = self.renderer, self._scratch
        tc = data.time - self.base.half_window
        for k, (t, _) in enumerate(self.base._plan):
            self.rings[i].load(tc + t, sc)
            r.update_scene(sc, camera=self.cam)
            self._apply_scene(sc, i)
            if self.interop is not None:
                self.interop.render_into(self._u8w)
            else:
                self._u8w.copy_(torch.from_numpy(r.render()))
            self._accumulate(i, k, self._u8w)
        if depth:  # one render at capture_t, as SimCamera's window
            self.rings[i].load(tc, sc)
            r.update_scene(sc, camera=self.cam)
            self._apply_scene(sc, i)
            if self.interop is not None:
                self.interop.render_scene(depth=True)
                self.interop.copy_depth(self._zs[i].data_ptr())
            else:
                r.enable_depth_rendering()
                self._zs[i].copy_(torch.from_numpy(r.render()))

    def _render_nonwindow(self, i, data, depth):
        """Single instant: GL + memcpy into raw slot i."""
        r = self.renderer
        r.update_scene(data, camera=self.cam)
        self._apply_scene(data, i)
        if self.interop is not None:
            self._grab(i, depth)
        else:  # ponytail: synchronous upload fallback, no pinned double buffer
            self._u8s[i].copy_(torch.from_numpy(r.render()))
            if depth:
                r.enable_depth_rendering()
                self._zs[i].copy_(torch.from_numpy(r.render()))

    def _apply_scene(self, data, i):
        if self.base.scene is not None:
            self.base.scene.apply(self.renderer.scene, data)
        if self._vibs[i] is not None:  # camera i's own mount vibration at this sub-render's time
            shake_glcameras(self.renderer.scene, self._vibs[i].matrix(data.time))

    def ingest(self, i, rgb, depth=None, capture_t=None):
        """Feed camera i a frame rendered elsewhere instead of rendering it (no GL, one device copy).

        The frame must be what ``render_into`` would have rendered: ``H_r x W_r`` at ``base.fovy_r``
        with the camera's ``sensorsize`` zeroed (``SimCamera._raw``), e.g. imported from a GL-only sim
        process with ``dmabuf.DmabufImporter.read``. Slot bookkeeping, ``process`` and the output ring
        are exactly as after ``render_into``.

        In window mode, ``rgb`` must be ``(n_sub, H_r, W_r, 3)`` uint8 on the batch device, with
        frames in ``base._plan`` order; each sub-render is accumulated into ``_wlin[i]``.

        Args:
            i: Camera index in ``[0, n)``.
            rgb: ``(H_r, W_r, 3)`` in non-window mode or ``(n_sub, H_r, W_r, 3)`` in window mode,
                uint8 tensor on the batch device, top row first (``Renderer.render()`` layout).
            depth: Optional ``(H_r, W_r)`` float32 depth in metres (``Renderer`` depth mode); needed
                before ``process(depth=True)`` and always for DoF configs.
            capture_t: Mid-exposure sim time of the frame (``render_into``'s ``data.time - half_window``);
                required with ``flicker_amp > 0`` (the row gains depend on it), else unused.

        Returns:
            Handle ``(i, slot)``, as ``render_into``.

        Raises:
            ValueError: Interop is active (the raw slots then hold GL's RGBA bottom-up layout; build the
                batch with ``interop=False`` or ``renderer=False``), camera i already has a pending
                frame (``pending(i)``), a shape/dtype/device is wrong, or a DoF config gets no depth.
                Nothing is written when it raises.
        """
        if self.interop is not None:
            raise ValueError("ingest needs interop=False (raw slots are then top-first RGB)")
        self._check_free(i)
        if self.window:
            want = (self.n_sub, self.H_r, self.W_r, 3)
        else:
            want = (self.H_r, self.W_r, 3)
        checks = [(rgb, "rgb", want, torch.uint8)]
        if depth is not None:
            checks.append((depth, "depth", (self.H_r, self.W_r), torch.float32))
        dev = self._lut.device
        for t, name, shape, dtype in checks:
            if tuple(t.shape) != shape or t.dtype != dtype or t.device != dev:
                raise ValueError(f"ingest: {name} must be {shape} {dtype} on {dev}, "
                                 f"got {tuple(t.shape)} {t.dtype} on {t.device}")
        if depth is None and self.base._dof is not None:
            raise ValueError("ingest: DoF needs depth")
        if capture_t is None and self.cfg.flicker_amp > 0:
            raise ValueError("ingest: flicker_amp > 0 needs capture_t (the row gains depend on the frame time)")
        self._tc[i] = 0.0 if capture_t is None else capture_t
        if self.window:
            for k in range(self.n_sub):
                self._accumulate(i, k, rgb[k])
        else:
            self._u8s[i].copy_(rgb)
        if depth is not None:
            if self._zs is None:
                self._zs = torch.empty((self.n, self.H_r, self.W_r), dtype=torch.float32, device=self.dev)
            self._zs[i].copy_(depth)
        return self._claim(i, depth is not None)

    def accumulate_all(self, k, u8):
        """Window mode: accumulate sub-render ``k`` of every camera at once (no per-camera Python loop).

        The batched form of ``ingest``'s window path for renderers that draw all n views in one pass (Genesis'
        Madrona BatchRenderer): call it once per ``base._plan`` sub-render, then ``claim_all``. Same LUT + band
        accumulate + remap ops and order as ``_accumulate``, so camera i's result is bit-identical to
        ``ingest(i, stack_i)``. Cameras are taken ``chunk`` at a time to bound the temporaries; per-band
        accumulators for all n cameras replace the shared one (about one float frame per camera).

        Args:
            k: Sub-render index into ``base._plan`` (all of them, in order, before ``claim_all``).
            u8: ``(n, H_r, W_r, 3)`` uint8 tensor on the batch device, top row first.

        Raises:
            ValueError: Not in window mode, interop active, or a wrong shape / dtype / device.
        """
        if not self.window:
            raise ValueError("accumulate_all is the window-mode path; use ingest_all")
        if self.interop is not None:
            raise ValueError("accumulate_all needs interop=False (raw slots are then top-first RGB)")
        want = (self.n, self.H_r, self.W_r, 3)
        if tuple(u8.shape) != want or u8.dtype != torch.uint8 or u8.device != self._lut.device:
            raise ValueError(f"accumulate_all: u8 must be {want} uint8 on {self._lut.device}, "
                             f"got {tuple(u8.shape)} {u8.dtype} on {u8.device}")
        base = self.base
        if self._baccn is None:
            self._baccn = [None if a is None else torch.empty((self.n, *a.shape), dtype=a.dtype, device=a.device)
                           for a in self._bacc]
        _, uses = base._plan[k]
        for c0, c1 in chunk_ranges(self.n, self.chunk):
            if self._lut_gather is not None:
                lin = self._lut_gather(self._lut, u8[c0:c1])
            else:
                idx = u8[c0:c1].permute(0, 3, 1, 2).to(torch.int32)
                lin = torch.index_select(self._lut, 0, idx.reshape(-1)).view(idx.shape)
            for b, first, last in uses:
                r0, r1, s0, s1 = base._rows[b]
                acc = self._baccn[b][c0:c1] if self._baccn[b] is not None else self._wlin[c0:c1, :, r0:r1]
                if first:
                    acc.copy_(lin[:, :, s0:s1])
                else:
                    acc.add_(lin[:, :, s0:s1])
                if last:
                    if base._N > 1:
                        acc.mul_(1.0 / base._N)
                    gr = self._bgrid[b]
                    if gr is not None:
                        self._wlin[c0:c1, :, r0:r1].copy_(F.grid_sample(
                            acc, gr.expand(c1 - c0, *gr.shape[1:]), mode="bilinear", padding_mode="zeros",
                            align_corners=True))

    def ingest_all(self, rgb, depth=None, capture_t=None):
        """Non-window mode: every camera's frame at once (``ingest`` for all i), then ``claim_all``.

        Args:
            rgb: ``(n, H_r, W_r, 3)`` uint8 tensor on the batch device, top row first.
            depth: Optional ``(n, H_r, W_r)`` float32 depth in metres (as ``ingest``).
            capture_t: Mid-exposure sim time (as ``ingest``).

        Returns:
            ``claim_all``'s handles.

        Raises:
            ValueError: Window mode (use ``accumulate_all``), interop active, a wrong shape / dtype / device, or
                as ``claim_all``.
        """
        if self.window:
            raise ValueError("window mode: accumulate_all per sub-render, then claim_all")
        if self.interop is not None:
            raise ValueError("ingest_all needs interop=False (raw slots are then top-first RGB)")
        want = (self.n, self.H_r, self.W_r, 3)
        if tuple(rgb.shape) != want or rgb.dtype != torch.uint8 or rgb.device != self._lut.device:
            raise ValueError(f"ingest_all: rgb must be {want} uint8 on {self._lut.device}, "
                             f"got {tuple(rgb.shape)} {rgb.dtype} on {rgb.device}")
        for i in range(self.n):
            self._check_free(i)
        self._u8s.copy_(rgb)
        return self.claim_all(depth, capture_t)

    def claim_all(self, depth=None, capture_t=None):
        """Book every camera's pending frame after ``accumulate_all`` / ``ingest_all`` (``ingest``'s bookkeeping).

        Args:
            depth: Optional ``(n, H_r, W_r)`` float32 depth in metres; needed for ``process(depth=True)`` and for
                DoF configs.
            capture_t: Mid-exposure sim time of the frames; required with ``flicker_amp > 0``.

        Returns:
            List of ``(i, slot)`` handles, as ``ingest`` returns per camera.

        Raises:
            ValueError: A camera already has a pending frame, a wrong depth shape / dtype / device, a DoF config
                without depth, or flicker without ``capture_t``.
        """
        for i in range(self.n):
            self._check_free(i)
        if depth is not None:
            want = (self.n, self.H_r, self.W_r)
            if tuple(depth.shape) != want or depth.dtype != torch.float32 or depth.device != self._lut.device:
                raise ValueError(f"claim_all: depth must be {want} float32 on {self._lut.device}, "
                                 f"got {tuple(depth.shape)} {depth.dtype} on {depth.device}")
        if depth is None and self.base._dof is not None:
            raise ValueError("claim_all: DoF needs depth")
        if capture_t is None and self.cfg.flicker_amp > 0:
            raise ValueError("claim_all: flicker_amp > 0 needs capture_t (the row gains depend on the frame time)")
        self._tc[:] = 0.0 if capture_t is None else capture_t
        if depth is not None:
            if self._zs is None:
                self._zs = torch.empty((self.n, self.H_r, self.W_r), dtype=torch.float32, device=self.dev)
            self._zs.copy_(depth)
        return [self._claim(i, depth is not None) for i in range(self.n)]

    def pending(self, i):
        """Whether camera i has a frame (``render_into``/``ingest``) awaiting ``process``.

        Args:
            i: Camera index in ``[0, n)``.

        Returns:
            True if a second ``render_into``/``ingest`` of camera i would raise until the next ``process``.
        """
        return bool(self._pending[i] >= 0)

    def _check_free(self, i):
        if self._pending[i] >= 0:
            raise ValueError(f"camera {i}: render_into/ingest twice without process()")

    def _claim(self, i, depth):
        """Book camera i's next ring slot for the pending frame: ``(i, r_i % ring)``."""
        slot = int(self._r[i] % self.ring)
        self._r[i] += 1
        self._pending[i], self._dok[i] = slot, depth
        return i, slot

    def _grab(self, i, depth):
        """Raw interop copies into slot i (``GlInterop.copy_color``/``copy_depth``: no flip/convert kernel)."""
        io = self.interop
        io.render_scene()
        io.copy_color(self._u8s[i].data_ptr(), self.W_r * self._ch)
        if depth:
            io.render_scene(depth=True)
            io.copy_depth(self._zs[i].data_ptr())

    # ── process: one launch set per chunk ─────────────────────────────
    def process(self, depth=False, mask=None):
        """Post-process the pending renders, one launch set per chunk of processed cameras.

        Only processed cameras advance their generators and AE/AWB state; the others are untouched
        (a dropped frame never renders, so never updates, as in ``SimCamera``).

        Results are enqueued on ``torch.cuda.current_stream(dev)``; other streams must wait on an
        event recorded after this call before reading the slots.

        Args:
            depth: Also write output-space depth (m, 0 = invalid) to ``depth[i, slot]``; with
                ``False`` the depth slot keeps stale data.
            mask: ``(n,)`` bool array, cameras to process; default: every camera rendered since the
                last ``process``. Cameras left out keep their pending render for a later ``process``.

        Returns:
            ``(n,)`` int64 slot of each camera's result in ``rgb``/``depth``, -1 if not processed.

        Raises:
            ValueError: ``mask`` is not an ``(n,)`` bool array, a masked camera has no render since
                the last ``process``, or ``depth=True`` and a processed camera was rendered without
                depth.
        """
        H, W, n = self.cfg.height, self.cfg.width, self.n
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (n,):
                raise ValueError(f"mask must be ({n},) bool, got {mask.dtype} {mask.shape}")
        sel = np.flatnonzero(self._pending >= 0 if mask is None else mask)
        bad = sel[self._pending[sel] < 0]
        if bad.size:
            raise ValueError(f"cameras {bad.tolist()} have no render since the last process()")
        bad = sel[~self._dok[sel]]
        if depth and bad.size:
            raise ValueError(f"process(depth=True): cameras {bad.tolist()} were rendered without depth "
                             f"(render_into(i, data, depth=True))")
        slots = np.full(n, -1, np.int64)
        slots[sel] = self._pending[sel]
        need_d = depth or self.base._dof is not None
        if need_d and self._dbuf is None:
            C, f32 = self.chunk, dict(dtype=torch.float32, device=self.dev)
            self._dbuf = (torch.empty((C, self.H_r, self.W_r), **f32),
                          torch.empty((C, H, W), **f32) if self._didx is not None else None)
        if depth and self.depth is None:
            self.depth = torch.empty((n, self.ring, H, W), dtype=torch.float32, device=self.dev)
        # maximal contiguous runs, then chunks: every row access below is a slice view, so the steady state
        # launches no gather kernel (the module docstring lists the paths that still allocate)
        runs = [(r0 + c0, r0 + c1) for r0, r1 in contiguous_runs(sel) for c0, c1 in chunk_ranges(r1 - r0, self.chunk)]
        for lo, hi in runs:
            rows, k = slice(lo, hi), hi - lo
            js = np.arange(lo, hi)
            if self.window:
                w = self._wlin[rows]
                d = self._depth(rows, k) if need_d else None
                self._finish(w, d, js, rows, slots[js], depth)
            else:
                src = self._u8s[rows]
                if self.interop is not None:
                    src = src.flip(1)  # GL rows are bottom-up
                idx, lin = self._idx[:k], self._lin[:k]
                idx.copy_(src[..., :3].permute(0, 3, 1, 2))
                torch.index_select(self._lut, 0, idx.view(-1), out=lin.view(-1))
                w = self._work[:k]
                if self._grid is not None:
                    w.copy_(F.grid_sample(lin, self._grid.expand(k, -1, -1, -1), mode="bilinear",
                                          padding_mode="zeros", align_corners=True))
                d = self._depth(rows, k) if need_d else None
                self._finish(w, d, js, rows, slots[js], depth)
        self._pending[sel] = -1
        self._dok[sel] = False
        return slots

    def _depth(self, rows, k):
        draw, d = self._dbuf
        draw = draw[:k]
        if self.interop is None:
            draw.copy_(self._zs[rows])
        else:  # batched reversed-Z -> metres, same code as GlInterop.depth_into
            self.interop.z_to_metres(self._zs[rows], draw, self._ortho)
        if self._didx is None:
            return draw
        d = d[:k]
        torch.index_select(draw.view(k, -1), 1, self._didx, out=d.view(k, -1))
        return d

    def _dof(self, w, d):
        cfg, base = self.cfg, self.base
        coc = torch.reciprocal(d)
        coc.sub_(1.0 / cfg.focus_m).abs_().mul_(cfg.aperture_m * base.f_px)
        coc.masked_fill_(d <= 0, 0)
        coc.masked_fill_(d >= base._zfar * (1 - 1e-6), 0)
        for c, js in dof_groups(coc.amax((1, 2)).tolist()).items():  # one sync per chunk
            ix = torch.tensor(js, device=self.dev)
            wg, cg = w.index_select(0, ix), coc.index_select(0, ix)
            edges = (c / 8, c / (2 * math.sqrt(2)), c / math.sqrt(2))
            sig_prev = 0.0
            with _no_tf32():
                for edge, level in zip(edges, (c / 4, c / 2, c)):
                    sig = level / 4
                    s = math.sqrt(sig * sig - sig_prev * sig_prev)
                    sig_prev = sig
                    ks = int(round(s * 8 + 1)) | 1
                    kk = torch.from_numpy(cv2.getGaussianKernel(ks, s, cv2.CV_32F).ravel()).to(self.dev)
                    r = ks // 2
                    x = F.pad(wg, (r, r, r, r), mode="reflect")
                    x = F.conv2d(x, kk.view(1, 1, 1, -1).expand(3, 1, 1, ks), groups=3)
                    x = F.conv2d(x, kk.view(1, 1, -1, 1).expand(3, 1, ks, 1), groups=3)
                    wg = torch.where((cg >= edge)[:, None], x, wg)
            w.index_copy_(0, ix, wg)

    def _lens(self, w, js, rows):
        """PSF, colour matrix, flicker row gains, lens dirt + veil on the chunk ``w`` (k, 3, H, W) in place."""
        cfg = self.cfg
        if cfg.psf_sigma_px > 0:
            w.copy_(lt.blur(w, cfg.psf_sigma_px))
        if self._ccm is not None:
            w.copy_(lt.ccm(w, self._ccm))
        if cfg.flicker_amp > 0:
            g = np.stack([flicker_rows(w.shape[2], self._tc[j], cfg.exposure_s, cfg.readout_s, cfg.flicker_amp,
                                       cfg.flicker_hz, cfg.flicker_phase) for j in js])
            lt.row_gain_(w, to_device(g, self.dev))  # pinned + non_blocking: no stream sync
        if self._T is not None:
            glare = None if self._sD is None else lt.veil(w, self._sD[rows])
            w.mul_(self._T[rows])
            if glare is not None:
                w.add_(glare)

    def _finish(self, w, d, js, rows, slots, depth):
        cfg, base, k = self.cfg, self.base, len(js)
        if base._dof is not None:
            self._dof(w, d)
        if base.lens_on:
            self._lens(w, js, rows)
        w.mul_(self._mul[rows][:, None] if self._mul.shape[0] > 1 else self._mul)
        scale = 2.0 ** (self.ev[js] - cfg.exposure_ev)
        if (scale != 1.0).any():
            w.mul_(to_device(scale.astype(np.float32), self.dev).view(k, 1, 1, 1))
        if cfg.awb_tau_s > 0:
            w.mul_(to_device(self.wb[js].astype(np.float32), self.dev).view(k, 3, 1, 1))
        if self._fused and not (scale != 1.0).any():  # scale != 1 only if a caller wrote `ev` by hand
            nz = None
            if self._noise:
                nz = self._nz[:k]
                for j, i in enumerate(js):
                    nz[j].normal_(generator=self._gens[i])
            out = _tone()(w, self._add[rows][:, None] if self._add.shape[0] > 1 else self._add, nz,
                          cfg.sigma_shot * cfg.sigma_shot, cfg.sigma_read * cfg.sigma_read, 1.0 / cfg.gamma)
            self._store(self.rgb, rows.start, slots, out)
            self._finish_depth(d, js, rows, slots, depth)
            return
        w.add_(self._add[rows][:, None] if self._add.shape[0] > 1 else self._add)
        if base._ae:
            means = torch.stack([w[j].mean((1, 2)) for j in range(k)]).tolist()
        if self._noise:
            nz = self._nz[:k]
            for j, i in enumerate(js):
                nz[j].normal_(generator=self._gens[i])
            ss, sr = cfg.sigma_shot, cfg.sigma_read
            if self._s is None:  # fused batch fell back (caller wrote `ev`): allocate the scratch once
                self._s = torch.empty_like(self._nz)
            s = torch.clamp(w, min=0, out=self._s[:k])
            s.mul_(ss * ss).add_(sr * sr).sqrt_().mul_(nz)
            w.add_(s)
        if cfg.bayer:
            w.clamp_(0, 1).mul_(self._cfa)
            with _no_tf32():
                x = F.conv2d(w, self._bk, padding=1, groups=3)
            torch.div(x, self._bnorm, out=w)
        w.clamp_(0, 1).pow_(1.0 / cfg.gamma).mul_(255).round_()
        self._store(self.rgb, rows.start, slots, w.permute(0, 2, 3, 1))  # float -> uint8 cast inside copy_
        if base._ae:
            for j, i in enumerate(js):
                self.ev[i], self.wb[i] = ae_awb_step(self.ev[i], tuple(self.wb[i]), means[j], cfg)
        self._finish_depth(d, js, rows, slots, depth)

    @staticmethod
    def _store(ring, lo, slots, src):
        """``ring[lo + j, slots[j]] = src[j]`` with no index tensors and no temporary (dtype cast inside ``copy_``).

        One strided copy when the run shares a slot (the usual batch/RL case), else one per camera.
        """
        if (slots == slots[0]).all():
            ring[lo:lo + len(slots), int(slots[0])].copy_(src)
        else:
            for j, s in enumerate(slots):
                ring[lo + j, int(s)].copy_(src[j])

    def _finish_depth(self, d, js, rows, slots, depth):
        cfg, base = self.cfg, self.base
        if depth:
            z = d  # chunk scratch: noise in place, then into the ring
            if self._depth_noise:
                a, b, lo, hi = cfg.depth_sigma_a, cfg.depth_sigma_b, cfg.depth_min_m, cfg.depth_max_m
                if cfg.depth_edge_hole_px > 0:  # per-camera uniform draw first, then the per-camera normal draw
                    u = torch.empty_like(z)
                    for j, i in enumerate(js):
                        u[j].uniform_(generator=self._dgens[i])
                    edge_holes_(z, u, cfg.depth_edge_hole_px, base._zfar)
                m = (z <= 0) | (z >= base._zfar * (1 - 1e-6))
                if lo > 0:
                    m |= z < lo
                if hi > 0:
                    m |= z > hi
                if a != 0 or b != 0:
                    nz = torch.empty_like(z)
                    for j, i in enumerate(js):
                        nz[j].normal_(generator=self._dgens[i])
                    z.add_(z.mul(z).mul_(b).add_(a).mul_(nz))
                z.masked_fill_(m, 0)
            self._store(self.depth, rows.start, slots, z)
