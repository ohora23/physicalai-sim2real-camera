"""Optional torch/CUDA post-processing backend: ``SimCamera(..., device="cuda")``.

Same stages and math as ``camera.py``'s CPU path, in CHW float32 on the GPU, after
one upload of the uint8 render. MuJoCo renders straight into pinned host buffers
(``cam._rgb`` / ``cam._dbuf``), which are copied ``non_blocking`` and fenced by one
event before the next render reuses them. With ``interop=True`` the colour render is copied
GPU->GPU into ``_u8`` and the depth render into ``_draw`` instead (``interop.py``; MSAA off;
the pinned buffers stay allocated but unused). Differences from the CPU path:

- noise: a per-camera CUDA ``torch.Generator`` (``SeedSequence([seed + cam, 1])``, depth
  ``[seed + cam, 2]``) instead of cv2's global stream: statistically equivalent, not
  bit-equal; identical run-to-run for the same seed.
- DoF: every cascade step is a full-res blur (no half-res trick for σ > 4 px).
- Bayer: normalised bilinear demosaic (3x3 conv); matches cv2's interior, not its border
  rows/cols, and skips the uint16 quantisation.

Cameras sharing one ``mujoco.Renderer`` (``SimCamera(..., renderer=)``) share one ``GlInterop``
(``_INTEROP``: renderer -> [GlInterop, refcount]); each ``GpuPipeline.release()`` drops a
reference and the last one (or the renderer's owner) unregisters it. Everything else (pinned
buffers, work tensors, generators) stays per camera, so shared == own output, bit for bit.
"""

from __future__ import annotations

import math
import weakref

import cv2
import numpy as np

from .lens import flicker_rows
from .sensor import ae_awb_step

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("device='cuda' needs torch: pip install 'physicalai-sim2real-camera[gpu]'") from e

from . import lens_torch as lt  # noqa: E402 (after the torch check, for its install hint)

# ponytail: the GlInterop value holds its renderer, so entries leave only via release(), not GC
_INTEROP = weakref.WeakKeyDictionary()  # mujoco.Renderer -> [GlInterop, refcount]


def edge_band_torch(z, k):
    """Depth-discontinuity band of ``z`` (``(..., H, W)`` float32): ``(dilate - erode) > 0.1 * z``.

    ``F.max_pool2d(z, 2k+1, 1, k)`` / ``-F.max_pool2d(-z, ...)`` equal cv2.dilate/erode with a
    (2k+1)^2 kernel (the -inf padding matches cv2's default border). Returns ``(mx, mn, band)``;
    mx/mn are new tensors.

    Args:
        z: Depth tensor, shape ``(..., H, W)``, float32.
        k: Half-width of the discontinuity kernel (kernel size ``2k+1``).

    Returns:
        ``(mx, mn, band)``: max-pool, min-pool, and the boolean discontinuity band.
    """
    x = z.reshape(-1, 1, *z.shape[-2:])
    mx = F.max_pool2d(x, 2 * k + 1, 1, k).reshape(z.shape)
    mn = F.max_pool2d(-x, 2 * k + 1, 1, k).neg_().reshape(z.shape)
    band = (mx - mn) > z * 0.1  # two separate float32 ops (same rounding as the CPU)
    return mx, mn, band


def edge_holes_(z, u, k, zfar):
    """Edge holes / flying pixels of ``depth.depth_noise`` on ``z`` in place (``(..., H, W)`` float32, CUDA).

    ``u`` is the uniform draw for ``z`` (same shape; consumed, it becomes ``2u - 1``). In the band of
    ``edge_band_torch(z, k)``: ``u < 0.5`` or a far-clip / invalid neighbour -> hole (0), else flying pixel
    ``mn + (2u - 1) * (mx - mn)`` as separate ops (no fused multiply-add: CPU, GPU and batch round alike).

    Args:
        z: Clean output-space depth in metres, edited in place.
        u: Uniform [0, 1) draw of ``z``'s shape (one per camera image), overwritten.
        k: ``cfg.depth_edge_hole_px`` (> 0).
        zfar: Far clip in metres (background pixels sit there).
    """
    mx, mn, band = edge_band_torch(z, k)
    hole = band & ((u < 0.5) | (mx >= zfar * (1 - 1e-6)) | (mn <= 0))
    fly = band & ~hole
    f = mx - mn
    u.mul_(2).sub_(1)
    f.mul_(u)
    f.add_(mn)
    z.copy_(torch.where(fly, f, z))
    z.masked_fill_(hole, 0)


def _grid(mx, my, h, w, dev):
    """Convert a cv2.remap map to a ``grid_sample(align_corners=True)`` grid (pixel-centre convention).

    cv2 5.0 samples float32 images at the exact float coords (no 1/32 fixed-point table): measured.
    """
    gx = 2 * mx.astype(np.float64) / (w - 1) - 1
    gy = 2 * my.astype(np.float64) / (h - 1) - 1
    return torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32))[None].to(dev)


def _remap(src, grid, dst):
    # zeros padding == cv2.remap's default BORDER_CONSTANT 0 (maps stay inside the margin anyway)
    # ponytail: grid_sample has no out=; its temporary comes from torch's caching allocator
    dst.copy_(F.grid_sample(src[None], grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)[0])


def _no_tf32():
    return torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False)


class GpuPipeline:
    """CUDA post-processing backend for one ``SimCamera``; see module docstring for the pipeline."""

    def __init__(self, cam, device="cuda", interop=False):
        """Allocate pinned/device buffers for ``cam`` and, if ``interop``, register GL-CUDA interop.

        Args:
            cam: the owning SimCamera (its ``cfg``, ``_lin`` shape and ``renderer`` are read).
            device: CUDA device string, e.g. "cuda" or "cuda:0".
            interop: copy the colour/depth render GPU->GPU via CUDA-GL interop instead of
                readPixels + upload (falls back with a warning if unavailable).
        """
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        self.cam, cfg, dev = cam, cam.cfg, torch.device(device)
        self.dev = dev
        H, W = cfg.height, cfg.width
        H_r, W_r = cam._lin.shape[:2]
        f32 = dict(dtype=torch.float32, device=dev)

        self._pin = torch.empty((H_r, W_r, 3), dtype=torch.uint8, pin_memory=True)
        self._pin_d = torch.empty((H_r, W_r), dtype=torch.float32, pin_memory=True)
        cam._rgb, cam._dbuf = self._pin.numpy(), self._pin_d.numpy()  # MuJoCo renders into these
        self._up = torch.cuda.Event()  # recorded after each upload: pinned buffers free again
        self._u8 = torch.empty((H_r, W_r, 3), dtype=torch.uint8, device=dev)
        self.interop = None  # set: colour -> _u8 and depth -> _draw land GPU->GPU (no readPixels/upload)
        if interop:
            try:
                ent = _INTEROP.get(cam.renderer)
                if ent is None:
                    from .interop import GlInterop
                    ent = _INTEROP[cam.renderer] = [GlInterop(cam.renderer, dev), 0]
                ent[1] += 1
                self.interop = ent[0]
            except Exception as e:  # missing cuda-python, EGL/driver without GL interop, ...
                print(f"WARNING: CUDA-GL interop unavailable ({e}); falling back to readPixels + upload")
        self._idx = torch.empty((3, H_r, W_r), dtype=torch.int32, device=dev)
        self._lut = torch.from_numpy(cam.lut.ravel().copy()).to(dev)
        self._lin = torch.empty((3, H_r, W_r), **f32)
        self._draw = torch.empty((H_r, W_r), **f32)

        maps = cam.maps
        if maps is None:
            self._grid = self._didx = None
            self._work, self._d = self._lin, self._draw
        else:
            self._grid = _grid(*maps, H_r, W_r, dev)
            mx, my = (np.rint(m).astype(np.int64) for m in maps)  # cv2 INTER_NEAREST rounds
            self._didx = torch.from_numpy(
                (np.clip(my, 0, H_r - 1) * W_r + np.clip(mx, 0, W_r - 1)).ravel()).to(dev)
            self._work = torch.empty((3, H, W), **f32)
            self._d = torch.empty((H, W), **f32)
        self._mul = torch.from_numpy(cam._mul).permute(2, 0, 1).contiguous().to(dev)
        self._add = torch.from_numpy(cam._add).permute(2, 0, 1).contiguous().to(dev)
        # camera-side DR on the device (lens_torch, same stage order as SimCamera._finish); MJPEG stays CPU-only
        self._ccm = None if cam._ccm is None else torch.from_numpy(cam._ccm).to(dev)
        self._T3 = self._sD = None
        if cam._dirt is not None:
            T3, sD = cam._dirt
            self._T3 = torch.from_numpy(T3).permute(2, 0, 1).contiguous().to(dev)
            if sD is not None:
                self._sD = torch.from_numpy(np.ascontiguousarray(sD)).permute(2, 0, 1)[None].contiguous().to(dev)
        self._n = torch.empty((3, H, W), **f32)
        self._s = torch.empty((3, H, W), **f32)
        s = cfg.seed + cam.cam
        self._gen = torch.Generator(device=dev).manual_seed(int(np.random.SeedSequence([s, 1]).generate_state(1)[0]))
        self._dgen = torch.Generator(device=dev).manual_seed(int(np.random.SeedSequence([s, 2]).generate_state(1)[0]))
        self._mask = torch.empty((H, W), dtype=torch.bool, device=dev)
        self._tmp = torch.empty((H, W), dtype=torch.bool, device=dev)
        self._coc = torch.empty((H, W), **f32)
        self._dn = torch.empty((H, W), **f32)
        if cfg.depth_edge_hole_px > 0:
            self._du = torch.empty((H, W), **f32)

        if cfg.bayer:  # RGGB masks + normalised bilinear kernels (R/B: 1-2-1 tent, G: cross)
            cfa = torch.zeros((3, H, W), **f32)
            cfa[0, 0::2, 0::2] = cfa[1, 0::2, 1::2] = cfa[1, 1::2, 0::2] = cfa[2, 1::2, 1::2] = 1
            rb = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], **f32)
            g = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], **f32)
            self._cfa, self._bk = cfa, torch.stack([rb, g, rb])[:, None]
            with _no_tf32():
                self._bnorm = F.conv2d(cfa[None], self._bk, padding=1, groups=3)[0]

        if cam._ring is not None:  # window: per-band accumulators (+ band grids)
            if maps is None:
                self._work = torch.empty((3, H, W), **f32)
            self._bacc, self._bgrid = [], []
            for (r0, r1, s0, s1), bm in zip(cam._rows, cam._bmaps):
                if bm is None:
                    self._bacc.append(self._work[:, r0:r1])
                    self._bgrid.append(None)
                else:
                    self._bacc.append(torch.empty((3, s1 - s0, W_r), **f32))
                    self._bgrid.append(_grid(*bm, s1 - s0, W_r, dev))

    def release(self):
        """Drop this camera's interop reference; unregister on the last one or when the owner closes."""
        ent = _INTEROP.get(self.cam.renderer) if self.interop is not None else None
        self.interop = None
        if ent is None:
            return
        ent[1] -= 1
        if ent[1] <= 0 or self.cam.owns_renderer:  # the renderer (GL context) goes away with its owner
            del _INTEROP[self.cam.renderer]
            ent[0].close()

    # ── stages ────────────────────────────────────────────────────────
    def _linearize(self):
        if self.interop is None:
            self._u8.copy_(self._pin, non_blocking=True)
            self._up.record()
        self._idx.copy_(self._u8.permute(2, 0, 1))
        torch.index_select(self._lut, 0, self._idx.view(-1), out=self._lin.view(-1))

    def _depth(self):
        if self.interop is None:
            self._draw.copy_(self._pin_d, non_blocking=True)
            self._up.record()
        if self._didx is not None:
            torch.index_select(self._draw.view(-1), 0, self._didx, out=self._d.view(-1))
        return self._d

    def add_noise(self, w):
        """``w += sqrt(σ_shot²·max(w,0) + σ_read²)·n`` in place, ``n`` from the camera's CUDA generator."""
        ss, sr = self.cam.cfg.sigma_shot, self.cam.cfg.sigma_read
        if ss == 0 and sr == 0:
            return w
        n = self._n.normal_(generator=self._gen)
        s = torch.clamp(w, min=0, out=self._s)
        s.mul_(ss * ss).add_(sr * sr).sqrt_().mul_(n)
        return w.add_(s)

    def _dof(self, w, d):
        cfg, cam, m = self.cam.cfg, self.cam, self._mask
        coc = torch.reciprocal(d, out=self._coc)
        coc.sub_(1.0 / cfg.focus_m).abs_().mul_(cfg.aperture_m * cam.f_px)
        coc.masked_fill_(torch.le(d, 0, out=m), 0)
        coc.masked_fill_(torch.ge(d, cam._zfar * (1 - 1e-6), out=m), 0)
        c = min(float(coc.max()), 32.0)
        if c < 0.5:
            return
        edges = (c / 8, c / (2 * math.sqrt(2)), c / math.sqrt(2))
        sig_prev = 0.0
        with _no_tf32():
            for edge, level in zip(edges, (c / 4, c / 2, c)):
                sig = level / 4
                s = math.sqrt(sig * sig - sig_prev * sig_prev)
                sig_prev = sig
                ks = int(round(s * 8 + 1)) | 1  # cv2.GaussianBlur's float32 ksize (4σ radius)
                k = torch.from_numpy(cv2.getGaussianKernel(ks, s, cv2.CV_32F).ravel()).to(self.dev)
                r = ks // 2
                x = F.pad(w[None], (r, r, r, r), mode="reflect")  # == BORDER_REFLECT_101
                x = F.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, ks), groups=3)
                x = F.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, ks, 1), groups=3)
                torch.where(torch.ge(coc, edge, out=m), x[0], w, out=w)

    def _bayer(self, w):
        w.clamp_(0, 1).mul_(self._cfa)
        with _no_tf32():
            x = F.conv2d(w[None], self._bk, padding=1, groups=3)[0]
        torch.div(x, self._bnorm, out=w)

    def _depth_noise(self, z):
        cfg, m, t = self.cam.cfg, self._mask, self._tmp
        if cfg.depth_edge_hole_px > 0:  # u is drawn before the normal draw below (as depth.depth_noise)
            edge_holes_(z, self._du.uniform_(generator=self._dgen), cfg.depth_edge_hole_px, self.cam._zfar)
        a, b, lo, hi = cfg.depth_sigma_a, cfg.depth_sigma_b, cfg.depth_min_m, cfg.depth_max_m
        torch.le(z, 0, out=m)
        m |= torch.ge(z, self.cam._zfar * (1 - 1e-6), out=t)
        if lo > 0:
            m |= torch.lt(z, lo, out=t)
        if hi > 0:
            m |= torch.gt(z, hi, out=t)
        if a != 0 or b != 0:
            s = torch.mul(z, z, out=self._dn)
            s.mul_(b).add_(a).mul_(self._s[0].normal_(generator=self._dgen))
            z.add_(s)
        z.masked_fill_(m, 0)

    # ── entry points (called from SimCamera) ──────────────────────────
    def frame(self, depth, as_tensor, t):
        """Single instant: the render already sits in the pinned buffers; ``t`` = mid-exposure (flicker phase)."""
        self._linearize()
        d = self._depth() if depth or self.cam._dof is not None else None
        if self._grid is not None:
            _remap(self._lin, self._grid, self._work)
        return self._finish(self._work, d, depth, as_tensor, t)

    def window(self, data, depth, as_tensor):
        """GPU twin of ``SimCamera._render_window`` (same grid-snap plan; each band remaps only its rows)."""
        cam = self.cam
        tc, sc, inv_n = data.time - cam.half_window, cam._scratch, 1.0 / cam._N
        for t, uses in cam._plan:
            cam._ring.load(tc + t, sc)
            self._up.synchronize()  # previous upload done reading the pinned buffer
            cam._raw(sc, False)
            self._linearize()
            for b, first, last in uses:
                r0, r1, s0, s1 = cam._rows[b]
                acc = self._bacc[b]
                if first:
                    acc.copy_(self._lin[:, s0:s1])
                else:
                    acc.add_(self._lin[:, s0:s1])
                if last:
                    if cam._N > 1:
                        acc.mul_(inv_n)
                    if self._bgrid[b] is not None:
                        _remap(acc, self._bgrid[b], self._work[:, r0:r1])
        d = None
        if depth or cam._dof is not None:
            cam._ring.load(tc, sc)
            self._up.synchronize()
            cam._raw(sc, True, rgb=False)
            d = self._depth()
        return self._finish(self._work, d, depth, as_tensor, tc)

    def _lens(self, w, t):
        """PSF, colour matrix, mains flicker, lens dirt + veil on ``w`` (3, H, W) in place (``SimCamera._finish``)."""
        cfg = self.cam.cfg
        if cfg.psf_sigma_px > 0:
            w.copy_(lt.blur(w[None], cfg.psf_sigma_px)[0])
        if self._ccm is not None:
            w.copy_(lt.ccm(w[None], self._ccm)[0])
        if cfg.flicker_amp > 0:
            g = flicker_rows(w.shape[1], t, cfg.exposure_s, cfg.readout_s, cfg.flicker_amp, cfg.flicker_hz,
                             cfg.flicker_phase)
            w.mul_(torch.from_numpy(g).pin_memory().to(self.dev, non_blocking=True)[:, None])  # no stream sync
        if self._T3 is not None:
            glare = None if self._sD is None else lt.veil(w[None], self._sD)[0]
            w.mul_(self._T3)
            if glare is not None:
                w.add_(glare)

    def _finish(self, w, d, depth, as_tensor, t):
        cam, cfg = self.cam, self.cam.cfg
        if cam._dof is not None:
            self._dof(w, d)
        if cam.lens_on:
            self._lens(w, t)
        w.mul_(self._mul)
        scale = 2.0 ** (cam._ev - cfg.exposure_ev)
        if scale != 1.0:
            w.mul_(scale)
        if cfg.awb_tau_s > 0:
            for c in range(3):
                w[c].mul_(cam._wb[c])
        w.add_(self._add)
        if cam._ae:
            means = w.mean((1, 2)).tolist()  # small sync; AE math stays on the CPU
        self.add_noise(w)
        if cfg.bayer:
            self._bayer(w)
        w.clamp_(0, 1).pow_(1.0 / cfg.gamma).mul_(255).round_()
        rgb = torch.empty(w.shape[1:] + (3,), dtype=torch.uint8, device=self.dev)
        rgb.copy_(w.permute(1, 2, 0))
        if cam._ae:
            cam._ev, cam._wb = ae_awb_step(cam._ev, cam._wb, means, cfg)
        if depth:
            if cam._drng is not None:
                self._depth_noise(d)
            d = d.clone()
        self._up.synchronize()  # pinned buffers are free before MuJoCo renders into them again
        if not as_tensor:
            rgb = rgb.cpu().numpy()
            d = d.cpu().numpy() if depth else None
        return (rgb, d) if depth else rgb
