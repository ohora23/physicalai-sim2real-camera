"""Depth sensor noise (stereo / ToF style), float32.

Order: edge holes / flying pixels -> invalid mask -> axial noise -> zero.

Edge holes / flying pixels with k = ``cfg.depth_edge_hole_px`` (> 0):
kernel 2k+1, ``mx = cv2.dilate(z)``, ``mn = cv2.erode(z)``; edge band
``band = (mx - mn) > 0.1 * z``.  In the band draw one uniform u per pixel:
u < 0.5 or (mx >= zfar*(1-1e-6)) or (mn <= 0) -> hole (z = 0), else
flying pixel ``z = mn + (2u-1)*(mx-mn)``.  Boolean temporaries may be
allocated; the "allocation-free" claim then means "no float allocations".
With distortion the remap fills pixels outside the render with 0, so band
pixels next to that border have ``mn <= 0`` and become holes: the invalid
border grows by k pixels (same in the CPU, GPU and batch paths).

First pass: axial noise ``sigma(z) = a + b*z**2`` plus range / far-clip
invalidation (invalid pixels -> 0, the RealSense "no data" convention).
"""

from __future__ import annotations

import cv2
import numpy as np


def depth_buffers(H: int, W: int, edge: bool = False) -> dict:
    """Preallocate the per-frame scratch used by :func:`depth_noise`.

    Args:
        H: Depth height, px.
        W: Depth width, px.
        edge: Also allocate the edge-band buffers (``mx``, ``mn``, ``u``, ``f``, ``band``) that
            ``depth_edge_hole_px > 0`` needs (5 more (H, W) arrays).

    Returns:
        Dict of (H, W) arrays: ``n`` (standard normal draw), ``s`` (sigma(z) * n, also the edge band's
        ``0.1 * z`` scratch), ``mask``/``tmp`` (invalid pixels) [, edge buffers].
    """
    bufs = {
        "n": np.empty((H, W), np.float32),     # standard normal draw
        "s": np.empty((H, W), np.float32),     # sigma(z) * n
        "mask": np.empty((H, W), bool),        # invalid pixels
        "tmp": np.empty((H, W), bool),
    }
    if edge:
        bufs.update(mx=np.empty((H, W), np.float32), mn=np.empty((H, W), np.float32),
                    u=np.empty((H, W), np.float32), f=np.empty((H, W), np.float32),
                    band=np.empty((H, W), bool))
    return bufs


def edge_band(z: np.ndarray, k: int, bufs: dict) -> np.ndarray:
    """Depth-discontinuity band of ``z``: pixels whose (2k+1)x(2k+1) neighbourhood spans more than 10 % of z.

    ``mx = dilate(z)``, ``mn = erode(z)`` (rectangular kernel, cv2 default border: no effect at the edge),
    band = ``(mx - mn) > 0.1 * z``. Fills ``bufs["mx"]``, ``bufs["mn"]``, ``bufs["band"]`` and returns the band.

    Args:
        z: float32 (H, W) depth map (clean, pre-noise).
        k: half-kernel size; kernel is ``(2*k+1) x (2*k+1)``.
        bufs: ``depth_buffers(H, W, edge=True)``: ``mx``, ``mn``, ``f``, ``band`` are written, ``s`` is scratch.

    Returns:
        The bool (H, W) band array (also written into ``bufs["band"]``).
    """
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)  # explicit: kernel=None would be 3x3
    mx, mn, band, f, s = bufs["mx"], bufs["mn"], bufs["band"], bufs["f"], bufs["s"]
    cv2.dilate(z, kernel, dst=mx)
    cv2.erode(z, kernel, dst=mn)
    np.subtract(mx, mn, out=f)  # left in bufs["f"] for depth_noise's flying-pixel value
    np.multiply(z, np.float32(0.1), out=s)
    np.greater(f, s, out=band)
    return band


def zfar_of(model) -> float:
    """Far clip in metres, i.e. the depth MuJoCo writes for background pixels."""
    return float(model.vis.map.zfar * model.stat.extent)


def depth_noise(z: np.ndarray, cfg, rng: np.random.Generator, bufs: dict,
                zfar: float) -> np.ndarray:
    """Add axial noise to metric depth ``z`` (float32 (H,W)) in place and return it.

    Order: edge holes / flying pixels -> invalid mask -> axial noise -> zero.

    With ``k = cfg.depth_edge_hole_px`` (> 0), a (2k+1)² kernel computes the
    edge band; inside it draws one uniform u per pixel.  u < 0.5 or an
    invalid neighbour -> hole (0), else flying pixel ``mn + (2u-1)*(mx-mn)``.
    Boolean temporaries for hole/fly may be allocated; "allocation-free" then
    means "no float allocations".

    ``rng`` is created by the caller as
    ``np.random.default_rng([cfg.seed + cam_id, 2])`` -- a numpy stream
    separate from cv2's global RNG (stream index 2), so depth noise is
    reproducible per camera and independent of the RGB noise.

    Invalid -> 0: ``z <= 0``; background (``z >= zfar*(1-1e-6)``); ``z <
    depth_min_m`` / ``z > depth_max_m`` when those are > 0.  MuJoCo renders
    background at the far clip, not 0 -- pass ``zfar=zfar_of(model)``.
    """
    a, b = cfg.depth_sigma_a, cfg.depth_sigma_b
    lo, hi = cfg.depth_min_m, cfg.depth_max_m
    k = cfg.depth_edge_hole_px
    if a == 0 and b == 0 and lo <= 0 and hi <= 0 and k == 0:
        return z
    if k > 0:
        band = edge_band(z, k, bufs)
        u = bufs["u"]
        rng.random(out=u, dtype=np.float32)
        mx, mn, f = bufs["mx"], bufs["mn"], bufs["f"]  # f = mx - mn from edge_band
        hole = band & ((u < np.float32(0.5)) | (mx >= np.float32(zfar * (1 - 1e-6))) | (mn <= np.float32(0)))
        fly = band & ~hole
        np.multiply(u, np.float32(2), out=u)
        np.subtract(u, np.float32(1), out=u)
        f *= u
        f += mn
        np.copyto(z, f, where=fly)
        np.copyto(z, 0, where=hole)
    mask, tmp = bufs["mask"], bufs["tmp"]
    np.less_equal(z, 0, out=mask)
    np.greater_equal(z, np.float32(zfar * (1 - 1e-6)), out=tmp)
    mask |= tmp
    if lo > 0:
        np.less(z, lo, out=tmp)
        mask |= tmp
    if hi > 0:
        np.greater(z, hi, out=tmp)
        mask |= tmp
    if a != 0 or b != 0:
        n, s = bufs["n"], bufs["s"]
        rng.standard_normal(out=n, dtype=np.float32)
        np.multiply(z, z, out=s)
        s *= np.float32(b)
        s += np.float32(a)
        s *= n
        z += s
    np.copyto(z, 0, where=mask)
    return z
