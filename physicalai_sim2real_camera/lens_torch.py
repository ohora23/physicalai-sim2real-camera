"""GPU twins of camera-side lens effects (mirror of lens.py).

All image tensors are float32, shape ``(n, 3, H, W)`` (channel-first batch), on any device; nothing moves to the CPU.
Each function matches its OpenCV / ``lens.py`` counterpart (tests/test_lens_torch.py); ``veil`` exactly only when H
and W are multiples of 16 (area-resize bins).
"""

from __future__ import annotations

import functools

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ccm(w: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 color correction matrix to a batch of images.

    Args:
        w: Float32 tensor of shape ``(n, 3, H, W)``.
        M: Float32 tensor of shape ``(3, 3)`` on ``w``'s device.

    Returns:
        New float32 tensor of shape ``(n, 3, H, W)`` with ``M`` applied.
    """
    out = torch.empty_like(w)
    for c in range(3):  # 9 elementwise mul-adds: exact float32, unaffected by the matmul TF32 setting
        o = out[:, c]
        torch.mul(w[:, 0], M[c, 0], out=o)
        o.add_(w[:, 1] * M[c, 1]).add_(w[:, 2] * M[c, 2])
    return out


def row_gain_(w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Multiply each row of a batch of images by a per-row gain.

    Args:
        w: Float32 tensor of shape ``(n, 3, H, W)``.
        g: Float32 tensor of shape ``(n, H)`` on ``w``'s device.

    Returns:
        ``w`` after in-place multiplication (same object).
    """
    w.mul_(g[:, None, :, None])
    return w


@functools.lru_cache(maxsize=64)
def _kernel_on(sigma: float, dev: torch.device) -> torch.Tensor:
    ksize = int(round(sigma * 8 + 1)) | 1
    return torch.from_numpy(cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel().copy()).to(dev)


def gaussian_kernel(sigma: float, device: str | torch.device) -> torch.Tensor:
    """Create a 1-D Gaussian kernel using OpenCV.

    Args:
        sigma: Standard deviation of the Gaussian.
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        Float32 1-D tensor of odd length ``ksize = int(round(sigma * 8 + 1)) | 1``.
    """
    return _kernel_on(float(sigma), torch.device(device))  # cached on the device: no per-frame upload / sync


_BORDER = {"reflect": cv2.BORDER_REFLECT, "reflect101": cv2.BORDER_REFLECT_101}


@functools.lru_cache(maxsize=64)
def _pad_index(n: int, r: int, border: str) -> np.ndarray:
    """Source index of each of the ``n + 2r`` padded positions, from OpenCV's own ``borderInterpolate``.

    Exact for any ``n`` (also ``r >= n``, where OpenCV reflects repeatedly), so the blur below matches
    ``cv2.GaussianBlur`` at every image size.
    """
    return np.array([cv2.borderInterpolate(p, n, _BORDER[border]) for p in range(-r, n + r)], np.int64)


@functools.lru_cache(maxsize=64)
def _pad_index_on(n: int, r: int, border: str, dev: torch.device) -> torch.Tensor:
    return torch.from_numpy(_pad_index(n, r, border)).to(dev)


def blur(w: torch.Tensor, sigma: float, border: str = "reflect") -> torch.Tensor:
    """Separable Gaussian blur matching ``cv2.GaussianBlur(x, (0, 0), sigma, borderType=...)``.

    Kernel from :func:`gaussian_kernel`; padding by gathering OpenCV's border indices on the device; horizontal
    then vertical ``F.conv2d`` (``groups=3``, TF32 off).

    Args:
        w: Float32 tensor of shape ``(n, 3, H, W)``.
        sigma: Standard deviation of the Gaussian in pixels.
        border: ``"reflect"`` (``cv2.BORDER_REFLECT``) or ``"reflect101"`` (``cv2.BORDER_REFLECT_101``).

    Returns:
        New float32 tensor of the same shape as ``w``.
    """
    k = gaussian_kernel(sigma, w.device)
    r = k.numel() // 2
    H, W = w.shape[-2:]
    iy, ix = _pad_index_on(H, r, border, w.device), _pad_index_on(W, r, border, w.device)
    x = w.index_select(-1, ix).index_select(-2, iy)  # (n, 3, H + 2r, W + 2r)
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        x = F.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        return F.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)


def veil(w: torch.Tensor, sD: torch.Tensor) -> torch.Tensor:
    """Veiling glare, the device twin of ``lens.veil(w, D, scatter)`` with ``sD = scatter * D``.

    Area resize to 1/16, Gaussian sigma 2 (``BORDER_REFLECT_101``, cv2's default), bilinear back up; all on w's device.
    Exact (1e-4) when H and W are multiples of 16; otherwise torch's integer area bins differ from cv2's fractional
    INTER_AREA weights (5e-3 measured on 120 x 150).

    Args:
        w: Float32 tensor of shape ``(n, 3, H, W)`` (linear light).
        sD: Float32 tensor ``(1, 1, H, W)`` or ``(n, 1, H, W)``: scatter times dirt weight.

    Returns:
        Float32 glare tensor of shape ``(n, 3, H, W)``.
    """
    H, W = w.shape[-2:]
    lp = F.interpolate(w, size=(max(1, H // 16), max(1, W // 16)), mode="area")
    lp = blur(lp, 2.0, border="reflect101")
    return sD * F.interpolate(lp, size=(H, W), mode="bilinear", align_corners=False)
