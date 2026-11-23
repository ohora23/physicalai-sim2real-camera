"""Camera-side DR pure functions: mains flicker per row, lens dirt + veiling glare, MJPEG round trip."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray


def flicker_rows(
    H: int,
    t_c: float,
    exposure_s: float,
    readout_s: float,
    amp: float,
    hz: float,
    phase: float,
) -> NDArray[np.float32]:
    """Mains-driven flicker gain for every row of a rolling-shutter frame.

    Illumination: L(t) = 1 + amp * cos(2*pi*hz*t + phase).
    Exposure-averaged gain for row y is:
        g_y = 1 + amp * sinc(hz * e) * cos(2*pi*hz*t_y + phase)

    Args:
        H: Number of rows (height).
        t_c: Mid-exposure time of the centre row (s).
        exposure_s: Exposure duration per row.
        readout_s: Total readout time (full frame).
        amp: Flicker amplitude (0-1).
        hz: Light modulation frequency in Hz (twice the mains frequency: 100 on a 50 Hz grid).
        phase: Phase offset in radians.

    Returns:
        Float32 array of shape ``(H,)`` with per-row gain values.
    """
    t_y = t_c + readout_s * ((np.arange(H) + 0.5) / H - 0.5)  # mid-exposure time of each row, top first
    return (1 + amp * np.sinc(hz * exposure_s) * np.cos(2 * math.pi * hz * t_y + phase)).astype(np.float32)


def dirt_maps(
    H: int,
    W: int,
    density: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Lens dust and smudges as soft Gaussian blobs (out of focus on the lens, so soft).

    Specks: ``round(200 * density)`` blobs, sigma 0.5-2 % of ``min(H, W)``, opacity 0.1-0.3; smudges:
    ``1 + round(5 * density)`` blobs, sigma 8-25 %, opacity 0.05-0.15. Opacities add up; per group the draws are
    cx, cy, sigma, opacity (fixed order: one seed, one map).

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        density: Dirt density (0 = clean lens).
        rng: Generator the blob parameters are drawn from.

    Returns:
        ``(T, D)``, float32 ``(H, W, 1)`` each: transmission ``T = clip(1 - opacity, 0.3, 1)`` and scatter weight
        ``D = 1 - T``.
    """
    if density <= 0:
        return np.ones((H, W, 1), np.float32), np.zeros((H, W, 1), np.float32)
    s = min(H, W)
    A = np.zeros((H, W), np.float64)
    for n, (s_lo, s_hi), (a_lo, a_hi) in ((int(round(200 * density)), (0.005, 0.02), (0.1, 0.3)),
                                          (1 + int(round(5 * density)), (0.08, 0.25), (0.05, 0.15))):
        cx, cy = rng.uniform(0, W, n), rng.uniform(0, H, n)
        sig, amp = rng.uniform(s_lo, s_hi, n) * s, rng.uniform(a_lo, a_hi, n)
        for x0, y0, sg, am in zip(cx, cy, sig, amp):  # each blob only inside its +-3 sigma box
            r = int(math.ceil(3 * sg))
            ys = slice(max(0, int(y0) - r), min(H, int(y0) + r + 1))
            xs = slice(max(0, int(x0) - r), min(W, int(x0) + r + 1))
            yy, xx = np.mgrid[ys, xs]
            A[ys, xs] += am * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sg * sg))
    T = np.clip(1 - A, 0.3, 1).astype(np.float32)[..., None]
    return T, 1 - T


def veil(
    w: NDArray[np.float32],
    D: NDArray[np.float32],
    scatter: float,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Compute veiling glare from dirt-scattered light.

    Light scattered by dirt is spread widely via a strongly
    low-passed version of the input image.

    Args:
        w: Linear RGB image, float32 ``(H, W, 3)``.
        D: Dirt weight map, float32 ``(H, W, 1)``.
        scatter: Scatter strength (>= 0).
        out: Optional pre-allocated output array.

    Returns:
        Glare map float32 ``(H, W, 3)``. Writes into ``out`` when given.
    """
    h, w_ = w.shape[:2]
    lp = cv2.resize(
        w,
        (max(1, w_ // 16), max(1, h // 16)),
        interpolation=cv2.INTER_AREA,
    )
    lp = cv2.GaussianBlur(lp, (0, 0), 2.0)
    sD = D if scatter == 1 else scatter * D  # (H,W,1); a caller that pre-scales D passes scatter 1 (no allocation)
    if out is None:
        return sD * cv2.resize(lp, (w_, h), interpolation=cv2.INTER_LINEAR)
    cv2.resize(lp, (w_, h), dst=out, interpolation=cv2.INTER_LINEAR)
    np.multiply(out, sD, out=out)  # same product as sD * lp (commutative), so both paths are bit-identical
    return out


def jpeg_roundtrip(rgb: NDArray[np.uint8], quality: int) -> NDArray[np.uint8]:
    """MJPEG transport simulation: encode to JPEG then decode back.

    Args:
        rgb: uint8 ``(H, W, 3)`` in RGB order.
        quality: JPEG quality in [1, 100].

    Returns:
        Decoded uint8 ``(H, W, 3)`` in RGB order.

    Raises:
        ValueError: If quality is outside [1, 100].
        RuntimeError: If OpenCV fails to encode or decode.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")

    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed")

    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
