"""Realistic sensor signal-processing primitives.

Pure functions only — no classes.  All arrays float32, linear domain
[0, 1] where 1.0 = saturation.  Uses ``out=`` / ``dst=`` buffers
everywhere the frame-loop runs (performance rule: no float64 promotion;
constants must be ``np.float32``; no new array allocation per frame in
hot paths).
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def linearize(u8: np.ndarray, gamma: float) -> np.ndarray:
    """Return ``(u8 / 255) ** gamma`` as a float32 array.

    Implemented as a 256-entry lookup table applied with ``cv2.LUT``
    (SIMD gather, ~0.15 ms at 640x480x3 vs ~0.9 ms for ``np.power`` on
    every pixel) since *u8* only has 256 distinct input values.

    Args:
    u8:
        uint8 array of any shape.
    gamma:
        Gamma value (e.g. 2.2).

    Returns:
        np.ndarray: float32 array, same shape as *u8*, values in ``[0, 1]``.
    """
    # ponytail: called once per camera at init (camera.py caches the table); no cache here
    lut = np.power(np.arange(256, dtype=np.float32) / np.float32(255.0), np.float32(gamma))
    # cv2.LUT drops a trailing singleton channel dim; reshape guards it.
    return cv2.LUT(u8, lut).reshape(u8.shape)


def encode(I: np.ndarray, gamma: float) -> np.ndarray:
    """Clip to ``[0, 1]``, power ``(1/γ)``, ``round(*255)``, cast to uint8.

    Uses ``cv2.pow`` (SIMD) instead of ``np.power`` for the ``1/γ``
    step — faster than numpy's per-element pow for large arrays.

    Args:
    I:
        float32 array in linear domain ``[0, 1]``.
    gamma:
        Gamma value (e.g. 2.2).

    Returns:
        np.ndarray: uint8 array, same shape as *I*, values in ``[0, 255]``.
    """
    np.clip(I, np.float32(0.0), np.float32(1.0), out=I)  # I is clobbered (caller's scratch)
    cv2.pow(I, 1.0 / gamma, dst=I)
    I *= np.float32(255.0)
    np.rint(I, out=I)
    return I.astype(np.uint8)  # fresh array on purpose: CameraStream holds returned frames


def make_fpn(
    H: int,
    W: int,
    sigma_prnu: float,
    sigma_dsnu: float,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Fixed-Pattern-Noise gain and offset maps.

    Args:
    H:
        Image height in pixels.
    W:
        Image width in pixels.
    sigma_prnu:
        PRNU standard deviation (gain noise, mean ≈ 1.0).
    sigma_dsnu:
        DSNU standard deviation (offset noise, mean ≈ 0.0).
    rng:
        ``np.random.Generator`` instance (caller constructs, e.g.
        ``np.random.default_rng(seed)``).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(g, o)`` gain and offset maps, each shape ``(H, W, 1)`` float32.
    """
    g = rng.normal(np.float32(1.0), np.float32(sigma_prnu), size=(H, W, 1)).astype(np.float32)
    o = rng.normal(np.float32(0.0), np.float32(sigma_dsnu), size=(H, W, 1)).astype(np.float32)
    return g, o


# Lazy per-shape scratch cache for add_noise (shape → float32 array).
# Warm after first call; no further per-frame allocation.
_add_noise_scratch: dict[tuple, np.ndarray] = {}


def add_noise(
    I: np.ndarray,
    sigma_shot: float,
    sigma_read: float,
    buf: np.ndarray,
) -> np.ndarray:
    """Add photon shot noise + read noise to *I* in-place.

    The caller is responsible for calling ``cv2.setRNGSeed(seed)`` before
    use; cv2's RNG is thread-local/global state, so determinism only
    holds for sequential calls from one calling thread, in the given order.

    Args:
    I:
        float32 array, shape ``(H, W, 3)``, linear domain ``[0, 1]``.
        Modified in place and returned.
    sigma_shot:
        Shot-noise coefficient (proportional to √signal).
    sigma_read:
        Read-noise coefficient (constant).
    buf:
        PREALLOCATED float32 array of shape ``(H, W * 3)`` — NOT
        ``(H, W, 3)``.

        Why?  ``cv2.randn`` on a 3-channel Mat only applies the given
        std to channel 0, so a ``(H, W, 3)`` buffer would leave
        channels 1 and 2 as zero noise; using a ``(H, W * 3)``
        single-channel buffer and viewing it as ``(H, W, 3)`` via
        ``reshape`` avoids this.

    Returns:
        np.ndarray: The same array *I* (modified in place).
    """
    s_shot = np.float32(sigma_shot)
    s_read = np.float32(sigma_read)
    zero = np.float32(0.0)

    # Fast path: no noise requested.
    if s_shot == zero and s_read == zero:
        return I

    H, W = I.shape[:2]
    shape_key = (H, W, 3)

    # Lazily cache a per-shape scratch array (one-time allocation).
    if shape_key not in _add_noise_scratch:
        _add_noise_scratch[shape_key] = np.empty((H, W, 3), dtype=np.float32)
    scratch = _add_noise_scratch[shape_key]

    # Fill buf with unit normal noise, view as (H, W, 3).
    # cv2.randn on a 3-channel Mat applies std to channel 0 only (measured std [1,0,0]) → buffer must be (H, W·3)
    cv2.randn(buf, 0.0, 1.0)
    N = buf.reshape(H, W, 3)

    # sigma = sqrt(sigma_shot² · max(I, 0) + sigma_read²), all in scratch
    np.maximum(I, zero, out=scratch)
    scratch *= s_shot * s_shot
    scratch += s_read * s_read
    cv2.sqrt(scratch, dst=scratch)

    # I += sigma * N  (in-place on I)
    np.multiply(scratch, N, out=scratch)
    np.add(I, scratch, out=I)
    return I


def ae_awb_step(ev: float, wb: tuple, means, cfg) -> tuple[float, tuple]:
    """One AE (+ gray-world AWB) update from this frame's metering.

    *means* = per-channel linear means of ``w`` after ``_mul/_add``, before
    noise (deterministic).  Luma = plain mean of the 3 channel means.
    ``α = 1 − exp(−1/(fps·τ))`` per rendered frame (τ ≤ 0 → frozen); dropped
    frames have no render, so the caller skips the update.  *wb* = (R, G, B)
    gains, G fixed at 1.  Incremental log form: the relaxation form
    ``log wb += α(log g − log wb)`` converges to the wrong point.
    """
    r, g, b = (max(float(m), 1e-6) for m in means)
    if cfg.ae_tau_s > 0:
        a = 1.0 - math.exp(-1.0 / (cfg.fps * cfg.ae_tau_s))
        ev += a * math.log2(cfg.ae_target / ((r + g + b) / 3.0))
        # ponytail: fixed ±6 EV range; real AE also moves exposure_s (blur length)
        ev = min(max(ev, cfg.exposure_ev - 6.0), cfg.exposure_ev + 6.0)
    if cfg.awb_tau_s > 0:
        a = 1.0 - math.exp(-1.0 / (cfg.fps * cfg.awb_tau_s))
        wb = tuple(min(max(w * (g / m) ** a, 0.25), 4.0) for w, m in ((wb[0], r), (1.0, g), (wb[2], b)))
    return float(ev), wb


def bayer(w: np.ndarray, raw_u16: np.ndarray, rgb_u16: np.ndarray) -> np.ndarray:
    """RGGB mosaic + demosaic of *w* (H, W, 3) float32 in place (after noise, before encode).

    *raw_u16* (H, W) and *rgb_u16* (H, W, 3) uint16 are preallocated.
    """
    np.clip(w, np.float32(0.0), np.float32(1.0), out=w)
    w *= np.float32(65535.0)
    w += np.float32(0.5)  # round, not truncate, on the uint16 cast below
    raw_u16[0::2, 0::2] = w[0::2, 0::2, 0]
    raw_u16[0::2, 1::2] = w[0::2, 1::2, 1]
    raw_u16[1::2, 0::2] = w[1::2, 0::2, 1]
    raw_u16[1::2, 1::2] = w[1::2, 1::2, 2]
    # cv2 names Bayer codes by the 2x2 at pixel (1,1), so our R-at-(0,0) RGGB
    # layout is COLOR_BayerRGGB2RGB == COLOR_BayerBG2RGB; COLOR_BayerRG2RGB swaps R/B.
    # Pinned by tests/test_sensor.py::test_bayer_gradient.
    cv2.cvtColor(raw_u16, cv2.COLOR_BayerRGGB2RGB, dst=rgb_u16)
    np.multiply(rgb_u16, np.float32(1.0 / 65535.0), out=w)
    return w
