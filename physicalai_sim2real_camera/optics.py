"""Optics helpers: render intrinsics, distortion maps, vignette masks, and DoF."""

from __future__ import annotations

import math
import warnings

import cv2
import numpy as np


def is_rational(dist) -> bool:
    """True when ``dist`` uses the OpenCV rational terms: k4, k5 or k6 non-zero."""
    return dist is not None and len(dist) > 5 and bool(np.any(np.asarray(dist, np.float64)[5:]))


def _distort_normalized(x, y, d):
    """OpenCV rational distortion of normalised points: ``d = (k1, k2, p1, p2, k3, k4, k5, k6)``."""
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def rational_domain(dist, r_max: float = 10.0, n: int = 100001):
    """Radial domain on which the rational model is invertible: ``(ru_max, rd_max, ru_table, rd_table)``.

    The radial map ``r_d = r_u * radial(r_u^2)`` of a rational model often folds back (stops increasing) at a radius
    the calibration never covered, e.g. some wide-angle factory models fold before the image corners. ``ru_max`` is the
    last radius before the first fold (or ``r_max``), ``rd_max`` its distorted radius; points beyond ``rd_max`` have
    no undistorted preimage. The tables (increasing) hold the radial map on ``[0, ru_max]``.

    Args:
        dist: Distortion coefficients with the rational terms (at least 8; terms past k6 must be zero).
        r_max: Largest undistorted normalised radius searched (10 = 84 deg off axis).
        n: Table size.

    Returns:
        ``(ru_max, rd_max, ru_table, rd_table)``.
    """
    d = np.zeros(8)
    d[:min(len(dist), 8)] = np.asarray(dist, np.float64)[:8]
    ru = np.linspace(0.0, r_max, n)
    r2 = ru * ru
    rd = ru * (1 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))) / (1 + r2 * (d[5] + r2 * (d[6] + r2 * d[7])))
    fold = np.flatnonzero(np.diff(rd) <= 0)
    end = int(fold[0]) + 1 if fold.size else n
    return float(ru[end - 1]), float(rd[end - 1]), ru[:end], rd[:end]


def _undistort_rational(pts, K, dist, P, iters: int = 20, return_ok: bool = False):
    """Invert a rational model: radial table inverse, then Newton on the full model inside the monotonic domain.

    Points Newton cannot invert to 1e-9 (normalised) -- past the fold (``rational_domain``), or in the thin band
    where the tangential terms move the fold -- keep the radial-table estimate, clamped to the domain edge along
    their ray. ``return_ok`` also returns the per-point success mask.
    """
    if np.any(np.asarray(dist, np.float64)[8:]):
        raise ValueError(f"thin-prism / tilt terms (past k6) are not modelled: dist={list(dist)}")
    d = np.zeros(8)
    d[:min(len(dist), 8)] = np.asarray(dist, np.float64)[:8]
    K = np.asarray(K, np.float64)
    q = np.asarray(pts, np.float64).reshape(-1, 2)
    yd = (q[:, 1] - K[1, 2]) / K[1, 1]  # like cv2, K's skew term is ignored
    xd = (q[:, 0] - K[0, 2]) / K[0, 0]
    ru_max, rd_max, ru_t, rd_t = rational_domain(d)
    rdist = np.hypot(xd, yd)
    s = np.interp(np.minimum(rdist, rd_max), rd_t, ru_t) / np.maximum(rdist, 1e-300)
    s[rdist == 0] = 1.0
    x0, y0 = xd * s, yd * s
    x, y = x0, y0
    h = 1e-7
    for _ in range(iters):  # Newton with a central-difference 2x2 Jacobian (tangential terms are small)
        fx, fy = _distort_normalized(x, y, d)
        ex, ey = fx - xd, fy - yd
        ax, ay = _distort_normalized(x + h, y, d)
        bx, by = _distort_normalized(x - h, y, d)
        cx_, cy_ = _distort_normalized(x, y + h, d)
        dx_, dy_ = _distort_normalized(x, y - h, d)
        j11, j21 = (ax - bx) / (2 * h), (ay - by) / (2 * h)
        j12, j22 = (cx_ - dx_) / (2 * h), (cy_ - dy_) / (2 * h)
        det = j11 * j22 - j12 * j21
        det = np.copysign(np.maximum(np.abs(det), 1e-12), det)
        x, y = x - (j22 * ex - j12 * ey) / det, y - (j11 * ey - j21 * ex) / det
        r = np.hypot(x, y)
        c = np.where(r > ru_max, ru_max / np.maximum(r, 1e-300), 1.0)  # stay on the invertible branch
        x, y = x * c, y * c
    fx, fy = _distort_normalized(x, y, d)
    ok = np.hypot(fx - xd, fy - yd) < 1e-9
    x, y = np.where(ok, x, x0), np.where(ok, y, y0)
    out = np.stack([x, y], -1)
    if P is not None:
        P = np.asarray(P, np.float64)
        out = np.stack([P[0, 0] * x + P[0, 1] * y + P[0, 2], P[1, 1] * y + P[1, 2]], -1)  # affine: P's 3rd row unused
    out = out.reshape(-1, 1, 2).astype(np.asarray(pts).dtype if np.asarray(pts).dtype.kind == "f" else np.float64)
    return (out, ok) if return_ok else out


def outside_domain_fraction(K, dist, W: int, H: int, step: int = 4) -> float:
    """Fraction of a W x H image (sampled every ``step`` px) the rational model cannot invert (past its fold)."""
    v, u = np.mgrid[0:H:step, 0:W:step].astype(np.float64)
    _, ok = _undistort_rational(np.stack([u, v], -1).reshape(-1, 1, 2), K, dist, None, return_ok=True)
    return float(1.0 - ok.mean())


def undistort_points(pts, K, dist, P, criteria):
    """Iterative ``cv2.undistortPoints`` with explicit ``criteria`` on OpenCV 4.x and 5.x.

    OpenCV 4.x exposes the criteria overload as ``undistortPointsIter``; 5.x takes ``criteria=`` on
    ``undistortPoints`` (and drops ``undistortPointsIter``). Rational models (``is_rational``: k4..k6 non-zero, e.g.
    wide-angle factory calibrations) go through ``_undistort_rational`` instead: OpenCV's fixed-point iteration does
    not converge near the edge of strongly distorted wide-angle models, and points past the model's fold have no
    preimage at all (``rational_domain``; they are clamped to the domain edge).

    Args:
        pts: Distorted pixel points, shape (N, 1, 2), float32 or float64.
        K: 3x3 camera matrix of the distorted image.
        dist: Distortion coefficients ``(k1, k2, p1, p2[, k3[, k4, k5, k6]])``.
        P: Output projection; ``None`` returns normalised coordinates, ``K`` returns pixels (the rational path takes
            P as an affine camera matrix: its third row is taken as (0, 0, 1)).
        criteria: ``cv2.TermCriteria`` tuple for the iterative inversion (OpenCV path only).

    Returns:
        Undistorted points, shape (N, 1, 2).
    """
    if is_rational(dist):
        return _undistort_rational(pts, K, dist, P)
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(pts, K, dist, None, P, criteria)
    return cv2.undistortPoints(pts, K, dist, None, P, criteria=criteria)


def render_intrinsics(
    K_out: np.ndarray,
    dist: tuple[float, ...],
    W: int,
    H: int,
    margin_px: int = 2,
) -> tuple[np.ndarray, int, int, float]:
    """Compute the camera matrix and output size for the distorted render.

    When *dist* is all-zeros and K_out is a centred fx==fy pinhole the
    output is unchanged (K_out, W, H).
    Otherwise the full perimeter of the W×H grid is undistorted to
    normalised coordinates and the bounding rect defines the new
    output size.

    Returns ``(K_render, W_r, H_r, fovy_deg)``.
    """
    # MuJoCo renders a centred, square-pixel pinhole (fovy only), so an
    # off-centre / fx≠fy K_out (e.g. sensorsize+principal camera) needs the
    # remap path even without distortion.
    centred = (K_out[0, 0] == K_out[1, 1] and K_out[0, 2] == (W - 1) / 2.0
               and K_out[1, 2] == (H - 1) / 2.0)
    if np.allclose(dist, 0.0) and centred:
        fy = K_out[1, 1]
        fovy = 2.0 * math.atan((H / 2.0) / fy) * 180.0 / math.pi
        return K_out, W, H, fovy

    if is_rational(dist):
        frac = outside_domain_fraction(K_out, dist, W, H)
        if frac > 0:
            warnings.warn(f"{100 * frac:.1f} % of the {W}x{H} image lies past the rational distortion model's fold "
                          "(rational_domain); those pixels are clamped to the domain edge", stacklevel=2)
    pts = _perimeter_points(W, H)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    undist = undistort_points(pts, K_out, dist, None, criteria)
    xn = undist[:, 0, 0]
    yn = undist[:, 0, 1]
    X = float(np.max(np.abs(xn)))
    Y = float(np.max(np.abs(yn)))

    fx = K_out[0, 0]
    fy = K_out[1, 1]
    f_r = max(fx, fy)

    W_r = int(2 * math.ceil(f_r * X + 0.5) + margin_px)
    H_r = int(2 * math.ceil(f_r * Y + 0.5) + margin_px)

    K_render = np.array(
        [[f_r, 0.0, (W_r - 1) / 2.0],
         [0.0, f_r, (H_r - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=K_out.dtype,
    )
    fovy = 2.0 * math.atan((H_r / 2.0) / f_r) * 180.0 / math.pi
    return K_render, W_r, H_r, fovy


def distortion_maps(
    K_out: np.ndarray,
    dist: tuple[float, ...],
    W: int,
    H: int,
    K_render: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build remap lookup tables for undistorting the render image.

    For each pixel (u, v) in the distorted output image, returns the
    coordinates in the K_render-pixel space where to sample.

    Returns ``(map_x, map_y)`` both ``float32`` arrays of shape ``(H, W)``.
    """
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    result = undistort_points(pts, K_out, dist, K_render, criteria)

    map_x = result[:, 0, 0].reshape(H, W).astype(np.float32)
    map_y = result[:, 0, 1].reshape(H, W).astype(np.float32)
    return map_x, map_y


def vignette_map(
    K_out: np.ndarray,
    dist: tuple[float, ...],
    W: int,
    H: int,
    strength: float,
) -> np.ndarray:
    """Build a vignette mask for the output image.

    Each pixel is attenuated by ``(1 + r_n²) ^ (-2·strength)`` where
    r_n is the normalised radial distance after undistortion.

    Returns ``V`` of shape ``(H, W, 1)`` dtype ``float32``.
    """
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    result = undistort_points(pts, K_out, dist, None, criteria)
    xn = result[:, 0, 0].reshape(H, W)
    yn = result[:, 0, 1].reshape(H, W)

    r2 = (xn ** 2 + yn ** 2).astype(np.float32)
    V = (1.0 + r2) ** (-2.0 * strength)
    return V[:, :, np.newaxis].astype(np.float32)


def dof_buffers(H: int, W: int) -> dict:
    """Preallocate the scratch arrays :func:`dof` reuses every frame."""
    return {"coc": np.empty((H, W), np.float32),
            "mask": np.empty((H, W), bool),
            "buf": np.empty((H, W, 3), np.float32),
            "half": np.empty((H // 2, W // 2, 3), np.float32),
            "half2": np.empty((H // 2, W // 2, 3), np.float32)}


def dof(w: np.ndarray, depth: np.ndarray, f_px: float, focus_m: float,
        aperture_m: float, bufs: dict, zfar: float) -> None:
    """Depth-guided defocus blur, in place on linear float32 *w* (H, W, 3).

    ``CoC_px = aperture_m·f_px·|1/z − 1/focus_m|`` from metric *depth* (H, W);
    z ≤ 0 or background (z ≥ *zfar*, see ``depth.zfar_of``) is treated as in focus.
    Pixels are quantised to the nearest of 3 levels {c/4, c/2, c} in log
    space (c = min(max CoC, 32 px); sharp below c/8), so the applied CoC is
    off by at most √2.  A disc of diameter CoC is approximated by a Gaussian
    with the same second moment, σ = CoC/4.
    ponytail: no occlusion handling — blurred background leaks over in-focus
    foreground edges (and vice versa) at depth discontinuities; upgrade to
    layered compositing if that matters.
    """
    if aperture_m == 0 or focus_m == 0:
        return
    coc, mask, buf = bufs["coc"], bufs["mask"], bufs["buf"]
    with np.errstate(divide="ignore"):
        np.reciprocal(depth, out=coc)
    np.subtract(coc, np.float32(1.0 / focus_m), out=coc)
    np.abs(coc, out=coc)
    np.multiply(coc, np.float32(aperture_m * f_px), out=coc)
    np.less_equal(depth, 0, out=mask)
    np.copyto(coc, 0, where=mask)
    np.greater_equal(depth, np.float32(zfar * (1 - 1e-6)), out=mask)
    np.copyto(coc, 0, where=mask)
    c = min(float(coc.max()), 32.0)
    if c < 0.5:
        return
    # lower edges = log-space midpoints between levels (sharp | c/4 | c/2 | c)
    edges = (c / 8, c / (2 * math.sqrt(2)), c / math.sqrt(2))
    sig_prev = 0.0
    for edge, level in zip(edges, (c / 4, c / 2, c)):
        sig = level / 4
        # cascade: pixels of this level already hold the previous level's blur,
        # so add only the missing variance (interior σ is exact, kernels shrink)
        s = math.sqrt(sig * sig - sig_prev * sig_prev)
        if s > 4:
            # ponytail: half-res blur for wide kernels (σ=6.9: 3.8 -> 0.75 ms). The area-down /
            # bilinear-up pair adds ~1 px² of variance, subtracted here; residual σ error <= 1.5%
            # (step edge); vs full-res |Δ| <= 0.023 linear on the P4 checker at c=32 (measured).
            h, h2 = bufs["half"], bufs["half2"]
            cv2.resize(w, h.shape[1::-1], dst=h, interpolation=cv2.INTER_AREA)
            cv2.GaussianBlur(h, (0, 0), math.sqrt(s * s - 1.0) / 2, dst=h2)
            cv2.resize(h2, w.shape[1::-1], dst=buf, interpolation=cv2.INTER_LINEAR)
        else:
            cv2.GaussianBlur(w, (0, 0), s, dst=buf)
        sig_prev = sig
        np.greater_equal(coc, edge, out=mask)
        cv2.copyTo(buf, mask.view(np.uint8), w)  # in place; np.copyto(where=(H,W,1)) is ~15x slower


# ── private helpers ──────────────────────────────────────────────────

def _perimeter_points(W: int, H: int) -> np.ndarray:
    """Return all perimeter pixel coordinates as (N, 1, 2) float32.

    Coordinates are in (x, y) pixel-space order.
    """
    pts: list[np.ndarray] = []
    # top row: y = 0, x = 0..W-1
    pts.append(np.mgrid[0:W, 0:1].T.reshape(-1, 2))
    # bottom row: y = H-1, x = 0..W-1
    pts.append(np.mgrid[0:W, H - 1:H].T.reshape(-1, 2))
    # left col: x = 0, y = 1..H-2  (swap ranges so first dim is x)
    pts.append(np.mgrid[0:1, 1:H - 1].T.reshape(-1, 2))
    # right col: x = W-1, y = 1..H-2
    pts.append(np.mgrid[W - 1:W, 1:H - 1].T.reshape(-1, 2))
    return np.vstack(pts).astype(np.float32).reshape(-1, 1, 2)
