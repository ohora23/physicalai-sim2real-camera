"""Geometric calibration: intrinsics from a checkerboard, flat-field from bright/dark stacks."""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from physicalai_sim2real_camera.optics import undistort_points
from physicalai_sim2real_camera.sensor import linearize


def fit_intrinsics(images, pattern=(9, 6), square_m=0.05):
    """Fit camera intrinsics and distortion from checkerboard images.

    Detects checkerboard corners (`pattern` = inner corners, `square_m` = square
    size in metres) in each of `images` via cv2.findChessboardCornersSB, then
    runs cv2.calibrateCamera (5 distortion coefficients) to recover K, dist, the
    RMS reprojection error, and per-view errors (views with per-view error > 2 px
    should be flagged as likely blurred/shaky and are candidates for exclusion).
    Views where detection fails are skipped with a warning; `per_view_err` covers
    only the used views. Returns (K, dist, rms, per_view_err) with dist a tuple of
    5 floats (k1, k2, p1, p2, k3). The fit itself is `fit_intrinsics_points`, which the
    online observer calls with its own correspondences.
    """
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    objs, pts = [], []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if not ok:
            warnings.warn(f"fit_intrinsics: no checkerboard in view {i}, skipped")
            continue
        objs.append(obj)
        pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
    H, W = gray.shape
    K, dist, rms, per_view, _ = fit_intrinsics_points(objs, pts, (W, H))
    return K, dist, rms, per_view


def fit_intrinsics_points(objs, pts, size, K0=None, flags=0):
    """Fit K and 5 distortion coefficients from per-view planar correspondences (shared by C1 and the observer).

    Args:
        objs: List of ``(n_i, 3)`` float32 object points per view (planar, z = 0; metres).
        pts: List of ``(n_i, 1, 2)`` float32 image points per view, same order as ``objs``.
        size: ``(W, H)`` image size in px.
        K0: Optional initial intrinsics; when given, ``cv2.CALIB_USE_INTRINSIC_GUESS`` is added to ``flags``.
        flags: Extra ``cv2.calibrateCamera`` flags.

    Returns:
        ``(K, dist, rms, per_view_err, std)`` with ``dist`` a tuple of 5 floats ``(k1, k2, p1, p2, k3)``,
        ``per_view_err`` the per-view RMS reprojection error (px) and ``std`` a dict of 1-sigma standard
        deviations ``{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"}`` from
        ``cv2.calibrateCameraExtended`` (OpenCV's Jacobian-based estimate, already scaled by the residual
        variance and in pixels: do not rescale it by the RMS). Views with per-view error > 2 px raise a warning.

    Raises:
        ValueError: Fewer than 3 views.
    """
    if len(pts) < 3:
        raise ValueError(f"fit_intrinsics_points: only {len(pts)} usable views (need >= 3)")
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = np.asarray(K0, np.float64).copy()
        K0 /= K0[2, 2]  # OpenCV insists on the exact [fx 0 cx; 0 fy cy; 0 0 1] shape
        K0[0, 1] = K0[1, 0] = K0[2, 0] = K0[2, 1] = 0.0
    objs = [np.asarray(o, np.float32) for o in objs]
    pts = [np.asarray(q, np.float32).reshape(-1, 1, 2) for q in pts]
    rms, K, dist, rvecs, tvecs, std_in, _std_ex, _pv = cv2.calibrateCameraExtended(
        objs, pts, tuple(size), K0, None, flags=flags)
    per_view = np.array([
        np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, dist)[0] - p) ** 2, axis=-1)))
        for o, p, r, t in zip(objs, pts, rvecs, tvecs)])
    for i in np.flatnonzero(per_view > 2.0):
        warnings.warn(f"fit_intrinsics_points: view {i} reprojection error {per_view[i]:.2f} px > 2 px")
    s = std_in.ravel()  # fx, fy, cx, cy, k1, k2, p1, p2, k3, ...
    std = dict(zip(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"), (float(x) for x in s[:9])))
    return K, tuple(float(x) for x in dist.ravel()[:5]), float(rms), per_view, std


def _lin_mean(stack, gamma):
    """Time-average of linearised frames, channel-averaged to (H, W, 1) float64."""
    acc = np.zeros(stack.shape[1:3], np.float64)
    for f in stack:  # linearise per frame, then average (averaging DN first biases dark low)
        acc += linearize(f, gamma).mean(axis=2)
    return (acc / len(stack))[:, :, None]


def undistorted_r2(K, dist, W, H):
    """Squared undistorted-normalized radius per pixel, (H, W) float64. Also used by scripts/calibrate.py."""
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    grid = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    d = None if dist is None else np.asarray(dist, np.float64)
    n = undistort_points(grid, np.asarray(K, np.float64), d, None, crit)
    return (n[:, 0, 0] ** 2 + n[:, 0, 1] ** 2).reshape(H, W)


def fit_flat_field(bright_stack, dark_stack, K, gamma, dist=None):
    """Fit vignetting strength, PRNU/DSNU gain and offset maps, and black level.

    Linearizes `bright_stack` and `dark_stack` (uint8 stacks) through `gamma`,
    time-averages each to get `bright_lin`/`dark_lin`. `black_level` is the mean
    of `dark_lin`; `dsnu_o = dark_lin - black_level` and `sigma_dsnu` its std.
    `bright_lin - dark_lin` is fit against a radial vignetting model
    V = (1 + r^2)^(-2*vignette_strength), where r is the undistorted normalized
    radius computed from `K` (and `dist`, same as `optics.vignette_map`; pass the
    C1 fit — `dist=None` treats the lens as distortion-free), via log-linear least
    squares to solve for vignette_strength. `prnu_g = (bright_lin - dark_lin) /
    (V_fit * center_brightness)` normalised to mean 1, and `sigma_prnu` its std.
    Maps are (H, W, 1) float32 like `sensor.make_fpn`. Returns
    (vignette_strength, prnu_g, dsnu_o, sigma_prnu, sigma_dsnu, black_level).
    """
    bright_lin = _lin_mean(bright_stack, gamma)
    dark_lin = _lin_mean(dark_stack, gamma)
    black_level = float(dark_lin.mean())
    dsnu_o = (dark_lin - black_level).astype(np.float32)

    H, W = dark_lin.shape[:2]
    r2 = undistorted_r2(K, dist, W, H)[:, :, None].astype(np.float64)

    S = bright_lin - dark_lin
    m = S > 0.05
    x = np.log1p(r2[m])
    A = np.stack([np.ones_like(x), -2.0 * x], axis=1)
    (logS0, s), *_ = np.linalg.lstsq(A, np.log(S[m]), rcond=None)
    g = S / ((1.0 + r2) ** (-2.0 * s) * np.exp(logS0))
    prnu_g = (g / g.mean()).astype(np.float32)
    return (float(s), prnu_g, dsnu_o, float(prnu_g.std()), float(dsnu_o.std()), black_level)


def distortion_field_error(K_true, d_true, K_est, d_est, hull_px, W, H, step=16):
    """Max pixel difference between two (K, dist) models over a grid inside the convex hull of ``hull_px``.

    Each grid pixel is undistorted with the true model and reprojected with the estimate, so the number compares
    the whole lens map where the views constrained it (coefficients trade off against each other outside).

    Args:
        K_true: (3, 3) true intrinsics.
        d_true: True distortion (5,).
        K_est: (3, 3) estimated intrinsics.
        d_est: Estimated distortion (5,).
        hull_px: ``(n, 2)`` image corners whose convex hull bounds the comparison.
        W: Image width, px.
        H: Image height, px.
        step: Grid step, px.

    Returns:
        Max reprojection difference, px.
    """
    xs, ys = np.meshgrid(np.arange(0, W, step, dtype=np.float64), np.arange(0, H, step, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    hull = cv2.convexHull(np.asarray(hull_px, np.float32))
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in pts])
    pts = pts[inside]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    norm = undistort_points(pts.reshape(-1, 1, 2), np.asarray(K_true, float), np.asarray(d_true, float), None, crit)
    obj = np.concatenate([norm.reshape(-1, 2), np.ones((len(pts), 1))], 1)
    back, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), np.asarray(K_est, float), np.asarray(d_est, float))
    return float(np.max(np.linalg.norm(back.reshape(-1, 2) - pts, axis=1)))
