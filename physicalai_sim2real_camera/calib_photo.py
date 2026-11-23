"""Photometric calibration: response curve, photon transfer curve, exposure matching."""

from __future__ import annotations

import warnings

import numpy as np


def fit_response(patch_means_dn, reflectances, black_level_lin=0.0):
    """Fit the camera's gamma response curve from a gray-patch chart.

    Regresses log(DN/255) = (1/gamma) * log(reflectance) + b linearly over
    `patch_means_dn`/`reflectances`, excluding saturated (DN >= 250) and
    near-black (DN <= 5) patches. Only gamma is meaningful for the sim config;
    the intercept b mixes illumination and exposure and is discarded (not
    transferable to sim; exposure_ev comes from C6 match_exposure). Run this
    before fit_flat_field/fit_ptc since they need gamma to linearize.

    A black level adds a constant in linear space, flattening the curve at low
    reflectance and biasing gamma high (2.2 -> 2.54 at black 0.02). `black_level_lin` > 0 is subtracted in
    linear domain, (DN/255)^gamma - black_level_lin, before the log fit; that
    needs gamma, so the fit is iterated to a fixed point (with 0 it is a single
    plain regression). Two-pass is the caller's job: C6 fits with 0 first
    (gamma for C2), then refits with C2's black_level.
    Returns (gamma, residual), residual = RMS of the log fit.
    """
    dn = np.asarray(patch_means_dn, np.float64)
    keep = (dn > 5) & (dn < 250)
    x = np.log(np.asarray(reflectances, np.float64)[keep])
    u = dn[keep] / 255.0
    gamma = 1.0
    for _ in range(100):
        # log(u^g - black)/g == log(u) when black == 0, so gamma is exact after one step
        y = np.log(np.maximum(u**gamma - black_level_lin, 1e-12)) / gamma
        slope, b = np.polyfit(x, y, 1)
        done = abs(1.0 / slope - gamma) < 1e-9
        gamma = 1.0 / slope
        if done:
            break
    return float(gamma), float(np.sqrt(np.mean((y - (slope * x + b)) ** 2)))


CLIP_FRAC = 1e-3  # a level with more than this share of samples at 0 or 255 DN is left out of the PTC


def ptc_points(level_stacks_dn, gamma, dark_mean_lin):
    """Per-level (m_lin, v_lin) photon-transfer points (pixel-averaged); see fit_ptc.

    Levels whose mean DN is outside (5, 250) or with more than ``CLIP_FRAC`` of their samples clipped are skipped.

    Also used by C6 for the mean-variance report CSV.
    """
    m_pts, v_pts = [], []
    for stack in level_stacks_dn:
        s = np.array(stack, np.float64)  # a copy: linearised in place below (offline stacks are full frames)
        m = s.mean(0)
        if not 5 < m.mean() < 250:
            continue
        if np.mean((s <= 0) | (s >= 255)) > CLIP_FRAC:  # clipped samples cut the variance (black squares at a low
            continue                                      # black level read sigma_read 25 % low)
        # linearise every sample, then take the temporal statistics: the delta method on the DN variance read
        # sigma_read 3.5 % high (O6: 9 %) once a level near the black level was in the fit
        s /= 255.0
        s **= gamma
        dI_dDN = gamma * (m / 255.0) ** (gamma - 1) / 255.0
        m_pts.append(s.mean() - dark_mean_lin)
        v_pts.append((s.var(0, ddof=1) - dI_dDN**2 / 12).mean())  # Sheppard: one DN step in linear units
    return np.array(m_pts), np.array(v_pts)


def fit_ptc(level_stacks_dn, gamma, dark_mean_lin):
    """Fit shot/read noise sigmas from a photon transfer curve (multiple exposure levels).

    For each level in `level_stacks_dn`, linearises every sample
    ((DN/255)^gamma), takes the per-pixel temporal variance v_lin minus the
    Sheppard quantization term (dI/dDN)^2 / 12 with
    dI/dDN = gamma * (m_DN/255)^(gamma-1) / 255, and m_lin = mean of the
    linearised samples - dark_mean_lin. Then fits v_lin = slope * m_lin + intercept by least squares
    over the per-level pixel means, excluding saturated (DN >= 250),
    near-black (DN <= 5) and clipped (``CLIP_FRAC``) levels.

    The sensor model applies shot noise to the whole signal including the black
    level (sensor.add_noise sees I + black_level), so the intercept on the
    dark-subtracted axis is sigma_read^2 + sigma_shot^2 * dark_mean_lin; the
    second term is removed (without it sigma_read reads ~22% high at
    black_level=0.02). Returns (sigma_shot, sigma_read, r2).
    """
    return ptc_line(*ptc_points(level_stacks_dn, gamma, dark_mean_lin), dark_mean_lin)[:3]


def ptc_line(m, v, dark_mean_lin, cov=False):
    """Least-squares line ``v = slope m + icpt`` through PTC points -> ``(sigma_shot, sigma_read, r2, C)``.

    ``sigma_shot = sqrt(slope)``, ``sigma_read^2 = icpt - slope * dark_mean_lin`` (see ``fit_ptc``). ``C`` is the
    2x2 covariance of ``(slope, icpt)`` from ``np.polyfit`` when ``cov`` and more than 3 points, else all inf.
    """
    if cov and len(m) > 3:
        (slope, icpt), C = np.polyfit(m, v, 1, cov=True)
    else:
        (slope, icpt), C = np.polyfit(m, v, 1), np.full((2, 2), np.inf)
    r2 = 1.0 - np.sum((v - (slope * m + icpt)) ** 2) / max(np.sum((v - v.mean()) ** 2), 1e-300)
    return (float(np.sqrt(max(slope, 0.0))), float(np.sqrt(max(icpt - slope * dark_mean_lin, 0.0))), float(r2), C)


def match_exposure(real_patch_mean_lin, sim_patch_mean_lin):
    """exposure_ev = log2(real / sim) for the same gray patch(es) in linear [0,1] units.

    `real_patch_mean_lin`: real D4 patch mean, linearised with the fitted gamma
    and black level subtracted. `sim_patch_mean_lin`: the same patch rendered in
    sim with fitted K/dist/gamma, noise 0, exposure_ev 0 (scripts/calibrate.py
    renders it with --sim-*; its analytic fallback is the reflectance, illum=1).
    Non-positive or non-finite inputs return 0.0 with a warning. Pure function.
    """
    real, sim = float(real_patch_mean_lin), float(sim_patch_mean_lin)
    if not (np.isfinite(real) and np.isfinite(sim) and real > 0 and sim > 0):
        warnings.warn(f"match_exposure: invalid means real={real}, sim={sim}: exposure_ev=0")
        return 0.0
    return float(np.log2(real / sim))
