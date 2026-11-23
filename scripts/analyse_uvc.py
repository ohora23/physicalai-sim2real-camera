"""Analyse a ``capture_uvc.py`` recording of a fixed camera: timing, response (gamma) and photon transfer.

    python scripts/analyse_uvc.py uvc_capture.npz [--relative-time inverse|direct] [--out uvc_report.json]

- Timing: ``TimingObserver`` on the driver buffer stamps (``ts_ms``) and host arrival times; frame numbers are
  recovered from the stamp spacing (a gap of k periods = k - 1 drops). Also reported: arrival - stamp, the delay
  the driver queue adds on top of the camera latency.
- Response: the crop is cut into 16 x 16 px tiles; each tile's mean DN over the exposure steps must be linear in the
  relative exposure time after linearisation, ``(DN / 255)^gamma = a_tile * t + b_tile``. Gamma is the value that
  makes all tiles most linear; ``t`` is ``1 / exposure_absolute`` for cameras whose control runs backwards
  (``--relative-time inverse``, the Logitech Webcam Pro 9000) or the control value itself (``direct``).
- PTC: every (tile, exposure) stack is one level for ``calib_photo.ptc_points`` / ``ptc_line`` with that gamma and
  the median intercept as the black level. Tiles that move (temporal std of the tile mean > 3x the median),
  clip, or drift (1 % between halves) are left out. ``identifiable`` is False when the line fit is poor (r2 < 0.9)
  or the pixel noise is under 1 DN rms (8-bit quantisation then dominates: a dim scene).
"""

from __future__ import annotations

import argparse
import ast
import json

import numpy as np

from physicalai_sim2real_camera.calib_photo import ptc_line, ptc_points
from physicalai_sim2real_camera.observer_photo import drifted
from physicalai_sim2real_camera.observer_timing import TimingObserver

TILE = 16


def timing(t_arrival, ts_s):
    """Timing estimates from arrival times and driver stamps (s)."""
    dts = np.diff(ts_s)
    period = float(np.median(dts))
    if not period > 0:
        raise SystemExit("driver stamps do not advance (CAP_PROP_POS_MSEC unsupported?)")
    # frame numbers gap by gap (a gap of k periods = k - 1 drops): dividing the elapsed time by one median period
    # accumulates its error over the run and slips frames
    fn = np.r_[0, np.cumsum(np.maximum(1, np.round(dts / period)))].astype(int)
    obs = TimingObserver(readout_s=0.0, height_px=480)
    for k, (a, s, f) in enumerate(zip(t_arrival, ts_s, fn)):
        obs.push(a, s, k, int(f))
    est = obs.estimate()
    lag = t_arrival - ts_s
    return dict(frames=len(ts_s), period_s=period, fps=1 / period, drops=int(fn[-1] + 1 - len(fn)),
                jitter_theta_s=est["jitter_theta"], jitter_theta_std_s=est["jitter_theta_std"],
                arrival_minus_stamp_s=dict(mean=float(lag.mean()), sd=float(lag.std())))


def tiles(stacks):
    """``(n_exp, n_frames, h, w, c)`` -> ``(n_exp, n_tiles, n_frames, TILE, TILE, c)`` float64 (c = 3 RGB, 1 luma)."""
    n_e, n_f, h, w, c = stacks.shape
    ty, tx = h // TILE, w // TILE
    s = stacks[:, :, :ty * TILE, :tx * TILE].astype(np.float64)
    s = s.reshape(n_e, n_f, ty, TILE, tx, TILE, c).transpose(0, 2, 4, 1, 3, 5, 6)
    return s.reshape(n_e, ty * tx, n_f, TILE, TILE, c)


def usable(T):
    """Mask ``(n_exp, n_tiles)``: still (tile-mean temporal std <= 3x median), not clipped, no 1 % drift."""
    tm = T.mean(axis=(3, 4, 5))  # (n_exp, n_tiles, n_frames)
    sd = tm.std(axis=2)
    ok = sd <= 3 * np.median(sd, axis=1, keepdims=True)  # per exposure: noise scales with the signal
    ok &= (T.reshape(*T.shape[:2], -1) >= 250).mean(axis=2) == 0
    ok &= (T.reshape(*T.shape[:2], -1) <= 0).mean(axis=2) == 0
    ok &= np.array([[not drifted(T[e, i].mean(axis=(1, 2, 3))) for i in range(T.shape[1])] for e in range(T.shape[0])])
    return ok


def fit_gamma(means, t, ok, grid=np.linspace(0.8, 3.0, 221)):
    """Gamma that makes ``(mean / 255)^gamma`` most linear in ``t`` over the tiles; returns (gamma, black, r2)."""
    best = None
    cols = [i for i in range(means.shape[1]) if ok[:, i].sum() >= 4]
    if not cols:
        return None
    for g in grid:
        res, tot, icpt = 0.0, 0.0, []
        for i in cols:
            m = ok[:, i]
            y = (means[m, i] / 255.0) ** g
            A = np.c_[t[m], np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            res += float(np.sum((A @ coef - y) ** 2))
            tot += float(np.sum((y - y.mean()) ** 2))
            icpt.append(coef[1])
        r2 = 1 - res / tot
        if best is None or r2 > best[2]:
            best = (float(g), float(np.median(icpt)), float(r2))
    return best


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture")
    ap.add_argument("--relative-time", choices=["inverse", "direct"], default="inverse")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = np.load(a.capture)
    rep = dict(timing=timing(d["t_arrival"], d["ts_ms"] / 1e3))
    sweep = ast.literal_eval(str(d["ctl_sweep"]))
    changed = sorted({k for c in sweep for k in c if c[k] != sweep[0][k]} - {"exposure_absolute"})
    if changed:
        rep["warning"] = f"controls other than exposure changed during the sweep: {changed}"
    exp = d["exposures"].astype(float)
    t = 1.0 / exp if a.relative_time == "inverse" else exp
    t = t / t.max()
    T = tiles(d["stacks"])
    if T.shape[-1] == 1:  # YUYV luma is limited range (16-235): map it to full-range DN before linearising
        rep["luma_raw_range"] = [int(d["stacks"].min()), int(d["stacks"].max())]
        # ponytail: one raw step becomes 1.16 DN, so ptc_points' 1/12 Sheppard term is ~36 % small (sub-DN noise only)
        T = np.clip((T - 16.0) * 255.0 / 219.0, 0.0, 255.0)
    ok = usable(T)
    means = T.mean(axis=(2, 3, 4, 5))  # (n_exp, n_tiles)
    fit = fit_gamma(means, t, ok)
    if fit is None:
        rep["response"] = dict(identifiable=False, reason="no tile is still and unclipped over >= 4 exposures")
        return _report(rep, a.out)
    gamma, black, r2 = fit
    rep["response"] = dict(gamma=gamma, black_level_lin=black, linearity_r2=r2, relative_time=a.relative_time,
                           gamma_at_grid_edge=gamma in (0.8, 3.0),
                           exposures=exp.tolist(), usable_tiles=int(ok.sum()), tiles=int(ok.size))
    stacks = [T[e, i] for e in range(T.shape[0]) for i in range(T.shape[1]) if ok[e, i]]
    m, v = ptc_points(stacks, gamma, max(black, 0.0))
    shot, read, pr2, _ = ptc_line(m, v, max(black, 0.0))
    # noise below ~1 DN rms is quantisation-limited at 8 bit: the PTC is then not identifiable
    dn_noise = float(np.sqrt(np.median(T.var(axis=2, ddof=1)[ok])))
    rep["ptc"] = dict(sigma_shot=shot, sigma_read=read, r2=pr2, n_levels=len(m), m_lin_range=[float(m.min()),
                      float(m.max())], pixel_noise_dn_rms=dn_noise,
                      identifiable=bool(pr2 >= 0.9 and dn_noise >= 1.0),
                      controls=ast.literal_eval(str(d["ctl_sweep"]))[0])
    return _report(rep, a.out)


def _report(rep, out):
    print(json.dumps(rep, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    main()
