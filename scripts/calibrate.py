#!/usr/bin/env python3
"""C6: calib_data/{D1..D5} -> camera_config.json + calib_uncertainty.json + calib_report/*.csv.

Usage: calibrate.py ROOT [--out camera_config.json] [--report calib_report] [--pattern 7,5]
       [--square-m 0.05] [--sim-xml M --sim-cam C --sim-roi x0,y0,x1,y1 --sim-reflectance r]
       [--ctrl-hz 500]

Layout: docs/capture_protocol.md. PNGs are read with cv2 (BGR) and converted to RGB.
Order: C1 -> C3 (black 0) <-> black level from D3 (iterated to a fixed point) -> C2 -> C4 -> C5
-> exposure match. Aborts (exit 2, "AE" in stderr) if any D2 level drifts > 1%.
D1/D2 are required; a missing D3 (black 0, no DSNU), D4 (gamma 2.2, exposure_ev 0) or D5 (timing
defaults, no latency/age reports) is skipped with a warning.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from physicalai_sim2real_camera.calib_geom import (  # noqa: E402
    _lin_mean,
    fit_flat_field,
    fit_intrinsics,
    undistorted_r2,
)
from physicalai_sim2real_camera.calib_photo import fit_ptc, fit_response, match_exposure, ptc_points  # noqa: E402
from physicalai_sim2real_camera.calib_timing import fit_timing  # noqa: E402
from physicalai_sim2real_camera.config import CameraConfig, scale_intrinsics  # noqa: E402
from physicalai_sim2real_camera.optics import vignette_map  # noqa: E402
from physicalai_sim2real_camera.sensor import linearize  # noqa: E402
from physicalai_sim2real_camera.timing import CameraStream  # noqa: E402

DRIFT_MAX = 0.01


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def read_dir(d, required=True):
    files = sorted(glob.glob(os.path.join(d, "*.png")),
                   key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])
    if not files:
        if required:
            die(f"no PNGs in {d}")
        return None
    return np.stack([cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in files])


def skip(warn, key, msg):
    warn.setdefault(key, []).append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        f.writelines(",".join("" if v is None else f"{v:.9g}" if isinstance(v, float) else str(v)
                              for v in r) + "\n" for r in rows)


class _Data:  # minimal mjData stand-in for CameraStream
    time = 0.0


class _NullCam:
    def render(self, data):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="calib_data/{D1..D5} -> camera_config.json")
    ap.add_argument("root")
    ap.add_argument("--out", default="camera_config.json")
    ap.add_argument("--report", default="calib_report")
    ap.add_argument("--pattern", default="7,5")
    ap.add_argument("--square-m", type=float, default=0.05)
    ap.add_argument("--sim-xml")
    ap.add_argument("--sim-cam")
    ap.add_argument("--sim-roi")
    ap.add_argument("--sim-reflectance", type=float)
    ap.add_argument("--ctrl-hz", type=float, default=500.0)
    ap.add_argument("--flat-field", action="store_true",
                    help="store a measured (H,W,1) vignette*PRNU map instead of the analytic model "
                         "(sets vignette=0, sigma_prnu=0 to avoid double-application)")
    a = ap.parse_args(argv)
    R, warn = a.root, {"C1": [], "C2": [], "C3": [], "C4": [], "C5": [], "exposure": []}
    sim_args = [a.sim_xml, a.sim_cam, a.sim_roi, a.sim_reflectance]
    if any(x is not None for x in sim_args) and not all(x is not None for x in sim_args):
        die("--sim-xml/--sim-cam/--sim-roi/--sim-reflectance must be given together")
    pattern = tuple(int(x) for x in a.pattern.split(","))

    # --- load + AE check first (fail fast before any fit) ---
    levels = sorted(glob.glob(os.path.join(R, "D2", "level_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    if not levels:
        die(f"no D2/level_<k> dirs under {R}")
    stacks = [read_dir(p) for p in levels]
    for p, s in zip(levels, stacks):
        m = s.reshape(len(s), -1).mean(axis=1)
        drift = abs(np.polyfit(np.arange(len(m)), m, 1)[0]) * (len(m) - 1) / m.mean()
        if drift > DRIFT_MAX:
            die(f"{os.path.relpath(p, R)} frame-mean drift {drift:.2%} > {DRIFT_MAX:.0%}: "
                "AE/auto-gain appears to be on — lock exposure/gain and recapture D2")
    d1 = read_dir(os.path.join(R, "D1"))
    H1, W1 = d1.shape[1:3]
    dark = read_dir(os.path.join(R, "D3"), required=False)
    if dark is None:
        skip(warn, "C2", "no D3: black_level 0, sigma_dsnu 0, vignette/PRNU fit with black included")
    patches = read_dir(os.path.join(R, "D4"), required=False)
    refl_csv = os.path.join(R, "D4", "reflectances.csv")
    if patches is not None and os.path.exists(refl_csv):
        refl = np.loadtxt(refl_csv, ndmin=1)
        if len(refl) != len(patches):
            die(f"D4: {len(patches)} images but {len(refl)} reflectances")
    else:
        patches = None
        skip(warn, "C3", "no D4 images/reflectances.csv: gamma 2.2, exposure_ev 0")
    ev_csv, tm = os.path.join(R, "D5", "events.csv"), None
    if os.path.exists(ev_csv):
        ev_tab = np.genfromtxt(ev_csv, delimiter=",", names=True)
        fr = ~np.isnan(ev_tab["avail_t"]) & ~np.isnan(ev_tab["frame_seq"])
        o = np.argsort(ev_tab["frame_seq"][fr], kind="stable")
        av, sq = ev_tab["avail_t"][fr][o], ev_tab["frame_seq"][fr][o].astype(np.int64)
        seen = ev_tab["event_seen"][fr][o].astype(np.int64)
        event_t = np.sort(ev_tab["event_t"][~np.isnan(ev_tab["event_t"])])
    else:
        skip(warn, "C5", "no D5/events.csv: timing fields keep defaults, no latency/age reports")

    # --- C1 ---
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        K, dist, rms, per_view = fit_intrinsics(list(d1), pattern, a.square_m)
    warn["C1"] = [str(x.message) for x in w]

    # --- C3 <-> black level fixed point (C3 with black 0 is biased; black needs gamma) ---
    gamma, res, it, black = CameraConfig.gamma, None, 0, 0.0
    if patches is not None:
        dn = patches.reshape(len(patches), -1).mean(axis=1)
        gamma, res = fit_response(dn, refl, 0.0)
    gamma0 = gamma
    if dark is not None:
        for it in range(1, 51):
            black = float(_lin_mean(dark, gamma).mean())
            if patches is None:
                break
            g_new, res = fit_response(dn, refl, black)
            done = abs(g_new - gamma) < 1e-7
            gamma = g_new
            if done:
                break
        else:
            skip(warn, "C3", f"C3<->black not converged in 50 iterations (last gamma {gamma:.5f})")

    # --- C2 (K rescaled if D2/D3 were captured at another resolution) ---
    H2, W2 = stacks[0].shape[1:3]
    sx, sy = W2 / W1, H2 / H1
    K2 = scale_intrinsics(K, sx, sy)
    means = [s.mean(axis=0) for s in stacks]
    ok = [i for i, m in enumerate(means) if np.percentile(m, 99) < 250]
    if not ok:
        die("every D2 level is saturated (99th pct >= 250 DN)")
    ib = ok[-1]
    # ponytail: no D3 -> an all-zero dark frame (black 0); vignette then absorbs the true black a bit
    dark_c2 = dark if dark is not None else np.zeros_like(stacks[ib][:1])
    vign, prnu_g, dsnu_o, s_prnu, s_dsnu, black_c2 = fit_flat_field(stacks[ib], dark_c2, K2, gamma, dist)

    # --- C4 ---
    m_pts, v_pts = ptc_points(stacks, gamma, black)
    if len(m_pts) >= 2:
        s_shot, s_read, r2 = fit_ptc(stacks, gamma, black)
        slope, icpt = np.polyfit(m_pts, v_pts, 1)
    else:  # clipped / saturated / near-black levels are left out of the PTC: too few may remain
        skip(warn, "C4", f"only {len(m_pts)} usable D2 level(s) for the PTC: sigma_shot / sigma_read keep defaults")
        s_shot, s_read, r2, slope, icpt = 0.0, 0.0, float("nan"), float("nan"), float("nan")

    # --- C5 ---
    timing = {}
    if os.path.exists(ev_csv):
        tm = fit_timing(event_t, av, sq, seen)
        warn["C5"] += tm["warnings"]
        if tm["n_samples"] == 0 or not np.isfinite(tm["latency_s"]):
            skip(warn, "C5", "no usable LED samples: timing fields keep defaults")
            tm = None
        else:
            timing = dict(fps=tm["fps"], latency_s=tm["latency_s"], jitter_k=2.0,
                          jitter_theta=tm["jitter_theta"], drop_prob=tm["drop_prob"])

    cfg = CameraConfig(W1, H1, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                       cy=float(K[1, 2]), dist=dist, vignette=vign, gamma=gamma,
                       sigma_shot=s_shot, sigma_read=s_read, sigma_prnu=s_prnu, sigma_dsnu=s_dsnu,
                       black_level=black, **timing)

    # --- exposure match: real D4 patch (linear, black removed) vs sim patch per unit reflectance ---
    ev, ev_method, real, keep = 0.0, "skipped (no D4)", None, np.zeros(0, bool)
    if patches is not None:
        keep = (dn > 5) & (dn < 250)
        real = float(np.sum((dn[keep] / 255.0) ** gamma - black))
        if a.sim_xml is not None:
            import mujoco

            from physicalai_sim2real_camera import SimCamera

            model = mujoco.MjModel.from_xml_path(a.sim_xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            cam = SimCamera(model, a.sim_cam, dataclasses.replace(
                cfg, sigma_shot=0, sigma_read=0, sigma_prnu=0, sigma_dsnu=0, black_level=0, exposure_ev=0))
            try:
                img = cam.render(data)
            finally:
                cam.close()
            x0, y0, x1, y1 = (int(v) for v in a.sim_roi.split(","))
            unit = float(linearize(img[y0:y1, x0:x1], gamma).mean()) / a.sim_reflectance
            ev_method = "sim render"
        else:
            unit, ev_method = 1.0, "analytic (illum=1)"
            skip(warn, "exposure", "no --sim-*: exposure_ev assumes sim illumination 1 "
                 "(patch linear value = reflectance)")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev = match_exposure(real, unit * float(np.sum(refl[keep])))
        warn["exposure"] += [str(x.message) for x in w]
    cfg = dataclasses.replace(cfg, exposure_ev=ev)

    flat_field = None
    if a.flat_field:
        flat_field = (vignette_map(K2, dist, W2, H2, vign) * prnu_g).astype(np.float32)
        if (H2, W2) != (H1, W1):
            skip(warn, "C2", f"--flat-field: resizing measured map {W2}x{H2} -> {W1}x{H1} "
                 "(D2/D3 captured at a different resolution than D1)")
            flat_field = cv2.resize(flat_field[..., 0], (W1, H1),
                                    interpolation=cv2.INTER_LINEAR)[..., None].astype(np.float32)
        skip(warn, "C2", "--flat-field: PRNU baked into the measured map -> vignette=0, sigma_prnu=0 "
             "saved (avoids double application); per-episode PRNU resampling is disabled")
        cfg = dataclasses.replace(cfg, flat_field=flat_field, vignette=0.0, sigma_prnu=0.0)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(a.out)
    unc = {
        "C1": dict(rms_px=rms, per_view_err_px=per_view.tolist(), n_views=len(per_view), n_images=len(d1),
                   warnings=warn["C1"]),
        "C3": dict(gamma_black0=gamma0, gamma=gamma, residual=res, iterations=it,
                   n_patches=int(keep.sum()), warnings=warn["C3"]),
        "C2": dict(bright_level=os.path.basename(levels[ib]), black_level_c2=black_c2, n_bright=len(stacks[ib]),
                   n_dark=0 if dark is None else len(dark), resolution=[W2, H2], warnings=warn["C2"]),
        "C4": dict(r2=r2, slope=float(slope), intercept=float(icpt), n_levels=len(m_pts), warnings=warn["C4"]),
        "C5": dict(n_samples=tm["n_samples"] if tm else 0,
                   latency_sample_std_s=float(np.std(tm["samples"])) if tm else None, warnings=warn["C5"]),
        "exposure": dict(ev=ev, method=ev_method, real_lin_sum=real, warnings=warn["exposure"]),
    }
    with open(os.path.join(out_dir, "calib_uncertainty.json"), "w") as f:
        json.dump(unc, f, indent=2)

    # --- reports ---
    rep = a.report
    os.makedirs(rep, exist_ok=True)
    write_csv(os.path.join(rep, "ptc.csv"), "kind,m_lin,v_lin",
              [("point", float(m), float(v)) for m, v in zip(m_pts, v_pts)]
              + [("fit_slope_intercept", float(slope), float(icpt))])

    S = _lin_mean(stacks[ib], gamma)[..., 0] - _lin_mean(dark_c2, gamma)[..., 0]
    r = np.sqrt(undistorted_r2(K2, dist, W2, H2))
    V = (1.0 + r * r) ** (-2.0 * vign)
    S0 = float(np.median(S / V))
    edges = np.linspace(0, r.max() + 1e-9, 21)
    b = np.digitize(r, edges) - 1
    write_csv(os.path.join(rep, "radial_profile.csv"), "r_lo,r_hi,n_px,measured_lin,fitted_lin",
              [(float(edges[k]), float(edges[k + 1]), int((b == k).sum()), float(S[b == k].mean()),
                float(S0 * V[b == k].mean())) for k in range(20) if (b == k).any()])

    ks = None
    if tm is not None:
        write_csv(os.path.join(rep, "latency_samples.csv"), "sample_s", [(float(x),) for x in tm["samples"]])
        # age: virtual control clock at ctrl_hz, sample-and-hold. Real capture_t = c + n*P from C5's
        # event-bracketed grid (independent of the fitted latency). On an irregular grid there is no
        # independent capture time, so the age report is skipped instead of deriving c from the fitted
        # latency (that would make the KS blind to latency error).
        P, c = tm["period"], tm["grid_offset"]
    if tm is not None and c is None:
        skip(warn, "AGE", "irregular capture grid: age.csv skipped (capture times not recoverable "
             "independently of the fitted latency)")
    if tm is not None and c is not None:
        o = np.argsort(av, kind="stable")
        av_s, sq_s = av[o], np.maximum.accumulate(sq[o])
        t_ctrl = np.arange(av_s[0], av_s[-1], 1.0 / a.ctrl_hz)
        real_age = t_ctrl - (c + sq_s[np.searchsorted(av_s, t_ctrl, side="right") - 1] * P)
        stream, data = CameraStream(_NullCam(), cfg), _Data()
        sim_age = []
        for t in np.arange(0.0, av_s[-1] - av_s[0], 1.0 / a.ctrl_hz):
            data.time = float(t)
            _, cap_t, _ = stream.step(data)
            if cap_t is not None:
                sim_age.append(t - cap_t)
        sim_age = np.array(sim_age)
        grid = np.sort(np.concatenate([real_age, sim_age]))
        ks = float(np.max(np.abs(np.searchsorted(np.sort(real_age), grid, side="right") / len(real_age)
                                 - np.searchsorted(np.sort(sim_age), grid, side="right") / len(sim_age))))
        n_age = max(len(real_age), len(sim_age))
        with open(os.path.join(rep, "age.csv"), "w") as f:
            f.write(f"ks,{ks:.6f}\nreal_age_s,sim_age_s\n")
            f.writelines(f"{real_age[i] if i < len(real_age) else ''},{sim_age[i] if i < len(sim_age) else ''}\n"
                         for i in range(n_age))
    # reprojection residual per 32x24 cell: corners re-detected, pose by solvePnP with the fitted K, dist
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * a.square_m
    Kd, dd = np.asarray(K, np.float64), np.asarray(dist, np.float64)
    pts, errs = [], []
    for img in d1:
        found, cor = cv2.findChessboardCornersSB(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), pattern)
        if not found:
            continue
        cor = cor.reshape(-1, 2).astype(np.float64)
        _, rv, tv = cv2.solvePnP(obj, cor, Kd, dd)
        proj = cv2.projectPoints(obj, rv, tv, Kd, dd)[0].reshape(-1, 2)
        pts.append(cor)
        errs.append(np.linalg.norm(proj - cor, axis=1))
    pts, errs = np.vstack(pts), np.concatenate(errs)
    gx = np.minimum((pts[:, 0] / W1 * 32).astype(int), 31)
    gy = np.minimum((pts[:, 1] / H1 * 24).astype(int), 23)
    rows = []
    for j in range(24):
        for i in range(32):
            e = errs[(gx == i) & (gy == j)]
            rows.append(((i + 0.5) * W1 / 32, (j + 0.5) * H1 / 24, len(e),
                         float(e.mean()) if len(e) else None))
    write_csv(os.path.join(rep, "reprojection_grid.csv"), "u,v,n_corners,mean_residual_px", rows)

    print(f"C1  K: fx={cfg.fx:.3f} fy={cfg.fy:.3f} cx={cfg.cx:.3f} cy={cfg.cy:.3f} rms={rms:.4f}px "
          f"views={len(per_view)}/{len(d1)}")
    print(f"C1  dist={np.round(dist, 5).tolist()}")
    print(f"C3  gamma={gamma:.4f} (black-0 pass {gamma0:.4f}, {it} iters) residual={res}")
    print(f"C2  vignette={vign:.4f} sigma_prnu={s_prnu:.5f} sigma_dsnu={s_dsnu:.5f} "
          f"black_level={black:.5f} (bright={os.path.basename(levels[ib])})")
    print(f"C4  sigma_shot={s_shot:.5f} sigma_read={s_read:.5f} r2={r2:.5f} levels={len(m_pts)}")
    print(f"C5  fps={cfg.fps:.4f} latency_s={cfg.latency_s:.5f} jitter_theta={cfg.jitter_theta:.5f} "
          f"drop_prob={cfg.drop_prob:.4f} n={tm['n_samples'] if tm else 0}")
    print(f"EV  exposure_ev={ev:.4f} ({ev_method})")
    if flat_field is not None:
        print(f"C2  flat_field: measured {flat_field.shape[1]}x{flat_field.shape[0]}x1 map saved "
              f"(vignette*PRNU baked in; vignette=0 sigma_prnu=0; per-episode PRNU resampling disabled)")
    if ks is not None:
        print(f"AGE KS={ks:.4f} (real n={len(real_age)}, sim n={len(sim_age)}, ctrl {a.ctrl_hz:g} Hz)")
    print(f"OUT {a.out}, {os.path.join(out_dir, 'calib_uncertainty.json')}")
    print(f"REP {rep}/{{ptc,radial_profile,latency_samples,age,reprojection_grid}}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
