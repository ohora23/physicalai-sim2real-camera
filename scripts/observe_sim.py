"""O6 closed loop: the observer on the simulated prism sweep, estimate ± std vs truth over seeds.

    MUJOCO_GL=egl python scripts/observe_sim.py --seeds 20 --out out/o6

Per seed and shutter setting (``rolling``: exposure 8 ms / readout 16 ms; ``global``: readout 0) it writes the 1 Hz
estimate series and the final errors against the truth to ``<out>/<shutter>_<seed>.json`` and a summary to
``<out>/summary.json``. The observer's gamma / black level come from ``scripts/calibrate.py`` run on
``tests/synth.write_calib_data`` output for the same truth (leakage item (c)); its prism is ``perturb_prism`` of the
built one (b), its initial K the nominal fovy K x 1.05 (a), its head input lags 4 ms with noise (d); the distortion
model has the same 5-coefficient form in the simulator and the fit (e, allowed and stated). The rolling-shutter
readout is the true value (allowed and stated: it is an offline D5b input to the observer).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "prism")
W, H, FOVY = 640, 480, 45.0
SHUTTERS = {"rolling": 0.016, "global": 0.0}
PRISM_YAW_DEG = -60.0  # full face 4 faces the head, the wedge face (5) at 60 deg - orbit (scenes.prism_scene)
FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")


def truth_config(seed, readout_s):
    """The simulator's camera: K off the nominal start, distortion, noise, latency / jitter / drops, 8 ms exposure."""
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.observer_geom import nominal_K
    f_nom = nominal_K(FOVY, W, H)[0, 0]
    return CameraConfig(W, H, fx=0.98 * f_nom, fy=0.98 * f_nom, cx=(W - 1) / 2 + 7, cy=(H - 1) / 2 - 5,
                        dist=(-0.3, 0.1, 0.001, -0.001, 0.0), gamma=2.2, black_level=0.02, sigma_shot=0.02,
                        sigma_read=0.005, fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.05,
                        exposure_s=0.008, readout_s=readout_s, seed=seed)


def calibrate_photometric(cfg, workdir):
    """``(gamma, black_level)`` from ``scripts/calibrate.py`` on synthetic D1-D5 data of ``cfg`` (never ``cfg``)."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from synth import write_calib_data

    from physicalai_sim2real_camera import CameraConfig
    root, out = os.path.join(workdir, "calib_data"), os.path.join(workdir, "calib_out")
    write_calib_data(dataclasses.replace(cfg, exposure_s=0.0, readout_s=0.0), root, np.random.default_rng(0),
                     flat_size=(64, 48))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "calibrate.py"), root, "--out",
                    f"{out}/camera_config.json", "--report", f"{out}/calib_report"], check=True, capture_output=True,
                   env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    est = CameraConfig.load(f"{out}/camera_config.json")
    return est.gamma, est.black_level


def run_seed(seed, readout_s, gamma, black, duration_s=16.6, every_s=1.0, cfg=None):
    """One closed-loop run: truth, 1 Hz estimate series, corner hull and realised camera drop rate.

    The realised rate (the drops this run's random draws produced) is the reference for the drop estimate. ``cfg``
    replaces ``truth_config(seed, readout_s)`` as the simulated camera (the feedback closure runs the fed-back one).
    """
    import cv2

    from physicalai_sim2real_camera import SimCamera
    from physicalai_sim2real_camera.observer_replay import Observer, perturb_prism, replay, sim_records
    from physicalai_sim2real_camera.scenes import HeadPoseRing, prism_scene
    from physicalai_sim2real_camera.target import PrismTarget
    cfg = truth_config(seed, readout_s) if cfg is None else cfg
    model, data, sweep = prism_scene(os.path.join(ASSETS, "prism.xml"), width=W, height=H, prism_yaw_deg=PRISM_YAW_DEG)
    cam = SimCamera(model, "cam", cfg)
    cv2.setRNGSeed(seed)  # SimCamera seeds cv2's global noise RNG once per process
    target = perturb_prism(PrismTarget.from_json(os.path.join(ASSETS, "prism.json")), np.random.default_rng(seed))
    obs = Observer(target, (W, H), FOVY, gamma, black, exposure_s=cfg.exposure_s, readout_s=readout_s)
    ring = HeadPoseRing()
    stats = {}
    try:
        records = sim_records(model, data, sweep, cam, cfg, duration_s, ring=ring, seed=seed, stats=stats)
        series = replay(records, obs, ring, every_s=every_s, rng=np.random.default_rng(seed))
        K_true = cam.K_out.copy()
    finally:
        cam.close()
        ring.close()
    _, pts = obs.geom.window.views()
    hull = np.concatenate([p.reshape(-1, 2) for p in pts]) if pts else np.zeros((0, 2))
    fns = stats["frame_numbers"]
    return cfg, K_true, series, hull, stats["dropped"] / (fns[-1] - fns[0])


def errors(cfg, K_true, est, hull, drop_true):
    """Final errors against the truth: O2 (relative f, centre px, field px, |err|/σ per field), O3, O4.

    Drops are compared with the realised rate ``drop_true``: with ~500 frames the draw alone scatters ±1 %p around
    ``drop_prob``.
    """
    from physicalai_sim2real_camera.calib_geom import distortion_field_error
    out = {}
    g = est["geom"]
    if g is not None:
        K, d, std = g["K"], g["dist"], g["std"]
        truth = dict(zip(FIELDS, (K_true[0, 0], K_true[1, 1], K_true[0, 2], K_true[1, 2], *cfg.dist)))
        val = dict(zip(FIELDS, (K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d)))
        out.update(fx_rel=max(abs(val["fx"] / truth["fx"] - 1), abs(val["fy"] / truth["fy"] - 1)),
                   centre_px=max(abs(val["cx"] - truth["cx"]), abs(val["cy"] - truth["cy"])),
                   field_px=distortion_field_error(K_true, cfg.dist, K, d, hull, W, H),
                   z={f: abs(val[f] - truth[f]) / max(std[f], 1e-12) for f in FIELDS}, n_views=g["n_views"],
                   sigma_centre=max(std["cx"], std["cy"]))
    p = est["photo"]
    if p is not None and p["sigma_shot"] is not None:
        out.update(shot_rel=p["sigma_shot"] / cfg.sigma_shot - 1, read_rel=p["sigma_read"] / cfg.sigma_read - 1,
                   photo_intervals=p["n_intervals"], photo_span=p["span"],
                   shot_z=abs(p["sigma_shot"] - cfg.sigma_shot) / p["sigma_shot_std"],
                   read_z=abs(p["sigma_read"] - cfg.sigma_read) / p["sigma_read_std"])
    t = est["timing"]
    if "latency_s" in t:
        out.update(latency_err_s=t["latency_s"] - cfg.latency_s, latency_std_s=t["latency_std"])
    if "jitter_theta" in t:
        out.update(theta_rel=t["jitter_theta"] / cfg.jitter_theta - 1, drop_err=t["camera_drop_rate"] - drop_true,
                   miss_rate=t["observer_miss_rate"], theta_z=abs(t["jitter_theta"] - cfg.jitter_theta) /
                   t["jitter_theta_std"], drop_z=abs(t["camera_drop_rate"] - cfg.drop_prob) / t["camera_drop_std"])
    return out


def _shift_geom(est, cx=0.0, f=1.0):
    """Copy of ``est`` whose camera moved: principal point ``+cx`` px, focal length ``x f``."""
    g = dict(est["geom"])
    K = np.array(g["K"], float)
    K[0, 0] *= f
    K[1, 1] *= f
    K[0, 2] += cx
    g["K"] = K
    return {**est, "geom": g}


def _shift_latency(est, dt):
    """Copy of ``est`` whose latency moved by ``dt`` s."""
    return {**est, "timing": {**est["timing"], "latency_s": est["timing"].get("latency_s", np.nan) + dt}}


# changes the O8 gate should catch between two windows (newer window's estimate shifted by the change)
CHANGES = {"cx+3px": ("geom", lambda e: _shift_geom(e, cx=3.0)), "f+1%": ("geom", lambda e: _shift_geom(e, f=1.01)),
           "latency+5ms": ("timing", lambda e: _shift_latency(e, 5e-3))}


GROUP_Z = dict(geom=[f"z.{f}" for f in FIELDS], photo=["shot_z", "read_z"], timing=["latency_z", "theta_z"],
               drop=["drop_z"])


def closure(a, b, readout_s, gamma, black, seeds):
    """Feedback closure: feed two truth runs back, simulate the fed-back camera, re-estimate.

    ``cfg1, spread = estimate_to_config(est_b, est_a, truth)``; two runs of ``cfg1`` (``seeds``) are re-estimated.
    A group is a fixed point when every field of both re-estimates is within 3 of its own std of ``cfg1`` (the
    calibrated O6 bound); the DR covers the truth when every fed-back field's truth lies in ``cfg1 +- spread``
    (``randomize``'s convention).

    Args:
        a: ``run_seed`` result of the first truth run.
        b: ``run_seed`` result of the second truth run (disjoint window: another seed).
        readout_s: Rolling-shutter readout.
        gamma: Offline gamma.
        black: Offline black level.
        seeds: Two seeds for the runs of the fed-back camera.

    Returns:
        dict ``report`` (estimate_to_config), ``fixed`` (group -> bool, updated groups only), ``covered`` (field ->
        bool), ``z`` (the re-estimates' |err| / std per field, worst of the two runs).
    """
    from physicalai_sim2real_camera.calib import _ABSOLUTE, estimate_to_config
    cfg_t, K_t = b[0], a[1]
    # base = the truth with the offline photometric constants the observer uses (fields not fed back keep those)
    base = dataclasses.replace(cfg_t, gamma=gamma, black_level=black)
    cfg1, spread, report = estimate_to_config(b[2][-1][1], a[2][-1][1], base)
    z = {}
    for seed in seeds:
        c = run_seed(seed, readout_s, gamma, black, cfg=dataclasses.replace(cfg1, seed=seed))
        e = errors(c[0], c[1], c[2][-1][1], c[3], c[4])
        flat = {**{f"z.{f}": v for f, v in e.get("z", {}).items()},
                **{k: e[k] for k in ("shot_z", "read_z", "theta_z", "drop_z") if k in e}}
        if "latency_err_s" in e:
            flat["latency_z"] = abs(e["latency_err_s"]) / e["latency_std_s"]
        for k, v in flat.items():
            z[k] = max(z.get(k, 0.0), float(v))
    fixed = {g: all(z.get(k, np.inf) <= 3 for k in GROUP_Z[g]) for g, r in report.items() if r == "updated"}
    truth = dict(fx=K_t[0, 0], fy=K_t[1, 1], cx=K_t[0, 2], cy=K_t[1, 2], sigma_shot=cfg_t.sigma_shot,
                 sigma_read=cfg_t.sigma_read, latency_s=cfg_t.latency_s, jitter_theta=cfg_t.jitter_theta,
                 drop_prob=cfg_t.drop_prob)
    covered = {}
    for f, s_ in spread.items():
        if f == "dist" and isinstance(s_, tuple):
            for k, (v1, vt, sk) in enumerate(zip(cfg1.dist, cfg_t.dist, s_)):
                covered[FIELDS[4 + k]] = abs(v1 - vt) <= sk
        elif f in truth and report.get({"fx": "geom", "fy": "geom", "cx": "geom", "cy": "geom", "sigma_shot": "photo",
                                         "sigma_read": "photo", "latency_s": "timing", "jitter_theta": "timing",
                                         "drop_prob": "drop"}[f]) == "updated":
            v1 = getattr(cfg1, f)
            covered[f] = abs(v1 - truth[f]) <= (s_ if f in _ABSOLUTE else s_ * abs(v1))
    return dict(report=report, fixed=fixed, covered=covered, z=z)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--shutter", choices=sorted(SHUTTERS) + ["both"], default="both")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "o6"))
    ap.add_argument("--closure", type=int, default=0,
                    help="feedback closure: this many trials per shutter instead of the O6 statistics")
    args = ap.parse_args(argv)
    from physicalai_sim2real_camera.calib import estimate_to_config
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        gamma, black = calibrate_photometric(truth_config(0, 0.0), tmp)
    print(f"offline photometric constants: gamma {gamma:.4f}, black {black:.5f}", flush=True)
    summary = {}
    if args.closure:
        for name in SHUTTERS if args.shutter == "both" else [args.shutter]:
            trials = []
            for k in range(args.first_seed, args.first_seed + args.closure):
                ra, rb = (run_seed(s, SHUTTERS[name], gamma, black) for s in (2 * k, 2 * k + 1))
                trials.append(closure(ra, rb, SHUTTERS[name], gamma, black, (1000 + 2 * k, 1001 + 2 * k)))
                print(f"{name} closure {k}: {trials[-1]['report']} fixed {trials[-1]['fixed']} "
                      f"covered {trials[-1]['covered']}", flush=True)
            # over all trials: a group that was not updated counts as not fixed, a field not fed back as not covered
            fields = list(FIELDS) + ["sigma_shot", "sigma_read", "latency_s", "jitter_theta", "drop_prob"]
            summary[name] = dict(
                n=len(trials), fixed={g: float(np.mean([t["fixed"].get(g, False) for t in trials])) for g in GROUP_Z},
                updated={g: float(np.mean([g in t["fixed"] for t in trials])) for g in GROUP_Z},
                covered={f: float(np.mean([t["covered"].get(f, False) for t in trials])) for f in fields},
                trials=trials)
            print(name, json.dumps({k: v for k, v in summary[name].items() if k != "trials"}, indent=1), flush=True)
        with open(os.path.join(args.out, "closure.json"), "w") as f:
            json.dump(_jsonable(dict(gamma=gamma, black_level=black, **summary)), f, indent=1)
        return
    for name in SHUTTERS if args.shutter == "both" else [args.shutter]:
        rows, finals = [], []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            cfg, K_true, series, hull, drop_true = run_seed(seed, SHUTTERS[name], gamma, black)
            err = errors(cfg, K_true, series[-1][1], hull, drop_true)
            rows.append(err)
            finals.append((cfg, series[-1][1]))
            ts = [dict(t=t, err=errors(cfg, K_true, e, hull, drop_true),
                       geom_std=None if e["geom"] is None else e["geom"]["std"], health=e["health"]) for t, e in series]
            with open(os.path.join(args.out, f"{name}_{seed}.json"), "w") as f:
                json.dump(_jsonable(dict(final=err, series=ts, coverage=series[-1][1]["geom"]["coverage"]
                                         if series[-1][1]["geom"] else None, estimate=series[-1][1])), f, indent=1)
            print(f"{name} seed {seed}: " + ", ".join(f"{k} {v:.4g}" for k, v in err.items()
                                                        if isinstance(v, float)), flush=True)
        z_ok = {f: float(np.mean([r["z"][f] <= 3 for r in rows if "z" in r])) for f in FIELDS}
        worst = {k: float(np.max(np.abs([r[k] for r in rows if k in r])))
                 for k in ("fx_rel", "centre_px", "field_px", "shot_rel", "read_rel", "latency_err_s", "theta_rel",
                           "drop_err") if any(k in r for r in rows)}
        # O8: consecutive seeds are two disjoint windows of the same camera; the gate should pass them, and catch the
        # same pairs with the newer window's camera changed (re-mount / refocus / timing change)
        pairs = list(zip(finals[::2], finals[1::2]))
        agree = [estimate_to_config(b, a, cfg)[2] for (_, a), (cfg, b) in pairs]
        updated = {g: float(np.mean([r[g] == "updated" for r in agree])) for g in agree[0]} if agree else {}
        caught = {name_: float(np.mean([estimate_to_config(change(b), a, cfg)[2][grp] != "updated"
                                        for (_, a), (cfg, b) in pairs]))
                  for name_, (grp, change) in CHANGES.items()} if pairs else {}
        z_more = {k: float(np.mean([r[k] <= 3 for r in rows if k in r]))
                  for k in ("shot_z", "read_z", "theta_z", "drop_z")}
        summary[name] = dict(n=len(rows), z_le_3=z_ok, z_le_3_photo_timing=z_more, worst=worst,
                             o8_updated=updated, o8_caught=caught, o8_reports=agree)
        print(name, json.dumps(summary[name], indent=1), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(gamma=gamma, black_level=black, **summary), f, indent=1)


if __name__ == "__main__":
    main()
