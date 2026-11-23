"""Calibration facade: re-exports calib_geom/calib_photo/calib_timing, plus randomize() and estimate_to_config()."""

from __future__ import annotations

import dataclasses
from statistics import NormalDist

import numpy as np

from .calib_geom import fit_flat_field, fit_intrinsics
from .calib_photo import fit_ptc, fit_response, match_exposure
from .calib_timing import fit_timing
from .optics import is_rational

__all__ = [
    "fit_intrinsics",
    "fit_flat_field",
    "fit_response",
    "fit_ptc",
    "match_exposure",
    "fit_timing",
    "randomize",
    "estimate_to_config",
]

# spread semantics: "abs" fields get v + s*U(-1,1); everything else gets the
# relative form v*(1 + s*U(-1,1)). `dist` (abs) is perturbed per-coefficient.
_DEFAULT_SPREAD = {
    "dist": 0.02,          # absolute +/-, per coefficient
    "sigma_shot": 0.2,     # relative
    "sigma_read": 0.2,     # relative
    "sigma_prnu": 0.2,     # relative
    "sigma_dsnu": 0.2,     # relative
    "vignette": 0.2,       # relative
    "latency_s": 0.2,      # relative
    "jitter_theta": 0.2,   # relative
    "gamma": 0.05,         # relative
    "exposure_ev": 0.3,    # absolute +/-
    "drop_prob": 0.02,     # absolute +/-
}


def _dist_spread(s, dist) -> np.ndarray:
    """Per-coefficient ``dist`` spread: scalar -> every term of a 4/5-term dist, none of a rational one.

    A shorter tuple is zero-padded.

    A scalar per-coefficient spread suits narrow lenses only. On a wide-angle rational model the high-order terms
    multiply r^4-r^6 > 60 at the image edge, so a small spread per term can push a large part of the image past the
    model's fold. Such a calibrated lens keeps its model; pass an explicit per-term tuple to randomize it anyway.
    """
    s = np.asarray(s, np.float64)
    if s.ndim == 0:
        return np.full(len(dist), 0.0 if is_rational(dist) else float(s))
    if s.size > len(dist):
        raise ValueError(f"dist spread has {s.size} terms for a {len(dist)}-term dist")
    return np.pad(s, (0, len(dist) - s.size))


_ABSOLUTE = {"dist", "ccm", "exposure_ev", "drop_prob", "cx", "cy", "flicker_phase"}
_PROB_FIELDS = {"drop_prob", "flicker_amp", "dirt_scatter"}
_SIGMA_FIELDS = {"sigma_shot", "sigma_read", "sigma_prnu", "sigma_dsnu", "vignette", "jitter_theta",
                 "dirt_density", "psf_sigma_px", "mount_vib_rot_deg", "mount_vib_trans_m"}


def randomize(cfg, spread=None, rng=None):
    """Domain-randomize a fit CameraConfig for one training episode.

    For each field in `spread` (default `_DEFAULT_SPREAD`), samples a new
    value around the current one: fields in `_ABSOLUTE` (`dist`, `ccm`, `exposure_ev`,
    `drop_prob`, `cx`, `cy`, `flicker_phase`) get `v + s * U(-1, 1)` (`dist` and `ccm` are perturbed
    per-coefficient, `s` a scalar or a per-coefficient tuple; a `None` ccm starts from the identity; a scalar
    `s` leaves a rational, wide-angle `dist` unchanged, see `_dist_spread`);
    every other field gets the relative form `v * (1 + s * U(-1, 1))`, so a field at 0 stays 0 (start DR of
    `dirt_*` / `flicker_amp` from a non-zero fit). `seed`
    is always resampled as `int(rng.integers(2**31))` so per-pixel FPN (and the lens-dirt pattern) differs
    every episode. Fractions (`drop_prob`, `flicker_amp`, `dirt_scatter`) are clipped to [0, 1], sigma-like fields
    (`sigma_shot`, `sigma_read`, `sigma_prnu`, `sigma_dsnu`, `vignette`, `jitter_theta`, `dirt_density`,
    `psf_sigma_px`, `mount_vib_rot_deg`, `mount_vib_trans_m`) to >= 0, and `jpeg_quality` is rounded into [1, 100]
    (0 = off stays off).
    Returns a new `CameraConfig` via `dataclasses.replace`; does not mutate `cfg`.
    """
    if spread is None:
        spread = _DEFAULT_SPREAD
    if rng is None:
        rng = np.random.default_rng()

    updates = {}
    for field, s in spread.items():
        v = getattr(cfg, field)
        if field == "ccm" and v is None:  # randomize around the identity matrix
            v = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if field == "dist":
            new_v = tuple(float(x + si * rng.uniform(-1, 1)) for x, si in zip(v, _dist_spread(s, v)))
        elif field == "ccm":
            new_v = tuple(float(x + si * rng.uniform(-1, 1)) for x, si in zip(v, np.broadcast_to(s, len(v))))
        elif field in _ABSOLUTE:
            new_v = float(v + s * rng.uniform(-1, 1))
        else:
            new_v = float(v * (1 + s * rng.uniform(-1, 1)))
            if field in _SIGMA_FIELDS:
                new_v = max(new_v, 0.0)
        if field in _PROB_FIELDS:
            new_v = min(max(new_v, 0.0), 1.0)
        if field == "jpeg_quality":  # relative spread around the fit quality, a valid int; off (0) stays off
            new_v = 0 if v == 0 else int(min(max(round(new_v), 1), 100))
        updates[field] = new_v
    updates["seed"] = int(rng.integers(2**31))
    return dataclasses.replace(cfg, **updates)


# --- O8: observer estimate -> CameraConfig + DR spread ---
GEOM_FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
# floors for the estimated fields that randomize() has no default for: the O2 tolerances (0.5 %, 3 px)
_ESTIMATE_FLOOR = {**_DEFAULT_SPREAD, "fx": 0.005, "fy": 0.005, "cx": 3.0, "cy": 3.0}
GEOM_MIN_VIEWS = 15     # O6 ends a 16.6 s sweep with 20 views
GEOM_MIN_COVERAGE = 0.5  # fraction of coverage cells with a view (O6: 0.58-0.67 of the 4 x 3 grid)
PHOTO_MIN_INTERVALS = 3
TIMING_MIN_FRAMES = 300  # 10 s at 30 fps
AGREE_ALPHA = 0.05       # per-group false-disagreement rate of two windows of one camera (Bonferroni over its fields)
SPREAD_K = 2.0           # spread = max(sqrt(3) * SPREAD_K * std, floor): U(-s, s) has the std of k * std
GROUPS = ("geom", "photo", "timing", "drop")


def observer_fields(est):
    """Flatten an ``observer_replay.Observer.estimate`` dict into groups of ``field -> (value, std_gate, std)``.

    ``std`` is the reported uncertainty against the truth (it sizes the DR spread). ``std_gate`` is the part that
    differs between two windows of the same camera: for the geometric group the fit's own ``std_fit``, without the
    detector-bias floor that both windows share; elsewhere ``std``. A group is updated as a whole (K
    with its distortion, shot with read noise, latency with the jitter it was corrected by). A group is left out when
    its own gate fails (too few views / intervals / frames, too little coverage or level span), when any value or std
    is not finite or a std is not positive, or when a relative-spread field (all but ``cx``, ``cy``, distortion and
    ``drop_prob``) is not positive (theta 0 and a non-positive latency are not identifiable).

    Args:
        est: ``dict(geom=..., photo=..., timing=...)``.

    Returns:
        dict ``group -> {field: (value, std_gate, std)}`` with groups ``geom`` (``GEOM_FIELDS``), ``photo``
        (``sigma_shot``, ``sigma_read``), ``timing`` (``jitter_theta`` and, when estimated, ``latency_s``) and
        ``drop`` (``drop_prob``).
    """
    out = {}
    g = est.get("geom")
    if g is not None and g["n_views"] >= GEOM_MIN_VIEWS and np.mean(np.asarray(g["coverage"]) > 0) >= GEOM_MIN_COVERAGE:
        K, d = np.asarray(g["K"]), np.asarray(g["dist"])
        vals = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d)
        fit = g.get("std_fit", g["std"])
        out["geom"] = {f: (float(v), float(fit[f]), float(g["std"][f])) for f, v in zip(GEOM_FIELDS, vals)}
    p = est.get("photo")
    if (p is not None and p.get("sigma_shot") is not None and not p["insufficient_levels"]
            and p["n_intervals"] >= PHOTO_MIN_INTERVALS):
        out["photo"] = {f: (p[f], p[f + "_std"], p[f + "_std"]) for f in ("sigma_shot", "sigma_read")}
    t = est.get("timing") or {}
    if t.get("n_frames", 0) >= TIMING_MIN_FRAMES and "jitter_theta" in t:
        tg = {"jitter_theta": (t["jitter_theta"], t["jitter_theta_std"], t["jitter_theta_std"])}
        if np.isfinite(t.get("latency_s", np.nan)):  # latency_s = delta - 2 theta + p: fed back with its theta
            tg["latency_s"] = (t["latency_s"], t["latency_std"], t["latency_std"])
        out["timing"] = tg
        out["drop"] = {"drop_prob": (t["camera_drop_rate"], t["camera_drop_std"], t["camera_drop_std"])}
    absolute = _ABSOLUTE | set(GEOM_FIELDS[4:])
    return {grp: fs for grp, fs in out.items()
            if all(np.isfinite(x) for v in fs.values() for x in v) and all(v[1] > 0 and v[2] > 0 for v in fs.values())
            and all(f in absolute or v[0] > 0 for f, v in fs.items())}


def agree_k(n_fields):
    """Per-field |a - b| / std bound for a group of ``n_fields``: Bonferroni at ``AGREE_ALPHA`` (1.96 for one field)."""
    return NormalDist().inv_cdf(1 - AGREE_ALPHA / (2 * n_fields))


def _spread(field, value, std):
    """``max(sqrt(3) k std, floor)`` in randomize()'s convention (absolute for ``_ABSOLUTE``, else relative)."""
    floor = _ESTIMATE_FLOOR["dist" if field in GEOM_FIELDS[4:] else field]
    s = np.sqrt(3) * SPREAD_K * std
    absolute = field in _ABSOLUTE or field in GEOM_FIELDS[4:]
    return max(s if absolute else s / abs(value), floor)


def estimate_to_config(est, est_prev, base, base_spread=None):
    """Feed observer estimates back into a config and its domain-randomization spread.

    A field group is updated only when both estimates pass their own gate (``observer_fields``) and, field by
    field, ``|a - b| <= agree_k(n) * hypot(std_gate_a, std_gate_b)`` (Bonferroni over the group's ``n`` fields,
    ``AGREE_ALPHA`` per group). ``est`` and ``est_prev`` must come from **disjoint** windows (the caller's job: e.g.
    two observers on consecutive stretches of the stream); the agreement check is what catches a re-mount, a refocus
    or a gain change between them, which no residual inside one window shows. The newer ``est``
    supplies the value. The spread is ``max(sqrt(3) * SPREAD_K * std, floor)`` with the floors of ``randomize`` (plus
    0.5 % for fx / fy and 3 px for cx / cy): the estimated std shrinks as 1/sqrt(N) and alone would shrink the DR to
    a point. ``phase_s`` and ``fps`` are reported by the observer, not fed back (DR keeps the phase uniform).

    Args:
        est: Newer ``Observer.estimate`` dict.
        est_prev: Older estimate on a window disjoint from ``est``'s; ``None`` updates nothing.
        base: ``CameraConfig`` to update (the offline calibration or the previous feedback).
        base_spread: Spread that goes with ``base`` (the previous feedback's); default ``randomize``'s defaults.
            Groups that are not updated keep it, so a fed-back K stays randomized when a later round holds it back.

    Returns:
        ``(config, spread, report)``: the updated ``CameraConfig``; the spread dict for ``randomize`` (``dist`` a
        5-tuple when updated); ``report`` maps each group to ``"updated"``, ``"gate"`` (missing or below its own gate
        in either window, or a different field set), ``"gate: rational base dist ..."`` (geom held back: the
        observer fits 5 terms and would replace a rational wide-angle model) or ``"disagree: <fields>"``.
    """
    new = observer_fields(est)
    old = {} if est_prev is None else observer_fields(est_prev)
    updates, spread, report = {}, dict(_DEFAULT_SPREAD if base_spread is None else base_spread), {}
    for group in GROUPS:
        if group not in new or group not in old or new[group].keys() != old[group].keys():
            report[group] = "gate"
            continue
        k = agree_k(len(new[group]))
        bad = [f for f, (v, sg, _) in new[group].items()
               if not abs(v - old[group][f][0]) <= k * np.hypot(sg, old[group][f][1])]
        if bad:
            report[group] = "disagree: " + ", ".join(bad)
            continue
        report[group] = "updated"
        fields = new[group]
        if group == "geom" and is_rational(base.dist):  # the observer fits 5 terms: keep the wide-angle model
            report[group] = "gate: rational base dist (observer fits k1, k2, p1, p2, k3 only)"
            continue
        if group == "geom":
            updates.update({f: fields[f][0] for f in GEOM_FIELDS[:4]})
            updates["dist"] = tuple(fields[f][0] for f in GEOM_FIELDS[4:])
            spread.update({f: _spread(f, fields[f][0], fields[f][2]) for f in GEOM_FIELDS[:4]})
            spread["dist"] = tuple(_spread(f, fields[f][0], fields[f][2]) for f in GEOM_FIELDS[4:])
        else:
            updates.update({f: v for f, (v, _, _) in fields.items()})
            spread.update({f: _spread(f, v, sd) for f, (v, _, sd) in fields.items()})
    return dataclasses.replace(base, **updates), spread, report
