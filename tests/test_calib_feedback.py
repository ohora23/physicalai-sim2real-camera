"""O8 feedback: observer estimate -> CameraConfig + DR spread, two-window agreement gate."""

import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.calib import (
    _DEFAULT_SPREAD,
    agree_k,
    estimate_to_config,
    observer_fields,
    randomize,
)
from physicalai_sim2real_camera.observer_timing import TimingObserver

# reported std (with the detector-bias floor) and the fit's own std_fit: 20-seed means of the O6 estimates at the end
# of the 16.6 s sweep (rolling)
STD = dict(fx=1.14, fy=1.11, cx=1.92, cy=2.08, k1=0.0064, k2=0.031, p1=3.7e-4, p2=3.5e-4, k3=0.050)
STD_FIT = dict(fx=1.13, fy=1.10, cx=1.20, cy=1.45, k1=0.0063, k2=0.031, p1=3.7e-4, p2=3.4e-4, k3=0.050)


def est(cx=326.5, f=590.0, shot=0.02, latency=0.05, theta=0.005, n_views=20, cov_cells=8, n_frames=500):
    """A synthetic ``Observer.estimate`` dict: distortion (-0.3, 0.1, ...), 4 x 3 grid, 500 frames."""
    cov = np.zeros((3, 4), int)
    cov.flat[:cov_cells] = 2
    K = np.array([[f, 0, cx], [0, f, 234.5], [0, 0, 1]])
    return dict(geom=dict(K=K, dist=np.array([-0.3, 0.1, 0.001, -0.001, 0.0]), std=dict(STD), std_fit=dict(STD_FIT),
                          n_views=n_views, coverage=cov),
                photo=dict(sigma_shot=shot, sigma_read=0.005, sigma_shot_std=4e-4, sigma_read_std=3e-4, n_intervals=4,
                           insufficient_levels=False),
                timing=dict(n_frames=n_frames, latency_s=latency, latency_std=8e-4, jitter_theta=theta,
                            jitter_theta_std=2.5e-4, camera_drop_rate=0.05, camera_drop_std=0.01))


BASE = CameraConfig(640, 480, gamma=2.1774, black_level=0.02081, latency_s=0.03, sigma_shot=0.01, sigma_read=0.01)


def test_agreeing_windows_update_every_group():
    cfg, spread, report = estimate_to_config(est(), est(cx=326.9, shot=0.0204), BASE)
    assert set(report.values()) == {"updated"}
    assert (cfg.fx, cfg.cx, cfg.dist[0], cfg.sigma_shot, cfg.latency_s) == (590.0, 326.5, -0.3, 0.02, 0.05)
    assert (cfg.gamma, cfg.black_level) == (BASE.gamma, BASE.black_level)  # offline constants untouched
    # spread = max(sqrt(3) * 2 * std, floor) on the reported std (with the floor): fx sqrt(3) 2 1.14 / 590 = 0.67 %
    # over 0.5 %, cx 6.65 px over 3 px; k1 0.022, k2 0.107, k3 0.17 over 0.02, p1 / p2 0.0013 under it; drop 0.035
    s = np.sqrt(3) * 2
    assert spread["fx"] == pytest.approx(s * 1.14 / 590) and spread["cx"] == pytest.approx(s * 1.92)
    assert spread["dist"] == pytest.approx((s * 0.0064, s * 0.031, 0.02, 0.02, s * 0.050))
    assert spread["drop_prob"] == pytest.approx(s * 0.01)
    assert spread["latency_s"] == 0.2 and spread["gamma"] == _DEFAULT_SPREAD["gamma"]


def test_gate_uses_the_fit_std_not_the_shared_floor():
    """Gate on std_fit (combined 1.70 px for cx), not the reported std whose 1.5 px detector floor both windows share
    (2.72 px): a 5 px principal-point jump is 2.95 > agree_k(9) = 2.77 combined std_fit, caught (it was 1.84 on the
    reported std). A 3 px jump (1.77) is below what two 16.6 s windows resolve: O6 caught it in 0 of 10 pairs,
    6 px in 10 of 10. f +1 % (5.9 px on fx, 3.7) is caught."""
    assert agree_k(1) == pytest.approx(1.96, abs=1e-3) and agree_k(9) == pytest.approx(2.773, abs=1e-3)
    cfg, spread, report = estimate_to_config(est(cx=331.5), est(), BASE)
    assert report["geom"] == "disagree: cx" and report["photo"] == "updated"
    assert cfg.fx is None and cfg.dist == BASE.dist and "fx" not in spread and spread["dist"] == 0.02
    assert estimate_to_config(est(f=595.9), est(), BASE)[2]["geom"] == "disagree: fx, fy"
    assert estimate_to_config(est(cx=329.5), est(), BASE)[2]["geom"] == "updated"


def test_latency_is_fed_back_with_its_theta():
    """latency_s = delta - 2 theta + p: latency and theta are one group; a theta change holds both back."""
    _, _, report = estimate_to_config(est(latency=0.0535), est(), BASE)  # 3.1 combined std > agree_k(2) = 2.24
    assert report["timing"] == "disagree: latency_s"
    cfg, _, report = estimate_to_config(est(theta=0.0065), est(), BASE)
    assert report["timing"] == "disagree: jitter_theta" and cfg.latency_s == BASE.latency_s
    e = est()
    e["timing"]["latency_s"] = float("nan")  # one window without latency: different field sets
    assert estimate_to_config(e, est(), BASE)[2]["timing"] == "gate"


def test_without_a_second_window_nothing_is_fed_back():
    cfg, spread, report = estimate_to_config(est(), None, BASE)
    assert cfg == BASE and spread == _DEFAULT_SPREAD and set(report.values()) == {"gate"}


@pytest.mark.parametrize("kw, group", [(dict(n_views=10), "geom"), (dict(cov_cells=5), "geom"),
                                       (dict(n_frames=200), "timing"), (dict(theta=0.0), "timing"),
                                       (dict(latency=-0.001), "timing")])
def test_own_gates(kw, group):
    """Too few views, coverage below half the cells, under 10 s of frames, theta 0 or a non-positive latency (not
    identifiable): the group is not offered, in either window."""
    assert group not in observer_fields(est(**kw)) and group in observer_fields(est())
    assert estimate_to_config(est(**kw), est(), BASE)[2][group] == "gate"
    assert estimate_to_config(est(), est(**kw), BASE)[2][group] == "gate"


@pytest.mark.parametrize("where, key", [("timing", "latency_std"), ("photo", "sigma_read_std"),
                                        ("timing", "camera_drop_std")])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_non_finite_or_zero_std_is_not_fed_back(where, key, bad):
    """A NaN / inf / zero std drops the group (a NaN once passed the gate and gave a NaN spread)."""
    e = est()
    e[where][key] = bad
    group = {"latency_std": "timing", "sigma_read_std": "photo", "camera_drop_std": "drop"}[key]
    assert group not in observer_fields(e)
    _, spread, report = estimate_to_config(e, est(latency=0.09), BASE)
    assert report[group] == "gate" and all(np.all(np.isfinite(v)) for v in spread.values())


def test_photo_gate():
    e = est()
    e["photo"]["insufficient_levels"] = True
    assert "photo" not in observer_fields(e) and "drop" in observer_fields(e)
    assert set(observer_fields(dict(geom=None, photo=None, timing={}))) == set()


def test_held_back_group_keeps_the_previous_spread():
    """Round 2 with K held back (cx jump): the fed-back K of round 1 keeps its spread, so it stays randomized."""
    cfg1, spread1, _ = estimate_to_config(est(), est(), BASE)
    cfg2, spread2, report = estimate_to_config(est(cx=331.5), est(), cfg1, spread1)
    assert report["geom"].startswith("disagree") and cfg2.cx == cfg1.cx
    assert spread2["cx"] == spread1["cx"] and spread2["dist"] == spread1["dist"]


def test_randomize_takes_the_fed_back_spread():
    """Per-coefficient dist spread and absolute cx / cy (px), relative fx: the output stays inside them."""
    cfg, spread, _ = estimate_to_config(est(), est(), BASE)
    for seed in range(20):
        r = randomize(cfg, spread, np.random.default_rng(seed))
        assert abs(r.cx - cfg.cx) <= spread["cx"] and abs(r.fx / cfg.fx - 1) <= spread["fx"]
        assert all(abs(a - b) <= s for a, b, s in zip(r.dist, cfg.dist, spread["dist"]))


def test_timing_stds():
    """jitter_theta_std (Gamma(2) sample variance) and camera_drop_std (binomial) match the scatter of 200 seeds."""
    thetas, stds, drops, dstds = [], [], [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        obs = TimingObserver(readout_s=0.0, height_px=480)
        fn = np.flatnonzero(rng.random(600) > 0.05)  # 5 % drops
        ts = fn / 30.0
        for k, (f, t) in enumerate(zip(fn, ts)):
            obs.push(t + 0.05 + rng.gamma(2, 0.005), t, k, int(f))
        e = obs.estimate()
        thetas.append(e["jitter_theta"])
        stds.append(e["jitter_theta_std"])
        drops.append(e["camera_drop_rate"])
        dstds.append(e["camera_drop_std"])
    assert np.std(thetas) == pytest.approx(np.mean(stds), rel=0.15)
    assert np.std(drops) == pytest.approx(np.mean(dstds), rel=0.15)


def test_rational_base_keeps_geometry():
    """A rational (wide-angle) base dist is not replaced by the observer's 5-term fit; other groups still update."""
    import dataclasses

    base = dataclasses.replace(BASE, dist=(0.23, -0.07, 0.0, 0.0, 0.0, 0.56, -0.07, -0.02))
    cfg, _, report = estimate_to_config(est(cx=329.5), est(), base)
    assert report["geom"].startswith("gate: rational") and cfg.dist == base.dist and cfg.cx == base.cx
