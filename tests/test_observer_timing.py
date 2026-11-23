"""Timing observer: pure-numpy synthetic with the sensor clock model, no rendering.

Truth: fps 30, latency 50 ms, jitter Gamma(2, 5 ms), 5 % camera drops, 1 % observer misses, sensor clock
``(1 + 50 ppm) t + 1.234 s + N(0, 50 us)``, proprioception delay 4 ms, head trajectory a sum of sinusoids
(|omega| peaks ~1 rad/s). The observer never sees the true capture times, and its |omega_cam| is a central
difference of noisy synthetic face rotations (as from solvePnP), not the true angular speed.
"""

import numpy as np
import pytest

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.observer_timing import (
    TimingObserver,
    detrended_delay,
    drop_accounting,
    fit_delay,
    jitter_theta,
    omega_central,
)
from physicalai_sim2real_camera.timing import schedule

L, THETA, DROP, P, RHO, C0 = 0.050, 0.005, 0.05, 0.004, 50e-6, 1.234
JITTER_K_THETA = 2 * THETA


def head_angles(t):
    """Yaw and pitch (rad) of the scripted head sweep; |omega| of Rz(yaw) Ry(pitch) is hypot(yaw', pitch')."""
    t = np.asarray(t, float)
    return 0.8 * np.sin(2 * np.pi * 0.30 * t), 0.5 * np.sin(2 * np.pi * 0.45 * t + 1.0)


def rot(yaw, pitch):
    """Rz(yaw) @ Ry(pitch)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])


def small_rotation(rng, sigma):
    """Random rotation of ~``sigma`` rad (pose noise of a solvePnP estimate)."""
    import cv2
    return cv2.Rodrigues(rng.normal(0, sigma, 3))[0]


def head_omega_true(t):
    """|omega| of a scripted head sweep: yaw and pitch sinusoids at different frequencies (rad/s)."""
    t = np.asarray(t, float)
    wy = 0.8 * 2 * np.pi * 0.30 * np.cos(2 * np.pi * 0.30 * t)
    wp = 0.5 * 2 * np.pi * 0.45 * np.cos(2 * np.pi * 0.45 * t + 1.0)
    return np.hypot(wy, wp)


def synth(seed, seconds=15.0, readout_s=0.0, H=480, miss_rate=0.01, fixed_row=None):
    rng = np.random.default_rng(seed)
    cfg = CameraConfig(640, H, fps=30, latency_s=L, jitter_k=2, jitter_theta=THETA, drop_prob=DROP, readout_s=readout_s)
    t_c, lat, dropped = schedule(cfg, int(seconds * cfg.fps), rng)
    keep = ~dropped
    t_c, lat, fn = t_c[keep], lat[keep], np.flatnonzero(keep)
    arrival = t_c + lat
    ts_sensor = (1 + RHO) * t_c + C0 + rng.normal(0, 50e-6, len(t_c))
    # the tracked face drifts slowly across the image (as it does under a smooth head sweep)
    row = (0.5 + 0.3 * np.sin(0.7 * t_c) if fixed_row is None else np.full(len(t_c), fixed_row)) * H
    t_rows = t_c + (row / H - 0.5) * readout_s                       # rolling shutter: the face's rows read then
    R = [rot(*head_angles(t)) @ small_rotation(rng, 3e-4) for t in t_rows]  # face pose per frame, 0.3 mrad noise
    seq = np.arange(len(t_c))
    got = rng.random(len(t_c)) >= miss_rate                          # frames the observer actually receives
    # |omega_cam| as the observer computes it: central difference over the neighbouring RECEIVED frames, sensor stamps
    gi = np.flatnonzero(got)
    omega_cam = np.full(len(t_c), np.nan)
    for a, b, c in zip(gi[:-2], gi[1:-1], gi[2:]):
        omega_cam[b] = omega_central(R[a], R[c], ts_sensor[a], ts_sensor[c])
    span = slice(int(fn[got][0]), int(fn[got][-1]) + 1)
    return dict(arrival=arrival[got], ts=ts_sensor[got], seq=seq[got], fn=fn[got], omega=omega_cam[got],
                row=row[got], n_dropped=int(dropped[span].sum()), drop_rate=float(dropped.mean()),
                n_missed=int((~got)[1:-1].sum()) if got[0] and got[-1] else int((~got).sum()))


def head_omega_reported(t):
    """What the robot reports at wall time t: the true pose P seconds ago."""
    return head_omega_true(np.asarray(t, float) - P)


def run(seed, readout_s):
    d = synth(seed, readout_s=readout_s)
    obs = TimingObserver(readout_s=readout_s, height_px=480, proprio_delay_s=P)
    for a, ts, s, f, w, r in zip(d["arrival"], d["ts"], d["seq"], d["fn"], d["omega"], d["row"]):
        obs.push(a, ts, s, f, omega_cam=w, row_mean_px=r)
    return obs.estimate(head_omega_reported, rng=np.random.default_rng(seed)), d


def test_components():
    d = synth(0)
    res = detrended_delay(d["arrival"], d["ts"])
    assert abs(res.mean()) < 1e-9 and abs(jitter_theta(res) - THETA) / THETA < 0.2
    cam, miss, ncd, nom = drop_accounting(d["fn"], d["seq"])
    assert ncd == d["n_dropped"] and nom == d["n_missed"], (ncd, d["n_dropped"], nom, d["n_missed"])
    assert abs(cam - d["drop_rate"]) < 0.01 and miss < 0.03
    ok = np.isfinite(d["omega"])
    delta, std, _ = fit_delay(d["omega"][ok], d["arrival"][ok], head_omega_reported)
    assert abs(delta - (L + 2 * THETA - P)) < 0.002 and 0 < std < 0.003


def test_timing_observer_global_and_rolling():
    for readout_s in (0.0, 0.008, 0.016):
        errs = []
        for seed in range(5):
            est, d = run(seed, readout_s)
            assert abs(est["fps"] - 30) < 0.1
            assert abs(est["jitter_theta"] - THETA) / THETA < 0.2
            assert abs(est["camera_drop_rate"] - d["drop_rate"]) < 0.01 and est["n_camera_drops"] == d["n_dropped"]
            assert est["n_observer_misses"] == d["n_missed"]
            errs.append(est["latency_s"] - L)
            assert abs(est["latency_s"] - L) < 0.002, f"readout {readout_s}: latency err {errs[-1] * 1e3:.2f} ms"
        print(f"timing observer readout {readout_s * 1e3:.0f} ms: latency err {np.abs(errs).max() * 1e3:.2f} ms max "
              f"over 5 seeds, std {est['latency_std'] * 1e3:.2f} ms")


def test_omega_central():
    """Central difference of Rz rotations: 0.1 rad over 0.05 s is 2 rad/s; non-increasing stamps give nan."""
    assert omega_central(rot(0.0, 0.0), rot(0.1, 0.0), 1.0, 1.05) == pytest.approx(2.0)
    assert omega_central(rot(0.0, 0.2), rot(0.0, 0.2), 0.0, 0.1) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(omega_central(rot(0, 0), rot(0.1, 0), 1.0, 1.0))


def test_fit_delay_without_overlap_is_nan():
    """A head record that no lagged camera sample reaches gives nan, not the first grid point."""
    delta, std, cmin = fit_delay(np.ones(30), np.linspace(100, 101, 30), lambda t: np.full(np.shape(t), np.nan))
    assert np.isnan(delta) and np.isnan(std) and np.isnan(cmin)


def test_no_true_capture_time_input():
    """The API has no way to pass the true capture time: the sim's trailer value cannot leak into the estimate."""
    import inspect
    params = set(inspect.signature(TimingObserver.push).parameters)
    assert "capture_t" not in params and "t_capture" not in params and params == {
        "self", "t_arrival", "ts_sensor", "seq", "frame_number", "omega_cam", "row_mean_px", "omega_span"}


def test_window_average_at_stops():
    """Stop-and-go head (1 rad/s, then 0 for 1 s): central differences average |omega| over two frame periods. When
    detection keeps fewer frames on the moving side, matching them against the head's point value biases the lag;
    matching against the head's average over the same window does not."""
    rng = np.random.default_rng(0)
    head = lambda t: (np.mod(np.asarray(t, float), 2.0) < 1.0).astype(float)  # noqa: E731
    T, L = 1 / 30, 0.05
    c = np.arange(1.0, 15.0, T) + rng.uniform(0, T)
    u = np.linspace(-1, 1, 201)
    w = head(c[:, None] + T * u).mean(axis=1)                      # ideal central difference over c - T .. c + T
    keep = (w < 0.99) & (w > 0.01) | (rng.random(len(c)) < 0.3)    # window on a step: kept; plain motion / rest: 30 %
    keep &= ~((w > 0.5) & (rng.random(len(c)) < 0.7))              # the moving side of a step is lost more often
    a, wk = c[keep] + L, w[keep] + rng.normal(0, 0.01, keep.sum())
    d_point, _, _ = fit_delay(wk, a, head, n_boot=0)
    d_win, _, _ = fit_delay(wk, a, head, n_boot=0, half_window_s=np.full(len(a), T))
    assert abs(d_win - L) < 1e-3 < abs(d_point - L), (d_win, d_point)


def test_rolling_shutter_bias_without_correction():
    """A face held near the bottom of a rolling-shutter frame is read (0.85 - 0.5) * 16 ms = 5.6 ms after mid-frame:
    without the row-time shift the latency is biased by that much; with it the error is back under 2 ms."""
    d = synth(1, readout_s=0.016, fixed_row=0.85)
    ests = {}
    for label, readout in (("uncorrected", 0.0), ("corrected", 0.016)):
        obs = TimingObserver(readout_s=readout, height_px=480, proprio_delay_s=P)
        for a, ts, s, f, w, r in zip(d["arrival"], d["ts"], d["seq"], d["fn"], d["omega"], d["row"]):
            obs.push(a, ts, s, f, omega_cam=w, row_mean_px=r)
        ests[label] = obs.estimate(head_omega_reported)["latency_s"] - L
    assert abs(ests["uncorrected"]) > 0.003 and abs(ests["corrected"]) < 0.002, ests
    print(f"rolling shutter face at 0.85 H: uncorrected err {ests['uncorrected'] * 1e3:.2f} ms, "
          f"corrected {ests['corrected'] * 1e3:.2f} ms")



def test_fit_delay_rejects_anti_correlated_lags():
    """Periodic head speed (period 0.4 s, lag 210 ms): a negative-slope match 200 ms away must not win (review H1)."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 15, 450))
        head = lambda q: 1 + np.sin(2 * np.pi * np.asarray(q) / 0.4)  # noqa: E731
        w = head(t - 0.21) + rng.normal(0, 0.3, len(t))
        assert abs(fit_delay(w, t, head, n_boot=0)[0] - 0.21) < 3e-3, seed


def test_stop_and_go_with_jitter():
    """Stop-and-go head (|omega| steps to 0 for 1 s every 2 s), Gamma(2, 5 ms) jitter, central differences with their
    spans: the observer's frame-time axis (arrival vs sensor line) keeps the latency within 2 ms, while the same fit on
    the raw arrival times is pulled off by the jitter meeting the steps."""
    rng = np.random.default_rng(3)
    cfg = CameraConfig(640, 480, fps=30, latency_s=L, jitter_k=2, jitter_theta=THETA, drop_prob=DROP)
    t_c, lat, dropped = schedule(cfg, 450, rng)
    t_c, lat, fn = t_c[~dropped], lat[~dropped], np.flatnonzero(~dropped)
    true = lambda t: 0.8 * (np.mod(np.asarray(t, float), 2.0) < 1.0)  # noqa: E731
    reported = lambda t: true(np.asarray(t, float) - P)  # noqa: E731
    ts = (1 + RHO) * t_c + C0 + rng.normal(0, 50e-6, len(t_c))
    u = np.linspace(-1, 1, 101)
    obs = TimingObserver(proprio_delay_s=P)
    for i in range(len(t_c)):
        if 0 < i < len(t_c) - 1:
            w = true(t_c[i - 1] + (t_c[i + 1] - t_c[i - 1]) * (u + 1) / 2).mean() + rng.normal(0, 0.01)
            obs.push(t_c[i] + lat[i], ts[i], i, fn[i], w, row_mean_px=0.0, omega_span=(ts[i - 1], ts[i + 1]))
        else:
            obs.push(t_c[i] + lat[i], ts[i], i, fn[i])
    est = obs.estimate(reported, rng=rng)
    assert abs(est["latency_s"] - L) < 2e-3, est["latency_s"]
    mid = np.asarray(obs.omega_ts)
    arrival = np.interp(mid, ts, t_c + lat)  # the old axis: jittered arrival times
    d_raw, _, _ = fit_delay(obs.omega_cam, arrival, reported, n_boot=0, half_window_s=np.asarray(obs.omega_hw))
    assert abs(d_raw - JITTER_K_THETA - (L - P)) > 2e-3 > abs(est["delta"] - JITTER_K_THETA - (L - P))


def test_parabolic_min_and_drop_clamp():
    from physicalai_sim2real_camera.observer_timing import _parabolic_min
    g = np.arange(5.0)
    assert _parabolic_min(g, (g - 2.3) ** 2) == pytest.approx(2.3)
    assert _parabolic_min(g, np.array([np.nan, np.nan, 0.0, np.nan, 1.0])) == 2.0  # nan neighbour: grid point
    assert _parabolic_min(g, g) == 0.0                                               # edge: no refinement
    assert drop_accounting([10, 11, 11, 12], [1, 2, 3, 4])[2:] == (0, 1)             # duplicates never go negative


def test_span_ignores_row_and_window_limits_samples():
    obs = TimingObserver(readout_s=0.016, window_s=5.0)
    for i in range(400):
        obs.push(i / 30 + 0.05, i / 30, i, i, 0.5 + 0.1 * np.sin(i / 5), row_mean_px=480.0,
                 omega_span=(i / 30 - 1 / 30, i / 30 + 1 / 30))
    assert set(obs.row_shift) == {0.0}  # the span is row-corrected already
    seen = []
    obs.estimate(lambda q: seen.append(np.size(q)) or np.ones(np.shape(q)), rng=np.random.default_rng(0))
    assert max(seen) <= 2501 * 151 * 7  # only ~5 s of samples (151 at 30 fps) enter the cost


if __name__ == "__main__":
    test_components()
    test_timing_observer_global_and_rolling()
    test_omega_central()
    test_no_true_capture_time_input()
    test_rolling_shutter_bias_without_correction()
    print("observer_timing OK")
