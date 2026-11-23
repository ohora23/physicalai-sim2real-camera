"""Photometric observer: synthetic still intervals through the sensor model, no rendering."""

import numpy as np

from physicalai_sim2real_camera.observer_photo import (
    MAX_FRAMES,
    PhotoObserver,
    common_mode_scale,
    drifted,
    flicker_normalise,
)

GAMMA, BLACK, SHOT, READ = 2.2, 0.02, 0.02, 0.005
REFL = (0.03, 0.08, 0.18, 0.36, 0.6, 0.9)


def frame(rng, levels, h=24, w=24, flicker=1.0, boxes_only=False, gain=1.0, black=BLACK):
    """One uint8 frame, one 24x24 patch per level: I = refl, noise on I + black (as sensor.add_noise), gamma encode.

    ``gain`` amplifies the electrons: the same output level then carries var = gain * SHOT^2 * I + gain^2 * READ^2.
    """
    img = np.zeros((h, w * len(levels)), np.float64)
    boxes = []
    for i, lin in enumerate(levels):
        sig = lin * flicker + black
        noisy = sig + np.sqrt(gain * SHOT**2 * sig + gain**2 * READ**2) * rng.standard_normal((h, w))
        img[:, i * w:(i + 1) * w] = noisy
        boxes.append((i * w + 2, 2, (i + 1) * w - 2, h - 2))
    dn = np.clip(np.round(255.0 * np.clip(img, 0, 1) ** (1 / GAMMA)), 0, 255).astype(np.uint8)
    return dn, boxes


def run(seed, n_intervals=4, n_frames=30, flicker=False, drift=False, levels=REFL, n_dark=0, black=BLACK):
    """Stills through a PhotoObserver; ``n_dark`` black squares (reflectance 0) go in as ``dark_boxes``."""
    rng = np.random.default_rng(seed)
    obs = PhotoObserver(GAMMA, black, eps_speed=1e-3, tau_dn=1.0)
    levels = tuple(levels) + (0.0,) * n_dark
    n = len(levels) - n_dark
    for k in range(n_intervals):
        for _ in range(10):  # moving frames between stills: rejected
            f, b = frame(rng, levels, black=black)
            obs.push(f, speed=0.5, boxes=b[:n], exposure=8, gain=1, dark_boxes=b[n:])
        for j in range(n_frames):
            fl = 1.0 + (0.03 * np.sin(j) if flicker else 0.0)
            lv = tuple(x * (1 + 0.03 * j / n_frames) for x in levels) if drift else levels  # AE creeping 3 %
            f, b = frame(rng, lv, flicker=fl, black=black)
            obs.push(f, speed=0.0, boxes=b[:n], exposure=8, gain=1, dark_boxes=b[n:])
    obs.flush()
    return obs, obs.estimate()


def test_photo_observer_recovers_sigmas():
    """4 stills x 30 frames, 6 wedge levels: sigma_shot within 10 %, sigma_read within 20 %, over 50 seeds."""
    shots, reads = [], []
    for seed in range(50):
        obs, est = run(seed)
        assert est is not None and not est["insufficient_levels"] and est["n_intervals"] == 4
        assert obs.n_rejected_motion == 40 and obs.n_rejected_drift == 0
        shots.append(est["sigma_shot"])
        reads.append(est["sigma_read"])
        # O8 std: the fit's covariance with a 2 % floor covers the error (|z| <= 3 every seed)
        assert abs(est["sigma_shot"] - SHOT) <= 3 * est["sigma_shot_std"], seed
        assert abs(est["sigma_read"] - READ) <= 3 * est["sigma_read_std"], seed
    shots, reads = np.array(shots), np.array(reads)
    print(f"photo observer: sigma_shot {shots.mean():.4f} +- {shots.std():.4f} (true {SHOT}), sigma_read "
          f"{reads.mean():.4f} +- {reads.std():.4f} (true {READ}), span {est['span']:.1f}, r2 {est['r2']:.3f}")
    assert np.all(np.abs(shots - SHOT) / SHOT <= 0.10), f"shot worst {np.abs(shots - SHOT).max() / SHOT:.1%}"
    assert np.all(np.abs(reads - READ) / READ <= 0.20), f"read worst {np.abs(reads - READ).max() / READ:.1%}"


def test_black_squares_anchor_read_noise():
    """Four black squares (reflectance 0, signal = black level) as dark boxes with the wedge, 50 seeds: sigma_read
    within 3 % (per-sample linearisation; the delta method read it 3.5 % high here) and scatter below wedge-only.
    With 3 % flicker they stay unbiased (they are not flicker-scaled). Black squares plus one wedge step have no
    span and are reported as insufficient."""
    reads, wedge = [], []
    for seed in range(50):
        _, est = run(seed, n_dark=4)
        assert not est["insufficient_levels"] and est["n_levels"] == 40  # 4 intervals x (6 wedge + 4 black)
        reads.append(est["sigma_read"] / READ - 1)
        wedge.append(run(seed)[1]["sigma_read"] / READ - 1)
    assert np.max(np.abs(reads)) <= 0.03 and np.std(reads) < 0.7 * np.std(wedge), (np.max(np.abs(reads)), np.std(wedge))
    fl = [run(s, n_dark=4, flicker=True)[1] for s in range(10)]
    assert abs(np.mean([e["sigma_read"] for e in fl]) / READ - 1) <= 0.02
    assert abs(np.mean([e["sigma_shot"] for e in fl]) / SHOT - 1) <= 0.02
    assert run(0, levels=(0.6,), n_dark=4)[1]["insufficient_levels"]


def test_black_squares_at_a_low_black_level():
    """Black level 0 / 0.005: clipped black squares are left out of the fit and never trip the drift guard."""
    for black in (0.0, 0.005):
        obs, est = run(0, n_dark=4, black=black)
        assert obs.n_rejected_drift == 0 and est is not None and est["n_levels"] == 24, (black, est)  # wedge only
        assert abs(est["sigma_read"] / READ - 1) <= 0.20 and abs(est["sigma_shot"] / SHOT - 1) <= 0.10, (black, est)


def test_flicker_is_removed_and_drift_is_rejected():
    obs, est = run(0, flicker=True)
    assert est is not None and abs(est["sigma_shot"] - SHOT) / SHOT <= 0.10, "3 % flicker must not inflate shot noise"
    obs, est = run(0, drift=True)
    assert est is None and obs.n_rejected_drift == 4, "a 3 % AE creep inside every interval rejects all of them"


def test_insufficient_levels_and_metadata_change():
    obs, est = run(0, levels=(0.03, 0.04))  # span 1.3: slope not identifiable
    assert est["insufficient_levels"] and est["span"] < 8
    rng = np.random.default_rng(1)
    obs = PhotoObserver(GAMMA, BLACK, eps_speed=1e-3, tau_dn=1.0)
    for j in range(40):
        f, b = frame(rng, REFL)
        obs.push(f, speed=0.0, boxes=b, exposure=8, gain=1 if j < 20 else 2)  # gain steps mid-still: new interval
    obs.flush()
    assert obs.n_rejected_meta == 1 and len(obs.intervals) == 2 and obs.intervals[-1][1] == 2


def test_gain_step_is_not_pooled():
    """Two stills at gain 1, then two at gain 2 (same output levels): the fit uses only the gain-2 intervals, so
    sigma_shot is SHOT * sqrt(2) within 10 %; pooling both gains would land between the two."""
    rng = np.random.default_rng(3)
    obs = PhotoObserver(GAMMA, BLACK, eps_speed=1e-3, tau_dn=1.0)
    for g in (1, 1, 2, 2):
        f, b = frame(rng, REFL)
        obs.push(f, speed=0.5, boxes=b, exposure=8, gain=g)  # a moving frame separates the stills
        for _ in range(30):
            f, b = frame(rng, REFL, gain=g)
            obs.push(f, speed=0.0, boxes=b, exposure=8, gain=g)
    obs.flush()
    est = obs.estimate()
    assert est["gain"] == 2 and est["n_intervals"] == 2
    assert abs(est["sigma_shot"] - SHOT * np.sqrt(2)) / (SHOT * np.sqrt(2)) <= 0.10, est["sigma_shot"]


def test_long_still_is_bounded_and_box_jitter_does_not_split():
    """A 130-frame still is cut into MAX_FRAMES intervals (the 10-frame tail is too short) and only the patch crops
    are held; boxes jittering by 1 px keep one interval going."""
    rng = np.random.default_rng(4)
    obs = PhotoObserver(GAMMA, BLACK, eps_speed=1e-3, tau_dn=1.0)
    for j in range(2 * MAX_FRAMES + 10):
        f, b = frame(rng, REFL)
        b = [(x0 + j % 2, y0, x1 + j % 2, y1) for x0, y0, x1, y1 in b]  # 1-px corner-noise jitter
        obs.push(f, speed=0.0, boxes=b, exposure=8, gain=1)
        if j == 5:
            assert obs._cur[0][0].shape == (20, 20), "only the patch crops are kept"
    obs.flush()
    assert len(obs.intervals) == 2 and all(len(st[0]) == MAX_FRAMES for st, _, _ in obs.intervals)


def test_helpers():
    rng = np.random.default_rng(0)
    f, b = frame(rng, REFL)
    stack = np.stack([f, f]).astype(np.float64)
    stack[1] *= 1.05  # common-mode gain on frame 1
    norm = flicker_normalise(stack, b)
    assert np.allclose(norm[0][2:22, 2:22].mean(), norm[1][2:22, 2:22].mean(), rtol=1e-3)
    ramp = np.array([1.0, 2.0])[:, None, None] * np.ones((2, 3, 3))  # frame levels 1 and 1.5 over the two patches
    assert np.allclose(common_mode_scale([np.ones((2, 3, 3)), ramp]), [1.25 / 1.0, 1.25 / 1.5])
    assert not drifted(np.ones((30, 4, 4)))
    assert drifted(np.concatenate([np.ones((15, 4, 4)), 1.02 * np.ones((15, 4, 4))]))


if __name__ == "__main__":
    test_helpers()
    test_photo_observer_recovers_sigmas()
    test_flicker_is_removed_and_drift_is_rejected()
    test_insufficient_levels_and_metadata_change()
    print("observer_photo OK")
