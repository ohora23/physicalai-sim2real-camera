"""scripts/analyse_uvc.py on synthetic captures: drop counting, gamma recovery, motion mask, empty-scene guard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyse_uvc as au  # noqa: E402


def stamps(n, period=0.032, jitter=0.0015, drop_every=25, seed=0):
    """Driver stamps of a camera at ``period`` dropping every ``drop_every``-th frame, plus arrival times."""
    rng = np.random.default_rng(seed)
    k = np.array([i for i in range(n) if i % drop_every != drop_every - 1])
    ts = k * period + rng.normal(0, 50e-6, len(k))
    return ts + 0.034 + rng.gamma(2, jitter / 2, len(k)), ts, n - len(k)


def test_drops_counted_gap_by_gap():
    """1900 frames at 31.25 fps with 1.5 ms jitter and every 25th dropped: the drop count is exact (a single
    median period over the whole run slipped frames)."""
    ta, ts, n_drop = stamps(1900)
    t = au.timing(ta, ts)
    assert t["drops"] == n_drop - (1900 % 25 == 0)  # a trailing drop leaves no gap
    assert t["period_s"] == pytest.approx(0.032, rel=1e-3) and t["arrival_minus_stamp_s"]["mean"] > 0.034
    with pytest.raises(SystemExit):
        au.timing(ta, np.zeros_like(ts))


def synth_stacks(gamma=1.45, n_frames=12, rng=None, move=False):
    """``(n_exp, n_frames, 64, 64, 1)`` uint8: tile albedos 0.2-0.9, linear signal ~ 1/exposure, encoded with gamma."""
    rng = rng or np.random.default_rng(1)
    exposures = np.array([335, 420, 500, 700, 1000, 1400, 2000], float)
    albedo = np.kron(rng.uniform(0.2, 0.9, (4, 4)), np.ones((16, 16)))[..., None]
    out = []
    for e in exposures:
        lin = albedo * (335 / e) * 0.9
        f = np.stack([lin + rng.normal(0, 0.004, lin.shape) for _ in range(n_frames)])
        if move:
            f[n_frames // 2:, :16, :16] *= 1.5  # a tile changes half-way: motion
        out.append(np.clip(np.round(255 * np.clip(f, 0, 1) ** (1 / gamma)), 0, 255).astype(np.uint8))
    return np.stack(out), exposures


def test_gamma_recovered_and_motion_masked():
    """Gamma 1.45 from tiles linear in 1/exposure; the tile that changes mid-stack is masked out."""
    stacks, exp = synth_stacks(move=True)
    T = au.tiles(stacks)
    ok = au.usable(T)
    assert not ok[:, 0].any() and ok[:, 1:].all()
    g, black, r2 = au.fit_gamma(T.mean(axis=(2, 3, 4, 5)), (1 / exp) / (1 / exp).max(), ok)
    assert g == pytest.approx(1.45, abs=0.05) and abs(black) < 0.01 and r2 > 0.999


def test_no_usable_tile_is_reported(tmp_path):
    """A saturated scene (no tile unclipped) gives identifiable=False, not a crash."""
    stacks, exp = synth_stacks()
    stacks = np.full_like(stacks, 255)
    ta, ts, _ = stamps(200)
    p = tmp_path / "cap.npz"
    np.savez(p, t_arrival=ta, ts_ms=ts * 1e3, exposures=exp, stacks=stacks, box=np.array([0, 0, 64, 64]),
             ctl_timing="{}", ctl_sweep=str([{"gain": 255, "exposure_absolute": int(e)} for e in exp]))
    rep = au.main([str(p), "--out", str(tmp_path / "rep.json")])
    assert rep["response"]["identifiable"] is False and (tmp_path / "rep.json").exists()
