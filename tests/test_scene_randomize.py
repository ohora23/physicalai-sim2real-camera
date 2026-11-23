"""Tests for scene_dr.randomize_scene — pure-dataclass perturbation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physicalai_sim2real_camera.scene_dr import Lamp, SceneConfig, Walker, randomize_scene

BASE = SceneConfig(
    light_gain=0.8,
    light_tint=(1.0, 0.9, 0.8),
    ambient=0.1,
    color_jitter=0.2,
    marks=("x",),
    marks_per_geom=6,
    mark_size=0.3,
    walkers=(Walker(phase=0.2, period_s=6.0),),
    lamps=(Lamp(rgb=(1.0, 0.8, 0.6), dir=(0.0, 1.0, -1.0)),),
)

FULL_SPREAD = dict(
    light_gain=0.5,
    light_tint=0.2,
    ambient=0.1,
    color_jitter=0.5,
    mark_size=0.5,
    marks_per_geom=0.5,
    walker_phase=1.0,
    walker_period=0.3,
    lamp_rgb=0.2,
    lamp_dir=0.3,
)


def test_ranges():
    """500 draws with default_rng(0) — all results valid and in bounds."""
    rng = np.random.default_rng(0)
    results = [randomize_scene(BASE, FULL_SPREAD, rng) for _ in range(500)]
    seeds = [r.seed for r in results]
    for r in results:
        assert 0.4 - 1e-12 <= r.light_gain <= 1.2 + 1e-12
        for i, t in enumerate((1.0, 0.9, 0.8)):
            assert t * (1 - 0.2) - 1e-12 <= r.light_tint[i] <= t * (1 + 0.2) + 1e-12
        assert 0 <= r.ambient <= 0.2
        assert 0 <= r.color_jitter < 1
        assert 0.15 - 1e-12 <= r.mark_size <= 0.45 + 1e-12
        assert 3 <= r.marks_per_geom <= 9
        for w in r.walkers:
            assert 0 <= w.phase < 1 and 4.2 - 1e-12 <= w.period_s <= 7.8 + 1e-12
        for l in r.lamps:
            norm = sum(c * c for c in l.dir) ** 0.5
            assert abs(norm - 1.0) < 1e-9
        for i, c in enumerate((1.0, 0.8, 0.6)):
            assert c * (1 - 0.2) - 1e-12 <= r.lamps[0].rgb[i] <= c * (1 + 0.2) + 1e-12
    assert len(set(seeds)) >= 490


def test_deterministic():
    """Same seed → identical; different seed → different."""
    r1 = randomize_scene(BASE, FULL_SPREAD, np.random.default_rng(5))
    r2 = randomize_scene(BASE, FULL_SPREAD, np.random.default_rng(5))
    assert r1 == r2
    r3 = randomize_scene(BASE, FULL_SPREAD, np.random.default_rng(6))
    assert dataclasses.replace(r1, seed=0) != dataclasses.replace(r3, seed=0)  # fields differ, not only the seed
    rev = dict(reversed(list(FULL_SPREAD.items())))
    assert randomize_scene(BASE, rev, np.random.default_rng(5)) == r1  # fixed key order, not dict order


def test_zero_spread_identity():
    """Every key at 0 → result equals dataclasses.replace(base, seed=result.seed)."""
    zero = {k: 0 for k in FULL_SPREAD}
    r = randomize_scene(BASE, zero, np.random.default_rng(42))
    expected = dataclasses.replace(BASE, seed=r.seed)
    assert r == expected


def test_only_given_keys():
    """spread={'light_gain': 0.5} → every field except light_gain and seed equals base."""
    sparse = {"light_gain": 0.5}
    r = randomize_scene(BASE, sparse, np.random.default_rng(0))
    expected = dataclasses.replace(BASE, seed=r.seed)
    # Fields that should be unchanged
    for field in ("light_tint", "ambient", "color_jitter", "mark_size", "marks_per_geom",
                  "walkers", "lamps", "marks", "color_geoms", "shadow_clip"):
        assert getattr(r, field) == getattr(expected, field)
    assert r.light_gain != BASE.light_gain


def test_unknown_key_and_not_mutated():
    """{'nope': 1} raises ValueError; base unchanged after full-spread call."""
    with pytest.raises(ValueError):
        randomize_scene(BASE, {"nope": 1}, np.random.default_rng(0))
    before_copy = dataclasses.replace(BASE)
    randomize_scene(BASE, FULL_SPREAD, np.random.default_rng(0))
    assert BASE == before_copy


def test_clipping():
    """Over-spread: light_gain >= 0, color_jitter <= 0.99."""
    cfg = SceneConfig(light_gain=0.1, color_jitter=0.9)
    spread = {"light_gain": 5, "color_jitter": 5}
    rng = np.random.default_rng(0)
    rs = [randomize_scene(cfg, spread, rng) for _ in range(200)]
    assert all(r.light_gain >= 0 and r.color_jitter <= 0.99 for r in rs)
    assert any(r.light_gain == 0 for r in rs) and any(r.color_jitter == 0.99 for r in rs)  # the clips are reached


def test_other_walker_and_lamp_fields_kept():
    """Only phase / period and rgb / dir change: path, size, colour, position, cone and shadow flag survive."""
    w = Walker(p0=(0.5, -1.0), p1=(1.5, 2.0), radius=0.2, height=1.8, z0=0.05, rgba=(0.1, 0.2, 0.3, 1.0))
    lamp = Lamp(pos=(1.0, 2.0, 3.0), cutoff=30.0, exponent=4.0, directional=True, castshadow=False)
    r = randomize_scene(SceneConfig(walkers=(w,), lamps=(lamp,)), FULL_SPREAD, np.random.default_rng(1))
    assert dataclasses.replace(r.walkers[0], phase=w.phase, period_s=w.period_s) == w
    assert dataclasses.replace(r.lamps[0], rgb=lamp.rgb, dir=lamp.dir) == lamp


def test_distractors_spread_and_order():
    """distractors spread [3,9] over 50 draws with >1 distinct value; light_gain spread unchanged for old keys."""
    cfg = SceneConfig(distractors=6)
    rng = np.random.default_rng(0)
    results = [randomize_scene(cfg, {"distractors": 0.5}, rng) for _ in range(50)]
    counts = [r.distractors for r in results]
    assert all(3 <= c <= 9 for c in counts)
    assert len(set(counts)) > 1
    base = SceneConfig(light_gain=0.8)
    rng_a = np.random.default_rng(42)
    r1 = randomize_scene(base, {"light_gain": 0.3}, rng_a)
    assert r1.light_gain == 0.9314989033068625
