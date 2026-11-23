"""Acceptance tests for physicalai_sim2real_camera.timing (T3)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

import numpy as np

from physicalai_sim2real_camera import CameraConfig, CameraStream, CameraStreamBatch, latest_ready, ring_depth, schedule


class Data:
    time = 0.0


class Cam:
    def __init__(self):
        self.calls = 0

    def render(self, data):
        self.calls += 1
        return self.calls


def cfg(**kw):
    base = dict(width=8, height=8, fps=30, latency_s=0.05)
    base.update(kw)
    return CameraConfig(**base)


def run(c, times):
    """Yield (t, frame, capture_t, n, stream, cam) per step."""
    cam, d = Cam(), Data()
    s = CameraStream(cam, c)
    for t in times:
        d.time = t
        f, ct, n = s.step(d)
        yield t, f, ct, n, s, cam


def test_exact_schedule():
    for k, (t, f, ct, n, s, cam) in enumerate(run(cfg(), (k / 500 for k in range(10000)))):
        if k < 25:
            assert n is None and ct is None and f is None, k
            continue
        exp = max(m for m in range(k // 16 + 2) if (50 * m + 2) // 3 + 25 <= k)
        assert n == exp, (k, n, exp)
        assert abs(ct - ((50 * n + 2) // 3) / 500) < 1e-12, (k, ct)


def test_multi_capture():
    cam, d = Cam(), Data()
    s = CameraStream(cam, cfg(latency_s=0.0))
    steps_any = doubles = 0
    prev_hi = -1
    for k in range(2000):
        d.time = k / 20
        before = cam.calls
        s.step(d)
        hi = (3 * k) // 2  # max n with n/30 <= k/20
        passed = hi - prev_hi
        prev_hi = hi
        assert cam.calls - before == (1 if passed else 0), k
        if passed:
            steps_any += 1
            assert s._next - 1 == hi, (k, s._next, hi)
            assert s._buf[-1][2] == hi
        if passed == 2:
            doubles += 1
    assert doubles > 0
    assert cam.calls == steps_any, (cam.calls, steps_any)


def test_drop():
    c = cfg(drop_prob=0.1, seed=7)
    N = 10000
    *_, (t, f, ct, n, s, cam) = run(c, (k / 30 for k in range(N)))  # one capture per step
    assert s._next == N
    rng = np.random.default_rng([7, 1])
    drops = int((rng.random(N) < 0.1).sum())  # same draw order as CameraStream (no jitter)
    rate = drops / N
    print(f"drop rate {rate:.4f}")
    assert 0.09 <= rate <= 0.11
    assert cam.calls == N - drops, (cam.calls, N - drops)


def test_jitter():
    lats, prev, maxbuf = [], -1.0, 0
    calls = 0
    for t, f, ct, n, s, cam in run(cfg(jitter_k=2, jitter_theta=0.005), (k / 500 for k in range(10000))):
        if cam.calls != calls:
            calls = cam.calls
            e = s._buf[-1]
            lats.append(e[1] - e[0])
        if ct is not None:
            assert ct >= prev, (t, ct, prev)
            prev = ct
        maxbuf = max(maxbuf, len(s._buf))
    m = float(np.mean(lats))
    print(f"jitter mean latency {m:.5f} (exp 0.06), max buf {maxbuf}")
    assert abs(m - 0.06) <= 0.05 * 0.06
    assert maxbuf <= 10


def test_pure():
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1)
    t_c, lat, dr = schedule(c, 300, np.random.default_rng(0))
    assert t_c.shape == lat.shape == dr.shape == (300,)
    assert np.allclose(np.diff(t_c), 1 / 30) and (lat >= 0.05).all() and dr.dtype == bool
    t_c, lat, dr = schedule(cfg(), 5, np.random.default_rng(0))
    assert (lat == 0.05).all() and not dr.any()
    cap = np.array([0.0, 0.1, 0.2])
    rdy = np.array([0.05, 0.3, 0.22])
    assert latest_ready(cap, rdy, 0.01) is None
    assert latest_ready(cap, rdy, 0.25) == 2  # newest ready, older 0.1 still pending
    assert latest_ready(cap, rdy, 0.12) == 0


def ages(c, N):
    a = np.array([t - ct for t, f, ct, n, s, cam in run(c, (k / 500 for k in range(N))) if ct is not None])
    return a


def test_age():
    a = ages(cfg(), 10000)
    exp = 0.05 + 1 / 60 - 1 / 1000  # capture snaps to next ctrl tick: -dt/2
    print(f"age (jitter 0): mean {a.mean():.5f} exp {exp:.5f} min {a.min():.5f} max {a.max():.5f}")
    assert abs(a.mean() - exp) <= 0.02 * exp
    assert a.min() >= 0.05 - 1e-9
    assert a.max() <= 0.05 + 1 / 30 + 1 / 500 + 1e-9
    a = ages(cfg(jitter_k=2, jitter_theta=0.005), 100000)
    exp = 0.05 + 2 * 0.005 + 1 / 60
    print(f"age (Gamma 2,0.005): mean {a.mean():.5f} exp {exp:.5f}")
    assert abs(a.mean() - exp) <= 0.05 * exp


def test_perf():
    cam, d = Cam(), Data()
    s = CameraStream(cam, cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1))
    N = 10000
    ts = [k / 500 for k in range(N)]
    t0 = time.perf_counter()
    for t in ts:
        d.time = t
        s.step(d)
    us = (time.perf_counter() - t0) / N * 1e6
    print(f"step avg {us:.2f} us")
    assert us <= 20


def ks(a, b):
    """Two-sample KS statistic (same as tests/test_axon_bridge.py)."""
    grid = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a)
                               - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


def batch_vs_single(c, n_cam=8, N=2500, seed=None):
    """500 Hz for N steps -> per camera (batch, single) dicts of ct/seq/fired/age."""
    d = Data()
    fired_b = [[] for _ in range(n_cam)]
    k_now = [0]

    def render(i, data, n):
        assert data is d
        fired_b[i].append((k_now[0], n))
        return (i, n)  # slot handle

    b = CameraStreamBatch(c, n_cam, seed=seed, render=render)
    cs = c if seed is None else replace(c, seed=seed)
    singles = [CameraStream(Cam(), cs, cam_id=i) for i in range(n_cam)]
    fired_s = [[] for _ in range(n_cam)]
    B = {"ct": [], "seq": []}
    S = {"ct": [], "seq": []}
    prev = np.full(n_cam, -1)
    for k in range(N):
        d.time = k / 500
        k_now[0] = k
        ready, ct, seq, h, new, fired = b.step([d] * n_cam)
        assert np.array_equal(new, seq != prev)
        prev = seq
        B["ct"].append(ct)
        B["seq"].append(seq)
        row_ct, row_seq = [], []
        for i, s in enumerate(singles):
            calls = s.cam.calls
            f, cti, ni = s.step(d)
            if s.cam.calls != calls:
                fired_s[i].append(k)
            assert fired[i] == (s.cam.calls != calls), (k, i)
            row_ct.append(np.nan if cti is None else cti)
            row_seq.append(-1 if ni is None else ni)
            assert ready[i] == (cti is not None)
            assert h[i] == (None if ni is None else (i, ni)), (k, i, h[i], ni)
        S["ct"].append(row_ct)
        S["seq"].append(row_seq)
    for D in (B, S):
        D["ct"], D["seq"] = np.array(D["ct"]), np.array(D["seq"])
    return B, S, fired_b, fired_s


def test_batch_exact():
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1, seed=3)
    B, S, fb, fs = batch_vs_single(c)
    assert np.array_equal(B["ct"], S["ct"], equal_nan=True)
    assert np.array_equal(B["seq"], S["seq"])
    assert [[k for k, n in f] for f in fb] == fs
    assert len({tuple(f) for f in fb}) == 8  # cameras really are independent
    print(f"(a) batch n=8 500Hz 5s: capture_t/seq/fired identical per camera; renders {[len(f) for f in fb]}")


def test_batch_hold():
    d = Data()
    slots = iter(range(100, 10000))
    b = CameraStreamBatch(cfg(), 2, render=lambda i, data, n: next(slots))
    held, changes = [None, None], 0
    for k in range(1000):
        d.time = k / 500
        ready, ct, seq, h, new, fired = b.step([d, d])
        for i in range(2):
            if h[i] != held[i]:
                assert h[i] is not None and (held[i] is None or h[i] > held[i])
                changes += 1
                held[i] = h[i]
            assert h[i] == held[i] and ready[i] == (h[i] is not None)
    assert changes >= 2 * 50, changes
    print(f"(b) slot handles held until next fire: {changes} changes over 2 cams")


def test_batch_age():
    B, S, *_ = batch_vs_single(cfg(), N=5000)
    t = np.arange(5000)[:, None] / 500
    ab, as_ = t - B["ct"], t - S["ct"]
    worst = 0.0
    for i in range(8):
        a, s = ab[:, i], as_[:, i]
        a, s = a[~np.isnan(a)], s[~np.isnan(s)]
        worst = max(worst, ks(a, s))
    pooled = ab[~np.isnan(ab)]
    exp = 0.05 + 1 / 60 - 1 / 1000
    print(f"(c) age KS(batch_i, single_i) max {worst}, pooled mean {pooled.mean():.5f} exp {exp:.5f}")
    assert worst == 0.0
    assert abs(pooled.mean() - exp) <= 0.02 * exp


def test_batch_drop():
    d = Data()
    calls = []
    b = CameraStreamBatch(cfg(drop_prob=0.3), 4, seed=11, render=lambda i, data, n: calls.append((i, n)) or n)
    for k in range(3000):
        d.time = k / 30  # one capture per step per camera
        b.step([d] * 4)
    ok = True
    for i in range(4):
        rng = np.random.default_rng([11 + i, 1])
        kept = np.flatnonzero(~(rng.random(3000) < 0.3))
        ok &= [n for j, n in calls if j == i] == kept.tolist()
    print(f"(d) drop 0.3: renders {len(calls)}/12000, only non-dropped captures rendered: {ok}")
    assert ok


def test_batch_seed_override():
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1, seed=3)
    B, S, fb, fs = batch_vs_single(c, n_cam=4, seed=42)
    assert np.array_equal(B["ct"], S["ct"], equal_nan=True) and np.array_equal(B["seq"], S["seq"])
    assert [[k for k, n in f] for f in fb] == fs
    B0, *_ = batch_vs_single(c, n_cam=4)
    assert not np.array_equal(B["seq"], B0["seq"])  # the override really changed the draws
    print("seed override: CameraStreamBatch(seed=42) == CameraStream(replace(cfg, seed=42), cam_id=i)")


# ---- unit tests for the pure pieces ----

def test_render_hook_substitution():
    class NoRender:
        def render(self, data):
            raise AssertionError("cam.render must not be called when a hook is given")

    d, got = Data(), []
    s = CameraStream(NoRender(), cfg(latency_s=0.0), render=lambda data, n: got.append((data, n)) or f"h{n}")
    for k in range(10):
        d.time = k / 30
        f, ct, n = s.step(d)
        assert f == f"h{k}" and n == k and abs(ct - k / 30) < 1e-12
    assert got == [(d, k) for k in range(10)]
    s = CameraStream(None, cfg(latency_s=0.0), render=lambda data, n: n)  # cam may be None with a hook
    d.time = 0.0
    assert s.step(d) == (0, 0.0, 0)


def test_batch_not_ready_sentinel():
    d = Data()
    b = CameraStreamBatch(cfg(), 3, render=lambda i, data, n: (i, n))
    ready, ct, seq, h, new, fired = b.step([d] * 3)  # t=0 < latency 0.05: nothing ready
    assert ready.dtype == bool and ready.shape == (3,) and not ready.any()
    assert ct.shape == (3,) and np.isnan(ct).all()
    assert seq.shape == (3,) and np.issubdtype(seq.dtype, np.integer) and (seq == -1).all()
    assert h == [None, None, None]
    d.time = 0.05
    assert fired.all() and not new.any()  # capture 0 rendered, not delivered yet
    ready, ct, seq, h, new, fired = b.step([d] * 3)
    assert ready.all() and (ct == 0.0).all() and (seq == 0).all() and h == [(0, 0), (1, 0), (2, 0)]
    assert new.all() and fired.all()  # capture 0 delivered; capture 1 (t=1/30 <= 0.05) rendered
    ready, ct, seq, h, new, fired = b.step([d] * 3)  # same time again: held, nothing new
    assert (seq == 0).all() and not new.any() and not fired.any()
    e = CameraStreamBatch(cfg(), 0, render=lambda i, data, n: n).step([])
    assert [a.dtype for a in (e[0], e[1], e[2], e[4], e[5])] == [bool, np.float64, np.int64, bool, bool]


def seqs(b, N=1000):
    d, out = Data(), []
    for k in range(N):
        d.time = k / 500
        out.append(b.step([d] * len(b.streams))[2])
    return np.array(out)


def test_batch_seed_derivation():
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.3)
    hook = lambda i, data, n: n
    a = seqs(CameraStreamBatch(replace(c, seed=5), 4, render=hook))
    b = seqs(CameraStreamBatch(replace(c, seed=0), 4, seed=5, render=hook))
    assert np.array_equal(a, b)  # seed= overrides cfg.seed
    wide = seqs(CameraStreamBatch(replace(c, seed=4), 5, render=hook))
    assert np.array_equal(a[:, :3], wide[:, 1:4])  # camera i draws from seed + i


def test_ring_depth():
    assert ring_depth(cfg()) == 4  # ceil(0.05*30)=2, +2
    assert ring_depth(cfg(latency_s=0.1)) == 5  # 0.1*30 = 3 exactly, float noise must not round up
    assert ring_depth(cfg(latency_s=0.0)) == 2
    # Gamma(2, 0.005): 0.01 + 6*sqrt(2)*0.005 = 0.052426; (0.05 + 0.052426)*30 = 3.07 -> 4, +2
    assert ring_depth(cfg(jitter_k=2, jitter_theta=0.005)) == 6
    # k < 1 clamps the shape to 1: (0.05 + (1 + 6)*0.01)*30 = 3.6 -> 4, +2
    assert ring_depth(cfg(jitter_k=0.5, jitter_theta=0.01)) == 6
    assert ring_depth(cfg(fps=60, jitter_k=1, jitter_theta=0.01)) == 10  # (0.05 + 0.01 + 0.06)*60 = 7.2 -> 8, +2


def ring_mismatch(c, K, n_cam=8, N=2500):
    """Hook renders into slot (i, r_i % K), r_i = camera i's render count; count held slots != seq.

    Keyed on the render count, not n: dropped captures leave gaps in n, so n % K would wrap early.
    """
    d, buf, r = Data(), {}, [0] * n_cam

    def render(i, data, n):
        slot = (i, r[i] % K)
        r[i] += 1
        buf[slot] = n  # "pixels" of capture n
        return slot

    b = CameraStreamBatch(c, n_cam, render=render)
    bad = total = 0
    for k in range(N):
        d.time = k / 500
        ready, ct, seq, h, new, fired = b.step([d] * n_cam)
        for i in np.flatnonzero(ready):
            total += 1
            bad += buf[h[i]] != seq[i]
    return bad, total


def test_ring_lifetime():
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1)
    K = ring_depth(c)
    bad, total = ring_mismatch(c, K)
    bad1, total1 = ring_mismatch(c, 1)
    print(f"K-ring: K=ring_depth={K} content==seq mismatches {bad}/{total}; K=1 (one slot per camera) {bad1}/{total1}")
    assert bad == 0
    # one slot per camera: captures rendered while an older frame is still in flight overwrite it,
    # so the delivered pixels are newer than capture_t (latency 0.05 s > 1/fps)
    assert bad1 > 0.9 * total1


def test_batch_len_mismatch():
    d = Data()
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.3)
    b = CameraStreamBatch(c, 3, render=lambda i, data, n: n)
    for bad in ([d] * 2, [d] * 4):
        try:
            b.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("length mismatch must raise")
    # nothing was stepped before the raise: the batch still matches a fresh one
    assert np.array_equal(seqs(b), seqs(CameraStreamBatch(c, 3, render=lambda i, data, n: n)))


def test_none_handle_is_ready():
    d = Data()
    b = CameraStreamBatch(cfg(latency_s=0.0), 2, render=lambda i, data, n: None)
    ready, ct, seq, h, new, fired = b.step([d, d])
    assert ready.all() and (seq == 0).all() and (ct == 0.0).all() and h == [None, None]


def test_validation():
    for make in (lambda: CameraStream(None, cfg()),
                 lambda: CameraStreamBatch(cfg(), 2),
                 lambda: CameraStreamBatch(cfg(), 2, cams=[Cam(), object()])):
        try:
            make()
        except ValueError:
            pass
        else:
            raise AssertionError("missing render must raise")
    CameraStreamBatch(cfg(), 2, cams=[Cam(), Cam()])


def test_batch_cams_push():
    class PushCam(Cam):
        def __init__(self):
            super().__init__()
            self.pushes = 0

        def push(self, data):
            self.pushes += 1

    cams, d = [PushCam() for _ in range(3)], Data()
    b = CameraStreamBatch(cfg(), 3, cams=cams)  # no hook: cams[i].render is used
    for k in range(100):
        d.time = k / 500
        ready, ct, seq, h, new, fired = b.step([d] * 3)
    assert [c.pushes for c in cams] == [100] * 3
    assert h == (seq + 1).tolist()  # no drops: capture n is the (n+1)-th cam.render call


def test_last_render():
    """last_render = (capture_t, ready_t, n) per rendered capture; step() delivers n at the first step >= ready_t."""
    c = cfg(latency_s=0.03, jitter_k=2, jitter_theta=0.005, drop_prob=0.3)
    s, d, ready, first = CameraStream(Cam(), c), Data(), {}, {}
    assert s.last_render is None
    for k in range(2000):
        d.time = k / 500
        before = s.last_render
        _, _, n = s.step(d)
        if s.last_render is not before:  # rendered this step
            t_c, t_r, m = s.last_render
            assert abs(t_c - d.time) < 1e-12 and t_r - t_c >= 0.03
            ready[m] = t_r
        if n is not None:
            first.setdefault(n, d.time)
    assert 0.5 * len(ready) < len(first) <= len(ready)  # ~30 % dropped before render, rest delivered
    for n, t in first.items():  # delivered within one control step after ready_t
        assert ready[n] - 1e-9 <= t < ready[n] + 1 / 500 + 1e-9


def test_stagger_phases():
    from physicalai_sim2real_camera import stagger_phases
    c = cfg(fps=30, phase_s=0.01)
    assert np.array_equal(stagger_phases(c, 1), [0.01])
    p = stagger_phases(c, 4)
    assert p.shape == (4,) and p.dtype == np.float64
    assert np.allclose(np.diff(p), 1 / 120, atol=1e-12)
    assert np.all(np.diff(p) > 0) and p[-1] < 0.01 + 1 / 30
    print(f"stagger n=4 fps=30: {p}")


def test_batch_phase_s():
    """Stream i == CameraStream(replace(cfg, phase_s=p[i]), cam_id=i) sample for sample; None unchanged; bad length."""
    from physicalai_sim2real_camera import stagger_phases
    c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.1, seed=3, fps=30)
    n = 8
    p = stagger_phases(c, n)
    d = Data()
    b = CameraStreamBatch(c, n, render=lambda i, data, k: (i, k), phase_s=p)
    singles = [CameraStream(Cam(), replace(c, phase_s=float(p[i])), cam_id=i) for i in range(n)]
    n_fire = 0
    for k in range(300):
        d.time = k * 0.002
        ready, ct, seq, h, new, fired = b.step([d] * n)
        for i, s in enumerate(singles):
            calls = s.cam.calls
            f, cti, ni = s.step(d)
            assert (s.cam.calls != calls) == bool(fired[i]), (k, i)
            assert (cti is None and np.isnan(ct[i])) or cti == ct[i], (k, i)
            assert (ni if ni is not None else -1) == seq[i], (k, i)
        assert fired.sum() <= 1  # spacing 1/(8*30) = 4.17 ms > 2 ms: cameras no longer fire together
        n_fire += int(fired.sum())
    assert n_fire > 0
    try:
        CameraStreamBatch(c, n, render=lambda i, data, n: None, phase_s=p[:-1])
    except ValueError as e:
        assert "phase_s" in str(e)
    else:
        raise AssertionError("length n-1 phase_s accepted")
    print(f"(b) batch phase_s: {n} staggered streams identical to singles over 300 steps, {n_fire} fires")


def test_stagger_max_fired():
    """Deterministic gate (no GPU): staggered cameras fire at most ceil(DT*N*fps) per 2 ms tick."""
    from physicalai_sim2real_camera import stagger_phases
    for N, want in ((8, 1), (16, 1), (17, 2)):
        c = cfg(jitter_k=2, jitter_theta=0.005, drop_prob=0.05, fps=30, exposure_s=0.008, readout_s=0.016,
                latency_s=0.05)
        b = CameraStreamBatch(c, N, render=lambda i, d, n: None, phase_s=stagger_phases(c, N))
        d = Data()
        mx = 0
        for k in range(1, 1501):  # 3 s at 500 Hz
            d.time = k * 0.002
            mx = max(mx, int(b.step([d] * N)[5].sum()))
        assert mx == want, (N, mx)
        print(f"stagger N={N}: max fired per tick {mx}")


if __name__ == "__main__":
    test_exact_schedule()
    test_multi_capture()
    test_drop()
    test_jitter()
    test_pure()
    test_age()
    test_perf()
    test_batch_exact()
    test_batch_hold()
    test_batch_age()
    test_batch_drop()
    test_batch_seed_override()
    test_render_hook_substitution()
    test_batch_not_ready_sentinel()
    test_batch_seed_derivation()
    test_batch_cams_push()
    test_ring_depth()
    test_ring_lifetime()
    test_batch_len_mismatch()
    test_none_handle_is_ready()
    test_validation()
    test_last_render()
    test_stagger_phases()
    test_batch_phase_s()
    test_stagger_max_fired()
    print("all timing tests passed")
