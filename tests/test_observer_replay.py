"""O6 closed loop: the observer on the rendered prism sweep, plus unit tests of the replay pieces.

The 20-seed statistics come from ``scripts/observe_sim.py`` (minutes per seed pair); the acceptance test here runs
two seeds per shutter setting with the same code and the O2-O4 tolerances.
"""

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from physicalai_sim2real_camera.observer_geom import CoverageWindow, GeomObserver, nominal_K
from physicalai_sim2real_camera.observer_replay import FrameRecord, perturb_prism, sensor_clock
from physicalai_sim2real_camera.observer_timing import fit_delay, omega_triplet
from physicalai_sim2real_camera.target import PrismTarget

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "prism")
needs_prism = pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "prism.xml")),
                                 reason="run scripts/make_prism.py --out assets/prism first")


def test_frame_record_has_no_capture_time():
    """Leakage (B1): what the observer receives never contains the true capture time."""
    names = {f.name for f in dataclasses.fields(FrameRecord)}
    assert names == {"frame", "t_arrival", "ts_sensor", "seq", "frame_number", "exposure", "gain"}


def test_sensor_clock():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 2001)
    ts = np.array([sensor_clock(x, rng) for x in t])
    a, b = np.polyfit(t, ts, 1)
    assert abs(a - (1 + 50e-6)) < 2e-6 and abs(b - 1.234) < 1e-5
    assert 40e-6 < np.std(ts - (a * t + b)) < 60e-6


@needs_prism
def test_perturb_prism_moves_faces_not_boards():
    """Leakage (b): face poses move by ~0.5 mm / 0.3 deg; the face-frame corners (all the fit uses) do not."""
    import cv2
    tgt = PrismTarget.from_json(os.path.join(ASSETS, "prism.json"))
    p = perturb_prism(tgt, np.random.default_rng(1))
    for f0, f1 in zip(tgt.faces, p.faces):
        dt = np.linalg.norm(f1.T_prism_face[:3, 3] - f0.T_prism_face[:3, 3])
        ang = np.degrees(np.linalg.norm(cv2.Rodrigues(f1.T_prism_face[:3, :3] @ f0.T_prism_face[:3, :3].T)[0]))
        assert 0 < dt < 3e-3 and 0 < ang < 1.5
        assert (f1.cols, f1.rows, f1.square_m, f1.ids, f1.wedge) == (f0.cols, f0.rows, f0.square_m, f0.ids, f0.wedge)
        np.testing.assert_array_equal(p.board(f1.index).getChessboardCorners(),
                                      tgt.board(f0.index).getChessboardCorners())


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def test_omega_triplet_rejects_flips():
    """A steady 0.6 rad/s turn passes; a mirrored pose at either end or in the middle is refused."""
    T = 1 / 30
    R = [_rz(0.6 * k * T) for k in range(3)]
    assert omega_triplet(*R, 0, T, 2 * T) == pytest.approx(0.6, rel=1e-6)
    for flip in (np.diag([1.0, -1.0, -1.0]), _rx(np.radians(25))):  # 180 deg, and a 25 deg mirror (~13 rad/s)
        for k in range(3):
            bad = list(R)
            bad[k] = flip @ R[k]
            assert np.isnan(omega_triplet(*bad, 0, T, 2 * T)), k
    stop = [_rz(0.0), _rz(0.6 * T), _rz(0.6 * T)]  # a stop between the half-steps: 0.6 -> 0 rad/s is real motion
    assert omega_triplet(*stop, 0, T, 2 * T) == pytest.approx(0.3, rel=1e-6)


def test_fit_delay_affine_absorbs_a_speed_scale():
    """|omega_cam| read 10 % low with a 0.02 rad/s floor (solvePnP on a K still off): the affine match keeps the lag."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 15, 450))
    head = lambda q: np.abs(np.sin(1.3 * np.asarray(q))) + 0.5 * np.abs(np.sin(0.7 * np.asarray(q)))  # noqa: E731
    w = 0.9 * head(t - 0.056) + 0.02 + rng.normal(0, 0.01, len(t))
    d_aff, std, _ = fit_delay(w, t, head, rng=rng)
    d_raw, _, _ = fit_delay(w, t, head, rng=rng, affine=False)
    assert abs(d_aff - 0.056) < 1e-3 and std < 2e-3
    assert abs(d_raw - 0.056) > 3e-3  # without the affine match the scale moves the lag


def test_coverage_window_dedups_on_shared_corners_and_evicts_by_age():
    """A still re-seen with one edge corner more or less is the same pose; views older than the age limit go."""
    g = np.mgrid[0:4, 0:3].T.reshape(-1, 2).astype(np.float32)
    obj = np.c_[g * 0.06, np.zeros(len(g))].astype(np.float32)
    px = (g * 40 + 100).astype(np.float32)
    w = CoverageWindow((640, 480))
    w.add(obj, px, t=0.0)
    w.add(obj[1:], px[1:] + 0.5, t=0.1)  # one corner fewer, 0.5 px: same pose, replaces
    assert len(w) == 1 and w.views()[0][0].shape[0] == len(obj) - 1
    w.add(obj[:6], px[:6] + 5.0, t=0.2)  # shares half the corners but moved 5 px: a new pose
    w.add(obj, px + 300, t=5.0)
    assert len(w) == 3
    assert w.evict(1.0) == 2 and len(w) == 1


def test_gap_limit_follows_the_frame_period():
    """After 5 intervals the gap limit is 2.5 periods: at 60 fps a 0.05 s gap (3 periods) no longer vouches."""
    obs = GeomObserver(None, (640, 480), 45.0)
    assert obs.max_gap_s == 0.1
    for k in range(7):
        obs.push_detections([], k / 60)
    assert obs.max_gap_s == pytest.approx(2.5 / 60)
    ids = np.arange(9)
    px = np.stack(np.meshgrid(np.arange(3) * 30.0 + 200, np.arange(3) * 30.0 + 200), -1).reshape(-1, 2)
    obj = np.c_[px / 1000, np.zeros(9)].astype(np.float32)
    obs.push_detections([(0, ids, px, obj)], 7 / 60)
    obs.push_detections([(0, ids, px, obj)], 7 / 60 + 0.05)
    assert obs.n_rejected_gap == 1 and obs.n_rejected_motion == 0 and obs.n_accepted == 0


@pytest.fixture(scope="module")
def photometric_constants(tmp_path_factory):
    """Leakage (c): gamma / black level from scripts/calibrate.py on synthetic D1-D5 data, never the true config."""
    from observe_sim import calibrate_photometric, truth_config
    return calibrate_photometric(truth_config(0, 0.0), str(tmp_path_factory.mktemp("o6_calib")))


@needs_prism
@pytest.mark.parametrize("readout_s", [0.016, 0.0], ids=["rolling", "global"])
def test_closed_loop(photometric_constants, readout_s):
    """Two seeds of the 16.6 s sweep, the true camera hidden behind the leakage list: O2 fx <= 0.5 %, centre <= 3 px,
    field <= 4.5 px (the prism turned -60 deg to show the wedge face costs ~1 px of field over the O2 test's 3.5 px),
    |err| / sigma <= 3 on every field; O3 sigma_shot within 10 %, sigma_read within 20 %; O4 latency
    within 2 ms (with the 4 ms proprioception delay removed), theta within 20 %, drops within 1 %p of the realised rate;
    8 ms exposure and
    either a 16 ms rolling readout or a global shutter. Rolling also closes the feedback loop: the two runs
    fed back through ``estimate_to_config``, the fed-back camera simulated twice and re-estimated, every group a
    fixed point (re-estimates within 3 std of the fed-back value) and every fed-back field's truth inside the DR."""
    from observe_sim import closure, errors, run_seed
    gamma, black = photometric_constants
    assert abs(gamma - 2.2) > 1e-6 and abs(black - 0.02) > 1e-7  # fitted values, not the truth
    runs = []
    for seed in (0, 1):
        runs.append(run_seed(seed, readout_s, gamma, black))
        cfg, K_true, series, hull, drop_true = runs[-1]
        assert not np.allclose(nominal_K(45.0, 640, 480, 1.05), K_true)  # leakage (a): the start is off
        e = errors(cfg, K_true, series[-1][1], hull, drop_true)
        print(f"readout {readout_s * 1e3:.0f} ms seed {seed}: " +
              ", ".join(f"{k} {v:.4g}" for k, v in e.items() if isinstance(v, float)))
        assert e["fx_rel"] <= 0.005 and e["centre_px"] <= 3.0 and e["field_px"] <= 4.5, e
        assert max(e["z"].values()) <= 3.0, e["z"]
        assert abs(e["shot_rel"]) <= 0.10 and abs(e["read_rel"]) <= 0.20, e
        assert abs(e["latency_err_s"]) <= 0.002 and abs(e["theta_rel"]) <= 0.20 and abs(e["drop_err"]) <= 0.01, e
        assert len(series) >= 15  # the 1 Hz series exists for the convergence plot
    if readout_s:
        # one trial of a statistical criterion (10 trials per shutter: groups fixed in 90-100 %),
        # calibrated to seeds 0 / 1 / 1000 / 1001 (worst z 2.10): a failure after an unrelated change (RNG stream,
        # renderer, cv2) means re-running `observe_sim.py --closure 10`, not necessarily a bug
        c = closure(runs[0], runs[1], readout_s, gamma, black, (1000, 1001))
        print(f"closure: {c['report']}, worst z {max(c['z'].values()):.2f}")
        assert set(c["report"].values()) == {"updated"} and all(c["fixed"].values()), c
        assert all(c["covered"].values()) and len(c["covered"]) == 14, c["covered"]


def test_drop_accounting_ignores_delivery_order():
    """Frame 11 delivered after 12 (jitter) is not a drop; the real drop (14) and miss (16) are counted once each."""
    from physicalai_sim2real_camera.observer_timing import drop_accounting
    fn = [10, 12, 11, 13, 15, 17]
    seq = [1, 3, 2, 4, 5, 7]  # published: 10 11 12 13 15 16 17 = seq 1..7; 14 dropped, 16 (seq 6) missed
    _, _, drops, misses = drop_accounting(fn, seq)
    assert (drops, misses) == (1, 1)


def test_replay_estimates_once_after_a_pause():
    """A 5 s gap in arrivals gives one estimate, not five; the last estimate follows flush."""
    from physicalai_sim2real_camera.observer_replay import replay

    class Obs:
        def __init__(self):
            self.n_push = self.flushed = 0

        def push(self, rec, fn):
            self.n_push += 1

        def flush(self):
            self.flushed += 1

        def estimate(self, fn, rng=None):
            return self.n_push

    class Head:
        def omega_fn(self):
            return None

    t = list(np.arange(0, 2, 0.1)) + list(np.arange(7, 8, 0.1))
    recs = [FrameRecord(None, a, a, k, k) for k, a in enumerate(t)]
    obs = Obs()
    series = replay(recs, obs, Head(), every_s=1.0)
    times = [round(a, 1) for a, _ in series]
    assert times == [1.0, 7.0, 7.9] and obs.flushed == 1 and series[-1][1] == len(t)


def _same(a, b):
    """Equality of nested estimate dicts (arrays and nan included) up to a relative 1e-5.

    Multi-threaded OpenCV / BLAS sum in a different order run to run: the geometric std moved by up to 2e-6 (relative)
    between two replays of the same records. A replay that got one frame or head sample wrong moves the estimates by
    orders of magnitude more.
    """
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (np.ndarray, float, np.floating)):
        return np.allclose(a, b, rtol=1e-5, atol=1e-12, equal_nan=True)
    return a == b


@needs_prism
def test_recording_replays_same_estimates(tmp_path):
    """O7: a 6 s run recorded while it is replayed live, then read back, gives the same estimate series (``_same``)."""
    from observe_sim import ASSETS as A
    from observe_sim import FOVY, PRISM_YAW_DEG, H, W, truth_config

    from physicalai_sim2real_camera import SimCamera
    from physicalai_sim2real_camera.observer_replay import Observer, load_recording, record, replay, sim_records
    from physicalai_sim2real_camera.scenes import HeadPoseRing, prism_scene
    cfg = truth_config(0, 0.016)
    model, data, sweep = prism_scene(os.path.join(A, "prism.xml"), width=W, height=H, prism_yaw_deg=PRISM_YAW_DEG)
    cam = SimCamera(model, "cam", cfg)
    target = PrismTarget.from_json(os.path.join(A, "prism.json"))
    ring = HeadPoseRing()
    try:
        recs = record(sim_records(model, data, sweep, cam, cfg, 6.0, ring=ring, seed=0), ring, str(tmp_path / "rec"))
        obs = Observer(target, (W, H), FOVY, 2.2, 0.02, exposure_s=cfg.exposure_s, readout_s=cfg.readout_s)
        live = replay(recs, obs, ring, every_s=2.0, rng=np.random.default_rng(0))
    finally:
        cam.close()
        ring.close()
    recs, head = load_recording(str(tmp_path / "rec"))
    obs = Observer(target, (W, H), FOVY, 2.2, 0.02, exposure_s=cfg.exposure_s, readout_s=cfg.readout_s)
    again = replay(recs, obs, head, every_s=2.0, rng=np.random.default_rng(0))
    assert len(live) == len(again) >= 3
    assert "latency_s" in live[-1][1]["timing"] and live[-1][1]["health"]["n_frames"] > 150
    assert _same(live, again)
    # the CLI on the same recording: same final estimate (config carries gamma / black / readout only)
    import json

    import observe
    import observe_sim
    dataclasses.replace(cfg, gamma=2.2, black_level=0.02).save(str(tmp_path / "cfg.json"))
    observe.main([str(tmp_path / "rec"), "--config", str(tmp_path / "cfg.json"), "--every", "2",
                  "--out", str(tmp_path / "est.json")])
    with open(tmp_path / "est.json") as f:
        est = json.load(f)
    assert len(est) == len(again)
    for e, (_, g) in zip(est, again):  # every estimate, every field: floats survive JSON exactly (NaN as NaN)
        assert _same(e["estimate"], json.loads(json.dumps(observe_sim._jsonable(g))))


def test_head_log_matches_the_ring_after_a_wrap():
    """HeadLog shows the samples HeadPoseRing.read shows: all before a wrap, the last n - 1 after it."""
    from physicalai_sim2real_camera.observer_replay import HeadLog
    from physicalai_sim2real_camera.scenes import HeadPoseRing
    ring = HeadPoseRing(n=8)
    rows = []
    try:
        for k in range(13):
            ring.write(0.1 * k, [0, 0, k], [1, 0, 0, 0], [0, 0, 0.5 * k])
            rows.append(ring.read()[-1].copy())
            log = HeadLog(np.array(rows), ring.n)
            q = np.linspace(-0.1, 1.3, 29)
            np.testing.assert_array_equal(log.omega_fn()(q), ring.omega_fn()(q))
        assert ring.count == 13
    finally:
        ring.close()


def test_unwrap_counter():
    """A 16-bit frame counter wrapping with one reordered frame unwraps to a monotonic count; drops survive."""
    from physicalai_sim2real_camera.observer_timing import drop_accounting, unwrap_counter
    true = np.array([65530, 65531, 65533, 65532, 65534, 65535, 65536, 65538, 65539])  # 65537 dropped, 2 reordered
    raw = true % 65536
    np.testing.assert_array_equal(unwrap_counter(raw, 16) - true, np.zeros(len(true), int) + (raw[0] - true[0]))
    assert drop_accounting(unwrap_counter(raw, 16), np.arange(len(true)))[2] == 1
    assert drop_accounting(raw, np.arange(len(true)))[2] != 1  # without the unwrap the span is wrong


@pytest.mark.parametrize("pre, per_frame", [(0, 3), (20, 3), (20, 9)])
def test_record_head_rows_match_the_live_ring(tmp_path, pre, per_frame):
    """Recording that starts on a ring already wrapped (``pre`` = 20 samples into a ring of 8) and a writer adding
    samples between frames: at every frame and after the last one, the replayed HeadLog shows what the ring showed.
    ``per_frame`` = 9 > 7 loses samples between two frames: warned, and the windows still match."""
    import warnings

    from physicalai_sim2real_camera.observer_replay import load_recording, record
    from physicalai_sim2real_camera.scenes import HeadPoseRing
    ring, k = HeadPoseRing(n=8), 0
    q = np.linspace(-1.0, 6.0, 71)

    def write(m):
        nonlocal k
        for _ in range(m):
            ring.write(0.1 * k, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0.3 * k + 0.01 * k * k])
            k += 1

    def frames():
        for j in range(6):
            write(per_frame)
            yield FrameRecord(np.full((4, 6, 3), j, np.uint8), 0.1 * k, 0.1 * k, j, j, np.float32(0.008), 1.0)

    try:
        write(pre)
        live = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in record(frames(), ring, str(tmp_path / "r")):
                live.append(ring.omega_fn()(q))
            live.append(ring.omega_fn()(q))  # after the stream: the final estimate's view
        assert bool(w) == (per_frame > 7)
        recs, head = load_recording(str(tmp_path / "r"))
        again = [head.omega_fn()(q) for _ in recs] + [head.omega_fn()(q)]
        for a, b in zip(live, again, strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        ring.close()


def test_interrupted_recording_keeps_its_head_rows(tmp_path):
    """A consumer that stops (exception) and closes the ring before the generator is finalised: no crash (the tail
    read is not in ``finally``), and head.npy / info.json hold the rows up to the last frame."""
    from physicalai_sim2real_camera.observer_replay import load_recording, record
    from physicalai_sim2real_camera.scenes import HeadPoseRing
    ring = HeadPoseRing(n=8)

    def frames():
        for j in range(5):
            ring.write(0.1 * j, [0, 0, 0], [1, 0, 0, 0], [0, 0, 1.0])
            yield FrameRecord(np.zeros((4, 6, 3), np.uint8), 0.1 * j, 0.1 * j, j, j)

    g = record(frames(), ring, str(tmp_path / "r"))
    for j, _ in enumerate(g):
        if j == 2:
            break
    ring.close()
    g.close()  # finalised after the ring is gone
    recs, head = load_recording(str(tmp_path / "r"))
    assert len(list(recs)) == 3 and len(head.rows) == 3
