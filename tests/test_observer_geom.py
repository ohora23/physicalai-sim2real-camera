"""Geometric observer: pure-logic tests and the render-based convergence / calibration test."""

import os

import numpy as np
import pytest

from physicalai_sim2real_camera.calib_geom import distortion_field_error
from physicalai_sim2real_camera.observer_geom import (
    DETECTOR_BIAS_PX,
    CoverageWindow,
    GeomObserver,
    detector_bias_floor,
    image_speed_px,
    motion_gate,
    nominal_K,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "prism")


def synthetic_views(K, d, n, rng, noise_px=0.0):
    """``n`` random poses of a 7x5 planar grid (40 mm) projected with ``K``, ``d``: ``(objs, pts)``."""
    import cv2
    g = np.mgrid[0:7, 0:5].T.reshape(-1, 2) * 0.04
    obj = np.c_[g - g.mean(0), np.zeros(len(g))].astype(np.float32)
    objs, pts = [], []
    for _ in range(n):
        r = rng.uniform(-0.5, 0.5, 3)
        t = np.r_[rng.uniform(-0.1, 0.1, 2), rng.uniform(0.5, 0.8)]
        q = cv2.projectPoints(obj.astype(np.float64), r, t, K, d)[0]
        pts.append((q + noise_px * rng.standard_normal(q.shape)).astype(np.float32))
        objs.append(obj)
    return objs, pts


class _ScriptedTarget:
    """Stand-in for ``PrismTarget``: ``detect`` returns the next scripted detection list."""

    def __init__(self, script):
        self.script = iter(script)

    def detect(self, img):
        return next(self.script)


def test_gate_needs_the_next_sighting_and_a_short_gap():
    """8 + 16 ms capture: the last frame of a still (the head moves 60 px/s right after it) is rejected because the
    gate also looks forward; frames 0.2 s apart never vouch for each other."""
    ids = np.arange(9)
    base = np.stack(np.meshgrid(np.arange(3) * 30.0 + 200, np.arange(3) * 30.0 + 200), -1).reshape(-1, 2)
    obj = np.zeros((9, 3), np.float32)
    det = lambda shift: [(0, ids, base + [shift, 0], obj)]  # noqa: E731
    shifts = [0, 0, 0, 0, 2, 4, 6]  # four still frames at 30 fps, then 60 px/s
    obs = GeomObserver(_ScriptedTarget([det(x) for x in shifts]), (640, 480), 45.0, exposure_s=0.008, readout_s=0.016)
    got = [obs.push(None, k / 30) for k in range(len(shifts))]
    assert got == [0, 0, 1, 1, 0, 0, 0]  # frames 1, 2 in; frame 3 (still's end) out
    assert len(obs.window) == 1  # frames 1 and 2 are the same pose: one view
    obs = GeomObserver(_ScriptedTarget([det(0)] * 4), (640, 480), 45.0, exposure_s=0.008, readout_s=0.016)
    assert sum(obs.push(None, 0.2 * k) for k in range(4)) == 0  # gap 0.2 s > MAX_GAP_S
    obs = GeomObserver(_ScriptedTarget([det(0)] * 4), (640, 480), 45.0, exposure_s=0.008, readout_s=0.016)
    assert sum(obs.push(None, 0.2 * k, ts_sensor=k / 30) for k in range(4)) == 2  # sensor stamps are the time base


def test_reset_and_repeated_estimates():
    """Re-estimating an unchanged window returns the same K every time (no state drifts); reset() empties it."""
    rng = np.random.default_rng(5)
    K1 = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    obs = GeomObserver(_ScriptedTarget([]), (640, 480), 45.0)
    for o, p in zip(*synthetic_views(K1, np.array([-0.3, 0.1, 0.0, 0.0, 0.0]), 30, rng, 0.05)):
        obs.window.add(o, p.reshape(-1, 2))
    Ks = [obs.estimate()["K"] for _ in range(6)]
    # 1e-6 px, not bit-equal: multi-threaded BLAS sums in a different order per call (3.3e-8 at 16 threads)
    assert all(np.allclose(K, Ks[0], rtol=0, atol=1e-6) for K in Ks) and abs(Ks[0][0, 2] - 319.5) < 1.0
    obs.reset()
    assert len(obs.window) == 0 and obs.estimate() is None and obs.n_resets == 1

def test_nominal_K():
    K = nominal_K(45.0, 640, 480)
    assert abs(K[0, 0] - 0.5 * 480 / np.tan(np.radians(22.5))) < 1e-9 and K[0, 2] == 319.5 and K[1, 2] == 239.5
    assert nominal_K(45.0, 640, 480, scale=1.05)[0, 0] == K[0, 0] * 1.05


def test_motion_gate():
    """0.2 px during capture: 8 + 16 ms window admits <= 8.3 px/s; a global shutter 8 ms admits <= 25 px/s."""
    assert motion_gate(8.0, 0.008, 0.016) and not motion_gate(9.0, 0.008, 0.016)
    assert motion_gate(24.0, 0.008, 0.0) and not motion_gate(26.0, 0.008, 0.0)
    assert not motion_gate(float("inf"), 0.0, 0.0)


def test_image_speed_px():
    ids0, px0 = np.array([1, 2, 3]), np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    ids1, px1 = np.array([2, 3, 4]), np.array([[10.0, 3.0], [20.0, 3.0], [30.0, 3.0]])
    assert abs(image_speed_px((ids0, px0), (ids1, px1), 0.5) - 6.0) < 1e-9  # shared ids 2, 3 moved 3 px in 0.5 s
    assert image_speed_px((ids0, px0), (np.array([9]), px1[:1]), 0.5) == float("inf")
    assert image_speed_px((ids0, px0), (ids1, px1), 0.0) == float("inf")


def test_coverage_window():
    """Per-cell cap with oldest-out replacement; coverage map; views() in insertion order."""
    w = CoverageWindow((640, 480), grid=(4, 3), cap=2)
    obj = np.zeros((4, 3), np.float32)
    corner = lambda x, y: np.array([[x, y], [x + 10, y], [x, y + 10], [x + 10, y + 10]], np.float32)  # noqa: E731
    assert w.add(obj, corner(10, 10)) == (0, 0)
    assert w.add(obj, corner(20, 20)) == (0, 0)
    assert w.add(obj, corner(30, 30)) == (0, 0) and len(w) == 2  # cap 2: the (10, 10) view was replaced
    objs, pts = w.views()
    assert pts[0][0, 0, 0] == 20 and pts[1][0, 0, 0] == 30
    assert w.add(obj, corner(600, 440)) == (3, 2)
    cov = w.coverage()
    assert cov.shape == (3, 4) and cov[0, 0] == 2 and cov[2, 3] == 1 and cov.sum() == 3
    w.clear()
    assert len(w) == 0 and w.coverage().sum() == 0
    obj2 = np.arange(12, dtype=np.float32).reshape(4, 3)
    w.add(obj2, corner(100, 100))
    w.add(obj2, corner(101, 100))  # same corners 1 px away: the same pose, replaces
    w.add(obj, corner(101, 100))   # other corners (another face): kept
    w.add(obj2, corner(300, 100))  # 200 px away: a new pose
    assert len(w) == 3 and w.views()[1][0][0, 0, 0] == 101


def test_detector_bias_floor():
    """Exact synthetic views: cx/cy floors equal the bias, k1 moves when the centre is forced off, fx/fy stay small."""
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    objs, pts = synthetic_views(K, np.array([-0.2, 0.05, 0.0, 0.0, 0.0]), 12, np.random.default_rng(0))
    floor = detector_bias_floor(objs, pts, (640, 480), K)
    assert floor["cx"] == pytest.approx(DETECTOR_BIAS_PX, abs=1e-6)
    assert floor["cy"] == pytest.approx(DETECTOR_BIAS_PX, abs=1e-6)
    assert floor["k1"] > 0 and floor["fx"] < 0.02 * K[0, 0]


def run_observer(seed, n_target=60, exposure_s=0.008, readout_s=0.016):
    """Drive the stop-and-go sweep through SimCamera (distortion + noise) and feed the observer at ~30 fps."""
    import cv2

    from physicalai_sim2real_camera import CameraConfig, SimCamera
    from physicalai_sim2real_camera.scenes import STILLS, drive, prism_scene
    from physicalai_sim2real_camera.target import PrismTarget
    W, H = 640, 480
    f_nom = nominal_K(45.0, W, H)[0, 0]  # the true K is off the observer's nominal start in every field (no leak)
    cfg = CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), sigma_shot=0.02, sigma_read=0.005, seed=seed,
                       fx=0.98 * f_nom, fy=0.98 * f_nom, cx=(W - 1) / 2 + 7, cy=(H - 1) / 2 - 5)
    target = PrismTarget.from_json(os.path.join(ASSETS, "prism.json"))
    model, data, sweep = prism_scene(os.path.join(ASSETS, "prism.xml"), width=W, height=H)
    cam = SimCamera(model, "cam", cfg)
    cv2.setRNGSeed(seed)  # SimCamera seeds cv2's global noise RNG once per process: without this the noise depends
    #                       on which tests ran before (make cov vs standalone gave k1 0.85 vs 0.95)
    obs = GeomObserver(target, (W, H), fovy_deg=45.0, exposure_s=exposure_s, readout_s=readout_s)
    snaps = {}
    hull = []
    try:
        k = 0
        while data.time < 16.0 and obs.n_accepted < n_target:
            drive(model, data, sweep)
            k += 1
            if k % 16:  # ~31 fps sampling of the 500 Hz sim
                continue
            img = cam.render(data)
            before = obs.n_accepted
            obs.push(img, data.time)
            if obs.n_accepted > before:
                for _, _, px, _ in target.detect(img):
                    hull.append(px)
            if "two_stills" not in snaps and data.time > STILLS[1][1]:
                snaps["two_stills"] = obs.estimate()
    finally:
        cam.close()
    return cam.K_out, np.array(cfg.dist), obs, snaps, np.concatenate(hull) if hull else np.zeros((0, 2))


def test_geom_observer_converges_and_std_is_calibrated():
    """End of the 16 s stop-and-go sweep (12 stills, ~23 distinct views in the window): fx/fy <= 0.5 %; the reported
    sigma is calibrated (|err| / sigma <= 3 for every field in >= 95 % of 20 seeds) and informative (sigma_cx, sigma_cy
    <= 2.5 px including the detector-bias floor, i.e. of the size of the 3 px centre tolerance, and the fit's own
    sigma <= 0.5 x its value after the first two stills); the distortion field inside the corner hull is within
    3.5 px and the centre within 3 px; moving frames are rejected by the gate (count reported). The absolute centre
    tolerance is set by the
    measured information limit of this geometry: an analytic twin of the same correspondences with 0.1 px noise
    gives cx/cy errors of 0.4-0.8 px and a field of 0.6-0.7 px, and the rendered pipeline adds ~1 px of systematic
    corner bias. Calibration holds because the window keeps distinct poses only: with
    near-duplicate still frames counted as independent views, sigma was several times too small."""
    if not os.path.exists(os.path.join(ASSETS, "prism.xml")):
        pytest.skip("run scripts/make_prism.py --out assets/prism first")
    W, H = 640, 480
    ratios, z, sig = [], [], []
    worst = dict(fx=0, cx=0, field=0)
    fails = []  # every seed runs before judging, so one run gives the whole statistics
    for seed in range(20):
        K_true, d_true, obs, snaps, hull = run_observer(seed, n_target=10**9)  # whole sweep
        est = obs.estimate()
        assert est is not None, (f"seed {seed}: only {obs.n_accepted} accepted frames "
                                 f"(rejected {obs.n_rejected_motion})")
        K, d, std = est["K"], est["dist"], est["std"]
        fx_err = max(abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0], abs(K[1, 1] - K_true[1, 1]) / K_true[1, 1])
        c_err = max(abs(K[0, 2] - K_true[0, 2]), abs(K[1, 2] - K_true[1, 2]))
        field = distortion_field_error(K_true, d_true, K, d, hull, W, H)
        worst = dict(fx=max(worst["fx"], fx_err), cx=max(worst["cx"], c_err), field=max(worst["field"], field))
        if not (fx_err <= 0.005 and field <= 3.5 and c_err <= 3.0):
            fails.append(f"seed {seed}: fx {fx_err:.3%} c {c_err:.2f} field {field:.2f}")
        if max(std["cx"], std["cy"]) > 2.5:
            fails.append(f"seed {seed}: centre sigma {std['cx']:.2f} / {std['cy']:.2f} px")
        assert snaps.get("two_stills") is not None, f"seed {seed}: fewer than 3 distinct views after two stills"
        ratios.append(est["std_fit"]["cy"] / snaps["two_stills"]["std_fit"]["cy"])
        sig.append(max(std["cx"], std["cy"]))
        truth = dict(fx=K_true[0, 0], fy=K_true[1, 1], cx=K_true[0, 2], cy=K_true[1, 2], k1=d_true[0], k2=d_true[1],
                     p1=d_true[2], p2=d_true[3], k3=d_true[4])
        estv = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4])
        z.append({f: abs(estv[f] - truth[f]) / max(std[f], 1e-12) for f in truth})
        if seed == 0:
            print(f"seed 0: accepted {obs.n_accepted} (window {est['n_views']} views), rejected by the motion gate "
                  f"{obs.n_rejected_motion}, coverage\n{est['coverage']}, cells to cover {est['cells_to_cover']}")
    frac_ok = {f: np.mean([zz[f] <= 3 for zz in z]) for f in z[0]}
    z_max = {f: round(float(np.max([zz[f] for zz in z])), 2) for f in z[0]}
    print(f"20 seeds: worst fx err {worst['fx']:.3%}, worst centre {worst['cx']:.2f} px, worst field "
          f"{worst['field']:.2f} px; centre sigma max {max(sig):.2f} px; sigma(end)/sigma(two stills) median "
          f"{np.median(ratios):.2f}; |err|/sigma <= 3 fraction per field {frac_ok}; max |err|/sigma {z_max}")
    assert not fails, fails
    assert np.median(ratios) <= 0.5, f"sigma ratio {np.median(ratios):.2f}"
    assert all(v >= 0.95 for v in frac_ok.values()), f"uncertainty not calibrated: {frac_ok}"


if __name__ == "__main__":
    test_nominal_K()
    test_motion_gate()
    test_image_speed_px()
    test_coverage_window()
    print("observer_geom logic OK")
