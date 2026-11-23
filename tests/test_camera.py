"""Tests for intrinsics_from_model and SimCamera — plain script, no pytest."""

import dataclasses
import math
import sys
import time

import cv2
import mujoco
import numpy as np

from physicalai_sim2real_camera import CameraConfig, SimCamera, intrinsics_from_model
from physicalai_sim2real_camera.optics import undistort_points


def test_intrinsics():
    # ── CHECK A: fovy camera ──────────────────────────────────────────
    xml_fovy = (
        '<mujoco>'
        '<worldbody><body><camera name="cam" fovy="45"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    model_a = mujoco.MjModel.from_xml_string(xml_fovy)
    cam_id = model_a.camera("cam").id
    W, H = 640, 480
    K = intrinsics_from_model(model_a, cam_id, W, H)

    expected_fy = 240.0 / math.tan(math.radians(22.5))
    print(f"CHECK A — K =\n{K}")
    print(f"CHECK A — expected_fy = {expected_fy}")

    assert abs(K[1, 1] - expected_fy) / expected_fy < 1e-9, "fy mismatch"
    assert K[0, 0] == K[1, 1], "fx != fy"
    assert abs(K[0, 2] - 319.5) < 1e-9, f"cx = {K[0, 2]}"
    assert abs(K[1, 2] - 239.5) < 1e-9, f"cy = {K[1, 2]}"

    # ── CHECK B: sensorsize camera (requires EGL) ────────────────────
    xml_sensor = """<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="flat" rgb1="0 0 0" width="2" height="2"/>
    <material name="dark" texture="grid" rgba="0.02 0.02 0.02 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" diffuse="1 1 1"/>
    <camera name="cam2" pos="0 0 1"
            sensorsize="0.00368 0.00276"
            focal="0.0022 0.0022"
            principal="0.0003 0.0002"
            resolution="640 480"/>
    <geom name="plane" type="plane" size="1 1 0.01" material="dark"/>
    <geom name="ball" type="sphere" size="0.01" pos="0.1 0.05 0" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>"""
    model_b = mujoco.MjModel.from_xml_string(xml_sensor)
    cam_id = model_b.camera("cam2").id
    data = mujoco.MjData(model_b)
    mujoco.mj_forward(model_b, data)

    W, H = 640, 480
    K = intrinsics_from_model(model_b, cam_id, W, H)
    print(f"CHECK B — K =\n{K}")

    # render
    renderer = mujoco.Renderer(model_b, height=480, width=640)
    renderer.update_scene(data, camera="cam2")
    img = renderer.render()  # (480, 640, 3) uint8
    renderer.close()

    gray = img.astype(np.float64).max(axis=2)
    threshold = 0.8 * gray.max()
    mask = gray > threshold
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))  # (x, y)
    print(f"CHECK B — centroid = {centroid}")

    # expected projection
    cam_xpos = data.cam_xpos[cam_id]
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p_w = data.geom("ball").xpos
    p_cam = R.T @ (p_w - cam_xpos)
    p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]], dtype=np.float64)

    pts, _ = cv2.projectPoints(
        p_cv.reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        None,
    )
    u, v = pts[0, 0, 0], pts[0, 0, 1]
    print(f"CHECK B — projected point = ({u:.4f}, {v:.4f})")

    assert abs(u - centroid[0]) <= 0.5, (
        f"u = {u:.4f} vs centroid x = {centroid[0]:.4f}"
    )
    assert abs(v - centroid[1]) <= 0.5, (
        f"v = {v:.4f} vs centroid y = {centroid[1]:.4f}"
    )


# ── T4a: SimCamera integration ──────────────────────────────────────
W, H = 640, 480
DIST = (-0.3, 0.1, 0.001, -0.001, 0.0)
SQ, NX, NY = 0.05, 8, 6          # 8×6 squares → 7×5 inner corners
BX, BY, TOP = 0.05, 0.0, 1e-4    # board centre, box top height (thin: white corners lie on z=0)
CAM_Z = 0.35
CAM_POSE = f'pos="0.05 0 {CAM_Z}" xyaxes="1 0 0 0 1 0"'
SS = 'sensorsize="0.00368 0.00276" focal="0.0022 0.0022" principal="0.0003 0.0002" resolution="640 480"'


def board_scene():
    boxes = "".join(
        f'<geom type="box" size="{SQ/2} {SQ/2} {TOP/2}" rgba="0 0 0 1" '
        f'pos="{BX + (i - NX/2 + 0.5)*SQ} {BY + (j - NY/2 + 0.5)*SQ} {TOP/2}"/>'
        for i in range(NX) for j in range(NY) if (i + j) % 2 == 0)
    xml = f"""<mujoco><visual><global offwidth="900" offheight="700"/></visual><worldbody>
  <light pos="0 0 2" directional="true" diffuse="1 1 1"/>
  <geom type="plane" size="1 1 0.01" rgba="1 1 1 1"/>{boxes}
  <camera name="cam" fovy="45" {CAM_POSE}/>
  <camera name="cam_ss" {CAM_POSE} {SS}/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def inner_corners_world():
    xs = BX + (np.arange(NX - 1) - (NX/2 - 1)) * SQ
    ys = BY + (np.arange(NY - 1) - (NY/2 - 1)) * SQ
    return np.array([[x, y, TOP] for y in ys for x in xs])


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ok, c = cv2.findChessboardCornersSB(gray, (NX - 1, NY - 1))
    assert ok, "chessboard not found"
    return c.reshape(-1, 2).astype(np.float64)


def nn_rms(a, b):
    """RMS of each point in a to its nearest neighbour in b (1:1 check)."""
    dd = np.linalg.norm(a[:, None] - b[None], axis=2)
    idx = dd.argmin(1)
    assert len(set(idx)) == len(a), "ambiguous corner matching"
    return float(np.sqrt((dd.min(1) ** 2).mean())), b[idx]


def geometry_rms(cam_name, dist=DIST):
    model, data = board_scene()
    sc = SimCamera(model, cam_name, CameraConfig(W, H, dist=dist))
    det = detect(sc.render(data))
    cam = sc.cam
    R = data.cam_xmat[cam].reshape(3, 3)
    p = (inner_corners_world() - data.cam_xpos[cam]) @ R      # = Rᵀ(p_w − c)
    p_cv = p * np.array([1.0, -1.0, -1.0])
    proj, _ = cv2.projectPoints(p_cv.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                sc.K_out, np.array(dist))
    proj = proj.reshape(-1, 2)
    rms, det_aligned = nn_rms(proj, det)
    sc.close()
    return rms, proj, det_aligned, sc.K_out


def test_geometry():
    rms, *_ = geometry_rms("cam")
    print(f"T4a-1 geometry RMS (cam) = {rms:.4f} px")
    assert rms < 0.5


def test_geometry_rational():
    """A rational (k4-k6) model renders to the geometry OpenCV projects with it (same 0.5 px bar as 5 terms).

    Plumbing only: on this narrow camera OpenCV's iteration would also converge; the solver itself (where OpenCV
    fails, a wide-angle factory-style model) is tested in test_optics.test_undistort_rational.
    """
    rational = (0.25, -0.05, 0.001, -0.001, 0.0, 0.6, -0.05, 0.0)
    rms, *_ = geometry_rms("cam", dist=rational)
    rms5, *_ = geometry_rms("cam", dist=(0.25, -0.05, 0.001, -0.001, 0.0))
    print(f"rational geometry RMS (cam) = {rms:.4f} px")
    assert rms < 0.5 and rms5 < 0.5


def test_distortion_applied():
    _, _, det, K = geometry_rms("cam")
    ideal = inner_corners_world()[:, :2]                       # aligned with det

    def h_rms(pts):
        Hm, _ = cv2.findHomography(ideal, pts, 0)
        pr = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        return float(np.sqrt(((pr - pts) ** 2).sum(1).mean()))

    und = undistort_points(det.reshape(-1, 1, 2), K, np.array(DIST), K,
                           (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    r_und, r_raw = h_rms(und.reshape(-1, 2)), h_rms(det)
    print(f"T4a-2 homography RMS undistorted = {r_und:.4f} px, raw = {r_raw:.4f} px")
    assert r_und < 0.5 and r_raw > 2.0


def test_border():
    xml = f"""<mujoco><visual><global offwidth="900" offheight="700"/></visual>
  <asset><material name="glow" rgba="1 1 1 1" emission="1"/></asset><worldbody>
  <geom type="plane" size="10 10 0.01" material="glow"/>
  <camera name="cam" fovy="45" {CAM_POSE}/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sc = SimCamera(model, "cam", CameraConfig(W, H, dist=DIST))
    img = sc.render(data)
    sc.close()
    print(f"T4a-3 border min pixel = {img.min()}")
    assert img.min() > 0


def test_sensorsize():
    rms, *_ = geometry_rms("cam_ss")
    # dist=0 must still honour the off-centre principal point (remap path)
    rms0, *_ = geometry_rms("cam_ss", dist=(0.0,) * 5)
    print(f"T4a-4 geometry RMS (cam_ss) = {rms:.4f} px, dist=0: {rms0:.4f} px")
    assert rms < 0.5 and rms0 < 0.5


def test_no_leak():
    model, data = board_scene()
    ids = [model.camera(n).id for n in ("cam", "cam_ss")]
    fovy0, ss0 = model.cam_fovy.copy(), model.cam_sensorsize.copy()
    plain = mujoco.Renderer(model, H, W)

    def shot(name):
        plain.update_scene(data, camera=name)
        return plain.render()

    before = [shot("cam"), shot("cam_ss")]
    for name in ("cam", "cam_ss"):
        sc = SimCamera(model, name, CameraConfig(W, H, dist=DIST))
        sc.render(data)
        sc.render(data, depth=True)
        sc.close()
    after = [shot("cam"), shot("cam_ss")]
    plain.close()
    assert all(np.array_equal(a, b) for a, b in zip(before, after)), "render leak"
    assert np.array_equal(model.cam_fovy, fovy0) and np.array_equal(model.cam_sensorsize, ss0)
    assert all(model.cam_sensorsize[i].any() == (i == ids[1]) for i in ids)
    print("T4a-5 no leak OK")


def test_depth_identity():
    model, data = board_scene()
    sc = SimCamera(model, "cam", CameraConfig(W, H, dist=DIST))
    rgb, d = sc.render(data, depth=True)
    sc.close()
    assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
    assert d.dtype == np.float32 and d.shape == (H, W)
    c = d[H//2 - 20:H//2 + 20, W//2 - 20:W//2 + 20]
    assert np.isfinite(c).all() and (c > 0).all()
    assert abs(float(np.median(c)) - (CAM_Z - TOP)) < 0.01, np.median(c)

    ident = SimCamera(model, "cam", CameraConfig(W, H))
    assert ident.identity
    out = ident.render(data)
    ident.close()
    plain = mujoco.Renderer(model, H, W)
    plain.update_scene(data, camera="cam")
    ref = plain.render()
    plain.close()
    assert np.array_equal(out, ref), "identity path not bit-identical"
    print("T4a-6 depth + identity OK")


def test_pipeline_math():
    """No-noise pipeline == encode(linearize(plain)·V·2^ev + black) within 1 LSB (catches order/wiring)."""
    from physicalai_sim2real_camera.sensor import encode, linearize
    model, data = board_scene()
    cfg = CameraConfig(W, H, vignette=0.5, exposure_ev=-1.0, black_level=0.02)
    sc = SimCamera(model, "cam", cfg)
    out = sc.render(data)
    V = sc.vignette
    sc.close()
    plain = mujoco.Renderer(model, H, W)
    plain.update_scene(data, camera="cam")
    exp = encode(linearize(plain.render(), cfg.gamma) * V * np.float32(0.5) + np.float32(0.02), cfg.gamma)
    plain.close()
    diff = np.abs(out.astype(int) - exp).max()
    print(f"T5 pipeline max |diff| = {diff} LSB")
    assert diff <= 1


# ── PI: Phase 2 wiring ────────────────────────────────────────────────

def _plain(model, data, name="cam"):
    r = mujoco.Renderer(model, H, W)
    r.update_scene(data, camera=name)
    img = r.render()
    r.close()
    return img


def _gradient_ff():
    return np.linspace(0.5, 1.0, W, dtype=np.float32)[None, :, None].repeat(H, 0)


PI_EFFECTS = {
    "flat_field": dict(flat_field=_gradient_ff()),
    "dof": dict(aperture_m=0.016, focus_m=0.5),
    "ae": dict(ae_tau_s=1.0),
    "bayer": dict(bayer=True),
    "depth_noise": dict(depth_sigma_b=0.002),
}


def _shot(model, data, depth=False, **kw):
    sc = SimCamera(model, "cam", CameraConfig(W, H, **kw))
    out = sc.render(data, depth=depth)
    sc.close()
    return out


def test_pi_identity():
    model, data = board_scene()
    assert np.array_equal(_shot(model, data), _plain(model, data))
    print("PI-1 default config == plain Renderer (bit)")


def test_pi_each_effect():
    model, data = board_scene()
    base = dict(vignette=0.3, sigma_shot=0.0)  # non-identity baseline so every stage runs
    off_rgb, off_d = _shot(model, data, depth=True, **base)
    for name, kw in PI_EFFECTS.items():
        rgb, d = _shot(model, data, depth=True, **base, **kw)
        assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3), name
        assert d.dtype == np.float32 and d.shape == (H, W), name
        # AE's first frame is (by design) the AE-off frame; compare its 2nd frame
        if name == "ae":
            sc = SimCamera(model, "cam", CameraConfig(W, H, **base, **kw))
            sc.render(data)
            rgb = sc.render(data)
            sc.close()
        changed = not np.array_equal(d, off_d) if name == "depth_noise" else not np.array_equal(rgb, off_rgb)
        assert changed, f"{name} had no effect"
    print(f"PI-2 each effect changes output, depth=True works: {list(PI_EFFECTS)}")


def test_pi_ae():
    from physicalai_sim2real_camera.sensor import linearize
    model, data = board_scene()
    off = _shot(model, data, exposure_ev=-0.3)
    sc = SimCamera(model, "cam", CameraConfig(W, H, exposure_ev=-0.3, ae_tau_s=1.0, ae_target=0.18))
    first = sc.render(data)
    assert np.array_equal(first, off), "AE first frame != AE-off (double exposure_ev?)"
    for _ in range(199):
        out = sc.render(data)
    ev = sc._ev
    sc.close()
    m0 = float(linearize(first, 2.2).mean())
    m = float(linearize(out, 2.2).mean())
    print(f"PI-3 AE first frame == AE off; linear mean {m0:.4f} -> {m:.4f} after 200 frames (ev {ev:+.3f})")
    assert abs(m - 0.18) / 0.18 < 0.10


def test_pi_flat_field():
    from physicalai_sim2real_camera.optics import vignette_map
    model, data = board_scene()
    sc = SimCamera(model, "cam", CameraConfig(W, H, dist=DIST, vignette=1.0))
    ref = sc.render(data)
    ff = vignette_map(sc.K_out, DIST, W, H, 1.0)
    sc.close()
    out = _shot(model, data, dist=DIST, flat_field=ff)
    assert np.array_equal(out, ref)
    print("PI-4 flat_field == cos^4 vignette(1.0) (bit)")


def test_pi_depth_noise():
    xml = f"""<mujoco><visual><global offwidth="900" offheight="700"/></visual><worldbody>
  <light pos="0 0 2" directional="true" diffuse="1 1 1"/>
  <geom type="box" size="0.1 0.1 0.01" rgba="1 1 1 1"/>
  <camera name="cam" fovy="45" {CAM_POSE}/></worldbody></mujoco>"""  # small board, far-clip background
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, clean = _shot(model, data, depth=True, vignette=0.3)
    _, noisy = _shot(model, data, depth=True, vignette=0.3, depth_sigma_b=0.002)
    board = clean < 1.0
    bg = clean > 5.0
    assert board.any() and bg.any()
    assert not np.array_equal(noisy[board], clean[board]) and (noisy[board] > 0).all()
    assert (noisy[bg] == 0).all()
    print(f"PI-5 depth noise: board std {float(np.std(noisy[board] - clean[board])):.2e} m, "
          f"background -> 0 ({int(bg.sum())} px)")


# ── bench() ───────────────────────────────────────────────────────────

def bench():
    """Benchmark SimCamera render timing.

    Post-processing time measured as mean(render wall) - mean(plain renderer.render wall)
    (subtractive, not exact per-call isolation).
    """
    W, H = 640, 480
    model, data = board_scene()
    cfg = CameraConfig(
        width=W, height=H,
        dist=(-0.3, 0.1, 0.001, -0.001, 0.0),
        vignette=1.0, exposure_ev=-0.3,
        sigma_shot=0.02, sigma_read=0.005,
        sigma_prnu=0.02, sigma_dsnu=0.005,
        black_level=0.02, gamma=2.2,
    )
    sc = SimCamera(model, "cam", cfg)

    # warm up
    for _ in range(20):
        sc.render(data)

    # full render timing
    n = 200
    full_times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sc.render(data)
        full_times.append((time.perf_counter() - t0) * 1000)

    # plain renderer timing (same margin size)
    W_r, H_r = sc.renderer.width, sc.renderer.height
    model.vis.global_.offwidth, model.vis.global_.offheight = W_r, H_r
    plain = mujoco.Renderer(model, H_r, W_r)
    plain_times = []
    for _ in range(20):
        plain.update_scene(data, camera=sc.cam)
        plain.render()
    for _ in range(n):
        t0 = time.perf_counter()
        plain.update_scene(data, camera=sc.cam)
        plain.render()
        plain_times.append((time.perf_counter() - t0) * 1000)
    plain.close()
    sc.close()

    def stats(ms):
        m = np.mean(ms)
        return m, float(np.median(ms)), float(np.percentile(ms, 95))

    full_mean, full_med, full_p95 = stats(full_times)
    plain_mean, plain_med, plain_p95 = stats(plain_times)
    post_mean = full_mean - plain_mean
    post_med = full_med - plain_med
    post_p95 = full_p95 - plain_p95

    print(f"full render:   mean={full_mean:.2f} ms  median={full_med:.2f} ms  p95={full_p95:.2f} ms")
    print(f"post-proc:     mean={post_mean:.2f} ms  median={post_med:.2f} ms  p95={post_p95:.2f} ms")
    if post_mean > 5.0:
        print(f"WARNING: post-processing {post_mean:.2f} ms > 5 ms")

    # PI: each Phase 2 effect individually (on top of the Phase 1 config above)
    model, data = board_scene()
    base = dataclasses.asdict(cfg)
    for name, kw in [("phase1", {})] + list(PI_EFFECTS.items()):
        c = CameraConfig(**{**base, **kw})
        sc = SimCamera(model, "cam", c)
        dep = name == "depth_noise"
        for _ in range(10):
            sc.render(data, depth=dep)
        t0 = time.perf_counter()
        for _ in range(100):
            sc.render(data, depth=dep)
        ms = (time.perf_counter() - t0) * 10
        sc.close()
        print(f"effect {name:12s}: {ms:.2f} ms/frame" + (" (depth=True)" if dep else ""))
        if ms > full_mean + 5.0:
            print(f"WARNING: {name} adds {ms - full_mean:.2f} ms/frame")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        test_intrinsics()
        test_geometry()
        test_distortion_applied()
        test_border()
        test_sensorsize()
        test_no_leak()
        test_depth_identity()
        test_pipeline_math()
        test_pi_identity()
        test_pi_each_effect()
        test_pi_ae()
        test_pi_flat_field()
        test_pi_depth_noise()
        print("OK")
