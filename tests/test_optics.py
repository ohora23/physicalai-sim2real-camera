"""Tests for optics helpers — plain script, no pytest."""

from __future__ import annotations

import math
import time

import cv2
import mujoco
import numpy as np

from physicalai_sim2real_camera import intrinsics_from_model
from physicalai_sim2real_camera.optics import (
    _undistort_rational,
    distortion_maps,
    dof,
    dof_buffers,
    is_rational,
    outside_domain_fraction,
    rational_domain,
    render_intrinsics,
    undistort_points,
    vignette_map,
)

# Base camera intrinsics
K_OUT = np.array(
    [[500.0, 0.0, 319.5],
     [0.0, 500.0, 239.5],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
W, H = 640, 480
cx, cy = K_OUT[0, 2], K_OUT[1, 2]
fx, fy = K_OUT[0, 0], K_OUT[1, 1]


def bilinear_sample(map_x: np.ndarray, map_y: np.ndarray, pts: np.ndarray):
    """Bilinearly sample (H, W) float arrays at float (x, y) coordinates.

    *pts* is ``(N, 2)`` in ``(x, y)`` order.  Returns ``(N, 2)`` of
    interpolated values — one row per channel (map_x then map_y).
    Only valid for points strictly inside ``[0, W-1] x [0, H-1]``.
    """
    h, w = map_x.shape
    xs = pts[:, 0]
    ys = pts[:, 1]

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    # clip
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    dx = xs - np.floor(xs)
    dy = ys - np.floor(ys)

    def _sample(m: np.ndarray):
        return (
            (1 - dx) * (1 - dy) * m[y0, x0]
            + dx * (1 - dy) * m[y0, x1]
            + (1 - dx) * dy * m[y1, x0]
            + dx * dy * m[y1, x1]
        )

    return np.stack([_sample(map_x), _sample(map_y)], axis=-1)


def test_identity_no_distortion():
    print("test_identity_no_distortion ...")
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    K_render, W_r, H_r, fovy = render_intrinsics(K_OUT, dist, W, H)
    assert W_r == 640 and H_r == 480
    assert np.allclose(K_render, K_OUT)

    map_x, map_y = distortion_maps(K_OUT, dist, W, H, K_render)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    assert float(np.max(np.abs(map_x - xs))) < 1e-3, "map_x drift"
    assert float(np.max(np.abs(map_y - ys))) < 1e-3, "map_y drift"
    print(f"  fovy = {fovy:.4f}°  map_x max|err| = {np.max(np.abs(map_x - xs)):.2e}  "
          f"map_y max|err| = {np.max(np.abs(map_y - ys)):.2e}")


def test_forward_consistency():
    print("test_forward_consistency ...")
    dist = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    K_render, W_r, H_r, _ = render_intrinsics(K_OUT, dist, W, H)
    map_x, map_y = distortion_maps(K_OUT, dist, W, H, K_render)

    rng = np.random.default_rng(0)
    N = 500
    # safe undistorted pixel range
    u_safe = rng.uniform(50, 590, N)
    v_safe = rng.uniform(50, 430, N)
    z = rng.uniform(0.5, 3.0, N)
    xn = (u_safe - cx) / fx
    yn = (v_safe - cy) / fy
    X3d = xn * z
    Y3d = yn * z
    object_pts = np.stack([X3d, Y3d, z], axis=-1).astype(np.float64)  # (N, 3)

    # distorted coords in K_out space
    u_d, _ = cv2.projectPoints(
        object_pts.reshape(1, N, 3),
        np.zeros(3), np.zeros(3),
        K_OUT, dist,
    )
    u_d = u_d.reshape(N, 2)  # (N, 2)

    # ideal coords in K_render space (no distortion)
    u_r, _ = cv2.projectPoints(
        object_pts.reshape(1, N, 3),
        np.zeros(3), np.zeros(3),
        K_render, np.zeros(5),
    )
    u_r = u_r.reshape(N, 2)

    sampled = bilinear_sample(map_x, map_y, u_d)  # (N, 2)
    err_per_point = np.max(np.abs(sampled - u_r), axis=-1)
    mean_err = float(np.mean(err_per_point))
    max_err = float(np.max(err_per_point))
    print(f"  mean max|err| = {mean_err:.4f} px  max max|err| = {max_err:.4f} px")
    assert mean_err < 0.05, f"mean error {mean_err:.4f} >= 0.05"
    assert max_err < 0.1, f"max error {max_err:.4f} >= 0.1"


def test_map_bounds():
    print("test_map_bounds ...")
    dist = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])

    k_off = K_OUT.copy()
    k_off[0, 2] = 334.5  # cx += 15
    for label, k in [("centered", K_OUT), ("off-center", k_off)]:
        k = k.astype(np.float64)
        K_r, W_r, H_r, _ = render_intrinsics(k, dist, W, H)
        mx, my = distortion_maps(k, dist, W, H, K_r)
        assert float(np.min(mx)) >= 0 and float(np.max(mx)) <= W_r - 2, \
            f"{label} map_x out of bounds [{np.min(mx):.1f}, {np.max(mx):.1f}] vs [0, {W_r-2}]"
        assert float(np.min(my)) >= 0 and float(np.max(my)) <= H_r - 2, \
            f"{label} map_y out of bounds [{np.min(my):.1f}, {np.max(my):.1f}] vs [0, {H_r-2}]"
        print(f"  {label}  W_r={W_r} H_r={H_r}  map_x ∈ [{np.min(mx):.1f}, {np.max(mx):.1f}]  "
              f"map_y ∈ [{np.min(my):.1f}, {np.max(my):.1f}]")


def test_render_size_ratio():
    print("test_render_size_ratio ...")
    dist = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    _, W_r, H_r, _ = render_intrinsics(K_OUT, dist, W, H)
    ratio = W_r / H_r
    aspect = 4.0 / 3.0
    rel_err = abs(ratio - aspect) / aspect
    print(f"  W_r={W_r}  H_r={H_r}  ratio={ratio:.4f}  target={aspect:.4f}  rel_err={rel_err:.4f}")
    assert rel_err <= 0.05, f"ratio {ratio:.4f} off by {rel_err:.4f} (>0.05)"


def test_vignette():
    print("test_vignette ...")
    dist = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    strength = 0.5

    V = vignette_map(K_OUT, dist, W, H, strength)
    assert V.shape == (H, W, 1), f"shape {V.shape}"

    # (a) Analytic check at exact principal point
    pt_cp = np.array([[[cx, cy]]], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    und = undistort_points(pt_cp, K_OUT, dist, None, crit)
    xn_cp = und[0, 0, 0]
    yn_cp = und[0, 0, 1]
    r2_cp = float(xn_cp ** 2 + yn_cp ** 2)
    v_center_expected = (1.0 + r2_cp) ** (-2.0 * strength)
    print(f"  center analytic: r2={r2_cp:.2e}  V_expected={v_center_expected:.6f}")
    assert abs(v_center_expected - 1.0) < 1e-6, f"center V_expected {v_center_expected}"

    # (b) Corner pixel (0,0)
    corner_row = int(round(cy))
    corner_col = int(round(cx))
    v_center_actual = float(V[corner_row % H, corner_col % W, 0])
    print(f"  V at nearest pixel ({corner_row},{corner_col}) = {v_center_actual:.6f}")

    # (c) Actual corner (0,0) — compute cos^4(theta)^strength
    pt_00 = np.array([[[0.0, 0.0]]], dtype=np.float32)
    und_00 = undistort_points(pt_00, K_OUT, dist, None, crit)
    xn_00 = float(und_00[0, 0, 0])
    yn_00 = float(und_00[0, 0, 1])
    r2_00 = xn_00 ** 2 + yn_00 ** 2
    cos_t = (1.0 + r2_00) ** -0.5
    v_corner_expected = cos_t ** (4.0 * strength)
    v_corner_actual = float(V[0, 0, 0])
    print(f"  corner(0,0): r2={r2_00:.6f}  V_expected={v_corner_expected:.6f}  V_actual={v_corner_actual:.6f}")
    assert abs(v_corner_actual - v_corner_expected) < 1e-4, \
        f"corner error {abs(v_corner_actual - v_corner_expected):.2e}"


# ── P4: depth of field ───────────────────────────────────────────────

def dof_scene():
    """Checker planes seen straight down: left half at 0.5 m, right half at 1.0 m.

    Returns linear float32 frame (γ=1: DN/255) and metric depth from a plain
    Renderer (no SimCamera noise/distortion), plus f_px.
    """
    xml = """<mujoco><visual><global offwidth="640" offheight="480"/></visual>
<asset><texture name="chk" type="2d" builtin="checker" width="512" height="512"
  rgb1="1 1 1" rgb2="0.1 0.1 0.1"/>
  <material name="chk" texture="chk" texrepeat="10 10" texuniform="true"/></asset>
<worldbody><light pos="0 0 2" directional="true" diffuse="1 1 1"/>
  <geom type="plane" pos="0.5 0 0" size="0.5 1 0.01" material="chk"/>
  <geom type="plane" pos="-0.5 0 0.5" size="0.5 1 0.01" material="chk"/>
  <camera name="cam" pos="0 0 1" fovy="45"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 480, 640)
    r.update_scene(data, "cam")
    w = r.render().astype(np.float32) / 255.0
    r.enable_depth_rendering()
    r.update_scene(data, "cam")
    depth = r.render().astype(np.float32)
    r.close()
    f_px = float(intrinsics_from_model(model, 0, 640, 480)[1, 1])
    return w, depth, f_px


def edge_width(g, x0, x1, h=12):
    """10–90% width (px) of the vertical checker edge nearest the middle of [x0, x1).

    Profile = mean of 7 rows centred in the tallest checker square (no horizontal
    edge), ±h px around the strongest step closest to the range centre.
    """
    gy = np.abs(np.diff(g[:, x0:x1].mean(1)))
    rows = np.flatnonzero(gy > 0.5 * gy.max())
    k = int(np.argmax(np.diff(rows)))
    mid = (rows[k] + rows[k + 1]) // 2
    prof = g[mid - 3:mid + 4, x0:x1].mean(0)
    gx = np.abs(np.diff(prof))
    peaks = np.flatnonzero(gx > 0.5 * gx.max())
    e = int(peaks[np.argmin(np.abs(peaks - (x1 - x0) / 2))])
    seg = prof[e - h:e + h + 2]
    t = (seg - seg[:3].mean()) / (seg[-3:].mean() - seg[:3].mean())
    t = np.maximum.accumulate(t)                     # rising, monotone for interp
    xs = np.arange(len(t))
    return float(np.interp(0.9, t, xs) - np.interp(0.1, t, xs))


def test_dof():
    """P4 acceptance: in-focus edge unchanged, out-of-focus 4σ == CoC ±30%, identity, far clip, bench."""
    w0, depth, f_px = dof_scene()
    A, zf, zfar = 0.016, 0.5, 1e3
    bufs = dof_buffers(480, 640)
    w = w0.copy()
    dof(w, depth, f_px, zf, 0.0, bufs, zfar)
    assert np.array_equal(w, w0)                                    # 3. aperture 0 -> identical
    dof(w, depth, f_px, zf, A, bufs, zfar)
    g0, g = w0.mean(2), w.mean(2)
    e0, e1 = edge_width(g0, 0, 300), edge_width(g, 0, 300)         # left plane z = 0.5 (focus)
    print(f"  DoF in-focus edge 10-90%: {e1:.2f} px vs sharp {e0:.2f} px")
    assert abs(e1 - e0) <= 1.0                                      # 1.
    r0, r1 = edge_width(g0, 340, 640), edge_width(g, 340, 640)      # right plane z = 1.0
    coc = A * f_px * abs(1 / 1.0 - 1 / zf)
    four_sigma = 4 * math.sqrt(r1 ** 2 - r0 ** 2) / 2.563
    print(f"  DoF 1.0 m plane: 4σ = {four_sigma:.2f} px vs CoC {coc:.2f} px ({four_sigma / coc - 1:+.1%})")
    assert abs(four_sigma - coc) <= 0.3 * coc                       # 2.
    w = w0.copy()                                                   # 2'. CoC ~32 px: half-res blur path (σ > 4)
    A2 = 0.055
    dof(w, depth, f_px, zf, A2, bufs, zfar)
    r2 = edge_width(w.mean(2), 340, 640, h=30)
    coc2 = A2 * f_px * abs(1 / 1.0 - 1 / zf)
    four_sigma2 = 4 * math.sqrt(r2 ** 2 - edge_width(g0, 340, 640, h=30) ** 2) / 2.563
    print(f"  DoF 1.0 m plane, A={A2}: 4σ = {four_sigma2:.2f} px vs CoC {coc2:.2f} px "
          f"({four_sigma2 / coc2 - 1:+.1%})")
    assert abs(four_sigma2 - coc2) <= 0.3 * coc2
    w = w0.copy()                                                   # far-clip background stays sharp
    bg = np.full_like(depth, zf)
    bg[:, 320:] = 20.0
    dof(w, bg, f_px, zf, A, bufs, 20.0)
    assert np.array_equal(w, w0)
    for _ in range(3):
        dof(w0.copy(), depth, f_px, zf, A, bufs, zfar)
    t0 = time.perf_counter()
    for _ in range(20):
        dof(w, depth, f_px, zf, A, bufs, zfar)
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  dof avg = {ms:.2f} ms (budget 4 ms)" + ("  WARNING > 4 ms" if ms > 4 else ""))


def test_undistort_points():
    """undistort_points inverts the distortion to 1e-4 px and dispatches on undistortPointsIter."""
    K = K_OUT
    dist = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    u, v = np.meshgrid(np.linspace(20, 620, 13), np.linspace(20, 460, 9))
    pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
    norm = undistort_points(pts, K, dist, None, crit)
    xyz = np.concatenate([norm.reshape(-1, 2), np.ones((pts.shape[0], 1))], 1)
    reproj, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, dist)
    assert np.abs(reproj - pts).max() < 1e-4
    pix = undistort_points(pts, K, dist, K, crit)  # P=K: pixels = K applied to the normalised points
    assert np.allclose(pix.reshape(-1, 2), norm.reshape(-1, 2) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]])
    # Iter present -> called with the criteria (4.x branch, faked on 5.x); 5.x itself exercises the keyword path above.
    calls = []
    saved = getattr(cv2, "undistortPointsIter", None)
    cv2.undistortPointsIter = lambda *a: calls.append(a) or "iter"
    try:
        assert undistort_points(pts, K, dist, None, crit) == "iter"
        assert len(calls) == 1 and calls[0][5] == crit
    finally:
        if saved is None:
            del cv2.undistortPointsIter
        else:
            cv2.undistortPointsIter = saved


if __name__ == "__main__":
    test_identity_no_distortion()
    test_forward_consistency()
    test_map_bounds()
    test_render_size_ratio()
    test_vignette()
    test_dof()
    test_undistort_points()
    print("ALL TESTS PASSED")


# a wide-angle rational factory-style model (640x400) that folds before the image corners
WIDE_K = np.array([[290.0, 0.0, 319.5], [0.0, 290.0, 199.5], [0.0, 0.0, 1.0]])
WIDE_DIST = (0.23, -0.07, 0.0, 0.0, -0.003, 0.55, -0.07, -0.015)  # synthetic; ~9 % of 640x400 past the fold


def test_is_rational_and_domain():
    """is_rational keys on non-zero k4..k6; rational_domain stops at the fold, or runs to r_max without one."""
    assert not is_rational((0.1, 0.0, 0.0, 0.0, 0.0)) and not is_rational((0.1,) + (0.0,) * 7)
    assert is_rational(WIDE_DIST) and not is_rational(None)
    ru_max, rd_max, ru_t, rd_t = rational_domain(WIDE_DIST)
    assert 1.9 < ru_max < 2.1 and 1.0 < rd_max < 1.2 and np.all(np.diff(rd_t) > 0) and ru_t[-1] == ru_max
    assert rational_domain((0.0,) * 5 + (0.01, 0.0, 0.0))[0] == 10.0  # gentle model: monotonic to r_max


def test_undistort_rational():
    """Rational inversion: exact (1e-6 px) where invertible, clamped past the fold, warned in render_intrinsics."""
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    v, u = np.mgrid[0:400:5, 0:640:5].astype(np.float64)
    pts = np.stack([u, v], -1).reshape(-1, 1, 2)
    norm, ok = _undistort_rational(pts, WIDE_K, WIDE_DIST, None, return_ok=True)
    assert np.array_equal(norm, undistort_points(pts, WIDE_K, WIDE_DIST, None, crit))  # dispatch
    xyz = np.concatenate([norm.reshape(-1, 2), np.ones((pts.shape[0], 1))], 1)
    reproj, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), WIDE_K, np.array(WIDE_DIST))
    err = np.abs(reproj - pts).reshape(-1, 2).max(1)
    assert 0.9 < ok.mean() < 1.0 and err[ok].max() < 1e-6
    assert np.hypot(*norm.reshape(-1, 2)[~ok].T).max() <= rational_domain(WIDE_DIST)[0] + 1e-9  # clamped to the domain
    assert abs(outside_domain_fraction(WIDE_K, WIDE_DIST, 640, 400, step=5) - (1 - ok.mean())) < 1e-12
    pix = undistort_points(pts, WIDE_K, WIDE_DIST, WIDE_K, crit)
    assert np.allclose(pix.reshape(-1, 2), norm.reshape(-1, 2) * [290.0, 290.0] + [319.5, 199.5])
    # a zero rational tail takes the OpenCV path: identical to the 5-term call
    d5 = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    assert np.array_equal(undistort_points(pts, WIDE_K, np.r_[d5, 0, 0, 0], None, crit),
                          undistort_points(pts, WIDE_K, d5, None, crit))
    try:
        _undistort_rational(pts, WIDE_K, WIDE_DIST + (0.01, 0, 0, 0), None)
        raise AssertionError("thin-prism terms must be refused")
    except ValueError:
        pass
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        K_r, W_r, H_r, _ = render_intrinsics(WIDE_K, WIDE_DIST, 640, 400)
    assert any("fold" in str(x.message) for x in w)
    mx, my = distortion_maps(WIDE_K, WIDE_DIST, 640, 400, K_r)
    xr, yr = (mx[::5, ::5] - K_r[0, 2]) / K_r[0, 0], (my[::5, ::5] - K_r[1, 2]) / K_r[1, 1]
    back, _ = cv2.projectPoints(np.stack([xr, yr, np.ones_like(xr)], -1).reshape(-1, 1, 3).astype(np.float64),
                                np.zeros(3), np.zeros(3), WIDE_K, np.array(WIDE_DIST))
    assert np.abs(back - pts).reshape(-1, 2).max(1)[ok].max() < 1e-3  # float32 maps
