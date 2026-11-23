"""Tests for depth_noise — plain script, no pytest.  MUJOCO_GL=egl."""

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np

from physicalai_sim2real_camera.depth import depth_buffers, depth_noise, edge_band, zfar_of

W, H = 640, 480


def cfg(**kw):
    base = dict(depth_sigma_a=0.0, depth_sigma_b=0.0, depth_min_m=0.0,
                depth_max_m=0.0, depth_edge_hole_px=0, seed=0)
    base.update(kw)
    return SimpleNamespace(**base)


def render_plane(h, half=0.5):
    """Downward camera at height h over a box top at z=0 (extent pinned -> zfar 20 m)."""
    xml = f"""<mujoco><statistic extent="0.4" center="0 0 0"/>
<visual><global offwidth="{W}" offheight="{H}"/></visual><worldbody>
  <light pos="0 0 3" directional="true"/>
  <geom type="box" size="{half} {half} 0.01" pos="0 0 -0.01"/>
  <camera name="cam" fovy="45" pos="0 0 {h}"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=H, width=W)
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam")
    z = r.render().astype(np.float32, copy=True)
    r.close()
    return z, zfar_of(model)


def _simple_zfar():
    """Minimal model for zfar (used by synthetic depth tests)."""
    xml = """<mujoco><statistic extent="1" center="0 0 0"/>
<visual><global offwidth="640" offheight="480"/></visual><worldbody>
  <camera name="cam" fovy="45" pos="0 0 1"/>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    return zfar_of(model)


def test_edge_band_synthetic():
    """Test edge_band on synthetic depth maps.

    (a) 40×60 float32 array of 1.0 with a step to 2.0 on columns >= 30;
        k in (1, 2, 3): band is exactly columns 30-k .. 29+k (all rows).
    (b) constant array -> empty band.
    (c) random array (rng seeded) -> band equals a numpy reference computed
        with a sliding-window max/min via stride_tricks.
    """
    z = np.ones((40, 60), np.float32)
    z[:, 30:] = 2.0
    bufs = depth_buffers(40, 60, edge=True)
    for k in (1, 2, 3):
        band = edge_band(z, k, bufs)
        expected = np.zeros((40, 60), bool)
        c0 = max(0, 30 - k)
        c1 = min(60, 30 + k)
        expected[:, c0:c1] = True
        exp_sum = (c1 - c0) * 40
        assert np.array_equal(
            band, expected), f"k={k}: sum={band.sum()} exp={exp_sum}"
    # constant -> empty band
    z_const = np.full((40, 60), 1.0, np.float32)
    band_c = edge_band(z_const, 2, bufs)
    assert not band_c.any(), "constant depth should produce empty band"
    # random array reference comparison
    rng = np.random.default_rng(99)
    z_rand = (1.0 + 0.001 * np.arange(60, dtype=np.float32))[None].repeat(40, 0)  # smooth ramp: no band on its own
    z_rand[rng.random((40, 60)) < 0.02] += 0.3  # sparse spikes: the band is their (2k+1)^2 neighbourhoods only
    band_r = edge_band(z_rand, 2, bufs).copy()
    mx_r, mn_r = bufs["mx"].copy(), bufs["mn"].copy()
    # reference: sliding-window max over a -inf padding and min over a +inf padding (cv2's default border)
    win = lambda fill: np.lib.stride_tricks.sliding_window_view(np.pad(z_rand, 2, constant_values=fill), (5, 5))  # noqa: E731
    z_max = win(-np.inf).max(axis=(-2, -1)).astype(np.float32)
    z_min = win(np.inf).min(axis=(-2, -1)).astype(np.float32)
    ref = (z_max - z_min) > z_rand * np.float32(0.1)
    assert np.array_equal(mx_r, z_max) and np.array_equal(mn_r, z_min), "dilate/erode differ from the reference"
    assert np.array_equal(band_r, ref) and 0.05 < ref.mean() < 0.95, f"random band mismatch (mean {ref.mean():.2f})"
    # a smooth ramp (0.1 %/px) has no band away from steps, so a padding mistake would show at the image border
    ramp = (1.0 + 0.001 * np.arange(60, dtype=np.float32))[None].repeat(40, 0)
    assert not edge_band(ramp, 2, bufs).any(), "smooth ramp must give an empty band, including the border"
    print("test_edge_band_synthetic PASSED")


def test_edge_holes_two_depth():
    """Accept 1, CPU: clean depth of two_depth_scene, k=2, no axial noise.

    Hole ratio in [0.45, 0.55]; flying >= 0.9*(valid-holes); flying in [mn,mx];
    outside band == clean bit-exactly; differs from k=0.
    """
    from test_gpu import two_depth_scene
    model, data = two_depth_scene()
    r = mujoco.Renderer(model, height=480, width=640)
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam")
    clean = r.render().astype(np.float32, copy=True)
    r.close()
    zfar = zfar_of(model)
    c = cfg(depth_edge_hole_px=2, depth_sigma_a=0.0, depth_sigma_b=0.0)
    bufs = depth_buffers(480, 640, edge=True)
    band = edge_band(clean, 2, depth_buffers(480, 640, edge=True)).copy()  # own buffers: depth_noise reuses bufs
    valid = band & (clean > 0) & (clean < zfar * (1 - 1e-6))
    out = depth_noise(clean.copy(), c, np.random.default_rng([0, 2]), bufs, zfar)
    hole_ratio = np.mean(out[valid] == 0)
    print(f"two_depth: hole_ratio={hole_ratio:.4f} valid={valid.sum()}")
    assert 0.45 <= hole_ratio <= 0.55, f"hole ratio {hole_ratio:.4f} not in [0.45, 0.55]"
    flying = valid & (out != 0) & (out != clean)
    assert flying.sum() >= 0.9 * (valid.sum() - (out[valid] == 0).sum()), \
        f"flying pixels {flying.sum()} < 0.9*(valid-holes)"
    assert np.all((out[flying] >= bufs["mn"][flying]) & (out[flying] <= bufs["mx"][flying])), "flying outside [mn, mx]"
    outside = ~band & (clean > 0) & (clean < zfar * (1 - 1e-6))
    assert np.array_equal(out[outside], clean[outside]), "outside band must match clean"
    out_k0 = depth_noise(clean.copy(), cfg(), np.random.default_rng([0, 2]), depth_buffers(480, 640, edge=True), zfar)
    assert not np.array_equal(out, out_k0), "k=2 must differ from k=0"
    print(f"test_edge_holes_two_depth PASSED (hole={hole_ratio:.4f}, flying={flying.sum()})")


def test_edge_holes_plane():
    """Accept 2: plane without background (k=2 == clean), silhouette case.

    Plane (no bg): k=2 alone == clean bit-exactly; k=2+a=0.001 gives 0 holes.
    Silhouette: band pixels adjacent to bg -> 0; no pixel exceeds plane+0.01.
    """
    clean, zfar = render_plane(1.0, half=2.0)
    # k=2 alone, no axial noise: no holes, no flying, == clean
    c_k = cfg(depth_edge_hole_px=2, depth_sigma_a=0.0, depth_sigma_b=0.0)
    out_k = depth_noise(clean.copy(), c_k, np.random.default_rng([0, 2]),
                        depth_buffers(480, 640, edge=True), zfar)
    assert np.array_equal(out_k, clean), "k=2 no noise on clean plane should be bit-identical"
    # k=2 + a=0.001: no holes
    c_ka = cfg(depth_edge_hole_px=2, depth_sigma_a=0.001, depth_sigma_b=0.0)
    out_ka = depth_noise(clean.copy(), c_ka, np.random.default_rng([0, 2]),
                         depth_buffers(480, 640, edge=True), zfar)
    assert (out_ka == 0).sum() == 0, f"k=2+a=0.001 on plane: {(out_ka == 0).sum()} holes"
    # Silhouette (default half=0.5, background visible)
    sil, zfar_s = render_plane(1.0, half=0.5)
    c_s = cfg(depth_edge_hole_px=2, depth_sigma_a=0.001, depth_sigma_b=0.0)
    out_s = depth_noise(sil.copy(), c_s, np.random.default_rng([0, 2]),
                        depth_buffers(480, 640, edge=True), zfar_s)
    plane_z = float(np.median(sil[sil > 0]))
    # Every band pixel adjacent to bg (mx >= zfar*(1-1e-6)) should be 0
    bufs_s = depth_buffers(480, 640, edge=True)
    edge_band(sil, 2, bufs_s)
    bg_adj = bufs_s["mx"] >= zfar_s * (1 - 1e-6)
    assert np.all(out_s[bg_adj] == 0), f"bg-adjacent band pixels: {(out_s[bg_adj] != 0).sum()} non-zero"
    assert np.all(out_s <= plane_z + 0.01), f"flying pixel exceeded: max={out_s.max():.4f}"
    print(f"test_edge_holes_plane PASSED (plane z={plane_z:.4f}, sil max={out_s.max():.4f})")


def test_edge_holes_k0():
    """Accept 3: cfg(depth_edge_hole_px=0) output is bit-equal to k=0 copy."""
    clean, zfar = render_plane(1.0, half=2.0)
    c = cfg(depth_edge_hole_px=0, depth_sigma_a=0.001, depth_sigma_b=0.0)
    out = depth_noise(clean.copy(), c, np.random.default_rng([0, 2]),
                      depth_buffers(480, 640, edge=True), zfar)
    out_ref = depth_noise(clean.copy(), cfg(depth_sigma_a=0.001, depth_sigma_b=0.0),
                          np.random.default_rng([0, 2]), depth_buffers(480, 640, edge=True), zfar)
    assert np.array_equal(out, out_ref), "k=0 must be bit-equal to k=0-only-noise"
    print("test_edge_holes_k0 PASSED")


def test_depth_identity():
    """Line 51: a==0, b==0, lo<=0, hi<=0 -> return z unchanged."""
    z = np.ones((H, W), np.float32) * 0.5
    c = cfg()
    bufs = depth_buffers(H, W, edge=True)
    zfar = _simple_zfar()
    out = depth_noise(z.copy(), c, np.random.default_rng([0, 2]), bufs, zfar=zfar)
    assert np.array_equal(out, z)


def test_depth_min_range():
    """Lines 57-58: lo > 0 masks pixels below lo threshold."""
    z = np.ones((H, W), np.float32) * 0.3  # all below lo=0.4
    c = cfg(depth_min_m=0.4)
    bufs = depth_buffers(H, W, edge=True)
    zfar = _simple_zfar()
    out = depth_noise(z.copy(), c, np.random.default_rng([0, 2]), bufs, zfar=zfar)
    assert np.all(out == 0), "pixels below min_range should be invalidated"


def test_depth_max_range():
    """Lines 60-61: hi > 0 masks pixels above hi threshold."""
    z = np.ones((H, W), np.float32) * 5.0  # all above hi=2.0
    c = cfg(depth_max_m=2.0)
    bufs = depth_buffers(H, W, edge=True)
    zfar = _simple_zfar()
    assert zfar > 5.0  # background-invalidation branch must not fire for this test
    out = depth_noise(z.copy(), c, np.random.default_rng([0, 2]), bufs, zfar=zfar)
    assert np.all(out == 0), "pixels above max_range should be invalidated"


def test_depth_noise_addition():
    """Lines 62->70: a!=0 or b!=0 branch adds noise to valid pixels."""
    z = np.full((H, W), 1.0, np.float32)
    c = cfg(depth_sigma_a=0.01, depth_sigma_b=0.0)
    bufs = depth_buffers(H, W, edge=True)
    zfar = _simple_zfar()
    rng = np.random.default_rng([42, 2])
    out = depth_noise(z.copy(), c, rng, bufs, zfar=zfar)
    assert not np.array_equal(out, z), "noise should change the values"
    assert np.all(out > 0), "valid pixels should stay positive"


def main():
    c = cfg(depth_sigma_a=0.001, depth_sigma_b=0.002)
    bufs = depth_buffers(H, W, edge=True)

    # 1 + 2: three planes, fit sigma = a + b z^2; background -> 0 and excluded
    zs, sig = [], []
    for h in (0.5, 1.0, 2.0):
        clean, zfar = render_plane(h)
        bg = clean >= zfar * (1 - 1e-6)
        noisy = depth_noise(clean.copy(), c, np.random.default_rng([0, 2]), bufs, zfar=zfar)
        assert np.all(noisy[bg] == 0), "background not invalidated"
        valid = noisy > 0
        assert not np.any(valid & bg)
        if h == 2.0:
            assert bg.any(), "expected background at 2 m"
            print(f"zfar={zfar:.4f}  clean bg={clean[bg].mean():.4f}")
            assert abs(clean[bg].mean() - 20.0) < 0.01
        zs.append(float(np.median(clean[valid])))
        sig.append(float(np.std(noisy[valid] - clean[valid])))
    A = np.stack([np.ones(3), np.array(zs) ** 2], 1)
    fa, fb = np.linalg.lstsq(A, np.array(sig), rcond=None)[0]
    print(f"planes z={zs} std={sig}  fit a={fa:.5f} (0.001) b={fb:.5f} (0.002)")
    assert abs(fb - 0.002) / 0.002 < 0.15, fb
    assert abs(fa - 0.001) / 0.001 < 0.30, fa

    # 3: min range
    clean, zfar = render_plane(0.3)
    out = depth_noise(clean.copy(), cfg(depth_sigma_a=0.001, depth_min_m=0.4),
                      np.random.default_rng([0, 2]), bufs, zfar=zfar)
    assert np.all(out == 0), "0.3 m plane should be fully invalid with min 0.4"

    # 4: identity
    clean, zfar = render_plane(1.0)
    out = depth_noise(clean.copy(), cfg(), np.random.default_rng([0, 2]), bufs, zfar=zfar)
    assert np.array_equal(out, clean)

    # 5: determinism per seed / cam_id
    run = lambda cam: depth_noise(clean.copy(), c, np.random.default_rng([c.seed + cam, 2]), bufs, zfar=zfar)
    assert np.array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))

    # 6: bench (warning only)
    z = clean.copy()
    rng = np.random.default_rng([0, 2])
    for _ in range(5):
        np.copyto(z, clean); depth_noise(z, c, rng, bufs, zfar=zfar)
    N = 50
    t = 0.0
    for _ in range(N):
        np.copyto(z, clean)
        t0 = time.perf_counter(); depth_noise(z, c, rng, bufs, zfar=zfar); t += time.perf_counter() - t0
    ms = t / N * 1e3
    print(f"bench depth_noise 640x480: {ms:.2f} ms" + ("  WARNING > 3 ms" if ms > 3 else ""))
    print("ALL DEPTH TESTS PASSED")


if __name__ == "__main__":
    test_edge_band_synthetic()
    test_edge_holes_two_depth()
    test_edge_holes_plane()
    test_edge_holes_k0()
    test_depth_identity()
    test_depth_min_range()
    test_depth_max_range()
    test_depth_noise_addition()
    main()
