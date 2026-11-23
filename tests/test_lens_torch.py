"""Tests for physicalai_sim2real_camera.lens_torch — GPU twins of camera-side effects."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physicalai_sim2real_camera import lens  # noqa: E402
from physicalai_sim2real_camera.lens_torch import blur, ccm, gaussian_kernel, row_gain_, veil  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)


def test_ccm() -> None:
    """Random w (2,3,8,9), random M: equals np.einsum within 1e-6."""
    w_np = rng.standard_normal((2, 3, 8, 9)).astype(np.float32)
    M_np = rng.standard_normal((3, 3)).astype(np.float32)
    w_t = torch.from_numpy(w_np).to(device)
    M_t = torch.from_numpy(M_np).to(device)
    out = ccm(w_t, M_t)
    expected = np.einsum("ck,nkhw->nchw", M_np, w_np)
    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=0, atol=1e-6)


def test_row_gain() -> None:
    """g (2,8) random: w * g[:, None, :, None] within 1e-7; returns the same object."""
    w_np = rng.standard_normal((2, 3, 8, 9)).astype(np.float32)
    g_np = rng.standard_normal((2, 8)).astype(np.float32)
    w_t = torch.from_numpy(w_np).to(device)
    g_t = torch.from_numpy(g_np).to(device)
    out = row_gain_(w_t, g_t)
    expected = w_np * g_np[:, None, :, None]
    assert out is w_t
    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=0, atol=1e-7)


def test_gaussian_kernel() -> None:
    """sigma in (0.5, 1.0, 2.0, 3.0): equals cv2.getGaussianKernel with the stated ksize (1e-7),
    sums to 1 (1e-6), odd length."""
    for sigma in (0.5, 1.0, 2.0, 3.0):
        ksize = int(round(sigma * 8 + 1)) | 1
        cv2_kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()
        t_kernel = gaussian_kernel(sigma, device)
        assert t_kernel.shape == (ksize,)
        assert ksize % 2 == 1
        np.testing.assert_allclose(t_kernel.cpu().numpy(), cv2_kernel, rtol=0, atol=1e-7)
        np.testing.assert_allclose(t_kernel.cpu().sum().item(), 1.0, rtol=0, atol=1e-6)


def test_blur_matches_cv2() -> None:
    """Image (1,3,47,61) random; sigma in (0.8, 2.0, 3.0); compare with cv2.GaussianBlur."""
    w_np = rng.standard_normal((1, 3, 47, 61)).astype(np.float32)
    for sigma in (0.8, 2.0, 3.0):
        for border, cv2_border in (("reflect", cv2.BORDER_REFLECT),
                                    ("reflect101", cv2.BORDER_REFLECT_101)):
            # Convert to HWC for cv2
            w_hwc = w_np[0].transpose(1, 2, 0)  # (47, 61, 3)
            cv2_out = np.stack(
                [cv2.GaussianBlur(w_hwc[:, :, c], (0, 0), sigma, borderType=cv2_border)
                 for c in range(3)],
                axis=2,
            ).astype(np.float32)  # (H, W, 3)
            # Convert back to CHW for comparison
            cv2_out_chw = cv2_out.transpose(2, 0, 1)[None]  # (1, 3, H, W)
            w_t = torch.from_numpy(w_np).to(device)
            out = blur(w_t, sigma, border=border)
            np.testing.assert_allclose(out.cpu().numpy(), cv2_out_chw, rtol=0, atol=2e-5)


def test_veil_matches_lens() -> None:
    """veil(w, sD) vs lens.veil(w_hwc, D, 0.3) for divisible and non-divisible sizes."""
    # Divisible: H=128, W=160
    H, W = 128, 160
    D = lens.dirt_maps(H, W, 0.5, np.random.default_rng(1))[1]  # (H, W, 1)
    w_np = rng.random((H, W, 3)).astype(np.float32)  # [0, 1]
    sD_np = 0.3 * D
    # lens.veil expects (H, W, 3) and returns (H, W, 3)
    lens_out = lens.veil(w_np, D, 0.3)  # (H, W, 3)
    # Torch path: (1, 3, H, W) -> (1, 1, H, W)
    w_t = torch.from_numpy(w_np).transpose(0, 2).transpose(1, 2).unsqueeze(0).to(device)  # (1, 3, H, W)
    sD_t = torch.from_numpy(sD_np).transpose(0, 2).transpose(1, 2).unsqueeze(0).to(device)  # (1, 1, H, W)
    out = veil(w_t, sD_t).cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, 3)
    np.testing.assert_allclose(out, lens_out, rtol=0, atol=1e-4)

    # Non-divisible: H=120, W=150 (area resize differs at fractional steps)
    H2, W2 = 120, 150
    D2 = lens.dirt_maps(H2, W2, 0.5, np.random.default_rng(2))[1]
    w_np2 = rng.random((H2, W2, 3)).astype(np.float32)
    lens_out2 = lens.veil(w_np2, D2, 0.3)
    w_t2 = torch.from_numpy(w_np2).transpose(0, 2).transpose(1, 2).unsqueeze(0).to(device)
    sD_t2 = torch.from_numpy(0.3 * D2).transpose(0, 2).transpose(1, 2).unsqueeze(0).to(device)
    out2 = veil(w_t2, sD_t2).cpu().numpy()[0].transpose(1, 2, 0)
    # Area resize produces different results at fractional step sizes
    np.testing.assert_allclose(out2, lens_out2, rtol=0, atol=5e-3)


def test_blur_batch_independent() -> None:
    """Two different images in one batch give the same result as blurring each alone."""
    w1_np = rng.random((1, 3, 64, 80)).astype(np.float32)
    w2_np = rng.random((1, 3, 64, 80)).astype(np.float32)
    batch_np = np.concatenate([w1_np, w2_np], axis=0)  # (2, 3, 64, 80)
    w1_t = torch.from_numpy(w1_np).to(device)
    w2_t = torch.from_numpy(w2_np).to(device)
    batch_t = torch.from_numpy(batch_np).to(device)
    out1 = blur(w1_t, 2.0, border="reflect")
    out2 = blur(w2_t, 2.0, border="reflect")
    out_batch = blur(batch_t, 2.0, border="reflect")
    np.testing.assert_allclose(out_batch[0].cpu().numpy(), out1[0].cpu().numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(out_batch[1].cpu().numpy(), out2[0].cpu().numpy(), rtol=0, atol=1e-6)


def test_blur_small_image_matches_cv2() -> None:
    """Images smaller than the kernel radius (veil's 1/16 image: 7 x 9 px vs radius 8): OpenCV reflects repeatedly."""
    for h, w in ((7, 9), (3, 20), (1, 1)):
        img = rng.random((h, w, 3)).astype(np.float32)
        t = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        for border, cvb in (("reflect", cv2.BORDER_REFLECT), ("reflect101", cv2.BORDER_REFLECT_101)):
            if border == "reflect101" and min(h, w) == 1:
                continue  # OpenCV itself treats 1-px reflect101 as replicate; not a case veil can hit at H, W >= 16
            ref = cv2.GaussianBlur(img, (0, 0), 2.0, borderType=cvb)
            out = blur(t, 2.0, border)[0].permute(1, 2, 0).cpu().numpy()
            np.testing.assert_allclose(out, ref, rtol=0, atol=2e-5)
