"""C0b: synthetic "real" calibration data (D1-D5) from a known CameraConfig.

Pure functions; every generator is deterministic given ``rng`` (np.random.Generator).
Loop-closure tests (C1-C6) fit the config back from these and compare to the
returned truth.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import mujoco
import numpy as np
from test_camera import BX, BY, NX, NY, SQ, TOP, board_scene

from physicalai_sim2real_camera import SimCamera
from physicalai_sim2real_camera.config import scale_intrinsics
from physicalai_sim2real_camera.optics import vignette_map
from physicalai_sim2real_camera.sensor import add_noise, encode, make_fpn
from physicalai_sim2real_camera.timing import schedule

PATTERN = (NX - 1, NY - 1)  # (7, 5) inner corners


def _look_quat(pos, target, roll):
    """MuJoCo camera quat looking from pos at target (camera -z forward), rolled."""
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def synth_geometry(cfg, rng, n_views=20):
    """D1: checkerboard views with distortion only (no noise/FPN/vignette/exposure).

    Returns ``(images, pattern=(7,5), square_m=0.05, K_true, dist_true)``; only
    views where ``findChessboardCornersSB`` finds all 35 corners are kept.
    """
    model, data = board_scene()
    cam = SimCamera(model, "cam", replace(cfg, sigma_shot=0, sigma_read=0, sigma_prnu=0,
                                          sigma_dsnu=0, vignette=0, black_level=0, exposure_ev=0))
    cid, centre = cam.cam, np.array([BX, BY, TOP])
    images = []
    try:
        for _ in range(50 * n_views):
            if len(images) == n_views:
                break
            d = rng.uniform(0.30, 0.55)
            ax, ay = np.radians(rng.uniform(-25, 25, 2))
            dirv = np.array([math.sin(ay), -math.sin(ax) * math.cos(ay), math.cos(ax) * math.cos(ay)])
            target = centre + [*rng.uniform(-0.03, 0.03, 2), 0.0]
            pos = target + d * dirv
            model.cam_pos[cid] = pos  # static worldbody camera: pos/quat are world pose
            model.cam_quat[cid] = _look_quat(pos, target, np.radians(rng.uniform(-10, 10)))
            mujoco.mj_forward(model, data)
            img = cam.render(data)
            ok, _ = cv2.findChessboardCornersSB(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), PATTERN)
            if ok:
                images.append(img)
        else:
            raise RuntimeError(f"only {len(images)}/{n_views} detectable views")
    finally:
        cam.close()
    return images, PATTERN, SQ, cam.K_out.copy(), tuple(cfg.dist)


def default_K(cfg):
    """K used by synth_flat when none is given: cfg fx/fy/cx/cy, else board_scene's fovy=45 pinhole."""
    f = (cfg.height / 2) / math.tan(math.radians(22.5))
    return np.array([[cfg.fx or f, 0, (cfg.width - 1) / 2 if cfg.cx is None else cfg.cx],
                     [0, cfg.fy or f, (cfg.height - 1) / 2 if cfg.cy is None else cfg.cy],
                     [0, 0, 1.0]])


def synth_flat(cfg, rng, level, n_frames, K=None):
    """D2 (level>0) / D3 (level=0): uniform linear input through the sensor pipeline.

    ``I = level·V·2^ev·g + o + black_level`` → shot+read noise → gamma encode.
    FPN uses ``default_rng([cfg.seed, 0])`` (= SimCamera cam_id 0).
    Returns ``(stack uint8 (n,H,W,3), truth)``.
    """
    W, H = cfg.width, cfg.height
    cv2.setRNGSeed(int(rng.integers(2**31)))
    K = default_K(cfg) if K is None else K
    V = vignette_map(K, cfg.dist, W, H, cfg.vignette) if cfg.vignette > 0 and level > 0 else None
    g, o = make_fpn(H, W, cfg.sigma_prnu, cfg.sigma_dsnu, np.random.default_rng([cfg.seed, 0]))
    gain = np.float32(level * 2.0 ** cfg.exposure_ev) * g
    if V is not None:
        gain *= V
    base = np.repeat(gain + o + np.float32(cfg.black_level), 3, axis=2)
    work, buf = np.empty_like(base), np.empty((H, W * 3), np.float32)
    stack = np.empty((n_frames, H, W, 3), np.uint8)
    for i in range(n_frames):
        np.copyto(work, base)
        add_noise(work, cfg.sigma_shot, cfg.sigma_read, buf)
        # ponytail: encode's astype allocs 0.9 MB/frame; reuse keeps bit-identity with SimCamera
        stack[i] = encode(work, cfg.gamma)
    truth = dict(g=g, o=o, vignette_map=V, K=K, black_level=cfg.black_level,
                 sigma_shot=cfg.sigma_shot, sigma_read=cfg.sigma_read,
                 sigma_prnu=cfg.sigma_prnu, sigma_dsnu=cfg.sigma_dsnu,
                 vignette=cfg.vignette, gamma=cfg.gamma, exposure_ev=cfg.exposure_ev)
    return stack, truth


def synth_dark(cfg, rng, n_frames):
    """D3: lens-capped frames (level 0, no vignette), same FPN/noise/black level."""
    return synth_flat(cfg, rng, 0.0, n_frames)


def synth_chart(cfg, reflectances, illum=1.0):
    """D4: noiseless mean DN of uniform grey patches, ``encode(r·illum·2^ev + black)``."""
    lin = np.asarray(reflectances, np.float32) * np.float32(illum * 2.0 ** cfg.exposure_ev)
    lin += np.float32(cfg.black_level)
    return encode(lin, cfg.gamma).astype(np.float64)


def synth_timing(cfg, rng, n_toggles=300, t_end=None):
    """D5a: LED toggles + frame availability on one clock.

    Returns dict with, per surviving (non-dropped) frame: ``avail_t`` (= t_c + latency),
    ``frame_seq`` (capture index n; gaps = drops), ``event_seen`` (# toggles with
    event_t <= t_c(n)); plus ``event_t`` and truth ``t_c``, ``latency``, ``n_frames``.
    """
    event_t = cfg.phase_s + np.cumsum(rng.uniform(0.3, 0.7, n_toggles))
    if t_end is None:
        t_end = event_t[-1] + 1.0
    n_frames = int(math.ceil((t_end - cfg.phase_s) * cfg.fps))
    t_c, lat, dropped = schedule(cfg, n_frames, rng)
    keep = ~dropped
    t_c, lat = t_c[keep], lat[keep]
    return dict(event_t=event_t, avail_t=t_c + lat, frame_seq=np.flatnonzero(keep),
                event_seen=np.searchsorted(event_t, t_c, side="right"),
                t_c=t_c, latency=lat, n_frames=n_frames)


def write_calib_data(cfg, root, rng, n_views=20, levels=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8),
                     n_frames=100, reflectances=(0.03, 0.09, 0.19, 0.36, 0.59, 0.9),
                     n_toggles=300, flat_size=None):
    """C6: write a full D1-D5 ``calib_data`` tree (docs/capture_protocol.md layout) under ``root``.

    PNGs go through ``cv2.imwrite`` in BGR order; scripts/calibrate.py reads them with
    cv2 and converts back to RGB, so channels round-trip. ``flat_size=(w, h)`` renders
    D2/D3 at a smaller resolution (K scaled by w/W, h/H with pixel-centre offsets; the
    CLI rescales the D1 K the same way) to keep the stacks small. D4 = one uniform 16x16
    image per patch (noiseless chart DN, illum=1) + ``reflectances.csv``. D5 = one
    ``events.csv``: toggle rows fill only ``event_t``; frame rows leave it empty.
    Returns truth: cfg fields + ``K``, ``dist``, ``K_flat``, FPN maps ``g``/``o`` (D2 res).
    """
    import dataclasses
    import os

    def put(path, img):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        assert cv2.imwrite(path, np.ascontiguousarray(img[..., ::-1]))

    images, _, _, K, dist = synth_geometry(cfg, rng, n_views)
    for i, img in enumerate(images):
        put(f"{root}/D1/view_{i:03d}.png", img)

    w, h = flat_size or (cfg.width, cfg.height)
    sx, sy = w / cfg.width, h / cfg.height
    K_flat = scale_intrinsics(K, sx, sy)
    small = replace(cfg, width=w, height=h, fx=K_flat[0, 0], fy=K_flat[1, 1],
                    cx=K_flat[0, 2], cy=K_flat[1, 2])
    for k, lv in enumerate(levels):
        stack, truth = synth_flat(small, rng, lv, n_frames, K=K_flat)
        for i, f in enumerate(stack):
            put(f"{root}/D2/level_{k}/{i:04d}.png", f)
    for i, f in enumerate(synth_dark(small, rng, n_frames)[0]):
        put(f"{root}/D3/{i:04d}.png", f)

    for i, dn in enumerate(synth_chart(cfg, reflectances)):
        put(f"{root}/D4/patch_{i:02d}.png", np.full((16, 16, 3), dn, np.uint8))
    with open(f"{root}/D4/reflectances.csv", "w") as f:
        f.writelines(f"{r}\n" for r in reflectances)

    d = synth_timing(cfg, rng, n_toggles)
    os.makedirs(f"{root}/D5", exist_ok=True)
    with open(f"{root}/D5/events.csv", "w") as f:
        f.write("event_t,avail_t,frame_seq,event_seen\n")
        f.writelines(f"{t:.9f},,,\n" for t in d["event_t"])
        f.writelines(f",{a:.9f},{n},{s}\n" for a, n, s in zip(d["avail_t"], d["frame_seq"], d["event_seen"]))

    return dict(dataclasses.asdict(cfg), K=K, dist=dist, K_flat=K_flat, g=truth["g"], o=truth["o"])
