"""Camera configuration and intrinsics helpers."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

# What an intrinsic / extrinsic calibration does not measure, at typical values of a consumer RGB-D camera
# (D435i-class: auto exposure, rolling-shutter RGB, USB transport). Placeholders, not a measured camera:
# ``calib.randomize`` around them, and replace any field ``scripts/calibrate.py`` fits. Published anchors:
# - latency_s: ~66 ms mean colour latency (rs-latency-tool, realsense-ros issue #2686); host and USB dependent.
# - readout_s: the RGB sensor (OV2740, rolling shutter) reads 1080p at up to 60 fps, so a frame reads out in <= 16 ms.
# - depth_sigma_b: Intel's depth RMS = z^2 * subpixel / (f * B); subpixel 0.08 (well calibrated: < 0.1), baseline
#   B = 50 mm, f = 447 px at 848x480 (87 deg HFOV) -> 0.0036 / m.
# Noise, vignetting, AE, jitter and drops have no published per-model figure (drops come mostly from USB
# bandwidth). Mount vibration is not in the preset: it is opt-in (docs/vibration.md); measure it on your robot
# (IMU PSD) before turning it on.
TYPICAL_UNCALIBRATED = dict(
    vignette=0.4, gamma=2.2, black_level=0.01, ae_tau_s=0.5, ae_target=0.18,             # optics, response, AE
    sigma_shot=0.02, sigma_read=0.004, sigma_prnu=0.01, sigma_dsnu=0.003,                  # sensor noise
    exposure_s=0.008, readout_s=0.016,                                                     # blur, rolling shutter
    fps=30.0, latency_s=0.066, jitter_k=2.0, jitter_theta=0.003, drop_prob=0.02,           # timing, drops
    depth_sigma_a=0.001, depth_sigma_b=0.0036, depth_min_m=0.2, depth_max_m=10.0, depth_edge_hole_px=2)  # depth
PRESETS = {"ideal": {}, "calibrated": TYPICAL_UNCALIBRATED}


@dataclass
class CameraConfig:
    """Realistic camera model for MuJoCo sensor simulation.

    Fields control distortion, noise, vignetting, timing, and sample
    variance.  ``exposure_ev`` adjusts brightness only in the shadows:
    the renderer outputs clipped uint8, so highlights that saturated to
    255 cannot be recovered by lowering exposure.
    """

    width: int  # required
    height: int  # required
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)  # OpenCV (k1, k2, p1, p2, k3) or rational (+ k4, k5, k6)
    vignette: float = 0.0
    gamma: float = 2.2
    exposure_ev: float = 0.0
    sigma_shot: float = 0.0
    sigma_read: float = 0.0
    sigma_prnu: float = 0.0
    sigma_dsnu: float = 0.0
    black_level: float = 0.0
    fps: float = 30.0
    phase_s: float = 0.0
    latency_s: float = 0.0
    jitter_k: float = 0.0
    jitter_theta: float = 0.0
    drop_prob: float = 0.0
    seed: int = 0
    # Phase 2 (all identity by default)
    exposure_s: float = 0.0
    readout_s: float = 0.0
    blur_samples: int = 4
    rs_bands: int = 8
    focus_m: float = 0.0
    aperture_m: float = 0.0
    ae_tau_s: float = 0.0
    ae_target: float = 0.18
    awb_tau_s: float = 0.0
    bayer: bool = False
    depth_sigma_a: float = 0.0
    depth_sigma_b: float = 0.0
    depth_min_m: float = 0.0
    depth_max_m: float = 0.0
    depth_edge_hole_px: int = 0
    # camera-side DR (all identity by default; MJPEG is CPU SimCamera only, the rest also cuda / batch)
    ccm: tuple[float, ...] | None = None  # row-major 3x3 linear-RGB colour matrix (sensor response)
    flicker_amp: float = 0.0              # mains light modulation depth, illumination 1 + amp*cos(2 pi hz t + phase)
    flicker_hz: float = 100.0             # twice the mains frequency (50 Hz grid -> 100, 60 Hz -> 120)
    flicker_phase: float = 0.0            # rad
    dirt_density: float = 0.0             # lens dust/smudges (pattern drawn from seed); 0 = clean lens
    dirt_scatter: float = 0.0             # veiling glare: fraction of low-passed light scattered by the dirt
    jpeg_quality: int = 0                 # MJPEG transport quality 1..100; 0 = raw
    psf_sigma_px: float = 0.0             # lens PSF / OLPF / pixel aperture as one Gaussian (output px); 0 = sharp
    render_scale: int = 1                 # supersample: render s x s per pixel, area-average in linear light
    # camera-mount vibration (vibration.MountVibration; off while both RMS values are 0): bracket / neck flex that a
    # rigid-body model lacks, a resonant random process applied to every sub-render (blur and rolling shutter see it)
    mount_vib_rot_deg: float = 0.0        # RMS per axis, deg (camera frame)
    mount_vib_trans_m: float = 0.0        # RMS per axis, m (camera frame)
    mount_vib_hz: float = 12.0            # natural frequency of the mount, Hz
    mount_vib_zeta: float = 0.1           # damping ratio
    # (H, W, 1) float32 gain map; compare=False: ndarray == is ambiguous in dataclass __eq__
    flat_field: np.ndarray | None = field(default=None, compare=False)

    def lens_fields(self) -> list[str]:
        """Names of the camera-side DR fields set on this config (ccm, flicker, lens dirt, MJPEG, PSF)."""
        on = dict(ccm=self.ccm is not None, flicker_amp=self.flicker_amp > 0, dirt_density=self.dirt_density > 0,
                  jpeg_quality=self.jpeg_quality > 0, psf_sigma_px=self.psf_sigma_px > 0)
        return [k for k, v in on.items() if v]

    def __post_init__(self):
        """Validate ``dist`` (4, 5 or 8 terms), ``flat_field`` (float32 ``(height, width, 1)``) and the DR fields."""
        if len(self.dist) not in (4, 5, 8):
            raise ValueError(f"dist must have 4, 5 or 8 terms (OpenCV k1, k2, p1, p2[, k3[, k4, k5, k6]]), got "
                             f"{len(self.dist)}: {tuple(self.dist)}")
        if self.ccm is not None:  # any 9-element / 3x3 input -> tuple, so a save/load round trip compares equal
            ccm = np.ravel(np.asarray(self.ccm, np.float64))
            if ccm.size != 9:
                raise ValueError(f"ccm must be 3x3 (9 entries, row-major), got shape {np.shape(self.ccm)}")
            self.ccm = tuple(float(x) for x in ccm)
        if not 0 <= self.flicker_amp <= 1:
            raise ValueError(f"flicker_amp must be in [0, 1], got {self.flicker_amp}")
        if self.jpeg_quality != 0 and not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be 0 (off) or in [1, 100], got {self.jpeg_quality}")
        if (isinstance(self.render_scale, bool) or not isinstance(self.render_scale, (int, np.integer))
                or not 1 <= self.render_scale <= 8):
            raise ValueError(f"render_scale must be an int in [1, 8], got {self.render_scale!r}")
        if self.psf_sigma_px < 0:
            raise ValueError(f"psf_sigma_px must be >= 0, got {self.psf_sigma_px}")
        if min(self.mount_vib_rot_deg, self.mount_vib_trans_m) < 0 or min(self.mount_vib_hz, self.mount_vib_zeta) <= 0:
            raise ValueError("mount_vib_rot_deg / mount_vib_trans_m must be >= 0, mount_vib_hz / mount_vib_zeta > 0")
        if self.dirt_density < 0 or not 0 <= self.dirt_scatter <= 1:
            raise ValueError("dirt_density must be >= 0 and dirt_scatter in [0, 1]")
        if self.flat_field is not None:
            want = (self.height, self.width, 1)
            if self.flat_field.dtype != np.float32 or self.flat_field.shape != want:
                raise ValueError(
                    f"flat_field must be float32 shape {want}, got dtype={self.flat_field.dtype} "
                    f"shape={self.flat_field.shape}")

    def save(self, path):
        """Write this config to `path` as JSON. Returns `path`.

        ``flat_field`` goes to ``<path>.flat_field.npy``; the json stores its
        basename (resolved relative to the json's directory on load).
        """
        d = dataclasses.asdict(self)
        if self.flat_field is not None:
            npy = os.fspath(path) + ".flat_field.npy"
            np.save(npy, self.flat_field)
            d["flat_field"] = os.path.basename(npy)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return path

    @classmethod
    def load(cls, path) -> "CameraConfig":
        """Read a config back from JSON, restoring `dist` as a tuple of floats (5 or 8 terms)."""
        with open(path) as f:
            data = json.load(f)
        data["dist"] = tuple(float(x) for x in data["dist"])
        if data.get("ccm") is not None:
            data["ccm"] = tuple(float(x) for x in data["ccm"])
        if data.get("flat_field"):
            data["flat_field"] = np.load(os.path.join(os.path.dirname(os.fspath(path)), data["flat_field"]))
        return cls(**data)


    @classmethod
    def preset(cls, name, width, height, K=None, dist=None, **overrides) -> "CameraConfig":
        """A config for a common setup (``PRESETS``).

        ``"calibrated"`` is the usual robot setup: intrinsics (and extrinsics, ``set_extrinsics``) calibrated, so the
        geometry is the measured one, while everything a calibration does not reach is on at typical values
        (``TYPICAL_UNCALIBRATED``: sensor noise, vignetting, auto exposure, motion blur, rolling shutter, latency,
        jitter, drops, depth noise / holes / range), so the policy and the frame handling meet them in simulation.
        ``"ideal"`` is the identity pipeline.

        Args:
            name: ``"calibrated"`` or ``"ideal"``.
            width: Output width, px.
            height: Output height, px.
            K: Calibrated 3x3 intrinsics at ``width x height`` (for rectified frames the rectified / new camera
                matrix, e.g. ROS ``P[:3, :3]``); None keeps the simulator camera's K. Rescale a calibration made at
                another resolution with ``scale_intrinsics``.
            dist: OpenCV distortion ``(k1, k2, p1, p2[, k3[, k4, k5, k6]])`` (the rational terms k4-k6 cover
                wide-angle factory calibrations): pass it when the policy sees raw frames. None means
                rectified frames (``cv2.undistort`` / ROS ``image_proc``): no distortion.
            **overrides: Any ``CameraConfig`` field, applied last (e.g. measured ``latency_s``).

        Returns:
            The ``CameraConfig``.

        Raises:
            ValueError: Unknown ``name``, ``K`` not 3x3, or ``dist`` with fewer than 4 terms or non-zero terms past
                k6 (thin-prism / tilt models are not modelled).
        """
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        kw = dict(PRESETS[name])
        if K is not None:
            K = np.asarray(K, np.float64)
            if K.shape != (3, 3):
                raise ValueError(f"K must be 3x3, got shape {K.shape}")
            kw.update(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))
        if dist is not None:
            d = np.ravel(np.asarray(dist, np.float64))
            if d.size < 4 or np.any(d[8:]):
                raise ValueError(f"dist must be OpenCV (k1, k2, p1, p2[, k3[, k4, k5, k6]]), got {d.size} terms "
                                 f"{d.tolist()}")
            n = 8 if np.any(d[5:8]) else 5  # keep the 5-term form unless the rational terms are used
            kw["dist"] = tuple(float(x) for x in np.pad(d[:n], (0, n - min(d.size, n))))
        return cls(width=width, height=height, **{**kw, **overrides})

def intrinsics_from_model(model, cam_id, W, H) -> np.ndarray:
    """Build a 3×3 intrinsic matrix from MuJoCo model data.

    Uses *sensorsize + focal* when available, falls back to *fovy* otherwise.
    """
    sensorsize = model.cam_sensorsize[cam_id]
    sensor_w, sensor_h = sensorsize[0], sensorsize[1]

    if sensor_w == 0.0 and sensor_h == 0.0:
        # fovy-based camera
        fovy_deg = model.cam_fovy[cam_id]
        fy = (H / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        fx = fy
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
    else:
        # sensorsize-based camera (units are metres)
        intrinsic = model.cam_intrinsic[cam_id]
        focal_x, focal_y, principal_x, principal_y = intrinsic

        fx = focal_x / sensor_w * W
        fy = focal_y / sensor_h * H
        cx = (W - 1) / 2.0 - principal_x / sensor_w * W
        cy = (H - 1) / 2.0 - principal_y / sensor_h * H

    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_intrinsics(K, sx, sy) -> np.ndarray:
    """Rescale intrinsics `K` for an image resized by (`sx`, `sy`).

    Keeps pixel centres aligned: focal lengths scale directly, principal point
    maps as (c + 0.5)*s - 0.5.
    """
    K = np.asarray(K, np.float64)
    return np.array([[K[0, 0] * sx, 0.0, (K[0, 2] + 0.5) * sx - 0.5],
                     [0.0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def set_extrinsics(model, cam_name, T_body_cam, optical=True):
    """Put MJCF camera ``cam_name`` at a calibrated extrinsic (writes ``model.cam_pos`` / ``cam_quat`` in place).

    Build ``SimCamera`` / ``SimCameraBatch`` after this and ``mj_forward`` the data; for Genesis pass
    ``GenesisCamera(extrinsics=...)``.

    Args:
        model: ``mujoco.MjModel``.
        cam_name: Fixed-mode MJCF camera.
        T_body_cam: ``(4, 4)`` camera pose in the frame of the body the camera is attached to (its MJCF parent,
            ``model.cam_bodyid``), metres.
        optical: True when the camera axes of ``T_body_cam`` are the OpenCV / ROS optical frame (x right, y down,
            z forward) that calibration tools report; False for MuJoCo's camera frame (x right, y up, looks -z).

    Raises:
        ValueError: Unknown or non-fixed camera, or ``T_body_cam`` not 4x4.
    """
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"camera {cam_name!r} not found")
    if model.cam_mode[cid] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:  # tracking modes recompute the pose every step
        raise ValueError(f"camera {cam_name!r} is not mode='fixed'")
    T = np.asarray(T_body_cam, np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_body_cam must be 4x4, got shape {T.shape}")
    R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0]) if optical else T[:3, :3]
    model.cam_pos[cid] = T[:3, 3]
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    model.cam_quat[cid] = q


def perturb_extrinsics(T_body_cam, rng, rot_deg=0.5, trans_m=0.003):
    """Draw a camera mount pose around a calibrated one (per-episode domain randomisation).

    Models what an extrinsic calibration leaves: its residual error and the mount's flex and settling, which differ
    between units and over time. The rotation turns about a uniformly random axis by an angle drawn uniformly in
    ``[0, rot_deg]``; the offset is uniform in ``[-trans_m, trans_m]`` per axis; both act in the camera frame. The
    defaults are placeholders, not a measurement.

    Args:
        T_body_cam: ``(4, 4)`` calibrated pose, as passed to ``set_extrinsics`` (OpenCV optical axes).
        rng: ``numpy.random.Generator``.
        rot_deg: Largest rotation, degrees.
        trans_m: Largest offset per axis, metres.

    Returns:
        A new ``(4, 4)`` float64 pose; ``T_body_cam`` is left unchanged.

    Raises:
        ValueError: ``T_body_cam`` not 4x4, or a negative ``rot_deg`` / ``trans_m``.
    """
    T = np.asarray(T_body_cam, np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_body_cam must be 4x4, got shape {T.shape}")
    if rot_deg < 0 or trans_m < 0:
        raise ValueError(f"rot_deg and trans_m must be >= 0, got {rot_deg}, {trans_m}")
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = math.radians(rot_deg) * rng.uniform()
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    D = np.eye(4)
    D[:3, :3] = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k  # Rodrigues
    D[:3, 3] = rng.uniform(-trans_m, trans_m, 3)
    return T @ D
