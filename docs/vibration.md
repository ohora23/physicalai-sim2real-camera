# Camera-mount vibration

A rigid-body simulator already moves the camera with the robot body (gait bob, sway, turns), and the window render
turns that motion into motion blur and rolling-shutter skew. This module adds an optional **mount flex**: a head
bracket, neck or housing that vibrates at its own natural frequency, applied to every sub-render so that motion blur
and the rolling shutter see the same time-continuous motion.

## Model

`physicalai_sim2real_camera/vibration.py`: each of six axes (rotation vector and translation, MuJoCo camera frame)
is a stationary second-order resonant process,

    x'' + 2 zeta w x' + w^2 x = s n(t),   w = 2 pi hz,   s = RMS * sqrt(4 zeta w^3)

integrated at 1 kHz from its stationary distribution and linearly interpolated, so the offset is a deterministic
function of time per seed. Every sub-render of a window (exposure samples, rolling-shutter bands) uses the offset of
its own time.

| field | meaning | default | placeholder for a head bracket |
|---|---|---|---|
| `mount_vib_rot_deg` | RMS rotation per axis, deg | 0 (off) | 0.15 (placeholder) |
| `mount_vib_trans_m` | RMS translation per axis, m | 0 (off) | 0.0003 (placeholder) |
| `mount_vib_hz` | natural frequency, Hz | 12 | 12 |
| `mount_vib_zeta` | damping ratio | 0.1 | 0.1 |

Mount vibration is **opt-in**: no preset turns it on, `calibrated` included. Turn it on with
`CameraConfig.preset("calibrated", W, H, mount_vib_rot_deg=0.15, mount_vib_trans_m=0.0003)` (or
`dataclasses.replace`). The values in the last column are placeholders, not a measurement; measure them with an IMU on
the camera (the PSD peak gives `hz`, its width `zeta`, the integral the RMS).

## Where it applies

- `SimCamera` (CPU and CUDA) and `SimCameraBatch` move the rendered camera (`mjvScene` eyes) after `update_scene`;
  camera i of a batch has the vibration of `SimCamera(cfg.seed = seed_base + i)` (bit-equal, `tests/test_batch.py`).
- `GenesisCamera` multiplies the mount offset into the margin camera's attachment before every sub-render. Genesis
  attaches one offset for the whole batch, so all envs of a batched Genesis scene share one vibration.
- The ideal camera (`render_ideal`) stays rigid.
- Per-episode mount offsets (`perturb_extrinsics`) and the vibration combine: the first moves the mean pose, the
  second shakes around it.

## Tests

`tests/test_vibration.py`: per-axis RMS within 12 %, spectral peak at the natural frequency (12 Hz and 4 Hz),
determinism and query-order independence, smooth 1 ms steps across chunk seams, and in MuJoCo and Genesis a red
sphere rendered at several times lands where the vibrating camera projects it (within 0.14 px, while it moves
by more than 30 px), and the config validation.
