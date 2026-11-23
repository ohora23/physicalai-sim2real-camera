# G1 Humanoid Head Camera Demo

Mount a `head_cam` at the D435i position on the head of the Unitree G1 (mujoco_menagerie) and quantify the effects of the ideal camera and the realistic (D435i-like) camera (motion blur, rolling shutter, noise, vignette, latency) along a walking-like trajectory.

## Fetch assets

```bash
bash scripts/fetch_menagerie.sh   # third_party/mujoco_menagerie/{unitree_g1,realsense_d435i} (gitignored, safe to re-run)
```

## Run

```bash
MUJOCO_GL=egl .venv/bin/python examples/g1_walk.py --seconds 5 --preset d435i_like --fast --out out/g1
MUJOCO_GL=egl .venv/bin/python examples/g1_walk.py --seconds 5 --preset ideal --out out/g1_ideal
MUJOCO_GL=egl .venv/bin/python tests/test_g1.py
```

Options: `--ctrl-hz 500`, `--fast` (rs_bands 2, blur_samples 2), `--bayer`, `--v-fwd 0.8`, `--a-z 0.02`,
`--osc 1.0` (multiplier for all angular and gait oscillations). Without `--fast`, d435i_like is N·B=32 → a WARNING is printed for 13 renders per frame.

## Note: the base motion is prescribed

The pelvis is welded to a mocap body and its pose is driven analytically (forward velocity, vertical bob at 1.8 Hz,
pitch ±2°@f, roll ±2°/yaw ±3°@f/2). The joints track a sinusoid around `knees_bent` via the menagerie position servo
(kp 500, dampratio 1; `ctrl = q_ref`). Foot-floor contact is off (it would conflict with the prescribed base).
In other words, this is **a device for producing a realistic camera trajectory, not emergent gait**.

## Output (`--out` directory)

| File | Contents |
|---|---|
| `frames_ideal_XXXX.png`, `frames_real_XXXX.png` | 6 evenly spaced frames (XXXX = frame index n) |
| `strip.png` | 6 rows × (ideal \| real), half resolution |
| `per_step.csv` | Per control step: `step, t, n, capture_t, age` (= t − capture_t, age of the frame the policy sees), `pelvis_z, target_z` |
| `per_frame.csv` | Per capture frame (below) |
| `summary.txt` | Number of delivered/dropped frames, age mean/p95 (ms), predicted blur and RS skew mean/max (px), Laplacian variance ratio, pelvis height error, runtime, ms/frame |

`per_frame.csv` columns:

- `n, capture_t` — frame index, exposure-center time
- `speed_v, v_perp` — camera linear speed magnitude / component perpendicular to the optical axis (m/s, finite difference over the exposure window)
- `speed_w` — camera angular speed magnitude (rad/s)
- `Z` — image-center depth (m, ideal depth)
- `blur_px` — predicted blur length `f_px·(v_perp/Z + ω)·exposure_s`
- `rs_px` — predicted rolling-shutter skew, same formula with `readout_s`
- `lap_ideal, lap_real, lap_ratio` — Laplacian variance (sharpness) and the real/ideal ratio (noise raises it, blur lowers it)
- `mean_ideal, mean_real` — mean brightness (gray 0–255)
- `ae_ev` — AE ev used for this frame
- `delivered` — whether it was actually delivered to the policy (1)

## Virtual preset values

**All values are virtual (VIRTUAL).** The real values are measured via Phase 3 calibration (`scripts/calibrate.py`,
`docs/capture_protocol.md`) and loaded via `CameraConfig.load`.

| Field | ideal | d435i_like |
|---|---|---|
| Resolution / fovy | 640×480 / 42° | 640×480 / 42° (model camera) |
| dist (k1,k2,p1,p2,k3) | 0 | (−0.05, 0.01, 0.0005, −0.0005, 0) |
| vignette / gamma / exposure_ev | 0 / 2.2 / 0 | 0.4 / 2.2 / 0 |
| sigma_shot / read / prnu / dsnu | 0 | 0.02 / 0.004 / 0.01 / 0.003 |
| black_level | 0 | 0.01 |
| fps / latency_s | 30 / 0 | 30 / 0.06 |
| jitter_k / jitter_theta / drop_prob | 0 | 2 / 0.003 / 0.02 |
| exposure_s / readout_s | 0 | 0.008 / 0.020 |
| blur_samples / rs_bands | – | 4 / 8 (`--fast`: 2 / 2) |
| ae_tau_s / ae_target | 0 | 0.5 / 0.18 |
| bayer | False | False (`--bayer`) |
