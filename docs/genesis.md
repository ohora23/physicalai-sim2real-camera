# Genesis Backend: `GenesisCamera`

The camera in [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) outputs ideal renders only. It lacks noise, vignetting, blur, rolling shutter, and exposure models, and the camera sensor rejects `delay`/`jitter` options. `physicalai_sim2real_camera.genesis_bridge` runs the MuJoCo-side pipeline unchanged on top of Genesis renders. This document covers the structure, installation and usage, and the agreement with MuJoCo measured on WIRobotics ALLEX, Unitree G1 and Apptronik Apollo.

## Status and direction (2026-11-20)

**Development continues on MuJoCo. The Genesis backend is kept as it is.** New features go into the MuJoCo path
(`SimCamera`, `SimCameraBatch`) only; the Genesis backend is guarded by its tests and gets no new features. On 2026-11-20
all 17 tests in `tests/test_genesis_bridge.py` and `tests/test_genesis_apollo.py` passed (0 skipped) with Genesis 1.4.5,
mujoco 3.14 and torch 2.11+cu128 on an RTX 5080. They cover the camera mount, joint-state interpolation (quaternions
included), projection, the window time axis, batched-versus-single identity, the headlight and the Apollo example; they
do not cover mount vibration in Genesis (`_shake`).

An English white paper with figures, charts and videos summarises this document: [`docs/whitepaper/genesis/index.html`](whitepaper/genesis/index.html).

### How Genesis is supported

1. Genesis goes in its own venv, because it installs mujoco 4 and opencv-python ([Installation](#installation)).
2. The user builds and steps the physics scene as usual. `GenesisCamera(xml, cam, cfg, dt)` builds a separate, render-only
   mirror scene from the same MJCF and only reads `qpos` from the physics scene.
3. Frames come through the same `CameraStream` as in MuJoCo, so latency, jitter, drops and the age of the frame the policy
   holds are computed the same way.
4. For many environments use `n_envs` (rasterizer, each environment rendered separately) or `renderer="madrona"`
   (BatchRenderer, frames stay on the GPU).

### What this repository adds on top of Genesis

Genesis itself is not modified (upstream 1.4.5). Everything below is in this repository's `genesis_bridge.py` and in the
examples and checks.

| Component | Why it was needed (Genesis side) | Where |
|---|---|---|
| Image-pipeline bridge | Genesis cameras output the ideal render only, and the camera sensor rejects `delay` / `jitter` | `GenesisCamera` → `SimCameraBatch(renderer=False)` |
| Render-only mirror scene | `RigidSolver.set_qpos` changes later physics steps even when it rewrites the same values (ALLEX fingers 0.04 rad after 7 s) | `GenesisCamera.__init__`, `mirror` |
| Joint-state ring and interpolation (window mode) | Blur and rolling-shutter sub-frames fall on a 2 ms grid; physics steps are 5 ms | `push`, `interp_qpos` |
| Floating-base interpolation | Quaternions of free and ball joints must not be interpolated linearly | `quat_starts`, `slerp` |
| Camera mount resolution | Genesis merges joint-less bodies into their parent link | `camera_mount` |
| Depth background correction | A 24-bit depth buffer resolves z near the far plane coarsely, so the background could be classified as a surface | `zfar_of`, `_bg` |
| Batched environments | Many environments with different poses rendered from one scene | `n_envs` |
| Camera-following headlight | Genesis lights are fixed, so brightness drifts from MuJoCo's headlight as the head turns | `headlight` |
| Large-scale path | The rasterizer is slow at hundreds of environments | Madrona BatchRenderer, batched accumulation, fused LUT |
| Calibrated mount pose and per-episode randomisation | Real robots use a calibrated extrinsic | `extrinsics`, `set_extrinsics` |
| Mount vibration | Each sub-render needs its own mount pose so that vibration shows up as blur | margin camera re-attached per sub-render (`_shake`) |
| Validation scripts and examples | Agreement with MuJoCo has to be measured | ALLEX (`examples/genesis_allex.py`), G1 (`scripts/genesis_g1_check.py`), Apollo (`examples/genesis_apollo.py`) |

### Not supported in Genesis

- The headlight works only in unbatched rasterizer scenes (see the limitations under [Structure](#structure)).
- All batched environments capture at the same time (no per-environment `stagger_phases`).
- The mirror scene doubles render geometry and build time.

### If Genesis work resumes

1. After a Genesis upgrade, re-run the 17 tests above and `scripts/genesis_g1_check.py` and
   `scripts/genesis_allex_check.py`, and confirm that the agreement with MuJoCo (tables below) still holds.
2. If Genesis gains per-environment light nodes, or light poses that can change between renders in the BatchRenderer,
   `GenesisCamera._shot` can move the light per environment just before each render and lift the headlight restriction.
3. If `set_qpos` stops perturbing the physics state, render directly from the physics scene without a mirror scene; render
   geometry and build time then halve.

## Structure

```
physics scene (Genesis, stepped)          mirror scene (Genesis, never stepped)          physicalai-sim2real-camera
  entity.get_qpos() ──push(data)──▶ ring of (t, qpos) ──interp──▶ set_qpos ▶ margin render ──ingest──▶ SimCameraBatch
                                                                    (H_r x W_r @ fovy_r)            (renderer=False)
                                                                                                    ▶ process ▶ rgb, depth
```

- **The entry point is `SimCameraBatch(model, cam, cfg, 1, renderer=False).ingest`.** Since the MJCF `MjModel` is passed to the batch as-is, `K`, distortion margin, `zfar`, window planning (`base._plan`), and seed are identical to MuJoCo `SimCamera`.
- **It shares the same stream interface as `SimCamera`.** It has `push(data)`, `render(data, depth=False, as_tensor=False)`, and `cam`, allowing `CameraStream` to operate directly. `data` is an object containing `time` and the `qpos` of the physical entity.
- **The camera mounting position is determined by `camera_mount`.** Genesis' MJCF loader does not import `<camera>` elements (`genesis/utils/mjcf.py` has no camera handling), so the bridge adds its own cameras and attaches them. Genesis may merge a body without joints into its parent link, so we walk upward from the camera body to find remaining links in Genesis and calculate the offset using the MuJoCo `qpos0` world pose. Since both simulators’ cameras look in the −z direction and have +y as up, no axis transformation is needed.
- **Window mode (motion blur, rolling shutter) renders using interpolated qpos.** The sub-frame timestamps are on a 2 ms grid (`min(8 ms/4, 20 ms/8)`), which cannot align with the 5 ms physics step. Therefore, similar to MuJoCo `StateRing`, we store `(t, qpos)` at each step and linearly interpolate the qpos at the `capture_t + t_k` timestamp for rendering. Depth is rendered once at `capture_t`.
- **Rendering is performed on the mirror scene, not the physics scene.** Genesis’s `RigidSolver.set_qpos` alters subsequent physics steps even if the current qpos is written back unchanged. Measurements during development (with the earlier code, which rendered directly from the physics scene) showed that ALLEX’s finger joints differed by 0.04 rad after 7 seconds. This persisted even when restoring constraint warm start (`is_warmstart`, `qacc_ws`) or skipping forward kinematics with `skip_forward=True`. Thus, the camera is placed in a separate scene that is never stepped, while the physics scene only reads qpos. The trade-off is that the geometry and build time for rendering are doubled.
- **The depth background is aligned using `zfar_of`.** The pipeline determines the background as `z >= zfar·(1 − 1e-6)` (`zfar = zfar_of(model)`). A 24-bit depth buffer can only distinguish z values near the far plane in increments of approximately `far²/near × 2⁻²⁴`, causing the read background value to fluctuate above and below `zfar_of`. The pre-alignment values (measured during development) were 68.7800 m (Genesis) and 68.7818 m (MuJoCo) for ALLEX, both exceeding `zfar_of` of 68.7795 m, thus passing the background check. However, on the test rig, Genesis’s background was 54.8240 m, which is less than `zfar_of` of 54.8301 m, causing it to be treated as a real surface. Therefore, we replace the range within 8 times this precision at the far plane (16 cm for ALLEX, 13 cm for the rig) with the exact `zfar_of` before passing it.

Limitations (marked with `# ponytail:`):
- In batch mode, all envs capture at the same `data.time`. To stagger capture phases per env (`stagger_phases`), per-env streams are required.
- **The headlight works only in unbatched rasterizer scenes.** Providing `n_envs > 0` or `renderer="madrona"` raises a `ValueError`. The cause lies within Genesis.
  - Split env rendering in the rasterizer shares the scene’s light nodes across all envs. Since each env has a different camera pose, a single node cannot create lights that follow the camera per env.
  - Madrona (BatchRenderer) fixes lights during build. There is no API to change light poses during rendering.
  - Therefore, batch/Madrona scenes use fixed lights (`scene_lights(forward=False)`, `FLAT_MADRONA`). Without lights following the camera, the result of matching brightness with MuJoCo headlight (the `--headlight` row in the lighting table below) holds only for unbatched scenes.
  - To resolve this, Genesis must support per-env light nodes (split rendering) or the ability to change light poses between renders (BatchRenderer). If supported, `GenesisCamera._shot` would move lights to per-env poses just before rendering (similar to what is currently done in a single env).

## Installation

Genesis installs mujoco 4 and opencv-python together, so use a separate virtual environment. Revert mujoco to version 3.14.

```bash
uv venv -p 3.12 .venv-genesis
uv pip install -p .venv-genesis torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv-genesis genesis-world==1.4.5 mujoco==3.14.0 pytest pytest-cov
uv pip install -p .venv-genesis --no-deps -e .
scripts/fetch_allex.sh            # third_party/allex_model @ 5e12052 (mesh redistribution is prohibited: do not commit)
```

## Usage

```python
import types

import genesis as gs
from physicalai_sim2real_camera import CameraConfig, CameraStream
from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005))
robot = scene.add_entity(gs.morphs.MJCF(file=xml))
cam = GenesisCamera(xml, "realsense_color_optical_frame", cfg, dt=0.005)   # Builds its own mirror scene
scene.build()
stream = CameraStream(cam, cfg)
for _ in range(n):
    scene.step()
    frame, capture_t, k = stream.step(types.SimpleNamespace(time=scene.sim.cur_t, qpos=robot.get_qpos()))
```

ALLEX example (recorded-motion replay, saved captures, video, live window):

```bash
PY=.venv-genesis/bin/python
$PY examples/genesis_allex.py --sim genesis --cfg real --lights --seconds 42.7
$PY examples/genesis_allex.py --sim genesis --cfg real --lights --seconds 42.7 --video out/allex.mp4
$PY examples/genesis_allex.py --sim genesis --cfg real --lights --seconds 42.7 --live   # Genesis viewer + head camera window
```

`--cfg` selects one of four setups that switch the effects on step by step: `geom` (distortion only), `wingeom` (+ motion blur, rolling shutter), `static` (distortion + noise, vignetting, gamma; no AE and window), `real` (everything, including AE). The values are the made-up `d435i_like` values from `examples/g1_walk.py`.

## Agreement with MuJoCo, measured on ALLEX

The target is ALLEX’s `realsense_color_optical_frame` (fovy 42°, 640×480, margin render 656×494 @ 43.11°), and the motion is `demo1` (42.7 s, waist and neck rotating ±65° between 14–42 s). On the MuJoCo side, the playback control law published by ALLEX (position + velocity servo, gravity compensation feed-forward, `clip(PD + G, ±τ)`) is used. On the Genesis side, `control_dofs_position_velocity` with the same gains and gravity compensation 1 is used. Five boxes with distinct colors are placed in the scene to obtain silhouettes independent of lighting. RTX 5080, GPU not shared with other tasks.

Reproduction:

```bash
for c in geom wingeom real; do for s in mujoco genesis; do $PY examples/genesis_allex.py --sim $s --cfg $c --seconds 42.7; done; done
$PY examples/genesis_allex.py --sim genesis --cfg real --lights --seconds 42.7
$PY examples/genesis_allex.py --sim genesis --cfg none --seconds 42.7          # Physics baseline without camera
for c in geom static; do for s in mujoco genesis; do CFG=$c $PY scripts/genesis_allex_check.py checker-render $s; done; done
$PY scripts/genesis_allex_check.py physics; $PY scripts/genesis_allex_check.py compare; $PY scripts/genesis_allex_check.py lights
for c in geom static; do CFG=$c $PY scripts/genesis_allex_check.py checker; done
```

### Physics

| Item | Result |
|---|---|
| Model loading | The Genesis MJCF loader reads 60 DoFs and 12 polynomial `equality joint`s for fingers and waist exactly as defined |
| `demo1` joint trajectory, Genesis vs MuJoCo RMS (`physics`) | Waist/neck avg 0.0004, max 0.0012 rad / Arms avg 0.0036, max 0.0128 rad / Hands avg 0.0127, max 0.0656 rad (`R_Thumb_MCP`) |
| Genesis trajectory with window camera vs without camera | Max difference 0 rad (mirror scene) |
| Torque limits | Genesis gravity compensation is outside the torque clip, so it differs from `clip(PD + G, ±τ)` only in steps where torque saturates |

### Camera

| Item | MuJoCo | Genesis | Notes |
|---|---|---|---|
| Timing (42.7 s) | 1252 frames, 27 gaps, avg age 68.4 ms, p99 84 ms | Same | `CameraStream` with same seed |
| Window output silhouette IoU (`wingeom`) | — | Median 0.988, rotation segment 0.987, rotation segment p10 0.974 | |
| Blur level `1 − gradient energy(window)/gradient energy(GS)` | 0.095 | 0.114 | 83 turning frames with visible blur; difference 20.7 %, per-frame correlation 0.79 |
| Static segment window = global shutter (IoU) | Median 1.000, min 0.984 | Median 1.000, min 0.991 | |
| Depth, box pixels | — | Difference median 0.15 mm, worst frame 2.28 mm | Background: MuJoCo 68.7818 m, Genesis 68.7795 m (aligned via `zfar_of`). Both are classified as background |

The 20.7 % difference in blur level is due to renderer shading differences (Genesis PBR vs MuJoCo Phong) contributing to gradient energy. Dividing by the GS output of the same capture cancels most of this, but not completely.

### Lighting (`real`)

`genesis_allex_check.py lights` divides `demo1` into four segments and computes the average brightness of ideal rendering and the segment average of AE gain `2^ev`. 2–14 s is the segment where the head is stationary. 14–30 s involves waist and neck rotation, but the field of view is similar to the stationary segment. 30–42 s involves the neck bending down, filling the screen with the floor. 14–42 s covers the entire rotation segment.

| Relative to MuJoCo (brightness / mean AE gain) | 2–14 s | 14–30 s | 30–42 s | 14–42 s |
|---|---|---|---|---|
| MuJoCo `scene.xml` (absolute values) | 73.3 / 1.641 | 73.3 / 1.668 | 82.9 / 1.652 | 77.4 / 1.661 |
| Genesis default lighting | −46.0 % / +258.7 % | −45.4 % / +251.3 % | −30.3 % / +150.4 % | −38.5 % / +208.2 % |
| Genesis `--lights` (fixed forward light) | +3.4 % / +10.8 % | +3.2 % / +10.3 % | −27.8 % / +105.9 % | −11.0 % / +51.2 % |
| Genesis `--headlight` | −0.6 % / +0.5 % | −0.7 % / +0.7 % | +1.1 % / +3.1 % | +0.1 % / +1.7 % |

`--lights` (`scene_lights(forward=True)`) consists of a directional light shining from above (intensity 3), a fixed +x directional light (intensity 1, replacing MuJoCo headlight), ambient 0.25, and background (0.2, 0.33, 0.43) (replacing skybox). These values are tuned only for the front view; when looking down at the floor with the head bent, it is 28 % darker and the AE doubles the gain. The initial report calculated the AE gain as the median, and that value (+11 %) masked this segment.

`--headlight` adds `GenesisCamera(headlight=2.0)` to `scene_lights(forward=False)` (overhead directional light of intensity 8, ambient 0.3, background (0.12, 0.198, 0.258)). The headlight moves the directional light node to the camera pose just before rendering (directional light points along −z of its pose, and the camera also looks along −z). Since Genesis renders the floor darker and a plain background brighter than MuJoCo, adjusting light intensity alone cannot simultaneously match views showing the sky and views showing the floor. Therefore the overhead light and the headlight were raised and the background lowered to match both pose types in `demo1` (`genesis_allex_check.py light-poses`):

| `demo1` Pose | MuJoCo Brightness | Genesis `--headlight` |
|---|---|---|
| 8 s (Front: sky, box) | 73.3 | 72.9 (−0.6 %) |
| 32 s (Floor) | 80.4 | 80.6 (+0.2 %) |
| 36 s (Floor) | 80.9 | 81.4 (+0.6 %) |
| 40 s (Floor, hand) | 86.1 | 83.0 (−3.6 %) |

These values were tuned using `demo1` poses and verified with the same `demo1`. It has not yet been verified if they hold for other motions or scenes.

With default lighting, the AE applies a 3.6x gain, increasing shot noise proportionally. To compare exposure and noise parameters in Genesis, lighting must be matched first. The fact that the headlight follows the camera is visible in the test: the fixed +x directional light fails to illuminate the front box when viewing −y (red 99.5), but the headlight illuminates +x (167.6) and −y (163.1) equally.

### Checkerboard K/D Recovery (`calib_geom`, 9×6 internal corners, 4 cm, 15 poses, k3 fixed)

| | fx / fy | cx / cy | k1 (set to −0.05) | lens-field error vs the set lens |
|---|---|---|---|---|
| MuJoCo `geom` | −0.09 % / −0.10 % | −0.17 / +0.20 px | −0.031 | 0.33 px |
| Genesis `geom` | −0.12 % / −0.12 % | −0.30 / +0.03 px | −0.041 | 0.46 px |
| MuJoCo `static` | −0.06 % / −0.06 % | −0.18 / +0.28 px | −0.042 | 0.38 px |
| Genesis `static` | −0.14 % / −0.14 % | −0.27 / +0.08 px | −0.044 | 0.52 px |

Directly comparing corners of the same board pose in Genesis and MuJoCo yields an average of 0.26–0.30 px and a max of 0.70 px, which does not increase with radius. The initial "3 px corner difference" observed with color box silhouettes was not due to geometry, but rather different face shading per renderer altering the color mask boundaries. Since k1, k2, and k3 change complementarily within this field of view, lens field error is a more suitable comparison metric than individual coefficients.

**Residual bias:** Genesis output is translated by (−0.21, −0.12) px relative to MuJoCo (`static`: (−0.21, −0.14) px) (signed average of checkerboard corners output by `checker`, independent of radius). Matching K causes the principal point to absorb this bias (cx difference 0.13 px, cy difference 0.17 px). On the small test rig, the color mask boundary effect under Genesis default lighting adds to the deviation, shifting it by 0.70 px from the analytical silhouette (MuJoCo `SimCamera` 0.17 px, `tests/test_genesis_bridge.py`).

## Floating base: Unitree G1

`interp_qpos` interpolates the quaternion of the free/ball joint (`quat_starts`, found in MJCF's `jnt_type`/`jnt_qposadr`) via shortest-path slerp. This is equivalent to `mj_differentiatePos` + `mj_integratePos` used by `StateRing.load`: maximum difference 4.4e-16 across 200 pairs of random free/ball/hinge poses (`tests/test_genesis_bridge.py`).

The validation scene is the Menagerie G1 with the same head camera as `g1_walk.py` and ten coloured boxes on a 2.5 m ring. It moves the pelvis free joint kinematically along a trajectory (forward 0.4 m/s, yaw 2.5 rad/s, 1.8 Hz vertical/pitch/roll oscillation, arm swinging) without physics, feeding the same qpos into both simulators every 2 ms, and compares 58 captures in the 0.1–2.0 s window in window mode (`WINDOW`, distorted). The two renderers shade faces differently, so a dark face can drop out of one colour mask only; this check therefore uses ambient light only and compares geometry.

```bash
$PY scripts/genesis_g1_check.py render mujoco; $PY scripts/genesis_g1_check.py render genesis
$PY scripts/genesis_g1_check.py report
```

| Item | Result |
|---|---|
| Genesis qpos layout = MuJoCo (same qpos → link positions) | Max difference 3.1e-7 m |
| Window output silhouette IoU, Genesis vs MuJoCo | Median 0.995, p10 0.992, min 0.990 |
| Window output centroid offset | Median 0.46 px, max 1.21 px |
| Note: Window centroid vs `capture_t` static render | MuJoCo 6.95 px, Genesis 7.05 px (vs end-time pose: 27.5 / 27.7 px) |

The last row is not an error. The rolling shutter records each row at a different time and the boxes enter and leave the frame edges, so the centroid is not symmetric; both simulators shift by the same amount. The time axis itself is validated by the free joint rig test with exposure only enabled, using a 0.5 px threshold.

## Batched environments

`GenesisCamera(..., n_envs=n)` builds the mirror scene with `n` environments and renders each environment separately (`split_envs=True`). `push`/`render`
accept `(n, nq)` qpos inputs and append the environment axis to the output. Since the camera for environment `i` in the pipeline is `SimCameraBatch`'s camera `i`, it must match a single `GenesisCamera` with seed `cfg.seed + i` according to the seed rules. Quaternion interpolation also performs operations identical to a single path for each environment.

| Item | Result |
|---|---|
| 4 environments (each with different yaw speed, window + depth + noise) vs 4 single cameras with seeds `cfg.seed + i` | RGB difference 0 DN, depth difference 0 m (identical at the bit level) |

Throughput (G1 scene, window mode: 13 subframes per capture + depth, margin 656×494, RTX 5080, GPU not shared with other tasks):

```bash
$PY scripts/genesis_g1_check.py bench 0 1 4 16
```

| n_envs | Per capture | Per environment |
|---|---|---|
| 0 (not batched) | 25.4 ms | 25.4 ms |
| 1 | 24.9 ms | 24.9 ms |
| 4 | 45.3 ms | 11.3 ms |
| 16 | 114.9 ms | 7.2 ms |

Although the Genesis rasterizer renders the environments one by one in split mode, the cost per environment drops 3.5x at 16 environments. It is not yet clear which stage contributes to this reduction.

## Large-scale: Madrona Batch Renderer and Batch Accumulation

We measured where time is spent step-by-step in the G1 scene (window mode, 13 subframes per capture + depth, margin 656×494) when increasing the number of environments. In the Genesis rasterizer path, one capture for 16 environments (117.8 ms) consisted of 74% Genesis render and CPU read, 14% copy, 5% the per-environment `ingest` loop, and 0.6% post-processing (`process`) in this package. Since heavy computations are already running in CUDA kernels and C++, moving code to another language would not reduce the time. Therefore, we systematically eliminated the measured bottlenecks.

1. **`GenesisCamera(renderer="madrona")`**: Renders the scene using Genesis’s BatchRenderer (Madrona, rasterizer mode). All environments are rendered on the GPU at once, and the results are output as CUDA tensors, so they do not pass through the CPU. Since Madrona does not use the lighting and ambient settings from `VisOptions`, lighting is provided via `madrona_lights` (argument to `scene.add_light`). All standard cameras must have the same resolution, so the ideal camera is set to the debug camera. Since lighting is fixed at build time, headlight is not supported.
2. **`SimCameraBatch.accumulate_all` / `ingest_all` / `claim_all`**: Accumulates frames for all cameras at once for each subframe rendered. This eliminates the buffer of size env × subframe (3 GB for 256 environments) and the per-environment Python loop. The operations and order are identical to `_accumulate`, so the result for camera `i` is bitwise identical to `ingest(i, …)` (`tests/test_batch.py::test_batch_ingest_all`, including window and global shutter, noise, AE, flicker, depth noise, and `chunk` split paths).
3. **`compile=True`**: Combines the LUT lookup for accumulation (axis rearrangement + int conversion + gather) into a single `torch.compile` kernel. Since it is a gather operation, the results are identical (same tests), and for 64 environments, this step went from 21.1 ms to 5.2 ms.

Throughput (per environment, G1 window capture, RTX 5080, GPU not shared with other tasks, `genesis_g1_check.py bench [madrona] [compile] n`):

| env count | Rasterizer | Madrona | Madrona + `compile` |
|---|---|---|---|
| 1 | 24.9 ms | 25.2 ms | — |
| 16 | 7.2 ms | 2.32 ms | 2.10 ms |
| 64 | — | 1.48 ms | 1.22 ms |
| 256 | (out of memory: stacked buffer) | 1.38 ms | 1.12 ms |

For 64 environments, the breakdown of one Madrona capture (approx. 100 ms, without `compile`) is: Madrona render 31%, accumulation 24%, render scene `set_qpos` 8%, camera update and memory alignment 6%, qpos interpolation 3%, post-processing 3%. Most of the remaining time is also GPU computation, with Python consuming only a few percent.

The geometry of the Madrona path is the same as or better than the rasterizer path (`genesis_g1_check.py render madrona`, same G1 trajectory, directional lights from six directions, no shadows):

| G1 window output, vs MuJoCo | Rasterizer | Madrona |
|---|---|---|
| Silhouette IoU median (min) | 0.995 (0.990) | 0.995 (0.993) |
| Centroid offset median (max) | 0.46 px (1.21 px) | 0.12 px (0.65 px) |

The analytical silhouette center error on the test rig is also 0.44 px for Madrona and 0.70 px for the rasterizer. Batch processing with 2 environments and single-camera captures are bitwise identical in Madrona as well.

Note: The Genesis rasterizer corrupts the heap and exits in some scenes if the light list is empty, such as `VisOptions(lights=[])` (`malloc(): corrupted top size`, reproducible on the free-joint rig). To use only ambient light, place one directional light with intensity 0.

## Apptronik Apollo Example

`examples/genesis_apollo.py` runs the same pipeline on Apptronik Apollo (MuJoCo Menagerie, Apache-2.0) with a D435-class camera mounted on its head. Unlike ALLEX, the mesh allows redistribution and commercial use, so this example is used for public demos and videos.

- **Camera mounting**: Apollo’s two default cameras are OAK-D units on the torso pointing downward. Therefore, a new `head_cam` (fovy 42°) is attached to the `neck_pitch_link`. The coordinate system of this link points approximately 40° upward in the `stand` pose, so the camera is positioned relative to the world (face front at (0.14, 0, 1.62) m, 35° downward) and then transformed into the link coordinate system. Due to ±0.08 rad pitch oscillation of the neck, the line of sight sweeps between 30° and 40° downward during execution.
- **Collision geometry**: Apollo’s collision geometry belongs to group 3 (translucent green). MuJoCo hides this group by default, but Genesis renders it, so it is changed to transparent (rgba alpha 0).
- **Motion**: The trajectory is provided directly without physics. The torso is fixed at the `stand` keyframe, and the neck rotates slowly, then quickly at 8-12, 18-22, and 28-33 s (exceeding 1 rad/s), while the right arm waves within the frame during 13-18.5 s. All joints remain within their limits.
- **Configuration**: `--cfg calibrated` corresponds to `CameraConfig.preset("calibrated")` (calibrated frames, so no distortion; depth effects included). `geom` / `static` / `wingeom` are not cumulative; each adds one effect group each to the example’s distortion value `DIST` (raw frames; not part of the preset) using preset typical values: distortion only / sensor noise, vignetting, response, AE / motion blur, rolling shutter. Timing uses all preset values, and depth effects are present only in `calibrated`. The default execution lasts 10 s, and the full motion (34 s) accumulates frames in approximately 3 GB of memory.

```bash
.venv/bin/python examples/genesis_apollo.py --preview out/genesis_apollo/preview.jpg          # MuJoCo static frame, check composition
.venv/bin/python examples/genesis_apollo.py --sim mujoco --cfg wingeom --seconds 10
.venv-genesis/bin/python examples/genesis_apollo.py --sim genesis --cfg calibrated --seconds 34 \
    --video out/genesis_apollo/apollo.mp4                                                      # External view + ideal + output
```

The test (`tests/test_genesis_apollo.py`, skipped if Menagerie is missing) verifies the camera’s parent link, position, and orientation (within XML round-trip error), whether the camera follows when the neck rotates, collision geometry hiding, joint limits of the 34 s trajectory and rapid neck rotation, and whether frames arrive late by the preset delay in MuJoCo 0.5 s captures.

## Tests

```bash
make test                                                     # default venv: the 6 Genesis tests are skipped
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-genesis/bin/python -m pytest tests/test_genesis_bridge.py -s
```

`tests/test_genesis_bridge.py` contains unit tests for `camera_mount` (link pose × offset = camera world pose, 1e-9), `quat_starts`, and `interp_qpos` (including quaternions, 1e-9 compared to MuJoCo), as well as integration tests that run only when Genesis is available. One test compares the silhouette and output positions of projected box vertices using K and distortion (1 px), while the other verifies the window time axis by checking whether the center of the window output on a rotating rig matches the static render of the `capture_t` pose (0.5 px; differs by 3.07 px from the final pose). The same verification is performed on a yawing free-joint rig (floating base). Batch tests verify that 4 environments match 4 single cameras bitwise, and the headlight test verifies that lights follow the camera. Since `GenesisCamera` requires Genesis and a GPU, it was excluded from coverage measurement (`# pragma: no cover`).
