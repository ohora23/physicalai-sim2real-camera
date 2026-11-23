# Online camera observer

The offline calibration (`scripts/calibrate.py`, D1-D5) fits a `CameraConfig` once. The **observer** keeps estimating
the camera while the robot moves: intrinsics and distortion, shot / read noise, latency / jitter / drops, each with
an uncertainty, and feeds them back into the config and its domain-randomization spread.

```
frames + sensor stamps ─┐          ┌─ GeomObserver   (K, dist ± σ, coverage)      ┐
                        ├─ Observer ┼─ PhotoObserver  (σ_shot, σ_read ± σ)          ├─ estimate ─ estimate_to_config ─ CameraConfig + spread ─ randomize
head pose ring ─────────┘          └─ TimingObserver (latency, θ, drops ± σ)       ┘     (two disjoint windows must agree)
```

## Target

A ChArUco **prism** (`scripts/make_prism.py` → `assets/prism/`): six faces with disjoint marker ids, one face carrying a
grey wedge (reflectance 0.03-0.9). `PrismTarget.detect` runs one ArUco marker pass per frame and interpolates each
face's ChArUco corners from its own markers (15 ms per 640x480 frame single-threaded). Print the measured prism
dimensions and wedge reflectances into `prism.json` for a real target.

## Estimators

| Observer | Input | Output | Notes |
|---|---|---|---|
| `GeomObserver` | still views of each face (motion gate: blur < 0.2 px during exposure + readout) | K, 5-term distortion, per-field σ (fit σ + 1.5 px detector-bias floor), coverage grid | sliding window of distinct poses (shared-corner de-duplication), age eviction, `reset()` on a known re-mount |
| `PhotoObserver` | still intervals (head still, image still, constant exposure / gain) | σ_shot, σ_read at the latest gain, σ of each | wedge steps + black squares (`dark_boxes`, read-noise anchor); per-sample linearisation; clipped levels skipped; 1 % drift guard; flicker removed as a common-mode gain |
| `TimingObserver` | arrival time, sensor stamp, transport seq, camera frame number, camera angular speed from face poses | latency (± σ), jitter θ (± σ), camera drop rate, observer miss rate | latency from matching |ω_cam| against the reported head speed (affine, positive slope, 30 s window, bootstrap σ) |

The observer never sees simulator truth: the closed loop (O6) starts from a nominal K × 1.05, perturbs the prism by
0.5 mm / 0.3°, takes γ and black level from the offline calibration, delays the head state by 4 ms and adds noise.

## Feedback (`calib.estimate_to_config`)

```python
from physicalai_sim2real_camera.calib import estimate_to_config, randomize
cfg, spread, report = estimate_to_config(est_new, est_old, base_cfg, base_spread)
episode_cfg = randomize(cfg, spread, rng)
```

- Groups: `geom` (K + distortion), `photo` (shot + read), `timing` (latency + θ), `drop`.
- A group is updated only when **both** windows pass its own gate (≥ 15 views and ≥ 50 % coverage; ≥ 3 photometric
  intervals with level span ≥ 8; ≥ 300 frames; finite positive σ) and every field agrees:
  `|a − b| ≤ k_n · hypot(σ_a, σ_b)`, Bonferroni `k_n` at α = 0.05 per group (2.77 for 9 fields). The geometric gate
  uses the fit σ (`std_fit`): the detector-bias floor is common to both windows.
- Spread = max(√3 · 2 · σ, floor) in `randomize`'s convention (absolute for distortion, cx, cy, drop; relative
  otherwise). The floor keeps the DR from collapsing as σ shrinks with 1/√N.
- `report[group]` is `updated`, `gate` or `disagree: <fields>`.

What two 16.6 s windows resolve (O6, 20 seeds): principal point 6 px, focal length 1 %, latency 3 ms (global
shutter); a 3 px re-mount is not caught, and a rolling-shutter latency change only partly (20 / 60 / 80 % at
5 / 7 / 10 ms). Longer windows resolve smaller changes.

## Accuracy (O6 closed loop, 20 seeds × rolling / global shutter)

| | rolling | global | tolerance |
|---|---|---|---|
| fx / fy | 0.30 % | 0.30 % | 0.5 % |
| principal point | 2.25 px | 2.18 px | 3 px |
| distortion field (corner hull) | 3.75 px | 3.95 px | 4.5 px |
| σ_shot / σ_read | 1.4 % / 4.0 % | 1.5 % / 4.1 % | 10 % / 20 % |
| latency (worst) | 2.0 ms | 1.6 ms | 2 ms |
| θ | 12.8 % | 12.1 % | 20 % |
| \|err\| / σ ≤ 3 | 100 % of seeds, every field | 100 % | ≥ 95 % |

Feedback closure (10 trials per shutter: estimate → `estimate_to_config` → simulate the fed-back camera →
re-estimate): every group updated; re-estimates within 3 σ of the fed-back value in 90-100 % of trials per group
(4 misses in 20 trials, worst z 3.13-3.30);
the truth inside the fed-back DR for all 14 fields in every trial.

## Running it

- Simulator closed loop: `python scripts/observe_sim.py --seeds 20 --out out/o6` (statistics, O8 agreement and gate
  power), `--closure 10` (feedback closure).
- Recording / replay (O7): `observer_replay.record(records, ring, path)` tees a stream to disk;
  `python scripts/observe.py <recording> --config calib_out/camera_config.json --out est.json` replays it. A replay
  feeds the observer bit-identical inputs: its estimates match an in-memory replay of the same records up to
  multi-threaded summation order (a live run may skip frames; the claim is for the
  frames consumed). Real recordings: `docs/capture_protocol.md` D7.
- Live (O5): `examples/virtual_sensor.py --scene prism --observer [--observer-cams 0]` adds an observer process
  (one camera, nice 10, one OpenCV thread) that writes `camera_config_estimate.json` at ≤ 1 Hz. Cost on the
  real-time path: `docs/perf.md` (Observer).
