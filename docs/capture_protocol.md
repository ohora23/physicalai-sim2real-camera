# Real-camera calibration capture protocol

A single-session checklist for collecting the data fed into the Phase 3 calibration (`physicalai_sim2real_camera/calib*.py`, `scripts/calibrate.py`).
Save the results under `calib_data/` as described in "File layout" below.

## 0. Common setup — fix exposure/gain manually

D2, D3, D4 and D5 must all be captured **with AE/AWB/automatic gain turned off**. If any automatic adjustment is
on, the flat-field, PTC and response-curve fits are all meaningless.

- [ ] In the camera app/driver settings, fix exposure, gain and white balance manually.
- [ ] For USB webcams with no manual-fix UI, force it with `v4l2-ctl`. Example:
  ```sh
  v4l2-ctl -d /dev/video0 --list-ctrls          # list supported controls
  v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1,exposure_time_absolute=300,gain=50,white_balance_automatic=0
  ```
  (The values are examples. The point is to turn off `auto_exposure`/`gain`/`white_balance_automatic`
  entirely and pin them to fixed values.)
- [ ] Record the fixed exposure/gain values (D2 and D3, and D2 and D4, must use the same settings).
- [ ] Right after the D2 capture, check the frame-mean brightness drift across the stack. **Above 1% is
      evidence that AE is still on**, so recheck the settings and recapture.

### 0a. Lighting flicker check (once per installation site)

AC-powered lights flicker at 2x the mains frequency (100 Hz in 50 Hz regions, 120 Hz in 60 Hz regions). It differs per site, so
measure it at first installation and match the capture exposure and fps to it (`physicalai_sim2real_camera.flicker`).

- [ ] Capture a static scene at **27 fps** (`flicker.MEASURE_FPS`, not a divisor of either 100 or 120), at manual exposures of
      1, 2, 3, 5, 8.33, 10 ms, 90 frames each, and collect the per-frame mean brightness. Capturing at a divisor of 120 such as 30 fps
      locks the phase and hides the flicker.
- [ ] `detect_mains({exposure_s: frame_means, ...})`: determines the mains frequency from whichever of 8.33 ms and 10 ms cancels the flicker
      (independent of waveform). If `confident` is False, capture again.
- [ ] `recommend(res["flicker_hz"], fps_options=...)`: returns the exposures that cancel the flicker (integer multiples of the period), the fps with
      no beating (divisors of the flicker frequency), and the fps at which that exposure fits in one frame. Choose the D2–D5 capture exposures from this list too.
- [ ] Pass `res["config"]` (`flicker_hz`, `flicker_amp`) to the simulator as is. For non-sinusoidal lighting,
      `flicker_amp` is an approximation.

## D1 — Geometry (K, D)

Obtain the intrinsic/distortion parameters with a checkerboard.

- [ ] Checkerboard: 9×6 inner corners; measure and record the side length of one square. Pass it to the CLI as `--pattern 9,6 --square-m <meters>` (the CLI default `--pattern 7,5` is for the synthetic board).
- [ ] Capture at varied tilts so the board covers the four corners and edges of the image, not only the centre.
- [ ] Capture 20~40 images. Blurred frames are filtered out by reprojection error in the fit stage, but
      frames that already look blurry can be excluded at this stage.
- [ ] Save to: `calib_data/D1/*.png`
- [ ] Wide-angle lenses whose factory calibration is an 8-coefficient rational model can be passed as is through `CameraConfig.preset(..., dist=8 coefficients)`.
      However, the factory model often folds over before the image corner, so pixels outside that domain are clamped to the boundary
      and a warning is raised (`optics.rational_domain`, `outside_domain_fraction`). `scripts/calibrate.py` and the online observer
      still fit only 5 coefficients.

## D2 — Noise · vignetting (flat-field)

- [ ] Hold an opal diffuser in front of the lens, or defocus and capture a uniform white wall.
- [ ] Fix AE/AWB/gain manually (see section 0 above).
- [ ] Capture while varying illumination or exposure time over 6~8 levels (each level has the same gain and white balance,
      only the brightness differs).
- [ ] At least 100 frames per level (needed for temporal mean/variance computation).
- [ ] Save to: `calib_data/D2/level_<k>/*.png` (k = 0..7, level index)

## D3 — FPN · black level (dark frame)

- [ ] With the lens cap on, capture with the **same gain and exposure** as D2.
- [ ] At least 100 frames.
- [ ] Save to: `calib_data/D3/*.png`

## D4 — Response curve (gamma)

- [ ] Prepare a grayscale chart with known reflectances (6 neutral ColorChecker patches, or a printed gray wedge).
- [ ] Use the same manual settings as D2 and capture from the front with uniform illumination on the chart.
- [ ] Save **1 cropped uniform PNG** per patch (do not include a photo of the whole chart: the CLI uses the mean of the whole image). The filename numeric order = the `reflectances.csv` order.
- [ ] Record the known reflectance of each patch in `reflectances.csv` (match the patch order).
- [ ] Save to: `calib_data/D4/*.png`, `calib_data/D4/reflectances.csv`

## D5 — Timing

### D5a (required) — baseline, photon→policy end-to-end

- [ ] Drive the LED directly from host GPIO or USB-serial (if the controller drives it, clock
      synchronisation via PTP/NTP is mandatory).
- [ ] **Same host clock rule**: always record the LED toggle command time and the time the policy process
      judged the frame to be available using the **same host `CLOCK_MONOTONIC`**. Do not mix
      clocks of different machines without synchronisation.
- [ ] Also record the driver timestamps (V4L2 buffer timestamp, RealSense `SENSOR_TIMESTAMP` /
      `BACKEND_TIMESTAMP` / `TIME_OF_ARRIVAL` metadata) and the frame number (`frame_seq`).
      They are used later to split the analysis into sensor→host→policy segments.
- [ ] Fix the exposure manually and record the value.
- [ ] **Random toggle phase is required**: make the toggle interval longer than the expected latency upper bound, with a
      random phase unrelated to the frame period (e.g. uniform distribution over 0.3~0.7 s). If the interval is correlated with the frame period, the latency estimate
      is aliased.
- [ ] At least 300 toggles.
- [ ] Save to: `calib_data/D5/events.csv` (columns: `event_t,avail_t,frame_seq,event_seen`)

### D5b (optional, sub-frame)

- [ ] Capture a 1 kHz binary counter built from 8 LEDs, or a millisecond stopwatch display, and read the
      exposure time directly from each frame (no quantisation bias).
- [ ] For a rolling-shutter camera, the difference between the readings of the top and bottom rows of a frame is the readout time.
- [ ] 100 frames.

### D5c (optional, batch/policy perspective)

- [ ] Attach an AprilTag to a robot link and drive the joint along a known sinusoidal trajectory (0.5~2 Hz).
- [ ] Estimate the time offset between the encoder (controller clock) time series and the tag pose time series estimated from the images
      by cross-correlation or least squares (the same principle as camera-IMU temporal calibration).
- [ ] 30 s × 3 runs.

## D6 (optional, Phase 2) — AE time constant

- [ ] With AE on, switch the light off and on and record continuously.
- [ ] 5 s × 3 runs.

## D7 (optional) — Prism sweep recording (for online observer replay)

After D1~D5 are done, record the frames together with the robot state while the robot moves, and replay them with `scripts/observe.py`.
From this recording alone, the observer re-estimates K and distortion, shot/read noise, and latency, jitter and drops.

- [ ] Fix the prism target (`assets/prism`, printed with `scripts/make_prism.py`) at a working distance of 0.75 m. Record the measured
      dimensions of the pillar (face spacing, height) and the printed reflectance of the wedge faces.
- [ ] **Fix exposure and gain manually** (AE/AWB off). Save the SDK exposure and gain metadata with every frame.
- [ ] Have the head camera orbit the pillar, alternating between standing still and moving (at least 12 stops of ≥ 1 s, with the pillar
      in a different region of the image at each stop). The geometry and photometric observers use the stationary segments, and the timing observer uses the moving ones.
- [ ] For every frame, record the following (all times and exposures in seconds): host arrival time (`t_arrival`, on the **same host monotonic clock** as the robot state), SDK
      timestamp (`ts_sensor`), SDK `frame_number`, transport-layer sequence number (`seq`), exposure, gain, and the number of head-state
      samples recorded up to the moment that frame was received (`head_count`).
- [ ] Record the head state (position, orientation quaternion wxyz, angular velocity) every control period on the same host clock. If the controller is on a different
      machine, synchronise the clocks with PTP.
- [ ] 60 s × 3 runs.

Recording folder layout (`record` in `physicalai_sim2real_camera.observer_replay` writes the same format from the simulator):

```
recording/
  info.json        # {"ring_n": 16384, "head_base": 0, "frame_number_bits": 32 | null, "seq_bits": null}
  frames.jsonl     # one line per frame: t_arrival, ts_sensor, seq, frame_number, exposure, gain, head_count
  frames/000000.png ...   # lossless PNG (cv2.imwrite, BGR), numbered in the same order as the lines of frames.jsonl
  head.npy         # (m, 11) float64: t, pos(3), quat wxyz(4), omega(3) — only the magnitude of the angular velocity is used
```

- `head_count` is the number of `head.npy` rows received up to that frame's `t_arrival` (counted from row 0). The rows of `head.npy` are in time order
  and t must be strictly increasing. The number of samples the ring had already discarded before the recording started is `info.json`'s `head_base` (usually 0 for real recordings).
- If a counter can wrap around, write its bit width in `info.json` (52 bits or fewer). `load_recording` unwraps it.
- Pass the `prism.json` corrected with the measured pillar dimensions and wedge reflectance to `scripts/observe.py --target` (the default is the as-fabricated values).
- `ring_n` is how long the head state is kept during replay (the live ring size). Real recordings
  use 16384 (32 s at 500 Hz).
- Replay: `python scripts/observe.py recording/ --config calib_out/camera_config.json --out est.json`. The observer takes only the gamma, black level and rolling-shutter readout
  from the config.

### Example: characterising a fixed UVC camera without a target

Without a target, some parameters of a fixed camera can still be measured remotely. On the camera host, use `scripts/capture_uvc.py` (no project imports, Python 3.8 / OpenCV 4.2 or later, requires `v4l-utils`) to capture the timing stream and still frames per exposure step, and analyse them on this PC with `scripts/analyse_uvc.py`.

```bash
python3 capture_uvc.py --out /tmp/uvc.npz [--raw]          # on the camera host, ~1.5 min, static scene
.venv/bin/python scripts/analyse_uvc.py uvc.npz --out uvc_report.json
```

Example results for a Logitech Webcam Pro 9000 (046d:0809, 640×480 YUYV):

| Item | Value |
|---|---|
| Exposure control | **Works backwards**: the larger `exposure_absolute`, the darker the image, and frame time × value ≈ constant (55 → 5 fps, 325 or more → 30 fps). `exposure_auto_priority=0` does not prevent the fps drop either. OpenCV `CAP_PROP_EXPOSURE` turns anything below 55 into 55 |
| Gain | Stepwise (0 = 64, 128 ≈ 2x, 255 ≈ 3.4x) |
| fps / jitter θ | 31.25 fps (period 32.0 ms) / 1.5~1.8 ms |
| Drops | 0 (1805 and 603 frames at 30 fps, 601 frames at 10 fps; frame numbers are reconstructed from each timestamp interval) |
| Driver timestamp → app arrival | ≈ 1 frame (34 ms at 30 fps, 101 ms at 10 fps). It looks like the OpenCV V4L2 queue, but it also contains the USB transfer and the timestamp reference point (estimate) |
| Response | Treating exposure time ∝ 1/value, γ ≈ 1.45 (RGB 1.44, Y 1.49; r² 0.999), black level ≈ 0. Y has a limited range (measured 16~80) |
| Noise (PTC) | **Cannot be estimated** under indoor lighting: even at maximum gain the pixel noise is 0.8~1.8 DN rms, at the 8-bit quantisation level, and the linear fit r² is 0.43~0.49 |

What cannot be done remotely (physical work required): K and distortion (a board or prism in front of the camera), latency (a stimulus such as an LED), noise (a bright uniform scene or a gray card, and a cover for dark frames).

## File layout

```
calib_data/
  D1/*.png
  D2/level_<k>/*.png        # k = 0..7, exposure/illumination level index
  D3/*.png
  D4/*.png                  # 1 per patch (a uniform image cropped from each gray patch), filename numeric order (natural sort) = patch order
  D4/reflectances.csv       # one reflectance per line, matching the sorted order of the D4 images
  D5/events.csv             # event_t,avail_t,frame_seq,event_seen
```

- Read PNGs with cv2 (BGR) and convert to RGB — if they were also saved with cv2.imwrite (BGR), the channel order round-trips unchanged.
- D2/D3 may have a different resolution from D1 (the CLI scales C1's K to the D2 resolution and uses it for the vignetting fit).
- `D5/events.csv` mixes two kinds of rows after the 1 header line: **toggle rows** fill in only `event_t`
  and leave the other cells empty (`0.4035,,,`), **frame rows** (1 line per surviving frame) leave `event_t` empty
  and fill in `avail_t,frame_seq,event_seen` (`,0.0558,0,0`). `event_seen` = the number of
  toggles up to that frame's capture time.

When the capture is finished, pass the whole `calib_data/` to `scripts/calibrate.py` to generate `camera_config.json` and
`calib_report/*.csv`. The D1~D5 captures can be done first, independently of the code work.

## Partial sessions

- D1 (geometry) and D2 (flat-field) are required. **If D3/D4/D5 are missing, the CLI emits a warning, skips the corresponding fit and keeps the defaults (black 0, gamma 2.2, ev 0, default timing values).**
- Filename numbers are natural-sorted (`img_2` < `img_10`). The row order of `events.csv` does not matter (the CLI sorts by frame_seq).
