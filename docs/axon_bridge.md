# Axon SDK bridge

Build and Python-binding verification results for the [Axon SDK](https://github.com/ohora23/Axon_SDK_for_PhysicalAI), a
zero-copy tensor transport library for sensor→accelerator→RT control loops (C++, pybind11 bindings, Apache-2.0,
pre-alpha). The source is cloned into `third_party/axon/`
(already in `.gitignore`, not committed).

## Build

```bash
./scripts/build_axon.sh
```

Internally:

```bash
cmake -S third_party/axon -B third_party/axon/build \
    -DCMAKE_BUILD_TYPE=Release -DAXON_BUILD_PYTHON=ON \
    -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
    -Dpybind11_DIR=<venv pybind11 cmake dir>   # see the blocker below
cmake --build third_party/axon/build -j
```

If `nvcc`/`/usr/local/cuda` is detected, it automatically adds `-DAXON_WITH_CUDA=ON`
(CUDA VMM `Accelerator` pool backend, `PoolBackend.Accelerator` + framework
`__cuda_array_interface__` zero-copy). In this environment (CUDA 12.8, driver 580)
the CUDA build succeeded.

Metadata channel: `AXON_WITH_ICEORYX2` is OFF by default and Iceoryx2 is not a hard
dependency — the default is the library's own seqlock/shared-memory backend
(`src/metadata_channel.cpp`), so no extra action is needed.

PYTHONPATH:
```bash
export PYTHONPATH=third_party/axon/build/python
```

## Blocker (resolved): pybind11-dev (apt) ↔ NumPy 2.x ABI mismatch

The apt `pybind11-dev` is 2.11.1, which predates NumPy 2.0 C-API support (added in
pybind11 2.12+). The numpy in this project's venv is 2.5.3, so building with this combination
**breaks every stride of `TensorView.data` to 0** — i.e.
`view.data` has the right shape, but it is a "zero-stride view" in which every element is broadcast from
the first byte of the buffer, so it is not the real data. Axon's own regression test
(`python/tests/test_axon.py`) also failed because of this
(`ValueError: To change to a dtype of a different size, the last axis must be
contiguous` — the result of reshape hitting a non-contiguous view).

This symptom was identified as a pybind11/numpy version-combination problem,
not an axon library bug (reproduced: build with apt pybind11 2.11.1 → fail /
rebuild with venv-local pybind11 3.1.0 → pass).

**Fix**: without sudo, install a recent pybind11 only into the venv and point CMake at it
(already reflected in `scripts/build_axon.sh`):

```bash
uv pip install --python .venv/bin/python "pybind11>=2.12"
# -> .venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11
```

Because it is solved without touching system apt packages, there is no `apt install` blocker.
(For reference, `apt install pybind11-dev` was already installed but the version was too old, which
was the problem — upgrading it would need `sudo apt install --only-upgrade pybind11-dev`,
but the repository may not have a newer version, so the venv-local install was chosen.)

## Verification results

### ctest (third_party/axon/build)

```
100% tests passed, 0 tests failed out of 11
```

`python_binding`, `python_device_array` and `python_device_consumer` all PASS
(before the pybind11 upgrade only `python_binding` FAILED, because of the blocker above).

### Import check

```bash
PYTHONPATH=third_party/axon/build/python .venv/bin/python -c "import axon; print(axon.__file__)"
# -> third_party/axon/build/python/axon.cpython-312-x86_64-linux-gnu.so
```

### Smoke test (multiprocessing, two processes)

Create a `TensorPool`, put a `TensorPublisher`/`TensorSubscriber` in two `multiprocessing`
processes, publish (480, 640, 3) uint8 frames, and receive them with `latest_view()`.

| backend | seqno | staleness_ns | shape | data match |
|---|---|---|---|---|
| `PoolBackend.Custom` | 1 | 843,467 | (480, 640, 3) | True |
| `PoolBackend.UDMABUF` | 1 | 760,928 | (480, 640, 3) | True |

Both backends work. `/dev/udmabuf` is `crw-rw---- root:plugdev` and the current
user belongs to the `plugdev` group, so the UDMABUF backend was usable
right away without any extra udev/group changes.

staleness_ns was measured on the first frame the consumer received, with the producer publishing 20 frames at 10 ms intervals and the consumer
running a 1 ms polling loop, so it depends on the polling timing, not on the actual zero-copy round-trip
latency (microsecond scale, normal range).

## Python API check

### `TensorPool`

```python
axon.TensorPool.create(n_buffers, buffer_size,
                        backend=axon.PoolBackend.Custom,  # V4L2/UDMABUF/Custom/Accelerator
                        v4l2_device="")
pool.generation() -> int
pool.buffer_count() -> int
pool.device_ptr(index) -> int          # CUdeviceptr, 0 for host backends
pool.device_array(index, shape, dtype=U8, read_only=False) -> DeviceArray  # __cuda_array_interface__
pool.retire_and_reallocate(new_buffer_size)
```

### `TensorPublisher`

```python
axon.TensorPublisher.create(service_name, pool)
pub.handshake_pool() -> int            # number of connected consumers
pub.publish(numpy_array, dtype=U8)     # host copy publish
pub.acquire() -> AcquiredDescriptor    # prepare a device publish (exposes only buffer_index)
pub.publish_device(acquired, shape, dtype)  # publish the GPU buffer as is (no host copy)
pub.reannounce_pool()
```

### `TensorSubscriber`

```python
axon.TensorSubscriber.create(service_name)
sub.wait_handshake(timeout_ms=5000) -> int
sub.set_fallback_policy(policy)        # LastKnownGood/ZeroCommand/UserCallback/AbortLoop
sub.pool_handshake_count() -> int
sub.fallback_invocation_count() -> int
sub.latest_view(max_retry=8) -> TensorView | None
```

### `TensorView` fields (all of them)

- `seqno` — sequence number
- `staleness_ns` — time elapsed since publication (ns)
- `seqlock_retries` — number of seqlock retries incurred by this read
- `dtype`
- `shape`
- `data` — a read-only NumPy view pointing directly at the dma-buf (host frames; an empty array
  for device frames)
- `device_ptr` — the consumer-side device pointer (int) of a device frame, 0 for
  host frames
- `__cuda_array_interface__` (property) — the CAI dict only when `device_ptr != 0`,
  otherwise `None`

### Whether custom metadata (capture timestamp) can be attached

**Not possible** — `publish()`/`publish_device()` take only `dtype` and `shape`,
and `TensorView` has no field for a capture timestamp or arbitrary metadata
(`staleness_ns` is merely a value derived from Axon's internal clock, not a timestamp
supplied by the user). The bridge has to **carry the capture timestamp (`capture_t`) directly in a payload trailer**
— e.g. append 8 bytes (or 16 bytes, monotonic + wall clock) to the end of the NumPy array being published,
and have the receiver
slice `view.data` after the known frame byte count to recover the timestamp.

## Bridge

`examples/axon_bridge.py` — a publisher/subscriber example that sends `CameraStream` frames out on the Axon topic `realsensor/head_cam`
(two `multiprocessing` spawn processes).

```bash
MUJOCO_GL=egl PYTHONPATH=third_party/axon/build/python .venv/bin/python examples/axon_bridge.py --seconds 5 [--backend udmabuf]
MUJOCO_GL=egl .venv/bin/python tests/test_axon_bridge.py   # the binding path is added automatically via a sys.path fallback; SKIP (exit 0) if not built
```

- **publisher**: board scene + `SimCamera` (distortion, vignetting, noise) + `CameraStream(fps=30,
  latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.05)`. `mj_step` runs at 500 Hz locked to
  wall-clock with a `perf_counter` absolute schedule (`t0 + k·0.002`), so sim time == wall time.
  Publishes once immediately on the tick where `step()` yields a new frame (capture_t changed); dropped frames are
  not published. The physics loop is not tied to publishing. At every control tick, records CameraStream's own
  age (`t − capture_t`, the ground truth) to `out/axon/pub_age.csv`.
- **subscriber**: after `wait_handshake()`, uses `LastKnownGood` and polls at 500 Hz with the same `t0`
  (`t0 + k·0.002 + 0.5 ms`), recovers `capture_t` and `n` from the trailer →
  `out/axon/sub_age.csv` (`t, n, capture_t, age_sub, age_tick, staleness_ns, seqno`).
- Pool: `n_buffers=8`, `buffer_size = H·W·3 + 16`.

### Trailer layout

The payload is a single 1-D uint8 `(H·W·3 + 16,)` array (a preallocated array reused every frame):

| Offset | Size | Content |
|---|---|---|
| 0 | H·W·3 | RGB frame (row-major) |
| H·W·3 | 8 | `capture_t` float64 little-endian (sim time = seconds relative to t0) |
| H·W·3+8 | 8 | capture index `n` int64 little-endian |

### Clock comparability

On Linux, `time.perf_counter()` is `CLOCK_MONOTONIC`, which is system-wide, so it is directly comparable across
processes. Measuring `perf_counter() − clock_gettime(CLOCK_MONOTONIC)` in each process right after the barrier
gives pub −0.7~−1.1 µs, sub −1.2~−1.9 µs — merely the gap between the two calls, so the offset is 0.
So the publisher passes `t0 = perf_counter() + 0.2 s` through a shared `Value` and both sides use it as is.

### age vs staleness (5 s, measured)

| | custom | udmabuf |
|---|---|---|
| published frames | 141 | 141 |
| subscriber samples | 2470 | 2470 |
| pub age (ground truth) mean / p95 | 78.77 / 104.00 ms | 78.77 / 104.00 ms |
| sub age (wall) mean / p95 | 79.47 / 104.55 ms | 79.49 / 104.55 ms |
| staleness mean | 19.27 ms | 19.31 ms |
| mean(age_sub − staleness) | 60.19 ms | 60.18 ms |

Axon `staleness_ns` measures only the publish→read elapsed time (= on which tick the same frame is being read under sample-and-hold).
So `age_sub ≈ staleness + (avail − capture)`: 60.2 ms ≈ `latency_s` 50 ms +
the Gamma(2, 5 ms) mean of 10 ms, and mean staleness ≈ 1/(2·30 Hz) + the drop intervals. **The age from the policy's point of view is recovered only
from the trailer's capture_t, not from staleness.**

### Distribution match (tests/test_axon_bridge.py)

| | custom | udmabuf |
|---|---|---|
| KS(age_tick vs pub age) | 0.0040 | 0.0049 |
| KS(age_sub wall vs pub age) (for reference) | 0.0607 | 0.0615 |
| \|mean(age_sub) − mean(pub age)\| | 0.69 ms | 0.72 ms |
| seqno coverage | 1.000 | 1.000 |
| run time | 5.8 s | 5.8 s |

Pass criteria: KS < 0.05, |Δmean| < 3 ms, coverage ≥ 0.9, < 90 s → both PASS.

- **Polling phase**: when the subscriber polled at exactly the same phase as the publisher tick, it always missed that tick's publish
  (mj_step + 921 KB copy), so every frame appeared 1 tick (2 ms) late (Δmean +2.1 ms).
  Reading after `POLL_LAG = 0.5 ms` gives Δmean ≈ 0.7 ms (= lag + transfer/sleep overshoot).
- **KS is judged on tick times because of discretisation**: pub age is an atomic
  distribution on the 2 ms control grid, so however slightly the wall-clock age_sub shifts continuously, KS is pinned at the mass of one atom (~0.06).
  So KS is judged on `age_tick`, which measures the same reads at the loop's logical tick `k·0.002`
  (rounded to µs — to keep the 1e-15 s difference between the `data.time` running sum and `k·DT` from
  splitting tied atoms), and the wall-clock age is judged by the difference of means. Over the 5 s, the ticks at which pub/sub
  saw different frames were 9 of 2470 (ticks where the publisher was late because it overlapped with render).

### Notes

- Axon is **pre-alpha** — the API/ABI may change, and since there is no custom metadata field, the trailer
  convention is this repo's own. It must be built with pybind11 ≥ 2.12 (see the blocker above).
- Wall-clock pacing is `time.sleep`-based, so on a heavily loaded machine the publisher may fall
  behind (render ticks). The results are for this machine (14 cores, idle).

## Virtual sensor topology (R5)

`examples/virtual_sensor.py` — the 3-process pipeline of Option B in `docs/gpu_rl_design.md`. The policy subscribes to the same
topics in sim and real.

```
 SIM (GL/EGL only, no torch/CUDA)                  VIRTUAL SENSOR (CUDA)                        POLICY (Axon only)
 board scene, N MjData, 1 mujoco.Renderer          DmabufImporter (EGL display, no context)     N TensorSubscriber
 CameraStreamBatch(cfg, N, render=hook) @500 Hz    SimCameraBatch(renderer=False)  (no GL)      500 Hz poll, t0 + k·2 ms + 0.5 ms
 hook: update_scene(env i) -> DmabufExporter       fence wait -> read(slot) -> ingest(i, raw)   trailer -> capture_t, n
       .render(i·Ks + r_i % Ks) -> fence fd        one process() per notification batch        age_tick = k·DT − capture_t
   │  fds × N·Ks once (SCM_RIGHTS)                 hold, publish at max(ready_t, T − DT/2)          │
   ├──────────────── Unix SOCK_SEQPACKET ─────────>│  Axon Accelerator pool (CUDA VMM)              │
   │  NOTE(cam, slot, n, capture_t, ready_t)       │  topic realsensor/head_cam/<i> ──────────────>│
   │  + native fence fd, per fired camera          │  frame | capture_t f64 | n i64                 │
 pub_age.csv (CameraStreamBatch ages: truth)       vs_publish.csv                               sub_age.csv
```

- **Slots**: the exporter has `N·Ks` slots, default `Ks = 4` (`--sim-slots`). Camera i's r-th render goes to
  `flat_slot(i, r, Ks) = i·Ks + r % Ks`. The virtual sensor copies the slot as soon as the notification arrives and keeps it in its own output ring (`ring_depth(cfg)`
  = 6), so the sim side needs no ring of delay depth. **There is no release fence back to the sim**: the sim reuses a slot
  after Ks renders (= after capture_t + Ks/fps), so safety rests on timing alone. The virtual sensor **drops and counts as an "overrun drop"**
  the frame of any notification older than `(Ks−1)/fps` (100 ms at Ks = 4) (the same as a real camera
  dropping frames when the host stalls; `vsensor.overrun_drops` in `summary.json`). Only beyond 4x the slot window (400 ms) is the virtual sensor
  hopelessly behind, and only then does it raise `RuntimeError`. There is no path that silently reads wrong pixels.
- **Notification** `NOTE = struct "<iiqdd"` (32 B): `cam` int32 (−1 = terminate), `slot` int32, `n` int64, `capture_t`,
  `ready_t` float64 (seconds relative to t0). In fence mode, 1 native fence fd is attached to the same message via SCM_RIGHTS.
  `ready_t` comes from `CameraStream.last_render` (a read-only attribute added in R5). The reason the Axon `raw_meta` topic is not used:
  `latest_view` is lossy and returns only the latest 1 message, and the fence fd needs SCM_RIGHTS.
- **Stall**: if the virtual sensor stalls for longer than one frame period, the same camera arrives twice in one drain. If `SimCameraBatch.pending(i)`
  holds, everything before it is `process()`ed first and then it continues (test: `--stall-ms 150` with the default slots: notifications older than 100 ms are
  overrun drops, the rest are flushed early).
- **Publish time**: the raw frame goes straight to the virtual sensor at the capture tick, and effect processing (~0.5 ms) finishes within the 50 ms latency budget.
  Publication happens at `max(ready_t, T − DT/2)` (T = the first control tick after ready_t = the tick `CameraStream` delivers): the policy polls at tick +
  POLL_LAG, so a publication within `(T − DT + POLL_LAG, T + POLL_LAG]` sees exactly the same frame as the sim.
  Publishing at `ready_t` as is made the 25 % of cases where ready_t falls within POLL_LAG just after a tick show up one tick early (tick match rate 0.989).
  It never publishes before ready_t. If a newer frame has already gone out from the same camera, the old frame is dropped.
  **Why this is faithful**: `latency_s` is calibrated **end to end** from mid-exposure on the robot to reception in the policy process,
  so adding the driver→Axon hop latency here once more would double-count it. Transport delay appears only when publication is late, and that is
  caught by the overrun gate below. The KS gate proves the delivery **schedule**, and the tick match rate and p99 latency gates prove the **overrun**.
  **Meaning in sim versus real**: this publish rule reproduces the sim's tick-quantised delivery (a sim artifact: `CameraStream` hands over frames only at control ticks).
  Compared with the continuous-time ready_t, it holds the frame 0–DT/2 (≤ 1 ms) longer, and assumes the policy runs with the same DT and the same
  t0 phase. A real robot's driver publishes at ready_t without quantisation, so on the real robot this rule is not needed
  (the policy just reads the latest frame at its own tick), and the policy-side age distribution matches under the same sample-and-hold rule.
- **Topics**: 1 per camera, `realsensor/head_cam/<i>` — if camera indices were mixed into one topic, `latest_view()` would return only the latest
  message and the policy would miss other cameras' frames. The payload is the same `frame | capture_t | n` as the trailer layout above
  (with `--depth`, float32 depth H·W·4 B follows the frame).
- **Pool**: default `PoolBackend.Accelerator` — the binding exposes `pool.device_array(k)` (`__cuda_array_interface__`) +
  `acquire()`/`publish_device()`, so the D2D copy + 16 B trailer are written straight into the pool buffer with no host copy.
  Builds without `PoolBackend.Accelerator` (AXON_WITH_CUDA off) SKIP. With `--pool custom|udmabuf` it copies to host and calls
  `publish()`, and the policy has no CUDA at all (the test checks this via `sys.modules` and compute-apps).
- **Policy-side CUDA (Accelerator pool)**: the policy process's CUDA context is created not by reading the trailer but by Axon's own
  `accel_import` (`cuDevicePrimaryCtxRetain` when the subscriber imports the VMM fd) — the 260 MiB is that cost.
  The example reads the trailer with a synchronous `cuMemcpyDtoH`, but a real policy running inference on the default stream would stall on it, so
  reading with **`cuMemcpyDtoHAsync` on a non-blocking stream** (pinned 16 B) is recommended. Follow-up: the Axon C++ already has
  a host-side `TensorDescriptor.capture_ts_ns`, but the Python binding (`publish`/`TensorView`) does not expose it.
  Exposing it would deliver the capture time through the descriptor and remove the device read itself (n can be replaced by seqno).
- **API additions**: `SimCameraBatch.ingest(i, rgb, depth=None)` (fills the raw slot + pending from a raw frame rendered externally,
  same slot rule as `render_into`, a wrong shape/dtype/device raises ValueError without writing anything),
  `SimCameraBatch.pending(i)`, `SimCameraBatch(renderer=False)` / `SimCamera(renderer=False)` (post-processing only with no GL context,
  `render` raises ValueError), `CameraStream.last_render`. `tests/test_batch.py::test_batch_ingest`
  pins that the ingest batch == the render_into batch bit for bit (colour, DoF + depth noise + AE/AWB).

Run:

```bash
MUJOCO_GL=egl .venv/bin/python examples/virtual_sensor.py --seconds 5 --n 4 [--depth] [--pool accelerator|custom|udmabuf] \
    [--sim-slots 4] [--stall-ms 0] [--run-name main]   # output: out/virtual_sensor/<run-name>/
MUJOCO_GL=egl .venv/bin/python tests/test_virtual_sensor.py   # SKIP (exit 0) without Axon/cuda-python/EGL extensions
```

### Results (5 s, N=4, 640×480, RTX 5080 / driver 580, 2026-09-28)

GPU condition: another benchmark was using the same GPU at the time — the timing numbers
may include contention.

| | accelerator | custom | udmabuf | accelerator `--depth` |
|---|---|---|---|---|
| KS(age_tick vs sim age) cameras 0–3 / pooled | 0.0000 ×4 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| seqno coverage (per-camera minimum) | 1.000 | 1.000 | 1.000 | 1.000 |
| per-tick frame match rate (gate ≥ 0.99) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| p99(publish − ready_t) (gate < 2 ms) / p99(publish − scheduled) | 1.17 / 0.43 ms | 1.20 ms | 1.20 ms | 1.16 ms |
| virtual sensor process (mean / p99; gate p99 < 50 ms) | 0.55 / 1.29 ms | 0.56 / 0.92 | 0.53 / 0.93 | 0.77 / 1.09 |
| \|Δmean(wall age)\| max | 0.58 ms | | | |
| sim: `render` call (fence, mean / p95) | 0.098 / 0.167 ms | | | |
| sim: ticks more than 1 tick late | 0 / 2500 | 6 / 2500 | 0 / 2500 | 0 / 2500 |
| memory sim (nvidia-smi table, G process) | 190 MiB, not in compute-apps | 190 | 190 | 213 |
| memory virtual sensor (compute-apps) ≤ bound | **506 ≤ 515.3** | 474 ≤ 480.1 | 474 ≤ 480.1 | 578 ≤ 603.2 |
| memory policy (compute-apps) | 260 MiB (Axon accel_import) | none (CUDA-free) | none | 260 |
| run time | 7.4 s | 7.2 s | 7.3 s | 7.2 s |

KS being 0 is the result of aligning the publish time to the control tick: over the 5 s there was not a single tick where the policy saw a different frame from the sim (the earlier version
published right at `ready_t`, giving KS 0.004–0.013 and a tick match rate of 0.989).

The main run on a quiet GPU (no external compute processes at start, util 1 %) after switching to the Ks = 4 default: KS 0.0000 (per camera and pooled),
coverage 1.000, tick match rate 1.0000, p99(publish − ready_t) 1.10 ms, process 0.52 / p99 0.88 ms, memory 506 ≤ 515.3, sim 205 MiB.
As a contrast, in a contended run where another GPU process (12.9 GB, util 96–98 %) occupied the GPU, the tick match rate was
0.970–0.974 and process was 3.9–4.7 ms/batch, so the tick-match-rate gate failed — hence the GPU condition is recorded on every run.

Stall test (5 s, default `--sim-slots 4`, `--stall-ms 150 --pool custom`, quiet GPU): 8 overrun drops, 2 mid-run flushes,
no crash, pooled KS 0.0264, minimum coverage 0.979, tick match rate 0.9707, policy CUDA-free. A stall shifts the mean age by design, so for this run
|Δmean| and overrun gates are not applied.

**GPU condition record**: every run records the external compute processes (`pid, process_name, used_memory`) and GPU utilisation at start and end in
`gpu_at_start`/`gpu_at_end` of `summary.json`, and prints a "GPU contended" warning to stderr if an external process exceeds 4 GB or utilisation exceeds 50 %
(the gates are not skipped). Every gate-failure message in the test carries both conditions.
The output goes to a per-run directory (`out/virtual_sensor/main/`, `.../stall/`), so a stall run does not overwrite the main run's evidence.

**Virtual sensor memory bound**:
`1.1 × (CUDA ctx + chunk buffers + N·(raw slot + output ring + per-camera FPN) + shared FPN + staging + Axon pool)`
= 1.1 × (304 + 101.1 + 4 × 6.53 + 2.34 + 2.9 + 32) = 515.3 MiB (accelerator).

- CUDA ctx: the plan said 270 MiB (docs/gpu_rl_design.md, an earlier measurement), but on this machine today the compute-apps
  row right after `torch.zeros(1)` is 304 MiB. The virtual sensor measures its own row right after the context and uses it in the bound, and the test pins it at ≤ 1.2 × 270.
- Chunk buffers 101.1 = resident working buffers 87.0 (`test_batch.chunk_bytes`) + the (C,3,H,W)
  f32 output 14.1 that `grid_sample` freshly allocates per chunk.
- Per-camera 6.53 = raw slot 764×574×3 + output ring 6×640×480×3. FPN (`sigma_prnu = sigma_dsnu = 0`) is a (1,H,W)
  shared map, so it is not multiplied by N and 2.34 MiB is counted only once.
- Axon pool 32 = 4 cameras × 4 buffers × VMM 2 MiB (the test pins it as `N·POOL_BUFFERS·2`).
- The imported dma-buf slots are sim-owned memory and do not show up in the virtual sensor's row (shrinking the slots from 24 to 8 leaves 506 MiB unchanged)
  → removed from the bound. For reference, `import_mib = slots × 2` (32 at Ks = 4) is recorded.
- The bound terms (`chunk_mib`, `per_cam_mib`, `shared_mib`, `staging_mib`) are computed with **formulas** such as H_r·W_r·3 + K·H·W·3,
  and the virtual sensor asserts they equal the live tensor sizes (so that neither one changes silently).
- Measured 506 ≈ 304 ctx + torch reserved 162 (allocated 118.4, peak 141.8; only the virtual sensor uses `expandable_segments:True`) + pool 32 +
  kernel modules ~8.

- **Role check**: sim `sys.modules` has no `torch`/`cuda.*` + not in compute-apps; policy `sys.modules` has no `OpenGL`/
  `mujoco`/`torch` (with a host pool, no `cuda.*` either, and not in compute-apps).
- **Fail fast**: the children use `barrier.wait(60)`/`ready_vs.wait(60)`, and the parent waits with `multiprocessing.connection.wait` on each child's result
  `Pipe` and the child sentinels together, so if any 1 child finishes before the 3 results arrive, it immediately kills them all and fails.

## Reverse direction: Axon udmabuf as GL render target

Spike `scripts/spike_udmabuf_render_target.py` (2026-09-28, RTX 5080 / driver 580.178.04). The Axon Python binding does not expose
the dma-buf fd of a UDMABUF pool buffer, so a buffer of the same kind (memfd + `F_SEAL_SHRINK` + `/dev/udmabuf`
`UDMABUF_CREATE`) was created directly and tested.

| Step | Result |
|---|---|
| **Control**: import the block-linear dma-buf (modifier `0x300000000e08014`) exported by R4 `DmabufExporter` → `glEGLImageTargetRenderbufferStorageOES` → FBO → blit → `glReadPixels` | **Success**: GL error 0, FBO complete, bit-identical to `Renderer.render()` |
| `UDMABUF_CREATE` (640×480×4 B) | ok |
| `eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, AB24, modifier LINEAR, pitch W·4)` | ok |
| `glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, img)` | **Failure: GL_INVALID_OPERATION (0x502)** |
| (for reference) `glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img)` | Failure: GL_INVALID_OPERATION |

Since the control succeeds through the same call path, the mechanism is correct, and the cause of the failure narrows to LINEAR/system memory.
`eglQueryDmaBufModifiersEXT(AB24)` also reports the 12 NVIDIA block-linear modifiers (`0x3000000006060xx`, `0x300000000e080xx`) as
renderable, while **LINEAR (0x0) is `external_only=True`** (for `GL_TEXTURE_EXTERNAL_OES` sampling only). Since udmabuf is
host memory, only LINEAR is possible, so it cannot be a GL render target. Conclusion: **the SIM→virtual sensor hop cannot be made Axon-native (rendering
directly into a udmabuf pool)** — doing so would need a GPU→host copy (`glReadPixels`/PBO into the mmapped pool buffer), which is worse than the R4
dma-buf path (zero-copy GPU memory). The Axon Accelerator pool is a CUDA VMM fd, so GL cannot import it (`docs/gpu_rl_design.md` Option B).
We keep the current configuration (sim→virtual sensor via R4 dma-buf + Unix socket, virtual sensor→policy via Axon).
