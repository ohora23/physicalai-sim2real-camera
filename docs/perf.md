# Performance notes (optimization pass, 2026-09-27)

640×480 output, Phase 1 effects on (dist, vignette, FPN, shot/read noise, γ 2.2), board scene,
RTX 5080 + EGL, mujoco 3.14, cv2 5.0. Median ms/frame, 100 frames after warm-up.
Script: `tests/test_temporal.py --bench`, `tests/test_camera.py --bench` (numbers are printed, never asserted).

## What was measured

cProfile of a window render (N=4, B=8, before): 32 raw renders were ~83% of the frame;
`_render_window` bookkeeping (32× full-frame add, 8× fill, 8× scale) ~3.3 ms; 32 full-frame
`cv2.LUT` ~2.0 ms. Single-instant post-processing ~1.5 ms, of which `cv2.randn` 1.2 ms.

Render cost depends on whether the GPU is shared. With another process holding the GPU
(a local LLM at 98% utilization) every back-to-back `mjr_readPixels` waits ~2.3 ms for a
time slice, independent of MSAA/shadow size; on an idle GPU a render is ~0.5 ms.
Both regimes are reported.

## Before / after

| case | idle GPU before | idle GPU after | shared GPU before | shared GPU after | renders |
|---|---|---|---|---|---|
| single instant | 3.15 | 3.15 | 3.27 | 3.27 | 1 |
| window N4 B1 (8 ms exp.) | 4.8 | 4.8 | 10.5 | 10.4 | 4 → 4 |
| window N1 B8 (16 ms readout) | 7.7 | 6.0 | 18.2 | 18.2 | 8 → 8 |
| window N4 B8 (8/16 ms) | 19.6 | 8.4 | 73.4 | 25.2 | 32 → 11 |
| window N4 B8 (10/20 ms) | 20.0 | 8.4 | 73.6 | 25.2 | 32 → 11 |
| DoF on (A=0.016, CoC ≈ 9 px) | 5.73 | 5.77 | 8.0 | 8.0 | 2 |
| `dof()` alone, CoC ≈ 32 px | 6.53 | 3.50 | – | – | – |
| depth_noise (depth=True) | 6.08 | 6.11 | 8.4 | 8.5 | 2 |

## What changed

- **Grid-snap render cache** (`SimCamera._plan_window`, `_render_window`): sample times
  `t_bi` are snapped to the grid `δ = min(exposure_s/N, readout_s/B)` (non-zero terms),
  anchored at the first sample. Each distinct grid time is rendered once, in time order, and
  shared by every band/sample that lands on it. Each band accumulates only the source rows its
  own remap reads (row range computed from the distortion map at init; map_y shifted by an
  integer, so the remap is bit-identical). Memory is about one frame, so no cache cap or
  fallback is needed. The first sample goes straight into the band accumulator, so there is
  no `fill`. With no distortion, bands accumulate directly into the output rows.
  When `exposure_s/N` and `readout_s/B` are multiples of δ the output is **bit-identical** to
  the previous per-band loop (checked on a moving scene for 8/16 and 10/20 ms, N=4, B=8, with and without
  distortion).
- **DoF half-res blur** (`optics.dof`): cascade steps with σ > 4 px blur a half-resolution copy
  (INTER_AREA down, INTER_LINEAR up, ~1 px² of added variance subtracted). σ error ≤ 1.5% on a
  step edge; |Δ| ≤ 0.023 linear vs full-res on the P4 checker at CoC 32 px; 4σ vs CoC −0.0%
  (new test case, A = 0.055). The P4 case (CoC 9 px, σ ≤ 2) does not take this path.

## Caveat: grid-snap time error

Per-sample time error is ≤ δ/2. It is 0 when `exposure_s/N == readout_s/B` (or one is a multiple
of the other). The actual maximum is `SimCamera._snap_err`, printed by the bench and
`test_grid_snap`. Example: 6 ms/N=4 with 16 ms/B=8 gives δ = 1.5 ms and 0.5 ms error (13 renders).

## N·B guidance

Cost ≈ renders × (0.5 ms idle GPU, ~2.3 ms shared GPU) + ~0.2 ms/render (ring load + LUT/add)
+ ~2.5 ms fixed post-processing. With `exposure_s/N == readout_s/B`, renders = N + B − 1 instead
of N·B (N=4, B=8 → 11). Choose N and B like that, and keep renders ≤ 8 for an RL-rate budget
(~17 ms on a shared GPU). The init warning counts snapped renders.

## Tried and rejected (numbers from this machine)

- `Renderer.render(out=)` is already used. Inside, mujoco still does `out[:] = np.flipud(out)`,
  a 1.3 MB temporary taking 0.06 ms per render. Avoiding it means calling `mjr_render`/`mjr_readPixels` on
  private `Renderer` attributes plus `cv2.flip` (0.018 ms). That saves ≤ 0.04 ms per render and
  ties us to mujoco internals, so it was not done.
- Accumulate: `acc += lin` 0.072 ms, `cv2.add` 0.073, `cv2.accumulate` 0.081, `np.add(out=)`
  0.092 (full frame). Kept `+=`. Scale: `*=` 0.035 vs `cv2.multiply(scale=)` 0.32.
- Depth `render(out=)`: 0.57 → 0.53 ms. mujoco still allocates float64 temporaries (~11 MB
  peak) for the NDC→metric conversion, and without distortion the raw depth is handed to the caller, so it must
  be a fresh array. Not done. The NEAREST depth remap also stays fresh (returned to the caller).
- Depth render moved after the RGB post-processing (to overlap the readback wait): 0.2–1 ms
  saved only on a shared GPU, 0 on an idle one. Dropped to keep one render call.
- Depth RNG: `standard_normal(float32)` 2.2 ms (numpy ziggurat, ~7 ns/sample); `cv2.randn`
  0.39 ms but uses cv2's global stream (would couple depth to the RGB noise); `cv2.RNG` is not
  exposed in the Python bindings; SFC64 1.97 ms changes the stream for −10%. Kept.
- `cv2.randn` (RGB, 1.2 ms) kept as required. tracemalloc (≥ 1 MB per render) finds only the
  intentional fresh outputs in our code: `encode` uint8 and the depth remap. The rest is
  mujoco-internal (flipud, depth float64).

## Remaining hot spots (why left)

- Raw renders: GPU/driver bound. Snapping cuts the count; the per-render cost is mujoco's.
- `cv2.randn` 1.2 ms + `cv2.pow` 0.6 ms in `encode`/`add_noise`: required RNG, already SIMD.
- Depth noise normal draw 2.2 ms: see above (separate numpy stream per spec).
- DoF at small CoC: 3 full-res GaussianBlurs (σ ≤ 4), ~1.7 ms.

## GPU backend (`device="cuda"`, `physicalai_sim2real_camera/gpu.py`)

`SimCamera(model, cam, cfg, device="cuda")` (needs `pip install 'physicalai-sim2real-camera[gpu]'`;
tested with torch 2.11.0+cu128) runs everything after the render on the GPU. MuJoCo renders
straight into pinned host buffers; one `non_blocking` upload (1.3 MB margin render), LUT
(`index_select`), remap (`grid_sample`, bilinear, `align_corners=True`), DoF, gain, AE metering,
noise, Bayer and encode in CHW float32, then one download (`render(...)` → numpy) or none
(`render(..., as_tensor=True)` → fresh CUDA uint8 (H,W,3) [+ float32 depth]). The CPU path is
unchanged and the default; the identity config stays the plain CPU render on any device.
Window path: same grid-snap plan, each sub-render uploaded once, bands accumulate on the GPU
and each band remaps only its own rows (cheaper than B full-frame remaps).

Parity vs CPU (`tests/test_gpu.py`, 640×480): Phase 1 noise off max 1 DN / mean 0.0000 DN;
DoF (A=0.016, CoC ≈ 9 px) max 1 DN; window N4B8 max 1 DN; AE ev after 30 frames |Δ| 3e-8;
depth remap bit-equal; Bayer interior max 1 DN (2-px border up to 7 DN: normalised bilinear
demosaic vs cv2's border rule). cv2 5.0 `remap` samples float32 images at exact float coords
(no 1/32 table: quantising the grid made it 5 DN worse). Measured, not assumed.

Differences: noise uses a per-camera CUDA `torch.Generator` (`SeedSequence([seed+cam, 1])`,
depth `[seed+cam, 2]`), statistically equivalent to cv2/numpy but not bit-equal (PTC slope/intercept
within 1%); deterministic run-to-run. DoF blurs every cascade step at full res (no half-res
trick; differs from CPU only when a step σ > 4 px, CoC > ~18 px). Convs run with TF32 off.
`grid_sample`/`conv2d` temporaries come from torch's caching allocator (no `out=`).

Idle GPU (0 %, no other process), ms/frame, `tests/test_gpu.py --bench`:

| case | CPU | GPU → numpy | GPU → tensor |
|---|---|---|---|
| single instant (Phase 1) | 3.15 | 0.67 | 0.53 |
| + DoF | 5.69 | 1.66 | 1.50 |
| + Bayer | 3.61 | 0.71 | 0.58 |
| + depth noise (depth=True) | 6.04 | 1.41 | 1.13 |
| window N4 B8 (8/16 ms, 11 renders) | 8.52 | 5.43 | 5.30 |

Plain render at margin size 0.31 ms, upload 0.03 ms, download 0.06 ms, so single-instant
post-processing is ~0.35 ms on the GPU (vs ~2.8 ms CPU). The window path is now render bound
(11 × render + ring loads).

## Stage 3: readback removal (`interop=True`, `physicalai_sim2real_camera/interop.py`)

`SimCamera(model, cam, cfg, device="cuda", interop=True)` copies each colour render
GPU→GPU instead of `mjr_readPixels` (GPU→CPU) + pinned upload (CPU→GPU). Needs
`cuda-python` (now in the `gpu` extra; tested cuda-bindings 13.4.3, driver 580 / CUDA 13).

Mechanism (verified under `MUJOCO_GL=egl`, RTX 5080): with `model.vis.quality.offsamples = 0`
set before the `Renderer` is created, `MjrContext.offColor` is a single-sample
`GL_RENDERBUFFER` with internal format `GL_RGBA8` (`offSamples == 0`, `offFBO_r` unused).
Once: `cuGraphicsGLRegisterImage(offColor, GL_RENDERBUFFER, READ_ONLY)` in torch's primary
CUDA context with MuJoCo's EGL context current. Per render: `mjr_render` →
`cuGraphicsMapResources` (orders after the GL draw) → `cuGraphicsSubResourceGetMappedArray` →
`cuMemcpy2DAsync` (array → device (H,W,4) buffer) → unmap → `copy_(buf[..., :3].flip(0))`
into the pipeline's (H,W,3) uint8 tensor (GL is bottom-up; `Renderer.render` flips too), all
on torch's current stream. Used for every colour sub-render (single instant and window path).

Depth (`GlInterop.depth_into`): the offscreen depth renderbuffer is `GL_DEPTH32F_STENCIL8`,
which `cuGraphicsGLRegisterImage` rejects (`CUDA_ERROR_INVALID_VALUE`), so the depth render is
read with `glReadPixels(GL_DEPTH_COMPONENT, GL_FLOAT)` into a pixel-pack buffer (a GPU→GPU
copy, no CPU stall) registered once with `cuGraphicsGLRegisterBuffer`, then map →
`cuMemcpyDtoDAsync` → unmap → flip → the same reversed-Z→metres conversion as
`Renderer.render` (float64 with float32-rounded frustum coefficients). Verified: the raw PBO
read equals `mjr_readPixels` bit-for-bit, and the converted depth equals `Renderer.render()`
bit-for-bit (20 moved frames), so DoF and depth noise outputs are bit-equal with/without
interop.

Correctness (`tests/test_gpu.py`): the interop copy equals `Renderer.render()` of an
offsamples=0 renderer bit-exactly over 20 frames with the board moved; interop vs
non-interop `device="cuda"` (same model, noise on, same seed) bit-equal for single instant,
depth=True and window N4B8; vs CPU (noise off) max 1 DN; identity config == plain render.

Timing (`tests/test_gpu.py --bench`, 640×480 output, 764×574 margin render). **Measured with
the GPU shared** (another process at 98 % util, 12.7 GB; never idle during this run), so the
absolute numbers are inflated and noisy; two back-to-back runs, ms/frame:

| case | readPixels + upload | interop |
|---|---|---|
| render + read (margin size) | 2.06 / 1.97 | 1.40 / 1.36 |
| single instant → tensor | 3.82 / 2.05 | 1.59 / 1.50 |
| window N4B8 → tensor (11 renders) | 29.8 / 34.9 | 4.5 / 15.5 |

(Spike, 808×606, same load: 1.35–1.85 vs 0.97–1.36 ms.) Interop was never slower. Under
contention it wins big because `glReadPixels` is a blocking CPU↔GPU round trip per sub-render,
while the interop path stays asynchronous on the GPU.

**Idle GPU** (2026-09-27, every other GPU process stopped; 548 MiB / 0 % before the run; `out/bench_idle_gpu.txt`), ms/frame:

| case | readPixels + upload | interop | ratio |
|---|---|---|---|
| render + read (margin size) | 0.306 | 0.194 | 1.6× |
| single instant → tensor | 0.54 | 0.30 | 1.8× |
| window N4B8 → tensor (11 renders) | 5.11 | 2.55 | 2.0× |
| single instant + depth → tensor (2 renders) | 1.38 | 0.54 | 2.6× |

The idle win is smaller in absolute terms but still ~2× in window mode: each `glReadPixels`
costs a GL flush + ~0.1 ms of blocked CPU per sub-render, which the async device copy avoids.
Same run, non-interop reference (cpu / gpu / gpu tensor): single 3.22 / 0.68 / 0.55, window
N4B8 8.59 / 5.49 / 5.34 ms. So the full RL path is now 3.22 → 0.30 ms single and
8.59 → 2.55 ms window (N4B8) from the CPU baseline.

Caveats:
- **MSAA off**: interop needs a single-sample FBO, so `interop=True` sets
  `model.vis.quality.offsamples = 0` (WARNING once, not restored); every Renderer created from
  that model afterwards renders without MSAA (edges alias; images differ from an offsamples=4
  render). Compare interop output only against offsamples=0 renders.
- GL context: `render_into` makes the renderer's EGL context current before `mjr_render`, so
  several cameras/renderers per process are fine. `SimCamera.close()` unregisters the resource
  before the renderer's GL context is freed.
- Fallback: if `cuda-python` is missing or registration fails (e.g. a GL driver without CUDA
  interop), one `WARNING: CUDA-GL interop unavailable (...)` and the camera keeps the
  readPixels + upload path (MSAA stays off since the model was already changed).
- Identity configs and `device="cpu"` ignore `interop` (plain render, MSAA untouched).
- Depth via interop still issues one `glReadPixels`, but into a GL buffer: the GL driver
  schedules it on the GPU and the CPU never waits for the data (the 1.38 → 0.54 ms row above
  is single + depth, i.e. two renders; the depth readback alone was ~0.8 ms of blocked CPU).

> Shared-GPU numbers in this section were taken with another process holding the GPU at 98 %; idle-GPU numbers are in the table above.

## Stage 4: batched pipeline (`SimCameraBatch`, `physicalai_sim2real_camera/batch.py`)

`SimCameraBatch(model, cam, cfg, n, chunk=16)` renders n cameras on one shared `mujoco.Renderer`
(`render_into(i, data_i)`: GL + `cuMemcpy` into camera i's raw slot, 0 CUDA kernels) and
post-processes them in chunks of 16 on shared `(16,3,H,W)` f32 work buffers (`process()`), writing
into a per-camera output ring `rgb (n, ring, H, W, 3)`. Bench: `scripts/bench_batch.py` →
`out/batch_bench.csv` (`tests/test_batch.py --bench` calls it). Config: board scene, Phase 1
effects + shot/read noise, 30 fps / 50 ms latency → `ring_depth(cfg) = 4`, interop on (asserted),
3 warm-up + 50 timed steps per method. Each case is timed two ways, for the batch and the
single-camera reference alike:

- **serialised**: batch = `render_into` × n, sync, `process`, sync (the two halves are the
  `render_into`/`process` columns, their sum `serial ms`); single = one `render`, sync, per frame.
- **pipelined**: the same loop with one sync at the end, so the CPU/GL work of the next step can
  overlap the GPU post-processing of the previous one (`pipelined ms`; ms/camera = pipelined / n).

Memory is `torch.cuda.memory_allocated` / `max_memory_allocated` after one step, relative to
before construction (torch only: the shared Renderer's GL memory is not in it). Reference rows:
one `SimCamera(device="cuda", interop=True)` → tensor.

GPU condition (2026-09-30, after the pinned-upload change below): RTX 5080, no other GPU process
before and after, `nvidia-smi` 431 MiB / 1 % before. The script runs the largest cases first (an
earlier run lost the 1280×720 depth n=64 cases to OOM when another process loaded a 13 GB model
mid-run). ms, ring = 4 (ring = 1 rows in the CSV):

| n | resolution | depth | ring | render_into ms | process ms | serial ms | pipelined ms | pipelined ms/cam | × single pipelined | × single serial | alloc MiB | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (SimCamera) | 640×480 | no | – | – | – | 0.34 | 0.29 | 0.294 | 1.00 | 1.00 | 42.9 | 46.4 |
| 1 | 640×480 | no | 4 | 0.31 | 0.22 | 0.53 | 0.34 | 0.340 | 1.16 | 1.58 | 36.4 | 41.6 |
| 16 | 640×480 | no | 4 | 3.13 | 2.60 | 5.74 | 5.63 | 0.352 | 1.20 | 1.07 | 457.9 | 578.9 |
| 64 | 640×480 | no | 4 | 11.99 | 10.14 | 22.13 | 22.05 | 0.345 | 1.17 | 1.03 | 823.4 | 944.4 |
| 1 (SimCamera) | 640×480 | yes | – | – | – | 0.61 | 0.50 | 0.502 | 1.00 | 1.00 | 42.9 | 56.3 |
| 1 | 640×480 | yes | 4 | 0.48 | 0.27 | 0.75 | 0.50 | 0.503 | 1.00 | 1.23 | 45.9 | 61.7 |
| 16 | 640×480 | yes | 4 | 5.83 | 3.23 | 9.06 | 8.98 | 0.561 | 1.12 | 0.93 | 606.2 | 848.5 |
| 64 | 640×480 | yes | 4 | 23.08 | 12.94 | 36.02 | 35.75 | 0.559 | 1.11 | 0.92 | 1276.9 | 1519.2 |
| 1 (SimCamera) | 1280×720 | no | – | – | – | 0.49 | 0.44 | 0.438 | 1.00 | 1.00 | 137.5 | 148.1 |
| 1 | 1280×720 | no | 4 | 0.34 | 0.34 | 0.68 | 0.45 | 0.450 | 1.03 | 1.38 | 116.6 | 133.1 |
| 16 | 1280×720 | no | 4 | 3.47 | 9.43 | 12.90 | 12.71 | 0.795 | 1.81 | 1.64 | 1461.6 | 1835.4 |
| 64 | 1280×720 | no | 4 | 13.34 | 37.38 | 50.72 | 50.57 | 0.790 | 1.80 | 1.61 | 2585.9 | 2959.6 |
| 1 (SimCamera) | 1280×720 | yes | – | – | – | 0.85 | 0.74 | 0.745 | 1.00 | 1.00 | 137.5 | 185.1 |
| 1 | 1280×720 | yes | 4 | 0.56 | 0.36 | 0.92 | 0.75 | 0.754 | 1.01 | 1.09 | 145.9 | 199.7 |
| 16 | 1280×720 | yes | 4 | 7.06 | 12.38 | 19.44 | 19.41 | 1.213 | 1.63 | 1.44 | 1928.8 | 2761.3 |
| 64 | 1280×720 | yes | 4 | 27.41 | 48.39 | 75.80 | 75.35 | 1.177 | 1.58 | 1.40 | 4004.6 | 4837.0 |

(× single = batch ms/camera ÷ single-camera ms/frame at the same resolution and depth flag, with
the same timing method.)

**Time: batching still does not buy speed per camera; `process()` no longer blocks the host.**
Pipelined vs pipelined — the fair comparison, since both loops then overlap whatever they can —
every batch case with n ≥ 16 is 1.1–1.8× slower per camera than a single `SimCamera` (640×480
n=64: 0.345 vs 0.294 ms colour, 0.559 vs 0.502 depth; 1280×720: 1.8× colour, 1.6× depth), while
n = 1 is now within 1.00–1.16× of it (0.34 vs 0.29, 0.50 vs 0.50, 0.45 vs 0.44, 0.75 vs 0.75 ms).

Until 2026-09-30 `process()` was host-synchronous: each chunk uploaded its camera/slot indices
(and, for non-contiguous chunks, its row gather index) with a pageable `torch.from_numpy(a).to(dev)`,
which PyTorch completes with a `cudaStreamSynchronize`. torch.profiler counted 8 syncs in one
`process()` at n=64 (4 chunks × `ix`/`sl`) and 6 for 32 non-contiguous cameras (2 chunks ×
`rows`/`ix`/`sl`). `batch.to_device` now stages those arrays in pinned memory from PyTorch's
caching host allocator and copies them `non_blocking`: 0 syncs in both cases
(`tests/test_batch.py::test_batch_no_sync` runs `process()` under
`torch.cuda.set_sync_debug_mode("error")`), and the host time of one `process()` call at n=64
drops from 10.1 ms to 1.1 ms (5.2 → 0.65 ms for the 32 gathered cameras). AE/AWB metering keeps one sync per
chunk by design (`.tolist()` of the means); DoF keeps its grouping sync plus, per blur group, a pageable
upload of the group index and of the three Gaussian kernels (1 + 4 syncs per group; not changed here). Before → after, pipelined ms:

| case | before (2026-09-28) | after |
|---|---|---|
| n=1 640×480 colour | 0.54 | 0.34 |
| n=1 640×480 depth | 0.65 | 0.50 |
| n=1 1280×720 colour | 0.67 | 0.45 |
| n=1 1280×720 depth | 0.93 | 0.75 |
| n=64 640×480 colour | 22.31 | 22.05 |
| n=64 1280×720 colour | 51.73 | 50.57 |

That removed the n = 1 overhead, where the post-processing's GPU time is small next to its launch
overhead, but not the n ≥ 16 one: serial ≈ pipelined within ~2 % there, so GL rendering and
post-processing still alternate. Likely cause (not profiled): every camera renders into the one
shared renderbuffer registered with CUDA, and its copies are queued on torch's stream behind the
previous step's `process()` kernels. `cuGraphicsUnmapResources` makes later GL work wait for all
earlier work on that stream, so the second `render_into` of step k+1 waits for `process(k)` to finish
on the GPU. The side-stream fix was tried and rejected (A/B below): interleaving GL renders with queued
CUDA kernels costs ~2 ms per render, consistent with GL/CUDA context time-slicing (inferred, not traced). Serialised vs serialised the batch
ties or wins slightly at 640×480 (1.03× colour, 0.92× depth, n=64) and still loses at 1280×720
(1.4–1.6×). `render_into` costs what a single render + interop copy costs (0.19 ms/camera colour,
0.36 depth at 640×480; sequential GL work either way), and `process` is dearer per camera than the
single pipeline's post-processing (0.16 ms vs ~0.10 ms at 640×480 — the single figure is derived
by subtraction, 0.29 total − 0.19 render + copy from Stage 3, not measured; 0.58 ms at 1280×720).
What the batch buys is memory (below) and a constant launch count.

**Process is bandwidth bound.** Per chunk, `process` makes an estimated ≈ 17 full passes over the
`(16,3,H,W)` f32 chunk tensor (59 MB at 640×480, 177 MB at 1280×720) — an estimate from the
19 kernels per chunk, several of which (LUT, index copy, remap input) run at the margin render
size (84 MB at 764×574), not a profile. 640×480 colour: 10.14 ms / 4 chunks = 2.54 ms for
≈ 1.0 GB of passes, ≈ 0.39 TB/s counting each pass once (≈ 0.8 TB/s if each reads and writes),
against the RTX 5080's 0.96 TB/s. It scales with pixels, not launches: 1280×720 is 3.0× the
pixels and 3.8× the process time. A single 640×480 frame (3.7 MB f32) stays in the 64 MB L2
between passes while a 59 MB chunk does not — the likely reason (not profiled) that per-camera
post-processing is cheaper unbatched. Going below 16 ms at n=64 needs kernel fusion (which must
also go into `gpu.py` to keep the bit-equality with `SimCamera`), or a smaller chunk; neither is
done here.

**Launch counts (torch.profiler, 640×480, chunk 16).** `render_into`: 0 CUDA kernels.
`process`: 19 kernels per chunk (18 before the output ring added one cast + `index_put_`) plus
one `normal_` per camera (per-camera Philox generators are required for bit-equality with
`SimCamera`); n=64 / n=16 = 4.00 (4 chunks / 1 chunk), n=64 / n=8 = 5.19 (n=8 is a half chunk).

**Ring memory cost.** The output ring is `ring × H × W × 3` B per camera (+ `ring × H × W × 4` with
depth); ring = 4 vs 1 at n=64:

| resolution | depth | ring 1 MiB | ring 4 MiB | cost of 3 extra slots |
|---|---|---|---|---|
| 640×480 | no | 653.6 | 823.4 | 169.8 MiB (2.65 MiB/camera) |
| 640×480 | yes | 883.1 | 1276.9 | 393.8 MiB (6.15 MiB/camera) |
| 1280×720 | no | 2078.6 | 2585.9 | 507.3 MiB (7.9 MiB/camera) |
| 1280×720 | yes | 2823.4 | 4004.6 | 1181.2 MiB (18.5 MiB/camera) |

Timing is unaffected (ring 1 and 4 within ~5 %). ring = 1 is only safe when every frame is
consumed before the camera renders again (see the R1 ring tests).

**Depth-enabled n=64 vs the R1 memory bound.** R1 Accept 3 bounds colour-only memory by
`1.1 × (chunk buffers + n × (7 B/px at render size + ring × 3 + 4 B/px at output size))`
(`bench_batch.r1_mem_bound`). With depth on, n=64 **exceeds** it: 640×480 allocates
1276.9 MiB vs a 957.3 MiB bound (+33 %), 1280×720 4004.6 vs 3061.3 MiB (+31 %). Colour-only
stays within (823.4 ≤ 907.2, 2585.9 ≤ 2897.7 MiB at ring 4). Depth adds a raw Z slot
(4 B/px at render size, 1.67 MiB at 764×574) and a depth ring (`ring × 4` B/px, 4.69 MiB at
ring 4) per camera — 6.4 MiB/camera, not the ~2.8 MiB the R1 reviewer estimated for ring 1.
Adding those two terms (`r1_mem_bound(..., depth=True)`) bounds it: 1276.9 ≤ 1405.1 and
4004.6 ≤ 4458.2 MiB.

**Memory per camera, before/after renderer sharing.** Before R0 every camera had its own
Renderer: 137.5 MiB/camera at 640×480 by `nvidia-smi` (≈ 94 MiB GL + 44 MiB torch; 256 MiB at
1280×720). With the batch the GL side is one renderer for all cameras. At n=64, ring 4:

| resolution | depth | marginal (n=16 → 64) | amortised ((alloc + 209 GL) / 64) |
|---|---|---|---|
| 640×480 | no | 7.6 MiB/camera | 16.1 MiB/camera |
| 640×480 | yes | 14.0 | 23.2 |
| 1280×720 | no | 23.4 | 43.7 |
| 1280×720 | yes | 43.2 | 65.8 |

The marginal figure excludes the fixed ~340–1240 MiB of chunk buffers; the amortised one includes
them and the GL renderer (209 MiB is the EGL context + renderer from the 640×480 / 900×700
budget run in the design note, not measured on this bench's renderer, so the amortised figures
underestimate GL at 1280×720 (209 MiB constant is from the 900×700 run)).

### Side-stream A/B (2026-10-02, reverted)

Hypothesis: issue the raw interop copies (map → `cuMemcpy2DAsync` → unmap) on a side CUDA stream with
per-chunk "slot read" events, so the unmap no longer orders the next GL render after `process(k)` and step
k+1's renders overlap it. Implemented (`GlInterop.copy_color/copy_depth(stream=)`, `SimCameraBatch._cs`/`_rev`),
verified bit-equal to the serial reference under forced contention (negative controls with a host-released
`cuStreamWaitValue32` gate: 96 (wait_event) and 128 (wait_stream) of 128 slots differ when the respective
wait is removed), then measured with `scripts/bench_batch.py --repeat 5` on both builds (median of 5, ring K,
quiet GPU, driver 595.91.07 open; the A run used an earlier build whose bench had a
`--no-side-stream` toggle and a `side_stream` CSV column):

| case | serial ms (A = side stream / B = current stream) | pipelined ms B | pipelined ms A | A pipelined / serial |
|---|---|---|---|---|
| n=1 640×480 colour | 0.54 / 0.52 | 0.34 | 0.37 | 0.67 (1.06 × B) |
| n=16 640×480 colour | 5.70 / 5.71 | 5.60 | **36.2** | 6.36 |
| n=64 640×480 colour | 21.9 / 22.0 | 21.9 | **144.9** | 6.63 |
| n=1 640×480 depth | 0.78 / 0.76 | 0.48 | 4.46 | 5.74 |
| n=64 640×480 depth | 35.0 / 35.3 | 34.9 | 285.9 | 8.16 |
| n=16 1280×720 colour | 12.8 / 12.8 | 12.6 | 36.8 | 2.88 |
| n=64 1280×720 colour | 50.2 / 50.3 | 50.2 | 147.3 | 2.94 |
| n=64 1280×720 depth | 74.8 / 75.0 | 74.6 | 291.0 | 3.89 |

Verdict (rule: keep only if n=64 colour pipelined ≤ 0.85 × serial and n=1 colour and depth ≤ 1.05 × B):
**reverted** (n=64 colour 6.6×, n=1 colour 1.06×, n=1 depth 9.3×). For n ≥ 16 the pipelined side-stream run
costs ≈ 2.26 ms per camera in total (colour; B: 0.34, so +1.9 ms) or ≈ 4.5 ms (depth, two GL renders per
camera; +3.9 ms) regardless of resolution; n=1 640×480 colour is the one case that stays cheap (0.37 ms).
Likely mechanism (inferred from a micro-test on the same machine, not traced with nsys — the plan's optional
nsys step was skipped because the bench and micro-test were decisive): a GL render + copy issued on a side
stream while the current stream holds a long kernel completes in 2.5 ms instead of 0.28 ms on an idle GPU, and
32 renders issued under a spinning kernel progressed at ~2 ms each, which fits GL and CUDA being separate GPU
contexts that are time-sliced rather than run concurrently (each GL render issued while CUDA kernels are queued
waits for a context switch of ~2 ms). With everything on the current stream the unmap makes the GL renders of
step k+1 wait for `process(k)` to drain, after which the renders run back-to-back with no CUDA work in flight —
the "alternation" is the cheap schedule. What is
kept: `bench_batch.py --repeat N` (median of N runs). Pipelining across steps therefore cannot come from
stream tricks; it would need the GL work and the CUDA work to run in different processes (the R5 GL-only sim +
virtual sensor already does that) or a longer GPU time slice.

## Stage 5: window mode in the batch path (motion blur, rolling shutter)

`SimCameraBatch` now accepts `exposure_s`/`readout_s` > 0. Each camera has a `StateRing` (`b.rings[i]`);
`render_into` renders `SimCamera`'s grid-snapped sub-frame plan from it (N4B8: 4 exposure samples × 8
readout bands = 32 sample times, snapped to 11 renders) and accumulates each sub-frame at once into a
per-camera output-space linear frame `_wlin[i]` (`GpuPipeline.window`'s LUT / band accumulate / band remap),
so camera i stays bit-equal to `SimCamera(seed+i, cuda)`. `process()` starts from `_wlin`. The R5 virtual
sensor carries the same windows across processes: the GL-only sim renders each capture as a group of 11
colour sub-frames (+1 depth render at `capture_t`) into consecutive dma-buf slots, and the virtual sensor
ingests the `(11, H_r, W_r, 3)` stack.

Bench: `scripts/bench_batch.py --window` → `out/batch_bench_window.csv`, 640×480 colour, N4B8
(`exposure_s` 8 ms, `readout_s` 16 ms), config otherwise as Stage 4; every timed step includes one `mj_step`
and the ring push on both sides. GPU condition (2026-09-30): RTX 5080, no other GPU process, 313 MiB / 1 %
before, one CPU core offline:

| n | window | ring | render_into ms | process ms | serial ms | pipelined ms | pipelined ms/cam | × single pipelined | alloc MiB |
|---|---|---|---|---|---|---|---|---|---|
| 1 (SimCamera) | N4B8 | – | – | – | 2.59 | 2.56 | 2.557 | 1.00 | 51.1 |
| 1 | N4B8 | 4 | 2.62 | 0.19 | 2.82 | 2.57 | 2.565 | 1.00 | 42.2 |
| 16 | N4B8 | 4 | 39.87 | 1.77 | 41.64 | 41.36 | 2.585 | 1.01 | 288.1 |
| 64 | N4B8 | 4 | 158.43 | 6.79 | 165.22 | 165.36 | 2.584 | 1.01 | 741.6 |

**Time is render bound and equal to the single camera.** 2.58 ms per camera per frame is 11 sub-renders at
~0.23 ms each (GL render + interop copy + LUT/accumulate kernels); `process` is 0.11 ms per camera, 4 % of
the frame, because the LUT and remap already happened per sub-frame. Batching neither helps nor hurts time
(1.00–1.01× the single camera), unlike the single-instant batch of Stage 4, where the chunked post-processing
was 1.1–1.8× slower per camera than the unbatched pipeline: here the per-camera work is the renders, which
are sequential GL either way. At 30 fps a camera needs 2.58 ms per 33.3 ms frame, so one GL context renders
~12 N4B8 cameras in real time; RL training is not wall-clock bound, so this is a throughput figure, not a limit.

**Memory.** Marginal cost n=16 → 64: (741.6 − 288.1) / 48 = 9.45 MiB per camera, = `_wlin` 3.52 MiB
(12 B/px, output size) + output ring 3.52 MiB (4 × 3 B/px) + per-camera FPN 2.34 MiB (8 B/px), 9.38 MiB
by formula. The single-instant batch's raw colour slot (1.75 MiB at the 764×574 render size) and chunk
LUT/remap buffers are not allocated in window mode. A single windowed `SimCamera` costs 51.1 MiB of torch
memory plus its own GL renderer (92–137 MiB, Phase 4 budget), so 64 windowed cameras on one renderer save
~8–11 GB (64 × (51 + 92…137) MiB − 742 MiB) against 64 `SimCamera`s.

**Virtual sensor (R5) real-time capacity.** In the three-process run the sim renders a group of 11 renders
in 0.53 ms mean (0.86 ms p99) at N=4; with every camera firing on the same tick (shared `phase_s`) N groups
must fit in the 2 ms control tick. N=4 passes every R5 gate (KS ≤ 0.002, tick match 0.9998, p99 publish − ready_t 1.90 ms of the 2 ms gate, late ticks
1/2500; measured with the child processes untraced: under `make cov`'s subprocess tracing its p99 processing rises
3 → 13 ms, so that strict run strips `COVERAGE_PROCESS_CONFIG`);
N=8 (4.4 ms of renders on the firing tick) still delivers the right frames (KS < 0.05, coverage ≥ 0.9) but is
late on those ticks (late ticks 47/1000, p99 publish − ready_t 13 ms). Training does not run at wall-clock
pace, so this only matters for the real-time virtual sensor.

**Staggered phases (`--stagger`, 2026-09-30).** `CameraStreamBatch(phase_s=stagger_phases(cfg, N))`
gives camera i `phase_s = i / (N·fps)`, so the capture instants form a lattice of spacing `1/(N·fps)` and at most
`⌈DT·N·fps⌉` cameras fire on one 2 ms tick: 1 for N ≤ 16 at 30 fps (`tests/test_timing.py::test_stagger_max_fired`
checks 8, 16 → 1 and 17 → 2 without a GPU). Measured (single runs of 2026-09-30, untraced, uncontended GPU; a
second run of each gave N=8: late ticks 4/2500, p99 1.84 ms, tick match 0.9998 and N=16: p99 3.03 ms, so the N=8
strict margin under the 2 ms gate is only 0.16–0.24 ms):

| N | max fired / tick | group ms mean / p99 | late ticks | p99 publish − ready_t | tick match | vsensor process ms / note | policy poll p99 |
|---|---|---|---|---|---|---|---|
| 8, shared phase | 8 | 0.55 / — | 47 / 1000 | 13 ms | — | — | — |
| 8, staggered | 1 | 0.63 / 1.03 | 3 / 2500 | **1.76 ms** | 1.0000 | 1.02 | 0.10 ms |
| 16, staggered | 1 | 0.61 / 1.11 | 4 / 1500 | 2.88 ms | 0.9963 | 1.04 | 0.19 ms |

N=8 now passes every strict R5 gate (the 352-descriptor transfer still goes in two messages). N=16 keeps the sim
on time (one 11-render group per tick) but misses the 2 ms publish gate: p99 2.9–3.0 ms. The likely cause is the
virtual sensor's fixed cost per note (~1.04 ms: fence wait, 11 dma-buf reads, `ingest`, a one-camera `process`
launch set, then a `synchronize` in `publish`), which fills half of the 2.08 ms note interval, so publishes that
fall into a busy window slip; that mechanism alone accounts for ~2 ms (DT/2 publish rule + one note), so ~1 ms is
not yet explained (not profiled). The limit moved from the sim (N ≈ 4 shared-phase) to the virtual sensor
(estimated N ≈ 12 staggered, not measured); the N=16 test keeps every delivery gate, dmean, tick match, the
latency budget and `max_fired_per_tick`, and enforces late ticks ≤ 1 % and p99 < 2·DT unless the GPU was contended
(a temporary relaxation, **reverted to the strict
gate below**). GL slot memory at N=16: 704 render slots × 764×574×4 B = 1178 MiB; virtual sensor
790 MiB of its 797 MiB bound.

**Virtual sensor per-note cost (2026-10-04).** Lateness decomposes as `late = hold + overdue`: hold =
`t_due − ready_t` is the publish rule's own 0–DT/2 (p99 0.98 ms), so the loop has ≈ 1 ms of the 2 ms gate. Before the
publish-guard fix, overdue was bimodal — 31–33 % < 0.3 ms (an uncontested publish) and 45–47 % at 0.6–0.9 ms — because the
note lands at tick + ≈ 0.6 ms (after the sim's `mj_step`s and the 0.5 ms group render), the virtual sensor's note
segment (`process_ms`: fence wait 0.27 + reads 0.22 + `ingest` 0.37 + one-camera `process` launch 0.15 ≈ 1.0 ms)
then ran to tick + ≈ 1.7 ms, and half of the due times sit exactly at the tick midpoint (tick + 1.0 ms) inside it:
a phase-locked collision, not random load. The earlier "unexplained ~1 ms" was simply this segment; the publish's
own `synchronize` is ≈ 0.01 ms even without the fix (`publish_sync` in a `--publish-guard-ms 0` run), so the
earlier guess that it queued behind the just-launched `process` was wrong. One change fixed it: a publish due
within `--publish-guard-ms` (1.5 ms ≈ the note segment p99) runs *before* a new note segment starts
(`publish_first`, a pure function; the note is delayed by at most the guard against its 50 ms budget, visible as
`process_ms` p99 2.2 ms since that timer starts at note receipt). A second change — `publish()` waiting only on
its own batch's CUDA event on a dedicated stream — was measured too: same p99 (1.30–1.33 ms) but
+68 MiB of driver memory for the extra stream (872 vs 804 MiB at N=16, bound 813), so it was dropped. Per-note
timers (`note_ms` in the summary, permanent). N=16 staggered N4B8, 5 s, untraced, driver 595.91.07; all runs
uncontended (the three baselines are flagged `contended` in their JSON only by the old rule that counted this
run's own ~50 % end-of-run utilisation, which this change fixes):

| | p99 publish − ready_t | overdue p99 | share 0.6–0.9 ms | `process_ms` mean / p99 | note_ms mean (fence / read / ingest / process / publish_copy / publish_sync) |
|---|---|---|---|---|---|
| before (main 417ff14), 3 runs | 1.93 / 1.98 / 2.07 ms | 1.19–1.83 | 0.45–0.47 | 1.04 / 1.30–1.47 ms (receipt → flush) | 0.27 / 0.24 / 0.37 / 0.16 / 0.04 / 0.01 (from the `--publish-guard-ms 0` run of the new code: p99 1.81 ms) |
| after (final code, publish-first), 7 runs (ranges exclude the noisy run) | **1.34 / 1.35 / 1.39 / 1.40 / 1.40 / 1.43 ms** + one 2.47 ms run in which the sim itself missed 13 ticks (vs 4–6) | 1.25–1.43 | 0.02–0.03 | 1.26 / 2.2 ms (receipt → flush, includes the ≤ guard deferral) | 0.27 / 0.22 / 0.37 / 0.15 / 0.08 / 0.01 |
| after with V2 (measured, dropped), 3 runs | 1.30 / 1.33 / 1.33 ms | 1.28–1.29 | 0.02–0.03 | — | 0.27 / 0.21 / 0.36 / 0.15 / 0.08 / 0.00 |

Tick match 0.997, late sim ticks 4–6/2500, KS pooled ≤ 0.003 (per camera ≤ 0.007) unchanged (the noisy run: tick match 0.996, 13 late ticks, overdue p99 2.12, excluded from the after-row ranges). `ingest` (11 accumulate
launches) is now the largest piece; the "natural mini-batch" never happens at N ≤ 16 (`flush_batch` mean 1.01: one
note per `process`), so V3 (fence fds in `select.poll`) was not needed and was not built. The N=16 test enforces the
original strict gates again (p99 < DT), with the contended-GPU rule kept for the two timing gates only — and that
rule now ignores this run's own processes. The remaining risk is machine noise: when the sim process misses ticks
(the 2.47 ms run), the virtual sensor cannot recover them; the gate is a single run.

### Deterministic stall A and the gen2-GC stall B in the N=16 gate (2026-10-04)

`test_virtual_sensor_window_16` (5 s, N=16 staggered) failed about one run in seven on the p99 publish gate
(< 2 ms, which allows ≈ 22 of 2255 publishes over 2 ms). Decomposing the nine clean runs
(`vs_publish.csv`, late = t_publish − ready_t): **13–17 publishes over 2 ms in every run, all inside one
≈ 22 ms stall at t0 + 1.085–1.118 s** (hold 0, overdue up to 22–24 ms); outside that window, 0 publishes over
2 ms and p99 1.18–1.28 ms. The noisy run added a second burst (3.0–3.1 s, 13 late sim ticks) and failed.

Finding the stall (diagnostic copies of the example, not included):

- Per-phase laps over 2 ms with their time: the stalled note's **`process` lap was 21–23 ms**, and
  `memory_reserved` grew by +40 MiB right after.
- Flush log: until 1.084 s every flush had k = 1 (or the contiguous pair [13, 14], a slice view); at **1.1096 s
  the first non-contiguous batch k = 3 [0, 1, 15]** (camera 15 and camera 0's phases meet for the first time) ran
  the `index_select` gather path. The k = 11 flush right after it was the consequence (backlog), not the cause.
- Memory history (`torch.cuda.memory._record_memory_history`): the allocations at the stall were the gather
  temporary `_wlin.index_select(0, rows)` (k, 3, H, W) f32 (7.0 MiB at k = 2) and `_finish`'s
  `w.permute(0, 2, 3, 1).to(uint8)` temporary (8.8 MiB at k = 10). But in that run the stall came with **no**
  memory growth, and other first-seen temporary sizes (1.76 / 8.79 / 2.64 MiB) cost nothing.
- `CUDA_MODULE_LOADING=EAGER` control: **0 publishes over 2 ms in 1.0–1.2 s, p99 1.30 ms**, while the +20 MiB
  growth still happened. So the cost is the **lazy module load of a kernel the warm-up never launched**
  (the gather), not the allocator. EAGER itself is not a fix: the virtual sensor's memory went 804 → 1498 MiB
  against a bound of 813.
- Ruled out: Python GC (gen2 never runs in the loop; the one 31 ms gen2 seen in the first diagnostic was caused
  by the diagnostic itself calling `torch.cuda.memory_reserved()` per note, which builds the full `memory_stats`
  dict — never poll memory stats in a hot loop), pinned-host growth (9 blocks, 376 B), Axon pool reclaim (4
  buffers per camera wrap at 0.17 s), a per-minute system timer (exits immediately; A/B runs
  overlapping the minute boundary showed nothing).

Fix (F2, `batch.py`, no new buffers because the vsensor memory gates have 4–11 MiB of headroom): `process()`
splits the selected cameras into maximal contiguous runs (`contiguous_runs`) and then chunks, so every row
access is a slice view (no gather kernel, no `to_device(js)` index upload); `_finish` writes results with
`rgb[lo:hi, s].copy_(w.permute(0, 2, 3, 1))` when the run shares a slot, else one `copy_` per camera (the
float → uint8 cast happens inside `copy_`, bit-equal to `.to(uint8)` on already-rounded values; same for depth and
the `compile=True` path). Steady-state `process()` now launches only kernels the first call already ran and
allocates nothing; the DoF path is the exception.

After F2 (same 5 s run, memory-history on): **0 publishes over 2 ms of 2255, p99 1.24 ms**, 0 allocations ≥ 1 MiB
from `batch.py` in the loop, reserved memory unchanged, virtual sensor 804 → 764 MiB. Unit guard:
`tests/test_batch.py::test_batch_steady_state_kernels` profiles a vsensor-style warm-up and then non-contiguous
mixed-slot `process()` calls in both modes: no new kernel name, allocator peak ≤ 1 MiB above the baseline.

Stall B turned out to be mostly **Python's gen2 garbage collection**. In 15 s runs every process collects
gen2 once (30–45 ms, scanning the setup-time heap: torch, Axon, 700+ dma-buf imports), at a time that drifts with
the allocation pattern (2.2 s, 6.0 s, 8.1 s, 9.6 s in different runs) and lands wherever the interpreter is: a
`publish_sync` lap, a `process` lap, or the poll wait. The 5 s runs usually ended before their first gen2 (one
uninstrumented 5 s run had it at 4.78 s), which is why GC had been ruled out. Fix: each process runs `gc.collect(); gc.freeze()` right before the start barrier
(`_gc_arm`), so later collections only scan the per-note churn; GC stays enabled, and a `gc.callbacks` hook
records per-generation counts and maximum durations into the summary (`gc`). After it (15 s, N=16): gen2 0 in all
three processes, the only lap over 2 ms is the first note's dma-buf `read` (7 ms at t0 + 21 ms, harmless), p99
1.19–1.21 ms, and the sim's startup late ticks dropped from 4 to 0–2. The one event not explained by A or GC is
one run's sim late-tick burst (13 ticks at 3.0 s), itself consistent with a sim-side gen2.

Gates (`window_gates`, enforced when the run is `realtime` — always for the N=2/N=8 window tests, only when the
GPU is uncontended for N=16 — and the sim had ≤ 6 late ticks; reported otherwise): reserved memory unchanged
in the loop (colour path only: the depth window test still shows +20 MiB of **unexplained** growth — not
`_finish_depth`, whose noise code never runs with the virtual sensor's config, and the depth-noise kernel guard
shows no growth — so depth is reported until a memory-history run with `--depth` finds it), and no GC collection
≥ 2 ms in any process; slow `process` laps are reported with their time. Still allocating in the steady state, served from the allocator cache after first use:
DoF, AE/AWB uploads, depth noise, `compile=True`'s output tensor and the instant path's `grid_sample` output. The N=16 test runs **15 s** (budget ≈ 68 late publishes) with every threshold unchanged.
Ten consecutive 15 s runs: p99 1.22 / 1.20 / 1.22 / 1.21 / 1.20 / 1.19 / 1.20 / 1.21 / 1.20 / 1.18 ms; 0 publishes over 2 ms in any run (6802 publishes each, max 1.5–1.8 ms, 0 late sim ticks), and 0 publishes over 2 ms in 1.0–1.2 s in any run.

Side effect on the batch bench (`--repeat 3`, same day as Stage 6): nothing slower; n = 1 pipelined −18 % (640×480
colour 0.338 → 0.276 ms, the per-call index uploads and the `index_put_` temporary are gone), n = 16 / 64
`process` −3 to −4 %. The Stage 4 / 6 tables are left as measured then. A scattered mask is not slower either:
n = 64 640×480 colour, `process(mask)` with every other camera (32 runs of k = 1) 2.72 ms vs the first 32 cameras
as two chunks of 16 4.91 ms vs all 64 9.72 ms — the k = 1 runs are faster per camera than the chunked path,
which points at the chunk path's `grid_sample` on an `expand`ed grid (follow-up, not this PR).

The warm-up of the virtual sensor now covers exactly what its loop does — the dma-buf `read`/`read_depth` (their
first launch cost 7 ms at t0 + 21 ms in every run; gone, N=16 now has zero laps over 2 ms), `ingest` and
`process`. The +20 MiB the depth window test still shows (110 → 130 MiB, N=2 depth) is not explained yet: the
virtual sensor's config has no depth-noise fields, so `_finish_depth` is a plain `copy_` there, and the depth-noise
kernel guard shows no growth either. Follow-up: `out/flake/vs_diag5.py --depth --n 2` (memory history with Python
stacks); candidates are the loop's `read_depth`, `ingest` and the depth publish copy.

## Depth edge holes / flying pixels (`depth_edge_hole_px`)

`k > 0` adds, per depth frame, a (2k+1)² dilate + erode, one uniform draw and the band select. Cost at
640×480, k=2, with axial noise on (2026-09-30, quiet GPU): CPU `depth_noise` 2.32 → 3.17 ms (+0.85 ms,
cv2 morphology + numpy masks); GPU `SimCamera.render(depth=True)` 0.87 → 0.99 ms (+0.12 ms,
`F.max_pool2d` ×2 + `uniform_`). The batch path pays the same kernels once per chunk plus one `uniform_`
per camera. Statistics on `two_depth_scene` (0.5 m / 1.0 m planes, band 1920 px): hole ratio 0.508 (CPU) /
0.503 (GPU), flying pixels 944 / 955, all within `[erode, dilate]`; the two paths' band masks are bit-equal
and the batch is bit-equal to the GPU `SimCamera`.

## Stage 6: `torch.compile` on the batch tone chain (`SimCameraBatch(compile=True)`, 2026-10-04)

The eager tail of `_finish` (FPN offset add, shot/read noise, clamp, gamma `pow`, ×255, round, NHWC
permute, uint8 cast) is ~12 elementwise kernels, each one full pass over the `(k, 3, H, W)` float32 chunk.
`tone_fused` (`batch.py`) is the same arithmetic as one pure function; `compile=True` runs it through
`torch.compile(dynamic=True)` (inductor fuses it into a few Triton kernels). Only the plain path is
fused: AE, AWB and Bayer keep the eager chain (they need per-camera means or a conv in the middle). The
vignette/PRNU multiply, the LUT gather and the `grid_sample` remap stay eager in front of it.

Cost: `__init__` compiles both guard variants (`k == 1` and `k > 1`; dynamo specialises size-1 dims
even under `dynamic=True`), measured ≈ 4 s once per process (`parity_check.py`); `tone_fused` is
compiled lazily so importing `batch` never triggers inductor, and no step stalls on a compile. Output differs from eager by at most 1 DN on ≈ 3 × 10⁻⁶ of the
pixels (fused rounding of the `sqrt`/`pow` chain); `tests/test_batch.py::test_batch_compile` gates
≤ 1 DN on ≤ 10⁻⁴ of pixels, and bit-equality on the AE fallback. `test_tone_fused` checks the pure
function against the eager chain on CPU. Default stays `compile=False`, so every existing bit-equality
test is untouched.

`scripts/bench_batch.py --repeat 3` vs `--repeat 3 --compile`, same matrix as Stage 4, ring 4 rows,
idle GPU (`out/compile/{eager,compile}.csv`):

| n | resolution | depth | process ms eager → compile | pipelined ms eager → compile | ms / camera eager → compile | pipelined speed-up |
|---|---|---|---|---|---|---|
| 1 | 640×480 | no | 0.21 → 0.23 | 0.34 → 0.37 | 0.340 → 0.368 | 0.92× |
| 1 | 640×480 | yes | 0.27 → 0.29 | 0.47 → 0.50 | 0.471 → 0.504 | 0.93× |
| 1 | 1280×720 | no | 0.32 → 0.24 | 0.42 → 0.37 | 0.421 → 0.372 | 1.13× |
| 1 | 1280×720 | yes | 0.34 → 0.30 | 0.72 → 0.66 | 0.725 → 0.657 | 1.10× |
| 16 | 640×480 | no | 2.57 → 1.40 | 5.59 → 4.26 | 0.349 → 0.266 | 1.31× |
| 16 | 640×480 | yes | 3.19 → 2.02 | 8.86 → 7.55 | 0.553 → 0.472 | 1.17× |
| 16 | 1280×720 | no | 9.35 → 4.39 | 12.61 → 7.65 | 0.788 → 0.478 | 1.65× |
| 16 | 1280×720 | yes | 11.99 → 7.02 | 18.74 → 13.77 | 1.171 → 0.860 | 1.36× |
| 64 | 640×480 | no | 9.99 → 5.32 | 21.88 → 17.12 | 0.342 → 0.267 | 1.28× |
| 64 | 640×480 | yes | 12.68 → 8.01 | 34.92 → 30.32 | 0.546 → 0.474 | 1.15× |
| 64 | 1280×720 | no | 37.04 → 17.57 | 50.14 → 30.69 | 0.783 → 0.480 | 1.63× |
| 64 | 1280×720 | yes | 47.84 → 28.43 | 74.70 → 55.32 | 1.167 → 0.864 | 1.35× |

Reading: `process()` is 1.6–2.1× faster wherever a chunk is full (n ≥ 16); the pipelined step gains
1.15–1.65× because `render_into` (GL + interop copies) is untouched and now dominates. At 1280×720 colour
the per-camera cost drops from 0.78 to 0.48 ms, close to the 640×480 figure, which says the eager chain
was bandwidth-bound on the work buffer. n = 1 at 640×480 is 8 % slower (≈ 30 µs of guard checks and
launch bookkeeping on a 0.2 ms chunk), so the single-camera `SimCamera` path is left eager. Memory is
unchanged (same work buffers; inductor allocates nothing persistent).

Not done, and why: fusing the LUT gather + remap into the same kernel (inductor does not fuse
`index_select` into `grid_sample`; a hand-written Triton kernel could, but the remaining `process` is
now ≤ 35 % of the step), CUDA graphs (the chunk loop has host-side RNG and index work between launches),
and compiling `GpuPipeline` in `gpu.py` for the single-camera path (no gain at n = 1, see above).


## Observer: cost of the live observer on the real-time path (2026-10-14)

`examples/virtual_sensor.py --scene prism --observer`: a 4th process (nice 10, one OpenCV thread) subscribes to
camera 0, copies frame + trailer device -> host (0.9 MB per frame), and runs `observer_replay.Observer` in a worker
thread. Prism scene, window mode (exposure 8 ms, readout 16 ms), N = 4, 15 s per run, quiet machine (load ~0.9,
GPU idle at start).

| | observer off | observer on |
|---|---|---|
| p99(publish − ready_t), 8 runs | 1.181 1.260 1.306 1.454 1.267 1.274 1.303 1.184 ms | 1.440 1.291 1.526 1.575 1.320 1.281 1.328 1.503 ms |
| median | 1.271 ms | 1.384 ms |
| sim late ticks / 7500 | 8 16 9 12 8 8 8 8 | 19 9 14 20 8 9 8 14 |
| policy poll p99 | 0.061-0.072 ms | 0.066-0.073 ms |

- Median difference **+0.113 ms** (criterion ≤ 0.15 ms: pass); mean difference +0.129 ± 0.051 ms (se). The observer
  costs about 0.1 ms of p99, 75 % (median) to 86 % (mean) of the budget; the mean's upper 95 % bound (~0.23 ms) is above it, so this is not
  shown to be under 0.15 ms with confidence. The first 8-vs-8 set (before this one) gave +0.104 ms (median),
  +0.044 ± 0.056 (mean).
- Run-to-run scatter with the observer off is 0.085 ms (sd, this set) to 0.135 ms (the first 8-run set): 3 runs per
  arm (the plan's first design) measured +0.21 ms on the same code. The A/B is therefore 8 runs per arm in ABBA blocks and opt-in
  (`REALSENSOR_AB=1 pytest tests/test_virtual_sensor.py -k observer_ab`, ~7 min); the pre-push gate runs the
  deterministic part (`test_virtual_sensor_observer`: every window gate with the observer up, miss rate ≤ 1 %).
  Re-run the A/B whenever the observer process, its receive loop, the handshake or the prism step changes.
- Observer throughput: 427 / 427 frames per 15 s (0 % miss), worker 15.7 ms mean / 21-23 ms p99 per frame, one
  estimate per second (calibrateCamera + latency bootstrap) 140-420 ms, absorbed by the 32-frame queue.
- What made it fit: `PrismTarget.detect` ran ArUco marker detection once per face (six passes, 88 ms per frame
  single-threaded, 56 % of frames dropped); one pass shared by all faces is 15 ms with identical corners. The sim's
  prism step uses `mj_kinematics` + `mj_camlight` instead of `mj_forward` (same camera / light / geom poses).
- Diagnostic (2 runs each, noisy): observer up without copying (1.34 / 1.30 ms), copying without processing
  (1.31 / 1.45 ms), full (1.34 / 1.45 ms) vs off (1.30 / 1.17 ms): no single component stands out at this resolution.

## Pre-push gate wall time (2026-10-14)

`make lint && make cov` (the pre-push hook) on a quiet machine. Everything in this section ran with the
single-thread pins (OMP / OpenBLAS / MKL / OpenCV = 1) that were removed later; the first multi-threaded gates took
6:34 and 6:37 (250 tests).

| when | suite | wall time |
|---|---|---|
| 2026-10-12, three back-to-back gates | 202-207 tests | 30:32, 30:34, 30:41 |
| 2026-10-14, before the detector change | 214-237 tests | 31:02-31:05 |
| 2026-10-14, after it (one-pass `PrismTarget.detect`) | 250 tests | 9:50, 10:31, 10:30, 10:27, 10:40 |

The suite went from ~30 to ~10 min with the one-pass detector, not with a quieter machine: every observer test that renders
frames detects the prism on each one (`GeomObserver.push`), and the detector ran ArUco marker detection once per
face (six passes, 88 ms per 640x480 frame single-threaded) where one shared pass (15 ms) gives the same corners.
The last three runs (10:27-10:40, 250 tests, coverage 96.69 %) meet the 12 min target with every process
single-threaded and no tolerance changed. The margin is ~11 %: one more 20-seed statistical test would use it up.
Running a gate next to a simulation or another gate still slows both (they compete for cores and the GPU tests).
