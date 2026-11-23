# GPU-RL topology: leaving the GPU to the policy

Design note (2026-09-27, no implementation yet). Goal: the policy process should keep as much
GPU memory and time as possible; the camera sim should cost little and never stall the GPU.

## What runs where today

| stage | API | hardware |
|---|---|---|
| render (colour, depth) | OpenGL (EGL), MuJoCo's GL shaders | SM CUDA cores + raster/ROP |
| interop copy | CUDA driver memcpy (`cuMemcpy2DAsync`, `cuMemcpyDtoDAsync`) | copy engine (DMA), no compute |
| post-processing (LUT, remap, exposure, noise, Bayer, depth noise) | torch CUDA kernels | SM CUDA cores, elementwise |
| DoF conv | cuDNN, `allow_tf32=True` by default | may use Tensor Cores (TF32) |

GL shaders and CUDA kernels share the SMs, so moving post-processing into GLSL saves no GPU
time. What a GL-only sim process saves is the CUDA context and torch's footprint.

## Measured (RTX 5080, driver 580, `measure_gpu_rl_budget.py`)

| item | MiB |
|---|---|
| EGL display init, no context (policy side, to import dma-buf images) | 0 |
| GL-only sim: EGL context + MuJoCo renderer (offscreen 900×700) | 209 |
| CUDA context (torch init), per process | 270 |
| first cuda camera (buffers + cuDNN/cuBLAS handles + allocator) | 203 |
| each extra cuda camera, 640×480, own Renderer/EGL FBO | 137.5 (torch 43.8) |
| each extra cuda camera, 128×128, own Renderer | 91.8 |

Resolution sweep (idle GPU, `measure_resolution_sweep.py`; pipeline = distortion + vignetting + noise):

| resolution | plain `Renderer.render` | cuda+interop → tensor | + depth | MiB per camera (torch) |
|---|---|---|---|---|
| 224×224 | 0.120 ms | 0.239 | 0.430 | 90 (7) |
| 256×256 | 0.155 | 0.242 | 0.436 | 84 (2) |
| 640×480 | 0.279 | 0.295 | 0.523 | 136 (44) |
| 1280×720 | 0.489 | 0.439 | 0.785 | 256 (137) |

Budget the **native camera resolution** (640×480 / 1280×720), not the model's input size: VLA
inputs (224², 256²) are produced by the policy's own resize, and the sensor effects must be
applied at sensor resolution first, exactly as on the robot. At native resolution the
post-processing (~0.10 ms/camera) is mostly memory-bound passes, only ~0.06 ms is launch
overhead, so batching buys little time; the per-camera `mujoco.Renderer` (~90–120 MiB,
resolution-independent) and the torch work buffers (pixel-proportional) are what matter.
With one `SimCamera` per env, N=64 at 640×480 ≈ 8.9 GB and ≈19 ms/step.

## Option A — single process, vectorised (recommended first)

Policy and N envs in one process; one CUDA context (the policy's). Share one `mujoco.Renderer`
across envs of the same model (`update_scene(data_i)` per env), stack renders into
`(N,3,H,W)` via interop, run the torch pipeline once per step (grid_sample, index_select,
randn, conv, Bayer are all batch-native; per-camera state such as AE ev / AWB gains / randomised
parameters become length-N tensors; `CameraStream` holds slot indices instead of frame arrays).
Process envs in chunks (e.g. 16) so f32 work buffers are reused and only the uint8 observation (0.9 MiB/env at 640×480) persists per env. Projected N=64 at 640×480: ≈1.0 GB (chunk 16), ≈15 ms/step (render-bound). Carries over unchanged to a batched
renderer (Madrona-style), whose output is already a CUDA tensor.

## Option B — GL-only sim process + dma-buf (for an Axon-style split)

The sim process creates no CUDA context. `offColor` → `eglCreateImageKHR(GL_RENDERBUFFER)`
→ `eglExportDMABUFImageMESA` → fd sent once (SCM_RIGHTS) → policy:
`eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, fd, fourcc, modifier)` → `cuGraphicsEGLRegisterImage`
→ `cuGraphicsResourceGetMappedEglFrame` → `cuMemcpy2DAsync` into a tensor.

Spike (`scripts/spike_dmabuf_export.py`, single process, same calls): fourcc `AB24`, modifier
`0x300000000e08014` (NVIDIA block-linear; the modifier attributes are required on import),
`CU_EGL_FRAME_TYPE_ARRAY`, pixels **bit-exact** with `Renderer.render()` over two frames.

Open items for B: cross-process sync (`glFinish` or EGL fence fd before signalling seqno),
double-buffered FBOs (blit `offColor` into two of our own), depth (export is colour formats
only; blit depth into an R32F colour buffer or skip), and the reverse direction — importing an
Axon `udmabuf` buffer as the GL render target (`EGL_EXT_image_dma_buf_import` +
`glEGLImageTargetRenderbufferStorageOES`) so Axon's seqno/staleness channel is reused. Axon
accelerator pools export CUDA VMM fds, which GL cannot import. GL contexts are not MPS clients,
so keep one sim process.

## Decision rule

If policy and envs can live in one process, A is sufficient. If the policy must be a separate
process, build A's batched pipeline first and add B on top.

## Status (2026-09-28): topology B implemented (R0–R5)

| task | result |
|---|---|
| R0 shared Renderer | bit-equal across cpu / cuda / cuda+interop; per-camera GPU increment measured by `tests/test_batch.py --mem` (own renderer → shared): cpu 116.9 → 0.0 MiB, cuda 151.2 → 40.5, cuda+interop 136.8 → 38.0 |
| R1 `SimCameraBatch` | chunked post-processing over a shared renderer, output ring `(i, r_i mod K)`, bit-equal to `SimCamera(seed+i)`; `GlInterop` public copy API |
| R2 `CameraStream(render=)` + `CameraStreamBatch` | per-camera exact timing, `ring_depth(cfg)`, `(ready, capture_t, seq, handles, new, fired)` |
| R3 bench | batching buys **memory** (137.5 → 7.6 MiB marginal, ≈16 MiB amortised per camera at n=64), not time: pipelined-vs-pipelined the batch is 1.1–1.9× slower per camera; the `process()` host sync (pageable H2D copy of slot indices) was removed later with pinned `non_blocking` uploads (0 syncs, host time 10.1 → 1.1 ms at n=64; n=1 now within 1.16× of a single camera), n ≥ 16 still alternates GL and CUDA via the shared renderbuffer's unmap (`perf.md` Stage 4); depth n=64 exceeds the colour-only bound by 31–33 % (1280×720 / 640×480; ring + raw depth) and fits the depth-extended bound |
| R4 dma-buf split | GL-only sim → fd → CUDA receiver, colour + depth bit-exact, fence sync, sim 193 MiB with no CUDA context |
| R5 virtual sensor | 3 processes (sim GL-only / virtual sensor CUDA effects / policy Axon-only): KS 0.0000 per camera, tick match 1.0000, p99(publish − ready_t) 1.10 ms, VS 506 ≤ 515 MiB, stall-safe (overrun drops); publish rule `t_pub = max(ready_t, T − DT/2)`; the reverse direction (render into an Axon udmabuf buffer) is impossible on this driver: LINEAR system-memory images can only be sampled, proven with a block-linear positive control |
| Phase 5 window mode | motion blur + rolling shutter in `SimCameraBatch` (per-camera `StateRing`, sub-frames accumulated into `_wlin`, bit-equal to `SimCamera` for blur/RS/N4B8, maps/no maps, DoF, AE) and in the R5 virtual sensor (groups of 11 colour sub-frames + 1 depth per capture over dma-buf, fds sent in ≤ 250-fd messages); N4B8 costs 2.58 ms/camera/frame in the batch = the single camera (render bound), 9.45 MiB/camera marginal; R5 real-time up to ~4 cameras firing on one tick (`perf.md` Stage 5) |

Follow-ups: ~~stagger `phase_s` across cameras so the R5 window groups (0.53 ms each) spread over the frame~~ (done 2026-09-30: `stagger_phases` + `CameraStreamBatch(phase_s=)`, `--stagger`; N=8 N4B8 passes every strict gate, N=16 is limited by the virtual sensor's ~1 ms per-note cost, `perf.md` Stage 5); ~~mini-batch notes in the virtual sensor for N > ~12~~ (done differently 2026-10-04: the lateness was a phase-locked collision between the note segment and the tick-midpoint due times; a publish-first lookahead brings N=16 p99 to 1.34–1.43 ms (an event-scoped publish stream was measured and dropped), `perf.md` Stage 5); ~~device-side slot indices in `batch.py`~~ (done: pinned `non_blocking` uploads); ~~interop copies on a side stream so `render_into` of step k+1 overlaps `process(k)`~~ (tried and reverted 2026-10-02: GL/CUDA context time-slicing makes it 3–8× slower, `perf.md` Stage 4 side-stream A/B); expose Axon `TensorDescriptor.capture_ts_ns` in the Python binding so the policy needs no device read for the trailer; ~~fuse the batch tone chain~~ (done 2026-10-04: `SimCameraBatch(compile=True)`, `process()` 1.6–2.1× at n ≥ 16, `perf.md` Stage 6); ~~subprocess coverage for `dmabuf.py`~~ (done: coverage `patch = ["subprocess"]`, `dmabuf.py` 0 → 90 %, total 81 → 95 %); check that no other process holds the GPU (`nvidia-smi`) before GPU tests.
