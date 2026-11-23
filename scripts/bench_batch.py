#!/usr/bin/env python3
"""R3: SimCameraBatch bench -> ``out/batch_bench.csv`` + a markdown table on stdout.

For n in {1, 16, 64} x {640x480, 1280x720} x {colour, depth} x ring in {ring_depth(cfg), 1}, ms/step
two ways: serialised (``render_into`` x n, sync, ``process``, sync; the two halves are reported and
their sum is ``serial_ms``) and pipelined (``steps`` x (``render_into`` x n; ``process``), one sync at
the end, so the CPU GL work of step k+1 overlaps the GPU post-processing of step k: ``pipelined_ms``,
``ms_per_camera = pipelined_ms / n``). Memory: ``torch.cuda.memory_allocated`` /
``max_memory_allocated`` after one step, relative to before the batch was built.
Config: board scene, Phase 1 effects + shot/read noise, 30 fps, 50 ms latency (ring_depth = 4),
chunk 16, interop. Reference rows (``ring = 0``): one ``SimCamera(device="cuda", interop=True)``
rendering to a tensor, timed the same two ways (sync every frame / one sync at the end), so the
table shows the batching ratio (batch ms/camera / single ms/frame) for both methods. A case that
runs out of GPU memory is skipped with a note. The CSV header comment records ``nvidia-smi`` and
the other GPU processes before and after the run: stop them (e.g. ``ollama stop <model>``) right before this script.

Window mode (``--window``): uses ``WIN`` config (N4B8: 11 grid-snapped renders/frame). Each camera
advances the sim and pushes to its ``StateRing`` before ``render_into`` / ``render``. The window
matrix runs 640x480, depth=False, single + batch n in {1, 16, 64} with ring=K only.

Usage: MUJOCO_GL=egl .venv/bin/python scripts/bench_batch.py [--out out/batch_bench.csv] [--steps 50]
       MUJOCO_GL=egl .venv/bin/python scripts/bench_batch.py --window [--out out/batch_bench_window.csv]
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COLS = ("n", "width", "height", "depth", "ring", "render_into_ms", "process_ms", "serial_ms", "pipelined_ms",
        "ms_per_camera", "alloc_mib", "peak_mib", "window")
TIMING = ("render_into_ms", "process_ms", "serial_ms", "pipelined_ms", "ms_per_camera")
WARMUP = 3
NS = (1, 16, 64)
RESOLUTIONS = ((640, 480), (1280, 720))
CHUNK = 16
# Phase 1 effects + noise, as tests/test_batch.py P1 + NOISE; 30 fps / 50 ms latency -> ring_depth 4
CFG_KW = dict(dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=-0.3, sigma_prnu=0.02,
              sigma_dsnu=0.005, black_level=0.02, gamma=2.2, sigma_shot=0.02, sigma_read=0.005,
              fps=30.0, latency_s=0.05)
MIB = 2**20
# Window-mode camera config: N4B8 = 11 grid-snapped renders per frame
WIN = dict(exposure_s=0.008, readout_s=0.016, blur_samples=4, rs_bands=8)


def r1_mem_bound(chunk_bytes, n, h_r, w_r, h, w, ring, depth=False):
    """R1 Accept 3 memory bound (bytes) for ``SimCameraBatch``, optionally extended for depth.

    Colour: ``1.1 * (chunk + n * (7 B/px at render size + ring * 3 B/px + 4 B/px at output size))``
    (raw RGBA slot + output ring + FPN, with the render-size slack covering the FPN's other 4 B/px),
    as in ``tests/test_batch.py::test_batch_mem``. ``depth=True`` adds the per-camera depth terms the
    R1 bound leaves out: the raw Z slot (4 B/px at render size) and the depth ring (``ring * 4`` B/px).

    Args:
        chunk_bytes: Chunk work buffers + maps + interop buffers (``chunk_bytes(batch)``).
        n: Number of cameras.
        h_r: Render (margin) height, px.
        w_r: Render (margin) width, px.
        h: Output height, px.
        w: Output width, px.
        ring: Output slots per camera.
        depth: Include the per-camera depth slot and ring.

    Returns:
        The bound in bytes (float).
    """
    per = h_r * w_r * 3 + h_r * w_r * 4 + ring * h * w * 3 + h * w * 4
    if depth:
        per += h_r * w_r * 4 + ring * h * w * 4
    return 1.1 * (chunk_bytes + n * per)


def csv_row(r):
    """One CSV line (no newline) for a result dict with ``COLS`` keys; None -> empty field.

    Args:
        r: Result dict (ints for n/width/height/depth/ring, floats for ms and MiB).

    Returns:
        Comma-separated fields in ``COLS`` order, floats with 3 decimals (ms) or 1 (MiB).
    """
    out = []
    for c in COLS:
        v = r.get(c)
        if v is None:
            out.append("")
        elif isinstance(v, float):
            out.append(f"{v:.1f}" if c.endswith("_mib") else f"{v:.3f}")
        else:
            out.append(str(int(v)))
    return ",".join(out)


def markdown_table(rows):
    """Markdown table of the bench rows; ``ring == 0`` rows are the single-SimCamera reference.

    Args:
        rows: Result dicts (``COLS`` keys). Batch rows get two ``x single`` columns: their pipelined and
            serialised ms/camera over the reference's pipelined and serialised ms/frame at the same
            resolution and depth flag (blank without a reference).

    Returns:
        The table as a string (header, separator, one line per row, in input order).
    """
    ref = {(r["width"], r["height"], r["depth"], r.get("window", 0)): r for r in rows if r["ring"] == 0}
    lines = ["| n | resolution | depth | window | ring | render_into ms | process ms | serial ms | pipelined ms "
             "| pipelined ms/cam | × single pipelined | × single serial | alloc MiB | peak MiB |", "|" + "---|" * 14]
    f = lambda v, p=2: "–" if v is None else f"{v:.{p}f}"  # noqa: E731
    wv = lambda v: "N4B8" if v else "–"  # noqa: E731
    for r in rows:
        single = r["ring"] == 0
        base = ref.get((r["width"], r["height"], r["depth"], r.get("window", 0)))
        rp = rs = "–"
        if base is not None:
            rp = f"{r['pipelined_ms'] / r['n'] / base['pipelined_ms']:.2f}"
            rs = f"{r['serial_ms'] / r['n'] / base['serial_ms']:.2f}"
        lines.append(f"| {'1 (SimCamera)' if single else r['n']} | {r['width']}×{r['height']} "
                     f"| {'yes' if r['depth'] else 'no'} | {wv(r.get('window', 0))} | "
                     f"{'–' if single else r['ring']} "
                     f"| {f(r['render_into_ms'])} | {f(r['process_ms'])} | {f(r['serial_ms'])} "
                     f"| {f(r['pipelined_ms'])} | {f(r['ms_per_camera'], 3)} | {rp} | {rs} "
                     f"| {f(r['alloc_mib'], 1)} | {f(r['peak_mib'], 1)} |")
    return "\n".join(lines)


def gpu_condition():
    """Snapshot of the GPU's load for the CSV header.

    Returns:
        One line: ``nvidia-smi`` name, memory used/total and utilisation, and the ``ollama ps`` model names
        (``none`` if no model is loaded or ollama is absent).
    """
    q = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]
    try:
        smi = subprocess.check_output(q, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as e:
        smi = f"nvidia-smi unavailable ({e})"
    try:
        ps = subprocess.check_output(["ollama", "ps"], text=True, timeout=10).strip().splitlines()[1:]
    except (OSError, subprocess.SubprocessError):
        ps = []
    return f"nvidia-smi before run: {smi}; ollama ps: {'; '.join(' '.join(p.split()[:1]) for p in ps) or 'none'}"


def chunk_bytes(b):
    """Bytes of a batch's shared (per-chunk, not per-camera) buffers: work, maps, LUT, depth, interop.

    Window mode adds its shared sub-render and band buffers (``_wlin`` is per camera, not counted here);
    buffers a mode does not allocate are None and skipped.
    """
    ts = [b._idx, b._lin, b._nz, b._s, b._grid, b._didx, b._lut]
    ts += [] if b._work is b._lin else [b._work]
    ts += list(b._dbuf or ()) + ([b.interop._buf, b.interop._dbuf] if b.interop is not None else [])
    # window-mode buffers
    ts += [b._u8w, b._lin1, b._idx1] + list(b._bacc or ()) + list(b._bgrid or ())
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def median_row(rows):
    """One result dict from repeated runs of a case: median of the timing columns, last run's memory.

    Medians are taken per column, so ``serial_ms`` need not equal ``render_into_ms + process_ms`` exactly.

    Args:
        rows: Result dicts of the same case (``COLS`` keys), at least one.

    Returns:
        Copy of the last row with ``TIMING`` columns replaced by their median over the rows (None stays None).
    """
    out = dict(rows[-1])
    for k in TIMING:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = statistics.median(vals) if vals else None
    return out


def _reset():
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def bench_batch(n, w, h, depth, ring, steps=50, window=False, compile=False):
    """Time and measure one ``SimCameraBatch`` case on a fresh board scene (serialised and pipelined).

    Args:
        n: Number of cameras.
        w: Output width, px.
        h: Output height, px.
        depth: Render and process depth too.
        ring: Output slots per camera.
        steps: Timed steps per method (after ``WARMUP`` steps).
        window: Use window mode (sim step + ring push every frame).
        compile: ``SimCameraBatch(compile=True)``: fused torch.compile tone chain.

    Returns:
        ``(row, bound_info)``: the result dict and ``(chunk_bytes, H_r, W_r)`` for ``r1_mem_bound``.
    """
    import mujoco
    import torch
    from test_camera import board_scene

    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.batch import SimCameraBatch
    model, data = board_scene()  # SimCamera raises offwidth/offheight before the shared renderer exists
    a0 = _reset()
    cfg_kw = CFG_KW if not window else {**CFG_KW, **WIN}
    b = SimCameraBatch(model, "cam", CameraConfig(w, h, **cfg_kw), n, interop=True, chunk=CHUNK, ring=ring,
                       compile=compile)

    def advance():  # window: the sim moves on and every camera's ring sees the new state
        if window:
            mujoco.mj_step(model, data)
            for i in range(n):
                b.rings[i].push(data)

    def step():
        advance()
        for i in range(n):
            b.render_into(i, data, depth=depth)
        b.process(depth=depth)
    try:
        assert b.interop is not None, "CUDA-GL interop unavailable: the bench would time readPixels"
        step()
        torch.cuda.synchronize()
        alloc, peak = torch.cuda.memory_allocated() - a0, torch.cuda.max_memory_allocated() - a0
        info = (chunk_bytes(b), b.H_r, b.W_r)
        for _ in range(WARMUP):  # cuDNN, allocator
            step()
        tr = tp = 0.0
        for _ in range(steps):  # serialised: sync after each half
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            advance()
            for i in range(n):
                b.render_into(i, data, depth=depth)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            b.process(depth=depth)
            torch.cuda.synchronize()
            tr, tp = tr + t1 - t0, tp + time.perf_counter() - t1
        t0 = time.perf_counter()
        for _ in range(steps):  # pipelined: one sync at the end
            step()
        torch.cuda.synchronize()
        pipe = (time.perf_counter() - t0) / steps * 1e3
    finally:
        b.close()
    tr, tp = tr / steps * 1e3, tp / steps * 1e3
    row = dict(n=n, width=w, height=h, depth=int(depth), ring=b.ring, render_into_ms=tr, process_ms=tp,
               serial_ms=tr + tp, pipelined_ms=pipe, ms_per_camera=pipe / n, alloc_mib=alloc / MIB,
               peak_mib=peak / MIB, window=int(window))
    return row, info


def bench_single(w, h, depth, steps=50, window=False):
    """Reference: one ``SimCamera(device="cuda", interop=True)`` rendering to a tensor (``ring = 0`` row).

    Args:
        w: Output width, px.
        h: Output height, px.
        depth: Also render depth.
        steps: Timed frames per method (after ``WARMUP`` frames).
        window: Use window mode (sim step + push every frame).

    Returns:
        Result dict: ``serial_ms`` with a sync after every frame, ``pipelined_ms`` (= ``ms_per_camera``) with
        one sync at the end, and memory after one frame relative to before construction.
    """
    import mujoco
    import torch
    from test_camera import board_scene

    from physicalai_sim2real_camera import CameraConfig, SimCamera
    model, data = board_scene()
    a0 = _reset()
    cfg_kw = CFG_KW if not window else {**CFG_KW, **WIN}
    cam = SimCamera(model, "cam", CameraConfig(w, h, **cfg_kw), device="cuda", interop=True)

    def frame():  # window: the sim moves on and the ring sees the new state (as the batch's advance())
        if window:
            mujoco.mj_step(model, data)
            cam.push(data)
        cam.render(data, depth=depth, as_tensor=True)
    try:
        assert cam._gpu.interop is not None, "CUDA-GL interop unavailable: the bench would time readPixels"
        frame()
        torch.cuda.synchronize()
        alloc, peak = torch.cuda.memory_allocated() - a0, torch.cuda.max_memory_allocated() - a0
        for _ in range(WARMUP):
            frame()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            frame()
            torch.cuda.synchronize()
        ser = (time.perf_counter() - t0) / steps * 1e3
        t0 = time.perf_counter()
        for _ in range(steps):
            frame()
        torch.cuda.synchronize()
        pipe = (time.perf_counter() - t0) / steps * 1e3
    finally:
        cam.close()
    return dict(n=1, width=w, height=h, depth=int(depth), ring=0, render_into_ms=None, process_ms=None,
                serial_ms=ser, pipelined_ms=pipe, ms_per_camera=pipe, alloc_mib=alloc / MIB, peak_mib=peak / MIB,
                window=int(window))


def main(argv=None):
    """Run the matrix, write the CSV (with GPU-condition and memory-bound comments), print the table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="CSV path (default out/batch_bench.csv, or "
                    "out/batch_bench_window.csv with --window)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--window", action="store_true", help="Run window-mode matrix only (640x480, depth=False)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each case and keep the median of the timing columns")
    ap.add_argument("--compile", action="store_true",
                    help="fused torch.compile tone chain (SimCameraBatch(compile=True))")
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import torch

    from physicalai_sim2real_camera import CameraConfig, ring_depth
    cond = gpu_condition()
    print(cond)
    K = ring_depth(CameraConfig(640, 480, **CFG_KW))
    torch.zeros(1, device="cuda")  # CUDA context before any baseline

    out = args.out or os.path.join("out", "batch_bench_window.csv" if args.window else "batch_bench.csv")
    groups, notes = [], []
    # window: 640x480 colour only, single + batch at ring K; otherwise the full matrix. Largest cases first:
    # another process reloading a model mid-run can take ~13 GB of the GPU back
    matrix = ([(640, 480, False)] if args.window else
              [(w, h, depth) for w, h in reversed(RESOLUTIONS) for depth in (True, False)])
    for w, h, depth in matrix:
        groups.append(rows := [])
        rings = (K,) if args.window else (K, 1)
        cases = [("single", None, None)] + [("batch", n, r) for n in NS for r in rings]
        for kind, n, ring in cases:
            name = (f"{w}x{h} depth={depth}{' window' * args.window} "
                    + ("SimCamera" if kind == "single" else f"n={n} ring={ring}"))
            try:
                if kind == "single":
                    rows.append(median_row([bench_single(w, h, depth, args.steps, window=args.window)
                                            for _ in range(args.repeat)]))
                else:
                    reps = [bench_batch(n, w, h, depth, ring, args.steps, window=args.window, compile=args.compile)
                            for _ in range(args.repeat)]
                    row, (cb, h_r, w_r) = median_row([r for r, _ in reps]), reps[-1][1]
                    rows.append(row)
                    if n == 64 and ring == K and not args.window:
                        b1 = r1_mem_bound(cb, n, h_r, w_r, h, w, ring) / MIB
                        bd = r1_mem_bound(cb, n, h_r, w_r, h, w, ring, depth=True) / MIB
                        verdict = "exceeds" if row["alloc_mib"] > b1 else "within"
                        notes.append(f"R1 bound {name}: allocated {row['alloc_mib']:.1f} MiB {verdict} R1 "
                                     f"colour bound {b1:.1f} MiB; depth-extended bound {bd:.1f} MiB "
                                     f"(chunk buffers {cb / MIB:.1f} MiB, render {w_r}x{h_r})")
                print(csv_row(rows[-1]), flush=True)
            except torch.cuda.OutOfMemoryError:
                notes.append(f"skipped {name}: CUDA out of memory")
                print(notes[-1], flush=True)
    rows = [r for g in reversed(groups) for r in g]  # table order: 640x480 first, colour before depth
    after = gpu_condition().replace("before run", "after run (bench process still alive)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# {cond}\n# {after}\n")
        if args.window:
            f.write(f"# window mode: {WIN} (sim step + ring push timed on both sides)\n")
        f.write(f"# chunk={CHUNK}, ring_depth={K} (30 fps, 50 ms latency), interop, compile={int(args.compile)}, "
                f"steps={args.steps}, warm-up={WARMUP}, repeat={args.repeat}; "
                f"ring=0 rows: single SimCamera(cuda, interop) reference; "
                "MiB relative to before construction\n")
        f.write(",".join(COLS) + "\n")
        f.writelines(csv_row(r) + "\n" for r in rows)
        f.writelines(f"# {s}\n" for s in notes)
    print(markdown_table(rows))
    for s in notes + [after]:
        print(s)
    print(f"wrote {out}")
    return rows, notes


if __name__ == "__main__":
    main()
