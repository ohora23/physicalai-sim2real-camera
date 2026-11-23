"""Axon bridge: CameraStream frames -> Axon zero-copy topic -> subscriber, with age logging.

Two spawned processes share a CLOCK_MONOTONIC epoch ``t0``:
- publisher: board scene, SimCamera + CameraStream, mj_step at 500 Hz wall-clock paced
  (sim time == wall time since t0); each NEW frame is published once as
  ``frame bytes | capture_t float64 | n int64`` (16-byte trailer, Axon has no metadata field).
  Logs CameraStream's own age per control step -> out/axon/pub_age.csv (the truth).
- subscriber: polls ``latest_view()`` at 500 Hz wall-clock, decodes the trailer,
  age_sub = (now - t0) - capture_t -> out/axon/sub_age.csv.

Run: MUJOCO_GL=egl PYTHONPATH=third_party/axon/build/python .venv/bin/python examples/axon_bridge.py --seconds 5
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco`

import argparse
import csv
import multiprocessing as mp
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.append(os.path.join(ROOT, "third_party", "axon", "build", "python"))  # fallback if PYTHONPATH unset

TOPIC = "realsensor/head_cam"
W, H = 640, 480
FRAME = H * W * 3
TRAILER = 16  # capture_t float64 + n int64, little-endian
DT = 0.002    # 500 Hz control / poll rate
POLL_LAG = 0.0005  # subscriber poll phase after the publisher's tick (> mj_step + publish time)
OUT = os.path.join(ROOT, "out", "axon")


def _wait_until(target):
    d = target - time.perf_counter()
    if d > 0:
        time.sleep(d)


def _clock_offset():
    """perf_counter - CLOCK_MONOTONIC in this process (0 when perf_counter is CLOCK_MONOTONIC)."""
    return time.perf_counter() - time.clock_gettime(time.CLOCK_MONOTONIC)


def publisher(backend, n_steps, ready, barrier, t0_val, clk):
    import axon
    import mujoco
    from test_camera import board_scene

    from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera

    cfg = CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0,
                       sigma_shot=0.02, sigma_read=0.005, black_level=0.02,
                       fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.05)
    model, data = board_scene()
    model.opt.timestep = DT
    cam = SimCamera(model, "cam", cfg)
    stream = CameraStream(cam, cfg)
    cam.render(data)  # warm-up (GL context, LUTs) outside the paced loop

    pool = axon.TensorPool.create(n_buffers=8, buffer_size=FRAME + TRAILER, backend=backend)
    pub = axon.TensorPublisher.create(TOPIC, pool)
    ready.set()
    pub.handshake_pool()
    buf = np.empty(FRAME + TRAILER, np.uint8)  # the one preallocated payload
    meta_t = buf[FRAME:FRAME + 8].view("<f8")
    meta_n = buf[FRAME + 8:].view("<i8")

    barrier.wait()
    t0_val.value = time.perf_counter() + 0.2
    clk[0] = _clock_offset()
    barrier.wait()
    t0 = t0_val.value
    rows, prev = [], None
    for k in range(1, n_steps + 1):
        _wait_until(t0 + k * DT)
        mujoco.mj_step(model, data)  # data.time == k*DT == wall time since t0
        frame, cap_t, n = stream.step(data)
        if cap_t is None:
            continue
        if cap_t != prev:  # new frame ready -> publish now; dropped frames never get here
            prev = cap_t
            buf[:FRAME] = frame.reshape(-1)
            meta_t[0], meta_n[0] = cap_t, n
            pub.publish(buf, axon.DType.U8)
        rows.append((data.time, n, cap_t, data.time - cap_t))
    cam.close()
    _write(os.path.join(OUT, "pub_age.csv"), "t,n,capture_t,age", rows)


def subscriber(n_steps, ready, barrier, t0_val, clk):
    import axon
    ready.wait()
    sub = axon.TensorSubscriber.create(TOPIC)
    if sub.wait_handshake(10000) != 0:
        raise RuntimeError("subscriber: handshake failed")
    sub.set_fallback_policy(axon.FallbackPolicy.LastKnownGood)

    barrier.wait()
    clk[1] = _clock_offset()
    barrier.wait()
    t0 = t0_val.value
    rows = []
    for k in range(1, n_steps + 1):
        # Poll POLL_LAG after the publisher's tick: exactly in phase it always misses that tick's
        # publish (+1 tick = +2 ms bias), and a half-tick lag shifts every sample by 1 ms against the
        # 2 ms-quantized pub ages (KS ~0.06). Expected age_sub - age = POLL_LAG + transport.
        _wait_until(t0 + k * DT + POLL_LAG)
        v = sub.latest_view()
        now = time.perf_counter() - t0
        if v is None:
            continue
        tr = v.data.reshape(-1)[FRAME:]
        cap_t = float(tr[:8].view("<f8")[0])
        n = int(tr[8:].view("<i8")[0])
        # age_tick: same read, timed by the loop's logical tick k*DT (on the pub's 2 ms age grid)
        rows.append((now, n, cap_t, now - cap_t, k * DT - cap_t, v.staleness_ns, v.seqno))
    _write(os.path.join(OUT, "sub_age.csv"), "t,n,capture_t,age_sub,age_tick,staleness_ns,seqno", rows)


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        csv.writer(f).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--backend", choices=["custom", "udmabuf"], default="custom")
    a = ap.parse_args()
    import axon
    backend = {"custom": axon.PoolBackend.Custom, "udmabuf": axon.PoolBackend.UDMABUF}[a.backend]
    os.makedirs(OUT, exist_ok=True)

    ctx = mp.get_context("spawn")
    n_steps = int(round(a.seconds / DT))
    ready, barrier = ctx.Event(), ctx.Barrier(2)
    t0_val, clk = ctx.Value("d", 0.0), ctx.Array("d", 2)
    procs = [ctx.Process(target=publisher, args=(backend, n_steps, ready, barrier, t0_val, clk)),
             ctx.Process(target=subscriber, args=(n_steps, ready, barrier, t0_val, clk))]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=a.seconds + 120)
    if any(p.exitcode != 0 for p in procs):
        for p in procs:
            p.kill()
        sys.exit(f"bridge failed: exit codes {[p.exitcode for p in procs]}")

    pa = np.genfromtxt(os.path.join(OUT, "pub_age.csv"), delimiter=",", names=True)
    sa = np.genfromtxt(os.path.join(OUT, "sub_age.csv"), delimiter=",", names=True)
    ms = lambda x: f"mean {1e3 * np.mean(x):.2f} ms, p95 {1e3 * np.percentile(x, 95):.2f} ms"  # noqa: E731
    stale = sa["staleness_ns"] * 1e-9
    print(f"clock: perf_counter - CLOCK_MONOTONIC = pub {clk[0] * 1e6:+.1f} us, sub {clk[1] * 1e6:+.1f} us")
    print(f"frames published: {len(np.unique(pa['capture_t']))} ({a.backend})")
    print(f"subscriber samples: {len(sa)} (unique seqnos {len(np.unique(sa['seqno']))})")
    print(f"pub age: {ms(pa['age'])}")
    print(f"sub age: {ms(sa['age_sub'])}")
    print(f"staleness: mean {1e3 * stale.mean():.3f} ms; mean(age_sub - staleness) = "
          f"{1e3 * np.mean(sa['age_sub'] - stale):.2f} ms (~ avail - capture)")


if __name__ == "__main__":
    main()
