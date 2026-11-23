"""dma-buf bridge (Phase 4 R4): GL-only sim process -> dma-buf -> CUDA receiver process.

Two spawned processes connected by a SOCK_SEQPACKET Unix socket (abstract address):
- sim (GL/EGL only, never imports torch/cuda -- it checks its own ``sys.modules``): board scene,
  ``offsamples=0``, ``DmabufExporter`` with 2 slots (colour + depth). fds + metadata go over the
  socket once (SCM_RIGHTS). Per frame: move the board, render into slot k%2, notify
  ``(slot, seqno, capture_t)`` (+ the native fence fd in fence mode), then render the reference
  with a plain ``Renderer.render()`` (colour and depth) of the same scene while the receiver reads.
  A slot is re-rendered only after the receiver released it.
- receiver (CUDA): ``DmabufImporter``; per notification wait the fence, read the slot into a new
  tensor, re-read the PREVIOUS slot and compare it with the copy taken when it arrived (the sim
  has rendered the next frame into the other slot meanwhile: double buffer not overwritten), then
  release the previous slot.
The parent reads ``nvidia-smi`` memory.used before/after the sim starts (sim GPU memory, taken
before the receiver creates its CUDA context) and ``--query-compute-apps`` while both run,
compares frames and writes out/dmabuf/summary.json.

Run: MUJOCO_GL=egl .venv/bin/python examples/dmabuf_bridge.py [--frames 24] [--sync auto|fence|finish]
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco` (children inherit it)

import argparse
import json
import multiprocessing as mp
import re
import socket
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

W, H = 640, 480
DT = 1 / 30
NOTE = struct.Struct("<iqd")  # slot (-1 = end), seqno, capture_t
ACK = struct.Struct("<i")     # released slot
OUT = os.path.join(ROOT, "out", "dmabuf")


def sim(addr, n, sync, ready, skip, exit_ok, q):
    import mujoco
    from test_camera import board_scene

    from physicalai_sim2real_camera.dmabuf import DmabufExporter

    model, data = board_scene()
    model.vis.quality.offsamples = 0  # the blit into our renderbuffers needs a single-sample FBO
    r = mujoco.Renderer(model, H, W)
    r.update_scene(data, "cam")
    r.render()  # warm-up
    try:
        ex = DmabufExporter(r, 2, depth=True, sync=sync)
        descs = ex.export()
    except RuntimeError as e:
        q.put(dict(who="sim", skip=str(e)))
        skip.value = 1
        ready.set()
        return
    lst = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    lst.bind(addr)
    lst.listen(1)
    ready.set()
    conn, _ = lst.accept()
    ex.send_fds(conn)

    boxes = model.geom_type == mujoco.mjtGeom.mjGEOM_BOX
    ref = np.empty((n, H, W, 3), np.uint8)
    refd = np.empty((n, H, W), np.float32)
    busy, waits = [False, False], []
    for k in range(n):
        slot = k % 2
        while busy[slot]:
            busy[ACK.unpack(conn.recv(ACK.size))[0]] = False
        model.geom_pos[boxes, 0] += 0.004
        mujoco.mj_forward(model, data)
        r.update_scene(data, "cam")
        t = time.perf_counter()
        fd = ex.render(slot)  # fence mode: fd; finish mode: returns after glFinish
        waits.append(time.perf_counter() - t)
        msg = NOTE.pack(slot, k, k * DT)
        if fd is None:
            conn.send(msg)
        else:
            socket.send_fds(conn, [msg], [fd])
            os.close(fd)
        busy[slot] = True
        # reference while the receiver reads: plain Renderer.render() of the same scene
        r.render(out=ref[k])
        r.enable_depth_rendering()
        r.render(out=refd[k])
        r.disable_depth_rendering()
    conn.send(NOTE.pack(-1, n, 0.0))
    np.save(os.path.join(OUT, "ref_rgb.npy"), ref)
    np.save(os.path.join(OUT, "ref_depth.npy"), refd)
    mods = sys.modules
    q.put(dict(who="sim", pid=os.getpid(), sync=ex.sync, fence_supported=ex.fence_supported,
               wait_ms=[1e3 * w for w in waits], torch_absent="torch" not in mods,
               cuda_absent=not any(m == "cuda" or m.startswith("cuda.") for m in mods),
               fourcc=descs[0]["fourcc"].to_bytes(4, "little").decode(), modifier=hex(descs[0]["modifier"])))
    exit_ok.wait(60)
    ex.close()
    conn.close()


def receiver(addr, n, exit_ok, q):
    import torch

    from physicalai_sim2real_camera.dmabuf import DmabufImporter, recv_descs, wait_fence

    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.connect(addr)
    descs, _ = recv_descs(s)
    im = DmabufImporter(descs)
    dev = torch.device("cuda")
    rgb = np.empty((n, H, W, 3), np.uint8)
    dep = np.empty((n, H, W), np.float32)
    scratch = torch.empty((H, W, 3), dtype=torch.uint8, device=dev)
    dz = torch.empty((H, W), dtype=torch.float32, device=dev)
    seqs, retained, fence_ms, read_ms = [], [], [], []
    prev = None
    while True:
        msg, fds, _, _ = socket.recv_fds(s, NOTE.size, 1)
        slot, seq, _cap_t = NOTE.unpack(msg)
        if slot < 0:
            break
        t = time.perf_counter()
        if fds:
            wait_fence(fds[0])
        t1 = time.perf_counter()
        cur = torch.empty((H, W, 3), dtype=torch.uint8, device=dev)  # new tensor per frame; prev is kept
        im.read(slot, cur)
        im.read_depth(slot, dz)
        torch.cuda.synchronize()
        fence_ms.append(1e3 * (t1 - t))
        read_ms.append(1e3 * (time.perf_counter() - t1))
        if prev is not None:  # sim rendered this frame into the other slot while we held prev's slot
            p, p_copy, p_slot = prev
            im.read(p_slot, scratch)
            torch.cuda.synchronize()
            retained.append(bool(torch.equal(scratch, p_copy) and torch.equal(p, p_copy)))
            s.send(ACK.pack(p_slot))
        prev = (cur, cur.clone(), slot)
        i = len(seqs)
        if i < n:
            rgb[i], dep[i] = cur.cpu().numpy(), dz.cpu().numpy()
        seqs.append(seq)
    np.save(os.path.join(OUT, "recv_rgb.npy"), rgb[:len(seqs)])
    np.save(os.path.join(OUT, "recv_depth.npy"), dep[:len(seqs)])
    q.put(dict(who="receiver", pid=os.getpid(), seqs=seqs, retained=retained, fence_ms=fence_ms, read_ms=read_ms))
    exit_ok.wait(60)
    im.close()


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


def _mem_used():
    return int(_smi("--query-gpu=memory.used", "--format=csv,noheader,nounits").split()[0])


def _stats(x):
    return dict(mean=float(np.mean(x)), p95=float(np.percentile(x, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--sync", choices=["auto", "fence", "finish"], default="auto")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n, addr = a.frames, f"\0realsensor-dmabuf-{os.getpid()}"
    ctx = mp.get_context("spawn")
    q, ready, exit_ok, skip = ctx.Queue(), ctx.Event(), ctx.Event(), ctx.Value("b", 0)

    base = _mem_used()
    ps = ctx.Process(target=sim, args=(addr, n, a.sync, ready, skip, exit_ok, q))
    ps.start()
    if not ready.wait(60):
        ps.kill()
        sys.exit("sim did not come up")
    sim_mib = _mem_used() - base  # sim alone: GL context + FBOs + exported renderbuffers
    pr = ctx.Process(target=receiver, args=(addr, n, exit_ok, q))
    if skip.value:
        print(f"SKIP: {q.get(timeout=10)['skip']}")
        ps.join(10)
        return
    pr.start()
    res = {}
    try:
        for _ in range(2):
            m = q.get(timeout=90)
            res[m["who"]] = m
    except Exception:
        for p in (ps, pr):
            if p.is_alive():
                p.kill()
        sys.exit(f"bridge failed: got {sorted(res)}")
    apps = [int(x) for x in _smi("--query-compute-apps=pid", "--format=csv,noheader").split()]
    table = _smi()
    exit_ok.set()
    for p in (ps, pr):
        p.join(30)
    if any(p.exitcode != 0 for p in (ps, pr)):
        sys.exit(f"bridge failed: exit codes {[p.exitcode for p in (ps, pr)]}")

    s, r = res["sim"], res["receiver"]
    ref, got = np.load(os.path.join(OUT, "ref_rgb.npy")), np.load(os.path.join(OUT, "recv_rgb.npy"))
    refd, gotd = np.load(os.path.join(OUT, "ref_depth.npy")), np.load(os.path.join(OUT, "recv_depth.npy"))
    seqs = r["seqs"]
    exact = [bool(np.array_equal(got[i], ref[q_])) for i, q_ in enumerate(seqs)]
    exact_d = [bool(np.array_equal(gotd[i], refd[q_])) for i, q_ in enumerate(seqs)]
    row = re.search(rf"\b{s['pid']}\b.*?(\d+)MiB", table)  # GL ("G") processes appear here, not in compute-apps
    summary = dict(
        frames_sent=n, frames_received=len(seqs), bitexact_rgb=sum(exact), bitexact_depth=sum(exact_d),
        seqno_gaps=n - len(set(seqs)) + sum(1 for x, y in zip(seqs, seqs[1:]) if y != x + 1),
        retention_ok=sum(r["retained"]), retention_checks=len(r["retained"]),
        moved=bool(not np.array_equal(ref[0], ref[-1])),
        sync=s["sync"], fence_supported=s["fence_supported"],
        sim_wait_ms=_stats(s["wait_ms"]), recv_fence_wait_ms=_stats(r["fence_ms"]), recv_read_ms=_stats(r["read_ms"]),
        sim_gpu_mib=sim_mib, sim_gpu_mib_smi_table=int(row.group(1)) if row else None,
        sim_in_compute_apps=s["pid"] in apps, receiver_in_compute_apps=r["pid"] in apps,
        sim_torch_absent=s["torch_absent"], sim_cuda_absent=s["cuda_absent"],
        fourcc=s["fourcc"], modifier=s["modifier"],
        depth="exported as float32 bits in an RGBA8 renderbuffer (R32F not exportable)")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
