"""R4: dma-buf cross-process bridge (GL-only sim -> CUDA receiver); pytest-collectable, also runs standalone.

Runs examples/dmabuf_bridge.py (subprocess, 120 s timeout) with glFinish sync and with the native
fence (when the driver has it) and checks out/dmabuf/summary.json. Skips (exit 0) when cuda-python
or the PyOpenGL EGL extension modules are missing, or the driver lacks dma-buf export.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_checks(sync):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "dmabuf_bridge.py"), "--sync", sync],
                       env=dict(os.environ, MUJOCO_GL="egl"), capture_output=True, text=True, timeout=120)
    if p.returncode == 0 and p.stdout.startswith("SKIP"):
        return p.stdout.strip()
    assert p.returncode == 0, p.stdout + p.stderr
    with open(os.path.join(ROOT, "out", "dmabuf", "summary.json")) as f:
        s = json.load(f)
    n = s["frames_sent"]
    print(f"[{s['sync']}] bit-exact rgb {s['bitexact_rgb']}/{n} depth {s['bitexact_depth']}/{n}, "
          f"gaps {s['seqno_gaps']}, retention {s['retention_ok']}/{s['retention_checks']}, "
          f"sim wait {s['sim_wait_ms']['mean']:.3f} ms (p95 {s['sim_wait_ms']['p95']:.3f}), "
          f"sim GPU {s['sim_gpu_mib']} MiB, fence supported {s['fence_supported']}")
    assert n >= 20 and s["frames_received"] == n and s["moved"]
    assert s["bitexact_rgb"] == n and s["bitexact_depth"] == n
    assert s["seqno_gaps"] == 0
    assert s["retention_checks"] == n - 1 and s["retention_ok"] == n - 1
    assert not s["sim_in_compute_apps"] and s["receiver_in_compute_apps"]
    assert s["sim_torch_absent"] and s["sim_cuda_absent"]
    assert s["sim_gpu_mib"] <= 250, s["sim_gpu_mib"]
    return s


def _skip(msg):
    """Skip under pytest; print and return False when run as a script."""
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    print(msg if msg.startswith("SKIP") else f"SKIP: {msg}")
    return False


def _run_all():
    try:
        from cuda.bindings import driver  # noqa: F401
        from OpenGL.EGL.ANDROID import native_fence_sync  # noqa: F401
        from OpenGL.EGL.EXT import image_dma_buf_import  # noqa: F401
        from OpenGL.EGL.MESA import image_dma_buf_export  # noqa: F401
    except ImportError as e:
        return _skip(str(e))
    s = _run_checks("finish")
    if isinstance(s, str):
        return _skip(s)
    if s["fence_supported"]:
        _run_checks("fence")
    else:
        print("NOTE: EGL native fence sync unavailable; glFinish only")
    return True


def test_dmabuf():
    """pytest entry point (skips via ``pytest.skip`` when prerequisites are missing)."""
    _run_all()


def main():
    if _run_all():
        print("test_dmabuf: PASS")


def test_fd_chunking(n=251):
    """send_fds/recv_descs split more than MAX_FDS descriptors over several messages (no GPU: /dev/null fds)."""
    import json
    import socket
    import types

    from physicalai_sim2real_camera.dmabuf import MAX_FDS, RECV_BYTES, DmabufExporter, recv_descs
    assert n > MAX_FDS  # so the transfer takes two messages
    descs = [dict(slot=i, kind="rgb", fd=os.open(os.devnull, os.O_RDONLY), width=4, height=2) for i in range(n)]
    exporter = types.SimpleNamespace(export=lambda: descs)
    opened = []
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        # raw messages: `more` must be True then False (a wrong flag would make recv_descs wait forever)
        DmabufExporter.send_fds(exporter, a, K=3)
        flags = []
        for _ in range(2):
            body, fds, _, _ = socket.recv_fds(b, RECV_BYTES, MAX_FDS)
            opened += fds
            flags.append(json.loads(body)["more"])
        assert flags == [True, False], flags
        DmabufExporter.send_fds(exporter, a, K=3, n_sub=11)
        got, meta = recv_descs(b)
        opened += [d["fd"] for d in got]
        assert meta == dict(K=3, n_sub=11), meta  # first message's meta, `descs`/`more` removed
        assert [d["slot"] for d in got] == list(range(n))
        b.setblocking(False)
        try:
            recv_descs(b)
        except ValueError:
            pass
        else:
            raise AssertionError("recv_descs accepted a non-blocking socket")
    finally:
        for fd in [d["fd"] for d in descs] + opened:
            os.close(fd)
        a.close()
        b.close()
    print(f"R4-chunk: {n} descriptors > MAX_FDS={MAX_FDS} in 2 messages (more True, False), all received in order")


if __name__ == "__main__":
    test_fd_chunking()
    main()
