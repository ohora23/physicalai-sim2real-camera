"""Spike (2026-09-28, R5): can the GL sim render straight into a udmabuf (Axon's host pool memory)?

Axon's ``UDMABUF`` pool buffers are memfd pages wrapped by ``/dev/udmabuf`` (``UDMABUF_CREATE``); the Python
bindings do not expose the per-buffer dma-buf fd, so this spike makes the same kind of buffer itself
(memfd + F_SEAL_SHRINK + UDMABUF_CREATE) and tries the reverse direction of R4:
dma-buf fd -> ``eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, AB24, LINEAR)`` -> ``glEGLImageTargetRenderbufferStorageOES``
-> FBO colour attachment -> blit MuJoCo's offColor into it -> ``glFinish`` -> read the memfd pages on the CPU
and compare with ``Renderer.render()`` (bottom-up rows, RGBA).
Each step prints ok / the failing call and its EGL/GL error; ``eglQueryDmaBufModifiersEXT`` shows which
modifiers the driver accepts for AB24 and whether they are ``external_only`` (sampling only, not renderable).

Run: MUJOCO_GL=egl .venv/bin/python scripts/spike_udmabuf_render_target.py
"""
import ctypes
import fcntl
import mmap
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco
import mujoco.egl as me
import numpy as np
from OpenGL import EGL, GL
from OpenGL.EGL.EXT import image_dma_buf_import as imp
from OpenGL.EGL.KHR import image as khr
from test_camera import board_scene

from physicalai_sim2real_camera.dmabuf import DmabufExporter

H, W = 480, 640
AB24 = 0x34324241  # DRM_FORMAT_ABGR8888 == GL_RGBA8 byte order (R, G, B, A)
UDMABUF_CREATE = (1 << 30) | (24 << 16) | (ord("u") << 8) | 0x42  # _IOW('u', 0x42, struct udmabuf_create)
F_ADD_SEALS, F_SEAL_SHRINK = 1033, 0x0002


def udmabuf(size):
    """Create a sealed memfd and wrap it with /dev/udmabuf into a dma-buf fd, as Axon's UDMABUF backend does."""
    mfd = os.memfd_create("spike", os.MFD_ALLOW_SEALING)
    os.ftruncate(mfd, size)
    fcntl.fcntl(mfd, F_ADD_SEALS, F_SEAL_SHRINK)
    dev = os.open("/dev/udmabuf", os.O_RDWR)
    try:
        fd = fcntl.ioctl(dev, UDMABUF_CREATE, bytearray(struct.pack("<IIQQ", mfd, 0, 0, size)), True)  # -> new fd
    finally:
        os.close(dev)
    return mfd, fd


def step(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        sys.exit(0)  # the spike's outcome is the first failing call


def positive_control(r, disp, target, ref):
    """Same import + glEGLImageTargetRenderbufferStorageOES on a block-linear GPU dma-buf (R4's exporter)."""
    d = DmabufExporter(r, 1).export()[0]
    attrs = [imp.EGL_LINUX_DRM_FOURCC_EXT, d["fourcc"], EGL.EGL_WIDTH, W, EGL.EGL_HEIGHT, H,
             imp.EGL_DMA_BUF_PLANE0_FD_EXT, d["fd"], imp.EGL_DMA_BUF_PLANE0_OFFSET_EXT, d["offset"],
             imp.EGL_DMA_BUF_PLANE0_PITCH_EXT, d["stride"], 0x3443, d["modifier"] & 0xFFFFFFFF,
             0x3444, d["modifier"] >> 32, EGL.EGL_NONE]
    img = khr.eglCreateImageKHR(disp, EGL.EGL_NO_CONTEXT, imp.EGL_LINUX_DMA_BUF_EXT, None,
                                (ctypes.c_int * len(attrs))(*attrs))
    rb, fbo = int(GL.glGenRenderbuffers(1)), int(GL.glGenFramebuffers(1))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glGetError()
    target(GL.GL_RENDERBUFFER, img if isinstance(img, int) else ctypes.cast(img, ctypes.c_void_p).value)
    err = GL.glGetError()
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)
    st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, r._mjr_context.offFBO)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
    GL.glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
    got = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)[::-1]
    mujoco.mjr_restoreBuffer(r._mjr_context)
    print(f"control: block-linear dma-buf (modifier {d['modifier']:#x}) import {'ok' if img else 'FAIL'}, "
          f"glEGLImageTargetRenderbufferStorageOES GL error {err:#x}, FBO status {int(st):#x}, "
          f"render through it == Renderer.render(): {np.array_equal(got, ref)}")


def main():
    m, d = board_scene()
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, H, W)
    mujoco.mj_forward(m, d)
    r.update_scene(d, "cam")
    ref = r.render().copy()
    r._gl_context.make_current()
    disp = me.EGL_DISPLAY
    exts = EGL.eglQueryString(disp, EGL.EGL_EXTENSIONS).decode().split()
    step("EGL_EXT_image_dma_buf_import", "EGL_EXT_image_dma_buf_import" in exts)
    gl_exts = {GL.glGetStringi(GL.GL_EXTENSIONS, i).decode() for i in range(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS))}
    step("GL_OES_EGL_image", "GL_OES_EGL_image" in gl_exts)

    if "EGL_EXT_image_dma_buf_import_modifiers" in exts:
        q = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(
            ctypes.cast(EGL.eglGetProcAddress(b"eglQueryDmaBufModifiersEXT"), ctypes.c_void_p).value)
        n = ctypes.c_int()
        dp = ctypes.cast(disp, ctypes.c_void_p).value if not isinstance(disp, int) else disp
        q(dp, AB24, 0, None, None, ctypes.byref(n))
        mods, ext_only = (ctypes.c_uint64 * n.value)(), (ctypes.c_uint * n.value)()
        q(dp, AB24, n.value, mods, ext_only, ctypes.byref(n))
        print(f"info AB24 modifiers: {[(hex(x), bool(e)) for x, e in zip(mods, ext_only)]} (modifier, external_only)")

    target = ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_void_p)(
        ctypes.cast(EGL.eglGetProcAddress(b"glEGLImageTargetRenderbufferStorageOES"), ctypes.c_void_p).value)
    mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
    positive_control(r, disp, target, ref)

    size = H * W * 4
    try:
        mfd, fd = udmabuf(size)
    except OSError as e:
        step("UDMABUF_CREATE", False, str(e))
    step("UDMABUF_CREATE", True, f"memfd {mfd} -> dma-buf fd {fd}")
    attrs = [imp.EGL_LINUX_DRM_FOURCC_EXT, AB24, EGL.EGL_WIDTH, W, EGL.EGL_HEIGHT, H,
             imp.EGL_DMA_BUF_PLANE0_FD_EXT, fd, imp.EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
             imp.EGL_DMA_BUF_PLANE0_PITCH_EXT, W * 4, 0x3443, 0, 0x3444, 0,  # modifier LINEAR
             EGL.EGL_NONE]
    img = khr.eglCreateImageKHR(disp, EGL.EGL_NO_CONTEXT, imp.EGL_LINUX_DMA_BUF_EXT, None,
                                (ctypes.c_int * len(attrs))(*attrs))
    step("eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, AB24, LINEAR)", bool(img), f"EGL error {EGL.eglGetError():#x}")

    rb = int(GL.glGenRenderbuffers(1))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glGetError()
    target(GL.GL_RENDERBUFFER, img if isinstance(img, int) else ctypes.cast(img, ctypes.c_void_p).value)
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:  # also try the texture path before giving up (external_only rules both out)
        tex2d = ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_void_p)(
            ctypes.cast(EGL.eglGetProcAddress(b"glEGLImageTargetTexture2DOES"), ctypes.c_void_p).value)
        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        tex2d(GL.GL_TEXTURE_2D, img if isinstance(img, int) else ctypes.cast(img, ctypes.c_void_p).value)
        print(f"info glEGLImageTargetTexture2DOES(GL_TEXTURE_2D): GL error {GL.glGetError():#x}")
    step("glEGLImageTargetRenderbufferStorageOES", err == GL.GL_NO_ERROR, f"GL error {err:#x}")
    fbo = int(GL.glGenFramebuffers(1))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)
    st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    step("FBO complete", st == GL.GL_FRAMEBUFFER_COMPLETE, f"status {int(st):#x}")

    mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, r._mjr_context.offFBO)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
    GL.glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    GL.glFinish()
    err = GL.glGetError()
    step("blit offColor -> udmabuf renderbuffer", err == GL.GL_NO_ERROR, f"GL error {err:#x}")
    with mmap.mmap(mfd, size, prot=mmap.PROT_READ) as mm:
        got = np.frombuffer(mm, np.uint8).reshape(H, W, 4)[::-1, :, :3].copy()  # GL rows are bottom-up
    same = np.array_equal(got, ref)
    step("udmabuf pages == Renderer.render()", same,
         f"mismatching bytes {int((got != ref).sum())} of {got.size}, max |diff| "
         f"{int(np.abs(got.astype(int) - ref).max())}, nonzero {int((got != 0).sum())}")
    print("RESULT: GL renders straight into a udmabuf -> the sim->virtual-sensor hop could be Axon-native")


if __name__ == "__main__":
    main()
