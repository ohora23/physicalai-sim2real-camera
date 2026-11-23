"""dma-buf frame export from a GL-only sim process to a CUDA receiver process (Phase 4 R4).

SIM process (``DmabufExporter``; GL/EGL only -- this module never imports torch or cuda at
module level): ``n_buffers`` own FBOs, each with a GL_RGBA8 renderbuffer of the renderer's
``width x height``. ``render(slot)`` = ``mjr_render`` into MuJoCo's offscreen FBO, then
``glBlitFramebuffer`` offColor -> our renderbuffer[slot] (GPU-side copy; MuJoCo's FBO is left as
is). The blit needs a single-sample offscreen FBO: set ``model.vis.quality.offsamples = 0``
BEFORE creating the ``mujoco.Renderer`` (MSAA off; same requirement as ``interop.py``).
``export()`` turns each renderbuffer into an EGLImage (``eglCreateImageKHR(GL_RENDERBUFFER)``)
and a dma-buf fd (``eglExportDMABUFImageQueryMESA`` / ``eglExportDMABUFImageMESA``);
``send_fds(sock)`` passes all fds + metadata ONCE over a Unix socket (SCM_RIGHTS), the receiver
picks them up with ``recv_descs(sock)``.

Sync (measured 2026-09-28, RTX 5080 / driver 580): the display exposes ``EGL_KHR_fence_sync`` +
``EGL_ANDROID_native_fence_sync``, so ``render`` returns a native fence fd (stage 2: sync_file
created after the blit, ``glFlush``, ``eglDupNativeFenceFDANDROID``) instead of blocking in
``glFinish``; the receiver ``poll()``s that fd before reading (CPU wait moves to the receiver).
Without the extensions (or ``sync="finish"``) ``render`` does ``glFinish()`` and returns None.

RECEIVER process (``DmabufImporter``): EGL display only (``eglInitialize`` via
``mujoco.egl.create_initialized_egl_device_display`` -- importing ``mujoco.egl`` does NOT
initialise it; no GL context: ``eglCreateImageKHR(EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT)`` works
without one), imports each fd as an EGLImage (with the block-linear modifier,
EGL_DMA_BUF_PLANE0_MODIFIER_LO/HI) and registers it with ``cuGraphicsEGLRegisterImage`` once.
``read(slot, out)``: ``cuMemcpy2DAsync`` from the ``cuGraphicsResourceGetMappedEglFrame`` array
(CU_EGL_FRAME_TYPE_ARRAY; an EGL-registered image is always mapped, ``cuGraphicsMapResources``
on it returns CUDA_ERROR_UNKNOWN) -> flip + drop alpha into ``out`` (H, W, 3) uint8, top row first,
== ``Renderer.render()``.

Depth (``depth=True``): an R32F colour renderbuffer/texture cannot be exported -- on this driver
``eglCreateImageKHR`` itself fails with EGL_BAD_PARAMETER for R32F, R32UI and RGBA8UI (RG16F/R16
create but ``eglExportDMABUFImageQueryMESA`` fails; exportable: RGBA8 AB24, RGBA16F AB4H, RG16
RG32, RGB10_A2 AB30). So the float bits are carried in an RGBA8 renderbuffer instead: a depth-mode
``mjr_render`` (segment flags, like ``Renderer.render``), ``glBlitFramebuffer`` of MuJoCo's
GL_DEPTH32F_STENCIL8 renderbuffer (not sampleable) into our FBO whose depth attachment is a
GL_DEPTH32F_STENCIL8 *texture*, then one fullscreen-triangle GLSL pass writes
``floatBitsToUint(depth)`` as 4 unorm bytes (k/255 -> k is exact). The receiver views the bytes
as float32 (little-endian, R = low byte) and applies ``Renderer.render``'s reversed-Z -> metres
conversion (float64, float32 coefficients): bit-exact.
"""

from __future__ import annotations

import ctypes
import json
import os
import select
import socket

import mujoco
import numpy as np
from OpenGL import EGL, GL
from OpenGL.EGL.ANDROID import native_fence_sync as nfs
from OpenGL.EGL.EXT import image_dma_buf_import as imp
from OpenGL.EGL.KHR import fence_sync as fs
from OpenGL.EGL.KHR import gl_renderbuffer_image as rbi
from OpenGL.EGL.KHR import image as khr
from OpenGL.EGL.MESA import image_dma_buf_export as mesa

_MOD_LO, _MOD_HI = 0x3443, 0x3444  # EGL_DMA_BUF_PLANE0_MODIFIER_LO/HI_EXT (not in PyOpenGL's EXT module)

MAX_FDS = 250  # descriptors per send_fds message: SCM_MAX_FD is 253
RECV_BYTES = 1 << 16  # recv_descs buffer for one JSON message

_VS = """#version 330
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }"""
_FS = """#version 330
uniform sampler2D z; out vec4 o;
void main() { uint u = floatBitsToUint(texelFetch(z, ivec2(gl_FragCoord.xy), 0).r);
              o = vec4(uvec4(u, u >> 8, u >> 16, u >> 24) & 0xffu) / 255.0; }"""


def _egl_display():
    import mujoco.egl as me  # needs MUJOCO_GL=egl (PyOpenGL EGL platform)
    if me.EGL_DISPLAY is None:
        me.EGL_DISPLAY = me.create_initialized_egl_device_display()
        # PyOpenGL checks extension entry points against eglGetCurrentDisplay() or the default display;
        # with no context current (receiver) that is uninitialised -> EGL_NOT_INITIALIZED. Point it here.
        from OpenGL.raw.EGL._types import EGLQuerier
        EGLQuerier.getDisplay = lambda: me.EGL_DISPLAY  # ponytail: PyOpenGL internal, instance attribute only
    return me.EGL_DISPLAY


def _handle(p):
    return p if isinstance(p, int) else ctypes.cast(p, ctypes.c_void_p).value


def _shader(kind, src):
    s = GL.glCreateShader(kind)
    GL.glShaderSource(s, src)
    GL.glCompileShader(s)
    if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
    return s


class DmabufExporter:
    """SIM-process side of the dma-buf export pipeline; see module docstring for the full protocol."""

    def __init__(self, renderer, n_buffers=2, depth=False, sync="auto"):
        """Allocate ``n_buffers`` FBOs (colour, plus depth if requested) sized to ``renderer``.

        Args:
            renderer: mujoco.Renderer with a single-sample offscreen FBO (offSamples == 0).
            n_buffers: number of ping-pong slots to export.
            depth: also allocate and export a depth renderbuffer per slot.
            sync: "auto" (fence if the display supports it, else "finish"), "fence", or "finish".
        """
        ctx, self.r = renderer._mjr_context, renderer
        if ctx.offSamples != 0:
            raise RuntimeError(f"offscreen FBO is multisampled (offSamples={ctx.offSamples}); set "
                               "model.vis.quality.offsamples = 0 before creating the Renderer")
        W, H = self.W, self.H = renderer.width, renderer.height
        renderer._gl_context.make_current()
        self._disp = _egl_display()
        exts = set(EGL.eglQueryString(self._disp, EGL.EGL_EXTENSIONS).decode().split())
        missing = {"EGL_KHR_image_base", "EGL_KHR_gl_renderbuffer_image", "EGL_MESA_image_dma_buf_export"} - exts
        if missing:
            raise RuntimeError(f"missing EGL extensions: {sorted(missing)}")
        self.fence_supported = {"EGL_KHR_fence_sync", "EGL_ANDROID_native_fence_sync"} <= exts
        self.sync = ("fence" if self.fence_supported else "finish") if sync == "auto" else sync
        if self.sync == "fence" and not self.fence_supported:
            raise RuntimeError("sync='fence' needs EGL_KHR_fence_sync + EGL_ANDROID_native_fence_sync")
        self.depth = depth
        kinds = ("rgb", "depth") if depth else ("rgb",)
        self._rb = [{k: self._target(W, H) for k in kinds} for _ in range(n_buffers)]
        if depth:
            self._ztex = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._ztex)
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_DEPTH32F_STENCIL8, W, H)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self._zfbo = int(GL.glGenFramebuffers(1))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._zfbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                      GL.GL_TEXTURE_2D, self._ztex, 0)
            GL.glDrawBuffer(GL.GL_NONE)
            GL.glReadBuffer(GL.GL_NONE)
            self._check_fbo()
            self._prog = GL.glCreateProgram()
            for s in (_shader(GL.GL_VERTEX_SHADER, _VS), _shader(GL.GL_FRAGMENT_SHADER, _FS)):
                GL.glAttachShader(self._prog, s)
            GL.glLinkProgram(self._prog)
            if not GL.glGetProgramiv(self._prog, GL.GL_LINK_STATUS):
                raise RuntimeError(GL.glGetProgramInfoLog(self._prog).decode())
            self._vao = int(GL.glGenVertexArrays(1))
            m = renderer._model
            self._zmeta = dict(znear=float(np.float32(m.vis.map.znear * m.stat.extent)),
                               zfar=float(np.float32(m.vis.map.zfar * m.stat.extent)))
        mujoco.mjr_restoreBuffer(ctx)
        self._imgs, self._fds, self._descs = [], [], []

    def _target(self, W, H):
        """(fbo, renderbuffer): one GL_RGBA8 colour attachment."""
        rb = int(GL.glGenRenderbuffers(1))
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, W, H)
        fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)
        self._check_fbo()
        return fbo, rb

    @staticmethod
    def _check_fbo():
        st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if st != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: {int(st):#x}")

    def _blit(self, dst, mask):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.r._mjr_context.offFBO)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, dst)
        GL.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, mask, GL.GL_NEAREST)
        mujoco.mjr_restoreBuffer(self.r._mjr_context)  # mjr_render draws into whatever FBO is bound

    def render(self, slot, depth=True, fence=True):
        """Render the renderer's current scene into ``slot`` (colour, and depth if enabled).

        Args:
            slot: Renderbuffer slot index to render into.
            depth: Also render depth (requires a depth renderbuffer). Default ``True``.
            fence: End with the sync step (a native fence fd, or ``glFinish``). ``False`` only queues the
                render + blit(s) and returns None: a later ``render(..., fence=True)`` on the same GL context
                orders them (one fence per window group).

        Returns:
            A native fence fd (caller sends then closes it) in fence mode, else
            ``None`` after ``glFinish``, or ``None`` when ``fence=False``.
        """
        r, ctx = self.r, self.r._mjr_context
        r._gl_context.make_current()
        mujoco.mjr_render(r._rect, r._scene, ctx)
        self._blit(self._rb[slot]["rgb"][0], GL.GL_COLOR_BUFFER_BIT)
        if self.depth and depth:
            flags = r._scene.flags.copy()
            r._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
            r._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
            mujoco.mjr_render(r._rect, r._scene, ctx)
            np.copyto(r._scene.flags, flags)
            self._blit(self._zfbo, GL.GL_DEPTH_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._rb[slot]["depth"][0])
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_VIEWPORT_BIT | GL.GL_COLOR_BUFFER_BIT)
            for cap in (GL.GL_BLEND, GL.GL_DEPTH_TEST, GL.GL_SCISSOR_TEST, GL.GL_CULL_FACE, GL.GL_STENCIL_TEST):
                GL.glDisable(cap)
            GL.glColorMask(True, True, True, True)
            GL.glViewport(0, 0, self.W, self.H)
            GL.glUseProgram(self._prog)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._ztex)
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glBindVertexArray(0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glUseProgram(0)
            GL.glPopAttrib()
        mujoco.mjr_restoreBuffer(ctx)
        if not fence:
            return None
        if self.sync == "fence":
            s = fs.eglCreateSyncKHR(self._disp, nfs.EGL_SYNC_NATIVE_FENCE_ANDROID, None)
            GL.glFlush()  # the fence must be submitted before its fd can be duplicated
            fd = nfs.eglDupNativeFenceFDANDROID(self._disp, s)
            fs.eglDestroySyncKHR(self._disp, s)
            if fd < 0:
                raise RuntimeError(f"eglDupNativeFenceFDANDROID failed: {EGL.eglGetError():#x}")
            return fd
        GL.glFinish()
        return None

    def export(self):
        """Export each renderbuffer as a dma-buf fd; one descriptor dict per (slot, kind).

        Each dict is {slot, kind, fd, width, height, fourcc, modifier, stride, offset}
        (+ znear/zfar/orthographic for depth). Exports once (later calls return the same
        list); the fds stay owned by the exporter (closed in close()).
        """
        if self._descs:
            return self._descs
        self.r._gl_context.make_current()
        out = []
        for slot, rbs in enumerate(self._rb):
            for kind, (_, rb) in rbs.items():
                img = khr.eglCreateImageKHR(self._disp, self.r._gl_context._context,
                                            rbi.EGL_GL_RENDERBUFFER_KHR, ctypes.c_void_p(rb), None)
                self._imgs.append(img)
                fourcc, nplanes, mod = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
                if not mesa.eglExportDMABUFImageQueryMESA(self._disp, img, fourcc, nplanes, mod) or nplanes.value != 1:
                    raise RuntimeError(f"dma-buf export query failed (planes={nplanes.value})")
                fd, stride, off = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
                if not mesa.eglExportDMABUFImageMESA(self._disp, img, fd, stride, off):
                    raise RuntimeError(f"eglExportDMABUFImageMESA failed: {EGL.eglGetError():#x}")
                self._fds.append(fd.value)
                d = dict(slot=slot, kind=kind, fd=fd.value, width=self.W, height=self.H, fourcc=fourcc.value,
                         modifier=mod.value, stride=stride.value, offset=off.value)
                if kind == "depth":
                    d.update(self._zmeta, orthographic=bool(self.r._scene.camera[0].orthographic))
                out.append(d)
        self._descs = out
        return out

    def send_fds(self, sock, **meta):
        """Export and send descriptors + fds over a Unix socket (SCM_RIGHTS) in chunks.

        Sends descriptors in messages of at most 250 fds each (SCM_MAX_FD is 253)
        with JSON body ``dict(meta, descs=[...chunk without fd...], more=<bool>)``;
        meta is included in every chunk. The receiver loops until ``more`` is False.

        Args:
            sock: A connected Unix ``SOCK_SEQPACKET`` socket.
            **meta: Additional key-value pairs included in every chunk's JSON body.

        Returns:
            The full list of descriptor dicts (as returned by ``export()``).
        """
        descs = self.export()
        for start in range(0, len(descs), MAX_FDS):
            chunk = descs[start:start + MAX_FDS]
            body = json.dumps(dict(meta, descs=[{k: v for k, v in d.items() if k != "fd"} for d in chunk],
                                   more=start + MAX_FDS < len(descs)), separators=(",", ":")).encode()
            if len(body) >= RECV_BYTES:
                raise RuntimeError(f"descriptor message of {len(body)} B exceeds the {RECV_BYTES} B receive buffer")
            socket.send_fds(sock, [body], [d["fd"] for d in chunk])
        return descs

    def close(self):
        """Destroy the EGLImages, close the exported fds, and delete the GL renderbuffers/FBOs."""
        self.r._gl_context.make_current()
        for img in self._imgs:
            khr.eglDestroyImageKHR(self._disp, img)
        for fd in self._fds:
            os.close(fd)
        for rbs in self._rb:
            for fbo, rb in rbs.values():
                GL.glDeleteFramebuffers(1, [fbo])
                GL.glDeleteRenderbuffers(1, [rb])
        if self.depth:
            GL.glDeleteFramebuffers(1, [self._zfbo])
            GL.glDeleteTextures(1, [self._ztex])
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteProgram(self._prog)
        self._imgs, self._fds, self._rb, self._descs = [], [], [], []


def recv_descs(sock, maxfds=None):
    """Receiver side of ``DmabufExporter.send_fds``: (descs with fds re-attached, meta dict).

    Handles both the chunked protocol (``more`` field, ``SCM_MAX_FD``-limited chunks)
    and the original single-chunk format (absent ``more``, treated as a single message).

    Args:
        sock: A connected, **blocking** Unix ``SOCK_SEQPACKET`` socket (later chunks may not have arrived yet).
        maxfds: Per-message file-descriptor limit; default ``MAX_FDS`` (250, under SCM_MAX_FD 253).

    Returns:
        ``(descs, meta)`` where ``descs`` is the full list of descriptor dicts
        with fds re-attached, and ``meta`` is the control dict from the first
        message.
    """
    if not sock.getblocking():
        raise ValueError("recv_descs needs a blocking socket")
    all_descs, meta = [], None
    more = True
    while more:
        body, fds, _, _ = socket.recv_fds(sock, RECV_BYTES, maxfds or MAX_FDS)
        msg = json.loads(body)
        if meta is None:
            meta = msg
        descs = msg.pop("descs")
        if len(fds) != len(descs):
            for fd in fds + [d["fd"] for d in all_descs]:
                os.close(fd)
            raise RuntimeError(f"got {len(fds)} fds for {len(descs)} descriptors")
        for d, fd in zip(descs, fds):
            d["fd"] = fd
        all_descs.extend(descs)
        more = msg.pop("more", False)
    return all_descs, meta


def wait_fence(fd, timeout_ms=1000):
    """Block until a native fence fd (sync_file) signals, then close it."""
    try:
        p = select.poll()
        p.register(fd, select.POLLIN)
        if not p.poll(timeout_ms):
            raise TimeoutError("fence not signalled")
    finally:
        os.close(fd)


class DmabufImporter:
    """RECEIVER-process side of the dma-buf export pipeline; see module docstring for the full protocol."""

    def __init__(self, descs, device="cuda"):
        """Import each descriptor's dma-buf fd as an EGLImage and register it with CUDA.

        Args:
            descs: descriptor dicts from ``DmabufExporter.export``/``send_fds`` with fds
                attached (e.g. via ``recv_descs``).
            device: CUDA device string the imported frames are copied into.
        """
        import torch
        from cuda.bindings import driver as cu
        self._cu, self._torch, self.dev = cu, torch, torch.device(device)
        self._disp = _egl_display()
        torch.empty(1, device=self.dev)  # make torch's primary CUDA context current
        self._res, self._fds, self._imgs, self._desc, self._frame = {}, [], [], {}, {}
        for d in descs:
            attrs = [imp.EGL_LINUX_DRM_FOURCC_EXT, d["fourcc"], EGL.EGL_WIDTH, d["width"], EGL.EGL_HEIGHT, d["height"],
                     imp.EGL_DMA_BUF_PLANE0_FD_EXT, d["fd"], imp.EGL_DMA_BUF_PLANE0_OFFSET_EXT, d["offset"],
                     imp.EGL_DMA_BUF_PLANE0_PITCH_EXT, d["stride"]]
            if d["modifier"] not in (0, 0x00FFFFFFFFFFFFFF):  # LINEAR / INVALID need no modifier attribs
                attrs += [_MOD_LO, d["modifier"] & 0xFFFFFFFF, _MOD_HI, d["modifier"] >> 32]
            attrs += [EGL.EGL_NONE]
            img = khr.eglCreateImageKHR(self._disp, EGL.EGL_NO_CONTEXT, imp.EGL_LINUX_DMA_BUF_EXT, None,
                                        (ctypes.c_int * len(attrs))(*attrs))
            if not img:
                raise RuntimeError(f"dma-buf import failed: {EGL.eglGetError():#x}")
            self._fds.append(d["fd"])
            self._imgs.append(img)
            key = (d["slot"], d["kind"])
            self._res[key] = self._ck(cu.cuGraphicsEGLRegisterImage(
                _handle(img), cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY))
            self._desc[key] = d
            self._frame[key] = self._ck(cu.cuGraphicsResourceGetMappedEglFrame(self._res[key], 0, 0))
        H, W = d["height"], d["width"]
        self._buf = torch.empty((H, W, 4), dtype=torch.uint8, device=self.dev)
        p = self._p = cu.CUDA_MEMCPY2D()
        p.dstMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
        p.dstDevice, p.dstPitch, p.WidthInBytes, p.Height = self._buf.data_ptr(), W * 4, W * 4, H

    def _ck(self, r):
        err, *rest = r
        if err != self._cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"dma-buf import: {err.name}")
        return rest[0] if rest else None

    def _copy(self, key):
        # An EGL-registered image is permanently mapped (cuGraphicsMapResources -> CUDA_ERROR_UNKNOWN);
        # ordering vs the producer comes from the fence fd / glFinish before the notification.
        cu, f, p = self._cu, self._frame[key], self._p
        if f.frameType == cu.CUeglFrameType.CU_EGL_FRAME_TYPE_ARRAY:
            p.srcMemoryType, p.srcArray = cu.CUmemorytype.CU_MEMORYTYPE_ARRAY, f.frame.pArray[0]
        else:
            p.srcMemoryType, p.srcDevice, p.srcPitch = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE, f.frame.pPitch[0], f.pitch
        self._ck(cu.cuMemcpy2DAsync(p, self._torch.cuda.current_stream(self.dev).cuda_stream))

    def read(self, slot, out):
        """Colour of ``slot`` -> ``out`` (H, W, 3) uint8 CUDA, top row first (== Renderer.render())."""
        self._copy((slot, "rgb"))
        out.copy_(self._buf[..., :3].flip(0))
        return out

    def read_depth(self, slot, out):
        """Depth of ``slot`` -> ``out`` (H, W) float32 CUDA metres (== Renderer.render() depth mode)."""
        d = self._desc[(slot, "depth")]
        self._copy((slot, "depth"))
        z = self._buf.view(self._torch.float32)[..., 0].flip(0).double()  # raw reversed Z
        zn, zf = np.float32(d["znear"]), np.float32(d["zfar"])
        if d["orthographic"]:
            out.copy_(zf - z * (zf - zn))
        else:
            c = np.float32(-0.5) * (-(zf + zn) / (zf - zn)) - np.float32(0.5)
            dd = np.float32(-0.5) * (-(np.float32(2) * zf * zn) / (zf - zn))
            out.copy_(float(dd) / (z + float(c)))
        return out

    def close(self):
        """Unregister the CUDA resources, destroy the EGLImages, and close the imported fds."""
        for res in self._res.values():
            self._ck(self._cu.cuGraphicsUnregisterResource(res))
        for img in self._imgs:
            khr.eglDestroyImageKHR(self._disp, img)
        for fd in self._fds:
            os.close(fd)
        self._res, self._imgs, self._fds = {}, [], []
