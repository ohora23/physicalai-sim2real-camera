"""CUDA-GL interop: MuJoCo's offscreen colour buffer straight into a CUDA tensor.

Skips ``mjr_readPixels`` (GPU->CPU) and the upload (CPU->GPU). Needs a single-sample
offscreen FBO (``model.vis.quality.offsamples = 0`` before the Renderer is created: MSAA
off) and ``cuda-python`` (``cuda.bindings.driver``). Path: register ``offColor`` (an
RGBA8/RGB8 GL_RENDERBUFFER) once with ``cuGraphicsGLRegisterImage``; per frame
``cuGraphicsMapResources`` (orders after the GL render) -> ``cuGraphicsSubResourceGetMappedArray``
-> ``cuMemcpy2DAsync`` into a device buffer -> unmap -> flip rows (GL is bottom-up) and drop
alpha into ``out``, all on torch's current stream.

Depth: the offscreen depth renderbuffer (``GL_DEPTH32F_STENCIL8``) cannot be registered with
CUDA (``CUDA_ERROR_INVALID_VALUE``), so ``depth_into`` does ``glReadPixels(GL_DEPTH_COMPONENT)``
into a pixel-pack buffer (a GPU->GPU copy, no CPU stall) registered with
``cuGraphicsGLRegisterBuffer``, maps it, ``cuMemcpyDtoDAsync`` -> flip -> the same reversed-Z
to metres conversion as ``Renderer.render`` (float64, coefficients rounded to float32 like
glFrustum), bit-exact with it.
"""

from __future__ import annotations

import ctypes

import mujoco
import numpy as np
import torch
from cuda.bindings import driver as cu
from OpenGL import GL


def _ck(r):
    err, *rest = r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA-GL interop: {err.name}")
    return rest[0] if rest else None


class GlInterop:
    """CUDA-GL interop registration for one ``mujoco.Renderer``; see module docstring for the path.

    Not thread-safe: the copy descriptor is shared mutable state and a GL context belongs to one
    thread anyway. All users of one instance (SimCamera, SimCameraBatch) run on the same thread.
    """

    def __init__(self, renderer, device):
        """Register ``renderer``'s offscreen colour/depth buffers for CUDA-GL interop.

        Args:
            renderer: mujoco.Renderer with a single-sample offscreen FBO (offSamples == 0).
            device: CUDA device string the interop buffers/tensors are allocated on.
        """
        ctx, self.r, self.dev = renderer._mjr_context, renderer, torch.device(device)
        if ctx.offSamples != 0:
            raise RuntimeError(f"offscreen FBO is multisampled (offSamples={ctx.offSamples}); needs offsamples=0")
        renderer._gl_context.make_current()
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, ctx.offColor)
        fmt = GL.glGetRenderbufferParameteriv(GL.GL_RENDERBUFFER, GL.GL_RENDERBUFFER_INTERNAL_FORMAT)
        C = {GL.GL_RGBA8: 4, GL.GL_RGB8: 3}.get(int(fmt))
        if C is None:
            raise RuntimeError(f"offColor internal format {int(fmt):#x} is not RGBA8/RGB8")
        H, W = renderer.height, renderer.width
        self._buf = torch.empty((H, W, C), dtype=torch.uint8, device=self.dev)  # also makes torch's ctx current
        self._res = _ck(cu.cuGraphicsGLRegisterImage(
            ctx.offColor, GL.GL_RENDERBUFFER, cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY))
        p = self._p = cu.CUDA_MEMCPY2D()
        p.srcMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_ARRAY
        p.dstMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
        p.dstDevice, p.dstPitch, p.WidthInBytes, p.Height = self._buf.data_ptr(), W * C, W * C, H
        # depth: pixel-pack buffer registered with CUDA + device buffer for the raw reversed-Z
        self._pbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, H * W * 4, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._pres = _ck(cu.cuGraphicsGLRegisterBuffer(
            self._pbo, cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY))
        self._dbuf = torch.empty((H, W), dtype=torch.float32, device=self.dev)
        m = renderer._model
        self._zn, self._zf = np.float32(m.vis.map.znear * m.stat.extent), np.float32(m.vis.map.zfar * m.stat.extent)

    @property
    def channels(self):
        """Bytes per pixel of ``offColor`` (4 for RGBA8, 3 for RGB8): the row layout ``copy_color`` writes."""
        return self._buf.shape[2]

    def render_scene(self, depth=False):
        """``mjr_render`` the renderer's current scene (after ``update_scene``) into its offscreen FBO.

        Args:
            depth: Render with segmentation flags like ``Renderer.render`` does for depth (cheaper
                shading, same depth buffer); restores the scene flags afterwards.
        """
        r = self.r
        r._gl_context.make_current()
        if not depth:
            mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
            return
        flags = r._scene.flags.copy()
        r._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        r._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
        np.copyto(r._scene.flags, flags)

    def copy_color(self, dst_ptr, dst_pitch):
        """Copy ``offColor`` as-is (GL rows, bottom-up; ``channels`` bytes/px) into a device buffer.

        Map -> ``cuMemcpy2DAsync`` -> unmap on torch's current stream; no kernel. The map orders the
        copy after the GL render, the unmap orders later GL work after the copy.

        Args:
            dst_ptr: Device pointer of an (H, W, ``channels``) uint8 destination.
            dst_pitch: Destination row pitch in bytes (``W * channels`` when contiguous).
        """
        s = torch.cuda.current_stream(self.dev).cuda_stream
        _ck(cu.cuGraphicsMapResources(1, self._res, s))
        try:
            self._p.srcArray = _ck(cu.cuGraphicsSubResourceGetMappedArray(self._res, 0, 0))
            self._p.dstDevice, self._p.dstPitch = dst_ptr, dst_pitch
            _ck(cu.cuMemcpy2DAsync(self._p, s))
        finally:
            _ck(cu.cuGraphicsUnmapResources(1, self._res, s))

    def copy_depth(self, dst_ptr):
        """Copy the depth buffer's raw reversed Z (GL rows, bottom-up) into a device buffer.

        ``glReadPixels(GL_DEPTH_COMPONENT)`` into the registered pixel-pack buffer (GPU->GPU), then
        map -> ``cuMemcpyDtoDAsync`` -> unmap on torch's current stream; no kernel. Call after
        ``render_scene(depth=True)`` (GL context current). Convert with ``z_to_metres``.

        Args:
            dst_ptr: Device pointer of an (H, W) float32 destination.
        """
        H, W = self._dbuf.shape
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
        GL.glReadPixels(0, 0, W, H, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, ctypes.c_void_p(0))  # GPU->PBO
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        s = torch.cuda.current_stream(self.dev).cuda_stream
        _ck(cu.cuGraphicsMapResources(1, self._pres, s))
        try:
            ptr = _ck(cu.cuGraphicsResourceGetMappedPointer(self._pres))
            _ck(cu.cuMemcpyDtoDAsync(dst_ptr, ptr, H * W * 4, s))
        finally:
            _ck(cu.cuGraphicsUnmapResources(1, self._pres, s))

    def z_to_metres(self, z, out, orthographic=False):
        """Raw reversed Z from ``copy_depth`` -> metres, rows flipped to top-first (== ``Renderer.render``).

        Float64 math with glFrustum's float32-rounded coefficients, as ``Renderer.render``.

        Args:
            z: (..., H, W) float32 CUDA raw depth in [0, 1], GL rows (bottom-up); batches allowed.
            out: Tensor of the same shape receiving float32 metres.
            orthographic: The rendered camera is orthographic (``model.cam_projection``).

        Returns:
            ``out``.
        """
        zn, zf = self._zn, self._zf
        z = z.flip(-2).double()  # reversed Z in [0,1] (readDepthMap ZEROFAR; window == NDC)
        if orthographic:
            out.copy_(zf - z * (zf - zn))
        else:
            c = np.float32(-0.5) * (-(zf + zn) / (zf - zn)) - np.float32(0.5)
            d = np.float32(-0.5) * (-(np.float32(2) * zf * zn) / (zf - zn))
            out.copy_(float(d) / (z + float(c)))
        return out

    def render_into(self, out):
        """Render the current scene and copy it GPU->GPU into ``out``.

        ``out`` is (H, W, 3) uint8 CUDA, top row first (== ``Renderer.render()`` output
        with the same offsamples).
        """
        self.render_scene()
        self.copy_color(self._buf.data_ptr(), self._buf.shape[1] * self.channels)
        # ponytail: flip() allocates a temp from torch's caching allocator; negligible at 1-2 MB
        out.copy_(self._buf[..., :3].flip(0))
        return out

    def depth_into(self, out):
        """Render depth (segmentation flags like ``Renderer.render``) and copy it into ``out``.

        ``out`` is (H, W) float32 CUDA metres, top row first (== ``Renderer.render()``
        with depth rendering enabled).
        """
        self.render_scene(depth=True)
        self.copy_depth(self._dbuf.data_ptr())
        return self.z_to_metres(self._dbuf, out, bool(self.r._scene.camera[0].orthographic))

    def close(self):
        """Unregister the CUDA-GL resources and delete the pixel-pack buffer."""
        if self._res is not None:
            self.r._gl_context.make_current()
            _ck(cu.cuGraphicsUnregisterResource(self._res))
            _ck(cu.cuGraphicsUnregisterResource(self._pres))
            GL.glDeleteBuffers(1, [self._pbo])
            self._res = None
