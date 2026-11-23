"""Spike (2026-09-27): dma-buf export/import round trip, bit-exact against Renderer.render().

MuJoCo offColor renderbuffer -> EGLImage -> dma-buf fd -> re-import as EGLImage ->
cuGraphicsEGLRegisterImage -> CUDA copy == Renderer.render() bit-exactly. The fd round trip is
what a GL-only sim process would send (SCM_RIGHTS) to a policy process; here both ends live in
one process.
Run: MUJOCO_GL=egl .venv/bin/python scripts/spike_dmabuf_export.py   (needs cuda-python, PyOpenGL EGL ext modules)
Result on RTX 5080 / driver 580: fourcc AB24, modifier 0x300000000e08014 (NVIDIA block-linear),
CU_EGL_FRAME_TYPE_ARRAY, bit-exact.
"""
import ctypes
import os
import sys

sys.path.insert(0, 'tests')

import mujoco
import mujoco.egl as me
import numpy as np
import torch
from cuda.bindings import driver as cu
from OpenGL import EGL
from OpenGL.EGL.EXT import image_dma_buf_import as imp
from OpenGL.EGL.KHR import gl_renderbuffer_image as rbi
from OpenGL.EGL.KHR import image as khr
from OpenGL.EGL.MESA import image_dma_buf_export as mesa
from test_camera import board_scene


def ck(r):
    err, *rest = r
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if rest else None

m, d = board_scene(); m.vis.quality.offsamples = 0
H, W = 480, 640
r = mujoco.Renderer(m, H, W); ctx = r._mjr_context
mujoco.mj_forward(m, d); r.update_scene(d, 'cam'); ref = r.render().copy()
r._gl_context.make_current()
disp = me.EGL_DISPLAY; eglctx = r._gl_context._context
# 1) renderbuffer -> EGLImage
img = khr.eglCreateImageKHR(disp, eglctx, rbi.EGL_GL_RENDERBUFFER_KHR, ctypes.c_void_p(ctx.offColor), None)
print('EGLImage from offColor:', bool(img), 'egl err', hex(EGL.eglGetError()))
# 2) export dma-buf
fourcc, nplanes, mod = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
ok = mesa.eglExportDMABUFImageQueryMESA(disp, img, fourcc, nplanes, mod)
fd, stride, off = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
ok2 = mesa.eglExportDMABUFImageMESA(disp, img, fd, stride, off)
print(f'export query ok={ok} fourcc={fourcc.value:#x} ({fourcc.value.to_bytes(4, "little")}) '
      f'planes={nplanes.value} modifier={mod.value:#x}; export ok={ok2} fd={fd.value} '
      f'stride={stride.value} offset={off.value}')
# 3) import fd as a new EGLImage (this is what the policy process would do with the received fd)
attrs = [imp.EGL_LINUX_DRM_FOURCC_EXT, fourcc.value, EGL.EGL_WIDTH, ctx.offWidth, EGL.EGL_HEIGHT, ctx.offHeight,
         imp.EGL_DMA_BUF_PLANE0_FD_EXT, fd.value, imp.EGL_DMA_BUF_PLANE0_OFFSET_EXT, off.value,
         imp.EGL_DMA_BUF_PLANE0_PITCH_EXT, stride.value]
if mod.value not in (0, 0x00ffffffffffffff):
    attrs += [0x3443, mod.value & 0xffffffff, 0x3444, mod.value >> 32]  # EGL_DMA_BUF_PLANE0_MODIFIER_LO/HI_EXT
attrs += [EGL.EGL_NONE]
arr = (ctypes.c_int * len(attrs))(*attrs)
img2 = khr.eglCreateImageKHR(disp, EGL.EGL_NO_CONTEXT, imp.EGL_LINUX_DMA_BUF_EXT, None, arr)
print('EGLImage from dma-buf:', bool(img2), 'egl err', hex(EGL.eglGetError()))
# 4) CUDA: register the imported EGLImage
torch.empty(1, device='cuda')
h2 = img2 if isinstance(img2, int) else int.from_bytes(bytes(img2), "little")
res = ck(cu.cuGraphicsEGLRegisterImage(h2, cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY))
print('cuGraphicsEGLRegisterImage: ok')
frame = ck(cu.cuGraphicsResourceGetMappedEglFrame(res, 0, 0))
print('frame type', frame.frameType, 'planes', frame.planeCount, 'w,h', frame.width, frame.height,
      'pitch', frame.pitch, 'fmt', frame.eglColorFormat)
buf = torch.empty((ctx.offHeight, ctx.offWidth, 4), dtype=torch.uint8, device='cuda')
p = cu.CUDA_MEMCPY2D()
if frame.frameType == cu.CUeglFrameType.CU_EGL_FRAME_TYPE_ARRAY:
    p.srcMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_ARRAY; p.srcArray = frame.frame.pArray[0]
else:
    p.srcMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
    p.srcDevice = frame.frame.pPitch[0]
    p.srcPitch = frame.pitch
p.dstMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
p.dstDevice, p.dstPitch, p.WidthInBytes, p.Height = buf.data_ptr(), ctx.offWidth * 4, ctx.offWidth * 4, ctx.offHeight
ck(cu.cuMemcpy2D(p))
got = buf[:H, :W, :3].flip(0).cpu().numpy()
print('dma-buf round trip == Renderer.render():', np.array_equal(got, ref), 'max|d|',
      int(np.abs(got.astype(int) - ref).max()))
# second frame: move the board, re-render, re-read (registration persists)
m.geom_pos[m.geom_type == mujoco.mjtGeom.mjGEOM_BOX, 0] += 0.01
mujoco.mj_forward(m, d); r.update_scene(d, 'cam'); ref2 = r.render().copy()
ck(cu.cuMemcpy2D(p)); got2 = buf[:H, :W, :3].flip(0).cpu().numpy()
print('frame 2 == :', np.array_equal(got2, ref2), 'changed vs frame 1:', not np.array_equal(ref2, ref))
os.close(fd.value)
