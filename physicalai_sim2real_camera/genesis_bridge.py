"""Genesis backend: a sim2real camera on a Genesis entity loaded from an MJCF that also defines the camera.

``GenesisCamera`` has ``SimCamera``'s stream interface (``push(data)``, ``render(data, depth=False)``, ``cam``), so a
``CameraStream`` drives it unchanged; ``data`` carries ``time`` and the physics entity's ``qpos``. Genesis renders the
margin view (``H_r x W_r`` at ``fovy_r``) and ``SimCameraBatch(model, renderer=False).ingest`` runs the image
pipeline, so K, margin, zfar, the window plan and the seeds are exactly the MuJoCo camera's.

Renders happen in a kinematic mirror scene that is never stepped: ``RigidSolver.set_qpos`` perturbs the following
physics steps even when it writes the current qpos back (measured on ALLEX: 4e-2 rad on the fingers after 7 s,
with or without the constraint warm start restored), so the physics scene is only read.

Window mode (``exposure_s``/``readout_s`` > 0): every ``push`` stores ``(time, qpos)``; a capture renders the
``base._plan`` sub-frame times from interpolated qpos (lerp, slerp for free/ball-joint quaternions), as
``StateRing`` does in MuJoCo. Genesis is imported
lazily, so this module (and its pure helpers) import without it.
"""

from __future__ import annotations

import collections
import math

import mujoco
import numpy as np

from .config import set_extrinsics
from .depth import zfar_of
from .vibration import mount_vibration


def camera_mount(model, cam_name, link_names):
    """Where to attach a Genesis camera for an MJCF camera: the nearest kept ancestor link and the offset.

    Genesis may merge joint-less bodies into their parent, so the camera body itself is not always a link. The
    offset is taken from MuJoCo world poses at ``qpos0``; MuJoCo and Genesis cameras both look along -z with +y up,
    so no axis change is needed.

    Args:
        model: ``mujoco.MjModel`` containing camera ``cam_name``.
        cam_name: MJCF camera name.
        link_names: Names of the links the Genesis entity kept.

    Returns:
        ``(link_name, offset)``: the link to attach to and the camera pose in that link's frame, ``(4, 4)``.

    Raises:
        ValueError: Unknown camera, or no ancestor body of the camera is among ``link_names``.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"camera {cam_name!r} not found")
    kept = set(link_names)
    body = int(model.cam_bodyid[cid])
    while model.body(body).name not in kept:
        if body == 0:
            raise ValueError(f"no ancestor of camera {cam_name!r} is a kept link")
        body = int(model.body_parentid[body])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    T_link, T_cam = np.eye(4), np.eye(4)
    T_link[:3, :3], T_link[:3, 3] = d.xmat[body].reshape(3, 3), d.xpos[body]
    T_cam[:3, :3], T_cam[:3, 3] = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    return model.body(body).name, np.linalg.inv(T_link) @ T_cam


def quat_starts(model):
    """Return the qpos start index of every free-joint and ball-joint quaternion (``(w, x, y, z)``) of ``model``.

    Args:
        model: ``mujoco.MjModel``.

    Returns:
        Tuple of indices into qpos; each quaternion occupies ``[i, i + 4)``.
    """
    free, ball = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)
    types = [int(x) for x in model.jnt_type]  # numpy ints do not compare equal to the mjtJoint enum
    return tuple(int(model.jnt_qposadr[j]) + (3 if types[j] == free else 0)
                 for j in range(model.njnt) if types[j] in (free, ball))


def slerp(q0, q1, a):
    """Shortest-arc spherical interpolation of unit quaternions (numpy arrays or torch tensors), normalised.

    Equals ``mj_differentiatePos`` + ``mj_integratePos`` on a ball/free joint (``StateRing.load``): MuJoCo turns
    ``q0^-1 q1`` into a rotation vector of angle <= pi and integrates the fraction ``a`` of it.
    """
    d = float((q0 * q1).sum())
    if d < 0:  # q and -q are the same rotation: take the short way round
        q1, d = -q1, -d
    th = math.acos(min(d, 1.0))
    if th < 1e-9:
        w0, w1 = 1 - a, a
    else:
        w0, w1 = math.sin((1 - a) * th) / math.sin(th), math.sin(a * th) / math.sin(th)
    q = w0 * q0 + w1 * q1
    return q / float((q * q).sum()) ** 0.5


def interp_qpos(ring, t, quats=()):
    """Return the qpos at time ``t`` from a time-ordered sequence of ``(time, qpos)``, clamped at the ends.

    Linear per coordinate (exact for hinge/slide joints and free-joint positions); the quaternions starting at the
    qpos indices ``quats`` (``quat_starts``) are slerped, as ``StateRing.load`` does through ``mj_integratePos``.

    Args:
        ring: Non-empty sequence of ``(time, qpos)``, times increasing; ``qpos`` ``(nq,)`` or ``(n_envs, nq)``
            numpy arrays or torch tensors.
        t: Query time, s.
        quats: qpos start indices of free/ball-joint quaternions.

    Returns:
        The interpolated qpos (the end sample outside the covered range).
    """
    if t <= ring[0][0]:
        return ring[0][1]
    if t >= ring[-1][0]:
        return ring[-1][1]
    j = next(k for k, (tk, _) in enumerate(ring) if tk >= t - 1e-12)
    t0, q0 = ring[j - 1]
    t1, q1 = ring[j]
    if abs(t1 - t) < 1e-12:
        return q1
    a = (t - t0) / (t1 - t0)
    q = q0 + a * (q1 - q0)
    for i in quats:
        if q.ndim == 1:
            q[i:i + 4] = slerp(q0[i:i + 4], q1[i:i + 4], a)
        else:  # (n_envs, nq): row by row, so env i is computed exactly as a single-env qpos would be
            for r in range(q.shape[0]):
                q[r, i:i + 4] = slerp(q0[r, i:i + 4], q1[r, i:i + 4], a)
    return q


class GenesisCamera:  # pragma: no cover -- needs genesis + a GPU: tests/test_genesis_bridge.py in the Genesis venv
    """sim2real camera for a Genesis rigid entity built from ``xml`` (renders in its own kinematic mirror scene).

    Call after ``gs.init``. The mirror scene is built in ``__init__``; the physics scene is never touched.

    Batched (``n_envs > 0``): the mirror scene has ``n_envs`` environments rendered split (each in its own world),
    ``data.qpos`` is ``(n_envs, nq)``, outputs gain a leading env axis, and env i's pipeline is ``SimCameraBatch``
    camera i, i.e. a single ``GenesisCamera`` with ``cfg.seed + i``. All envs capture at the same ``data.time``.
    """

    def __init__(self, xml, cam_name, cfg, dt, vis_options=None, ideal=False, extra_cameras=(), extra_morphs=(),
                 n_envs=0, headlight=None, renderer="rasterizer", madrona_lights=(), compile=False,
                 extrinsics=None):
        """Build the mirror scene, its cameras and the post-processing batch.

        Args:
            xml: Path of the MJCF (robot + scene) the physics entity was loaded from.
            cam_name: MJCF camera to emulate.
            cfg: ``CameraConfig``.
            dt: Physics step, s (rounds the window ring size, as ``StateRing`` uses the model timestep).
            vis_options: Optional ``gs.options.VisOptions`` (lights) for the mirror scene.
            ideal: Also add an ideal ``W x H`` camera at the MJCF fovy (``render_ideal``).
            extra_cameras: ``scene.add_camera`` keyword dicts for more (unattached) mirror-scene cameras
                (``self.extra``; pose the entity with ``mirror`` and render them with ``force_render=True``).
            extra_morphs: More ``gs.morphs`` for the mirror scene only (``self.extra_entities``), e.g. a calibration
                target posed with ``set_qpos`` between renders.
            n_envs: Number of environments of the batched physics scene; 0 for an unbatched one.
            headlight: Intensity of a white directional light that follows the rendering camera, as MuJoCo's
                headlight does (re-posed before every render); None for none. Unbatched rasterizer scenes only.
            renderer: ``"rasterizer"`` (Genesis' OpenGL rasterizer: envs rendered one by one and read back to the
                host) or ``"madrona"`` (Genesis' BatchRenderer: every env in one GPU pass, frames stay on the GPU).
            madrona_lights: ``scene.add_light`` keyword dicts for the Madrona scene (it ignores ``VisOptions``
                lights and ambient).
            compile: ``SimCameraBatch(compile=True)``: fused LUT gather in the window accumulation (exact) and the
                fused tone tail (may differ by 1 DN, see ``batch.py``). Worth it from tens of envs.
            extrinsics: Calibrated camera pose, ``(4, 4)`` in the camera's MJCF parent body frame, OpenCV optical
                axes (``config.set_extrinsics``); None keeps the MJCF pose.

        Raises:
            ValueError: ``headlight`` with ``n_envs > 0`` (one light node is shared by all split-rendered envs) or with
                ``renderer="madrona"`` (static lights), or an unknown ``renderer``.
        """
        import genesis as gs

        from .batch import SimCameraBatch

        if renderer not in ("rasterizer", "madrona"):
            raise ValueError(f"renderer must be 'rasterizer' or 'madrona', got {renderer!r}")
        if headlight is not None and (n_envs or renderer == "madrona"):
            # ponytail: one light node for all split envs, and Madrona's lights are fixed at build; per-env /
            # per-render light poses need support in Genesis
            raise ValueError("headlight needs an unbatched rasterizer scene")
        self.madrona = renderer == "madrona"

        self.model = mj = mujoco.MjModel.from_xml_path(str(xml))
        mj.opt.timestep = dt
        self.cam = mj.camera(cam_name).id
        if extrinsics is not None:
            set_extrinsics(mj, cam_name, extrinsics)
        self.quats = quat_starts(mj)  # Genesis' MJCF qpos layout is MuJoCo's (scripts/genesis_g1_check.py)
        self.n_envs = n_envs
        self.batch = b = SimCameraBatch(mj, cam_name, cfg, max(n_envs, 1), renderer=False, ring=1, compile=compile)
        self.half = b.base.half_window
        self.zfar = zfar_of(mj)
        # background band: a 24-bit depth buffer resolves z near the far plane to ~far^2 / near * 2^-24
        self._bg = self.zfar * (1 - 8 * self.zfar / (mj.vis.map.znear * mj.stat.extent) * 2.0 ** -24)
        if n_envs and not self.madrona:  # every env rendered in its own world, one image per env
            vis_options = (gs.options.VisOptions(split_envs=True) if vis_options is None
                           else vis_options.model_copy(update={"split_envs": True}))
        kw = {} if vis_options is None else {"vis_options": vis_options}
        if self.madrona:  # rasterizer mode of the BatchRenderer: same geometry as its ray tracer, 1.4x faster (G1)
            kw["renderer"] = gs.options.renderers.BatchRenderer(use_rasterizer=True)
        self.scene = gs.Scene(show_viewer=False, **kw)
        self.entity = self.scene.add_entity(gs.morphs.MJCF(file=str(xml)))
        link, offset = camera_mount(mj, cam_name, [lk.name for lk in self.entity.links])
        near, far = mj.vis.map.znear * mj.stat.extent, mj.vis.map.zfar * mj.stat.extent
        self.margin = self.scene.add_camera(res=(b.W_r, b.H_r), fov=float(b.base.fovy_r), near=near, far=far,
                                            GUI=False)
        # Madrona renders every batch camera at one resolution: the ideal view is a debug (rasterizer) camera there
        self.ideal = (self.scene.add_camera(res=(cfg.width, cfg.height), fov=float(mj.cam_fovy[self.cam]), near=near,
                                            far=far, GUI=False, **({"debug": True} if self.madrona else {}))
                      if ideal else None)
        self._mount = (self.entity.get_link(link), offset)
        for c in (self.margin, self.ideal):
            if c is not None:
                c.attach(*self._mount)
        # one mount vibration for every env: Genesis attaches one camera offset for the whole batch
        self._vib = mount_vibration(cfg, cfg.seed, self.cam)
        self.extra = [self.scene.add_camera(GUI=False, **k) for k in extra_cameras]
        self.extra_entities = [self.scene.add_entity(m) for m in extra_morphs]
        for light in madrona_lights if self.madrona else ():
            self.scene.add_light(**light)
        self.scene.build(n_envs=n_envs)
        self._headlight = None
        if headlight is not None:
            from genesis.ext import pyrender

            self._headlight = self.scene.visualizer.context.add_node(
                pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=float(headlight)), pose=np.eye(4))
        self.link = link
        self._ring = collections.deque(maxlen=math.ceil(2 * self.half / dt) + 3)

    def push(self, data):
        """Record ``data.qpos`` (the physics entity's qpos) at ``data.time``; ``CameraStream.step`` calls it."""
        if self._ring and data.time < self._ring[-1][0]:
            self._ring.clear()  # episode reset, as StateRing.push
        self._ring.append((data.time, data.qpos.clone() if hasattr(data.qpos, "clone") else data.qpos.copy()))

    def set_extrinsics(self, T_body_cam):
        """Move the camera to a new calibrated or perturbed mount pose (e.g. per episode).

        Args:
            T_body_cam: ``(4, 4)`` pose in the camera's MJCF parent body frame, OpenCV optical axes
                (``config.set_extrinsics``; ``config.perturb_extrinsics`` draws one).
        """
        name = self.model.camera(self.cam).name
        set_extrinsics(self.model, name, T_body_cam)
        link, offset = camera_mount(self.model, name, [lk.name for lk in self.entity.links])
        self._mount = (self.entity.get_link(link), offset)
        for c in (self.margin, self.ideal):
            if c is not None:  # attach() again only replaces the link and the offset
                c.attach(*self._mount)

    def _shake(self, t):
        if self._vib is not None:  # the margin camera at the mount pose of time t (the ideal one stays rigid)
            link, offset = self._mount
            self.margin.attach(link, offset @ self._vib.matrix(t))

    def mirror(self, qpos):
        """Pose the mirror scene's entity (e.g. before rendering ``extra`` cameras)."""
        self.entity.set_qpos(qpos)

    def _shot(self, cam, depth=False):
        """One render with a leading env axis, ``(n, H, W[, 3])`` (n = 1 unbatched).

        A CUDA tensor from the Madrona margin camera, a numpy array otherwise.
        """
        cam.move_to_attach()
        if self._headlight is not None:  # a directional light shines along its pose's -z: the camera's view axis
            self.scene.visualizer.context.set_node_pose(self._headlight, cam.transform)
        out = cam.render(rgb=not depth, depth=depth, force_render=True)  # the mirror scene never steps
        out = out[1] if depth else out[0]
        out = out.contiguous() if hasattr(out, "contiguous") else np.ascontiguousarray(out)
        return out if out.ndim == (3 if depth else 4) else out[None]  # Genesis drops the env axis when n_envs == 1

    def render_ideal(self):
        """Ideal ``W x H`` render of the mirror scene's current pose (``ideal=True``), env axis when batched."""
        out = self._shot(self.ideal)
        out = out.cpu().numpy() if hasattr(out, "cpu") else out
        return out if self.n_envs else out[0]

    def render(self, data, depth=False, as_tensor=False):
        """Capture ending at ``data.time``: window sub-renders (or one render), depth at ``capture_t``, pipeline.

        Args:
            data: Object with ``time`` and ``qpos`` (``(n_envs, nq)`` when batched); ``push(data)`` must have run
                for this time (``CameraStream`` does it).
            depth: Also return output-space depth (m).
            as_tensor: Return torch tensors on the batch device (copies of the output slots) instead of numpy arrays.

        Returns:
            ``rgb`` uint8 ``(H, W, 3)``, or ``(rgb, depth)``; ``(n_envs, ...)`` when batched. The mirror scene is
            left at ``capture_t``.
        """
        import torch

        b = self.batch
        tc = data.time - self.half
        ring = self._ring if self._ring else [(data.time, data.qpos)]
        # every sub-frame of every env goes straight into the batch's accumulators (accumulate_all): no
        # (n, n_sub, H_r, W_r, 3) stack, no per-env Python loop; Madrona frames never leave the GPU
        frame = None
        for k, t_k in enumerate([t for t, _ in b.base._plan] if b.window else [0.0]):  # sub-frame offsets from tc
            self.mirror(interp_qpos(ring, tc + t_k, self.quats))
            self._shake(tc + t_k)
            frame = torch.as_tensor(self._shot(self.margin)).to(b.dev)
            if b.window:
                b.accumulate_all(k, frame)
        self.mirror(interp_qpos(ring, tc, self.quats))
        self._shake(tc)
        z = None
        if depth or b.base._dof is not None:  # DoF needs depth even when it is not returned
            zn = self._shot(self.margin, depth=True)
            # depth-buffer precision: Genesis' background can read back below zfar_of (test rig: 54.8240 vs 54.8301 m)
            # and the pipeline recognises background by z >= zfar * (1 - 1e-6): snap the far band to zfar exactly
            zn[zn >= self._bg] = self.zfar
            z = torch.as_tensor(zn).to(b.dev)
        if b.window:
            b.claim_all(z, capture_t=tc)
        else:
            b.ingest_all(frame, z, capture_t=tc)
        slots = b.process(depth=depth).tolist()
        envs = torch.arange(b.n, device=b.dev)
        sl = torch.as_tensor(slots, device=b.dev)  # ring=1: all 0, but index per env to stay general
        rgb = b.rgb[envs, sl]
        out = (rgb, b.depth[envs, sl]) if depth else (rgb,)
        out = tuple(x if self.n_envs else x[0] for x in out)
        out = out if as_tensor else tuple(x.cpu().numpy() for x in out)
        return out if depth else out[0]
