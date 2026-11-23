"""Prism scene for the observer: a mocap head with the camera sweeping over a ChArUco prism.

``prism_scene`` loads the prism model written by ``scripts/make_prism.py``, adds a floor, lights and a
mocap body ``head`` carrying camera ``cam`` at the working distance, and returns the model, data and a
``HeadSweep`` that drives the head: an orbit around the prism axis, a vertical bob and yaw / pitch offsets with a
look-at, frozen during the 12 ``STILLS`` (stop-and-go: the stills feed the geometric and photometric observers,
the motion between them the timing observer). ``HeadPoseRing`` is the
shared-memory ring the simulator writes once per tick (time, position, quaternion, angular velocity) so an
observer process can read the head trajectory without a socket send in the 500 Hz loop.
"""

from __future__ import annotations

import json
import math
from multiprocessing import shared_memory
from pathlib import Path

import mujoco
import numpy as np

REC = 11  # t, pos(3), quat(4), omega(3)
# stop-and-go sweep: 12 still intervals of 1.0 s every 1.3 s. The geometric observer admits only frames whose
# corners move < 0.2 px during a capture, i.e. the stills; between them the head moves on so the
# prism lands on a different part of the frame (coverage). The photometric observer uses the same stills.
STILLS = tuple((1.0 + 1.3 * k, 2.0 + 1.3 * k) for k in range(12))


class HeadPoseRing:
    """Shared-memory ring of head poses: writer = sim (one record per tick), readers = observer processes."""

    def __init__(self, n=16384, name=None):
        """Create (``name=None``) or attach to (``name`` given) a ring of ``n`` records (16384 = 32 s at 500 Hz)."""
        nbytes = 8 + n * REC * 8
        if name is None:
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self.owner = True
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self.owner = False
        self.n = n
        self._count = np.ndarray((1,), np.int64, self.shm.buf, 0)
        self._rec = np.ndarray((n, REC), np.float64, self.shm.buf, 8)
        if self.owner:
            self._count[0] = 0

    @property
    def name(self):
        """Shared-memory name to hand to another process."""
        return self.shm.name

    @property
    def count(self):
        """Records written so far (including those overwritten)."""
        return int(self._count[0])

    def write(self, t, pos, quat, omega):
        """Append one record (overwrites the oldest after ``n``)."""
        k = int(self._count[0])
        self._rec[k % self.n, 0] = t
        self._rec[k % self.n, 1:4] = pos
        self._rec[k % self.n, 4:8] = quat
        self._rec[k % self.n, 8:11] = omega
        self._count[0] = k + 1

    def read(self):
        """Records written so far (oldest first), as an ``(m, 11)`` copy.

        After a wrap the oldest slot is left out: the writer may be overwriting it during the copy, which would put a
        newer time at the start and break ``np.interp``.
        """
        k = int(self._count[0])
        if k <= self.n:
            return self._rec[:k].copy()
        j = k % self.n
        return np.concatenate([self._rec[j + 1:], self._rec[:j]]).copy()

    def omega_fn(self):
        """``t -> |omega|`` interpolated over the records (robot-state clock): the timing observer's head input.

        Outside the recorded span it returns ``nan`` (``fit_delay`` skips those), never the end value.
        """
        r = self.read()
        t, w = r[:, 0], np.linalg.norm(r[:, 8:11], axis=1)
        return lambda q: np.interp(np.asarray(q, float), t, w, left=np.nan, right=np.nan)

    def close(self, unlink=None):
        """Detach; the creator also unlinks unless ``unlink=False``."""
        self.shm.close()
        if self.owner if unlink is None else unlink:
            self.shm.unlink()


def _camera_quat_looking_plus_x():
    """Quaternion of a MuJoCo camera (looks along -z, +y up) rotated to look along the body's +x with +z up."""
    R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: x_c, y_c, z_c in body axes
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def _euler_zy_quat(yaw, pitch):
    """Quaternion (w, x, y, z) of yaw about z then pitch about the new y."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


class HeadSweep:
    """Scripted head trajectory with the geometry a planar calibration needs.

    Three motions share one phase ``s`` (which pauses inside ``STILLS``): the head **orbits** the prism axis by
    ``a_orbit`` and bobs in height (so the faces are seen at 0 to ~40 deg of tilt: fx and the principal point
    separate only with tilted views), and a yaw / pitch **offset** on top of the look-at direction moves the prism
    to every part of the frame (coverage for the distortion field).
    """

    def __init__(self, pos, fovy_deg, aspect, margin=0.5, f_yaw=0.45, f_pitch=0.29, stills=STILLS,
                 a_orbit=0.6, f_orbit=0.19, a_z=0.25, f_z=0.23, target=(0.0, 0.0, None)):
        """Set amplitudes from the field of view so the prism (at the origin) reaches the corners.

        Args:
            pos: Nominal head position ``(-d, 0, z)``; ``d`` is the orbit radius and ``z`` the nominal height.
            fovy_deg: Vertical field of view.
            aspect: Image aspect ratio ``W / H``.
            margin: Fraction of the half-FOV used as the frame-position amplitude.
            f_yaw: Yaw-offset frequency, Hz, of the motion phase (which pauses during stills).
            f_pitch: Pitch-offset frequency, Hz.
            stills: Still intervals ``((t0, t1), ...)``.
            a_orbit: Orbit half-angle around the prism axis, rad (0.6 rad = 34 deg: faces tilt up to ~40 deg).
            f_orbit: Orbit frequency, Hz.
            a_z: Height bob amplitude, m.
            f_z: Height frequency, Hz.
            target: Look-at point; ``None`` for z means the nominal head height.
        """
        self.pos0 = np.asarray(pos, float)
        self.d = float(np.hypot(self.pos0[0], self.pos0[1]))
        tx, ty, tz = target
        self.target = np.array([tx, ty, self.pos0[2] if tz is None else tz])
        half_v = math.radians(fovy_deg) / 2
        half_h = math.atan(math.tan(half_v) * aspect)
        self.a_yaw, self.a_pitch = margin * half_h, margin * half_v
        self.f_yaw, self.f_pitch, self.stills = f_yaw, f_pitch, stills
        self.a_orbit, self.f_orbit, self.a_z, self.f_z = a_orbit, f_orbit, a_z, f_z

    def phase(self, t):
        """Motion phase: advances with time outside the still intervals, frozen inside them."""
        s = t
        for t0, t1 in self.stills:
            if t > t1:
                s -= t1 - t0
            elif t > t0:
                s -= t - t0
        return s

    def still(self, t):
        """True inside a still interval."""
        return any(t0 <= t <= t1 for t0, t1 in self.stills)

    def pose(self, t):
        """``(pos, quat)`` of the head at ``t``: orbit + bob, look-at the target, then the frame-position offset."""
        s = self.phase(t)
        psi = self.a_orbit * math.sin(2 * math.pi * self.f_orbit * s)
        z = self.pos0[2] + self.a_z * math.sin(2 * math.pi * self.f_z * s + 0.7)
        pos = np.array([-self.d * math.cos(psi), -self.d * math.sin(psi), z])
        f = self.target - pos
        yaw0 = math.atan2(f[1], f[0])
        pitch0 = math.atan2(-f[2], math.hypot(f[0], f[1]))
        yaw = yaw0 + self.a_yaw * math.sin(2 * math.pi * self.f_yaw * s)
        pitch = pitch0 + self.a_pitch * math.sin(2 * math.pi * self.f_pitch * s + 1.0)
        return pos, _euler_zy_quat(yaw, pitch)

    def omega(self, t, dt=1e-3):
        """Angular velocity (body frame, rad/s) by finite differences of the quaternion."""
        _, q0 = self.pose(t - dt / 2)
        _, q1 = self.pose(t + dt / 2)
        w = np.zeros(3)
        mujoco.mju_subQuat(w, q1, q0)
        return w / dt


def prism_scene(prism_xml, distance_m=0.75, fovy_deg=45.0, width=640, height=480, offsamples=0, timestep=0.002,
                prism_yaw_deg=0.0):
    """Load the prism model and add floor, lights and the camera-carrying mocap head.

    Args:
        prism_xml: Path to ``assets/prism/prism.xml`` from ``scripts/make_prism.py`` (body ``prism`` at the origin).
        distance_m: Head (camera) distance from the prism axis along -x (0.75 m: a 0.36 m face at ~0.44 m fills the
            frame with ~7 px marker cells, so faces tilted up to ~50 deg stay readable).
        fovy_deg: Camera vertical FOV.
        width: Offscreen width (``offwidth``).
        height: Offscreen height.
        offsamples: MSAA samples (0 for interop).
        timestep: Simulation step.
        prism_yaw_deg: Prism rotation about its axis. At 0 face 3 faces the head and the wedge face (5) points
            120 deg away (never seen); at -60 face 4 faces the head and the wedge face is readable (tilt <= 60 deg)
            on one side of the orbit, which the closed loop (O6) needs for the photometric observer. (-90, both at
            ~30 deg, cost O2 a third of its views and left the latency barely identifiable.)

    Returns:
        ``(model, data, sweep)`` with ``sweep`` a ``HeadSweep`` whose ``pose(t)`` drives ``data.mocap_pos/quat[0]``.
    """
    spec = mujoco.MjSpec.from_file(str(prism_xml))
    spec.option.timestep = timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max(width, 640), max(height, 480)
    spec.visual.quality.offsamples = offsamples
    spec.visual.headlight.active = 1  # light from the camera: the vertical faces stay readable at any yaw
    spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient[:] = [0.3, 0.3, 0.3]
    wb = spec.worldbody
    if prism_yaw_deg:
        a = math.radians(prism_yaw_deg) / 2
        spec.body("prism").quat[:] = [math.cos(a), 0.0, 0.0, math.sin(a)]
    prism_h = json.loads(Path(prism_xml).with_name("prism.json").read_text())["height_m"]
    sun = wb.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], specular=[0, 0, 0])
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    wb.add_light(pos=[-2, 1, 2], dir=[1, -0.5, -1], diffuse=[0.5, 0.5, 0.5], specular=[0, 0, 0])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.05], pos=[0, 0, -0.01],
                rgba=[0.6, 0.6, 0.6, 1])
    pos = [-distance_m, 0.0, prism_h / 2]
    head = wb.add_body(name="head", mocap=True, pos=pos)
    head.add_camera(name="cam", fovy=fovy_deg, quat=_camera_quat_looking_plus_x())
    model = spec.compile()
    data = mujoco.MjData(model)
    sweep = HeadSweep(pos, fovy_deg, width / height, target=(0.0, 0.0, prism_h / 2))
    data.mocap_pos[0], data.mocap_quat[0] = sweep.pose(0.0)
    mujoco.mj_forward(model, data)
    return model, data, sweep


def drive(model, data, sweep, ring=None):
    """Advance one step: set the head pose for ``data.time``, write the ring, ``mj_step``."""
    t = data.time
    pos, quat = sweep.pose(t)
    data.mocap_pos[0], data.mocap_quat[0] = pos, quat
    if ring is not None:
        ring.write(t, pos, quat, sweep.omega(t))
    mujoco.mj_step(model, data)
