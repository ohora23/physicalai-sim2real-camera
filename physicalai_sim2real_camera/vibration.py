"""Camera-mount vibration: a per-axis second-order resonant random process, deterministic per seed.

Models the bracket / neck flex that a rigid-body simulator lacks (the body's own motion is already in the render).
Each of six axes (a rotation vector in rad and a translation in m, both in the MuJoCo camera frame: x right, y up,
-z along the view) follows ``x'' + 2 zeta w x' + w^2 x = s n(t)`` with ``w = 2 pi hz`` and ``n`` white noise; ``s``
is set so the stationary RMS is the requested one (the stationary variance of that system is ``s^2 / (4 zeta w^3)``).
The process is integrated at a fixed rate (semi-implicit Euler, 1 kHz by default) from its stationary distribution,
in 10 s chunks with one generator, and linearly interpolated: a time gives the same offset whatever order times are
queried in. The renderers apply the offset of each sub-render's own time, so motion blur and the rolling shutter
see the same vibration.
"""

from __future__ import annotations

import math

import numpy as np


class MountVibration:
    """Resonant random offset of a camera mount."""

    def __init__(self, rot_deg, trans_m, hz, zeta, seed, rate=1000.0, keep_s=30.0):
        """Start the process from its stationary distribution.

        Args:
            rot_deg: RMS rotation per axis, degrees.
            trans_m: RMS translation per axis, metres.
            hz: Natural frequency of the mount, Hz.
            zeta: Damping ratio.
            seed: Seed of the generator (anything ``numpy.random.default_rng`` takes).
            rate: Integration rate, Hz.
            keep_s: Seconds of history to keep in memory (default 30.0). Older samples are pruned; querying
                a time outside the kept window regenerates the process from the stored seed.

        Raises:
            ValueError: A negative RMS, ``hz`` or ``zeta`` not positive, ``keep_s`` not positive,
                or ``rate`` below ``10 * hz``.
        """
        if rot_deg < 0 or trans_m < 0 or hz <= 0 or zeta <= 0 or keep_s <= 0 or rate < 10 * hz:
            raise ValueError(f"need rot_deg, trans_m >= 0, hz, zeta, keep_s > 0, rate >= 10 hz; got {rot_deg}, "
                             f"{trans_m}, {hz}, {zeta}, {keep_s}, {rate}")
        self.active = rot_deg > 0 or trans_m > 0
        if not self.active:
            return
        self.rate, self._w, self._zeta, self.keep_s = float(rate), 2 * math.pi * hz, zeta, keep_s
        self._rms = np.array([math.radians(rot_deg)] * 3 + [trans_m] * 3)
        self._s = self._rms * math.sqrt(4 * zeta * self._w ** 3)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._x = (self._rms * self._rng.standard_normal(6))[None]  # sample k at time k / rate
        self._v = self._rms * self._w * self._rng.standard_normal(6)
        self._i0 = 0  # absolute index of self._x[0]

    def _extend(self, n):
        """Append chunks of 10 s until absolute sample ``n`` exists, then prune leading samples beyond ``keep_s``."""
        dt, chunk = 1 / self.rate, int(10 * self.rate)
        while self._i0 + len(self._x) <= n:
            noise = self._s * self._rng.standard_normal((chunk, 6)) / math.sqrt(dt)
            x, v, out = self._x[-1].copy(), self._v, np.empty((chunk, 6))
            for k in range(chunk):  # ponytail: plain loop, ~20 ms per simulated 10 s; vectorise if it ever shows
                v = v + (noise[k] - 2 * self._zeta * self._w * v - self._w ** 2 * x) * dt
                x = x + v * dt
                out[k] = x
            self._x, self._v = np.concatenate([self._x, out]), v
        max_keep = int(self.keep_s * self.rate) + chunk
        if len(self._x) > max_keep:
            n_drop = len(self._x) - max_keep
            self._i0 += n_drop
            self._x = self._x[n_drop:]

    def offset(self, t):
        """6-vector (rotation vector rad, translation m) at time ``t`` s (clamped at 0); zeros when off.

        If ``t`` falls outside the kept history window, the process is regenerated from the stored seed so
        that output is bit-identical to the original implementation.  A jump beyond ``3600 * rate`` new samples
        in one call raises ``ValueError``.
        """
        if not self.active:
            return np.zeros(6)
        k = max(float(t), 0.0) * self.rate
        i = int(k)
        # Check if either needed sample (i or i+1) is outside the kept window
        if i < self._i0 or i + 1 >= self._i0 + len(self._x):
            self._regenerate_and_extend(i)
        f = k - i
        ri = i - self._i0
        return self._x[ri] * (1 - f) + self._x[ri + 1] * f

    def _restart(self):
        """Back to time 0 from the stored seed: the same draws as ``__init__``, so every value repeats exactly."""
        self._rng = np.random.default_rng(self._seed)
        self._x = (self._rms * self._rng.standard_normal(6))[None]
        self._v = self._rms * self._w * self._rng.standard_normal(6)
        self._i0 = 0

    def _regenerate_and_extend(self, i):
        """Make samples ``i`` and ``i + 1`` available: restart for a time before the kept window, else extend."""
        if i < self._i0:
            self._restart()
        missing = i + 2 - (self._i0 + len(self._x))
        if missing > 3600 * self.rate:
            raise ValueError(f"t = {i / self.rate:g} s needs {missing} new samples in one call; the limit is one hour "
                             f"({3600 * self.rate:g}) - a wall-clock time passed as sim time?")
        self._extend(i + 1)  # keeps keep_s + one chunk >= 2 samples, so i stays inside the window

    def matrix(self, t):
        """4x4 offset of the mount at time ``t`` in the camera frame (Rodrigues rotation, translation); I when off."""
        return _matrix(self.offset(t)) if self.active else np.eye(4)


def _matrix(o):
    """4x4 transform of a 6-vector offset (rotation vector, translation) by Rodrigues."""
    T = np.eye(4)
    a = float(np.linalg.norm(o[:3]))
    if a > 0:
        k = o[:3] / a
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    T[:3, 3] = o[3:]
    return T


def mount_vibration(cfg, seed, cam):
    """Mount vibration for ``cfg``'s ``mount_vib_*`` fields, seeded by (``seed``, ``cam``); None when off."""
    v = MountVibration(cfg.mount_vib_rot_deg, cfg.mount_vib_trans_m, cfg.mount_vib_hz, cfg.mount_vib_zeta,
                       seed=[int(seed), int(cam), 0x7669])
    return v if v.active else None


def shake_glcameras(scene, T):
    """Move both eyes of an ``mjvScene`` by ``T`` (4x4 in the MuJoCo camera frame), after ``update_scene``."""
    for k in range(2):
        g = scene.camera[k]
        fwd, up = np.array(g.forward, float), np.array(g.up, float)
        R = np.column_stack([np.cross(fwd, up), up, -fwd])
        Rn = R @ T[:3, :3]
        g.pos[:] = np.array(g.pos, float) + R @ T[:3, 3]
        g.forward[:], g.up[:] = -Rn[:, 2], Rn[:, 1]
