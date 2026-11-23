"""Gymnasium wrapper: replace a MuJoCo env's ideal camera images with realistic ones.

Needs ``gymnasium`` (``pip install physicalai-sim2real-camera[gym]``); not imported by the package ``__init__``.
"""

from __future__ import annotations

import weakref

import cv2
import gymnasium as gym
import mujoco

from .camera import SimCamera
from .timing import CameraStream


class _StepHook:
    """Stands in for ``mujoco`` inside one module and runs a per-``data`` hook after each ``mj_step``.

    Hooks live in ``hooks[id(data)]``, a list of ``weakref.WeakMethod`` s (nested or rebound wrappers of one env
    each get called), so several wrapped envs can share the module.
    ``nstep > 1`` runs the hook once, after the last step.
    """

    def __init__(self):
        self.hooks = {}

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, model, data, *args, **kwargs):
        mujoco.mj_step(model, data, *args, **kwargs)
        for ref in self.hooks.get(id(data), ()):
            fn = ref()
            if fn is not None:
                fn(data)


def _unhook(module, hook, key, ref):
    """Drop this wrapper's ``ref`` under ``key``; put the real ``mujoco`` back in ``module`` once none is left."""
    refs = hook.hooks.get(key, [])
    if ref in refs:
        refs.remove(ref)
    if not refs:
        hook.hooks.pop(key, None)
    if not hook.hooks and module.mujoco is hook:
        module.mujoco = mujoco


class RealSensorWrapper(gym.Wrapper):
    """Swap ``obs["pixels"][key]`` for frames from ``SimCamera`` + ``CameraStream`` (optics, noise, timing).

    The wrapped env must expose ``unwrapped.model`` / ``unwrapped.data`` (``mujoco.MjModel`` / ``MjData``) and
    return a dict observation; image envs such as gym-hil's ``PandaPickCubeGymEnv(image_obs=True)`` already
    carry ``obs["pixels"]``, whose ideal frames are overwritten (the key is created otherwise).

    Timing: each stream steps after every physics step when ``substep_module`` is given, so captures fire at
    their own times (2 ms grid for gym-hil, whose 10 Hz control step spans three 30 fps captures) and
    motion-blur windows see every state. For gym-hil pass
    ``gym_hil.mujoco_gym_env``: its ``apply_action`` loop calls ``mujoco.mj_step``, and that module's
    ``mujoco`` is replaced by a proxy (``_StepHook``) until ``close()``. Without it the streams step once per
    ``env.step`` (capture times snap to the control period, and captures inside one step are missed).

    Each observation holds the newest frame delivered by its stream (sample-and-hold, latency/jitter/drops),
    resized with ``cv2.INTER_AREA`` when the config size differs from the observation's (a real driver's
    resize). ``info["camera_age_s"][key]`` is the delivered frame's age: ``data.time`` minus its
    mid-exposure capture time; ``info["ideal_pixels"]`` keeps the env's own (ideal, current-state) frames,
    empty when the env renders none. After ``reset`` the streams restart at ``data.time`` and, until their first
    frame is delivered, the held frame is a render of the reset state (the camera ran while the robot sat
    there).

    Several wrappers may share one ``substep_module``: one proxy dispatches each ``mj_step`` to the wrapper
    whose env owns that ``data``, and the module gets the real ``mujoco`` back when the last wrapper is closed
    or garbage-collected. ``observation_space`` is left as is (it must already hold the ``pixels`` keys), and
    a held frame is the same array on every step until a newer one arrives: copy before editing it in place.
    ``info`` keys are merged (``setdefault``), which assumes the env returns a fresh ``info`` dict per step.
    A wrapper replaced by a new one on the same env keeps rendering until it is closed or collected (the output
    stays correct; the render cost doubles): close the old one.

    ``depth=True`` adds ``obs["depth"][key]``: the output-space depth (metres, float32) of the same capture as
    the held colour frame, resized with ``cv2.INTER_NEAREST``. It is one extra depth render per capture; colour
    frames stay the same up to rare 1-DN rasterisation rounding on windowed (blur) cameras.
    """

    def __init__(self, env, cams, substep_module=None, scene=None, depth=False):
        """Build one ``SimCamera`` (own renderer) and ``CameraStream`` per observation key.

        Args:
            env: Gymnasium env with ``unwrapped.model`` / ``unwrapped.data`` and dict observations.
            cams: ``{obs_key: (mujoco_camera_name, CameraConfig)}``.
            substep_module: Module whose ``mujoco.mj_step`` calls drive the physics inside ``env.step``
                (``gym_hil.mujoco_gym_env`` for gym-hil), or None to step the streams once per ``env.step``.
            scene: optional ``scene_dr.SceneConfig`` applied by every camera (see ``SimCamera``).
            depth: also deliver ``obs["depth"]`` (see class docstring).
        """
        super().__init__(env)
        self._model, self._data = env.unwrapped.model, env.unwrapped.data
        self.cams = {k: SimCamera(self._model, name, cfg, scene=scene) for k, (name, cfg) in cams.items()}
        self._streams, self._frames = {}, {}
        self._depth = depth
        self._module = substep_module
        self._unhook = None
        if substep_module is not None:
            hook = substep_module.mujoco if isinstance(substep_module.mujoco, _StepHook) else _StepHook()
            key = id(self._data)
            ref = weakref.WeakMethod(self._advance)
            hook.hooks.setdefault(key, []).append(ref)
            substep_module.mujoco = hook
            self._unhook = weakref.finalize(self, _unhook, substep_module, hook, key, ref)

    def _advance(self, data):
        for k, s in self._streams.items():
            f = s.step(data)
            if f[0] is not None:
                self._frames[k] = f

    def _observe(self, obs, info):
        t = self._data.time
        pix = obs.setdefault("pixels", {})
        info.setdefault("ideal_pixels", dict(pix))  # nested wrappers: the innermost (truly ideal) frames
        ages = {}
        if not self._depth:
            obs.pop("depth", None)  # an inner depth wrapper's depth would not match these frames
        dep = obs.setdefault("depth", {}) if self._depth else None
        for k, (img, capture_t, _) in self._frames.items():
            if dep is not None:
                img, d = img
            shape = pix[k].shape if k in pix else img.shape
            pix[k] = img if img.shape == shape else cv2.resize(img, shape[1::-1], interpolation=cv2.INTER_AREA)
            if dep is not None:
                dep[k] = d if d.shape == shape[:2] else cv2.resize(d, shape[1::-1], interpolation=cv2.INTER_NEAREST)
            ages[k] = t - capture_t
        info.setdefault("camera_age_s", {}).update(ages)
        return obs, info

    def reset(self, **kwargs):
        """Reset the env, restart every stream and hold a render of the reset state."""
        obs, info = self.env.reset(**kwargs)
        d, t = self._data, self._data.time
        self._streams = {k: CameraStream(c, c.cfg, render=self._render_hook(c)) for k, c in self.cams.items()}
        for c in self.cams.values():
            c.reset_ring()  # forget the last episode: the held frame (and the first window) is this state
        self._frames = {k: (c.render(d, depth=self._depth), t, None) for k, c in self.cams.items()}
        self._advance(d)
        return self._observe(obs, info)

    def _render_hook(self, cam):
        """Stream render hook: ``(rgb, depth)`` per capture when depth is on, else the stream's default."""
        return (lambda data, n: cam.render(data, depth=True)) if self._depth else None

    def step(self, action):
        """Step the env; streams advance per physics step (``substep_module``) or once here."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._module is None:
            self._advance(self._data)
        obs, info = self._observe(obs, info)
        return obs, reward, terminated, truncated, info

    def close(self):
        """Restore the patched module, close the cameras' renderers, then the env."""
        if self._unhook is not None:
            self._unhook()  # idempotent
        for c in self.cams.values():
            c.close()
        super().close()
