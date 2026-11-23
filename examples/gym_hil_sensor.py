"""gym-hil Franka pick-cube with realistic front + wrist cameras (``RealSensorWrapper``): ideal vs real.

A scripted pick (proportional control on the true block position) moves the arm; every step the env's own
ideal frames (current state) and the wrapper's realistic frames (latency, jitter, drops, motion blur,
rolling shutter, distortion, vignetting, noise) are kept. Writes ``compare.png`` (rows: ideal / real per
camera, columns: six moments) and ``summary.json`` (frame-age statistics, per-step time, success).

gym-hil pins MuJoCo < 3.9 (``MjData.qM``), so it runs in its own venv::

    uv venv .venv-hil --python 3.12
    uv pip install --python .venv-hil/bin/python gym-hil "opencv-python-headless>=4.8"
    uv pip install --python .venv-hil/bin/python --no-deps -e .
    MUJOCO_GL=egl .venv-hil/bin/python examples/gym_hil_sensor.py --out out/gym_hil

Preset values are VIRTUAL (the D435i-like set of ``examples/g1_walk.py`` at 256 x 256, blur 4 x 2 bands);
real values come from calibration (scripts/calibrate.py, docs/capture_protocol.md).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco`

import argparse
import json
import time

import cv2
import gym_hil.mujoco_gym_env as hil_env
import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv

from physicalai_sim2real_camera import CameraConfig
from physicalai_sim2real_camera.gym_wrapper import RealSensorWrapper

REALISTIC = dict(width=256, height=256, dist=(-0.05, 0.01, 0.0005, -0.0005, 0.0), vignette=0.4, gamma=2.2,
                 sigma_shot=0.02, sigma_read=0.004, sigma_prnu=0.01, sigma_dsnu=0.003, black_level=0.01, fps=30,
                 latency_s=0.06, jitter_k=2, jitter_theta=0.003, drop_prob=0.02, exposure_s=0.008,
                 readout_s=0.020, blur_samples=4, rs_bands=2)
CAMS = {"front": "front", "wrist": "handcam_rgb"}


def scripted_action(data, phase):
    """Pick by P-control on the true block position: above -> down -> close -> lift. Returns (action, phase)."""
    tcp, block = data.sensor("2f85/pinch_pos").data, data.sensor("block_pos").data
    goal = block + [0, 0, 0.10] if phase == 0 else block if phase == 1 else tcp if phase == 2 else block + [0, 0, 0.2]
    err = goal - tcp
    a = np.zeros(7, np.float32)
    a[:3] = np.clip(0.5 * err, -0.01, 0.01)
    a[6] = 0.05 if phase >= 2 else -0.05
    if phase < 2 and np.linalg.norm(err) < 0.01:
        phase += 1
    elif phase == 2 and data.ctrl[-1] > 0.9 * 255:  # gripper actuator is last (fingers_actuator)
        phase = 3
    return a, phase


def run(steps=150, seed=0, out="out/gym_hil"):
    """Run the scripted pick with the wrapper; write compare.png and summary.json; return the summary."""
    os.makedirs(out, exist_ok=True)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    cfg = CameraConfig(**REALISTIC, seed=seed)  # seed: FPN, noise, jitter and drops
    env = RealSensorWrapper(PandaPickCubeGymEnv(image_obs=True, seed=seed),
                            {k: (cam, cfg) for k, cam in CAMS.items()}, substep_module=hil_env)
    try:
        obs, info = env.reset(seed=seed)
        data = env.unwrapped.data
        phase, ages, frames, success = 0, {k: [] for k in CAMS}, [], False
        t0 = time.perf_counter()
        for _ in range(steps):
            a, phase = scripted_action(data, phase)
            obs, _, term, trunc, info = env.step(a)
            frames.append({k: (info["ideal_pixels"][k], obs["pixels"][k]) for k in CAMS})
            for k in CAMS:
                ages[k].append(info["camera_age_s"][k])
            success |= bool(info.get("succeed"))
            if term or trunc:
                break
        ms = 1e3 * (time.perf_counter() - t0) / len(frames)
    finally:
        env.close()
    keep = np.linspace(min(10, len(frames) - 1), len(frames) - 1, 6).round().astype(int)
    rows = [np.hstack([frames[i][k][j] for i in keep]) for k in CAMS for j in (0, 1)]
    cv2.imwrite(os.path.join(out, "compare.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    summary = dict(steps=len(frames), ms_per_step=ms, success=success, keep_steps=keep.tolist(),
                   rows=[f"{k}/{j}" for k in CAMS for j in ("ideal", "real")],
                   age_s={k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
                          for k, v in ages.items()})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/gym_hil")
    a = ap.parse_args()
    print(json.dumps(run(a.steps, a.seed, a.out), indent=1))


if __name__ == "__main__":
    main()
