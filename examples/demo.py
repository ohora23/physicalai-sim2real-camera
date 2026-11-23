"""Demo: ideal vs realistic render of a checkerboard, plus a 2 s timing log.

Run: MUJOCO_GL=egl .venv/bin/python examples/demo.py  →  out/ideal.png, out/real.png, out/timing.csv
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco`

import csv
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_camera import board_scene  # noqa: E402  (same scene as the integration tests)

from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera  # noqa: E402

W, H = 640, 480
REAL = CameraConfig(W, H, dist=(-0.3, 0.1, 0.001, -0.001, 0.0), vignette=1.0, exposure_ev=-0.3,
                    sigma_shot=0.02, sigma_read=0.005, sigma_prnu=0.02, sigma_dsnu=0.005,
                    black_level=0.02, gamma=2.2,
                    fps=30, latency_s=0.05, jitter_k=2, jitter_theta=0.005, drop_prob=0.05)


def main():
    os.makedirs("out", exist_ok=True)
    model, data = board_scene()
    model.opt.timestep = 0.002

    plain = mujoco.Renderer(model, H, W)
    plain.update_scene(data, camera="cam")
    cv2.imwrite("out/ideal.png", cv2.cvtColor(plain.render(), cv2.COLOR_RGB2BGR))
    plain.close()

    cam = SimCamera(model, "cam", REAL)
    cv2.imwrite("out/real.png", cv2.cvtColor(cam.render(data), cv2.COLOR_RGB2BGR))

    stream = CameraStream(cam, REAL)
    rows, ages, delivered, prev = [], [], 0, None
    for i in range(1000):  # 2 s at 500 Hz
        mujoco.mj_step(model, data)
        t = data.time
        _, capture_t, n = stream.step(data)
        if capture_t is None:
            rows.append((i, f"{t:.4f}", "", "", ""))
            continue
        if capture_t != prev:
            delivered += 1
        prev = capture_t
        ages.append(t - capture_t)
        rows.append((i, f"{t:.4f}", n, f"{capture_t:.4f}", f"{t - capture_t:.6f}"))
    cam.close()

    with open("out/timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "n", "capture_t", "age"])
        w.writerows(rows)

    print(f"frames delivered: {delivered}")
    print(f"never delivered (dropped or not yet ready at end): {round(2 * REAL.fps) - delivered}")
    print(f"mean age ms: {np.mean(ages) * 1000:.2f}")


if __name__ == "__main__":
    main()
