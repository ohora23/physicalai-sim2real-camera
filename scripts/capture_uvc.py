"""Remote capture on a fixed UVC camera (no target needed): timing stream + exposure sweep for gamma / PTC.

Runs on the camera host (Python >= 3.8, OpenCV >= 4.2, numpy >= 1.17; no project imports)::

    python3 capture_uvc.py --out /tmp/uvc_capture.npz [--device 0] [--seconds 60]

A. Timing: ``--seconds`` of frames at a fixed manual exposure; per frame the host monotonic arrival time, the
   driver's buffer timestamp (``CAP_PROP_POS_MSEC``) and the read index.
B. Exposure sweep: for each ``exposure_absolute`` step, drop ``--settle`` frames then keep
   ``--frames`` frames of the central crop at full resolution. Gain, white balance and auto exposure are fixed
   so the scene's linear signal scales with exposure only; keep the scene still while it runs (~1 min).

Controls go through ``v4l2-ctl`` (package v4l-utils). Measured on a Logitech Webcam Pro 9000 (046d:0809): the
exposure control runs backwards (larger ``exposure_absolute`` = darker, frame time x value ~ constant, 30 fps only
from ~325 up) and ``exposure_auto_priority=0`` does not stop the frame-rate drop; gain moves in steps.

The output ``.npz`` holds ``t_arrival``, ``ts_ms``, ``exposures``, ``stacks`` (n_exp, n_frames, h, w, 3) uint8, the
controls read back after each set, and the crop box. Analyse with ``scripts/analyse_uvc.py``.
"""

import argparse
import subprocess
import time

import cv2
import numpy as np


def open_camera(device, width, height, fps):
    """Open a V4L2 camera in YUYV at the given mode; return the capture."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"cannot open /dev/video{device}")
    return cap


def v4l2(device, **ctrls):
    """Set V4L2 controls with ``v4l2-ctl`` (OpenCV's property mapping is unreliable on old UVC firmware)."""
    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}",
                    "--set-ctrl=" + ",".join(f"{k}={v}" for k, v in ctrls.items())], check=True)


def controls(device):
    """Read back every V4L2 control (``v4l2-ctl --list-ctrls``) as ``{name: value}``."""
    out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}", "--list-ctrls"], capture_output=True,
                         text=True).stdout
    return {ln.split()[0]: int(ln.rsplit("value=", 1)[1].split()[0]) for ln in out.splitlines() if "value=" in ln}


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--timing-exposure", type=int, default=335,
                    help="exposure_absolute for part A (Webcam Pro 9000: >= 325 keeps 30 fps, larger = shorter)")
    ap.add_argument("--exposures", default="335,420,500,700,1000,1400,2000")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--gain", type=int, default=255)
    ap.add_argument("--raw", action="store_true",
                    help="sharpness 0, backlight compensation off, and keep the luma (Y) plane instead of RGB")
    a = ap.parse_args()
    cap = open_camera(a.device, 640, 480, 30)
    if (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (640, 480):
        raise SystemExit(f"camera did not take 640x480: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                         f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    for _ in range(5):  # let OpenCV finish configuring before the controls are set
        cap.read()
    v4l2(a.device, exposure_auto=1, exposure_auto_priority=0, white_balance_temperature_auto=0, gain=a.gain,
         exposure_absolute=a.timing_exposure)
    if a.raw:  # in-camera sharpening mixes neighbouring pixels' noise; YUYV chroma is shared by pixel pairs
        v4l2(a.device, sharpness=0, backlight_compensation=0)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    for _ in range(a.settle):
        cap.read()
    ctl_timing = controls(a.device)
    t_arrival, ts_ms = [], []
    t_end = time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        ok, _ = cap.read()
        t = time.monotonic()
        if ok:
            t_arrival.append(t)
            ts_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    exposures = [int(x) for x in a.exposures.split(",")]
    H, W = 480, 640
    box = (W // 4, H // 4, 3 * W // 4, 3 * H // 4)  # x0, y0, x1, y1
    stacks, ctl_sweep = [], []
    for e in exposures:
        v4l2(a.device, exposure_absolute=e)
        for _ in range(a.settle):
            cap.read()
        st = []
        while len(st) < a.frames:
            ok, f = cap.read()
            if ok:
                if a.raw and f.size != 480 * 640 * 2:
                    raise SystemExit(f"--raw expects a YUYV buffer of {480 * 640 * 2} bytes, got {f.shape}")
                crop = f.reshape(480, 640, 2)[..., 0:1] if a.raw else f[..., ::-1]  # Y plane / RGB
                st.append(crop[box[1]:box[3], box[0]:box[2]].copy())
        stacks.append(np.stack(st))
        ctl_sweep.append(controls(a.device))
        print(f"exposure {e}: mean DN {stacks[-1].mean():.1f}, controls {ctl_sweep[-1]}", flush=True)
    cap.release()
    np.savez(a.out, t_arrival=np.array(t_arrival), ts_ms=np.array(ts_ms), exposures=np.array(exposures),
             stacks=np.stack(stacks), box=np.array(box), ctl_timing=str(ctl_timing), ctl_sweep=str(ctl_sweep))
    dt = np.diff(t_arrival)
    print(f"timing: {len(t_arrival)} frames in {a.seconds:.0f} s, arrival dt {1e3 * dt.mean():.2f} ms "
          f"(sd {1e3 * dt.std():.2f}), controls {ctl_timing} -> {a.out}")


if __name__ == "__main__":
    main()
