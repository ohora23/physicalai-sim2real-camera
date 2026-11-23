"""Apptronik Apollo head camera via physicalai-sim2real-camera in Genesis (and in MuJoCo for comparison).

A D435-class camera is mounted on Apollo's head (``neck_pitch_link``, the model's own OAK-D cameras sit on the torso
and look down); a table with coloured boxes stands in front, taller boxes ring the room 3.3 m away. The motion is a
kinematic replay (no physics, the base stays at the ``stand`` keyframe): the neck scans, turns fast in three windows
(8-12, 18-22, 28-33 s) and the right arm waves in view (13-18.5 s), so blur, rolling shutter and latency show.
Captures (ideal at capture_t, pipeline output, depth, age) go to ``out/genesis_apollo``; ``--video`` writes a
3-panel mp4 (external view, ideal now, the frame the policy holds) and ``--preview`` writes MuJoCo stills.

Asset: ``bash scripts/fetch_menagerie.sh`` (apptronik_apollo, Apache-2.0). Genesis runs in its own venv
(``examples/genesis_allex.py`` has the setup line).

Run: ``.venv-genesis/bin/python examples/genesis_apollo.py --sim genesis --cfg calibrated --seconds 34 --video
out/genesis_apollo/apollo.mp4``
"""

import argparse
import collections
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco`

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APOLLO = ROOT / "third_party" / "mujoco_menagerie" / "apptronik_apollo"
OUT = ROOT / "out" / "genesis_apollo"
CAM = "head_cam"
W, H, DT, FPS = 640, 480, 0.002, 30
CAM_WORLD, CAM_PITCH_DEG = (0.14, 0.0, 1.62), 35.0  # at the stand pose: in front of the face plate, pitched down
BOXES = [  # (name, centre, half size, rgba): a table in reach, then a ring of taller boxes for depth range and edges
    ("table", (0.85, 0.0, 0.36), (0.30, 0.55, 0.36), (0.55, 0.42, 0.30, 1)),
    ("red", (0.75, 0.25, 0.78), (0.06, 0.06, 0.06), (1, 0, 0, 1)),
    ("green", (0.95, -0.05, 0.80), (0.08, 0.08, 0.08), (0, 1, 0, 1)),
    ("blue", (0.70, -0.30, 0.77), (0.05, 0.05, 0.05), (0, 0, 1, 1)),
    ("yellow", (0.98, 0.35, 0.79), (0.07, 0.04, 0.07), (1, 1, 0, 1)),
] + [(f"far{k}", (3.3 * math.cos(math.radians(-70 + 20 * k)), 3.3 * math.sin(math.radians(-70 + 20 * k)), 0.35),
      (0.2, 0.2, 0.35), [(1, 0, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1), (0, 0, 1, 1)][k % 4]) for k in range(8)]
EXT = dict(res=(1280, 720), pos=(2.1, -1.7, 1.75), lookat=(0.45, 0.0, 1.0), fov=45)  # --video external view
DIST = (-0.05, 0.01, 0.0005, -0.0005, 0.0)
CFGS = ("geom", "static", "wingeom", "calibrated")


def make_cfg(name):
    """Camera config ``name`` (one of ``CFGS``).

    ``calibrated`` is ``CameraConfig.preset("calibrated")``: rectified (no distortion), every effect a calibration does
    not remove on, depth included. The others are each the example distortion ``DIST`` (raw frames; it is not part of
    the preset) plus one effect group at the preset's typical values, not cumulative: ``geom`` nothing more,
    ``static`` sensor noise / vignetting / response / auto exposure, ``wingeom`` motion blur and rolling shutter. All
    keep the preset's timing (fps, latency, jitter, drops); depth effects are in ``calibrated`` only.
    """
    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.config import TYPICAL_UNCALIBRATED as T

    if name == "calibrated":
        return CameraConfig.preset("calibrated", W, H)
    keys = dict(geom=(), static=("vignette", "gamma", "black_level", "sigma_shot", "sigma_read", "sigma_prnu",
                                 "sigma_dsnu", "ae_tau_s", "ae_target"), wingeom=("exposure_s", "readout_s"))[name]
    timing = ("fps", "latency_s", "jitter_k", "jitter_theta", "drop_prob")
    return CameraConfig(W, H, dist=DIST, **{k: T[k] for k in (*timing, *keys)})


def write_xml():
    """Apollo ``scene.xml`` + head camera + boxes, written next to it (meshdir is file-relative). Returns the path."""
    spec = mujoco.MjSpec.from_file(str(APOLLO / "scene.xml"))
    spec.stat.extent = 5.0
    # the camera is placed in the world at the stand pose, then expressed in neck_pitch_link, whose frame is pitched
    # up ~40 deg there
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetDataKeyframe(m0, d0, 0)
    mujoco.mj_forward(m0, d0)
    b = m0.body("neck_pitch_link").id
    R_wb, x_wb = d0.xmat[b].reshape(3, 3), d0.xpos[b]
    p = math.radians(CAM_PITCH_DEG)
    fwd, up = np.array([math.cos(p), 0, -math.sin(p)]), np.array([math.sin(p), 0, math.cos(p)])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, (R_wb.T @ np.column_stack([np.cross(fwd, up), up, -fwd])).ravel())
    spec.body("neck_pitch_link").add_camera(name=CAM, pos=list(R_wb.T @ (np.array(CAM_WORLD) - x_wb)), quat=list(q),
                                            fovy=42.0)
    for g in spec.geoms:  # collision geoms (group 3): MuJoCo hides the group by default, Genesis would draw them
        if g.group == 3:
            g.rgba = [0, 0, 0, 0]
    for name, pos, size, rgba in BOXES:
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(size),
                                rgba=list(rgba), contype=0, conaffinity=0)
    path = APOLLO / "scene_headcam.xml"
    path.write_text(spec.to_xml())
    return path


def gate(t, a, b, ramp=0.6):
    """0 outside [a, b], 1 inside, sin^2 ramps of ``ramp`` s at both ends."""
    return math.sin(0.5 * math.pi * min(1.0, max(0.0, (t - a) / ramp), max(0.0, (b - t) / ramp))) ** 2


class Motion:
    """Kinematic Apollo pose at time t: the ``stand`` keyframe plus neck scan / fast turns, torso sway, arm wave."""

    def __init__(self, m):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self.q0 = d.qpos.copy()
        self.adr = {m.joint(j).name: m.joint(j).qposadr[0] for j in range(m.njnt)}

    def __call__(self, t):
        q = self.q0.copy()
        fast = sum(gate(t, a, b) for a, b in ((8, 12), (18, 22), (28, 33)))
        wave = gate(t, 13, 18.5)
        q[self.adr["neck_yaw"]] += 0.45 * math.sin(2 * math.pi * t / 10) + 0.35 * fast * math.sin(2 * math.pi * t / 2.5)
        q[self.adr["neck_pitch"]] += 0.08 * math.sin(2 * math.pi * t / 7)
        q[self.adr["torso_yaw"]] += 0.12 * math.sin(2 * math.pi * t / 14)
        q[self.adr["r_shoulder_fe"]] += -1.1 * wave
        q[self.adr["r_shoulder_aa"]] += -0.25 * wave
        q[self.adr["r_elbow_fe"]] += wave * (-0.9 - 0.45 * math.sin(2 * math.pi * 1.1 * t))
        return q


def label(img, text, sub=""):
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def preview(path, times=(0.0, 9.0, 15.5, 30.0)):
    """MuJoCo stills of the head camera (left) and an external view (right) at ``times``."""
    m = mujoco.MjModel.from_xml_path(str(write_xml()))
    d, motion, r = mujoco.MjData(m), Motion(m), mujoco.Renderer(m, H, W)
    ext = mujoco.MjvCamera()
    ext.lookat[:], ext.distance, ext.azimuth, ext.elevation = EXT["lookat"], 3.9, 140, -22
    tiles = []
    for t in times:
        d.qpos[:] = motion(t)
        mujoco.mj_forward(m, d)
        r.update_scene(d, CAM)
        head = r.render().copy()
        r.update_scene(d, ext)
        tiles.append(np.hstack([head, r.render()]))
    r.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))


def capture(sim, cfg, seconds, video=None):
    """Replay ``seconds`` of motion in ``sim``; returns the per-capture rows (first delivery of each frame)."""
    from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera

    assert video is None or sim == "genesis", "video needs the Genesis mirror scene"
    xml = write_xml()
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.opt.timestep = DT
    motion = Motion(m)
    if sim == "mujoco":
        real, ideal, sc = SimCamera(m, CAM, cfg), SimCamera(m, CAM, CameraConfig(W, H)), mujoco.MjData(m)
        d = mujoco.MjData(m)

        def render(data, n):
            rgb, z = real.render(data, depth=True)
            sc.qpos[:] = motion(max(0.0, data.time - real.half_window))  # ideal at capture_t (clamped at 0, as Genesis)
            mujoco.mj_forward(m, sc)
            return ideal.render(sc).copy(), rgb, z

        cam = real
    else:
        import genesis as gs

        from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        vis = gs.options.VisOptions(  # overhead light + a camera-following headlight; flat background
            lights=[gs.options.vis.DirectionalLight(dir=(-0.3, 0.2, -1), color=(1, 1, 1), intensity=6.0)],
            ambient_light=(0.3, 0.3, 0.3), background_color=(0.12, 0.198, 0.258))
        cam = GenesisCamera(xml, CAM, cfg, DT, vis_options=vis, ideal=True, headlight=2.0,
                            extra_cameras=[EXT] if video else [])

        def render(data, n):
            rgb, z = cam.render(data, depth=True)  # leaves the mirror at capture_t: the ideal matches it
            return cam.render_ideal(), rgb, z

    stream = CameraStream(cam, cfg, render=render)
    rows, vw, nxt, held, tc = collections.defaultdict(list), None, 0.0, None, None
    for i in range(int(round(seconds / DT)) + 1):
        t = i * DT
        q = motion(t)
        if sim == "mujoco":
            d.time, d.qpos[:] = t, q
            mujoco.mj_forward(m, d)
            data = d
        else:
            data = types.SimpleNamespace(time=t, qpos=q)
        frame, ctime, n = stream.step(data)
        if frame is not None and not (rows["n"] and rows["n"][-1] == n):
            for k, v in zip(("age", "n", "ideal", "real", "depth"), (t - ctime, n, *frame)):
                rows[k].append(v)
        if frame is not None:
            held, tc = frame[1], ctime
        if video and t + 1e-9 >= nxt:
            nxt += 1 / FPS
            cam.mirror(q)
            now = cam.render_ideal()
            hold = np.zeros_like(now) if held is None else held.copy()
            ext = np.ascontiguousarray(cam.extra[0].render(rgb=True, force_render=True)[0])
            img = np.vstack([label(cv2.resize(ext, (1280, 720)), "Genesis - Apptronik Apollo"),
                             np.hstack([label(cv2.resize(now, (640, 480)), "head cam: ideal (now)", f"t = {t:5.2f} s"),
                                        label(cv2.resize(hold, (640, 480)), "physicalai-sim2real-camera",
                                              "" if tc is None else f"age {1e3 * (t - tc):.0f} ms")])])
            if vw is None:
                import imageio_ffmpeg

                Path(video).parent.mkdir(parents=True, exist_ok=True)
                vw = imageio_ffmpeg.write_frames(str(video), img.shape[1::-1], fps=FPS, codec="libx264",
                                                 pix_fmt_out="yuv420p", quality=8)
                vw.send(None)
            vw.send(np.ascontiguousarray(img))
    if vw is not None:
        vw.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", choices=("genesis", "mujoco"), default="genesis")
    ap.add_argument("--cfg", choices=CFGS, default="calibrated")
    ap.add_argument("--seconds", type=float, default=10.0, help="34 covers the whole motion (~3 GB of frames)")
    ap.add_argument("--video", help="Genesis: write a 3-panel mp4 here")
    ap.add_argument("--preview", help="MuJoCo: write head-camera / external-view stills here and exit")
    a = ap.parse_args()
    if not APOLLO.is_dir():
        ap.error(f"{APOLLO} missing: bash scripts/fetch_menagerie.sh")
    if a.preview:
        preview(a.preview)
        print(a.preview)
        return
    if a.video and a.sim != "genesis":
        ap.error("--video needs --sim genesis")
    rows = capture(a.sim, make_cfg(a.cfg), a.seconds, a.video)
    if not rows["age"]:
        ap.error(f"no frame was delivered in {a.seconds} s: run longer")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cam_{a.sim}_{a.cfg}.npz"
    np.savez_compressed(path, **{k: np.array(v) if k in ("age", "n") else np.stack(v) for k, v in rows.items()})
    age = np.array(rows["age"]) * 1e3
    print(f"[apollo {a.sim} {a.cfg}] {len(age)} frames in {a.seconds:.1f} s; capture gaps>1: "
          f"{int((np.diff(rows['n']) > 1).sum())}; age ms mean {age.mean():.1f} p99 {np.percentile(age, 99):.1f} "
          f"-> {path}")


if __name__ == "__main__":
    main()
