"""WIRobotics ALLEX head camera via physicalai-sim2real-camera in Genesis (and in MuJoCo for comparison).

The robot replays a recorded ALLEX motion (``demo1.npz``) under PD position control in either simulator; the head
RealSense colour camera of the MJCF goes through ``CameraStream`` + the image pipeline: ``GenesisCamera``
(``physicalai_sim2real_camera.genesis_bridge``) in Genesis, ``SimCamera`` in MuJoCo. Captures are saved for
``scripts/genesis_allex_check.py``; ``--video`` writes a demo mp4 and ``--live`` opens the Genesis viewer plus a
head-camera window (Genesis only).

Setup: ``scripts/fetch_allex.sh``; Genesis lives in its own venv (it pulls mujoco 4 and opencv-python):
``uv venv -p 3.12 .venv-genesis && uv pip install -p .venv-genesis torch --index-url
https://download.pytorch.org/whl/cu128 && uv pip install -p .venv-genesis genesis-world==1.4.5 mujoco==3.14.0 &&
uv pip install -p .venv-genesis --no-deps -e .``

Run: ``.venv-genesis/bin/python examples/genesis_allex.py --sim genesis --cfg wingeom --seconds 42.7``
(``--cfg calibrated``: the ``CameraConfig.preset("calibrated")`` setup, rectified geometry with every effect a
calibration cannot remove on).
"""

import argparse
import collections
import dataclasses
import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLEX = ROOT / "third_party" / "allex_model"
OUT = ROOT / "out" / "genesis_allex"
CAM = "realsense_color_optical_frame"
W, H = 640, 480
TIMING = dict(fps=30, latency_s=0.06, jitter_k=2, jitter_theta=0.003, drop_prob=0.02)
WINDOW = dict(exposure_s=0.008, readout_s=0.020, blur_samples=4, rs_bands=8)
DIST = (-0.05, 0.01, 0.0005, -0.0005, 0.0)
PHOTO = dict(vignette=0.4, gamma=2.2, sigma_shot=0.02, sigma_read=0.004, sigma_prnu=0.01, sigma_dsnu=0.003,
             black_level=0.01)
CFGS = {  # g1_walk.py "d435i_like" VIRTUAL values, split so each effect can be checked on its own
    "geom": dict(dist=DIST),  # distortion only: margin render + remap
    "wingeom": dict(dist=DIST, **WINDOW),  # + motion blur and rolling shutter
    "static": dict(dist=DIST, **PHOTO),  # + noise, vignetting, black level, gamma (no AE, no window)
    "real": dict(dist=DIST, **PHOTO, **WINDOW, ae_tau_s=0.5, ae_target=0.18),
}
BOXES = (("red", "1.0 0.25 0.9", 0.08, "1 0 0"), ("green", "1.3 -0.2 1.1", 0.1, "0 1 0"),
         ("blue", "1.6 0.0 0.8", 0.12, "0 0 1"), ("yellow", "0.7 -0.1 0.95", 0.04, "1 1 0"),
         ("cyan", "1.0 0.42 1.30", 0.04, "0 1 1"))  # saturated: lighting-independent hue silhouettes
TARGETS = "".join(f'\n    <geom name="t_{n}" type="box" pos="{p}" size="{s} {s} {s}" rgba="{c} 1" contype="0" '
                  f'conaffinity="0"/>' for n, p, s, c in BOXES) + "\n  </worldbody>"


def scene_variant(name, extra):
    """Write ``mjcf/<name>.xml`` = ALLEX ``scene.xml`` + ``extra`` worldbody content (meshdir is file-relative)."""
    src = (ALLEX / "mjcf" / "scene.xml").read_text()
    path = ALLEX / "mjcf" / f"{name}.xml"
    path.write_text(src.replace("</worldbody>", extra, 1))
    return path


def make_cfg(name):
    from physicalai_sim2real_camera import CameraConfig

    if name == "calibrated":  # the usual robot setup: calibrated + rectified geometry, every uncalibrated effect on
        return CameraConfig.preset("calibrated", W, H)
    return CameraConfig(width=W, height=H, **TIMING, **CFGS[name])


HEADLIGHT = 2.0  # GenesisCamera headlight intensity standing in for MuJoCo's (diffuse 0.6, follows the camera)


def scene_lights(gs, forward=True):
    """Genesis lights approximating ALLEX scene.xml as seen by the head camera.

    scene.xml: overhead directional light + MuJoCo headlight (diffuse 0.6, ambient 0.3; it follows the rendering
    camera) + gradient skybox, which becomes a flat background. Two fits of ideal brightness against MuJoCo:

    - ``forward=True``: a fixed +x directional light stands in for the headlight. Fitted on the initial head view
      only (+3 %); looking down at the floor (demo1 30-42 s) it is ~28 % too dark.
    - ``forward=False``, with ``GenesisCamera(headlight=HEADLIGHT)``: fitted on demo1 poses at 8 s (sky + boxes) and
      32 / 36 / 40 s (floor): within -4..+1 %. Brighter overhead light and darker background, because Genesis
      renders the floor darker and the flat background brighter than MuJoCo's skybox.
    """
    if forward:
        return gs.options.VisOptions(
            lights=[gs.options.vis.DirectionalLight(dir=(0, 0, -1), color=(1, 1, 1), intensity=3.0),
                    gs.options.vis.DirectionalLight(dir=(1, 0, 0), color=(1, 1, 1), intensity=1.0)],
            ambient_light=(0.25, 0.25, 0.25), background_color=(0.2, 0.33, 0.43))
    return gs.options.VisOptions(
        lights=[gs.options.vis.DirectionalLight(dir=(0, 0, -1), color=(1, 1, 1), intensity=8.0)],
        ambient_light=(0.3, 0.3, 0.3), background_color=(0.12, 0.198, 0.258))


def load_motion(path, xml):
    """Return ``(names, dt, q, qd, kp, kv)``; motions without per-frame gains get the MJCF's nominal gains."""
    import mujoco

    z = np.load(path, allow_pickle=True)
    names, q = [str(x) for x in z["joint_names"]], z["q"]
    if "kp" in z and "kv" in z:
        return names, float(z["dt"]), q, z["qd"], z["kp"], z["kv"]
    m = mujoco.MjModel.from_xml_path(str(xml))
    gains = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])):
             (m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]) for a in range(m.nu)}  # position: bias [0,-kp,-kv]
    kp, kv = (np.broadcast_to(np.array([gains[n][i] for n in names]), q.shape) for i in (0, 1))
    return names, float(z["dt"]), q, z["qd"], kp, kv


def warm_targets(home, q0, dt):
    """The ALLEX replay's 0.5 s ramp from the rest pose to the first frame (zero velocity target)."""
    n = max(1, int(round(0.5 / dt)))
    return [home * (1 - (k + 1) / n) + q0 * ((k + 1) / n) for k in range(n)]


def run_mujoco(xml, motion, cb):
    """ALLEX's published MuJoCo replay law: position + velocity servos, gravity comp fed forward, clip(PD + G, ±τ)."""
    import mujoco

    names, dt, q, qd, kp, kv = motion
    spec = importlib.util.spec_from_file_location("allex_replay", ALLEX / "examples" / "mujoco" / "replay.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    m = ref.build_model(xml)
    m.opt.timestep = dt
    d, sd = mujoco.MjData(m), mujoco.MjData(m)
    pos_act, vel_act = {}, {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        is_vel = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a).endswith("_Velocity")
        (vel_act if is_vel else pos_act)[jn] = (a, jid)
    pairs = [(names.index(jn), pa, vel_act[jn][0], int(m.jnt_dofadr[jid]), jid,
              float(m.jnt_actfrcrange[jid, 1]) if m.jnt_actfrclimited[jid] else None)
             for jn, (pa, jid) in pos_act.items() if jn in names]  # as replay.py: recorded joints only
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])

    def step(fr, frd, kpf, kvf):
        sd.qpos[:], sd.qvel[:] = d.qpos, 0.0
        mujoco.mj_forward(m, sd)
        d.qfrc_applied[:] = sd.qfrc_bias
        for c, pa, va, dof, jid, tau in pairs:
            m.actuator_gainprm[pa, 0], m.actuator_biasprm[pa, 1] = kpf[c], -kpf[c]
            m.actuator_gainprm[va, 0], m.actuator_biasprm[va, 2] = kvf[c], -kvf[c]
            d.ctrl[pa], d.ctrl[va] = fr[c], frd[c]
            if tau is not None:
                g = d.qfrc_applied[dof]
                m.jnt_actfrcrange[jid] = (-tau - g, tau - g)
        mujoco.mj_step(m, d)

    for t in warm_targets(d.qpos[qadr].copy(), q[0], dt):
        step(t, np.zeros(len(names)), kp[0], kv[0])
    traj = np.empty_like(q)
    for i in range(len(q)):
        step(q[i], qd[i], kp[i], kv[i])
        traj[i] = d.qpos[qadr]
        cb(m, d)
    return traj


def run_genesis(xml, motion, setup, cb, live=False, lights=None):
    """Same replay in Genesis: control_dofs_position_velocity with per-frame gains, force range ±τ, gravity comp 1.

    ponytail: Genesis gravity compensation sits outside the torque clip, so only torque-saturated steps differ
    from clip(PD + G, ±τ); feed G forward through control_dofs_force if saturation matters.
    """
    import genesis as gs
    import mujoco
    import torch

    names, dt, q, qd, kp, kv = motion
    mj = mujoco.MjModel.from_xml_path(str(xml))
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    kw = {"vis_options": scene_lights(gs, forward=lights == "fixed")} if lights else {}
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=live,
                     viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, 1.6, 1.0), camera_lookat=(0.15, 0, 0.35),
                                                             camera_fov=45, refresh_rate=60), **kw)
    robot = scene.add_entity(gs.morphs.MJCF(file=str(xml)), material=gs.materials.Rigid(gravity_compensation=1.0))
    ctx = setup(gs, robot, dt, kw.get("vis_options"))
    scene.build()
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
    lim = [mj.jnt_actfrcrange[mj.joint(n).id, 1] if mj.jnt_actfrclimited[mj.joint(n).id] else 1e9 for n in names]
    robot.set_dofs_force_range(-np.array(lim), np.array(lim), dofs)

    def T(a):
        return torch.as_tensor(a, dtype=gs.tc_float, device=gs.device)

    def step(fr, frd, kpf, kvf):
        robot.set_dofs_kp(T(kpf), dofs)
        robot.set_dofs_kv(T(kvf), dofs)
        robot.control_dofs_position_velocity(T(fr), T(frd), dofs)
        scene.step()

    warm = warm_targets(robot.get_dofs_position(dofs).cpu().numpy(), q[0], dt)
    for t in warm:
        step(t, np.zeros(len(names)), kp[0], kv[0])
    traj = torch.empty((len(q), len(names)), dtype=gs.tc_float, device=gs.device)
    for i in range(len(q)):
        step(q[i], qd[i], kp[i], kv[i])
        traj[i] = robot.get_dofs_position(dofs)
        cb(ctx, robot, (len(warm) + i + 1) * dt)
    return traj.cpu().numpy()


class Log:
    """First delivery of each capture: age, index, ideal (at capture_t), pipeline output, depth, AE ev."""

    def __init__(self):
        self.rows = collections.defaultdict(list)

    def add(self, t, frame, tc, n):
        if frame is None or (self.rows.get("n") and self.rows["n"][-1] == n):
            return
        for k, v in zip(("age", "n", "ideal", "real", "depth", "ev"), (t - tc, n, *frame)):
            self.rows[k].append(v)


def label(img, text, sub=""):
    import cv2

    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def capture_mujoco(xml, motion, cfg):
    import mujoco

    from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera

    log, st = Log(), {}

    def cb(m, d):
        if not st:
            st["ideal"], st["real"] = SimCamera(m, CAM, CameraConfig(W, H)), SimCamera(m, CAM, cfg, device="cuda")
            st["sc"] = mujoco.MjData(m)

            def render(data, n):
                rgb, z = st["real"].render(data, depth=True)
                sc, half = st["sc"], st["real"].half_window
                if half > 0:  # ideal at capture_t (mid-exposure), as the Genesis side renders it
                    st["real"]._ring.load(data.time - half, sc)
                else:
                    sc.qpos[:], sc.time = data.qpos, data.time
                    mujoco.mj_forward(m, sc)
                return st["ideal"].render(sc).copy(), rgb, z, float(st["real"]._ev)

            st["stream"] = CameraStream(st["real"], cfg, render=render)
        log.add(d.time, *st["stream"].step(d))

    return log, run_mujoco(xml, motion, cb)


def capture_genesis(xml, motion, cfg, lights=None, video=None, live=False):
    import cv2

    from physicalai_sim2real_camera import CameraStream
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    log, vw = Log(), []

    def setup(gs, robot, dt, vis):
        if cfg is None:  # --cfg none: physics only (reference trajectory for the no-disturbance check)
            return None
        extra = [dict(res=(960, 540), pos=(2.0, 1.6, 1.0), lookat=(0.15, 0.0, 0.35), fov=45)] if video else []
        cam = GenesisCamera(xml, CAM, cfg, dt, vis_options=vis, ideal=True, extra_cameras=extra,
                            headlight=HEADLIGHT if lights == "headlight" else None)

        def render(data, n):
            rgb, z = cam.render(data, depth=True)  # leaves the mirror at capture_t: the ideal matches it
            return cam.render_ideal(), rgb, z, float(cam.batch.ev[0])

        print(f"camera on link {cam.link!r}; margin {cam.batch.W_r}x{cam.batch.H_r} @ fovy {cam.batch.base.fovy_r:.2f}"
              f"; {cam.batch.n_sub} sub-render(s)/frame")
        return {"cam": cam, "stream": CameraStream(cam, cfg, render=render), "next": 0.0, "wall0": None}

    def panels(ctx, robot, t, frame, tc):
        cam = ctx["cam"]
        cam.mirror(robot.get_qpos())
        now = cam.render_ideal()
        held = np.zeros_like(now) if frame is None else frame[1].copy()
        age = "" if tc is None else f"age {1e3 * (t - tc):.0f} ms"
        return (label(now.copy(), "head cam: ideal (now)", f"t = {t:5.2f} s"),
                label(held, "physicalai-sim2real-camera", age))

    def cb(ctx, robot, t):
        if ctx is None:
            return
        data = types.SimpleNamespace(time=t, qpos=robot.get_qpos())
        frame, tc, n = ctx["stream"].step(data)
        log.add(t, frame, tc, n)
        if not (video or live) or t + 1e-9 < ctx["next"]:
            return
        ctx["next"] += 1 / 30
        now, held = panels(ctx, robot, t, frame, tc)
        if video:
            import imageio_ffmpeg

            ext = np.ascontiguousarray(ctx["cam"].extra[0].render(rgb=True, force_render=True)[0])
            img = np.vstack([label(ext, "Genesis - ALLEX demo1"),
                             np.hstack([cv2.resize(now, (480, 360)), cv2.resize(held, (480, 360))])])
            if not vw:
                vw.append(imageio_ffmpeg.write_frames(str(video), img.shape[1::-1], fps=30, codec="libx264",
                                                      pix_fmt_out="yuv420p", quality=7))
                vw[0].send(None)
            vw[0].send(np.ascontiguousarray(img))
        if live:  # paced to wall clock: sleep only the remainder
            cv2.imshow("ALLEX head camera", cv2.cvtColor(np.hstack([now, held]), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            ctx["wall0"] = ctx["wall0"] or time.perf_counter() - t
            time.sleep(max(0.0, ctx["wall0"] + t - time.perf_counter()))

    traj = run_genesis(xml, motion, setup, cb, live=live, lights=lights)
    if vw:
        vw[0].close()
    return log, traj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", choices=("genesis", "mujoco"), default="genesis")
    ap.add_argument("--cfg", choices=[*sorted(CFGS), "calibrated", "none"], default="real",
                    help="calibrated: CameraConfig.preset('calibrated'); none: Genesis physics only")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--motion", default=str(ALLEX / "examples" / "motions" / "demo1.npz"))
    ap.add_argument("--lights", action="store_true", help="Genesis: lights approximating scene.xml (scene_lights)")
    ap.add_argument("--headlight", action="store_true", help="Genesis: scene_lights with a camera-following headlight")
    ap.add_argument("--video", help="Genesis: write a demo mp4 here")
    ap.add_argument("--live", action="store_true", help="Genesis: viewer + head-camera window, real time")
    a = ap.parse_args()
    if a.cfg == "none" and a.sim != "genesis":
        ap.error("--cfg none is the Genesis no-camera reference")
    if (a.video or a.live or a.lights or a.headlight) and (a.sim != "genesis" or a.cfg == "none"):
        ap.error("--video / --live / --lights need --sim genesis and a camera cfg")
    os.environ.setdefault("MUJOCO_GL", "egl")
    xml = scene_variant("scene_targets", TARGETS)
    names, dt, q, qd, kp, kv = load_motion(a.motion, xml)
    T = int(round(a.seconds / dt))
    motion = (names, dt, q[:T], qd[:T], kp[:T], kv[:T])
    cfg = None if a.cfg == "none" else make_cfg(a.cfg)
    if a.sim == "mujoco":
        log, traj = capture_mujoco(xml, motion, cfg)
    else:
        lights = "headlight" if a.headlight else ("fixed" if a.lights else None)
        log, traj = capture_genesis(xml, motion, cfg, lights=lights, video=a.video, live=a.live)
    if cfg is not None and not log.rows.get("age"):
        ap.error(f"no frame was delivered in {a.seconds} s (latency {cfg.latency_s} s): run longer")
    r = {k: np.array(v) if k in ("age", "n", "ev") else np.stack(v) for k, v in log.rows.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.sim}_{a.cfg}{'_head' if a.headlight else '_lit' if a.lights else ''}"
    np.savez_compressed(OUT / f"cam_{tag}.npz", traj=traj, cfg=str(cfg and dataclasses.asdict(cfg)), **r)
    if cfg is None:
        print(f"[{tag}] {len(traj)} physics steps -> {OUT / f'cam_{tag}.npz'}")
        return
    age = r["age"]
    print(f"[{tag}] {len(r['n'])} frames in {a.seconds:.1f} s; capture gaps>1: {int((np.diff(r['n']) > 1).sum())}; "
          f"age ms mean {1e3 * age.mean():.1f} p99 {1e3 * np.percentile(age, 99):.1f} -> {OUT / f'cam_{tag}.npz'}")


if __name__ == "__main__":
    main()
