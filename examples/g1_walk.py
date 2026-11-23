"""Unitree G1 head-camera characterization demo (ideal vs realistic D435i-like camera).

IMPORTANT: the base motion is PRESCRIBED, not emergent locomotion. The pelvis is
welded to a mocap body whose pose is driven analytically (forward speed, vertical
bob, roll/pitch/yaw sway); the joints track scripted gait-like sinusoids around the
``knees_bent`` pose with the model's position servos (kp=500, dampratio=1 in
menagerie's g1.xml; ctrl = q_ref is the PD law). The feet do not contact the floor
(a prescribed base would drag the stance foot and fight the weld). The goal is a
realistic head-camera trajectory for effect characterization, not a controller.

All camera preset values are VIRTUAL placeholders; real values come from Phase 3
calibration (scripts/calibrate.py).

Asset: ``bash scripts/fetch_menagerie.sh`` (-> third_party/mujoco_menagerie).
Run:   MUJOCO_GL=egl .venv/bin/python examples/g1_walk.py --seconds 5 --preset d435i_like --fast --out out/g1
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede `import mujoco`

import argparse
import csv
import math
import time
from collections import deque

import cv2
import mujoco
import numpy as np

from physicalai_sim2real_camera import CameraConfig, CameraStream, SimCamera

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENAGERIE = os.path.join(ROOT, "third_party", "mujoco_menagerie")
W, H = 640, 480
# knees_bent key from menagerie unitree_g1/scene_mjx.xml (g1.xml itself only ships "stand")
KNEES_BENT = np.array([0, 0, 0.755, 1, 0, 0, 0,
                       -0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
                       0, 0, 0.073,
                       0.2, 0.22, 0, 1, 0, 0, 0, 0.2, -0.22, 0, 1, 0, 0, 0])
CAM_POS = (0.09, 0.0, 0.43)  # on torso_link, just in front of the head mesh (x_max 0.078)
CAM_PITCH_DEG = 20.0  # downward

# VIRTUAL values (not measured). fovy 42 (D435 RGB vertical FOV) is set on the model camera.
PRESETS = {
    "ideal": dict(width=W, height=H),
    "d435i_like": dict(width=W, height=H, dist=(-0.05, 0.01, 0.0005, -0.0005, 0.0), vignette=0.4,
                       gamma=2.2, exposure_ev=0.0, sigma_shot=0.02, sigma_read=0.004, sigma_prnu=0.01,
                       sigma_dsnu=0.003, black_level=0.01, fps=30, latency_s=0.06, jitter_k=2,
                       jitter_theta=0.003, drop_prob=0.02, exposure_s=0.008, readout_s=0.020,
                       blur_samples=4, rs_bands=8, ae_tau_s=0.5, ae_target=0.18, bayer=False),
}


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat2quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def build_model():
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "unitree_g1", "g1.xml"))
    spec.option.timestep = 0.002
    spec.visual.headlight.active = 0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.stat.extent = 5.0  # zfar = 50*extent

    tex = mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    spec.add_texture(name="floor_tex", type=tex[0], builtin=tex[1], rgb1=[0.55, 0.55, 0.52],
                     rgb2=[0.3, 0.32, 0.35], width=256, height=256)
    spec.add_texture(name="tag_tex", type=tex[0], builtin=tex[1], rgb1=[1, 1, 1], rgb2=[0.05, 0.05, 0.05],
                     width=64, height=64)
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    fm = spec.add_material(name="floor_mat", texuniform=True, texrepeat=[1, 1])  # 1 m period = 0.5 m squares
    fm.textures[rgb_role] = "floor_tex"
    wb = spec.worldbody
    # contype 0: no foot/floor contact (base is prescribed)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.05], material="floor_mat",
                contype=0, conaffinity=0)
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 5], dir=[0.4, 0.3, -1],
                 diffuse=[0.45, 0.45, 0.45], ambient=[0.25, 0.25, 0.25], specular=[0.1, 0.1, 0.1])

    B, C = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    objects = [  # (type, pos, size, rgba, texrepeat)
        (B, [1.8, 0.9, 0.3], [0.3, 0.3, 0.3], [0.9, 0.3, 0.25, 1], 4),
        (C, [2.4, -0.9, 0.5], [0.2, 0.5, 0], [0.25, 0.6, 0.9, 1], 3),
        (B, [3.0, 1.0, 0.45], [0.2, 0.4, 0.45], [0.3, 0.8, 0.35, 1], 5),
        (C, [4.2, -1.0, 0.35], [0.3, 0.35, 0], [0.9, 0.75, 0.2, 1], 3),
        (B, [5.5, 0.9, 0.6], [0.25, 0.25, 0.6], [0.7, 0.35, 0.8, 1], 4),
        (B, [9.0, 0.0, 1.2], [0.05, 2.5, 1.2], [0.8, 0.8, 0.8, 1], 3),  # textured wall at the end of the path
    ]
    for i, (typ, pos, size, rgba, rep) in enumerate(objects):
        m = spec.add_material(name=f"obj{i}", rgba=rgba, texrepeat=[rep, rep])
        m.textures[rgb_role] = "tag_tex"
        wb.add_geom(name=f"obj{i}", type=typ, pos=pos, size=size, material=f"obj{i}")

    wb.add_body(name="base_target", mocap=True, pos=KNEES_BENT[:3], quat=KNEES_BENT[3:7])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="base_target", name2="pelvis",
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      # anchor 0, relpose identity (pelvis frame == mocap frame), torquescale 1.
                      # Explicit: MjSpec's default data is [0,1,0,...] (anchor 1 m off in y).
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

    # head camera: MuJoCo cameras look along -z with +y up; columns = (right, up, back)
    p = math.radians(CAM_PITCH_DEG)
    fwd, up = np.array([math.cos(p), 0, -math.sin(p)]), np.array([math.sin(p), 0, math.cos(p)])
    Rc = np.column_stack([np.cross(fwd, up), up, -fwd])
    torso = spec.body("torso_link")
    torso.add_camera(name="head_cam", pos=list(CAM_POS), quat=mat2quat(Rc), fovy=42.0)
    # D435i mesh: its lenses face body +z, so device frame = camera frame rotated 180° about x,
    # origin 5 mm behind the camera so the camera sits in front of the housing.
    d435 = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "realsense_d435i", "d435i.xml"))
    frame = torso.add_frame(pos=list(np.array(CAM_POS) - 0.005 * fwd),
                            quat=mat2quat(Rc @ np.diag([1.0, -1.0, -1.0])))
    frame.attach_body(d435.body("d435i"), "d435i_", "")
    model = spec.compile()
    for g in range(model.ngeom):  # the housing's collision capsule would touch the head/arms
        if model.geom_bodyid[g] == model.body("d435i_d435i").id:
            model.geom_contype[g] = model.geom_conaffinity[g] = 0
    return model


class Motion:
    """Prescribed base pose + joint references. ``osc`` scales every oscillation (angles and gait)."""

    def __init__(self, model, v_fwd=0.8, a_z=0.02, f=1.8, osc=1.0):
        self.v, self.az, self.f, self.osc = v_fwd, a_z, f, osc
        self.q0 = KNEES_BENT[7:].copy()
        j = lambda n: model.joint(n).qposadr[0] - 7  # noqa: E731
        self.i = {n: j(n) for n in (
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint",
            "right_elbow_joint", "waist_yaw_joint")}
        lo, hi = model.jnt_range[1:].T
        self.lo, self.hi = lo, hi

    def base(self, t):
        w, d, o = 2 * math.pi * self.f, math.radians, self.osc
        pos = KNEES_BENT[:3] + [self.v * t, 0.0, self.az * math.sin(w * t)]
        R = rot(o * d(2) * math.sin(w * t / 2), o * d(2) * math.sin(w * t), o * d(3) * math.sin(w * t / 2))
        return pos, mat2quat(R)

    def qref(self, t):
        s = math.sin(math.pi * self.f * t)  # stride phase: one stride = two bob periods
        a, i, q = self.osc, self.i, self.q0.copy()
        swing_l, swing_r = max(0.0, s), max(0.0, -s)
        q[i["left_hip_pitch_joint"]] -= a * 0.30 * s  # negative hip pitch = leg forward
        q[i["right_hip_pitch_joint"]] += a * 0.30 * s
        q[i["left_knee_joint"]] += a * 0.45 * swing_l
        q[i["right_knee_joint"]] += a * 0.45 * swing_r
        q[i["left_ankle_pitch_joint"]] -= a * 0.2 * swing_l
        q[i["right_ankle_pitch_joint"]] -= a * 0.2 * swing_r
        q[i["left_shoulder_pitch_joint"]] += a * 0.35 * s  # arms counter-swing
        q[i["right_shoulder_pitch_joint"]] -= a * 0.35 * s
        q[i["left_elbow_joint"]] += a * 0.15 * swing_r
        q[i["right_elbow_joint"]] += a * 0.15 * swing_l
        q[i["waist_yaw_joint"]] += a * 0.05 * s
        return np.clip(q, self.lo, self.hi)

    def apply(self, data, t):
        data.mocap_pos[0], data.mocap_quat[0] = self.base(t)
        data.ctrl[:] = self.qref(t)  # position servos: ctrl = q_ref


class Tap:
    """Wraps the realistic SimCamera for CameraStream.

    Per capture also renders the ideal frame + depth at capture_t and computes
    camera speed over the exposure window.
    """

    def __init__(self, model, real, ideal, hist, ctrl_dt, keep_n):
        self.model, self.real, self.ideal, self.hist = model, real, ideal, hist
        self.cam, self.push, self.half = real.cam, real.push, real.half_window
        self.cfg = real.cfg
        self.sc = mujoco.MjData(model)
        self.w = max(self.cfg.exposure_s, 2 * ctrl_dt)  # finite-difference window
        self.keep_n, self.kept, self.rows = list(keep_n), [], []

    def load(self, t):
        ts = np.array([h[0] for h in self.hist])
        _, qpos, qvel, mp, mq = self.hist[int(np.argmin(abs(ts - t)))]
        self.sc.qpos[:], self.sc.qvel[:], self.sc.mocap_pos[:], self.sc.mocap_quat[:] = qpos, qvel, mp, mq
        mujoco.mj_forward(self.model, self.sc)
        return self.sc.cam_xpos[self.cam].copy(), self.sc.cam_xmat[self.cam].reshape(3, 3).copy()

    def render(self, data):
        ev = float(getattr(self.real, "_ev", 0.0))  # AE ev used for THIS frame
        img = self.real.render(data)
        tc = data.time - self.half
        n = math.floor((tc + 1e-9 - self.cfg.phase_s) * self.cfg.fps)
        t0, t1 = max(tc - self.w / 2, self.hist[0][0]), min(tc + self.w / 2, data.time)
        p0, R0 = self.load(t0)
        p1, R1 = self.load(t1)
        dt = max(t1 - t0, 1e-9)  # first frame of an episode: no history yet -> speed 0
        v = (p1 - p0) / dt
        omega = np.linalg.norm(cv2.Rodrigues(R0.T @ R1)[0]) / dt
        self.load(tc)
        rgb_i, depth = self.ideal.render(self.sc, depth=True)
        axis = -R1[:, 2]
        v_perp = np.linalg.norm(v - (v @ axis) * axis)
        Z = float(depth[H // 2, W // 2])
        rate = self.real.f_px * (v_perp / Z + omega)  # px/s
        g_i = cv2.cvtColor(rgb_i, cv2.COLOR_RGB2GRAY)
        g_r = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap_i = cv2.Laplacian(g_i, cv2.CV_64F).var()
        lap_r = cv2.Laplacian(g_r, cv2.CV_64F).var()
        self.rows.append(dict(n=n, capture_t=tc, speed_v=float(np.linalg.norm(v)), v_perp=v_perp,
                              speed_w=omega, Z=Z, blur_px=rate * self.cfg.exposure_s,
                              rs_px=rate * self.cfg.readout_s, lap_ideal=lap_i, lap_real=lap_r,
                              lap_ratio=lap_r / lap_i, mean_ideal=float(g_i.mean()),
                              mean_real=float(g_r.mean()), ae_ev=ev))
        if self.keep_n and n >= self.keep_n[0]:
            while self.keep_n and n >= self.keep_n[0]:
                self.keep_n.pop(0)
            self.kept.append((n, rgb_i.copy(), img.copy()))
        return img


def run(seconds=5.0, ctrl_hz=500.0, preset="d435i_like", out="out/g1", fast=False, bayer=False,
        v_fwd=0.8, a_z=0.02, osc=1.0, settle_s=0.5):
    os.makedirs(out, exist_ok=True)
    model = build_model()
    data = mujoco.MjData(model)
    kw = dict(PRESETS[preset])
    if fast and kw.get("exposure_s"):
        kw.update(rs_bands=2, blur_samples=2)
    if bayer:
        kw["bayer"] = True
    cfg = CameraConfig(**kw)
    motion = Motion(model, v_fwd=v_fwd, a_z=a_z, osc=osc)
    dt = model.opt.timestep
    n_sub = max(1, round(1.0 / (ctrl_hz * dt)))
    ctrl_dt = n_sub * dt

    # settle at the t=0 references (servo sag under gravity), then restart the clock
    data.qpos[:] = KNEES_BENT
    motion.apply(data, 0.0)
    for _ in range(int(settle_s / dt)):
        mujoco.mj_step(model, data)
    data.time = 0.0
    mujoco.mj_forward(model, data)

    real = SimCamera(model, "head_cam", cfg)
    ideal = SimCamera(model, "head_cam", CameraConfig(W, H))
    hist = deque(maxlen=max(8, math.ceil(2 * (real.half_window + cfg.exposure_s) / ctrl_dt) + 8))
    n_total = int(seconds * cfg.fps)
    keep = np.linspace(0, max(0, n_total - 4), 6).round().astype(int)
    tap = Tap(model, real, ideal, hist, ctrl_dt, keep)
    stream = CameraStream(tap, cfg)
    pelvis = model.body("pelvis").id

    steps, seen = [], set()
    t_start = time.perf_counter()
    for k in range(int(round(seconds / ctrl_dt))):
        motion.apply(data, data.time)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)  # live xpos/cam poses consistent with qpos at data.time
        hist.append((data.time, data.qpos.copy(), data.qvel.copy(), data.mocap_pos.copy(),
                     data.mocap_quat.copy()))
        _, capture_t, n = stream.step(data)
        age = None if capture_t is None else data.time - capture_t
        if n is not None:
            seen.add(n)
        steps.append((k, data.time, n, capture_t, age, data.xpos[pelvis, 2], data.mocap_pos[0, 2]))
    wall = time.perf_counter() - t_start
    real.close()
    ideal.close()

    for r in tap.rows:
        r["delivered"] = int(r["n"] in seen)
    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tap.rows[0]))
        w.writeheader()
        w.writerows(tap.rows)
    with open(os.path.join(out, "per_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "n", "capture_t", "age", "pelvis_z", "target_z"])
        w.writerows(["" if v is None else v for v in s] for s in steps)

    tiles = []
    for n, gi, gr in tap.kept:
        cv2.imwrite(os.path.join(out, f"frames_ideal_{n:04d}.png"), cv2.cvtColor(gi, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, f"frames_real_{n:04d}.png"), cv2.cvtColor(gr, cv2.COLOR_RGB2BGR))
        tiles.append(cv2.resize(np.hstack([gi, gr]), (W, H // 2), interpolation=cv2.INTER_AREA))
    if tiles:
        cv2.imwrite(os.path.join(out, "strip.png"), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))

    ages = np.array([s[4] for s in steps if s[4] is not None]) * 1e3
    col = lambda key: np.array([r[key] for r in tap.rows])  # noqa: E731
    captured = len(tap.rows)
    summary = dict(
        preset=preset + (" (fast)" if fast else "") + (" +bayer" if bayer else ""),
        seconds=seconds, ctrl_hz=1 / ctrl_dt,
        frames_captured=captured, frames_delivered=len(seen),
        frames_dropped=math.floor((data.time + 1e-9 - real.half_window - cfg.phase_s) * cfg.fps)
        + 1 - captured,
        age_mean_ms=ages.mean(), age_p95_ms=np.percentile(ages, 95),
        blur_mean_px=col("blur_px").mean(), blur_max_px=col("blur_px").max(),
        rs_skew_mean_px=col("rs_px").mean(), rs_skew_max_px=col("rs_px").max(),
        lap_ratio_mean=col("lap_ratio").mean(),
        pelvis_z_err_max_m=max(abs(s[5] - KNEES_BENT[2]) for s in steps),
        run_s=wall, ms_per_frame=wall / max(1, captured) * 1e3,
    )
    with open(os.path.join(out, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.4g}\n" if isinstance(v, float) else f"{k}: {v}\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--ctrl-hz", type=float, default=500.0)
    ap.add_argument("--preset", choices=list(PRESETS), default="d435i_like")
    ap.add_argument("--out", default="out/g1")
    ap.add_argument("--fast", action="store_true", help="rs_bands 2, blur_samples 2")
    ap.add_argument("--bayer", action="store_true")
    ap.add_argument("--v-fwd", type=float, default=0.8)
    ap.add_argument("--a-z", type=float, default=0.02)
    ap.add_argument("--osc", type=float, default=1.0, help="scale of all angle/gait oscillations")
    a = ap.parse_args()
    s = run(a.seconds, a.ctrl_hz, a.preset, a.out, a.fast, a.bayer, a.v_fwd, a.a_z, a.osc)
    print(open(os.path.join(a.out, "summary.txt")).read(), end="")
    return s


if __name__ == "__main__":
    main()
