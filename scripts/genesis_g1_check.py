"""Floating-base check: Unitree G1 head camera, Genesis (GenesisCamera) vs MuJoCo (SimCamera), window mode.

The pelvis free joint follows a kinematic trajectory (no physics: both simulators get the same qpos every 2 ms):
forward 0.4 m/s, yaw 2.5 rad/s, small pitch / roll bob, swinging arms. Coloured boxes ring the robot. Each capture
renders the exposure-window sub-frames from the qpos ring, where the pelvis quaternion is interpolated (slerp in
``interp_qpos``, ``mj_integratePos`` in ``StateRing``).

Usage (one simulator per process; Genesis from its venv, see docs/genesis.md):
  python scripts/genesis_g1_check.py render mujoco|genesis|madrona   -> out/genesis_g1/<sim>.npz (madrona: Genesis'
                                                                         BatchRenderer)
  python scripts/genesis_g1_check.py report                  -> silhouette IoU, centroid vs q(capture_t)
  python scripts/genesis_g1_check.py bench [madrona] [compile] [n ...] -> GenesisCamera ms per capture (Genesis)
"""

import math
import os
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))

from g1_walk import CAM_PITCH_DEG, CAM_POS, KNEES_BENT, H, W, mat2quat  # noqa: E402
from genesis_allex_check import iou, mask  # noqa: E402

G1 = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
OUT = ROOT / "out" / "genesis_g1"
DT, T_END, FPS = 0.002, 2.0, 30.0
WINDOW = dict(exposure_s=0.008, readout_s=0.020, blur_samples=4, rs_bands=8)
DIST = (-0.05, 0.01, 0.0005, -0.0005, 0.0)
HUES = ("1 0 0", "1 1 0", "0 1 0", "0 1 1", "0 0 1")
# Madrona has no ambient light: six shadowless directional lights along +-x, +-y, +-z light every face about evenly
# (Madrona wants a position even for a directional light: 10 m up-light of the origin)
FLAT_MADRONA = [dict(pos=tuple(-10.0 * float(v) * s for v in axis), dir=tuple(float(v) * s for v in axis),
                     directional=True, castshadow=False, intensity=0.5) for axis in np.eye(3) for s in (1, -1)]


def write_xml():
    """G1 + head camera (g1_walk's mount) + 10 saturated boxes on a 2.5 m ring; free pelvis, no mocap weld."""
    spec = mujoco.MjSpec.from_file(str(G1 / "g1.xml"))
    spec.stat.extent = 5.0
    # flat lighting (ambient only) in both simulators: this check is about geometry and interpolation, and the two
    # renderers shade faces differently (a dark face drops out of one hue mask and not the other)
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = [1.0] * 3, [0.0] * 3, [0.0] * 3
    p = math.radians(CAM_PITCH_DEG)
    fwd, up = np.array([math.cos(p), 0, -math.sin(p)]), np.array([math.sin(p), 0, math.cos(p)])
    spec.body("torso_link").add_camera(name="head_cam", pos=list(CAM_POS), fovy=42.0,
                                       quat=mat2quat(np.column_stack([np.cross(fwd, up), up, -fwd])))
    for k in range(10):
        a = 2 * math.pi * k / 10
        spec.worldbody.add_geom(name=f"t{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.18, 0.18, 0.18],
                                pos=[2.5 * math.cos(a), 2.5 * math.sin(a), 0.5 + 0.15 * (k % 3)],
                                rgba=[float(x) for x in HUES[k % 5].split()] + [1], contype=0, conaffinity=0)
    path = G1 / "g1_headcam.xml"  # next to g1.xml: meshdir is relative to the model file
    path.write_text(spec.to_xml())
    return path


def quat_ypr(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(roll / 2), math.sin(roll / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def qpos_at(m, t):
    """Kinematic G1 pose: pelvis forward 0.4 m/s, yaw 2.5 rad/s, 1.8 Hz pitch/roll bob; shoulders swing."""
    q = KNEES_BENT.copy()
    q[:3] = [0.4 * t, 0.0, KNEES_BENT[2] + 0.02 * math.sin(2 * math.pi * 1.8 * t)]
    q[3:7] = quat_ypr(2.5 * t, 0.05 * math.sin(2 * math.pi * 1.8 * t), 0.04 * math.sin(2 * math.pi * 0.9 * t))
    for side, s in (("left", 1), ("right", -1)):
        q[m.joint(f"{side}_shoulder_pitch_joint").qposadr[0]] += 0.5 * s * math.sin(2 * math.pi * 0.9 * t)
    return q


def cfg(window=True):
    from physicalai_sim2real_camera import CameraConfig

    return CameraConfig(width=W, height=H, dist=DIST, latency_s=0.05, **(WINDOW if window else {}))


def render(sim):
    xml = write_xml()
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.opt.timestep = DT
    m.light_diffuse[:] = m.light_specular[:] = m.light_ambient[:] = 0  # g1.xml's sun: headlight ambient only
    steps = int(round(T_END / DT))
    capture = {int(round((k / FPS) / DT)) for k in range(3, int(T_END * FPS) + 1)}  # skip the first 0.1 s
    win, gs_tc, gs_end, times = [], [], [], []
    if sim == "mujoco":
        from physicalai_sim2real_camera import SimCamera

        d = mujoco.MjData(m)
        cw, cg = SimCamera(m, "head_cam", cfg(), device="cuda"), SimCamera(m, "head_cam", cfg(False), device="cuda")
        sc = mujoco.MjData(m)
        for i in range(steps + 1):
            d.time, d.qpos[:] = i * DT, qpos_at(m, i * DT)
            mujoco.mj_forward(m, d)
            cw.push(d)
            if i in capture:
                win.append(cw.render(d))
                for t, out in ((d.time - cw.half_window, gs_tc), (d.time, gs_end)):
                    sc.time, sc.qpos[:] = t, qpos_at(m, t)
                    mujoco.mj_forward(m, sc)
                    out.append(cg.render(sc))
                times.append(d.time)
        layout = None
    else:
        import genesis as gs

        from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        # ambient only; a zero-intensity light rather than lights=[]: Genesis' rasterizer corrupts the heap with an
        # empty light list on some scenes ("malloc(): corrupted top size", reproduced on a free-joint rig)
        flat = gs.options.VisOptions(
            lights=[gs.options.vis.DirectionalLight(dir=(0, 0, -1), color=(1, 1, 1), intensity=0.0)],
            ambient_light=(1.0, 1.0, 1.0))
        kw = (dict(renderer="madrona", madrona_lights=FLAT_MADRONA) if sim == "madrona" else dict(vis_options=flat))
        cw = GenesisCamera(xml, "head_cam", cfg(), DT, **kw)
        cg = GenesisCamera(xml, "head_cam", cfg(False), DT, **kw)
        layout = qpos_layout_error(m, cw)
        for i in range(steps + 1):
            t = i * DT
            data = types.SimpleNamespace(time=t, qpos=qpos_at(m, t))
            cw.push(data)
            if i in capture:
                win.append(cw.render(data))
                for tt, out in ((t - cw.half, gs_tc), (t, gs_end)):
                    out.append(cg.render(types.SimpleNamespace(time=tt, qpos=qpos_at(m, tt))))
                times.append(t)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{sim}.npz", win=np.stack(win), gs_tc=np.stack(gs_tc), gs_end=np.stack(gs_end),
                        t=np.array(times), layout=np.array(-1.0 if layout is None else layout))
    print(f"[{sim}] {len(win)} captures" + ("" if layout is None else f"; qpos layout: max link pose error "
                                                                      f"{layout:.2e} m"))


def qpos_layout_error(m, cam):
    """Same qpos into MuJoCo and the Genesis mirror entity -> max link-position difference (layout check)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos_at(m, 0.37)
    mujoco.mj_forward(m, d)
    cam.mirror(d.qpos.copy())
    err = 0.0
    for link in cam.entity.links:
        pos = link.get_pos().cpu().numpy().reshape(-1)
        err = max(err, float(np.abs(pos - d.xpos[m.body(link.name).id]).max()))
    return err


def report(other="genesis"):
    a, b = (np.load(OUT / f"{s}.npz") for s in ("mujoco", other))
    print(f"--- MuJoCo vs {other}")
    ious = np.array([iou(mask(x), mask(y)) for x, y in zip(a["win"], b["win"])])
    print(f"Genesis qpos layout vs MuJoCo: max link position error {float(b['layout']):.2e} m")
    print(f"window silhouettes Genesis vs MuJoCo over {np.isfinite(ious).sum()} captures with targets: IoU median "
          f"{np.nanmedian(ious):.3f}, p10 {np.nanpercentile(ious, 10):.3f}, min {np.nanmin(ious):.3f}")
    d = [np.linalg.norm(np.argwhere(mask(x)).mean(0) - np.argwhere(mask(y)).mean(0))
         for x, y in zip(a["win"], b["win"]) if mask(x).sum() > 500 and mask(y).sum() > 500]
    print(f"window centroid |Genesis - MuJoCo|: median {np.median(d):.2f} px, max {np.max(d):.2f} px")
    # reference only: with a rolling shutter and boxes crossing the frame edge the window centroid is not the
    # q(capture_t) one in either simulator; tests/test_genesis_bridge.py checks the time base (exposure only)
    for sim, z in (("mujoco", a), (other, b)):
        e_tc, e_end = [], []
        for w, g0, g1 in zip(z["win"], z["gs_tc"], z["gs_end"]):
            mw, m0, m1 = mask(w), mask(g0), mask(g1)
            if mw.sum() > 500 and m0.sum() > 500 and m1.sum() > 500:
                c = np.argwhere(mw).mean(0)
                e_tc.append(np.linalg.norm(c - np.argwhere(m0).mean(0)))
                e_end.append(np.linalg.norm(c - np.argwhere(m1).mean(0)))
        print(f"{sim}: window centroid vs global shutter at q(capture_t): median {np.median(e_tc):.2f} px, "
              f"vs q(end): median {np.median(e_end):.2f} px ({len(e_tc)} captures)")


def bench(ns, renderer="rasterizer", compile=False):
    """Window-mode capture time of a batched GenesisCamera on the G1 scene (13 sub-renders + depth per capture)."""
    import time

    import genesis as gs
    import torch

    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = write_xml()
    m = mujoco.MjModel.from_xml_path(str(xml))
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    for n in ns:
        cam = GenesisCamera(xml, "head_cam", cfg(), DT, n_envs=n, renderer=renderer, compile=compile,
                            madrona_lights=FLAT_MADRONA if renderer == "madrona" else ())
        q = lambda t: np.repeat(qpos_at(m, t)[None], n, 0) if n else qpos_at(m, t)  # noqa: E731
        for i in range(30):
            cam.push(types.SimpleNamespace(time=i * DT, qpos=q(i * DT)))
        data = types.SimpleNamespace(time=29 * DT, qpos=q(29 * DT))
        cam.render(data, depth=True)  # warm-up (kernel compile, first allocations, torch.compile)
        cam.render(data, depth=True)
        torch.cuda.synchronize()
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            cam.render(data, depth=True)
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0) / reps
        print(f"{renderer}{'+compile' if compile else ''} n_envs {n:3d}: {ms:7.1f} ms per capture, "
              f"{ms / max(n, 1):6.2f} ms per env "
              f"({cam.batch.n_sub} sub-renders + depth, {cam.batch.W_r}x{cam.batch.H_r})")


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    if sys.argv[1] == "render":
        render(sys.argv[2])
    elif sys.argv[1] == "bench":
        flags = [a for a in sys.argv[2:] if not a.isdigit()]  # [madrona] [compile] n ...
        bench([int(a) for a in sys.argv[2:] if a.isdigit()] or [0, 4, 16], "madrona" if "madrona" in flags else
              "rasterizer", "compile" in flags)
    else:
        report()
        if (OUT / "madrona.npz").exists():
            report("madrona")


if __name__ == "__main__":
    main()
