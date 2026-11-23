"""Genesis vs MuJoCo agreement of the ALLEX head camera (numbers in docs/genesis.md).

Subcommands (run the captures first with examples/genesis_allex.py, see docs/genesis.md):
  physics             joint-trajectory RMS, Genesis (no camera) vs MuJoCo, per joint group
  compare             window-mode silhouettes, blur amount, static window == global shutter, depth, and the
                      Genesis physics trajectory with vs without the camera
  lights              ideal brightness and AE gain per demo1 segment, MuJoCo vs Genesis (default, --lights,
                      --headlight)
  light-poses         Genesis --headlight lights on demo1 poses at 8 / 32 / 36 / 40 s vs MuJoCo ideal (the fit)
  checker-render SIM  checkerboard renders through the pipeline (CFG env: geom | static); one sim per process
  checker             K / D recovery with physicalai_sim2real_camera.calib_geom and corner agreement by image radius
"""

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import genesis_allex as ga  # noqa: E402

HUES = ((0, 10), (170, 180), (25, 35), (50, 70), (85, 95), (115, 125))  # red (2 bands), yellow, green, cyan, blue
FPS = 30.0
TURN = (14.0, 42.0)  # demo1: waist / neck turning segment, s
COLS, ROWS, SQ = 10, 7, 0.04  # checkerboard squares (9 x 6 inner corners), 4 cm


def mask(rgb):
    """Union of the saturated target-box hues: a silhouette that does not depend on the renderer's shading."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sel = np.zeros(hsv.shape[:2], bool)
    for lo, hi in HUES:
        sel |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
    return sel & (hsv[..., 1] > 150) & (hsv[..., 2] > 40)


def iou(a, b):
    """Intersection over union of two boolean masks (nan when both are empty)."""
    u = (a | b).sum()
    return np.nan if u == 0 else (a & b).sum() / u


def grad_energy(rgb):
    """Mean |Sobel| of the grey image: drops with motion blur."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))))


def load(tag):
    with np.load(ga.OUT / f"cam_{tag}.npz") as z:  # materialise once: NpzFile re-reads an array per key access
        return {k: z[k] for k in z.files}


def compare():
    mw, gw, mg, gg = (load(t) for t in ("mujoco_wingeom", "genesis_wingeom", "mujoco_geom", "genesis_geom"))
    assert np.array_equal(mw["n"], gw["n"]) and np.array_equal(mg["n"], gg["n"]), "capture indices differ"
    if mw["n"][-1] / FPS < TURN[1]:
        raise SystemExit(f"captures end at {mw['n'][-1] / FPS:.1f} s: rerun genesis_allex.py with --seconds 42.7")
    t = mw["n"] / FPS
    turn = (t >= TURN[0]) & (t <= TURN[1])
    ious = np.array([iou(mask(a), mask(b)) for a, b in zip(mw["real"], gw["real"])])
    print(f"window silhouettes Genesis vs MuJoCo: IoU median {np.nanmedian(ious):.3f}, turning median "
          f"{np.nanmedian(ious[turn]):.3f}, turning p10 {np.nanpercentile(ious[turn], 10):.3f}")
    # blur amount = 1 - gradient energy (window) / gradient energy (global shutter) of the same capture, per sim:
    # the ratio cancels each renderer's own shading
    _, ia, ib = np.intersect1d(mw["n"], mg["n"], return_indices=True)
    blur = {s: np.array([1 - grad_energy(w["real"][i]) / grad_energy(g["real"][j]) for i, j in zip(ia, ib)])
            for s, w, g in (("mujoco", mw, mg), ("genesis", gw, gg))}
    tt = mw["n"][ia] / FPS
    moving = (tt >= TURN[0]) & (tt <= TURN[1]) & (blur["mujoco"] > 0.05)
    bm, bg = np.median(blur["mujoco"][moving]), np.median(blur["genesis"][moving])
    corr = np.corrcoef(blur["mujoco"][moving], blur["genesis"][moving])[0, 1]
    print(f"blur amount over {moving.sum()} blurred turning frames: MuJoCo {bm:.3f}, Genesis {bg:.3f} "
          f"(rel diff {abs(bg - bm) / bm:.1%}, per-frame corr {corr:.2f})")
    static = tt < TURN[0] - 0.5
    for s, w, g in (("mujoco", mw, mg), ("genesis", gw, gg)):
        v = np.array([iou(mask(w["real"][i]), mask(g["real"][j])) for i, j in zip(ia[static], ib[static])])
        print(f"static head, window vs global shutter ({s}): IoU median {np.nanmedian(v):.3f}, min {np.nanmin(v):.3f}")
    dd = []
    for a, b, ra, rb in zip(mg["depth"], gg["depth"], mg["real"], gg["real"]):
        m = mask(ra) & mask(rb) & (a > 0) & (b > 0)
        if m.sum() > 100:
            dd.append(np.median(np.abs(a[m] - b[m])))
    print(f"depth on targets |Genesis - MuJoCo|: median {1e3 * np.median(dd):.2f} mm, worst frame {1e3 * max(dd):.2f} "
          f"mm; background {mg['depth'][0].max():.4f} m vs {gg['depth'][0].max():.4f} m")
    ref = ga.OUT / "cam_genesis_none.npz"
    if ref.exists():
        r = load("genesis_none")["traj"]
        print(f"Genesis physics with the window camera vs without: max |dq| "
              f"{np.abs(gw['traj'] - r[:len(gw['traj'])]).max():.2e} rad")


def physics():
    names = [str(x) for x in np.load(ga.ALLEX / "examples" / "motions" / "demo1.npz")["joint_names"]]
    g, m = load("genesis_none")["traj"], load("mujoco_geom")["traj"]
    rms = np.sqrt(np.mean((g - m[:len(g)]) ** 2, axis=0))
    groups = {"waist/neck": ("Waist", "Neck"), "arms": ("Shoulder", "Elbow", "Wrist"),
              "hands": ("Thumb", "Index", "Middle", "Ring", "Little")}
    for gname, keys in groups.items():
        ix = [i for i, n in enumerate(names) if any(k in n for k in keys)]
        worst = ix[int(np.argmax(rms[ix]))]
        print(f"joint RMS Genesis vs MuJoCo, {gname} ({len(ix)}): mean {rms[ix].mean():.4f} rad, max {rms[worst]:.4f} "
              f"rad ({names[worst]})")


def lights():
    """Ideal brightness and AE gain (segment means, medians in brackets) per segment of demo1.

    2-14 s: head still, AE settled. 14-30 s: waist / neck turning, views much like the still ones. 30-42 s: neck
    pitched down, the floor fills the view. 14-42 s: the whole turning part.
    """
    segments = (("2-14 s", 2.0, 14.0), ("14-30 s", 14.0, 30.0), ("30-42 s", 30.0, 42.0), ("14-42 s", 14.0, 42.0))

    def summary(z, lo, hi):
        t = z["n"] / FPS
        sel = (t > lo) & (t < hi)
        gain = 2.0 ** z["ev"][sel]  # exposure_ev = 0 in "real"
        return z["ideal"][sel].mean(), gain.mean(), np.median(gain)

    m = load("mujoco_real")
    for seg, lo, hi in segments:
        mb, mg, mmed = summary(m, lo, hi)
        print(f"[{seg}] MuJoCo          ideal mean {mb:6.1f}  AE gain mean {mg:.3f} (median {mmed:.3f})")
        for name, tag in (("Genesis default", "genesis_real"), ("Genesis fixed", "genesis_real_lit"),
                          ("Genesis head", "genesis_real_head")):
            if (ga.OUT / f"cam_{tag}.npz").exists():
                b, g, med = summary(load(tag), lo, hi)
                print(f"[{seg}] {name:15s} ideal mean {b:6.1f}  AE gain mean {g:.3f} (median {med:.3f})  "
                      f"brightness {b / mb - 1:+.1%}, gain mean {g / mg - 1:+.1%}")

def light_poses():
    """The scene_lights(forward=False) + headlight fit: ideal brightness on demo1 poses vs MuJoCo (Genesis venv)."""
    import genesis as gs
    import mujoco

    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

    xml = ga.ALLEX / "mjcf" / "scene_targets.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    names = [str(x) for x in np.load(ga.ALLEX / "examples" / "motions" / "demo1.npz")["joint_names"]]
    mz, traj = load("mujoco_real"), load("genesis_none")["traj"]  # traj step k is at (101 + k) * 5 ms
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    cam = GenesisCamera(xml, ga.CAM, CameraConfig(ga.W, ga.H, latency_s=0.05), 0.005,
                        vis_options=ga.scene_lights(gs, forward=False), ideal=True, headlight=ga.HEADLIGHT)
    for tt in (8.0, 32.0, 36.0, 40.0):
        q = np.zeros(m.nq, np.float32)
        for j, n in enumerate(names):
            q[m.joint(n).qposadr[0]] = traj[int(round(tt / 0.005)) - 101, j]
        cam.mirror(q)
        want = float(mz["ideal"][int(np.argmin(np.abs(mz["n"] / FPS - tt)))].mean())
        got = float(cam.render_ideal().mean())
        print(f"demo1 pose at {tt:4.1f} s: MuJoCo ideal {want:5.1f}, Genesis headlight {got:5.1f} "
              f"({got / want - 1:+.1%})")


def board_geoms():
    """White plate in the body's y-z plane facing -x, black squares 0.5 mm in front: no textures needed."""
    g = [f'<geom type="box" size="0.002 {(COLS + 2) * SQ / 2} {(ROWS + 2) * SQ / 2}" rgba="1 1 1 1" contype="0" '
         'conaffinity="0"/>']
    for i in range(COLS):
        for j in range(ROWS):
            if (i + j) % 2 == 0:
                y, z = (i + 0.5 - COLS / 2) * SQ, (j + 0.5 - ROWS / 2) * SQ
                g.append(f'<geom type="box" size="0.0005 {SQ / 2} {SQ / 2}" pos="-0.0025 {y:.4f} {z:.4f}" '
                         'rgba="0 0 0 1" contype="0" conaffinity="0"/>')
    return "\n      ".join(g)


def board_poses(n=15, seed=0):
    """Board poses in front of the head camera (~(0.154, 0.033, 1.0), looking +x): distance, offset, tilt."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.uniform(0.62, 0.95)
        pos = np.array([x, 0.033 + rng.uniform(-0.12, 0.12) * x, 1.0 + rng.uniform(-0.08, 0.08) * x])
        yaw, pitch, roll = np.radians(rng.uniform([-25, -25, -15], [25, 25, 15]))
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        q = np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy])  # z-y-x Euler -> (w, x, y, z)
        out.append((pos, q))
    return out


def checker_render(sim):
    import mujoco

    cfg = ga.make_cfg(os.environ.get("CFG", "geom"))
    xml = ga.scene_variant("scene_board", f'    <body name="board" mocap="true" pos="0.8 0 1.0">\n      '
                                          f'{board_geoms()}\n    </body>\n  </worldbody>')
    out = []
    if sim == "mujoco":
        from physicalai_sim2real_camera import SimCamera

        m = mujoco.MjModel.from_xml_path(str(xml))
        d = mujoco.MjData(m)
        cam = SimCamera(m, ga.CAM, cfg, device="cuda")
        for pos, q in board_poses():
            d.mocap_pos[0], d.mocap_quat[0] = pos, q
            mujoco.mj_forward(m, d)
            out.append(cam.render(d).copy())
    else:
        import types

        import genesis as gs

        from physicalai_sim2real_camera.genesis_bridge import GenesisCamera

        ent = ga.ALLEX / "mjcf" / "board_entity.xml"
        ent.write_text(f'<mujoco model="board">\n  <worldbody>\n    <body name="board" pos="0.8 0 1.0">\n      '
                       f'<freejoint/>\n      {board_geoms()}\n    </body>\n  </worldbody>\n</mujoco>\n')
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        cam = GenesisCamera(ga.ALLEX / "mjcf" / "scene.xml", ga.CAM, cfg, 0.005, vis_options=ga.scene_lights(gs),
                            extra_morphs=[gs.morphs.MJCF(file=str(ent))])
        board = cam.extra_entities[0]
        data = types.SimpleNamespace(time=0.0, qpos=cam.entity.get_qpos())  # robot at qpos0, static renders
        for pos, q in board_poses():
            board.set_qpos(np.concatenate([pos, q]))
            out.append(cam.render(data))
    np.save(ga.OUT / f"checker_{sim}_{os.environ.get('CFG', 'geom')}.npy", np.stack(out))
    print(f"[checker {sim}] {len(out)} views")


def checker():
    import mujoco

    from physicalai_sim2real_camera.calib_geom import distortion_field_error, fit_intrinsics_points
    from physicalai_sim2real_camera.config import intrinsics_from_model

    name = os.environ.get("CFG", "geom")
    cfg = ga.make_cfg(name)
    m = mujoco.MjModel.from_xml_path(str(ga.ALLEX / "mjcf" / "scene_board.xml"))
    K0, d0 = intrinsics_from_model(m, m.camera(ga.CAM).id, ga.W, ga.H), np.array(cfg.dist)
    pattern = (COLS - 1, ROWS - 1)
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * SQ
    found = {}
    for sim in ("mujoco", "genesis"):
        cs = []
        for im in np.load(ga.OUT / f"checker_{sim}_{name}.npy"):
            ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), pattern, cv2.CALIB_CB_ACCURACY)
            cs.append(c.reshape(-1, 2) if ok else None)
        used = [c for c in cs if c is not None]
        # k3 fixed: with corners out to r ~ 300 px, k1/k2/k3 trade off; the lens field is the robust comparison
        K, d, rms, _, std = fit_intrinsics_points([obj] * len(used), [c.reshape(-1, 1, 2) for c in used],
                                                  (ga.W, ga.H), flags=cv2.CALIB_FIX_K3)
        field = distortion_field_error(K0, d0, K, d, np.concatenate(used), ga.W, ga.H)
        found[sim] = (K, d, cs)
        print(f"[{sim}/{name}] views {len(used)}/{len(cs)} rms {rms:.3f} px | fx {K[0, 0] / K0[0, 0] - 1:+.2%} "
              f"fy {K[1, 1] / K0[1, 1] - 1:+.2%} cx {K[0, 2] - K0[0, 2]:+.2f} cy {K[1, 2] - K0[1, 2]:+.2f} px | "
              f"k1 {d[0]:+.4f} (cfg {d0[0]:+.4f}, sd {std['k1']:.4f}) | lens field vs cfg {field:.2f} px")
    (_, _, cm), (_, _, cg) = found["mujoco"], found["genesis"]
    pairs = [(a, b) for a, b in zip(cm, cg) if a is not None and b is not None]
    a, b = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    diff, r = np.linalg.norm(b - a, axis=1), np.linalg.norm(a - [(ga.W - 1) / 2, (ga.H - 1) / 2], axis=1)
    print(f"  corners Genesis - MuJoCo, signed mean (dx, dy): {np.round((b - a).mean(0), 3)} px")
    for lo, hi in ((0, 150), (150, 250), (250, 400)):
        s = (r >= lo) & (r < hi)
        if s.any():
            print(f"  corners r {lo}-{hi} px (n {s.sum()}): |Genesis - MuJoCo| mean {diff[s].mean():.3f}, "
                  f"max {diff[s].max():.3f} px")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "checker-render":
        checker_render(sys.argv[2])
    else:
        {"compare": compare, "lights": lights, "checker": checker, "physics": physics,
         "light-poses": light_poses}[cmd]()


if __name__ == "__main__":
    main()
