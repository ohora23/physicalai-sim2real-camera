"""Unit tests for ChArUco prism target module."""
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_prism import build_prism

from physicalai_sim2real_camera.target import PrismTarget


def test_prism_json_roundtrip(tmp_path: Path) -> None:
    """Generate prism assets via make_prism, then round-trip through PrismTarget.from_json."""
    build_prism(6, 5, 7, 1.0, 600, str(tmp_path))

    target = PrismTarget.from_json(tmp_path / "prism.json")
    faces = target.faces
    assert len(faces) == 6

    # IDs disjoint: each face has its own range, no overlaps.
    all_ids: set[int] = set()
    for f in faces:
        f_set = set(f.ids)
        assert f_set & all_ids == set(), f"Face {f.index} IDs overlap with others"
        all_ids |= f_set

    expected_total = 17 * 5 + 12
    assert len(all_ids) == expected_total
    assert all_ids == set(range(expected_total))

    # Every T_prism_face is a rigid transform.
    for f in faces:
        R = f.T_prism_face[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), \
            f"Face {f.index} R^T R != I"
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-9, f"Face {f.index} det(R) = {det}"

    # Face normals at ~60 deg steps.
    normals = np.array([f.T_prism_face[:3, 2] for f in faces])
    for i in range(6):
        for j in range(i + 1, 6):
            cos_a = np.dot(normals[i], normals[j])
            expected_cos = math.cos(2.0 * math.pi * abs(j - i) / 6.0)
            assert abs(cos_a - expected_cos) < 0.01, \
                f"Normals {i},j angle cos={cos_a} expected {expected_cos}"

    # Wedge face has 6 wedge boxes with increasing reflectances.
    wedge_face = faces[-1]
    assert wedge_face.wedge is not None
    assert len(wedge_face.wedge) == 6
    assert wedge_face.wedge_reflectances is not None
    for i in range(5):
        assert wedge_face.wedge_reflectances[i] < wedge_face.wedge_reflectances[i + 1]


def test_detect_warped_board(tmp_path: Path) -> None:
    """Detect ChArUco on a homography-warped board image."""
    out_dir = tmp_path / "detect_test"
    out_dir.mkdir()
    build_prism(6, 5, 7, 1.0, 600, str(out_dir))

    target = PrismTarget.from_json(out_dir / "prism.json")
    face_0 = target.faces[0]

    # Load the board image.
    board_img = cv2.imread(str(out_dir / "face_0.png"), cv2.IMREAD_COLOR)

    # Warp with a known homography onto a white canvas.
    H = np.array([[0.9, 0.08, 60], [-0.05, 0.95, 40], [1e-5, 2e-5, 1]], dtype=np.float64)
    canvas_w, canvas_h = 1400, 1000
    warped = cv2.warpPerspective(board_img, H, (canvas_w, canvas_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=(255, 255, 255))

    # Detect.
    results = target.detect(warped)
    face_0_results = [r for r in results if r[0] == 0]
    assert len(face_0_results) == 1, f"Expected 1 result for face 0, got {len(face_0_results)}"

    face_idx, corner_ids, px, obj = face_0_results[0]
    assert len(corner_ids) >= 6

    # Verify corner positions against expected board image corners mapped through H.
    px_per_m = 2000  # 20 px/cm = 2000 px/m
    margin_px = int(0.5 * face_0.square_m * px_per_m)  # margin_m = 0.5 * square_m

    # Check each detected corner: compute expected warped position and compare.
    for idx in range(len(corner_ids)):
        obj_pt = obj[idx][:2]
        # Convert board-frame metres to board-image pixel coords.
        bx = margin_px + obj_pt[0] * px_per_m
        by = margin_px + obj_pt[1] * px_per_m
        # Warp through H to get expected position on warped image.
        src = np.array([[bx, by]], dtype=np.float64).reshape(-1, 1, 2)
        expected = cv2.perspectiveTransform(src, H).reshape(-1, 2)[0]
        actual = px[idx]
        dist = np.linalg.norm(actual - expected)
        assert dist < 3.0, \
            f"Corner {idx}: detected {actual}, expected {expected}, dist={dist:.1f} px"

    # Detect on blank canvas -> empty.
    blank = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    blank_results = target.detect(blank)
    assert len(blank_results) == 0


def test_detect_matches_per_face_detect_board(tmp_path: Path) -> None:
    """detect()'s single marker pass gives the same faces, ids and px as each face's own detectBoard (6 passes)."""
    build_prism(6, 5, 7, 1.0, 600, str(tmp_path))
    target = PrismTarget.from_json(tmp_path / "prism.json")
    board_img = cv2.imread(str(tmp_path / "face_0.png"), cv2.IMREAD_COLOR)
    H = np.array([[0.9, 0.08, 60], [-0.05, 0.95, 40], [1e-5, 2e-5, 1]], dtype=np.float64)
    warped = cv2.warpPerspective(board_img, H, (1400, 1000), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)  # what detect() does with a 3-channel image
    ref = {}
    for i, det in enumerate(target._detectors):
        cc, ci, _, _ = det.detectBoard(gray)
        if cc is not None and ci is not None and len(cc) >= 6:
            ref[target.faces[i].index] = (ci.reshape(-1), target.board(i).matchImagePoints(cc, ci)[1].reshape(-1, 2))
    got = {f: (ids, px) for f, ids, px, _ in target.detect(warped)}
    assert sorted(got) == sorted(ref) and 0 in got, f"faces {sorted(got)} vs per-face {sorted(ref)}"
    for f, (ids, px) in ref.items():
        assert np.array_equal(got[f][0], ids), f"face {f}: corner ids differ"
        assert np.abs(got[f][1] - px).max() < 1e-4, f"face {f}: px differ by {np.abs(got[f][1] - px).max()}"


def test_face_ids_must_be_disjoint() -> None:
    """Overlapping marker ids between faces are rejected (detect() splits one marker pass by id)."""
    f = PrismTarget.from_json(Path(__file__).resolve().parent.parent / "assets" / "prism" / "prism.json").faces
    with pytest.raises(AssertionError, match="overlap"):
        PrismTarget([f[0], f[0]], 0.1, 0.2, 0.5, 600.0)


def test_wedge_patches(tmp_path: Path) -> None:
    """wedge_patches returns 6 non-overlapping boxes >= 20 px wide."""
    out_dir = tmp_path / "wedge_test"
    out_dir.mkdir()
    build_prism(6, 5, 5, 0.75, 600, str(out_dir))  # the make_prism.py defaults

    target = PrismTarget.from_json(out_dir / "prism.json")
    wedge_idx = len(target.faces) - 1

    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    rvec = np.zeros(3, dtype=np.float64)
    wedge_face = target.faces[wedge_idx]
    w = np.array(wedge_face.wedge)  # (x0, y0, x1, y1) per step, face frame; put the wedge's centre on the optical axis
    centre = [(w[:, 0].min() + w[:, 2].max()) / 2, (w[:, 1].min() + w[:, 3].max()) / 2]
    tvec = np.array([-centre[0], -centre[1], 2.0 * (wedge_face.cols * wedge_face.square_m + wedge_face.square_m)])

    boxes = target.wedge_patches(wedge_idx, rvec, tvec, K, dist, shrink_px=2)
    assert len(boxes) == 6, f"Expected 6 boxes, got {len(boxes)}"

    # All inside 640x480.
    for (x0, y0, x1, y1) in boxes:
        assert x0 >= 0 and y0 >= 0 and x1 <= 640 and y1 <= 480, \
            f"Box {(x0,y0,x1,y1)} outside 640x480"

    # Non-overlapping.
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            x0_i, y0_i, x1_i, y1_i = boxes[i]
            x0_j, y0_j, x1_j, y1_j = boxes[j]
            # Two boxes overlap if neither is completely above/below/left/right.
            no_overlap = (y1_i <= y0_j + 1 or y1_j <= y0_i + 1 or
                          x1_i <= x0_j + 1 or x1_j <= x0_i + 1)
            assert no_overlap, \
                f"Boxes {i} {boxes[i]} and {j} {boxes[j]} overlap"

    # Each >= 20 px wide.
    for (x0, y0, x1, y1) in boxes:
        assert x1 - x0 >= 20, f"Box width {x1 - x0} < 20 px"


def test_face_poses_place_every_board_one_margin_below_the_top() -> None:
    """Every face's board (the shorter wedge-face board included) starts one margin (half a square) below the prism
    top, spans its columns centred on the face, and lies on the face plane: catches a board pose offset that the
    per-face solvePnP reprojection checks cannot see."""
    target = PrismTarget.from_json(Path(ASSETS) / "prism.json")
    import json
    meta = json.loads((Path(ASSETS) / "prism.json").read_text())
    for face in target.faces:
        T = face.T_prism_face
        corners = np.array([[0, 0, 0], [face.cols * face.square_m, face.rows * face.square_m, 0]], float)
        p = (T[:3, :3] @ corners.T).T + T[:3, 3]
        assert p[0, 2] == pytest.approx(meta["height_m"] - face.square_m / 2, abs=1e-9), f"face {face.index} top"
        assert p[1, 2] > face.square_m / 2 - 1e-9, f"face {face.index} board runs off the bottom"
        normal = -T[:3, 2]  # z into the board -> outward normal is -z
        assert np.dot(p[0], normal) == pytest.approx(meta["apothem_m"], abs=1e-9)
        assert np.dot(p[0] + p[1], T[:3, 0]) / 2 == pytest.approx(0, abs=1e-9), f"face {face.index} not centred"


# ---- render-based acceptance: needs EGL; the prism assets come from scripts/make_prism.py ----

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "prism")


def visible_faces(target, model, data, K, W, H, max_obliquity_deg=60.0, min_inside=0.5):
    """Faces a detector can be expected to read: outward-facing, viewed at <= 60 deg obliquity (ArUco's practical
    limit) and with at least half of their corners inside the frame (a face three-quarters out of frame shows too few
    markers; ChArUco ids survive partial views beyond that)."""
    import mujoco
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    c_pos, c_mat = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    out = []
    for f in target.faces:
        R, t = f.T_prism_face[:3, :3], f.T_prism_face[:3, 3]
        w, h = f.cols * f.square_m, f.rows * f.square_m
        corners = (R @ np.array([[0, 0, 0], [w, 0, 0], [0, h, 0], [w, h, 0]], float).T).T + t
        centre, normal = corners.mean(0), -R[:, 2]  # board z points into the board; outward normal is -z
        v = centre - c_pos
        if normal @ v >= 0:
            continue
        if np.degrees(np.arccos(-normal @ v / np.linalg.norm(v))) > max_obliquity_deg:
            continue
        pc = (c_mat.T @ (corners - c_pos).T).T
        if np.any(pc[:, 2] >= 0):  # MuJoCo camera looks along -z
            continue
        u = K[0, 0] * (-pc[:, 0] / -pc[:, 2]) + K[0, 2]
        vv = K[1, 1] * (pc[:, 1] / -pc[:, 2]) + K[1, 2]
        if np.mean((u >= 0) & (u < W) & (vv >= 0) & (vv < H)) >= min_inside:
            out.append(f.index)
    return out


def test_prism_scene_detection_and_reprojection():
    """>= 90 % of the geometrically visible faces detected, reprojection with the true K < 0.3 px (no blur)."""
    import mujoco

    from physicalai_sim2real_camera import intrinsics_from_model
    from physicalai_sim2real_camera.scenes import prism_scene
    from physicalai_sim2real_camera.target import PrismTarget
    if not os.path.exists(os.path.join(ASSETS, "prism.xml")):
        pytest.skip("run scripts/make_prism.py --out assets/prism first")
    W, H = 640, 480
    target = PrismTarget.from_json(os.path.join(ASSETS, "prism.json"))
    model, data, sweep = prism_scene(os.path.join(ASSETS, "prism.xml"), width=W, height=H)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    K = intrinsics_from_model(model, cam, W, H)
    r = mujoco.Renderer(model, H, W)
    vis_total = det_total = 0
    errs = []
    try:
        for t in np.linspace(0.3, 16.0, 14):
            data.mocap_pos[0], data.mocap_quat[0] = sweep.pose(t)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera="cam")
            img = r.render()
            vis = visible_faces(target, model, data, K, W, H)
            det = target.detect(img)
            found = {f for f, *_ in det}
            vis_total += len(vis)
            det_total += len(found & set(vis))
            for f, ids, px, obj in det:
                ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), px.astype(np.float64), K, None)
                assert ok
                proj = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, None)[0].reshape(-1, 2)
                errs.append(np.sqrt(np.mean(np.sum((proj - px) ** 2, axis=1))))
    finally:
        r.close()
    rate = det_total / max(vis_total, 1)
    print(f"prism_scene: {det_total}/{vis_total} visible faces detected ({rate:.0%}), reprojection RMS "
          f"median {np.median(errs):.3f} px, max {np.max(errs):.3f} px over {len(errs)} face views")
    assert vis_total >= 10 and rate >= 0.9, f"detection rate {rate:.0%}"
    assert np.median(errs) < 0.3, f"median reprojection {np.median(errs):.3f} px"


def test_prism_scene_detection_under_blur():
    """N4 motion blur (exposure 8 ms) during the sweep's motion phases: >= 90 % of readable faces detected,
    reprojection with the true K < 0.8 px. The plan asks for <= 300 px/s; the sweep's motion phases measure a median
    ~300 and max ~680 px/s (0.1 s sampling; the old 0.5 s grid averaged over stills and read ~280), so passing here
    covers the plan's condition a fortiori."""
    import mujoco

    from physicalai_sim2real_camera import CameraConfig, SimCamera, intrinsics_from_model
    from physicalai_sim2real_camera.scenes import drive, prism_scene
    from physicalai_sim2real_camera.target import PrismTarget
    if not os.path.exists(os.path.join(ASSETS, "prism.xml")):
        pytest.skip("run scripts/make_prism.py --out assets/prism first")
    W, H = 640, 480
    target = PrismTarget.from_json(os.path.join(ASSETS, "prism.json"))
    model, data, sweep = prism_scene(os.path.join(ASSETS, "prism.xml"), width=W, height=H)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    K = intrinsics_from_model(model, cam_id, W, H)
    cam = SimCamera(model, "cam", CameraConfig(W, H, exposure_s=0.008, blur_samples=4))
    vis_total = det_total = 0
    errs, speeds = [], []
    prev = None
    try:
        while data.time < 16.0:
            drive(model, data, sweep)
            cam.push(data)
            if abs((data.time / 0.1) - round(data.time / 0.1)) > 1e-6 or data.time < 0.5 or sweep.still(data.time):
                continue  # sample every 0.1 s of motion (the stop-and-go sweep moves only 0.3 s per 1.3 s)
            img = cam.render(data)
            det = target.detect(img)
            found = {f: (ids, px) for f, ids, px, _ in det}
            # image speed from the previous sample's corners (same face, shared ids), px/s
            if prev is not None and data.time - prev[0] < 0.15:  # consecutive only: a still between dilutes the speed
                for f, (ids, px) in found.items():
                    if f in prev[1]:
                        ids0, px0 = prev[1][f]
                        c, i0, i1 = np.intersect1d(ids0, ids, return_indices=True)
                        if len(c):
                            speeds.append(np.mean(np.linalg.norm(px[i1] - px0[i0], axis=1)) / (data.time - prev[0]))
            prev = (data.time, found)
            vis = visible_faces(target, model, data, K, W, H)
            vis_total += len(vis)
            det_total += len(set(found) & set(vis))
            for f, ids, px, obj in det:
                ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), px.astype(np.float64), K, None)
                proj = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, None)[0].reshape(-1, 2)
                errs.append(np.sqrt(np.mean(np.sum((proj - px) ** 2, axis=1))))
    finally:
        cam.close()
    rate = det_total / max(vis_total, 1)
    print(f"prism_scene blur N4 8 ms: {det_total}/{vis_total} readable faces detected ({rate:.0%}), reprojection RMS "
          f"median {np.median(errs):.3f} px, max {np.max(errs):.3f}; image speed median {np.median(speeds):.0f} px/s, "
          f"max {np.max(speeds):.0f}")
    assert np.median(speeds) >= 100, "sweep too slow to exercise the motion blur"
    assert vis_total >= 10 and rate >= 0.9, f"detection rate {rate:.0%}"
    assert np.median(errs) < 0.8, f"median reprojection {np.median(errs):.3f} px"


def test_black_squares_are_black_on_the_board_image(tmp_path: Path) -> None:
    """Fronto-parallel wedge face: the 8 boxes of a 5 x 3 board (row + col even) all land on black, eroded 4 px."""
    build_prism(6, 5, 5, 0.75, 600, str(tmp_path))
    target = PrismTarget.from_json(tmp_path / "prism.json")
    i = len(target.faces) - 1
    face = target.faces[i]
    ppm = 800.0  # board image px per face metre; the camera below maps face metres to the same pixels
    img = target.board(i).generateImage((int(face.cols * face.square_m * ppm), int(face.rows * face.square_m * ppm)))
    K = np.array([[ppm, 0, 0], [0, ppm, 0], [0, 0, 1.0]])
    boxes = target.black_squares(i, np.zeros(3), np.array([0, 0, 1.0]), K, np.zeros(5))
    assert len(boxes) == 8
    side = face.square_m * ppm - 8  # eroded by 4 px per side
    for x0, y0, x1, y1 in boxes:
        assert abs((x1 - x0) - side) <= 2 and abs((y1 - y0) - side) <= 2
        assert img[y0:y1, x0:x1].max() == 0, (x0, y0, x1, y1)
    assert target.black_squares(i, np.zeros(3), np.array([0, 0, 50.0]), K, np.zeros(5)) == []  # < 8 px: none


@pytest.mark.parametrize("cols, rows", [(4, 4), (5, 3), (6, 5), (3, 6)])
def test_black_square_parity_on_any_board(cols, rows) -> None:
    """OpenCV's (non-legacy) ChArUco boards put the solid black squares at row + col even, for even and odd sizes."""
    board = cv2.aruco.CharucoBoard((cols, rows), 0.06, 0.045, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    img = board.generateImage((cols * 100, rows * 100))
    for r in range(rows):
        for c in range(cols):
            corner = int(img[r * 100 + 5, c * 100 + 5])  # inside the square, outside any marker's quiet zone
            assert (corner == 0) == ((r + c) % 2 == 0), (cols, rows, r, c)
