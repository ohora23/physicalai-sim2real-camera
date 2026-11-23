#!/usr/bin/env python3
"""Generate ChArUco prism target assets: JSON config, XML, textures, README.

Usage:
    make_prism.py --faces 6 --cols 5 --rows 7 --distance-m 1.0 --focal-px 600 --out assets/prism
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# OpenCV 5.0: dict and CharucoBoard constructors accept numpy arrays for ids.
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


def _round_up_to_5mm(v: float) -> float:
    """Round up to the nearest 5 mm."""
    return math.ceil(v / 0.005) * 0.005


def _quat_from_r(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion.

    Args:
        r: 3x3 rotation matrix.

    Returns:
        (4,) float64 quaternion [w, x, y, z].
    """
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([
            0.25 / s,
            (r[2, 1] - r[1, 2]) * s,
            (r[0, 2] - r[2, 0]) * s,
            (r[1, 0] - r[0, 1]) * s,
        ], dtype=np.float64)
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        return np.array([
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        ], dtype=np.float64)
    if r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        return np.array([
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        ], dtype=np.float64)
    s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
    return np.array([
        (r[1, 0] - r[0, 1]) / s,
        (r[0, 2] + r[2, 0]) / s,
        (r[1, 2] + r[2, 1]) / s,
        0.25 * s,
    ], dtype=np.float64)


def _marker_ids_for_face(face_index: int, n_markers: int) -> Tuple[int, ...]:
    """Return disjoint marker ids for face k with n_markers markers.

    Args:
        face_index: Face index (0-based).
        n_markers: Number of markers on this face.

    Returns:
        Tuple of marker ids.
    """
    return tuple(range(n_markers * face_index, n_markers * face_index + n_markers))


def _t_prism_face(face_index: int, n_faces: int, cols: int, rows: int,
                  square_m: float, margin_m: float, apothem_m: float,
                  height_m: float) -> np.ndarray:
    """Compute the face-to-prism transform for face k.

    The face frame is the OpenCV board frame: origin at the top-left corner of the squares area as seen
    from outside the prism, x along columns (viewer's right), y along rows (down), z into the board.

    Args:
        face_index: Face index (0-based).
        n_faces: Number of faces.
        cols: Chessboard columns.
        rows: Chessboard rows.
        square_m: Square side length in metres.
        margin_m: Margin around the squares area.
        apothem_m: Prism apothem.
        height_m: Prism height.

    Returns:
        4x4 transform matrix.
    """
    phi_k = 2.0 * math.pi * face_index / n_faces
    n_k = np.array([math.cos(phi_k), math.sin(phi_k), 0.0])        # outward face normal
    t_k = np.array([-math.sin(phi_k), math.cos(phi_k), 0.0])       # viewer's right, seen from outside
    up = np.array([0.0, 0.0, 1.0])
    # OpenCV board frame: x right, y down, z into the board (away from the viewer) -> x = t_k, y = -up, z = -n_k
    R = np.column_stack((t_k, -up, -n_k))
    c_k = apothem_m * n_k + (height_m / 2.0) * up                   # face centre
    # top-left corner of the squares: one margin below the face top (the wedge face's shorter board sits at the top
    # too, its strip below the squares, so centring on its own rows put it 1.5 squares low)
    origin = c_k - (cols * square_m / 2.0) * t_k + (height_m / 2.0 - margin_m) * up
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T


def _n_markers(cols: int, rows: int) -> int:
    """Number of markers in a checkerboard pattern on a cols x rows board."""
    return (cols * rows) // 2


def build_prism(faces: int, cols: int, rows: int, distance_m: float, focal_px: int, out_dir: str,
                cell_px: float = 6.0, square_m_override: float | None = None, px_per_m: float = 2000) -> None:
    """Build prism target assets.

    Args:
        faces: Number of faces.
        cols: Chessboard columns per face.
        rows: Chessboard rows per face.
        distance_m: Working distance in metres.
        focal_px: Focal length in pixels.
        out_dir: Output directory.
        cell_px: Marker cell size at the working distance, px (default 6).
        square_m_override: Fixed square size in metres instead of the formula.
        px_per_m: Texture resolution (default 2000 = 20 px/cm).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Size formula ---
    # marker = 0.75 square, DICT_4X4 = 6 cells across the marker (4 + border); cell_px cells at the working distance
    square_m = square_m_override or _round_up_to_5mm(max(0.05, cell_px * 6 / 0.75 * distance_m / focal_px))
    marker_m = 0.75 * square_m

    # --- Face geometry ---
    margin_m = 0.5 * square_m
    w = cols * square_m + 2 * margin_m
    h = rows * square_m + 2 * margin_m
    apothem_m = w / (2.0 * math.tan(math.pi / faces))

    # --- Per-face data ---
    face_data: List[dict] = []
    px_per_m = int(px_per_m)  # texture resolution

    # Pre-calculate marker counts per face for disjoint ID allocation.
    n_markers_per_face: List[int] = []
    for k in range(faces):
        if k == faces - 1:
            n_markers_per_face.append(_n_markers(cols, rows - 2))
        else:
            n_markers_per_face.append(_n_markers(cols, rows))

    cumulative = 0
    for k in range(faces):
        n_m = n_markers_per_face[k]
        ids = tuple(range(cumulative, cumulative + n_m))
        cumulative += n_m

        n_rows = rows - 2 if k == faces - 1 else rows
        T = _t_prism_face(k, faces, cols, n_rows, square_m, margin_m,
                          apothem_m, h)

        board = cv2.aruco.CharucoBoard(
            (cols, n_rows), square_m, marker_m, DICT,
            ids=np.array(ids, dtype=np.int32),
        )

        board_px_w = int((cols * square_m + 2 * margin_m) * px_per_m)
        board_px_h = int((n_rows * square_m + 2 * margin_m) * px_per_m)
        img = board.generateImage((board_px_w, board_px_h),
                                  marginSize=int(margin_m * px_per_m))

        face_path = out / f"face_{k}.png"
        cv2.imwrite(str(face_path), img)

        face_data.append({
            "index": k,
            "cols": cols,
            "rows": n_rows,
            "square_m": square_m,
            "marker_m": marker_m,
            "ids": list(ids),
            "T_prism_face": T.tolist(),
            "wedge": None,
            "wedge_reflectances": None,
        })

    # --- Wedge face: a strip of equal-width grey patches in the two freed bottom rows ---
    wedge_idx = faces - 1
    wedge_rows = rows - 2
    refl = [0.03, 0.08, 0.18, 0.36, 0.6, 0.9]
    n_wedge = len(refl)
    margin_px = int(margin_m * px_per_m)
    board_w_px = int(cols * square_m * px_per_m)
    strip_h_px = int(2 * square_m * px_per_m)
    wedge_img = cv2.imread(str(out / f"face_{wedge_idx}.png"), cv2.IMREAD_GRAYSCALE)
    strip = np.full((strip_h_px, wedge_img.shape[1]), 255, dtype=np.uint8)  # white margins like the board
    step_px = board_w_px / n_wedge
    for i_w, r in enumerate(refl):
        px_val = int(max(0, min(255, round(255.0 * r ** (1.0 / 2.2)))))  # sRGB-like encode of the reflectance
        x0 = margin_px + int(round(i_w * step_px))
        x1 = margin_px + int(round((i_w + 1) * step_px))
        strip[:, x0:x1] = px_val
    # the board image ends with its bottom margin; the strip goes right below the squares, the margin follows
    body, bottom = wedge_img[:-margin_px], wedge_img[-margin_px:]
    cv2.imwrite(str(out / f"face_{wedge_idx}.png"), np.vstack([body, strip, bottom]))
    # patch boxes in face (board) metres: x along columns, y down; the strip starts at the bottom of the squares
    step_m = cols * square_m / n_wedge
    wedge_steps = [(i_w * step_m, wedge_rows * square_m, (i_w + 1) * step_m, (wedge_rows + 2) * square_m)
                   for i_w in range(n_wedge)]
    face_data[wedge_idx]["wedge"] = tuple(wedge_steps)
    face_data[wedge_idx]["wedge_reflectances"] = refl
    patch_w_px = step_m * focal_px / distance_m
    print(f"Wedge patch width at the working distance: {patch_w_px:.1f} px (spec min 20 px)")
    if patch_w_px < 20:
        print("WARNING: wedge patch < 20 px; increase square_m, reduce distance_m, or increase focal_px")

    # --- prism.json ---
    prism_json = {
        "faces": face_data,
        "apothem_m": apothem_m,
        "height_m": h,
        "working_distance_m": distance_m,
        "focal_px": focal_px,
        "notes": f"Prism target built with square_m={square_m}, marker_m={marker_m}, "
                 f"distance_m={distance_m}, focal_px={focal_px}",
    }
    with open(out / "prism.json", "w") as f:
        json.dump(prism_json, f, indent=2)

    # --- prism.xml (MuJoCo) ---
    xml_lines: List[str] = []
    xml_lines.append('<?xml version="1.0" encoding="utf-8"?>')
    xml_lines.append('<mujoco model="prism">')
    xml_lines.append('  <compiler texturedir="."/>')
    xml_lines.append('  <asset>')

    for k in range(faces):
        xml_lines.append(f'    <texture name="face_{k}" type="2d" file="face_{k}.png"/>')
        xml_lines.append(
            f'    <material name="face_{k}_mat" texture="face_{k}" '
            f'texuniform="false" specular="0" shininess="0" reflectance="0"/>'
        )

    xml_lines.append('  </asset>')
    xml_lines.append('  <worldbody>')
    xml_lines.append('    <body name="prism" pos="0 0 0">')

    for k in range(faces):
        phi_k = 2.0 * math.pi * k / faces
        n_k = np.array([math.cos(phi_k), math.sin(phi_k), 0.0])
        t_k = np.array([-math.sin(phi_k), math.cos(phi_k), 0.0])
        up_k = np.array([0.0, 0.0, 1.0])

        R_geom = np.column_stack((t_k, up_k, n_k))
        q_k = _quat_from_r(R_geom)
        half_w = w / 2.0
        half_h = h / 2.0
        thickness = 0.002
        # the textured outer surface must lie exactly on the apothem plane that prism.json assumes: a box of
        # half-thickness `thickness` centred at apothem - thickness puts its +z face at the apothem
        c_k = (apothem_m - thickness) * n_k + (h / 2.0) * np.array([0.0, 0.0, 1.0])

        pos_str = f"{c_k[0]:.6f} {c_k[1]:.6f} {c_k[2]:.6f}"
        quat_str = f"{q_k[0]:.6f} {q_k[1]:.6f} {q_k[2]:.6f} {q_k[3]:.6f}"

        xml_lines.append(
            f'      <geom name="face_{k}" type="box" '
            f'size="{half_w:.6f} {half_h:.6f} {thickness:.6f}" '
            f'pos="{pos_str}" quat="{quat_str}" '
            f'material="face_{k}_mat"/>'
        )

    xml_lines.append('    </body>')
    xml_lines.append('  </worldbody>')
    xml_lines.append('</mujoco>')

    with open(out / "prism.xml", "w") as f:
        f.write("\n".join(xml_lines) + "\n")

    # --- README.md ---
    readme = f"""# Prism Target

## Build parameters

| Parameter | Value |
|-----------|-------|
| Faces | {faces} |
| Board size | {cols} x {rows} squares |
| Square size | {square_m * 100:.1f} cm |
| Marker size | {marker_m * 100:.1f} cm |
| Working distance | {distance_m:.2f} m |
| Focal length | {focal_px} px |
| Apothem | {apothem_m:.4f} m |
| Face width | {w * 100:.1f} cm |
| Face height | {h * 100:.1f} cm |

## Formulas

- `square_m = max(0.05, cell_px * 6 cells / 0.75 * distance_m / focal_px)` rounded up to 5 mm (cell_px 6 by default)
- `marker_m = 0.75 * square_m`
- `w = cols * square_m + 2 * margin` (margin = 0.5 * square_m)
- `apothem = w / (2 * tan(pi / N))`
- Face k angle: `phi_k = 2*pi*k/N`

## Files

- `prism.json` — target configuration (boards, transforms, wedge)
- `prism.xml` — MuJoCo model with textures and geoms
- `face_<k>.png` — ChArUco board texture for face k
- `README.md` — this file

## Build notes

- Print at 1:1 scale; measure edge lengths to verify.
- Use a rigid board (not paper) to avoid warping.
- Laser-cut or 3D-print the prism frame from the apothem and height.
"""
    with open(out / "README.md", "w") as f:
        f.write(readme)

    print(f"Wrote {len(list(out.iterdir()))} files to {out}")


def main() -> None:
    """Parse CLI args and build prism target."""
    parser = argparse.ArgumentParser(description="Generate ChArUco prism target assets")
    parser.add_argument("--faces", type=int, default=6, help="Number of faces")
    parser.add_argument("--cols", type=int, default=5, help="Chessboard columns")
    parser.add_argument("--rows", type=int, default=5,
                        help="Chessboard rows (5 x 5 squares: a 0.36 m face fills a 45 deg frame at 0.44 m)")
    parser.add_argument("--cell-px", type=float, default=6.0,
                        help="Marker cell size at the working distance, px (ArUco reads from ~6 px; tilt shrinks it)")
    parser.add_argument("--square-m", type=float, default=None, help="Override the square size (metres)")
    parser.add_argument("--px-per-m", type=float, default=2000, help="Texture resolution, px per metre")
    parser.add_argument("--distance-m", type=float, default=0.75,
                        help="Working distance in metres")
    parser.add_argument("--focal-px", type=int, default=600,
                        help="Focal length in pixels")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    build_prism(args.faces, args.cols, args.rows, args.distance_m, args.focal_px, args.out,
                cell_px=args.cell_px, square_m_override=args.square_m, px_per_m=args.px_per_m)


if __name__ == "__main__":
    main()
