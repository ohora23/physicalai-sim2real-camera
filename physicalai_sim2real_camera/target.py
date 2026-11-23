"""ChArUco prism target: per-face boards with disjoint marker ids, detection, and grey-wedge patch lookup."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


@dataclass(frozen=True)
class Face:
    """Single face of the ChArUco prism target.

    Attributes:
        index: Face index (0-based).
        cols: Chessboard squares along x.
        rows: Chessboard squares along y.
        square_m: Square side length in metres.
        marker_m: Marker side length in metres.
        ids: Marker ids for this face, disjoint across faces.
        T_prism_face: (4, 4) face-to-prism transform; face frame = board frame
            (x along columns, y along rows, z into the board, metres).
        wedge: Per-step bounding boxes (x0, y0, x1, y1) in face metres.
        wedge_reflectances: Reflectance values for each wedge step, dark -> bright.
    """

    index: int
    cols: int
    rows: int
    square_m: float
    marker_m: float
    ids: Tuple[int, ...]
    T_prism_face: np.ndarray
    wedge: Optional[Tuple[Tuple[float, float, float, float], ...]] = None
    wedge_reflectances: Optional[Tuple[float, ...]] = None


class PrismTarget:
    """Loads prism.json; one CharucoBoard + CharucoDetector per face.

    detect(img) -> list of (face_index, corner_ids, px, obj_xyz_face) for every
    face with >= 6 charuco corners found.  img may be RGB or gray.

    wedge_patches(face_index, rvec, tvec, K, dist, shrink_px=2) -> list of int
    pixel bounding boxes for the wedge face, projected from the face frame; black_squares(...) the same for the
    solid black chessboard squares.
    """

    def __init__(self, faces: List[Face], apothem_m: float, height_m: float,
                 working_distance_m: float, focal_px: float) -> None:
        """Initialize PrismTarget.

        Args:
            faces: List of Face objects, one per prism face.
            apothem_m: Prism apothem in metres.
            height_m: Prism height in metres.
            working_distance_m: Working distance in metres.
            focal_px: Focal length in pixels.
        """
        self._faces = faces
        self._apothem_m = apothem_m
        self._height_m = height_m
        self._working_distance_m = working_distance_m
        self._focal_px = focal_px
        self._boards: List[cv2.aruco.CharucoBoard] = []
        self._detectors: List[cv2.aruco.CharucoDetector] = []
        for f in faces:
            board = cv2.aruco.CharucoBoard(
                (f.cols, f.rows),
                f.square_m,
                f.marker_m,
                DICT,
                ids=np.array(f.ids, dtype=np.int32),
            )
            self._boards.append(board)
            self._detectors.append(cv2.aruco.CharucoDetector(board))
        self._ids = [np.array(f.ids) for f in faces]
        # detect() runs one marker pass and hands each face the markers with its ids: they must not overlap
        all_ids = np.concatenate(self._ids) if self._ids else np.zeros(0)
        assert len(np.unique(all_ids)) == len(all_ids), "PrismTarget: face marker ids overlap"
        self._markers = cv2.aruco.ArucoDetector(DICT)

    @classmethod
    def from_json(cls, path: str) -> "PrismTarget":
        """Load prism target configuration from a JSON file.

        Args:
            path: Path to the prism.json file.

        Returns:
            PrismTarget instance.
        """
        with open(path, "r") as f:
            cfg = json.load(f)

        faces: List[Face] = []
        for fc in cfg["faces"]:
            T = np.array(fc["T_prism_face"], dtype=np.float64).reshape(4, 4)
            wedge = None
            wedge_refl = None
            if fc.get("wedge"):
                wedge = tuple(tuple(v for v in step) for step in fc["wedge"])
                wedge_refl = tuple(fc["wedge_reflectances"])  # type: ignore[arg-type]
            faces.append(Face(
                index=fc["index"],
                cols=fc["cols"],
                rows=fc["rows"],
                square_m=fc["square_m"],
                marker_m=fc["marker_m"],
                ids=tuple(fc["ids"]),
                T_prism_face=T,
                wedge=wedge,
                wedge_reflectances=wedge_refl,
            ))

        return cls(
            faces=faces,
            apothem_m=cfg["apothem_m"],
            height_m=cfg["height_m"],
            working_distance_m=cfg["working_distance_m"],
            focal_px=cfg["focal_px"],
        )

    @property
    def faces(self) -> List[Face]:
        """List of Face objects."""
        return self._faces

    @property
    def apothem_m(self) -> float:
        """Prism apothem in metres."""
        return self._apothem_m

    @property
    def height_m(self) -> float:
        """Prism height in metres."""
        return self._height_m

    @property
    def working_distance_m(self) -> float:
        """Working distance in metres."""
        return self._working_distance_m

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return self._focal_px

    def board(self, i: int) -> cv2.aruco.CharucoBoard:
        """Return the CharucoBoard for face i.

        Args:
            i: Face index.

        Returns:
            CharucoBoard object.
        """
        return self._boards[i]

    def detect(self, img: np.ndarray
               ) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Detect ChArUco corners on all faces.

        Args:
            img: Input image, RGB (H, W, 3) uint8 or grayscale (H, W).

        Returns:
            List of (face_index, corner_ids (n,), px (n, 2) float32,
            obj_xyz_face (n, 3) float32) for faces with >= 6 corners.
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        results: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        # one marker pass for all faces (their ids are disjoint), then each face interpolates its ChArUco corners from
        # its own markers: six detectBoard marker passes took 88 ms per 640x480 frame single-threaded, this 15 ms, with
        # identical corners
        marker_corners, marker_ids, _ = self._markers.detectMarkers(gray)
        if marker_ids is None:
            return results
        marker_ids = marker_ids.reshape(-1)
        for i, (board, detector) in enumerate(zip(self._boards, self._detectors)):
            sel = np.isin(marker_ids, self._ids[i])
            if not sel.any():
                continue
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(
                gray, markerCorners=tuple(c for c, k in zip(marker_corners, sel) if k),
                markerIds=marker_ids[sel].reshape(-1, 1))
            if charuco_corners is None or charuco_ids is None:
                continue
            if len(charuco_corners) < 6:
                continue
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is None or img_points is None:
                continue
            px = img_points.reshape(-1, 2).astype(np.float32)
            obj = obj_points.reshape(-1, 3).astype(np.float32)
            results.append((i, charuco_ids.reshape(-1), px, obj))
        return results

    def wedge_patches(self, face_index: int, rvec: np.ndarray, tvec: np.ndarray,
                      K: np.ndarray, dist: np.ndarray,
                      shrink_px: int = 2
                      ) -> List[Tuple[int, int, int, int]]:
        """Compute pixel bounding boxes for wedge patches.

        Projects each wedge step's 4 corners (face frame) with the face pose from ``cv2.solvePnP`` on that
        face's corners, takes the inscribed axis-aligned box, shrinks it inward by ``shrink_px`` and drops boxes
        smaller than 8 px in either direction.

        Args:
            face_index: Index of the wedge face.
            rvec: Rotation vector (3,) for the face pose.
            tvec: Translation vector (3,) for the face pose.
            K: 3x3 camera intrinsic matrix.
            dist: Distortion coefficients (5,).
            shrink_px: Pixels to shrink each bounding box on all sides.

        Returns:
            List of (x0, y0, x1, y1) int pixel bounding boxes.
        """
        face = self._faces[face_index]
        if face.wedge is None or face.wedge_reflectances is None:
            return []
        return _inscribed_boxes(face.wedge, rvec, tvec, K, dist, shrink_px)

    def black_squares(self, face_index: int, rvec: np.ndarray, tvec: np.ndarray,
                      K: np.ndarray, dist: np.ndarray, shrink_px: int = 4
                      ) -> List[Tuple[int, int, int, int]]:
        """Compute pixel bounding boxes for the solid black chessboard squares of a face.

        ChArUco markers sit in the white squares; the black ones (row + column even, as ``generateImage`` draws them)
        are uniform and give the photometric observer a level near the black level, which anchors the read noise.
        Boxes are eroded by ``shrink_px`` (blur radius margin) like ``wedge_patches``.

        Args:
            face_index: Face index.
            rvec: Rotation vector (3,) for the face pose.
            tvec: Translation vector (3,) for the face pose.
            K: 3x3 camera intrinsic matrix.
            dist: Distortion coefficients (5,).
            shrink_px: Pixels to shrink each bounding box on all sides.

        Returns:
            List of (x0, y0, x1, y1) int pixel bounding boxes, row-major over the board.
        """
        face = self._faces[face_index]
        s = face.square_m
        rects = [(c * s, r * s, (c + 1) * s, (r + 1) * s)
                 for r in range(face.rows) for c in range(face.cols) if (r + c) % 2 == 0]
        return _inscribed_boxes(rects, rvec, tvec, K, dist, shrink_px)


def _inscribed_boxes(rects_m, rvec, tvec, K, dist, shrink_px):
    """Project face-frame rectangles ``(x0, y0, x1, y1)`` m with the face pose; inscribed boxes >= 8 px, shrunk."""
    boxes_px: List[Tuple[int, int, int, int]] = []
    for (x0_m, y0_m, x1_m, y1_m) in rects_m:
        # rvec/tvec is the FACE pose (solvePnP on this face's planar corners), so project face-frame points
        corners_m = np.array([[x0_m, y0_m, 0.0], [x1_m, y0_m, 0.0], [x1_m, y1_m, 0.0], [x0_m, y1_m, 0.0]])
        corners_px = cv2.projectPoints(corners_m, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64),
                                       np.asarray(K, np.float64), np.asarray(dist, np.float64))[0].reshape(-1, 2)
        # inscribed box (corners 0..3 = top-left, top-right, bottom-right, bottom-left): a foreshortened patch's
        # bounding box would take in the neighbouring steps; then shrink inward to stay off the edges
        x0 = int(np.ceil(max(corners_px[0, 0], corners_px[3, 0]) + shrink_px))
        y0 = int(np.ceil(max(corners_px[0, 1], corners_px[1, 1]) + shrink_px))
        x1 = int(np.floor(min(corners_px[1, 0], corners_px[2, 0]) - shrink_px))
        y1 = int(np.floor(min(corners_px[2, 1], corners_px[3, 1]) - shrink_px))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            boxes_px.append((x0, y0, x1, y1))
    return boxes_px
