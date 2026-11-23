"""Online geometric observer: K and distortion from a ChArUco prism seen while the robot moves.

Streaming API: ``push(frame, t_arrival, ts_sensor, seq, head_state)`` runs the prism detector, applies the
motion gate (image speed × (exposure + readout) < ``GATE_PX``: a frame whose corners move more than a
fifth of a pixel during capture would bias K and the distortion field under blur / rolling shutter; the speed is
checked towards both the previous and the next sighting, so a face is accepted one frame late, and sightings more
than ``GAP_PERIODS`` median frame intervals apart (``MAX_GAP_S`` before the period is known) never vouch for each
other: those are counted apart from the motion rejections) and
stores every detected face as its own **planar view** in a coverage-weighted sliding set: the image is
divided into ``grid`` cells, each cell keeps at most ``cap`` views whose corner centroid falls in it, and
the oldest view in a full cell is replaced; views older than ``max_age_s`` are evicted. A view sharing most of its
corners with a held one, within ``DUP_PX`` on them, replaces it:
a still delivers ~30 near-identical frames whose corner bias is shared, and counting them as independent views made
OpenCV's σ several times too small (80 held views were ~10 distinct poses; |err|/σ up to 4.7 for p1/p2).
``estimate()`` refits ``fit_intrinsics_points`` on the set (per-face planar Zhang views: prism build error between
faces never enters the fit), reports
``std`` = OpenCV's Jacobian σ (already residual-scaled, in pixels) combined with the ``detector_bias_floor``, the
coverage map and the cells still empty. ``reset()`` drops the window; the caller calls it on a known re-mount or
refocus. There is no automatic reset on a residual jump: planar views absorb a changed K into their poses (centre
+20 px and f +3 % moved the new views' median RMS only 0.079 -> 0.102 px), so the residual misses a real change and
trips on noise; detecting a change is the two-window agreement gate of O8. The initial K is
the nominal fovy intrinsics (never the simulator's own K): the observer must not see simulator truth.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .calib_geom import fit_intrinsics_points

GATE_PX = 0.2        # max corner motion during capture (px) for a frame to enter the fit
MAX_GAP_S = 0.1      # sightings further apart than this (s) do not measure speed, until the period is known
GAP_PERIODS = 2.5    # once >= GAP_MIN_DT frame intervals are seen, the gap limit is this x their median
GAP_MIN_DT = 5
DUP_PX = 2.0         # a view sharing >= DUP_SHARE of its corners with a held one, moved < this px on them, replaces it
DUP_SHARE = 0.5
MAX_AGE_S = 120.0    # views older than this (sensor time) are evicted: an unannounced refocus ages out
MIN_VIEWS = 3        # calibrateCamera needs >= 3 planar views
MIN_CORNERS = 6      # a face with fewer charuco corners is ignored
DETECTOR_BIAS_PX = 1.5  # principal-point floor: corner detectors bias foreshortened faces by ~0.05 px per corner,
#                         which the 5-coefficient fit turns into ~1-1.5 px of centre error that more views do not
#                         average out (invariant to MSAA, texture resolution and distortion)


def detector_bias_floor(objs, pts, size, K):
    """Per-field σ floor from a ``DETECTOR_BIAS_PX`` principal-point bias, propagated through the fit.

    Refits with the principal point held ``DETECTOR_BIAS_PX`` off along x, then along y; the shift of each field is
    how far the detector bias drags it (the distortion coefficients trade off against the centre, so a cx/cy-only
    floor left k1/k2 under-covered: |err|/σ <= 3 in 75 / 90 % of seeds).

    Args:
        objs: Per-view object points, as for ``fit_intrinsics_points``.
        pts: Per-view image points, same order.
        size: ``(W, H)`` image size in px.
        K: (3, 3) free-fit intrinsics around which to perturb.

    Returns:
        Dict ``{field: floor}`` over ``fx, fy, cx, cy, k1, k2, p1, p2, k3`` (cx, cy get ``DETECTOR_BIAS_PX``).
    """
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    K = np.asarray(K, np.float64)
    shifts = []
    base = None
    for dx, dy in ((0.0, 0.0), (DETECTOR_BIAS_PX, 0.0), (0.0, DETECTOR_BIAS_PX)):
        K1 = K.copy()
        K1[0, 2] += dx
        K1[1, 2] += dy
        Kf, df, _, _, _ = fit_intrinsics_points(objs, pts, size, K0=K1, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
        v = np.array([Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2], *df])
        if base is None:
            base = v  # same solver on the unshifted centre, so solver differences cancel
        else:
            shifts.append(v - base)
    return dict(zip(names, (float(x) for x in np.hypot(*np.abs(shifts)))))


def nominal_K(fovy_deg, W, H, scale=1.0):
    """Pinhole K from a vertical field of view (the MJCF ``fovy``), principal point centred.

    Args:
        fovy_deg: Vertical field of view in degrees.
        W: Image width, px.
        H: Image height, px.
        scale: Multiplier on the focal length (``1.05`` gives a deliberately-off initial guess).

    Returns:
        ``(3, 3)`` float64 intrinsics.
    """
    f = 0.5 * H / math.tan(math.radians(fovy_deg) / 2) * scale
    return np.array([[f, 0.0, (W - 1) / 2], [0.0, f, (H - 1) / 2], [0.0, 0.0, 1.0]])


def image_speed_px(prev, cur, dt):
    """Mean corner speed (px/s) between two detections of the same face, over the ids they share.

    Args:
        prev: ``(ids, px)`` of the earlier detection.
        cur: ``(ids, px)`` of the later detection.
        dt: Time between them, s.

    Returns:
        Mean displacement / ``dt``, or ``inf`` when ``dt <= 0`` or no id is shared (so the gate rejects).
    """
    ids0, px0 = prev
    ids1, px1 = cur
    common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
    if len(common) == 0 or dt <= 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(px1[i1] - px0[i0], axis=1)) / dt)


def motion_gate(speed_px_s, exposure_s, readout_s, gate_px=GATE_PX):
    """True when the corners move less than ``gate_px`` during one capture (exposure + readout)."""
    return speed_px_s * (exposure_s + readout_s) < gate_px


def _corner_keys(obj):
    """Integer keys of face-frame corners (0.1 mm grid), to match the corners two views share."""
    return np.round(np.asarray(obj, np.float64)[:, :2] * 1e4).astype(np.int64) @ np.array([1, 1 << 32], np.int64)


def _same_pose(key0, px0, key1, px1):
    """True when two views share >= ``DUP_SHARE`` of the smaller corner set and moved < ``DUP_PX`` on them."""
    common, i0, i1 = np.intersect1d(key0, key1, return_indices=True)
    if len(common) < max(1, DUP_SHARE * min(len(np.unique(key0)), len(np.unique(key1)))):
        return False
    return float(np.linalg.norm(px0[i0] - px1[i1], axis=1).mean()) < DUP_PX


class CoverageWindow:
    """Sliding set of distinct planar views bucketed by corner-centroid cell, ``cap`` per cell, oldest out first."""

    def __init__(self, size, grid=(4, 3), cap=3):
        """Create an empty window.

        Args:
            size: ``(W, H)`` image size in px.
            grid: ``(nx, ny)`` cells across the image.
            cap: Maximum views kept per cell.
        """
        self.W, self.H = size
        self.nx, self.ny = grid
        self.cap = cap
        self.cells = {}  # (ix, iy) -> list of (stamp, obj, pts, t, corner keys)
        self._stamp = 0

    def cell_of(self, px):
        """Cell index of a view from its corner centroid."""
        cx, cy = px.mean(axis=0)
        return (min(int(cx / self.W * self.nx), self.nx - 1), min(int(cy / self.H * self.ny), self.ny - 1))

    def add(self, obj, px, t=None):
        """Insert a view; it replaces a held view of the same pose, else in a full cell the oldest view is dropped.

        Same pose: the views share at least ``DUP_SHARE`` of the smaller corner set (edge corners come and go
        between frames of one still) and the shared corners moved less than ``DUP_PX`` on average. Keys are
        face-frame coordinates, equal on every face board, so another face at the same image pose also counts: as a
        planar view it is the same constraint.

        Args:
            obj: ``(n, 3)`` face-frame corner coordinates.
            px: ``(n, 2)`` image corners.
            t: View time, s (for ``evict``); ``None`` = never evicted by age.

        Returns:
            The cell it went to.
        """
        obj, px = np.asarray(obj, np.float32), np.asarray(px, np.float32)
        key = _corner_keys(obj)
        for c0, vs in self.cells.items():
            k = next((k for k, v in enumerate(vs) if _same_pose(v[4], v[2], key, px)), None)
            if k is not None:
                del vs[k]
                if not vs:
                    del self.cells[c0]
                break
        c = self.cell_of(px)
        views = self.cells.setdefault(c, [])
        self._stamp += 1
        views.append((self._stamp, obj, px, t, key))
        if len(views) > self.cap:
            views.sort(key=lambda v: v[0])
            del views[0]
        return c

    def evict(self, t_min):
        """Drop views stamped before ``t_min`` (views without a time stay).

        Returns:
            Number of views dropped.
        """
        n = len(self)
        for c in list(self.cells):
            self.cells[c] = [v for v in self.cells[c] if v[3] is None or v[3] >= t_min]
            if not self.cells[c]:
                del self.cells[c]
        return n - len(self)

    def views(self):
        """All views as ``(objs, pts)`` lists in insertion order."""
        allv = sorted((v for vs in self.cells.values() for v in vs), key=lambda v: v[0])
        return [v[1] for v in allv], [v[2].reshape(-1, 1, 2) for v in allv]

    def coverage(self):
        """``(ny, nx)`` int array of views per cell."""
        m = np.zeros((self.ny, self.nx), int)
        for (ix, iy), vs in self.cells.items():
            m[iy, ix] = len(vs)
        return m

    def clear(self):
        """Drop every view (window reset)."""
        self.cells.clear()

    def __len__(self):
        """Number of views held."""
        return sum(len(v) for v in self.cells.values())


class GeomObserver:
    """Streaming estimator of K and distortion; see the module docstring."""

    def __init__(self, target, size, fovy_deg, exposure_s=0.0, readout_s=0.0, grid=(4, 3), cap=10, k_scale=1.05,
                 max_age_s=MAX_AGE_S):
        """Set up the detector, the deliberately-off initial K and an empty coverage window.

        Args:
            target: ``PrismTarget`` (``detect(img) -> [(face, ids, px, obj)]``).
            size: ``(W, H)`` px.
            fovy_deg: Nominal vertical FOV for the initial K (the camera's spec sheet / MJCF value).
            exposure_s: Exposure used by the motion gate.
            readout_s: Rolling-shutter readout used by the motion gate (0 for global shutter).
            grid: Coverage grid ``(nx, ny)``.
            cap: Views kept per cell (10: with 12 still poses x 2 faces the centre std is ~0.6 px; 3 gives ~1.1 px).
            k_scale: Initial focal-length multiplier (1.05: start off on purpose).
            max_age_s: Views older than this (s, on the gate's time base) are evicted; ``None`` keeps them.
        """
        self.target, self.size = target, tuple(size)
        self.exposure_s, self.readout_s, self.max_age_s = exposure_s, readout_s, max_age_s
        self.K0 = nominal_K(fovy_deg, *size, scale=k_scale)
        self.window = CoverageWindow(size, grid, cap)
        self._last = {}     # face -> (t, ids, px) of the previous sighting
        self._pending = {}  # face -> (obj, px, t) that passed the backward gate, waiting for the next sighting
        self._t_prev, self._dts = None, []  # frame intervals for the gap limit
        self.K, self.dist, self.std, self.rms = self.K0.copy(), (0.0,) * 5, None, None
        self.n_pushed = self.n_accepted = self.n_rejected_motion = self.n_rejected_gap = self.n_resets = 0
        self.n_evicted = 0

    @property
    def max_gap_s(self):
        """Longest sighting interval that still measures speed: ``GAP_PERIODS`` x the median frame interval."""
        if len(self._dts) < GAP_MIN_DT:
            return MAX_GAP_S
        return GAP_PERIODS * float(np.median(self._dts))

    def push(self, frame, t_arrival, ts_sensor=None, seq=None, head_state=None):
        """Detect faces in ``frame`` and add the still-enough ones to the window.

        Args:
            frame: ``(H, W, 3)`` uint8 RGB or ``(H, W)`` gray.
            t_arrival: Receipt time on the observer's clock, s (the speed's time base when ``ts_sensor`` is None).
            ts_sensor: Sensor timestamp, s; preferred for the speed (no delivery jitter).
            seq: Unused here.
            head_state: Unused here (the gate uses corner motion, which needs no hand-eye transform).

        Returns:
            Number of faces accepted into the window by this frame (the previous frame's sightings that this one
            confirmed still).
        """
        return self.push_detections(self.target.detect(frame), t_arrival, ts_sensor)

    def push_detections(self, detections, t_arrival, ts_sensor=None):
        """``push`` on detections already made (``target.detect`` output), so a caller detects once per frame.

        Returns:
            Number of faces accepted into the window by this frame.
        """
        self.n_pushed += 1
        t = t_arrival if ts_sensor is None else ts_sensor
        if self._t_prev is not None and t > self._t_prev:
            self._dts = (self._dts + [t - self._t_prev])[-30:]  # ponytail: last 30 intervals, a frame-rate change
        self._t_prev = t                                         # re-learns within a second
        if self.max_age_s is not None:
            self.n_evicted += self.window.evict(t - self.max_age_s)
        gap = self.max_gap_s
        accepted = 0
        for face, ids, px, obj in detections:
            if len(ids) < MIN_CORNERS:
                continue
            prev = self._last.get(face)
            self._last[face] = (t, ids, px)
            dt = None if prev is None else t - prev[0]
            near = dt is not None and dt <= gap
            ok = near and motion_gate(image_speed_px((prev[1], prev[2]), (ids, px), dt), self.exposure_s,
                                      self.readout_s)
            pending = self._pending.pop(face, None)
            if pending is not None and ok:  # still towards the next sighting too: motion did not start mid-capture
                self.window.add(*pending)
                accepted += 1
            elif pending is not None:
                self._count_reject(near)
            if ok:
                self._pending[face] = (obj, px, t)
            elif prev is not None:
                self._count_reject(near)
        self.n_accepted += accepted
        return accepted

    def _count_reject(self, near):
        if near:
            self.n_rejected_motion += 1
        else:
            self.n_rejected_gap += 1

    def reset(self):
        """Drop the window and the pending sightings (call on a known re-mount or refocus)."""
        self.window.clear()
        self._last.clear()
        self._pending.clear()
        self._t_prev, self._dts = None, []
        self.n_resets += 1

    def estimate(self):
        """Refit on the current window.

        Returns:
            ``None`` when fewer than ``MIN_VIEWS`` views are held; otherwise a dict with ``K`` (3,3), ``dist`` (5,),
            ``std`` (per field, px / coefficient units; includes the ``detector_bias_floor``), ``std_fit``
            (the fit's own σ), ``rms`` px, ``n_views``, ``per_view`` (RMS px per view), ``coverage`` (ny, nx) and
            ``cells_to_cover`` (list of empty cells, i.e. where to point the camera next). The cx/cy floor equals
            ``DETECTOR_BIAS_PX``, so their |err|/σ <= 3 holds whenever the centre error is <= 4.5 px.
        """
        objs, pts = self.window.views()
        if len(pts) < MIN_VIEWS:
            return None
        # OpenCV's stdDeviationsIntrinsics are already in pixels and already scaled by the residual variance
        # (checked: scaling them again by the RMS under-reported σ nine-fold)
        K, dist, rms, per_view, std_fit = fit_intrinsics_points(objs, pts, self.size, K0=self.K0)
        bias = detector_bias_floor(objs, pts, self.size, K)
        # the fit's σ shrinks with views; the detector's systematic bias does not
        std = {f: float(np.hypot(std_fit[f], bias[f])) for f in std_fit}
        self.K, self.dist, self.std, self.rms = K, dist, std, rms
        cov = self.window.coverage()
        empty = [(int(ix), int(iy)) for iy in range(cov.shape[0]) for ix in range(cov.shape[1]) if cov[iy, ix] == 0]
        return dict(K=K, dist=np.array(dist), std=std, std_fit=std_fit, rms=rms, n_views=len(pts), per_view=per_view,
                    coverage=cov, cells_to_cover=empty)
