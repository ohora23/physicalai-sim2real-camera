"""Online photometric observer: shot and read noise from still intervals of the live stream.

The photon-transfer fit of Phase 3 (``calib_photo.ptc_points`` / ``fit_ptc``) needs stacks of frames of a
static scene at several brightness levels. Online, the levels come from the grey-wedge patches (and the black squares)
of the prism target, and the stacks from **still intervals**: runs of frames in which the robot reports no motion
(joint speed <= ``eps``), the image itself does not change (median absolute change of the patch means <= ``tau`` DN)
and the camera's exposure / gain metadata are constant. An interval is dropped when a patch mean drifts by more than 1 %
between its halves (auto-exposure creeping, the same guard as ``scripts/calibrate.py``). Each frame is divided by
its all-patch mean before the temporal statistics are taken, which removes mains flicker (a per-frame common-mode
gain) without touching the per-pixel noise. Gamma and black level are offline inputs (D3/D4). The fit is refused
when the brightest and darkest usable patch differ by less than ``MIN_SPAN`` in linear units: the slope of the
mean-variance line is then not identifiable. Shot noise scales with analogue gain, so the fit uses only the intervals
taken at the latest gain and reports that gain. Only the patch crops are kept (a parked robot would otherwise hold
full frames without bound), and an interval is closed at ``MAX_FRAMES``.
"""

from __future__ import annotations

import numpy as np

from .calib_photo import ptc_line, ptc_points

MIN_SPAN = 8.0        # m_lin max / min (lit patches) needed for an identifiable slope
DRIFT_TOL = 0.01      # max relative change of a linearised patch mean between interval halves (AE guard)
MIN_FRAMES = 20       # frames per still interval before it counts
MAX_INTERVALS = 16    # sliding set of intervals kept for the fit
MAX_FRAMES = 60       # an interval is closed (and a new one started) at this many frames: bounds the memory
BOX_TOL_PX = 2        # boxes moving more than this (px) break the interval; <= wedge_patches' shrink_px: crops stay in
STD_FLOOR_REL = 0.02  # relative term added in quadrature to the sigma std: the line fit's own std reads the shot
                      # scatter 1.4x low over 50 synthetic seeds, and O6 measured a 1.6 % model error


def flicker_normalise(frames, boxes):
    """Divide each frame by its all-patch mean, relative to the stack's mean (removes a per-frame common-mode gain).

    Args:
        frames: ``(n, H, W)`` or ``(n, H, W, 3)`` float array (DN).
        boxes: Patch boxes ``(x0, y0, x1, y1)``.

    Returns:
        Normalised copy of ``frames``.
    """
    f = np.asarray(frames, np.float64)
    scale = common_mode_scale([f[:, y0:y1, x0:x1] for x0, y0, x1, y1 in boxes])
    return f * scale.reshape((-1,) + (1,) * (f.ndim - 1))


def common_mode_scale(stacks):
    """Per-frame factor that divides out a common-mode gain: stack-mean all-patch level over each frame's level.

    Args:
        stacks: Per-patch ``(n, h, w[, 3])`` arrays over the same ``n`` frames.

    Returns:
        ``(n,)`` float64 factors.
    """
    per_frame = np.mean([s.reshape(len(s), -1).mean(axis=1) for s in stacks], axis=0)
    return per_frame.mean() / np.maximum(per_frame, 1e-9)


def drifted(stack, tol=DRIFT_TOL):
    """True when a patch's mean changes by more than ``tol`` between the first and second half of the interval."""
    n = len(stack) // 2
    a, b = float(np.mean(stack[:n])), float(np.mean(stack[n:]))
    return abs(b - a) > tol * max(abs(a), 1e-9)


class PhotoObserver:
    """Streaming photon-transfer estimator over still intervals; see the module docstring."""

    def __init__(self, gamma, black_level_lin, eps_speed=1e-3, tau_dn=1.0, min_frames=MIN_FRAMES):
        """Set the offline photometric constants and the still-interval thresholds.

        Args:
            gamma: Display gamma from D4 (``CameraConfig.gamma``).
            black_level_lin: Black level in linear units from D3 (``CameraConfig.black_level``).
            eps_speed: Max reported joint / head speed for a frame to count as still.
            tau_dn: Max median absolute change of the patch means (DN) between consecutive frames.
            min_frames: Frames an interval needs before it is used.
        """
        self.gamma, self.black, self.min_frames = gamma, black_level_lin, min_frames
        self.eps, self.tau = eps_speed, tau_dn
        self._cur = []            # per frame, the patch crops (DN) of the interval being collected
        self._cur_boxes = None    # patch boxes of the interval (fixed while still)
        self._cur_meta = None     # (exposure, gain) of the interval
        self._cur_lit = 0         # the first _cur_lit boxes are lit (wedge), the rest dark (black squares)
        self._prev = None
        self.intervals = []       # list of (patch stacks [(n, h, w[,3]) ...], gain, number of lit stacks)
        self.n_frames = self.n_still = self.n_rejected_motion = self.n_rejected_meta = self.n_rejected_drift = 0

    def push(self, frame, speed, boxes, exposure=None, gain=None, dark_boxes=()):
        """Feed one frame; closes the current interval when motion, a metadata change or a patch change breaks it.

        Args:
            frame: ``(H, W[, 3])`` uint8 frame.
            speed: Reported robot speed (joint or head), any consistent unit.
            boxes: Lit patch boxes ``(x0, y0, x1, y1)`` for this frame (wedge steps); ``[]`` = no target.
            exposure: Camera exposure metadata (any unit); ``None`` when unavailable.
            gain: Camera gain metadata.
            dark_boxes: Boxes at the black level (black squares). They enter the PTC fit (they pin the intercept,
                i.e. sigma_read) but not the drift guard (1 % of the black level is below the noise), the flicker
                scale (no lit signal to flicker) or the level span.

        Returns:
            ``True`` when the frame was added to a still interval.
        """
        self.n_frames += 1
        f = np.asarray(frame)
        n_lit = len(boxes)
        boxes = list(boxes) + list(dark_boxes)
        still = n_lit > 0 and speed <= self.eps
        # image stillness from the patch MEANS (spatial averages suppress the pixel noise, which at low light is
        # several DN after gamma encoding and would otherwise read as motion), not from a per-pixel difference
        means = np.array([f[y0:y1, x0:x1].mean() for x0, y0, x1, y1 in boxes[:n_lit]]) if n_lit else None
        if still and self._prev is not None and len(self._prev) == len(means):
            # compare the patch pattern with the common-mode gain divided out (flicker is not motion); scale the
            # relative change back to DN of the frame so ``tau`` keeps its unit
            rel, prev = means / max(means.mean(), 1e-9), self._prev / max(self._prev.mean(), 1e-9)
            still = float(np.median(np.abs(rel - prev)) * means.mean()) <= self.tau
        self._prev = means
        meta = (exposure, gain)
        moved = bool(self._cur) and (len(boxes) != len(self._cur_boxes) or n_lit != self._cur_lit or
                                     np.abs(np.subtract(boxes, self._cur_boxes)).max() > BOX_TOL_PX)
        if not still or (self._cur and (meta != self._cur_meta or moved or len(self._cur) >= MAX_FRAMES)):
            if self._cur and meta != self._cur_meta and still:
                self.n_rejected_meta += 1
            self._close()
            if not still:
                self.n_rejected_motion += 1
                return False
        if not self._cur:  # boxes frozen for the interval: 1-px corner noise must not shift the crops
            self._cur_boxes, self._cur_meta, self._cur_lit = [tuple(b) for b in boxes], meta, n_lit
        self._cur.append([f[y0:y1, x0:x1].copy() for x0, y0, x1, y1 in self._cur_boxes])
        self.n_still += 1
        return True

    def _close(self):
        if len(self._cur) >= self.min_frames:
            raw = [np.stack([fr[j] for fr in self._cur]).astype(np.float64) for j in range(len(self._cur_boxes))]
            # the AE guard runs on the raw (not flicker-normalised) crops, linearised so ``DRIFT_TOL`` is a gain
            # tolerance: a creeping exposure is a common-mode gain that the normalisation below would otherwise hide
            lit = self._cur_lit
            if any(drifted(np.clip(r / 255.0, 0.0, 1.0) ** self.gamma) for r in raw[:lit]):
                self.n_rejected_drift += 1
            else:
                scale = common_mode_scale(raw[:lit])
                stacks = [r * scale.reshape((-1,) + (1,) * (r.ndim - 1)) for r in raw[:lit]] + raw[lit:]
                self.intervals.append((stacks, self._cur_meta[1], lit))
                self.intervals = self.intervals[-MAX_INTERVALS:]
        self._cur, self._cur_boxes, self._cur_meta = [], None, None

    def flush(self):
        """Close the interval being collected (call at the end of a run)."""
        self._close()

    def estimate(self):
        """Fit shot and read noise on the kept intervals taken at the latest gain.

        Returns:
            ``None`` without intervals; else dict ``sigma_shot``, ``sigma_read``, ``r2``, ``n_intervals`` (used, i.e. at
            ``gain``), ``n_levels``, ``span`` (m_lin max / min over the lit patches), ``gain``
            (of the last interval), ``insufficient_levels`` (True when the span is below ``MIN_SPAN``, in which case
            the sigmas are reported but should not be used), ``sigma_shot_std`` / ``sigma_read_std`` (the line fit's
            covariance through the delta method, with ``STD_FLOOR_REL`` of the value added in quadrature).
        """
        if not self.intervals:
            return None
        gain = self.intervals[-1][1]
        used = [iv for iv in self.intervals if iv[1] == gain]  # shot noise scales with gain: never pool gains
        stacks = [s for ivs, _, _ in used for s in ivs]
        m, v = ptc_points(stacks, self.gamma, self.black)
        # span over the lit patches only: a black square's m_lin is ~0 and would make one wedge step look identifiable
        lit = np.array([((s / 255.0) ** self.gamma).mean() - self.black for ivs, _, n in used for s in ivs[:n]])
        pos = lit[lit > 0]
        if len(pos) < 2 or len(m) < 2:
            return dict(sigma_shot=None, sigma_read=None, r2=None, n_intervals=len(used), n_levels=len(m),
                        span=0.0, gain=gain, insufficient_levels=True)
        span = float(pos.max() / pos.min())
        # fit_ptc's line on the points already computed, with its covariance for the std (delta method:
        # shot = sqrt(slope), read^2 = icpt - slope * black)
        shot, read, r2, C = ptc_line(m, v, self.black, cov=True)
        var_r2 = C[1, 1] + self.black**2 * C[0, 0] - 2 * self.black * C[0, 1]
        shot_std = np.hypot(np.sqrt(C[0, 0]) / (2 * max(shot, 1e-12)), STD_FLOOR_REL * shot)
        read_std = np.hypot(np.sqrt(max(var_r2, 0.0)) / (2 * max(read, 1e-12)), STD_FLOOR_REL * read)
        return dict(sigma_shot=shot, sigma_read=read, r2=r2, n_intervals=len(used), n_levels=len(m),
                    span=span, gain=gain, insufficient_levels=span < MIN_SPAN, sigma_shot_std=float(shot_std),
                    sigma_read_std=float(read_std))
