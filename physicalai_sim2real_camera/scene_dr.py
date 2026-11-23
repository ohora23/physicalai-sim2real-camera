"""Render-time scene DR — perturb lights, colours, stains, walkers, distractors, lamps and the floor texture.

Perturb what the camera sees by editing the ``mujoco.MjvScene`` AFTER
``mjv_updateScene`` and BEFORE rendering.  The ``MjModel`` and ``MjData``
are never modified so physics is untouched.
The floor texture is the one exception to editing the ``MjvScene``: it
is uploaded into the camera's own GL context from a private copy of the model.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import cv2
import mujoco
import numpy as np

MAX_RENDER_LIGHTS = 8  # mjr_render shades with the first 8 scene lights (headlight included); later ones are dropped


@dataclasses.dataclass(frozen=True)
class Lamp:
    """An extra light source (desk lamp = spot, window / sun = directional).

    Attributes:
        pos: Position in world space. For a directional lamp MuJoCo centres the shadow map on the line through
            ``pos`` along ``dir``: put ``pos`` on a line through the area whose shadows matter (e.g. the table
            minus 2 m along ``dir``), or shadows there are clipped.
        dir: Direction (normalised in apply).
        rgb: Diffuse colour.
        cutoff: Spot cone half-angle, degrees.
        exponent: Spot exponent.
        directional: True for directional (sun/window), False for spot.
        castshadow: Whether this lamp casts shadows.
    """

    pos: tuple = (0.0, 0.0, 2.0)
    dir: tuple = (0.0, 0.0, -1.0)
    rgb: tuple = (1.0, 1.0, 1.0)
    cutoff: float = 45.0
    exponent: float = 2.0
    directional: bool = False
    castshadow: bool = True


@dataclasses.dataclass(frozen=True)
class Walker:
    """A person passing by: vertical capsule moving p0 -> p1 -> p0.

    Triangle-wave motion in time with configurable period and phase offset.

    Attributes:
        p0: Floor start position (x, y).
        p1: Floor end position (x, y).
        period_s: Full round-trip period in seconds.
        phase: Fraction of a period offset, [0, 1).
        radius: Capsule radius.
        height: Capsule total height.
        z0: Floor height (z).
        rgba: Walker colour and alpha.
    """

    p0: tuple = (1.0, -1.0)
    p1: tuple = (1.0, 1.0)
    period_s: float = 6.0
    phase: float = 0.0
    radius: float = 0.18
    height: float = 1.7
    z0: float = 0.0
    rgba: tuple = (0.3, 0.3, 0.35, 1.0)


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    """Configuration for scene DR effects applied during rendering.

    All effects are applied to the ``MjvScene`` without touching the model.

    Attributes:
        light_gain: Multiplier on diffuse and specular of model lights. MuJoCo shades in render values, which
            ``SimCamera`` decodes with ``cfg.gamma``: one unsaturated light scales linear radiance by
            ``light_gain ** gamma`` (0.5 -> 0.22 at gamma 2.2). The same holds per channel for ``light_tint``.
        light_tint: Per-channel multiply on the same model lights.
        ambient: Added to the ambient of every model light (room bounce), so the total grows with the number of
            lights. ``light_gain`` does not scale ambient (gym-hil's headlight has ambient 0.5): a dimmed room
            darkens less than ``light_gain ** gamma``.
        lamps: Extra light sources (desk lamps, windows, sun).
        walkers: People walking past who cast shadows.
        marks: Geom names that get stains.
        marks_per_geom: Number of stains per marked geom.
        mark_size: Mean stain semi-axis as fraction of the face's smaller
            half-extent.
        color_jitter: Per-geom per-channel rgb multiply range.
        color_geoms: Geom names to jitter; empty = every geom.
        floor_texture: Replace the 2D RGB texture of ``floor_geom``'s material by a random pattern (checker, stripes,
            blotches or flat; colours and scale from ``seed``), uploaded into the camera's own GL context only;
            the texture is replaced for every material that uses it.
        floor_geom: Geom whose material texture ``floor_texture`` replaces.
        floor_images: Image files (e.g. floor photos); with ``floor_texture`` one of them, chosen by ``seed`` and
            turned by a random multiple of 90 degrees when the texture is square, replaces the random pattern
            (resized to the texture with area averaging; the aspect ratio is not kept). Any format OpenCV reads.
        distractors: Number of static clutter objects (box, cylinder, sphere, capsule; random size, yaw and colour)
            drawn resting on z = 0; render-only, they cast shadows but have no physics;
            they can visually intersect moving bodies (no collision).
        distractor_region: Floor rectangle (x0, y0, x1, y1) in metres where distractors are placed.
        distractor_keepout: Optional rectangle (x0, y0, x1, y1) kept free of distractors (e.g. where the task object
            spawns); empty = none.
        distractor_size: (min, max) half-size in metres.
        shadow_clip: Half-size in metres of the region a light's shadow map covers, set on the camera's own
            renderer (``MjrContext.shadowClip``; 0 keeps the model's ``vis.map.shadowclip * stat.extent``). Raise
            it when walkers stand outside the scene's extent, or their shadows vanish (gym-hil: 0.8 m -> 3 m).
        seed: Random seed for reproducibility.
    """

    light_gain: float = 1.0
    light_tint: tuple = (1.0, 1.0, 1.0)
    ambient: float = 0.0
    lamps: tuple = ()
    walkers: tuple = ()
    marks: tuple = ()
    marks_per_geom: int = 6
    mark_size: float = 0.25
    color_jitter: float = 0.0
    color_geoms: tuple = ()
    floor_texture: bool = False
    floor_geom: str = "floor"
    floor_images: tuple = ()
    distractors: int = 0
    distractor_region: tuple = (-0.5, -0.5, 0.5, 0.5)
    distractor_keepout: tuple = ()
    distractor_size: tuple = (0.015, 0.05)
    shadow_clip: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate configuration constraints.

        Raises:
            ValueError: If any parameter violates its constraints.
        """
        if self.light_gain < 0:
            raise ValueError("light_gain must be >= 0")
        if self.ambient < 0:
            raise ValueError("ambient must be >= 0")
        if self.color_jitter < 0 or self.color_jitter >= 1:
            raise ValueError("color_jitter must be in [0, 1)")
        if self.marks_per_geom < 0:
            raise ValueError("marks_per_geom must be >= 0")
        if self.shadow_clip < 0:
            raise ValueError("shadow_clip must be >= 0")
        if self.mark_size <= 0:
            raise ValueError("mark_size must be > 0")
        if len(self.light_tint) != 3 or min(self.light_tint) < 0:
            raise ValueError("light_tint must be 3 values >= 0")
        object.__setattr__(self, "light_tint", tuple(float(x) for x in self.light_tint))
        object.__setattr__(self, "floor_images", tuple(str(f) for f in self.floor_images))  # hashable, Path == str
        if self.floor_images and not self.floor_texture:
            raise ValueError("floor_images needs floor_texture=True")
        if self.distractors < 0:
            raise ValueError("distractors must be >= 0")
        for name in ("distractor_region", "distractor_keepout"):
            r = getattr(self, name)
            if (name == "distractor_region" or r) and (len(r) != 4 or r[0] >= r[2] or r[1] >= r[3]):
                raise ValueError(f"{name} must be (x0, y0, x1, y1) with x0 < x1 and y0 < y1")
        if len(self.distractor_size) != 2 or not 0 < self.distractor_size[0] <= self.distractor_size[1]:
            raise ValueError("distractor_size must be (min, max) with 0 < min <= max")
        for w in self.walkers:
            if w.period_s <= 0:
                raise ValueError("walker period_s must be > 0")
            if w.radius <= 0:
                raise ValueError("walker radius must be > 0")
            if w.height < 2 * w.radius:
                raise ValueError("walker height must be >= 2 * radius")


def walker_pos(w: Walker, t: float) -> np.ndarray:
    """Compute walker position at time t.

    Triangle-wave motion between p0 and p1 with period_s and phase offset.

    Args:
        w: Walker configuration.
        t: Current simulation time.

    Returns:
        Float64 (3,) array = (x, y, z0 + height / 2), the capsule centre.
    """
    s = (t / w.period_s + w.phase) % 1.0
    u = 1.0 - abs(2.0 * s - 1.0)
    x = w.p0[0] + u * (w.p1[0] - w.p0[0])
    y = w.p0[1] + u * (w.p1[1] - w.p0[1])
    return np.array([x, y, w.z0 + w.height / 2], dtype=np.float64)


def mark_frames(gtype: int, size: np.ndarray, n: int, mean_size: float,
                rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, tuple, np.ndarray]]:
    """Stains in the geom's local frame: thin ellipsoids ON a face, the face drawn with probability ~ its area.

    Args:
        gtype: ``mujoco.mjtGeom`` type code (box or plane).
        size: The geom's ``model.geom_size`` row.
        n: Number of stains.
        mean_size: Mean stain semi-axis as a fraction of the face's smaller half-extent.
        rng: Generator; per stain it draws face (box only), u, v, a, b, angle, grey, alpha in that order.

    Returns:
        ``(pos (3,), R (3, 3), size (3,), rgba (4,) float32)`` per stain: R's columns are the stain's x, y axes
        and the face's outward normal; size = (a, b, 0.0004) ellipsoid semi-axes.

    Raises:
        ValueError: If gtype is not mjGEOM_BOX or mjGEOM_PLANE.
    """
    # faces as (normal axis, sign, in-plane axes b < c); a box has 6, a plane its +z face
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        h = np.asarray(size[:3], float)
        faces = [(a, sg, *[k for k in range(3) if k != a]) for a in range(3) for sg in (1.0, -1.0)]
        area = np.array([4.0 * h[b] * h[c] for _, _, b, c in faces])
        p_face = area / area.sum()
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        # ponytail: an infinite plane (half-extent 0) is stained over a 1 m half-extent square
        h = np.array([size[0] or 1.0, size[1] or 1.0, 0.0])
        faces, p_face = [(2, 1.0, 0, 1)], None
    else:
        raise ValueError(f"Unsupported geom type for marks: {gtype}")
    e = np.eye(3)
    out = []
    for _ in range(n):
        a, sg, b, c = faces[rng.choice(6, p=p_face)] if p_face is not None else faces[0]
        hb, hc = h[b], h[c]
        u, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
        sa = mean_size * min(hb, hc) * rng.uniform(0.5, 1.5)
        sb = mean_size * min(hb, hc) * rng.uniform(0.5, 1.5)
        angle = rng.uniform(0, math.pi)
        grey, alpha = rng.uniform(0.12, 0.35), rng.uniform(0.5, 0.9)
        r = max(sa, sb)  # keep the stain inside the face (no fin past the outline)
        pos = sg * h[a] * e[a] + u * max(hb - r, 0) * e[b] + v * max(hc - r, 0) * e[c]
        normal = sg * e[a]
        x = math.cos(angle) * e[b] + math.sin(angle) * e[c]
        out.append((pos, np.column_stack((x, np.cross(normal, x), normal)), (sa, sb, 0.0004),
                    np.array([grey, 0.85 * grey, 0.7 * grey, alpha], np.float32)))
    return out


def color_factors(n: int, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Generate per-geom colour multiply factors.

    Args:
        n: Number of geoms.
        jitter: Jitter amount in [0, 1). 0 means identity (no draw).
        rng: Numpy random generator.

    Returns:
        Float32 array of shape (n, 3).  If jitter == 0, returns ones
        without drawing from rng.
    """
    if jitter == 0.0:
        return np.ones((n, 3), dtype=np.float32)
    return rng.uniform(1 - jitter, 1 + jitter, (n, 3)).astype(np.float32)


TEXTURE_KINDS = ("checker", "stripes", "blotches", "flat")


def floor_texture_image(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A random tileable floor pattern: checker, stripes, blotches or flat, with two random colours and grain.

    Every pattern repeats seamlessly across the texture's edges (even cell and stripe counts, axis-aligned
    stripes, wrapped blotch grid), so a material's ``texrepeat`` shows no seams.

    Args:
        h: Texture height in pixels.
        w: Texture width in pixels.
        rng: Generator for every draw (kind, colours, scale, grain).

    Returns:
        uint8 (h, w, 3) image.
    """
    kind = TEXTURE_KINDS[int(rng.integers(len(TEXTURE_KINDS)))]
    c = rng.uniform(0.1, 0.95, (2, 3))
    y = np.arange(h)[:, None] / h
    x = np.arange(w)[None, :] / w
    if kind == "checker":
        n = 2 * int(rng.integers(1, 9))  # even: the last cell differs from the first, so tiles alternate
        t = (np.floor(x * n) + np.floor(y * n)) % 2
    elif kind == "stripes":
        n = 2 * int(rng.integers(2, 13))  # even, as for the checker
        t = np.floor((x if rng.random() < 0.5 else y) * n) % 2
    elif kind == "blotches":
        n = int(rng.integers(4, 17))
        g = rng.random((n, n))
        fy, fx = y * n, x * n
        iy, ix = np.floor(fy).astype(int) % n, np.floor(fx).astype(int) % n
        sy, sx = fy - np.floor(fy), fx - np.floor(fx)
        sy, sx = sy * sy * (3 - 2 * sy), sx * sx * (3 - 2 * sx)  # smoothstep
        top = g[iy, ix] * (1 - sx) + g[iy, (ix + 1) % n] * sx
        bot = g[(iy + 1) % n, ix] * (1 - sx) + g[(iy + 1) % n, (ix + 1) % n] * sx
        t = top * (1 - sy) + bot * sy
    else:  # flat
        t = np.zeros((h, w))
    t = np.broadcast_to(t, (h, w))[..., None]
    img = c[0] * (1 - t) + c[1] * t
    img = img * (1 + rng.normal(0.0, 0.04, (h, w, 1)))  # grain (per-pixel, i.i.d.: no seam)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def floor_photo(paths: tuple, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """One of ``paths`` as an (h, w, 3) uint8 RGB texture: chosen by ``rng``, area-resized, turned by k * 90 degrees.

    The turn (k from ``rng``, drawn even for a non-square texture so the stream does not depend on the shape) is
    applied only when ``h == w``.

    Args:
        paths: Image files readable by OpenCV.
        h: Texture height in pixels.
        w: Texture width in pixels.
        rng: Generator for the choice and the turn.

    Returns:
        uint8 (h, w, 3) RGB image.

    Raises:
        ValueError: A file that OpenCV cannot read.
    """
    path = paths[int(rng.integers(len(paths)))]
    k = int(rng.integers(4))
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read floor image {path!r}")
    img = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.rot90(img, k)) if h == w else img


_DISTRACTOR_TYPES = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE,
                     mujoco.mjtGeom.mjGEOM_CAPSULE)


def distractor_geoms(n: int, region: tuple, keepout: tuple, size: tuple, rng: np.random.Generator) -> list:
    """Static clutter objects resting on z = 0 inside ``region`` and clear of ``keepout``.

    Each candidate draws (x, y, type, half-size, two aspect factors, yaw, rgb) in that order; a candidate whose
    footprint radius reaches into ``keepout`` is rejected and redrawn (at most ``100 * n`` candidates).

    Args:
        n: Number of objects.
        region: Floor rectangle (x0, y0, x1, y1).
        keepout: Rectangle (x0, y0, x1, y1) to keep free, or empty.
        size: (min, max) half-size in metres.
        rng: Generator for every draw.

    Returns:
        List of ``(gtype, size3, pos3, mat9, rgba4)`` for ``_add_geom``; fewer than ``n`` only if the region is
        (almost) all keep-out.
    """
    out = []
    for _ in range(100 * n):
        if len(out) == n:
            break
        x, y = rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])
        gtype = _DISTRACTOR_TYPES[int(rng.integers(len(_DISTRACTOR_TYPES)))]
        s = rng.uniform(size[0], size[1])
        a, b = rng.uniform(0.5, 1.5, 2)
        yaw = rng.uniform(0.0, 2 * np.pi)
        rgb = rng.uniform(0.05, 0.95, 3)
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            sz, z, r = (s, s * a, s * b), s * b, s * max(1.0, a) * math.sqrt(2)
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            sz, z, r = (s, s * a, 0.0), s * a, s
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            sz, z, r = (s, 0.0, 0.0), s, s
        else:  # capsule, upright
            sz, z, r = (0.6 * s, s * a, 0.0), 0.6 * s + s * a, 0.6 * s
        if keepout and keepout[0] - r < x < keepout[2] + r and keepout[1] - r < y < keepout[3] + r:
            continue
        c, si = math.cos(yaw), math.sin(yaw)
        mat = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        out.append((gtype, np.array(sz), np.array([x, y, z]), mat, np.array([*rgb, 1.0], np.float32)))
    return out


def _geom_id(model, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"Geom not found: {name}")
    return gid


class SceneDR:
    """Render-time scene DR: edits a ``mujoco.MjvScene`` between ``mjv_updateScene`` and the render.

    ``MjModel`` and ``MjData`` are never touched. Stains and colour factors are drawn once here (stains
    first, so turning colour jitter on never changes the stains); the floor texture and the distractors each use their
    own stream from ``seed``, so turning one on never changes the other or the stains.

    Attributes:
        cfg: The SceneConfig this instance applies.
    """

    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig) -> None:
        """Resolve geom names and draw the stains, colour, floor texture and distractors.

        Args:
            model: The MuJoCo model.
            cfg: Scene configuration.

        Raises:
            ValueError: An unknown geom name in ``marks`` / ``color_geoms``, a stained geom that is not a box or
                plane, or ``floor_texture`` on a geom whose material lacks a 2D RGB texture.
        """
        rng = np.random.default_rng([cfg.seed, 11])
        self._marks = [(gid, *f) for gid in (_geom_id(model, nm) for nm in cfg.marks)
                       for f in mark_frames(model.geom_type[gid], model.geom_size[gid], cfg.marks_per_geom,
                                            cfg.mark_size, rng)]
        self._color = {}
        if cfg.color_jitter > 0:
            ids = [_geom_id(model, nm) for nm in cfg.color_geoms] or list(range(model.ngeom))
            self._color = dict(zip(ids, color_factors(len(ids), cfg.color_jitter, rng)))
        self._floor = None
        if cfg.floor_texture:
            fid = _geom_id(model, cfg.floor_geom)
            mat = model.geom_matid[fid]
            tex = int(model.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB]) if mat >= 0 else -1
            if tex < 0 or model.tex_type[tex] != mujoco.mjtTexture.mjTEXTURE_2D or model.tex_nchannel[tex] != 3:
                raise ValueError(f"floor_texture needs a 2D RGB texture on geom {cfg.floor_geom!r}'s material")
            h, w = int(model.tex_height[tex]), int(model.tex_width[tex])
            frng = np.random.default_rng([cfg.seed, 12])
            self._floor = (tex, floor_photo(cfg.floor_images, h, w, frng) if cfg.floor_images
                           else floor_texture_image(h, w, frng))
        self._distractors = (distractor_geoms(cfg.distractors, cfg.distractor_region, cfg.distractor_keepout,
                                              cfg.distractor_size, np.random.default_rng([cfg.seed, 13]))
                             if cfg.distractors else [])
        self._model = model
        self.cfg = cfg

    def bind(self, renderer: mujoco.Renderer) -> None:
        """Upload the drawn floor texture into ``renderer``'s own GL context (no-op without ``floor_texture``).

        The texture stays in that context: every camera rendering with this renderer sees it. ``SimCamera``
        therefore refuses ``floor_texture`` on a shared renderer, and refuses to share a renderer that already
        carries a random floor.

        Args:
            renderer: The camera's ``mujoco.Renderer``.
        """
        if self._floor is None:
            return
        tex, pixels = self._floor
        twin = copy.deepcopy(self._model)  # the shared model stays untouched; the copy lives only for the upload
        adr = int(twin.tex_adr[tex])
        twin.tex_data[adr:adr + pixels.size] = pixels.ravel()
        renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(twin, renderer._mjr_context, tex)
        renderer._realsensor_floor = True  # sharers would see this floor: SimCamera refuses to share such a renderer

    def apply(self, scn: mujoco.MjvScene, data: mujoco.MjData) -> None:
        """Model lights, colour, stains, walkers, distractors, lamps, in that order, on ``scn`` in place.

        Args:
            scn: The MjvScene to modify (after mjv_updateScene, before render).
            data: The rendered state (geom poses for stains, ``time`` for walkers).

        Raises:
            RuntimeError: ``scn`` has no free geom slot, or already holds ``MAX_RENDER_LIGHTS`` lights.
        """
        cfg = self.cfg
        if cfg.light_gain != 1.0 or cfg.light_tint != (1.0, 1.0, 1.0) or cfg.ambient != 0.0:
            gain, tint = cfg.light_gain, np.array(cfg.light_tint, np.float32)
            for i in range(scn.nlight):
                L = scn.lights[i]
                L.diffuse[:] = L.diffuse * gain * tint  # (diffuse * gain) * tint: keep this float order
                L.specular[:] = L.specular * gain * tint
                L.ambient[:] = L.ambient + cfg.ambient
        if self._color:
            for i in range(scn.ngeom):
                g = scn.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self._color:
                    g.rgba[:3] = np.clip(g.rgba[:3] * self._color[g.objid], 0, 1)
        for gid, pos, R, size, rgba in self._marks:
            X = data.geom_xmat[gid].reshape(3, 3)
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, data.geom_xpos[gid] + X @ pos, X @ R, rgba)
        for w in cfg.walkers:
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, (w.radius, w.height / 2 - w.radius, 0),
                      walker_pos(w, data.time), np.eye(3), np.array(w.rgba, np.float32))
        for gtype, size, pos, mat, rgba in self._distractors:
            _add_geom(scn, gtype, size, pos, mat, rgba)
        for lamp in cfg.lamps:
            _add_lamp(scn, lamp)


def _add_geom(scn, gtype, size, pos, mat, rgba):
    """Append one geom; ``mjv_initGeom`` leaves the slot's old identity, so it is re-tagged.

    ``segid`` = its scene index, as ``mjv_updateScene`` does (unique; depth rendering skips ``segid < 0``); objtype
    UNKNOWN, objid -1; category DYNAMIC (a stale DECOR category would drop the geom from the shadow pass).
    """
    if scn.ngeom >= scn.maxgeom:
        raise RuntimeError("MjvScene full")
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), pos, np.asarray(mat, np.float64).ravel(), rgba)
    g.segid, g.objtype, g.objid = scn.ngeom, mujoco.mjtObj.mjOBJ_UNKNOWN, -1
    g.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
    scn.ngeom += 1


def _add_lamp(scn, lamp):
    if scn.nlight >= min(MAX_RENDER_LIGHTS, len(scn.lights)):
        raise RuntimeError(f"MjvScene full: mjr_render uses only {MAX_RENDER_LIGHTS} lights")
    L = scn.lights[scn.nlight]
    L.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if lamp.directional else mujoco.mjtLightType.mjLIGHT_SPOT
    d = np.array(lamp.dir, np.float32)
    rgb = np.array(lamp.rgb, np.float32)
    L.pos[:], L.dir[:] = lamp.pos, d / (np.linalg.norm(d) + 1e-12)
    L.diffuse[:], L.specular[:], L.ambient[:], L.attenuation[:] = rgb, 0.1 * rgb, 0.0, (1, 0, 0)
    L.cutoff, L.exponent, L.castshadow = lamp.cutoff, lamp.exponent, int(lamp.castshadow)
    L.headlight, L.bulbradius, L.range, L.id, L.texid = 0, 0.02, 10, -1, -1
    scn.nlight += 1


_SCENE_SPREAD_KEYS = ("light_gain", "light_tint", "ambient", "color_jitter", "mark_size", "marks_per_geom",
                      "walker_phase", "walker_period", "lamp_rgb", "lamp_dir", "distractors")


def randomize_scene(cfg: SceneConfig, spread: dict, rng: np.random.Generator) -> SceneConfig:
    """A new scene drawn around ``cfg``: the scene twin of ``calib.randomize`` for DR training.

    Only keys in ``spread`` are perturbed, in the fixed order of ``_SCENE_SPREAD_KEYS`` (so a seed always gives the
    same scene); each draw is ``u ~ U(-1, 1)``. Relative: ``light_gain``, ``light_tint`` (per channel),
    ``color_jitter`` (clipped to 0.99), ``mark_size`` (>= 1e-3), ``marks_per_geom`` (rounded int), ``walker_period``
    (>= 0.1 s), ``lamp_rgb`` (per channel), ``distractors`` (rounded int >= 0); absolute: ``ambient`` (>= 0),
    ``walker_phase`` (mod 1), ``lamp_dir`` (per component of the unit direction, renormalised). Spreads are scalars.
    The ``seed`` (stains, colour factors, floor texture, distractor layout) is always redrawn last.

    Args:
        cfg: Scene to perturb (frozen; not mutated).
        spread: ``{key: scale}`` with keys from ``_SCENE_SPREAD_KEYS``.
        rng: Generator for every draw.

    Returns:
        The perturbed ``SceneConfig``.

    Raises:
        ValueError: On an unknown spread key.
    """
    bad = set(spread) - set(_SCENE_SPREAD_KEYS)
    if bad:
        raise ValueError(f"unknown scene spread keys {sorted(bad)}; known: {_SCENE_SPREAD_KEYS}")

    def rel(v, s):
        return v * (1 + s * rng.uniform(-1, 1))

    new, walkers, lamps = {}, cfg.walkers, cfg.lamps
    for k in _SCENE_SPREAD_KEYS:
        if k not in spread:
            continue
        s = spread[k]
        if k == "light_gain":
            new[k] = max(0.0, rel(cfg.light_gain, s))
        elif k == "light_tint":
            new[k] = tuple(max(0.0, rel(t, s)) for t in cfg.light_tint)
        elif k == "ambient":
            new[k] = max(0.0, cfg.ambient + s * rng.uniform(-1, 1))
        elif k == "color_jitter":
            new[k] = min(max(0.0, rel(cfg.color_jitter, s)), 0.99)
        elif k == "mark_size":
            new[k] = max(1e-3, rel(cfg.mark_size, s))
        elif k == "marks_per_geom":
            new[k] = max(0, int(round(rel(cfg.marks_per_geom, s))))
        elif k == "walker_phase":
            walkers = tuple(dataclasses.replace(w, phase=((w.phase + s * rng.uniform(-1, 1)) % 1.0) % 1.0)
                            for w in walkers)  # second % 1.0: (-1e-20) % 1.0 == 1.0
        elif k == "walker_period":
            walkers = tuple(dataclasses.replace(w, period_s=max(0.1, rel(w.period_s, s))) for w in walkers)
        elif k == "lamp_rgb":
            lamps = tuple(dataclasses.replace(lp, rgb=tuple(max(0.0, rel(c, s)) for c in lp.rgb)) for lp in lamps)
        elif k == "distractors":
            new[k] = max(0, int(round(rel(cfg.distractors, s))))
        else:  # lamp_dir
            lamps = tuple(dataclasses.replace(lp, dir=_perturbed_dir(lp.dir, s, rng)) for lp in lamps)
    return dataclasses.replace(cfg, **new, walkers=walkers, lamps=lamps, seed=int(rng.integers(2**31)))  # as calib


def _perturbed_dir(d, s, rng):
    """``d/|d| + s * u`` per component (3 draws, also at s = 0), renormalised; the old ``d`` at s = 0 or if degenerate.

    Normalising first makes the angular spread independent of how ``d`` was written ((0, 1, -1) vs its unit vector).
    """
    v = np.asarray(d, float)
    v = v / max(float(np.linalg.norm(v)), 1e-12) + s * rng.uniform(-1, 1, 3)
    n = float(np.linalg.norm(v))
    return tuple(d) if s == 0 or n < 1e-9 else tuple(float(x) for x in v / n)
