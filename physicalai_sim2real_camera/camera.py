"""SimCamera: mujoco.Renderer wrapper + sim2real image pipeline."""

from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np

from .config import CameraConfig, intrinsics_from_model
from .depth import depth_buffers, depth_noise, zfar_of
from .lens import dirt_maps, flicker_rows, jpeg_roundtrip, veil
from .optics import distortion_maps, dof, dof_buffers, render_intrinsics, vignette_map
from .scene_dr import SceneDR
from .sensor import add_noise, ae_awb_step, bayer, encode, linearize, make_fpn
from .vibration import mount_vibration, shake_glcameras

# cv2's RNG is global/thread-local and the last setRNGSeed wins, so seed it
# once per process (first render), not per camera.
_RNG_SEEDED = False


class StateRing:
    """Circular buffer of recent sim states for renders inside the exposure window.

    Holds ``ceil(2*half/timestep) + 3`` samples (window ``[t - 2*half, t]``; pushes
    are at most once per timestep). ``push``/``load`` allocate nothing.
    Interpolation accuracy is limited by the push (control) rate.
    """

    def __init__(self, model, half):
        """Allocate ring buffers sized for a ``half``-second window of ``model``'s state.

        Args:
            model: mujoco.MjModel whose timestep and state layout size the ring.
            half: half-window length in seconds, ``(exposure_s + readout_s) / 2``.
        """
        self.model = model
        self.cap = cap = math.ceil(2 * half / model.opt.timestep) + 3
        self.time = np.empty(cap)
        self.fields = ["qpos", "qvel"]
        if model.nmocap > 0:
            self.fields += ["mocap_pos", "mocap_quat"]
        if model.na > 0:
            self.fields.append("act")
        # probe shapes from a throwaway MjData (once, at init)
        d = mujoco.MjData(model)
        self.buf = {f: np.empty((cap,) + getattr(d, f).shape) for f in self.fields}
        self._dq = np.empty(model.nv)
        self.n = 0  # samples stored
        self.head = 0  # next write slot

    def push(self, data):
        """Record ``data``'s state (qpos/qvel/... at ``data.time`` s) into the ring, one sample per call."""
        if self.n and data.time < self.time[(self.head - 1) % self.cap]:
            self.n = 0  # episode reset
        i = self.head
        self.time[i] = data.time
        for f in self.fields:
            self.buf[f][i] = getattr(data, f)
        self.head = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def _copy(self, i, scratch):
        for f in self.fields:
            getattr(scratch, f)[:] = self.buf[f][i]

    def load(self, t, scratch):
        """Set ``scratch`` to the state at time ``t`` and ``mj_forward`` it (never mj_step)."""
        if self.n == 0:
            raise RuntimeError("StateRing.load before any push")
        cap, T = self.cap, self.time
        # walk newest -> oldest to the first sample with time <= t
        j = (self.head - 1) % cap
        for k in range(self.n):
            i = (self.head - 1 - k) % cap
            if T[i] <= t + 1e-12:
                break
            j = i
        else:
            i = None  # t older than the ring: clamp to oldest (only the first frame of an episode)
        if i is None or abs(t - T[i]) < 1e-12 or i == j:
            self._copy(j if i is None else i, scratch)  # verbatim sample (also t beyond newest)
        else:
            m, B = self.model, self.buf
            a = (t - T[i]) / (T[j] - T[i])
            q = scratch.qpos
            q[:] = B["qpos"][i]
            # exact lerp for hinge/slide, geodesic for free/ball quaternions
            mujoco.mj_differentiatePos(m, self._dq, 1.0, B["qpos"][i], B["qpos"][j])
            mujoco.mj_integratePos(m, q, self._dq, a)
            for f in self.fields[1:]:
                dst = getattr(scratch, f)
                np.subtract(B[f][j], B[f][i], out=dst)
                dst *= a
                dst += B[f][i]
            if "mocap_quat" in B:
                # ponytail: lerp + normalise, assumes small rotation per push (nlerp ≈ slerp)
                for qk in scratch.mocap_quat:
                    qk /= math.sqrt(qk @ qk)
        scratch.time = t
        mujoco.mj_forward(self.model, scratch)


class SimCamera:
    """Render ``cam_name`` with distortion, vignetting, exposure, FPN and noise.

    Note: raises ``model.vis.global_.offwidth/offheight`` to fit the
    margin-FOV render (harmless model change). ``cam_fovy`` /
    ``cam_sensorsize`` are overridden only for the duration of each render
    and restored in ``finally``.

    Temporal noise uses cv2's global RNG, seeded once per process with the
    first-rendered camera's ``cfg.seed``; deterministic only when rendered from one
    calling thread in the same render order. FPN uses ``default_rng([seed + cam_id, 0])``.

    Temporal window: ``half_window = (exposure_s + readout_s)/2``. When > 0 the
    camera owns a ``StateRing`` fed by ``push(data)`` (``CameraStream`` calls it
    every step) and ``render_at(t)`` renders the interpolated state at ``t`` on a
    private scratch MjData (live ``data`` untouched). ``render(data)`` then renders
    the motion-blur / rolling-shutter window ending at ``data.time`` from the ring
    (``_render_window``); ``_render_data(data)`` is the single-instant render.
    Note: right after ``mj_step`` live ``xpos`` lags ``qpos`` by one step, so
    ``render_at(data.time)`` equals ``_render_data(data)`` only after ``mj_forward``.
    ``render_at`` runs the full pipeline, so it also advances AE state and cv2's RNG.

    Pipeline (each stage off at its default; all off == plain Renderer, bit-identical)::

        render uint8 (margin FOV) -> LUT linearise [-> window: sum/N per band]
         -> remap (output space; depth NEAREST)
         -> DoF(output-space depth)          aperture_m>0 & focus_m>0
         -> [Gaussian PSF sigma psf_sigma_px]  lens PSF / OLPF / pixel aperture
         -> [x ccm 3x3] [x flicker row gain] [x dirt transmission + veiling glare]   camera-side DR (CPU / cuda)
         -> x _mul (flat_field or cos^4 vignette, 2^exposure_ev, PRNU) x AE 2^(ev-exposure_ev)
            [x AWB wb]  + _add (DSNU, black)       ae_tau_s>0 / awb_tau_s>0; ev starts at exposure_ev
         -> meter cv2.mean(w)  (after gain: closed loop; before noise: deterministic)
         -> add_noise -> [bayer RGGB mosaic/demosaic] -> encode uint8 [-> MJPEG round trip (CPU only)]
         -> AE/AWB update for the NEXT frame
        depth out: remap NEAREST -> depth_noise (invalid -> 0; edge holes / flying pixels if depth_edge_hole_px > 0)

    DoF needs depth even for ``render(data)``: with DoF on, every frame costs one
    extra depth render (+1R, ~2 ms). ``flat_field`` replaces the cos^4 vignette.

    ``device="cuda"`` runs everything after the render on the GPU (``gpu.py``, needs
    torch); ``render(..., as_tensor=True)`` then returns CUDA tensors. The identity
    config always stays the plain CPU render. ``interop=True`` (with ``device="cuda"``)
    copies the colour render GPU->GPU via CUDA-GL interop (``interop.py``, needs
    ``cuda-python``) instead of readPixels + upload; it sets ``model.vis.quality.offsamples = 0``
    (MSAA off, for every Renderer made from this model afterwards). Depth goes GPU->GPU too
    (pixel-pack buffer), bit-exact with ``Renderer.render``.

    ``renderer=`` shares an existing ``mujoco.Renderer`` (e.g. ``other_cam.renderer``) instead of
    creating one: several cameras of one model then cost one GL context. It must come from the
    same model and be exactly ``render_scale * (H_r x W_r)`` (this camera's margin render size), and with interop
    have ``offSamples == 0``; otherwise ``ValueError`` (``offwidth/offheight``/``offsamples`` only
    act before a Renderer exists, so a sharing camera never changes ``model.vis``). Output is
    bit-identical to an own-renderer camera with the same config and seed. ``owns_renderer`` is
    False for a sharer and ``close()`` then leaves the renderer open; close sharers before the
    owner. With interop the sharers also share one ``GlInterop`` registration (``gpu.py``).
    """

    def __init__(self, model, cam_name: str, cfg: CameraConfig, device: str = "cpu",
                 interop: bool = False, renderer=None, scene=None):
        """Build the render + sim2real pipeline for one camera.

        Args:
            model: mujoco.MjModel containing camera ``cam_name``.
            cam_name: name of the MuJoCo camera to render.
            cfg: CameraConfig with intrinsics (px), distortion, sensor and noise parameters.
            device: "cpu" or "cuda"; see class docstring for the GPU pipeline.
            interop: enable CUDA-GL interop for the colour/depth readback; see class docstring.
            renderer: optional shared mujoco.Renderer; see class docstring. ``False`` creates no
                renderer at all (geometry, maps and LUTs only, ``render`` unusable): for a process
                that receives frames rendered elsewhere (``SimCameraBatch(renderer=False)``).
            scene: optional ``scene_dr.SceneConfig``: lights, extra lamps, walkers casting shadows, stains and
                colour jitter applied to the ``MjvScene`` of every render (model and physics untouched).
                ``floor_texture`` needs the camera's own renderer (and that renderer is not shareable afterwards);
                with ``renderer=False`` no texture is uploaded: the process that renders must do it.
        """
        shared = renderer is not None and renderer is not False
        if shared and scene is not None and scene.floor_texture:
            raise ValueError("floor_texture uploads into the camera's own GL context: use an unshared renderer")
        if shared and getattr(renderer, "_realsensor_floor", False):
            raise ValueError("this renderer carries another camera's random floor texture: do not share it")
        self.model, self.cfg = model, cfg
        self.scene = None if scene is None else SceneDR(model, scene)
        self.cam = cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam < 0:
            raise ValueError(f"camera {cam_name!r} not found")
        W, H = cfg.width, cfg.height

        K = intrinsics_from_model(model, cam, W, H)
        for (i, j), v in {(0, 0): cfg.fx, (1, 1): cfg.fy, (0, 2): cfg.cx, (1, 2): cfg.cy}.items():
            if v is not None:
                K[i, j] = v
        self.K_out = K
        self.K_render, W_r, H_r, self.fovy_r = render_intrinsics(K, cfg.dist, W, H)

        self.maps = (None if (W_r, H_r) == (W, H) and np.array_equal(self.K_render, K)
                     else distortion_maps(K, cfg.dist, W, H, self.K_render))
        if cfg.flat_field is not None:  # measured map (float32 (H,W,1), checked in config) replaces cos^4
            self.vignette = cfg.flat_field
        else:
            self.vignette = vignette_map(K, cfg.dist, W, H, cfg.vignette) if cfg.vignette > 0 else None
        s = cfg.seed + cam  # per-camera seed: stereo pairs get uncorrelated FPN
        self.fpn = (make_fpn(H, W, cfg.sigma_prnu, cfg.sigma_dsnu, np.random.default_rng([s, 0]))
                    if cfg.sigma_prnu or cfg.sigma_dsnu else None)
        self.gain = np.float32(2.0 ** cfg.exposure_ev)
        # camera-side DR: colour matrix, mains flicker, lens dirt, MJPEG
        self._ccm = None if cfg.ccm is None else np.asarray(cfg.ccm, np.float32).reshape(3, 3)
        self._dirt = None
        if cfg.dirt_density > 0:
            T, D = dirt_maps(H, W, cfg.dirt_density, np.random.default_rng([s, 3]))
            # veil weight pre-scaled (scatter * D) and an output buffer: no per-frame allocation in veil()
            self._dirt = (np.repeat(T, 3, axis=2), cfg.dirt_scatter * D if cfg.dirt_scatter > 0 else None)
            self._glare = np.empty((H, W, 3), np.float32) if cfg.dirt_scatter > 0 else None
        self.lens_on = bool(cfg.lens_fields())
        if device != "cpu" and cfg.jpeg_quality > 0:  # ponytail: imencode needs host pixels; GPU nvjpeg if needed
            raise ValueError("jpeg_quality runs on the CPU pipeline only (device='cpu')")
        self.lut = linearize(np.arange(256, dtype=np.uint8), cfg.gamma)  # reuse sensor's gamma LUT

        self.identity = (self.maps is None and self.vignette is None and cfg.exposure_ev == 0
                         and self.fpn is None and cfg.sigma_shot == 0 and cfg.sigma_read == 0
                         and cfg.black_level == 0 and (W_r, H_r) == (W, H)
                         and np.array_equal(self.K_render, K)
                         and cfg.exposure_s == 0 and cfg.readout_s == 0
                         and cfg.aperture_m == 0 and cfg.ae_tau_s == 0 and cfg.awb_tau_s == 0
                         and not cfg.bayer and cfg.depth_sigma_a == 0 and cfg.depth_sigma_b == 0
                         and cfg.depth_min_m == 0 and cfg.depth_max_m == 0 and cfg.depth_edge_hole_px == 0
                         and not self.lens_on and cfg.render_scale == 1)

        self.f_px = float(K[0, 0])
        self._vib = mount_vibration(cfg, cfg.seed, cam)  # None when off
        self._dof = dof_buffers(H, W) if cfg.aperture_m > 0 and cfg.focus_m > 0 else None
        self._bayer = ((np.empty((H, W), np.uint16), np.empty((H, W, 3), np.uint16))
                       if cfg.bayer else None)
        self._drng = None
        if cfg.depth_sigma_a or cfg.depth_sigma_b or cfg.depth_min_m or cfg.depth_max_m or cfg.depth_edge_hole_px:
            self._drng = np.random.default_rng([s, 2])
            self._dbufs = depth_buffers(H, W, edge=cfg.depth_edge_hole_px > 0)
        self._zfar = zfar_of(model)  # background depth: DoF treats it as in focus, depth_noise -> 0
        self._ae = cfg.ae_tau_s > 0 or cfg.awb_tau_s > 0
        self._ev, self._wb = cfg.exposure_ev, (1.0, 1.0, 1.0)  # ev starts at exposure_ev: no double gain

        self._rgb = np.empty((H_r, W_r, 3), np.uint8)
        # supersampling: the GL render is s x larger; _raw area-averages it in linear light into _rgb (re-encoded, so
        # the window / remap / GPU paths downstream are unchanged). ponytail: 8-bit re-encode, +-0.5 LSB
        sc = self.render_scale = cfg.render_scale
        Hb, Wb = H_r * sc, W_r * sc
        if sc > 1:
            self._big = np.empty((Hb, Wb, 3), np.uint8)
            self._big_lin = np.empty((Hb, Wb, 3), np.float32)
            self._big_d = np.empty((Hb, Wb), np.float32)
            self._small_lin = np.empty((H_r, W_r, 3), np.float32)
        self._lin = np.empty((H_r, W_r, 3), np.float32)
        self._work = self._lin if self.maps is None else np.empty((H, W, 3), np.float32)
        self._noise = np.empty((H, W * 3), np.float32)
        # ponytail: vignette·2^ev·PRNU folded into one gain map and DSNU+black into one
        # offset (same math as the per-step order); contiguous (H,W,3) — (H,W,1) broadcasts cost 0.3–0.4 ms each
        one = np.ones((H, W, 1), np.float32)
        v = one if self.vignette is None else self.vignette
        g, o = (one, 0 * one) if self.fpn is None else self.fpn
        self._mul = np.repeat(v * self.gain * g, 3, axis=2)
        self._add = np.repeat(o + np.float32(cfg.black_level), 3, axis=2)

        interop = interop and device != "cpu" and not self.identity and renderer is not False
        if interop and sc > 1:
            raise ValueError("render_scale > 1 needs the readPixels path: use interop=False")
        self.owns_renderer = renderer is None
        if renderer is False:  # post-processing only (SimCameraBatch ingest path): no GL context
            self.renderer = None
        elif renderer is not None:
            if renderer.model is not model:
                raise ValueError("shared renderer was made from a different model")
            if (renderer.height, renderer.width) != (Hb, Wb):
                raise ValueError(f"shared renderer is {renderer.height}x{renderer.width}, "
                                 f"camera {cam_name!r} needs render_scale * (H_r x W_r) = {Hb}x{Wb}")
            if interop and renderer._mjr_context.offSamples != 0:
                raise ValueError(f"interop needs a single-sample renderer (offSamples="
                                 f"{renderer._mjr_context.offSamples}); create the owner with interop=True "
                                 f"or set model.vis.quality.offsamples = 0 before it")
            self.renderer = renderer
        else:
            g = model.vis.global_
            g.offwidth, g.offheight = max(g.offwidth, Wb), max(g.offheight, Hb)
            if interop and model.vis.quality.offsamples != 0:
                print("WARNING: interop=True disables MSAA (model.vis.quality.offsamples = 0)")
                model.vis.quality.offsamples = 0
            self.renderer = mujoco.Renderer(model, Hb, Wb)
        if self.scene is not None and scene.shadow_clip > 0 and self.renderer is not None:
            self.renderer._mjr_context.shadowClip = scene.shadow_clip  # shared renderer: its sharers get it too
        if self.scene is not None and self.renderer is not None:
            self.scene.bind(self.renderer)

        self.half_window = (cfg.exposure_s + cfg.readout_s) / 2
        self._ring = self._scratch = None
        if self.half_window > 0:
            self._ring = StateRing(model, self.half_window)
            self._scratch = mujoco.MjData(model)
            N = self._N = cfg.blur_samples if cfg.exposure_s > 0 else 1
            B = self._B = cfg.rs_bands if cfg.readout_s > 0 else 1
            if self.maps is None:  # _work aliases _lin, which every sub-render overwrites
                self._work = np.empty((H, W, 3), np.float32)
            self._plan_window(N, B, H_r)
            n_r = len(self._plan)
            if n_r > 8:
                print(f"WARNING: {n_r} renders/frame (blur_samples*rs_bands = {N * B}, grid-snapped; "
                      f"RL budget <= 8 renders, see docs/perf.md)")

        self.device, self._dbuf, self._gpu = device, None, None
        if device != "cpu" and not self.identity:
            from .gpu import GpuPipeline
            self._gpu = GpuPipeline(self, device, interop)

    def _plan_window(self, N, B, H_r):
        """Grid-snap render cache: which renders feed which band (init only).

        Sample ``t_bi`` (see ``_render_window``) is snapped to the grid
        ``delta = min(exposure_s/N, readout_s/B)`` (non-zero terms) anchored at
        the first sample, and each distinct grid time is rendered once, in time
        order, shared by every band/sample that snaps to it (N=4, B=8 with
        exposure = readout/2: 32 -> 11 renders). Time error per sample <= delta/2
        (0 when exposure/N and readout/B are multiples of delta); the actual max
        is ``self._snap_err``. Each band accumulates only the render rows its
        remap reads (``_rows``), so memory is ~one frame and there is no cache cap.
        """
        cfg, H = self.cfg, self.cfg.height
        e, r = cfg.exposure_s, cfg.readout_s
        delta = min(x for x in (e / N if e > 0 else math.inf, r / B if r > 0 else math.inf))
        off = [[r * ((b + 0.5) / B - 0.5) + e * ((i + 0.5) / N - 0.5) for i in range(N)]
               for b in range(B)]
        o0 = off[0][0]
        use = {}  # grid index -> [(band, first, last)] in (b, i) order
        self._snap_err = 0.0
        for b in range(B):
            for i in range(N):
                k = round((off[b][i] - o0) / delta)
                self._snap_err = max(self._snap_err, abs(o0 + k * delta - off[b][i]))
                use.setdefault(k, []).append((b, i == 0, i == N - 1))
        self._plan = [(o0 + k * delta, use[k]) for k in sorted(use)]
        # per band: source rows [s0, s1) its remap reads, a row accumulator, maps shifted to s0
        self._rows, self._bacc, self._bmaps = [], [], []
        for b in range(B):
            r0, r1 = b * H // B, (b + 1) * H // B
            if self.maps is None:
                s0, s1 = r0, r1
                self._bacc.append(self._work[r0:r1])  # accumulate straight into the output rows
                self._bmaps.append(None)
            else:
                my = self.maps[1][r0:r1]
                s0 = min(max(0, math.floor(float(my.min()))), H_r - 1)
                s1 = min(H_r, max(s0 + 1, math.floor(float(my.max())) + 2))  # +1 bilinear neighbour
                self._bacc.append(np.empty((s1 - s0,) + self._lin.shape[1:], np.float32))
                self._bmaps.append((self.maps[0][r0:r1], my - np.float32(s0)))  # exact: integer shift
            self._rows.append((r0, r1, s0, s1))

    def close(self):
        """Release the GPU pipeline (if any) and close the renderer, if this camera owns it."""
        try:
            if self._gpu is not None:
                self._gpu.release()
        finally:
            if self.owns_renderer:
                self.renderer.close()

    def push(self, data):
        """Feed ``data``'s state into the state ring."""
        if self._ring is not None:
            self._ring.push(data)

    def reset_ring(self):
        """Episode reset: forget the ring's samples; the next ``render`` is a single-instant render of its data."""
        if self._ring is not None:
            self._ring.n = 0

    def render_at(self, t, depth=False, as_tensor=False):
        """Render the ring's state at sim time ``t`` (requires half_window > 0)."""
        self._ring.load(t, self._scratch)
        return self._render_data(self._scratch, depth, as_tensor, t)

    def render(self, data, depth=False, as_tensor=False):
        """Render the frame ending at ``data.time`` s.

        Runs the windowed (motion-blur/rolling-shutter) render when ``half_window > 0``
        and the ring already holds samples, otherwise a single-instant render of ``data``.

        Args:
            data: mujoco.MjData with the current state (only ``data.time`` is used once
                the ring holds samples; otherwise this state is rendered directly).
            depth: also return an output-space depth map in metres.
            as_tensor: return CUDA tensors instead of numpy arrays (needs ``device="cuda"``).
        """
        if as_tensor and self.device == "cpu":
            raise ValueError("as_tensor=True needs device='cuda'")
        if self._ring is None or self._ring.n == 0:  # empty ring: nothing to integrate yet
            # same mid-exposure time as the window path, so a flicker phase does not jump on the first frame
            return self._render_data(data, depth, as_tensor, data.time - self.half_window)
        return self._render_window(data, depth, as_tensor)

    def _render_window(self, data, depth, as_tensor=False):
        """Render the window ending at ``data.time`` from the ring (call after ``push(data)``).

        ``capture_t = data.time - half``; band b (output rows, top first) averages N
        linear sub-renders at ``t_bi = capture_t + readout_s*((b+.5)/B - .5)
        + exposure_s*((i+.5)/N - .5)`` (grid-snapped, see ``_plan_window``: each distinct
        time rendered once), then remaps only its own rows. Depth is one
        render at ``capture_t``. Blur averages already uint8-clipped renders (same
        no-HDR limit as Phase 1).
        """
        if self._gpu is not None:
            return self._gpu.window(data, depth, as_tensor)
        tc = data.time - self.half_window
        sc, lin, inv_n = self._scratch, self._lin, np.float32(1 / self._N)
        for t, uses in self._plan:  # distinct grid times, in time order
            self._ring.load(tc + t, sc)
            rgb, _ = self._raw(sc, False)
            for b, first, last in uses:
                r0, r1, s0, s1 = self._rows[b]
                acc = self._bacc[b]
                if first:  # first sample goes straight into the band accumulator (no fill)
                    cv2.LUT(rgb[s0:s1], self.lut, dst=acc)
                else:
                    tmp = lin[:s1 - s0]
                    cv2.LUT(rgb[s0:s1], self.lut, dst=tmp)
                    acc += tmp
                if last:
                    if self._N > 1:
                        acc *= inv_n
                    if self.maps is not None:
                        cv2.remap(acc, *self._bmaps[b], cv2.INTER_LINEAR, dst=self._work[r0:r1])
        d = None
        if depth or self._dof is not None:
            self._ring.load(tc, sc)
            _, d = self._raw(sc, True, rgb=False)
            if self.maps is not None:
                d = cv2.remap(d, *self.maps, cv2.INTER_NEAREST)
        return self._finish(self._work, d, depth, tc)

    def _raw(self, data, depth, rgb=True):
        """Render rgb (into ``self._rgb`` unless identity; skipped if not ``rgb``) and optional depth."""
        global _RNG_SEEDED
        if not _RNG_SEEDED:
            cv2.setRNGSeed(self.cfg.seed)
            _RNG_SEEDED = True

        m, cam, r = self.model, self.cam, self.renderer
        if r is None:
            raise ValueError("this SimCamera has no renderer (renderer=False): it cannot render")
        fovy0, ss0 = float(m.cam_fovy[cam]), m.cam_sensorsize[cam].copy()
        d = None
        try:
            m.cam_sensorsize[cam, :] = 0  # sensorsize≠0 cameras ignore cam_fovy
            m.cam_fovy[cam] = self.fovy_r
            r.update_scene(data, camera=cam)  # fovy is read here, not at render()
            if self.scene is not None:  # every sub-render of a window sees the scene at its own data.time
                self.scene.apply(r.scene, data)
            if self._vib is not None:  # the mount pose at this sub-render's time: blur and rolling shutter follow it
                shake_glcameras(r.scene, self._vib.matrix(data.time))
            if rgb and self._gpu is not None and self._gpu.interop is not None:
                rgb = self._gpu.interop.render_into(self._gpu._u8)
            elif rgb and self.render_scale > 1:  # pixel-area integration in linear light, then back to 8 bit
                cv2.LUT(r.render(out=self._big), self.lut, dst=self._big_lin)
                cv2.resize(self._big_lin, self._rgb.shape[1::-1], dst=self._small_lin, interpolation=cv2.INTER_AREA)
                np.copyto(self._rgb, encode(self._small_lin, self.cfg.gamma))
                rgb = self._rgb
            elif rgb:
                rgb = r.render() if self.identity else r.render(out=self._rgb)
            if depth and self._gpu is not None and self._gpu.interop is not None:
                d = self._gpu.interop.depth_into(self._gpu._draw)
            elif depth:
                r.enable_depth_rendering()
                if self.render_scale > 1:  # depth must not be averaged across edges (no in-between surfaces)
                    s, k = self.render_scale, self.render_scale // 2
                    big = r.render(out=self._big_d)
                    if s % 2:  # odd: the block's centre sample
                        sub = big[k::s, k::s]
                    else:  # even: the nearest of the 4 samples around the centre (symmetric, no 1/2s px shift)
                        sub = np.minimum(np.minimum(big[k - 1::s, k - 1::s], big[k - 1::s, k::s]),
                                         np.minimum(big[k::s, k - 1::s], big[k::s, k::s]))
                    if self._dbuf is None:
                        d = np.ascontiguousarray(sub)
                    else:
                        np.copyto(self._dbuf, sub)
                        d = self._dbuf
                else:
                    d = r.render(out=self._dbuf)  # pinned buffer on the GPU path, fresh otherwise
        finally:
            r.disable_depth_rendering()
            m.cam_fovy[cam] = fovy0
            m.cam_sensorsize[cam] = ss0
        return rgb, d

    def _render_data(self, data, depth, as_tensor=False, t=None):
        rgb, d = self._raw(data, depth or self._dof is not None)
        if self.identity:
            if as_tensor:  # plain render, uploaded as-is
                import torch
                rgb, d = (None if x is None else torch.from_numpy(x).to(self.device) for x in (rgb, d))
            return rgb if d is None else (rgb, d)
        if self._gpu is not None:
            return self._gpu.frame(depth, as_tensor, data.time if t is None else t)
        # windowed cameras keep a separate _work even without remap: single-instant renders then finish in _lin
        w = self._lin if self.maps is None else self._work
        cv2.LUT(rgb, self.lut, dst=self._lin)
        if self.maps is not None:
            cv2.remap(self._lin, *self.maps, cv2.INTER_LINEAR, dst=w)
            if d is not None:
                d = cv2.remap(d, *self.maps, cv2.INTER_NEAREST)
        return self._finish(w, d, depth, data.time if t is None else t)

    def _finish(self, w, d, depth, t):
        """Output-space linear ``w`` (+ depth ``d``) -> uint8 [, depth]; ``t`` = mid-exposure of the centre row."""
        cfg = self.cfg
        if self._dof is not None:
            dof(w, d, self.f_px, cfg.focus_m, cfg.aperture_m, self._dof, self._zfar)
        if cfg.psf_sigma_px > 0:  # lens PSF + OLPF + pixel aperture on scene light, before row timing and noise
            cv2.GaussianBlur(w, (0, 0), cfg.psf_sigma_px, dst=w, borderType=cv2.BORDER_REFLECT)
        if self._ccm is not None:  # sensor colour response, on scene light (before every gain)
            np.copyto(w, cv2.transform(w, self._ccm))
        if cfg.flicker_amp > 0:  # exposure-averaged mains illumination per row (rolling shutter: bands)
            w *= flicker_rows(w.shape[0], t, cfg.exposure_s, cfg.readout_s, cfg.flicker_amp, cfg.flicker_hz,
                              cfg.flicker_phase)[:, None, None]
        if self._dirt is not None:  # lens dirt: veil from the incoming light, then transmission
            T3, D = self._dirt
            glare = None if D is None else veil(w, D, 1.0, out=self._glare)
            cv2.multiply(w, T3, dst=w)
            if glare is not None:
                w += glare
        # AE gain 2^(ev − exposure_ev) rides the _mul multiply as cv2's scale (1.0 when AE off: bit-identical,
        # ~0 ms vs 0.7 ms for a separate pass). ponytail: no HDR headroom — render is already uint8-clipped
        cv2.multiply(w, self._mul, dst=w, scale=2.0 ** (self._ev - cfg.exposure_ev))
        if cfg.awb_tau_s > 0:
            cv2.multiply(w, (*self._wb, 0.0), dst=w)  # per-channel Scalar (0.4 ms; a (3,) broadcast is 0.8)
        w += self._add
        if self._ae:
            means = cv2.mean(w)[:3]
        add_noise(w, cfg.sigma_shot, cfg.sigma_read, self._noise)
        if self._bayer is not None:
            bayer(w, *self._bayer)
        rgb = encode(w, cfg.gamma)
        if cfg.jpeg_quality:
            rgb = jpeg_roundtrip(rgb, cfg.jpeg_quality)
        if self._ae:  # for the NEXT frame; dropped frames never render, so never update
            self._ev, self._wb = ae_awb_step(self._ev, self._wb, means, cfg)
        if not depth:
            return rgb
        if self._drng is not None:
            depth_noise(d, cfg, self._drng, self._dbufs, self._zfar)
        return rgb, d
