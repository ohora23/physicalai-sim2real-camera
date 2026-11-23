"""O7: run the observer on a recording and write the estimate series.

    python scripts/observe.py <recording> --config calib_out/camera_config.json --out est.json

The recording layout is in ``physicalai_sim2real_camera.observer_replay`` (``record`` writes it from the simulator; a
real camera recorder writes the same files, ``docs/capture_protocol.md`` D7). ``--config`` is the offline calibration
(``scripts/calibrate.py``): the observer takes its gamma, black level and rolling-shutter readout (D3 / D4 / D5b) and
nothing else; the exposure comes from the recording's metadata unless ``--exposure-s`` is given. The output JSON holds
the estimate at every ``--every`` seconds of arrival time and the final one after the stream ends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    """CLI entry point."""
    from observe_sim import _jsonable

    from physicalai_sim2real_camera import CameraConfig
    from physicalai_sim2real_camera.observer_replay import Observer, load_recording, replay
    from physicalai_sim2real_camera.target import PrismTarget
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("--config", required=True, help="offline camera_config.json (gamma, black_level, readout_s)")
    ap.add_argument("--target", default=os.path.join(ROOT, "assets", "prism", "prism.json"))
    ap.add_argument("--fovy", type=float, default=45.0, help="nominal vertical field of view for the initial K, deg")
    ap.add_argument("--exposure-s", type=float, default=None, help="default: the first frame's exposure metadata")
    ap.add_argument("--every", type=float, default=1.0, help="estimate period, s of arrival time")
    ap.add_argument("--seed", type=int, default=0, help="latency bootstrap seed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = CameraConfig.load(a.config)
    records, head = load_recording(a.recording)
    first = next(records, None)
    if first is None:
        sys.exit(f"{a.recording}: no frames")
    if not len(head.rows):
        sys.exit(f"{a.recording}: no head samples (head.npy is empty)")
    exposure = a.exposure_s if a.exposure_s is not None else first.exposure
    if exposure is None:
        sys.exit("no exposure metadata in the recording: pass --exposure-s")
    h, w = first.frame.shape[:2]
    obs = Observer(PrismTarget.from_json(a.target), (w, h), a.fovy, cfg.gamma, cfg.black_level, exposure_s=exposure,
                   readout_s=cfg.readout_s)

    def stream():
        yield first
        yield from records

    series = replay(stream(), obs, head, every_s=a.every, rng=np.random.default_rng(a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(_jsonable([dict(t=t, estimate=e) for t, e in series]), f, indent=1)
    g = series[-1][1]["geom"]
    print(f"{len(series)} estimates -> {a.out}" + ("" if g is None else f"; final fx {g['K'][0, 0]:.2f} "
                                                       f"cx {g['K'][0, 2]:.2f} cy {g['K'][1, 2]:.2f}"))


if __name__ == "__main__":
    main()
