"""MuJoCo real-sensor plugin."""

from .camera import SimCamera
from .config import CameraConfig, intrinsics_from_model, perturb_extrinsics, set_extrinsics
from .timing import CameraStream, CameraStreamBatch, latest_ready, ring_depth, schedule, stagger_phases

# ``SimCameraBatch`` is deliberately NOT in ``__all__``: a star-import would load torch, which
# GL-only (sim) processes must not do. Import it by name: ``from physicalai_sim2real_camera import SimCameraBatch``.
__all__ = [
    "CameraConfig",
    "CameraStream",
    "CameraStreamBatch",
    "SimCamera",
    "intrinsics_from_model",
    "latest_ready",
    "perturb_extrinsics",
    "ring_depth",
    "schedule",
    "set_extrinsics",
    "stagger_phases",
]


def __getattr__(name):
    """Lazy ``SimCameraBatch`` export: ``batch`` imports torch, which GL-only (sim) processes must not load."""
    if name == "SimCameraBatch":
        from .batch import SimCameraBatch

        globals()[name] = SimCameraBatch  # resolve once
        return SimCameraBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Keep ``dir()`` consistent with the lazy export."""
    return sorted(set(globals()) | {"SimCameraBatch"})
