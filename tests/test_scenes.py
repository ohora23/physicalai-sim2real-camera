"""HeadSweep / HeadPoseRing logic (no rendering); prism_scene itself is exercised in test_target with the assets."""

import numpy as np

from physicalai_sim2real_camera.scenes import STILLS, HeadPoseRing, HeadSweep


def test_sweep_phase_and_stills():
    s = HeadSweep([-1, 0, 0.2], 45.0, 640 / 480, target=(0.0, 0.0, 0.2))
    assert s.phase(0.5) == 0.5 and s.phase(1.5) == 1.0 and abs(s.phase(2.2) - 1.2) < 1e-12
    assert s.still(1.5) and not s.still(2.2)
    p0, q0 = s.pose(1.0)
    p1, q1 = s.pose(2.0)
    assert np.allclose(q0, q1) and np.allclose(p0, p1), "pose frozen during a still interval"
    assert np.linalg.norm(s.omega(1.5)) < 1e-9 and np.linalg.norm(s.omega(2.2)) > 0.05
    assert 0 < s.a_pitch < np.radians(22.5) and s.a_yaw > s.a_pitch  # yaw amplitude follows the wider horizontal FOV
    p_a, _ = s.pose(2.2)
    p_b, _ = s.pose(5.0)
    assert abs(np.hypot(p_a[0], p_a[1]) - 1.0) < 1e-9 and not np.allclose(p_a, p_b), "orbit keeps the radius, moves"
    assert len(STILLS) == 12 and all(abs((t1 - t0) - 1.0) < 1e-9 for t0, t1 in STILLS)


def test_head_pose_ring_roundtrip():
    ring = HeadPoseRing(n=16)
    try:
        other = HeadPoseRing(n=16, name=ring.name)
        for k in range(20):  # wraps once
            ring.write(k * 0.002, [k, 0, 0], [1, 0, 0, 0], [0, 0, 0.5 * k])
        r = other.read()
        assert r.shape == (15, 11) and r[0, 0] == 5 * 0.002 and r[-1, 1] == 19  # oldest slot left out after a wrap
        f = other.omega_fn()
        assert abs(f(10 * 0.002) - 5.0) < 1e-9 and abs(f(10.5 * 0.002) - 5.25) < 1e-9
        assert np.isnan(f(0.0)) and np.isnan(f(1.0)), "outside the record: nan, not the end value"
        other.close(unlink=False)
    finally:
        ring.close()


if __name__ == "__main__":
    test_sweep_phase_and_stills()
    test_head_pose_ring_roundtrip()
    print("scenes OK")
