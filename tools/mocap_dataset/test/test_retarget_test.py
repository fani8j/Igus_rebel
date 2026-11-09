"""Joint-1 retarget math of retarget_test.py: identity at d_theta = 0, FK(q_new) = Rz(d) FK(q), ramp and checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retarget_test as rt  # noqa: E402

pytestmark = pytest.mark.skipif(not rt.URDF.is_file(), reason="URDF scan_results file missing")

FPS = 30.0


@pytest.fixture(scope="module")
def chain():
    return rt.Chain()


def _traj(n=400):
    t = np.arange(n) / FPS
    base = np.radians([5, 60, 40, -30, 20, 100])
    amp = np.radians([10, 20, 15, 30, 10, 40])
    return base + amp * np.sin(0.4 * t[:, None] + np.arange(6))


def test_zero_delta_is_identity():
    q = _traj()
    w = rt.ramp_weights(len(q), 100, 250)
    q0 = rt.retarget_q(q, 0.0, w)
    assert np.array_equal(q0, q)
    assert np.array_equal(rt.next_state_action(q0), rt.next_state_action(q))


def test_ramp_weights_shape():
    w = rt.ramp_weights(400, 100, 250)
    assert w[:100].max() == 0.0 and w[250:].min() == 1.0
    assert np.all(np.diff(w) >= -1e-12)
    # min-jerk: zero slope at both ends, midpoint 0.5
    assert w[175] == pytest.approx(0.5, abs=0.02)
    assert (w[101] - w[100]) < 1e-3 and (w[250] - w[249]) < 1e-3


@pytest.mark.parametrize("deg", [-4.0, 4.0, 7.5])
def test_fk_after_ramp_is_source_rotated_about_base_z(chain, deg):
    q = _traj()
    d = np.radians(deg)
    w = rt.ramp_weights(len(q), 100, 250)
    qn = rt.retarget_q(q, d, w)
    assert np.array_equal(qn[:, 1:], q[:, 1:])                       # only joint 1 changes
    for k in (250, 300, 399):                                         # w == 1
        T0, T1 = chain.fk(q[k]), chain.fk(qn[k])
        Rz = rt.rotz(d)
        assert np.allclose(T1[:3, 3], Rz @ T0[:3, 3], atol=1e-9)
        assert np.allclose(T1[:3, :3], Rz @ T0[:3, :3], atol=1e-9)
    for k in (0, 50, 100):                                            # w == 0: untouched
        assert np.allclose(chain.fk(qn[k]), chain.fk(q[k]), atol=1e-12)
    k = 175                                                           # inside the ramp: rotated by d * w
    T0, T1 = chain.fk(q[k]), chain.fk(qn[k])
    assert np.allclose(T1[:3, 3], rt.rotz(d * w[k]) @ T0[:3, 3], atol=1e-9)


def test_checks_flag_violations(chain):
    q = _traj()
    ok = rt.check_q(q, chain)
    assert ok["limits_ok"] and ok["speed_ok"] and ok["step_ok"]
    bad = q.copy()
    bad[10, 0] = chain.hi[0]                                          # inside the 2 deg inset
    assert not rt.check_q(bad, chain)["limits_ok"]
    jump = q.copy()
    jump[50:, 2] += np.radians(8)
    assert not rt.check_q(jump, chain)["step_ok"]


def test_wrist_homography_identity_and_rotation(chain):
    T = chain.fk(np.radians([5, 60, 40, -30, 20, 100]))
    assert np.allclose(rt.wrist_homography(T, 0.0), np.eye(3), atol=1e-9)
    Hm = rt.wrist_homography(T, np.radians(4))
    # a rotation homography keeps the principal point within a few tens of px for 4 degrees
    p = Hm @ np.array([320.0, 240.0, 1.0])
    p = p[:2] / p[2]
    assert np.linalg.norm(p - [320, 240]) < 0.08 * 460 * 1.5
    assert np.linalg.det(Hm / Hm[2, 2]) == pytest.approx(1.0, abs=0.2)


def test_head_camera_pose_is_sane():
    T = rt.T_base_cam_optical()
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    assert 0.5 < T[2, 3] < 1.2                                         # camera ~0.87 m above the base
