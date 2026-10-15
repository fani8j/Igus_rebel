"""Orbit / roll / range pose generation and the in-view prediction (wrist_poses.py)."""
import math

import numpy as np
import pytest

from rebel_dual_arm_calibration import wrist_poses as wp
from rebel_dual_arm_calibration import wrist_synthetic as W
from rebel_dual_arm_calibration.transforms import inv_T, make_T, rotvec_to_R, xyz_rpy_to_T

K, DIST, SIZE = W.WRIST_K, W.WRIST_DIST, W.WRIST_SIZE
BOARD = (0.199, 0.160)
X_TRUE = xyz_rpy_to_T([-0.02, -0.09, -0.04], np.radians([10, -5, 95]))
# tilted board like the earlier real reading (normal 15 deg off vertical): board z INTO the table
B_TILT = xyz_rpy_to_T([0.59, -0.154, -0.186], np.radians([180 - 15, 0.0, 10.0]))


def seed_pose(B, X, dist=0.38, tilt_deg=10.0):
    """link6 pose whose camera looks at the board centre from ``dist`` along (almost) the board normal."""
    n_out = -B[:3, 2]
    ax = np.cross(n_out, [1.0, 0, 0])
    eye = B[:3, 3] + dist * (rotvec_to_R(ax / np.linalg.norm(ax) * math.radians(tilt_deg)) @ n_out)
    cam = W.look_at(eye, B[:3, 3], B[:3, 1])
    return cam @ inv_T(X)


OPT = wp.PoseOptions()


def gen(A0, B=B_TILT, X=X_TRUE, opt=OPT):
    return wp.generate_poses(A0, X, B, K, DIST, SIZE, BOARD, opt)


def test_nominal_guess_geometry():
    T = wp.nominal_T_link6_optical()
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9) and np.linalg.det(T[:3, :3]) > 0
    # optical +z points along the CAD front-face normal, 18 deg off the tool axis; the tool z in link6 is +z (yaw pi about z only)
    z_link6 = T[:3, 2]
    assert z_link6[2] == pytest.approx(0.9511, abs=1e-3)
    assert np.linalg.norm(T[:3, 3]) < 0.2        # camera is within 20 cm of the flange
    # roll parameter only rotates about the optical axis
    T90 = wp.nominal_T_link6_optical(90.0)
    assert np.allclose(T90[:3, 2], T[:3, 2], atol=1e-9) and np.allclose(T90[:3, 3], T[:3, 3])
    assert not np.allclose(T90[:3, 0], T[:3, 0])


def test_orbit_keeps_board_centre_fixed_in_the_camera_for_any_mount():
    A0 = seed_pose(B_TILT, X_TRUE)
    for X_wrong in (X_TRUE, wp.nominal_T_link6_optical(), X_TRUE @ make_T(rotvec_to_R([0.1, -0.05, 0.2]), [0.01, 0, 0.02])):
        A0w = seed_pose(B_TILT, X_wrong)                  # the operator centred the board; X_wrong is the ASSUMED mount
        cands, _ = gen(A0w, X=X_wrong)
        assert len(cands) > 10
        c0 = inv_T(A0w @ X_wrong) @ np.append(B_TILT[:3, 3], 1)
        for c in cands:
            if c.dist_factor != 1.0:
                continue
            cc = inv_T(c.T_base_link6 @ X_wrong) @ np.append(B_TILT[:3, 3], 1)
            # rigid rotation about the centre / the line to it: the centre keeps its camera coordinates (orbit and roll)
            assert np.allclose(cc[:3], c0[:3], atol=1e-6), c.desc


def test_distance_limits_and_range_factors():
    A0 = seed_pose(B_TILT, X_TRUE, dist=0.40)
    cands, rej = gen(A0)
    d0 = 0.40
    for c in cands:
        d = c.info["distance_m"]
        assert d >= OPT.min_dist_m - 1e-9 and d <= OPT.max_dist_m + 1e-9
        assert d == pytest.approx(d0 * c.dist_factor, abs=2e-3)
    assert {c.dist_factor for c in cands} >= {0.8, 1.0, 1.2}
    # closer than 0.25 m is rejected and counted
    A_close = seed_pose(B_TILT, X_TRUE, dist=0.27)
    cands, rej = gen(A_close)
    assert all(c.info["distance_m"] >= 0.25 - 1e-9 for c in cands)
    assert rej.get("too close", 0) > 0


def test_in_view_check_rejects_board_outside_margin_and_behind():
    A0 = seed_pose(B_TILT, X_TRUE)
    ok, why, info = wp.view_check(A0, X_TRUE, B_TILT, K, DIST, SIZE, BOARD, OPT)
    assert ok, why
    assert info["margin_px"] > 0 and len(info["outline_px"]) == 4
    # shift the camera sideways so the board leaves the image margin
    A_off = A0.copy()
    A_off[:3, 3] += (A0[:3, :3] @ X_TRUE[:3, :3])[:, 0] * 0.25
    assert not wp.view_check(A_off, X_TRUE, B_TILT, K, DIST, SIZE, BOARD, OPT)[0]
    # camera below the board plane / looking away
    A_flip = A0 @ make_T(rotvec_to_R([math.pi, 0, 0]), [0, 0, 0])
    assert not wp.view_check(A_flip, X_TRUE, B_TILT, K, DIST, SIZE, BOARD, OPT)[0]


def test_every_candidate_passes_the_in_view_check_with_a_margin():
    A0 = seed_pose(B_TILT, X_TRUE)
    cands, rej = gen(A0)
    assert len(cands) >= 20
    for c in cands:
        ok, why, info = wp.view_check(c.T_base_link6, X_TRUE, B_TILT, K, DIST, SIZE, BOARD, OPT)
        assert ok, why
        uv = np.array(info["outline_px"])
        assert uv[:, 0].min() >= 0.1 * SIZE[0] and uv[:, 0].max() <= 0.9 * SIZE[0]
        assert uv[:, 1].min() >= 0.1 * SIZE[1] and uv[:, 1].max() <= 0.9 * SIZE[1]


def test_candidates_cover_tilts_azimuths_and_rolls_and_use_the_real_board_normal():
    A0 = seed_pose(B_TILT, X_TRUE)
    cands, _ = gen(A0, opt=wp.PoseOptions(max_diverse=400))
    assert {c.roll_deg for c in cands} >= {-45.0, 0.0, 45.0}
    assert len({c.az_deg for c in cands if c.tilt_deg}) >= 6
    n_real = -B_TILT[:3, 2]
    assert math.degrees(math.acos(n_real[2])) == pytest.approx(15.0, abs=0.5)      # the board is tilted
    # all camera positions are on the board-normal side of the board plane
    for c in cands:
        o = (c.T_base_link6 @ X_TRUE)[:3, 3]
        assert float(n_real @ (o - B_TILT[:3, 3])) > 0
    # diversity order: first = the seed view, and the rotation span of the first 12 is large
    assert cands[0].tilt_deg == 0 and cands[0].roll_deg == 0 and cands[0].dist_factor == 1.0
    R = [(c.T_base_link6 @ X_TRUE)[:3, :3] for c in cands[:12]]
    span = max(math.degrees(math.acos(max(-1, min(1, (np.trace(a.T @ b) - 1) / 2)))) for a in R for b in R)
    assert span > 40


def test_orbit_axes_lie_in_the_board_plane():
    A0 = seed_pose(B_TILT, X_TRUE)
    c = B_TILT[:3, 3]
    bx, by, n = wp.board_axes(B_TILT)
    for az in (0, 90, 215):
        axis = math.cos(math.radians(az)) * bx + math.sin(math.radians(az)) * by
        assert abs(float(axis @ n)) < 1e-9
        M = wp.rotate_about(c, axis, math.radians(20))
        assert np.allclose(M @ np.append(c, 1), np.append(c, 1))                  # centre is a fixed point


def test_joint_ordering_is_greedy_nearest():
    order = wp.order_by_joint_distance([0, 0], [("far", [3, 3]), ("near", [0.1, 0]), ("mid", [1, 1])])
    assert order == ["near", "mid", "far"]
    assert wp.max_joint_change_deg([0, 0], [math.radians(30), 0]) == pytest.approx(30.0)
