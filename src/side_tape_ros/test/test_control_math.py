"""Pure-math tests for side_tape_ros.control_math. No rclpy, no node spin.

Run: source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash \
     && export PYTHONPATH=~/igus_rebel_ros2/src/side_tape_ros:$PYTHONPATH \
     && python3 -m pytest -q -p no:anyio -p no:cacheprovider ~/igus_rebel_ros2/src/side_tape_ros/test/test_control_math.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from side_tape_ros import control_math as cm
from precision_box_detector.side_tape.supervisor import ServoStatus
from precision_box_detector.side_tape.types import ControlOutput, Phase


def test_quat_identity_is_identity_matrix():
    R = cm.quat_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quat_90deg_about_z_maps_x_to_y():
    # 90 deg about +Z: qz = sin(45deg), qw = cos(45deg)
    q = math.sin(math.radians(45.0))
    w = math.cos(math.radians(45.0))
    R = cm.quat_to_rotation_matrix(0.0, 0.0, q, w)
    x_axis = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(R @ x_axis, [0.0, 1.0, 0.0], atol=1e-9)


def test_freeze_axes_identity_rotation_passthrough():
    R = np.eye(3)
    cut, lat, depth = cm.freeze_axes(R, [0, -1, 0], [1, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(cut, [0, -1, 0])
    np.testing.assert_allclose(lat, [1, 0, 0])
    np.testing.assert_allclose(depth, [0, 0, 1])


def test_freeze_axes_rotated_tool():
    # tool rotated 90 deg about base Z: base = R @ tool_axis
    q = math.sin(math.radians(45.0))
    w = math.cos(math.radians(45.0))
    R = cm.quat_to_rotation_matrix(0.0, 0.0, q, w)
    cut, lat, depth = cm.freeze_axes(R, [0, -1, 0], [1, 0, 0], [0, 0, 1])
    # tool -Y -> base -(-X)?? compute directly: R@[0,-1,0]
    np.testing.assert_allclose(cut, R @ np.array([0.0, -1.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(lat, R @ np.array([1.0, 0.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(depth, [0.0, 0.0, 1.0], atol=1e-9)


def _vecs(p0=(0.0, 0.0, 0.0)):
    cut = np.array([0.0, -1.0, 0.0])
    lat = np.array([1.0, 0.0, 0.0])
    depth = np.array([0.0, 0.0, 1.0])
    return cm.TcpVectors(cut=cut, lateral=lat, depth=depth, p0_m=np.asarray(p0, dtype=np.float64))


def test_tcp_state_mm_projects_onto_frozen_axes():
    vecs = _vecs()
    # moved +5mm along cut(-Y) => p = (0, -0.005, 0); +2mm lateral (+X); +1mm depth(+Z)
    p = np.array([0.002, -0.005, 0.001])
    along, lat, depth = cm.tcp_state_mm(p, vecs)
    assert along == pytest.approx(5.0)
    assert lat == pytest.approx(2.0)
    assert depth == pytest.approx(1.0)


def test_tcp_state_mm_relative_to_frozen_origin():
    vecs = _vecs(p0=(0.010, 0.0, 0.0))  # frozen 10mm away in +X (lateral) at start
    p = np.array([0.010, 0.0, 0.0])  # tool hasn't moved since freeze
    along, lat, depth = cm.tcp_state_mm(p, vecs)
    assert (along, lat, depth) == (pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0))


def test_compose_linear_velocity_sums_axes():
    vecs = _vecs()
    out = ControlOutput(v_cut_mps=0.01, v_lat_mps=-0.002, v_depth_mps=0.003, phase=Phase.CUT)
    v = cm.compose_linear_velocity(out, vecs)
    expected = 0.01 * vecs.cut + (-0.002) * vecs.lateral + 0.003 * vecs.depth
    np.testing.assert_allclose(v, expected)


def test_compose_linear_velocity_zero_output_is_zero_twist():
    vecs = _vecs()
    v = cm.compose_linear_velocity(ControlOutput.zero(Phase.HOLD), vecs)
    np.testing.assert_allclose(v, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("code,label", [
    (0, "OK"),
    (1, "WARN"), (3, "WARN"), (6, "WARN"),
    (2, "HALT"), (4, "HALT"), (5, "HALT"),
    (99, "UNKNOWN"),
])
def test_map_servo_status_int(code, label):
    assert cm.map_servo_status_int(code) == label


def test_map_servo_status_int_none_is_unknown():
    assert cm.map_servo_status_int(None) == "UNKNOWN"


def test_servo_status_for_supervisor_direct_labels():
    assert cm.servo_status_for_supervisor("OK", dry_run=False) == ServoStatus.OK
    assert cm.servo_status_for_supervisor("WARN", dry_run=False) == ServoStatus.WARN
    assert cm.servo_status_for_supervisor("HALT", dry_run=False) == ServoStatus.HALT
    assert cm.servo_status_for_supervisor("HALT", dry_run=True) == ServoStatus.HALT


def test_servo_status_for_supervisor_unknown_collapses_by_dry_run():
    assert cm.servo_status_for_supervisor("UNKNOWN", dry_run=True) == ServoStatus.OK
    assert cm.servo_status_for_supervisor("UNKNOWN", dry_run=False) == ServoStatus.WARN


@pytest.mark.parametrize("statuses,expected", [
    ([], False),
    ([4], False),  # SUCCEEDED
    ([5], False),  # CANCELED
    ([6], False),  # ABORTED
    ([1], True),   # ACCEPTED
    ([2], True),   # EXECUTING
    ([3], True),   # CANCELING
    ([4, 2], True),
])
def test_trajectory_goal_active(statuses, expected):
    assert cm.trajectory_goal_active(statuses) is expected


def test_arm_owned_requires_fresh_tf():
    assert cm.arm_owned(tf_fresh=False, trajectory_active=False, dry_run=True,
                         servo_unpaused_by_us=True, servo_status_age_s=0.0) is False


def test_arm_owned_requires_no_active_trajectory():
    assert cm.arm_owned(tf_fresh=True, trajectory_active=True, dry_run=True,
                         servo_unpaused_by_us=True, servo_status_age_s=0.0) is False


def test_arm_owned_dry_run_bypasses_servo_checks():
    assert cm.arm_owned(tf_fresh=True, trajectory_active=False, dry_run=True,
                         servo_unpaused_by_us=False, servo_status_age_s=None) is True


def test_arm_owned_live_requires_servo_unpaused_and_fresh_status():
    base = dict(tf_fresh=True, trajectory_active=False, dry_run=False)
    assert cm.arm_owned(**base, servo_unpaused_by_us=True, servo_status_age_s=0.5) is True
    assert cm.arm_owned(**base, servo_unpaused_by_us=False, servo_status_age_s=0.5) is False
    assert cm.arm_owned(**base, servo_unpaused_by_us=True, servo_status_age_s=1.5) is False
    assert cm.arm_owned(**base, servo_unpaused_by_us=True, servo_status_age_s=None) is False


def test_sanitize_linear_velocity_passthrough_within_limit():
    v = np.array([0.001, 0.0, 0.0])
    v_safe, nan = cm.sanitize_linear_velocity(v, v_cut_mps=0.015, v_lat_max_mps=0.005,
                                               v_retract_mps=0.01)
    assert nan is False
    np.testing.assert_allclose(v_safe, v)


def test_sanitize_linear_velocity_clamps_overspeed():
    limit = max(0.015, 0.005, 0.01) * 1.2  # 0.018
    v = np.array([1.0, 0.0, 0.0])  # way over
    v_safe, nan = cm.sanitize_linear_velocity(v, v_cut_mps=0.015, v_lat_max_mps=0.005,
                                               v_retract_mps=0.01)
    assert nan is False
    assert np.linalg.norm(v_safe) == pytest.approx(limit)
    np.testing.assert_allclose(v_safe / np.linalg.norm(v_safe), [1.0, 0.0, 0.0])


def test_sanitize_linear_velocity_nan_zeros_and_flags():
    v = np.array([float("nan"), 0.0, 0.0])
    v_safe, nan = cm.sanitize_linear_velocity(v, v_cut_mps=0.015, v_lat_max_mps=0.005,
                                               v_retract_mps=0.01)
    assert nan is True
    np.testing.assert_allclose(v_safe, [0.0, 0.0, 0.0])


# -- ACM editing (finding 8) -------------------------------------------------

def test_acm_add_allowed_pairs_grows_matrix_from_empty():
    names, enabled = cm.acm_add_allowed_pairs([], [], "carton", ["tool_tip", "handpiece"])
    assert set(names) == {"carton", "tool_tip", "handpiece"}
    i = names.index("carton")
    for link in ("tool_tip", "handpiece"):
        j = names.index(link)
        assert enabled[i][j] is True
        assert enabled[j][i] is True
    # unrelated pair (tool_tip vs handpiece) untouched (still False).
    a, b = names.index("tool_tip"), names.index("handpiece")
    assert enabled[a][b] is False


def test_acm_add_allowed_pairs_preserves_existing_entries():
    names = ["link_a", "link_b", "carton"]
    enabled = [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]
    new_names, new_enabled = cm.acm_add_allowed_pairs(names, enabled, "carton", ["link_a"])
    assert cm.acm_matrices_equal(new_names, new_enabled, ["link_a", "link_b", "carton"],
                                  [[False, True, True], [True, False, False], [True, False, False]])
    # original untouched (function does not mutate inputs)
    assert enabled == [[False, True, False], [True, False, False], [False, False, False]]


def test_acm_add_allowed_pairs_idempotent():
    names, enabled = cm.acm_add_allowed_pairs([], [], "carton", ["tool_tip"])
    names2, enabled2 = cm.acm_add_allowed_pairs(names, enabled, "carton", ["tool_tip"])
    assert cm.acm_matrices_equal(names, enabled, names2, enabled2)
    assert len(names2) == 2  # no duplicate rows/cols


def test_acm_add_allowed_pairs_adds_link_not_previously_in_matrix():
    names = ["carton"]
    enabled = [[False]]
    new_names, new_enabled = cm.acm_add_allowed_pairs(names, enabled, "carton", ["new_link"])
    assert "new_link" in new_names
    i, j = new_names.index("carton"), new_names.index("new_link")
    assert new_enabled[i][j] is True and new_enabled[j][i] is True


def test_acm_matrices_equal_ignores_row_order():
    names_a = ["x", "y"]
    enabled_a = [[False, True], [True, False]]
    names_b = ["y", "x"]
    enabled_b = [[False, True], [True, False]]
    assert cm.acm_matrices_equal(names_a, enabled_a, names_b, enabled_b)


# -- Servo parameter contract (finding 7d) -----------------------------------

class _FakeParamValue:
    def __init__(self, type_, **kwargs):
        self.type = type_
        self.string_value = kwargs.get("string_value", "")
        self.bool_value = kwargs.get("bool_value", False)
        self.integer_value = kwargs.get("integer_value", 0)
        self.double_value = kwargs.get("double_value", 0.0)


def test_parse_get_parameters_result_decodes_by_type():
    names = ["a", "b", "c", "d"]
    values = [
        _FakeParamValue(4, string_value="base_link"),
        _FakeParamValue(2, integer_value=5),
        _FakeParamValue(3, double_value=0.1),
        _FakeParamValue(0),
    ]
    out = cm.parse_get_parameters_result(names, values)
    assert out == {"a": "base_link", "b": 5, "c": pytest.approx(0.1), "d": None}


def test_servo_params_ok_matches_config():
    ok, msg = cm.servo_params_ok("base_link", "speed_units", 0.1, "base_link", "speed_units", 0.3)
    assert ok is True and msg == ""


def test_servo_params_ok_rejects_frame_mismatch():
    ok, msg = cm.servo_params_ok("tool0", "speed_units", 0.1, "base_link", "speed_units", 0.3)
    assert ok is False and "robot_link_command_frame" in msg


def test_servo_params_ok_rejects_command_in_type_mismatch():
    ok, msg = cm.servo_params_ok("base_link", "unit_less", 0.1, "base_link", "speed_units", 0.3)
    assert ok is False and "command_in_type" in msg


def test_servo_params_ok_rejects_timeout_over_configured_max():
    ok, msg = cm.servo_params_ok("base_link", "speed_units", 0.5, "base_link", "speed_units", 0.3)
    assert ok is False and "incoming_command_timeout" in msg


def test_servo_params_ok_rejects_missing_params():
    ok, msg = cm.servo_params_ok(None, "speed_units", 0.1, "base_link", "speed_units", 0.3)
    assert ok is False and msg == "servo_params_unavailable"


def test_servo_incoming_timeout_warning_recommends_point_one():
    assert cm.servo_incoming_timeout_warning(0.3) is not None
    assert cm.servo_incoming_timeout_warning(0.1) is None
    assert cm.servo_incoming_timeout_warning(0.05) is None
