"""Tests of the pure functions of a2_ab_test.py and a2_ab_analyze.py (no ROS graph, no bags).

    cd tools && python3 -m pytest -p no:anyio test_a2_ab.py
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a2_ab_analyze as an  # noqa: E402
import a2_ab_test as ab  # noqa: E402


# ------------------------------------------------------------------ trajectory
def test_quintic_has_zero_velocity_and_acceleration_at_the_end_and_reaches_the_goal():
    start = [0.0, 0.5, -1.0, 0.0, 0.2, 0.0]
    goal = [0.0, 0.7, -1.0, 0.1, 0.2, -0.3]
    points = ab.quintic_trajectory(start, goal, 8.0)
    assert len(points) == ab.TRAJECTORY_POINTS
    t, pos, vel, acc = points[-1]
    assert t == pytest.approx(8.0)
    assert pos == pytest.approx(goal)
    assert vel == pytest.approx([0.0] * 6, abs=1e-12)
    assert acc == pytest.approx([0.0] * 6, abs=1e-12)
    # strictly increasing times, monotonic joint 2 (quintic min-jerk is monotonic)
    times = [p[0] for p in points]
    assert all(b > a for a, b in zip(times, times[1:]))
    j2 = [p[1][1] for p in points]
    assert all(b >= a for a, b in zip(j2, j2[1:]))
    # holding joints stay put
    assert all(p[1][0] == 0.0 and p[2][0] == 0.0 for p in points)
    assert all(p[1][2] == -1.0 for p in points)


def test_quintic_velocity_is_the_derivative_of_position():
    start, goal, duration = [0.0], [1.0], 10.0
    points = ab.quintic_trajectory(start, goal, duration, n_points=1000)
    for (t0, p0, _, _), (t1, p1, v1, _) in zip(points, points[1:]):
        assert (p1[0] - p0[0]) / (t1 - t0) == pytest.approx(v1[0], abs=2e-3)


def test_planned_peak_velocity_matches_the_analytic_value():
    peak = ab.planned_peak_velocity([0.0], [0.4], 8.0)
    assert peak == pytest.approx(1.875 * 0.4 / 8.0)
    points = ab.quintic_trajectory([0.0], [0.4], 8.0, n_points=1001)
    assert max(abs(p[2][0]) for p in points) == pytest.approx(peak, rel=1e-4)


def test_quintic_rejects_bad_input():
    with pytest.raises(ValueError):
        ab.quintic_trajectory([0.0], [0.0, 1.0], 5.0)
    with pytest.raises(ValueError):
        ab.quintic_trajectory([0.0], [1.0], 0.0)


# ------------------------------------------------------------------ limit checks
START = [0, -30, 60, 0, 45, 0]


def goal_with(**delta):
    goal = list(START)
    for name, value in delta.items():
        goal[int(name[1:]) - 1] += value
    return goal


def test_ok_jtc_and_movejoint_requests_pass():
    assert ab.validate_args(START, goal_with(j2=15), "jtc", 8.0, 10, 10) == []
    assert ab.validate_args(START, goal_with(j2=15), "movejoint", 0, 10, 10) == []
    assert ab.validate_args(START, goal_with(j2=15), "movejoint", 0, 25, 25) == []


def test_delta_over_40_deg_is_refused_on_any_joint():
    assert ab.validate_args(START, goal_with(j2=40.0), "movejoint", 0, 10, 10) == []
    errors = ab.validate_args(START, goal_with(j5=-40.5), "movejoint", 0, 10, 10)
    assert any("joint5" in e for e in errors)


def test_identical_start_and_goal_and_wrong_lengths_are_refused():
    assert ab.validate_args(START, START, "movejoint", 0, 10, 10)
    assert ab.validate_args(START[:5], START, "movejoint", 0, 10, 10)
    assert ab.validate_args(START, goal_with(j1=float("nan")), "movejoint", 0, 10, 10)


def test_jtc_duration_below_4_s_is_refused():
    assert ab.validate_args(START, goal_with(j2=2), "jtc", 3.99, 10, 10)
    assert ab.validate_args(START, goal_with(j2=2), "jtc", float("nan"), 10, 10)
    assert ab.validate_args(START, goal_with(j2=2), "jtc", 4.0, 10, 10) == []


def test_jtc_peak_velocity_over_02_rad_s_is_refused():
    # 40 deg in 8 s: 1.875 * 0.698 / 8 = 0.164 rad/s -> ok; in 6 s: 0.218 -> refused
    assert ab.validate_args(START, goal_with(j2=40), "jtc", 8.0, 10, 10) == []
    errors = ab.validate_args(START, goal_with(j2=40), "jtc", 6.0, 10, 10)
    assert any("peak joint velocity" in e for e in errors)
    # a 10 s duration cannot hide a fast joint: the largest joint decides
    assert ab.validate_args(START, goal_with(j1=40, j2=1), "jtc", 6.0, 10, 10)


def test_movejoint_velocity_and_acceleration_cap_25_percent():
    assert ab.validate_args(START, goal_with(j2=5), "movejoint", 0, 25.1, 10)
    assert ab.validate_args(START, goal_with(j2=5), "movejoint", 0, 10, 25.1)
    assert ab.validate_args(START, goal_with(j2=5), "movejoint", 0, 0, 10)
    assert ab.validate_args(START, goal_with(j2=5), "movejoint", 0, float("inf"), 10)


def test_deadband_only_for_jtc_and_in_range():
    assert ab.validate_args(START, goal_with(j2=5), "jtc", 8, 10, 10, deadband=0.5) == []
    assert ab.validate_args(START, goal_with(j2=5), "jtc", 8, 10, 10, deadband=2.5)
    assert ab.validate_args(START, goal_with(j2=5), "movejoint", 0, 10, 10, deadband=0.5)


def test_start_pose_must_be_within_2_deg():
    current = [0.5, -30.0, 61.9, 0.0, 45.0, -1.9]
    assert ab.start_pose_errors(current, START) == []
    current[2] = 62.1
    errors = ab.start_pose_errors(current, START)
    assert len(errors) == 1 and "joint3" in errors[0]
    assert ab.start_pose_errors([float("nan")] + START[1:], START)


def hw(**overrides):
    state = dict(latched=False, latch_reason="", connected=True, activated=True,
                 status_fresh=True, move_active=False, cart_jog_active=False,
                 joint_jog_active=False)
    state.update(overrides)
    return SimpleNamespace(**state)


def test_hardware_refusals():
    assert ab.hardware_refusals(hw(), "jtc", "active", "jtc") == []
    assert ab.hardware_refusals(hw(), "movejoint", "inactive", "jtc") == []
    assert ab.hardware_refusals(hw(), "movejoint", None, "jtc") == []
    assert any("LATCHED" in e for e in
               ab.hardware_refusals(hw(latched=True, latch_reason="x"), "jtc", "active", "jtc"))
    assert any("not connected" in e for e in
               ab.hardware_refusals(hw(connected=False), "jtc", "active", "jtc"))
    assert ab.hardware_refusals(None, "jtc", "active", "jtc")
    assert ab.hardware_refusals(hw(status_fresh=False), "jtc", "active", "jtc")
    assert ab.hardware_refusals(hw(joint_jog_active=True), "jtc", "active", "jtc")


def test_controller_state_rules_per_mode():
    assert ab.hardware_refusals(hw(), "jtc", "inactive", "jtc")
    assert ab.hardware_refusals(hw(), "jtc", None, "jtc")
    assert ab.hardware_refusals(hw(), "movejoint", "active", "jtc")
    assert ab.controller_state_wrong("jtc", "inactive")
    assert not ab.controller_state_wrong("jtc", "active")
    assert ab.controller_state_wrong("movejoint", "active")
    assert not ab.controller_state_wrong("movejoint", "inactive")
    assert not ab.controller_state_wrong("movejoint", None)


def test_switch_command_text_and_bag_command():
    assert ab.switch_controllers_command("jtc", "rebel_arm_trajectory_controller", "/controller_manager") == (
        "ros2 control switch_controllers -c /controller_manager --activate "
        "rebel_arm_trajectory_controller")
    assert "--deactivate" in ab.switch_controllers_command("movejoint", "c", "/controller_manager")
    topics = ab.default_topics("igus_rebel", "rebel_arm_trajectory_controller")
    assert "/igus_rebel/cri_diagnostics" in topics
    assert "/rebel_arm_trajectory_controller/controller_state" in topics
    assert "/rosout" in topics and "/dynamic_joint_states" in topics
    command = ab.bag_record_command("/tmp/x", topics, distro="humble")
    assert command == ["ros2", "bag", "record", "-o", "/tmp/x"] + topics  # Humble has no --topics
    assert ab.bag_record_command("/tmp/x", topics, distro="jazzy")[5] == "--topics"


def test_dry_run_validates_without_ros(capsys):
    argv = ["--mode", "jtc", "--duration", "8", "--dry-run", "--start-deg", *map(str, START),
            "--goal-deg", *map(str, goal_with(j2=10))]
    assert ab.main(argv) == 0
    assert "Dry run OK" in capsys.readouterr().out
    argv[argv.index("8")] = "2"
    assert ab.main(argv) == 1


# ------------------------------------------------------------------ analyzer
def test_sign_changes_skips_zeros_and_counts_flips():
    assert an.sign_changes([]) == 0
    assert an.sign_changes([0, 0, 0]) == 0
    assert an.sign_changes([0.3, 0.3, 0.3]) == 0
    assert an.sign_changes([0.3, -0.3, 0.3, -0.3]) == 3
    assert an.sign_changes([0.3, 0, -0.3, 0, 0, 0.3]) == 2  # zeros between do not hide a flip
    assert an.sign_changes([0.3, 0, 0.3, 0, 0.3]) == 0  # a pause is not a flip
    assert an.sign_changes([0.3, float("nan"), 0.4]) == 0


def test_small_jog_fraction_counts_only_open_interval_0_to_1():
    jogs = [0.0, 0.5, -0.5, 1.0, -1.0, 3.0, 0.999, 0.0]
    assert an.small_jog_fraction(jogs) == pytest.approx(3 / 8)
    assert an.small_jog_fraction([]) == 0.0


def test_joint_stats():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    currents = [100, -300, 200, -100, 0]
    jogs = [0.4, -0.4, 0.4, -0.4, 5.0]
    stats = an.joint_stats(times, currents, jogs)
    assert stats["frames"] == 5
    assert stats["peak_current"] == 300
    assert stats["rms_current"] == pytest.approx(math.sqrt((1e4 + 9e4 + 4e4 + 1e4 + 0) / 5))
    assert stats["peak_jog"] == 5.0
    assert stats["sign_changes"] == 4
    assert stats["sign_changes_per_s"] == pytest.approx(2.0)
    assert stats["small_jog_fraction"] == pytest.approx(4 / 5)
    assert an.joint_stats([], [], [])["peak_current"] == 0


def test_classify_joints_holding_vs_moving():
    groups = an.classify_joints([0, -30, 60, 0, 45, 0], [0, -10, 60, 0, 45, 0.5])
    assert groups == ["holding", "moving", "holding", "holding", "holding", "moving"]


def synthetic_run(n=100, dt=0.01, dither_joint=0):
    """Joint 2 moves with a steady jog; `dither_joint` (0-based) flips +-0.3 % every frame."""
    t0 = 1000.0
    diag = {"t": [t0 + i * dt for i in range(n)], "current": [[0] * n for _ in range(6)],
            "jog": [[0.0] * n for _ in range(6)], "position_deg": [[0.0] * n for _ in range(6)],
            "source": ["jtc"] * n}
    for i in range(n):
        diag["jog"][1][i] = 4.0
        diag["current"][1][i] = 2000 + i
        diag["jog"][dither_joint][i] = 0.3 if i % 2 == 0 else -0.3
        diag["current"][dither_joint][i] = 100 if i % 2 == 0 else -100
    return diag


def test_analyse_run_separates_holding_and_moving_joints():
    diag = synthetic_run()
    groups = an.classify_joints([0, -30, 60, 0, 45, 0], [0, -10, 60, 0, 45, 0])
    groups[5] = "moving"
    result = an.analyse_run(diag, groups)
    assert result["frames"] == 100
    holding = result["holding"]
    assert holding["joints"] == [1, 3, 4, 5]
    assert holding["peak_current"] == 100
    assert holding["peak_jog"] == pytest.approx(0.3)
    assert holding["small_jog_fraction"] == pytest.approx(100 / 400)  # only joint 1 dithers
    assert holding["sign_changes_per_s"] == pytest.approx((99 / 0.99) / 4)
    moving = result["moving"]
    assert moving["joints"] == [2, 6]
    assert moving["peak_current"] == 2099
    assert moving["peak_jog"] == 4.0
    assert result["per_joint"][1]["sign_changes"] == 0


def test_window_restricts_the_frames():
    diag = synthetic_run()
    groups = ["moving"] * 6
    result = an.analyse_run(diag, groups, t_start=1000.20, t_end=1000.39)
    assert result["frames"] == 20


def test_trip_events_from_rosout_and_hardware_state():
    rosout = [
        (10.0, 20, "igus_rebel", "Receiving write() commands: forwarding jog commands"),
        (12.5, 20, "igus_rebel", "Rebel MESSAGE: Overcurrent axis 2"),
        (12.6, 40, "igus_rebel", "robot controller reports a stop after the arm was enabled"),
        (13.0, 30, "other", "something unrelated"),
    ]
    hardware = [(11.0, False, "", 0), (12.7, True, "overcurrent", 0), (12.8, True, "overcurrent", 0),
                (12.9, True, "overcurrent", 0x02)]
    events = an.find_trip_events(rosout, hardware)
    assert [e["t"] for e in events] == [12.5, 12.6, 12.7, 12.9]
    assert an.first_trip_time(events) == 12.5
    assert an.first_trip_time([]) is None
    assert an.find_trip_events([(1.0, 20, "x", "all fine")], [(1.0, False, "", 0)]) == []


def test_a2_current_in_the_last_second_before_the_trip():
    times = [i * 0.1 for i in range(100)]            # 0 .. 9.9 s
    currents = [100] * 80 + [7600] * 20              # the last 2 s are high
    jogs = [0.0] * 100
    summary = an.trip_window_summary(times, currents, jogs, t_trip=9.5)
    assert summary["frames"] == 11                    # 8.5 .. 9.5 inclusive
    assert summary["peak_current"] == 7600
    assert an.trip_window_summary(times, currents, jogs, None) is None
    only_before = an.trip_window_summary(times, currents, jogs, t_trip=5.0)
    assert only_before["peak_current"] == 100


def test_analyse_run_includes_trip_window_over_the_whole_recording():
    diag = synthetic_run()
    events = [{"t": 1000.9, "source": "rosout", "level": 40, "detail": "x"}]
    result = an.analyse_run(diag, ["moving"] * 6, t_start=1000.0, t_end=1000.3, trip_events=events)
    assert result["trip_time"] == 1000.9
    assert result["trip_a2"]["frames"] == 91         # 1000.0 .. 1000.9 inside a 1 s window
    text = an.format_report("run", {"label": "a", "mode": "jtc", "result": {"success": False}}, result)
    assert "TRIP" in text and "A2" in text


def test_extract_diagnostics_from_message_like_objects():
    def message(sec, nanosec):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
            current_raw=[1, 2, 3, 4, 5, 6], jog_percent_sent=[0.5] * 6,
            position_deg=[10.0] * 6, jog_source="jtc")

    diag = an.extract_diagnostics([message(5, 500_000_000), message(6, 0)])
    assert diag["t"] == [5.5, 6.0]
    assert diag["current"][2] == [3, 3]
    assert diag["jog"][0] == [0.5, 0.5]
    assert diag["source"] == ["jtc", "jtc"]
    text = an.format_report("run", None, an.analyse_run(diag, ["holding"] * 6))
    assert "holding" in text and "no overcurrent" in text
