import math

import pytest

from dual_teleop.selector_core import (
    allow_same_arm_pairs,
    STATE_EXECUTING, STATE_IDLE, STATE_PLANNED, STATE_STOPPED, STATE_WAITING_BARRIER,
    Command, Config, SelectorCore, is_zero)


def twist(vx=0.0, vy=0.0, vz=0.0, wz=0.0, frame="base_link"):
    return Command("twist", frame=frame, linear=(vx, vy, vz), angular=(0.0, 0.0, wz))


def joint(index, vel, names=None):
    v = [0.0] * 6
    v[index] = vel
    return Command("joint", joint_names=names or [f"joint{n}" for n in range(1, 7)], velocities=v)


def ready_core(now=100.0, coordinator=STATE_IDLE, arm="left", enabled=True, **cfg):
    """Core with both controllers active, both Servos running and a fresh coordinator status."""
    core = SelectorCore(Config(**cfg), default_arm=arm, enabled=enabled)
    core.update_scene(now)
    for a in ("left", "right"):
        core.update_controller(a, True, now)
        core.set_servo_running(a, True)
    if coordinator is not None:
        core.update_coordinator(coordinator, now)
    core.refresh(now)
    # a first neutral command clears the "needs neutral" latch
    core.handle_command(twist(), now)
    return core


def test_start_state_is_disabled_and_drops_everything():
    core = SelectorCore(Config(), default_arm="left")
    core.update_controller("left", True, 1.0)
    core.update_controller("right", True, 1.0)
    core.set_servo_running("left", True)
    out, why = core.handle_command(twist(vx=0.02), 1.0)
    assert why.startswith("blocked") and "disabled" in why
    assert not [o for o in out if not is_zero(o)]
    assert core.status(1.0)["state"] == "PAUSED"
    assert not core.desired_unpaused("left", 1.0)


def test_forward_to_selected_arm_translates_frames_and_joints():
    core = ready_core()
    out, why = core.handle_command(twist(vx=0.02, frame="base_link"), 101.0)
    assert why == ""
    assert [(o.arm, o.kind, o.frame) for o in out] == [("left", "twist", "world")]
    out, _ = core.handle_command(twist(vy=0.02, frame="xeg32_tool_tip"), 101.1)
    assert out[0].frame == "arm_left_xeg32_tool_tip"
    out, _ = core.handle_command(joint(2, 0.1), 101.2)
    assert out[0].joint_names == [f"arm_left_joint{n}" for n in range(1, 7)]
    assert out[0].arm == "left"
    core.select("right", 102.0)
    core.update_controller("right", True, 102.5)
    core.update_controller("left", True, 102.5)
    out, why = core.handle_command(twist(), 103.0)  # neutral
    out, why = core.handle_command(twist(vx=0.02, frame="xeg32_tool_tip"), 103.1)
    assert why == "" and out[0].arm == "right" and out[0].frame == "arm_right_xeg32_tool_tip"


def test_foreign_frames_and_joints_are_dropped():
    core = ready_core()
    _, why = core.handle_command(twist(vx=0.01, frame="arm_right_xeg32_tool_tip"), 101.0)
    assert "foreign" in why
    _, why = core.handle_command(twist(vx=0.01, frame="camera_link"), 101.1)
    assert "foreign" in why
    names = [f"arm_right_joint{n}" for n in range(1, 7)]
    _, why = core.handle_command(joint(0, 0.05, names=names), 101.2)
    assert "does not belong to arm_left" in why
    _, why = core.handle_command(twist(vx=math.nan), 101.3)
    assert why == "non-finite command"


def test_switching_zeroes_the_old_arm_first():
    core = ready_core(arm="left")
    out, _ = core.handle_command(twist(vx=0.03), 101.0)
    assert out[0].arm == "left" and not is_zero(out[0])
    zeros = core.select("right", 101.2)
    assert {(o.arm, o.kind) for o in zeros} == {("left", "twist"), ("left", "joint")}
    assert all(is_zero(o) for o in zeros)
    assert core.selected == "right"
    # selecting the same arm again changes nothing
    assert core.select("right", 101.3) == []
    # the old arm's Servo is no longer wanted unpaused; it may be paused after pause_delay_s
    assert not core.desired_unpaused("left", 101.3)
    assert not core.pause_due("left", 101.3)  # zeros need a moment to take effect
    assert core.pause_due("left", 101.2 + core.cfg.pause_delay_s + 0.01)


def test_command_held_through_a_switch_is_not_forwarded_to_the_new_arm():
    core = ready_core(arm="left")
    t = 101.0
    core.handle_command(twist(vx=0.03), t)
    core.select("right", t + 0.01)
    # operator still holds the key: continuous non-zero stream with small gaps -> dropped
    for i in range(1, 30):
        out, why = core.handle_command(twist(vx=0.03), t + 0.01 + i * 0.03)
        assert why.startswith("waiting for a neutral"), why
        assert not [o for o in out if o.arm == "right" and not is_zero(o)]
    # after the key is released (a zero command) the next command goes to the right arm
    core.handle_command(twist(), t + 1.5)
    out, why = core.handle_command(twist(vx=0.03), t + 1.55)
    assert why == "" and out[0].arm == "right"


def test_new_command_after_a_pause_is_accepted_without_a_zero():
    core = ready_core(arm="left")
    core.select("right", 101.0)
    out, why = core.handle_command(twist(vx=0.02), 101.5)  # first input after a >0.3 s gap
    assert why == "" and out[0].arm == "right"


@pytest.mark.parametrize("state", [STATE_EXECUTING, STATE_WAITING_BARRIER])
def test_coordinator_executing_blocks_both_arms(state):
    core = ready_core()
    out, why = core.handle_command(twist(vx=0.02), 101.0)
    assert why == ""
    core.update_coordinator(state, 101.5)
    out, why = core.handle_command(twist(vx=0.02), 101.6)
    assert why.startswith("blocked") and "coordinator" in why
    assert all(is_zero(o) for o in out), "only zeros may leave while blocked"
    assert any(o.arm == "left" for o in out), "the moving arm gets a zero command when teleop closes"
    st = core.status(101.6)
    assert st["state"] == "BLOCKED" and st["coordinator_blocking"]
    # BOTH servos are to be paused (also the one that was not selected)
    assert not core.desired_unpaused("left", 101.6) and not core.desired_unpaused("right", 101.6)
    assert core.pause_due("right", 101.6)
    assert not core.pause_due("left", 101.6)
    assert core.pause_due("left", 101.6 + core.cfg.pause_delay_s + 0.01)
    # selecting the other arm does not bypass it
    core.select("right", 101.7)
    _, why = core.handle_command(twist(vx=0.02), 102.0)
    assert why.startswith("blocked")


def test_coordinator_going_idle_resumes_only_after_a_neutral_command():
    core = ready_core()
    core.update_coordinator(STATE_EXECUTING, 101.0)
    core.handle_command(twist(vx=0.02), 101.1)
    core.update_coordinator(STATE_IDLE, 101.2)
    _, why = core.handle_command(twist(vx=0.02), 101.25)
    assert why.startswith("waiting for a neutral")
    core.handle_command(twist(), 101.3)
    out, why = core.handle_command(twist(vx=0.02), 101.35)
    assert why == "" and out[0].arm == "left"


@pytest.mark.parametrize("state", [STATE_IDLE, STATE_PLANNED, STATE_STOPPED])
def test_non_executing_states_do_not_block(state):
    core = ready_core(coordinator=state)
    _, why = core.handle_command(twist(vx=0.02), 101.0)
    assert why == "", "STOPPED must allow recovery jogging"


def test_unknown_coordinator_allows_teleop_with_a_warning():
    core = ready_core(coordinator=None)
    _, why = core.handle_command(twist(vx=0.02), 101.0)
    assert why == ""
    st = core.status(101.0)
    assert st["state"] == "ACTIVE" and st["warnings"] and "UNKNOWN" in st["coordinator"]


def test_stale_idle_status_becomes_unknown_but_stale_executing_stays_blocked():
    core = ready_core(coordinator=STATE_IDLE)
    late = 101.0 + core.cfg.coordinator_timeout_s + 1.0
    core.update_controller("left", True, late)
    core.update_controller("right", True, late)
    core.update_scene(late)
    _, why = core.handle_command(twist(vx=0.02), late)
    assert why == "" and core.status(late)["warnings"]

    core = ready_core(coordinator=STATE_EXECUTING)
    late = 101.0 + core.cfg.coordinator_timeout_s + 1.0
    core.update_controller("left", True, late)
    core.update_controller("right", True, late)
    core.update_scene(late)
    _, why = core.handle_command(twist(vx=0.02), late)
    assert why.startswith("blocked") and "STALE" in why
    core.ignore_stale = True
    core.handle_command(twist(), late + 0.1)
    _, why = core.handle_command(twist(vx=0.02), late + 0.15)
    assert why == ""


def test_inactive_or_unknown_controller_blocks_that_arm_only():
    core = ready_core(arm="left")
    core.update_controller("left", False, 101.0)
    _, why = core.handle_command(twist(vx=0.02), 101.1)
    assert why.startswith("blocked") and "arm_left_trajectory_controller" in why
    assert not core.desired_unpaused("left", 101.1)
    core.select("right", 101.2)
    core.handle_command(twist(), 101.8)
    _, why = core.handle_command(twist(vx=0.02), 101.85)
    assert why == "" and core.desired_unpaused("right", 101.85)


def test_controller_poll_not_answering_fails_closed():
    core = ready_core()
    late = 100.0 + core.cfg.controller_timeout_s + 0.5
    core.update_coordinator(STATE_IDLE, late)
    _, why = core.handle_command(twist(vx=0.02), late)
    assert why.startswith("blocked")


def test_servo_not_running_blocks():
    core = ready_core()
    core.set_servo_running("left", False)
    _, why = core.handle_command(twist(vx=0.02), 101.0)
    assert why.startswith("blocked") and "Servo" in why


def test_disable_sends_zeros_and_stops_forwarding():
    core = ready_core()
    core.handle_command(twist(vx=0.02), 101.0)
    zeros = core.set_enabled(False, 101.1)
    assert zeros and all(is_zero(o) and o.arm == "left" for o in zeros)
    _, why = core.handle_command(twist(vx=0.02), 101.2)
    assert why.startswith("blocked")


def test_speed_scaling_and_clamping():
    core = ready_core(speed_scale=0.5, max_linear_mps=0.05, max_angular_rps=0.3, max_joint_rps=0.1)
    out, _ = core.handle_command(twist(vx=0.04, wz=0.2), 101.0)
    assert out[0].linear[0] == pytest.approx(0.02) and out[0].angular[2] == pytest.approx(0.1)
    out, _ = core.handle_command(twist(vx=0.3, vy=0.4), 101.05)  # gamepad full stick: norm 0.5 -> 0.25 -> clamp 0.05
    assert math.hypot(out[0].linear[0], out[0].linear[1]) == pytest.approx(0.05)
    assert out[0].linear[1] / out[0].linear[0] == pytest.approx(4.0 / 3.0)  # direction preserved
    out, _ = core.handle_command(joint(3, 0.4), 101.1)
    assert out[0].velocities[3] == pytest.approx(0.1)
    out, _ = core.handle_command(joint(3, -0.04), 101.15)
    assert out[0].velocities[3] == pytest.approx(-0.02)


def test_base_frame_mode_arm_base():
    core = ready_core(base_command_frame="arm_base", arm="right")
    out, _ = core.handle_command(twist(vx=0.02), 101.0)
    assert out[0].frame == "arm_right_base_link"


def test_default_arm_validation():
    with pytest.raises(ValueError):
        SelectorCore(default_arm="middle")
    core = SelectorCore()
    with pytest.raises(ValueError):
        core.select("both", 0.0)


def test_missing_or_stale_scene_blocks_fail_closed():
    core = ready_core()
    _, why = core.handle_command(twist(vx=0.02), 101.0)
    assert why == ""
    late = 100.0 + core.cfg.scene_timeout_s + 1.0
    core.update_controller("left", True, late)
    core.update_controller("right", True, late)
    core.update_coordinator(STATE_IDLE, late)
    _, why = core.handle_command(twist(vx=0.02), late)
    assert why.startswith("blocked") and "planning scene" in why
    core.update_scene(late)
    core.handle_command(twist(), late + 0.1)
    _, why = core.handle_command(twist(vx=0.02), late + 0.15)
    assert why == ""
    # no scene ever
    c2 = SelectorCore(Config(), enabled=True)
    c2.update_controller("left", True, 1.0)
    c2.set_servo_running("left", True)
    c2.update_coordinator(STATE_IDLE, 1.0)
    assert any("planning scene" in r for r in c2.arm_block_reasons("left", 1.0))
    # and the check can be switched off
    c3 = SelectorCore(Config(require_scene=False), enabled=True)
    assert not any("planning scene" in r for r in c3.global_block_reasons(1.0))


def test_allow_same_arm_pairs_keeps_cross_arm_table_and_objects():
    names = ["arm_left_link5", "arm_left_knife_c_wrist_camera", "arm_right_link5", "arm_right_knife_c_blade",
             "table", "detected_carton"]
    n = len(names)
    rows = [[False] * n for _ in range(n)]
    rows[0][5] = rows[5][0] = True  # an existing allowance (carton contact) stays
    out = allow_same_arm_pairs(names, rows)
    idx = {k: i for i, k in enumerate(names)}
    assert out[idx["arm_left_link5"]][idx["arm_left_knife_c_wrist_camera"]]
    assert out[idx["arm_left_knife_c_wrist_camera"]][idx["arm_left_link5"]]
    assert out[idx["arm_right_link5"]][idx["arm_right_knife_c_blade"]]
    assert not out[idx["arm_left_link5"]][idx["arm_right_link5"]], "cross-arm pair must stay checked"
    assert not out[idx["arm_left_link5"]][idx["table"]], "arm vs table must stay checked"
    assert not out[idx["arm_right_link5"]][idx["detected_carton"]]
    assert out[idx["arm_left_link5"]][idx["detected_carton"]], "existing allowances are kept"
    assert rows[0][1] is False, "input rows are not modified"


# ---------------------------------------------------------------- touch mode
from dual_teleop.selector_core import (  # noqa: E402
    STATE_PLANNING, allow_touch_pairs, knife_links)


def touch_core(**cfg):
    core = ready_core(**cfg)
    ok, msg, zeros = core.set_touch(True, 100.5)
    assert ok, msg
    return core


def test_touch_mode_is_off_by_default_and_needs_an_explicit_request():
    core = ready_core()
    assert not core.touch_on and core.status(100.5)["touch"]["on"] is False
    ok, msg, zeros = core.set_touch(True, 100.5)
    assert ok and core.touch_on and core.touch_arm == "left"
    assert all(is_zero(z) and z.arm == "left" for z in zeros), "entering sends a stop first"
    st = core.status(100.5)["touch"]
    assert st["on"] and st["arm"] == "left" and st["remaining_s"] == pytest.approx(120.0)
    ok, msg, _ = core.set_touch(False, 101.0)
    assert ok and not core.touch_on and "turned off by the operator" in core.touch_off_reason


@pytest.mark.parametrize("coordinator", [None, STATE_IDLE, STATE_STOPPED])
def test_touch_mode_entry_allowed_states(coordinator):
    core = ready_core(coordinator=coordinator)
    assert core.set_touch(True, 100.5)[0]


@pytest.mark.parametrize("state", [STATE_PLANNING, STATE_PLANNED, STATE_WAITING_BARRIER, STATE_EXECUTING])
def test_touch_mode_entry_refused_for_any_other_coordinator_state(state):
    core = ready_core(coordinator=state)
    ok, msg, _ = core.set_touch(True, 100.5)
    assert not ok and "coordinator" in msg and not core.touch_on


def test_touch_mode_entry_refused_when_disabled_or_controller_inactive_or_no_scene():
    core = ready_core(enabled=False)
    assert not core.set_touch(True, 100.5)[0]
    core = ready_core()
    core.update_controller("right", False, 100.4)
    ok, msg, _ = core.set_touch(True, 100.5)
    assert not ok and "arm_right_trajectory_controller" in msg
    core = SelectorCore(Config(), enabled=True)
    core.update_controller("left", True, 1.0)
    core.update_controller("right", True, 1.0)
    assert not core.set_touch(True, 1.0)[0], "no scene yet"


def test_touch_auto_off_on_arm_switch():
    core = touch_core()
    zeros = core.select("right", 101.0)
    assert not core.touch_on and "arm switched" in core.touch_off_reason
    assert zeros and all(z.arm == "left" for z in zeros)
    assert core.effective_scale("right") == core.cfg.speed_scale


def test_touch_auto_off_on_disable():
    core = touch_core()
    core.set_enabled(False, 101.0)
    assert not core.touch_on and "teleop disabled" in core.touch_off_reason


@pytest.mark.parametrize("state", [STATE_PLANNING, STATE_PLANNED, STATE_WAITING_BARRIER, STATE_EXECUTING])
def test_touch_auto_off_on_coordinator_activity(state):
    core = touch_core()
    core.update_coordinator(state, 101.0)
    out = core.refresh(101.0)
    assert not core.touch_on and "coordinator" in core.touch_off_reason
    assert any(o.arm == "left" and is_zero(o) for o in out)


def test_touch_stays_on_for_idle_stopped_and_unknown_coordinator():
    core = touch_core()
    core.update_coordinator(STATE_STOPPED, 101.0)
    core.refresh(101.0)
    assert core.touch_on
    core.update_coordinator(STATE_IDLE, 101.5)
    core.refresh(101.5)
    assert core.touch_on


def test_touch_auto_off_when_either_controller_goes_inactive():
    for arm in ("left", "right"):
        core = touch_core()
        core.update_controller(arm, False, 101.0)
        core.refresh(101.0)
        assert not core.touch_on and "not active" in core.touch_off_reason


def test_touch_auto_off_when_the_controller_poll_stops_answering():
    core = touch_core()
    late = 100.0 + core.cfg.controller_timeout_s + 0.5
    core.update_coordinator(STATE_IDLE, late)
    core.update_scene(late)
    core.refresh(late)
    assert not core.touch_on


def test_touch_auto_off_on_stale_scene():
    core = touch_core()
    late = 100.0 + core.cfg.scene_timeout_s + 0.5
    core.update_controller("left", True, late)
    core.update_controller("right", True, late)
    core.update_coordinator(STATE_IDLE, late)
    core.refresh(late)
    assert not core.touch_on and "scene" in core.touch_off_reason


def test_touch_timeout():
    core = touch_core(touch_mode_timeout_s=30.0)
    for t in (110.0, 120.0, 130.0):  # keep everything else fresh
        for a in ("left", "right"):
            core.update_controller(a, True, t)
        core.update_coordinator(STATE_IDLE, t)
        core.update_scene(t)
        core.refresh(t)
        assert core.touch_on
    for a in ("left", "right"):
        core.update_controller(a, True, 130.6)
    core.update_coordinator(STATE_IDLE, 130.6)
    core.update_scene(130.6)
    core.refresh(130.6)
    assert not core.touch_on and "timeout" in core.touch_off_reason
    # timeout 0 disables it
    core = touch_core(touch_mode_timeout_s=0.0)
    for a in ("left", "right"):
        core.update_controller(a, True, 500.0)
    core.update_coordinator(STATE_IDLE, 500.0)
    core.update_scene(500.0)
    core.refresh(500.0)
    assert core.touch_on


def test_touch_speed_is_reduced_and_never_above_the_normal_scale():
    core = ready_core(speed_scale=0.6, touch_speed_scale=0.2)
    out, _ = core.handle_command(twist(vx=0.06), 101.0)
    assert out[0].linear[0] == pytest.approx(0.036)
    core.set_touch(True, 101.1)
    core.handle_command(twist(), 101.2)
    out, why = core.handle_command(twist(vx=0.06), 101.25)
    assert why == "" and out[0].linear[0] == pytest.approx(0.012)
    out, _ = core.handle_command(joint(1, 0.3), 101.3)
    assert out[0].velocities[1] == pytest.approx(0.06)
    # touch_speed_scale above the normal scale is clamped to it
    core = ready_core(speed_scale=0.3, touch_speed_scale=0.9)
    assert core.effective_scale("left") == pytest.approx(0.3)
    core.set_touch(True, 100.5)
    assert core.effective_scale("left") == pytest.approx(0.3)
    # default: about a third
    assert Config().touch_speed_scale == pytest.approx(Config().speed_scale / 3, rel=0.01)
    # back to the normal scale after OFF; the command held through the change waits for a neutral
    core = ready_core(speed_scale=0.6, touch_speed_scale=0.2)
    core.set_touch(True, 100.5)
    core.handle_command(twist(vx=0.06), 100.9)  # dropped (neutral latch), but keeps the input stream alive
    core.set_touch(False, 101.0)
    _, why = core.handle_command(twist(vx=0.06), 101.02)
    assert why.startswith("waiting for a neutral")


def test_touch_acm_allows_exactly_the_knife_vs_carton_and_table():
    names = [f"arm_{a}_{l}" for a in ("left", "right") for l in
             ("link4", "link5", "link6", "knife_c_blade", "knife_c_handpiece", "knife_c_holder",
              "knife_c_wrist_camera", "xeg32_tool_tip")]
    names += ["table", "camera_pole", "camera_head"]
    n = len(names)
    rows = [[False] * n for _ in range(n)]
    new_names, new_rows = allow_touch_pairs(names, rows, "left", ["detected_carton"])
    idx = {k: i for i, k in enumerate(new_names)}
    assert "detected_carton" in idx and len(new_rows) == len(new_names) == n + 1
    assert all(len(r) == n + 1 for r in new_rows)
    allowed = {(a, b) for a in new_names for b in new_names if new_rows[idx[a]][idx[b]]}
    expected = set()
    for k in knife_links("left"):
        for t in ("detected_carton", "table"):
            expected |= {(k, t), (t, k)}
    assert allowed == expected
    assert knife_links("left") == ["arm_left_knife_c_blade", "arm_left_knife_c_handpiece", "arm_left_knife_c_holder",
                                   "arm_left_xeg32_tool_tip"]
    # not the arm's other links, not the other arm, not the pole/head
    for forbidden in ("arm_left_link4", "arm_left_link5", "arm_left_link6", "arm_left_knife_c_wrist_camera",
                      "arm_right_knife_c_blade", "arm_right_knife_c_holder"):
        assert not new_rows[idx[forbidden]][idx["detected_carton"]]
        assert not new_rows[idx[forbidden]][idx["table"]]
    for target in ("camera_pole", "camera_head", "arm_right_knife_c_blade", "arm_right_link5"):
        assert not new_rows[idx["arm_left_knife_c_blade"]][idx[target]]
    assert rows == [[False] * n for _ in range(n)], "input untouched"


def test_touch_acm_keeps_existing_carton_row_and_other_entries():
    names = ["arm_left_knife_c_blade", "arm_left_link5", "detected_carton", "table"]
    rows = [[False] * 4 for _ in range(4)]
    rows[1][2] = rows[2][1] = True  # an existing allowance of the stack stays as it is
    n2, r2 = allow_touch_pairs(names, rows, "left", ["detected_carton"])
    assert n2 == names and r2[1][2] and r2[0][2] and r2[0][3] and not r2[1][3]


# ---------------------------------------------------------------- retreat window after touch mode
def fresh(core, t):
    for a in ("left", "right"):
        core.update_controller(a, True, t)
    core.update_coordinator(STATE_IDLE, t)
    core.update_scene(t)


def test_operator_off_opens_a_retreat_window_that_keeps_the_knife_allowances_and_the_slow_speed():
    core = touch_core()
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.set_touch(False, 101.0)
    assert not core.touch_on and core.retreat_arm == "left" and core.acm_touch_arm == "left"
    assert core.effective_scale("left") == pytest.approx(core.cfg.touch_speed_scale)
    st = core.status(101.0)["touch"]
    assert st["on"] is False and st["retreat"] is True


def test_retreat_ends_after_the_tcp_moved_far_enough():
    core = touch_core(retreat_distance_m=0.08)
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.set_touch(False, 101.0)
    fresh(core, 102.0)
    core.update_tcp("left", (0.5, 0.0, 0.35))
    core.refresh(102.0)
    assert core.retreat_arm == "left"
    core.update_tcp("left", (0.5, 0.0, 0.39))
    core.refresh(102.5)
    assert core.retreat_arm is None and core.acm_touch_arm is None
    assert core.effective_scale("left") == core.cfg.speed_scale
    assert "retreated far enough" in core.touch_off_reason


def test_retreat_ends_by_time():
    core = touch_core(retreat_window_s=10.0)
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.set_touch(False, 101.0)
    fresh(core, 105.0)
    core.refresh(105.0)
    assert core.retreat_arm == "left"
    fresh(core, 111.5)
    core.refresh(111.5)
    assert core.retreat_arm is None and "retreat window over" in core.touch_off_reason


def test_retreat_ends_on_arm_switch_disable_and_blocking_coordinator_and_controller():
    def make():
        c = touch_core()
        c.update_tcp("left", (0.5, 0.0, 0.3))
        c.set_touch(False, 101.0)
        assert c.retreat_arm == "left"
        return c
    c = make(); c.select("right", 101.5); assert c.retreat_arm is None
    c = make(); c.set_enabled(False, 101.5); assert c.retreat_arm is None
    c = make(); fresh(c, 101.5); c.update_coordinator(STATE_EXECUTING, 101.5); c.refresh(101.5); assert c.retreat_arm is None
    c = make(); fresh(c, 101.5); c.update_controller("left", False, 101.5); c.refresh(101.5); assert c.retreat_arm is None


def test_retreat_follows_timeout_and_planning_but_not_the_safety_triggers():
    core = touch_core(touch_mode_timeout_s=5.0)
    core.update_tcp("left", (0.5, 0.0, 0.3))
    fresh(core, 110.0)
    core.refresh(110.0)  # timeout
    assert not core.touch_on and core.retreat_arm == "left"
    core = touch_core()
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.update_coordinator(STATE_PLANNING, 101.0)
    core.refresh(101.0)
    assert not core.touch_on and core.retreat_arm == "left"
    core = touch_core()
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.update_coordinator(STATE_EXECUTING, 101.0)
    core.refresh(101.0)
    assert not core.touch_on and core.retreat_arm is None, "executing: no unchecked knife"
    core = touch_core()
    core.update_tcp("left", (0.5, 0.0, 0.3))
    core.update_controller("left", False, 101.0)
    core.refresh(101.0)
    assert not core.touch_on and core.retreat_arm is None


def test_touch_on_again_ends_the_retreat_and_off_without_a_known_tcp_still_times_out():
    core = touch_core()
    core.set_touch(False, 101.0)  # no TCP known: window by time only
    assert core.retreat_arm == "left" and core.retreat_origin is None
    assert core.set_touch(True, 102.0)[0]
    assert core.retreat_arm is None and core.touch_on


def test_retreat_acm_also_allows_link6_and_wrist_camera_but_not_link4_5():
    names = [f"arm_left_{l}" for l in ("link4", "link5", "link6", "knife_c_blade", "knife_c_wrist_camera")] + ["table"]
    rows = [[False] * len(names) for _ in names]
    idx = {k: i for i, k in enumerate(names)}
    _, strict = allow_touch_pairs(names, rows, "left", [])
    _, retreat = allow_touch_pairs(names, rows, "left", [], retreat=True)
    for k in ("arm_left_knife_c_blade",):
        assert strict[idx[k]][idx["table"]] and retreat[idx[k]][idx["table"]]
    for k in ("arm_left_link6", "arm_left_knife_c_wrist_camera"):
        assert not strict[idx[k]][idx["table"]] and retreat[idx[k]][idx["table"]]
    for k in ("arm_left_link4", "arm_left_link5"):
        assert not retreat[idx[k]][idx["table"]]
