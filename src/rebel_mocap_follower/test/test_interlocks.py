from rebel_mocap_follower.interlocks import (
    STATE_EXECUTING, STATE_WAITING_BARRIER, controller_gate, coordinator_gate, exclusivity_gate)


def test_coordinator_idle_ok():
    for state in (0, 1, 2, 5):
        assert coordinator_gate(state, 0.5, 2.0) is None


def test_coordinator_busy_holds():
    for state in (STATE_WAITING_BARRIER, STATE_EXECUTING):
        assert "coordinator is" in coordinator_gate(state, 0.1, 2.0)


def test_coordinator_silent_holds():
    assert "silent" in coordinator_gate(None, 0.0, 2.0)
    assert "silent" in coordinator_gate(0, 2.5, 2.0)
    assert "silent" in coordinator_gate(0, float("inf"), 2.0)
    assert coordinator_gate(0, 2.0, 2.0) is None


def test_controller_active_ok():
    assert controller_gate("c", "active", 0.9) is None
    assert controller_gate("c", "active", 3.0) is None


def test_controller_not_active_holds():
    assert "not active" in controller_gate("c", "inactive", 0.1)
    assert "not active" in controller_gate("c", "not loaded", 0.1)


def test_controller_stale_or_unknown_holds():
    assert "unknown" in controller_gate("c", "active", 3.1)
    assert "unknown" in controller_gate("c", None, 1e9)
    assert "unknown" in controller_gate("c", "active", float("inf"))


def test_exclusivity_only_own_servo_ok():
    own = "/mocap/arm_left/servo"
    assert exclusivity_gate("/t", [], own) is None
    assert exclusivity_gate("/t", [("/mocap/arm_left/servo", "servo_node")], own) is None


def test_exclusivity_names_other_publisher():
    own = "/mocap/arm_left/servo"
    reason = exclusivity_gate("/t", [("/mocap/arm_left/servo", "servo_node"), ("/arm_left/servo", "servo_node")], own)
    assert "/arm_left/servo/servo_node" in reason and "/mocap/arm_left/servo/servo_node" not in reason


def test_exclusivity_other_arm_servo_is_foreign():
    assert exclusivity_gate("/t", [("/mocap/arm_right/servo", "servo_node")], "/mocap/arm_left/servo")
    assert exclusivity_gate("/t", [("/", "x")], "/mocap/arm_left/servo") is not None


def test_exclusivity_prefix_not_substring():
    assert exclusivity_gate("/t", [("/mocap/arm_left/servo2", "n")], "/mocap/arm_left/servo")


def test_exclusivity_allowed_publisher():
    own = "/mocap/arm_left/servo"
    assert exclusivity_gate("/t", [("/", "dual_cut_coordinator")], own, ["/dual_cut_coordinator"]) is None
    reason = exclusivity_gate("/t", [("/", "dual_cut_coordinator"), ("/arm_left/servo", "servo_node")], own,
                              ["/dual_cut_coordinator"])
    assert "/arm_left/servo/servo_node" in reason and "dual_cut_coordinator" not in reason


def test_exclusivity_servo_relay_node_is_own():
    # servo_relay:=true: the only publisher on the JTC topic is rebel_servo_relay's node, launched inside the Servo
    # namespace, so it counts as our own publisher (no extra exclusive_allowed_publishers entry needed).
    own = "/mocap/arm_left/servo"
    assert exclusivity_gate("/t", [("/mocap/arm_left/servo", "trajectory_relay")], own) is None
    assert exclusivity_gate("/t", [("/mocap/arm_left/servo", "trajectory_relay"),
                                   ("/mocap/arm_left/servo", "servo_node")], own) is None
    # the other arm's relay (or a relay elsewhere) is a foreign publisher
    assert exclusivity_gate("/t", [("/mocap/arm_right/servo", "trajectory_relay")], own)
    assert exclusivity_gate("/t", [("/", "trajectory_relay")], own)


RELAY = "/mocap/arm_left/servo/trajectory_relay"
OWN = "/mocap/arm_left/servo"


def test_required_own_exactly_the_relay():
    assert exclusivity_gate("/t", [(OWN, "trajectory_relay")], OWN, [], RELAY) is None
    # the coordinator may still publish (its own gate), the relay stays required
    assert exclusivity_gate("/t", [(OWN, "trajectory_relay"), ("/", "dual_cut_coordinator")], OWN,
                            ["/dual_cut_coordinator"], RELAY) is None


def test_required_own_second_relay_or_servo_still_wired_holds():
    # a second relay node (any name) under the Servo namespace
    reason = exclusivity_gate("/t", [(OWN, "trajectory_relay"), (OWN, "trajectory_relay2")], OWN, [], RELAY)
    assert reason and "exactly one" in reason and "trajectory_relay2" in reason
    # same node name twice (two processes of the relay): two publishers, one path
    reason = exclusivity_gate("/t", [(OWN, "trajectory_relay"), (OWN, "trajectory_relay")], OWN, [], RELAY)
    assert reason and "exactly one" in reason
    # Servo itself still wired to the JTC topic next to the relay
    reason = exclusivity_gate("/t", [(OWN, "trajectory_relay"), (OWN, "servo_node")], OWN, [], RELAY)
    assert reason and "servo_node" in reason


def test_required_own_wrong_or_missing_relay_holds():
    assert "not running" in exclusivity_gate("/t", [], OWN, [], RELAY)
    reason = exclusivity_gate("/t", [(OWN, "servo_node")], OWN, [], RELAY)  # Servo direct, no relay
    assert reason and "exactly one" in reason
    assert exclusivity_gate("/t", [("/mocap/arm_right/servo", "trajectory_relay")], OWN, [], RELAY)  # foreign


def test_required_own_empty_keeps_old_behaviour():
    assert exclusivity_gate("/t", [(OWN, "servo_node"), (OWN, "trajectory_relay")], OWN) is None
    assert exclusivity_gate("/t", [], OWN) is None


def test_single_arm_relay_gate():
    # mocap_follower.launch.py (servo_relay:=true): own_servo_node == own_publisher_node == the relay node.
    relay = "/servo_trajectory_relay"
    assert exclusivity_gate("/jtc", [("/", "servo_trajectory_relay")], relay, [], relay) is None
    # Servo still wired to the JTC next to the relay, or alone: foreign publisher -> hold
    reason = exclusivity_gate("/jtc", [("/", "servo_trajectory_relay"), ("/", "servo_node")], relay, [], relay)
    assert reason and "/servo_node" in reason
    assert exclusivity_gate("/jtc", [("/", "servo_node")], relay, [], relay)
    # a second relay process (same node name twice) -> more than one own publisher
    reason = exclusivity_gate("/jtc", [("/", "servo_trajectory_relay"), ("/", "servo_trajectory_relay")], relay, [], relay)
    assert reason and "exactly one" in reason
    # relay missing
    assert "not running" in exclusivity_gate("/jtc", [], relay, [], relay)
    # some other node (teleop, executor) publishing on the JTC topic
    assert exclusivity_gate("/jtc", [("/", "servo_trajectory_relay"), ("/", "box_h_cut_executor")], relay, [], relay)
