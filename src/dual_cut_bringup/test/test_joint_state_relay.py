import importlib.util
import math
import os
import struct

import pytest
from sensor_msgs.msg import JointState

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "joint_state_relay.py")
_spec = importlib.util.spec_from_file_location("joint_state_relay", _SCRIPT)
relay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay)


def dual_message(order=None, drop=(), velocity=True, effort=True):
    names = [f"arm_{arm}_joint{n}" for arm in ("left", "right") for n in range(1, 7)]
    if order is not None:
        names = [names[i] for i in order]
    names = [n for n in names if n not in drop]
    msg = JointState()
    msg.header.stamp.sec, msg.header.stamp.nanosec = 123, 456789
    msg.header.frame_id = "dual"
    msg.name = names
    # Awkward doubles that any float32 / arithmetic round trip would change.
    msg.position = [math.pi * (i + 1) / 7.0 + 1e-17 * i for i in range(len(names))]
    msg.velocity = [0.1 * (i + 1) / 3.0 for i in range(len(names))] if velocity else []
    msg.effort = [-0.7 * (i + 1) / 11.0 for i in range(len(names))] if effort else []
    return msg


def bits(x):
    return struct.pack("<d", x)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_mapping_is_bit_exact(arm):
    msg = dual_message()
    out, reason = relay.map_joint_state(msg, arm)
    assert reason is None
    assert list(out.name) == [f"joint{n}" for n in range(1, 7)]
    for single_name, position, velocity, effort in zip(out.name, out.position, out.velocity, out.effort):
        i = msg.name.index(f"arm_{arm}_{single_name}")
        assert bits(position) == bits(msg.position[i])
        assert bits(velocity) == bits(msg.velocity[i])
        assert bits(effort) == bits(msg.effort[i])
    assert out.header.stamp.sec == 123 and out.header.stamp.nanosec == 456789
    assert out.header.frame_id == "dual"


def test_arms_do_not_mix():
    msg = dual_message()
    left, _ = relay.map_joint_state(msg, "left")
    right, _ = relay.map_joint_state(msg, "right")
    assert list(left.position) == list(msg.position[0:6])
    assert list(right.position) == list(msg.position[6:12])


def test_shuffled_order_still_maps_by_name():
    order = [11, 3, 7, 0, 9, 5, 1, 10, 2, 8, 4, 6]
    base = dual_message()
    msg = dual_message()
    msg.name = [base.name[i] for i in order]
    msg.position = [base.position[i] for i in order]
    msg.velocity = [base.velocity[i] for i in order]
    msg.effort = [base.effort[i] for i in order]
    out, reason = relay.map_joint_state(msg, "right")
    assert reason is None
    reference, _ = relay.map_joint_state(base, "right")
    assert list(out.name) == list(reference.name)
    assert list(out.position) == list(reference.position)
    assert list(out.velocity) == list(reference.velocity)
    assert list(out.effort) == list(reference.effort)


def test_extra_joints_are_ignored():
    msg = dual_message()
    msg.name = list(msg.name) + ["gripper"]
    msg.position = list(msg.position) + [0.5]
    msg.velocity = list(msg.velocity) + [0.0]
    msg.effort = list(msg.effort) + [0.0]
    out, reason = relay.map_joint_state(msg, "left")
    assert reason is None and len(out.name) == 6


def test_missing_joint_drops_message():
    out, reason = relay.map_joint_state(dual_message(drop=("arm_left_joint4",)), "left")
    assert out is None and "arm_left_joint4" in reason
    # The other arm is unaffected by the left arm's missing joint.
    out, reason = relay.map_joint_state(dual_message(drop=("arm_left_joint4",)), "right")
    assert reason is None and len(out.position) == 6


def test_empty_and_other_arm_only_messages_drop():
    out, reason = relay.map_joint_state(JointState(), "left")
    assert out is None
    msg = dual_message(drop=tuple(f"arm_left_joint{n}" for n in range(1, 7)))
    out, reason = relay.map_joint_state(msg, "left")
    assert out is None


def test_array_length_mismatch_drops():
    msg = dual_message()
    msg.position = list(msg.position)[:-1]
    assert relay.map_joint_state(msg, "left")[0] is None
    msg = dual_message()
    msg.velocity = list(msg.velocity)[:5]
    assert relay.map_joint_state(msg, "left")[0] is None


def test_duplicate_joint_drops():
    msg = dual_message()
    msg.name = list(msg.name) + ["arm_left_joint1"]
    msg.position = list(msg.position) + [9.0]
    msg.velocity = list(msg.velocity) + [0.0]
    msg.effort = list(msg.effort) + [0.0]
    out, reason = relay.map_joint_state(msg, "left")
    assert out is None and "duplicate" in reason


def test_missing_velocity_effort_stay_empty():
    out, reason = relay.map_joint_state(dual_message(velocity=False, effort=False), "left")
    assert reason is None and len(out.position) == 6
    assert len(out.velocity) == 0 and len(out.effort) == 0


def test_bad_arm_rejected():
    with pytest.raises(ValueError):
        relay.mapping_for("middle")
