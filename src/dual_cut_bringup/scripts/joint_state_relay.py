#!/usr/bin/env python3
"""Relay the dual-cell /joint_states of ONE arm as a single-arm /arm_X/joint_states.

Subscribes to the dual ``/joint_states`` (joint_state_broadcaster, names ``arm_left_joint1`` ..
``arm_right_joint6``) and republishes ``joint_states`` (relative, so inside the ``arm_X`` namespace it is
``/arm_X/joint_states``) with ``arm_X_jointN`` renamed to ``jointN`` in the order joint1..joint6. This is what
the per-arm single-arm move_group and box_h_cut_executor (unprefixed URDF) read.

Rules (checked by test/test_joint_state_relay.py):
- values are copied unchanged (no arithmetic, bit-identical doubles); the header (stamp, frame_id) is kept;
- a message missing any of the 6 joints of the arm, listing one twice, or with a position/velocity/effort array
  that does not match ``name`` is DROPPED (never partially forwarded); drops are counted and logged at most
  every ``log_period_s`` seconds;
- velocity / effort are forwarded only when the input carries them (same length as ``name``), otherwise left empty.
"""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

JOINT_SUFFIXES = tuple(f"joint{n}" for n in range(1, 7))


def mapping_for(arm):
    """{dual name: single-arm name}, e.g. arm_left_joint1 -> joint1."""
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return {f"arm_{arm}_{suffix}": suffix for suffix in JOINT_SUFFIXES}


def map_joint_state(msg, arm):
    """Return (JointState for the single-arm stack, None), or (None, reason) when the message must be dropped."""
    wanted = mapping_for(arm)
    names = list(msg.name)
    count = len(names)
    if len(msg.position) != count:
        return None, f"position has {len(msg.position)} entries for {count} names"
    for field in ("velocity", "effort"):
        size = len(getattr(msg, field))
        if size not in (0, count):
            return None, f"{field} has {size} entries for {count} names"
    index = {}
    for position_index, name in enumerate(names):
        if name in wanted:
            if name in index:
                return None, f"duplicate joint {name}"
            index[name] = position_index
    missing = [name for name in wanted if name not in index]
    if missing:
        return None, "missing " + ",".join(missing)
    out = JointState()
    out.header = msg.header
    out.name = [wanted[name] for name in wanted]
    order = [index[name] for name in wanted]
    out.position = [msg.position[i] for i in order]
    if len(msg.velocity) == count:
        out.velocity = [msg.velocity[i] for i in order]
    if len(msg.effort) == count:
        out.effort = [msg.effort[i] for i in order]
    return out, None


class JointStateRelay(Node):
    def __init__(self):
        super().__init__("joint_state_relay")
        self.arm = self.declare_parameter("arm", "left").value
        mapping_for(self.arm)  # fail at startup on a bad value
        input_topic = self.declare_parameter("input_topic", "/joint_states").value
        output_topic = self.declare_parameter("output_topic", "joint_states").value
        self.log_period = float(self.declare_parameter("log_period_s", 5.0).value)
        self.forwarded = 0
        self.dropped = 0
        self.last_reason = ""
        self.publisher = self.create_publisher(JointState, output_topic, 10)
        self.subscription = self.create_subscription(JointState, input_topic, self.on_message, 10)
        self.create_timer(self.log_period, self.report)
        self.get_logger().info(
            f"relaying {input_topic} arm_{self.arm}_jointN -> {self.publisher.topic_name} jointN")

    def on_message(self, msg):
        out, reason = map_joint_state(msg, self.arm)
        if out is None:
            self.dropped += 1
            self.last_reason = reason
            return
        self.forwarded += 1
        self.publisher.publish(out)

    def report(self):
        if self.dropped:
            self.get_logger().warning(
                f"dropped {self.dropped} joint_states message(s) (forwarded {self.forwarded}); "
                f"last reason: {self.last_reason}")
            self.dropped = 0
        elif self.forwarded == 0:
            self.get_logger().warning("no joint_states received on the dual topic yet")
        self.forwarded = 0


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = JointStateRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
