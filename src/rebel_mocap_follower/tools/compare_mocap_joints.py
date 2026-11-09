#!/usr/bin/env python3
"""Read-only: print mocap target vs real /joint_states per joint (no commands are sent).

    python3 compare_mocap_joints.py [arm_left|arm_right]

The prefix selects the mocap joints (arm_left_joint1..6). The robot joints are read as <prefix>_joint1..6 when
/joint_states has them (dual-arm table cell), else as joint1..6 (single arm).
"""
import math
import sys

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

prefix = (sys.argv[1] if len(sys.argv) > 1 else "arm_left") + "_"
rclpy.init()
node = rclpy.create_node("compare_mocap_joints")
data = {}
node.create_subscription(JointState, "/dual_arm/isaac_joint_commands",
                         lambda m: data.__setitem__("mocap", dict(zip(m.name, m.position))),
                         qos_profile_sensor_data)
node.create_subscription(JointState, "/joint_states",
                         lambda m: data.__setitem__("robot", dict(zip(m.name, m.position))), 10)
end = node.get_clock().now().nanoseconds + 5e9
while rclpy.ok() and len(data) < 2 and node.get_clock().now().nanoseconds < end:
    rclpy.spin_once(node, timeout_sec=0.1)
if len(data) < 2:
    print("missing:", [k for k in ("mocap", "robot") if k not in data])
else:
    print(f"{'joint':8s} {'robot rad':>10s} {'mocap rad':>10s} {'diff rad':>9s} {'robot deg':>10s} {'mocap deg':>10s}")
    for i in range(1, 7):
        r = data["robot"].get(f"{prefix}joint{i}", data["robot"].get(f"joint{i}"))
        m = data["mocap"].get(f"{prefix}joint{i}")
        if r is None or m is None:
            print(f"joint{i}: missing (robot {r}, mocap {m})")
            continue
        print(f"joint{i:<3d} {r:10.3f} {m:10.3f} {m - r:9.3f} {math.degrees(r):10.1f} {math.degrees(m):10.1f}"
              + ("   <-- > 0.5" if abs(m - r) > 0.5 else ""))
node.destroy_node()
rclpy.shutdown()
