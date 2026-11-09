#!/usr/bin/env python3
"""Read-only: watch the mocap topic and print every big jump between consecutive messages.

    python3 mocap_input_steps.py [arm_left|arm_right] [threshold_rad=0.05] [seconds=30]

Each line shows the joint, the value before and after, and the time since the previous message,
so you can tell a real fast movement (big dt) from a snap/jump in the source (small dt).
"""
import math
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

prefix = (sys.argv[1] if len(sys.argv) > 1 else "arm_left") + "_"
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
names = [f"{prefix}joint{i}" for i in range(1, 7)]
TOPIC = "/dual_arm/isaac_joint_commands"

rclpy.init()
node = rclpy.create_node("mocap_input_steps")
st = {"prev": None, "t": None, "n": 0, "dts": [], "jumps": 0}


def cb(m):
    t = time.monotonic()
    d = dict(zip(m.name, m.position))
    if any(n not in d for n in names):
        print("message without", [n for n in names if n not in d])
        return
    q = [d[n] for n in names]
    st["n"] += 1
    if st["prev"] is not None:
        dt = t - st["t"]
        st["dts"].append(dt)
        for i, (a, b) in enumerate(zip(st["prev"], q)):
            if abs(b - a) > thr:
                st["jumps"] += 1
                print(f"{t:.3f} joint{i + 1}: {a:+.3f} -> {b:+.3f} rad (step {b - a:+.3f}, "
                      f"{math.degrees(b - a):+.1f} deg) after dt {dt * 1000:.0f} ms, "
                      f"stamp {m.header.stamp.sec}.{m.header.stamp.nanosec:09d}")
    st["prev"], st["t"] = q, t


node.create_subscription(JointState, TOPIC, cb, qos_profile_sensor_data)
time.sleep(0.5)
pubs = node.get_publishers_info_by_topic(TOPIC)
print(f"{len(pubs)} publisher(s) on {TOPIC}: " + ", ".join(f"{p.node_namespace.rstrip('/')}/{p.node_name}"
                                                           for p in pubs))
print(f"watching {dur:.0f} s, printing steps > {thr} rad ... (Ctrl-C to stop)")
end = time.monotonic() + dur
try:
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
except KeyboardInterrupt:
    pass
dts = sorted(st["dts"])
if dts:
    print(f"\n{st['n']} messages, mean rate {1.0 / (sum(dts) / len(dts)):.1f} Hz, dt median "
          f"{dts[len(dts) // 2] * 1000:.0f} ms, max {dts[-1] * 1000:.0f} ms, gaps > 50 ms: "
          f"{sum(1 for x in dts if x > 0.05)}, jumps > {thr} rad: {st['jumps']}")
else:
    print("no messages received")
node.destroy_node()
rclpy.shutdown()
