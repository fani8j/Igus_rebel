"""Fake mocap source for mock testing: publishes JointState on /mocap/joint_command.

The centre pose is the first /joint_states received. Modes:
  sine     smooth sine on joint1/joint4/joint6 (--amp rad, --freq Hz), continuous
  jump     sine, then at --event-time a step of --jump rad on joint1 that stays
  nan      sine, with NaN positions for 0.3 s at --event-time
  stop     sine, stops publishing at --event-time
  far      centre offset by --jump rad on joint1 from the start (engage refusal test)
  badnames sine, but message lacks joint6
  limit    sine, but joint1 set outside the URDF limits at --event-time (stays)

--dual publishes the real dual-arm message shape on /dual_arm/isaac_joint_commands: arm_left_joint1..6 +
arm_right_joint1..6 plus extra junk joints (lift, slider, gripper) that the followers must ignore; the sine runs on
both arms. --event-arm left|right|both (default both) selects which arm the mode's event (jump, nan, stop, far,
badnames, limit) hits, so the other arm can be shown to keep following.
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", default="sine",
                    choices=["sine", "jump", "nan", "stop", "far", "badnames", "limit"])
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--amp", type=float, default=0.1)
    ap.add_argument("--freq", type=float, default=0.1)
    ap.add_argument("--event-time", type=float, default=6.0)
    ap.add_argument("--jump", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0, help="exit after this many s (0 = run forever)")
    ap.add_argument("--topic", default=None,
                    help="default /mocap/joint_command, or /dual_arm/isaac_joint_commands with --dual")
    ap.add_argument("--dual", action="store_true", help="publish arm_left_*/arm_right_* names + junk joints")
    ap.add_argument("--event-arm", default="both", choices=["left", "right", "both"])
    opts, ros_args = ap.parse_known_args(sys.argv[1:] if args is None else args)
    topic = opts.topic or ("/dual_arm/isaac_joint_commands" if opts.dual else "/mocap/joint_command")
    prefixes = ["arm_left_", "arm_right_"] if opts.dual else [""]
    event_prefixes = prefixes if opts.event_arm == "both" or not opts.dual else [f"arm_{opts.event_arm}_"]
    names_all = [f"{pf}joint{i}" for pf in prefixes for i in range(1, 7)]
    junk = {"lift_joint": 0.0, "slider_joint": 0.0, "gripper_left_joint": 0.01} if opts.dual else {}
    rclpy.init(args=ros_args)
    node = rclpy.create_node("fake_mocap_publisher")
    center = {}

    def on_js(msg):
        if not center:
            center.update(zip(msg.name, msg.position))

    node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
    pub = node.create_publisher(JointState, topic, qos_profile_sensor_data)
    while rclpy.ok() and not all(n in center for n in names_all):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not rclpy.ok():
        return
    node.get_logger().info(f"center: {[round(center[n], 3) for n in names_all]} mode={opts.mode}")
    if opts.mode == "far":
        for pf in event_prefixes:
            center[f"{pf}joint1"] += opts.jump
    t0 = time.monotonic()
    period = 1.0 / opts.rate
    next_t = t0
    while rclpy.ok():
        t = time.monotonic() - t0
        if opts.duration and t > opts.duration:
            break
        pos = dict(center)
        s = math.sin(2 * math.pi * opts.freq * t)
        for pf in prefixes:
            for name, k in (("joint1", 1.0), ("joint4", 0.8), ("joint6", 1.2)):
                pos[pf + name] = center[pf + name] + k * opts.amp * s
        names = list(names_all)
        publish = True
        if t >= opts.event_time:
            for pf in event_prefixes:
                if opts.mode == "jump":
                    pos[pf + "joint1"] += opts.jump
                elif opts.mode == "nan" and t < opts.event_time + 0.3:
                    pos[pf + "joint1"] = float("nan")
                elif opts.mode == "stop":
                    if len(event_prefixes) == len(prefixes):
                        publish = False
                    else:
                        names = [n for n in names if not n.startswith(pf)]
                elif opts.mode == "limit":
                    pos[pf + "joint1"] = 10.0
        if opts.mode == "badnames":
            names = [n for n in names if not any(n == f"{pf}joint6" for pf in event_prefixes)]
        if publish:
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = names + list(junk)
            msg.position = [pos[n] for n in names] + list(junk.values())
            pub.publish(msg)
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
