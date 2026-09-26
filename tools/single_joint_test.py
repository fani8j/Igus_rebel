#!/usr/bin/env python3
"""Supervised single-joint motion test through the joint trajectory controller.

Reads the live joint state, moves ONE joint by a small angle and back, and reports the
tracking result. Every motion needs a typed "yes". Intended for the first moves on the
real ReBeL with an operator holding the E-stop; run it against mock hardware first.

    python3 tools/single_joint_test.py --joint joint6 --delta-deg 5 --duration 6
"""

import argparse
import math
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ACTION = "/rebel_arm_trajectory_controller/follow_joint_trajectory"
MAX_DELTA_DEG = 10.0
MIN_DURATION_S = 4.0


class SingleJointTest(Node):
    def __init__(self):
        super().__init__("single_joint_test")
        self._state = None
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self._client = ActionClient(self, FollowJointTrajectory, ACTION)

    def _on_state(self, msg):
        self._state = msg

    def current_positions(self, timeout_s=5.0):
        self._state = None
        end = time.monotonic() + timeout_s
        while self._state is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._state is None:
            raise RuntimeError("no /joint_states received (same ROS_DOMAIN_ID as the driver?)")
        by_name = dict(zip(self._state.name, self._state.position))
        velocities = dict(zip(self._state.name, self._state.velocity or []))
        missing = [j for j in JOINTS if j not in by_name]
        if missing:
            raise RuntimeError(f"joint_states is missing {missing}")
        moving = [j for j in JOINTS if abs(velocities.get(j, 0.0)) > 1e-3]
        if moving:
            raise RuntimeError(f"arm is not at rest (velocity on {moving})")
        return [by_name[j] for j in JOINTS]

    def move(self, target, duration_s):
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"action server {ACTION} not available")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = list(target)
        point.velocities = [0.0] * len(JOINTS)
        point.time_from_start = Duration(
            sec=int(duration_s), nanosec=int((duration_s % 1.0) * 1e9))
        goal.trajectory.points = [point]

        send = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("goal rejected by the trajectory controller")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration_s + 10.0)
        if not result_future.done():
            handle.cancel_goal_async()
            raise RuntimeError("no result in time; goal cancelled")
        result = result_future.result().result
        return result.error_code, result.error_string


def confirm(prompt):
    try:
        return input(f"{prompt} Type 'yes' to continue: ").strip().lower() == "yes"
    except EOFError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--joint", default="joint6", choices=JOINTS)
    parser.add_argument("--delta-deg", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=6.0, help="seconds per move")
    parser.add_argument("--no-return", action="store_true", help="do not move back")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        sys.exit("refusing to run without an interactive terminal (confirmation required)")
    if not math.isfinite(args.delta_deg) or not 0.0 < abs(args.delta_deg) <= MAX_DELTA_DEG:
        sys.exit(f"--delta-deg must be non-zero and at most {MAX_DELTA_DEG} deg")
    if not math.isfinite(args.duration) or args.duration < MIN_DURATION_S:
        sys.exit(f"--duration must be at least {MIN_DURATION_S} s")

    rclpy.init()
    node = SingleJointTest()
    idx = JOINTS.index(args.joint)
    try:
        start = node.current_positions()
        target = list(start)
        target[idx] += math.radians(args.delta_deg)
        speed = abs(math.radians(args.delta_deg)) / args.duration
        print("Current joints [deg]: " + " ".join(f"{math.degrees(p):8.2f}" for p in start))
        print(f"Move {args.joint}: {math.degrees(start[idx]):.2f} -> "
              f"{math.degrees(target[idx]):.2f} deg over {args.duration:.1f} s "
              f"(avg {math.degrees(speed):.2f} deg/s); all other joints hold.")
        if not confirm("E-stop in hand and the path clear?"):
            print("Aborted, nothing sent.")
            return 1

        code, text = node.move(target, args.duration)
        reached = node.current_positions()
        err = math.degrees(reached[idx] - target[idx])
        print(f"Result: error_code={code} {text!r}; {args.joint} final error {err:+.3f} deg")
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            print("Motion did NOT succeed; not moving back. Check the driver log.")
            return 2

        if args.no_return:
            return 0
        if not confirm(f"Move {args.joint} back to {math.degrees(start[idx]):.2f} deg?"):
            print("Left at the target position.")
            return 0
        back = list(reached)
        back[idx] = start[idx]
        code, text = node.move(back, args.duration)
        reached = node.current_positions()
        err = math.degrees(reached[idx] - start[idx])
        print(f"Return: error_code={code} {text!r}; {args.joint} final error {err:+.3f} deg")
        return 0 if code == FollowJointTrajectory.Result.SUCCESSFUL else 2
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 3
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
