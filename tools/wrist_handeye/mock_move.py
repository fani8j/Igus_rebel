#!/usr/bin/env python3
"""MOCK-ONLY helper: send the arm of a mock stack to a joint configuration (plain joint trajectory, no collision check).
Refuses unless ROS_DOMAIN_ID is set, != 11 and the hardware is mock_components/GenericSystem.
usage: ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 python3 mock_move.py <seconds> q1 q2 q3 q4 q5 q6   (radians)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ros_io
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

dom = os.environ.get("ROS_DOMAIN_ID", "")
if not dom.isdigit() or dom == "11":
    sys.exit("refusing: explicit non-11 ROS_DOMAIN_ID required")
dur = float(sys.argv[1]); q = [float(v) for v in sys.argv[2:8]]
io = ros_io.IO()
try:
    comps = io.hardware_components()
    if not comps or any(c != "mock_components/GenericSystem" for _, c, _ in comps):
        sys.exit("refusing: hardware is not mock")
    pub = io.node.create_publisher(JointTrajectory, "/rebel_arm_trajectory_controller/joint_trajectory", 1)
    time.sleep(1.0)
    m = JointTrajectory(); m.joint_names = ros_io.JOINTS
    p = JointTrajectoryPoint(); p.positions = q; p.time_from_start.sec = int(dur); p.time_from_start.nanosec = int((dur % 1) * 1e9)
    m.points = [p]; pub.publish(m)
    time.sleep(dur + 2.0)
    io.wait_standstill(1.0, 0.002, timeout=30)
    print("now at", np.round(io.wait_joint_states(), 3).tolist())
finally:
    io.cleaning = True; io.shutdown()
