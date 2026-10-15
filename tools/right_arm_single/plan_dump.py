#!/usr/bin/env python3
"""Dump the executor's retained plan (read only): cut targets, start/end joint state, frame info.

usage: plan_dump.py [--node /box_h_cut_executor] [--out FILE.json]   (uses the caller's ROS env)
"""
import argparse
import json
import sys

import rclpy
from dual_cut_msgs.srv import GetPlannedSolution
from rclpy.node import Node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="/box_h_cut_executor")
    parser.add_argument("--out")
    args = parser.parse_args()
    rclpy.init()
    node = Node("plan_dump")
    client = node.create_client(GetPlannedSolution, f"{args.node}/get_planned_solution")
    if not client.wait_for_service(timeout_sec=10.0):
        print("get_planned_solution not available", file=sys.stderr)
        return 2
    r = None
    for _ in range(3):  # the first call right after discovery can be lost
        future = client.call_async(GetPlannedSolution.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
        r = future.result()
        if r is not None:
            break
    if r is None:
        print("get_planned_solution did not answer", file=sys.stderr)
        return 2
    doc = {
        "success": r.success, "message": r.message, "plan_seq": int(r.plan_seq),
        "snapshot_id": r.snapshot_id, "acquisition_stamp_ns": int(r.acquisition_stamp_ns),
        "planning_frame": r.planning_frame, "tool_link": r.tool_link, "cut_pattern": r.cut_pattern,
        "snapshot_frame_mode": r.snapshot_frame_mode, "snapshot_relabel": r.snapshot_relabel,
        "snapshot_planning_base_xyz": list(r.snapshot_planning_base_xyz),
        "snapshot_planning_base_rpy": list(r.snapshot_planning_base_rpy),
        "middle_cut_fraction": r.middle_cut_fraction,
        "cut_targets": [[p.pose.position.x, p.pose.position.y, p.pose.position.z,
                         p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z,
                         p.pose.orientation.w] for p in r.cut_targets],
        "start_state": dict(zip(r.start_state.name, r.start_state.position)),
        "end_state": dict(zip(r.end_state.name, r.end_state.position)),
        "segments": [{"stage": s.stage_name, "cut_id": s.cut_id, "kind": int(s.kind),
                      "points": len(s.trajectory.joint_trajectory.points)} for s in r.segments],
        "summary": r.summary,
    }
    text = json.dumps(doc, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if r.success else 1


if __name__ == "__main__":
    sys.exit(main())
