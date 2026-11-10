#!/usr/bin/env python3
"""Dual-arm task behaviour tree: H-cut the carton -> open & hold the lid -> pick item to shelf.

First, minimal runnable version (see README.md in this folder). Only the H-cut stage talks to
real ROS services; the lid and pick/shelve stages are NotImplementedStub placeholders.

Needs the dual-arm stack running, e.g. (mock hardware, no real robot):
    ros2 launch igus_rebel_moveit_config dual_arm_rig_isaac.launch.py \\
        hardware_protocol:=mock_hardware use_sim_time:=false

Examples:
    python3 scripts/task_tree/run_task_tree.py --offline-snapshot   # no camera needed
    python3 scripts/task_tree/run_task_tree.py                      # needs the camera (Isaac Sim)
    python3 scripts/task_tree/run_task_tree.py --render             # draw the tree, no ROS

Execution is plan-only unless BOTH --execute and --i-understand-this-moves-the-real-robot are
given, and even then box_h_cut_executor itself refuses unless its own execution_enabled
parameter is true (this script never changes that parameter).
"""
import argparse
import operator
import sys
import time
from pathlib import Path

import py_trees
import py_trees_ros
import rclpy
from compal_box_msgs.srv import CaptureBoxSnapshot
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behaviours import (  # noqa: E402
    CheckResponseSuccess,
    DeriveSnapshotPath,
    LogAndSucceed,
    NotImplementedStub,
    SetRemoteParameter,
    WriteFixtureSnapshot,
)

PERCEPTION_NODE = "/box_perception"
EXECUTOR_NODE = "/box_h_cut_executor"
SNAPSHOT_DIR = Path("~/dual_arm_box_snapshots").expanduser()   # dual_arm_box_perception.yaml
FIXTURE_FILE = SNAPSHOT_DIR / "offline_fixture.json"
TICK_PERIOD_MS = 500


def service_call(name, srv_type, service, response_key):
    """Send an empty (constant) request and store the response on the blackboard.

    wait_for_server_timeout_sec=0: don't block tree setup; if the server is missing when the
    behaviour starts, it returns FAILURE (main() checks the required servers up front anyway).
    """
    return py_trees_ros.service_clients.FromConstant(
        name=name,
        service_type=srv_type,
        service_name=service,
        service_request=srv_type.Request(),
        key_response=response_key,
        wait_for_server_timeout_sec=0.0,
    )


def is_true(name, key):
    """Condition leaf: SUCCESS if blackboard[key] is True, else FAILURE."""
    return py_trees.behaviours.CheckBlackboardVariableValue(
        name=name,
        check=py_trees.common.ComparisonExpression(variable=key, value=True, operator=operator.eq),
    )


def build_h_cut():
    # --- 1. get a carton snapshot: offline fixture, or the camera (retried) ---
    offline = py_trees.composites.Sequence("OfflineFixture", memory=True, children=[
        is_true("OfflineMode?", "offline"),
        WriteFixtureSnapshot("WriteFixtureSnapshot", FIXTURE_FILE),
    ])
    live = py_trees.decorators.Retry("Retry x3", num_failures=3, child=py_trees.composites.Sequence(
        "LiveCapture", memory=True, children=[
            service_call("CaptureSnapshot", CaptureBoxSnapshot,
                         f"{PERCEPTION_NODE}/capture_snapshot", "capture_response"),
            CheckResponseSuccess("CheckCapture", "capture_response", "capture_message"),
            DeriveSnapshotPath("DeriveSnapshotPath", "capture_response", SNAPSHOT_DIR),
        ]))
    get_snapshot = py_trees.composites.Selector("GetSnapshot", memory=True, children=[offline, live])

    # --- 2. plan (replaces the manual `ros2 param set ... external_snapshot_file`) ---
    set_param = SetRemoteParameter("SetExecutorSnapshotParam", EXECUTOR_NODE,
                                   "external_snapshot_file", "snapshot_file")
    plan = service_call("PlanHCut", Trigger, f"{EXECUTOR_NODE}/plan_h_cut", "plan_response")
    check_plan = CheckResponseSuccess("CheckPlan", "plan_response", "plan_message")

    # --- 3. execute only when explicitly allowed; otherwise record "plan only" ---
    execute = py_trees.composites.Sequence("Execute", memory=True, children=[
        is_true("ExecutionAllowed?", "execute_allowed"),
        service_call("ExecuteHCut", Trigger, f"{EXECUTOR_NODE}/execute_planned_h_cut", "execute_response"),
        CheckResponseSuccess("CheckExecute", "execute_response", "execute_message"),
    ])
    skip = LogAndSucceed("SkipExecution", "plan only - execution not requested", "execute_message")
    execute_or_skip = py_trees.composites.Selector("ExecuteOrSkip", memory=True, children=[execute, skip])

    return py_trees.composites.Sequence("HCut", memory=True, children=[
        get_snapshot, set_param, plan, check_plan, execute_or_skip])


def build_tree():
    open_lid = py_trees.composites.Sequence("OpenAndHoldLid", memory=True, children=[
        NotImplementedStub("OpenLidFlaps"),
        NotImplementedStub("HoldLid"),
    ])
    pick_and_shelve = py_trees.composites.Sequence("PickAndShelve", memory=True, children=[
        NotImplementedStub("PickItem"),
        NotImplementedStub("PlaceOnShelf"),
    ])
    return py_trees.composites.Sequence("CutOpenShelve", memory=True, children=[
        build_h_cut(), open_lid, pick_and_shelve])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline-snapshot", action="store_true",
                        help="skip the camera; use a snapshot of the USD scene's carton")
    parser.add_argument("--execute", action="store_true",
                        help="call execute_planned_h_cut after a successful plan (needs the next flag too)")
    parser.add_argument("--i-understand-this-moves-the-real-robot", action="store_true",
                        dest="i_understand")
    parser.add_argument("--render", action="store_true",
                        help="write the tree diagram (dot/png/svg) to scripts/task_tree/ and exit")
    parser.add_argument("--service-wait", type=float, default=10.0,
                        help="seconds to wait for the required ROS services before giving up")
    args = parser.parse_args()
    if args.execute != args.i_understand:
        parser.error("--execute and --i-understand-this-moves-the-real-robot must be given together")
    return args


def wait_for_services(node, services, timeout):
    """Return the services that are still missing after `timeout` seconds."""
    clients = {name: node.create_client(srv_type, name) for name, srv_type in services}
    deadline = time.monotonic() + timeout
    missing = list(clients)
    while missing and time.monotonic() < deadline:
        missing = [name for name, client in clients.items() if not client.wait_for_service(timeout_sec=0.5)]
    for client in clients.values():
        node.destroy_client(client)
    return missing


def print_summary(root, blackboard):
    print("\n" + "=" * 70)
    print(py_trees.display.unicode_tree(root, show_status=True))
    print("=" * 70)
    for stage in root.children:
        print(f"  {stage.name:16s} {stage.status.value}")
    for key in ("snapshot_file", "capture_message", "plan_message", "execute_message"):
        if blackboard.exists(key):
            print(f"  {key:16s} {blackboard.get(key)}")
    print(f"  => task {root.status.value}")
    print("=" * 70)


def main():
    args = parse_args()
    root = build_tree()

    if args.render:
        out = Path(__file__).resolve().parent
        files = py_trees.display.render_dot_tree(root, name="dual_arm_task_tree", target_directory=str(out))
        print("\n".join(files.values()))
        return 0

    rclpy.init()
    tree = py_trees_ros.trees.BehaviourTree(root)
    blackboard = py_trees.blackboard.Client(name="run_task_tree")
    for key in ("offline", "execute_allowed", "snapshot_file", "capture_message", "plan_message",
                "execute_message"):
        blackboard.register_key(key, access=py_trees.common.Access.WRITE)
    blackboard.offline = args.offline_snapshot
    blackboard.execute_allowed = args.execute and args.i_understand

    try:
        tree.setup(node_name="dual_arm_task_tree", timeout=15.0)
        required = [(f"{EXECUTOR_NODE}/plan_h_cut", Trigger)]
        if not args.offline_snapshot:
            required.append((f"{PERCEPTION_NODE}/capture_snapshot", CaptureBoxSnapshot))
        missing = wait_for_services(tree.node, required, args.service_wait)
        if missing:
            print(f"ERROR: services not available after {args.service_wait:.0f}s: {missing}\n"
                  "Is dual_arm_rig_isaac.launch.py running (with box_h_cut_executor / box_perception)?",
                  file=sys.stderr)
            return 1

        done = rclpy.task.Future()

        def stop_when_finished(behaviour_tree):
            if behaviour_tree.root.status != py_trees.common.Status.RUNNING and not done.done():
                done.set_result(behaviour_tree.root.status)

        tree.tick_tock(period_ms=TICK_PERIOD_MS, post_tick_handler=stop_when_finished)
        rclpy.spin_until_future_complete(tree.node, done)
        print_summary(root, blackboard)
        return 0 if done.result() == py_trees.common.Status.SUCCESS else 1
    except KeyboardInterrupt:
        return 130
    finally:
        tree.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
