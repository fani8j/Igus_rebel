#!/usr/bin/env python3
"""H-cut behaviour tree: cut the carton tape (H pattern) -> check it is fully cut -> re-cut if not.

    -^- HCutAndVerify xN                 retry only when the check says NOT_FULLY_CUT
        {-} Attempt
            --> BeginAttempt              attempt += 1, previous check report -> re-cut hint
            {-} StartCutCheck             /cut_check/start: start recording the wrist cameras
            {-} HCut                      YOUR action server (h_cut_interface.py)
            {-} CheckCut                  /cut_check/check_cut: judge the recording
                                          -> blackboard is_tape_cut

Needs three ROS processes (see README.md in this folder):
    python3 scripts/task_tree/h_cut/mock_h_cut_server.py      # or YOUR H-cut action server
    python3 scripts/task_tree/h_cut/cut_check_server.py        # tape check (cut_check_code)
    python3 scripts/task_tree/h_cut/h_cut_tree.py              # this tree

Examples:
    python3 scripts/task_tree/h_cut/h_cut_tree.py                            # plan only
    python3 scripts/task_tree/h_cut/h_cut_tree.py --check-run-folder RUN_DIR # judge a recorded run
    python3 scripts/task_tree/h_cut/h_cut_tree.py --render                   # draw the tree, no ROS

The arm can move only if BOTH --execute and --i-understand-this-moves-the-real-robot are given
(then make_goal() in h_cut_interface.py sets the goal's safety flags). Exit code 0 = tape fully cut.
"""
import argparse
import operator
import sys
import time
from pathlib import Path

import py_trees
import py_trees_ros
import rclpy
import rclpy.action
import rclpy.signals
from std_srvs.srv import Trigger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import h_cut_interface as hcut  # noqa: E402
from behaviours import CheckResponseSuccess  # noqa: E402
from compal_cut_task_msgs.action import CheckCut  # noqa: E402
from h_cut_behaviours import ActionLeaf, BeginAttempt, RetryWhenNotFullyCut  # noqa: E402

CUT_CHECK_NODE = "/cut_check"
TICK_PERIOD_MS = 200


def check_goal(ctx):
    """CheckCut goal: --check-run-folder > folder reported by the H-cut server > own recording."""
    folder = ctx.check_run_folder
    if not folder and ctx.h_cut_result is not None and ctx.h_cut_result.raw is not None:
        folder = hcut.run_folder_from_result(ctx.h_cut_result.raw) or ""
    return CheckCut.Goal(run_folder=folder)


def check_interpret(result):
    recut = "; re-cut " + ", ".join(result.recut) if result.recut else ""
    return result.success, f"{result.verdict}: {result.message}{recut}"


def build_tree(max_attempts):
    start_check = py_trees.composites.Sequence("StartCutCheck", memory=True, children=[
        py_trees_ros.service_clients.FromConstant(
            name="StartRecording", service_type=Trigger, service_name=f"{CUT_CHECK_NODE}/start",
            service_request=Trigger.Request(), key_response="start_response",
            wait_for_server_timeout_sec=0.0),
        CheckResponseSuccess("RecordingStarted?", "start_response", "start_message"),
    ])
    h_cut = py_trees.composites.Sequence("HCut", memory=True, children=[
        ActionLeaf("HCutAction", hcut.ACTION_TYPE, hcut.ACTION_NAME, hcut.make_goal, hcut.interpret_result,
                   "h_cut_result", read_keys=("execute_allowed", "attempt", "recut_hint_json"),
                   feedback_text=hcut.feedback_text),
        CheckResponseSuccess("HCutSucceeded?", "h_cut_result", "h_cut_message"),
    ])
    check = py_trees.composites.Sequence("CheckCut", memory=True, children=[
        ActionLeaf("CheckCutAction", CheckCut, f"{CUT_CHECK_NODE}/check_cut", check_goal, check_interpret,
                   "check_result", read_keys=("check_run_folder", "h_cut_result"),
                   feedback_text=lambda fb: f"{fb.stage} {fb.message}"),
        CheckResponseSuccess("IsTapeFullyCut?", "check_result", "check_message", flag_key="is_tape_cut"),
    ])
    attempt = py_trees.composites.Sequence("Attempt", memory=True, children=[
        BeginAttempt("BeginAttempt"), start_check, h_cut, check])
    return RetryWhenNotFullyCut(f"HCutAndVerify x{max_attempts}", child=attempt, num_failures=max_attempts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="cut -> check at most this many times (1 = never re-cut)")
    parser.add_argument("--check-run-folder", default="",
                        help="judge this recorded run folder (wrist.mp4 / right_wrist.mp4) instead of "
                             "the live recording - for testing the check on old runs")
    parser.add_argument("--execute", action="store_true",
                        help="let the H-cut server move the arm (needs the next flag too)")
    parser.add_argument("--i-understand-this-moves-the-real-robot", action="store_true", dest="i_understand")
    parser.add_argument("--server-wait", type=float, default=10.0,
                        help="seconds to wait for the action servers / services before giving up")
    parser.add_argument("--render", action="store_true", help="write h_cut_tree.png/.svg/.dot here and exit")
    args = parser.parse_args()
    if args.execute != args.i_understand:
        parser.error("--execute and --i-understand-this-moves-the-real-robot must be given together")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.check_run_folder:
        args.check_run_folder = str(Path(args.check_run_folder).expanduser().resolve())
    return args


def wait_for_servers(node, timeout):
    """Return the names of the servers still missing after `timeout` seconds."""
    actions = {hcut.ACTION_NAME: rclpy.action.ActionClient(node, hcut.ACTION_TYPE, hcut.ACTION_NAME),
               f"{CUT_CHECK_NODE}/check_cut": rclpy.action.ActionClient(node, CheckCut, f"{CUT_CHECK_NODE}/check_cut")}
    services = {f"{CUT_CHECK_NODE}/start": node.create_client(Trigger, f"{CUT_CHECK_NODE}/start")}
    deadline = time.monotonic() + timeout
    while True:
        missing = [n for n, c in actions.items() if not c.server_is_ready()]
        missing += [n for n, c in services.items() if not c.service_is_ready()]
        if not missing or time.monotonic() > deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    for client in actions.values():
        client.destroy()
    for client in services.values():
        node.destroy_client(client)
    return missing


def print_summary(root, bb):
    print("\n" + "=" * 78)
    print(py_trees.display.unicode_tree(root, show_status=True))
    print("=" * 78)
    print(f"  attempts         {bb.attempt}")
    for key in ("start_message", "h_cut_message", "check_message"):
        if bb.exists(key):
            print(f"  {key:16s} {bb.get(key)}")
    result = bb.check_result
    raw = getattr(result, "raw", None)
    if raw is not None and raw.report_dir:
        print(f"  report_dir       {raw.report_dir}")
    print(f"  is_tape_cut      {bb.is_tape_cut}")
    print(f"  => H-cut task {root.status.value}"
          + ("  (tape fully cut, OK to open the lid)" if root.status == py_trees.common.Status.SUCCESS else ""))
    print("=" * 78)


def main():
    args = parse_args()
    root = build_tree(args.max_attempts)

    if args.render:
        files = py_trees.display.render_dot_tree(root, name="h_cut_tree", target_directory=str(HERE))
        print("\n".join(files.values()))
        return 0

    # Keep rclpy's context alive on Ctrl-C (its default SIGINT handler shuts it down first), so the
    # KeyboardInterrupt handler below can still send the cancel for a running H-cut goal.
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    tree = py_trees_ros.trees.BehaviourTree(root)
    bb = py_trees.blackboard.Client(name="h_cut_tree")
    initial = {"execute_allowed": args.execute and args.i_understand, "attempt": 0, "recut_hint_json": "",
               "check_run_folder": args.check_run_folder, "h_cut_result": None, "check_result": None,
               "is_tape_cut": False}
    for key, value in initial.items():
        bb.register_key(key, access=py_trees.common.Access.WRITE)
        bb.set(key, value)
    for key in ("start_message", "h_cut_message", "check_message"):  # set by the stages that ran
        bb.register_key(key, access=py_trees.common.Access.READ)

    try:
        tree.setup(node_name="h_cut_tree", timeout=15.0)
        missing = wait_for_servers(tree.node, args.server_wait)
        if missing:
            print(f"ERROR: not available after {args.server_wait:.0f}s: {missing}\n"
                  "Is your H-cut action server running (h_cut_interface.ACTION_NAME = "
                  f"{hcut.ACTION_NAME!r})? Is cut_check_server.py running?", file=sys.stderr)
            return 1
        mode = "EXECUTE (the arm may move)" if bb.execute_allowed else "plan only (the arm must not move)"
        print(f"H-cut tree: {hcut.ACTION_NAME} [{hcut.ACTION_TYPE.__name__}], max {args.max_attempts} attempts, {mode}")

        done = rclpy.task.Future()

        def stop_when_finished(behaviour_tree):
            if behaviour_tree.root.status != py_trees.common.Status.RUNNING and not done.done():
                done.set_result(behaviour_tree.root.status)

        tree.tick_tock(period_ms=TICK_PERIOD_MS, post_tick_handler=stop_when_finished)
        rclpy.spin_until_future_complete(tree.node, done)
        print_summary(root, bb)
        return 0 if done.result() == py_trees.common.Status.SUCCESS else 1
    except KeyboardInterrupt:
        if root.status == py_trees.common.Status.RUNNING:
            print("\nCtrl-C: stopping the tree and cancelling the running goal", file=sys.stderr)
            if getattr(tree, "timer", None) is not None:
                tree.timer.cancel()   # no more ticks: otherwise the tree would start a new attempt
            root.stop(py_trees.common.Status.INVALID)   # ActionLeaf.terminate() sends the cancel
            for _ in range(10):
                rclpy.spin_once(tree.node, timeout_sec=0.05)
        return 130
    finally:
        tree.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
