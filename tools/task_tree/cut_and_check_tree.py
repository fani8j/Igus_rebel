#!/usr/bin/env python3
"""Cut-and-check behaviour tree: cut the tape -> check the carton is cut open -> SUCCESS / FAILURE.

Both sub-actions are MOCKS for now (MockActionLeaf: RUNNING for a few ticks, then a fake
result); nothing moves and no other ROS node is needed. Each sub-action is "action leaf ->
CheckResponseSuccess", so switching to a real action server only changes the leaf built in
make_cut_tape() / make_check_carton() (see README.md, "Cut-and-check tree").

If the check fails, the whole cut -> check is retried up to --max-attempts times.
The verdict ends up in blackboard `is_tape_cut`; exit code 0 = SUCCESS, 1 = FAILURE.

Examples:
    python3 scripts/task_tree/cut_and_check_tree.py                      # SUCCESS
    python3 scripts/task_tree/cut_and_check_tree.py --mock-check-fail 1  # check fails once -> re-cut -> SUCCESS
    python3 scripts/task_tree/cut_and_check_tree.py --mock-check-fail 5  # retries used up -> FAILURE
    python3 scripts/task_tree/cut_and_check_tree.py --render             # draw the tree, no ROS

--execute / --i-understand-this-moves-the-real-robot must be given together; they are recorded
on the blackboard (execute_allowed) for the future CutBox goal and do nothing in mock mode.
"""
import argparse
import sys
from pathlib import Path

import py_trees
import py_trees_ros
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behaviours import CheckResponseSuccess, MockActionLeaf  # noqa: E402

TICK_PERIOD_MS = 200


def make_cut_tape(args):
    """Leaf that cuts the tape and stores its result at `cut_result`.

    Real action later (igus_rebel_msgs/action/CutBox.action):
        goal = CutBox.Goal(execute=allowed, i_understand_this_moves_the_real_robot=allowed)
        return ActionResultToBlackboard("CutTape", CutBox, "/cut_box", goal, "cut_result")
    """
    return MockActionLeaf("CutTape", "cut_result", ticks=args.mock_ticks, fail_first=args.mock_cut_fail,
                          success_message="tape cut along seam", failure_message="cut trajectory failed")


def make_check_carton(args):
    """Leaf that checks whether the carton is cut open and stores its result at `check_result`.

    Real action later: a not-yet-defined e.g. CheckCartonCut.action whose Result has
    `bool success` / `string message`, wrapped in ActionResultToBlackboard(..., "check_result").
    """
    return MockActionLeaf("CheckCarton", "check_result", ticks=args.mock_ticks, fail_first=args.mock_check_fail,
                          success_message="carton is cut open", failure_message="tape still intact")


def build_tree(args):
    cut = py_trees.composites.Sequence("CutTape", memory=True, children=[
        make_cut_tape(args),
        CheckResponseSuccess("CutSucceeded?", "cut_result", "cut_message"),
    ])
    check = py_trees.composites.Sequence("CheckCarton", memory=True, children=[
        make_check_carton(args),
        CheckResponseSuccess("IsCartonCut?", "check_result", "check_message", flag_key="is_tape_cut"),
    ])
    cut_then_check = py_trees.composites.Sequence("CutThenCheck", memory=True, children=[cut, check])
    return py_trees.decorators.Retry(f"CutAndVerify x{args.max_attempts}", num_failures=args.max_attempts,
                                     child=cut_then_check)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="how many times to run cut -> check before giving up (1 = no retry)")
    parser.add_argument("--mock-ticks", type=int, default=5,
                        help="ticks each mock sub-action stays RUNNING")
    parser.add_argument("--mock-cut-fail", type=int, default=0,
                        help="the first N mock cuts report failure")
    parser.add_argument("--mock-check-fail", type=int, default=0,
                        help="the first N mock checks report 'tape still intact'")
    parser.add_argument("--execute", action="store_true",
                        help="allow the (future, real) cut action to move the arm; needs the next flag too")
    parser.add_argument("--i-understand-this-moves-the-real-robot", action="store_true",
                        dest="i_understand")
    parser.add_argument("--render", action="store_true",
                        help="write the tree diagram (dot/png/svg) to scripts/task_tree/ and exit")
    args = parser.parse_args()
    if args.execute != args.i_understand:
        parser.error("--execute and --i-understand-this-moves-the-real-robot must be given together")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    return args


def print_summary(root, blackboard):
    print("\n" + "=" * 70)
    print(py_trees.display.unicode_tree(root, show_status=True))
    print("=" * 70)
    for key in ("cut_message", "check_message", "is_tape_cut", "execute_allowed"):
        if blackboard.exists(key):
            print(f"  {key:16s} {blackboard.get(key)}")
    print(f"  => task {root.status.value}")
    print("=" * 70)


def main():
    args = parse_args()
    root = build_tree(args)

    if args.render:
        out = Path(__file__).resolve().parent
        files = py_trees.display.render_dot_tree(root, name="cut_and_check_tree", target_directory=str(out))
        print("\n".join(files.values()))
        return 0

    rclpy.init()
    tree = py_trees_ros.trees.BehaviourTree(root)
    blackboard = py_trees.blackboard.Client(name="cut_and_check_tree")
    for key in ("execute_allowed", "is_tape_cut", "cut_message", "check_message"):
        blackboard.register_key(key, access=py_trees.common.Access.WRITE)
    blackboard.execute_allowed = args.execute and args.i_understand
    blackboard.is_tape_cut = False

    try:
        tree.setup(node_name="cut_and_check_tree", timeout=15.0)
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
