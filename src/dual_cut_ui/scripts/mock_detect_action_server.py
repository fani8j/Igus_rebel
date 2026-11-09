#!/usr/bin/env python3
"""MOCK ONLY - a compal_box_msgs/action/DetectBox action server that stands in for /box_seam_detector/detect_box when testing
the dual-cut UI's action-mode Detect without a camera.  NOT FOR THE REAL CELL.

For each accepted goal it publishes a few feedback messages (capture / detect / geometry / snapshot), then copies the recorded
snapshot given by --source to --snapshot with ``acquisition_stamp_ns`` = now (atomic replace, fresh mtime) and returns
success.  Like the real node it rejects a goal while another one runs, honours cancel (result status "cancelled"), and does
not write anything for a ``dry_run`` goal.  --reject returns status "rejected" without writing.  With --output-dir it also
creates the run directory the UI's detection overlay looks for (like mock_detect.py).

The DetectBox interface only exists in the box-seam-detection install, so source it before starting this script:

    source /opt/ros/humble/setup.bash
    source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash
    ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=173 python3 mock_detect_action_server.py \\
        --source src/dual_cut_ui/test/data/snapshot_MOCK_ONLY.json --snapshot /tmp/mock_det/seam_3d_base_link.json

Refuses to run unless ROS_LOCALHOST_ONLY=1 and ROS_DOMAIN_ID is set explicitly and is not 11, 12 or 219..232 (the same check as
mock_camera_publisher.py), and refuses to write the real snapshot file or anything below the real capture directory (the same
check as mock_detect.py).
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(os.path.realpath(__file__))):
    if _d not in sys.path:
        sys.path.append(_d)
from mock_camera_publisher import check_environment      # noqa: E402  (domain / localhost refusal rules)
from mock_detect import refuse_reason, synth_overlay      # noqa: E402  (real snapshot / capture dir refusal, overlay)

DEFAULT_ACTION = "/box_seam_detector/detect_box"
STAGES = (("capture", "grabbing RGB-D frames"), ("detect", "locating the carton and tape seam"),
          ("geometry", "measuring the carton top"), ("snapshot", "writing the planner snapshot"))


def empty_result(**over) -> dict:
    r = {"success": False, "status": "error", "reason": "", "reasons": [], "cut_permission": "n/a", "snapshot_id": "",
         "snapshot_path": "", "capture_id": "", "seam_status": "", "attempts": 0, "elapsed_s": 0.0}
    r.update(over)
    return r


def write_snapshot(source: str, dest: str, output_dir: str = "") -> dict:
    """Copy ``source`` to ``dest`` with acquisition_stamp_ns = now.  Returns the snapshot document."""
    with open(source) as f:
        doc = json.load(f)
    doc["acquisition_stamp_ns"] = time.time_ns()
    doc["mock_restamped"] = True
    raw = (json.dumps(doc, indent=2) + "\n").encode()
    capture_id = str(doc.get("capture_id") or "mock_capture")
    if output_dir:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_mock"
        run_dir = os.path.join(output_dir, "pipeline", capture_id, run)
        os.makedirs(os.path.join(run_dir, "autolabel_v2_overlays"), exist_ok=True)
        with open(os.path.join(run_dir, "planner_snapshot.json"), "wb") as f:
            f.write(raw)
        synth_overlay(doc, os.path.join(run_dir, "autolabel_v2_overlays", f"{capture_id}_autolabel_v2.png"))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, dest)
    return doc


def run_goal(goal, args, feedback, cancelled) -> dict:
    """The whole mock detection for one goal.  ``goal`` has dry_run; ``feedback(stage, attempt, message)``;
    ``cancelled()`` is polled between stages.  Returns the result fields as a dict."""
    t0 = time.monotonic()
    for stage, message in STAGES[:3]:
        if cancelled():
            return empty_result(status="cancelled", reason="cancelled by the client", reasons=["cancelled"], attempts=1,
                                elapsed_s=time.monotonic() - t0)
        feedback(stage, 1, "MOCK: " + message)
        time.sleep(args.step_s)
    if args.reject:
        return empty_result(status="rejected", reason="MOCK: rejected on request (--reject)", reasons=["mock_reject"],
                            attempts=1, elapsed_s=time.monotonic() - t0)
    if cancelled():
        return empty_result(status="cancelled", reason="cancelled by the client", reasons=["cancelled"], attempts=1,
                            elapsed_s=time.monotonic() - t0)
    feedback(*(STAGES[3][0], 1, "MOCK: " + STAGES[3][1]))
    if getattr(goal, "dry_run", False):
        return empty_result(success=True, status="accepted", reason="MOCK: dry run, nothing written", cut_permission="allowed",
                            seam_status="ok", attempts=1, elapsed_s=time.monotonic() - t0)
    doc = write_snapshot(args.source, args.snapshot, args.output_dir)
    cap = str(doc.get("capture_id") or "mock_capture")
    return empty_result(success=True, status="accepted", reason="MOCK: snapshot written", cut_permission="allowed",
                        snapshot_id=str(doc.get("snapshot_id") or cap), snapshot_path=os.path.abspath(args.snapshot),
                        capture_id=cap, seam_status="ok", attempts=1, elapsed_s=time.monotonic() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="recorded box_h_cut_snapshot/v1 JSON to replay")
    ap.add_argument("--snapshot", required=True, help="where the snapshot is written (the UI's snapshot_file)")
    ap.add_argument("--output-dir", default="", help="optional: also create the pipeline/<capture>/<run>/ overlay directory here")
    ap.add_argument("--action", default=DEFAULT_ACTION)
    ap.add_argument("--node-name", default="mock_box_seam_detector")
    ap.add_argument("--reject", action="store_true", help="answer every goal with status rejected and write nothing")
    ap.add_argument("--step-s", type=float, default=0.5, help="seconds per stage (feedback interval)")
    a = ap.parse_args(argv)
    why = check_environment()
    if why:
        print(f"mock_detect_action_server: refusing to run: {why}", file=sys.stderr)
        return 2
    why = refuse_reason(a.snapshot, os.path.dirname(os.path.abspath(a.snapshot))) or (
        refuse_reason(a.snapshot, a.output_dir) if a.output_dir else None)
    if why:
        print(f"mock_detect_action_server: {why}", file=sys.stderr)
        return 3
    if not os.path.isfile(a.source):
        print(f"mock_detect_action_server: --source {a.source} not found", file=sys.stderr)
        return 2
    try:
        import rclpy
        from compal_box_msgs.action import DetectBox
    except ImportError as e:
        print(f"mock_detect_action_server: cannot import DetectBox ({e}); source the box-seam-detection install first:\n"
              "  source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash", file=sys.stderr)
        return 2
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    node = rclpy.create_node(a.node_name)
    lock = threading.Lock()
    busy = [False]

    def on_goal(goal):
        with lock:
            if busy[0]:
                node.get_logger().warn("goal rejected: a detection is already running")
                return GoalResponse.REJECT
            busy[0] = True
        return GoalResponse.ACCEPT

    def on_cancel(_handle):
        return CancelResponse.ACCEPT

    def execute(handle):
        goal = handle.request
        node.get_logger().info(f"goal: dry_run={goal.dry_run} max_attempts={goal.max_attempts} "
                               f"allow_needs_review={goal.allow_needs_review} allow_cut_blocked={goal.allow_cut_blocked}")

        def feedback(stage, attempt, message):
            fb = DetectBox.Feedback()
            fb.stage, fb.attempt, fb.message = stage, attempt, message
            handle.publish_feedback(fb)
        try:
            res = run_goal(goal, a, feedback, lambda: handle.is_cancel_requested)
        except Exception as e:  # noqa: BLE001
            node.get_logger().error(f"mock detection failed: {e}")
            res = empty_result(status="error", reason=str(e), reasons=["mock_exception"])
        finally:
            with lock:
                busy[0] = False
        out = DetectBox.Result()
        for k, v in res.items():
            setattr(out, k, v)
        node.get_logger().info(f"result: status={res['status']} success={res['success']} {res['reason']}")
        if res["status"] == "cancelled":
            handle.canceled()
        else:
            handle.succeed()
        return out

    group = ReentrantCallbackGroup()
    _server = ActionServer(node, DetectBox, a.action, execute_callback=execute, goal_callback=on_goal,
                           cancel_callback=on_cancel, callback_group=group)
    node.get_logger().info(f"MOCK DetectBox server on {a.action} (domain {os.environ.get('ROS_DOMAIN_ID')}), "
                           f"snapshot -> {a.snapshot}{' (--reject)' if a.reject else ''}")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
