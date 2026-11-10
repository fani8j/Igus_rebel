#!/usr/bin/env python3
"""ROS 2 wrapper around cut_check_code/cut_check: was the tape fully cut?

Provides (node name `cut_check`):
    /cut_check/start      std_srvs/Trigger   start recording the two wrist-camera topics
    /cut_check/check_cut  compal_cut_task_msgs/action/CheckCut
                          run_folder ""  : stop recording, judge the recording
                          run_folder DIR : judge that recorded run instead (wrist.mp4 / right_wrist.mp4)

The judging is cut_check_code/cut_check/check_run.py (analyze_video.py per arm), the same tool
used on the robot recordings, so a recording written here can also be re-checked by hand:
    python3 scripts/task_tree/cut_check_code/cut_check/check_run.py ~/cut_check_runs/<run>

Examples:
    python3 cut_check_server.py                                     # real check, default topics
    python3 cut_check_server.py --ros-args -p left_topic:=/my/left/image_raw -p right_topic:=/my/right/image_raw
    python3 cut_check_server.py --ros-args -p mock_verdicts:=NOT_FULLY_CUT,FULLY_CUT   # no camera needed

Parameters:
    left_topic / right_topic   sensor_msgs/Image of the left / right wrist camera
    fps                        frame rate written into the recorded mp4 (the wrist cameras send 15)
    output_dir                 recordings + reports go to <output_dir>/<YYYYmmdd_HHMMSS>/
    require_both_arms          true: an arm without video / without a cutting pass -> NOT_CHECKED
    ignore_edge_passes         true: only top-seam passes decide (edge results are estimates)
    mock_verdicts              "" = real check. Else a comma list (FULLY_CUT, NOT_FULLY_CUT,
                               NOT_CHECKED) answered one per goal, the last one repeating;
                               nothing is recorded or analysed.
"""
import datetime
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import rclpy
from compal_cut_task_msgs.action import CheckCut
from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

CUT_CHECK_DIR = Path(__file__).resolve().parent.parent / "cut_check_code" / "cut_check"
sys.path.insert(0, str(CUT_CHECK_DIR))
import check_run  # noqa: E402  (cut_check_code/cut_check/check_run.py)

# camera name used by cut_check (file name + calibration in cut_detector.CAMERAS) -> arm label
ARMS = {"wrist": "left arm", "right_wrist": "right arm"}
VERDICTS = ("FULLY_CUT", "NOT_FULLY_CUT", "NOT_CHECKED")


class Recorder:
    """Writes one image topic to <run>/<camera>.mp4 between start() and stop()."""

    def __init__(self, camera, fps):
        self.camera = camera
        self.fps = fps
        self.lock = threading.Lock()
        self.writer = None
        self.path = None
        self.frames = 0
        self.size = None
        self.active = False

    def start(self, run_dir):
        with self.lock:
            self._close()
            self.path = run_dir / f"{self.camera}.mp4"
            self.frames, self.size, self.active = 0, None, True

    def write(self, frame):
        with self.lock:
            if not self.active:
                return
            h, w = frame.shape[:2]
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self.size = (w, h)
            if (w, h) != self.size:
                frame = cv2.resize(frame, self.size)
            self.writer.write(frame)
            self.frames += 1

    def stop(self):
        with self.lock:
            self.active = False
            self._close()
            return self.frames

    def _close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class CutCheckServer(Node):
    def __init__(self):
        super().__init__("cut_check")
        self.declare_parameter("left_topic", "/wrist_left/rgb/image_raw")
        self.declare_parameter("right_topic", "/wrist_right/rgb/image_raw")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("output_dir", "~/cut_check_runs")
        self.declare_parameter("require_both_arms", True)
        self.declare_parameter("ignore_edge_passes", False)
        self.declare_parameter("mock_verdicts", "")

        p = self.get_parameter
        self.output_dir = Path(p("output_dir").value).expanduser()
        self.mock = [v.strip().upper() for v in p("mock_verdicts").value.split(",") if v.strip()]
        bad = [v for v in self.mock if v not in VERDICTS]
        if bad:
            raise SystemExit(f"mock_verdicts: unknown {bad}, use {VERDICTS}")
        self.mock_count = 0
        self.bridge = CvBridge()
        self.recorders = {cam: Recorder(cam, float(p("fps").value)) for cam in ARMS}
        self.topics = {"wrist": p("left_topic").value, "right_wrist": p("right_topic").value}
        self.run_dir = None
        self.started_at = None
        self.busy = threading.Lock()

        images = ReentrantCallbackGroup()
        for cam, topic in self.topics.items():
            self.create_subscription(Image, topic, lambda msg, c=cam: self._on_image(c, msg),
                                     qos_profile_sensor_data, callback_group=images)
        control = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, "~/start", self._on_start, callback_group=control)
        ActionServer(self, CheckCut, "~/check_cut", execute_callback=self._on_check,
                     cancel_callback=lambda _: CancelResponse.REJECT, callback_group=ReentrantCallbackGroup())
        mode = f"MOCK verdicts {self.mock}" if self.mock else f"left={self.topics['wrist']} right={self.topics['right_wrist']}"
        self.get_logger().info(f"cut_check ready ({mode}); output in {self.output_dir}")

    # ---- recording ------------------------------------------------------------
    def _on_image(self, camera, msg):
        recorder = self.recorders[camera]
        if not recorder.active:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # wrong encoding etc.: log once in a while, keep going
            self.get_logger().error(f"{camera}: cannot convert image ({exc})", throttle_duration_sec=5.0)
            return
        recorder.write(frame)

    def _on_start(self, _request, response):
        if self.mock:
            response.success, response.message = True, "mock mode - nothing recorded"
            return response
        self.run_dir = self.output_dir / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        for recorder in self.recorders.values():
            recorder.start(self.run_dir)
        response.success, response.message = True, f"recording wrist cameras to {self.run_dir}"
        self.get_logger().info(response.message)
        return response

    def _stop_recording(self):
        """Close the videos and write clip.json (the run-folder format check_run.py reads)."""
        cameras = {}
        for cam, recorder in self.recorders.items():
            frames = recorder.stop()
            cameras[cam] = {"status": "ok" if frames else "no frames", "frames": frames, "topic": self.topics[cam]}
            if not frames and recorder.path is not None and recorder.path.exists():
                recorder.path.unlink()
        clip = {"start_local": self.started_at, "cameras": cameras, "source": "scripts/task_tree/h_cut/cut_check_server.py"}
        (self.run_dir / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
        return cameras

    # ---- the check --------------------------------------------------------------
    def _on_check(self, goal_handle):
        result = CheckCut.Result()
        if not self.busy.acquire(blocking=False):
            result.verdict, result.message = "NOT_CHECKED", "another check is still running"
            goal_handle.succeed()
            return result
        try:
            if self.mock:
                return self._mock_check(goal_handle, result)
            return self._real_check(goal_handle, result)
        except Exception as exc:  # report, don't kill the node
            self.get_logger().error(f"check failed: {exc!r}")
            result.success, result.verdict, result.message = False, "NOT_CHECKED", f"internal error: {exc!r}"
            goal_handle.succeed()
            return result
        finally:
            self.busy.release()

    def _feedback(self, goal_handle, stage, message=""):
        goal_handle.publish_feedback(CheckCut.Feedback(stage=stage, message=message))
        self.get_logger().info(f"{stage} {message}")

    def _real_check(self, goal_handle, result):
        folder = goal_handle.request.run_folder
        if folder:
            run_dir = Path(folder).expanduser().resolve()
            if any(r.active for r in self.recorders.values()):
                self._stop_recording()  # a recording was running but the caller asked for another folder
            if not any((run_dir / f"{cam}.mp4").is_file() for cam in ARMS):
                return self._done(goal_handle, result, "NOT_CHECKED",
                                  f"{run_dir}: no wrist.mp4 / right_wrist.mp4 found")
            report_root = self.output_dir / "offline_checks"   # never write into the caller's folder
            if not (run_dir / "clip.json").is_file():
                # check_run.py needs a clip.json; for a bare video folder link the videos next to one
                linked = self.output_dir / "offline_inputs" / run_dir.name
                linked.mkdir(parents=True, exist_ok=True)
                for cam in ARMS:
                    src, dst = run_dir / f"{cam}.mp4", linked / f"{cam}.mp4"
                    if src.is_file() and not dst.exists():
                        dst.symlink_to(src)
                (linked / "clip.json").write_text(json.dumps({"cameras": {}}), encoding="utf-8")
                run_dir = linked
        else:
            if self.run_dir is None or not any(r.active for r in self.recorders.values()):
                return self._done(goal_handle, result, "NOT_CHECKED",
                                  "nothing recorded: call /cut_check/start before the cut")
            run_dir = self.run_dir
            report_root = run_dir / "cut_check_report"
            cameras = self._stop_recording()
            frames = ", ".join(f"{ARMS[c]} {v['frames']} frames" for c, v in cameras.items())
            self._feedback(goal_handle, "recording stopped", frames)
            if not any(v["frames"] for v in cameras.values()):
                return self._done(goal_handle, result, "NOT_CHECKED",
                                  f"no wrist-camera frames received on {list(self.topics.values())} "
                                  "(camera topics wrong / not publishing?)", report_dir=str(run_dir))

        self._feedback(goal_handle, "analyzing", f"{run_dir} (analyze_video.py per arm, this takes a while)")
        t0 = time.monotonic()
        report = check_run.check_run(str(run_dir), str(report_root))
        report_dir = report_root / report["run"]
        verdict, message, recut = self._verdict(report)
        result.recut = recut
        result.report_json = json.dumps(report)
        return self._done(goal_handle, result, verdict, f"{message} ({time.monotonic() - t0:.0f} s)",
                          report_dir=str(report_dir))

    def _verdict(self, report):
        """check_run.py report -> (verdict, message, recut list). Stricter than check_run's own
        box verdict: with require_both_arms an arm that could not be judged makes it NOT_CHECKED."""
        require_both = self.get_parameter("require_both_arms").value
        ignore_edges = self.get_parameter("ignore_edge_passes").value
        recut, unjudged, judged = [], [], 0
        for arm, a in report["arms"].items():
            passes = [p for p in a.get("passes", []) if not (ignore_edges and p["cut"] == "edge")]
            judged_here = [p for p in passes if p["cut_properly_percent"] is not None]
            if not judged_here:
                unjudged.append(f"{arm}: {a.get('verdict', 'no cutting pass')}")
            judged += len(judged_here)
            for p in passes:
                for section in p["recut"]:
                    recut.append(f"{arm} {p['cut']}: {section}")
        if recut:
            return "NOT_FULLY_CUT", f"{len(recut)} section(s) to re-cut", recut
        if judged == 0 or (require_both and unjudged):
            return "NOT_CHECKED", "could not judge - " + "; ".join(unjudged or ["no cutting pass found"]), recut
        note = f" (not judged: {'; '.join(unjudged)})" if unjudged else ""
        return "FULLY_CUT", f"{judged} cutting pass(es), all cut properly{note}", recut

    def _mock_check(self, goal_handle, result):
        verdict = self.mock[min(self.mock_count, len(self.mock) - 1)]
        self.mock_count += 1
        self._feedback(goal_handle, "analyzing", f"[MOCK] check {self.mock_count}")
        time.sleep(1.5)
        if verdict == "NOT_FULLY_CUT":
            result.recut = ["left arm top seam: 30-45% [MOCK]"]
        result.report_json = json.dumps({"mock": True, "verdict": verdict, "recut": list(result.recut)})
        return self._done(goal_handle, result, verdict, f"[MOCK] check {self.mock_count}")

    def _done(self, goal_handle, result, verdict, message, report_dir=""):
        result.success = verdict == "FULLY_CUT"
        result.verdict, result.message, result.report_dir = verdict, message, report_dir
        text = f"{verdict}: {message}" + (f" | re-cut {list(result.recut)}" if result.recut else "")
        if result.success:  # rclpy: one call site must keep one severity, so two separate calls
            self.get_logger().info(text)
        else:
            self.get_logger().warning(text)
        goal_handle.succeed()
        return result


def main():
    rclpy.init()
    node = CutCheckServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for recorder in node.recorders.values():
            recorder.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
