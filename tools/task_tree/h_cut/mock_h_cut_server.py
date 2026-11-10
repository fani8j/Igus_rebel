#!/usr/bin/env python3
"""Stand-in for the real H-cut action server (compal_cut_task_msgs/action/HCut on /h_cut).

NOTHING MOVES - it only pretends to cut, so the behaviour tree and cut_check_server.py can be run
end to end before the real server is plugged in. It is also a minimal example of an HCut server:
copy its structure (goal checks, feedback, success=false instead of abort) into your own.

    python3 mock_h_cut_server.py                                   # every cut "succeeds" after 3 s
    python3 mock_h_cut_server.py --ros-args -p fail_attempts:=1    # first cut reports PLANNING_FAILED
    python3 mock_h_cut_server.py --ros-args -p replay_run_folder:=/path/to/20261020_093201

replay_run_folder: while "cutting", publish that run's wrist.mp4 / right_wrist.mp4 on the
wrist-camera topics in real time, so cut_check_server.py records and judges real cut footage.
"""
import time
from pathlib import Path

import cv2
import rclpy
from compal_cut_task_msgs.action import HCut
from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

STAGES = ["perception", "planning", "left arm cutting", "right arm cutting", "retract"]


class MockHCutServer(Node):
    def __init__(self):
        super().__init__("mock_h_cut_server")
        self.declare_parameter("duration_s", 3.0)
        self.declare_parameter("fail_attempts", 0)
        self.declare_parameter("replay_run_folder", "")
        self.declare_parameter("left_topic", "/wrist_left/rgb/image_raw")
        self.declare_parameter("right_topic", "/wrist_right/rgb/image_raw")
        self.goals = 0
        self.running = False
        self.bridge = CvBridge()
        self.publishers_ = {
            "wrist": self.create_publisher(Image, self.get_parameter("left_topic").value, qos_profile_sensor_data),
            "right_wrist": self.create_publisher(Image, self.get_parameter("right_topic").value, qos_profile_sensor_data),
        }
        ActionServer(self, HCut, "/h_cut", execute_callback=self.execute,
                     goal_callback=self.on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
                     callback_group=ReentrantCallbackGroup())
        self.get_logger().warning("MOCK H-cut server on /h_cut - nothing moves")

    def on_goal(self, _goal):
        # one cut at a time; a second goal is rejected (the tree reports "goal rejected")
        return GoalResponse.REJECT if self.running else GoalResponse.ACCEPT

    def execute(self, goal_handle):
        self.running = True
        try:
            return self._execute(goal_handle)
        finally:
            self.running = False

    def _execute(self, goal_handle):
        goal = goal_handle.request
        self.goals += 1
        t0 = time.monotonic()
        result = HCut.Result(did_move=False)

        def finish(code, message):
            result.success = code == HCut.Result.OK
            result.error_code = code
            result.message = f"{message} ({time.monotonic() - t0:.1f} s)"
            goal_handle.succeed()  # expected failures: succeed() with success=false, not abort()
            self.get_logger().info(f"attempt {goal.attempt}: success={result.success} {message}")
            return result

        # Safety contract: both flags or neither.
        if goal.execute != goal.i_understand_this_moves_the_real_robot:
            return finish(HCut.Result.BAD_GOAL, "execute needs i_understand_this_moves_the_real_robot too")
        mode = "EXECUTE requested (mock: still nothing moves)" if goal.execute else "plan only"
        hint = " with re-cut hint" if goal.recut_hint_json else ""
        self.get_logger().info(f"attempt {goal.attempt}{hint}: {mode}")

        frames = self._open_replay()
        duration = self.get_parameter("duration_s").value
        start = time.monotonic()
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warning(f"attempt {goal.attempt}: cancelled by the client")
                return HCut.Result(success=False, error_code=HCut.Result.CANCELLED, message="cancelled")
            elapsed = time.monotonic() - start
            if frames is not None:
                more = self._publish_due_frames(frames, elapsed)
                progress = min(1.0, elapsed / max(frames["length_s"], 1e-3))
                if not more:
                    break
            else:
                progress = min(1.0, elapsed / max(duration, 1e-3))
                if elapsed >= duration:
                    break
            stage = STAGES[min(int(progress * len(STAGES)), len(STAGES) - 1)]
            goal_handle.publish_feedback(HCut.Feedback(stage=stage, progress=progress, message="[MOCK]"))
            time.sleep(0.02 if frames is not None else 0.2)

        if self.goals <= self.get_parameter("fail_attempts").value:
            return finish(HCut.Result.PLANNING_FAILED, "[MOCK] IK found no solution for the right-arm edge")
        return finish(HCut.Result.OK, f"[MOCK] H-cut done ({mode})")

    # ---- optional replay of a recorded run on the wrist-camera topics --------------------
    def _open_replay(self):
        folder = self.get_parameter("replay_run_folder").value
        if not folder:
            return None
        caps = {}
        for cam in self.publishers_:
            path = Path(folder).expanduser() / f"{cam}.mp4"
            if path.is_file():
                caps[cam] = cv2.VideoCapture(str(path))
        if not caps:
            self.get_logger().error(f"replay_run_folder {folder}: no wrist.mp4 / right_wrist.mp4")
            return None
        fps = {cam: cap.get(cv2.CAP_PROP_FPS) or 15.0 for cam, cap in caps.items()}
        length = max(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps[cam] for cam, cap in caps.items())
        self.get_logger().info(f"replaying {sorted(caps)} from {folder} ({length:.0f} s)")
        return {"caps": caps, "fps": fps, "sent": {cam: 0 for cam in caps}, "length_s": length}

    def _publish_due_frames(self, frames, elapsed):
        """Publish every frame whose time has come; False once all videos have ended."""
        alive = False
        for cam, cap in list(frames["caps"].items()):
            while frames["sent"][cam] < elapsed * frames["fps"][cam]:
                ok, image = cap.read()
                if not ok:
                    cap.release()
                    del frames["caps"][cam]
                    break
                msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"{cam}_camera"
                self.publishers_[cam].publish(msg)
                frames["sent"][cam] += 1
            alive = alive or cam in frames["caps"]
        return alive


def main():
    rclpy.init()
    node = MockHCutServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
