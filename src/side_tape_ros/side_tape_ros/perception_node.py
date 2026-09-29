"""Single-threaded knife-camera tracking node. Never commands the robot."""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np

from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage, Image
from side_tape_interfaces.msg import Observation as ObservationMsg, SupervisorStatus
from side_tape_interfaces.srv import SetSeed
from std_srvs.srv import Trigger

from . import core_path  # noqa: F401  -- core must be on sys.path before imports
from .conversions import observation_to_msg, stamp_to_s
from precision_box_detector.side_tape.backends.classical import ClassicalBackend
from precision_box_detector.side_tape.config import load_config
from precision_box_detector.side_tape.geometry import (derotate_image, line_from_points,
                                                        lookahead_center, roi_around)
from precision_box_detector.side_tape.overlay import draw_overlay
from precision_box_detector.side_tape.replay import load_mask
from precision_box_detector.side_tape.tracker import EdgeTracker


class SideTapePerception(Node):
    def __init__(self):
        super().__init__("side_tape_perception")
        self.declare_parameter("config_path", core_path.DEFAULT_CONFIG)
        config_path = self.get_parameter("config_path").value
        self.cfg = load_config(config_path)
        self.declare_parameter("image_topic", self.cfg.camera.image_topic)
        self.declare_parameter("debug_rate_hz", 5.0)
        # Raw 1280x720 frames over DDS arrive at ~2-8 Hz (measured); the JPEG
        # image_transport topic arrives at ~28 Hz and decodes in ~3 ms.
        self.declare_parameter("use_compressed", True)
        debug_rate = float(self.get_parameter("debug_rate_hz").value)
        if debug_rate < 0:
            raise ValueError("debug_rate_hz must be >= 0")
        self.debug_period = 1.0 / debug_rate if debug_rate else None
        self.bridge = CvBridge()
        self._provisional = self.cfg.mount.tip_uv is None or self.cfg.mount.J is None
        if self._provisional:
            # Preview before calibration: rough tip/J so the debug image works; e_lat is meaningless
            # and control stays blocked (mount.calibrated is false).
            if self.cfg.mount.tip_uv is None:
                self.cfg.mount.tip_uv = [self.cfg.camera.width / 2.0, self.cfg.camera.height / 2.0]
            if self.cfg.mount.J is None:
                self.cfg.mount.J = [[-10.0, 0.0], [0.0, -10.0]]
            self.cfg.mount.calibrated = False
            self.get_logger().warn("UNCALIBRATED PREVIEW: tip_uv/J missing -> using rough values; "
                                   "run calib_tip_mask and calib_jacobian. Control is blocked.")
        backend = ClassicalBackend(self.cfg.detector, load_mask(self.cfg, config_path))
        self.tracker = EdgeTracker(self.cfg, backend)
        self._seq = 0
        self._last_debug_s = float("-inf")
        self._proc_samples = []
        self._mode = "align"
        self._phase = "IDLE"
        self._seed_line = None
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        topic = self.get_parameter("image_topic").value
        if bool(self.get_parameter("use_compressed").value):
            topic = topic.rstrip("/") + "/compressed"
            self.create_subscription(CompressedImage, topic, self._on_compressed, image_qos)
        else:
            self.create_subscription(Image, topic, self._on_image, image_qos)
        self._topic = topic
        self.observation_pub = self.create_publisher(ObservationMsg, "/side_tape/observation", 10)
        self.debug_pub = self.create_publisher(Image, "/side_tape/debug_image", 1)
        self.create_subscription(SupervisorStatus, "/side_tape/status", self._on_status, 10)
        # Latched: the sequencer sets how far the tip is in front of the face (parallax correction).
        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float64, "/side_tape/standoff_mm", self._on_standoff, latched)
        self.create_service(SetSeed, "/side_tape/set_seed", self._on_seed)
        self.create_service(Trigger, "/side_tape/reset_tracker", self._on_reset)
        self.create_timer(5.0, self._log_stats)
        self.get_logger().info(f"tracking {topic} (no robot output)")

    def _on_standoff(self, msg):
        self.tracker.set_standoff_mm(msg.data)
        self.get_logger().info(f"standoff {msg.data:.1f} mm -> effective offset "
                               f"{self.tracker.effective_offset_mm:+.2f} mm")

    def _on_status(self, msg):
        self._phase = msg.phase
        mode = "cut" if msg.phase == "CUT" else "align"
        if mode != self._mode:
            self.tracker.set_mode(mode)
            self._mode = mode

    def _on_seed(self, request, response):
        if not hasattr(self.tracker, "set_seed"):
            response.accepted = False
            response.message = "this tracker does not implement set_seed"
            return response
        try:
            line = line_from_points((request.u0, request.v0), (request.u1, request.v1))
            self.tracker.set_seed(line)
        except ValueError as exc:
            response.accepted = False
            response.message = str(exc)
            return response
        self._seed_line = line
        response.accepted = True
        response.message = "seed accepted"
        return response

    def _on_reset(self, _request, response):
        self.tracker.reset()
        self._seed_line = None
        response.success = True
        response.message = "tracker reset"
        return response

    def _on_compressed(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f"JPEG decode failed ({msg.format})", throttle_duration_sec=5.0)
            return
        self._process(image, msg.header)

    def _on_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image decode failed: {exc}", throttle_duration_sec=5.0)
            return
        self._process(image, msg.header)

    def _process(self, image, header):
        h, w = image.shape[:2]
        if (w, h) != (self.cfg.camera.width, self.cfg.camera.height):
            self.get_logger().error(
                f"image {w}x{h} != configured "
                f"{self.cfg.camera.width}x{self.cfg.camera.height}; frame skipped",
                throttle_duration_sec=5.0)
            return
        image = derotate_image(image, self.cfg.camera.rotate_180)
        obs = self.tracker.step(image, stamp_to_s(header.stamp), self._seq)
        self._seq += 1
        if obs.state.value == "LOST" and self._phase in ("IDLE", "DONE"):
            # LOST is sticky in the tracker; re-search automatically only while the
            # controller is not using observations (never mid-align/cut/hold).
            self.tracker.reset()
            if self._seed_line is not None and hasattr(self.tracker, "set_seed"):
                self.tracker.set_seed(self._seed_line)
        result = observation_to_msg(obs, ObservationMsg, Time, frame_id=header.frame_id)
        result.header.stamp = header.stamp  # preserve the exact captured nanoseconds
        self.observation_pub.publish(result)
        self._proc_samples.append(obs.proc_ms)
        now = time.monotonic()
        if self.debug_period is not None and now - self._last_debug_s >= self.debug_period:
            center = (lookahead_center(self.cfg.tip, self.cfg.J, self.cfg.detector.lookahead_mm)
                      if self._mode == "cut" else self.cfg.tip)
            roi = roi_around(center, self.cfg.detector.roi_half_w,
                             self.cfg.detector.roi_half_h,
                             self.cfg.camera.width, self.cfg.camera.height)
            overlay = draw_overlay(image, obs, self.cfg, roi=roi, seed=self._seed_line,
                                   offset_mm=self.tracker.effective_offset_mm)
            if self._provisional:
                cv2.putText(overlay, "UNCALIBRATED PREVIEW (no J) - e_lat not meaningful", (20, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
            self._last_debug_s = now

    def _log_stats(self):
        if self._proc_samples:
            samples = self._proc_samples
            self._proc_samples = []
            self.get_logger().info(f"tracking: {len(samples)} frames / 5 s, "
                                   f"proc_ms p50={statistics.median(samples):.1f} "
                                   f"p95={sorted(samples)[int(0.95 * (len(samples) - 1))]:.1f}")


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = SideTapePerception()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
