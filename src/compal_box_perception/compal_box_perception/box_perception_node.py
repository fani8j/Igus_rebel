"""Live synchronized RGB-D carton seam perception node."""

from __future__ import annotations

import json
import threading
from pathlib import Path
import math
from typing import Dict, Iterable, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped conversions
from compal_box_msgs.msg import BoxSnapshot
from compal_box_msgs.srv import CaptureBoxSnapshot
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

import detect_tape_seam

from .depth_utils import (
    build_xyzrgb_points,
    decode_compressed_depth,
    depth_scale_for_encoding,
    depth_from_plane_at_pixel,
    fit_top_plane_quad,
    get_median_depth,
    inset_segment,
    normalized_camera_rays,
    pixel_to_camera_xyz,
)
from .stability import (
    TemporalArrayMedian,
    TemporalScalarMedian,
    TemporalUnitVectorMedian,
)


class DetectionRejected(RuntimeError):
    """A frame failed a required perception invariant."""


class BoxPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("box_perception")
        self._declare_parameters()
        self._validate_parameters()

        self._bridge = CvBridge()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._markers_visible = False
        self._last_rgb = None
        self._camera_info = None
        self._pointcloud_geometry_key = None
        self._pointcloud_rays = None
        self._supports_quad_edge_refinement = hasattr(
            detect_tape_seam, "refine_quad_edges_by_gradient"
        )
        self._supports_seam_refinement = hasattr(
            detect_tape_seam, "refine_seam_line_by_gradient"
        )
        if bool(self._parameter("use_rgb_quad_edge_refinement")) and not self._supports_quad_edge_refinement:
            self.get_logger().warning(
                "Copied detector has no refine_quad_edges_by_gradient(); skipping optional RGB quad-edge refinement"
            )
        if (
            bool(self._parameter("use_rgb_front_edge_refinement"))
            or bool(self._parameter("use_rgb_seam_refinement"))
        ) and not self._supports_seam_refinement:
            self.get_logger().warning(
                "Copied detector has no refine_seam_line_by_gradient(); skipping optional RGB line refinement"
            )
        self._last_search_roi = None
        self._seam_ratio_filter = TemporalScalarMedian(
            window_size=int(self._parameter("seam_ratio_filter_window")),
            max_jump=float(self._parameter("seam_ratio_max_jump")),
        )
        self._corner_filter = TemporalArrayMedian(
            window_size=int(self._parameter("corner_filter_window")),
            max_point_jump=float(self._parameter("max_corner_jump_px")),
        )
        self._front_ratio_filter = TemporalScalarMedian(
            window_size=int(self._parameter("front_ratio_filter_window")),
            max_jump=float(self._parameter("front_ratio_max_jump")),
        )
        self._normal_filter = TemporalUnitVectorMedian(
            window_size=int(self._parameter("normal_filter_window")),
            max_angle_jump_deg=float(self._parameter("max_normal_jump_deg")),
        )
        self._last_detection_roi = None
        self._rgbd_lock = threading.Lock()
        self._original_detector_lock = threading.Lock()
        self._latest_rgbd = None
        self._last_raw_rgb = None
        self._start_pub = self.create_publisher(PointStamped, "/box_seam/start", 10)
        self._end_pub = self.create_publisher(PointStamped, "/box_seam/end", 10)
        self._center_pub = self.create_publisher(PointStamped, "/box_seam/center", 10)
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/box_seam/markers", marker_qos
        )
        self._normal_pub = self.create_publisher(
            Vector3Stamped, "/box_seam/surface_normal", marker_qos
        )
        self._debug_pub = self.create_publisher(Image, "/box_detection/debug_image", 10)
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, "/box_detection/diagnostics", 10
        )
        self._snapshot_service = self.create_service(
            CaptureBoxSnapshot, "~/capture_snapshot", self._capture_snapshot
        )
        self._original_detector_service = self.create_service(
            Trigger, "~/run_original_detector", self._run_original_detector
        )
        cloud_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/box_detection/registered_points", cloud_qos
        )

        rgb_topic = self._parameter("rgb_topic")
        depth_topic = self._parameter("depth_topic")
        camera_info_topic = self._parameter("camera_info_topic")
        self._use_compressed_rgb = bool(self._parameter("use_compressed_rgb"))
        self._use_compressed_depth = bool(self._parameter("use_compressed_depth"))
        rgb_type = CompressedImage if self._use_compressed_rgb else Image
        depth_type = CompressedImage if self._use_compressed_depth else Image
        self._rgb_sub = message_filters.Subscriber(
            self, rgb_type, rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = message_filters.Subscriber(
            self, depth_type, depth_topic, qos_profile=qos_profile_sensor_data
        )
        self._info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=int(self._parameter("sync_queue_size")),
            slop=float(self._parameter("sync_slop_s")),
        )
        self._synchronizer.registerCallback(self._on_rgbd)
        self.get_logger().info(
            f"Waiting for synchronized {rgb_topic} and {depth_topic}; "
            f"intrinsics from {camera_info_topic}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "use_compressed_rgb": False,
            "use_compressed_depth": False,
            "robot_base_frame": "base_link",
            "use_auto_roi": True,
            "manual_roi": [590, 90, 860, 300],
            "manual_roi_reference_size": [1280, 720],
            "constrain_auto_roi": True,
            "auto_roi_search_margin_ratio": 0.25,
            "sync_queue_size": 10,
            "use_rgb_seam_refinement": True,
            "seam_ratio_min": 0.20,
            "seam_ratio_max": 0.60,
            "seam_ratio_step": 0.005,
            "seam_min_gradient_score": 20.0,
            "seam_ratio_filter_window": 15,
            "seam_ratio_max_jump": 0.04,
            "sync_slop_s": 0.05,
            "corner_filter_window": 15,
            "max_corner_jump_px": 10.0,
            "use_rgb_quad_edge_refinement": True,
            "quad_edge_search_range_px": 8.0,
            "quad_edge_offset_step_px": 0.5,
            "quad_edge_min_gradient_score": 20.0,
            "use_rgb_front_edge_refinement": False,
            "front_ratio_min": 0.70,
            "front_ratio_max": 1.00,
            "front_ratio_step": 0.005,
            "front_min_gradient_score": 40.0,
            "front_ratio_filter_window": 15,
            "front_ratio_max_jump": 0.05,
            "use_depth_plane_top_face": True,
            "plane_fit_height_ratio": 0.65,
            "plane_max_normal_deviation_deg": 20.0,
            "plane_sample_stride": 2,
            "plane_distance_threshold_m": 0.008,
            "plane_ransac_iterations": 100,
            "plane_min_inliers": 80,
            "plane_min_inlier_ratio": 0.25,
            "seam_inset_ratio": 0.10,
            "depth_patch_radius": 3,
            "min_valid_depth_samples": 5,
            "min_depth_m": 0.15,
            "max_depth_m": 2.0,
            "min_seam_length_m": 0.02,
            "max_seam_length_m": 1.0,
            "max_surface_depth_difference_m": 0.10,
            "min_top_face_area_ratio": 0.005,
            "max_top_face_height_width_ratio": 0.70,
            "max_opposite_edge_angle_difference_deg": 15.0,
            "normal_filter_window": 15,
            "max_normal_jump_deg": 10.0,
            "tf_timeout_s": 0.20,
            "marker_lifetime_s": 30.0,
            "publish_debug_image": True,
            "publish_local_pointcloud": True,
            "pointcloud_stride": 4,
            "snapshot_output_dir": "~/box_snapshots",
            "snapshot_max_receipt_age_s": 1.0,
            "publish_markers": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self) -> None:
        inset = float(self._parameter("seam_inset_ratio"))
        if not 0.0 <= inset < 0.5:
            raise ValueError("seam_inset_ratio must be in [0.0, 0.5)")
        roi = list(self._parameter("manual_roi"))
        if len(roi) != 4:
            raise ValueError("manual_roi must contain x0, y0, x1, y1")
        roi_reference_size = list(self._parameter("manual_roi_reference_size"))
        if len(roi_reference_size) != 2 or any(
            int(value) <= 0 for value in roi_reference_size
        ):
            raise ValueError("manual_roi_reference_size must contain positive width, height")
        seam_ratio_min = float(self._parameter("seam_ratio_min"))
        seam_ratio_max = float(self._parameter("seam_ratio_max"))
        if not 0.0 <= seam_ratio_min < seam_ratio_max <= 1.0:
            raise ValueError("seam ratio bounds must satisfy 0 <= min < max <= 1")
        if int(self._parameter("seam_ratio_filter_window")) < 1:
            raise ValueError("seam_ratio_filter_window must be positive")
        if float(self._parameter("seam_ratio_max_jump")) <= 0.0:
            raise ValueError("seam_ratio_max_jump must be positive")
        if int(self._parameter("corner_filter_window")) < 1:
            raise ValueError("corner_filter_window must be positive")
        if float(self._parameter("max_corner_jump_px")) <= 0.0:
            raise ValueError("max_corner_jump_px must be positive")
        front_ratio_min = float(self._parameter("front_ratio_min"))
        front_ratio_max = float(self._parameter("front_ratio_max"))
        if not 0.0 <= front_ratio_min < front_ratio_max <= 1.0:
            raise ValueError("front ratio bounds must satisfy 0 <= min < max <= 1")
        search_margin = float(self._parameter("auto_roi_search_margin_ratio"))
        if search_margin < 0.0:
            raise ValueError("auto_roi_search_margin_ratio must be non-negative")
        if float(self._parameter("max_top_face_height_width_ratio")) <= 0.0:
            raise ValueError("max_top_face_height_width_ratio must be positive")
        plane_fit_height_ratio = float(self._parameter("plane_fit_height_ratio"))
        if not 0.0 < plane_fit_height_ratio <= 1.0:
            raise ValueError("plane_fit_height_ratio must be in (0.0, 1.0]")
        if int(self._parameter("plane_sample_stride")) < 1:
            raise ValueError("plane_sample_stride must be positive")
        if int(self._parameter("normal_filter_window")) < 1:
            raise ValueError("normal_filter_window must be positive")
        if not 0.0 < float(self._parameter("max_normal_jump_deg")) <= 180.0:
            raise ValueError("max_normal_jump_deg must be in (0, 180]")
        if int(self._parameter("sync_queue_size")) < 1:
            raise ValueError("sync_queue_size must be positive")
        if int(self._parameter("depth_patch_radius")) < 0:
            raise ValueError("depth_patch_radius must be non-negative")
        if int(self._parameter("pointcloud_stride")) < 1:
            raise ValueError("pointcloud_stride must be positive")
        if float(self._parameter("snapshot_max_receipt_age_s")) <= 0.0:
            raise ValueError("snapshot_max_receipt_age_s must be positive")
        if int(self._parameter("min_valid_depth_samples")) < 1:
            raise ValueError("min_valid_depth_samples must be positive")
        if float(self._parameter("min_depth_m")) >= float(self._parameter("max_depth_m")):
            raise ValueError("min_depth_m must be less than max_depth_m")

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _on_camera_info(self, info_msg: CameraInfo) -> None:
        self._camera_info = info_msg

    def _on_rgbd(
        self, rgb_msg: Image | CompressedImage, depth_msg: Image | CompressedImage
    ) -> None:
        if self._camera_info is None:
            self._publish_diagnostic(
                DiagnosticStatus.WARN, "Waiting for CameraInfo", {}
            )
            return
        info_msg = self._camera_info
        with self._rgbd_lock:
            self._latest_rgbd = (
                rgb_msg,
                depth_msg,
                info_msg,
                self.get_clock().now(),
            )
        if (
            not bool(self._parameter("publish_local_pointcloud"))
            or self._cloud_pub.get_subscription_count() == 0
        ):
            return
        try:
            rgb, depth, depth_encoding = self._decode_rgbd(rgb_msg, depth_msg)
            self._validate_rgbd_geometry(rgb, depth, depth_msg, info_msg)
            self._publish_local_pointcloud(
                depth, rgb, depth_encoding, depth_msg.header, info_msg
            )
        except (DetectionRejected, ValueError, CvBridgeError) as exc:
            self.get_logger().warning(
                f"Point-cloud publication failed: {exc}", throttle_duration_sec=2.0
            )

    def _decode_rgbd(
        self, rgb_msg: Image | CompressedImage, depth_msg: Image | CompressedImage
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if self._use_compressed_rgb:
            rgb = self._bridge.compressed_imgmsg_to_cv2(
                rgb_msg, desired_encoding="bgr8"
            )
        else:
            rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        if self._use_compressed_depth:
            depth, depth_encoding = decode_compressed_depth(
                bytes(depth_msg.data), depth_msg.format
            )
        else:
            depth = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            depth_encoding = depth_msg.encoding
        return rgb, depth, depth_encoding

    @staticmethod
    def _validate_rgbd_geometry(
        rgb: np.ndarray, depth: np.ndarray, depth_msg: Image | CompressedImage,
        info_msg: CameraInfo,
    ) -> str:
        if rgb.shape[:2] != depth.shape[:2]:
            raise DetectionRejected(
                f"RGB/depth dimensions differ: {rgb.shape[1]}x{rgb.shape[0]} versus "
                f"{depth.shape[1]}x{depth.shape[0]}"
            )
        if (info_msg.height, info_msg.width) != rgb.shape[:2]:
            raise DetectionRejected("CameraInfo dimensions do not match synchronized images")
        source_frame = depth_msg.header.frame_id
        if not source_frame or info_msg.header.frame_id != source_frame:
            raise DetectionRejected(
                f"Depth/CameraInfo frames differ: {source_frame!r} versus "
                f"{info_msg.header.frame_id!r}"
            )
        return source_frame

    def _process(
        self,
        rgb_msg: Image | CompressedImage,
        depth_msg: Image | CompressedImage,
        info_msg: CameraInfo,
    ) -> Tuple[
        Dict[str, PointStamped],
        np.ndarray,
        Dict[str, PointStamped],
        np.ndarray,
        Dict[str, float],
    ]:
        rgb, depth, depth_encoding = self._decode_rgbd(rgb_msg, depth_msg)
        self._last_rgb = rgb
        self._last_raw_rgb = rgb.copy()
        source_frame = self._validate_rgbd_geometry(
            rgb, depth, depth_msg, info_msg
        )

        roi = self._resolve_roi(rgb)
        # The copied revised detector owns all 2D carton/seam geometry. Depth is
        # used only to reconstruct those RGB pixels in 3D for TF and planning.
        result = detect_tape_seam.detect_seam(rgb, roi)
        plane_metrics = {}
        plane_normal = None
        plane_offset = None
        if bool(self._parameter("use_depth_plane_top_face")):
            expected_normal = self._expected_top_normal_camera(
                source_frame, depth_msg.header.stamp
            )
            x0, y0, x1, y1 = roi
            fit_y1 = y0 + round(
                (y1 - y0) * float(self._parameter("plane_fit_height_ratio"))
            )
            _, plane_normal, plane_offset, inlier_ratio, plane_rms = fit_top_plane_quad(
                depth,
                info_msg.k,
                info_msg.d,
                scale=depth_scale_for_encoding(depth_encoding),
                fit_roi=(x0, y0, x1, fit_y1),
                classify_roi=roi,
                sample_stride=int(self._parameter("plane_sample_stride")),
                distance_threshold_m=float(
                    self._parameter("plane_distance_threshold_m")
                ),
                iterations=int(self._parameter("plane_ransac_iterations")),
                min_inliers=int(self._parameter("plane_min_inliers")),
                min_inlier_ratio=float(self._parameter("plane_min_inlier_ratio")),
                expected_normal=expected_normal,
                max_normal_deviation_deg=float(
                    self._parameter("plane_max_normal_deviation_deg")
                ),
            )
            plane_metrics = {
                "plane_inlier_ratio": inlier_ratio,
                "plane_rms_m": plane_rms,
                "plane_normal_x": float(plane_normal[0]),
                "plane_normal_y": float(plane_normal[1]),
                "plane_normal_z": float(plane_normal[2]),
            }
        detection_debug = detect_tape_seam.draw_visualization(rgb, result)
        self._last_rgb = detection_debug
        self._validate_detection_geometry(result, rgb.shape)

        if plane_normal is not None and plane_offset is not None:
            start_uv = np.asarray(result["seam_line"][0], dtype=np.float64)
            end_uv = np.asarray(result["seam_line"][1], dtype=np.float64)
            depths = [
                depth_from_plane_at_pixel(
                    uv[0],
                    uv[1],
                    info_msg.k,
                    info_msg.d,
                    plane_normal,
                    plane_offset,
                )
                for uv in (start_uv, end_uv)
            ]
            plane_metrics["endpoint_inset_ratio_used"] = 0.0
        else:
            start_uv, end_uv = inset_segment(
                result["seam_line"][0],
                result["seam_line"][1],
                float(self._parameter("seam_inset_ratio")),
            )
            scale = depth_scale_for_encoding(depth_encoding)
            depths = [
                get_median_depth(
                    depth,
                    uv[0],
                    uv[1],
                    radius=int(self._parameter("depth_patch_radius")),
                    scale=scale,
                    min_valid_samples=int(
                        self._parameter("min_valid_depth_samples")
                    ),
                )
                for uv in (start_uv, end_uv)
            ]
            if any(value is None for value in depths):
                raise DetectionRejected(
                    "Insufficient valid depth samples at seam endpoints"
                )
            plane_metrics["endpoint_inset_ratio_used"] = float(
                self._parameter("seam_inset_ratio")
            )
        start_depth, end_depth = float(depths[0]), float(depths[1])
        self._validate_depths(start_depth, end_depth)

        camera_xyz = [
            pixel_to_camera_xyz(uv[0], uv[1], z, info_msg.k, info_msg.d)
            for uv, z in zip((start_uv, end_uv), (start_depth, end_depth))
        ]
        source_points = [
            self._make_point(source_frame, depth_msg.header.stamp, xyz)
            for xyz in camera_xyz
        ]
        target_frame = str(self._parameter("robot_base_frame"))
        timeout = Duration(seconds=float(self._parameter("tf_timeout_s")))
        transformed = [
            self._tf_buffer.transform(point, target_frame, timeout=timeout)
            for point in source_points
        ]
        seam_length = self._distance(transformed[0], transformed[1])
        if not (
            float(self._parameter("min_seam_length_m"))
            <= seam_length
            <= float(self._parameter("max_seam_length_m"))
        ):
            raise DetectionRejected(f"Implausible seam length: {seam_length:.4f} m")

        center = self._midpoint(transformed[0], transformed[1])
        points = {"start": transformed[0], "end": transformed[1], "center": center}
        metrics = {
            "start_depth_m": start_depth,
            "end_depth_m": end_depth,
            "seam_length_m": seam_length,
            "start_u": float(start_uv[0]),
            "start_v": float(start_uv[1]),
            "end_u": float(end_uv[0]),
            "end_v": float(end_uv[1]),
        }
        metrics.update(plane_metrics)
        normal_base = self._surface_normal_to_base(
            plane_normal, source_frame, depth_msg.header.stamp
        )
        normal_base = self._normal_filter.update(normal_base)
        corners_base = self._transform_result_corners(
            result,
            info_msg,
            plane_normal,
            plane_offset,
            source_frame,
            depth_msg.header.stamp,
            target_frame,
            timeout,
        )
        return points, normal_base, corners_base, detection_debug, metrics

    def _transform_result_corners(
        self,
        result,
        info_msg,
        plane_normal,
        plane_offset,
        source_frame,
        stamp,
        target_frame,
        timeout,
    ) -> Dict[str, PointStamped]:
        if plane_normal is None or plane_offset is None:
            raise DetectionRejected("Plane is required for a frozen H-cut snapshot")
        transformed = {}
        for name in ("back_left", "back_right", "front_left", "front_right"):
            u, v = result[name]
            depth = depth_from_plane_at_pixel(
                u,
                v,
                info_msg.k,
                info_msg.d,
                plane_normal,
                plane_offset,
            )
            xyz = pixel_to_camera_xyz(u, v, depth, info_msg.k, info_msg.d)
            camera_point = self._make_point(source_frame, stamp, xyz)
            transformed[name] = self._tf_buffer.transform(
                camera_point, target_frame, timeout=timeout
            )
        return transformed

    def _make_snapshot_candidate(
        self,
        points: Dict[str, PointStamped],
        normal: np.ndarray,
        corners: Dict[str, PointStamped],
        metrics: Dict[str, float],
        debug_image: np.ndarray,
    ) -> dict:
        snapshot = BoxSnapshot()
        snapshot.header = points["center"].header
        snapshot.snapshot_id = (
            f"{snapshot.header.stamp.sec}_{snapshot.header.stamp.nanosec}"
        )
        snapshot.seam_start = points["start"].point
        snapshot.seam_end = points["end"].point
        snapshot.seam_center = points["center"].point
        snapshot.back_left = corners["back_left"].point
        snapshot.back_right = corners["back_right"].point
        snapshot.front_left = corners["front_left"].point
        snapshot.front_right = corners["front_right"].point
        snapshot.surface_normal.x = float(normal[0])
        snapshot.surface_normal.y = float(normal[1])
        snapshot.surface_normal.z = float(normal[2])
        snapshot.plane_rms_m = float(metrics["plane_rms_m"])
        snapshot.confirmation_count = 1
        return {
            "message": snapshot,
            "rgb": self._last_raw_rgb.copy(),
            "debug": debug_image.copy(),
        }

    def _run_original_detector(self, _request, response):
        """Run the unmodified RGB detector on the newest synchronized RGB frame."""
        with self._original_detector_lock:
            with self._rgbd_lock:
                cached_rgbd = self._latest_rgbd
            if cached_rgbd is None:
                response.success = False
                response.message = "No synchronized RGB frame is available"
                return response
            rgb_msg = cached_rgbd[0]
            try:
                if self._use_compressed_rgb:
                    rgb = self._bridge.compressed_imgmsg_to_cv2(
                        rgb_msg, desired_encoding="bgr8"
                    )
                else:
                    rgb = self._bridge.imgmsg_to_cv2(
                        rgb_msg, desired_encoding="bgr8"
                    )
                roi = self._resolve_roi(rgb)
                result = detect_tape_seam.detect_seam(rgb, roi)
                if result is None or "seam_line" not in result:
                    response.success = False
                    response.message = "Original detector found no seam"
                    return response
                seam_start, seam_end = result["seam_line"]
                debug = detect_tape_seam.draw_visualization(rgb, result)
                debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                debug_msg.header = rgb_msg.header
                self._debug_pub.publish(debug_msg)
                response.success = True
                response.message = (
                    "Original detector succeeded; ROI "
                    f"{tuple(int(value) for value in roi)}, seam pixels "
                    f"({int(seam_start[0])}, {int(seam_start[1])}) -> "
                    f"({int(seam_end[0])}, {int(seam_end[1])}); "
                    "debug image published on /box_detection/debug_image"
                )
            except Exception as exc:
                self.get_logger().error(f"Original detector service failed: {exc}")
                response.success = False
                response.message = f"Original detector failed: {exc}"
        return response

    def _capture_snapshot(self, _request, response):
        """Run one RGB-D detection and return its frozen snapshot."""
        with self._rgbd_lock:
            cached_rgbd = self._latest_rgbd
        if cached_rgbd is None:
            response.success = False
            response.message = "No synchronized RGB-D frame is available"
            return response
        rgb_msg, depth_msg, info_msg, received = cached_rgbd
        age = (self.get_clock().now() - received).nanoseconds / 1e9
        if age > float(self._parameter("snapshot_max_receipt_age_s")):
            response.success = False
            response.message = f"Latest synchronized RGB-D frame is {age:.2f}s old"
            return response
        try:
            points, normal, corners, debug_image, metrics = self._process(
                rgb_msg, depth_msg, info_msg
            )
            if bool(self._parameter("publish_debug_image")):
                debug_msg = self._bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                debug_msg.header = rgb_msg.header
                self._debug_pub.publish(debug_msg)
            self._publish_points(points)
            self._publish_surface_normal(normal, points["start"].header)
            candidate = self._make_snapshot_candidate(
                points, normal, corners, metrics, debug_image
            )
            if bool(self._parameter("publish_markers")):
                self._publish_markers(points)
            self._publish_diagnostic(DiagnosticStatus.OK, "captured seam", metrics)
        except (DetectionRejected, ValueError, CvBridgeError, TransformException) as exc:
            self.get_logger().warning(str(exc))
            self._publish_failure_debug(rgb_msg.header)
            self._reset_geometry_filters()
            self._clear_markers()
            self._publish_diagnostic(DiagnosticStatus.WARN, str(exc), {})
            response.success = False
            response.message = str(exc)
            return response
        except Exception as exc:
            self.get_logger().error(f"Detection failed: {exc}")
            self._publish_failure_debug(rgb_msg.header)
            self._reset_geometry_filters()
            self._clear_markers()
            self._publish_diagnostic(DiagnosticStatus.ERROR, str(exc), {})
            response.success = False
            response.message = f"Detection failed: {exc}"
            return response

        snapshot = candidate["message"]
        output_dir = Path(
            str(self._parameter("snapshot_output_dir"))
        ).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / snapshot.snapshot_id
        rgb_path = prefix.with_name(f"{prefix.name}_rgb.png")
        debug_path = prefix.with_name(f"{prefix.name}_debug.png")
        json_path = prefix.with_suffix(".json")
        if not cv2.imwrite(str(rgb_path), candidate["rgb"]):
            raise RuntimeError(f"Failed to save {rgb_path}")
        if not cv2.imwrite(str(debug_path), candidate["debug"]):
            raise RuntimeError(f"Failed to save {debug_path}")

        def coordinates(point):
            return {"x": point.x, "y": point.y, "z": point.z}

        report = {
            "snapshot_id": snapshot.snapshot_id,
            "frame_id": snapshot.header.frame_id,
            "stamp": {
                "sec": snapshot.header.stamp.sec,
                "nanosec": snapshot.header.stamp.nanosec,
            },
            "seam_start": coordinates(snapshot.seam_start),
            "seam_end": coordinates(snapshot.seam_end),
            "seam_center": coordinates(snapshot.seam_center),
            "back_left": coordinates(snapshot.back_left),
            "back_right": coordinates(snapshot.back_right),
            "front_left": coordinates(snapshot.front_left),
            "front_right": coordinates(snapshot.front_right),
            "surface_normal": coordinates(snapshot.surface_normal),
            "plane_rms_m": snapshot.plane_rms_m,
            "confirmation_count": snapshot.confirmation_count,
            "rgb_path": str(rgb_path),
            "debug_path": str(debug_path),
        }
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        response.success = True
        response.message = f"Captured frozen snapshot {snapshot.snapshot_id}"
        response.snapshot = snapshot
        return response

    @staticmethod
    def _seam_line_at_ratio(result: dict, ratio: float) -> list:
        back_left = np.asarray(result["back_left"], dtype=np.float64)
        back_right = np.asarray(result["back_right"], dtype=np.float64)
        front_left = np.asarray(result["front_left"], dtype=np.float64)
        front_right = np.asarray(result["front_right"], dtype=np.float64)
        left = back_left + ratio * (front_left - back_left)
        right = back_right + ratio * (front_right - back_right)
        return np.stack((left, right)).tolist()

    def _expected_top_normal_camera(self, camera_frame: str, stamp) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            camera_frame,
            str(self._parameter("robot_base_frame")),
            Time.from_msg(stamp),
            timeout=Duration(seconds=float(self._parameter("tf_timeout_s"))),
        )
        rotation = transform.transform.rotation
        x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
        return np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )

    def _surface_normal_to_base(
        self, normal_camera: np.ndarray, camera_frame: str, stamp
    ) -> np.ndarray:
        if normal_camera is None:
            raise DetectionRejected("Surface normal is unavailable")
        transform = self._tf_buffer.lookup_transform(
            str(self._parameter("robot_base_frame")),
            camera_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=float(self._parameter("tf_timeout_s"))),
        )
        return self._rotate_vector(normal_camera, transform.transform.rotation)

    @staticmethod
    def _rotate_vector(vector: np.ndarray, quaternion) -> np.ndarray:
        q = np.array(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype=np.float64,
        )
        q_xyz = q[:3]
        return (
            vector
            + 2.0 * np.cross(q_xyz, np.cross(q_xyz, vector) + q[3] * vector)
        )

    def _publish_surface_normal(self, normal: np.ndarray, header) -> None:
        message = Vector3Stamped()
        message.header = header
        message.vector.x = float(normal[0])
        message.vector.y = float(normal[1])
        message.vector.z = float(normal[2])
        self._normal_pub.publish(message)

    @staticmethod
    def _apply_front_ratio(result: dict, ratio: float) -> dict:
        back_left = np.asarray(result["back_left"], dtype=np.float64)
        back_right = np.asarray(result["back_right"], dtype=np.float64)
        front_left = np.asarray(result["front_left"], dtype=np.float64)
        front_right = np.asarray(result["front_right"], dtype=np.float64)
        refined_front_left = back_left + ratio * (front_left - back_left)
        refined_front_right = back_right + ratio * (front_right - back_right)
        corners = np.stack(
            (back_left, back_right, refined_front_right, refined_front_left)
        )
        return BoxPerceptionNode._result_from_corners(result["roi"], corners)

    def _reset_geometry_filters(self) -> None:
        self._seam_ratio_filter.reset()
        self._front_ratio_filter.reset()
        self._corner_filter.reset()
        self._normal_filter.reset()

    @staticmethod
    def _result_from_corners(roi, corners: np.ndarray) -> dict:
        geometry = detect_tape_seam.order_quad_and_find_seam(corners)

        def as_list(value):
            return np.asarray(value, dtype=float).tolist()

        return {
            "roi": tuple(int(value) for value in roi),
            "top_face_corners": as_list(geometry["corners_ordered"]),
            "left_edge": as_list(geometry["left_edge"]),
            "right_edge": as_list(geometry["right_edge"]),
            "back_edge": as_list(geometry["back_edge"]),
            "front_edge": as_list(geometry["front_edge"]),
            "seam_line": as_list(geometry["seam_line"]),
            "back_left": as_list(geometry["back_left"]),
            "back_right": as_list(geometry["back_right"]),
            "front_left": as_list(geometry["front_left"]),
            "front_right": as_list(geometry["front_right"]),
        }

    def _scaled_manual_roi(self, rgb: np.ndarray) -> Tuple[int, int, int, int]:
        reference_width, reference_height = (
            int(value) for value in self._parameter("manual_roi_reference_size")
        )
        scale_x = rgb.shape[1] / reference_width
        scale_y = rgb.shape[0] / reference_height
        x0, y0, x1, y1 = (int(value) for value in self._parameter("manual_roi"))
        return (
            round(x0 * scale_x),
            round(y0 * scale_y),
            round(x1 * scale_x),
            round(y1 * scale_y),
        )

    @staticmethod
    def _expand_roi(
        roi: Tuple[int, int, int, int], image_shape: tuple, margin_ratio: float
    ) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = roi
        height, width = image_shape[:2]
        margin_x = round((x1 - x0) * margin_ratio)
        margin_y = round((y1 - y0) * margin_ratio)
        return (
            max(0, x0 - margin_x),
            max(0, y0 - margin_y),
            min(width, x1 + margin_x),
            min(height, y1 + margin_y),
        )

    def _resolve_roi(self, rgb: np.ndarray) -> Tuple[int, int, int, int]:
        manual_roi = self._scaled_manual_roi(rgb)
        if bool(self._parameter("use_auto_roi")):
            search_roi = None
            roi_kwargs = {}
            if bool(self._parameter("constrain_auto_roi")):
                search_roi = self._expand_roi(
                    manual_roi,
                    rgb.shape,
                    float(self._parameter("auto_roi_search_margin_ratio")),
                )
                width, height = rgb.shape[1], rgb.shape[0]
                center_x = (manual_roi[0] + manual_roi[2]) / 2.0
                center_y = (manual_roi[1] + manual_roi[3]) / 2.0
                search_half_diagonal = math.hypot(
                    (search_roi[2] - search_roi[0]) / 2.0,
                    (search_roi[3] - search_roi[1]) / 2.0,
                )
                roi_kwargs = {
                    "expected_center_ratio": (center_x / width, center_y / height),
                    "max_center_dist_ratio": search_half_diagonal / math.hypot(width, height),
                }
            roi, _ = detect_tape_seam.estimate_auto_roi(rgb, **roi_kwargs)
            self._last_search_roi = search_roi
            self._last_detection_roi = tuple(int(value) for value in roi)
            return self._last_detection_roi
        self._last_search_roi = manual_roi
        self._last_detection_roi = manual_roi
        return manual_roi

    def _publish_local_pointcloud(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        depth_encoding: str,
        header,
        info_msg: CameraInfo,
    ) -> None:
        stride = int(self._parameter("pointcloud_stride"))
        geometry_key = (
            depth.shape[1],
            depth.shape[0],
            tuple(info_msg.k),
            tuple(info_msg.d),
            stride,
        )
        if geometry_key != self._pointcloud_geometry_key:
            self._pointcloud_rays = normalized_camera_rays(
                depth.shape[1],
                depth.shape[0],
                info_msg.k,
                info_msg.d,
                stride,
            )
            self._pointcloud_geometry_key = geometry_key
        points = build_xyzrgb_points(
            depth,
            rgb,
            self._pointcloud_rays,
            scale=depth_scale_for_encoding(depth_encoding),
            stride=stride,
        )
        cloud = PointCloud2()
        cloud.header = header
        cloud.height, cloud.width = points.shape
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = points.dtype.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = points.tobytes(order="C")
        cloud.is_dense = False
        self._cloud_pub.publish(cloud)

    def _validate_detection_geometry(self, result: dict, image_shape: tuple) -> None:
        height, width = image_shape[:2]
        corners = np.asarray(result["top_face_corners"], dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise DetectionRejected("Detector returned invalid top-face corners")
        area_ratio = abs(cv2.contourArea(corners)) / float(width * height)
        if area_ratio < float(self._parameter("min_top_face_area_ratio")):
            raise DetectionRejected(f"Detected top face is too small: ratio={area_ratio:.5f}")
        x_span = float(np.ptp(corners[:, 0]))
        y_span = float(np.ptp(corners[:, 1]))
        if x_span <= 0.0:
            raise DetectionRejected("Detected top face has zero image width")
        height_width_ratio = y_span / x_span
        max_ratio = float(self._parameter("max_top_face_height_width_ratio"))
        if height_width_ratio > max_ratio:
            raise DetectionRejected(
                f"Top-face height/width ratio {height_width_ratio:.3f} exceeds "
                f"{max_ratio:.3f}; likely includes a vertical carton face"
            )
        if not cv2.isContourConvex(corners.astype(np.int32)):
            raise DetectionRejected("Detected top-face quadrilateral is not convex")
        back_edge = np.asarray(result["back_edge"], dtype=np.float64)
        front_edge = np.asarray(result["front_edge"], dtype=np.float64)
        back_angle = math.degrees(
            math.atan2(*(back_edge[1] - back_edge[0])[::-1])
        )
        front_angle = math.degrees(
            math.atan2(*(front_edge[1] - front_edge[0])[::-1])
        )
        angle_difference = abs((back_angle - front_angle + 90.0) % 180.0 - 90.0)
        max_angle_difference = float(
            self._parameter("max_opposite_edge_angle_difference_deg")
        )
        if angle_difference > max_angle_difference:
            raise DetectionRejected(
                f"Opposite top-face edges differ by {angle_difference:.1f} degrees; "
                f"limit is {max_angle_difference:.1f}"
            )
        seam = np.asarray(result["seam_line"], dtype=np.float64)
        if seam.shape != (2, 2) or not np.all(np.isfinite(seam)):
            raise DetectionRejected("Detector returned invalid seam endpoints")
        if np.any(seam[:, 0] < 0) or np.any(seam[:, 0] >= width):
            raise DetectionRejected("Detected seam lies outside image width")
        if np.any(seam[:, 1] < 0) or np.any(seam[:, 1] >= height):
            raise DetectionRejected("Detected seam lies outside image height")

    def _publish_failure_debug(self, header) -> None:
        if not bool(self._parameter("publish_debug_image")) or self._last_rgb is None:
            return
        preview = self._last_rgb.copy()
        if self._last_search_roi is not None:
            x0, y0, x1, y1 = self._last_search_roi
            cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 255, 0), 2)
        if self._last_detection_roi is not None:
            x0, y0, x1, y1 = self._last_detection_roi
            cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(
            preview,
            "DETECTION REJECTED",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        debug_msg = self._bridge.cv2_to_imgmsg(preview, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    def _validate_depths(self, start_depth: float, end_depth: float) -> None:
        minimum = float(self._parameter("min_depth_m"))
        maximum = float(self._parameter("max_depth_m"))
        if not (minimum <= start_depth <= maximum and minimum <= end_depth <= maximum):
            raise DetectionRejected(
                f"Depth outside [{minimum:.3f}, {maximum:.3f}] m: "
                f"{start_depth:.3f}, {end_depth:.3f}"
            )
        difference = abs(start_depth - end_depth)
        limit = float(self._parameter("max_surface_depth_difference_m"))
        if difference > limit:
            raise DetectionRejected(
                f"Endpoint depth difference {difference:.4f} m exceeds {limit:.4f} m"
            )

    @staticmethod
    def _make_point(frame_id: str, stamp, xyz: Iterable[float]) -> PointStamped:
        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = stamp
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        return point

    @staticmethod
    def _distance(first: PointStamped, second: PointStamped) -> float:
        return math.sqrt(
            (first.point.x - second.point.x) ** 2
            + (first.point.y - second.point.y) ** 2
            + (first.point.z - second.point.z) ** 2
        )

    @staticmethod
    def _midpoint(first: PointStamped, second: PointStamped) -> PointStamped:
        center = PointStamped()
        center.header = first.header
        center.point.x = (first.point.x + second.point.x) / 2.0
        center.point.y = (first.point.y + second.point.y) / 2.0
        center.point.z = (first.point.z + second.point.z) / 2.0
        return center

    def _publish_points(self, points: Dict[str, PointStamped]) -> None:
        self._start_pub.publish(points["start"])
        self._end_pub.publish(points["end"])
        self._center_pub.publish(points["center"])

    def _publish_markers(self, points: Dict[str, PointStamped]) -> None:
        lifetime = Duration(seconds=float(self._parameter("marker_lifetime_s"))).to_msg()
        markers = MarkerArray()
        for marker_id, (name, color) in enumerate(
            (("start", (0.0, 1.0, 0.0)), ("end", (1.0, 0.0, 0.0)))
        ):
            marker = Marker()
            marker.header = points[name].header
            marker.ns = "box_seam"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = points[name].point
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.025
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.lifetime = lifetime
            markers.markers.append(marker)

        line = Marker()
        line.header = points["start"].header
        line.ns = "box_seam"
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.points = [points["start"].point, points["end"].point]
        line.scale.x = 0.008
        line.color.r = 0.15
        line.color.g = 0.65
        line.color.b = 1.0
        line.color.a = 1.0
        line.lifetime = lifetime
        markers.markers.append(line)
        self._marker_pub.publish(markers)
        self._markers_visible = True

    def _clear_markers(self) -> None:
        if not self._markers_visible:
            return
        marker = Marker()
        marker.action = Marker.DELETEALL
        self._marker_pub.publish(MarkerArray(markers=[marker]))
        self._markers_visible = False

    def _publish_diagnostic(
        self, level: int, message: str, metrics: Dict[str, float]
    ) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.level = level
        status.name = "compal_box_perception/seam_detection"
        status.hardware_id = "gemini_rgbd"
        status.message = message
        status.values = [
            KeyValue(key=key, value=f"{value:.6f}") for key, value in metrics.items()
        ]
        array.status = [status]
        self._diagnostic_pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoxPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
