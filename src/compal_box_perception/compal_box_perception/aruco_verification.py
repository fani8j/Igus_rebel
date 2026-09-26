"""ArUco marker pose verification for camera-to-base calibration checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import (
    normalize_quaternion,
    quaternion_angle_deg,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_to_matrix,
)


class DetectionRejected(RuntimeError):
    """A frame failed a required calibration-verification invariant."""


@dataclass(frozen=True)
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    reprojection_error_px: float
    alternate_reprojection_error_px: Optional[float]


class StablePoseWindow:
    """Require consecutive poses to agree in translation and SO3 orientation."""

    def __init__(
        self,
        required_count: int,
        max_position_delta_m: float,
        max_rotation_delta_deg: float,
    ) -> None:
        if required_count < 1:
            raise ValueError("stable_count must be positive")
        if max_position_delta_m <= 0.0:
            raise ValueError("max_stable_position_delta_m must be positive")
        if not 0.0 < max_rotation_delta_deg <= 180.0:
            raise ValueError("max_stable_rotation_delta_deg must be in (0, 180]")
        self.required_count = required_count
        self.max_position_delta_m = max_position_delta_m
        self.max_rotation_delta_deg = max_rotation_delta_deg
        self._poses: deque[PoseStamped] = deque(maxlen=required_count)

    @property
    def confirmation_count(self) -> int:
        return len(self._poses)

    def reset(self) -> None:
        self._poses.clear()

    def update(self, pose: PoseStamped) -> int:
        if not self._poses:
            self._poses.append(pose)
            return len(self._poses)
        if not all(self._within_limits(pose, previous) for previous in self._poses):
            self._poses.clear()
        self._poses.append(pose)
        return len(self._poses)

    def stable(self) -> bool:
        return len(self._poses) >= self.required_count

    def _within_limits(self, first: PoseStamped, second: PoseStamped) -> bool:
        position_delta = float(
            np.linalg.norm(_pose_position(first) - _pose_position(second))
        )
        if position_delta > self.max_position_delta_m:
            return False
        angle_delta = quaternion_angle_deg(
            _pose_quaternion(first), _pose_quaternion(second)
        )
        return angle_delta <= self.max_rotation_delta_deg


def _aruco_dictionary_constants() -> dict[str, int]:
    if not hasattr(cv2, "aruco"):
        return {}
    constants = {}
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(cv2.aruco, name)
        if isinstance(value, int):
            constants[name] = value
    return constants


def _resolve_aruco_dictionary(name: str):
    constants = _aruco_dictionary_constants()
    if name not in constants:
        available = ", ".join(sorted(constants)) or "none"
        raise ValueError(
            f"Unsupported marker_dictionary {name!r}; use an explicit OpenCV "
            f"cv2.aruco DICT_* name. Available: {available}"
        )
    dictionary_id = constants[name]
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def _create_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 30
        parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


# Hover-test chart (aruco_tcp_hover_test_A4.pdf): marker centre = origin,
# marker-local X = right, Y = up (same convention as _marker_object_points).
# (name, x_mm, y_mm) measured from the chart; all within 100 mm of centre.
HOVER_TEST_POINTS_MM = (
    ("CENTER", 0, 0),
    ("TOP", 0, 60),
    ("BOTTOM", 0, -60),
    ("LEFT", -60, 0),
    ("RIGHT", 60, 0),
    ("P1", 45, 60),
    ("P2", -70, 35),
    ("P3", 20, -85),
)


def hover_test_points(
    origin: np.ndarray, rotation: np.ndarray, hover_height_m: float
) -> list:
    """Return ``[(name, xyz)]`` hover targets above the chart in the pose frame.

    Each target is the chart offset in marker-local X/Y raised by
    ``hover_height_m`` along marker-local +Z (out of the paper), expressed in
    the same frame as ``origin``/``rotation`` (the stable board pose frame).
    """
    points = []
    for name, x_mm, y_mm in HOVER_TEST_POINTS_MM:
        local = np.array([x_mm / 1000.0, y_mm / 1000.0, hover_height_m], dtype=np.float64)
        points.append((name, origin + rotation @ local))
    return points


def _solve_marker_pose_ippe(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_size_m: float,
    max_reprojection_error_px: float,
    min_ippe_error_ratio: float,
) -> PoseEstimate:
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if image_points.shape != (4, 2) or not np.all(np.isfinite(image_points)):
        raise DetectionRejected("Target marker corners are invalid")
    object_points = _marker_object_points(marker_size_m)
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error as exc:
        raise DetectionRejected(f"IPPE_SQUARE solvePnP failed: {exc}") from exc

    ok, rvecs, tvecs = result[:3]
    if not ok or len(rvecs) == 0:
        raise DetectionRejected("IPPE_SQUARE produced no pose")

    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        error = _reprojection_error_px(
            object_points, image_points, camera_matrix, distortion, rvec, tvec
        )
        candidates.append(
            PoseEstimate(
                rvec=rvec,
                tvec=tvec,
                rotation_matrix=rotation_matrix,
                reprojection_error_px=error,
                alternate_reprojection_error_px=None,
            )
        )
    candidates.sort(key=lambda candidate: candidate.reprojection_error_px)
    best = candidates[0]

    if float(best.tvec[2, 0]) <= 0.0:
        raise DetectionRejected(
            f"IPPE best pose has non-positive depth: {float(best.tvec[2, 0]):.4f} m"
        )
    if best.reprojection_error_px > max_reprojection_error_px:
        raise DetectionRejected(
            f"Marker reprojection error {best.reprojection_error_px:.3f} px exceeds "
            f"{max_reprojection_error_px:.3f} px"
        )

    marker_normal_camera = best.rotation_matrix[:, 2]
    camera_to_marker = best.tvec.reshape(3)
    if float(np.dot(marker_normal_camera, camera_to_marker)) >= -1e-9:
        raise DetectionRejected("Marker +Z normal is not outward camera-facing")

    alternate_error = None
    if len(candidates) > 1:
        alternate_error = candidates[1].reprojection_error_px
        ratio = alternate_error / max(best.reprojection_error_px, 1e-9)
        if ratio < min_ippe_error_ratio:
            raise DetectionRejected(
                f"Ambiguous IPPE pose: alternate/best reprojection ratio "
                f"{ratio:.2f} is below {min_ippe_error_ratio:.2f}"
            )

    return PoseEstimate(
        rvec=best.rvec,
        tvec=best.tvec,
        rotation_matrix=best.rotation_matrix,
        reprojection_error_px=best.reprojection_error_px,
        alternate_reprojection_error_px=alternate_error,
    )


def _reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residuals = projected.reshape(4, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _pose_from_estimate(
    estimate: PoseEstimate, frame_id: str, stamp
) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(estimate.tvec[0, 0])
    pose.pose.position.y = float(estimate.tvec[1, 0])
    pose.pose.position.z = float(estimate.tvec[2, 0])
    qx, qy, qz, qw = quaternion_from_matrix(estimate.rotation_matrix)
    pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def _pose_position(pose: PoseStamped) -> np.ndarray:
    position = pose.pose.position
    return np.array([position.x, position.y, position.z], dtype=np.float64)


def _pose_quaternion(pose: PoseStamped) -> np.ndarray:
    orientation = pose.pose.orientation
    return normalize_quaternion(
        (orientation.x, orientation.y, orientation.z, orientation.w)
    )


def _stamp_is_zero(stamp) -> bool:
    return int(stamp.sec) == 0 and int(stamp.nanosec) == 0


def _seconds_between(first, second) -> float:
    return abs((Time.from_msg(first) - Time.from_msg(second)).nanoseconds) / 1e9


class ArucoVerificationNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("aruco_verification", **kwargs)
        self._declare_parameters()
        self._validate_parameters()

        self._bridge = CvBridge()
        self._camera_info: Optional[CameraInfo] = None
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._markers_visible = False
        self._last_published_pose: Optional[PoseStamped] = None
        self._last_processed_stamp_ns: Optional[int] = None
        self._read_only_parameters = {
            "rgb_topic",
            "camera_info_topic",
            "use_compressed_rgb",
            "robot_base_frame",
            "marker_dictionary",
            "marker_id",
            "marker_size_m",
            "hover_height_m",
            "publish_hover_targets",
            "stable_count",
            "max_stable_position_delta_m",
            "max_stable_rotation_delta_deg",
            "max_reprojection_error_px",
            "min_ippe_error_ratio",
            "min_marker_side_px",
            "max_image_age_s",
            "max_camera_info_delta_s",
            "tf_timeout_s",
            "max_tf_age_s",
            "marker_lifetime_s",
            "publish_debug_image",
            "publish_markers",
        }
        self.add_on_set_parameters_callback(self._reject_runtime_parameter_updates)

        self._dictionary_name = str(self._parameter("marker_dictionary")).strip()
        self._target_marker_id = int(self._parameter("marker_id"))
        self._configured = bool(self._dictionary_name) and self._target_marker_id >= 0
        self._dictionary = None
        self._detector_parameters = None
        self._detector = None
        if self._configured:
            self._dictionary = _resolve_aruco_dictionary(self._dictionary_name)
            marker_count = getattr(self._dictionary, "bytesList", np.empty((0,))).shape[0]
            if marker_count and self._target_marker_id >= marker_count:
                raise ValueError(
                    f"marker_id {self._target_marker_id} is outside {self._dictionary_name} "
                    f"range [0, {marker_count - 1}]"
                )
            self._detector_parameters = _create_detector_parameters()
            if hasattr(cv2.aruco, "ArucoDetector"):
                self._detector = cv2.aruco.ArucoDetector(
                    self._dictionary, self._detector_parameters
                )
        else:
            self.get_logger().warning(
                "ArUco verification disabled until marker_dictionary and marker_id "
                "are explicitly configured"
            )

        self._stable_window = StablePoseWindow(
            required_count=int(self._parameter("stable_count")),
            max_position_delta_m=float(
                self._parameter("max_stable_position_delta_m")
            ),
            max_rotation_delta_deg=float(
                self._parameter("max_stable_rotation_delta_deg")
            ),
        )

        self._pose_pub = self.create_publisher(
            PoseStamped, "/calibration_verification/board_pose", 10
        )
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray,
            "/calibration_verification/detection_markers",
            marker_qos,
        )
        self._hover_pub = self.create_publisher(
            MarkerArray,
            "/calibration_verification/hover_targets",
            marker_qos,
        )
        self._debug_pub = self.create_publisher(Image, "~/debug_image", 10)

        self._use_compressed_rgb = bool(self._parameter("use_compressed_rgb"))
        image_type = CompressedImage if self._use_compressed_rgb else Image
        self._image_sub = self.create_subscription(
            image_type,
            str(self._parameter("rgb_topic")),
            self._on_image,
            qos_profile_sensor_data,
        )
        self._info_sub = self.create_subscription(
            CameraInfo,
            str(self._parameter("camera_info_topic")),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Waiting for {self._parameter('rgb_topic')} and "
            f"{self._parameter('camera_info_topic')}; publishing stable poses in base_link"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "rgb_topic": "/camera/color/image_raw/compressed",
            "camera_info_topic": "/camera/color/camera_info",
            "use_compressed_rgb": True,
            "robot_base_frame": "base_link",
            "marker_dictionary": "",
            "marker_id": -1,
            "marker_size_m": 0.060,
            "hover_height_m": 0.020,
            "publish_hover_targets": True,
            "stable_count": 5,
            "max_stable_position_delta_m": 0.010,
            "max_stable_rotation_delta_deg": 3.0,
            "max_reprojection_error_px": 1.0,
            "min_ippe_error_ratio": 1.5,
            "min_marker_side_px": 30.0,
            "max_image_age_s": 0.5,
            "max_camera_info_delta_s": 0.5,
            "tf_timeout_s": 0.20,
            "max_tf_age_s": 0.5,
            "marker_lifetime_s": 1.0,
            "publish_debug_image": True,
            "publish_markers": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self) -> None:
        if str(self._parameter("robot_base_frame")) != "base_link":
            raise ValueError("robot_base_frame must remain base_link for this verifier")
        dictionary_name = str(self._parameter("marker_dictionary")).strip()
        marker_id = int(self._parameter("marker_id"))
        if bool(dictionary_name) != (marker_id >= 0):
            raise ValueError(
                "marker_dictionary and marker_id must be explicitly configured together"
            )
        if float(self._parameter("marker_size_m")) <= 0.0:
            raise ValueError("marker_size_m must be positive")
        if not 0.0 <= float(self._parameter("hover_height_m")) <= 0.5:
            raise ValueError("hover_height_m must be in [0, 0.5]")
        if int(self._parameter("stable_count")) < 1:
            raise ValueError("stable_count must be positive")
        if float(self._parameter("max_stable_position_delta_m")) <= 0.0:
            raise ValueError("max_stable_position_delta_m must be positive")
        max_rotation = float(self._parameter("max_stable_rotation_delta_deg"))
        if not 0.0 < max_rotation <= 180.0:
            raise ValueError("max_stable_rotation_delta_deg must be in (0, 180]")
        if float(self._parameter("max_reprojection_error_px")) <= 0.0:
            raise ValueError("max_reprojection_error_px must be positive")
        if float(self._parameter("min_ippe_error_ratio")) <= 1.0:
            raise ValueError("min_ippe_error_ratio must be greater than 1.0")
        if float(self._parameter("min_marker_side_px")) <= 0.0:
            raise ValueError("min_marker_side_px must be positive")
        for name in (
            "max_image_age_s",
            "max_camera_info_delta_s",
            "tf_timeout_s",
            "max_tf_age_s",
            "marker_lifetime_s",
        ):
            if float(self._parameter(name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _reject_runtime_parameter_updates(self, parameters) -> SetParametersResult:
        changed = [
            parameter.name
            for parameter in parameters
            if parameter.name in self._read_only_parameters
        ]
        if changed:
            return SetParametersResult(
                successful=False,
                reason=(
                    "ArUco verifier parameters are read-only; restart the node to "
                    f"change: {', '.join(changed)}"
                ),
            )
        return SetParametersResult(successful=True)


    def _on_camera_info(self, info_msg: CameraInfo) -> None:
        self._camera_info = info_msg

    def _on_image(self, image_msg: Image | CompressedImage) -> None:
        # Decode once; the failure path reuses the decoded frame for its debug image.
        rgb = None
        try:
            rgb = self._decode_rgb(image_msg)
            self._process_image(image_msg, rgb)
        except (DetectionRejected, TransformException, ValueError, CvBridgeError) as exc:
            self._stable_window.reset()
            self._last_published_pose = None
            self._clear_markers()
            self.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            self._publish_failure_debug(image_msg, rgb, str(exc))
        except Exception as exc:
            self._stable_window.reset()
            self._last_published_pose = None
            self._clear_markers()
            self.get_logger().error(f"ArUco verification failed: {exc}")
            self._publish_failure_debug(
                image_msg, rgb, f"ArUco verification failed: {exc}"
            )

    def _process_image(
        self, image_msg: Image | CompressedImage, rgb: np.ndarray
    ) -> Optional[PoseStamped]:
        if not self._configured:
            self._stable_window.reset()
            self._last_published_pose = None
            self._clear_markers()
            self._publish_debug_image(
                rgb,
                image_msg.header,
                "disabled: set marker_dictionary and marker_id",
                published=False,
            )
            return None
        if self._camera_info is None:
            raise DetectionRejected("Waiting for CameraInfo")

        self._validate_image_freshness(image_msg.header.stamp)
        self._validate_monotonic_image_stamp(image_msg.header.stamp)
        camera_matrix, distortion = self._validated_camera_model(
            self._camera_info, rgb.shape, image_msg.header
        )
        corners, ids = self._detect_markers(rgb)
        target_corners = self._target_marker_corners(corners, ids)
        estimate = _solve_marker_pose_ippe(
            target_corners,
            camera_matrix,
            distortion,
            float(self._parameter("marker_size_m")),
            float(self._parameter("max_reprojection_error_px")),
            float(self._parameter("min_ippe_error_ratio")),
        )
        camera_pose = _pose_from_estimate(
            estimate,
            image_msg.header.frame_id,
            image_msg.header.stamp,
        )
        base_pose = self._transform_pose_to_base(camera_pose)
        self._validate_image_freshness(image_msg.header.stamp)
        confirmation_count = self._stable_window.update(base_pose)
        is_stable = self._stable_window.stable()
        if is_stable:
            self._pose_pub.publish(base_pose)
            self._last_published_pose = base_pose
            if bool(self._parameter("publish_markers")):
                self._publish_markers(base_pose)
            if bool(self._parameter("publish_hover_targets")):
                self._publish_hover_targets(base_pose)
            status = (
                f"published stable {confirmation_count}/{self._stable_window.required_count}; "
                f"reprojection {estimate.reprojection_error_px:.2f} px"
            )
        else:
            self._last_published_pose = None
            self._clear_markers()
            status = (
                f"tracking {confirmation_count}/{self._stable_window.required_count}; "
                f"reprojection {estimate.reprojection_error_px:.2f} px"
            )
        self._publish_debug_image(
            rgb,
            image_msg.header,
            status,
            corners=target_corners,
            camera_matrix=camera_matrix,
            distortion=distortion,
            estimate=estimate,
            published=is_stable,
        )
        return base_pose if is_stable else None

    def _decode_rgb(self, image_msg: Image | CompressedImage) -> np.ndarray:
        if self._use_compressed_rgb:
            return self._bridge.compressed_imgmsg_to_cv2(
                image_msg, desired_encoding="bgr8"
            )
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    def _validate_image_freshness(self, stamp) -> None:
        if _stamp_is_zero(stamp):
            raise DetectionRejected("Image stamp is zero; refusing freshness-unknown pose")
        age_s = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
        if age_s > float(self._parameter("max_image_age_s")):
            raise DetectionRejected(
                f"Image is stale: age {age_s:.3f}s exceeds "
                f"{float(self._parameter('max_image_age_s')):.3f}s"
            )

    def _validate_monotonic_image_stamp(self, stamp) -> None:
        stamp_ns = Time.from_msg(stamp).nanoseconds
        if (
            self._last_processed_stamp_ns is not None
            and stamp_ns <= self._last_processed_stamp_ns
        ):
            raise DetectionRejected(
                "Image stamp did not advance; refusing to reuse a frame for stability"
            )
        self._last_processed_stamp_ns = stamp_ns

    def _validated_camera_model(
        self, info_msg: CameraInfo, image_shape: tuple, image_header
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (info_msg.height, info_msg.width) != image_shape[:2]:
            raise DetectionRejected(
                f"CameraInfo dimensions {info_msg.width}x{info_msg.height} do not "
                f"match image {image_shape[1]}x{image_shape[0]}"
            )
        if not image_header.frame_id or image_header.frame_id != info_msg.header.frame_id:
            raise DetectionRejected(
                f"Image/CameraInfo frames differ: {image_header.frame_id!r} versus "
                f"{info_msg.header.frame_id!r}"
            )
        if not _stamp_is_zero(info_msg.header.stamp):
            delta_s = _seconds_between(image_header.stamp, info_msg.header.stamp)
            if delta_s > float(self._parameter("max_camera_info_delta_s")):
                raise DetectionRejected(
                    f"CameraInfo stamp differs from image by {delta_s:.3f}s"
                )

        camera_matrix = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(camera_matrix)):
            raise DetectionRejected("CameraInfo intrinsics contain non-finite values")
        if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
            raise DetectionRejected("CameraInfo focal lengths must be positive")
        if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0]):
            raise DetectionRejected("CameraInfo K matrix bottom row is invalid")

        model = str(info_msg.distortion_model or "")
        coefficients = np.asarray(info_msg.d, dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise DetectionRejected("CameraInfo distortion contains non-finite values")
        if model in ("", "none"):
            if coefficients.size and not np.allclose(coefficients, 0.0):
                raise DetectionRejected(
                    "CameraInfo has distortion coefficients but no supported model"
                )
            return camera_matrix, np.zeros((0, 1), dtype=np.float64)
        if model == "plumb_bob":
            # Some camera drivers pad the five-term model to OpenCV
            # rational-model length. Only exact zero padding is equivalent.
            if coefficients.size == 8 and np.all(coefficients[5:] == 0.0):
                coefficients = coefficients[:5]
            if coefficients.size != 5:
                raise DetectionRejected("plumb_bob distortion must have 5 coefficients")
            return camera_matrix, coefficients.reshape(-1, 1)
        if model == "rational_polynomial":
            if coefficients.size != 8:
                raise DetectionRejected(
                    "rational_polynomial distortion must have 8 coefficients"
                )
            return camera_matrix, coefficients.reshape(-1, 1)
        raise DetectionRejected(f"Unsupported distortion model: {model!r}")

    def _detect_markers(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                gray,
                self._dictionary,
                parameters=self._detector_parameters,
            )
        return corners, ids

    def _target_marker_corners(self, corners, ids) -> np.ndarray:
        if ids is None or len(ids) == 0:
            raise DetectionRejected(
                f"Target marker id {self._target_marker_id} was not detected"
            )
        flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        matches = np.flatnonzero(flat_ids == self._target_marker_id)
        if len(matches) == 0:
            raise DetectionRejected(
                f"Target marker id {self._target_marker_id} was not detected"
            )
        if len(matches) > 1:
            raise DetectionRejected(
                f"Detected marker id {self._target_marker_id} more than once"
            )
        target = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
        edges = np.linalg.norm(np.roll(target, -1, axis=0) - target, axis=1)
        min_side = float(self._parameter("min_marker_side_px"))
        if float(np.min(edges)) < min_side:
            raise DetectionRejected(
                f"Marker side {float(np.min(edges)):.1f}px is below {min_side:.1f}px"
            )
        if not cv2.isContourConvex(target.astype(np.float32)):
            raise DetectionRejected("Detected marker quadrilateral is not convex")
        return target

    def _transform_pose_to_base(self, camera_pose: PoseStamped) -> PoseStamped:
        source_frame = camera_pose.header.frame_id
        target_frame = str(self._parameter("robot_base_frame"))
        if source_frame == target_frame:
            base_pose = PoseStamped()
            base_pose.header = camera_pose.header
            base_pose.pose = camera_pose.pose
            return base_pose
        image_time = Time.from_msg(camera_pose.header.stamp)
        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            image_time,
            timeout=Duration(seconds=float(self._parameter("tf_timeout_s"))),
        )
        self._validate_transform_stamp(transform, camera_pose.header.stamp)
        return _transform_pose(camera_pose, transform)

    def _validate_transform_stamp(self, transform: TransformStamped, image_stamp) -> None:
        if _stamp_is_zero(transform.header.stamp):
            return
        delta_s = _seconds_between(image_stamp, transform.header.stamp)
        if delta_s > float(self._parameter("max_tf_age_s")):
            raise DetectionRejected(
                f"TF {transform.header.frame_id}->{transform.child_frame_id} is stale: "
                f"{delta_s:.3f}s from image stamp"
            )

    def _publish_debug_image(
        self,
        rgb: np.ndarray,
        header,
        status: str,
        *,
        corners: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        estimate: Optional[PoseEstimate] = None,
        published: bool,
    ) -> None:
        if not bool(self._parameter("publish_debug_image")):
            return
        # ``rgb`` is a per-callback decoded frame that is not reused afterwards,
        # so annotate it in place unless the decoder returned a read-only view.
        debug = rgb if rgb.flags.writeable else rgb.copy()
        if corners is not None:
            polygon = corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(debug, [polygon], True, (0, 255, 255), 2)
            for index, point in enumerate(corners):
                cv2.circle(debug, tuple(np.round(point).astype(int)), 3, (0, 128, 255), -1)
                cv2.putText(
                    debug,
                    str(index),
                    tuple(np.round(point + np.array([4.0, -4.0])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
        if (
            estimate is not None
            and camera_matrix is not None
            and distortion is not None
            and hasattr(cv2, "drawFrameAxes")
        ):
            cv2.drawFrameAxes(
                debug,
                camera_matrix,
                distortion,
                estimate.rvec,
                estimate.tvec,
                float(self._parameter("marker_size_m")) * 0.5,
            )
        color = (0, 180, 0) if published else (0, 165, 255)
        self._put_status_lines(debug, status, color)
        debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    def _publish_failure_debug(
        self,
        image_msg: Image | CompressedImage,
        rgb: Optional[np.ndarray],
        status: str,
    ) -> None:
        if not bool(self._parameter("publish_debug_image")) or rgb is None:
            return
        self._publish_debug_image(
            rgb,
            image_msg.header,
            status,
            published=False,
        )

    @staticmethod
    def _put_status_lines(image: np.ndarray, status: str, color) -> None:
        words = status.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 54 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        if not lines:
            lines = [status]
        cv2.rectangle(image, (10, 10), (630, 34 + 24 * len(lines)), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (20, 38 + 24 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

    def _publish_markers(self, pose: PoseStamped) -> None:
        marker_array = MarkerArray()
        lifetime = Duration(seconds=float(self._parameter("marker_lifetime_s"))).to_msg()
        origin = _pose_position(pose)
        rotation = quaternion_to_matrix(_pose_quaternion(pose))
        half = float(self._parameter("marker_size_m")) / 2.0
        corners_local = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )
        marker_array.markers.append(
            self._square_marker(pose.header, origin, rotation, corners_local, lifetime)
        )
        axis_specs = [
            ("X right", rotation[:, 0], (1.0, 0.0, 0.0), 1),
            ("Y up", rotation[:, 1], (0.0, 1.0, 0.0), 2),
            ("Z out", rotation[:, 2], (0.1, 0.3, 1.0), 3),
        ]
        axis_length = float(self._parameter("marker_size_m")) * 0.75
        for label, direction, color, marker_id in axis_specs:
            end = origin + axis_length * direction
            marker_array.markers.append(
                self._axis_marker(pose.header, origin, end, color, marker_id, lifetime)
            )
            marker_array.markers.append(
                self._text_marker(
                    pose.header,
                    end,
                    label,
                    color,
                    marker_id + 10,
                    lifetime,
                    scale=0.025,
                )
            )
        marker_array.markers.append(
            self._text_marker(
                pose.header,
                origin + rotation[:, 2] * axis_length * 1.15,
                f"{self._dictionary_name} id {self._target_marker_id}",
                (1.0, 1.0, 1.0),
                20,
                lifetime,
                scale=0.030,
            )
        )
        self._marker_pub.publish(marker_array)
        self._markers_visible = True

    def _square_marker(
        self,
        header,
        origin: np.ndarray,
        rotation: np.ndarray,
        corners_local: np.ndarray,
        lifetime,
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "aruco_verification"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.points = [
            _point_from_array(origin + rotation @ corner) for corner in corners_local
        ]
        marker.scale.x = 0.004
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = lifetime
        return marker

    @staticmethod
    def _axis_marker(header, start: np.ndarray, end: np.ndarray, color, marker_id: int, lifetime) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "aruco_verification_axes"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [_point_from_array(start), _point_from_array(end)]
        marker.scale.x = 0.006
        marker.scale.y = 0.012
        marker.scale.z = 0.018
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.lifetime = lifetime
        return marker

    @staticmethod
    def _text_marker(
        header,
        position: np.ndarray,
        text: str,
        color,
        marker_id: int,
        lifetime,
        *,
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "aruco_verification_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = _point_from_array(position)
        marker.pose.orientation.w = 1.0
        marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.text = text
        marker.lifetime = lifetime
        return marker

    def _clear_markers(self) -> None:
        if not self._markers_visible or not bool(self._parameter("publish_markers")):
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self._marker_pub.publish(marker_array)
        self._markers_visible = False
        if bool(self._parameter("publish_hover_targets")):
            clear = MarkerArray()
            wipe = Marker()
            wipe.action = Marker.DELETEALL
            clear.markers.append(wipe)
            self._hover_pub.publish(clear)

    def _publish_hover_targets(self, pose: PoseStamped) -> None:
        lifetime = Duration(seconds=float(self._parameter("marker_lifetime_s"))).to_msg()
        origin = _pose_position(pose)
        rotation = quaternion_to_matrix(_pose_quaternion(pose))
        hover_h = float(self._parameter("hover_height_m"))
        array = MarkerArray()
        for index, (name, xyz) in enumerate(hover_test_points(origin, rotation, hover_h)):
            sphere = Marker()
            sphere.header = pose.header
            sphere.ns = "tcp_hover_targets"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = _point_from_array(xyz)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.008
            sphere.color.r = 0.2
            sphere.color.g = 1.0
            sphere.color.b = 0.2
            sphere.color.a = 1.0
            sphere.lifetime = lifetime
            array.markers.append(sphere)
            array.markers.append(
                self._text_marker(
                    pose.header, xyz + rotation[:, 2] * 0.015, name,
                    (0.2, 1.0, 0.2), 100 + index, lifetime, scale=0.015,
                )
            )
        self._hover_pub.publish(array)


def _transform_pose(pose: PoseStamped, transform: TransformStamped) -> PoseStamped:
    translation = transform.transform.translation
    transform_q = np.array(
        [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    rotation = quaternion_to_matrix(transform_q)
    position = rotation @ _pose_position(pose) + np.array(
        [translation.x, translation.y, translation.z], dtype=np.float64
    )
    pose_q = _pose_quaternion(pose)
    transformed_q = quaternion_multiply(transform_q, pose_q)

    transformed = PoseStamped()
    transformed.header.stamp = pose.header.stamp
    transformed.header.frame_id = transform.header.frame_id
    transformed.pose.position = _point_from_array(position)
    transformed.pose.orientation = Quaternion(
        x=float(transformed_q[0]),
        y=float(transformed_q[1]),
        z=float(transformed_q[2]),
        w=float(transformed_q[3]),
    )
    return transformed


def _point_from_array(values: Sequence[float]) -> Point:
    point = Point()
    point.x = float(values[0])
    point.y = float(values[1])
    point.z = float(values[2])
    return point


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoVerificationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
