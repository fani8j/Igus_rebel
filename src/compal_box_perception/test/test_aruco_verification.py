import math
import time

import cv2
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from compal_box_perception.aruco_verification import (
    ArucoVerificationNode,
    DetectionRejected,
    StablePoseWindow,
    _solve_marker_pose_ippe,
)


CAMERA_MATRIX = np.array(
    [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DISTORTION = np.zeros((0, 1), dtype=np.float64)
MARKER_SIZE_M = 0.060
MARKER_ID = 7
DICTIONARY_NAME = "DICT_4X4_50"


def _contract_marker_object_points(marker_size_m):
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


def _front_facing_tilted_pose():
    front_parallel = np.diag([1.0, -1.0, -1.0])
    tilt, _ = cv2.Rodrigues(np.array([[0.4], [0.2], [0.0]], dtype=np.float64))
    rotation = tilt @ front_parallel
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.02], [0.01], [0.55]], dtype=np.float64)
    return rvec, tvec, rotation


def _projected_marker_corners():
    rvec, tvec, _rotation = _front_facing_tilted_pose()
    corners, _ = cv2.projectPoints(
        _contract_marker_object_points(MARKER_SIZE_M),
        rvec,
        tvec,
        CAMERA_MATRIX,
        DISTORTION,
    )
    return corners.reshape(4, 2), tvec


def _marker_image(size_px=240):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, DICTIONARY_NAME)
    )
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, MARKER_ID, size_px)
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, MARKER_ID, size_px, marker, 1)
    return marker


def _synthetic_compressed_marker_image():
    corners, _tvec = _projected_marker_corners()
    source = np.array([[0, 0], [239, 0], [239, 239], [0, 239]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    marker = _marker_image()
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(
        cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR),
        homography,
        (640, 480),
        flags=cv2.INTER_NEAREST,
        borderValue=(255, 255, 255),
    )
    mask = cv2.warpPerspective(
        np.full(marker.shape, 255, np.uint8),
        homography,
        (640, 480),
        flags=cv2.INTER_NEAREST,
    )
    image[mask > 0] = warped[mask > 0]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _pose(x=0.0, y=0.0, z=0.0, quaternion=(0.0, 0.0, 0.0, 1.0)):
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose

def _stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stamp_from_ns(nanoseconds):
    stamp = TimeMsg()
    stamp.sec = int(nanoseconds // 1_000_000_000)
    stamp.nanosec = int(nanoseconds % 1_000_000_000)
    return stamp


def _next_stamp_after(node, previous):
    now_ns = _stamp_to_ns(node.get_clock().now().to_msg())
    previous_ns = _stamp_to_ns(previous)
    return _stamp_from_ns(max(now_ns, previous_ns + 1_000_000))


def _camera_info(stamp):
    info_msg = CameraInfo()
    info_msg.header.frame_id = "camera_color_optical_frame"
    info_msg.header.stamp = stamp
    info_msg.width = 640
    info_msg.height = 480
    info_msg.k = CAMERA_MATRIX.reshape(-1).tolist()
    info_msg.distortion_model = "plumb_bob"
    info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return info_msg


def _compressed_image(stamp):
    image_msg = CompressedImage()
    image_msg.header.frame_id = "camera_color_optical_frame"
    image_msg.header.stamp = stamp
    image_msg.format = "png"
    image_msg.data = _synthetic_compressed_marker_image()
    return image_msg


def _static_camera_transform():
    transform = TransformStamped()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = "camera_color_optical_frame"
    transform.transform.rotation.w = 1.0
    return transform

def _point_xyz(point):
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _pose_origin(pose):
    return _point_xyz(pose.pose.position)


def _rotation_matrix_from_orientation(orientation):
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    assert norm > 0.0
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _marker_color(marker):
    return np.array([marker.color.r, marker.color.g, marker.color.b], dtype=np.float64)


def _spin_for(executor, duration_s):
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))


def _spin_until(executor, predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def _publish_frame(info_pub, image_pub, stamp, executor=None):
    info_pub.publish(_camera_info(stamp))
    if executor is not None:
        _spin_for(executor, 0.05)
    image_pub.publish(_compressed_image(stamp))


def test_ippe_pose_uses_centered_camera_facing_marker_frame():
    corners, expected_tvec = _projected_marker_corners()

    estimate = _solve_marker_pose_ippe(
        corners,
        CAMERA_MATRIX,
        DISTORTION,
        MARKER_SIZE_M,
        max_reprojection_error_px=1.0,
        min_ippe_error_ratio=1.5,
    )

    np.testing.assert_allclose(estimate.tvec, expected_tvec, atol=1e-9)
    assert estimate.reprojection_error_px < 1e-9
    assert estimate.alternate_reprojection_error_px > 1.0
    assert np.dot(estimate.rotation_matrix[:, 2], estimate.tvec.reshape(3)) < 0.0


def test_ippe_error_ratio_rejects_untrusted_planar_ambiguity():
    corners, _expected_tvec = _projected_marker_corners()

    with pytest.raises(DetectionRejected, match="Ambiguous IPPE pose"):
        _solve_marker_pose_ippe(
            corners,
            CAMERA_MATRIX,
            DISTORTION,
            MARKER_SIZE_M,
            max_reprojection_error_px=1.0,
            min_ippe_error_ratio=1e12,
        )


def test_stable_window_resets_on_position_or_so3_jump():
    window = StablePoseWindow(
        required_count=2,
        max_position_delta_m=0.010,
        max_rotation_delta_deg=5.0,
    )
    assert window.update(_pose()) == 1
    assert window.update(_pose(x=0.005)) == 2
    assert window.stable()

    assert window.update(_pose(x=0.050)) == 1
    assert not window.stable()

    yaw = math.radians(12.0)
    quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    assert window.update(_pose(quaternion=quaternion)) == 1
    assert not window.stable()


def test_transport_publishes_only_after_fresh_monotonic_stable_frames():
    context = Context()
    rclpy.init(args=None, context=context)
    rgb_topic = "/test_aruco_verification/rgb/compressed"
    info_topic = "/test_aruco_verification/camera_info"
    overrides = [
        Parameter("rgb_topic", Parameter.Type.STRING, rgb_topic),
        Parameter("camera_info_topic", Parameter.Type.STRING, info_topic),
        Parameter("marker_dictionary", Parameter.Type.STRING, DICTIONARY_NAME),
        Parameter("marker_id", Parameter.Type.INTEGER, MARKER_ID),
        Parameter("stable_count", Parameter.Type.INTEGER, 2),
        Parameter("max_image_age_s", Parameter.Type.DOUBLE, 5.0),
    ]
    verifier = ArucoVerificationNode(context=context, parameter_overrides=overrides)
    driver = rclpy.create_node("aruco_transport_contract_driver", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(verifier)
    executor.add_node(driver)
    try:
        pose_messages = []
        debug_messages = []
        marker_messages = []
        driver.create_subscription(
            PoseStamped,
            "/calibration_verification/board_pose",
            pose_messages.append,
            10,
        )
        driver.create_subscription(
            Image,
            "/aruco_verification/debug_image",
            debug_messages.append,
            10,
        )
        driver.create_subscription(
            MarkerArray,
            "/calibration_verification/detection_markers",
            marker_messages.append,
            10,
        )
        image_pub = driver.create_publisher(
            CompressedImage, rgb_topic, qos_profile_sensor_data
        )
        info_pub = driver.create_publisher(
            CameraInfo, info_topic, qos_profile_sensor_data
        )
        broadcaster = StaticTransformBroadcaster(driver)

        result = verifier.set_parameters(
            [Parameter("marker_id", Parameter.Type.INTEGER, MARKER_ID + 1)]
        )[0]
        assert not result.successful
        assert verifier.get_parameter("marker_id").value == MARKER_ID

        assert _spin_until(
            executor,
            lambda: image_pub.get_subscription_count() > 0
            and info_pub.get_subscription_count() > 0,
        )
        broadcaster.sendTransform(_static_camera_transform())
        _spin_for(executor, 0.2)

        first_stamp = driver.get_clock().now().to_msg()
        _publish_frame(info_pub, image_pub, first_stamp, executor)
        _spin_for(executor, 0.25)
        assert pose_messages == []

        _publish_frame(info_pub, image_pub, first_stamp, executor)
        _spin_for(executor, 0.25)
        assert pose_messages == []

        second_stamp = _next_stamp_after(driver, first_stamp)
        _publish_frame(info_pub, image_pub, second_stamp, executor)
        _spin_for(executor, 0.25)
        assert pose_messages == []

        third_stamp = _next_stamp_after(driver, second_stamp)
        _publish_frame(info_pub, image_pub, third_stamp, executor)
        assert _spin_until(
            executor,
            lambda: len(pose_messages) == 1 and len(marker_messages) == 1,
        )

        pose = pose_messages[0]
        assert pose.header.frame_id == "base_link"
        assert pose.header.stamp == third_stamp
        assert pose.pose.position.z == pytest.approx(0.55, abs=0.010)
        assert debug_messages

        markers = marker_messages[0].markers
        assert all(marker.header.frame_id == "base_link" for marker in markers)
        assert all(marker.header.stamp == third_stamp for marker in markers)

        square = next(marker for marker in markers if marker.type == Marker.LINE_STRIP)
        square_points = np.array([_point_xyz(point) for point in square.points])
        assert square_points.shape == (5, 3)
        np.testing.assert_allclose(square_points[0], square_points[-1], atol=1e-9)
        edge_lengths = np.linalg.norm(np.diff(square_points, axis=0), axis=1)
        np.testing.assert_allclose(edge_lengths, MARKER_SIZE_M, atol=0.002)
        origin = _pose_origin(pose)
        np.testing.assert_allclose(np.mean(square_points[:4], axis=0), origin, atol=0.002)

        rotation = _rotation_matrix_from_orientation(pose.pose.orientation)
        arrows = [marker for marker in markers if marker.type == Marker.ARROW]
        assert len(arrows) == 3
        axis_specs = [
            (np.array([1.0, 0.0, 0.0]), rotation[:, 0]),
            (np.array([0.0, 1.0, 0.0]), rotation[:, 1]),
            (np.array([0.1, 0.3, 1.0]), rotation[:, 2]),
        ]
        for expected_color, expected_direction in axis_specs:
            arrow = next(
                marker
                for marker in arrows
                if np.allclose(_marker_color(marker), expected_color, atol=1e-6)
            )
            assert len(arrow.points) == 2
            np.testing.assert_allclose(_point_xyz(arrow.points[0]), origin, atol=0.002)
            vector = _point_xyz(arrow.points[1]) - _point_xyz(arrow.points[0])
            assert np.linalg.norm(vector) == pytest.approx(
                MARKER_SIZE_M * 0.75, abs=0.002
            )
            direction = vector / np.linalg.norm(vector)
            assert np.dot(direction, expected_direction) > 0.999

        labels = {marker.text for marker in markers if marker.type == Marker.TEXT_VIEW_FACING}
        assert {"X right", "Y up", "Z out", f"{DICTIONARY_NAME} id {MARKER_ID}"} <= labels
    finally:
        executor.shutdown()
        verifier.destroy_node()
        driver.destroy_node()
        rclpy.shutdown(context=context)


def test_default_configuration_idles_without_marker_guessing():
    context = Context()
    rclpy.init(args=None, context=context)
    rgb_topic = "/test_aruco_verification/default/rgb/compressed"
    overrides = [
        Parameter("rgb_topic", Parameter.Type.STRING, rgb_topic),
    ]
    verifier = ArucoVerificationNode(context=context, parameter_overrides=overrides)
    driver = rclpy.create_node("aruco_default_idle_driver", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(verifier)
    executor.add_node(driver)
    try:
        pose_messages = []
        debug_messages = []
        driver.create_subscription(
            PoseStamped,
            "/calibration_verification/board_pose",
            pose_messages.append,
            10,
        )
        driver.create_subscription(
            Image,
            "/aruco_verification/debug_image",
            debug_messages.append,
            10,
        )
        image_pub = driver.create_publisher(
            CompressedImage, rgb_topic, qos_profile_sensor_data
        )

        assert _spin_until(executor, lambda: image_pub.get_subscription_count() > 0)
        stamp = driver.get_clock().now().to_msg()
        image_pub.publish(_compressed_image(stamp))

        assert _spin_until(executor, lambda: len(debug_messages) == 1)
        _spin_for(executor, 0.2)
        assert pose_messages == []
        assert debug_messages[0].header.stamp == stamp
    finally:
        executor.shutdown()
        verifier.destroy_node()
        driver.destroy_node()
        rclpy.shutdown(context=context)

def test_zero_padded_plumb_bob_preserves_marker_pose():
    info = _camera_info(TimeMsg())
    info.d = [0.007052, -0.049205, 0.0002613, 0.00009177, 0.033208, 0.0, 0.0, 0.0]
    rvec, tvec, rotation = _front_facing_tilted_pose()
    corners, _ = cv2.projectPoints(
        _contract_marker_object_points(MARKER_SIZE_M), rvec, tvec,
        CAMERA_MATRIX, np.asarray(info.d[:5]))
    matrix, distortion = ArucoVerificationNode._validated_camera_model(
        None, info, (480, 640, 3), info.header)
    pose = _solve_marker_pose_ippe(
        corners.reshape(4, 2), matrix, distortion, MARKER_SIZE_M, 1.0, 1.5)
    np.testing.assert_allclose(pose.tvec, tvec, atol=1e-6)
    np.testing.assert_allclose(pose.rotation_matrix, rotation, atol=1e-5)


@pytest.mark.parametrize("coefficients", [
    [0.0] * 4, [0.0] * 6, [0.0] * 7 + [1e-10], [0.0] * 7 + [float("nan")],
])
def test_plumb_bob_rejects_invalid_or_nonzero_extra_terms(coefficients):
    info = _camera_info(TimeMsg())
    info.d = coefficients
    with pytest.raises(DetectionRejected):
        ArucoVerificationNode._validated_camera_model(
            None, info, (480, 640, 3), info.header)
