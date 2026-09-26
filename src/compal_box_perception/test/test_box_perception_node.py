"""Node-level snapshot tests with synthetic RGB-D frames (no spin, no camera).

The RGB detector is replaced by a fixed synthetic result so these tests
exercise the node's depth-plane, TF, snapshot and IO path deterministically.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import rclpy
import yaml
from compal_box_msgs.srv import CaptureBoxSnapshot
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo

from compal_box_perception import detect_tape_seam
from compal_box_perception.box_perception_node import BoxPerceptionNode

WIDTH, HEIGHT = 160, 120
FX = FY = 100.0
CX, CY = 80.0, 60.0
CAMERA_FRAME = "test_camera_optical_frame"
BASE_FRAME = "base_link"
CARTON_DEPTH_M = 0.8
BACKGROUND_DEPTH_M = 1.5
# Carton top face in pixels (x0, y0, x1, y1); the seam runs along v == CY.
TOP_FACE = (40, 40, 120, 80)
CONFIG = Path(__file__).resolve().parents[1] / "config" / "box_perception.yaml"


def _fake_detection(_rgb, roi):
    x0, y0, x1, y1 = TOP_FACE
    return {
        "roi": tuple(int(value) for value in roi),
        "top_face_corners": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "left_edge": [[x0, y0], [x0, y1]],
        "right_edge": [[x1, y0], [x1, y1]],
        "back_edge": [[x0, y0], [x1, y0]],
        "front_edge": [[x0, y1], [x1, y1]],
        "seam_line": [[x0, CY], [x1, CY]],
        "back_left": [x0, y0],
        "back_right": [x1, y0],
        "front_left": [x0, y1],
        "front_right": [x1, y1],
    }


def _depth_image(tilt_deg):
    """Carton plane ``z - tan(tilt) * y = CARTON_DEPTH_M`` over a far background.

    The tilt is about the camera x axis, so both seam endpoints (v == CY) keep
    the same depth and only the plane normal changes between frames.
    """
    depth = np.full((HEIGHT, WIDTH), BACKGROUND_DEPTH_M, dtype=np.float32)
    x0, y0, x1, y1 = TOP_FACE
    v = np.arange(y0, y1, dtype=np.float64)[:, None]
    yn = (v - CY) / FY
    carton = CARTON_DEPTH_M / (1.0 - math.tan(math.radians(tilt_deg)) * yn)
    depth[y0:y1, x0:x1] = np.broadcast_to(carton, (y1 - y0, x1 - x0))
    return depth


def _expected_base_normal(tilt_deg):
    # Camera-frame normal facing the camera is (0, tan, -1); base_link is the
    # camera rotated 180 deg about x, i.e. diag(1, -1, -1).
    tangent = math.tan(math.radians(tilt_deg))
    normal = np.array([0.0, -tangent, 1.0])
    return normal / np.linalg.norm(normal)


@pytest.fixture
def perception(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_tape_seam, "detect_seam", _fake_detection)
    context = Context()
    rclpy.init(args=None, context=context)
    overrides = [
        Parameter("robot_base_frame", Parameter.Type.STRING, BASE_FRAME),
        Parameter("use_auto_roi", Parameter.Type.BOOL, False),
        Parameter("manual_roi", Parameter.Type.INTEGER_ARRAY, [30, 30, 130, 90]),
        Parameter(
            "manual_roi_reference_size", Parameter.Type.INTEGER_ARRAY, [WIDTH, HEIGHT]
        ),
        Parameter("plane_min_inliers", Parameter.Type.INTEGER, 50),
        Parameter("publish_local_pointcloud", Parameter.Type.BOOL, False),
        Parameter("snapshot_output_dir", Parameter.Type.STRING, str(tmp_path / "out")),
    ]
    node = BoxPerceptionNode(context=context, parameter_overrides=overrides)
    transform = TransformStamped()
    transform.header.frame_id = BASE_FRAME
    transform.child_frame_id = CAMERA_FRAME
    transform.transform.translation.z = 1.0
    transform.transform.rotation.x = 1.0  # looking straight down
    transform.transform.rotation.w = 0.0
    node._tf_buffer.set_transform_static(transform, "test")
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)


def _cache_frame(node, tilt_deg, age_s=0.0):
    bridge = CvBridge()
    stamp = node.get_clock().now().to_msg()
    rgb_msg = bridge.cv2_to_imgmsg(
        np.full((HEIGHT, WIDTH, 3), 180, dtype=np.uint8), encoding="bgr8"
    )
    depth_msg = bridge.cv2_to_imgmsg(_depth_image(tilt_deg), encoding="32FC1")
    info = CameraInfo()
    info.width, info.height = WIDTH, HEIGHT
    info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    info.d = [0.0] * 5
    info.distortion_model = "plumb_bob"
    for message in (rgb_msg, depth_msg, info):
        message.header.frame_id = CAMERA_FRAME
        message.header.stamp = stamp
    received = node.get_clock().now() - Duration(seconds=age_s)
    node._latest_rgbd = (rgb_msg, depth_msg, info, received)


def _capture(node):
    return node._capture_snapshot(
        CaptureBoxSnapshot.Request(), CaptureBoxSnapshot.Response()
    )


def _normal(response):
    normal = response.snapshot.surface_normal
    return np.array([normal.x, normal.y, normal.z])


def test_capture_reconstructs_synthetic_top_face(perception, tmp_path):
    _cache_frame(perception, tilt_deg=0.0)
    response = _capture(perception)

    assert response.success, response.message
    snapshot = response.snapshot
    assert snapshot.header.frame_id == BASE_FRAME
    np.testing.assert_allclose(_normal(response), [0.0, 0.0, 1.0], atol=1e-4)
    # Seam at depth 0.8 m below a camera 1.0 m above base: z = 0.2 m, and
    # 80 px at fx=100 spans 0.64 m.
    for point in (snapshot.seam_start, snapshot.seam_end, snapshot.seam_center):
        assert point.z == pytest.approx(1.0 - CARTON_DEPTH_M, abs=1e-3)
    length = math.dist(
        (snapshot.seam_start.x, snapshot.seam_start.y),
        (snapshot.seam_end.x, snapshot.seam_end.y),
    )
    assert length == pytest.approx(0.64, abs=2e-3)

    report_path = tmp_path / "out" / f"{snapshot.snapshot_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["surface_normal"]["z"] == pytest.approx(snapshot.surface_normal.z)
    assert Path(report["rgb_path"]).is_file()
    assert Path(report["debug_path"]).is_file()


def test_new_capture_never_inherits_previous_carton_normal(perception):
    # Regression: a temporal normal filter kept across one-shot captures made
    # a second carton tilted beyond its jump limit report the first carton's
    # normal. Each capture must use only its own frame.
    _cache_frame(perception, tilt_deg=0.0)
    first = _capture(perception)
    assert first.success, first.message

    _cache_frame(perception, tilt_deg=15.0)
    second = _capture(perception)
    assert second.success, second.message

    angle = math.degrees(
        math.acos(np.clip(np.dot(_normal(second), _expected_base_normal(15.0)), -1, 1))
    )
    assert angle < 0.5
    assert np.dot(_normal(first), _normal(second)) < math.cos(math.radians(14.0))


def test_capture_rejects_stale_frame(perception):
    assert perception.get_parameter("snapshot_max_receipt_age_s").value == 1.0
    _cache_frame(perception, tilt_deg=0.0, age_s=2.0)
    response = _capture(perception)
    assert not response.success
    assert "old" in response.message


def test_snapshot_io_failure_returns_failure(perception, tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    perception.set_parameters(
        [Parameter("snapshot_output_dir", Parameter.Type.STRING, str(blocker / "sub"))]
    )
    _cache_frame(perception, tilt_deg=0.0)
    response = _capture(perception)
    assert not response.success
    assert "not saved" in response.message


def test_shipped_config_only_sets_declared_parameters(perception):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parameters = config["box_perception"]["ros__parameters"]
    undeclared = sorted(
        name for name in parameters if not perception.has_parameter(name)
    )
    assert undeclared == []
    assert parameters["snapshot_max_receipt_age_s"] <= 1.0
