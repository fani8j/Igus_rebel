import json
import time

import numpy as np
import pytest
import rclpy
from rclpy.node import Node

from box_seam_detection.ros import cloud
from box_seam_detection.ros.capture import CaptureTimeout, SceneCapture

from .synthetic import H, K, W, SyntheticCamera, make_images


def _capture(tmp_path, name, **kw):
    node = Node(name)
    cap = SceneCapture(node, output_root=tmp_path, use_tf=False, **kw)
    return node, cap


def test_fresh_frames_are_captured_and_files_written(tmp_path, spinner):
    cam = SyntheticCamera()
    node, cap = _capture(tmp_path, "cap1")
    spinner(cam, node)
    files = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    assert files.stamp_ns > 0
    assert files.image_path.exists() and files.pcd_path.exists() and files.manifest_path.exists()
    man = json.loads(files.manifest_path.read_text())
    assert man["schema"] == "rgbd_capture/v1" and man["capture_id"] == files.capture_id
    assert man["image"]["header"]["stamp_ns"] == files.stamp_ns
    assert man["camera_info"]["k"] == K
    assert "driver_extrinsic" not in man  # tf disabled here
    rgb, depth = make_images()
    from box_seam_detection.ros.fake_camera import read_pcd_xyzrgb
    xyz, _ = read_pcd_xyzrgb(files.pcd_path)
    assert xyz.shape[0] == man["pointcloud"]["point_count"] == int((depth > 0).sum())
    import cv2
    png = cv2.imread(str(files.image_path), cv2.IMREAD_COLOR)[:, :, ::-1]
    assert (png == rgb).all()
    # a frame is used once: the next grab needs a newer one
    files2 = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    assert files2.stamp_ns > files.stamp_ns and files2.capture_id != files.capture_id


def test_stale_frames_rejected_then_timeout(tmp_path, spinner):
    cam = SyntheticCamera(stamp_offset_s=-5.0)  # camera stamps 5 s in the past
    node, cap = _capture(tmp_path, "cap2")
    spinner(cam, node)
    t0 = time.monotonic()
    with pytest.raises(CaptureTimeout) as ei:
        cap.grab(cap.now_ns(), timeout_s=1.0)
    assert 0.9 < time.monotonic() - t0 < 3.0
    assert "synced" in str(ei.value)
    assert cap.diagnostics()["counts"]["synced"] > 0  # frames arrived, they were just stale
    # ... but stamps after an old request time are accepted
    fs, _ = cap.grab(cap.now_ns() - int(10e9), timeout_s=3.0)
    assert fs.color_stamp_ns > 0


def test_margin_demands_newer_frames(tmp_path, spinner):
    cam = SyntheticCamera()
    node, cap = _capture(tmp_path, "cap3", margin_s=30.0)
    spinner(cam, node)
    with pytest.raises(CaptureTimeout):
        cap.grab(cap.now_ns(), timeout_s=1.0)


def test_no_topics_times_out_with_diagnostics(tmp_path, spinner):
    node, cap = _capture(tmp_path, "cap4")
    spinner(node)
    with pytest.raises(CaptureTimeout, match="camera_info=NO"):
        cap.grab(cap.now_ns(), timeout_s=0.5)


def test_unaligned_depth_refused_and_nothing_written(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2))
    node, cap = _capture(tmp_path, "cap5")
    spinner(cam, node)
    # colour/depth stamps are identical so they sync; the capture itself must refuse the pair
    with pytest.raises(cloud.CaptureError, match="not aligned"):
        cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    assert not list((tmp_path / "pointcloud").glob("*"))
    assert not list((tmp_path / "captures").glob("*"))


def test_float_depth_and_bgr_colour(tmp_path, spinner):
    cam = SyntheticCamera(encoding="bgr8", depth_encoding="32FC1")
    node, cap = _capture(tmp_path, "cap6")
    spinner(cam, node)
    files = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    from box_seam_detection.ros.fake_camera import read_pcd_xyzrgb
    xyz, rgbf = read_pcd_xyzrgb(files.pcd_path)
    rgb, depth = make_images()
    vs, us = np.nonzero(depth > 0)
    np.testing.assert_allclose(xyz[:, 2], depth[vs, us] * 0.001, rtol=1e-6)
    assert (rgbf[0] & 0xFFFFFF) == (int(rgb[vs[0], us[0], 0]) << 16 | int(rgb[vs[0], us[0], 1]) << 8
                                    | int(rgb[vs[0], us[0], 2]))
    man = json.loads(files.manifest_path.read_text())
    assert man["pointcloud"]["source"]["depth_encoding"] == "32FC1"
    assert man["image"]["encoding"] == "bgr8"


def test_driver_extrinsic_from_tf(tmp_path, spinner):
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    cam = SyntheticCamera()
    node = Node("cap7")
    cap = SceneCapture(node, output_root=tmp_path, use_tf=True, tf_timeout_s=3.0)
    tfb = StaticTransformBroadcaster(node)
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = "camera_link", "camera_color_optical_frame"
    t.transform.translation.x = 0.0016
    t.transform.rotation.w = 1.0
    tfb.sendTransform(t)
    spinner(cam, node)
    files = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    ext = json.loads(files.manifest_path.read_text())["driver_extrinsic"]
    assert ext["parent_frame"] == "camera_link" and ext["child_frame"] == "camera_color_optical_frame"
    assert ext["translation"][0] == pytest.approx(0.0016) and ext["source"] == "tf_live"
    assert ext["stamp_ns"] == files.stamp_ns


def test_extrinsic_required_fails_closed(tmp_path, spinner):
    cam = SyntheticCamera(frame="some_unlinked_optical_frame")
    node = Node("cap8")
    cap = SceneCapture(node, output_root=tmp_path, use_tf=True, tf_timeout_s=0.3, extrinsic_required=True)
    spinner(cam, node)
    with pytest.raises(cloud.CaptureError, match="driver extrinsic"):
        cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
