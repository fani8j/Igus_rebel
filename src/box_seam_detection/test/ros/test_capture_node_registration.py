"""SceneCapture node-side depth registration path (second synchroniser: colour + raw depth)."""
import json
import time
import types

import numpy as np
import pytest
from rclpy.node import Node

from box_seam_detection.ros import cloud
from box_seam_detection.ros.capture import CaptureTimeout, SceneCapture
from box_seam_detection.ros.fake_camera import read_pcd_xyzrgb

from .synthetic import (BASELINE_X, H, K, K_RAW, RAW_FRAME, RAW_MM, RH, RW, W, SyntheticCamera, make_images,
                        static_tf_buffer)

COLOR_FRAME = "camera_color_optical_frame"


def _cap(tmp_path, name, tf=True, **kw):
    node = Node(name)
    if tf is True:
        kw["tf_buffer"] = static_tf_buffer()
    elif tf is not None:
        kw["tf_buffer"] = tf
    kw.setdefault("use_tf", False)
    return node, SceneCapture(node, output_root=tmp_path, **kw)


def _expected_registration():
    """The registration of the synthetic raw depth, computed directly (the capture must produce the same cloud)."""
    raw = np.full((RH, RW), RAW_MM, dtype=np.uint16)
    raw[:3, :] = 0
    T = cloud.transform_matrix([BASELINE_X, 0, 0], [0, 0, 0, 1])
    return cloud.register_depth_to_color(raw, 0.001, K_RAW, [0.0] * 5, K, [0.0] * 5, "plumb_bob", T, W, H)


def test_node_registration_when_driver_never_aligns(tmp_path, spinner):
    # image_raw always carries an unaligned (2x) depth -> the driver path never has a candidate
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr1")
    spinner(cam, node)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.source == "node" and fs.depth_info.width == RW and fs.extrinsic["parent_frame"] == COLOR_FRAME
    assert fs.extrinsic["child_frame"] == RAW_FRAME
    files = cap.write_capture(fs, info)
    assert files.image_path.exists() and files.pcd_path.exists() and files.manifest_path.exists()
    expected = _expected_registration()
    n = int((expected > 0).sum())
    assert 0 < n < W * H           # the raw FOV is narrower than the colour one: not every colour pixel has depth
    man = json.loads(files.manifest_path.read_text())
    assert man["schema"] == "rgbd_capture/v1"
    assert (man["image"]["width"], man["image"]["height"]) == (W, H)
    assert man["pointcloud"]["point_count"] == n
    xyz, _ = read_pcd_xyzrgb(files.pcd_path)
    assert xyz.shape[0] == n
    np.testing.assert_allclose(xyz[:, 2], RAW_MM * 0.001, atol=1e-3)          # plane at 1 m, no z in the extrinsic
    vs, us = np.nonzero(expected)
    np.testing.assert_allclose(xyz[:, 0], (us - K[2]) * 1.0 / K[0], atol=2e-3)  # colour-frame x of the (u, v) pixel
    # manifest: a node-registered cloud lives in the COLOUR frame (consumers require image == cloud == camera_info)
    assert man["pointcloud"]["header"]["frame_id"] == COLOR_FRAME == man["image"]["header"]["frame_id"]
    assert man["pointcloud"]["header"]["stamp_ns"] == fs.depth_stamp_ns
    assert man["camera_info"]["k"] == K and man["camera_info"]["width"] == W
    assert man["topics"]["depth"] == "/camera/depth/image_unaligned"
    reg = man["depth_registration"]
    assert reg["source"] == "node" and reg["raw_depth_topic"] == "/camera/depth/image_unaligned"
    assert reg["depth_info_topic"] == "/camera/depth/camera_info" and reg["raw_sync_slop_s"] == 0.03
    assert reg["depth_k"] == K_RAW and abs(reg["color_depth_dt_ms"]) < 5.0
    assert reg["t_color_depth"]["parent_frame"] == COLOR_FRAME and reg["t_color_depth"]["child_frame"] == RAW_FRAME
    assert reg["t_color_depth"]["translation"] == pytest.approx([BASELINE_X, 0, 0])
    assert reg["stats"]["output_valid"] == n and reg["raw_depth_header"]["frame_id"] == RAW_FRAME
    # the colour PNG is the colour frame, untouched
    import cv2
    png = cv2.imread(str(files.image_path), cv2.IMREAD_COLOR)[:, :, ::-1]
    assert (png == make_images()[0]).all()
    d = cap.diagnostics()
    assert d["counts"]["captures_node"] == 1 and d["counts"]["captures_driver"] == 0
    assert d["counts"]["raw_synced"] >= 1 and d["node_registration"]["have_extrinsic"]
    # the depth CameraInfo of the aligned (colour) size was NOT used: the depth intrinsics are the raw-size ones
    assert reg["depth_k"][0] == K_RAW[0]
    # a consumer of the manifest accepts it as is
    from box_seam_detection.live.query_depth import load_capture_manifest
    load_capture_manifest(str(files.manifest_path), str(files.pcd_path))


def test_capture_helper_end_to_end_and_used_once(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr1b")
    spinner(cam, node)
    f1 = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    f2 = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    assert f2.stamp_ns > f1.stamp_ns and f1.capture_id != f2.capture_id
    assert cap.diagnostics()["counts"]["captures_node"] == 2


def test_raw_depth_without_colour_inside_the_slop_is_not_used(tmp_path, spinner):
    # raw depth stamps are 3 s in the future: no colour frame within the sync slop (and the queue is far too short to wait)
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_stamp_offset_s=3.0)
    node, cap = _cap(tmp_path, "nr2")
    spinner(cam, node)
    with pytest.raises(CaptureTimeout) as ei:
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.5)
    c = cap.diagnostics()["counts"]
    assert c["raw_depth"] > 0 and c["raw_synced"] == 0 and c["captures_node"] == 0
    assert "0 synced with colour" in str(ei.value)
    assert not list((tmp_path / "pointcloud").glob("*"))


def test_missing_tf_gives_no_node_capture_and_a_clear_reason(tmp_path, spinner):
    import tf2_ros
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr3", tf=tf2_ros.Buffer())      # empty buffer: no transform
    spinner(cam, node)
    with pytest.raises(CaptureTimeout) as ei:
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.5)
    msg = str(ei.value)
    assert f"no tf {COLOR_FRAME} <- {RAW_FRAME}" in msg and "not usable" in msg
    c = cap.diagnostics()["counts"]
    assert c["raw_synced"] > 0 and c["captures_node"] == 0
    assert not list((tmp_path / "captures").glob("*"))


def test_tf_disabled_is_reported(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr3b", tf=None)
    spinner(cam, node)
    with pytest.raises(CaptureTimeout, match="tf disabled"):
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.0)


def test_tf_appearing_later_recovers(tmp_path, spinner):
    import tf2_ros
    buf = tf2_ros.Buffer()
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr3c", tf=buf)
    spinner(cam, node)
    with pytest.raises(CaptureTimeout):
        cap.grab(cap.now_ns(), timeout_s=0.6)
    good = static_tf_buffer()
    buf.set_transform_static(good.lookup_transform(COLOR_FRAME, RAW_FRAME, __import__("rclpy").time.Time()), "late")
    fs, _ = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.source == "node"


def test_colour_size_equals_raw_size_refuses_the_node_path(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(W, H))
    node, cap = _cap(tmp_path, "nr4")
    spinner(cam, node)
    with pytest.raises(CaptureTimeout) as ei:
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.5)
    c = cap.diagnostics()["counts"]
    assert c["raw_rejected_same_size"] > 0 and c["captures_node"] == 0
    assert "same_size" in str(ei.value) and "ambiguous" in str(ei.value)


def test_same_frame_id_refused(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_frame=COLOR_FRAME)
    node, cap = _cap(tmp_path, "nr4b")
    spinner(cam, node)
    with pytest.raises(CaptureTimeout, match="same_frame"):
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.0)


def test_stale_raw_sets_rejected_then_accepted_for_an_old_request(tmp_path, spinner):
    cam = SyntheticCamera(stamp_offset_s=-5.0, depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr5")
    spinner(cam, node)
    with pytest.raises(CaptureTimeout):
        cap.grab(cap.now_ns(), timeout_s=1.0)
    assert cap.diagnostics()["counts"]["raw_synced"] > 0         # arrived, just stale
    fs, _ = cap.grab(cap.now_ns() - int(10e9), timeout_s=3.0)
    assert fs.source == "node" and fs.color_stamp_ns > 0


def test_freshness_requires_the_raw_depth_stamp_too(tmp_path, spinner):
    # raw depth is stamped 15 ms before its colour partner (inside the 30 ms slop; the next raw frame is 35 ms away at
    # 20 Hz, so the pairing is unambiguous); with a 30 ms margin a set is only fresh when the RAW stamp exceeds the
    # threshold, i.e. the colour stamp must exceed it by > 15 ms
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_stamp_offset_s=-0.015)
    node, cap = _cap(tmp_path, "nr6", margin_s=0.03)
    spinner(cam, node)
    for _ in range(6):
        thr = cap.now_ns()
        fs, _ = cap.grab(thr, timeout_s=5.0)
        assert fs.source == "node"
        assert fs.depth_stamp_ns > thr + int(0.03e9)
        assert fs.color_stamp_ns - fs.depth_stamp_ns == pytest.approx(0.015e9, abs=3e6)
        time.sleep(0.05)


def test_driver_aligned_sets_still_work_with_raw_path_enabled(tmp_path, spinner):
    cam = SyntheticCamera()                                    # aligned depth, no raw stream at all
    node, cap = _cap(tmp_path, "nr7")
    spinner(cam, node)
    files = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    man = json.loads(files.manifest_path.read_text())
    assert man["depth_registration"]["source"] == "driver"
    assert man["depth_registration"]["depth_topic"] == "/camera/depth/image_raw"
    assert man["pointcloud"]["point_count"] == int((make_images()[1] > 0).sum())
    d = cap.diagnostics()
    assert d["counts"]["captures_driver"] == 1 and d["counts"]["captures_node"] == 0


def test_mixed_stream_returns_the_earliest_fresh_set_of_either_kind(tmp_path, spinner):
    cam = SyntheticCamera(raw_size=(RW, RH))                   # aligned AND raw frames every tick, same stamps
    node, cap = _cap(tmp_path, "nr8")
    spinner(cam, node)
    fs, _ = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.source == "driver"                               # same colour frame: the driver set wins the tie
    # the frame is used once, whichever kind: the next set is a different (newer) colour frame
    fs2, _ = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs2.color_stamp_ns > fs.color_stamp_ns


def test_mixed_stream_driver_rarely_aligns_node_fills_in(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), aligned_every=10 ** 6, raw_size=(RW, RH), raw_every=3)
    node, cap = _cap(tmp_path, "nr9")
    spinner(cam, node)
    sources = {cap.grab(cap.now_ns(), timeout_s=5.0)[0].source for _ in range(3)}
    assert sources == {"node"}


def test_node_registration_disabled_by_parameter(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr10", node_depth_registration=False)
    spinner(cam, node)
    with pytest.raises(CaptureTimeout, match="node registration disabled"):
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.0)
    assert cap.diagnostics()["counts"]["raw_depth"] == 0 and not hasattr(cap, "_raw_sync")


def test_bgr_colour_with_raw_depth(tmp_path, spinner):
    cam = SyntheticCamera(encoding="bgr8", depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr11")
    spinner(cam, node)
    files = cap.capture(after_ns=cap.now_ns(), timeout_s=5.0)
    assert json.loads(files.manifest_path.read_text())["image"]["encoding"] == "bgr8"


def test_write_capture_refuses_a_mismatched_colour_info_and_writes_nothing(tmp_path, spinner):
    from sensor_msgs.msg import CameraInfo
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr12")
    spinner(cam, node)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    bad = CameraInfo(width=W * 2, height=H * 2, distortion_model="plumb_bob", d=[0.0] * 5, k=K)
    bad.header = info.header
    with pytest.raises(cloud.CaptureError, match="exact colour pixel grid"):
        cap.write_capture(fs, bad)
    wrong_frame = CameraInfo(width=W, height=H, distortion_model="plumb_bob", d=[0.0] * 5, k=K)
    wrong_frame.header.frame_id = "other"
    with pytest.raises(cloud.CaptureError, match="frame"):
        cap.write_capture(fs, wrong_frame)
    assert not list((tmp_path / "pointcloud").glob("*")) and not list((tmp_path / "captures").glob("*"))


def test_write_capture_refuses_depth_with_lens_distortion(tmp_path, spinner):
    import dataclasses
    from sensor_msgs.msg import CameraInfo
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr13")
    spinner(cam, node)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    di = CameraInfo(width=RW, height=RH, distortion_model="plumb_bob", d=[0.1, 0, 0, 0, 0], k=K_RAW)
    with pytest.raises(cloud.CaptureError, match="non-zero distortion"):
        cap.write_capture(dataclasses.replace(fs, depth_info=di), info)
    assert not list((tmp_path / "captures").glob("*"))


def test_unknown_colour_distortion_model_fails_closed(tmp_path, spinner):
    import dataclasses
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "nr14")
    spinner(cam, node)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    info = type(info)(width=info.width, height=info.height, distortion_model="fisheye", d=[0.0] * 4, k=K)
    info.header = fs.color.header
    with pytest.raises(cloud.CaptureError, match="distortion_model"):
        cap.write_capture(fs, info)


FALLBACK = [float(RW), float(RH), K_RAW[0], K_RAW[4], K_RAW[2], K_RAW[5]]      # [w, h, fx, fy, cx, cy] of the raw camera


def test_config_fallback_used_when_no_raw_size_camera_info_arrives(tmp_path, spinner):
    # like the live driver while its alignment works: depth/camera_info only at the aligned (colour) size
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_info=False)
    node, cap = _cap(tmp_path, "fb1", raw_depth_intrinsics=FALLBACK)
    spinner(cam, node)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.source == "node" and fs.depth_info_source == "config_fallback"
    assert (fs.depth_info.width, fs.depth_info.height) == (RW, RH) and list(fs.depth_info.k) == K_RAW
    assert fs.depth_info.header.frame_id == RAW_FRAME
    files = cap.write_capture(fs, info)
    man = json.loads(files.manifest_path.read_text())
    reg = man["depth_registration"]
    assert reg["source"] == "node" and reg["intrinsics_source"] == "config_fallback"
    assert reg["depth_k"] == K_RAW and reg["depth_d"] == [0.0] * 5
    assert man["pointcloud"]["point_count"] == int((_expected_registration() > 0).sum())     # same cloud as live info
    d = cap.diagnostics()
    assert d["node_registration"]["depth_info_sizes"] == [f"{W}x{H}"]       # no live raw-size info was ever seen
    assert d["node_registration"]["fallback_intrinsics"] is True
    assert d["node_registration"]["fallback_size"] == f"{RW}x{RH}"
    assert d["node_registration"]["last_intrinsics_source"] == "config_fallback"


def test_live_raw_size_camera_info_wins_over_the_fallback(tmp_path, spinner):
    other = [float(RW), float(RH), 55.0, 56.0, 18.0, 13.0]          # differs from the live K_RAW
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "fb2", raw_depth_intrinsics=other)
    spinner(cam, node)
    deadline = time.monotonic() + 5.0
    while (RW, RH) not in cap._depth_infos and time.monotonic() < deadline:
        time.sleep(0.02)
    fs, info = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.depth_info_source == "camera_info" and list(fs.depth_info.k) == K_RAW
    reg = json.loads(cap.write_capture(fs, info).manifest_path.read_text())["depth_registration"]
    assert reg["intrinsics_source"] == "camera_info" and reg["depth_k"] == K_RAW
    nr = cap.diagnostics()["node_registration"]
    assert nr["fallback_intrinsics"] is True and nr["last_intrinsics_source"] == "camera_info"


def test_fallback_of_another_size_is_not_used_and_the_reason_is_reported(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_info=False)
    node, cap = _cap(tmp_path, "fb3", raw_depth_intrinsics=[RW * 2.0, RH * 2.0, 80.0, 80.0, 39.0, 29.0])
    spinner(cam, node)
    with pytest.raises(CaptureTimeout) as ei:
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.5)
    msg = str(ei.value)
    assert f"no depth camera_info of the raw depth size {RW}x{RH}" in msg
    assert f"raw_depth_intrinsics fallback is for {RW * 2}x{RH * 2}" in msg and "fallback size mismatch" in msg
    d = cap.diagnostics()
    assert d["counts"]["captures_node"] == 0 and d["node_registration"]["last_intrinsics_source"] == ""
    assert not list((tmp_path / "captures").glob("*"))


def test_no_fallback_fails_closed_as_before(tmp_path, spinner):
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH), raw_info=False)
    node, cap = _cap(tmp_path, "fb4")
    spinner(cam, node)
    with pytest.raises(CaptureTimeout) as ei:
        cap.capture(after_ns=cap.now_ns(), timeout_s=1.0)
    assert "fallback" not in str(ei.value).replace("config_fallback", "")
    nr = cap.diagnostics()["node_registration"]
    assert nr["fallback_intrinsics"] is False and nr["fallback_size"] == ""


@pytest.mark.parametrize("bad", [
    [1280.0, 800.0, 624.6, 624.6, 633.5],                       # 5 numbers
    [1280.0, 800.0, 624.6, 624.6, 633.5, 399.0, 1.0],           # 7 numbers
    [1280.0, 800.0, 624.6, 624.6, float("nan"), 399.0],
    [1280.0, 800.0, float("inf"), 624.6, 633.5, 399.0],
    [1280.5, 800.0, 624.6, 624.6, 633.5, 399.0],                # fractional width
    [0.0, 800.0, 624.6, 624.6, 633.5, 399.0],
    [1280.0, 800.0, 0.0, 624.6, 633.5, 399.0],                  # fx <= 0
    [1280.0, 800.0, 624.6, -1.0, 633.5, 399.0],                 # fy <= 0
    [1280.0, 800.0, 624.6, 624.6, 1280.0, 399.0],               # cx outside (0, width)
    [1280.0, 800.0, 624.6, 624.6, 633.5, 0.0],                  # cy outside (0, height)
])
def test_invalid_raw_depth_intrinsics_are_refused(tmp_path, bad):
    node = Node("fb5")
    with pytest.raises(ValueError, match="raw_depth_intrinsics"):
        SceneCapture(node, output_root=tmp_path, use_tf=False, raw_depth_intrinsics=bad)
    node.destroy_node()


def test_live_info_differing_from_the_fallback_warns_once(tmp_path, spinner):
    warnings = []
    cam = SyntheticCamera(depth_size=(W * 2, H * 2), raw_size=(RW, RH))
    node, cap = _cap(tmp_path, "fb6", raw_depth_intrinsics=[float(RW), float(RH), 44.0, 40.0, 19.5, 14.5])  # fx +10 %
    real = node.get_logger()
    node.get_logger = lambda: types.SimpleNamespace(warn=lambda m: warnings.append(m), info=real.info,
                                                    error=real.error, debug=real.debug)
    spinner(cam, node)
    deadline = time.monotonic() + 5.0
    while cap.diagnostics()["counts"]["depth_info"] < 6 and time.monotonic() < deadline:
        time.sleep(0.02)
    mism = [w for w in warnings if "raw_depth_intrinsics" in w]
    assert len(mism) == 1 and "differs" in mism[0] and "live camera_info is used" in mism[0]
    fs, _ = cap.grab(cap.now_ns(), timeout_s=5.0)
    assert fs.depth_info_source == "camera_info"                  # the live one is used regardless


@pytest.mark.parametrize("fb, warns", [
    ([40.15, 40.15, 19.5, 14.5], False),        # 0.4 % in fx/fy: within tolerance
    ([40.0, 40.0, 21.4, 14.5], False),          # 1.9 px in cx
    ([40.0, 40.0, 19.5, 16.6], True),           # 2.1 px in cy
    ([40.0, 40.3, 19.5, 14.5], True),           # 0.75 % in fy
])
def test_fallback_mismatch_tolerances(tmp_path, fb, warns):
    node, cap = _cap(tmp_path, "fb7", raw_depth_intrinsics=[float(RW), float(RH), *fb])
    from sensor_msgs.msg import CameraInfo
    got = []
    cap._log_once = lambda key, text: got.append(key)
    cap._on_depth_info(CameraInfo(width=RW, height=RH, k=K_RAW))
    assert (got == ["fallback_mismatch"]) is warns
    cap._on_depth_info(CameraInfo(width=W, height=H, k=K))        # another size never compares
    assert len(got) == int(warns)
    node.destroy_node()
