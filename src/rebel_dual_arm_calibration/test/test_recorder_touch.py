"""Recorder: table-board observations, ~/capture_touch and saving the URDF (rclpy, isolated DDS domain)."""
import json
import os
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from rebel_dual_arm_calibration import synthetic as S   # noqa: E402
from rebel_dual_arm_calibration.modes import mode_parameters   # noqa: E402
from rebel_dual_arm_calibration.targets import default_targets_path, load_targets_all   # noqa: E402
from test_modes import _feed, _images, _stamped   # noqa: E402
from test_targets import _scene_with_boards   # noqa: E402


@pytest.fixture()
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "177")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init()
    yield
    rclpy.shutdown()


def _node(tmp_path, mode, **extra):
    from rclpy.parameter import Parameter

    from rebel_dual_arm_calibration.recorder_node import CalibrationRecorder
    params = dict(mode_parameters(mode))
    params["output_dir"] = str(tmp_path / mode)
    params["stationary_window_s"] = 0.5     # default 2.0 s; see test_capture_quality
    params.update(extra)
    return CalibrationRecorder(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])


def _set(node, **kv):
    from rclpy.parameter import Parameter
    node.set_parameters([Parameter(k, value=v) for k, v in kv.items()])


def _call(fn):
    resp = type("R", (), {})()
    fn(None, resp)
    return resp


def _ready(node, names, tcp_xyz=(0.3, 0.1, 0.02)):
    _feed(node, names, 0.0)
    time.sleep(0.7)
    _feed(node, names, 0.0, n=3)
    _stamped(node, "base_link", "link6", (0.3, 0.1, 0.2))
    _stamped(node, "base_link", "xeg32_tool_tip", tcp_xyz)
    _stamped(node, "camera_link", "camera_color_optical_frame", (0.0, 0.0, 0.0), static=True)


def test_tcp_frames_are_derived_from_the_tip_frames(ros, tmp_path):
    n = _node(tmp_path, "dual")
    try:
        assert n.p["left_tcp_frame"] == "arm_left_xeg32_tool_tip" and n.p["right_tcp_frame"] == "arm_right_xeg32_tool_tip"
        assert n.table_spec is not None and sorted(n.table_spec.touch_points) == [1, 2, 3, 4]
    finally:
        n.destroy_node()
    n = _node(tmp_path, "single_right")
    try:
        assert n.p["right_tcp_frame"] == "xeg32_tool_tip"
        meta = json.load(open(os.path.join(n.session_dir, "session.json")))
        assert "table_board" in meta["targets"] and meta["parameters"]["right_tcp_frame"] == "xeg32_tool_tip"
    finally:
        n.destroy_node()


def test_capture_touch_records_tcp_and_joints_and_validates(ros, tmp_path):
    names = [f"joint{i}" for i in range(1, 7)]
    n = _node(tmp_path, "single_left")
    try:
        r = _call(n._srv_capture_touch)
        assert not r.success and "touch_point_id" in r.message           # point id not set yet
        _set(n, touch_point_id=9)
        assert "touch_point_id" in _call(n._srv_capture_touch).message   # not a board point
        _set(n, touch_point_id=2)
        r = _call(n._srv_capture_touch)
        assert not r.success and "joint_states" in r.message             # nothing received yet
        _ready(n, names)
        r = _call(n._srv_capture_touch)
        assert r.success, r.message
        assert "arm left, point 2" in r.message and "0.3000, 0.1000, 0.0200" in r.message
        rows = [json.loads(l) for l in open(os.path.join(n.session_dir, "touches.jsonl"))]
        assert len(rows) == 1 and rows[0]["arm"] == "left" and rows[0]["point_id"] == 2
        assert np.allclose(rows[0]["p_board"], [0.06, -0.04, 0.0]) and rows[0]["tcp_frame"] == "xeg32_tool_tip"
        assert np.allclose(np.array(rows[0]["T_base_tcp"])[:3, 3], [0.3, 0.1, 0.02])
        assert sorted(rows[0]["joints"]) == sorted(names)
        assert n.touch_counts["left"] == 1
        assert "touches left 1" in _call(n._srv_status).message
        # the touched point must be touched with the arm at rest
        _feed(n, names, 0.0, n=20, vel=0.5)
        r = _call(n._srv_capture_touch)
        assert not r.success and "stationary" in r.message
        assert len(open(os.path.join(n.session_dir, "touches.jsonl")).readlines()) == 1
    finally:
        n.destroy_node()


def test_capture_touch_stores_link6_pose_and_the_link6_tcp_transform(ros, tmp_path):
    """T_base_link6 (contemporaneous) and T_link6_tcp are stored with every touch, so a TCP change can be re-based."""
    from rebel_dual_arm_calibration.session import load_touches
    names = [f"joint{i}" for i in range(1, 7)]
    n = _node(tmp_path, "single_left")
    try:
        _set(n, touch_point_id=2)
        _ready(n, names)
        r = _call(n._srv_capture_touch)
        assert r.success, r.message
        row = json.loads(open(os.path.join(n.session_dir, "touches.jsonl")).readline())
        T6, X = np.array(row["T_base_link6"]), np.array(row["T_link6_tcp"])
        assert np.allclose(T6[:3, 3], [0.3, 0.1, 0.2]) and np.allclose(X[:3, 3], [0.0, 0.0, -0.18], atol=1e-9)
        assert np.allclose(T6 @ X, np.array(row["T_base_tcp"]), atol=1e-9)
        t = load_touches(n.session_dir)[0]
        assert np.allclose(t.T_link6_tcp, X) and np.allclose(t.T_base_link6, T6) and t.tcp_source == "recorded"
        # no TF base -> link6: the touch is refused (not stored without it)
        n.tf_buffer.clear()
        _set(n, touch_point_id=2)
        assert not _call(n._srv_capture_touch).success
        assert len(open(os.path.join(n.session_dir, "touches.jsonl")).readlines()) == 1
    finally:
        n.destroy_node()


def test_dual_mode_needs_touch_arm(ros, tmp_path):
    n = _node(tmp_path, "dual")
    try:
        _set(n, touch_point_id=1)
        r = _call(n._srv_capture_touch)
        assert not r.success and "touch_arm" in r.message
        _set(n, touch_arm="middle")
        assert "touch_arm" in _call(n._srv_capture_touch).message
    finally:
        n.destroy_node()


def test_capture_stores_table_board_corners_separately_from_arm_samples(ros, tmp_path):
    from sensor_msgs.msg import CameraInfo, Image
    arms, table = load_targets_all(default_targets_path())
    names = [f"joint{i}" for i in range(1, 7)]
    n = _node(tmp_path, "single_left")
    try:
        gray, _ = _scene_with_boards({"table": table}, px_per_m=4000.0, scale=0.3, noise=1.0)   # table board only
        _ready(n, names)
        info = CameraInfo()
        info.width, info.height = 1920, 1080
        info.k = [960.0, 0.0, 960.0, 0.0, 960.0, 540.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        n._on_info(info)
        _images(n, gray)
        r = _call(n._srv_capture)
        assert r.success, r.message
        assert "table board:" in r.message and "left: not detected" in r.message
        assert r.message.startswith("NO ARM SAMPLE RECORDED")        # success=True must not pass for a good capture
        assert not os.path.exists(os.path.join(n.session_dir, "samples.jsonl"))       # no arm sample
        rows = [json.loads(l) for l in open(os.path.join(n.session_dir, "table_obs.jsonl"))]
        assert len(rows) == 1 and len(rows[0]["ids"]) == 40 + 8       # 10 markers x 4 + 8 chess corners
        assert not {1000, 1003, 1008, 1011} & set(rows[0]["ids"])      # touch corners are not observations
        assert os.path.exists(os.path.join(n.session_dir, rows[0]["image"]))
        from rebel_dual_arm_calibration.session import load_table_obs
        assert len(load_table_obs(n.session_dir)) == 1 and n.n_table_obs == 1
    finally:
        n.destroy_node()


def test_urdf_saved_from_file_once(ros, tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(S.rebel_urdf_text(True) if S.rebel_urdf_path() else "<robot/>")
    if S.rebel_urdf_path() is None:
        pytest.skip("igus_rebel2.urdf not found")
    n = _node(tmp_path, "single_left", robot_description_file=str(urdf))
    try:
        saved = os.path.join(n.session_dir, "robot_description.urdf")
        assert n.urdf_saved and os.path.exists(saved)
        assert "robot_description.urdf: saved" in _call(n._srv_status).message
        before = open(saved).read()
    finally:
        n.destroy_node()
    # restarting in the same session directory keeps the first copy
    n = _node(tmp_path, "single_left", robot_description_file=str(urdf))
    try:
        assert n.urdf_saved and open(saved).read() == before
    finally:
        n.destroy_node()


def test_urdf_saved_from_robot_description_topic(ros, tmp_path):
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    if S.rebel_urdf_path() is None:
        pytest.skip("igus_rebel2.urdf not found")
    n = _node(tmp_path, "single_left")
    try:
        assert not n.urdf_saved
        pub = n.create_publisher(String, "robot_description",
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                            reliability=ReliabilityPolicy.RELIABLE))
        pub.publish(String(data=S.rebel_urdf_text(True)))
        t0 = time.time()
        while not n.urdf_saved and time.time() - t0 < 5:
            rclpy.spin_once(n, timeout_sec=0.1)
        assert n.urdf_saved and os.path.exists(os.path.join(n.session_dir, "robot_description.urdf"))
        n._save_urdf("<broken", "test")       # a second / invalid description is ignored
    finally:
        n.destroy_node()
