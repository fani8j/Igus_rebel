"""Recorder modes (single-arm stacks) and the recorder's handling of a missing second arm."""
import os

import pytest

from rebel_dual_arm_calibration.modes import MODES, active_arms_of, mode_parameters


def test_mode_parameters():
    d = mode_parameters("dual")
    assert d["active_arms"] == ["left", "right"]
    assert (d["left_base_frame"], d["left_tip_frame"]) == ("arm_left_base_link", "arm_left_link6")
    assert (d["right_base_frame"], d["right_tip_frame"]) == ("arm_right_base_link", "arm_right_link6")
    assert (d["left_joint_prefix"], d["right_joint_prefix"]) == ("arm_left_", "arm_right_")
    for arm in ("left", "right"):
        p = mode_parameters(f"single_{arm}")
        assert p == {"active_arms": [arm], f"{arm}_joint_prefix": "",
                     f"{arm}_base_frame": "base_link", f"{arm}_tip_frame": "link6"}
    assert set(MODES) == {"dual", "single_left", "single_right"}
    with pytest.raises(ValueError):
        mode_parameters("triple")
    with pytest.raises(ValueError):
        active_arms_of([])
    with pytest.raises(ValueError):
        active_arms_of(["middle"])


# --- the recorder node itself (needs rclpy; isolated DDS domain, never the robot's) -----------
rclpy = pytest.importorskip("rclpy")


@pytest.fixture()
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "174")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init()
    yield
    rclpy.shutdown()


def _node(tmp_path, mode):
    from rclpy.parameter import Parameter

    from rebel_dual_arm_calibration.recorder_node import CalibrationRecorder
    params = dict(mode_parameters(mode))
    params["output_dir"] = str(tmp_path / mode)
    params["stationary_window_s"] = 0.5     # the default is 2.0 s (test_capture_quality); these tests wait wall time
    return CalibrationRecorder(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])


def _feed(node, names, t0, n=40, dt=0.02, vel=0.0):
    from sensor_msgs.msg import JointState
    for k in range(n):
        m = JointState()
        m.name = list(names)
        m.position = [0.1 * (i + 1) + (vel * k * dt if i == 0 else 0.0) for i in range(len(names))]
        m.velocity = [vel if i == 0 else 0.0 for i in range(len(names))]
        # use header stamp 0 -> node uses its own clock for the sample time
        node._on_joints(m)


@pytest.mark.parametrize("mode", ["single_left", "single_right"])
def test_single_arm_recorder_ignores_missing_other_arm(ros, tmp_path, mode):
    import time
    names = [f"joint{i}" for i in range(1, 7)]
    node = _node(tmp_path, mode)
    try:
        arm = mode.split("_")[1]
        assert node.active_arms == [arm]
        other = "right" if arm == "left" else "left"
        # joint_states of the single-arm stack: unprefixed names, only one arm
        _feed(node, names, 0.0)
        time.sleep(0.7)  # stationary window is 0.5 s of wall time
        _feed(node, names, 0.0, n=3)
        since, why = node.stationary_since()
        assert since is not None, why
        # status must only ask for the active arm
        resp = type("R", (), {})()
        node._srv_status(None, resp)
        assert f"TF {arm}:" in resp.message and f"TF {other}:" not in resp.message
        assert f"active arms {arm}" in resp.message
    finally:
        node.destroy_node()


def test_dual_recorder_still_requires_both_prefixes(ros, tmp_path):
    import time
    node = _node(tmp_path, "dual")
    try:
        assert node.active_arms == ["left", "right"]
        left_only = [f"arm_left_joint{i}" for i in range(1, 7)]
        _feed(node, left_only, 0.0)
        since, why = node.stationary_since()
        assert since is None and "arm_right_" in why
        both = left_only + [f"arm_right_joint{i}" for i in range(1, 7)]
        time.sleep(0.7)
        _feed(node, both, 0.0, n=3)
        # history still holds the left-only messages (right joints absent there), which must not count as motion
        since, why = node.stationary_since()
        assert since is not None, why
    finally:
        node.destroy_node()


def test_single_mode_detects_motion(ros, tmp_path):
    import time
    node = _node(tmp_path, "single_left")
    try:
        names = [f"joint{i}" for i in range(1, 7)]
        _feed(node, names, 0.0, n=20, vel=0.5)
        time.sleep(0.05)
        since, why = node.stationary_since()
        assert since is None
    finally:
        node.destroy_node()


def _images(node, gray, second=None, gap=0.5, encoding="mono8"):
    """Deliver two images ``gap`` s apart (stamps now-gap and now); ``second`` defaults to the same picture."""
    from rclpy.duration import Duration
    from sensor_msgs.msg import Image
    now = node.get_clock().now()
    node.image_hist.clear()                  # a scripted image sequence replaces whatever came before
    for g, back in ((gray, gap), (gray if second is None else second, 0.0)):
        img = Image()
        img.header.stamp = (now - Duration(seconds=back)).to_msg()
        img.width, img.height, img.encoding, img.step = 1920, 1080, encoding, 1920
        img.data = g.tobytes()
        node._on_image(img)


def _stamped(node, parent, child, xyz, static=False):
    from geometry_msgs.msg import TransformStamped
    m = TransformStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id, m.child_frame_id = parent, child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = xyz
    m.transform.rotation.w = 1.0
    node.tf_buffer.set_transform_static(m, "test") if static else node.tf_buffer.set_transform(m, "test")


@pytest.mark.parametrize("mode", ["single_left", "single_right"])
def test_single_arm_capture_end_to_end(ros, tmp_path, mode):
    """Full capture with only this arm's TF/joints present; the other arm's board is visible but ignored."""
    import json
    import time

    from sensor_msgs.msg import CameraInfo, Image

    from rebel_dual_arm_calibration.targets import default_targets_path, load_targets
    from test_targets import _scene_with_boards
    arm = mode.split("_")[1]
    other = "right" if arm == "left" else "left"
    node = _node(tmp_path, mode)
    try:
        specs = load_targets(default_targets_path())
        gray, _ = _scene_with_boards(specs, px_per_m=4000.0, scale=0.3, noise=1.0)   # both boards in view
        names = [f"joint{i}" for i in range(1, 7)]
        _feed(node, names, 0.0)
        time.sleep(0.7)
        _feed(node, names, 0.0, n=3)
        # TF: ONLY base_link->link6 of this arm and camera_link->optical (no arm_*_ frames exist)
        _stamped(node, "base_link", "link6", (0.3, 0.1, 0.2))
        _stamped(node, "camera_link", "camera_color_optical_frame", (0.0, 0.0, 0.0), static=True)
        info = CameraInfo()
        info.width, info.height = 1920, 1080
        info.k = [960.0, 0.0, 960.0, 0.0, 960.0, 540.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        node._on_info(info)
        _images(node, gray)                      # two images 0.5 s apart (the two-frame stillness check)
        resp = type("R", (), {})()
        node._srv_capture(None, resp)
        assert resp.success, resp.message
        assert f"{arm}:" in resp.message and f"{other}:" not in resp.message, resp.message
        rows = [json.loads(line) for line in open(os.path.join(node.session_dir, "samples.jsonl"))]
        assert [r["arm"] for r in rows] == [arm]
        assert all(n.startswith("joint") for n in rows[0]["joints"]) and len(rows[0]["joints"]) == 6
        assert node.counts[arm] == 1 and node.counts[other] == 0
    finally:
        node.destroy_node()
