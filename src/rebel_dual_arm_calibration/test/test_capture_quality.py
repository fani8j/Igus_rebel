"""Capture quality: 2 s stationary window, two-frame corner check, free stickers, capture_intrinsics, session info."""
import json
import os
import time

import cv2
import numpy as np
import pytest

from rebel_dual_arm_calibration import intrinsics as I

rclpy = pytest.importorskip("rclpy")

from rebel_dual_arm_calibration.modes import mode_parameters   # noqa: E402
from rebel_dual_arm_calibration.targets import default_targets_path, load_targets_all   # noqa: E402
from test_modes import _feed, _images, _stamped   # noqa: E402
from test_recorder_touch import _call, _node, _ready, _set   # noqa: E402,F401
from test_targets import _scene_with_boards   # noqa: E402

NAMES = [f"joint{i}" for i in range(1, 7)]


@pytest.fixture()
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "178")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init()
    yield
    rclpy.shutdown()


# ---- recorder ---------------------------------------------------------------------------------------------------
def test_default_stationary_window_is_two_seconds(ros, tmp_path):
    from rclpy.parameter import Parameter

    from rebel_dual_arm_calibration.recorder_node import CalibrationRecorder
    params = dict(mode_parameters("single_left"))
    params["output_dir"] = str(tmp_path / "d")
    n = CalibrationRecorder(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    try:
        assert n.p["stationary_window_s"] == 2.0 and n.p["max_corner_motion_px"] == 0.2 and n.p["min_image_gap_s"] == 0.4
        assert n.p["max_corner_single_px"] == 1.0
        _feed(n, NAMES, 0.0)
        time.sleep(0.7)
        _feed(n, NAMES, 0.0, n=3)
        since, why = n.stationary_since()
        assert since is None and ("2.00 s" in why or "history" in why)          # 0.7 s of stillness is not enough any more
    finally:
        n.destroy_node()


def _fresh(n):
    """Joint states and TF are only valid for 0.5 / 1 s: refresh them after the (slow) image detections."""
    _feed(n, NAMES, 0.0, n=3)
    _stamped(n, "base_link", "link6", (0.3, 0.1, 0.2))
    _stamped(n, "base_link", "xeg32_tool_tip", (0.3, 0.1, 0.02))


def _table_scene(angle=0.0):
    _, table = load_targets_all(default_targets_path())
    gray, _ = _scene_with_boards({"table": table}, px_per_m=4000.0, scale=0.3, angle_deg=angle, noise=1.0)
    return gray


def _shifted(gray, dx):
    return cv2.warpAffine(gray, np.float32([[1, 0, dx], [0, 1, 0]]), (gray.shape[1], gray.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _info(n):
    from sensor_msgs.msg import CameraInfo
    info = CameraInfo()
    info.width, info.height = 1920, 1080
    info.k = [960.0, 0.0, 960.0, 0.0, 960.0, 540.0, 0.0, 0.0, 1.0]
    info.d = [0.0] * 5
    n._on_info(info)


def test_two_frame_check_rejects_a_moving_board_and_needs_an_earlier_frame(ros, tmp_path):
    n = _node(tmp_path, "single_left")
    try:
        _ready(n, NAMES)
        _info(n)
        gray = _table_scene()
        # (1) one image only: nothing to compare with
        _images(n, gray)
        n.image_hist.clear()
        n.image_hist.append((n._image_stamp(n.image), n.image))
        _fresh(n)
        r = _call(n._srv_capture)
        assert not r.success and "no earlier image" in r.message
        # (2) a board that moved 0.6 px between the two frames is rejected, not recorded
        _images(n, gray, second=_shifted(gray, 0.6))
        _fresh(n)
        r = _call(n._srv_capture)
        assert not r.success and "board still moving (common" in r.message and "single max" in r.message, r.message
        assert not os.path.exists(os.path.join(n.session_dir, "table_obs.jsonl"))
        # (3) the same board in both frames: recorded
        _images(n, gray)
        _fresh(n)
        r = _call(n._srv_capture)
        assert r.success and "table board:" in r.message and "image #1" in r.message, r.message
        # (4) two frames only 0.2 s apart are not enough
        n.image_hist.clear()
        _images(n, gray, gap=0.2)
        _fresh(n)
        r = _call(n._srv_capture)
        assert not r.success and "no earlier image" in r.message
        # (5) a frame from before the stationary start cannot be used
        n.image_hist.clear()
        _images(n, gray, gap=3.0)
        _fresh(n)
        r = _call(n._srv_capture)
        assert not r.success and "no earlier image" in r.message
    finally:
        n.destroy_node()
    # (6) the check can be switched off (parameter corner_check:=false), and the status says so
    n = _node(tmp_path / "off", "single_left", corner_check=False)
    try:
        _ready(n, NAMES)
        _info(n)
        _images(n, gray, second=_shifted(gray, 0.6))
        _fresh(n)
        assert "still moving" not in _call(n._srv_capture).message
        assert "two-frame corner check OFF" in _call(n._srv_status).message
    finally:
        n.destroy_node()


def test_touch_checks_the_table_board_when_visible_and_skips_otherwise(ros, tmp_path):
    n = _node(tmp_path, "single_left")
    try:
        _ready(n, NAMES)
        _info(n)
        _set(n, touch_point_id=1)
        gray = _table_scene()
        _images(n, gray, second=_shifted(gray, 0.7))
        _fresh(n)
        r = _call(n._srv_capture_touch)
        assert not r.success and "table board in the image" in r.message and "still moving" in r.message, r.message
        assert not os.path.exists(os.path.join(n.session_dir, "touches.jsonl"))
        _images(n, gray)
        _fresh(n)
        r = _call(n._srv_capture_touch)
        assert r.success and "image check: still (common" in r.message, r.message
        _images(n, np.full_like(gray, 190))                    # no board in view: the touch itself needs no camera
        _fresh(n)
        r = _call(n._srv_capture_touch)
        assert r.success and "not visible" in r.message, r.message
    finally:
        n.destroy_node()


def test_touch_skips_the_image_check_for_a_partly_hidden_board(ros, tmp_path):
    """The arm in front of the table board: the corner detection of a partly hidden board jumps by about a pixel, which
    is not board motion.  Fewer than 90 % of the corners -> the image check is skipped (and says so)."""
    n = _node(tmp_path, "single_left")
    try:
        _ready(n, NAMES)
        _info(n)
        _set(n, touch_point_id=1)
        gray = _table_scene()
        hidden = gray.copy()
        hidden[:, 650:] = 190                                   # right part of the board hidden
        from rebel_dual_arm_calibration.targets import TargetDetector
        _, table = load_targets_all(default_targets_path())
        n_vis = len(TargetDetector({"table": table}).detect(hidden)["table"].points)
        assert 16 <= n_vis < 0.9 * 48, n_vis
        _images(n, hidden, second=_shifted(hidden, 0.7))
        _fresh(n)
        r = _call(n._srv_capture_touch)
        assert r.success and "partly hidden" in r.message and "skipped" in r.message, r.message
    finally:
        n.destroy_node()


def test_free_stickers_are_recorded_with_their_kind(ros, tmp_path):
    n = _node(tmp_path, "single_left")
    try:
        _ready(n, NAMES)
        assert n.stickers.fit_points == [101, 102, 103, 104] and n.stickers.check_points == [111, 112]
        for pid, kind in ((111, "check"), (101, "fit"), (2, "board")):
            _set(n, touch_point_id=pid)
            _fresh(n)
            r = _call(n._srv_capture_touch)
            assert r.success, r.message
            assert (kind + " sticker" in r.message) or kind == "board"
        rows = [json.loads(l) for l in open(os.path.join(n.session_dir, "touches.jsonl"))]
        assert [(r["point_id"], r["kind"]) for r in rows] == [(111, "check"), (101, "fit"), (2, "board")]
        assert rows[0]["p_board"] == [0.0, 0.0, 0.0] and np.allclose(rows[2]["p_board"], [0.06, -0.04, 0.0])
        _set(n, touch_point_id=7)
        _fresh(n)
        r = _call(n._srv_capture_touch)
        assert not r.success and "sticker ids" in r.message and "111" in r.message
        meta = json.load(open(os.path.join(n.session_dir, "session.json")))
        assert meta["targets"]["touch_stickers"] == {"fit_points": [101, 102, 103, 104], "check_points": [111, 112]}
        from rebel_dual_arm_calibration.session import load_touches
        assert [t.kind for t in load_touches(n.session_dir)] == ["check", "fit", "board"]
    finally:
        n.destroy_node()


def test_capture_intrinsics_saves_sharp_distinct_views_only(ros, tmp_path):
    n = _node(tmp_path, "single_left")
    try:
        _info(n)
        assert "intrinsics views 0" in _call(n._srv_status).message
        a = _table_scene(0.0)
        r = _call(n._srv_capture_intrinsics)
        assert not r.success and "no image" in r.message
        _images(n, np.full_like(a, 190))
        r = _call(n._srv_capture_intrinsics)
        assert not r.success and "not detected" in r.message
        _images(n, a, second=_shifted(a, 0.6))                        # blurred / moving
        r = _call(n._srv_capture_intrinsics)
        assert not r.success and "still moving" in r.message and "blurred" in r.message
        _images(n, a)
        r = _call(n._srv_capture_intrinsics)
        assert r.success and "intrinsics view #1" in r.message, r.message
        _images(n, a)                                                  # the same view again
        r = _call(n._srv_capture_intrinsics)
        assert not r.success and "same view" in r.message
        _images(n, _table_scene(20.0))                                 # board moved: a new view
        r = _call(n._srv_capture_intrinsics)
        assert r.success and "view #2" in r.message, r.message
        rows = [json.loads(l) for l in open(os.path.join(n.session_dir, "intrinsics.jsonl"))]
        assert len(rows) == 2 and rows[0]["target"] == "table" and rows[0]["image_size"] == [1920, 1080]
        assert np.allclose(np.array(rows[0]["K"])[0, 0], 960.0) and rows[0]["board_distance_m"] > 0
        assert os.path.exists(os.path.join(n.session_dir, rows[0]["image"])) and "intrinsics views 2" in _call(n._srv_status).message
        assert len(rows[0]["ids"]) == len(rows[0]["pix"]) == len(rows[0]["obj"]) >= 16
    finally:
        n.destroy_node()


def test_intrinsics_file_replaces_camera_info_in_recorded_samples(ros, tmp_path):
    p = str(tmp_path / "intr.yaml")
    K = np.array([[1010.0, 0, 955.0], [0, 1008.0, 545.0], [0, 0, 1.0]])
    I.write_intrinsics_file(p, K, np.array([-0.05, 0.02, 0.0, 0.0, 0.0]), (1920, 1080))
    n = _node(tmp_path, "single_left", intrinsics_file=p)
    try:
        _ready(n, NAMES)
        _info(n)
        gray = _table_scene()
        _images(n, gray)
        _fresh(n)
        r = _call(n._srv_capture)
        assert r.success, r.message
        row = json.loads(open(os.path.join(n.session_dir, "table_obs.jsonl")).readline())
        assert np.allclose(np.array(row["K"])[0, 0], 1010.0) and row["dist"][0] == pytest.approx(-0.05)
        assert "intrinsics file overrides camera_info" in _call(n._srv_status).message
    finally:
        n.destroy_node()


def test_session_info_is_stored_and_shown(ros, tmp_path):
    n = _node(tmp_path, "single_left", ambient_temp_c=21.5, ambient_rh_pct=44.0, warmup_min=35.0, operator_note="cold room")
    try:
        meta = json.load(open(os.path.join(n.session_dir, "session.json")))
        assert meta["session_info"] == {"ambient_temp_c": 21.5, "ambient_rh_pct": 44.0, "warmup_min": 35.0,
                                        "operator_note": "cold room"}
        st = _call(n._srv_status).message
        assert "ambient_temp_c 21.5" in st and "warmup_min 35.0" in st and "operator_note cold room" in st
        _set(n, ambient_temp_c=23.0)
        _call(n._srv_status)
        assert json.load(open(os.path.join(n.session_dir, "session.json")))["session_info"]["ambient_temp_c"] == 23.0
    finally:
        n.destroy_node()
    # not set: absent values are None, the status says how to set them
    n = _node(tmp_path / "other", "single_left")
    try:
        meta = json.load(open(os.path.join(n.session_dir, "session.json")))
        assert meta["session_info"]["ambient_temp_c"] is None
        assert "set ambient_temp_c" in _call(n._srv_status).message
    finally:
        n.destroy_node()
