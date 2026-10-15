"""Recorder ~/status_json: the machine-readable status used by the calibration UI (rclpy, isolated DDS domain)."""
import json
import os
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from test_recorder_touch import _call, _node, _ready, _set   # noqa: E402


@pytest.fixture()
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "178")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init()
    yield
    rclpy.shutdown()


def test_status_json_content_and_touch_ids(ros, tmp_path):
    names = [f"joint{i}" for i in range(1, 7)]
    n = _node(tmp_path, "single_left")
    try:
        r = _call(n._srv_status_json)
        assert r.success
        st = json.loads(r.message)                       # strict JSON: no NaN
        assert st["session_dir"] == n.session_dir and st["active_arms"] == ["left"]
        assert st["counts"] == {"left": 0, "right": 0} and st["capture_count"] == 0 and st["n_table_obs"] == 0
        assert st["touched_ids"] == {"left": [], "right": []} and st["n_intrinsics"] == 0
        assert st["intrinsics_centres"] == [] and st["intrinsics_bboxes"] == []
        assert st["image"] == {"ok": False, "age_s": None, "size": None} and st["camera_info"] is False
        assert st["stationary"]["ok"] is False and st["tf"]["left"]["ok"] is False
        assert st["session_info"] == {"ambient_temp_c": None, "ambient_rh_pct": None, "warmup_min": None,
                                      "operator_note": ""}
        assert [t["id"] for t in st["touch_list"] if t["kind"] == "board"] == [1, 2, 3, 4]
        assert st["min_corners"]["left"] == 8 and st["min_corners"]["table"] == 16
        _set(n, touch_point_id=2, ambient_temp_c=21.5)
        _ready(n, names)
        assert _call(n._srv_capture_touch).success
        assert _call(n._srv_capture_touch).success
        st = json.loads(_call(n._srv_status_json).message)
        assert st["touched_ids"]["left"] == [2] and st["touch_point_counts"]["left"] == {"2": 2}
        assert st["touch_counts"]["left"] == 2 and st["stationary"]["ok"] is True and st["tf"]["left"]["ok"] is True
        assert st["session_info"]["ambient_temp_c"] == 21.5
        # intrinsics coverage entries come from the stored views and survive a recorder restart
        from rebel_dual_arm_calibration.recorder_node import _bbox_px, _centre_px
        pix = np.array([[100.0, 50.0], [300.0, 50.0], [300.0, 250.0], [100.0, 250.0]])
        n.intrinsics_views.append((_centre_px(pix), _bbox_px(pix)))
        st = json.loads(_call(n._srv_status_json).message)
        assert st["intrinsics_centres"] == [[200.0, 150.0]] and st["intrinsics_bboxes"] == [[100.0, 50.0, 300.0, 250.0]]
        session_dir = n.session_dir
    finally:
        n.destroy_node()
    # restart on the same session dir: touches are read back
    n = _node(tmp_path, "single_left")
    try:
        assert n.session_dir == session_dir
        assert json.loads(_call(n._srv_status_json).message)["touched_ids"]["left"] == [2]
    finally:
        n.destroy_node()
