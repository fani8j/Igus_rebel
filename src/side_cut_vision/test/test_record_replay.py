import json
import os

import cv2
import numpy as np
import pytest

import synth
from side_cut_vision import replay
from side_cut_vision.calib import CALIB_SEED
from side_cut_vision.measure import measure
from side_cut_vision.overlay import draw_overlay
from side_cut_vision.record import RecordError, load_record, look_dir_name, save_record


@pytest.fixture(scope="module")
def look(tmp_path_factory):
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T, tip, _ = synth.approach_pose(model)
    color, depth, _ = synth.render(cam, T, model, synth.SceneSpec())
    m = measure(color, depth, cam, T, model, tip_world=tip, meta={"calib_status": CALIB_SEED})
    ov = draw_overlay(color, cam, T, model, m, tip, state="LOOK saved", calib_status=CALIB_SEED,
                      snapshot_id="capture_test", ros_time_s=123.456)
    side = tmp_path_factory.mktemp("clip") / "side_cut_vision"
    name = look_dir_name(str(side), "left", 1792470300.0)
    path = save_record(str(side / name), "left", color, depth, cam, T, np.block(
        [[np.eye(3), tip[:, None]], [np.zeros((1, 3)), 1]]), {"name": ["j1"], "position": [0.0]},
        synth.SNAPSHOT, synth.HANDEYE, m, ov)
    return path, m


def test_record_layout_and_names(look):
    path, m = look
    for f in ("color.png", "depth.png", "camera_info.json", "T_world_optical.json", "T_world_tip.json",
              "joint_state.json", "snapshot.json", "calib.yaml", "measurement.json", "overlay.png"):
        assert os.path.isfile(os.path.join(path, f)), f
    assert os.path.basename(path).startswith("left_01_")
    side = os.path.dirname(path)
    os.makedirs(os.path.join(side, "left_02_000000"))
    assert look_dir_name(side, "left").startswith("left_03_")
    assert look_dir_name(side, "right").startswith("right_01_")
    d = cv2.imread(os.path.join(path, "depth.png"), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16
    with open(os.path.join(path, "measurement.json")) as f:
        json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))   # strict JSON


def test_replay_reproduces_the_saved_measurement(look, tmp_path):
    path, m = look
    d = replay.replay_record(path, out_dir=str(tmp_path))
    assert d["ok"] and d["seam_ok"]
    for k, v in m.to_dict()["delta_mm"].items():
        assert abs(d["delta_mm"][k] - v) < 0.05
    assert os.path.isfile(tmp_path / "replay_overlay.png")
    ov = cv2.imread(str(tmp_path / "replay_overlay.png"))
    assert ov.shape == (600, 960, 3)


def test_replay_cli_record(look, tmp_path, capsys):
    path, _ = look
    rc = replay.main(["record", path, "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ok=True" in out and "delta(a,c,n)" in out
    assert (tmp_path / "out" / os.path.basename(path) / "replay_measurement.json").is_file()


def test_replay_cli_reports_errors_without_crashing(tmp_path, capsys):
    rc = replay.main(["record", str(tmp_path / "nothing_here")])
    assert rc == 1
    assert "ERROR" in capsys.readouterr().err
    with pytest.raises(RecordError):
        load_record(str(tmp_path))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "color.png").write_bytes(b"not a png")
    with pytest.raises(RecordError):
        load_record(str(bad))


def test_overlay_banners_and_failure_modes(look):
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T, tip, _ = synth.approach_pose(model)
    color, depth, _ = synth.render(cam, T, model, synth.SceneSpec())
    # no depth -> banner, still an image of the right size
    m = measure(color, np.zeros_like(depth), cam, T, model, tip_world=tip)
    ov = draw_overlay(color, cam, T, model, m, tip, calib_status=CALIB_SEED)
    assert ov.shape == (600, 960, 3) and ov.dtype == np.uint8
    # model only, no measurement, non-finite pose: must not raise
    draw_overlay(color, cam, np.full((4, 4), np.nan), model, None, [np.nan] * 3)
    draw_overlay(color, cam, T, model, None, None, state="LIVE")


def test_library_has_no_ros_imports():
    import re
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "side_cut_vision")
    pat = re.compile(r"^\s*(import|from)\s+(rclpy|tf2_ros|sensor_msgs|std_msgs|dual_cut_msgs)", re.M)
    for f in os.listdir(root):
        if f.endswith(".py") and f not in ("probe_node.py", "mock_camera.py"):   # the only ROS-facing modules
            assert not pat.search(open(os.path.join(root, f)).read()), f
