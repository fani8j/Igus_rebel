"""Static checks of the automatic cut recorder in launch/move_group.launch.py (single arm; no process is started).

Run (sourced ROS + workspace): python3 -m pytest -p no:anyio src/igus_rebel_moveit_config/test/test_move_group_cut_recorder_launch.py
"""
import importlib.util
import os
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, LogInfo

_PATH = os.path.join(os.path.dirname(__file__), "..", "launch", "move_group.launch.py")
_spec = importlib.util.spec_from_file_location("move_group_launch", _PATH)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

FAKE_SCRIPT = "/fake/robot_remote/recording/record_raw_cut.py"


def declared():
    return {a.name: a for a in mg.generate_launch_description().entities if isinstance(a, DeclareLaunchArgument)}


def ctx(**configs):
    context = LaunchContext()
    defaults = {"record_cut_video": "true", "cut_recorder_backend": "raw", "robot_remote_dir": "~/robot_remote",
                "cut_video_cameras": "", "cut_video_dir": "",
                "run_box_h_cut_executor": "true", "hardware_protocol": "rebel"}
    defaults.update(configs)
    context.launch_configurations.update(defaults)
    return context


def cmd_of(action):
    return " ".join("".join(x.text for x in c) if isinstance(c, list) else getattr(c, "text", str(c)) for c in action.cmd)


@pytest.fixture
def raw_ok(monkeypatch):
    monkeypatch.setattr(mg, "raw_ready", lambda d: (Path(FAKE_SCRIPT), ""))


def test_arguments():
    args = declared()
    assert args["cut_recorder_backend"].default_value[0].text == "raw"
    assert set(args["cut_recorder_backend"].choices) == {"raw", "video"}
    assert set(args["record_cut_video"].choices) == {"true", "false", "always"}
    assert args["record_cut_video"].default_value[0].text == "true"
    assert args["robot_remote_dir"].default_value[0].text == "~/robot_remote"


def test_only_with_the_hcut_executor_on_rebel_unless_always(raw_ok):
    assert mg.cut_recorder_actions(ctx(hardware_protocol="mock_hardware")) == []
    assert mg.cut_recorder_actions(ctx(run_box_h_cut_executor="false")) == []
    assert mg.cut_recorder_actions(ctx(record_cut_video="false")) == []
    assert len(mg.cut_recorder_actions(ctx())) == 1
    assert len(mg.cut_recorder_actions(ctx(record_cut_video="always", hardware_protocol="mock_hardware",
                                           run_box_h_cut_executor="false"))) == 1


def test_raw_command_uses_the_fjt_status_trigger(raw_ok):
    (action,) = mg.cut_recorder_actions(ctx(cut_video_dir="/clips"))
    cmd = cmd_of(action)
    assert cmd.startswith("nice -n 10 python3 " + FAKE_SCRIPT)
    for part in ("--trigger fjt-status", "--fjt-hold-s 5.0", "--camera-keys " + mg.RAW_CAMERAS, "--clip-dir /clips"):
        assert part in cmd, (part, cmd)
    for absent in ("--camera-domain", "dual", "lerobot", "--source", "--root"):
        assert absent not in cmd
    assert action._ExecuteLocal__sigterm_timeout[0].text == "60"


def test_old_camera_names_are_mapped(raw_ok):
    (action,) = mg.cut_recorder_actions(ctx(cut_video_cameras="knife,scene,wrist,right_wrist"))
    assert "--camera-keys knife,head,wrist_left_knife,wrist_right_knife" in cmd_of(action)


def test_video_backend_is_the_old_recorder_unchanged():
    (action,) = mg.cut_recorder_actions(ctx(cut_recorder_backend="video"))
    cmd = cmd_of(action)
    assert "tools/cut_recorder/cut_recorder.py" in cmd
    assert "--cameras knife,scene,wrist,right_wrist" in cmd
    assert "--coordinator-status-topic" not in cmd            # trigger = the FJT action status (its default)
    (action,) = mg.cut_recorder_actions(ctx(cut_recorder_backend="video", cut_video_cameras="head,wrist_right_gripper"))
    assert "--cameras scene,right_gripper" in cmd_of(action)


def test_falls_back_to_video_and_never_fails(tmp_path, monkeypatch):
    actions = mg.cut_recorder_actions(ctx(robot_remote_dir=str(tmp_path / "missing")))
    assert isinstance(actions[0], LogInfo) and "falling back" in actions[0].msg[0].text
    assert "cut_recorder.py" in cmd_of(actions[1])
    (tmp_path / "recording").mkdir()
    (tmp_path / "recording" / "record_raw_cut.py").write_text("")

    class Failed:
        returncode = 1
    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: Failed())
    actions = mg.cut_recorder_actions(ctx(robot_remote_dir=str(tmp_path)))
    assert isinstance(actions[0], LogInfo) and "rosbag2_py" in actions[0].msg[0].text
    assert "cut_recorder.py" in cmd_of(actions[1])
