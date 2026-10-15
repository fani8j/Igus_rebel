"""Static checks of launch/dual_cut_system.launch.py (no processes are started)."""
import importlib.util
import os

import pytest
from launch.actions import DeclareLaunchArgument

_PATH = os.path.join(os.path.dirname(__file__), "..", "launch", "dual_cut_system.launch.py")
_spec = importlib.util.spec_from_file_location("dual_cut_system_launch", _PATH)
system = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(system)


def declared():
    return {a.name: a for a in system.generate_launch_description().entities
            if isinstance(a, DeclareLaunchArgument)}


def test_hardware_protocol_has_no_default():
    argument = declared()["hardware_protocol"]
    assert argument.default_value is None
    assert set(argument.choices) == {"mock_hardware", "rebel"}


def test_safe_defaults():
    args = declared()
    assert args["activate_controllers"].default_value[0].text == "auto"
    assert args["use_rviz"].default_value[0].text == "false"
    assert "execution_enabled" not in args    # enables are never launch arguments


def write(tmp_path, body):
    path = tmp_path / "coordinator.yaml"
    path.write_text(body)
    return str(path)


def test_coordinator_config_with_enabled_execution_is_refused(tmp_path):
    for name in system.ENABLE_PARAMS:
        path = write(tmp_path, f"dual_cut_coordinator:\n  ros__parameters:\n    {name}: true\n")
        with pytest.raises(RuntimeError, match=name):
            system.check_coordinator_config(path)


def test_coordinator_config_with_disabled_execution_is_accepted(tmp_path):
    system.check_coordinator_config(write(
        tmp_path, "dual_cut_coordinator:\n  ros__parameters:\n    execution_enabled: false\n    swept_padding_m: 0.05\n"))


def test_checklist_for_rebel_has_the_identity_steps():
    text = "\n".join(system.operator_checklist("rebel", "192.168.3.11", "192.168.3.12", "auto"))
    assert "pendant" in text and "dual_arm_recover.py activate" in text and "INACTIVE" in text
    assert "192.168.3.11" in text and "192.168.3.12" in text
    text = "\n".join(system.operator_checklist("rebel", "a", "b", "true"))
    assert "NO identity check" in text


def test_checklist_for_mock_is_short_and_mentions_mock():
    lines = system.operator_checklist("mock_hardware", "", "", "auto")
    assert "MOCK" in "\n".join(lines) and len(lines) <= 6
