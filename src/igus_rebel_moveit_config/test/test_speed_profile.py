"""Speed profile of launch/move_group.launch.py (single arm) and config/speed_profile_fast.yaml. No process is started.

Run: PYTEST_ADDOPTS="-p no:anyio" python3 -m pytest src/igus_rebel_moveit_config/test/test_speed_profile.py
"""
import importlib.util
import os

import pytest
import yaml
from launch.actions import DeclareLaunchArgument

HERE = os.path.dirname(__file__)
CONFIG = os.path.join(HERE, "..", "config")
BASE = os.path.join(CONFIG, "box_h_cut_executor.yaml")
PROFILE = os.path.join(CONFIG, "speed_profile_fast.yaml")
_spec = importlib.util.spec_from_file_location("move_group_launch_sp", os.path.join(HERE, "..", "launch", "move_group.launch.py"))
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

DEFAULTS = {"velocity_scaling": 0.1376, "acceleration_scaling": 0.0763,
            "transit_velocity_scaling": 0.20, "transit_acceleration_scaling": 0.0763}
FAST = {"velocity_scaling": 0.2064, "acceleration_scaling": 0.0763,
        "transit_velocity_scaling": 0.40, "transit_acceleration_scaling": 0.0763}


def test_argument_defaults_to_default():
    args = {a.name: a for a in mg.generate_launch_description().entities if isinstance(a, DeclareLaunchArgument)}
    assert args["speed_profile"].default_value[0].text == "default"
    assert set(args["speed_profile"].choices) == {"default", "fast"}


def test_default_loads_no_extra_file_and_keeps_the_yaml_speeds():
    assert mg.speed_profile_file("default") is None
    assert mg.effective_speed_scalings(BASE, None) == DEFAULTS


def test_fast_file_is_listed_and_its_values_win():
    path = mg.speed_profile_file("fast")
    assert os.path.basename(path) == "speed_profile_fast.yaml"
    assert mg.effective_speed_scalings(BASE, PROFILE) == FAST


def test_unknown_profile_is_refused():
    with pytest.raises(RuntimeError, match="speed_profile"):
        mg.speed_profile_file("turbo")


def test_profile_file_sets_only_the_four_scalings_for_the_executor():
    document = yaml.safe_load(open(PROFILE))
    assert set(document["box_h_cut_executor"]["ros__parameters"]) == set(FAST)
    assert set(document) == {"box_h_cut_executor", "dual_cut_coordinator"}


def test_fast_accelerations_are_capped_at_0_4_rad_s2():
    limits = yaml.safe_load(open(os.path.join(CONFIG, "joint_limits.yaml")))["joint_limits"]
    for joint, entry in limits.items():
        assert entry["max_acceleration"] * FAST["acceleration_scaling"] <= 0.4 + 1e-3, joint
        assert entry["max_acceleration"] * FAST["transit_acceleration_scaling"] <= 0.4 + 1e-3, joint
        assert entry["max_velocity"] * FAST["transit_velocity_scaling"] <= 0.3141, joint
        assert entry["max_velocity"] * FAST["velocity_scaling"] <= 0.1621, joint


def test_summary():
    text = mg.speed_profile_summary("fast", FAST)
    assert "speed_profile=fast" in text and "0.314 rad/s" in text and "0.400 rad/s^2" in text


def test_default_accelerations_are_the_same_0_4_rad_s2_cap():
    limits = yaml.safe_load(open(os.path.join(CONFIG, "joint_limits.yaml")))["joint_limits"]
    for joint, entry in limits.items():
        assert entry["max_acceleration"] * DEFAULTS["acceleration_scaling"] <= 0.4 + 1e-3, joint
        assert entry["max_acceleration"] * DEFAULTS["transit_acceleration_scaling"] <= 0.4 + 1e-3, joint
