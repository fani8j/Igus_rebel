"""Static checks of launch/per_arm_planning.launch.py: the right base pose comes from the calibration YAML."""
import importlib.util
import os

import pytest
from launch.actions import DeclareLaunchArgument

_HERE = os.path.dirname(__file__)
_PATH = os.path.join(_HERE, "..", "launch", "per_arm_planning.launch.py")
_spec = importlib.util.spec_from_file_location("per_arm_planning_launch", _PATH)
planning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planning)

OVERRIDES_RIGHT = os.path.join(_HERE, "..", "config", "arm_right_executor_overrides.yaml")
OVERRIDES_LEFT = os.path.join(_HERE, "..", "config", "arm_left_executor_overrides.yaml")

CALIBRATION = """dual_arm_calibration:
  source: test
  arm_right_base:
    xyz: [0.011, -0.4321, 0.0123]
    rpy: [0.001, -0.002, 0.0345]
"""


def write(tmp_path, body, name="calibration.yaml"):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


@pytest.fixture
def base_config(tmp_path):
    return write(tmp_path, "box_h_cut_executor:\n  ros__parameters:\n    cut_pattern: t_cut\n", "base.yaml")


def test_calibration_file_is_a_launch_argument():
    names = [e.name for e in planning.generate_launch_description().entities if isinstance(e, DeclareLaunchArgument)]
    assert "calibration_file" in names


def test_right_base_values_come_from_the_calibration_yaml(tmp_path, base_config):
    calibration = write(tmp_path, CALIBRATION)
    params = planning.load_executor_parameters(base_config, "", "right", calibration)
    assert params["snapshot_planning_base_xyz"] == [0.011, -0.4321, 0.0123]
    assert params["snapshot_planning_base_rpy"] == [0.001, -0.002, 0.0345]


def test_calibration_wins_over_a_stale_overrides_file_and_the_mismatch_is_reported(tmp_path, base_config):
    # The overrides file may still carry an OLD right base (it is owned elsewhere): the calibration YAML wins
    # and the launch reports the disagreement.
    calibration = write(tmp_path, CALIBRATION)
    params = planning.load_executor_parameters(base_config, OVERRIDES_RIGHT, "right", calibration)
    assert params["snapshot_planning_base_xyz"] == [0.011, -0.4321, 0.0123]
    overrides = planning.node_parameters(planning.load_yaml(__import__("pathlib").Path(OVERRIDES_RIGHT)), OVERRIDES_RIGHT)
    notes = planning.overrides_mismatches("right", calibration, overrides)
    if "snapshot_planning_base_xyz" in overrides:
        assert any("snapshot_planning_base_xyz" in n and "wins" in n for n in notes)
    assert planning.overrides_mismatches("left", calibration, overrides) == []


def test_installed_calibration_always_wins(base_config):
    # Whatever the shipped overrides file says today, the executor gets arm_right_base of the installed YAML.
    params = planning.load_executor_parameters(base_config, OVERRIDES_RIGHT, "right", planning.default_calibration_file())
    xyz, rpy = planning.load_arm_right_base(planning.default_calibration_file())
    assert params["snapshot_planning_base_xyz"] == xyz and params["snapshot_planning_base_rpy"] == rpy


def test_overrides_without_the_values_are_fine(tmp_path, base_config):
    overrides = write(tmp_path, "box_h_cut_executor:\n  ros__parameters:\n    plan_export_only: true\n", "o.yaml")
    params = planning.load_executor_parameters(base_config, overrides, "right", write(tmp_path, CALIBRATION))
    assert params["snapshot_planning_base_xyz"] == [0.011, -0.4321, 0.0123]
    assert params["plan_export_only"] is True


def test_left_arm_gets_no_snapshot_transform(tmp_path, base_config):
    params = planning.load_executor_parameters(base_config, OVERRIDES_LEFT, "left", write(tmp_path, CALIBRATION))
    assert "snapshot_planning_base_xyz" not in params or params.get("snapshot_frame_mode") != "fixed_transform"


def test_bad_calibration_is_refused(tmp_path, base_config):
    with pytest.raises(RuntimeError, match="arm_right_base"):
        planning.load_executor_parameters(base_config, "", "right", write(tmp_path, "dual_arm_calibration: {}\n"))
    with pytest.raises(RuntimeError, match="3 values"):
        planning.load_executor_parameters(
            base_config, "", "right", write(tmp_path, "dual_arm_calibration:\n  arm_right_base: {xyz: [1, 2], rpy: [0, 0, 0]}\n"))


def test_execution_stays_disabled(tmp_path, base_config):
    params = planning.load_executor_parameters(base_config, "", "right", write(tmp_path, CALIBRATION))
    assert params["execution_enabled"] is False and params["debug_motion_execution_enabled"] is False


# ---- blade collision (opt-in) ----
def test_blade_collision_is_a_launch_argument_default_true():
    args = {e.name: e for e in planning.generate_launch_description().entities if isinstance(e, DeclareLaunchArgument)}
    assert "blade_collision" in args and args["blade_collision"].default_value[0].text == "true"


def test_carton_contact_links_only_with_blade_collision(tmp_path, base_config):
    calibration = write(tmp_path, CALIBRATION)
    off = planning.load_executor_parameters(base_config, "", "left", calibration)
    assert "carton_contact_links" not in off
    config = write(tmp_path, "box_h_cut_executor:\n  ros__parameters:\n    tool_link: xeg32_tool_tip\n", "tool.yaml")
    for arm in ("left", "right"):
        on = planning.load_executor_parameters(config, "", arm, calibration, blade_collision=True)
        assert on["carton_contact_links"] == ["xeg32_tool_tip", "knife_c_blade"]
    # the launch value wins over a stale YAML / overrides entry
    stale = write(tmp_path, "box_h_cut_executor:\n  ros__parameters:\n    tool_link: xeg32_tool_tip\n"
                  "    carton_contact_links: [link6]\n", "stale.yaml")
    assert planning.load_executor_parameters(stale, "", "left", calibration, True)["carton_contact_links"] == \
        ["xeg32_tool_tip", "knife_c_blade"]
    with pytest.raises(RuntimeError):
        planning.load_executor_parameters(base_config, "", "left", calibration, blade_collision=True)  # no tool_link


def test_blade_srdf_fragment_is_inserted_before_the_closing_robot_tag():
    from ament_index_python.packages import get_package_share_directory
    share = get_package_share_directory("igus_rebel_moveit_config")
    srdf = open(os.path.join(share, "config", "igus_rebel2.srdf")).read()
    fragment = open(os.path.join(share, "config", planning.BLADE_SRDF_FRAGMENT)).read()
    assert "knife_c_blade" not in srdf          # the default single-arm SRDF stays untouched
    merged = planning.with_blade_disable_pairs(srdf, fragment)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(merged)
    pairs = [(d.get("link1"), d.get("link2")) for d in root.findall("disable_collisions")]
    assert ("knife_c_blade", "knife_c_handpiece") in pairs
    assert [p for p in pairs if "knife_c_blade" in p] == [("knife_c_blade", "knife_c_handpiece")]
    assert merged.rstrip().endswith("</robot>")
    with pytest.raises(RuntimeError):
        planning.with_blade_disable_pairs("<robot/>", "<disable_collisions/>")
