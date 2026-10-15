"""Fake per-arm executor exports for the WP6 mock tests: /arm_left/box_h_cut_executor and
/arm_right/box_h_cut_executor serving plan_h_cut_from_state / get_planned_solution from a scenario.

    ros2 launch dual_cut_coordinator fake_providers.launch.py scenario:=clear   (clear | collide | serial | path)

lie_arm:=right exports a base transform 5 mm off the dual URDF (the frame-consistency gate must refuse);
cut_offset_arm:=right cut_offset_m:=0.005 shifts the exported cut targets (the tip-on-line gate must refuse);
snapshot_age_s sets the age of the fake snapshot at start (it keeps growing).
"""
import os
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def setup(context):
    me = get_package_share_directory("dual_cut_coordinator")
    desc = get_package_share_directory("rebel_dual_table_description")
    cfg = get_package_share_directory("rebel_dual_table_moveit_config")
    scenario = LaunchConfiguration("scenario").perform(context)
    path = scenario if os.path.isfile(scenario) else os.path.join(me, "config", "scenarios", scenario + ".yaml")
    calibration = LaunchConfiguration("calibration_file").perform(context) or os.path.join(
        desc, "config", "dual_arm_calibration.yaml")
    urdf = ParameterValue(Command([FindExecutable(name="xacro"), " ", os.path.join(desc, "urdf", "rebel_dual_table.urdf.xacro"),
                                   " include_ros2_control:=false calibration_file:=", calibration]), value_type=str)
    srdf = Path(os.path.join(cfg, "config", "rebel_dual_table.srdf")).read_text()
    lie = LaunchConfiguration("lie_arm").perform(context)
    off_arm = LaunchConfiguration("cut_offset_arm").perform(context)
    stamp_ns = int(time.time() * 1e9 - float(LaunchConfiguration("snapshot_age_s").perform(context)) * 1e9)
    nodes = []
    for side in ("left", "right"):
        nodes.append(Node(
            package="dual_cut_coordinator", executable="fake_plan_provider", namespace=f"arm_{side}",
            name="box_h_cut_executor", output="screen",
            parameters=[{
                "arm": side, "scenario_file": path, "calibration_file": calibration,
                "robot_description": urdf, "robot_description_semantic": srdf,
                "snapshot_stamp_ns": stamp_ns,
                "lie_transform": lie == side,
                "cut_line_offset_m": float(LaunchConfiguration("cut_offset_m").perform(context)) if off_arm == side else 0.0,
                "plan_delay_s": float(LaunchConfiguration("plan_delay_s").perform(context)),
            }]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="clear"),
        DeclareLaunchArgument("calibration_file", default_value=""),
        DeclareLaunchArgument("snapshot_age_s", default_value="5.0"),
        DeclareLaunchArgument("lie_arm", default_value="none"),
        DeclareLaunchArgument("cut_offset_arm", default_value="none"),
        DeclareLaunchArgument("cut_offset_m", default_value="0.005"),
        DeclareLaunchArgument("plan_delay_s", default_value="0.0"),
        OpaqueFunction(function=setup),
    ])
