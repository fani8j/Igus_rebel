"""Both per-arm planning stacks (arm_left, arm_right); expects the dual stack to be running already.

    ros2 launch dual_cut_bringup both_arms_planning.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    path = os.path.join(get_package_share_directory("dual_cut_bringup"), "launch",
                        "per_arm_planning.launch.py")
    common = {"use_sim_time": LaunchConfiguration("use_sim_time"),
              "executor_config": LaunchConfiguration("executor_config"),
              "box_h_cut_start_delay": LaunchConfiguration("box_h_cut_start_delay"),
              "calibration_file": LaunchConfiguration("calibration_file"),
              "blade_collision": LaunchConfiguration("blade_collision"),
              "table_in_arm_scene": LaunchConfiguration("table_in_arm_scene"),
              "table_padding_m": LaunchConfiguration("table_padding_m"),
              "speed_profile": LaunchConfiguration("speed_profile")}

    def arm(name):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(path),
            launch_arguments={**common, "arm": name,
                              "executor_overrides": LaunchConfiguration(f"executor_overrides_{name}")}.items())

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("executor_config", default_value=""),
        DeclareLaunchArgument("executor_overrides_left", default_value=""),
        DeclareLaunchArgument("executor_overrides_right", default_value=""),
        DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
        DeclareLaunchArgument("blade_collision", default_value="true", choices=["true", "false"],
                              description="per-arm models with the knife_c_blade box + carton_contact_links "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("table_in_arm_scene", default_value="true", choices=["true", "false"],
                              description="the dual cell's table as a collision object in both per-arm scenes "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("table_padding_m", default_value="0.010",
                              description="padding of that table box (default 10 mm = the coordinator's table_min_clearance_m; "
                                          "the base link is exempt, every other link keeps the margin)"),
        DeclareLaunchArgument("speed_profile", default_value="default", choices=["default", "fast"],
                              description="opt-in speed profile of the per-arm executors (see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("calibration_file", default_value="",
                              description="dual_arm_calibration YAML for the right arm's snapshot transform"),
        arm("left"),
        arm("right"),
    ])
