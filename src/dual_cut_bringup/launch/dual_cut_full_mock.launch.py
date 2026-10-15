"""Dual stack + both per-arm planning stacks, MOCK ONLY.

    ROS_DOMAIN_ID=<unused> ROS_LOCALHOST_ONLY=1 ros2 launch dual_cut_bringup dual_cut_full_mock.launch.py use_gui:=false

Starts rebel_dual_table_description dual_table.launch.py (hardware_protocol:=mock_hardware), the dual
rebel_dual_table_moveit_config move_group, and both_arms_planning.launch.py (arm_left and arm_right).
Use per_arm_planning.launch.py directly to bring up only one arm.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, launch_file, arguments, condition=None):
    path = os.path.join(get_package_share_directory(package), "launch", launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path), launch_arguments=arguments.items(), condition=condition)


def generate_launch_description():
    calibration_file = LaunchConfiguration("calibration_file")
    # The SAME calibration file for the driver/URDF, the dual move_group and the per-arm planners.
    dual_args = {"hardware_protocol": "mock_hardware", "calibration_file": calibration_file}
    forwarded = {name: LaunchConfiguration(name) for name in (
        "use_sim_time", "executor_config", "executor_overrides_left", "executor_overrides_right",
        "box_h_cut_start_delay", "calibration_file", "blade_collision")}
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="false", description="RViz of the dual move_group"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("executor_config", default_value=""),
        DeclareLaunchArgument("executor_overrides_left", default_value=""),
        DeclareLaunchArgument("executor_overrides_right", default_value=""),
        DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
        DeclareLaunchArgument("blade_collision", default_value="true", choices=["true", "false"],
                              description="per-arm planners: knife_c_blade collision box + carton_contact_links "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("calibration_file", default_value=os.path.join(
            get_package_share_directory("rebel_dual_table_description"), "config", "dual_arm_calibration.yaml"),
                              description="dual_arm_calibration YAML for the right arm's snapshot transform"),
        DeclareLaunchArgument("run_dual_move_group", default_value="true"),
        include("rebel_dual_table_description", "dual_table.launch.py", dual_args),
        include("rebel_dual_table_moveit_config", "move_group.launch.py",
                {**dual_args, "use_gui": LaunchConfiguration("use_gui")},
                condition=IfCondition(LaunchConfiguration("run_dual_move_group"))),
        include("dual_cut_bringup", "both_arms_planning.launch.py", forwarded),
    ])
