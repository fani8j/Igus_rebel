"""dual_cut_coordinator node with config/coordinator.yaml (override the file with config:=...).

    ros2 launch dual_cut_coordinator coordinator.launch.py

speed_profile_file:=<yaml> (default "" = none) is loaded AFTER config, so its dual_cut_coordinator section wins (used by
dual_cut_system.launch.py speed_profile:=fast for the JTC sub-chord limits cri_subchord_*).

Needs, in the same ROS domain: the dual ros2_control stack (rebel_dual_table_description
dual_table.launch.py), the dual move_group (robot model and planning scene), and the two per-arm
executors (or tools/fake_plan_provider via fake_providers.launch.py). Nothing executes unless
execution_enabled and a per-phase enable are set to true at runtime.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def coordinator_node(context):
    parameters = [LaunchConfiguration("config")]
    profile = LaunchConfiguration("speed_profile_file").perform(context).strip()
    if profile:
        parameters.append(os.path.expanduser(profile))
    return [Node(package="dual_cut_coordinator", executable="dual_cut_coordinator_node", name="dual_cut_coordinator",
                 output="screen", parameters=parameters)]


def generate_launch_description():
    share = get_package_share_directory("dual_cut_coordinator")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "coordinator.yaml")),
        DeclareLaunchArgument("speed_profile_file", default_value="",
                              description="optional extra parameter YAML loaded after config (speed profile); empty = none"),
        OpaqueFunction(function=coordinator_node),
    ])
