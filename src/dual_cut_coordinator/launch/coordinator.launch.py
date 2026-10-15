"""dual_cut_coordinator node with config/coordinator.yaml (override the file with config:=...).

    ros2 launch dual_cut_coordinator coordinator.launch.py

Needs, in the same ROS domain: the dual ros2_control stack (rebel_dual_table_description
dual_table.launch.py), the dual move_group (robot model and planning scene), and the two per-arm
executors (or tools/fake_plan_provider via fake_providers.launch.py). Nothing executes unless
execution_enabled and a per-phase enable are set to true at runtime.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("dual_cut_coordinator")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "coordinator.yaml")),
        Node(package="dual_cut_coordinator", executable="dual_cut_coordinator_node", name="dual_cut_coordinator",
             output="screen", parameters=[LaunchConfiguration("config")]),
    ])
