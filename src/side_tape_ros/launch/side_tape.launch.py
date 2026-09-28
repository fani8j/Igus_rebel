"""Start perception and (dry-run by default) the separate controller process."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from side_tape_ros.core_path import DEFAULT_CONFIG


def generate_launch_description():
    config = LaunchConfiguration("config_path")
    dry_run = LaunchConfiguration("dry_run")
    use_controller = LaunchConfiguration("use_controller")
    core = LaunchConfiguration("side_tape_core")

    def set_core(context):
        directory = core.perform(context)
        return [SetEnvironmentVariable("SIDE_TAPE_CORE", directory)] if directory else []

    return LaunchDescription([
        DeclareLaunchArgument("config_path", default_value=DEFAULT_CONFIG),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("use_controller", default_value="true"),
        DeclareLaunchArgument("side_tape_core", default_value="",
                              description="Optional SIDE_TAPE_CORE directory"),
        OpaqueFunction(function=set_core),
        Node(package="side_tape_ros", executable="perception_node", name="side_tape_perception",
             parameters=[{"config_path": config}]),
        Node(package="side_tape_ros", executable="controller_node", name="side_tape_controller",
             condition=IfCondition(use_controller),
             parameters=[{"config_path": config, "dry_run": dry_run}]),
    ])
