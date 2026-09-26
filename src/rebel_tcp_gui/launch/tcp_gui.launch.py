"""Launch the interactive TCP pivot-calibration GUI."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="rebel_tcp_gui",
                executable="tcp_gui",
                name="tcp_gui",
                output="screen",
            ),
        ]
    )
