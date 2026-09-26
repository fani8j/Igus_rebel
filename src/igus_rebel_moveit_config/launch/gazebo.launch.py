import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config_share = get_package_share_directory("igus_rebel_moveit_config")
    # package://igus_rebel_description/... mesh URIs resolve against the directory that
    # contains the package share directory.
    description_share_parent = os.path.dirname(
        get_package_share_directory("igus_rebel_description"))
    use_sim_time = LaunchConfiguration("use_sim_time")

    resource_paths = [
        # Gazebo Garden and newer.
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", description_share_parent),
        # Gazebo Fortress (ROS 2 Humble).
        AppendEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", description_share_parent),
    ]

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        launch_arguments=[("gz_args", LaunchConfiguration("gz_args"))],
    )

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="param_bridge",
        output="screen",
        arguments=[
            "--ros-args", "-p",
            "config_file:=" + os.path.join(moveit_config_share, "config", "gz_bridge.yaml"),
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Spawn the robot model published by robot_state_publisher.
    spawn_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "igus_rebel"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    ros_controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_share, "launch", "ros_controllers.launch.py")
        ),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use simulation (Gazebo) clock if true"),
        DeclareLaunchArgument(
            "gz_args", default_value="-r empty.sdf",
            description="Gazebo arguments; add -s for a headless server"),
        *resource_paths,
        gz_bridge,
        gz_sim_launch,
        spawn_node,
        TimerAction(period=10.0, actions=[ros_controllers_launch]),
    ])
