import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    moveit_config_share = get_package_share_directory("igus_rebel_moveit_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Same arm + XEG-32 + TCP model as the real/mock stack, with gz_ros2_control hardware.
    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            os.path.join(get_package_share_directory("igus_rebel_description"),
                         "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro"),
            " hardware_protocol:=gazebo",
        ]),
        value_type=str,
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "gz_args": LaunchConfiguration("gz_args"),
        }.items(),
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
        output="both",
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_share, "launch", "move_group.launch.py")),
        launch_arguments={
            "hardware_protocol": "gazebo",
            "use_sim_time": use_sim_time,
            "use_gui": LaunchConfiguration("use_gui"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true", description="Start RViz"),
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="Use sim time if true"),
        DeclareLaunchArgument("gz_args", default_value="-r empty.sdf",
                              description="Gazebo arguments; add -s for a headless server"),
        gazebo_launch,
        robot_state_pub_node,
        moveit_launch,
    ])
