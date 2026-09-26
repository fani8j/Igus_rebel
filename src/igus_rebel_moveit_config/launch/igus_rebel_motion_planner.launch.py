import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Arguments forwarded unchanged to move_group.launch.py.
FORWARDED_ARGUMENTS = (
    "hardware_protocol",
    "use_sim_time",
    "use_gui",
    "run_teleop",
    "run_gamepad",
    "run_camera_pole",
    "run_box_h_cut_executor",
    "box_h_cut_start_delay",
)


def generate_launch_description():
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("igus_rebel_moveit_config"),
                "launch",
                "move_group.launch.py",
            )
        ),
        launch_arguments={
            name: LaunchConfiguration(name) for name in FORWARDED_ARGUMENTS
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "hardware_protocol",
                default_value="rebel",
                choices=["mock_hardware", "gazebo", "rebel"],
            ),
            DeclareLaunchArgument(
                "run_teleop", default_value="false",
                description="Start MoveIt Servo (needed by keyboard and gamepad teleop)"),
            DeclareLaunchArgument(
                "run_gamepad", default_value="false",
                description="Start joy_node + gamepad teleop (also needs run_teleop:=true)"),
            DeclareLaunchArgument(
                "run_camera_pole", default_value="true",
                description="Maintain the external camera_pole collision object"),
            DeclareLaunchArgument(
                "run_box_h_cut_executor",
                default_value="true",
                description="Start the external-snapshot plan-only H-cut MTC executor",
            ),
            DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
            moveit_launch,
        ]
    )
