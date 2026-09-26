import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_gui_arg = DeclareLaunchArgument("use_gui", default_value="false")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")
    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="rebel",
        choices=["mock_hardware", "gazebo", "rebel"],
    )
    run_h_cut_executor_arg = DeclareLaunchArgument(
        "run_box_h_cut_executor",
        default_value="true",
        description="Start the external-snapshot plan-only H-cut MTC executor",
    )
    h_cut_delay_arg = DeclareLaunchArgument(
        "box_h_cut_start_delay", default_value="3.0"
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("igus_rebel_moveit_config"),
                "launch",
                "move_group.launch.py",
            )
        ),
        launch_arguments={
            "hardware_protocol": LaunchConfiguration("hardware_protocol"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "use_gui": LaunchConfiguration("use_gui"),
            "run_box_h_cut_executor": LaunchConfiguration(
                "run_box_h_cut_executor"
            ),
            "box_h_cut_start_delay": LaunchConfiguration(
                "box_h_cut_start_delay"
            ),
        }.items(),
    )

    return LaunchDescription(
        [
            use_gui_arg,
            use_sim_time_arg,
            hardware_protocol_arg,
            run_h_cut_executor_arg,
            h_cut_delay_arg,
            moveit_launch,
        ]
    )
