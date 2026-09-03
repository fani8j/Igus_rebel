import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
    run_mtc_program_arg = DeclareLaunchArgument(
        "run_mtc_program",
        default_value="false",
        description="Run the legacy configured PTP/LIN MTC task",
    )
    execute_mtc_arg = DeclareLaunchArgument(
        "execute_mtc",
        default_value="false",
        description="Execute the legacy PTP/LIN task; unrelated to box motion",
    )
    run_perception_arg = DeclareLaunchArgument(
        "run_box_perception",
        default_value="true",
        description="Start compressed RGB-D box perception",
    )
    publish_calibration_arg = DeclareLaunchArgument(
        "publish_calibration",
        default_value="true",
        description="Publish the saved easy_handeye2 camera calibration",
    )
    calibration_name_arg = DeclareLaunchArgument(
        "calibration_name", default_value="gemini336_world_v4_t"
    )
    run_box_executor_arg = DeclareLaunchArgument(
        "run_box_motion_executor",
        default_value="false",
        description="Start the legacy plan-only single-hover executor",
    )
    box_motion_delay_arg = DeclareLaunchArgument(
        "box_motion_start_delay", default_value="3.0"
    )
    run_h_cut_executor_arg = DeclareLaunchArgument(
        "run_box_h_cut_executor",
        default_value="true",
        description="Start the frozen-snapshot plan-only H-cut MTC executor",
    )
    h_cut_delay_arg = DeclareLaunchArgument(
        "box_h_cut_start_delay", default_value="3.0"
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("compal_box_perception"),
                "launch",
                "box_perception.launch.py",
            )
        ),
        condition=IfCondition(LaunchConfiguration("run_box_perception")),
        launch_arguments={
            "rviz": "false",
            "publish_calibration": LaunchConfiguration("publish_calibration"),
            "calibration_name": LaunchConfiguration("calibration_name"),
        }.items(),
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
            "run_mtc_program": LaunchConfiguration("run_mtc_program"),
            "execute_mtc": LaunchConfiguration("execute_mtc"),
            "run_box_motion_executor": LaunchConfiguration(
                "run_box_motion_executor"
            ),
            "box_motion_start_delay": LaunchConfiguration(
                "box_motion_start_delay"
            ),
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
            run_mtc_program_arg,
            execute_mtc_arg,
            run_perception_arg,
            publish_calibration_arg,
            calibration_name_arg,
            run_box_executor_arg,
            box_motion_delay_arg,
            run_h_cut_executor_arg,
            h_cut_delay_arg,
            perception_launch,
            moveit_launch,
        ]
    )