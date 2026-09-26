"""ArUco/TCP verification; no detection-triggered motion or calibration changes."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    motion = get_package_share_directory("igus_rebel_moveit_config")
    perception = get_package_share_directory("compal_box_perception")

    def include(package, filename, arguments, condition=None):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(package, "launch", filename)),
            launch_arguments=arguments.items(), condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware",
                              choices=["mock_hardware", "rebel"]),
        DeclareLaunchArgument("start_control", default_value="true",
                              description="False when the hardware driver is already running"),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("execution_allowed", default_value="false",
                              description="Allows explicit execute_plan requests; never auto-executes"),
        DeclareLaunchArgument("publish_calibration", default_value="true",
                              description="False if this camera TF already has a publisher"),
        DeclareLaunchArgument("calibration_name", default_value="rebel_handeye_h_v2"),
        DeclareLaunchArgument("detector_config", default_value=os.path.join(
            perception, "config", "aruco_verification.yaml")),
        DeclareLaunchArgument("motion_config", default_value=os.path.join(
            motion, "config", "calibration_verification.yaml")),
        include(get_package_share_directory("igus_rebel"), "rebel.launch.py", {
            "hardware_protocol": LaunchConfiguration("hardware_protocol")},
            IfCondition(LaunchConfiguration("start_control"))),
        include(perception, "box_perception.launch.py", {
            "run_detection": "false", "rviz": "false",
            "publish_calibration": LaunchConfiguration("publish_calibration"),
            "calibration_name": LaunchConfiguration("calibration_name")}),
        Node(package="compal_box_perception", executable="aruco_verification",
             name="aruco_verification", parameters=[LaunchConfiguration("detector_config")],
             output="screen"),
        include(motion, "move_group.launch.py", {
            "hardware_protocol": LaunchConfiguration("hardware_protocol"),
            "use_gui": LaunchConfiguration("use_gui"),
            "run_teleop": "false", "run_box_h_cut_executor": "false",
            "run_calibration_verification": "true",
            "calibration_verification_config": LaunchConfiguration("motion_config"),
            "calibration_execution_allowed": LaunchConfiguration("execution_allowed"),
            "rviz_config": os.path.join(motion, "config", "calibration_verification.rviz")}),
    ])
