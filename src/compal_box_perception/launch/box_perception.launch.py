import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def calibration_nodes(context):
    if LaunchConfiguration("publish_calibration").perform(context).lower() != "true":
        return []
    name = LaunchConfiguration("calibration_name").perform(context)
    path = os.path.expanduser(
        f"~/.ros2/easy_handeye2/calibrations/{name}.calib"
    )
    with open(path, encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    parameters = calibration["parameters"]
    transform = calibration["transform"]
    translation = transform["translation"]
    rotation = transform["rotation"]
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="box_camera_calibration",
            arguments=[
                "--x", str(translation["x"]),
                "--y", str(translation["y"]),
                "--z", str(translation["z"]),
                "--qx", str(rotation["x"]),
                "--qy", str(rotation["y"]),
                "--qz", str(rotation["z"]),
                "--qw", str(rotation["w"]),
                "--frame-id", parameters["robot_base_frame"],
                "--child-frame-id", parameters["tracking_base_frame"],
            ],
            output="screen",
        )
    ]

def generate_launch_description():
    package_share = get_package_share_directory("compal_box_perception")
    default_config = os.path.join(package_share, "config", "box_perception.yaml")
    rviz_config = os.path.join(package_share, "config", "box_perception.rviz")
    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config, description="Perception parameter YAML"
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="false", description="Start the dedicated RViz view"
    )
    calibration_arg = DeclareLaunchArgument(
        "publish_calibration",
        default_value="true",
        description="Publish the saved eye-on-base calibration",
    )
    calibration_name_arg = DeclareLaunchArgument(
        "calibration_name",
        default_value="gemini336_world_v4_t",
        description="Saved easy_handeye2 calibration name",
    )
    return LaunchDescription(
        [
            config_arg,
            rviz_arg,
            calibration_arg,
            calibration_name_arg,
            OpaqueFunction(function=calibration_nodes),
            Node(
                package="compal_box_perception",
                executable="box_perception_node",
                name="box_perception",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="box_perception_rviz",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
