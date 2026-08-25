from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


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
        default_value="gemini336_world_v3e",
        description="Saved easy_handeye2 calibration name",
    )
    return LaunchDescription(
        [
            config_arg,
            rviz_arg,
            calibration_arg,
            calibration_name_arg,
            Node(
                package="easy_handeye2",
                executable="handeye_publisher",
                name="handeye_publisher",
                parameters=[{"name": LaunchConfiguration("calibration_name")}],
                condition=IfCondition(LaunchConfiguration("publish_calibration")),
                output="screen",
            ),
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
