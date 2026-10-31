"""Gamepad teleop through the controller's own Cartesian jog (the "CRI route").

Starts joy_node and the gamepad node with output:=cri. NO MoveIt, NO Servo, NO controllers: the
TwistStamped goes to the Igus/Rebel driver's ~/cart_jog_cmd and the ReBeL controller does the IK.

Prerequisites (the driver refuses the mode otherwise, with a message):
  * the stack is running, e.g. ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel
  * the trajectory controller is DEACTIVATED:
      ros2 control switch_controllers --deactivate rebel_arm_trajectory_controller
The node calls /igus_rebel/cart_jog_enable true on start (cri_auto_enable:=false: press the toggle
button instead) and false on Ctrl+C. The arm moves only while the enable button (gamepad.yaml
button_enable) is held.

Keyboard: not launched here (it needs a terminal):
  ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard --ros-args -p output:=cri
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE = "igus_rebel_moveit_config"


def generate_launch_description():
    gamepad_yaml = os.path.join(get_package_share_directory(PACKAGE), "config", "gamepad.yaml")

    args = [
        DeclareLaunchArgument("device_id", default_value="0", description="joy_node device id"),
        DeclareLaunchArgument(
            "cri_cmd_topic", default_value="/igus_rebel/cart_jog_cmd",
            description="Driver topic for the Cartesian jog command (TwistStamped, base frame)"),
        DeclareLaunchArgument(
            "cri_enable_service", default_value="/igus_rebel/cart_jog_enable",
            description="Driver service (std_srvs/SetBool) that turns the Cartesian jog mode on/off"),
        DeclareLaunchArgument(
            "cri_auto_enable", default_value="true",
            description="Enable the mode on start (false: press button_cri_toggle)"),
        DeclareLaunchArgument(
            "button_cri_toggle", default_value="6",
            description="Joy button that toggles the mode (-1 disables the button)"),
        DeclareLaunchArgument(
            "cri_max_linear_vel", default_value="0.05",
            description="Full-stick linear speed [m/s]; the driver scales and clamps it (UNMEASURED mapping)"),
    ]

    joy_node = Node(
        package="joy",
        executable="joy_node",
        # autorepeat keeps the Joy stream alive at 20 Hz while a stick is held still (normal gap
        # 50 ms); the gamepad node zeroes the twist if Joy stops for joy_timeout_s (0.15 s, the
        # same order as the driver's own cart_jog_cmd watchdog).
        parameters=[{
            "device_id": ParameterValue(LaunchConfiguration("device_id"), value_type=int),
            "autorepeat_rate": 20.0,
        }],
        output="screen",
    )
    gamepad_node = Node(
        package=PACKAGE,
        executable="rebel_servo_teleop_gamepad",
        parameters=[
            gamepad_yaml,
            {
                "output": "cri",
                "cri_cmd_topic": LaunchConfiguration("cri_cmd_topic"),
                "cri_enable_service": LaunchConfiguration("cri_enable_service"),
                "cri_auto_enable": ParameterValue(LaunchConfiguration("cri_auto_enable"), value_type=bool),
                "button_cri_toggle": ParameterValue(LaunchConfiguration("button_cri_toggle"), value_type=int),
                "cri_max_linear_vel": ParameterValue(LaunchConfiguration("cri_max_linear_vel"), value_type=float),
            },
        ],
        output="screen",
    )
    return LaunchDescription(args + [joy_node, gamepad_node])
