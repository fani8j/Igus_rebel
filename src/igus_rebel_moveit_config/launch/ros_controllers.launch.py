from launch import LaunchDescription
from launch_ros.actions import Node


def spawner(controller):
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[controller, "--controller-manager", "/controller_manager"],
        output="both",
    )


def generate_launch_description():
    # Controllers are defined in config/ros_controllers.yaml (loaded by gz_ros2_control).
    return LaunchDescription([
        spawner("rebel_arm_trajectory_controller"),
        spawner("joint_state_broadcaster"),
    ])
