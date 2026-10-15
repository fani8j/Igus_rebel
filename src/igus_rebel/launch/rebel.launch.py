from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="rebel",
        choices=["mock_hardware", "gazebo", "isaac_topic", "rebel"],
        description="Which hardware protocol or mock hardware should be used",
    )
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.3.11",
        description="CRI controller IPv4 address (192.168.3.11 wired LAN, 192.168.50.133 Wi-Fi)",
    )

    tool_variant_arg = DeclareLaunchArgument(
        "tool_variant",
        default_value="knife_c",
        choices=["knife_c", "xeg32_knife_v1"],
        description="Tool model on link6 (xacro arg tool_variant): knife_c = variant-C knife "
                    "holder (default), xeg32_knife_v1 = previous XEG-32 gripper + old knife. "
                    "Must match the move_group launch.",
    )

    igus_rebel_pkg = get_package_share_directory("igus_rebel")
    igus_rebel_desc_pkg = get_package_share_directory("igus_rebel_description")

    controller_config_path = PathJoinSubstitution(
        [
            igus_rebel_pkg,
            "config",
            PythonExpression(
                [
                    "'controller.yaml' if '",
                    LaunchConfiguration("hardware_protocol"),
                    "' == 'rebel' else 'controller_mock.yaml'",
                ]
            ),
        ]
    )

    robot_description_file = os.path.join(
        igus_rebel_desc_pkg,
        "urdf",
        "igus_rebel_xeg32_wrapper.urdf.xacro",
    )

    robot_description = {"robot_description": ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                robot_description_file,
                " hardware_protocol:=",
                LaunchConfiguration("hardware_protocol"),
                " robot_ip:=",
                LaunchConfiguration("robot_ip"),
                " tool_variant:=",
                LaunchConfiguration("tool_variant"),
            ]
        ),
        value_type=str,
    )}

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            controller_config_path],
        output="both",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="both",
    )

    robot_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "rebel_arm_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="both",
    )

    # Two spawners calling load_controller concurrently fail at random on hosts
    # without RT scheduling (either controller can be the one that fails). Start
    # the trajectory spawner only after the broadcaster spawner has exited.
    # Ported from github.com/ccjameslai/compal_robot commit efecba2.
    delayed_robot_trajectory_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_trajectory_controller_spawner],
        )
    )

    return LaunchDescription([
        hardware_protocol_arg,
        robot_ip_arg,
        tool_variant_arg,
        ros2_control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delayed_robot_trajectory_controller_spawner,
    ])
