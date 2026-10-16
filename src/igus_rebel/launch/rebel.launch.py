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
from launch.conditions import IfCondition
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
        description="Tool variant passed to the xacro",
    )
    mock_move_joint_arg = DeclareLaunchArgument(
        "mock_move_joint",
        default_value="false",
        choices=["true", "false"],
        description="mock_hardware only: serve the move_joint action with a mock (streams "
        "positions through an inactive forward_command_controller). Ignored for other "
        "hardware_protocol values; the real driver serves the action itself.",
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

    use_mock_move_joint = IfCondition(
        PythonExpression([
            "'", LaunchConfiguration("mock_move_joint"), "' == 'true' and '",
            LaunchConfiguration("hardware_protocol"), "' == 'mock_hardware'",
        ])
    )

    mock_move_joint_server = Node(
        package="igus_rebel",
        executable="mock_move_joint_server",
        # Same node name as the real driver's plugin node, so the action is /igus_rebel/move_joint.
        name="igus_rebel",
        output="both",
        condition=use_mock_move_joint,
    )

    mock_move_joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "mock_move_joint_controller",
            "--controller-manager",
            "/controller_manager",
            "--inactive",
        ],
        output="both",
        condition=use_mock_move_joint,
    )

    # Same one-at-a-time rule as above: after the trajectory spawner exited.
    delayed_mock_move_joint_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_trajectory_controller_spawner,
            on_exit=[mock_move_joint_controller_spawner],
        )
    )

    return LaunchDescription([
        hardware_protocol_arg,
        robot_ip_arg,
        tool_variant_arg,
        mock_move_joint_arg,
        ros2_control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delayed_robot_trajectory_controller_spawner,
        mock_move_joint_server,
        delayed_mock_move_joint_controller_spawner,
    ])
