from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
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
    move_joint_vel_arg = DeclareLaunchArgument(
        "move_joint_max_velocity_percent",
        default_value="25",
        description="CRI MoveJoint velocity cap in percent (driver default 25, hard cap 100); raise only "
                    "for supervised real-arm tests",
    )
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.3.11",
        description="CRI controller IPv4 address (192.168.3.11 wired LAN, 192.168.50.133 Wi-Fi)",
    )

    jog_velocity_scale_arg = DeclareLaunchArgument(
        "jog_velocity_scale",
        default_value="1.65",
        description="Joint jog scale: jog [%] = scale * velocity [deg/s]. 1.65 since 2026-10-31 (real left arm: "
                    "the robot moves 0.600 deg/s per jog %, exact scale 1.667, 10-100 %; the C++ fallback "
                    "when the parameter is absent stays 2.0). controller.yaml gains assume this scale.",
    )
    cart_jog_percent_per_mps_arg = DeclareLaunchArgument(
        "cart_jog_percent_per_mps",
        default_value="200.0",
        description="Cartesian jog (CRI route): jog [%] per m/s of linear speed. UNMEASURED placeholder.",
    )
    cart_jog_percent_per_radps_arg = DeclareLaunchArgument(
        "cart_jog_percent_per_radps",
        default_value="100.0",
        description="Cartesian jog: jog [%] per rad/s of angular speed. UNMEASURED placeholder.",
    )
    cart_jog_max_percent_arg = DeclareLaunchArgument(
        "cart_jog_max_percent",
        default_value="10.0",
        description="Cartesian jog: per-axis clamp [%].",
    )
    cart_jog_allow_rotation_arg = DeclareLaunchArgument(
        "cart_jog_allow_rotation",
        default_value="false",
        description="Cartesian jog: false zeroes the angular input (true only after measuring A/B/C).",
    )
    cart_jog_signs_arg = DeclareLaunchArgument(
        "cart_jog_signs",
        default_value="1,1,1,1,1,1",
        description="Cartesian jog: per-axis sign (1 or -1) for x,y,z,rx,ry,rz, to fix directions without a rebuild.",
    )
    cart_jog_timeout_s_arg = DeclareLaunchArgument(
        "cart_jog_timeout_s",
        default_value="0.15",
        description="Cartesian jog: zero jog if no cart_jog_cmd arrives for this long [s].",
    )
    cart_jog_idle_exit_s_arg = DeclareLaunchArgument(
        "cart_jog_idle_exit_s",
        default_value="10.0",
        description="Cartesian jog: leave the mode after this many seconds without a cart_jog_cmd (0 = never).",
    )
    cart_jog_base_frame_arg = DeclareLaunchArgument(
        "cart_jog_base_frame",
        default_value="base_link",
        description="Cartesian jog: header.frame_id a cart_jog_cmd must carry (empty is accepted).",
    )
    joint_jog_max_percent_arg = DeclareLaunchArgument(
        "joint_jog_max_percent",
        default_value="10.0",
        description="Direct joint jog (CRI route): per-joint clamp [%] on top of jog_velocity_scale.",
    )
    joint_jog_timeout_s_arg = DeclareLaunchArgument(
        "joint_jog_timeout_s",
        default_value="0.15",
        description="Direct joint jog: zero jog if no joint_jog_cmd arrives for this long [s].",
    )
    joint_jog_idle_exit_s_arg = DeclareLaunchArgument(
        "joint_jog_idle_exit_s",
        default_value="10.0",
        description="Direct joint jog: leave the mode after this many seconds without a joint_jog_cmd (0 = never).",
    )
    robot_port_arg = DeclareLaunchArgument(
        "robot_port",
        default_value="3920",
        description="CRI TCP port (3920 on the real controller; change only for a loopback fake)",
    )

    controller_params_file_arg = DeclareLaunchArgument(
        "controller_params_file",
        default_value="",
        description="absolute path of a ros2_control parameter file replacing the default "
                    "controller.yaml (empty = default, nothing changes)",
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

    # controller_params_file: absolute path of a replacement ros2_control parameter file (default
    # "" = the package's controller.yaml / controller_mock.yaml, i.e. no change in behaviour).
    controller_config_path = PythonExpression(
        [
            "'",
            LaunchConfiguration("controller_params_file"),
            "' if '",
            LaunchConfiguration("controller_params_file"),
            "' != '' else '",
            os.path.join(igus_rebel_pkg, "config"),
            "/' + ('controller.yaml' if '",
            LaunchConfiguration("hardware_protocol"),
            "' == 'rebel' else 'controller_mock.yaml')",
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
                " robot_port:=",
                LaunchConfiguration("robot_port"),
                " jog_velocity_scale:=",
                LaunchConfiguration("jog_velocity_scale"),
                " cart_jog_percent_per_mps:=",
                LaunchConfiguration("cart_jog_percent_per_mps"),
                " cart_jog_percent_per_radps:=",
                LaunchConfiguration("cart_jog_percent_per_radps"),
                " cart_jog_max_percent:=",
                LaunchConfiguration("cart_jog_max_percent"),
                " cart_jog_allow_rotation:=",
                LaunchConfiguration("cart_jog_allow_rotation"),
                " cart_jog_signs:=",
                LaunchConfiguration("cart_jog_signs"),
                " cart_jog_timeout_s:=",
                LaunchConfiguration("cart_jog_timeout_s"),
                " cart_jog_idle_exit_s:=",
                LaunchConfiguration("cart_jog_idle_exit_s"),
                " cart_jog_base_frame:=",
                LaunchConfiguration("cart_jog_base_frame"),
                " joint_jog_max_percent:=",
                LaunchConfiguration("joint_jog_max_percent"),
                " joint_jog_timeout_s:=",
                LaunchConfiguration("joint_jog_timeout_s"),
                " joint_jog_idle_exit_s:=",
                LaunchConfiguration("joint_jog_idle_exit_s"),
                " tool_variant:=",
                LaunchConfiguration("tool_variant"),
                " move_joint_max_velocity_percent:=",
                LaunchConfiguration("move_joint_max_velocity_percent"),
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
        jog_velocity_scale_arg,
        cart_jog_percent_per_mps_arg,
        cart_jog_percent_per_radps_arg,
        cart_jog_max_percent_arg,
        cart_jog_allow_rotation_arg,
        cart_jog_signs_arg,
        cart_jog_timeout_s_arg,
        cart_jog_idle_exit_s_arg,
        cart_jog_base_frame_arg,
        joint_jog_max_percent_arg,
        joint_jog_timeout_s_arg,
        joint_jog_idle_exit_s_arg,
        robot_port_arg,
        controller_params_file_arg,
        move_joint_vel_arg,
        tool_variant_arg,
        mock_move_joint_arg,
        ros2_control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delayed_robot_trajectory_controller_spawner,
        mock_move_joint_server,
        delayed_mock_move_joint_controller_spawner,
    ])
