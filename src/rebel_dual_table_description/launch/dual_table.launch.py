"""Bring-up of the dual-arm test table: one controller_manager, two arms.

    ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware

Starts ros2_control_node (robot_description from rebel_dual_table.urdf.xacro), robot_state_publisher,
the controller spawners (strictly one after the other: joint_state_broadcaster, then
arm_left_trajectory_controller, then arm_right_trajectory_controller, and in mock with mock_move_joint:=true then
the two INACTIVE arm_<side>_mock_move_joint_controller) and, by default, a static
transform world -> camera_link taken from the calibration YAML. In mock_hardware (no camera driver)
it also publishes the static camera_link -> camera_color_optical_frame transform that the Orbbec
driver provides on the real cell (publish_optical_tf:=auto), which camera_pole_collision.py needs.
"""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE = "rebel_dual_table_description"
CM = "/controller_manager"


def rpy_to_quat(roll, pitch, yaw):
    """URDF fixed-axis rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def load_calibration(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return data["dual_arm_calibration"]
    except (KeyError, TypeError):
        raise RuntimeError(f"{path}: missing the top-level 'dual_arm_calibration' key")


def launch_setup(context):
    hw = LaunchConfiguration("hardware_protocol").perform(context)
    left_ip = LaunchConfiguration("left_robot_ip").perform(context).strip()
    right_ip = LaunchConfiguration("right_robot_ip").perform(context).strip()
    left_jog_scale = LaunchConfiguration("left_jog_velocity_scale").perform(context).strip()
    right_jog_scale = LaunchConfiguration("right_jog_velocity_scale").perform(context).strip()
    calibration_file = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context))

    if hw == "rebel" and left_ip == right_ip:
        raise RuntimeError(
            f"hardware_protocol:=rebel needs two different controller addresses, but "
            f"left_robot_ip == right_robot_ip == '{left_ip}'. Both arms on one IP would send "
            f"both arms' jog commands to the same robot. Pass left_robot_ip:=... right_robot_ip:=...")
    if not os.path.isfile(calibration_file):
        raise RuntimeError(f"calibration_file '{calibration_file}' does not exist")
    calib = load_calibration(calibration_file)
    source = str(calib.get("source", "unknown"))
    calibration_warning = []
    if hw == "rebel" and source != "calibrated":
        # Not a blocker (commissioning needs the stack before calibration exists), but the arm_right
        # base pose is NOT trusted for collision avoidance between the arms until the trust gate of
        # rebel_dual_arm_calibration passed (source: calibrated).
        calibration_warning = [LogInfo(msg=(
            f"WARNING: calibration source is '{source}' (not 'calibrated'): the arm_right base pose "
            f"in {calibration_file} has not passed the trust gate. Do not rely on MoveIt to keep "
            f"the two arms apart; move one arm at a time with the other parked."))]

    share = get_package_share_directory(PACKAGE)
    controllers = os.path.join(
        share, "config", "controllers.yaml" if hw == "rebel" else "controllers_mock.yaml")

    robot_description = {"robot_description": ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            os.path.join(share, "urdf", "rebel_dual_table.urdf.xacro"),
            " hardware_protocol:=", hw,
            " left_robot_ip:=", left_ip,
            " right_robot_ip:=", right_ip,
            " left_jog_velocity_scale:=", left_jog_scale,
            " right_jog_velocity_scale:=", right_jog_scale,
            " calibration_file:=", calibration_file,
        ]),
        value_type=str,
    )}

    ros2_control_node = Node(
        package="controller_manager", executable="ros2_control_node",
        parameters=[robot_description, controllers], output="both")
    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[robot_description], output="both")

    # On real hardware the trajectory controllers are only loaded and configured (inactive) unless
    # activate_controllers:=true: the operator first checks which physical arm answers which IP
    # (move each arm from its pendant, watch /joint_states), then activates them deliberately with
    # 'ros2 control switch_controllers --activate arm_left_trajectory_controller ...'.
    activate = LaunchConfiguration("activate_controllers").perform(context).strip().lower()
    if activate == "auto":
        activate = "true" if hw == "mock_hardware" else "false"
    if activate not in ("true", "false"):
        raise RuntimeError(f"activate_controllers must be auto|true|false, got '{activate}'")

    def spawner(name, inactive=False):
        arguments = [name, "--controller-manager", CM]
        if inactive:
            arguments.append("--inactive")
        return Node(package="controller_manager", executable="spawner",
                    arguments=arguments, output="both")

    jsb = spawner("joint_state_broadcaster")
    left = spawner("arm_left_trajectory_controller", inactive=activate == "false")
    right = spawner("arm_right_trajectory_controller", inactive=activate == "false")
    # Spawners calling load_controller concurrently fail at random on hosts without RT scheduling
    # (see igus_rebel/launch/rebel.launch.py), so each one starts only after the previous exited.
    chain = [
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[left])),
        RegisterEventHandler(OnProcessExit(target_action=left, on_exit=[right])),
    ]

    # MOCK ONLY: the robot-side joint move (MoveJoint action + hardware_state + robot_joint_states) of each arm, served by
    # igus_rebel's mock_move_joint_server under the node name the real driver has (node_name hardware parameter,
    # igus_rebel_left / igus_rebel_right): /igus_rebel_<side>/{move_joint,hardware_state,robot_joint_states}. Never with
    # hardware_protocol:=rebel (the real driver serves these itself).
    mock_move_joint = LaunchConfiguration("mock_move_joint").perform(context).strip().lower()
    if mock_move_joint not in ("true", "false"):
        raise RuntimeError(f"mock_move_joint must be true|false, got '{mock_move_joint}'")
    mock_servers = []
    if mock_move_joint == "true" and hw == "mock_hardware":
        ml = spawner("arm_left_mock_move_joint_controller", inactive=True)
        mr = spawner("arm_right_mock_move_joint_controller", inactive=True)
        chain += [
            RegisterEventHandler(OnProcessExit(target_action=right, on_exit=[ml])),
            RegisterEventHandler(OnProcessExit(target_action=ml, on_exit=[mr])),
        ]
        igus_rebel = get_package_share_directory("igus_rebel")
        for side in ("left", "right"):
            mock_servers.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(igus_rebel, "launch", "mock_move_joint_arm.launch.py")),
                launch_arguments={
                    "node_name": f"igus_rebel_{side}", "joint_prefix": f"arm_{side}_",
                    "controller_name": f"arm_{side}_mock_move_joint_controller",
                    "trajectory_controller_name": f"arm_{side}_trajectory_controller",
                    "component_name": f"arm_{side}_hw"}.items()))
    elif mock_move_joint == "true" and hw == "rebel":
        calibration_warning = [*calibration_warning, LogInfo(msg=(
            "mock_move_joint is ignored with hardware_protocol:=rebel (the real driver serves move_joint itself)"))]

    actions = [*calibration_warning, ros2_control_node, robot_state_publisher, jsb, *chain, *mock_servers]

    cam = calib["camera_link"]
    x, y, z = (float(v) for v in cam["xyz"])
    qx, qy, qz, qw = rpy_to_quat(*(float(v) for v in cam["rpy"]))
    actions.append(Node(
        condition=IfCondition(LaunchConfiguration("publish_camera_tf")),
        package="tf2_ros", executable="static_transform_publisher", name="camera_link_static_tf",
        arguments=["--x", repr(x), "--y", repr(y), "--z", repr(z),
                   "--qx", repr(qx), "--qy", repr(qy), "--qz", repr(qz), "--qw", repr(qw),
                   "--frame-id", "world", "--child-frame-id", "camera_link"],
        output="both"))
    optical = LaunchConfiguration("publish_optical_tf").perform(context).strip().lower()
    if optical == "auto":
        optical = "true" if hw == "mock_hardware" else "false"
    if optical not in ("true", "false"):
        raise RuntimeError(f"publish_optical_tf must be auto|true|false, got '{optical}'")
    if optical == "true":
        # ROS optical convention relative to the body frame camera_link (x forward, y left, z up):
        # optical z forward, x right, y down = rpy (-pi/2, 0, -pi/2). Translation 0 (the real colour
        # sensor offset of ~17 mm does not matter for the pole direction).
        oqx, oqy, oqz, oqw = rpy_to_quat(-math.pi / 2, 0.0, -math.pi / 2)
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="camera_optical_static_tf",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--qx", repr(oqx), "--qy", repr(oqy), "--qz", repr(oqz), "--qw", repr(oqw),
                       "--frame-id", "camera_link", "--child-frame-id", "camera_color_optical_frame"],
            output="both"))
    return actions


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    return LaunchDescription([
        DeclareLaunchArgument(
            "hardware_protocol", default_value="mock_hardware", choices=["mock_hardware", "rebel"],
            description="mock_hardware (default) or rebel (real CRI drivers; never without operator approval)"),
        DeclareLaunchArgument("left_robot_ip", default_value="192.168.3.11",
                              description="CRI controller of arm_left (rebel only)"),
        DeclareLaunchArgument("right_robot_ip", default_value="192.168.3.12",
                              description="CRI controller of arm_right (rebel only); must differ from left_robot_ip"),
        DeclareLaunchArgument(
            "left_jog_velocity_scale", default_value="1.65",
            description="arm_left joint jog scale: jog [%] = scale * velocity [deg/s] (rebel only). 1.65 since "
                        "2026-10-31 (real left arm 0.600 deg/s per jog %, exact 1.667); controllers.yaml gains assume it"),
        DeclareLaunchArgument(
            "right_jog_velocity_scale", default_value="1.65",
            description="arm_right joint jog scale (rebel only), see left_jog_velocity_scale"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value=os.path.join(share, "config", "dual_arm_calibration.yaml"),
            description="dual_arm_calibration YAML: arm_right_base and camera_link poses in world"),
        DeclareLaunchArgument(
            "activate_controllers", default_value="auto", choices=["auto", "true", "false"],
            description="auto: active in mock, loaded INACTIVE with rebel (activate deliberately)"),
        DeclareLaunchArgument("publish_camera_tf", default_value="true",
                              description="publish static TF world -> camera_link from the calibration YAML"),
        DeclareLaunchArgument(
            "publish_optical_tf", default_value="auto", choices=["auto", "true", "false"],
            description="static TF camera_link -> camera_color_optical_frame: auto = only with "
                        "mock_hardware (the real camera driver publishes it)"),
        DeclareLaunchArgument(
            "mock_move_joint", default_value="true", choices=["true", "false"],
            description="mock_hardware only: serve the robot-side joint move (/igus_rebel_<side>/move_joint, hardware_state, "
                        "robot_joint_states) with igus_rebel's mock server and spawn the two INACTIVE mock forward "
                        "controllers. Ignored (never started) with hardware_protocol:=rebel."),
        OpaqueFunction(function=launch_setup),
    ])
