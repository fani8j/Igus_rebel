"""Per-arm jog/teleop of the dual-arm cell (one MoveIt Servo per arm, an arm selector, optional keyboard/gamepad).

Run next to the dual stack (all in the same ROS domain):
    rebel_dual_table_description dual_table.launch.py      (ros2_control, TF)
    rebel_dual_table_moveit_config move_group.launch.py    (robot model + planning scene: carton, camera pole, table)
Optional: dual_cut_coordinator (its CoordinatorStatus interlocks teleop while a phase executes).

    ros2 launch dual_teleop dual_teleop.launch.py arm:=left keyboard:=true

Teleop starts PAUSED: nothing moves until it is enabled (console key 'e', gamepad Back, or
`ros2 service call /dual_teleop/enable std_srvs/srv/Trigger`) or start_enabled:=true is given.

keyboard:=true opens the unmodified single-arm keyboard node in its own terminal (keyboard_prefix, default
gnome-terminal) because it needs a TTY; without a display run it by hand:
    ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard --ros-args -r __ns:=/dual_teleop/keyboard \
      -r delta_twist_cmds:=/dual_teleop/in/delta_twist_cmds -r delta_joint_cmds:=/dual_teleop/in/delta_joint_cmds \
      -r servo_node/start_servo:=/dual_teleop/servo_start_stub -r /status:=/dual_teleop/servo_alive
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PythonExpression
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE = "dual_teleop"
IS_HUMBLE = os.environ.get("ROS_DISTRO") == "humble"

# The unmodified single-arm teleop nodes publish with hard-coded names; remap them into the selector.
TELEOP_REMAPS = [
    ("delta_twist_cmds", "/dual_teleop/in/delta_twist_cmds"),
    ("delta_joint_cmds", "/dual_teleop/in/delta_joint_cmds"),
    ("servo_node/start_servo", "/dual_teleop/servo_start_stub"),
    ("servo_node/switch_command_type", "/dual_teleop/servo_start_stub"),
    ("/status", "/dual_teleop/servo_alive"),
]


def launch_setup(context):
    share = get_package_share_directory(PACKAGE)
    desc_share = get_package_share_directory("rebel_dual_table_description")
    mc_share = get_package_share_directory("rebel_dual_table_moveit_config")
    hw = LaunchConfiguration("hardware_protocol").perform(context)
    calibration_file = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        os.path.join(desc_share, "urdf", "rebel_dual_table.urdf.xacro"),
        " hardware_protocol:=", hw,
        " calibration_file:=", calibration_file,
    ]).perform(context)
    moveit_args = {
        "robot_description": robot_description,
        "robot_description_semantic": Path(os.path.join(mc_share, "config", "rebel_dual_table.srdf")).read_text(),
        "robot_description_kinematics": load_yaml(Path(os.path.join(mc_share, "config", "kinematics.yaml"))),
        "robot_description_planning": load_yaml(Path(os.path.join(mc_share, "config", "joint_limits.yaml"))),
    }

    actions = []
    servo_libexec = os.path.join(get_package_prefix("moveit_servo"), "lib", "moveit_servo")
    servo_executable = "servo_node" if os.path.exists(os.path.join(servo_libexec, "servo_node")) else "servo_node_main"
    base = load_yaml(Path(os.path.join(share, "config", "servo_arm.yaml")))
    for side in ("left", "right"):
        servo = dict(base)
        servo.update({
            "move_group_name": f"arm_{side}",
            "ee_frame_name": f"arm_{side}_xeg32_tool_tip",
            "planning_frame": "world",
            "robot_link_command_frame": "world",
            "command_out_topic": f"/arm_{side}_trajectory_controller/joint_trajectory",
        })
        if IS_HUMBLE:
            # Same translation as igus_rebel_moveit_config/launch/move_group.launch.py.
            servo["joint_limit_margin"] = servo.pop("joint_limit_margins", [0.1])[0]
            servo.pop("use_smoothing", None)
            servo.pop("check_octomap_collisions", None)
            servo["use_gazebo"] = False
        actions.append(Node(
            condition=IfCondition(LaunchConfiguration("run_servo")),
            package="moveit_servo", executable=servo_executable,
            name="servo_node", namespace=f"/arm_{side}/servo",
            # Servo's planning scene monitor uses namespaced relative names by default. Its scene comes from the
            # selector (/dual_teleop/servo_scene_<side> = dual /move_group scene polled at 2 Hz, same-arm pairs allowed);
            # the initial state is requested from /move_group. Global (not node-prefixed) rules: the monitor lives
            # in a private sub-node of servo_node. (The camera pole is added through apply_planning_scene, which
            # never appears on the /planning_scene diff topic, hence the poll.)
            arguments=["--ros-args", "-r", f"planning_scene:=/dual_teleop/servo_scene_{side}",
                       "-r", "get_planning_scene:=/get_planning_scene"],
            parameters=[{"use_sim_time": use_sim_time}, {"moveit_servo": servo},
                        {"planning_group_name": f"arm_{side}"}, moveit_args],
            output="screen"))

    actions.append(Node(
        package=PACKAGE, executable="teleop_selector", name="selector", namespace="/dual_teleop",
        parameters=[{
            "default_arm": LaunchConfiguration("arm"),
            "start_enabled": LaunchConfiguration("start_enabled"),
            "speed_scale": LaunchConfiguration("speed_scale"),
            "base_command_frame": LaunchConfiguration("base_command_frame"),
            "coordinator_status_topic": LaunchConfiguration("coordinator_status_topic"),
            "touch_speed_scale": LaunchConfiguration("touch_speed_scale"),
            "touch_mode_timeout_s": LaunchConfiguration("touch_mode_timeout_s"),
            "joy_buttons": LaunchConfiguration("gamepad"),
            "joy_topic": "/dual_teleop/gamepad/joy",
        }],
        output="screen"))

    remaps = [(a, b) for a, b in TELEOP_REMAPS]
    keyboard_prefix = LaunchConfiguration("keyboard_prefix").perform(context)
    actions.append(Node(
        condition=IfCondition(LaunchConfiguration("keyboard")),
        package="igus_rebel_moveit_config", executable="rebel_servo_teleop_keyboard",
        namespace="/dual_teleop/keyboard", remappings=remaps, prefix=keyboard_prefix, output="screen"))
    actions.append(Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("keyboard"), "' == 'true' or '", LaunchConfiguration("console"), "' == 'true'"])),
        package=PACKAGE, executable="teleop_console", prefix=keyboard_prefix, output="screen"))
    actions.append(Node(
        condition=IfCondition(LaunchConfiguration("gamepad")),
        package="joy", executable="joy_node", namespace="/dual_teleop/gamepad",
        parameters=[{"use_sim_time": use_sim_time,
                     "device_id": ParameterValue(LaunchConfiguration("joy_device_id"), value_type=int)}],
        output="screen"))
    actions.append(Node(
        condition=IfCondition(LaunchConfiguration("gamepad")),
        package="igus_rebel_moveit_config", executable="rebel_servo_teleop_gamepad",
        namespace="/dual_teleop/gamepad", remappings=remaps,
        parameters=[{"use_sim_time": use_sim_time},
                    os.path.join(get_package_share_directory("igus_rebel_moveit_config"), "config", "gamepad.yaml")],
        output="screen"))
    return actions


def generate_launch_description():
    desc_share = get_package_share_directory("rebel_dual_table_description")
    return LaunchDescription([
        DeclareLaunchArgument("arm", default_value="left", choices=["left", "right"],
                              description="arm selected at start (switch at runtime: console l/r, gamepad LB/RB)"),
        DeclareLaunchArgument("keyboard", default_value="false",
                              description="start the single-arm keyboard teleop node (+ console) in a new terminal"),
        DeclareLaunchArgument("gamepad", default_value="false", description="start joy_node + gamepad teleop node"),
        DeclareLaunchArgument("joy_device_id", default_value="0", description="SDL device id of the gamepad (joy_node)"),
        DeclareLaunchArgument("console", default_value="false", description="start the operator screen / arm keys"),
        DeclareLaunchArgument("keyboard_prefix", default_value="gnome-terminal --wait --",
                              description="launch prefix giving the keyboard/console node a TTY"),
        DeclareLaunchArgument("start_enabled", default_value="false",
                              description="true: teleop is enabled at start; default: PAUSED until enabled"),
        DeclareLaunchArgument("speed_scale", default_value="0.5",
                              description="multiplier on every keyboard/gamepad command, (0, 1]"),
        DeclareLaunchArgument("touch_speed_scale", default_value="-1.0",
                              description="speed scale while TOUCH MODE is on; -1 = speed_scale / 3; never above speed_scale"),
        DeclareLaunchArgument("touch_mode_timeout_s", default_value="120.0",
                              description="touch mode turns itself off after this long (0 = never)"),
        DeclareLaunchArgument("base_command_frame", default_value="world", choices=["world", "arm_base"],
                              description="frame 'base_link' (keyboard 'w') is mapped to: world or the arm's own base"),
        DeclareLaunchArgument("coordinator_status_topic", default_value="/dual_cut_coordinator/status"),
        DeclareLaunchArgument("run_servo", default_value="true", description="start the two Servo nodes"),
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware", choices=["mock_hardware", "rebel"],
                              description="only used to generate the model; must match the dual stack"),
        DeclareLaunchArgument("calibration_file",
                              default_value=os.path.join(desc_share, "config", "dual_arm_calibration.yaml"),
                              description="must be the same file as dual_table.launch.py / move_group.launch.py"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
