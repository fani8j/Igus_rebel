"""Per-arm jog/teleop of the dual-arm cell (one MoveIt Servo per arm, an arm selector, optional keyboard/gamepad).

Run next to the dual stack (all in the same ROS domain):
    rebel_dual_table_description dual_table.launch.py      (ros2_control, TF)
    rebel_dual_table_moveit_config move_group.launch.py    (robot model + planning scene: carton, camera pole, table)
Optional: dual_cut_coordinator (its CoordinatorStatus interlocks teleop while a phase executes).

    ros2 launch dual_teleop dual_teleop.launch.py arm:=left keyboard:=true

Servo speed (launch argument servo_relay, default true): Servo 2.5.10 anchors each command on the measured joint
state and the JTC restarts every stamp-0 trajectory from its current state, which realises only ~1/3 of the
commanded speed. With servo_relay:=true each Servo runs a pass-through smoothing plugin and publishes to
/arm_<side>/servo/trajectory_out; a rebel_servo_relay node (/arm_<side>/servo/trajectory_relay) integrates Servo's
velocity into its own reference and publishes the JTC trajectories. servo_relay:=false is the old direct
Servo -> JTC wiring. See the README.

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
# rebel_servo_relay's pass-through online_signal_smoothing plugin (Servo's smoothing_filter_plugin_name).
SERVO_PASS_THROUGH_PLUGIN = "rebel_servo_relay::PassThroughFilterPlugin"

# Relay lead bound (rebel_servo_relay): |reference - measured| <= min(max_lead, lead_time x |v| + 0.005 rad). The cap stays
# well below the JTC state tolerance of 0.25 rad (lead + v x horizon + tracking lag must fit into it).
MAX_RELAY_MAX_LEAD_RAD = 0.15
MAX_RELAY_LEAD_TIME_S = 1.0

TELEOP_REMAPS = [
    ("delta_twist_cmds", "/dual_teleop/in/delta_twist_cmds"),
    ("delta_joint_cmds", "/dual_teleop/in/delta_joint_cmds"),
    ("servo_node/start_servo", "/dual_teleop/servo_start_stub"),
    ("servo_node/switch_command_type", "/dual_teleop/servo_start_stub"),
    ("/status", "/dual_teleop/servo_alive"),
]


def relay_lead_args(context):
    """(max_lead_rad, lead_time_s) from the launch arguments; RuntimeError instead of clamping."""
    out = []
    for name, lo_excl, hi in (("relay_max_lead_rad", True, MAX_RELAY_MAX_LEAD_RAD),
                              ("relay_lead_time_s", False, MAX_RELAY_LEAD_TIME_S)):
        raw = LaunchConfiguration(name).perform(context)
        try:
            val = float(raw)
        except ValueError:
            val = float("nan")
        if not ((val > 0.0 if lo_excl else val >= 0.0) and val <= hi):  # also rejects NaN
            raise RuntimeError(f"{name}:={raw!r} is invalid: it must be a number in "
                               f"{'(0' if lo_excl else '[0'}, {hi}]. Refusing to start instead of clamping.")
        out.append(val)
    return out


def launch_setup(context):
    servo_relay_arg = LaunchConfiguration("servo_relay").perform(context)
    if servo_relay_arg.strip().lower() not in ("true", "false"):
        raise RuntimeError(f"servo_relay:={servo_relay_arg!r} is invalid: it must be 'true' or 'false'. "
                           "Refusing to start instead of guessing.")
    servo_relay = servo_relay_arg.strip().lower() == "true"
    relay_max_lead, relay_lead_time = relay_lead_args(context)
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
        jtc_topic = servo["command_out_topic"]
        relay_in = f"/arm_{side}/servo/trajectory_out"
        if servo_relay:
            # Servo -> relay -> JTC. The pass-through plugin keeps Servo's output velocity equal to the
            # commanded one (the default Butterworth filter would remove ~25 % of it).
            servo["command_out_topic"] = relay_in
            servo["smoothing_filter_plugin_name"] = SERVO_PASS_THROUGH_PLUGIN
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
        if servo_relay:
            actions.append(Node(
                condition=IfCondition(LaunchConfiguration("run_servo")),
                package="rebel_servo_relay", executable="servo_trajectory_relay",
                name="trajectory_relay", namespace=f"/arm_{side}/servo",
                parameters=[{
                    "input_topic": relay_in, "output_topic": jtc_topic, "joint_states_topic": "/joint_states",
                    "publish_period_s": float(servo["publish_period"]),
                    "max_lead_rad": relay_max_lead, "lead_time_s": relay_lead_time,
                    # dual_table.launch.py's robot_state_publisher holds the dual-table URDF (joint limits)
                    "robot_description_node": "/robot_state_publisher",
                }],
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
        DeclareLaunchArgument("servo_relay", default_value="true", choices=["true", "false"],
                              description="true: pass-through Servo smoothing + rebel_servo_relay between Servo and "
                                          "the JTC (arm moves at the commanded speed); false: old direct wiring "
                                          "(~1/3 of the commanded speed)"),
        DeclareLaunchArgument("relay_max_lead_rad", default_value="0.05",
                              description="servo_relay: bound on |reference - measured| in rad, in (0, 0.15]"),
        DeclareLaunchArgument("relay_lead_time_s", default_value="0.2",
                              description="servo_relay: lead per rad/s of joint velocity in s, in [0, 1]"),
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware", choices=["mock_hardware", "rebel"],
                              description="only used to generate the model; must match the dual stack"),
        DeclareLaunchArgument("calibration_file",
                              default_value=os.path.join(desc_share, "config", "dual_arm_calibration.yaml"),
                              description="must be the same file as dual_table.launch.py / move_group.launch.py"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
