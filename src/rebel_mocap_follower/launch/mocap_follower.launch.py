"""Start MoveIt Servo (optional) and the mocap follower.

start_servo:=true  starts a Servo node with the same servo.yaml/robot model as
                   move_group.launch.py's run_teleop, plus start_servo.py. Do NOT also
                   launch move_group.launch.py with run_teleop:=true (two Servo nodes).
start_servo:=false Servo is already running (e.g. run_teleop:=true on move_group.launch.py).
Requires rebel.launch.py (ros2_control) and move_group.launch.py to be running already.

servo_relay:=true (default) with start_servo:=true: Servo runs the pass-through smoothing plugin and publishes to
/servo_trajectory_out; rebel_servo_relay (/servo_trajectory_relay) turns that into the JTC trajectories, so the arm
moves at the commanded speed instead of ~1/3 of it. servo_relay:=false is the old direct Servo -> JTC wiring.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_param_builder import load_yaml
from launch_ros.actions import Node

MOVEIT_PKG = "igus_rebel_moveit_config"
SERVO_PASS_THROUGH_PLUGIN = "rebel_servo_relay::PassThroughFilterPlugin"
RELAY_INPUT_TOPIC = "/servo_trajectory_out"
RELAY_NODE = "/servo_trajectory_relay"

# Relay lead bound (rebel_servo_relay): |reference - measured| <= min(max_lead, lead_time x |v| + 0.005 rad). The cap stays
# well below the JTC state tolerance of 0.25 rad (lead + v x horizon + tracking lag must fit into it).
MAX_RELAY_MAX_LEAD_RAD = 0.15
MAX_RELAY_LEAD_TIME_S = 1.0


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


def _setup(context):
    hardware_protocol = LaunchConfiguration("hardware_protocol").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    moveit_share = Path(get_package_share_directory(MOVEIT_PKG))
    desc_share = Path(get_package_share_directory("igus_rebel_description"))
    own_share = Path(get_package_share_directory("rebel_mocap_follower"))

    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        str(desc_share / "urdf" / "igus_rebel_xeg32_wrapper.urdf.xacro"),
        " hardware_protocol:=", hardware_protocol,
        " tool_variant:=", LaunchConfiguration("tool_variant"),
    ]).perform(context)
    srdf = (moveit_share / "config" / "igus_rebel2.srdf").read_text()
    moveit_args = {
        "robot_description": robot_description,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": load_yaml(moveit_share / "config" / "kinematics.yaml"),
        "robot_description_planning": load_yaml(moveit_share / "config" / "joint_limits.yaml"),
    }

    servo_relay_arg = LaunchConfiguration("servo_relay").perform(context)
    if servo_relay_arg.strip().lower() not in ("true", "false"):
        raise RuntimeError(f"servo_relay:={servo_relay_arg!r} is invalid: it must be 'true' or 'false'. "
                           "Refusing to start instead of guessing.")
    servo_relay = servo_relay_arg.strip().lower() == "true"
    relay_max_lead, relay_lead_time = relay_lead_args(context)
    start_servo = LaunchConfiguration("start_servo").perform(context).strip().lower() == "true"
    if servo_relay and not start_servo:
        raise RuntimeError(
            "servo_relay:=true needs start_servo:=true: the relay only works with the Servo node this launch starts "
            "(its output topic is rewired). Use servo_relay:=false together with start_servo:=false. "
            "Refusing to start.")
    use_relay = servo_relay

    servo_context = load_yaml(moveit_share / "config" / "servo.yaml")
    jtc_topic = "/" + str(servo_context["command_out_topic"]).lstrip("/")
    if use_relay:
        servo_context["command_out_topic"] = RELAY_INPUT_TOPIC
        servo_context["smoothing_filter_plugin_name"] = SERVO_PASS_THROUGH_PLUGIN
    if os.environ.get("ROS_DISTRO") == "humble":
        # Same Humble translation as move_group.launch.py.
        servo_context["joint_limit_margin"] = servo_context.pop("joint_limit_margins", [0.1])[0]
        servo_context.pop("use_smoothing", None)
        servo_context.pop("check_octomap_collisions", None)
        servo_context["planning_frame"] = "base_link"
        servo_context["ee_frame_name"] = "xeg32_tool_tip"
        servo_context["robot_link_command_frame"] = "base_link"
        servo_context["use_gazebo"] = hardware_protocol == "gazebo"
    servo_dir = os.path.join(get_package_prefix("moveit_servo"), "lib", "moveit_servo")
    servo_exe = "servo_node" if os.path.exists(os.path.join(servo_dir, "servo_node")) \
        else "servo_node_main"

    servo = Node(
        condition=IfCondition(LaunchConfiguration("start_servo")),
        package="moveit_servo", executable=servo_exe,
        parameters=[{"use_sim_time": use_sim_time}, {"moveit_servo": servo_context},
                    {"planning_group_name": "igus_rebel_arm"}, moveit_args],
        output="screen")
    starter = Node(
        condition=IfCondition(LaunchConfiguration("start_servo")),
        package=MOVEIT_PKG, executable="start_servo.py", output="screen")
    follower = Node(
        package="rebel_mocap_follower", executable="mocap_follower", name="mocap_follower",
        parameters=[LaunchConfiguration("params_file").perform(context),
                    {"use_sim_time": use_sim_time},
                    # servo_relay:=true: same JTC exclusivity gate as the dual launch. The only allowed publisher on the
                    # JTC topic is the relay; own_servo_node is the relay's own path (single arm: no shared Servo
                    # namespace), so Servo (/servo_node) still wired to the JTC, a second relay, or any other
                    # publisher holds the follower, and so does a missing relay.
                    *([{"exclusive_command_topic": jtc_topic, "own_servo_node": RELAY_NODE,
                        "own_publisher_node": RELAY_NODE}] if use_relay else [])],
        output="screen")
    actions = [servo, starter, follower]
    if use_relay:
        actions.append(Node(
            package="rebel_servo_relay", executable="servo_trajectory_relay", name="servo_trajectory_relay",
            parameters=[{"input_topic": RELAY_INPUT_TOPIC, "output_topic": jtc_topic,
                         "joint_states_topic": "/joint_states",
                         "publish_period_s": float(servo_context["publish_period"]),
                         "max_lead_rad": relay_max_lead, "lead_time_s": relay_lead_time,
                         "robot_description_node": "/robot_state_publisher"}],
            output="screen"))
    return actions


def generate_launch_description():
    share = get_package_share_directory("rebel_mocap_follower")
    return LaunchDescription([
        DeclareLaunchArgument("hardware_protocol", default_value="rebel",
                              choices=["mock_hardware", "gazebo", "rebel"],
                              description="Only used to build the Servo robot model"),
        DeclareLaunchArgument("tool_variant", default_value="knife_c",
                              choices=["knife_c", "xeg32_knife_v1"]),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("start_servo", default_value="true"),
        DeclareLaunchArgument("servo_relay", default_value="true", choices=["true", "false"],
                              description="true: pass-through Servo smoothing + rebel_servo_relay between Servo and "
                                          "the JTC (arm moves at the commanded speed); false: old direct wiring "
                                          "(~1/3 of the commanded speed)"),
        DeclareLaunchArgument("relay_max_lead_rad", default_value="0.05",
                              description="servo_relay: bound on |reference - measured| in rad, in (0, 0.15]"),
        DeclareLaunchArgument("relay_lead_time_s", default_value="0.2",
                              description="servo_relay: lead per rad/s of joint velocity in s, in [0, 1]"),
        DeclareLaunchArgument("params_file",
                              default_value=os.path.join(share, "config", "mocap_follower.yaml")),
        OpaqueFunction(function=_setup),
    ])
