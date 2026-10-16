"""One mocap stream drives BOTH arms of the dual-arm table cell, each arm independently.

Starts, per selected arm: a MoveIt Servo (/mocap/arm_<side>/servo, built exactly like dual_teleop's, collision
checking on) and a mocap_follower_<side> node; plus one planning-scene relay (/mocap/servo_scene) and a one-shot
start_servo caller. Needs the dual stack already running (dual_cut_system.launch.py, or at least
dual_table.launch.py + the dual move_group). Do NOT run dual_teleop at the same time: the followers detect its
Servo on the controller command topic and hold.

    ros2 launch rebel_mocap_follower mocap_follower_dual.launch.py hardware_protocol:=mock_hardware

Servo speed (launch argument servo_relay, default true): Servo 2.5.10 anchors each command on the measured joint
state and the JTC restarts each stamp-0 trajectory from its current state, which realises only ~1/3 of the commanded
speed. With servo_relay:=true each Servo runs a pass-through smoothing plugin and publishes to
/mocap/arm_<side>/servo/trajectory_out; a rebel_servo_relay node (/mocap/arm_<side>/servo/trajectory_relay)
integrates Servo's velocity into its own reference and publishes the JTC trajectories. servo_relay:=false is the old
direct Servo -> JTC wiring. See the README.

Servo's SCENE collision slow-down distance (camera pole / head, carton) is lowered from dual_teleop's 0.05 m to
0.025 m for these Servo nodes only (launch argument scene_collision_proximity_threshold, see the README).
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_param_builder import load_yaml
from launch_ros.actions import Node

IS_HUMBLE = os.environ.get("ROS_DISTRO") == "humble"

# Upper bound of the scene_collision_proximity_threshold argument = dual_teleop's servo_arm.yaml value.
MAX_SCENE_COLLISION_PROXIMITY_THRESHOLD = 0.05
# rebel_servo_relay's pass-through online_signal_smoothing plugin (Servo's smoothing_filter_plugin_name).
SERVO_PASS_THROUGH_PLUGIN = "rebel_servo_relay::PassThroughFilterPlugin"

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
    hw = LaunchConfiguration("hardware_protocol").perform(context)
    calibration_file = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    arms = LaunchConfiguration("arms").perform(context)
    sides = ("left", "right") if arms == "both" else (arms,)
    params_file = LaunchConfiguration("params_file").perform(context)
    servo_relay_arg = LaunchConfiguration("servo_relay").perform(context)
    if servo_relay_arg.strip().lower() not in ("true", "false"):
        raise RuntimeError(f"servo_relay:={servo_relay_arg!r} is invalid: it must be 'true' or 'false'. "
                           "Refusing to start instead of guessing.")
    servo_relay = servo_relay_arg.strip().lower() == "true"
    relay_max_lead, relay_lead_time = relay_lead_args(context)
    if servo_relay and LaunchConfiguration("start_servo").perform(context).strip().lower() != "true":
        raise RuntimeError(
            "servo_relay:=true needs start_servo:=true: the relay only works with the Servo nodes this launch starts "
            "(their output topic is rewired). Use servo_relay:=false together with start_servo:=false. "
            "Refusing to start.")
    scene_threshold_arg = LaunchConfiguration("scene_collision_proximity_threshold").perform(context)
    try:
        scene_threshold = float(scene_threshold_arg)
    except ValueError:
        scene_threshold = float("nan")
    if not (0.0 < scene_threshold <= MAX_SCENE_COLLISION_PROXIMITY_THRESHOLD):  # also rejects NaN
        raise RuntimeError(
            f"scene_collision_proximity_threshold:={scene_threshold_arg!r} is invalid: it must be a number in "
            f"(0, {MAX_SCENE_COLLISION_PROXIMITY_THRESHOLD}] metres (0.05 is dual_teleop's value). "
            "Refusing to start instead of clamping.")
    desc_share = get_package_share_directory("rebel_dual_table_description")
    mc_share = get_package_share_directory("rebel_dual_table_moveit_config")
    teleop_share = get_package_share_directory("dual_teleop")

    actions = []
    if LaunchConfiguration("start_servo").perform(context).lower() == "true":
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
        servo_libexec = os.path.join(get_package_prefix("moveit_servo"), "lib", "moveit_servo")
        servo_executable = "servo_node" if os.path.exists(os.path.join(servo_libexec, "servo_node")) \
            else "servo_node_main"
        base = load_yaml(Path(os.path.join(teleop_share, "config", "servo_arm.yaml")))
        for side in sides:
            servo = dict(base)
            servo.update({
                "move_group_name": f"arm_{side}",
                "ee_frame_name": f"arm_{side}_xeg32_tool_tip",
                "planning_frame": "world",
                "robot_link_command_frame": "world",
                "command_out_topic": f"/arm_{side}_trajectory_controller/joint_trajectory",
            })
            # Mocap-only override of servo_arm.yaml (dual_teleop keeps 0.05). Servo's slow-down is exponential
            # (scale 1 at the threshold, 0.001 at contact), so a jog toward a scene object stops at about
            # 0.44 x threshold: ~11 mm at 0.025 instead of ~22 mm at 0.05. Trade-off: less stopping margin to the
            # camera pole / head / carton, in exchange for the left arm (wrist camera 32 mm from the camera_head
            # box at home) no longer crawling from home. self_collision_proximity_threshold (other arm, table)
            # is deliberately left at servo_arm.yaml's 0.07.
            servo["scene_collision_proximity_threshold"] = scene_threshold
            relay_in = f"/mocap/arm_{side}/servo/trajectory_out"
            jtc_topic = servo["command_out_topic"]
            if servo_relay:
                # Servo -> relay -> JTC. The pass-through plugin keeps Servo's output velocity equal to the
                # commanded one (the default Butterworth filter would remove ~25 % of it).
                servo["command_out_topic"] = relay_in
                servo["smoothing_filter_plugin_name"] = SERVO_PASS_THROUGH_PLUGIN
            if IS_HUMBLE:
                # Same translation as dual_teleop.launch.py / move_group.launch.py.
                servo["joint_limit_margin"] = servo.pop("joint_limit_margins", [0.1])[0]
                servo.pop("use_smoothing", None)
                servo.pop("check_octomap_collisions", None)
                servo["use_gazebo"] = False
            actions.append(Node(
                package="moveit_servo", executable=servo_executable,
                name="servo_node", namespace=f"/mocap/arm_{side}/servo",
                arguments=["--ros-args", "-r", "planning_scene:=/mocap/servo_scene",
                           "-r", "get_planning_scene:=/get_planning_scene"],
                parameters=[{"use_sim_time": use_sim_time}, {"moveit_servo": servo},
                            {"planning_group_name": f"arm_{side}"}, moveit_args],
                output="screen"))
            if servo_relay:
                # Inside the Servo namespace on purpose: the follower's exclusivity gate treats every node under
                # own_servo_node (/mocap/arm_<side>/servo) as our own publisher on the JTC topic.
                actions.append(Node(
                    package="rebel_servo_relay", executable="servo_trajectory_relay",
                    name="trajectory_relay", namespace=f"/mocap/arm_{side}/servo",
                    parameters=[{
                        "input_topic": relay_in, "output_topic": jtc_topic, "joint_states_topic": "/joint_states",
                        "publish_period_s": float(servo["publish_period"]),
                        "max_lead_rad": relay_max_lead, "lead_time_s": relay_lead_time,
                        "robot_description_node": "/robot_state_publisher",
                    }],
                    output="screen"))
        actions.append(Node(package="rebel_mocap_follower", executable="mocap_servo_scene_relay",
                            name="servo_scene_relay", namespace="/mocap",
                            parameters=[{"use_sim_time": use_sim_time}], output="screen"))
        actions.append(TimerAction(period=3.0, actions=[Node(
            package="rebel_mocap_follower", executable="mocap_servo_starter", name="mocap_servo_starter",
            arguments=[f"/mocap/arm_{side}/servo" for side in sides], output="screen")]))

    for side in sides:
        actions.append(Node(
            package="rebel_mocap_follower", executable="mocap_follower", name=f"mocap_follower_{side}",
            parameters=[params_file, {"use_sim_time": use_sim_time},
                        # exactly this one own publisher on the JTC topic (see interlocks.exclusivity_gate)
                        *([{"own_publisher_node": f"/mocap/arm_{side}/servo/trajectory_relay"}] if servo_relay else [])],
            output="screen"))
    return actions


def generate_launch_description():
    share = get_package_share_directory("rebel_mocap_follower")
    desc_share = get_package_share_directory("rebel_dual_table_description")
    return LaunchDescription([
        DeclareLaunchArgument("arms", default_value="both", choices=["both", "left", "right"]),
        DeclareLaunchArgument("start_servo", default_value="true",
                              description="start the two mocap Servo nodes, scene relay and start_servo caller"),
        DeclareLaunchArgument("hardware_protocol", choices=["mock_hardware", "rebel"],
                              description="only used to build the Servo robot model; must match the dual stack"),
        DeclareLaunchArgument("calibration_file",
                              default_value=os.path.join(desc_share, "config", "dual_arm_calibration.yaml"),
                              description="must be the same file as the dual stack was started with"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("scene_collision_proximity_threshold", default_value="0.025",
                              description="Servo scene-collision slow-down distance in m, mocap Servo nodes only; "
                                          "must be in (0, 0.05] (0.05 = dual_teleop's value)"),
        DeclareLaunchArgument("servo_relay", default_value="true", choices=["true", "false"],
                              description="true: pass-through Servo smoothing + rebel_servo_relay between Servo and "
                                          "the JTC (arm moves at the commanded speed); false: old direct wiring "
                                          "(~1/3 of the commanded speed)"),
        DeclareLaunchArgument("relay_max_lead_rad", default_value="0.05",
                              description="servo_relay: bound on |reference - measured| in rad, in (0, 0.15]"),
        DeclareLaunchArgument("relay_lead_time_s", default_value="0.2",
                              description="servo_relay: lead per rad/s of joint velocity in s, in [0, 1]"),
        DeclareLaunchArgument("params_file",
                              default_value=os.path.join(share, "config", "mocap_follower_dual.yaml")),
        OpaqueFunction(function=_setup),
    ])
