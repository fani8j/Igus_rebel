"""MoveIt - medicine grasp: plan-only MoveIt + ros2_control bring-up for the dual-arm ReBeL rig with the suction tool (left) and the LEHF20K2-48 gripper (right)
(mock hardware or Isaac topics).  Humble.  ADAPTED from rebel_dual_arm_stand_moveit_config/launch/dual_arm.launch.py (cutting rig); same arguments except
`add_cell_scene` -> `add_medicine_shelf` (+ `pull_out`), and no `tool_config` (the tools are fixed).

DERIVED (structure, argument names, node set) FROM compal_robot igus_rebel_moveit_config/launch/dual_arm_rig_isaac.launch.py @ bb76327
(MIT, Copyright (c) 2026 James, see ../NOTICE); rewritten for Humble following igus_rebel_moveit_config/launch/move_group.launch.py, with the
controller spawners sequenced like igus_rebel/launch/rebel.launch.py (joint_state_broadcaster first, each next one after the previous spawner
has exited).  There is NO real-hardware option: hardware_protocol is mock_hardware | isaac_topic.

Starts: robot_state_publisher, ros2_control_node (+ spawners: joint_state_broadcaster, gantry_, arm_left_, arm_right_trajectory_controller, right_gripper_controller),
move_group (OMPL + Pilz pipelines; allow_trajectory_execution DEFAULT false = plan-only), optional rviz2, optional add_medicine_shelf.py.

  export ROS_DOMAIN_ID=<unused> ROS_LOCALHOST_ONLY=1        # NEVER the real robot's domain
  ros2 launch rebel_medicine_grasp_moveit_config medicine_grasp.launch.py hardware_protocol:=mock_hardware
  ros2 launch rebel_medicine_grasp_moveit_config medicine_grasp.launch.py hardware_protocol:=isaac_topic use_sim_time:=true use_rviz:=true
"""
import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DESC_PKG = "rebel_medicine_grasp_description"
PKG = "rebel_medicine_grasp_moveit_config"
CONTROLLERS = ("gantry_trajectory_controller", "arm_left_trajectory_controller", "arm_right_trajectory_controller", "right_gripper_controller")


def opaque_func(context, *args, **kwargs):
    cfg = Path(get_package_share_directory(PKG)) / "config"
    desc = Path(get_package_share_directory(DESC_PKG))
    hardware_protocol = LaunchConfiguration("hardware_protocol").perform(context)
    if hardware_protocol not in ("mock_hardware", "isaac_topic"):
        raise RuntimeError("hardware_protocol must be mock_hardware or isaac_topic (no real-hardware option), got %r" % hardware_protocol)
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    allow_exec = LaunchConfiguration("allow_trajectory_execution").perform(context).lower() in ("true", "1")

    # subprocess with an argument list (not launch's Command, which shlex-splits one string): install paths may contain spaces (external drive scratch workspace)
    xacro_cmd = ["xacro", str(desc / "urdf" / "rebel_medicine_grasp.urdf.xacro"),
                 "hardware_protocol:=" + hardware_protocol,
                 "initial_positions_file:=" + str(cfg / "initial_positions.yaml")]
    robot_description = {"robot_description": subprocess.run(xacro_cmd, check=True, capture_output=True, text=True).stdout}
    robot_description_semantic = {"robot_description_semantic": (cfg / "rebel_medicine_grasp.srdf").read_text()}

    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="both",
               parameters=[robot_description, use_sim_time])
    control = Node(package="controller_manager", executable="ros2_control_node", output="both",
                   parameters=[robot_description, str(cfg / "ros2_controllers.yaml"), use_sim_time])

    def spawner(name):
        return Node(package="controller_manager", executable="spawner", output="both",
                    arguments=[name, "--controller-manager", "/controller_manager"])

    # Concurrent spawners calling load_controller fail at random on hosts without RT scheduling: run them strictly one after the other.
    chain = [spawner("joint_state_broadcaster")] + [spawner(n) for n in CONTROLLERS]
    handlers = [RegisterEventHandler(OnProcessExit(target_action=chain[i], on_exit=[chain[i + 1]])) for i in range(len(chain) - 1)]

    # ---- move_group (Humble parameter layout) ----
    kinematics = load_yaml(cfg / "kinematics.yaml")
    joint_limits = load_yaml(cfg / "joint_limits.yaml")
    joint_limits.update(load_yaml(cfg / "pilz_cartesian_limits.yaml"))          # cartesian_limits for Pilz LIN/CIRC
    ompl = load_yaml(cfg / "ompl_planning.yaml")
    pilz = load_yaml(cfg / "pilz_planning.yaml")
    controllers = load_yaml(cfg / "moveit_controllers.yaml")
    trajectory_execution = controllers.pop("trajectory_execution")
    moveit_args = {
        **robot_description,
        **robot_description_semantic,
        "robot_description_kinematics": kinematics,
        "robot_description_planning": joint_limits,
        "planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
        "pilz_industrial_motion_planner": pilz,
        "trajectory_execution": trajectory_execution,
        "allow_trajectory_execution": allow_exec,        # plan-only unless explicitly enabled
        "moveit_manage_controllers": False,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_planning_scene_hz": 2.0,
    }
    if allow_exec:
        moveit_args["moveit_simple_controller_manager"] = controllers
        moveit_args["moveit_controller_manager"] = "moveit_simple_controller_manager/MoveItSimpleControllerManager"
    # else: plan-only.  Besides allow_trajectory_execution:=false, move_group is given NO controller manager / controller list, so even a stray
    # /execute_trajectory goal could not reach a controller (belt and braces; Humble's /execute_trajectory action server exists regardless).
    move_group = Node(package="moveit_ros_move_group", executable="move_group", output="screen",
                      parameters=[use_sim_time, moveit_args], condition=IfCondition(LaunchConfiguration("run_move_group")))

    scene = Node(package=PKG, executable="add_medicine_shelf.py", name="add_medicine_shelf", output="screen",
                 parameters=[use_sim_time, {"scene_file": LaunchConfiguration("scene_file"),
                                            "pull_out": ParameterValue(LaunchConfiguration("pull_out"), value_type=float)}],
                 condition=IfCondition(LaunchConfiguration("add_medicine_shelf")))
    banner = LogInfo(msg="MoveIt - medicine grasp: hardware_protocol=%s, allow_trajectory_execution=%s (%s); left = suction tool v5 (TCP arm_left_suction_tcp), "
                         "right = LEHF20K2-48 gripper (TCP arm_right_gripper_tcp)" % (hardware_protocol, allow_exec, "EXECUTION ENABLED" if allow_exec else "plan-only"))

    rviz = Node(package="rviz2", executable="rviz2", output="log", arguments=["-d", LaunchConfiguration("rviz_config")],
                parameters=[robot_description, robot_description_semantic, {"robot_description_kinematics": kinematics},
                            {"robot_description_planning": joint_limits}, use_sim_time],
                condition=IfCondition(LaunchConfiguration("use_rviz")))

    return [banner, rsp, control, chain[0], *handlers, move_group, scene, rviz]


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory(PKG), "config")
    return LaunchDescription([
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware", choices=["mock_hardware", "isaac_topic"],
                              description="mock_components/GenericSystem, or topic_based_ros2_control talking to Isaac Sim "
                                          "(/dual_arm/isaac_joint_commands, /dual_arm/isaac_joint_states). No real-hardware option."),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("allow_trajectory_execution", default_value="false",
                              description="false = plan-only: move_group loads no controller manager and cannot execute. "
                                          "true also needs the controllers (mock or Isaac) running."),
        DeclareLaunchArgument("run_move_group", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(cfg, "rebel_medicine_grasp.rviz")),
        DeclareLaunchArgument("add_medicine_shelf", default_value="true",
                              description="add the shelf, plinth and packed cartons (config/medicine_shelf_scene.yaml) to the planning scene"),
        DeclareLaunchArgument("scene_file", default_value=os.path.join(cfg, "medicine_shelf_scene.yaml")),
        DeclareLaunchArgument("pull_out", default_value="0.0",
                              description="pull the target carton this far (m) out of the shelf toward the robot (0.100 = the handoff state)"),
        OpaqueFunction(function=opaque_func),
    ])
