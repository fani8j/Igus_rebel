"""MoveIt move_group (+ optional RViz) for the dual-arm test table.

Run next to rebel_dual_table_description dual_table.launch.py (which owns ros2_control and TF):

    ros2 launch rebel_dual_table_moveit_config move_group.launch.py hardware_protocol:=mock_hardware use_gui:=false

robot_description is generated from the same rebel_dual_table.urdf.xacro and calibration YAML
(argument calibration_file, default the description package's config) as dual_table.launch.py.

run_camera_pole (default true) starts the single-arm stack's igus_rebel_moveit_config
camera_pole_collision.py UNCHANGED, with base_frame:=world: it publishes the fixed external
camera_pole / camera_head boxes into the planning scene from the TF camera_link and
camera_color_optical_frame (world -> camera_link from the calibration YAML via dual_table.launch.py;
camera_link -> camera_color_optical_frame from the camera driver, or from dual_table.launch.py in mock).
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_param_builder import load_yaml
from launch_ros.actions import Node

PACKAGE = "rebel_dual_table_moveit_config"
DESCRIPTION = "rebel_dual_table_description"
IS_HUMBLE = os.environ.get("ROS_DISTRO") == "humble"

# Humble takes one planning plugin and a whitespace-delimited request-adapter chain (post-processing
# runs from the last adapter to the first, so time parameterization is listed last to run first).
HUMBLE_OMPL_REQUEST_ADAPTERS = (
    "default_planner_request_adapters/ResolveConstraintFrames",
    "default_planner_request_adapters/FixWorkspaceBounds",
    "default_planner_request_adapters/FixStartStateBounds",
    "default_planner_request_adapters/FixStartStateCollision",
    "default_planner_request_adapters/FixStartStatePathConstraints",
    "default_planner_request_adapters/AddTimeOptimalParameterization",
)


def to_humble_pipeline(pipeline, request_adapters):
    pipeline["planning_plugin"] = pipeline.pop("planning_plugins")[0]
    pipeline["request_adapters"] = " ".join(request_adapters)
    pipeline.pop("response_adapters", None)
    return pipeline


def launch_setup(context):
    share = get_package_share_directory(PACKAGE)
    desc_share = get_package_share_directory(DESCRIPTION)
    hw = LaunchConfiguration("hardware_protocol").perform(context)
    calibration_file = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        os.path.join(desc_share, "urdf", "rebel_dual_table.urdf.xacro"),
        " hardware_protocol:=", hw,
        " calibration_file:=", calibration_file,
    ]).perform(context)
    robot_description_semantic = Path(os.path.join(share, "config", "rebel_dual_table.srdf")).read_text()
    kinematics = load_yaml(Path(os.path.join(share, "config", "kinematics.yaml")))
    joint_limits = load_yaml(Path(os.path.join(share, "config", "joint_limits.yaml")))

    ompl_file = load_yaml(Path(os.path.join(share, "config", "ompl_planning.yaml")))
    ompl = ompl_file["move_group"]
    if IS_HUMBLE:
        to_humble_pipeline(ompl, HUMBLE_OMPL_REQUEST_ADAPTERS)
    ompl_context = dict(ompl)
    for key in ("planner_configs", "arm_left", "arm_right", "both_arms"):
        ompl_context[key] = ompl_file[key]

    moveit_args = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics,
        "robot_description_planning": joint_limits,
        "moveit_simple_controller_manager": load_yaml(Path(os.path.join(share, "config", "moveit_controllers.yaml"))),
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": False,
        "publish_planning_scene_hz": 2.0,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.02,  # >= JTC goal tolerance
        # Dual cell: MoveIt cancels a trajectory that overruns 1.2 x its duration + 0.5 s
        # (an overrun means an arm is not following; stopping is the safe outcome).
        "trajectory_execution.execution_duration_monitoring": True,
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_context,
    }
    if not IS_HUMBLE:
        moveit_args["publish_robot_description"] = True
        moveit_args["publish_robot_description_semantic"] = True

    move_group = Node(
        package="moveit_ros_move_group", executable="move_group",
        parameters=[{"use_sim_time": use_sim_time}, moveit_args], output="screen")

    rviz = Node(
        condition=IfCondition(LaunchConfiguration("use_gui")),
        package="rviz2", executable="rviz2", output={"both": "log"},
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{
            "robot_description": robot_description,
            "robot_description_semantic": robot_description_semantic,
            "robot_description_kinematics": kinematics,
            "robot_description_planning": joint_limits,
        }])
    camera_pole = Node(
        condition=IfCondition(LaunchConfiguration("run_camera_pole")),
        package="igus_rebel_moveit_config", executable="camera_pole_collision.py",
        parameters=[{"use_sim_time": use_sim_time, "base_frame": "world",
                     "camera_frame": "camera_link", "optical_frame": "camera_color_optical_frame"}],
        output="screen")
    return [move_group, rviz, camera_pole]


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    desc_share = get_package_share_directory(DESCRIPTION)
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware",
                              choices=["mock_hardware", "rebel"]),
        DeclareLaunchArgument("use_gui", default_value="true", description="start RViz (MotionPlanning)"),
        DeclareLaunchArgument("run_camera_pole", default_value="true",
                              description="maintain the camera_pole / camera_head collision boxes from camera TF"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "config", "rebel_dual_table.rviz")),
        DeclareLaunchArgument(
            "calibration_file", default_value=os.path.join(desc_share, "config", "dual_arm_calibration.yaml"),
            description="must be the same file passed to dual_table.launch.py"),
        OpaqueFunction(function=launch_setup),
    ])
