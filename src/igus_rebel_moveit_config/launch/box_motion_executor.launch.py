import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    hardware_protocol = LaunchConfiguration("hardware_protocol")
    use_sim_time = LaunchConfiguration("use_sim_time")

    description_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_description"), "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro"]
    )
    robot_description = Command(
        [FindExecutable(name="xacro"), " ", description_file, " hardware_protocol:=", hardware_protocol]
    )
    semantic_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_moveit_config"), "config", "igus_rebel2.srdf"]
    )
    robot_description_semantic = Command([FindExecutable(name="cat"), " ", semantic_file])

    kinematics_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_moveit_config"), "config", "kinematics.yaml"]
    )
    limits_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_moveit_config"), "config", "joint_limits.yaml"]
    )
    ompl_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_moveit_config"), "config", "ompl_planning.yaml"]
    )
    executor_file = PathJoinSubstitution(
        [FindPackageShare("igus_rebel_moveit_config"), "config", "box_motion_executor.yaml"]
    )
    ompl = load_yaml(Path(ompl_file.perform(context)))
    pipeline = dict(ompl["move_group"])
    if os.environ.get("ROS_DISTRO") == "humble":
        pipeline["planning_plugin"] = pipeline.pop("planning_plugins")[0]
        pipeline["request_adapters"] = " ".join(
            [
                "default_planner_request_adapters/ResolveConstraintFrames",
                "default_planner_request_adapters/FixWorkspaceBounds",
                "default_planner_request_adapters/FixStartStateBounds",
                "default_planner_request_adapters/FixStartStateCollision",
                "default_planner_request_adapters/FixStartStatePathConstraints",
                "default_planner_request_adapters/AddTimeOptimalParameterization",
            ]
        )
        pipeline.pop("response_adapters", None)
    pipeline["planner_configs"] = ompl["planner_configs"]
    pipeline["igus_rebel_arm"] = ompl["igus_rebel_arm"]
    executor_pipeline = {"ompl": pipeline}
    return [
        Node(
            package="igus_rebel_moveit_config",
            executable="box_motion_executor",
            output="screen",
            parameters=[
                executor_file,
                {"use_sim_time": use_sim_time},
                {"robot_description": robot_description},
                {"robot_description_semantic": robot_description_semantic.perform(context)},
                {"robot_description_kinematics": load_yaml(Path(kinematics_file.perform(context)))},
                {"robot_description_planning": load_yaml(Path(limits_file.perform(context)))},
                executor_pipeline,
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "hardware_protocol",
                default_value="rebel",
                choices=["mock_hardware", "gazebo", "rebel"],
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
