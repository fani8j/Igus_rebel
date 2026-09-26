import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PACKAGE = "igus_rebel_moveit_config"
IS_HUMBLE = os.environ.get("ROS_DISTRO") == "humble"

# Humble takes one planning plugin and a whitespace-delimited request-adapter chain
# (post-processing runs from the last adapter to the first, so time parameterization
# is listed last to run first). The *.yaml files use the Jazzy list format.
HUMBLE_OMPL_REQUEST_ADAPTERS = (
    "default_planner_request_adapters/ResolveConstraintFrames",
    "default_planner_request_adapters/FixWorkspaceBounds",
    "default_planner_request_adapters/FixStartStateBounds",
    "default_planner_request_adapters/FixStartStateCollision",
    "default_planner_request_adapters/FixStartStatePathConstraints",
    "default_planner_request_adapters/AddTimeOptimalParameterization",
)
HUMBLE_CHOMP_REQUEST_ADAPTERS = HUMBLE_OMPL_REQUEST_ADAPTERS
HUMBLE_PILZ_REQUEST_ADAPTERS = (
    "default_planner_request_adapters/FixWorkspaceBounds",
    "default_planner_request_adapters/FixStartStateBounds",
    "default_planner_request_adapters/FixStartStateCollision",
)


def package_file(package, *parts):
    return PathJoinSubstitution([FindPackageShare(package), *parts])


def config_file(filename):
    return package_file(PACKAGE, "config", filename)


def load_config(context, filename):
    return load_yaml(Path(config_file(filename).perform(context)))


def to_humble_pipeline(pipeline, request_adapters):
    """Translate a Jazzy-format planning pipeline dict to Humble parameter names."""
    pipeline["planning_plugin"] = pipeline.pop("planning_plugins")[0]
    pipeline["request_adapters"] = " ".join(request_adapters)
    pipeline.pop("response_adapters", None)
    return pipeline


def opaque_func(context, *args, **kwargs):
    hardware_protocol = LaunchConfiguration("hardware_protocol")
    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_description = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            package_file("igus_rebel_description", "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro"),
            " hardware_protocol:=",
            hardware_protocol,
        ]
    ).perform(context)
    robot_description_semantic = Path(
        config_file("igus_rebel2.srdf").perform(context)
    ).read_text()
    kinematics_config = load_config(context, "kinematics.yaml")
    joint_limits_config = load_config(context, "joint_limits.yaml")

    ompl_context = load_config(context, "ompl_planning.yaml")
    ompl_pipeline = ompl_context["move_group"]
    if IS_HUMBLE:
        to_humble_pipeline(ompl_pipeline, HUMBLE_OMPL_REQUEST_ADAPTERS)
    executor_ompl_context = dict(ompl_pipeline)
    executor_ompl_context["planner_configs"] = ompl_context["planner_configs"]
    executor_ompl_context["igus_rebel_arm"] = ompl_context["igus_rebel_arm"]

    chomp_context = load_config(context, "chomp_planning.yaml")
    pilz_context = load_config(context, "pilz_industrial_motion_planner_planning.yaml")
    if IS_HUMBLE:
        to_humble_pipeline(chomp_context, HUMBLE_CHOMP_REQUEST_ADAPTERS)
        to_humble_pipeline(pilz_context, HUMBLE_PILZ_REQUEST_ADAPTERS)

    moveit_args = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics_config,
        "robot_description_planning": joint_limits_config,
        "moveit_simple_controller_manager": load_config(context, "moveit_controllers.yaml"),
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": False,
        "publish_planning_scene_hz": 2.0,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.02,  # >= JTC goal tolerance
        "trajectory_execution.execution_duration_monitoring": False,
        # The H-cut executor dispatches retained MTC solutions through this capability.
        "capabilities": "move_group/ExecuteTaskSolutionCapability",
        # All pipelines stay loaded so the executor's motion_profile can pick OMPL, CHOMP or Pilz.
        "planning_pipelines": ["ompl", "chomp", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "ompl",
        "ompl": executor_ompl_context,
        "chomp": chomp_context,
        "pilz_industrial_motion_planner": pilz_context,
    }
    if not IS_HUMBLE:
        # move_group parameters introduced after Humble.
        moveit_args["publish_robot_description"] = True
        moveit_args["publish_robot_description_semantic"] = True

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[{"use_sim_time": use_sim_time}, moveit_args],
    )

    box_h_cut_node = Node(
        condition=IfCondition(LaunchConfiguration("run_box_h_cut_executor")),
        package=PACKAGE,
        executable="box_h_cut_executor",
        parameters=[
            config_file("box_h_cut_executor.yaml"),
            {"use_sim_time": use_sim_time},
            moveit_args,
        ],
        additional_env={"FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"},
        output="screen",
    )
    delayed_box_h_cut_node = TimerAction(
        period=LaunchConfiguration("box_h_cut_start_delay"), actions=[box_h_cut_node]
    )
    camera_pole_node = Node(
        condition=IfCondition(LaunchConfiguration("run_camera_pole")),
        package=PACKAGE,
        executable="camera_pole_collision.py",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    calibration_params = load_yaml(Path(
        LaunchConfiguration("calibration_verification_config").perform(context)
    ))["calibration_verification"]["ros__parameters"]
    calibration_params["execution_allowed"] = LaunchConfiguration("calibration_execution_allowed")
    calibration_node = Node(
        condition=IfCondition(LaunchConfiguration("run_calibration_verification")),
        package=PACKAGE,
        executable="calibration_verification",
        parameters=[moveit_args, calibration_params, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    servo_context = load_config(context, "servo.yaml")
    if IS_HUMBLE:
        # Translate parameters renamed or removed after Humble's MoveIt Servo release.
        joint_limit_margins = servo_context.pop("joint_limit_margins", [0.1])
        servo_context["joint_limit_margin"] = joint_limit_margins[0]
        servo_context.pop("use_smoothing", None)
        servo_context.pop("check_octomap_collisions", None)
        servo_context["planning_frame"] = "base_link"
        servo_context["ee_frame_name"] = "xeg32_tool_tip"
        servo_context["robot_link_command_frame"] = "base_link"
        servo_context["use_gazebo"] = hardware_protocol.perform(context) == "gazebo"

    moveit_servo_libexec = os.path.join(get_package_prefix("moveit_servo"), "lib", "moveit_servo")
    servo_executable = (
        "servo_node"
        if os.path.exists(os.path.join(moveit_servo_libexec, "servo_node"))
        else "servo_node_main"
    )
    servo_node = Node(
        condition=IfCondition(LaunchConfiguration("run_teleop")),
        package="moveit_servo",
        executable=servo_executable,
        parameters=[
            {"use_sim_time": use_sim_time},
            {"moveit_servo": servo_context},
            # Planning group for the acceleration-limiting filter.
            {"planning_group_name": "igus_rebel_arm"},
            moveit_args,
        ],
        output="screen",
    )

    joy_node = Node(
        condition=IfCondition(LaunchConfiguration("run_gamepad")),
        package="joy",
        executable="joy_node",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )
    teleop_twist_joy_node = Node(
        condition=IfCondition(LaunchConfiguration("run_gamepad")),
        package=PACKAGE,
        executable="rebel_servo_teleop_gamepad",
        parameters=[{"use_sim_time": use_sim_time}, config_file("gamepad.yaml")],
        output="screen",
    )

    launch_rviz = Node(
        condition=IfCondition(LaunchConfiguration("use_gui")),
        package="rviz2",
        executable="rviz2",
        output={"both": "log"},
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            {
                "robot_description": robot_description,
                "robot_description_semantic": robot_description_semantic,
                "robot_description_kinematics": kinematics_config,
                "robot_description_planning": joint_limits_config,
            },
        ],
    )

    return [
        move_group_node,
        camera_pole_node,
        calibration_node,
        servo_node,
        joy_node,
        teleop_twist_joy_node,
        delayed_box_h_cut_node,
        launch_rviz,
    ]


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="Use sim time if true"),
        DeclareLaunchArgument(
            "hardware_protocol",
            default_value="rebel",
            choices=["mock_hardware", "gazebo", "rebel"],
            description="Which hardware protocol or mock hardware should be used",
        ),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("rviz_config",
                              default_value=os.path.join(share, "launch", "moveit.rviz")),
        DeclareLaunchArgument(
            "run_teleop", default_value="false",
            description="Start MoveIt Servo (needed by keyboard and gamepad teleop)"),
        DeclareLaunchArgument(
            "run_gamepad", default_value="false",
            description="Start joy_node + gamepad teleop (also needs run_teleop:=true); "
                        "keyboard teleop is a separate manual node"),
        DeclareLaunchArgument(
            "run_camera_pole", default_value="true",
            description="Maintain the external camera_pole collision object from camera TF "
                        "(required by H-cut and calibration verification planning)"),
        DeclareLaunchArgument(
            "run_box_h_cut_executor",
            default_value="false",
            description="Start the frozen-snapshot plan-only H-cut MTC executor",
        ),
        DeclareLaunchArgument(
            "box_h_cut_start_delay",
            default_value="3.0",
            description="Seconds to wait for MoveIt before starting the H-cut executor",
        ),
        DeclareLaunchArgument("run_calibration_verification", default_value="false"),
        DeclareLaunchArgument("calibration_execution_allowed", default_value="false"),
        DeclareLaunchArgument(
            "calibration_verification_config",
            default_value=os.path.join(share, "config", "calibration_verification.yaml")),
        OpaqueFunction(function=opaque_func),
    ])
