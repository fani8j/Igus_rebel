import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def opaque_func(context, *args, **kwargs):
    
    namespace = LaunchConfiguration("namespace")
    hardware_protocol = LaunchConfiguration('hardware_protocol')
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_mtc_program = LaunchConfiguration("run_mtc_program")
    execute_mtc = LaunchConfiguration("execute_mtc")
    mtc_start_delay = LaunchConfiguration("mtc_start_delay")
    run_box_motion_executor = LaunchConfiguration("run_box_motion_executor")
    box_motion_start_delay = LaunchConfiguration("box_motion_start_delay")
    run_box_h_cut_executor = LaunchConfiguration("run_box_h_cut_executor")
    box_h_cut_start_delay = LaunchConfiguration("box_h_cut_start_delay")

    joint_limits_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "joint_limits.yaml",
        ]
    )

    robot_description_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_description"),
            "urdf",
            "igus_rebel_xeg32_wrapper.urdf.xacro",
        ]
    )
    robot_description = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_file,
            " hardware_protocol:=",
            hardware_protocol,
        ]
    )
    
    robot_description_semantic_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "igus_rebel2.srdf",
        ]
    )
    
    robot_description_semantic = Command(
        [
            FindExecutable(name="cat"),
            " ",
            robot_description_semantic_file,
        ]
    )

    controllers_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "moveit_controllers.yaml",   
        ]
    )

    controllers_dict = load_yaml(Path(controllers_file.perform(context)))

    ompl_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "ompl_planning.yaml",
        ]
    )
    chomp_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "chomp_planning.yaml",
        ]
    )
    pilz_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "pilz_industrial_motion_planner_planning.yaml",
        ]
    )
    ompl_context = load_yaml(Path(ompl_file.perform(context)))
    if os.environ.get("ROS_DISTRO") == "humble":
        # Humble accepts one planner plugin and a whitespace-delimited request-adapter chain.
        ompl_pipeline = ompl_context["move_group"]
        ompl_pipeline["planning_plugin"] = ompl_pipeline.pop("planning_plugins")[0]
        ompl_pipeline["request_adapters"] = " ".join([
            "default_planner_request_adapters/ResolveConstraintFrames",
            "default_planner_request_adapters/FixWorkspaceBounds",
            "default_planner_request_adapters/FixStartStateBounds",
            "default_planner_request_adapters/FixStartStateCollision",
            "default_planner_request_adapters/FixStartStatePathConstraints",
            "default_planner_request_adapters/AddTimeOptimalParameterization",
        ])
        ompl_pipeline.pop("response_adapters", None)
    ompl = {"ompl": ompl_context}
    executor_ompl_context = dict(ompl_context["move_group"])
    executor_ompl_context["planner_configs"] = ompl_context["planner_configs"]
    executor_ompl_context["igus_rebel_arm"] = ompl_context["igus_rebel_arm"]
    executor_ompl = {"ompl": executor_ompl_context}
    chomp_context = load_yaml(Path(chomp_file.perform(context)))
    pilz_context = load_yaml(Path(pilz_file.perform(context)))
    if os.environ.get("ROS_DISTRO") == "humble":
        pilz_context["planning_plugin"] = pilz_context.pop("planning_plugins")[0]
        pilz_context["request_adapters"] = " ".join(
            [
                "default_planner_request_adapters/FixWorkspaceBounds",
                "default_planner_request_adapters/FixStartStateBounds",
                "default_planner_request_adapters/FixStartStateCollision",
            ]
        )
    named_pipelines = {
        "planning_pipelines": [
            "ompl",
            "chomp",
            "pilz_industrial_motion_planner",
        ],
        "default_planning_pipeline": "ompl",
        "chomp": chomp_context,
        "pilz_industrial_motion_planner": pilz_context,
    }
    executor_extra_pipelines = {
        "chomp": chomp_context,
        "pilz_industrial_motion_planner": pilz_context,
    }

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_dict,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    robot_description_kinematics_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "kinematics.yaml",
        ]
    )

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    
    kinematics_config = load_yaml(Path(robot_description_kinematics_file.perform(context)))
    joint_limits_config = load_yaml(Path(joint_limits_file.perform(context)))

    moveit_args_not_concatenated = [
        {"robot_description": robot_description.perform(context)},
        {"robot_description_semantic": robot_description_semantic.perform(context)},
        {"robot_description_kinematics": kinematics_config},
        {"robot_description_planning": joint_limits_config},
        moveit_controllers,
        planning_scene_monitor_parameters,
        {
            "publish_robot_description": True,
            "publish_robot_description_semantic": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
            "capabilities": "move_group/ExecuteTaskSolutionCapability",
        },
        executor_ompl,
        named_pipelines,
    ]

    # Concatenate all dictionaries together, else moveitpy won't read all parameters
    moveit_args = dict()
    for d in moveit_args_not_concatenated:
        moveit_args.update(d)
                        
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            moveit_args,
        ],
    )

    mtc_params_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "mtc_ptp_lin.yaml",
        ]
    )
    mtc_node = Node(
        condition=IfCondition(run_mtc_program),
        package="igus_rebel_moveit_config",
        executable="rebel_mtc_ptp_lin",
        namespace=namespace,
        parameters=[
            mtc_params_file,
            {"execute": execute_mtc},
            {"use_sim_time": use_sim_time},
            moveit_args,
        ],
        output="screen",
    )
    delayed_mtc_node = TimerAction(period=mtc_start_delay, actions=[mtc_node])

    box_motion_params_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "box_motion_executor.yaml",
        ]
    )
    box_motion_node = Node(
        condition=IfCondition(run_box_motion_executor),
        package="igus_rebel_moveit_config",
        executable="box_motion_executor",
        namespace=namespace,
        parameters=[
            box_motion_params_file,
            {"use_sim_time": use_sim_time},
            moveit_args,
            executor_ompl,
            executor_extra_pipelines,
        ],
        output="screen",
    )
    delayed_box_motion_node = TimerAction(
        period=box_motion_start_delay, actions=[box_motion_node]
    )
    box_h_cut_params_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "box_h_cut_executor.yaml",
        ]
    )
    box_h_cut_node = Node(
        condition=IfCondition(run_box_h_cut_executor),
        package="igus_rebel_moveit_config",
        executable="box_h_cut_executor",
        namespace=namespace,
        parameters=[
            box_h_cut_params_file,
            {"use_sim_time": use_sim_time},
            moveit_args,
            executor_ompl,
            executor_extra_pipelines,
        ],
        additional_env={"FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"},
        output="screen",
    )
    delayed_box_h_cut_node = TimerAction(
        period=box_h_cut_start_delay, actions=[box_h_cut_node]
    )

    servo_params_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "servo.yaml",
        ]
    )
    servo_context = load_yaml(Path(servo_params_file.perform(context)))
    if os.environ.get("ROS_DISTRO") == "humble":
        # Translate parameters renamed or removed after Humble's MoveIt Servo release.
        joint_limit_margins = servo_context.pop("joint_limit_margins", [0.1])
        servo_context["joint_limit_margin"] = joint_limit_margins[0]
        servo_context.pop("use_smoothing", None)
        servo_context.pop("check_octomap_collisions", None)
        servo_context["planning_frame"] = "base_link"
        servo_context["ee_frame_name"] = "xeg32_tool_tip"
        servo_context["robot_link_command_frame"] = "base_link"
        servo_context["use_gazebo"] = hardware_protocol.perform(context) == "gazebo"
    servo_params = {
        "moveit_servo": servo_context
    }
    # This sets the update rate and planning group name for the acceleration limiting filter.
    planning_group_name = {"planning_group_name": "igus_rebel_arm"}

    moveit_servo_libexec = os.path.join(get_package_prefix("moveit_servo"), "lib", "moveit_servo")
    servo_executable = (
        "servo_node"
        if os.path.exists(os.path.join(moveit_servo_libexec, "servo_node"))
        else "servo_node_main"
    )

    servo_node = Node(
        package="moveit_servo",
        executable=servo_executable,
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            servo_params,
            planning_group_name,
            moveit_args,
        ],
        output="screen",
    )

    # Launch gamepad
    joy_node = Node(
        package="joy",
        executable="joy_node",
        parameters=[{'use_sim_time': use_sim_time}],
        output="screen",
    )
    
    teleop_joy_twist_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "gamepad.yaml",
        ]
    )
    teleop_twist_joy_node = Node(
        package="igus_rebel_moveit_config",
        executable="rebel_servo_teleop_gamepad",
        parameters=[{'use_sim_time': use_sim_time}, teleop_joy_twist_file],
        output="screen",
    )

    default_rviz_file = os.path.join(
        get_package_share_directory("igus_rebel_moveit_config"),
        "launch",
        "moveit.rviz",
    )
    
    rviz_parameters = [
        {
            "robot_description_kinematics": kinematics_config,
            "robot_description_semantic": robot_description_semantic.perform(context),
            "robot_description_planning": joint_limits_config,
            "robot_description": robot_description.perform(context),
        },
    ]

    launch_rviz = Node(
        condition=IfCondition(LaunchConfiguration("use_gui")),
        package="rviz2",
        executable="rviz2",
        output={"both": "log"},
        arguments=["-d", default_rviz_file],
        parameters=rviz_parameters,
    )
    
    return [
        move_group_node,
        servo_node,
        joy_node,
        teleop_twist_joy_node,
        delayed_mtc_node,
        delayed_box_motion_node,
        delayed_box_h_cut_node,
        launch_rviz
    ]


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument("namespace", default_value="")
    prefix_arg = DeclareLaunchArgument("prefix", default_value="")
    use_gui_arg = DeclareLaunchArgument("use_gui", default_value="true")
    run_mtc_program_arg = DeclareLaunchArgument(
        "run_mtc_program",
        default_value="false",
        description="Plan the configured PTP then LIN task",
    )
    execute_mtc_arg = DeclareLaunchArgument(
        "execute_mtc",
        default_value="false",
        description="Execute the MTC solution after planning; false is planning-only",
    )
    mtc_start_delay_arg = DeclareLaunchArgument(
        "mtc_start_delay",
        default_value="3.0",
        description="Seconds to wait for controllers before starting the MTC task",
    )
    run_box_motion_executor_arg = DeclareLaunchArgument(
        "run_box_motion_executor",
        default_value="false",
        description="Start the plan-only carton hover MTC executor",
    )
    box_motion_start_delay_arg = DeclareLaunchArgument(
        "box_motion_start_delay",
        default_value="3.0",
        description="Seconds to wait for MoveIt before starting the box executor",
    )
    run_box_h_cut_executor_arg = DeclareLaunchArgument(
        "run_box_h_cut_executor",
        default_value="false",
        description="Start the frozen-snapshot plan-only H-cut MTC executor",
    )
    box_h_cut_start_delay_arg = DeclareLaunchArgument(
        "box_h_cut_start_delay",
        default_value="3.0",
        description="Seconds to wait for MoveIt before starting the H-cut executor",
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='false', 
        description='Use sim time if true')

    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="rebel",
        choices=["mock_hardware", "gazebo", "rebel"],
        description="Which hardware protocol or mock hardware should be used",
    )

    ld = LaunchDescription()
    ld.add_action(use_sim_time_arg)
    ld.add_action(namespace_arg)
    ld.add_action(use_gui_arg)
    ld.add_action(hardware_protocol_arg)
    ld.add_action(run_mtc_program_arg)
    ld.add_action(execute_mtc_arg)
    ld.add_action(mtc_start_delay_arg)
    ld.add_action(run_box_motion_executor_arg)
    ld.add_action(box_motion_start_delay_arg)
    ld.add_action(run_box_h_cut_executor_arg)
    ld.add_action(box_h_cut_start_delay_arg)

    ld.add_action(OpaqueFunction(function=opaque_func))

    return ld