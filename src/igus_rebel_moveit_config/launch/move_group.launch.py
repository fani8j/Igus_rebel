import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
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
            "igus_rebel_robot2.urdf.xacro",
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
    ompl_context = load_yaml(Path(ompl_file.perform(context)))
    ompl = {"ompl": ompl_context}

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
        },
        ompl_context,
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

    # Get parameters for the Servo node
    servo_params_file = PathJoinSubstitution(
        [
            FindPackageShare("igus_rebel_moveit_config"),
            "config",
            "servo.yaml",
        ]
    )
    servo_context = load_yaml(Path(servo_params_file.perform(context)))
    servo_params = {
        "moveit_servo": servo_context
    }
    # This sets the update rate and planning group name for the acceleration limiting filter.
    planning_group_name = {"planning_group_name": "igus_rebel_arm"}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
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
        get_package_share_directory('igus_rebel_moveit_config'),
        'launch',
        'moveit.rviz'
    )
    
    rviz_parameters = [
        {'robot_description_kinematics': kinematics_config,
         "robot_description_semantic": robot_description_semantic.perform(context),
         "robot_description_planning": joint_limits_config,
         'robot_description': robot_description},
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
        # move_group_node,
        # servo_node,
        # joy_node,
        # teleop_twist_joy_node,
        launch_rviz
    ]


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument("namespace", default_value="")
    prefix_arg = DeclareLaunchArgument("prefix", default_value="")
    use_gui_arg = DeclareLaunchArgument("use_gui", default_value="true")
    
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

    ld.add_action(OpaqueFunction(function=opaque_func))

    return ld