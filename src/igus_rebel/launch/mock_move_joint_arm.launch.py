"""Mock robot-side joint move (MoveJoint action + hardware_state + robot_joint_states) for ONE arm
of a mock_hardware stack, for dual-arm launch files:

    IncludeLaunchDescription(PythonLaunchDescriptionSource(<igus_rebel share>/launch/mock_move_joint_arm.launch.py),
        launch_arguments={"node_name": "igus_rebel_left", "joint_prefix": "arm_left_",
                          "controller_name": "arm_left_mock_move_joint_controller",
                          "trajectory_controller_name": "arm_left_trajectory_controller",
                          "component_name": "arm_left_hw"}.items())

Only the server node. The forward_command_controller named `controller_name` must exist in the
controller_manager (loaded and spawned --inactive, one spawner at a time after the others); see
config/controller_mock.yaml for the single-arm definition. The node name is the name the real
driver's node has for this arm (the ros2_control `node_name` hardware parameter), so the
interface is identical in mock and on the robot: /<node_name>/move_joint, /<node_name>/hardware_state,
/<node_name>/robot_joint_states.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        "node_name": ("igus_rebel", "node name = real driver node name of this arm"),
        "joint_prefix": ("", "joint name prefix of this arm, e.g. arm_left_"),
        "controller_name": ("mock_move_joint_controller", "this arm's own forward_command_controller"),
        "trajectory_controller_name": ("rebel_arm_trajectory_controller",
                                       "this arm's JTC, which must be inactive during a move"),
        "component_name": ("", "<ros2_control name=...> of this arm (HardwareState.component, activation)"),
        "controller_manager": ("/controller_manager", "controller manager name"),
        "joint_state_topic": ("/joint_states", "joint state topic carrying this arm's joints"),
        "max_velocity_percent": ("25.0", "MoveJoint velocity cap in percent (hard cap 100)"),
    }
    declared = [DeclareLaunchArgument(k, default_value=v[0], description=v[1]) for k, v in args.items()]
    server = Node(
        package="igus_rebel",
        executable="mock_move_joint_server",
        name=LaunchConfiguration("node_name"),
        output="both",
        parameters=[{
            "joint_prefix": LaunchConfiguration("joint_prefix"),
            "controller_name": LaunchConfiguration("controller_name"),
            "trajectory_controller_name": LaunchConfiguration("trajectory_controller_name"),
            "component_name": LaunchConfiguration("component_name"),
            "controller_manager": LaunchConfiguration("controller_manager"),
            "joint_state_topic": LaunchConfiguration("joint_state_topic"),
            "max_velocity_percent": ParameterValue(LaunchConfiguration("max_velocity_percent"), value_type=float),
        }],
    )
    return LaunchDescription([*declared, server])
