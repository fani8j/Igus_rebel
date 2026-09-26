from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


REPLAY_PARAMETER_NAMES = (
    "planning_group",
    "model_source_node",
    "wait_for_enter",
    "velocity_scale",
    "acceleration_scale",
    "settle_seconds",
)


def generate_launch_description():
    replay_parameters = {
        name: LaunchConfiguration(name) for name in REPLAY_PARAMETER_NAMES
    }
    return LaunchDescription([
        DeclareLaunchArgument(
            "planning_group",
            default_value="igus_rebel_arm",
            description="MoveIt planning group",
        ),
        DeclareLaunchArgument(
            "model_source_node",
            default_value="/move_group",
            description="Running MoveIt node from which to copy URDF and SRDF parameters",
        ),
        DeclareLaunchArgument(
            "wait_for_enter",
            default_value="true",
            description=(
                "Require ENTER before each pose; the node refuses to start if "
                "stdin is not a terminal"
            ),
        ),
        DeclareLaunchArgument(
            "velocity_scale",
            default_value="0.05",
            description="MoveIt velocity scaling factor",
        ),
        DeclareLaunchArgument(
            "acceleration_scale",
            default_value="0.05",
            description="MoveIt acceleration scaling factor",
        ),
        DeclareLaunchArgument(
            "settle_seconds",
            default_value="1.0",
            description="Pause after reaching each calibration pose",
        ),

        DeclareLaunchArgument(
            "interactive_terminal",
            default_value="true",
            description=(
                "Open replay in GNOME Terminal so ENTER reaches the replay process; "
                "with false, wait_for_enter:=true is refused unless stdin is a terminal"
            ),
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("interactive_terminal")),
            package="handeye_pose_replay",
            executable="replay_handeye",
            name="handeye_pose_replay",
            output="screen",
            prefix="gnome-terminal --wait -- bash -ic 'exec \"$@\"' --",
            parameters=[replay_parameters],
        ),
        Node(
            condition=UnlessCondition(LaunchConfiguration("interactive_terminal")),
            package="handeye_pose_replay",
            executable="replay_handeye",
            name="handeye_pose_replay",
            output="screen",
            emulate_tty=True,
            parameters=[replay_parameters],
        ),
    ])
