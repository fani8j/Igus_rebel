from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


REPLAY_PARAMETER_NAMES = (
    "planning_group",
    "model_source_node",
    "wait_for_enter",
    "velocity_scale",
    "acceleration_scale",
    "settle_seconds",
    "poses_file",
    "arm",
    "start_index",
    "max_poses",
)

STRING_PARAMETER_NAMES = ("poses_file", "arm")


def generate_launch_description():
    replay_parameters = {
        name: LaunchConfiguration(name) for name in REPLAY_PARAMETER_NAMES
    }
    # Keep these as strings (an empty value or "~/..." must not be re-parsed as YAML).
    for name in STRING_PARAMETER_NAMES:
        replay_parameters[name] = ParameterValue(
            LaunchConfiguration(name), value_type=str
        )
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
            "poses_file",
            default_value="",
            description=(
                "samples.jsonl of a calibration session, or the session directory "
                "(~ is expanded); empty = the built-in 19 poses"
            ),
        ),
        DeclareLaunchArgument(
            "arm",
            default_value="",
            description=(
                "Only replay samples of this arm (left/right); required when the "
                "poses file contains more than one arm"
            ),
        ),
        DeclareLaunchArgument(
            "start_index",
            default_value="0",
            description="0-based index into the filtered pose list, to resume a replay",
        ),
        DeclareLaunchArgument(
            "max_poses",
            default_value="0",
            description="Replay at most this many poses (0 = all)",
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
