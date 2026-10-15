"""Start the calibration recorder for one of three set-ups (``mode``).

    # dual-arm bring-up (rebel_dual_table_description dual_table.launch.py): both arms in one session
    ros2 launch rebel_dual_arm_calibration record.launch.py mode:=dual output_dir:=~/dual_arm_calibration/s1

    # today's single-arm driver, one arm at a time (record each arm in its own session, then
    #   ros2 run rebel_dual_arm_calibration solve <left_dir> <right_dir> --out dual_arm_calibration.yaml)
    ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_left  output_dir:=~/dual_arm_calibration/left
    ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_right output_dir:=~/dual_arm_calibration/right

The recorder also offers ~/capture_touch (table-board touches) and saves robot_description.urdf into the session
(see the README).  Frames and joint prefixes per mode are defined in rebel_dual_arm_calibration/modes.py. This launch file
only starts the recorder; it never starts or moves a robot.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rebel_dual_arm_calibration.modes import MODES, mode_parameters


def launch_setup(context):
    mode = LaunchConfiguration("mode").perform(context)
    params = dict(mode_parameters(mode))
    for arg, key in (("output_dir", "output_dir"), ("targets_file", "targets_file"),
                     ("image_topic", "image_topic"), ("camera_info_topic", "camera_info_topic"),
                     ("joint_states_topic", "joint_states_topic"),
                     ("robot_description_file", "robot_description_file"), ("intrinsics_file", "intrinsics_file"),
                     ("intrinsics_target", "intrinsics_target")):
        value = os.path.expanduser(LaunchConfiguration(arg).perform(context))
        if value:
            params[key] = value
    for arg in ("ambient_temp_c", "ambient_rh_pct", "warmup_min",        # session metadata (floats), optional
                "max_corner_motion_px"):                                # stillness limit override, optional
        value = LaunchConfiguration(arg).perform(context)
        if value:
            params[arg] = float(value)
    note = LaunchConfiguration("operator_note").perform(context)
    if note:
        params["operator_note"] = note
    return [Node(package="rebel_dual_arm_calibration", executable="recorder_node",
                 name="dual_arm_calibration_recorder", output="screen", parameters=[params])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="dual", choices=list(MODES),
                              description="dual | single_left | single_right (see modes.py)"),
        DeclareLaunchArgument("output_dir", default_value="",
                              description="session directory (use a new one for every session)"),
        DeclareLaunchArgument("targets_file", default_value="",
                              description="measured targets YAML (default: the package's config/targets.yaml)"),
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument("robot_description_file", default_value="",
                              description="URDF to save into the session for joint offsets (default: taken from the "
                                          "robot_description topic / robot_state_publisher parameter)"),
        DeclareLaunchArgument("intrinsics_file", default_value="",
                              description="intrinsics YAML from calibrate_intrinsics: replaces camera_info for K / distortion"),
        DeclareLaunchArgument("intrinsics_target", default_value="",
                              description="board used by capture_intrinsics (default: the table board spec)"),
        DeclareLaunchArgument("ambient_temp_c", default_value="", description="session metadata: ambient temperature [C]"),
        DeclareLaunchArgument("ambient_rh_pct", default_value="", description="session metadata: relative humidity [%]"),
        DeclareLaunchArgument("warmup_min", default_value="", description="session metadata: minutes since switch-on"),
        DeclareLaunchArgument("operator_note", default_value="", description="session metadata: free text"),
        DeclareLaunchArgument("max_corner_motion_px", default_value="",
                              description="two-frame stillness limit on the common board motion [px] (default 0.2)"),
        OpaqueFunction(function=launch_setup),
    ])
