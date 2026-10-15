"""Calibration web UI (and, by default, the recorder it talks to).

    ros2 launch rebel_dual_arm_calibration calibration_ui.launch.py mode:=dual output_dir:=~/dual_arm_calibration/s1
    # the log prints  http://127.0.0.1:8095/?token=...  (a new token at every start); open exactly that URL

``hover_correction_left_mm`` / ``hover_correction_right_mm`` ("x,y,z" mm in that arm's base frame, default off) add a
constant shift to every approach-assist prediction; the page shows it while active and every hover record logs it.
Single-arm modes also get the "Wrist camera" tab (eye-in-hand hand-eye calibration of the camera on the arm; see README):
``wrist_image_topic`` / ``wrist_camera_info_topic`` (default /wrist_camera/color/...), ``wrist_handeye_output_dir`` (default
``<output_dir>/wrist_handeye``), ``wrist_camera_serial`` (recorded in the result), ``wrist_cad_roll_deg`` (image roll of the
rough CAD mount guess).  Its automatic sequence on REAL hardware additionally needs ``allow_real_move:=true``.
The "0. TCP" tab (TCP pivot calibration) reads the xacro given by ``tcp_xacro_path`` and never writes it.
``start_recorder:=false`` attaches the UI to a recorder that is already running (record.launch.py).  The UI itself never
starts a robot; arm jogging goes only through its jog controller after the operator enables it on the page.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from rebel_dual_arm_calibration.modes import MODES
from rebel_dual_arm_calibration.ui.hover_ui import validate_correction_mm


def launch_setup(context):
    mode = LaunchConfiguration("mode").perform(context)
    targets = os.path.expanduser(LaunchConfiguration("targets_file").perform(context))
    params = {"mode": mode, "port": int(LaunchConfiguration("port").perform(context)),
              "image_topic": LaunchConfiguration("image_topic").perform(context),
              "camera_info_topic": LaunchConfiguration("camera_info_topic").perform(context)}
    if targets:
        params["targets_file"] = targets
    session_dir = os.path.expanduser(LaunchConfiguration("output_dir").perform(context))
    if session_dir:        # the suggested-pose panel reads the samples from here (the recorder's own status wins)
        params["session_dir"] = session_dir
    xacro = os.path.expanduser(LaunchConfiguration("tcp_xacro_path").perform(context))
    if xacro:        # the TCP tab shows its patch against this xacro (read only); the node default is the source file
        params["tcp_xacro_path"] = xacro
    for arm in ("left", "right"):       # opt-in hover prediction correction [mm, arm base frame]; empty = zero = not passed
        name = f"hover_correction_{arm}_mm"
        vec = validate_correction_mm(LaunchConfiguration(name).perform(context), name)      # bad value: the launch fails
        if any(vec):        # passed as a string ("x,y,z"), never YAML-evaluated; the node validates it again
            params[name] = ParameterValue(",".join(repr(c) for c in vec), value_type=str)
    wd = os.path.expanduser(LaunchConfiguration("wrist_handeye_output_dir", default="").perform(context))
    if wd:
        params["wrist_handeye_output_dir"] = wd
    elif session_dir:
        params["wrist_handeye_output_dir"] = os.path.join(session_dir, "wrist_handeye")
    params["wrist_image_topic"] = LaunchConfiguration("wrist_image_topic", default="/wrist_camera/color/image_raw").perform(context)
    params["wrist_camera_info_topic"] = LaunchConfiguration("wrist_camera_info_topic", default="/wrist_camera/color/camera_info").perform(context)
    params["wrist_camera_serial"] = LaunchConfiguration("wrist_camera_serial", default="").perform(context)
    params["wrist_cad_roll_deg"] = float(LaunchConfiguration("wrist_cad_roll_deg", default="0.0").perform(context))
    params["allow_real_move"] = LaunchConfiguration("allow_real_move").perform(context).strip().lower() in ("1", "true", "yes")
    return [Node(package="rebel_dual_arm_calibration", executable="calibration_ui", name="calibration_ui",
                 output="screen", emulate_tty=True, parameters=[params])]


def generate_launch_description():
    args = [
        DeclareLaunchArgument("mode", default_value="dual", choices=list(MODES)),
        DeclareLaunchArgument("output_dir", default_value="", description="session directory (new one per session)"),
        DeclareLaunchArgument("targets_file", default_value=""),
        DeclareLaunchArgument("intrinsics_file", default_value=""),
        DeclareLaunchArgument("ambient_temp_c", default_value=""),
        DeclareLaunchArgument("ambient_rh_pct", default_value=""),
        DeclareLaunchArgument("warmup_min", default_value=""),
        DeclareLaunchArgument("max_corner_motion_px", default_value="", description="recorder stillness limit [px]"),
        DeclareLaunchArgument("start_recorder", default_value="true", description="also start the recorder"),
        DeclareLaunchArgument("port", default_value="8095"),
        DeclareLaunchArgument("allow_real_move", default_value="false",
                              description="allow 'Move there' on REAL hardware (only after the supervised first test)"),
        DeclareLaunchArgument("tcp_xacro_path", default_value="",
                              description="xacro SOURCE the TCP tab makes its patch against (read only; default: the "
                                          "igus_rebel_description source file)"),
        DeclareLaunchArgument("hover_correction_left_mm", default_value="",
                              description="OPT-IN stopgap: [x,y,z] mm added to the approach-assist predictions of the LEFT arm "
                                          "(its base frame), e.g. '-5.3,8.1,0' or '[-5.3, 8.1, 0]'; |v| <= 20 mm; empty = off"),
        DeclareLaunchArgument("hover_correction_right_mm", default_value="",
                              description="same for the RIGHT arm"),
        DeclareLaunchArgument("wrist_image_topic", default_value="/wrist_camera/color/image_raw",
                              description="wrist (eye-in-hand) camera colour image, single_* modes"),
        DeclareLaunchArgument("wrist_camera_info_topic", default_value="/wrist_camera/color/camera_info"),
        DeclareLaunchArgument("wrist_handeye_output_dir", default_value="",
                              description="where the wrist hand-eye samples / result go (default <output_dir>/wrist_handeye)"),
        DeclareLaunchArgument("wrist_camera_serial", default_value="", description="recorded in the wrist result YAML"),
        DeclareLaunchArgument("wrist_cad_roll_deg", default_value="0.0",
                              description="image roll [deg] about the optical axis of the rough CAD mount guess (0 = optical x "
                                          "along tool +Y); only used until the seed estimate exists"),
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
    ]
    recorder = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("rebel_dual_arm_calibration"), "launch", "record.launch.py"])),
        launch_arguments={k: LaunchConfiguration(k) for k in (
            "mode", "output_dir", "targets_file", "intrinsics_file", "ambient_temp_c", "ambient_rh_pct",
            "warmup_min", "max_corner_motion_px", "image_topic", "camera_info_topic")}.items(),
        condition=IfCondition(LaunchConfiguration("start_recorder")))
    return LaunchDescription(args + [recorder, OpaqueFunction(function=launch_setup)])
