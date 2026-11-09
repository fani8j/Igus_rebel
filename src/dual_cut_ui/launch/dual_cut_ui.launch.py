"""Operator web UI for the dual-arm cutting cell.

    ros2 launch dual_cut_ui dual_cut_ui.launch.py
    # the log prints  http://127.0.0.1:8096/?token=...  (a new token at every start); open exactly that URL
    # add &view=present for the read-only projector view

Arguments: port, snapshot_file, detect_cwd, detect_output_dir, detect_command (one string, split like a shell would),
cameras ("name=/topic,..." ; empty = the six default cameras), cameras_off (start switched off in the page; default knife), plan_timeout_s,
camera_domain_id / detect_domain_id (cameras and the detector in another ROS domain, e.g. camera_domain_id:=11; -1 = off),
camera_localhost_only (ROS_LOCALHOST_ONLY of the camera context and the detector, to match the camera stack; -1 = inherit).  The UI listens on 127.0.0.1 only.
It closes the execution gates when it starts and when it stops.

Detect: detect_mode (action = the DetectBox action goal {dry_run: false}, the default; command = the detect_command subprocess),
detect_action (/box_seam_detector/detect_box), detect_timeout_s (300).  Action mode needs compal_box_msgs/action/DetectBox, which only
exists in the box-seam-detection install: start the UI with
    source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash
sourced (after the other setup files), otherwise the Detect step fails with that hint.
placement_limits (JSON, thresholds of the information-only placement pre-check; see dual_cut_ui/placement.py).
Plan: schedule_mode (default sequence) is set on the coordinator before every plan call; empty = leave the coordinator's value alone.
Enable execution writes only execution_enabled and execute_phase_a_enabled true (phase B is not touched); every off path writes all three false.
"""
import shlex

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)
    params = {"port": int(arg("port")), "plan_timeout_s": float(arg("plan_timeout_s")),
              "detect_mode": arg("detect_mode"), "detect_action": arg("detect_action"),
              "detect_timeout_s": float(arg("detect_timeout_s")), "schedule_mode": arg("schedule_mode"),
              "camera_domain_id": int(arg("camera_domain_id")), "detect_domain_id": int(arg("detect_domain_id")),
              "camera_localhost_only": int(arg("camera_localhost_only"))}
    for name in ("snapshot_file", "detect_cwd", "detect_output_dir", "cameras", "cameras_off"):
        if arg(name):
            params[name] = arg(name)
    if arg("placement_limits").strip():
        params["placement_limits"] = arg("placement_limits")          # JSON object, validated by the node (info-only thresholds)
    if arg("detect_command").strip():
        params["detect_command"] = shlex.split(arg("detect_command"))      # list of strings, never run through a shell
    return [Node(package="dual_cut_ui", executable="dual_cut_ui", name="dual_cut_ui", output="screen",
                 emulate_tty=True, parameters=[params])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8096"),
        DeclareLaunchArgument("snapshot_file", default_value="", description="default /home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"),
        DeclareLaunchArgument("detect_cwd", default_value="", description="working directory of the detect command"),
        DeclareLaunchArgument("detect_output_dir", default_value="", description="where the detector writes pipeline/<capture>/<run>/ (overlay lookup)"),
        DeclareLaunchArgument("detect_command", default_value="", description="full detect command as ONE string (shlex-split); empty = the capture_and_query_depth.py default"),
        DeclareLaunchArgument("cameras", default_value="", description='"name=/topic,..." CompressedImage topics; empty = defaults'),
        DeclareLaunchArgument("cameras_off", default_value="",
                              description='"name,..." cameras that start switched off in the page (empty = knife, none = all on); each browser remembers its own on/off choice'),
        DeclareLaunchArgument("plan_timeout_s", default_value="300"),
        DeclareLaunchArgument("detect_mode", default_value="action",
                              description="action = send a DetectBox goal (needs the box-seam-detection install sourced); "
                                          "command = run detect_command as a subprocess"),
        DeclareLaunchArgument("detect_action", default_value="/box_seam_detector/detect_box",
                              description="DetectBox action name used when detect_mode is action"),
        DeclareLaunchArgument("detect_timeout_s", default_value="300",
                              description="seconds to wait for the DetectBox result before cancelling the goal"),
        DeclareLaunchArgument("schedule_mode", default_value="sequence",
                              description="coordinator schedule_mode set before every plan call (phases|sequence); empty = leave it alone"),
        DeclareLaunchArgument("camera_domain_id", default_value="-1",
                              description="ROS domain the camera topics are read from (read-only second context); -1 = the UI's own domain"),
        DeclareLaunchArgument("detect_domain_id", default_value="-1",
                              description="ROS_DOMAIN_ID of the detector subprocess; -1 = camera_domain_id if set, else inherit"),
        DeclareLaunchArgument("camera_localhost_only", default_value="-1",
                              description="ROS_LOCALHOST_ONLY (0/1) of the camera context and the detector; -1 = inherit. "
                                          "Must match the camera stack (the cell may run with 0)"),
        DeclareLaunchArgument("placement_limits", default_value="",
                              description='JSON object overriding the (information only) placement pre-check thresholds, e.g. '
                                          '{"right_table_x_max_m": 0.41, "left_reach_enabled": false}; empty = defaults (see placement.py)'),
        OpaqueFunction(function=launch_setup),
    ])
