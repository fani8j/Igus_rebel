"""The whole dual-arm cutting stack in one command (cameras are NOT included).

    ROS_DOMAIN_ID=<unused> ROS_LOCALHOST_ONLY=1 \\
      ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=mock_hardware
    ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=rebel \\
      left_robot_ip:=192.168.3.11 right_robot_ip:=192.168.3.12     # real arms, operator approval only

`hardware_protocol` has NO default (the single-arm rebel.launch.py defaults to the real robot; this launch
refuses to guess). Start-up order, with delays so that each part finds the previous one:

  t = 0                  dual driver: rebel_dual_table_description dual_table.launch.py (one controller_manager,
                         arm_left_hw / arm_right_hw in fault group dual_arm_table, JSB, both trajectory
                         controllers; with `rebel` the controllers stay INACTIVE unless activate_controllers:=true)
  t = move_group_delay   dual move_group (planning scene of the cell, camera pole; RViz only with use_rviz:=true)
  t = planning_delay     both per-arm plan-only stacks (move_group + executor + camera pole + joint-state relay)
  t = coordinator_delay  dual_cut_coordinator, execution_enabled and the per-phase enables all false

Speed profile (arg speed_profile, default|fast, default `default` = nothing changes): `fast` loads
igus_rebel_moveit_config/config/speed_profile_fast.yaml AFTER the executor YAML and the per-arm overrides (both executors:
contact and transit scalings, planned accelerations <= 0.4 rad/s^2) and gives the coordinator the matching JTC sub-chord
limits. Opt-in, NOT validated on the real arms (the real JTC lags ~0.4 s x v with today's gains).

Safety defaults: nothing here enables execution; the dual move_group is plan-only (allow_move_group_execution:=false;
the per-arm move_groups are always plan-only), so RViz Plan & Execute is refused. The coordinator's enables are runtime parameters that the
operator sets on purpose (see README.md); a coordinator_config that sets any of them to true is refused.
"""
import os
import subprocess
from pathlib import Path

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, TimerAction, UnsetLaunchConfiguration)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

DRIVER_PACKAGE = "rebel_dual_table_description"
MOVEIT_PACKAGE = "rebel_dual_table_moveit_config"
COORDINATOR_PACKAGE = "dual_cut_coordinator"
THIS = "dual_cut_bringup"
MOVEIT_CONFIG_PACKAGE = "igus_rebel_moveit_config"
SPEED_PROFILES = ("default", "fast")
SPEED_PROFILE_FILE = "speed_profile_fast.yaml"
# cri_transit_enabled (robot-side MoveJoint for transit legs, stage 1) is an enable in the same sense: it hands a move to the
# robot controller, which completes it even if the driver or the link dies. Runtime parameter, never set at launch.
ENABLE_PARAMS = ("execution_enabled", "execute_phase_a_enabled", "execute_phase_b_enabled", "cri_transit_enabled")
REAL_ROS_DOMAIN = "11"


def include(package, launch_file, arguments):
    path = os.path.join(get_package_share_directory(package), "launch", launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=arguments.items())


def driver_include(arguments, unset=()):
    """The dual driver include in its own launch-configuration scope.

    Launch configurations are global: this file declares left_robot_ip / right_robot_ip with default "", so the
    included dual_table.launch.py would inherit that "" and its own DeclareLaunchArgument defaults
    (192.168.3.11 / .12) would never apply (2026-10-20: the driver got empty IPs). `unset` names the
    configurations that must not leak in; the scope is popped again after the include, so nothing leaks out."""
    return GroupAction([*[UnsetLaunchConfiguration(name) for name in unset],
                        include(DRIVER_PACKAGE, "dual_table.launch.py", arguments)], scoped=True)


def check_coordinator_config(path):
    """Refuse a coordinator parameter file that enables execution (the enables are set at runtime)."""
    with open(path) as handle:
        document = yaml.safe_load(handle)
    for key, body in (document or {}).items():
        params = body.get("ros__parameters", {}) if isinstance(body, dict) else {}
        for name in ENABLE_PARAMS:
            if params.get(name) not in (None, False):
                raise RuntimeError(
                    f"coordinator_config {path}: {name} is {params.get(name)!r} under {key!r}. Execution "
                    f"enables (and cri_transit_enabled) must stay false at launch; the operator sets them at runtime with "
                    f"'ros2 param set /dual_cut_coordinator {name} true' (see the dual_cut_bringup README).")


def speed_profile_setup(profile):
    """(coordinator speed_profile_file or "", log lines). `default` = ("", one line): nothing is loaded or changed.
    The coordinator gets the file only for `fast`; its dual_cut_coordinator section carries the JTC sub-chord limits."""
    if profile not in SPEED_PROFILES:
        raise RuntimeError(f"speed_profile must be one of {SPEED_PROFILES}, got {profile!r}")
    if profile == "default":
        return "", ["speed_profile=default: executor speeds from box_h_cut_executor.yaml, coordinator cri_subchord_* from "
                    "the coordinator config (nothing overridden)"]
    path = os.path.join(get_package_share_directory(MOVEIT_CONFIG_PACKAGE), "config", SPEED_PROFILE_FILE)
    with open(path) as handle:
        document = yaml.safe_load(handle) or {}
    coordinator = (document.get("dual_cut_coordinator") or {}).get("ros__parameters") or {}
    if set(coordinator) != {"cri_subchord_velocity_rad_s", "cri_subchord_acceleration_rad_s2"}:
        raise RuntimeError(f"{path}: the dual_cut_coordinator section must define exactly cri_subchord_velocity_rad_s and "
                           f"cri_subchord_acceleration_rad_s2, got {sorted(coordinator)}")
    for name, value in coordinator.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0.0:
            raise RuntimeError(f"{path}: {name} must be a positive number, got {value!r}")
    return path, [f"speed_profile=fast ({path}): both per-arm executors get the fast scalings (applied last; see the "
                  f"per-arm 'speed_profile=fast' lines), coordinator cri_subchord_velocity_rad_s="
                  f"{coordinator['cri_subchord_velocity_rad_s']} cri_subchord_acceleration_rad_s2="
                  f"{coordinator['cri_subchord_acceleration_rad_s2']}. NOT validated on the real arms."]


def operator_checklist(hw, left_ip, right_ip, activate):
    rule = "=" * 78
    if hw == "rebel":
        controllers = ("ACTIVE at start-up (activate_controllers:=true was given: NO identity check will be "
                       "asked)" if activate == "true" else
                       "loaded INACTIVE (activate_controllers:=false)")
        return [
            rule,
            "REAL HARDWARE (hardware_protocol:=rebel). Execution stays disabled; the arms do not move.",
            f"  arm_left  -> {left_ip}    arm_right -> {right_ip}",
            f"  Trajectory controllers: {controllers}.",
            "OPERATOR CHECKLIST",
            "  1. Verify arm identity: jog ONE arm from its pendant and watch /joint_states; the joints of",
            "     arm_left_* must be the ones that move for the left-IP arm (and the same for the right).",
            "     A swapped IP pair sends each arm the other arm's trajectories.",
            "  2. Only then activate the controllers, with the identity prompt:",
            "       ros2 run dual_cut_bringup dual_arm_recover.py activate",
            "  3. Check the state any time:  ros2 run dual_cut_bringup dual_arm_recover.py status",
            "  4. A fault on one arm latches both (fault group dual_arm_table): see the README, recover with",
            "       ros2 run dual_cut_bringup dual_arm_recover.py recover --root left|right|auto",
            "     then rearm the coordinator yourself (ros2 service call /dual_cut_coordinator/rearm ...).",
            rule,
        ]
    return [
        rule,
        "MOCK (hardware_protocol:=mock_hardware): no robot is involved. Execution stays disabled.",
        "  Controllers start active (activate_controllers:=auto). Do NOT use ROS domain 11 (real robot).",
        rule,
    ]


# Camera names of tools/cut_recorder (backend `video`) -> camera keys of robot_remote's raw recorder (backend `raw`).
VIDEO_TO_RAW = {"knife": "knife", "scene": "head", "wrist": "wrist_left_knife", "right_wrist": "wrist_right_knife",
                    "left_gripper": "wrist_left_gripper", "right_gripper": "wrist_right_gripper",
                    "left_vision": "left_vision", "right_vision": "right_vision"}
RAW_TO_VIDEO = {new: old for old, new in VIDEO_TO_RAW.items()}
RAW_CAMERAS = "head,knife,wrist_left_knife,wrist_right_knife,wrist_left_gripper,wrist_right_gripper,left_vision,right_vision"
VIDEO_CAMERAS = "knife,scene,wrist,right_wrist,left_vision,right_vision"


def map_cameras(value, mapping):
    """Comma list of camera names through `mapping` (names already in the target vocabulary pass unchanged)."""
    out = []
    for name in (n.strip() for n in value.split(",")):
        name = mapping.get(name, name)
        if name and name not in out:
            out.append(name)
    return ",".join(out)


def raw_ready(robot_remote_dir):
    """(recorder script or None, why not): robot_remote's raw recorder needs its script and a python that imports rosbag2_py
    (no LeRobot needed)."""
    script = Path(os.path.expanduser(robot_remote_dir)) / "recording" / "record_raw_cut.py"
    if not script.is_file():
        return None, f"{script} not found"
    try:
        ok = subprocess.run(["python3", "-c", "import rosbag2_py"], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"cannot run python3: {exc}"
    return (script, "") if ok else (None, "python3 cannot import rosbag2_py")


def video_recorder_actions(context, hw):
    """Backend `video`: the passive mp4 recorder (tools/cut_recorder, subscriptions only): one clip per coordinator
    execution (recording while /dual_cut_coordinator/status is WAITING_BARRIER or EXECUTING)."""
    candidates = [Path(__file__).resolve().parents[3] / "tools" / "cut_recorder" / "cut_recorder.py",
                  Path.home() / "igus_rebel_ros2" / "tools" / "cut_recorder" / "cut_recorder.py"]
    script = next((c for c in candidates if c.is_file()), None)
    if script is None:
        return [LogInfo(msg="record_cut_video: tools/cut_recorder/cut_recorder.py not found; camera recording disabled")]
    cameras = LaunchConfiguration("cut_video_cameras").perform(context).strip() or VIDEO_CAMERAS
    cmd = ["nice", "-n", "10", "python3", str(script),
           "--coordinator-status-topic", "/dual_cut_coordinator/status", "--post-roll-s", "5.0",
           "--cameras", map_cameras(cameras, RAW_TO_VIDEO)]
    cam_domain = camera_domain(context, hw)
    if cam_domain is not None:
        cmd += ["--camera-domain", str(cam_domain)]
    out_dir = LaunchConfiguration("cut_video_dir").perform(context)
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return [ExecuteProcess(cmd=cmd, name="cut_recorder", output="screen")]


def raw_recorder_actions(context, hw, script):
    """Backend `raw` (default): robot_remote's read-only raw recorder (recording/record_raw_cut.py): one rosbag2 (sqlite3)
    per coordinator execution (all phases) with every camera as published (full-resolution JPEG), joint states, controller
    states, coordinator status / reports, tf. Kept whatever the outcome (marked in clip.json). The episode folder is
    <cut_video_dir>/<YYYYmmdd_HHMMSS>/ (clip.json + bag/), the one side_cut_vision's probe looks for; index.jsonl lists them."""
    cameras = LaunchConfiguration("cut_video_cameras").perform(context).strip() or RAW_CAMERAS
    cmd = ["nice", "-n", "10", "python3", str(script), "--trigger", "dual-cut",
           "--camera-keys", map_cameras(cameras, VIDEO_TO_RAW)]
    cam_domain = camera_domain(context, hw)
    if cam_domain is not None:
        cmd += ["--camera-domain", str(cam_domain)]
    out_dir = LaunchConfiguration("cut_video_dir").perform(context)
    if out_dir:
        cmd += ["--clip-dir", out_dir]
    # SIGINT (Ctrl-C of the launch) closes the open bag within seconds; the timeouts only matter for a backed-up disk queue.
    return [ExecuteProcess(cmd=cmd, name="cut_recorder", output="screen", sigterm_timeout="60", sigkill_timeout="15")]


def cut_recorder_actions(context, hw):
    """Automatic cut recording. record_cut_video true = real hardware only, always = also in mock (tests), false = off.
    cut_recorder_backend raw (default) falls back to video when robot_remote or rosbag2_py is missing - never fails."""
    mode = LaunchConfiguration("record_cut_video").perform(context).strip().lower()
    if mode == "false" or (mode == "true" and hw != "rebel"):
        return []
    if LaunchConfiguration("cut_recorder_backend").perform(context).strip().lower() == "raw":
        script, why = raw_ready(LaunchConfiguration("robot_remote_dir").perform(context))
        if script is not None:
            return raw_recorder_actions(context, hw, script)
        return [LogInfo(msg=f"record_cut_video: raw recorder unavailable ({why}); falling back to the video backend "
                            "(tools/cut_recorder)"), *video_recorder_actions(context, hw)]
    return video_recorder_actions(context, hw)


def camera_domain(context, hw):
    """ROS domain of the cameras, or None = the stack's own domain. auto: 11 on real hardware (this cell runs
    rebel_cameras in domain 11, the robot stack elsewhere), the stack's domain in mock."""
    value = LaunchConfiguration("camera_domain_id").perform(context).strip().lower()
    if value in ("", "auto"):
        return 11 if hw == "rebel" else None
    if value == "same":
        return None
    domain = int(value)
    if not 0 <= domain <= 232:
        raise RuntimeError(f"camera_domain_id must be 0..232 (Fast DDS), auto or same, got {value!r}")
    return domain


def side_cut_probe_actions(context, hw):
    """Measure-only side-cut camera probe (src/side_cut_vision, passive: subscriptions plus its own overlay and
    measurement topics, no service/action clients, no parameter sets). It measures while the coordinator is
    WAITING_BARRIER or EXECUTING; its overlays go into the cut_recorder clip (left_vision/right_vision).
    auto = real hardware only; true also starts it in mock (tests, with a mock camera)."""
    mode = LaunchConfiguration("run_side_cut_probe").perform(context).strip().lower()
    if mode == "false" or (mode == "auto" and hw != "rebel"):
        return []
    try:
        get_package_share_directory("side_cut_vision")
    except PackageNotFoundError:
        return [LogInfo(msg="run_side_cut_probe: package side_cut_vision not found (source install_vision/setup.bash "
                            "or build it into install); side-cut probe disabled")]
    cam_domain = camera_domain(context, hw)
    probe_args = {"camera_domain_id": str(cam_domain if cam_domain is not None else -1)}
    out_dir = LaunchConfiguration("cut_video_dir").perform(context)
    if out_dir:
        probe_args["out_dir"] = out_dir
    return [TimerAction(period=float(LaunchConfiguration("coordinator_delay").perform(context)),
                        actions=[include("side_cut_vision", "side_cut_probe.launch.py", probe_args)])]


def launch_setup(context):
    hw = LaunchConfiguration("hardware_protocol").perform(context).strip()
    if hw not in ("mock_hardware", "rebel"):
        raise RuntimeError(f"hardware_protocol must be mock_hardware or rebel, got {hw!r}")
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if hw == "mock_hardware" and domain == REAL_ROS_DOMAIN:
        raise RuntimeError(
            f"hardware_protocol:=mock_hardware refused: ROS_DOMAIN_ID={REAL_ROS_DOMAIN} is the real robot's "
            f"domain. Use an unused domain with ROS_LOCALHOST_ONLY=1.")
    if hw == "mock_hardware" and os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError(
            "hardware_protocol:=mock_hardware refused: set ROS_LOCALHOST_ONLY=1 so the mock stack cannot be "
            "seen from (or reach) other machines.")

    left_ip = LaunchConfiguration("left_robot_ip").perform(context).strip()
    right_ip = LaunchConfiguration("right_robot_ip").perform(context).strip()
    activate = LaunchConfiguration("activate_controllers").perform(context).strip()
    calibration = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context).strip())
    if not calibration:
        calibration = os.path.join(get_package_share_directory(DRIVER_PACKAGE), "config",
                                   "dual_arm_calibration.yaml")
    coordinator_config = os.path.expanduser(LaunchConfiguration("coordinator_config").perform(context).strip())
    if not coordinator_config:
        coordinator_config = os.path.join(get_package_share_directory(COORDINATOR_PACKAGE), "config",
                                          "coordinator.yaml")
    check_coordinator_config(coordinator_config)

    driver_args = {"hardware_protocol": hw, "calibration_file": calibration, "activate_controllers": activate}
    # An empty argument is NOT forwarded and is removed from the driver include's scope, so that
    # dual_table.launch.py's own defaults apply (equal IPs are still refused by dual_table.launch.py).
    driver_unset = []
    left_jog_scale = LaunchConfiguration("left_jog_velocity_scale").perform(context).strip()
    right_jog_scale = LaunchConfiguration("right_jog_velocity_scale").perform(context).strip()
    for name, value in (("left_robot_ip", left_ip), ("right_robot_ip", right_ip),
                        ("left_jog_velocity_scale", left_jog_scale),
                        ("right_jog_velocity_scale", right_jog_scale)):
        if value:
            driver_args[name] = value
        else:
            driver_unset.append(name)
    # Shown in the checklist; the driver's own defaults when not given.
    shown_left = left_ip or "192.168.3.11 (dual_table.launch.py default)"
    shown_right = right_ip or "192.168.3.12 (dual_table.launch.py default)"

    move_group_args = {"hardware_protocol": hw, "calibration_file": calibration,
                       "use_gui": LaunchConfiguration("use_rviz"),
                       "run_camera_pole": LaunchConfiguration("run_camera_pole"),
                       # The coordinator executes (checked); the RViz panel / execute_trajectory must not.
                       "allow_trajectory_execution": LaunchConfiguration("allow_move_group_execution"),
                       "use_sim_time": LaunchConfiguration("use_sim_time")}
    planning_args = {name: LaunchConfiguration(name) for name in (
        "use_sim_time", "executor_config", "executor_overrides_left", "executor_overrides_right",
        "box_h_cut_start_delay", "blade_collision", "table_in_arm_scene", "table_padding_m", "speed_profile")}
    planning_args["calibration_file"] = calibration   # the SAME file as the driver and the dual move_group

    speed_profile = LaunchConfiguration("speed_profile", default="default").perform(context).strip()
    coordinator_profile_file, speed_lines = speed_profile_setup(speed_profile)

    def after(delay, action):
        return TimerAction(period=float(LaunchConfiguration(delay).perform(context)), actions=[action])

    return [
        *[LogInfo(msg=line) for line in operator_checklist(hw, shown_left, shown_right, activate)],
        *[LogInfo(msg=f"[{THIS}] {line}") for line in speed_lines],
        driver_include(driver_args, driver_unset),
        after("move_group_delay", include(MOVEIT_PACKAGE, "move_group.launch.py", move_group_args)),
        after("planning_delay", include(THIS, "both_arms_planning.launch.py", planning_args)),
        after("coordinator_delay", include(COORDINATOR_PACKAGE, "coordinator.launch.py",
                                           {"config": coordinator_config,
                                            "speed_profile_file": coordinator_profile_file})),
        *cut_recorder_actions(context, hw),
        *side_cut_probe_actions(context, hw),
    ]


def generate_launch_description():
    return LaunchDescription([
        # No default on purpose: DeclareLaunchArgument without a default aborts the launch when it is missing.
        DeclareLaunchArgument(
            "hardware_protocol", choices=["mock_hardware", "rebel"],
            description="REQUIRED. mock_hardware, or rebel (real CRI drivers; operator approval only)"),
        DeclareLaunchArgument("left_robot_ip", default_value="",
                              description="CRI controller of arm_left (rebel); empty = dual_table.launch.py default"),
        DeclareLaunchArgument("right_robot_ip", default_value="",
                              description="CRI controller of arm_right (rebel); empty = dual_table.launch.py default"),
        DeclareLaunchArgument("left_jog_velocity_scale", default_value="",
                              description="arm_left jog [%] = scale * velocity [deg/s] (rebel); empty = "
                                          "dual_table.launch.py default 1.65 (real left arm 2026-10-31; the "
                                          "controllers.yaml gains assume it)"),
        DeclareLaunchArgument("right_jog_velocity_scale", default_value="",
                              description="arm_right jog scale (rebel); empty = dual_table.launch.py default 1.65"),
        DeclareLaunchArgument("calibration_file", default_value="",
                              description="dual_arm_calibration YAML (empty = rebel_dual_table_description default); "
                                          "passed to the driver AND the dual move_group"),
        DeclareLaunchArgument("activate_controllers", default_value="auto", choices=["auto", "true", "false"],
                              description="auto: active in mock, INACTIVE with rebel (activate with dual_arm_recover.py)"),
        DeclareLaunchArgument("speed_profile", default_value="default", choices=list(SPEED_PROFILES),
                              description="default = unchanged speeds; fast = opt-in profile (igus_rebel_moveit_config "
                                          "config/speed_profile_fast.yaml: faster transits and cut strokes, planned "
                                          "accelerations <= 0.4 rad/s^2), applied after the executor YAML / overrides and to "
                                          "the coordinator's JTC sub-chord limits. NOT validated on the real arms"),
        DeclareLaunchArgument("use_rviz", default_value="false", description="RViz of the dual move_group"),
        DeclareLaunchArgument("allow_move_group_execution", default_value="false", choices=["true", "false"],
                              description="false (default): the dual /move_group is plan-only, so RViz 'Plan & "
                                          "Execute' cannot drive the arms around the dual_cut_coordinator's checks; "
                                          "true restores MoveIt trajectory execution (operator decision)"),
        DeclareLaunchArgument("run_camera_pole", default_value="true",
                              description="camera_pole / camera_head boxes in the dual scene (needs camera TF)"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("executor_config", default_value="",
                              description="base executor YAML for both per-arm planners"),
        DeclareLaunchArgument("executor_overrides_left", default_value=""),
        DeclareLaunchArgument("executor_overrides_right", default_value=""),
        DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
        DeclareLaunchArgument("blade_collision", default_value="true", choices=["true", "false"],
                              description="per-arm planners: knife_c_blade collision box + carton_contact_links "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("table_in_arm_scene", default_value="true", choices=["true", "false"],
                              description="per-arm planners: the dual cell's table as collision object `table` "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("table_padding_m", default_value="0.010",
                              description="padding of that table box (default 10 mm = the coordinator's table_min_clearance_m; "
                                          "the base link is exempt, every other link keeps the margin)"),
        DeclareLaunchArgument("coordinator_config", default_value="",
                              description="coordinator parameter YAML (default: dual_cut_coordinator "
                                          "config/coordinator.yaml); the execution enables must be false in it"),
        DeclareLaunchArgument("move_group_delay", default_value="2.0",
                              description="seconds after the driver before the dual move_group starts"),
        DeclareLaunchArgument("planning_delay", default_value="4.0",
                              description="seconds after the driver before the per-arm planners start"),
        DeclareLaunchArgument("coordinator_delay", default_value="10.0",
                              description="seconds after the driver before the coordinator starts"),
        DeclareLaunchArgument("record_cut_video", default_value="true", choices=["true", "false", "always"],
                              description="record the cuts automatically: true = rebel only, always = also in mock "
                                          "(tests), false = off"),
        DeclareLaunchArgument("cut_recorder_backend", default_value="raw", choices=["raw", "video"],
                              description="raw = robot_remote's raw rosbag2 recorder, full resolution (falls back to video when "
                                          "robot_remote or rosbag2_py is missing); video = tools/cut_recorder, passive mp4 clips"),
        DeclareLaunchArgument("robot_remote_dir", default_value="~/robot_remote",
                              description="robot_remote checkout (recording/record_raw_cut.py) for the raw backend"),
        DeclareLaunchArgument("cut_video_cameras", default_value="",
                              description="comma list; empty = all (raw: head,knife,wrist_left_knife,wrist_right_knife,"
                                          "wrist_left_gripper,wrist_right_gripper,left_vision,right_vision; video: knife,scene,"
                                          "wrist,right_wrist,left_vision,right_vision). Old names (scene, wrist, ...) work with both"),
        DeclareLaunchArgument("camera_domain_id", default_value="auto",
                              description="ROS domain of the cameras for the recorder and the side-cut probe: "
                                          "auto (11 with rebel, the stack's domain in mock), same, or a number"),
        DeclareLaunchArgument("run_side_cut_probe", default_value="auto", choices=["auto", "true", "false"],
                              description="measure-only side-cut camera probe (side_cut_vision, passive): "
                                          "auto = rebel only, true = also in mock"),
        DeclareLaunchArgument("cut_video_dir", default_value="",
                              description="clip / sidecar directory (empty = the recorder's default, external drive if mounted)"),
        OpaqueFunction(function=launch_setup),
    ])
