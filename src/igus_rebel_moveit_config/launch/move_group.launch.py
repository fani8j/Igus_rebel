import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
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


SPEED_PROFILES = ("default", "fast")
SPEED_PROFILE_FILE = "speed_profile_fast.yaml"
EXECUTOR_NODE = "box_h_cut_executor"
SPEED_SCALING_KEYS = ("velocity_scaling", "acceleration_scaling", "transit_velocity_scaling",
                      "transit_acceleration_scaling")
# joint_limits.yaml: the scalings are fractions of these (used for the log line only).
JOINT_LIMIT_VELOCITY_RAD_S = 0.785
JOINT_LIMIT_ACCELERATION_RAD_S2 = 5.24


def speed_profile_file(profile):
    """Path of the speed profile YAML to load AFTER box_h_cut_executor.yaml, or None for `default` (nothing is loaded:
    the effective executor parameters are exactly those of box_h_cut_executor.yaml). Fail closed on an unknown profile."""
    if profile not in SPEED_PROFILES:
        raise RuntimeError(f"speed_profile must be one of {SPEED_PROFILES}, got {profile!r}")
    if profile == "default":
        return None
    return os.path.join(get_package_share_directory(PACKAGE), "config", SPEED_PROFILE_FILE)


def effective_speed_scalings(base_file, profile_file):
    """The four scalings the executor will see: base YAML, then the profile file (which wins)."""
    values = {}
    for path in (base_file, profile_file):
        if path is None:
            continue
        section = (load_yaml(Path(path)) or {}).get(EXECUTOR_NODE, {}).get("ros__parameters", {})
        values.update({name: float(section[name]) for name in SPEED_SCALING_KEYS if name in section})
    missing = [name for name in SPEED_SCALING_KEYS if name not in values]
    if missing:
        raise RuntimeError(f"no value for {missing} in {base_file} / {profile_file}")
    return values


def speed_profile_summary(profile, v):
    a_lim, v_lim = JOINT_LIMIT_ACCELERATION_RAD_S2, JOINT_LIMIT_VELOCITY_RAD_S
    return (f"speed_profile={profile}: contact velocity_scaling={v['velocity_scaling']:g} ({v['velocity_scaling'] * v_lim:.3f} rad/s) "
            f"acceleration_scaling={v['acceleration_scaling']:g} ({v['acceleration_scaling'] * a_lim:.3f} rad/s^2); "
            f"transit_velocity_scaling={v['transit_velocity_scaling']:g} ({v['transit_velocity_scaling'] * v_lim:.3f} rad/s) "
            f"transit_acceleration_scaling={v['transit_acceleration_scaling']:g} "
            f"({v['transit_acceleration_scaling'] * a_lim:.3f} rad/s^2)")


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


def cut_recorder_script():
    """tools/cut_recorder/cut_recorder.py of this workspace (it is not installed)."""
    candidates = [
        # --symlink-install: this file resolves into src/<pkg>/launch/
        Path(__file__).resolve().parents[3] / "tools" / "cut_recorder" / "cut_recorder.py",
        Path.home() / "igus_rebel_ros2" / "tools" / "cut_recorder" / "cut_recorder.py",
    ]
    return next((c for c in candidates if c.is_file()), None)


# Camera names of tools/cut_recorder (backend `video`) -> camera keys of robot_remote's raw recorder (backend `raw`);
# mirrors dual_cut_bringup/launch/dual_cut_system.launch.py.
VIDEO_TO_RAW = {"knife": "knife", "scene": "head", "wrist": "wrist_left_knife", "right_wrist": "wrist_right_knife",
                    "left_gripper": "wrist_left_gripper", "right_gripper": "wrist_right_gripper",
                    "left_vision": "left_vision", "right_vision": "right_vision"}
RAW_TO_VIDEO = {new: old for old, new in VIDEO_TO_RAW.items()}
RAW_CAMERAS = "head,knife,wrist_left_knife,wrist_right_knife,wrist_left_gripper,wrist_right_gripper,left_vision,right_vision"
VIDEO_CAMERAS = "knife,scene,wrist,right_wrist"


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


def video_recorder_actions(context):
    """Backend `video`: the passive mp4 recorder (tools/cut_recorder, trigger = FJT action status)."""
    script = cut_recorder_script()
    if script is None:
        return [LogInfo(msg="record_cut_video: tools/cut_recorder/cut_recorder.py not found; "
                            "camera recording disabled")]
    cameras = LaunchConfiguration("cut_video_cameras").perform(context).strip() or VIDEO_CAMERAS
    cmd = ["nice", "-n", "10", "python3", str(script), "--cameras", map_cameras(cameras, RAW_TO_VIDEO)]
    out_dir = LaunchConfiguration("cut_video_dir").perform(context)
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return [ExecuteProcess(
        cmd=cmd,
        name="cut_recorder",
        output="screen",
    )]


def raw_recorder_actions(context, script):
    """Backend `raw` (default): robot_remote's read-only raw recorder. Trigger fjt-status = any FollowJointTrajectory goal on
    the arm's controller (like the video recorder), 5 s hold; one rosbag2 (sqlite3) per cut with the cameras as published
    (full-resolution JPEG), joint / controller state and tf, kept whatever the outcome."""
    cameras = LaunchConfiguration("cut_video_cameras").perform(context).strip() or RAW_CAMERAS
    cmd = ["nice", "-n", "10", "python3", str(script), "--trigger", "fjt-status", "--fjt-hold-s", "5.0",
           "--camera-keys", map_cameras(cameras, VIDEO_TO_RAW)]
    out_dir = LaunchConfiguration("cut_video_dir").perform(context)
    if out_dir:
        cmd += ["--clip-dir", out_dir]
    return [ExecuteProcess(cmd=cmd, name="cut_recorder", output="screen", sigterm_timeout="60", sigkill_timeout="15")]


def cut_recorder_actions(context):
    """Automatic cut recording (see tools/cut_recorder/README.md and robot_remote's README "Record cutting runs").

    record_cut_video true = real hardware with the H-cut executor, always = also in mock (tests), false = off. It only
    subscribes; it never commands anything. cut_recorder_backend raw (default) falls back to video when
    robot_remote or rosbag2_py is missing - the launch never fails because of it.
    """
    mode = LaunchConfiguration("record_cut_video").perform(context).strip().lower()
    if mode == "false" or (mode == "true" and (
            LaunchConfiguration("run_box_h_cut_executor").perform(context).lower() != "true"
            or LaunchConfiguration("hardware_protocol").perform(context) != "rebel")):
        return []
    if LaunchConfiguration("cut_recorder_backend").perform(context).strip().lower() == "raw":
        script, why = raw_ready(LaunchConfiguration("robot_remote_dir").perform(context))
        if script is not None:
            return raw_recorder_actions(context, script)
        return [LogInfo(msg=f"record_cut_video: raw recorder unavailable ({why}); falling back to the video backend "
                            "(tools/cut_recorder)"), *video_recorder_actions(context)]
    return video_recorder_actions(context)


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
            " tool_variant:=",
            LaunchConfiguration("tool_variant"),
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

    # Opt-in speed profile: its file is listed right after the base YAML, so its values win (default: nothing extra).
    speed_profile = LaunchConfiguration("speed_profile", default="default").perform(context).strip()
    profile_file = speed_profile_file(speed_profile)
    base_executor_yaml = config_file("box_h_cut_executor.yaml").perform(context)
    speed_log = LogInfo(msg="[igus_rebel_moveit_config] " + speed_profile_summary(
        speed_profile, effective_speed_scalings(base_executor_yaml, profile_file)))
    box_h_cut_node = Node(
        condition=IfCondition(LaunchConfiguration("run_box_h_cut_executor")),
        package=PACKAGE,
        executable="box_h_cut_executor",
        parameters=[
            config_file("box_h_cut_executor.yaml"),
            *([profile_file] if profile_file else []),
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
    start_servo_node = Node(
        condition=IfCondition(LaunchConfiguration("run_teleop")),
        package=PACKAGE,
        executable="start_servo.py",
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
        start_servo_node,
        joy_node,
        teleop_twist_joy_node,
        delayed_box_h_cut_node,
        speed_log,
        launch_rviz,
        *cut_recorder_actions(context),
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
        DeclareLaunchArgument(
            "tool_variant",
            default_value="knife_c",
            choices=["knife_c", "xeg32_knife_v1"],
            description="Tool model on link6 (xacro arg tool_variant): knife_c = variant-C knife "
                        "holder (default), xeg32_knife_v1 = previous XEG-32 gripper + old knife. "
                        "Must match the rebel.launch.py robot_state_publisher.",
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
            "speed_profile", default_value="default", choices=list(SPEED_PROFILES),
            description="H-cut executor speeds: default = box_h_cut_executor.yaml unchanged; fast = opt-in "
                        "config/speed_profile_fast.yaml loaded after it (faster transits and cut strokes, planned "
                        "accelerations <= 0.4 rad/s^2; NOT validated on the real arm)"),
        DeclareLaunchArgument(
            "box_h_cut_start_delay",
            default_value="3.0",
            description="Seconds to wait for MoveIt before starting the H-cut executor",
        ),
        DeclareLaunchArgument(
            "record_cut_video", default_value="true", choices=["true", "false", "always"],
            description="Record the cuts automatically (every trajectory execution): true = with the H-cut "
                        "executor on real hardware, always = also in mock (tests), false = off"),
        DeclareLaunchArgument(
            "cut_recorder_backend", default_value="raw", choices=["raw", "video"],
            description="raw = robot_remote's raw rosbag2 recorder, full resolution (falls back to video when robot_remote or "
                        "rosbag2_py is missing); video = tools/cut_recorder, passive mp4 clips"),
        DeclareLaunchArgument(
            "robot_remote_dir", default_value="~/robot_remote",
            description="robot_remote checkout (recording/record_raw_cut.py) for the raw backend"),
        DeclareLaunchArgument(
            "cut_video_cameras", default_value="",
            description="Cameras for record_cut_video, comma list; empty = all (raw: head,knife,wrist_left_knife,"
                        "wrist_right_knife,wrist_left_gripper,wrist_right_gripper,left_vision,right_vision; video: "
                        "knife,scene,wrist,right_wrist). Old names (scene, wrist, ...) work with both backends"),
        DeclareLaunchArgument(
            "cut_video_dir", default_value="",
            description="Clip / sidecar folder root for record_cut_video (empty: the recorder default, "
                        "the external drive's igus_nvidia/cut_videos)"),
        DeclareLaunchArgument("run_calibration_verification", default_value="false"),
        DeclareLaunchArgument("calibration_execution_allowed", default_value="false"),
        DeclareLaunchArgument(
            "calibration_verification_config",
            default_value=os.path.join(share, "config", "calibration_verification.yaml")),
        OpaqueFunction(function=opaque_func),
    ])
