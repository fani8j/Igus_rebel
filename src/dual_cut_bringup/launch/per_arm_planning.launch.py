"""Per-arm, plan-only copy of the single-arm cutting planner for the dual cell (mock first).

    ros2 launch dual_cut_bringup per_arm_planning.launch.py arm:=left   # or right

Everything runs in namespace /arm_<arm>:

  move_group            single-arm knife_c URDF + SRDF, allow_trajectory_execution:=false, no controllers.
                        Its /tf and /tf_static are REMAPPED to unused topics: the single-arm URDF root link is
                        `world`, the dual TF tree is also rooted at `world`, so importing the dual TF would place
                        this scene's frames relative to the wrong `world`.
  box_h_cut_executor    unchanged installed binary. Every absolute topic of the P3/P4 design table is remapped to
                        /arm_<arm>/...; the executor YAML is loaded into a dict and passed as parameters (a params
                        file keyed `box_h_cut_executor:` is NOT applied to /arm_<arm>/box_h_cut_executor).
                        execution_enabled / debug_motion_execution_enabled are forced to false.
  camera_pole_collision camera_pole_collision.py on the DUAL TF tree (base_frame:=arm_<arm>_base_link),
                        publishing in frame base_link (output_frame_id) into /arm_<arm>/ apply_planning_scene and
                        /arm_<arm>/box_motion/camera_pole_collision.
  table_collision       table_collision.py (arg table_in_arm_scene, default true): the dual cell's `table` box as
                        collision object `table` in this arm's scene, in the arm's planning frame (base_link), from
                        the same calibration YAML and table size/thickness/setback as the dual URDF, padded by
                        table_padding_m (default 10 mm; the arm's base_link is exempt through an ACM entry, every other
                        link keeps the margin; the coordinator's table_min_clearance_m gate uses the same value).
                        Without it only the coordinator's
                        final world check sees the table and refuses plans the per-arm planner could have routed
                        around it.
  joint_state_relay     /joint_states (dual) -> /arm_<arm>/joint_states with arm_<arm>_jointN -> jointN.

Executor parameters = box_h_cut_executor.yaml (arg executor_config) + config/arm_<arm>_executor_overrides.yaml
(arg executor_overrides) + execution_enabled:=false. Read-only parameters (e.g. external_snapshot_file in a
test) are set only here, at launch.

SPEED PROFILE (arg speed_profile, default|fast, default `default`): `default` changes nothing. `fast` applies the four
speed scalings of igus_rebel_moveit_config/config/speed_profile_fast.yaml AFTER the executor YAML, the overrides file and
the launch-owned values, so neither file can undo them (an overrides file that sets a scaling gets a WARNING). The active
profile and the effective scalings are logged at start-up.

BLADE COLLISION (arg blade_collision, default true since 2026-10-15 when the main install got the executor with
carton_contact_links; knife_c only): the single-arm knife_c model has no collision
geometry for the 14.4 mm exposed blade, but the dual model has one (<prefix>knife_c_blade), and the coordinator's
strict world check tests it against detected_carton. So the per-arm model is built with blade_collision:=true
(same box as the dual model), the SRDF gets the one blade-handpiece disable pair
(igus_rebel_moveit_config/config/knife_c_blade_disable_collisions.srdf.inc), and the executor parameter
carton_contact_links = [<tool_link>, knife_c_blade] lets the blade touch the carton ONLY while a cut stage's scoped
allowance is installed (free-space transits keep the blade fully checked). It is set here, after the YAML and the
overrides file, and wins over both. blade_collision:=false reproduces the previous stack exactly (no blade link,
default carton_contact_links). The default stays false until the installed executor binary has the
carton_contact_links parameter: a blade link with an executor that does not know the parameter would
be checked against the carton during the cut strokes and break every cut plan.

RIGHT ARM: the perception snapshot is written in the LEFT base frame (`base_link` = left base). The right executor
brings it into its own frame with a fixed transform (snapshot_frame_mode fixed_transform, set in
config/arm_right_executor_overrides.yaml). The transform values snapshot_planning_base_xyz / _rpy have ONE source:
arm_right_base of the dual_arm_calibration.yaml given by `calibration_file` (the file the dual driver and the dual
move_group use). They are passed as node parameters AFTER the overrides file, so they win; if the overrides file
also carries them and they differ, the launch logs a WARNING and the calibration YAML still wins (the file should
not carry them at all).
"""
import os
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_param_builder import load_yaml
from launch_ros.actions import Node

MOVEIT_CONFIG = "igus_rebel_moveit_config"
DESCRIPTION = "igus_rebel_description"
THIS = "dual_cut_bringup"
IS_HUMBLE = os.environ.get("ROS_DISTRO") == "humble"
EXECUTOR_NODE = "box_h_cut_executor"
BLADE_LINK = "knife_c_blade"
BLADE_SRDF_FRAGMENT = "knife_c_blade_disable_collisions.srdf.inc"

# Same adapter chain as igus_rebel_moveit_config/launch/move_group.launch.py (Humble format).
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

# Absolute names used by the executor (design table section 1) -> relative to the arm namespace.
EXECUTOR_TOPICS = (
    "/joint_states",
    "/collision_object",
    "/box_motion/h_cut_markers",
    "/box_motion/cut_target_poses",
    "/box_motion/camera_pole_collision",
    "/delta_twist_cmds",
    "/delta_joint_cmds",
)


def to_humble_pipeline(pipeline, request_adapters):
    pipeline["planning_plugin"] = pipeline.pop("planning_plugins")[0]
    pipeline["request_adapters"] = " ".join(request_adapters)
    pipeline.pop("response_adapters", None)
    return pipeline


def node_parameters(document, source):
    """Merge the ros__parameters of every node key that can match /arm_x/box_h_cut_executor."""
    merged = {}
    if document is None:
        return merged
    if not isinstance(document, dict):
        raise RuntimeError(f"{source}: top level must be a mapping")
    for key, body in document.items():
        if key not in (EXECUTOR_NODE, "/**", f"/{EXECUTOR_NODE}"):
            raise RuntimeError(f"{source}: unexpected top-level key {key!r}")
        if body is None:
            continue
        params = body.get("ros__parameters") if isinstance(body, dict) else None
        if not isinstance(params, dict):
            raise RuntimeError(f"{source}: {key!r} needs a ros__parameters mapping")
        merged.update(params)
    return merged


SPEED_PROFILES = ("default", "fast")
SPEED_SCALING_KEYS = ("velocity_scaling", "acceleration_scaling", "transit_velocity_scaling",
                      "transit_acceleration_scaling")
SPEED_PROFILE_FILE = "speed_profile_fast.yaml"
# joint_limits.yaml (igus_rebel_moveit_config): the scalings are fractions of these; only used for the log line.
JOINT_LIMIT_VELOCITY_RAD_S = 0.785
JOINT_LIMIT_ACCELERATION_RAD_S2 = 5.24


def speed_profile_path():
    return os.path.join(get_package_share_directory(MOVEIT_CONFIG), "config", SPEED_PROFILE_FILE)


def speed_profile_parameters(profile, profile_file=None):
    """Executor parameters the speed profile sets ({} for `default`). Fail closed: an unknown profile, or a fast profile file
    that does not define exactly the four finite scalings in (0, 1], is a launch error."""
    if profile not in SPEED_PROFILES:
        raise RuntimeError(f"speed_profile must be one of {SPEED_PROFILES}, got {profile!r}")
    if profile == "default":
        return {}
    path = profile_file or speed_profile_path()
    params = node_parameters({k: v for k, v in (load_yaml(Path(path)) or {}).items() if k == EXECUTOR_NODE}, path)
    if set(params) != set(SPEED_SCALING_KEYS):
        raise RuntimeError(f"{path}: the {EXECUTOR_NODE} section must define exactly {SPEED_SCALING_KEYS}, got {sorted(params)}")
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < float(value) <= 1.0):
            raise RuntimeError(f"{path}: {name} must be a number in (0, 1], got {value!r}")
    return {name: float(value) for name, value in params.items()}


def speed_profile_summary(profile, params):
    """One log line: the active profile and the effective scalings with their physical values (rad/s, rad/s^2).
    A scaling the YAML does not set is shown as the executor's code default."""
    code_defaults = {"velocity_scaling": 0.1376, "acceleration_scaling": 0.0763,
                     "transit_velocity_scaling": 0.1376, "transit_acceleration_scaling": 0.0763}

    def show(name, limit):
        value = float(params.get(name, code_defaults[name]))
        return f"{name}={value:g} ({value * limit:.3f} {'rad/s' if limit == JOINT_LIMIT_VELOCITY_RAD_S else 'rad/s^2'})"

    v, a = JOINT_LIMIT_VELOCITY_RAD_S, JOINT_LIMIT_ACCELERATION_RAD_S2
    return (f"speed_profile={profile}: contact {show('velocity_scaling', v)} {show('acceleration_scaling', a)}; "
            f"{show('transit_velocity_scaling', v)} {show('transit_acceleration_scaling', a)}")


TABLE_SCRIPT = "table_collision.py"


def table_script_installed():
    """True when table_collision.py is in this package's install. The launch files of a --symlink-install tree
    change with the source, the script only after a rebuild: without this check a stale install would abort the
    whole launch on 'executable not found'."""
    try:
        prefix = get_package_prefix(THIS)
    except PackageNotFoundError:
        return False
    return os.path.isfile(os.path.join(prefix, "lib", THIS, TABLE_SCRIPT))


def default_calibration_file():
    return os.path.join(get_package_share_directory("rebel_dual_table_description"), "config",
                        "dual_arm_calibration.yaml")


def load_arm_right_base(calibration_file):
    """(xyz, rpy) of arm_right_base_link in world from a dual_arm_calibration YAML, as plain float lists."""
    document = load_yaml(Path(calibration_file))
    try:
        base = document["dual_arm_calibration"]["arm_right_base"]
        xyz = [float(v) for v in base["xyz"]]
        rpy = [float(v) for v in base["rpy"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{calibration_file}: no usable dual_arm_calibration.arm_right_base xyz/rpy ({error})")
    if len(xyz) != 3 or len(rpy) != 3:
        raise RuntimeError(f"{calibration_file}: arm_right_base xyz and rpy need 3 values each")
    return xyz, rpy


def overrides_mismatches(arm, calibration_file, overrides):
    """Human-readable notes for snapshot_planning_base_* values in an overrides file that disagree with the
    calibration YAML (empty list when they agree or the file does not carry them)."""
    if arm != "right":
        return []
    xyz, rpy = load_arm_right_base(calibration_file)
    notes = []
    for name, value in (("snapshot_planning_base_xyz", xyz), ("snapshot_planning_base_rpy", rpy)):
        if name in overrides and [float(v) for v in overrides[name]] != value:
            notes.append(f"overrides file carries {name}={list(overrides[name])!r} but {calibration_file} says "
                         f"{value!r}; the calibration YAML wins (remove {name} from the overrides file)")
    return notes


def snapshot_transform_parameters(arm, calibration_file):
    """Parameters that carry the right base pose into the executor ({} for the left arm, which uses none).
    The calibration YAML is the single source."""
    if arm != "right":
        return {}
    xyz, rpy = load_arm_right_base(calibration_file)
    return {"snapshot_planning_base_xyz": xyz, "snapshot_planning_base_rpy": rpy}


def with_blade_disable_pairs(srdf_text, fragment_text):
    """Insert the blade SRDF fragment before the closing </robot> of the single-arm SRDF."""
    marker = "</robot>"
    index = srdf_text.rfind(marker)
    if index < 0 or BLADE_LINK not in fragment_text:
        raise RuntimeError("cannot insert the blade disable_collisions fragment into the SRDF")
    return srdf_text[:index] + fragment_text.rstrip("\n") + "\n" + srdf_text[index:]


BLADE_MIN_CLEARANCES_M = {"hover_clearance_m": 0.012, "left_lateral_standoff_m": 0.0165}


def blade_contact_links(executor_params):
    """carton_contact_links for the per-arm planner: the kinematic TCP link plus the blade box."""
    tool_link = executor_params.get("tool_link")
    if not isinstance(tool_link, str) or not tool_link:
        raise RuntimeError("the executor parameters need a tool_link to build carton_contact_links")
    return [tool_link, BLADE_LINK]


def load_executor_parameters(executor_config, overrides_file, arm=None, calibration_file=None,
                             blade_collision=False, speed_profile="default", speed_profile_file=None):
    base = node_parameters(load_yaml(Path(executor_config)), executor_config)
    overrides = {}
    if overrides_file:
        overrides = node_parameters(load_yaml(Path(overrides_file)), overrides_file)
    for name in ("execution_enabled", "debug_motion_execution_enabled"):
        if overrides.get(name) not in (None, False):
            raise RuntimeError(f"{overrides_file}: {name} must stay false in the per-arm planning stack")
    merged = dict(base)
    merged.update(overrides)
    # Applied AFTER the overrides: the calibration YAML wins (and a disagreeing overrides file is refused).
    merged.update(snapshot_transform_parameters(arm, calibration_file or default_calibration_file()))
    merged["execution_enabled"] = False
    merged["debug_motion_execution_enabled"] = False
    if blade_collision:
        # Launch-owned (read-only executor parameter): wins over the YAML and the overrides file.
        merged["carton_contact_links"] = blade_contact_links(merged)
        # The blade box sits ~1 mm from the carton at the 10 mm hover/standoff poses, so an OMPL approach transit
        # can graze it (2026-10-15, 1 of 3 strict 09:20 runs refused); 12 mm passed both captures.
        # left_lateral_standoff_m is measured from the cut line, which is left_cut_inward_m (7 mm) inside the wall,
        # so the left hover TCP is only standoff - 7 mm from the wall and the box reaches ~6 mm ahead of the TCP:
        # at 12 mm it overlapped the carton at the hover pose and the left face planned at 11 of 44 yaw cells.
        # 15 mm: 34 of 44, min free-space blade-box clearance 0.81 mm (scan_results/left_face_blade_20261018/).
        # 16.5 mm (2026-10-18 sweep 14..22 mm, 25/25/split settings): same windows as 15-16 mm, min clearance 1.77 mm;
        # 17 mm loses yaw -5 on 0749 (scan_results/standoff_sweep_20261018/).
        for name in BLADE_MIN_CLEARANCES_M:
            merged[name] = max(float(merged.get(name, 0.0)), BLADE_MIN_CLEARANCES_M[name])
    # LAST: the speed profile wins over the executor YAML and the overrides file ({} for `default`: nothing changes).
    merged.update(speed_profile_parameters(speed_profile, speed_profile_file))
    return merged


def launch_setup(context):
    arm = LaunchConfiguration("arm").perform(context).strip()
    if arm not in ("left", "right"):
        raise RuntimeError(f"arm must be left or right, got {arm!r}")
    ns = f"arm_{arm}"
    use_sim_time = LaunchConfiguration("use_sim_time")
    config_share = get_package_share_directory(MOVEIT_CONFIG)
    desc_share = get_package_share_directory(DESCRIPTION)
    this_share = get_package_share_directory(THIS)

    def cfg(name):
        return load_yaml(Path(config_share) / "config" / name)

    blade_collision = LaunchConfiguration("blade_collision").perform(context).strip().lower()
    if blade_collision not in ("true", "false"):
        raise RuntimeError(f"blade_collision must be true or false, got {blade_collision!r}")
    use_blade = blade_collision == "true"
    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        os.path.join(desc_share, "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro"),
        " hardware_protocol:=mock_hardware",
        " tool_variant:=", LaunchConfiguration("tool_variant"),
        " blade_collision:=", blade_collision,
    ]).perform(context)
    robot_description_semantic = (Path(config_share) / "config" / "igus_rebel2.srdf").read_text()
    if use_blade:
        robot_description_semantic = with_blade_disable_pairs(
            robot_description_semantic, (Path(config_share) / "config" / BLADE_SRDF_FRAGMENT).read_text())
    kinematics_config = cfg("kinematics.yaml")
    joint_limits_config = cfg("joint_limits.yaml")

    ompl_context = cfg("ompl_planning.yaml")
    ompl_pipeline = ompl_context["move_group"]
    if IS_HUMBLE:
        to_humble_pipeline(ompl_pipeline, HUMBLE_OMPL_REQUEST_ADAPTERS)
    executor_ompl_context = dict(ompl_pipeline)
    executor_ompl_context["planner_configs"] = ompl_context["planner_configs"]
    executor_ompl_context["igus_rebel_arm"] = ompl_context["igus_rebel_arm"]
    chomp_context = cfg("chomp_planning.yaml")
    pilz_context = cfg("pilz_industrial_motion_planner_planning.yaml")
    if IS_HUMBLE:
        to_humble_pipeline(chomp_context, HUMBLE_CHOMP_REQUEST_ADAPTERS)
        to_humble_pipeline(pilz_context, HUMBLE_PILZ_REQUEST_ADAPTERS)

    # Same dict as the single-arm launch, so the executor sees the same robot/pipelines.
    moveit_args = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics_config,
        "robot_description_planning": joint_limits_config,
        "moveit_simple_controller_manager": cfg("moveit_controllers.yaml"),
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": False,
        "publish_planning_scene_hz": 2.0,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.02,
        "trajectory_execution.execution_duration_monitoring": False,
        "capabilities": "move_group/ExecuteTaskSolutionCapability",
        "planning_pipelines": ["ompl", "chomp", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "ompl",
        "ompl": executor_ompl_context,
        "chomp": chomp_context,
        "pilz_industrial_motion_planner": pilz_context,
    }
    if not IS_HUMBLE:
        moveit_args["publish_robot_description"] = True
        moveit_args["publish_robot_description_semantic"] = True

    # Plan-only move_group: no trajectory execution, no controller manager, no execute capability.
    move_group_args = {k: v for k, v in moveit_args.items()
                       if k not in ("moveit_simple_controller_manager", "moveit_controller_manager",
                                    "capabilities")}
    move_group_args["allow_trajectory_execution"] = False
    move_group_node = Node(
        package="moveit_ros_move_group", executable="move_group", namespace=ns,
        parameters=[{"use_sim_time": use_sim_time}, move_group_args],
        # Isolate from the dual TF tree (also rooted at `world`); see the module docstring.
        remappings=[("/tf", f"/{ns}/unused_tf"), ("/tf_static", f"/{ns}/unused_tf_static")],
        output="screen")

    executor_config = LaunchConfiguration("executor_config").perform(context).strip()
    if not executor_config:
        executor_config = os.path.join(config_share, "config", "box_h_cut_executor.yaml")
    overrides = LaunchConfiguration("executor_overrides").perform(context).strip()
    if not overrides:
        overrides = os.path.join(this_share, "config", f"arm_{arm}_executor_overrides.yaml")
    calibration = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context).strip()) \
        or default_calibration_file()
    speed_profile = LaunchConfiguration("speed_profile", default="default").perform(context).strip()
    executor_params = load_executor_parameters(executor_config, overrides, arm, calibration, use_blade, speed_profile)
    override_params = node_parameters(load_yaml(Path(overrides)), overrides) if overrides else {}
    speed_notes = [f"[{THIS}] arm_{arm}: {speed_profile_summary(speed_profile, executor_params)}"]
    if speed_profile != "default":
        speed_notes += [f"[{THIS}] WARNING arm_{arm}: {overrides} sets {name}={override_params[name]!r}; the {speed_profile} "
                        f"speed profile wins ({executor_params[name]!r})"
                        for name in SPEED_SCALING_KEYS if name in override_params]
    overrides_notes = overrides_mismatches(
        arm, calibration, node_parameters(load_yaml(Path(overrides)), overrides) if overrides else {})

    # No name= here: it would add a process-wide `-r __node:=box_h_cut_executor` that also renames the helper
    # node MTC creates inside the executor process, leaving two nodes with the same name (clashing parameter
    # services). The executor names itself box_h_cut_executor in its constructor.
    executor_node = Node(
        package=MOVEIT_CONFIG, executable="box_h_cut_executor", namespace=ns,
        parameters=[executor_params, {"use_sim_time": use_sim_time}, moveit_args],
        remappings=[(topic, f"/{ns}{topic}") for topic in EXECUTOR_TOPICS],
        additional_env={"FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"},
        output="screen")
    delayed_executor = TimerAction(
        period=LaunchConfiguration("box_h_cut_start_delay"),
        actions=[executor_node],
        condition=IfCondition(LaunchConfiguration("run_executor")))

    camera_pole = Node(
        condition=IfCondition(LaunchConfiguration("run_camera_pole")),
        package=MOVEIT_CONFIG, executable="camera_pole_collision.py", name="camera_pole_collision",
        namespace=ns,
        parameters=[{"use_sim_time": use_sim_time,
                     "base_frame": f"arm_{arm}_base_link",
                     "output_frame_id": "base_link",
                     "camera_frame": "camera_link",
                     "optical_frame": "camera_color_optical_frame"}],
        remappings=[("/box_motion/camera_pole_collision", f"/{ns}/box_motion/camera_pole_collision")],
        output="screen")

    table_wanted = LaunchConfiguration("table_in_arm_scene").perform(context).strip().lower() == "true"
    table_missing = table_wanted and not table_script_installed()
    table = Node(
        condition=IfCondition(LaunchConfiguration("table_in_arm_scene")),
        package=THIS, executable=TABLE_SCRIPT, name="table_collision", namespace=ns,
        parameters=[{"use_sim_time": use_sim_time,
                     "arm": arm,
                     "calibration_file": calibration,   # the SAME file as the dual URDF / right base transform
                     "output_frame_id": "base_link",
                     "table_size": float(LaunchConfiguration("table_size").perform(context)),
                     "table_thickness": float(LaunchConfiguration("table_thickness").perform(context)),
                     "table_base_setback": float(LaunchConfiguration("table_base_setback").perform(context)),
                     "padding_m": float(LaunchConfiguration("table_padding_m").perform(context))}],
        output="screen")

    relay = Node(
        condition=IfCondition(LaunchConfiguration("run_relay")),
        package=THIS, executable="joint_state_relay.py", name="joint_state_relay", namespace=ns,
        parameters=[{"use_sim_time": use_sim_time, "arm": arm}], output="screen")

    actions = [LogInfo(msg=f"[{THIS}] arm_{arm}: executor parameters from {executor_config} + {overrides}"),
               *[LogInfo(msg=note) for note in speed_notes],
               move_group_node, relay, camera_pole, delayed_executor]
    if table_missing:
        actions.insert(0, LogInfo(msg=(
            f"[{THIS}] WARNING arm_{arm}: table_in_arm_scene:=true but {TABLE_SCRIPT} is not in the install of "
            f"{THIS} (rebuild the package); the table is NOT in this arm's planning scene, only the coordinator's "
            "final world check sees it")))
    elif table_wanted:
        actions.append(table)
    if arm == "right":
        actions.insert(0, LogInfo(msg=(
            f"[{THIS}] arm_right: snapshot_planning_base_xyz/rpy = arm_right_base of {calibration} "
            f"(xyz {executor_params['snapshot_planning_base_xyz']}, rpy {executor_params['snapshot_planning_base_rpy']}). "
            "Execution is disabled (plan_export_only).")))
    actions[0:0] = [LogInfo(msg=f"[{THIS}] WARNING arm_{arm}: {note}") for note in overrides_notes]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("arm", description="left or right"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("tool_variant", default_value="knife_c",
                              choices=["knife_c", "xeg32_knife_v1"],
                              description="single-arm tool model; must match the dual URDF tool"),
        DeclareLaunchArgument("blade_collision", default_value="true", choices=["true", "false"],
                              description="build the per-arm model with the knife_c_blade collision box and let the "
                                          "executor allow it (with the TCP link) to touch the carton during cut "
                                          "strokes; false = the previous stack (no blade link)"),
        DeclareLaunchArgument("executor_config", default_value="",
                              description="base executor YAML (default: igus_rebel_moveit_config "
                                          "config/box_h_cut_executor.yaml)"),
        DeclareLaunchArgument("executor_overrides", default_value="",
                              description="per-arm override YAML (default: this package's "
                                          "config/arm_<arm>_executor_overrides.yaml)"),
        DeclareLaunchArgument("calibration_file", default_value="",
                              description="dual_arm_calibration YAML (empty = rebel_dual_table_description default); "
                                          "source of the right arm's snapshot_planning_base_xyz/rpy"),
        DeclareLaunchArgument("speed_profile", default_value="default", choices=list(SPEED_PROFILES),
                              description="default = box_h_cut_executor.yaml speeds unchanged; fast = the opt-in profile of "
                                          "igus_rebel_moveit_config/config/speed_profile_fast.yaml (applied last, wins over "
                                          "the overrides file); NOT validated on the real arms"),
        DeclareLaunchArgument("run_executor", default_value="true"),
        DeclareLaunchArgument("run_camera_pole", default_value="true"),
        DeclareLaunchArgument("run_relay", default_value="true"),
        DeclareLaunchArgument("table_in_arm_scene", default_value="true", choices=["true", "false"],
                              description="put the dual cell's table into this arm's planning scene as the collision "
                                          "object `table` (false = the previous stack: only the coordinator's final "
                                          "world check sees the table)"),
        DeclareLaunchArgument("table_size", default_value="0.65",
                              description="table edge length; must equal the xacro arg table_size of "
                                          "rebel_dual_table.urdf.xacro"),
        DeclareLaunchArgument("table_thickness", default_value="0.04",
                              description="must equal the xacro arg table_thickness"),
        DeclareLaunchArgument("table_base_setback", default_value="0.110",
                              description="must equal the xacro arg table_base_setback"),
        DeclareLaunchArgument("table_padding_m", default_value="0.010",
                              description="grows the table box on every face (default 10 mm, = the coordinator's "
                                          "table_min_clearance_m); the base link is exempt (ACM), every other "
                                          "link keeps the margin; 0 = no margin"),
        DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
        OpaqueFunction(function=launch_setup),
    ])
