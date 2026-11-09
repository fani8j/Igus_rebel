"""Static checks of launch/dual_cut_system.launch.py (no processes are started)."""
import importlib.util
import os

import pytest
from launch.actions import DeclareLaunchArgument

_PATH = os.path.join(os.path.dirname(__file__), "..", "launch", "dual_cut_system.launch.py")
_spec = importlib.util.spec_from_file_location("dual_cut_system_launch", _PATH)
system = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(system)


def declared():
    return {a.name: a for a in system.generate_launch_description().entities
            if isinstance(a, DeclareLaunchArgument)}


def test_hardware_protocol_has_no_default():
    argument = declared()["hardware_protocol"]
    assert argument.default_value is None
    assert set(argument.choices) == {"mock_hardware", "rebel"}


def test_safe_defaults():
    args = declared()
    assert args["activate_controllers"].default_value[0].text == "auto"
    assert args["use_rviz"].default_value[0].text == "false"
    assert "execution_enabled" not in args    # enables are never launch arguments


def write(tmp_path, body):
    path = tmp_path / "coordinator.yaml"
    path.write_text(body)
    return str(path)


def test_coordinator_config_with_enabled_execution_is_refused(tmp_path):
    for name in system.ENABLE_PARAMS:
        path = write(tmp_path, f"dual_cut_coordinator:\n  ros__parameters:\n    {name}: true\n")
        with pytest.raises(RuntimeError, match=name):
            system.check_coordinator_config(path)


def test_coordinator_config_with_disabled_execution_is_accepted(tmp_path):
    system.check_coordinator_config(write(
        tmp_path, "dual_cut_coordinator:\n  ros__parameters:\n    execution_enabled: false\n    swept_padding_m: 0.05\n"))


def test_checklist_for_rebel_has_the_identity_steps():
    text = "\n".join(system.operator_checklist("rebel", "192.168.3.11", "192.168.3.12", "auto"))
    assert "pendant" in text and "dual_arm_recover.py activate" in text and "INACTIVE" in text
    assert "192.168.3.11" in text and "192.168.3.12" in text
    text = "\n".join(system.operator_checklist("rebel", "a", "b", "true"))
    assert "NO identity check" in text


def test_checklist_for_mock_is_short_and_mentions_mock():
    lines = system.operator_checklist("mock_hardware", "", "", "auto")
    assert "MOCK" in "\n".join(lines) and len(lines) <= 6


def ctx(**configs):
    from launch import LaunchContext
    context = LaunchContext()
    defaults = {"camera_domain_id": "auto", "run_side_cut_probe": "auto", "record_cut_video": "true",
                "cut_recorder_backend": "video", "robot_remote_dir": "~/robot_remote",
                "cut_video_cameras": "wrist", "cut_video_dir": "", "coordinator_delay": "10.0"}
    defaults.update(configs)
    context.launch_configurations.update(defaults)
    return context


def test_camera_domain_auto_is_11_on_rebel_and_same_in_mock():
    assert system.camera_domain(ctx(), "rebel") == 11
    assert system.camera_domain(ctx(), "mock_hardware") is None
    assert system.camera_domain(ctx(camera_domain_id="same"), "rebel") is None
    assert system.camera_domain(ctx(camera_domain_id="203"), "mock_hardware") == 203
    with pytest.raises(RuntimeError):
        system.camera_domain(ctx(camera_domain_id="250"), "rebel")   # Fast DDS refuses > 232


def cmd_of(action):
    return " ".join("".join(x.text for x in c) if isinstance(c, list) else getattr(c, "text", str(c)) for c in action.cmd)


def test_recorder_gets_the_camera_domain_on_rebel():
    (action,) = system.cut_recorder_actions(ctx(), "rebel")
    assert "--camera-domain 11" in cmd_of(action)


def test_side_cut_probe_defaults_rebel_only():
    assert system.side_cut_probe_actions(ctx(), "mock_hardware") == []
    assert system.side_cut_probe_actions(ctx(run_side_cut_probe="false"), "rebel") == []
    actions = system.side_cut_probe_actions(ctx(), "rebel")
    assert len(actions) == 1   # the delayed probe include, or a LogInfo when side_cut_vision is not installed


def test_vision_overlays_are_recorded_by_default():
    # empty = every camera of the chosen backend
    assert declared()["cut_video_cameras"].default_value[0].text == ""
    assert "left_vision" in system.VIDEO_CAMERAS.split(",") and "right_vision" in system.VIDEO_CAMERAS.split(",")
    assert "left_vision" in system.RAW_CAMERAS.split(",") and "right_vision" in system.RAW_CAMERAS.split(",")


# --- automatic cut recording: backend raw (default) / video, record_cut_video true | false | always ---

FAKE_SCRIPT = "/fake/robot_remote/recording/record_raw_cut.py"


@pytest.fixture
def raw_ok(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(system, "raw_ready", lambda d: (Path(FAKE_SCRIPT), ""))


def test_backend_and_recording_arguments():
    args = declared()
    assert args["cut_recorder_backend"].default_value[0].text == "raw"
    assert set(args["cut_recorder_backend"].choices) == {"raw", "video"}
    assert args["record_cut_video"].default_value[0].text == "true"
    assert set(args["record_cut_video"].choices) == {"true", "false", "always"}
    assert args["robot_remote_dir"].default_value[0].text == "~/robot_remote"


def test_recording_is_rebel_only_unless_always(raw_ok):
    for backend in ("raw", "video"):
        assert system.cut_recorder_actions(ctx(cut_recorder_backend=backend), "mock_hardware") == []
        assert system.cut_recorder_actions(ctx(cut_recorder_backend=backend, record_cut_video="false"), "rebel") == []
        assert system.cut_recorder_actions(ctx(cut_recorder_backend=backend, record_cut_video="false"), "mock_hardware") == []
        assert len(system.cut_recorder_actions(ctx(cut_recorder_backend=backend), "rebel")) >= 1
        assert len(system.cut_recorder_actions(ctx(cut_recorder_backend=backend, record_cut_video="always"),
                                               "mock_hardware")) >= 1


def test_raw_backend_command(raw_ok):
    (action,) = system.cut_recorder_actions(
        ctx(cut_recorder_backend="raw", cut_video_cameras="", camera_domain_id="214", record_cut_video="always",
            cut_video_dir="/clips"), "mock_hardware")
    cmd = cmd_of(action)
    assert cmd.startswith("nice -n 10 python3 " + FAKE_SCRIPT)
    for part in ("--trigger dual-cut", "--camera-domain 214", "--camera-keys " + system.RAW_CAMERAS, "--clip-dir /clips"):
        assert part in cmd, (part, cmd)
    for absent in ("lerobot", "record_dataset", "--root", "--fps", "--source"):
        assert absent not in cmd                           # no LeRobot, no dataset options
    assert not action.additional_env                       # no HF_HUB_OFFLINE needed either


def test_raw_backend_without_camera_domain_or_dirs(raw_ok):
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", cut_video_cameras=""), "rebel")
    assert "--camera-domain 11" in cmd_of(action)           # auto on rebel, as for the video recorder
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", record_cut_video="always",
                                                cut_video_cameras=""), "mock_hardware")
    cmd = cmd_of(action)
    assert "--camera-domain" not in cmd and "--clip-dir" not in cmd


def test_old_camera_names_still_work_with_both_backends(raw_ok):
    old = "knife,scene,wrist,right_wrist,left_vision,right_vision"
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", cut_video_cameras=old), "rebel")
    assert "--camera-keys knife,head,wrist_left_knife,wrist_right_knife,left_vision,right_vision" in cmd_of(action)
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="video", cut_video_cameras=old), "rebel")
    assert f"--cameras {old}" in cmd_of(action)
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="video", cut_video_cameras="head,wrist_left_gripper"), "rebel")
    assert "--cameras scene,left_gripper" in cmd_of(action)


def test_video_backend_is_the_old_recorder_unchanged():
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="video", cut_video_cameras=""), "rebel")
    cmd = cmd_of(action)
    assert "tools/cut_recorder/cut_recorder.py" in cmd
    assert "--coordinator-status-topic /dual_cut_coordinator/status --post-roll-s 5.0" in cmd
    assert f"--cameras {system.VIDEO_CAMERAS}" in cmd and "--camera-domain 11" in cmd


def test_raw_falls_back_to_video_and_never_fails(tmp_path, monkeypatch):
    from launch.actions import LogInfo
    missing = ctx(cut_recorder_backend="raw", robot_remote_dir=str(tmp_path / "nope"))
    actions = system.cut_recorder_actions(missing, "rebel")
    assert isinstance(actions[0], LogInfo) and "falling back" in actions[0].msg[0].text
    assert "cut_recorder.py" in cmd_of(actions[1])
    # robot_remote exists but python3 cannot import rosbag2_py
    (tmp_path / "recording").mkdir()
    (tmp_path / "recording" / "record_raw_cut.py").write_text("")

    class Failed:
        returncode = 1
    monkeypatch.setattr(system.subprocess, "run", lambda *a, **k: Failed())
    actions = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", robot_remote_dir=str(tmp_path)), "rebel")
    assert isinstance(actions[0], LogInfo) and "raw" in actions[0].msg[0].text
    class Ok:
        returncode = 0
    monkeypatch.setattr(system.subprocess, "run", lambda *a, **k: Ok())
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", robot_remote_dir=str(tmp_path)), "rebel")
    assert str(tmp_path / "recording" / "record_raw_cut.py") in cmd_of(action)


def test_signal_timeouts_for_the_raw_backend(raw_ok):
    (action,) = system.cut_recorder_actions(ctx(cut_recorder_backend="raw", cut_video_cameras=""), "rebel")
    assert action._ExecuteLocal__sigterm_timeout[0].text == "60"       # SIGINT -> wait -> SIGTERM -> wait -> SIGKILL
    assert action._ExecuteLocal__sigkill_timeout[0].text == "15"


# --- robot IP defaults (2026-10-20: the real-cell driver got "" because the parent's declared "" was inherited) ---

def resolve_driver(**configs):
    """Resolve the dual driver part of the launch description WITHOUT starting anything and return
    (robot_description xacro output of ros2_control_node, launch configurations seen while the driver launch ran).
    Nodes, timers and event handlers are not executed; everything else (includes, groups, declare/unset/set
    actions, OpaqueFunctions) is visited for real."""
    from launch import LaunchContext
    from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
    from launch_ros.actions import Node

    context = LaunchContext()
    context.launch_configurations.update(configs)
    seen = {}
    descriptions = []
    queue = list(system.generate_launch_description().entities)
    while queue:
        entity = queue.pop(0)
        if isinstance(entity, Node):   # (Node is an ExecuteProcess: never executed here)
            if entity.node_executable == "ros2_control_node":
                from launch.utilities import perform_substitutions
                robot_description = next(
                    value for p in entity._Node__parameters if isinstance(p, dict) for key, value in p.items()
                    if perform_substitutions(context, list(key)) == "robot_description")
                descriptions.append(perform_substitutions(context, list(robot_description.value)))
                seen.update(context.launch_configurations)
            continue
        if isinstance(entity, (TimerAction, RegisterEventHandler, ExecuteProcess)):
            continue
        sub = entity.visit(context)
        queue[0:0] = list(sub or [])
    assert len(descriptions) == 1, "expected exactly one ros2_control_node in the driver launch"
    return descriptions[0], seen


def robot_ips(urdf):
    import re
    return re.findall(r'<param name="robot_ip">([^<]*)</param>', urdf)


def run_resolve(monkeypatch, **configs):
    monkeypatch.setenv("ROS_DOMAIN_ID", "221")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    # Only what the operator types on the command line; every other value comes from the launch file defaults.
    base = {"hardware_protocol": "rebel"}
    base.update(configs)
    return resolve_driver(**base)


def test_driver_gets_its_own_ip_defaults_when_no_ip_is_given(monkeypatch):
    urdf, _ = run_resolve(monkeypatch)
    assert robot_ips(urdf) == ["192.168.3.11", "192.168.3.12"]


def test_driver_gets_explicit_ips(monkeypatch):
    urdf, _ = run_resolve(monkeypatch, left_robot_ip="10.0.0.5", right_robot_ip="10.0.0.6")
    assert robot_ips(urdf) == ["10.0.0.5", "10.0.0.6"]


def test_only_one_ip_given_the_other_keeps_the_driver_default(monkeypatch):
    urdf, _ = run_resolve(monkeypatch, left_robot_ip="10.0.0.5")
    assert robot_ips(urdf) == ["10.0.0.5", "192.168.3.12"]
    urdf, _ = run_resolve(monkeypatch, right_robot_ip="10.0.0.6")
    assert robot_ips(urdf) == ["192.168.3.11", "10.0.0.6"]


def jog_velocity_scales(urdf):
    import re
    return re.findall(r'<param name="jog_velocity_scale">([^<]*)</param>', urdf)


def test_driver_gets_jog_velocity_scale_1_65_for_both_arms_by_default(monkeypatch):
    # Real left arm 2026-10-31: 0.600 deg/s per jog % -> scale 1.65; the controllers.yaml gains assume it.
    urdf, _ = run_resolve(monkeypatch)
    assert jog_velocity_scales(urdf) == ["1.65", "1.65"]


def test_driver_jog_velocity_scale_can_be_set_per_arm(monkeypatch):
    urdf, _ = run_resolve(monkeypatch, left_jog_velocity_scale="1.7", right_jog_velocity_scale="1.6")
    assert jog_velocity_scales(urdf) == ["1.7", "1.6"]
    urdf, _ = run_resolve(monkeypatch, right_jog_velocity_scale="1.6")
    assert jog_velocity_scales(urdf) == ["1.65", "1.6"]


def test_mock_description_has_no_jog_velocity_scale(monkeypatch):
    urdf, _ = run_resolve(monkeypatch, hardware_protocol="mock_hardware")
    assert jog_velocity_scales(urdf) == []


def test_equal_ips_are_still_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="two different controller addresses"):
        run_resolve(monkeypatch, left_robot_ip="10.0.0.5", right_robot_ip="10.0.0.5")
    # one given equal to the other's DEFAULT is also two equal addresses
    with pytest.raises(RuntimeError, match="two different controller addresses"):
        run_resolve(monkeypatch, left_robot_ip="192.168.3.12")


def test_other_driver_arguments_still_reach_the_driver(monkeypatch, tmp_path):
    calibration = tmp_path / "calib.yaml"
    source = os.path.join(os.path.dirname(__file__), "..", "..", "rebel_dual_table_description", "config",
                          "dual_arm_calibration.yaml")
    calibration.write_text(open(source).read())
    _, seen = run_resolve(monkeypatch, calibration_file=str(calibration), activate_controllers="true")
    assert seen["hardware_protocol"] == "rebel"
    assert seen["calibration_file"] == str(calibration)
    assert seen["activate_controllers"] == "true"
    assert seen["left_robot_ip"] == "192.168.3.11" and seen["right_robot_ip"] == "192.168.3.12"


def test_driver_include_is_scoped_and_unsets_only_the_empty_arguments():
    """The unset sits between a push and a pop of the launch configurations, so the parent's values are back
    afterwards (the checklist and later includes still see what the operator gave)."""
    from launch import LaunchContext
    group = system.driver_include({"hardware_protocol": "mock_hardware"}, ["left_robot_ip"])
    entities = list(group.visit(LaunchContext()))
    names = [type(e).__name__ for e in entities]
    assert names[0] == "PushLaunchConfigurations" and names[-1] == "PopLaunchConfigurations"
    assert names.count("UnsetLaunchConfiguration") == 1
    assert names.index("UnsetLaunchConfiguration") < names.index("IncludeLaunchDescription")


def test_checklist_shows_the_driver_defaults_when_no_ip_is_given():
    import inspect
    source = inspect.getsource(system.launch_setup)
    assert "192.168.3.11 (dual_table.launch.py default)" in source


# --- the dual move_group must not execute trajectories (RViz Plan & Execute bypassed the coordinator) ---

def move_group_parameters(**configs):
    """Parameters of the move_group Node of rebel_dual_table_moveit_config move_group.launch.py, resolved without
    starting anything. {name: value} of the dict parameters with the substitution keys performed."""
    from launch import LaunchContext
    from launch.actions import ExecuteProcess
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    import yaml
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory("rebel_dual_table_moveit_config"), "launch", "move_group.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(configs)
    queue = list(PythonLaunchDescriptionSource(path).get_launch_description(context).entities)
    found = {}
    while queue:
        entity = queue.pop(0)
        if isinstance(entity, Node):
            if entity.node_executable == "move_group":
                for p in entity._Node__parameters:
                    if isinstance(p, dict):
                        found.update({perform_substitutions(context, list(k)): v for k, v in p.items()})
            continue
        if isinstance(entity, ExecuteProcess):
            continue
        queue[0:0] = list(entity.visit(context) or [])
    assert found, "no move_group node"
    return found


def test_system_launch_makes_the_dual_move_group_plan_only_by_default():
    assert declared()["allow_move_group_execution"].default_value[0].text == "false"
    import inspect
    assert '"allow_trajectory_execution": LaunchConfiguration("allow_move_group_execution")' in \
        inspect.getsource(system.launch_setup)


def test_dual_move_group_plan_only_has_no_controller_manager():
    params = move_group_parameters(allow_trajectory_execution="false")
    assert params["allow_trajectory_execution"] is False
    assert not any(k.startswith(("moveit_controller_manager", "moveit_simple_controller_manager")) for k in params)
    assert "robot_description" in params and "planning_pipelines" in params   # planning stays configured


def test_standalone_dual_move_group_keeps_its_execution_default():
    params = move_group_parameters()
    assert "allow_trajectory_execution" not in params
    assert "".join(sub.text for sub in params["moveit_controller_manager"]).startswith(
        "moveit_simple_controller_manager/MoveItSimpleControllerManager")
    assert any(k.startswith("moveit_simple_controller_manager.") for k in params)


# --- the table in the per-arm planning scenes (table_collision.py) ---

def load_launch(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "..", "launch", name + ".launch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_of(module, name):
    args = {a.name: a for a in module.generate_launch_description().entities if isinstance(a, DeclareLaunchArgument)}
    return args[name].default_value[0].text


def test_table_in_arm_scene_defaults_true_and_padding_5mm_everywhere():
    for module in (system, load_launch("dual_cut_full_mock"), load_launch("both_arms_planning"),
                   load_launch("per_arm_planning")):
        assert default_of(module, "table_in_arm_scene") == "true"
        assert default_of(module, "table_padding_m") == "0.010"


def test_table_args_are_passed_through_to_every_arm():
    import inspect
    assert '"table_in_arm_scene"' in inspect.getsource(system.launch_setup)
    assert '"table_padding_m"' in inspect.getsource(system.launch_setup)
    assert '"table_in_arm_scene"' in inspect.getsource(load_launch("dual_cut_full_mock").generate_launch_description)
    both = load_launch("both_arms_planning")
    from launch.actions import IncludeLaunchDescription
    includes = [e for e in both.generate_launch_description().entities if isinstance(e, IncludeLaunchDescription)]
    assert len(includes) == 2
    for include_action in includes:
        names = {k for k, _ in include_action.launch_arguments}
        assert {"table_in_arm_scene", "table_padding_m", "arm"} <= names


def per_arm_actions(arm, installed=True, **configs):
    from launch import LaunchContext
    planning = load_launch("per_arm_planning")
    planning.table_script_installed = lambda: installed
    context = LaunchContext()
    values = {"arm": arm, "use_sim_time": "false", "tool_variant": "knife_c", "blade_collision": "true",
              "executor_config": "", "executor_overrides": "", "calibration_file": "", "run_executor": "true",
              "run_camera_pole": "true", "run_relay": "true", "box_h_cut_start_delay": "3.0",
              "table_in_arm_scene": "true", "table_size": "0.65", "table_thickness": "0.04",
              "table_base_setback": "0.110", "table_padding_m": "0.010"}
    values.update(configs)
    context.launch_configurations.update(values)
    return context, planning.launch_setup(context)


def per_arm_nodes(arm, **configs):
    from launch_ros.actions import Node
    context, actions = per_arm_actions(arm, **configs)
    return context, [a for a in actions if isinstance(a, Node)]


def test_per_arm_launch_adds_one_table_node_per_arm_with_the_calibration_file():
    import yaml
    from launch.utilities import perform_substitutions
    for arm in ("left", "right"):
        context, nodes = per_arm_nodes(arm)
        tables = [n for n in nodes if n.node_executable == "table_collision.py"]
        assert len(tables) == 1
        def plain(v):
            # launch_ros normalises parameter values to YAML text
            return yaml.safe_load(perform_substitutions(context, list(v))) if isinstance(v, tuple) else v
        params = {plain(k): plain(v) for k, v in tables[0]._Node__parameters[0].items()}
        assert params["arm"] == arm and params["output_frame_id"] == "base_link"
        assert params["calibration_file"].endswith("dual_arm_calibration.yaml")
        assert params["table_size"] == 0.65 and params["table_thickness"] == 0.04
        assert params["padding_m"] == 0.010
        assert tables[0].condition is not None
        assert tables[0].condition.evaluate(context) is True


def test_table_in_arm_scene_false_turns_the_node_off():
    context, nodes = per_arm_nodes("left", table_in_arm_scene="false")
    assert not [n for n in nodes if n.node_executable == "table_collision.py"]
    # the other nodes of the stack are unchanged
    assert {n.node_executable for n in nodes} >= {"move_group", "camera_pole_collision.py", "joint_state_relay.py"}


def test_stale_install_without_the_script_warns_instead_of_aborting_the_launch():
    from launch.actions import LogInfo
    from launch_ros.actions import Node
    context, actions = per_arm_actions("right", installed=False)
    assert not [a for a in actions if isinstance(a, Node) and a.node_executable == "table_collision.py"]
    text = " ".join(a.msg[0].text if isinstance(a.msg, list) else str(a.msg) for a in actions if isinstance(a, LogInfo))
    assert "table_collision.py is not in the install" in text and "NOT in this arm's planning scene" in text
    # everything else still starts
    assert {a.node_executable for a in actions if isinstance(a, Node)} >= {"move_group", "joint_state_relay.py"}


def test_table_collision_script_is_installed():
    cmake = open(os.path.join(os.path.dirname(__file__), "..", "CMakeLists.txt")).read()
    assert "scripts/table_collision.py" in cmake
    assert os.access(os.path.join(os.path.dirname(__file__), "..", "scripts", "table_collision.py"), os.X_OK)
