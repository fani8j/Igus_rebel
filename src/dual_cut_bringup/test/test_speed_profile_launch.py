"""Speed profile (speed_profile:=default|fast) of the dual launch files: selection, precedence, default unchanged.

No process is started. Run: PYTEST_ADDOPTS="-p no:anyio" python3 -m pytest src/dual_cut_bringup/test/test_speed_profile_launch.py
"""
import importlib.util
import os
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
MOVEIT_CONFIG = os.path.join(SRC, "igus_rebel_moveit_config", "config")
BASE_YAML = os.path.join(MOVEIT_CONFIG, "box_h_cut_executor.yaml")
PROFILE_YAML = os.path.join(MOVEIT_CONFIG, "speed_profile_fast.yaml")
OVERRIDES = {arm: os.path.join(HERE, "..", "config", f"arm_{arm}_executor_overrides.yaml") for arm in ("left", "right")}
SCALINGS = ("velocity_scaling", "acceleration_scaling", "transit_velocity_scaling", "transit_acceleration_scaling")
DEFAULTS = {"velocity_scaling": 0.1376, "acceleration_scaling": 0.0763,
            "transit_velocity_scaling": 0.20, "transit_acceleration_scaling": 0.0763}
FAST = {"velocity_scaling": 0.2064, "acceleration_scaling": 0.0763,
        "transit_velocity_scaling": 0.40, "transit_acceleration_scaling": 0.0763}


def load_launch(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "launch", name + ".launch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planning = load_launch("per_arm_planning")
system = load_launch("dual_cut_system")


def params_of(path):
    return yaml.safe_load(open(path))["box_h_cut_executor"]["ros__parameters"]


def load(arm, profile=None, **kwargs):
    extra = {} if profile is None else {"speed_profile": profile}
    return planning.load_executor_parameters(BASE_YAML, OVERRIDES[arm], arm, planning.default_calibration_file(),
                                             True, speed_profile_file=PROFILE_YAML, **extra, **kwargs)


def test_shipped_yaml_values_are_the_documented_defaults():
    base = params_of(BASE_YAML)
    assert {k: base[k] for k in SCALINGS} == DEFAULTS


@pytest.mark.parametrize("arm", ["left", "right"])
def test_default_profile_is_byte_for_byte_the_old_merge(arm):
    # Independent re-implementation of the pre-profile merge: base, overrides, calibration, forced flags, blade.
    expected = dict(params_of(BASE_YAML))
    expected.update(params_of(OVERRIDES[arm]))
    expected.update(planning.snapshot_transform_parameters(arm, planning.default_calibration_file()))
    expected["execution_enabled"] = False
    expected["debug_motion_execution_enabled"] = False
    expected["carton_contact_links"] = planning.blade_contact_links(expected)
    for name, minimum in planning.BLADE_MIN_CLEARANCES_M.items():
        expected[name] = max(float(expected.get(name, 0.0)), minimum)
    assert load(arm) == expected                       # argument omitted
    assert load(arm, "default") == expected            # explicit
    for name in SCALINGS:
        assert load(arm, "default")[name] == DEFAULTS[name]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_fast_profile_changes_exactly_the_two_velocity_scalings(arm):
    # Since 2026-10-31 the default accelerations are already the 0.4 rad/s^2 cap (0.0763), so the FAST profile
    # only raises the two velocity scalings; it still sets all four values.
    default, fast = load(arm, "default"), load(arm, "fast")
    changed = {k for k in set(default) | set(fast) if default.get(k) != fast.get(k)}
    assert changed == {"velocity_scaling", "transit_velocity_scaling"}
    assert {k: fast[k] for k in SCALINGS} == FAST


def test_overrides_files_set_no_scaling_today():
    for path in OVERRIDES.values():
        assert not set(params_of(path)) & set(SCALINGS)


def test_overrides_cannot_undo_the_fast_profile(tmp_path):
    overrides = tmp_path / "o.yaml"
    overrides.write_text("box_h_cut_executor:\n  ros__parameters:\n" + "".join(f"    {k}: 0.05\n" for k in SCALINGS))
    base = planning.load_executor_parameters(BASE_YAML, str(overrides), "left", planning.default_calibration_file(),
                                             speed_profile_file=PROFILE_YAML)
    assert {k: base[k] for k in SCALINGS} == {k: 0.05 for k in SCALINGS}          # default: the override applies as before
    fast = planning.load_executor_parameters(BASE_YAML, str(overrides), "left", planning.default_calibration_file(),
                                             speed_profile="fast", speed_profile_file=PROFILE_YAML)
    assert {k: fast[k] for k in SCALINGS} == FAST                                  # fast: the profile wins


def test_unknown_profile_is_refused():
    with pytest.raises(RuntimeError, match="speed_profile"):
        planning.speed_profile_parameters("turbo")


@pytest.mark.parametrize("body, match", [
    ("box_h_cut_executor:\n  ros__parameters:\n    velocity_scaling: 0.2\n", "exactly"),
    ("", "exactly"),
    ("box_h_cut_executor:\n  ros__parameters:\n" + "".join(f"    {k}: 0.2\n" for k in SCALINGS[:3]) +
     "    transit_acceleration_scaling: 1.5\n", "in .0, 1."),
    ("box_h_cut_executor:\n  ros__parameters:\n" + "".join(f"    {k}: 0.2\n" for k in SCALINGS) + "    cut_pattern: h\n", "exactly"),
])
def test_bad_profile_file_fails_closed(tmp_path, body, match):
    path = tmp_path / "p.yaml"
    path.write_text(body)
    with pytest.raises(RuntimeError, match=match):
        planning.speed_profile_parameters("fast", str(path))


def test_fast_profile_respects_the_acceleration_cap_and_joint_limits():
    limits = yaml.safe_load(open(os.path.join(MOVEIT_CONFIG, "joint_limits.yaml")))["joint_limits"]
    for joint, entry in limits.items():
        v, a = entry["max_velocity"], entry["max_acceleration"]
        assert a * FAST["acceleration_scaling"] <= 0.4 + 1e-3, joint
        assert a * FAST["transit_acceleration_scaling"] <= 0.4 + 1e-3, joint
        assert v * FAST["transit_velocity_scaling"] == pytest.approx(0.314, abs=1e-3), joint


def test_summary_names_profile_and_physical_speeds():
    text = planning.speed_profile_summary("fast", planning.speed_profile_parameters("fast", PROFILE_YAML))
    assert "speed_profile=fast" in text and "0.314 rad/s" in text and "0.400 rad/s^2" in text and "0.162 rad/s" in text
    text = planning.speed_profile_summary("default", params_of(BASE_YAML))
    assert "speed_profile=default" in text and "0.157 rad/s" in text and "0.400 rad/s^2" in text


# --- launch plumbing ---

def default_of(module, name):
    args = {a.name: a for a in module.generate_launch_description().entities if isinstance(a, DeclareLaunchArgument)}
    return args[name].default_value[0].text, args[name].choices


def test_speed_profile_argument_everywhere_defaults_to_default():
    for module in (system, load_launch("dual_cut_full_mock"), load_launch("both_arms_planning"), planning):
        value, choices = default_of(module, "speed_profile")
        assert value == "default" and set(choices) == {"default", "fast"}


def test_both_arms_planning_forwards_the_profile_to_each_arm():
    both = load_launch("both_arms_planning")
    includes = [e for e in both.generate_launch_description().entities if isinstance(e, IncludeLaunchDescription)]
    assert len(includes) == 2 and all("speed_profile" in {k for k, _ in i.launch_arguments} for i in includes)
    import inspect
    assert '"speed_profile"' in inspect.getsource(load_launch("dual_cut_full_mock").generate_launch_description)
    assert '"speed_profile"' in inspect.getsource(system.launch_setup)


def per_arm_executor_scalings(arm, profile):
    from launch_ros.actions import Node
    planning_module = load_launch("per_arm_planning")
    planning_module.table_script_installed = lambda: True
    context = LaunchContext()
    context.launch_configurations.update({
        "arm": arm, "use_sim_time": "false", "tool_variant": "knife_c", "blade_collision": "true", "executor_config": "",
        "executor_overrides": "", "calibration_file": "", "run_executor": "true", "run_camera_pole": "true",
        "run_relay": "true", "box_h_cut_start_delay": "3.0", "table_in_arm_scene": "true", "table_size": "0.65",
        "table_thickness": "0.04", "table_base_setback": "0.110", "table_padding_m": "0.010"})
    if profile is not None:
        context.launch_configurations["speed_profile"] = profile
    actions = planning_module.launch_setup(context)
    from launch.actions import TimerAction
    timers = [a for a in actions if isinstance(a, TimerAction)]
    node = [n for t in timers for n in t.actions if isinstance(n, Node)][0]
    from launch.utilities import perform_substitutions

    def plain(v):   # launch_ros normalises parameter values to YAML text
        return yaml.safe_load(perform_substitutions(context, list(v))) if isinstance(v, tuple) else v
    executor_params = {}
    for k, v in node._Node__parameters[0].items():
        if plain(k) in SCALINGS:     # (other values, e.g. float lists, are tuples of plain numbers)
            executor_params[plain(k)] = plain(v)
    logs = [perform_substitutions(context, list(a.msg)) for a in actions if isinstance(a, LogInfo)]
    return {k: executor_params[k] for k in SCALINGS}, logs


@pytest.mark.parametrize("arm", ["left", "right"])
def test_per_arm_launch_setup_applies_and_logs_the_profile(arm):
    default, logs = per_arm_executor_scalings(arm, None)           # no configuration at all = default
    assert default == DEFAULTS and any("speed_profile=default" in m for m in logs)
    fast, logs = per_arm_executor_scalings(arm, "fast")
    assert fast == FAST
    assert any("speed_profile=fast" in m and "0.314 rad/s" in m for m in logs)


def test_system_default_does_not_touch_the_coordinator():
    path, lines = system.speed_profile_setup("default")
    assert path == "" and "default" in lines[0]


def test_system_fast_hands_the_coordinator_the_subchord_limits():
    path, lines = system.speed_profile_setup("fast")
    document = yaml.safe_load(open(path))
    assert Path(path).name == "speed_profile_fast.yaml"
    coordinator = document["dual_cut_coordinator"]["ros__parameters"]
    assert coordinator == {"cri_subchord_velocity_rad_s": 0.314, "cri_subchord_acceleration_rad_s2": 0.4}
    assert "speed_profile=fast" in lines[0] and "0.314" in lines[0]
    with pytest.raises(RuntimeError):
        system.speed_profile_setup("turbo")


def test_coordinator_launch_loads_the_profile_last():
    coordinator = load_launch_path(os.path.join(SRC, "dual_cut_coordinator", "launch", "coordinator.launch.py"))
    from launch_ros.actions import Node
    context = LaunchContext()
    context.launch_configurations.update({"config": "/c.yaml", "speed_profile_file": ""})
    node, = coordinator.coordinator_node(context)
    assert isinstance(node, Node) and len(node._Node__parameters) == 1
    context.launch_configurations["speed_profile_file"] = PROFILE_YAML
    node, = coordinator.coordinator_node(context)
    assert len(node._Node__parameters) == 2


def load_launch_path(path):
    spec = importlib.util.spec_from_file_location("coordinator_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinator_default_subchord_limits_are_the_default_transit_limits():
    # cri_subchord_* in coordinator.yaml (only used with cri_transit_enabled, default false) = the executors' default transit
    # limits read as fractions of joint_limits.yaml: 0.20 x 0.785 = 0.157 rad/s and 0.0763 x 5.24 = 0.400 rad/s^2 (the
    # 2026-10-31 planned-acceleration cap; was 0.10 x 5.24 = 0.524).
    coordinator = yaml.safe_load(open(os.path.join(SRC, "dual_cut_coordinator", "config", "coordinator.yaml")))
    p = coordinator["dual_cut_coordinator"]["ros__parameters"]
    assert p["cri_subchord_velocity_rad_s"] == pytest.approx(DEFAULTS["transit_velocity_scaling"] * 0.785, abs=1e-3)
    assert p["cri_subchord_acceleration_rad_s2"] == pytest.approx(DEFAULTS["transit_acceleration_scaling"] * 5.24, abs=1e-3)
    assert p["cri_subchord_acceleration_rad_s2"] <= 0.400 + 1e-9
