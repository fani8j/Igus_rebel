"""dual_teleop.launch.py: servo_relay wiring (Servo -> relay -> JTC vs the direct Servo -> JTC wiring).

Needs a sourced ROS environment (launch_ros, launch_param_builder, moveit_servo, the dual-table packages); skipped
otherwise. The launch's Node action is replaced by a recorder, so nothing is started and xacro is not run.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("launch_ros")
pytest.importorskip("launch_param_builder")
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory  # noqa: E402
from launch import LaunchContext  # noqa: E402
from launch.actions import DeclareLaunchArgument  # noqa: E402

LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "dual_teleop.launch.py"
PLUGIN = "rebel_servo_relay::PassThroughFilterPlugin"
BUTTERWORTH = "online_signal_smoothing::ButterworthFilterPlugin"

for _pkg in ("moveit_servo", "rebel_dual_table_description", "rebel_dual_table_moveit_config", "dual_teleop"):
    try:
        get_package_share_directory(_pkg)
    except PackageNotFoundError:
        pytest.skip(f"package {_pkg} not found (source the workspace)", allow_module_level=True)


class RecNode:
    def __init__(self, **kw):
        self.kw = kw


class FakeCommand:
    def __init__(self, *a, **k):
        pass

    def perform(self, context):
        return "<robot name='fake'/>"


def expand(monkeypatch, **overrides):
    spec = importlib.util.spec_from_file_location("dual_teleop_launch_under_test", LAUNCH_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "Node", RecNode)
    monkeypatch.setattr(mod, "Command", FakeCommand)
    ld = mod.generate_launch_description()
    context = LaunchContext()
    for ent in ld.entities:
        if isinstance(ent, DeclareLaunchArgument):
            context.launch_configurations[ent.name] = ent.default_value[0].perform(context) \
                if ent.default_value else ""
    context.launch_configurations["hardware_protocol"] = "mock_hardware"
    for k, v in overrides.items():
        assert k in context.launch_configurations, k
        context.launch_configurations[k] = str(v)
    return mod, mod.launch_setup(context)


def by_pkg(actions, package, executable):
    return [a for a in actions if a.kw.get("package") == package and a.kw.get("executable") == executable]


def servo_nodes(actions):
    nodes = [a for a in actions if a.kw.get("package") == "moveit_servo"]
    assert sorted(a.kw["namespace"] for a in nodes) == ["/arm_left/servo", "/arm_right/servo"]
    return {a.kw["namespace"].split("/")[1][4:]: a for a in nodes}


def servo_params(node):
    (p,) = [d["moveit_servo"] for d in node.kw["parameters"] if isinstance(d, dict) and "moveit_servo" in d]
    return p


def test_relay_true_is_default_and_wires_servo_to_relay_to_jtc(monkeypatch):
    _, actions = expand(monkeypatch)
    servos = servo_nodes(actions)
    relays = by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay")
    assert len(relays) == 2
    for side in ("left", "right"):
        sp = servo_params(servos[side])
        assert sp["command_out_topic"] == f"/arm_{side}/servo/trajectory_out"
        assert sp["smoothing_filter_plugin_name"] == PLUGIN
        (relay,) = [r for r in relays if r.kw["namespace"] == f"/arm_{side}/servo"]
        assert relay.kw["name"] == "trajectory_relay"
        (rp,) = relay.kw["parameters"]
        assert rp["input_topic"] == sp["command_out_topic"]
        assert rp["output_topic"] == f"/arm_{side}_trajectory_controller/joint_trajectory"
        assert rp["joint_states_topic"] == sp["joint_topic"] == "/joint_states"
        assert rp["publish_period_s"] == sp["publish_period"] == pytest.approx(0.034)
        assert rp["robot_description_node"] == "/robot_state_publisher"
        assert rp["max_lead_rad"] == pytest.approx(0.05)
        assert rp["lead_time_s"] == pytest.approx(0.2)
    # the two relays publish to different JTCs
    assert len({r.kw["parameters"][0]["output_topic"] for r in relays}) == 2


def test_relay_false_is_the_direct_wiring(monkeypatch):
    _, actions = expand(monkeypatch, servo_relay="false")
    assert by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay") == []
    for side, node in servo_nodes(actions).items():
        sp = servo_params(node)
        assert sp["command_out_topic"] == f"/arm_{side}_trajectory_controller/joint_trajectory"
        assert sp["smoothing_filter_plugin_name"] == BUTTERWORTH


def test_relay_true_and_false_servo_params_differ_only_in_topic_and_plugin(monkeypatch):
    _, on = expand(monkeypatch, servo_relay="true")
    _, off = expand(monkeypatch, servo_relay="false")
    for side in ("left", "right"):
        a, b = servo_params(servo_nodes(on)[side]), servo_params(servo_nodes(off)[side])
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff == {"command_out_topic", "smoothing_filter_plugin_name"}


def test_relay_follows_run_servo(monkeypatch):
    _, actions = expand(monkeypatch, run_servo="false")
    for a in by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay") + \
            [a for a in actions if a.kw.get("package") == "moveit_servo"]:
        assert a.kw["condition"] is not None  # both are gated by the same run_servo condition


def test_relay_lead_args_are_passed_through(monkeypatch):
    _, actions = expand(monkeypatch, relay_max_lead_rad="0.15", relay_lead_time_s="0.6")
    for r in by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay"):
        assert r.kw["parameters"][0]["max_lead_rad"] == pytest.approx(0.15)
        assert r.kw["parameters"][0]["lead_time_s"] == pytest.approx(0.6)


@pytest.mark.parametrize("bad", ["maybe", "", "1", "True1"])
def test_invalid_servo_relay_raises(monkeypatch, bad):
    with pytest.raises(RuntimeError, match="servo_relay"):
        expand(monkeypatch, servo_relay=bad)


@pytest.mark.parametrize("name,bad", [
    ("relay_max_lead_rad", "0"), ("relay_max_lead_rad", "-0.01"), ("relay_max_lead_rad", "0.2"),
    ("relay_max_lead_rad", "nan"), ("relay_max_lead_rad", "abc"), ("relay_max_lead_rad", "inf"),
    ("relay_lead_time_s", "-0.1"), ("relay_lead_time_s", "1.5"), ("relay_lead_time_s", "nan"),
    ("relay_lead_time_s", "x"),
])
@pytest.mark.parametrize("relay", ["true", "false"])
def test_invalid_lead_args_raise(monkeypatch, name, bad, relay):
    with pytest.raises(RuntimeError, match=name):
        expand(monkeypatch, servo_relay=relay, **{name: bad})


def test_boundary_lead_values_are_accepted(monkeypatch):
    _, actions = expand(monkeypatch, relay_max_lead_rad="0.15", relay_lead_time_s="0")
    assert len(by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay")) == 2
    _, actions = expand(monkeypatch, relay_max_lead_rad="0.001", relay_lead_time_s="1.0")
    assert len(by_pkg(actions, "rebel_servo_relay", "servo_trajectory_relay")) == 2
