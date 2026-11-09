"""Round 5: cameras and the detector in another ROS domain (no ROS context is ever created here)."""
import inspect
import os
import sys

import pytest

from dual_cut_ui import server
from dual_cut_ui.camera import CameraHub, DEFAULT_CAMERAS
from dual_cut_ui.domains import (DomainPlan, check_domain_param, check_localhost_param, localhost_only_env, plan_domains,
                                 select_camera_source, startup_message)
from dual_cut_ui.jobs import ProcessRunner
from helpers import Env, FakeBridge, FakeClock


@pytest.mark.parametrize("node,cam,det,expect", [
    (12, -1, -1, DomainPlan(None, False, None)),                     # defaults: nothing changes
    (12, 11, -1, DomainPlan(11, True, 11)),                          # the real cell: cameras and detector in domain 11
    (12, 11, 173, DomainPlan(11, True, 173)),
    (12, 12, -1, DomainPlan(12, False, 12)),                         # same domain: no second context
    (12, -1, 40, DomainPlan(None, False, 40)),                       # detector only
    (0, 232, 0, DomainPlan(232, True, 0)),
])
def test_plan_domains(node, cam, det, expect):
    assert plan_domains(node, cam, det) == expect


@pytest.mark.parametrize("bad", [-2, 233, 1000, "11", 11.0, None, True])
def test_domain_params_are_validated(bad):
    with pytest.raises(ValueError):
        check_domain_param("camera_domain_id", bad)
    with pytest.raises(ValueError):
        plan_domains(12, bad, -1)
    with pytest.raises(ValueError):
        plan_domains(12, -1, bad)


def test_startup_message_names_the_camera_domain():
    assert startup_message(plan_domains(12, 11, -1), 12) == "cameras read from ROS domain 11 (read-only); coordinator in ROS domain 12"
    assert "own ROS domain 12" in startup_message(plan_domains(12, -1, -1), 12)


def test_camera_source_selection_only_builds_a_second_context_when_needed():
    main, made = object(), []
    cam = object()

    def factory(domain):
        made.append(domain)
        return cam
    assert select_camera_source(plan_domains(12, -1, -1), main, factory) is main
    assert select_camera_source(plan_domains(12, 12, -1), main, factory) is main
    assert made == []
    assert select_camera_source(plan_domains(12, 11, -1), main, factory) is cam
    assert made == [11]


def test_the_camera_hub_subscribes_through_the_camera_source_only():
    clock = FakeClock()
    main, camera = FakeBridge(clock), FakeBridge(clock)
    hub = CameraHub(select_camera_source(plan_domains(12, 11, -1), main, lambda d: camera), DEFAULT_CAMERAS, clock=clock)
    hub.request("scene", "small")
    assert camera.count("subscribe") == 1 and main.count("subscribe") == 0
    clock.advance(11)
    hub.sweep()
    assert camera.count("unsubscribe") == 1 and main.count("unsubscribe") == 0


def test_without_a_camera_domain_the_main_bridge_serves_the_cameras():
    clock = FakeClock()
    main = FakeBridge(clock)
    hub = CameraHub(select_camera_source(plan_domains(12, -1, -1), main, lambda d: pytest.fail("no second context expected")),
                    DEFAULT_CAMERAS, clock=clock)
    hub.request("left_gripper", "small")
    assert main.count("subscribe") == 1


# ---- the detector's environment --------------------------------------------------------------------------------------------
def env_probe(snapshot):
    return [sys.executable, "-c", "import os; print('DOM=' + str(os.environ.get('ROS_DOMAIN_ID')), 'LOC=' + str(os.environ.get('ROS_LOCALHOST_ONLY')), flush=True); "
            f"os.utime({snapshot!r})"]


def run_probe(tmp_path, monkeypatch, **cfg):
    monkeypatch.setenv("ROS_DOMAIN_ID", "12")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0")
    e = Env(tmp_path, **cfg)
    try:
        snap = str(tmp_path / "seam_3d_base_link.json")
        e.cfg.detect_command = env_probe(snap)
        e.bridge.push_status(state=0)
        assert e.act("detect")[0] == 200
        return e.wait_job()
    finally:
        e.close()


def test_the_detector_gets_the_configured_domain_and_keeps_localhost_only(tmp_path, monkeypatch):
    job = run_probe(tmp_path, monkeypatch, detect_domain_id=11)
    assert job.state == "done"
    assert "DOM=11 LOC=0" in job.log                                   # ROS_DOMAIN_ID overridden, ROS_LOCALHOST_ONLY as inherited
    running = next(line for line in job.log if line.startswith("[ui] running:"))
    assert running.endswith("ROS_DOMAIN_ID 11, ROS_LOCALHOST_ONLY 0)") and "cwd " in running


def test_the_detector_gets_the_configured_localhost_only(tmp_path, monkeypatch):
    job = run_probe(tmp_path, monkeypatch, detect_domain_id=11, detect_localhost_only=1)
    assert job.state == "done" and "DOM=11 LOC=1" in job.log
    assert next(line for line in job.log if line.startswith("[ui] running:")).endswith("ROS_DOMAIN_ID 11, ROS_LOCALHOST_ONLY 1)")
    assert os.environ["ROS_LOCALHOST_ONLY"] == "0"                     # the UI process keeps its own value


def test_without_an_override_the_detector_inherits(tmp_path, monkeypatch):
    job = run_probe(tmp_path, monkeypatch)
    assert "DOM=12 LOC=0" in job.log
    assert next(line for line in job.log if line.startswith("[ui] running:")).endswith("ROS_DOMAIN_ID 12, ROS_LOCALHOST_ONLY 0)")


def test_the_override_never_touches_the_ui_process_environment(tmp_path, monkeypatch):
    run_probe(tmp_path, monkeypatch, detect_domain_id=11)
    assert os.environ["ROS_DOMAIN_ID"] == "12" and os.environ["ROS_LOCALHOST_ONLY"] == "0"


def test_process_runner_env_overrides_are_applied_to_a_copy(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "12")
    r = ProcessRunner([sys.executable, "-c", "import os; print(os.environ['ROS_DOMAIN_ID'], os.environ.get('X_TEST'))"],
                      env_overrides={"ROS_DOMAIN_ID": "5", "X_TEST": "y"})
    assert r.run() == 0 and r.tail[-1] == "5 y" and os.environ["ROS_DOMAIN_ID"] == "12"


@pytest.mark.parametrize("value,expected", [(-1, None), (0, 0), (1, 1)])
def test_check_localhost_param_accepts(value, expected):
    assert check_localhost_param("camera_localhost_only", value) == expected


@pytest.mark.parametrize("value", [2, -2, "1", True, 1.0, None])
def test_check_localhost_param_rejects(value):
    with pytest.raises(ValueError):
        check_localhost_param("camera_localhost_only", value)


def test_localhost_only_env_sets_and_restores(monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    with localhost_only_env(0):
        assert os.environ["ROS_LOCALHOST_ONLY"] == "0"
    assert os.environ["ROS_LOCALHOST_ONLY"] == "1"
    monkeypatch.delenv("ROS_LOCALHOST_ONLY")
    with pytest.raises(RuntimeError):
        with localhost_only_env(0):
            assert os.environ["ROS_LOCALHOST_ONLY"] == "0"
            raise RuntimeError("boom")
    assert "ROS_LOCALHOST_ONLY" not in os.environ
    with localhost_only_env(None):
        assert "ROS_LOCALHOST_ONLY" not in os.environ


# ---- shutdown order --------------------------------------------------------------------------------------------------------
def test_shutdown_steps_put_the_camera_context_last(tmp_path):
    e = Env(tmp_path)
    try:
        calls = []
        core, hub, ctx = (type("C", (), {"shutdown": lambda self, budget_s=0: calls.append("core")})(),
                          type("H", (), {"shutdown": lambda self: calls.append("hub")})(),
                          type("X", (), {"shutdown": lambda self: calls.append("pools")})())
        steps = server.shutdown_steps(core, hub, ctx, lambda: calls.append("camera_ctx"))
        assert [n for n, _ in steps] == ["core (gates false, job stopped)", "cameras", "pools", "camera ROS context"]
        server.run_shutdown(steps, lambda m: None, budget_s=2.0)
        assert calls == ["core", "hub", "pools", "camera_ctx"]
        assert [n for n, _ in server.shutdown_steps(core, hub, ctx)] == ["core (gates false, job stopped)", "cameras", "pools"]
    finally:
        e.close()


def test_a_failing_camera_teardown_does_not_stop_the_shutdown():
    res = server.run_shutdown([("core", lambda: None), ("camera ROS context", lambda: 1 / 0)], lambda m: None, budget_s=1.0)
    assert res["core"] == "ok" and res["camera ROS context"].startswith("error")


def test_camera_context_teardown_order():
    pytest.importorskip("rclpy")
    from dual_cut_ui.ros_bridge import shutdown_camera_context
    calls = []

    class Rec:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, attr):
            return lambda *a, **k: calls.append(f"{self.name}.{attr}")
    shutdown_camera_context(Rec("executor"), Rec("thread"), Rec("node"), Rec("context"))
    assert calls == ["executor.shutdown", "thread.join", "node.destroy_node", "context.shutdown"]


def test_main_runs_the_camera_context_teardown_inside_run_shutdown_before_ros_goes_down():
    src = inspect.getsource(server.main)
    assert src.index("run_shutdown(shutdown_steps(") < src.index("executor.shutdown()") < src.rindex("rclpy.shutdown()")
    assert "camera_ctx[0].teardown" in src and "plan_domains(" in src and "select_camera_source(" in src


def test_the_camera_context_runs_nothing_but_subscriptions():
    pytest.importorskip("rclpy")
    from dual_cut_ui import ros_bridge
    text = inspect.getsource(ros_bridge.CameraSource) + inspect.getsource(ros_bridge.CameraContext)
    for forbidden in ("create_client", "create_service", "create_publisher", "declare_parameter", "set_parameters", "call_async",
                      "create_timer", "ActionClient"):
        assert forbidden not in text, forbidden
    assert "start_parameter_services=False" in text and "enable_rosout=False" in text and "domain_id=domain_id" in text
    assert "SignalHandlerOptions.NO" in text


def test_camera_source_subscribes_with_sensor_qos_on_its_own_node():
    pytest.importorskip("rclpy")
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from dual_cut_ui.ros_bridge import CameraSource
    seen = []

    class Node:
        def create_subscription(self, msg_type, topic, cb, qos, callback_group=None):
            seen.append((msg_type, topic, qos))
            return object()

        def destroy_subscription(self, sub):
            seen.append("destroyed")
    src = CameraSource(Node())
    src.subscribe_camera("scene", "/camera/color/image_raw/compressed", lambda b: None)
    src.subscribe_camera("scene", "/camera/color/image_raw/compressed", lambda b: None)          # idempotent
    src.unsubscribe_camera("scene")
    src.unsubscribe_camera("scene")
    assert seen == [(CompressedImage, "/camera/color/image_raw/compressed", qos_profile_sensor_data), "destroyed"]


def test_launch_arguments_for_the_domains():
    pytest.importorskip("launch_ros")
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launch", "dual_cut_ui.launch.py")
    spec = importlib.util.spec_from_file_location("dual_cut_ui_launch2", path)
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    args = {e.name: e for e in mod.generate_launch_description().entities if hasattr(e, "name")}
    for name in ("camera_domain_id", "detect_domain_id", "camera_localhost_only"):
        assert name in args and args[name].default_value[0].text == "-1"
    assert "int(arg(\"camera_domain_id\"))" in inspect.getsource(mod.launch_setup)
