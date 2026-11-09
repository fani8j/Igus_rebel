"""The MOCK ONLY helper scripts: safety refusals and what mock_detect writes (no ROS node is ever started here)."""
import importlib.util
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from dual_cut_ui.snapshot import find_overlay, load_snapshot, summarize
from helpers import MOCK_DETECT, RECORDED_SNAPSHOT, SCRIPTS

CAM_SCRIPT = os.path.join(SCRIPTS, "mock_camera_publisher.py")
REAL_SNAPSHOT = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True          # no __pycache__ next to the installed scripts
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


cam = load_module(CAM_SCRIPT, "mock_camera_publisher")


def run(script, *args, env=None):
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True, timeout=60, env=env)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROS_")}
    env.update(extra)
    return env


@pytest.mark.parametrize("env,ok", [
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "173"}, True),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "40"}, True),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "11"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "12"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "219"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "225"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "232"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "218"}, True),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "233"}, True),
    ({"ROS_LOCALHOST_ONLY": "1"}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": ""}, False),
    ({"ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "abc"}, False),
    ({"ROS_DOMAIN_ID": "173"}, False),
    ({"ROS_LOCALHOST_ONLY": "0", "ROS_DOMAIN_ID": "173"}, False),
    ({}, False),
])
def test_camera_publisher_environment_check(env, ok):
    assert (cam.check_environment(env) is None) is ok


@pytest.mark.parametrize("env", [clean_env(), clean_env(ROS_LOCALHOST_ONLY="1"), clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="11"),
                                 clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="12"), clean_env(ROS_DOMAIN_ID="173"),
                                 clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="222")])
def test_camera_publisher_refuses_to_start(env):
    r = run(CAM_SCRIPT, env=env)
    assert r.returncode == 2 and "refusing to run" in r.stderr


def test_mock_detect_refuses_the_real_snapshot_and_capture_dir(tmp_path):
    before = os.stat(REAL_SNAPSHOT).st_mtime_ns if os.path.exists(REAL_SNAPSHOT) else None
    r = run(MOCK_DETECT, "--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", str(tmp_path), "--planner-output-json", REAL_SNAPSHOT, "--sleep", "0")
    assert r.returncode == 3 and "REAL snapshot" in r.stderr
    link = tmp_path / "link.json"
    link.symlink_to(REAL_SNAPSHOT)
    r = run(MOCK_DETECT, "--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", str(tmp_path), "--planner-output-json", str(link), "--sleep", "0")
    assert r.returncode == 3
    r = run(MOCK_DETECT, "--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", "/home/farhan/Igus_rebel/outputs/rgbd_captures",
            "--planner-output-json", str(tmp_path / "x.json"), "--sleep", "0")
    assert r.returncode == 3 and "real capture directory" in r.stderr and not (tmp_path / "x.json").exists()
    after = os.stat(REAL_SNAPSHOT).st_mtime_ns if os.path.exists(REAL_SNAPSHOT) else None
    assert before == after


def test_mock_detect_writes_a_fresh_snapshot_and_a_findable_overlay(tmp_path):
    snap, out = tmp_path / "seam.json", tmp_path / "out"
    t0 = time.time()
    r = run(MOCK_DETECT, "--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", str(out), "--planner-output-json", str(snap),
            "--autolabel-v2", "--allow-needs-review", "--allow-cut-blocked", "--target-frame", "base_link", "--sleep", "0.3")
    assert r.returncode == 0 and "MOCK detector" in r.stdout
    doc, raw, mtime = load_snapshot(str(snap))
    assert abs(mtime - time.time()) < 5 and doc["acquisition_stamp_ns"] * 1e-9 >= t0 - 1
    s = summarize(doc, mtime, time.time(), 240.0)
    assert s["fresh"] and s["age_s"] < 10
    recorded = json.load(open(RECORDED_SNAPSHOT))
    assert recorded["acquisition_stamp_ns"] < doc["acquisition_stamp_ns"] and doc["seam_start"] == recorded["seam_start"]
    found = find_overlay(str(out), str(snap))
    assert found and found["overlay"].endswith(f"{doc['capture_id']}_autolabel_v2.png")
    img = cv2.imread(found["overlay"])
    assert img is not None and img.shape[0] > 100


def test_mock_detect_uses_a_given_overlay_and_can_simulate_failures(tmp_path):
    png = tmp_path / "given.png"
    cv2.imwrite(str(png), np.full((50, 60, 3), 200, np.uint8))
    snap, out = tmp_path / "seam.json", tmp_path / "out"
    base = ["--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", str(out), "--planner-output-json", str(snap), "--sleep", "0"]
    assert run(MOCK_DETECT, *base, "--overlay-png", str(png)).returncode == 0
    found = find_overlay(str(out), str(snap))
    assert cv2.imread(found["overlay"]).shape[:2] == (50, 60)
    snap.unlink()
    assert run(MOCK_DETECT, *base, "--exit-code", "2").returncode == 2 and not snap.exists()


def test_the_scripts_are_installed_as_mock_only_names():
    text = open(os.path.join(os.path.dirname(SCRIPTS), "setup.py")).read()
    assert 'glob("scripts/mock_*.py")' in text
    for f in (f for f in os.listdir(SCRIPTS) if f.endswith(".py")):
        assert f.startswith("mock_")
        assert "MOCK ONLY" in open(os.path.join(SCRIPTS, f)).read(400)


def test_ros_bridge_module_imports_when_ros_is_available():
    pytest.importorskip("rclpy")
    pytest.importorskip("dual_cut_msgs.msg")
    from dual_cut_ui import ros_bridge, bridge as B
    for name in ("status", "add_status_listener", "plan_report", "exec_reports", "health", "gates", "trigger", "set_gates",
                 "set_schedule_mode", "detect", "cancel_all", "subscribe_camera", "unsubscribe_camera"):
        assert getattr(ros_bridge.RosBridge, name) is not getattr(B.Bridge, name)       # every interface method is implemented
    assert set(ros_bridge.STATUS_FIELDS) >= {"state", "plan_seq", "plan_retained", "phase_validated", "stop_latched", "stop_reason"}


def test_launch_file_builds():
    pytest.importorskip("launch_ros")
    mod = load_module(os.path.join(os.path.dirname(SCRIPTS), "launch", "dual_cut_ui.launch.py"), "dual_cut_ui_launch")
    ld = mod.generate_launch_description()
    names = {e.name for e in ld.entities if hasattr(e, "name")}
    assert {"port", "snapshot_file", "detect_cwd", "detect_output_dir", "detect_command", "cameras", "plan_timeout_s",
            "detect_mode", "detect_action", "detect_timeout_s", "schedule_mode"} <= names
    args = {e.name: e for e in ld.entities if hasattr(e, "name")}
    defaults = {n: args[n].default_value[0].text for n in ("detect_mode", "detect_action", "detect_timeout_s", "schedule_mode")}
    assert defaults == {"detect_mode": "action", "detect_action": "/box_seam_detector/detect_box", "detect_timeout_s": "300",
                        "schedule_mode": "sequence"}


# ---------------------------------------------------------------- mock_detect_action_server.py (never started as a node here)
ACTION_SERVER = os.path.join(SCRIPTS, "mock_detect_action_server.py")
srv = load_module(ACTION_SERVER, "mock_detect_action_server")


class Args:
    def __init__(self, tmp_path, reject=False, output_dir=""):
        self.source, self.snapshot = RECORDED_SNAPSHOT, str(tmp_path / "seam.json")
        self.output_dir, self.reject, self.step_s = output_dir, reject, 0.0


class Goal:
    dry_run = False


def test_mock_action_server_reuses_the_environment_check():
    for env in (clean_env(), clean_env(ROS_LOCALHOST_ONLY="1"), clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="11"),
                clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="12"), clean_env(ROS_DOMAIN_ID="173"),
                clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="219"), clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="232")):
        r = run(ACTION_SERVER, "--source", RECORDED_SNAPSHOT, "--snapshot", "/tmp/never_written.json", env=env)
        assert r.returncode == 2 and "refusing to run" in r.stderr, (env, r.stderr)
    assert not os.path.exists("/tmp/never_written.json")


def test_mock_action_server_refuses_the_real_snapshot():
    env = clean_env(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="173")
    r = run(ACTION_SERVER, "--source", RECORDED_SNAPSHOT, "--snapshot", REAL_SNAPSHOT, env=env)
    assert r.returncode == 3 and "REAL snapshot" in r.stderr
    r = run(ACTION_SERVER, "--source", RECORDED_SNAPSHOT, "--snapshot", "/home/farhan/Igus_rebel/outputs/other.json", env=env)
    assert r.returncode == 3


def test_mock_action_server_success_writes_a_fresh_snapshot_after_feedback(tmp_path):
    seen = []
    t0 = time.time()
    res = srv.run_goal(Goal(), Args(tmp_path, output_dir=str(tmp_path / "out")), lambda *a: seen.append(a), lambda: False)
    assert [s[0] for s in seen] == ["capture", "detect", "geometry", "snapshot"] and all(s[1] == 1 for s in seen)
    assert res["success"] is True and res["status"] == "accepted" and res["snapshot_path"] == str(tmp_path / "seam.json")
    doc, raw, mtime = load_snapshot(res["snapshot_path"])
    assert doc["acquisition_stamp_ns"] * 1e-9 >= t0 - 1 and abs(mtime - time.time()) < 5
    assert doc["seam_start"] == json.load(open(RECORDED_SNAPSHOT))["seam_start"]
    assert find_overlay(str(tmp_path / "out"), res["snapshot_path"])


def test_mock_action_server_reject_writes_nothing(tmp_path):
    res = srv.run_goal(Goal(), Args(tmp_path, reject=True), lambda *a: None, lambda: False)
    assert res["success"] is False and res["status"] == "rejected" and not (tmp_path / "seam.json").exists()


def test_mock_action_server_dry_run_writes_nothing(tmp_path):
    g = Goal()
    g.dry_run = True
    res = srv.run_goal(g, Args(tmp_path), lambda *a: None, lambda: False)
    assert res["success"] is True and res["snapshot_path"] == "" and not (tmp_path / "seam.json").exists()


def test_mock_action_server_honours_cancel(tmp_path):
    n = [0]

    def cancelled():
        n[0] += 1
        return n[0] > 2
    res = srv.run_goal(Goal(), Args(tmp_path), lambda *a: None, cancelled)
    assert res["status"] == "cancelled" and res["success"] is False and not (tmp_path / "seam.json").exists()


def test_mock_action_server_source_check_in_the_script_text():
    text = open(ACTION_SERVER).read()
    assert "MOCK ONLY" in text[:200] and "box-seam-detection/install/setup.bash" in text and "busy" in text


@pytest.mark.parametrize("schedule,expected", [("sequence", "sequence"), ("", ""), ("phases", "phases")])
def test_launch_passes_the_detect_and_schedule_parameters(schedule, expected):
    pytest.importorskip("launch_ros")
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    mod = load_module(os.path.join(os.path.dirname(SCRIPTS), "launch", "dual_cut_ui.launch.py"), "dual_cut_ui_launch3")
    ctx = LaunchContext()
    for e in mod.generate_launch_description().entities:
        if hasattr(e, "name"):
            ctx.launch_configurations[e.name] = e.default_value[0].text if e.default_value else ""
    ctx.launch_configurations["schedule_mode"] = schedule
    node = mod.launch_setup(ctx)[0]
    params = getattr(node, "_Node__parameters", None)
    if params is None:
        pytest.skip("launch_ros Node does not expose its parameters")
    p = evaluate_parameters(ctx, params)[0]
    assert p["schedule_mode"] == expected and p["detect_mode"] == "action"
    assert p["detect_action"] == "/box_seam_detector/detect_box" and p["detect_timeout_s"] == 300.0
