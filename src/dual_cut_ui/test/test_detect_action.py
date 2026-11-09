"""Detect through the DetectBox action (default mode): goal fields, log lines, result handling, cancel paths.  The fake bridge
stands in for the action (the real RosBridge.detect is covered in test_ros_bridge_stubs.py)."""
import os
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from dual_cut_ui.core import Config
from dual_cut_ui.detect_action import ActionRunner
from helpers import Env

ACTION = "/box_seam_detector/detect_box"
HINT = "source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash"


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path, detect_mode="action")
    e.bridge.push_status(state=B.STATE_IDLE)
    yield e
    e.close()


def log_text(job):
    return "\n".join(job.log)


def test_config_defaults_and_validation():
    c = Config()
    assert (c.detect_mode, c.detect_action, c.detect_timeout_s, c.schedule_mode) == ("action", ACTION, 300.0, "sequence")
    with pytest.raises(ValueError):
        Config(detect_mode="subprocess")
    with pytest.raises(ValueError):
        Config(schedule_mode="both")
    assert Config(schedule_mode="").schedule_mode == ""


def test_action_detect_sends_the_operators_goal_and_logs_like_the_cli(env):
    assert env.act("detect")[0] == 200
    job = env.wait_job()
    assert job.state == "done", job.to_dict()
    assert env.bridge.detect_goals == [(ACTION, {"max_attempts": 0, "allow_needs_review": False, "allow_cut_blocked": False,
                                                 "dry_run": False})]
    assert B.DETECT_GOAL == env.bridge.detect_goals[0][1]
    assert env.bridge.names("detect")[0][3] == 300.0                       # detect_timeout_s
    text = log_text(job)
    assert f"[ui] send_goal {ACTION} DetectBox {{dry_run: false}}" in text
    assert "[capture] attempt 1: grabbing frames" in text and "[detect] attempt 1: seam found" in text
    assert "result: success=true status=accepted reason=snapshot written" in text
    assert "cut_permission: allowed" in text and "attempts: 1  elapsed: 4.5 s" in text
    step = job.step("Detect")
    assert step["state"] == "ok" and "seam yaw" in step["detail"]           # the same detail text as command mode
    assert env.bridge.count("trigger") == 0                                # Detect alone never plans
    assert job.runner is None


def test_detect_and_plan_runs_the_plan_after_the_action(env):
    assert env.act("detect_and_plan")[0] == 200
    job = env.wait_job()
    assert job.state == "done" and [s["state"] for s in job.steps] == ["ok", "ok"]
    names = [c[0] for c in env.bridge.calls if c[0] in ("detect", "set_schedule_mode", "trigger")]
    assert names == ["detect", "set_schedule_mode", "trigger"]


def test_result_success_false_fails_the_step_with_the_reason(env):
    env.bridge.detect_result = {**env.bridge.detect_result, "success": False, "status": "rejected", "reason": "no seam found",
                                "reasons": ["seam_missing", "low_confidence"], "cut_permission": "n/a", "attempts": 3}
    env.bridge.detect_hook = None
    assert env.act("detect_and_plan")[0] == 200
    job = env.wait_job()
    assert job.state == "failed" and job.message == "detector rejected: no seam found"
    assert job.step("Detect")["state"] == "failed" and "no seam found" in job.step("Detect")["detail"]
    assert job.step("Plan")["state"] == "skipped"
    assert "reasons: seam_missing, low_confidence" in log_text(job)
    assert env.bridge.count("trigger") == 0 and env.bridge.count("set_schedule_mode") == 0


def test_result_without_reason_uses_the_reason_codes(env):
    env.bridge.detect_result = {**env.bridge.detect_result, "success": False, "status": "blocked", "reason": "",
                                "reasons": ["cut_blocked"]}
    env.act("detect")
    assert env.wait_job().message == "detector blocked: cut_blocked"


def test_success_but_the_snapshot_was_not_rewritten_fails(env):
    old = time.time() - 600
    os.utime(env.snap, (old, old))
    env.bridge.detect_hook = None
    env.act("detect")
    job = env.wait_job()
    assert job.state == "failed" and "reported success but did not write a new snapshot file" in job.message
    assert job.step("Detect")["state"] == "failed"


def test_success_but_the_snapshot_is_unusable_fails(env):
    env.bridge.detect_hook = lambda: open(env.snap, "w").write("{not json")
    env.act("detect")
    job = env.wait_job()
    assert job.state == "failed" and "reported success but the snapshot is unusable" in job.message


def test_snapshot_path_mismatch_is_flagged_in_the_log(env):
    env.bridge.detect_result = {**env.bridge.detect_result, "snapshot_path": "/somewhere/else.json"}
    env.act("detect")
    job = env.wait_job()
    assert job.state == "done" and "WARNING: the detector wrote /somewhere/else.json" in log_text(job)


@pytest.mark.parametrize("error", ["detector action server not available (is box_seam_detector running?)",
                                   "detector busy / goal rejected",
                                   "the DetectBox action interface is not available (No module named 'compal_box_msgs.action'); "
                                   "start the UI after sourcing the box-seam-detection install:  " + HINT,
                                   "no result within 300 s; the goal was cancelled"])
def test_transport_failures_are_shown_to_the_operator(env, error):
    env.bridge.detect_error = error
    env.act("detect_and_plan")
    job = env.wait_job()
    assert job.state == "failed" and job.message == error
    assert job.step("Detect")["state"] == "failed" and job.step("Detect")["detail"].startswith(error[:40])
    assert job.step("Plan")["state"] == "skipped" and env.bridge.count("trigger") == 0


def test_a_bridge_exception_fails_the_step(env):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    env.bridge.detect = boom
    env.act("detect")
    job = env.wait_job()
    assert job.state == "failed" and "kaboom" in job.message


def test_command_mode_is_unchanged(tmp_path):
    e = Env(tmp_path)                                  # helpers default: detect_mode="command" with the mock detector subprocess
    try:
        assert e.cfg.detect_mode == "command"
        e.bridge.push_status(state=B.STATE_IDLE)
        old = time.time() - 600
        os.utime(e.snap, (old, old))
        e.act("detect")
        job = e.wait_job()
        assert job.state == "done", job.to_dict()
        assert e.bridge.count("detect") == 0
        assert "[ui] running: " in log_text(job) and "send_goal" not in log_text(job)
    finally:
        e.close()


# ---------------------------------------------------------------- cancel paths
def start_long_detect(env):
    env.bridge.block["detect"] = threading.Event()
    assert env.act("detect")[0] == 200
    return env.wait_job_started_detect()


def test_cancel_button_cancels_the_goal(env):
    start_long_detect(env)
    s, r = env.act("cancel_job")
    assert s == 200 and "cancelling the detector" in r["message"]
    assert env.bridge.detect_cancelled.wait(3)
    done = env.wait_job()
    assert done.state == "cancelled" and done.message == "detection cancelled"
    assert done.step("Detect")["detail"] == "cancelled"
    assert "cancelling the detector goal" in log_text(done)


def test_software_stop_cancels_the_goal_and_runs_the_other_steps(env):
    start_long_detect(env)
    s, r = env.act("software_stop")
    assert s == 200
    assert r["steps"]["stop_job"] == {"ok": True, "message": "detect job cancelled"}
    assert env.bridge.detect_cancelled.wait(3)
    assert env.wait_job().state == "cancelled"
    env.core.join_background()
    assert env.bridge.count("cancel") >= 2 and [c[1] for c in env.bridge.names("trigger")] == ["stop"]
    assert (False, B.GATE_PARAMS) in env.bridge.gate_param_writes


def test_shutdown_cancels_the_goal(env):
    start_long_detect(env)
    env.core.shutdown(budget_s=3.0)
    assert env.bridge.detect_cancelled.is_set()
    assert env.wait_job().state == "cancelled"


def test_cancel_before_the_goal_is_sent_never_sends_it(env):
    env.bridge.block["detect"] = threading.Event()
    r = ActionRunner(env.bridge, ACTION, B.DETECT_GOAL, 5)
    assert r.wait_gone(0.1) is True                    # not started: nothing to wait for
    r.terminate()
    ok, info = r.run(lambda *a: None)
    assert (ok, info) == (False, {"cancelled": True}) and env.bridge.count("detect") == 0
