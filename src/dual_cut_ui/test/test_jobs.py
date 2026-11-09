"""Detect / plan jobs: exclusivity, cancel, exit codes, log tail, detect_and_plan, plan refusals."""
import os
import sys
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from dual_cut_ui.jobs import ProcessRunner, exit_message
from helpers import Env


def sleeper(seconds=30):
    return [sys.executable, "-c", f"import time; print('started', flush=True); time.sleep({seconds})"]


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    e.bridge.push_status(state=B.STATE_IDLE)
    yield e
    e.close()


def test_detect_succeeds_and_finds_the_overlay(env):
    s, r = env.act("detect")
    assert s == 200 and r["job"] == "job1"
    job = env.wait_job()
    assert job.state == "done" and job.steps[0]["state"] == "ok"
    assert "seam yaw" in job.steps[0]["detail"]
    assert any("MOCK detector" in line for line in job.log)
    assert env.core.detection_info() is not None and env.core.detection_info()["overlay"].endswith("_autolabel_v2.png")
    assert env.bridge.count("trigger") == 0                       # detect only never plans
    st = env.core.state("op")
    assert st["detection"]["available"] is True and st["job"]["state"] == "done"


@pytest.mark.parametrize("code,text", [(1, "rejected"), (2, "is the scene camera running?")])
def test_detector_exit_codes_have_clear_messages(tmp_path, code, text):
    e = Env(tmp_path, detect_args=("--exit-code", str(code)))
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("detect")
        job = e.wait_job()
        assert job.state == "failed" and text in job.message and text in job.steps[0]["detail"]
        assert any("simulated failure" in line for line in job.log)
    finally:
        e.close()


def test_exit_messages():
    assert "scene camera" in exit_message(2)
    assert "rejected" in exit_message(1)
    assert "exit code 7" in exit_message(7)
    assert "SIGTERM" in exit_message(-15)
    assert "could not be started" in exit_message(127)


def test_missing_command_fails_cleanly(tmp_path):
    e = Env(tmp_path, detect_command=["/nonexistent/detector"])
    try:
        e.act("detect")
        job = e.wait_job()
        assert job.state == "failed" and "could not be started" in job.message
        assert any("cannot start" in line for line in job.log)
    finally:
        e.close()


def test_exit_zero_without_a_new_snapshot_is_a_failure(tmp_path):
    e = Env(tmp_path, detect_command=[sys.executable, "-c", "print('did nothing')"])
    try:
        import os, time
        old = time.time() - 600
        os.utime(e.snap, (old, old))
        e.act("detect")
        job = e.wait_job()
        assert job.state == "failed" and "did not write a new snapshot" in job.message
    finally:
        e.close()


def test_only_one_job_at_a_time(tmp_path):
    e = Env(tmp_path, detect_command=sleeper())
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        assert e.act("detect")[0] == 200
        e.wait_job_started_process()
        for a in ("detect", "plan", "detect_and_plan"):
            s, r = e.act(a)
            assert s == 409 and "busy" in r["message"]
        s, r = e.act("rearm")
        assert s == 409
        assert e.bridge.count("trigger") == 0
        assert e.act("cancel_job")[0] == 200
        job = e.wait_job()
        assert job.state == "cancelled" and job.steps[0]["detail"] == "cancelled" and job.message == "detection cancelled"
        assert e.act("detect")[0] == 200                          # free again
        e.act("cancel_job")
        e.wait_job()
    finally:
        e.close()


def test_cancel_without_job_and_during_plan(env):
    assert env.act("cancel_job")[0] == 409
    gate = threading.Event()
    env.bridge.block["plan"] = gate
    env.act("plan")
    time.sleep(0.1)
    s, r = env.act("cancel_job")
    assert s == 409 and "Software stop" in r["message"]
    gate.set()
    env.wait_job()


def test_log_tail_keeps_the_last_80_lines_and_stderr(tmp_path):
    code = "import sys\nfor i in range(300): print('line', i)\nprint('to-stderr', file=sys.stderr)\n"
    r = ProcessRunner([sys.executable, "-c", code])
    assert r.run() == 0
    tail = list(r.tail)
    assert len(tail) == 80 and tail[-1] == "to-stderr" and tail[-2] == "line 299"
    e = Env(tmp_path, detect_command=[sys.executable, "-c", code + "import os; os.utime(%r)" % str(tmp_path / "seam_3d_base_link.json")])
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("detect")
        job = e.wait_job()
        assert len(job.log) == 80
        assert len(e.core.state("op")["job"]["log"]) == 80
    finally:
        e.close()


def test_runner_terminate_kills_the_whole_process_group():
    code = ("import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(p.pid, flush=True)\ntime.sleep(60)\n")
    r = ProcessRunner([sys.executable, "-c", code], kill_grace_s=0.5)
    lines = []
    t = threading.Thread(target=lambda: r.run(on_line=lines.append))
    t.start()
    deadline = time.time() + 5
    while not lines and time.time() < deadline:
        time.sleep(0.02)
    grandchild = int(lines[0])
    r.terminate()
    t.join(10)
    assert not t.is_alive() and r.returncode == -15
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(grandchild, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    else:
        pytest.fail("grandchild survived terminate()")


def test_terminate_before_start_does_not_run():
    r = ProcessRunner(sleeper())
    r.terminate()
    assert r.run() == -15 and r.pid is None


def test_detect_and_plan_runs_both_steps(env):
    s, r = env.act("detect_and_plan")
    assert s == 200
    job = env.wait_job()
    assert job.state == "done" and [x["state"] for x in job.steps] == ["ok", "ok"]
    assert job.message == "top split left_max: ok" and job.plan_ok is True
    assert [c[1:3] for c in env.bridge.names("trigger")] == [("plan", env.cfg.plan_timeout_s)]


def test_detect_and_plan_stops_after_a_failed_detect(tmp_path):
    e = Env(tmp_path, detect_args=("--exit-code", "2"))
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("detect_and_plan")
        job = e.wait_job()
        assert job.state == "failed" and [x["state"] for x in job.steps] == ["failed", "skipped"]
        assert e.bridge.count("trigger") == 0
    finally:
        e.close()


def test_plan_failure_message_is_shown(env):
    env.bridge.results["plan"] = (False, "top seam coverage refused: GAP between the cuts")
    env.act("plan")
    job = env.wait_job()
    assert job.state == "failed" and job.steps[0]["state"] == "failed"
    assert job.steps[0]["detail"] == "top seam coverage refused: GAP between the cuts" and job.plan_ok is False


@pytest.mark.parametrize("state", [B.STATE_EXECUTING, B.STATE_WAITING_BARRIER])
@pytest.mark.parametrize("action", ["plan", "detect_and_plan"])
def test_plan_refused_while_executing(env, state, action):
    env.bridge.push_status(state=state)
    s, r = env.act(action)
    assert s == 409 and "executing" in r["message"]
    assert env.core._job is None and env.bridge.count("trigger") == 0


def test_plan_refused_while_planning_or_offline(env):
    env.bridge.push_status(state=B.STATE_PLANNING)
    assert env.act("plan")[0] == 409
    env.bridge.push_status(state=B.STATE_IDLE)
    env.clock.advance(5)
    s, r = env.act("plan")
    assert s == 409 and "stale or missing" in r["message"]


def test_detect_only_is_allowed_while_executing(env):
    env.bridge.push_status(state=B.STATE_EXECUTING)
    assert env.act("detect")[0] == 200
    env.wait_job()


def test_plan_is_rechecked_just_before_the_call(tmp_path):
    e = Env(tmp_path, detect_command=[sys.executable, "-c", "import time; time.sleep(0.4); import os; os.utime(%r)" % str(tmp_path / "seam_3d_base_link.json")])
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("detect_and_plan")
        e.wait_job_started_process()
        e.bridge.push_status(state=B.STATE_EXECUTING)             # somebody started a run meanwhile
        job = e.wait_job()
        assert job.state == "failed" and "executing" in job.message
        assert e.bridge.count("trigger") == 0
    finally:
        e.close()


def test_rearm_calls_the_coordinator_and_reports(env):
    s, r = env.act("rearm")
    assert s == 200 and r["message"] == "rearmed"
    env.bridge.results["rearm"] = (False, "refused: arms are not at rest")
    s, r = env.act("rearm")
    assert s == 200 and r["ok"] is False and r["message"] == "refused: arms are not at rest"
