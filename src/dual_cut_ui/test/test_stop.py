"""Software stop: independence, per-step results, repeated cancels, job kill."""
import sys
import threading
import time

import pytest

from dual_cut_ui.core import Config
from helpers import Env


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def test_default_repeat_delays_are_half_and_one_and_a_half_seconds():
    assert Config().stop_repeat_delays == (0.5, 1.5)
    assert Config().stop_step_timeout_s == 1.0


def test_stop_runs_all_steps_and_reports_them(env):
    status, reply = env.act("software_stop")
    assert status == 200 and reply["ok"] is True
    assert set(reply["steps"]) == {"coordinator_stop", "cancel_left", "cancel_right", "gates_off", "stop_job"}
    assert all(s["ok"] for s in reply["steps"].values())
    assert env.core.last_stop["page"] == "op"
    env.core.join_background()


def test_repeated_cancels_follow_the_schedule(tmp_path):
    e = Env(tmp_path, stop_repeat_delays=(0.2, 0.6))
    try:
        t0 = time.monotonic()
        e.act("software_stop")
        e.core.join_background(5)
        for arm in ("left", "right"):
            times = [c[-1] - t0 for c in e.bridge.names("cancel") if c[1] == arm]
            assert len(times) == 3, times
            assert times[0] < 0.15
            assert 0.15 < times[1] < 0.45 and 0.5 < times[2] < 0.9
        # the coordinator stop and the gates are not repeated
        assert e.bridge.count("trigger") == 1 and e.bridge.count("set_gates") == 1
        assert len(e.core.last_stop["repeats"]["cancel_left"]) == 2
    finally:
        e.close()


def test_coordinator_absent_still_cancels_controllers_and_closes_gates(env):
    env.bridge.unavailable.add("stop")
    status, reply = env.act("software_stop")
    assert status == 200 and reply["ok"] is False
    assert reply["steps"]["coordinator_stop"]["ok"] is False
    assert reply["steps"]["cancel_left"]["ok"] and reply["steps"]["cancel_right"]["ok"]
    assert reply["steps"]["gates_off"]["ok"] is True
    assert env.bridge.names("set_gates")[0][1] is False
    assert "did not confirm" in reply["message"] and "hardware e-stop" in reply["message"]


def test_everything_absent_is_reported_not_raised(env):
    env.bridge.unavailable.update({"stop", "gates", "cancel_left", "cancel_right"})
    status, reply = env.act("software_stop")
    assert status == 200 and reply["ok"] is False
    assert not any(s["ok"] for k, s in reply["steps"].items() if k != "stop_job")
    assert env.bridge.count("cancel") >= 2


def test_a_raising_step_does_not_stop_the_others(env):
    env.bridge.raises.update({"stop", "cancel_left"})
    status, reply = env.act("software_stop")
    assert status == 200
    assert reply["steps"]["coordinator_stop"]["ok"] is False and "boom" in reply["steps"]["coordinator_stop"]["message"]
    assert reply["steps"]["cancel_left"]["ok"] is False
    assert reply["steps"]["cancel_right"]["ok"] is True and reply["steps"]["gates_off"]["ok"] is True


def test_a_hanging_step_does_not_hold_the_stop_hostage(env):
    gate = threading.Event()
    env.bridge.block["cancel_left"] = gate
    t0 = time.monotonic()
    try:
        status, reply = env.act("software_stop")
    finally:
        gate.set()
    assert time.monotonic() - t0 < env.cfg.stop_step_timeout_s + 2.0
    assert reply["steps"]["cancel_left"] == {"ok": False, "message": "no result"}
    assert reply["steps"]["coordinator_stop"]["ok"] and reply["steps"]["cancel_right"]["ok"]


def test_stop_is_not_blocked_by_a_running_plan(env):
    gate = threading.Event()
    env.bridge.block["plan"] = gate
    env.bridge.push_status(state=0)
    assert env.act("plan")[0] == 200
    time.sleep(0.1)
    assert env.core._job_running()
    t0 = time.monotonic()
    status, reply = env.act("software_stop")
    assert status == 200 and time.monotonic() - t0 < 1.5
    assert reply["steps"]["stop_job"]["message"] == "plan job marked cancelled"
    gate.set()
    job = env.wait_job()
    assert job.state in ("done", "cancelled")


def test_stop_kills_a_running_detect_job(tmp_path):
    cmd = [sys.executable, "-c", "import time; print('capturing', flush=True); time.sleep(120)"]
    e = Env(tmp_path, detect_command=cmd)
    try:
        assert e.act("detect")[0] == 200
        job = e.wait_job_started_process()
        pid = job.runner.pid
        time.sleep(0.3)
        status, reply = e.act("software_stop")
        assert reply["steps"]["stop_job"]["message"] == "detect job terminated"
        job = e.wait_job(10)
        assert job.state == "cancelled" and job.steps[0]["detail"] == "cancelled"
        def gone():
            try:
                with open(f"/proc/{pid}/stat") as f:
                    return f.read().rsplit(")", 1)[1].split()[0] == "Z"
            except OSError:
                return True
        deadline = time.time() + 8
        while time.time() < deadline and not gone():
            time.sleep(0.05)
        assert gone()
    finally:
        e.close()


def test_stop_ignores_term_blocking_child_with_sigkill(tmp_path):
    code = ("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('up', flush=True)\n"
            "time.sleep(120)\n")
    e = Env(tmp_path, detect_command=[sys.executable, "-c", code], kill_grace_s=0.5)
    try:
        e.act("detect")
        job = e.wait_job_started_process()
        time.sleep(0.5)
        e.act("software_stop")
        job = e.wait_job(10)
        assert job.state == "cancelled"
    finally:
        e.close()


def test_stop_aborts_the_second_step_of_detect_and_plan(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(0.5)"]
    e = Env(tmp_path, detect_command=cmd)
    try:
        e.bridge.push_status(state=0)
        e.act("detect_and_plan")
        e.wait_job_started_process()
        e.act("software_stop")
        job = e.wait_job()
        assert e.bridge.count("trigger") == 1           # only the stop, never the plan
        assert job.steps[1]["state"] == "skipped"
    finally:
        e.close()
