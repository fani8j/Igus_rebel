"""Round 4: deterministic interleavings (events / barriers on the fake bridge) for the concurrency fixes."""
import sys
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from dual_cut_ui.jobs import ProcessRunner
from helpers import Env


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def spawn(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    t.start()
    return t, out


def wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def gate_writes(env):
    return [c[1] for c in env.bridge.names("set_gates")]


def final_gates(env):
    return {p: env.bridge.gates()[0][p] for p in B.GATE_PARAMS}


# ============================================================== 1a: execute and plan exclude each other
def test_plan_cannot_start_while_an_execute_is_in_flight(env):
    gate = threading.Event()
    env.bridge.block["execute"] = gate
    t, out = spawn(lambda: env.act("execute", {"plan_seq": 3}))
    assert wait_for(lambda: env.bridge.count("trigger") == 1)
    for action in ("plan", "detect_and_plan"):
        s, r = env.act(action)
        assert s == 409 and "execute request is in flight" in r["message"]
    assert env.core._job is None and env.bridge.count("trigger") == 1
    gate.set()
    t.join(5)
    assert out[0][0] == 200


def test_execute_is_refused_once_a_plan_job_exists(tmp_path):
    e = Env(tmp_path, detect_command=[sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        assert e.act("detect_and_plan")[0] == 200
        e.wait_job_started_process()
        e.bridge.set_ready()
        s, r = e.act("execute", {"plan_seq": 3})
        assert s == 409 and "a detect / plan job is running" in r["message"]
        assert e.bridge.count("trigger") == 0
        e.act("cancel_job")
        e.wait_job()
    finally:
        e.close()


# ============================================================== 1b: plan identity guard after the dispatch
def test_plan_seq_change_right_after_execute_triggers_a_software_stop(env):
    s, r = env.act("execute", {"plan_seq": 3})
    assert s == 200 and r["ok"]
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=4, plan_retained=True, phase_validated=True)     # a new plan appeared
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute", "stop"]
    assert {c[1] for c in env.bridge.names("cancel")} == {"left", "right"}
    assert env.core.last_stop["reason"] == "plan changed under the Cut confirmation"
    st = env.core.state("op")
    assert "plan changed under the Cut confirmation" in st["alert"]["message"] and "plan 3" in st["alert"]["message"]
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=5)           # the guard fired once
    env.core.join_background()
    assert env.bridge.count("cancel") >= 2 and [c[1] for c in env.bridge.names("trigger")].count("stop") == 1


def test_guard_is_disarmed_when_the_confirmed_plan_starts(env):
    env.act("execute", {"plan_seq": 3})
    env.bridge.push_status(state=B.STATE_WAITING_BARRIER, plan_seq=3, plan_retained=True, phase_validated=True)
    env.bridge.push_status(state=B.STATE_EXECUTING, plan_seq=9)         # whatever happens later is not this guard's business
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute"]
    assert env.core.state("op")["alert"] is None


def test_guard_trips_even_if_the_run_started_on_another_plan(env):
    env.act("execute", {"plan_seq": 3})
    env.bridge.push_status(state=B.STATE_WAITING_BARRIER, plan_seq=4)
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute", "stop"]


def test_guard_expires_after_the_window(env):
    env.act("execute", {"plan_seq": 3})
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    env.clock.advance(3.5)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=4)
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute"]


def test_guard_covers_a_status_that_overtakes_the_execute_reply(env):
    gate = threading.Event()
    env.bridge.block["execute"] = gate
    t, out = spawn(lambda: env.act("execute", {"plan_seq": 3}))
    assert wait_for(lambda: env.bridge.count("trigger") == 1)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=4, plan_retained=True, phase_validated=True)     # before the reply
    gate.set()
    t.join(5)
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute", "stop"]


def test_a_refused_execute_leaves_no_guard(env):
    env.bridge.results["execute"] = (False, "refused: stop is latched")
    assert env.act("execute", {"plan_seq": 3})[1]["ok"] is False
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=4)
    env.core.join_background()
    assert [c[1] for c in env.bridge.names("trigger")] == ["execute"]


def test_the_guard_never_blocks_the_ros_callback(env):
    env.act("execute", {"plan_seq": 3})
    gate = threading.Event()
    env.bridge.block["stop"] = gate
    t0 = time.monotonic()
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=4)
    assert time.monotonic() - t0 < 0.3                                  # the stop runs on its own thread
    gate.set()
    env.core.join_background()


# ============================================================== 2a: disable dominates the enable path
def start_enable(env, page="op"):
    gate = threading.Event()
    env.bridge.block["gates_true"] = gate
    t, out = spawn(lambda: env.act("set_execution_enabled", {"enabled": True}, page=page))
    assert wait_for(lambda: True in gate_writes(env))
    return gate, t, out


def test_enable_racing_a_software_stop_ends_false(env):
    gate, t, out = start_enable(env)
    t0 = time.monotonic()
    s, r = env.act("software_stop")                                      # must not wait for the enable that is still in flight
    assert s == 200 and time.monotonic() - t0 < 1.5
    gate.set()
    t.join(5)
    status, reply = out[0]
    assert status == 409 and "a Software stop happened while enabling" in reply["message"] and "closed again" in reply["message"]
    assert final_gates(env) == {p: False for p in B.GATE_PARAMS}
    writes = gate_writes(env)
    assert writes[0] is True and writes[-1] is False and writes.count(False) >= 2        # stop's write + the re-write


def test_enable_racing_a_shutdown_ends_false(env):
    gate, t, out = start_enable(env)
    env.core.shutdown(budget_s=2.0)
    gate.set()
    t.join(5)
    assert out[0][0] == 409 and "shutting down" in out[0][1]["message"]
    assert final_gates(env) == {p: False for p in B.GATE_PARAMS}


def test_enable_racing_a_takeover_ends_false(env):
    gate, t, out = start_enable(env)
    assert env.act("take_control", {"force": True}, page="other")[0] == 200
    gate.set()
    t.join(5)
    assert out[0][0] == 409 and "no longer the operator" in out[0][1]["message"]
    assert final_gates(env) == {p: False for p in B.GATE_PARAMS}


def test_an_unconfirmed_enable_is_rolled_back(env):
    env.bridge.fail_true = True
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 409 and "not confirmed" in r["message"]
    assert gate_writes(env) == [True, False]


def test_a_clean_enable_is_a_single_write(env):
    env.bridge.set_gate_values(False)
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 200 and gate_writes(env) == [True]
    assert final_gates(env) == {"execution_enabled": True, "execute_phase_a_enabled": True, "execute_phase_b_enabled": False}


def test_enable_is_refused_while_the_ui_is_closing(env):
    env.core._closing.set()
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 409 and gate_writes(env) == []


def test_disable_is_never_held_up_by_an_enable(env):
    gate, t, out = start_enable(env)
    t0 = time.monotonic()
    s, r = env.act("set_execution_enabled", {"enabled": False})
    assert s == 200 and time.monotonic() - t0 < 1.0
    gate.set()
    t.join(5)


# ============================================================== 2b: run-end closure keeps trying
def finish_run(env):
    env.bridge.push_status(state=B.STATE_EXECUTING)
    env.bridge.push_status(state=B.STATE_PLANNED)


def test_run_end_closure_retries_until_a_false_write_succeeds(tmp_path):
    e = Env(tmp_path, gates_retry_s=0.02)
    try:
        e.bridge.fail_false = True
        finish_run(e)
        assert wait_for(lambda: gate_writes(e).count(False) >= 6)         # well beyond the old 3 tries
        e.bridge.fail_false = False
        assert wait_for(lambda: e.core.last_gates_off["ok"])
        n = gate_writes(e).count(False)
        time.sleep(0.15)
        assert gate_writes(e).count(False) == n                          # stopped after the success
        e.core.join_background(3)
    finally:
        e.close()


def test_a_newer_operator_enable_ends_the_retries(tmp_path):
    e = Env(tmp_path, gates_retry_s=0.02)
    try:
        e.bridge.fail_false = True
        finish_run(e)
        assert wait_for(lambda: gate_writes(e).count(False) >= 3)
        assert e.act("set_execution_enabled", {"enabled": True})[0] == 200
        time.sleep(0.1)
        n = gate_writes(e).count(False)
        time.sleep(0.2)
        assert gate_writes(e).count(False) == n                          # the retry loop gave way to the operator's new intent
        assert all(final_gates(e)[p] is True for p in B.ENABLE_PARAMS)
        e.core.join_background(3)
    finally:
        e.close()


def test_shutdown_ends_the_retries(tmp_path):
    e = Env(tmp_path, gates_retry_s=0.02)
    e.bridge.fail_false = True
    finish_run(e)
    assert wait_for(lambda: gate_writes(e).count(False) >= 3)
    e.core.shutdown(budget_s=1.0)
    e.core.join_background(3)
    n = gate_writes(e).count(False)
    time.sleep(0.15)
    assert gate_writes(e).count(False) == n


def test_a_second_run_end_while_retrying_is_not_lost(tmp_path):
    e = Env(tmp_path, gates_retry_s=0.02)
    try:
        e.bridge.fail_false = True
        finish_run(e)
        assert wait_for(lambda: gate_writes(e).count(False) >= 2)
        finish_run(e)
        e.bridge.fail_false = False
        assert wait_for(lambda: e.core.last_gates_off["ok"])
        e.core.join_background(3)
        assert e.core._gates_thread is None
    finally:
        e.close()


# ============================================================== 3: rearm versus stop
def test_a_stop_during_a_rearm_is_sent_again_after_it(env):
    gate = threading.Event()
    env.bridge.block["rearm"] = gate
    t, out = spawn(lambda: env.act("rearm"))
    assert wait_for(lambda: env.bridge.count("trigger") == 1)
    env.act("software_stop")
    gate.set()
    t.join(5)
    s, r = out[0]
    assert s == 200 and r["ok"] is False and "raced with a Software stop" in r["message"] and "sent again" in r["message"]
    assert [c[1] for c in env.bridge.names("trigger")] == ["rearm", "stop", "stop"]       # stop dominates: it is the LAST word
    env.core.join_background()


def test_a_rearm_without_a_stop_is_untouched(env):
    s, r = env.act("rearm")
    assert s == 200 and r["ok"] and [c[1] for c in env.bridge.names("trigger")] == ["rearm"]


def test_actions_admitted_before_a_takeover_finish(env):
    gate = threading.Event()
    env.bridge.block["rearm"] = gate
    t, out = spawn(lambda: env.act("rearm"))
    assert wait_for(lambda: env.bridge.count("trigger") == 1)
    assert env.act("take_control", {"force": True}, page="other")[0] == 200
    gate.set()
    t.join(5)
    assert out[0][0] == 200 and out[0][1]["ok"] is True


# ============================================================== 4b: independent cancel repeats
def test_cancel_repeats_are_not_delayed_by_a_slow_first_call(tmp_path):
    e = Env(tmp_path, stop_repeat_delays=(0.1, 0.25))
    try:
        e.bridge.cancel_delays["left"] = [0.5]                            # the first left cancel takes 0.5 s
        t0 = time.monotonic()
        e.act("software_stop")
        e.core.join_background(5)
        left = sorted(c[-1] - t0 for c in e.bridge.names("cancel") if c[1] == "left")
        right = sorted(c[-1] - t0 for c in e.bridge.names("cancel") if c[1] == "right")
        assert len(left) == 3 and len(right) == 3
        assert left[0] < 0.08 and 0.07 < left[1] < 0.2 and 0.2 < left[2] < 0.4          # NOT after the 0.5 s call finished
        assert right[1] < 0.2 and right[2] < 0.4
    finally:
        e.close()


# ============================================================== 5: cancel return codes
@pytest.mark.parametrize("code,n,ok,text", [
    (0, 2, True, "2 goal(s) cancelling (code 0)"),
    (0, 0, True, "no active goal (code 0)"),
    (1, 0, False, "cancel REJECTED by the controller (code 1)"),
    (1, 1, True, "1 goal(s) cancelling (code 1)"),
    (2, 0, True, "no active goal (code 2)"),
    (3, 0, True, "no active goal (code 3)"),
    (3, 2, True, "2 goal(s) cancelling (code 3)"),
    (9, 0, False, "unexpected cancel return code 9"),
])
def test_interpret_cancel(code, n, ok, text):
    assert B.interpret_cancel(code, n) == (ok, text)


def test_ros_bridge_cancel_all_uses_the_return_code():
    pytest.importorskip("rclpy")
    from action_msgs.msg import GoalInfo
    from action_msgs.srv import CancelGoal
    from dual_cut_ui.ros_bridge import RosBridge
    bridge = object.__new__(RosBridge)
    bridge._cancel = {"left": object()}

    def answer(code, n):
        resp = CancelGoal.Response()
        resp.return_code = code
        resp.goals_canceling = [GoalInfo() for _ in range(n)]
        bridge._call = lambda client, req, timeout, require_ready=True: (resp, "")
        return bridge.cancel_all("left", 1.0)
    assert answer(1, 0) == (False, "cancel left: cancel REJECTED by the controller (code 1)")
    assert answer(0, 1) == (True, "cancel left: 1 goal(s) cancelling (code 0)")
    assert answer(2, 0)[0] is True
    bridge._call = lambda *a, **k: (None, "no answer within 1.0 s (the request was sent)")
    assert bridge.cancel_all("left", 1.0)[0] is False


def test_a_rejected_cancel_is_a_failed_stop_step(env):
    env.bridge.cancel_results["right"] = (False, "cancel right: cancel REJECTED by the controller (code 1)")
    s, r = env.act("software_stop")
    assert r["ok"] is False and r["steps"]["cancel_right"]["ok"] is False and "REJECTED" in r["message"]
    assert r["steps"]["cancel_left"]["ok"] is True
    env.core.join_background()


# ============================================================== 6: process groups
def running(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def tree_script(parent_exits, grandchild_closes_stdout=False):
    child = ("import signal, sys, time, os\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
             + ("sys.stdout.close(); sys.stderr.close(); os.close(1); os.close(2)\n" if grandchild_closes_stdout else "")
             + "time.sleep(120)\n")
    return ("import subprocess, sys, time\n"
            f"p = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "print(p.pid, flush=True)\n"
            + ("sys.exit(0)\n" if parent_exits else "time.sleep(120)\n"))


def run_tree(runner):
    lines = []
    t = threading.Thread(target=lambda: runner.run(on_line=lines.append), daemon=True)
    t.start()
    assert wait_for(lambda: bool(lines))
    return int(lines[0]), t


def test_a_descendant_holding_the_pipe_cannot_hang_run_and_is_killed():
    r = ProcessRunner([sys.executable, "-c", tree_script(parent_exits=True)], reader_join_s=0.3)
    t0 = time.monotonic()
    grandchild, t = run_tree(r)
    t.join(10)
    assert not t.is_alive() and time.monotonic() - t0 < 3.0           # the reader was not waited for forever
    assert r.returncode == 0
    assert wait_for(lambda: not running(grandchild), 3.0)             # the TERM-ignoring leftover got SIGKILL


def test_terminate_kills_a_term_ignoring_grandchild_after_the_grace():
    r = ProcessRunner([sys.executable, "-c", tree_script(parent_exits=False)], kill_grace_s=0.4, reader_join_s=0.3)
    grandchild, t = run_tree(r)
    assert running(grandchild)
    r.terminate()
    t.join(10)
    assert not t.is_alive()
    assert wait_for(lambda: not running(grandchild), 3.0)


def test_terminate_signals_the_group_even_when_the_child_already_exited():
    r = ProcessRunner([sys.executable, "-c", tree_script(parent_exits=True, grandchild_closes_stdout=True)], kill_grace_s=0.3, reader_join_s=0.3)
    grandchild, t = run_tree(r)
    t.join(10)
    assert not t.is_alive() and not r.alive()                         # the direct child is gone ...
    assert running(grandchild) and r.group_alive()                    # ... its TERM-ignoring child is not
    r.terminate()
    assert wait_for(lambda: not running(grandchild), 3.0)


def test_kill_now_takes_the_group_even_after_the_leader_exited():
    r = ProcessRunner([sys.executable, "-c", tree_script(parent_exits=True, grandchild_closes_stdout=True)], reader_join_s=0.3)
    grandchild, t = run_tree(r)
    t.join(10)
    assert running(grandchild)
    r.kill_now()
    assert wait_for(lambda: not running(grandchild), 3.0)
    r.kill_now()                                                      # a second one on a dead group is harmless


def test_core_stop_kills_the_whole_detector_tree(tmp_path):
    e = Env(tmp_path, detect_command=[sys.executable, "-c", tree_script(parent_exits=False)], kill_grace_s=0.4)
    try:
        e.act("detect")
        job = e.wait_job_started_process()
        deadline = time.time() + 5
        while time.time() < deadline and not any(line.strip().isdigit() for line in list(job.log)):
            time.sleep(0.02)
        grandchild = int(next(line for line in job.log if line.strip().isdigit()))
        e.act("software_stop")
        e.wait_job(10)
        assert wait_for(lambda: not running(grandchild), 4.0)
    finally:
        e.close()
