"""The three execution gates are closed at start, shutdown, software stop and when a run ends."""
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from helpers import Env


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def gate_calls(env):
    return [c[1] for c in env.bridge.names("set_gates")]


def test_gates_closed_on_start(env):
    env.core.start()
    env.core.join_background()
    assert gate_calls(env) == [False]
    assert env.core.last_gates_off["reason"] == "UI start"


def test_start_retries_until_the_coordinator_answers(env):
    env.bridge.unavailable.add("gates")
    env.core.start()
    time.sleep(0.15)
    assert len(gate_calls(env)) >= 3
    env.bridge.unavailable.discard("gates")
    env.core.join_background(3)
    assert gate_calls(env)[-1] is False and env.core.last_gates_off["ok"] is True


def test_gates_closed_on_shutdown_and_job_terminated(tmp_path):
    import sys
    e = Env(tmp_path, detect_command=[sys.executable, "-c", "import time; time.sleep(60)"])
    e.act("detect")
    runner = e.wait_job_started_process().runner
    e.core.shutdown()
    assert gate_calls(e) == [False]
    assert runner.wait_gone(5) and runner.terminated
    e.core.shutdown()                    # twice is harmless


@pytest.mark.parametrize("end", [B.STATE_PLANNED, B.STATE_IDLE, B.STATE_STOPPED])
@pytest.mark.parametrize("start", [B.STATE_WAITING_BARRIER, B.STATE_EXECUTING])
def test_gates_close_when_a_run_ends(env, start, end):
    env.bridge.push_status(state=B.STATE_WAITING_BARRIER)
    env.bridge.push_status(state=B.STATE_EXECUTING) if start == B.STATE_EXECUTING else None
    env.core.join_background()
    assert gate_calls(env) == []
    env.bridge.push_status(state=end)
    env.core.join_background()
    assert gate_calls(env) == [False]
    env.bridge.push_status(state=end)                 # the next status messages do not repeat it
    env.core.join_background()
    assert gate_calls(env) == [False]


def test_barrier_straight_to_stopped_closes_the_gates(env):
    env.bridge.push_status(state=B.STATE_WAITING_BARRIER)
    env.bridge.push_status(state=B.STATE_STOPPED, stop_latched=True)
    env.core.join_background()
    assert gate_calls(env) == [False]


def test_a_second_run_closes_them_again(env):
    for _ in range(2):
        env.bridge.push_status(state=B.STATE_EXECUTING)
        env.bridge.push_status(state=B.STATE_PLANNED)
        env.core.join_background()
    assert gate_calls(env) == [False, False]


def test_no_gate_traffic_without_a_run(env):
    for st in (B.STATE_IDLE, B.STATE_PLANNING, B.STATE_PLANNED, B.STATE_PLANNED):
        env.bridge.push_status(state=st)
    env.core.join_background()
    assert gate_calls(env) == []


def test_status_listener_never_blocks_on_a_slow_coordinator(env):
    gate = threading.Event()
    env.bridge.block["gates"] = gate
    env.bridge.push_status(state=B.STATE_EXECUTING)
    t0 = time.monotonic()
    env.bridge.push_status(state=B.STATE_PLANNED)
    assert time.monotonic() - t0 < 0.5
    gate.set()
    env.core.join_background()
    assert gate_calls(env) == [False]


def test_gate_closing_is_retried_when_the_first_call_fails(env):
    env.bridge.unavailable.add("gates")
    env.bridge.push_status(state=B.STATE_EXECUTING)
    env.bridge.push_status(state=B.STATE_STOPPED)
    time.sleep(0.3)
    env.bridge.unavailable.discard("gates")
    env.core.join_background(3)
    assert len(gate_calls(env)) >= 2 and env.core.last_gates_off["ok"] is True


def test_set_execution_enabled_sets_and_reports(env):
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 200 and r["ok"] and gate_calls(env) == [True]
    s, r = env.act("set_execution_enabled", {"enabled": False})
    assert gate_calls(env) == [True, False]
    assert env.act("set_execution_enabled", {"enabled": "yes"})[0] == 400
    assert env.act("set_execution_enabled", {})[0] == 400


def test_enabling_is_refused_when_stop_latched_or_coordinator_offline(env):
    env.bridge.push_status(state=B.STATE_STOPPED, stop_latched=True, stop_reason="software stop")
    env.core.join_background()
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 409 and "stop is latched" in r["message"]
    env.clock.advance(5)
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 409 and "stale or missing" in r["message"]
    # closing is always allowed
    assert env.act("set_execution_enabled", {"enabled": False})[0] == 200


def test_state_shows_the_read_back_not_the_intent(env):
    env.bridge.set_gate_values(False)
    assert env.core.state("op")["gates"]["all_enabled"] is False
    env.bridge.set_gate_values(True, execute_phase_b_enabled=False)
    g = env.core.state("op")["gates"]
    # "enabled" means the two params the Enable switch writes; phase B is shown but not required
    assert g["all_enabled"] is True and g["execute_phase_a_enabled"] is True and g["execute_phase_b_enabled"] is False
    env.bridge.set_gate_values(True, execution_enabled=False)
    assert env.core.state("op")["gates"]["all_enabled"] is False
    env.bridge.set_gate_values(True, execute_phase_a_enabled=False)
    assert env.core.state("op")["gates"]["all_enabled"] is False
