"""Plan sets schedule_mode first; Enable writes only execution_enabled + execute_phase_a_enabled; every OFF path writes all
three gates false; the phases-plan blocker."""
import json
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from helpers import Env, read_data


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    e.bridge.push_status(state=B.STATE_IDLE)
    yield e
    e.close()


def order(env):
    return [c[0] if c[0] != "trigger" else "trigger:" + c[1] for c in env.bridge.calls if c[0] in ("set_schedule_mode", "trigger")]


def log_text(job):
    return "\n".join(job.log)


# ---------------------------------------------------------------- schedule_mode before plan
def test_plan_sets_schedule_mode_then_plans(env):
    assert env.act("plan")[0] == 200
    job = env.wait_job()
    assert job.state == "done"
    assert order(env) == ["set_schedule_mode", "trigger:plan"]
    assert env.bridge.schedule_modes == ["sequence"]
    assert "[ui] param set /dual_cut_coordinator schedule_mode sequence" in log_text(job)
    assert log_text(job).index("param set") < log_text(job).index("calling coordinator plan")


def test_plan_fails_without_calling_plan_when_the_set_fails(env):
    env.bridge.schedule_ok = False
    env.act("plan")
    job = env.wait_job()
    assert job.state == "failed" and job.plan_ok is False
    assert "could not set schedule_mode sequence before planning" in job.message and "service not available" in job.message
    assert job.step("Plan")["state"] == "failed"
    assert order(env) == ["set_schedule_mode"] and env.bridge.count("trigger") == 0


def test_a_raising_set_fails_the_plan_step_too(env):
    env.bridge.raises.add("schedule_mode")
    env.act("plan")
    job = env.wait_job()
    assert job.state == "failed" and "boom schedule_mode" in job.message and env.bridge.count("trigger") == 0


def test_empty_schedule_mode_leaves_the_coordinator_alone(tmp_path):
    e = Env(tmp_path, schedule_mode="")
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("plan")
        job = e.wait_job()
        assert job.state == "done" and e.bridge.count("set_schedule_mode") == 0 and e.bridge.count("trigger") == 1
        assert "param set" not in log_text(job)
    finally:
        e.close()


def test_a_configured_phases_mode_is_written(tmp_path):
    e = Env(tmp_path, schedule_mode="phases")
    try:
        e.bridge.push_status(state=B.STATE_IDLE)
        e.act("plan")
        e.wait_job()
        assert e.bridge.schedule_modes == ["phases"]
    finally:
        e.close()


def test_a_refused_plan_precheck_sets_nothing(env):
    env.bridge.push_status(state=B.STATE_EXECUTING)
    assert env.act("plan")[0] == 409
    assert env.bridge.count("set_schedule_mode") == 0


def test_every_plan_sets_the_mode_again(env):
    env.act("plan")
    env.wait_job()
    env.act("plan")
    env.wait_job()
    assert env.bridge.schedule_modes == ["sequence", "sequence"]


# ---------------------------------------------------------------- enable / disable writes
def gates_now(env):
    return {p: env.bridge.gates()[0][p] for p in B.GATE_PARAMS}


def test_gate_params_for():
    assert B.gate_params_for(True) == ("execution_enabled", "execute_phase_a_enabled") == B.ENABLE_PARAMS
    assert B.gate_params_for(False) == B.GATE_PARAMS and "execute_phase_b_enabled" in B.gate_params_for(False)
    assert set(B.ENABLE_PARAMS) < set(B.GATE_PARAMS)


def test_enable_writes_exactly_the_two_params_true_and_leaves_phase_b(env):
    env.bridge.set_gate_values(False)
    s, r = env.act("set_execution_enabled", {"enabled": True})
    assert s == 200 and env.bridge.gate_param_writes == [(True, B.ENABLE_PARAMS)]
    assert gates_now(env) == {"execution_enabled": True, "execute_phase_a_enabled": True, "execute_phase_b_enabled": False}
    st = env.core.state("op")["gates"]
    assert st["all_enabled"] is True and st["execute_phase_b_enabled"] is False


def all_off_writes(env):
    return [w for w in env.bridge.gate_param_writes if w[0] is False]


def assert_only_full_off_writes(env):
    offs = all_off_writes(env)
    assert offs and all(w[1] == B.GATE_PARAMS for w in offs)
    assert all(w[1] == B.ENABLE_PARAMS for w in env.bridge.gate_param_writes if w[0] is True)


def test_off_path_operator_switch(env):
    env.act("set_execution_enabled", {"enabled": True})
    env.act("set_execution_enabled", {"enabled": False})
    assert_only_full_off_writes(env)
    assert gates_now(env) == {p: False for p in B.GATE_PARAMS}


def test_off_path_software_stop(env):
    env.bridge.set_gate_values(True)
    env.act("software_stop")
    env.core.join_background()
    assert_only_full_off_writes(env)
    assert gates_now(env) == {p: False for p in B.GATE_PARAMS}


def test_off_path_ui_start_and_shutdown(env):
    env.core.start()
    env.core.join_background()
    assert_only_full_off_writes(env)
    n = len(env.bridge.gate_param_writes)
    env.core.shutdown()
    assert len(env.bridge.gate_param_writes) == n + 1 and env.bridge.gate_param_writes[-1] == (False, B.GATE_PARAMS)


def test_off_path_run_end(env):
    env.act("set_execution_enabled", {"enabled": True})
    env.bridge.push_status(state=B.STATE_EXECUTING)
    env.bridge.push_status(state=B.STATE_PLANNED)
    deadline = time.time() + 3
    while time.time() < deadline and not all_off_writes(env):
        time.sleep(0.01)
    env.core.join_background()
    assert_only_full_off_writes(env)
    assert gates_now(env) == {p: False for p in B.GATE_PARAMS}


def test_off_path_enable_races(env):
    gate = threading.Event()
    env.bridge.block["gates_true"] = gate
    out = []
    t = threading.Thread(target=lambda: out.append(env.act("set_execution_enabled", {"enabled": True})))
    t.start()
    deadline = time.time() + 3
    while time.time() < deadline and not env.bridge.gate_param_writes:
        time.sleep(0.01)
    env.act("software_stop")
    gate.set()
    t.join(5)
    assert out[0][0] == 409
    assert_only_full_off_writes(env)
    assert gates_now(env) == {p: False for p in B.GATE_PARAMS}


def test_off_path_unconfirmed_enable_rollback(env):
    env.bridge.fail_true = True
    assert env.act("set_execution_enabled", {"enabled": True})[0] == 409
    assert env.bridge.gate_param_writes[-1] == (False, B.GATE_PARAMS)


# ---------------------------------------------------------------- phases-plan blocker
PHASES_TEXT = "the retained plan is a 'phases' plan: with phase B off only phase A would run; plan again"


def blockers(env):
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)     # a PLANNED cell
    return env.core.cut_blockers("op", 3)


def test_phases_plan_with_phase_b_off_blocks_cut(env):
    env.set_plan_report("side_first_report.json")
    env.bridge.set_gate_values(True, execute_phase_b_enabled=False)
    b = blockers(env)
    assert len(b) == 1 and PHASES_TEXT in b[0] and "Plan sets schedule_mode sequence" in b[0]
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    s, r = env.act("execute", {"plan_seq": 3})
    assert s == 409 and PHASES_TEXT in r["message"] and env.bridge.count("trigger") == 0
    st = env.core.state("op")
    assert st["cut"]["can_cut"] is False and PHASES_TEXT in st["cut"]["blockers"][0]


def test_sequence_plan_with_phase_b_off_is_allowed(env):
    env.set_plan_report("sequence_plan_report.json")
    env.bridge.set_gate_values(True, execute_phase_b_enabled=False)
    assert blockers(env) == []
    s, r = env.act("execute", {"plan_seq": 3})
    assert s == 200 and env.bridge.names("trigger")[0][1] == "execute"


def test_phases_plan_with_phase_b_on_is_allowed(env):
    env.set_plan_report("side_first_report.json")
    env.bridge.set_gate_values(True)
    assert blockers(env) == []


def test_unknown_plan_mode_counts_as_not_sequence(env):
    env.bridge.set_gate_values(True, execute_phase_b_enabled=False)
    assert PHASES_TEXT in "; ".join(blockers(env))                    # no plan report at all
    env.bridge._plan_report = ("not json", env.clock())
    assert PHASES_TEXT in "; ".join(blockers(env))
    env.bridge._plan_report = ("{}", env.clock())
    assert PHASES_TEXT in "; ".join(blockers(env))


def test_a_sequence_request_that_fell_back_to_phases_counts_as_phases(env):
    doc = json.loads(read_data("sequence_plan_report.json"))
    doc["sequence"] = {"fallback_to_phases": True}
    env.bridge._plan_report = (json.dumps(doc), env.clock())
    env.bridge.set_gate_values(True, execute_phase_b_enabled=False)
    assert PHASES_TEXT in "; ".join(blockers(env))


def test_the_enable_params_must_both_read_back_true(env):
    env.set_plan_report("sequence_plan_report.json")
    env.bridge.set_gate_values(True, execute_phase_a_enabled=False, execute_phase_b_enabled=False)
    assert any("execute_phase_a_enabled not true" in b for b in blockers(env))
    env.bridge.set_gate_values(True, execution_enabled=False, execute_phase_b_enabled=False)
    assert any("execution_enabled not true" in b for b in blockers(env))
