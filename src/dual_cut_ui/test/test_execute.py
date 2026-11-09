"""Server-side pre-checks of the execute action; every refusal leaves the coordinator untouched."""
import sys
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from helpers import Env, write_snapshot


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def execute(env, seq=3, page="op"):
    return env.act("execute", {"plan_seq": seq}, page=page)


def refused(env, text, status=409, **kw):
    s, r = execute(env, **kw)
    assert s == status and r["ok"] is False, (s, r)
    assert text in r["message"], r["message"]
    assert env.bridge.count("trigger") == 0
    return r


def test_happy_path_calls_execute_once_and_returns_the_message_verbatim(env):
    s, r = execute(env)
    assert s == 200 and r["ok"] is True
    assert r["message"] == "dispatch accepted (barriers are checked next)"
    assert [c[:2] for c in env.bridge.names("trigger")] == [("trigger", "execute")]
    assert env.core.state("op")["cut"] == {"can_cut": True, "blockers": []}


def test_coordinator_refusal_is_passed_on_verbatim(env):
    env.bridge.results["execute"] = (False, "refused: snapshot is stale (250.0 s old, limit 240.0 s)")
    s, r = execute(env)
    assert s == 200 and r["ok"] is False and r["message"] == "refused: snapshot is stale (250.0 s old, limit 240.0 s)"


def test_read_only_page_is_403(env):
    env.core.touch("other")
    s, r = execute(env, page="other")
    assert s == 403 and env.bridge.count("trigger") == 0


@pytest.mark.parametrize("bad", [None, "3", 3.5, True, -1])
def test_plan_seq_is_required_and_must_be_an_int(env, bad):
    s, r = env.act("execute", {} if bad is None else {"plan_seq": bad})
    assert s == 400 and env.bridge.count("trigger") == 0


def test_wrong_plan_seq(env):
    refused(env, "you confirmed plan 2, the coordinator now holds plan 3", seq=2)


@pytest.mark.parametrize("state,name", [(B.STATE_IDLE, "IDLE"), (B.STATE_PLANNING, "PLANNING"), (B.STATE_WAITING_BARRIER, "WAITING_BARRIER"),
                                        (B.STATE_EXECUTING, "EXECUTING"), (B.STATE_STOPPED, "STOPPED")])
def test_state_must_be_planned(env, state, name):
    env.bridge.push_status(state=state, plan_seq=3, plan_retained=True, phase_validated=True)
    refused(env, f"coordinator state is {name}")


def test_status_must_be_fresh(env):
    env.clock.advance(2.5)
    env.bridge.set_health()
    env.bridge.set_gate_values()
    refused(env, "coordinator status is stale or missing")


def test_no_status_at_all(tmp_path):
    e = Env(tmp_path)
    try:
        e.bridge._status = None
        refused(e, "coordinator status is stale or missing")
    finally:
        e.close()


def test_plan_must_be_retained_and_validated(env):
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=False, phase_validated=True)
    refused(env, "no plan is retained")
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=False)
    refused(env, "not validated")


def test_stop_latched(env):
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True,
                           stop_latched=True, stop_reason="software stop")
    refused(env, "stop is latched (software stop)")


def test_snapshot_stale_by_acquisition_stamp(env):
    write_snapshot(env.snap, age_s=300.0, mtime_age_s=0.0)
    refused(env, "snapshot is stale (300 s old, limit 240 s)")


def test_snapshot_stale_by_file_mtime(env):
    write_snapshot(env.snap, age_s=0.0, mtime_age_s=500.0)          # the file is old even though the stamp says fresh
    refused(env, "snapshot is stale")


def test_snapshot_limit_comes_from_the_coordinator_parameter(env):
    write_snapshot(env.snap, age_s=50.0)
    env.bridge.set_gate_values(True, max_age=30.0)
    refused(env, "limit 30 s")
    env.bridge.set_gate_values(True, max_age=None)                  # unreadable: fall back to 240
    s, r = execute(env)
    assert s == 200 and r["ok"]


def test_snapshot_missing_or_without_stamp(env, tmp_path):
    def drop(doc):
        doc.pop("acquisition_stamp_ns")
    write_snapshot(env.snap, mutate=drop)
    refused(env, "no valid acquisition_stamp_ns")
    import os
    os.remove(env.snap)
    refused(env, "snapshot unavailable")


@pytest.mark.parametrize("kw,text", [({"ctrl_l": False}, "controllers left"), ({"ctrl_r": False}, "controllers right"),
                                     ({"hw_l": False}, "hardware left"), ({"hw_r": False}, "hardware right")])
def test_controllers_and_hardware_must_be_active(env, kw, text):
    env.bridge.set_health(**kw)
    refused(env, "not active: " + text)


def test_health_must_be_fresh_and_present(env):
    env.clock.advance(1.0)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    env.bridge.set_gate_values()
    env.clock.advance(2.5)                                           # health is now 3.5 s old, status 2.5 s: refresh status only
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    refused(env, "controller / hardware state is unknown")
    env.bridge._health = None
    refused(env, "controller / hardware state is unknown")


@pytest.mark.parametrize("gate", list(B.ENABLE_PARAMS))
def test_each_gate_must_read_back_true(env, gate):
    env.bridge.set_gate_values(True, **{gate: False})
    refused(env, f"execution is not enabled on the coordinator ({gate} not true)")


def test_gates_unknown_or_stale(env):
    env.bridge._gates = None
    refused(env, "execution gates unknown")
    env.bridge.set_gate_values()
    env.clock.advance(1.0)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    env.bridge.set_health()
    env.clock.advance(2.5)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)
    env.bridge.set_health()
    refused(env, "execution gates unknown")


def test_refused_while_a_job_is_running(tmp_path):
    e = Env(tmp_path, detect_command=[sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        e.act("detect")
        e.wait_job_started_process()
        refused(e, "a detect / plan job is running")
        e.act("cancel_job")
        e.wait_job()
    finally:
        e.close()


def test_all_blockers_are_listed_for_the_page(env):
    env.bridge.set_health(hw_l=False)
    env.bridge.set_gate_values(False)
    env.set_plan_report("sequence_plan_report.json")             # (a phases plan with phase B off would add a third blocker)
    st = env.core.state("op")
    assert st["cut"]["can_cut"] is False and len(st["cut"]["blockers"]) == 2
    other = env.core.state("viewer")
    assert any("read-only" in b for b in other["cut"]["blockers"])


def test_concurrent_executes_dispatch_only_one(env):
    gate = threading.Event()
    env.bridge.block["execute"] = gate
    results = []
    t = threading.Thread(target=lambda: results.append(execute(env)))
    t.start()
    time.sleep(0.2)
    s, r = execute(env)
    assert s == 409 and "already in progress" in r["message"]
    gate.set()
    t.join(5)
    assert env.bridge.count("trigger") == 1 and results[0][0] == 200


def test_debounce_blocks_an_immediate_second_dispatch(tmp_path):
    e = Env(tmp_path, execute_debounce_s=5.0)
    try:
        assert execute(e)[0] == 200
        s, r = execute(e)
        assert s == 409 and "a moment ago" in r["message"]
        assert e.bridge.count("trigger") == 1
    finally:
        e.close()


def test_nan_in_status_does_not_break_state_json(env):
    import json
    st = env.core.state("op")
    assert st["coordinator"]["min_cross_distance_m"] is None
    json.dumps(st, allow_nan=False)
