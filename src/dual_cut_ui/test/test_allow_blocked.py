"""The blocked-detection switch: coordinator parameter allow_blocked_detection, shown with the detection's cut_permission.

The coordinator side (refusing to execute a plan of a blocked detection while the parameter is false, plan reports carrying
``detection_blocked``) is NOT part of this package: the fake bridge stands in for it."""
import json
import threading
import time

import pytest

from dual_cut_ui import bridge as B
from dual_cut_ui.report import parse_plan_report
from helpers import Env, read_data, write_snapshot


def make_blocked(doc):
    doc["cut_permission"] = {"status": "blocked", "reasons": [{"code": "generic_seam_unconfirmed", "detail": "seam not confirmed"}], "flags": []}


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    write_snapshot(e.snap, mutate=make_blocked)
    e.bridge.set_ready()
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


def writes(env):
    return list(env.bridge.allow_blocked_writes)


def set_allow(env, on, page="op"):
    return env.act("set_allow_blocked_detection", {"enabled": on}, page=page)


def blocked_blockers(env):
    return [b for b in env.core.cut_blockers("op") if "BLOCKED" in b]


# ------------------------------------------------------------------------------------------------ what the page is told
def test_state_shows_the_permission_and_that_the_switch_is_needed(env):
    st = env.core.state("op")
    assert st["snapshot"]["cut_permission"]["status"] == "blocked"
    ab = st["allow_blocked"]
    assert ab["detection_blocked"] is True and ab["value"] is False and ab["known"] is True and ab["needed"] is True
    assert ab["reason_codes"] == ["generic_seam_unconfirmed"]


def test_an_allowed_detection_needs_nothing(tmp_path):
    e = Env(tmp_path)
    try:
        ab = e.core.state("op")["allow_blocked"]
        assert ab["detection_blocked"] is False and ab["needed"] is False
        assert blocked_blockers(e) == [] and e.core.state("op")["cut"]["can_cut"] is True
    finally:
        e.close()


def test_cut_is_blocked_with_a_clear_reason_until_the_switch_is_on(env):
    bl = blocked_blockers(env)
    assert len(bl) == 1 and "generic_seam_unconfirmed" in bl[0] and "Allow blocked detection" in bl[0]
    assert env.core.state("op")["cut"]["can_cut"] is False
    s, r = env.act("execute", {"plan_seq": 3})
    assert s == 409 and "BLOCKED" in r["message"] and env.bridge.count("trigger") == 0       # refused by the UI before the coordinator
    assert set_allow(env, True)[0] == 200
    assert blocked_blockers(env) == [] and env.core.state("op")["cut"]["can_cut"] is True
    s, r = env.act("execute", {"plan_seq": 3})
    assert s == 200 and r["ok"] is True and env.bridge.count("trigger") == 1


def test_a_retained_plan_that_says_detection_blocked_counts_even_if_the_file_says_allowed(tmp_path):
    e = Env(tmp_path)                                       # the snapshot file is 'allowed'
    try:
        report = json.loads(read_data("sequence_plan_report.json"))
        assert parse_plan_report(json.dumps(report))["detection_blocked"] is None          # absent: unknown
        report["detection_blocked"] = True
        e.bridge._plan_report = (json.dumps(report), e.clock())
        assert parse_plan_report(json.dumps(report))["detection_blocked"] is True
        assert e.core.state("op")["allow_blocked"]["detection_blocked"] is True and len(blocked_blockers(e)) == 1
        assert set_allow(e, True)[0] == 200                  # allowed to switch on: the plan, not the file, is blocked
        assert blocked_blockers(e) == []
    finally:
        e.close()


def test_the_report_flag_is_parsed_defensively():
    assert parse_plan_report('{"ok": true, "detection_blocked": false}')["detection_blocked"] is False
    assert parse_plan_report('{"ok": true, "detection_blocked": "yes"}')["detection_blocked"] is None


def test_unknown_readback_keeps_cut_blocked_and_says_why(env):
    env.bridge._allow_blocked = (None, env.clock())          # coordinator without the parameter
    ab = env.core.state("op")["allow_blocked"]
    assert ab["value"] is None and ab["known"] is False and ab["needed"] is True
    assert "does not report allow_blocked_detection" in blocked_blockers(env)[0]
    env.bridge._allow_blocked = (True, env.clock() - 100)    # a stale read-back never counts as 'on'
    assert env.core.state("op")["allow_blocked"]["value"] is None and len(blocked_blockers(env)) == 1


# ------------------------------------------------------------------------------------------------ the action
def test_switch_on_and_off_writes_only_the_one_parameter(env):
    gates_before = env.bridge.count("set_gates")
    s, r = set_allow(env, True)
    assert s == 200 and r["ok"] and writes(env) == [True]
    assert env.core.state("op")["allow_blocked"]["value"] is True
    s, r = set_allow(env, False)
    assert s == 200 and writes(env) == [True, False] and env.core.state("op")["allow_blocked"]["value"] is False
    assert env.bridge.count("set_gates") == gates_before                         # never part of the execution gates
    assert all(c[0] != "set_gates" or c[1] is False for c in env.bridge.calls)   # (only the auto-off gate writes, if any)


def test_enabling_it_does_not_enable_execution(env):
    env.bridge.set_gate_values(False)
    set_allow(env, True)
    assert env.core.state("op")["gates"]["all_enabled"] is False
    assert env.bridge.gate_param_writes == []


@pytest.mark.parametrize("bad", [None, "true", 1, 0, [], {}])
def test_enabled_must_be_a_bool(env, bad):
    s, _ = env.act("set_allow_blocked_detection", {"enabled": bad})
    assert s == 400 and writes(env) == []
    assert env.act("set_allow_blocked_detection", {})[0] == 400


def test_read_only_pages_get_403(env):
    env.core.touch("other")
    s, r = set_allow(env, True, page="other")
    assert s == 403 and "read-only" in r["message"] and env.bridge.count("set_allow_blocked") == 0


def test_refused_when_the_detection_is_not_blocked(tmp_path):
    e = Env(tmp_path)
    try:
        s, r = set_allow(e, True)
        assert s == 409 and "not blocked" in r["message"] and e.bridge.count("set_allow_blocked") == 0
    finally:
        e.close()


def test_refused_when_the_coordinator_is_offline_or_stop_is_latched(env):
    env.clock.advance(10)                                    # status older than status_fresh_s
    s, r = set_allow(env, True)
    assert s == 409 and "stale or missing" in r["message"]
    env.bridge.push_status(state=B.STATE_STOPPED, stop_latched=True, stop_reason="x")
    s, r = set_allow(env, True)
    assert s == 409 and "stop is latched" in r["message"] and writes(env) == []


def test_refused_while_the_ui_is_closing(env):
    env.core._closing.set()
    s, r = set_allow(env, True)
    assert s == 409 and "shutting down" in r["message"] and writes(env) == []


def test_an_unconfirmed_write_is_rolled_back(env):
    env.bridge.allow_blocked_ok = False                      # the coordinator has no such parameter
    s, r = set_allow(env, True)
    assert s == 409 and "not confirmed" in r["message"] and "service not available" in r["message"]
    assert writes(env) == [] and env.bridge.count("set_allow_blocked") >= 2        # the on-write failed, the roll-back was tried


def test_a_raising_bridge_does_not_break_the_action(env):
    env.bridge.raises.add("allow_blocked")
    s, r = set_allow(env, True)
    assert s == 409 and "exception" in r["message"]


# ------------------------------------------------------------------------------------------------ races: disable dominates
def start_enable(env, page="op"):
    gate = threading.Event()
    env.bridge.block["allow_blocked_true"] = gate
    t, out = spawn(lambda: set_allow(env, True, page=page))
    assert wait_for(lambda: env.bridge.count("set_allow_blocked") == 1)
    return gate, t, out


def test_enable_racing_a_software_stop_ends_false(env):
    gate, t, out = start_enable(env)
    t0 = time.monotonic()
    assert env.act("software_stop")[0] == 200 and time.monotonic() - t0 < 1.5
    gate.set()
    t.join(5)
    assert out[0][0] == 409 and "Software stop happened" in out[0][1]["message"] and "switched off again" in out[0][1]["message"]
    env.core.join_background(3)
    assert env.bridge.allow_blocked()[0] is False


def test_enable_racing_a_shutdown_and_a_takeover_ends_false(env):
    gate, t, out = start_enable(env)
    assert env.act("take_control", {"force": True}, page="other")[0] == 200
    gate.set()
    t.join(5)
    assert out[0][0] == 409 and "no longer the operator" in out[0][1]["message"] and env.bridge.allow_blocked()[0] is False


# ------------------------------------------------------------------------------------------------ automatic OFF
def on(env):
    assert set_allow(env, True)[0] == 200
    env.bridge.allow_blocked_writes.clear()


def settle(env):
    env.core.join_background(3)
    assert wait_for(lambda: env.core._allow_off_thread is None)


def test_software_stop_switches_it_off(env):
    on(env)
    assert env.act("software_stop")[0] == 200
    settle(env)
    assert writes(env) == [False] and env.bridge.allow_blocked()[0] is False


def test_the_end_of_a_run_switches_it_off(env):
    on(env)
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="A", plan_seq=3)
    env.bridge.push_status(state=B.STATE_IDLE, plan_seq=3)
    settle(env)
    assert writes(env) == [False] and env.bridge.allow_blocked()[0] is False


def test_switching_execution_off_switches_it_off(env):
    on(env)
    assert env.act("set_execution_enabled", {"enabled": False})[0] == 200
    settle(env)
    assert writes(env) == [False]


def test_a_new_detect_switches_it_off(env):
    on(env)
    env.bridge.push_status(state=B.STATE_IDLE)
    assert env.act("detect")[0] == 200
    env.wait_job()
    settle(env)
    assert writes(env) == [False]


def test_ui_start_and_shutdown_switch_it_off(tmp_path):
    e = Env(tmp_path)
    try:
        e.bridge._allow_blocked = (True, e.clock())          # left on by a previous UI
        e.core.start()
        assert wait_for(lambda: e.bridge.allow_blocked()[0] is False)
        e.bridge._allow_blocked = (True, e.clock())
        e.core.shutdown(budget_s=2.0)
        assert e.bridge.allow_blocked()[0] is False
    finally:
        e.close()


def test_already_off_costs_no_write(env):
    env.act("software_stop")
    settle(env)
    assert writes(env) == []


def test_a_failing_off_write_never_disturbs_the_gates_or_the_stop(env):
    env.bridge.allow_blocked_ok = False
    env.bridge._allow_blocked = (None, env.clock())          # a coordinator without the parameter
    s, reply = env.act("software_stop")
    assert s == 200 and reply["ok"] is True and all(step["ok"] for step in reply["steps"].values())
    assert set(reply["steps"]) == {"coordinator_stop", "cancel_left", "cancel_right", "gates_off", "stop_job"}
    settle(env)
    assert env.bridge.names("set_gates")[0][1] is False
    # no retry loop when the parameter is simply missing
    assert env.bridge.count("set_allow_blocked") == 1


def test_a_failing_off_write_is_retried_while_it_reads_back_true(tmp_path):
    e = Env(tmp_path, gates_retry_s=0.01, allow_off_retries=4)
    try:
        calls = []
        orig = e.bridge.set_allow_blocked

        def flaky(enabled, timeout_s):
            calls.append(enabled)
            if len(calls) < 3:
                e.bridge.record("set_allow_blocked", enabled, timeout_s)
                return False, "transient"
            return orig(enabled, timeout_s)
        e.bridge.set_allow_blocked = flaky
        e.bridge._allow_blocked = (True, e.clock())
        e.core._allow_blocked_off("test")
        assert wait_for(lambda: e.bridge.allow_blocked()[0] is False)
        assert calls == [False, False, False]
    finally:
        e.close()


def test_auto_off_leaves_the_execution_gates_logic_alone(env):
    """The run-end gate closing still happens exactly as before (all three gates false), whatever the allow flag does."""
    env.bridge.allow_blocked_ok = False
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="A", plan_seq=3)
    env.bridge.push_status(state=B.STATE_IDLE, plan_seq=3)
    assert wait_for(lambda: env.bridge.gate_param_writes and env.bridge.gate_param_writes[-1] == (False, B.GATE_PARAMS))


def test_the_parameter_is_not_a_gate_parameter():
    assert B.ALLOW_BLOCKED_PARAM == "allow_blocked_detection"
    assert B.ALLOW_BLOCKED_PARAM not in B.GATE_PARAMS and B.ALLOW_BLOCKED_PARAM not in B.DISPLAY_PARAMS
    assert B.ALLOW_BLOCKED_PARAM not in B.gate_params_for(True) and B.ALLOW_BLOCKED_PARAM not in B.gate_params_for(False)
