"""Action whitelist and the single-operator lock."""
import pytest

from dual_cut_ui.core import ACTIONS
from helpers import Env

OPERATOR_ONLY = sorted(a for a, (_m, needs) in ACTIONS.items() if needs)
ARGS = {"set_execution_enabled": {"enabled": True}, "set_allow_blocked_detection": {"enabled": True}, "execute": {"plan_seq": 3}}


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def test_whitelist_is_exactly_the_approved_set():
    assert set(ACTIONS) == {"take_control", "release_control", "software_stop", "detect", "plan", "detect_and_plan",
                            "set_execution_enabled", "set_allow_blocked_detection", "execute", "rearm", "cancel_job"}
    assert {a for a, (_m, needs) in ACTIONS.items() if not needs} == {"software_stop", "take_control"}


@pytest.mark.parametrize("name", ["call_service", "set_parameter", "publish", "trigger", "ros2", "shell", "exec", "",
                                  "Software_Stop", "software_stop ", "__init__", "start", None, 5, ["detect"], {"a": 1}])
def test_unknown_actions_are_400_and_touch_nothing(env, name):
    status, reply = env.act(name, {"service": "/x", "name": "execution_enabled", "value": True})
    assert status == 400 and reply["ok"] is False
    assert env.bridge.calls == []


def test_args_must_be_an_object(env):
    assert env.act("rearm", "boom")[0] == 400
    assert env.act("rearm", [1])[0] == 400
    assert env.bridge.calls == []


def test_extra_args_are_ignored_not_passed_through(env):
    status, _ = env.act("rearm", {"service": "/controller_manager/switch_controller", "param": "x"})
    assert status == 200
    assert [c[:2] for c in env.bridge.calls] == [("trigger", "rearm")]


def test_first_polling_page_becomes_operator_second_is_read_only(env):
    env.core.touch("other")
    assert env.core.is_operator("op") and not env.core.is_operator("other")
    st = env.core.state("other")
    assert st["operator"]["is_operator"] is False and st["operator"]["other_operator_active"] is True


def test_present_view_never_auto_claims(tmp_path):
    e = Env(tmp_path)
    try:
        e.core._holder = None
        e.core.touch("projector", "present")
        assert not e.core.is_operator("projector")
        e.core.touch("laptop")
        assert e.core.is_operator("laptop")
    finally:
        e.close()


@pytest.mark.parametrize("action", OPERATOR_ONLY)
def test_read_only_page_gets_403_for_every_operator_action(env, action):
    env.core.touch("other")
    before = len(env.bridge.calls)
    status, reply = env.act(action, ARGS.get(action, {}), page="other")
    assert status == 403 and "read-only" in reply["message"]
    assert len(env.bridge.calls) == before


def test_page_without_id_is_read_only_but_can_stop(env):
    assert env.act("plan", {}, page=None)[0] == 403
    status, reply = env.act("software_stop", {}, page=None)
    assert status == 200 and env.bridge.count("trigger") == 1


def test_software_stop_from_a_read_only_page_works(env):
    env.core.touch("other")
    status, reply = env.act("software_stop", {}, page="other")
    assert status == 200 and reply["ok"] is True
    assert [c[1] for c in env.bridge.names("trigger")] == ["stop"]
    assert {c[1] for c in env.bridge.names("cancel")} == {"left", "right"}
    assert env.bridge.names("set_gates")[0][1] is False


def test_grace_expiry_lets_another_page_claim(env):
    env.core.touch("other")
    assert not env.core.is_operator("other")
    env.clock.advance(env.cfg.control_grace_s + 1)        # the holder stopped polling
    env.core.touch("other")
    assert env.core.is_operator("other") and not env.core.is_operator("op")
    assert env.act("rearm", page="op")[0] == 403


def test_holder_that_only_paused_keeps_control_if_nobody_claimed(env):
    env.clock.advance(env.cfg.control_grace_s + 5)
    assert env.core.is_operator("op")
    env.core.touch("op")
    assert env.core.operator_info("op")["holder_idle_s"] == 0


def test_take_control_needs_force_while_holder_is_active(env):
    status, reply = env.act("take_control", {}, page="other")
    assert status == 409 and reply["needs_force"] is True
    assert env.core.is_operator("op")
    status, reply = env.act("take_control", {"force": "yes"}, page="other")
    assert status == 400
    status, reply = env.act("take_control", {"force": True}, page="other")
    assert status == 200 and env.core.is_operator("other") and not env.core.is_operator("op")
    assert env.act("plan", {}, page="op")[0] == 403


def test_take_control_after_grace_needs_no_force(env):
    env.clock.advance(env.cfg.control_grace_s + 1)
    assert env.act("take_control", {}, page="other")[0] == 200
    assert env.core.is_operator("other")


def test_take_control_when_free_and_own_page(env):
    assert env.act("release_control")[0] == 200
    assert not env.core.is_operator("op")
    assert env.act("take_control", {}, page="x")[0] == 200
    assert env.act("take_control", {}, page="x")[1]["message"].startswith("you are already")
    assert env.act("take_control", {}, page=None)[0] == 400


def test_state_reports_operator_info(env):
    st = env.core.state("op")
    assert st["operator"]["is_operator"] is True and st["operator"]["control_grace_s"] == env.cfg.control_grace_s
