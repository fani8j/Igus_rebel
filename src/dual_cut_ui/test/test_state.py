"""Fields of /api/state that the page relies on: freshness flags, run / phase progress, short step details."""
import pytest

from dual_cut_ui import bridge as B
from dual_cut_ui.core import shorten
from helpers import Env


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def test_health_and_gate_readbacks_carry_a_fresh_flag(env):
    st = env.core.state("op")
    assert st["health"]["fresh"] is True and st["gates"]["fresh"] is True and st["config"]["health_fresh_s"] == 3.0
    env.clock.advance(3.5)
    env.bridge.push_status(state=B.STATE_PLANNED, plan_seq=3, plan_retained=True, phase_validated=True)   # coordinator still alive
    st = env.core.state("op")
    assert st["coordinator"]["online"] is True
    assert st["health"]["fresh"] is False and st["gates"]["fresh"] is False
    assert st["gates"]["all_enabled"] is True            # the stale value is still sent; the page must not trust it


def test_execution_progress_follows_the_status_stream(env):
    assert env.core.state("op")["execution"] == {"active": False, "phase": None, "run_elapsed_s": None, "phase_elapsed_s": None}
    env.bridge.push_status(state=B.STATE_WAITING_BARRIER, phase="A")
    env.clock.advance(2.0)
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="A")
    env.clock.advance(5.0)
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="A")
    ex = env.core.state("op")["execution"]
    assert ex["active"] and ex["phase"] == "A" and ex["run_elapsed_s"] == pytest.approx(7.0)   # barrier wait (2 s) counts
    env.clock.advance(1.0)
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="B")
    env.clock.advance(3.0)
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="B")
    ex = env.core.state("op")["execution"]
    assert ex["phase"] == "B" and ex["phase_elapsed_s"] == pytest.approx(3.0) and ex["run_elapsed_s"] == pytest.approx(11.0)
    env.bridge.push_status(state=B.STATE_PLANNED)
    env.core.join_background()
    assert env.core.state("op")["execution"]["active"] is False


def test_execution_is_not_active_when_the_coordinator_went_quiet(env):
    env.bridge.push_status(state=B.STATE_EXECUTING, phase="A")
    env.clock.advance(5)
    assert env.core.state("op")["execution"]["active"] is False


def test_step_details_are_short_and_the_full_message_is_kept_once(env):
    long_msg = "phase order side_first: top split left_max: " + "x" * 400
    env.bridge.results["plan"] = (True, long_msg)
    env.bridge.push_status(state=B.STATE_IDLE)
    env.act("detect_and_plan")
    job = env.wait_job()
    detect, plan = job.steps
    assert plan["detail"] == "plan OK" and len(detect["detail"]) < 80 and "seam yaw" in detect["detail"]
    assert job.message == long_msg                                   # the full coordinator message lives in one place
    env.bridge.results["plan"] = (False, "refused: " + "y" * 400)
    env.act("plan")
    job = env.wait_job()
    assert len(job.steps[0]["detail"]) <= 120 and job.steps[0]["detail"].endswith("…")


def test_shorten():
    assert shorten("a  b\n c", 10) == "a b c"
    assert shorten("x" * 50, 10) == "x" * 9 + "…"
