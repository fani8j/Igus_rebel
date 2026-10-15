from dual_teleop.screen import render
from dual_teleop.selector_core import STATE_EXECUTING, Config, SelectorCore


def status_for(core, now):
    st = core.status(now)
    st["servo"] = {a: {"running": core.servo_running[a], "status": "NO_WARNING"} for a in ("left", "right")}
    st["counts"] = {"forwarded": 3, "dropped": 4}
    return st


def test_screen_says_blocked_by_the_coordinator():
    core = SelectorCore(Config(), enabled=True)
    for a in ("left", "right"):
        core.update_controller(a, True, 1.0)
        core.set_servo_running(a, True)
    core.update_scene(1.0)
    core.update_coordinator(STATE_EXECUTING, 1.0)
    text = render(status_for(core, 1.0))
    assert "TELEOP BLOCKED" in text and "coordinator EXECUTING" in text and "BOTH arms" in text
    assert "selected arm : LEFT" in text


def test_screen_says_paused_and_warns_when_the_coordinator_is_unknown():
    core = SelectorCore(Config(), enabled=False)
    text = render(status_for(core, 1.0))
    assert "TELEOP PAUSED" in text and "WARNING: coordinator status unknown" in text


def test_screen_shows_the_touch_mode_banner():
    core = SelectorCore(Config(), enabled=True)
    for a in ("left", "right"):
        core.update_controller(a, True, 1.0)
        core.set_servo_running(a, True)
    core.update_scene(1.0)
    ok, _, _ = core.set_touch(True, 1.0)
    assert ok
    text = render(status_for(core, 2.0))
    first = text.splitlines()[0]
    assert first.startswith("!!! TOUCH MODE LEFT: knife may contact carton/table")
    core.select("right", 3.0)
    text = render(status_for(core, 3.0))
    assert "TOUCH MODE" not in text.splitlines()[0]
    assert "touch mode   : OFF - arm switched" in text
