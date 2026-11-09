"""Orderly shutdown: gates false within a bounded time, logged where it survives a closed stderr, no hostage taking."""
import logging
import sys
import threading
import time

import pytest

from dual_cut_ui import server
from dual_cut_ui.camera import CameraHub, DEFAULT_CAMERAS
from helpers import Env


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    @property
    def text(self):
        return "\n".join(self.lines)


@pytest.fixture
def caplog():
    """Own capture on the dual_cut_ui logger (importing launch_ros swaps the logger class and stops propagation)."""
    log = logging.getLogger("dual_cut_ui")
    h = ListHandler()
    old_level = log.level
    log.addHandler(h)
    log.setLevel(logging.INFO)

    class Cap:
        text = property(lambda self: h.text)

        def set_level(self, *a, **k):
            pass
    yield Cap()
    log.removeHandler(h)
    log.setLevel(old_level)


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def test_core_shutdown_closes_the_gates_and_logs_it(env, caplog):
    env.core.shutdown()
    assert [c[1] for c in env.bridge.names("set_gates")] == [False]
    text = caplog.text
    assert "UI shutdown: closing the execution gates" in text
    assert "execution gates -> false (UI shutdown)" in text
    assert "UI shutdown finished" in text and "gates: closed" in text


def test_core_shutdown_is_bounded_when_the_coordinator_hangs(env, caplog):
    gate = threading.Event()
    env.bridge.block["gates"] = gate
    t0 = time.monotonic()
    try:
        env.core.shutdown(budget_s=1.0)
    finally:
        gate.set()
    assert time.monotonic() - t0 < 1.6
    assert "gates: NOT confirmed" in caplog.text


def test_core_shutdown_kills_a_detector_that_ignores_sigterm_quickly(tmp_path):
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('up', flush=True)\ntime.sleep(120)\n"
    e = Env(tmp_path, detect_command=[sys.executable, "-c", code], kill_grace_s=30.0)       # long grace: shutdown must not wait for it
    try:
        e.act("detect")
        runner = e.wait_job_started_process().runner
        time.sleep(0.5)
        t0 = time.monotonic()
        e.core.shutdown(budget_s=4.0)
        assert time.monotonic() - t0 < 3.5
        assert runner.wait_gone(2.0)
    finally:
        e.close()


def test_gates_are_closed_before_the_slow_job_cleanup(tmp_path):
    cmd = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(60)"]
    e = Env(tmp_path, detect_command=cmd)
    try:
        e.act("detect")
        e.wait_job_started_process()
        e.core.shutdown()
        assert e.bridge.count("set_gates") == 1
    finally:
        e.close()


def test_run_shutdown_runs_steps_in_order_logs_each_and_the_total():
    order, lines = [], []
    res = server.run_shutdown([("one", lambda: order.append(1)), ("two", lambda: order.append(2))], lines.append, budget_s=2.0)
    assert order == [1, 2] and res == {"one": "ok", "two": "ok"}
    assert lines[0].startswith("dual-cut UI shutting down") and "step one: ok" in lines[1] and "finished in" in lines[-1]


def test_run_shutdown_survives_hanging_and_failing_steps_within_the_budget():
    lines, release = [], threading.Event()

    def boom():
        raise RuntimeError("kaput")
    t0 = time.monotonic()
    res = server.run_shutdown([("hang", lambda: release.wait(30)), ("boom", boom), ("after", lambda: None)], lines.append, budget_s=1.0)
    release.set()
    assert time.monotonic() - t0 < 1.5
    assert res["hang"] == "timeout"
    assert res["boom"] in ("skipped",) or res["boom"].startswith("error")          # the hang may have used the whole budget
    assert res["after"] in ("ok", "skipped")


def test_run_shutdown_skips_steps_when_out_of_time():
    release = threading.Event()
    res = server.run_shutdown([("a", lambda: release.wait(30)), ("b", lambda: None)], lambda m: None, budget_s=0.3)
    release.set()
    assert res == {"a": "timeout", "b": "skipped"}


def test_the_real_sequence_with_a_fake_bridge(env, caplog):
    """The same steps main() runs, in the same order: gates false first, within the 4 s budget."""
    hub = CameraHub(env.bridge, DEFAULT_CAMERAS, clock=env.clock)
    hub.request("scene", "small")
    from dual_cut_ui.auth import Auth
    ctx = server.Context(env.core, hub, Auth(1, token="t"))
    lines = []
    t0 = time.monotonic()
    res = server.run_shutdown([("core (gates false, job stopped)", lambda: env.core.shutdown(budget_s=server.SHUTDOWN_BUDGET_S - 1.0)),
                               ("cameras", hub.shutdown), ("pools", ctx.shutdown)], lines.append)
    assert time.monotonic() - t0 < server.SHUTDOWN_BUDGET_S
    assert set(res.values()) == {"ok"}
    assert [c[1] for c in env.bridge.names("set_gates")] == [False]
    assert not env.bridge.subscriptions
    assert "execution gates -> false (UI shutdown)" in caplog.text


def test_shutdown_lines_reach_the_log_file_even_if_stderr_is_broken(tmp_path, env):
    path = tmp_path / "ui.log"
    server.setup_logging(str(path))
    try:
        class Broken:
            def write(self, *_):
                raise BrokenPipeError("stderr pipe closed")

            def flush(self):
                raise BrokenPipeError("stderr pipe closed")
        stream = next(h for h in logging.getLogger("dual_cut_ui").handlers if type(h) is logging.StreamHandler)
        stream.stream = Broken()
        env.core.shutdown()                                   # must not raise although every stderr write fails
        server.flush_logs()
        text = path.read_text()
        assert "execution gates -> false (UI shutdown)" in text and "UI shutdown finished" in text
    finally:
        _reset_logging()


def test_setup_logging_without_a_writable_path_still_works():
    server.setup_logging("/proc/nope/x.log")
    try:
        logging.getLogger("dual_cut_ui").info("still fine")
    finally:
        _reset_logging()


def test_main_wires_signals_and_the_shutdown_order():
    import inspect
    src = inspect.getsource(server.main)
    for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        assert f"signal.{sig}" in src
    assert "early_signal" in src and "stop_requested" in src                 # a signal during start-up is not lost
    assert src.index("run_shutdown(") < src.index("executor.shutdown()") < src.rindex("rclpy.shutdown()")
    steps_src = inspect.getsource(server.shutdown_steps)
    assert steps_src.index("core.shutdown(budget_s=") < steps_src.index("hub.shutdown")  # gates first
    assert "SignalHandlerOptions.NO" in src                                 # the lead's patch stays
    assert server.SHUTDOWN_BUDGET_S <= 4.0


def _reset_logging():
    log = logging.getLogger("dual_cut_ui")
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log.propagate = True
    logging.raiseExceptions = True
