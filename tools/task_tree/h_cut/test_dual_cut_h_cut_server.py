"""Tests for dual_cut_h_cut_server.py with an in-process fake coordinator and a fake DetectBox server. Nothing real is started.

Run (isolated domain, see the task rules):
    export ROS_DOMAIN_ID=227 ROS_LOCALHOST_ONLY=1
    source /opt/ros/humble/setup.bash; source <ws>/install/setup.bash; source <ws>/install_bt/local_setup.bash
    cd tools/task_tree/h_cut && nice -n 19 python3 -m pytest -p no:anyio test_dual_cut_h_cut_server.py -q
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest
import rclpy
from compal_box_msgs.action import DetectBox
from compal_cut_task_msgs.action import HCut
from dual_cut_msgs.msg import CoordinatorStatus
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

import dual_cut_h_cut_server as srv

S = CoordinatorStatus
PID = os.getpid()
COORD = f"/fake_cc_{PID}"
DETECT = f"/fake_detect_{PID}"
HCUT = f"/h_cut_test_{PID}"


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    dom = os.environ.get("ROS_DOMAIN_ID", "")
    assert dom and dom not in ("11", "12") and os.environ.get("ROS_LOCALHOST_ONLY") == "1", \
        "set ROS_DOMAIN_ID (not 11) and ROS_LOCALHOST_ONLY=1 before running these tests"
    rclpy.init()
    yield
    rclpy.shutdown()


class FakeCoordinator(Node):
    """The four Trigger services + status / exec_report publishers with scriptable behaviour."""

    def __init__(self):
        super().__init__(f"fake_cc_{PID}", namespace="")
        g = ReentrantCallbackGroup()
        self.calls = {"plan": 0, "execute": 0, "stop": 0, "rearm": 0}
        self.param_calls = 0
        self.state, self.phase, self.latched, self.reason = S.STATE_IDLE, "", False, ""
        self.retained = False
        self.publish_status_enabled = True
        self.plan_result = (True, "plan ok: fake")
        self.execute_result = (True, "dispatch accepted")
        self.execute_script = []      # list of ("state", state, phase) | ("report", dict) | ("stop", reason) | ("sleep", s) | ("hang",)
        for name in ("plan", "execute", "stop", "rearm"):
            self.create_service(Trigger, f"{COORD}/{name}", getattr(self, f"_{name}"), callback_group=g)
        # parameter services of this node are the real ones; count writes by wrapping the handler is not possible in rclpy,
        # so the tests instead check that the server has no parameter client at all (test_server_has_no_rearm_or_param_client).
        self.status_pub = self.create_publisher(S, f"{COORD}/status", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.report_pub = self.create_publisher(String, f"{COORD}/exec_report", QoSProfile(
            depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.05, self.publish_status, callback_group=g)
        self.script_thread = None

    def publish_status(self):
        if not self.publish_status_enabled:
            return
        m = S()
        m.header.stamp = self.get_clock().now().to_msg()
        m.state, m.phase = self.state, self.phase
        m.plan_retained = m.phase_validated = self.retained
        m.plan_seq = 1 if self.retained else 0
        m.stop_latched, m.stop_reason = self.latched, self.reason
        m.barrier_ok, m.barrier_message = True, "ok"
        self.status_pub.publish(m)

    def report(self, phase, completed=True, **extra):
        doc = {"phase": phase, "completed": completed, "stop_latched": self.latched, "stop_reason": self.reason}
        doc.update(extra)
        self.report_pub.publish(String(data=json.dumps(doc)))

    def _plan(self, _q, r):
        self.calls["plan"] += 1
        r.success, r.message = self.plan_result
        if r.success:
            self.state, self.retained = S.STATE_PLANNED, True
        return r

    def _execute(self, _q, r):
        self.calls["execute"] += 1
        r.success, r.message = self.execute_result
        if r.success:
            self.state = S.STATE_WAITING_BARRIER
            self.script_thread = threading.Thread(target=self._run_script, daemon=True)
            self.script_thread.start()
        return r

    def _run_script(self):
        for step in self.execute_script:
            if step[0] == "state":
                self.state, self.phase = step[1], step[2]
            elif step[0] == "report":
                self.report(**step[1])
            elif step[0] == "stop":
                self.state, self.latched, self.reason = S.STATE_STOPPED, True, step[1]
            elif step[0] == "sleep":
                time.sleep(step[1])
            elif step[0] == "silence":
                self.publish_status_enabled = False
            elif step[0] == "hang":
                while self.state in (S.STATE_WAITING_BARRIER, S.STATE_EXECUTING) and not self.latched:
                    time.sleep(0.05)
                return

    def _stop(self, _q, r):
        self.calls["stop"] += 1
        self.state, self.latched = S.STATE_STOPPED, True
        self.reason = self.reason or "operator ~/stop"
        r.success, r.message = True, "stop latched"
        return r

    def _rearm(self, _q, r):
        self.calls["rearm"] += 1
        self.state, self.latched, self.reason = S.STATE_IDLE, False, ""
        r.success, r.message = True, "rearmed"
        return r


class FakeDetect:
    def __init__(self, node, mode="ok"):
        self.mode, self.goals, self.cancelled = mode, 0, 0
        self.server = ActionServer(node, DetectBox, DETECT, execute_callback=self._exec,
                                   goal_callback=lambda _g: GoalResponse.ACCEPT,
                                   cancel_callback=lambda _h: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup())

    def _exec(self, gh):
        self.goals += 1
        res = DetectBox.Result()
        if self.mode == "hang":
            while not gh.is_cancel_requested:
                time.sleep(0.05)
            self.cancelled += 1
            gh.canceled()
            res.status, res.reason = "cancelled", "cancelled"
            return res
        if self.mode == "reject":
            res.success, res.status, res.reason = False, "rejected", "no carton found in the image"
        else:
            res.success, res.status, res.reason, res.cut_permission = True, "accepted", "ok", "allowed"
        gh.succeed()
        return res


class Env:
    def __init__(self, detect_mode="ok", **params_kw):
        self.cc = FakeCoordinator()
        self.det_node = Node(f"fake_detect_{PID}")
        self.det = FakeDetect(self.det_node, detect_mode)
        params = {"h_cut_action": HCUT, "coordinator": COORD, "detect_action": DETECT, "preflight_wait_s": 3.0,
                  "status_timeout_s": 1.0, "exec_report_grace_s": 1.0, "detect_cancel_wait_s": 3.0}
        params.update(params_kw)
        self.server = srv.DualCutHCutServer(node_name=f"dual_cut_h_cut_server_test_{PID}",
                                            parameter_overrides=[rclpy.parameter.Parameter(k, value=v) for k, v in params.items()])
        self.client_node = Node(f"hcut_client_{PID}")
        self.client = ActionClient(self.client_node, HCut, HCUT)
        self.ex = MultiThreadedExecutor(num_threads=10)
        for n in (self.cc, self.det_node, self.server, self.client_node):
            self.ex.add_node(n)
        self.thread = threading.Thread(target=self.ex.spin, daemon=True)
        self.thread.start()
        assert self.client.wait_for_server(timeout_sec=10.0)
        self.feedback = []

    def close(self):
        self.ex.shutdown()
        self.thread.join(timeout=5)
        for n in (self.cc, self.det_node, self.server, self.client_node):
            n.destroy_node()

    def send(self, execute=False, understand=False, wait=True, **kw):
        goal = HCut.Goal(execute=execute, i_understand_this_moves_the_real_robot=understand, attempt=kw.pop("attempt", 1),
                         recut_hint_json=kw.pop("hint", ""), timeout_sec=kw.pop("timeout_sec", 0.0))
        fut = self.client.send_goal_async(goal, feedback_callback=lambda m: self.feedback.append(m.feedback.stage))
        t = time.monotonic()
        while not fut.done() and time.monotonic() - t < 10:
            time.sleep(0.02)
        handle = fut.result()
        if not wait:
            return handle
        return handle, self.result(handle)

    def result(self, handle, timeout=30.0):
        assert handle is not None and handle.accepted
        rf = handle.get_result_async()
        t = time.monotonic()
        while not rf.done() and time.monotonic() - t < timeout:
            time.sleep(0.02)
        assert rf.done(), "no result"
        return rf.result()


@pytest.fixture
def make_env():
    envs = []

    def make(**kw):
        e = Env(**kw)
        envs.append(e)
        return e
    yield make
    for e in envs:
        e.close()


def assert_safe(env):
    assert env.cc.calls["rearm"] == 0, "the server must never call ~/rearm"


# ------------------------------------------------------------------------------------------------ tests
def test_plan_only_success(make_env):
    env = make_env()
    _, r = env.send()
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (True, HCut.Result.OK, False)
    assert res.message.startswith("plan only: plan ok: fake")
    assert res.run_folder == ""
    assert env.det.goals == 1 and env.cc.calls["plan"] == 1 and env.cc.calls["execute"] == 0 and env.cc.calls["stop"] == 0
    assert "preflight" in env.feedback and "detect" in env.feedback and "plan" in env.feedback
    assert_safe(env)


def test_skip_detect(make_env):
    env = make_env(skip_detect=True)
    _, r = env.send()
    assert r.result.success and env.det.goals == 0
    assert_safe(env)


def test_detect_rejected(make_env):
    env = make_env(detect_mode="reject")
    _, r = env.send()
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.PERCEPTION_FAILED, False)
    assert "no carton found" in res.message and "rejected" in res.message
    assert env.cc.calls["plan"] == 0
    assert_safe(env)


def test_detect_timeout(make_env):
    env = make_env(detect_mode="hang", detect_timeout_s=0.5)
    _, r = env.send()
    assert r.result.error_code == HCut.Result.PERCEPTION_FAILED and "timeout" in r.result.message
    assert env.det.cancelled == 1 and env.cc.calls["plan"] == 0
    assert_safe(env)


def test_plan_refused(make_env):
    env = make_env()
    env.cc.plan_result = (False, "refused: stale snapshot\n" + "x" * 2000)
    _, r = env.send()
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.PLANNING_FAILED, False)
    assert "stale snapshot" in res.message and "\n" not in res.message and len(res.message) < 450
    assert env.cc.calls["execute"] == 0
    assert_safe(env)


def test_execute_refused_gates_off(make_env):
    env = make_env()
    env.cc.execute_result = (False, "refused: execution_enabled is false")
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.EXECUTION_FAILED, False)
    assert "execution_enabled is false" in res.message
    assert env.cc.calls["execute"] == 1 and env.cc.calls["stop"] == 0   # refused: nothing started, nothing to stop
    assert_safe(env)


def test_execute_success_phases_a_and_b(make_env):
    env = make_env()
    env.cc.execute_script = [
        ("state", S.STATE_WAITING_BARRIER, "A"), ("sleep", 0.2), ("state", S.STATE_EXECUTING, "A"), ("sleep", 0.3),
        ("report", {"phase": "A"}),
        ("state", S.STATE_WAITING_BARRIER, "B"), ("sleep", 0.2), ("state", S.STATE_EXECUTING, "B"), ("sleep", 0.3),
        ("report", {"phase": "B"}), ("state", S.STATE_PLANNED, ""),
    ]
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (True, HCut.Result.OK, True), res.message
    assert "A+B" in res.message
    assert env.cc.calls["execute"] == 1 and env.cc.calls["stop"] == 0
    assert any(s.startswith("execute phase A") for s in env.feedback)
    assert any(s.startswith("execute phase B") for s in env.feedback)
    assert_safe(env)


def test_planned_again_without_report_is_not_success(make_env):
    """A barrier refusal returns the coordinator to PLANNED without a stop: must not count as done."""
    env = make_env()
    env.cc.execute_script = [("state", S.STATE_WAITING_BARRIER, "A"), ("sleep", 0.2), ("state", S.STATE_PLANNED, "A")]
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.EXECUTION_FAILED, True)
    assert "exec_report" in res.message
    assert_safe(env)


def test_phase_b_not_completed(make_env):
    env = make_env()
    env.cc.execute_script = [
        ("state", S.STATE_EXECUTING, "A"), ("sleep", 0.2), ("report", {"phase": "A"}),
        ("state", S.STATE_WAITING_BARRIER, "B"), ("sleep", 0.2), ("state", S.STATE_PLANNED, "B"),
    ]
    _, r = env.send(execute=True, understand=True)
    assert r.result.success is False and r.result.error_code == HCut.Result.EXECUTION_FAILED
    assert_safe(env)


def test_coordinator_stopped_during_execution(make_env):
    env = make_env()
    env.cc.execute_script = [("state", S.STATE_EXECUTING, "A"), ("sleep", 0.3), ("stop", "monitor: tracking error 0.3 rad")]
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.EXECUTION_FAILED, True)
    assert "tracking error 0.3 rad" in res.message
    assert env.cc.calls["stop"] == 0   # already latched by the coordinator itself
    assert_safe(env)


def test_status_stale_during_execution_calls_stop(make_env):
    env = make_env()
    env.cc.execute_script = [("state", S.STATE_EXECUTING, "A"), ("sleep", 0.3), ("silence",)]
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.EXECUTION_FAILED, True)
    assert "stale" in res.message
    assert env.cc.calls["stop"] == 1
    assert_safe(env)


def test_cancel_during_execution_calls_stop_exactly_once(make_env):
    env = make_env()
    env.cc.execute_script = [("state", S.STATE_EXECUTING, "A"), ("hang",)]
    handle = env.send(execute=True, understand=True, wait=False)
    t = time.monotonic()
    while env.cc.state != S.STATE_EXECUTING and time.monotonic() - t < 10:
        time.sleep(0.02)
    time.sleep(0.3)
    cf = handle.cancel_goal_async()
    r = env.result(handle)
    assert cf.done()
    res = r.result
    assert r.status == 5   # STATUS_CANCELED
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.CANCELLED, True)
    assert env.cc.calls["stop"] == 1
    assert_safe(env)


def test_cancel_during_detect_cancels_detector_no_stop(make_env):
    env = make_env(detect_mode="hang")
    handle = env.send(wait=False)
    time.sleep(0.8)
    handle.cancel_goal_async()
    r = env.result(handle)
    assert r.result.error_code == HCut.Result.CANCELLED and r.result.did_move is False
    assert env.det.cancelled == 1
    assert env.cc.calls["stop"] == 0 and env.cc.calls["execute"] == 0
    assert_safe(env)


def test_timeout_during_execution_calls_stop(make_env):
    env = make_env()
    env.cc.execute_script = [("state", S.STATE_EXECUTING, "A"), ("hang",)]
    _, r = env.send(execute=True, understand=True, timeout_sec=1.5)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.TIMEOUT, True)
    assert env.cc.calls["stop"] == 1
    assert_safe(env)


@pytest.mark.parametrize("latched", [True, False])
def test_stopped_at_preflight_never_rearms(make_env, latched):
    env = make_env()
    env.cc.state, env.cc.latched, env.cc.reason = S.STATE_STOPPED, latched, "operator ~/stop"
    _, r = env.send(execute=True, understand=True)
    res = r.result
    assert (res.success, res.did_move) == (False, False)
    assert res.error_code == HCut.Result.EXECUTION_FAILED
    assert "coordinator is STOPPED (operator ~/stop); the operator must rearm" in res.message
    assert env.det.goals == 0 and env.cc.calls["plan"] == 0 and env.cc.calls["execute"] == 0
    assert_safe(env)


def test_no_status_fails_preflight(make_env):
    env = make_env(preflight_wait_s=0.5)
    env.cc.publish_status_enabled = False
    _, r = env.send()
    assert r.result.error_code == HCut.Result.INTERNAL_ERROR and "status" in r.result.message
    assert env.det.goals == 0
    assert_safe(env)


def test_plan_from_planned_state_without_rearm(make_env):
    env = make_env()
    env.cc.state, env.cc.retained = S.STATE_PLANNED, True
    _, r = env.send()
    assert r.result.success and env.cc.calls["plan"] == 1
    assert_safe(env)


@pytest.mark.parametrize("execute,understand", [(True, False), (False, True)])
def test_bad_goal(make_env, execute, understand):
    env = make_env()
    _, r = env.send(execute=execute, understand=understand)
    res = r.result
    assert (res.success, res.error_code, res.did_move) == (False, HCut.Result.BAD_GOAL, False)
    assert env.det.goals == 0 and env.cc.calls["plan"] == 0 and env.cc.calls["execute"] == 0
    assert_safe(env)


def test_second_goal_rejected_while_one_runs(make_env):
    env = make_env(detect_mode="hang")
    h1 = env.send(wait=False)
    time.sleep(0.5)
    h2 = env.send(wait=False)
    assert h2 is not None and not h2.accepted
    h1.cancel_goal_async()
    r = env.result(h1)
    assert r.result.error_code == HCut.Result.CANCELLED
    # and a new goal is accepted again afterwards
    env.det.mode = "ok"
    _, r2 = env.send()
    assert r2.result.success
    assert_safe(env)


def test_recut_hint_ignored_full_flow(make_env):
    env = make_env()
    _, r = env.send(attempt=2, hint='{"x": 1}')
    assert r.result.success and env.cc.calls["plan"] == 1
    assert_safe(env)


def test_server_has_no_rearm_or_param_client():
    code = Path(srv.__file__).read_text()
    assert not re.search(r"create_client\([^)]*rearm", code)
    assert "SetParameters" not in code and "set_parameters" not in code and 'f"{coord}/rearm"' not in code
