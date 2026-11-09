"""RosBridge.detect / set_gates / set_schedule_mode logic with stub action-client and service objects (no ROS graph, no node)."""
import sys
import threading
import types

import pytest

from dual_cut_ui import bridge as B

pytest.importorskip("rclpy")
pytest.importorskip("dual_cut_msgs.msg")
from dual_cut_ui import ros_bridge  # noqa: E402

ACTION = "/box_seam_detector/detect_box"
HINT = "source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash"


class Fut:
    def __init__(self, value=None, exc=None):
        self.value, self.exc = value, exc

    def add_done_callback(self, cb):
        cb(self)

    def result(self):
        if self.exc:
            raise self.exc
        return self.value


class Handle:
    def __init__(self, accepted=True, result=None, status=4):
        self.accepted, self._result, self.status = accepted, result, status
        self.cancels = 0

    def get_result_async(self):
        return Fut(types.SimpleNamespace(result=self._result, status=self.status))

    def cancel_goal_async(self):
        self.cancels += 1
        return Fut()


class Client:
    def __init__(self, ready=True, handle=None, feedback=()):
        self.ready, self.handle, self.feedback, self.sent = ready, handle, feedback, []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, feedback_callback=None):
        self.sent.append(goal)
        for stage, attempt, message in self.feedback:
            feedback_callback(types.SimpleNamespace(feedback=types.SimpleNamespace(stage=stage, attempt=attempt, message=message)))
        return Fut(self.handle)


def stub_bridge(client, monkeypatch, import_error=None):
    rb = ros_bridge.RosBridge.__new__(ros_bridge.RosBridge)
    rb._lock = threading.Lock()
    rb._detect_clients = {ACTION: client}
    rb._detect_import_error = import_error
    rb._detect_type = None if import_error else types.SimpleNamespace(Goal=lambda: types.SimpleNamespace())
    rb.node = types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(warn=lambda *a, **k: None))
    return rb


RESULT = types.SimpleNamespace(success=True, status="accepted", reason="ok", reasons=["a", "b"], cut_permission="allowed",
                               snapshot_id="s", snapshot_path="/x.json", capture_id="c", seam_status="ok", attempts=2, elapsed_s=3.5)


def test_ros_bridge_detect_success_sets_the_goal_fields_and_streams_feedback(monkeypatch):
    client = Client(handle=Handle(result=RESULT), feedback=[("capture", 1, "go")])
    rb = stub_bridge(client, monkeypatch)
    seen = []
    ok, info = rb.detect(ACTION, B.DETECT_GOAL, lambda *a: seen.append(a), threading.Event(), 10)
    assert ok is True and seen == [("capture", 1, "go")]
    g = client.sent[0]
    assert (g.max_attempts, g.allow_needs_review, g.allow_cut_blocked, g.dry_run) == (0, False, False, False)
    assert info["status"] == "accepted" and info["reasons"] == ["a", "b"] and info["attempts"] == 2 and "cancelled" not in info


def test_ros_bridge_detect_server_unavailable(monkeypatch):
    monkeypatch.setattr(ros_bridge, "DETECT_SERVER_WAIT_S", 0.2)
    rb = stub_bridge(Client(ready=False), monkeypatch)
    ok, info = rb.detect(ACTION, B.DETECT_GOAL, lambda *a: None, threading.Event(), 10)
    assert ok is False and info == {"error": "detector action server not available (is box_seam_detector running?)"}


def test_ros_bridge_detect_goal_rejected(monkeypatch):
    rb = stub_bridge(Client(handle=Handle(accepted=False)), monkeypatch)
    ok, info = rb.detect(ACTION, B.DETECT_GOAL, lambda *a: None, threading.Event(), 10)
    assert ok is False and info == {"error": "detector busy / goal rejected"}


def test_ros_bridge_detect_cancel_sends_cancel_and_reports_cancelled(monkeypatch):
    class Late(Handle):
        def get_result_async(self):
            self.fut = types.SimpleNamespace(cbs=[])
            outer = self

            class F:
                def add_done_callback(self, cb):
                    outer.fut.cbs.append(cb)

                def result(self):
                    return types.SimpleNamespace(result=RESULT_CANCELLED, status=5)
            self.F = F()
            return self.F

        def cancel_goal_async(self):
            super().cancel_goal_async()
            for cb in self.fut.cbs:
                cb(self.F)
            return Fut()
    RESULT_CANCELLED = types.SimpleNamespace(**{**vars(RESULT), "success": False, "status": "cancelled"})
    h = Late()
    rb = stub_bridge(Client(handle=h), monkeypatch)
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    ok, info = rb.detect(ACTION, B.DETECT_GOAL, lambda *a: None, cancel, 10)
    assert ok is False and info["cancelled"] is True and h.cancels == 1


def test_ros_bridge_import_error_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "compal_box_msgs.action", None)          # makes the import fail
    rb = stub_bridge(None, monkeypatch)
    rb._detect_type = None
    msg = rb.detect_import_error()
    assert msg and HINT in msg and "DetectBox" in msg
    assert rb.detect_import_error() == msg                                     # cached
    ok, info = rb.detect(ACTION, B.DETECT_GOAL, lambda *a: None, threading.Event(), 10)
    assert ok is False and info == {"error": msg}


# ---------------------------------------------------------------- gate / schedule parameter writes
class ParamClient:
    def __init__(self, ok=True):
        self.ok, self.requests = ok, []


def param_bridge(ok=True, reply=True):
    rb = ros_bridge.RosBridge.__new__(ros_bridge.RosBridge)
    rb._set_params = ParamClient(ok)
    rb.polled = 0
    rb._poll_gates = lambda: setattr(rb, "polled", rb.polled + 1)
    rb._poll_allow_blocked = lambda: setattr(rb, "polled_allow", getattr(rb, "polled_allow", 0) + 1)

    def call(client, request, timeout_s, require_ready=True):
        client.requests.append(request)
        if not reply:
            return None, "no answer"
        results = [types.SimpleNamespace(successful=ok, reason="" if ok else "read only") for _ in request.parameters]
        return types.SimpleNamespace(results=results), ""
    rb._call = call
    return rb


def written(rb):
    req = rb._set_params.requests[-1]
    return [(p.name, p.value.bool_value if p.value.type == 1 else p.value.string_value) for p in req.parameters]


def test_enable_writes_exactly_the_two_enable_params_true():
    rb = param_bridge()
    ok, msg = rb.set_gates(True, 1.0)
    assert ok and written(rb) == [("execution_enabled", True), ("execute_phase_a_enabled", True)]
    assert "execute_phase_b_enabled" not in msg


def test_disable_writes_all_three_gate_params_false():
    rb = param_bridge()
    ok, _ = rb.set_gates(False, 1.0)
    assert ok and written(rb) == [("execution_enabled", False), ("execute_phase_a_enabled", False), ("execute_phase_b_enabled", False)]


def test_a_rejected_gate_write_is_reported():
    ok, msg = param_bridge(ok=False).set_gates(False, 1.0)
    assert not ok and "execution gates not set" in msg
    ok, msg = param_bridge(reply=False).set_gates(True, 1.0)
    assert not ok and "no answer" in msg


def test_set_schedule_mode_writes_a_string_parameter():
    rb = param_bridge()
    ok, msg = rb.set_schedule_mode("sequence", 1.0)
    assert ok and written(rb) == [("schedule_mode", "sequence")] and rb.polled == 1
    ok, msg = rb.set_schedule_mode("phases", 1.0)
    assert ok and written(rb) == [("schedule_mode", "phases")]
    n = len(rb._set_params.requests)
    ok, msg = rb.set_schedule_mode("sideways", 1.0)
    assert not ok and "schedule_mode must be one of" in msg and len(rb._set_params.requests) == n


def test_set_schedule_mode_failures():
    ok, msg = param_bridge(ok=False).set_schedule_mode("sequence", 1.0)
    assert not ok and "schedule_mode not set: read only" in msg
    ok, msg = param_bridge(reply=False).set_schedule_mode("sequence", 1.0)
    assert not ok and "no answer" in msg


def test_set_allow_blocked_writes_one_bool_parameter_and_never_a_gate():
    rb = param_bridge()
    ok, msg = rb.set_allow_blocked(True, 1.0)
    assert ok and written(rb) == [("allow_blocked_detection", True)] and rb.polled_allow == 1 and rb.polled == 0
    ok, _ = rb.set_allow_blocked(False, 1.0)
    assert ok and written(rb) == [("allow_blocked_detection", False)]
    ok, msg = param_bridge(ok=False).set_allow_blocked(True, 1.0)
    assert not ok and "allow_blocked_detection not set: read only" in msg
    ok, msg = param_bridge(reply=False).set_allow_blocked(True, 1.0)
    assert not ok and "no answer" in msg


def test_allow_blocked_poll_is_separate_and_tolerates_a_coordinator_without_the_parameter():
    from rcl_interfaces.msg import ParameterType, ParameterValue

    rb = ros_bridge.RosBridge.__new__(ros_bridge.RosBridge)
    rb._lock = threading.Lock()
    rb._allow_blocked = None
    rb._pending = {}
    rb.node = types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(warn=lambda *a, **k: None))
    sent = []

    class Getter:
        def __init__(self, values=None, exc=None):
            self.values, self.exc = values, exc

        def service_is_ready(self):
            return True

        def call_async(self, req):
            sent.append(list(req.names))
            return Fut(types.SimpleNamespace(values=self.values) if self.exc is None else None, self.exc)
    rb._get_params = Getter([ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True)])
    rb._poll_allow_blocked()
    assert sent == [["allow_blocked_detection"]] and rb.allow_blocked()[0] is True
    rb._pending.clear()
    rb._get_params = Getter([])                                      # undeclared parameter: empty reply
    rb._poll_allow_blocked()
    assert rb.allow_blocked()[0] is None
    rb._pending.clear()
    rb._get_params = Getter([ParameterValue(type=ParameterType.PARAMETER_STRING, string_value="x")])
    rb._poll_allow_blocked()
    assert rb.allow_blocked()[0] is None                             # not a bool
    rb._pending.clear()
    rb._get_params = Getter(exc=RuntimeError("down"))
    rb._poll_allow_blocked()                                         # a failing poll keeps the last value and does not raise
    assert rb.allow_blocked()[0] is None
