"""Shared test fakes (no ROS): a fake clock, a fake Bridge that records every call, and a ready-to-cut environment."""
import json
import os
import sys
import threading
import time

from dual_cut_ui import bridge as B
from dual_cut_ui.core import Config, UiCore

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
MOCK_DETECT = os.path.join(SCRIPTS, "mock_detect.py")
RECORDED_SNAPSHOT = os.path.join(DATA, "snapshot_MOCK_ONLY.json")


def read_data(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read()


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def status_fields(**over):
    f = {"state": B.STATE_IDLE, "phase": "", "plan_seq": 0, "plan_retained": False, "phase_validated": False,
         "min_cross_distance_m": float("nan"), "closest_link_a": "", "closest_link_b": "",
         "joint_states_age_s": 0.05, "stop_reason": "", "stop_latched": False, "barrier_ok": True,
         "barrier_message": ""}
    f.update(over)
    return f


class FakeBridge(B.Bridge):
    def __init__(self, clock):
        self.clock = clock
        self.lock = threading.Lock()
        self.calls = []                     # (name, args..., monotonic_time)
        self._status = None
        self._listeners = []
        self._plan_report = None
        self._exec_reports = []
        self._health = None
        self._gates = None
        self.results = {"plan": (True, "top split left_max: ok"), "execute": (True, "dispatch accepted (barriers are checked next)"),
                        "stop": (True, "stopped"), "rearm": (True, "rearmed")}
        self.unavailable = set()            # names whose calls fail like a missing service: "stop", "plan", "gates", "cancel_left", ...
        self.block = {}                     # name -> threading.Event the call waits for
        self.raises = set()
        self.subscriptions = {}
        self.fail_false = False             # writes of gates=false fail (coordinator rejects / times out)
        self.fail_true = False
        self.cancel_delays = {}             # "left"/"right" -> list of seconds the next calls sleep (consumed in order)
        self.cancel_results = {}            # arm -> (ok, message)
        self.gate_param_writes = []         # (enabled, params written) per set_gates call that reached the coordinator
        self.schedule_modes = []            # modes written by set_schedule_mode
        self.schedule_ok = True
        self.detect_goals = []              # (action, goal) per detect call
        self.detect_feedback = [("capture", 1, "grabbing frames"), ("detect", 1, "seam found")]
        self.detect_result = {"success": True, "status": "accepted", "reason": "snapshot written", "reasons": [],
                              "cut_permission": "allowed", "snapshot_id": "snap1", "snapshot_path": "", "capture_id": "cap1",
                              "seam_status": "ok", "attempts": 1, "elapsed_s": 4.5}
        self.detect_error = None            # transport failure text (server missing, goal rejected, import error ...)
        self.detect_hook = None             # called when the goal is accepted (Env: writes the snapshot file)
        self.detect_cancelled = threading.Event()
        self._allow_blocked = (False, clock())      # polled read-back of allow_blocked_detection: (value | None, rx_time)
        self.allow_blocked_writes = []              # enabled values written by set_allow_blocked (reached the coordinator)
        self.allow_blocked_ok = True                # False: writes fail like a coordinator without the parameter

    # ---- helpers for tests
    def record(self, *call):
        with self.lock:
            self.calls.append(call + (time.monotonic(),))

    def names(self, name):
        with self.lock:
            return [c for c in self.calls if c[0] == name]

    def count(self, name):
        return len(self.names(name))

    def push_status(self, **over):
        fields = status_fields(**over)
        self._status = (fields, self.clock())
        for fn in list(self._listeners):
            fn(fields, self.clock())

    def set_health(self, ctrl_l=True, ctrl_r=True, hw_l=True, hw_r=True):
        self._health = ({"controllers": {"left": ctrl_l, "right": ctrl_r}, "hardware": {"left": hw_l, "right": hw_r}}, self.clock())

    def set_gate_values(self, on=True, max_age=240.0, **over):
        vals = {p: on for p in B.GATE_PARAMS}
        vals.update({"phase_order": "top_first", "schedule_mode": "sequence", "top_split_mode": "left_max",
                     "external_snapshot_max_age_s": max_age})
        vals.update(over)
        self._gates = (vals, self.clock())

    def set_ready(self, plan_seq=3):
        self.push_status(state=B.STATE_PLANNED, plan_seq=plan_seq, plan_retained=True, phase_validated=True)
        self.set_health()
        self.set_gate_values(True)

    # ---- Bridge interface
    def status(self):
        return self._status

    def add_status_listener(self, fn):
        self._listeners.append(fn)

    def plan_report(self):
        return self._plan_report

    def exec_reports(self):
        return list(self._exec_reports)

    def health(self):
        return self._health

    def gates(self):
        return self._gates

    def allow_blocked(self):
        return self._allow_blocked

    def set_allow_blocked(self, enabled, timeout_s):
        self.record("set_allow_blocked", enabled, timeout_s)
        self._wait("allow_blocked")
        self._wait("allow_blocked_true" if enabled else "allow_blocked_false")
        if "allow_blocked" in self.raises:
            raise RuntimeError("boom allow_blocked")
        if "allow_blocked" in self.unavailable or not self.allow_blocked_ok:
            return False, "allow_blocked_detection: service not available"
        self.allow_blocked_writes.append(enabled)
        self._allow_blocked = (enabled, self.clock())
        return True, f"allow_blocked_detection set {enabled}"

    def _wait(self, key):
        ev = self.block.get(key)
        if ev is not None:
            ev.wait(30)

    def trigger(self, name, timeout_s):
        self.record("trigger", name, timeout_s)
        self._wait(name)
        if name in self.raises:
            raise RuntimeError("boom " + name)
        if name in self.unavailable:
            return False, f"{name}: service not available"
        return self.results[name]

    def set_gates(self, enabled, timeout_s):
        self.record("set_gates", enabled, timeout_s)
        self._wait("gates")
        self._wait("gates_true" if enabled else "gates_false")
        if "gates" in self.raises:
            raise RuntimeError("boom gates")
        if "gates" in self.unavailable:
            return False, "execution gates: service not available"
        if (self.fail_false and not enabled) or (self.fail_true and enabled):
            return False, "execution gates: parameter write failed"
        params = B.gate_params_for(enabled)
        self.gate_param_writes.append((enabled, params))
        if self._gates is not None:
            vals = dict(self._gates[0])
            vals.update({p: enabled for p in params})
            self._gates = (vals, self.clock())
        return True, f"execution gates set {enabled}"

    def set_schedule_mode(self, mode, timeout_s):
        self.record("set_schedule_mode", mode, timeout_s)
        self._wait("schedule_mode")
        if "schedule_mode" in self.raises:
            raise RuntimeError("boom schedule_mode")
        if not self.schedule_ok:
            return False, "schedule_mode: service not available"
        self.schedule_modes.append(mode)
        if self._gates is not None:
            vals = dict(self._gates[0])
            vals["schedule_mode"] = mode
            self._gates = (vals, self.clock())
        return True, f"schedule_mode set {mode}"

    def detect(self, action, goal, on_feedback, cancel, timeout_s):
        self.record("detect", action, dict(goal), timeout_s)
        self.detect_goals.append((action, dict(goal)))
        if self.detect_error:
            return False, {"error": self.detect_error}
        for fb in self.detect_feedback:
            on_feedback(*fb)
        if self.detect_hook is not None:
            self.detect_hook()
        ev = self.block.get("detect")
        while ev is not None and not ev.is_set():          # a long-running goal: only a cancel (or the test) ends it
            if cancel.wait(0.01):
                self.detect_cancelled.set()
                self.record("detect_cancel")
                return False, {"cancelled": True}
        return bool(self.detect_result["success"]), dict(self.detect_result)

    def cancel_all(self, arm, timeout_s):
        self.record("cancel", arm, timeout_s)
        delays = self.cancel_delays.get(arm)
        if delays:
            time.sleep(delays.pop(0))
        self._wait("cancel_" + arm)
        if "cancel_" + arm in self.raises:
            raise RuntimeError("boom cancel")
        if "cancel_" + arm in self.unavailable:
            return False, f"cancel {arm}: service not available"
        return self.cancel_results.get(arm, (True, "no active goal"))

    def subscribe_camera(self, name, topic, callback):
        self.record("subscribe", name, topic)
        self.subscriptions[name] = callback

    def unsubscribe_camera(self, name):
        self.record("unsubscribe", name)
        self.subscriptions.pop(name, None)


def write_snapshot(path, age_s=0.0, mtime_age_s=None, mutate=None):
    """A recorded snapshot re-stamped to ``age_s`` seconds ago; the file mtime is set to ``mtime_age_s`` ago."""
    doc = json.loads(read_data("snapshot_MOCK_ONLY.json"))
    doc["acquisition_stamp_ns"] = int((time.time() - age_s) * 1e9)
    if mutate:
        mutate(doc)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    t = time.time() - (age_s if mtime_age_s is None else mtime_age_s)
    os.utime(path, (t, t))
    return doc


class Env:
    """A UiCore with a FakeBridge in the ready-to-cut state; page "op" is the operator."""

    def __init__(self, tmp_path, detect_args=(), **cfg_over):
        self.clock = FakeClock()
        self.wall_offset = 0.0
        self.snap = str(tmp_path / "seam_3d_base_link.json")
        self.out = str(tmp_path / "rgbd_captures")
        write_snapshot(self.snap)
        self.bridge = FakeBridge(self.clock)
        cmd = [sys.executable, MOCK_DETECT, "--source-snapshot", RECORDED_SNAPSHOT, "--output-dir", self.out,
               "--planner-output-json", self.snap, "--sleep", "0", *detect_args]
        kw = dict(snapshot_file=self.snap, detect_output_dir=self.out, detect_cwd=str(tmp_path), detect_command=cmd,
                  detect_mode="command",        # the subprocess tests; action-mode tests pass detect_mode="action"
                  stop_repeat_delays=(0.05, 0.15), startup_retry_s=0.01, kill_grace_s=1.0, execute_debounce_s=0.0)
        kw.update(cfg_over)
        self.cfg = Config(**kw)
        self.core = UiCore(self.bridge, self.cfg, clock=self.clock, wall=lambda: time.time() + self.wall_offset)
        self.bridge.set_ready()
        self.bridge.detect_hook = lambda: write_snapshot(self.snap)       # an action-mode detector writes a fresh snapshot
        self.core.touch("op")

    def act(self, action, args=None, page="op"):
        return self.core.dispatch(action, args, page)

    def set_plan_report(self, name="sequence_plan_report.json"):
        self.bridge._plan_report = (read_data(name), self.clock())

    def wait_job(self, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.core._job
            if job is not None and job.state != "running":
                return job
            time.sleep(0.02)
        raise AssertionError("job did not finish: " + str(self.core._job and self.core._job.to_dict()))

    def wait_job_started_process(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.core._job
            if job is not None and job.runner is not None and job.runner.pid is not None:
                return job
            time.sleep(0.02)
        raise AssertionError("detect process did not start")

    def wait_job_started_detect(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.bridge.count("detect") > 0 and self.core._job is not None and self.core._job.runner is not None:
                return self.core._job
            time.sleep(0.02)
        raise AssertionError("detect goal was not sent")

    def close(self):
        self.core.shutdown()
        self.core.join_background(3)


def old_placement_limits():
    """The pre-2026-11-07 placement checks only (single-arm left_reach on, two_arm_reach off), for the tests of those checks."""
    from dual_cut_ui.placement import PlacementLimits
    return PlacementLimits(left_reach_enabled=True, two_arm_reach_enabled=False)
