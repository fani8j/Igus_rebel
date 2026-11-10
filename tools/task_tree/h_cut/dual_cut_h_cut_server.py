#!/usr/bin/env python3
"""HCut action server (compal_cut_task_msgs/action/HCut on /h_cut) that wraps the dual-arm cutting flow.

For one goal it does: preflight -> detect the box -> plan -> (only if allowed) execute -> wait until finished.

  detect   compal_box_msgs/action/DetectBox on `detect_action` (default /box_seam_detector/detect_box), sent like the operator UI
           does (default goal, nothing overridden). Not accepted / rejected / blocked / timeout -> PERCEPTION_FAILED.
  plan     Trigger `<coordinator>/plan` (default coordinator /dual_cut_coordinator). success=false -> PLANNING_FAILED.
  execute  Trigger `<coordinator>/execute`, then follow `<coordinator>/status` and `<coordinator>/exec_report`.
           "Done" means: the state went back to PLANNED after the phases had run, the stop is not latched, and every phase
           that was seen on the status has an exec_report with completed=true. Anything else is EXECUTION_FAILED.
  stop     Trigger `<coordinator>/stop` on cancel, on timeout, on node shutdown and when the status stream goes stale
           (no message for `status_timeout_s`) while an execution is running. Best effort, the reply is logged.

Run (the coordinator and the detector must already run; nothing here starts them):

    source /opt/ros/humble/setup.bash
    source /home/farhan/igus_rebel_ros2/install/setup.bash            # dual_cut_msgs, compal_box_msgs
    source /home/farhan/igus_rebel_ros2/install_bt/local_setup.bash   # compal_cut_task_msgs
    python3 dual_cut_h_cut_server.py [--ros-args -p skip_detect:=true -p detect_timeout_s:=300.0]

Safety rules (execute-path code, fail closed):
  * The robot may move only if goal.execute and goal.i_understand_this_moves_the_real_robot are BOTH true. If they differ the
    goal ends with BAD_GOAL and nothing is done. Both false = plan only (did_move=false).
  * This server NEVER calls `<coordinator>/rearm` (it has no client for it) and NEVER sets a parameter on any node. The
    execution gates (execution_enabled, execute_phase_*_enabled, ...) stay operator controlled; if they are off the coordinator
    refuses ~/execute and the refusal is reported as the failure message with did_move=false.
  * A STOPPED / latched coordinator fails the goal ("coordinator is STOPPED (<reason>); the operator must rearm").
  * One goal at a time; a second goal is rejected.
  * attempt > 1 / recut_hint_json: the planner cannot re-cut partially, the hint is ignored and the whole flow runs again.
  * A cancel during planning does not stop the coordinator's plan call (nothing moves while planning); the coordinator
    finishes it by itself. A cancel after ~/execute was requested always calls ~/stop.
"""
import json
import re
import signal
import threading
import time

import rclpy
from compal_box_msgs.action import DetectBox
from compal_cut_task_msgs.action import HCut
from dual_cut_msgs.msg import CoordinatorStatus
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger

STATE_IDLE = CoordinatorStatus.STATE_IDLE
STATE_PLANNING = CoordinatorStatus.STATE_PLANNING
STATE_PLANNED = CoordinatorStatus.STATE_PLANNED
STATE_WAITING_BARRIER = CoordinatorStatus.STATE_WAITING_BARRIER
STATE_EXECUTING = CoordinatorStatus.STATE_EXECUTING
STATE_STOPPED = CoordinatorStatus.STATE_STOPPED
STATE_NAMES = {0: "IDLE", 1: "PLANNING", 2: "PLANNED", 3: "WAITING_BARRIER", 4: "EXECUTING", 5: "STOPPED"}
POLL_S = 0.05


def oneline(text, limit=300):
    """One line, whitespace collapsed, the START kept (the cause is usually first)."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= limit else s[: max(0, limit - 3)] + "..."


class _Interrupted(Exception):
    """cancel / timeout / shutdown while a stage was waiting."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class _Finish(Exception):
    """An expected end of the goal: error code + one-line message."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class _Run:
    """State of the one running goal."""

    def __init__(self, goal_handle, timeout_s):
        self.gh = goal_handle
        self.t0 = time.monotonic()
        self.deadline = self.t0 + timeout_s if timeout_s > 0 else None
        self.execute_requested = False   # ~/execute was (possibly) sent: a cancel / timeout must call ~/stop
        self.did_move = False
        self.stop_called = 0
        self.stage = ""
        self.last_fb = 0.0


class DualCutHCutServer(Node):
    def __init__(self, node_name="dual_cut_h_cut_server", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self.declare_parameter("h_cut_action", "/h_cut")
        self.declare_parameter("coordinator", "/dual_cut_coordinator")
        self.declare_parameter("detect_action", "/box_seam_detector/detect_box")
        self.declare_parameter("skip_detect", False)
        self.declare_parameter("detect_timeout_s", 300.0)
        self.declare_parameter("detect_server_wait_s", 10.0)
        self.declare_parameter("detect_cancel_wait_s", 5.0)
        self.declare_parameter("plan_timeout_s", 300.0)
        self.declare_parameter("execute_call_timeout_s", 15.0)
        self.declare_parameter("stop_call_timeout_s", 5.0)
        self.declare_parameter("status_timeout_s", 3.0)
        self.declare_parameter("preflight_wait_s", 5.0)
        self.declare_parameter("exec_report_grace_s", 3.0)
        self.declare_parameter("default_timeout_s", 1800.0)   # whole goal, used when goal.timeout_sec <= 0
        self._p = lambda n: self.get_parameter(n).value

        coord = "/" + self._p("coordinator").strip("/")
        self._group = ReentrantCallbackGroup()
        # NOTE: no ~/rearm client and no parameter client exist on purpose (see the module docstring).
        self._plan = self.create_client(Trigger, f"{coord}/plan", callback_group=self._group)
        self._execute = self.create_client(Trigger, f"{coord}/execute", callback_group=self._group)
        self._stop = self.create_client(Trigger, f"{coord}/stop", callback_group=self._group)
        self._detect = ActionClient(self, DetectBox, self._p("detect_action"), callback_group=self._group)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched20 = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CoordinatorStatus, f"{coord}/status", self._on_status, reliable, callback_group=self._group)
        self.create_subscription(String, f"{coord}/exec_report", self._on_exec_report, latched20, callback_group=self._group)

        self._lock = threading.Lock()
        self._status = None          # (CoordinatorStatus, rx monotonic)
        self._reports = []           # parsed exec_report dicts since the last clear
        self._busy = False
        self._shutdown = threading.Event()
        self._server = ActionServer(self, HCut, self._p("h_cut_action"), execute_callback=self._on_execute,
                                    goal_callback=self._on_goal, cancel_callback=lambda _h: CancelResponse.ACCEPT,
                                    callback_group=self._group)
        self.get_logger().info(f"HCut server on {self._p('h_cut_action')} wrapping {coord} "
                               f"(detect {self._p('detect_action')}{', SKIPPED' if self._p('skip_detect') else ''}); "
                               "never rearms, never sets parameters")

    # ------------------------------------------------------------------ subscriptions
    def _on_status(self, msg):
        with self._lock:
            self._status = (msg, time.monotonic())

    def _on_exec_report(self, msg):
        try:
            doc = json.loads(msg.data)
            if not isinstance(doc, dict):
                raise ValueError("not an object")
        except Exception as e:  # noqa: BLE001 - an unreadable report never counts as "completed"
            self.get_logger().warn(f"unreadable exec_report ignored: {e}")
            doc = {"phase": "?", "completed": False, "unreadable": True}
        with self._lock:
            self._reports.append(doc)

    def _status_now(self):
        with self._lock:
            return self._status

    def _fresh_status(self):
        st = self._status_now()
        if st is None or time.monotonic() - st[1] > self._p("status_timeout_s"):
            return None
        return st[0]

    # ------------------------------------------------------------------ goal / cancel plumbing
    def _on_goal(self, _goal):
        with self._lock:
            if self._busy:
                self.get_logger().warn("goal rejected: an H-cut goal is already running")
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def request_shutdown(self):
        self._shutdown.set()

    def busy(self):
        with self._lock:
            return self._busy

    def _check(self, run):
        """Raise _Interrupted if the goal was cancelled / timed out / the node is shutting down."""
        if self._shutdown.is_set():
            raise _Interrupted("shutdown")
        if run.gh.is_cancel_requested:
            raise _Interrupted("cancel")
        if run.deadline is not None and time.monotonic() > run.deadline:
            raise _Interrupted("timeout")

    def _wait(self, run, event, timeout_s, interruptible=True):
        """Wait for a threading.Event without spinning; returns True if it was set."""
        end = time.monotonic() + timeout_s
        while True:
            if event.wait(POLL_S):
                return True
            if interruptible:
                self._check(run)
            if time.monotonic() >= end:
                return False

    def _call(self, run, client, timeout_s, interruptible=True):
        """Trigger call from inside the execute callback: returns the response or None on timeout."""
        done = threading.Event()
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda _f: done.set())
        try:
            ok = self._wait(run, done, timeout_s, interruptible)
        except _Interrupted:
            client.remove_pending_request(fut)
            raise
        if not ok:
            client.remove_pending_request(fut)
            return None
        try:
            return fut.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"service call failed: {e}")
            return None

    def _feedback(self, run, stage, progress, message="", force=True):
        now = time.monotonic()
        if not force and stage == run.stage and now - run.last_fb < 1.0:   # a new stage is always reported
            return
        run.last_fb = now
        run.stage = stage
        try:
            run.gh.publish_feedback(HCut.Feedback(stage=stage, progress=float(progress), message=oneline(message, 200)))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"feedback failed: {e}")

    def _stop_best_effort(self, run, why):
        """~/stop, not interruptible, never raises. Called at most once per goal."""
        if run.stop_called:
            return
        run.stop_called += 1
        self.get_logger().warn(f"calling {self._stop.srv_name}: {why}")
        try:
            res = self._call(run, self._stop, self._p("stop_call_timeout_s"), interruptible=False)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"stop call raised: {e}")
            return
        if res is None:
            self.get_logger().error("stop: no reply (timeout / service failure); the coordinator may still be executing")
        else:
            self.get_logger().warn(f"stop reply: success={res.success} message={oneline(res.message)}")

    # ------------------------------------------------------------------ the goal
    def _on_execute(self, goal_handle):
        req = goal_handle.request
        timeout = float(req.timeout_sec)
        run = _Run(goal_handle, timeout if timeout > 0 else self._p("default_timeout_s"))
        result = HCut.Result(success=False, error_code=HCut.Result.INTERNAL_ERROR, message="", did_move=False, run_folder="")
        interrupted = None
        try:
            try:
                msg = self._flow(run, req)
                result.success, result.error_code, result.message = True, HCut.Result.OK, msg
            except _Finish as f:
                result.error_code, result.message = f.code, f.message
            except _Interrupted as i:
                interrupted = i.kind
                if run.execute_requested:
                    self._stop_best_effort(run, f"{i.kind} after ~/execute was requested")
                code = HCut.Result.TIMEOUT if i.kind == "timeout" else HCut.Result.CANCELLED
                result.error_code = code
                result.message = {"timeout": "timeout", "cancel": "cancelled", "shutdown": "cancelled: server shutting down"}[i.kind]
                result.message += (" (stop sent, robot may have moved)" if run.execute_requested and run.did_move else
                                   " (stop sent)" if run.stop_called else "")
            except Exception as e:  # noqa: BLE001 - internal error: fail closed, stop if motion may be running
                self.get_logger().error(f"internal error: {e!r}")
                if run.execute_requested:
                    self._stop_best_effort(run, f"internal error: {e!r}")
                result.error_code, result.message = HCut.Result.INTERNAL_ERROR, oneline(f"internal error: {e!r}")
            result.did_move = bool(run.did_move)
            result.message = oneline(f"{result.message} ({time.monotonic() - run.t0:.1f} s)", 400)
            self.get_logger().info(f"goal finished: success={result.success} code={result.error_code} did_move={result.did_move}: "
                                   f"{result.message}")
            if interrupted == "shutdown" and not goal_handle.is_cancel_requested:
                goal_handle.abort()
            elif goal_handle.is_cancel_requested:
                # succeed() is not allowed once a cancel was accepted; a cancel that arrives after the work finished is
                # still reported as cancelled (did_move tells the truth).
                if not interrupted:
                    result.success, result.error_code = False, HCut.Result.CANCELLED
                    result.message = oneline("cancelled (after the flow had finished): " + result.message, 400)
                goal_handle.canceled()
            elif result.error_code == HCut.Result.INTERNAL_ERROR:
                goal_handle.abort()
            else:
                goal_handle.succeed()   # expected failures: succeed() with success=false
        finally:
            with self._lock:
                self._busy = False
        return result

    def _flow(self, run, req):
        log = self.get_logger()
        if req.execute != req.i_understand_this_moves_the_real_robot:
            raise _Finish(HCut.Result.BAD_GOAL, "execute and i_understand_this_moves_the_real_robot must be equal "
                                                "(both true to move, both false for plan only)")
        allowed = bool(req.execute and req.i_understand_this_moves_the_real_robot)
        if req.attempt > 1 or req.recut_hint_json:
            log.warn(f"attempt {req.attempt}: recut_hint_json ({len(req.recut_hint_json)} chars) is ignored, "
                     "the planner cannot re-cut partially; running the full flow again")
        log.info(f"goal: attempt={req.attempt} {'EXECUTE requested' if allowed else 'plan only'} timeout_sec={req.timeout_sec}")

        self._feedback(run, "preflight", 0.02, "checking the coordinator")
        self._preflight(run)

        if self._p("skip_detect"):
            log.warn("skip_detect=true: planning on the snapshot the executors already have")
        else:
            self._feedback(run, "detect", 0.05, "detecting the box")
            self._detect_box(run)

        self._feedback(run, "plan", 0.25, "planning")
        plan_msg = self._plan_step(run)

        if not allowed:
            return f"plan only: {oneline(plan_msg, 200)}"

        self._feedback(run, "execute", 0.4, "requesting execution")
        return self._execute_step(run, plan_msg)

    # ------------------------------------------------------------------ preflight
    def _preflight(self, run):
        end = time.monotonic() + self._p("preflight_wait_s")
        while True:
            self._check(run)
            missing = [c.srv_name for c in (self._plan, self._execute, self._stop) if not c.service_is_ready()]
            st = self._fresh_status()
            if not missing and st is not None:
                break
            if time.monotonic() >= end:
                if missing:
                    raise _Finish(HCut.Result.INTERNAL_ERROR, "coordinator services not available: " + ", ".join(missing))
                raise _Finish(HCut.Result.INTERNAL_ERROR, "no fresh coordinator status received "
                                                          f"(within {self._p('status_timeout_s'):.1f} s): is the coordinator running?")
            time.sleep(POLL_S)
        self._require_usable(st)

    def _require_usable(self, st):
        """State must be IDLE or PLANNED and not latched; raises _Finish otherwise."""
        if st.stop_latched or st.state == STATE_STOPPED:
            raise _Finish(HCut.Result.EXECUTION_FAILED,
                          f"coordinator is STOPPED ({oneline(st.stop_reason, 150) or 'no reason given'}); the operator must rearm")
        if st.state not in (STATE_IDLE, STATE_PLANNED):
            raise _Finish(HCut.Result.BUSY, f"coordinator is busy (state {STATE_NAMES.get(st.state, st.state)})")

    # ------------------------------------------------------------------ detect
    def _detect_box(self, run):
        log = self.get_logger()
        end = time.monotonic() + self._p("detect_server_wait_s")
        while not self._detect.server_is_ready():
            self._check(run)
            if time.monotonic() >= end:
                raise _Finish(HCut.Result.PERCEPTION_FAILED, "detector action server not available "
                                                             f"({self._p('detect_action')}; is box_seam_detector running?)")
            time.sleep(POLL_S)
        goal = DetectBox.Goal()   # the UI's default goal: max_attempts 0, no overrides, dry_run false
        goal.max_attempts, goal.allow_needs_review, goal.allow_cut_blocked, goal.dry_run = 0, False, False, False

        def fb(msg):
            try:
                f = msg.feedback
                self._feedback(run, "detect", 0.1, f"{f.stage}: {f.message}", force=False)
            except Exception:  # noqa: BLE001
                pass

        accepted = threading.Event()
        fut = self._detect.send_goal_async(goal, feedback_callback=fb)
        fut.add_done_callback(lambda _f: accepted.set())
        try:
            if not self._wait(run, accepted, 15.0):
                fut.add_done_callback(self._cancel_late)
                raise _Finish(HCut.Result.PERCEPTION_FAILED, "detector did not answer the goal (no accept / reject in time)")
        except _Interrupted:
            fut.add_done_callback(self._cancel_late)
            raise
        handle = fut.result()
        if handle is None or not handle.accepted:
            raise _Finish(HCut.Result.PERCEPTION_FAILED, "detector rejected the goal (busy?)")
        done = threading.Event()
        rfut = handle.get_result_async()
        rfut.add_done_callback(lambda _f: done.set())
        t_end = time.monotonic() + self._p("detect_timeout_s")
        try:
            while not done.wait(POLL_S):
                self._check(run)
                if time.monotonic() > t_end:
                    self._cancel_detect(handle, done)
                    raise _Finish(HCut.Result.PERCEPTION_FAILED,
                                  f"detector timeout: no result within {self._p('detect_timeout_s'):.0f} s (goal cancelled)")
        except _Interrupted:
            self._cancel_detect(handle, done)
            raise
        res = rfut.result().result
        if not res.success:
            why = res.reason or ", ".join(res.reasons) or "no reason given"
            raise _Finish(HCut.Result.PERCEPTION_FAILED, oneline(f"detector {res.status or 'failed'}: {why}"))
        log.info(f"detect ok: status={res.status} capture={res.capture_id} snapshot={res.snapshot_path} "
                 f"cut_permission={res.cut_permission}")

    def _cancel_detect(self, handle, done):
        try:
            handle.cancel_goal_async()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"detect cancel failed: {e}")
            return
        if not done.wait(self._p("detect_cancel_wait_s")):
            self.get_logger().warn("the detector did not confirm the cancel in time")

    @staticmethod
    def _cancel_late(fut):
        try:
            h = fut.result()
            if h is not None and h.accepted:
                h.cancel_goal_async()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ plan
    def _plan_step(self, run):
        res = self._call(run, self._plan, self._p("plan_timeout_s"))
        if res is None:
            raise _Finish(HCut.Result.PLANNING_FAILED, f"plan: no reply within {self._p('plan_timeout_s'):.0f} s")
        if not res.success:
            raise _Finish(HCut.Result.PLANNING_FAILED, oneline("plan refused: " + res.message))
        self.get_logger().info(f"plan ok: {oneline(res.message, 400)}")
        return res.message

    # ------------------------------------------------------------------ execute
    def _execute_step(self, run, plan_msg):
        # The plan must be the retained, validated one and the coordinator must still be usable.
        end = time.monotonic() + self._p("preflight_wait_s")
        while True:
            self._check(run)
            st = self._fresh_status()
            if st is not None:
                self._require_usable(st)
                if st.state == STATE_PLANNED and st.plan_retained and st.phase_validated:
                    break
            if time.monotonic() >= end:
                raise _Finish(HCut.Result.EXECUTION_FAILED, "after the plan the coordinator does not report PLANNED with a "
                                                            "retained, validated plan; not executing")
            time.sleep(POLL_S)

        with self._lock:
            self._reports.clear()
        t_req_ns = self.get_clock().now().nanoseconds
        # From here on a cancel / timeout / shutdown calls ~/stop (the request may be accepted at any moment).
        run.execute_requested = True
        res = self._call(run, self._execute, self._p("execute_call_timeout_s"), interruptible=False)
        if res is None:
            run.did_move = True   # unknown: report conservatively
            self._stop_best_effort(run, "no reply to ~/execute")
            raise _Finish(HCut.Result.EXECUTION_FAILED, "execute: no reply from the coordinator; stop sent, the robot may have moved")
        if not res.success:
            run.execute_requested = False   # refused: nothing started, no stop needed
            raise _Finish(HCut.Result.EXECUTION_FAILED, oneline("execute refused: " + res.message))
        run.did_move = True
        self.get_logger().warn(f"execute accepted: {oneline(res.message)}")
        self._check(run)   # a cancel that arrived during the call is handled by the caller (stop + CANCELLED)
        phases = self._follow_execution(run, t_req_ns)
        return f"H-cut executed, phases {'+'.join(phases)} completed"

    def _follow_execution(self, run, t_req_ns):
        """Follow ~/status until the execution finished; returns the completed phase labels or raises."""
        seen_active = False
        phases_seen = []
        t_wait = None            # first time the state was PLANNED after the phases (grace for the exec reports)
        status_timeout = self._p("status_timeout_s")
        last_rx = None
        while True:
            self._check(run)
            snap = self._status_now()
            if snap is None or time.monotonic() - snap[1] > status_timeout:
                self._stop_best_effort(run, "status stream went stale during the execution")
                raise _Finish(HCut.Result.EXECUTION_FAILED, f"coordinator status stale for more than {status_timeout:.1f} s "
                                                            "during the execution; stop sent")
            st, rx = snap
            if st.stop_latched or st.state == STATE_STOPPED:
                raise _Finish(HCut.Result.EXECUTION_FAILED,
                              oneline(f"execution stopped by the coordinator: {st.stop_reason or 'no reason given'}", 300))
            if st.state in (STATE_WAITING_BARRIER, STATE_EXECUTING):
                seen_active = True
                t_wait = None
                if st.phase and st.phase not in phases_seen:
                    phases_seen.append(st.phase)
                if rx != last_rx:
                    last_rx = rx
                    prog = 0.5 + 0.1 * min(len(phases_seen), 4)
                    name = "execute phase " + st.phase if st.phase else "execute"
                    self._feedback(run, name + (" (barrier)" if st.state == STATE_WAITING_BARRIER else ""), prog,
                                   st.barrier_message if st.state == STATE_WAITING_BARRIER else "", force=False)
            elif st.state == STATE_PLANNING:
                raise _Finish(HCut.Result.EXECUTION_FAILED, "coordinator went to PLANNING during the execution")
            else:   # PLANNED or IDLE
                stamp_ns = st.header.stamp.sec * 1_000_000_000 + st.header.stamp.nanosec
                if seen_active or stamp_ns > t_req_ns:
                    if st.state == STATE_IDLE:
                        raise _Finish(HCut.Result.EXECUTION_FAILED, "coordinator went to IDLE during the execution "
                                                                    "(plan cleared / rearmed by someone else)")
                    # PLANNED again: either all phases ran, or a barrier / goal build refused (no stop is latched then).
                    verdict = self._judge_reports(phases_seen)
                    if verdict == "ok":
                        return phases_seen
                    if isinstance(verdict, str) and verdict != "wait":
                        raise _Finish(HCut.Result.EXECUTION_FAILED, oneline(verdict + self._barrier_hint(st)))
                    t_wait = t_wait or time.monotonic()
                    if time.monotonic() - t_wait > self._p("exec_report_grace_s"):
                        raise _Finish(HCut.Result.EXECUTION_FAILED, oneline(
                            "coordinator is PLANNED again but the exec_report of phase "
                            f"{'/'.join(phases_seen) or '?'} did not arrive: not treating the execution as done"
                            + self._barrier_hint(st)))
            time.sleep(POLL_S)

    @staticmethod
    def _barrier_hint(st):
        return f" (barrier: {st.barrier_message})" if st.barrier_message and st.barrier_message != "ok" else ""

    def _judge_reports(self, phases_seen):
        """'ok' if every seen phase has a completed=true report and none is completed=false;
        'wait' if reports are still missing; otherwise a failure text."""
        with self._lock:
            reports = list(self._reports)
        if not phases_seen:
            return "wait"
        for rep in reports:
            if rep.get("completed") is not True or rep.get("stop_latched") is True:
                return f"phase {rep.get('phase', '?')} did not complete: {rep.get('stop_reason') or 'no stop reason'}"
        have = {str(r.get("phase")) for r in reports}
        if all(p in have for p in phases_seen):
            return "ok"
        return "wait"


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # we handle SIGINT/SIGTERM so ~/stop can still be called
    node = DualCutHCutServer()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)
        if node.busy():
            node.get_logger().warn("shutdown while a goal runs: stopping it (~/stop if an execute was requested)")
            node.request_shutdown()
            t_end = time.monotonic() + 15.0
            while node.busy() and time.monotonic() < t_end:   # keep spinning so the service calls can finish
                executor.spin_once(timeout_sec=0.1)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
