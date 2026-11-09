"""The testable heart of the dual-cut UI: operator lock, the action whitelist, job state machine, execute pre-checks,
software stop and the automatic closing of the execution gates.  No ROS and no HTTP; everything outside goes through
the ``Bridge`` interface (see bridge.py).

Safety model (the UI is NOT an emergency stop):
* The whitelist ``ACTIONS`` is the only way to make the cell do anything.  There is no service / topic / parameter
  passthrough.
* ``software_stop`` and ``take_control`` work from any page; every other action needs the operator role (403 otherwise).
* The three execution gates (execution_enabled, execute_phase_a_enabled, execute_phase_b_enabled) are set false at UI
  start, at shutdown, on a software stop and whenever a run ends (the coordinator leaves WAITING_BARRIER / EXECUTING).
  "Enable execution" only writes execution_enabled and execute_phase_a_enabled true (a sequence plan never uses phase B).
"""
import collections
import copy
import dataclasses
import logging
import math
import os
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

from . import bridge as B
from .detect_action import ActionRunner
from .jobs import ProcessRunner, exit_message
from .placement import PlacementLimits
from .report import clean, parse_exec_report, parse_plan_report
from .snapshot import SnapshotWatcher, find_overlay
from .topdown import render_topdown_svg

LOG = logging.getLogger("dual_cut_ui")

DEFAULT_SNAPSHOT_FILE = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"
DEFAULT_DETECT_CWD = "/home/farhan/Downloads/seam_handoff_20260908/scripts"
DEFAULT_DETECT_OUTPUT_DIR = "/home/farhan/Igus_rebel/outputs/rgbd_captures"
DEFAULT_DETECT_ACTION = "/box_seam_detector/detect_box"
DETECT_MODES = ("action", "command")
LOG_TAIL = 80
SHORT_DETAIL = 120


def shorten(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def default_detect_command(snapshot_file: str, output_dir: str) -> List[str]:
    return ["python3", "capture_and_query_depth.py", "--autolabel-v2", "--allow-needs-review", "--allow-cut-blocked",
            "--output-dir", output_dir, "--planner-output-json", snapshot_file, "--target-frame", "base_link"]


@dataclasses.dataclass
class Config:
    snapshot_file: str = DEFAULT_SNAPSHOT_FILE
    detect_command: Optional[List[str]] = None           # None / empty: default_detect_command(...)
    detect_cwd: str = DEFAULT_DETECT_CWD
    detect_output_dir: str = DEFAULT_DETECT_OUTPUT_DIR
    detect_domain_id: Optional[int] = None         # ROS_DOMAIN_ID for the detector subprocess (None: inherit)
    detect_localhost_only: Optional[int] = None    # ROS_LOCALHOST_ONLY (0/1) for the detector subprocess (None: inherit)
    detect_mode: str = "action"                    # "action": DetectBox goal (default) | "command": the subprocess above
    detect_action: str = DEFAULT_DETECT_ACTION
    detect_timeout_s: float = 300.0
    schedule_mode: str = "sequence"                # set on the coordinator before every plan call; "" = leave it alone
    coordinator: str = "/dual_cut_coordinator"     # only used for the job log text
    schedule_set_timeout_s: float = 5.0
    plan_timeout_s: float = 300.0
    control_grace_s: float = 10.0
    status_fresh_s: float = 2.0            # the coordinator status counts as live for this long
    health_fresh_s: float = 3.0            # controllers / hardware / gate read-backs
    default_max_age_s: float = 240.0       # snapshot limit when the coordinator parameter cannot be read
    stop_step_timeout_s: float = 1.0
    stop_repeat_delays: Tuple[float, ...] = (0.5, 1.5)
    gates_timeout_s: float = 2.0
    gates_retry_s: float = 1.0             # run-end gate closing keeps retrying at this interval
    plan_guard_s: float = 3.0              # after an accepted execute, a changed plan_seq within this window triggers a stop
    startup_retry_s: float = 2.0
    kill_grace_s: float = 3.0
    execute_debounce_s: float = 2.0
    allow_off_retries: int = 10            # auto-off writes of allow_blocked_detection retried this often (1 per gates_retry_s) while it reads back true
    placement_limits: PlacementLimits = dataclasses.field(default_factory=PlacementLimits)   # info-only placement pre-check thresholds

    def __post_init__(self):
        if self.detect_mode not in DETECT_MODES:
            raise ValueError(f"detect_mode must be one of {', '.join(DETECT_MODES)}, got {self.detect_mode!r}")
        if self.schedule_mode and self.schedule_mode not in B.SCHEDULE_MODES:
            raise ValueError(f"schedule_mode must be empty or one of {', '.join(B.SCHEDULE_MODES)}, got {self.schedule_mode!r}")

    def command(self) -> List[str]:
        return list(self.detect_command) if self.detect_command else default_detect_command(
            self.snapshot_file, self.detect_output_dir)


class Refusal(Exception):
    """A request the server will not perform; ``status`` is the HTTP code."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


class Job:
    def __init__(self, job_id: str, kind: str, steps: List[str], page_id: str):
        self.id, self.kind, self.page_id = job_id, kind, page_id
        self.state = "running"                  # running | done | failed | cancelled
        self.steps = [{"name": n, "state": "pending", "detail": ""} for n in steps]
        self.log: Deque[str] = collections.deque(maxlen=LOG_TAIL)
        self.message = ""
        self.started_wall = time.time()
        self.ended_wall: Optional[float] = None
        self.runner: Optional[object] = None            # ProcessRunner (command mode) or ActionRunner (action mode)
        self.cancel_requested = False
        self.thread: Optional[threading.Thread] = None
        self.plan_ok: Optional[bool] = None

    def step(self, name: str) -> dict:
        return next(s for s in self.steps if s["name"] == name)

    def set_step(self, name: str, state: str, detail: str = "") -> None:
        s = self.step(name)
        s["state"], s["detail"] = state, detail

    def note(self, line: str) -> None:
        self.log.append("[ui] " + line)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "steps": [dict(s) for s in self.steps],
                "message": self.message, "log": list(self.log), "started_at": self.started_wall,
                "ended_at": self.ended_wall, "plan_ok": self.plan_ok}


ACTIONS: Dict[str, Tuple[str, bool]] = {
    # name: (method, needs_operator).  The complete list of things a page can ask for.
    "take_control": ("_a_take_control", False),
    "release_control": ("_a_release_control", True),
    "software_stop": ("_a_software_stop", False),
    "detect": ("_a_detect", True),
    "plan": ("_a_plan", True),
    "detect_and_plan": ("_a_detect_and_plan", True),
    "set_execution_enabled": ("_a_set_execution_enabled", True),
    "set_allow_blocked_detection": ("_a_set_allow_blocked_detection", True),
    "execute": ("_a_execute", True),
    "rearm": ("_a_rearm", True),
    "cancel_job": ("_a_cancel_job", True),
}


def _bool_text(v) -> str:
    return "?" if v is None else ("on" if v else "off")


class UiCore:
    def __init__(self, bridge: B.Bridge, cfg: Optional[Config] = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.bridge, self.cfg = bridge, cfg or Config()
        self.clock, self.wall = clock, wall
        self.snapshot = SnapshotWatcher(self.cfg.snapshot_file, wall, self.cfg.placement_limits)
        self._ctl_lock = threading.Lock()
        self._holder: Optional[str] = None
        self._holder_seen = 0.0
        self._job_lock = threading.Lock()
        self._job: Optional[Job] = None
        self._job_n = 0
        self._exec_lock = threading.Lock()          # execute (check + dispatch) and the start of any Plan job exclude each other
        self._exec_dispatched_at = -1e9
        self._guard: Optional[dict] = None          # plan identity guard after an execute dispatch
        self._alert: Optional[dict] = None
        self._alert_n = 0
        self._enable_seq = 0                        # counts operator enable requests (a newer enable ends run-end gate retries)
        self._stop_epoch = 0
        self._closing = threading.Event()
        self._exec_seen = False
        self._exec_started = 0.0
        self._exec_phase: Optional[str] = None
        self._exec_phase_started = 0.0
        self._gates_lock = threading.Lock()
        self._gates_pending: Optional[Tuple[str, int]] = None
        self._gates_thread: Optional[threading.Thread] = None
        self.last_gates_off: Optional[dict] = None
        self._last_stop: Optional[dict] = None
        self._stop_lock = threading.Lock()
        self._allow_lock = threading.Lock()
        self._allow_off_pending: Optional[str] = None
        self._allow_off_thread: Optional[threading.Thread] = None
        self._threads: List[threading.Thread] = []
        self._parse_cache: Dict[str, dict] = {}
        self._overlay_cache: Tuple[object, Optional[dict], float] = (None, None, 0.0)
        bridge.add_status_listener(self.on_status)

    # ================================================================ lifecycle
    def start(self) -> None:
        """Close the gates at UI start (retried while the coordinator is not up yet)."""
        t = threading.Thread(target=self._startup_gates_off, name="ui-startup-gates", daemon=True)
        self._threads.append(t)
        t.start()

    def _startup_gates_off(self) -> None:
        """Keep trying until the coordinator answers (it may start after the UI) or the UI shuts down."""
        n = 0
        while not self._closing.is_set():
            ok, msg = self._set_gates(False, "UI start")
            if ok:
                self._allow_blocked_off("UI start")
                return
            n += 1
            if n % 15 == 1:
                LOG.error("could not close the execution gates yet (coordinator not reachable?): %s", msg)
            self._closing.wait(self.cfg.startup_retry_s)

    def shutdown(self, budget_s: float = 4.0) -> None:
        """Close the gates and stop a running detect job within ``budget_s`` seconds in total (ros2 launch escalates
        SIGTERM to SIGKILL, and a hung coordinator must not hold the UI's exit hostage).  The gates come first and run
        in parallel with the job termination.  Safe to call twice."""
        t0 = time.monotonic()
        self._closing.set()
        LOG.warning("UI shutdown: closing the execution gates and stopping any job (budget %.1f s)", budget_s)
        gates_t = threading.Thread(
            target=lambda: self._set_gates(False, "UI shutdown", timeout=min(self.cfg.gates_timeout_s, 0.4 * budget_s)),
            name="ui-shutdown-gates", daemon=True)
        gates_t.start()
        allow_t = threading.Thread(target=lambda: self._allow_blocked_off_now("UI shutdown", min(self.cfg.gates_timeout_s, 0.4 * budget_s)),
                                   name="ui-shutdown-allow-blocked", daemon=True)
        allow_t.start()
        job = self._job
        runner = job.runner if job is not None else None
        self._terminate_job(grace_s=1.0)
        if runner is not None and not runner.wait_gone(max(0.1, min(1.5, budget_s - (time.monotonic() - t0)))):
            LOG.warning("detector ignored SIGTERM: SIGKILL to its process group")
            runner.kill_now()
            runner.wait_gone(0.5)
        gates_t.join(max(0.0, budget_s - (time.monotonic() - t0)))
        allow_t.join(max(0.0, budget_s - (time.monotonic() - t0)))
        LOG.warning("UI shutdown finished in %.2f s (gates: %s)", time.monotonic() - t0,
                    "closed" if (self.last_gates_off or {}).get("reason") == "UI shutdown" and self.last_gates_off.get("ok")
                    else "NOT confirmed")

    def join_background(self, timeout: float = 5.0) -> None:
        """For tests: wait for the helper threads (gate closing, job, stop repeats)."""
        deadline = time.monotonic() + timeout
        for t in list(self._threads) + [self._gates_thread]:
            if t is not None and t.is_alive():
                t.join(max(0.0, deadline - time.monotonic()))

    # ================================================================ operator lock
    def touch(self, page_id: Optional[str], view: Optional[str] = None) -> None:
        """Called on every /api/state poll: refreshes the holder, or lets a free page claim control (not in present view)."""
        if not page_id:
            return
        now = self.clock()
        with self._ctl_lock:
            if page_id == self._holder:
                self._holder_seen = now
            elif view != "present" and not self._holder_active(now):
                LOG.info("control: page %s is the operator (free)", page_id[:8])
                self._holder, self._holder_seen = page_id, now

    def _holder_active(self, now: float) -> bool:
        return self._holder is not None and now - self._holder_seen <= self.cfg.control_grace_s

    def is_operator(self, page_id: Optional[str]) -> bool:
        with self._ctl_lock:
            return page_id is not None and page_id == self._holder

    def operator_info(self, page_id: Optional[str]) -> dict:
        now = self.clock()
        with self._ctl_lock:
            mine = page_id is not None and page_id == self._holder
            other_active = self._holder is not None and not mine and self._holder_active(now)
            return {"is_operator": mine, "other_operator_active": other_active, "control_grace_s": self.cfg.control_grace_s,
                    "holder_idle_s": None if self._holder is None else now - self._holder_seen}

    # ================================================================ dispatch
    def dispatch(self, action, args, page_id: Optional[str]) -> Tuple[int, dict]:
        """Run one whitelisted action.  Returns (http_status, reply); never raises."""
        if not isinstance(action, str) or action not in ACTIONS:
            return 400, {"ok": False, "message": f"unknown action {action!r}"}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return 400, {"ok": False, "message": "args must be a JSON object"}
        method, needs_operator = ACTIONS[action]
        if needs_operator and not self.is_operator(page_id):
            return 403, {"ok": False, "message": "read-only: you are not the operator (use Take control)"}
        try:
            reply = getattr(self, method)(args, page_id)
            return 200, clean({"ok": True, **reply} if "ok" not in reply else reply)
        except Refusal as r:
            return r.status, clean({"ok": False, "message": r.message, **r.extra})
        except Exception as e:  # noqa: BLE001 - a bug must not kill the request thread silently
            LOG.exception("action %s failed", action)
            return 500, {"ok": False, "message": f"internal error: {e}"}

    # ---------------------------------------------------------------- control actions
    def _a_take_control(self, args: dict, page_id: Optional[str]) -> dict:
        if not page_id:
            raise Refusal(400, "X-Page-Id header missing or malformed")
        force = args.get("force")
        if force is not None and not isinstance(force, bool):
            raise Refusal(400, "force must be true or false")
        now = self.clock()
        with self._ctl_lock:
            if self._holder == page_id:
                self._holder_seen = now
                return {"message": "you are already the operator"}
            if self._holder_active(now) and not force:
                raise Refusal(409, "another screen is the operator; take over from it?", needs_force=True)
            if self._holder is not None:
                LOG.warning("control: page %s takes over from %s%s", page_id[:8], self._holder[:8],
                            "" if self._holder_active(now) else " (it had gone quiet)")
            self._holder, self._holder_seen = page_id, now
        return {"message": "you are the operator"}

    def _a_release_control(self, args: dict, page_id: Optional[str]) -> dict:
        with self._ctl_lock:
            if self._holder == page_id:
                self._holder = None
        return {"message": "control released"}

    # ---------------------------------------------------------------- software stop
    def _a_software_stop(self, args: dict, page_id: Optional[str]) -> dict:
        return self.software_stop(page_id)

    def software_stop(self, page_id: Optional[str] = None, reason: str = "") -> dict:
        """Not an e-stop.  Concurrently: coordinator ~/stop, cancel-all on both trajectory actions (repeated at the
        configured delays, default +0.5 s and +1.5 s), gates false, kill a running detect job.  One failing step never
        keeps the others from running.  Returns after the first round; the repeats continue in the background."""
        t0 = time.monotonic()
        LOG.warning("SOFTWARE STOP requested by %s%s", ("page " + page_id[:8]) if page_id else "the UI server",
                    f" ({reason})" if reason else "")
        self._stop_epoch += 1
        cfg = self.cfg
        T = cfg.stop_step_timeout_s
        results: Dict[str, dict] = {}
        lock = self._stop_lock
        firsts = {n: threading.Event() for n in ("coordinator_stop", "cancel_left", "cancel_right", "gates_off", "stop_job")}
        record = {"at": self.wall(), "page": (page_id or "?")[:8], "reason": reason, "steps": results, "repeats": {}}
        with lock:
            self._last_stop = record

        def put(name, ok, msg, first=True):
            with lock:
                if first:
                    results[name] = {"ok": bool(ok), "message": str(msg)}
                else:
                    record["repeats"].setdefault(name, []).append({"ok": bool(ok), "message": str(msg)})
            if first:
                firsts[name].set()

        def guarded(name, fn, first=True):
            try:
                ok, msg = fn()
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"exception: {e}"
            put(name, ok, msg, first)
            if not ok:
                LOG.error("stop step %s: %s", name, msg)

        def cancel_once(arm, first):
            guarded(f"cancel_{arm}", lambda: self.bridge.cancel_all(arm, T), first=first)

        def repeat_at(arm, delay):
            """Each repeat has its own thread: a slow first call (up to the 1 s timeout) never delays or skips it."""
            wait = t0 + delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            cancel_once(arm, False)

        jobs = [("coordinator_stop", lambda: guarded("coordinator_stop", lambda: self.bridge.trigger("stop", T))),
                ("cancel_left", lambda: cancel_once("left", True)),
                ("cancel_right", lambda: cancel_once("right", True)),
                ("gates_off", lambda: guarded("gates_off", lambda: self._set_gates(False, "software stop", timeout=T))),
                ("stop_job", lambda: guarded("stop_job", lambda: (True, self._terminate_job())))]
        threads = []
        for name, fn in jobs:
            t = threading.Thread(target=fn, name=f"ui-stop-{name}", daemon=True)
            threads.append(t)
        for arm in B.ARMS:
            for delay in cfg.stop_repeat_delays:
                threads.append(threading.Thread(target=repeat_at, args=(arm, delay), name=f"ui-stop-repeat-{arm}-{delay}", daemon=True))
        for t in threads:
            t.start()
        self._threads.extend(threads)
        self._allow_blocked_off("software stop")          # not a stop step: the gates above already make execution impossible
        deadline = time.monotonic() + T + 1.5
        for name, ev in firsts.items():
            ev.wait(max(0.0, deadline - time.monotonic()))
        with lock:
            steps = {n: dict(results.get(n, {"ok": False, "message": "no result"})) for n in firsts}
        failed = [f"{n}: {s['message']}" for n, s in steps.items() if not s["ok"]]
        coord_ok = steps["coordinator_stop"]["ok"]
        if not failed:
            msg = "Software stop sent: coordinator held, trajectory goals cancelled, execution gates closed."
        else:
            msg = "Software stop sent, but some steps FAILED: " + "; ".join(failed)
            if not coord_ok:
                msg += ". The coordinator did not confirm: check the arms and use the hardware e-stop if needed."
        return {"ok": coord_ok and not failed, "message": msg, "steps": steps}

    @property
    def last_stop(self) -> Optional[dict]:
        with self._stop_lock:
            return copy.deepcopy(self._last_stop)

    def _terminate_job(self, grace_s: Optional[float] = None) -> str:
        job = self._job
        if job is None or job.state != "running":
            return "no job running"
        job.cancel_requested = True
        runner = job.runner
        if runner is not None:
            runner.terminate(grace_s)
            if isinstance(runner, ActionRunner):
                job.note("cancelling the detector goal")
            else:
                job.note("terminating the detector (SIGTERM, then SIGKILL after %.0f s)"
                         % (self.cfg.kill_grace_s if grace_s is None else grace_s))
        if self._detect_active(job):
            return "detect job cancelled" if isinstance(runner, ActionRunner) else "detect job terminated"
        return "plan job marked cancelled"

    @staticmethod
    def _detect_active(job: "Job") -> bool:
        return any(s["name"] == "Detect" and s["state"] in ("pending", "running") for s in job.steps)

    # ---------------------------------------------------------------- gates
    def _set_gates(self, enabled: bool, reason: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        t = self.cfg.gates_timeout_s if timeout is None else timeout
        try:
            ok, msg = self.bridge.set_gates(enabled, t)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"exception: {e}"
        if not enabled:
            self.last_gates_off = {"reason": reason, "ok": ok, "message": msg, "at": self.wall()}
            (LOG.info if ok else LOG.error)("execution gates -> false (%s): %s", reason, msg)
        else:
            LOG.warning("execution gates -> TRUE (%s): %s", reason, msg)
        return ok, msg

    def _request_gates_off(self, reason: str) -> None:
        """Non-blocking (called from the ROS callback thread): a helper thread closes the gates and keeps retrying (every
        ``gates_retry_s``) until a false write succeeds, the UI closes, or the operator enabled execution again AFTER this
        run ended (that newer enable is then the operator's intent, not something to undo)."""
        self._allow_blocked_off(reason)
        with self._gates_lock:
            self._gates_pending = (reason, self._enable_seq)
            if self._gates_thread is not None:
                return
            t = threading.Thread(target=self._gates_off_worker, name="ui-gates-off", daemon=True)
            self._gates_thread = t
            t.start()

    def _gates_off_worker(self) -> None:
        while True:
            with self._gates_lock:
                item, self._gates_pending = self._gates_pending, None
                if item is None:
                    self._gates_thread = None        # decided under the lock, so a concurrent request starts a new worker
                    return
            reason, seq0 = item
            while not self._closing.is_set():
                if self._enable_seq != seq0:
                    LOG.warning("run-end gate closing (%s) abandoned: the operator enabled execution again since", reason)
                    break
                ok, _msg = self._set_gates(False, reason)
                if ok:
                    break
                if self._closing.wait(self.cfg.gates_retry_s):
                    break

    def on_status(self, fields: dict, rx: float) -> None:
        """Status listener (ROS thread, must not block): remembers when a run / phase started; a run that ends closes the gates."""
        state = fields.get("state")
        self._check_plan_guard(fields)
        if state in (B.STATE_WAITING_BARRIER, B.STATE_EXECUTING):
            now = self.clock()
            if not self._exec_seen:
                self._exec_started, self._exec_phase = now, None
            phase = fields.get("phase") or ""
            if phase != self._exec_phase:
                self._exec_phase, self._exec_phase_started = phase, now
            self._exec_seen = True
        elif self._exec_seen:
            self._exec_seen = False
            self._request_gates_off(f"execute ended ({B.STATE_NAMES.get(state, state)})")

    def _check_plan_guard(self, fields: dict) -> None:
        """After an accepted execute: if the coordinator's plan_seq is not the one the operator confirmed (a plan was made
        meanwhile), stop at once.  Runs in the ROS callback: the stop itself is handed to a thread."""
        g = self._guard
        if g is None:
            return
        if g["until"] is not None and self.clock() > g["until"]:
            self._guard = None                         # the window is over: later plan changes are not this confirmation's business
            return
        seq_now = fields.get("plan_seq")
        if seq_now != g["seq"]:
            self._guard = None
            msg = (f"plan changed under the Cut confirmation: you confirmed plan {g['seq']}, the coordinator now reports "
                   f"plan {seq_now}; Software stop sent")
            LOG.error(msg)
            self._alert_n += 1
            self._alert = {"id": self._alert_n, "message": msg, "at": self.wall()}
            t = threading.Thread(target=self.software_stop, kwargs={"reason": "plan changed under the Cut confirmation"},
                                 name="ui-guard-stop", daemon=True)
            self._threads.append(t)
            t.start()
        elif fields.get("state") in (B.STATE_WAITING_BARRIER, B.STATE_EXECUTING):
            self._guard = None                         # the confirmed plan is the one that started

    # ---------------------------------------------------------------- gate switch
    def _a_set_execution_enabled(self, args: dict, page_id: Optional[str]) -> dict:
        enabled = args.get("enabled")
        if not isinstance(enabled, bool):
            raise Refusal(400, "enabled must be true or false")
        if not enabled:
            ok, msg = self._set_gates(False, "operator switch")
            self._allow_blocked_off("execution switch off")
            return {"ok": ok, "message": msg}
        st = self._status()
        if st is None:
            raise Refusal(409, "refused: the coordinator status is stale or missing (coordinator offline?)")
        if st[0].get("stop_latched"):
            raise Refusal(409, "refused: stop is latched; press Rearm first")
        if self._closing.is_set():
            raise Refusal(409, "refused: the UI is shutting down")
        epoch = self._stop_epoch
        self._enable_seq += 1
        LOG.warning("operator %s enables execution", (page_id or "?")[:8])
        ok, msg = self._set_gates(True, "operator switch")
        # Disable dominates: a Software stop, a shutdown or a takeover that happened while the write was in flight must
        # not leave the gates true.  No lock is held across the write, so stop / shutdown never wait for an enable.
        why = None
        if self._stop_epoch != epoch:
            why = "a Software stop happened while enabling"
        elif self._closing.is_set():
            why = "the UI is shutting down"
        elif not self.is_operator(page_id):
            why = "you are no longer the operator"
        elif not ok:
            why = "the enable was not confirmed"
        if why:
            LOG.error("enable raced (%s): writing the gates false again", why)
            ok2, msg2 = self._set_gates(False, f"enable raced: {why}")
            raise Refusal(409, f"Execution NOT enabled: {why}; the gates were closed again ({'ok' if ok2 else 'FAILED: ' + msg2})"
                          + ("" if ok else f" [enable reply: {msg}]"))
        return {"ok": ok, "message": msg}

    # ---------------------------------------------------------------- blocked-detection switch
    def _allow_blocked_value(self) -> Tuple[Optional[bool], Optional[float]]:
        """(value, age_s) of the coordinator's allow_blocked_detection read-back; (None, age) when the coordinator does not
        report it, (None, None) when never polled.  A read-back older than ``health_fresh_s`` counts as unknown."""
        got = self.bridge.allow_blocked()
        if got is None:
            return None, None
        age = self.clock() - got[1]
        return (got[0] if age <= self.cfg.health_fresh_s else None), age

    def _detection_blocked(self, snap: Optional[dict] = None) -> Tuple[bool, List[str]]:
        """(blocked, reason codes): the current snapshot says cut_permission.status 'blocked', or the retained plan report says
        detection_blocked (the coordinator refuses to execute such a plan unless the switch is on)."""
        snap = snap if snap is not None else self.snapshot.summary(self._max_age())
        cp = snap.get("cut_permission") or {}
        blocked = cp.get("status") == "blocked"
        codes = [r.get("code") for r in cp.get("reasons") or [] if r.get("code")]
        pr = self.bridge.plan_report()
        if pr is not None and self._parsed("plan", pr[0]).get("detection_blocked") is True:
            blocked = True
        return blocked, codes

    def _a_set_allow_blocked_detection(self, args: dict, page_id: Optional[str]) -> dict:
        """Switch for the coordinator parameter allow_blocked_detection.  Turning it ON is refused unless the current detection
        is blocked (nothing to allow otherwise), the coordinator is live, no stop is latched and the UI is not closing; it goes
        OFF again by itself (Software stop, run end, execution switch off, the next Detect, UI start / stop).  Disable dominates
        exactly like the execution switch.  It never touches the execution gates."""
        enabled = args.get("enabled")
        if not isinstance(enabled, bool):
            raise Refusal(400, "enabled must be true or false")
        if not enabled:
            ok, msg = self._write_allow_blocked(False, "operator switch")
            return {"ok": ok, "message": msg}
        st = self._status()
        if st is None:
            raise Refusal(409, "refused: the coordinator status is stale or missing (coordinator offline?)")
        if st[0].get("stop_latched"):
            raise Refusal(409, "refused: stop is latched; press Rearm first")
        if self._closing.is_set():
            raise Refusal(409, "refused: the UI is shutting down")
        blocked, codes = self._detection_blocked()
        if not blocked:
            raise Refusal(409, "refused: the current detection is not blocked (cut_permission is not 'blocked'): nothing to allow")
        epoch = self._stop_epoch
        LOG.warning("operator %s allows cutting a BLOCKED detection (%s)", (page_id or "?")[:8], ", ".join(codes) or "no reason codes")
        ok, msg = self._write_allow_blocked(True, "operator switch")
        why = None
        if self._stop_epoch != epoch:
            why = "a Software stop happened while switching it on"
        elif self._closing.is_set():
            why = "the UI is shutting down"
        elif not self.is_operator(page_id):
            why = "you are no longer the operator"
        elif not ok:
            why = "the write was not confirmed"
        if why:
            LOG.error("allow_blocked_detection raced (%s): writing it false again", why)
            ok2, msg2 = self._write_allow_blocked(False, f"enable raced: {why}")
            raise Refusal(409, f"Blocked detection NOT allowed: {why}; it was switched off again ({'ok' if ok2 else 'FAILED: ' + msg2})"
                          + ("" if ok else f" [reply: {msg}]"))
        return {"ok": True, "message": msg}

    def _write_allow_blocked(self, enabled: bool, reason: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        t = self.cfg.gates_timeout_s if timeout is None else timeout
        try:
            ok, msg = self.bridge.set_allow_blocked(enabled, t)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"exception: {e}"
        (LOG.warning if enabled else LOG.info)("%s -> %s (%s): %s %s", B.ALLOW_BLOCKED_PARAM, "TRUE" if enabled else "false", reason, ok, msg)
        return ok, msg

    def _allow_blocked_off_now(self, reason: str, timeout: Optional[float] = None) -> None:
        """Synchronous best-effort OFF write.  Retried (every gates_retry_s) only while the read-back says true, so a
        coordinator without the parameter costs one quick failed call, not a retry loop."""
        for i in range(max(1, self.cfg.allow_off_retries)):
            got = self.bridge.allow_blocked()
            if got is not None and got[0] is False and i == 0:
                return                                     # known false: nothing to switch off
            ok, _msg = self._write_allow_blocked(False, reason, timeout)
            if ok or self._closing.is_set():
                return
            got = self.bridge.allow_blocked()
            if got is None or got[0] is not True:
                return                                     # parameter missing / unknown: do not loop
            if self._closing.wait(self.cfg.gates_retry_s):
                return

    def _allow_blocked_off(self, reason: str) -> None:
        """Non-blocking (also called from the ROS callback thread): a helper thread switches the permission off."""
        with self._allow_lock:
            self._allow_off_pending = reason
            if self._allow_off_thread is not None:
                return
            t = threading.Thread(target=self._allow_off_worker, name="ui-allow-blocked-off", daemon=True)
            self._allow_off_thread = t
            t.start()
        self._threads.append(t)

    def _allow_off_worker(self) -> None:
        while True:
            with self._allow_lock:
                reason, self._allow_off_pending = self._allow_off_pending, None
                if reason is None:
                    self._allow_off_thread = None
                    return
            self._allow_blocked_off_now(reason)

    # ---------------------------------------------------------------- rearm
    def _a_rearm(self, args: dict, page_id: Optional[str]) -> dict:
        if self._job_running():
            raise Refusal(409, "busy: a detect / plan job is running")
        epoch = self._stop_epoch
        ok, msg = self.bridge.trigger("rearm", 10.0)
        LOG.warning("rearm by %s: %s %s", (page_id or "?")[:8], ok, msg)
        if self._stop_epoch != epoch:
            # a Software stop happened while the rearm call was in flight: the stop dominates, whatever order they landed in
            ok2, msg2 = self.bridge.trigger("stop", self.cfg.stop_step_timeout_s)
            LOG.error("rearm raced with a Software stop: stop re-sent (%s %s)", ok2, msg2)
            return {"ok": False, "message": f"Rearm raced with a Software stop: the stop was sent again "
                                            f"({'ok' if ok2 else 'FAILED: ' + str(msg2)}); rearm reply was: {msg}"}
        return {"ok": ok, "message": msg}

    # ---------------------------------------------------------------- execute
    def _status(self) -> Optional[Tuple[dict, float]]:
        """(fields, age_s) of a live coordinator status, else None."""
        got = self.bridge.status()
        if got is None:
            return None
        age = self.clock() - got[1]
        return (got[0], age) if age <= self.cfg.status_fresh_s else None

    def _max_age(self) -> float:
        got = self.bridge.gates()
        v = None if got is None else got[0].get("external_snapshot_max_age_s")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 \
            else self.cfg.default_max_age_s

    def _health(self) -> Tuple[Optional[dict], Optional[float]]:
        got = self.bridge.health()
        return (None, None) if got is None else (got[0], self.clock() - got[1])

    def _gate_values(self) -> Tuple[Optional[dict], Optional[float]]:
        got = self.bridge.gates()
        return (None, None) if got is None else (got[0], self.clock() - got[1])

    def cut_blockers(self, page_id: Optional[str], plan_seq: Optional[int] = None) -> List[str]:
        """Every reason why Cut/execute is not allowed right now (empty list: allowed).  The page greys out its Cut
        button from this; ``execute`` re-checks it server-side."""
        b: List[str] = []
        if not self.is_operator(page_id):
            b.append("read-only: you are not the operator")
        if self._job_running():
            b.append("a detect / plan job is running")
        st = self._status()
        if st is None:
            b.append("coordinator status is stale or missing (coordinator offline?)")
        else:
            f = st[0]
            if f.get("stop_latched"):
                b.append(f"stop is latched ({f.get('stop_reason') or 'no reason given'}): press Rearm and plan again")
            if f.get("state") != B.STATE_PLANNED:
                b.append(f"coordinator state is {B.STATE_NAMES.get(f.get('state'), f.get('state'))}, it must be PLANNED")
            if not f.get("plan_retained"):
                b.append("no plan is retained")
            if not f.get("phase_validated"):
                b.append("the retained plan is not validated")
            if plan_seq is not None and f.get("plan_seq") != plan_seq:
                b.append(f"the plan changed: you confirmed plan {plan_seq}, the coordinator now holds plan {f.get('plan_seq')}; "
                         "look at the result again")
        snap = self.snapshot.summary(self._max_age())
        if not snap["available"]:
            b.append(f"snapshot unavailable: {snap['error']}")
        elif not snap["fresh"]:
            if snap["age_s"] is None:
                b.append("snapshot has no usable acquisition time: " + "; ".join(snap["problems"]))
            else:
                b.append(f"snapshot is stale ({snap['age_s']:.0f} s old, limit {snap['max_age_s']:.0f} s): detect again")
        health, h_age = self._health()
        if health is None or h_age > self.cfg.health_fresh_s:
            b.append("controller / hardware state is unknown (no fresh poll)")
        else:
            off = [f"{kind} {arm}" for kind in ("controllers", "hardware") for arm in B.ARMS
                   if not health.get(kind, {}).get(arm)]
            if off:
                b.append("not active: " + ", ".join(off))
        blocked, codes = self._detection_blocked(snap)
        if blocked:
            allow, _age = self._allow_blocked_value()
            if allow is not True:
                b.append("the detection is BLOCKED (cut_permission" + (": " + ", ".join(codes) if codes else "") +
                         "): switch on 'Allow blocked detection' to cut it" +
                         ("" if allow is False else " (the coordinator does not report allow_blocked_detection)"))
        gates, g_age = self._gate_values()
        if gates is None or g_age > self.cfg.health_fresh_s:
            b.append("execution gates unknown (no fresh read-back from the coordinator)")
        else:
            off = [p for p in B.ENABLE_PARAMS if gates.get(p) is not True]
            if off:
                b.append("execution is not enabled on the coordinator (" + ", ".join(off) + " not true)")
            if gates.get("execute_phase_b_enabled") is not True and self._retained_schedule_mode() != "sequence":
                b.append("the retained plan is a 'phases' plan: with phase B off only phase A would run; plan again "
                         "(Plan sets schedule_mode sequence)")
        return b

    def _retained_schedule_mode(self) -> Optional[str]:
        """schedule_mode of the plan report the coordinator published last (None: no report / unparsable / unknown)."""
        pr = self.bridge.plan_report()
        if pr is None:
            return None
        plan = self._parsed("plan", pr[0])
        if (plan.get("sequence") or {}).get("fallback_to_phases"):
            return "phases"                                    # a sequence request that fell back is executed as phases
        return plan.get("schedule_mode")

    def _a_execute(self, args: dict, page_id: Optional[str]) -> dict:
        seq = args.get("plan_seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise Refusal(400, "plan_seq (the number shown in the confirm dialog) is required")
        if not self._exec_lock.acquire(blocking=False):
            raise Refusal(409, "an execute request is already in progress")
        try:
            blockers = self.cut_blockers(page_id, seq)
            if blockers:
                LOG.warning("execute refused: %s", "; ".join(blockers))
                raise Refusal(409, "Cut refused: " + "; ".join(blockers), blockers=blockers)
            if self.clock() - self._exec_dispatched_at < self.cfg.execute_debounce_s:
                raise Refusal(409, "Cut refused: an execute was dispatched a moment ago")
            LOG.warning("EXECUTE plan_seq=%d requested by %s", seq, (page_id or "?")[:8])
            self._guard = {"seq": seq, "until": None}      # armed BEFORE the call: a status may overtake the reply
            ok, msg = self.bridge.trigger("execute", 5.0)
            LOG.warning("execute -> %s: %s", ok, msg)
            if ok:
                self._exec_dispatched_at = self.clock()
                if self._guard is not None:
                    self._guard["until"] = self.clock() + self.cfg.plan_guard_s
            else:
                self._guard = None
            return {"ok": ok, "message": msg}
        finally:
            self._exec_lock.release()

    # ---------------------------------------------------------------- jobs
    def _job_running(self) -> bool:
        j = self._job
        return j is not None and j.state == "running"

    def _a_detect(self, args, page_id):
        return self._start_job("detect", ["Detect"], page_id)

    def _a_plan(self, args, page_id):
        return self._start_job("plan", ["Plan"], page_id)

    def _a_detect_and_plan(self, args, page_id):
        return self._start_job("detect_and_plan", ["Detect", "Plan"], page_id)

    def _plan_precheck(self) -> None:
        st = self._status()
        if st is None:
            raise Refusal(409, "refused: the coordinator status is stale or missing (coordinator offline?)")
        state = st[0].get("state")
        if state in (B.STATE_EXECUTING, B.STATE_WAITING_BARRIER):
            raise Refusal(409, "refused: the cell is executing; wait for it to finish or use Software stop")
        if state == B.STATE_PLANNING:
            raise Refusal(409, "refused: the coordinator is already planning")

    def _start_job(self, kind: str, steps: List[str], page_id: Optional[str]) -> dict:
        takes_exec_lock = "Plan" in steps
        if takes_exec_lock and not self._exec_lock.acquire(blocking=False):
            raise Refusal(409, "refused: an execute request is in flight; a plan cannot start now")
        try:
            return self._start_job_locked(kind, steps, page_id)
        finally:
            if takes_exec_lock:
                self._exec_lock.release()

    def _start_job_locked(self, kind: str, steps: List[str], page_id: Optional[str]) -> dict:
        with self._job_lock:
            if self._job_running():
                raise Refusal(409, f"busy: a {self._job.kind} job is already running")
            if "Plan" in steps:
                self._plan_precheck()
            self._job_n += 1
            job = Job(f"job{self._job_n}", kind, steps, page_id or "?")
            self._job = job
            job.note(f"{kind} started")
            job.thread = threading.Thread(target=self._run_job, args=(job,), name=f"ui-job-{job.id}", daemon=True)
            self._threads.append(job.thread)
            job.thread.start()
        return {"message": f"{kind} started", "job": job.id}

    def _a_cancel_job(self, args, page_id):
        job = self._job
        if job is None or job.state != "running":
            raise Refusal(409, "no job is running")
        if self._detect_active(job):
            self._terminate_job()
            return {"message": "cancelling the detector"}
        raise Refusal(409, "the plan call cannot be cancelled from here; use Software stop to abort planning")

    def _run_job(self, job: Job) -> None:
        epoch = self._stop_epoch
        try:
            ok = True
            if "Detect" in [s["name"] for s in job.steps]:
                self._allow_blocked_off("new detection")      # a permission belongs to one detection
                ok = self._step_detect(job)
                if not ok:
                    if job.plan_ok is None and any(s["name"] == "Plan" for s in job.steps):
                        job.set_step("Plan", "skipped", "detect did not succeed")
                    self._finish(job, False)
                    return
            if any(s["name"] == "Plan" for s in job.steps):
                if job.cancel_requested or self._stop_epoch != epoch:
                    job.set_step("Plan", "skipped", "stopped")
                    job.message = "stopped before planning"
                    self._finish(job, False)
                    return
                ok = self._step_plan(job)
            self._finish(job, ok)
        except Exception as e:  # noqa: BLE001
            LOG.exception("job %s crashed", job.id)
            job.message = f"internal error: {e}"
            self._finish(job, False)

    def _finish(self, job: Job, ok: bool) -> None:
        job.ended_wall = time.time()
        job.state = "cancelled" if (job.cancel_requested and not ok) else ("done" if ok else "failed")
        job.note(f"job {job.state}: {shorten(job.message, SHORT_DETAIL)}")
        LOG.info("job %s (%s) %s: %s", job.id, job.kind, job.state, job.message)

    def _step_detect(self, job: Job) -> bool:
        if self.cfg.detect_mode == "action":
            return self._step_detect_action(job)
        cfg = self.cfg
        cmd = cfg.command()
        job.set_step("Detect", "running")
        overrides = {} if cfg.detect_domain_id is None else {"ROS_DOMAIN_ID": str(cfg.detect_domain_id)}
        if cfg.detect_localhost_only is not None:
            overrides["ROS_LOCALHOST_ONLY"] = str(cfg.detect_localhost_only)
        domain = overrides.get("ROS_DOMAIN_ID") or os.environ.get("ROS_DOMAIN_ID") or "unset"
        localhost = overrides.get("ROS_LOCALHOST_ONLY") or os.environ.get("ROS_LOCALHOST_ONLY") or "unset"
        job.note("running: " + " ".join(cmd) + f"  (cwd {cfg.detect_cwd}, ROS_DOMAIN_ID {domain}, ROS_LOCALHOST_ONLY {localhost})")
        runner = ProcessRunner(cmd, cwd=cfg.detect_cwd, tail_lines=LOG_TAIL, kill_grace_s=cfg.kill_grace_s, env_overrides=overrides)
        job.runner = runner
        if job.cancel_requested:
            runner.terminate()
        started = self.wall()
        rc = runner.run(on_line=job.log.append)
        job.runner = None
        if job.cancel_requested or runner.terminated:
            job.set_step("Detect", "failed", "cancelled")
            job.message = "detection cancelled"
            return False
        if rc != 0:
            job.message = exit_message(rc)
            job.set_step("Detect", "failed", shorten(job.message, SHORT_DETAIL))
            return False
        return self._detect_finish(job, started, "detector exited 0")

    def _detect_finish(self, job: Job, started: float, who: str) -> bool:
        """The detector says it worked: the snapshot file must really be usable and newly written."""
        cfg = self.cfg
        snap = self.snapshot.summary(self._max_age())
        try:
            mtime = os.path.getmtime(cfg.snapshot_file)
        except OSError:
            mtime = 0.0
        if not snap["available"]:
            job.message = f"{who} but the snapshot is unusable: {snap['error']}"
            job.set_step("Detect", "failed", shorten(job.message, SHORT_DETAIL))
            return False
        if mtime < started - 2.0:
            job.message = f"{who} but did not write a new snapshot file"
            job.set_step("Detect", "failed", shorten(job.message, SHORT_DETAIL))
            return False
        job.message = exit_message(0)
        placement = snap.get("placement")
        detail = job.message
        if placement:
            detail = f"seam yaw {placement['yaw_deg']:.1f} deg, x {placement['centre_x_m']:.3f} y {placement['centre_y_m']:.3f}"
            if any(w.get("level") == "warn" for w in placement.get("warnings") or []):
                detail += ", PLACEMENT WARNING"
            if self._detection_blocked(snap)[0]:
                detail += ", BLOCKED"
        job.set_step("Detect", "ok", detail)
        return True

    def _step_detect_action(self, job: Job) -> bool:
        """Detect through the DetectBox action, with the goal of the operator's ``ros2 action send_goal`` command."""
        cfg = self.cfg
        job.set_step("Detect", "running")
        goal = dict(B.DETECT_GOAL)
        job.note(f"send_goal {cfg.detect_action} DetectBox {{dry_run: {'true' if goal['dry_run'] else 'false'}}}")
        runner = ActionRunner(self.bridge, cfg.detect_action, goal, cfg.detect_timeout_s)
        job.runner = runner
        if job.cancel_requested:
            runner.terminate()
        started = self.wall()

        def feedback(stage: str, attempt: int, message: str) -> None:
            job.log.append(f"[{stage}] attempt {attempt}: {message}")
        ok, info = runner.run(feedback)
        job.runner = None
        if job.cancel_requested or runner.terminated or info.get("cancelled"):
            job.log.append("detect goal cancelled" + (f": {info['error']}" if info.get("error") else ""))
            job.set_step("Detect", "failed", "cancelled")
            job.message = "detection cancelled"
            return False
        if info.get("error") and "status" not in info:
            job.message = str(info["error"])
            job.log.append("detect failed: " + job.message)
            job.set_step("Detect", "failed", shorten(job.message, SHORT_DETAIL))
            return False
        self._log_detect_result(job, info)
        written = info.get("snapshot_path")
        if written and os.path.realpath(str(written)) != os.path.realpath(cfg.snapshot_file):
            job.log.append(f"WARNING: the detector wrote {written} but the UI reads {cfg.snapshot_file}")
        if not ok:
            status = info.get("status") or "failed"
            reason = info.get("reason") or "; ".join(str(r) for r in (info.get("reasons") or [])) or "no reason given"
            if info.get("error"):
                reason += f" ({info['error']})"
            job.message = f"detector {status}: {reason}"
            job.set_step("Detect", "failed", shorten(job.message, SHORT_DETAIL))
            return False
        return self._detect_finish(job, started, "the detector reported success")

    @staticmethod
    def _log_detect_result(job: Job, info: dict) -> None:
        reasons = info.get("reasons") or []
        job.log.append(f"result: success={'true' if info.get('success') else 'false'} status={info.get('status')} "
                       f"reason={info.get('reason') or ''}")
        job.log.append("reasons: " + (", ".join(str(r) for r in reasons) if reasons else "-"))
        job.log.append(f"cut_permission: {info.get('cut_permission') or '-'}  seam_status: {info.get('seam_status') or '-'}")
        job.log.append(f"snapshot_path: {info.get('snapshot_path') or '-'}")
        job.log.append(f"attempts: {info.get('attempts')}  elapsed: {info.get('elapsed_s')} s")

    def _step_plan(self, job: Job) -> bool:
        job.set_step("Plan", "running")
        try:
            self._plan_precheck()
        except Refusal as r:
            job.message = r.message
            job.set_step("Plan", "failed", shorten(r.message, SHORT_DETAIL))
            job.plan_ok = False
            return False
        mode = self.cfg.schedule_mode
        if mode:
            job.note(f"param set {self.cfg.coordinator} schedule_mode {mode}")
            try:
                ok, msg = self.bridge.set_schedule_mode(mode, self.cfg.schedule_set_timeout_s)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"exception: {e}"
            if not ok:
                job.plan_ok = False
                job.message = f"could not set schedule_mode {mode} before planning: {msg}"
                job.set_step("Plan", "failed", shorten(job.message, SHORT_DETAIL))
                return False
        job.note(f"calling coordinator plan (waits up to {self.cfg.plan_timeout_s:.0f} s)")
        ok, msg = self.bridge.trigger("plan", self.cfg.plan_timeout_s)
        job.plan_ok = bool(ok)
        job.message = msg
        job.set_step("Plan", "ok" if ok else "failed", "plan OK" if ok else shorten(msg, SHORT_DETAIL))   # full text: job.message
        return bool(ok)

    # ================================================================ state for the page
    def _parsed(self, kind: str, text: str) -> dict:
        key = kind + "\0" + text
        got = self._parse_cache.get(key)
        if got is None:
            got = (parse_plan_report if kind == "plan" else parse_exec_report)(text)
            if len(self._parse_cache) > 64:
                self._parse_cache.clear()
            self._parse_cache[key] = got
        return got

    def detection_info(self) -> Optional[dict]:
        """The overlay of the current snapshot (cached by snapshot signature; a miss is retried every 2 s)."""
        summ = self.snapshot.summary(self._max_age())
        sig = tuple(summ.get("signature") or ()) or None
        key, val, at = self._overlay_cache
        now = self.clock()
        if key == sig and (val is not None or now - at < 2.0):
            return val
        val = find_overlay(self.cfg.detect_output_dir, self.cfg.snapshot_file) if sig else None
        self._overlay_cache = (sig, val, now)
        return val

    def _execution_view(self, coord: dict) -> dict:
        now = self.clock()
        active = bool(coord.get("online")) and coord.get("state") in (B.STATE_WAITING_BARRIER, B.STATE_EXECUTING) and self._exec_seen
        return {"active": active, "phase": coord.get("phase") or None if active else None,
                "run_elapsed_s": now - self._exec_started if active else None,
                "phase_elapsed_s": now - self._exec_phase_started if active else None}

    def state(self, page_id: Optional[str] = None, view: Optional[str] = None) -> dict:
        cfg, now = self.cfg, self.clock()
        self.touch(page_id, view)
        got = self.bridge.status()
        coord = {"online": False, "age_s": None, "state": None, "state_name": "COORDINATOR OFFLINE"}
        if got is not None:
            age = now - got[1]
            f = dict(got[0])
            online = age <= cfg.status_fresh_s
            coord = {**f, "online": online, "age_s": age,
                     "state_name": B.STATE_NAMES.get(f.get("state"), "?") if online else "COORDINATOR OFFLINE"}
        health, h_age = self._health()
        gate_vals, g_age = self._gate_values()
        max_age = self._max_age()
        snap = self.snapshot.summary(max_age)
        pr = self.bridge.plan_report()
        plan = None
        if pr is not None:
            plan = dict(self._parsed("plan", pr[0]))
            plan["report_age_s"] = now - pr[1]
        coverage = plan.get("top_coverage") if plan else None
        svg = render_topdown_svg(snap.get("points") or {}, coverage) if snap.get("available") else None
        execs = [dict(self._parsed("exec", t), age_s=now - rx) for t, rx in self.bridge.exec_reports()[-2:]]
        det = self.detection_info()
        job = self._job
        blockers = self.cut_blockers(page_id)
        gates_view = None
        if gate_vals is not None:
            gates_view = {**{p: gate_vals.get(p) for p in B.GATE_PARAMS + B.DISPLAY_PARAMS}, "age_s": g_age,
                          "fresh": g_age <= cfg.health_fresh_s,
                          "all_enabled": all(gate_vals.get(p) is True for p in B.ENABLE_PARAMS)}
        allow_val, allow_age = self._allow_blocked_value()
        det_blocked, det_codes = self._detection_blocked(snap)
        allow_view = {"value": allow_val, "known": allow_val is not None, "age_s": allow_age,
                      "detection_blocked": det_blocked, "reason_codes": det_codes,
                      "needed": det_blocked and allow_val is not True}
        return clean({
            "ok": True, "server_time": self.wall(),
            "operator": self.operator_info(page_id),
            "coordinator": coord,
            "health": None if health is None else {**health, "age_s": h_age, "fresh": h_age <= cfg.health_fresh_s,
                                                     "all_active": all(health.get(k, {}).get(a) for k in ("controllers", "hardware") for a in B.ARMS)},
            "gates": gates_view, "allow_blocked": allow_view,
            "snapshot": snap,
            "detection": {"available": det is not None, "mtime": None if det is None else det["mtime"]},
            "plan": plan, "svg": svg, "exec_reports": execs,
            "job": None if job is None else job.to_dict(),
            "cut": {"can_cut": not blockers, "blockers": blockers},
            "last_stop": self.last_stop, "last_gates_off": self.last_gates_off,
            "execution": self._execution_view(coord), "alert": self._alert,
            "config": {"health_fresh_s": cfg.health_fresh_s, "plan_timeout_s": cfg.plan_timeout_s, "control_grace_s": cfg.control_grace_s,
                       "snapshot_file": cfg.snapshot_file, "max_age_s": max_age},
        })
