"""Suggested next calibration pose: ROS glue and state for the web UI (the maths is in ``pose_suggest.py``).

    SuggestManager   pure state machine (no ROS imports): compute in a worker thread, selection, send to RViz, plan / move
                     pass-through to the jog controller, stale marking.  All collaborators are injected, so tests use fakes.
    MoveItBridge     the ROS side: MoveIt ``/compute_ik`` and ``/check_state_validity`` service calls (blocking, worker
                     thread only) and ONE publisher, the RViz goal-state topic below.

Safety: the only thing this module can publish is a ``moveit_msgs/RobotState`` on RViz's "update custom goal state" topic.
That topic only moves the interactive goal marker in the MotionPlanning panel.  Nothing here publishes a plan, an
execute request or a trajectory, and nothing here talks to a controller.  The robot only moves through the jog
controller's ``execute_move`` (same enable / busy / STOP rules as a jog step).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

GOAL_STATE_TOPIC = "/rviz/moveit/update_custom_goal_state"
N_SUGGESTIONS = 5
IK_TIMEOUT_S = 0.1
SERVICE_TIMEOUT_S = 3.0
RVIZ_HINT = ("goal shown in RViz (MotionPlanning > Context: tick 'Allow External Comm.' if nothing changes); "
             "then press Plan in RViz, check the preview, and Execute in RViz (or use Plan preview / Move there here)")


class SuggestRefused(Exception):
    """A request that is refused with a clear message (HTTP ``status``)."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.message, self.status = message, status


def clean(v):
    """JSON-safe copy: numpy -> python, non-finite floats -> None, tuples -> lists."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if v is None or isinstance(v, str):
        return v
    return str(v)


def _default_engine():
    from .. import pose_suggest
    return pose_suggest


def _default_load_samples(session_dir: str):
    from ..session import load_samples
    return load_samples(session_dir)


class SampleCounter:
    """Number of recorded arm samples in a session, cached by file (mtime, size): safe to call on every state poll."""

    def __init__(self):
        self._key: Optional[Tuple[str, int, int]] = None
        self._counts: Dict[str, int] = {}

    def count(self, session_dir: Optional[str], arm: str) -> Optional[int]:
        if not session_dir:
            return None
        path = os.path.join(session_dir, "samples.jsonl")
        try:
            st = os.stat(path)
        except OSError:
            return 0 if os.path.isdir(session_dir) else None
        key = (path, st.st_mtime_ns, st.st_size)
        if key != self._key:
            counts: Dict[str, int] = {}
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            a = str(json.loads(line).get("arm"))
                        except (ValueError, AttributeError):
                            continue
                        counts[a] = counts.get(a, 0) + 1
            except OSError:
                return None
            self._key, self._counts = key, counts
        return self._counts.get(arm, 0)


class SuggestManager:
    """State and actions of the "Suggested pose" panel.

    mode          recorder mode; suggestions are offered in single_left / single_right only
    cfgs          jog.arm_configs(mode): {arm: ArmConfig(group, joint_prefix, joint_names, ...)}
    jog           JogController-like: status(), joint_window(arms, s) or joint_snapshot(arm), plan_move, execute_move,
                  clear_plan
    session_dir_fn  () -> session directory or None
    moveit        MoveItBridge-like: ready() -> (bool, msg), ik(cfg, T, seed_q) -> dict|None, valid(cfg, q) -> (bool, str),
                  publish_goal(names, positions); None = MoveIt not available
    engine        module-like with fit_arm / suggest (default: pose_suggest)
    load_samples  session_dir -> samples (default: session.load_samples)
    """

    def __init__(self, mode: str, cfgs: Dict[str, Any], jog, session_dir_fn: Callable[[], Optional[str]],
                 moveit=None, engine=None, load_samples: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic, n: int = N_SUGGESTIONS):
        self.mode, self.cfgs, self.jog = mode, cfgs, jog
        self.session_dir_fn, self.moveit = session_dir_fn, moveit
        self._engine, self._load = engine, load_samples or _default_load_samples
        self.clock, self.n = clock, n
        self.log = logging.getLogger("calibration_ui.suggest")
        self.lock = threading.Lock()
        self.counter = SampleCounter()
        self._gen = 0
        self._state, self._message = "idle", ""
        self._arm: Optional[str] = None
        self._items: List[dict] = []
        self._public: List[dict] = []
        self._stats: Dict[str, int] = {}
        self._based_on: Optional[int] = None
        self._computed_at: Optional[float] = None
        self._selected: Optional[int] = None
        self._plan: Optional[dict] = None            # {plan_id, index, arm} of the plan stored in the jog

    # ---- helpers ---------------------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return self.mode in ("single_left", "single_right")

    def _the_arm(self, arm: Optional[str]) -> str:
        if not self.supported:
            raise SuggestRefused("suggested poses: single-arm modes only", 400)
        only = next(iter(self.cfgs))
        if arm not in (None, only):
            raise SuggestRefused(f"arm must be '{only}' in mode {self.mode}", 400)
        return only

    def _jog_blocked(self) -> str:
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            return f"jog state unknown ({e})"
        if st.get("available") is False:
            return "jog unavailable: " + str(st.get("error", ""))
        if st.get("busy") or st.get("stopping") or st.get("fault"):
            return "the arm is busy (a jog step / move is running, stopping or in fault)"
        return ""

    def _current_q(self, arm: str) -> Optional[Dict[str, float]]:
        cfg = self.cfgs[arm]
        jw = getattr(self.jog, "joint_window", None)
        if jw is not None:
            r = jw([arm], 0.5)
            if r is not None:
                q = r[0]
                if all(n in q for n in cfg.joint_names):
                    return {n: float(q[n]) for n in cfg.joint_names}
            return None
        snap = getattr(self.jog, "joint_snapshot", None) or getattr(getattr(self.jog, "core", None),
                                                                    "joint_snapshot", None)
        if snap is None:
            return None
        v = snap(arm)
        return None if v is None else {n: float(x) for n, x in zip(cfg.joint_names, v)}

    def _clear_jog_plan(self) -> None:
        self._plan = None
        fn = getattr(self.jog, "clear_plan", None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                self.log.exception("jog.clear_plan failed")

    def _get_engine(self):
        if self._engine is None:
            self._engine = _default_engine()
        return self._engine

    # ---- compute ---------------------------------------------------------------------------------
    def compute(self, arm: Optional[str] = None) -> str:
        """Validate, fit the arm's samples (a ValueError text is returned as a refusal) and start the search in a worker
        thread.  Blocking for the fit only; call from a worker pool, not from the Tornado loop."""
        arm = self._the_arm(arm)
        cfg = self.cfgs[arm]
        with self.lock:
            if self._state == "computing":
                raise SuggestRefused("already computing a suggestion", 409)
        why = self._jog_blocked()
        if why:
            raise SuggestRefused("cannot suggest now: " + why, 409)
        if self.moveit is None:
            raise SuggestRefused("MoveIt services are not available in this UI (no ROS bridge)", 503)
        ok, msg = self.moveit.ready()
        if not ok:
            raise SuggestRefused(msg, 503)
        session_dir = self.session_dir_fn()
        if not session_dir:
            raise SuggestRefused("session directory unknown (recorder status not available yet)", 503)
        q0 = self._current_q(arm)
        if q0 is None:
            raise SuggestRefused("current joint states of the arm are not available", 503)
        engine = self._get_engine()
        try:
            samples = [s for s in self._load(session_dir) if getattr(s, "arm", None) == arm]
        except Exception as e:  # noqa: BLE001
            raise SuggestRefused(f"cannot read the session samples: {e}", 500)
        try:
            fit = engine.fit_arm(samples, arm)
        except ValueError as e:
            with self.lock:
                self._set_error(str(e))
            raise SuggestRefused(str(e), 400)
        with self.lock:
            if self._state == "computing":
                raise SuggestRefused("already computing a suggestion", 409)
            self._gen += 1
            gen = self._gen
            self._state, self._message, self._arm = "computing", "searching for a pose ...", arm
            self._items, self._public, self._stats, self._selected = [], [], {}, None
            self._based_on, self._computed_at = len(samples), None
            self._clear_jog_plan()
        threading.Thread(target=self._worker, args=(gen, arm, cfg, engine, fit, q0), name="suggest-pose",
                         daemon=True).start()
        return f"computing a suggestion from {len(samples)} samples"

    def _set_error(self, message: str) -> None:
        self._gen += 1
        self._state, self._message = "error", message
        self._items, self._public, self._stats, self._selected = [], [], {}, None
        self._based_on, self._computed_at = None, None
        self._clear_jog_plan()

    def _worker(self, gen: int, arm: str, cfg, engine, fit, q0: Dict[str, float]) -> None:
        try:
            ik_errors: Dict[str, Any] = {"n": 0, "first": None}

            def ik(T, seed_q=None):
                # pose_suggest.suggest() calls ik(T, seed_q=current_q) BY KEYWORD: the parameter must be called seed_q.
                # The bridge call stays positional.  An exception is counted by the engine as "ik_error" (it cannot
                # say why), so the first one is logged and kept for the message.
                try:
                    return self.moveit.ik(cfg, T, {} if seed_q is None else seed_q)
                except Exception as e:  # noqa: BLE001
                    ik_errors["n"] += 1
                    if ik_errors["first"] is None:
                        ik_errors["first"] = f"{type(e).__name__}: {e}"
                        self.log.error("IK call failed (first error of this search): %s", ik_errors["first"])
                    raise

            valid = lambda q: self.moveit.valid(cfg, q)                # noqa: E731
            sugg, stats = engine.suggest(fit, ik, valid, q0, n=self.n)
            items, public, bad = [], [], 0
            for s in sugg:
                try:
                    q = {n: float(s.q[n]) for n in cfg.joint_names}
                    if not all(math.isfinite(v) for v in q.values()):
                        raise ValueError("non-finite joint value")
                    items.append({"arm": arm, "q": q, "s": s})
                    public.append(clean({
                        "rank": s.rank, "description": s.description, "gains": s.gains,
                        "centre_px": s.centre_px, "outline_px": s.outline_px,
                        "q_deg": [math.degrees(q[n]) for n in cfg.joint_names]}))
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    bad += 1
                    self.log.warning("dropping a malformed suggestion: %s", e)
            stats = {str(k): int(v) for k, v in dict(stats or {}).items()}
            if bad:
                stats["malformed"] = bad
            with self.lock:
                if gen != self._gen:
                    return
                self._items, self._public, self._stats = items, public, stats
                self._computed_at = self.clock()
                if items:
                    self._state, self._selected = "ready", 0
                    self._message = f"{len(items)} suggestion(s)"
                else:
                    self._state, self._selected = "error", None
                    n_err = max(int(stats.get("ik_error", 0)), int(ik_errors["n"]))
                    if n_err > 0 and int(stats.get("accepted", 0)) == 0:
                        self._message = (f"IK service calls failed ({n_err} errors): is move_group running? "
                                         f"see the UI log" + (f" [first error: {str(ik_errors['first'])[:150]}]"
                                                              if ik_errors["first"] else ""))
                    else:
                        self._message = "no reachable collision-free pose found" + (
                            " (" + ", ".join(f"{k}: {v}" for k, v in sorted(stats.items())) + ")" if stats else "")
        except Exception as e:  # noqa: BLE001 - never leave the panel in "computing"
            self.log.exception("suggestion search failed")
            with self.lock:
                if gen == self._gen:
                    self._set_error(f"suggestion search failed: {e}")

    # ---- selection / RViz ------------------------------------------------------------------------
    def _check_index(self, index) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise SuggestRefused("index must be an integer", 400)
        with self.lock:
            n = len(self._items) if self._state == "ready" else 0
        if n == 0:
            raise SuggestRefused("no suggestion yet: press 'Suggest pose'", 409)
        if not 0 <= index < n:
            raise SuggestRefused(f"index must be 0..{n - 1}", 400)
        return index

    def select(self, index=0) -> int:
        index = self._check_index(index)
        with self.lock:
            changed = index != self._selected
            self._selected = index
            if changed and self._plan is not None and self._plan["index"] != index:
                self._clear_jog_plan()           # a stored plan must always belong to the selected suggestion
        return index

    def send_rviz(self, index=0) -> str:
        index = self._check_index(index)
        if self.moveit is None:
            raise SuggestRefused("no ROS bridge: cannot publish to RViz", 503)
        with self.lock:
            item = self._items[index]
            self._selected = index
        cfg = self.cfgs[item["arm"]]
        names = list(cfg.joint_names)
        self.moveit.publish_goal(names, [item["q"][n] for n in names])
        return RVIZ_HINT

    # ---- plan / move (through the jog controller) ---------------------------------------------------
    def plan(self, page_id: str, index=0, epoch=None) -> dict:
        index = self._check_index(index)
        why = self._jog_blocked()
        if why:
            raise SuggestRefused("cannot plan now: " + why, 409)
        fn = getattr(self.jog, "plan_move", None)
        if fn is None:
            raise SuggestRefused("jog module has no plan_move", 503)
        with self.lock:
            item = self._items[index]
            self._selected = index
        res = fn(page_id, item["arm"], dict(item["q"]), **({} if epoch is None else {"epoch": epoch}))
        res = dict(res) if isinstance(res, dict) else {"ok": False, "message": "bad answer from the jog module"}
        with self.lock:
            if res.get("ok") and res.get("plan_id") is not None:
                self._plan = {"plan_id": res["plan_id"], "index": index, "arm": item["arm"]}
            else:
                self._plan = None
        res["index"] = index
        return res

    def move(self, page_id: str, plan_id, epoch=None) -> dict:
        with self.lock:
            mine = None if self._plan is None else dict(self._plan)
            sel = self._selected
        if mine is None or mine["plan_id"] != plan_id:
            raise SuggestRefused("no stored plan with this id: press 'Plan preview' first", 409)
        if sel != mine["index"]:
            raise SuggestRefused("the stored plan is for another suggestion: press 'Plan preview' again", 409)
        fn = getattr(self.jog, "execute_move", None)
        if fn is None:
            raise SuggestRefused("jog module has no execute_move", 503)
        res = fn(page_id, plan_id, **({} if epoch is None else {"epoch": epoch}))
        res = dict(res) if isinstance(res, dict) else {"ok": False, "message": "bad answer from the jog module"}
        if res.get("ok"):
            with self.lock:
                self._plan = None                 # the jog consumes the plan
        return res

    # ---- views -----------------------------------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            st, msg, arm = self._state, self._message, self._arm
            public, sel, stats = list(self._public), self._selected, dict(self._stats)
            based, at, plan = self._based_on, self._computed_at, (dict(self._plan) if self._plan else None)
        count_arm = arm or (next(iter(self.cfgs)) if self.supported and self.cfgs else None)
        try:
            now_n = self.counter.count(self.session_dir_fn(), count_arm) if count_arm else None
        except Exception:  # noqa: BLE001
            now_n = None
        return {"supported": self.supported, "mode": self.mode,
                "state": st, "message": msg if self.supported else "single-arm modes only",
                "arm": arm, "based_on_samples": based, "session_samples": now_n,
                "stale": bool(st == "ready" and based is not None and now_n is not None and now_n != based),
                "age_s": None if at is None else max(0.0, self.clock() - at),
                "items": public, "selected": sel, "stats": stats, "plan": plan}

    def overlay(self) -> Optional[dict]:
        """{outline_px, label} of the selected suggestion for the live image, or None."""
        with self.lock:
            if self._state != "ready" or self._selected is None or self._selected >= len(self._public):
                return None
            it = self._public[self._selected]
            return {"outline_px": it.get("outline_px"), "label": f"suggested #{self._selected + 1}"}


# ---------------------------------------------------------------------------------------------------
# ROS side

class MoveItBridge:
    """MoveIt services plus the RViz goal-state publisher.  The blocking calls must run in a worker thread."""

    def __init__(self, node):
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetPositionIK, GetStateValidity
        from rclpy.callback_groups import ReentrantCallbackGroup
        self.node = node
        self._RobotState = RobotState
        self._cb = ReentrantCallbackGroup()
        self._cli_ik = node.create_client(GetPositionIK, "/compute_ik", callback_group=self._cb)
        self._cli_valid = node.create_client(GetStateValidity, "/check_state_validity", callback_group=self._cb)
        self._pub = node.create_publisher(RobotState, GOAL_STATE_TOPIC, 10)

    def ready(self) -> Tuple[bool, str]:
        if not self._cli_ik.service_is_ready():
            return False, "/compute_ik is not available (is move_group running?)"
        if not self._cli_valid.service_is_ready():
            return False, "/check_state_validity is not available (is move_group running?)"
        return True, ""

    def _call(self, client, request, timeout_s: float):
        done = threading.Event()
        fut = client.call_async(request)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            try:
                client.remove_pending_request(fut)
            except Exception:  # noqa: BLE001
                pass
            return None
        try:
            return fut.result()
        except Exception:  # noqa: BLE001
            return None

    def _state(self, q: Dict[str, float]):
        rs = self._RobotState()
        rs.joint_state.name = list(q)
        rs.joint_state.position = [float(v) for v in q.values()]
        return rs

    @staticmethod
    def base_frame(cfg) -> str:
        return f"{cfg.joint_prefix}base_link"

    @staticmethod
    def ik_link(cfg) -> str:
        return f"{cfg.joint_prefix}link6"

    def ik(self, cfg, T: np.ndarray, seed_q: Dict[str, float]) -> Optional[Dict[str, float]]:
        from builtin_interfaces.msg import Duration
        from moveit_msgs.msg import MoveItErrorCodes
        from moveit_msgs.srv import GetPositionIK
        from ..transforms import R_to_quat
        T = np.asarray(T, dtype=float).reshape(4, 4)
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = cfg.group, self.ik_link(cfg), True
        r.robot_state = self._state({n: seed_q[n] for n in cfg.joint_names if n in seed_q})
        r.pose_stamped.header.frame_id = self.base_frame(cfg)
        p, o = r.pose_stamped.pose.position, r.pose_stamped.pose.orientation
        p.x, p.y, p.z = (float(v) for v in T[:3, 3])
        o.x, o.y, o.z, o.w = (float(v) for v in R_to_quat(T[:3, :3]))
        r.timeout = Duration(sec=0, nanosec=int(IK_TIMEOUT_S * 1e9))
        res = self._call(self._cli_ik, req, SERVICE_TIMEOUT_S + IK_TIMEOUT_S)
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if not all(n in sol for n in cfg.joint_names):
            return None
        return {n: float(sol[n]) for n in cfg.joint_names}

    def valid(self, cfg, q: Dict[str, float]) -> Tuple[bool, str]:
        from moveit_msgs.srv import GetStateValidity
        req = GetStateValidity.Request()
        req.robot_state = self._state({n: q[n] for n in cfg.joint_names})
        req.group_name = cfg.group
        res = self._call(self._cli_valid, req, SERVICE_TIMEOUT_S)
        if res is None:
            return False, "/check_state_validity did not answer"
        if res.valid:
            return True, ""
        what = ""
        if res.contacts:
            c = res.contacts[0]
            what = f" ({c.contact_body_1} / {c.contact_body_2})"
        return False, "invalid state" + what

    def publish_goal(self, names: List[str], positions: List[float]) -> None:
        msg = self._RobotState()
        msg.joint_state.name = [str(n) for n in names]
        msg.joint_state.position = [float(v) for v in positions]
        msg.is_diff = False
        self._pub.publish(msg)
