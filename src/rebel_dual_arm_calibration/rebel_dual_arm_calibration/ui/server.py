"""Calibration web UI: Tornado on 127.0.0.1, rclpy on a background executor thread.

Every /api request needs the per-run token in ``X-Token``; POSTs also need a local Origin (or none).  The only motion
path is the injected jog controller; this module never publishes robot commands itself.
"""
import asyncio
import functools
import json
import logging
import math
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web

from .auth import Auth
from .autocapture import AutoCapture, CHANGE_CHOICES_DEG, DEFAULT_CHANGE_DEG, JointReading
from .hover_ui import (DEFAULT_DESCEND_MM, HoverManager, HoverRefused, corrections_from_params, warn_corrections)
from .jobs import JobBusy, JobManager
from .suggest import MoveItBridge, SuggestManager, SuggestRefused
from .tcp_ui import DEFAULT_XACRO_PATH, TcpManager, TcpRefused, make_tf_source
from .wrist_ui import WristManager, WristRefused

ARMS = ("left", "right")
ACTIVE_ARMS = {"dual": ("left", "right"), "single_left": ("left",), "single_right": ("right",)}
SESSION_NUMBERS = ("ambient_temp_c", "ambient_rh_pct", "warmup_min")
JOB_KIND = "calibrate_intrinsics"


def default_job_command(session_dir: str, out: str):
    return ["nice", "-n", "19", "ros2", "run", "rebel_dual_arm_calibration", "calibrate_intrinsics", session_dir,
            "--out", out]


def find_static_dir():
    """ui/static next to this file (source tree, colcon --symlink-install), else the installed share directory."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.isfile(os.path.join(here, "index.html")):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory("rebel_dual_arm_calibration"), "ui_static")
        if os.path.isdir(share):
            return share
    except Exception:  # noqa: BLE001
        pass
    return here


class UnavailableJog:
    """Stand-in when the jog module cannot be loaded: the UI still records, but nothing can move."""

    def __init__(self, reason: str):
        self.reason = reason

    def status(self):
        return {"available": False, "enabled": False, "error": self.reason}

    def _no(self, *a, **k):
        return {"ok": False, "id": None, "message": "jog unavailable: " + self.reason}

    enable_real = step = clear_fault = plan_move = execute_move = descend = _no

    def tcp_pose_base(self, arm, require_standstill=False):
        return None, "jog unavailable: " + self.reason

    tcp_fk = lambda self, arm, q: (None, "jog unavailable: " + self.reason)          # noqa: E731
    link6_to_tcp = lambda self, arm: (None, "jog unavailable: " + self.reason)       # noqa: E731

    def reserve_motion(self):
        return {"ok": False, "message": "jog unavailable: " + self.reason}

    def release_motion(self, token):
        return None

    def clear_plan(self):
        return None

    def stop(self):
        return {"ok": True, "message": "nothing to stop (jog unavailable)"}

    def disable(self, reason=""):
        return None

    def shutdown(self):
        return None


class Context:
    """Everything the handlers need.  ``bridge`` (recorder/camera access) and ``jog`` are injected, so tests use fakes.

    bridge: health() -> dict, recorder_status() -> dict|None, trigger(name, timeout_s) -> (ok, msg),
            set_parameters(dict) -> (ok, msg)
    pipeline: detections() -> dict, render_jpeg(width, overlay, status) -> bytes|None
    jog: JogController interface (status/enable_real/disable/step/stop/shutdown, and joint_window for the auto capture)
    auto: AutoCapture (default: built from the above; its clock / joint source / capture call can be injected)
    suggest: SuggestManager (default: one without MoveIt access, which refuses to compute; main() injects the real one)
    hover: HoverManager, the approach assist (default: one without camera / MoveIt access, which refuses; main() injects)
    tcp: TcpManager, the TCP pivot tab (default: one without a TF source, which refuses captures; main() injects)
    wrist: WristManager, the eye-in-hand wrist-camera tab (default None = "not available"; main() injects)"""

    def __init__(self, bridge, jog, pipeline, auth, mode="dual", jobs=None, job_command=default_job_command, auto=None,
                 suggest=None, hover=None, tcp=None, wrist=None):
        if mode not in ACTIVE_ARMS:
            raise ValueError(f"unknown mode '{mode}'")
        self.bridge, self.jog, self.pipeline, self.auth, self.mode = bridge, jog, pipeline, auth, mode
        self.jobs = jobs or JobManager()
        self.job_command = job_command
        self.action_lock = threading.Lock()          # one recorder action (params + trigger) at a time
        self.action_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-action")
        self.jog_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-jog")      # ordered enable/step
        self.stop_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-stop")    # STOP never queues behind others
        self.render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-render")
        self.auto_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-auto")    # one auto-capture tick at a time
        self.suggest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-suggest")   # fit + start of a search
        self.hover_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-hover")       # detect points, RViz goal
        self.tcp_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-tcp")           # TCP capture / solve / write
        self.wrist_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-wrist")       # wrist tab (locate, scene, ...)
        self.wrist = wrist
        self.log = logging.getLogger("calibration_ui")
        self.auto = auto or AutoCapture(
            time.monotonic, self._joint_reading if hasattr(jog, "joint_window") else None, self._auto_capture_call,
            self._jog_busy, lambda: bool((self.bridge.health() or {}).get("recorder")),
            unavailable="joint states are not available (jog module not loaded)")
        self._auto_tick_future = None
        self.suggest = suggest or SuggestManager(mode, self._arm_cfgs(mode), jog, self._session_dir)
        self.hover = hover or HoverManager(mode, self._arm_cfgs(mode), jog, self._session_dir)
        self.tcp = tcp or TcpManager(mode, self._arm_cfgs(mode), jog, self._session_dir)

    @staticmethod
    def _arm_cfgs(mode):
        try:
            from .jog import arm_configs
            return arm_configs(mode)
        except Exception:  # noqa: BLE001 - the panel then says "not available"
            return {}

    def _session_dir(self, fallback=None):
        """The recorder's own session directory (it writes the samples), else the ``session_dir`` parameter."""
        try:
            d = (self.bridge.recorder_status() or {}).get("session_dir")
        except Exception:  # noqa: BLE001
            d = None
        return d or fallback

    def shutdown(self):
        self.auto.stop(None, "auto capture stopped: UI shutting down")
        if self.wrist is not None:
            try:
                self.wrist.shutdown()           # stops a running sequence, removes the safety scene
            except Exception:  # noqa: BLE001
                self.log.exception("wrist shutdown failed")
        for pool in (self.action_pool, self.jog_pool, self.stop_pool, self.render_pool, self.auto_pool,
                     self.suggest_pool, self.hover_pool, self.tcp_pool, self.wrist_pool):
            pool.shutdown(wait=False, cancel_futures=True)

    # ---- auto capture plumbing (see autocapture.py) ----------------------------------------------
    def _joint_reading(self, window_s):
        r = self.jog.joint_window(self.active_arms, window_s)
        return None if r is None else JointReading(*r)

    def _jog_busy(self):
        try:
            st = self.jog.status()
        except Exception:  # noqa: BLE001 - unknown jog state: do not capture
            return True
        return bool(st.get("busy") or st.get("stopping") or st.get("fault"))

    def _auto_capture_call(self):
        """The manual Capture path (same lock, same trigger); None if a manual action holds it (try again later)."""
        if not self.action_lock.acquire(blocking=False):
            return None
        try:
            return self.bridge.trigger("capture", 20.0)
        finally:
            self.action_lock.release()

    def submit_auto_tick(self):
        """Called by the periodic callback on the loop thread: run one tick on the worker unless one is still running."""
        fut = self._auto_tick_future
        if fut is not None and not fut.done():
            return
        try:
            self._auto_tick_future = self.auto_pool.submit(self._run_auto_tick)
        except RuntimeError:                   # pool already shut down
            pass

    def _run_auto_tick(self):
        try:
            self.auto.tick()
        except Exception as e:  # noqa: BLE001 - never leave a run going that cannot be supervised
            self.log.exception("auto capture tick failed")
            self.auto.stop(None, f"auto capture stopped: internal error ({e})")

    @property
    def active_arms(self):
        return ACTIVE_ARMS[self.mode]

    def touch_list(self, status):
        """[{id, kind, done:{left,right}}] from the recorder status; None if the recorder status is unavailable."""
        if not status or "touch_list" not in status:
            return None
        touched = status.get("touched_ids") or {}
        return [{"id": int(t["id"]), "kind": t["kind"],
                 "done": {a: int(t["id"]) in set(touched.get(a, [])) for a in ARMS}} for t in status["touch_list"]]

    def _page_touch_list(self, status):
        """The touch list for the page: the recorder's configured touch ids plus (approach assist) hover tile / sheet points
        the targets file defines but the recorder does not list, as kind "hover" / capturable False - the page shows them so
        the mark can be hovered over, but /api/touch still only accepts the recorder's own ids.  Every item that is a hover
        point also carries ``tile`` (its source, "tile A") and ``tile_name`` ("A1") so the page can label it."""
        base = self.touch_list(status)
        if base is None:
            return []
        try:
            cat = {int(i): (n, src) for i, n, src in self.hover.catalog_points() if src != "table"}
            for t in base:
                if t["id"] in cat:
                    t["tile_name"], t["tile"] = cat[t["id"]]
            have = {t["id"] for t in base}
            extra = [{"id": int(i), "kind": "hover", "name": n, "tile": src, "tile_name": n, "capturable": False,
                      "done": {a: False for a in ARMS}}
                     for i, (n, src) in sorted(cat.items()) if int(i) not in have]
        except Exception:  # noqa: BLE001 - the state poll must survive a broken panel
            extra = []
        return base + extra

    def state(self, page_id=None):
        status = self.bridge.recorder_status()
        try:
            jog = self.jog.status()
        except Exception as e:  # noqa: BLE001 - the state poll must survive a broken jog module
            jog = {"available": False, "error": f"jog.status failed: {e}"}
        return {"ok": True, "mode": self.mode, "health": self.bridge.health(), "recorder": status, "jog": jog,
                "jobs": self.jobs.snapshot(), "touch_list": self._page_touch_list(status),
                "detections": self.pipeline.detections() if self.pipeline else {},
                "auto_capture": self.auto.snapshot(page_id), "suggest": self._suggest_state(),
                "hover": self._hover_state(), "tcp": self._tcp_state(page_id), "wrist": self._wrist_state()}

    def _wrist_state(self):
        if self.wrist is None:
            return {"supported": False, "message": "wrist-camera panel not available in this UI"}
        try:
            return self.wrist.state()
        except Exception as e:  # noqa: BLE001 - the state poll must survive a broken panel
            self.log.exception("wrist state failed")
            return {"supported": False, "message": f"wrist state failed: {e}"}

    def _suggest_state(self):
        try:
            return self.suggest.state()
        except Exception as e:  # noqa: BLE001 - the state poll must survive a broken panel
            return {"supported": False, "state": "error", "message": f"suggest state failed: {e}", "items": [],
                    "selected": None, "stats": {}, "based_on_samples": None, "stale": False, "plan": None}


    def _tcp_state(self, page_id=None):
        try:
            return self.tcp.state(page_id)
        except Exception as e:  # noqa: BLE001 - the state poll must survive a broken panel
            self.log.exception("tcp state failed")
            return {"supported": False, "message": f"tcp state failed: {e}", "arm": None, "samples": [],
                    "checklist": {"items": [], "all_checked": False}, "live": None, "coverage": None, "result": None,
                    "mechanical": None, "written": None, "diff": None, "warnings": [], "apply_steps": []}

    def _hover_state(self):
        try:
            return self.hover.state()
        except Exception as e:  # noqa: BLE001 - the state poll must survive a broken panel
            return {"supported": False, "state": "error", "message": f"hover state failed: {e}", "points": [],
                    "arm": None, "calib_samples": None, "prepared": None, "plan": None, "selected": None,
                    "test_summary": {"n": 0, "points": [], "mean_lateral_mm": None, "max_lateral_mm": None}}


class ApiHandler(tornado.web.RequestHandler):
    @property
    def ctx(self) -> Context:
        return self.application.settings["ctx"]

    def set_default_headers(self):
        self.set_header("Cache-Control", "no-store")

    def prepare(self):
        auth = self.ctx.auth
        if not auth.host_ok(self.request.headers.get("Host")):
            self.fail(403, "bad Host header")
        elif not auth.valid(self.request.headers.get("X-Token")):
            self.fail(401, "missing or wrong token")
        elif self.request.method == "POST" and not auth.origin_ok(self.request.headers.get("Origin")):
            self.fail(403, "bad Origin")

    def fail(self, status, message):
        self.set_status(status)
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({"ok": False, "message": message}))

    def reply(self, obj):
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(obj, allow_nan=False, default=str))

    def page_id(self):
        pid = self.request.headers.get("X-Page-Id")
        return pid if self.ctx.auth.page_id_ok(pid) else None

    def json_body(self):
        raw = self.request.body
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def write_error(self, status_code, **kwargs):
        if not self._finished:
            self.fail(status_code, self._reason)

    async def run(self, pool, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(pool, functools.partial(fn, *args))

    def reserve_motion(self):
        """HTTP ingress gate for plan / step / move requests: BEFORE the request is queued on the single jog worker,
        refuse at once (409) if the jog is busy / stopping / in fault or another motion request is already queued.
        Returns (ok, epoch, token); on ok the caller must run the request through run_motion().  A jog without the
        gate (fakes, UnavailableJog) is not gated here."""
        fn = getattr(self.ctx.jog, "reserve_motion", None)
        if fn is None:
            return True, None, None
        res = fn()
        if not res.get("ok"):
            self.fail(409, str(res.get("message") or "busy: the jog cannot take a motion request now"))
            return False, None, None
        return True, res.get("epoch"), res.get("token")

    async def run_motion(self, epoch, token, fn, *args, **kw):
        """Run a reserved motion request on the single jog worker.  The request carries the STOP epoch seen at ingress
        (a STOP pressed meanwhile voids it in begin_*) and gives the queue slot back when it is done."""
        ctx = self.ctx
        started = [False]

        def work():
            started[0] = True
            try:
                return fn(*args, **({"epoch": epoch} if epoch is not None else {}), **kw)
            finally:
                rel = getattr(ctx.jog, "release_motion", None)
                if rel is not None and token is not None:
                    rel(token)
        try:
            return await self.run(ctx.jog_pool, work)
        except RuntimeError:                       # the pool refused the task (shut down): give the slot back
            rel = getattr(ctx.jog, "release_motion", None)
            if rel is not None and token is not None and not started[0]:
                rel(token)
            raise

    async def run_locked(self, fn, *args):
        """Run a recorder action on the action pool; a second concurrent action gets 409 instead of queueing."""
        ctx = self.ctx
        if not ctx.action_lock.acquire(blocking=False):
            self.fail(409, "another action is in progress")
            return None, False

        def work():
            try:
                return fn(*args)
            finally:
                ctx.action_lock.release()
        return await self.run(ctx.action_pool, work), True


class StateHandler(ApiHandler):
    def get(self):
        pid = self.page_id()
        self.ctx.auto.heartbeat(pid)             # the owner page polling keeps its auto capture alive
        if self.ctx.wrist is not None:
            self.ctx.wrist.heartbeat(pid)        # ... and a running wrist sequence (a lost browser stops it)
        self.reply(self.ctx.state(pid))


class ImageHandler(ApiHandler):
    async def get(self):
        try:
            width = int(self.get_query_argument("w", "960"))
        except ValueError:
            return self.fail(400, "w must be an integer")
        overlay = self.get_query_argument("overlay", "detect")
        if overlay not in ("detect", "intrinsics", "none"):
            return self.fail(400, "overlay must be detect|intrinsics|none")
        ctx = self.ctx
        if ctx.pipeline is None:
            return self.fail(404, "no image yet")
        render = functools.partial(ctx.pipeline.render_jpeg, width, overlay, ctx.bridge.recorder_status())
        if self.get_query_argument("suggest", "0") == "1":
            ov = ctx.suggest.overlay()
            if ov is not None:
                render = functools.partial(render, suggest=ov)
        if self.get_query_argument("hover", "0") == "1":
            try:
                sel = int(self.get_query_argument("sel", ""))
            except ValueError:
                sel = None
            ov = ctx.hover.overlay(sel)
            if ov is not None:
                render = functools.partial(render, hover=ov)
        data = await self.run(ctx.render_pool, render)
        if data is None:
            return self.fail(404, "no image yet")
        self.set_header("Content-Type", "image/jpeg")
        self.finish(data)


class SessionHandler(ApiHandler):
    async def post(self):
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        values = {}
        for k in SESSION_NUMBERS:
            if body.get(k) is None:
                continue
            v = body[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return self.fail(400, f"{k} must be a finite number")
            values[k] = float(v)
        if body.get("operator_note") is not None:
            if not isinstance(body["operator_note"], str) or len(body["operator_note"]) > 1000:
                return self.fail(400, "operator_note must be a string of at most 1000 characters")
            values["operator_note"] = body["operator_note"]
        if not values:
            return self.fail(400, "give at least one of " + ", ".join(SESSION_NUMBERS + ("operator_note",)))
        res, ran = await self.run_locked(self.ctx.bridge.set_parameters, values)
        if ran:
            self.reply({"ok": res[0], "message": res[1]})


class CaptureHandler(ApiHandler):
    service = "capture"

    async def post(self):
        res, ran = await self.run_locked(self.ctx.bridge.trigger, self.service, 20.0)
        if ran:
            self.reply({"ok": res[0], "message": res[1]})


class IntrinsicsCaptureHandler(CaptureHandler):
    service = "capture_intrinsics"


class AutoCaptureHandler(ApiHandler):
    """Start / stop the server-side auto capture.  It never moves the robot, so it needs no "Enable motion"; it is owned by
    the page that started it (X-Page-Id)."""
    action = ""

    def post(self):
        auto, pid = self.ctx.auto, self.page_id()
        if pid is None:
            return self.fail(400, "X-Page-Id header missing or malformed")
        if self.action == "start":
            body = self.json_body()
            if body is None:
                return self.fail(400, "body must be a JSON object")
            deg = body.get("min_change_deg", DEFAULT_CHANGE_DEG)
            if isinstance(deg, bool) or not isinstance(deg, (int, float)) or float(deg) not in CHANGE_CHOICES_DEG:
                return self.fail(400, f"min_change_deg must be one of {[int(c) for c in CHANGE_CHOICES_DEG]}")
            ok, msg = auto.start(pid, float(deg))
        else:
            ok, msg = auto.stop(pid, "auto capture stopped")
        if not ok:
            return self.fail(409, msg)
        self.reply({"ok": True, "message": msg, "auto_capture": auto.snapshot(pid)})


def _auto_handler(action):
    return type(f"Auto_{action}", (AutoCaptureHandler,), {"action": action})


class TouchHandler(ApiHandler):
    async def post(self):
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        ctx = self.ctx
        arm, pid = body.get("arm"), body.get("point_id")
        if arm not in ARMS or arm not in ctx.active_arms:
            return self.fail(400, f"arm must be one of {list(ctx.active_arms)}")
        if isinstance(pid, bool) or not isinstance(pid, int):
            return self.fail(400, "point_id must be an integer")
        touch_list = ctx.touch_list(ctx.bridge.recorder_status())
        if touch_list is None:
            return self.fail(503, "recorder status not available yet; cannot validate point_id")
        if pid not in {t["id"] for t in touch_list}:
            return self.fail(400, f"point_id {pid} is not a touch point {sorted(t['id'] for t in touch_list)}")

        def work():
            ok, msg = ctx.bridge.set_parameters({"touch_arm": arm, "touch_point_id": pid})
            if not ok:
                return False, "cannot select the touch point: " + msg, None
            try:
                n_before = ctx.hover.touch_count()
            except Exception:  # noqa: BLE001
                n_before = None
            ok, msg = ctx.bridge.trigger("capture_touch", 20.0)
            off = None
            if ok and n_before is not None:       # approach assist: predicted vs touched (None without a prediction)
                try:
                    off = ctx.hover.after_touch(arm, pid, n_before)
                except Exception:  # noqa: BLE001
                    ctx.log.exception("hover comparison failed")
            return ok, msg, off
        res, ran = await self.run_locked(work)
        if ran:
            out = {"ok": res[0], "message": res[1]}
            if res[2]:
                out["hover_offset"] = res[2]
                out["message"] = f"{res[1]}; {res[2]['text']}"
            self.reply(out)


class CalibrateHandler(ApiHandler):
    def post(self):
        ctx = self.ctx
        status = ctx.bridge.recorder_status()
        if not status or not status.get("session_dir"):
            return self.fail(503, "recorder status not available yet; session directory unknown")
        if not status.get("n_intrinsics"):
            return self.fail(400, "no intrinsics views captured yet")
        session_dir = os.path.abspath(status["session_dir"])
        out = os.path.normpath(os.path.join(session_dir, "..", "intrinsics.yaml"))
        try:
            job = ctx.jobs.start(JOB_KIND, ctx.job_command(session_dir, out), result_file=out)
        except JobBusy:
            return self.fail(409, "a job is already running")
        self.reply({"ok": True, "job": job["id"], "message": f"started {job['id']}: {job['command']}"})


class JogHandler(ApiHandler):
    """Jog endpoints.  enable/step need the page id; STOP, clear_fault and disable do not and never wait for any
    other work."""
    action = ""

    async def post(self):
        ctx, jog = self.ctx, self.ctx.jog
        try:
            if self.action == "stop":
                res = await self.run(ctx.stop_pool, jog.stop)
            elif self.action == "clear_fault":
                res = await self.run(ctx.stop_pool, jog.clear_fault)
            elif self.action == "disable":
                await self.run(ctx.jog_pool, jog.disable, "operator")
                res = {"ok": True, "message": "jog disabled"}
            else:
                pid = self.page_id()
                if pid is None:
                    return self.fail(400, "X-Page-Id header missing or malformed")
                if self.action == "enable":
                    res = await self.run(ctx.jog_pool, jog.enable_real, pid)
                else:
                    body = self.json_body()
                    if body is None:
                        return self.fail(400, "body must be a JSON object")
                    missing = [k for k in ("arm", "mode", "axis", "sign", "size") if k not in body]
                    if missing:
                        return self.fail(400, "missing " + ", ".join(missing))
                    ok, epoch, token = self.reserve_motion()
                    if not ok:
                        return
                    res = await self.run_motion(epoch, token, jog.step, pid, body["arm"], body["mode"], body["axis"],
                                                body["sign"], body["size"], body.get("frame", "world"))
        except Exception as e:  # noqa: BLE001
            ctx.log.exception("jog %s failed", self.action)
            return self.fail(500, f"jog {self.action} failed: {e}")
        self.reply(res)


def _jog_handler(action):
    return type(f"Jog_{action}", (JogHandler,), {"action": action})


class SuggestHandler(ApiHandler):
    """Suggested-pose endpoints.  None of them moves the robot except ``move``, which goes through the jog controller
    (its enable / busy / STOP rules apply).  compute / plan / move need X-Page-Id."""
    action = ""

    def index_of(self, body):
        return body.get("index", 0)

    async def post(self):
        ctx, sg = self.ctx, self.ctx.suggest
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        pid = self.page_id()
        if self.action in ("plan", "move") and pid is None:
            return self.fail(400, "X-Page-Id header missing or malformed")
        epoch = token = None
        if self.action in ("plan", "move"):
            ok, epoch, token = self.reserve_motion()
            if not ok:
                return
        try:
            if self.action == "compute":
                arm = body.get("arm")
                if arm is not None and arm not in ARMS:
                    return self.fail(400, f"arm must be one of {list(ARMS)}")
                msg = await self.run(ctx.suggest_pool, sg.compute, arm)
                res = {"ok": True, "message": msg}
            elif self.action == "select":
                idx = await self.run(ctx.suggest_pool, sg.select, self.index_of(body))
                res = {"ok": True, "message": f"suggestion {idx + 1} selected", "selected": idx}
            elif self.action == "send_rviz":
                msg = await self.run(ctx.suggest_pool, sg.send_rviz, self.index_of(body))
                res = {"ok": True, "message": msg}
            elif self.action == "plan":
                res = await self.run_motion(epoch, token, sg.plan, pid, self.index_of(body))
            else:
                res = await self.run_motion(epoch, token, sg.move, pid, body.get("plan_id"))
        except SuggestRefused as e:
            return self.fail(e.status, e.message)
        except Exception as e:  # noqa: BLE001
            ctx.log.exception("suggest %s failed", self.action)
            return self.fail(500, f"suggest {self.action} failed: {e}")
        self.reply(res)


def _suggest_handler(action):
    return type(f"Suggest_{action}", (SuggestHandler,), {"action": action})


class HoverHandler(ApiHandler):
    """Approach-assist endpoints.  prepare / move / descend can move the robot (prepare only plans); they pass the same
    ingress gate (reserve_motion) and carry the STOP epoch like a jog step or "Move there", and need X-Page-Id.  compute,
    send_rviz (one RobotState on RViz's goal topic) and record_offset do not move anything."""
    action = ""
    MOTION = ("prepare", "move", "descend")

    async def post(self):
        ctx, hv = self.ctx, self.ctx.hover
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        pid = self.page_id()
        if self.action in self.MOTION and pid is None:
            return self.fail(400, "X-Page-Id header missing or malformed")
        point = body.get("point_id")
        if self.action in ("prepare", "send_rviz", "descend", "record_offset") and (isinstance(point, bool)
                                                                                       or not isinstance(point, int)):
            return self.fail(400, "point_id must be an integer")
        stage = body.get("stage", "prehover")
        if self.action == "prepare" and not isinstance(stage, str):
            return self.fail(400, "stage must be a string")
        height = body.get("height_mm", DEFAULT_DESCEND_MM)
        if self.action == "descend" and (isinstance(height, bool) or not isinstance(height, (int, float))):
            return self.fail(400, "height_mm must be a number (10, 20 or 30)")
        arm = body.get("arm")
        if self.action == "compute" and arm is not None and arm not in ARMS:
            return self.fail(400, f"arm must be one of {list(ARMS)}")
        epoch = token = None
        if self.action in self.MOTION:
            ok, epoch, token = self.reserve_motion()
            if not ok:
                return
        try:
            if self.action == "compute":
                res = {"ok": True, "message": await self.run(ctx.hover_pool, hv.compute, arm)}
            elif self.action == "send_rviz":
                res = {"ok": True, "message": await self.run(ctx.hover_pool, hv.send_rviz, point)}
            elif self.action == "record_offset":
                res = await self.run(ctx.hover_pool, hv.record_offset, point)
            elif self.action == "prepare":
                res = await self.run_motion(epoch, token, hv.prepare, pid, point, stage)
            elif self.action == "descend":
                res = await self.run_motion(epoch, token, hv.descend, pid, point, height)
            else:
                res = await self.run_motion(epoch, token, hv.move, pid, body.get("plan_id"))
        except HoverRefused as e:
            return self.fail(e.status, e.message)
        except Exception as e:  # noqa: BLE001
            ctx.log.exception("hover %s failed", self.action)
            return self.fail(500, f"hover {self.action} failed: {e}")
        self.reply(res)


def _hover_handler(action):
    return type(f"Hover_{action}", (HoverHandler,), {"action": action})


class TcpHandler(ApiHandler):
    """TCP pivot tab endpoints.  None of them moves the robot (the operator positions the arm with the Servo teleop) and
    none writes the xacro: ``write`` stores ``tcp_result_<arm>.yaml`` in the session directory only."""
    action = ""

    async def post(self):
        ctx, tcp = self.ctx, self.ctx.tcp
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        try:
            pid = self.page_id()
            if self.action in ("checklist", "capture") and pid is None:
                return self.fail(400, "X-Page-Id header missing or malformed")
            if self.action == "checklist":
                items = body.get("items")
                if not isinstance(items, dict) or not items or any(not isinstance(v, bool) for v in items.values()):
                    return self.fail(400, "items must be an object of checklist key -> true/false")
                await self.run(ctx.tcp_pool, tcp.set_checklist, items, pid)
                res = {"ok": True, "message": "checklist updated"}
            elif self.action == "capture":
                role = body.get("role")
                if not isinstance(role, str):
                    return self.fail(400, "role must be 'fit' or 'validate'")
                res = {"ok": True, **await self.run(ctx.tcp_pool, tcp.capture, role, body.get("note") or "", pid)}
            elif self.action == "delete":
                res = {"ok": True, **await self.run(ctx.tcp_pool, tcp.delete, body.get("index"))}
            elif self.action == "solve":
                res = {"ok": True, **await self.run(ctx.tcp_pool, tcp.solve)}
            elif self.action == "mechanical":
                res = {"ok": True, **await self.run(ctx.tcp_pool, tcp.mechanical, body.get("length_mm"))}
            else:
                res = {"ok": True, **await self.run(ctx.tcp_pool, tcp.write)}
        except TcpRefused as e:
            return self.fail(e.status, e.message)
        except Exception as e:  # noqa: BLE001
            ctx.log.exception("tcp %s failed", self.action)
            return self.fail(500, f"tcp {self.action} failed: {e}")
        res["tcp"] = ctx._tcp_state(self.page_id())
        self.reply(res)


def _tcp_handler(action):
    return type(f"Tcp_{action}", (TcpHandler,), {"action": action})


class WristHandler(ApiHandler):
    """Wrist-camera (eye-in-hand) tab.  Only ``start`` leads to robot motion, and that goes through the jog controller's
    plan_move / execute_move (see ui/wrist_ui.py); ``stop`` also sends the jog STOP.  start needs X-Page-Id and the explicit
    per-run confirmation in the body ({"confirm": true})."""
    action = ""
    PLAIN = ("locate_board", "scene_add", "scene_remove", "seed", "generate", "capture", "solve", "write", "publish_tf")

    async def post(self):
        ctx, w = self.ctx, self.ctx.wrist
        if w is None:
            return self.fail(503, "wrist-camera panel not available in this UI")
        body = self.json_body()
        if body is None:
            return self.fail(400, "body must be a JSON object")
        try:
            if self.action == "stop":
                res = await self.run(ctx.stop_pool, w.stop)
            elif self.action == "start":
                pid = self.page_id()
                if pid is None:
                    return self.fail(400, "X-Page-Id header missing or malformed")
                res = await self.run(ctx.wrist_pool, w.start, pid, body.get("confirm"))
            elif self.action == "mock_board":
                res = {"ok": True, "board": await self.run(ctx.wrist_pool, w.mock_board, body)}
            elif self.action == "delete_sample":
                res = await self.run(ctx.wrist_pool, w.delete_sample, body.get("index"))
            elif self.action in self.PLAIN:
                fn = {"locate_board": w.locate_board, "scene_add": w.scene_add, "scene_remove": w.scene_remove,
                      "seed": w.use_seed, "generate": w.generate, "capture": w.capture_now, "solve": w.solve,
                      "write": w.write, "publish_tf": w.publish_tf}[self.action]
                out = await self.run(ctx.wrist_pool, fn)
                res = out if isinstance(out, dict) and "ok" in out else {"ok": True, "message": out if isinstance(out, str)
                                                                         else f"{self.action} done", "result": out}
            else:
                return self.fail(404, "unknown wrist action")
        except WristRefused as e:
            return self.fail(e.status, e.message)
        except Exception as e:  # noqa: BLE001
            ctx.log.exception("wrist %s failed", self.action)
            return self.fail(500, f"wrist {self.action} failed: {e}")
        res = dict(res)
        res["wrist"] = ctx._wrist_state()
        self.reply(res)


def _wrist_handler(action):
    return type(f"Wrist_{action}", (WristHandler,), {"action": action})


class WristImageHandler(ApiHandler):
    async def get(self):
        w = self.ctx.wrist
        if w is None:
            return self.fail(404, "no wrist panel")
        try:
            width = int(self.get_query_argument("w", "960"))
        except ValueError:
            return self.fail(400, "w must be an integer")
        data = await self.run(self.ctx.render_pool, w.render_jpeg, max(160, min(width, 1920)))
        if data is None:
            return self.fail(404, "no wrist image yet")
        self.set_header("Content-Type", "image/jpeg")
        self.finish(data)


class Index(tornado.web.RequestHandler):
    """The page itself is public on localhost (it carries no data); everything it fetches needs the token."""

    def get(self):
        path = os.path.join(self.application.settings["static_dir"], "index.html")
        if not os.path.isfile(path):
            self.set_status(404)
            return self.finish("index.html not found")
        self.set_header("Cache-Control", "no-store")
        with open(path, "rb") as f:
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.finish(f.read())


def _log_request(handler):
    if handler.get_status() < 400 and handler.request.path in ("/api/state", "/api/image.jpg"):
        return
    logging.getLogger("calibration_ui.http").info("%d %s %s", handler.get_status(), handler.request.method,
                                                  handler.request.path)


def make_app(ctx, static_dir=None):
    static_dir = static_dir or find_static_dir()
    routes = [
        (r"/", Index),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_dir}),
        (r"/api/state", StateHandler),
        (r"/api/image\.jpg", ImageHandler),
        (r"/api/session", SessionHandler),
        (r"/api/capture", CaptureHandler),
        (r"/api/auto_capture/start", _auto_handler("start")),
        (r"/api/auto_capture/stop", _auto_handler("stop")),
        (r"/api/touch", TouchHandler),
        (r"/api/intrinsics/capture", IntrinsicsCaptureHandler),
        (r"/api/intrinsics/calibrate", CalibrateHandler),
    ] + [(rf"/api/jog/{a}", _jog_handler(a)) for a in ("enable", "disable", "step", "stop", "clear_fault")] \
        + [(rf"/api/suggest/{a}", _suggest_handler(a)) for a in ("compute", "select", "send_rviz", "plan", "move")] \
        + [(rf"/api/hover/{a}", _hover_handler(a))
           for a in ("compute", "prepare", "send_rviz", "move", "descend", "record_offset")] \
        + [(rf"/api/tcp/{a}", _tcp_handler(a)) for a in ("checklist", "capture", "delete", "solve", "mechanical", "write")] \
        + [(rf"/api/wrist/{a}", _wrist_handler(a))
           for a in ("locate_board", "mock_board", "scene_add", "scene_remove", "seed", "generate", "start", "stop",
                     "capture", "solve", "write", "publish_tf", "delete_sample")] \
        + [(r"/api/wrist/image\.jpg", WristImageHandler)]
    return tornado.web.Application(routes, ctx=ctx, static_dir=static_dir, log_function=_log_request)


def make_hover_scene(targets_file, min_marker_perimeter_rate=0.02):
    """scene_fn for the HoverManager: (table_spec, sheet_spec, detector) from the targets file, built once on first use.
    ``sheet_spec`` is the list of hover targets (the ``hover_tiles`` tiles, then a legacy ``hover_sheet``; None if there are
    none).  The detector is its own TargetDetector (the live-image pipeline's runs on another thread); it holds the arm
    boards and the table board, the hover tiles are handled by hover.detect_targets itself (their specs are passed along)."""
    from types import SimpleNamespace

    from ..targets import TargetDetector, load_hover_sheets, load_targets_all
    cache = {}
    lock = threading.Lock()

    def scene():
        with lock:
            if "scene" not in cache:
                specs, table = load_targets_all(targets_file)
                sheet = load_hover_sheets(targets_file) or None
                if table is None and sheet is None:
                    cache["scene"] = None
                else:
                    all_specs = dict(specs)
                    if table is not None:
                        all_specs["table"] = table
                    cache["scene"] = SimpleNamespace(table_spec=table, sheet_spec=sheet,
                                                     detector=TargetDetector(all_specs, float(min_marker_perimeter_rate)))
            return cache["scene"]
    return scene


def make_wrist_manager(node, mode, bridge, jog, moveit, targets_file, session_dir, gp):
    """The wrist-camera tab (single-arm modes): own subscriptions for the wrist topics, the table-board detector, the
    output directory ``wrist_handeye_output_dir`` (default ``<session dir>/wrist_handeye``)."""
    from ..targets import TargetDetector, load_targets_all
    from .jog import arm_configs
    from .wrist_ros import WristRos
    cfgs = arm_configs(mode)
    specs, table = load_targets_all(targets_file)
    if table is None:
        raise RuntimeError("the targets file has no table_board")
    cache, lock = {}, threading.Lock()

    class LockedDetector:           # the live preview and the capture run on different threads
        def __init__(self, det):
            self.det, self.lock, self.specs = det, threading.Lock(), det.specs

        def detect(self, image):
            with self.lock:
                return self.det.detect(image)

    def detector():
        with lock:
            if "d" not in cache:
                cache["d"] = LockedDetector(TargetDetector({"table": table}, float(gp("min_marker_perimeter_rate"))))
            return cache["d"], table
    explicit = os.path.expanduser(str(gp("wrist_handeye_output_dir") or ""))

    def out_dir():
        if explicit:
            return explicit
        base = (bridge.recorder_status() or {}).get("session_dir") or session_dir
        return os.path.join(base, "wrist_handeye") if base else None
    ros = WristRos(node, str(gp("wrist_image_topic")), str(gp("wrist_camera_info_topic")), bridge)
    return WristManager(mode, cfgs, jog, ros, moveit, out_dir, detector, allow_real_move=bool(gp("allow_real_move")),
                        camera_serial=str(gp("wrist_camera_serial") or ""), cad_roll_deg=float(gp("wrist_cad_roll_deg")))


def main(args=None):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from ..targets import TargetDetector, default_targets_path, load_targets_all
    from .overlay import ImagePipeline
    from .ros_bridge import RosBridge

    rclpy.init(args=args)
    node = rclpy.create_node("calibration_ui")
    for name, default in (("mode", "dual"), ("port", 8095), ("image_topic", "/camera/color/image_raw"),
                          ("camera_info_topic", "/camera/color/camera_info"), ("targets_file", ""),
                          ("min_marker_perimeter_rate", 0.02), ("session_dir", ""),
                          ("allow_real_move", False), ("tcp_xacro_path", DEFAULT_XACRO_PATH),
                          ("wrist_image_topic", "/arm_left/wrist_camera/color/image_raw"),
                          ("wrist_camera_info_topic", "/arm_left/wrist_camera/color/camera_info"),
                          ("wrist_handeye_output_dir", ""), ("wrist_camera_serial", ""), ("wrist_cad_roll_deg", 0.0)):
        node.declare_parameter(name, default)
    from rcl_interfaces.msg import ParameterDescriptor
    for name in ("hover_correction_left_mm", "hover_correction_right_mm"):     # "x,y,z" string or a 3-number array
        node.declare_parameter(name, "", ParameterDescriptor(dynamic_typing=True))
    gp = lambda n: node.get_parameter(n).value  # noqa: E731
    mode, port = str(gp("mode")), int(gp("port"))
    logging.basicConfig(level=logging.INFO)
    try:        # an invalid hover correction stops the UI here (NOT inside the optional-panel try blocks below)
        hover_correction = corrections_from_params(gp("hover_correction_left_mm"), gp("hover_correction_right_mm"))
    except ValueError as e:
        node.get_logger().error(f"invalid hover prediction correction, refusing to start: {e}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"calibration UI: invalid hover prediction correction: {e}")
    warn_corrections(hover_correction, mode, node.get_logger().warning)
    import cv2
    cv2.setNumThreads(2)

    targets_file = os.path.expanduser(gp("targets_file") or "") or default_targets_path()

    def make_detector():
        specs, table = load_targets_all(targets_file)
        all_specs = dict(specs)
        if table is not None:
            all_specs["table"] = table
        return TargetDetector(all_specs, float(gp("min_marker_perimeter_rate")))

    bridge = RosBridge(node, gp("image_topic"), gp("camera_info_topic"))
    pipeline = ImagePipeline(bridge.latest_frame, make_detector,
                             min_corners_fn=lambda: (bridge.recorder_status() or {}).get("min_corners"))
    try:
        from .jog import JogController
        jog = JogController(node, mode, allow_real_move=bool(gp("allow_real_move")))
        if gp("allow_real_move"):
            node.get_logger().warning("allow_real_move:=true: 'Move there' may move the REAL robot (with Enable motion + confirm)")
    except Exception as e:  # noqa: BLE001 - recording still works without jog
        node.get_logger().error(f"jog module unavailable, jogging disabled: {e}")
        jog = UnavailableJog(str(e))
    auth = Auth(port)
    session_dir = os.path.expanduser(str(gp("session_dir") or ""))
    suggest = None
    try:
        from .jog import arm_configs
        suggest = SuggestManager(mode, arm_configs(mode), jog,
                                 lambda: (bridge.recorder_status() or {}).get("session_dir") or session_dir or None,
                                 MoveItBridge(node))
    except Exception as e:  # noqa: BLE001 - recording and jogging still work without the suggestion panel
        node.get_logger().error(f"suggested-pose panel unavailable: {e}")
    hover = None
    try:
        from .jog import arm_configs
        hover = HoverManager(mode, arm_configs(mode), jog,
                             lambda: (bridge.recorder_status() or {}).get("session_dir") or session_dir or None,
                             MoveItBridge(node) if suggest is None else suggest.moveit,
                             image_fn=bridge.latest_frame, scene_fn=make_hover_scene(targets_file, gp("min_marker_perimeter_rate")),
                             correction_mm=hover_correction)
    except Exception as e:  # noqa: BLE001 - recording, jogging and the suggestions still work without the approach assist
        node.get_logger().error(f"approach assist unavailable: {e}")
    tcp = None
    try:
        from .jog import arm_configs
        tcp = TcpManager(mode, arm_configs(mode), jog,
                         lambda: (bridge.recorder_status() or {}).get("session_dir") or session_dir or None,
                         tf_fn=make_tf_source(node), xacro_path=str(gp("tcp_xacro_path") or DEFAULT_XACRO_PATH))
    except Exception as e:  # noqa: BLE001 - recording, jogging and the other panels still work without the TCP tab
        node.get_logger().error(f"TCP calibration tab unavailable: {e}")
    wrist = None
    if mode in ("single_left", "single_right"):
        try:
            wrist = make_wrist_manager(node, mode, bridge, jog, suggest.moveit if suggest is not None else MoveItBridge(node),
                                       targets_file, session_dir, gp)
        except Exception as e:  # noqa: BLE001 - everything else still works without the wrist tab
            node.get_logger().error(f"wrist-camera tab unavailable: {e}")
    ctx = Context(bridge, jog, pipeline, auth, mode, suggest=suggest, hover=hover, tcp=tcp, wrist=wrist)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, name="ros-executor", daemon=True)
    ros_thread.start()
    pipeline.start()

    async def serve():
        app = make_app(ctx)
        server = app.listen(port, address="127.0.0.1")
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        tornado.ioloop.PeriodicCallback(ctx.submit_auto_tick, 500).start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        node.get_logger().info(f"calibration UI (mode {mode}) listening on 127.0.0.1:{port}")
        node.get_logger().info(f"open {auth.url()}")
        print(f"calibration UI: {auth.url()}", flush=True)
        await stopped.wait()
        server.stop()

    try:
        asyncio.run(serve())
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):      # a second signal (the launcher forwards one too) must not cut the cleanup
            try:                                         # short: it removes the wrist safety scene from the planning scene
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        try:
            jog.stop()
        except Exception:  # noqa: BLE001
            pass
        for fn in (jog.shutdown, ctx.jobs.shutdown, pipeline.stop, ctx.shutdown):
            try:
                fn()
            except Exception:  # noqa: BLE001
                node.get_logger().error(f"shutdown step {fn} failed")
        executor.shutdown()
        ros_thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
