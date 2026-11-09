"""Dual-cut operator web UI: Tornado on 127.0.0.1, rclpy on a background executor thread.

Every /api request needs the per-run token in ``X-Token``; POSTs also need a local Origin (or none).  The page can only
ask for the whitelisted actions of core.ACTIONS; ``software_stop`` runs on its own thread pool and never waits behind
other work.  This UI is NOT an emergency stop.
"""
import asyncio
import functools
import json
import logging
import logging.handlers
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import tornado.ioloop
import tornado.web

from .auth import Auth
from .camera import DEFAULT_CAMERAS_OFF, CameraHub, overlay_jpeg, parse_camera_off, parse_camera_spec
from .placement import PlacementLimits
from .domains import check_localhost_param, plan_domains, select_camera_source, startup_message
from .core import (Config, DEFAULT_DETECT_ACTION, DEFAULT_DETECT_CWD, DEFAULT_DETECT_OUTPUT_DIR, DEFAULT_SNAPSHOT_FILE,
                   UiCore)

LOG = logging.getLogger("dual_cut_ui")


def find_static_dir() -> str:
    """static/ next to this file (source tree, colcon --symlink-install), else the installed share directory."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.isfile(os.path.join(here, "index.html")):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory("dual_cut_ui"), "ui_static")
        if os.path.isdir(share):
            return share
    except Exception:  # noqa: BLE001
        pass
    return here


class Context:
    def __init__(self, core: UiCore, hub: CameraHub, auth: Auth, overlay_widths=None, jpeg_quality: int = 80):
        self.core, self.hub, self.auth = core, hub, auth
        self.overlay_widths = overlay_widths or {"small": 640, "large": 1280}
        self.jpeg_quality = jpeg_quality
        self.action_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ui-action")
        self.stop_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-stop")      # STOP never queues behind others
        self.image_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-image")
        # /api/state never runs on the IOLoop thread (and not on the stop or action pools): the loop must always be free
        # to accept a Software stop POST, however slow a state build is
        self.state_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-state")
        self._overlay_cache = {}
        self._overlay_lock = threading.Lock()

    def overlay_bytes(self, path: str, mtime: float, size: str) -> Optional[bytes]:
        key = (path, mtime, size)
        with self._overlay_lock:
            got = self._overlay_cache.get(key)
        if got is not None:
            return got
        data = overlay_jpeg(path, self.overlay_widths[size], self.jpeg_quality)
        if data is not None:
            with self._overlay_lock:
                if len(self._overlay_cache) > 6:
                    self._overlay_cache.clear()
                self._overlay_cache[key] = data
        return data

    def shutdown(self) -> None:
        for pool in (self.action_pool, self.stop_pool, self.image_pool, self.state_pool):
            pool.shutdown(wait=False, cancel_futures=True)


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

    def write_error(self, status_code, **kwargs):
        if not self._finished:
            self.fail(status_code, self._reason)

    async def run(self, pool, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(pool, functools.partial(fn, *args))


class StateHandler(ApiHandler):
    async def get(self):
        ctx = self.ctx
        view, pid = self.get_query_argument("view", ""), self.page_id()

        def build():
            state = ctx.core.state(pid, view)
            state["cameras"] = ctx.hub.status()
            state["cameras_default_off"] = ctx.hub.default_off
            return state
        try:
            state = await self.run(ctx.state_pool, build)
        except RuntimeError:                      # pool shut down
            return self.fail(503, "server is shutting down")
        self.reply(state)


class CameraHandler(ApiHandler):
    async def get(self, name):
        ctx = self.ctx
        if name not in ctx.hub.cameras:
            return self.fail(404, "unknown camera")
        size = self.get_query_argument("size", "small")
        if size not in ctx.hub.SIZES:
            return self.fail(400, "size must be small or large")
        data, reason = await self.run(ctx.image_pool, ctx.hub.request, name, size)
        if data is None:
            return self.fail(404 if reason == "unknown camera" else 503, reason)
        self.set_header("Content-Type", "image/jpeg")
        self.finish(data)


class DetectionHandler(ApiHandler):
    async def get(self):
        ctx = self.ctx
        size = self.get_query_argument("size", "large")
        if size not in ctx.overlay_widths:
            return self.fail(400, "size must be small or large")
        info = await self.run(ctx.image_pool, ctx.core.detection_info)       # globbing / comparing files: not on the IOLoop
        if info is None:
            return self.fail(404, "no detection overlay for the current snapshot")
        data = await self.run(ctx.image_pool, ctx.overlay_bytes, info["overlay"], info["mtime"], size)
        if data is None:
            return self.fail(404, "overlay not readable")
        self.set_header("Content-Type", "image/jpeg")
        self.finish(data)


class ActionHandler(ApiHandler):
    async def post(self):
        raw = self.request.body
        try:
            body = json.loads(raw) if raw.strip() else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return self.fail(400, "body must be a JSON object")
        action = body.get("action")
        ctx = self.ctx
        pool = ctx.stop_pool if action == "software_stop" else ctx.action_pool
        try:
            status, reply = await self.run(pool, ctx.core.dispatch, action, body.get("args"), self.page_id())
        except RuntimeError:                      # pool shut down
            return self.fail(503, "server is shutting down")
        self.set_status(status)
        self.reply(reply)


class Index(tornado.web.RequestHandler):
    """The page itself is public on localhost (it carries no data); everything it fetches needs the token."""

    def get(self):
        path = os.path.join(self.application.settings["static_dir"], "index.html")
        if not os.path.isfile(path):
            self.set_status(404)
            return self.finish("index.html not found")
        self.set_header("Cache-Control", "no-store")
        with open(path, "rb") as f:
            page = f.read()
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.finish(versioned_assets(page, self.application.settings["static_dir"]))


def versioned_assets(page: bytes, static_dir: str) -> bytes:
    """/static/app.js -> /static/app.js?v=<mtime>, same for style.css: a rebuilt UI is picked up even by a browser that
    cached the old file."""
    for name in ("app.js", "style.css"):
        try:
            v = str(os.stat(os.path.join(static_dir, name)).st_mtime_ns).encode()
        except OSError:
            continue
        page = page.replace(b'"/static/' + name.encode() + b'"', b'"/static/' + name.encode() + b'?v=' + v + b'"')
    return page


class StaticHandler(tornado.web.StaticFileHandler):
    """Revalidate every time (ETag -> 304 when unchanged), so a rebuilt UI never runs a stale app.js from the browser cache."""

    def set_extra_headers(self, path):
        self.set_header("Cache-Control", "no-cache")


def _log_request(handler):
    if handler.get_status() < 400 and (handler.request.path in ("/api/state", "/api/detection.jpg")
                                        or handler.request.path.startswith("/api/camera/")):
        return
    logging.getLogger("dual_cut_ui.http").info("%d %s %s", handler.get_status(), handler.request.method, handler.request.path)


def make_app(ctx: Context, static_dir: Optional[str] = None):
    static_dir = static_dir or find_static_dir()
    routes = [
        (r"/", Index),
        (r"/static/(.*)", StaticHandler, {"path": static_dir}),
        (r"/api/state", StateHandler),
        (r"/api/camera/([A-Za-z0-9_]+)\.jpg", CameraHandler),
        (r"/api/detection\.jpg", DetectionHandler),
        (r"/api/action", ActionHandler),
    ]
    return tornado.web.Application(routes, ctx=ctx, static_dir=static_dir, log_function=_log_request)


SHUTDOWN_BUDGET_S = 4.0          # ros2 launch escalates SIGTERM -> SIGKILL; everything below must fit well inside this
DEFAULT_LOG_FILE = "~/.ros/dual_cut_ui/dual_cut_ui.log"


def _ui_logger() -> logging.Logger:
    return logging.getLogger("dual_cut_ui")


def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> None:
    """Handlers on the "dual_cut_ui" logger itself (stderr, plus best effort a rotating file).  The file matters at
    shutdown: when the launcher dies first, stderr is a closed pipe and the shutdown lines would be lost.  Logging
    problems never raise into the shutdown path."""
    logging.raiseExceptions = False
    log = _ui_logger()
    log.setLevel(logging.INFO)
    log.propagate = False                     # we own the handlers (and a foreign root configuration cannot swallow the lines)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in list(log.handlers):
        log.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    path = os.path.expanduser(log_file) if log_file else ""
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2)
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError as e:
            log.warning("no log file %s: %s", path, e)


def flush_logs() -> None:
    for h in list(_ui_logger().handlers):
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass


def run_shutdown(steps: List[Tuple[str, Callable[[], None]]], log: Callable[[str], None],
                 budget_s: float = SHUTDOWN_BUDGET_S, clock: Callable[[], float] = time.monotonic) -> dict:
    """Run the shutdown steps in order, each on a helper thread, so that the whole sequence ends within ``budget_s`` even
    if a step hangs.  Every step and the total are logged (and the logs flushed).  Returns {name: "ok" | "timeout" |
    "skipped" | "error: ..."}."""
    t0 = clock()
    results = {}
    log(f"dual-cut UI shutting down (budget {budget_s:.1f} s)")
    flush_logs()
    for name, fn in steps:
        remaining = budget_s - (clock() - t0)
        if remaining <= 0.05:
            results[name] = "skipped"
            log(f"shutdown step {name}: skipped (out of time)")
            continue
        out = {}

        def work(fn=fn, out=out):
            try:
                fn()
                out["r"] = "ok"
            except Exception as e:  # noqa: BLE001
                out["r"] = f"error: {e}"
        t = threading.Thread(target=work, name=f"shutdown-{name}", daemon=True)
        t.start()
        t.join(remaining)
        results[name] = out.get("r", "timeout")
        log(f"shutdown step {name}: {results[name]}")
        flush_logs()
    log(f"dual-cut UI shutdown finished in {clock() - t0:.2f} s")
    flush_logs()
    return results


def shutdown_steps(core, hub, ctx, camera_context_teardown: Optional[Callable[[], None]] = None) -> List[Tuple[str, Callable[[], None]]]:
    """The ordered shutdown steps of main(): gates first, then the camera subscriptions, the pools, and (only if a second ROS
    context for the cameras exists) that context last, after all the main node's work."""
    steps = [("core (gates false, job stopped)", lambda: core.shutdown(budget_s=SHUTDOWN_BUDGET_S - 1.0)),
             ("cameras", hub.shutdown), ("pools", ctx.shutdown)]
    if camera_context_teardown is not None:
        steps.append(("camera ROS context", camera_context_teardown))
    return steps


def main(args=None):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .ros_bridge import CameraContext, RosBridge

    from rclpy.signals import SignalHandlerOptions
    # our own SIGINT/SIGTERM handling: the gates must be closed (needs a live context) before rclpy shuts down
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop_requested = threading.Event()

    def early_signal(signum, _frame):           # a signal during start-up still leads to the orderly shutdown below
        stop_requested.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, early_signal)
    node = rclpy.create_node("dual_cut_ui")
    for name, default in (("port", 8096), ("snapshot_file", DEFAULT_SNAPSHOT_FILE), ("detect_cwd", DEFAULT_DETECT_CWD),
                          ("detect_output_dir", DEFAULT_DETECT_OUTPUT_DIR), ("cameras", ""), ("cameras_off", DEFAULT_CAMERAS_OFF), ("plan_timeout_s", 300.0),
                          ("control_grace_s", 10.0), ("camera_small_width", 320), ("camera_large_width", 960),
                          ("camera_jpeg_quality", 70), ("camera_idle_s", 10.0), ("coordinator", "/dual_cut_coordinator"),
                          ("log_file", DEFAULT_LOG_FILE), ("camera_domain_id", -1), ("detect_domain_id", -1),
                          ("camera_localhost_only", -1), ("detect_mode", "action"), ("detect_action", DEFAULT_DETECT_ACTION),
                          ("detect_timeout_s", 300.0), ("schedule_mode", "sequence"), ("placement_limits", "")):
        node.declare_parameter(name, default)
    node.declare_parameter("detect_command", [""])        # string array; [""] = not given (use the default command)
    gp = lambda n: node.get_parameter(n).value  # noqa: E731
    setup_logging(str(gp("log_file")))
    port = int(gp("port"))
    try:
        cameras = parse_camera_spec(str(gp("cameras") or ""))
    except ValueError as e:
        node.get_logger().error(f"bad cameras parameter: {e}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"dual_cut_ui: bad cameras parameter: {e}")
    try:
        node_domain = node.context.get_domain_id()
        domains = plan_domains(node_domain, int(gp("camera_domain_id")), int(gp("detect_domain_id")))
        camera_localhost = check_localhost_param("camera_localhost_only", int(gp("camera_localhost_only")))
    except ValueError as e:
        node.get_logger().error(str(e))
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"dual_cut_ui: {e}")
    cmd = [c for c in (gp("detect_command") or []) if c != ""]
    try:
        placement_limits = PlacementLimits.from_json(str(gp("placement_limits")))      # info-only thresholds; "" = defaults
    except ValueError as e:
        node.get_logger().error(f"bad placement_limits parameter: {e}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"dual_cut_ui: bad placement_limits parameter: {e}")
    try:
        cfg = Config(placement_limits=placement_limits, snapshot_file=os.path.expanduser(str(gp("snapshot_file"))), detect_command=cmd or None,
                     detect_cwd=os.path.expanduser(str(gp("detect_cwd"))),
                     detect_output_dir=os.path.expanduser(str(gp("detect_output_dir"))), detect_domain_id=domains.detect_domain,
                     detect_localhost_only=camera_localhost, detect_mode=str(gp("detect_mode")).strip(),
                     detect_action=str(gp("detect_action")).strip() or DEFAULT_DETECT_ACTION,
                     detect_timeout_s=float(gp("detect_timeout_s")), schedule_mode=str(gp("schedule_mode")).strip(),
                     coordinator="/" + str(gp("coordinator")).strip("/"),
                     plan_timeout_s=float(gp("plan_timeout_s")), control_grace_s=float(gp("control_grace_s")))
    except ValueError as e:
        node.get_logger().error(str(e))
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"dual_cut_ui: {e}")
    import cv2
    cv2.setNumThreads(1)             # one decode/encode at a time per worker; extra OpenCV threads only add CPU overhead
    bridge = RosBridge(node, str(gp("coordinator")), {"left": "/arm_left_trajectory_controller",
                                                      "right": "/arm_right_trajectory_controller"})
    camera_ctx = []

    def make_camera_source(domain):                 # a second rclpy context, used ONLY for the camera subscriptions
        camera_ctx.append(CameraContext(domain, args, localhost_only=camera_localhost))
        return camera_ctx[0].source
    camera_source = select_camera_source(domains, bridge, make_camera_source)
    for line in (startup_message(domains, node_domain),
                 (f"detector runs with ROS_DOMAIN_ID {domains.detect_domain}" if domains.detect_domain is not None
                  else "detector inherits the ROS environment") if cfg.detect_mode == "command"
                 else f"detector action is called in the UI's own ROS domain {node_domain}",
                 (f"camera context{' and detector' if cfg.detect_mode == 'command' else ''} use ROS_LOCALHOST_ONLY {camera_localhost}"
                  if camera_localhost is not None
                  else f"camera context{' and detector' if cfg.detect_mode == 'command' else ''} inherit ROS_LOCALHOST_ONLY")):
        LOG.info(line)
        node.get_logger().info(line)
    if cfg.detect_mode == "action":
        why = bridge.detect_import_error()           # lazy import of compal_box_msgs.action.DetectBox, once, at start
        line = (f"Detect uses the action {cfg.detect_action} (timeout {cfg.detect_timeout_s:.0f} s)" if why is None
                else f"Detect will FAIL until this is fixed: {why}")
        (LOG.info if why is None else LOG.error)(line)
        (node.get_logger().info if why is None else node.get_logger().error)(line)
    else:
        LOG.info("Detect runs the command %s", " ".join(cfg.command()))
    LOG.info("Plan sets the coordinator's schedule_mode to %s first" % cfg.schedule_mode if cfg.schedule_mode
             else "Plan leaves the coordinator's schedule_mode alone")
    core = UiCore(bridge, cfg)
    hub = CameraHub(camera_source, cameras, small_width=int(gp("camera_small_width")), large_width=int(gp("camera_large_width")),
                    quality=int(gp("camera_jpeg_quality")), idle_s=float(gp("camera_idle_s")),
                    default_off=parse_camera_off(str(gp("cameras_off")), cameras))
    auth = Auth(port)
    ctx = Context(core, hub, auth)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, name="ros-executor", daemon=True)
    ros_thread.start()
    core.start()

    async def serve():
        app = make_app(ctx)
        server = app.listen(port, address="127.0.0.1")
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        tornado.ioloop.PeriodicCallback(lambda: ctx.image_pool.submit(hub.sweep), 1000).start()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, stopped.set)
        if stop_requested.is_set():
            stopped.set()
        node.get_logger().info(f"dual-cut UI listening on 127.0.0.1:{port}")
        node.get_logger().info(f"open {auth.url()}")
        print(f"dual-cut UI: {auth.url()}", flush=True)
        await stopped.wait()
        server.stop()

    def node_log(msg):
        LOG.warning(msg)
        try:
            node.get_logger().warning(msg)         # also lands in ~/.ros/log, independent of a closed stderr pipe
        except Exception:  # noqa: BLE001
            pass

    try:
        asyncio.run(serve())
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):      # a second signal must not cut the cleanup short
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        # the gates must be closed while the executor still runs (the parameter call needs it), so ROS goes down last
        run_shutdown(shutdown_steps(core, hub, ctx, camera_ctx[0].teardown if camera_ctx else None), node_log)
        try:
            executor.shutdown()
            ros_thread.join(timeout=1.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        flush_logs()


if __name__ == "__main__":
    main()
