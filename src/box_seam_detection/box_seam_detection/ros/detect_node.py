"""DetectBox action server: capture a fresh RGB-D frame set, run the box/seam detector, publish the result.

The node never commands the robot. It only (a) captures from the scene camera, (b) calls
``box_seam_detection.live.orchestrator.run_detection`` (which owns the fail-closed snapshot file rules) and
(c) publishes status / markers / overlay. ``live.orchestrator`` and ``core.sam3_daemon`` are imported lazily, so
the node module imports (and tests can inject a fake) without them. See DESIGN.md and config/box_seam_detection.yaml.
"""

from __future__ import annotations

import os

# Must run before anything imports numpy (sensor_msgs does): same BLAS pin as the CLI detector.
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ.setdefault("AUTOLABEL_V2_BLAS_PINNED", "1")

import dataclasses
import importlib
import inspect
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from compal_box_msgs.action import DetectBox

from . import markers as markers_mod
from .capture import CaptureAborted, SceneCapture

DEFAULT_SNAPSHOT = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"
DEFAULT_OUTPUT_ROOT = "/home/farhan/Igus_rebel/outputs/rgbd_captures"
DEFAULT_SAM3_PYTHON = "/home/farhan/miniconda3/envs/book_grasping/bin/python"
DEFAULT_CALIB = "~/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class DetectionCancelled(RuntimeError):
    pass


def _import_orchestrator():
    return importlib.import_module("box_seam_detection.live.orchestrator")


def _cut_permission_text(cp) -> str:
    """allowed | blocked | overridden (blocked but allow_cut_blocked used) | n/a."""
    if not isinstance(cp, dict):
        return "n/a"
    if cp.get("override") or cp.get("overridden"):
        return "overridden"
    return str(cp.get("status", "n/a"))


def _result_summary(res: Any) -> dict:
    cp = getattr(res, "cut_permission", None)
    snap = getattr(res, "snapshot", None) or {}
    return {
        "success": bool(res.success), "status": str(res.status), "reason": str(res.reason or ""),
        "reasons": [str(r) for r in (res.reasons or [])],
        "cut_permission": _cut_permission_text(cp),
        "snapshot_id": str(snap.get("snapshot_id", "")) if isinstance(snap, dict) else "",
        "snapshot_path": str(res.snapshot_path or ""),
        "capture_id": str(res.capture_id or ""), "seam_status": str(res.seam_status or ""),
        "attempts": len(res.attempts or []), "elapsed_s": float(res.elapsed_s or 0.0),
        "finished_at_ns": time.time_ns(),
    }


class DetectNode(Node):
    # (parameter name, default); None-valued defaults are resolved in __init__
    DETECT_PARAMS = {
        "max_attempts": 4, "retry_policy": "smart", "allow_needs_review": False, "allow_cut_blocked": False,
        "dry_run": False, "rect_tol_mm": -1.0, "sam3_python": os.environ.get("BOX_SEAM_SAM3_PYTHON", DEFAULT_SAM3_PYTHON),
        "sam3_daemon": "auto", "downstream_inprocess": True, "calib_file": DEFAULT_CALIB,
        "driver_extrinsic": "recorded", "target_frame": "base_link", "side_seed": True,
        "output_root": DEFAULT_OUTPUT_ROOT, "snapshot_path": DEFAULT_SNAPSHOT,
        "scene_config": "",     # "" = table_cell_v1 (today's values); path or shipped name e.g. dual_arm_v1
    }

    def __init__(self, *, orchestrator=None, node_name: str = "box_seam_detector", **kwargs):
        super().__init__(node_name, **kwargs)
        self._orch_override = orchestrator
        p = self.declare_parameter
        for k, v in self.DETECT_PARAMS.items():
            p(k, v)
        p("capture_source", "node")  # node | recorder
        p("color_topic", "/camera/color/image_raw")
        p("depth_topic", "/camera/depth/image_raw")
        p("camera_info_topic", "/camera/color/camera_info")
        p("sync_slop_s", 0.04)
        p("sync_queue_size", 10)
        p("capture_margin_s", 0.0)
        p("capture_timeout_s", 10.0)
        p("driver_parent_frame", "camera_link")
        p("extrinsic_required", False)
        p("use_tf", True)
        p("warm_daemon", True)
        p("overlay_glob", "autolabel_v2_overlays/*_autolabel_v2.png")
        p("status_period_s", 2.0)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self._gp = gp
        self.capture_source = str(gp("capture_source"))
        if self.capture_source not in ("node", "recorder"):
            raise ValueError("capture_source must be 'node' or 'recorder'")
        self.output_root = Path(str(gp("output_root"))).expanduser()
        sp = str(gp("snapshot_path")).strip()
        self.snapshot_path: Path | None = Path(sp).expanduser() if sp else None  # empty = never publish

        for name in ("sync_slop_s", "capture_timeout_s", "status_period_s"):
            if float(gp(name)) <= 0:
                raise ValueError(f"parameter {name} must be > 0")
        if int(gp("sync_queue_size")) < 1:
            raise ValueError("parameter sync_queue_size must be >= 1")
        if float(gp("capture_margin_s")) < 0:
            raise ValueError("parameter capture_margin_s must be >= 0")
        if bool(self.get_parameter("use_sim_time").value):
            self.get_logger().warn("use_sim_time=true: capture freshness uses the node (sim) clock; camera stamps "
                                   "must be on the same clock or every capture times out")
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._cancel = False
        self._progress = {"stage": "idle", "attempt": 0, "message": ""}
        self._last_result: dict | None = None
        self._daemon: dict = {"ready": False, "state": "not_started", "error": ""}
        self._goal_handle = None

        self.sub_group = MutuallyExclusiveCallbackGroup()
        self.action_group = ReentrantCallbackGroup()
        self.capture: SceneCapture | None = None
        if self.capture_source == "node":
            self.capture = SceneCapture(
                self, color_topic=gp("color_topic"), depth_topic=gp("depth_topic"),
                info_topic=gp("camera_info_topic"), output_root=self.output_root,
                sync_slop_s=float(gp("sync_slop_s")), queue_size=int(gp("sync_queue_size")),
                margin_s=float(gp("capture_margin_s")), driver_parent_frame=gp("driver_parent_frame"),
                use_tf=bool(gp("use_tf")), extrinsic_required=bool(gp("extrinsic_required")),
                files_cls=self._files_cls(), callback_group=self.sub_group)

        self.status_pub = self.create_publisher(String, "~/status", LATCHED)
        self.markers_pub = self.create_publisher(MarkerArray, "~/markers", LATCHED)
        self.overlay_pub = self.create_publisher(Image, "~/overlay", LATCHED)
        self._action = ActionServer(
            self, DetectBox, "~/detect_box", execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=self._on_cancel, callback_group=self.action_group)
        self.create_timer(max(float(gp("status_period_s")), 0.2), self.publish_status,
                          callback_group=self.sub_group)
        self.publish_status()
        if bool(gp("warm_daemon")) and str(gp("sam3_daemon")) != "off":
            threading.Thread(target=self._warm_daemon, name="sam3-warmup", daemon=True).start()
        else:
            self._daemon = {"ready": False, "state": "disabled", "error": ""}
        self.get_logger().info(
            f"box_seam_detector ready: action ~/detect_box, capture_source={self.capture_source}, snapshot={self.snapshot_path or '(none: not published)'}")

    # ------------------------------------------------------------------ helpers
    def _files_cls(self):
        try:
            return _import_orchestrator().CaptureFiles if self._orch_override is None else self._orch_override.CaptureFiles
        except Exception:  # live.orchestrator not available yet -> local dataclass with the same fields
            return None

    def _orch(self):
        return self._orch_override if self._orch_override is not None else _import_orchestrator()

    def _warm_daemon(self):
        self._set_daemon(state="starting")
        try:
            mod = importlib.import_module("box_seam_detection.core.sam3_daemon")
        except Exception as exc:  # core not installed yet: log and continue (detector spawns lazily)
            self.get_logger().warn(f"SAM3 daemon warm-up skipped (core.sam3_daemon unavailable: {exc})")
            self._set_daemon(state="unavailable", error=str(exc))
            return
        try:
            info = mod.ensure_daemon(str(self._gp("sam3_python")))
            self.get_logger().info(f"SAM3 daemon ready (spawned={info.get('spawned')})")
            self._set_daemon(ready=True, state="ready")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"SAM3 daemon warm-up failed: {exc}")
            self._set_daemon(state="failed", error=str(exc))

    def _set_daemon(self, ready=False, state="", error=""):
        with self._lock:
            self._daemon = {"ready": ready, "state": state, "error": error}
        self.publish_status()

    def build_config(self, goal=None):
        """DetectConfig from the parameters (+ the goal's overrides: max_attempts>0 and the OR of the bool flags)."""
        orch = self._orch()
        vals = {k: self._gp(k) for k in self.DETECT_PARAMS}
        vals["rect_tol_mm"] = None if float(vals["rect_tol_mm"]) < 0 else float(vals["rect_tol_mm"])
        scene = str(vals.get("scene_config") or "").strip()
        vals["scene_config"] = scene or None
        if scene and str(vals["calib_file"]) == str(DEFAULT_CALIB):
            # the scene's calib file is the default when the calib_file parameter is left alone
            from ..core import scene as _scene  # noqa: PLC0415
            vals["calib_file"] = _scene.load_scene(scene).calib_file
        vals["output_root"] = self.output_root
        vals["snapshot_path"] = self.snapshot_path  # None = no publish
        if goal is not None:
            if int(goal.max_attempts) > 0:
                vals["max_attempts"] = int(goal.max_attempts)
            for f in ("allow_needs_review", "allow_cut_blocked", "dry_run"):
                vals[f] = bool(vals[f]) or bool(getattr(goal, f))
        names = {f.name for f in dataclasses.fields(orch.DetectConfig)}
        dropped = sorted(set(vals) - names)
        if dropped:
            self.get_logger().warn(f"DetectConfig has no field(s) {dropped}; parameters ignored", once=True)
        return orch.DetectConfig(**{k: v for k, v in vals.items() if k in names})

    # ------------------------------------------------------------------ status / publishing
    def _status_dict(self) -> dict:
        with self._lock:
            d = {"state": "busy" if self._busy else "idle", "progress": dict(self._progress),
                 "last_result": self._last_result, "daemon": dict(self._daemon),
                 "capture_source": self.capture_source, "snapshot_path": str(self.snapshot_path or "")}
        d["camera"] = self.capture.diagnostics() if self.capture is not None else None
        d["stamp_ns"] = time.time_ns()
        return d

    def publish_status(self):
        try:
            self.status_pub.publish(String(data=json.dumps(self._status_dict())))
        except Exception:  # context shut down
            pass

    def _publish_outputs(self, res: Any):
        snap = getattr(res, "snapshot", None)
        if res.success and isinstance(snap, dict) and snap:
            self.markers_pub.publish(markers_mod.snapshot_markers(snap, self.get_clock().now().to_msg()))
        overlay = getattr(res, "overlay_path", None)
        pdir = getattr(res, "pipeline_dir", None)
        if overlay and Path(overlay).is_file():
            self._publish_overlay(Path(overlay).parent, Path(overlay))
        elif pdir:
            self._publish_overlay(Path(pdir))

    def _publish_overlay(self, pdir: Path, path: Path | None = None):
        try:
            import cv2
            files = ([path] if path is not None else
                     sorted(pdir.glob("**/" + str(self._gp("overlay_glob"))), key=lambda f: f.stat().st_mtime))
            if not files:
                self.overlay_pub.publish(self._empty_overlay())
                return
            img = cv2.imread(str(files[-1]), cv2.IMREAD_COLOR)
            if img is None:
                return
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "box_seam_overlay"
            msg.height, msg.width, msg.encoding = int(img.shape[0]), int(img.shape[1]), "bgr8"
            msg.step = int(img.shape[1] * 3)
            msg.data = img.tobytes()
            self.overlay_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001 - the overlay is cosmetic
            self.get_logger().warn(f"overlay publish failed: {exc}")

    # ------------------------------------------------------------------ action
    def _on_goal(self, goal):
        with self._lock:
            if self._busy:
                self.get_logger().warn("DetectBox goal rejected: a detection is already running")
                return GoalResponse.REJECT
            self._busy = True
            self._cancel = False
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        with self._lock:
            self._cancel = True
        return CancelResponse.ACCEPT

    def _aborting(self) -> bool:
        """abort_check for run_detection: goal cancel requested or node shutting down."""
        return bool(self._cancel) or self._shutdown.is_set()

    @staticmethod
    def _empty_overlay():
        m = Image()
        m.encoding = "bgr8"  # 0x0 image = cleared
        return m

    def destroy_node(self):
        self._shutdown.set()
        return super().destroy_node()

    def _set_progress(self, stage: str, attempt: int, message: str, goal_handle=None):
        with self._lock:
            self._progress = {"stage": stage, "attempt": int(attempt), "message": message}
        gh = goal_handle or self._goal_handle
        if gh is not None:
            fb = DetectBox.Feedback()
            fb.stage, fb.attempt, fb.message = str(stage), int(attempt), str(message)
            try:
                gh.publish_feedback(fb)
            except Exception:  # noqa: BLE001
                pass
        self.publish_status()

    def _make_next_capture(self) -> Callable[[int], Any]:
        timeout = float(self._gp("capture_timeout_s"))
        if self.capture_source == "node":
            def next_capture(attempt: int):
                if self._cancel:
                    raise DetectionCancelled("cancelled before capture")
                self._set_progress("capture", attempt, "waiting for a fresh frame")
                files = self.capture.capture(after_ns=self.capture.now_ns(), timeout_s=timeout,
                                             abort=self._aborting)
                self._set_progress("capture", attempt, f"captured {files.capture_id}")
                return files
            return next_capture
        # recorder source: the live builder provides box_seam_detection.live.recorder.capture_once(
        #   output_root: Path, attempt: int, timeout_s: float) -> CaptureFiles  (today's rgbd_recorder path).
        rec = importlib.import_module("box_seam_detection.live.recorder")
        capture_once = getattr(rec, "capture_once", None)
        if capture_once is None:
            raise NotImplementedError("capture_source=recorder needs live.recorder.capture_once(output_root, attempt, timeout_s)")

        def next_capture(attempt: int):
            if self._cancel:
                raise DetectionCancelled("cancelled before capture")
            self._set_progress("capture", attempt, "recorder capture")
            return capture_once(self.output_root, attempt, timeout)
        return next_capture

    def _fail_closed_snapshot_removal(self, dry_run: bool):
        if dry_run or self.snapshot_path is None:
            return
        try:
            self.snapshot_path.unlink()
            self.get_logger().warn(f"removed snapshot {self.snapshot_path} after a failed run")
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.get_logger().error(f"could not remove snapshot {self.snapshot_path}: {exc}")

    def _execute(self, goal_handle):
        result = DetectBox.Result()
        self._goal_handle = goal_handle
        res = None
        error = ""
        cancelled = False
        cancelled_exc = False
        eff_dry = bool(goal_handle.request.dry_run) or bool(self._gp("dry_run"))
        self.overlay_pub.publish(self._empty_overlay())  # never show an old box as current
        try:
            self.markers_pub.publish(markers_mod.clear_markers())
            cfg = self.build_config(goal_handle.request)
            self._set_progress("detect", 0, "starting", goal_handle)

            def progress(stage: str, attempt: int, message: str):
                self._set_progress(stage, attempt, message, goal_handle)

            kwargs = {}
            run = self._orch().run_detection
            if "abort_check" in inspect.signature(run).parameters:
                kwargs["abort_check"] = self._aborting
            res = run(self._make_next_capture(), cfg, progress, **kwargs)
        except (DetectionCancelled, CaptureAborted) as exc:
            error = f"cancelled: {exc}"
            cancelled_exc = True
            self._fail_closed_snapshot_removal(eff_dry)
        except Exception as exc:  # noqa: BLE001 - every failure is reported as status=error
            error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"detection failed: {error}")
            self._fail_closed_snapshot_removal(eff_dry)
        try:
            if res is not None:
                summary = _result_summary(res)
                if not self._shutdown.is_set():
                    self._publish_outputs(res)
            else:
                summary = {"success": False, "status": "cancelled" if cancelled_exc else "error", "reason": error,
                           "reasons": ["cancelled"] if cancelled_exc else ["node_exception"],
                           "cut_permission": "n/a", "snapshot_id": "", "snapshot_path": "", "capture_id": "",
                           "seam_status": "", "attempts": 0, "elapsed_s": 0.0, "finished_at_ns": time.time_ns()}
            result.success = bool(summary["success"])
            result.status = summary["status"]
            result.reason = summary["reason"]
            result.reasons = summary["reasons"]
            result.cut_permission = summary["cut_permission"]
            result.snapshot_id = summary["snapshot_id"]
            result.snapshot_path = summary["snapshot_path"]
            result.capture_id = summary["capture_id"]
            result.seam_status = summary["seam_status"]
            result.attempts = int(summary["attempts"])
            result.elapsed_s = float(summary["elapsed_s"])
            with self._lock:
                self._last_result = summary
                self._progress = {"stage": "idle", "attempt": 0, "message": ""}
                # a cancel that arrives after the snapshot was published leaves the goal SUCCEEDED
                cancelled = (summary["status"] == "cancelled" or (self._cancel and not summary["success"]))
        finally:
            with self._lock:
                self._busy = False
                self._cancel = False
            self._goal_handle = None
            self.publish_status()
        if cancelled and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif cancelled:
            goal_handle.abort()  # aborted by node shutdown, not by a client cancel
        elif result.success or result.status in ("accepted", "rejected", "blocked"):
            goal_handle.succeed()  # the action completed; result.success says whether a snapshot was produced
        else:
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._shutdown.set()  # abort_check -> the worker stops at its next check
        t0 = time.monotonic()
        while node._busy and time.monotonic() - t0 < 10.0:
            time.sleep(0.1)
        stuck = node._busy
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if stuck:  # a detection is still inside a long call: do not keep the process alive for it
            os._exit(0)


if __name__ == "__main__":
    main()
