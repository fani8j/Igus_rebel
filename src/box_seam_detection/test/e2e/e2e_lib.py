"""Helpers for the box_seam_detection end-to-end check (node + replayed capture, no robot, no camera driver).

Everything runs on an isolated ROS domain (default 87, ROS_LOCALHOST_ONLY=1); processes are started in their own
session and stopped by process group (never by name). The snapshot/output paths always point into a scratch dir.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

DOMAIN_ID = os.environ.get("BOX_SEAM_E2E_DOMAIN", "87")
CAPTURES = Path(os.environ.get("BOX_SEAM_E2E_CAPTURES", "/home/farhan/Igus_rebel/outputs/rgbd_captures/captures"))
REAL_SNAPSHOT = Path("/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json")  # never written by the e2e check
ACTION = "/box_seam_detector/detect_box"

POINT_KEYS = ("seam_start", "seam_end", "seam_center", "back_left", "back_right", "front_left", "front_right",
              "back_middle_left", "front_middle_left", "back_middle_right", "front_middle_right")


def isolated_env() -> dict:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = DOMAIN_ID
    env["ROS_LOCALHOST_ONLY"] = "1"
    # SHM-only FastDDS profile (see the xml): loopback UDP drops most full-HD image messages
    env.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(Path(__file__).with_name("fastdds_shm_only.xml")))
    return env


# --------------------------------------------------------------------------------------------- executor parser port
def executor_snapshot_problems(doc: dict, *, faces=("left", "right", "back", "front")) -> list[str]:
    """Python port of box_h_cut_executor.cpp parseSnapshotFromDocument + validateSnapshot + the side-seed parse
    (~lines 5490-5930). Freshness is not checked here (the caller compares stamps). Empty list = executor would parse."""
    problems: list[str] = []

    def s(name):
        v = doc.get(name)
        return isinstance(v, str) and bool(v)

    def pt(d, name):
        c = d.get(name) if isinstance(d, dict) else None
        if not isinstance(c, dict):
            return None
        try:
            v = np.array([float(c["x"]), float(c["y"]), float(c["z"])])
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.all(np.isfinite(v)) else None

    if doc.get("schema") != "box_h_cut_snapshot/v1" or not all(s(k) for k in ("snapshot_id", "capture_id", "frame_id", "source_frame", "source_file")):
        problems.append("schema or identity fields invalid")
    stamp = doc.get("acquisition_stamp_ns")
    if not isinstance(stamp, int) or stamp <= 0:
        problems.append("acquisition_stamp_ns invalid")
    rms = doc.get("plane_rms_m")
    if not isinstance(rms, (int, float)) or not math.isfinite(rms) or rms < 0:
        problems.append("plane_rms_m invalid")
    pts = {k: pt(doc, k) for k in POINT_KEYS}
    normal = pt(doc, "surface_normal")
    bad = [k for k, v in pts.items() if v is None] + (["surface_normal"] if normal is None else [])
    if bad:
        return problems + [f"required geometry invalid: {bad}"]
    if doc.get("frame_id") != "base_link":
        problems.append(f"frame_id {doc.get('frame_id')} != base_link")
    if abs(np.linalg.norm(normal) - 1.0) > 0.05:
        problems.append("surface_normal not unit")
    bl, br, fl, fr = (pts[k] for k in ("back_left", "back_right", "front_left", "front_right"))
    back, left, right, front = br - bl, fl - bl, fr - br, fr - fl
    width, depth = np.linalg.norm(back), np.linalg.norm(left)
    if min(width, depth, np.linalg.norm(right), np.linalg.norm(front)) < 0.04:
        return problems + ["carton edges degenerate"]
    if abs(left @ right / (depth * np.linalg.norm(right))) < 0.95 or abs(back @ front / (width * np.linalg.norm(front))) < 0.95:
        problems.append("corner ordering not rectangular")

    def on_top(p):
        rel = p - bl
        x, y = rel @ back / width, rel @ left / depth
        return -0.01 <= x <= width + 0.01 and -0.01 <= y <= depth + 0.01 and abs(rel @ normal) <= 0.01

    for k in ("seam_start", "seam_end", "seam_center", "back_middle_left", "front_middle_left", "back_middle_right", "front_middle_right"):
        if not on_top(pts[k]):
            problems.append(f"{k} outside the carton top face")
    for k, a, b in (("back_middle_left", "back_left", "seam_start"), ("front_middle_left", "front_left", "seam_start"),
                    ("back_middle_right", "back_right", "seam_end"), ("front_middle_right", "front_right", "seam_end")):
        if np.max(np.abs(pts[k] - (pts[a] + pts[b]) / 2)) > 1e-6:
            problems.append(f"{k} is not the midpoint of {a},{b}")
    seed = doc.get("side_tape_seed")
    if not isinstance(seed, dict):
        return problems + ["no side_tape_seed block"]
    if seed.get("schema") not in ("side_tape_seed/v1", "side_tape_seed/v1.1"):
        problems.append("side_tape_seed schema")
    if seed.get("frame_id") != "base_link" or seed.get("units") != "m":
        problems.append("side_tape_seed frame/units")
    src = seed.get("source") or {}
    if src.get("snapshot_id") != doc.get("snapshot_id") or src.get("acquisition_stamp_ns") != stamp:
        problems.append("side_tape_seed.source does not match snapshot_id/stamp")
    for face in faces:
        fn = (seed.get("faces") or {}).get(face)
        if not isinstance(fn, dict):
            problems.append(f"seed face {face} missing")
            continue
        pl = fn.get("plane")
        if pt(pl, "point") is None or pt(pl, "normal") is None or np.linalg.norm(pt(pl, "normal")) < 1e-6:
            problems.append(f"seed face {face} plane invalid")
        edge = fn.get("wall_top_edge") or fn.get("top_edge")
        if not isinstance(edge, dict) or pt(edge, "start") is None or pt(edge, "end") is None:
            problems.append(f"seed face {face} top edge invalid")
    return problems


def max_point_diff_mm(a: dict, b: dict, keys=POINT_KEYS + ("surface_normal",)) -> tuple[float, dict]:
    """(max over keys, per-key) euclidean difference in mm between two snapshot dicts (normal in 1e-3 units)."""
    per = {}
    for k in keys:
        pa = np.array([a[k][c] for c in "xyz"])
        pb = np.array([b[k][c] for c in "xyz"])
        per[k] = float(np.linalg.norm(pa - pb) * 1000.0)
    return max(per.values()), per


def run_cli_snapshot(capture_id: str, out_json: Path, work_dir: Path, timeout_s: float = 600) -> subprocess.CompletedProcess:
    """CLI reference: box_seam_detect --from-image on the saved capture, planner snapshot to out_json (scratch only)."""
    img = CAPTURES.parent / "rgb" / f"{capture_id}.png"
    manifest = CAPTURES / f"{capture_id}.json"
    assert out_json.resolve() != REAL_SNAPSHOT
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "box_seam_detection.live.orchestrator", "--from-image", str(img),
           "--capture-manifest", str(manifest), "--driver-extrinsic", "recorded", "--planner-output-json", str(out_json),
           "--predict-out-dir", str(work_dir), "--output-dir", str(work_dir)]
    return subprocess.run(cmd, env=isolated_env(), capture_output=True, text=True, timeout=timeout_s)


# --------------------------------------------------------------------------------------------- process management
class Proc:
    """A subprocess in its own session; stop() = SIGINT then SIGTERM to its process group only."""

    def __init__(self, cmd: list[str], log: Path):
        self.log = open(log, "w")
        self.p = subprocess.Popen(cmd, env=isolated_env(), stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.pgid = os.getpgid(self.p.pid)

    def stop(self, timeout_s: float = 10.0):
        if self.p.poll() is None:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    os.killpg(self.pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.p.wait(timeout_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.log.close()


def start_node(scratch: Path, extra_params: list[str] | None = None) -> Proc:
    from ament_index_python.packages import get_package_share_directory
    cfg = Path(get_package_share_directory("box_seam_detection")) / "config" / "box_seam_detection.yaml"
    cmd = ["ros2", "run", "box_seam_detection", "box_seam_detector_node", "--ros-args", "--params-file", str(cfg),
           "-p", f"snapshot_path:={scratch / 'snap.json'}", "-p", f"output_root:={scratch / 'out'}"]
    for p in extra_params or []:
        cmd += ["-p", p]
    return Proc(cmd, scratch / "node.log")


def start_camera(scratch: Path, capture_id: str, rate: float = 15.0) -> Proc:
    return Proc([sys.executable, "-m", "box_seam_detection.ros.fake_camera", str(CAPTURES / f"{capture_id}.json"),
                 "--rate", str(rate)], scratch / f"camera_{capture_id}.log")


# --------------------------------------------------------------------------------------------- rclpy client
class E2EClient:
    """rclpy node (spinning in a thread) that sends DetectBox goals and records feedback/status/markers/overlay."""

    def __init__(self):
        os.environ.update(isolated_env())  # same domain / SHM profile as the spawned processes (before rclpy.init)
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        from compal_box_msgs.action import DetectBox

        self.rclpy, self.DetectBox = rclpy, DetectBox
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("box_seam_e2e_client")
        self.client = ActionClient(self.node, DetectBox, ACTION)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.lock = threading.Lock()
        self.status_msgs: list[tuple[float, dict]] = []
        self.marker_msgs: list[tuple[float, object]] = []
        self.overlay_msgs: list[tuple[float, object]] = []
        self.image_stamps: set[int] = set()
        self.node.create_subscription(String, "/box_seam_detector/status", self._on_status, latched)
        self.node.create_subscription(MarkerArray, "/box_seam_detector/markers", lambda m: self._add(self.marker_msgs, m), latched)
        self.node.create_subscription(Image, "/box_seam_detector/overlay", lambda m: self._add(self.overlay_msgs, m), latched)
        self.node.create_subscription(Image, "/camera/color/image_raw", self._on_image, qos_profile_sensor_data)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _add(self, store, msg):
        with self.lock:
            store.append((time.time(), msg))

    def _on_status(self, msg):
        try:
            self._add(self.status_msgs, json.loads(msg.data))
        except ValueError:
            pass

    def _on_image(self, msg):
        with self.lock:
            self.image_stamps.add(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)

    def wait_ready(self, timeout_s: float = 90.0) -> bool:
        """Action server present and SAM3 daemon reported ready in the latched status."""
        end = time.time() + timeout_s
        while time.time() < end:
            if self.client.wait_for_server(timeout_sec=1.0):
                with self.lock:
                    st = self.status_msgs[-1][1] if self.status_msgs else None
                if st and (st.get("daemon") or {}).get("ready", True):
                    return True
        return False

    def send(self, timeout_s: float = 180.0, **goal_fields) -> dict:
        """Send one goal, block for the result. Returns accepted/status/result fields/feedback/markers/overlay/status."""
        goal = self.DetectBox.Goal()
        for k, v in goal_fields.items():
            setattr(goal, k, v)
        feedback: list[dict] = []
        with self.lock:
            n_status, n_mark, n_over = len(self.status_msgs), len(self.marker_msgs), len(self.overlay_msgs)
            stamps_before = set(self.image_stamps)
        t0 = time.time()
        fut = self.client.send_goal_async(goal, feedback_callback=lambda fb: feedback.append(
            {"stage": fb.feedback.stage, "attempt": fb.feedback.attempt, "message": fb.feedback.message, "t": time.time() - t0}))
        out = self._wait(fut, 30.0)
        if out is None:
            return {"accepted": False, "error": "no goal response"}
        handle = out
        if not handle.accepted:
            return {"accepted": False, "feedback": feedback, "elapsed_s": time.time() - t0}
        rfut = handle.get_result_async()
        res = self._wait(rfut, timeout_s)
        if res is None:
            return {"accepted": True, "error": "result timeout", "feedback": feedback, "handle": handle}
        time.sleep(0.5)  # let the status / marker / overlay publications land
        r = res.result
        with self.lock:
            statuses = [m for _, m in self.status_msgs[n_status:]]
            markers = [m for _, m in self.marker_msgs[n_mark:]]
            overlays = [m for _, m in self.overlay_msgs[n_over:]]
            stamps = set(self.image_stamps)
        return {
            "accepted": True, "goal_status": int(res.status), "elapsed_s": time.time() - t0, "feedback": feedback,
            "result": {f: getattr(r, f) if not isinstance(getattr(r, f), (list, tuple)) else list(getattr(r, f))
                       for f in ("success", "status", "reason", "reasons", "cut_permission", "snapshot_id", "snapshot_path",
                                 "capture_id", "seam_status", "attempts", "elapsed_s")},
            "statuses": statuses, "markers": markers, "overlays": overlays, "image_stamps": stamps,
            "new_image_stamps": stamps - stamps_before,
        }

    def _wait(self, fut, timeout_s):
        end = time.time() + timeout_s
        while time.time() < end:
            if fut.done():
                return fut.result()
            time.sleep(0.02)
        return None

    def close(self):
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


GOAL_SUCCEEDED = 4
