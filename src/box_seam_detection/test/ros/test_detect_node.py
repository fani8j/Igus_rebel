import dataclasses
import json
import threading
import time
import types
from pathlib import Path

import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from compal_box_msgs.action import DetectBox
from box_seam_detection.ros.detect_node import DetectNode

from .synthetic import SyntheticCamera

SNAP = {"schema": "box_h_cut_snapshot/v1", "snapshot_id": "capture_x", "frame_id": "base_link",
        "seam_start": {"x": 0.1, "y": 0.0, "z": 0.09}, "seam_end": {"x": 0.2, "y": 0.0, "z": 0.09},
        "seam_center": {"x": 0.15, "y": 0.0, "z": 0.09},
        **{c: {"x": 0.1 * i, "y": 0.1, "z": 0.09} for i, c in
           enumerate(("back_left", "back_right", "front_right", "front_left"))}}


def make_fake_orchestrator():
    @dataclasses.dataclass
    class CaptureFiles:
        capture_id: str
        image_path: Path
        pcd_path: Path
        manifest_path: Path
        stamp_ns: int

    @dataclasses.dataclass
    class DetectConfig:
        max_attempts: int = 4
        retry_policy: str = "smart"
        allow_needs_review: bool = False
        allow_cut_blocked: bool = False
        dry_run: bool = False
        rect_tol_mm: float | None = None
        sam3_python: str = ""
        sam3_daemon: str = "auto"
        downstream_inprocess: bool = True
        calib_file: str = ""
        driver_extrinsic: str = "recorded"
        target_frame: str = "base_link"
        side_seed: bool = True
        output_root: Path = Path(".")
        snapshot_path: Path = Path("snap.json")

    @dataclasses.dataclass
    class DetectResult:
        success: bool = True
        status: str = "accepted"
        reason: str = "ok"
        reasons: list = dataclasses.field(default_factory=list)
        cut_permission: dict | None = dataclasses.field(default_factory=lambda: {"status": "allowed"})
        snapshot: dict | None = dataclasses.field(default_factory=lambda: dict(SNAP))
        snapshot_path: str | None = "/tmp/snap.json"
        capture_id: str | None = "cid"
        seam_status: str | None = "ok"
        attempts: list = dataclasses.field(default_factory=lambda: [{"attempt": 1}])
        elapsed_s: float = 0.5
        pipeline_dir: str | None = None

    orch = types.SimpleNamespace(CaptureFiles=CaptureFiles, DetectConfig=DetectConfig, DetectResult=DetectResult)
    orch.calls = []
    orch.behaviour = lambda next_capture, cfg, progress: DetectResult()

    def run_detection(next_capture, cfg, progress=None):
        orch.calls.append(cfg)
        return orch.behaviour(next_capture, cfg, progress)

    orch.run_detection = run_detection
    return orch


class Harness:
    def __init__(self, tmp_path, spinner, camera=True, **params):
        self.orch = make_fake_orchestrator()
        self.snap = tmp_path / "seam.json"
        self.node = DetectNode(
            orchestrator=self.orch,
            parameter_overrides=[rclpy.parameter.Parameter(k, value=v) for k, v in {
                "snapshot_path": str(self.snap), "output_root": str(tmp_path / "out"), "use_tf": False,
                "warm_daemon": False, "capture_timeout_s": 3.0, "status_period_s": 0.5, **params}.items()])
        self.client_node = Node("test_client")
        self.client = ActionClient(self.client_node, DetectBox, "/box_seam_detector/detect_box")
        self.status = []
        self.markers = []
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.client_node.create_subscription(String, "/box_seam_detector/status",
                                             lambda m: self.status.append(json.loads(m.data)), latched)
        self.client_node.create_subscription(MarkerArray, "/box_seam_detector/markers",
                                             lambda m: self.markers.append(m), latched)
        nodes = [self.node, self.client_node]
        if camera:
            nodes.append(SyntheticCamera())
        spinner(*nodes)
        assert self.client.wait_for_server(timeout_sec=10.0)

    def send(self, **goal_kw):
        fb = []
        goal = DetectBox.Goal(**goal_kw)
        fut = self.client.send_goal_async(goal, feedback_callback=lambda m: fb.append(m.feedback))
        return fut, fb

    def run(self, **goal_kw):
        fut, fb = self.send(**goal_kw)
        gh = self._wait(fut)
        if not gh.accepted:
            return gh, None, fb
        res = self._wait(gh.get_result_async())
        return gh, res, fb

    @staticmethod
    def _wait(fut, timeout=20.0):
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError
            time.sleep(0.02)
        return fut.result()


@pytest.fixture
def harness(tmp_path, spinner):
    return Harness(tmp_path, spinner)


def test_accepted_result_feedback_status_markers(harness):
    def behaviour(next_capture, cfg, progress):
        progress("capture", 1, "grab")
        files = next_capture(1)
        assert files.manifest_path.exists() and files.capture_id
        progress("detect", 1, "running")
        r = harness.orch.DetectResult(capture_id=files.capture_id)
        return r
    harness.orch.behaviour = behaviour
    gh, res, fb = harness.run(max_attempts=2)
    assert gh.accepted and res.result.success and res.result.status == "accepted"
    assert res.result.cut_permission == "allowed" and res.result.snapshot_id == "capture_x"
    assert res.result.attempts == 1 and res.result.capture_id.startswith("frame_")
    stages = [f.stage for f in fb]
    assert "capture" in stages and "detect" in stages
    assert harness.orch.calls[0].max_attempts == 2
    assert harness.orch.calls[0].snapshot_path == harness.snap
    time.sleep(1.0)
    last = harness.status[-1]
    assert last["state"] == "idle" and last["last_result"]["status"] == "accepted"
    assert last["daemon"]["state"] == "disabled" and last["camera"]["counts"]["synced"] > 0
    names = {m.ns for arr in harness.markers for m in arr.markers}
    assert {"box_seam/seam_start", "box_seam/seam_line", "box_seam/back_left", "box_seam/top_outline"} <= names


@pytest.mark.parametrize("status,reason", [("rejected", "no_box"), ("blocked", "cut_blocked")])
def test_rejected_and_blocked_report_failure(harness, status, reason):
    harness.orch.behaviour = lambda nc, cfg, pr: harness.orch.DetectResult(
        success=False, status=status, reason=reason, reasons=[reason], snapshot=None, snapshot_path=None,
        cut_permission={"status": "blocked"} if status == "blocked" else None)
    gh, res, _ = harness.run()
    assert gh.accepted and not res.result.success and res.result.status == status
    assert list(res.result.reasons) == [reason]
    assert res.result.cut_permission == ("blocked" if status == "blocked" else "n/a")
    assert res.result.snapshot_path == ""


def test_error_exception_removes_stale_snapshot(harness):
    harness.snap.write_text("{}")

    def boom(*a):
        raise RuntimeError("detector exploded")
    harness.orch.behaviour = boom
    gh, res, _ = harness.run()
    assert not res.result.success and res.result.status == "error"
    assert "detector exploded" in res.result.reason
    assert not harness.snap.exists()  # fail closed
    # the node is usable again
    harness.orch.behaviour = lambda nc, cfg, pr: harness.orch.DetectResult()
    assert harness.run()[1].result.success


def test_dry_run_exception_keeps_snapshot(harness):
    harness.snap.write_text("{}")

    def boom(*a):
        raise RuntimeError("x")
    harness.orch.behaviour = boom
    _, res, _ = harness.run(dry_run=True)
    assert res.result.status == "error" and harness.snap.exists()
    assert harness.orch.calls[0].dry_run is True


def test_busy_goal_rejected_while_running(harness):
    release = threading.Event()
    started = threading.Event()

    def slow(nc, cfg, pr):
        started.set()
        release.wait(10)
        return harness.orch.DetectResult()
    harness.orch.behaviour = slow
    fut1, _ = harness.send()
    gh1 = harness._wait(fut1)
    assert gh1.accepted and started.wait(5)
    gh2 = harness._wait(harness.send()[0])
    assert not gh2.accepted
    time.sleep(0.7)
    assert harness.status[-1]["state"] == "busy"
    release.set()
    assert harness._wait(gh1.get_result_async()).result.success
    # accepted again after completion
    harness.orch.behaviour = lambda nc, cfg, pr: harness.orch.DetectResult()
    assert harness.run()[0].accepted


def test_capture_timeout_surfaces_as_error(tmp_path, spinner):
    h = Harness(tmp_path, spinner, camera=False, capture_timeout_s=0.5)
    h.orch.behaviour = lambda nc, cfg, pr: (nc(1), h.orch.DetectResult())[1]
    _, res, _ = h.run()
    assert res.result.status == "error" and "no fresh synced" in res.result.reason


def test_goal_flags_and_param_overrides(tmp_path, spinner):
    h = Harness(tmp_path, spinner, max_attempts=3, rect_tol_mm=2.5, allow_cut_blocked=True)
    h.run()
    c = h.orch.calls[0]
    assert (c.max_attempts, c.rect_tol_mm, c.allow_cut_blocked, c.allow_needs_review) == (3, 2.5, True, False)
    h.run(allow_needs_review=True, max_attempts=7)
    c = h.orch.calls[1]
    assert (c.max_attempts, c.allow_needs_review) == (7, True)


def test_recorder_source_without_live_recorder_reports_error(tmp_path, spinner):
    h = Harness(tmp_path, spinner, camera=False, capture_source="recorder")
    h.orch.behaviour = lambda nc, cfg, pr: (nc(1), h.orch.DetectResult())[1]
    _, res, _ = h.run()
    assert res.result.status == "error"
