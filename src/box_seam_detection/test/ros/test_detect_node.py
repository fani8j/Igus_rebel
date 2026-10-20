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
        capture_staging_dir: Path | None = None

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

    def run_detection(next_capture, cfg, progress=None, abort_check=None):
        orch.calls.append(cfg)
        orch.abort_check = abort_check
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


def test_cancel_during_detection_no_snapshot(harness):
    harness.snap.write_text("{}")
    started = threading.Event()

    def cancellable(nc, cfg, pr):
        started.set()
        for _ in range(200):
            if harness.orch.abort_check():
                return harness.orch.DetectResult(success=False, status="cancelled", reason="cancelled",
                                                 snapshot=None, snapshot_path=None, cut_permission=None)
            time.sleep(0.05)
        return harness.orch.DetectResult()
    harness.orch.behaviour = cancellable
    gh = harness._wait(harness.send()[0])
    assert gh.accepted and started.wait(5)
    harness._wait(gh.cancel_goal_async())
    res = harness._wait(gh.get_result_async())
    from action_msgs.msg import GoalStatus
    assert res.status == GoalStatus.STATUS_CANCELED
    assert res.result.status == "cancelled" and not res.result.success and res.result.snapshot_path == ""


def test_cancel_during_capture_wait(tmp_path, spinner):
    h = Harness(tmp_path, spinner, camera=False, capture_timeout_s=30.0)
    h.snap.write_text("{}")
    h.orch.behaviour = lambda nc, cfg, pr: (nc(1), h.orch.DetectResult())[1]
    gh = h._wait(h.send()[0])
    time.sleep(0.8)
    t0 = time.monotonic()
    h._wait(gh.cancel_goal_async())
    res = h._wait(gh.get_result_async())
    from action_msgs.msg import GoalStatus
    assert res.status == GoalStatus.STATUS_CANCELED and time.monotonic() - t0 < 3.0
    assert res.result.status == "cancelled" and not h.snap.exists()


def test_cancel_after_publish_stays_succeeded(harness):
    def late(nc, cfg, pr):
        time.sleep(0.5)
        return harness.orch.DetectResult()  # snapshot already published
    harness.orch.behaviour = late
    gh = harness._wait(harness.send()[0])
    harness._wait(gh.cancel_goal_async())
    res = harness._wait(gh.get_result_async())
    from action_msgs.msg import GoalStatus
    assert res.status == GoalStatus.STATUS_SUCCEEDED and res.result.success


def test_param_dry_run_exception_keeps_snapshot(tmp_path, spinner):
    h = Harness(tmp_path, spinner, dry_run=True)
    h.snap.write_text("{}")

    def boom(*a):
        raise RuntimeError("x")
    h.orch.behaviour = boom
    _, res, _ = h.run()
    assert res.result.status == "error" and h.snap.exists()


def test_empty_snapshot_path_passes_none(tmp_path, spinner):
    h = Harness(tmp_path, spinner, snapshot_path="")
    h.run()
    assert h.orch.calls[0].snapshot_path is None


def test_override_maps_to_overridden(harness):
    harness.orch.behaviour = lambda nc, cfg, pr: harness.orch.DetectResult(
        cut_permission={"status": "blocked", "override": "allow_cut_blocked"})
    assert harness.run()[1].result.cut_permission == "overridden"


def test_overlay_cleared_at_start(harness):
    got = []
    from sensor_msgs.msg import Image
    harness.client_node.create_subscription(Image, "/box_seam_detector/overlay", got.append,
                                            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
    harness.run()
    time.sleep(0.5)
    assert got and got[-1].width == 0


def test_parameter_validation(tmp_path):
    with pytest.raises(ValueError, match="capture_timeout_s"):
        DetectNode(orchestrator=make_fake_orchestrator(), parameter_overrides=[
            rclpy.parameter.Parameter("capture_timeout_s", value=0.0),
            rclpy.parameter.Parameter("warm_daemon", value=False)])


@pytest.mark.parametrize("slop", [0.0, -0.01, 0.06, 0.25])
def test_raw_sync_slop_validation(slop):
    with pytest.raises(ValueError, match="raw_sync_slop_s"):
        DetectNode(orchestrator=make_fake_orchestrator(), parameter_overrides=[
            rclpy.parameter.Parameter("raw_sync_slop_s", value=slop),
            rclpy.parameter.Parameter("warm_daemon", value=False)])


@pytest.mark.parametrize("bad", [
    [1280.0, 800.0, 624.6, 624.6, 633.5],
    [1280.0, 800.0, float("nan"), 624.6, 633.5, 399.0],
    [1280.5, 800.0, 624.6, 624.6, 633.5, 399.0],
    [1280.0, 800.0, 624.6, 0.0, 633.5, 399.0],
    [1280.0, 800.0, 624.6, 624.6, 1280.0, 399.0],
    [1280.0, 800.0, 624.6, 624.6, 633.5, 800.0],
])
def test_raw_depth_intrinsics_validation(bad):
    with pytest.raises(ValueError, match="raw_depth_intrinsics"):
        DetectNode(orchestrator=make_fake_orchestrator(), parameter_overrides=[
            rclpy.parameter.Parameter("raw_depth_intrinsics", value=bad),
            rclpy.parameter.Parameter("warm_daemon", value=False)])


def test_raw_depth_intrinsics_reach_scene_capture_and_status(tmp_path, spinner):
    h0 = Harness(tmp_path / "default", spinner, camera=False)
    assert h0.node.capture.raw_intrinsics is None                  # empty default = no fallback
    fb = [1280.0, 800.0, 624.600830078125, 624.600830078125, 633.5, 399.0]
    h = Harness(tmp_path, spinner, camera=False, node_depth_registration=True, raw_depth_intrinsics=fb)
    assert h.node.capture.raw_intrinsics == (1280, 800, fb[2], fb[3], fb[4], fb[5])
    assert _wait_for(lambda: h.status and h.status[-1]["camera"]["node_registration"]["fallback_intrinsics"])
    nr = h.status[-1]["camera"]["node_registration"]
    assert nr["fallback_size"] == "1280x800" and nr["last_intrinsics_source"] == ""


def test_shipped_config_declares_a_valid_raw_depth_intrinsics_fallback():
    import yaml
    from box_seam_detection.ros.capture import parse_raw_depth_intrinsics
    cfg = Path(__file__).resolve().parents[2] / "config" / "box_seam_detection.yaml"
    params = yaml.safe_load(cfg.read_text())["box_seam_detector"]["ros__parameters"]
    assert parse_raw_depth_intrinsics(params["raw_depth_intrinsics"]) == (
        1280, 800, 624.600830078125, 624.600830078125, 633.5, 399.0)


def test_node_registration_parameters_reach_scene_capture(tmp_path, spinner):
    h0 = Harness(tmp_path / "default", spinner, camera=False)
    assert h0.node.capture.node_registration
    h = Harness(tmp_path, spinner, camera=False, node_depth_registration=True, raw_sync_slop_s=0.04,
                raw_depth_topic="/cam2/depth/raw",
                depth_info_topic="/cam2/depth/info")
    cap = h.node.capture
    assert cap.node_registration and cap.raw_sync_slop_s == 0.04
    assert cap.raw_depth_topic == "/cam2/depth/raw" and cap.depth_info_topic == "/cam2/depth/info"
    assert _wait_for(lambda: h.status and h.status[-1]["camera"]["node_registration"]["enabled"])
    h2 = Harness(tmp_path / "b", spinner, camera=False, node_depth_registration=False)
    assert not h2.node.capture.node_registration


def _wait_for(predicate, timeout=10.0):
    t0 = time.monotonic()
    while not predicate():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


def test_staged_captures_are_archived_after_the_result(tmp_path, spinner):
    stage = tmp_path / "stage"
    h = Harness(tmp_path, spinner, capture_staging_dir=str(stage))
    out = tmp_path / "out"
    seen = {}

    def behaviour(next_capture, cfg, progress):
        seen["cfg_dir"] = cfg.capture_staging_dir
        rejected = next_capture(1)
        winner = next_capture(2)
        seen["staged"] = (rejected.image_path, winner.manifest_path)
        return h.orch.DetectResult(capture_id=winner.capture_id,
                                   attempts=[{"attempt": 1, "capture_id": rejected.capture_id},
                                             {"attempt": 2, "capture_id": winner.capture_id}])
    h.orch.behaviour = behaviour
    _, res, _ = h.run()
    assert res.result.success
    cid = res.result.capture_id
    assert seen["cfg_dir"] is not None and Path(seen["cfg_dir"]) in seen["staged"][0].parents
    assert _wait_for(lambda: (out / "captures" / f"{cid}.json").is_file()
                     and len(list((out / "failed_recent" / "captures").glob("*.json"))) == 1)
    from box_seam_detection.live.orchestrator import load_capture_manifest
    load_capture_manifest(out / "captures" / f"{cid}.json", str(out / "rgb" / f"{cid}.png"),
                          str(out / "pointcloud" / f"{cid}.pcd"))
    assert not list(Path(seen["cfg_dir"]).glob("*/*.*"))           # RAM staging emptied
    assert h.node._status_dict()["capture_staging_dir"] == str(seen["cfg_dir"])


def test_staged_capture_of_a_run_that_raised_goes_to_failed_recent(tmp_path, spinner):
    h = Harness(tmp_path, spinner, capture_staging_dir=str(tmp_path / "stage"))
    out = tmp_path / "out"

    def boom(next_capture, cfg, progress):
        next_capture(1)
        raise RuntimeError("detector exploded")
    h.orch.behaviour = boom
    _, res, _ = h.run()
    assert res.result.status == "error"
    assert _wait_for(lambda: len(list((out / "failed_recent" / "captures").glob("*.json"))) == 1)
    assert not (out / "captures").exists() or not list((out / "captures").glob("*.json"))


def test_staging_off_param_writes_directly_to_output_root(tmp_path, spinner):
    h = Harness(tmp_path, spinner, capture_staging_dir="off")

    def behaviour(next_capture, cfg, progress):
        files = next_capture(1)
        assert tmp_path / "out" in files.image_path.parents
        return h.orch.DetectResult(capture_id=files.capture_id)
    h.orch.behaviour = behaviour
    assert h.run()[1].result.success
    assert h.node._staging is None and h.orch.calls[0].capture_staging_dir is None
