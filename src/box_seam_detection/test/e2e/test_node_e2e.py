"""Node end-to-end: DetectBox goal -> real detection (SAM3) on replayed captures -> snapshot file. No robot, no driver.

Needs BOX_SEAM_E2E=1 (see conftest). Tests run in file order and share one node; the snapshot is in the scratch dir
(never /home/farhan/Igus_rebel/outputs/seam_3d_base_link.json).
"""

import json
import os
import threading
import time

import pytest

from . import e2e_lib as L
from .conftest import CAPTURE_IDS

TOL_MM = float(os.environ.get("BOX_SEAM_E2E_TOL_MM", "1.0"))  # node (rebuilt cloud) vs CLI; 0.44 mm worst seen on R2


def _accept(stack, replay, key):
    with replay(CAPTURE_IDS[key]):
        res = stack["client"].send(max_attempts=1)
    assert res["accepted"], res
    return res


@pytest.mark.parametrize("key", ["accepted_a", "accepted_b", "r2_highyaw"])
def test_accepted_capture_writes_executor_valid_snapshot(stack, replay, key):
    res = _accept(stack, replay, key)
    r = res["result"]
    assert res["goal_status"] == L.GOAL_SUCCEEDED
    assert (r["success"], r["status"], r["cut_permission"]) == (True, "accepted", "allowed"), r
    assert r["snapshot_path"] == str(stack["snap"]) and stack["snap"].is_file()
    doc = json.loads(stack["snap"].read_text())
    assert doc["schema"] == "box_h_cut_snapshot/v1"
    assert doc["cut_permission"]["status"] == "allowed"
    assert L.executor_snapshot_problems(doc) == []
    assert doc["snapshot_id"] == r["snapshot_id"] and doc["capture_id"] == r["capture_id"]
    # acquisition stamp = the colour image stamp the fake camera published, after the request
    assert doc["acquisition_stamp_ns"] in res["new_image_stamps"], "stamp is not a replayed colour image stamp"
    assert abs(time.time_ns() - doc["acquisition_stamp_ns"]) < 120e9
    # feedback covers the stages in order; status topic went busy -> idle with the last result
    stages = [f["stage"] for f in res["feedback"]]
    for s in ("capture", "detect", "base_link", "snapshot", "side_seed", "done"):
        assert s in stages, stages
    assert [s["state"] for s in res["statuses"]][0] == "busy" and res["statuses"][-1]["state"] == "idle"
    assert res["statuses"][-1]["last_result"]["status"] == "accepted"
    # markers (clear + seam/corner markers) and overlay image
    assert any(len(m.markers) > 1 for m in res["markers"]) and res["overlays"]
    assert res["overlays"][-1].width > 0 and res["overlays"][-1].encoding == "bgr8"
    # keep for the CLI comparison
    (stack["scratch"] / f"node_{key}.json").write_text(json.dumps(doc))
    (stack["scratch"] / f"node_{key}.capture_id").write_text(CAPTURE_IDS[key])


@pytest.mark.parametrize("key", ["accepted_a", "accepted_b", "r2_highyaw"])
def test_node_snapshot_matches_cli(stack, key):
    node_file = stack["scratch"] / f"node_{key}.json"
    if not node_file.is_file():
        pytest.skip("accepted test did not run")
    cli = stack["scratch"] / f"cli_{key}.json"
    p = L.run_cli_snapshot(CAPTURE_IDS[key], cli, stack["scratch"] / "cli_work" / key)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    worst, per = L.max_point_diff_mm(json.loads(node_file.read_text()), json.loads(cli.read_text()))
    print(f"\n{key}: node vs CLI max point difference {worst:.4f} mm")
    assert worst <= TOL_MM, per


@pytest.mark.parametrize("key,status", [("rejected_rect", "rejected"), ("blocked_top_face", "blocked"),
                                        ("blocked_soft_range", "rejected")])
def test_failed_detection_removes_previous_snapshot(stack, replay, key, status):
    # a fresh accepted snapshot exists first; the failing run must remove it (fail closed)
    _accept(stack, replay, "accepted_a")
    assert stack["snap"].is_file()
    res = _accept(stack, replay, key)
    r = res["result"]
    assert (r["success"], r["status"], r["cut_permission"]) == (False, status, "blocked"), r
    assert r["snapshot_path"] == "" and r["reason"]
    assert not stack["snap"].exists(), "snapshot must be absent after a rejected/blocked detection"
    assert res["statuses"][-1]["state"] == "idle"


def test_busy_goal_is_rejected(stack, replay):
    with replay(CAPTURE_IDS["accepted_a"]):
        first: list = []
        t = threading.Thread(target=lambda: first.append(stack["client"].send(max_attempts=1)))
        t.start()
        end = time.time() + 15
        while time.time() < end and not any(s["state"] == "busy" for _, s in stack["client"].status_msgs[-3:]):
            time.sleep(0.05)
        second = stack["client"].send(max_attempts=1, timeout_s=10)
        t.join(120)
    assert second["accepted"] is False, second
    assert first and first[0]["result"]["status"] == "accepted", first


def test_capture_timeout_is_error_and_fail_closed(stack, replay):
    _accept(stack, replay, "accepted_a")
    assert stack["snap"].is_file()
    # fake camera stopped (replay context exited): the node must time out, report error, and leave no snapshot
    res = stack["client"].send(max_attempts=1, timeout_s=60)
    r = res["result"]
    assert (r["success"], r["status"]) == (False, "error"), r
    assert "CaptureTimeout" in r["reason"]
    assert not stack["snap"].exists()
    assert res["goal_status"] == 6  # ABORTED
    # the node recovers once the camera is back
    res2 = _accept(stack, replay, "accepted_b")
    assert res2["result"]["status"] == "accepted"
