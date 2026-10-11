# Provenance: ported from rgbd_captures/tests/test_autolabel_v2.py (HEAD 6df5994); imports made package-relative.
"""Tests for scripts/autolabel_v2 (box_autolabel/v5 pipeline).

Fast synthetic-only tests: schema invariants, corner-order mapping,
seam-never-midpoints, confidence bounds, and graceful failure paths.
No PCD/RGB dataset reads, no model downloads.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


from box_seam_detection.core import coarse as _coarse
from box_seam_detection.core.config import CORNER_ORDER, default_config
from box_seam_detection.core.overlay import draw_overlay
from box_seam_detection.core.pipeline import (_fail_record,
                                   compute_confidence_quad,
                                   compute_confidence_seam,
                                   process_capture)
from box_seam_detection.core.sam3_live import Sam3LiveUnavailable, run_sam3_live

REQUIRED_TOP = {"schema", "capture_id", "image", "image_size", "provenance",
                "top_face", "side_faces", "seam", "checks", "confidence",
                "confidence_quad", "confidence_seam", "overlay"}


def _assert_schema(rec: dict):
    assert REQUIRED_TOP <= set(rec), f"missing keys: {REQUIRED_TOP - set(rec)}"
    # Contract: top-level confidence is the quad confidence.
    assert rec["confidence"] == rec["confidence_quad"]
    assert set(rec["checks"]) >= {"rgb_depth_quad_iou", "wrong_face_suspect",
                                  "needs_review", "reasons"}
    assert rec["schema"] == "box_autolabel/v5"
    assert set(rec["top_face"]) == {"corners_px", "corners_xyz_camera_m", "plane"}
    assert set(rec["seam"]) == {"status", "endpoints_px", "axis", "score",
                                "source", "estimate"}
    assert rec["seam"]["status"] in ("ok", "estimated", "ambiguous",
                                     "invisible")
    if rec["seam"]["status"] != "ok":
        assert rec["seam"]["endpoints_px"] is None, \
            "non-ok seam must carry null endpoints (never midpoints)"
    assert set(rec["checks"]) == {"rgb_depth_quad_iou", "wrong_face_suspect",
                                  "needs_review",
                                  "edge_residual_rms_px", "support_score",
                                  "corner_flags", "reasons"}
    assert 0.0 <= float(rec["confidence"]) <= 1.0
    if rec["top_face"]["corners_px"] is not None:
        assert set(rec["top_face"]["corners_px"]) == set(CORNER_ORDER)


# ---- corner-order mapping ----------------------------------------------

def test_archived_mapping_matches_compare_convention(tmp_path):
    """Archived corners_px row i maps to CORNER_ORDER[i], exactly like
    run_auto_guide_compare.build_auto_terminal_doc's positional zip."""
    row = {"model": "sam3", "checkpoint_sha256": "abc",
           "prompt": "cardboard box top face",
           "corners_px": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
           "selected_mask": {"concept_score": 0.9, "prompt_used": "x"},
           "status": "ok", "quad_valid": True}
    (tmp_path / "cid.json").write_text(json.dumps(row), encoding="utf-8")
    corners, prov = _coarse.load_sam3_archived("cid", archive_dir=tmp_path)
    assert corners is not None
    for i, name in enumerate(CORNER_ORDER):
        assert list(corners[i]) == row["corners_px"][i], \
            f"row {i} must map to {name} (compare-script convention)"
    assert prov["sam3"]["checkpoint_sha256"] == "abc"
    assert prov["sam3"]["concept_score"] == pytest.approx(0.9)


def test_archived_missing_row_returns_reason(tmp_path):
    corners, prov = _coarse.load_sam3_archived("nope", archive_dir=tmp_path)
    assert corners is None
    assert "reason" in prov


def test_depth_quad_to_named_known_square():
    tltr = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
    got = _coarse.depth_quad_to_named(tltr)
    assert got.tolist() == [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]


# ---- seam never equals quad midpoints -----------------------------------

def _edge_midpoints(quad):
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return (q[3] + q[0]) / 2.0, (q[1] + q[2]) / 2.0  # left-edge, right-edge mids




def test_estimator_axis_values_on_synthetic_stripes():
    """Estimator axis picks the taped direction (lr/bf, not long/short)."""
    from box_seam_detection.core.seam_estimator import estimate_seam
    for axis in ("lr", "bf"):
        bgr, quad = _tape_face(None, axis=axis)
        xyz = {n: [float(i), 0.0, 0.5] for i, n in enumerate(
            ("back_left", "back_right", "front_right", "front_left"))}
        res = estimate_seam(bgr, quad, corners_xyz=xyz)
        assert res["axis"] == axis, (axis, res.get("candidates"))


# ---- confidence ----------------------------------------------------------

def test_confidence_quad_missing_evidence_penalised():
    """Missing evidence keeps its weight as a 0 term: absent validation
    lowers quad confidence instead of renormalising away."""
    v2 = default_config()["v2"]
    assert compute_confidence_quad(None, None, None, None, v2) == 0.0
    full = compute_confidence_quad(1.0, 0.0, 1.0, 1.0, v2)
    assert full == pytest.approx(1.0)
    # Concept-only frame: other weights stay in the denominator as zeros.
    only_concept = compute_confidence_quad(0.6, None, None, None, v2)
    assert only_concept == pytest.approx(0.6 * v2["w_concept"])
    mid = compute_confidence_quad(0.9, 1.5, 0.9, 0.9, v2)
    assert 0.0 < mid < 1.0


def test_confidence_seam_non_ok_is_zero():
    assert compute_confidence_seam("ambiguous", 0.9) == 0.0
    assert compute_confidence_seam("invisible", None) == 0.0
    assert compute_confidence_seam("ok", 0.55) == pytest.approx(0.55)
    assert compute_confidence_seam("ok", None) == 0.0


# ---- graceful failures ----------------------------------------------------

def test_fail_record_schema_and_null_seam():
    cfg = default_config()
    rec = _fail_record("cid", {"image": "/tmp/x.png"}, cfg, "refine_rejected",
                       ["left: only 2/10 valid profiles"], "sam3_archived",
                       image_size=[1280, 720], concept_score=0.7)
    _assert_schema(rec)
    assert rec["top_face"]["corners_px"] is None
    assert rec["seam"] == {"status": "invisible", "endpoints_px": None,
                           "axis": None, "score": None, "source": None,
                           "estimate": None}
    # Null-corner records carry 0.0 confidence despite partial evidence.
    assert rec["confidence"] == 0.0
    assert rec["confidence_quad"] == 0.0
    assert rec["confidence_seam"] == 0.0
    assert rec["checks"]["needs_review"] is True


def test_process_capture_missing_manifest_never_raises(tmp_path):
    cfg = default_config()
    rec, debug = process_capture(
        "ghost",
        {"image": str(tmp_path / "x.png"), "pcd": str(tmp_path / "x.pcd"),
         "manifest": str(tmp_path / "missing.json")},
        cfg, source="depth_only")
    _assert_schema(rec)
    assert rec["top_face"]["corners_px"] is None
    assert any("unreadable_manifest" in r for r in rec["checks"]["reasons"])


def test_sam3_live_missing_interpreter_fails_cleanly():
    with pytest.raises(Sam3LiveUnavailable):
        run_sam3_live("/tmp/does_not_matter.png",
                      python_bin="/tmp/no_such_python_xyz")


class _DeadSession:
    """Session whose worker is gone: query raises Sam3LiveUnavailable."""

    def query(self, image_path, timeout_s=300):
        raise Sam3LiveUnavailable("sam3 worker process is not running")


def test_process_capture_sam3_live_unavailable_is_record(tmp_path):
    cfg = default_config()
    img = np.zeros((60, 80, 3), np.uint8)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    man = {"schema": "rgbd_capture/v1", "capture_id": "x",
           "image": {"path": str(p), "width": 80, "height": 60}}
    mp = tmp_path / "x.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    rec, _ = process_capture(
        "x", {"image": str(p), "pcd": str(tmp_path / "x.pcd"),
              "manifest": str(mp)},
        cfg, source="sam3_live", session=_DeadSession())
    _assert_schema(rec)
    assert rec["provenance"]["mask_source"] == "sam3_live"
    assert any("sam3_live_unavailable" in r for r in rec["checks"]["reasons"])


def test_gate_rejected_detections_preserved_in_attempt(tmp_path):
    """Sanity-gate rejections keep the detection list + per-detection
    rejection reasons in the recorded sam3_live attempt."""
    cfg = default_config()
    img = np.zeros((60, 80, 3), np.uint8)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    man = {"schema": "rgbd_capture/v1", "capture_id": "x",
           "image": {"path": str(p), "width": 80, "height": 60}}
    mp = tmp_path / "x.json"
    mp.write_text(json.dumps(man), encoding="utf-8")

    class _GateRejectSession:
        checkpoint_sha256 = "sha"
        model_revision = "rev"

        def query(self, image_path, timeout_s=300):
            return {"ok": False, "id": 1,
                    "reason": "all_detections_failed_sanity_gate",
                    "prompt": "cardboard box top face",
                    "n_detections": 2,
                    "detections": [
                        {"score": 0.99, "box_xyxy": [0, 0, 70, 50],
                         "area_frac": 0.30, "brown_frac": 0.01,
                         "rejected": "brown_too_low"},
                        {"score": 0.80, "box_xyxy": [5, 5, 9, 9],
                         "area_frac": 0.0005, "brown_frac": 0.9,
                         "rejected": "area_out_of_range"}]}

    rec, _ = process_capture(
        "x", {"image": str(p), "pcd": str(tmp_path / "x.pcd"),
              "manifest": str(mp)},
        cfg, source="sam3_live", session=_GateRejectSession())
    _assert_schema(rec)
    assert rec["top_face"]["corners_px"] is None
    assert any("all_detections_failed_sanity_gate" in r
               for r in rec["checks"]["reasons"])
    live_attempts = [a for a in rec["provenance"]["attempts"]
                     if a["source"] == "sam3_live"]
    assert live_attempts, rec["provenance"]["attempts"]


def test_sam3_live_attempt_carries_detections_on_gate_reject(tmp_path):
    """_live_coarse threads the worker's detection list (with rejection
    reasons) into the candidate prov even when no quad results."""
    from box_seam_detection.core.pipeline import _live_coarse

    class _GateRejectSession:
        checkpoint_sha256 = "sha"
        model_revision = "rev"

        def query(self, image_path, timeout_s=300):
            return {"ok": False, "id": 1,
                    "reason": "all_detections_failed_sanity_gate",
                    "prompt": "cardboard box top face",
                    "detections": [
                        {"score": 0.5, "box_xyxy": [0, 0, 10, 10],
                         "area_frac": 0.5, "brown_frac": 0.0,
                         "rejected": "brown_too_low"}]}

    corners, prov, reason = _live_coarse("/tmp/x.png", _GateRejectSession(),
                                        {})
    assert corners is None
    assert "all_detections_failed_sanity_gate" in reason
    dets = (prov.get("sam3") or {}).get("detections")
    assert dets == [{"score": 0.5, "box_xyxy": [0, 0, 10, 10],
                     "area_frac": 0.5, "brown_frac": 0.0,
                     "rejected": "brown_too_low"}]


def test_worker_query_timeout_is_real_and_kills_worker():
    """A sleeping fake worker: query(timeout=1) returns within ~1.5 s and
    the worker is marked dead (subsequent queries raise immediately)."""
    import time
    from box_seam_detection.core.sam3_session import Sam3Worker

    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; time.sleep(30)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1)
    worker = Sam3Worker.__new__(Sam3Worker)
    worker.proc = proc
    worker._seq = 0
    import threading as _th
    worker._lock = _th.Lock()
    worker.model_revision = None
    worker.checkpoint_sha256 = None
    worker.threshold = 0.5
    t0 = time.monotonic()
    with pytest.raises(Sam3LiveUnavailable, match="timed out"):
        worker.query("/tmp/does_not_matter.png", timeout_s=1)
    dt = time.monotonic() - t0
    assert dt < 1.5, f"timeout not real: took {dt:.2f}s"
    assert worker.proc is None  # marked dead
    with pytest.raises(Sam3LiveUnavailable):
        worker.query("/tmp/does_not_matter.png", timeout_s=1)
    proc.wait(timeout=10)


def test_runner_null_record_has_full_failure_fields():
    """run._null_record routes through pipeline._fail_record: 0.0
    confidences, needs_review true, reasons preserved."""
    from box_seam_detection.core.run import _null_record
    cfg = default_config()
    rec = _null_record("cid", {"image": "/tmp/x.png"}, cfg, "missing_input:pcd")
    _assert_schema(rec)
    assert rec["confidence"] == 0.0
    assert rec["confidence_quad"] == 0.0
    assert rec["confidence_seam"] == 0.0
    assert rec["checks"]["needs_review"] is True
    assert rec["checks"]["reasons"] == ["missing_input:pcd"]


def test_overlay_draws_on_failure_record():
    cfg = default_config()
    rec = _fail_record("cid", {"image": ""}, cfg, "no_coarse_quad",
                       ["no_coarse_quad"], "depth_only", image_size=[64, 48])
    bgr = np.zeros((48, 64, 3), np.uint8)
    out = draw_overlay(bgr, rec, {})
    assert out.shape == bgr.shape


# ---- wrong-face gate ------------------------------------------------------

def _fake_depth_est_ok(quad):
    return {"ok": True, "quad": np.asarray(quad, dtype=np.float64),
            "hull": None, "support": None, "reason": None,
            "depth": {"status": "ok"}}


class _FakePlane:
    def __init__(self, normal, d):
        self.normal = normal
        self.d = d


class _FakeEst:
    def __init__(self, rms_mm, n_inliers, inlier_ratio):
        self.rms_mm = rms_mm
        self.n_inliers = n_inliers
        self.inlier_ratio = inlier_ratio


class _FakeGuided:
    def __init__(self, plane, est, xyz, erosion_px=8):
        self.status = "OK"
        self.plane = plane
        self.estimate = est
        self.camera_corners_xyz = xyz
        self.erosion_px = erosion_px
        self.reasons = []


def _synth_pcd_two_planes(seed=0):
    """Top plane z=0.42 + front wall y-tilted; camera-frame metric points."""
    rng = np.random.default_rng(seed)
    n_top, n_wall = 4000, 1500
    top = np.column_stack([
        rng.uniform(-0.15, 0.15, n_top),
        rng.uniform(-0.10, 0.10, n_top),
        0.42 + rng.normal(0, 0.0006, n_top)])
    # Same (x, y) footprint as the top's lower half, but 4 cm closer: at
    # shared pixels nearest-depth-wins keeps the wall, so a spilling quad
    # sees a genuine two-surface interior while the clean quad is pure top.
    wall = np.column_stack([
        rng.uniform(-0.15, 0.15, n_wall),
        rng.uniform(-0.10, 0.10, n_wall),
        0.38 + rng.normal(0, 0.0006, n_wall)])
    return np.vstack([top, wall]).astype(np.float64)


def _write_pcd_ascii(path, xyz):
    n = xyz.shape[0]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n"
                 "TYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA ascii\n")
        for row in xyz:
            fh.write(f"{row[0]:.5f} {row[1]:.5f} {row[2]:.5f}\n")


def test_depth_cross_flags_top_side_quad():
    """A quad spilling off the measured top trips the IoU gate: wrong_face
    plus low_rgb_depth_iou / wrong_face_suspect reasons."""
    from box_seam_detection.core.pipeline import check_depth_cross
    cfg = default_config()
    W, H = 1920, 1080
    top = np.array([[799.0, 642.0], [1001.0, 702.0],
                    [888.0, 925.0], [674.0, 831.0]])
    manifest = {"image": {"path": "/tmp/x.png", "width": W, "height": H},
                "pointcloud": {"path": "/tmp/x.pcd"},
                "camera_info": {"width": W, "height": H}}
    est = _fake_depth_est_ok(top)
    ok_cross, _ = check_depth_cross(top, manifest, est, None, W, H, cfg)
    side = top.copy()
    side[2, 1] += 220
    side[3, 1] += 220
    bad_cross, _ = check_depth_cross(side, manifest, est, None, W, H, cfg)
    assert ok_cross["iou"] is not None and ok_cross["iou"] > 0.9
    assert not ok_cross["wrong_face"]
    assert bad_cross["wrong_face"]
    assert any(r.startswith("low_rgb_depth_iou") for r in bad_cross["reasons"])
    assert "wrong_face_suspect" in bad_cross["reasons"]


def test_plane_support_arm_fires_with_iou_none(tmp_path, monkeypatch):
    """Regression: a top+side quad with NO independent depth quad (iou None)
    still triggers wrong_face via the plane-support arm (inlier ratio /
    all-point p95), which the old inlier-RMS arm could never fire."""
    from box_seam_detection.core import pipeline as _pipe
    cfg = default_config()
    W, H = 640, 480
    fx = fy = 500.0
    cx, cy = W / 2.0, H / 2.0
    xyz = _synth_pcd_two_planes()
    pcd_path = tmp_path / "two_planes.pcd"
    _write_pcd_ascii(str(pcd_path), xyz)
    import cv2 as _cv2
    _cv2.imwrite(str(tmp_path / "x.png"),
                np.zeros((H, W, 3), dtype=np.uint8))
    manifest = {
        "schema": "rgbd_capture/v1",
        "capture_id": "synth",
        "image": {"path": str(tmp_path / "x.png"), "width": W, "height": H,
                  "header": {"frame_id": "camera_color_optical_frame"}},
        "pointcloud": {"path": str(pcd_path),
                       "header": {"frame_id": "camera_color_optical_frame"}},
        "camera_info": {"width": W, "height": H,
                        "header": {"frame_id": "camera_color_optical_frame"},
                        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                        "d": [], "distortion_model": "plumb_bob",
                        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0,
                              0.0, 0.0, 1.0, 0.0]},
    }
    # Project the top points to find their pixel box, then build a clean
    # quad (top only) and a spill quad reaching into wall pixels.
    import sys as _sys
    import box_seam_detection.core.pbd.camera_model as _cm
    cam = _cm.CameraModel.from_mapping(
        {"frame_id": "camera_color_optical_frame", "width": W, "height": H,
         "k": manifest["camera_info"]["k"], "d": [],
         "distortion_model": "plumb_bob",
         "p": manifest["camera_info"]["p"]})
    uv = np.asarray(_cm.project_xyz(xyz[:4000], cam), dtype=float)
    u0, v0 = float(uv[:, 0].min()), float(uv[:, 1].min())
    u1, v1 = float(uv[:, 0].max()), float(uv[:, 1].max())
    clean = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    spill = clean.copy()
    spill[2, 1] = min(H - 1, v1 + 0.5 * (v1 - v0))
    spill[3, 1] = min(H - 1, v1 + 0.5 * (v1 - v0))

    import box_seam_detection.core.pbd.sam3_depth_fusion as _fusion  # noqa: F401
    real_fit = _fusion.fit_sam3_guided_depth_plane

    def _fit(manifest_arg, quad, **kw):
        return real_fit(manifest_arg, quad, **kw)

    monkeypatch.setattr(_fusion, "fit_sam3_guided_depth_plane", _fit)
    # No independent estimate (depth_no_top) -> iou None on both. The
    # scene itself is two-surface (wall under the whole footprint), so even
    # the clean quad is suspect: the arm fires on both, harder on spill.
    ok_cross, _ = _pipe.check_depth_cross(clean, manifest, None,
                                          "depth_no_top", W, H, cfg)
    bad_cross, _ = _pipe.check_depth_cross(spill, manifest, None,
                                           "depth_no_top", W, H, cfg)
    assert ok_cross["iou"] is None
    assert bad_cross["iou"] is None
    assert bad_cross["wrong_face"], bad_cross["reasons"]
    assert any(r.startswith("low_plane_inlier_ratio")
               for r in bad_cross["reasons"])
    assert "wrong_face_suspect" in bad_cross["reasons"]
    assert (bad_cross.get("plane_p95_all_mm", 0)
            >= ok_cross.get("plane_p95_all_mm", 0))


def test_depth_cross_never_invents_plane():
    """Without a depth estimate the cross-check reports the missing depth
    reason and no plane — never a fabricated agreement."""
    from box_seam_detection.core.pipeline import check_depth_cross
    cfg = default_config()
    quad = np.array([[10.0, 10.0], [50.0, 10.0],
                     [50.0, 40.0], [10.0, 40.0]])
    manifest = {"image": {"path": "/tmp/x.png", "width": 64, "height": 48},
                "pointcloud": {"path": "/tmp/x.pcd"},
                "camera_info": {"width": 64, "height": 48}}
    cross, _ = check_depth_cross(quad, manifest, None,
                                 "depth_no_top", 64, 48, cfg)
    assert cross["iou"] is None
    assert cross["plane"] is None
    assert not cross["wrong_face"]
    assert any("depth_no_top" in r for r in cross["reasons"])


# ---- SAM3 sanity gate (distractor rejection) ----------------------------------

def test_sanity_gate_rejects_absurd_and_keeps_top():
    """Unit test with a distractor: a huge high-score non-cardboard mask is
    rejected by the area/brown sanity gate; the smaller brown top wins."""
    from box_seam_detection.core import sam3_worker as _w
    dets = [
        {"score": 0.99, "box_xyxy": [0, 0, 600, 440],
         "area_frac": 0.30, "brown_frac": 0.01},   # distractor: big, not brown
        {"score": 0.90, "box_xyxy": [100, 100, 300, 260],
         "area_frac": 0.04, "brown_frac": 0.60},   # true top
    ]
    order = sorted(range(len(dets)), key=lambda di: dets[di]["score"],
                   reverse=True)
    picked = None
    for di in order:
        dd = dets[di]
        if not (_w.MIN_MASK_AREA_FRAC <= dd["area_frac"]
                <= _w.MAX_MASK_AREA_FRAC):
            continue
        if dd["brown_frac"] is not None and dd["brown_frac"] < \
                _w.MIN_MASK_BROWN_FRAC:
            continue
        picked = di
        break
    assert picked == 1, dets
    # And a run of pure distractors picks nothing.
    only_bad = [{"score": 0.99, "box_xyxy": [0, 0, 600, 440],
                 "area_frac": 0.30, "brown_frac": 0.01}]
    picked2 = None
    for di in range(len(only_bad)):
        dd = only_bad[di]
        if not (_w.MIN_MASK_AREA_FRAC <= dd["area_frac"]
                <= _w.MAX_MASK_AREA_FRAC):
            continue
        if dd["brown_frac"] is not None and dd["brown_frac"] < \
                _w.MIN_MASK_BROWN_FRAC:
            continue
        picked2 = di
        break
    assert picked2 is None


# ---- fallback chain ---------------------------------------------------------

def test_auto_chain_skips_identical_archived(tmp_path):
    """When live and archived quads agree, archived is skipped as duplicate;
    the chain still ends with the depth_only candidate slot."""
    from box_seam_detection.core.pipeline import _gather_coarse_candidates
    cfg = default_config()
    quad = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    row = {"model": "sam3", "checkpoint_sha256": "abc",
           "prompt": "cardboard box top face", "corners_px": quad,
           "selected_mask": {"concept_score": 0.9}, "status": "ok",
           "quad_valid": True}
    (tmp_path / "cid.json").write_text(json.dumps(row), encoding="utf-8")

    class _Live:
        checkpoint_sha256 = "live-sha"
        model_revision = "rev"

        def query(self, image_path, timeout_s=300):
            return {"ok": True, "id": 1, "corners_px": quad,
                    "concept_score": 0.8, "prompt": "cardboard box top face"}

    cands = _gather_coarse_candidates("cid", "auto", "/tmp/x.png", _Live(),
                                      tmp_path, None, "depth_no_top", cfg)
    assert [c["source"] for c in cands] == ["sam3_live", "sam3_archived",
                                            "depth_only"]
    assert cands[1].get("skipped")
    assert cands[1]["reason"] == "sam3_archived_same_as_live"
    assert cands[2]["coarse"] is None


# ---- estimator integration --------------------------------------------------

def _tape_face(tmp_path, axis="lr", tape=True, seed=0, size=(640, 480),
               gap=True):
    """Synthetic brown face with an optional low-sat tape band + flap gap
    (rect-space face 400x300 warped to a fixed perspective quad). The gap
    is a 2-px dark valley at the seam row/col. Returns (bgr, quad)."""
    from box_seam_detection.core.seam_estimator import rectify_face  # noqa: F401

    rng = np.random.default_rng(seed)
    W, H = 400, 300
    sat = np.full((H + 1, W + 1), 95.0)
    val = np.full((H + 1, W + 1), 165.0)
    hue = np.full((H + 1, W + 1), 15.0)
    if tape:
        hw = 7
        if axis == "lr":
            sat[150 - hw:150 + hw + 1, :] -= 28.0
            val[150 - hw:150 + hw + 1, :] += 8.0
        else:
            sat[:, 200 - hw:200 + hw + 1] -= 28.0
            val[:, 200 - hw:200 + hw + 1] += 8.0
    if gap:
        if axis == "lr":
            val[149:151, :] = np.minimum(val[149:151, :], 105.0)
        else:
            val[:, 199:201] = np.minimum(val[:, 199:201], 105.0)
    sat = np.clip(sat + rng.normal(0, 2.0, sat.shape), 0, 255).astype(np.uint8)
    val = np.clip(val + rng.normal(0, 2.0, val.shape), 0, 255).astype(np.uint8)
    hue = np.clip(hue + rng.normal(0, 1.0, hue.shape), 0, 179).astype(np.uint8)
    face = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)
    quad = np.array([(120, 90), (520, 140), (480, 400), (90, 360)], float)
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    img = np.full((size[1], size[0], 3), 60, np.uint8)
    cv2.warpPerspective(face, M, (size[0], size[1]), dst=img,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return img, quad


def test_compute_seam_stage_ok_estimate_wins():
    """Confirm mode: prior confirmed by gap -> ok, endpoints = prior ∩
    edges re-rounded to 0.1 px; gap line kept in estimate for audit."""
    from box_seam_detection.core.pipeline import compute_seam_stage
    cfg = default_config()
    bgr, quad = _tape_face(None)
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    seam, stage = compute_seam_stage(bgr, quad, cfg, corners_xyz=xyz)
    assert seam["status"] in ("ok", "estimated"), stage
    assert seam["estimate"]["status"] in ("ok", "estimated")
    if seam["status"] == "ok":
        assert seam["source"] == "prior_confirmed_by_gap"
        # Final endpoints are the prior endpoints re-rounded to 0.1 px.
        for k in ("seam_left", "seam_right"):
            for a, b in zip(seam["endpoints_px"][k],
                            seam["estimate"]["prior_endpoints_px"][k]):
                assert a == pytest.approx(b, abs=0.06)
        assert seam["estimate"]["observed_line"] is not None
    else:
        assert seam["endpoints_px"] is None


def test_compute_seam_stage_refine_mode_publishes_gap():
    """Refine mode (comparison): ok publishes the observed gap line."""
    from box_seam_detection.core.pipeline import compute_seam_stage
    cfg = default_config()
    cfg["seam_estimator"] = dict(cfg["seam_estimator"],
                                 seam_mode="refine")
    bgr, quad = _tape_face(None)
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    seam, stage = compute_seam_stage(bgr, quad, cfg, corners_xyz=xyz)
    assert seam["status"] == "ok", stage
    assert seam["source"] == "observed_gap"
    for k in ("seam_left", "seam_right"):
        for a, b in zip(seam["endpoints_px"][k],
                        seam["estimate"]["observed_line"][k]):
            assert a == pytest.approx(b, abs=0.06)


def test_compute_seam_stage_estimated_keeps_null_endpoints():
    """No gap: estimator 'estimated' -> endpoints null, estimate kept."""
    from box_seam_detection.core.pipeline import compute_seam_stage
    cfg = default_config()
    bgr, quad = _tape_face(None, tape=False, gap=False)
    seam, stage = compute_seam_stage(bgr, quad, cfg)
    assert seam["estimate"] is not None
    if seam["estimate"]["status"] == "estimated":
        assert seam["endpoints_px"] is None
        # Distinct top-level "estimated": axis clear, gap unobserved.
        assert seam["status"] == "estimated"
        assert seam["axis"] == seam["estimate"]["axis"]
        assert seam["source"] == "prior"
        # Estimate-only endpoints carry the prior line.
        assert seam["estimate"]["estimate_endpoints_px"] == \
            seam["estimate"]["prior_endpoints_px"]
    else:
        # Estimator ambiguous on textureless synthetic input is also a
        # valid fail-closed outcome; endpoints must still be null.
        assert seam["estimate"]["status"] == "ambiguous"
        assert seam["endpoints_px"] is None


def test_reseam_record_keeps_corners_byte_identical(tmp_path):
    """Reseam round-trip: only seam fields/reasons/overlay/confidence_seam
    change; provenance.reseam = {from, code_git_rev, config} recorded,
    original config and code_git_rev kept."""
    from box_seam_detection.core.run import reseam_record
    cfg = default_config()
    cfg["corner_fold"]["enabled"] = False  # seam-only behaviour under test
    bgr, quad = _tape_face(None)
    p = tmp_path / "face.png"
    cv2.imwrite(str(p), bgr)
    names = ("back_left", "back_right", "front_right", "front_left")
    corners = {n: [round(float(quad[i][0]), 1), round(float(quad[i][1]), 1)]
               for i, n in enumerate(names)}
    src = {"schema": "box_autolabel/v5", "capture_id": "synth",
           "image": str(p), "image_size": [640, 480],
           "provenance": {"mask_source": "sam3_live", "config": {"old": 1},
                          "code_git_rev": "abc123"},
           "top_face": {"corners_px": corners,
                        "corners_xyz_camera_m": None, "plane": None},
           "side_faces": [],
           "seam": {"status": "ambiguous", "endpoints_px": None,
                    "axis": None, "score": None},
           "checks": {"rgb_depth_quad_iou": None,
                      "wrong_face_suspect": False, "needs_review": True,
                      "edge_residual_rms_px": 1.0, "support_score": 0.9,
                      "reasons": ["seed_reason"]},
           "confidence": 0.5, "confidence_quad": 0.5, "confidence_seam": 0.0,
           "overlay": "old.png"}
    rec, debug = reseam_record(src, {"image": str(p)}, cfg, "srcdir")
    assert rec["top_face"] == src["top_face"]
    assert rec["image_size"] == src["image_size"]
    assert rec["provenance"]["mask_source"] == "sam3_live"
    assert rec["provenance"]["config"] == {"old": 1}
    assert rec["provenance"]["code_git_rev"] == "abc123"
    assert rec["provenance"]["reseam"]["from"] == "srcdir"
    assert rec["provenance"]["reseam"]["config"] == cfg["seam_estimator"]
    assert "seed_reason" in rec["checks"]["reasons"]
    assert set(rec["seam"]) == {"status", "endpoints_px", "axis", "score",
                                "source", "estimate"}
    assert rec["seam"]["estimate"] is not None
    if rec["seam"]["status"] == "ok":
        assert rec["seam"]["endpoints_px"] is not None
    else:
        assert rec["seam"]["endpoints_px"] is None
    assert debug.get("bgr") is not None


def test_corner_shift_gate_flags_pushed_corner():
    """A refined corner pushed >10 px (scaled) from coarse flags
    large_shift and the report sets any_flag (needs_review upstream)."""
    from box_seam_detection.core import corner_checks
    gray = np.full((200, 200), 150.0)
    gray[:, 100:] = 60.0  # vertical edge at x=100
    coarse = np.array([[40, 40], [120, 40], [120, 160], [40, 160]], float)
    refined = np.array([[40, 40], [120, 40], [120, 160], [40, 160]], float)
    rep = corner_checks.check_corners(gray, coarse, refined, [1280, 720])
    assert rep["any_flag"] is False
    pushed = refined.copy()
    pushed[1] += [15.0, 0.0]
    rep2 = corner_checks.check_corners(gray, coarse, pushed, [1280, 720])
    assert "large_shift" in rep2["corners"]["back_right"]["flags"]
    assert rep2["any_flag"] is True


def test_corner_border_gate_flags_edge_corner():
    """A refined corner within 8 px (scaled) of the image border flags."""
    from box_seam_detection.core import corner_checks
    gray = np.full((720, 1280), 150.0)
    coarse = np.array([[40, 40], [120, 40], [120, 160], [40, 160]], float)
    refined = np.array([[40, 40], [1277, 40], [120, 160], [40, 160]], float)
    rep = corner_checks.check_corners(gray, coarse, refined, [1280, 720])
    assert "near_border" in rep["corners"]["back_right"]["flags"]


def test_corner_ambiguous_edge_rejects_print_not_gap():
    """A double edge (parallel foreign line ≥4 px away, ≥60% gradient)
    on the incident edges records ambiguous_frac; it flags
    ambiguous_edge only when corner_cfg flag_ambiguous is true
    (default false: too noisy on tape stripes)."""
    from box_seam_detection.core import corner_checks
    gray = np.full((300, 300), 150.0)
    gray[95:105, :] = 60.0  # strong edge band at y~100
    gray[150:160, :] = 60.0  # second parallel band 55 px away
    coarse = np.array([[20, 100], [280, 100], [280, 250], [20, 250]], float)
    rep = corner_checks.check_corners(gray, coarse, coarse, [1280, 720])
    c = rep["corners"]["back_left"]
    assert c["ambiguous_frac"] is not None and c["ambiguous_frac"] > 0.3
    assert "ambiguous_edge" not in c["flags"]
    rep2 = corner_checks.check_corners(
        gray, coarse, coarse, [1280, 720],
        corner_cfg={"flag_ambiguous": True})
    assert "ambiguous_edge" in rep2["corners"]["back_left"]["flags"]


def test_corner_depth_reports_without_flag_by_default():
    """Depth membership values persist; off-plane only flags when
    corner_cfg flag_depth is true (default false: too noisy at tips)."""
    from box_seam_detection.core import corner_checks
    gray = np.full((100, 100), 150.0)
    coarse = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    uv = np.array([[10, 10], [11, 10], [90, 10]])
    res = np.array([50.0, 60.0, 0.0])  # first corner far off-plane
    rep = corner_checks.check_corners(gray, coarse, coarse, [1280, 720],
                                      uv_win=uv, res_win=res)
    c = rep["corners"]["back_left"]
    assert c["depth_frac"] == 0.0 and c["depth_n"] == 2
    assert "depth_off_plane" not in c["flags"]
    rep2 = corner_checks.check_corners(
        gray, coarse, coarse, [1280, 720], uv_win=uv, res_win=res,
        corner_cfg={"flag_depth": True})
    assert "depth_off_plane" in rep2["corners"]["back_left"]["flags"]


def test_recheck_record_recomputes_corners_without_sam3(tmp_path):
    """recheck_record: reseam + corner gates from stored corners; flags
    persist in checks.corner_flags and set needs_review, never reject."""
    from box_seam_detection.core.run import recheck_record
    cfg = default_config()
    bgr, quad = _tape_face(None)
    p = tmp_path / "face.png"
    cv2.imwrite(str(p), bgr)
    names = ("back_left", "back_right", "front_right", "front_left")
    corners = {n: [round(float(quad[i][0]), 1), round(float(quad[i][1]), 1)]
               for i, n in enumerate(names)}
    coarse = [[c[0] + 30.0, c[1]] for c in quad]  # force large_shift
    src = {"schema": "box_autolabel/v5", "capture_id": "synth",
           "image": str(p), "image_size": [640, 480],
           "provenance": {"mask_source": "sam3_live", "config": {"old": 1},
                          "code_git_rev": "abc123",
                          "attempts": [{"source": "sam3_live",
                                        "coarse": coarse,
                                        "status": "accepted"}]},
           "top_face": {"corners_px": corners,
                        "corners_xyz_camera_m": None, "plane": None},
           "side_faces": [],
           "seam": {"status": "ambiguous", "endpoints_px": None,
                    "axis": None, "score": None},
           "checks": {"rgb_depth_quad_iou": None,
                      "wrong_face_suspect": False, "needs_review": False,
                      "edge_residual_rms_px": 1.0, "support_score": 0.9,
                      "reasons": []},
           "confidence": 0.5, "confidence_quad": 0.5, "confidence_seam": 0.0,
           "overlay": "old.png"}
    man = tmp_path / "m.json"
    man.write_text("{}")
    rec, debug = recheck_record(
        src, {"image": str(p), "manifest": str(man)}, cfg, "srcdir")
    assert rec["top_face"] == src["top_face"]
    assert rec["provenance"]["recheck"]["from"] == "srcdir"
    cf = rec["checks"]["corner_flags"]
    assert cf["any_flag"] is True
    assert rec["checks"]["needs_review"] is True
    assert any("corner_flags(" in r for r in rec["checks"]["reasons"])
    assert rec["top_face"]["corners_px"] is not None  # never rejected


def test_overlay_draws_estimate_candidates_and_source():
    """Overlay with an estimate block draws without error and labels the
    winning source; failed records still render."""
    bgr = np.zeros((120, 160, 3), np.uint8)
    rec = {"capture_id": "x", "confidence": 0.5,
           "provenance": {"mask_source": "sam3_live"},
           "top_face": {"corners_px": {
               "back_left": [10, 10], "back_right": [150, 10],
               "front_right": [150, 110], "front_left": [10, 110]}},
           "seam": {"status": "ok",
                    "endpoints_px": {"seam_left": [10, 60],
                                     "seam_right": [150, 60]},
                    "axis": "lr", "score": 0.4,
                    "source": "observed_gap",
                    "estimate": {"axis": "lr", "candidates": [
                        {"axis": "lr", "endpoints_px": {
                            "seam_left": [10, 60],
                            "seam_right": [150, 60]}},
                        {"axis": "bf", "endpoints_px": {
                            "seam_left": [80, 10],
                            "seam_right": [80, 110]}}]}},
           "checks": {"rgb_depth_quad_iou": None,
                      "wrong_face_suspect": False, "reasons": []}}
    out = draw_overlay(bgr, rec, {})
    assert out.shape == bgr.shape
    # Magenta final seam pixel at the line centre.
    assert (out[60, 80] == np.array([255, 0, 255])).all()


def test_reseam_skips_fold_for_already_folded_source(tmp_path, monkeypatch):
    """A source record already carrying provenance.corner_fold is never
    folded again; the skip is recorded in provenance.reseam.corner_fold."""
    from box_seam_detection.core import run as run_mod
    cfg = default_config()
    assert cfg["corner_fold"]["enabled"]
    bgr, quad = _tape_face(None)
    p = tmp_path / "face.png"
    cv2.imwrite(str(p), bgr)
    names = ("back_left", "back_right", "front_right", "front_left")
    corners = {n: [round(float(quad[i][0]), 1), round(float(quad[i][1]), 1)]
               for i, n in enumerate(names)}
    src = {"schema": "box_autolabel/v5", "capture_id": "synth",
           "image": str(p), "image_size": [640, 480],
           "provenance": {"mask_source": "sam3_live",
                          "corner_fold": {"changed": True,
                                          "config": {"hidden_mode": "keep"}}},
           "top_face": {"corners_px": corners,
                        "corners_xyz_camera_m": None, "plane": None},
           "side_faces": [],
           "seam": {"status": "ambiguous", "endpoints_px": None,
                    "axis": None, "score": None},
           "checks": {"reasons": []},
           "confidence": 0.5, "confidence_quad": 0.5, "confidence_seam": 0.0,
           "overlay": None}

    def _boom(*a, **k):
        raise AssertionError("fold_record must not run on a folded source")
    monkeypatch.setattr(run_mod, "fold_record", _boom)
    rec, _ = run_mod.reseam_record(src, {"image": str(p)}, cfg, "srcdir")
    assert rec["top_face"]["corners_px"] == corners
    assert rec["provenance"]["corner_fold"] == src["provenance"]["corner_fold"]
    assert rec["provenance"]["reseam"]["corner_fold"].startswith(
        "skipped_already_folded")
    assert "hidden_mode=keep" in rec["provenance"]["reseam"]["corner_fold"]
