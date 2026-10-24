# Provenance: ported from rgbd_captures/tests/test_autolabel_v2_alternates.py (HEAD 6df5994); imports made package-relative.
"""SAM3 alternate-candidate fallback: worker returns ``alternates``; the
pipeline tries them (as sam3_live candidates) only after the primary fails."""
import contextlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


from box_seam_detection.core import pipeline as _p
from box_seam_detection.core.config import default_config

PRIMARY = [[10.0, 10.0], [70.0, 10.0], [70.0, 50.0], [10.0, 50.0]]
ALT1 = [[20.0, 12.0], [60.0, 12.0], [60.0, 30.0], [20.0, 30.0]]
ALT2 = [[22.0, 14.0], [58.0, 14.0], [58.0, 28.0], [22.0, 28.0]]


# ---- worker ----------------------------------------------------------------

class _T:
    """Minimal tensor stand-in (detach/cpu/numpy/iteration)."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __iter__(self):
        return iter(self.a)

    def __len__(self):
        return len(self.a)


def _fake_torch():
    m = types.ModuleType("torch")
    m.bfloat16 = "bf16"
    m.inference_mode = lambda: contextlib.nullcontext()
    m.autocast = lambda *a, **k: contextlib.nullcontext()
    return m


def _run_worker(monkeypatch, tmp_path, scores, rects):
    from box_seam_detection.core import sam3_worker as w
    H, W = 200, 300
    img = np.full((H, W, 3), (60, 110, 170), np.uint8)  # BGR brown-ish
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    masks = np.zeros((len(rects), H, W), np.uint8)
    boxes = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        masks[i, y0:y1, x0:x1] = 1
        boxes.append([x0, y0, x1, y1])

    class _Proc:
        def __call__(self, images, text, return_tensors):
            return {}

        def post_process_instance_segmentation(self, outputs, threshold,
                                               target_sizes):
            return [{"scores": _T(scores), "boxes": _T(boxes),
                     "masks": _T(masks)}]

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    return w._handle_one(lambda **k: None, _Proc(), "cpu",
                         {"id": 1, "image": str(p), "threshold": 0.5})


def test_worker_returns_alternates(monkeypatch, tmp_path):
    # det0 = top face (0.668), det1 = whole box (0.742, winner),
    # det2 = tiny speck (fails area gate), det3 = another lower-score box.
    rects = [(50, 50, 150, 120), (50, 50, 150, 170), (5, 5, 8, 8),
             (160, 60, 240, 130)]
    resp = _run_worker(monkeypatch, tmp_path, [0.668, 0.742, 0.65, 0.6],
                       rects)
    assert resp["ok"], resp
    assert resp["picked"] == 1
    assert resp["concept_score"] == 0.742
    alts = resp["alternates"]
    assert [a["det_index"] for a in alts] == [0, 3]  # desc score, gate-passed
    assert [a["concept_score"] for a in alts] == [0.668, 0.6]
    for a in alts:
        assert len(a["corners_px"]) == 4 and len(a["corners_px"][0]) == 2
        assert {"box_xyxy", "mask_quad_iou"} <= set(a)
    # winner fields unchanged: alternates did not mutate detections
    assert len(resp["detections"]) == 4
    assert all("rejected" not in d for d in resp["detections"][:2])
    xs = [c[0] for c in alts[0]["corners_px"]]
    assert min(xs) == 50 and max(xs) >= 148


def test_worker_single_detection_has_empty_alternates(monkeypatch, tmp_path):
    resp = _run_worker(monkeypatch, tmp_path, [0.8], [(50, 50, 150, 120)])
    assert resp["ok"], resp
    assert resp["alternates"] == []


# ---- pipeline --------------------------------------------------------------

class _Session:
    checkpoint_sha256 = "sha"
    model_revision = "rev"

    def __init__(self, alternates):
        self.alternates = alternates

    def query(self, image_path, timeout_s=300):
        resp = {"ok": True, "id": 1, "corners_px": PRIMARY,
                "concept_score": 0.742, "prompt": "cardboard box top face",
                "n_detections": 3, "picked": 0, "box_xyxy": [10, 10, 70, 50]}
        if self.alternates is not None:
            resp["alternates"] = self.alternates
        return resp


def _alts():
    return [{"corners_px": ALT1, "concept_score": 0.668,
             "box_xyxy": [20, 12, 60, 30], "mask_quad_iou": 0.97,
             "det_index": 1},
            {"corners_px": ALT2, "concept_score": 0.6,
             "box_xyxy": [22, 14, 58, 28], "mask_quad_iou": 0.95,
             "det_index": 2}]


def test_gather_appends_alternates_after_primary():
    cands = _p._gather_coarse_candidates(
        "cid", "auto", "/tmp/x.png", _Session(_alts()), "/nonexistent", None,
        "depth_no_top", default_config())
    assert [c["source"] for c in cands] == [
        "sam3_live", "sam3_live", "sam3_live", "sam3_archived", "depth_only"]
    assert cands[0].get("alternate_rank") is None
    assert [c.get("alternate_rank") for c in cands[1:3]] == [1, 2]
    assert cands[1]["concept_score"] == 0.668
    assert cands[1]["prov"]["mask_source"] == "sam3_live"
    np.testing.assert_allclose(cands[1]["coarse"], ALT1)


def test_gather_no_alternates_unchanged():
    for alts in (None, []):
        cands = _p._gather_coarse_candidates(
            "cid", "auto", "/tmp/x.png", _Session(alts), "/nonexistent",
            None, "depth_no_top", default_config())
        assert [c["source"] for c in cands] == [
            "sam3_live", "sam3_archived", "depth_only"]


def _setup(tmp_path):
    img = np.zeros((60, 80, 3), np.uint8)
    img[:] = (60, 110, 170)
    ip = tmp_path / "x.png"
    cv2.imwrite(str(ip), img)
    man = {"schema": "rgbd_capture/v1", "capture_id": "x",
           "image": {"path": str(ip), "width": 80, "height": 60}}
    mp = tmp_path / "x.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    return {"image": str(ip), "pcd": str(tmp_path / "x.pcd"),
            "manifest": str(mp)}


def _patch_chain(monkeypatch, wrong_quads):
    """Refine accepts everything (returns the coarse quad); the depth cross
    check flags the quads whose first corner is in ``wrong_quads``."""
    def refine(gray, coarse, cfg):
        r = types.SimpleNamespace(accepted=True, reasons=[],
                                  corners_px=np.asarray(coarse, float))
        return r, 0.5, 0.9

    def cross(quad, manifest, depth_est, depth_err, W, H, cfg, **kw):
        wrong = tuple(np.asarray(quad)[0].tolist()) in wrong_quads
        return ({"wrong_face": wrong, "iou": 0.1 if wrong else 0.9,
                 "reasons": ["iou_low"] if wrong else []}, None)

    monkeypatch.setattr(_p, "_refine_once", refine)
    monkeypatch.setattr(_p, "check_depth_cross", cross)


def test_alternates_tried_only_after_primary_rejected(monkeypatch, tmp_path):
    paths = _setup(tmp_path)
    calls = []
    _patch_chain(monkeypatch, {(10.0, 10.0), (20.0, 12.0), (22.0, 14.0)})
    rec, _ = _p.process_capture("x", paths, default_config(),
                                source="sam3_live",
                                session=_Session(_alts()))
    att = rec["provenance"]["attempts"]
    assert [a["source"] for a in att] == ["sam3_live"] * 3
    assert [a.get("alternate_rank") for a in att] == [None, 1, 2]
    assert [a["status"] for a in att] == ["depth_cross_rejected"] * 3
    assert rec["top_face"]["corners_px"] is None


def test_alternate_wins_and_is_flagged(monkeypatch, tmp_path):
    paths = _setup(tmp_path)
    _patch_chain(monkeypatch, {(10.0, 10.0)})
    # Downstream (seam etc.) is out of scope: stop right after the chain.
    seen = {}

    real = _p.compute_confidence_quad

    def stop(concept_score, *a, **k):
        seen["score"] = concept_score
        raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(_p, "compute_confidence_quad", stop)
    try:
        _p.process_capture("x", paths, default_config(), source="sam3_live",
                           session=_Session(_alts()))
    except _Stop:
        pass
    assert seen["score"] == 0.668  # alternate 1 won and supplied its score


def test_primary_pass_never_touches_alternates(monkeypatch, tmp_path):
    """Primary accepted -> attempts hold exactly one entry, no alternate keys;
    a session with/without alternates yields the same attempts."""
    paths = _setup(tmp_path)
    _patch_chain(monkeypatch, set())
    seen = {}

    class _Stop(Exception):
        pass

    def stop(concept_score, *a, **k):
        seen["score"] = concept_score
        raise _Stop()

    monkeypatch.setattr(_p, "compute_confidence_quad", stop)
    calls = []
    orig = _p.check_depth_cross

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(_p, "check_depth_cross", spy)
    for alts in (None, _alts()):
        calls.clear()
        try:
            _p.process_capture("x", paths, default_config(),
                               source="sam3_live", session=_Session(alts))
        except _Stop:
            pass
        assert seen["score"] == 0.742
        assert len(calls) == 1
