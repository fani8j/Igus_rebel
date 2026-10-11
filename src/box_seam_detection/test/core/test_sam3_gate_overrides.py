# Provenance: ported from rgbd_captures/tests/test_sam3_gate_overrides.py (HEAD 6df5994); imports made package-relative.
"""sam3_worker per-request sanity-gate overrides (stub model, no GPU)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

CORE = Path(__file__).resolve().parents[2] / "box_seam_detection" / "core"
_spec = importlib.util.spec_from_file_location(
    "sam3_worker_gate_t", str(CORE / "sam3_worker.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

H, W = 100, 200


class _Proc:
    """Stub processor: three rectangular masks of different area / colour."""

    def __init__(self, masks, scores):
        self.masks, self.scores = masks, scores

    def __call__(self, images, text, return_tensors):
        return {"x": torch.zeros(1)}

    def post_process_instance_segmentation(self, outputs, threshold,
                                           target_sizes):
        boxes = torch.tensor([[0., 0., 10., 10.]] * len(self.masks))
        return [{"scores": torch.tensor(self.scores), "boxes": boxes,
                 "masks": torch.tensor(np.stack(self.masks))}]


class _Model:
    def __call__(self, **kw):
        return None


def _mask(y0, y1, x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


@pytest.fixture()
def img(tmp_path):
    # left half brown, right half blue (BGR)
    a = np.zeros((H, W, 3), np.uint8)
    a[:, :100] = (40, 90, 150)      # brown-ish
    a[:, 100:] = (200, 60, 20)      # blue
    p = tmp_path / "im.png"
    cv2.imwrite(str(p), a)
    return str(p)


def _run(img, masks, scores, **extra):
    req = {"id": 1, "image": img, "prompts": ["p"], **extra}
    return w._handle_one(_Model(), _Proc(masks, scores), "cpu", req)


def test_defaults_unchanged(img):
    big = _mask(0, 100, 0, 200)         # area 1.0 > 0.25
    ok = _mask(10, 60, 10, 90)          # brown, area 0.2
    r = _run(img, [big, ok], [0.9, 0.5])
    assert r["ok"] and r["picked"] == 1
    assert r["detections"][0]["rejected"] == "area_out_of_range"


def test_max_area_override_admits_big_mask(img):
    big = _mask(0, 100, 0, 200)
    ok = _mask(10, 60, 10, 90)
    r = _run(img, [big, ok], [0.9, 0.5], gates={"max_area_frac": 1.0})
    assert r["ok"] and r["picked"] == 0
    assert "rejected" not in r["detections"][0]


def test_min_area_override_rejects_small(img):
    ok = _mask(10, 60, 10, 90)          # 0.2
    r = _run(img, [ok], [0.9], gates={"min_area_frac": 0.22})
    assert not r["ok"] and r["reason"] == "all_detections_failed_sanity_gate"
    assert r["detections"][0]["rejected"] == "area_out_of_range"


def test_brown_override(img):
    blue = _mask(10, 60, 110, 190)      # not brown
    r = _run(img, [blue], [0.9])
    assert not r["ok"] and r["detections"][0]["rejected"] == "brown_too_low"
    r = _run(img, [blue], [0.9], gates={"min_brown_frac": 0.0})
    assert r["ok"]


def test_alternates_use_overrides(img):
    a = _mask(10, 60, 10, 90)
    b = _mask(0, 100, 0, 200)
    r = _run(img, [a, b], [0.9, 0.5])
    assert r["ok"] and not r.get("alternates")
    r2 = _run(img, [a, b], [0.9, 0.5], gates={"max_area_frac": 1.0})
    assert r2["ok"]
    assert [x["det_index"] for x in r2.get("alternates", [])] == [1]


@pytest.mark.parametrize("gates", [
    "x", {"bogus": 1}, {"min_area_frac": "abc"}, {"min_area_frac": -1},
    {"min_area_frac": 0.5, "max_area_frac": 0.1}])
def test_bad_gates_fail_cleanly(img, gates):
    r = _run(img, [_mask(0, 10, 0, 10)], [0.9], gates=gates)
    assert r["ok"] is False and r["id"] == 1


def test_resolve_defaults_and_none():
    d = (w.MIN_MASK_AREA_FRAC, w.MAX_MASK_AREA_FRAC, w.MIN_MASK_BROWN_FRAC)
    assert w._resolve_gates({}) [:3] == d
    assert w._resolve_gates({"gates": {}})[:3] == d
    assert w._resolve_gates({"gates": {"max_area_frac": None}})[:3] == d
    assert w._resolve_gates({"gates": {"max_area_frac": 0.6}})[1] == 0.6


def test_daemon_passes_request_dict_through():
    src = (CORE / "sam3_daemon.py").read_text()
    assert "sam3_worker._handle_one(\n            model, processor, device, req)" in src
