# Provenance: ported from rgbd_captures/tests/test_autolabel_v2_roi_retry.py (HEAD 6df5994); imports made package-relative.
"""Depth cross-check ROI retry: a same-height surface elsewhere in the frame
must not decide the IoU arm (synthetic dense cloud, no dataset reads)."""
import sys
import types
from pathlib import Path

import numpy as np


from box_seam_detection.core.autolabel import depth_top as _depth
from box_seam_detection.core import pipeline as _p
from box_seam_detection.core.config import default_config

W, H = 640, 480
FX = FY = 500.0
CX, CY = 320.0, 240.0
CAM = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}
Z_TABLE, Z_TOP = 0.80, 0.714      # top 86 mm above the table (nearer camera)
BOX = (220, 180, 330, 300)        # x0, y0, x1, y1 (pixels)
DESK = (480, 40, 630, 440)        # same-height distractor plateau


def _cloud(with_box=True, with_desk=True):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = np.full(uu.shape, Z_TABLE)
    def paint(r):
        m = (uu >= r[0]) & (uu < r[2]) & (vv >= r[1]) & (vv < r[3])
        z[m] = Z_TOP
    if with_desk:
        paint(DESK)
    if with_box:
        paint(BOX)
    x = (uu - CX) / FX * z
    y = (vv - CY) / FY * z
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _cfg():
    cfg = default_config()
    return cfg


def _rect(r):
    x0, y0, x1, y1 = r
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _patch_plane(monkeypatch, ratio=0.99, p95=1.0):
    est = types.SimpleNamespace(rms_mm=1.0, n_inliers=5000, inlier_ratio=ratio)
    gp = types.SimpleNamespace(
        status="OK", reasons=[], estimate=est,
        plane=types.SimpleNamespace(normal=np.array([0, 0, 1.0]), d=-Z_TOP),
        camera_corners_xyz=np.zeros((4, 3)), erosion_px=0)
    from box_seam_detection.core.pbd import sam3_depth_fusion as fusion
    monkeypatch.setattr(fusion, "fit_sam3_guided_depth_plane",
                        lambda *a, **k: gp)
    monkeypatch.setattr(_p, "_plane_p95_all_mm", lambda *a, **k: p95)


def test_global_estimate_is_fooled_by_desk():
    """Sanity of the scene: the global estimate lands on the desk."""
    cfg = _cfg()
    est = _depth.estimate_top(_cloud(), CAM, (H, W), cfg["depth_top"])
    assert est["ok"]
    cx = float(np.asarray(est["quad"])[:, 0].mean())
    assert cx > 450, est["quad"]


def test_roi_retry_finds_box(monkeypatch):
    cfg = _cfg()
    xyz = _cloud()
    est = _depth.estimate_top(xyz, CAM, (H, W), cfg["depth_top"])
    _patch_plane(monkeypatch)
    ctx = {"xyz": xyz, "cam": CAM}
    cross, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg,
                                    depth_ctx=ctx)
    r = cross["depth_top_roi_retry"]
    assert r["used"] and r["result"] == "ok"
    assert r["global_quad_centroid"][0] > 450
    assert cross["iou"] is not None and cross["iou"] > 0.6
    assert not cross["wrong_face"]
    assert "depth_top_elsewhere" not in cross["reasons"]


def test_no_ctx_keeps_old_path(monkeypatch):
    cfg = _cfg()
    xyz = _cloud()
    est = _depth.estimate_top(xyz, CAM, (H, W), cfg["depth_top"])
    _patch_plane(monkeypatch)
    cross, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg)
    assert "depth_top_roi_retry" not in cross
    assert cross["iou"] == 0.0 and cross["wrong_face"]


def test_overlapping_global_estimate_takes_old_path(monkeypatch):
    cfg = _cfg()
    xyz = _cloud(with_desk=False)
    est = _depth.estimate_top(xyz, CAM, (H, W), cfg["depth_top"])
    _patch_plane(monkeypatch)
    a, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg)
    b, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg,
                                depth_ctx={"xyz": xyz, "cam": CAM})
    assert "depth_top_roi_retry" not in b
    assert a["iou"] == b["iou"] and a["iou"] > 0.6
    assert a["reasons"] == b["reasons"]


def test_retry_failure_makes_iou_unavailable_not_wrong_face(monkeypatch):
    """No box in the ROI (flat table): IoU None + depth_top_elsewhere, and
    with good plane support it is NOT wrong_face."""
    cfg = _cfg()
    xyz = _cloud(with_box=False)
    est = _depth.estimate_top(xyz, CAM, (H, W), cfg["depth_top"])
    assert est["ok"]
    _patch_plane(monkeypatch)
    cross, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg,
                                    depth_ctx={"xyz": xyz, "cam": CAM})
    assert cross["iou"] is None
    assert "depth_top_elsewhere" in cross["reasons"]
    assert cross["depth_top_roi_retry"]["result"] in ("failed", "no_overlap")
    assert not cross["wrong_face"]


def test_plane_arms_still_reject_top_plus_front(monkeypatch):
    """Top+front quad: even when the IoU arm is unavailable, the plane
    support arms fire wrong_face exactly as before."""
    cfg = _cfg()
    xyz = _cloud(with_box=False)
    est = _depth.estimate_top(xyz, CAM, (H, W), cfg["depth_top"])
    _patch_plane(monkeypatch, ratio=0.77, p95=50.0)
    cross, _ = _p.check_depth_cross(_rect(BOX), {}, est, None, W, H, cfg,
                                    depth_ctx={"xyz": xyz, "cam": CAM})
    assert cross["iou"] is None
    assert cross["wrong_face"]
    assert any(r.startswith("low_plane_inlier_ratio") for r in cross["reasons"])
    assert any(r.startswith("high_plane_p95") for r in cross["reasons"])
