# Provenance: ported from rgbd_captures/tests/test_autolabel_v2_plane_trim.py (HEAD 6df5994); imports made package-relative.
"""Depth-plane-trimmed fallback outline: a SAM3 quad that spills onto a side
face is re-derived from the cloud points on the top plane (synthetic scene,
no dataset reads, no SAM3)."""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


from box_seam_detection.core import pipeline as _p
from box_seam_detection.core.config import default_config

W, H = 640, 480
FX = FY = 500.0
CX, CY = 320.0, 240.0
CAM = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}
Z_TABLE, Z_TOP = 0.80, 0.714
TOP = (220, 180, 330, 300)         # true top face (pixels)
SIDE_ROWS = (300, 360)             # visible side face below the front edge
PLANE = types.SimpleNamespace(normal=np.array([0, 0, 1.0]), d=-Z_TOP)
TRUE_TOP = np.array([[220, 180], [330, 180], [330, 300], [220, 300]], float)


def _cloud():
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = np.full(uu.shape, Z_TABLE)
    in_x = (uu >= TOP[0]) & (uu < TOP[2])
    z[in_x & (vv >= TOP[1]) & (vv < TOP[3])] = Z_TOP
    side = in_x & (vv >= SIDE_ROWS[0]) & (vv < SIDE_ROWS[1])
    z[side] = Z_TOP + (vv[side] - SIDE_ROWS[0]) / 60.0 * (Z_TABLE - Z_TOP)
    x = (uu - CX) / FX * z
    y = (vv - CY) / FY * z
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _spilled(d):
    """Coarse quad whose front-right corner sits d px down the side face."""
    return np.array([[220, 180], [330, 180], [330, 300 + d], [220, 300]],
                    float)


def _ray_xyz(quad):
    """Pixel corners back-projected onto the top plane (like the guided fit)."""
    return np.array([[(u - CX) / FX * Z_TOP, (v - CY) / FY * Z_TOP, Z_TOP]
                     for u, v in quad])


def _ordered(q):
    q = np.asarray(q, float)
    return q[np.lexsort((q[:, 1], q[:, 0]))]


# ---- plane_trimmed_quad ------------------------------------------------------

def test_trim_recovers_top_face_from_spilled_quad():
    ctx = {"xyz": _cloud(), "cam": CAM}
    quad, info = _p.plane_trimmed_quad(_spilled(40), PLANE, ctx, W, H)
    assert quad is not None, info
    assert info["n_points"] > 5000 and info["hull_area_px"] > 10000
    # The whole side face is off the plane: no trimmed corner may sit on it.
    assert quad[:, 1].max() < 300 + 3
    assert np.abs(_ordered(quad) - _ordered(TRUE_TOP)).max() < 3.0


def test_trim_without_plane_or_cloud_is_none():
    assert _p.plane_trimmed_quad(TRUE_TOP, None, {"xyz": _cloud(),
                                                  "cam": CAM}, W, H)[0] is None
    assert _p.plane_trimmed_quad(TRUE_TOP, PLANE, None, W, H)[0] is None
    far = np.array([[500, 400], [600, 400], [600, 470], [500, 470]], float)
    q, info = _p.plane_trimmed_quad(far, PLANE, {"xyz": _cloud(),
                                                 "cam": CAM}, W, H)
    assert q is None and "reason" in info


def test_rect_diffs_threshold_geometry():
    sq = np.array([[0, 0, 1], [0.1, 0, 1], [0.1, 0.1, 1], [0, 0.1, 1.0]])
    assert _p._rect_diffs_mm(sq) == (0.0, 0.0) or max(
        _p._rect_diffs_mm(sq)) < 1e-9
    stretched = sq.copy()
    stretched[2] += [0.0, 0.0151, 0.0]  # right side +15.1 mm
    assert max(_p._rect_diffs_mm(stretched)) > _p.GROSS_RECT_MM
    assert _p._rect_diffs_mm(None) is None
    bad = sq.copy()
    bad[0, 0] = np.nan
    assert _p._rect_diffs_mm(bad) is None


# ---- pipeline wiring -----------------------------------------------------------

def _setup(tmp_path):
    img = np.full((H, W, 3), (60, 110, 170), np.uint8)
    ip = tmp_path / "x.png"
    cv2.imwrite(str(ip), img)
    man = {"schema": "rgbd_capture/v1", "capture_id": "x",
           "image": {"path": str(ip), "width": W, "height": H}}
    mp = tmp_path / "x.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    return {"image": str(ip), "pcd": str(tmp_path / "x.pcd"),
            "manifest": str(mp)}


class _Session:
    checkpoint_sha256 = "sha"
    model_revision = "rev"

    def __init__(self, quad):
        self.quad = quad

    def query(self, image_path, timeout_s=300):
        return {"ok": True, "id": 1,
                "corners_px": np.asarray(self.quad).tolist(),
                "concept_score": 0.8, "prompt": "cardboard box top face",
                "n_detections": 1, "picked": 0, "box_xyxy": [0, 0, 1, 1]}


def _patch(monkeypatch, reject_spilled=False):
    from box_seam_detection.core.autolabel import depth_top as _depth
    from box_seam_detection.core.autolabel import pcd as _pcd

    xyz = _cloud()
    monkeypatch.setattr(_depth, "load_capture_camera",
                        lambda m: (CAM, "synthetic"))
    monkeypatch.setattr(_pcd, "read_pcd_xyz", lambda p: (xyz, None))
    monkeypatch.setattr(_depth, "estimate_top",
                        lambda *a, **k: {"ok": False, "reason": "synthetic"})

    def refine(gray, coarse, cfg):
        r = types.SimpleNamespace(accepted=True, reasons=[],
                                  corners_px=np.asarray(coarse, float))
        return r, 0.5, 0.9

    calls = []

    def cross(quad, manifest, depth_est, depth_err, W_, H_, cfg, **kw):
        quad = np.asarray(quad, float)
        calls.append(quad.copy())
        spill = float(quad[:, 1].max()) > 300 + 3
        wrong = bool(reject_spilled and spill)
        out = {"wrong_face": wrong, "iou": 0.9, "reasons": [],
               "plane": PLANE, "plane_rms_mm": 1.0, "plane_inliers": 5000,
               "plane_inlier_ratio": 0.97, "plane_p95_all_mm": 1.0,
               "corners_xyz": _ray_xyz(quad), "depth_quad": None}
        if wrong:
            out["reasons"] = ["low_plane_inlier_ratio(0.77)",
                              "wrong_face_suspect"]
        return out, None

    monkeypatch.setattr(_p, "_refine_once", refine)
    monkeypatch.setattr(_p, "check_depth_cross", cross)
    return calls


def _run(tmp_path, quad):
    return _p.process_capture("x", _setup(tmp_path), default_config(),
                              source="sam3_live", session=_Session(quad))


def test_gross_rect_triggers_trim_and_improves(monkeypatch, tmp_path):
    calls = _patch(monkeypatch)
    rec, _ = _run(tmp_path, _spilled(40))     # 40 px * 1.43 = 57 mm diff
    pt = rec["provenance"]["sam3"]["plane_trimmed"]
    assert pt["from"] == "primary" and pt["why"] == "gross_rect"
    assert pt["n_points"] > 5000 and pt["hull_area_px"] > 10000
    assert pt["original_rect_diffs_mm"][1] > _p.GROSS_RECT_MM
    assert max(pt["rect_diffs_mm"]) < 3.0
    assert rec["provenance"]["mask_source"] == "sam3_live"
    att = rec["provenance"]["attempts"]
    assert [a.get("plane_trimmed") for a in att] == [None, True]
    ys = [c[1] for c in rec["top_face"]["corners_px"].values()]
    assert max(ys) < 300 + 4
    assert len(calls) == 2


def test_below_gross_threshold_keeps_original(monkeypatch, tmp_path):
    calls = _patch(monkeypatch)
    rec, _ = _run(tmp_path, _spilled(8))      # 8 px * 1.43 = 11 mm < 15
    assert len(calls) == 1
    assert "plane_trimmed" not in rec["provenance"]["sam3"]
    assert len(rec["provenance"]["attempts"]) == 1


def test_just_above_gross_threshold_triggers(monkeypatch, tmp_path):
    calls = _patch(monkeypatch)
    _run(tmp_path, _spilled(12))              # 12 px * 1.43 = 17 mm > 15
    assert len(calls) == 2


def test_clean_primary_unchanged(monkeypatch, tmp_path):
    calls = _patch(monkeypatch)
    rec, _ = _run(tmp_path, TRUE_TOP)
    assert len(calls) == 1
    assert "plane_trimmed" not in rec["provenance"]["sam3"]
    assert [a["status"] for a in rec["provenance"]["attempts"]] == [
        "accepted"]


def test_all_rejected_falls_back_to_trim(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, reject_spilled=True)
    rec, _ = _run(tmp_path, _spilled(40))
    att = rec["provenance"]["attempts"]
    assert [a["status"] for a in att] == ["depth_cross_rejected", "accepted"]
    assert att[1]["plane_trimmed"] is True
    pt = rec["provenance"]["sam3"]["plane_trimmed"]
    assert pt["why"] == "all_rejected" and pt["from"] == "primary"
    assert rec["top_face"]["corners_px"] is not None
    assert rec["provenance"]["mask_source"] == "sam3_live"


def test_all_rejected_and_trim_fails_stays_failed(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, reject_spilled=True)
    # Quad far from any plane points -> no trimmed candidate.
    far = np.array([[500, 400], [600, 400], [600, 470], [500, 470]], float)
    far_cross = _p.check_depth_cross

    def always_wrong(quad, *a, **k):
        out, g = far_cross(quad, *a, **k)
        out["wrong_face"] = True
        out["reasons"] = ["wrong_face_suspect"]
        return out, g

    monkeypatch.setattr(_p, "check_depth_cross", always_wrong)
    rec, _ = _run(tmp_path, far)
    assert rec["top_face"]["corners_px"] is None
    assert "no_coarse_quad" in rec["checks"]["reasons"]


# ---- table-parallel plane choice (front face larger than top) ----------------

def _oblique_scene(box_h=0.12, elev_deg=44.0, step=0.003):
    """Table + box seen obliquely (view angle 90-elev from the table normal).
    Returns (cloud xyz cam-frame, cam dict, top_px(4,2), cand quad, front
    plane n/d in the camera frame, #top px, #front px)."""
    e = np.radians(elev_deg)
    dvec = np.array([0.0, np.cos(e), -np.sin(e)])
    cx_ = np.array([1.0, 0.0, 0.0])
    cy_ = np.cross(dvec, cx_)
    Rm = np.stack([cx_, cy_, dvec])
    p0 = np.array([0.0, 0.0, 0.04]) - 0.8 * dvec

    def cam(p):
        return (np.atleast_2d(p) - p0) @ Rm.T

    def grid(a, b, fix):
        A, B = np.meshgrid(np.arange(a[0], a[1] + 1e-9, step),
                           np.arange(b[0], b[1] + 1e-9, step))
        return fix(A.ravel(), B.ravel())

    hx, hy = 0.055, 0.049
    table = grid((-0.45, 0.45), (-0.35, 0.45),
                 lambda a, b: np.stack([a, b, np.zeros_like(a)], 1))
    ins = ~((np.abs(table[:, 0]) < hx) & (np.abs(table[:, 1]) < hy))
    table = table[ins]
    top = grid((-hx, hx), (-hy, hy),
               lambda a, b: np.stack([a, b, np.full_like(a, box_h)], 1))
    front = grid((-hx, hx), (0.0, box_h),
                 lambda a, z: np.stack([a, np.full_like(a, -hy), z], 1))
    xyz = cam(np.vstack([table, top, front]))
    cam_d = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}

    def proj(p):
        c = cam(p)
        return np.stack([c[:, 0] / c[:, 2] * FX + CX,
                         c[:, 1] / c[:, 2] * FY + CY], 1)

    top_c = np.array([[-hx, hy, box_h], [hx, hy, box_h],
                      [hx, -hy, box_h], [-hx, -hy, box_h]])
    top_px = proj(top_c)
    fr_c = np.array([[-hx, -hy, box_h], [hx, -hy, box_h],
                     [hx, -hy, 0.0], [-hx, -hy, 0.0]])
    # SAM3-style candidate spanning top + front: back edge of the top down to
    # the bottom edge of the front face.
    cand = np.vstack([top_px[:2], proj(fr_c)[2:]]).astype(float)
    ft = cam(np.array([0.0, -hy, 0.0]))[0]
    fn = Rm @ np.array([0.0, -1.0, 0.0])
    fplane = types.SimpleNamespace(normal=fn, d=-float(fn @ ft))
    def area(p):
        p = np.asarray(p, np.float32)
        return float(cv2.contourArea(cv2.convexHull(p)))
    return (xyz, cam_d, top_px, cand, fplane, area(top_px), area(proj(fr_c)))


def test_table_parallel_plane_beats_larger_front_face():
    xyz, cam_d, top_px, cand, fplane, a_top, a_front = _oblique_scene()
    assert a_front > a_top        # the dominant plane in the quad is the front
    ctx = {"xyz": xyz, "cam": cam_d}
    quad, info = _p.plane_trimmed_quad(cand, fplane, ctx, W, H)
    assert quad is not None, info
    pc = info["plane_choice"]
    assert pc["method"] == "table_parallel"
    assert pc["angle_deg"] < 3.0 and abs(pc["height_mm"] - 120.0) < 8.0
    assert len(pc["table_normal"]) == 3
    assert np.abs(_ordered(quad) - _ordered(top_px)).max() < 5.0


def test_dominant_plane_kept_when_nothing_is_table_parallel():
    xyz, cam_d, top_px, cand, fplane, _a, _b = _oblique_scene()
    # A "table" that matches neither the top nor the front face.
    tn = np.array([0.5, 0.5, 0.7071])
    tn /= np.linalg.norm(tn)
    ctx = {"xyz": xyz, "cam": cam_d, "_table": {"n": tn, "d": 0.8}}
    quad, info = _p.plane_trimmed_quad(cand, fplane, ctx, W, H)
    assert info["plane_choice"]["method"] == "dominant"
    assert info["plane_choice"]["table_normal"] is not None
    # no table at all -> dominant as well
    ctx2 = {"xyz": xyz, "cam": cam_d, "_table": None}
    _q, info2 = _p.plane_trimmed_quad(cand, fplane, ctx2, W, H)
    assert info2["plane_choice"] == {"method": "dominant",
                                     "table_normal": None,
                                     "angle_deg": None, "height_mm": None}


def test_already_table_parallel_dominant_plane_is_kept_unchanged():
    ctx = {"xyz": _cloud(), "cam": CAM}
    quad, info = _p.plane_trimmed_quad(_spilled(40), PLANE, ctx, W, H)
    assert info["plane_choice"]["method"] == "table_parallel"
    assert abs(info["plane_choice"]["height_mm"] - 86.0) < 3.0
    assert np.abs(_ordered(quad) - _ordered(TRUE_TOP)).max() < 3.0


def test_no_guided_plane_uses_table_parallel_plane_only():
    xyz, cam_d, top_px, cand, _fplane, _a, _b = _oblique_scene()
    ctx = {"xyz": xyz, "cam": cam_d}
    # refine-rejected candidate: no guided plane. Needs the explicit opt-in.
    assert _p.plane_trimmed_quad(cand, None, ctx, W, H)[0] is None
    quad, info = _p.plane_trimmed_quad(cand, None, ctx, W, H,
                                       allow_no_plane=True)
    assert quad is not None, info
    assert info["plane_choice"]["method"] == "table_parallel"
    assert np.abs(_ordered(quad) - _ordered(top_px)).max() < 5.0
    # no table-parallel plane available -> nothing (never a dominant plane)
    tn = np.array([0.5, 0.5, 0.7071])
    ctx2 = {"xyz": xyz, "cam": cam_d,
            "_table": {"n": tn / np.linalg.norm(tn), "d": 0.8}}
    q2, info2 = _p.plane_trimmed_quad(cand, None, ctx2, W, H,
                                      allow_no_plane=True)
    assert q2 is None and info2["reason"] == "no_table_parallel_plane"
