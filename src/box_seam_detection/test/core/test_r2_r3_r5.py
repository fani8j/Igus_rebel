# Provenance: ported from rgbd_captures/tests/test_r2_r3_r5.py (HEAD 6df5994); imports made package-relative.
"""R2 (high-yaw outline), R3 (k4/hexagon rectangle fit) and R5 (needs_review
clearing) of the box top-face detection (synthetic scenes; no dataset reads,
no SAM3)."""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


from box_seam_detection.core import fold_edge as _fold
from box_seam_detection.core import pipeline as _p
from box_seam_detection.core.config import default_config

W, H = 1280, 720
FX = FY = 1040.0
CX, CY = 640.0, 360.0
CAM = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}
MANIFEST_CAM = {"k": [FX, 0, CX, 0, FY, CY, 0, 0, 1.0]}


def make_scene(yaw_deg=0.0, box_h=0.12, hx=0.055, hy=0.049, elev_deg=44.0,
               step=0.0007):
    """Table (z=0) + box (footprint 2hx x 2hy rotated by yaw about z, height
    box_h) seen from an oblique camera. Returns a dict with the camera-frame
    cloud, true top corners (px, ring order), and a SAM3-style candidate that
    spans the top face plus the visible side walls ("spill")."""
    e = np.radians(elev_deg)
    dvec = np.array([0.0, np.cos(e), -np.sin(e)])
    cx_ = np.array([1.0, 0.0, 0.0])
    cy_ = np.cross(dvec, cx_)
    Rm = np.stack([cx_, cy_, dvec])
    p0 = np.array([0.0, 0.0, 0.04]) - 0.5 * dvec
    th = np.radians(yaw_deg)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def cam(p):
        return (np.atleast_2d(p) - p0) @ Rm.T

    def proj(p):
        c = cam(p)
        return np.stack([c[:, 0] / c[:, 2] * FX + CX,
                         c[:, 1] / c[:, 2] * FY + CY], 1)

    def rot(a, b):
        xy = np.stack([a, b], 1) @ Rz.T
        return xy[:, 0], xy[:, 1]

    A, B = np.meshgrid(np.arange(-0.25, 0.25 + 1e-9, step),
                       np.arange(-0.2, 0.25 + 1e-9, step))
    tx, ty = A.ravel(), B.ravel()
    # box-frame coordinates of the table points: drop the footprint
    bx, by = (np.stack([tx, ty], 1) @ Rz).T
    keep = ~((np.abs(bx) < hx) & (np.abs(by) < hy))
    table = np.stack([tx[keep], ty[keep], np.zeros(keep.sum())], 1)

    def face(u_rng, v_rng, fn):
        U, V = np.meshgrid(np.arange(u_rng[0], u_rng[1] + 1e-9, step),
                           np.arange(v_rng[0], v_rng[1] + 1e-9, step))
        return fn(U.ravel(), V.ravel())

    def top_fn(u, v):
        x, y = rot(u, v)
        return np.stack([x, y, np.full_like(x, box_h)], 1)

    top = face((-hx, hx), (-hy, hy), top_fn)
    walls = []
    for sx, sy, ax in ((0, -1, "x"), (-1, 0, "y"), (1, 0, "y"), (0, 1, "x")):
        if ax == "x":
            w = face((-hx, hx), (0.0, box_h),
                     lambda u, z, sy=sy: np.stack(
                         [*rot(u, np.full_like(u, sy * hy)), z], 1))
        else:
            w = face((-hy, hy), (0.0, box_h),
                     lambda v, z, sx=sx: np.stack(
                         [*rot(np.full_like(v, sx * hx), v), z], 1))
        # keep only walls whose outward normal faces the camera
        nrm = np.array([*(Rz @ np.array([sx, sy], float)), 0.0])
        cen = np.array([*(Rz @ np.array([sx * hx, sy * hy], float)),
                        box_h / 2])
        if float(nrm @ (p0 - cen)) > 0:
            walls.append(w)
    xyz = cam(np.vstack([table, top] + walls))
    ring = np.array([[-hx, hy], [hx, hy], [hx, -hy], [-hx, -hy]])
    top_px = proj(np.array([[*(Rz @ c), box_h] for c in ring]))
    foot_px = proj(np.array([[*(Rz @ c), 0.0] for c in ring]))
    # SAM3-style candidate: the image-lowest top corner is pulled down onto
    # the visible wall (the outline spills onto the side face), which makes
    # the quad grossly non-rectangular in 3-D (> r_keep_original_rect_mm).
    cand = top_px.copy()
    cand[int(np.argmax(top_px[:, 1])), 1] += 60.0
    spill_hull = cv2.convexHull(cand.astype(np.float32))
    return {"xyz": xyz, "top_px": top_px, "foot_px": foot_px, "cand": cand,
            "spill_hull": spill_hull.reshape(-1, 2).astype(float),
            "box_h": box_h, "dims_mm": (2000 * hx, 2000 * hy)}


def _ctx(scene):
    return {"xyz": scene["xyz"], "cam": CAM}


def _ordered(q):
    q = np.asarray(q, float).reshape(4, 2)
    return q[np.lexsort((q[:, 1], q[:, 0]))]


def _max_err(q, ref):
    """Max corner distance, minimised over cyclic start/orientation."""
    return _p._quad_shift_px(q, ref)


def _spill_quad(scene, extra=18.0):
    """Top face quad whose two near corners are pulled down onto the wall."""
    q = scene["top_px"].copy()
    d = q[:, 1].argsort()[2:]          # the two lowest corners in the image
    q[d, 1] += extra
    return q


# ---- R2: view geometry -----------------------------------------------------

def test_view_geometry_yaw_and_walls():
    s0 = make_scene(0.0)
    g0 = _p.box_view_geometry(s0["top_px"], MANIFEST_CAM, _ctx(s0))
    assert g0 is not None and g0["yaw45"] < 4.0
    s30 = make_scene(30.0)
    g30 = _p.box_view_geometry(s30["top_px"], MANIFEST_CAM, _ctx(s30))
    assert abs(g30["yaw45"] - 30.0) < 4.0
    rcfg = default_config()["reliability"]
    # square-on, centred: only the front wall faces the camera
    assert sum(c > rcfg["r2_wall_cos"] for c in g0["wall_cos"].values()) == 1
    assert view_trim(g0, rcfg) is False
    # yawed: two walls face the camera
    assert sum(c > rcfg["r2_wall_cos"] for c in g30["wall_cos"].values()) == 2
    assert view_trim(g30, rcfg) is True


def view_trim(geom, rcfg):
    return _p.view_needs_trim(geom, rcfg)[0]


def test_view_geometry_unavailable_without_table_or_camera():
    s = make_scene(30.0)
    assert _p.box_view_geometry(s["top_px"], MANIFEST_CAM, None) is None
    assert _p.box_view_geometry(s["top_px"], {}, _ctx(s)) is None
    ctx = _ctx(s)
    ctx["_table"] = None
    assert _p.box_view_geometry(s["top_px"], MANIFEST_CAM, ctx) is None
    assert _p.view_needs_trim(None, default_config()["reliability"]) == (
        False, [])


# ---- R3: orthogonal rectangle in the top plane ----------------------------------

def test_fit_plane_rectangle_recovers_top_from_hexagon_quad():
    for yaw in (0.0, 25.0):
        s = make_scene(yaw)
        rcfg = default_config()["reliability"]
        quad, info = _p.fit_plane_rectangle(s["cand"], MANIFEST_CAM,
                                            _ctx(s), W, H, rcfg)
        assert quad is not None, info
        assert _max_err(quad, s["top_px"]) < 10.0, (yaw, info)
        long_, short_ = info["rect_mm"]
        assert abs(long_ - 110.0) < 6.0 and abs(short_ - 98.0) < 6.0
        assert info["dims"]["nearest_class"] == "small"
        assert info["dims"]["within_tol"] is True
        assert info["plane_choice"]["method"] == "table_parallel"


def test_fit_plane_rectangle_rejects_unrelated_quad_and_missing_cloud():
    s = make_scene(0.0)
    rcfg = default_config()["reliability"]
    far = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], float)
    q, info = _p.fit_plane_rectangle(far, MANIFEST_CAM, _ctx(s), W, H, rcfg)
    assert q is None and "reason" in info
    assert _p.fit_plane_rectangle(s["cand"], MANIFEST_CAM, None, W, H,
                                  rcfg)[0] is None
    assert _p.fit_plane_rectangle(s["cand"], {}, _ctx(s), W, H,
                                  rcfg)[0] is None
    # the fitted rectangle must overlap the k4 quad
    strict = dict(rcfg, r3_min_iou=0.999)
    q2, info2 = _p.fit_plane_rectangle(s["cand"], MANIFEST_CAM, _ctx(s), W, H,
                                       strict)
    assert q2 is None and info2["reason"].startswith("rect_far_from_k4")


# ---- R5: independent top confirmation ----------------------------------------------

def test_independent_top_confirm():
    s = make_scene(20.0)
    rcfg = default_config()["reliability"]
    ok = _p.independent_top_confirm(s["top_px"], _ctx(s), W, H, rcfg)
    assert ok["confirmed"] is True and ok["iou"] >= rcfg["r5_min_iou"]
    assert ok["plane_choice"]["method"] == "table_parallel"
    # quad that includes the wall: not confirmed
    bad = _p.independent_top_confirm(_spill_quad(s, 60.0), _ctx(s), W, H,
                                     rcfg)
    assert bad["confirmed"] is False and bad["iou"] < ok["iou"]
    # nothing on the quad's location: no confirmation, never raises
    far = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], float)
    r = _p.independent_top_confirm(far, _ctx(s), W, H, rcfg)
    assert r["confirmed"] is False and "reason" in r
    assert _p.independent_top_confirm(s["top_px"], None, W, H,
                                      rcfg)["confirmed"] is False


# ---- fold outward guard --------------------------------------------------------------

def test_fold_outward_guard_blocks_move_onto_side_wall(monkeypatch):
    quad = np.array([[300, 200], [700, 200], [700, 500], [300, 500]], float)
    img = np.full((720, 1280), 120, np.uint8)
    plane = {"normal": [0.0, 0.0, 1.0], "offset_m": -0.7}

    def fake_edge(gray, corners, ei, km, n, d, height, c, scale, W_, H_):
        rec = {"method": "fold_shoulder", "w_pred_px": 20.0}
        if ei != 2:                      # only the (bottom) front edge moves
            return {"method": "keep"}, None
        p0, p1 = corners[2], corners[3]
        a = p1 - p0
        nv = np.array([-a[1], a[0]]) / np.linalg.norm(a)
        if nv @ (0.5 * (p0 + p1) - corners.mean(axis=0)) < 0:
            nv = -nv
        pt = p0 + nv * 5.0               # 5 px OUTWARD (toward the wall)
        return rec, np.array([nv[0], nv[1], -float(nv @ pt)])

    monkeypatch.setattr(_fold, "_process_edge", fake_edge)
    free = _fold.relocate_edges(img, quad, MANIFEST_CAM, plane, 0.1)
    assert free["changed"] is True
    assert free["corners_px"][2:, 1].min() > 504.0
    guarded = _fold.relocate_edges(img, quad, MANIFEST_CAM, plane, 0.1,
                                   outward_limit_px=1.5)
    assert guarded["changed"] is False
    assert guarded["per_edge"]["front"]["reason"] == \
        "outward_blocked_side_wall"
    # an inward move is not blocked
    def inward(gray, corners, ei, *a):
        rec, line = fake_edge(gray, corners, ei, *a)
        if line is not None:
            line = np.array([line[0], line[1], line[2] + 10.0])  # -> inward
        return rec, line

    monkeypatch.setattr(_fold, "_process_edge", inward)
    ok = _fold.relocate_edges(img, quad, MANIFEST_CAM, plane, 0.1,
                              outward_limit_px=1.5)
    assert ok["changed"] is True


# ---- pipeline wiring -------------------------------------------------------------------

def _setup(tmp_path):
    img = np.full((H, W, 3), (60, 110, 170), np.uint8)
    ip = tmp_path / "x.png"
    cv2.imwrite(str(ip), img)
    man = {"schema": "rgbd_capture/v1", "capture_id": "x",
           "image": {"path": str(ip), "width": W, "height": H},
           "camera_info": {"width": W, "height": H, "k": MANIFEST_CAM["k"]}}
    mp = tmp_path / "x.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    return {"image": str(ip), "pcd": str(tmp_path / "x.pcd"),
            "manifest": str(mp)}


class _Session:
    checkpoint_sha256 = "sha"
    model_revision = "rev"

    def __init__(self, quad, k4=False):
        self.quad, self.k4 = quad, k4

    def query(self, image_path, timeout_s=300):
        r = {"ok": True, "id": 1,
             "corners_px": np.asarray(self.quad).tolist(),
             "concept_score": 0.8, "prompt": "cardboard box top face",
             "n_detections": 1, "picked": 0, "box_xyxy": [0, 0, 1, 1]}
        if self.k4:
            r["quad_method"] = "min_area_k4"
        return r


def _patch(monkeypatch, scene, cross_iou=0.9, reject_when=None,
           corner_flag=False):
    from box_seam_detection.core.autolabel import depth_top as _depth
    from box_seam_detection.core.autolabel import pcd as _pcd

    xyz = scene["xyz"]
    monkeypatch.setattr(_depth, "load_capture_camera",
                        lambda m: (CAM, "synthetic"))
    monkeypatch.setattr(_pcd, "read_pcd_xyz", lambda p: (xyz, None))
    monkeypatch.setattr(_depth, "estimate_top",
                        lambda *a, **k: {"ok": False, "reason": "synthetic"})
    tn = np.array(_p.estimate_table_plane({"xyz": xyz, "cam": CAM})["n"])

    def refine(gray, coarse, cfg):
        r = types.SimpleNamespace(accepted=True, reasons=[],
                                  corners_px=np.asarray(coarse, float))
        return r, 0.5, 0.9

    seen = []

    def cross(quad, manifest, depth_est, depth_err, W_, H_, cfg, **kw):
        quad = np.asarray(quad, float)
        seen.append(quad.copy())
        wrong = bool(reject_when and reject_when(quad))
        # top plane of the true box: table-parallel at box height
        n = tn
        pts_h = float(scene["box_h"])
        d = _p.estimate_table_plane({"xyz": xyz, "cam": CAM})["d"] - pts_h
        plane = types.SimpleNamespace(normal=n, d=d)
        from box_seam_detection.core import fold_edge as F
        cxyz = F.corners_xyz_on_plane(quad, MANIFEST_CAM, plane)
        out = {"wrong_face": wrong, "iou": cross_iou, "reasons": [],
               "plane": plane, "plane_rms_mm": 1.0, "plane_inliers": 5000,
               "plane_inlier_ratio": 0.97, "plane_p95_all_mm": 1.0,
               "corners_xyz": cxyz, "depth_quad": None}
        if wrong:
            out["reasons"] = ["low_plane_inlier_ratio(0.5)",
                              "wrong_face_suspect"]
        return out, None

    monkeypatch.setattr(_p, "_refine_once", refine)
    monkeypatch.setattr(_p, "check_depth_cross", cross)
    if corner_flag:
        monkeypatch.setattr(
            _p._corner_checks, "check_corners",
            lambda *a, **k: {"any_flag": True, "corners": {
                "back_left": {"flags": ["large_shift"]}}})
    return seen


def _run(tmp_path, quad, k4=False, cfg=None):
    return _p.process_capture("x", _setup(tmp_path), cfg or default_config(),
                              source="sam3_live",
                              session=_Session(quad, k4=k4))


def _corners(rec):
    return np.array(list(rec["top_face"]["corners_px"].values()), float)


def test_r2_trims_spilled_high_yaw_quad_before_refine(monkeypatch, tmp_path):
    s = make_scene(30.0)
    seen = _patch(monkeypatch, s)
    rec, _ = _run(tmp_path, s["cand"])
    att = rec["provenance"]["attempts"]
    assert att[0]["r2_trim"] is True and att[0]["selected"] is True
    assert len(att) == 2 and "r2_trim" not in att[1]   # original evaluated too
    assert _max_err(att[0]["coarse"], s["top_px"]) < 18.0
    # the refine/depth check saw the trimmed quad, not the SAM3 outline
    assert _max_err(seen[0], s["top_px"]) < 18.0
    assert _max_err(s["cand"], s["top_px"]) > 30.0
    pr = rec["provenance"]["sam3"]["r2_trim"]
    assert pr["geometry"]["yaw45"] > 20 and pr["max_corner_shift_px"] > 20
    used = [r for r in rec["checks"]["reasons"] if r.startswith("r2_trim_used(")]
    assert used and "vs original" in used[0]
    assert "plane_trimmed" not in rec["provenance"]["sam3"]
    assert _max_err(_corners(rec), s["top_px"]) < 18.0


def test_r2_disabled_takes_the_old_path(monkeypatch, tmp_path):
    s = make_scene(30.0)
    seen = _patch(monkeypatch, s)
    cfg = default_config()
    cfg["reliability"]["r2_enabled"] = False
    rec, _ = _run(tmp_path, s["cand"], cfg=cfg)
    assert "r2_trim" not in rec["provenance"]["attempts"][0]
    assert "r2_trim" not in rec["provenance"]["sam3"]
    assert _max_err(seen[0], s["cand"]) < 1e-6


def test_r2_low_yaw_clean_quad_is_untouched(monkeypatch, tmp_path):
    s = make_scene(0.0)
    seen = _patch(monkeypatch, s)
    rec, _ = _run(tmp_path, s["top_px"])
    att = rec["provenance"]["attempts"]
    assert len(att) == 1 and "r2_trim" not in att[0]
    assert "r2_trim" not in rec["provenance"]["sam3"]
    assert np.allclose(seen[0], s["top_px"])
    assert not [r for r in rec["checks"]["reasons"] if r.startswith("r2_")]


def test_r2_high_yaw_but_no_spill_keeps_original(monkeypatch, tmp_path):
    s = make_scene(30.0)
    seen = _patch(monkeypatch, s)
    cfg = default_config()
    cfg["reliability"]["r2_min_spill_px"] = 30.0   # trim moves < 30 px
    rec, _ = _run(tmp_path, s["top_px"], cfg=cfg)  # outline already exact
    att = rec["provenance"]["attempts"]
    assert len(att) == 1 and "r2_trim" not in att[0]
    assert att[0].get("r2_skipped", "").startswith("no_spill")
    assert np.allclose(seen[0], s["top_px"])


def test_r2_rewrite_not_used_when_original_already_rectangle(monkeypatch,
                                                              tmp_path):
    s = make_scene(30.0)
    seen = _patch(monkeypatch, s)
    cfg = default_config()
    cfg["reliability"]["r2_min_spill_px"] = 0.0     # always rewrite...
    rec, _ = _run(tmp_path, s["top_px"], cfg=cfg)   # ...but the outline is fine
    att = rec["provenance"]["attempts"]
    assert len(att) == 2 and att[0]["r2_trim"] is True
    assert att[0].get("not_selected") is True and att[1].get("selected") is True
    assert "r2_trim" not in rec["provenance"]["sam3"]
    assert any(r.startswith("r2_trim_not_better_original_used(")
               for r in rec["checks"]["reasons"])
    assert np.allclose(seen[-1], s["top_px"])       # the original was kept


def test_r2_trimmed_attempt_rejected_falls_back_to_original(monkeypatch,
                                                            tmp_path):
    s = make_scene(30.0)
    cand = s["cand"]

    def reject_trimmed(quad):          # the depth check rejects the trim
        return _max_err(quad, s["top_px"]) < 18.0 and _max_err(quad, cand) > 20.0

    seen = _patch(monkeypatch, s, reject_when=reject_trimmed)
    rec, _ = _run(tmp_path, cand)
    att = rec["provenance"]["attempts"]
    # [rejected R2 attempt, accepted original, (old gross_rect trim attempt,
    # rejected by the same fake depth check)]
    assert [a["status"] for a in att][:2] == ["depth_cross_rejected",
                                               "accepted"]
    assert att[0]["r2_trim"] is True and "r2_trim" not in att[1]
    assert att[1]["rewrite_failed"] == "r2_trim"
    assert att[2].get("plane_trimmed") is True       # old path, unchanged
    assert _max_err(seen[1], cand) < 1e-6
    reasons = rec["checks"]["reasons"]
    assert "r2_trim_unusable_original_used" in reasons
    # the failed R2 attempt's depth_cross reason is not left behind; only
    # the old gross_rect attempt's one remains
    assert sum("depth_cross_rejected" in r for r in reasons) == 1


def test_r3_k4_quad_replaced_by_plane_rectangle(monkeypatch, tmp_path):
    s = make_scene(15.0)
    seen = _patch(monkeypatch, s)
    rec, _ = _run(tmp_path, s["cand"], k4=True)
    att = rec["provenance"]["attempts"]
    assert att[0]["r3_rect"] is True and att[0]["selected"] is True
    assert _max_err(att[0]["coarse"], s["top_px"]) < 10.0
    r3 = rec["provenance"]["sam3"]["r3_rect"]
    assert r3["kind"] == "k4_plane_rectangle" and r3["dims"]["within_tol"]
    assert rec["provenance"]["sam3"]["quad_method"] == "min_area_k4"
    assert any(r.startswith("r3_rect_used(") for r in rec["checks"]["reasons"])
    # a non-k4 quad of the same scene does not take the R3 path
    seen.clear()
    rec2, _ = _run(tmp_path, s["cand"], k4=False)
    assert "r3_rect" not in rec2["provenance"]["sam3"]
    disabled = default_config()
    disabled["reliability"]["r3_enabled"] = False
    disabled["reliability"]["r2_enabled"] = False
    rec3, _ = _run(tmp_path, s["cand"], k4=True, cfg=disabled)
    assert "r3_rect" not in rec3["provenance"]["sam3"]


# ---- R5 wiring ---------------------------------------------------------------------------

def _no_check_run(monkeypatch, tmp_path, quad, scene, cfg=None,
                  corner_flag=False):
    _patch(monkeypatch, scene, cross_iou=None, corner_flag=corner_flag)
    return _run(tmp_path, quad, cfg=cfg)


def _r2_off():
    cfg = default_config()
    cfg["reliability"]["r2_enabled"] = False
    return cfg


def test_r5_clears_needs_review_when_top_plane_confirms(monkeypatch,
                                                        tmp_path):
    s = make_scene(0.0)
    rec, _ = _no_check_run(monkeypatch, tmp_path, s["top_px"], s)
    chk = rec["checks"]
    assert chk["needs_review"] is False
    assert chk["rgb_depth_quad_iou"] is None
    itc = chk["independent_top_check"]
    assert itc["confirmed"] is True and itc["cleared_needs_review"] is True
    assert "no_independent_depth_check" not in chk["reasons"]
    assert any(r.startswith("needs_review_cleared(") for r in chk["reasons"])
    # not capped at max_conf_no_depth_check any more (independent IoU used)
    assert rec["confidence"] > default_config()["v2"][
        "max_conf_no_depth_check"]


def test_r5_keeps_needs_review_when_not_confirmed(monkeypatch, tmp_path):
    s = make_scene(0.0)
    bad = _spill_quad(s, 60.0)
    rec, _ = _no_check_run(monkeypatch, tmp_path, bad, s, cfg=_r2_off())
    chk = rec["checks"]
    assert chk["needs_review"] is True
    assert "no_independent_depth_check" in chk["reasons"]
    assert chk["independent_top_check"]["confirmed"] is False
    assert not chk["independent_top_check"].get("cleared_needs_review")


def test_r5_keeps_needs_review_with_corner_flags(monkeypatch, tmp_path):
    s = make_scene(0.0)
    rec, _ = _no_check_run(monkeypatch, tmp_path, s["top_px"], s,
                           corner_flag=True)
    chk = rec["checks"]
    assert chk["needs_review"] is True
    assert "no_independent_depth_check" in chk["reasons"]
    assert not chk["independent_top_check"].get("cleared_needs_review")


def test_r5_disabled_and_normal_depth_check_untouched(monkeypatch, tmp_path):
    s = make_scene(0.0)
    off = default_config()
    off["reliability"]["r5_enabled"] = False
    rec, _ = _no_check_run(monkeypatch, tmp_path, s["top_px"], s, cfg=off)
    assert rec["checks"]["needs_review"] is True
    assert "independent_top_check" not in rec["checks"]
    # a capture that has the RGB/depth IoU never computes the extra check
    _patch(monkeypatch, s, cross_iou=0.9)
    rec2, _ = _run(tmp_path, s["top_px"])
    assert rec2["checks"]["needs_review"] is False
    assert "independent_top_check" not in rec2["checks"]
