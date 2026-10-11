# Provenance: ported from rgbd_captures/tests/test_fold_edge.py (HEAD 6df5994); imports made package-relative.
"""Tests for scripts/autolabel_v2/fold_edge.py (side-face fold model).

Synthetic box renders from a real camera model: a box of known height
sits on a bright table, seen from ~0.45 m at 45 deg (like the rig). The
top face, the visible side faces and the table are rendered with 8x
supersampling (area anti-aliasing), so the true fold (top-face polygon)
and the side-face widths are known exactly.

* side_face_widths reproduces the rendered side widths (visible > 0,
  hidden <= 0);
* narrow visible side face, input edge on its outer silhouette -> the
  edge relocates to the fold (method silhouette_minus_width) within 1 px;
* input edge already on the fold -> stays within 1 px (fold_shoulder);
* wide face with a bright rim band, input on the rim->side step -> moves
  to the rim's inner edge (the fold) within 1 px;
* hidden side face with default hidden_mode -> unchanged, reason
  side_hidden; flat image -> unchanged, weak-evidence reasons;
  bad plane / box class / intrinsics -> unchanged with reason.
"""

import math
import sys
from pathlib import Path

import numpy as np
import cv2


from box_seam_detection.core import fold_edge as fe  # noqa: E402

W, H = 1280, 720
KM = np.array([[692.0, 0.0, 651.0], [0.0, 692.0, 366.0], [0.0, 0.0, 1.0]])
MAN = {"camera_info": {"k": KM.ravel().tolist()}}
HEIGHT = fe.BOX_HEIGHT_M["small"]
SS = 8  # supersampling


def _proj(x):
    x = np.atleast_2d(x)
    return np.stack([KM[0, 0] * x[:, 0] / x[:, 2] + KM[0, 2],
                     KM[1, 1] * x[:, 1] / x[:, 2] + KM[1, 2]], axis=1)


def _scene(lateral_m):
    """Top-face corners (BL, BR, FR, FL) in 3-D and the plane (n, d).

    Camera looks down at 45 deg; the top face is 111 x 100 mm.
    ``lateral_m`` shifts the box sideways (x): x < 0 exposes the right
    side face (camera sees the +x wall) as a narrow strip.
    """
    tilt = math.radians(45.0)
    n = np.array([0.0, -math.sin(tilt), -math.cos(tilt)])  # up, to camera
    e1 = np.array([1.0, 0.0, 0.0])                         # box x (right)
    e2 = np.cross(n, e1)                                   # box depth
    e2 /= np.linalg.norm(e2)
    center = np.array([lateral_m, 0.08, 0.42])
    a, b = 0.111 / 2, 0.100 / 2
    # back = further from camera: +e2 or -e2 whichever increases z
    back = e2 if e2[2] > 0 else -e2
    top = np.array([center - a * e1 + b * back, center + a * e1 + b * back,
                    center + a * e1 - b * back, center - a * e1 - b * back])
    d = -float(n @ center)
    return top, n, d


def _render(top, n, rim_px=0, noise=1.0, seed=0, top_val=150.0,
            side_val=95.0, table_val=232.0, rim_val=225.0):
    """Supersampled render; returns uint8 gray image."""
    bot = top - HEIGHT * n[None, :]
    big = np.full((H * SS, W * SS), table_val, np.float32)

    def poly(pts3, val):
        uv = (_proj(pts3) + 0.5) * SS - 0.5  # pixel-centre convention
        cv2.fillPoly(big, [np.round(uv * 16).astype(np.int32)], float(val),
                     lineType=cv2.LINE_8, shift=4)

    centre = top.mean(axis=0) - 0.5 * HEIGHT * n
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        face = np.array([top[i], top[j], bot[j], bot[i]])
        fc = face.mean(axis=0)
        normal = np.cross(top[j] - top[i], n)
        if normal @ (fc - centre) < 0:
            normal = -normal
        if normal @ (-fc) > 0:  # camera (origin) on the outer side
            poly(face, side_val)
            if rim_px:
                # bright rim band on the side face just below the fold
                k = rim_px / (float(np.linalg.norm(
                    _proj(top[i] - HEIGHT * n) - _proj(top[i]))) + 1e-9)
                rim = np.array([top[i], top[j],
                                top[j] - k * HEIGHT * n,
                                top[i] - k * HEIGHT * n])
                poly(rim, rim_val)
    poly(top, top_val)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _line_dist(quad, ei, pts):
    """Unsigned distances of points to edge ei of quad."""
    i0, i1 = fe._EDGE_PAIRS[ei]
    p0, p1 = quad[i0], quad[i1]
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-t[1], t[0]])
    return np.abs((np.asarray(pts) - p0) @ nrm)


def _edge_err(quad_out, quad_true, ei):
    """Max distance of the output edge's sample points to the true edge."""
    i0, i1 = fe._EDGE_PAIRS[ei]
    ts = np.linspace(0.15, 0.85, 8)
    pts = quad_out[i0][None, :] + ts[:, None] * (quad_out[i1] - quad_out[i0])
    return float(_line_dist(quad_true, ei, pts).max())


def _silhouette_quad(top, n, edge):
    """Top quad with one edge replaced by the side face's outer silhouette
    (projection of the bottom edge) - what the argmax refiner locks onto."""
    q = _proj(top)
    bot = _proj(top - HEIGHT * n[None, :])
    ei = fe.EDGE_NAMES.index(edge)
    lines = []
    for k, (i, j) in enumerate(fe._EDGE_PAIRS):
        if k == ei:
            lines.append(fe._line_through(bot[i], bot[j]))
        else:
            lines.append(fe._line_through(q[i], q[j]))
    back, right, front, left = lines
    return np.array([fe._intersect(a, b) for a, b in
                     ((left, back), (back, right), (right, front),
                      (front, left))])


def _plane(n, d):
    return {"normal": n.tolist(), "offset_m": d}


def test_side_face_widths_match_render_geometry():
    top, n, d = _scene(-0.10)
    q = _proj(top)
    for ei, (i0, i1) in enumerate(fe._EDGE_PAIRS):
        p0, p1 = q[i0], q[i1]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        out_n = np.array([-t[1], t[0]])
        if out_n @ ((p0 + p1) / 2 - q.mean(axis=0)) < 0:
            out_n = -out_n
        st = p0[None, :] + np.outer([0.5], p1 - p0)
        x_edge = fe._backproject(np.stack([p0, p1]), KM, n, d)
        w, lo, hi = fe.side_face_widths(st, x_edge, out_n, KM, n, d,
                                        HEIGHT, 3.0)
        xm = fe._backproject(st, KM, n, d)[0]
        w_true = float((_proj(xm - HEIGHT * n) - _proj(xm))[0] @ out_n)
        assert abs(w[0] - w_true) < 1e-6
        assert lo[0] <= w[0] <= hi[0]
    names = dict(zip(fe.EDGE_NAMES, range(4)))
    # front wall faces the camera (wide), back wall hidden
    xm_f = (top[2] + top[3]) / 2
    assert (_proj(xm_f - HEIGHT * n) - _proj(xm_f))[0][1] > 30
    assert names["back"] == 0


def test_narrow_visible_side_relocates_from_silhouette_to_fold():
    top, n, d = _scene(-0.10)
    img = _render(top, n)
    q_true = _proj(top)
    q_in = _silhouette_quad(top, n, "right")
    ri = fe.EDGE_NAMES.index("right")
    gap = _edge_err(q_in, q_true, ri)
    assert 3.0 < gap < 30.0, gap  # a narrow but real side face
    out = fe.relocate_edges(img, q_in, MAN, _plane(n, d), "small")
    rec = out["per_edge"]["right"]
    assert out["changed"], out["per_edge"]
    assert rec["method"] == "silhouette_minus_width", rec
    assert rec["side_face"] == "narrow"
    assert rec["role"] == "current_is_silhouette"
    err = _edge_err(np.asarray(out["corners_px"]), q_true, ri)
    assert err < 1.0, (err, rec)


def test_edge_on_fold_stays_on_fold():
    top, n, d = _scene(-0.10)
    img = _render(top, n)
    q_true = _proj(top)
    out = fe.relocate_edges(img, q_true.copy(), MAN, _plane(n, d), "small")
    rec = out["per_edge"]["right"]
    assert rec["method"] == "fold_shoulder", rec
    assert rec["role"] == "current_is_fold"
    q_out = np.asarray(out["corners_px"])
    for e in ("right", "front"):
        assert _edge_err(q_out, q_true, fe.EDGE_NAMES.index(e)) < 1.0


def test_wide_face_rim_band_moves_to_fold():
    top, n, d = _scene(0.0)
    img = _render(top, n, rim_px=3)
    q_true = _proj(top)
    fi = fe.EDGE_NAMES.index("front")
    # input front edge on the rim -> side step (3 px outside the fold)
    q_in = q_true.copy()
    i0, i1 = fe._EDGE_PAIRS[fi]
    q_in[i0, 1] += 3.0
    q_in[i1, 1] += 3.0
    assert _edge_err(q_in, q_true, fi) > 2.5
    out = fe.relocate_edges(img, q_in, MAN, _plane(n, d), "small")
    rec = out["per_edge"]["front"]
    assert rec["side_face"] == "wide", rec
    assert rec["method"] == "fold_shoulder", rec
    err = _edge_err(np.asarray(out["corners_px"]), q_true, fi)
    assert err < 1.0, (err, rec)


def test_hidden_side_kept_with_reason_by_default():
    top, n, d = _scene(-0.10)
    img = _render(top, n)
    q_true = _proj(top)
    out = fe.relocate_edges(img, q_true.copy(), MAN, _plane(n, d), "small")
    for e in ("back", "left"):
        rec = out["per_edge"][e]
        assert rec["side_face"] == "hidden", rec
        assert rec["method"] == "keep" and rec["reason"] == "side_hidden"
    # hidden edges keep their exact input line
    q_out = np.asarray(out["corners_px"])
    bi = fe.EDGE_NAMES.index("back")
    assert _edge_err(q_out, q_true, bi) < 1e-6
    # opt-in shoulder mode moves them but stays within 1 px of the fold
    out2 = fe.relocate_edges(img, q_true.copy(), MAN, _plane(n, d), "small",
                             {"hidden_mode": "shoulder"})
    assert out2["per_edge"]["back"]["method"] == "fold_shoulder"
    assert _edge_err(np.asarray(out2["corners_px"]), q_true, bi) < 1.0


def test_weak_evidence_unchanged_with_reason():
    top, n, d = _scene(-0.10)
    rng = np.random.default_rng(3)
    flat = np.clip(150 + rng.normal(0, 1.0, (H, W)), 0, 255).astype(np.uint8)
    q = _proj(top)
    out = fe.relocate_edges(flat, q.copy(), MAN, _plane(n, d), "small",
                            {"hidden_mode": "shoulder"})
    assert not out["changed"]
    assert np.allclose(out["corners_px"], q)
    for rec in out["per_edge"].values():
        assert rec["method"] == "keep"
        assert rec["reason"] in ("weak_evidence", "no_silhouette"), rec


def test_bad_inputs_unchanged_with_reason():
    top, n, d = _scene(-0.10)
    img = _render(top, n)
    q = _proj(top) + 2.0
    cases = [
        ((MAN, None, "small"), "bad_plane"),
        ((MAN, {"normal": [0, 0, 0], "offset_m": 0.4}, "small"), "bad_plane"),
        ((MAN, _plane(n, d), "medium"), "bad_box_class"),
        (({"camera_info": {}}, _plane(n, d), "small"), "no_intrinsics"),
    ]
    for (man, plane, cls), reason in cases:
        out = fe.relocate_edges(img, q, man, plane, cls)
        assert not out["changed"]
        assert np.allclose(out["corners_px"], q)
        assert all(r["reason"] == reason for r in out["per_edge"].values())


def test_box_class_and_corners_xyz():
    top, n, d = _scene(0.0)
    assert fe.box_class_from_xyz(top) == "small"
    assert fe.box_class_from_xyz(top * np.array([1.9, 1.0, 1.0])) == "big"
    xyz = fe.corners_xyz_on_plane(_proj(top), MAN, _plane(n, d))
    assert np.allclose(xyz, top, atol=1e-9)
    # plane given with the opposite sign convention gives the same points
    xyz2 = fe.corners_xyz_on_plane(_proj(top), MAN, _plane(-n, -d))
    assert np.allclose(xyz2, top, atol=1e-9)


def test_grazing_side_face_treated_like_hidden():
    # choose the lateral offset where the right wall is barely visible
    for lat in np.linspace(-0.02, -0.06, 41):
        top, n, d = _scene(lat)
        q = _proj(top)
        p0, p1 = q[1], q[2]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        out_n = np.array([-t[1], t[0]])
        if out_n @ ((p0 + p1) / 2 - q.mean(axis=0)) < 0:
            out_n = -out_n
        st = ((p0 + p1) / 2)[None, :]
        x_edge = fe._backproject(np.stack([p0, p1]), KM, n, d)
        _w, _lo, hi = fe.side_face_widths(st, x_edge, out_n, KM, n, d,
                                          HEIGHT, 3.0)
        if 1.5 < hi[0] < 3.5:
            break
    else:
        raise AssertionError("no grazing configuration found")
    img = _render(top, n)
    out = fe.relocate_edges(img, q.copy(), MAN, _plane(n, d), "small")
    rec = out["per_edge"]["right"]
    assert rec["side_face"] == "grazing", rec
    assert rec["method"] == "keep" and rec["reason"] == "side_grazing"


def test_distortion_roundtrip_and_bad_image():
    man = {"camera_info": {"k": KM.ravel().tolist(),
                           "d": [0.007, -0.049, 0.0, 0.0, 0.033],
                           "distortion_model": "plumb_bob"}}
    cam = fe._intrinsics(man)
    assert cam is not None and cam.dist is not None
    top, n, d = _scene(-0.05)
    uv = fe._project(top, cam)
    assert not np.allclose(uv, _proj(top))  # distortion applied
    xyz = fe.corners_xyz_on_plane(uv, man, _plane(n, d))
    assert np.allclose(xyz, top, atol=1e-7)
    out = fe.relocate_edges(None, _proj(top), MAN, _plane(n, d), "small")
    assert not out["changed"]
    assert all(r["reason"].startswith("bad_image")
               for r in out["per_edge"].values())


def test_pipeline_apply_corner_fold_and_fold_record(tmp_path):
    import json

    from box_seam_detection.core.config import default_config
    from box_seam_detection.core.pipeline import apply_corner_fold, fold_reason
    from box_seam_detection.core.run import fold_record

    top, n, d = _scene(-0.10)
    img = _render(top, n)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    q_in = _silhouette_quad(top, n, "right")
    cfg = default_config()
    # default since 2026-09-27: enabled, hidden_mode "shoulder"
    assert cfg["corner_fold"]["enabled"] is True
    assert cfg["corner_fold"]["hidden_mode"] == "shoulder"
    cfg["corner_fold"]["enabled"] = True
    xyz_in = fe.corners_xyz_on_plane(q_in, MAN, _plane(n, d))
    quad, xyz, info = apply_corner_fold(bgr, q_in, MAN, _plane(n, d),
                                        xyz_in, cfg["corner_fold"])
    assert info["changed"] and info["box_class"] == "small"
    assert info["per_edge"]["right"]["method"] == "silhouette_minus_width"
    assert np.allclose(fe._project(xyz, KM), quad, atol=1e-6)
    assert fold_reason(info).startswith("corner_fold_moved(")
    # no metric corners -> unchanged with reason
    q2, x2, info2 = apply_corner_fold(bgr, q_in, MAN, _plane(n, d), None,
                                      cfg["corner_fold"])
    assert q2 is q_in and info2["reason"] == "no_metric_corners"

    keys = fe._CORNER_KEYS
    man_path = tmp_path / "m.json"
    man_path.write_text(json.dumps(MAN))
    src = {"top_face": {
        "corners_px": {k: [round(float(q_in[i][0]), 1),
                           round(float(q_in[i][1]), 1)]
                       for i, k in enumerate(keys)},
        "corners_xyz_camera_m": {k: [float(v) for v in xyz_in[i]]
                                 for i, k in enumerate(keys)},
        "plane": _plane(n, d)},
        "checks": {"reasons": ["x"]}, "provenance": {"config": {}},
        "seam": {"status": "estimated"}}
    rec = fold_record(src, {"manifest": str(man_path)}, cfg, bgr, "src")
    assert src["top_face"]["corners_px"]["back_right"] == \
        [round(float(q_in[1][0]), 1), round(float(q_in[1][1]), 1)]
    assert rec["top_face"]["corners_px"] != src["top_face"]["corners_px"]
    assert rec["provenance"]["corner_fold"]["changed"] is True
    assert rec["provenance"]["corner_fold"]["from"] == "src"
    assert rec["checks"]["reasons"][-1].startswith("corner_fold_moved(")
    # every other field untouched
    for k in ("seam", "checks"):
        if k == "checks":
            assert rec[k]["reasons"][:-1] == src[k]["reasons"]
        else:
            assert rec[k] == src[k]
    assert rec["top_face"]["plane"] == src["top_face"]["plane"]


def test_cli_fold_flags_and_reseam_requires_recheck(tmp_path):
    import pytest as _pt
    import sys as _sys
    from box_seam_detection.core import run as _run
    base = ["--reseam-from", str(tmp_path / "missing_src"),
            "--run-name", "x", "--out-root", str(tmp_path / "out")]
    # fold is on by default -> a seam-only reseam must opt out or recheck
    with _pt.raises(SystemExit, match="recheck-corners"):
        _run.main(base)
    with _pt.raises(SystemExit, match="conflict"):
        _run.main(base + ["--corner-fold", "--no-corner-fold"])
    # opting out / rechecking gets past the guard to the missing source dir
    with _pt.raises(SystemExit, match="source dir missing"):
        _run.main(base + ["--no-corner-fold"])
    with _pt.raises(SystemExit, match="source dir missing"):
        _run.main(base + ["--recheck-corners"])
