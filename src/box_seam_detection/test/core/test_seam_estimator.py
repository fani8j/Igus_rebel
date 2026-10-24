# Provenance: ported from rgbd_captures/tests/test_seam_estimator.py (HEAD 6df5994); imports made package-relative.
"""Tests for scripts/autolabel_v2/seam_estimator.py.

Synthetic-only: rectangles under a known homography (projective midline
exactness, vanishing point at infinity), synthetic flap gap + tape stripe
(axis choice + subpixel gap refinement, gap-follows-gap-not-tape),
no-gap images (fail-closed statuses), printed-line / shadow / tape-edge
negatives. No dataset reads, no model downloads.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


from box_seam_detection.core.seam_estimator import (axis_evidence, estimate_seam,
                                         expected_view_shift,
                                         projective_midlines, refine_on_gap,
                                         refine_on_darkline, refine_on_tape,
                                         rectify_face, view_geometry)


def _warp_rect_to_quad(rect_pts, quad, size=(640, 480)):
    """Render rect-space points into an image viewed at ``quad``."""
    W, H = 400, 300
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    dst = np.asarray(quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = []
    for p in rect_pts:
        q = M @ np.array([p[0], p[1], 1.0])
        out.append(q[:2] / q[2])
    return np.array(out), M


def _synth_face(seam_axis="lr", tape=True, seed=0, size=(640, 480),
                gap=True, tape_offset_px=0, gap_offset_px=0):
    """Synthetic top-face view: brown cardboard + tape stripe + flap gap.

    Rect-space face is 400x300 with the seam at row 150 (lr) or col 200
    (bf); the tape is a low-saturation band ~14 px wide and the flap gap
    is a 2-px dark valley (gray -60, bright cardboard shoulders) at the
    seam row/col, optionally offset from the tape centre by
    ``tape_offset_px`` (tape follows the gap only when offset is 0). The
    face is warped by a fixed perspective quad. Returns (bgr, corners, gt).
    """
    rng = np.random.default_rng(seed)
    W, H = 400, 300
    sat = np.full((H + 1, W + 1), 95.0)
    val = np.full((H + 1, W + 1), 165.0)
    hue = np.full((H + 1, W + 1), 15.0)
    gt_rect = None
    if tape:
        hw = 7
        off = int(tape_offset_px)
        if seam_axis == "lr":
            sat[150 + off - hw:150 + off + hw + 1, :] -= 28.0
            val[150 + off - hw:150 + off + hw + 1, :] += 8.0
            gt_rect = [(0, 150), (W, 150)]
        else:
            sat[:, 200 + off - hw:200 + off + hw + 1] -= 28.0
            val[:, 200 + off - hw:200 + off + hw + 1] += 8.0
            gt_rect = [(200, 0), (200, H)]
    else:
        gt_rect = [(0, 150), (W, 150)] if seam_axis == "lr" \
            else [(200, 0), (200, H)]
    if gap:
        g = int(gap_offset_px)
        if seam_axis == "lr":
            val[149 + g:151 + g, :] = np.minimum(val[149 + g:151 + g, :],
                                                 105.0)
        else:
            val[:, 199 + g:201 + g] = np.minimum(val[:, 199 + g:201 + g],
                                                 105.0)
    tex = rng.normal(0, 2.0, sat.shape)
    sat = np.clip(sat + tex, 0, 255).astype(np.uint8)
    val = np.clip(val + rng.normal(0, 2.0, val.shape), 0, 255).astype(np.uint8)
    hue = np.clip(hue + rng.normal(0, 1.0, hue.shape), 0, 179).astype(np.uint8)
    face = cv2.merge([hue, sat, val])
    face = cv2.cvtColor(face, cv2.COLOR_HSV2BGR)
    quad = [(120, 90), (520, 140), (480, 400), (90, 360)]  # BL,BR,FR,FL
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    img = np.full((size[1], size[0], 3), 60, np.uint8)
    cv2.warpPerspective(face, M, (size[0], size[1]), dst=img,
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    gt = None
    if gt_rect is not None:
        gt, _ = _warp_rect_to_quad(gt_rect, quad)
        gt = np.array(sorted(gt, key=lambda p: p[0]))
    return img, np.array(quad, float), gt


def test_projective_midline_exact_on_projected_midline():
    # Ground truth: rect-space midlines warped by the same homography.
    img, corners, _ = _synth_face()
    pr = projective_midlines(corners)
    for axis, gt_rect in (("lr", [(0, 150), (400, 150)]),
                          ("bf", [(200, 0), (200, 300)])):
        gt, _ = _warp_rect_to_quad(gt_rect, corners)
        gt = np.array(sorted(gt, key=lambda p: p[0]))
        est = pr[axis]["endpoints"]
        err = (np.linalg.norm(est[0] - gt[0])
               + np.linalg.norm(est[1] - gt[1])) / 2.0
        assert err < 1e-6, (axis, err)


def test_projective_midline_parallelogram_vp_at_infinity():
    corners = np.array([[100, 100], [400, 100], [350, 300], [50, 300]], float)
    pr = projective_midlines(corners)
    assert pr["lr"]["vp"] is None  # back/front edges parallel
    est = pr["lr"]["endpoints"]
    assert abs(est[0][1] - 200.0) < 1e-9
    assert abs(est[1][1] - 200.0) < 1e-9


def test_axis_choice_on_synthetic_stripe():
    for axis in ("lr", "bf"):
        img, corners, _ = _synth_face(seam_axis=axis)
        ev_lr = axis_evidence(img, corners, "lr")
        ev_bf = axis_evidence(img, corners, "bf")
        picked = "lr" if ev_lr > ev_bf else "bf"
        assert picked == axis, (axis, ev_lr, ev_bf)
        res = estimate_seam(img, corners)
        assert res["axis"] == axis, (axis, res["candidates"])
        # Without a metric scale ok is impossible; axis must still be
        # right and the outcome must be estimated (never ok).
        assert res["status"] == "estimated"


def test_refinement_recovers_gap_subpixel():
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    for axis in ("lr", "bf"):
        img, corners, gt = _synth_face(seam_axis=axis)
        eps, score, _ = refine_on_gap(img, corners, axis,
                                      corners_xyz=xyz)
        assert eps is not None
        err = (np.linalg.norm(eps[0] - gt[0])
               + np.linalg.norm(eps[1] - gt[1])) / 2.0
        assert err < 0.5, (axis, err)


def test_gap_follows_gap_not_tape_centre():
    """Translucent tape stripe offset from the gap: the fit must follow
    the gap valley, not the tape centreline."""
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    img, corners, gt = _synth_face(seam_axis="lr", tape_offset_px=6)
    eps, _, _ = refine_on_gap(img, corners, "lr", corners_xyz=xyz)
    assert eps is not None
    err = (np.linalg.norm(eps[0] - gt[0])
           + np.linalg.norm(eps[1] - gt[1])) / 2.0
    assert err < 1.0, err
    # The tape centreline is ~6 rect px off; the gap fit must beat it.
    teps, _, _ = refine_on_tape(img, corners, "lr", corners_xyz=xyz)
    if teps is not None:
        terr = (np.linalg.norm(teps[0] - gt[0])
                + np.linalg.norm(teps[1] - gt[1])) / 2.0
    assert err < terr, (err, terr)


def test_either_confirm_ok_if_either_agrees():
    """either_confirm publishes the prior when at least one fit agrees:
    source records which confirmed; estimated only when both reject."""
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    img, corners, _ = _synth_face(seam_axis="lr")
    res = estimate_seam(img, corners,
                        est_cfg={"seam_mode": "either_confirm"},
                        corners_xyz=xyz)
    assert res["status"] == "ok", res.get("estimate")
    assert res["source"] in ("prior_confirmed_by_gap",
                             "prior_confirmed_by_darkline")
    assert res["refined_endpoints_px"] is not None
    for k in ("seam_left", "seam_right"):
        for a, b in zip(res["refined_endpoints_px"][k],
                        res["prior_endpoints_px"][k]):
            assert abs(a - b) < 0.05
    assert res["estimate"]["estimate_endpoints_px"] == \
        res["prior_endpoints_px"]
    img2, corners2 = _synth_face_variant("print")
    res2 = estimate_seam(img2, corners2,
                         est_cfg={"seam_mode": "either_confirm"},
                         corners_xyz=xyz)
    assert res2["status"] in ("estimated", "ambiguous")
    assert res2["refined_endpoints_px"] is None


def test_darkline_follows_dark_line_under_offset_tape():
    """Synthetic dark line under an offset tape stripe: darkline follows
    the dark line (err < 1 px), not the tape centre."""
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    img, corners, gt = _synth_face(seam_axis="lr", tape_offset_px=6)
    eps, _, info = refine_on_darkline(img, corners, "lr",
                                      corners_xyz=xyz)
    assert eps is not None, info
    err = (np.linalg.norm(eps[0] - gt[0])
           + np.linalg.norm(eps[1] - gt[1])) / 2.0
    assert err < 1.0, err
    res = estimate_seam(img, corners, est_cfg={"seam_mode": "darkline"},
                        corners_xyz=xyz)
    assert res["status"] == "ok", res.get("estimate")
    assert res["source"] == "observed_darkline"
    assert res["refined_endpoints_px"] is not None


def test_darkline_negatives_fail_closed():
    """Print/glare/tape-without-gap must not become darkline-ok; a short
    dark block (barcode) must fail coverage, not publish."""
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    for kind in ("print", "glare"):
        img, corners = _synth_face_variant(kind)
        res = estimate_seam(img, corners,
                            est_cfg={"seam_mode": "darkline"},
                            corners_xyz=xyz)
        assert res["status"] in ("estimated", "ambiguous"), kind
        assert res["refined_endpoints_px"] is None, kind
    img, corners = _synth_face_variant("barcode")
    res = estimate_seam(img, corners, est_cfg={"seam_mode": "darkline"},
                        corners_xyz=xyz)
    assert res["status"] in ("estimated", "ambiguous"), "barcode"
    assert res["refined_endpoints_px"] is None, "barcode"
    img, corners, _ = _synth_face(tape=True, gap=False)
    res = estimate_seam(img, corners, est_cfg={"seam_mode": "darkline"},
                        corners_xyz=xyz)
    assert res["status"] in ("estimated", "ambiguous"), "tape-no-gap"
    assert res["refined_endpoints_px"] is None, "tape-no-gap"
    img, corners, _ = _synth_face(tape=False, gap=False)
    res = estimate_seam(img, corners)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["refined_endpoints_px"] is None
    assert res["source"] in ("prior", None)
    # No endpoints that could be mistaken for a cutting target.
    if res["status"] == "ambiguous":
        assert res["prior_endpoints_px"] is None
        assert res["axis"] is None
    else:
        assert res["axis"] in ("lr", "bf")


def test_tape_without_gap_never_ok():
    """Tape stripe but no flap gap: a printed/tape line without the dark
    valley must not become ok."""
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    img, corners, _ = _synth_face(tape=True, gap=False)
    res = estimate_seam(img, corners, corners_xyz=xyz)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["refined_endpoints_px"] is None
    assert res["observed_line"] is None


def test_rectify_face_maps_corners():
    _, corners, _ = _synth_face()
    M, Mi, W, H = rectify_face(np.zeros((480, 640, 3), np.uint8), corners)
    dst = np.array([[0, H], [W, H], [W, 0], [0, 0]], float)
    for p, q in zip(corners, dst):
        r = M @ np.array([p[0], p[1], 1.0])
        assert np.linalg.norm(r[:2] / r[2] - q) < 1e-6


# ---- observed-tape negatives -------------------------------------------------

def _synth_face_variant(kind="print", seed=0, size=(640, 480)):
    """Tapeless faces with tape-lookalike distractors (rect-space face
    400x300 warped to the standard perspective quad):

    - print: pale printed stripe (shallow wide sat dip, no sharp edges)
    - glare: bright low-sat band, wide with soft edges (one-sided gradient)
    - barcode: dark low-sat block near the midline but short (no span)
    """
    rng = np.random.default_rng(seed)
    W, H = 400, 300
    sat = np.full((H + 1, W + 1), 95.0)
    val = np.full((H + 1, W + 1), 165.0)
    hue = np.full((H + 1, W + 1), 15.0)
    if kind == "print":
        sat[139:162, :] -= 8.0
    elif kind == "glare":
        sat[128:173, :] -= 20.0
        val[128:173, :] += 40.0
    elif kind == "barcode":
        sat[146:154, 160:240] -= 45.0
        val[146:154, 160:240] -= 30.0
    else:
        raise ValueError(kind)
    sat = np.clip(sat + rng.normal(0, 2.0, sat.shape), 0, 255).astype(np.uint8)
    val = np.clip(val + rng.normal(0, 2.0, val.shape), 0, 255).astype(np.uint8)
    hue = np.clip(hue + rng.normal(0, 1.0, hue.shape), 0, 179).astype(np.uint8)
    face = cv2.merge([hue, sat, val])
    face = cv2.cvtColor(face, cv2.COLOR_HSV2BGR)
    quad = [(120, 90), (520, 140), (480, 400), (90, 360)]
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    img = np.full((size[1], size[0], 3), 60, np.uint8)
    cv2.warpPerspective(face, M, (size[0], size[1]), dst=img,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return img, np.array(quad, float)


def test_print_stripe_not_ok():
    """Pale printed stripe: shallow dip without dark valley -> never ok."""
    img, corners = _synth_face_variant("print")
    res = estimate_seam(img, corners)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["refined_endpoints_px"] is None
    assert res["observed_line"] is None


def test_glare_stripe_not_ok():
    """Bright low-sat glare band: one-sided gradient, no valley."""
    img, corners = _synth_face_variant("glare")
    res = estimate_seam(img, corners)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["refined_endpoints_px"] is None
    assert res["observed_line"] is None


def test_barcode_near_midline_not_ok():
    """Short dark block near the midline: no face-span support -> not ok."""
    img, corners = _synth_face_variant("barcode")
    res = estimate_seam(img, corners)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["refined_endpoints_px"] is None
    assert res["observed_line"] is None


def test_shadow_without_shoulders_not_ok():
    """Broad dark shadow band (wide valley, dim shoulders) is not a gap:
    width + shoulder gates must reject it."""
    rng = np.random.default_rng(0)
    W, H = 400, 300
    sat = np.full((H + 1, W + 1), 95.0)
    val = np.full((H + 1, W + 1), 165.0)
    hue = np.full((H + 1, W + 1), 15.0)
    val[140:161, :] -= 18.0  # 21-px-wide shallow shadow, no bright gap walls
    sat = np.clip(sat + rng.normal(0, 2.0, sat.shape), 0, 255).astype(np.uint8)
    val = np.clip(val + rng.normal(0, 2.0, val.shape), 0, 255).astype(np.uint8)
    hue = np.clip(hue + rng.normal(0, 1.0, hue.shape), 0, 179).astype(np.uint8)
    face = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)
    quad = [(120, 90), (520, 140), (480, 400), (90, 360)]
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.warpPerspective(face, M, (640, 480), dst=img,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    res = estimate_seam(img, np.array(quad, float), corners_xyz=xyz)
    assert res["status"] in ("estimated", "ambiguous")
    assert res["observed_line"] is None


def test_line_agreement_rect_beats_image_scale_under_anisotropy():
    """Rect-space agreement (not image px × rect scale): an anisotropic
    homography stretches one direction, so the same two lines have
    different perpendicular distances in image vs rectified space. The
    gate must use the rectified distance (reviewer BLOCKER: stored
    1.492 mm image-scale vs 1.691 mm rectified on
    frame_000001_20260909_062436_099339)."""
    from box_seam_detection.core.seam_estimator import (
        _eps_to_rect, _line_agreement, _line_agreement_rect,
        _rect_px_to_mm, rectify_face)
    # Anisotropic view: strong perspective tilt compresses one diagonal.
    quad = np.array([(60, 60), (560, 110), (500, 420), (40, 380)], float)
    img = np.zeros((480, 640, 3), np.uint8)
    M, _, W, H = rectify_face(img, quad)
    # Two lines 3 rect-px apart across the seam direction.
    prior_rect = ((0.0, H / 2.0), (float(W), H / 2.0))
    gap_rect = ((0.0, H / 2.0 + 3.0), (float(W), H / 2.0 + 3.0))
    Mi = np.linalg.inv(np.asarray(M, dtype=np.float64))

    def _back(pt):
        p = Mi @ np.array([pt[0], pt[1], 1.0])
        return p[:2] / p[2]

    prior_img = {"seam_left": [float(v) for v in _back(prior_rect[0])],
                 "seam_right": [float(v) for v in _back(prior_rect[1])]}
    gap_img = {"seam_left": [float(v) for v in _back(gap_rect[0])],
               "seam_right": [float(v) for v in _back(gap_rect[1])]}
    mm = 0.65  # representative across-seam scale
    rect = _line_agreement_rect(prior_rect, gap_rect, mm)
    assert rect is not None
    assert rect["offset_px"] == [3.0, 3.0, 3.0], rect
    assert rect["max_offset_mm"] == pytest.approx(3.0 * mm, abs=1e-9)
    # Round-trip through the image must recover the same rect distance:
    # the gate path maps published endpoints back with the SAME M.
    assert _eps_to_rect(prior_img, M) is not None
    back_rect = _line_agreement_rect(
        _eps_to_rect(prior_img, M), _eps_to_rect(gap_img, M), mm)
    assert back_rect is not None
    assert back_rect["max_offset_px"] == pytest.approx(3.0, abs=1e-6)
    # The legacy image-space shim measures a DIFFERENT distance here
    # (that is the bug being fixed); it must not equal the rect value
    # on this anisotropic quad.
    legacy = _line_agreement(prior_img, gap_img, mm)
    assert legacy is not None
    assert legacy["max_offset_px"] != pytest.approx(
        rect["max_offset_px"], abs=1e-6)


def test_observed_line_matches_refined_and_reports_support():
    """Refine mode (comparison): observed gap line published as refined.
    Confirm mode (default): prior published, gap kept in estimate."""
    img, corners, gt = _synth_face(seam_axis="lr")
    xyz = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(
        ("back_left", "back_right", "front_right", "front_left"))}
    res = estimate_seam(img, corners, corners_xyz=xyz,
                        est_cfg={"seam_mode": "refine"})
    assert res["status"] == "ok", res.get("refine_info")
    assert res["observed_line"] == res["refined_endpoints_px"]
    assert res["source"] == "observed_gap"
    assert res["coverage"] is not None and res["coverage"] >= 0.40
    assert res["residual_px"] is not None
    assert res["estimate"]["median_gap_width_mm"] is not None
    assert res["estimate"]["per_slice_support"]["n_inliers"] >= 4
    eps, score, info = refine_on_gap(img, corners, "lr",
                                     corners_xyz=xyz)
    assert eps is not None
    assert info["coverage"] >= 0.40
    assert info["max_gap_frac"] <= 0.35
    err = (np.linalg.norm(eps[0] - gt[0])
           + np.linalg.norm(eps[1] - gt[1])) / 2.0
    assert err < 0.5, err
    # Confirm mode publishes the prior; observed gap stays for audit.
    # Refined carries FULL precision (pipeline rounds once to 0.1 px),
    # so compare within half a publish step, not exact equality.
    resc = estimate_seam(img, corners, corners_xyz=xyz)
    assert resc["status"] in ("ok", "estimated")
    if resc["refined_endpoints_px"] is None:
        assert resc["status"] == "estimated"
    else:
        for k in ("seam_left", "seam_right"):
            for a, b in zip(resc["refined_endpoints_px"][k],
                            resc["prior_endpoints_px"][k]):
                assert a == pytest.approx(b, abs=0.05)
    assert resc["observed_line"] is not None
    assert resc["estimate"]["estimate_endpoints_px"] == \
        resc["prior_endpoints_px"]


def test_missing_metric_scale_never_ok():
    """width_mm None (missing/invalid corners_xyz) must NOT allow ok:
    axis-clear gap evidence degrades to estimated with no_metric_scale."""
    img, corners, _ = _synth_face(seam_axis="lr")
    res = estimate_seam(img, corners, corners_xyz=None)
    assert res["axis"] == "lr"
    assert res["status"] == "estimated"
    assert res["observed_line"] is None
    assert res["refined_endpoints_px"] is None
    assert (res.get("refine_info") or {}).get("reject") == "no_metric_scale"
    eps, score, info = refine_on_gap(img, corners, "lr",
                                     corners_xyz=None)
    assert eps is None
    assert info["reject"] == "no_metric_scale"
    bad = {n: [0.0, 0.0, float("nan")] for n in
           ("back_left", "back_right", "front_right", "front_left")}
    res2 = estimate_seam(img, corners, corners_xyz=bad)
    assert res2["status"] == "estimated"
    assert (res2.get("refine_info") or {}).get("reject") == \
        "no_metric_scale"


# ---- viewing shift (confirmation vs prior + expected apparent shift) ----

_NAMES = ("back_left", "back_right", "front_right", "front_left")


def _small_box_xyz(back_dir=(0.0, -0.8, 0.6), center=(0.0, 0.0, 0.45),
                   across_m=0.112, along_m=0.100):
    """Planar small-box top (112 mm front->back, 100 mm left->right) in the
    camera frame. The default tilt puts the back edge farther from the
    camera, i.e. rect +y (toward BL/BR) is "away"; the synthetic rect row
    scale is then 112 mm / 300 rows = 0.373 mm per synthetic row."""
    b = np.asarray(back_dir, float)
    b = b / np.linalg.norm(b)
    r = np.array([1.0, 0.0, 0.0])
    c = np.asarray(center, float)
    ha, hl = across_m / 2.0, along_m / 2.0
    pts = [c + ha * b - hl * r, c + ha * b + hl * r,
           c - ha * b + hl * r, c - ha * b - hl * r]
    return {n: [float(v) for v in p] for n, p in zip(_NAMES, pts)}


def _shift_cfg(shift_mm, enabled=True):
    return {"seam_mode": "either_confirm",
            "view_shift": {"enabled": enabled,
                           "shift_mm": {"big": 0.0, "small": shift_mm},
                           "k_tan_mm": {"big": 0.0, "small": 0.0}}}


def test_view_geometry_away_sign_and_class():
    xyz = _small_box_xyz()
    geo = view_geometry(xyz, "lr")
    assert geo["box_class"] == "small"
    assert geo["long_side_mm"] == pytest.approx(112.0, abs=0.1)
    assert geo["away_rect_sign"] == 1          # back edge is farther
    assert 0.0 < geo["view_angle_deg"] < 90.0
    # Back edge nearer the camera -> rect +y is TOWARD the camera.
    near = _small_box_xyz(back_dir=(0.0, -0.8, -0.6), center=(0, 0, 0.45))
    assert view_geometry(near, "lr")["away_rect_sign"] == -1
    big = _small_box_xyz(across_m=0.193, along_m=0.105)
    assert view_geometry(big, "lr")["box_class"] == "big"
    # Collinear / missing xyz -> no geometry, zero shift (plain prior).
    line = {n: [float(i) * 0.112, 0.0, 0.5] for i, n in enumerate(_NAMES)}
    assert view_geometry(line, "lr") is None
    vs = expected_view_shift(line, "lr", _shift_cfg(2.0))
    assert vs["expected_shift_mm"] == 0.0
    assert vs["reason"] == "no_view_geometry"
    vs2 = expected_view_shift(xyz, "lr", _shift_cfg(2.0))
    assert vs2["expected_shift_mm"] == pytest.approx(2.0)
    assert vs2["box_class"] == "small"
    vs3 = expected_view_shift(xyz, "lr", {"view_shift": {
        "shift_mm": {"small": 1.0}, "k_tan_mm": {"small": 2.0}}})
    t = np.tan(np.radians(vs3["view_angle_across_deg"]))
    assert vs3["expected_shift_mm"] == pytest.approx(1.0 + 2.0 * t,
                                                     abs=1e-3)


def test_view_shift_confirms_line_at_expected_shift():
    """Dark gap drawn ~2.05 mm AWAY from the midline (synthetic rows 155-156
    vs prior row 150, 0.373 mm/row): with expected shift 2.0 mm it
    confirms; the published line is still the UNSHIFTED prior and the
    estimate records expected_shift_mm and the signed observed offsets."""
    xyz = _small_box_xyz()
    img, corners, _ = _synth_face(seam_axis="lr", gap_offset_px=6)
    res = estimate_seam(img, corners, est_cfg=_shift_cfg(2.0),
                        corners_xyz=xyz)
    assert res["axis"] == "lr"
    assert res["status"] == "ok", (res.get("confirm_reject"),
                                   res["estimate"].get("observed_offset_mm"))
    assert res["source"] in ("prior_confirmed_by_gap",
                             "prior_confirmed_by_darkline")
    for k in ("seam_left", "seam_right"):
        for a, b in zip(res["refined_endpoints_px"][k],
                        res["prior_endpoints_px"][k]):
            assert abs(a - b) < 0.05
    est = res["estimate"]
    assert est["expected_shift_mm"] == pytest.approx(2.0)
    assert res["expected_shift_mm"] == pytest.approx(2.0)
    assert est["view_shift"]["box_class"] == "small"
    assert est["confirm_target"] == "prior_plus_expected_shift"
    obs = [v for v in est["observed_offset_mm"].values() if v is not None]
    assert obs, est["observed_offset_mm"]
    for v in obs:                      # observed ~ +2 mm away from prior
        assert 1.2 < v < 2.8, est["observed_offset_mm"]
    # Unshifted agreement would have failed the 1.5 mm gate.
    un = est["agreement_unshifted"] or est["darkline_agreement_unshifted"]
    assert un["max_offset_mm"] > 1.5


def test_view_shift_rejects_line_offset_the_other_way():
    """Same gap drawn ~2.4 mm TOWARD the camera: ~4.4 mm from the expected
    apparent line -> estimated (prior not confirmed)."""
    xyz = _small_box_xyz()
    img, corners, _ = _synth_face(seam_axis="lr", gap_offset_px=-6)
    res = estimate_seam(img, corners, est_cfg=_shift_cfg(2.0),
                        corners_xyz=xyz)
    assert res["axis"] == "lr"
    assert res["status"] == "estimated", res.get("source")
    assert res["refined_endpoints_px"] is None
    assert res["estimate"]["expected_shift_mm"] == pytest.approx(2.0)
    obs = [v for v in res["estimate"]["observed_offset_mm"].values()
           if v is not None]
    for v in obs:
        assert v < 0.0


def test_view_shift_disabled_keeps_plain_prior_confirmation():
    """Shift disabled: the +2 mm line fails the plain-prior 1.5 mm gate
    (the pre-v10 small-box failure mode) while a centred gap confirms."""
    xyz = _small_box_xyz()
    img, corners, _ = _synth_face(seam_axis="lr", gap_offset_px=6)
    res = estimate_seam(img, corners, est_cfg=_shift_cfg(2.0, False),
                        corners_xyz=xyz)
    assert res["status"] == "estimated"
    assert res["estimate"]["expected_shift_mm"] == 0.0
    assert res["estimate"]["confirm_target"] == "prior"
    img2, corners2, _ = _synth_face(seam_axis="lr")
    res2 = estimate_seam(img2, corners2, est_cfg=_shift_cfg(2.0, False),
                         corners_xyz=xyz)
    assert res2["status"] == "ok"


def _one_sided_highlight_face(size=(640, 480)):
    """Noise-free face: plain cardboard (gray ~150) with a bright tape-edge
    highlight band (rows 146-149) and a slightly brighter band above it,
    NO gap. The darkest point next to the highlight is only darker than
    the highlight side (one-sided step)."""
    W, H = 400, 300
    face = np.full((H + 1, W + 1, 3), 150, np.uint8)
    face[:146, :] = 172
    face[146:150, :] = 225
    quad = [(120, 90), (520, 140), (480, 400), (90, 360)]
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    img = np.full((size[1], size[0], 3), 60, np.uint8)
    cv2.warpPerspective(face, M, (size[0], size[1]), dst=img,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return img, np.array(quad, float)


def test_darkline_rejects_one_sided_tape_edge_highlight():
    """Without the per-shoulder gate the darkest point beside a highlight
    band forms a coherent false line; the gate (valley darker than BOTH
    shoulders) drops those stations."""
    xyz = _small_box_xyz()
    img, corners = _one_sided_highlight_face()
    eps0, _, info0 = refine_on_darkline(
        img, corners, "lr", est_cfg={"darkline_min_shoulder": 0.0},
        corners_xyz=xyz)
    assert eps0 is not None, info0          # the failure mode exists
    eps, _, info = refine_on_darkline(img, corners, "lr",
                                      corners_xyz=xyz)
    assert eps is None
    assert info["n_one_sided"] > 0
    assert info["darkline_min_shoulder"] == pytest.approx(2.0)


def test_no_view_shift_cli_reproduces_pre_v10_confirmation():
    """run.py --no-view-shift must reproduce the pre-change (git HEAD
    before the view shift, all853_v8) seam results: view shift off AND the
    darkline per-shoulder gate off. Expected values below were produced by
    the pre-change estimator on these synthetic faces."""
    from box_seam_detection.core.config import default_config
    from box_seam_detection.core.run import _parse_args, apply_seam_cli_overrides

    cfg = apply_seam_cli_overrides(
        default_config(), _parse_args(["--run-name", "x",
                                       "--no-view-shift"]))
    se = cfg["seam_estimator"]
    assert se["view_shift"]["enabled"] is False
    assert se["darkline_min_shoulder"] == 0.0
    only_gate = apply_seam_cli_overrides(
        default_config(), _parse_args(["--run-name", "x",
                                       "--no-shoulder-gate"]))
    assert only_gate["seam_estimator"]["darkline_min_shoulder"] == 0.0
    assert only_gate["seam_estimator"]["view_shift"]["enabled"] is True
    default = default_config()["seam_estimator"]
    assert default["view_shift"]["enabled"] is True
    assert default["darkline_min_shoulder"] == 2.0

    xyz = _small_box_xyz()
    expected = {   # pre-change: status, source, gap / darkline max mm
        0: ("ok", "prior_confirmed_by_gap", None, 0.203),
        6: ("estimated", "prior", 2.025, 2.082),
        -6: ("estimated", "prior", 2.441, 2.527),
    }
    for off, (st, src, gmm, dmm) in expected.items():
        img, corners, _ = _synth_face(seam_axis="lr", gap_offset_px=off)
        res = estimate_seam(img, corners, est_cfg=se, corners_xyz=xyz)
        assert (res["status"], res["source"]) == (st, src), off
        assert res["estimate"]["expected_shift_mm"] == 0.0
        assert res["estimate"]["confirm_target"] == "prior"
        assert res["darkline_agreement"]["max_offset_mm"] == \
            pytest.approx(dmm, abs=1e-3), off
        if gmm is not None:
            assert res["agreement"]["max_offset_mm"] == \
                pytest.approx(gmm, abs=1e-3), off
        if st == "ok":
            for k in ("seam_left", "seam_right"):
                for a, b in zip(res["refined_endpoints_px"][k],
                                res["prior_endpoints_px"][k]):
                    assert abs(a - b) < 0.05
    # Pre-change darkline accepted the one-sided highlight line (rect
    # intercept 139.5, coverage 1.0); gate off reproduces it exactly.
    img, corners = _one_sided_highlight_face()
    eps, _, info = refine_on_darkline(img, corners, "lr", est_cfg=se,
                                      corners_xyz=xyz)
    assert eps is not None
    assert info["intercept_rect"] == pytest.approx(139.5, abs=1e-6)
    assert info["coverage"] == pytest.approx(1.0)
    assert info["n_one_sided"] == 0


# ---- seam_publish="observed" (large_392) -------------------------------------------------

def _xyz_rect(w=0.40, h=0.30, z=1.2):
    # BL, BR, FR, FL in the camera frame (a flat 400 x 300 mm face)
    return {"back_left": [0.0, 0.0, z], "back_right": [w, 0.0, z],
            "front_right": [w, h, z], "front_left": [0.0, h, z]}


def test_observed_profile_publishes_the_gap_not_the_midline():
    img, corners, gt = _synth_face(seam_axis="lr", gap_offset_px=6)
    xyz = _xyz_rect()
    res = estimate_seam(img, corners, corners_xyz=xyz, profile="large_392")
    assert res["status"] == "ok" and res["source"] == "observed_gap"
    pub = res["refined_endpoints_px"]
    eps = np.array([pub["seam_left"], pub["seam_right"]])
    assert (np.linalg.norm(eps[0] - gt[0]) + np.linalg.norm(eps[1] - gt[1])) / 2.0 > 3.0  # not the midline
    mid = np.array(projective_midlines(corners)["lr"]["endpoints"])
    assert abs(eps[:, 1].mean() - mid[:, 1].mean()) > 3.0
    inner = res["estimate"]
    assert inner["publish"] == "observed" and inner["published_line"] == "observed_gap"
    assert res["seam_curvature_mm"] is not None and res["seam_curvature_mm"] < 1.0   # straight synthetic gap
    assert len(inner["observed_points_px"]) > 50
    assert abs(inner["gap_shape"]["offset_median_mm"]) < 0.3     # the line sits on the gap points
    # the same face under a legacy profile keeps publishing the (projective) prior
    old = estimate_seam(img, corners, corners_xyz=xyz, profile="small")
    assert old.get("source") != "observed_gap" and "seam_curvature_mm" not in old


def test_observed_profile_without_gap_is_estimated():
    img, corners, _ = _synth_face(tape=True, gap=False)
    res = estimate_seam(img, corners, corners_xyz=_xyz_rect(), profile="large_392")
    assert res["status"] != "ok"
    if res["status"] == "estimated":
        assert res["source"] == "prior" and res["refined_endpoints_px"] is None


def test_gap_shape_curvature_and_outlier_robustness():
    from box_seam_detection.core.seam_estimator import gap_shape
    t = np.arange(0.0, 400.0, 2.0)
    bow = 4.0 * (1.0 - ((t - 200.0) / 200.0) ** 2)            # 4 px bow
    line = (0.0, 2.0)                                          # published line sits at the ends
    g = gap_shape(t, 2.0 + bow, line, 1.0)
    assert 3.9 < g["curvature_mm"] < 4.1
    spike = np.full_like(t, 2.0)
    spike[50] += 20.0                                          # one bad slice is not curvature
    assert gap_shape(t, spike, line, 1.0)["curvature_mm"] < 0.1
    assert gap_shape(t[:3], t[:3], line, 1.0) is None and gap_shape(t, bow, line, None) is None
