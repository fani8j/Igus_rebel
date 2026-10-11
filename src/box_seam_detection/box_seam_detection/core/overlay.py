# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/overlay.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Overlay PNG writer for v2 autolabels.

Top quad green, depth quad (independent check) cyan, seam magenta with
endpoint dots (or "seam: <status>" text when absent), named corners, and
a confidence + reasons footer. Seam audit lines (thin): observed gap fit
green, darkline red, confirmation target (prior + expected viewing shift)
dashed orange, with the class / view angle / shift text line.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CORNER_ORDER


def draw_overlay(bgr: np.ndarray, record: dict, debug: dict) -> np.ndarray:
    img = np.ascontiguousarray(bgr.copy())
    H, W = img.shape[:2]
    top = (record.get("top_face") or {}).get("corners_px")
    if top:
        quad = np.array([[top[n][0], top[n][1]] for n in CORNER_ORDER],
                        dtype=np.float64)
        cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        for name, color in (("back_left", (0, 255, 255)),
                            ("back_right", (0, 255, 255)),
                            ("front_right", (0, 255, 0)),
                            ("front_left", (0, 255, 0))):
            pt = (int(round(top[name][0])), int(round(top[name][1])))
            cv2.circle(img, pt, 5, color, -1)
            cv2.putText(img, name, (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    dq = debug.get("depth_quad")
    if dq is not None:
        try:
            cv2.polylines(img, [np.asarray(dq, dtype=np.float64).astype(np.int32)],
                          True, (255, 255, 0), 1)
        except cv2.error:
            pass
    def _eps(d):
        try:
            return ((float(d["seam_left"][0]), float(d["seam_left"][1])),
                    (float(d["seam_right"][0]), float(d["seam_right"][1])))
        except (TypeError, KeyError, ValueError):
            return None

    def _dashed(p0, p1, color, dashes=8):
        p0 = (float(p0[0]), float(p0[1]))
        p1 = (float(p1[0]), float(p1[1]))
        for i in range(dashes):
            t0, t1 = i / dashes, (i + 0.55) / dashes
            a = (int(round(p0[0] + (p1[0] - p0[0]) * t0)),
                 int(round(p0[1] + (p1[1] - p0[1]) * t0)))
            b = (int(round(p0[0] + (p1[0] - p0[0]) * t1)),
                 int(round(p0[1] + (p1[1] - p0[1]) * t1)))
            cv2.line(img, a, b, color, 1)

    seam = record.get("seam") or {}
    est = seam.get("estimate") or {}
    # Estimate candidates dashed: picked axis green-ish, other grey.
    cands = est.get("candidates") or []
    if isinstance(cands, list) and top:
        picked = est.get("axis")
        for cand in cands:
            eps = _eps((cand or {}).get("endpoints_px") or {})
            if eps is None:
                continue
            col = (0, 255, 0) if cand.get("axis") == picked else (140, 140, 140)
            _dashed(eps[0], eps[1], col)
            mid = (int(round((eps[0][0] + eps[1][0]) / 2)),
                   int(round((eps[0][1] + eps[1][1]) / 2)))
            cv2.putText(img, str(cand.get("axis")), mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    # Audit lines (thin): observed gap fit green, darkline red, and the
    # confirmation target prior + expected viewing shift dashed orange.
    inner = est.get("estimate") if isinstance(est.get("estimate"), dict) \
        else {}
    obs_gap = _eps(est.get("observed_line") or {})
    obs_dark = _eps(((inner.get("darkline") or {}).get("endpoints_px"))
                    or {})
    for oeps, col in ((obs_gap, (0, 200, 0)), (obs_dark, (0, 0, 255))):
        if oeps is not None:
            cv2.line(img, (int(round(oeps[0][0])), int(round(oeps[0][1]))),
                     (int(round(oeps[1][0])), int(round(oeps[1][1]))),
                     col, 1, cv2.LINE_AA)
    vshift = inner.get("view_shift") or {}
    exp = _eps(vshift.get("expected_line_endpoints_px") or {})
    if exp is not None:
        _dashed(exp[0], exp[1], (0, 165, 255), dashes=16)
    if vshift:
        off = inner.get("observed_offset_mm") or {}
        cv2.putText(
            img,
            f"{vshift.get('box_class')} view {vshift.get('view_angle_deg')}deg "
            f"exp_shift {vshift.get('expected_shift_mm')}mm "
            f"obs gap {off.get('gap')} dark {off.get('darkline')}",
            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
            cv2.LINE_AA)
    eps = (seam.get("endpoints_px") or {})
    if seam.get("status") == "ok" and eps:
        p0 = (int(round(eps["seam_left"][0])), int(round(eps["seam_left"][1])))
        p1 = (int(round(eps["seam_right"][0])), int(round(eps["seam_right"][1])))
        cv2.line(img, p0, p1, (255, 0, 255), 2)
        cv2.circle(img, p0, 5, (255, 0, 255), -1)
        cv2.circle(img, p1, 5, (255, 0, 255), -1)
        src = seam.get("source") or "?"
        cv2.putText(img, f"seam:{seam.get('axis')}:{src}", (p0[0], p0[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    elif seam.get("status") == "estimated":
        # Axis clear, gap unobserved: no cutting endpoints; draw the
        # estimate-only prior line dashed yellow + label axis/coverage.
        est = seam.get("estimate") or {}
        est_eps = (est.get("estimate_endpoints_px")
                   or est.get("prior_endpoints_px") or {})
        peps = _eps(est_eps)
        if peps is not None:
            _dashed(peps[0], peps[1], (0, 255, 255), dashes=12)
        cv2.putText(
            img,
            f"seam: estimated {est.get('axis') or '?'} "
            f"(cov={est.get('coverage')})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            cv2.LINE_AA)
    else:
        cv2.putText(img, f"seam: {seam.get('status', '?')}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    checks = record.get("checks") or {}
    msrc = (record.get("provenance") or {}).get("mask_source", "?")
    line1 = (f"{record.get('capture_id')} src={msrc} "
             f"conf={record.get('confidence')} "
             f"iou={checks.get('rgb_depth_quad_iou')} "
             f"wf={checks.get('wrong_face_suspect')}")
    cv2.putText(img, line1, (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    why = "; ".join(checks.get("reasons") or [])[:160]
    cv2.putText(img, why, (10, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def write_overlay(out_path: str | Path, bgr: np.ndarray, record: dict,
                  debug: dict) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_overlay(bgr, record, debug))
    return str(out)
