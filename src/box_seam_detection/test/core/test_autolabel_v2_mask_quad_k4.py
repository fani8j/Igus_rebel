# Provenance: ported from rgbd_captures/tests/test_autolabel_v2_mask_quad_k4.py (HEAD 6df5994); imports made package-relative.
"""_mask_to_quad min-area-k4 fallback for a cut/crushed corner (synthetic)."""
import sys
from pathlib import Path

import cv2
import numpy as np


from box_seam_detection.core import sam3_worker as w

H, W = 480, 640


def _rot_rect(angle=35.0, c=(320, 240), size=(260, 160)):
    r = cv2.boxPoints(((c[0], c[1]), size, angle))
    return r.astype(np.float64)


def _fill(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m


def _cut(rect, k=0, frac=0.22):
    """Rectangle polygon with corner k replaced by a chord (two points)."""
    a, b, c = rect[k - 1], rect[k], rect[(k + 1) % 4]
    p1 = b + (a - b) * frac
    p2 = b + (c - b) * frac
    pts = [rect[i] for i in range(4)]
    return np.array(pts[:k] + [p1, p2] + pts[k + 1:])


def _match(q, rect):
    d = [min(np.linalg.norm(p - r) for r in rect) for p in q]
    return max(d)


def test_cut_corner_restored(monkeypatch):
    # approxPolyDP happily returns a (degraded) 4-gon for clean pentagons, so
    # force it to find nothing to exercise the fallback path deterministically.
    monkeypatch.setattr(w, "APPROX_EPS", ())
    rect = _rot_rect()
    q, iou, method = w._mask_to_quad_ex(_fill(_cut(rect, 2)), H, W)
    assert method == "min_area_k4"
    assert q.shape == (4, 2)
    assert iou >= w.K4_MIN_IOU
    assert _match(q, rect) <= 2.5


def test_cut_each_corner(monkeypatch):
    monkeypatch.setattr(w, "APPROX_EPS", ())
    for ang in (-20.0, 10.0, 35.0):
        rect = _rot_rect(angle=ang)
        for k in range(4):
            q, _, method = w._mask_to_quad_ex(_fill(_cut(rect, k)), H, W)
            assert method == "min_area_k4"
            assert _match(q, rect) <= 2.5, (ang, k)


def test_rounded_crushed_corner_hull_reduction():
    rect = _rot_rect()
    a, b, c = rect[1], rect[2], rect[3]
    p1, p2 = b + (a - b) * 0.2, b + (c - b) * 0.2
    mid = (p1 + p2) / 2
    arc = [(1 - s) ** 2 * p1 + 2 * s * (1 - s) * (mid + (b - mid) * .3)
           + s * s * p2 for s in np.linspace(0, 1, 6)]
    poly = np.array([rect[0], rect[1]] + arc + [rect[3]])
    hull = cv2.convexHull(np.round(poly).astype(np.int32)).reshape(-1, 2)
    q = w._hull_min_area_k4(hull)
    assert q is not None and len(q) == 4
    assert _match(q, rect) <= 3.0


def test_wrapper_two_tuple_and_success_unchanged():
    rect = _rot_rect()
    m = _fill(rect)
    q, iou, method = w._mask_to_quad_ex(m, H, W)
    assert method == "approx_poly"
    q2, iou2 = w._mask_to_quad(m, H, W)
    assert np.array_equal(q, q2) and iou == iou2


def test_garbage_blob_none():
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    r = 40 + rng.uniform(0, 120, 40)
    pts = np.stack([320 + r * np.cos(ang), 240 + r * np.sin(ang)], 1)
    q, iou, method = w._mask_to_quad_ex(_fill(pts), H, W)
    assert q is None and method is None
    q, iou = w._mask_to_quad(np.zeros((H, W), np.uint8), H, W)
    assert q is None
