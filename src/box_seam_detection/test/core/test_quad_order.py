# Provenance: ported from rgbd_captures/tests/test_quad_order.py (HEAD 6df5994); imports made package-relative.
"""Focused tests for order_corners (scripts/autolabel/quad).

The sums/differences heuristic mapped two output corners to one input
point on strongly oblique quads (a collapsed triangle with a zero-length
edge that validate_quad could only reject). The centroid-angle ordering
must return [TL, TR, BR, BL] for any four distinct points: permutation
invariant, four distinct corners, legacy-compatible start corner, and
honest ValueErrors on degenerate input (wrong count, non-finite,
duplicates) so callers reject instead of measuring fabricated geometry.
"""
import itertools
import sys

import cv2
import numpy as np
import pytest


from box_seam_detection.core.autolabel.quad import order_corners, quad_area  # noqa: E402


def _assert_canonical(q):
    q = np.asarray(q, dtype=float).reshape(4, 2)
    assert len({tuple(p) for p in np.round(q, 6)}) == 4
    assert bool(cv2.isContourConvex(q.astype(np.float32)))
    sums = q[:, 0] + q[:, 1]
    assert q[0][0] + q[0][1] == sums.min()  # TL is the legacy min-sum corner
    tied = q[sums == sums.min()]
    assert q[0][0] == tied[:, 0].min()  # ...tie-break smallest x


def test_axis_rect_permutation_invariant():
    quad = np.array([[100.0, 100.0], [500.0, 100.0],
                     [500.0, 400.0], [100.0, 400.0]])
    outs = {tuple(order_corners(quad[list(p)]).ravel())
            for p in itertools.permutations(range(4))}
    assert len(outs) == 1
    np.testing.assert_allclose(
        order_corners(quad),
        [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]])


def test_oblique_quad_permutation_invariant():
    quad = np.array([[500.0, 300.0], [780.0, 260.0],
                     [860.0, 460.0], [540.0, 540.0]])
    outs = {tuple(np.round(order_corners(quad[list(p)]), 6).ravel())
            for p in itertools.permutations(range(4))}
    assert len(outs) == 1
    _assert_canonical(order_corners(quad))


def test_diamond_quad_starts_at_min_sum_corner():
    quad = np.array([[640.0, 150.0], [950.0, 400.0],
                     [640.0, 650.0], [330.0, 400.0]])
    outs = {tuple(np.round(order_corners(quad[list(p)]), 6).ravel())
            for p in itertools.permutations(range(4))}
    assert len(outs) == 1
    out = order_corners(quad)
    # Legacy min-sum corner (left vertex), repeated for no input: the old
    # heuristic also picked it, but only by luck of distinct extrema.
    np.testing.assert_allclose(out[0], [330.0, 400.0])
    _assert_canonical(out)


def _old_order(quad):
    """The pre-fix sums/differences heuristic, for back-compat comparison."""
    q = np.asarray(quad, dtype=float).reshape(4, 2)
    s = q.sum(axis=1)
    d = np.diff(q, axis=1).ravel()
    return np.stack([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]])


def test_matches_legacy_where_legacy_was_unambiguous():
    """Wherever the old heuristic produced four distinct corners, the new
    ordering must be identical: the fix changes behavior ONLY on inputs
    the old code collapsed (rectification correspondence preserved)."""
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(600):
        cx, cy = rng.uniform(200, 1000, 2)
        angs = np.sort(rng.uniform(0, 2 * np.pi, 4))
        r = rng.uniform(80, 400)
        quad = np.stack([cx + r * np.cos(angs), cy + r * np.sin(angs)], axis=1)
        if not bool(cv2.isContourConvex(quad.astype(np.float32))):
            continue
        s = quad.sum(axis=1)
        d = np.diff(quad, axis=1).ravel()
        idx = [int(np.argmin(s)), int(np.argmin(d)),
               int(np.argmax(s)), int(np.argmax(d))]
        if len(set(idx)) < 4:
            continue  # ambiguous: old code duplicated; new code defines it
        np.testing.assert_allclose(order_corners(quad), _old_order(quad),
                                   rtol=0, atol=1e-9)
        checked += 1
    assert checked > 100


def test_sum_diff_collision_quad_yields_four_distinct_corners():
    """Convex quad where argmax(sum) == argmax(diff): the old heuristic
    returned the same point for BR and BL (zero-length edge)."""
    quad = np.array([[485.4, 196.8], [557.2, 212.6],
                     [377.4, 598.7], [371.0, 374.4]])
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    assert int(np.argmax(s)) == int(np.argmax(d))  # old code duplicated here
    out = order_corners(quad)
    assert len({tuple(p) for p in np.round(out, 6)}) == 4
    _assert_canonical(out)
    # Full area preserved: the old output collapsed to ~half (a triangle).
    assert quad_area(out) == pytest.approx(quad_area(quad[np.argsort(
        np.arctan2(quad[:, 1] - quad[:, 1].mean(),
                   quad[:, 0] - quad[:, 0].mean()), kind="stable")]), rel=1e-9)


def test_random_convex_quads_permutation_invariant():
    rng = np.random.default_rng(7)
    quads = []
    while len(quads) < 15:
        # Cyclic quads (sorted angles on a circle) are strictly convex.
        cx, cy = rng.uniform(200, 1000, 2)
        angs = np.sort(rng.uniform(0, 2 * np.pi, 4))
        r = rng.uniform(80, 400)
        quads.append(np.stack([cx + r * np.cos(angs),
                               cy + r * np.sin(angs)], axis=1))
    while len(quads) < 30:
        # Affine images of the unit square: parallelograms with strong
        # obliquity, strictly convex by construction.
        while True:
            m = rng.uniform(-2, 2, (2, 2))
            if abs(np.linalg.det(m)) > 0.5:
                break
        t = rng.uniform(100, 800, 2)
        quads.append((np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ m.T) * 200 + t)
    for quad in quads:
        assert bool(cv2.isContourConvex(quad.astype(np.float32)))
        outs = {tuple(np.round(order_corners(quad[list(p)]), 6).ravel())
                for p in itertools.permutations(range(4))}
        assert len(outs) == 1
        _assert_canonical(order_corners(quad))


@pytest.mark.parametrize("bad", [
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],                      # 3 points
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2, 2]],  # 5 points
    [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],          # duplicate
    [[0.0, 0.0], [1.0, 0.0], [1.0 + 1e-9, 0.0], [0.0, 1.0]],   # coincident
    [[np.inf, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],       # non-finite
    [[np.nan, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],       # non-finite
])
def test_degenerate_input_raises(bad):
    with pytest.raises(ValueError):
        order_corners(np.asarray(bad, dtype=float))
