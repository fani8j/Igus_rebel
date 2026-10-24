"""Known-size rectangle fit score (shadow diagnostic, 2026-10-12).

Ported (idea and scoring, re-expressed for our units) from the conveyor box
detector of a colleague, ``conveyor_box/rgbd_detector.py::_fit_partition``
(minAreaRect yaw candidates, 2-98 % extents for the centre, score =
inside_ratio * exp(-mean outside distance / tol) * min(occupancy / 0.60, 1) on
a 10 x 6 grid). Not imported; used on the top-plane points of one box in the
plane frame, tolerances scaled to this box (``tol_mm``, default 4 mm instead
of their 25 mm). The result is only RECORDED: advisors measured that the top
points inside the plane band stop 8-29 mm short of the edges and the top is
tilted/bulged, so the fit cannot place a corner to millimetres; the score is a
sanity check of "is this blob the known box" and the corner disagreement with
the published quad is a diagnostic.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

GRID_X, GRID_Y = 10, 6
COVERAGE_FULL = 0.60


def fit(points_xy: np.ndarray, length_mm: float, width_mm: float,
        tol_mm: float = 4.0) -> dict | None:
    """Best known-size rectangle for 2-D points (mm). ``length_mm`` >= ``width_mm``.

    Returns ``{score, inside_ratio, coverage, outside_mean_mm, centre,
    yaw_deg, corners (4, 2)}`` or None when the points do not allow a fit.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 20:
        return None
    rect = cv2.minAreaRect(pts.astype(np.float32))
    measured = np.asarray(rect[1], dtype=np.float64)
    if np.any(measured <= 0) or not np.all(np.isfinite(measured)):
        return None
    free = np.asarray(cv2.boxPoints(rect), dtype=np.float64)
    edges = np.roll(free, -1, axis=0) - free
    lengths = np.linalg.norm(edges, axis=1)
    longest = int(np.argmax(lengths))
    seed = float(math.atan2(edges[longest, 1], edges[longest, 0]) % math.pi)
    best = None
    tol = max(float(tol_mm), 1e-3)
    # yaw candidates: the minAreaRect seed of the colleague's code (+ 90 deg), plus the
    # principal axes of the points (the minAreaRect of a point set with a ragged fringe
    # can be rotated by several degrees; the fit itself never refines the yaw)
    _w, vec = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
    pca = float(math.atan2(vec[1, 1], vec[0, 1]) % math.pi)
    cands = []
    for y in (seed, (seed + math.pi / 2.0) % math.pi, pca, (pca + math.pi / 2.0) % math.pi):
        if all(abs(((y - c0) + math.pi / 2) % math.pi - math.pi / 2) > 1e-6 for c0 in cands):
            cands.append(y)
    for yaw in cands:
        rot = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        proj = pts @ rot.T
        lo, hi = np.percentile(proj, (2.0, 98.0), axis=0)
        centre = ((lo + hi) / 2.0) @ rot
        local = (pts - centre) @ rot.T
        ox = np.maximum(np.abs(local[:, 0]) - length_mm / 2.0, 0.0)
        oy = np.maximum(np.abs(local[:, 1]) - width_mm / 2.0, 0.0)
        out_d = np.hypot(ox, oy)
        inside = out_d <= tol
        if not np.any(inside):
            continue
        il = local[inside]
        xb = np.clip(((il[:, 0] / length_mm + 0.5) * GRID_X).astype(int), 0, GRID_X - 1)
        yb = np.clip(((il[:, 1] / width_mm + 0.5) * GRID_Y).astype(int), 0, GRID_Y - 1)
        occ = np.zeros((GRID_Y, GRID_X), dtype=bool)
        occ[yb, xb] = True
        cov = float(occ.mean())
        score = float(inside.mean()) * math.exp(-float(out_d.mean()) / tol) \
            * min(cov / COVERAGE_FULL, 1.0)
        if best is None or score > best["score"]:
            c, s = math.cos(yaw), math.sin(yaw)
            r2 = np.array([[c, -s], [s, c]])
            loc = np.array([[-length_mm / 2, -width_mm / 2], [length_mm / 2, -width_mm / 2],
                            [length_mm / 2, width_mm / 2], [-length_mm / 2, width_mm / 2]])
            best = {"score": score, "inside_ratio": float(inside.mean()),
                    "coverage": cov, "outside_mean_mm": float(out_d.mean()),
                    "centre": centre, "yaw_deg": math.degrees(yaw),
                    "corners": loc @ r2.T + centre}
    return best


def corner_disagreement_mm(fit_corners: np.ndarray, quad_corners: np.ndarray) -> list:
    """Per quad corner: distance (mm) to the nearest fitted corner under the
    best cyclic/reflected matching (the fit's corner order is arbitrary)."""
    f = np.asarray(fit_corners, dtype=np.float64).reshape(4, 2)
    q = np.asarray(quad_corners, dtype=np.float64).reshape(4, 2)
    best = None
    for rev in (False, True):
        ff = f[::-1] if rev else f
        for k in range(4):
            fk = np.roll(ff, k, axis=0)
            dist = np.linalg.norm(fk - q, axis=1)
            if best is None or dist.max() < best.max():
                best = dist
    return [round(float(x), 2) for x in best]
