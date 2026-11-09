"""Colour-only qualitative detector for recorded video frames (no depth, no metric output).

Used for the 2026-10-20 clips, which have colour only. It finds, in PIXELS:
  * the strongest long straight edge between cardboard and background (the cut-side top edge), and
  * dark-line (flap gap) candidates perpendicular to it, scored by black-hat column response.
It makes no metric claim and no model assumption. On wrinkled glossy tape the seam ranking is weak;
labels, printing and wrinkles produce false candidates. The result is for human review only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .calib import DEFAULT_TOOL_POLYGON_PX
from .detect import tool_mask_full


@dataclass
class SeamCandidate:
    p_top: Tuple[float, float]       # pixel at the far (inner) end of the search band
    p_edge: Tuple[float, float]      # pixel where it meets the edge line
    confidence: float
    peak: float


@dataclass
class ColorOnlyResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    edge_p0: Optional[Tuple[float, float]] = None
    edge_p1: Optional[Tuple[float, float]] = None
    edge_score: float = 0.0
    edge_support_px: float = 0.0
    side_contrast: float = 0.0       # cardboard-coloured pixel % on the carton side minus the other side
    seams: List[SeamCandidate] = field(default_factory=list)
    other_edges: list = field(default_factory=list)

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else [round(float(v[0]), 1), round(float(v[1]), 1)]
        return {"ok": self.ok, "reasons": self.reasons, "edge_p0": r(self.edge_p0),
                "edge_p1": r(self.edge_p1), "edge_score": round(self.edge_score, 1),
                "edge_support_px": round(self.edge_support_px, 1),
                "side_contrast": round(self.side_contrast, 1),
                "seams": [{"p_top": r(s.p_top), "p_edge": r(s.p_edge),
                           "confidence": round(s.confidence, 2), "peak": round(s.peak, 2)}
                          for s in self.seams]}


def _line_through(seg_pts: np.ndarray):
    """Unit direction and a point of the best line through points (N,2)."""
    vx, vy, x0, y0 = cv2.fitLine(seg_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([vx, vy]), np.array([x0, y0])


def detect_color_only(rgb: np.ndarray, orientation: str = "any", tool_polygon_px=None,
                      n_seams: int = 3) -> ColorOnlyResult:
    res = ColorOnlyResult(ok=False)
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        res.reasons.append("bad_image")
        return res
    H, W = rgb.shape[:2]
    s = W / 1280.0
    tool = tool_mask_full(W, H, tool_polygon_px if tool_polygon_px is not None else DEFAULT_TOOL_POLYGON_PX)
    tool_d = cv2.dilate(tool.astype(np.uint8), np.ones((int(15 * s) | 1, int(15 * s) | 1), np.uint8)).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(gray, (0, 0), 1.6)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag[tool_d] = 0
    hsv0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cardboard = (hsv0[..., 0] >= 3) & (hsv0[..., 0] <= 30) & (hsv0[..., 1] >= 30) & (hsv0[..., 2] >= 90)
    # ---- edge: long straight segments, scored by length x gradient along them
    edges = cv2.Canny(g, 30, 90)
    edges[tool_d] = 0
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, int(70 * s), minLineLength=int(220 * s),
                           maxLineGap=int(30 * s))
    if segs is None:
        res.reasons.append("no_long_straight_edge")
        return res
    cand = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        horiz = min(ang, 180 - ang) < 35
        vert = abs(ang - 90) < 35
        if orientation == "horizontal" and not horiz:
            continue
        if orientation == "vertical" and not vert:
            continue
        if not (horiz or vert):
            continue
        L = np.hypot(x2 - x1, y2 - y1)
        t = np.linspace(0, 1, max(int(L / 4), 2))
        xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, W - 1)
        ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, H - 1)
        strength = float(mag[ys, xs].mean())
        # cardboard on one side of the segment and not on the other (hue 5-30, saturated)
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        fr = []
        for sgn in (+1, -1):
            xx = np.clip((xs[::3, None] + sgn * nx * np.arange(12, 60, 6)[None, :] * s).astype(int), 0, W - 1)
            yy = np.clip((ys[::3, None] + sgn * ny * np.arange(12, 60, 6)[None, :] * s).astype(int), 0, H - 1)
            fr.append(float(cardboard[yy, xx].mean()))
        contrast = abs(fr[0] - fr[1])
        cand.append((L * min(strength, 40.0) * (0.05 + max(fr) * (0.2 + contrast)), L, ang, (x1, y1, x2, y2)))
    if not cand:
        res.reasons.append("no_long_straight_edge")
        return res
    cand.sort(key=lambda c: -c[0])
    best = cand[0]
    # merge collinear segments with the best one
    d0, p0 = _line_through(np.array([[best[3][0], best[3][1]], [best[3][2], best[3][3]]], float))
    nrm = np.array([-d0[1], d0[0]])
    pts = []
    support = 0.0
    for sc, L, ang, (x1, y1, x2, y2) in cand:
        da = abs(((ang - best[2]) + 90) % 180 - 90)
        e1 = abs((np.array([x1, y1]) - p0) @ nrm)
        e2 = abs((np.array([x2, y2]) - p0) @ nrm)
        if da <= 4 and max(e1, e2) <= 12 * s:
            pts += [[x1, y1], [x2, y2]]
            support += L
    d, p0 = _line_through(np.array(pts, float))
    nrm = np.array([-d[1], d[0]])
    proj = (np.array(pts) - p0) @ d
    pa, pb = p0 + proj.min() * d, p0 + proj.max() * d
    res.edge_p0, res.edge_p1 = tuple(pa), tuple(pb)
    res.edge_score = best[0]
    res.edge_support_px = support
    # ---- carton side (info only): more cardboard-coloured pixels in a 15-85 px strip on one side
    mid = (pa + pb) / 2
    sides = []
    for sgn in (+1, -1):
        vals = []
        for off in np.linspace(15 * s, 85 * s, 8):
            for tt in np.linspace(-0.4, 0.4, 9):
                p = mid + tt * (pb - pa) + sgn * off * nrm
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < W and 0 <= y < H and not tool_d[y, x]:
                    vals.append(float(cardboard[y, x]))
        sides.append(float(np.mean(vals)) if vals else 0.0)
    res.side_contrast = abs(sides[0] - sides[1]) * 100.0
    # ---- other long edges (distinct from the best), for the reviewer
    for sc, L, ang, (x1, y1, x2, y2) in cand[1:]:
        q = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if abs((q - p0) @ nrm) < 30 * s:
            continue
        if any(abs((q - (np.array(e[0]) + np.array(e[1])) / 2) @ nrm) < 30 * s for e in res.other_edges):
            continue
        res.other_edges.append(((x1, y1), (x2, y2)))
        if len(res.other_edges) >= 2:
            break
    # ---- seam candidates: dark lines perpendicular to the edge family, searched over +-18 deg of tilt
    # (independent of which long edge was picked: only the orientation hint of the edge is used)
    vertical_seams = abs(((np.degrees(np.arctan2(d[1], d[0])) % 180) + 90) % 180 - 90) < 45
    work_img = gray if vertical_seams else cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    work_valid = (~tool_d) if vertical_seams else cv2.rotate((~tool_d).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
    # close the cardboard mask so a thin dark seam line / tape glare inside the carton stays in it
    card_closed = cv2.morphologyEx(cardboard.astype(np.uint8), cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(41 * s) | 1, int(41 * s) | 1))) > 0
    work_card = card_closed if vertical_seams else cv2.rotate(card_closed.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
    wh, ww = work_img.shape
    # rows that contain the carton: cardboard-coloured fraction >= 0.25 of the valid columns
    cardrows = (work_card & work_valid).mean(axis=1) > 0.15
    rows = np.nonzero(cardrows)[0]
    if len(rows) < 80 * s:
        res.reasons.append("no_carton_band")
        return res
    y0, y1_ = int(rows.min()), int(rows.max())
    kw = int(round(31 * s)) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    kg = np.exp(-0.5 * (np.arange(-8, 9) / 2.5) ** 2)
    kg /= kg.sum()
    cands = []
    centre = (ww / 2.0, (y0 + y1_) / 2.0)
    for phi in range(-18, 19, 3):
        M = cv2.getRotationMatrix2D(centre, phi, 1.0)
        wi = cv2.warpAffine(work_img, M, (ww, wh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wv = cv2.warpAffine(work_valid.astype(np.uint8), M, (ww, wh), flags=cv2.INTER_NEAREST) > 0
        wc = cv2.warpAffine(work_card.astype(np.uint8), M, (ww, wh), flags=cv2.INTER_NEAREST) > 0
        m = wv & wc
        bi = wi[y0:y1_ + 1]
        mb = m[y0:y1_ + 1]
        if mb.mean() < 0.1:
            continue
        med = float(np.median(bi[mb]))
        fl = np.where(mb, bi, int(med)).astype(np.uint8)
        bh = cv2.morphologyEx(cv2.GaussianBlur(fl, (0, 0), 1.0), cv2.MORPH_BLACKHAT, kern).astype(np.float64)
        wt = mb.astype(np.float64)
        cover = wt.mean(axis=0)
        prof = (bh * wt).sum(0) / np.maximum(wt.sum(0), 1)
        prof[cover < 0.35] = np.nan
        prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof) if np.isfinite(prof).any() else 0)
        sm = np.convolve(np.pad(prof, 8, mode="edge"), kg, mode="valid")
        medb = float(np.median(sm))
        bgd = medb + float(np.median(np.abs(sm - medb))) * 1.4826 + 1.0
        for pk in np.argsort(sm)[::-1][:6]:
            cands.append((float(sm[pk] / bgd), float(sm[pk]), phi, int(pk), M, cover[pk]))
    cands.sort(key=lambda c: -c[0])
    chosen = []
    for conf, peak, phi, pk, M, cov in cands:
        Minv = cv2.invertAffineTransform(M)
        top = Minv @ np.array([pk, y0, 1.0])
        bot = Minv @ np.array([pk, y1_, 1.0])
        # same line as an already chosen one? compare at mid height
        midp = (top + bot) / 2
        if any(np.hypot(*(midp - c[2])) < 45 * s for c in chosen):
            continue
        chosen.append((conf, peak, midp, top, bot))
        if len(chosen) >= n_seams:
            break
    for conf, peak, midp, top, bot in chosen:
        if not vertical_seams:                                  # map back from the 90-degree rotation
            top = np.array([top[1], H - 1 - top[0]])
            bot = np.array([bot[1], H - 1 - bot[0]])
        res.seams.append(SeamCandidate(tuple(top), tuple(bot), conf, peak))
    res.ok = True
    return res


def draw_color_only(rgb: np.ndarray, res: ColorOnlyResult, label: str = "") -> np.ndarray:
    img = rgb.copy()
    h, w = img.shape[:2]
    if res.edge_p0 is not None:
        cv2.line(img, tuple(np.rint(res.edge_p0).astype(int)), tuple(np.rint(res.edge_p1).astype(int)),
                 (0, 220, 255), 3, cv2.LINE_AA)
    for (e0, e1) in res.other_edges:
        cv2.line(img, tuple(np.rint(e0).astype(int)), tuple(np.rint(e1).astype(int)), (200, 120, 255), 2, cv2.LINE_AA)
    for i, sc in enumerate(res.seams):
        col = (40, 230, 60) if i == 0 else (200, 200, 60)
        cv2.line(img, tuple(np.rint(sc.p_top).astype(int)), tuple(np.rint(sc.p_edge).astype(int)), col,
                 3 if i == 0 else 1, cv2.LINE_AA)
        cv2.putText(img, f"{i + 1}: {sc.confidence:.1f}", tuple(np.rint(sc.p_top).astype(int) + [4, 14]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    lines = ["COLOR-ONLY: pixels only, no metric output", label]
    if res.ok:
        lines.append(f"edge support {res.edge_support_px:.0f}px side contrast {res.side_contrast:.0f}")
        if res.seams:
            lines.append("seam candidates (conf): " + " ".join(f"{s.confidence:.1f}" for s in res.seams))
    else:
        lines.append("FAILED: " + ", ".join(res.reasons))
    y = 8
    for t in lines:
        if not t:
            continue
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (6, y), (12 + tw, y + th + 8), (0, 0, 0), -1)
        cv2.putText(img, t, (9, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += th + 10
    return img
