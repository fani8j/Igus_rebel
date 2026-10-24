# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/tape_strip.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Tape-strip detector on a rectified top face (pure functions, no I/O).

Input: the rectified top face (BGR, e.g. from ``seam_estimator.rectify_face``
+ warp) and optionally the metric scale (mm per rect px).  Output: a
JSON-serialisable dict of strip candidates (see ``detect_tape_strips``).

Design (see docs/multi_box_plan.md, "Advisor review" / M4):

* A tape strip runs parallel to a box edge, so only the two face axes are
  searched ("h": strip along rect x, cross coordinate y, published seam axis
  "lr"; "v": strip along rect y, cross coordinate x, axis "bf"), each with a
  small yaw search (+-3 deg in 0.5 deg steps; the image is deskewed about the
  face centre and the 1-D cross profile re-evaluated).
* Per axis/yaw, robust 1-D cross profiles (median along the strip over the
  middle span) of: saturation, brightness (V) and paired-edge gradient
  energy.  A strip hypothesis (position a, width w) is scored per channel by
  its PAIRED contrast (inside vs. both flanks, same sign on both sides, so a
  single step such as a face border scores 0) and z-scored against the
  face's own hypothesis distribution (median / MAD with an absolute noise
  floor).  Greedy non-maximum suppression keeps several strips per axis.
* The centreline comes from the two strip EDGES only (derivative extrema of
  the weighted channel profile, per block along the strip, robust line fit),
  never from brightness peaks, so a specular streak cannot move it.
* The gap valley (narrow dark line, the physical flap gap) is searched inside
  the strip and reported separately with its offset from the centreline.
* ``ok`` needs a strip with >= 2 agreeing channels (sat/val/edge z >= z_agree),
  edge coverage >= min_coverage and an axis margin >= min_margin.

Conventions: rect pixel centres are integer coordinates (cv2).  ``yaw_deg``
is the line rotation relative to its ideal axis, positive counter-clockwise
as displayed (image y down).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CHANNELS = ("sat", "val", "edge")

DEFAULTS: dict = {
    "min_width_mm": 10.0,
    "max_width_mm": 80.0,
    # used when no metric scale is known (fraction of the cross size)
    "min_width_frac": 0.02,
    "max_width_frac": 0.35,
    "yaw_max_deg": 3.0,
    "yaw_step_deg": 0.5,
    "top_k": 6,
    "z_agree": 5.0,
    "z_cap": 12.0,
    "floor_sat": 2.0,
    "floor_val": 2.0,
    "floor_edge": 0.6,
    "floor_gap": 2.0,
    "gap_z_min": 3.0,
    "n_blocks": 6,
    "span": (0.2, 0.8),          # central span for the scan profile
    "along": (0.04, 0.96),       # extent used for blocks / coverage
    # a strip edge this close to the face boundary is the flap region between
    # a tape edge and the face edge, not a tape strip
    "border_margin_px": 4.0,
    "border_margin_frac": 0.08,
    "yaw_slack_deg": 1.0,        # residual slope allowed beyond yaw_max
    "min_coverage": 0.6,
    "min_margin": 0.25,
    "nms_overlap": 0.5,
    "width_ratio": 1.08,
    "width_max_slack": 1.1,   # grid extends this far past max_width
}


# ---------------------------------------------------------------- helpers

def mm_per_px_from_xyz(corners_xyz, W: int, H: int):
    """(mm per rect px along x, along y) from the 4 top-corner xyz (metres),
    corners in [back_left, back_right, front_right, front_left] order
    (dict name->xyz or (4,>=3) array).  None if unavailable."""
    names = ("back_left", "back_right", "front_right", "front_left")
    try:
        if isinstance(corners_xyz, dict):
            xm = np.array([corners_xyz[n] for n in names], float)
        else:
            xm = np.asarray(corners_xyz, float).reshape(4, -1)
        if not np.all(np.isfinite(xm)) or W <= 0 or H <= 0:
            return None
        sx = (np.linalg.norm(xm[1, :3] - xm[0, :3])
              + np.linalg.norm(xm[2, :3] - xm[3, :3])) / 2.0
        sy = (np.linalg.norm(xm[3, :3] - xm[0, :3])
              + np.linalg.norm(xm[2, :3] - xm[1, :3])) / 2.0
        return (float(sx * 1000.0 / W), float(sy * 1000.0 / H))
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _mm_pair(mm_per_px):
    if mm_per_px is None:
        return None
    if isinstance(mm_per_px, (int, float)):
        v = float(mm_per_px)
        return (v, v) if v > 0 else None
    try:
        a, b = float(mm_per_px[0]), float(mm_per_px[1])
        return (a, b) if a > 0 and b > 0 else None
    except (TypeError, ValueError, IndexError):
        return None


def rect_points_to_image(Mi, pts) -> list:
    """Map rect-face points (N,2) to original image px with the inverse
    homography ``Mi`` (as returned by rectify_face)."""
    p = np.asarray(pts, float).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(p, np.asarray(Mi, float))
    return [[float(x), float(y)] for x, y in out.reshape(-1, 2)]


def candidate_lines_image(result: dict, Mi) -> list:
    """For every candidate: its centreline, edges and gap line endpoints in
    original image px (list of dicts parallel to result["candidates"])."""
    out = []
    for c in result.get("candidates", []):
        d = {"centre": rect_points_to_image(Mi, c["line_rect"]),
             "edge_a": rect_points_to_image(Mi, c["edge_lines_rect"][0]),
             "edge_b": rect_points_to_image(Mi, c["edge_lines_rect"][1])}
        g = c.get("gap_line")
        d["gap"] = rect_points_to_image(Mi, g["line_rect"]) if g else None
        out.append(d)
    return out


def _cfg(cfg):
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    return c


def _mad(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _zscore(vals, valid, floor, sigma=0.0, base=0.0):
    """(vals - base) / max(sigma, floor) over the valid entries.  ``sigma``
    is the face's own robust noise of the profile (not of the hypothesis
    distribution: a wide strip would inflate that)."""
    z = np.zeros_like(vals)
    z[valid] = (vals[valid] - base) / max(float(sigma), floor)
    return z


def _profile_sigma(p, k=31):
    """Robust noise of a 1-D profile: 1.4826*MAD of p - median_filter(p, k)
    (texture + slow blotches; edges and strips are sparse outliers)."""
    p = np.asarray(p, np.float64)
    if p.size < k:
        k = max(3, p.size // 2 * 2 - 1)
    pad = k // 2
    pp = np.pad(p, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pp, k)
    return _mad(p - np.median(win, axis=1))


# ------------------------------------------------------------ frames

class _Frame:
    """Deskewed channel images in the strip frame of one axis
    (cross coordinate = rows, along = columns)."""

    def __init__(self, sat, val, gray, axis):
        if axis == "v":
            sat, val, gray = sat.T, val.T, gray.T
        self.sat = np.ascontiguousarray(sat)
        self.val = np.ascontiguousarray(val)
        self.gray = np.ascontiguousarray(gray)
        self.n_cross, self.n_along = self.gray.shape

    def central_profiles(self, span, close_px: int = 7) -> dict:
        i0 = int(span[0] * self.n_along)
        i1 = max(i0 + 1, int(span[1] * self.n_along))
        raw = {k: np.median(getattr(self, k)[:, i0:i1], axis=1)
               for k in ("sat", "val", "gray")}
        return _clean_profiles(raw, close_px)

    def block_profiles(self, nb, along, close_px: int = 7) -> tuple:
        L = self.n_along
        e = np.linspace(int(along[0] * L), int(along[1] * L), nb + 1).astype(int)
        e[1:] = np.maximum(e[1:], e[:-1] + 1)
        out = {}
        for k in ("sat", "val", "gray"):
            A = getattr(self, k)
            raw = np.stack([np.median(A[:, e[j]:e[j + 1]], axis=1)
                            for j in range(nb)])
            out[k + "_raw"] = raw
            out[k] = np.stack([_close1(r, close_px) for r in raw])
        xc = (e[:-1] + e[1:] - 1) / 2.0
        return out, xc


def _close1(p, k):
    """1-D grey closing (removes dark valleys narrower than k px, i.e. the
    flap gap / adhesive rim, so they cannot split a strip into halves)."""
    if k <= 1:
        return p
    ker = np.ones((k, 1), np.uint8)
    x = p.reshape(-1, 1).astype(np.float32)
    return cv2.morphologyEx(x, cv2.MORPH_CLOSE, ker,
                            borderType=cv2.BORDER_REPLICATE)[:, 0]


def _clean_profiles(raw, close_px):
    """Add closed (gap-free) profiles and the paired-edge energy to the raw
    median profiles.  ``sat``/``val`` are the closed versions used for
    scoring; ``*_raw`` keep the originals; ``grad`` is the edge energy
    max(|d gray|, |d sat|) of the closed profiles (central difference)."""
    out = {}
    for k in ("sat", "val", "gray"):
        out[k + "_raw"] = raw[k]
        out[k] = _close1(raw[k], close_px)
    def cd(p):
        d = np.zeros_like(p, dtype=np.float64)
        d[1:-1] = np.abs(p[2:] - p[:-2]) / 2.0
        return d
    out["grad"] = np.maximum(cd(out["gray"]), cd(out["sat"]))
    return out


def _rot_images(sat, val, gray, yaw_deg):
    """Deskew by rotating the content so a line at ccw angle ``yaw_deg``
    becomes axis-aligned.  Returns (images, forward 2x3 matrix)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -yaw_deg, 1.0)
    if abs(yaw_deg) < 1e-9:
        return (sat, val, gray), M
    st = np.dstack([sat, val, gray])
    r = cv2.warpAffine(st, M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return (r[:, :, 0], r[:, :, 1], r[:, :, 2]), M


# ---------------------------------------------------------- hypotheses

def _width_grid(wmin, wmax, ratio):
    ws, w = [], float(max(wmin, 3))
    while w <= wmax:
        ws.append(int(round(w)))
        w *= ratio
        if int(round(w)) == ws[-1]:
            w = ws[-1] + 1
    return sorted(set(ws))


def _csum(p):
    return np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])


def _hypotheses(prof, widths, c) -> list:
    """All (a, w) strip hypotheses of one profile set, z-scored per channel.
    Returns a list of dicts (best few per scan, NMS applied by the caller)."""
    n = len(prof["gray"])
    cs_sat, cs_val = _csum(prof["sat"]), _csum(prof["val"])
    e = prof["grad"]
    e_s = cv2.dilate(e.reshape(-1, 1).astype(np.float32),
                     np.ones((3, 1), np.uint8))[:, 0]
    floors = {"sat": c["floor_sat"], "val": c["floor_val"],
              "edge": c["floor_edge"]}
    sigma = {"sat": _profile_sigma(prof["sat"]),
             "val": _profile_sigma(prof["val"]),
             "edge": _mad(e), }
    e_med = float(np.median(e))
    A_all, W_all, S_all, Z_all, SG_all = [], [], [], [], []
    for w in widths:
        if w >= n - 8:
            continue
        a = np.arange(2, n - w - 1)
        if a.size < 3:
            continue
        b = a + w
        f = int(np.clip(round(0.5 * w), 3, 12))
        ls = np.maximum(0, a - 2 - f)
        ll = (a - 2) - ls
        re = np.minimum(n, b + 2 + f)
        rl = re - (b + 2)
        valid = (ll >= 3) & (rl >= 3)
        if valid.sum() < 3:
            continue
        ia, ib = (a + 1, b - 1) if w >= 6 else (a, b)
        zs, sgn = [], []
        for cs, name in ((cs_sat, "sat"), (cs_val, "val")):
            m_in = (cs[ib] - cs[ia]) / np.maximum(ib - ia, 1)
            m_l = (cs[a - 2] - cs[ls]) / np.maximum(ll, 1)
            m_r = (cs[re] - cs[b + 2]) / np.maximum(rl, 1)
            dl, dr = m_in - m_l, m_in - m_r
            same = (dl * dr) > 0
            p = np.where(same, np.minimum(np.abs(dl), np.abs(dr)), 0.0)
            sgn.append(np.where(same, np.sign(dl), 0.0))
            zs.append(_zscore(p, valid, floors[name], sigma[name]))
        pe = np.minimum(e_s[a], e_s[b - 1]).astype(np.float64)
        zs.append(_zscore(pe, valid, floors["edge"], sigma["edge"], e_med))
        sgn.append(np.zeros_like(pe))
        Z = np.stack(zs)                       # (3, na)
        # soft cap: strong channels must not tie across yaws / strips
        Zp = np.maximum(Z, 0.0)
        cap = c["z_cap"]
        S = np.where(Zp <= cap, Zp, cap + 0.1 * (Zp - cap)).sum(axis=0)
        S[~valid] = -1.0
        A_all.append(a)
        W_all.append(np.full(a.shape, w))
        S_all.append(S)
        Z_all.append(Z)
        SG_all.append(np.stack(sgn))
    if not A_all:
        return []
    A = np.concatenate(A_all)
    Wd = np.concatenate(W_all)
    S = np.concatenate(S_all)
    Z = np.concatenate(Z_all, axis=1)
    SG = np.concatenate(SG_all, axis=1)
    kk = min(400, S.size)
    idx = np.argpartition(-S, kk - 1)[:kk]
    idx = idx[np.argsort(-S[idx])]
    keep, ov = [], c["nms_overlap"]
    for i in idx:
        if S[i] <= 0:
            break
        lo, hi = int(A[i]), int(A[i] + Wd[i])
        clash = False
        for j in keep:
            o = min(hi, j["b"]) - max(lo, j["a"])
            if o > 0 and o / float(min(hi - lo, j["b"] - j["a"])) >= ov:
                clash = True
                break
        if clash:
            continue
        keep.append({"a": lo, "b": hi, "S": float(S[i]),
                     "z": Z[:, i].copy(), "sign": SG[:, i].copy()})
        if len(keep) >= 2 * c["top_k"]:
            break
    return keep


def _overlap_clash(a, b, others, ov):
    for o in others:
        x = min(b, o["b"]) - max(a, o["a"])
        if x > 0 and x / float(min(b - a, o["b"] - o["a"])) >= ov:
            return True
    return False


# ------------------------------------------------------------- refinement

def _mid(nb):
    """Slice of the central ~60 % of the blocks."""
    lo = int(round(0.2 * nb))
    hi = max(lo + 1, int(round(0.8 * nb)))
    return slice(lo, hi)


def _smooth1(x, sigma=1.0):
    k = int(math.ceil(3 * sigma))
    t = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    xp = np.pad(x, k, mode="edge")
    return np.convolve(xp, g, mode="valid")


def _peak_pos(d, lo, hi, sign):
    """Sub-pixel location (between-pixel convention: index i+0.5 for a step
    between pixels i and i+1) of the extreme of ``sign*d`` in [lo, hi)."""
    lo, hi = max(0, lo), min(len(d), hi)
    if hi - lo < 1:
        return None, 0.0
    seg = sign * d[lo:hi]
    k = int(np.argmax(seg))
    i = lo + k
    off = 0.0
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return i + 0.5 + off, float(seg[k])


def _theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    sl = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[j] - x[i]) > 1e-9:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(sl)) if sl else 0.0


def _refine(frame, hyp, c, scales) -> dict | None:
    """Edge positions per block (deskewed frame) -> robust line fit."""
    nb = int(c["n_blocks"])
    B, xc = frame.block_profiles(nb, c["along"])
    a0, b0 = hyp["a"], hyp["b"]
    w0 = b0 - a0
    z = hyp["z"]
    # weighted signed combination of the colour channels
    wts, sgs = [], []
    for k, name in enumerate(("sat", "val")):
        wts.append(float(z[k]) if z[k] >= 2.0 and hyp["sign"][k] != 0 else 0.0)
        sgs.append(float(hyp["sign"][k]) if hyp["sign"][k] != 0 else 1.0)
    if sum(wts) <= 0:
        # edge-energy only: use the stronger of sat/val by its scan z
        k = int(np.argmax(z[:2]))
        wts[k] = 1.0
        if hyp["sign"][k] == 0:
            # sign from the raw profile contrast
            pass
    q = np.zeros_like(B["sat"])
    for k, name in enumerate(("sat", "val")):
        if wts[k] > 0:
            q += (wts[k] * sgs[k] / scales[name]) * B[name]
    # sign ambiguity when sign==0 (no paired contrast): derive from profile
    r = max(3, int(round(0.35 * w0)))
    mid = (a0 + b0) / 2.0
    a_pos, b_pos, strength = [], [], []
    kk = int(min(7, max(3, w0 // 3)))
    ker = np.ones((kk, 1), np.uint8)
    for j in range(nb):
        # open + close: drops narrow bright streaks / dark rims so only the
        # strip's own steps remain (the strip is positive after sign alignment)
        qj = q[j].reshape(-1, 1).astype(np.float32)
        qj = cv2.morphologyEx(qj, cv2.MORPH_OPEN, ker,
                              borderType=cv2.BORDER_REPLICATE)
        qj = cv2.morphologyEx(qj, cv2.MORPH_CLOSE, ker,
                              borderType=cv2.BORDER_REPLICATE)
        qs = _smooth1(qj[:, 0], 1.0)
        d = np.diff(qs)
        pa, sa = _peak_pos(d, int(a0 - r), int(min(a0 + r, mid)), +1.0)
        pb, sb = _peak_pos(d, int(max(b0 - 1 - r, mid)), int(b0 - 1 + r), -1.0)
        a_pos.append(pa)
        b_pos.append(pb)
        strength.append(min(sa, sb))
    a_pos = np.array([np.nan if v is None else v for v in a_pos])
    b_pos = np.array([np.nan if v is None else v for v in b_pos])
    strength = np.array(strength)
    ok = np.isfinite(a_pos) & np.isfinite(b_pos) & (b_pos > a_pos)
    if ok.sum() < 3:
        return None
    # block consistency
    ma, mb = np.median(a_pos[ok]), np.median(b_pos[ok])
    tol = max(2.0, 0.2 * (mb - ma))
    # reference edge strength = that of the CENTRAL profile (median of the
    # central blocks), so a pattern that exists in only a few blocks (label,
    # occluder) cannot define its own reference
    qc = np.median(q[_mid(nb)], axis=0).reshape(-1, 1)
    qc = cv2.morphologyEx(qc.astype(np.float32), cv2.MORPH_OPEN, ker,
                          borderType=cv2.BORDER_REPLICATE)
    qc = cv2.morphologyEx(qc, cv2.MORPH_CLOSE, ker,
                          borderType=cv2.BORDER_REPLICATE)
    dc = np.diff(_smooth1(qc[:, 0], 1.0))
    _, sca = _peak_pos(dc, int(a0 - r), int(min(a0 + r, mid)), +1.0)
    _, scb = _peak_pos(dc, int(max(b0 - 1 - r, mid)), int(b0 - 1 + r), -1.0)
    ref_str = min(sca, scb)
    cons = (ok & (np.abs(a_pos - ma) <= tol) & (np.abs(b_pos - mb) <= tol)
            & (strength >= 0.4 * ref_str) & (strength > 0))
    if cons.sum() < 3:
        return None
    cen = (a_pos + b_pos) / 2.0
    wid = b_pos - a_pos
    slope = _theil_sen(xc[cons], cen[cons])
    xmid = frame.n_along / 2.0
    c_mid = float(np.median(cen[cons] - slope * (xc[cons] - xmid)))
    width = float(np.median(wid[cons]))
    return {"center": c_mid, "width": width, "slope": slope,
            "coverage": float(cons.sum()) / nb, "xc": xc,
            "block_centers": cen, "consistent": cons,
            "n_blocks": nb, "B": B}


def _gap_in_strip(frame, ref, c, hyp_scan_scale) -> dict | None:
    """Narrow dark valley inside the strip band (deskewed frame)."""
    B, xc = ref["B"]["gray_raw"], ref["xc"]
    nb = B.shape[0]
    cen, wid, slope = ref["center"], ref["width"], ref["slope"]
    xmid = frame.n_along / 2.0
    a = cen - wid / 2.0
    b = cen + wid / 2.0

    def valley(g):
        gs = _smooth1(g, 0.7)
        n = len(gs)
        cs = _csum(gs)
        v = np.full(n, -1e9)
        for i in range(7, n - 7):
            m = ((cs[i - 2] - cs[i - 7]) + (cs[i + 8] - cs[i + 3])) / 10.0
            v[i] = m - gs[i]
        return v

    vc = valley(np.median(B[_mid(nb)], axis=0))
    finite = vc > -1e8
    if finite.sum() < 10:
        return None
    lo, hi = int(math.ceil(a + 1.5)), int(math.floor(b - 1.5))
    if hi - lo < 1:
        return None
    seg = vc[lo:hi + 1]
    k = int(np.argmax(seg))
    i = lo + k
    vmax = float(seg[k])
    z = (vmax - float(np.median(vc[finite]))) / max(_mad(vc[finite]),
                                                    c["floor_gap"])
    if z < c["gap_z_min"] or vmax <= 0:
        return None
    # half-depth width of the valley on the smoothed profile
    gs = _smooth1(np.median(B[_mid(nb)], axis=0), 0.7)
    base = gs[i] + vmax
    half = gs[i] + 0.5 * vmax
    lft = i
    while lft > 0 and gs[lft] < half:
        lft -= 1
    rgt = i
    while rgt < len(gs) - 1 and gs[rgt] < half:
        rgt += 1
    width_px = float(rgt - lft)
    # sub-pixel parabola on the valley score
    off = 0.0
    if 0 < i < len(vc) - 1:
        y0, y1, y2 = vc[i - 1], vc[i], vc[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    pos0 = i + off
    # per-block consistency of the valley position (relative to the strip
    # line so a yaw residual does not matter)
    pos = []
    for j in range(nb):
        vj = valley(B[j])
        shift = slope * (xc[j] - xmid)
        l2 = int(max(0, math.floor(pos0 + shift - 3)))
        h2 = int(min(len(vj), math.ceil(pos0 + shift + 4)))
        if h2 - l2 < 2:
            continue
        kk = int(np.argmax(vj[l2:h2]))
        if vj[l2 + kk] > 0:
            pos.append((j, l2 + kk - shift))
    ok_blocks = [p for j, p in pos if abs(p - pos0) <= 1.5]
    cov = len(ok_blocks) / float(nb)
    return {"center": float(pos0), "z": float(z), "width_px": width_px,
            "coverage": float(cov), "offset_px": float(pos0 - cen),
            "inside_band": bool(abs(pos0 - cen) <= 0.25 * wid)}


# ------------------------------------------------------------------ main

def _line_endpoints_rect(axis, c_mid, slope, M_fwd, W, H):
    """Endpoints (rect px, original frame) of the deskewed-frame line
    cross=c_mid + slope*(along-xmid), mapped back through the inverse of the
    deskew affine."""
    Mi = cv2.invertAffineTransform(M_fwd)
    if axis == "h":
        xm = (W + 1) / 2.0
        pts = [(0.0, c_mid + slope * (0.0 - xm)),
               (float(W), c_mid + slope * (float(W) - xm))]
    else:
        ym = (H + 1) / 2.0
        pts = [(c_mid + slope * (0.0 - ym), 0.0),
               (c_mid + slope * (float(H) - ym), float(H))]
    out = []
    for x, y in pts:
        out.append([float(Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2]),
                    float(Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2])])
    return out


def _yaw_of(axis, p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if axis == "h":
        if dx < 0:
            dx, dy = -dx, -dy
        return math.degrees(math.atan2(-dy, dx))
    if dy < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dx, dy))


def _at_centre(axis, p0, p1, W, H):
    """Cross coordinate of a rect-frame line at the face centre."""
    if axis == "h":
        x = W / 2.0
        t = (x - p0[0]) / (p1[0] - p0[0])
        return p0[1] + t * (p1[1] - p0[1])
    y = H / 2.0
    t = (y - p0[1]) / (p1[1] - p0[1])
    return p0[0] + t * (p1[0] - p0[0])


def detect_tape_strips(face_bgr: np.ndarray, mm_per_px=None,
                       cfg: dict | None = None) -> dict:
    """Detect tape strips on a rectified top face.

    ``mm_per_px``: float or (mm/px along x, mm/px along y), or None.
    Returns {"candidates": [...], "best_axis": "lr"|"bf"|None,
    "axis_margin": float, "n_strips_per_axis": {"lr": n, "bf": n},
    "ok": bool, "reasons": [...], "face_wh": [W, H]}.
    Candidate keys: axis ("lr"|"bf"), rect_axis ("h"|"v"), center_px (cross
    coordinate of the centreline at the face centre), width_px, width_mm,
    yaw_deg, edges_px [a, b] (cross coords at the face centre), channels
    {sat,val,edge,gap z}, n_channels_agree, qualified, coverage, score,
    line_rect (centreline endpoints, rect px), edge_lines_rect, gap_line
    (None or {center_px, offset_px, z, width_px, coverage, inside_band,
    agrees_with_centre, line_rect}).
    """
    c = _cfg(cfg)
    img = np.asarray(face_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mm = _mm_pair(mm_per_px)

    yaws = np.arange(-c["yaw_max_deg"], c["yaw_max_deg"] + 1e-9,
                     c["yaw_step_deg"])
    rots = {}
    for y in yaws:
        rots[round(float(y), 3)] = _rot_images(sat, val, gray, float(y))

    per_axis = {}
    frames = {}
    for axis in ("h", "v"):
        n_cross = H if axis == "h" else W
        if mm is not None:
            mmpp = mm[1] if axis == "h" else mm[0]
            wmin = c["min_width_mm"] / mmpp
            wmax = c["max_width_mm"] / mmpp
        else:
            wmin = c["min_width_frac"] * n_cross
            wmax = c["max_width_frac"] * n_cross
        wmax = min(wmax, 0.5 * n_cross / c["width_max_slack"])
        widths = _width_grid(wmin, wmax * c["width_max_slack"],
                             c["width_ratio"])
        pooled = []
        for y in yaws:
            key = round(float(y), 3)
            (rs, rv, rg), _ = rots[key]
            fr = _Frame(rs, rv, rg, axis)
            frames[(axis, key)] = fr
            prof = fr.central_profiles(c["span"])
            for h in _hypotheses(prof, widths, c):
                h["yaw"] = key
                pooled.append(h)
        pooled.sort(key=lambda h: -h["S"])
        kept = []
        for h in pooled:
            if _overlap_clash(h["a"], h["b"], kept, c["nms_overlap"]):
                continue
            kept.append(h)
            if len(kept) >= c["top_k"]:
                break
        per_axis[axis] = (kept, n_cross, wmin, wmax)

    candidates = []
    for axis in ("h", "v"):
        kept, n_cross, wmin, wmax = per_axis[axis]
        for h in kept:
            fr = frames[(axis, h["yaw"])]
            prof = fr.central_profiles(c["span"])
            scales = {k: max(_mad(prof[k]), c["floor_" + k])
                      for k in ("sat", "val")}
            ref = _refine(fr, h, c, scales)
            if ref is None:
                # edges not consistent along the strip: keep the scan
                # hypothesis visible (coverage 0 -> never qualified)
                B, xc = fr.block_profiles(int(c["n_blocks"]), c["along"])
                ref = {"center": (h["a"] + h["b"] - 1) / 2.0,
                       "width": float(h["b"] - h["a"]), "slope": 0.0,
                       "coverage": 0.0, "xc": xc, "B": B}
            _, M_fwd = rots[h["yaw"]]
            p0, p1 = _line_endpoints_rect(axis, ref["center"], ref["slope"],
                                          M_fwd, W, H)
            yaw = _yaw_of(axis, p0, p1)
            if abs(yaw) > c["yaw_max_deg"] + c["yaw_slack_deg"]:
                continue        # strips run parallel to a box edge
            half = ref["width"] / 2.0
            ea = _line_endpoints_rect(axis, ref["center"] - half,
                                      ref["slope"], M_fwd, W, H)
            eb = _line_endpoints_rect(axis, ref["center"] + half,
                                      ref["slope"], M_fwd, W, H)
            c0 = _at_centre(axis, p0, p1, W, H)
            ca = _at_centre(axis, *ea, W, H)
            cb = _at_centre(axis, *eb, W, H)
            width_px = float(abs(cb - ca))
            lo_w, hi_w = per_axis[axis][2], per_axis[axis][3]
            if (width_px < 0.8 * lo_w
                    or width_px > 1.2 * hi_w * c["width_max_slack"]):
                continue
            mmpp = None
            if mm is not None:
                mmpp = mm[1] if axis == "h" else mm[0]
            z = h["z"]
            agree = int(sum(1 for k in range(3) if z[k] >= c["z_agree"]))
            margin_px = max(c["border_margin_px"],
                            c["border_margin_frac"] * per_axis[axis][1])
            touches = bool(min(ca, cb) < margin_px
                           or max(ca, cb) > per_axis[axis][1] - 1 - margin_px)
            gap = _gap_in_strip(fr, ref, c, scales)
            gap_d = None
            if gap is not None:
                gl = _line_endpoints_rect(axis, gap["center"], ref["slope"],
                                          M_fwd, W, H)
                gcen = _at_centre(axis, *gl, W, H)
                gap_d = {"center_px": float(gcen),
                         "offset_px": float(gcen - c0),
                         "z": gap["z"], "width_px": gap["width_px"],
                         "coverage": gap["coverage"],
                         "inside_band": gap["inside_band"],
                         "agrees_with_centre": gap["inside_band"],
                         "line_rect": gl}
            cand = {
                "axis": "lr" if axis == "h" else "bf",
                "rect_axis": axis,
                "center_px": float(c0),
                "width_px": width_px,
                "width_mm": (None if mmpp is None
                             else float(width_px * mmpp)),
                "yaw_deg": float(yaw),
                "edges_px": [float(min(ca, cb)), float(max(ca, cb))],
                "channels": {"sat": float(z[0]), "val": float(z[1]),
                             "edge": float(z[2]),
                             "gap": (gap["z"] if gap else 0.0)},
                "n_channels_agree": agree,
                "touches_face_border": touches,
                "qualified": bool(agree >= 2 and not touches
                                  and ref["coverage"] >= c["min_coverage"]),
                "coverage": float(ref["coverage"]),
                "score": float(h["S"] * ref["coverage"]),
                "score_raw": float(h["S"]),
                "line_rect": [p0, p1],
                "edge_lines_rect": [ea, eb],
                "gap_line": gap_d,
            }
            candidates.append(cand)

    # drop duplicates produced by different yaws of the same strip
    candidates.sort(key=lambda d: -d["score"])
    uniq = []
    for d in candidates:
        e = {"a": d["edges_px"][0], "b": d["edges_px"][1]}
        same = [u for u in uniq if u["rect_axis"] == d["rect_axis"]]
        if _overlap_clash(e["a"], e["b"],
                          [{"a": u["edges_px"][0], "b": u["edges_px"][1]}
                           for u in same], 0.5):
            continue
        uniq.append(d)
    candidates = _drop_between_strips(uniq)

    # ---- axis decision
    best = {"lr": 0.0, "bf": 0.0}
    nstrips = {"lr": 0, "bf": 0}
    for d in candidates:
        if d["qualified"]:
            nstrips[d["axis"]] += 1
            best[d["axis"]] = max(best[d["axis"]], d["score"])
    reasons = []
    if best["lr"] <= 0 and best["bf"] <= 0:
        best_axis, margin = None, 0.0
        reasons.append("no_tape_candidate")
    else:
        best_axis = "lr" if best["lr"] >= best["bf"] else "bf"
        other = "bf" if best_axis == "lr" else "lr"
        margin = float((best[best_axis] - best[other]) / best[best_axis])
        if margin < c["min_margin"]:
            reasons.append("ambiguous_axis")
        if nstrips[best_axis] > 1:
            reasons.append("multiple_strips_same_axis")
        if nstrips[other] > 0:
            reasons.append("strips_on_both_axes")
    ok = best_axis is not None and "ambiguous_axis" not in reasons
    return {"candidates": candidates, "best_axis": best_axis,
            "axis_margin": margin, "n_strips_per_axis": nstrips,
            "ok": bool(ok), "reasons": reasons, "face_wh": [int(W), int(H)]}


def _drop_between_strips(cands: list, tol: float = 3.0) -> list:
    """Remove "inverse strips": the bare gap between two strips is bounded by
    tape on both sides and so also scores as a paired-edge hypothesis."""
    out = []
    for x in cands:
        a, b = x["edges_px"]
        same = [y for y in cands if y is not x and y["qualified"]
                and y["rect_axis"] == x["rect_axis"]]
        left = any(abs(y["edges_px"][1] - a) <= tol for y in same)
        right = any(abs(y["edges_px"][0] - b) <= tol for y in same)
        if not (left and right):
            out.append(x)
    return out


def best_candidate(result: dict, axis: str | None = None) -> dict | None:
    """Highest-score qualified candidate (optionally on one axis)."""
    cs = [d for d in result.get("candidates", [])
          if d["qualified"] and (axis is None or d["axis"] == axis)]
    return max(cs, key=lambda d: d["score"]) if cs else None


def detect_on_image(bgr: np.ndarray, corners, corners_xyz=None,
                    cfg: dict | None = None) -> dict:
    """Convenience: rectify the quad (seam_estimator.rectify_face), detect,
    and add ``image_lines`` (candidate lines in original image px)."""
    from .seam_estimator import rectify_face
    M, Mi, W, H = rectify_face(bgr, corners)
    face = cv2.warpPerspective(bgr, M, (W + 1, H + 1), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    mm = mm_per_px_from_xyz(corners_xyz, W, H) if corners_xyz is not None \
        else None
    res = detect_tape_strips(face, mm, cfg)
    res["image_lines"] = candidate_lines_image(res, Mi)
    return res
