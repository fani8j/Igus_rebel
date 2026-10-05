#!/usr/bin/env python3
"""Decide how the labelled spine connects to the front and back faces (by image evidence).

For every box it enumerates the physically possible configurations
  front photo -> +Z   in each rotation that maps the photo's height edge to X
  back  photo -> -Z   likewise
  spine image -> as rectified from source/photos, or rotated 180 deg
and scores how well the artwork continues across the two spine folds:
  fold F: front.png top row   <->  top.png top row    (top.png reversed: right = -X there, +X in front.png)
  fold B: back.png  top row   <->  top.png bottom row (both have +X on the left)
A 6 px strip just inside each fold line (the textures are already trimmed by
OUTPUT_INSET_PX = 2) is reduced to one Lab colour per sample along the fold.  Metrics:
  dE   mean CIEDE2000 per sample (lower is better), also with the median colour offset
       between the two sides removed (dE0), since the photos have different exposure
  corr Pearson correlation of the smoothed L/a/b colour profiles along the fold
Only the sum over the two folds is ranked.  Mirror images are not physical (both faces
are photographed from outside) and are not enumerated.  The tuck tab ("Open here") edges
(TAB_EDGE, in the rectified photo frame) give an independent check.

Run:  python3 sim_assets/medical_boxes/tools/match_net.py [--write-previews] [box ...]
Writes usd/previews/<box>_net_match.png with --write-previews.
Geometry of the net (all with +X at the top of the picture): back | spine | front, spine
rotated 90 deg CW, front 90 deg CCW, back 90 deg CW, so the shared folds touch.
"""
import itertools
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_textures as ex  # noqa: E402

ROOT = ex.ROOT
PREV = os.path.join(ROOT, "usd", "previews")
STRIP = (2, 8)            # rows (counted from the fold edge) averaged for the colour samples
SMOOTH_SIGMA = 3.0        # px, for the 1-D profile correlation

# Tuck tab ("Open here" / "Push here to open") edge as seen in the rectified photo frame
# (T/B/L/R of the upright photo; hand-marked on source/face_photos/*, None = not visible).
TAB_EDGE = {
    "agent_monorail":    {"front": None, "back": "R"},   # back: blue 'Push here to open' tab at right
    "angiolite":         {"front": None, "back": None},  # no tab visible on either big face
    "accuforce":         {"front": None, "back": "R"},   # back: 'Open here' tab at right
    "ultimaster_nagomi": {"front": "R",  "back": "R"},   # front: flap notch at right; back: 'Open here'
    "conqueror_nc_pro":  {"front": None, "back": "R"},   # back: 'OPEN HERE' flap at right
}


# --------------------------------------------------------------------------- colour
def srgb_to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def ciede2000(lab1, lab2):
    """Vectorised CIEDE2000 (Sharma et al.), arrays (..., 3)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hm = np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                  np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    hm = np.where(C1p * C2p == 0, hs, hm)
    T = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    dth = 30 * np.exp(-(((hm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# --------------------------------------------------------------------------- geometry
def load_dims(box):
    with open(os.path.join(ROOT, "boxes.yaml")) as f:
        return yaml.safe_load(f)["boxes"][box]["dims_mm"]


CYC = ["T", "L", "B", "R"]    # edge moves one step along this list per CCW quarter turn


def edge_after(edge, k):
    return None if edge is None else CYC[(CYC.index(edge) + k) % 4]


# box edge reached by a texture edge (T/B/L/R of front.png / back.png)
FRONT_EDGE = {"T": "+Y", "B": "-Y", "R": "+X", "L": "-X"}
BACK_EDGE = {"T": "+Y", "B": "-Y", "L": "+X", "R": "-X"}


def face_candidates(box, side):
    """[(label, aspect, k, tex_bgr, valid)] : photo rotated k quarter turns CCW."""
    out = []
    for asp in ex.ASPECT_CANDIDATES.get(box, ("L",)):
        P, V = ex.photo_face(box, side, asp)
        ks = (1, 3) if asp == "L" else (0, 2)
        for k in ks:
            tex = np.ascontiguousarray(np.rot90(P, k))
            val = np.ascontiguousarray(np.rot90(V, k))
            out.append(("%s%s k=%d (%s)" % (side[0].upper(), asp, k,
                        {0: "photo as is", 1: "photo rot 90 CCW", 2: "photo rot 180",
                         3: "photo rot 90 CW"}[k]), asp, k, tex, val))
    return out


def samples(strip_bgr):
    """Rows STRIP of an edge-first strip -> Lab per column (W,3)."""
    return srgb_to_lab(strip_bgr[STRIP[0]:STRIP[1]]).mean(axis=0)


def fold_metrics(a_lab, b_lab, valid):
    """a_lab, b_lab: (W,3) Lab samples along the fold, aligned; valid: (W,) bool."""
    a, b = a_lab[valid], b_lab[valid]
    if len(a) < 50:
        return dict(dE=np.nan, dE0=np.nan, corr=np.nan, n=len(a))
    de = ciede2000(a, b)
    off = np.median(b - a, axis=0)
    de0 = ciede2000(a + off, b)
    cs = []
    for c in range(3):
        sa = cv2.GaussianBlur(a[:, c].reshape(1, -1), (0, 0), SMOOTH_SIGMA).ravel()
        sb = cv2.GaussianBlur(b[:, c].reshape(1, -1), (0, 0), SMOOTH_SIGMA).ravel()
        if sa.std() < 1e-3 or sb.std() < 1e-3:
            cs.append(0.0)
        else:
            cs.append(float(np.corrcoef(sa, sb)[0, 1]))
    return dict(dE=float(de.mean()), dE0=float(de0.mean()), corr=float(np.mean(cs)),
                n=int(valid.sum()))


def spine_variants(box):
    S = ex.rectify_spine(box, load_dims(box))
    return {0: S, 180: np.ascontiguousarray(np.rot90(S, 2))}


def analyse(box):
    S = spine_variants(box)
    fronts, backs = face_candidates(box, "front"), face_candidates(box, "back")
    ones = np.ones(S[0].shape[1], bool)
    rows = []
    fold = {}
    for sf, Sv in S.items():
        s_top = samples(Sv)                     # top row = front fold (strip from the top edge)
        s_bot = samples(Sv[::-1])               # bottom row = back fold
        for fi, f in enumerate(fronts):
            tex, val = f[3], f[4]
            fl = samples(tex)
            v = val[STRIP[0]:STRIP[1]].all(axis=0)
            fold[("F", sf, fi)] = fold_metrics(fl, s_top[::-1], v)   # spine reversed: +X right
        for bi, b in enumerate(backs):
            tex, val = b[3], b[4]
            bl = samples(tex)
            v = val[STRIP[0]:STRIP[1]].all(axis=0)
            fold[("B", sf, bi)] = fold_metrics(bl, s_bot, v)
    for sf in S:
        for fi, f in enumerate(fronts):
            for bi, b in enumerate(backs):
                mf, mb = fold[("F", sf, fi)], fold[("B", sf, bi)]
                tf = edge_after(TAB_EDGE[box]["front"], f[2])
                tb = edge_after(TAB_EDGE[box]["back"], b[2])
                ef = FRONT_EDGE[tf] if tf else None
                eb = BACK_EDGE[tb] if tb else None
                rows.append(dict(sf=sf, fi=fi, bi=bi, f=f[0], b=b[0],
                                 dE=(mf["dE"] + mb["dE"]) / 2, dE0=(mf["dE0"] + mb["dE0"]) / 2,
                                 corr=(mf["corr"] + mb["corr"]) / 2,
                                 mf=mf, mb=mb, tab_f=ef, tab_b=eb,
                                 tab_ok=(ef is None or eb is None or ef == eb)))
    return S, fronts, backs, rows


def rank(rows, tab_only=True):
    """Ascending dE.  Configurations whose front and back tuck-tab edges land on different
    box edges are physically impossible and are dropped when tab_only."""
    rs = [r for r in rows if r["tab_ok"] or not tab_only]
    return sorted(rs, key=lambda r: (r["dE"] if not np.isnan(r["dE"]) else 1e9))


def confidence(ranked):
    """clear: margin >= 3 dE and both corr and dE0 agree with the winner; weak: margin >= 1 dE
    (or two of the three agree); else undecided.  Also returns the spine-flip-only margin."""
    a, b = ranked[0], ranked[1]
    m = b["dE"] - a["dE"]
    agree = (a["corr"] >= b["corr"]) + (a["dE0"] <= b["dE0"])
    if m >= 3 and agree == 2:
        lab = "clear"
    elif m >= 1 and agree >= 1:
        lab = "weak"
    else:
        lab = "undecided"
    other = [r for r in ranked if r["sf"] != a["sf"] and r["fi"] == a["fi"] and r["bi"] == a["bi"]]
    flip_m = other[0]["dE"] - a["dE"] if other else float("nan")
    return lab, m, flip_m


def describe(r):
    return "spine %s | front %s | back %s" % ("as photographed" if r["sf"] == 0 else "rot 180",
                                             r["f"], r["b"])


def report(box, S, fronts, backs, rows, top=6):
    ranked = rank(rows)
    print("=" * 100)
    print(box)
    print("%-3s %-8s %-30s %-30s %7s %7s %7s  tabF tabB ok" %
          ("#", "spine", "front", "back", "dE", "dE0", "corr"))
    for i, r in enumerate(ranked[:top]):
        print("%-3d %-8s %-30s %-30s %7.2f %7.2f %7.3f  %-4s %-4s %s" %
              (i + 1, "as-is" if r["sf"] == 0 else "rot180", r["f"], r["b"], r["dE"], r["dE0"],
               r["corr"], r["tab_f"], r["tab_b"], r["tab_ok"]))
    lab, m, fm = confidence(ranked)
    w = ranked[0]
    print("margin best-vs-2nd dE %.2f | spine-flip-only margin %.2f | confidence: %s" % (m, fm, lab))
    print("winner: %s | tab edges F=%s B=%s (spine is +Y => 'spine opposite tab' %s)" % (
        describe(w), w["tab_f"], w["tab_b"],
        "n/a (no tab seen)" if not (w["tab_f"] or w["tab_b"]) else
        str(all(t == "-Y" for t in (w["tab_f"], w["tab_b"]) if t))))
    unc = rank(rows, tab_only=False)[0]
    if unc is not w:
        print("unconstrained best (tab-inconsistent, excluded): dE %.2f %s" % (unc["dE"], describe(unc)))
    for k, m_ in (("front fold", w["mf"]), ("back fold", w["mb"])):
        print("  %s: dE %.2f dE0 %.2f corr %.3f valid samples %d" % (k, m_["dE"], m_["dE0"], m_["corr"], m_["n"]))
    return ranked


# --------------------------------------------------------------------------- net preview
def net_row(S, front_tex, back_tex, scale):
    """back | spine | front, +X at the top, shared folds touching (no gaps)."""
    sp = np.rot90(S, -1)          # CW: spine top row (front fold) -> right, left end (+X) -> top
    fr = np.rot90(front_tex, 1)   # CCW: top row -> left, right side (+X) -> top
    bk = np.rot90(back_tex, -1)   # CW: top row -> right, left side (+X) -> top
    h = max(sp.shape[0], fr.shape[0], bk.shape[0])
    def pad(x):
        return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = np.hstack([pad(bk), pad(sp), pad(fr)])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def write_preview(box, S, fronts, backs, ranked, scale=0.22):
    def used(r):
        return (r["sf"] == (180 if ex.SPINE_FLIP[box] else 0)
                and fronts[r["fi"]][1:3] == (ex.FACE_ORIENT[box]["front"])
                and backs[r["bi"]][1:3] == (ex.FACE_ORIENT[box]["back"]))
    cur = [r for r in ranked if used(r)][0]
    alt = [r for r in ranked if r is not cur][0]
    rowimgs = []
    for tag, r in (("TEXTURES (rank %d)" % (ranked.index(cur) + 1), cur),
                   ("RUNNER-UP (rank %d)" % (ranked.index(alt) + 1), alt)):
        img = net_row(S[r["sf"]], fronts[r["fi"]][3], backs[r["bi"]][3], scale)
        txt = "%s dE %.1f corr %.2f spine %s F k=%d B k=%d  [back|spine|front, +X up]" % (
            tag, r["dE"], r["corr"], "as-is" if r["sf"] == 0 else "rot180",
            fronts[r["fi"]][2], backs[r["bi"]][2])
        rowimgs.append(label(img, txt))
    w = max(i.shape[1] for i in rowimgs)
    rowimgs = [cv2.copyMakeBorder(i, 0, 6, 0, w - i.shape[1], cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255)) for i in rowimgs]
    out = np.vstack(rowimgs)
    os.makedirs(PREV, exist_ok=True)
    path = os.path.join(PREV, "%s_net_match.png" % box)
    cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return path


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ex.FACE_CORNERS)
    for name in names:
        S, fronts, backs, rows = analyse(name)
        ranked = report(name, S, fronts, backs, rows)
        if "--write-previews" in sys.argv:
            print("wrote", write_preview(name, S, fronts, backs, ranked))
