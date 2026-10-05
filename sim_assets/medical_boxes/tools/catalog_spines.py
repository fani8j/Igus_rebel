#!/usr/bin/env python3
"""Extract a clean, rectified spine texture for every catalog family (deterministic).

Run:  nice -n 19 python3 sim_assets/medical_boxes/tools/catalog_spines.py [--only fam,fam] [--debug DIR]

Input : catalog/families_draft.yaml (exemplar photo + rough spine corners),
        optional catalog/dims.yaml  {family_id: {spine_length_mm: .., thickness_mm: ..}}
Output: catalog/textures/<family_id>/top.png  (spine, 4 px/mm, text upright left->right)
        catalog/spines.yaml                   (refined corners, mm used + source, photo_top_end,
                                               visible fraction, end-band colours, grade)

Output image convention: printed text reads upright left -> right.  `photo_top_end`
says which end of the output image (left|right) was at the TOP in the photo.
Corners are stored in photo pixel coordinates in OUTPUT order (output TL, TR, BR, BL).

Per-family manual corrections live in OVERRIDES below (photo-pixel corner fixes picked on
zoomed crops, rotation, visible-part notes).  Everything is deterministic: no randomness,
fixed OpenCV calls, sorted iteration.
"""
import argparse
import math
import os
import re
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT = os.path.join(ROOT, "catalog")
PHOTOS = os.path.join(ROOT, "source", "setup_photos_full")
PX_MM = 4
TYPE_LEN = {"coronary_des": 220, "coronary_balloon_sc": 220, "coronary_balloon_nc": 220,
            "coronary_dcb": 220, "scoring_cutting_balloon": 220, "pta_balloon": 230,
            "guiding_catheter": 1000}
DEFAULT_LEN = 220

# Per-family manual data.  rot: 'ccw' (default; photo top -> output left), 'cw' (photo top ->
# output right, used where the text reads bottom->top in the photo), 'none' (box lying flat).
# quad: 4 photo (x, y) corners picked on zoomed crops, in the order of the ccw-rotated view
# (top-left, top-right, bottom-right, bottom-left of the ccw view; photo order for rot 'none'),
# replaces the rough corners from families_draft.yaml.  Corners are always snapped afterwards.
# visible: fraction of the full spine that is visible (the texture then covers only that part);
# cut: which photo end ('top'|'bottom') is cut off.  cap: maximum grade.  note: free text.
OVERRIDES = {
    'meril_nexgen_balloon': dict(redraw=True, cap='C', note='fanned stack seen from above: label band is a trapezoid and the spine edges are not parallel; printed text is skewed'),
    'medalliance_selution_slr_dcb': dict(redraw=True, cap='C', note='stack tilted ~10 deg, diagonal printed design; rows overlap in the photo so neighbour print enters the top and bottom edge'),
    'boston_scientific_mustang': dict(quad=[(3005.6, 121.0), (2947.4, 892.0), (2902.5, 892.0), (2957.5, 121.0)], fit=False, note='top box of a stack under the shelf lip; quad re-picked on 4x crops to cover the full spine thickness (red BSC end band, MUSTANG panel, red 035 end band); the stack is tilted ~4 deg'),
    'trireme_tri_wedge': dict(quad=[(308.9, 1084.2), (302.4, 1815.8), (279.9, 1815.8), (246.7, 1076.5)], fit=False),
    'medtronic_resolute_onyx': dict(rot='cw', quad=[(1049.1, 1752.7), (1035.5, 2251.2), (997.3, 2251.2), (1010.9, 1752.7)], fit=False),
    'boston_scientific_synergy_xd': dict(quad=[(1450.0, 3614.2), (1467.9, 4018.9), (1433.7, 4018.9), (1419.0, 3614.2)], fit=False),
    'ivascular_oceanus_35': dict(quad=[(816.0, 1128.9), (842.5, 1821.9), (788.8, 1821.9), (760.5, 1128.9)], fit=False),
    'bsci_wolverine_cutting_balloon': dict(quad=[(318.9, 2740.9), (385.4, 3303.8), (341.2, 3303.8), (273.8, 2740.9)], fit=False),
    'biotronik_passeo_18': dict(quad=[(1079.3, 1982.7), (1099.5, 2591.3), (1054.9, 2591.3), (1035.6, 1982.7)], fit=False, note='tilted stack, neighbour edges close; corners picked by hand'),
    'biotronik_orsiro_mission': dict(quad=[(2496.9, 2333.1), (2507.5, 2821.9), (2468.4, 2821.9), (2458.6, 2333.1)], fit=False),
    'bard_conquest_pta': dict(rot='cw', face=dict(quad=[(2323, 1438), (2472, 1436), (2418, 2745), (2269, 2745)], base='#1F63A8', stretch=0.92), note='large blue FRONT face with hanger hole and label; logo QUEST runs bottom-to-top; the crop is the box front, not a spine'),
    'boston_scientific_v18_control_wire': dict(quad=[(1940.2, 1907.4), (2069.8, 2666.5), (2023.5, 2666.5), (1893.9, 1907.4)], note='second carton of the tilted V-18 stack (300 cm, lot 2027-08-27); the rough corners covered the neighbour box'),
    'biotronik_pk_papyrus': dict(quad=[(549.5, 2380.4), (596.6, 2877.1), (559.6, 2877.1), (510.3, 2380.4)], note='lower carton of two stacked PK Papyrus; tilted'),
    'ivascular_oceanus_14_pro': dict(quad=[(1175.4, 3410.4), (1146.6, 4032.0), (1108.2, 4032.0), (1126.0, 3410.4)], visible=0.93, cut='bottom', note='bottom (iVascular) end cut by the photo edge; rough corners covered 3 cartons'),
    'medtronic_nc_euphora_edge': dict(redraw=True, rot='cw', quad=[(2948.0, 2998.9), (2959.2, 3370.8), (2924.4, 3370.8), (2932.0, 2998.9)], note='lower carton of a tilted, shadowed pair seen from above; perspective is strong', cap='C'),
    'nse_alpha_scoring_balloon': dict(rot='cw', quad=[(448.9, 3494.4), (471.5, 4006.1), (428.6, 4006.1), (412.1, 3494.4)], note='carton lies on the shelf, spine strip seen at a grazing angle; left maroon end face excluded', cap='B'),
    'orbusneich_combo_plus': dict(redraw=True, quad=[(958.3, 1062.0), (1065.4, 1625.7), (1025.4, 1625.7), (918.4, 1062.0)], cap='C', note='tilted stack seen from above; row "3.0 / 33"; neighbour rows overlap the edges'),
    'terumo_navicross': dict(rot='none', quad=[(402.4, 2693.9), (1219.5, 2710.1), (1219.5, 2775.9), (402.4, 2750.3)], note='carton lies flat; the crop is its narrow long SIDE (NaviCross, size, barcode, TERUMO), i.e. the labelled spine, not the large front face; confirmed by eye; text reads left to right', cap='B'),
    'ivascular_essential_pro_balloon': dict(redraw=True, quad=[(2452.6, 2345.1), (2463.5, 3780.6), (2349.1, 3780.6), (2352.3, 2345.1)], note='spine of the "4 mm / 30 mm" carton in a fanned stack; spine seen obliquely, edges ambiguous, printed text is skewed', cap='C'),
    'bsci_nc_emerge_monorail': dict(rot='cw'),
    'bsci_permanent_sled_bag_carton': dict(rot='cw'),
    'terumo_ryurei_balloon': dict(rot='cw'),
    'medtronic_amphirion_deep': dict(rot='cw', quad=[(2095.1, 3435.8), (2068.7, 4032.0), (2023.9, 4032.0), (2050.3, 3435.8)], fit=False, visible=0.95, cut='bottom', note='bottom (Medtronic) end cut by the photo edge'),
    'medtronic_export_advance_edge': dict(rot='cw'),
    'medtronic_pacific_plus': dict(rot='cw'),
    'medtronic_prevail_dcb': dict(rot='cw'),
    'teleflex_guideliner': dict(rot='cw', visible=0.95, cut='bottom', note='white spine with 6F tag, GuideLiner logo, REF 5571 and LOT text (verified by eye on the stack: it is the narrow labelled edge, not a plain large face)', cap='B'),
    'bsci_mach1_guiding_catheter': dict(face=dict(quad=[(1679, 1426), (1988, 1424), (1956, 2834), (1680, 2834)], base='#E4E6E7'), note='large white FRONT face of the long box (label near the top); the lower part is hidden by the bin rim'),
    'terumo_heartrail_ii_guiding_catheter': dict(face=dict(quad=[(1132, 1384), (1376, 1384), (1383, 2836), (1152, 2836)], base='#E6E8EA', extend='last_rows'), note='large white FRONT face with green Heartrail II label near the top and a blue TERUMO band below; the lower end is hidden by the bin rim'),
    'orbusneich_sapphire_3': dict(redraw=True, rot='cw', visible=0.9, cut='top', note='top end hidden behind the folder; pale-blue left part may belong to the neighbouring Conqueror carton', cap='C'),
}

# ----------------------------------------------------------------------------------------
# geometry: ordering, edge snapping
# ----------------------------------------------------------------------------------------
def photo_path(name):
    return os.path.join(PHOTOS, name if name.endswith(".jpg") else name + ".jpg")


def order_quad(q, rot):
    """Photo corners TL,TR,BR,BL -> output order (output TL,TR,BR,BL) for a rotation.
    ccw: photo rotated 90 deg CCW (photo top -> output left); cw: photo top -> output right;
    none: as is (photo left stays left); flip: 180 deg."""
    tl, tr, br, bl = [np.asarray(p, float) for p in q]
    if rot == "ccw":
        out = [tr, br, bl, tl]
    elif rot == "cw":
        out = [bl, tl, tr, br]
    elif rot == "flip":
        out = [br, bl, tl, tr]
    else:
        out = [tl, tr, br, bl]
    return np.array(out)


def photo_top_end(rot, quad_out):
    """Which end of the output image was at the photo's top (smaller y)."""
    left_y = (quad_out[0][1] + quad_out[3][1]) / 2
    right_y = (quad_out[1][1] + quad_out[2][1]) / 2
    if abs(left_y - right_y) < 1e-6:
        return "left"
    return "left" if left_y < right_y else "right"


def _strip(img, p0, p1, s_half, s_step=0.5, t_step=1.5, t_lo=0.04, t_hi=0.96):
    """Sample img along the segment p0->p1 (t) and across it (s). Returns array [S,T,3]."""
    L = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    ts = np.arange(t_lo * L, t_hi * L, t_step)
    ss = np.arange(-s_half, s_half + 1e-6, s_step)
    X = p0[0] + ts[None, :] * u[0] + ss[:, None] * n[0]
    Y = p0[1] + ts[None, :] * u[1] + ss[:, None] * n[1]
    S = cv2.remap(img, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    return S, ts, ss, u, n, L


def snap_line(img, p0, p1, half, prior=0.15, smooth=5, t_lo=0.04, t_hi=0.96):
    """Offsets (a at p0, b at p1; along the left-normal of p0->p1) of the strongest straight
    edge within +-half px of the segment.  Returns (a, b, confidence)."""
    S, ts, ss, u, n, L = _strip(img, p0, p1, half, t_lo=t_lo, t_hi=t_hi)
    g = np.linalg.norm(np.diff(S.astype(np.float32), axis=0), axis=2)      # [S-1,T]
    g = cv2.blur(g, (smooth, 1))
    cap = np.percentile(g, 85)
    g = np.minimum(g, cap)
    sg = (ss[:-1] + ss[1:]) / 2
    nS, nT = g.shape
    step = 1.0 if half <= 40 else 2.0
    offs = np.arange(-half, half + 1e-6, step)
    f = (ts - ts[0]) / max(ts[-1] - ts[0], 1)
    best = (-1e9, 0, 0)
    base = np.mean(g)
    scores = {}
    for ai, a in enumerate(offs):
        for bi, b in enumerate(offs):
            rows = np.clip(np.round((a + (b - a) * f - sg[0]) / 0.5).astype(int), 0, nS - 1)
            sc = g[rows, np.arange(nT)].mean() - prior * base * (abs(a) + abs(b)) / (2 * half)
            if sc > best[0]:
                best = (sc, a, b)
    sc, a, b = best
    conf = (sc + 0) / max(base, 1e-6)
    return a, b, conf, n


def refine_quad(img, q, long_half_frac=0.22, end_half=14):
    """q: output-ordered corners A(TL) B(TR) C(BR) D(BL). Long edges AB (top) and DC (bottom).
    Returns refined (A,B,C,D), confidences."""
    A, B, C, D = [np.array(p, float) for p in q]
    th = (np.linalg.norm(D - A) + np.linalg.norm(C - B)) / 2
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    half = max(6.0, long_half_frac * th)
    # top edge: normal chosen to point "up" (away from the quad); snap_line uses left-normal
    res = {}
    for name, (p0, p1) in {"top": (A, B), "bot": (D, C)}.items():
        a, b, conf, n = snap_line(blur, p0, p1, half)
        res[name] = (p0 + a * n, p1 + b * n, conf)
    (A2, B2, ct), (D2, C2, cb) = res["top"], res["bot"]

    def inter(p, d, q0, d2):
        M = np.array([d, -d2]).T
        t = np.linalg.solve(M, q0 - p)
        return p + t[0] * d

    def end_snap(Pt, Pb, outward):
        # rough end edge from Pt (on top line) to Pb (on bottom line)
        a, b, conf, n = snap_line(blur, Pt, Pb, end_half, t_lo=0.12, t_hi=0.88)
        return a, b, conf, n

    # end positions: move the rough end points onto the refined long lines first
    def project(P, p0, p1):
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        return p0 + d * np.dot(P - p0, d), d
    At, dt = project(A, A2, B2)
    Dt, db = project(D, D2, C2)
    Bt, _ = project(B, A2, B2)
    Ct, _ = project(C, D2, C2)
    out = []
    confs = [ct, cb]
    ends = []
    for (Pt, Pb, dtp, dbp, sign) in ((At, Dt, dt, db, 1), (Bt, Ct, dt, db, -1)):
        a, b, conf, n = end_snap(Pt, Pb, sign)
        # n is the left-normal of the end segment Pt->Pb; shift along the long direction
        # by projecting the normal offset onto the long axis.
        ends.append((Pt, Pb, dtp, dbp, a, b, n, conf))
    pts = []
    for (Pt, Pb, dtp, dbp, a, b, n, conf) in ends:
        # points offset along n by a, b; intersect with the long lines
        Pt2, Pb2 = Pt + a * n, Pb + b * n
        d_end = Pb2 - Pt2
        P1 = inter(Pt2, d_end, A2, dtp)
        P2 = inter(Pt2, d_end, D2, dbp)
        pts.append((P1, P2))
        confs.append(conf)
    A3, D3 = pts[0]
    B3, C3 = pts[1]
    return np.array([A3, B3, C3, D3]), confs


# ----------------------------------------------------------------------------------------
# dims
# ----------------------------------------------------------------------------------------
def parse_mm(v):
    """Number from an estimate that may be a string like '140-170 (nominal 155)' or '>=190'."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().lower()
    if t.startswith("unknown") or t == "none":
        return None
    m = re.search(r"nominal\s*([\d.]+)", t)
    if m:
        return float(m.group(1))
    m = re.match(r"^[~>=<\s]*([\d.]+)\s*(?:\(|$)", t)       # leading number, then paren/end
    if m:
        return float(m.group(1))
    m = re.match(r"^[~>=<\s]*([\d.]+)\s*-\s*([\d.]+)", t)    # range
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"([\d.]+)", t)
    return float(m.group(1)) if m else None


def resolve_dims(fid, fam, aspect, dims_override):
    """Returns (length_mm, thickness_mm, source string) for the FULL spine."""
    if fid in dims_override:
        d = dims_override[fid]
        if "dims_mm" in d:                       # catalog/dims.yaml: height = spine length
            return float(d["dims_mm"]["height"]), float(d["dims_mm"]["thickness"]), "catalog/dims.yaml"
        return float(d["spine_length_mm"]), float(d["thickness_mm"]), "catalog/dims.yaml"
    est = fam.get("photo_estimate_mm") or {}
    L, T = parse_mm(est.get("spine_length")), parse_mm(est.get("thickness"))
    dtype = fam.get("device_type", "")
    if L and T:
        r = (L / T) / aspect
        if 0.5 <= r <= 1.6:
            return L, T, "photo_estimate_mm"
        # estimate disagrees with the measured pixel aspect: trust the part that is physically sound
        if dtype == "guiding_catheter":
            L = TYPE_LEN["guiding_catheter"]
            return L, round(L / aspect, 1), "type default length 1000 mm; width from px ratio (estimate aspect off by x%.1f)" % r
        return round(T * aspect, 1), T, "thickness from photo_estimate_mm; length from px ratio (estimate length/thickness off by x%.1f)" % r
    if T and not L:
        return round(T * aspect, 1), T, "thickness from photo_estimate_mm; length from px ratio"
    if L and not T:
        return L, round(L / aspect, 1), "length from photo_estimate_mm; thickness from px ratio"
    L = float(TYPE_LEN.get(dtype, DEFAULT_LEN))
    return L, round(L / aspect, 1), "type default length %d mm (%s); thickness from px ratio" % (L, dtype)


# ----------------------------------------------------------------------------------------
# image processing
# ----------------------------------------------------------------------------------------
def luma(img):
    f = img.astype(np.float32)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def rectify(img, quad, w, h, margin=2):
    """Warp quad (A,B,C,D = TL,TR,BR,BL of the output) to a w x h image; a margin-px ring is cut."""
    cw_, ch_ = w + 2 * margin, h + 2 * margin
    dst = np.float32([[0, 0], [cw_, 0], [cw_, ch_], [0, ch_]])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    big = cv2.warpPerspective(img, M, (cw_, ch_), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return big[margin:margin + h, margin:margin + w].copy()


def white_balance(img):
    """Anchor on the label stock (near-white, low chroma, bright); grey-world fallback.
    Returns (img, info string)."""
    f = img.astype(np.float32)
    mx, mn = f.max(2), f.min(2)
    L = luma(img)
    chroma = (mx - mn) / np.maximum(mx, 1)
    thr = np.percentile(L, 75)
    sel = (L >= max(thr, 120)) & (chroma < 0.18)
    frac = float(sel.mean())
    if frac >= 0.03:
        med = np.array([np.median(f[..., c][sel]) for c in range(3)])         # B,G,R
        tint = med.mean() / med
        scale = 238.0 / med.mean()
        scale = float(np.clip(scale, 0.75, 1.6))
        g = np.clip(tint, 0.8, 1.25) * scale
        mode = "label-stock (%.0f%% white px)" % (100 * frac)
    else:
        mean = f.reshape(-1, 3).mean(0)
        tint = mean.mean() / mean
        g = 1.0 + 0.6 * (np.clip(tint, 0.85, 1.18) - 1.0)
        mode = "grey-world (no label stock)"
    out = np.clip(f * g[None, None, :], 0, 255)
    return (out + 0.5).astype(np.uint8), mode + " gains(BGR)=%s" % np.round(g, 2).tolist()


def remove_flat_glare(img):
    """Blown-out, thick, desaturated blobs on a non-white surround (specular reflection on
    ink): inpaint from the surround.  Text strokes are thinner than the erosion kernel."""
    f = img.astype(np.float32)
    L = luma(img)
    mn = f.min(2)
    hot = (mn >= 246).astype(np.uint8)
    bgmed = cv2.medianBlur(L.astype(np.uint8), 31).astype(np.float32)
    hot[bgmed > 205] = 0                            # white label stock is meant to be white
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core = cv2.erode(hot, k)
    if core.sum() == 0:
        return img, 0.0
    mask = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    frac = float(mask.mean())
    out = cv2.inpaint(img, mask * 255, 3, cv2.INPAINT_TELEA)
    return out, frac


def fold_glare(img, n=3, thr=22.0):
    """Replace the outer n rows (top/bottom fold edges) per column by the nearest clean row where
    they are much brighter than the interior."""
    out = img.copy()
    L = luma(img)
    H = img.shape[0]
    ref_t = np.median(L[n + 1:n + 4], axis=0)
    ref_b = np.median(L[H - n - 4:H - n - 1], axis=0)
    cnt = 0
    for i in range(n):
        bad = L[i] - ref_t > thr
        out[i][bad] = out[n][bad]
        cnt += int(bad.sum())
        j = H - 1 - i
        bad = L[j] - ref_b > thr
        out[j][bad] = out[H - 1 - n][bad]
        cnt += int(bad.sum())
    return out, cnt / float(2 * n * img.shape[1])


def unsharp(img, sigma=1.2, amount=0.55, thresh=3):
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    d = f - blur
    d[np.abs(d) < thresh] = 0
    return np.clip(f + amount * d + 0.5, 0, 255).astype(np.uint8)


def process_spine(photo, quad_out, len_mm, th_mm):
    """photo BGR, quad_out photo corners in output order. Returns (final BGR, info dict)."""
    q = np.asarray(quad_out, float)
    Lpx = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    Tpx = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    w0, h0 = max(int(round(Lpx)), 16), max(int(round(Tpx)), 8)
    nat = rectify(photo, q, w0, h0)
    nat, wb = white_balance(nat)
    nat, glare_frac = remove_flat_glare(nat)
    nat = cv2.fastNlMeansDenoisingColored(nat, None, 3, 3, 5, 15)
    W, H = int(round(len_mm * PX_MM)), int(round(th_mm * PX_MM))
    up = cv2.resize(nat, (W, H), interpolation=cv2.INTER_LANCZOS4)
    up, fold_frac = fold_glare(up)
    up = unsharp(up)
    sharp = float(cv2.Laplacian(luma(nat), cv2.CV_32F).var())
    info = dict(native_px=[w0, h0], px_per_mm=round(w0 / len_mm, 2), wb=wb,
                flat_glare_area=round(glare_frac, 4), fold_glare_frac=round(fold_frac, 3),
                sharpness=round(sharp, 1))
    return up, info


def band_colors(img):
    H, W = img.shape[:2]
    y0, y1 = int(H * 0.25), int(H * 0.75)

    def med(x0, x1):
        p = img[y0:y1, x0:x1].reshape(-1, 3)
        b = np.median(p, 0)
        return [int(round(v)) for v in b[::-1]]
    e = max(int(W * 0.08), 6)
    left, right = med(int(W * 0.01), e), med(W - e, int(W * 0.99))
    # dominant colours: 4-bit quantised histogram, top 3 distinct bins (sorted, deterministic)
    q = (img.reshape(-1, 3) >> 4).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    order = np.lexsort((vals, -counts))[:8]
    dom = []
    for o in order:
        b, g, r = vals[o] // 256, (vals[o] // 16) % 16, vals[o] % 16
        c = [int(r * 16 + 8), int(g * 16 + 8), int(b * 16 + 8)]
        if all(sum(abs(a - b2) for a, b2 in zip(c, d)) > 60 for d in dom):
            dom.append(c)
        if len(dom) == 3:
            break
    hexs = lambda c: "#%02X%02X%02X" % tuple(c)
    return dict(left_end_rgb=left, right_end_rgb=right, left_end_hex=hexs(left), right_end_hex=hexs(right),
                dominant_hex=[hexs(c) for c in dom])


# ----------------------------------------------------------------------------------------
# clean procedural redraws (families whose photo spine is fanned / tilted / contaminated)
# ----------------------------------------------------------------------------------------
FONT_DIRS = ["/usr/share/fonts/truetype/liberation2", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/dejavu"]
FONTS = {"sans": "LiberationSans-Regular.ttf", "sansb": "LiberationSans-Bold.ttf",
         "sansbi": "LiberationSans-BoldItalic.ttf", "wide": "DejaVuSans-Bold.ttf",
         "dj": "DejaVuSans.ttf", "djb": "DejaVuSans-Bold.ttf"}
SS = 4                                     # supersampling of the redraw canvas


def font_path(key):
    for d in FONT_DIRS:
        p = os.path.join(d, FONTS[key])
        if os.path.isfile(p):
            return p
    raise RuntimeError("font %s not found under %s" % (FONTS[key], FONT_DIRS))


class Canvas:
    """RGB drawing canvas at SS x supersampling; all coordinates are in final pixels."""

    def __init__(self, w, h, bg):
        from PIL import Image, ImageDraw
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w * SS, h * SS), tuple(bg))
        self.d = ImageDraw.Draw(self.im)
        self._fonts = {}

    def font(self, key, size):
        from PIL import ImageFont
        k = (key, int(size))
        if k not in self._fonts:
            self._fonts[k] = ImageFont.truetype(font_path(key), int(size))
        return self._fonts[k]

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0):
        b = [x0 * SS, y0 * SS, x1 * SS - 1, y1 * SS - 1]
        if r:
            self.d.rounded_rectangle(b, radius=r * SS, fill=fill, outline=outline, width=max(int(width * SS), 1))
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=max(int(width * SS), 1))

    def line(self, x0, y0, x1, y1, fill, width=1.0):
        self.d.line([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=fill, width=max(int(width * SS), 1))

    def ellipse(self, cx, cy, rx, ry, fill=None, outline=None, width=1.0):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=fill,
                       outline=outline, width=max(int(width * SS), 1))

    def polygon(self, pts, fill):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)

    def text(self, s, x0, y0, x1, y1, key="sansb", fill=(0, 0, 0), align="l", spacing=0.0, max_h=None):
        """Draw s in the largest font that fits the box (x0,y0,x1,y1); vertical centre of the box."""
        bw, bh = (x1 - x0) * SS, (y1 - y0) * SS
        if max_h:
            bh = min(bh, max_h * SS)
        lo, hi = 4, int(bh * 1.6)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            f = self.font(key, mid)
            w, h = self._measure(s, f, spacing * mid)
            if w <= bw and h <= bh:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        f = self.font(key, best)
        w, h = self._measure(s, f, spacing * best)
        if align == "l":
            x = x0 * SS
        elif align == "r":
            x = x1 * SS - w
        else:
            x = (x0 + x1) * SS / 2 - w / 2
        cy = (y0 + y1) * SS / 2
        self._draw(s, f, x, cy, fill, spacing * best)
        return best / SS

    def _measure(self, s, f, sp):
        bb = self.d.textbbox((0, 0), s, font=f, anchor="ls")
        if sp:
            w = sum(self.d.textlength(ch, font=f) for ch in s) + sp * (len(s) - 1)
        else:
            w = bb[2] - bb[0]
        return w, bb[3] - bb[1]

    def _draw(self, s, f, x, cy, fill, sp):
        bb = self.d.textbbox((0, 0), s, font=f, anchor="ls")
        y_base = cy - (bb[1] + bb[3]) / 2.0
        if sp:
            for ch in s:
                self.d.text((x, y_base), ch, font=f, fill=fill, anchor="ls")
                x += self.d.textlength(ch, font=f) + sp
        else:
            self.d.text((x - bb[0], y_base), s, font=f, fill=fill, anchor="ls")

    def bars(self, x0, y0, x1, y1, seed, fill=(20, 20, 20), unit=1.0):
        rng = _lcg(seed)
        x = x0
        while x < x1:
            wbar = unit * (1 + int(rng() * 3))
            if int(rng() * 2) == 0:
                self.rect(x, y0, min(x + wbar, x1), y1, fill=fill)
            x += wbar + unit * (0 if int(rng() * 2) else 1)

    def matrix(self, x0, y0, x1, y1, seed, n=12, fill=(20, 20, 20)):
        rng = _lcg(seed)
        cw, ch = (x1 - x0) / n, (y1 - y0) / n
        for i in range(n):
            for j in range(n):
                on = (i == 0 or j == n - 1) or ((i == n - 1 or j == 0) and (i + j) % 2 == 0) or rng() > 0.5
                if on:
                    self.rect(x0 + i * cw, y0 + j * ch, x0 + (i + 1) * cw, y0 + (j + 1) * ch, fill=fill)

    def result(self):
        arr = np.asarray(self.im, dtype=np.float32).reshape(self.h, SS, self.w, SS, 3).mean((1, 3))
        return cv2.cvtColor(np.clip(arr + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _lcg(seed):
    state = [seed & 0x7FFFFFFF]

    def nxt():
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return (state[0] >> 8) / float(1 << 23)
    return nxt


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def redraw_essential_pro(W, H):
    pink, lav, ink = hexrgb("#D81B73"), hexrgb("#B8B0E0"), (28, 26, 44)
    c = Canvas(W, H, pink)
    bw = 0.11 * W
    c.rect(bw, 0, W - bw, H, fill=lav)
    # CARDIO emblem: ring + heart
    cx, cy, r = bw / 2, H * 0.43, H * 0.30
    c.ellipse(cx, cy, r, r, outline=(255, 255, 255), width=1.6)
    c.ellipse(cx - r * 0.28, cy - r * 0.12, r * 0.30, r * 0.30, fill=(255, 255, 255))
    c.ellipse(cx + r * 0.28, cy - r * 0.12, r * 0.30, r * 0.30, fill=(255, 255, 255))
    c.polygon([(cx - r * 0.56, cy), (cx + r * 0.56, cy), (cx, cy + r * 0.62)], fill=(255, 255, 255))
    c.text("CARDIO", bw * 0.15, H * 0.80, bw * 0.85, H * 0.96, "sansb", (255, 255, 255), "c")
    x = bw
    c.text("Rx", x + 0.008 * W, H * 0.10, x + 0.03 * W, H * 0.30, "sansb", ink)
    c.text("essential pro", x + 0.035 * W, H * 0.06, x + 0.27 * W, H * 0.60, "dj", ink)
    c.rect(x + 0.037 * W, H * 0.68, x + 0.058 * W, H * 0.90, outline=ink, width=0.8)
    c.text("REF", x + 0.039 * W, H * 0.70, x + 0.056 * W, H * 0.88, "sansb", ink, "c")
    c.text("BCDPR14N150400030", x + 0.064 * W, H * 0.66, x + 0.27 * W, H * 0.92, "dj", ink)
    c.rect(x + 0.30 * W, H * 0.12, x + 0.325 * W, H * 0.88, outline=ink, width=0.8)
    c.text("BALLOON", x + 0.302 * W, H * 0.16, x + 0.323 * W, H * 0.84, "sansb", ink, "c", max_h=H * 0.2)
    c.text("4", x + 0.345 * W, H * 0.08, x + 0.375 * W, H * 0.66, "sansb", ink, "c")
    c.text("mm", x + 0.340 * W, H * 0.70, x + 0.380 * W, H * 0.90, "sans", ink, "c")
    c.text("30", x + 0.425 * W, H * 0.08, x + 0.485 * W, H * 0.66, "sansb", ink, "c")
    c.text("mm", x + 0.430 * W, H * 0.70, x + 0.480 * W, H * 0.90, "sans", ink, "c")
    c.text("LOT 2510245", x + 0.52 * W, H * 0.08, x + 0.64 * W, H * 0.34, "sans", ink)
    c.text("2028-07-21", x + 0.52 * W, H * 0.38, x + 0.64 * W, H * 0.64, "sans", ink)
    c.text("2025-07-22", x + 0.52 * W, H * 0.68, x + 0.64 * W, H * 0.94, "sans", ink)
    c.bars(x + 0.67 * W, H * 0.12, x + 0.77 * W, H * 0.88, 11, ink, unit=1.0)
    c.text("iVascular", W - bw + 0.012 * W, H * 0.30, W - 0.012 * W, H * 0.70, "sansbi", (255, 255, 255), "c")
    return c.result()


def redraw_selution(W, H):
    paper, ink, grey = hexrgb("#E8E8E6"), (18, 18, 18), (95, 95, 95)
    c = Canvas(W, H, paper)
    c.rect(0, 0, 0.014 * W, H, fill=hexrgb("#BCBAB7"))
    c.rect(W - 0.03 * W, 0, W, H, fill=hexrgb("#E5E6E5"))
    # sunburst + MedAlliance wordmark
    sx, sy = 0.165 * W, H * 0.40
    for k in range(18):
        a = math.radians(-100 + k * 11)
        c.line(sx + math.cos(a) * H * 0.16, sy + math.sin(a) * H * 0.16,
               sx + math.cos(a) * H * 0.46, sy + math.sin(a) * H * 0.46, grey, 0.9)
    c.text("MedAlliance", 0.03 * W, H * 0.46, 0.215 * W, H * 0.80, "sansb", grey, "c")
    c.text("SELUTION SLR™ 014 PTCA", 0.235 * W, H * 0.06, 0.58 * W, H * 0.48, "sansb", ink)
    c.text("BALLOON CATHETER", 0.235 * W, H * 0.52, 0.52 * W, H * 0.92, "sansb", ink)
    c.rect(0.60 * W, H * 0.06, 0.715 * W, H * 0.94, fill=(10, 10, 10))
    c.text("Ø", 0.605 * W, H * 0.10, 0.625 * W, H * 0.46, "dj", (255, 255, 255))
    c.text("3.50", 0.625 * W, H * 0.10, 0.705 * W, H * 0.74, "sansb", (255, 255, 255), "c")
    c.text("mm", 0.66 * W, H * 0.74, 0.70 * W, H * 0.93, "sans", (255, 255, 255), "c", max_h=H * 0.18)
    c.rect(0.725 * W, H * 0.06, 0.80 * W, H * 0.94, outline=ink, width=1.0)
    c.text("↔", 0.730 * W, H * 0.10, 0.752 * W, H * 0.40, "dj", ink)
    c.text("25", 0.735 * W, H * 0.30, 0.795 * W, H * 0.80, "sansb", ink, "c")
    c.text("mm", 0.76 * W, H * 0.78, 0.795 * W, H * 0.96, "sans", ink, "c", max_h=H * 0.16)
    for i, s in enumerate(["REF SC14-350025", "LOT 18472396", "2027-06-05"]):
        y = H * (0.08 + i * 0.29)
        c.rect(0.815 * W, y + 1, 0.826 * W, y + H * 0.22, outline=ink, width=0.6)
        c.text(s, 0.833 * W, y, 0.965 * W, y + H * 0.26, "sans", (40, 40, 40))
    return c.result()


def redraw_nexgen(W, H):
    white, maroon, blue = hexrgb("#ECEDEC"), hexrgb("#9B1B30"), hexrgb("#1F5FBF")
    ink = (14, 14, 14)
    c = Canvas(W, H, maroon)
    c.rect(0, 0, 0.84 * W, H, fill=white, r=H * 0.12)
    c.rect(0.70 * W, 0, 0.84 * W, H, fill=white)
    c.text("NEXGEN", 0.03 * W, H * 0.30, 0.37 * W, H * 0.78, "wide", ink, "l", spacing=0.06)
    c.text("TM", 0.375 * W, H * 0.16, 0.40 * W, H * 0.34, "sans", ink, max_h=H * 0.14)
    c.rect(0.405 * W, H * 0.08, 0.60 * W, H * 0.92, fill=white, outline=ink, width=1.0, r=H * 0.05)
    c.text("4.00 X 24", 0.412 * W, H * 0.14, 0.595 * W, H * 0.86, "sans", ink, "c")
    c.text("LOT NF22", 0.615 * W, H * 0.10, 0.735 * W, H * 0.34, "sans", (60, 60, 60))
    c.text("REF NAG40024", 0.615 * W, H * 0.42, 0.735 * W, H * 0.66, "sans", (60, 60, 60))
    c.text("2025-07-17", 0.755 * W, H * 0.10, 0.82 * W, H * 0.34, "sans", (60, 60, 60))
    c.text("2028-07-16", 0.755 * W, H * 0.40, 0.82 * W, H * 0.64, "sans", (60, 60, 60))
    c.matrix(0.755 * W, H * 0.68, 0.755 * W + H * 0.26, H * 0.68 + H * 0.26, 7, 10, ink)
    c.rect(0.885 * W, H * 0.18, 0.965 * W, H * 0.82, fill=blue)
    c.text("Meril", 0.89 * W, H * 0.26, 0.96 * W, H * 0.74, "sans", (255, 255, 255), "c")
    return c.result()


def redraw_combo(W, H):
    navy, white, ink = hexrgb("#1B3F9A"), hexrgb("#F2F2F0"), (24, 24, 28)
    c = Canvas(W, H, navy)
    bw = 0.095 * W
    c.rect(bw, 0, W - bw, H, fill=white)
    c.text("COMBO", bw + 0.012 * W, H * 0.28, bw + 0.17 * W, H * 0.70, "sansb", (128, 128, 130))
    c.text("+", bw + 0.175 * W, H * 0.22, bw + 0.205 * W, H * 0.74, "sansb", (128, 128, 130), "c")
    c.text("3.0", bw + 0.235 * W, H * 0.22, bw + 0.30 * W, H * 0.78, "sansb", ink, "c")
    c.rect(bw + 0.315 * W, H * 0.08, bw + 0.375 * W, H * 0.92, fill=(8, 8, 8))
    c.text("33", bw + 0.320 * W, H * 0.16, bw + 0.370 * W, H * 0.84, "sansb", (255, 255, 255), "c")
    c.bars(bw + 0.40 * W, H * 0.14, bw + 0.64 * W, H * 0.74, 5, (70, 70, 74), unit=1.2)
    c.text("COMBO Plus PTCA 3.0 x 33", bw + 0.40 * W, H * 0.78, bw + 0.64 * W, H * 0.94, "sans", (70, 70, 74))
    c.text("LOT 2511", bw + 0.675 * W, H * 0.12, bw + 0.79 * W, H * 0.46, "sans", ink)
    c.text("2027-11", bw + 0.675 * W, H * 0.54, bw + 0.79 * W, H * 0.88, "sans", ink)
    c.text("OrbusNeich", W - bw + 0.004 * W, H * 0.34, W - 0.004 * W, H * 0.66, "sansb", (235, 238, 250), "c")
    return c.result()


def redraw_sapphire(W, H):
    pale, black, orange = hexrgb("#E4E7E8"), hexrgb("#1B1B1F"), hexrgb("#F26B3A")
    white, ink = (246, 247, 247), (18, 18, 20)
    c = Canvas(W, H, pale)
    c.rect(0.045 * W, 0, 0.255 * W, H, fill=black)
    c.text("SAPPHIRE", 0.055 * W, H * 0.28, 0.185 * W, H * 0.62, "sansb", (255, 255, 255), "l")
    c.text("3", 0.195 * W, H * 0.04, 0.245 * W, H * 0.96, "sansb", orange, "c")
    c.rect(0.255 * W, 0, 0.84 * W, H, fill=white)
    c.text("3.0", 0.265 * W, H * 0.14, 0.335 * W, H * 0.86, "sansb", ink, "c")
    c.rect(0.345 * W, H * 0.08, 0.395 * W, H * 0.92, fill=(8, 8, 8))
    c.text("15", 0.349 * W, H * 0.16, 0.391 * W, H * 0.84, "sansb", (255, 255, 255), "c")
    c.bars(0.425 * W, H * 0.10, 0.71 * W, H * 0.58, 3, ink, unit=1.3)
    c.text("Sapphire 3 PTCA Balloon 3.0 mm / 15 mm", 0.425 * W, H * 0.64, 0.71 * W, H * 0.92, "sans", (50, 50, 52))
    c.polygon([(0.745 * W, H * 0.16), (0.765 * W, H * 0.16), (0.755 * W, H * 0.5)], fill=ink)
    c.polygon([(0.745 * W, H * 0.84), (0.765 * W, H * 0.84), (0.755 * W, H * 0.5)], fill=ink)
    c.text("2027-11-07", 0.775 * W, H * 0.12, 0.835 * W, H * 0.46, "sans", ink)
    c.text("LOT FA2510297", 0.775 * W, H * 0.54, 0.835 * W, H * 0.88, "sans", ink)
    c.text("ORBUSNEICH", 0.85 * W, H * 0.34, 0.99 * W, H * 0.66, "sansb", (90, 96, 100), "c")
    return c.result()


def redraw_euphora(W, H):
    black, mint, teal = hexrgb("#1A1A1A"), hexrgb("#A8E4C8"), hexrgb("#2AA89A")
    c = Canvas(W, H, hexrgb("#40B0D2"))
    c.rect(0.035 * W, 0, 0.255 * W, H, fill=black)
    c.ellipse(0.075 * W, H * 0.5, H * 0.34, H * 0.34, fill=hexrgb("#2DBE60"))
    c.ellipse(0.075 * W, H * 0.5, H * 0.20, H * 0.20, outline=black, width=1.4)
    c.text("NC Euphora™", 0.105 * W, H * 0.30, 0.248 * W, H * 0.70, "sansbi", (245, 245, 245))
    c.rect(0.255 * W, 0, 0.285 * W, H, fill=hexrgb("#4CC3A8"))
    c.rect(0.285 * W, 0, 0.325 * W, H, fill=hexrgb("#F0A030"))
    c.rect(0.325 * W, 0, 0.365 * W, H, fill=(240, 240, 240))
    c.rect(0.332 * W, H * 0.2, 0.358 * W, H * 0.8, outline=(40, 40, 40), width=0.8)
    c.rect(0.365 * W, 0, 0.93 * W, H, fill=mint)
    c.text("NC Euphora™ Edge", 0.38 * W, H * 0.06, 0.62 * W, H * 0.46, "sansb", (30, 52, 46))
    c.text("NONCOMPLIANT BALLOON DILATATION CATHETER", 0.38 * W, H * 0.52, 0.72 * W, H * 0.80, "sans", (30, 52, 46))
    c.rect(0.76 * W, H * 0.10, 0.85 * W, H * 0.90, fill=(255, 255, 255), outline=(30, 52, 46), width=1.0)
    c.text("3.5 x 20", 0.765 * W, H * 0.18, 0.845 * W, H * 0.82, "sansb", (30, 52, 46), "c")
    c.bars(0.86 * W, H * 0.16, 0.92 * W, H * 0.84, 9, (30, 52, 46), unit=1.1)
    c.rect(0.93 * W, 0, W, H, fill=teal)
    c.text("Medtronic", 0.935 * W, H * 0.34, 0.995 * W, H * 0.66, "sansb", (255, 255, 255), "c")
    return c.result()


REDRAW = {
    "ivascular_essential_pro_balloon": (redraw_essential_pro, "4 x 30 mm (REF BCDPR14N150400030)", "iVascular",
                                        ["#D81B73", "#B8B0E0"]),
    "medalliance_selution_slr_dcb": (redraw_selution, "3.50 x 25 mm", "MedAlliance", ["#BCBAB7", "#E8E8E6"]),
    "meril_nexgen_balloon": (redraw_nexgen, "4.00 x 24", "Meril", ["#9B1B30", "#ECEDEC", "#1F5FBF"]),
    "orbusneich_combo_plus": (redraw_combo, "3.0 x 33", "OrbusNeich (inferred)", ["#1B3F9A", "#F2F2F0"]),
    "orbusneich_sapphire_3": (redraw_sapphire, "3.0 x 15", "OrbusNeich (inferred)", ["#E4E7E8", "#1B1B1F", "#F26B3A"]),
    "medtronic_nc_euphora_edge": (redraw_euphora, "3.5 x 20 (from research front image; none in inventory)", "Medtronic",
                                  ["#40B0D2", "#1A1A1A", "#A8E4C8", "#2AA89A"]),
}


# ----------------------------------------------------------------------------------------
# grading, io, main
# ----------------------------------------------------------------------------------------
GRADES = "ABC"


def grade(info, confs, ov, vis):
    reasons = []
    g = 0
    ppm = info["px_per_mm"]
    if ppm < 2.0:
        g = max(g, 2); reasons.append("low resolution %.1f px/mm" % ppm)
    elif ppm < 3.0:
        g = max(g, 1); reasons.append("%.1f px/mm" % ppm)
    else:
        reasons.append("%.1f px/mm" % ppm)
    if info["sharpness"] < 350:
        g = max(g, 1); reasons.append("blurry source photo (lap var %.0f)" % info["sharpness"])
    elif info["sharpness"] < 1000:
        g = max(g, 1); reasons.append("soft source photo (lap var %.0f)" % info["sharpness"])
    if min(confs) < 1.3:
        g = max(g, 1); reasons.append("weak edge fit (%.1f)" % min(confs))
    if vis < 0.85:
        g = max(g, 2); reasons.append("only %.0f%% visible" % (vis * 100))
    elif vis < 1.0:
        g = max(g, 1); reasons.append("%.0f%% visible" % (vis * 100))
    cap = ov.get("cap")
    if cap and GRADES.index(cap) > g:
        g = GRADES.index(cap)
        if ov.get("note"):
            reasons.append(ov["note"].split(";")[0])
    return GRADES[g], "; ".join(reasons)


def save_png(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("png encode failed for " + path)
    tmp = "%s.tmp-%d" % (path, os.getpid())
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_text(path, text):
    tmp = "%s.tmp-%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def rnd_pts(q):
    return [[round(float(x), 1), round(float(y), 1)] for x, y in q]


def face_entry(fid, fam, ov, photo, path, dims_override, a):
    """Families whose photo crop is the box's big FRONT face (not a spine): write the rectified
    face crop (front_photo_crop.png, image right = +X = photo top, up = +Y) and a plain-colour top."""
    d = dims_override[fid]["dims_mm"]
    H_mm, L_mm, T_mm = float(d["height"]), float(d["length"]), float(d["thickness"])
    fq = ov["face"]
    q = order_quad(fq["quad"], "cw")                       # photo top -> image right
    w0 = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h0 = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    nat = rectify(photo, q, max(int(round(w0)), 16), max(int(round(h0)), 8))
    nat, wb = white_balance(nat)
    nat = cv2.fastNlMeansDenoisingColored(nat, None, 3, 3, 5, 15)
    ppm_src = nat.shape[0] / L_mm                           # photo px per mm across the face width
    covers = nat.shape[1] / ppm_src
    stretch = float(fq.get("stretch", 1.0))                  # applied along the box height only
    covers *= stretch
    clipped_mm = 0.0
    if covers > H_mm:                                        # photo shows more than the box height:
        keep = H_mm / covers                                 # drop the photo-bottom end (image left)
        x0 = int(round(nat.shape[1] * (1 - keep)))
        nat = nat[:, x0:]
        clipped_mm = covers - H_mm
        covers = H_mm
    ppm = min(float(PX_MM), 2048.0 / H_mm)
    crop = cv2.resize(nat, (int(round(covers * ppm)), int(round(L_mm * ppm))), interpolation=cv2.INTER_AREA)
    base = hexrgb(fq["base"])
    top = np.empty((int(round(T_mm)), int(round(H_mm)), 3), np.uint8)             # plain colour: 1 px/mm
    top[:] = base[::-1]
    if not a.no_write:
        save_png(os.path.join(a.out_root, "textures", fid, "front_photo_crop.png"), crop)
        save_png(os.path.join(a.out_root, "textures", fid, "top.png"), top)
    return dict(
        brand=fam.get("brand"), product=fam.get("product"), device_type=fam.get("device_type"),
        source="plain", photo=os.path.basename(path), rotation="cw",
        note="the photo quad is the big FRONT face, not the spine; top.png is the plain base colour",
        plain_rgb=list(base), plain_hex=fq["base"],
        mm_used=dict(spine_length=H_mm, thickness=T_mm, full_spine_length=H_mm, source="catalog/dims.yaml"),
        texture=dict(file="textures/%s/top.png" % fid, size_px=[int(top.shape[1]), int(top.shape[0])],
                     px_per_mm=1),
        front_face_crop=dict(
            file="textures/%s/front_photo_crop.png" % fid, size_px=[int(crop.shape[1]), int(crop.shape[0])],
            px_per_mm=round(ppm, 3), photo_quad_tl_tr_br_bl=rnd_pts(fq["quad"]),
            orientation="image right = +X = photo top; image up = +Y = photo left",
            covers_height_mm=round(covers, 1), full_height_mm=H_mm, face_length_mm=L_mm,
            placed_from="photo-top end (+X, image right edge)",
            stretch_along_height=round(stretch, 3), clipped_photo_bottom_mm=round(clipped_mm, 1),
            remainder_fill=fq.get("extend", "base"), base_rgb=list(base), wb=wb),
        occlusion_notes=ov.get("note"),
        grade="B", grade_reason="front face crop at %.2f px/mm (source), covers %.0f%% of the height" % (
            ppm_src, 100 * covers / H_mm),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--sheets", default="", help="directory for contact sheet + corner overlay sheet")
    ap.add_argument("--no-write", action="store_true", help="do not write textures/spines.yaml")
    ap.add_argument("--out-root", default=CAT)
    a = ap.parse_args()
    draft = yaml.safe_load(open(os.path.join(CAT, "families_draft.yaml")))
    fams = draft["families"]
    dims_path = os.path.join(CAT, "dims.yaml")
    dims_override = {}
    if os.path.isfile(dims_path):
        dims_override = yaml.safe_load(open(dims_path)) or {}
        dims_override = dims_override.get("families", dims_override)
    only = [x for x in a.only.split(",") if x]
    result = {}
    skipped = {}
    tiles, ctiles = [], []
    cache = {}
    for fid in sorted(fams):
        fam = fams[fid]
        if only and fid not in only:
            continue
        ex = fam.get("exemplar") or {}
        if fam.get("existing_asset"):
            skipped[fid] = "existing_asset: %s" % fam["existing_asset"]
            continue
        if not ex.get("corners_px"):
            skipped[fid] = "no corners_px"
            continue
        ov = OVERRIDES.get(fid, {})
        path = photo_path(ex["photo"])
        photo = cache.setdefault(path, cv2.imread(path, cv2.IMREAD_COLOR))
        rot = ov.get("rot", "ccw")
        if "face" in ov:
            result[fid] = face_entry(fid, fam, ov, photo, path, dims_override, a)
            continue
        rough = order_quad(ex["corners_px"], rot)
        if "quad" in ov:
            vq = np.array(ov["quad"], float)                    # ccw-view order
            start = vq if rot in ("ccw", "none") else vq[[2, 3, 0, 1]]
        else:
            start = rough
        if ov.get("fit", True):
            quad, confs = refine_quad(photo, start)
        else:
            quad, confs = np.asarray(start, float), [9.0] * 4
        Lpx = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        Tpx = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        aspect = Lpx / Tpx
        L_full, T, src = resolve_dims(fid, fam, aspect, dims_override)
        vis = float(ov.get("visible", 1.0))
        L_tex = round(L_full * vis, 1)
        img, info = process_spine(photo, quad, L_tex, T)
        if vis < 1.0 and not ov.get("redraw"):
            # the photo hides part of the spine (photo edge, shelf lip): continue the end band on
            # the hidden end so the texture covers the full spine length at the right aspect
            full_w = int(round(L_full * PX_MM))
            if full_w > img.shape[1]:
                pte0 = photo_top_end(rot, quad)
                cut = ov.get("cut", "bottom")
                hidden = pte0 if cut == "top" else ("right" if pte0 == "left" else "left")
                n = full_w - img.shape[1]
                edge = img[:, -6:] if hidden == "right" else img[:, :6]
                band = np.repeat(np.median(edge, axis=1, keepdims=True).astype(img.dtype), n, axis=1)
                img = np.hstack([img, band] if hidden == "right" else [band, img])
                L_tex = L_full
        redrawn = None
        if ov.get("redraw"):
            fn, size_txt, maker, palette = REDRAW[fid]
            redrawn = fn(int(round(L_full * PX_MM)), int(round(T * PX_MM)))
            if not a.no_write:
                save_png(os.path.join(a.out_root, "textures", fid, "top_photo.png"), img)
                save_png(os.path.join(a.out_root, "textures", fid, "top.png"), redrawn)
        elif not a.no_write:
            save_png(os.path.join(a.out_root, "textures", fid, "top.png"), img)
        col = band_colors(redrawn if redrawn is not None else img)
        gr, why = grade(info, confs, ov, vis)
        if redrawn is not None:
            col["dominant_hex"] = list(palette)
            gr, why = "A", "clean procedural redraw (photo spine unusable: %s)" % (ov.get("note") or "").split(";")[0]
        pte = photo_top_end(rot, quad)
        entry = dict(
            brand=fam.get("brand"), product=fam.get("product"), device_type=fam.get("device_type"),
            source="redrawn" if redrawn is not None else "photo",
            photo=os.path.basename(path), rotation=rot,
            corners_px=rnd_pts(quad),
            corners_order="output top-left, top-right, bottom-right, bottom-left, in photo pixels",
            corners_source=("manual quad, no snap" if not ov.get("fit", True) else
                            "manual quad + edge snap" if "quad" in ov else "draft corners + edge snap"),
            edge_fit_confidence=[round(float(c), 2) for c in confs],
            mm_used=dict(spine_length=(L_full if redrawn is not None else L_tex), thickness=T,
                         full_spine_length=round(L_full, 1),
                         source=src, px_aspect=round(float(aspect), 2)),
            photo_top_end=pte,
            visible_fraction=(1.0 if redrawn is not None else vis),
            occlusion_notes=(ov.get("note") or "none") + (
                ("; cut off at the %s of the photo" % ov["cut"]) if ov.get("cut") else ""),
            texture=dict(file="textures/%s/top.png" % fid,
                         size_px=[int((redrawn if redrawn is not None else img).shape[1]),
                                  int((redrawn if redrawn is not None else img).shape[0])],
                         px_per_mm=PX_MM),
            processing=info,
            band_colors=col,
            grade=gr, grade_reason=why,
        )
        if redrawn is not None:
            entry["photo_reference"] = dict(file="textures/%s/top_photo.png" % fid,
                                            size_px=[int(img.shape[1]), int(img.shape[0])],
                                            mm_covered=[L_tex, T], note="warped photo spine kept for reference only")
            entry["redraw"] = dict(size_text=size_txt, maker_word=maker, palette=list(palette),
                                   fonts="Liberation Sans / DejaVu Sans (system TTF)",
                                   layout="end band, label panel, product name, size boxes, small print; as seen in the photo")
        if a.sheets:
            tiles.append((fid, gr, redrawn if redrawn is not None else img))
            ctiles.append((fid, photo, rough, quad, rot))
        result[fid] = entry
    if not a.no_write and not only:
        doc = dict(generator="tools/catalog_spines.py", dims_file_used=bool(dims_override),
                   skipped=skipped, families=result)
        save_text(os.path.join(a.out_root, "spines.yaml"),
                  yaml.safe_dump(to_py(doc), sort_keys=False, default_flow_style=None, width=120))
    if a.sheets:
        os.makedirs(a.sheets, exist_ok=True)
        make_sheets(a.sheets, tiles, ctiles)
    grades = {}
    for e in result.values():
        grades[e["grade"]] = grades.get(e["grade"], 0) + 1
    print("processed %d, skipped %d %s, grades %s" % (len(result), len(skipped), sorted(skipped), grades))
    for fid, e in sorted(result.items()):
        if e["grade"] == "C":
            print("C", fid, e["grade_reason"])


def make_sheets(outdir, tiles, ctiles):
    TW = 900
    rows = []
    for fid, gr, img in tiles:
        s = TW / img.shape[1]
        t = cv2.resize(img, (TW, max(int(img.shape[0] * s), 1)), interpolation=cv2.INTER_AREA)
        pad = np.full((18, TW, 3), 40, np.uint8)
        cv2.putText(pad, "%s [%s]" % (fid, gr), (3, 13), 0, 0.45, (255, 255, 255), 1)
        rows.append(np.vstack([pad, t]))
    def stack(rs, cols):
        out = []
        for i in range(0, len(rs), cols):
            grp = rs[i:i + cols]
            h = max(r.shape[0] for r in grp)
            grp = [np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0))) for r in grp]
            out.append(np.hstack(grp))
        w = max(r.shape[1] for r in out)
        return np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in out])
    per = 14
    for k in range(0, len(rows), per):
        cv2.imwrite(os.path.join(outdir, "contact_%d.png" % (k // per)), stack(rows[k:k + per], 2))
    crow = []
    for fid, photo, rough, quad, rot in ctiles:
        allp = np.vstack([rough, quad])
        pad = 40
        x0, y0 = np.maximum(allp.min(0) - pad, 0).astype(int)
        x1, y1 = np.minimum(allp.max(0) + pad, [photo.shape[1], photo.shape[0]]).astype(int)
        c = photo[y0:y1, x0:x1].copy()
        s = min(3.0, 1100.0 / max(c.shape[:2]))
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for pts, col in ((rough, (0, 0, 255)), (quad, (0, 255, 0))):
            cv2.polylines(c, [((pts - [x0, y0]) * s).astype(np.int32)], True, col, 1)
        # rotate so that the output text is upright
        k = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE}.get(rot)
        if rot == "ccw" or rot == "cw":
            if c.shape[0] > c.shape[1]:
                c = cv2.rotate(c, k)
        cv2.putText(c, fid, (3, 14), 0, 0.5, (255, 255, 0), 1)
        crow.append(c)
    per = 5
    for k in range(0, len(crow), per):
        grp = crow[k:k + per]
        w = max(r.shape[1] for r in grp)
        out = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in grp])
        cv2.imwrite(os.path.join(outdir, "corners_%d.png" % (k // per)), out)


if __name__ == "__main__":
    main()
