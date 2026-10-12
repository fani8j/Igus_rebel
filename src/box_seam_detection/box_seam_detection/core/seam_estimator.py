# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/seam_estimator.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Flap-gap seam estimation from 4 top-face corners + observed gap evidence.

The seam is the physical GAP where the two top flaps meet, NOT the packing
tape centreline (tape is often off-centre). Pipeline: projective-midline
priors for both axes ("lr" joins the left/right edges, "bf" joins the
back/front edges) -> pick the axis from tape evidence measured in the
rectified face (low-saturation tape band + tape-edge energy; optional SAM3
"packing tape" mask bonus) with per-axis gap support recorded alongside ->
refine toward observed flap-gap evidence: a thin dark valley in rectified
gray with bright cardboard shoulders on BOTH sides (the gap is visible
through or beside translucent tape, often strongest mid-face) -> robust
line fit on gap detections only (no blending with the prior) -> intersect
with the two face edges in the ORIGINAL image.

``ok`` means the gap was observed: thin-valley detections with bright
shoulders supporting >= ``gap_min_coverage`` of the face with no gap >
``gap_max_gap_frac``, residual <= ``gap_max_residual_px``, plausible gap
width in mm via the top-plane scale (REQUIRES ``corners_xyz``), line inside
the search window. ``estimated`` means the axis is clear but the gap is not
observed: the seam is the prior line (endpoints in the estimate block only,
never a cutting target). ``ambiguous`` means the axis itself is unclear.

Corner order: [back_left, back_right, front_right, front_left].
``seam_left`` / ``seam_right`` are image-relative (smaller/larger x).

Dev-set note: thresholds below were tuned on the 42 reviewed ok gap seams
(22 pilot + 20 batch-2) in reviewed/autolabel_reviews_v5.json; numbers
reported by the eval driver are dev-set results, not generalisation claims.
Gap width measured on the dev set via the top-plane scale: median rect
detection width 2..4 px at 0.4..0.9 mm/px, i.e. ~1..3 mm; gate [0.5, 4.0]
mm keeps every dev pass with margin while rejecting broad shadow bands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import box_profiles as _profiles

CORNER_NAMES = ("back_left", "back_right", "front_right", "front_left")

# Axis evidence gates (dev-set: true-axis score 14.8..47.9, wrong-axis
# 2.4..8.6, synthetic uniform/texture noise < 1.0).
AXIS_MIN_EVIDENCE = 8.0
AXIS_MIN_MARGIN_RATIO = 0.25
# Observed-gap acceptance gates. Gap width measured on the 42 dev gap seams
# via the top-plane scale: rect detection width 2..4 px at 0.4..0.9 mm/px
# (dev set, not generalisation). Gate [0.5, 4.0] mm keeps all dev passes
# with margin while rejecting broad shadow/glare bands whose valley widths
# fall outside a real flap gap. Coverage gates chosen from dev
# observed-fit stats (35/42 pass at min_coverage 0.40 with the tuned
# detector; the rest degrade to estimated = prior line).
GAP_MIN_DEPTH = 12.0
GAP_MIN_SHOULDER = 8.0
GAP_MAX_WIDTH_PX = 6
MIN_SUPPORT_FRAC = 0.5
MIN_COVERAGE = 0.40
MAX_GAP_FRAC = 0.35
MAX_RESIDUAL_PX = 2.5
GAP_WIDTH_MM_MIN = 0.5
GAP_WIDTH_MM_MAX = 4.0
RESID_INLIER_PX = 2.0
# Legacy tape detector (refine_on_tape) keeps its own tuned gates: sat-dip
# score >= 3.0 with paired edges >= 1.5, coverage >= 0.45, gap <= 0.30,
# width in [4, 15] mm. It now serves axis audit + the eval driver's
# old-tape comparison only.
TAPE_MIN_SLICE_SCORE = 3.0
TAPE_MIN_EDGE_NEG = 1.5
TAPE_MIN_EDGE_POS = 1.5
TAPE_MIN_COVERAGE = 0.45
TAPE_MAX_GAP_FRAC = 0.30
TAPE_MAX_RESIDUAL_PX = 2.5
TAPE_WIDTH_MM_MIN = 4.0
TAPE_WIDTH_MM_MAX = 15.0
MIN_SLICE_SCORE = TAPE_MIN_SLICE_SCORE
MIN_EDGE_NEG = TAPE_MIN_EDGE_NEG
MIN_EDGE_POS = TAPE_MIN_EDGE_POS
# Confirm-mode agreement gates (default "confirm" in estimate_seam):
# published line is always the prior; ok needs gap agreement with the
# prior SHIFTED by the expected viewing shift (VIEW_SHIFT_MM below) within
# 1.5 mm at 25/50/75% + angle <= 2 deg with coverage >= 0.40.
CONFIRM_MAX_OFFSET_MM = 1.5
CONFIRM_MAX_ANGLE_DEG = 2.0
CONFIRM_MIN_COVERAGE = 0.40
# Darkline detector (seam_mode "darkline"): stations every ~2-3 px along
# the middle ~80% of the prior line; perpendicular gray profile ±8 px at
# 720p scaled with face size; darkest point kept when shoulder contrast
# (mean of shoulders − valley) clears DARKLINE_MIN_CONTRAST. Robust
# Huber fit on kept points; ok needs coverage + residual gates below.
DARKLINE_MIN_CONTRAST = 6.0
# Per-side shoulder gate (each shoulder mean − valley, gray levels): the
# valley must be darker than BOTH shoulders, not just their mean, so the
# dark side of a one-sided tape-edge highlight step is not a station.
DARKLINE_MIN_SHOULDER = 2.0
# Viewing shift (apparent seam position vs the physical, centred seam).
# The flap gap / dark line is SEEN displaced AWAY from the camera (along
# the top plane, perpendicular to the seam) relative to the projective
# midline, although the user measured the small box flaps centred (gap at
# 56 of 112 mm). Fitted on UNLABELLED all853_v8 corners (no human labels):
# per capture, signed mid-face offset of the gated gap-valley fit from the
# prior, + = away from the camera, with the search window centred on the
# candidate shift until self-consistent (the prior-centred window
# truncates the far shoulder and biases the fit toward the prior).
# Class medians (bootstrap 95%): small 3.17 mm [3.00, 3.25], n=142,
# MAD 0.34; big 0.52 mm [0.41, 0.64], n=572, MAD 0.44. The gap-valley
# detector is used because its position barely depends on the window
# (small 2.87 -> 3.17 mm for window centres 0 -> 4.9 mm); the darkline
# (darkest point) sits further away and moves with the window (small
# 2.99 -> 3.98 mm; fixed point ~3.9 mm, big ~0.7 mm). Within a class the
# offset does not scale with tan(view angle) (corr +0.18..+0.29; a k*tan
# fit has larger residual MAD, 0.49 vs 0.34 on small), so the default
# model is a per-class constant; ``k_tan_mm`` allows s = c + k*tan(theta)
# with theta = tilt of the camera ray at the face centre from the plane
# normal, in the across-seam plane. Physical reading: s ~ h*tan(theta)
# for an apparent lip height h above the corner plane (small: h ~ 5 mm at
# theta ~ 32 deg); the point-cloud profile across the small-box seam shows
# the camera-side flap raised (~0.8 mm, sensor-blurred) over the far
# flap and the mean gray profile shows a bright near-lip highlight
# (+2.2 mm) followed by the dark gap (+3.5..4 mm), so the gap is seen
# beyond the raised near lip. The PUBLISHED line stays the unshifted
# prior; only the confirmation compares observed lines with prior + s.
VIEW_SHIFT_MM = _profiles.view_shift_mm_table()
VIEW_SHIFT_K_TAN_MM = {"big": 0.0, "small": 0.0}
BIG_LONG_SIDE_M = _profiles.LEGACY_BIG_LONG_SIDE_M
DARKLINE_MIN_COVERAGE = 0.40
DARKLINE_MAX_GAP_FRAC = 0.35
DARKLINE_MAX_RESIDUAL_PX = 2.5

# ---- geometry -----------------------------------------------------------

def _as_quad(corners) -> np.ndarray:
    q = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(q)):
        raise ValueError("corners must be finite")
    return q


def _line_int(p1, p2, q1, q2):
    """Intersection of lines p1-p2 and q1-q2; None if parallel."""
    l1 = np.cross(np.append(p1, 1.0), np.append(p2, 1.0))
    l2 = np.cross(np.append(q1, 1.0), np.append(q2, 1.0))
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-12:
        return None
    return x[:2] / x[2]


def _line_edge_int(p, dv, q1, q2):
    """Intersection of line through p with direction dv and line q1-q2."""
    return _line_int(p, p + np.asarray(dv, dtype=float), q1, q2)


def projective_midlines(corners) -> dict:
    """Projective-correct midlines for both axes.

    center = quad diagonal intersection; the seam line runs through the
    center and the vanishing point of the two edges PARALLEL to the seam
    (i.e. the edges it does NOT join). Near-parallel edges (vanishing
    point at infinity) fall back to the mean edge direction.
    Returns {"lr": {"endpoints": (2, 2) float array, ...}, "bf": {...}}
    with endpoints ordered (smaller-x first) and the raw center/vp kept
    for debugging.
    """
    q = _as_quad(corners)
    BL, BR, FR, FL = q
    center = _line_int(BL, FR, BR, FL)
    if center is None:  # degenerate quad: fall back to vertex mean
        center = q.mean(axis=0)

    out = {}
    for axis, join_a, join_b, par_a, par_b in (
            ("lr", (BL, FL), (BR, FR), (BL, BR), (FL, FR)),
            ("bf", (BL, BR), (FL, FR), (BL, FL), (BR, FR))):
        vp = _line_int(par_a[0], par_a[1], par_b[0], par_b[1])
        if vp is None or not np.all(np.isfinite(vp)):
            d1 = par_a[1] - par_a[0]
            d2 = par_b[1] - par_b[0]
            n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
            dv = (d1 / n1 + d2 / n2) / 2.0 if n1 > 1e-9 and n2 > 1e-9 \
                else np.array([1.0, 0.0])
            at_infinity = True
        else:
            dv = vp - center
            if float(np.linalg.norm(dv)) < 1e-9:
                dv = (par_a[1] - par_a[0]) + (par_b[1] - par_b[0])
            at_infinity = False
        e1 = _line_edge_int(center, dv, join_a[0], join_a[1])
        e2 = _line_edge_int(center, dv, join_b[0], join_b[1])
        if e1 is None or e2 is None:  # truly degenerate: edge midpoints
            e1 = (join_a[0] + join_a[1]) / 2.0
            e2 = (join_b[0] + join_b[1]) / 2.0
        eps = np.array(sorted([e1, e2], key=lambda p: p[0]))
        out[axis] = {"endpoints": eps, "center": np.asarray(center, float),
                     "vp": (None if at_infinity else np.asarray(vp, float))}
    return out


def rectify_face(bgr: np.ndarray, corners) -> tuple:
    """Homography mapping the quad to an upright rectangle.

    Destination: BL->(0,H), BR->(W,H), FR->(W,0), FL->(0,0), so rect row
    y == H/2 is the "lr" prior line and col x == W/2 the "bf" prior line.
    Returns (M, Mi, W, H).
    """
    q = _as_quad(corners)
    BL, BR, FR, FL = q

    def _len(a, b):
        return float(np.linalg.norm(a - b))

    W = max(40, int(round((_len(BL, BR) + _len(FL, FR)) / 2.0)))
    H = max(40, int(round((_len(BL, FL) + _len(BR, FR)) / 2.0)))
    src = q.astype(np.float32)
    dst = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Mi = np.linalg.inv(M.astype(np.float64))
    return M, Mi, W, H


def _warp_rect(img: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.warpPerspective(img, M, (W + 1, H + 1),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _face_sat(bgr: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _warp_rect(hsv[:, :, 1], M, W, H).astype(np.float64)


def _est_cfg(est_cfg: dict | None, key: str, default: float) -> float:
    try:
        return float((est_cfg or {}).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _estimate_tape_hw(sat: np.ndarray, axis: str, prior: int) -> tuple:
    """Half-width (px, rect) of the tape dip from the mean profile.

    Seeds the per-slice search window only; acceptance never uses this
    (width is gated per-slice in mm via paired-edge widths)."""
    H, W_ = sat.shape
    horizontal = (axis == "lr") or (axis == "h")
    if horizontal:
        x0, x1 = int(W_ * 0.25), int(W_ * 0.75)
        v = cv2.GaussianBlur(sat[:, x0:x1].mean(axis=1).reshape(-1, 1),
                             (1, 7), 0)[:, 0]
    else:
        y0, y1 = int(H * 0.25), int(H * 0.75)
        v = cv2.GaussianBlur(sat[y0:y1, :].mean(axis=0).reshape(1, -1),
                             (7, 1), 0)[0, :]
    n = len(v)
    side = (v[max(0, prior - 30):max(0, prior - 10)].mean()
            + v[min(n, prior + 11):min(n, prior + 31)].mean()) / 2.0
    depth = float(side - v[max(0, prior - 4):min(n, prior + 5)].mean())
    dip = v[max(0, prior - 15):min(n, prior + 16)]
    below = np.where(dip < side - 0.5 * depth)[0] if depth > 0 else np.array([])
    hw = float(np.clip(len(below) / 2.0 if len(below) else 4.0, 2.5, 12.0))
    return hw, depth


def _band_axis_score(v: np.ndarray) -> tuple[float, int]:
    """Tape-band score of a 1-D mean profile.

    Tape reads as a low-saturation band: surround-minus-band depth plus
    half the local gradient (the two tape edges). Scans every row/col,
    returns (best score, index).
    """
    n = len(v)
    g = np.abs(np.gradient(v.astype(float)))
    best, bi = 0.0, n // 2
    for i in range(12, n - 12):
        band = v[max(0, i - 4):min(n, i + 5)].mean()
        out = (v[max(0, i - 20):max(0, i - 6)].mean()
               + v[min(n, i + 7):min(n, i + 21)].mean()) / 2.0
        edge = g[max(0, i - 5):min(n - 1, i + 5)].max()
        s = float((out - band) + 0.5 * edge)
        if s > best:
            best, bi = s, i
    return best, bi


def axis_evidence(bgr: np.ndarray, corners, candidate: str,
                  _sat: np.ndarray | None = None,
                  _shape: tuple | None = None) -> float:
    """Score tape presence along a candidate midline band ("lr"|"bf").

    Rectifies the top face, averages saturation over the middle 60% of
    the across-seam span (away from face edges), and returns the
    surround-minus-band + tape-edge score. Higher = more tape-like.
    """
    if candidate not in ("lr", "bf"):
        raise ValueError("candidate must be 'lr' or 'bf'")
    if _sat is None:
        M, _, W, H = rectify_face(bgr, corners)
        sat = _face_sat(bgr, M, W, H)
    else:
        sat = _sat
        W, H = _shape
    Hh, Ww = sat.shape
    if candidate == "lr":  # seam runs horizontally: scan rows
        x0, x1 = int(Ww * 0.2), int(Ww * 0.8)
        prof = sat[:, x0:x1].mean(axis=1)
    else:
        y0, y1 = int(Hh * 0.2), int(Hh * 0.8)
        prof = sat[y0:y1, :].mean(axis=0)
    score, _ = _band_axis_score(np.asarray(prof, float))
    return float(score)


def sam3_tape_evidence(image_path: str, python_bin: str,
                        timeout_s: int = 300) -> dict:
    """Query the existing sam3_worker with tape prompts (drives, not edits).

    Spawns the worker subprocess directly with prompts
    ["packing tape", "tape"] and returns {"ok", "score", "quad",
    "prompt", "reason"}. Unavailable worker -> ok False, caller treats
    the bonus as 0. One model instance per call; the eval driver opens
    one worker per --sam3 run and reuses it via ``session``.
    """
    from pathlib import Path as _P
    worker = _P(__file__).resolve().parent / "sam3_worker.py"
    env = dict(__import__("os").environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        proc = subprocess.Popen(
            [python_bin, str(worker)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)
    except OSError as exc:
        return {"ok": False, "reason": f"spawn_failed: {exc}"}
    try:
        import select as _select
        fd = proc.stdout.fileno()
        _r, _, _ = _select.select([fd], [], [], timeout_s)
        hello = proc.stdout.readline() if _r else ""
        try:
            h = json.loads(hello) if hello else None
        except ValueError:
            h = None
        if not isinstance(h, dict) or h.get("type") != "handshake":
            return {"ok": False, "reason": "no_handshake"}
        req = {"id": 1, "image": str(image_path),
               "prompts": ["packing tape", "tape"], "threshold": 0.5}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        _r, _, _ = _select.select([fd], [], [], timeout_s)
        line = proc.stdout.readline() if _r else ""
        try:
            resp = json.loads(line) if line else None
        except ValueError:
            resp = None
        if not isinstance(resp, dict) or not resp.get("ok", False):
            reason = (resp or {}).get("reason", "no_response") \
                if isinstance(resp, dict) else "no_response"
            return {"ok": False, "reason": str(reason)}
        return {"ok": True, "score": float(resp.get("concept_score", 0.0)),
                "quad": resp.get("corners_px"),
                "prompt": resp.get("prompt")}
    finally:
        try:
            proc.kill()
        except OSError:
            pass

# ---- viewing shift ----------------------------------------------------------

def _xyz_quad(corners_xyz) -> np.ndarray | None:
    """(4, 3) camera-frame corners in CORNER_NAMES order, or None."""
    try:
        if isinstance(corners_xyz, dict):
            x = np.array([corners_xyz[n] for n in CORNER_NAMES], float)
        else:
            x = np.asarray(corners_xyz, dtype=float)
        x = x.reshape(4, -1)[:, :3]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if x.shape != (4, 3) or not np.all(np.isfinite(x)):
        return None
    return x


def view_geometry(corners_xyz, axis: str,
                  big_long_side_m: float = BIG_LONG_SIDE_M) -> dict | None:
    """Camera-relative geometry of the seam for the viewing-shift model.

    ``corners_xyz`` are the top-face corners in the CAMERA frame (camera
    centre at the origin). Returns {box_class, long_side_mm,
    away_rect_sign, view_angle_deg, view_angle_across_deg} or None when
    the corners are missing/degenerate (collinear, zero area).
    ``away_rect_sign`` is +1 when the rect +across direction (rect +y =
    toward the back edge for "lr"; rect +x = toward the right edge for
    "bf") increases the distance from the camera centre along the top
    plane, else -1. ``view_angle_deg`` = angle between the camera ray at
    the face centre and the plane normal; ``view_angle_across_deg`` = that
    ray's tilt from the normal within the (across-seam, normal) plane,
    signed + when the ray leans away along the across direction.
    """
    if axis not in ("lr", "bf"):
        return None
    X = _xyz_quad(corners_xyz)
    if X is None:
        return None
    BL, BR, FR, FL = X
    nrm = np.cross(BR - BL, FL - BL) + np.cross(FL - FR, BR - FR)
    nn = float(np.linalg.norm(nrm))
    if nn < 1e-9:
        return None
    nrm = nrm / nn
    if axis == "lr":
        u = (BL + BR) / 2.0 - (FL + FR) / 2.0
    else:
        u = (BR + FR) / 2.0 - (BL + FL) / 2.0
    u = u - float(u @ nrm) * nrm
    un = float(np.linalg.norm(u))
    P = X.mean(axis=0)
    pn = float(np.linalg.norm(P))
    if un < 1e-9 or pn < 1e-9:
        return None
    u = u / un
    ray = P / pn
    along_u = float(ray @ u)
    away = 1.0 if along_u >= 0.0 else -1.0
    rn = abs(float(ray @ nrm))
    theta = float(np.degrees(np.arccos(min(1.0, rn))))
    theta_u = float(np.degrees(np.arctan2(abs(along_u), rn)))
    sides = [float(np.linalg.norm(X[(i + 1) % 4] - X[i])) for i in range(4)]
    long_side = max(0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3]))
    return {"box_class": "big" if long_side > big_long_side_m else "small",
            "long_side_mm": round(long_side * 1000.0, 1),
            "away_rect_sign": int(away),
            "view_angle_deg": round(theta, 2),
            "view_angle_across_deg": round(theta_u, 2)}


def _class_value(table, cls: str, default: float) -> float:
    try:
        return float((table or {}).get(cls, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def expected_view_shift(corners_xyz, axis: str,
                        est_cfg: dict | None = None,
                        profile: str | None = None) -> dict:
    """Expected apparent seam shift s (mm, + = away from the camera).

    s = shift_mm[class] + k_tan_mm[class] * tan(view_angle_across) from
    ``est_cfg["view_shift"]`` (defaults VIEW_SHIFT_MM /
    VIEW_SHIFT_K_TAN_MM, fitted on unlabelled captures; see module
    constants). Degenerate/missing geometry or ``enabled`` False -> s = 0
    with ``reason`` set (the confirmation then compares with the plain
    prior, as before).

    ``profile`` (box profile name, optional): ``"generic"`` -> s = 0 with
    ``reason="generic_profile"`` (an unknown box has no validated shift);
    any other value / None keeps the per-class behaviour.
    """
    vcfg = (est_cfg or {}).get("view_shift") if isinstance(est_cfg, dict) \
        else None
    vcfg = vcfg if isinstance(vcfg, dict) else {}
    enabled = bool(vcfg.get("enabled", True))
    try:
        big_side = float(vcfg.get("big_long_side_m", BIG_LONG_SIDE_M))
    except (TypeError, ValueError):
        big_side = BIG_LONG_SIDE_M
    geo = view_geometry(corners_xyz, axis, big_side)
    out = {"enabled": enabled, "model": "class_const_plus_k_tan",
           "box_class": None, "view_angle_deg": None,
           "view_angle_across_deg": None, "away_rect_sign": None,
           "expected_shift_mm": 0.0, "reason": None}
    if geo is None:
        out["reason"] = "no_view_geometry"
        return out
    out.update({k: geo[k] for k in ("box_class", "long_side_mm",
                                    "view_angle_deg",
                                    "view_angle_across_deg",
                                    "away_rect_sign")})
    if not enabled:
        out["reason"] = "disabled"
        return out
    if profile == "generic":
        out["reason"] = "generic_profile"
        return out
    if profile in _profiles.PROFILES and _profiles.PROFILES[profile].own_constants:
        # later profile (large_392): its OWN shift, never the legacy class table
        out["reason"] = f"profile_{profile}"
        out["expected_shift_mm"] = float(_profiles.PROFILES[profile].view_shift_mm)
        return out
    cls = geo["box_class"]
    c = _class_value(vcfg.get("shift_mm"), cls, VIEW_SHIFT_MM.get(cls, 0.0))
    k = _class_value(vcfg.get("k_tan_mm"), cls,
                     VIEW_SHIFT_K_TAN_MM.get(cls, 0.0))
    s = c + k * float(np.tan(np.radians(geo["view_angle_across_deg"])))
    out["expected_shift_mm"] = round(float(s), 3) if np.isfinite(s) else 0.0
    return out


def _shift_rect_line(line_rect, axis: str, shift_rect: float):
    """Translate a rect line across the seam (rect y for lr, x for bf)."""
    (x0, y0), (x1, y1) = line_rect
    if axis == "lr":
        return ((x0, y0 + shift_rect), (x1, y1 + shift_rect))
    return ((x0 + shift_rect, y0), (x1 + shift_rect, y1))


def _signed_offsets_rect(prior_rect, obs_rect, axis: str, away: int,
                         mm_per_px) -> dict | None:
    """Signed across-seam offset of the observed line from the prior.

    Measured along the rect across axis (y for lr, x for bf) at 25/50/75%
    of the prior's rect span; + = away from the camera (``away`` =
    rect sign of "away"). Returns {"offset_mm": [..3], "mid_mm",
    "offset_rect_px": [..3]} (mm None without a scale) or None.
    """
    try:
        (px0, py0), (px1, py1) = prior_rect
        (gx0, gy0), (gx1, gy1) = obs_rect
    except (TypeError, ValueError, IndexError):
        return None
    offs = []
    for f in (0.25, 0.5, 0.75):
        sx, sy = px0 + f * (px1 - px0), py0 + f * (py1 - py0)
        if axis == "lr":
            dx = gx1 - gx0
            if abs(dx) < 1e-9:
                return None
            gy = gy0 + (sx - gx0) * (gy1 - gy0) / dx
            offs.append((gy - sy) * away)
        else:
            dy = gy1 - gy0
            if abs(dy) < 1e-9:
                return None
            gx = gx0 + (sy - gy0) * (gx1 - gx0) / dy
            offs.append((gx - sx) * away)
    ok_mm = mm_per_px is not None and np.isfinite(mm_per_px)
    mm = [round(float(o * mm_per_px), 3) for o in offs] if ok_mm else None
    return {"offset_rect_px": [round(float(o), 2) for o in offs],
            "offset_mm": mm, "mid_mm": (mm[1] if mm else None)}

# ---- observed-tape support ------------------------------------------------

def _rect_gray(bgr: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _warp_rect(gray, M, W, H).astype(np.float64)


def _gap_halfwin(W: int, H: int, est_cfg: dict | None) -> int:
    """Search half-window (rect px) around the prior midline.

    Default ±8 px at 720p scaled by face size (min(W, H) / 150, clamped to
    [6, 20]); ``gap_halfwin_px`` overrides. ``gap_band_mm`` (via the plane
    scale) overrides when a valid scale is available — see refine_on_gap.
    """
    direct = (est_cfg or {}).get("gap_halfwin_px") \
        if isinstance(est_cfg, dict) else None
    try:
        if direct is not None:
            return max(2, int(direct))
    except (TypeError, ValueError):
        pass
    return int(round(float(np.clip(8.0 * min(W, H) / 150.0, 6.0, 20.0))))


def _gap_slice_support(v: np.ndarray, prior: int, halfwin: int,
                       min_depth: float, min_shoulder: float,
                       max_width: int) -> dict | None:
    """Best thin dark valley in the search window for one slice.

    The flap gap reads as a narrow dark valley in rectified gray with
    bright cardboard on BOTH sides (visible through or beside translucent
    tape): valley depth = min(left-shoulder, right-shoulder) - valley must
    clear ``min_depth``, each shoulder alone must clear ``min_shoulder``
    (rejects one-sided shadow/tape edges), and the below-half-depth valley
    width must fit ``max_width`` rect px (rejects broad shadow bands;
    width is plausibility-checked at line level, in mm). Returns
    {"center", "depth", "width"} or None. ``center`` is subpixel via a
    parabola fit on the raw profile.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    lo, hi = max(8, prior - halfwin), min(n - 9, prior + halfwin)
    if hi < lo:
        return None
    vs = cv2.GaussianBlur(v.reshape(-1, 1), (1, 3), 0)[:, 0]
    fast = bool(np.all(np.isfinite(vs)) and not np.any(np.signbit(vs)))
    if fast:
        # Speed: the scan below only indexes, slices, takes maxima and
        # compares, so running it on a Python float list gives identical
        # values (max over finite, sign-bit-free floats is exact and
        # unambiguous) without per-element numpy scalar overhead. Other
        # profiles keep the numpy path.
        vs = vs.tolist()
    best = None
    for j in range(lo, hi + 1):
        f = float(vs[j])
        lseg = vs[max(0, j - 8):max(0, j - 1)]
        rseg = vs[j + 2:min(n, j + 9)]
        if len(lseg) < 2 or len(rseg) < 2:
            continue
        if fast:
            lref, rref = float(max(lseg)), float(max(rseg))
        else:
            lref, rref = float(lseg.max()), float(rseg.max())
        depth = min(lref, rref) - f
        if depth < min_depth:
            continue
        if lref - f < min_shoulder or rref - f < min_shoulder:
            continue
        mid = min(lref, rref) - depth / 2.0
        a = j
        while a - 1 >= max(0, j - 9) and vs[a - 1] < mid:
            a -= 1
        b = j
        while b + 1 < min(n, j + 10) and vs[b + 1] < mid:
            b += 1
        width = int(b - a + 1)
        if width > max_width:
            continue
        j0 = max(1, min(n - 2, j))
        y0, y1, y2 = v[j0 - 1], v[j0], v[j0 + 1]
        den = y0 - 2.0 * y1 + y2
        off = (y0 - y2) / (2.0 * den) if den > 1e-9 else 0.0
        cc = float(j0 + float(np.clip(off, -1.0, 1.0)))
        if best is None or depth > best[0]:
            best = (depth, cc, depth, width)
    if best is None:
        return None
    return {"center": best[1], "depth": best[2], "width": best[3]}


def _slice_support(v: np.ndarray, prior: int, halfwin: int, hw: int,
                   smin: float, neg_min: float, pos_min: float) -> dict | None:
    """Best dip in the search window + paired tape-edge check for one slice.

    Returns {"center", "score", "neg", "pos", "width", "valid"} or None when
    the window is degenerate. ``valid`` requires a deep-enough dip AND a
    negative lobe (leading tape edge) AND a positive lobe (trailing tape
    edge): a pale printed stripe or glare band has the dip but not the
    paired edges, so it fails here. ``width`` is the lobe-peak separation
    in rect px (plausibility-checked at line level, in mm). Legacy tape
    detector kept for the axis-audit tape stats and the eval driver's
    old-tape comparison.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    r = int(round(hw))
    gap, span = r + 2, r + 14
    lo, hi = max(span, prior - halfwin), min(n - span - 1, prior + halfwin)
    if hi < lo:
        return None
    best = None
    for j in range(lo, hi + 1):
        b0, b1 = max(0, j - r), min(n, j + r + 1)
        s0a, s0b = max(0, j - span), max(0, j - gap)
        s1a, s1b = min(n, j + gap + 1), min(n, j + span + 1)
        if s0b <= s0a or s1b <= s1a:
            continue
        s = float((v[s0a:s0b].mean() + v[s1a:s1b].mean()) / 2.0
                  - v[b0:b1].mean())
        if best is None or s > best[0]:
            best = (s, j)
    if best is None:
        return None
    s, j = best
    b0, b1 = max(0, j - r - 2), min(n, j + r + 3)
    seg = v[b0:b1]
    w = np.clip(seg.max() - seg, 0.0, None)
    idx = np.arange(b0, b1, dtype=float)
    cc = float((w * idx).sum() / w.sum()) if w.sum() > 0 else float(j)
    g = np.gradient(v)
    gs = cv2.GaussianBlur(g.reshape(-1, 1), (1, 5), 0)[:, 0]
    jj = int(round(cc))
    elo, ehi = max(0, jj - r - 4), min(n, jj + r + 5)
    neg = float(gs[elo:jj + 1].min()) if jj > elo else 0.0
    pos = float(gs[jj:ehi].max()) if jj < ehi - 1 else 0.0
    pn = int(np.argmin(gs[elo:jj + 1])) + elo if jj > elo else jj
    pp = int(np.argmax(gs[jj:ehi])) + jj if jj < ehi - 1 else jj
    width = int(pp - pn)
    valid = bool(s >= smin and neg <= -neg_min and pos >= pos_min)
    return {"center": cc, "score": s, "neg": neg, "pos": pos,
            "width": width, "valid": valid}

def _unblended_huber(t: np.ndarray, c: np.ndarray,
                     iters: int = 8) -> tuple:
    """Huber-IRLS line fit on observed slice centers. No slope shrinkage,
    no blending toward the prior: the line is observations alone. The
    prior only defined the search window the centers came from."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.polyfit(t, c, 1)
    for _ in range(iters):
        r = np.abs(c - (A[0] * t + A[1]))
        w = 1.0 / np.maximum(1.0, r / 1.5)
        Wm = np.diag(w)
        X = np.vstack([t, np.ones_like(t)]).T
        A = np.linalg.inv(X.T @ Wm @ X) @ (X.T @ Wm @ c)
    r = np.abs(c - (A[0] * t + A[1]))
    return float(A[0]), float(A[1]), r


def _rect_px_to_mm(corners_xyz, axis: str, W: int, H: int) -> float | None:
    """mm per rect px along the across-seam direction, via the top-plane
    scale. ``corners_xyz`` is a name->xyz dict or a (4, 2+) array in
    CORNER_NAMES order. None when unavailable or degenerate."""
    try:
        if isinstance(corners_xyz, dict):
            xm = np.array([corners_xyz[n] for n in CORNER_NAMES], float)
        else:
            xm = np.asarray(corners_xyz, dtype=float).reshape(4, -1)
        if not np.all(np.isfinite(xm)):
            return None
        if axis == "h":
            side_m = (np.linalg.norm(xm[3, :3] - xm[0, :3])
                      + np.linalg.norm(xm[2, :3] - xm[1, :3])) / 2.0
            return float(side_m * 1000.0 / H) if H > 0 else None
        side_m = (np.linalg.norm(xm[1, :3] - xm[0, :3])
                  + np.linalg.norm(xm[2, :3] - xm[3, :3])) / 2.0
        return float(side_m * 1000.0 / W) if W > 0 else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


# ---- user darkline detector ---------------------------------------------

def _bilinear_sample(gray: np.ndarray, xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of a float gray image at fractional coords."""
    h, w = gray.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    dx = xs - x0
    dy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    g = np.asarray(gray, dtype=np.float64)
    return ((1.0 - dx) * (1.0 - dy) * g[y0c, x0c]
            + dx * (1.0 - dy) * g[y0c, x1c]
            + (1.0 - dx) * dy * g[y1c, x0c]
            + dx * dy * g[y1c, x1c])


def _darkline_halfwin(W: int, H: int, est_cfg: dict | None) -> int:
    """Perpendicular profile half-width (rect px) for darkline stations.

    Default ±8 px at 720p scaled by face size (min(W, H) / 150, clamped
    to [5, 16]); ``darkline_halfwin_px`` overrides.
    """
    direct = (est_cfg or {}).get("darkline_halfwin_px") \
        if isinstance(est_cfg, dict) else None
    try:
        if direct is not None:
            return max(3, int(direct))
    except (TypeError, ValueError):
        pass
    return int(round(float(np.clip(8.0 * min(W, H) / 150.0, 5.0, 16.0))))


def refine_on_darkline(bgr: np.ndarray, corners, candidate: str,
                       halfwin_px: int | None = None,
                       est_cfg: dict | None = None,
                       corners_xyz=None,
                       center_offset_rect: float = 0.0,
                       keep_slices: bool = False) -> tuple:
    """Fit the darkest-contrast line along the prior (user method).

    Stations every ~2-3 px along the middle ~80% of the prior line; at
    each station the perpendicular grayscale profile (±halfwin, lightly
    smoothed) contributes its darkest point when shoulder contrast
    (mean of both shoulders − valley) clears ``darkline_min_contrast``.
    Robust Huber fit on kept points; endpoints = line ∩ opposite face
    edges. Returns (endpoints (2,2) smaller-x-first or None, score,
    info). Fail-closed like refine_on_gap: coverage/residual/gap gates;
    the mm width gate is SKIPPED (a 1-px dark line has no valley
    width); ``corners_xyz`` is still REQUIRED for ok so the confirm
    agreement that consumes this line has a valid plane scale
    (reject ``no_metric_scale`` otherwise).

    Each shoulder must ALSO be brighter than the valley on its own
    (``darkline_min_shoulder``, mean of that side − valley): a station
    whose darkest point is the dark side of a one-sided step (tape-edge
    highlight next to plain cardboard) is dropped even when the mean of
    both shoulders clears the contrast gate. ``center_offset_rect``
    shifts the station profiles across the seam (rect px, + = rect
    +y/+x) so the window is centred on the expected apparent line
    (prior + viewing shift); the fit itself stays observation-only.
    """
    if candidate not in ("lr", "bf"):
        raise ValueError("candidate must be 'lr' or 'bf'")
    min_contrast = _est_cfg(est_cfg, "darkline_min_contrast",
                             DARKLINE_MIN_CONTRAST)
    min_side = _est_cfg(est_cfg, "darkline_min_shoulder",
                        DARKLINE_MIN_SHOULDER)
    min_cov = _est_cfg(est_cfg, "darkline_min_coverage",
                        DARKLINE_MIN_COVERAGE)
    max_gap = _est_cfg(est_cfg, "darkline_max_gap_frac",
                        DARKLINE_MAX_GAP_FRAC)
    max_res = _est_cfg(est_cfg, "darkline_max_residual_px",
                        DARKLINE_MAX_RESIDUAL_PX)
    axis = "h" if candidate == "lr" else "v"
    M, Mi, W, H = rectify_face(bgr, corners)
    gray = cv2.GaussianBlur(_rect_gray(bgr, M, W, H), (0, 0), 1.0)
    Hh, Ww = gray.shape
    prior = float(Hh // 2 if axis == "h" else Ww // 2)
    try:
        c_off = float(center_offset_rect)
    except (TypeError, ValueError):
        c_off = 0.0
    if not np.isfinite(c_off):
        c_off = 0.0
    prior += c_off
    if halfwin_px is None:
        halfwin = _darkline_halfwin(W, H, est_cfg)
    else:
        halfwin = max(3, int(halfwin_px))
    info = {"darkline_min_contrast": round(float(min_contrast), 2),
            "darkline_min_shoulder": round(float(min_side), 2),
            "prior_rect": round(float(prior), 2),
            "center_offset_rect": round(float(c_off), 2),
            "halfwin": int(halfwin)}
    span = (Ww if axis == "h" else Hh)
    lo, hi = span * 0.10, span * 0.90
    step = float(np.clip(span * 0.70 / 100.0, 2.0, 3.0))
    ts = np.arange(lo, hi + 1e-9, step)
    across = np.arange(-halfwin, halfwin + 1, dtype=np.float64)
    T, C, Ct = [], [], []
    n_one_sided = 0
    for t in ts:
        if axis == "h":
            xs = np.full_like(across, float(t))
            ys = float(prior) + across
            ok = (ys >= 0) & (ys <= Hh - 1)
        else:
            xs = float(prior) + across
            ys = np.full_like(across, float(t))
            ok = (xs >= 0) & (xs <= Ww - 1)
        if int(ok.sum()) < 2 * halfwin:
            continue
        v = _bilinear_sample(gray, xs[ok], ys[ok])
        a = across[ok]
        j = int(np.argmin(v))
        valley = float(v[j])
        lsh = v[a <= a[j] - 3.0]
        rsh = v[a >= a[j] + 3.0]
        if len(lsh) == 0 or len(rsh) == 0:
            continue
        contrast = float((lsh.mean() + rsh.mean()) / 2.0 - valley)
        if contrast < min_contrast:
            continue
        if min(float(lsh.mean()), float(rsh.mean())) - valley < min_side:
            n_one_sided += 1
            continue
        j1 = max(0, min(len(v) - 1, j))
        y0 = float(v[max(0, j1 - 1)])
        y1 = float(v[j1])
        y2 = float(v[min(len(v) - 1, j1 + 1)])
        den = y0 - 2.0 * y1 + y2
        off = (y0 - y2) / (2.0 * den) if den > 1e-9 else 0.0
        cc = float(a[j1] + float(np.clip(off, -1.0, 1.0)))
        T.append(float(t))
        C.append(float(prior + cc))
        Ct.append(contrast)
    n_stations = len(ts)
    info.update({"n_slices": len(T), "n_slices_total": int(n_stations),
                 "n_one_sided": int(n_one_sided),
                 "support_frac": round(float(len(T) / max(1, n_stations)),
                                       3)})
    if len(T) < 4:
        info["reject"] = "insufficient_darkline_support"
        return None, 0.0, info
    t = np.array(T, dtype=float)
    cc = np.array(C, dtype=float)
    m, b, resid = _unblended_huber(t, cc)
    inres = resid[resid < RESID_INLIER_PX]
    resid_px = float(np.median(inres)) if len(inres) else float(
        np.median(resid))
    inl = np.abs(cc - (m * t + b)) < RESID_INLIER_PX
    det = {}
    for tt, ccc in zip(T, C):
        det[round(float(tt), 6)] = float(ccc)
    # Coverage over the FULL station span (misses + outliers unsupported),
    # not just kept stations — a short block like a barcode scores
    # n_stations_span coverage, never 1.0.
    keys = np.array([float(t) for t in ts])
    sup_full = np.array(
        [(round(float(k), 6) in det
          and abs(det[round(float(k), 6)] - (m * k + b)) < RESID_INLIER_PX)
         for k in keys])
    coverage = float(sup_full.mean()) if len(sup_full) else 0.0
    gap = run = 0
    for f in list(sup_full) + [True]:
        if not f:
            run += 1
        else:
            gap = max(gap, run)
            run = 0
    gap_frac = float(gap / max(1, len(sup_full)))
    med_contrast = float(np.median(np.array(Ct))) if Ct else 0.0
    mm_per_px = _rect_px_to_mm(corners_xyz, axis, W, H)
    if keep_slices:
        info["_slice_t"] = [float(v) for v in T]
        info["_slice_c"] = [float(v) for v in C]
        info["_line"] = (float(m), float(b))
    info.update({"slope": round(float(m), 4),
                 "intercept_rect": round(float(b), 2),
                 "coverage": round(coverage, 3),
                 "max_gap_frac": round(gap_frac, 3),
                 "residual_px": round(resid_px, 2),
                 "median_contrast": round(med_contrast, 1),
                 "mm_per_px": (round(float(mm_per_px), 4)
                               if mm_per_px is not None else None),
                 "n_inliers": int(inl.sum())})
    if coverage < min_cov:
        info["reject"] = "low_coverage"
        return None, 0.0, info
    if gap_frac > max_gap:
        info["reject"] = "gappy_support"
        return None, 0.0, info
    if resid_px > max_res:
        info["reject"] = "high_residual"
        return None, 0.0, info
    if mm_per_px is None:
        info["reject"] = "no_metric_scale"
        return None, 0.0, info

    def _back(pt):
        p = Mi @ np.array([pt[0], pt[1], 1.0])
        return p[:2] / p[2]

    if axis == "h":
        e1, e2 = _back([0.0, b]), _back([float(W), m * W + b])
    else:
        e1, e2 = _back([b, 0.0]), _back([m * H + b, float(H)])
    eps = np.array(sorted([e1, e2], key=lambda p: p[0]))
    score = float(np.clip(coverage, 0.0, 1.0)
                  * np.clip(1.0 - resid_px / (max_res + 1.0), 0.0, 1.0))
    info["score"] = round(score, 3)
    return eps, score, info


def _gap_support_for_axis(gray_rect: np.ndarray, rect_axis: str, prior: int,
                          halfwin: int, min_depth: float,
                          min_shoulder: float, max_width: int) -> float:
    """Fraction of slices with a gap detection (axis scoring aid)."""
    Hh, Ww = gray_rect.shape
    hit = tot = 0
    if rect_axis == "h":
        for x in range(int(Ww * 0.15), int(Ww * 0.85), 2):
            tot += 1
            if _gap_slice_support(gray_rect[:, x], prior, halfwin,
                                  min_depth, min_shoulder,
                                  max_width) is not None:
                hit += 1
    else:
        for y in range(int(Hh * 0.15), int(Hh * 0.85), 2):
            tot += 1
            if _gap_slice_support(gray_rect[y, :], prior, halfwin,
                                  min_depth, min_shoulder,
                                  max_width) is not None:
                hit += 1
    return float(hit / max(1, tot))


def refine_on_gap(bgr: np.ndarray, corners, candidate: str,
                  halfwin_px: int | None = None,
                  est_cfg: dict | None = None,
                  corners_xyz=None,
                  center_offset_rect: float = 0.0,
                  keep_slices: bool = False) -> tuple:
    """Fit the flap-gap line from observations alone.

    The prior is used ONLY for the search window (rect row/col +/- halfwin
    around the prior midline; ``gap_band_mm`` in est_cfg optionally sets
    the half-window in mm via the plane scale). Per-slice thin-valley
    centers (dark gap, bright shoulders both sides) are fit with an
    unblended Huber line; the published endpoints are that observed line
    intersected with the face edges. Returns (endpoints (2,2)
    smaller-x-first or None, score float, info dict). Fail-closed:
    weak/partial evidence -> (None, 0, info) with reject set.
    ``corners_xyz`` is REQUIRED for ok: without a valid metric scale the
    gap width cannot be checked in mm (reject ``no_metric_scale``).
    ``center_offset_rect`` (rect px, rounded) moves the search window
    centre across the seam onto the expected apparent line (prior +
    viewing shift); ``line_left_window`` is checked against that centre.
    """
    if candidate not in ("lr", "bf"):
        raise ValueError("candidate must be 'lr' or 'bf'")
    min_depth = _est_cfg(est_cfg, "gap_min_depth", GAP_MIN_DEPTH)
    min_sh = _est_cfg(est_cfg, "gap_min_shoulder", GAP_MIN_SHOULDER)
    max_w = int(_est_cfg(est_cfg, "gap_max_width_px", GAP_MAX_WIDTH_PX))
    min_cov = _est_cfg(est_cfg, "min_coverage", MIN_COVERAGE)
    max_gap = _est_cfg(est_cfg, "max_gap_frac", MAX_GAP_FRAC)
    max_res = _est_cfg(est_cfg, "max_residual_px", MAX_RESIDUAL_PX)
    wmin = _est_cfg(est_cfg, "gap_width_mm_min", GAP_WIDTH_MM_MIN)
    wmax = _est_cfg(est_cfg, "gap_width_mm_max", GAP_WIDTH_MM_MAX)
    axis = "h" if candidate == "lr" else "v"
    M, Mi, W, H = rectify_face(bgr, corners)
    gray = _rect_gray(bgr, M, W, H)
    Hh, Ww = gray.shape
    prior = Hh // 2 if axis == "h" else Ww // 2
    try:
        c_off = int(round(float(center_offset_rect)))
    except (TypeError, ValueError, OverflowError):
        c_off = 0
    prior += c_off
    if halfwin_px is None:
        halfwin = _gap_halfwin(W, H, est_cfg)
        band_mm = (est_cfg or {}).get("gap_band_mm") \
            if isinstance(est_cfg, dict) else None
        try:
            if band_mm is not None:
                mm_per_px = _rect_px_to_mm(corners_xyz, axis, W, H)
                if mm_per_px is not None and mm_per_px > 1e-9:
                    halfwin = max(2, int(round(float(band_mm)
                                              / mm_per_px)))
        except (TypeError, ValueError):
            pass
    else:
        halfwin = int(halfwin_px)
    info = {"gap_min_depth": round(float(min_depth), 2),
            "gap_min_shoulder": round(float(min_sh), 2),
            "gap_max_width_px": int(max_w),
            "prior_rect": int(prior), "center_offset_rect": int(c_off),
            "halfwin": int(halfwin)}
    T, C, Dp, Wd = [], [], [], []
    if axis == "h":
        rng = range(int(Ww * 0.15), int(Ww * 0.85), 2)
        for x in rng:
            sup = _gap_slice_support(gray[:, x], prior, halfwin,
                                     min_depth, min_sh, max_w)
            if sup is None:
                continue
            T.append(x)
            C.append(sup["center"])
            Dp.append(sup["depth"])
            Wd.append(sup["width"])
    else:
        rng = range(int(Hh * 0.15), int(Hh * 0.85), 2)
        for y in rng:
            sup = _gap_slice_support(gray[y, :], prior, halfwin,
                                     min_depth, min_sh, max_w)
            if sup is None:
                continue
            T.append(y)
            C.append(sup["center"])
            Dp.append(sup["depth"])
            Wd.append(sup["width"])
    t = np.array(T, dtype=float)
    cc = np.array(C, dtype=float)
    n_slices = len(list(rng))
    info.update({"n_slices": len(T), "n_slices_total": int(n_slices),
                 "support_frac": round(float(len(T) / max(1, n_slices)),
                                       3)})
    if len(T) < 4:
        info["reject"] = "insufficient_gap_support"
        return None, 0.0, info
    m, b, resid = _unblended_huber(t, cc)
    inres = resid[resid < RESID_INLIER_PX]
    resid_px = float(np.median(inres)) if len(inres) else float(
        np.median(resid))
    inl = np.abs(cc - (m * t + b)) < RESID_INLIER_PX
    # Coverage counts every non-inlier slice (miss or outlier) as
    # unsupported over the full middle-70% span.
    det = {int(tt): float(ccc) for tt, ccc in zip(T, C)}
    sup_full = np.array(
        [(tt in det and abs(det[tt] - (m * tt + b)) < RESID_INLIER_PX)
         for tt in rng])
    coverage = float(sup_full.mean()) if len(sup_full) else 0.0
    gap = run = 0
    for f in list(sup_full) + [True]:
        if not f:
            run += 1
        else:
            gap = max(gap, run)
            run = 0
    gap_frac = float(gap / max(1, len(sup_full)))
    med_width = float(np.median(np.array(Wd))) if Wd else 0.0
    med_depth = float(np.median(np.array(Dp))) if Dp else 0.0
    mm_per_px = _rect_px_to_mm(corners_xyz, axis, W, H)
    width_mm = (float(med_width * mm_per_px)
                if mm_per_px is not None else None)
    max_dev = float(np.abs((m * t[inl] + b) - prior).max()) \
        if inl.sum() else float(halfwin + 1)
    if keep_slices:
        # lists: dropped by every JSON-facing filter of the info dict
        info["_slice_t"] = [float(v) for v in T]
        info["_slice_c"] = [float(v) for v in C]
        info["_line"] = (float(m), float(b))
    info.update({"slope": round(float(m), 4),
                 "intercept_rect": round(float(b), 2),
                 "coverage": round(coverage, 3),
                 "max_gap_frac": round(gap_frac, 3),
                 "residual_px": round(resid_px, 2),
                 "median_width_rect": round(med_width, 1),
                 "median_depth": round(med_depth, 1),
                 "width_mm": (round(width_mm, 2)
                              if width_mm is not None else None),
                 "mm_per_px": (round(float(mm_per_px), 4)
                               if mm_per_px is not None else None),
                 "n_inliers": int(inl.sum()),
                 "max_dev_from_prior": round(max_dev, 1)})
    if coverage < min_cov:
        info["reject"] = "low_coverage"
        return None, 0.0, info
    if gap_frac > max_gap:
        info["reject"] = "gappy_support"
        return None, 0.0, info
    if resid_px > max_res:
        info["reject"] = "high_residual"
        return None, 0.0, info
    if max_dev > halfwin:
        info["reject"] = "line_left_window"
        return None, 0.0, info
    # A valid metric scale is required: without corners_xyz/plane the
    # gap width cannot be checked in mm, so the gap cannot be confirmed.
    if width_mm is None:
        info["reject"] = "no_metric_scale"
        return None, 0.0, info
    if not (wmin <= width_mm <= wmax):
        info["reject"] = "implausible_gap_width"
        return None, 0.0, info

    def _back(pt):
        p = Mi @ np.array([pt[0], pt[1], 1.0])
        return p[:2] / p[2]

    if axis == "h":
        e1, e2 = _back([0.0, b]), _back([float(W), m * W + b])
    else:
        e1, e2 = _back([b, 0.0]), _back([m * H + b, float(H)])
    eps = np.array(sorted([e1, e2], key=lambda p: p[0]))
    score = float(np.clip(coverage, 0.0, 1.0)
                  * np.clip(1.0 - resid_px / (max_res + 1.0), 0.0, 1.0))
    info["score"] = round(score, 3)
    return eps, score, info

def refine_on_tape(bgr: np.ndarray, corners, candidate: str,
                   halfwin_px: int | None = None,
                   est_cfg: dict | None = None,
                   corners_xyz=None) -> tuple:
    """Fit the tape centreline from observations alone.

    The prior is used ONLY for the search window (rect row/col +/- halfwin
    around the prior midline). Per-slice dip centers with paired tape
    edges are fit with an unblended Huber line; the published endpoints
    are that observed line intersected with the face edges. Returns
    (endpoints (2,2) smaller-x-first or None, score float, info dict).
    Fail-closed: weak/partial evidence -> (None, 0, info) with reject set.
    ``corners_xyz`` is REQUIRED for ok: without a valid metric scale the
    tape width cannot be checked in mm (reject ``no_metric_scale``).
    """
    if candidate not in ("lr", "bf"):
        raise ValueError("candidate must be 'lr' or 'bf'")
    smin = _est_cfg(est_cfg, "min_slice_score", TAPE_MIN_SLICE_SCORE)
    neg_min = _est_cfg(est_cfg, "min_edge_neg", TAPE_MIN_EDGE_NEG)
    pos_min = _est_cfg(est_cfg, "min_edge_pos", TAPE_MIN_EDGE_POS)
    min_sup = _est_cfg(est_cfg, "min_support_frac", MIN_SUPPORT_FRAC)
    min_cov = _est_cfg(est_cfg, "min_coverage", TAPE_MIN_COVERAGE)
    max_gap = _est_cfg(est_cfg, "max_gap_frac", TAPE_MAX_GAP_FRAC)
    max_res = _est_cfg(est_cfg, "max_residual_px", TAPE_MAX_RESIDUAL_PX)
    wmin = _est_cfg(est_cfg, "tape_width_mm_min", TAPE_WIDTH_MM_MIN)
    wmax = _est_cfg(est_cfg, "tape_width_mm_max", TAPE_WIDTH_MM_MAX)
    axis = "h" if candidate == "lr" else "v"
    M, Mi, W, H = rectify_face(bgr, corners)
    sat = _face_sat(bgr, M, W, H)
    Hh, Ww = sat.shape
    prior = Hh // 2 if axis == "h" else Ww // 2
    if halfwin_px is None:
        halfwin = int(round(
            float(np.clip(15.0 * min(W, H) / 150.0, 8.0, 25.0))))
    else:
        halfwin = int(halfwin_px)
    hw, depth = _estimate_tape_hw(sat, axis, prior)
    info = {"tape_hw_rect": round(float(hw), 2),
            "band_depth": round(float(depth), 2),
            "prior_rect": int(prior), "halfwin": int(halfwin)}
    T, C, V, Wd = [], [], [], []
    if axis == "h":
        rng = range(int(Ww * 0.15), int(Ww * 0.85), 2)
        for x in rng:
            col = cv2.GaussianBlur(sat[:, x:x + 1], (1, 5), 0)[:, 0]
            sup = _slice_support(col, prior, halfwin, hw, smin,
                                 neg_min, pos_min)
            if sup is None:
                continue
            T.append(x)
            C.append(sup["center"])
            V.append(sup["valid"])
            Wd.append(sup["width"])
    else:
        rng = range(int(Hh * 0.15), int(Hh * 0.85), 2)
        for y in rng:
            row = cv2.GaussianBlur(sat[y:y + 1, :], (5, 1), 0)[0, :]
            sup = _slice_support(row, prior, halfwin, hw, smin,
                                 neg_min, pos_min)
            if sup is None:
                continue
            T.append(y)
            C.append(sup["center"])
            V.append(sup["valid"])
            Wd.append(sup["width"])
    t = np.array(T, dtype=float)
    cc = np.array(C, dtype=float)
    vv = np.array(V, dtype=bool)
    support_frac = float(vv.mean()) if len(vv) else 0.0
    info.update({"n_slices": len(T),
                 "support_frac": round(support_frac, 3)})
    if len(T) < 4 or vv.sum() < 4:
        info["reject"] = "insufficient_paired_edge_support"
        return None, 0.0, info
    if support_frac < min_sup:
        info["reject"] = "low_slice_support"
        return None, 0.0, info
    m, b, resid = _unblended_huber(t[vv], cc[vv])
    inres = resid[resid < RESID_INLIER_PX]
    resid_px = float(np.median(inres)) if len(inres) else float(
        np.median(resid))
    inl = np.abs(cc - (m * t + b)) < RESID_INLIER_PX
    sup = vv & inl
    coverage = float(sup.mean())
    gap = run = 0
    for f in list(sup) + [True]:
        if not f:
            run += 1
        else:
            gap = max(gap, run)
            run = 0
    gap_frac = float(gap / max(1, len(sup)))
    med_width = float(np.median(np.array(Wd)[vv])) if vv.sum() else 0.0
    mm_per_px = _rect_px_to_mm(corners_xyz, axis, W, H)
    width_mm = (float(med_width * mm_per_px)
                if mm_per_px is not None else None)
    max_dev = float(np.abs((m * t[sup] + b) - prior).max()) \
        if sup.sum() else float(halfwin + 1)
    info.update({"slope": round(float(m), 4),
                 "intercept_rect": round(float(b), 2),
                 "coverage": round(coverage, 3),
                 "max_gap_frac": round(gap_frac, 3),
                 "residual_px": round(resid_px, 2),
                 "median_width_rect": round(med_width, 1),
                 "width_mm": (round(width_mm, 1)
                              if width_mm is not None else None),
                 "max_dev_from_prior": round(max_dev, 1)})
    if coverage < min_cov:
        info["reject"] = "low_coverage"
        return None, 0.0, info
    if gap_frac > max_gap:
        info["reject"] = "gappy_support"
        return None, 0.0, info
    if resid_px > max_res:
        info["reject"] = "high_residual"
        return None, 0.0, info
    if max_dev > halfwin:
        info["reject"] = "line_left_window"
        return None, 0.0, info
    # A valid metric scale is required: without corners_xyz/plane the
    # tape width cannot be checked in mm, so tape cannot be confirmed.
    if width_mm is None:
        info["reject"] = "no_metric_scale"
        return None, 0.0, info
    if not (wmin <= width_mm <= wmax):
        info["reject"] = "implausible_tape_width"
        return None, 0.0, info

    def _back(pt):
        p = Mi @ np.array([pt[0], pt[1], 1.0])
        return p[:2] / p[2]

    if axis == "h":
        e1, e2 = _back([0.0, b]), _back([float(W), m * W + b])
    else:
        e1, e2 = _back([b, 0.0]), _back([m * H + b, float(H)])
    eps = np.array(sorted([e1, e2], key=lambda p: p[0]))
    score = float(np.clip(coverage, 0.0, 1.0)
                  * np.clip(1.0 - resid_px / (max_res + 1.0), 0.0, 1.0))
    info["score"] = round(score, 3)
    return eps, score, info

def _named(corners) -> dict:
    q = _as_quad(corners)
    return {name: [round(float(q[i][0]), 2), round(float(q[i][1]), 2)]
            for i, name in enumerate(CORNER_NAMES)}


GEOM_AXIS_DEFAULTS = {
    "enabled": True,
    "small_long_mm": _profiles.PROFILES["small"].r10_long_mm,
    "small_short_mm": _profiles.PROFILES["small"].r10_short_mm,
    "min_ratio": 1.05,
    "min_diff_mm": 5.0,
    "opposite_tol_mm": 3.0,
}


def geometry_class_axis(corners_xyz, est_cfg: dict | None = None,
                        profile: str | None = None
                        ) -> tuple[str | None, dict]:
    """R10: seam axis from the SMALL box top-face geometry.

    The small box's flap gap runs parallel to the SHORT side, joining the
    midpoints of the two LONG sides. Returns ``(axis | None, info)`` where
    axis is ``"lr"`` (seam joins the BL-FL and BR-FR sides, i.e. those are
    the long sides) or ``"bf"``. None unless ``corners_xyz`` is valid, the
    quad is rectangular (opposite 3-D sides within ``opposite_tol_mm``),
    the long/short side lengths are in the small-class ranges, and
    long/short >= ``min_ratio`` and long - short >= ``min_diff_mm``.

    ``profile`` (box profile name, optional): R10 is a construction fact of
    the small box only, so a profile without ``geometry_axis_validated``
    (big, generic) vetoes the result AFTER the legacy computation (None +
    ``reason="profile_not_validated"``); None keeps the legacy behaviour."""
    axis, info = _geometry_class_axis(corners_xyz, est_cfg)
    if (axis is not None and profile is not None
            and profile in _profiles.PROFILES
            and not _profiles.PROFILES[profile].geometry_axis_validated):
        info["reason"] = "profile_not_validated"
        info["vetoed_axis"] = axis
        return None, info
    return axis, info


def _geometry_class_axis(corners_xyz, est_cfg: dict | None = None
                         ) -> tuple[str | None, dict]:
    g = dict(GEOM_AXIS_DEFAULTS)
    sub = (est_cfg or {}).get("geometry_axis") \
        if isinstance(est_cfg, dict) else None
    if isinstance(sub, dict):
        g.update(sub)
    info: dict = {}
    if not g.get("enabled", True):
        info["reason"] = "disabled"
        return None, info
    X = _xyz_quad(corners_xyz) if corners_xyz is not None else None
    if X is None:
        info["reason"] = "no_corners_xyz"
        return None, info
    BL, BR, FR, FL = X
    d = lambda a, b: float(np.linalg.norm(a - b)) * 1000.0  # noqa: E731
    a1, a2 = d(BL, BR), d(FL, FR)      # sides along "lr" direction
    b1, b2 = d(BL, FL), d(BR, FR)      # sides the "lr" seam joins
    info["sides_mm"] = [round(a1, 1), round(a2, 1), round(b1, 1),
                        round(b2, 1)]
    tol = float(g["opposite_tol_mm"])
    if abs(a1 - a2) > tol or abs(b1 - b2) > tol:
        info["reason"] = "not_rectangular"
        return None, info
    a, b = (a1 + a2) / 2.0, (b1 + b2) / 2.0
    long_, short_ = max(a, b), min(a, b)
    info["long_short_mm"] = [round(long_, 1), round(short_, 1)]
    lo, hi = g["small_long_mm"]
    slo, shi = g["small_short_mm"]
    if not (lo <= long_ <= hi and slo <= short_ <= shi):
        info["reason"] = "not_small_class"
        return None, info
    if short_ <= 0 or long_ / short_ < float(g["min_ratio"]) \
            or long_ - short_ < float(g["min_diff_mm"]):
        info["reason"] = "near_square"
        return None, info
    return ("lr" if b > a else "bf"), info


def estimate_seam(bgr: np.ndarray, corners,
                  sam3_bonus: tuple[float, float] | None = None,
                  axis_margin: float | None = None,
                  axis_min: float | None = None,
                  est_cfg: dict | None = None,
                  corners_xyz=None,
                  profile: str | None = None) -> dict:
    """Estimate the flap-gap seam (see :func:`_estimate_seam_core`).

    ``profile``: box profile name (None = legacy behaviour, today's two
    boxes). It gates R10 (geometry_axis_validated only) and the view shift
    (generic: 0).

    Adds ``axis_source`` (``"evidence"`` normally; ``"geometry_class"``
    when the tape evidence was ambiguous and R10 chose the axis from the
    small-box top-face geometry). A geometry-axis result is NEVER ``ok``:
    it is demoted to ``estimated`` (source ``prior``), and carries
    ``seam_axis_from_geometry=True``."""
    res = _estimate_seam_core(bgr, corners, sam3_bonus, axis_margin,
                              axis_min, est_cfg, corners_xyz, profile)
    src = res.get("axis_source")
    if src == "geometry_class":
        res["seam_axis_from_geometry"] = True
        if res.get("status") == "ok":
            res["confirm_reject"] = "geometry_axis_never_ok"
            res["refined_endpoints_px"] = None
            res["refine_score"] = 0.0
            res.update({"source": "prior", "status": "estimated"})
            # publish the prior, never an observed/darkline line (refine /
            # darkline modes put those in estimate_endpoints_px on ok)
            if isinstance(res.get("estimate"), dict) and res.get("prior_endpoints_px"):
                res["estimate"]["estimate_endpoints_px"] = dict(res["prior_endpoints_px"])
        if isinstance(res.get("estimate"), dict):
            res["estimate"]["axis_source"] = "geometry_class"
    elif src is None and res.get("axis") is not None:
        res["axis_source"] = "evidence"
    if res.get("axis") is not None and isinstance(res.get("estimate"), dict):
        res["estimate"].setdefault("axis_source", res.get("axis_source"))
    return res


def _estimate_seam_core(bgr: np.ndarray, corners,
                        sam3_bonus: tuple[float, float] | None = None,
                        axis_margin: float | None = None,
                        axis_min: float | None = None,
                        est_cfg: dict | None = None,
                        corners_xyz=None,
                        profile: str | None = None) -> dict:
    """Estimate the flap-gap seam; returns the contract dict.

    Mode (``seam_mode`` in est_cfg, default ``"confirm"``):
    - ``confirm`` (default): the published line is ALWAYS the projective
      prior on the evidence-chosen axis. ``ok`` (source
      ``prior_confirmed_by_gap``) requires the observed gap line to agree
      with the prior (offset <= ``confirm_max_offset_mm`` at 25/50/75%,
      angle <= ``confirm_max_angle_deg``) with sufficient coverage; else
      ``estimated`` (source ``prior``). Endpoints = prior ∩ face edges.
    - ``refine``: legacy comparison mode; ``ok`` publishes the observed
      gap line itself (source ``observed_gap``).
    - ``darkline``: user method as publisher; stations along the prior,
      darkest-contrast perpendicular points, Huber fit; ``ok`` publishes
      the darkline fit itself (source ``observed_darkline``).
    - ``darkline_confirm``: darkline as confirmation of the prior; ``ok``
      publishes the PRIOR (source ``prior_confirmed_by_darkline``) iff
      the darkline fit agrees within the confirm gates; else
      ``estimated`` (source ``prior``).
    - ``either_confirm``: prior published + ok if EITHER the gap-valley
      fit OR the darkline fit confirms the prior within the confirm
      gates (source records which agreed); else ``estimated``.
    ``ambiguous`` means the axis itself is unclear. Observed lines and
    their stats are always kept in ``estimate`` for audit.

    Viewing shift (all confirm modes): the observed gap/dark line is
    compared with the prior translated by ``expected_shift_mm`` (see
    :func:`expected_view_shift`; + = away from the camera along the top
    plane), and the gap/darkline search windows are centred there
    (``view_shift.center_windows``). The published endpoints are still
    the UNSHIFTED prior (the physical seam is centred). ``estimate``
    records ``view_shift``, ``expected_shift_mm``, the signed observed
    offsets vs the prior (``observed_offset_mm`` at mid-face,
    ``observed_offsets`` at 25/50/75%) and the unshifted agreements.
    Degenerate ``corners_xyz`` -> shift 0 (plain-prior comparison).
    """
    if axis_margin is None:
        axis_margin = _est_cfg(est_cfg, "axis_min_margin_ratio",
                               AXIS_MIN_MARGIN_RATIO)
    if axis_min is None:
        axis_min = _est_cfg(est_cfg, "axis_min_evidence", AXIS_MIN_EVIDENCE)
    priors = projective_midlines(corners)
    M, _, W, H = rectify_face(bgr, corners)
    sat = _face_sat(bgr, M, W, H)
    gray = _rect_gray(bgr, M, W, H)
    ev = {}
    for cand in ("lr", "bf"):
        ev[cand] = axis_evidence(bgr, corners, cand, _sat=sat,
                                 _shape=(W, H))
    if sam3_bonus is not None:
        ev["lr"] += float(sam3_bonus[0])
        ev["bf"] += float(sam3_bonus[1])
    _gd = _est_cfg(est_cfg, "gap_min_depth", GAP_MIN_DEPTH)
    _gs = _est_cfg(est_cfg, "gap_min_shoulder", GAP_MIN_SHOULDER)
    _gw = int(_est_cfg(est_cfg, "gap_max_width_px", GAP_MAX_WIDTH_PX))
    _hwax = _gap_halfwin(W, H, est_cfg)
    gap_sup = {}
    for _cand, _ra in (("lr", "h"), ("bf", "v")):
        _prc = H // 2 if _ra == "h" else W // 2
        gap_sup[_cand] = _gap_support_for_axis(gray, _ra, _prc, _hwax,
                                               _gd, _gs, _gw)
    cands = []
    for cand in ("lr", "bf"):
        eps = priors[cand]["endpoints"]
        sl, sr = eps[0], eps[1]
        cands.append({"axis": cand,
                      "endpoints_px": {"seam_left": [float(sl[0]), float(sl[1])],
                                       "seam_right": [float(sr[0]), float(sr[1])]},
                      "evidence": round(float(ev[cand]), 3),
                      "gap_support": round(float(gap_sup[cand]), 3)})
    order = sorted(cands, key=lambda c: c["evidence"], reverse=True)
    best, second = order[0], order[1]
    margin = ((best["evidence"] - second["evidence"])
              / best["evidence"]) if best["evidence"] > 0 else 0.0
    axis_clear = bool(best["evidence"] >= axis_min
                      and margin >= axis_margin)
    geom_info = None
    if not axis_clear:
        g_axis, geom_info = geometry_class_axis(corners_xyz, est_cfg, profile)
        if g_axis is None:
            return {"method": "projective_midline", "axis": None,
                    "candidates": cands, "prior_endpoints_px": None,
                    "estimate": None,
                    "refined_endpoints_px": None, "refine_score": None,
                    "observed_line": None, "coverage": None,
                    "residual_px": None,
                    "source": None, "status": "ambiguous",
                    "axis_source": None, "geometry_axis": geom_info}
        best = next(c for c in cands if c["axis"] == g_axis)
        geom_info["axis"] = g_axis
    axis = best["axis"]
    prior_eps = np.array([best["endpoints_px"]["seam_left"],
                          best["endpoints_px"]["seam_right"]])
    # Full-precision prior ∩ face edges. Published ONCE at 0.1 px by the
    # pipeline's _rounded_eps (never pre-rounded here: a 0.01 px staging
    # round plus a 0.1 px publish round disagrees with a single 0.1 px
    # round on ~9% of endpoints). Audit copies below keep 0.1 px.
    prior_full = {"seam_left": [float(prior_eps[0][0]),
                                float(prior_eps[0][1])],
                  "seam_right": [float(prior_eps[1][0]),
                                 float(prior_eps[1][1])]}
    prior_block = {
        "seam_left": [round(float(prior_eps[0][0]), 1),
                      round(float(prior_eps[0][1]), 1)],
        "seam_right": [round(float(prior_eps[1][0]), 1),
                       round(float(prior_eps[1][1]), 1)]}
    # Viewing shift: the observed gap / dark line is expected at prior +
    # s (mm along the plane, + = away from the camera). The PUBLISHED
    # line stays the prior; s only moves the confirmation target (and
    # centres the gap/darkline search windows on it).
    mmpp = _rect_px_to_mm(corners_xyz, "h" if axis == "lr" else "v",
                          W, H)
    vshift = expected_view_shift(corners_xyz, axis, est_cfg, profile)
    s_mm = float(vshift.get("expected_shift_mm") or 0.0)
    away = vshift.get("away_rect_sign")
    shift_rect = (float(away) * s_mm / float(mmpp)
                  if away and mmpp is not None and mmpp > 1e-9
                  and np.isfinite(mmpp) else 0.0)
    if shift_rect == 0.0:
        s_mm = 0.0
    vshift["expected_shift_mm"] = round(s_mm, 3)
    vshift["expected_shift_rect_px"] = round(float(shift_rect), 2)
    _vcfg = (est_cfg or {}).get("view_shift") \
        if isinstance(est_cfg, dict) else None
    center_windows = bool((_vcfg or {}).get("center_windows", True)) \
        if isinstance(_vcfg, dict) else True
    vshift["center_windows"] = center_windows
    win_off = shift_rect if center_windows else 0.0
    _prof = _profiles.PROFILES.get(profile) if profile else None
    publish_obs = bool(_prof is not None and _prof.seam_publish == "observed")
    gap_hw = None
    if publish_obs and mmpp is not None and mmpp > 1e-9 and np.isfinite(mmpp):
        # the search window must cover at least +-seam_gap_window_mm around the midline
        gap_hw = max(_gap_halfwin(W, H, est_cfg),
                     int(np.ceil(float(_prof.seam_gap_window_mm) / float(mmpp))))
    observed, score, info = refine_on_gap(
        bgr, corners, axis, halfwin_px=gap_hw, est_cfg=est_cfg,
        corners_xyz=corners_xyz, center_offset_rect=win_off,
        keep_slices=publish_obs)
    tape_eps, tape_score, tape_info = refine_on_tape(
        bgr, corners, axis, est_cfg=est_cfg, corners_xyz=corners_xyz)
    dark_eps, dark_score, dark_info = refine_on_darkline(
        bgr, corners, axis, est_cfg=est_cfg, corners_xyz=corners_xyz,
        center_offset_rect=win_off, keep_slices=publish_obs)
    obs = ({"seam_left": [round(float(observed[0][0]), 2),
                           round(float(observed[0][1]), 2)],
            "seam_right": [round(float(observed[1][0]), 2),
                           round(float(observed[1][1]), 2)]}
           if observed is not None else None)
    dark = ({"seam_left": [round(float(dark_eps[0][0]), 2),
                           round(float(dark_eps[0][1]), 2)],
            "seam_right": [round(float(dark_eps[1][0]), 2),
                           round(float(dark_eps[1][1]), 2)]}
           if dark_eps is not None else None)
    est_block = {"axis": axis, "prior_endpoints_px": dict(prior_block),
                 **({"geometry_axis": geom_info}
                    if geom_info is not None else {}),
                 "gap_support": {k: round(float(v), 3)
                                 for k, v in gap_sup.items()},
                 "coverage": info.get("coverage"),
                 "residual_px": info.get("residual_px"),
                 "median_gap_width_mm": info.get("width_mm"),
                 "median_gap_width_rect": info.get("median_width_rect"),
                 "per_slice_support": {
                     "n_slices": info.get("n_slices"),
                     "n_slices_total": info.get("n_slices_total"),
                     "n_inliers": info.get("n_inliers"),
                     "max_gap_frac": info.get("max_gap_frac")},
                 "tape": {"score": round(float(tape_score), 3),
                          "width_mm": tape_info.get("width_mm"),
                          "coverage": tape_info.get("coverage"),
                          "residual_px": tape_info.get("residual_px"),
                          "reject": tape_info.get("reject")},
                 "darkline": {"endpoints_px": dark,
                              "score": round(float(dark_score), 3),
                              "coverage": dark_info.get("coverage"),
                              "residual_px": dark_info.get("residual_px"),
                              "median_contrast":
                                  dark_info.get("median_contrast"),
                              "n_inliers": dark_info.get("n_inliers"),
                              "max_gap_frac":
                                  dark_info.get("max_gap_frac"),
                              "reject": dark_info.get("reject")},
                 "gap": {k: (round(float(v), 4)
                             if isinstance(v, float) else v)
                         for k, v in info.items()
                         if not isinstance(v, list)}}
    result = {"method": "projective_midline", "axis": axis,
              "axis_source": ("geometry_class" if geom_info is not None
                              else "evidence"),
              "candidates": cands,
              "prior_endpoints_px": dict(prior_block),
              "estimate": dict(est_block),
              "observed_line": obs,
              "coverage": info.get("coverage"),
              "residual_px": info.get("residual_px"),
              "refine_info": {k: (round(float(v), 4)
                                  if isinstance(v, float) else v)
                              for k, v in info.items()
                              if not isinstance(v, list)}}
    if observed is None:
        _mode_none = True
    else:
        _mode_none = False
    mode = ((est_cfg or {}).get("seam_mode", "confirm")
            if isinstance(est_cfg, dict) else "confirm")
    mode = str(mode).lower() if mode is not None else "confirm"
    if mode not in ("confirm", "refine", "darkline", "darkline_confirm",
                    "either_confirm"):
        raise ValueError(
            "seam_mode must be 'confirm', 'refine', 'darkline', "
            "'darkline_confirm' or 'either_confirm'")
    max_off = _est_cfg(est_cfg, "confirm_max_offset_mm",
                        CONFIRM_MAX_OFFSET_MM)
    if profile in _profiles.PROFILES and _profiles.PROFILES[profile].own_constants:
        # Generic box / later profile (large_392): no validated view-shift
        # model, so the confirmation tolerance comes from the profile (small/big
        # keep the config value above: bit-identical). Still the evidence path
        # (axis_evidence + either_confirm); no ``observed_tape`` publisher
        # yet (waits for real tape data, docs/multi_box_plan.md M4).
        _gp = _profiles.PROFILES[profile].confirm_max_offset_mm
        if _gp is not None:
            max_off = float(_gp)
    max_ang = _est_cfg(est_cfg, "confirm_max_angle_deg",
                        CONFIRM_MAX_ANGLE_DEG)
    min_cov = _est_cfg(est_cfg, "confirm_min_coverage",
                        CONFIRM_MIN_COVERAGE)
    prior_rect = _eps_to_rect(prior_block, M)
    # Confirmation target = prior shifted by the expected viewing shift.
    target_rect = (_shift_rect_line(prior_rect, axis, shift_rect)
                   if prior_rect is not None else None)
    gap_rect = (_eps_to_rect(obs, M) if obs is not None else None)
    dark_rect = (_eps_to_rect(dark, M) if dark is not None else None)
    agreement = (_line_agreement_rect(target_rect, gap_rect, mmpp)
                 if target_rect is not None and gap_rect is not None
                 else None)
    _away = int(away) if away else 0
    observed_offsets = {}
    for _k, _r in (("gap", gap_rect), ("darkline", dark_rect)):
        observed_offsets[_k] = (
            _signed_offsets_rect(prior_rect, _r, axis, _away, mmpp)
            if prior_rect is not None and _r is not None and _away
            else None)
    exp_line = None
    if target_rect is not None and shift_rect != 0.0:
        try:
            Mi_ = np.linalg.inv(np.asarray(M, dtype=np.float64))
            _pts = []
            for _p in target_rect:
                _q = Mi_ @ np.array([_p[0], _p[1], 1.0])
                _pts.append(_q[:2] / _q[2])
            _pts = sorted(_pts, key=lambda p: p[0])
            exp_line = {"seam_left": [round(float(_pts[0][0]), 1),
                                      round(float(_pts[0][1]), 1)],
                        "seam_right": [round(float(_pts[1][0]), 1),
                                       round(float(_pts[1][1]), 1)]}
        except (np.linalg.LinAlgError, ValueError, TypeError):
            exp_line = None
    vshift["expected_line_endpoints_px"] = exp_line
    est_block["view_shift"] = dict(vshift)
    est_block["expected_shift_mm"] = vshift["expected_shift_mm"]
    est_block["observed_offset_mm"] = {
        k: (v or {}).get("mid_mm") for k, v in observed_offsets.items()}
    est_block["observed_offsets"] = observed_offsets
    est_block["confirm_target"] = ("prior_plus_expected_shift"
                                   if shift_rect != 0.0 else "prior")
    est_block["agreement"] = agreement
    est_block["agreement_unshifted"] = (
        _line_agreement_rect(prior_rect, gap_rect, mmpp)
        if prior_rect is not None and gap_rect is not None else None)
    est_block["mm_per_px_across"] = (round(float(mmpp), 4)
                                     if mmpp is not None else None)
    result["agreement"] = agreement
    result["expected_shift_mm"] = est_block["expected_shift_mm"]
    result["observed_offset_mm"] = dict(est_block["observed_offset_mm"])
    if publish_obs:
        return _publish_observed(
            result, est_block, _prof, M, axis, mmpp, prior_block,
            observed, score, info, obs, dark_eps, dark_score, dark_info, dark,
            agreement, dark_rect, target_rect, max_ang)
    if mode == "refine":
        if _mode_none:
            result.update({"refined_endpoints_px": None,
                           "refine_score": 0.0,
                           "source": "prior", "status": "estimated"})
            est_block["estimate_endpoints_px"] = dict(prior_block)
            result["estimate"] = dict(est_block)
        else:
            result.update(
                {"refined_endpoints_px": obs,
                 "refine_score": round(float(score), 3),
                 "source": "observed_gap", "status": "ok"})
            est_block["estimate_endpoints_px"] = dict(obs)
            result["estimate"] = dict(est_block)
        return result
    if mode == "darkline":
        if dark_eps is None:
            result.update({"refined_endpoints_px": None,
                           "refine_score": 0.0,
                           "source": "prior", "status": "estimated"})
            est_block["estimate_endpoints_px"] = dict(prior_block)
            result["estimate"] = dict(est_block)
        else:
            result.update(
                {"refined_endpoints_px": dark,
                 "refine_score": round(float(dark_score), 3),
                 "source": "observed_darkline", "status": "ok"})
            est_block["estimate_endpoints_px"] = dict(dark)
            result["estimate"] = dict(est_block)
        return result
    if mode == "darkline_confirm":
        # Darkline as confirmation of the prior: ok publishes the PRIOR
        # (source prior_confirmed_by_darkline) iff the darkline fit
        # agrees with it within the confirm gates; else estimated.
        dark_agreement = (
            _line_agreement_rect(target_rect, dark_rect, mmpp)
            if target_rect is not None and dark_rect is not None
            else None)
        est_block["darkline_agreement"] = dark_agreement
        est_block["darkline_agreement_unshifted"] = (
            _line_agreement_rect(prior_rect, dark_rect, mmpp)
            if prior_rect is not None and dark_rect is not None
            else None)
        result["darkline_agreement"] = dark_agreement
        est_block["estimate_endpoints_px"] = dict(prior_block)
        est_block["confirm_gates"] = {
            "min_coverage": round(float(min_cov), 3),
            "max_offset_mm": round(float(max_off), 3),
            "max_angle_deg": round(float(max_ang), 3),
            "expected_shift_mm": est_block["expected_shift_mm"]}
        result["estimate"] = dict(est_block)
        result["refined_endpoints_px"] = dict(prior_full)
        result["refine_score"] = round(float(dark_score), 3) \
            if dark_eps is not None else 0.0
        dcov = dark_info.get("coverage")
        if dark_eps is None:
            result["confirm_reject"] = "no_darkline"
            result["refined_endpoints_px"] = None
            result["refine_score"] = 0.0
            result.update({"source": "prior", "status": "estimated"})
            return result
        if dcov is None or dcov < min_cov:
            result["confirm_reject"] = (f"low_darkline_coverage({dcov})"
                                        if dcov is not None
                                        else "low_darkline_coverage(?)")
            result["refined_endpoints_px"] = None
            result["refine_score"] = 0.0
            result.update({"source": "prior", "status": "estimated"})
            return result
        if dark_agreement is None or \
                dark_agreement.get("max_offset_mm") is None:
            result["confirm_reject"] = "darkline_disagrees(no_mm_scale)"
            result["refined_endpoints_px"] = None
            result["refine_score"] = 0.0
            result.update({"source": "prior", "status": "estimated"})
            return result
        if dark_agreement["max_offset_mm"] > max_off or \
                dark_agreement["angle_deg"] > max_ang:
            result["confirm_reject"] = (
                f"darkline_disagrees({dark_agreement['max_offset_mm']}mm,"
                f"{dark_agreement['angle_deg']}deg)")
            result["refined_endpoints_px"] = None
            result["refine_score"] = 0.0
            result.update({"source": "prior", "status": "estimated"})
            return result
        result.update({"source": "prior_confirmed_by_darkline",
                       "status": "ok"})
        return result
    if mode == "either_confirm":
        # Prior published + ok if EITHER the gap-valley fit OR the
        # darkline fit confirms the prior within the confirm gates
        # (same coverage + offset/angle gates, darkline coverage from
        # its own fit). First agreement wins for the source label;
        # both agreements persist in the estimate block for audit.
        dark_agreement = (
            _line_agreement_rect(target_rect, dark_rect, mmpp)
            if target_rect is not None and dark_rect is not None
            else None)
        est_block["darkline_agreement"] = dark_agreement
        est_block["darkline_agreement_unshifted"] = (
            _line_agreement_rect(prior_rect, dark_rect, mmpp)
            if prior_rect is not None and dark_rect is not None
            else None)
        result["darkline_agreement"] = dark_agreement
        est_block["estimate_endpoints_px"] = dict(prior_block)
        est_block["confirm_gates"] = {
            "min_coverage": round(float(min_cov), 3),
            "max_offset_mm": round(float(max_off), 3),
            "max_angle_deg": round(float(max_ang), 3),
            "expected_shift_mm": est_block["expected_shift_mm"]}
        result["estimate"] = dict(est_block)
        result["refined_endpoints_px"] = dict(prior_full)
        _scores = [s for s in (score, dark_score) if s is not None]
        result["refine_score"] = round(float(max(_scores)), 3) \
            if _scores and (observed is not None or dark_eps is not None) \
            else 0.0
        gap_ok = (observed is not None
                  and (info.get("coverage") or 0) >= min_cov
                  and agreement is not None
                  and agreement.get("max_offset_mm") is not None
                  and agreement["max_offset_mm"] <= max_off
                  and agreement["angle_deg"] <= max_ang)
        dark_ok = (dark_eps is not None
                   and (dark_info.get("coverage") or 0) >= min_cov
                   and dark_agreement is not None
                   and dark_agreement.get("max_offset_mm") is not None
                   and dark_agreement["max_offset_mm"] <= max_off
                   and dark_agreement["angle_deg"] <= max_ang)
        if gap_ok or dark_ok:
            result.update({"source": (
                "prior_confirmed_by_gap" if gap_ok
                else "prior_confirmed_by_darkline"), "status": "ok"})
            return result
        why = []
        if observed is None:
            why.append(f"gap:{info.get('reject') or 'no_fit'}")
        elif (info.get("coverage") or 0) < min_cov:
            why.append(f"gap:low_coverage({info.get('coverage')})")
        elif agreement is None or \
                agreement.get("max_offset_mm") is None:
            why.append("gap:no_mm_scale")
        else:
            why.append(f"gap:disagrees({agreement['max_offset_mm']}mm,"
                       f"{agreement['angle_deg']}deg)")
        if dark_eps is None:
            why.append(f"darkline:{dark_info.get('reject') or 'no_fit'}")
        elif (dark_info.get("coverage") or 0) < min_cov:
            why.append(
                f"darkline:low_coverage({dark_info.get('coverage')})")
        elif dark_agreement is None or \
                dark_agreement.get("max_offset_mm") is None:
            why.append("darkline:no_mm_scale")
        else:
            why.append(
                f"darkline:disagrees({dark_agreement['max_offset_mm']}mm,"
                f"{dark_agreement['angle_deg']}deg)")
        result["confirm_reject"] = "either_disagrees(" + ";".join(why) + ")"
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    cov = info.get("coverage")
    est_block["estimate_endpoints_px"] = dict(prior_block)
    est_block["confirm_gates"] = {
        "min_coverage": round(float(min_cov), 3),
        "max_offset_mm": round(float(max_off), 3),
        "max_angle_deg": round(float(max_ang), 3),
        "expected_shift_mm": est_block["expected_shift_mm"]}
    result["estimate"] = dict(est_block)
    if _mode_none:
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    result["refined_endpoints_px"] = dict(prior_full)
    result["refine_score"] = round(float(score), 3)
    if cov is None or cov < min_cov:
        result["confirm_reject"] = (f"low_gap_coverage({cov})"
                                    if cov is not None
                                    else "low_gap_coverage(?)")
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    if agreement is None or agreement.get("max_offset_mm") is None:
        result["confirm_reject"] = "gap_disagrees(no_mm_scale)"
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    if agreement["max_offset_mm"] > max_off or \
            agreement["angle_deg"] > max_ang:
        result["confirm_reject"] = (
            f"gap_disagrees({agreement['max_offset_mm']}mm,"
            f"{agreement['angle_deg']}deg)")
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    result.update({"source": "prior_confirmed_by_gap", "status": "ok"})
    return result

def gap_shape(slice_t, slice_c, line, mmpp, smooth: int = 7) -> dict | None:
    """Shape of an observed gap against its published straight line.

    ``slice_t`` / ``slice_c``: per-slice along / across rect coordinates,
    ``line`` = (slope, intercept) of ``c = m t + b``. Residuals (mm, perpendicular,
    + = larger rect across coordinate) are median-filtered over ``smooth`` neighbouring
    slices (single-slice noise is not curvature); ``curvature_mm`` = max |smoothed residual|,
    ``offset_median_mm`` / ``offset_max_mm`` = median / max |raw residual| of the points
    against the line. None without a metric scale or points.
    """
    if mmpp is None or not np.isfinite(mmpp) or len(slice_t) < 4:
        return None
    t = np.asarray(slice_t, float)
    c = np.asarray(slice_c, float)
    order = np.argsort(t)
    t, c = t[order], c[order]
    m, b = line
    k = float(mmpp) / float(np.sqrt(1.0 + m * m))
    r = (c - (m * t + b)) * k
    h = smooth // 2
    sm = np.array([np.median(r[max(0, i - h):i + h + 1]) for i in range(len(r))])
    return {"curvature_mm": round(float(np.max(np.abs(sm))), 3),
            "offset_median_mm": round(float(np.median(r)), 3),
            "offset_max_mm": round(float(np.max(np.abs(r))), 3),
            "n_points": int(len(r))}


def _publish_observed(result, est_block, prof, M, axis, mmpp, prior_block,
                      observed, score, info, obs, dark_eps, dark_score,
                      dark_info, dark, agreement, dark_rect, target_rect,
                      max_ang) -> dict:
    """Seam publisher of profiles with ``seam_publish == "observed"``.

    Publishes the straight line fitted to the observed flap-gap valley (dark line
    only when the gap is not found). ``ok`` needs coverage >= ``seam_min_obs_coverage``
    of the slices, a fitted line inside the search window and within ``max_ang`` of the
    midline (the existing gap plausibility checks - width, residual, gaps - are already
    part of the fit); else ``estimated`` with the midline as estimate-only endpoints.
    Records the per-slice gap points and ``seam_curvature_mm``.
    """
    min_cov = float(prof.seam_min_obs_coverage)
    win = float(prof.seam_gap_window_mm)
    dark_agreement = (_line_agreement_rect(target_rect, dark_rect, mmpp)
                      if target_rect is not None and dark_rect is not None else None)
    est_block["darkline_agreement"] = dark_agreement
    result["darkline_agreement"] = dark_agreement

    def _usable(eps, inf, agr):
        if eps is None:
            return f"{inf.get('reject') or 'no_fit'}"
        if (inf.get("coverage") or 0) < min_cov:
            return f"low_coverage({inf.get('coverage')}<{min_cov})"
        if agr is None or agr.get("max_offset_mm") is None:
            return "no_mm_scale"
        if agr["max_offset_mm"] > win:
            return f"outside_window({agr['max_offset_mm']}mm)"
        if agr["angle_deg"] > max_ang:
            return f"angle({agr['angle_deg']}deg)"
        return None

    why_gap = _usable(observed, info, agreement)
    why_dark = _usable(dark_eps, dark_info, dark_agreement)
    est_block["publish"] = "observed"
    est_block["publish_gates"] = {"min_coverage": round(min_cov, 3),
                                  "window_mm": round(win, 3),
                                  "max_angle_deg": round(float(max_ang), 3)}
    if why_gap is None:
        src, eps_pub, eps_obs, sc, inf = "observed_gap", observed, obs, score, info
    elif why_dark is None and observed is None:
        # dark line only as the fallback when the gap valley is not found at all
        src, eps_pub, eps_obs, sc, inf = ("observed_darkline", dark_eps, dark,
                                          dark_score, dark_info)
    else:
        src = None
    shape = None
    if src is not None and "_slice_t" in inf:
        shape = gap_shape(inf["_slice_t"], inf["_slice_c"], inf["_line"], mmpp)
    if src is None:
        # report the shape of whatever line was fitted (audit), publish the midline unconfirmed
        if observed is not None and "_slice_t" in info:
            shape = gap_shape(info["_slice_t"], info["_slice_c"], info["_line"], mmpp)
        est_block["estimate_endpoints_px"] = dict(prior_block)
        est_block["seam_curvature_mm"] = (shape or {}).get("curvature_mm")
        est_block["gap_shape"] = shape
        result["seam_curvature_mm"] = est_block["seam_curvature_mm"]
        result["estimate"] = dict(est_block)
        result["confirm_reject"] = f"observed_unusable(gap:{why_gap};darkline:{why_dark})"
        result["refined_endpoints_px"] = None
        result["refine_score"] = 0.0
        result.update({"source": "prior", "status": "estimated"})
        return result
    est_block["estimate_endpoints_px"] = {k: list(v) for k, v in eps_obs.items()}
    est_block["published_line"] = src
    est_block["seam_curvature_mm"] = (shape or {}).get("curvature_mm")
    est_block["gap_shape"] = shape
    try:
        Mi = np.linalg.inv(np.asarray(M, dtype=np.float64))
        pts = []
        for t_, c_ in zip(inf["_slice_t"], inf["_slice_c"]):
            xy = (t_, c_) if axis == "lr" else (c_, t_)
            q = Mi @ np.array([xy[0], xy[1], 1.0])
            pts.append([round(float(q[0] / q[2]), 1), round(float(q[1] / q[2]), 1)])
        est_block["observed_points_px"] = pts
    except (np.linalg.LinAlgError, ValueError, TypeError, KeyError):
        est_block["observed_points_px"] = None
    result["seam_curvature_mm"] = est_block["seam_curvature_mm"]
    result["estimate"] = dict(est_block)
    result["refined_endpoints_px"] = {
        "seam_left": [float(eps_pub[0][0]), float(eps_pub[0][1])],
        "seam_right": [float(eps_pub[1][0]), float(eps_pub[1][1])]}
    result["refine_score"] = round(float(sc), 3)
    result.update({"source": src, "status": "ok"})
    return result


# ---- confirm mode: prior confirmed by gap ----------------------------------

# Confirm-mode agreement gates are defined next to the other gates at
# the top of this module (CONFIRM_*); _line_agreement computes them.


def _eps_to_rect(eps, M) -> tuple | None:
    """Map {seam_left, seam_right} image endpoints to rect coordinates."""
    try:
        pts = []
        for k in ("seam_left", "seam_right"):
            q = np.asarray(M, dtype=np.float64) @ np.array(
                [float(eps[k][0]), float(eps[k][1]), 1.0])
            pts.append(q[:2] / q[2])
        return (tuple(map(float, pts[0])), tuple(map(float, pts[1])))
    except (TypeError, KeyError, ValueError, IndexError):
        return None


def _line_agreement_rect(prior_rect, gap_rect, mm_per_px) -> dict | None:
    """Agreement of the observed gap line with the prior line in RECTIFIED
    coordinates (the gap fit and its width/residual gates all live there).

    Perpendicular offset of the PRIOR line sampled at 25/50/75% of its
    rect span from the gap line (rect px and mm via ``mm_per_px``, the
    across-seam top-plane scale) plus the acute angle. Returns None when
    either endpoint pair is unusable. Image-space distances are NOT mixed
    with the rect scale here: under an anisotropic homography the two
    differ (reviewer example: stored 1.492 mm image-scale vs 1.691 mm
    rectified on frame_000001_20260909_062436_099339).
    """
    import math as _m
    try:
        (px0, py0), (px1, py1) = prior_rect
        (gx0, gy0), (gx1, gy1) = gap_rect
    except (TypeError, ValueError, IndexError):
        return None
    gdx, gdy = gx1 - gx0, gy1 - gy0
    pdx, pdy = px1 - px0, py1 - py0
    gn, pn = _m.hypot(gdx, gdy), _m.hypot(pdx, pdy)
    if gn < 1e-12 or pn < 1e-12:
        return None
    offs = []
    for f in (0.25, 0.5, 0.75):
        sx, sy = px0 + f * pdx, py0 + f * pdy
        offs.append(abs((sx - gx0) * gdy - (sy - gy0) * gdx) / gn)
    c = max(-1.0, min(1.0, (pdx * gdx + pdy * gdy) / (pn * gn)))
    ac = _m.degrees(_m.acos(c))
    ang = min(ac, 180.0 - ac)
    mm = ([round(float(o * mm_per_px), 3) for o in offs]
          if mm_per_px is not None and np.isfinite(mm_per_px) else None)
    return {"offset_px": [round(float(o), 2) for o in offs],
            "offset_mm": mm,
            "max_offset_px": round(float(max(offs)), 2),
            "max_offset_mm": (round(float(max(mm)), 3)
                              if mm is not None else None),
            "angle_deg": round(float(ang), 2)}


def _line_agreement(prior_eps, gap_eps, mm_per_px) -> dict | None:
    """Agreement of the observed gap line with the prior line.

    BACKWARD-COMPATIBLE shim: image-space perpendicular offsets scaled by
    ``mm_per_px``. WRONG under an anisotropic homography (mixes image px
    with the rectified-face scale); kept only so older recorded estimates
    still parse. New code MUST use ``_line_agreement_rect``.
    """
    import math as _m
    try:
        px0, py0 = float(prior_eps["seam_left"][0]), float(
            prior_eps["seam_left"][1])
        px1, py1 = float(prior_eps["seam_right"][0]), float(
            prior_eps["seam_right"][1])
        gx0, gy0 = float(gap_eps["seam_left"][0]), float(
            gap_eps["seam_left"][1])
        gx1, gy1 = float(gap_eps["seam_right"][0]), float(
            gap_eps["seam_right"][1])
    except (TypeError, KeyError, ValueError, IndexError):
        return None
    gdx, gdy = gx1 - gx0, gy1 - gy0
    pdx, pdy = px1 - px0, py1 - py0
    gn, pn = _m.hypot(gdx, gdy), _m.hypot(pdx, pdy)
    if gn < 1e-12 or pn < 1e-12:
        return None
    offs = []
    for f in (0.25, 0.5, 0.75):
        sx, sy = px0 + f * pdx, py0 + f * pdy
        offs.append(abs((sx - gx0) * gdy - (sy - gy0) * gdx) / gn)
    c = max(-1.0, min(1.0, (pdx * gdx + pdy * gdy) / (pn * gn)))
    ac = _m.degrees(_m.acos(c))
    ang = min(ac, 180.0 - ac)
    mm = ([round(float(o * mm_per_px), 3) for o in offs]
          if mm_per_px is not None and np.isfinite(mm_per_px) else None)
    return {"offset_px": [round(float(o), 2) for o in offs],
            "offset_mm": mm,
            "max_offset_px": round(float(max(offs)), 2),
            "max_offset_mm": (round(float(max(mm)), 3)
                              if mm is not None else None),
            "angle_deg": round(float(ang), 2)}


def _px_to_mm(corners_px, corners_xyz) -> float:
    px = np.asarray([[corners_px[n][0], corners_px[n][1]]
                     for n in CORNER_NAMES], float)
    xyz = np.asarray([corners_xyz[n] for n in CORNER_NAMES], float)
    s = []
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        dp = float(np.linalg.norm(px[j] - px[i]))
        dm = float(np.linalg.norm(xyz[j] - xyz[i])) * 1000.0
        if dp > 1e-9:
            s.append(dm / dp)
    return float(np.mean(s)) if s else float("nan")


def _line_of_eps(eps) -> tuple | None:
    """Unit-direction line through {seam_left, seam_right} endpoints."""
    try:
        ax, ay = float(eps["seam_left"][0]), float(eps["seam_left"][1])
        bx, by = float(eps["seam_right"][0]), float(eps["seam_right"][1])
    except (TypeError, KeyError, ValueError, IndexError):
        return None
    dx, dy = bx - ax, by - ay
    n = float((dx * dx + dy * dy) ** 0.5)
    if not np.isfinite(n) or n < 1e-12:
        return None
    return ((ax, ay), (dx / n, dy / n))


def _perp_line_err(pred_eps, gt_eps, fracs=(0.25, 0.5, 0.75)) -> dict | None:
    """Perpendicular distance of the human line from the predicted line.

    Sampled at ``fracs`` along the HUMAN line (endpoints uncertain along
    the edge, so the mid-face perpendicular offset is primary), plus the
    acute angle between the lines. Secondary: mean endpoint distance.
    """
    import math as _m
    try:
        hx0, hy0 = float(gt_eps["seam_left"][0]), float(
            gt_eps["seam_left"][1])
        hx1, hy1 = float(gt_eps["seam_right"][0]), float(
            gt_eps["seam_right"][1])
        px0, py0 = float(pred_eps["seam_left"][0]), float(
            pred_eps["seam_left"][1])
        px1, py1 = float(pred_eps["seam_right"][0]), float(
            pred_eps["seam_right"][1])
    except (TypeError, KeyError, ValueError, IndexError):
        return None
    pdx, pdy = px1 - px0, py1 - py0
    hdx, hdy = hx1 - hx0, hy1 - hy0
    pn, hn = _m.hypot(pdx, pdy), _m.hypot(hdx, hdy)
    if pn < 1e-12 or hn < 1e-12:
        return None
    perps = []
    for f in fracs:
        sx, sy = hx0 + f * hdx, hy0 + f * hdy
        perps.append(abs((sx - px0) * pdy - (sy - py0) * pdx) / pn)
    c = max(-1.0, min(1.0, (pdx * hdx + pdy * hdy) / (pn * hn)))
    ang = min(_m.degrees(_m.acos(c)), 180.0 - _m.degrees(_m.acos(c)))
    mean_ep = (float(np.linalg.norm([px0 - hx0, py0 - hy0]))
               + float(np.linalg.norm([px1 - hx1, py1 - hy1]))) / 2.0
    return {"perp_25_px": round(perps[0], 2),
            "perp_50_px": round(perps[1], 2),
            "perp_75_px": round(perps[2], 2),
            "perp_median_px": round(float(np.median(perps)), 2),
            "angle_deg": round(float(ang), 2),
            "endpoint_mean_px": round(float(mean_ep), 2)}


def _match_err(est, gt) -> tuple:
    """Mean endpoint distance (sorted by x) + perpendicular distance."""
    e = np.asarray([[est["seam_left"][0], est["seam_left"][1]],
                    [est["seam_right"][0], est["seam_right"][1]]], float)
    g = np.asarray([[gt["seam_left"][0], gt["seam_left"][1]],
                    [gt["seam_right"][0], gt["seam_right"][1]]], float)
    mean = float((np.linalg.norm(e[0] - g[0])
                  + np.linalg.norm(e[1] - g[1])) / 2.0)
    d = g[1] - g[0]
    ln = float(np.linalg.norm(d))
    perp = float(abs(d[0] * (e - g[0])[:, 1] - d[1] * (e - g[0])[:, 0]).mean()
                 / ln) if ln > 1e-9 else float("nan")
    return mean, perp


def _oracle_axis(corners, gt) -> str:
    pr = projective_midlines(np.asarray(corners, dtype=float))
    err = {}
    for ax in ("lr", "bf"):
        eps = pr[ax]["endpoints"]
        est = {"seam_left": [float(eps[0][0]), float(eps[0][1])],
               "seam_right": [float(eps[1][0]), float(eps[1][1])]}
        err[ax], _ = _match_err(est, gt)
    return min(err, key=lambda a: err[a])


def _summ(xs: list) -> dict:
    a = np.asarray([x for x in xs if np.isfinite(x)], float)
    if len(a) == 0:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 2),
            "p90": round(float(sorted(a)[min(len(a) - 1,
                                             int(0.9 * len(a)))]), 2),
            "max": round(float(a.max()), 2)}


def _draw_overlay(bgr, corners, gt, res) -> np.ndarray:
    ov = bgr.copy()
    q = np.asarray(corners, float).reshape(4, 2)
    cv2.polylines(ov, [q.astype(np.int32)], True, (255, 0, 0), 2)
    for key, col in (("prior_endpoints_px", (0, 255, 255)),
                     ("refined_endpoints_px", (0, 255, 0))):
        eps = res.get(key)
        if eps:
            a = (float(eps["seam_left"][0]), float(eps["seam_left"][1]))
            b = (float(eps["seam_right"][0]), float(eps["seam_right"][1]))
            cv2.line(ov, (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))), col, 2)
    if gt:
        a = (int(round(gt["seam_left"][0])), int(round(gt["seam_left"][1])))
        b = (int(round(gt["seam_right"][0])), int(round(gt["seam_right"][1])))
        cv2.line(ov, a, b, (0, 0, 255), 2)
    cv2.putText(ov, f"{res.get('axis')} {res.get('status')}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return ov


def run_eval(reviews_path: str, autolabels_dir: str, out_dir: str,
             corners_src: str = "reviewed", use_sam3: bool = False,
             python_bin: str = "/home/farhan/miniconda3/envs/book_grasping/bin/python",
             overlay_n: int = 6) -> dict:
    """Eval driver: per-capture estimate + error report (see task spec).

    Primary metric: perpendicular distance of the human seam LINE from the
    predicted line at 25/50/75% along the human line (px and mm via the
    top plane: ray/plane back-projection with the capture's camera_info),
    plus angle. Secondary: endpoint distance. Compares prior only vs the
    gap-refined line; the legacy tape-centreline fit is reported separately
    via ``tape_line`` so tape-following regressions show up.
    """
    reviews = json.load(open(reviews_path))["records"]
    out = Path(out_dir)
    (out / "captures").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    per, n_axis_ok, n_axis_tot = [], 0, 0
    prior_med, ref_med, tape_med = [], [], []
    prior_mm, ref_mm, tape_mm = [], [], []
    prior_ang, ref_ang, tape_ang = [], [], []
    prior_ep, ref_ep, tape_ep = [], [], []
    from collections import Counter
    status_counts = Counter()
    for cid, rec in sorted(reviews.items()):
        gt_status = rec.get("seam_status")
        gt_eps = rec.get("seam_endpoints_px")
        auto_p = Path(autolabels_dir) / f"{cid}.json"
        if not auto_p.is_file():
            continue
        auto = json.load(open(auto_p))
        if corners_src == "reviewed" and rec.get("corners_px"):
            cn = rec["corners_px"]
            corners = [cn[n] for n in CORNER_NAMES]
        else:
            tcp = (auto.get("top_face") or {}).get("corners_px")
            if not tcp:
                continue
            corners = [tcp[n] for n in CORNER_NAMES]
        img_p = (auto.get("image") or "").strip()
        if not img_p or not Path(img_p).is_file():
            rgb_p = Path(
                "/home/farhan/Igus_rebel/outputs/rgbd_captures/rgb") / f"{cid}.png"
            img_p = str(rgb_p) if rgb_p.is_file() else img_p
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bonus = None
        if use_sam3:
            sev = sam3_tape_evidence(str(img_p), python_bin)
            # bonus kept informational (see above): never applied.
        xyz = (auto.get("top_face") or {}).get("corners_xyz_camera_m")
        res = estimate_seam(bgr, corners, sam3_bonus=bonus, corners_xyz=xyz)
        row = {"capture_id": cid, "gt_seam_status": gt_status,
               "corners_source": ("reviewed" if corners_src == "reviewed"
                                  and rec.get("corners_px") else "autolabel"),
               "result": res}
        # Legacy tape-centreline fit on the WINNING axis (comparison only):
        # shows what the old tape-observed stage would have published.
        if res.get("axis") in ("lr", "bf"):
            try:
                _teps, _tsc, _tinfo = refine_on_tape(
                    bgr, corners, res["axis"], corners_xyz=xyz)
                if _teps is not None:
                    row["tape_line"] = {
                        "seam_left": [round(float(_teps[0][0]), 2),
                                      round(float(_teps[0][1]), 2)],
                        "seam_right": [round(float(_teps[1][0]), 2),
                                       round(float(_teps[1][1]), 2)]}
            except (ValueError, cv2.error):
                pass
        if gt_eps and gt_status == "ok":
            oracle = _oracle_axis(np.array(corners, float), gt_eps)
            row["oracle_axis"] = oracle
            if res["axis"] is not None:
                n_axis_tot += 1
                n_axis_ok += int(res["axis"] == oracle)
                row["axis_correct"] = bool(res["axis"] == oracle)
            xyz = (auto.get("top_face") or {}).get("corners_xyz_camera_m")
            mm_px = _px_to_mm(
                {n: corners[i] for i, n in enumerate(CORNER_NAMES)},
                xyz) if xyz and all(xyz.get(n) for n in CORNER_NAMES) \
                else float("nan")
            # Ray/plane mm scale for THIS capture (task spec): camera
            # rays through sampled points ∩ top plane from the stored
            # corners_xyz fit (plane normal·x + d = 0).
            plane_mm = None
            try:
                _pts = np.array(
                    [xyz[n] for n in CORNER_NAMES], float) \
                    if xyz and all(xyz.get(n) for n in CORNER_NAMES) \
                    else None
                if _pts is not None and np.all(np.isfinite(_pts)):
                    _e1 = _pts[1] - _pts[0]
                    _e2 = _pts[3] - _pts[0]
                    _n = np.cross(_e1, _e2)
                    _nn = float(np.linalg.norm(_n))
                    if _nn > 1e-12:
                        _n = _n / _nn
                        _c = _pts.mean(axis=0)
                        _d = -float(_n @ _c)
                        _man = json.load(open(
                            Path("/home/farhan/Igus_rebel/outputs/"
                                 "rgbd_captures/captures") / f"{cid}.json"))
                        _k = _man["camera_info"]["k"]
                        _fx, _fy, _cx, _cy = (float(_k[0]),
                                                     float(_k[4]),
                                                     float(_k[2]),
                                                     float(_k[5]))

                        def _bp(u, v):
                            _dx = (float(u) - _cx) / _fx
                            _dy = (float(v) - _cy) / _fy
                            _den = (float(_n[0]) * _dx
                                    + float(_n[1]) * _dy
                                    + float(_n[2]))
                            if abs(_den) < 1e-12:
                                return None
                            _t = -_d / _den
                            if not np.isfinite(_t) or _t <= 0:
                                return None
                            return np.array([_t * _dx, _t * _dy, _t])
                        plane_mm = _bp
            except (OSError, ValueError, KeyError, IndexError):
                plane_mm = None
            for key, med_list, mm_list, ang_list, ep_list in (
                    ("prior_endpoints_px", prior_med, prior_mm,
                     prior_ang, prior_ep),
                    ("refined_endpoints_px", ref_med, ref_mm,
                     ref_ang, ref_ep),
                    ("tape_line", tape_med, tape_mm, tape_ang, tape_ep)):
                eps = (res.get(key) if key != "tape_line"
                       else row.get("tape_line"))
                if eps:
                    err = _perp_line_err(eps, gt_eps)
                    if err is None:
                        continue
                    med_list.append(err["perp_median_px"])
                    ang_list.append(err["angle_deg"])
                    ep_list.append(err["endpoint_mean_px"])
                    row[key + "_perp_median_px"] = err["perp_median_px"]
                    row[key + "_perp_25_px"] = err["perp_25_px"]
                    row[key + "_perp_50_px"] = err["perp_50_px"]
                    row[key + "_perp_75_px"] = err["perp_75_px"]
                    row[key + "_angle_deg"] = err["angle_deg"]
                    row[key + "_endpoint_mean_px"] = err[
                        "endpoint_mean_px"]
                    if plane_mm is not None:
                        try:
                            _hx0, _hy0 = (float(gt_eps["seam_left"][0]),
                                          float(gt_eps["seam_left"][1]))
                            _hx1, _hy1 = (float(gt_eps["seam_right"][0]),
                                           float(gt_eps["seam_right"][1]))
                            _px0, _py0 = (float(eps["seam_left"][0]),
                                           float(eps["seam_left"][1]))
                            _px1, _py1 = (float(eps["seam_right"][0]),
                                            float(eps["seam_right"][1]))
                            _pdx, _pdy = _px1 - _px0, _py1 - _py0
                            _pn = float((_pdx ** 2 + _pdy ** 2) ** 0.5)
                            _mm = []
                            for _f in (0.25, 0.5, 0.75):
                                _sx = _hx0 + _f * (_hx1 - _hx0)
                                _sy = _hy0 + _f * (_hy1 - _hy0)
                                _t = ((_sx - _px0) * _pdx
                                      + (_sy - _py0) * _pdy) / (_pn ** 2)
                                _fxp = _px0 + _t * _pdx
                                _fyp = _py0 + _t * _pdy
                                _s3 = plane_mm(_sx, _sy)
                                _f3 = plane_mm(_fxp, _fyp)
                                if _s3 is not None and _f3 is not None:
                                    _mm.append(float(
                                        np.linalg.norm(_s3 - _f3)) * 1000.0)
                            if _mm:
                                mm_list.append(float(np.median(_mm)))
                                row[key + "_perp_median_mm"] = round(
                                    float(np.median(_mm)), 2)
                        except (TypeError, ValueError, KeyError,
                                IndexError):
                            pass
        status_counts[res["status"]] += 1
        per.append(row)
        json.dump(row, open(out / "captures" / f"{cid}.json", "w"), indent=1)
    per_sorted = per
    overlay_ids = [r["capture_id"] for r in per_sorted[:overlay_n]]
    for cid in overlay_ids:
        rec = reviews[cid]
        row = next(r for r in per if r["capture_id"] == cid)
        auto = json.load(open(Path(autolabels_dir) / f"{cid}.json"))
        img_p = (auto.get("image") or "").strip()
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        cn = rec.get("corners_px") or (auto.get("top_face") or {}) \
            .get("corners_px")
        corners = [cn[n] for n in CORNER_NAMES]
        ov = _draw_overlay(bgr, np.array(corners, float),
                           rec.get("seam_endpoints_px"), row["result"])
        cv2.imwrite(str(out / "overlays" / f"{cid}.png"), ov)
    summary = {
        "corners": corners_src, "sam3": bool(use_sam3),
        "n_captures": len(per),
        "axis_accuracy": (round(n_axis_ok / n_axis_tot, 3)
                          if n_axis_tot else None),
        "axis_n": n_axis_tot,
        "status_counts": dict(status_counts),
        "prior_perp_median_px": _summ(prior_med),
        "refined_perp_median_px": _summ(ref_med),
        "tape_perp_median_px": _summ(tape_med),
        "prior_angle_deg": _summ(prior_ang),
        "refined_angle_deg": _summ(ref_ang),
        "tape_angle_deg": _summ(tape_ang),
        "prior_endpoint_mean_px": _summ(prior_ep),
        "refined_endpoint_mean_px": _summ(ref_ep),
        "tape_endpoint_mean_px": _summ(tape_ep),
        "prior_perp_median_mm": _summ(prior_mm),
        "refined_perp_median_mm": _summ(ref_mm),
        "tape_perp_median_mm": _summ(tape_mm),
    }
    json.dump({"summary": summary, "per_capture": per},
              open(out / "report.json", "w"), indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tape-seam estimator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="eval on reviewed seams")
    ev.add_argument("--reviews", required=True)
    ev.add_argument("--autolabels", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--corners", choices=["reviewed", "autolabel"],
                    default="reviewed")
    ev.add_argument("--sam3", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "eval":
        summary = run_eval(args.reviews, args.autolabels, args.out,
                           corners_src=args.corners, use_sam3=args.sam3)
        print(json.dumps(summary, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
