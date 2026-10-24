# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/corner_names.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Canonical top-face corner names (rule ``long_side_back_v1``).

User-approved naming rule (2026-09-27) for the four top-face corners
``back_left, back_right, front_right, front_left``:

* The BACK edge is the LONG side of the top face (small box 112 x ~100 mm,
  big box ~193 x ~105 mm) that is FARTHER from the camera: 3-D distance from
  the camera centre to the edge midpoint, both edge end points back-projected
  onto the top plane (pixel rays with lens distortion, like
  :func:`fold_edge.corners_xyz_on_plane`).
* ``back_left`` / ``back_right`` = the back edge's left / right end in the
  IMAGE (smaller / larger image x).
* ``front_left`` / ``front_right`` = the corners adjacent to ``back_left`` /
  ``back_right`` (the other end of the short side leaving that corner).

Long vs short is decided on the MEAN 3-D length of each pair of opposite
sides. When ``|long - short| < AMBIGUOUS_LONG_MM`` (3 mm) the decision is
still made (larger mean wins; exact tie -> the pair containing the first
convex-ring edge) but flagged ``ambiguous_long_side``. Likewise when the two
long sides' midpoints are within ``AMBIGUOUS_BACK_MM`` (5 mm) of the same
distance from the camera (long sides running away from the camera) the
farther one still wins but the record is flagged ``ambiguous_back_side``.

Tie rule for a near-vertical back edge (its image direction within
``NEAR_VERTICAL_DEG`` = 10 deg of vertical, where "smaller x" is decided by
a few pixels of click noise): ``back_left`` is the back-edge end from which
the ring back_left -> back_right -> front_right -> front_left runs CLOCKWISE
in the image (y down). That is the viewer's left hand when standing at the
front edge looking across the face at the back edge, and it coincides with
"smaller x" whenever the face lies below its back edge in the image (the
normal case for a far back edge). Flag ``near_vertical_back_edge``. Outside
that band the literal smaller-x rule is used; if it yields a
counter-clockwise ring the result is flagged ``ccw_ring`` (never seen on
real captures, kept visible rather than silently re-oriented).

No plane / no camera / a ray missing the plane -> image fallback (method
``image_fallback``, flag ``no_plane_image_fallback``): lengths in pixels
(ambiguous when the relative difference is < 3 %), "farther" = the long
side whose midpoint is higher in the image (smaller y).

Input points may come in ANY order (named dict or 4-list); they are first
ordered convexly (clockwise by angle about the centroid).
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

RULE = "long_side_back_v1"
CORNER_ORDER = ("back_left", "back_right", "front_right", "front_left")
EDGE_NAMES = ("back", "right", "front", "left")
#: edge name -> its two corner names (ring order)
EDGE_CORNERS = {"back": ("back_left", "back_right"),
                "right": ("back_right", "front_right"),
                "front": ("front_right", "front_left"),
                "left": ("front_left", "back_left")}
AMBIGUOUS_LONG_MM = 3.0
AMBIGUOUS_LONG_REL_IMAGE = 0.03
#: back vs front midpoint distance margin below which the "farther" choice
#: is flagged ``ambiguous_back_side`` (long sides running away from the
#: camera; still decided deterministically: larger distance wins)
AMBIGUOUS_BACK_MM = 5.0
NEAR_VERTICAL_DEG = 10.0

_SWAP_AXIS = {"lr": "bf", "bf": "lr"}


# ---------------------------------------------------------------------------
# Camera / plane helpers
# ---------------------------------------------------------------------------

def camera_from_manifest(manifest):
    """(K 3x3, dist or None) from a capture manifest dict (``camera_info``
    k/d/distortion_model) or a camera_info dict. None when unusable."""
    if not isinstance(manifest, dict):
        return None
    info = manifest.get("camera_info") if isinstance(
        manifest.get("camera_info"), dict) else manifest
    try:
        k = np.asarray(info["k"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        return None
    model = str(info.get("distortion_model", "plumb_bob"))
    dist = info.get("d") if model in ("plumb_bob",
                                      "rational_polynomial") else None
    if dist is not None:
        try:
            dist = [float(v) for v in list(dist)[:8]]
        except (TypeError, ValueError):
            dist = None
    return k, dist


def _plane_nd(plane):
    """(unit n, d) with n.x + d = 0 from {'normal','offset_m'} / object with
    .normal/.d / (n, d); None when unusable."""
    try:
        if plane is None:
            return None
        if hasattr(plane, "normal"):
            n = np.asarray(plane.normal, dtype=np.float64).ravel()
            d = float(plane.d)
        elif isinstance(plane, dict):
            n = np.asarray(plane["normal"], dtype=np.float64).ravel()
            d = float(plane.get("offset_m", plane.get("d")))
        else:
            n = np.asarray(plane[0], dtype=np.float64).ravel()
            d = float(plane[1])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if n.shape != (3,) or not np.all(np.isfinite(n)) or not math.isfinite(d):
        return None
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return None
    return n / norm, d / norm


def backproject(uv, plane, K, dist=None):
    """Pixels (N,2) -> (N,3) camera-frame points on the plane (distortion-
    aware via fold_edge's back-projection); NaN rows where the ray misses.
    None when the plane/camera are unusable."""
    nd = _plane_nd(plane)
    if nd is None or K is None:
        return None
    from .fold_edge import Camera, _backproject
    n, d = nd
    if d < 0:
        n, d = -n, -d
    cam = Camera(K, dist)
    return _backproject(np.asarray(uv, dtype=np.float64).reshape(-1, 2),
                        cam, n, d)


# ---------------------------------------------------------------------------
# Core rule
# ---------------------------------------------------------------------------

def _as_points(corners_px):
    """(pts (4,2), input_names or None)."""
    if isinstance(corners_px, dict):
        names = [k for k in CORNER_ORDER if k in corners_px]
        if len(names) != 4:
            raise ValueError("corners_px dict needs all 4 corner names")
        pts = np.array([[float(corners_px[k][0]), float(corners_px[k][1])]
                        for k in CORNER_ORDER], dtype=np.float64)
        return pts, list(CORNER_ORDER)
    pts = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    return pts, None


def convex_ring(pts):
    """Indices of ``pts`` ordered clockwise in the image (y down: increasing
    atan2 about the centroid), starting at the smallest angle."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return [int(i) for i in np.argsort(ang, kind="stable")]


def canonical_order(pts, plane=None, K=None, dist=None):
    """Core of the rule. ``pts`` (4,2) any order. Returns ``(idx, info)``:
    ``idx[j]`` = index into ``pts`` of CORNER_ORDER[j]."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(p)):
        raise ValueError("corner points must be finite")
    ring = convex_ring(p)
    rp = p[ring]
    flags: list[str] = []
    X = None
    if plane is not None and K is not None:
        X = backproject(rp, plane, K, dist)
        if X is not None and not np.all(np.isfinite(X)):
            X = None
    info: dict = {"rule": RULE}
    # side k joins ring[k] -> ring[k+1]
    if X is not None:
        method = "plane_3d"
        lens = [float(np.linalg.norm(X[(k + 1) % 4] - X[k])) * 1000.0
                for k in range(4)]
        mids = [0.5 * (X[k] + X[(k + 1) % 4]) for k in range(4)]
        far = [float(np.linalg.norm(m)) for m in mids]
    else:
        method = "image_fallback"
        flags.append("no_plane_image_fallback")
        lens = [float(np.linalg.norm(rp[(k + 1) % 4] - rp[k]))
                for k in range(4)]
        mids = [0.5 * (rp[k] + rp[(k + 1) % 4]) for k in range(4)]
        far = [-float(m[1]) for m in mids]  # higher in image = farther
    mean_a = 0.5 * (lens[0] + lens[2])
    mean_b = 0.5 * (lens[1] + lens[3])
    long_pair = (0, 2) if mean_a >= mean_b else (1, 3)
    long_len, short_len = max(mean_a, mean_b), min(mean_a, mean_b)
    if method == "plane_3d":
        if long_len - short_len < AMBIGUOUS_LONG_MM:
            flags.append("ambiguous_long_side")
    elif long_len <= 0 or (long_len - short_len) / long_len \
            < AMBIGUOUS_LONG_REL_IMAGE:
        flags.append("ambiguous_long_side")
    k0, k1 = long_pair
    kb = k0 if far[k0] >= far[k1] else k1  # back side
    kf = k1 if kb == k0 else k0
    if method == "plane_3d" and (far[kb] - far[kf]) * 1000.0 \
            < AMBIGUOUS_BACK_MM:
        flags.append("ambiguous_back_side")
    a, b = kb, (kb + 1) % 4  # ring positions of the back edge ends
    dx = float(rp[b][0] - rp[a][0])
    dy = float(rp[b][1] - rp[a][1])
    near_vertical = abs(dx) < math.tan(math.radians(NEAR_VERTICAL_DEG)) \
        * abs(dy)
    if near_vertical:
        flags.append("near_vertical_back_edge")
        cw = True  # clockwise ring: back_left = ring[a]
    else:
        cw = rp[a][0] <= rp[b][0]
    if cw:
        pos = [a, b, (a + 2) % 4, (a + 3) % 4]
    else:
        flags.append("ccw_ring")
        pos = [b, a, (a + 3) % 4, (a + 2) % 4]
    idx = [ring[q] for q in pos]
    info.update({
        "method": method,
        "flags": flags,
        "ambiguous": ("ambiguous_long_side" in flags
                      or "ambiguous_back_side" in flags),
        "orientation": "cw" if cw else "ccw",
    })
    if method == "plane_3d":
        info.update({"long_side_mm": round(long_len, 2),
                     "short_side_mm": round(short_len, 2),
                     "back_mid_dist_m": round(far[kb], 5),
                     "front_mid_dist_m": round(far[kf], 5)})
    else:
        info.update({"long_side_mm": None, "short_side_mm": None,
                     "long_side_px": round(long_len, 2),
                     "short_side_px": round(short_len, 2),
                     "back_mid_dist_m": None, "front_mid_dist_m": None})
    return idx, info


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True when segments p1-p2 and p3-p4 properly intersect."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def ring_is_simple(pts) -> bool:
    """True when the 4-point ring (in the given order) does not self-cross."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    return not (_segments_cross(p[0], p[1], p[2], p[3])
                or _segments_cross(p[1], p[2], p[3], p[0]))


def canonical_corner_names(corners_px, plane=None, K=None, dist=None,
                           strict_ring=True):
    """Name 4 top-face corners by the ``long_side_back_v1`` rule.

    ``corners_px``: named dict (current names) or 4 points in any order.
    ``plane``: {'normal', 'offset_m'} (camera frame, n.x + d = 0) or None.
    ``K``: 3x3 intrinsics (or 9-list); ``dist``: plumb_bob coefficients.
    Returns ``(named, info)``: ``named`` = {name: [x, y]} (input values,
    unchanged), ``info`` = {rule, renamed, method, flags, ambiguous,
    orientation, long_side_mm, short_side_mm, back_mid_dist_m,
    front_mid_dist_m, source_index, from_names (dict input only)}.
    ``renamed`` is True when a named input changes names (for a list input
    the list order is read as CORNER_ORDER).
    ``strict_ring``: a named input whose stored ring self-crosses is a bad
    quad, not a naming problem - it is returned unchanged with flag
    ``input_not_simple`` (renaming would silently straighten it). Pass False
    for points clicked in arbitrary order (review tool).
    """
    pts, names = _as_points(corners_px)
    if names is not None and strict_ring and not ring_is_simple(pts):
        named = {k: [float(corners_px[k][0]), float(corners_px[k][1])]
                 for k in CORNER_ORDER}
        return named, {"rule": RULE, "method": "not_named",
                       "flags": ["input_not_simple"], "ambiguous": True,
                       "orientation": None, "source_index": [0, 1, 2, 3],
                       "renamed": False,
                       "from_names": {k: k for k in CORNER_ORDER}}
    if K is not None:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    idx, info = canonical_order(pts, plane, K, dist)
    if isinstance(corners_px, dict):
        src = [corners_px[k] for k in CORNER_ORDER]
    else:
        src = [list(v) for v in np.asarray(corners_px,
                                           dtype=np.float64).reshape(4, 2)]
    named = {CORNER_ORDER[j]: [float(src[idx[j]][0]), float(src[idx[j]][1])]
             for j in range(4)}
    info["source_index"] = idx
    info["renamed"] = idx != [0, 1, 2, 3]
    if names is not None:
        info["from_names"] = {CORNER_ORDER[j]: names[idx[j]]
                              for j in range(4)}
    return named, info


def canonical_names_for_manifest(corners_px, plane, manifest,
                                 strict_ring=True):
    """:func:`canonical_corner_names` with the camera of a capture manifest
    (``camera_info`` k/d); image fallback when the manifest is unusable."""
    cam = camera_from_manifest(manifest) if manifest is not None else None
    K, dist = cam if cam is not None else (None, None)
    return canonical_corner_names(corners_px, plane if cam else None,
                                  K, dist, strict_ring=strict_ring)


# ---------------------------------------------------------------------------
# Record renaming (autolabel box_autolabel/v5 records)
# ---------------------------------------------------------------------------

def edge_map(new2old: dict) -> dict:
    """new edge name -> old edge name for a corner renaming."""
    old_by_set = {frozenset(v): k for k, v in EDGE_CORNERS.items()}
    out = {}
    for e, (c0, c1) in EDGE_CORNERS.items():
        out[e] = old_by_set[frozenset((new2old[c0], new2old[c1]))]
    return out


def axis_swapped(new2old: dict) -> bool:
    """True when the renaming turns the old left/right edges into back/front
    (seam axis "lr" <-> "bf")."""
    return edge_map(new2old)["left"] in ("back", "front")


def swap_axis_labels(obj):
    """Deep copy of ``obj`` with seam axis labels swapped: dict keys "lr"/
    "bf" and string values "lr"/"bf" under any key named ``axis`` or ending
    in ``_axis``. Rect-frame labels ("h"/"v") are left alone."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _SWAP_AXIS.get(k, k) if isinstance(k, str) else k
            if (isinstance(v, str) and isinstance(k, str)
                    and (k == "axis" or k.endswith("_axis"))):
                out[nk] = _SWAP_AXIS.get(v, v)
            else:
                out[nk] = swap_axis_labels(v)
        return out
    if isinstance(obj, list):
        return [swap_axis_labels(v) for v in obj]
    return obj


_CORNER_RE = re.compile(
    r"(?<![A-Za-z0-9])(back_left|back_right|front_right|front_left)(?![A-Za-z0-9])")
_FOLD_RE = re.compile(r"corner_fold_moved\(([a-z,]*)\)")


def _rename_reason(s: str, old2new: dict, eold2new: dict) -> str:
    if not isinstance(s, str):
        return s

    def fold(m):
        parts = [eold2new.get(p, p) for p in m.group(1).split(",") if p]
        order = {e: i for i, e in enumerate(EDGE_NAMES)}
        parts.sort(key=lambda e: order.get(e, 9))
        return "corner_fold_moved(" + ",".join(parts) + ")"

    s = _FOLD_RE.sub(fold, s)

    def corner(m):
        return old2new.get(m.group(1), m.group(1))

    s = _CORNER_RE.sub(corner, s)

    # corner_flags(a,b): keep the sorted listing sorted after renaming
    def flags(m):
        names = sorted(p for p in m.group(1).split(",") if p)
        return "corner_flags(" + ",".join(names) + ")"

    return re.sub(r"corner_flags\(([a-z_,]*)\)", flags, s)


def rename_record(rec: dict, manifest=None, source: str | None = None,
                  code_git_rev: str | None = None) -> tuple[dict, dict]:
    """Apply the naming rule to one box_autolabel/v5 record WITHOUT touching
    geometry. Returns ``(new_record, info)`` (deep copy).

    Permuted consistently: top_face.corners_px / corners_xyz_camera_m,
    checks.corner_flags.corners, provenance.corner_fold.corners_px_before
    and .per_edge (edge keys), the accepted attempt's ``coarse`` rows (so a
    later recheck pairs coarse and refined corners correctly), corner/edge
    names inside checks.reasons, and the seam axis label ("lr" = seam joins
    the left & right edges, "bf" = joins back & front) everywhere in
    ``seam`` when the renaming rotates by an odd step. Seam endpoints
    seam_left/seam_right are image-relative and never change. Rect-frame
    diagnostics inside seam.estimate ("h"/"v", *_rect values) stay in the
    corner order the seam was computed with (``rect_frame`` in the stamp).
    Stamps ``provenance.corner_names``. Records without corners are returned
    unchanged apart from the stamp.
    """
    out = json.loads(json.dumps(rec))
    tf = out.get("top_face") or {}
    named = tf.get("corners_px")
    stamp: dict = {"rule": RULE}
    if source is not None:
        stamp["from"] = str(source)
    if code_git_rev is not None:
        stamp["code_git_rev"] = code_git_rev
    if not isinstance(named, dict) or not all(
            isinstance(named.get(k), (list, tuple)) for k in CORNER_ORDER):
        stamp.update({"renamed": False, "reason": "no_corners"})
        out.setdefault("provenance", {})["corner_names"] = stamp
        return out, stamp
    _, info = canonical_names_for_manifest(named, tf.get("plane"), manifest)
    new2old = info["from_names"]
    old2new = {v: k for k, v in new2old.items()}
    emap = edge_map(new2old)
    eold2new = {v: k for k, v in emap.items()}
    swap = axis_swapped(new2old)
    renamed = bool(info["renamed"])
    stamp.update({k: info.get(k) for k in (
        "method", "flags", "ambiguous", "orientation", "long_side_mm",
        "short_side_mm", "back_mid_dist_m", "front_mid_dist_m")})
    stamp.update({"renamed": renamed,
                  "mapping_new_from_old": new2old if renamed else None,
                  "edge_mapping_new_from_old": emap if renamed else None,
                  "axis_swapped": bool(renamed and swap)})
    if renamed:
        stamp["rect_frame"] = "seam.estimate rect diagnostics keep the " \
            "pre-rename corner order"

        def perm_named(d):
            if not isinstance(d, dict):
                return d
            if not all(k in d for k in CORNER_ORDER):
                return {old2new.get(k, k): v for k, v in d.items()}
            rest = {k: v for k, v in d.items() if k not in CORNER_ORDER}
            return {**{k: d[new2old[k]] for k in CORNER_ORDER}, **rest}

        tf["corners_px"] = perm_named(named)
        if isinstance(tf.get("corners_xyz_camera_m"), dict):
            tf["corners_xyz_camera_m"] = perm_named(
                tf["corners_xyz_camera_m"])
        # K1 / D1 records (top_face.top_from_gap, top_face.depth_diag): corner dicts
        # and per-edge dicts follow the renaming (absent on records of other profiles)
        tfg = tf.get("top_from_gap")
        if isinstance(tfg, dict):
            for qk in ("rgb_quad", "gap_anchored_quad"):
                if isinstance(tfg.get(qk), dict):
                    for ck in ("corners_px", "corners_xyz_camera_m"):
                        if isinstance(tfg[qk].get(ck), dict):
                            tfg[qk][ck] = perm_named(tfg[qk][ck])
            for ek in ("deltas_mm",):
                pe = tfg.get(ek)
                if isinstance(pe, dict) and all(e in pe for e in EDGE_NAMES):
                    tfg[ek] = {e: pe[emap[e]] for e in EDGE_NAMES}
            if isinstance(tfg.get("sides_mm"), list) and len(tfg["sides_mm"]) == 4:
                # ring-order side lengths (edge i joins ring corners i, i+1): follow the edges
                old_by_name = dict(zip(EDGE_NAMES, tfg["sides_mm"]))
                tfg["sides_mm"] = [old_by_name[emap[e]] for e in EDGE_NAMES]
        dd = tf.get("depth_diag")
        if isinstance(dd, dict):
            es = dd.get("edge_support")
            if isinstance(es, dict):
                pe = es.get("edges")
                if isinstance(pe, dict) and all(e in pe for e in EDGE_NAMES):
                    es["edges"] = {e: pe[emap[e]] for e in EDGE_NAMES}
                if isinstance(es.get("unsupported"), list):
                    es["unsupported"] = sorted(
                        (eold2new.get(e, e) for e in es["unsupported"]),
                        key=lambda e: EDGE_NAMES.index(e) if e in EDGE_NAMES else 9)
            ksf = dd.get("known_size_fit")
            if isinstance(ksf, dict) and isinstance(ksf.get("corner_disagreement_mm"), list) \
                    and len(ksf["corner_disagreement_mm"]) == 4:
                old_by = dict(zip(CORNER_ORDER, ksf["corner_disagreement_mm"]))
                ksf["corner_disagreement_mm"] = [old_by[new2old[k]] for k in CORNER_ORDER]
        checks = out.get("checks") or {}
        cf = checks.get("corner_flags")
        if isinstance(cf, dict) and isinstance(cf.get("corners"), dict):
            cf["corners"] = perm_named(cf["corners"])
        if isinstance(checks.get("reasons"), list):
            checks["reasons"] = [_rename_reason(r, old2new, eold2new)
                                 for r in checks["reasons"]]
        prov = out.get("provenance") or {}
        fold = prov.get("corner_fold")
        if isinstance(fold, dict):
            if isinstance(fold.get("corners_px_before"), dict):
                fold["corners_px_before"] = perm_named(
                    fold["corners_px_before"])
            pe = fold.get("per_edge")
            if isinstance(pe, dict) and all(e in pe for e in EDGE_NAMES):
                fold["per_edge"] = {e: pe[emap[e]] for e in EDGE_NAMES}
        oldi = {k: i for i, k in enumerate(CORNER_ORDER)}
        for att in prov.get("attempts") or []:
            if (isinstance(att, dict) and att.get("status") == "accepted"
                    and isinstance(att.get("coarse"), list)
                    and len(att["coarse"]) == 4):
                c = att["coarse"]
                att["coarse"] = [c[oldi[new2old[k]]] for k in CORNER_ORDER]
        if swap and isinstance(out.get("seam"), dict):
            out["seam"] = swap_axis_labels(out["seam"])
    out.setdefault("provenance", {})["corner_names"] = stamp
    return out, stamp


def canonicalise_named(named: dict, plane, manifest) -> tuple[dict, dict]:
    """Rename a {name: [x, y]} dict (e.g. a human label) by the rule; values
    are the input points unchanged. ``manifest``: capture manifest or
    camera_info dict (K + dist)."""
    return canonical_names_for_manifest(named, plane, manifest)
