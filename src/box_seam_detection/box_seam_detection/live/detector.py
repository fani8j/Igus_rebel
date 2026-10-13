# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/autolabel_v2_detect.py
# at commit 08d1041 (working tree identical to HEAD, unmodified). Changes: package-relative imports (no sys.path,
# no file-path loading), --autolabel-repo accepted but unused; logic unchanged.
# -*- coding: utf-8 -*-
"""autolabel_v2_detect.py (now box_seam_detection.live.detector) — live single-capture adapter for the autolabel_v2
detector (SAM3 live coarse -> refine -> depth cross-check -> corner fold ->
flap-gap seam estimator -> long_side_back_v1 corner names).

Runs ``autolabel_v2.pipeline.process_capture`` on ONE capture with the SAM3
live source (a persistent-worker subprocess in the book_grasping
interpreter, exactly like ``autolabel_v2/run.py``; HF_HUB_OFFLINE=1) and
converts the box_autolabel/v5 record into the orchestrator's canonical
six-point seam JSON (same keys/binding fields as guided_seam_adapter.py).

Seam: left_mid/right_mid are the PUBLISHED seam line endpoints —
``seam.endpoints_px`` when status is ok, else the estimator's published
prior line (``seam.estimate.estimate.estimate_endpoints_px`` = geometric
projective midline) when status is estimated. ambiguous/invisible seams
(no axis) are rejected. ``seam_observed`` = (status == ok).

Contract corner names: the robot contract assumes the seam joins the LEFT
edge (back_left-front_left) to the RIGHT edge (back_right-front_right).
The rule names (long_side_back_v1: back = long side farther from camera)
are relabelled by a cyclic shift of the ring [BL, BR, FR, FL] (orientation
preserved) so left_mid lies on the contract left edge and right_mid on the
contract right edge: identity when the seam already joins the rule
left/right edges (axis lr), else the shift (of the two possible) that makes
left_mid the image-left seam endpoint. Both endpoints must lie within
``--edge-tol-px`` of their contract edges, else rejection. The rule names and
mapping are kept under the ``autolabel_v2`` key.

Exit codes (fail-closed, same as guided_seam_adapter.py):
  0 — accepted; canonical seam JSON written atomically.
  1 — scene rejection: no top face, wrong-face suspect, needs_review
      (unless --allow-needs-review), top face not a rectangle in 3-D
      (rectangle_inconsistent, retryable) or not verifiable
      (rectangle_unverifiable), seam ambiguous/invisible, endpoint not on
      contract edges. No output file; a reject JSON (IDs + reasons +
      ``retryable``) is written to --overlay-dir (or --work-dir).
  2 — prerequisite failure: missing input, manifest/image binding mismatch,
      SAM3 interpreter/worker unavailable, import failure, write failure,
      invalid --edge-tol-px (must be 0 < tol <= 5), --rect-tol-mm (must be
      0 < tol <= 20) or --sam3-timeout, or an unexpected internal error while
      processing / building the document.
Any pre-existing --out-seam-json is removed before any other work.

Server mode (``--serve --work-dir DIR [--sam3-python P] [--autolabel-repo R]``):
starts the SAM3 worker immediately (the orchestrator captures meanwhile),
prints one status line ({"type": "ready"} / {"type": "startup_error"}) on
stdout, then answers one JSON request per stdin line ({"id", "argv": [the
one-shot CLI arguments]}) with one JSON response line ({"type": "response",
"id", "exit_code", "out_seam_json", "reject_json", "timing", ...}). Each
request runs exactly the one-shot per-capture code (same files, same exit
code as ``exit_code``); only the SAM3 worker is reused across requests.
EOF or {"type": "shutdown"} stops it. See :func:`serve`.

Rectangle gate (default on; --no-rect-check disables): a real box top is a
rectangle, so opposite 3-D side lengths of ``top_face.corners_xyz_camera_m``
(rule names) must agree: pair A = |BL-BR| vs |FL-FR| (back vs front), pair
B = |BL-FL| vs |BR-FR| (left vs right). Either |difference| > --rect-tol-mm
(default 3.0 mm) -> exit 1, reason ``rectangle_inconsistent(A=..mm,B=..mm)``,
``"retryable": true`` in the reject JSON (a skewed quad from one capture's
edge jitter; a fresh capture usually fixes it). Missing/invalid 3-D corners
or a side shorter than 1 mm -> ``rectangle_unverifiable`` (not retryable).
The lengths/diffs are recorded under ``autolabel_v2.rect_check``.

Depth edge fix (default on; --no-depth-edge disables), applied to the record
BEFORE the rectangle gate and the contract relabel: of the two top-face edges
the published seam crosses (axis bf -> back & front, lr -> left & right; rule
names) one can have two RGB candidate lines a few mm apart (the near edge
beside a visible side wall), which skews the quad into a trapezoid. The
ANCHOR edge is the one with the lower RGB ambiguity (edge_ambiguous_frac
re-sampled on the image; fallback: the record's per-corner ambiguous_frac),
ties (< 0.10 apart) -> the edge farther from the camera. The flat top's
DEPTH extent along the seam direction u_seam (top-plane cloud points
|n.x+d| < 4 mm, central band |a| < 30 mm along the anchor edge; robust
p0.5/p99.5) gives one boundary per side. The other edge becomes the line
PARALLEL to the anchor edge (in the top plane) AT the other-side depth
boundary (no offset; the anchor's fold-vs-depth offset delta = anchor RGB
edge - anchor-side depth boundary is measured and recorded only); its
corners are its intersections with the two short-side RGB lines (3-D in the
plane, projected to pixels WITH lens distortion). Applied only when the RGB
other edge disagrees with that line by > 1.0 mm at the band centre or at
either end (a rotated edge agrees at the centre). When applied, the
published seam line is RECOMPUTED as the projective midline of the fixed
quad (diagonal intersection -> meeting point of the two short sides, in the
top plane), ends on the two seam-crossed edges; seam.status is kept, but an
``ok`` seam whose recomputed line moved > 1.5 mm laterally from the old
(confirmed) line is published with ``seam_observed`` false
(``seam_status_downgraded``). Everything (applied/skip reason, anchor, widths, delta,
per-corner moves, band point count) is recorded under
``autolabel_v2.depth_edge`` and in the written record (``depth_edge``). Any
failure only skips the fix (the rectangle gate still guards the output).

Reliability extensions R1 (2026-10-04; every path is recorded in
``depth_edge.mode`` ("seam" | "quad_geometry") and ``depth_edge.reasons``):
 (a) no published seam / seam without axis, SKEWED quad only: the two long
     sides (both pairs when square-ish) are the candidate seam-crossed edges
     (mode quad_geometry; needs placement support >= 0.60; no seam recompute);
 (b) both anchors ambiguous: both are evaluated (``candidates``), best depth
     support AND smaller rectangle residual wins;
 (c) centre move 3.5..10 mm allowed when the boundary is contiguous, the line
     is supported along >= 80 % of the edge (5 mm bins) and the moved edge
     stays inside the SAM3 box + 10 %/side; the 10 mm corner cap stays;
 (d) long side > 150 mm: allowed only for a skewed quad whose FIXED dims match
     a known class (small ~110x98, big ~193x105, +-8 mm);
 (e) the length difference of the seam-crossed pair left after the placement
     (non-parallel short sides) is repaired by placing the OTHER pair from
     depth (anchor = better supported edge, support >= 0.80), kept only if
     the residual improves.
 The result is dropped (original kept) when the max 3-D side difference ends
 > 0.5 mm worse than before, or a gate-passing quad would fail.
R7 (2026-10-04): the rectangle gate also checks the 4 corner angles of the 3-D
quad: any |angle - 90| > RECT_MAX_CORNER_DEV_DEG (5.0, --rect-max-corner-dev-deg)
-> ``rectangle_skewed(max_dev=..deg)`` (retryable, same exit path as
rectangle_inconsistent); the max deviation is recorded in ``rect_check``.
R8: (a) the second pair is placed perpendicular to the first anchor axis;
(b) [withdrawn after the replay: no minimum support in seam mode]; (c) support also
passes via a RANSAC line fit of the per-bin depth boundary (>= 0.60 inliers at
1.5 mm, within 2 deg of the anchor edge; criterion recorded in
``support_criterion``); (d) a placement moving the edge > 5 mm farther from the
quad centroid is refused (``depth_edge_moved_outward``).
R9: LAST RESORT, only for a quad the rectangle gate would reject: joint
rectangle fit in the top plane (shared orientation = circular mean of 4x the
edge angles, each edge a line through its midpoint at theta / theta+90,
corners = intersections); accepted only if every edge residual <= 3 deg, every
corner moves <= 4 mm in 3-D and the repaired quad passes the gate (incl. R7);
the seam endpoints are recomputed from the repaired quad (as after the depth
edge fix); flag ``rectangle_repaired_joint_fit`` (``autolabel_v2.soft_flags``,
``rect_repair`` with the per-corner moves).
R6: ``seam_direction`` (record + doc autolabel_v2) = seam direction in the
camera top-plane frame and approx. in base_link (hand-eye calibration x
recorded driver extrinsic); ``seam_direction_outside_soft_range`` when it is
beyond +-45 deg of 83 deg. Logged/flagged only, never a rejection.
"""

from __future__ import annotations

import os as _blas_os  # noqa: E402  (BLAS pin must precede the first numpy import)
if _blas_os.environ.get("AUTOLABEL_V2_BLAS_THREADS", "").strip().lower() != "default":
    if "OPENBLAS_NUM_THREADS" not in _blas_os.environ:   # RANSAC gemv: 7-8x faster
        _blas_os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _blas_os.environ["AUTOLABEL_V2_BLAS_PINNED"] = "1"  # ours, not the user's

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from ..core import box_profiles  # noqa: E402

DEFAULT_SAM3_PYTHON = "/home/farhan/miniconda3/envs/book_grasping/bin/python"

METHOD = "autolabel_v2_sam3_fold_midline"
GENERATOR = "autolabel_v2 (sam3_live + refine + corner_fold + seam_estimator)"
RING = ("back_left", "back_right", "front_right", "front_left")
DEFAULT_EDGE_TOL_PX = 2.0
MAX_EDGE_TOL_PX = 5.0
DEFAULT_RECT_TOL_MM = 3.0
MAX_RECT_TOL_MM = 20.0
MIN_RECT_SIDE_MM = 1.0
# R7 corner-angle gate: any top-face corner deviating more than this from 90 deg
# (3-D, camera frame) -> rectangle_skewed (retryable).
RECT_MAX_CORNER_DEV_DEG = 5.0
MAX_RECT_CORNER_DEV_DEG = 45.0
# R9 joint rectangle repair (last resort, only for a quad the rectangle gate rejects).
JOINT_FIT_MAX_RESIDUAL_DEG = 3.0     # per-edge angle residual to the shared orientation
JOINT_FIT_MAX_MOVE_MM = 4.0          # per-corner 3-D move
# Depth edge fix (see module docstring).
DEPTH_EDGE_PLANE_TOL_M = 0.004       # |n.x + d| for top-plane points
DEPTH_EDGE_RADIUS_M = 0.120          # min crop radius around the quad centre
DEPTH_EDGE_HALF_M = 0.080            # min |a|,|b| crop half-size
DEPTH_EDGE_BAND_M = 0.030            # central band |a| along the anchor edge
DEPTH_EDGE_MARGIN_M = 0.020          # crop margin beyond the RGB quad
DEPTH_EDGE_PCT = (0.5, 99.5)         # robust boundary percentiles
DEPTH_EDGE_MIN_POINTS = 2000         # band points required
DEPTH_EDGE_MIN_MOVE_MM = 0.0         # always apply on the small box (one edge convention per capture; live 2026-09-28)
DEPTH_EDGE_AMBIG_TIE = 0.10          # ambiguity difference counted as a tie
DEPTH_EDGE_MAX_WIDTH_DIFF_MM = 8.0   # |depth width - RGB width| sanity bound
DEPTH_EDGE_MAX_DELTA_MM = 6.0        # |fold-vs-depth offset| sanity bound
DEPTH_EDGE_CLIP_MM = 2.0             # boundary this close to the crop -> skip
SEAM_DOWNGRADE_SHIFT_MM = 1.5        # recomputed-vs-old seam shift -> not observed
DEPTH_EDGE_MAX_CENTRE_MOVE_MM = 3.5  # post-fix guard: edge move at band centre
DEPTH_EDGE_MAX_CORNER_MOVE_MM = 10.0  # post-fix guard: any moved corner
DEPTH_EDGE_AMBIG_BOTH = 0.3          # both edges this ambiguous + tie -> skip
DEPTH_EDGE_GAP_MM = 3.0              # boundary contiguity: max empty run
DEPTH_EDGE_GAP_DENSITY = 0.10        # "empty" = < 10% of the median density
DEPTH_EDGE_SMALL_MAX_LONG_M = box_profiles.LEGACY_BIG_LONG_SIDE_M  # today's class split (small box only)
# Reliability fixes R1 (2026-10-03): support-based move guard, skewed-quad class
# override, quad-geometry mode, both-anchor evaluation, second pair from depth.
DEPTH_EDGE_SUPPORTED_CENTRE_MOVE_MM = 10.0  # centre-move cap when depth-supported
DEPTH_EDGE_SUPPORT_MIN_FRAC = 0.80   # min fraction of supported bins along the edge
DEPTH_EDGE_SUPPORT_BIN_MM = 5.0
DEPTH_EDGE_SUPPORT_DEV_MM = 2.0      # per-bin boundary vs the placed line
DEPTH_EDGE_SUPPORT_MIN_PTS = 3
DEPTH_EDGE_SECOND_PAIR_MIN_SUPPORT = 0.80  # second pair placed from depth only when this well supported
DEPTH_EDGE_GEOMETRY_MIN_SUPPORT = 0.60     # quad-geometry mode (no published seam) first placement
DEPTH_EDGE_RANSAC_MIN_INLIER = 0.60  # R8(c): per-bin boundary line inlier fraction ...
DEPTH_EDGE_RANSAC_TOL_MM = 1.5       # ... within this tolerance ...
DEPTH_EDGE_RANSAC_MAX_ANGLE_DEG = 2.0  # ... and this angle to the anchor edge
DEPTH_EDGE_MAX_OUTWARD_MM = 5.0      # R8(d): moved edge may not end up farther out than this
DEPTH_EDGE_SAM_BOX_DILATE = 0.10     # moved edge must stay inside the SAM3 box + 10 %/side
DEPTH_EDGE_MAX_WIDTH_DIFF_EXT_MM = 12.0  # width sanity bound when the move cap is extended
DEPTH_EDGE_WORSE_MM = 0.5            # fix kept only if max rect diff <= before + this
DEPTH_EDGE_BOX_CLASSES = {n: box_profiles.PROFILES[n].top_dims_mm
                          for n in ("small", "big")}  # (long, short) mm, from box_profiles
DEPTH_EDGE_CLASS_TOL_MM = box_profiles.PROFILES["small"].dims_tol_mm
DEPTH_EDGE_SQUARE_RATIO = 1.12       # long/short below this = square-ish (try both pairs)
DEPTH_EDGE_PLACEMENT = (
    "anchor = RGB fold edge; moved edge = depth boundary (no offset; delta "
    "recorded only); widens/narrows the top by ~delta, which moves the seam "
    "endpoints ALONG the seam by ~delta/2 (a bf seam's lateral position comes "
    "from the short sides)")
PROG = "autolabel_v2_detect.py"
RELIABILITY_RULES = ("a", "c", "d", "r7", "r9", "r10")


# R8(a) (second pair perpendicular to the first anchor) and R8(c) (RANSAC-line
# support) are OFF by default: on the 431-capture replay (2026-10-04) they
# changed 7 previously accepted outputs by 1-5 px with no evidence of
# improvement (2 less skewed, 5 with new 1-2.5 mm opposite-side mismatches)
# and enabled only 1 extra capture. Kept for validation against labels.
DEFAULT_RULES_OFF = ("a", "c")


def rules_off() -> set:
    """Reliability sub-rules disabled via AUTOLABEL_V2_R8_OFF (comma list of
    a,c,d,r7,r9,r10). Unset -> DEFAULT_RULES_OFF; set (even empty) -> exactly
    that list ("" = all on). Read at call time."""
    raw = os.environ.get("AUTOLABEL_V2_R8_OFF")
    if raw is None:
        return set(DEFAULT_RULES_OFF)
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def reliability_rules() -> dict:
    off = rules_off()
    return {"off": sorted(off), "on": [r for r in RELIABILITY_RULES if r not in off],
            "env": "AUTOLABEL_V2_R8_OFF"}


class Reject(Exception):
    """Honest scene rejection (exit 1)."""

    def __init__(self, stage: str, reasons: list, retryable: bool = False,
                 details: dict = None):
        super().__init__(f"{stage}: {reasons}")
        self.stage = stage
        self.reasons = list(reasons)
        # True only when a fresh capture of the same scene is expected to fix
        # it (capture-level noise, e.g. a skewed top-face quad).
        self.retryable = bool(retryable)
        self.details = dict(details or {})


class Prerequisite(Exception):
    """Prerequisite failure (exit 2)."""


# ---------------------------------------------------------------- helpers

def write_json_atomically(path: Path, value: dict) -> None:
    """Same-directory temp file + os.replace; the temp file is removed on any
    failure (serialisation, fsync or rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _pair(value, name: str) -> list:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError) as error:
        raise Reject("record", [f"{name} is not a coordinate pair: {value!r}"]) from error
    if not (math.isfinite(x) and math.isfinite(y)):
        raise Reject("record", [f"{name} is not finite: {value!r}"])
    return [x, y]


def point_segment_distance(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def contract_edges(ring_points: dict) -> dict:
    """Named edges of a [BL, BR, FR, FL] quad."""
    return {
        "left": (ring_points["front_left"], ring_points["back_left"]),
        "right": (ring_points["back_right"], ring_points["front_right"]),
        "back": (ring_points["back_left"], ring_points["back_right"]),
        "front": (ring_points["front_right"], ring_points["front_left"]),
    }


def _image_left_first(p, q):
    """(image-left, image-right) by u, ties broken by v."""
    return (p, q) if (p[0], p[1]) <= (q[0], q[1]) else (q, p)


def contract_relabel(rule_corners: dict, seam_a, seam_b,
                     tol_px: float = DEFAULT_EDGE_TOL_PX) -> dict:
    """Cyclic relabel of the rule-named quad so the seam joins contract L/R.

    ``rule_corners``: {back_left, back_right, front_right, front_left} -> [u, v].
    ``seam_a``/``seam_b``: the two published seam endpoints (any order).
    Returns {"corners": contract-named points, "contract_from_rule":
    {contract_name: rule_name}, "shift": k, "left_mid", "right_mid",
    "geometric_axis": "lr"|"bf", "edge_distance_px": {...}}. Raises Reject
    when no cyclic relabel puts both endpoints within ``tol_px`` of their
    contract edges.
    """
    ring = [_pair(rule_corners.get(name), f"corner {name}") for name in RING]
    image_left, image_right = _image_left_first(_pair(seam_a, "seam endpoint"),
                                                _pair(seam_b, "seam endpoint"))
    candidates = []
    for shift in range(4):
        names = {RING[j]: RING[(j + shift) % 4] for j in range(4)}  # contract <- rule
        corners = {RING[j]: ring[(j + shift) % 4] for j in range(4)}
        edges = contract_edges(corners)
        for left_mid, right_mid in ((image_left, image_right), (image_right, image_left)):
            d_left = point_segment_distance(left_mid, *edges["left"])
            d_right = point_segment_distance(right_mid, *edges["right"])
            candidates.append({
                "shift": shift, "contract_from_rule": names, "corners": corners,
                "left_mid": left_mid, "right_mid": right_mid,
                "image_left_is_left_mid": left_mid is image_left,
                "edge_distance_px": {"left_mid": round(d_left, 3),
                                     "right_mid": round(d_right, 3)},
                "valid": max(d_left, d_right) <= tol_px,
            })
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        best = min(candidates, key=lambda c: max(c["edge_distance_px"].values()))
        raise Reject("contract_edges", [
            f"seam endpoints not within {tol_px} px of any contract left/right edge "
            f"pair (best shift {best['shift']}: {best['edge_distance_px']})"])
    identity = [c for c in valid if c["shift"] == 0]
    if identity:
        # Seam already joins the rule left/right edges: keep the rule names.
        chosen = min(identity, key=lambda c: max(c["edge_distance_px"].values()))
    else:
        preferred = [c for c in valid if c["image_left_is_left_mid"]] or valid
        chosen = min(preferred, key=lambda c: (max(c["edge_distance_px"].values()), c["shift"]))
    out = {key: chosen[key] for key in ("shift", "contract_from_rule", "corners",
                                        "left_mid", "right_mid", "edge_distance_px")}
    out["geometric_axis"] = "lr" if chosen["shift"] % 2 == 0 else "bf"
    return out


def published_seam_line(record: dict) -> dict:
    """Published seam endpoints of a box_autolabel/v5 record.

    ok -> seam.endpoints_px; estimated -> seam.estimate.estimate.
    estimate_endpoints_px (published prior = geometric midline), falling
    back to seam.estimate.prior_endpoints_px. ambiguous/invisible -> Reject.
    """
    seam = record.get("seam") or {}
    status = seam.get("status")
    endpoints = None
    source = None
    if status == "ok":
        endpoints = seam.get("endpoints_px")
        source = "seam.endpoints_px"
    elif status == "estimated":
        estimate = seam.get("estimate") or {}
        inner = estimate.get("estimate") or {}
        if isinstance(inner, dict) and inner.get("estimate_endpoints_px"):
            endpoints = inner["estimate_endpoints_px"]
            source = "seam.estimate.estimate.estimate_endpoints_px"
        elif estimate.get("prior_endpoints_px"):
            endpoints = estimate["prior_endpoints_px"]
            source = "seam.estimate.prior_endpoints_px"
    else:
        raise Reject("seam", [f"seam_status_{status}(no published seam axis)"])
    if not isinstance(endpoints, dict) or "seam_left" not in endpoints \
            or "seam_right" not in endpoints:
        raise Reject("seam", [f"seam status {status} carries no published endpoints"])
    return {"status": status, "observed": status == "ok", "source": source,
            "axis": seam.get("axis"), "seam_source": seam.get("source"),
            "seam_left": _pair(endpoints["seam_left"], "seam_left"),
            "seam_right": _pair(endpoints["seam_right"], "seam_right")}


RECT_SIDES = {
    # name: (rule corner a, rule corner b)
    "back": ("back_left", "back_right"),
    "front": ("front_left", "front_right"),
    "left": ("back_left", "front_left"),
    "right": ("back_right", "front_right"),
}


def corner_angles_deg(points: dict) -> dict:
    """Interior angle (deg) at each RING corner of the 3-D quad ``points``
    ({name: xyz}); the angle between the two adjacent edge vectors."""
    angles = {}
    for i, name in enumerate(RING):
        prev, nxt = points[RING[i - 1]], points[RING[(i + 1) % 4]]
        v1 = [float(a) - float(b) for a, b in zip(prev, points[name])]
        v2 = [float(a) - float(b) for a, b in zip(nxt, points[name])]
        n1 = math.sqrt(sum(c * c for c in v1))
        n2 = math.sqrt(sum(c * c for c in v2))
        cos = sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)
        angles[name] = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return angles


def _gap_anchored_rgb_xyz(record: dict):
    """3-D corners {name: xyz} of the RGB quad when the published quad is the
    gap-anchored one (top_face.top_from_gap applied); None otherwise (records of
    every other profile)."""
    tfg = (record.get("top_face") or {}).get("top_from_gap")
    if not (isinstance(tfg, dict) and tfg.get("applied")):
        return None
    xyz = (tfg.get("rgb_quad") or {}).get("corners_xyz_camera_m")
    try:
        return {k: [float(c) for c in xyz[k]] for k in RING}
    except (KeyError, TypeError, ValueError):
        return None


def _rgb_quad_rect_diffs(record: dict):
    """(|back-front|, |left-right|) mm of the measured RGB quad of a K1 record, or None."""
    pts = _gap_anchored_rgb_xyz(record)
    if pts is None:
        return None
    lengths = {side: 1000.0 * math.dist(pts[a], pts[b]) for side, (a, b) in RECT_SIDES.items()}
    return (abs(lengths["back"] - lengths["front"]), abs(lengths["left"] - lengths["right"]))


def rectangle_check(record: dict, tol_mm: float,
                    max_corner_dev_deg: float = RECT_MAX_CORNER_DEV_DEG) -> dict:
    """3-D rectangle consistency of the top face (rule corner names).

    Pair A = back |BL-BR| vs front |FL-FR|; pair B = left |BL-FL| vs right
    |BR-FR|. Returns {"enabled", "tol_mm", "lengths_mm", "diffs_mm", "pass"}
    when both |differences| <= tol_mm. Raises Reject("rectangle",
    ["rectangle_inconsistent(A=..mm,B=..mm)"], retryable=True) when either
    exceeds tol_mm, and Reject("rectangle", ["rectangle_unverifiable(...)"],
    retryable=False) when the 3-D corners are missing/invalid/degenerate.
    """
    xyz = (record.get("top_face") or {}).get("corners_xyz_camera_m")
    if not isinstance(xyz, dict):
        raise Reject("rectangle", ["rectangle_unverifiable(no corners_xyz_camera_m)"])
    points = {}
    for name in RING:
        value = xyz.get(name)
        try:
            point = [float(c) for c in value]
        except (TypeError, ValueError):
            point = None
        if point is None or len(point) != 3 or not all(math.isfinite(c) for c in point):
            raise Reject("rectangle", [f"rectangle_unverifiable({name} xyz invalid: {value!r})"])
        points[name] = point
    lengths = {side: 1000.0 * math.dist(points[a], points[b])
               for side, (a, b) in RECT_SIDES.items()}
    short = sorted(side for side, length in lengths.items() if length < MIN_RECT_SIDE_MM)
    if short:
        raise Reject("rectangle", [f"rectangle_unverifiable(degenerate sides {short})"],
                     details={"lengths_mm": {k: round(v, 3) for k, v in lengths.items()}})
    diff_a = abs(lengths["back"] - lengths["front"])
    diff_b = abs(lengths["left"] - lengths["right"])
    rgb_diffs = _rgb_quad_rect_diffs(record)
    if rgb_diffs is not None:
        # K1 published the gap-anchored quad: the MEASURED (RGB) quad must pass the same gate
        diff_a, diff_b = max(diff_a, rgb_diffs[0]), max(diff_b, rgb_diffs[1])
    angles = corner_angles_deg(points)
    max_dev = max(abs(a - 90.0) for a in angles.values())
    lengths_ok = diff_a <= tol_mm and diff_b <= tol_mm
    angles_ok = max_dev <= max_corner_dev_deg
    result = {"enabled": True, "tol_mm": float(tol_mm),
              "lengths_mm": {k: round(v, 3) for k, v in lengths.items()},
              "diffs_mm": {"A": round(diff_a, 3), "B": round(diff_b, 3)},
              "pairs": {"A": ["back", "front"], "B": ["left", "right"]},
              **({"rgb_quad_diffs_mm": {"A": round(rgb_diffs[0], 3), "B": round(rgb_diffs[1], 3)}}
                 if rgb_diffs is not None else {}),
              "corner_angles_deg": {k: round(v, 3) for k, v in angles.items()},
              "max_corner_dev_deg": round(max_dev, 3),
              "corner_dev_limit_deg": float(max_corner_dev_deg),
              "pass": lengths_ok and angles_ok}
    if not lengths_ok:
        raise Reject("rectangle",
                     [f"rectangle_inconsistent(A={diff_a:.2f}mm,B={diff_b:.2f}mm)"],
                     retryable=True, details=result)
    if not angles_ok:
        raise Reject("rectangle", [f"rectangle_skewed(max_dev={max_dev:.2f}deg)"],
                     retryable=True, details=result)
    return result


# ---------------------------------------------------------------- depth edge

# Seam-crossed edges per published axis (rule names), and the short-side
# pairing (anchor-side corner <-> other-side corner on the same short side).
SEAM_CROSSED_EDGES = {
    "bf": {"back": ("back_left", "back_right"), "front": ("front_left", "front_right")},
    "lr": {"left": ("back_left", "front_left"), "right": ("back_right", "front_right")},
}
SHORT_SIDE_PARTNER = {
    "bf": {"back_left": "front_left", "front_left": "back_left",
           "back_right": "front_right", "front_right": "back_right"},
    "lr": {"back_left": "back_right", "back_right": "back_left",
           "front_left": "front_right", "front_right": "front_left"},
}
_SEAM_FIELD_PATHS = {
    "seam.endpoints_px": ("endpoints_px",),
    "seam.estimate.estimate.estimate_endpoints_px": ("estimate", "estimate",
                                                     "estimate_endpoints_px"),
    "seam.estimate.prior_endpoints_px": ("estimate", "prior_endpoints_px"),
}


class _DepthEdgeSkip(Exception):
    """The depth edge fix does not apply (reason recorded, record unchanged).
    ``info`` carries the diagnostics gathered before the skip."""

    def __init__(self, reason: str, info: dict = None):
        super().__init__(reason)
        self.info = dict(info or {})


def camera_from_manifest(manifest):
    """(K 3x3, dist or None) from the manifest camera_info (same convention as
    autolabel_v2.fold_edge: plumb_bob/rational_polynomial d, <= 8 coeffs), or
    None when unusable."""
    import numpy as np  # noqa: PLC0415

    info = (manifest or {}).get("camera_info") or {}
    try:
        k = np.asarray(info.get("k"), dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        return None
    dist = None
    if str(info.get("distortion_model", "plumb_bob")) in ("plumb_bob", "rational_polynomial"):
        try:
            d = np.asarray(info.get("d") or [], dtype=np.float64).ravel()[:8]
        except (TypeError, ValueError):
            return None
        if d.size and np.any(d != 0.0):
            if not np.all(np.isfinite(d)):
                return None
            dist = np.concatenate([d, np.zeros(max(0, 5 - d.size))]) if d.size < 5 else d
    return k, dist


def project_to_px(xyz, camera):
    """Camera-frame points (N,3) -> distorted pixels (N,2)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    k, dist = camera
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k,
                              dist if dist is not None else np.zeros(5))
    return uv.reshape(-1, 2)


def backproject_to_plane(uv, camera, n, d):
    """Pixels (N,2) -> rays (undistorted) intersected with n.x + d = 0."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    k, dist = camera
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if dist is not None:
        xy = cv2.undistortPointsIter(
            uv.reshape(-1, 1, 2), k, dist, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)).reshape(-1, 2)
    else:
        xy = np.stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1]], axis=1)
    rays = np.concatenate([xy, np.ones((len(uv), 1))], axis=1)
    den = rays @ n
    if np.any(np.abs(den) < 1e-9):
        raise _DepthEdgeSkip("depth_edge_degenerate(ray parallel to plane)")
    t = -d / den
    if np.any(t <= 0):
        raise _DepthEdgeSkip("depth_edge_degenerate(plane behind camera)")
    return rays * t[:, None]


def choose_anchor(edges: dict, tie: float = DEPTH_EDGE_AMBIG_TIE):
    """Pick the anchor among the two seam-crossed edges.

    ``edges``: {name: {"ambiguous_frac": float|None, "distance_m": float}}.
    Lower ambiguity wins when both are known and differ by >= ``tie``; else
    the edge farther from the camera. Returns (anchor_name, reason). Raises
    _DepthEdgeSkip("depth_edge_anchor_ambiguous") when both edges are
    ambiguous (>= 0.3) and tie: no trustworthy anchor.
    """
    (name_a, a), (name_b, b) = sorted(edges.items())
    fa, fb = a.get("ambiguous_frac"), b.get("ambiguous_frac")
    if (fa is not None and fb is not None and min(fa, fb) >= DEPTH_EDGE_AMBIG_BOTH
            and abs(fa - fb) < tie):
        raise _DepthEdgeSkip(f"depth_edge_anchor_ambiguous({name_a}={fa:.3f},"
                             f"{name_b}={fb:.3f})")
    if fa is not None and fb is not None and abs(fa - fb) >= tie:
        anchor = name_a if fa < fb else name_b
        return anchor, (f"lower_ambiguity({name_a}={fa:.3f},{name_b}={fb:.3f})")
    anchor = name_a if a["distance_m"] >= b["distance_m"] else name_b
    why = ("ambiguity_tie" if fa is not None and fb is not None else "ambiguity_unknown")
    return anchor, (f"{why}->farther({name_a}={a['distance_m']:.4f}m,"
                    f"{name_b}={b['distance_m']:.4f}m)")


def _edge_ambiguity(record: dict, gray, corners_px: dict, pair) -> tuple:
    """(ambiguous_frac|None, source) of one edge: re-sampled on the image with
    autolabel_v2.corner_checks.edge_ambiguous_frac when available, else the
    record's per-corner values (each = max of its two edges, so the min over
    the edge's endpoints bounds the edge value from above)."""
    if gray is not None:
        try:
            from ..core.corner_checks import edge_ambiguous_frac  # noqa: PLC0415
            refine = (((record.get("provenance") or {}).get("config") or {})
                      .get("refine") or {})
            frac, _ = edge_ambiguous_frac(
                gray, corners_px[pair[0]], corners_px[pair[1]],
                int(refine.get("profile_half_width_px", 12)),
                float(refine.get("sample_spacing_px", 6.0)),
                float(refine.get("min_gradient", 4.0)))
            if frac is not None:
                return float(frac), "image_resampled"
        except Exception:  # noqa: BLE001 — fall back to the record values
            pass
    corners = (((record.get("checks") or {}).get("corner_flags") or {}).get("corners") or {})
    values = []
    for name in pair:
        value = (corners.get(name) or {}).get("ambiguous_frac")
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if len(values) == 2:
        return min(values), "record_corner_min"
    return None, "unavailable"


def _plane_nd(plane):
    import numpy as np  # noqa: PLC0415

    if not isinstance(plane, dict):
        raise _DepthEdgeSkip("depth_edge_no_plane")
    try:
        n = np.asarray(plane.get("normal"), dtype=np.float64).reshape(3)
        d = float(plane.get("offset_m"))
    except (TypeError, ValueError):
        raise _DepthEdgeSkip("depth_edge_no_plane") from None
    norm = float(np.linalg.norm(n))
    if not (np.all(np.isfinite(n)) and math.isfinite(d)) or norm < 1e-9:
        raise _DepthEdgeSkip("depth_edge_no_plane")
    n, d = n / norm, d / norm
    if d < 0:
        n, d = -n, -d
    return n, d


def _boundary_gap(b_band, b_lo: float, b_hi: float, bin_m: float = 0.001):
    """Contiguity of the flat top between the centre and each chosen depth
    boundary: a run of 1 mm bins longer than DEPTH_EDGE_GAP_MM whose counts are
    all < DEPTH_EDGE_GAP_DENSITY x the median bin count (over [b_lo, b_hi])
    means the percentile boundary jumped across a gap onto a neighbouring
    same-height surface. Returns {"side", "start_mm", "length_mm"} or None."""
    import numpy as np  # noqa: PLC0415

    edges = np.arange(b_lo, b_hi + bin_m, bin_m)
    if len(edges) < 4:
        return None
    counts, _ = np.histogram(b_band, bins=edges)
    median = float(np.median(counts))
    if median <= 0:
        return None
    low = counts < DEPTH_EDGE_GAP_DENSITY * median
    centres = 0.5 * (edges[:-1] + edges[1:])
    need = int(math.floor(DEPTH_EDGE_GAP_MM / (1000 * bin_m))) + 1  # > 3 mm
    for side, order in (("other", np.flatnonzero(centres < 0)[::-1]),
                        ("anchor", np.flatnonzero(centres >= 0))):
        run = 0
        for position, index in enumerate(order[:-1]):  # boundary bin is partial
            run = run + 1 if low[index] else 0
            if run >= need:
                start = centres[order[position - (need - 1)]]  # gap start (centre side)
                return {"side": side, "start_mm": round(1000 * float(start), 2),
                        "length_mm": float(need)}
    return None


def _line_b_at(p_ab, q_ab, a0: float = 0.0) -> float:
    """b of the (a,b)-plane line through p,q at a = a0."""
    (pa, pb), (qa, qb) = p_ab, q_ab
    if abs(qa - pa) < 1e-9:
        raise _DepthEdgeSkip("depth_edge_degenerate(edge along u_seam)")
    return pb + (a0 - pa) * (qb - pb) / (qa - pa)


def depth_edge_fix(record: dict, seam, cloud_xyz, camera, gray=None,
                   min_points: int = DEPTH_EDGE_MIN_POINTS,
                   min_move_mm: float = DEPTH_EDGE_MIN_MOVE_MM):
    """Depth-placed replacement of the ambiguous seam-crossed top-face edge(s).

    ``record``: box_autolabel/v5 record (rule names; top_face.plane,
    corners_px, corners_xyz_camera_m). ``seam``: :func:`published_seam_line`
    output, or None/axis-less (the quad's own geometry then picks the
    candidate edges, only for a skewed quad). ``cloud_xyz``: (N,3)
    camera-frame cloud. ``camera``: (K, dist).
    Returns ``(record_out, seam_out, info)``; record_out/seam_out are new
    objects (the inputs are never mutated) and equal copies when not applied.
    Raises _DepthEdgeSkip(reason) when the fix cannot be evaluated.
    """
    info = {"applied": False}
    try:
        return _depth_edge_fix_body(record, seam, cloud_xyz, camera, gray,
                                    min_points, min_move_mm, info)
    except _DepthEdgeSkip as skip:
        skip.info = {**info, **skip.info, "applied": False}
        raise


def _rect_diffs_mm(xyz: dict) -> dict:
    """{"A": |back-front|, "B": |left-right|} 3-D side differences in mm."""
    import numpy as np  # noqa: PLC0415

    length = {side: 1000.0 * float(np.linalg.norm(np.asarray(xyz[b]) - np.asarray(xyz[a])))
              for side, (a, b) in RECT_SIDES.items()}
    return {"A": abs(length["back"] - length["front"]),
            "B": abs(length["left"] - length["right"])}


def _quad_dims_mm(xyz: dict):
    """(long, short) mean opposite-side lengths in mm."""
    import numpy as np  # noqa: PLC0415

    length = {side: 1000.0 * float(np.linalg.norm(np.asarray(xyz[b]) - np.asarray(xyz[a])))
              for side, (a, b) in RECT_SIDES.items()}
    pair = (0.5 * (length["back"] + length["front"]), 0.5 * (length["left"] + length["right"]))
    return max(pair), min(pair)


def _class_match(xyz: dict):
    """Known box class whose (long, short) dims the quad matches within
    DEPTH_EDGE_CLASS_TOL_MM, or None."""
    long_mm, short_mm = _quad_dims_mm(xyz)
    profile, _ = box_profiles.classify(long_mm, short_mm)
    return None if profile.is_generic else profile.name


def _geometry_axes(xyz: dict) -> list:
    """Candidate seam axes from the quad alone: the axis whose seam-crossed
    edges are the two LONG sides ("bf" = back/front); both when square-ish."""
    import numpy as np  # noqa: PLC0415

    def mean_len(edge_pair):
        return float(np.mean([np.linalg.norm(xyz[b] - xyz[a])
                              for a, b in edge_pair]))

    len_bf = mean_len(SEAM_CROSSED_EDGES["bf"].values())
    len_lr = mean_len(SEAM_CROSSED_EDGES["lr"].values())
    if len_bf > DEPTH_EDGE_SQUARE_RATIO * len_lr:
        return ["bf"]
    if len_lr > DEPTH_EDGE_SQUARE_RATIO * len_bf:
        return ["lr"]
    return ["bf", "lr"]


def _sam_box(record: dict):
    """SAM3 box (x0, y0, x1, y1) of the record, or None."""
    sam3 = (record.get("provenance") or {}).get("sam3") or {}
    box = sam3.get("box_xyxy")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return tuple(float(v) for v in box)
        except (TypeError, ValueError):
            pass
    return _sam3_box_px(record)


def _inside_dilated_box(points_px, box, frac: float = DEPTH_EDGE_SAM_BOX_DILATE) -> bool:
    x0, y0, x1, y1 = box
    mx, my = frac * (x1 - x0), frac * (y1 - y0)
    return all(x0 - mx <= float(p[0]) <= x1 + mx and y0 - my <= float(p[1]) <= y1 + my
               for p in points_px)


def _edge_support(pa, pb, a_lo: float, a_hi: float, b_target: float) -> float:
    """Fraction of 5 mm bins along the edge (minus 3 mm at each end) whose
    flat-top depth boundary (p1 of b) lies within DEPTH_EDGE_SUPPORT_DEV_MM of
    ``b_target`` with >= DEPTH_EDGE_SUPPORT_MIN_PTS points: how well a LINE
    parallel to the anchor at b_target is supported by the depth along the
    whole edge (0 when too short to bin)."""
    import numpy as np  # noqa: PLC0415

    step = DEPTH_EDGE_SUPPORT_BIN_MM / 1000.0
    lo, hi = min(a_lo, a_hi) + 0.003, max(a_lo, a_hi) - 0.003
    if hi - lo < 2 * step:
        return 0.0
    edges = np.arange(lo, hi + 1e-9, step)
    ok = 0
    for left, right in zip(edges[:-1], edges[1:]):
        sel = pb[(pa >= left) & (pa < right)]
        if len(sel) >= DEPTH_EDGE_SUPPORT_MIN_PTS and \
                abs(float(np.percentile(sel, 1.0)) - b_target) * 1000 <= DEPTH_EDGE_SUPPORT_DEV_MM:
            ok += 1
    return ok / max(1, len(edges) - 1)


def _edge_support_ex(pa, pb, a_lo: float, a_hi: float, b_target: float,
                     anchor_angle_deg: float = 0.0) -> dict:
    """R8(c): :func:`_edge_support` plus a RANSAC criterion.

    ``fraction`` passes when the raw fraction >= DEPTH_EDGE_SUPPORT_MIN_FRAC.
    ``ransac`` passes when a line fitted (exhaustive 2-point RANSAC, then least
    squares on its inliers) to the per-bin depth-boundary points (p1 of b in
    each 5 mm bin) has >= DEPTH_EDGE_RANSAC_MIN_INLIER inliers (of ALL bins,
    tol DEPTH_EDGE_RANSAC_TOL_MM), is within DEPTH_EDGE_RANSAC_MAX_ANGLE_DEG of
    the anchor edge and passes within DEPTH_EDGE_SUPPORT_DEV_MM of
    ``b_target`` mid-edge. ``effective`` is the value the support gates use:
    the raw fraction, lifted to the fraction gate when only RANSAC passes.
    """
    import numpy as np  # noqa: PLC0415

    frac = _edge_support(pa, pb, a_lo, a_hi, b_target)
    out = {"frac": frac, "ransac_inlier_frac": 0.0, "ransac_angle_deg": None,
           "ransac_ok": False}
    step = DEPTH_EDGE_SUPPORT_BIN_MM / 1000.0
    lo, hi = min(a_lo, a_hi) + 0.003, max(a_lo, a_hi) - 0.003
    if hi - lo >= 2 * step:
        edges = np.arange(lo, hi + 1e-9, step)
        n_bins = max(1, len(edges) - 1)
        pts = []
        for left, right in zip(edges[:-1], edges[1:]):
            sel = pb[(pa >= left) & (pa < right)]
            if len(sel) >= DEPTH_EDGE_SUPPORT_MIN_PTS:
                pts.append((0.5 * (left + right), float(np.percentile(sel, 1.0))))
        tol = DEPTH_EDGE_RANSAC_TOL_MM / 1000.0
        if len(pts) >= 3:
            arr = np.asarray(pts)
            best_n, best_line = 0, None
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    da = arr[j, 0] - arr[i, 0]
                    if abs(da) < 1e-9:
                        continue
                    m = (arr[j, 1] - arr[i, 1]) / da
                    c = arr[i, 1] - m * arr[i, 0]
                    inl = np.abs(arr[:, 1] - (m * arr[:, 0] + c)) <= tol
                    if int(inl.sum()) > best_n and int(inl.sum()) >= 2:
                        best_n, best_line, best_inl = int(inl.sum()), (m, c), inl
            if best_line is not None:
                if best_n >= 3:
                    m, c = np.polyfit(arr[best_inl, 0], arr[best_inl, 1], 1)
                    inl = np.abs(arr[:, 1] - (m * arr[:, 0] + c)) <= tol
                    if int(inl.sum()) >= best_n:
                        best_n, best_line = int(inl.sum()), (float(m), float(c))
                m, c = best_line
                angle = math.degrees(math.atan(m))
                inlier_frac = best_n / n_bins
                a_mid = 0.5 * (lo + hi)
                near_target = abs(m * a_mid + c - b_target) * 1000 <= DEPTH_EDGE_SUPPORT_DEV_MM
                out["ransac_inlier_frac"] = round(inlier_frac, 3)
                out["ransac_angle_deg"] = round(angle - anchor_angle_deg, 3)
                out["ransac_ok"] = bool(
                    "c" not in rules_off() and inlier_frac >= DEPTH_EDGE_RANSAC_MIN_INLIER
                    and abs(angle - anchor_angle_deg) <= DEPTH_EDGE_RANSAC_MAX_ANGLE_DEG
                    and near_target)
    if frac >= DEPTH_EDGE_SUPPORT_MIN_FRAC:
        out["criterion"], out["effective"] = "fraction", frac
    elif out["ransac_ok"]:
        out["criterion"], out["effective"] = "ransac", DEPTH_EDGE_SUPPORT_MIN_FRAC
    else:
        out["criterion"], out["effective"] = "none", frac
    return out


def _place_edge(ctx: dict, axis: str, anchor: str, xyz: dict, px: dict,
                extended_ok: bool) -> dict:
    """One depth placement: keep the ``anchor`` seam-crossed edge, re-place the
    other edge of ``axis`` parallel to it at the other-side depth boundary.

    ``xyz``/``px``: current (on-plane) corners. Returns {"info", "axis",
    "anchor", "other", "moved": {name: (xyz, px)}, "support", "noop"}; raises
    _DepthEdgeSkip(reason, info) when the placement does not apply.
    """
    other = next(name for name in SEAM_CROSSED_EDGES[axis] if name != anchor)
    local = {"axis": axis, "anchor": anchor, "moved_edge": other}
    try:
        return _place_edge_body(ctx, axis, anchor, other, xyz, px, extended_ok, local)
    except _DepthEdgeSkip as skip:
        skip.info = {**local, **skip.info}
        raise


def _place_edge_body(ctx, axis, anchor, other, xyz, px, extended_ok, local) -> dict:
    import numpy as np  # noqa: PLC0415

    n, d, camera = ctx["n"], ctx["d"], ctx["camera"]
    edges = SEAM_CROSSED_EDGES[axis]
    a0, a1 = edges[anchor]
    partner = SHORT_SIDE_PARTNER[axis]

    # In-plane frame: origin = quad centre, u_long along the anchor edge,
    # u_seam = n x u_long pointing from the centre toward the anchor edge.
    origin = np.mean([xyz[k] for k in RING], axis=0)
    origin = origin - n * (float(origin @ n) + d)
    u_ref = ctx.get("u_ref")
    if u_ref is not None and "a" not in rules_off():
        # R8(a): second pair -> the frame is perpendicular to the FIRST anchor
        # axis, so the moved edge is placed perpendicular to it (not parallel
        # to this anchor edge, whose own direction may be off).
        u_long = np.cross(n, u_ref)
    else:
        u_long = xyz[a1] - xyz[a0]
    u_long = u_long - n * float(u_long @ n)
    if np.linalg.norm(u_long) < 1e-6:
        raise _DepthEdgeSkip("depth_edge_degenerate(anchor edge)")
    u_long /= np.linalg.norm(u_long)
    u_seam = np.cross(n, u_long)
    if float((0.5 * (xyz[a0] + xyz[a1]) - origin) @ u_seam) < 0:
        u_seam = -u_seam

    def to_ab(p):
        q = np.asarray(p, dtype=np.float64) - origin
        return np.array([float(q @ u_long), float(q @ u_seam)])

    def from_ab(ab_):
        return origin + ab_[0] * u_long + ab_[1] * u_seam

    ab = {k: to_ab(xyz[k]) for k in RING}
    radius = max(DEPTH_EDGE_RADIUS_M, max(np.linalg.norm(xyz[k] - origin) for k in RING)
                 + DEPTH_EDGE_MARGIN_M)
    a_lim = max(DEPTH_EDGE_HALF_M, max(abs(ab[k][0]) for k in RING) + DEPTH_EDGE_MARGIN_M)
    b_lim = max(DEPTH_EDGE_HALF_M, max(abs(ab[k][1]) for k in RING) + DEPTH_EDGE_MARGIN_M)

    cloud = ctx["cloud_plane"]
    rel = cloud - origin
    keep = np.linalg.norm(rel, axis=1) < radius
    rel = rel[keep]
    pa, pb = rel @ u_long, rel @ u_seam
    keep = (np.abs(pa) < a_lim) & (np.abs(pb) < b_lim)
    pa, pb = pa[keep], pb[keep]
    band = np.abs(pa) < DEPTH_EDGE_BAND_M
    n_band = int(band.sum())
    local["n_band_points"] = n_band
    if n_band < int(ctx["min_points"]):
        raise _DepthEdgeSkip(f"depth_edge_insufficient_points({n_band}<{int(ctx['min_points'])})")
    b_lo, b_hi = (float(v) for v in np.percentile(pb[band], DEPTH_EDGE_PCT))
    local["depth_boundaries_mm"] = {"anchor_side": round(1000 * b_hi, 3),
                                    "other_side": round(1000 * b_lo, 3)}
    gap = _boundary_gap(pb[band], b_lo, b_hi)
    if gap is not None:
        local["boundary_gap"] = gap
        raise _DepthEdgeSkip(f"depth_edge_boundary_gap({gap['side']} side, "
                             f"{gap['length_mm']:.1f}mm at b={gap['start_mm']:.1f}mm)")
    if b_hi > b_lim - DEPTH_EDGE_CLIP_MM / 1000 or b_lo < -b_lim + DEPTH_EDGE_CLIP_MM / 1000:
        raise _DepthEdgeSkip("depth_edge_boundary_clipped")

    b_anchor = _line_b_at(ab[a0], ab[a1])
    o0, o1 = partner[a0], partner[a1]
    b_other_rgb = _line_b_at(ab[o0], ab[o1])
    delta = b_anchor - b_hi                   # >0: RGB edge outside the depth edge
    # Placement (lead decision): the anchor stays the RGB FOLD edge, the moved
    # edge is the DEPTH boundary itself (no offset; delta recorded only).
    b_target = b_lo                           # other edge AT its depth boundary
    local["placement"] = DEPTH_EDGE_PLACEMENT
    depth_width = b_hi - b_lo
    rgb_width = b_anchor - b_other_rgb
    local.update({"depth_width_mm": round(1000 * depth_width, 3),
                  "rgb_width_mm": round(1000 * rgb_width, 3),
                  "delta_mm": round(1000 * delta, 3),
                  "new_width_mm": round(1000 * (b_anchor - b_target), 3)})
    d_anchor = ab[a1] - ab[a0]
    anchor_angle = (math.degrees(math.atan(d_anchor[1] / d_anchor[0]))
                    if abs(d_anchor[0]) > 1e-9 else 90.0)
    sup = _edge_support_ex(pa, pb, min(ab[a0][0], ab[a1][0]), max(ab[a0][0], ab[a1][0]),
                           b_target, anchor_angle)
    support = sup["effective"]
    local["support_frac"] = round(sup["frac"], 3)
    local["support_criterion"] = sup["criterion"]
    local["support_ransac"] = {"inlier_frac": sup["ransac_inlier_frac"],
                               "angle_deg": sup["ransac_angle_deg"], "ok": sup["ransac_ok"]}
    disagreement = {"centre": round(1000 * abs(b_other_rgb - b_target), 3),
                    o0: round(1000 * abs(ab[o0][1] - b_target), 3),
                    o1: round(1000 * abs(ab[o1][1] - b_target), 3)}
    local["disagreement_mm"] = disagreement
    if abs(delta) * 1000 > DEPTH_EDGE_MAX_DELTA_MM:
        raise _DepthEdgeSkip(f"depth_edge_anchor_offset_implausible({1000 * delta:.2f}mm)")
    centre_move = disagreement["centre"]
    wide = extended_ok and centre_move > DEPTH_EDGE_MAX_CENTRE_MOVE_MM
    width_bound = DEPTH_EDGE_MAX_WIDTH_DIFF_EXT_MM if wide else DEPTH_EDGE_MAX_WIDTH_DIFF_MM
    if abs(depth_width - rgb_width) * 1000 > width_bound:
        raise _DepthEdgeSkip(f"depth_edge_implausible(depth {1000 * depth_width:.1f}mm vs "
                             f"rgb {1000 * rgb_width:.1f}mm)")
    if centre_move > (DEPTH_EDGE_SUPPORTED_CENTRE_MOVE_MM if extended_ok
                      else DEPTH_EDGE_MAX_CENTRE_MOVE_MM):
        raise _DepthEdgeSkip(f"depth_edge_move_too_large(centre {centre_move:.2f}mm > "
                             f"{(DEPTH_EDGE_SUPPORTED_CENTRE_MOVE_MM if extended_ok else DEPTH_EDGE_MAX_CENTRE_MOVE_MM):g}mm)")
    if max(disagreement.values()) <= float(ctx["min_move_mm"]):
        local["reason"] = f"rgb_agrees_with_depth(<= {ctx['min_move_mm']:g}mm)"
        return {"info": local, "axis": axis, "anchor": anchor, "other": other,
                "moved": {}, "support": support, "noop": True}

    # New other-edge corners: line b = b_target meets the short-side RGB lines.
    moved = {}
    for anchor_corner in (a0, a1):
        oc = partner[anchor_corner]
        pa_, qa_ = ab[anchor_corner], ab[oc]
        if abs(qa_[1] - pa_[1]) < 1e-6:
            raise _DepthEdgeSkip("depth_edge_degenerate(short side parallel to edge)")
        t = (b_target - pa_[1]) / (qa_[1] - pa_[1])
        if not (0.5 < t < 1.5):
            raise _DepthEdgeSkip(f"depth_edge_degenerate(short side t={t:.3f})")
        moved[oc] = from_ab(pa_ + t * (qa_ - pa_))
    names = sorted(moved)
    new_px_arr = project_to_px(np.stack([moved[k] for k in names]), camera)
    new_px = {k: new_px_arr[i] for i, k in enumerate(names)}
    local["moved_mm"] = {k: round(1000 * float(np.linalg.norm(moved[k] - xyz[k])), 3)
                         for k in names}
    local["moved_px"] = {k: round(float(np.linalg.norm(new_px[k] - px[k])), 3) for k in names}
    if max(local["moved_mm"].values()) > DEPTH_EDGE_MAX_CORNER_MOVE_MM:
        raise _DepthEdgeSkip(f"depth_edge_move_too_large(corner "
                             f"{max(local['moved_mm'].values()):.2f}mm > "
                             f"{DEPTH_EDGE_MAX_CORNER_MOVE_MM:g}mm)")
    if centre_move > DEPTH_EDGE_MAX_CENTRE_MOVE_MM:
        # Support-based guard (replaces the fixed cap up to 10 mm): contiguous
        # boundary (the gap check above passed), supported along the whole
        # edge, moved edge inside the SAM3 box dilated by 10 %.
        box = ctx.get("sam_box")
        inside = bool(box is not None
                      and _inside_dilated_box([new_px[k] for k in names], box))
        local["extended_move"] = {"centre_mm": centre_move, "support_frac": round(support, 3),
                                  "inside_sam_box": inside}
        if support < DEPTH_EDGE_SUPPORT_MIN_FRAC or not inside:
            why = ("no_sam_box" if box is None else
                   "outside_sam_box" if not inside else
                   f"support {support:.2f}<{DEPTH_EDGE_SUPPORT_MIN_FRAC:g}")
            raise _DepthEdgeSkip(f"depth_edge_move_too_large(centre {centre_move:.2f}mm > "
                                 f"{DEPTH_EDGE_MAX_CENTRE_MOVE_MM:g}mm; unsupported: {why})")
    # R8(d): never put the moved edge more than 5 mm farther from the quad
    # centroid (the frame origin) than the original edge's midpoint.
    # (the other edge lies at negative b: u_seam points from the centroid to the anchor)
    outward = 1000 * (0.5 * (ab[o0][1] + ab[o1][1]) - b_target)
    local["outward_mm"] = round(outward, 3)
    if outward > DEPTH_EDGE_MAX_OUTWARD_MM and "d" not in rules_off():
        raise _DepthEdgeSkip(f"depth_edge_moved_outward({outward:.2f}mm > "
                             f"{DEPTH_EDGE_MAX_OUTWARD_MM:g}mm)")

    local["reason"] = (f"rgb_{other}_edge_disagrees_with_depth("
                       f"{max(disagreement.values()):.2f}mm>{ctx['min_move_mm']:g}mm)")
    return {"info": local, "axis": axis, "anchor": anchor, "other": other,
            "moved": {k: (moved[k], new_px[k]) for k in names},
            "support": support, "noop": False}


def _apply_moved(xyz: dict, px: dict, res: dict):
    xyz2, px2 = dict(xyz), dict(px)
    for k, (x_, p_) in res["moved"].items():
        xyz2[k], px2[k] = x_, p_
    return xyz2, px2


def _pick_best(results: list, xyz: dict, px: dict) -> dict:
    """Best of several candidate placements: better depth support AND smaller
    rectangle residual; when the two criteria conflict the smaller residual
    (the gate's own measure) wins unless the residuals are within 0.3 mm."""
    for res in results:
        xyz_after, _ = _apply_moved(xyz, px, res)
        res["resid_mm"] = max(_rect_diffs_mm(xyz_after).values())
    best = results[0]
    for res in results[1:]:
        d_resid = res["resid_mm"] - best["resid_mm"]
        d_support = res["support"] - best["support"]
        if (d_resid <= 0 and d_support >= 0 and (d_resid < 0 or d_support > 0)):
            best = res
        elif d_resid < -0.3 or (abs(d_resid) <= 0.3 and d_support > 0):
            best = res
    return best


def _seam_after(seam, axis, xyz_old, px_old, xyz_new, px_new, n, d, camera, info):
    """Recompute the published seam as the projective midline of the FIXED
    quad (through the diagonal intersection toward the meeting point of the
    two short sides, in the top plane); its ends lie on the two seam-crossed
    edges. Returns (new_end_px {seam_left, seam_right}, downgraded)."""
    import numpy as np  # noqa: PLC0415

    edges = SEAM_CROSSED_EDGES[axis]
    e_first, e_second = list(edges)
    partner = SHORT_SIDE_PARTNER[axis]
    a0, a1 = edges[e_first]
    origin = np.mean([xyz_new[k] for k in RING], axis=0)
    origin = origin - n * (float(origin @ n) + d)
    u_long = xyz_new[a1] - xyz_new[a0]
    u_long = u_long - n * float(u_long @ n)
    u_long /= np.linalg.norm(u_long)
    u_seam = np.cross(n, u_long)
    if float((np.mean([xyz_new[k] for k in edges[e_second]], axis=0) - origin) @ u_seam) > 0:
        u_seam = -u_seam      # e_first side is at +b (as in the anchor frame)

    def to_ab(p):
        q = np.asarray(p, dtype=np.float64) - origin
        return np.array([float(q @ u_long), float(q @ u_seam)])

    def from_ab(ab_):
        return origin + ab_[0] * u_long + ab_[1] * u_seam

    def cross2(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    def meet(p, r, q, s_):
        den = cross2(r, s_)
        if abs(den) < 1e-12:
            return None
        return p + (cross2(q - p, s_) / den) * r

    ab_new = {k: to_ab(xyz_new[k]) for k in RING}
    bl, br, fr, fl = (ab_new[k] for k in RING)
    centre = meet(bl, fr - bl, br, fl - br)
    if centre is None:
        raise _DepthEdgeSkip("depth_edge_degenerate(quad diagonals)")
    d0 = ab_new[partner[a0]] - ab_new[a0]
    d1 = ab_new[partner[a1]] - ab_new[a1]
    vanish = meet(ab_new[a0], d0, ab_new[a1], d1)
    if vanish is None or np.linalg.norm(vanish - centre) > 1e3:
        direction = d0 / np.linalg.norm(d0) + d1 / np.linalg.norm(d1)
    else:
        direction = vanish - centre
        if float(direction @ d0) < 0:
            direction = -direction
    direction = direction / np.linalg.norm(direction)
    ends_ab = {}
    for name in (e_first, e_second):
        p, q = (ab_new[c] for c in edges[name])
        point = meet(centre, direction, p, q - p)
        if point is None:
            raise _DepthEdgeSkip("depth_edge_degenerate(midline parallel to edges)")
        ends_ab[name] = point

    ends = {key: np.asarray(seam[key], dtype=np.float64) for key in ("seam_left", "seam_right")}

    def edge_dist_old(point, name):
        return point_segment_distance(point, px_old[edges[name][0]], px_old[edges[name][1]])

    keys = list(ends)
    straight = (edge_dist_old(ends[keys[0]], e_first) + edge_dist_old(ends[keys[1]], e_second))
    crossed = (edge_dist_old(ends[keys[1]], e_first) + edge_dist_old(ends[keys[0]], e_second))
    on_edge = ({keys[0]: e_first, keys[1]: e_second} if straight <= crossed
               else {keys[1]: e_first, keys[0]: e_second})
    s_xyz = backproject_to_plane(np.stack([ends["seam_left"], ends["seam_right"]]),
                                 camera, n, d)
    old_ab = {"seam_left": to_ab(s_xyz[0]), "seam_right": to_ab(s_xyz[1])}
    normal2 = np.array([-direction[1], direction[0]])
    lateral = {key: round(1000 * abs(float((old_ab[key] - centre) @ normal2)), 3)
               for key in old_ab}
    new_end_px = {key: project_to_px(from_ab(ends_ab[on_edge[key]]).reshape(1, 3), camera)[0]
                  for key in ends}
    edge_dist = {}
    for key in ends:
        c0, c1 = edges[on_edge[key]]
        edge_dist[key] = point_segment_distance(new_end_px[key], px_new[c0], px_new[c1])
    lateral_max = max(lateral.values())
    downgraded = bool(seam.get("status") == "ok" and seam.get("observed")
                      and lateral_max > SEAM_DOWNGRADE_SHIFT_MM)
    info["seam_line_recomputed"] = True
    info["seam_endpoints_before"] = {k: [float(v) for v in ends[k]] for k in ends}
    info["seam_endpoints_after"] = {k: [round(float(v), 2) for v in new_end_px[k]]
                                    for k in ends}
    info["seam_on_moved_edge"] = next((k for k, v in on_edge.items()
                                       if v == info.get("moved_edge")), keys[0])
    info["seam_lateral_shift_mm"] = {**lateral, "max": lateral_max}
    info["seam_edge_distance_px"] = {k: round(float(v), 3) for k, v in edge_dist.items()}
    info["seam_status_downgraded"] = downgraded
    return new_end_px, downgraded


def _depth_edge_fix_body(record, seam, cloud_xyz, camera, gray, min_points,
                         min_move_mm, info):
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    axis = (seam or {}).get("axis")
    seam_axis_ok = axis in SEAM_CROSSED_EDGES and bool(seam) \
        and "seam_left" in seam and "seam_right" in seam
    top = record.get("top_face") or {}
    n, d = _plane_nd(top.get("plane"))
    px_in = top.get("corners_px") or {}
    xyz_in = top.get("corners_xyz_camera_m") or {}
    try:
        px = {k: np.asarray(px_in[k], dtype=np.float64).reshape(2) for k in RING}
        xyz = {k: np.asarray(xyz_in[k], dtype=np.float64).reshape(3) for k in RING}
    except (KeyError, TypeError, ValueError):
        raise _DepthEdgeSkip("depth_edge_no_corners") from None
    if not all(np.all(np.isfinite(v)) for v in (*px.values(), *xyz.values())):
        raise _DepthEdgeSkip("depth_edge_no_corners")
    # Corners exactly on the plane (the record's are pixel rays on it, rounded).
    xyz = {k: v - n * (float(v @ n) + d) for k, v in xyz.items()}
    xyz0, px0 = dict(xyz), dict(px)
    sides = [float(np.linalg.norm(xyz[RING[(i + 1) % 4]] - xyz[RING[i]])) for i in range(4)]
    long_side = max(0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3]))
    info["top_long_side_mm"] = round(1000 * long_side, 2)
    diffs0 = _rect_diffs_mm(xyz)
    info["rect_before_mm"] = {k: round(v, 3) for k, v in diffs0.items()}
    tol = DEFAULT_RECT_TOL_MM
    skewed = max(diffs0.values()) > tol
    reasons = []
    info["reasons"] = reasons
    # today's 150 mm split (classify_legacy: > 150 mm -> big, not validated)
    class_gate = not box_profiles.classify_legacy(long_side).depth_edge_validated
    if class_gate:
        if not skewed:
            raise _DepthEdgeSkip(f"depth_edge_class_not_validated(long side "
                                 f"{1000 * long_side:.1f}mm > "
                                 f"{1000 * DEPTH_EDGE_SMALL_MAX_LONG_M:.0f}mm)")
        reasons.append(f"class_gate_overridden_skewed(long side {1000 * long_side:.1f}mm, "
                       f"max diff {max(diffs0.values()):.2f}mm)")

    # --- candidate (axis, anchor) list of the first placement -------------
    if seam_axis_ok:
        info["mode"] = "seam"
        info["seam_axis"] = axis
        edges = SEAM_CROSSED_EDGES[axis]
        edge_eval = {}
        for name, pair in edges.items():
            frac, source = _edge_ambiguity(record, gray, px_in, pair)
            mid = 0.5 * (xyz[pair[0]] + xyz[pair[1]])
            edge_eval[name] = {"ambiguous_frac": None if frac is None else round(frac, 3),
                               "ambiguity_source": source,
                               "distance_m": round(float(np.linalg.norm(mid)), 4)}
        info["edges"] = edge_eval
        try:
            anchor, why = choose_anchor(edge_eval)
            candidates = [(axis, anchor, why)]
        except _DepthEdgeSkip as ambiguous:
            reasons.append("anchor_ambiguous->both_anchors_evaluated")
            candidates = [(axis, name, f"both_anchors({ambiguous})") for name in edges]
    else:
        if not skewed:
            reason = ("depth_edge_no_published_seam" if not seam else
                      f"depth_edge_no_seam_axis({axis})")
            raise _DepthEdgeSkip(reason)
        info["mode"] = "quad_geometry"
        reasons.append("no_published_seam->quad_geometry" if not seam
                       else f"no_seam_axis({axis})->quad_geometry")
        axes = _geometry_axes(xyz)
        info["geometry_axes"] = axes
        candidates = [(ax, name, "quad_geometry_both_anchors")
                      for ax in axes for name in SEAM_CROSSED_EDGES[ax]]

    cloud = np.asarray(cloud_xyz, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    cloud = cloud[np.abs(cloud @ n + d) < DEPTH_EDGE_PLANE_TOL_M]
    ctx = {"n": n, "d": d, "camera": camera, "cloud_plane": cloud, "min_points": min_points,
           "min_move_mm": min_move_mm, "sam_box": _sam_box(record)}

    def run(cands, xyz_s, px_s, ctx_run=None):
        results, skips = [], []
        for ax, anchor_name, why in cands:
            try:
                res = _place_edge(ctx_run or ctx, ax, anchor_name, xyz_s, px_s,
                                  extended_ok=True)
                res["anchor_reason"] = why
                results.append(res)
            except _DepthEdgeSkip as skip:
                skips.append((ax, anchor_name, skip))
        return results, skips

    def summary(res):
        return {"axis": res["axis"], "anchor": res["anchor"], "support_frac": round(
            res["support"], 3), "resid_mm": round(res.get("resid_mm", float("nan")), 3),
            "noop": res["noop"]}

    results, skips = run(candidates, xyz, px)
    if info["mode"] == "quad_geometry":
        weak = [(r["axis"], r["anchor"], r["support"]) for r in results
                if r["support"] < DEPTH_EDGE_GEOMETRY_MIN_SUPPORT and not r["noop"]]
        results = [r for r in results if r["noop"]
                   or r["support"] >= DEPTH_EDGE_GEOMETRY_MIN_SUPPORT]
        for ax, name, sup in weak:
            skips.append((ax, name, _DepthEdgeSkip(
                f"depth_edge_unsupported(support {sup:.2f}<{DEPTH_EDGE_GEOMETRY_MIN_SUPPORT:g})")))
    variants = [{"axis": a, "anchor": n_, "skipped": str(s)} for a, n_, s in skips]
    if not results:
        first = skips[0][2]
        text = str(first)
        if len(skips) > 1:
            text += " [other candidates: " + "; ".join(
                f"{a}/{n_}: {s}" for a, n_, s in skips[1:]) + "]"
        info["candidates"] = variants
        raise _DepthEdgeSkip(text, {**first.info, "candidates": variants})
    best = _pick_best(results, xyz, px)
    variants += [summary(r) for r in results]
    if len(results) > 1:
        info["candidates"] = variants
        reasons.append(f"best_of_{len(results)}: {best['axis']}/{best['anchor']}")
    elif variants and skips:
        info["candidates"] = variants
    first_info = best["info"]
    info.update({k: v for k, v in first_info.items()})
    info["anchor"], info["moved_edge"] = best["anchor"], best["other"]
    info["anchor_reason"] = best.get("anchor_reason")
    if "extended_move" in first_info:
        reasons.append("move_cap_extended_by_support")
    steps = [summary(best)]
    first_axis = best["axis"]
    xyz1, px1 = _apply_moved(xyz, px, best)
    moved_any = bool(best["moved"])
    moved_names = set(best["moved"])

    # --- (e) the OTHER pair still skewed: place it from depth too ----------
    if moved_any or best["noop"]:
        diffs1 = _rect_diffs_mm(xyz1)
        other_axis = "lr" if first_axis == "bf" else "bf"
        # The first placement makes its moved edge parallel to the anchor, so
        # what is left is the length difference of the SEAM-CROSSED edges (pair A
        # for bf, B for lr), caused by non-parallel short sides: those are the
        # OTHER edge pair, which is then placed from depth the same way.
        other_pair = "A" if first_axis == "bf" else "B"
        if diffs1[other_pair] > tol:
            cands2 = [(other_axis, name, "second_pair_both_anchors")
                      for name in SEAM_CROSSED_EDGES[other_axis]]
            # R8(a): the second pair must be perpendicular to the FIRST anchor axis.
            f0, f1 = SEAM_CROSSED_EDGES[first_axis][best["anchor"]]
            u_first = xyz1[f1] - xyz1[f0]
            u_first = u_first - n * float(u_first @ n)
            u_first = u_first / np.linalg.norm(u_first)
            res2, skips2 = run(cands2, xyz1, px1, {**ctx, "u_ref": u_first})
            info["second_pair"] = {"axis": other_axis, "diff_before_mm": round(
                diffs1[other_pair], 3),
                "skipped": [f"{a}/{n_}: {s}" for a, n_, s in skips2]}
            weak = [f"{r['anchor']}: support {r['support']:.2f}" for r in res2
                    if not r["noop"] and r["support"] < DEPTH_EDGE_SECOND_PAIR_MIN_SUPPORT]
            if weak:
                info["second_pair"]["unsupported"] = weak
            moves2 = [r for r in res2 if not r["noop"]
                      and r["support"] >= DEPTH_EDGE_SECOND_PAIR_MIN_SUPPORT]
            if moves2:
                best2 = _pick_best(moves2, xyz1, px1)
                xyz2, px2 = _apply_moved(xyz1, px1, best2)
                if (max(_rect_diffs_mm(xyz2).values())
                        < max(diffs1.values()) - 1e-6):
                    xyz1, px1 = xyz2, px2
                    moved_names |= set(best2["moved"])
                    steps.append(summary(best2))
                    info["second_pair"].update({"applied": True, "anchor": best2["anchor"],
                                                "info": best2["info"]})
                    reasons.append(f"second_pair_from_depth({other_axis}/{best2['anchor']})")
                    moved_any = True
                else:
                    info["second_pair"]["applied"] = False
                    info["second_pair"]["why"] = "residual_not_improved"
            else:
                info["second_pair"]["applied"] = False
                info["second_pair"]["why"] = ("unsupported" if weak else "no_placement")
    info["steps"] = steps

    diffs_f = _rect_diffs_mm(xyz1)
    info["rect_after_mm"] = {k: round(v, 3) for k, v in diffs_f.items()}
    record_out = copy.deepcopy(record)
    seam_out = dict(seam) if seam else None
    if not moved_any:
        info["reason"] = first_info.get("reason", "no_move")
        return record_out, seam_out, info
    worst0, worst_f = max(diffs0.values()), max(diffs_f.values())
    if (worst0 <= tol < worst_f) or worst_f > worst0 + DEPTH_EDGE_WORSE_MM:
        raise _DepthEdgeSkip(f"depth_edge_residual_worse(before {worst0:.2f}mm, after "
                             f"{worst_f:.2f}mm; original kept)", {"rect_after_mm": info["rect_after_mm"]})
    if class_gate:
        match = _class_match(xyz1)
        info["class_match"] = match
        if match is None:
            dims = _quad_dims_mm(xyz1)
            raise _DepthEdgeSkip(
                f"depth_edge_class_not_validated(long side {1000 * long_side:.1f}mm; fixed dims "
                f"{dims[0]:.1f}x{dims[1]:.1f}mm match no known box class)")
    names = [k for k in RING if k in moved_names]
    info["moved_mm"] = {k: round(1000 * float(np.linalg.norm(xyz1[k] - xyz0[k])), 3)
                        for k in names}
    info["moved_px"] = {k: round(float(np.linalg.norm(px1[k] - px0[k])), 3) for k in names}
    info["corners_px_before"] = {k: list(px_in[k]) for k in names}
    info["corners_xyz_before"] = {k: list(xyz_in[k]) for k in names}

    if seam_axis_ok:
        new_end_px, downgraded = _seam_after(seam, axis, xyz0, px0, xyz1, px1, n, d,
                                             camera, info)
    else:
        new_end_px, downgraded = None, False
        info["seam_line_recomputed"] = False
    for k in names:
        record_out["top_face"]["corners_px"][k] = [round(float(v), 2) for v in px1[k]]
        record_out["top_face"]["corners_xyz_camera_m"][k] = [round(float(v), 5)
                                                             for v in xyz1[k]]
    if seam_axis_ok:
        path = _SEAM_FIELD_PATHS.get(seam.get("source"))
        node = record_out.get("seam")
        if path:
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
        for key in ("seam_left", "seam_right"):
            value = [round(float(v), 2) for v in new_end_px[key]]
            seam_out[key] = value
            if path and isinstance(node, dict):
                node[key] = list(value)
        if downgraded:
            seam_out["observed"] = False
    info["applied"] = True
    info["reason"] = first_info.get("reason", "applied") if len(steps) == 1 else (
        f"{first_info.get('reason', 'applied')}+second_pair")
    return record_out, seam_out, info


def apply_depth_edge(record: dict, seam: dict, manifest: dict, pcd_path, gray=None,
                     profile=None):
    """Load the cloud/camera and run :func:`depth_edge_fix`; never raises.

    Returns (record_out, seam_out, info); on any skip/failure the inputs are
    returned unchanged with ``info = {"applied": False, "reason": ...}``.
    """
    t0 = time.monotonic()
    try:
        if profile is not None and not profile.depth_edge_validated \
                and profile.own_constants:
            # The fix is validated for the known boxes only (per-profile flag);
            # 'big' keeps today's own class gate inside depth_edge_fix.
            raise _DepthEdgeSkip(f"depth_edge_not_validated_for_profile({profile.name})")
        _plane_nd((record.get("top_face") or {}).get("plane"))
        camera = camera_from_manifest(manifest)
        if camera is None:
            raise _DepthEdgeSkip("depth_edge_no_camera")
        from ..core.autolabel.pcd import read_pcd_xyz  # noqa: PLC0415
        cloud, _ = read_pcd_xyz(str(pcd_path))
        record_out, seam_out, info = depth_edge_fix(record, seam, cloud, camera, gray=gray)
    except _DepthEdgeSkip as skip:
        record_out, seam_out = record, seam
        info = {**skip.info, "applied": False, "reason": str(skip)}
    except Exception as error:  # noqa: BLE001 — the fix is optional; never fail here
        record_out, seam_out = record, seam
        info = {"applied": False,
                "reason": f"depth_edge_error({type(error).__name__}: {error})"}
    info["time_s"] = round(time.monotonic() - t0, 3)
    return record_out, seam_out, info


# ---- box profile + cut permission (M1/M4 live) -------------------------------

def resolve_box_profile(record: dict):
    """(profile, info) of a v5 record. Prefers the record's own ``box_profile``
    ({"name": ...}, written by the pipeline); otherwise today's rule on the
    top-face long side (``classify_legacy``: > 150 mm big, else small), so
    today's two boxes never become 'generic' behind the pipeline's back."""
    recorded = record.get("box_profile")
    name = recorded.get("name") if isinstance(recorded, dict) else (
        recorded if isinstance(recorded, str) else None)
    if name in box_profiles.PROFILES:
        return box_profiles.PROFILES[name], {
            "name": name, "source": "record",
            "record": recorded if isinstance(recorded, dict) else None}
    info = {"source": "corners_legacy"}
    if name is not None:
        info["ignored_record_profile"] = str(name)
    xyz = (record.get("top_face") or {}).get("corners_xyz_camera_m")
    try:
        pts = [[float(v) for v in xyz[k]] for k in RING]
        sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        long_side = max(0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3]))
        info["top_long_side_mm"] = round(1000.0 * long_side, 2)
        # Today's 150 mm rule (identical to the old live behaviour); only the
        # leave-one-type-out switch can turn a known class into generic. The
        # pipeline's record profile (preferred above) does the full resolve.
        # classify_legacy already maps an excluded class to generic
        # (AUTOLABEL_V2_PROFILES_EXCLUDE); record which class was excluded.
        profile = box_profiles.classify_legacy(long_side)
        raw = box_profiles.legacy_class_name(long_side)
        if profile.is_generic and raw != "generic":
            info["excluded_by_env"] = raw
    except (KeyError, TypeError, ValueError, IndexError):
        # No profile name and no corners: nothing is known -> generic (never a guess).
        profile = box_profiles.PROFILES["generic"]
        info["source"] = "default_generic(no corners_xyz)"
    info["name"] = profile.name
    return profile, info


CUT_BORDER_PX = 20.0        # top-face corner this close to the image border -> blocked (retryable)
CUT_MIN_GAP_COVERAGE = 0.02  # seam.estimate.coverage (gap-fit slice coverage) below this -> blocked.
#                              Placeholder: real accepted captures reach 0.041, so this blocks none.
CUT_SEAM_BAD_STATUS = ("ambiguous", "invisible")


def _cp_reason(code: str, retryable: bool, detail: str) -> dict:
    return {"code": code, "retryable": bool(retryable), "detail": detail}


def _measured_height_mm(record: dict):
    """Measured height of a record: box_profile.measured_height_mm (pipeline format),
    else a top-level measured_height_mm; None when unavailable."""
    recorded = record.get("box_profile")
    for value in ((recorded.get("measured_height_mm") if isinstance(recorded, dict) else None),
                  record.get("measured_height_mm")):
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(value):
            return float(value)
    return None


def _support_hidden_height(record: dict) -> bool:
    """The pipeline could not measure the height because the support under the
    box is hidden (scene ``hidden_stand``): box_profile.height_unavailable ==
    "support_hidden". Not a measurement failure: a KNOWN profile then uses its
    own height (generic has none, so it is still blocked)."""
    recorded = record.get("box_profile")
    return (isinstance(recorded, dict)
            and recorded.get("height_unavailable") == "support_hidden")


GENERIC_EDGE_MIN_VALID_FRAC = 0.8
#: waiver: coplanar top beyond each quad edge (edge_support extent / share of stations);
#: good dual-arm tops <= 5.8 mm / 0.1, the new generic box <= 2.5 mm / 0.0 (2026-10-13)
GENERIC_EDGE_MAX_EXTENT_MM = 8.0
GENERIC_EDGE_MAX_CONTINUE_FRAC = 0.2


#: waiver with a depth-fitted (rectangle) quad: the selected attempt's SAM3 outline must agree
#: (2026-10-13 new box: IoU 0.965, corners <= 9.2 px)
GENERIC_RGB_MIN_IOU = 0.92
GENERIC_DEPTH_MIN_IOU = 0.9   # primary depth_top quad vs RGB quad (new box: 0.94-0.96)
GENERIC_RGB_MAX_CORNER_PX = 12.0


def _rgb_quad_agrees(record: dict) -> bool:
    """The SAM3 (RGB) outline of the selected attempt (``original_coarse`` before the depth
    rewrite, else ``coarse``) overlaps the final top quad by >= GENERIC_RGB_MIN_IOU and every
    final corner lies within GENERIC_RGB_MAX_CORNER_PX of an RGB corner. False when unknown."""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        attempts = [a for a in ((record.get("provenance") or {}).get("attempts") or [])
                    if isinstance(a, dict)]
        sel = [a for a in attempts if a.get("selected") is True]
        if len(sel) != 1:
            return False
        rgb = sel[0].get("original_coarse") or sel[0].get("coarse")
        c = (record.get("top_face") or {}).get("corners_px") or {}
        fin = np.asarray([c[k] for k in ("back_left", "back_right", "front_right", "front_left")],
                         dtype=np.float64).reshape(4, 2)
        rgb = np.asarray(rgb, dtype=np.float64).reshape(4, 2)
        if not (np.all(np.isfinite(fin)) and np.all(np.isfinite(rgb))):
            return False
        lo = np.floor(np.minimum(fin.min(0), rgb.min(0))) - 2
        hi = np.ceil(np.maximum(fin.max(0), rgb.max(0))) + 2
        w, h = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1
        if w <= 0 or h <= 0 or w * h > 4_000_000:
            return False
        ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(ma, np.round(fin - lo).astype(np.int32), 1)
        cv2.fillConvexPoly(mb, np.round(rgb - lo).astype(np.int32), 1)
        union = int((ma | mb).sum())
        iou = int((ma & mb).sum()) / union if union else 0.0
        corner = max(float(np.min(np.linalg.norm(rgb - p, axis=1))) for p in fin)
        return iou >= GENERIC_RGB_MIN_IOU and corner <= GENERIC_RGB_MAX_CORNER_PX
    except Exception:  # noqa: BLE001 - evidence unknown -> no waiver
        return False


def _depth_confirmed_and_supported(record: dict) -> bool:
    """Fail-closed evidence for the generic_cut_without_height waiver:
    checks.independent_top_check.confirmed is True (and not a circular
    rectangle-fitted skip) AND top_face.depth_diag.edge_support.edges holds ALL
    four edges, each evaluated (status ok, valid_frac >= GENERIC_EDGE_MIN_VALID_FRAC),
    AND none is unsupported. A missing depth_diag (e.g. AUTOLABEL_V2_TOP_DIAG_OFF)
    or edges with insufficient depth never qualify."""
    prof = record.get("box_profile")
    clearance = prof.get("support_clearance_mm") if isinstance(prof, dict) else None
    if not (isinstance(clearance, (int, float)) and not isinstance(clearance, bool)
            and math.isfinite(clearance) and clearance > 0.0) \
            or prof.get("support_gravity_unchecked"):
        return False        # support not measured below the top, or chosen without gravity check
    # the primary depth check (depth_top found the top on its own and it overlaps the RGB quad)
    # is confirmation; the independent re-check only runs when that one is unavailable
    iou = (record.get("checks") or {}).get("rgb_depth_quad_iou")
    if isinstance(iou, (int, float)) and not isinstance(iou, bool) and math.isfinite(iou) \
            and iou >= GENERIC_DEPTH_MIN_IOU:
        check = {"confirmed": True}
    else:
        check = (record.get("checks") or {}).get("independent_top_check")
    if not isinstance(check, dict) or check.get("confirmed") is not True:
        return False
    if check.get("skipped_rectangle_fitted") and not _rgb_quad_agrees(record):
        # the final quad came from the depth plane itself (circular depth check): only an
        # agreeing SAM3 (RGB) outline of the same attempt is independent evidence
        return False
    diag = (record.get("top_face") or {}).get("depth_diag")
    support = diag.get("edge_support") if isinstance(diag, dict) else None
    if not isinstance(support, dict) or support.get("unsupported") != []:
        return False
    edges = support.get("edges")
    if not isinstance(edges, dict):
        return False
    def _num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))

    for name in ("back", "front", "left", "right"):
        e = edges.get(name)
        if not isinstance(e, dict):
            return False
        vf, ext, cf = e.get("valid_frac"), e.get("extent_mm"), e.get("continue_frac")
        # evaluated AND located: the coplanar top must end at the quad edge (no size check
        # backs a generic box, so a top continuing past an edge would put the cut ends short)
        if not (e.get("status") == "ok" and _num(vf) and vf >= GENERIC_EDGE_MIN_VALID_FRAC
                and _num(ext) and ext <= GENERIC_EDGE_MAX_EXTENT_MM
                and _num(cf) and cf <= GENERIC_EDGE_MAX_CONTINUE_FRAC):
            return False
    return True


def _height_mismatch_flag(record: dict) -> bool:
    """box_profile.flags (pipeline.resolve_box_profile) contains height_mismatch."""
    recorded = record.get("box_profile")
    flags = recorded.get("flags") if isinstance(recorded, dict) else None
    return isinstance(flags, list) and "height_mismatch" in flags


# Known profile: the FINAL top face must be within this of the profile's top dims.
# 431-capture replay (2026-10-06): correct quads deviate <= 4.0 mm (p95), 9.3 mm max;
# truncated quads (front edge on the tape edge, ~100 x 77 mm, seam ~10 mm off) and
# tape-band quads (101 x 33 mm) deviate 18-66 mm.
CUT_MAX_TOP_DIMS_DEV_MM = 12.0


def _final_top_dims_dev_mm(record: dict, profile):
    """Max |final top-face side - profile top dim| (mm, long/short sorted) for a
    known profile with top dims; None for generic / unknown dims / no corners."""
    if profile is None or profile.is_generic or not profile.top_dims_mm:
        return None
    xyz = (record.get("top_face") or {}).get("corners_xyz_camera_m")
    # K1 publishes a known-size quad: the size check must look at the MEASURED (RGB) one
    xyz = _gap_anchored_rgb_xyz(record) or xyz
    try:
        pts = [[float(v) for v in xyz[k]] for k in RING]
    except (KeyError, TypeError, ValueError):
        return None
    sides = [1000.0 * math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lo, sh = sorted((0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3])), reverse=True)
    pl, ps = sorted(profile.top_dims_mm, reverse=True)
    return max(abs(lo - pl), abs(sh - ps))


def evaluate_cut_permission(record: dict, seam: dict, seam_direction: dict = None,
                            seam_geometry: dict = None, profile=None,
                            image_size=None, border_px: float = CUT_BORDER_PX,
                            generic_cut_without_height: bool = False) -> dict:
    """Cut permission of an accepted detection: ``{"status": "allowed"|"blocked",
    "reasons": [{"code", "retryable", "detail"}]}``. Pure; reads only the
    arguments. Detection and cut permission are separate: a blocked capture
    still has a seam, but must not be cut without ``--allow-cut-blocked``.

    Rules: geometry-axis seam outside / without the soft direction range
    (needs review), geometry-axis seam on a generic profile, seam status
    ambiguous/invisible, gap-fit coverage below CUT_MIN_GAP_COVERAGE, top-face
    corner within ``border_px`` of the image border (retryable: re-capture),
    measured-height mismatch flag, more than one tape candidate
    (record.seam.tape_evidence.candidates, when present).
    """
    reasons = []
    seam = seam or {}
    seam_direction = seam_direction or {}
    estimate = (record.get("seam") or {}).get("estimate") or {}
    geom_axis = bool(seam_geometry) or estimate.get("axis_source") == "geometry_class"
    if geom_axis:
        if profile is not None and profile.own_constants:
            reasons.append(_cp_reason(
                "geometry_axis_generic_profile", False,
                "seam axis from box geometry only; generic profile has no validated geometry rule"))
        outside = bool(seam_direction.get("seam_direction_outside_soft_range"))
        if outside:
            reasons.append(_cp_reason(
                "seam_axis_from_geometry_outside_soft_range", False,
                f"offset_from_prior={seam_direction.get('offset_from_prior_deg')}deg"))
        elif seam_direction.get("base_link_angle_deg") is None:
            reasons.append(_cp_reason("seam_axis_from_geometry_direction_unknown", False,
                                      "base_link seam direction unavailable"))
    status = seam.get("status") or (record.get("seam") or {}).get("status")
    if status in CUT_SEAM_BAD_STATUS:
        reasons.append(_cp_reason(f"seam_status_{status}", False, "no usable seam axis"))
    coverage = estimate.get("coverage")
    if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) \
            and math.isfinite(coverage) and coverage < CUT_MIN_GAP_COVERAGE:
        reasons.append(_cp_reason("seam_coverage_low", False,
                                  f"coverage {coverage:.3f} < {CUT_MIN_GAP_COVERAGE:g}"))
    size = image_size or record.get("image_size")
    corners = (record.get("top_face") or {}).get("corners_px")
    # Visibility is about the MEASURED outline: with top_from_gap the published corners
    # are model-placed (e.g. the front edge moved ~6 mm inwards), so use the RGB quad.
    rgb_quad = (((record.get("top_face") or {}).get("top_from_gap") or {})
                .get("rgb_quad") or {}).get("corners_px")
    if isinstance(rgb_quad, dict) and rgb_quad:
        corners = rgb_quad
    if size and isinstance(corners, dict):
        try:
            width, height = float(size[0]), float(size[1])
            margin = min(min(float(c[0]), float(c[1]), width - 1.0 - float(c[0]),
                             height - 1.0 - float(c[1])) for c in corners.values())
            if margin < border_px:
                reasons.append(_cp_reason(
                    "top_face_touches_image_border", True,
                    f"nearest corner {margin:.1f}px from the border (< {border_px:g}px)"))
        except (TypeError, ValueError, IndexError):
            pass
    if profile is not None and profile.own_constants and status != "ok":
        reasons.append(_cp_reason(
            "generic_seam_unconfirmed", False,
            f"seam status {status}: line not confirmed by gap/darkline evidence"))
    limit = getattr(profile, "seam_max_curvature_mm", None) if profile is not None else None
    curvature = estimate.get("seam_curvature_mm")
    if limit is not None and isinstance(curvature, (int, float)) \
            and not isinstance(curvature, bool) and math.isfinite(curvature) \
            and curvature > float(limit):
        reasons.append(_cp_reason(
            "seam_too_curved", False,
            f"observed flap gap bows {curvature:.1f} mm from the straight cut line "
            f"(> {float(limit):g} mm)"))
    # Any profile: without a measured height a known footprint could be a taller box.
    # Applies to the pipeline's own identification (record box_profile) and to generic;
    # records of the old format (no box_profile at all) have no height to demand.
    hidden_known = bool(profile is not None and not profile.is_generic
                        and profile.height_mm is not None
                        and _support_hidden_height(record))
    flags = []
    if profile is not None and (profile.is_generic or isinstance(record.get("box_profile"), dict)) \
            and _measured_height_mm(record) is None and not hidden_known:
        if (generic_cut_without_height and profile.is_generic
                and _support_hidden_height(record)
                and _depth_confirmed_and_supported(record)):
            # scene option: the cut only needs the top face + seam (the executor has its
            # own carton height); the depth check confirmed the top and no edge is truncated
            flags.append("height_waived(support_hidden)")
        else:
            reasons.append(_cp_reason("height_unavailable", False,
                                      "no measured box height (a taller box of the same footprint "
                                      "cannot be excluded)"))
    if _height_mismatch_flag(record):
        reasons.append(_cp_reason("height_mismatch", False,
                                  "measured box height disagrees with the profile (needs review)"))
    # Only with the pipeline's own identification (record box_profile, which uses
    # the measured height); a corner-fallback class says nothing beyond size.
    final_dev = (_final_top_dims_dev_mm(record, profile)
                 if isinstance(record.get("box_profile"), dict) else None)
    if final_dev is not None and final_dev > max(CUT_MAX_TOP_DIMS_DEV_MM,
                                                 float(profile.dims_tol_mm)):
        # Known box (legacy class / measured height) but the FINAL top-face quad
        # (after the depth-edge fix / R9) is far from its top dims: the quad is
        # wrong (e.g. a 101 x 33 mm tape band instead of the 110 x 98 mm top).
        # A fresh capture may segment it correctly, hence retryable.
        reasons.append(_cp_reason(
            "top_face_does_not_match_profile", True,
            f"final top face {final_dev:.1f} mm from the {profile.name} profile dims"))
    if profile is not None and getattr(profile, "edge_support_cut_rule", False):
        # D1: the coplanar top continues beyond a quad edge = a truncated quad (e.g. a quad
        # that ends on the flap gap, a tape band). Evidence from top_face.depth_diag.
        diag = (record.get("top_face") or {}).get("depth_diag")
        support = diag.get("edge_support") if isinstance(diag, dict) else None
        unsupported = support.get("unsupported") if isinstance(support, dict) else None
        for edge in (unsupported if isinstance(unsupported, list) else []):
            info = ((support.get("edges") or {}).get(edge)) or {}
            reasons.append(_cp_reason(
                f"top_edge_unsupported({edge})", True,
                f"the coplanar top continues {info.get('extent_mm')} mm beyond the {edge} edge "
                f"({info.get('continue_frac')} of its stations): the quad is truncated"))
    evidence = (record.get("seam") or {}).get("tape_evidence")
    candidates = evidence.get("candidates") if isinstance(evidence, dict) else None
    if isinstance(candidates, list):
        # only qualified candidates (ok / qualified flag) count, and only differing axes conflict
        qualified = [c for c in candidates if isinstance(c, dict)
                     and (c.get("ok") is True or c.get("qualified") is True)]
        axes = sorted({str(c.get("axis")) for c in qualified})
        if len(qualified) > 1 and len(axes) > 1:
            reasons.append(_cp_reason("multiple_tape_candidates", False,
                                      f"{len(qualified)} qualified tape candidates "
                                      f"(axes {axes})"))
    out = {"status": "blocked" if reasons else "allowed", "reasons": reasons}
    if flags:
        out["flags"] = flags
    return out


MULTI_BOX_HINT = ("more than one box (or a hand / object on the top) seen: leave only the box to cut "
                  "in view, with clear space around it, and capture again")


def multi_box_suspect(record: dict, profile):
    """F4: corroborated evidence of several boxes / an object on the top (the
    pipeline's ``top_face.depth_diag.multi_box``: another box-sized component
    coplanar with the top, or an off-plane blob inside the top). Only for a
    profile with ``multi_box_reject``; returns the evidence dict or None."""
    if profile is None or not getattr(profile, "multi_box_reject", False):
        return None
    diag = (record.get("top_face") or {}).get("depth_diag")
    multi = diag.get("multi_box") if isinstance(diag, dict) else None
    if isinstance(multi, dict) and multi.get("suspect"):
        return {k: multi.get(k) for k in ("evidence", "other_box_sized", "step_blobs",
                                          "own_area_ratio", "components")}
    return None


def reject_cut_permission(error) -> dict:
    """cut_permission block of a reject JSON: a rejected capture is blocked."""
    details = getattr(error, "details", None) or {}
    if error.stage == "cut_permission" and isinstance(details.get("cut_permission"), dict):
        return details["cut_permission"]
    return {"status": "blocked",
            "reasons": [_cp_reason(f"rejected_{error.stage}", error.retryable,
                                   "; ".join(str(r) for r in error.reasons)[:300])]}


# ---- R9: joint rectangle repair (last resort) -------------------------------

REPAIRABLE_REASONS = ("rectangle_inconsistent", "rectangle_skewed")


def joint_rectangle_repair(record: dict, seam, camera, tol_mm: float,
                           max_corner_dev_deg: float = RECT_MAX_CORNER_DEV_DEG):
    """Replace a quad the rectangle gate rejects by its joint rectangle fit.

    In the top plane (corners projected onto it, 2-D frame at the centroid):
    shared orientation theta = circular mean of 4 x (edge angle) over the 4
    edges, / 4; each edge becomes the line at theta (or theta + 90 deg,
    whichever it is closest to) through its midpoint; consecutive lines meet
    in the 4 new corners (3-D, then projected to pixels WITH lens distortion).
    Accepted only when every edge's angular residual to its fitted direction
    <= JOINT_FIT_MAX_RESIDUAL_DEG, every corner moves <= JOINT_FIT_MAX_MOVE_MM
    in 3-D and the repaired quad passes :func:`rectangle_check` (incl. R7).
    The seam endpoints are re-derived from the repaired quad with
    :func:`_seam_after` (as after the depth edge fix).
    Returns ``(record_out, seam_out, info, rect)``; raises
    _DepthEdgeSkip(reason, info) when the repair does not apply (inputs
    untouched).
    """
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    info = {"applied": False, "flag": "rectangle_repaired_joint_fit"}
    try:
        top = record.get("top_face") or {}
        n, d = _plane_nd(top.get("plane"))
        px_in = top.get("corners_px") or {}
        xyz_in = top.get("corners_xyz_camera_m") or {}
        try:
            px = {k: np.asarray(px_in[k], dtype=np.float64).reshape(2) for k in RING}
            xyz = {k: np.asarray(xyz_in[k], dtype=np.float64).reshape(3) for k in RING}
        except (KeyError, TypeError, ValueError):
            raise _DepthEdgeSkip("joint_fit_no_corners") from None
        if not all(np.all(np.isfinite(v)) for v in (*px.values(), *xyz.values())):
            raise _DepthEdgeSkip("joint_fit_no_corners")
        if camera is None:
            raise _DepthEdgeSkip("joint_fit_no_camera")
        xyz = {k: v - n * (float(v @ n) + d) for k, v in xyz.items()}
        origin = np.mean([xyz[k] for k in RING], axis=0)
        e1 = xyz["back_right"] - xyz["back_left"]
        e1 = e1 - n * float(e1 @ n)
        if np.linalg.norm(e1) < 1e-6:
            raise _DepthEdgeSkip("joint_fit_degenerate(back edge)")
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        def to2(p):
            q = p - origin
            return np.array([float(q @ e1), float(q @ e2)])

        p2 = {k: to2(xyz[k]) for k in RING}
        edge_names = ["back", "right", "front", "left"]
        edge_pts = [(RING[i], RING[(i + 1) % 4]) for i in range(4)]
        phi = [math.atan2(*(p2[b] - p2[a])[::-1]) for a, b in edge_pts]
        theta = math.atan2(sum(math.sin(4 * f) for f in phi),
                           sum(math.cos(4 * f) for f in phi)) / 4.0
        residual, kind = [], []
        for f in phi:
            k = round((f - theta) / (math.pi / 2))
            residual.append(math.degrees((f - theta) - k * math.pi / 2))
            kind.append(int(k) % 2)
        info["theta_deg"] = round(math.degrees(theta), 3)
        info["edge_residual_deg"] = {nm: round(r, 3) for nm, r in zip(edge_names, residual)}
        if kind[0] != kind[2] or kind[1] != kind[3] or kind[0] == kind[1]:
            raise _DepthEdgeSkip("joint_fit_orientation_inconsistent")
        worst = max(abs(r) for r in residual)
        info["max_residual_deg"] = round(worst, 3)
        if worst > JOINT_FIT_MAX_RESIDUAL_DEG:
            raise _DepthEdgeSkip(f"joint_fit_residual({worst:.2f}deg > "
                                 f"{JOINT_FIT_MAX_RESIDUAL_DEG:g}deg)")
        lines = []
        for (a, b), kd in zip(edge_pts, kind):
            ang = theta + kd * math.pi / 2
            lines.append((0.5 * (p2[a] + p2[b]), np.array([math.cos(ang), math.sin(ang)])))

        def meet(i, j):
            (p, r), (q, t) = lines[i], lines[j]
            den = float(r[0] * t[1] - r[1] * t[0])
            if abs(den) < 1e-9:
                raise _DepthEdgeSkip("joint_fit_degenerate(parallel edges)")
            return p + (float((q - p)[0] * t[1] - (q - p)[1] * t[0]) / den) * r

        # back=0 right=1 front=2 left=3
        c2 = {"back_left": meet(0, 3), "back_right": meet(0, 1),
              "front_right": meet(1, 2), "front_left": meet(2, 3)}
        xyz_new = {k: origin + c2[k][0] * e1 + c2[k][1] * e2 for k in RING}
        moves = {k: 1000.0 * float(np.linalg.norm(xyz_new[k] - xyz[k])) for k in RING}
        info["corner_moves_mm"] = {k: round(v, 3) for k, v in moves.items()}
        if max(moves.values()) > JOINT_FIT_MAX_MOVE_MM:
            raise _DepthEdgeSkip(f"joint_fit_corner_move({max(moves.values()):.2f}mm > "
                                 f"{JOINT_FIT_MAX_MOVE_MM:g}mm)")
        new_px_arr = project_to_px(np.stack([xyz_new[k] for k in RING]), camera)
        px_new = {k: new_px_arr[i] for i, k in enumerate(RING)}
        record_out = copy.deepcopy(record)
        for k in RING:
            record_out["top_face"]["corners_px"][k] = [round(float(v), 2) for v in px_new[k]]
            record_out["top_face"]["corners_xyz_camera_m"][k] = [
                round(float(v), 5) for v in xyz_new[k]]
        try:
            rect = rectangle_check(record_out, float(tol_mm), float(max_corner_dev_deg))
        except Reject as error:
            raise _DepthEdgeSkip(f"joint_fit_gate_failed({'; '.join(error.reasons)})") from None
        seam_out = dict(seam) if seam else None
        axis = (seam or {}).get("axis")
        if axis in SEAM_CROSSED_EDGES and "seam_left" in seam and "seam_right" in seam:
            seam_info = {}
            new_end_px, downgraded = _seam_after(seam, axis, xyz, px, xyz_new, px_new,
                                                 n, d, camera, seam_info)
            info["seam"] = seam_info
            path = _SEAM_FIELD_PATHS.get(seam.get("source"))
            node = record_out.get("seam")
            if path:
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
            for key in ("seam_left", "seam_right"):
                value = [round(float(v), 2) for v in new_end_px[key]]
                seam_out[key] = value
                if path and isinstance(node, dict):
                    node[key] = list(value)
            if downgraded:
                seam_out["observed"] = False
            info["seam_status_downgraded"] = downgraded
        else:
            info["seam"] = {"seam_line_recomputed": False}
        info["applied"] = True
        info["corners_px_before"] = {k: [float(v) for v in px_in[k]] for k in RING}
        info["corners_xyz_before"] = {k: [float(v) for v in xyz_in[k]] for k in RING}
        rect["repaired"] = True
        return record_out, seam_out, info, rect
    except _DepthEdgeSkip as skip:
        skip.info = {**info, **getattr(skip, "info", {}), "applied": False}
        raise


def rectangle_gate_with_repair(record: dict, seam, manifest: dict, tol_mm: float,
                               max_corner_dev_deg: float = RECT_MAX_CORNER_DEV_DEG):
    """Rectangle gate, then (only for rectangle_inconsistent/rectangle_skewed)
    the R9 joint repair as the last resort. Returns ``(record, seam, rect,
    repair_info|None)``; a quad that passes the gate is returned untouched
    (repair never runs). Re-raises the ORIGINAL Reject when the repair does
    not apply or fails its gates."""
    try:
        return record, seam, rectangle_check(record, tol_mm, max_corner_dev_deg), None
    except Reject as error:
        if "r9" in rules_off() or error.stage != "rectangle" or not any(
                str(r).startswith(REPAIRABLE_REASONS) for r in error.reasons):
            raise
        try:
            camera = camera_from_manifest(manifest)
            record_out, seam_out, info, rect = joint_rectangle_repair(
                record, seam, camera, tol_mm, max_corner_dev_deg)
        except _DepthEdgeSkip as skip:
            error.repair = {**skip.info, "applied": False, "reason": str(skip)}
            raise error from None
        except Exception as repair_error:  # noqa: BLE001 — repair is optional
            error.repair = {"applied": False, "flag": "rectangle_repaired_joint_fit",
                            "reason": f"joint_fit_error({type(repair_error).__name__}: "
                                      f"{repair_error})"}
            raise error from None
        return record_out, seam_out, rect, info


# ---- R10 (live path): seam axis from the FINAL quad's geometry ---------------

def geometry_seam(record: dict, seam_reject: Reject, est_cfg=None,
                  profile=None):
    """The record's seam is ambiguous: after the depth edge fix / rectangle gate
    the FINAL quad may be a clean small-box rectangle, whose seam axis
    ``seam_estimator.geometry_class_axis`` knows (seam joins the midpoints of
    the two long sides). Publishes the projective midline of the final PIXEL
    quad for that axis: status estimated, observed False, axis_source
    geometry_class. Returns ``(record_out, seam, info)``; raises the original
    ambiguity Reject (plus the geometry reason) when no axis is returned."""
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    if profile is not None and not profile.geometry_axis_validated:
        raise seam_reject               # R10 only inside a validated profile (small)
    from ..core import seam_estimator as est  # noqa: PLC0415

    top = record.get("top_face") or {}
    xyz = top.get("corners_xyz_camera_m")
    axis, geom = est.geometry_class_axis(
        {k: xyz[k] for k in RING} if isinstance(xyz, dict) and all(k in xyz for k in RING)
        else None, est_cfg)
    if axis is None:
        raise Reject("seam", [*seam_reject.reasons,
                              f"geometry_class_axis_none({geom.get('reason')})"],
                     details={"geometry_axis": geom})
    quad = np.array([top["corners_px"][k] for k in RING], dtype=np.float64)
    eps = est.projective_midlines(quad)[axis]["endpoints"]
    left, right = ([round(float(v), 2) for v in eps[0]], [round(float(v), 2) for v in eps[1]])
    record_out = copy.deepcopy(record)
    prior = {"seam_left": left, "seam_right": right}
    record_out["seam"] = {
        "status": "estimated", "endpoints_px": None, "axis": axis, "score": 0.0,
        "source": "prior",
        "estimate": {"method": "projective_midline", "axis": axis,
                     "axis_source": "geometry_class", "seam_axis_from_geometry": True,
                     "prior_endpoints_px": dict(prior), "geometry_axis": geom,
                     "source": "prior", "status": "estimated"}}
    seam = {"status": "estimated", "observed": False,
            "source": "seam.estimate.prior_endpoints_px", "axis": axis,
            "seam_source": "prior", "seam_left": left, "seam_right": right}
    info = {"axis": axis, "axis_source": "geometry_class", "geometry_axis": geom,
            "flag": "seam_axis_from_geometry", "endpoints_px": prior}
    return record_out, seam, info


# ---- R6: seam direction soft prior (log/flag only, never rejects) -----------

SEAM_DIRECTION_PRIOR_DEG = 83.0      # observed base_link seam direction (median)
SEAM_DIRECTION_SOFT_RANGE_DEG = 45.0
SEAM_DIRECTION_CALIB = "/home/farhan/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"
_CAM_TO_BASE_CACHE = {}


def _camera_to_base_matrix(calib_file=None, manifest=None, scene=None):
    """base_link <- camera_color_optical_frame = hand-eye calibration composed
    with the driver extrinsic (the manifest's driver_extrinsic block when it has one, else the
    recorded constant; no ROS). ``calib_file`` is the SAME file the snapshot transform uses
    (default SEAM_DIRECTION_CALIB). Cached per (file, mtime, extrinsic); None when unavailable --
    then the seam direction is "unknown" (a clearly logged condition, never silent)."""
    calib_file = os.path.expanduser(str(calib_file or SEAM_DIRECTION_CALIB))
    try:
        mtime = os.stat(calib_file).st_mtime_ns
    except OSError:
        mtime = None
    block = (manifest or {}).get("driver_extrinsic") if isinstance(manifest, dict) else None
    block_key = None
    if isinstance(block, dict):
        try:
            block_key = (tuple(block["translation"]), tuple(block["quaternion_xyzw"]))
        except (KeyError, TypeError):
            block = None
    fallback = scene.driver_block("fallback") if scene is not None else None
    fb_key = None
    if fallback is not None:
        fb_key = (tuple(fallback["translation"]), tuple(fallback["quaternion_xyzw"]))
    key = (calib_file, mtime, block_key, fb_key)
    if key in _CAM_TO_BASE_CACHE:
        return _CAM_TO_BASE_CACHE[key]
    matrix = None
    try:
        from . import base_link as c2b  # noqa: PLC0415

        calibration = c2b.load_handeye_calibration(calib_file)
        recorded = fallback if fallback is not None else c2b.RECORDED_DRIVER_EXTRINSIC
        if block_key is not None:
            recorded = {"translation": block["translation"], "quaternion_xyzw": block["quaternion_xyzw"]}
        driver = c2b.transform_matrix(recorded["translation"], recorded["quaternion_xyzw"])
        matrix = calibration["matrix"] @ driver
    except Exception as error:  # noqa: BLE001 - optional estimate
        matrix = None
        _say(f"WARNING: seam-direction calibration unavailable ({calib_file}: "
             f"{type(error).__name__}: {error}); the seam direction is UNKNOWN for this capture "
             f"(a geometry-axis seam is then sent to review)")
    _CAM_TO_BASE_CACHE[key] = matrix
    return matrix


def compute_seam_direction(record: dict, seam, manifest: dict, camera_to_base=None,
                           calib_file=None, scene=None) -> dict:
    """Seam direction in the camera's top-plane frame and (approximately, from
    the hand-eye calibration + recorded driver extrinsic) in base_link.

    Axis angles are mod 180 deg. Plane frame: e1 = camera x projected on the
    top plane, e2 = n x e1. base_link angle = atan2(y, x) of the seam vector.
    ``seam_direction_outside_soft_range`` is True when the base_link axis is
    more than SEAM_DIRECTION_SOFT_RANGE_DEG from SEAM_DIRECTION_PRIOR_DEG.
    A SOFT PRIOR ONLY: logged/flagged, never a rejection. ``scene`` (a
    ``core.scene.Scene``) supplies the prior (``live.seam_direction_prior_deg``;
    None there = the prior is disabled: the angle is still reported, nothing is
    flagged) and the soft range; without a scene the constants above apply.
    """
    import numpy as np  # noqa: PLC0415

    prior_deg = SEAM_DIRECTION_PRIOR_DEG if scene is None else scene.seam_prior_deg
    soft_deg = SEAM_DIRECTION_SOFT_RANGE_DEG if scene is None else scene.seam_soft_range_deg
    out = {"available": False, "prior_deg": prior_deg,
           "soft_range_deg": soft_deg,
           "seam_direction_outside_soft_range": False}
    try:
        if not seam:
            out["reason"] = "no_published_seam"
            return out
        n, d = _plane_nd((record.get("top_face") or {}).get("plane"))
        camera = camera_from_manifest(manifest)
        if camera is None:
            out["reason"] = "no_camera"
            return out
        pts = backproject_to_plane(np.stack([np.asarray(seam["seam_left"], dtype=np.float64),
                                             np.asarray(seam["seam_right"], dtype=np.float64)]),
                                   camera, n, d)
        v = pts[1] - pts[0]
        v = v - n * float(v @ n)
        norm = float(np.linalg.norm(v))
        if norm < 1e-6:
            out["reason"] = "degenerate_seam"
            return out
        v /= norm
        e1 = np.array([1.0, 0.0, 0.0]) - n * n[0]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        out["available"] = True
        out["plane_frame_angle_deg"] = round(
            math.degrees(math.atan2(float(v @ e2), float(v @ e1))) % 180.0, 2)
        out["direction_camera"] = [round(float(c), 5) for c in v]
        matrix = (_camera_to_base_matrix(calib_file, manifest, scene)
                  if camera_to_base is None else camera_to_base)
        if matrix is None:
            out["base_link_angle_deg"] = None
            out["reason"] = "calibration_unavailable(plane-frame angle only)"
            return out
        vb = np.asarray(matrix, dtype=np.float64)[:3, :3] @ v
        angle = math.degrees(math.atan2(float(vb[1]), float(vb[0]))) % 180.0
        if prior_deg is None:
            offset = None
            outside = False
        else:
            offset = (angle - prior_deg + 90.0) % 180.0 - 90.0
            outside = bool(abs(offset) > soft_deg)
        out.update({"base_link_angle_deg": round(angle, 2),
                    "offset_from_prior_deg": None if offset is None else round(offset, 2),
                    "seam_direction_outside_soft_range": outside,
                    "base_link_source": "calib(rebel_handeye_h_v2)+recorded_driver_extrinsic",
                    "base_link_calib_file": os.path.expanduser(str(calib_file or SEAM_DIRECTION_CALIB)),
                    "note": "approximate; soft prior, flag only (never rejects)"})
    except Exception as error:  # noqa: BLE001 - optional, never fail the detection
        out["available"] = False
        out["reason"] = f"seam_direction_error({type(error).__name__}: {error})"
    return out


def record_reject_reasons(record: dict, allow_needs_review: bool) -> list:
    """Scene-level rejection reasons of a v5 record ([] = acceptable)."""
    reasons = []
    top = record.get("top_face") or {}
    checks = record.get("checks") or {}
    if not top.get("corners_px"):
        reasons.append("no_top_face")
        reasons.extend(str(r) for r in (checks.get("reasons") or []))
        return reasons
    if checks.get("wrong_face_suspect"):
        reasons.append("wrong_face_suspect")
    mask_source = (record.get("provenance") or {}).get("mask_source")
    if mask_source != "sam3_live":
        reasons.append(f"mask_source_{mask_source}(not sam3_live)")
    if checks.get("needs_review") and not allow_needs_review:
        reasons.append("needs_review")
        reasons.extend(str(r) for r in (checks.get("reasons") or [])
                       if not str(r).startswith("seam_"))
    return reasons


# ---- operator hint for scenes whose top face cannot be measured ------------

HINT_OBLIQUE_DEG = 62.0      # camera ray vs table normal above this = too oblique
HINT_SATURATED_FRAC = 0.15   # fraction of pixels >= 250 above this = overexposed
HINT_SATURATED_LEVEL = 250
HINT_EDGE_PX = 5             # SAM3 box within this of the border = outside
HINT_RING_FRAC = 0.5         # surroundings: box grown by this x its size per side


def hint_suggestions(view_angle_deg, saturated_frac, box_px, image_wh) -> list:
    """Plain-language suggestions from the measured numbers (pure; any input
    may be None = not measured). Order: outside, oblique, overexposed; the
    generic 'not found' text only when nothing else applies."""
    out = []
    if box_px is not None and image_wh is not None:
        x0, y0, x1, y1 = box_px
        w, h = image_wh
        if (x0 <= HINT_EDGE_PX or y0 <= HINT_EDGE_PX
                or x1 >= w - HINT_EDGE_PX or y1 >= h - HINT_EDGE_PX):
            out.append("box partly outside the image")
    if view_angle_deg is not None and view_angle_deg > HINT_OBLIQUE_DEG:
        out.append(f"box top seen too obliquely ({view_angle_deg:.0f} deg) "
                   "- move it toward the table centre")
    if saturated_frac is not None and saturated_frac > HINT_SATURATED_FRAC:
        out.append(f"image overexposed ({100.0 * saturated_frac:.0f}% saturated) "
                   "- reduce light / lock camera exposure")
    if not out:
        out.append("top face not found - re-run or check placement")
    return out


def _sam3_box_px(record: dict):
    """(x0, y0, x1, y1) of the first SAM3-live candidate quad, or None."""
    for att in ((record.get("provenance") or {}).get("attempts") or []):
        if att.get("source") == "sam3_live" and not att.get("plane_trimmed") \
                and att.get("coarse"):
            try:
                q = [[float(a), float(b)] for a, b in att["coarse"]]
            except (TypeError, ValueError):
                continue
            xs, ys = [p[0] for p in q], [p[1] for p in q]
            return min(xs), min(ys), max(xs), max(ys)
    return None


def saturated_fraction(gray, box_px, level: int = HINT_SATURATED_LEVEL):
    """Fraction of pixels >= ``level`` in the SAM3 box grown by
    ``HINT_RING_FRAC`` x its size per side (box + surroundings)."""
    import math  # noqa: PLC0415

    h, w = gray.shape[:2]
    x0, y0, x1, y1 = box_px
    mx, my = HINT_RING_FRAC * (x1 - x0), HINT_RING_FRAC * (y1 - y0)
    a, b = max(0, int(math.floor(x0 - mx))), max(0, int(math.floor(y0 - my)))
    c, d = min(w, int(math.ceil(x1 + mx))), min(h, int(math.ceil(y1 + my)))
    if c <= a or d <= b:
        return None
    return float((gray[b:d, a:c] >= level).mean())


def compute_operator_hint(record: dict, bgr, manifest_path, pcd_path) -> dict:
    """Why the top face could not be measured + what the operator can do.
    Additive diagnostics only; never raises (missing pieces become null)."""
    hint = {"view_angle_deg": None, "visible_top_area_px": None,
            "saturated_frac": None, "sam3_box_px": None, "image_size": None,
            "suggestion": None, "suggestions": []}
    box = _sam3_box_px(record)
    h, w = bgr.shape[:2]
    hint["image_size"] = [w, h]
    sat = None
    if box is not None:
        hint["sam3_box_px"] = [round(v, 1) for v in box]
        import cv2  # noqa: PLC0415

        sat = saturated_fraction(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), box)
        hint["saturated_frac"] = None if sat is None else round(sat, 4)
        try:
            ang, area = _hint_depth_metrics(box, manifest_path, pcd_path)
            hint["view_angle_deg"] = None if ang is None else round(ang, 1)
            hint["visible_top_area_px"] = area
        except Exception as error:  # noqa: BLE001 — hint must never break a reject
            hint["depth_note"] = f"{type(error).__name__}: {error}"
    hint["suggestions"] = hint_suggestions(hint["view_angle_deg"], sat, box, (w, h))
    hint["suggestion"] = "; ".join(hint["suggestions"])
    return hint


def _hint_depth_metrics(box, manifest_path, pcd_path):
    """(view angle deg between the camera ray through the SAM3 box centre and
    the table normal, estimated visible top-face area in px) from the cloud.
    The area counts cloud points inside the box on the most populated
    table-parallel height slab 30-400 mm above the table (+-6 mm)."""
    import numpy as np  # noqa: PLC0415

    from ..core.autolabel import depth_top as dt  # noqa: PLC0415
    from ..core.autolabel.pcd import project_points, read_pcd_xyz  # noqa: PLC0415
    from ..core.config import default_config  # noqa: PLC0415

    cam_dict, _src = dt.load_capture_camera(str(manifest_path))
    k = np.array([[cam_dict["fx"], 0.0, cam_dict["cx"]],
                  [0.0, cam_dict["fy"], cam_dict["cy"]], [0.0, 0.0, 1.0]])
    xyz, _ = read_pcd_xyz(str(pcd_path))
    xyz = np.asarray(xyz, dtype=np.float64)
    dcfg = dict(default_config()["depth_top"])
    cropped = xyz[dt.workspace_mask(xyz, dcfg)]
    dcfg["max_planes"] = 1
    dcfg["top_min_inliers"] = int(dcfg["table_min_inliers"])
    planes = dt.sequential_planes(cropped, dcfg)
    if not planes:
        return None, None
    n, d = np.asarray(planes[0]["n"], dtype=np.float64), float(planes[0]["d"])
    if d < 0:
        n, d = -n, -d
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    ray = np.array([(cx - k[0, 2]) / k[0, 0], (cy - k[1, 2]) / k[1, 1], 1.0])
    ray /= np.linalg.norm(ray)
    angle = float(np.degrees(np.arccos(np.clip(abs(float(ray @ n)), 0.0, 1.0))))
    # visible top-face area estimate
    area = None
    try:
        uv, _z, _ = project_points(xyz, cam_dict)
        ok = np.isfinite(xyz[:, 2]) & (xyz[:, 2] > 0.05)  # project_points gate
        pts = xyz[ok]
        inbox = ((uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2])
                 & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3]))
        sub = pts[inbox]
        height_mm = (sub @ n + d) * 1000.0
        sel = (height_mm > 30.0) & (height_mm < 400.0)
        if sel.sum() >= 50:
            hh = height_mm[sel]
            bins = np.arange(30.0, 400.0 + 5.0, 5.0)
            hist, _ = np.histogram(hh, bins=bins)
            peak = float(bins[int(np.argmax(hist))] + 2.5)
            area = int((np.abs(height_mm - peak) <= 6.0).sum())
        else:
            area = 0
    except Exception:  # noqa: BLE001
        area = None
    return angle, area


def sam3_unavailable_reason(record: dict):
    """The sam3_live transport/worker failure reason, if any (-> exit 2)."""
    for entry in ((record.get("provenance") or {}).get("attempts") or []):
        reason = str(entry.get("reason") or "")
        if entry.get("source") == "sam3_live" and reason.startswith("sam3_live_unavailable"):
            return reason
    return None


def _seam_curvature_mm(record: dict):
    """Observed-gap curvature (mm) recorded by the seam estimator (observed-publish profiles), else None."""
    value = ((record.get("seam") or {}).get("estimate") or {}).get("seam_curvature_mm")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) else None


def build_canonical_seam_doc(record: dict, image_path: str, width: int, height: int,
                             capture_id: str, seam: dict, relabel: dict,
                             record_path: str, timing: dict,
                             allow_needs_review: bool, rect_check: dict = None,
                             depth_edge: dict = None, seam_direction: dict = None,
                             rect_repair: dict = None,
                             seam_geometry: dict = None,
                             cut_permission: dict = None,
                             box_profile: dict = None) -> dict:
    corners = relabel["corners"]
    top = record.get("top_face") or {}
    checks = record.get("checks") or {}
    xyz_rule = top.get("corners_xyz_camera_m") or None
    xyz_contract = ({name: xyz_rule.get(rule) for name, rule in relabel["contract_from_rule"].items()}
                    if isinstance(xyz_rule, dict) else None)
    sam3 = (record.get("provenance") or {}).get("sam3") or {}
    names_prov = (record.get("provenance") or {}).get("corner_names") or {}
    quad = [corners[name] for name in RING]
    return {
        "source_image": str(image_path),
        "image_size": [width, height],
        "capture_id": capture_id,
        "method": METHOD,
        "back_left": corners["back_left"],
        "back_right": corners["back_right"],
        "front_left": corners["front_left"],
        "front_right": corners["front_right"],
        "left_mid": relabel["left_mid"],
        "right_mid": relabel["right_mid"],
        # Contract names are a deterministic relabel of the rule names; no
        # left/right ambiguity is passed to the planner.
        "lr_ambiguous": False,
        "incomplete_groups": [],
        "seam_status": seam["status"],
        "cut_permission": cut_permission,
        "seam_observed": bool(seam["observed"]),
        **({"seam_curvature_mm": _seam_curvature_mm(record)}
           if _seam_curvature_mm(record) is not None else {}),
        # The published line (observed-confirmed or geometric midline) is
        # the physical seam for these cartons (flap gap measured centred).
        "cutting_target_valid": True,
        "provenance": {"coarse_source": "sam3_live", "sam3_inference": True,
                       "sam3_prompt": sam3.get("prompt"),
                       "sam3_concept_score": sam3.get("concept_score"),
                       "sam3_model_revision": sam3.get("model_revision")},
        "depth_first": {
            "score": record.get("confidence"),
            "generator": GENERATOR,
            "quad": quad,
            "operator_accepted": bool(allow_needs_review and checks.get("needs_review")),
        },
        "autolabel_v2": {
            "schema": record.get("schema"),
            "rule_corner_names": {name: top["corners_px"][name] for name in RING},
            "rule": names_prov.get("rule"),
            "rule_ambiguous": names_prov.get("ambiguous"),
            "rule_flags": names_prov.get("flags"),
            "contract_from_rule": relabel["contract_from_rule"],
            "contract_shift": relabel["shift"],
            "contract_corners_xyz_camera_m": xyz_contract,
            "seam_axis": seam["axis"],
            "seam_geometric_axis_rule": relabel["geometric_axis"],
            "seam_status": seam["status"],
            "seam_source": seam["seam_source"],
            "seam_endpoint_field": seam["source"],
            "seam_endpoints_image": {"seam_left": seam["seam_left"],
                                     "seam_right": seam["seam_right"]},
            "edge_distance_px": relabel["edge_distance_px"],
            "needs_review": bool(checks.get("needs_review")),
            "allow_needs_review": bool(allow_needs_review),
            "wrong_face_suspect": bool(checks.get("wrong_face_suspect")),
            "corner_flags": checks.get("corner_flags"),
            "reasons": checks.get("reasons"),
            "confidence": record.get("confidence"),
            "confidence_quad": record.get("confidence_quad"),
            "confidence_seam": record.get("confidence_seam"),
            "plane": top.get("plane"),
            "rect_check": rect_check,
            "depth_edge": depth_edge,
            "seam_direction": seam_direction,
            "rect_repair": rect_repair,
            "seam_geometry": seam_geometry,
            "box_profile": box_profile,
            "measured_height_mm": _measured_height_mm(record),
            "cut_permission": cut_permission,
            "reliability_rules": reliability_rules(),
            "soft_flags": (["seam_direction_outside_soft_range"]
                           if (seam_direction or {}).get("seam_direction_outside_soft_range")
                           else [])
                          + (["rectangle_repaired_joint_fit"]
                             if (rect_repair or {}).get("applied") else [])
                          + (["seam_axis_from_geometry"] if seam_geometry else []),
            "seam_status_downgraded": bool((depth_edge or {}).get("seam_status_downgraded")
                                           or (rect_repair or {}).get("seam_status_downgraded")),
            "record_path": str(record_path),
            "timing_s": timing,
        },
    }


def draw_contract_overlay(bgr, overlay_path: Path, doc: dict) -> None:
    import cv2  # noqa: PLC0415

    image = bgr.copy()
    height, width = image.shape[:2]
    thick = max(2, min(width, height) // 360)

    def ip(point):
        return (int(round(point[0])), int(round(point[1])))

    ring = [doc[name] for name in RING]
    for index in range(4):
        cv2.line(image, ip(ring[index]), ip(ring[(index + 1) % 4]), (0, 255, 0), thick)
    cv2.line(image, ip(doc["left_mid"]), ip(doc["right_mid"]), (255, 0, 255), thick + 1)
    for name in RING:
        cv2.circle(image, ip(doc[name]), thick + 2, (0, 255, 0), -1)
        cv2.putText(image, name, ip(doc[name]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
    for key, label in (("left_mid", "LM"), ("right_mid", "RM")):
        cv2.circle(image, ip(doc[key]), thick + 1, (255, 0, 255), 2)
        cv2.putText(image, label, ip(doc[key]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{METHOD} seam={doc['seam_status']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), image):
        raise OSError(f"cv2.imwrite failed for {overlay_path}")


def load_capture_manifest(manifest_path: Path) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Prerequisite(f"cannot read capture manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != "rgbd_capture/v1":
        raise Prerequisite(f"unsupported capture manifest schema: {manifest.get('schema')!r}")
    if not manifest.get("capture_id"):
        raise Prerequisite("capture manifest lacks capture_id")
    image = manifest.get("image")
    if (not isinstance(image, dict) or not isinstance(image.get("width"), int)
            or not isinstance(image.get("height"), int)
            or image["width"] <= 0 or image["height"] <= 0):
        raise Prerequisite("capture manifest lacks a usable image width/height")
    return manifest


def import_autolabel(repo=None):
    """The core library (``repo`` is ignored: kept for the old call signature)."""
    try:
        from ..core import pipeline  # noqa: PLC0415
        from ..core.config import default_config  # noqa: PLC0415
        from ..core.overlay import write_overlay  # noqa: PLC0415
        from ..core.sam3_session import Sam3Worker  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001 — any import failure is exit 2
        raise Prerequisite(f"cannot import box_seam_detection.core ({type(error).__name__}: {error})") from error
    return pipeline, default_config, Sam3Worker, write_overlay


def start_sam3_worker(Sam3Worker, python_bin: str, stderr_log: Path):
    if not Path(python_bin).is_file():
        raise Prerequisite(f"SAM3 interpreter not found: {python_bin}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return Sam3Worker(python_bin=python_bin, stderr_log=str(stderr_log)).start()
    except Exception as error:  # noqa: BLE001 — SAM3 unavailable is exit 2
        raise Prerequisite(f"SAM3 worker unavailable: {error} (stderr log: {stderr_log})") from error


# ---------------------------------------------------------------- CLI

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="autolabel_v2 (SAM3 live + fold + seam estimator) -> canonical "
                    "six-point seam JSON. Exit 0 accepted, 1 scene rejection "
                    "(no file), 2 prerequisite failure.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--out-seam-json", required=True)
    parser.add_argument("--work-dir", required=True,
                        help="scratch dir: full v5 record, SAM3 worker stderr log, "
                             "reject JSON when no --overlay-dir")
    parser.add_argument("--overlay-dir", default=None,
                        help="accepted: <capture_id>_autolabel_v2.png (contract names) + "
                             "<capture_id>_autolabel_v2_record.png (rule names); "
                             "rejected: <capture_id>_autolabel_v2_reject.json")
    parser.add_argument("--allow-needs-review", action="store_true",
                        help="accept records flagged checks.needs_review (still recorded)")
    parser.add_argument("--allow-cut-blocked", action="store_true",
                        help="accept captures whose cut_permission is blocked (still recorded "
                             "as blocked); default: reject at stage cut_permission")
    parser.add_argument("--edge-tol-px", type=float, default=DEFAULT_EDGE_TOL_PX,
                        help="max seam-endpoint distance to its contract edge, "
                             "0 < tol <= 5 (default 2.0)")
    parser.add_argument("--rect-tol-mm", type=float, default=None,
                        help="max |difference| of opposite 3-D top-face side lengths "
                             "(back/front, left/right), 0 < tol <= 20 mm (default: the scene's "
                             "rect_tol_mm, else 3.0); "
                             "larger -> exit 1 rectangle_inconsistent (retryable)")
    parser.add_argument("--rect-max-corner-dev-deg", type=float,
                        default=RECT_MAX_CORNER_DEV_DEG,
                        help="max deviation of a top-face corner angle from 90 deg (3-D), "
                             f"0 < dev <= {MAX_RECT_CORNER_DEV_DEG:g} (default "
                             f"{RECT_MAX_CORNER_DEV_DEG:g}); larger -> exit 1 "
                             "rectangle_skewed (retryable)")
    parser.add_argument("--no-rect-check", action="store_true",
                        help="disable the 3-D rectangle consistency gate")
    parser.add_argument("--no-depth-edge", action="store_true",
                        help="disable the depth-placed seam-crossed edge fix (applied "
                             "before the rectangle gate)")
    parser.add_argument("--autolabel-repo", default=None,
                        help="ignored (kept so old callers/replay harnesses still work): the "
                             "library is box_seam_detection.core")
    parser.add_argument("--sam3-python", default=DEFAULT_SAM3_PYTHON,
                        help="SAM3 worker interpreter (book_grasping env)")
    parser.add_argument("--calib-file", default=None,
                        help="hand-eye calibration used for the seam-direction soft prior (the same "
                             "file as the snapshot transform; default rebel_handeye_h_v2.calib)")
    parser.add_argument("--scene-config", default=None,
                        help="scene config (YAML path or shipped name, e.g. dual_arm_v1): depth-top "
                             "workspace/support mode, guided-plane gate, seam-direction prior, driver "
                             "extrinsic reference, default calib. Default: table_cell_v1 (today's values)")
    parser.add_argument("--sam3-timeout", type=int, default=300,
                        help="per-image SAM3 query timeout in seconds")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _say(message: str) -> None:
    print(f"{PROG}: {message}", file=sys.stderr)


def _remove_partial_output(out_path: Path) -> None:
    """Failure paths never leave a canonical seam JSON behind."""
    try:
        out_path.unlink(missing_ok=True)
    except OSError as error:
        _say(f"warning: cannot remove {out_path}: {error}")


class _OneShotSam3:
    """SAM3 session per run (one-shot CLI): a fresh worker is started for the
    capture (stderr log ``<work-dir>/<capture_id>_sam3_worker.stderr.log``)
    and closed after process_capture."""

    mode = "oneshot"

    def acquire(self, Sam3Worker, args, work_dir: Path, capture_id: str):
        return start_sam3_worker(Sam3Worker, args.sam3_python,
                                 work_dir / f"{capture_id}_sam3_worker.stderr.log")

    def release(self, session, work_dir: Path, capture_id: str, broken: bool = False) -> None:
        session.close()

    def close(self) -> None:
        pass


def _is_sam3_unavailable(error: BaseException) -> bool:
    """True for autolabel_v2's Sam3LiveUnavailable (matched by class name: the
    autolabel_v2 package is imported lazily)."""
    return any(cls.__name__ == "Sam3LiveUnavailable" for cls in type(error).__mro__)


def _session_alive(session) -> bool:
    """False when a Sam3Worker's subprocess is gone (killed after a timeout,
    crashed); objects without ``proc`` (test fakes) count as alive."""
    if not hasattr(session, "proc"):
        return True
    proc = session.proc
    return proc is not None and proc.poll() is None


class _ServerSam3:
    """SAM3 session owned by ``--serve``: started ONCE when the server starts
    (so the model loads while the caller is still capturing) and reused by
    every request. A worker that died since the previous request is replaced
    by a fresh one (what the one-shot CLI would do anyway). The per-request
    query wrapper is removed after each request; the server-level worker log
    is copied to ``<work-dir>/<capture_id>_sam3_worker.stderr.log`` so each
    capture keeps the same debug file as in one-shot mode."""

    mode = "server"

    def __init__(self, python_bin: str, stderr_log: Path):
        self.python_bin = str(python_bin)
        self.stderr_log = Path(stderr_log)
        self.session = None
        self.error = None
        self.load_s = None
        self.Sam3Worker = None

    def start(self, Sam3Worker) -> None:
        self.Sam3Worker = Sam3Worker
        t0 = time.monotonic()
        try:
            self.session = start_sam3_worker(Sam3Worker, self.python_bin, self.stderr_log)
        except Prerequisite as error:
            self.error = error
        self.load_s = round(time.monotonic() - t0, 2)

    def acquire(self, Sam3Worker, args, work_dir: Path, capture_id: str):
        if Path(args.sam3_python).resolve() != Path(self.python_bin).resolve():
            raise Prerequisite(f"detector server runs SAM3 with {self.python_bin}, "
                               f"request asks for --sam3-python {args.sam3_python}")
        if self.error is not None:
            raise self.error
        if self.session is None or not _session_alive(self.session):
            if self.session is not None:
                try:
                    self.session.close()
                except Exception:  # noqa: BLE001 — already dead
                    pass
                self.session = None
            _say("SAM3 worker is not running any more; starting a fresh one")
            self.session = start_sam3_worker(Sam3Worker, self.python_bin, self.stderr_log)
        return self.session

    def release(self, session, work_dir: Path, capture_id: str, broken: bool = False) -> None:
        """``broken``: a query raised Sam3LiveUnavailable (timeout, EOF, bad
        payload). The session is then dropped (closed) so the next request
        starts a fresh worker, as the one-shot CLI does for every capture."""
        # Drop the per-request timed_query wrapper (an instance attribute).
        session.__dict__.pop("query", None)
        try:
            if self.stderr_log.is_file():
                import shutil  # noqa: PLC0415
                shutil.copyfile(self.stderr_log,
                                work_dir / f"{capture_id}_sam3_worker.stderr.log")
        except OSError as error:
            _say(f"warning: cannot copy SAM3 worker log: {error}")
        if broken and session is self.session:
            _say("SAM3 worker query failed (Sam3LiveUnavailable); dropping the session, "
                 "a fresh worker starts on the next request")
            self.session = None
            try:
                session.close()
            except Exception as error:  # noqa: BLE001 — already broken
                _say(f"warning: closing the failed SAM3 worker: {error}")

    def close(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            session.close()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--serve" in argv:
        return serve(argv)
    args = _parse_args(argv)
    return run_request(args, _OneShotSam3())["exit_code"]


def run_request(args, sam3) -> dict:
    """One capture -> canonical seam JSON (the whole one-shot CLI behaviour).

    ``sam3``: :class:`_OneShotSam3` (CLI) or :class:`_ServerSam3` (--serve).
    Returns {"exit_code", "capture_id", "out_seam_json" (str|None),
    "reject_json" (str|None), "timing"}; files written and exit codes are the
    same for both session modes.
    """
    image_path = Path(args.image).resolve()
    pcd_path = Path(args.pcd).resolve()
    manifest_path = Path(args.capture_manifest).resolve()
    out_path = Path(args.out_seam_json).resolve()
    work_dir = Path(args.work_dir).resolve()
    overlay_dir = Path(args.overlay_dir).resolve() if args.overlay_dir else None
    result = {"exit_code": 2, "capture_id": None, "out_seam_json": None,
              "reject_json": None, "timing": None}
    # Fail closed: supersede any previous output before anything else.
    if out_path.exists() or out_path.is_symlink():
        try:
            out_path.unlink()
        except OSError as error:
            _say(f"prerequisite failure: cannot remove stale output {out_path}: {error}")
            return result
    if not (math.isfinite(args.edge_tol_px) and 0.0 < args.edge_tol_px <= MAX_EDGE_TOL_PX):
        _say(f"prerequisite failure: --edge-tol-px must be in (0, {MAX_EDGE_TOL_PX:g}], "
             f"got {args.edge_tol_px!r}")
        return result
    scene_arg = None            # None = the built-in table_cell_v1 (no file read, no config key)
    calib_arg = getattr(args, "calib_file", None)
    if getattr(args, "scene_config", None):
        from ..core import scene as _scene  # noqa: PLC0415
        try:
            scene_arg = _scene.load_scene(args.scene_config)
        except _scene.SceneError as error:
            _say(f"prerequisite failure: scene config: {error}")
            return result
        if not calib_arg:
            calib_arg = scene_arg.calib_file
    rect_tol = getattr(args, "rect_tol_mm", None)
    if rect_tol is None:
        rect_tol = (scene_arg.rect_tol_mm if scene_arg is not None and scene_arg.rect_tol_mm is not None
                    else DEFAULT_RECT_TOL_MM)
    if not (math.isfinite(rect_tol) and 0.0 < rect_tol <= MAX_RECT_TOL_MM):
        _say(f"prerequisite failure: --rect-tol-mm must be in (0, {MAX_RECT_TOL_MM:g}], "
             f"got {rect_tol!r}")
        return result
    corner_dev = float(getattr(args, "rect_max_corner_dev_deg", RECT_MAX_CORNER_DEV_DEG))
    if not (math.isfinite(corner_dev) and 0.0 < corner_dev <= MAX_RECT_CORNER_DEV_DEG):
        _say(f"prerequisite failure: --rect-max-corner-dev-deg must be in "
             f"(0, {MAX_RECT_CORNER_DEV_DEG:g}], got {corner_dev!r}")
        return result
    if "r7" in rules_off():
        corner_dev = 180.0          # R7 disabled via AUTOLABEL_V2_R8_OFF
    if args.sam3_timeout <= 0:
        _say(f"prerequisite failure: --sam3-timeout must be positive, got {args.sam3_timeout!r}")
        return result
    capture_id = None
    depth_edge = None
    operator_hint = None
    seam_direction = None
    rect_repair = None
    seam_geometry = None
    box_profile = None
    box_profile_info = None
    est_cfg = None
    timing = {}
    result["timing"] = timing
    try:
        for label, path in (("image", image_path), ("pcd", pcd_path),
                            ("capture manifest", manifest_path)):
            if not path.is_file():
                raise Prerequisite(f"{label} not found: {path}")
        manifest = load_capture_manifest(manifest_path)
        capture_id = manifest["capture_id"]
        result["capture_id"] = capture_id
        width, height = int(manifest["image"]["width"]), int(manifest["image"]["height"])
        if Path(manifest["image"].get("path", "")).resolve() != image_path:
            raise Prerequisite("manifest image path does not match --image")
        pc_path = (manifest.get("pointcloud") or {}).get("path")
        if pc_path and Path(pc_path).resolve() != pcd_path:
            raise Prerequisite("manifest pointcloud path does not match --pcd")
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise Prerequisite(f"cannot create work dir {work_dir}: {error}") from error
        pipeline, default_config, Sam3Worker, write_overlay = import_autolabel()
        try:
            import cv2  # noqa: PLC0415
        except ImportError as error:
            raise Prerequisite(f"cv2 unavailable: {error}") from error
        probe = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if probe is None:
            raise Prerequisite(f"cannot decode image {image_path}")
        if [probe.shape[1], probe.shape[0]] != [width, height]:
            raise Prerequisite(f"real image size {[probe.shape[1], probe.shape[0]]} != "
                               f"manifest {[width, height]}")

        t0 = time.monotonic()
        session = sam3.acquire(Sam3Worker, args, work_dir, capture_id)
        timing["sam3_load"] = round(time.monotonic() - t0, 2)
        if sam3.mode == "server":
            timing["sam3_session"] = "server"
            timing["sam3_server_load"] = sam3.load_s
        query_times = []
        query_unavailable = []
        original_query = session.query

        def timed_query(*q_args, **q_kwargs):
            # pipeline._live_coarse calls session.query(image) without a
            # timeout; apply --sam3-timeout unless the caller passes one.
            if len(q_args) < 2:
                q_kwargs.setdefault("timeout_s", int(args.sam3_timeout))
            q0 = time.monotonic()
            try:
                return original_query(*q_args, **q_kwargs)
            except Exception as error:
                if _is_sam3_unavailable(error):
                    query_unavailable.append(error)
                raise
            finally:
                query_times.append(time.monotonic() - q0)

        session.query = timed_query
        try:
            cfg = default_config(scene_arg) if scene_arg is not None else default_config()
            cfg.setdefault("sam3", {})["checkpoint_sha256"] = None
            if (cfg.get("depth_top") or {}).get("support") == "hidden_stand":
                # hidden_stand: the floor is the support family whose normal is gravity
                # (base_link +z in the camera frame, from the hand-eye calibration); without
                # a calibration the largest plane is used (tagged support_gravity_unchecked)
                try:
                    import numpy as _np  # noqa: PLC0415
                    c2b = _camera_to_base_matrix(calib_arg, manifest, scene_arg)
                    if c2b is not None:
                        up = _np.asarray(c2b, dtype=float)[:3, :3].T @ _np.array([0.0, 0.0, 1.0])
                        cfg["depth_top"]["gravity_cam"] = [round(float(v), 6) for v in up / _np.linalg.norm(up)]
                except Exception as _error:  # noqa: BLE001 - optional, falls back to unchecked
                    _say(f"WARNING: gravity direction unavailable ({type(_error).__name__}: {_error})")
            est_cfg = cfg.get("seam_estimator") if isinstance(cfg, dict) else None
            paths = {"image": str(image_path), "pcd": str(pcd_path),
                     "manifest": str(manifest_path)}
            t1 = time.monotonic()
            record, debug = pipeline.process_capture(
                capture_id, paths, cfg, source="sam3_live", session=session)
            timing["pipeline_total"] = round(time.monotonic() - t1, 2)
            timing["sam3_inference"] = round(sum(query_times), 2)
            timing["post_sam3"] = round(timing["pipeline_total"] - timing["sam3_inference"], 2)
        finally:
            sam3.release(session, work_dir, capture_id, broken=bool(query_unavailable))
        record_path = work_dir / f"{capture_id}_autolabel_v2_record.json"
        try:
            write_json_atomically(record_path, record)
        except (OSError, TypeError, ValueError) as error:
            raise Prerequisite(f"cannot write record {record_path}: {error}") from error
        unavailable = sam3_unavailable_reason(record)
        if unavailable:
            raise Prerequisite(f"SAM3 live inference unavailable: {unavailable}")
        if args.debug:
            _say(f"debug: timing {timing}; record {record_path}")

        reasons = record_reject_reasons(record, args.allow_needs_review)
        if reasons:
            if "no_top_face" in reasons:
                try:
                    operator_hint = compute_operator_hint(
                        record, probe, manifest_path, pcd_path)
                except Exception as hint_error:  # noqa: BLE001 — additive only
                    operator_hint = {"suggestion": "top face not found - re-run or check placement",
                                     "hint_error": f"{type(hint_error).__name__}: {hint_error}"}
            raise Reject("record", reasons)
        box_profile, box_profile_info = resolve_box_profile(record)
        # The published seam is needed by the depth edge fix; its rejection
        # is deferred until after the rectangle gate (unchanged stage order).
        try:
            seam = published_seam_line(record)
            seam_reject = None
        except Reject as error:
            seam, seam_reject = None, error
        if args.no_depth_edge:
            depth_edge = {"applied": False, "reason": "disabled(--no-depth-edge)"}
        else:
            gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY).astype("float64")
            record, seam, depth_edge = apply_depth_edge(
                record, seam, manifest, pcd_path, gray=gray,
                profile=box_profile)
        timing["depth_edge"] = depth_edge.get("time_s")
        record["depth_edge"] = depth_edge
        seam_direction = compute_seam_direction(record, seam, manifest, calib_file=calib_arg, scene=scene_arg)
        record["seam_direction"] = seam_direction
        try:
            write_json_atomically(record_path, record)
        except (OSError, TypeError, ValueError) as error:
            raise Prerequisite(f"cannot write record {record_path}: {error}") from error
        if args.debug:
            _say(f"debug: depth_edge {depth_edge}")
        if args.no_rect_check:
            rect = {"enabled": False}
        else:
            try:
                record, seam, rect, rect_repair = rectangle_gate_with_repair(
                    record, seam, manifest, float(rect_tol), corner_dev)
            except Reject as rect_error:
                rect_repair = getattr(rect_error, "repair", None)
                raise
            if rect_repair is not None:
                record["rect_repair"] = rect_repair
                seam_direction = compute_seam_direction(record, seam, manifest, calib_file=calib_arg, scene=scene_arg)
                record["seam_direction"] = seam_direction
                try:
                    write_json_atomically(record_path, record)
                except (OSError, TypeError, ValueError) as error:
                    raise Prerequisite(f"cannot write record {record_path}: {error}") from error
        if seam_reject is not None:
            if ("r10" in rules_off() or not seam_reject.reasons
                    or not str(seam_reject.reasons[0]).startswith("seam_status_ambiguous")
                    # R10 (axis from box geometry) is validated for the known boxes only
                    or not box_profile.geometry_axis_validated):
                raise seam_reject
            record, seam, seam_geometry = geometry_seam(
                record, seam_reject, est_cfg, box_profile)
            record["seam_geometry"] = seam_geometry
            seam_direction = compute_seam_direction(record, seam, manifest, calib_file=calib_arg, scene=scene_arg)
            record["seam_direction"] = seam_direction
        elif ((record.get("seam") or {}).get("estimate") or {}).get(
                "axis_source") == "geometry_class":
            # The pipeline's own seam stage already took the axis from geometry.
            seam_geometry = {"axis": seam.get("axis"), "axis_source": "geometry_class",
                             "path": "pipeline_seam_stage",
                             "flag": "seam_axis_from_geometry"}
            record["seam_geometry"] = seam_geometry
        if seam_geometry is not None:
            # Safety: a seam axis chosen from box geometry alone (no tape
            # evidence) whose base_link direction is outside the soft range,
            # or unknown, needs an operator's review before cutting.
            outside = bool(seam_direction.get("seam_direction_outside_soft_range"))
            if outside or seam_direction.get("base_link_angle_deg") is None:
                why = ("seam_axis_from_geometry_outside_soft_range" if outside
                       else "seam_axis_from_geometry_direction_unknown")
                checks = record.setdefault("checks", {})
                checks["needs_review"] = True
                checks["reasons"] = [*(checks.get("reasons") or []), why]
                seam_geometry["needs_review_reason"] = why
                if not args.allow_needs_review:
                    try:
                        write_json_atomically(record_path, record)
                    except (OSError, TypeError, ValueError) as error:
                        raise Prerequisite(f"cannot write record {record_path}: {error}") from error
                    raise Reject("seam", [
                        f"needs_review({why}, offset_from_prior="
                        f"{seam_direction.get('offset_from_prior_deg')}deg)"],
                        details={"seam_geometry": seam_geometry,
                                 "seam_direction": seam_direction})
            try:
                write_json_atomically(record_path, record)
            except (OSError, TypeError, ValueError) as error:
                raise Prerequisite(f"cannot write record {record_path}: {error}") from error
        relabel = contract_relabel(record["top_face"]["corners_px"],
                                   seam["seam_left"], seam["seam_right"],
                                   tol_px=float(args.edge_tol_px))
        if seam["axis"] in ("lr", "bf") and seam["axis"] != relabel["geometric_axis"]:
            raise Reject("contract_edges", [
                f"seam axis label {seam['axis']} disagrees with geometry "
                f"({relabel['geometric_axis']} w.r.t. rule names)"])
        for key in ("back_left", "back_right", "front_left", "front_right"):
            x, y = relabel["corners"][key]
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise Reject("contract_edges", [f"{key} [{x}, {y}] outside {width}x{height}"])
        multi = multi_box_suspect(record, box_profile)
        if multi is not None:
            operator_hint = {"suggestion": MULTI_BOX_HINT, "evidence": multi}
            raise Reject("multi_box", [f"multiple_boxes_suspect({', '.join(multi['evidence'])})"],
                         retryable=False, details={"multi_box": multi})
        cut_permission = evaluate_cut_permission(
            record, seam, seam_direction, seam_geometry, box_profile, (width, height),
            generic_cut_without_height=bool(scene_arg is not None
                                            and scene_arg.generic_cut_without_height))
        if cut_permission["status"] == "blocked":
            if not getattr(args, "allow_cut_blocked", False):
                try:
                    write_json_atomically(record_path, record)
                except (OSError, TypeError, ValueError) as error:
                    raise Prerequisite(f"cannot write record {record_path}: {error}") from error
                raise Reject("cut_permission",
                             [f"cut_blocked({r['code']}: {r['detail']})"
                              for r in cut_permission["reasons"]],
                             retryable=all(r["retryable"] for r in cut_permission["reasons"]),
                             details={"cut_permission": cut_permission,
                                      "box_profile": box_profile_info})
            cut_permission = {**cut_permission, "override": "allow_cut_blocked"}
        doc = build_canonical_seam_doc(record, str(image_path), width, height, capture_id,
                                       seam, relabel, str(record_path), timing,
                                       args.allow_needs_review, rect_check=rect,
                                       depth_edge=depth_edge, seam_direction=seam_direction,
                                       rect_repair=rect_repair, seam_geometry=seam_geometry,
                                       cut_permission=cut_permission,
                                       box_profile=box_profile_info)
        if overlay_dir is not None:
            bgr = debug.get("bgr") if isinstance(debug, dict) else None
            if bgr is None:
                bgr = probe
            try:
                write_overlay(overlay_dir / f"{capture_id}_autolabel_v2_record.png",
                              bgr, record, debug)
                draw_contract_overlay(bgr, overlay_dir / f"{capture_id}_autolabel_v2.png", doc)
            except Exception as error:  # noqa: BLE001
                raise Prerequisite(f"cannot write overlay: {error}") from error
        try:
            write_json_atomically(out_path, doc)
        except OSError as error:
            raise Prerequisite(f"cannot write {out_path}: {error}") from error
    except Prerequisite as error:
        _say(f"prerequisite failure: {error}")
        _remove_partial_output(out_path)
        return result
    except Reject as error:
        diagnostic = {"status": "rejected", "capture_id": capture_id,
                      "method": METHOD, "stage": error.stage, "reasons": error.reasons,
                      "retryable": error.retryable}
        if error.details:
            diagnostic["rectangle" if error.stage == "rectangle" else "details"] = error.details
        if depth_edge is not None:
            diagnostic["depth_edge"] = depth_edge
        if seam_direction is not None:
            diagnostic["seam_direction"] = seam_direction
        if rect_repair is not None:
            diagnostic["rect_repair"] = rect_repair
        diagnostic["reliability_rules"] = reliability_rules()
        diagnostic["cut_permission"] = reject_cut_permission(error)
        if box_profile_info is not None:
            diagnostic["box_profile"] = box_profile_info
        if operator_hint is not None:
            diagnostic["operator_hint"] = operator_hint
        reject_dir = overlay_dir or work_dir
        reject_path = reject_dir / f"{capture_id}_autolabel_v2_reject.json"
        try:
            write_json_atomically(reject_path, diagnostic)
            result["reject_json"] = str(reject_path)
        except OSError as write_error:
            _say(f"warning: cannot write rejection diagnostic: {write_error}")
        _say(f"rejected: {capture_id} stage={error.stage} reasons={error.reasons} "
             f"retryable={error.retryable} (no output file written)")
        if operator_hint is not None:
            _say(f"operator hint: {operator_hint.get('suggestion')} "
                 f"(view_angle_deg={operator_hint.get('view_angle_deg')}, "
                 f"visible_top_area_px={operator_hint.get('visible_top_area_px')}, "
                 f"saturated_frac={operator_hint.get('saturated_frac')})")
        _remove_partial_output(out_path)
        result["exit_code"] = 1
        return result
    except Exception as error:  # noqa: BLE001 — internal error, NOT a scene rejection
        _say(f"internal error (exit 2, not a scene rejection): {type(error).__name__}: "
             f"{error} (capture {capture_id}; no output file written)")
        if args.debug:
            import traceback  # noqa: PLC0415
            traceback.print_exc()
        _remove_partial_output(out_path)
        return result
    rect_note = (f"rect=A{rect['diffs_mm']['A']:.2f}/B{rect['diffs_mm']['B']:.2f}mm "
                 if rect.get("enabled") else "rect=off ")
    edge_note = (f"depth_edge={'applied' if depth_edge.get('applied') else 'no'}"
                 f"({depth_edge.get('anchor', '-')}) ")
    print(f"{PROG}: accepted {capture_id} seam={doc['seam_status']} "
          f"axis={seam['axis']} shift={relabel['shift']} {rect_note}{edge_note}"
          f"timing={timing} -> {out_path}")
    result["exit_code"] = 0
    result["out_seam_json"] = str(out_path)
    return result


# ---------------------------------------------------------------- server

SERVER_PROTOCOL = 1
# Request keys -> one-shot CLI flags (a request may instead carry "argv": the
# exact one-shot argument list, which is what the orchestrator sends).
_REQUEST_VALUE_FLAGS = {
    "image": "--image", "pcd": "--pcd", "capture_manifest": "--capture-manifest",
    "out_seam_json": "--out-seam-json", "work_dir": "--work-dir",
    "overlay_dir": "--overlay-dir", "edge_tol_px": "--edge-tol-px",
    "rect_tol_mm": "--rect-tol-mm", "rect_max_corner_dev_deg": "--rect-max-corner-dev-deg",
    "autolabel_repo": "--autolabel-repo",
    "sam3_python": "--sam3-python", "sam3_timeout": "--sam3-timeout", "calib_file": "--calib-file",
    "scene_config": "--scene-config",
}
_REQUEST_SWITCHES = {"allow_needs_review": "--allow-needs-review",
                     "allow_cut_blocked": "--allow-cut-blocked",
                     "no_rect_check": "--no-rect-check",
                     "no_depth_edge": "--no-depth-edge", "debug": "--debug"}


def request_argv(request: dict) -> list:
    """One-shot CLI argv of a server request (raises ValueError when malformed)."""
    if not isinstance(request, dict):
        raise ValueError("request is not a JSON object")
    if "argv" in request:
        argv = request["argv"]
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError("'argv' must be a list of strings")
        extra = sorted(set(request) - {"argv", "id", "type"})
        if extra:
            raise ValueError(f"'argv' requests take no other options: {extra}")
        return list(argv)
    argv = []
    for key, value in request.items():
        if key in ("id", "type"):
            continue
        if key in _REQUEST_VALUE_FLAGS:
            if value is None:
                continue
            argv += [_REQUEST_VALUE_FLAGS[key], str(value)]
        elif key in _REQUEST_SWITCHES:
            if value:
                argv.append(_REQUEST_SWITCHES[key])
        else:
            raise ValueError(f"unknown request key {key!r}")
    return argv


def _exit_code_of(code) -> int:
    """sys.exit(code) -> process exit status (None 0, int as is, other 1)."""
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def _serve_one(argv: list, sam3) -> dict:
    try:
        args = _parse_args(argv)
    except SystemExit as stop:  # argparse error/--help: same status as the CLI
        return {"exit_code": _exit_code_of(stop.code), "capture_id": None,
                "out_seam_json": None, "reject_json": None, "timing": None}
    try:
        return run_request(args, sam3)
    except SystemExit as stop:
        return {"exit_code": _exit_code_of(stop.code), "capture_id": None,
                "out_seam_json": None, "reject_json": None, "timing": None}
    except Exception:  # noqa: BLE001 — an uncaught error exits 1 in the CLI
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        return {"exit_code": 1, "capture_id": None, "out_seam_json": None,
                "reject_json": None, "timing": None}


def _parse_serve_args(argv):
    parser = argparse.ArgumentParser(
        prog=f"{PROG} --serve",
        description="Detector server: loads the SAM3 worker immediately, then reads "
                    "one JSON request per line on stdin and writes one JSON line per "
                    "request on stdout (see serve()).")
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--work-dir", required=True,
                        help="server scratch dir (SAM3 worker stderr log)")
    parser.add_argument("--autolabel-repo", default=None)   # ignored (compat)
    parser.add_argument("--sam3-python", default=DEFAULT_SAM3_PYTHON)
    return parser.parse_args(argv)


def serve(argv, stdin=None, protocol_out=None) -> int:
    """``--serve``: persistent detector with a preloaded SAM3 worker.

    Starts the SAM3 worker at once (the caller captures meanwhile), then
    writes ONE status line: {"type": "ready"} or {"type": "startup_error",
    "reason"} (autolabel_v2 import / SAM3 worker failure; requests are still
    answered, each exiting 2 like the one-shot CLI would). Then, per stdin
    line (a JSON object: {"id", "argv": [one-shot CLI args]} or the keys of
    :data:`_REQUEST_VALUE_FLAGS` / :data:`_REQUEST_SWITCHES`), runs exactly
    the one-shot per-capture code (:func:`run_request`) and writes
    {"type": "response", "id", "exit_code", "capture_id", "out_seam_json",
    "reject_json", "timing", "elapsed_s"}. {"type": "shutdown"} or EOF stops
    the server and closes the SAM3 worker. Everything else the detector
    prints goes to stderr (stdout carries only protocol lines).
    """
    if protocol_out is None:
        # Protocol on a private dup of fd 1; fd 1 itself (print(), C-level
        # writes, child processes) is redirected to stderr.
        sys.stdout.flush()
        protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        os.dup2(2, 1)
    stdin = sys.stdin if stdin is None else stdin

    def emit(message: dict) -> None:
        sys.stdout.flush()
        protocol_out.write(json.dumps(message) + "\n")
        protocol_out.flush()

    try:
        args = _parse_serve_args(argv)
    except SystemExit as stop:
        emit({"type": "startup_error", "ok": False, "protocol": SERVER_PROTOCOL,
              "reason": f"invalid server arguments (exit {_exit_code_of(stop.code)})"})
        return 2
    work_dir = Path(args.work_dir).resolve()
    sam3 = _ServerSam3(args.sam3_python, work_dir / "sam3_worker.stderr.log")
    # Everything after the SAM3 session exists runs under the finally, so a
    # failure while reporting startup (e.g. BrokenPipe: the client is gone)
    # still closes the SAM3 worker.
    try:
        try:
            try:
                work_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise Prerequisite(f"cannot create server work dir {work_dir}: {error}") from error
            _, _, Sam3Worker, _ = import_autolabel()
            sam3.start(Sam3Worker)
        except Prerequisite as error:
            sam3.error = error
        if sam3.error is None:
            emit({"type": "ready", "ok": True, "protocol": SERVER_PROTOCOL, "pid": os.getpid(),
                  "sam3_load_s": sam3.load_s, "sam3_stderr_log": str(sam3.stderr_log)})
        else:
            _say(f"server startup failure: {sam3.error}")
            emit({"type": "startup_error", "ok": False, "protocol": SERVER_PROTOCOL,
                  "pid": os.getpid(), "reason": str(sam3.error)})
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            request_id = None
            try:
                request = json.loads(line)
                if isinstance(request, dict):
                    request_id = request.get("id")
                    if request.get("type") == "shutdown":
                        break
                request_argv_list = request_argv(request)
            except ValueError as error:
                _say(f"server: malformed request: {error}")
                emit({"type": "response", "id": request_id, "exit_code": 2,
                      "error": f"malformed request: {error}", "capture_id": None,
                      "out_seam_json": None, "reject_json": None, "timing": None})
                continue
            t0 = time.monotonic()
            response = _serve_one(request_argv_list, sam3)
            emit({"type": "response", "id": request_id, **response,
                  "elapsed_s": round(time.monotonic() - t0, 3)})
    finally:
        sam3.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
