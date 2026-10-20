# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/pipeline.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Per-capture v2 pipeline: coarse -> refine -> depth check -> seam -> record.

Corner order: [BL, BR, FR, FL] everywhere (ring order). Which physical
corner gets which name is fixed at the end by corner_names (rule
long_side_back_v1: back = long side farther from the camera). Seam endpoints come only from
the estimator's observed tape centreline (unblended fit of per-slice
paired tape edges); quad-edge midpoints are NEVER emitted as seam.

Seam endpoint construction: the accepted rectified seam line is intersected
with the top-face boundary (infinite line x rect bounds, mapped back to
source pixels), so endpoints sit ON the face boundary. ``seam_left`` /
``seam_right`` are image-relative: sorted by x (u), the smaller-u endpoint
is left. They are NOT box-fixed (which flap); the axis field (long/short)
records the box-relative direction instead.

``--source auto`` fallback chain: sam3_live -> sam3_archived (only when it
is a different quad) -> depth_only. Every attempt and its reject reasons
are recorded in ``provenance.attempts`` (and merged into
``checks.reasons``); the first attempt passing refinement AND the depth
cross-check wins. An attempt failing the depth cross-check is never picked
just to fill a slot.
"""

from __future__ import annotations

import copy
import os
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from . import box_profiles as _bp
from . import coarse as _coarse
from . import corner_checks as _corner_checks
from .config import CORNER_ORDER, default_config
from .sam3_live import Sam3LiveUnavailable


def _code_git_rev() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rev = (proc.stdout or "").strip()
    return rev or None


def _quad_iou(a: np.ndarray, b: np.ndarray, w: int, h: int) -> float | None:
    try:
        ma = np.zeros((h, w), np.uint8)
        mb = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(ma, np.asarray(a, dtype=np.float64).astype(np.int32), 255)
        cv2.fillConvexPoly(mb, np.asarray(b, dtype=np.float64).astype(np.int32), 255)
    except cv2.error:
        return None
    inter = float(((ma > 0) & (mb > 0)).sum())
    union = float(((ma > 0) | (mb > 0)).sum())
    return float(inter / union) if union > 0 else 0.0


def _edge_terms(refined) -> tuple[float | None, float | None]:
    rms_vals, sup_vals = [], []
    for e in getattr(refined, "edges", []) or []:
        r = float(getattr(e, "residual_rms_px", float("nan")))
        s = float(getattr(e, "support_score", 0.0))
        if np.isfinite(r):
            rms_vals.append(r)
        if np.isfinite(s):
            sup_vals.append(s)
    rms = float(np.mean(rms_vals)) if rms_vals else None
    sup = float(np.mean(sup_vals)) if sup_vals else None
    return rms, sup


def _concept_term(score) -> float | None:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(s):
        return None
    return float(min(1.0, max(0.0, s)))


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def compute_confidence_quad(concept, edge_rms_px, plane_inlier_ratio, iou,
                            v2cfg: dict) -> float:
    """Quad confidence: weighted mean over PRESENT evidence only.

    Terms (each mapped to [0, 1]):
    - concept: raw SAM3 concept score (w_concept)
    - edge: 1 - edge_rms/edge_rms_full_px (w_edge)
    - plane: eroded-interior inlier share (w_plane) — support evidence,
      NOT inlier RMS (which is < 3 mm by construction)
    - iou: independent RGB/depth quad agreement (w_iou)
    Missing evidence (None) is SKIPPED WITHOUT renormalising away the
    penalty: the weight stays in the denominator as a 0 term. Only the
    explicit ``max_conf_no_depth_check`` cap handles the depth-missing
    case (see process_capture). Rationale: absent validation must lower
    confidence, never hide.
    """
    num, den = 0.0, 0.0
    c = _concept_term(concept)
    den += float(v2cfg["w_concept"])
    num += float(v2cfg["w_concept"]) * (c if c is not None else 0.0)
    den += float(v2cfg["w_edge"])
    if edge_rms_px is not None and np.isfinite(edge_rms_px):
        num += float(v2cfg["w_edge"]) * _clamp01(
            1.0 - float(edge_rms_px) / float(v2cfg["edge_rms_full_px"]))
    den += float(v2cfg["w_plane"])
    if plane_inlier_ratio is not None and np.isfinite(plane_inlier_ratio):
        num += float(v2cfg["w_plane"]) * _clamp01(float(plane_inlier_ratio))
    den += float(v2cfg["w_iou"])
    if iou is not None and np.isfinite(iou):
        num += float(v2cfg["w_iou"]) * _clamp01(float(iou))
    return _clamp01(num / den) if den > 0 else 0.0


def compute_confidence_seam(seam_status: str, seam_score) -> float:
    """Seam confidence: raw seam score when ok, else 0/low.

    Estimated/ambiguous/invisible seams carry NO cutting evidence, so they
    score 0.0 rather than being dropped from a mean. A weak-but-accepted
    score below the detector floor is also 0.0 (callers only pass accepted
    scores, so this is a backstop).
    """
    if seam_status != "ok":
        return 0.0
    try:
        s = float(seam_score)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(s) or s <= 0:
        return 0.0
    return _clamp01(s)


def _load_manifest(manifest_path: str) -> dict:
    import json

    with open(manifest_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    return doc


def _quads_same(a: np.ndarray, b: np.ndarray, tol_px: float = 1.0) -> bool:
    try:
        aa = np.asarray(a, dtype=np.float64).reshape(4, 2)
        bb = np.asarray(b, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return False
    if aa.shape != (4, 2) or bb.shape != (4, 2):
        return False
    return bool(np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
                and float(np.abs(aa - bb).max()) <= tol_px)


# ----------------------------------------------------------------------
# Depth cross-check (shared by every attempt)
# ----------------------------------------------------------------------

ROI_DILATE_FRAC = 0.6      # ROI = candidate bbox grown by 60% of its size per side
ROI_CENTROID_DILATE = 1.5  # candidate quad scaled x1.5 about its centroid


def _global_quad_elsewhere(cand_quad: np.ndarray, dquad: np.ndarray,
                           iou: float | None) -> bool:
    """True when the global depth-top quad does not overlap the candidate:
    IoU == 0, or its centroid lies outside the candidate quad dilated 50%."""
    if iou is not None and iou <= 0.0:
        return True
    q = np.asarray(cand_quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    big = (c + (q - c) * ROI_CENTROID_DILATE).astype(np.float32)
    g = np.asarray(dquad, dtype=np.float64).reshape(4, 2).mean(axis=0)
    return cv2.pointPolygonTest(big.reshape(-1, 1, 2),
                                (float(g[0]), float(g[1])), False) < 0


def _depth_projection(depth_ctx: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(valid-row indices, Nx2 pixel projections) of the full cloud, cached
    in ``depth_ctx``. (None, None) when the cloud/camera is missing."""
    import sys

    from .autolabel.pcd import project_points

    xyz, cam = depth_ctx.get("xyz"), depth_ctx.get("cam")
    if xyz is None or cam is None:
        return None, None
    proj = depth_ctx.get("_proj")
    if proj is None:
        xyz = np.asarray(xyz, dtype=np.float64)
        ok = np.isfinite(xyz[:, 2]) & (xyz[:, 2] > 0.05)  # project_points gate
        uv, _z, _ = project_points(xyz, cam)
        proj = depth_ctx["_proj"] = (np.nonzero(ok)[0], uv)
    return proj


def _depth_top_roi(quad: np.ndarray, W: int, H: int, cfg: dict,
                   depth_ctx: dict) -> tuple[dict | None, list]:
    """Re-run ``estimate_top`` on the cloud points that project inside the
    candidate bbox dilated by 60% (clipped). The full-cloud projection is
    computed once and cached in ``depth_ctx``. Returns (estimate|None, roi)."""
    import sys

    from .autolabel.pcd import project_points
    from .autolabel import depth_top as _depth

    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    mx, my = ROI_DILATE_FRAC * (x1 - x0), ROI_DILATE_FRAC * (y1 - y0)
    roi = [int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my))),
           int(min(W, np.ceil(x1 + mx))), int(min(H, np.ceil(y1 + my)))]
    xyz, cam = depth_ctx.get("xyz"), depth_ctx.get("cam")
    if xyz is None or cam is None:
        return None, roi
    idx, uv = _depth_projection(depth_ctx)
    sel = ((uv[:, 0] >= roi[0]) & (uv[:, 0] < roi[2])
           & (uv[:, 1] >= roi[1]) & (uv[:, 1] < roi[3]))
    sub = np.asarray(xyz, dtype=np.float64)[idx[sel]]
    kw = {}
    if support_hidden(depth_ctx):
        # the sub-cloud does not contain the floor: reuse the full-cloud one
        tbl = estimate_table_plane(depth_ctx)
        if tbl is not None:
            kw["support_hint"] = tbl
    try:
        if support_hidden(depth_ctx):
            kw["prefer_quad"] = q
        est = _depth.estimate_top(sub, cam, (H, W), cfg["depth_top"], **kw)
    except Exception:  # noqa: BLE001 — retry failure == unavailable
        return None, roi
    return est, roi


# ----------------------------------------------------------------------
# Depth-plane-trimmed fallback outline
# ----------------------------------------------------------------------

GROSS_RECT_MM = 15.0         # opposite 3-D sides differ more than this -> try trim
TRIM_PLANE_BAND_MM = 5.0     # cloud points kept within this of the top plane
TRIM_QUAD_DILATE = 1.10      # candidate quad scaled about its centroid
TRIM_MIN_POINTS = 200        # fewer plane points than this -> no trimmed quad


def _rect_diffs_mm(corners_xyz) -> tuple[float, float] | None:
    """(|back-front|, |left-right|) in mm from ring-ordered 3-D corners.

    Opposite ring sides (0-2, 1-3) are invariant under the corner renaming,
    so this equals the live detector's rectangle_check pairs. None when the
    corners are missing or not finite."""
    if corners_xyz is None:
        return None
    c = np.asarray(corners_xyz, dtype=np.float64).reshape(-1, 3)
    if c.shape != (4, 3) or not np.all(np.isfinite(c)):
        return None
    s = [1000.0 * float(np.linalg.norm(c[i] - c[(i + 1) % 4]))
         for i in range(4)]
    return abs(s[0] - s[2]), abs(s[1] - s[3])


TABLE_PARALLEL_DEG = 10.0    # top plane normal within this of the table normal
BOX_H_MIN_MM = 30.0          # plausible box height above the table
BOX_H_MAX_MM = 700.0          # measurement range (boxes up to ~0.6 m tall, plan M3)
#: top-plane SELECTION limit for known boxes (and anything not proven tall):
#: the trimmed fallback takes the highest table-parallel plane, so a hand /
#: arm 0.4-0.7 m above the table must not win. 700 applies only inside
#: :func:`_h_max_scope` (generic candidate, or measured height > 0.35 m).
BOX_H_MAX_KNOWN_MM = 400.0
BOX_H_TALL_MM = 350.0
TRIM_MAX_PLANES = 3          # sequential RANSAC planes inside the quad


def _dcfg_of(depth_ctx: dict) -> dict:
    return depth_ctx.get("dcfg") or default_config()["depth_top"]


def support_hidden(depth_ctx: dict | None) -> bool:
    """True when the scene's support model is ``hidden_stand`` (the "table"
    plane of ``depth_ctx`` is then the FLOOR: heights above it are stand + box,
    never the box height)."""
    if not depth_ctx:
        return False
    return str((_dcfg_of(depth_ctx) or {}).get("support",
                                               "visible_table")) == "hidden_stand"


def _apply_scene_limits(depth_ctx: dict | None) -> None:
    """hidden_stand: floor-relative plane heights run up to ``offset_max`` (the
    scene's stand + box range) instead of the table-cell box limits."""
    if depth_ctx is not None and support_hidden(depth_ctx):
        lim = float(_dcfg_of(depth_ctx)["offset_max"]) * 1000.0
        depth_ctx["box_h_max_mm"] = lim
        depth_ctx["box_h_scope_max_mm"] = lim


def estimate_table_plane(depth_ctx: dict) -> dict | None:
    """Support (table) plane of the scene: the dominant plane of the
    workspace cloud, as ``estimate_top`` identifies it (largest sequential
    RANSAC plane). Cached in ``depth_ctx``. Returns ``{"n", "d"}`` with the
    normal oriented toward the camera (``d > 0``), or None."""
    if "_table" in depth_ctx:
        return depth_ctx["_table"]
    import sys

    from .autolabel import depth_top as _depth

    table = None
    try:
        dcfg = dict(_dcfg_of(depth_ctx))
        xyz = np.asarray(depth_ctx["xyz"], dtype=np.float64)
        cropped = xyz[_depth.workspace_mask(xyz, dcfg)]
        hidden = support_hidden(depth_ctx)
        fixed_cam = dcfg.get("fixed_floor") if hidden else None
        if fixed_cam:
            # scene option live.fixed_floor: the support is the measured plane, checked against this cloud
            # (fail closed: None + the diagnostics when the floor points do not sit on it)
            from . import fixed_floor as _ff  # noqa: PLC0415
            ftab, fdiag = _ff.evaluate(cropped, fixed_cam)
            depth_ctx["_table_fixed_floor"] = fdiag
            if ftab is not None:
                table = {"n": np.asarray(ftab["n"], dtype=np.float64), "d": float(ftab["d"]),
                         "n_inliers": int(ftab["n_inliers"]),
                         "rms_mm": float(ftab["rms"]) * 1000.0, "fixed": True}
            depth_ctx["_table"] = table
            return table
        if not hidden:
            dcfg["max_planes"] = 1
            dcfg["top_min_inliers"] = int(dcfg["table_min_inliers"])
        planes = _depth.sequential_planes(cropped, dcfg)
        if planes:
            # hidden_stand: the support is the farthest large plane family
            # (what estimate_top picks), not the first RANSAC plane (a
            # foreground table can be found before the floor fragments)
            k, _sd = 0, {}
            if hidden:
                # farthest large gravity-aligned plane family (floor), shared
                # with depth_top.identify_table_top; fail closed (no support)
                # when a calibration is present but no such family exists
                k, _sd = _depth.pick_hidden_support(planes, dcfg)
            if k is not None:
                n, d = np.asarray(planes[k]["n"], dtype=np.float64), planes[k]["d"]
                if d < 0:
                    n, d = -n, -d
                table = {"n": n, "d": float(d),
                         "n_inliers": int(planes[k]["n_inliers"]),
                         "rms_mm": float(planes[k]["rms"]) * 1000.0}
                if hidden and _sd.get("support_gravity_unchecked"):
                    table["support_gravity_unchecked"] = True
    except Exception:  # noqa: BLE001 — unavailable == dominant fallback
        table = None
    depth_ctx["_table"] = table
    return table


def _table_relation(n, d, table: dict, ref) -> tuple[float, float]:
    """(angle_deg to the table normal, height_mm above the table) for the
    plane ``n.p + d = 0``; height measured along the table normal at the
    the plane point nearest ``ref`` (positive = toward the camera)."""
    n = np.asarray(n, dtype=np.float64).ravel()
    nn = max(float(np.linalg.norm(n)), 1e-12)
    cosa = float(np.clip(abs(n @ table["n"]) / nn, 0.0, 1.0))
    ang = float(np.degrees(np.arccos(cosa)))
    ref = np.asarray(ref, dtype=np.float64).ravel()
    # point of the plane nearest ``ref`` (the candidate's cloud centroid)
    foot = ref - (float(n @ ref) + float(d)) * n / (nn * nn)
    height = (float(table["n"] @ foot) + table["d"]) * 1000.0
    return ang, height


def _choice_ok(ang: float, h_mm: float,
               h_max_mm: float = BOX_H_MAX_KNOWN_MM) -> bool:
    return ang < TABLE_PARALLEL_DEG and BOX_H_MIN_MM < h_mm < h_max_mm


class _h_max_scope:
    """Context manager: allow table-parallel top planes up to BOX_H_MAX_MM
    (``depth_ctx["box_h_max_mm"]``) while ``enabled``; no-op otherwise."""

    def __init__(self, depth_ctx, enabled: bool):
        self.ctx = depth_ctx if (depth_ctx and enabled) else None

    def __enter__(self):
        if self.ctx is not None:
            self.prev = self.ctx.get("box_h_max_mm")
            self.ctx["box_h_max_mm"] = float(self.ctx.get(
                "box_h_scope_max_mm", BOX_H_MAX_MM))
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            if self.prev is None:
                self.ctx.pop("box_h_max_mm", None)
            else:
                self.ctx["box_h_max_mm"] = self.prev
        return False


def _tall_quad(quad, depth_ctx, W: int, H: int) -> bool:
    """True when the quad's own central cloud height says generic/tall."""
    hq = quad_height_mm(quad, depth_ctx, W, H)
    return bool(hq is not None and (_bp.is_generic_height(hq)
                                    or hq > BOX_H_TALL_MM))


def measure_box_height(depth_ctx: dict | None, plane, corners_xyz,
                       top_rms_mm=None, top_inliers=None) -> dict:
    """Measured box height: the top-face centroid's distance to the table.

    ``plane`` is the fitted top plane (``.normal``/``.d``), ``corners_xyz``
    the camera-frame top corners (their centroid lies on it). Returns
    ``{"measured_height_mm", "height_uncertainty_mm", "table_inliers",
    "table_rms_mm", "top_rms_mm", "top_inliers", "height_unavailable"}``;
    on failure the height is None and ``height_unavailable`` says why (never
    a guess: no table plane, plane not table-parallel, top not above the
    table or outside [BOX_H_MIN_MM, BOX_H_MAX_MM])."""
    out: dict = {"measured_height_mm": None, "height_uncertainty_mm": None,
                 "table_inliers": None, "table_rms_mm": None,
                 "top_rms_mm": None if top_rms_mm is None
                 else round(float(top_rms_mm), 2),
                 "top_inliers": None if top_inliers is None
                 else int(top_inliers),
                 "height_unavailable": None}
    try:
        if not depth_ctx:
            out["height_unavailable"] = "no_depth"
            return out
        if support_hidden(depth_ctx):
            # the plane below the box is the floor under a stand that hides the
            # real support: its distance is stand + box, never the box height
            out["height_unavailable"] = "support_hidden"
            # recorded only: top-to-floor distance along the floor normal = box + stand,
            # an UPPER bound on the box height (never used as the height)
            try:
                tbl = estimate_table_plane(depth_ctx)
                cc = (None if corners_xyz is None
                      else np.asarray(corners_xyz, dtype=np.float64).reshape(-1, 3))
                if tbl is not None and cc is not None and cc.shape == (4, 3) \
                        and np.all(np.isfinite(cc)):
                    clr = (float(tbl["n"] @ cc.mean(axis=0)) + tbl["d"]) * 1000.0
                    if np.isfinite(clr):
                        out["support_clearance_mm"] = round(float(clr), 1)
                    if tbl.get("support_gravity_unchecked"):
                        out["support_gravity_unchecked"] = True
                    if tbl.get("fixed"):
                        out["support_source"] = "fixed_floor"
                elif tbl is None and depth_ctx.get("_table_fixed_floor") is not None:
                    # fixed floor configured but the cloud contradicts it: no support, say why
                    out["support_unavailable"] = "fixed_floor_mismatch"
                    out["fixed_floor"] = depth_ctx["_table_fixed_floor"]
            except Exception:  # noqa: BLE001
                pass
            return out
        table = estimate_table_plane(depth_ctx)
        if table is None:
            out["height_unavailable"] = "no_table_plane"
            return out
        out["table_inliers"] = table.get("n_inliers")
        if table.get("rms_mm") is not None:
            out["table_rms_mm"] = round(float(table["rms_mm"]), 2)
        c = (None if corners_xyz is None
             else np.asarray(corners_xyz, dtype=np.float64).reshape(-1, 3))
        if plane is None or c is None or c.shape != (4, 3) \
                or not np.all(np.isfinite(c)):
            out["height_unavailable"] = "no_top_plane"
            return out
        ang, h = _table_relation(np.asarray(plane.normal, dtype=np.float64),
                                 float(plane.d), table, c.mean(axis=0))
        if not np.isfinite(h):
            out["height_unavailable"] = "non_finite"
        elif ang >= TABLE_PARALLEL_DEG:
            out["height_unavailable"] = f"top_not_table_parallel({ang:.1f}deg)"
        elif not (BOX_H_MIN_MM <= h <= BOX_H_MAX_MM):
            out["height_unavailable"] = f"height_out_of_range({h:.0f}mm)"
        else:
            out["measured_height_mm"] = round(float(h), 2)
            terms = [v for v in (out["top_rms_mm"], out["table_rms_mm"])
                     if v is not None]
            out["height_uncertainty_mm"] = (
                round(float(np.sqrt(sum(v * v for v in terms))), 2)
                if terms else None)
    except Exception as exc:  # noqa: BLE001 — measurement never breaks a record
        out["measured_height_mm"] = None
        out["height_unavailable"] = f"error({type(exc).__name__})"
    return out


def early_height_mm(depth_est) -> float | None:
    """Box height from the independent depth estimate (``estimate_top``
    top-vs-table plane offset), available before the SAM3 quad; None when
    that estimate failed."""
    try:
        if not depth_est or not depth_est.get("ok"):
            return None
        h = float((depth_est.get("depth") or {}).get("top_offset_mm"))
    except (TypeError, ValueError):
        return None
    return h if (np.isfinite(h) and BOX_H_MIN_MM <= h <= BOX_H_MAX_MM) else None


def quad_height_mm(quad: np.ndarray, depth_ctx: dict | None, W: int, H: int,
                   core: float = 0.5) -> float | None:
    """Robust (90th percentile) height (mm) above the table of the cloud
    points inside the CENTRAL ``core`` fraction of a (coarse) quad, i.e. of the top face even
    when the outline spills onto side walls. None when there is no table
    plane / too few points. Never raises."""
    try:
        if not depth_ctx:
            return None
        table = estimate_table_plane(depth_ctx)
        if table is None:
            return None
        idx, uv = _depth_projection(depth_ctx)
        if idx is None:
            return None
        q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        c = q.mean(axis=0)
        sm = np.round(c + (q - c) * float(core)).astype(np.int32)
        region = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(sm).reshape(-1, 2), 1)
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        sel = np.zeros(len(u), dtype=bool)
        sel[inb] = region[v[inb], u[inb]] > 0
        if int(sel.sum()) < TRIM_MIN_POINTS:
            return None
        pts = np.asarray(depth_ctx["xyz"], dtype=np.float64)[idx[sel]]
        h = (pts @ np.asarray(table["n"], dtype=np.float64)
             + float(table["d"])) * 1000.0
        # upper envelope (90th percentile): the top face is the highest
        # surface under its own pixels, so points seen "through" it (clouds
        # that are not z-buffered) or low outliers cannot pull it down
        h = float(np.percentile(h, 90.0))
        return h if np.isfinite(h) else None
    except Exception:  # noqa: BLE001 — a hint only
        return None


def resolve_box_profile(corners_xyz, height: dict | None):
    """(profile, record dict) from the top-face corners and the measured
    height (see :func:`box_profiles.resolve`). ``corners_xyz`` missing ->
    legacy class unknown: ``(None, None)`` (callers keep today's path)."""
    cls = _fold_mod().box_class_from_xyz(corners_xyz) \
        if corners_xyz is not None else None
    if cls is None:
        return None, None
    c = np.asarray(corners_xyz, dtype=np.float64).reshape(4, 3)
    sides = [float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
    a, b = 0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3])
    h = (height or {}).get("measured_height_mm")
    prof, info = _bp.resolve(max(a, b), min(a, b), h)
    flags = []
    if info.get("profile_mismatch"):
        flags.append("profile_mismatch")
    if (not prof.is_generic and h is not None and prof.height_mm is not None
            and abs(float(h) - prof.height_mm) > _bp.HEIGHT_MISMATCH_MM):
        flags.append("height_mismatch")
    rec = {"name": prof.name, "legacy_class": info["legacy_class"],
           "profiles_excluded": info.get("profiles_excluded", []),
           "profiles_excluded_raw": info.get("profiles_excluded_raw", ""),
           "matched": info["matched"], "reason": info.get("reason"),
           "flags": flags,
           "top_dims_mm": [round(max(a, b) * 1000.0, 1),
                           round(min(a, b) * 1000.0, 1)],
           "profile_height_mm": prof.height_mm,
           "geometry_axis_validated": prof.geometry_axis_validated,
           "depth_edge_validated": prof.depth_edge_validated,
           "classification": info.get("classification"),
           **({"near_miss": info["near_miss"]} if "near_miss" in info else {}),
           **(height or {})}
    return prof, rec


def _fold_mod():
    from . import fold_edge as _fold
    return _fold


def choose_top_plane(pts: np.ndarray, plane, depth_ctx: dict
                     ) -> tuple[np.ndarray, float, dict]:
    """Pick the top plane for the trimmed fallback by ORIENTATION.

    ``plane`` is the dominant (guided) plane (None for a candidate that never
    reached the depth check: only a table-parallel plane qualifies then; the
    returned normal is None when there is none); it is kept whenever it is
    already table-parallel at a plausible box height. Otherwise up to
    ``TRIM_MAX_PLANES`` sequential RANSAC planes are fitted to ``pts`` (cloud
    points inside the dilated candidate) and the highest table-parallel one
    is used. No table-parallel plane -> the dominant plane (today's
    behaviour). Returns ``(normal, d, plane_choice)``."""
    n0 = d0 = None
    if plane is not None:
        n0 = np.asarray(plane.normal, dtype=np.float64).ravel()
        d0 = float(plane.d)
    table = estimate_table_plane(depth_ctx)
    h_max = float(depth_ctx.get("box_h_max_mm", BOX_H_MAX_KNOWN_MM))
    if table is None:
        return n0, d0, {"method": "dominant", "table_normal": None,
                        "angle_deg": None, "height_mm": None}
    tn = [round(float(v), 5) for v in table["n"]]
    ref = np.median(pts, axis=0)
    ang0 = h0 = None
    if n0 is not None:
        ang0, h0 = _table_relation(n0, d0, table, ref)
    if n0 is not None and _choice_ok(ang0, h0, h_max):
        return n0, d0, {"method": "table_parallel", "table_normal": tn,
                        "angle_deg": round(ang0, 2), "height_mm": round(h0, 1)}
    best = None
    try:
        import sys

        from .autolabel import depth_top as _depth

        dcfg = dict(_dcfg_of(depth_ctx))
        dcfg["max_planes"] = TRIM_MAX_PLANES
        dcfg["top_min_inliers"] = TRIM_MIN_POINTS
        for pl in _depth.sequential_planes(pts, dcfg):
            ang, h = _table_relation(pl["n"], pl["d"], table, pl["centroid"])
            if _choice_ok(ang, h, h_max) and (best is None or h > best[2]):
                best = (np.asarray(pl["n"], dtype=np.float64),
                        float(pl["d"]), h, ang)
    except Exception:  # noqa: BLE001
        best = None
    if best is not None:
        return best[0], best[1], {
            "method": "table_parallel", "table_normal": tn,
            "angle_deg": round(best[3], 2), "height_mm": round(best[2], 1)}
    return n0, d0, {"method": "dominant", "table_normal": tn,
                    "angle_deg": None if ang0 is None else round(ang0, 2),
                    "height_mm": None if h0 is None else round(h0, 1)}


def plane_trimmed_quad(quad: np.ndarray, plane, depth_ctx: dict | None,
                       W: int, H: int, allow_no_plane: bool = False,
                       dilate: float | None = None
                       ) -> tuple[np.ndarray | None, dict]:
    """Top-face outline from the cloud points ON the fitted top plane.

    Keeps cloud points within ``TRIM_PLANE_BAND_MM`` of ``plane`` whose
    projection lies inside ``quad`` dilated ``TRIM_QUAD_DILATE`` about its
    centroid, rasterises their pixels, closes + keeps the largest connected
    component, and fits the best 4-gon to its convex hull (the SAM3 worker's
    ``_mask_to_quad`` logic). Returns ``(named quad [BL,BR,FR,FL] | None,
    info)``; ``info["reason"]`` explains a None. Never raises on data.
    """
    info: dict = {}
    if not depth_ctx or (plane is None and not allow_no_plane):
        info["reason"] = "no_plane_or_cloud"
        return None, info
    try:
        idx, uv = _depth_projection(depth_ctx)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"projection_failed({exc})"
        return None, info
    if idx is None:
        info["reason"] = "no_cloud"
        return None, info
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    big = np.round(c + (q - c) * (TRIM_QUAD_DILATE if dilate is None
                                  else float(dilate))).astype(np.int32)
    region = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(big).reshape(-1, 2), 1)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    sel = np.zeros(len(u), dtype=bool)
    sel[inb] = region[v[inb], u[inb]] > 0
    if not sel.any():
        info["reason"] = "no_points_in_quad"
        return None, info
    pts = np.asarray(depth_ctx["xyz"], dtype=np.float64)[idx[sel]]
    n, d_plane, choice = choose_top_plane(pts, plane, depth_ctx)
    info["plane_choice"] = choice
    if n is None:
        info["reason"] = "no_table_parallel_plane"
        return None, info
    res_mm = np.abs(pts @ n + d_plane) * 1000.0
    keep = res_mm <= TRIM_PLANE_BAND_MM
    n_keep = int(keep.sum())
    info["n_points"] = n_keep
    if n_keep < TRIM_MIN_POINTS:
        info["reason"] = f"too_few_plane_points({n_keep})"
        return None, info
    mask = np.zeros((H, W), np.uint8)
    mask[v[sel][keep], u[sel][keep]] = 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    # Open removes thin fringes (mixed edge pixels bridging to a neighbour).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (9, 9)))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_cc < 2:
        info["reason"] = "no_component"
        return None, info
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        info["reason"] = "no_contour"
        return None, info
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    hull_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.reshape(-1, 2), 1)
    info["hull_area_px"] = int(hull_mask.sum())
    info["hull_mask"] = hull_mask  # in-memory only (callers pick keys)
    from . import sam3_worker as _w  # light module-level imports only

    quad4, iou, qmethod = _w._mask_to_quad_ex(hull_mask, H, W)
    if quad4 is None:
        info["reason"] = "no_4gon"
        return None, info
    info["hull_quad_iou"] = round(float(iou), 3)
    info["quad_method"] = qmethod
    return np.asarray(_w._to_named(quad4), dtype=np.float64).reshape(4, 2), info


# ----------------------------------------------------------------------
# R2 (high-yaw outline) / R3 (k4 rectangle) helpers
# ----------------------------------------------------------------------

def _rel_cfg(cfg: dict | None) -> dict:
    """The ``reliability`` config section merged over the defaults."""
    base = dict(default_config()["reliability"])
    base.update((cfg or {}).get("reliability") or {})
    return base


def _fold45(angle_deg: float) -> float:
    a = angle_deg % 90.0
    return min(a, 90.0 - a)


def box_view_geometry(quad: np.ndarray, manifest, depth_ctx: dict | None,
                      ref_height_m: float = 0.10) -> dict | None:
    """Box yaw vs the camera and visible side walls from a (coarse) quad.

    The quad corners are back-projected onto the plane parallel to the table
    at ``ref_height_m`` above it (the exact height only scales the result).
    ``yaw45``: edge directions in the table plane vs the camera x axis
    projected on it, folded to [0, 45] deg (0 = edges square to the camera;
    the definition of scripts/eval/live_reliability.py). ``wall_cos[edge]``:
    cosine between the edge's outward (horizontal) wall normal and the view
    direction from the edge midpoint to the camera; > 0 means the vertical
    wall below that edge faces the camera (visible). Edge order [back, right,
    front, left] for a [BL, BR, FR, FL] quad. None when there is no table
    plane / camera / finite back-projection."""
    from . import fold_edge as _fold

    if not depth_ctx:
        return None
    table = estimate_table_plane(depth_ctx)
    cam = _fold._intrinsics(manifest)
    if table is None or cam is None:
        return None
    n = np.asarray(table["n"], dtype=np.float64)
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    p = _fold._backproject(q, cam, n, float(table["d"]) - float(ref_height_m))
    if not np.all(np.isfinite(p)):
        return None
    ex = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(n, ex)
    z = 0j
    for i in range(4):
        v = p[(i + 1) % 4] - p[i]
        a = np.arctan2(float(v @ ey), float(v @ ex))
        z += complex(np.cos(4 * a), np.sin(4 * a))
    yaw = _fold45(float(np.degrees(np.arctan2(z.imag, z.real)) / 4.0))
    centre = p.mean(axis=0)
    cos = {}
    for i, name in enumerate(_fold.EDGE_NAMES):
        mid = 0.5 * (p[i] + p[(i + 1) % 4])
        m = np.cross(n, p[(i + 1) % 4] - p[i])
        nm = float(np.linalg.norm(m))
        if nm < 1e-9:
            return None
        m = m / nm
        if float(m @ (mid - centre)) < 0:
            m = -m
        v = -mid
        cos[name] = float(m @ v) / float(np.linalg.norm(v))
    return {"yaw45": round(float(yaw), 2),
            "wall_cos": {k: round(v, 3) for k, v in cos.items()}}


def view_needs_trim(geom: dict | None, rcfg: dict) -> tuple[bool, list[str]]:
    """R2 trigger: yaw >= ``r2_yaw_deg`` or a SECOND wall (besides the
    nearest-facing one) visible with view cos >= ``r2_wall_cos``."""
    if not geom:
        return False, []
    why = []
    if geom["yaw45"] >= float(rcfg["r2_yaw_deg"]):
        why.append(f"yaw{geom['yaw45']:.0f}")
    cs = sorted(geom["wall_cos"].values(), reverse=True)
    if len(cs) >= 2 and cs[1] >= float(rcfg["r2_wall_cos"]):
        why.append("second_wall_visible")
    return bool(why), why


def _quad_shift_px(a: np.ndarray, b: np.ndarray) -> float:
    """Largest corner distance between two quads, minimised over the cyclic
    start / orientation of ``b`` (names may differ between the two)."""
    a = np.asarray(a, dtype=np.float64).reshape(4, 2)
    b = np.asarray(b, dtype=np.float64).reshape(4, 2)
    best = float("inf")
    for bb in (b, b[::-1]):
        for k in range(4):
            best = min(best, float(np.linalg.norm(a - np.roll(bb, k, axis=0),
                                                  axis=1).max()))
    return best


def _quad_iou_px(a: np.ndarray, b: np.ndarray, W: int, H: int) -> float:
    ma = np.zeros((H, W), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(ma, np.round(np.asarray(a).reshape(4, 2)).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(np.asarray(b).reshape(4, 2)).astype(np.int32), 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / union if union else 0.0


def fit_plane_rectangle(quad: np.ndarray, manifest, depth_ctx: dict | None,
                        W: int, H: int, rcfg: dict
                        ) -> tuple[np.ndarray | None, dict]:
    """R3: orthogonal min-area rectangle of the TOP-PLANE inlier cloud.

    Cloud points inside ``quad`` dilated by ``r3_dilate`` that lie within
    ``TRIM_PLANE_BAND_MM`` of the table-parallel top plane (same plane
    choice as the plane trim; no dominant fallback) are cleaned by the
    plane trim's morphology (close + open + largest component), expressed
    in plane coordinates (mm) and fitted with ``cv2.minAreaRect``. The four
    rectangle corners are projected to pixels with the manifest camera
    (distortion applied) and returned as a named [BL, BR, FR, FL] quad. The
    known box sizes (``r3_known_dims_mm``) are a soft check only (recorded
    in ``info["dims"]``). Returns ``(named quad | None, info)``;
    ``info["reason"]`` explains a None."""
    from . import fold_edge as _fold
    from . import sam3_worker as _w

    info: dict = {}
    if not depth_ctx:
        info["reason"] = "no_depth"
        return None, info
    cam = _fold._intrinsics(manifest)
    if cam is None:
        info["reason"] = "no_intrinsics"
        return None, info
    try:
        idx, uv = _depth_projection(depth_ctx)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"projection_failed({exc})"
        return None, info
    if idx is None:
        info["reason"] = "no_cloud"
        return None, info
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    big = np.round(c + (q - c) * float(rcfg["r3_dilate"])).astype(np.int32)
    region = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(big).reshape(-1, 2), 1)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    sel = np.zeros(len(u), dtype=bool)
    sel[inb] = region[v[inb], u[inb]] > 0
    if not sel.any():
        info["reason"] = "no_points_in_quad"
        return None, info
    pts = np.asarray(depth_ctx["xyz"], dtype=np.float64)[idx[sel]]
    n, d_plane, choice = choose_top_plane(pts, None, depth_ctx)
    info["plane_choice"] = choice
    if n is None or choice.get("method") != "table_parallel":
        info["reason"] = "no_table_parallel_plane"
        return None, info
    n = np.asarray(n, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    n, d_plane = n / nn, float(d_plane) / nn
    band = float(rcfg.get("r3_band_mm") or TRIM_PLANE_BAND_MM)
    keep = np.abs(pts @ n + d_plane) * 1000.0 <= band
    if int(keep.sum()) < TRIM_MIN_POINTS:
        info["reason"] = f"too_few_plane_points({int(keep.sum())})"
        return None, info
    us, vs = u[sel][keep], v[sel][keep]
    mask = np.zeros((H, W), np.uint8)
    mask[vs, us] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_cc < 2:
        info["reason"] = "no_component"
        return None, info
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inl = lab[vs, us] == best
    P = pts[keep][inl]
    if len(P) < TRIM_MIN_POINTS:
        info["reason"] = f"too_few_component_points({len(P)})"
        return None, info
    ex = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(n, ex)
    xy = np.stack([P @ ex, P @ ey], axis=1) * 1000.0
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(xy.astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float64) / 1000.0
    corners3 = (box[:, 0:1] * ex[None, :] + box[:, 1:2] * ey[None, :]
                - d_plane * n[None, :])
    px = _fold._project(corners3, cam)
    info["n_points"] = int(len(P))
    info["rect_mm"] = [round(float(max(rw, rh)), 1), round(float(min(rw, rh)), 1)]
    dims = {}
    for cls, (la, lb) in (rcfg.get("r3_known_dims_mm") or {}).items():
        dims[cls] = round(max(abs(max(rw, rh) - max(la, lb)),
                              abs(min(rw, rh) - min(la, lb))), 1)
    if dims:
        cls = min(dims, key=dims.get)
        info["dims"] = {"nearest_class": cls, "max_dev_mm": dims[cls],
                        "within_tol": bool(dims[cls] <= float(
                            rcfg["r3_dims_tol_mm"]))}
    if (not np.all(np.isfinite(px)) or px[:, 0].min() < -50
            or px[:, 0].max() > W + 50 or px[:, 1].min() < -50
            or px[:, 1].max() > H + 50):
        info["reason"] = "rect_outside_image"
        return None, info
    named = np.asarray(_w._to_named(px), dtype=np.float64).reshape(4, 2)
    iou = _quad_iou_px(named, q, W, H)
    info["iou_with_k4"] = round(float(iou), 3)
    if iou < float(rcfg["r3_min_iou"]):
        info["reason"] = f"rect_far_from_k4(iou {iou:.2f})"
        return None, info
    return named, info


def independent_top_confirm(quad: np.ndarray, depth_ctx: dict | None,
                            W: int, H: int, rcfg: dict) -> dict:
    """R5: SAM3/RGB-independent check of the final top-face quad.

    The cloud points inside ``quad`` dilated by ``r5_dilate`` that lie on the
    TABLE-PARALLEL plane at a plausible box height (``choose_top_plane``
    with no dominant fallback) form a hull; the quad is confirmed when its
    pixel IoU with that hull is >= ``r5_min_iou`` (the box top is a flat,
    table-parallel face of the quad's extent, from depth only). Returns
    ``{"confirmed", "iou", "plane_choice", "reason"?}``. Never raises."""
    out: dict = {"confirmed": False, "iou": None}
    try:
        tq, info = plane_trimmed_quad(quad, None, depth_ctx, W, H,
                                      allow_no_plane=True,
                                      dilate=float(rcfg["r5_dilate"]))
        out["plane_choice"] = info.get("plane_choice")
        if tq is None:
            out["reason"] = str(info.get("reason"))
            return out
        hull = info["hull_mask"]
        qm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(qm, np.round(np.asarray(quad, dtype=np.float64)
                                        .reshape(4, 2)).astype(np.int32), 1)
        union = int((qm | hull).sum())
        iou = int((qm & hull).sum()) / union if union else 0.0
        out["iou"] = round(float(iou), 3)
        out["n_points"] = int(info.get("n_points", 0))
        out["confirmed"] = bool(iou >= float(rcfg["r5_min_iou"]))
        if not out["confirmed"]:
            out["reason"] = f"low_plane_hull_iou({iou:.2f})"
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"error({type(exc).__name__})"
    return out


def noise_adaptive_gate(cfg: dict, depth_est, z_top_m: float | None = None) -> dict:
    """Guided-plane gate thresholds for this scene's depth noise.

    Disabled (``v2.noise_adaptive.enabled`` false): today's fixed values
    (``guided_plane.ransac_threshold_m`` band, ``v2.max_plane_p95_mm``).
    Enabled: ``noise_mm`` = median RMS of the planes ``estimate_top`` fitted
    (``depth.noise_mm``); ``excess = max(0, noise - rms_floor_mm)``; the
    inlier band grows by ``band_k_mm_per_mm * excess`` (capped at
    ``max_band_mm``) and the p95 limit by ``p95_k_mm_per_mm * excess``; both
    are never below today's values, so a scene at or below the floor noise
    (the table cell: 0.7-1.6 mm) keeps exactly the fixed thresholds. No noise
    estimate (depth failed early) -> fixed thresholds. Returns the band (mm),
    p95 limit (mm) and the bulge parameters.

    ``noise_adaptive.range_model`` (dict or None): depth noise grows with the
    range of the BOX, which the support planes (floor, 1-2.5 m) do not
    represent and whose band-truncated RMS saturates near 2.9 mm. With a model
    the scene noise above ``rms_floor_mm`` only ACTIVATES the scaling, and the
    thresholds are ``band_ref_mm`` / ``p95_ref_mm`` x (z_top / z_ref_m) **
    exponent (z_top = mean range of the top corners; z_ref when not yet known),
    never below today's values, band capped at ``max_band_mm``. Constants fitted
    on the two dual-arm captures (1.17 m: ratio 0.87 @ 7 mm, p95 9.0; 1.91 m:
    ratio 0.88 @ 10 mm, p95 12.9 -> exponent 0.7-0.8)."""
    v2cfg, gcfg = cfg["v2"], cfg["guided_plane"]
    na = v2cfg.get("noise_adaptive") or {}
    gate: dict = {"enabled": bool(na.get("enabled")), "noise_mm": None,
                  "excess_mm": 0.0, "scaled": False,
                  "band_mm": float(gcfg["ransac_threshold_m"]) * 1000.0,
                  "p95_limit_mm": float(v2cfg["max_plane_p95_mm"]),
                  "bulge_mm": 0.0}
    if not gate["enabled"]:
        return gate
    noise = ((depth_est or {}).get("depth") or {}).get("noise_mm")
    try:
        noise = float(noise)
    except (TypeError, ValueError):
        noise = float("nan")
    if not np.isfinite(noise):
        gate["reason"] = "no_noise_estimate"
        return gate
    excess = max(0.0, noise - float(na["rms_floor_mm"]))
    gate["noise_mm"] = round(noise, 2)
    gate["excess_mm"] = round(excess, 2)
    rm = na.get("range_model")
    if excess > 0.0 and isinstance(rm, dict):
        zr = float(rm["z_ref_m"])
        f = (float(z_top_m) / zr) ** float(rm["exponent"]) if z_top_m else 1.0
        gate["z_top_m"] = None if not z_top_m else round(float(z_top_m), 3)
        gate["model"] = "range"
        gate["band_mm"] = min(float(na["max_band_mm"]),
                              max(gate["band_mm"], float(rm["band_ref_mm"]) * f))
        gate["p95_limit_mm"] = max(gate["p95_limit_mm"], float(rm["p95_ref_mm"]) * f)
        gate["scaled"] = True
    elif excess > 0.0:
        gate["band_mm"] = min(max(gate["band_mm"], float(na["max_band_mm"])),
                              gate["band_mm"] + float(na["band_k_mm_per_mm"]) * excess)
        gate["p95_limit_mm"] += float(na["p95_k_mm_per_mm"]) * excess
        gate["scaled"] = True
    return gate


def _bulge_allowance_mm(cfg: dict, corners_xyz) -> float:
    """p95 allowance (mm) for the lid bulge of a LARGE top face (a big flap
    lid sags/bulges ~+-2 mm): ``bulge_mm_per_m`` x longest side, capped at
    ``bulge_max_mm``, only for sides >= ``bulge_min_side_m``."""
    na = (cfg["v2"].get("noise_adaptive") or {})
    try:
        c = np.asarray(corners_xyz, dtype=np.float64).reshape(4, 3)
        side = max(float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(side) or side < float(na["bulge_min_side_m"]):
        return 0.0
    return float(min(float(na["bulge_max_mm"]), float(na["bulge_mm_per_m"]) * side))


def check_depth_cross(quad: np.ndarray, manifest: dict, depth_est, depth_err,
                      W: int, H: int, cfg: dict, depth_ctx: dict | None = None
                      ) -> tuple[dict, dict | None]:
    """Run the independent depth check for one refined quad.

    Returns ``(result, guided_plane)``. ``result`` carries quad/iou/plane
    fields, ``wrong_face`` bool and ``reasons`` list. Pure function of its
    inputs (imports resolved lazily so unit tests can call it directly).

    Wrong-face arms (either fires):
    - IoU arm: refined quad vs independent ``estimate_top`` quad below
      ``min_rgb_depth_iou``.
    - Plane-support arm: eroded-interior inlier share below
      ``min_plane_inlier_ratio``, or p95 absolute residual over ALL
      eroded-interior points (not just RANSAC inliers) above
      ``max_plane_p95_mm``. (The old inlier-RMS arm was dead: inlier RMS
      is < 3 mm by construction of the 3 mm RANSAC threshold.)
    """
    import sys

    from .pbd import sam3_depth_fusion as _fusion

    v2cfg = cfg["v2"]
    gate = noise_adaptive_gate(cfg, depth_est)
    reasons: list[str] = []
    out: dict = {"iou": None, "wrong_face": False, "plane": None,
                 "plane_rms_mm": None, "plane_inliers": None,
                 "plane_inlier_ratio": None, "plane_p95_all_mm": None,
                 "corners_xyz": None, "depth_quad": None}
    ambiguous_retry = None
    if (depth_ctx is not None and support_hidden(depth_ctx)
            and depth_est is not None and not depth_est.get("ok")
            and depth_est.get("reason") == "depth_ambiguous_top"):
        # hidden_stand: two parallel planes above the floor (this box top and
        # e.g. a desk) cannot be told apart by depth alone; the candidate's own
        # ROI decides (the floor is passed in as the support plane).
        est_a, roi_a = _depth_top_roi(quad, W, H, cfg, depth_ctx)
        ambiguous_retry = {"used": True, "after": "depth_ambiguous_top",
                           "roi": roi_a,
                           "result": "ok" if (est_a is not None
                                              and est_a.get("ok")) else "failed",
                           "reason": (est_a or {}).get("reason")}
        out["depth_top_roi_retry"] = ambiguous_retry
        if est_a is not None and est_a.get("ok"):
            depth_est = est_a
        else:
            reasons.append("depth_top_ambiguous_unresolved")
    if depth_est is not None and depth_est.get("ok"):
        dquad_tltr = np.asarray(depth_est["quad"], dtype=np.float64).reshape(4, 2)
        dquad = _coarse.depth_quad_to_named(dquad_tltr)
        out["depth_quad"] = dquad
        iou = _quad_iou(quad, dquad, W, H)
        if depth_ctx is not None and _global_quad_elsewhere(quad, dquad, iou):
            # The global top plane sits somewhere else (same-height surface,
            # e.g. a desk): re-estimate inside the candidate's ROI.
            g = np.asarray(dquad, dtype=np.float64).mean(axis=0)
            est_r, roi = _depth_top_roi(quad, W, H, cfg, depth_ctx)
            retry = {"used": True,
                     "global_quad_centroid": [round(float(g[0]), 1),
                                              round(float(g[1]), 1)],
                     "roi": roi}
            iou_r = None
            if est_r is not None and est_r.get("ok"):
                rq = _coarse.depth_quad_to_named(np.asarray(
                    est_r["quad"], dtype=np.float64).reshape(4, 2))
                iou_r = _quad_iou(quad, rq, W, H)
            if iou_r is not None and iou_r > 0.0:
                retry["result"] = "ok"
                dquad, iou = rq, iou_r
                out["depth_quad"] = dquad
            else:
                retry["result"] = "failed" if not (
                    est_r is not None and est_r.get("ok")) else "no_overlap"
                out["depth_quad"] = None
                iou = None
                reasons.append("depth_top_elsewhere")
            out["depth_top_roi_retry"] = retry
        out["iou"] = iou
        if iou is not None and iou < float(v2cfg["min_rgb_depth_iou"]):
            out["wrong_face"] = True
            reasons.append(f"low_rgb_depth_iou({iou:.2f})")
    else:
        reasons.append(str((depth_est or {}).get(
            "reason", depth_err or "depth_unavailable")))
    gcfg = cfg["guided_plane"]

    def _fit(g):
        return _fusion.fit_sam3_guided_depth_plane(
            manifest, quad,
            erosion_px=int(gcfg["erosion_px"]),
            ransac_threshold_m=(float(gcfg["ransac_threshold_m"])
                                if not g["scaled"] else g["band_mm"] / 1000.0),
            min_inliers=int(gcfg["min_inliers"]))
    try:
        gplane = _fit(gate)
        if (gate.get("model") == "range" and gplane is not None
                and gplane.status == "OK" and gplane.plane is not None):
            # second pass: the band for the range the first fit found
            g2 = noise_adaptive_gate(cfg, depth_est, float(np.mean(
                np.asarray(gplane.camera_corners_xyz, dtype=np.float64)[:, 2])))
            if abs(g2["band_mm"] - gate["band_mm"]) > 0.5:
                gate = g2
                gplane = _fit(gate)
            else:
                gate = g2
    except ValueError as exc:
        reasons.append(f"guided_plane_invalid_param({exc})")
        return {**out, "reasons": reasons}, None
    if gplane is not None and gplane.status == "OK" and gplane.plane is not None:
        est = gplane.estimate
        rms = float(getattr(est, "rms_mm", float("nan")))
        inl = int(getattr(est, "n_inliers", 0))
        ratio = float(getattr(est, "inlier_ratio", float("nan")))
        p95_all = _plane_p95_all_mm(gplane, quad, W, H, manifest)
        out["plane"] = gplane.plane
        out["plane_rms_mm"] = rms
        out["plane_inliers"] = inl
        out["plane_inlier_ratio"] = ratio
        out["plane_p95_all_mm"] = p95_all
        out["corners_xyz"] = np.asarray(gplane.camera_corners_xyz,
                                        dtype=np.float64)
        p95_limit = float(v2cfg["max_plane_p95_mm"])
        if gate["enabled"]:
            gate["bulge_mm"] = round(_bulge_allowance_mm(
                cfg, out["corners_xyz"]), 2)
            p95_limit = gate["p95_limit_mm"] + gate["bulge_mm"]
            gate["p95_limit_total_mm"] = round(p95_limit, 2)
            out["plane_gate"] = gate
        if (ratio is not None and np.isfinite(ratio)
                and ratio < float(v2cfg["min_plane_inlier_ratio"])):
            out["wrong_face"] = True
            reasons.append(f"low_plane_inlier_ratio({ratio:.2f})")
        if (p95_all is not None and np.isfinite(p95_all)
                and p95_all > p95_limit):
            out["wrong_face"] = True
            reasons.append(f"high_plane_p95({p95_all:.1f}mm)")
    elif gplane is not None:
        reasons.extend([f"guided_plane_{r}" for r in (gplane.reasons or [])])
    if out["wrong_face"]:
        reasons.append("wrong_face_suspect")
    out["reasons"] = reasons
    return out, gplane


def _plane_p95_all_mm(gplane, quad: np.ndarray, W: int, H: int,
                      manifest: dict | None = None) -> float | None:
    """p95 absolute residual (mm) over ALL eroded-interior PCD points.

    Loads the cloud from ``manifest`` (projection cached per process via
    ``sam3_depth_fusion.projected_depth_winners``), projects into the eroded quad with
    nearest-depth-wins, and scores every interior point against the fitted
    plane. Inlier-only RMS passes by construction; this sees the off-plane
    wedge of a top+side quad. Returns None when unavailable (never raises).
    """
    import sys

    try:
        import cv2 as _cv2
        from .pbd import sam3_depth_fusion as _fusion
        from .pbd.camera_model import CameraModel
        plane = gplane.plane
        n = np.asarray(plane.normal, dtype=np.float64).ravel()
        d = float(plane.d)
        if manifest is None or not isinstance(manifest, dict):
            return None
        info = manifest.get("camera_info") or {}
        cloud = manifest.get("pointcloud") or {}
        pcd_path = cloud.get("path")
        if not pcd_path:
            return None
        camera = CameraModel.from_mapping(
            {"frame_id": "camera_color_optical_frame",
             "width": int(manifest["image"]["width"]),
             "height": int(manifest["image"]["height"]),
             "k": info.get("k"), "d": info.get("d", ()),
             "distortion_model": info.get("distortion_model", "plumb_bob"),
             "p": info.get("p")})
        # Same projection + nearest-depth-wins (depth, then source row) as
        # the guided plane fit; the projection is computed once per process.
        proj = _fusion.projected_cloud(pcd_path, camera, W, H, 0.05)
        if proj.status != "ok":
            return None
        mask = np.zeros((H, W), np.uint8)
        _cv2.fillPoly(mask, [np.round(
            np.asarray(quad, dtype=np.float64)).astype(np.int32)], 1)
        if int(gplane.erosion_px) > 0:
            mask = _cv2.erode(mask, np.ones((3, 3), np.uint8),
                              iterations=int(gplane.erosion_px))
        # Only mask pixels are eligible: resolve winners over its bbox only
        # (identical rows, identical order).
        region = _fusion.mask_region(mask)
        if region is None:
            return None
        zbuf = _fusion.zbuffer_winners(proj, region)
        elig = mask[zbuf.wv, zbuf.wu] > 0
        pts = np.ascontiguousarray(zbuf.xyz_win[elig])
        if pts.shape[0] < 3:
            return None
        return float(np.percentile(np.abs(pts @ n + d) * 1000.0, 95))
    except Exception:  # noqa: BLE001 — diagnostic only, never raises
        return None


def _corner_depth_winners(manifest: dict | None, plane, W: int, H: int,
                          region: tuple | None = None
                          ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Z-buffer-winning pixels + plane residuals (mm) for corner gates.

    Wraps ``precision_box_detector`` projection (never edited): projects
    the cloud with nearest-depth-wins per pixel and scores winners against
    the fitted top plane. Returns (uv_win Nx2 int, res_mm N) or (None,
    None) when unavailable. Never raises. ``region`` (inclusive
    (u0, v0, u1, v1)) limits the winners to those pixels — same winners,
    same order, only the rest dropped (see
    ``corner_checks.depth_query_region``).
    """
    import sys

    try:
        from .pbd import sam3_depth_fusion as _fusion
        from .pbd.camera_model import CameraModel
        if plane is None or manifest is None \
                or not isinstance(manifest, dict):
            return None, None
        n = np.asarray(plane.normal, dtype=np.float64).ravel()
        d = float(plane.d)
        info = manifest.get("camera_info") or {}
        cloud = manifest.get("pointcloud") or {}
        pcd_path = cloud.get("path")
        if not pcd_path:
            return None, None
        camera = CameraModel.from_mapping(
            {"frame_id": "camera_color_optical_frame",
             "width": int(manifest["image"]["width"]),
             "height": int(manifest["image"]["height"]),
             "k": info.get("k"), "d": info.get("d", ()),
             "distortion_model": info.get("distortion_model", "plumb_bob"),
             "p": info.get("p")})
        # read_pcd_xyz drops non-finite rows, so the finite gate is the
        # plain z >= 0.05 gate of the guided fit, and a stable depth argsort
        # equals lexsort((row, depth)): the cached winners are identical.
        zbuf = _fusion.projected_depth_winners(pcd_path, camera, W, H, 0.05,
                                               region=region)
        if zbuf.status != "ok":
            return None, None
        uv_win = np.stack([zbuf.wu, zbuf.wv], axis=1).astype(np.int64)
        res_mm = (zbuf.xyz_win @ n + d) * 1000.0
        return uv_win, np.asarray(res_mm, dtype=np.float64)
    except Exception:  # noqa: BLE001 — diagnostic only, never raises
        return None, None


def apply_corner_fold(bgr: np.ndarray, quad: np.ndarray, manifest,
                      plane, corners_xyz, fold_cfg: dict | None,
                      outward_limit_px: float | None = None,
                      profile=None, measured_height_m: float | None = None
                      ) -> tuple[np.ndarray, object, dict]:
    """Optional fold relocation of a refined quad (``corner_fold`` config).

    Box class from the metric top-face corners (long side > 150 mm = big),
    height from ``fold_cfg["box_height_m"]``; edges relocated by
    :func:`fold_edge.relocate_edges`; new ``corners_xyz`` = pixel rays on the
    SAME plane (no refit). Returns ``(quad, corners_xyz, info)``; on any
    missing input the quad is returned unchanged and ``info["reason"]``
    says why. ``info`` carries per-edge method/evidence and the corners
    before relocation. Never raises on data.

    ``profile`` (box_profiles.BoxProfile, optional): a ``generic`` profile
    uses ``measured_height_m`` as the fold height (no measurement -> the
    fold is skipped, reason ``no_height_generic``); None / a known profile
    keeps the class-height path (bit-identical to before).
    """
    from . import fold_edge as _fold

    fcfg = dict(fold_cfg or {})
    q0 = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    info: dict = {
        "enabled": True, "changed": False, "box_class": None,
        "height_m": None,
        "corners_px_before": {
            name: [round(float(q0[i][0]), 2), round(float(q0[i][1]), 2)]
            for i, name in enumerate(CORNER_ORDER)},
        "per_edge": None}
    cls = (_fold.box_class_from_xyz(corners_xyz)
           if corners_xyz is not None else None)
    if cls is None:
        info["reason"] = "no_metric_corners"
        return quad, corners_xyz, info
    heights = fcfg.get("box_height_m") or _fold.BOX_HEIGHT_M
    height = heights.get(cls)
    if profile is not None and profile.own_constants:
        cls = profile.name
        height = measured_height_m
        info.update({"box_class": cls, "height_m": height})
        if height is None:
            info["reason"] = f"no_height_{cls}"
            return quad, corners_xyz, info
    info.update({"box_class": cls, "height_m": height})
    try:
        res = _fold.relocate_edges(bgr, q0, manifest, plane, height, fcfg,
                                   outward_limit_px=outward_limit_px)
    except Exception as exc:  # noqa: BLE001 — fold never breaks the record
        info["reason"] = f"fold_failed({type(exc).__name__}:{exc})"
        return quad, corners_xyz, info
    info["per_edge"] = res.get("per_edge")
    if not res.get("changed"):
        return quad, corners_xyz, info
    new_quad = np.asarray(res["corners_px"], dtype=np.float64).reshape(4, 2)
    new_xyz = _fold.corners_xyz_on_plane(new_quad, manifest, plane)
    if new_xyz is None:
        info["reason"] = "reprojection_failed_reverted"
        return quad, corners_xyz, info
    info["changed"] = True
    info["max_corner_shift_px"] = round(
        float(np.linalg.norm(new_quad - q0, axis=1).max()), 2)
    return new_quad, new_xyz, info


def fold_reason(info: dict | None) -> str | None:
    """Compact reason string for checks.reasons, or None."""
    if not info or not info.get("changed"):
        return None
    moved = [e for e, r in (info.get("per_edge") or {}).items()
             if r.get("method") not in (None, "keep")]
    return f"corner_fold_moved({','.join(moved)})"


def _share_image_shape(image_path, shape) -> None:
    """Tell the guided plane fit this image already decoded (size only), so
    it skips a second full PNG decode. Best-effort; never raises."""
    import sys

    try:
        from .pbd import sam3_depth_fusion as _fusion
        _fusion.remember_image_shape(image_path, shape)
    except Exception:  # noqa: BLE001 — speed-only hint
        pass


def _refine_once(gray: np.ndarray, coarse: np.ndarray,
                 cfg: dict) -> tuple[object, float | None, float | None]:
    import sys

    from .pbd import subpixel_edge as _edge

    rcfg = cfg["refine"]
    refined = _edge.refine_quad_edges(
        gray, np.asarray(coarse, dtype=np.float64).reshape(4, 2),
        profile_half_width_px=int(rcfg["profile_half_width_px"]),
        sample_spacing_px=float(rcfg["sample_spacing_px"]),
        min_gradient=float(rcfg["min_gradient"]),
        max_offset_px=float(rcfg["max_offset_px"]),
        min_samples=int(rcfg["min_samples"]))
    rms, sup = _edge_terms(refined)
    return refined, rms, sup


# ----------------------------------------------------------------------
# Coarse candidates
# ----------------------------------------------------------------------

def _live_coarse(image_path: str, session, coarse_prov_extra: dict):
    """Query the persistent worker; return (coarse|None, prov, reason|None)."""
    if session is None:
        return None, {"mask_source": "sam3_live"}, \
            "sam3_live_unavailable(no worker session)"
    try:
        resp = session.query(str(image_path))
    except Sam3LiveUnavailable as exc:
        return None, {"mask_source": "sam3_live"}, \
            f"sam3_live_unavailable({exc})"
    if not resp.get("ok", False):
        prov = {"mask_source": "sam3_live",
                "sam3": {"checkpoint_sha256": None,
                         "prompt": resp.get("prompt"),
                         "concept_score": None,
                         "detections": resp.get("detections"),
                         "n_detections": resp.get("n_detections")},
                "model_revision": resp.get("model_revision")}
        return None, prov, \
            f"sam3_live_{resp.get('reason', 'no_quad')}"
    try:
        corners = np.asarray(resp["corners_px"], dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError, KeyError):
        return None, {"mask_source": "sam3_live"}, "sam3_live_bad_payload"
    if not np.all(np.isfinite(corners)):
        return None, {"mask_source": "sam3_live"}, \
            "sam3_live_bad_payload(non_finite)"
    try:
        score = resp.get("concept_score")
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    sha = session.checkpoint_sha256 or coarse_prov_extra.get("checkpoint_sha256")
    prov = {"mask_source": "sam3_live",
            "sam3": {"checkpoint_sha256": sha,
                     "prompt": resp.get("prompt"),
                     "concept_score": score,
                     "box_xyxy": resp.get("box_xyxy"),
                     "n_detections": resp.get("n_detections"),
                     "picked": resp.get("picked"),
                     "detections": resp.get("detections"),
                     "mask_quad_iou": resp.get("mask_quad_iou")},
            "model_revision": resp.get("model_revision")
            or session.model_revision}
    if resp.get("quad_method") == "min_area_k4":
        # Recorded only for the fallback so approx_poly outputs stay
        # byte-identical to the pre-fallback provenance.
        prov["sam3"]["quad_method"] = "min_area_k4"
    alts = resp.get("alternates")
    if isinstance(alts, list) and alts:
        # Private hand-off to _gather_coarse_candidates (never serialized).
        prov["_alternates"] = alts
    return corners, prov, None


def _alternate_candidates(prov: dict, checkpoint_sha) -> list[dict]:
    """Build further sam3_live candidates from the worker's ``alternates``.

    Same ``mask_source`` (sam3_live) as the primary so the live detector
    accepts an alternate winner; rank (1-based) and score are recorded.
    Malformed alternates are skipped silently (the primary is unaffected).
    """
    out: list[dict] = []
    base = prov.get("sam3") or {}
    for rank, alt in enumerate(prov.get("_alternates") or [], start=1):
        try:
            corners = np.asarray(alt["corners_px"],
                                 dtype=np.float64).reshape(4, 2)
        except (TypeError, ValueError, KeyError):
            continue
        if not np.all(np.isfinite(corners)):
            continue
        try:
            score = alt.get("concept_score")
            score = float(score) if score is not None else None
        except (TypeError, ValueError, AttributeError):
            score = None
        aprov = {"mask_source": "sam3_live",
                 "sam3": {"checkpoint_sha256": base.get("checkpoint_sha256"),
                          "prompt": base.get("prompt"),
                          "concept_score": score,
                          "box_xyxy": alt.get("box_xyxy"),
                          "n_detections": base.get("n_detections"),
                          "picked": alt.get("det_index"),
                          "detections": base.get("detections"),
                          "mask_quad_iou": alt.get("mask_quad_iou")},
                 "model_revision": prov.get("model_revision"),
                 "alternate_rank": rank}
        if alt.get("quad_method") == "min_area_k4":
            aprov["sam3"]["quad_method"] = "min_area_k4"
        out.append({"source": "sam3_live", "coarse": corners, "prov": aprov,
                    "reason": None, "concept_score": score,
                    "prompt": base.get("prompt"),
                    "checkpoint_sha": base.get("checkpoint_sha256"),
                    "alternate_rank": rank})
    return out


def _drop_candidates_outside_work_area(cands: list, cand_filter) -> None:
    """Scene work area: a SAM3 candidate (live / archived) whose quad lies mostly outside the
    work-area prism in the image is ignored before ranking (``coarse`` None + a reason, recorded in
    the attempts like any other unusable candidate). In place."""
    from .work_area import CANDIDATE_DROP_REASON  # noqa: PLC0415

    for cand in cands:
        if (cand.get("source") not in ("sam3_live", "sam3_archived") or cand.get("skipped")
                or cand.get("coarse") is None):
            continue
        keep, info = cand_filter(cand["coarse"])
        cand.setdefault("prov", {})["work_area"] = info
        if not keep:
            cand["reason"] = (f"{CANDIDATE_DROP_REASON}(inside={info['inside_frac']:.2f}"
                              f"<{info['min_inside_frac']:.2f})")
            cand["coarse"] = None


def _gather_coarse_candidates(capture_id: str, want: str, image_path: str,
                              session, archive_dir, depth_est, depth_err,
                              cfg: dict, depth_join=None,
                              cand_filter=None) -> list[dict]:
    """Return ordered coarse candidates for the fallback chain.

    Each entry: {"source", "coarse"|None, "prov", "reason"|None,
    "concept_score", "prompt", "checkpoint_sha"}. Order for auto:
    sam3_live, sam3_archived (skipped when quad-equal to live), depth_only.
    """
    cands: list[dict] = []
    live_quad = None
    if want in ("auto", "sam3_live"):
        extra = {"checkpoint_sha256":
                 cfg.get("sam3", {}).get("checkpoint_sha256")}
        corners, prov, reason = _live_coarse(image_path, session, extra)
        live_quad = corners
        cands.append({"source": "sam3_live", "coarse": corners, "prov": prov,
                      "reason": reason,
                      "concept_score": (prov.get("sam3") or {}).get(
                          "concept_score"),
                      "prompt": (prov.get("sam3") or {}).get("prompt"),
                      "checkpoint_sha": (prov.get("sam3") or {}).get(
                          "checkpoint_sha256")})
        if corners is not None:
            cands.extend(_alternate_candidates(prov, None))
    if want in ("auto", "sam3_archived"):
        corners, prov = _coarse.load_sam3_archived(capture_id, archive_dir)
        sam3info = prov.get("sam3") or {}
        if corners is not None and live_quad is not None \
                and _quads_same(corners, live_quad):
            cands.append({"source": "sam3_archived", "coarse": None,
                          "prov": prov, "reason": "sam3_archived_same_as_live",
                          "concept_score": sam3info.get("concept_score"),
                          "prompt": sam3info.get("prompt"),
                          "checkpoint_sha": sam3info.get("checkpoint_sha256"),
                          "skipped": True})
        else:
            cands.append({
                "source": "sam3_archived", "coarse": corners, "prov": prov,
                "reason": None if corners is not None
                else prov.get("reason", "sam3_archived_missing"),
                "concept_score": sam3info.get("concept_score"),
                "prompt": sam3info.get("prompt"),
                "checkpoint_sha": sam3info.get("checkpoint_sha256")})
    if cand_filter is not None:
        _drop_candidates_outside_work_area(cands, cand_filter)
    if want in ("auto", "depth_only"):
        if depth_join is not None:
            depth_est, depth_err = depth_join()
        if depth_err is not None:
            cands.append({"source": "depth_only", "coarse": None,
                          "prov": {"mask_source": "depth_only"},
                          "reason": str(depth_err),
                          "concept_score": None, "prompt": None,
                          "checkpoint_sha": None})
        elif depth_est is not None and depth_est.get("ok"):
            cands.append({"source": "depth_only",
                          "coarse": _coarse.depth_quad_to_named(
                              np.asarray(depth_est["quad"], dtype=np.float64)),
                          "prov": {"mask_source": "depth_only",
                                   "depth_status": depth_est.get(
                                       "depth", {}).get("status")},
                          "reason": None, "concept_score": None,
                          "prompt": None, "checkpoint_sha": None})
        else:
            reason = (depth_est or {}).get("reason", "depth_no_quad") \
                if depth_est else (depth_err or "depth_no_quad")
            cands.append({"source": "depth_only", "coarse": None,
                          "prov": {"mask_source": "depth_only"},
                          "reason": str(reason), "concept_score": None,
                          "prompt": None, "checkpoint_sha": None})
    if want not in ("auto", "sam3_live", "sam3_archived", "depth_only"):
        cands.append({"source": want, "coarse": None,
                      "prov": {"mask_source": want},
                      "reason": f"unknown source {want!r}",
                      "concept_score": None, "prompt": None,
                      "checkpoint_sha": None})
    return cands


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

def _s1_off() -> bool:
    return os.environ.get("AUTOLABEL_V2_S1_OFF", "").strip() not in ("", "0")


def _run_depth_work(_depth, read_pcd_xyz, paths: dict, cam, H: int, W: int,
                    cfg: dict, s1: bool, want_proj: bool) -> dict:
    """SAM3-independent depth work of ``process_capture`` (thread-safe: all
    RNGs are per-call ``default_rng(seed)``, no shared mutable state).

    Returns ``{"xyz", "est", "err", "extra"}``; ``extra`` holds precomputed
    ``depth_ctx`` caches (``_table``, ``_proj``) that the lazy accessors would
    otherwise compute with identical results. With ``s1`` False this is the
    old code verbatim (no casts/caches).
    """
    depth_est, depth_err, extra = None, None, {}
    try:
        xyz, _ = read_pcd_xyz(paths["pcd"])
    except Exception as exc:  # noqa: BLE001 — recorded, not fatal
        return {"xyz": None, "est": None,
                "err": f"depth_pcd_unreadable({exc})", "extra": extra}
    if s1:
        # One float64 conversion of the whole cloud; every consumer casts to
        # float64 anyway (identical values, no further copies). A failure is
        # recorded exactly like a failing estimate_top (as before S1).
        try:
            xyz = np.asarray(xyz, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — recorded, not fatal
            return {"xyz": xyz, "est": None,
                    "err": f"depth_estimate_failed({exc})", "extra": extra}
    first: dict = {}
    try:
        kw = {"first_plane_out": first} if s1 else {}
        depth_est = _depth.estimate_top(
            np.asarray(xyz, dtype=np.float64), cam, (H, W),
            cfg["depth_top"], **kw)
    except Exception as exc:  # noqa: BLE001 — recorded, not fatal
        depth_err = f"depth_estimate_failed({exc})"
        depth_est = None
    if s1 and cam is not None:
        try:
            dcfg = cfg["depth_top"]
            # The table plane is the first plane of sequential_planes on the
            # same workspace-cropped cloud with the same seed; the only
            # differences are the min-inlier stop rules, so reuse it only
            # when both rules would pass exactly as for the full fit.
            tmin = int(dcfg["table_min_inliers"])
            if (first and first["n_pool"] >= tmin
                    and first["n_inliers"] >= tmin
                    and dcfg.get("support", "visible_table") != "hidden_stand"):
                n, d = first["n"], first["d"]
                if d < 0:
                    n, d = -n, -d
                extra["_table"] = {"n": n, "d": float(d),
                                   "n_inliers": int(first["n_inliers"])}
                if first.get("rms") is not None:
                    extra["_table"]["rms_mm"] = float(first["rms"]) * 1000.0
            if want_proj and xyz is not None:
                from .autolabel.pcd import project_points
                ok = np.isfinite(xyz[:, 2]) & (xyz[:, 2] > 0.05)
                uv, _z, _ = project_points(xyz, cam)
                extra["_proj"] = (np.nonzero(ok)[0], uv)
        except Exception:  # noqa: BLE001 — caches are optional (lazy path)
            extra.pop("_proj", None)
    return {"xyz": xyz, "est": depth_est, "err": depth_err, "extra": extra}


def _clone_depth_work(res: dict) -> dict:
    """Private copy of a :func:`_run_depth_work` result: the mutable small
    parts (``est``, the ``_table`` cache) are deep-copied, the big arrays
    (``xyz``, ``_proj``) are shared because every consumer only reads them."""
    extra = {k: (copy.deepcopy(v) if k == "_table" else v)
             for k, v in (res.get("extra") or {}).items()}
    return {"xyz": res["xyz"], "est": copy.deepcopy(res["est"]),
            "err": res["err"], "extra": extra}


def process_capture(capture_id: str, paths: dict, cfg: dict | None = None,
                    source: str = "auto", archive_dir=None,
                    session=None, _forced: dict | None = None
                    ) -> tuple[dict, dict]:
    """Run the full v2 pipeline for one capture.

    ``paths``: {"image", "pcd", "manifest"}. ``source``: auto |
    sam3_archived | depth_only | sam3_live (auto = live -> archived (when
    different) -> depth_only). ``session``: persistent
    :class:`sam3_session.Sam3Worker` or None (live attempts fail clean).
    Returns ``(record, debug)`` where debug holds intermediate arrays/dicts
    for the overlay writer (not serialized). Never raises on data: failures
    become records with null geometry and explicit reasons.

    ``_forced`` (private, truncation recovery): ``{"winner", "reasons",
    "attempts"}`` of an already accepted candidate; the candidate search is
    skipped and the downstream stages (profile, seam, record) run on it.
    The same dict may carry ``"depth"`` (a pristine :func:`_run_depth_work`
    result of the first pass, same capture/cfg/camera: reused instead of
    recomputing) and ``"bgr"`` (the first pass's decoded image: used instead
    of decoding the file again).
    """
    import sys

    cfg = cfg or default_config()
    from .autolabel import depth_top as _depth
    from .autolabel.pcd import read_pcd_xyz

    reasons: list[str] = []
    attempts: list[dict] = []
    debug: dict = {"capture_id": capture_id}
    v2cfg = cfg["v2"]

    # ---- inputs ------------------------------------------------------
    try:
        manifest = _load_manifest(paths["manifest"])
    except (OSError, ValueError) as exc:
        return _fail_record(capture_id, paths, cfg, "unreadable_manifest",
                            [f"unreadable_manifest({exc})"], source,
                            attempts=[]), debug
    img = manifest.get("image") or {}
    w, h = int(img.get("width", 0)), int(img.get("height", 0))
    image_path = img.get("path") or paths.get("image")
    bgr = (_forced.get("bgr") if _forced is not None else None)
    if bgr is None:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
    if bgr is None:
        return _fail_record(capture_id, paths, cfg, "unreadable_image",
                            ["unreadable_image"], source,
                            image_size=[w, h] if w and h else None,
                            attempts=[]), debug
    H, W = bgr.shape[:2]
    debug["bgr"] = bgr
    _share_image_shape(image_path, bgr.shape)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Depth estimate runs once (independent check + depth_only candidate).
    # S1: the SAM3-independent depth work (cloud parse, float64 cast,
    # estimate_top, table plane, cloud projection) runs on a thread while the
    # blocking SAM3 IPC query is in flight; results are joined before any
    # consumer reads them, so outputs are unchanged. AUTOLABEL_V2_S1_OFF=1
    # restores the old sequential path exactly.
    cam, depth_est, depth_err = None, None, None
    xyz = None
    try:
        cam, _cal_src = _depth.load_capture_camera(paths["manifest"])
    except (ValueError, OSError) as exc:
        depth_err = f"depth_bad_calibration({exc})"
    s1 = not _s1_off()
    box: dict = {"res": {"xyz": None, "est": None, "err": depth_err,
                         "extra": {}}}

    def _depth_work() -> None:
        try:
            box["res"] = _run_depth_work(
                _depth, read_pcd_xyz, paths, cam, H, W, cfg, s1,
                want_proj=(s1 and session is not None
                           and source in ("auto", "sam3_live")))
        except BaseException as exc:  # noqa: BLE001 — re-raised at join
            box["exc"] = exc

    thread = None
    reuse = _forced.get("depth") if _forced is not None else None
    if depth_err is None:
        if reuse is not None:
            # truncation re-run: the SAM3-independent depth work of this very
            # capture was already done by the first pass (deterministic)
            box["res"] = _clone_depth_work(reuse)
        elif s1:
            thread = threading.Thread(target=_depth_work, daemon=True,
                                      name="s1-depth")
            thread.start()
        else:
            _depth_work()

    def _depth_join():
        """(depth_est, depth_err) once the depth work is finished."""
        if thread is not None:
            thread.join()
        if "exc" in box:
            raise box["exc"]
        return box["res"]["est"], box["res"]["err"]

    # ---- 1. coarse candidates + fallback chain ------------------------
    cand_filter = None
    wa_runtime = cfg.get("work_area_runtime")
    if wa_runtime and cam is not None:
        from .work_area import CandidateFilter  # noqa: PLC0415
        cand_filter = CandidateFilter(wa_runtime, cam, W, H)
    try:
        if _forced is None:
            cands = _gather_coarse_candidates(
                capture_id, source, str(image_path), session, archive_dir,
                depth_est, depth_err, cfg, depth_join=_depth_join,
                cand_filter=cand_filter)
        else:
            cands = []
    finally:
        if thread is not None:
            thread.join()
    depth_est, depth_err = _depth_join()
    xyz = box["res"]["xyz"]
    # Pristine copy for the truncation re-run (taken before any consumer runs).
    depth_snapshot = None
    if (_forced is None and cam is not None and xyz is not None
            and _truncation_mode(cfg) != "off"):
        depth_snapshot = _clone_depth_work(box["res"])
    debug["depth_est"] = depth_est
    debug["cam"] = cam
    depth_ctx = ({"xyz": xyz, "cam": cam, "dcfg": cfg["depth_top"]}
                 if (xyz is not None and cam is not None) else None)
    if depth_ctx is not None:
        depth_ctx.update(box["res"]["extra"])
        _apply_scene_limits(depth_ctx)
    winner = None
    if _forced is not None:
        winner = _forced["winner"]
        reasons[:] = list(_forced["reasons"])
        attempts[:] = list(_forced["attempts"])

    rcfg = _rel_cfg(cfg)
    # Box-size awareness (multi-box plan M3). A candidate whose own central
    # cloud height is far from every known profile height is ``generic`` for
    # the pre-refine rewrite stage (R2 spill height, no r3 known-dims check,
    # depth-trimmed quad preferred); known boxes are unchanged.
    # The depth estimate alone is not trusted (42 of today's 421 captures
    # have a wrong early height ~60 mm): it only decides whether the
    # candidate's own central-quad height is worth computing.
    early_h = early_height_mm(depth_est)
    early_hint = bool(early_h is None or _bp.is_generic_height(early_h))

    def _cand_generic(coarse):
        """(generic?, height_mm | None) of a coarse quad: the median height of
        its central cloud points far from every known profile height."""
        if not early_hint:
            return False, None
        hq = quad_height_mm(coarse, depth_ctx, W, H)
        return bool(hq is not None and _bp.is_generic_height(hq)), hq

    def _pre_refine_rewrite(cand: dict):
        """R3 (k4 mask -> orthogonal top-plane rectangle) / R2 (yawed box or
        visible side wall -> table-parallel plane-trimmed quad) rewrite of a
        sam3_live coarse quad BEFORE edge refinement and the corner fold.
        Returns ``(rewritten candidate | None, info)``; ``info`` is {} when
        the old path applies unchanged (so such captures are identical)."""
        if (cand.get("source") != "sam3_live" or depth_ctx is None
                or cand.get("coarse") is None):
            return None, {}
        prov = cand.get("prov") or {}
        if prov.get("plane_trimmed") or prov.get("r2_trim") \
                or prov.get("r3_rect"):
            return None, {}
        coarse = np.asarray(cand["coarse"], dtype=np.float64).reshape(4, 2)
        info: dict = {}
        newq, key, detail = None, None, None
        generic_c, hq = _cand_generic(coarse)
        r2_ref_h = (hq / 1000.0 if generic_c
                    else float(rcfg["r2_ref_height_m"]))
        rcfg_r3 = ({**rcfg, "r3_known_dims_mm": {}} if generic_c else rcfg)
        is_k4 = (prov.get("sam3") or {}).get("quad_method") == "min_area_k4"
        _scope = _h_max_scope(depth_ctx, generic_c)
        _scope.__enter__()          # 700 mm plane limit only for generic
        try:
            if is_k4 and rcfg["r3_enabled"]:
                rq, rinfo = fit_plane_rectangle(coarse, manifest, depth_ctx,
                                                W, H, rcfg_r3)
                if rq is not None:
                    newq, key, detail = rq, "r3_rect", {
                        "kind": "k4_plane_rectangle", **rinfo}
                else:
                    info["r3_skipped"] = rinfo.get("reason")
            if newq is None and rcfg["r2_enabled"]:
                geom = box_view_geometry(coarse, manifest, depth_ctx, r2_ref_h)
                trig, why = view_needs_trim(geom, rcfg)
                if generic_c and geom is not None and not trig:
                    # tall / unknown box: side faces are large, so the SAM3
                    # outline may spill onto them even for a square view.
                    trig, why = True, ["generic_box"]
                if trig:
                    # Orthogonal rectangle of the top-plane inliers first (a
                    # hull 4-gon rounds off acute corners); the plane-trim
                    # hull quad only when no rectangle can be fitted.
                    tq, tinfo = fit_plane_rectangle(coarse, manifest, depth_ctx,
                                                    W, H, rcfg_r3)
                    method = "plane_rectangle"
                    if tq is None:
                        rect_reason = tinfo.get("reason")
                        tq, tinfo = plane_trimmed_quad(
                            coarse, None, depth_ctx, W, H, allow_no_plane=True)
                        method = "plane_hull_quad"
                        if tq is None:
                            info["r2_skipped"] = (f"trim_{tinfo.get('reason')}"
                                                  f"(rect:{rect_reason})")
                    if tq is not None:
                        shift = _quad_shift_px(coarse, tq)
                        if shift < float(rcfg["r2_min_spill_px"]):
                            info["r2_skipped"] = f"no_spill({shift:.1f}px)"
                        else:
                            newq, key, detail = tq, "r2_trim", {
                                "why": why, "method": method,
                                "geometry": geom,
                                "max_corner_shift_px": round(shift, 1),
                                "n_points": int(tinfo.get("n_points", 0)),
                                "plane_choice": tinfo.get("plane_choice")}
                            if tinfo.get("rect_mm"):
                                detail["rect_mm"] = tinfo["rect_mm"]
                                detail["dims"] = tinfo.get("dims")
        except Exception as exc:  # noqa: BLE001 — R2/R3 never break a capture
            return None, {"r_error": f"{type(exc).__name__}: {exc}"}
        finally:
            _scope.__exit__(None, None, None)
        if newq is None:
            return None, info
        tprov = dict(prov)
        tprov[key] = detail
        tcand = {**{k: v for k, v in cand.items()
                    if not k.startswith("_") and k != "skipped"},
                 "coarse": np.asarray(newq, dtype=np.float64).reshape(4, 2),
                 "prov": tprov}
        info["rewrite"] = key
        info["generic"] = bool(generic_c)
        info["rewrite_detail"] = detail
        info["original_coarse"] = coarse.tolist()
        return tcand, info

    def _evaluate(cand: dict, entry: dict):
        """R2/R3 pre-refine rewrite, then :func:`_evaluate_core`.

        A rewritten quad (top-plane rectangle / trim) is only used when it is
        measurably better than the SAM3 outline: both go through refine + the
        depth check and the rewritten one wins only if the original is not
        already rectangle-consistent (<= ``r_keep_original_rect_mm``, the old
        path, identical output) and the rewrite has strictly smaller 3-D
        opposite-side differences. If the rewrite is rejected or not better,
        the original quad is used (nothing the old path accepted is lost
        because of a bad rewrite)."""
        tcand, rinfo = _pre_refine_rewrite(cand)
        if tcand is None:
            if rinfo:
                entry.update({k: v for k, v in rinfo.items()
                              if k.endswith("_skipped") or k == "r_error"})
            return _evaluate_core(cand, entry)
        key = rinfo["rewrite"]
        n_reasons = len(reasons)
        tentry = dict(entry)
        tentry[key] = True
        tentry["original_coarse"] = rinfo["original_coarse"]
        win_t = _evaluate_core(tcand, tentry)
        if win_t is None:
            del reasons[n_reasons:]
            reasons.append(f"{key}_unusable_original_used")
            entry["rewrite_failed"] = key
            return _evaluate_core(cand, entry)
        n_after_t = len(reasons)
        win_o = _evaluate_core(cand, entry)
        del reasons[n_after_t:]  # the loser's rejection notes are not kept
        d_t = _rect_diffs_mm(win_t["cross"].get("corners_xyz"))
        d_o = (_rect_diffs_mm(win_o["cross"].get("corners_xyz"))
               if win_o is not None else None)
        keep_mm = float(rcfg["r_keep_original_rect_mm"])
        use_t = (win_o is None or d_o is None
                 or (max(d_o) > keep_mm and d_t is not None
                     and max(d_t) < max(d_o)))
        if rinfo.get("generic") and not use_t and d_t is not None:
            # generic box: the depth-trimmed table-parallel quad is the
            # primary source whenever it is at least as rectangle-consistent
            # (the rewrite exists only when the SAM3 quad spills >= r2_min_spill_px)
            use_t = max(d_t) <= max(d_o)
        fmt = (lambda d: "n/a" if d is None else f"{max(d):.1f}")
        if use_t:
            tentry["selected"] = True
            detail = rinfo["rewrite_detail"]
            detail["rect_mm_max_rewrite_vs_original"] = [
                None if d_t is None else round(max(d_t), 2),
                None if d_o is None else round(max(d_o), 2)]
            reasons.append(
                f"{key}_used({','.join(detail.get('why', [detail.get('kind', '')]))}"
                f";rect_mm {fmt(d_t)} vs original {fmt(d_o)})")
            return win_t
        entry["selected"] = True
        tentry["not_selected"] = True
        reasons.append(f"{key}_not_better_original_used("
                       f"rect_mm {fmt(d_t)} vs original {fmt(d_o)})")
        return win_o

    def _evaluate_core(cand: dict, entry: dict):
        """refine + depth cross-check one coarse candidate. Appends the
        attempt entry; returns the winner dict (accepted) or None."""
        entry["coarse"] = np.asarray(
            cand["coarse"], dtype=np.float64).reshape(4, 2).tolist()
        try:
            refined, edge_rms, edge_support = _refine_once(
                gray, np.asarray(cand["coarse"], dtype=np.float64), cfg)
        except ValueError as exc:
            entry.update({"status": "refine_rejected",
                          "reason": f"refine_invalid_input({exc})"})
            attempts.append(entry)
            reasons.append(f"{cand['source']}:refine_invalid_input({exc})")
            return None
        entry["edge_rms_px"] = edge_rms
        entry["edge_support"] = edge_support
        if not refined.accepted:
            entry.update({"status": "refine_rejected",
                          "reason": "; ".join(list(refined.reasons))})
            attempts.append(entry)
            reasons.append(
                f"{cand['source']}:refine_rejected("
                f"{'; '.join(list(refined.reasons))})")
            cand["_refine_rejected"] = True  # private: trim fallback pass 2
            return None
        quad = np.asarray(refined.corners_px, dtype=np.float64).reshape(4, 2)
        cross, _gplane = check_depth_cross(quad, manifest, depth_est,
                                           depth_err, W, H, cfg,
                                           depth_ctx=depth_ctx)
        entry["iou"] = cross.get("iou")
        if cross.get("depth_top_roi_retry") is not None:
            entry["depth_top_roi_retry"] = cross["depth_top_roi_retry"]
        entry["plane_inlier_ratio"] = cross.get("plane_inlier_ratio")
        entry["plane_p95_all_mm"] = cross.get("plane_p95_all_mm")
        if cross.get("plane_gate") is not None:
            entry["plane_gate"] = cross["plane_gate"]
        entry["wrong_face"] = bool(cross.get("wrong_face"))
        if cross.get("wrong_face"):
            entry.update({"status": "depth_cross_rejected",
                          "reason": "; ".join(cross.get("reasons", []))})
            attempts.append(entry)
            reasons.append(
                f"{cand['source']}:depth_cross_rejected("
                f"{'; '.join(cross.get('reasons', []))})")
            # Kept (private) so the plane-trimmed fallback can reuse it.
            cand["_rejected_eval"] = {"quad": quad, "cross": cross}
            return None
        entry.update({"status": "accepted"})
        attempts.append(entry)
        return {"cand": cand, "refined": refined, "quad": quad,
                "edge_rms": edge_rms, "edge_support": edge_support,
                "cross": cross}

    def _trimmed_attempt(src: dict, quad_src: np.ndarray, plane, why: str,
                         orig_diffs=None):
        """Build + evaluate a plane_trimmed candidate from ``src``'s quad.
        Returns the winner dict (accepted, and for ``gross_rect`` strictly
        less non-rectangular than ``orig_diffs``) or None."""
        rank = src.get("alternate_rank")
        frm = "primary" if rank is None else f"alternate_{int(rank)}"
        entry: dict = {"source": "sam3_live", "plane_trimmed": True,
                       "plane_trimmed_from": frm,
                       "plane_trimmed_why": why}
        if rank is not None:
            entry["alternate_rank"] = int(rank)
            entry["concept_score"] = src.get("concept_score")
        with _h_max_scope(depth_ctx, _tall_quad(quad_src, depth_ctx, W, H)):
            tq, tinfo = plane_trimmed_quad(quad_src, plane, depth_ctx, W, H,
                                           allow_no_plane=plane is None)
        if tq is None:
            entry.update({"status": "no_coarse",
                          "reason": f"plane_trimmed_{tinfo.get('reason')}"})
            attempts.append(entry)
            reasons.append(f"plane_trimmed({frm}):{tinfo.get('reason')}")
            return None
        pt = {"from": frm, "why": why,
              "n_points": int(tinfo.get("n_points", 0)),
              "hull_area_px": int(tinfo.get("hull_area_px", 0))}
        sprov = src.get("prov") or {}
        tprov = dict(sprov)
        tprov["plane_trimmed"] = pt
        if tinfo.get("plane_choice") is not None:
            pt["plane_choice"] = tinfo["plane_choice"]
        if tinfo.get("quad_method") == "min_area_k4":
            pt["quad_method"] = "min_area_k4"
        tcand = {**{k: v for k, v in src.items()
                    if not k.startswith("_") and k != "skipped"},
                 "coarse": tq, "prov": tprov}
        win = _evaluate(tcand, entry)
        if win is None:
            return None
        diffs = _rect_diffs_mm(win["cross"].get("corners_xyz"))
        pt["rect_diffs_mm"] = ([round(d, 2) for d in diffs]
                               if diffs is not None else None)
        if why == "gross_rect":
            pt["original_rect_diffs_mm"] = [round(d, 2) for d in orig_diffs]
            if diffs is None or max(diffs) >= max(orig_diffs):
                entry["status"] = "plane_trimmed_not_better"
                reasons.append(f"plane_trimmed({frm}):not_better")
                return None
            reasons.append(f"plane_trimmed_used({frm},gross_rect)")
        else:
            reasons.append(f"plane_trimmed_used({frm},{why})")
        return win

    live_evals: list[dict] = []  # sam3_live candidates evaluated, in order
    trimmed_done = False

    def _trim_fallback_all_rejected():
        """(a) every sam3_live candidate was rejected by the depth check."""
        for src in live_evals:
            ev = src.get("_rejected_eval")
            if ev is None or ev["cross"].get("plane") is None:
                continue
            win = _trimmed_attempt(src, ev["quad"], ev["cross"]["plane"],
                                   "all_rejected")
            if win is not None:
                return win
        # Pass 2 (only when pass 1 found nothing, so earlier outcomes are
        # unchanged): a live candidate whose edge refine failed never got a
        # depth plane; its coarse quad (e.g. top + overexposed front) is
        # trimmed with a TABLE-PARALLEL plane only (no dominant fallback).
        for src in live_evals:
            if not src.get("_refine_rejected") or src.get("coarse") is None:
                continue
            win = _trimmed_attempt(
                src, np.asarray(src["coarse"], dtype=np.float64).reshape(4, 2),
                None, "refine_rejected")
            if win is not None:
                return win
        return None

    for cand in cands:
        if (winner is None and not trimmed_done and live_evals
                and cand["source"] != "sam3_live"):
            trimmed_done = True
            winner = _trim_fallback_all_rejected()
            if winner is not None:
                break
        entry: dict = {"source": cand["source"]}
        if cand.get("alternate_rank") is not None:
            entry["alternate_rank"] = int(cand["alternate_rank"])
            entry["concept_score"] = cand.get("concept_score")
        if cand.get("skipped"):
            entry.update({"status": "skipped",
                          "reason": cand.get("reason")})
            attempts.append(entry)
            reasons.append(f"{cand['source']}:{cand.get('reason')}")
            continue
        if cand.get("coarse") is None:
            entry.update({"status": "no_coarse",
                          "reason": cand.get("reason")})
            attempts.append(entry)
            reasons.append(f"{cand['source']}:{cand.get('reason')}")
            continue
        if cand["source"] == "sam3_live":
            live_evals.append(cand)
        winner = _evaluate(cand, entry)
        if winner is not None:
            break
    if winner is None and not trimmed_done and live_evals:
        winner = _trim_fallback_all_rejected()
    elif winner is not None and _forced is None \
            and winner["cand"]["source"] == "sam3_live" \
            and not any((winner["cand"].get("prov") or {}).get(k)
                        for k in ("plane_trimmed", "r2_trim", "r3_rect")):
        # (b) accepted but grossly non-rectangular in 3-D: try the trim.
        diffs = _rect_diffs_mm(winner["cross"].get("corners_xyz"))
        if (diffs is not None and max(diffs) > GROSS_RECT_MM
                and winner["cross"].get("plane") is not None):
            better = _trimmed_attempt(
                winner["cand"], winner["quad"], winner["cross"]["plane"],
                "gross_rect", orig_diffs=diffs)
            if better is not None:
                winner = better
    if winner is None:
        reasons.append("no_coarse_quad")
        first = cands[0] if cands else {}
        return _fail_record(
            capture_id, paths, cfg, "no_coarse_quad", reasons,
            str(first.get("source", source)), image_size=[W, H],
            concept_score=first.get("concept_score"),
            prompt=first.get("prompt"),
            checkpoint_sha=first.get("checkpoint_sha"),
            model_revision=(first.get("prov") or {}).get("model_revision"),
            attempts=attempts), debug

    # ---- truncation recovery (scene flag; off = this block never runs) ------
    trunc_block = None
    trunc_mode = _truncation_mode(cfg)
    if (_forced is None and trunc_mode != "off" and depth_ctx is not None
            and winner["cand"]["source"] == "sam3_live"
            and winner["cross"].get("plane") is not None):
        try:
            recovered, trunc_block = _truncation_recovery(
                winner, cands, trunc_mode, cfg, depth_ctx, manifest, W, H,
                attempts, reasons, _evaluate,
                lambda w: process_capture(
                    capture_id, paths, cfg, source, archive_dir, session,
                    _forced={"winner": w, "reasons": list(reasons),
                             "attempts": list(attempts),
                             "depth": depth_snapshot,
                             # fresh copy: the re-run may draw on / alter it
                             "bgr": bgr.copy()}))
        except Exception as exc:  # noqa: BLE001 — recovery never breaks the old path
            recovered = None
            trunc_block = {"mode": trunc_mode, "triggered": True, "recovered": False,
                           "error": f"{type(exc).__name__}: {exc}"}
        if recovered is not None:
            return recovered

    cand = winner["cand"]
    mask_source = cand["source"]
    concept_score = cand.get("concept_score")
    prompt = cand.get("prompt")
    checkpoint_sha = cand.get("checkpoint_sha")
    model_revision = (cand.get("prov") or {}).get("model_revision")
    coarse_prov = cand.get("prov") or {}
    quad = winner["quad"]
    edge_rms, edge_support = winner["edge_rms"], winner["edge_support"]
    cross = winner["cross"]
    debug["coarse"] = np.asarray(cand["coarse"], dtype=np.float64).reshape(4, 2)
    debug["refined"] = winner["refined"]
    debug["edge_rms"] = edge_rms
    debug["edge_support"] = edge_support
    iou = cross.get("iou")
    wrong_face = False  # winner passed the cross-check by construction
    plane_inlier_ratio = cross.get("plane_inlier_ratio")
    plane = cross.get("plane")
    plane_rms_mm = cross.get("plane_rms_mm")
    plane_inliers = cross.get("plane_inliers")
    corners_xyz = cross.get("corners_xyz")
    if cross.get("depth_quad") is not None:
        debug["depth_quad"] = cross["depth_quad"]
    debug["iou"] = iou
    debug["wrong_face"] = wrong_face
    if plane is not None:
        plane_normal = [float(v) for v in np.asarray(
            plane.normal, dtype=np.float64).ravel()]
        plane_offset = float(plane.d)
    else:
        plane_normal, plane_offset = None, None
    # depth_only winners compare the refined quad against the SAME depth
    # estimate that produced the coarse quad: that IoU is self-agreement,
    # not an independent check. Downgrade it to None so it can never look
    # as validated as a live/archived cross-check.
    needs_review = False
    if mask_source == "depth_only":
        if iou is not None:
            reasons.append("depth_self_agreement")
        iou = None
        needs_review = True
    if iou is None:
        # No independent depth check exists (depth missing/rejected, or the
        # depth_only downgrade above): flag for review. Missing evidence
        # must lower confidence via the cap below, never renormalise away.
        needs_review = True
        if "no_independent_depth_check" not in reasons:
            reasons.append("no_independent_depth_check")
    # Non-gating depth notes still recorded (e.g. guided-plane details when
    # the independent estimate_top was fine but the eroded fit was thin).
    reasons.extend([r for r in cross.get("reasons", [])
                    if not r.startswith(("low_rgb_depth_iou",
                                         "low_plane_inlier_ratio",
                                         "high_plane_p95"))])
    # ---- optional fold relocation (corner_fold.enabled) -----------------
    # ---- box profile + measured height (M1/M2/M3) -----------------------
    height_rec = measure_box_height(
        depth_ctx, plane, corners_xyz, plane_rms_mm, plane_inliers)
    box_prof, box_profile_rec = resolve_box_profile(corners_xyz, height_rec)
    generic = bool(box_prof is not None and box_prof.is_generic)
    # generic and later profiles (large_392) take their height from the
    # measurement, else (support hidden / not measurable) from the PROFILE;
    # small/big keep the legacy class tables (bit-identical).
    own = bool(box_prof is not None and box_prof.own_constants)
    meas_h_m = (None if height_rec.get("measured_height_mm") is None
                else float(height_rec["measured_height_mm"]) / 1000.0)
    fold_h_m = meas_h_m
    if fold_h_m is None and own and not generic:
        fold_h_m = box_prof.height_m
    guard_ref_h = (fold_h_m if (own and fold_h_m is not None)
                   else float(rcfg["r2_ref_height_m"]))
    fold_info = None
    if (cfg.get("corner_fold") or {}).get("enabled"):
        guard_px, guard_geom = None, None
        if rcfg["r2_enabled"] and depth_ctx is not None:
            try:
                guard_geom = box_view_geometry(
                    quad, manifest, depth_ctx, guard_ref_h)
                if (rcfg.get("r2_fold_outward_px") is not None
                        and view_needs_trim(guard_geom, rcfg)[0]):
                    guard_px = float(rcfg["r2_fold_outward_px"])
            except Exception:  # noqa: BLE001
                guard_px = None
        quad, corners_xyz, fold_info = apply_corner_fold(
            bgr, quad, manifest, plane, corners_xyz, cfg["corner_fold"],
            outward_limit_px=guard_px,
            profile=box_prof if own else None,
            measured_height_m=fold_h_m)
        if guard_px is not None:
            fold_info["outward_guard"] = {"limit_px": guard_px,
                                          "geometry": guard_geom}
        debug["corner_fold"] = fold_info
        fr = fold_reason(fold_info)
        if fr:
            reasons.append(fr)
    # ---- corner quality gates (flag only, never reject) ------------------
    corner_report: dict = {"corners": {}, "any_flag": False}
    try:
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray_full = None
    if gray_full is not None:
        uv_win, res_win = _corner_depth_winners(
            manifest, plane, W, H,
            region=_corner_checks.depth_query_region(
                quad, [W, H], cfg.get("corner_checks") or {}))
        try:
            corner_report = _corner_checks.check_corners(
                np.asarray(gray_full, dtype=np.float64),
                np.asarray(cand["coarse"], dtype=np.float64).reshape(4, 2),
                quad, [W, H],
                uv_win=uv_win, res_win=res_win,
                refine_cfg=cfg.get("refine"),
                corner_cfg=(cfg.get("corner_checks") or {}))
        except Exception:  # noqa: BLE001 — gates never break the pipeline
            corner_report = {"corners": {}, "any_flag": False,
                             "error": "corner_check_failed"}
    debug["corner_report"] = corner_report
    if corner_report.get("any_flag"):
        needs_review = True
        flagged = sorted(n for n, c in
                         (corner_report.get("corners") or {}).items()
                         if (c.get("flags") if isinstance(c, dict) else []))
        reasons.append(f"corner_flags({','.join(flagged)})")
        for n in flagged:
            c = corner_report["corners"][n]
            reasons.append(f"corner_{n}({','.join(c.get('flags', []))})")
    # ---- R5: clear needs_review that an independent check confirms ------
    # The only sub-reason cleared is a missing RGB/depth IoU (no independent
    # depth check) on a sam3_live winner with NO corner flag: the quad is then
    # confirmed from depth alone (table-parallel top plane hull). Corner
    # flags (fold/refine disagreement) and depth_only winners stay flagged.
    r5_info = None
    conf_iou = iou
    if (rcfg["r5_enabled"] and mask_source == "sam3_live"
            and depth_ctx is not None
            and (needs_review or rcfg.get("r5_always_compute"))):
        with _h_max_scope(depth_ctx, bool(
                own or (meas_h_m is not None
                            and meas_h_m * 1000.0 > BOX_H_TALL_MM))):
            r5_info = independent_top_confirm(quad, depth_ctx, W, H, rcfg)
        rect_fitted = bool(coarse_prov.get("r2_trim")
                           or coarse_prov.get("r3_rect"))
        if needs_review and iou is None and rect_fitted:
            # The R2 trim / R3 rectangle is built from the very top-plane
            # cloud R5 confirms against: the hull IoU is circular.
            r5_info["skipped_rectangle_fitted"] = True
            r5_info["rectangle_source"] = ("r3_rect" if coarse_prov.get(
                "r3_rect") else "r2_trim")
        elif (needs_review and iou is None
                and not corner_report.get("any_flag")
                and r5_info["confirmed"]):
            needs_review = False
            reasons[:] = [r for r in reasons
                          if r != "no_independent_depth_check"]
            reasons.append(
                "needs_review_cleared(no_independent_depth_check:"
                f"top_plane_hull_iou={r5_info['iou']})")
            conf_iou = r5_info["iou"]  # independent evidence (confidence only)
            r5_info["cleared_needs_review"] = True
    seam_dict, seam_stage = compute_seam_stage(
        bgr, quad, cfg, corners_xyz=corners_xyz,
        profile=(box_prof.name if box_prof is not None else None))
    reasons.extend(seam_stage["reasons"])
    seam_status = seam_dict["status"]
    endpoints = seam_dict["endpoints_px"]
    seam_axis = seam_dict["axis"]
    seam_score = seam_dict["score"]
    seam_source = seam_dict["source"]
    estimate = seam_dict["estimate"]
    seam_info = {"estimate_status": estimate.get("status"),
                 "estimate_axis": estimate.get("axis"),
                 "reject": (estimate.get("refine_info") or {}).get(
                     "reject")}
    debug["seam"] = (seam_dict["endpoints_px"] and
                     [tuple(seam_dict["endpoints_px"]["seam_left"]),
                      tuple(seam_dict["endpoints_px"]["seam_right"])])
    debug["seam_info"] = seam_info
    debug["estimate"] = estimate

    # ---- depth diagnostics of the RGB quad + gap-anchored top (K1) --------
    # Recorded for every profile (shadow); K1 only changes what is published
    # for a profile with top_from_gap (large_392).
    depth_diag = top_depth_diagnostics(quad, corners_xyz, plane, plane_rms_mm,
                                       depth_ctx, manifest, W, H, box_prof)
    tfg = top_from_gap_stage(quad, corners_xyz, plane, depth_ctx, manifest, W, H,
                             box_prof, seam_dict)
    tfg_rec = None
    if tfg is not None:
        rgb_px, rgb_xyz = _named_px_xyz(quad, corners_xyz)
        tfg_mode = _top_from_gap_mode()
        tfg_rec = {"mode": tfg_mode, "applied": False, "published": "rgb",
                   "reason": tfg.get("reason"),
                   "rgb_quad": {"corners_px": rgb_px,
                                "corners_xyz_camera_m": rgb_xyz,
                                "plane": ({"normal": [round(float(v), 4) for v in
                                                      np.asarray(plane.normal).ravel()],
                                           "offset_m": round(float(plane.d), 5)}
                                          if plane is not None else None)}}
        for key in ("across_rgb_mm", "along_rgb_mm", "gap_from_centre_mm",
                    "rgb_rect_diffs_mm", "recentred_ends", "end_edge_angle_deg",
                    "deltas_mm", "sides_mm", "known_mm"):
            if key in tfg:
                tfg_rec[key] = tfg[key]
        if tfg.get("plane"):
            fit = tfg["plane"]
            tfg_rec["refit_plane"] = {
                "normal": [round(float(v), 4) for v in fit["n"]],
                "offset_m": round(float(fit["d"]), 5),
                "tilt_deg": round(float(fit["tilt_deg"]), 3),
                "rms_mm": round(float(fit["rms_mm"]), 2),
                "inliers": int(fit["n_inliers"]), "points": int(fit["n_points"])}
        if tfg.get("applied"):
            g_px, g_xyz = _named_px_xyz(tfg["quad_px"], tfg["corners_xyz"])
            tfg_rec["gap_anchored_quad"] = {"corners_px": g_px,
                                            "corners_xyz_camera_m": g_xyz}
            if tfg.get("seam_endpoints"):
                tfg_rec["seam_endpoints_gap_anchored"] = {
                    k: [round(float(v), 2) for v in pt]
                    for k, pt in tfg["seam_endpoints"].items()}
            if tfg_mode == "on":
                tfg_rec["applied"], tfg_rec["published"] = True, "gap_anchored"
                tfg_rec["seam_endpoints_rgb"] = dict(seam_dict["endpoints_px"])
                quad = np.asarray(tfg["quad_px"], dtype=np.float64).reshape(4, 2)
                corners_xyz = np.asarray(tfg["corners_xyz"], dtype=np.float64)
                fit = tfg["plane"]
                plane_normal = [float(v) for v in fit["n"]]
                plane_offset = float(fit["d"])
                plane_rms_mm, plane_inliers = fit["rms_mm"], fit["n_inliers"]
                if tfg.get("seam_endpoints"):
                    ep = tfg["seam_endpoints"]
                    new_eps = _rounded_eps([ep["seam_left"], ep["seam_right"]])
                    if new_eps is not None:
                        endpoints = new_eps
                        seam_dict["endpoints_px"] = new_eps
                        estimate["refined_endpoints_px"] = {
                            "seam_left": list(new_eps["seam_left"]),
                            "seam_right": list(new_eps["seam_right"])}
                        debug["seam"] = [tuple(new_eps["seam_left"]),
                                         tuple(new_eps["seam_right"])]
                debug["gap_anchored_quad"] = quad.copy()
            else:
                reasons.append("top_from_gap_shadow")
        else:
            reasons.append(f"top_from_gap_declined({tfg.get('reason')})")

    # ---- record ----------------------------------------------------------
    named = {name: [round(float(quad[i][0]), 1), round(float(quad[i][1]), 1)]
             for i, name in enumerate(CORNER_ORDER)}
    if corners_xyz is not None and np.all(np.isfinite(corners_xyz)):
        xyz_named = {name: [round(float(corners_xyz[i][0]), 4),
                            round(float(corners_xyz[i][1]), 4),
                            round(float(corners_xyz[i][2]), 4)]
                     for i, name in enumerate(CORNER_ORDER)}
    else:
        xyz_named = None
    if (plane_normal is not None and plane_offset is not None
            and plane_rms_mm is not None and np.isfinite(plane_rms_mm)):
        plane_rec = {"normal": [round(float(v), 4) for v in plane_normal],
                     "offset_m": round(float(plane_offset), 5),
                     "rms_mm": round(float(plane_rms_mm), 2),
                     "inliers": int(plane_inliers or 0)}
    else:
        plane_rec = None
    conf_quad = compute_confidence_quad(concept_score, edge_rms,
                                        plane_inlier_ratio, conf_iou, v2cfg)
    if needs_review:
        conf_quad = min(conf_quad, float(v2cfg["max_conf_no_depth_check"]))
    # Seam confidence is the estimator's observed-line score (refine_score);
    # compute_confidence_seam is 0.0 unless status is ok.
    conf_seam = compute_confidence_seam(seam_status, seam_score)
    record = {
        "schema": "box_autolabel/v5",
        "capture_id": capture_id,
        "image": str(image_path),
        "image_size": [W, H],
        "provenance": {
            "mask_source": mask_source,
            "sam3": ({"checkpoint_sha256": checkpoint_sha,
                      "prompt": prompt,
                      "concept_score": concept_score,
                      "model_revision": model_revision,
                      "box_xyxy": (coarse_prov.get("sam3") or {}).get("box_xyxy"),
                      "n_detections": (coarse_prov.get("sam3") or {}).get("n_detections"),
                      "picked": (coarse_prov.get("sam3") or {}).get("picked"),
                      "detections": (coarse_prov.get("sam3") or {}).get("detections"),
                      "mask_quad_iou": (coarse_prov.get("sam3") or {}).get("mask_quad_iou"),
                      **({"quad_method": "min_area_k4"}
                         if (coarse_prov.get("sam3") or {}).get("quad_method")
                         == "min_area_k4" else {}),
                      **({"used_alternate": int(coarse_prov["alternate_rank"])}
                         if coarse_prov.get("alternate_rank") is not None
                         else {}),
                      **({"plane_trimmed": coarse_prov["plane_trimmed"]}
                         if coarse_prov.get("plane_trimmed") else {}),
                      **({"r2_trim": coarse_prov["r2_trim"]}
                         if coarse_prov.get("r2_trim") else {}),
                      **({"r3_rect": coarse_prov["r3_rect"]}
                         if coarse_prov.get("r3_rect") else {})}
                     if mask_source in ("sam3_archived", "sam3_live") else None),
            "code_git_rev": _code_git_rev(),
            "config": cfg,
            "attempts": attempts,
            **({"corner_fold": fold_info} if fold_info is not None else {}),
            **({"r5": "skipped_rectangle_fitted"}
               if (r5_info or {}).get("skipped_rectangle_fitted") else {}),
        },
        "top_face": {"corners_px": named, "corners_xyz_camera_m": xyz_named,
                     "plane": plane_rec,
                     **({"top_from_gap": tfg_rec} if tfg_rec is not None else {}),
                     **({"depth_diag": depth_diag} if depth_diag is not None else {})},
        "side_faces": [],
        "seam": {"status": seam_status, "endpoints_px": endpoints,
                 "axis": seam_axis, "score": seam_score,
                 "source": seam_source, "estimate": estimate,
                 **({"tape_strip_shadow": seam_dict["tape_strip_shadow"]}
                    if "tape_strip_shadow" in seam_dict else {})},
        "checks": {"rgb_depth_quad_iou": round(float(iou), 3) if iou is not None else None,
                   "wrong_face_suspect": bool(wrong_face),
                   "needs_review": bool(needs_review),
                   **({"depth_top_roi_retry": cross["depth_top_roi_retry"]}
                      if cross.get("depth_top_roi_retry") is not None else {}),
                   **({"independent_top_check": r5_info}
                      if r5_info is not None else {}),
                   "edge_residual_rms_px": round(float(edge_rms), 2)
                   if edge_rms is not None else None,
                   "support_score": round(float(edge_support), 3)
                   if edge_support is not None else None,
                   "corner_flags": corner_report,
                   "reasons": reasons},
        "confidence": round(float(conf_quad), 3),
        "confidence_quad": round(float(conf_quad), 3),
        "confidence_seam": round(float(conf_seam), 3),
        "overlay": None,  # filled by the runner after writing the PNG
        # box type facts (box_profiles) + measured height; the live detector
        # reads name / depth_edge_validated / geometry_axis_validated.
        "box_profile": box_profile_rec if box_profile_rec is not None else {
            "name": None, **height_rec},
    }
    record = apply_corner_names(record, manifest, cfg, debug)
    if trunc_block is not None:
        record["truncation_recovery"] = trunc_block
    return record, debug


# ----------------------------------------------------------------------
# Top-face depth diagnostics + gap-anchored known-size top (2026-10-12)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Truncation recovery (2026-10-13)
# ----------------------------------------------------------------------
# SAM3 sometimes outlines only ONE FLAP of the lid (back edge -> flap gap). The
# half-top passes the depth cross-check (it is coplanar), the profile resolves
# `generic` and the seam is the quad's own front edge. D1 (core/edge_support.py)
# sees it: the coplanar top continues beyond an edge. Recovery tries (a) a SAM3
# alternate that contains the quad, (b) the connected coplanar region of the top
# plane grown from the quad; each candidate runs the full evaluation (refine +
# depth check) and the downstream stages, and replaces the detection only when
# nothing is truncated any more and the seam stage confirms.

# (b) has two coarse quads for the grown region: the 4-gon of its convex hull
# ("depth_grown") and, when that candidate fails, the rectangle anchored on the
# quad's supported edges ("depth_grown_anchored", top_diag.anchored_rect_quad):
# the hull 4-gon skews when a corner of the grown region is chopped (a lifted
# flap corner leaves the depth band), which left the refined back edge tilted
# (2026-10-21: still_truncated / generic_profile_fails_rectangle_gate on ~30 %
# of the dual-arm captures). ``anchored_rect``: ``fallback`` (default; only after
# the hull candidate failed, so everything the hull candidate recovers is
# unchanged), ``first`` or ``off``.

TRUNC_DEFAULTS = {"mode": "off", "min_contain": 0.90, "generic_rect_tol_mm": 20.0,
                  "anchored_rect": "fallback"}


def _truncation_mode(cfg: dict) -> str:
    """``off`` (default; the code path is skipped entirely), ``shadow``
    (recorded, never applied) or ``on``. ``AUTOLABEL_V2_TRUNC_RECOVERY``
    overrides the scene's ``truncation_recovery.mode``."""
    v = os.environ.get("AUTOLABEL_V2_TRUNC_RECOVERY")
    if v is None or not v.strip():
        v = (cfg.get("truncation_recovery") or {}).get("mode", "off")
    if v is True:           # unquoted YAML `on`
        v = "on"
    elif v is False:        # unquoted YAML `off`
        v = "off"
    v = str(v).strip().lower()
    return v if v in ("on", "shadow", "off") else "off"


def _contained_frac(inner: np.ndarray, outer: np.ndarray, W: int, H: int) -> float | None:
    """Share of the area of quad ``inner`` that lies inside quad ``outer``."""
    try:
        mi = np.zeros((H, W), np.uint8)
        mo = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mi, np.asarray(inner, dtype=np.float64).astype(np.int32), 1)
        cv2.fillConvexPoly(mo, np.asarray(outer, dtype=np.float64).astype(np.int32), 1)
    except cv2.error:
        return None
    a = float(mi.sum())
    return float((mi & mo).sum()) / a if a > 0 else None


def _trunc_edges_summary(es: dict, names: list) -> dict:
    ed = es.get("edges") or {}
    return {nm: {"extent_mm": (ed.get(nm) or {}).get("extent_mm"),
                 "continue_frac": (ed.get(nm) or {}).get("continue_frac")}
            for nm in names}


def _quad_json(q) -> list:
    return [[round(float(x), 1), round(float(y), 1)]
            for x, y in np.asarray(q, dtype=np.float64).reshape(4, 2)]


def _truncation_recovery(win0: dict, cands: list, mode: str, cfg: dict,
                         depth_ctx: dict, manifest: dict, W: int, H: int,
                         attempts: list, reasons: list, evaluate, rerun):
    """Check the accepted winner ``win0`` for a truncated top and try to recover.

    ``evaluate(cand, entry)`` = the pipeline's refine + depth check (appends to
    ``attempts``); ``rerun(winner)`` = ``process_capture`` forced onto an
    accepted candidate (profile, seam, record). Returns ``(result | None,
    block | None)``: ``result`` = ``(record, debug)`` of the recovered
    detection (mode ``on`` and every acceptance rule passed) else None;
    ``block`` = the ``truncation_recovery`` provenance record (None when the
    quad is not truncated)."""
    from . import top_diag as _td
    from . import sam3_worker as _w

    tcfg = {**TRUNC_DEFAULTS, **(cfg.get("truncation_recovery") or {})}
    fe = _fold_mod()
    cross0 = win0["cross"]
    n0, d0 = fe._plane_nd(cross0["plane"])
    idx, uv = _depth_projection(depth_ctx)
    if n0 is None or idx is None:
        return None, None
    xyz = depth_ctx["xyz"]
    es0 = _td.edge_check(win0["quad"], (n0, d0), cross0.get("plane_rms_mm"), xyz,
                         (idx, uv), _plane_backprojector(manifest, n0, d0), (W, H))
    bad0 = list(es0.get("unsupported") or [])
    if not bad0:
        return None, None
    block: dict = {
        "mode": mode, "triggered": True, "recovered": False, "applied": False,
        "method": None,
        "trigger": {"edges": bad0, **_trunc_edges_summary(es0, bad0)},
        "before": {"corners_px": _quad_json(win0["quad"]),
                   "top_dims_mm": None},
        "tried": []}
    d_before = _rect_diffs_mm(cross0.get("corners_xyz"))
    c0 = cross0.get("corners_xyz")
    if c0 is not None and np.all(np.isfinite(c0)):
        c = np.asarray(c0, dtype=np.float64).reshape(4, 3)
        sides = [1000.0 * float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
        block["before"]["top_dims_mm"] = [round(0.5 * (sides[0] + sides[2]), 1),
                                          round(0.5 * (sides[1] + sides[3]), 1)]
    block["before"]["rect_diffs_mm"] = (None if d_before is None
                                        else [round(v, 2) for v in d_before])
    n_att, n_rea = len(attempts), len(reasons)
    min_contain = float(tcfg["min_contain"])

    def _try(method: str, cand: dict, extra: dict):
        """evaluate + verify one candidate -> (record, debug) | None."""
        rec_t: dict = {"method": method, **extra}
        block["tried"].append(rec_t)
        entry: dict = {"source": "sam3_live", "truncation_recovery": method}
        if cand.get("alternate_rank") is not None:
            entry["alternate_rank"] = int(cand["alternate_rank"])
            entry["concept_score"] = cand.get("concept_score")
        win = evaluate(cand, entry)
        if win is None:
            rec_t["result"] = "rejected_by_existing_checks"
            return None
        quad2 = win["quad"]
        rec_t["quad_px"] = _quad_json(quad2)
        cf = _contained_frac(win0["quad"], quad2, W, H)
        rec_t["contains_original"] = None if cf is None else round(cf, 3)
        if cf is None or cf < min_contain:
            rec_t["result"] = "does_not_contain_original"
            return None
        pl2 = win["cross"].get("plane")
        n2, d2 = fe._plane_nd(pl2) if pl2 is not None else (None, None)
        if n2 is None:
            rec_t["result"] = "no_plane"
            return None
        es2 = _td.edge_check(quad2, (n2, d2), win["cross"].get("plane_rms_mm"), xyz,
                             (idx, uv), _plane_backprojector(manifest, n2, d2), (W, H))
        bad2 = es2.get("unsupported")
        if bad2 is None or bad2:
            rec_t["result"] = ("edge_check_error" if bad2 is None
                               else f"still_truncated({','.join(bad2)})")
            return None
        res = rerun(win)
        rec2 = res[0]
        prof = (rec2.get("box_profile") or {}).get("name")
        generic = prof in (None, "generic")
        rec_t["profile"] = prof
        rec_t["top_dims_mm"] = (rec2.get("box_profile") or {}).get("top_dims_mm")
        if generic:
            dd = _rect_diffs_mm(win["cross"].get("corners_xyz"))
            tol = float(tcfg["generic_rect_tol_mm"])
            if dd is None or max(dd) > tol:
                rec_t["result"] = "generic_profile_fails_rectangle_gate"
                return None
        seam = rec2.get("seam") or {}
        if seam.get("status") != "ok":
            rec_t["result"] = f"seam_not_ok({seam.get('status')})"
            return None
        if (not generic and getattr(_bp.get(prof), "top_from_gap", False)
                and seam.get("source") != "observed_gap"):
            rec_t["result"] = f"seam_not_observed_gap({seam.get('source')})"
            return None
        ds = ((rec2.get("top_face") or {}).get("depth_diag") or {}).get("edge_support") or {}
        if ds.get("unsupported") is None or ds.get("unsupported"):
            rec_t["result"] = "final_quad_edge_support_not_clean"
            return None
        rec_t["result"] = "verified"
        return res

    def _commit(res, method: str):
        block["method"] = method
        block["recovered"] = True
        rec2 = res[0]
        block["after"] = {"corners_px": (rec2.get("top_face") or {}).get("corners_px"),
                          "top_dims_mm": (rec2.get("box_profile") or {}).get("top_dims_mm"),
                          "profile": (rec2.get("box_profile") or {}).get("name")}

    result = None
    found = None
    # (a) SAM3 alternates of the same response that contain the current quad
    for cand in cands:
        if (cand is win0["cand"] or cand.get("source") != "sam3_live"
                or cand.get("alternate_rank") is None or cand.get("coarse") is None):
            continue
        cf = _contained_frac(win0["quad"], np.asarray(cand["coarse"]), W, H)
        if cf is None or cf < min_contain:
            continue
        res = _try("sam3_alternate", dict(cand), {"alternate_rank": int(cand["alternate_rank"])})
        if res is not None:
            found, result = "sam3_alternate", res
            break
    # (b) depth-grown coplanar region seeded by the quad
    if result is None:
        gate = cross0.get("plane_gate") or {}
        mask, region, ginfo = _td.grow_coplanar_region_ex(
            win0["quad"], (n0, d0), cross0.get("plane_rms_mm"), xyz, (idx, uv),
            depth_ctx["cam"], (W, H),
            band_mm=gate.get("band_mm", float(cfg["guided_plane"]["ransac_threshold_m"]) * 1000.0))
        anchored_mode = str(tcfg.get("anchored_rect", "fallback")).strip().lower()
        if anchored_mode not in ("fallback", "first", "off"):
            anchored_mode = "fallback"
        src = win0["cand"]

        def _grown_cand(coarse: np.ndarray, label: str, rec_info: dict) -> dict:
            prov = dict(src.get("prov") or {})
            prov[label] = rec_info
            return {**{k: v for k, v in src.items()
                       if not k.startswith("_") and k != "skipped"},
                    "coarse": coarse, "prov": prov}

        def _hull_candidate():
            """The convex-hull 4-gon of the grown region."""
            rec_g: dict = {"method": "depth_grown", **ginfo}
            coarse = None
            if mask is not None:
                q4, iou, qmethod = _w._mask_to_quad_ex(mask, H, W)
                if q4 is None:
                    rec_g["result"] = "no_4gon"
                else:
                    coarse = np.asarray(_w._to_named(q4), dtype=np.float64).reshape(4, 2)
                    rec_g["coarse_px"] = _quad_json(coarse)
                    rec_g["hull_quad_iou"] = round(float(iou), 3)
            else:
                rec_g["result"] = ginfo.get("reason", "no_region")
            if coarse is None:
                block["tried"].append(rec_g)
                return None
            return _try("depth_grown",
                        _grown_cand(coarse, "truncation_grown",
                                    {k: v for k, v in rec_g.items() if k != "method"}),
                        {k: v for k, v in rec_g.items() if k != "method"})

        def _anchored_candidate():
            """The rectangle anchored on the supported edges of the quad."""
            if region is None:
                return None
            from . import edge_support as _es
            tr_idx = [_es.EDGE_NAMES.index(nm) for nm in bad0 if nm in _es.EDGE_NAMES]
            q_a, ainfo = _td.anchored_rect_quad(
                win0["quad"], tr_idx, (n0, d0), region, depth_ctx["cam"], (W, H))
            rec_a: dict = {"method": "depth_grown_anchored",
                           "seed_coverage": ginfo.get("seed_coverage"), **ainfo}
            if q_a is None:
                rec_a["result"] = f"anchored_rect_{ainfo.get('reason', 'failed')}"
                block["tried"].append(rec_a)
                return None
            rec_a["coarse_px"] = _quad_json(q_a)
            return _try("depth_grown_anchored",
                        _grown_cand(np.asarray(q_a, dtype=np.float64),
                                    "truncation_anchored_rect",
                                    {k: v for k, v in rec_a.items() if k != "method"}),
                        {k: v for k, v in rec_a.items() if k != "method"})

        order = ([("depth_grown_anchored", _anchored_candidate),
                  ("depth_grown", _hull_candidate)] if anchored_mode == "first"
                 else [("depth_grown", _hull_candidate)]
                 + ([("depth_grown_anchored", _anchored_candidate)]
                    if anchored_mode == "fallback" else []))
        for name, fn in order:
            n_a, n_r = len(attempts), len(reasons)
            res = fn()
            if res is not None:
                found, result = name, res
                break
            # a failed candidate leaves only its ``tried`` entry, not its evaluation notes
            del attempts[n_a:]
            del reasons[n_r:]
    if result is None:
        # nothing recovered: the old behaviour; the extra attempts / reasons are
        # summarised in the block instead of the (unchanged) record lists
        del attempts[n_att:]
        del reasons[n_rea:]
        return None, block
    _commit(result, found)
    if mode != "on":
        del attempts[n_att:]
        del reasons[n_rea:]
        return None, block
    block["applied"] = True
    rec2, dbg2 = result
    rec2["truncation_recovery"] = block
    rs = (rec2.get("checks") or {}).get("reasons")
    if isinstance(rs, list):
        rs.append(f"truncation_recovered({found};trigger={','.join(bad0)})")
    return (rec2, dbg2), block


def _top_diag_off() -> bool:
    return os.environ.get("AUTOLABEL_V2_TOP_DIAG_OFF", "").strip() not in ("", "0")


def _top_from_gap_mode() -> str:
    """``on`` (default; only profiles with ``top_from_gap`` are affected),
    ``shadow`` (computed and recorded, not published) or ``off``."""
    v = os.environ.get("AUTOLABEL_V2_TOP_FROM_GAP", "on").strip().lower()
    return v if v in ("on", "shadow", "off") else "on"


def _plane_backprojector(manifest, n, d):
    """f(uv (N,2)) -> (N,3) camera-frame points where the pixel rays meet the
    plane (the manifest intrinsics, like the guided-plane corners)."""
    fe = _fold_mod()
    km = fe._intrinsics(manifest)
    if km is None:
        raise ValueError("no camera intrinsics")
    return lambda uv: fe._backproject(np.asarray(uv, dtype=np.float64).reshape(-1, 2),
                                      km, np.asarray(n, dtype=np.float64), float(d))


def _named_px_xyz(quad, corners_xyz) -> tuple[dict, dict | None]:
    px = {name: [round(float(quad[i][0]), 1), round(float(quad[i][1]), 1)]
          for i, name in enumerate(CORNER_ORDER)}
    if corners_xyz is not None and np.all(np.isfinite(corners_xyz)):
        xyz = {name: [round(float(corners_xyz[i][0]), 4),
                      round(float(corners_xyz[i][1]), 4),
                      round(float(corners_xyz[i][2]), 4)]
               for i, name in enumerate(CORNER_ORDER)}
    else:
        xyz = None
    return px, xyz


def top_depth_diagnostics(quad, corners_xyz, plane, plane_rms_mm, depth_ctx,
                          manifest, W: int, H: int, box_prof) -> dict | None:
    """D1 / F4 / shadow-fit diagnostics of the final RGB quad (core/top_diag.py).
    None when there is no cloud / plane or the diagnostics are switched off
    (``AUTOLABEL_V2_TOP_DIAG_OFF=1``); never raises."""
    if _top_diag_off() or depth_ctx is None or plane is None:
        return None
    try:
        from . import top_diag as _td

        n, d = _fold_mod()._plane_nd(plane)
        if n is None:
            return None
        idx, uv = _depth_projection(depth_ctx)
        if idx is None:
            return None
        dims = None
        if box_prof is not None and not box_prof.is_generic:
            if (box_prof.outer_dims_mm.get("along_seam")
                    and box_prof.outer_dims_mm.get("across_seam")):
                dims = (box_prof.outer_dims_mm["along_seam"],
                        box_prof.outer_dims_mm["across_seam"])
            elif box_prof.top_dims_mm:
                dims = tuple(box_prof.top_dims_mm)
        return _td.run(quad, corners_xyz, (n, d), plane_rms_mm,
                       depth_ctx["xyz"], (idx, uv), depth_ctx["cam"],
                       _plane_backprojector(manifest, n, d), (W, H), dims,
                       box_prof.name if box_prof is not None else None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def top_from_gap_stage(quad, corners_xyz, plane, depth_ctx, manifest, W: int, H: int,
                       box_prof, seam_dict) -> dict | None:
    """K1 for a profile with ``top_from_gap`` and an observed-gap seam: the raw
    :func:`top_from_gap.compute` result (``applied`` False + reason when it
    declines) or None when K1 does not concern this detection."""
    if (box_prof is None or not getattr(box_prof, "top_from_gap", False)
            or _top_from_gap_mode() == "off" or depth_ctx is None or plane is None):
        return None
    if (seam_dict.get("status") != "ok" or seam_dict.get("source") != "observed_gap"
            or not seam_dict.get("endpoints_px")):
        return {"applied": False, "reason": "seam_not_observed_gap"}
    try:
        from . import top_from_gap as _tfg

        fe = _fold_mod()
        n, d = fe._plane_nd(plane)
        if n is None:
            return {"applied": False, "reason": "no_plane"}
        idx, uv = _depth_projection(depth_ctx)
        if idx is None:
            return {"applied": False, "reason": "no_cloud"}
        km = fe._intrinsics(manifest)
        dims = box_prof.outer_dims_mm
        if not (dims.get("along_seam") and dims.get("across_seam")) or km is None:
            return {"applied": False, "reason": "no_known_dims"}
        return _tfg.compute(
            quad, seam_dict["endpoints_px"], seam_dict.get("axis"),
            depth_ctx["xyz"], (idx, uv), (n, d),
            lambda nn, dd: _plane_backprojector(manifest, nn, dd),
            lambda X: fe._project(np.asarray(X, dtype=np.float64), km),
            float(dims["along_seam"]), float(dims["across_seam"]), (W, H))
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "reason": f"error({type(exc).__name__}: {exc})"}



def apply_corner_names(record: dict, manifest, cfg: dict,
                       debug: dict | None = None,
                       source: str | None = None) -> dict:
    """Rename the final corners by ``corner_names.RULE`` (names only).

    Permutes corners_px / corners_xyz / corner flags / fold per-edge data
    and swaps the seam axis label when needed (see
    :func:`corner_names.rename_record`); geometry and seam endpoints are
    untouched. No-op when ``cfg["corner_names"]["enabled"]`` is false.
    ``debug["estimate"]`` follows the renamed record for the overlay.
    """
    from . import corner_names as _cn

    ncfg = (cfg or {}).get("corner_names") or {}
    if not ncfg.get("enabled"):
        return record
    out, _info = _cn.rename_record(record, manifest, source=source,
                                   code_git_rev=_code_git_rev())
    if debug is not None and isinstance(out.get("seam"), dict):
        debug["estimate"] = out["seam"].get("estimate")
    return out


def _rounded_eps(seam_pts) -> dict | None:
    """Sort [left, right] by x and round to 0.1 px; None when unusable."""
    try:
        pts = np.asarray(seam_pts, dtype=np.float64).reshape(2, 2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(pts)):
        return None
    sl, sr = sorted([tuple(map(float, pts[0])), tuple(map(float, pts[1]))],
                    key=lambda p: p[0])
    return {"seam_left": [round(sl[0], 1), round(sl[1], 1)],
            "seam_right": [round(sr[0], 1), round(sr[1], 1)]}

def compute_seam_stage(bgr: np.ndarray, quad: np.ndarray, cfg: dict,
                       corners_xyz=None, profile: str | None = None
                       ) -> tuple[dict, dict]:
    """Run the flap-gap seam stage for accepted corners (estimator only).

    Runs ``estimate_seam`` (projective-midline prior + observed flap-gap
    evidence) and applies the decision tree, returning ``(seam_dict,
    debug_dict)`` where ``seam_dict`` has keys status/endpoints_px/axis/
    score/source/estimate. Default mode is ``confirm``: the published line
    is always the prior on the evidence-chosen axis; ``refine`` mode
    (``seam_estimator.seam_mode``) publishes the observed gap line for
    comparison. Statuses: ok = prior confirmed by gap (confirm; source
    ``prior_confirmed_by_gap``) or gap observed (refine; source
    ``observed_gap``); estimated = axis clear but unconfirmed/unobserved
    (source ``prior``; endpoints None at top level, estimate-only
    endpoints carry the prior line, never a cutting target); ambiguous =
    axis unclear; invisible = no axis evidence. ``corners_xyz`` is
    REQUIRED for ok (no valid scale -> ``no_metric_scale`` -> estimated).
    ``profile``: box profile name (None = legacy behaviour); generic boxes get
    no R10 geometry axis and no view shift.
    """
    from . import seam_estimator as _est

    reasons: list[str] = []
    debug: dict = {}
    t_shadow = _tape_shadow_enabled()
    try:
        estimate = _est.estimate_seam(
            bgr, np.asarray(quad, dtype=np.float64).reshape(4, 2),
            est_cfg=cfg.get("seam_estimator"), corners_xyz=corners_xyz,
            profile=profile)
    except (ValueError, cv2.error) as exc:
        estimate = {"method": "projective_midline", "axis": None,
                    "candidates": [], "prior_endpoints_px": None,
                    "estimate": None,
                    "refined_endpoints_px": None, "refine_score": None,
                    "source": None, "status": "ambiguous",
                    "error": f"estimator_failed({exc})"}
        reasons.append(f"seam_estimator_failed({exc})")
    est_status = estimate.get("status")
    debug["estimate"] = estimate
    # ---- decision tree -------------------------------------------------
    # Confirm (default): ok = prior confirmed by the observed gap line
    # (source prior_confirmed_by_gap; endpoints = prior ∩ face edges).
    # Refine: ok = observed gap line (source observed_gap). Estimated =
    # axis clear but unconfirmed (source prior; top-level endpoints null,
    # never a cutting target; the prior line lives in estimate only).
    # "ambiguous" means the axis itself is unclear; "invisible" means no
    # axis evidence at all.
    if est_status == "ok" and estimate.get("refined_endpoints_px"):
        reps = estimate["refined_endpoints_px"]
        src = estimate.get("source") or "observed_gap"
        seam_dict = {
            "status": "ok",
            "endpoints_px": _rounded_eps(
                [reps["seam_left"], reps["seam_right"]]),
            "axis": estimate.get("axis"),
            "score": round(float(estimate.get("refine_score", 0.0)), 3),
            "source": src,
            "estimate": estimate,
        }
        reasons.append("seam_estimate_ok")
        if estimate.get("confirm_reject"):
            reasons.append(f"seam_{estimate['confirm_reject']}")
    elif est_status == "estimated":
        seam_dict = {"status": "estimated",
                     "endpoints_px": None, "axis": estimate.get("axis"),
                     "score": 0.0, "source": "prior",
                     "estimate": estimate}
        reasons.append("seam_estimate_estimated")
        if estimate.get("axis_source") == "geometry_class":
            reasons.append("seam_axis_from_geometry")
        reject = (estimate.get("refine_info") or {}).get("reject") \
            or estimate.get("confirm_reject")
        if reject:
            reasons.append(f"seam_{reject}")
    elif est_status == "ambiguous":
        seam_dict = {"status": "ambiguous", "endpoints_px": None,
                     "axis": None, "score": None, "source": None,
                     "estimate": estimate}
        reasons.append("seam_estimate_ambiguous")
    else:
        seam_dict = {"status": "invisible", "endpoints_px": None,
                     "axis": None, "score": None, "source": None,
                     "estimate": estimate}
        reasons.append(f"seam_estimate_{est_status}")
    if t_shadow:
        # SHADOW ONLY: recorded, never used for the published seam/axis/gates.
        seam_dict["tape_strip_shadow"] = tape_strip_shadow(
            bgr, quad, corners_xyz, seam_dict.get("axis"))
    return seam_dict, {"reasons": reasons, "debug": debug}


def _tape_shadow_enabled() -> bool:
    return os.environ.get("AUTOLABEL_V2_TAPE_SHADOW", "") == "1"


def tape_strip_shadow(bgr, quad, corners_xyz, published_axis) -> dict:
    """Compact summary of ``tape_strip.detect_on_image`` on the final quad.

    Keys: ok, best_axis (published-axis convention lr/bf), axis_margin,
    n_candidates / n_qualified per axis, best (centre_px, width_mm, coverage,
    n_channels_agree, channels), gap_line_found, agrees_with_published_axis,
    reasons, time_s. Never raises (``error`` instead)."""
    import time

    t0 = time.perf_counter()
    out: dict = {"ok": False, "best_axis": None, "axis_margin": None,
                 "n_candidates": {"lr": 0, "bf": 0},
                 "n_qualified": {"lr": 0, "bf": 0}, "best": None,
                 "gap_line_found": False,
                 "agrees_with_published_axis": None}
    try:
        from . import tape_strip as _ts

        cxyz = None
        if isinstance(corners_xyz, dict):
            corners_xyz = [corners_xyz[k] for k in CORNER_ORDER]
        if corners_xyz is not None:
            c = np.asarray(corners_xyz, dtype=np.float64)
            if c.shape == (4, 3) and np.all(np.isfinite(c)):
                cxyz = c
        res = _ts.detect_on_image(
            bgr, np.asarray(quad, dtype=np.float64).reshape(4, 2), cxyz)
        cands = res.get("candidates") or []
        for cd in cands:
            out["n_candidates"][cd["axis"]] += 1
            if cd.get("qualified"):
                out["n_qualified"][cd["axis"]] += 1
        best = _ts.best_candidate(res)
        out.update(ok=bool(res.get("ok")), best_axis=res.get("best_axis"),
                   axis_margin=round(float(res.get("axis_margin") or 0.0), 3),
                   reasons=list(res.get("reasons") or []))
        if best is not None:
            gl = best.get("gap_line")
            out["best"] = {
                "axis": best.get("axis"),
                "center_px": round(float(best["center_px"]), 2),
                "width_mm": (None if best.get("width_mm") is None
                             else round(float(best["width_mm"]), 2)),
                "coverage": round(float(best.get("coverage", 0.0)), 3),
                "n_channels_agree": best.get("n_channels_agree"),
                "channels": {k: round(float(v), 2) for k, v in
                             (best.get("channels") or {}).items()},
                "score": round(float(best.get("score", 0.0)), 3)}
            out["gap_line_found"] = gl is not None
        if published_axis in ("lr", "bf") and out["best_axis"] is not None:
            out["agrees_with_published_axis"] = bool(
                out["best_axis"] == published_axis)
    except Exception as exc:  # noqa: BLE001 — shadow never breaks the stage
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["time_s"] = round(time.perf_counter() - t0, 3)
    return out


def _fail_record(capture_id: str, paths: dict, cfg: dict, status: str,
                 reasons: list[str], mask_source: str, image_size=None,
                 concept_score=None, prompt=None, checkpoint_sha=None,
                 model_revision=None, depth_est=None,
                 depth_err=None, edge_rms=None, edge_support=None,
                 attempts=None) -> dict:
    v2cfg = cfg["v2"]
    _ = v2cfg
    # Null-corner records carry no geometry: confidence is 0.0 regardless
    # of partial evidence (concept/edge terms must not inflate a failure).
    image = paths.get("image")
    fail_reasons = list(reasons)
    return {
        "schema": "box_autolabel/v5",
        "capture_id": capture_id,
        "image": str(image) if image else "",
        "image_size": list(image_size) if image_size else [0, 0],
        "provenance": {
            "mask_source": mask_source,
            "sam3": ({"checkpoint_sha256": checkpoint_sha, "prompt": prompt,
                      "concept_score": concept_score,
                      "model_revision": model_revision}
                     if mask_source in ("sam3_archived", "sam3_live") else None),
            "code_git_rev": _code_git_rev(),
            "config": cfg,
            "attempts": list(attempts or []),
        },
        "top_face": {"corners_px": None, "corners_xyz_camera_m": None,
                     "plane": None},
        "side_faces": [],
        "seam": {"status": "invisible", "endpoints_px": None,
                 "axis": None, "score": None, "source": None,
                 "estimate": None},
        "checks": {"rgb_depth_quad_iou": None,
                   "wrong_face_suspect": False,
                   "needs_review": True,
                   "edge_residual_rms_px": (round(float(edge_rms), 2)
                                            if edge_rms is not None else None),
                   "support_score": (round(float(edge_support), 3)
                                     if edge_support is not None else None),
                   "corner_flags": {"corners": {}, "any_flag": False,
                                    "note": "no_quad"},
                   "reasons": fail_reasons},
        "confidence": 0.0,
        "confidence_quad": 0.0,
        "confidence_seam": 0.0,
        "overlay": None,
        "box_profile": None,
    }
