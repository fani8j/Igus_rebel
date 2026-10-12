# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/config.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""v2 pipeline thresholds (recorded verbatim in provenance.config).

Kept in one dict so outputs stay reproducible and reviewable. The seam
section reuses scripts/autolabel DEFAULTS values; the v2_* section holds
the new wrong-face/confidence gates.
"""

from __future__ import annotations

import copy
import os

from . import box_profiles as _bp
from .fold_edge import BOX_HEIGHT_M as _BOX_HEIGHT_M
from .fold_edge import DEFAULTS as _FOLD_DEFAULTS

V2_DEFAULTS: dict = {
    # Subpixel edge refinement (refine_quad_edges kwargs).
    "refine": {
        "profile_half_width_px": 12,
        "sample_spacing_px": 6,
        "min_gradient": 4,
        "max_offset_px": 15,
        "min_samples": 8,
    },
    # Guided depth plane (fit_sam3_guided_depth_plane kwargs).
    "guided_plane": {
        "erosion_px": 8,
        "ransac_threshold_m": 0.003,
        "min_inliers": 100,
    },
    # Seam search: exact copy of scripts/autolabel DEFAULTS["seam"] so the
    # frozen detector is reused, not re-tuned. Both rect axes are searched
    # by find_seam itself (no long-axis-only gate is applied in v2).
    "seam": {
        "rect_w": 400,
        "rect_h": 300,
        "min_span_frac": 0.55,
        "align_tol_deg": 15.0,
        "margin_frac": 0.08,
        "w_span": 0.50,
        "w_contrast": 0.40,
        "w_center": 0.10,
        "termination_ext_frac": 0.3,
        "termination_edge_frac": 0.25,
        # Termination-gate Canny thresholds on the full frame. The frozen
        # seam search uses (30, 90) on the rectified warp; the termination
        # gate here intentionally uses (50, 150) on full-frame blur to stay
        # specific to strong background lines, not weak print texture.
        "termination_canny_lo": 50,
        "termination_canny_hi": 150,
        "min_seam_score": 0.40,
        "ambiguity_margin": 0.08,
        "fallback_min_score": 0.44,
    },
    # Projective-midline flap-gap seam estimator (scripts/autolabel_v2/
    # seam_estimator.py). Axis gates from the reviewed ok gap seams
    # (dev set, not generalisation): true-axis tape evidence 14.8..47.9,
    # wrong-axis 2.4..8.6, synthetic noise < 1.0; per-axis gap support
    # recorded alongside (true axis 0.9..1.0, wrong axis 0.2..0.5).
    # Observed-gap gates: thin dark valley (depth >= 12 gray levels,
    # bright shoulders >= 8 each side, width <= 6 rect px) over >= 0.40
    # of the face, no gap > 0.35, residual <= 2.5 px, gap width in
    # [0.5, 4.0] mm via the top-plane scale (dev measured 1..3 mm).
    # Legacy tape_* keys kept so older stored configs still parse; the
    # estimator reads the gap_* keys (tape stats kept for axis audit).
    "seam_estimator": {
        "axis_min_evidence": 8.0,
        "axis_min_margin_ratio": 0.25,
        "seam_mode": "either_confirm",
        # R10: when the tape evidence cannot choose the axis, take it from
        # the small-box top-face geometry (seam joins the midpoints of the
        # two LONG sides). Published as estimated, never ok.
        "geometry_axis": {
            "enabled": True,
            # ranges come from the small profile; R10 is applied only when
            # the box profile has geometry_axis_validated (small).
            "small_long_mm": list(_bp.PROFILES["small"].r10_long_mm),
            "small_short_mm": list(_bp.PROFILES["small"].r10_short_mm),
            "min_ratio": 1.05,
            "min_diff_mm": 5.0,
            "opposite_tol_mm": 3.0,
        },
        "confirm_max_offset_mm": 1.5,
        "confirm_max_angle_deg": 2.0,
        "confirm_min_coverage": 0.40,
        "gap_min_depth": 12.0,
        "gap_min_shoulder": 8.0,
        "gap_max_width_px": 6,
        "min_support_frac": 0.5,
        "min_coverage": 0.40,
        "max_gap_frac": 0.35,
        "max_residual_px": 2.5,
        "gap_width_mm_min": 0.5,
        "gap_width_mm_max": 4.0,
        "darkline_min_contrast": 6.0,
        "darkline_halfwin_px": None,
        "darkline_min_coverage": 0.40,
        "darkline_max_gap_frac": 0.35,
        "darkline_max_residual_px": 2.5,
        # Each darkline shoulder alone must be >= this brighter than the
        # valley (rejects one-sided tape-edge-highlight steps). all853_v8:
        # one-sided stations are rare (median 0-1 per capture). With the
        # view shift ON (windows centred on the expected line) the gate at
        # 2.0 changes no seam status; with the shift OFF (prior-centred
        # windows) it drops 7 big-box darkline confirmations (480 -> 473
        # ok). run.py --no-view-shift therefore also sets this to 0.0
        # (pre-v10 behaviour); --no-shoulder-gate zeroes it alone.
        "darkline_min_shoulder": 2.0,
        # Viewing shift: observed gap/darkline appear displaced AWAY from
        # the camera vs the (physically centred) projective midline.
        # Confirmation compares them with prior + s, s = shift_mm[class] +
        # k_tan_mm[class] * tan(view angle across the seam); the published
        # line stays the prior. Fitted on UNLABELLED all853_v8 (class
        # medians of the gated gap-valley mid-face offset, search window
        # centred on s until self-consistent): small 3.17 mm (n=142, MAD
        # 0.34), big 0.52 mm (n=572, MAD 0.44); no tan(angle) trend within
        # a class -> k = 0. Class: top long side > 150 mm = big. See
        # seam_estimator.VIEW_SHIFT_MM for the derivation.
        "view_shift": {
            "enabled": True,
            # per-class shift from the box profiles (generic: 0, see
            # seam_estimator.expected_view_shift(profile=...))
            "shift_mm": _bp.view_shift_mm_table(),
            "k_tan_mm": {"big": 0.0, "small": 0.0},
            "big_long_side_m": _bp.LEGACY_BIG_LONG_SIDE_M,
            "center_windows": True,
        },
        "tape_width_mm_min": 4.0,
        "tape_width_mm_max": 15.0,
    },
    # Independent depth-top estimate (estimate_top dcfg).
    "depth_top": {
        "ws_x_min": -1.2,
        "ws_x_max": 1.2,
        "ws_y_min": -0.8,
        "ws_y_max": 0.8,
        "ws_z_min": 0.2,
        "ws_z_max": 2.0,
        "ransac_thresh": 0.005,
        "ransac_iters": 300,
        "ransac_seed": 0,
        "ransac_downsample": 20000,
        "max_planes": 4,
        "table_min_inliers": 20000,
        "top_min_inliers": 5000,
        "parallel_tol_deg": 10.0,
        "offset_min": 0.05,
        # see scripts/autolabel/config.py (0.60: boxes up to ~0.6 m tall;
        # planes beyond offset_above_check_from must be above the table)
        "offset_max": 0.60,
        "offset_above_check_from": 0.30,
        "support_expand_m": 0.10,
        # Support model of the scene (config/scenes/*.yaml):
        #   visible_table: the box stands on a table the camera sees (today);
        #     the top must lie inside the dominant plane's footprint.
        #   hidden_stand:  the support under the box is hidden (stand smaller
        #     than the box): the dominant plane is the floor, the top is a large
        #     parallel plane [offset_min, offset_max] above it; the floor
        #     distance is NEVER the box height (measured height unavailable).
        "support": "visible_table",
        # record the scene depth noise (median plane RMS) in the depth
        # estimate; read by the noise-adaptive guided-plane gate (v2.noise_adaptive)
        "measure_noise": False,
        "front_min_angle_deg": 75.0,
        "front_max_angle_deg": 90.0,
        "front_min_inliers": 2000,
        "front_adjacency_max_m": 0.8,
        "top_merge_max_offset": 0.01,
        "top_merge_max_gap": 0.01,
        "min_hull_area_frac": 0.01,
        "min_support_dominance": 0.5,
        "quad_approx_eps": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        "min_quad_iou": 0.80,
        "top_rms_max": 0.003,
        "top_min_object_ratio": 0.25,
        "seam_inside_tol_px": 5.0,
        "support_cell_px": 4,
        "seam_support_radius_px": 6.0,
        "seam_support_min_coverage": 0.6,
        "seam_support_max_gap_frac": 0.3,
        "seam_support_endpoint_radius_px": 12.0,
    },
    # v2 wrong-face / confidence gates.
    "v2": {
        "min_rgb_depth_iou": 0.50,
        # Guided-plane support gate (fires, unlike inlier RMS which is
        # < 3 mm by construction): eroded-interior inlier share below this
        # flags wrong_face_suspect.
        "min_plane_inlier_ratio": 0.80,
        # p95 absolute residual (mm) over ALL eroded-interior points (not
        # just RANSAC inliers) above this flags wrong_face_suspect.
        "max_plane_p95_mm": 8.0,
        # Noise-adaptive guided-plane gate (off = the two fixed thresholds above
        # and guided_plane.ransac_threshold_m). The scene noise is the median
        # RMS (mm) of the large planes estimate_top fitted (5 mm RANSAC band).
        # excess = max(0, noise - rms_floor_mm); then
        #   band    = guided_plane.ransac_threshold_m + band_k_mm_per_mm * excess
        #   p95 max = max_plane_p95_mm + p95_k_mm_per_mm * excess + bulge
        # (never below today's values; bulge = bulge_mm_per_m * longest top side
        # [m] capped at bulge_max_mm, only for tops >= bulge_min_side_m).
        "noise_adaptive": {
            "enabled": False,
            "rms_floor_mm": 1.8,
            "band_k_mm_per_mm": 4.0,
            "max_band_mm": 8.0,
            "p95_k_mm_per_mm": 2.0,
            "bulge_mm_per_m": 5.0,
            "bulge_max_mm": 2.0,
            "bulge_min_side_m": 0.25,
        },
        # Quad confidence cap when no independent depth check exists.
        "max_conf_no_depth_check": 0.50,
        # Confidence weights: concept, edge, plane-support, iou.
        # (Seam has its own confidence_seam; see compute_confidences.)
        "w_concept": 0.25,
        "w_edge": 0.30,
        "w_plane": 0.25,
        "w_iou": 0.20,
        # Edge residual RMS (px) mapping: 0 px -> 1, >= this -> 0.
        "edge_rms_full_px": 3.0,
    },
    # Corner quality gates (flag-only → needs_review, never reject).
    # Px thresholds scale by image_width/1280. Operating point: only
    # shift + border flag (catches all 6 bad captures, ~4 false flags on
    # dev); ambiguous_edge fires on ~25% of good corners (tape stripes
    # parallel to edges) and depth_off_plane fires on most good corners
    # (z-buffer winners near a box tip are usually background), so both
    # are informational by default (flag_ambiguous=False,
    # flag_depth=False) — per-corner values stay in checks.corner_flags.
    "corner_checks": {
        "shift_px": 10.0,
        "border_px": 8.0,
        "ambiguous_frac": 0.3,
        "flag_ambiguous": False,
        "depth_radius_px": 3.0,
        "depth_inlier_mm": 8.0,
        "depth_min_frac": 0.5,
        "flag_depth": False,
    },
    # Corner fold relocation (scripts/autolabel_v2/fold_edge.py): moves the
    # refined edges from the strongest-gradient feature (rim / side-face
    # silhouette) to the top/side fold using a side-face projection model
    # (top plane + intrinsics + box height by footprint class). Off by
    # default; hidden_mode "keep" leaves edges whose side face cannot be
    # visible untouched ("shoulder" also snaps those to the inner shoulder).
    # Box heights are the user-measured values (big / small).
    # Enabled 2026-09-27: on a fresh small-box blind batch (reviewed/blind_small2_*,
    # rule fixed before scoring: lowest median corner mm) "shoulder" won:
    # 1.71 mm vs 1.89 (keep) vs 2.38 (off).
    "corner_fold": {
        "enabled": True,
        "box_height_m": dict(_BOX_HEIGHT_M),
        **{**copy.deepcopy(_FOLD_DEFAULTS), "hidden_mode": "shoulder"},
    },
    # Reliability fixes R2/R3/R5 (docs/detection_optimization_plan.md s3.3).
    # R2: when the box is yawed vs the camera (>= r2_yaw_deg, folded to
    # [0,45]) or a second side wall is visible (view cos >= r2_wall_cos), the
    # SAM3 outline may include the visible side face; if the table-parallel
    # plane-trimmed quad moves a corner by >= r2_min_spill_px, the trimmed
    # quad is refined instead (the original quad is the fallback), and the
    # corner fold may not move an edge facing a visible wall more than
    # r2_fold_outward_px outward (onto the side wall; None = off). The
    # rewritten quad (R2/R3) is used only when it is strictly more
    # rectangle-consistent in 3-D than the SAM3 outline AND the outline is
    # grossly inconsistent (> r_keep_original_rect_mm; blind labels showed a
    # rewrite of an outline that is only a few mm off loses accuracy, the
    # fitted rectangle being ~3-5 mm larger than the box per side). Otherwise
    # the old path runs unchanged.
    # R3: a quad from the min-area k=4 fallback (hexagonal mask) is replaced
    # by an orthogonal min-area rectangle of the top-plane inlier cloud
    # (accepted when its pixel IoU with the k4 quad is >= r3_min_iou; known
    # box sizes are a soft check only).
    # R5: clear needs_review when every sub-reason is independently
    # confirmed (pipeline._r5_clear_needs_review). Each fix can be disabled.
    "reliability": {
        "r2_enabled": True,
        "r2_yaw_deg": 20.0,
        "r2_wall_cos": 0.25,
        "r2_ref_height_m": 0.10,
        "r2_min_spill_px": 4.0,
        "r_keep_original_rect_mm": 10.0,
        "r2_fold_outward_px": 1.5,
        "r3_enabled": True,
        "r3_dilate": 1.25,
        "r3_band_mm": 3.0,
        "r3_min_iou": 0.5,
        "r3_dims_tol_mm": 8.0,
        "r3_known_dims_mm": _bp.known_dims_mm_table(),
        "r5_enabled": True,
        "r5_dilate": 1.2,
        "r5_min_iou": 0.85,
        "r5_always_compute": False,
    },
    # Corner naming (scripts/autolabel_v2/corner_names.py, user-approved
    # 2026-09-27): back edge = the LONG top-face side farther from the
    # camera (3-D, on the top plane); back_left/back_right = its smaller/
    # larger image-x end; front_* adjacent. Applied to the FINAL corners
    # (after fold + corner checks + seam) by permuting names only.
    "corner_names": {
        "enabled": True,
        "rule": "long_side_back_v1",
    },
}

#: Fixed corner order used across the v2 pipeline (ring order; which
#: physical corner is "back_left" is set by corner_names.RULE).
CORNER_ORDER = ("back_left", "back_right", "front_right", "front_left")


def default_config(scene=None) -> dict:
    """Return a deep copy of the v2 defaults (``scene``: a
    :class:`scene.Scene` whose overrides are merged in; None = the defaults,
    which are the ``table_cell_v1`` scene).

    ``AUTOLABEL_V2_RELIABILITY_OFF=1`` in the environment switches the
    R2/R3/R5 reliability fixes off (A/B replays and rollback; the old path)."""
    cfg = copy.deepcopy(V2_DEFAULTS)
    if os.environ.get("AUTOLABEL_V2_RELIABILITY_OFF") == "1":
        for k in ("r2_enabled", "r3_enabled", "r5_enabled"):
            cfg["reliability"][k] = False
    if scene is not None:
        scene.apply(cfg)
    return cfg
