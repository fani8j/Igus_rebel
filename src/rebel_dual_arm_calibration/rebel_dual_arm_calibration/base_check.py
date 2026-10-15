"""Touch-only estimate of the right base, TCP consistency and the trust gate (numpy only, no ROS).

Both arms touch the same printed board points (same point ids).  Left TCP positions are in the world frame
(world = arm_left_base_link); right TCP positions are in the arm_right base frame.  A rigid Kabsch fit between
them gives ``T_world_rightbase`` WITHOUT the camera and WITHOUT the table board pose: an independent estimate
that only uses the nominal forward kinematics (TF) that the deployed robot also uses.

The trust gate decides whether the right base may be used for collision avoidance between the arms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrology import predict_base_noise, recommended_padding_mm, repeat_scatter, touch_kind
from .transforms import R_to_rpy, R_to_rotvec, inv_T, kabsch, rotation_angle_deg

ARMS = ("left", "right")


@dataclass
class GateOptions:
    min_common_points: int = 4          # touched by both arms (same point ids): board crosshairs + fit stickers
    max_touch_rms_mm: float = 1.5       # fit residual (FRE, in-sample), out-of-fit loop closure and held-out touch RMS
    max_joint_vs_touch_mm: float = 1.0  # FLOOR of the joint vs touch-only agreement (mm); the limit is max(floor, 3 sigma)
    max_joint_vs_touch_deg: float = 0.1  # FLOOR of the rotation agreement per axis (deg)
    agree_sigma_k: float = 3.0          # agreement limit = max(floor, k x predicted 1-sigma of the touch-only base)
    reach_m: float = 0.4                # representative reach point distance from the right base
    max_amplification: float = 2.0      # warn: touch noise -> reach-point error amplification of the Kabsch fit
    min_orientation_deg: float = 20.0   # wrist orientations closer than this do not test the TCP
    tcp_warn_mm: float = 1.5            # warn: TCP consistency spread above this
    min_tilt_sigma: float = 0.2         # TCP check: sigma_min(B)/sqrt(N) of B_i = R_i - mean(R) per point (pivot measure)
    # ---- held-out evidence ----
    max_heldout_px: float = 1.0         # held-out reprojection RMS of the written variant
    min_check_points: int = 2           # check points touched by both arms
    check_min_height_mm: float = 30.0   # at least one check point this far off the table-board plane
    check_min_dist_mm: float = 150.0    # at least one check point this far from the board centre
    max_check_error_mm: float = 3.0     # max check-point error (TRE, held out)
    max_yaw_sigma_deg: float = 0.15     # geometry adequacy: predicted 1-sigma yaw of the touch-only base
    max_reach_sigma_mm: float = 1.0     # ... and of the reach point (3D RMS)
    max_intrinsics_f_frac: float = 0.003   # intrinsics pull (what --refine-intrinsics would move) in f ...
    max_intrinsics_c_px: float = 3.0       # ... and in the principal point
    assumed_touch_sigma_mm: float = 0.5    # per-axis repeat scatter assumed when an arm has no repeated touches
    n_mc: int = 400                     # Monte Carlo draws of the touch-only base noise prediction


def _points_by_id(touches: Sequence, arm: str) -> Dict[int, list]:
    out: Dict[int, list] = {}
    for t in touches:
        if t.arm == arm:
            out.setdefault(int(t.point_id), []).append(t)
    return out


def tilt_observability(Rs: Sequence[np.ndarray]) -> dict:
    """Observability of the TCP offset from touches of ONE point at the rotations ``Rs`` (the pivot-method measure).

    The tip is p_i = c + R_i r (c: the point, r: the TCP offset in the tool frame).  With B_i = R_i - mean(R), the
    stacked (3N x 3) matrix B decides which components of r the position spread can see: ``sigma_min(B)/sqrt(N)``
    (0 for a pure roll about one axis, which leaves the TCP component along that axis invisible; about 0.8 for
    two-axis tilts of 30 deg or more) and ``weakest_direction_tool`` (unit vector in the tool frame, the right
    singular vector of sigma_min).  N = 2 always has rank 2 (the rotation axis between the two is invisible)."""
    Rs = [np.asarray(R, dtype=float)[:3, :3] for R in Rs]
    n = len(Rs)
    B = np.vstack([R - np.mean(Rs, axis=0) for R in Rs])
    _, sv, Vt = np.linalg.svd(B)
    rank = int(np.sum(sv > 1e-9 * max(1.0, sv[0])))
    return {"n": n, "rank": rank, "sigma_min_norm": float(sv[-1] / math.sqrt(n)),
            "weakest_direction_tool": [float(v) for v in Vt[-1]]}


def tcp_consistency(touches: Sequence, opts: Optional[GateOptions] = None) -> dict:
    """Spread of the tip position over touches of the SAME arm on the SAME point with different wrist
    orientations (rigid-motion invariant, so it needs no base pose): a CONSISTENCY INDICATOR, not an upper bound on
    the TCP error.  It contains the TCP offset error along the directions the tilts make visible, plus joint zero
    offsets / backlash / touch repeatability (about 1-2.5 mm for 0.4 deg joint zero errors in a Monte Carlo), and it
    is blind to a TCP error along an axis the tilts do not move.

    A point qualifies only with tilt diversity (:func:`tilt_observability`: rank 3 and sigma_min(B)/sqrt(N) >=
    ``opts.min_tilt_sigma``); a pure roll about one axis, or two touches, does not and the point is excluded from the
    number and listed (``insufficient_tilt``, with the weakest direction)."""
    opts = opts or GateOptions()
    out: dict = {"min_orientation_deg": opts.min_orientation_deg, "min_tilt_sigma": opts.min_tilt_sigma,
                 "arms": {}, "warnings": []}
    for arm in ARMS:
        groups = _points_by_id(touches, arm)
        per_point, skipped, insufficient = [], [], []
        for pid, ts in sorted(groups.items()):
            if len(ts) < 2:
                continue
            Rs = [t.T_base_tcp[:3, :3] for t in ts]
            ang = max(rotation_angle_deg(Rs[i], Rs[j]) for i in range(len(ts)) for j in range(i + 1, len(ts)))
            if ang < opts.min_orientation_deg:
                skipped.append({"point": pid, "n": len(ts), "max_orientation_deg": float(ang)})
                continue
            obs = tilt_observability(Rs)
            if obs["rank"] < 3 or obs["sigma_min_norm"] < opts.min_tilt_sigma:
                w = obs["weakest_direction_tool"]
                insufficient.append({"point": pid, "n": len(ts), "max_orientation_deg": float(ang),
                                     "sigma_min_norm": obs["sigma_min_norm"], "rank": obs["rank"],
                                     "weakest_direction_tool": w,
                                     "message": f"insufficient tilt diversity for a TCP check at point {pid}"})
                out["warnings"].append(
                    f"{arm}: insufficient tilt diversity for a TCP check at point {pid} ({len(ts)} touches, "
                    f"observability {obs['sigma_min_norm']:.2f} < {opts.min_tilt_sigma:.2f}; TCP direction "
                    f"[{w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f}] in the tool frame is not tested - tilt about two "
                    "different axes, >= 3 touches)")
                continue
            P = np.array([t.T_base_tcp[:3, 3] for t in ts])
            n = len(ts)
            dev = P - P.mean(axis=0)
            spread = float(np.sqrt(np.sum(dev ** 2) / (n - 1))) * 1e3     # unbiased RMS distance from the mean
            pair = max(float(np.linalg.norm(P[i] - P[j])) for i in range(n) for j in range(i + 1, n)) * 1e3
            per_point.append({"point": pid, "n": n, "max_orientation_deg": float(ang), "spread_rms_mm": spread,
                              "max_pair_mm": pair, "sigma_min_norm": obs["sigma_min_norm"],
                              "weakest_direction_tool": obs["weakest_direction_tool"]})
        extra = {"skipped_similar_orientation": skipped, "insufficient_tilt": insufficient}
        if per_point:
            rms = float(np.sqrt(np.mean([p["spread_rms_mm"] ** 2 for p in per_point])))
            weakest = min(per_point, key=lambda p: p["sigma_min_norm"])
            out["arms"][arm] = {"n_points": len(per_point), "spread_rms_mm": rms,
                                "max_pair_mm": max(p["max_pair_mm"] for p in per_point), "per_point": per_point,
                                "weakest_point": weakest["point"], "weakest_sigma_min_norm": weakest["sigma_min_norm"],
                                "weakest_direction_tool": weakest["weakest_direction_tool"], **extra}
            if rms > opts.tcp_warn_mm:
                out["warnings"].append(
                    f"{arm}: TCP consistency {rms:.2f} mm > {opts.tcp_warn_mm:.1f} mm over {len(per_point)} "
                    "point(s): TCP offset error, joint offsets or backlash")
        else:
            out["arms"][arm] = {"n_points": 0, "spread_rms_mm": None, "max_pair_mm": None, "per_point": [], **extra}
    return out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return kabsch(P, Q)


def _reach_point(P_right: np.ndarray, reach_m: float) -> np.ndarray:
    """Point ``reach_m`` from the right base origin towards the touched points (horizontal direction in the base
    frame), where the arm actually works."""
    d = P_right.mean(axis=0).copy()
    d[2] = 0.0
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return reach_m * d


def _diff(Ta: np.ndarray, Tb: np.ndarray, p_reach: np.ndarray) -> dict:
    pa = Ta[:3, :3] @ p_reach + Ta[:3, 3]
    pb = Tb[:3, :3] @ p_reach + Tb[:3, 3]
    rv = np.degrees(R_to_rotvec(Ta[:3, :3] @ Tb[:3, :3].T))        # world-frame rotation difference
    return {"origin_mm": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1e3),
            "reach_mm": float(np.linalg.norm(pa - pb) * 1e3),
            "rotation_deg": float(rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])),
            "rotation_components_deg": [float(v) for v in rv]}


def touch_only_base(touches: Sequence, T_joint: Optional[np.ndarray] = None,
                    opts: Optional[GateOptions] = None) -> dict:
    """Kabsch estimate of T_world_rightbase from the touches (and the comparison with ``T_joint``).

    Uses the board crosshairs AND the free fit stickers (kind ``fit``); check points (kind ``check``) are NEVER
    used.  Returns a dict with ``available`` and, if available: ``T`` (4x4 list), ``n_common``, ``points``,
    ``residual_rms_mm`` (raw; this is a fit residual, FRE, in-sample), ``residual_rms_dof_mm`` (corrected for the 6
    fitted DOF), per-point residuals, ``noise`` (predicted 1-sigma of this base from the geometry and the measured
    repeat scatter), ``comparison`` with the joint estimate (origin / reach point / rotation), the direct arm-to-arm
    mismatch with the joint base, the geometry of the points and ``warnings``."""
    opts = opts or GateOptions()
    out: dict = {"available": False, "warnings": [], "gate_options": vars(opts).copy()}
    touches = [t for t in touches if touch_kind(t) != "check"]
    left, right = _points_by_id(touches, "left"), _points_by_id(touches, "right")
    common = sorted(set(left) & set(right))
    out["n_common"], out["points"] = len(common), common
    kinds = {}
    for t in touches:
        kinds.setdefault(int(t.point_id), touch_kind(t))
    out["point_kinds"] = {int(k): kinds.get(k, "board") for k in common}
    out["points_left_only"] = sorted(set(left) - set(right))
    out["points_right_only"] = sorted(set(right) - set(left))
    if len(common) < 3:
        out["reason"] = f"{len(common)} point(s) touched by both arms; >= 3 non-collinear (4 recommended) needed"
        return out
    Q = np.array([np.mean([t.T_base_tcp[:3, 3] for t in left[k]], axis=0) for k in common])   # world
    P = np.array([np.mean([t.T_base_tcp[:3, 3] for t in right[k]], axis=0) for k in common])  # right base
    sv = np.linalg.svd(Q - Q.mean(axis=0), compute_uv=False)
    if sv[0] < 1e-9 or sv[1] / sv[0] < 0.05:
        out["reason"] = "the common touch points are (nearly) collinear; the base orientation is not determined"
        return out
    T = _kabsch(P, Q)
    res = Q - (P @ T[:3, :3].T + T[:3, 3])
    n = len(common)
    raw = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    dof = 3 * n - 6
    corr = float(np.sqrt(np.sum(res ** 2) / dof)) if dof > 0 else None
    p_reach = _reach_point(P, opts.reach_m)
    spread = np.sqrt(np.sum((Q - Q.mean(axis=0)) ** 2, axis=1).mean())        # RMS radius of the points
    amp = float(opts.reach_m / (spread * math.sqrt(n))) if spread > 1e-9 else float("inf")
    scatter = repeat_scatter(touches, opts.assumed_touch_sigma_mm)
    noise = predict_base_noise(P, Q, [len(left[k]) for k in common], [len(right[k]) for k in common],
                               scatter["left"]["sigma_axis_mm"], scatter["right"]["sigma_axis_mm"], p_reach,
                               n_mc=opts.n_mc)
    noise["scatter"] = scatter
    out.update(
        available=True, T=T.tolist(), residual_rms_mm=raw * 1e3,
        residual_rms_dof_mm=None if corr is None else corr * 1e3,
        residual_label="fit residual (FRE, in-sample)",
        per_point=[{"point": int(k), "residual_mm": float(np.linalg.norm(r) * 1e3)} for k, r in zip(common, res)],
        reach_point_in_right_base=p_reach.tolist(), noise=noise,
        geometry={"rms_radius_mm": float(spread * 1e3), "n_fit_points": n,
                  "principal_extents_mm": [float(v / math.sqrt(n) * 1e3) for v in sv],
                  "yaw_sensitivity_deg_per_mm": float(math.degrees(1e-3 / (spread * math.sqrt(n)))),
                  "reach_amplification": amp,
                  "note": "reach_amplification = error at the reach point per unit of touch-point error "
                          "(yaw lever arm = RMS radius x sqrt(n points)); computed from ALL fit points "
                          "(board crosshairs + fit stickers)"})
    if amp > opts.max_amplification:
        out["warnings"].append(
            f"touch points spread only {spread * 1e3:.0f} mm (RMS radius, {n} points): 1 mm of touch error becomes "
            f"about {amp:.1f} mm at {opts.reach_m:.1f} m reach ({out['geometry']['yaw_sensitivity_deg_per_mm']:.2f} "
            "deg yaw per mm); add fit stickers further apart and repeat each touch")
    if T_joint is not None:
        Tj = np.asarray(T_joint, dtype=float)
        d = _diff(Tj, T, p_reach)
        d["note"] = "joint estimate minus touch-only estimate (both with nominal FK)"
        out["comparison"] = d
        mism = Q - (P @ Tj[:3, :3].T + Tj[:3, 3])
        out["direct_mismatch_with_joint_base"] = {
            "rms_mm": float(np.sqrt(np.mean(np.sum(mism ** 2, axis=1))) * 1e3),
            "label": "fit residual (FRE, in-sample): the board / fit points were used to determine the bases",
            "per_point": [{"point": int(k), "mismatch_mm": float(np.linalg.norm(m) * 1e3)}
                          for k, m in zip(common, mism)]}
    return out


def pose_xyz_rpy(T) -> dict:
    T = np.asarray(T, dtype=float)
    return {"xyz": [float(v) for v in T[:3, 3]], "rpy": [float(v) for v in R_to_rpy(T[:3, :3])]}


def geometry_adequate(tb: dict, opts: GateOptions) -> Tuple[bool, str]:
    """Predicted 1-sigma of the touch-only base small enough?  (yaw <= max_yaw_sigma_deg, reach point <= max_reach_sigma_mm)"""
    nz = tb.get("noise") if tb.get("available") else None
    if not nz:
        return False, "no touch-only base estimate"
    ok = nz["yaw_sigma_deg"] <= opts.max_yaw_sigma_deg and nz["reach_sigma_mm"] <= opts.max_reach_sigma_mm
    txt = (f"predicted 1-sigma of the touch-only base: yaw {nz['yaw_sigma_deg']:.3f} deg (<= "
           f"{opts.max_yaw_sigma_deg:.2f}), reach point {nz['reach_sigma_mm']:.2f} mm (<= {opts.max_reach_sigma_mm:.1f}), "
           f"origin {nz['origin_sigma_mm']:.2f} mm; per-axis repeat scatter L "
           f"{nz['sigma_left_axis_mm']:.2f}{'*' if nz['scatter']['left']['assumed'] else ''} / R "
           f"{nz['sigma_right_axis_mm']:.2f}{'*' if nz['scatter']['right']['assumed'] else ''} mm")
    if not ok:
        txt += " -> touch spread insufficient: add fit stickers further apart (and repeat each touch)"
    return ok, txt


def _written_std(report: dict) -> Tuple[np.ndarray, np.ndarray]:
    wb = report.get("written_base") or {}
    j = report.get("joint", {}).get("arm_right_base", {})
    xyz = wb.get("std_xyz_mm") or j.get("std_xyz_mm") or [0.0, 0.0, 0.0]
    rpy = wb.get("std_rpy_deg") or j.get("std_rpy_deg") or [0.0, 0.0, 0.0]
    return np.asarray(xyz, float), np.asarray(rpy, float)


def _agreement(tb: dict, report: dict, opts: GateOptions) -> Tuple[bool, dict, str]:
    c, nz = tb.get("comparison"), tb.get("noise")
    sx, sr = _written_std(report)
    L = opts.reach_m
    k = opts.agree_sigma_k
    sig_o = math.hypot(nz["origin_sigma_mm"], float(np.linalg.norm(sx)))
    sig_r = math.hypot(nz["reach_sigma_mm"], math.hypot(float(np.linalg.norm(sx)), L * 1e3 * math.radians(float(np.linalg.norm(sr)))))
    sig_t = [math.hypot(a, b) for a, b in zip(nz["rotation_sigma_deg"], sr)]
    lim_o, lim_r = max(opts.max_joint_vs_touch_mm, k * sig_o), max(opts.max_joint_vs_touch_mm, k * sig_r)
    lim_t = [max(opts.max_joint_vs_touch_deg, k * s) for s in sig_t]
    rc = c["rotation_components_deg"]
    ok = c["origin_mm"] <= lim_o and c["reach_mm"] <= lim_r and all(abs(a) <= b for a, b in zip(rc, lim_t))
    val = {"origin_mm": c["origin_mm"], "reach_mm": c["reach_mm"], "rotation_components_deg": rc,
           "limit_origin_mm": lim_o, "limit_reach_mm": lim_r, "limit_rotation_deg": lim_t,
           "sigma_origin_mm": sig_o, "sigma_reach_mm": sig_r, "sigma_rotation_deg": sig_t}
    txt = (f"|joint - touch-only| <= max(floor {opts.max_joint_vs_touch_mm:.1f} mm / {opts.max_joint_vs_touch_deg:.2f} "
           f"deg, {k:.0f} sigma): origin {c['origin_mm']:.2f} mm (limit {lim_o:.2f}), reach point "
           f"{c['reach_mm']:.2f} mm (limit {lim_r:.2f}), rotation x/y/z "
           + "/".join(f"{abs(a):.3f}" for a in rc) + " deg (limits " + "/".join(f"{b:.3f}" for b in lim_t) + ")")
    return ok, val, txt


def _heldout(report: dict, opts: GateOptions) -> Tuple[bool, dict, str]:
    val = report.get("validation")
    if not val:
        return False, {"run": False}, "held-out validation not run (use --validate / provide the URDF)"
    fit = (report.get("written_base") or {}).get("fit", "nominal_fit")
    variant = "with_offsets" if fit == "offset_fit" else "without_offsets"
    v = val.get(variant) or {}
    px = v.get("heldout_rms_px")
    if px is None:
        return False, {"run": True, "variant": variant}, f"held-out validation ({variant}) produced no reprojection value"
    touch = v.get("heldout_touch_rms_nominal_mm", v.get("heldout_touch_rms_mm") if variant == "without_offsets" else None)
    ok_px = px <= opts.max_heldout_px
    ok_t = touch is None or touch <= opts.max_touch_rms_mm
    txt = (f"held-out ({variant}, the variant that is written): reprojection RMS {px:.3f} px (<= {opts.max_heldout_px:.2f})"
           + (f", held-out touch error {touch:.2f} mm nominal FK (<= {opts.max_touch_rms_mm:.2f})" if touch is not None
              else "; no held-out touches (not required)"))
    return ok_px and ok_t, {"variant": variant, "heldout_rms_px": px, "heldout_touch_nominal_mm": touch}, txt


def _intrinsics(report: dict, opts: GateOptions) -> Tuple[bool, dict, str]:
    src = (report.get("options") or {}).get("intrinsics_source", "camera_info")
    ir = (report.get("joint") or {}).get("intrinsics") or {}
    pf, pc = ir.get("pull_f_frac"), ir.get("pull_c_px")
    val = {"source": src, "pull_f_frac": pf, "pull_c_px": pc}
    if src == "file":
        extra = "" if pf is None else f" (data pull {100 * pf:.2f} % in f, {pc:.1f} px in c, informational)"
        return True, val, "intrinsics validated: --intrinsics-file supplied (overrides camera_info)" + extra
    if pf is None or pc is None:
        return False, val, ("intrinsics pull not computed (" + str(ir.get("error", "not available")) +
                            "): calibrate the camera intrinsics")
    ok = pf <= opts.max_intrinsics_f_frac and pc <= opts.max_intrinsics_c_px
    txt = (f"intrinsics validated: the data would pull f by {100 * pf:.2f} % (<= {100 * opts.max_intrinsics_f_frac:.1f} %) "
           f"and c by {pc:.1f} px (<= {opts.max_intrinsics_c_px:.0f})")
    if not ok:
        txt += " -> calibrate the camera intrinsics (calibrate_intrinsics) and pass --intrinsics-file"
    return ok, val, txt


def evaluate_gate(report: dict, opts: Optional[GateOptions] = None, base_from: str = "joint") -> dict:
    """PASS/FAIL gate for using arm_right's base in collision avoidance.  Reads ``report`` only.

    Every check evaluates VALUES against thresholds.  The in-sample fit residual (FRE) is reported and must be small,
    but it can never satisfy the gate on its own: the held-out check points (TRE), the held-out validation, the
    noise-aware agreement of two independent bases, the touch geometry and the intrinsics have to pass as well."""
    opts = opts or GateOptions()
    tb = report.get("touch_base") or {"available": False}
    lc = report.get("loop_closure")
    cp = report.get("check_points") or {"available": False, "n_points": 0, "points": []}
    checks: List[dict] = []

    # 1. common touch points (board crosshairs + fit stickers)
    n = int(tb.get("n_common", 0))
    checks.append({"name": "common_touch_points", "ok": n >= opts.min_common_points and bool(tb.get("available")),
                   "value": n, "threshold": opts.min_common_points,
                   "text": f">= {opts.min_common_points} fit points (board crosshairs + fit stickers) common to both arms "
                           f"(non-collinear): {n}" + ("" if tb.get("available") else f" ({tb.get('reason', 'no estimate')})")})

    # 2. fit residual (FRE, in-sample) and the out-of-fit loop closure; in-fit numbers alone never satisfy the gate
    vals, parts = [], []
    if tb.get("available"):
        if base_from == "touches":
            v = tb.get("residual_rms_dof_mm")
            v = tb["residual_rms_mm"] if v is None else v
            vals.append(v)
            parts.append(f"touch-only Kabsch fit residual (FRE, in-sample) {v:.2f} mm (dof-corrected)")
        elif tb.get("direct_mismatch_with_joint_base"):
            v = tb["direct_mismatch_with_joint_base"]["rms_mm"]
            vals.append(v)
            parts.append(f"fit residual (FRE, in-sample) arm-to-arm mismatch with the written base {v:.2f} mm")
    oof = lc.get("out_of_fit_rms_mm") if lc else None
    if oof is not None:
        vals.append(float(oof))
        parts.append(f"out-of-fit loop closure {oof:.2f} mm ({lc.get('out_of_fit_source')})")
    elif lc:
        parts.append("no out-of-fit loop-closure number (only in-fit data: it can never satisfy the gate)")
    worst = max(vals) if vals else None
    checks.append({"name": "touch_mismatch_rms", "ok": worst is not None and worst <= opts.max_touch_rms_mm and oof is not None,
                   "value": worst, "threshold": opts.max_touch_rms_mm,
                   "text": f"touch mismatch RMS <= {opts.max_touch_rms_mm:.2f} mm (nominal FK, worst of; needs an "
                           "out-of-fit number): " + ("; ".join(parts) if parts else "not available")})

    # 3. held-out check points (TRE)
    cp_pts = cp.get("points") or []
    n_cp = len(cp_pts)
    have_geo = cp.get("reference") is not None
    h_ok = have_geo and any(p.get("height_above_plane_mm", 0) >= opts.check_min_height_mm for p in cp_pts)
    d_ok = have_geo and any(p.get("dist_from_centre_mm", 0) >= opts.check_min_dist_mm for p in cp_pts)
    cov_ok = n_cp >= opts.min_check_points and h_ok and d_ok
    cov_txt = (f">= {opts.min_check_points} check points touched by both arms (never used in a fit), one >= "
               f"{opts.check_min_height_mm:.0f} mm off the table-board plane, one >= {opts.check_min_dist_mm:.0f} mm "
               f"from the board centre: {n_cp} point(s)")
    if n_cp and have_geo:
        cov_txt += (f", max height {cp.get('max_height_above_plane_mm', 0):.0f} mm, max distance "
                    f"{cp.get('max_dist_from_centre_mm', 0):.0f} mm")
    elif n_cp:
        cov_txt += " (board plane / centre not available: touch >= 3 board or fit points)"
    elif cp.get("incomplete"):
        cov_txt += " (touched by one arm only: " + ", ".join(str(i["point"]) for i in cp["incomplete"]) + ")"
    checks.append({"name": "check_point_coverage", "ok": cov_ok,
                   "value": {"n": n_cp, "max_height_mm": cp.get("max_height_above_plane_mm"),
                             "max_dist_mm": cp.get("max_dist_from_centre_mm")},
                   "threshold": {"n": opts.min_check_points, "height_mm": opts.check_min_height_mm,
                                 "dist_mm": opts.check_min_dist_mm}, "text": cov_txt})
    maxe = cp.get("max_mm")
    checks.append({"name": "check_point_error", "ok": maxe is not None and maxe <= opts.max_check_error_mm,
                   "value": maxe, "threshold": opts.max_check_error_mm,
                   "text": f"check-point error (TRE, held-out) max <= {opts.max_check_error_mm:.1f} mm: "
                           + (f"max {maxe:.2f} mm, RMS {cp['rms_mm']:.2f} mm over {n_cp} point(s) ("
                              + ", ".join(f"{p['point']}: {p['error_mm']:.2f}" for p in cp_pts) + ")"
                              if maxe is not None else "no check-point data")})

    # 4. geometry of the fit points: predicted noise of the touch-only base
    g_ok, g_txt = geometry_adequate(tb, opts)
    nz = tb.get("noise") or {}
    checks.append({"name": "touch_geometry", "ok": g_ok,
                   "value": {"yaw_sigma_deg": nz.get("yaw_sigma_deg"), "reach_sigma_mm": nz.get("reach_sigma_mm")},
                   "threshold": {"yaw_sigma_deg": opts.max_yaw_sigma_deg, "reach_sigma_mm": opts.max_reach_sigma_mm},
                   "text": g_txt})

    # 5. noise-aware agreement of the two independent bases
    if tb.get("available") and tb.get("comparison") and tb.get("noise"):
        a_ok, a_val, a_txt = _agreement(tb, report, opts)
        checks.append({"name": "joint_vs_touch", "ok": a_ok, "value": a_val,
                       "threshold": {"floor_mm": opts.max_joint_vs_touch_mm, "floor_deg": opts.max_joint_vs_touch_deg,
                                     "sigma_k": opts.agree_sigma_k}, "text": a_txt})
    else:
        checks.append({"name": "joint_vs_touch", "ok": False, "value": None, "threshold": None,
                       "text": "joint vs touch-only base agreement: no touch-only estimate"})

    # 6. held-out validation: values, not "it ran"
    ho_ok, ho_val, ho_txt = _heldout(report, opts)
    checks.append({"name": "heldout_validation", "ok": ho_ok, "value": ho_val,
                   "threshold": {"px": opts.max_heldout_px, "touch_mm": opts.max_touch_rms_mm}, "text": ho_txt})

    # 7. intrinsics
    i_ok, i_val, i_txt = _intrinsics(report, opts)
    checks.append({"name": "intrinsics_validated", "ok": i_ok, "value": i_val,
                   "threshold": {"f_frac": opts.max_intrinsics_f_frac, "c_px": opts.max_intrinsics_c_px}, "text": i_txt})

    passed = all(c["ok"] for c in checks)
    by = {c["name"]: c for c in checks}
    padding = recommended_padding_mm(maxe)
    wfit = (report.get("written_base") or {}).get("fit")
    wtxt = ("touch-only Kabsch" if base_from == "touches" else
            {"offset_fit": "offset fit (joint-1 absorbed)", "nominal_fit": "nominal fit"}.get(wfit, "joint (camera chain)"))
    lines = [f"TRUST GATE for using arm_right's base in collision avoidance: {'PASS' if passed else 'FAIL'}"
             f"   (base written: {wtxt})"]
    lines += [f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['text']}" for c in checks]
    lines.append("  scope: accuracy demonstrated at the check points only; a PASS does not certify the accuracy "
                 "elsewhere in the workspace")
    lines.append(f"  recommended_padding_mm = max(20, 2 x max check-point error + 10) = {padding:.0f}"
                 + ("" if maxe is not None else "  (no check points: the 20 mm minimum only)"))
    if not passed:
        lines.append("  -> YAML source is 'calibrated_unverified'; do not rely on this base for collision "
                     "avoidance between the arms until the failing checks are fixed")
        if not by["joint_vs_touch"]["ok"] and tb.get("available"):
            tc = [v["spread_rms_mm"] for v in (tb.get("tcp_consistency") or {}).get("arms", {}).values()
                  if v.get("spread_rms_mm") is not None]
            if tc and max(tc) <= opts.tcp_warn_mm and g_ok:
                lines.append(f"  -> joint vs touch-only disagree while TCP consistency is good ({max(tc):.2f} mm) and the "
                             "touch geometry is adequate: the camera chain is the suspect; the touch-only base is "
                             "preferred (--base-from touches)")
            elif tc:
                lines.append(f"  -> joint vs touch-only disagree and TCP consistency is {max(tc):.2f} mm / the touch "
                             "geometry amplifies noise: the arms' kinematics (joint offsets, TCP) are suspect and "
                             "the touch-only base inherits their error; neither base is trustworthy yet")
            else:
                lines.append("  -> joint vs touch-only disagree and TCP consistency was not measured (touch one "
                             "point >= 3 times with the tool tilted about two different axes): the touch-only base is preferred only if the arms' "
                             "kinematics and the touch geometry are known to be good")
    return {"passed": passed, "source": "calibrated" if passed else "calibrated_unverified", "base_from": base_from,
            "checks": checks, "text": lines, "recommended_padding_mm": padding,
            "scope": "accuracy demonstrated at the check points only",
            "summary": ("PASS" if passed else "FAIL: " + ", ".join(c["name"] for c in checks if not c["ok"]))}
