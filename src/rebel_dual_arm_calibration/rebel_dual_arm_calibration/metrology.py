"""Metrology of the touch data (numpy only, no ROS): held-out check points (TRE), noise-aware prediction of the
touch-only base, repeatability (ISO 9283-inspired) and comparison of two calibrations.

Vocabulary (Fitzpatrick/West): FRE = fiducial registration error = the residual of the points that were used in the
fit (in-sample, optimistic, it can be small for a bad registration).  TRE = target registration error = the error at
points that were NOT used in any fit; only TRE says something about the accuracy somewhere else.  The check points of
this module are the TRE measurement: ``||T_world_left p_left - T_world_right(base) p_right||`` with the WRITTEN base
and the NOMINAL forward kinematics (the model the deployed robot runs).  What it demonstrates is the agreement of the
two arms AT the check points only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

from .transforms import R_to_rotvec, kabsch_batch, rotation_angle_deg, xyz_rpy_to_T

ARMS = ("left", "right")


def touch_kind(t) -> str:
    return getattr(t, "kind", "board") or "board"


def group_by_id(touches: Sequence, arm: str) -> Dict[int, list]:
    out: Dict[int, list] = {}
    for t in touches:
        if t.arm == arm:
            out.setdefault(int(t.point_id), []).append(t)
    return out


def _mean_pos(ts) -> np.ndarray:
    return np.mean([t.T_base_tcp[:3, 3] for t in ts], axis=0)


def repeat_scatter(touches: Sequence, assumed_sigma_mm: float = 0.5) -> dict:
    """Measured repeat scatter of the TCP position per arm: pooled 1-sigma PER AXIS (mm) over all (arm, point) groups
    with >= 2 touches (the repeats of one point at different wrist configurations, so it includes the kinematic
    scatter that a real arm has).  An arm without repeats gets ``assumed_sigma_mm`` and ``assumed: True``."""
    out: dict = {}
    for arm in ARMS:
        ss, dof = 0.0, 0
        for ts in group_by_id(touches, arm).values():
            if len(ts) < 2:
                continue
            P = np.array([t.T_base_tcp[:3, 3] for t in ts])
            ss += float(np.sum((P - P.mean(axis=0)) ** 2))
            dof += 3 * (len(ts) - 1)
        if dof > 0:
            out[arm] = {"sigma_axis_mm": max(1e3 * math.sqrt(ss / dof), 0.02), "dof": dof, "assumed": False}
        else:
            out[arm] = {"sigma_axis_mm": float(assumed_sigma_mm), "dof": 0, "assumed": True}
    return out


# --------------------------------------------------------------------------
# Reference geometry of the board (plane and centre, measured from the touched positions)
# --------------------------------------------------------------------------
def _world_means(touches: Sequence, T_W_R: np.ndarray) -> Dict[int, np.ndarray]:
    """Per point id: mean world position over the arms that touched it (right mapped with ``T_W_R``)."""
    left, right = group_by_id(touches, "left"), group_by_id(touches, "right")
    out = {}
    for pid in sorted(set(left) | set(right)):
        pts = []
        if pid in left:
            pts.append(_mean_pos(left[pid]))
        if pid in right:
            pts.append(T_W_R[:3, :3] @ _mean_pos(right[pid]) + T_W_R[:3, 3])
        out[pid] = np.mean(pts, axis=0)
    return out


def reference_plane(touches: Sequence, T_W_R: np.ndarray) -> Optional[dict]:
    """Plane (unit normal, centroid) through the touched positions of the table-board crosshairs (kind ``board``);
    if there are none (e.g. a check-only session) through the fit stickers.  None if < 3 distinct points or collinear."""
    for kinds in (("board",), ("fit",)):
        sub = [t for t in touches if touch_kind(t) in kinds]
        pts = np.array(list(_world_means(sub, T_W_R).values()))
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        sv = np.linalg.svd(pts - c, compute_uv=True)
        if sv[1][1] < 1e-4:          # collinear
            continue
        n = sv[2][2]
        return {"normal": n.tolist(), "centre": c.tolist(), "source": kinds[0], "n_points": len(pts)}
    return None


# --------------------------------------------------------------------------
# Check points (TRE, held out)
# --------------------------------------------------------------------------
def check_point_report(touches: Sequence, T_W_R: np.ndarray) -> dict:
    """Arm-to-arm disagreement at the check points (never used in any fit), written base + nominal FK.

    ``touches`` may contain every touch of the session; only kind ``check`` is evaluated (the board / fit touches
    only define the reference plane and centre for the coverage numbers)."""
    T_W_R = np.asarray(T_W_R, dtype=float)
    chk = [t for t in touches if touch_kind(t) == "check"]
    left, right = group_by_id(chk, "left"), group_by_id(chk, "right")
    ref = reference_plane([t for t in touches if touch_kind(t) != "check"], T_W_R)
    pts, incomplete = [], []
    for pid in sorted(set(left) | set(right)):
        if pid not in left or pid not in right:
            incomplete.append({"point": pid, "touched_by": "left" if pid in left else "right"})
            continue
        pl = _mean_pos(left[pid])
        pr = T_W_R[:3, :3] @ _mean_pos(right[pid]) + T_W_R[:3, 3]
        d = pl - pr
        rec = {"point": pid, "n_left": len(left[pid]), "n_right": len(right[pid]),
               "error_mm": float(np.linalg.norm(d) * 1e3), "vector_mm": (d * 1e3).tolist(),
               "position_world": ((pl + pr) / 2.0).tolist(),
               "scatter_left_mm": _scatter3(left[pid]), "scatter_right_mm": _scatter3(right[pid])}
        if ref is not None:
            p = (pl + pr) / 2.0
            rec["height_above_plane_mm"] = float(abs(np.dot(np.array(ref["normal"]), p - np.array(ref["centre"]))) * 1e3)
            rec["dist_from_centre_mm"] = float(np.linalg.norm(p - np.array(ref["centre"])) * 1e3)
        pts.append(rec)
    out: dict = {"available": bool(pts), "n_points": len(pts), "points": pts, "incomplete": incomplete,
                 "reference": ref,
                 "label": "check-point error (TRE, held-out): arm-to-arm disagreement with the written base and "
                          "NOMINAL FK at points not used in any fit",
                 "scope": "accuracy demonstrated at the check points only"}
    if pts:
        e = np.array([p["error_mm"] for p in pts])
        out["rms_mm"] = float(np.sqrt(np.mean(e ** 2)))
        out["max_mm"] = float(e.max())
        out["recommended_padding_mm"] = recommended_padding_mm(out["max_mm"])
        if ref is not None:
            out["max_height_above_plane_mm"] = float(max(p["height_above_plane_mm"] for p in pts))
            out["max_dist_from_centre_mm"] = float(max(p["dist_from_centre_mm"] for p in pts))
    else:
        out["recommended_padding_mm"] = recommended_padding_mm(None)
    return out


def recommended_padding_mm(max_check_error_mm: Optional[float]) -> float:
    """``max(20, 2 x max check-point error + 10)``; without a check-point error only the 20 mm minimum is known."""
    if max_check_error_mm is None:
        return 20.0
    return float(max(20.0, 2.0 * max_check_error_mm + 10.0))


def _scatter3(ts) -> Optional[float]:
    if len(ts) < 2:
        return None
    P = np.array([t.T_base_tcp[:3, 3] for t in ts])
    return float(np.sqrt(np.sum((P - P.mean(axis=0)) ** 2) / (len(ts) - 1)) * 1e3)


# --------------------------------------------------------------------------
# Noise prediction of the touch-only (Kabsch) base
# --------------------------------------------------------------------------
def predict_base_noise(P_right: np.ndarray, Q_world: np.ndarray, n_left: Sequence[int], n_right: Sequence[int],
                       sigma_left_mm: float, sigma_right_mm: float, p_reach: np.ndarray, n_mc: int = 400,
                       seed: int = 20261008) -> dict:
    """1-sigma of the touch-only base from the fit-point geometry and the MEASURED repeat scatter of both arms.

    Small Monte Carlo of the Kabsch fit (the nonlinear version of the Fitzpatrick/West expression): the averaged
    point positions get independent Gaussian noise of (per-axis scatter / sqrt(repeats)), the points are
    re-registered and the spread of the result is taken.  Rotation errors are rotation vectors in the WORLD frame
    (z = yaw); the origin and the reach point are the right base origin and ``p_reach`` (right base frame) mapped to
    the world.  Deterministic (seeded)."""
    rng = np.random.default_rng(seed)
    P, Q = np.asarray(P_right, float), np.asarray(Q_world, float)
    nl, nr = np.maximum(np.asarray(n_left, float), 1.0), np.maximum(np.asarray(n_right, float), 1.0)
    sL = (sigma_left_mm * 1e-3 / np.sqrt(nl))[None, :, None]
    sR = (sigma_right_mm * 1e-3 / np.sqrt(nr))[None, :, None]
    Pt = P[None] + rng.normal(size=(n_mc,) + P.shape) * sR
    Qt = Q[None] + rng.normal(size=(n_mc,) + Q.shape) * sL
    Ts = kabsch_batch(Pt, Qt)
    T0 = kabsch_batch(P[None], Q[None])[0]
    dR = np.einsum("bij,jk->bik", Ts[:, :3, :3], T0[:3, :3].T)             # world-frame rotation error
    rv = np.array([R_to_rotvec(r) for r in dR])                            # (B, 3) rad
    d_org = Ts[:, :3, 3] - T0[:3, 3]
    reach = np.einsum("bij,j->bi", Ts[:, :3, :3], p_reach) + Ts[:, :3, 3]
    d_reach = reach - (T0[:3, :3] @ p_reach + T0[:3, 3])
    return {"method": f"Monte Carlo of the Kabsch fit ({n_mc} draws) from the fit-point geometry and the measured "
                      "repeat scatter",
            "n_mc": n_mc, "sigma_left_axis_mm": float(sigma_left_mm), "sigma_right_axis_mm": float(sigma_right_mm),
            "rotation_sigma_deg": [float(v) for v in np.degrees(rv.std(axis=0))],      # world x, y, z
            "yaw_sigma_deg": float(np.degrees(rv[:, 2].std())),
            "origin_sigma_mm": float(np.sqrt(np.sum(d_org.var(axis=0))) * 1e3),
            "origin_sigma_axis_mm": [float(v) for v in d_org.std(axis=0) * 1e3],
            "reach_sigma_mm": float(np.sqrt(np.sum(d_reach.var(axis=0))) * 1e3),
            "reach_sigma_axis_mm": [float(v) for v in d_reach.std(axis=0) * 1e3]}


# --------------------------------------------------------------------------
# Repeatability (ISO 9283-inspired)
# --------------------------------------------------------------------------
def repeatability(touches: Sequence, max_orientation_deg: float = 5.0, min_repeats: int = 5) -> dict:
    """RP per (arm, point): touches of the same id by the same arm at nearly the same wrist orientation (every
    member within ``max_orientation_deg`` of every other) and >= ``min_repeats`` repeats.
    RP = l_mean + 3 S_l with l_j = distance of touch j to the barycentre (ISO 9283 position repeatability).
    ISO 9283-INSPIRED only: the touches are made by an operator with a jog, not by a programmed approach."""
    out: dict = {"max_orientation_deg": max_orientation_deg, "min_repeats": min_repeats, "clusters": [],
                 "skipped": [], "note": "ISO 9283-inspired (RP = mean distance to barycentre + 3 sigma); the standard "
                                        "prescribes programmed repeated approaches, not operator touches"}
    for arm in ARMS:
        for pid, ts in sorted(group_by_id(touches, arm).items()):
            clusters: List[list] = []
            for t in ts:
                for c in clusters:
                    if all(rotation_angle_deg(t.T_base_tcp[:3, :3], m.T_base_tcp[:3, :3]) < max_orientation_deg
                           for m in c):
                        c.append(t)
                        break
                else:
                    clusters.append([t])
            for c in clusters:
                if len(c) < min_repeats:
                    if len(c) >= 2:
                        out["skipped"].append({"arm": arm, "point": pid, "n": len(c)})
                    continue
                P = np.array([t.T_base_tcp[:3, 3] for t in c])
                l = np.linalg.norm(P - P.mean(axis=0), axis=1)
                lm, sl = float(l.mean()), float(l.std(ddof=1))
                out["clusters"].append({"arm": arm, "point": pid, "kind": touch_kind(c[0]), "n": len(c),
                                        "l_mean_mm": lm * 1e3, "s_l_mm": sl * 1e3, "rp_mm": (lm + 3 * sl) * 1e3})
    out["max_rp_mm"] = max((c["rp_mm"] for c in out["clusters"]), default=None)
    return out


# --------------------------------------------------------------------------
# Calibration YAML I/O and comparison
# --------------------------------------------------------------------------
def load_calibration_yaml(path: str) -> dict:
    """{'T_R': 4x4, 'T_cam': 4x4, 'source': str, 'date': str} from a ``dual_arm_calibration.yaml``."""
    with open(path) as f:
        d = yaml.safe_load(f)["dual_arm_calibration"]
    return {"T_R": xyz_rpy_to_T(d["arm_right_base"]["xyz"], d["arm_right_base"]["rpy"]),
            "T_cam": xyz_rpy_to_T(d["camera_link"]["xyz"], d["camera_link"]["rpy"]),
            "source": d.get("source", ""), "date": str(d.get("date", "")), "path": path}


def compare_poses(T_new: np.ndarray, T_old: np.ndarray) -> dict:
    return {"position_mm": float(np.linalg.norm(T_new[:3, 3] - T_old[:3, 3]) * 1e3),
            "rotation_deg": float(rotation_angle_deg(T_new[:3, :3], T_old[:3, :3])),
            "delta_xyz_mm": [float(v) for v in (T_new[:3, 3] - T_old[:3, 3]) * 1e3]}


def compare_calibrations(T_R_new, T_cam_new, previous: dict, accept_mm: float = 2.0, accept_deg: float = 0.2) -> dict:
    """Difference of the written calibration to a previous one (reproducibility check)."""
    r = compare_poses(T_R_new, previous["T_R"])
    c = compare_poses(T_cam_new, previous["T_cam"])
    ok = max(r["position_mm"], c["position_mm"]) < accept_mm and max(r["rotation_deg"], c["rotation_deg"]) < accept_deg
    return {"previous": previous.get("path", ""), "previous_date": previous.get("date", ""),
            "arm_right_base": r, "camera_link": c, "accept_mm": accept_mm, "accept_deg": accept_deg,
            "within_guidance": bool(ok),
            "text": (f"vs {previous.get('path', 'previous')} ({previous.get('date', '?')}): arm_right base "
                     f"{r['position_mm']:.2f} mm / {r['rotation_deg']:.3f} deg, camera "
                     f"{c['position_mm']:.2f} mm / {c['rotation_deg']:.3f} deg -> "
                     + ("within" if ok else "EXCEEDS") + f" the reproducibility guidance ({accept_mm:.1f} mm / "
                     f"{accept_deg:.1f} deg)")}
