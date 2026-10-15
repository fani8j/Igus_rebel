"""Pivot calibration of the knife TCP (pure Python, numpy + yaml only, no ROS).

Model.  The knife tip rests in a fixed pivot ``c`` (base frame).  For every
sample the flange-side frame ``igus_tool0`` has pose ``(R_i, p_i)`` in the base
frame, and ``p_i + R_i r = c`` where ``r`` is the TCP in ``igus_tool0`` (the
origin xyz of the URDF joint ``igus_tool0_to_xeg32_tool_tip``).  Stacking
``[R_i, -I] [r; c] = -p_i`` gives a linear least-squares problem solved with an
SVD (``np.linalg.lstsq``).

What the result means (and does not mean).  Nominal-FK errors (joint zero
offsets, link lengths) are absorbed into ``r``: ``r`` is the *effective* tip
position for the poses that were measured, accurate to about 1-2 mm for
tilts up to +-35 deg, not an independent measurement of the blade geometry.
The residuals therefore measure tip slip plus un-absorbed FK error, and the
held-out samples (never used in the fit, fixed ``(r, c)``) are the honest test.

Observability.  Eliminating the pivot gives ``B_i = R_i - mean(R)``; ``r`` is
determined by ``B`` alone.  ``sigma_min(B)/sqrt(N)`` (``obs_min``) tells how
well the tilts separate ``r`` from ``c``; rank < 3 means all rotations share an
axis (the tool direction along that axis cannot be found).  The conditional
covariance of ``r`` is ``sigma^2 (B^T B)^-1`` with ``sigma^2 = SSE/(3N-6)``.

The module never writes the URDF: ``xacro_patch`` only returns a diff.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------- constants

#: URDF joint ``igus_tool0_to_xeg32_tool_tip`` origin xyz at the time of writing.
CURRENT_URDF_R_M = np.array([0.005937, 0.094121, 0.130699])
#: link6 -> igus_tool0 translation along link6 z (URDF: xyz 0 0 0.0527, rpy 0 0 pi).
LINK6_TO_TOOL0_Z_MM = 52.7
TCP_JOINT_NAME = "igus_tool0_to_xeg32_tool_tip"
RESULT_YAML_KIND = "rebel_tcp_pivot_calibration/v1"

_ORTHONORMAL_TOL = 1e-3
_RANK_REL_TOL = 1e-6


# ---------------------------------------------------------------- data classes

def _jsonable(v):
    """Convert numpy / nan / inf into plain JSON-safe python (nan/inf -> None)."""
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class PivotSample:
    """One captured pose with the tip resting in the pivot."""

    arm: str
    T_base_tool0: np.ndarray                      # 4x4 base -> igus_tool0
    T_base_link6: Optional[np.ndarray] = None
    joints: Dict[str, float] = field(default_factory=dict)
    stamp: float = 0.0
    role: str = "fit"                             # "fit" | "validate"
    note: str = ""

    def to_json_dict(self) -> dict:
        return {
            "arm": self.arm,
            "T_base_tool0": _jsonable(np.asarray(self.T_base_tool0, dtype=float)),
            "T_base_link6": (None if self.T_base_link6 is None
                             else _jsonable(np.asarray(self.T_base_link6, dtype=float))),
            "joints": {str(k): float(v) for k, v in self.joints.items()},
            "stamp": float(self.stamp),
            "role": self.role,
            "note": self.note,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "PivotSample":
        t6 = d.get("T_base_link6")
        role = d.get("role", "fit")
        if role not in ("fit", "validate"):
            raise ValueError(f"sample role must be 'fit' or 'validate', got {role!r}")
        return PivotSample(
            arm=str(d.get("arm", "")),
            T_base_tool0=np.array(d["T_base_tool0"], dtype=float),
            T_base_link6=None if t6 is None else np.array(t6, dtype=float),
            joints={str(k): float(v) for k, v in (d.get("joints") or {}).items()},
            stamp=float(d.get("stamp", 0.0)),
            role=role,
            note=str(d.get("note", "")),
        )


@dataclass
class PivotOptions:
    """Acceptance thresholds.  Defaults are the lead's decisions (mm unless noted)."""

    min_fit: int = 8                    # fewer fit poses -> INSUFFICIENT
    min_validate: int = 3               # fewer held-out poses -> INSUFFICIENT (no verdict without them)
    rms_pass_mm: float = 2.0            # fit residual RMS must be <= this for PASS
    max_pass_mm: float = 3.5            # worst fit residual must be <= this for PASS
    heldout_rms_pass_mm: float = 2.5    # RMS over validate samples with fixed (r, c)
    good_rms_mm: float = 1.0            # GOOD needs fit RMS <= this ...
    good_max_mm: float = 2.0            # ... and worst residual <= this (and no warnings)
    axis_sigma_pass_mm: float = 1.0     # 1-sigma of r along its worst principal direction
    obs_min: float = 0.2                # sigma_min(B)/sqrt(N); below -> geometry too weak
    dup_min_deg: float = 15.0           # orientations closer than this are duplicates
    outlier_factor: float = 2.0         # LOO error > factor x median(LOO) -> flagged (never dropped)
    outlier_min_mm: float = 0.5         # EXTRA (not in the spec): absolute floor so that exact /
                                        # sub-0.5 mm data never produces spurious outliers


@dataclass
class PivotResult:
    ok: bool
    verdict: str                        # "GOOD" | "PASS" | "FAIL" | "INSUFFICIENT"
    reasons: List[str]
    r_tool0: np.ndarray
    c_base: np.ndarray
    n_fit: int
    n_validate: int
    residuals_mm: List[float]
    rms_mm: float
    max_mm: float
    loo_mm: List[float]
    loo_rms_mm: float
    heldout_mm: List[float]
    heldout_rms_mm: Optional[float]
    cov_r: np.ndarray                   # m^2
    sigma_axes_mm: np.ndarray           # principal 1-sigma, descending
    weak_axis_tool0: np.ndarray         # unit vector (tool0) of the worst-determined direction
    obs_min: float
    cond_B: float
    rank: int
    outliers: List[int]                 # indices into the fit list
    tilt_deg: List[float]               # per fit sample, tool-axis tilt from the mean orientation's axis
    roll_deg: List[float]               # per fit sample, roll about the tool axis relative to the mean
    delta_to_current_mm: Optional[np.ndarray]

    def to_json_dict(self) -> dict:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}


# ---------------------------------------------------------------- rotation helpers

def _rot3(T) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, :3]


def rotation_distance_deg(Ta, Tb) -> float:
    """Angle (deg) of the relative rotation between two 4x4 (or 3x3) poses."""
    Ra = np.asarray(Ta, dtype=float)[:3, :3]
    Rb = np.asarray(Tb, dtype=float)[:3, :3]
    D = Ra.T @ Rb
    v = 0.5 * np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    return float(math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (np.trace(D) - 1.0))))


def duplicate_of(T_new: np.ndarray, existing: Sequence[np.ndarray], min_deg: float) -> Optional[int]:
    """Index of the closest existing pose whose rotation is within ``min_deg``, else None."""
    best, best_d = None, float(min_deg)
    for i, T in enumerate(existing):
        d = rotation_distance_deg(T_new, T)
        if d < best_d:
            best, best_d = i, d
    return best


def _axis_angle(axis, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-15 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def _rotvec_to_R(rotvec_deg) -> np.ndarray:
    v = np.radians(np.asarray(rotvec_deg, dtype=float))
    return _axis_angle(v, float(np.linalg.norm(v)))


def _swing_twist(Rrel: np.ndarray, a: np.ndarray):
    """Decompose ``Rrel = swing @ twist`` about unit axis ``a`` (tool frame).

    Returns (tilt_deg, azimuth_deg, roll_deg): tilt = angle between a and Rrel a,
    azimuth = direction of that lean around ``a`` (0 = towards tool +x), roll =
    remaining rotation about ``a``.
    """
    b = Rrel @ a
    cr = np.cross(a, b)
    tilt = math.atan2(float(np.linalg.norm(cr)), float(a @ b))
    swing = _axis_angle(cr, tilt)
    tw = swing.T @ Rrel
    v = 0.5 * np.array([tw[2, 1] - tw[1, 2], tw[0, 2] - tw[2, 0], tw[1, 0] - tw[0, 1]])
    roll = math.atan2(float(a @ v), 0.5 * (np.trace(tw) - 1.0))
    e1 = np.array([1.0, 0.0, 0.0]) - a[0] * a
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0]) - a[1] * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    d = b - (a @ b) * a
    az = math.atan2(float(d @ e2), float(d @ e1)) if np.linalg.norm(d) > 1e-9 else 0.0
    return math.degrees(tilt), math.degrees(az), math.degrees(roll)


def _mean_rotation(Rs: Sequence[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(np.asarray(Rs), axis=0))
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def _wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- linear algebra core

def _check_T(T, label: str) -> np.ndarray:
    A = np.asarray(T, dtype=float)
    if A.shape != (4, 4):
        raise ValueError(f"{label}: expected a 4x4 transform, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{label}: transform contains NaN/inf; re-capture that pose")
    R = A[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=_ORTHONORMAL_TOL) or abs(np.linalg.det(R) - 1.0) > _ORTHONORMAL_TOL:
        raise ValueError(f"{label}: rotation is not a proper rotation matrix; check the TF source")
    return A


def _fit_rc(Rs: Sequence[np.ndarray], ps: Sequence[np.ndarray]):
    n = len(Rs)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, (R, p) in enumerate(zip(Rs, ps)):
        A[3 * i:3 * i + 3, :3] = R
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -p
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], x[3:]


def _b_svd(Rs: Sequence[np.ndarray]):
    """SVD of the pivot-eliminated B; returns (s, Vt, rank)."""
    Rm = np.mean(np.asarray(Rs), axis=0)
    B = np.vstack([R - Rm for R in Rs])
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    rank = int(np.sum(s > _RANK_REL_TOL * max(1.0, float(s[0])))) if s.size else 0
    return s, Vt, rank


def _err_mm(p, R, r, c) -> float:
    return float(np.linalg.norm(p + R @ r - c) * 1000.0)


# ---------------------------------------------------------------- solver

def _nan3() -> np.ndarray:
    return np.full(3, np.nan)


def solve_pivot(samples: Sequence[PivotSample], options: Optional[PivotOptions] = None,
                current_r: Optional[np.ndarray] = None) -> PivotResult:
    """Fit ``r`` and ``c`` from the ``role == 'fit'`` samples and judge the result.

    ``validate`` samples are never used in the fit; they are evaluated with the
    fixed ``(r, c)``.  Poses in the ``reasons`` are numbered "#N", 1-based by their
    position in ``samples`` (fit and validate together), like a list of the poses.  Raises ``ValueError`` for malformed transforms, mixed
    arms or unknown roles (programming/data errors); everything the operator
    can fix is reported through ``verdict``/``reasons`` instead.
    """
    opt = options or PivotOptions()
    arms = {s.arm for s in samples}
    if len(arms) > 1:
        raise ValueError(f"samples from several arms {sorted(arms)}; solve one arm at a time")
    fit_s, val_s = [], []
    fit_no, val_no = [], []       # 1-based position of each fit / validate pose in `samples` (the "#N" of the reasons)
    for k, s in enumerate(samples):
        if s.role not in ("fit", "validate"):
            raise ValueError(f"sample {k}: role must be 'fit' or 'validate', got {s.role!r}")
        T = _check_T(s.T_base_tool0, f"sample {k}")
        (fit_s if s.role == "fit" else val_s).append(T)
        (fit_no if s.role == "fit" else val_no).append(k + 1)
    n_fit, n_val = len(fit_s), len(val_s)
    cur = None if current_r is None else np.asarray(current_r, dtype=float).reshape(3)

    reasons: List[str] = []
    structural: List[str] = []
    fails: List[str] = []
    warns: List[str] = []
    missing_val = False

    def empty(rank=0, cond=float("nan"), obs=float("nan"), weak=None, why=None) -> PivotResult:
        if why:
            reasons.extend(why)
        return PivotResult(
            ok=False, verdict="INSUFFICIENT", reasons=list(reasons),
            r_tool0=_nan3(), c_base=_nan3(), n_fit=n_fit, n_validate=n_val,
            residuals_mm=[float("nan")] * n_fit, rms_mm=float("nan"), max_mm=float("nan"),
            loo_mm=[float("nan")] * n_fit, loo_rms_mm=float("nan"),
            heldout_mm=[], heldout_rms_mm=None,
            cov_r=np.full((3, 3), np.nan),
            sigma_axes_mm=np.array([np.inf, np.nan, np.nan]),
            weak_axis_tool0=_nan3() if weak is None else weak,
            obs_min=obs, cond_B=cond, rank=rank, outliers=[],
            tilt_deg=[float("nan")] * n_fit, roll_deg=[float("nan")] * n_fit,
            delta_to_current_mm=None)

    if n_fit < 3:
        return empty(why=[f"only {n_fit} fit pose(s); at least {opt.min_fit} are needed "
                          f"(capture {opt.min_fit - n_fit} more, following the pose plan)"])

    Rs = [T[:3, :3] for T in fit_s]
    ps = [T[:3, 3] for T in fit_s]
    s, Vt, rank = _b_svd(Rs)
    obs = float(s[-1] / math.sqrt(n_fit))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    weak_b = Vt[-1] if Vt.shape[0] == 3 else _nan3()
    if rank < 3:
        why = []
        if n_fit < opt.min_fit:
            why.append(f"only {n_fit} fit poses; at least {opt.min_fit} are needed")
        why.append(_weak_axis_text(weak_b, "the fit poses only rotate about a single axis "
                                           f"(rank {rank} < 3), so the TCP offset along tool0 "
                                           f"{_fmt_vec(weak_b)} cannot be determined"))
        return empty(rank=rank, cond=cond, obs=obs, weak=_unit_sign(weak_b), why=why)

    r, c = _fit_rc(Rs, ps)
    res_m = np.array([np.linalg.norm(p + R @ r - c) for R, p in zip(Rs, ps)])
    residuals = (res_m * 1000.0).tolist()
    sse = float(np.sum(res_m ** 2))
    dof = 3 * n_fit - 6
    sigma2 = sse / dof if dof > 0 else float("nan")
    cov = sigma2 * (Vt.T @ np.diag(1.0 / s ** 2) @ Vt)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    sigma_axes = np.sqrt(np.clip(evals[order], 0.0, None)) * 1000.0
    weak = _unit_sign(evecs[:, order[0]])
    rms = float(math.sqrt(np.mean(np.square(residuals))))
    mx = float(np.max(residuals))

    # leave-one-out
    loo: List[float] = []
    for i in range(n_fit):
        idx = [j for j in range(n_fit) if j != i]
        Rr, pr = [Rs[j] for j in idx], [ps[j] for j in idx]
        if len(idx) < 3 or _b_svd(Rr)[2] < 3:
            loo.append(float("nan"))
            continue
        ri, ci = _fit_rc(Rr, pr)
        loo.append(_err_mm(ps[i], Rs[i], ri, ci))
    finite_loo = [v for v in loo if math.isfinite(v)]
    loo_rms = float(math.sqrt(np.mean(np.square(finite_loo)))) if finite_loo else float("nan")
    med = float(np.median(finite_loo)) if finite_loo else float("nan")
    outliers = [i for i, v in enumerate(loo)
                if math.isfinite(v) and v > max(opt.outlier_factor * med, opt.outlier_min_mm)]

    # held-out
    heldout = [_err_mm(T[:3, 3], T[:3, :3], r, c) for T in val_s]
    heldout_rms = float(math.sqrt(np.mean(np.square(heldout)))) if heldout else None

    # tilt / roll per fit sample (about the solved tool axis r/|r|)
    tilt_l: List[float] = []
    roll_l: List[float] = []
    rn = float(np.linalg.norm(r))
    if rn > 1e-6:
        a = r / rn
        Rbar = _mean_rotation(Rs)
        rolls = []
        for R in Rs:
            t, _az, ro = _swing_twist(Rbar.T @ R, a)
            tilt_l.append(t)
            rolls.append(ro)
        # roll relative to the circular mean so a constant offset cancels
        cm = math.degrees(math.atan2(np.mean(np.sin(np.radians(rolls))),
                                     np.mean(np.cos(np.radians(rolls)))))
        roll_l = [_wrap_deg(x - cm) for x in rolls]
    else:
        tilt_l = [float("nan")] * n_fit
        roll_l = [float("nan")] * n_fit

    # ----- judge
    if n_fit < opt.min_fit:
        structural.append(f"only {n_fit} fit poses; at least {opt.min_fit} are needed "
                          f"(capture {opt.min_fit - n_fit} more, following the pose plan)")
    if obs < opt.obs_min:
        structural.append(_weak_axis_text(
            weak_b, f"orientation spread is too small (observability {obs:.2f} < {opt.obs_min:.2f}); "
                    f"r along tool0 {_fmt_vec(weak_b)} is poorly separated from the pivot"))

    if rms > opt.rms_pass_mm:
        fails.append(f"fit RMS {rms:.2f} mm > {opt.rms_pass_mm:.1f} mm: the tip moved in the pivot or the "
                     "pivot is not fixed; re-seat the tip in the dimple (keep it seated while jogging) "
                     "and re-capture the worst poses")
    if mx > opt.max_pass_mm:
        worst = int(np.argmax(residuals))
        fails.append(f"worst residual {mx:.2f} mm (pose #{fit_no[worst]}) > {opt.max_pass_mm:.1f} mm: "
                     f"remove pose #{fit_no[worst]} and re-capture it with the tip fully seated")
    if sigma_axes[0] > opt.axis_sigma_pass_mm:
        fails.append(_weak_axis_text(
            weak, f"TCP uncertainty {sigma_axes[0]:.2f} mm (1-sigma) along tool0 {_fmt_vec(weak)} > "
                  f"{opt.axis_sigma_pass_mm:.1f} mm"))
    if n_val < opt.min_validate:
        missing_val = True
        reasons_val = (f"held-out validation missing: {n_val} validate pose(s), need {opt.min_validate}; "
                       f"capture {opt.min_validate - n_val} more at tilts different from the fit poses "
                       "(they are never used in the fit)")
    elif heldout_rms is not None and heldout_rms > opt.heldout_rms_pass_mm:
        fails.append(f"held-out RMS {heldout_rms:.2f} mm > {opt.heldout_rms_pass_mm:.1f} mm: r does not "
                     "predict poses it was not fitted on; re-seat the tip and re-capture, and add fit "
                     "poses with larger or more varied tilts")
    for i in outliers:
        warns.append(f"pose #{fit_no[i]} (fit) is an outlier: leave-one-out error {loo[i]:.2f} mm vs "
                     f"median {med:.2f} mm; the tip probably slipped there - remove it and re-capture")
    # duplicates (fit-fit)
    for i in range(n_fit):
        for j in range(i + 1, n_fit):
            d = rotation_distance_deg(fit_s[i], fit_s[j])
            if d < opt.dup_min_deg:
                warns.append(f"poses #{fit_no[i]} and #{fit_no[j]} (both fit) are near-duplicate orientations ({d:.1f} deg "
                             f"apart, < {opt.dup_min_deg:.0f}); they add no information - remove one and "
                             "capture a differently tilted pose instead")
    for k, T in enumerate(val_s):
        j = duplicate_of(T, fit_s, opt.dup_min_deg)
        if j is not None:
            warns.append(f"pose #{val_no[k]} (validate) is a near-duplicate of pose #{fit_no[j]} (fit); the held-out "
                         "error is then not independent - re-capture it at a different tilt")

    if structural:
        verdict = "INSUFFICIENT"
    elif fails:
        verdict = "FAIL"
    elif missing_val:
        verdict = "INSUFFICIENT"
    elif rms <= opt.good_rms_mm and mx <= opt.good_max_mm and not warns:
        verdict = "GOOD"
    else:
        verdict = "PASS"

    reasons.extend(structural)
    reasons.extend(fails)
    if missing_val:
        reasons.append(reasons_val)
    reasons.extend(warns)
    if verdict == "PASS" and not warns:
        reasons.append(f"PASS but not GOOD: fit RMS {rms:.2f} mm (GOOD needs <= {opt.good_rms_mm:.1f}) "
                       f"and worst {mx:.2f} mm (<= {opt.good_max_mm:.1f})")

    return PivotResult(
        ok=verdict in ("GOOD", "PASS"), verdict=verdict, reasons=reasons,
        r_tool0=r, c_base=c, n_fit=n_fit, n_validate=n_val,
        residuals_mm=residuals, rms_mm=rms, max_mm=mx,
        loo_mm=loo, loo_rms_mm=loo_rms, heldout_mm=heldout, heldout_rms_mm=heldout_rms,
        cov_r=cov, sigma_axes_mm=sigma_axes, weak_axis_tool0=weak,
        obs_min=obs, cond_B=cond, rank=rank, outliers=outliers,
        tilt_deg=tilt_l, roll_deg=roll_l,
        delta_to_current_mm=None if cur is None else (r - cur) * 1000.0)


def _unit_sign(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return v
    v = v / n
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def _fmt_vec(v) -> str:
    v = np.asarray(v, dtype=float)
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def _weak_axis_text(weak, lead: str) -> str:
    """Operator advice: swing the weak direction by rotating about the *other* tool axes."""
    w = np.abs(np.asarray(weak, dtype=float))
    if not np.all(np.isfinite(w)):
        return lead
    dominant = int(np.argmax(w))
    names = "xyz"
    others = [names[i] for i in range(3) if i != dominant]
    return (f"{lead}; add poses tilted about tool {others[0]} and tool {others[1]} "
            f"(not only about tool {names[dominant]}) so that direction swings")


# ---------------------------------------------------------------- pose plan and coverage

def _plan_item(pid, role, label, text, rotvec_deg=None, tilt_deg=None, roll_deg=None) -> dict:
    return {"id": pid, "role": role, "label": label, "text": text,
            "rotvec_deg": rotvec_deg, "tilt_deg": tilt_deg, "roll_deg": roll_deg}


_KEEP = "keeping the knife tip seated in the pivot dimple"

POSE_PLAN: List[dict] = [
    _plan_item("vertical", "fit", "knife vertical",
               f"Pose 1: hold the knife vertical, tip down in the pivot dimple.", [0, 0, 0], 0.0, None),
    _plan_item("x+30", "fit", "+30 deg about tool x",
               f"Rotate the tool +30 deg about its x axis (right-hand rule), {_KEEP}.", [30, 0, 0]),
    _plan_item("x-30", "fit", "-30 deg about tool x",
               f"Rotate the tool -30 deg about its x axis, {_KEEP}.", [-30, 0, 0]),
    _plan_item("y+30", "fit", "+30 deg about tool y",
               f"Rotate the tool +30 deg about its y axis, {_KEEP}.", [0, 30, 0]),
    _plan_item("y-30", "fit", "-30 deg about tool y",
               f"Rotate the tool -30 deg about its y axis, {_KEEP}.", [0, -30, 0]),
    _plan_item("xy++", "fit", "+21 about x, +21 about y",
               f"Tilt diagonally: +21 deg about tool x and +21 deg about tool y, {_KEEP}.", [21, 21, 0]),
    _plan_item("xy+-", "fit", "+21 about x, -21 about y",
               f"Tilt diagonally: +21 deg about tool x and -21 deg about tool y, {_KEEP}.", [21, -21, 0]),
    _plan_item("xy-+", "fit", "-21 about x, +21 about y",
               f"Tilt diagonally: -21 deg about tool x and +21 deg about tool y, {_KEEP}.", [-21, 21, 0]),
    _plan_item("xy--", "fit", "-21 about x, -21 about y",
               f"Tilt diagonally: -21 deg about tool x and -21 deg about tool y, {_KEEP}.", [-21, -21, 0]),
    _plan_item("roll+90", "fit", "roll +90 deg, tilt 25 deg",
               f"Turn the tool +90 deg about the knife axis (the gripper rolls, the tip stays put), "
               f"then tilt it about 25 deg away from vertical, {_KEEP}.", None, 25.0, 90.0),
    _plan_item("roll-90", "fit", "roll -90 deg, tilt 25 deg",
               f"Turn the tool -90 deg about the knife axis, then tilt it about 25 deg away from "
               f"vertical, {_KEEP}.", None, 25.0, -90.0),
    # validation poses: never used in the fit; >= 30 deg of rotation away from every fit pose
    _plan_item("val-r+45-t15", "validate", "validate: roll +45 deg, tilt 15 deg",
               f"Validation: turn the tool +45 deg about the knife axis, then tilt it about 15 deg away "
               f"from vertical (about tool x), {_KEEP}.", None, 15.0, 45.0),
    _plan_item("val-r-45-t15", "validate", "validate: roll -45 deg, tilt 15 deg",
               f"Validation: turn the tool -45 deg about the knife axis, then tilt it about 15 deg away "
               f"from vertical (about tool x), {_KEEP}.", None, 15.0, -45.0),
    _plan_item("val-r+45-t35", "validate", "validate: roll +45 deg, tilt 35 deg",
               f"Validation: turn the tool +45 deg about the knife axis, then tilt it about 35 deg away "
               f"from vertical (about tool x), {_KEEP}.", None, 35.0, 45.0),
    _plan_item("val-r-45-t35", "validate", "validate: roll -45 deg, tilt 35 deg",
               f"Validation: turn the tool -45 deg about the knife axis, then tilt it about 35 deg away "
               f"from vertical (about tool x), {_KEEP}.", None, 35.0, -45.0),
]


def plan_rotation(item: dict, axis_tool0: np.ndarray) -> np.ndarray:
    """Body-frame rotation (relative to the knife-vertical pose) that realises a plan item.

    Items with ``rotvec_deg`` are plain rotations about the tool axes; the others are
    "roll about the knife axis ``axis_tool0``, then tilt ``tilt_deg`` about tool x".
    Useful to generate mock/synthetic poses for tests.
    """
    if item["rotvec_deg"] is not None:
        return _rotvec_to_R(item["rotvec_deg"])
    a = np.asarray(axis_tool0, dtype=float)
    a = a / np.linalg.norm(a)
    return _rotvec_to_R([item["tilt_deg"], 0.0, 0.0]) @ _axis_angle(a, math.radians(item["roll_deg"]))


def _plan_target(item: dict, a: np.ndarray):
    """(tilt_deg, azimuth_deg or None, roll_deg or None) for a plan item about axis ``a``."""
    if item["rotvec_deg"] is not None:
        t, az, _ro = _swing_twist(_rotvec_to_R(item["rotvec_deg"]), a)
        return t, (az if t > 1.0 else None), None
    return float(item["tilt_deg"]), None, item["roll_deg"]


def coverage(samples: Sequence[PivotSample], r_tool0: Optional[np.ndarray] = None,
             vertical_tol_deg: float = 10.0, match_tilt_deg: float = 8.0,
             match_az_deg: float = 30.0, match_roll_deg: float = 30.0) -> dict:
    """Which planned poses are covered, plus tilt statistics and a next-step hint.

    The knife axis is ``r_tool0`` (default: current URDF TCP) and "vertical" means
    that axis pointing along base -z.  The fit sample closest to vertical is the
    reference; every other sample is described by (tilt, lean azimuth, roll)
    relative to it, and plan items are matched by those numbers within tolerance.
    Only samples with the matching role are matched against an item.
    """
    r = CURRENT_URDF_R_M if r_tool0 is None else np.asarray(r_tool0, dtype=float)
    a = r / np.linalg.norm(r)
    down = np.array([0.0, 0.0, -1.0])
    Ts = [(_check_T(s.T_base_tool0, f"sample {k}"), s.role) for k, s in enumerate(samples)]
    tilt_vert = [math.degrees(math.atan2(float(np.linalg.norm(np.cross(T[:3, :3] @ a, down))),
                                         float((T[:3, :3] @ a) @ down))) for T, _ in Ts]
    n_fit = sum(1 for _, ro in Ts if ro == "fit")
    n_val = sum(1 for _, ro in Ts if ro == "validate")

    ref = None
    cand = [i for i, (_, ro) in enumerate(Ts) if ro == "fit" and tilt_vert[i] <= vertical_tol_deg]
    if cand:
        ref = min(cand, key=lambda i: tilt_vert[i])

    covered, missing = [], []
    rel_stats: List[Optional[tuple]] = [None] * len(Ts)
    if ref is not None:
        Rref = Ts[ref][0][:3, :3]
        rel_stats = [_swing_twist(Rref.T @ T[:3, :3], a) for T, _ in Ts]
    for item in POSE_PLAN:
        ok = False
        if ref is not None:
            tt, taz, tro = _plan_target(item, a)
            for i, (_, ro) in enumerate(Ts):
                if ro != item["role"]:
                    continue
                t, az, roll = rel_stats[i]
                if abs(t - tt) > (match_tilt_deg if item["id"] != "vertical" else vertical_tol_deg):
                    continue
                if taz is not None and abs(_wrap_deg(az - taz)) > match_az_deg:
                    continue
                if tro is not None and abs(_wrap_deg(roll - tro)) > match_roll_deg:
                    continue
                ok = True
                break
        (covered if ok else missing).append(item["id"])

    edges = [0, 10, 20, 30, 40, 180]
    labels = ["0-10", "10-20", "20-30", "30-40", "40+"]
    tilt_hist = {lab: 0 for lab in labels}
    for t in tilt_vert:
        for k in range(5):
            if edges[k] <= t < edges[k + 1]:
                tilt_hist[labels[k]] += 1
                break
    az_hist = {f"{d}": 0 for d in range(0, 360, 45)}
    if ref is not None:
        for i, st in enumerate(rel_stats):
            if i == ref or st is None or st[0] < 5.0:
                continue
            sector = int(round((st[1] % 360.0) / 45.0)) % 8 * 45
            az_hist[str(sector)] += 1

    by_id = {it["id"]: it for it in POSE_PLAN}
    if not samples:
        suggestion = by_id["vertical"]["text"] + " Start here."
        nxt = "vertical"
    elif ref is None:
        suggestion = ("No fit pose with the knife vertical yet: " + by_id["vertical"]["text"] +
                      " (the other poses are described relative to it).")
        nxt = "vertical"
    else:
        todo = [i for i in missing if by_id[i]["role"] == "fit"] or \
               [i for i in missing if by_id[i]["role"] == "validate"]
        if todo:
            nxt = todo[0]
            suggestion = "Next: " + by_id[nxt]["text"]
        else:
            nxt = None
            suggestion = (f"Pose plan complete ({n_fit} fit + {n_val} validate poses). "
                          "Solve, then check the verdict.")
    return {
        "n_fit": n_fit, "n_validate": n_val, "reference_index": ref,
        "covered": covered, "missing": missing, "next": nxt, "suggestion": suggestion,
        "tilt_from_vertical_deg": tilt_vert, "max_tilt_deg": max(tilt_vert) if tilt_vert else 0.0,
        "tilt_hist": tilt_hist, "azimuth_hist_deg": az_hist,
        "complete": not missing,
    }


# ---------------------------------------------------------------- result file

def write_result_yaml(result: PivotResult, path: str, arm: str, urdf_hash: str,
                      xacro_path: str, current_r) -> None:
    """Write a versioned calibration record (atomic).  Does not touch the URDF."""
    import yaml

    cur = None if current_r is None else np.asarray(current_r, dtype=float).reshape(3)
    r = np.asarray(result.r_tool0, dtype=float)
    doc = {
        "kind": RESULT_YAML_KIND,
        "version": 1,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "arm": arm,
        "verdict": result.verdict,
        "usable": bool(result.ok),
        "reasons": list(result.reasons),
        "r_tool0_m": r.tolist(),
        "r_tool0_mm": (r * 1000.0).tolist(),
        "c_base_m": np.asarray(result.c_base, dtype=float).tolist(),
        "current_r_m": None if cur is None else cur.tolist(),
        "delta_to_current_mm": None if cur is None else ((r - cur) * 1000.0).tolist(),
        "n_fit": result.n_fit,
        "n_validate": result.n_validate,
        "rms_mm": result.rms_mm,
        "max_mm": result.max_mm,
        "loo_rms_mm": result.loo_rms_mm,
        "heldout_rms_mm": result.heldout_rms_mm,
        "sigma_axes_mm": np.asarray(result.sigma_axes_mm, dtype=float).tolist(),
        "weak_axis_tool0": np.asarray(result.weak_axis_tool0, dtype=float).tolist(),
        "obs_min": result.obs_min,
        "cond_B": result.cond_B,
        "rank": result.rank,
        "outliers": list(result.outliers),
        "urdf_hash": urdf_hash,
        "xacro_path": xacro_path,
        "joint": TCP_JOINT_NAME,
    }
    doc = _jsonable(doc)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- xacro diff

def _blank_comments(text: str) -> str:
    """Replace XML comment bodies with spaces (newlines kept) so offsets stay valid."""
    return re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def xacro_apply(xacro_text: str, r_new: np.ndarray, joint_name: str = TCP_JOINT_NAME) -> str:
    """Return ``xacro_text`` with only the origin xyz of ``joint_name`` replaced (6 decimals).

    Raises ``ValueError`` unless the joint exists exactly once (comments ignored)
    with exactly one ``<origin ... xyz="...">``.
    """
    r = np.asarray(r_new, dtype=float).reshape(3)
    if not np.all(np.isfinite(r)) or np.any(np.abs(r) > 1.0):
        raise ValueError(f"refusing to patch with implausible r_new={r.tolist()} (must be finite, |.| <= 1 m)")
    scan = _blank_comments(xacro_text)
    starts = [m for m in re.finditer(r"<joint\b[^>]*?\bname\s*=\s*([\"'])" + re.escape(joint_name) + r"\1[^>]*>", scan)]
    if len(starts) != 1:
        raise ValueError(f"joint {joint_name!r} found {len(starts)} times in the xacro; need exactly 1")
    end = scan.find("</joint>", starts[0].end())
    if end < 0:
        raise ValueError(f"joint {joint_name!r} has no closing </joint>")
    body_lo, body_hi = starts[0].end(), end
    origins = list(re.finditer(r"<origin\b[^>]*>", scan[body_lo:body_hi]))
    if len(origins) != 1:
        raise ValueError(f"joint {joint_name!r} has {len(origins)} <origin> elements; need exactly 1")
    o = origins[0]
    xm = list(re.finditer(r"(\bxyz\s*=\s*)([\"'])([^\"']*)\2", o.group(0)))
    if len(xm) != 1:
        raise ValueError(f"joint {joint_name!r} origin has no single xyz attribute")
    x = xm[0]
    new_val = " ".join(f"{v:.6f}" for v in r)
    abs_lo = body_lo + o.start() + x.start()
    abs_hi = body_lo + o.start() + x.end()
    new_attr = f"{x.group(1)}{x.group(2)}{new_val}{x.group(2)}"
    return xacro_text[:abs_lo] + new_attr + xacro_text[abs_hi:]


def xacro_patch(xacro_text: str, r_new: np.ndarray, joint_name: str = TCP_JOINT_NAME,
                path: str = "igus_rebel_xeg32_wrapper.urdf.xacro") -> str:
    """Unified diff changing only the origin xyz of ``joint_name`` (rpy untouched).

    Never writes a file; the operator applies it deliberately (``patch -p1``).
    Returns "" if the rounded values are already in the file.
    """
    new = xacro_apply(xacro_text, r_new, joint_name)
    return "".join(difflib.unified_diff(
        xacro_text.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3))


# ---------------------------------------------------------------- mechanical cross-check

def mechanical_check(r_new, measured_flange_to_tip_mm: float, tol_mm: float = 2.0) -> dict:
    """Compare a caliper length (link6 flange face -> tip, along link6 z) with the model.

    link6 -> igus_tool0 is a 52.7 mm translation along z plus a rotation about z,
    so the tip's height above the flange face is ``52.7 + r_z``.  This checks only
    r_z, not r_x / r_y.
    """
    r = np.asarray(r_new, dtype=float).reshape(3)
    meas = float(measured_flange_to_tip_mm)
    pred_new = LINK6_TO_TOOL0_Z_MM + float(r[2]) * 1000.0
    pred_cur = LINK6_TO_TOOL0_Z_MM + float(CURRENT_URDF_R_M[2]) * 1000.0
    d_new = meas - pred_new
    d_cur = meas - pred_cur
    ok_new = abs(d_new) <= tol_mm
    ok_cur = abs(d_cur) <= tol_mm
    text = (f"Caliper {meas:.1f} mm vs new model {pred_new:.1f} mm (52.7 + r_z {r[2] * 1000:.1f}): "
            f"{d_new:+.1f} mm; vs current URDF {pred_cur:.1f} mm: {d_cur:+.1f} mm. ")
    if ok_new:
        text += (f"The new r_z agrees with the caliper within {tol_mm:.1f} mm "
                 "(this checks r_z only, not r_x / r_y).")
    elif ok_cur:
        text += ("The caliper agrees with the CURRENT URDF but not with the new r_z: re-check the pivot "
                 "poses before applying the new value.")
    else:
        text += (f"Neither model agrees within {tol_mm:.1f} mm: re-measure from the link6 flange face to "
                 "the tip, and do not apply the new r until this is explained.")
    return {"measured_mm": meas, "predicted_new_mm": pred_new, "predicted_current_mm": pred_cur,
            "delta_new_mm": d_new, "delta_current_mm": d_cur, "tol_mm": tol_mm,
            "consistent_new": ok_new, "consistent_current": ok_cur, "text": text}
