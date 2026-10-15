"""Next-calibration-pose suggestion engine for one arm (numpy / scipy / cv2 only, no ROS).

Hand-eye model (see ``solver.py``): the board pose in the optical frame for a flange pose ``A = T_base_link6`` is
``T_optical_board = X @ A @ Z`` with ``X = T_optical_base`` and ``Z = T_link6_target``.  Given the fit of the samples
recorded so far, this module proposes flange poses that would improve the calibration the most:

1. ``generate_candidates`` casts rays through a 3x3 grid of image cells, places the board at several distances along each
   ray and tries a set of orientations (turn / tip / spin).  Cheap geometric filters (height above the base plane,
   reach, view angle, board fully inside the image) remove most candidates; the survivors are scored.
2. ``suggest`` walks the scored candidates best first, asks an injected IK / validity check for a joint goal and keeps
   the first ``n`` that are reachable, safe, close enough to the current configuration, new (not a repeat of a
   recorded sample) and mutually distinct.

Orientation convention
----------------------
The board frame is the one of ``targets.py``: origin at the board centre, x right, y down "as read", z = x cross y,
i.e. pointing *into* the board (away from a viewer looking at its printed face).  For each candidate a "line-of-sight
frame" ``R_face`` is built (minimal rotation taking the optical z axis onto the ray towards the board centre), and the
board orientation is

    R_optical_board = R_face @ Ry(turn) @ Rx(tip) @ Rz(spin) @ F

``F`` is the identity if the existing samples have the board z axis pointing away from the camera (this is the case
for the real data; ``F = Rx(pi)`` otherwise: the facing sign is *measured* from the samples, not assumed).  With
``turn = tip = 0`` the printed face looks straight at the camera.  ``spin`` is the in-plane roll; its base value is the
circular mean of the spins of the existing samples (so the wrist stays in a familiar configuration).

Sign conventions (verified numerically in ``test/test_pose_suggest.py``):

* positive ``turn`` (rotation about the line-of-sight frame's +y) moves the board's image-right side towards the
  camera -> "right edge closer".
* positive ``tip`` (rotation about +x) moves the board's top (image up) side towards the camera -> "top tipped towards".

The words depend only on the board *plane normal*, not on the spin.

Scoring
-------
``score = w_info * info_norm + w_rot * rot_norm + w_cell * empty_cell + w_dist * dist_bonus`` where

* ``info_gain`` = log det(I_old + J_c^T J_c) - log det(I_old) (nats).  ``I_old = J^T J`` of the existing samples'
  residuals at the fitted solution (12 parameters: dX, dZ; weights as in the solver), ``J_c`` the Jacobian of a
  synthetic sample (the candidate, its predicted pixels of all ``obj_full`` corners).  Information is additive so
  ``I_old`` is computed once.  This is the D-optimal design criterion: how much the parameter covariance volume
  shrinks.  ``info_norm = info_gain / max(info_gain over the scored batch)``.
* ``rot_gain`` = increase (deg) of the 2nd + 3rd principal spreads of ``rotation_diversity_deg`` over the link6
  rotations when the candidate is added.  Hand-eye needs at least two non-parallel rotation axes, and the solver warns
  below 30 deg of 2nd spread; D-optimality alone cannot see this because the information matrix only reflects the
  current linearisation.  ``rot_norm = max(rot_gain, 0) / max(rot_gain over the batch)``.
* ``empty_cell``: 1 if no existing sample's board centre lies in the candidate's 3x3 image cell (coverage of the image
  matters for distortion / intrinsics errors that the 12-parameter model cannot see).
* ``dist_bonus``: 1 if no existing sample is within half a distance step of the candidate's distance.

Default weights ``w_info=1.0, w_rot=1.0, w_cell=0.35, w_dist=0.15``: the two normalised terms are in [0, 1] and are the
direct, quantitative objectives, so they get equal weight; the coverage bonuses are binary heuristics that should break
near-ties and push into unexplored regions without overriding a clearly more informative pose (a full cell bonus is
worth about a third of the best information gain; the distance bonus is only a small nudge).

Speed: everything is vectorised over the whole candidate batch (batched numeric Jacobian by central differences, with
the same delta parametrisation and step as ``solver._ArmModel`` / ``_numeric_jacobian``; batched rotation spreads
replicating ``rotation_diversity_deg``), so *all* geometric survivors (~1000) are scored, no top-N pre-selection.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .solver import (Sample, SolverError, SolverOptions, _ArmModel, _flag_outliers,
                     _numeric_jacobian, project_points, rotation_diversity_deg, solve_arm_only)
from .targets import CHESS_ID_BASE
from .transforms import inv_T, make_T, rotvec_to_R, xyz_rpy_to_T

MIN_SAMPLES = 6
_JAC_STEP = 1e-6          # same central-difference step as solver._numeric_jacobian


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class ArmFit:
    arm: str
    X: np.ndarray            # T_optical_base (4x4)
    Z: np.ndarray            # T_link6_target (4x4)
    K: np.ndarray
    dist: np.ndarray
    image_size: Tuple[int, int]      # (w, h), from the samples
    obj_full: np.ndarray     # (N,3) board-frame corner points of the most complete detection (all corners)
    samples: list            # the samples used (outliers of the fit removed)
    # ---- extensions (optional, filled by fit_arm) ----
    ids_full: Optional[np.ndarray] = None       # (N,) point ids of obj_full (weights: markers noisier than saddles)
    board_size: Optional[Tuple[float, float]] = None   # (width, height) in m of the printed board (outline)
    info_old: Optional[np.ndarray] = None       # (12,12) J^T J of the existing samples at the solution


@dataclass
class Suggestion:
    rank: int
    arm: str
    T_base_link6: np.ndarray        # 4x4 flange target in the arm base frame
    q: Optional[Dict[str, float]]   # joint goal (rad) from the injected IK; None before IK
    outline_px: List[Tuple[float, float]]   # 4 outer board corners (TL, TR, BR, BL of the board), distorted pixels
    centre_px: Tuple[float, float]
    description: str
    gains: Dict[str, float]
    score: float


@dataclass
class SuggestOptions:
    # ---- candidate grid ----
    grid_cols: int = 3                                  # image cells (columns x rows); cell centres are the ray targets
    grid_rows: int = 3
    distances_m: Tuple[float, ...] = (0.5, 0.65, 0.8, 0.95)     # board-centre distance from the camera along the ray
    turns_deg: Tuple[float, ...] = (-40.0, -25.0, 0.0, 25.0, 40.0)   # rotation about the line-of-sight frame's y
    tips_deg: Tuple[float, ...] = (-35.0, -20.0, 0.0, 20.0, 35.0)    # rotation about the line-of-sight frame's x
    spin_offsets_deg: Tuple[float, ...] = (-30.0, 0.0, 30.0)          # added to the circular mean spin of the samples
    # ---- geometric filters ----
    min_height_m: float = 0.10          # board centre above the arm base plane (z = 0 = table top)
    max_reach_m: float = 0.62           # link6 position distance from the base origin (cheap pre-filter, IK decides)
    max_view_angle_deg: float = 60.0    # board normal vs line of sight
    image_margin_px: float = 40.0       # every outer board corner must be this far inside the image
    # ---- score weights (see module docstring) ----
    w_info: float = 1.0
    w_rot: float = 1.0
    w_cell: float = 0.35
    w_dist: float = 0.15
    # ---- suggest() ----
    max_joint_change_deg: float = 90.0  # per joint, versus current_q
    max_ik_calls: int = 150
    duplicate_joint_tol_deg: float = 2.0     # all joints within this of a recorded sample -> repeat
    # Two suggestions are "too close" if EITHER the flange poses are within min_separation_m AND min_separation_deg,
    # OR the board looks alike in the image: centres within min_separation_px_frac * image width AND board
    # orientations within min_view_separation_deg (distance along the ray is ignored: same view, same information).
    min_separation_m: float = 0.08
    min_separation_deg: float = 20.0
    min_separation_px_frac: float = 0.15
    min_view_separation_deg: float = 25.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _dist8(dist: np.ndarray) -> np.ndarray:
    d = np.zeros(8)
    d[: min(8, len(dist))] = np.asarray(dist, dtype=float).reshape(-1)[:8]
    return d


def _ray_frames(rays: np.ndarray) -> np.ndarray:
    """(B,3,3) minimal rotations taking the optical z axis onto each unit ray."""
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(ez, rays)
    s = np.linalg.norm(axis, axis=1)
    ang = np.arctan2(s, rays[:, 2])
    rv = np.where(s[:, None] > 1e-12, axis / np.maximum(s, 1e-12)[:, None] * ang[:, None], 0.0)
    return Rotation.from_rotvec(rv).as_matrix()


def _facing_flip(sign: float) -> np.ndarray:
    return np.eye(3) if sign >= 0 else Rotation.from_euler("x", np.pi).as_matrix()


def _board_poses_opt(fit: ArmFit, A: np.ndarray) -> np.ndarray:
    """T_optical_board = X A Z for flange poses A of shape (4,4) or (B,4,4)."""
    return fit.X @ A @ fit.Z


def _facing_sign(fit: ArmFit) -> float:
    """+1 if the board z axis points away from the camera when the printed face is seen (measured on the samples)."""
    vals = []
    for s in fit.samples:
        T = fit.X @ s.T_base_link6 @ fit.Z
        p = T[:3, 3]
        vals.append(float(T[:3, 2] @ p) / max(float(np.linalg.norm(p)), 1e-9))
    return 1.0 if not vals or float(np.median(vals)) >= 0.0 else -1.0


def _decompose(T_ob: np.ndarray, F: np.ndarray):
    """Line-of-sight frame decomposition of board poses (B,4,4): (R_face, M, spin_deg) with
    R_ob = R_face @ M @ F and M = Ry(turn) Rx(tip) Rz(spin)."""
    p = T_ob[:, :3, 3]
    rays = p / np.linalg.norm(p, axis=1, keepdims=True)
    Rf = _ray_frames(rays)
    M = np.swapaxes(Rf, 1, 2) @ T_ob[:, :3, :3] @ F.T
    return rays, Rf, M


def _spin_deg(M: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(M).as_euler("YXZ", degrees=True)[:, 2]


def _turn_tip_deg(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = M[:, :, 2]            # board z in the line-of-sight frame = (s*cos(tip), -sin(tip), c*cos(tip))
    turn = np.degrees(np.arctan2(n[:, 0], n[:, 2]))
    tip = np.degrees(np.arcsin(np.clip(-n[:, 1], -1.0, 1.0)))
    return turn, tip


def _board_size_from_points(ids: np.ndarray, obj: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(width, height), (cx, cy) of the printed board estimated from the detected corner points."""
    chess = obj[ids >= CHESS_ID_BASE]
    if len(chess) >= 4:
        ux, uy = np.unique(np.round(chess[:, 0], 6)), np.unique(np.round(chess[:, 1], 6))
        if len(ux) >= 2 and len(uy) >= 2:
            px, py = float(np.median(np.diff(ux))), float(np.median(np.diff(uy)))
            return ((ux[-1] - ux[0]) + 2 * px, (uy[-1] - uy[0]) + 2 * py), \
                   (0.5 * (ux[-1] + ux[0]), 0.5 * (uy[-1] + uy[0]))
    lo, hi = obj[:, :2].min(axis=0), obj[:, :2].max(axis=0)
    ext = (hi - lo) * 1.1
    return (float(ext[0]), float(ext[1])), (float(0.5 * (hi[0] + lo[0])), float(0.5 * (hi[1] + lo[1])))


def _outline_obj(fit: ArmFit) -> np.ndarray:
    """(4,3) outer board corners in the board frame: TL, TR, BR, BL (x right, y down)."""
    if fit.board_size is not None:
        w, h = fit.board_size
        c = np.zeros(2)
    else:
        (w, h), c = _board_size_from_points(
            fit.ids_full if fit.ids_full is not None else np.zeros(len(fit.obj_full), int), fit.obj_full)
    x0, x1, y0, y1 = c[0] - w / 2, c[0] + w / 2, c[1] - h / 2, c[1] + h / 2
    return np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])


def _project_batch(fit: ArmFit, T_ob: np.ndarray, pts: np.ndarray, zmin: float = 1e-3):
    """Project board-frame points (n,3) with poses T_ob (B,4,4): (B,n,2) pixels and the camera-frame z (B,n)."""
    P = np.einsum("bij,nj->bni", T_ob[:, :3, :3], pts) + T_ob[:, None, :3, 3]
    z = P[:, :, 2].copy()
    P[:, :, 2] = np.maximum(P[:, :, 2], zmin)
    uv = project_points(fit.K, fit.dist, P.reshape(-1, 3)).reshape(len(T_ob), len(pts), 2)
    return uv, z


def _cells(uv: np.ndarray, size: Tuple[int, int], cols: int, rows: int) -> np.ndarray:
    w, h = size
    c = np.clip(np.floor(uv[:, 0] / w * cols).astype(int), 0, cols - 1)
    r = np.clip(np.floor(uv[:, 1] / h * rows).astype(int), 0, rows - 1)
    return r * cols + c


def _position_words(cell: int, cols: int, rows: int) -> str:
    r, c = divmod(int(cell), cols)
    fr, fc = (r + 0.5) / rows, (c + 0.5) / cols
    rw = "upper" if fr < 1 / 3 else "lower" if fr > 2 / 3 else "middle"
    cw = "left" if fc < 1 / 3 else "right" if fc > 2 / 3 else "centre"
    if rw == "middle" and cw == "centre":
        return "centre of image"
    return f"{rw}-{cw} of image"


def _describe(cell: int, turn: float, tip: float, dist: float, opt: SuggestOptions, spin_rel: float = 0.0) -> str:
    side = "right" if turn >= 0.0 else "left"
    updown = "towards" if tip > 0.0 else "away"
    roll = f", wrist roll {spin_rel:+.0f} deg vs usual" if abs(spin_rel) >= 10.0 else ""
    return (f"{_position_words(cell, opt.grid_cols, opt.grid_rows)}, {side} edge closer {abs(turn):.0f} deg, "
            f"top tipped {updown} {abs(tip):.0f} deg{roll}, {dist:.2f} m")


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
def fit_arm(samples: Sequence[Sample], arm: str, target=None) -> ArmFit:
    """Fit X (T_optical_base) and Z (T_link6_target) of one arm from its samples.

    ``target`` (optional ``targets.TargetSpec``) gives the printed board size for the outline; without it the size is
    estimated from the corner ids/positions.  Raises ``ValueError`` if fewer than 6 usable samples."""
    usable = [s for s in samples if s.arm == arm and len(s.ids) >= 4]
    if len(usable) < MIN_SAMPLES:
        raise ValueError(f"arm '{arm}': {len(usable)} usable samples, at least {MIN_SAMPLES} are needed to suggest a "
                         "next pose; record more samples first")
    try:
        res = solve_arm_only(usable)
    except (SolverError, cv2.error, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        raise ValueError(f"arm '{arm}': the single-arm fit failed ({exc}); cannot suggest a pose") from exc
    X = np.asarray(res["T_optical_base"], dtype=float)
    Z = xyz_rpy_to_T(res["target_offset"]["xyz"], res["target_offset"]["rpy"])
    model = _ArmModel(usable, X, Z)
    sample_ok, _ = _flag_outliers(model, np.zeros(12), SolverOptions())
    used = [s for s, ok in zip(usable, sample_ok) if ok]
    if len(used) < MIN_SAMPLES:
        raise ValueError(f"arm '{arm}': only {len(used)} samples are consistent with the fit "
                         f"(of {len(usable)}); need at least {MIN_SAMPLES}")
    model = _ArmModel(used, X, Z)
    _, cmask = _flag_outliers(model, np.zeros(12), SolverOptions())
    J = _numeric_jacobian(lambda th: model.residuals(th, cmask), np.zeros(12))
    info_old = J.T @ J
    pts: Dict[int, np.ndarray] = {}
    for s in used:
        for i, o in zip(s.ids, s.obj):
            pts.setdefault(int(i), o)
    ids_full = np.array(sorted(pts), dtype=int)
    obj_full = np.array([pts[i] for i in ids_full]).reshape(-1, 3)
    board = None
    if target is not None:
        bw, bh = target.board_size()
        board = (float(bw), float(bh))
    s0 = used[0]
    return ArmFit(arm=arm, X=X, Z=Z, K=np.array(s0.K), dist=np.array(s0.dist), image_size=tuple(s0.image_size),
                  obj_full=obj_full, samples=used, ids_full=ids_full, board_size=board, info_old=info_old)


def predicted_outline(fit: ArmFit, T_base_link6: np.ndarray) -> List[Tuple[float, float]]:
    """4 outer board corners (TL, TR, BR, BL) predicted in the distorted image for a flange pose.  Corners at or behind
    the camera plane are clamped to z = 1 mm (finite but meaningless values)."""
    T_ob = _board_poses_opt(fit, np.asarray(T_base_link6, dtype=float).reshape(1, 4, 4))
    uv, _ = _project_batch(fit, T_ob, _outline_obj(fit))
    return [(float(u), float(v)) for u, v in uv[0]]


# --------------------------------------------------------------------------
# Batched terms
# --------------------------------------------------------------------------
def _info_old(fit: ArmFit) -> np.ndarray:
    if fit.info_old is not None:
        return fit.info_old
    model = _ArmModel(fit.samples, fit.X, fit.Z)
    _, cmask = _flag_outliers(model, np.zeros(12), SolverOptions())
    J = _numeric_jacobian(lambda th: model.residuals(th, cmask), np.zeros(12))
    return J.T @ J


def _point_sigma(fit: ArmFit) -> np.ndarray:
    n = len(fit.obj_full)
    if fit.ids_full is None:
        return np.ones(n)
    return np.where(fit.ids_full >= CHESS_ID_BASE, 1.0, float(SolverOptions().marker_sigma_ratio))


def _delta_T(j: int, h: float) -> np.ndarray:
    d = np.zeros(6)
    d[j] = h
    return make_T(rotvec_to_R(d[:3]), d[3:])


def info_gain_batch(fit: ArmFit, A: np.ndarray) -> np.ndarray:
    """log det(I_old + J_c^T J_c) - log det(I_old) for flange poses A (B,4,4).

    J_c is the numeric Jacobian (central differences, step 1e-6, delta parametrisation around the fitted X, Z; exactly
    the model of ``solver._ArmModel``) of the pixel residuals of a synthetic sample that sees every ``obj_full`` corner."""
    B, N = len(A), len(fit.obj_full)
    sig = _point_sigma(fit)
    cols = []
    for j in range(12):
        cc = []
        for sgn in (+1.0, -1.0):
            E = _delta_T(j % 6, sgn * _JAC_STEP)
            Xm, Zm = (fit.X @ E, fit.Z) if j < 6 else (fit.X, fit.Z @ E)
            uv, _ = _project_batch(fit, Xm @ A @ Zm, fit.obj_full)
            cc.append(uv)
        cols.append((cc[0] - cc[1]) / (2 * _JAC_STEP))                      # (B,N,2)
    J = np.stack(cols, axis=-1) / sig[None, :, None, None]                   # (B,N,2,12)
    Jf = J.reshape(B, N * 2, 12)
    Ic = np.einsum("bki,bkj->bij", Jf, Jf)
    I0 = _info_old(fit)
    return np.linalg.slogdet(I0[None] + Ic)[1] - np.linalg.slogdet(I0)[1]


def rotation_spreads_batch(R_exist: np.ndarray, R_cand: np.ndarray) -> np.ndarray:
    """(B,3) principal spreads (deg, descending) of ``R_exist`` (n,3,3) plus each candidate rotation (B,3,3);
    batched replica of ``solver.rotation_diversity_deg``."""
    B, n = len(R_cand), len(R_exist)
    q_old = Rotation.from_matrix(R_exist).as_quat()
    q_c = Rotation.from_matrix(R_cand).as_quat()
    Msum = q_old.T @ q_old + q_c[:, :, None] * q_c[:, None, :]               # (B,4,4)
    m = np.linalg.eigh(Msum)[1][:, :, -1]                                    # mean quaternion (x,y,z,w)
    q_all = np.concatenate([np.broadcast_to(q_old, (B, n, 4)), q_c[:, None, :]], axis=1)    # (B,n+1,4)
    a = np.concatenate([-m[:, :3], m[:, 3:]], axis=1)[:, None, :]            # conj(mean)
    av, aw, bv, bw = a[..., :3], a[..., 3:], q_all[..., :3], q_all[..., 3:]
    rel = np.concatenate([aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, axis=-1, keepdims=True)],
                         axis=-1)
    rv = Rotation.from_quat(rel.reshape(-1, 4)).as_rotvec().reshape(B, n + 1, 3)
    rv = rv - rv.mean(axis=1, keepdims=True)
    sv = np.linalg.svd(rv, compute_uv=False) / math.sqrt(n + 1)
    return np.degrees(sv)


def _terms(fit: ArmFit, opt: SuggestOptions, A: np.ndarray, T_ob: np.ndarray, outline: np.ndarray,
           centre_uv: np.ndarray) -> dict:
    """All raw (un-normalised) per-candidate terms for flange poses A (B,4,4) / board poses T_ob."""
    F = _facing_flip(_facing_sign(fit))
    rays, Rf, M = _decompose(T_ob, F)
    turn, tip = _turn_tip_deg(M)
    spin = _spin_deg(M)
    p = T_ob[:, :3, 3]
    dist = np.linalg.norm(p, axis=1)
    s = _facing_sign(fit)
    view = np.degrees(np.arccos(np.clip(s * np.einsum("bi,bi->b", T_ob[:, :3, 2], rays), -1.0, 1.0)))
    cell = _cells(centre_uv, fit.image_size, opt.grid_cols, opt.grid_rows)
    # existing samples
    S = np.array([x.T_base_link6 for x in fit.samples])
    T_s = _board_poses_opt(fit, S)
    cuv, _ = _project_batch(fit, T_s, np.zeros((1, 3)))
    ex_cells = set(_cells(cuv[:, 0, :], fit.image_size, opt.grid_cols, opt.grid_rows).tolist())
    empty = np.array([0.0 if int(c) in ex_cells else 1.0 for c in cell])
    ex_d = np.linalg.norm(T_s[:, :3, 3], axis=1)
    _, _, M_ex = _decompose(T_s, F)
    spin_rel = (spin - _circular_mean_deg(_spin_deg(M_ex)) + 180.0) % 360.0 - 180.0   # roll vs the usual roll
    step = np.median(np.diff(sorted(opt.distances_m))) if len(opt.distances_m) > 1 else 0.15
    dist_bonus = np.array([0.0 if np.any(np.abs(ex_d - d) < step / 2.0) else 1.0 for d in dist])
    # rotation diversity
    R_ex = S[:, :3, :3]
    before = rotation_diversity_deg(list(R_ex))
    after = rotation_spreads_batch(R_ex, A[:, :3, :3])
    rot_gain = (after[:, 1] - before[1]) + (after[:, 2] - before[2])
    info = info_gain_batch(fit, A)
    return dict(turn=turn, tip=tip, spin=spin, spin_rel=spin_rel, dist=dist, view=view, cell=cell, empty=empty, dist_bonus=dist_bonus,
                spread2_before=np.full(len(A), before[1]), spread2_after=after[:, 1],
                spread3_before=np.full(len(A), before[2]), spread3_after=after[:, 2], rot_gain=rot_gain, info=info)


def _score(t: dict, opt: SuggestOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    info_norm = np.maximum(t["info"], 0.0) / max(float(np.max(t["info"])), 1e-12) if len(t["info"]) else t["info"]
    rg = np.maximum(t["rot_gain"], 0.0)
    rot_norm = rg / max(float(np.max(rg)), 1e-12) if len(rg) else rg
    score = (opt.w_info * info_norm + opt.w_rot * rot_norm + opt.w_cell * t["empty"] + opt.w_dist * t["dist_bonus"])
    return score, info_norm, rot_norm


def _make_suggestions(fit: ArmFit, opt: SuggestOptions, A: np.ndarray, uv_outline: np.ndarray,
                      centre_uv: np.ndarray, t: dict) -> List[Suggestion]:
    score, info_norm, rot_norm = _score(t, opt)
    out = []
    for i in range(len(A)):
        gains = dict(info_gain=float(t["info"][i]), info_norm=float(info_norm[i]),
                     rot_spread2_before=float(t["spread2_before"][i]), rot_spread2_after=float(t["spread2_after"][i]),
                     rot_spread3_before=float(t["spread3_before"][i]), rot_spread3_after=float(t["spread3_after"][i]),
                     rot_gain_deg=float(t["rot_gain"][i]), rot_norm=float(rot_norm[i]),
                     empty_cell=float(t["empty"][i]), dist_bonus=float(t["dist_bonus"][i]),
                     view_angle_deg=float(t["view"][i]), distance_m=float(t["dist"][i]),
                     turn_deg=float(t["turn"][i]), tip_deg=float(t["tip"][i]), spin_deg=float(t["spin"][i]),
                     cell=float(t["cell"][i]), max_joint_change_deg=0.0)     # last one is filled by suggest() after IK
        out.append(Suggestion(
            rank=0, arm=fit.arm, T_base_link6=A[i].copy(), q=None,
            outline_px=[(float(u), float(v)) for u, v in uv_outline[i]],
            centre_px=(float(centre_uv[i, 0]), float(centre_uv[i, 1])),
            description=_describe(int(t["cell"][i]), float(t["turn"][i]), float(t["tip"][i]), float(t["dist"][i]), opt,
                                  float(t["spin_rel"][i])),
            gains=gains, score=float(score[i])))
    return out


def _eval_poses(fit: ArmFit, opt: SuggestOptions, A: np.ndarray) -> List[Suggestion]:
    T_ob = _board_poses_opt(fit, A)
    uv_o, _ = _project_batch(fit, T_ob, _outline_obj(fit))
    uv_c, _ = _project_batch(fit, T_ob, np.zeros((1, 3)))
    t = _terms(fit, opt, A, T_ob, uv_o, uv_c[:, 0, :])
    return _make_suggestions(fit, opt, A, uv_o, uv_c[:, 0, :], t)


def score_poses(fit: ArmFit, T_base_link6_list: Sequence[np.ndarray],
                options: Optional[SuggestOptions] = None) -> List[Suggestion]:
    """Score arbitrary flange poses with the same terms as the candidates (no geometric filtering).  Normalisation is
    relative to the best pose *in this list*.  Poses whose board centre is behind the camera are not allowed."""
    opt = options or SuggestOptions()
    A = np.array([np.asarray(T, dtype=float).reshape(4, 4) for T in T_base_link6_list]).reshape(-1, 4, 4)
    if not len(A):
        return []
    if np.any(_board_poses_opt(fit, A)[:, 2, 3] <= 0.0):
        raise ValueError("a pose has the board centre behind the camera")
    return _eval_poses(fit, opt, A)


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
def _circular_mean_deg(a: np.ndarray) -> float:
    if not len(a):
        return 0.0
    r = np.radians(a)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def generate_candidates(fit: ArmFit, options: Optional[SuggestOptions] = None) -> List[Suggestion]:
    """Geometrically valid candidate flange poses, scored and sorted best first (``q`` is None, ranks 1..n)."""
    opt = options or SuggestOptions()
    w, h = fit.image_size
    # --- rays through the cell centres (row-major cell index = row * cols + col)
    cu = (np.arange(opt.grid_cols) + 0.5) / opt.grid_cols * w
    cv_ = (np.arange(opt.grid_rows) + 0.5) / opt.grid_rows * h
    px = np.array([(u, v) for v in cv_ for u in cu], dtype=np.float64)
    xy = cv2.undistortPoints(px.reshape(-1, 1, 2), np.asarray(fit.K, dtype=np.float64),
                             _dist8(fit.dist)).reshape(-1, 2)
    rays = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    Rf = _ray_frames(rays)
    # --- orientation set (in the line-of-sight frame)
    sign = _facing_sign(fit)
    F = _facing_flip(sign)
    S = np.array([x.T_base_link6 for x in fit.samples])
    T_s = _board_poses_opt(fit, S)
    _, _, M_s = _decompose(T_s, F)
    spin0 = _circular_mean_deg(_spin_deg(M_s))
    turns, tips, spins = np.meshgrid(opt.turns_deg, opt.tips_deg, np.array(opt.spin_offsets_deg) + spin0, indexing="ij")
    M_set = Rotation.from_euler("YXZ", np.stack([turns.ravel(), tips.ravel(), spins.ravel()], axis=1),
                                degrees=True).as_matrix()                              # (m,3,3)
    # --- all combinations (ray, distance, orientation)
    nr, nd, nm = len(rays), len(opt.distances_m), len(M_set)
    ir, idd, im = np.meshgrid(np.arange(nr), np.arange(nd), np.arange(nm), indexing="ij")
    ir, idd, im = ir.ravel(), idd.ravel(), im.ravel()
    T_ob = np.tile(np.eye(4), (len(ir), 1, 1))
    T_ob[:, :3, :3] = Rf[ir] @ M_set[im] @ F
    T_ob[:, :3, 3] = rays[ir] * np.asarray(opt.distances_m)[idd][:, None]
    # --- cheap filters: height, reach (needs A), view angle
    A = inv_T(fit.X) @ T_ob @ inv_T(fit.Z)
    c_base = (inv_T(fit.X) @ T_ob)[:, :3, 3]
    keep = c_base[:, 2] >= opt.min_height_m
    keep &= np.linalg.norm(A[:, :3, 3], axis=1) <= opt.max_reach_m
    view = np.degrees(np.arccos(np.clip(sign * np.einsum("bi,bi->b", T_ob[:, :3, 2], rays[ir]), -1.0, 1.0)))
    keep &= (sign * np.einsum("bi,bi->b", T_ob[:, :3, 2], rays[ir]) > 0.0) & (view <= opt.max_view_angle_deg)
    A, T_ob = A[keep], T_ob[keep]
    if not len(A):
        return []
    # --- outline fully inside the image (with margin), in front of the camera
    uv_o, z_o = _project_batch(fit, T_ob, _outline_obj(fit))
    m = opt.image_margin_px
    ok = np.all(z_o > 0.05, axis=1)
    ok &= np.all((uv_o[:, :, 0] >= m) & (uv_o[:, :, 0] <= w - m) & (uv_o[:, :, 1] >= m) & (uv_o[:, :, 1] <= h - m),
                 axis=1)
    A, T_ob, uv_o = A[ok], T_ob[ok], uv_o[ok]
    if not len(A):
        return []
    uv_c, _ = _project_batch(fit, T_ob, np.zeros((1, 3)))
    t = _terms(fit, opt, A, T_ob, uv_o, uv_c[:, 0, :])
    sugg = _make_suggestions(fit, opt, A, uv_o, uv_c[:, 0, :], t)
    sugg.sort(key=lambda s: -s.score)
    for i, s in enumerate(sugg):
        s.rank = i + 1
    return sugg


# --------------------------------------------------------------------------
# IK / validity walk
# --------------------------------------------------------------------------
_JOINT_RE = re.compile(r"joint(\d+)$")


def _joint_map(q: Optional[Dict[str, float]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, v in (q or {}).items():
        m = _JOINT_RE.search(str(k))
        if m:
            out[int(m.group(1))] = float(v)
    return out


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def _too_close(fit: ArmFit, a: Suggestion, b: Suggestion, opt: SuggestOptions) -> bool:
    Ta, Tb = a.T_base_link6, b.T_base_link6
    if (float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])) < opt.min_separation_m
            and _rot_angle_deg(Ta[:3, :3], Tb[:3, :3]) < opt.min_separation_deg):
        return True
    px = math.hypot(a.centre_px[0] - b.centre_px[0], a.centre_px[1] - b.centre_px[1])
    if px < opt.min_separation_px_frac * fit.image_size[0]:
        Ra, Rb = (fit.X @ Ta @ fit.Z)[:3, :3], (fit.X @ Tb @ fit.Z)[:3, :3]
        return _rot_angle_deg(Ra, Rb) < opt.min_view_separation_deg
    return False


def suggest(fit: ArmFit,
            ik: Callable[[np.ndarray, Dict[str, float]], Optional[Dict[str, float]]],
            valid: Callable[[Dict[str, float]], Tuple[bool, str]],
            current_q: Dict[str, float], n: int = 5,
            options: Optional[SuggestOptions] = None) -> Tuple[List[Suggestion], Dict[str, int]]:
    """Walk the scored candidates best first and return up to ``n`` reachable, valid, new and mutually distinct ones.

    ``ik(T_base_link6, seed_q=current_q)`` -> joint dict or None; ``valid(q)`` -> (ok, reason).  Rejection order per
    candidate: too close to an accepted suggestion (no IK call spent), IK failure / exception, invalid, joint change
    from ``current_q`` above ``max_joint_change_deg``, joint goal within ``duplicate_joint_tol_deg`` (all joints) of an
    existing sample.  Stops after ``n`` accepted or ``max_ik_calls`` IK calls.  Joint names are matched by their
    trailing ``joint<N>``.

    Stats keys: candidates, ik_calls, accepted, too_close, ik_failed, ik_error, invalid, joint_change,
    duplicate_sample, max_ik_calls_reached (0/1), plus ``invalid: <reason>`` counts per validity reason."""
    opt = options or SuggestOptions()
    cands = generate_candidates(fit, opt)
    stats: Dict[str, int] = dict(candidates=len(cands), ik_calls=0, accepted=0, too_close=0, ik_failed=0, ik_error=0,
                                 invalid=0, joint_change=0, duplicate_sample=0, max_ik_calls_reached=0)
    cur = _joint_map(current_q)
    existing = [m for m in (_joint_map(s.joints) for s in fit.samples) if m]
    tol = math.radians(opt.duplicate_joint_tol_deg)
    accepted: List[Suggestion] = []
    for c in cands:
        if len(accepted) >= n:
            break
        if any(_too_close(fit, c, a, opt) for a in accepted):
            stats["too_close"] += 1
            continue
        if stats["ik_calls"] >= opt.max_ik_calls:
            stats["max_ik_calls_reached"] = 1
            break
        stats["ik_calls"] += 1
        try:
            q = ik(c.T_base_link6, seed_q=current_q)
        except Exception:  # noqa: BLE001 - an IK crash must not abort the walk
            stats["ik_error"] += 1
            continue
        if q is None:
            stats["ik_failed"] += 1
            continue
        try:
            ok, reason = valid(q)
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"validity check raised {type(exc).__name__}"
        if not ok:
            stats["invalid"] += 1
            key = f"invalid: {str(reason)[:60]}"
            stats[key] = stats.get(key, 0) + 1
            continue
        qm = _joint_map(q)
        common = [k for k in qm if k in cur]
        change = math.degrees(max((abs(qm[k] - cur[k]) for k in common), default=0.0))
        if change > opt.max_joint_change_deg:
            stats["joint_change"] += 1
            continue
        if any((lambda ks: bool(ks) and all(abs(qm[k] - e[k]) < tol for k in ks))([k for k in qm if k in e])
               for e in existing):
            stats["duplicate_sample"] += 1
            continue
        gains = dict(c.gains)
        gains["max_joint_change_deg"] = float(change)
        accepted.append(dataclasses.replace(c, q={k: float(v) for k, v in q.items()}, gains=gains))
        stats["accepted"] += 1
    for i, s in enumerate(accepted):
        s.rank = i + 1
    return accepted, stats
