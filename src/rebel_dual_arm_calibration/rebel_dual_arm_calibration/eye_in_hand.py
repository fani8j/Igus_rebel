"""Eye-in-hand (camera on the arm) hand-eye calibration: pure functions, no ROS.

Unknowns
    X = T_link6_optical   the rigid mount of the wrist camera's colour optical frame on link6   (what we want)
    B = T_base_board      the fixed table board in the arm base frame                          (nuisance, also reported)

Data (one WristSample per arm pose): A_i = T_base_link6 (TF at the image stamp), the ChArUco corners of the table board
(board-frame points ``obj``, pixels ``pix``) and the camera model K / distortion of the wrist camera.

Model: a board corner p (board frame) is seen at  project(K, dist, inv(A_i X) B p).

Solve
    1. per sample PnP (IPPE, both planar solutions kept, gross corner outliers trimmed) -> P_i = T_optical_board
    2. cv2.calibrateHandEye with Tsai, Park, Daniilidis: X_m; B_i = A_i X_m P_i must agree for all samples, so the method with the
       smallest board scatter is taken; the planar IPPE ambiguity is resolved against it (closed form = relative motions only)
    3. nonlinear least squares over (X, B) jointly (12 parameters), minimising the corner reprojection error with intrinsics FIXED,
       soft-L1 loss, marker corners weighted 1/2 against chessboard saddles.  GUARD: the arm's absolute positioning error (a few mm
       on the ReBeL) cannot be explained by X; the refinement absorbs it into a badly rotated X.  If X moved more than
       REFINE_MAX_MM / REFINE_MAX_DEG from the closed-form result the refinement is REJECTED (closed-form X, B kept, warning).  The
       same guarded procedure is used in every leave-one-out fold and in quick_estimate.
    4. diagnostics: reprojection RMS / max (informational), board pose scatter across samples, leave-one-out (held-out) board
       position, rotation diversity (span, distinct axes, planarity), agreement of the closed-form methods and the cross-check of B
       against the board pose measured by the static scene camera.

Gate (``GateLimits``; README has the table)
    PASS needs: board consistency <= 6 mm and <= 1 deg, held-out position <= 6 mm, >= 10 used samples, rotation span >= 30 deg, >= 3
    distinct axes, closed-form methods agree within 5 mm / 2 deg, scene cross-check <= 10 mm / 5 deg.  FAIL: reprojection RMS > 12 px,
    consistency > 10 mm or > 3 deg, held-out > 12 mm, < 6 samples, span < 15 deg, < 2 axes, cross-check > 25 mm or > 15 deg.
    Anything in between is WARN.  The absolute reprojection RMS is informational (it cannot reach 1 px on this arm).

Output frame: the Orbbec driver publishes wrist_camera_link -> ... -> wrist_camera_color_optical_frame, so the result is attached
as the joint link6 -> wrist_camera_link (X composed with the driver's transform); without that transform nothing is attached.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .solver import CHESS_ID_BASE, pnp_candidates
from .transforms import (inv_T, make_T, pose_to_T, R_to_quat, rotation_angle_deg, T_to_pose, T_to_xyz_rpy)

SCHEMA = "wrist_handeye/v1"
PARENT_FRAME = "link6"
CHILD_FRAME = "wrist_camera_link"            # what the result is attached to: the Orbbec driver publishes
                                             # wrist_camera_link -> ... -> wrist_camera_color_optical_frame (identity on the real camera)
OPTICAL_FRAME = "wrist_camera_color_optical_frame"
SAMPLES_FILE = "wrist_samples.jsonl"
SCENE_BOARD_FILE = "scene_board.json"
RESULT_YAML = "wrist_handeye_result.yaml"
RESULT_XACRO = "wrist_handeye.xacro.txt"
RESULT_REPORT = "wrist_handeye_report.json"
MARKER_WEIGHT = 0.5            # marker corners are ~2x noisier than chessboard saddles
MIN_CORNERS = 16               # per sample (spec)
MIN_SOLVE_SAMPLES = 4
HE_METHODS = (("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("PARK", cv2.CALIB_HAND_EYE_PARK),
              ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS))
# The joint reprojection refinement may absorb the ARM's absolute positioning error (several mm on the ReBeL) into a badly
# rotated X (real cell 2026-10-13: +37 mm / +49 deg for -1 px).  The closed-form AX = XB result uses relative motions only; the
# refinement is kept only if it stays this close to it, otherwise it is REJECTED (a warning is reported).
REFINE_MAX_MM = 5.0
REFINE_MAX_DEG = 2.0


class HandEyeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------
# data

@dataclass
class WristSample:
    index: int
    stamp: float
    T_base_link6: np.ndarray
    ids: np.ndarray
    obj: np.ndarray
    pix: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    image_size: Tuple[int, int]
    image: str = ""
    camera_frame_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T_base_link6 = np.asarray(self.T_base_link6, dtype=float).reshape(4, 4)
        self.ids = np.asarray(self.ids, dtype=int).reshape(-1)
        self.obj = np.asarray(self.obj, dtype=float).reshape(-1, 3)
        self.pix = np.asarray(self.pix, dtype=float).reshape(-1, 2)
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    def to_json_dict(self) -> dict:
        return {"index": int(self.index), "stamp": float(self.stamp), "T_base_link6": self.T_base_link6.tolist(),
                "ids": self.ids.tolist(), "obj": self.obj.tolist(), "pix": self.pix.tolist(), "K": self.K.tolist(),
                "dist": self.dist.tolist(), "image_size": list(self.image_size), "image": self.image,
                "camera_frame_id": self.camera_frame_id, "meta": self.meta}

    @classmethod
    def from_json_dict(cls, d: dict) -> "WristSample":
        return cls(index=int(d.get("index", 0)), stamp=float(d.get("stamp", 0.0)), T_base_link6=d["T_base_link6"],
                   ids=d["ids"], obj=d["obj"], pix=d["pix"], K=d["K"], dist=d.get("dist", []),
                   image_size=d["image_size"], image=str(d.get("image", "")),
                   camera_frame_id=str(d.get("camera_frame_id", "")), meta=dict(d.get("meta") or {}))


def append_sample(session_dir: str, sample: WristSample) -> None:
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, SAMPLES_FILE), "a") as f:
        f.write(json.dumps(sample.to_json_dict()) + "\n")


def rewrite_samples(session_dir: str, samples: Sequence[WristSample]) -> None:
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, SAMPLES_FILE)
    with open(path + ".tmp", "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json_dict()) + "\n")
    os.replace(path + ".tmp", path)


def load_samples(session_dir: str) -> List[WristSample]:
    path = os.path.join(session_dir, SAMPLES_FILE)
    out: List[WristSample] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(WristSample.from_json_dict(json.loads(line)))
    return out


def load_scene_board(session_dir: str) -> Optional[dict]:
    path = os.path.join(session_dir, SCENE_BOARD_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------
# geometry helpers

def project(T_opt_board: np.ndarray, obj: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rv = cv2.Rodrigues(np.ascontiguousarray(T_opt_board[:3, :3]))[0]
    uv, _ = cv2.projectPoints(np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3), rv, T_opt_board[:3, 3].reshape(3, 1),
                              np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1))
    return uv.reshape(-1, 2)


def _weights(ids: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(ids) >= CHESS_ID_BASE, 1.0, MARKER_WEIGHT)


def _T_opt_board(s: WristSample, X: np.ndarray, B: np.ndarray) -> np.ndarray:
    return inv_T(s.T_base_link6 @ X) @ B


def sample_residuals(s: WristSample, X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(N, 2) pixel residuals (predicted - measured)."""
    return project(_T_opt_board(s, X, B), s.obj, s.K, s.dist) - s.pix


def rms_of(res: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) if len(res) else float("nan")


def pose_delta(Ta: np.ndarray, Tb: np.ndarray) -> Tuple[float, float]:
    """(translation distance [m], rotation angle [deg]) between two poses."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])


def _pnp_subset(obj, pix, K, dist) -> List[Tuple[np.ndarray, float]]:
    class _S:       # pnp_candidates reads obj / pix / K / dist only
        pass
    o = _S()
    o.obj, o.pix, o.K, o.dist = obj, pix, K, dist
    return pnp_candidates(o)


def sample_pnp(s: WristSample) -> List[Tuple[np.ndarray, float]]:
    """PnP candidates [(T_optical_board, rms)] sorted by error (planar IPPE: up to two).  Gross corner outliers (a mis-detected
    corner) are trimmed: corners with a residual above max(4 px, 4 x median) of the first fit are dropped and the pose
    re-estimated (twice at most); the returned rms is over ALL corners."""
    cands = _pnp_subset(s.obj, s.pix, s.K, s.dist)
    if not cands:
        return cands
    keep = np.ones(len(s.pix), dtype=bool)
    for _ in range(2):
        res = np.linalg.norm(project(cands[0][0], s.obj, s.K, s.dist) - s.pix, axis=1)
        lim = max(4.0, 4.0 * float(np.median(res)))
        new = res <= lim
        if new.all() or new.sum() < 8 or (new == keep).all():
            break
        keep = new
        sub = _pnp_subset(s.obj[keep], s.pix[keep], s.K, s.dist)
        if not sub:
            break
        cands = sub
    if not keep.all():
        cands = sorted(((T, rms_of(project(T, s.obj, s.K, s.dist) - s.pix)) for T, _ in cands), key=lambda c: c[1])
    return cands


# ---------------------------------------------------------------------------------------------------
# closed-form initialisation

def _closed_form(A: List[np.ndarray], P: List[np.ndarray], method: int) -> np.ndarray:
    R_g2b = [a[:3, :3] for a in A]
    t_g2b = [a[:3, 3].reshape(3, 1) for a in A]
    R_t2c = [p[:3, :3] for p in P]
    t_t2c = [p[:3, 3].reshape(3, 1) for p in P]
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    return make_T(R, t.reshape(3))


def board_scatter(A: Sequence[np.ndarray], P: Sequence[np.ndarray], X: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """B_i = A_i X P_i must be the same pose for all i: (position RMS about the mean [m], rotation RMS [deg], B_mean)."""
    Bs = [a @ X @ p for a, p in zip(A, P)]
    ts = np.array([b[:3, 3] for b in Bs])
    tm = ts.mean(axis=0)
    # mean rotation by chordal L2 mean
    M = sum(b[:3, :3] for b in Bs) / len(Bs)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    pos = float(np.sqrt(np.mean(np.sum((ts - tm) ** 2, axis=1))))
    rot = float(np.sqrt(np.mean([rotation_angle_deg(Rm, b[:3, :3]) ** 2 for b in Bs])))
    return pos, rot, make_T(Rm, tm)


def _resolve_ambiguity(cands: List[List[Tuple[np.ndarray, float]]], A: List[np.ndarray], X: np.ndarray,
                       B: np.ndarray) -> List[np.ndarray]:
    """For each sample the PnP candidate closest to the model prediction inv(A X) B."""
    out = []
    for c, a in zip(cands, A):
        pred = inv_T(a @ X) @ B
        best = min(c, key=lambda ct: np.linalg.norm(ct[0][:3, 3] - pred[:3, 3])
                   + 0.05 * rotation_angle_deg(ct[0][:3, :3], pred[:3, :3]) / 90.0)
        out.append(best[0])
    return out


def closed_form_init(samples: Sequence[WristSample]) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, float]]:
    """(X0, B0, method name, {method: board position scatter mm}).  Raises HandEyeError."""
    X, B, name, scatters, _ = closed_form_all(samples)
    return X, B, name, scatters


def method_agreement(Xs: Dict[str, np.ndarray]) -> Optional[dict]:
    """Largest pairwise difference between the closed-form methods' X: {max_mm, max_deg}; None with fewer than two."""
    names = list(Xs)
    d = [pose_delta(Xs[a], Xs[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    if not d:
        return None
    return {"max_mm": max(x[0] for x in d) * 1e3, "max_deg": max(x[1] for x in d)}


def closed_form_all(samples: Sequence[WristSample]):
    """(X0, B0, best method, {method: scatter mm}, {method: X}) of the closed-form (AX = XB) hand-eye methods."""
    if len(samples) < 3:
        raise HandEyeError(f"need >= 3 samples for the closed-form hand-eye initialisation, got {len(samples)}")
    cands, keep = [], []
    for k, s in enumerate(samples):
        c = sample_pnp(s)
        if c:
            cands.append(c)
            keep.append(k)
    if len(keep) < 3:
        raise HandEyeError(f"PnP failed on too many samples ({len(keep)} usable)")
    A = [samples[k].T_base_link6 for k in keep]
    P = [c[0][0] for c in cands]
    best = None
    scatters: Dict[str, float] = {}
    for name, method in HE_METHODS:
        try:
            X = _closed_form(A, P, method)
            if not np.all(np.isfinite(X)):
                continue
            pos, rot, B = board_scatter(A, P, X)
        except cv2.error:
            continue
        score = pos + 0.002 * rot      # 1 deg ~ 2 mm for the choice only
        scatters[name] = pos * 1e3
        if best is None or score < best[0]:
            best = (score, name, X, B)
    if best is None:
        raise HandEyeError("cv2.calibrateHandEye failed for every method")
    _, name, X, B = best
    for _ in range(2):          # resolve the IPPE flip against the model, then recompute
        P = _resolve_ambiguity(cands, A, X, B)
        method = dict(HE_METHODS)[name]
        try:
            X = _closed_form(A, P, method)
        except cv2.error:
            break
        _, _, B = board_scatter(A, P, X)
    Xs = {}
    for n_, m_ in HE_METHODS:       # every method on the final (ambiguity-resolved) poses, for the agreement check
        try:
            Xm = _closed_form(A, P, m_)
            if np.all(np.isfinite(Xm)):
                Xs[n_] = Xm
        except cv2.error:
            pass
    return X, B, name, scatters, Xs


# ---------------------------------------------------------------------------------------------------
# nonlinear refinement

def refine(samples: Sequence[WristSample], X0: np.ndarray, B0: np.ndarray, f_scale: float = 1.0,
           max_nfev: int = 200) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Joint NLS over X and B (intrinsics fixed).  Returns (X, B, success)."""
    ws = [_weights(s.ids)[:, None] for s in samples]

    def res(x):
        X, B = pose_to_T(x[:6]), pose_to_T(x[6:])
        return np.concatenate([(sample_residuals(s, X, B) * w).ravel() for s, w in zip(samples, ws)])

    x0 = np.concatenate([T_to_pose(X0), T_to_pose(B0)])
    sol = least_squares(res, x0, method="trf", loss="soft_l1", f_scale=f_scale, max_nfev=max_nfev,
                        x_scale=np.concatenate([[0.01] * 3, [0.01] * 3, [0.01] * 3, [0.01] * 3]))
    ok = bool(sol.success and np.all(np.isfinite(sol.x)))
    return pose_to_T(sol.x[:6]), pose_to_T(sol.x[6:]), ok


# ---------------------------------------------------------------------------------------------------
# diagnostics

def rotation_diversity(samples: Sequence[WristSample], X: np.ndarray, min_pair_deg: float = 8.0,
                       cluster_deg: float = 20.0) -> dict:
    """Camera orientation spread and the distinct rotation axes of the relative camera rotations (base frame)."""
    Rs = [(s.T_base_link6 @ X)[:3, :3] for s in samples]
    n = len(Rs)
    span = 0.0
    axes: List[Tuple[float, np.ndarray]] = []
    for i in range(n):
        for j in range(i + 1, n):
            Rrel = Rs[j] @ Rs[i].T
            rv = cv2.Rodrigues(np.ascontiguousarray(Rrel))[0].reshape(3)
            ang = math.degrees(float(np.linalg.norm(rv)))
            span = max(span, ang)
            if ang >= min_pair_deg:
                a = rv / np.linalg.norm(rv)
                axes.append((ang, a))
    axes.sort(key=lambda t: -t[0])
    reps: List[np.ndarray] = []
    for _, a in axes:
        if not any(abs(float(a @ r)) > math.cos(math.radians(cluster_deg)) for r in reps):
            reps.append(a)
    M = sum(w * np.outer(a, a) for w, a in axes) if axes else np.zeros((3, 3))
    ev = np.sort(np.linalg.eigvalsh(M))[::-1] if axes else np.zeros(3)
    planarity = float(math.sqrt(max(ev[2], 0.0) / ev[0])) if ev[0] > 1e-12 else 0.0
    return {"span_deg": float(span), "n_axes": len(reps), "axis_planarity": planarity, "n_pairs": len(axes)}


def board_consistency(samples: Sequence[WristSample], X: np.ndarray, B: np.ndarray) -> dict:
    """Per-sample board pose from its own PnP (candidate nearest to the model) -> B_i = A_i X P_i vs the joint B."""
    cands = [sample_pnp(s) for s in samples]
    ok = [k for k, c in enumerate(cands) if c]
    if not ok:
        return {"position_rms_mm": None, "position_max_mm": None, "rotation_rms_deg": None, "rotation_max_deg": None,
                "per_sample_mm": []}
    P = _resolve_ambiguity([cands[k] for k in ok], [samples[k].T_base_link6 for k in ok], X, B)
    d = [pose_delta(samples[k].T_base_link6 @ X @ p, B) for k, p in zip(ok, P)]
    pos = np.array([x[0] for x in d]) * 1e3
    rot = np.array([x[1] for x in d])
    return {"position_rms_mm": float(np.sqrt(np.mean(pos ** 2))), "position_max_mm": float(pos.max()),
            "rotation_rms_deg": float(np.sqrt(np.mean(rot ** 2))), "rotation_max_deg": float(rot.max()),
            "per_sample_mm": [float(v) for v in pos]}


def scene_cross_check(B_wrist: np.ndarray, B_scene: Optional[np.ndarray]) -> Optional[dict]:
    if B_scene is None:
        return None
    dt, dr = pose_delta(B_wrist, B_scene)
    return {"position_mm": dt * 1e3, "rotation_deg": dr,
            "delta_base_mm": ((B_wrist[:3, 3] - B_scene[:3, 3]) * 1e3).tolist()}


# ---------------------------------------------------------------------------------------------------
# one fit (closed form + guarded refinement)

@dataclass
class _Fit:
    X: np.ndarray
    B: np.ndarray
    X0: np.ndarray
    B0: np.ndarray
    method: str
    scatters: Dict[str, float]
    agreement: Optional[dict]
    refine_delta: Optional[Tuple[float, float]]       # (mm, deg) the refinement moved X from the closed-form result
    refine_rejected: bool
    refine_converged: bool


def fit_once(pool: Sequence[WristSample], refine_nfev: int = 200) -> "_Fit":
    """Closed-form init, then the joint reprojection refinement - REJECTED (closed-form X, B kept) if it moves X more than
    REFINE_MAX_MM / REFINE_MAX_DEG from the closed-form result or does not converge."""
    X0, B0, method, scatters, Xs = closed_form_all(pool)
    X, B, ok = refine(pool, X0, B0, max_nfev=refine_nfev)
    delta = pose_delta(X0, X) if ok else None
    rejected = (not ok) or delta[0] * 1e3 > REFINE_MAX_MM or delta[1] > REFINE_MAX_DEG
    if rejected:
        X, B = X0, B0
    return _Fit(X, B, X0, B0, method, scatters, method_agreement(Xs), None if delta is None else (delta[0] * 1e3, delta[1]),
                rejected, ok)


def leave_one_out(samples: Sequence[WristSample], min_remaining: int = 4) -> dict:
    """Held-out error: refit WITHOUT sample i with the same guarded procedure as the full solve (closed form per fold, the
    refinement only if it stays close), then compare the board pose obtained from sample i's own PnP through X_-i with B_-i
    (position / rotation), and sample i's reprojection error predicted by (X_-i, B_-i)."""
    n = len(samples)
    empty = {"n": 0, "position_rms_mm": None, "position_max_mm": None, "rotation_rms_deg": None, "reproj_rms_px": None,
             "per_sample_mm": []}
    if n - 1 < min_remaining:
        return empty
    pos, rot, rp = [], [], []
    for i in range(n):
        rest = [s for k, s in enumerate(samples) if k != i]
        try:
            f = fit_once(rest, refine_nfev=60)
        except (HandEyeError, cv2.error, np.linalg.LinAlgError, ValueError):
            continue
        s = samples[i]
        c = sample_pnp(s)
        if not c:
            continue
        P = _resolve_ambiguity([c], [s.T_base_link6], f.X, f.B)[0]
        dt, dr = pose_delta(s.T_base_link6 @ f.X @ P, f.B)
        pos.append(dt * 1e3)
        rot.append(dr)
        rp.append(rms_of(sample_residuals(s, f.X, f.B)))
    if not pos:
        return empty
    pos, rot, rp = np.array(pos), np.array(rot), np.array(rp)
    return {"n": int(len(pos)), "position_rms_mm": float(np.sqrt(np.mean(pos ** 2))), "position_max_mm": float(pos.max()),
            "rotation_rms_deg": float(np.sqrt(np.mean(rot ** 2))),
            "reproj_rms_px": float(np.sqrt(np.mean(rp ** 2))), "per_sample_mm": [float(v) for v in pos]}


# ---------------------------------------------------------------------------------------------------
# gate

@dataclass(frozen=True)
class GateLimits:
    """The ReBeL's absolute positioning error (a few mm) cannot be explained by X: the absolute reprojection error is
    informational (FAIL only when huge) and the board pose consistency across the views is judged instead."""
    rms_px_info: float = 1.0           # above this: informational note only
    rms_px_fail: float = 12.0
    cons_mm_pass: float = 6.0
    cons_mm_fail: float = 10.0
    cons_deg_pass: float = 1.0
    cons_deg_fail: float = 3.0
    heldout_mm_pass: float = 6.0
    heldout_mm_fail: float = 12.0
    agree_mm: float = 5.0              # the closed-form methods must agree (WARN otherwise)
    agree_deg: float = 2.0
    min_samples_pass: int = 10
    min_samples_fail: int = 6
    span_deg_pass: float = 30.0
    span_deg_fail: float = 15.0
    min_axes_pass: int = 3
    min_axes_fail: int = 2
    planarity_warn: float = 0.10
    cross_mm_pass: float = 10.0
    cross_deg_pass: float = 5.0
    cross_mm_fail: float = 25.0
    cross_deg_fail: float = 15.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate_gate(stats: dict, limits: GateLimits = GateLimits()) -> dict:
    """{verdict: PASS|WARN|FAIL, reasons: [..]} from the stats dict of ``solve``."""
    fails, warns, notes = [], [], []
    n = int(stats.get("n_used", 0))
    rms = stats.get("rms_px")
    held = (stats.get("heldout") or {}).get("position_rms_mm")
    cons = stats.get("board_consistency") or {}
    div = stats.get("diversity") or {}
    cross = stats.get("scene_cross_check")
    agree = stats.get("method_agreement")

    def chk(name, val, pass_lim, fail_lim, unit, higher_is_worse=True, missing_warns=True):
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            if missing_warns:
                warns.append(f"{name}: not available")
            return
        bad = (lambda v, l: v > l) if higher_is_worse else (lambda v, l: v < l)
        if bad(val, fail_lim):
            fails.append(f"{name} {val:.2f} {unit} (fail limit {fail_lim:g})")
        elif bad(val, pass_lim):
            warns.append(f"{name} {val:.2f} {unit} (pass limit {pass_lim:g})")

    if rms is not None and math.isfinite(rms):
        if rms > limits.rms_px_fail:
            fails.append(f"reprojection RMS {rms:.2f} px (fail limit {limits.rms_px_fail:g})")
        elif rms > limits.rms_px_info:
            notes.append(f"reprojection RMS {rms:.2f} px > {limits.rms_px_info:g} (arm positioning accuracy; informational)")
    else:
        warns.append("reprojection RMS: not available")
    chk("board consistency (position)", cons.get("position_rms_mm"), limits.cons_mm_pass, limits.cons_mm_fail, "mm")
    chk("board consistency (rotation)", cons.get("rotation_rms_deg"), limits.cons_deg_pass, limits.cons_deg_fail, "deg")
    chk("held-out board position", held, limits.heldout_mm_pass, limits.heldout_mm_fail, "mm")
    chk("samples used", float(n), limits.min_samples_pass, limits.min_samples_fail, "", higher_is_worse=False)
    chk("rotation span", div.get("span_deg"), limits.span_deg_pass, limits.span_deg_fail, "deg", higher_is_worse=False)
    chk("distinct rotation axes", float(div.get("n_axes", 0)), limits.min_axes_pass, limits.min_axes_fail, "",
        higher_is_worse=False)
    if div.get("n_axes", 0) >= limits.min_axes_pass and div.get("axis_planarity", 1.0) < limits.planarity_warn:
        warns.append(f"rotation axes nearly coplanar (planarity {div.get('axis_planarity', 0):.2f} < "
                     f"{limits.planarity_warn:g}): the rotation of X is weakly observable")
    if agree is not None and (agree["max_mm"] > limits.agree_mm or agree["max_deg"] > limits.agree_deg):
        warns.append(f"closed-form methods disagree by {agree['max_mm']:.1f} mm / {agree['max_deg']:.1f} deg "
                     f"(limits {limits.agree_mm:g} mm / {limits.agree_deg:g} deg)")
    if cross is not None:
        if cross["position_mm"] > limits.cross_mm_fail or cross["rotation_deg"] > limits.cross_deg_fail:
            fails.append(f"scene-camera cross-check {cross['position_mm']:.1f} mm / {cross['rotation_deg']:.1f} deg "
                         f"(fail limit {limits.cross_mm_fail:g} mm / {limits.cross_deg_fail:g} deg)")
        elif cross["position_mm"] > limits.cross_mm_pass or cross["rotation_deg"] > limits.cross_deg_pass:
            warns.append(f"scene-camera cross-check {cross['position_mm']:.1f} mm / {cross['rotation_deg']:.1f} deg "
                         f"(pass limits {limits.cross_mm_pass:g} mm / {limits.cross_deg_pass:g} deg)")
    if stats.get("n_rejected"):
        warns.append(f"{stats['n_rejected']} sample(s) rejected as outliers")
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"
    return {"verdict": verdict, "reasons": fails + warns + notes, "limits": limits.to_dict()}


# ---------------------------------------------------------------------------------------------------
# the solve

@dataclass
class HandEyeResult:
    X: np.ndarray
    B: np.ndarray
    stats: dict
    gate: dict
    used: List[int]
    rejected: List[int]
    method: str
    per_sample_rms: List[float]
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        xyz, rpy = T_to_xyz_rpy(self.X)
        return {"xyz": xyz.tolist(), "rpy": rpy.tolist(), "verdict": self.gate["verdict"], "reasons": self.gate["reasons"],
                "warnings": list(self.warnings),
                "rms_px": self.stats.get("rms_px"), "n_used": self.stats.get("n_used"),
                "heldout_position_mm": (self.stats.get("heldout") or {}).get("position_rms_mm"),
                "span_deg": (self.stats.get("diversity") or {}).get("span_deg")}


def usable_samples(samples: Sequence[WristSample], min_corners: int = MIN_CORNERS) -> List[WristSample]:
    return [s for s in samples if len(s.ids) >= min_corners and np.all(np.isfinite(s.pix))
            and np.all(np.isfinite(s.T_base_link6))]


def solve(samples: Sequence[WristSample], T_base_board_scene: Optional[np.ndarray] = None,
          limits: GateLimits = GateLimits(), loo: bool = True, max_reject_frac: float = 0.3,
          X_start: Optional[np.ndarray] = None) -> HandEyeResult:
    """Full solve.  Raises HandEyeError when no solution can be formed (< 4 usable samples / init failure).
    ``X_start`` is accepted for compatibility and ignored (the closed form is the reference of the refinement guard)."""
    pool = usable_samples(samples)
    if len(pool) < MIN_SOLVE_SAMPLES:
        raise HandEyeError(f"need >= {MIN_SOLVE_SAMPLES} samples with >= {MIN_CORNERS} corners, have {len(pool)}")
    used = list(range(len(pool)))
    rejected: List[int] = []
    fit = fit_once(pool)
    for _ in range(max(1, int(max_reject_frac * len(pool)))):      # drop the clear outlier samples, one at a time
        cur = [pool[k] for k in used]
        if fit.refine_rejected:
            # arm positioning error makes the reprojection meaningless as a per-sample test: use the board pose scatter
            metric = np.array(board_consistency(cur, fit.X, fit.B)["per_sample_mm"])
            lim = max(25.0, 4.0 * float(np.median(metric))) if len(metric) == len(cur) else float("inf")
        else:
            metric = np.array([rms_of(sample_residuals(s, fit.X, fit.B)) for s in cur])
            lim = max(3.0, 4.0 * float(np.median(metric)))
        if not len(metric) == len(cur) or metric.max() <= lim or len(used) - 1 < MIN_SOLVE_SAMPLES:
            break
        rejected.append(used.pop(int(np.argmax(metric))))
        fit = fit_once([pool[k] for k in used])
    fitted = [pool[k] for k in used]
    X, B = fit.X, fit.B
    res_all = np.vstack([sample_residuals(s, X, B) for s in fitted])
    per_rms = [rms_of(sample_residuals(s, X, B)) for s in pool]
    warnings: List[str] = []
    if fit.refine_rejected:
        if fit.refine_delta is None:
            warnings.append(f"joint refinement did not converge: X = closed-form {fit.method} result")
        else:
            warnings.append(f"joint refinement REJECTED: it moved X by {fit.refine_delta[0]:.1f} mm / {fit.refine_delta[1]:.1f} deg "
                            f"from the closed-form {fit.method} result (limit {REFINE_MAX_MM:g} mm / {REFINE_MAX_DEG:g} deg), "
                            f"absorbing arm positioning error; X = closed-form {fit.method} result")
    stats = {
        "method_init": fit.method, "init_board_scatter_mm": fit.scatters, "method_agreement": fit.agreement,
        "refine_rejected": bool(fit.refine_rejected),
        "refine_delta_from_init": None if fit.refine_delta is None else {"mm": fit.refine_delta[0], "deg": fit.refine_delta[1]},
        "n_samples": len(samples), "n_pool": len(pool),
        "n_used": len(fitted), "n_rejected": len(rejected), "rejected_indices": [int(pool[k].index) for k in rejected],
        "n_corners": int(len(res_all)), "rms_px": rms_of(res_all),
        "max_px": float(np.max(np.linalg.norm(res_all, axis=1))),
        "diversity": rotation_diversity(fitted, X),
        "board_consistency": board_consistency(fitted, X, B),
        "heldout": leave_one_out(fitted) if loo else {"n": 0, "position_rms_mm": None},
        "scene_cross_check": scene_cross_check(B, T_base_board_scene),
        "mean_board_distance_m": float(np.mean([np.linalg.norm(_T_opt_board(s, X, B)[:3, 3]) for s in fitted])),
        "warnings": warnings,
    }
    gate = evaluate_gate(stats, limits)
    return HandEyeResult(X=X, B=B, stats=stats, gate=gate, used=[int(pool[k].index) for k in used],
                         rejected=[int(pool[k].index) for k in rejected], method=fit.method, per_sample_rms=per_rms,
                         warnings=warnings)


def quick_estimate(samples: Sequence[WristSample], X_start: Optional[np.ndarray] = None) -> Optional[HandEyeResult]:
    """Interim solve (no leave-one-out; same guarded closed-form / refinement) for the in-view prediction; None if it cannot
    be formed."""
    try:
        return solve(samples, loo=False)
    except (HandEyeError, cv2.error, np.linalg.LinAlgError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------
# outputs

def attach_transform(X_link6_optical: np.ndarray, T_camera_link_optical: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """T_link6_wrist_camera_link = X @ inv(T_wrist_camera_link_optical); None if that transform is unknown (fail closed)."""
    if T_camera_link_optical is None:
        return None
    T = np.asarray(T_camera_link_optical, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        return None
    return X_link6_optical @ inv_T(T)


def xacro_snippet(T_link6_camera_link: np.ndarray, arm: str, verdict: str, stamp: str) -> str:
    """URDF joint link6 -> wrist_camera_link (the optical frame hangs below it in the camera driver's own TF tree)."""
    xyz, rpy = T_to_xyz_rpy(T_link6_camera_link)
    return (f"<!-- {SCHEMA} arm={arm} {stamp} verdict={verdict}: LIVE-CHECK / REVIEW ONLY. Paste into the URDF only after the\n"
            f"     operator has accepted the result. The camera driver publishes {CHILD_FRAME} -> ... -> {OPTICAL_FRAME};\n"
            f"     the optical frame must NOT get a second parent. -->\n"
            f'<link name="{CHILD_FRAME}"/>\n'
            f'<joint name="{PARENT_FRAME}_to_{CHILD_FRAME}" type="fixed">\n'
            f'  <parent link="{PARENT_FRAME}"/>\n'
            f'  <child link="{CHILD_FRAME}"/>\n'
            f'  <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n'
            f"</joint>\n")


def result_document(res: HandEyeResult, arm: str, samples: Sequence[WristSample], camera: Optional[dict] = None,
                    T_base_board_scene: Optional[np.ndarray] = None, nominal: Optional[dict] = None,
                    created: Optional[str] = None, T_camera_link_optical: Optional[np.ndarray] = None) -> dict:
    T_att = attach_transform(res.X, T_camera_link_optical)
    q = None if T_att is None else R_to_quat(T_att[:3, :3])
    xyz, rpy = (None, None) if T_att is None else T_to_xyz_rpy(T_att)
    xyz_o, rpy_o = T_to_xyz_rpy(res.X)
    created = created or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    stamps = [s.stamp for s in samples if s.stamp]
    return {
        "schema": SCHEMA, "arm": arm, "parent_frame": PARENT_FRAME, "child_frame": CHILD_FRAME,
        "optical_frame": OPTICAL_FRAME,
        "xyz": None if xyz is None else [float(v) for v in xyz], "rpy": None if rpy is None else [float(v) for v in rpy],
        "quaternion_xyzw": None if q is None else [float(v) for v in q],
        "attach_note": ("xyz/rpy/quaternion = link6 -> wrist_camera_link (the joint to publish / paste)" if T_att is not None else
                        "UNKNOWN: TF wrist_camera_link -> optical frame was not available, so the link6 -> wrist_camera_link "
                        "joint is NOT given (fail closed); use T_link6_optical with the driver's transform"),
        "T_link6_wrist_camera_link": None if T_att is None else T_att.tolist(),
        "T_wrist_camera_link_optical": None if T_camera_link_optical is None else np.asarray(T_camera_link_optical).tolist(),
        "T_link6_optical": res.X.tolist(), "optical_xyz": [float(v) for v in xyz_o], "optical_rpy": [float(v) for v in rpy_o],
        "verdict": res.gate["verdict"], "gate": res.gate, "stats": res.stats, "warnings": list(res.warnings),
        "n_samples": len(samples), "n_used": res.stats["n_used"], "used_indices": res.used, "rejected_indices": res.rejected,
        "timestamps": {"created_utc": created, "first_sample_stamp": min(stamps) if stamps else None,
                       "last_sample_stamp": max(stamps) if stamps else None},
        "camera": camera or {},
        "board": {"T_base_board_wrist_solve": res.B.tolist(),
                  "T_base_board_scene": None if T_base_board_scene is None else np.asarray(T_base_board_scene).tolist()},
        "nominal_guess": nominal,
        "notes": ["Intrinsics held fixed (camera_info). Closed-form AX=XB result unless a refinement within "
                  f"{REFINE_MAX_MM:g} mm / {REFINE_MAX_DEG:g} deg was accepted (see warnings).",
                  "NOT applied to the URDF. 'Publish TF' only broadcasts link6 -> wrist_camera_link for a live check."],
    }


def write_outputs(out_dir: str, res: HandEyeResult, arm: str, samples: Sequence[WristSample], **kw) -> Dict[str, str]:
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    doc = result_document(res, arm, samples, **kw)
    out: Dict[str, str] = {}
    ypath = os.path.join(out_dir, RESULT_YAML)
    with open(ypath, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(doc)), f, sort_keys=False)
    out["yaml"] = ypath
    xpath = os.path.join(out_dir, RESULT_XACRO)
    if doc["T_link6_wrist_camera_link"] is not None:
        with open(xpath, "w") as f:
            f.write(xacro_snippet(np.array(doc["T_link6_wrist_camera_link"]), arm, res.gate["verdict"],
                                  doc["timestamps"]["created_utc"]))
        out["xacro"] = xpath
    elif os.path.exists(xpath):
        os.remove(xpath)            # never leave a snippet of an older result next to a result that has none
    rpath = os.path.join(out_dir, RESULT_REPORT)
    with open(rpath, "w") as f:
        json.dump(doc, f, indent=1)
    out["report"] = rpath
    return out


def format_summary(res: HandEyeResult) -> str:
    st = res.stats
    xyz, rpy = T_to_xyz_rpy(res.X)
    h = st.get("heldout") or {}
    c = st.get("board_consistency") or {}
    lines = [f"{res.gate['verdict']}: {st['n_used']}/{st['n_samples']} samples, reprojection RMS {st['rms_px']:.2f} px "
             f"(max {st['max_px']:.2f}; informational), init {res.method}",
             f"X = link6 -> optical: xyz [{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}] m, rpy "
             f"[{math.degrees(rpy[0]):.2f} {math.degrees(rpy[1]):.2f} {math.degrees(rpy[2]):.2f}] deg"]
    if c.get("position_rms_mm") is not None:
        lines.append(f"board consistency across samples: {c['position_rms_mm']:.2f} mm / {c['rotation_rms_deg']:.2f} deg RMS")
    if h.get("position_rms_mm") is not None:
        lines.append(f"held-out (leave-one-out) board position RMS {h['position_rms_mm']:.2f} mm "
                     f"(max {h['position_max_mm']:.2f}), reprojection {h['reproj_rms_px']:.2f} px")
    d = st["diversity"]
    lines.append(f"rotation span {d['span_deg']:.1f} deg, {d['n_axes']} distinct axes, planarity {d['axis_planarity']:.2f}")
    a = st.get("method_agreement")
    if a:
        lines.append(f"closed-form methods agree within {a['max_mm']:.1f} mm / {a['max_deg']:.2f} deg")
    cc = st.get("scene_cross_check")
    if cc:
        lines.append(f"board vs scene camera: {cc['position_mm']:.1f} mm / {cc['rotation_deg']:.2f} deg")
    lines += ["WARNING: " + w for w in res.warnings]
    lines += ["reason: " + r for r in res.gate["reasons"]]
    return "\n".join(lines)
