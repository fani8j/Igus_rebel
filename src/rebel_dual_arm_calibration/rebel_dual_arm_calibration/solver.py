"""Dual-arm eye-to-hand calibration solver (numpy / scipy / cv2 only, no ROS).

Model (all transforms are 4x4, ``T_a_b`` maps coordinates of frame b into frame a):

    T_O_target = inv(T_W_O) @ T_W_base(arm) @ T_base_link6 @ T_link6_target

  O          scene-camera optical frame (``camera_color_optical_frame``)
  W          world = ``arm_left_base_link``  (so T_W_base(left) = I)
  T_W_base(right)  unknown: pose of arm_right_base_link in world
  T_base_link6     measured (TF / forward kinematics) at every capture
  T_link6_target   unknown rigid mount of the printed target, one per arm

Pipeline: per-sample IPPE/PnP -> closed-form eye-to-hand (OpenCV
``calibrateRobotWorldHandEye``) per arm -> per-arm reprojection refinement ->
joint reprojection refinement of both arms (24 unknowns, robust loss) with
covariance from the Jacobian.  The joint result is converted to ``camera_link``
with the recorded ``T_camlink_optical``.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .targets import CHESS_ID_BASE
from .transforms import (R_to_rotvec, R_to_rpy, T_to_xyz_rpy, inv_T, kabsch, make_T, rotation_angle_deg,
                         rotvec_to_R, xyz_rpy_to_T)

ARMS = ("left", "right")


class SolverError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class Sample:
    arm: str                          # "left" | "right"
    T_base_link6: np.ndarray          # 4x4 pose of link6 in the arm base frame
    ids: np.ndarray                   # (N,) point ids
    obj: np.ndarray                   # (N,3) points in the target frame (m)
    pix: np.ndarray                   # (N,2) detected pixels
    K: np.ndarray                     # 3x3
    dist: np.ndarray                  # distortion coefficients
    image_size: Tuple[int, int]       # (width, height)
    T_camlink_optical: np.ndarray = field(default_factory=lambda: np.eye(4))
    joints: Dict[str, float] = field(default_factory=dict)
    stamp: float = 0.0
    image: str = ""
    sigma: Optional[np.ndarray] = None   # per-point noise factor (set by the solver)
    q: Optional[np.ndarray] = None       # joint values in chain order (for joint offsets); not serialised

    def __post_init__(self):
        self.arm = str(self.arm)
        self.T_base_link6 = np.asarray(self.T_base_link6, dtype=float).reshape(4, 4)
        self.ids = np.asarray(self.ids, dtype=int).reshape(-1)
        self.obj = np.asarray(self.obj, dtype=float).reshape(-1, 3)
        self.pix = np.asarray(self.pix, dtype=float).reshape(-1, 2)
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        self.T_camlink_optical = np.asarray(self.T_camlink_optical, dtype=float).reshape(4, 4)
        if self.sigma is None:
            self.sigma = np.ones(len(self.ids))

    def to_json_dict(self) -> dict:
        return dict(arm=self.arm, T_base_link6=self.T_base_link6.tolist(), joints=self.joints,
                    ids=self.ids.tolist(), obj=self.obj.tolist(), pix=self.pix.tolist(),
                    K=self.K.tolist(), dist=self.dist.tolist(), image_size=list(self.image_size),
                    T_camlink_optical=self.T_camlink_optical.tolist(), stamp=self.stamp, image=self.image)

    @staticmethod
    def from_json_dict(d: dict) -> "Sample":
        return Sample(arm=d["arm"], T_base_link6=d["T_base_link6"], ids=d["ids"], obj=d["obj"], pix=d["pix"],
                      K=d["K"], dist=d["dist"], image_size=tuple(d["image_size"]),
                      T_camlink_optical=d.get("T_camlink_optical", np.eye(4)), joints=d.get("joints", {}),
                      stamp=d.get("stamp", 0.0), image=d.get("image", ""))


@dataclass
class TableObservation:
    """Corners of the fixed table board detected in one image (not tied to an arm)."""
    ids: np.ndarray
    obj: np.ndarray                   # (N,3) board-frame points (m)
    pix: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    image_size: Tuple[int, int]
    T_camlink_optical: np.ndarray = field(default_factory=lambda: np.eye(4))
    stamp: float = 0.0
    image: str = ""
    sigma: Optional[np.ndarray] = None
    arm: str = "table"                # lets the camera-consistency check treat it like a sample

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=int).reshape(-1)
        self.obj = np.asarray(self.obj, dtype=float).reshape(-1, 3)
        self.pix = np.asarray(self.pix, dtype=float).reshape(-1, 2)
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        self.T_camlink_optical = np.asarray(self.T_camlink_optical, dtype=float).reshape(4, 4)
        if self.sigma is None:
            self.sigma = np.ones(len(self.ids))

    def to_json_dict(self) -> dict:
        return dict(ids=self.ids.tolist(), obj=self.obj.tolist(), pix=self.pix.tolist(), K=self.K.tolist(),
                    dist=self.dist.tolist(), image_size=list(self.image_size),
                    T_camlink_optical=self.T_camlink_optical.tolist(), stamp=self.stamp, image=self.image)

    @staticmethod
    def from_json_dict(d: dict) -> "TableObservation":
        return TableObservation(ids=d["ids"], obj=d["obj"], pix=d["pix"], K=d["K"], dist=d["dist"],
                                image_size=tuple(d["image_size"]),
                                T_camlink_optical=d.get("T_camlink_optical", np.eye(4)),
                                stamp=d.get("stamp", 0.0), image=d.get("image", ""))


@dataclass
class Touch:
    """The TCP of one arm resting on a known table-board point (arm stationary)."""
    arm: str
    point_id: int
    p_board: np.ndarray               # (3,) touched point in the board frame (m)
    T_base_tcp: np.ndarray            # 4x4 TCP pose in the arm base frame (TF)
    joints: Dict[str, float] = field(default_factory=dict)
    stamp: float = 0.0
    tcp_frame: str = ""
    q: Optional[np.ndarray] = None    # joint values in chain order; not serialised
    session: str = ""
    # "board": a printed crosshair of the table board (p_board known, used in the joint solve);
    # "fit": a free sticker (position unknown; used only in the touch-only Kabsch base);
    # "check": a free sticker that is NEVER used in any fit (held-out check point).
    # p_board of a sticker is a placeholder (zeros) and is never read.
    kind: str = "board"
    # contemporaneous TF base -> link6 and the link6 -> TCP transform in use when the touch was captured (recorder;
    # absent in old touches).  ``tcp_source`` (not serialised): "recorded" | "session_urdf" | "rebased" | "".
    T_base_link6: Optional[np.ndarray] = None
    T_link6_tcp: Optional[np.ndarray] = None
    tcp_source: str = ""

    def __post_init__(self):
        self.arm = str(self.arm)
        if self.T_base_link6 is not None:
            self.T_base_link6 = np.asarray(self.T_base_link6, dtype=float).reshape(4, 4)
        if self.T_link6_tcp is not None:
            self.T_link6_tcp = np.asarray(self.T_link6_tcp, dtype=float).reshape(4, 4)
        self.point_id = int(self.point_id)
        self.p_board = np.asarray(self.p_board, dtype=float).reshape(3)
        self.T_base_tcp = np.asarray(self.T_base_tcp, dtype=float).reshape(4, 4)
        self.kind = str(self.kind or "board")
        if self.kind not in ("board", "fit", "check"):
            raise ValueError(f"touch kind must be board|fit|check, got '{self.kind}'")

    def to_json_dict(self) -> dict:
        d = dict(arm=self.arm, point_id=self.point_id, p_board=self.p_board.tolist(),
                 T_base_tcp=self.T_base_tcp.tolist(), joints=self.joints, stamp=self.stamp,
                 tcp_frame=self.tcp_frame, kind=self.kind)
        if self.T_base_link6 is not None:
            d["T_base_link6"] = self.T_base_link6.tolist()
        if self.T_link6_tcp is not None:
            d["T_link6_tcp"] = self.T_link6_tcp.tolist()
        return d

    @staticmethod
    def from_json_dict(d: dict) -> "Touch":
        return Touch(arm=d["arm"], point_id=d["point_id"], p_board=d["p_board"], T_base_tcp=d["T_base_tcp"],
                     joints=d.get("joints", {}), stamp=d.get("stamp", 0.0), tcp_frame=d.get("tcp_frame", ""),
                     kind=d.get("kind", "board"), T_base_link6=d.get("T_base_link6"),
                     T_link6_tcp=d.get("T_link6_tcp"),
                     tcp_source="recorded" if d.get("T_link6_tcp") is not None else "")


# --------------------------------------------------------------------------
# TCP consistency and re-basing of touches (solve --tcp-file)
# --------------------------------------------------------------------------
TCP_X_TOL_M = 1e-5            # two link6 -> TCP transforms closer than this (m) / TCP_X_TOL_DEG are "the same TCP"
TCP_X_TOL_DEG = 1e-3
TCP_FILE_MAX_R_M = 0.5        # sanity limit of |r_tool0_m|
TCP_LINK6_MISMATCH_WARN_M = 5e-4     # warn if T_base_link6 @ X_old differs from the recorded T_base_tcp by more
# verdicts of a TCP result file that must not be applied (the exact vocabulary is owned by the TCP calibration tool)
BAD_TCP_VERDICTS = frozenset({"fail", "failed", "reject", "rejected", "insufficient", "unobservable", "bad", "poor",
                              "invalid", "error", "not_ok", "no_go", "nogo", "refused"})


def parse_tcp_file_args(items: Sequence[str]) -> Dict[str, str]:
    """``--tcp-file left=PATH[,right=PATH]`` (repeatable) -> {arm: path}."""
    import re
    out: Dict[str, str] = {}
    for item in items or []:
        for part in re.split(r",(?=(?:left|right)=)", item):
            arm, sep, path = part.partition("=")
            arm, path = arm.strip(), path.strip()
            if not sep or arm not in ARMS or not path:
                raise SolverError(f"--tcp-file expects left=PATH[,right=PATH], got '{part}'")
            if arm in out:
                raise SolverError(f"--tcp-file given twice for arm '{arm}'")
            out[arm] = path
    return out


def load_tcp_file(path: str, arm: str) -> dict:
    """Read a TCP result YAML: ``r_tool0_m`` (new igus_tool0 -> xeg32_tool_tip translation, 3 finite numbers in
    metres), ``arm`` (must match if present) and ``verdict`` (reported; a failing verdict is refused).  Nothing else
    is read."""
    import yaml
    try:
        with open(path) as f:
            d = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SolverError(f"--tcp-file {arm}: cannot read '{path}': {e}") from e
    if not isinstance(d, dict):
        raise SolverError(f"--tcp-file {arm}: '{path}' is not a YAML mapping")
    r = d.get("r_tool0_m")
    try:
        r = np.array(r, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        r = np.array([])
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise SolverError(f"--tcp-file {arm}: '{path}' needs r_tool0_m: [x, y, z] (3 finite numbers, metres)")
    if float(np.linalg.norm(r)) > TCP_FILE_MAX_R_M:
        raise SolverError(f"--tcp-file {arm}: |r_tool0_m| = {np.linalg.norm(r):.3f} m exceeds {TCP_FILE_MAX_R_M} m "
                          "(metres expected)")
    file_arm = d.get("arm")
    if file_arm is not None and str(file_arm) != arm:
        raise SolverError(f"--tcp-file {arm}: '{path}' is a TCP result for arm '{file_arm}'")
    verdict = d.get("verdict")
    if verdict is not None and str(verdict).strip().lower() in BAD_TCP_VERDICTS:
        raise SolverError(f"--tcp-file {arm}: '{path}' has verdict '{verdict}'; a TCP without an acceptable verdict "
                          "is not applied")
    return {"path": path, "arm": arm, "r_tool0_m": r, "verdict": None if verdict is None else str(verdict)}


def _x_diff(Xa: np.ndarray, Xb: np.ndarray) -> Tuple[float, float]:
    """(translation difference in m, rotation difference in deg) of two link6 -> TCP transforms."""
    return (float(np.linalg.norm(Xa[:3, 3] - Xb[:3, 3])), rotation_angle_deg(Xa[:3, :3], Xb[:3, :3]))


def prepare_touch_tcp(touches: Sequence["Touch"], kin: Optional[dict], tcp_files: Optional[Dict[str, dict]] = None):
    """Make the touches' TCP explicit and consistent.  Modifies ``touches`` in place when re-basing.

    Every touch carries ``T_link6_tcp`` (X_old): stored by the recorder, else the link6 -> TCP of its session URDF.

    * Without a TCP file for an arm: touches of that arm recorded with different X_old are REFUSED (a mixed
      data set would silently average two TCPs); nothing else changes.
    * With ``tcp_files[arm]`` (see :func:`load_tcp_file`): X_new = X_old with the igus_tool0 -> tip translation
      replaced by r_new (link6 -> igus_tool0 and all rotations kept) and
      ``T_base_tcp_new = T_base_tcp_old @ inv(X_old) @ X_new`` for every touch of that arm (any X_old); the touch
      is marked ``tcp_source = "rebased"``.  Refused if the X_old of a touch is unknown or the URDF chain is not
      link6 -> igus_tool0 -> tip.

    The FK used by the joint-offset path is patched to the TCP in use (X_new, or the common X_old), so that
    ``A = T_base_tcp @ inv(FK(q))`` stays a pure TF-vs-URDF residual.

    Returns ``(report, kin_new)``; ``report[arm]`` describes the TCP used, the touches and their shifts."""
    tcp_files = tcp_files or {}
    kin = dict(kin or {})
    report: Dict[str, dict] = {}
    for arm in ARMS:
        ts = [t for t in touches if t.arm == arm]
        tf = tcp_files.get(arm)
        if not ts:
            if tf:
                report[arm] = {"n_touches": 0, "source": "tcp_file (no touches of this arm: nothing re-based)",
                               "tcp_file": tf["path"], "verdict": tf["verdict"], "rebased": False}
            continue
        ak = kin.get(arm)
        known = [t for t in ts if t.T_link6_tcp is not None]
        unknown = [t for t in ts if t.T_link6_tcp is None]
        entry: dict = {"n_touches": len(ts), "n_unknown_tcp": len(unknown), "rebased": False, "warnings": []}
        report[arm] = entry
        # ---- the TCP the touches were recorded with ----------------------------------------------------------
        if known:
            ref = known[0].T_link6_tcp
            diffs = [_x_diff(t.T_link6_tcp, ref) for t in known]
            worst_t, worst_r = max(d[0] for d in diffs), max(d[1] for d in diffs)
            differing = worst_t > TCP_X_TOL_M or worst_r > TCP_X_TOL_DEG
        else:
            ref, differing, worst_t, worst_r = None, False, 0.0, 0.0
        split = None if ak is None else ak.tool_split()
        if differing and not tf:
            raise SolverError(
                f"arm {arm}: the touches were recorded with different TCPs (link6 -> TCP differs by up to "
                f"{worst_t * 1e3:.3f} mm / {worst_r:.3f} deg between touches, sessions: "
                f"{sorted({t.session or '?' for t in known})}).  Mixing them would average two TCPs.  Re-base all of "
                f"them onto one TCP with --tcp-file {arm}=<tcp result yaml>, or solve the sessions separately.")
        if unknown and tf:
            raise SolverError(
                f"arm {arm}: cannot re-base {len(unknown)} touch(es) (e.g. point {unknown[0].point_id}): the "
                "link6 -> TCP transform they were recorded with is neither stored in touches.jsonl nor available "
                "from the session robot_description.urdf")
        if ref is not None:
            entry["link6_tcp_xyz_m"] = [float(v) for v in ref[:3, 3]]
            entry["tcp_differs_between_touches"] = bool(differing)
            if split is not None:
                T1inv = np.linalg.inv(split[0])
                r_old = np.array([(T1inv @ t.T_link6_tcp)[:3, 3] for t in known])
                entry["r_tool0_tip_m"] = [float(v) for v in r_old[0]]
                entry["r_tool0_tip_spread_mm"] = float(np.max(np.linalg.norm(r_old - r_old[0], axis=1)) * 1e3)
        # stored transforms that disagree with the TF poses of the same touch
        bad = [t for t in known if t.T_base_link6 is not None and np.linalg.norm(
            (t.T_base_link6 @ t.T_link6_tcp)[:3, 3] - t.T_base_tcp[:3, 3]) > TCP_LINK6_MISMATCH_WARN_M]
        if bad:
            entry["warnings"].append(f"{len(bad)} touch(es) have T_base_link6 @ T_link6_tcp differing from T_base_tcp "
                                     f"by > {TCP_LINK6_MISMATCH_WARN_M * 1e3:.1f} mm (TF sampled at different times?)")
        # ---- re-base ------------------------------------------------------------------------------------------
        if tf:
            if split is None:
                raise SolverError(
                    f"arm {arm}: cannot re-base: the session URDF has no chain link6 -> igus_tool0 -> TCP "
                    "(fixed joints, the last starting at igus_tool0) to replace the igus_tool0 -> tip translation in")
            T1, _ = split
            T1inv = np.linalg.inv(T1)
            r_new = np.asarray(tf["r_tool0_m"], dtype=float)
            shifts, per = [], []
            for t in ts:
                X_old = t.T_link6_tcp
                T2 = T1inv @ X_old
                T2n = T2.copy()
                T2n[:3, 3] = r_new
                X_new = T1 @ T2n
                new = t.T_base_tcp @ np.linalg.inv(X_old) @ X_new
                shift = float(np.linalg.norm(new[:3, 3] - t.T_base_tcp[:3, 3]) * 1e3)
                shifts.append(shift)
                per.append({"point_id": t.point_id, "kind": t.kind, "session": t.session, "shift_mm": shift,
                            "r_tool0_tip_old_m": [float(v) for v in T2[:3, 3]]})
                t.T_base_tcp, t.T_link6_tcp, t.tcp_source = new, X_new, "rebased"
            entry.update(rebased=True, source="tcp_file", tcp_file=tf["path"], verdict=tf["verdict"],
                         r_tool0_tip_m_new=[float(v) for v in r_new],
                         shift_mm={"mean": float(np.mean(shifts)), "max": float(np.max(shifts)),
                                   "min": float(np.min(shifts)), "per_touch": per})
            ref = ts[0].T_link6_tcp
            entry["link6_tcp_xyz_m"] = [float(v) for v in ref[:3, 3]]
        else:
            entry["source"] = ("recorded with the touches (touches.jsonl)" if any(t.tcp_source == "recorded" for t in known)
                               else "session robot_description.urdf" if known else "unknown (no TCP stored, no URDF)")
        # ---- FK of the joint-offset path uses the TCP in use ---------------------------------------------------
        if ak is not None and ak.chain_tcp is not None and ak.chain_link6_tcp is not None and ref is not None \
                and not unknown:
            dt, dr = _x_diff(ref, ak.T_link6_tcp())
            if dt > TCP_X_TOL_M or dr > TCP_X_TOL_DEG:
                kin[arm] = ak.with_tcp_transform(ref)
                entry["fk_tcp_patched"] = True
                if not tf:
                    entry["warnings"].append(
                        f"the TCP of the touches differs from the session URDF by {dt * 1e3:.3f} mm / {dr:.3f} deg "
                        "(URDF saved before a TCP change?); the touches' TCP is used for the joint-offset FK")
    return report, kin


def format_tcp_report(report: Dict[str, dict]) -> List[str]:
    """Human-readable lines for :func:`prepare_touch_tcp`."""
    L = []
    for arm in ARMS:
        e = report.get(arm)
        if not e:
            continue
        if e.get("n_touches", 0) == 0:
            L.append(f"TCP {arm}: {e.get('source')}")
            continue
        x = e.get("link6_tcp_xyz_m")
        xs = "unknown" if x is None else "link6->TCP xyz [" + ", ".join(f"{v * 1e3:.3f}" for v in x) + "] mm"
        if e.get("rebased"):
            old = e.get("r_tool0_tip_m")
            r_old = "" if old is None else " (was [" + ", ".join(f"{v * 1e3:.3f}" for v in old) + "] mm)"
            sh = e["shift_mm"]
            L.append(f"TCP {arm}: RE-BASED with {e['tcp_file']} (verdict {e['verdict']}): igus_tool0->tip "
                     f"[" + ", ".join(f"{v * 1e3:.3f}" for v in e["r_tool0_tip_m_new"]) + f"] mm{r_old}, {xs}; "
                     f"{e['n_touches']} touches shifted by {sh['mean']:.3f} mm mean / {sh['max']:.3f} mm max")
        else:
            r = e.get("r_tool0_tip_m")
            rs = "" if r is None else ", igus_tool0->tip [" + ", ".join(f"{v * 1e3:.3f}" for v in r) + "] mm"
            L.append(f"TCP {arm}: as recorded ({e.get('source')}): {xs}{rs}; {e['n_touches']} touches"
                     + (f", {e['n_unknown_tcp']} without a known TCP" if e.get("n_unknown_tcp") else ""))
        for w in e.get("warnings", []):
            L.append(f"  WARNING TCP {arm}: {w}")
    return L


@dataclass
class SolverOptions:
    loss: str = "soft_l1"             # robust loss for the refinement
    f_scale_px: float = 1.0
    min_samples_warn: int = 20        # warn below this many distinct poses per arm
    recommended_samples: Tuple[int, int] = (30, 40)
    min_points: int = 4
    sample_outlier_px: float = 1.5    # floor for the sample-level outlier threshold (median residual)
    sample_outlier_factor: float = 5.0
    corner_outlier_px: float = 2.5
    corner_outlier_sigma: float = 5.0
    rot_diversity_warn_deg: float = 30.0
    pos_spread_warn_m: float = 0.05
    cond_warn: float = 1e5
    rms_warn_px: float = 1.0
    max_outer_iterations: int = 2
    # Marker corners are localised less precisely than the refined chessboard saddles
    # (ids >= CHESS_ID_BASE); they get this many times larger noise in the weighted fit.
    marker_sigma_ratio: float = 2.0
    # ---- loop closure, joint offsets, intrinsics, diagnostics -----------------------------------
    touch_sigma_m: float = 1e-3           # 1-sigma of one touch coordinate (m)
    nominal_px_sigma: float = 0.3         # px: scale that converts touch / prior residuals to the pixel scale
    joint_offsets: bool = False           # estimate 6 joint zero offsets per arm (needs FK from the URDF)
    offset_prior_deg: float = 0.3         # Gaussian prior sigma of every joint offset
    refine_intrinsics: bool = False       # estimate fx, fy, cx, cy (with priors)
    intrinsics_prior_frac: float = 0.01   # prior sigma of fx, fy as a fraction
    intrinsics_prior_px: float = 5.0      # prior sigma of cx, cy
    corr_warn: float = 0.8                # |correlation| above this is reported
    max_table_obs: int = 40               # evenly thinned if more table-board images are recorded
    touch_warn_factor: float = 2.5        # warn if touch RMS > this x touch_sigma
    intrinsics_report: bool = True        # when intrinsics are fixed: report how far the data would pull them
    loop_closure_variants: bool = True    # extra fits for the out-of-fit touch check
    keep_px_margin: float = 0.03          # joint offsets: held-out reprojection must improve by at least this fraction
    keep_touch_margin: float = 0.10       # ... and the held-out touch mismatch by at least this fraction
    obs_warn: float = 0.03                # warn if the smallest normalised data-only singular value is below this
    intrinsics_source: str = "camera_info"   # "file" when --intrinsics-file overrides camera_info (gate: intrinsics validated)
    intrinsics_file: str = ""


# --------------------------------------------------------------------------
# Projection and PnP
# --------------------------------------------------------------------------
def project_points(K: np.ndarray, dist: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project (N,3) camera-frame points with the OpenCV model (k1,k2,p1,p2,k3,k4,k5,k6)."""
    z = P[:, 2]
    x, y = P[:, 0] / z, P[:, 1] / z
    d = np.zeros(8)
    d[: min(8, len(dist))] = dist[:8]
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)


def _transform_points(T: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return obj @ T[:3, :3].T + T[:3, 3]


def reproj_rms(K, dist, T_O_target, obj, pix) -> float:
    P = _transform_points(T_O_target, obj)
    if np.any(P[:, 2] <= 1e-6):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((project_points(K, dist, P) - pix) ** 2, axis=1))))


def reproj_median(K, dist, T_O_target, obj, pix) -> float:
    """Median pixel error: robust sample score that ignores a few bad corners."""
    P = _transform_points(T_O_target, obj)
    if np.any(P[:, 2] <= 1e-6):
        return float("inf")
    return float(np.median(np.linalg.norm(project_points(K, dist, P) - pix, axis=1)))


def pnp_candidates(s: Sample) -> List[Tuple[np.ndarray, float]]:
    """Return PnP pose candidates [(T_O_target, rms_px)] sorted by error.

    Planar targets give the two IPPE solutions (the ambiguous flip is kept as the
    second candidate so it can be resolved against the hand-eye model later).
    """
    obj = s.obj.astype(np.float64).reshape(-1, 1, 3)
    pix = s.pix.astype(np.float64).reshape(-1, 1, 2)
    dist = s.dist if len(s.dist) else np.zeros(5)
    cands = []
    planar = np.allclose(s.obj[:, 2], s.obj[0, 2], atol=1e-9)
    try:
        if planar:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, pix, s.K, dist, flags=cv2.SOLVEPNP_IPPE)
        else:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, pix, s.K, dist, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return []
    for rv, tv in zip(rvecs, tvecs):
        try:
            rv, tv = cv2.solvePnPRefineLM(obj, pix, s.K, dist, rv, tv)
        except cv2.error:
            pass
        T = make_T(cv2.Rodrigues(rv)[0], tv.reshape(3))
        if T[2, 3] <= 0:
            continue
        cands.append((T, reproj_rms(s.K, dist, T, s.obj, s.pix)))
    cands.sort(key=lambda c: c[1])
    return cands


# --------------------------------------------------------------------------
# Closed-form initialisation
# --------------------------------------------------------------------------
def _hand_eye_closed_form(A: List[np.ndarray], P: List[np.ndarray], method: int):
    """A: T_base_link6 list, P: PnP T_O_target list.  Returns (T_O_base, T_link6_target)."""
    R_w2c = [p[:3, :3] for p in P]
    t_w2c = [p[:3, 3].reshape(3, 1) for p in P]
    R_b2g = [a[:3, :3] for a in A]
    t_b2g = [a[:3, 3].reshape(3, 1) for a in A]
    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(R_w2c, t_w2c, R_b2g, t_b2g, method=method)
    T_target_link6 = make_T(R_b2w, t_b2w.reshape(3))
    return make_T(R_g2c, t_g2c.reshape(3)), inv_T(T_target_link6)


def _predict_pose(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return X @ A @ Z


def _pose_dist(Ta: np.ndarray, Tb: np.ndarray) -> Tuple[float, float]:
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])


def closed_form_init(samples: Sequence[Sample], cands: Sequence[List[Tuple[np.ndarray, float]]]):
    """Per-arm closed-form init with ambiguity resolution and gross-outlier rejection.

    Returns (X, Z, use_mask) with X = T_O_base, Z = T_link6_target.
    """
    n = len(samples)
    if n < 3:
        raise SolverError(f"need >= 3 samples with a valid PnP for the closed-form init, got {n}")
    A = [s.T_base_link6 for s in samples]
    choice = [0] * n
    use = np.ones(n, bool)
    best = None
    for it in range(4):
        idx = np.flatnonzero(use)
        if len(idx) < 3:
            raise SolverError("closed-form init: too few consistent samples")
        P = [cands[i][choice[i]][0] for i in idx]
        sol = None
        for method in (cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI):
            try:
                X, Z = _hand_eye_closed_form([A[i] for i in idx], P, method)
            except cv2.error:
                continue
            cost = 0.0
            for i in idx:
                Tp = _predict_pose(X, A[i], Z)
                cost += reproj_median(samples[i].K, samples[i].dist, Tp, samples[i].obj, samples[i].pix) ** 2
            if sol is None or cost < sol[2]:
                sol = (X, Z, cost)
        if sol is None:
            raise SolverError("calibrateRobotWorldHandEye failed")
        X, Z, _ = sol
        best = (X, Z)
        # resolve IPPE ambiguity against the model; flag samples that disagree with both candidates
        errs = np.zeros(n)
        for i in range(n):
            Tp = _predict_pose(X, A[i], Z)
            d = [reproj_median(samples[i].K, samples[i].dist, Tp, samples[i].obj, samples[i].pix)]
            dd = [_pose_dist(Tp, c[0]) for c in cands[i]]
            score = [p / 0.05 + r / 10.0 for p, r in dd]  # 50 mm ~ 10 deg
            choice[i] = int(np.argmin(score))
            errs[i] = d[0]
        med = float(np.median(errs[use])) if use.any() else 0.0
        thr = max(10.0, 4.0 * med)
        new_use = errs <= thr
        if new_use.sum() < 3:
            break
        if np.array_equal(new_use, use) and it >= 1:
            break
        use = new_use
    return best[0], best[1], use


# --------------------------------------------------------------------------
# Reprojection model and refinement
# --------------------------------------------------------------------------
def _delta_to_T(T0: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """T0 @ exp(delta): local (right-multiplied) perturbation, delta = [rotvec, trans]."""
    return T0 @ make_T(rotvec_to_R(delta[:3]), delta[3:6])


class _ArmModel:
    """Single-arm model: params [dX(6), dZ(6)], X = T_O_base, Z = T_link6_target."""

    def __init__(self, samples: Sequence[Sample], X0, Z0):
        self.samples = list(samples)
        self.base = [np.asarray(X0), np.asarray(Z0)]
        self.n_params = 12

    def poses(self, theta):
        return _delta_to_T(self.base[0], theta[0:6]), _delta_to_T(self.base[1], theta[6:12])

    def target_poses(self, theta):
        X, Z = self.poses(theta)
        return [X @ s.T_base_link6 @ Z for s in self.samples]

    def residuals(self, theta: np.ndarray, mask: Optional[List[np.ndarray]] = None) -> np.ndarray:
        out = []
        for k, (s, T) in enumerate(zip(self.samples, self.target_poses(theta))):
            P = _transform_points(T, s.obj)
            if np.any(P[:, 2] <= 1e-6):
                r = np.full((len(s.obj), 2), 1e3)
            else:
                r = (project_points(s.K, s.dist, P) - s.pix) / s.sigma[:, None]
            if mask is not None:
                r = r[mask[k]]
            out.append(r.reshape(-1))
        return np.concatenate(out)


def _numeric_jacobian(fun, x0: np.ndarray, step: float = 1e-6) -> np.ndarray:
    f0 = fun(x0)
    J = np.empty((len(f0), len(x0)))
    for j in range(len(x0)):
        dx = np.zeros_like(x0)
        dx[j] = step
        J[:, j] = (fun(x0 + dx) - fun(x0 - dx)) / (2 * step)
    return J


def _fit(model, masks, options: SolverOptions, max_nfev: int = 200):
    x0 = np.zeros(model.n_params)
    res = least_squares(lambda th: model.residuals(th, masks), x0, method="trf", loss=options.loss,
                        f_scale=options.f_scale_px, x_scale="jac", diff_step=1e-7, max_nfev=max_nfev,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return res.x, res


def _residual_norms(model, theta, masks=None) -> List[np.ndarray]:
    out = []
    for s, T in zip(model.samples, model.target_poses(theta)):
        P = _transform_points(T, s.obj)
        if np.any(P[:, 2] <= 1e-6):
            out.append(np.full(len(s.obj), 1e3))
        else:
            out.append(np.linalg.norm(project_points(s.K, s.dist, P) - s.pix, axis=1))
    return out


def _flag_outliers(model, theta, options: SolverOptions):
    """Return (sample_ok mask, corner mask list) using robust sample and corner statistics."""
    return _flag_from_norms(_residual_norms(model, theta), options)


def _flag_from_norms(norms, options: SolverOptions):
    allv = np.concatenate(norms)
    sigma = max(float(np.median(allv)) / 1.1774, 0.05)
    sample_score = np.array([np.median(n) for n in norms])
    med_score = float(np.median(sample_score))
    s_thr = max(options.sample_outlier_px, options.sample_outlier_factor * med_score)
    sample_ok = sample_score <= s_thr
    c_thr = max(options.corner_outlier_px, options.corner_outlier_sigma * sigma)
    corner_masks = [(n <= c_thr) for n in norms]
    # a sample with too few remaining corners is dropped
    for i, m in enumerate(corner_masks):
        if m.sum() < 4:
            sample_ok[i] = False
    return sample_ok, corner_masks


# --------------------------------------------------------------------------
# Full joint model: camera, right base, target mounts, table board, joint offsets, intrinsics
# --------------------------------------------------------------------------
POSE_AXES = ("rx", "ry", "rz", "x", "y", "z")     # local perturbation [rotvec, translation]


@dataclass
class Params:
    """Absolute parameter values (the solution)."""
    T_W_O: np.ndarray                          # optical frame in world
    T_W_R: np.ndarray                          # arm_right_base in world
    Z: Dict[str, np.ndarray]                   # T_link6_target per arm
    T_W_B: Optional[np.ndarray] = None         # table board in world
    off: Dict[str, np.ndarray] = field(default_factory=lambda: {a: np.zeros(6) for a in ARMS})
    dK: np.ndarray = field(default_factory=lambda: np.zeros(4))   # added to fx, fy, cx, cy

    def copy(self) -> "Params":
        return Params(self.T_W_O.copy(), self.T_W_R.copy(), {a: z.copy() for a, z in self.Z.items()},
                      None if self.T_W_B is None else self.T_W_B.copy(),
                      {a: o.copy() for a, o in self.off.items()}, self.dK.copy())


@dataclass(frozen=True)
class Flags:
    board: bool = False        # estimate T_W_B (needs table observations and/or touches)
    offsets: bool = False
    intr: bool = False
    touches: bool = False      # touch residuals are part of the fit


def _project_arrays(fx, fy, cx, cy, d8, P):
    z = P[:, 2]
    x, y = P[:, 0] / z, P[:, 1] / z
    k1, k2, p1, p2, k3, k4, k5, k6 = (d8[:, i] for i in range(8))
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def _dist8(dist: np.ndarray) -> np.ndarray:
    d = np.zeros(8)
    d[: min(8, len(dist))] = dist[:8]
    return d


class _TouchBlock:
    """Vectorised touch residuals of a list of touches."""

    def __init__(self, touches: Sequence[Touch], kin, need_fk: bool):
        self.touches = list(touches)
        self.idx = {a: np.array([i for i, t in enumerate(self.touches) if t.arm == a], dtype=int) for a in ARMS}
        self.p_board = np.array([t.p_board for t in self.touches]).reshape(-1, 3)
        self.Trec = np.array([t.T_base_tcp for t in self.touches]).reshape(-1, 4, 4)
        self.A, self.q, self.chain = {}, {}, {}
        if need_fk and self.touches:
            for a in ARMS:
                ii = self.idx[a]
                if not len(ii):
                    continue
                ak = None if kin is None else kin.get(a)
                if ak is None or ak.chain_tcp is None:
                    raise SolverError(f"joint offsets need the TCP frame in the URDF of arm '{a}' (touch samples)")
                if any(self.touches[i].q is None for i in ii):
                    raise SolverError(f"touch joint values of arm '{a}' do not match the URDF joint names")
                q = np.array([self.touches[i].q for i in ii])
                self.q[a], self.chain[a] = q, ak.chain_tcp
                self.A[a] = self.Trec[ii] @ np.linalg.inv(ak.chain_tcp.fk(q))

    def tip_base(self, off: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """TCP position in the arm base frame, (n,3); joint offsets applied if ``off`` is given."""
        out = self.Trec[:, :3, 3].copy()
        if off is not None:
            for a, ii in self.idx.items():
                if len(ii):
                    out[ii] = (self.A[a] @ self.chain[a].fk(self.q[a] + off[a]))[:, :3, 3]
        return out

    def mismatch(self, params: Params, use_offsets: bool) -> np.ndarray:
        """TCP position in world minus table-board point in world (m), (n,3)."""
        if not len(self.touches):
            return np.zeros((0, 3))
        tip = self.tip_base(params.off if use_offsets else None)
        R, t = params.T_W_R[:3, :3], params.T_W_R[:3, 3]
        right = self.idx["right"]
        if len(right):
            tip[right] = tip[right] @ R.T + t
        board = self.p_board @ params.T_W_B[:3, :3].T + params.T_W_B[:3, 3]
        return tip - board


class _FullModel:
    """Parameter layout: [optical(6), right base(6), Z_left(6), Z_right(6)] (+ board(6)) (+ offsets 2..6 of L, R)
    (+ dfx, dfy, dcx, dcy).  Poses are perturbed by right-multiplied local increments around ``base``;
    offsets and intrinsics deltas are added to the base values."""

    def __init__(self, samples: Sequence[Sample], tables: Sequence[TableObservation], touches: Sequence[Touch],
                 base: Params, flags: Flags, options: SolverOptions, kin=None):
        self.samples, self.tables = list(samples), list(tables)
        self.items = self.samples + self.tables
        self.n_s = len(self.samples)
        self.options = options
        self.touch_block = _TouchBlock(touches if flags.touches else [], kin, flags.offsets)
        if flags.board and not self.tables and not len(self.touch_block.touches):
            flags = Flags(False, flags.offsets, flags.intr, flags.touches)
        self.flags = flags
        self.base = base.copy()
        if flags.board and self.base.T_W_B is None:
            raise SolverError("internal: board parameters requested without an initial board pose")
        self.sl, names, n = {}, [], 0
        for key, label, k in (("O", "optical", 6), ("R", "right_base", 6), ("Zl", "target_left", 6),
                              ("Zr", "target_right", 6)):
            self.sl[key] = slice(n, n + k)
            names += [f"{label}.{c}" for c in POSE_AXES]
            n += k
        if flags.board:
            self.sl["B"] = slice(n, n + 6)
            names += [f"table_board.{c}" for c in POSE_AXES]
            n += 6
        if flags.offsets:
            for a in ARMS:      # joint 1 is not separable from the base yaw: fixed (gauge), joints 2..6 estimated
                self.sl["off_" + a] = slice(n, n + 5)
                names += [f"offset_{a}.joint{i + 1}" for i in range(1, 6)]
                n += 5
        if flags.intr:
            self.sl["K"] = slice(n, n + 4)
            names += ["intrinsics.fx", "intrinsics.fy", "intrinsics.cx", "intrinsics.cy"]
            n += 4
        self.n_params, self.names = n, names
        # ---- pixel data, concatenated over items ----
        self.starts = np.cumsum([0] + [len(it.ids) for it in self.items])
        M = int(self.starts[-1])
        self.pidx = np.repeat(np.arange(len(self.items)), [len(it.ids) for it in self.items])
        self.obj = np.concatenate([it.obj for it in self.items]) if self.items else np.zeros((0, 3))
        self.pix = np.concatenate([it.pix for it in self.items]) if self.items else np.zeros((0, 2))
        self.sig = np.concatenate([it.sigma for it in self.items]) if self.items else np.zeros(0)
        self.K0 = np.array([[it.K[0, 0], it.K[1, 1], it.K[0, 2], it.K[1, 2]] for it in self.items]).reshape(-1, 4)
        self.d8 = np.array([_dist8(it.dist) for it in self.items]).reshape(-1, 8)
        assert M == len(self.pix)
        # ---- per-arm sample data ----
        self.arm_idx = {a: np.array([i for i, s in enumerate(self.samples) if s.arm == a], dtype=int) for a in ARMS}
        self.Trec = {a: np.array([self.samples[i].T_base_link6 for i in self.arm_idx[a]]).reshape(-1, 4, 4)
                     for a in ARMS}
        self.A, self.q, self.chain = {}, {}, {}
        if flags.offsets:
            for a in ARMS:
                ii = self.arm_idx[a]
                if not len(ii):
                    continue
                ak = None if kin is None else kin.get(a)
                if ak is None:
                    raise SolverError(f"joint offsets need the URDF kinematics of arm '{a}' "
                                      "(record with the current recorder, which saves robot_description.urdf)")
                if any(self.samples[i].q is None for i in ii):
                    raise SolverError(f"sample joint values of arm '{a}' do not match the URDF joint names")
                q = np.array([self.samples[i].q for i in ii])
                self.q[a], self.chain[a] = q, ak.chain6
                self.A[a] = self.Trec[a] @ np.linalg.inv(ak.chain6.fk(q))

    # ---- parameters -----------------------------------------------------------------------
    def params(self, theta: np.ndarray) -> Params:
        b, sl = self.base, self.sl
        out = Params(_delta_to_T(b.T_W_O, theta[sl["O"]]), _delta_to_T(b.T_W_R, theta[sl["R"]]),
                     {"left": _delta_to_T(b.Z["left"], theta[sl["Zl"]]),
                      "right": _delta_to_T(b.Z["right"], theta[sl["Zr"]])},
                     b.T_W_B, {a: b.off[a].copy() for a in ARMS}, b.dK.copy())
        if self.flags.board:
            out.T_W_B = _delta_to_T(b.T_W_B, theta[sl["B"]])
        if self.flags.offsets:
            for a in ARMS:
                out.off[a] = b.off[a].copy()
                out.off[a][1:] += theta[sl["off_" + a]]
        if self.flags.intr:
            out.dK = b.dK + theta[sl["K"]]
        return out

    def poses(self, theta):
        p = self.params(theta)
        return p.T_W_O, p.T_W_R, p.Z["left"], p.Z["right"]

    def base_link6(self, params: Params) -> Dict[str, np.ndarray]:
        out = {}
        for a in ARMS:
            if not len(self.arm_idx[a]):
                continue
            out[a] = (self.A[a] @ self.chain[a].fk(self.q[a] + params.off[a])) if self.flags.offsets else self.Trec[a]
        return out

    def item_poses_p(self, params: Params) -> np.ndarray:
        T = np.zeros((len(self.items), 4, 4))
        T_O_W = inv_T(params.T_W_O)
        X = {"left": T_O_W, "right": T_O_W @ params.T_W_R}
        Tb = self.base_link6(params)
        for a in ARMS:
            ii = self.arm_idx[a]
            if len(ii):
                T[ii] = X[a] @ Tb[a] @ params.Z[a]
        if self.tables:
            T[self.n_s:] = T_O_W @ params.T_W_B
        return T

    def target_poses(self, theta: np.ndarray) -> List[np.ndarray]:
        return list(self.item_poses_p(self.params(theta))[: self.n_s])

    # ---- residuals --------------------------------------------------------------------------
    def pixel_proj(self, params: Params) -> Tuple[np.ndarray, np.ndarray]:
        """(projected pixels (M,2), bad (M,) bool for points behind the camera)."""
        T = self.item_poses_p(params)[self.pidx]
        P = np.einsum("mij,mj->mi", T[:, :3, :3], self.obj) + T[:, :3, 3]
        bad = P[:, 2] <= 1e-6
        P = np.where(bad[:, None], np.array([0.0, 0.0, 1.0]), P)
        K = self.K0[self.pidx] + params.dK
        return _project_arrays(K[:, 0], K[:, 1], K[:, 2], K[:, 3], self.d8[self.pidx], P), bad

    def pixel_residuals(self, params: Params) -> np.ndarray:
        uv, bad = self.pixel_proj(params)
        r = (uv - self.pix) / self.sig[:, None]
        r[bad] = 1e3
        return r

    def pixel_norm_list(self, theta: np.ndarray) -> List[np.ndarray]:
        uv, bad = self.pixel_proj(self.params(theta))
        n = np.linalg.norm(uv - self.pix, axis=1)
        n[bad] = 1e3
        return [n[self.starts[i]:self.starts[i + 1]] for i in range(len(self.items))]

    def touch_residuals(self, params: Params) -> np.ndarray:
        m = self.touch_block.mismatch(params, self.flags.offsets)
        return (m / self.options.touch_sigma_m * self.options.nominal_px_sigma).reshape(-1)

    def prior_residuals(self, params: Params) -> np.ndarray:
        o, out = self.options, []
        if self.flags.offsets:
            sig = math.radians(o.offset_prior_deg)
            out += [params.off[a][1:] / sig * o.nominal_px_sigma for a in ARMS]
        if self.flags.intr:
            K0 = self.K0[0] if len(self.K0) else np.array([1000.0, 1000.0, 0.0, 0.0])
            sig = np.array([o.intrinsics_prior_frac * K0[0], o.intrinsics_prior_frac * K0[1],
                            o.intrinsics_prior_px, o.intrinsics_prior_px])
            out.append(params.dK / sig * o.nominal_px_sigma)
        return np.concatenate(out) if out else np.zeros(0)

    def residuals(self, theta: np.ndarray, masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        p = self.params(theta)
        r = self.pixel_residuals(p)
        if masks is not None:
            r = r[np.concatenate(masks)]
        return np.concatenate([r.reshape(-1), self.touch_residuals(p), self.prior_residuals(p)])

    def outputs(self, theta: np.ndarray, T_camlink_optical: np.ndarray) -> np.ndarray:
        """Quantities reported to the user: right base (xyz, rpy) and camera_link (xyz, rpy) in world."""
        p = self.params(theta)
        T_W_cam = p.T_W_O @ inv_T(T_camlink_optical)
        a, b = T_to_xyz_rpy(p.T_W_R), T_to_xyz_rpy(T_W_cam)
        return np.concatenate([a[0], a[1], b[0], b[1]])

    def touch_mismatch_m(self, theta: np.ndarray) -> np.ndarray:
        return self.touch_block.mismatch(self.params(theta), self.flags.offsets)


_kabsch = kabsch     # T with Q ~ T P (rigid, least squares); lives in transforms.py


def _init_board(tables: Sequence[TableObservation], touches: Sequence[Touch], T_W_O: np.ndarray,
                T_W_R: np.ndarray) -> Optional[np.ndarray]:
    """Initial T_W_B: rigid registration of the touches (best arm), else PnP of the table images."""
    best = None
    for a in ARMS:
        ts = [t for t in touches if t.arm == a]
        pts: Dict[int, List[np.ndarray]] = {}
        for t in ts:
            pts.setdefault(t.point_id, []).append(t)
        if len(pts) < 3:
            continue
        P = np.array([pts[k][0].p_board for k in sorted(pts)])
        Q = np.array([np.mean([x.T_base_tcp[:3, 3] for x in pts[k]], axis=0) for k in sorted(pts)])
        if a == "right":
            Q = Q @ T_W_R[:3, :3].T + T_W_R[:3, 3]
        if np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-6) < 2:
            continue
        if best is None or len(pts) > best[0]:
            best = (len(pts), _kabsch(P, Q))
    if best is not None:
        return best[1]
    if tables:
        res = []
        for tb in tables:
            c = pnp_candidates(tb)
            if c:
                res.append(c[0])
        if res:
            res.sort(key=lambda c: c[1])
            return T_W_O @ res[len(res) // 2][0]
    return None


def _attach_q(items, kin) -> None:
    """Fill ``q`` from the joint dictionaries for items that lack it (uses the arm's URDF chain)."""
    for it in items:
        if getattr(it, "q", None) is None and kin and it.arm in kin and getattr(it, "joints", None):
            try:
                it.q = kin[it.arm].q_from_joints(it.joints)
            except Exception:  # noqa: BLE001 - missing joint -> q stays None, reported later
                pass


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def _pose_dict(T: np.ndarray, std: Optional[np.ndarray] = None) -> dict:
    xyz, rpy = T_to_xyz_rpy(T)
    d = {"xyz": [float(v) for v in xyz], "rpy": [float(v) for v in rpy]}
    if std is not None:
        d["std_xyz_mm"] = [float(v) * 1e3 for v in std[:3]]
        d["std_rpy_deg"] = [float(math.degrees(v)) for v in std[3:6]]
    return d


def _propagate(fun, theta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    G = _numeric_jacobian(fun, theta, 1e-6)
    return np.sqrt(np.maximum(np.diag(G @ cov @ G.T), 0.0))


def _covariance(J: np.ndarray, rss: float, dof: int):
    sigma2 = rss / max(dof, 1)
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.pinv(JTJ, rcond=1e-14)
    except np.linalg.LinAlgError:
        cov = np.full_like(JTJ, np.inf)
    # normalise columns so the condition number does not depend on units (rad vs m)
    cn = np.linalg.norm(J, axis=0)
    cn[cn == 0] = 1.0
    sv = np.linalg.svd(J / cn, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return cov, cond, math.sqrt(sigma2)


def rotation_diversity_deg(Rs: Sequence[np.ndarray]) -> np.ndarray:
    """Principal angular spreads (deg, descending) of the rotations about their mean."""
    if len(Rs) < 2:
        return np.zeros(3)
    q = Rotation.from_matrix(np.array(Rs)).mean()
    rv = np.array([R_to_rotvec(q.as_matrix().T @ R) for R in Rs])
    sv = np.linalg.svd(rv - rv.mean(axis=0), compute_uv=False) / math.sqrt(len(Rs))
    out = np.zeros(3)
    out[: len(sv)] = np.degrees(sv)
    return out


def _distinct_poses(samples: Sequence[Sample], tol_rad: float = 0.01) -> int:
    reps: List[Dict[str, float]] = []
    count = 0
    for s in samples:
        if s.joints:
            q = s.joints
            new = True
            for r in reps:
                keys = set(q) & set(r)
                if keys and max(abs(q[k] - r[k]) for k in keys) < tol_rad:
                    new = False
                    break
            if new:
                reps.append(dict(q))
                count += 1
        else:
            count += 1
    return count


def conditioning_report(arm_samples: Dict[str, List[Sample]], options: SolverOptions) -> Tuple[dict, List[str]]:
    info, warns = {}, []
    for arm, ss in arm_samples.items():
        a = {"n_samples": len(ss), "n_distinct_poses": _distinct_poses(ss)}
        div = rotation_diversity_deg([s.T_base_link6[:3, :3] for s in ss])
        a["rotation_spread_deg"] = [float(v) for v in div]
        pos = np.array([s.T_base_link6[:3, 3] for s in ss])
        if len(pos) > 1:
            sp = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(pos.T)), 0))
            a["position_spread_mm"] = [float(v) * 1e3 for v in sp[::-1]]
        else:
            sp = np.zeros(3)
            a["position_spread_mm"] = [0.0] * 3
        if a["n_distinct_poses"] < options.min_samples_warn:
            warns.append(f"{arm}: only {a['n_distinct_poses']} distinct poses ({len(ss)} samples); "
                         f">= {options.min_samples_warn} recommended")
        if div[1] < options.rot_diversity_warn_deg:
            warns.append(f"{arm}: poor rotation diversity (2nd principal spread {div[1]:.1f} deg < "
                         f"{options.rot_diversity_warn_deg:.0f} deg); hand-eye needs at least two non-parallel "
                         "rotation axes, rotate the wrist about different axes")
        if len(pos) > 1 and sp[-1] < options.pos_spread_warn_m:
            warns.append(f"{arm}: link6 positions only spread {sp[-1] * 1e3:.0f} mm; cover more of the workspace")
        info[arm] = a
    return info, warns


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    T_world_optical: np.ndarray
    T_world_camera_link: np.ndarray
    T_world_arm_right_base: np.ndarray
    T_link6_target: Dict[str, np.ndarray]
    report: dict
    params: Optional["Params"] = None            # absolute solution (incl. board, offsets, intrinsics deltas)
    flags: Optional["Flags"] = None
    fit_samples: Optional[list] = None           # samples / table observations kept in the final fit
    fit_tables: Optional[list] = None
    # diagnostics: the same data refitted with joint offsets fixed to 0, and the touch-only base
    T_world_arm_right_base_nominal_refit: Optional[np.ndarray] = None
    T_world_camera_link_nominal_refit: Optional[np.ndarray] = None
    T_world_arm_right_base_touch: Optional[np.ndarray] = None
    input_samples: Optional[list] = None            # raw samples given to solve (validation folds start from these)
    input_tables: Optional[list] = None             # table observations after thinning, before any fit-based filtering

    def to_json(self) -> str:
        return json.dumps(self.report, indent=2)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def _set_weights(samples: Sequence[Sample], options: SolverOptions) -> None:
    for s in samples:
        s.sigma = np.where(s.ids >= CHESS_ID_BASE, 1.0, float(options.marker_sigma_ratio))


def _prepare(samples: Sequence[Sample], options: SolverOptions):
    _set_weights(samples, options)
    by_arm = {a: [s for s in samples if s.arm == a and len(s.ids) >= options.min_points] for a in ARMS}
    for a in ARMS:
        if len(by_arm[a]) < 3:
            raise SolverError(f"arm '{a}' has {len(by_arm[a])} usable samples; need >= 3 "
                              "(>= 12 recommended)")
    cands = {}
    for a in ARMS:
        keep, c = [], []
        for s in by_arm[a]:
            cc = pnp_candidates(s)
            if cc:
                keep.append(s)
                c.append(cc)
        by_arm[a], cands[a] = keep, c
        if len(keep) < 3:
            raise SolverError(f"arm '{a}': PnP failed for most samples")
    return by_arm, cands


def _refine_arm(samples, X0, Z0, options):
    model = _ArmModel(samples, X0, Z0)
    masks = None
    theta = np.zeros(12)
    sample_ok = np.ones(len(samples), bool)
    for _ in range(options.max_outer_iterations + 1):
        sub = [s for s, ok in zip(samples, sample_ok) if ok]
        if len(sub) < 3:
            raise SolverError("arm refinement: too few samples remain after outlier rejection")
        model = _ArmModel(sub, model.poses(theta)[0], model.poses(theta)[1])
        theta, _ = _fit(model, None, options)
        ok_sub, cmask = _flag_outliers(model, theta, options)
        if ok_sub.all():
            masks = cmask
            break
        ids_sub = [i for i, ok in enumerate(sample_ok) if ok]
        for j, ok in enumerate(ok_sub):
            if not ok:
                sample_ok[ids_sub[j]] = False
    X, Z = model.poses(theta)
    return X, Z, sample_ok, model


def solve_arm_only(samples: Sequence[Sample], options: Optional[SolverOptions] = None) -> dict:
    """Solve a single arm alone (camera pose in that arm's base frame + target offset)."""
    options = options or SolverOptions()
    _set_weights(samples, options)
    ss = [s for s in samples if len(s.ids) >= options.min_points]
    cands = []
    keep = []
    for s in ss:
        c = pnp_candidates(s)
        if c:
            keep.append(s)
            cands.append(c)
    X, Z, use = closed_form_init(keep, cands)
    keep2 = [s for s, u in zip(keep, use) if u]
    X, Z, ok, model = _refine_arm(keep2, X, Z, options)
    return _arm_only_summary(model, X, Z, options)


def _arm_only_summary(model: _ArmModel, X, Z, options) -> dict:
    # Re-centre the model on the solution and compute covariance there.
    m = _ArmModel(model.samples, X, Z)
    norms = _residual_norms(m, np.zeros(12))
    _, cmask = _flag_outliers(m, np.zeros(12), options)
    J = _numeric_jacobian(lambda th: m.residuals(th, cmask), np.zeros(12))
    r = m.residuals(np.zeros(12), cmask)
    cov, cond, sigma0 = _covariance(J, float(r @ r), len(r) - 12)
    cam_in_base = inv_T(X)
    std = _propagate(lambda th: np.concatenate(T_to_xyz_rpy(inv_T(m.poses(th)[0]))), np.zeros(12), cov)
    std_z = _propagate(lambda th: np.concatenate(T_to_xyz_rpy(m.poses(th)[1])), np.zeros(12), cov)
    rms = float(np.sqrt(np.mean(np.concatenate([n[k] for n, k in zip(norms, cmask)]) ** 2)))
    return {"n_samples": len(m.samples), "rms_px": rms, "sigma0_px": sigma0, "jacobian_cond": cond,
            "T_optical_base": X, "optical_in_arm_base": _pose_dict(cam_in_base, std),
            "target_offset": _pose_dict(Z, std_z)}


@dataclass
class _Fit:
    model: _FullModel                 # re-centred on the solution (theta = 0)
    params: Params
    masks: List[np.ndarray]           # corner masks per item (samples, then table observations)
    norms: List[np.ndarray]           # pixel residual norms per item
    ok: np.ndarray                    # per input item (samples + tables): kept in the fit


def _joint_refine(samples: Sequence[Sample], tables: Sequence[TableObservation], touches: Sequence[Touch],
                  params0: Params, flags: Flags, options: SolverOptions, kin=None,
                  fixed_masks: Optional[Dict[int, np.ndarray]] = None) -> _Fit:
    """Robust joint fit with sample-level and corner-level outlier rejection (the pipeline of the original solver).

    With ``fixed_masks`` ({id(item): corner mask}) no outlier detection is done: the items and corner masks of
    an earlier fit are reused and a single fit is run (variant fits, validation folds)."""
    items = list(samples) + list(tables)
    n_s = len(samples)
    ok = np.ones(len(items), bool)

    def build(params):
        idx = [i for i in range(len(items)) if ok[i]]
        ss = [items[i] for i in idx if i < n_s]
        tt = [items[i] for i in idx if i >= n_s]
        for a in ARMS:
            if sum(1 for s in ss if s.arm == a) < 3:
                raise SolverError(f"arm '{a}': too few samples remain after outlier rejection")
        return _FullModel(ss, tt, touches, params, flags, options, kin), idx

    if fixed_masks is not None:
        model, _ = build(params0)
        masks = [fixed_masks.get(id(it), np.ones(len(it.ids), bool)) for it in model.items]
        theta, _ = _fit(model, masks, options)
        model, _ = build(model.params(theta))
        theta = np.zeros(model.n_params)
        return _Fit(model, model.params(theta), masks, model.pixel_norm_list(theta), ok)

    params = params0
    masks = None
    for _ in range(options.max_outer_iterations + 1):
        model, idx = build(params)
        theta, _ = _fit(model, None, options)
        params = model.params(theta)
        ok_sub, cmask = _flag_from_norms(model.pixel_norm_list(theta), options)
        if ok_sub.all():
            masks = cmask
            break
        for j, o in enumerate(ok_sub):
            if not o:
                ok[idx[j]] = False
    if masks is None:
        model, idx = build(params)
        masks = [np.ones(len(it.ids), bool) for it in model.items]
    if not all(m.all() for m in masks):          # a polish fit only changes anything if corners are masked
        theta, _ = _fit(model, masks, options)
        params = model.params(theta)
    model, idx = build(params)
    theta = np.zeros(model.n_params)
    norms = model.pixel_norm_list(theta)
    _, masks = _flag_from_norms(norms, options)
    return _Fit(model, model.params(theta), masks, norms, ok)


def _stats(model: _FullModel, masks, want_jacobian: bool = False):
    fun = lambda th: model.residuals(th, masks)  # noqa: E731
    th0 = np.zeros(model.n_params)
    J = _numeric_jacobian(fun, th0)
    r = fun(th0)
    cov, cond, sigma0 = _covariance(J, float(r @ r), len(r) - model.n_params)
    if want_jacobian:
        return cov, cond, sigma0, r, J
    return cov, cond, sigma0, r


def _observability(model: _FullModel, J: np.ndarray, options: SolverOptions) -> dict:
    """Singular values of the DATA-ONLY Jacobian (pixel and touch rows; the prior rows are dropped), so that
    the priors cannot hide directions the data do not determine.  Parametrisation as in the correlation matrix
    (camera and right-base translations at the data centroid); columns are scaled to unit norm, singular values
    are divided by the largest.  A value near 0 means a combination of parameters the data cannot separate."""
    n_prior = len(model.prior_residuals(model.params(np.zeros(model.n_params))))
    Jd = J[: J.shape[0] - n_prior] if n_prior else J
    Jc = Jd @ np.linalg.inv(_centering_matrix(model))     # theta' = A theta  ->  J' = J A^-1
    cn = np.linalg.norm(Jc, axis=0)
    U, sv, Vt = np.linalg.svd(Jc / np.where(cn > 0, cn, 1.0), full_matrices=False)
    norm = sv / sv[0] if sv[0] > 0 else sv
    names = list(model.names)
    weak = []
    for k in range(len(norm) - 1, max(len(norm) - 7, -1), -1):      # the six weakest directions
        v = Vt[k]
        top = [i for i in np.argsort(-np.abs(v))[:4] if abs(v[i]) > 0.1]
        weak.append({"normalised_sv": float(norm[k]), "below_threshold": bool(norm[k] < options.obs_warn),
                     "top_parameters": [{"name": names[i], "weight": float(v[i])} for i in top],
                     "structural": _structural_direction([names[i] for i in top])})
    return {"basis": "data-only Jacobian (prior rows excluded), centred parametrisation, unit-norm columns, "
                     "singular values relative to the largest",
            "n_data_rows": int(Jd.shape[0]), "n_parameters": int(len(names)), "n_prior_rows_dropped": int(n_prior),
            "normalised_singular_values": [float(v) for v in norm], "smallest": float(norm[-1]),
            "threshold": options.obs_warn, "weak_directions": weak}


def _structural_direction(names: List[str]) -> bool:
    """True when the weak direction only mixes the structural pairs of ``_structural_pair``."""
    if len(names) < 2:
        return False
    return all(_structural_pair(a, b) for i, a in enumerate(names) for b in names[i + 1:])


def _rms(v) -> Optional[float]:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else None


def touch_summary(touches: Sequence[Touch], mism_m: np.ndarray) -> dict:
    """Per-touch mismatch records (mm) and RMS per arm."""
    mm = np.asarray(mism_m).reshape(-1, 3) * 1e3
    per = [{"arm": t.arm, "point": t.point_id, "dx_mm": float(m[0]), "dy_mm": float(m[1]), "dz_mm": float(m[2]),
            "norm_mm": float(np.linalg.norm(m))} for t, m in zip(touches, mm)]
    norms = np.linalg.norm(mm, axis=1) if len(mm) else np.zeros(0)
    arms = np.array([t.arm for t in touches])
    rms = {a: _rms(norms[arms == a]) for a in ARMS}
    rms["all"] = _rms(norms)
    return {"rms_mm": rms, "per_touch": per}


def _pose_diff(Ta: np.ndarray, Tb: np.ndarray) -> dict:
    return {"position_mm": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1e3),
            "rotation_deg": rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])}


def _prepare_tables(tables, options: SolverOptions) -> List[TableObservation]:
    tables = [t for t in (tables or []) if len(t.ids) >= 8]
    for t in tables:
        t.sigma = np.where(t.ids >= CHESS_ID_BASE, 1.0, float(options.marker_sigma_ratio))
    if len(tables) > options.max_table_obs:
        pick = np.unique(np.round(np.linspace(0, len(tables) - 1, options.max_table_obs)).astype(int))
        tables = [tables[i] for i in pick]
    return tables


def _fixed_masks(fit: _Fit) -> Dict[int, np.ndarray]:
    return {id(it): m for it, m in zip(fit.model.items, fit.masks)}


def _loop_closure(fit: _Fit, touches: Sequence[Touch], flags: Flags, options: SolverOptions, kin) -> dict:
    """Touch mismatch after the fit and out-of-fit checks (touches not used to determine what they test).

    The reported numbers evaluate the touches with NOMINAL forward kinematics (the TF poses as recorded), because
    the robot never applies fitted joint offsets; with ``flags.offsets`` the same numbers with the fitted offsets
    applied are given under ``offset_corrected`` as a diagnostic."""
    model = fit.model
    samples, tables = model.samples, model.tables
    touches = list(touches)
    evals = [False] + ([True] if flags.offsets else [])
    lc: dict = {"touch_sigma_mm": options.touch_sigma_m * 1e3, "n_touches": {a: sum(t.arm == a for t in touches)
                                                                          for a in ARMS},
                "points": {a: sorted({t.point_id for t in touches if t.arm == a}) for a in ARMS}}
    lc["model"] = ("nominal FK (as recorded); base/camera/board from the fit " +
                   ("with nuisance joint offsets" if flags.offsets else "without joint offsets"))
    blk = _TouchBlock(touches, kin, flags.offsets)
    out = {ev: {"camera_only": None, "cross_arm": {}} for ev in evals}
    for ev in evals:
        out[ev]["final"] = touch_summary(touches, blk.mismatch(fit.params, ev))
    lc["note"] = ("'final' = touches used in the fit.  'camera_only' = touches compared with a solution that does "
                  "not use them (board pose from the table-board images, right base from the camera chain).  "
                  "'cross_arm' = touches of one arm compared with the board pose defined by the other arm's "
                  "touches.  'final' is a fit residual (FRE, in-sample); only the out-of-fit numbers say anything about "
                  "the calibration, and only at the touched board points (held-out check points are the TRE).")
    lc["right_base_with_vs_without_touches"] = None
    if options.loop_closure_variants:
        flags_nt = Flags(board=bool(tables), offsets=flags.offsets, intr=flags.intr, touches=False)
        p0 = fit.params.copy()
        try:
            fit_nt = _joint_refine(samples, tables, [], p0, flags_nt, options, kin, _fixed_masks(fit))
            lc["right_base_with_vs_without_touches"] = {
                **_pose_diff(fit.params.T_W_R, fit_nt.params.T_W_R),
                "without_touches": _pose_dict(fit_nt.params.T_W_R), "with_touches": _pose_dict(fit.params.T_W_R)}
            if tables:
                for ev in evals:
                    out[ev]["camera_only"] = touch_summary(touches, blk.mismatch(fit_nt.params, ev))
        except SolverError as e:
            lc["variant_error"] = str(e)
        for a in ARMS:
            mine = [t for t in touches if t.arm == a]
            others = [t for t in touches if t.arm != a]
            if not mine or len({t.point_id for t in others}) < 3:
                continue
            try:
                fx = _joint_refine(samples, tables, others, p0, Flags(True, flags.offsets, flags.intr, True),
                                   options, kin, _fixed_masks(fit))
                bm = _TouchBlock(mine, kin, flags.offsets)
                for ev in evals:
                    out[ev]["cross_arm"][a] = touch_summary(mine, bm.mismatch(fx.params, ev))
            except SolverError as e:
                lc["variant_error"] = str(e)
    lc.update(out[False])
    if flags.offsets:
        lc["offset_corrected"] = {**out[True], "right_base_with_vs_without_touches":
                                  lc["right_base_with_vs_without_touches"],
                                  "note": "diagnostic only: touch mismatch with the fitted joint offsets applied; the "
                                          "robot never applies them (NEVER apply them to the robot/URDF)"}
        _set_out_of_fit(lc["offset_corrected"])
    _set_out_of_fit(lc)
    return lc


def _set_out_of_fit(loop: dict) -> None:
    """The out-of-fit touch mismatch: camera-only if available, else cross-arm.  When neither exists there is NO
    out-of-fit number (``None``): the in-fit residual (FRE) is never substituted, so it cannot satisfy the gate."""
    oof, src = None, None
    if loop["camera_only"] is not None:
        oof, src = loop["camera_only"]["rms_mm"]["all"], "camera_only"
    elif loop["cross_arm"]:
        vals = [v["rms_mm"][a] for a, v in loop["cross_arm"].items() if v["rms_mm"][a] is not None]
        if vals:
            oof, src = float(np.sqrt(np.mean(np.square(vals)))), "cross_arm"
    if oof is None:
        src = "none (only in-fit data: no out-of-fit estimate available)"
    loop["out_of_fit_rms_mm"], loop["out_of_fit_source"] = oof, src
    loop["fit_residual_fre_rms_mm"] = loop["final"]["rms_mm"]["all"]


def _skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _centering_matrix(model: _FullModel) -> np.ndarray:
    """A with theta' = A theta: the camera and right-base translations re-expressed at the data centroid.

    The solver perturbs a pose at its own origin; for the camera (about 1 m from the data) a rotation and the
    translation it induces at the targets are then trivially correlated (about 1.0).  Reporting the translation
    of the point that matters, ``t' = t + w x x_c`` (x_c: data centroid in that frame), removes this lever arm
    effect so that the correlations that remain describe real parameter coupling."""
    p = model.params(np.zeros(model.n_params))
    cen = np.mean([T[:3, 3] for T in [p.T_W_O @ Tt for Tt in
                                      [inv_T(p.T_W_O) @ (np.eye(4) if s.arm == "left" else p.T_W_R)
                                       @ s.T_base_link6 @ p.Z[s.arm] for s in model.samples]]], axis=0)
    A = np.eye(model.n_params)
    for key, T in (("O", p.T_W_O), ("R", p.T_W_R)):
        x_c = inv_T(T)[:3, :3] @ cen + inv_T(T)[:3, 3]
        sl = model.sl[key]
        A[sl.start + 3:sl.start + 6, sl.start:sl.start + 3] = -_skew(x_c)
    return A


def _centered_cov(model: _FullModel, cov: np.ndarray) -> np.ndarray:
    A = _centering_matrix(model)
    return A @ cov @ A.T


def _structural_pair(a: str, b: str) -> bool:
    """Couplings that are built into the model (expected, held by the priors): the parameters of one target
    mount among themselves, and a joint-6 offset with its arm's target mount (image data cannot tell them apart)."""
    ga, gb = a.split(".")[0], b.split(".")[0]
    if ga == gb and ga.startswith("target_"):
        return True
    for x, y, nx in ((a, b, ga), (b, a, gb)):
        if x.startswith("offset_") and x.endswith("joint6") and y.startswith("target_") \
                and nx.replace("offset_", "") == y.split(".")[0].replace("target_", ""):
            return True
    return False


def _correlation(model: _FullModel, cov: np.ndarray, options: SolverOptions) -> dict:
    cov = _centered_cov(model, cov)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ok = sd > 1e-12
    corr = np.zeros_like(cov)
    corr[np.ix_(ok, ok)] = cov[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
    np.fill_diagonal(corr, 1.0)
    pairs = []
    n = len(model.names)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > options.corr_warn:
                pairs.append({"a": model.names[i], "b": model.names[j], "corr": float(corr[i, j]),
                              "structural": _structural_pair(model.names[i], model.names[j])})
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return {"threshold": options.corr_warn, "names": list(model.names),
            "parametrisation": "camera and right-base translations are taken at the data centroid",
            "matrix": [[round(float(v), 4) for v in row] for row in corr], "high_pairs": pairs}


def _intrinsics_report(fit: _Fit, touches, flags: Flags, options: SolverOptions, kin, cov, model) -> dict:
    """Estimated (or, when fixed, would-be) intrinsics change.  Always reported."""
    names = ("fx", "fy", "cx", "cy")
    K0 = model.K0[0] if len(model.K0) else np.array([1.0, 1.0, 0.0, 0.0])

    def pull(dK):
        d = np.asarray(dK, float)
        return {"nominal_px": dict(zip(names, map(float, K0))),
                "pull_f_frac": float(max(abs(d[0]) / K0[0], abs(d[1]) / K0[1])),
                "pull_c_px": float(max(abs(d[2]), abs(d[3])))}

    if flags.intr:
        sl = model.sl["K"]
        std = np.sqrt(np.maximum(np.diag(cov)[sl], 0))
        return {"refined": True, "delta_px": dict(zip(names, map(float, fit.params.dK))),
                "std_px": dict(zip(names, map(float, std))), **pull(fit.params.dK),
                "source": options.intrinsics_source,
                "prior_sigma_px": [options.intrinsics_prior_frac * model.K0[0][0],
                                   options.intrinsics_prior_frac * model.K0[0][1],
                                   options.intrinsics_prior_px, options.intrinsics_prior_px]}
    out = {"refined": False, "delta_px": None, "source": options.intrinsics_source,
           "note": ("intrinsics fixed to the --intrinsics-file values" if options.intrinsics_source == "file"
                    else "intrinsics fixed to camera_info") + "; delta_px is how far the data would pull them "
                                                              "(fit with the priors)"}
    if not options.intrinsics_report:
        return out
    try:
        fl = Flags(fit.model.flags.board, flags.offsets, True, flags.touches)
        fi = _joint_refine(fit.model.samples, fit.model.tables, touches, fit.params, fl, options, kin,
                           _fixed_masks(fit))
        cov_i, _, _, _ = _stats(fi.model, fi.masks)
        std = np.sqrt(np.maximum(np.diag(cov_i)[fi.model.sl["K"]], 0))
        rms0 = _rms(np.concatenate([n[m] for n, m in zip(fit.norms, fit.masks)]))
        rms1 = _rms(np.concatenate([n[m] for n, m in zip(fi.norms, fi.masks)]))
        out.update(delta_px=dict(zip(names, map(float, fi.params.dK))), std_px=dict(zip(names, map(float, std))),
                   **pull(fi.params.dK),
                   prior_sigma_px=[options.intrinsics_prior_frac * fi.model.K0[0][0],
                                   options.intrinsics_prior_frac * fi.model.K0[0][1],
                                   options.intrinsics_prior_px, options.intrinsics_prior_px],
                   right_base_shift=_pose_diff(fi.params.T_W_R, fit.params.T_W_R),
                   camera_shift=_pose_diff(fi.params.T_W_O, fit.params.T_W_O),
                   rms_px_before=rms0, rms_px_after=rms1)
    except SolverError as e:
        out["error"] = str(e)
    return out


def _run_fit(samples: Sequence[Sample], options: SolverOptions, tables, touches, kin):
    """The whole fitting pipeline from RAW samples: PnP -> closed form per arm -> per-arm refinement ->
    stage A (pixels, outlier rejection) -> stage B (touches / offsets / intrinsics).  Everything (initialisation,
    outlier filtering) depends only on the items passed in, which is what makes a validation fold leak-free.
    Returns ``(by_arm, fit)``."""
    by_arm, cands = _prepare(samples, options)
    arm_init = {}
    for a in ARMS:
        X, Z, use = closed_form_init(by_arm[a], cands[a])
        keep = [s for s, u in zip(by_arm[a], use) if u]
        X, Z, ok, amodel = _refine_arm(keep, X, Z, options)
        arm_init[a] = (X, Z, [s for s, o in zip(keep, ok) if o])

    X_l, Z_l, ss_l = arm_init["left"]
    X_r, Z_r, ss_r = arm_init["right"]
    T_W_O0 = inv_T(X_l)                    # world == left base
    T_W_R0 = T_W_O0 @ X_r                  # inv(T_O_W) @ X_r  ->  T_W_O @ T_O_Rbase
    all_samples = ss_l + ss_r
    for t in touches:
        if t.arm not in ARMS:
            raise SolverError(f"touch with unknown arm '{t.arm}'")

    use_board = bool(tables or touches)
    T_W_B0 = _init_board(tables, touches, T_W_O0, T_W_R0) if use_board else None
    if use_board and T_W_B0 is None:
        raise SolverError("touch samples need >= 3 distinct touched points of one arm (non-collinear), or a "
                          "table-board image, to initialise the board pose")
    if options.joint_offsets:
        _attach_q(by_arm["left"] + by_arm["right"] + list(touches), kin)
    params0 = Params(T_W_O0, T_W_R0, {"left": Z_l, "right": Z_r}, T_W_B0)

    # ---- stage A: pixels only (the original robust joint fit) --------------
    flags_a = Flags(board=bool(tables))
    fit = _joint_refine(all_samples, tables, [], params0, flags_a, options, kin)
    # ---- stage B: touches / joint offsets / intrinsics ---------------------
    flags = Flags(board=use_board, offsets=options.joint_offsets, intr=options.refine_intrinsics,
                  touches=bool(touches))
    if touches or options.joint_offsets or options.refine_intrinsics:
        pB = fit.params.copy()
        if T_W_B0 is not None:
            tb = _init_board([], touches, pB.T_W_O, pB.T_W_R) if touches else None
            pB.T_W_B = tb if tb is not None else (fit.params.T_W_B if fit.params.T_W_B is not None else T_W_B0)
        # B1: from the stage-A inliers with a wide robust scale, so that the larger initial residuals (the
        # unmodelled offsets) are not down-weighted into a local minimum; B2: outlier handling on all samples.
        wide = dataclasses.replace(options, f_scale_px=5.0 * options.f_scale_px)
        b1 = _joint_refine(fit.model.samples, fit.model.tables, touches, pB, flags, wide, kin, _fixed_masks(fit))
        fit = _joint_refine(by_arm["left"] + by_arm["right"], tables, touches, b1.params, flags, options, kin)
    return by_arm, fit


def solve(samples: Sequence[Sample], options: Optional[SolverOptions] = None,
          dataset: str = "", tables: Optional[Sequence[TableObservation]] = None,
          touches: Optional[Sequence[Touch]] = None, kin: Optional[dict] = None,
          gate_options=None) -> CalibrationResult:
    """Solve the dual-arm calibration.

    ``tables``  corner observations of the fixed table board (extra reprojection residuals, unknown T_W_B).
    ``touches`` TCP-on-board-point records (3 position residuals each): the only coupling between the arms.
    ``kin``     {arm: ArmKinematics} for ``options.joint_offsets``.

    With ``joint_offsets`` the main fit (``joint``) is the one with the nuisance offsets; the deployed robot uses
    NOMINAL forward kinematics, so a second fit with the offsets fixed to zero (``nominal``) provides the base and
    camera that are written to the YAML and the touch mismatch that the trust gate uses.
    """
    from .base_check import GateOptions, tcp_consistency, touch_only_base
    from .metrology import check_point_report, repeatability
    options = options or SolverOptions()
    gate_options = gate_options or GateOptions()
    tables = _prepare_tables(tables, options)
    all_touches = list(touches or [])
    # Only the printed board crosshairs (kind "board", known board coordinates) enter the joint solve.  Free fit
    # stickers only widen the touch-only Kabsch base; check stickers are NEVER used in any fit.
    touches = [t for t in all_touches if t.kind == "board"]
    warnings: List[str] = []
    notes: List[str] = []
    by_arm, fit = _run_fit(samples, options, tables, touches, kin)
    cond_info, cw = conditioning_report(by_arm, options)
    warnings += cw
    for a in ARMS:
        n = cond_info[a]["n_distinct_poses"]
        if options.min_samples_warn <= n < options.recommended_samples[0]:
            notes.append(f"{a}: {n} distinct poses; {options.recommended_samples[0]}-{options.recommended_samples[1]} "
                         "per arm recommended")
    model, masks, params = fit.model, fit.masks, fit.params
    flags = model.flags
    theta = np.zeros(model.n_params)
    T_W_O, T_W_R, Zl, Zr = params.T_W_O, params.T_W_R, params.Z["left"], params.Z["right"]
    n_s = model.n_s
    norms_all = fit.norms

    # ---- statistics -------------------------------------------------------
    T_cl_o = np.mean([s.T_camlink_optical for s in model.samples], axis=0)
    Rm = Rotation.from_matrix(np.array([s.T_camlink_optical[:3, :3] for s in model.samples])).mean().as_matrix()
    T_cl_o = make_T(Rm, T_cl_o[:3, 3])
    spread = max(rotation_angle_deg(Rm, s.T_camlink_optical[:3, :3]) for s in model.samples)
    if spread > 0.05:
        warnings.append(f"camera_link->optical TF varied between captures by {spread:.2f} deg")

    cov, cond, sigma0, r, J_full = _stats(model, masks, want_jacobian=True)
    out_std = _propagate(lambda th: model.outputs(th, T_cl_o), theta, cov)
    T_W_cam = T_W_O @ inv_T(T_cl_o)

    # ---- plain nominal-FK refit (offsets fixed to 0): a DIAGNOSTIC alternative, see report['nominal_refit'] ------
    if flags.offsets:
        p_nom = params.copy()
        p_nom.off = {a: np.zeros(6) for a in ARMS}
        flags_nom = Flags(board=flags.board, offsets=False, intr=flags.intr, touches=flags.touches)
        fit_nom = _joint_refine(model.samples, model.tables, touches, p_nom, flags_nom, options, kin,
                                _fixed_masks(fit))
        cov_n, _, _, _ = _stats(fit_nom.model, fit_nom.masks)
        out_std_n = _propagate(lambda th: fit_nom.model.outputs(th, T_cl_o), np.zeros(fit_nom.model.n_params), cov_n)
    else:
        flags_nom, fit_nom, out_std_n = flags, fit, out_std
    T_W_R_nom = fit_nom.params.T_W_R
    T_W_cam_nom = fit_nom.params.T_W_O @ inv_T(T_cl_o)
    nominal_refit_loop = None

    def pose_std(get):
        return _propagate(lambda th: np.concatenate(T_to_xyz_rpy(get(model.params(th)))), theta, cov)

    std_O = pose_std(lambda p: p.T_W_O)
    std_Zl = pose_std(lambda p: p.Z["left"])
    std_Zr = pose_std(lambda p: p.Z["right"])

    poses_pred = model.target_poses(theta)
    per_sample, per_arm_err, per_arm_px = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
    counters = {a: 0 for a in ARMS}
    for k, (s, T) in enumerate(zip(model.samples, poses_pred)):
        n = norms_all[k]
        cands_k = pnp_candidates(s)
        pe = ae = depth = None
        if cands_k:
            best = min(cands_k, key=lambda c: _pose_dist(T, c[0])[0] / 0.05 + _pose_dist(T, c[0])[1] / 10.0)
            pe, ae = _pose_dist(T, best[0])
            depth = float(T[2, 3] - best[0][2, 3])
            per_arm_err[s.arm].append((pe * 1e3, abs(depth) * 1e3, ae))
        per_arm_px[s.arm].append(n[masks[k]])
        per_sample.append({
            "arm": s.arm, "index": counters[s.arm], "image": s.image, "n_points": int(len(n)),
            "rms_px": float(np.sqrt(np.mean(n[masks[k]] ** 2))) if masks[k].any() else float("nan"),
            "median_px": float(np.median(n)), "max_px": float(n.max()),
            "n_corner_outliers": int((~masks[k]).sum()),
            "pnp_position_diff_mm": None if pe is None else pe * 1e3,
            "pnp_rotation_diff_deg": ae, "outlier": False,
            "target_distance_m": float(np.linalg.norm(T[:3, 3])),
            "view_angle_deg": float(math.degrees(math.acos(min(1.0, abs(T[2, 2]))))),
        })
        counters[s.arm] += 1
    n_input = {a: sum(1 for s in samples if s.arm == a) for a in ARMS}
    n_used = {a: sum(1 for s in model.samples if s.arm == a) for a in ARMS}
    rejected_total = {a: n_input[a] - n_used[a] for a in ARMS}
    corner_outliers = int(sum((~m).sum() for m in masks))

    rms = {a: (float(np.sqrt(np.mean(np.concatenate(per_arm_px[a]) ** 2))) if per_arm_px[a] else float("nan"))
           for a in ARMS}
    rms["all"] = float(np.sqrt(np.mean(np.concatenate(sum(per_arm_px.values(), [])) ** 2)))
    if model.tables:
        rms["table_board"] = _rms(np.concatenate([n[m] for n, m in zip(norms_all[n_s:], masks[n_s:])]))
    err3d = {}
    for a in ARMS:
        e = np.array(per_arm_err[a]) if per_arm_err[a] else np.zeros((0, 3))
        err3d[a] = {"pnp_position_diff_rms_mm": float(np.sqrt(np.mean(e[:, 0] ** 2))) if len(e) else None,
                    "pnp_position_diff_max_mm": float(e[:, 0].max()) if len(e) else None,
                    "pnp_depth_diff_rms_mm": float(np.sqrt(np.mean(e[:, 1] ** 2))) if len(e) else None,
                    "pnp_rotation_diff_rms_deg": float(np.sqrt(np.mean(e[:, 2] ** 2))) if len(e) else None}

    # conditioning warnings from the fit
    if cond > options.cond_warn:
        warnings.append(f"Jacobian condition number {cond:.2e} > {options.cond_warn:.0e}: some parameters are "
                        "poorly determined (check std devs); collect more diverse poses")
    for a in ARMS:
        if n_used[a] < options.min_samples_warn:
            warnings.append(f"{a}: only {n_used[a]} samples used in the final fit (< {options.min_samples_warn})")
        if rejected_total[a]:
            warnings.append(f"{a}: {rejected_total[a]} sample(s) rejected as outliers")
    if rms["all"] > options.rms_warn_px:
        warnings.append(f"RMS reprojection error {rms['all']:.2f} px > {options.rms_warn_px:.1f} px: check the "
                        "measured target dimensions, intrinsics and joint/TF timing"
                        + ("" if options.joint_offsets else "; unmodelled joint zero offsets also raise it "
                           "(try --joint-offsets and compare the held-out error)"))
    angles = [p["view_angle_deg"] for p in per_sample]
    if angles and float(np.median(angles)) > 60.0:
        warnings.append("targets are viewed at grazing angles (median > 60 deg off the camera axis); "
                        "depth/orientation observability is poor")

    # ---- correlation -------------------------------------------------------
    corr = _correlation(model, cov, options)
    unexpected = [p for p in corr["high_pairs"] if not p["structural"]]
    if unexpected:
        top = "; ".join(f"{p['a']} ~ {p['b']} ({p['corr']:+.2f})" for p in unexpected[:6])
        warnings.append(f"{len(unexpected)} parameter pair(s) with |correlation| > {options.corr_warn}: {top}"
                        + (" ..." if len(unexpected) > 6 else "")
                        + " (strongest first; the full matrix is in the report): these parameters are poorly "
                          "separated by the data, collect more diverse poses")
    if len(corr["high_pairs"]) > len(unexpected):
        notes.append(f"{len(corr['high_pairs']) - len(unexpected)} further high-correlation pair(s) are structural "
                     "(target mount parameters among themselves; joint-6 offset vs target mount) and held by the priors")

    # ---- loop closure: NOMINAL-FK touch mismatch with the base/camera/board that are written ------------------
    # (the robot never applies the fitted joint offsets); the offset-corrected numbers are a diagnostic.
    loop = None
    if touches:
        loop = _loop_closure(fit, touches, flags, options, kin)
        if flags.offsets:
            nominal_refit_loop = touch_summary(touches, _TouchBlock(touches, kin, False).mismatch(
                fit_nom.params, False))["rms_mm"]
        thr = options.touch_warn_factor * options.touch_sigma_m * 1e3
        for key in ("final", "camera_only"):
            blk = loop[key]
            if blk and blk["rms_mm"]["all"] is not None and blk["rms_mm"]["all"] > thr:
                warnings.append(f"touch loop closure ({key}, nominal FK): RMS {blk['rms_mm']['all']:.2f} mm > "
                                f"{thr:.2f} mm ({options.touch_warn_factor} x touch sigma): the arms/camera/board "
                                "are not consistent (wrong board scale, TCP offset, base pose or joint offsets)")
        for a in ARMS:
            nd = len(loop["points"][a])
            if loop["n_touches"][a] and nd < 3:
                warnings.append(f"{a}: touches at only {nd} distinct board point(s); >= 3 non-collinear needed, "
                                "4 recommended")
    else:
        notes.append("no touch samples: the two arms are not coupled by a loop closure; the right base rests on "
                     "the camera chain only.  Record table-board touches (README) for an independent accuracy number.")

    # ---- touch-only base (board crosshairs + fit stickers), TCP consistency, check points ------------------
    touch_base = touch_only_base(all_touches, T_W_R, gate_options)
    touch_base["tcp_consistency"] = tcp_consistency(all_touches, gate_options)
    if all_touches:
        warnings += touch_base["warnings"] + touch_base["tcp_consistency"]["warnings"]
        if not touch_base["available"]:
            warnings.append("no touch-only base estimate: " + touch_base.get("reason", ""))
    check_points = check_point_report(all_touches, T_W_R)
    if not check_points["points"]:
        notes.append("no check points (kind 'check' stickers touched by both arms): nothing was held out, the "
                     "trust gate cannot pass; stick >= 2 check stickers (one on a ~40 mm block, one >= 150 mm from the "
                     "board) and touch them with both arms")
    # ---- observability of the data (priors excluded) ------------------------------------------------------
    obs = _observability(model, J_full, options)
    weak = [w for w in obs["weak_directions"] if w["below_threshold"]]
    if weak:
        top = "; ".join(f"sv {w['normalised_sv']:.1e}: " + " + ".join(p["name"] for p in w["top_parameters"])
                        + (" (structural)" if w["structural"] else "") for w in weak[:4])
        warnings.append(f"{len(weak)} parameter direction(s) are not determined by the data alone (smallest "
                        f"normalised singular value {obs['smallest']:.1e} < {options.obs_warn:.0e}, priors "
                        f"excluded): {top}; any value reported for them comes from the prior")

    # ---- offsets / intrinsics -----------------------------------------------
    offsets_rep = {"enabled": bool(flags.offsets), "prior_sigma_deg": options.offset_prior_deg}
    if flags.offsets:
        est, std = {}, {}
        for a in ARMS:
            sl = model.sl["off_" + a]
            est[a] = [math.degrees(v) for v in params.off[a]]
            std[a] = [None] + [math.degrees(v) for v in np.sqrt(np.maximum(np.diag(cov)[sl], 0))]
        offsets_rep.update(estimated_deg=est, std_deg=std, joint_names={
            a: (kin[a].chain6.joint_names if kin and a in kin else None) for a in ARMS},
            note="joint 1 of each arm is not separable from the base yaw: it is fixed to 0 (the right base yaw "
                 "and the camera absorb it; for NOMINAL FK, which the robot uses, that absorption is exactly what "
                 "is wanted).  Joint 6 is separable from the target mount only through touches.  These offsets are "
                 "nuisance parameters of THIS solve only: NEVER apply them to the robot or the URDF (re-referencing "
                 "invalidates them and they absorb backlash, compliance and link errors).  The YAML base/camera "
                 "come from the 'nominal' fit with the offsets fixed to 0.")
    intr = _intrinsics_report(fit, touches, flags, options, kin, cov, model)
    if options.intrinsics_source == "file" and intr.get("pull_f_frac") is not None \
            and (intr["pull_f_frac"] > 0.003 or intr["pull_c_px"] > 3.0):
        warnings.append(f"the data would pull the intrinsics of --intrinsics-file by {100 * intr['pull_f_frac']:.2f} % in f and "
                        f"{intr['pull_c_px']:.1f} px in c: the file may not belong to this camera / lens setting / resolution "
                        "(the gate still counts a supplied file as validated; check the held-out reprojection)")

    board_rep = None
    if flags.board:
        std_B = pose_std(lambda p: p.T_W_B)
        board_rep = _pose_dict(params.T_W_B, std_B)
        board_rep["n_table_images"] = len(model.tables)

    statement = []
    if not flags.offsets:
        statement.append("The std devs cover detection noise only (joint offsets are NOT modelled): the real error "
                         "is larger by the arm's kinematic error (about 1 mm or more).")
    else:
        statement.append("Joint zero offsets are modelled as regularised nuisance parameters, so the std devs "
                         "include their uncertainty (prior %.2f deg); other kinematic errors are still not covered."
                         % options.offset_prior_deg)
    if flags.offsets:
        statement.append("The fitted joint offsets are NEVER applied to the robot/URDF.  Whether the base written to the "
                         "YAML is the offset fit (joint-1 offset absorbed in the base yaw, correct for nominal FK) or "
                         "the plain nominal fit follows the held-out validation (report 'written_base'); the touch "
                         "mismatch and the trust gate are evaluated with NOMINAL FK, i.e. the model the robot "
                         "actually runs.")
    if touches or all_touches:
        statement.append("The touch residual of the board crosshairs is a fit residual (FRE, in-sample) and, out of fit, "
                         "a loop closure at the board; only the check-point error (TRE, held-out stickers touched by "
                         "both arms, never used in any fit) says how well the two arms agree elsewhere, and only AT "
                         "those check points: accuracy demonstrated at the check points only, no claim for the rest "
                         "of the workspace.  The numbers also contain the TCP offset error and the print/board "
                         "flatness.")
    else:
        statement.append("There is no independent accuracy measurement without touch samples.")

    # ---- per-arm-only solutions (nominal kinematics, no touches) --------------
    separate = {}
    X_sep = {}
    for a in ARMS:
        sub = [s for s in model.samples if s.arm == a]
        cs = [pnp_candidates(s) for s in sub]
        Xa, Za, use = closed_form_init(sub, cs)
        sub2 = [s for s, u in zip(sub, use) if u]
        Xa, Za, _, am = _refine_arm(sub2, Xa, Za, options)
        d = _arm_only_summary(am, Xa, Za, options)
        X_sep[a] = Xa
        separate[a] = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in d.items()}
        separate[a].pop("T_optical_base", None)
    T_W_R_sep = inv_T(X_sep["left"]) @ X_sep["right"]
    T_W_cam_sep = inv_T(X_sep["left"]) @ inv_T(T_cl_o)
    dpos = float(np.linalg.norm(T_W_R_sep[:3, 3] - T_W_R[:3, 3]) * 1e3)
    drot = rotation_angle_deg(T_W_R_sep[:3, :3], T_W_R[:3, :3])
    dpos_c = float(np.linalg.norm(T_W_cam_sep[:3, 3] - T_W_cam[:3, 3]) * 1e3)
    drot_c = rotation_angle_deg(T_W_cam_sep[:3, :3], T_W_cam[:3, :3])
    separate["composed"] = {"arm_right_base": _pose_dict(T_W_R_sep), "camera_link": _pose_dict(T_W_cam_sep),
                            "arm_right_base_diff_vs_joint": {"position_mm": dpos, "rotation_deg": drot},
                            "camera_link_diff_vs_joint": {"position_mm": dpos_c, "rotation_deg": drot_c},
                            "note": "per-arm fits with nominal kinematics and no touches/offsets/board"}

    report = {
        "date": datetime.date.today().isoformat(),
        "dataset": dataset,
        "n_samples_input": n_input, "n_samples_used": n_used, "n_samples_rejected": rejected_total,
        "n_corner_outliers": corner_outliers,
        "n_table_observations": len(model.tables), "n_touches": {a: sum(t.arm == a for t in touches) for a in ARMS},
        "n_sticker_touches": {a: {k: sum(t.arm == a and t.kind == k for t in all_touches) for k in ("fit", "check")}
                              for a in ARMS},
        "conditioning": cond_info,
        "joint": {
            "rms_px": rms, "sigma0_px": sigma0, "jacobian_cond": cond, "n_residuals": int(len(r)),
            "arm_right_base": _pose_dict(T_W_R, out_std[0:6]),
            "camera_link": _pose_dict(T_W_cam, out_std[6:12]),
            "optical_frame": _pose_dict(T_W_O, std_O),
            "target_offsets": {"left": _pose_dict(Zl, std_Zl), "right": _pose_dict(Zr, std_Zr)},
            "error_3d_mm": err3d,
            "calibration_position_std_mm": {"arm_right_base": float(np.linalg.norm(out_std[0:3]) * 1e3),
                                            "camera_link": float(np.linalg.norm(out_std[6:9]) * 1e3)},
            "T_camlink_optical": _pose_dict(T_cl_o),
            "table_board": board_rep,
            "joint_offsets": offsets_rep,
            "intrinsics": intr,
            "std_basis": "detection noise" + (" + joint offsets (prior)" if flags.offsets else "")
                         + (" + touch sigma %.2f mm" % (options.touch_sigma_m * 1e3) if touches else ""),
        },
        "nominal_refit": {
            "arm_right_base": _pose_dict(T_W_R_nom, out_std_n[0:6]),
            "camera_link": _pose_dict(T_W_cam_nom, out_std_n[6:12]),
            "same_as_joint": not flags.offsets,
            "rms_px_all": _rms(np.concatenate([n[m] for n, m in zip(fit_nom.norms[:fit_nom.model.n_s],
                                                                     fit_nom.masks[:fit_nom.model.n_s])])),
            "touch_rms_mm_nominal": nominal_refit_loop,
            "difference_vs_joint": {"arm_right_base": _pose_diff(T_W_R_nom, T_W_R),
                                    "camera_link": _pose_diff(T_W_cam_nom, T_W_cam)},
            "note": "DIAGNOSTIC: the same data refitted with the joint offsets forced to 0.  Not what is written: "
                    "under misfit kinematics the camera/base absorb the offsets' effect and predict the nominal "
                    "TCP positions worse than the offset-fit base (see README, 'nominal-model prediction error')."},
        "touch_base": touch_base,
        "check_points": check_points,
        "repeatability": repeatability(all_touches),
        "written_base": None,
        "observability": obs,
        "loop_closure": loop,
        "correlation": corr,
        "accuracy_statement": statement,
        "separate": separate,
        "warnings": warnings,
        "notes": notes,
        "samples": per_sample,
        "options": {k: v for k, v in vars(options).items()},
    }
    res = CalibrationResult(T_W_O, T_W_cam, T_W_R, {"left": Zl, "right": Zr}, report)
    res.T_world_arm_right_base_nominal_refit = T_W_R_nom
    res.T_world_camera_link_nominal_refit = T_W_cam_nom
    res.T_world_arm_right_base_touch = (np.array(touch_base["T"]) if touch_base.get("available") else None)
    res.input_samples = list(samples)
    res.input_tables = list(tables)
    res.params = params
    res.flags = flags
    res.fit_samples, res.fit_tables = list(model.samples), list(model.tables)
    res.touches_all, res.touches_fit, res.kin = all_touches, touches, kin
    res.gate_options, res.options = gate_options, options
    res.fit, res.fit_nominal, res.T_camlink_optical = fit, fit_nom, T_cl_o
    # provisional: the joint fit as before; solve_cli decides after the held-out validation (see set_written_base)
    set_written_base(res, "offset_fit" if flags.offsets else "nominal_fit",
                     "provisional (no held-out validation decision yet)")
    return res


def choose_written_fit(res: "CalibrationResult", validation: Optional[dict]) -> Tuple[str, str]:
    """Which fit is written: the offset fit only if ``--joint-offsets`` was used AND the held-out validation recommends
    KEEP; otherwise the nominal fit.  Returns (``offset_fit`` | ``nominal_fit``, reason)."""
    if not (res.flags and res.flags.offsets):
        return "nominal_fit", "no joint zero offsets were estimated (the nominal fit is the only fit)"
    if validation and validation.get("keep_joint_offsets") is True:
        return "offset_fit", ("--joint-offsets and the held-out validation recommends KEEP (reprojection"
                              + (" and touch" if not validation.get("keep_decision_weak") else " only: weak decision, "
                                 "no held-out touches") + " improve): the offset-fit base (joint-1 offset absorbed in "
                              "the yaw) is written; the offsets are never applied to the robot")
    why = ("the held-out validation was not run" if not validation else
           "the held-out validation could not evaluate the offsets" if validation.get("keep_joint_offsets") is None
           else "the held-out validation does not recommend KEEP")
    return "nominal_fit", f"--joint-offsets was used but {why}: the plain nominal fit is written"


def set_written_base(res: "CalibrationResult", which: str, reason: str = "") -> None:
    """Choose the base / camera that is written to the YAML and re-evaluate everything that depends on it
    (touch-only comparison, check points).

    ``which``: ``"offset_fit"`` = the joint fit with the nuisance joint offsets (its base has the joint-1 offset
    absorbed in the yaw, which is right for nominal FK), ``"nominal_fit"`` = the fit with the offsets fixed to 0
    (identical to the joint fit when no offsets were estimated).  The offsets themselves are never applied to the
    robot."""
    from .base_check import tcp_consistency, touch_only_base
    from .metrology import check_point_report
    rep = res.report
    if which not in ("offset_fit", "nominal_fit"):
        raise SolverError(f"unknown written fit '{which}'")
    if which == "offset_fit" and not (res.flags and res.flags.offsets):
        which = "nominal_fit"
    if which == "nominal_fit":
        T_R, T_C = res.T_world_arm_right_base_nominal_refit, res.T_world_camera_link_nominal_refit
        pose = rep["nominal_refit"]["arm_right_base"]
    else:
        T_R, T_C = res.T_world_arm_right_base, res.T_world_camera_link
        pose = rep["joint"]["arm_right_base"]
    res.written_fit, res.T_written_R, res.T_written_cam = which, T_R, T_C
    rep["written_base"] = {"fit": which, "reason": reason, "std_xyz_mm": pose.get("std_xyz_mm"),
                           "std_rpy_deg": pose.get("std_rpy_deg"), "arm_right_base": _pose_dict(T_R),
                           "camera_link": _pose_dict(T_C),
                           "offsets_note": "fitted joint offsets are never applied to the robot"}
    tcp = (rep.get("touch_base") or {}).get("tcp_consistency")
    tb = touch_only_base(res.touches_all, T_R, res.gate_options)
    tb["tcp_consistency"] = tcp if tcp is not None else tcp_consistency(res.touches_all, res.gate_options)
    rep["touch_base"] = tb
    rep["check_points"] = check_point_report(res.touches_all, T_R)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _fmt(v, p=3):
    return "[" + ", ".join(f"{x:.{p}f}" for x in v) + "]"


def format_report(rep: dict) -> str:
    j = rep["joint"]
    L = []
    L.append(f"Dual-arm calibration  {rep['date']}  dataset: {rep['dataset']}")
    L.append(f"samples used {rep['n_samples_used']} of {rep['n_samples_input']}  "
             f"(rejected {rep['n_samples_rejected']}, corner outliers {rep['n_corner_outliers']})")
    L.append(f"RMS reprojection px: left {j['rms_px']['left']:.3f}  right {j['rms_px']['right']:.3f}  "
             f"all {j['rms_px']['all']:.3f}   sigma0 {j['sigma0_px']:.3f}   cond(J) {j['jacobian_cond']:.2e}")

    def pose_line(name, p):
        xyz = _fmt(p["xyz"], 4)
        rpy = _fmt([math.degrees(v) for v in p["rpy"]], 3)
        s = f"  {name:<16} xyz [m] {xyz}  rpy [deg] {rpy}"
        if "std_xyz_mm" in p:
            s += f"\n  {'':<16} std xyz [mm] {_fmt(p['std_xyz_mm'], 2)}  std rpy [deg] {_fmt(p['std_rpy_deg'], 3)}"
        return s

    L.append("Joint result (world = arm_left_base_link)" +
             (" - WITH fitted nuisance joint offsets (this base is written; the offsets are never applied):" if
              j.get("joint_offsets", {}).get("enabled") else ":"))
    L.append(pose_line("arm_right_base", j["arm_right_base"]))
    L.append(pose_line("camera_link", j["camera_link"]))
    L.append(pose_line("optical_frame", j["optical_frame"]))
    L.append(pose_line("target_left", j["target_offsets"]["left"]) + "   (T_link6_target)")
    L.append(pose_line("target_right", j["target_offsets"]["right"]) + "   (T_link6_target)")
    nm = rep.get("nominal_refit")
    if nm and not nm["same_as_joint"]:
        L.append("Diagnostic: plain nominal-FK refit (offsets fixed to 0, NOT written; touch mismatch "
                 f"{_nz(nm['touch_rms_mm_nominal']['all'] if nm.get('touch_rms_mm_nominal') else None)} mm):")
        L.append(pose_line("arm_right_base", nm["arm_right_base"]))
        L.append(pose_line("camera_link", nm["camera_link"]))
        d = nm["difference_vs_joint"]
        L.append(f"  nominal refit vs offset-fit: right base {d['arm_right_base']['position_mm']:.2f} mm / "
                 f"{d['arm_right_base']['rotation_deg']:.3f} deg, camera {d['camera_link']['position_mm']:.2f} mm / "
                 f"{d['camera_link']['rotation_deg']:.3f} deg")
    if j.get("table_board"):
        b = j["table_board"]
        L.append(pose_line("table_board", b) + f"   ({b['n_table_images']} images)")
    if j.get("joint_offsets", {}).get("enabled"):
        o = j["joint_offsets"]
        L.append(f"Joint zero offsets [deg] (prior sigma {o['prior_sigma_deg']:.2f}; joint1 fixed = base yaw gauge; "
                 "nuisance parameters of this solve, NEVER apply them to the robot/URDF):")
        for a in ARMS:
            L.append(f"  {a:<6} " + "  ".join(
                f"j{i + 1} {v:+.3f}" + ("" if o['std_deg'][a][i] is None else f"+-{o['std_deg'][a][i]:.3f}")
                for i, v in enumerate(o["estimated_deg"][a])))
    ir = j.get("intrinsics")
    if ir and ir.get("delta_px"):
        d, sd = ir["delta_px"], ir["std_px"]
        L.append(("Intrinsics refined" if ir["refined"] else "Intrinsics NOT refined; the data would pull them by")
                 + " [px]: " + "  ".join(f"{k} {d[k]:+.2f}+-{sd[k]:.2f}" for k in ("fx", "fy", "cx", "cy"))
                 + ("" if ir["refined"] or "right_base_shift" not in ir else
                    f"   (right base would move {ir['right_base_shift']['position_mm']:.2f} mm, camera "
                    f"{ir['camera_shift']['position_mm']:.2f} mm)"))
    if ir and ir.get("pull_f_frac") is not None:
        L.append(f"  intrinsics source: {ir.get('source', 'camera_info')}; data pull {100 * ir['pull_f_frac']:.2f} % in f, "
                 f"{ir['pull_c_px']:.1f} px in c")
    wb = rep.get("written_base")
    if wb:
        L.append(f"Written base: {'offset fit (joint-1 offset absorbed in the base yaw; offsets never applied)' if wb['fit'] == 'offset_fit' else 'nominal fit (joint offsets, if any, not used)'}"
                 f" - {wb['reason']}")
    c = rep["separate"]["composed"]
    L.append("3D consistency vs per-sample PnP (upper bound, dominated by PnP depth noise):")
    for a in ARMS:
        e = j["error_3d_mm"][a]
        if e["pnp_position_diff_rms_mm"] is not None:
            L.append(f"  {a:<6} target-origin diff rms {e['pnp_position_diff_rms_mm']:.2f} mm "
                     f"(depth {e['pnp_depth_diff_rms_mm']:.2f}, max {e['pnp_position_diff_max_mm']:.2f}) "
                     f"rot {e['pnp_rotation_diff_rms_deg']:.2f} deg")
    L.append("Separate per-arm solutions (each arm alone, nominal kinematics, no touches):")
    for a in ARMS:
        s_ = rep["separate"][a]
        L.append(f"  {a:<6} n={s_['n_samples']} rms {s_['rms_px']:.3f} px, optical in arm base xyz "
                 f"{_fmt(s_['optical_in_arm_base']['xyz'], 4)}")
    L.append(f"  composed vs joint: right base {c['arm_right_base_diff_vs_joint']['position_mm']:.2f} mm / "
             f"{c['arm_right_base_diff_vs_joint']['rotation_deg']:.3f} deg, camera_link "
             f"{c['camera_link_diff_vs_joint']['position_mm']:.2f} mm / "
             f"{c['camera_link_diff_vs_joint']['rotation_deg']:.3f} deg")
    lc = rep.get("loop_closure")
    if lc:
        L.append(f"Loop closure through the table board (touch sigma {lc['touch_sigma_mm']:.2f} mm, "
                 f"touches L{lc['n_touches']['left']}/R{lc['n_touches']['right']}):")

        def rms_line(label, blk):
            if blk:
                r = blk["rms_mm"]
                L.append(f"  {label:<30} RMS mm: left {_nz(r['left'])}  right {_nz(r['right'])}  all {_nz(r['all'])}")
        L.append(f"  model: {lc.get('model', 'nominal FK')}")
        rms_line("in-fit (touches used)", lc["final"])
        rms_line("camera-only (touches unused)", lc.get("camera_only"))
        for a, blk in lc.get("cross_arm", {}).items():
            rms_line(f"{a} touches vs other arm's", blk)
        w = lc.get("right_base_with_vs_without_touches")
        if w:
            L.append(f"  right base with vs without touches: {w['position_mm']:.2f} mm / {w['rotation_deg']:.3f} deg")
        L.append("  fit residual (FRE, in-sample) of the board touches: " + _nz(lc["final"]["rms_mm"]["all"]) + " mm RMS")
        if lc.get("out_of_fit_rms_mm") is not None:
            L.append(f"  out-of-fit touch mismatch ({lc['out_of_fit_source']}): {lc['out_of_fit_rms_mm']:.2f} mm RMS "
                     "(at the board points only)")
        else:
            L.append(f"  out-of-fit touch mismatch: {lc['out_of_fit_source']}")
        L.append("  per touch (mm): " + "  ".join(f"{t['arm'][0].upper()}{t['point']}:{t['norm_mm']:.2f}"
                                               for t in lc["final"]["per_touch"]))
        if lc.get("camera_only"):
            L.append("  per touch camera-only: " + "  ".join(
                f"{t['arm'][0].upper()}{t['point']}:{t['norm_mm']:.2f}" for t in lc["camera_only"]["per_touch"]))
        oc = lc.get("offset_corrected")
        if oc:
            L.append(f"  diagnostic with fitted offsets applied (the robot does not): in-fit "
                     f"{_nz(oc['final']['rms_mm']['all'])} mm, out-of-fit {_nz(oc['out_of_fit_rms_mm'])} mm "
                     f"({oc['out_of_fit_source']})")
    tb = rep.get("touch_base")
    n_any = {a: (rep.get("n_touches") or {}).get(a, 0) + sum(((rep.get("n_sticker_touches") or {}).get(a) or {}).values())
             for a in ARMS}
    if tb and n_any["left"] and n_any["right"]:
        if tb.get("available"):
            c = tb.get("comparison", {})
            L.append(f"Touch-only right base (Kabsch of {tb['n_common']} common fit points {tb['points']} = board "
                     "crosshairs + fit stickers, no camera):")
            pt = tb["T"]
            from .base_check import pose_xyz_rpy
            L.append(pose_line("arm_right_base", pose_xyz_rpy(pt)))
            L.append(f"  fit residual (FRE, in-sample) RMS {tb['residual_rms_mm']:.2f} mm (dof-corrected "
                     f"{_nz(tb['residual_rms_dof_mm'])} mm); point spread {tb['geometry']['rms_radius_mm']:.0f} mm RMS "
                     f"radius, reach amplification x{tb['geometry']['reach_amplification']:.1f}")
            if c:
                L.append(f"  written (nominal) minus touch-only: {c['origin_mm']:.2f} mm at the base origin, "
                         f"{c['reach_mm']:.2f} mm at the reach point {_fmt(tb['reach_point_in_right_base'], 2)} m, "
                         f"{c['rotation_deg']:.3f} deg (x/y/z {_fmt(c['rotation_components_deg'], 3)})")
            nz = tb.get("noise")
            if nz:
                sc = nz["scatter"]
                L.append(f"  predicted 1-sigma of the touch-only base ({nz['n_mc']} Monte Carlo draws): origin "
                         f"{nz['origin_sigma_mm']:.2f} mm, reach point {nz['reach_sigma_mm']:.2f} mm, rotation x/y/z "
                         f"{_fmt(nz['rotation_sigma_deg'], 3)} deg (yaw {nz['yaw_sigma_deg']:.3f}); measured repeat "
                         f"scatter per axis left {nz['sigma_left_axis_mm']:.2f}{'*' if sc['left']['assumed'] else ''} "
                         f"right {nz['sigma_right_axis_mm']:.2f}{'*' if sc['right']['assumed'] else ''} mm "
                         "(* = assumed, no repeated touches)")
        else:
            L.append(f"Touch-only right base: not available ({tb.get('reason')})")
        for arm, v in tb.get("tcp_consistency", {}).get("arms", {}).items():
            if v["n_points"]:
                w_ = v.get("weakest_direction_tool") or [0, 0, 0]
                L.append(f"TCP consistency {arm}: {v['spread_rms_mm']:.2f} mm RMS spread over {v['n_points']} point(s) "
                         f"touched with tilt diversity (max pair {v['max_pair_mm']:.2f} mm; an indicator, NOT an upper "
                         "bound on the TCP error: it includes joint offset/backlash effects and is blind along the "
                         f"weakest tilt direction [{w_[0]:+.2f}, {w_[1]:+.2f}, {w_[2]:+.2f}] (tool frame, observability "
                         f"{v.get('weakest_sigma_min_norm', float('nan')):.2f}, point {v.get('weakest_point')}))")
            elif v["skipped_similar_orientation"]:
                L.append(f"TCP consistency {arm}: repeated touches only at similar wrist orientations (< "
                         f"{rep['touch_base']['tcp_consistency']['min_orientation_deg']:.0f} deg): not evaluated")
            for ins in v.get("insufficient_tilt", []):
                L.append(f"TCP consistency {arm}: {ins['message']} ({ins['n']} touches, observability "
                         f"{ins['sigma_min_norm']:.2f} < {rep['touch_base']['tcp_consistency']['min_tilt_sigma']:.2f}): "
                         "excluded from the number")
    cp = rep.get("check_points")
    if cp:
        L.append("Check points (" + cp["label"] + "):")
        if cp["points"]:
            for p_ in cp["points"]:
                geo = ("" if "height_above_plane_mm" not in p_ else
                       f", {p_['height_above_plane_mm']:.0f} mm off the board plane, {p_['dist_from_centre_mm']:.0f} mm "
                       "from the board centre")
                L.append(f"  point {p_['point']}: {p_['error_mm']:.2f} mm (L x{p_['n_left']}, R x{p_['n_right']}"
                         f"{geo})")
            L.append(f"  check-point error (TRE, held-out): RMS {cp['rms_mm']:.2f} mm, max {cp['max_mm']:.2f} mm over "
                     f"{cp['n_points']} point(s) - " + cp["scope"])
        else:
            L.append("  none (no check sticker touched by both arms)"
                     + (": touched by one arm only: " + ", ".join(str(i["point"]) for i in cp["incomplete"])
                        if cp.get("incomplete") else ""))
        L.append(f"  recommended_padding_mm = max(20, 2 x max check-point error + 10) = "
                 f"{cp['recommended_padding_mm']:.0f}")
    rp = rep.get("repeatability")
    if rp and rp.get("clusters"):
        L.append("Repeatability (ISO 9283-inspired: RP = mean distance to barycentre + 3 sigma, wrist orientation "
                 f"within {rp['max_orientation_deg']:.0f} deg, >= {rp['min_repeats']} repeats):")
        for c_ in rp["clusters"]:
            L.append(f"  {c_['arm']:<5} point {c_['point']:>3} n={c_['n']}: RP {c_['rp_mm']:.2f} mm "
                     f"(mean {c_['l_mean_mm']:.2f} + 3 x {c_['s_l_mm']:.2f})")
    elif rp and rp.get("skipped"):
        L.append("Repeatability: no (arm, point) has >= " + str(rp["min_repeats"]) + " repeats at one wrist orientation")
    if rep.get("compare"):
        L.append("Reproducibility: " + rep["compare"]["text"])
    mc = rep.get("uncertainty_mc")
    if mc:
        from .uncertainty import format_mc
        L += format_mc(mc)
    si = rep.get("session_info")
    if si:
        for lab, d_ in si:
            vals = ", ".join(f"{k} {v}" for k, v in d_.items() if v not in (None, ""))
            if vals:
                L.append(f"Session {lab}: {vals}")
    ob = rep.get("observability")
    if ob:
        L.append(f"Observability (data-only Jacobian, priors excluded): smallest normalised singular value "
                 f"{ob['smallest']:.2e} (warn below {ob['threshold']:.0e}); weakest directions:")
        for w in ob["weak_directions"][:3]:
            L.append(f"  sv {w['normalised_sv']:.2e}{' (structural)' if w['structural'] else ''}: " + " + ".join(
                f"{p['name']}({p['weight']:+.2f})" for p in w["top_parameters"]))
    bad = [s for s in rep["samples"] if s["n_corner_outliers"] or s["rms_px"] > 2 * j["rms_px"]["all"] + 0.5]
    if bad:
        L.append("Samples with corner outliers or high residual:")
        for s in bad:
            L.append(f"  {s['arm']}#{s['index']} {s['image']} rms {s['rms_px']:.2f} max {s['max_px']:.2f} "
                     f"corner outliers {s['n_corner_outliers']}")
    if "validation" in rep and rep["validation"]:
        from .validation import format_validation
        L += format_validation(rep["validation"])
    if rep.get("trust_gate"):
        L += rep["trust_gate"]["text"]
    cr = rep.get("correlation")
    if cr and cr["high_pairs"]:
        L.append(f"Parameter correlation |r| > {cr['threshold']} ({len(cr['high_pairs'])} pairs; full matrix in the "
                 "JSON report):")
        for p in cr["high_pairs"][:8]:
            L.append(f"  {p['a']:<22} ~ {p['b']:<22} {p['corr']:+.3f}" + ("  (structural)" if p["structural"] else ""))
    L.append("STD DEVS: " + " ".join(rep.get("accuracy_statement", [])))
    if rep.get("notes"):
        L.append("NOTES:")
        L += [f"  - {w}" for w in rep["notes"]]
    if rep["warnings"]:
        L.append("WARNINGS:")
        L += [f"  - {w}" for w in rep["warnings"]]
    else:
        L.append("No conditioning warnings.")
    return "\n".join(L)


def _nz(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def deployed_poses(res, base_from: str = "joint"):
    """(T_world_arm_right_base, T_world_camera_link) that are written to the YAML: the written fit chosen by
    :func:`set_written_base` (the offset fit, whose base has the joint-1 offset absorbed in the yaw, when
    ``--joint-offsets`` was used and the held-out validation recommends KEEP; otherwise the nominal fit; the offsets
    themselves are never applied), or with ``base_from='touches'`` the touch-only base."""
    T_R = getattr(res, "T_written_R", None)
    if T_R is not None:
        T_R, T_C = T_R, res.T_written_cam
    else:
        T_R, T_C = res.T_world_arm_right_base, res.T_world_camera_link
    if base_from == "touches":
        T_t = getattr(res, "T_world_arm_right_base_touch", None)
        if T_t is None:
            raise SolverError("--base-from touches: no touch-only base estimate (need >= 3 non-collinear board "
                              "points touched by BOTH arms, same point ids): "
                              + str((getattr(res, "report", {}).get("touch_base") or {}).get("reason", "")))
        T_R = T_t
    elif base_from != "joint":
        raise SolverError(f"unknown base_from '{base_from}'")
    return T_R, T_C


def calibration_yaml_dict(res: CalibrationResult, notes: str, date: Optional[str] = None,
                          source: str = "calibrated", base_from: str = "joint") -> dict:
    """Exactly the structure consumed by rebel_dual_table_description.  ``source`` is 'calibrated' only when
    the trust gate passed, else 'calibrated_unverified'."""
    T_R, T_C = deployed_poses(res, base_from)
    xr, rr = T_to_xyz_rpy(T_R)
    xc, rc = T_to_xyz_rpy(T_C)
    return {"dual_arm_calibration": {
        "source": source,
        "date": date or res.report["date"],
        "world_frame": "world",
        "arm_right_base": {"xyz": [float(v) for v in xr], "rpy": [float(v) for v in rr]},
        "camera_link": {"xyz": [float(v) for v in xc], "rpy": [float(v) for v in rc]},
        "notes": notes,
    }}


def write_calibration_yaml(path: str, res: CalibrationResult, notes: str, date: Optional[str] = None,
                           source: str = "calibrated", base_from: str = "joint") -> None:
    """Write the YAML with a stable key order and full float precision."""
    d = calibration_yaml_dict(res, notes, date, source, base_from)["dual_arm_calibration"]

    def vec(v):
        return "[" + ", ".join(repr(float(x)) for x in v) + "]"

    notes_q = json.dumps(d["notes"])  # JSON string is a valid YAML double-quoted scalar
    text = (
        "dual_arm_calibration:\n"
        f"  source: {d['source']}\n"
        f"  date: \"{d['date']}\"\n"
        f"  world_frame: {d['world_frame']}        # coincident with arm_left_base_link\n"
        "  arm_right_base:           # pose of arm_right_base_link in world\n"
        f"    xyz: {vec(d['arm_right_base']['xyz'])}\n"
        f"    rpy: {vec(d['arm_right_base']['rpy'])}       # URDF fixed-axis roll/pitch/yaw [rad]\n"
        "  camera_link:              # pose of camera_link in world\n"
        f"    xyz: {vec(d['camera_link']['xyz'])}\n"
        f"    rpy: {vec(d['camera_link']['rpy'])}\n"
        f"  notes: {notes_q}\n"
    )
    with open(path, "w") as f:
        f.write(text)
