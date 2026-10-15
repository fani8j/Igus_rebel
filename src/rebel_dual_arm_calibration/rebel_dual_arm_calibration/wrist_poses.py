"""Pose generation for the eye-in-hand calibration (pure maths, no ROS).

Everything is relative to the SEED: the operator jogs the arm until the table board is fully visible in the wrist image and
presses "Use as seed".  Candidate views are rigid rotations of the seed link6 pose about points / lines that only depend on
the BOARD centre and on the camera->board ray, never on the exact camera mount:

    orbit   rotate about an axis through the board centre that lies in the board plane (azimuth az), by a tilt of 0..30 deg;
            a rigid rotation about a point keeps the board centre at the SAME place in the camera image whatever the
            hand-eye transform is (this is what makes the orbit robust to an inaccurate camera-mount guess)
    roll    rotate about the camera->board-centre line by +-15..45 deg (needed for hand-eye observability)
    range   move along that line to 0.8..1.2 x the seed distance (never closer than ``min_dist_m``)

Each candidate is checked against the CURRENT estimate of X (the seed-derived estimate, else the CAD guess below, else the
interim solve): all four board corners (and the edge mid-points) must project inside the image with a margin, the board
must face the camera, and the range must be sane.  The tilts use the board's REAL normal (the board can stand on a tilted
holder), not base +z.

Nominal camera mount (ASSUMPTION, rough): the Gemini 305 body of the knife_c model (meshes/knife_c/visual/
gemini_305_body.stl, authored in igus_tool0 coordinates) is an 8-vertex box; its front face (the large face whose normal
points along +tool z, towards the knife tip) has the centre and normal below.  The colour lens is assumed at that face
centre with optical +Z along the face normal.  The image roll about +Z is NOT recoverable from the box (42 x 42 mm face): it
is a parameter (``roll_deg``, default 0: optical x = tool +Y).  igus_tool0 = link6 + 0.0527 m z, rotated pi about z (from
igus_rebel_xeg32_wrapper.urdf.xacro).  Expect errors of several mm / several degrees: the seed estimate replaces it as soon as
the board pose in the base is known.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .eye_in_hand import project
from .transforms import inv_T, make_T, rotvec_to_R

# --- CAD nominal --------------------------------------------------------------------------------------
LINK6_TO_TOOL0_XYZ = (0.0, 0.0, 0.0527)
LINK6_TO_TOOL0_YAW = math.pi
CAD_FRONT_CENTRE_TOOL0 = np.array([-0.0201, -0.0920, -0.0438])       # front face centre of gemini_305_body.stl
CAD_OPTICAL_Z_TOOL0 = np.array([-0.30901699, 0.0, 0.95105652])      # front face normal (18 deg off tool +z, towards -x)


def nominal_T_link6_optical(roll_deg: float = 0.0) -> np.ndarray:
    """CAD-based rough guess of T_link6_optical (see module docstring for the assumptions)."""
    T_l6_t0 = make_T(rotvec_to_R([0, 0, LINK6_TO_TOOL0_YAW]), LINK6_TO_TOOL0_XYZ)
    z = CAD_OPTICAL_Z_TOOL0 / np.linalg.norm(CAD_OPTICAL_Z_TOOL0)
    x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    R = np.column_stack([x, y, z]) @ rotvec_to_R([0, 0, math.radians(roll_deg)])
    return T_l6_t0 @ make_T(R, CAD_FRONT_CENTRE_TOOL0)


# --- options --------------------------------------------------------------------------------------------
@dataclass
class PoseOptions:
    tilts_deg: Tuple[float, ...] = (10.0, 20.0, 30.0)
    n_azimuths: int = 8
    rolls_deg: Tuple[float, ...] = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
    dist_factors: Tuple[float, ...] = (0.8, 1.0, 1.2)
    min_dist_m: float = 0.25
    max_dist_m: float = 0.65
    image_margin: float = 0.10                # fraction of width / height kept free at every image border
    max_view_angle_deg: float = 65.0          # between the board normal and the line of sight
    max_joint_jump_deg: float = 90.0
    max_diverse: int = 90                     # candidates kept after the diversity pre-selection (before IK)
    target_count: int = 18


@dataclass(eq=False)
class Candidate:
    T_base_link6: np.ndarray
    desc: str
    tilt_deg: float
    az_deg: float
    roll_deg: float
    dist_factor: float
    info: dict = field(default_factory=dict)
    q: Optional[Dict[str, float]] = None


# --- rigid motions --------------------------------------------------------------------------------------
def rotate_about(point: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """4x4 rigid motion: rotation by angle about the line through ``point`` with direction ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R = rotvec_to_R(axis * angle_rad)
    return make_T(R, point - R @ point)


def board_axes(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, outward normal) of the table board in base.  The board frame has z INTO the paper (a printed board
    object frame x right, y down), so the normal pointing out of the paper is -z."""
    return B[:3, 0].copy(), B[:3, 1].copy(), -B[:3, 2].copy()


def outline_points(board_size: Tuple[float, float]) -> np.ndarray:
    """Board outline corners and edge mid-points in the board frame (z = 0), shape (8, 3)."""
    w, h = board_size[0] / 2.0, board_size[1] / 2.0
    return np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0], [0, -h, 0], [w, 0, 0], [0, h, 0], [-w, 0, 0]],
                    dtype=float)


def board_corners_base(B: np.ndarray, board_size: Tuple[float, float]) -> np.ndarray:
    """The 4 outline corners of the board in base, (4, 3)."""
    pts = outline_points(board_size)[:4]
    return (B[:3, :3] @ pts.T).T + B[:3, 3]


def view_check(T_base_link6: np.ndarray, X: np.ndarray, B: np.ndarray, K: np.ndarray, dist: np.ndarray,
               image_size: Tuple[int, int], board_size: Tuple[float, float], opt: PoseOptions) -> Tuple[bool, str, dict]:
    """Predicted board visibility for a link6 pose.  Returns (ok, reason, info)."""
    T_cam = T_base_link6 @ X
    T_ob = inv_T(T_cam) @ B
    info: dict = {}
    centre_cam = T_ob[:3, 3]
    d = float(np.linalg.norm(centre_cam))
    info["distance_m"] = d
    if centre_cam[2] <= 0.0:
        return False, "board behind the camera", info
    if d < opt.min_dist_m:
        return False, f"too close ({d:.2f} m < {opt.min_dist_m:g} m)", info
    if d > opt.max_dist_m:
        return False, f"too far ({d:.2f} m > {opt.max_dist_m:g} m)", info
    _, _, n_out = board_axes(B)
    to_cam = T_cam[:3, 3] - B[:3, 3]
    cosang = float(n_out @ to_cam) / max(np.linalg.norm(to_cam), 1e-9)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    info["view_angle_deg"] = ang
    if cosang <= 0.0:
        return False, "camera behind the board plane", info
    if ang > opt.max_view_angle_deg:
        return False, f"board seen at {ang:.0f} deg from its normal (> {opt.max_view_angle_deg:g})", info
    pts_cam = (T_ob[:3, :3] @ outline_points(board_size).T).T + T_ob[:3, 3]
    if np.any(pts_cam[:, 2] <= 1e-3):
        return False, "part of the board behind the camera plane", info
    uv = project(T_ob, outline_points(board_size), K, dist)
    w, h = image_size
    lo = np.array([opt.image_margin * w, opt.image_margin * h])
    hi = np.array([(1 - opt.image_margin) * w, (1 - opt.image_margin) * h])
    margin = float(min(np.min(uv - lo), np.min(hi - uv)))
    info["margin_px"] = margin
    info["outline_px"] = uv[:4].tolist()
    if margin < 0.0:
        return False, f"board outside the image margin ({margin:.0f} px)", info
    return True, "", info


# --- generation -------------------------------------------------------------------------------------
def _describe(az_deg, tilt_deg, roll_deg, f) -> str:
    parts = []
    parts.append("seed view" if tilt_deg == 0 else f"orbit {tilt_deg:g} deg @ {az_deg:.0f} deg")
    if roll_deg:
        parts.append(f"roll {roll_deg:+.0f} deg")
    if abs(f - 1.0) > 1e-9:
        parts.append(f"range x{f:g}")
    return ", ".join(parts)


def generate_poses(A0: np.ndarray, X: np.ndarray, B: np.ndarray, K: np.ndarray, dist: np.ndarray,
                   image_size: Tuple[int, int], board_size: Tuple[float, float],
                   opt: PoseOptions = PoseOptions()) -> Tuple[List[Candidate], Dict[str, int]]:
    """All candidate link6 poses (rigid motions of the seed pose A0) whose predicted view passes ``view_check``, in a
    diversity order (farthest-point over camera orientation / position, the seed view first).  Also the rejection
    counts by reason."""
    c = B[:3, 3].copy()
    bx, by, _ = board_axes(B)
    o0 = (A0 @ X)[:3, 3]
    d0 = float(np.linalg.norm(c - o0))
    rejected: Dict[str, int] = {}
    cands: List[Candidate] = []
    orbits: List[Tuple[float, float]] = [(0.0, 0.0)]
    for t in opt.tilts_deg:
        for k in range(opt.n_azimuths):
            orbits.append((t, 360.0 * k / opt.n_azimuths))
    for tilt, az in orbits:
        axis = math.cos(math.radians(az)) * bx + math.sin(math.radians(az)) * by
        A1 = rotate_about(c, axis, math.radians(tilt)) @ A0 if tilt else A0.copy()
        o1 = (A1 @ X)[:3, 3]
        line = c - o1
        dist1 = float(np.linalg.norm(line))
        if dist1 < 1e-6:
            continue
        u = line / dist1
        for roll in opt.rolls_deg:
            A2 = rotate_about(c, u, math.radians(roll)) @ A1 if roll else A1
            for f in opt.dist_factors:
                A3 = A2.copy()
                A3[:3, 3] += (1.0 - f) * d0 * u
                if not np.all(np.isfinite(A3)):
                    continue
                ok, why, info = view_check(A3, X, B, K, dist, image_size, board_size, opt)
                if not ok:
                    key = why.split(" (")[0].split(" at ")[0] if why else "rejected"
                    rejected[key] = rejected.get(key, 0) + 1
                    continue
                cands.append(Candidate(A3, _describe(az, tilt, roll, f), tilt, az, roll, f, info))
    ordered = diverse_order(cands, A0, X, opt.max_diverse)
    return ordered, rejected


def _orientation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def diverse_order(cands: Sequence[Candidate], A0: np.ndarray, X: np.ndarray, limit: int) -> List[Candidate]:
    """Greedy farthest-point order (max-min of camera rotation angle + 0.5 rad per metre of position), the seed pose
    (smallest tilt / roll / range change) first."""
    if not cands:
        return []
    poses = [(c.T_base_link6 @ X) for c in cands]
    ref = A0 @ X
    first = min(range(len(cands)), key=lambda i: _orientation_angle(poses[i][:3, :3], ref[:3, :3])
                + np.linalg.norm(poses[i][:3, 3] - ref[:3, 3]))
    chosen = [first]
    dmin = np.array([_pair(poses[first], p) for p in poses])
    while len(chosen) < min(limit, len(cands)):
        nxt = int(np.argmax(dmin))
        if dmin[nxt] <= 1e-9:
            break
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.array([_pair(poses[nxt], p) for p in poses]))
    return [cands[i] for i in chosen]


def _pair(Ta: np.ndarray, Tb: np.ndarray) -> float:
    return _orientation_angle(Ta[:3, :3], Tb[:3, :3]) + 0.5 * float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def order_by_joint_distance(q_start: Sequence[float], items: Sequence[Tuple[object, Sequence[float]]]) -> List[object]:
    """Greedy nearest-neighbour order of (key, q) pairs by joint-space distance from ``q_start``."""
    remaining = list(items)
    cur = np.asarray(q_start, dtype=float)
    out = []
    while remaining:
        k = min(range(len(remaining)), key=lambda i: float(np.linalg.norm(np.asarray(remaining[i][1]) - cur)))
        key, q = remaining.pop(k)
        out.append(key)
        cur = np.asarray(q, dtype=float)
    return out


def max_joint_change_deg(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    return float(np.degrees(np.max(np.abs(np.asarray(q_a, dtype=float) - np.asarray(q_b, dtype=float)))))
