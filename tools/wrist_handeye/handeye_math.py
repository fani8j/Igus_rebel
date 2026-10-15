"""Pure geometry / hand-eye maths for wrist_handeye.py (no ROS imports; unit-tested).

Conventions
-----------
* T_a_b is the 4x4 pose of frame b expressed in frame a (p_a = T_a_b @ p_b).
* The camera frame is the OPTICAL frame (x right, y down, z forward).
* The board frame is the frame of ``targets.TargetSpec`` object points: origin = board CENTRE,
  x right, y down, z = board normal (into the board when seen from the front).
* X = T_link6_optical is what the hand-eye calibration solves for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------------------------------
# SE(3) helpers
# ----------------------------------------------------------------------------------------------


def make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def T_from_xyz_quat(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=float)
    return make_T(Rot.from_quat(q / np.linalg.norm(q)).as_matrix(), [x, y, z])


def T_to_xyz_quat(T: np.ndarray) -> Tuple[List[float], List[float]]:
    q = Rot.from_matrix(T[:3, :3]).as_quat()
    if q[3] < 0:
        q = -q
    return [float(v) for v in T[:3, 3]], [float(v) for v in q]


def T_to_xyz_rpy(T: np.ndarray) -> Tuple[List[float], List[float]]:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll) (fixed-axis xyz)."""
    rpy = Rot.from_matrix(T[:3, :3]).as_euler("xyz")
    return [float(v) for v in T[:3, 3]], [float(v) for v in rpy]


def T_from_xyz_rpy(x, y, z, roll, pitch, yaw) -> np.ndarray:
    return make_T(Rot.from_euler("xyz", [roll, pitch, yaw]).as_matrix(), [x, y, z])


def rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pose_diff(T1: np.ndarray, T2: np.ndarray) -> Tuple[float, float]:
    """(translation difference in mm, rotation difference in deg)."""
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]) * 1000.0), rot_angle_deg(T1[:3, :3], T2[:3, :3])


# ----------------------------------------------------------------------------------------------
# Nominal camera mount (from the CAD, ROUGH)
# ----------------------------------------------------------------------------------------------
# link `knife_c_wrist_camera` is a child of igus_tool0 with identity origin; its meshes are authored in igus_tool0
# coordinates.  igus_tool0 = link6 * T(0,0,0.0527) * Rz(pi).
#
# meshes/knife_c/visual/gemini_305_body.stl is a plain 8-vertex box, 42 mm (y) x 41.9 mm (u) x 23.1 mm (n), rotated
# 18 deg about tool0 y.  The two big faces (42 x 42 mm) have normals +/-(0.309, 0, -0.951); the camera looks FORWARD
# (towards +z of tool0, the knife side), so the front face is the one with normal n_f = (-sin18, 0, cos18) and its centre
# (-0.0201, -0.092, -0.0438) is taken as the optical centre ("lens near the centre of the front face").
# Optical +z = n_f.  The image axes cannot be read from a square box: ASSUMPTION optical x (image right) = tool0 +y,
# hence optical y (image down) = z x x = (-0.951, 0, -0.309).  The unknown in-plane rotation only moves the board
# around in the image; the orbit poses are generated for the camera optical axis so this has little effect on the
# plan, and `wrist_handeye.py` re-estimates X from the live view when the board is visible at start.  Override with
# --mount-xyz/--mount-rpy or --mount-yaml.
LINK6_TO_TOOL0 = make_T(Rot.from_euler("z", math.pi).as_matrix(), [0.0, 0.0, 0.0527])
_S18, _C18 = math.sin(math.radians(18.0)), math.cos(math.radians(18.0))
NOMINAL_OPTICAL_CENTRE_TOOL0 = np.array([-0.0201016, -0.092, -0.0438283])
NOMINAL_OPTICAL_Z_TOOL0 = np.array([-_S18, 0.0, _C18])
NOMINAL_OPTICAL_X_TOOL0 = np.array([0.0, 1.0, 0.0])


def nominal_T_tool0_optical() -> np.ndarray:
    z = NOMINAL_OPTICAL_Z_TOOL0
    x = NOMINAL_OPTICAL_X_TOOL0
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), NOMINAL_OPTICAL_CENTRE_TOOL0)


def nominal_T_link6_optical() -> np.ndarray:
    return LINK6_TO_TOOL0 @ nominal_T_tool0_optical()


def derive_nominal_from_mesh(stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(front-face centre, front-face normal) of gemini_305_body.stl in tool0 coordinates (needs trimesh).

    The front face is the largest-area face whose normal has a positive z component."""
    import trimesh  # optional dependency, only for this cross-check

    m = trimesh.load(stl_path)
    best = None
    for n in np.unique(np.round(m.face_normals, 3), axis=0):
        sel = np.abs(m.face_normals - n).max(axis=1) < 2e-3
        area = m.area_faces[sel].sum()
        if n[2] > 0.5 and (best is None or area > best[0]):
            best = (area, n, (m.triangles_center[sel] * m.area_faces[sel, None]).sum(axis=0) / area)
    return best[2], best[1]


# ----------------------------------------------------------------------------------------------
# Board / orbit geometry
# ----------------------------------------------------------------------------------------------


def board_corners_obj(w: float, h: float) -> np.ndarray:
    return np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=float)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def board_up_normal(T_base_board: np.ndarray) -> np.ndarray:
    """Unit board normal pointing to the side the camera views from (positive base z)."""
    n = T_base_board[:3, 2]
    return n if n[2] >= 0 else -n


def look_at_pose(T_base_board: np.ndarray, cam_pos: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """T_base_cam (optical): origin at cam_pos, z towards ``target``.  Roll 0 puts the image x axis along the board
    x axis (projected), then the camera is rolled by roll_deg about its view axis."""
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    ref = T_base_board[:3, 0]
    x0 = ref - np.dot(ref, z) * z
    if np.linalg.norm(x0) < 1e-6:
        ref = T_base_board[:3, 1]
        x0 = ref - np.dot(ref, z) * z
    x0 = x0 / np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    R0 = np.column_stack([x0, y0, z])
    R = R0 @ Rot.from_euler("z", roll_deg, degrees=True).as_matrix()
    return make_T(R, cam_pos)


@dataclass
class Candidate:
    T_base_cam: np.ndarray
    tilt: float
    azimuth: float
    roll: float
    dist: float
    tag: str = "orbit"
    q: Optional[np.ndarray] = None  # IK solution
    T_base_link6: Optional[np.ndarray] = None


def orbit_candidates(T_base_board: np.ndarray, tilts=(0, 15, 25, 35), n_az=8, rolls=(-40, -20, 0, 20, 40),
                     dists=(0.30, 0.36, 0.42)) -> List[Candidate]:
    """Cameras on a spherical shell about the board centre, all looking at it."""
    centre = T_base_board[:3, 3]
    u = board_up_normal(T_base_board)
    e1 = T_base_board[:3, 0] - np.dot(T_base_board[:3, 0], u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    out = []
    for tilt in tilts:
        azs = [0.0] if tilt == 0 else [360.0 * i / n_az for i in range(n_az)]
        for az in azs:
            th, ph = math.radians(tilt), math.radians(az)
            dirn = math.cos(th) * u + math.sin(th) * (math.cos(ph) * e1 + math.sin(ph) * e2)
            for d in dists:
                pos = centre + d * dirn
                for roll in rolls:
                    out.append(Candidate(look_at_pose(T_base_board, pos, centre, roll), float(tilt), az, float(roll), float(d)))
    return out


def project_board(T_base_cam: np.ndarray, T_base_board: np.ndarray, obj: np.ndarray, K: np.ndarray, D: np.ndarray):
    """Project board-frame points into the image of a camera at T_base_cam.  Returns (pix Nx2, depth N)."""
    T_cb = inv_T(T_base_cam) @ T_base_board
    pc = transform_points(T_cb, obj)
    if np.any(pc[:, 2] <= 1e-3):
        return None, pc[:, 2]
    pix, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return pix.reshape(-1, 2), pc[:, 2]


def corners_in_view(T_base_cam, T_base_board, w, h, K, D, size_wh, margin=0.10) -> bool:
    pix, _ = project_board(T_base_cam, T_base_board, board_corners_obj(w, h), K, D)
    if pix is None:
        return False
    W, H = size_wh
    return bool(np.all(pix[:, 0] >= margin * W) and np.all(pix[:, 0] <= (1 - margin) * W)
                and np.all(pix[:, 1] >= margin * H) and np.all(pix[:, 1] <= (1 - margin) * H))


def select_diverse(rots: Sequence[np.ndarray], n_pick: int, preselected: Sequence[int] = (),
                   first: Optional[int] = None) -> List[int]:
    """Greedy max-min rotation distance.  ``preselected`` are indices already chosen (kept out of the result)."""
    n = len(rots)
    if n == 0:
        return []
    chosen = list(preselected)
    picked: List[int] = []
    if not chosen:
        i0 = 0 if first is None else first
        chosen.append(i0)
        picked.append(i0)
    mind = np.full(n, np.inf)
    for c in chosen:
        mind = np.minimum(mind, [rot_angle_deg(rots[c], rots[j]) for j in range(n)])
    while len(picked) < n_pick and len(picked) + len(preselected) < n:
        mind_masked = mind.copy()
        mind_masked[chosen] = -1.0
        j = int(np.argmax(mind_masked))
        if mind_masked[j] < 0:
            break
        chosen.append(j)
        picked.append(j)
        mind = np.minimum(mind, [rot_angle_deg(rots[j], rots[k]) for k in range(n)])
    return picked


def order_by_joint_distance(Q: Sequence[np.ndarray], q_start: np.ndarray) -> List[int]:
    """Greedy nearest-neighbour ordering of joint configurations starting from q_start."""
    left = list(range(len(Q)))
    order, cur = [], np.asarray(q_start, dtype=float)
    while left:
        j = min(left, key=lambda k: float(np.linalg.norm(Q[k] - cur)))
        order.append(j)
        left.remove(j)
        cur = Q[j]
    return order


def plan_sequence(pool: List[Candidate], q_start: np.ndarray, ik_fn: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]],
                  X_link6_cam: np.ndarray, n_poses: int, max_step_deg: float = 90.0, max_j6_excursion: float = 3.14,
                  forced_first: Optional[Candidate] = None, max_rounds: int = 5, stop: Callable[[], None] = lambda: None):
    """Pick ~n_poses diverse candidates, order them by joint distance and re-solve IK along the chain.

    pool: candidates that already passed the first-stage filters (``q`` set, seeded from the start state).
    Returns (sequence, reasons) where reasons counts the rejections made here.
    """
    reasons: Dict[str, int] = {}

    def bump(k):
        reasons[k] = reasons.get(k, 0) + 1

    rots = [c.T_base_cam[:3, :3] for c in pool]
    remaining = set(range(len(pool)))
    selected: List[int] = []
    accepted: List[Candidate] = []
    for _ in range(max_rounds):
        need = n_poses - len(accepted)
        if need <= 0 or not remaining:
            break
        rem = sorted(remaining)
        if accepted:
            # diversity with respect to what is already accepted: put accepted rotations first
            allr = [c.T_base_cam[:3, :3] for c in accepted] + [rots[i] for i in rem]
            pre = list(range(len(accepted)))
            picks = select_diverse(allr, need, preselected=pre)
            new = [rem[p - len(accepted)] for p in picks]
        else:
            first = None
            if forced_first is not None:
                # forced first candidate is accepted outside the pool
                allr = [forced_first.T_base_cam[:3, :3]] + [rots[i] for i in rem]
                picks = select_diverse(allr, need, preselected=[0])
                new = [rem[p - 1] for p in picks]
            else:
                picks = select_diverse([rots[i] for i in rem], need)
                new = [rem[p] for p in picks]
        for i in new:
            remaining.discard(i)
        trial = ([forced_first] if forced_first is not None and not accepted else []) + accepted + [pool[i] for i in new]
        # order by joint distance, then verify along the chain with fresh IK
        Q = [c.q for c in trial]
        order = order_by_joint_distance(Q, q_start)
        chain: List[Candidate] = []
        prev = np.asarray(q_start, dtype=float)
        for oi in order:
            stop()
            c = trial[oi]
            T6 = c.T_base_cam @ inv_T(X_link6_cam)
            q = ik_fn(T6, prev)
            if q is None:
                bump("ik_fail_in_chain")
                continue
            if np.max(np.abs(q - prev)) > math.radians(max_step_deg):
                bump("joint_step>%gdeg" % max_step_deg)
                continue
            if abs(q[5] - q_start[5]) > max_j6_excursion:
                bump("joint6_travel>%.2f" % max_j6_excursion)
                continue
            c2 = Candidate(c.T_base_cam, c.tilt, c.azimuth, c.roll, c.dist, c.tag, q=q, T_base_link6=T6)
            chain.append(c2)
            prev = q
        accepted = chain
        forced_first = None  # only forced into the first round
    return accepted, reasons


# ----------------------------------------------------------------------------------------------
# Safety scene geometry
# ----------------------------------------------------------------------------------------------


def rect_axis_distance(xmin, xmax, ymin, ymax) -> float:
    """Distance from the z axis (origin in xy) to an axis-aligned rectangle."""
    dx = max(xmin, 0.0, -xmax)
    dy = max(ymin, 0.0, -ymax)
    return math.hypot(dx, dy)


def build_slab(corners_base: np.ndarray, top_offset=0.05, side=1.2, min_axis_dist=0.15, thickness=0.30) -> dict:
    """Horizontal keep-out slab.  top = highest board corner z + top_offset; the footprint (side x side centred on the
    board) is clipped by the half-plane (x>=d, x<=-d, y>=d or y<=-d) that keeps the most area while staying >= d from
    the base z axis and still covering the board corners."""
    zmax = float(np.max(corners_base[:, 2]))
    top = zmax + top_offset
    cx = float(np.mean(corners_base[:, 0]))
    cy = float(np.mean(corners_base[:, 1]))
    rect = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)  # xmin xmax ymin ymax
    d = min_axis_dist
    options = []
    if rect_axis_distance(*rect) >= d:
        options.append(rect)
    else:
        xmin, xmax, ymin, ymax = rect
        options += [(max(xmin, d), xmax, ymin, ymax), (xmin, min(xmax, -d), ymin, ymax),
                    (xmin, xmax, max(ymin, d), ymax), (xmin, xmax, ymin, min(ymax, -d))]
    best = None
    bx0, bx1 = corners_base[:, 0].min(), corners_base[:, 0].max()
    by0, by1 = corners_base[:, 1].min(), corners_base[:, 1].max()
    for r in options:
        if r[1] - r[0] <= 0 or r[3] - r[2] <= 0:
            continue
        if rect_axis_distance(*r) < d - 1e-9:
            continue
        if not (r[0] <= bx0 and r[1] >= bx1 and r[2] <= by0 and r[3] >= by1):
            continue
        area = (r[1] - r[0]) * (r[3] - r[2])
        if best is None or area > best[0]:
            best = (area, r)
    if best is None:
        raise ValueError("cannot build a slab that covers the board and stays %.2f m from the base axis" % d)
    xmin, xmax, ymin, ymax = best[1]
    return {"center": [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), top - thickness / 2],
            "size": [xmax - xmin, ymax - ymin, thickness], "top": top, "zmax_corner": zmax}


def build_board_stand(corners_base: np.ndarray, slab_top: float, margin=0.05, extra=0.03) -> dict:
    """Box over the board footprint (+margin), from the slab top up to slab_top + extra (see README, Mismatches)."""
    x0, x1 = corners_base[:, 0].min() - margin, corners_base[:, 0].max() + margin
    y0, y1 = corners_base[:, 1].min() - margin, corners_base[:, 1].max() + margin
    return {"center": [0.5 * (x0 + x1), 0.5 * (y0 + y1), slab_top + extra / 2],
            "size": [x1 - x0, y1 - y0, extra], "top": slab_top + extra}


def inflate_box(center, size, margin=0.05) -> dict:
    return {"center": [float(c) for c in center], "size": [float(s) + 2.0 * margin for s in size]}


# ----------------------------------------------------------------------------------------------
# Hand-eye solve
# ----------------------------------------------------------------------------------------------


@dataclass
class Sample:
    T_base_link6: np.ndarray
    obj: np.ndarray  # Nx3 board-frame points
    pix: np.ndarray  # Nx2 detected pixels
    name: str = ""


def solve_pnp(obj: np.ndarray, pix: np.ndarray, K: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """T_cam_board and RMS reprojection (px)."""
    obj = np.ascontiguousarray(obj, dtype=np.float64).reshape(-1, 3)
    pix = np.ascontiguousarray(pix, dtype=np.float64).reshape(-1, 1, 2)
    flat = np.ptp(obj[:, 2]) < 1e-9
    ok, rv, tv = False, None, None
    if flat and len(obj) >= 4:
        try:
            ok, rv, tv = cv2.solvePnP(obj, pix, K, D, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            ok = False
    if not ok:
        ok, rv, tv = cv2.solvePnP(obj, pix, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rv, tv = cv2.solvePnPRefineLM(obj, pix, K, D, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pix.reshape(-1, 2)) ** 2, axis=1))))
    return make_T(cv2.Rodrigues(rv)[0], tv.ravel()), rms


REFINE_MAX_MM = 5.0
REFINE_MAX_DEG = 2.0


def _pack(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.concatenate([Rot.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3], Rot.from_matrix(B[:3, :3]).as_rotvec(), B[:3, 3]])


def _unpack(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return make_T(Rot.from_rotvec(p[0:3]).as_matrix(), p[3:6]), make_T(Rot.from_rotvec(p[6:9]).as_matrix(), p[9:12])


def reprojection_residuals(p, samples: Sequence[Sample], K, D) -> np.ndarray:
    X, B = _unpack(p)
    res = []
    for s in samples:
        T_cb = inv_T(s.T_base_link6 @ X) @ B
        rv = cv2.Rodrigues(T_cb[:3, :3])[0]
        proj, _ = cv2.projectPoints(s.obj.reshape(-1, 1, 3), rv, T_cb[:3, 3], K, D)
        res.append((proj.reshape(-1, 2) - s.pix.reshape(-1, 2)).ravel())
    return np.concatenate(res)


def refine(samples: Sequence[Sample], K, D, X0, B0, loss="huber", f_scale=1.0, max_nfev=60) -> Tuple[np.ndarray, np.ndarray, "least_squares"]:
    sol = least_squares(reprojection_residuals, _pack(X0, B0), args=(samples, K, D), loss=loss, f_scale=f_scale,
                        x_scale="jac", max_nfev=max_nfev)
    X, B = _unpack(sol.x)
    return X, B, sol


def board_poses_from_pnp(samples, T_cb_list, X) -> List[np.ndarray]:
    return [s.T_base_link6 @ X @ T_cb for s, T_cb in zip(samples, T_cb_list)]


def mean_pose(Ts: Sequence[np.ndarray]) -> np.ndarray:
    R = Rot.from_matrix(np.array([T[:3, :3] for T in Ts])).mean().as_matrix()
    return make_T(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def consistency(Ts: Sequence[np.ndarray], ref: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Spread of a list of board poses (mm, deg) about ``ref`` (default: their mean)."""
    ref = mean_pose(Ts) if ref is None else ref
    d = [pose_diff(T, ref) for T in Ts]
    mm = np.array([a for a, _ in d])
    dg = np.array([b for _, b in d])
    return {"pos_rms_mm": float(np.sqrt(np.mean(mm ** 2))), "pos_max_mm": float(mm.max()),
            "rot_rms_deg": float(np.sqrt(np.mean(dg ** 2))), "rot_max_deg": float(dg.max())}


def rotation_diversity(Rs: Sequence[np.ndarray]) -> Dict[str, object]:
    """Span (largest pairwise angle) and axis diversity of the gripper orientations.

    eig = normalised eigenvalues of sum_ij angle_ij * a_ij a_ij^T over all relative rotations (axis a in the gripper
    frame); eig[1] ~ 0 means all rotations are about (nearly) one axis, which leaves X unobservable."""
    n = len(Rs)
    M = np.zeros((3, 3))
    span = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rel = Rot.from_matrix(Rs[i].T @ Rs[j])
            rv = rel.as_rotvec()
            ang = float(np.linalg.norm(rv))
            span = max(span, math.degrees(ang))
            if ang > math.radians(3.0):
                a = rv / ang
                M += ang * np.outer(a, a)
    tr = np.trace(M)
    eig = np.sort(np.linalg.eigvalsh(M / tr))[::-1] if tr > 1e-12 else np.zeros(3)
    return {"span_deg": span, "axis_eig": [float(e) for e in eig]}


def _handeye_init(samples, T_cb_list, K) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    Rg = [s.T_base_link6[:3, :3] for s in samples]
    tg = [s.T_base_link6[:3, 3].reshape(3, 1) for s in samples]
    Rt = [T[:3, :3] for T in T_cb_list]
    tt = [T[:3, 3].reshape(3, 1) for T in T_cb_list]
    methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
    Xs, info = {}, {}
    for name, m in methods.items():
        try:
            Rc, tc = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
        except cv2.error:
            continue
        X = make_T(Rc, tc.ravel())
        if not np.all(np.isfinite(X)):
            continue
        Bs = board_poses_from_pnp(samples, T_cb_list, X)
        c = consistency(Bs)
        Xs[name] = X
        info[name] = dict(c, score=c["pos_rms_mm"] + 2.0 * c["rot_rms_deg"])
    return Xs, info


def _stats(samples, K, D, X, B) -> Dict[str, object]:
    per, allsq = [], []
    for s in samples:
        r = reprojection_residuals(_pack(X, B), [s], K, D).reshape(-1, 2)
        sq = np.sum(r ** 2, axis=1)
        per.append(float(np.sqrt(np.mean(sq))))
        allsq.append(sq)
    sq = np.concatenate(allsq)
    return {"rms_px": float(np.sqrt(np.mean(sq))), "max_px": float(np.sqrt(sq.max())), "per_sample_rms_px": per}


def solve_handeye(samples: Sequence[Sample], K: np.ndarray, D: np.ndarray, loss="huber", f_scale=1.0, do_loo=True,
                  T_base_board_scene: Optional[np.ndarray] = None, X_hint: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Initialise with the three OpenCV methods, refine jointly over X and T_base_board on the reprojection error."""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    D = np.asarray(D, dtype=float).ravel()
    if len(samples) < 3:
        raise ValueError("need at least 3 samples, got %d" % len(samples))
    T_cb_list, pnp_rms = [], []
    for s in samples:
        T, rms = solve_pnp(s.obj, s.pix, K, D)
        T_cb_list.append(T)
        pnp_rms.append(rms)
    Xs, init_info = _handeye_init(samples, T_cb_list, K)
    if not Xs and X_hint is None:
        raise RuntimeError("all cv2.calibrateHandEye methods failed")
    if Xs:
        best = min(init_info, key=lambda k: init_info[k]["score"])
        X0 = Xs[best]
    else:
        best, X0 = "hint", X_hint
    B0 = mean_pose(board_poses_from_pnp(samples, T_cb_list, X0))
    X, B, sol = refine(samples, K, D, X0, B0, loss, f_scale)
    # The joint reprojection refinement can absorb the ARM's positioning error (a few mm on the ReBeL) into a
    # badly rotated X (seen on the real cell 2026-10-13: +35 mm / +48 deg for -1 px). The closed-form AX=XB
    # initialisation uses only relative motions and is the robust answer; keep it when the refinement wanders.
    refine_rejected = None
    d_mm, d_deg = pose_diff(X0, X)
    if d_mm > REFINE_MAX_MM or d_deg > REFINE_MAX_DEG:
        refine_rejected = {"mm": d_mm, "deg": d_deg}
        X, B = X0, B0
    st = _stats(samples, K, D, X, B)
    cons_each = board_poses_from_pnp(samples, T_cb_list, X)
    cons = consistency(cons_each, B)
    div = rotation_diversity([s.T_base_link6[:3, :3] for s in samples])

    out: Dict[str, object] = {
        "X": X, "T_base_board": B, "n_samples": len(samples), "init_method": best, "init": init_info,
        "refine_cost": float(sol.cost), "pnp_rms_px": pnp_rms,
        "rms_px": st["rms_px"], "max_px": st["max_px"], "per_sample_rms_px": st["per_sample_rms_px"],
        "board_consistency": cons, "rotation": div, "warnings": [],
    }
    out["refine_delta_from_init"] = {"mm": d_mm, "deg": d_deg}
    out["refine_rejected"] = refine_rejected
    if refine_rejected is not None:
        out["warnings"].append("joint refinement rejected (it moved X by %.1f mm / %.1f deg from the AX=XB init, "
                               "absorbing arm positioning error); X = %s init" % (d_mm, d_deg, best))
    if len(Xs) >= 2:
        names = list(Xs)
        agree = [pose_diff(Xs[a], Xs[b]) for i, a in enumerate(names) for b in names[i + 1:]]
        out["method_agreement"] = {"max_mm": max(a for a, _ in agree), "max_deg": max(b for _, b in agree)}

    loo = None
    if do_loo and len(samples) >= 4:
        errs_mm, errs_deg, errs_px = [], [], []
        for i in range(len(samples)):
            rest = [s for j, s in enumerate(samples) if j != i]
            if refine_rejected is not None and best in Xs:
                rest_cb = [t for j, t in enumerate(T_cb_list) if j != i]
                Xr, _ = _handeye_init(rest, rest_cb, K)
                if best not in Xr:
                    continue
                Xl = Xr[best]
                Bl = mean_pose(board_poses_from_pnp(rest, rest_cb, Xl))
            else:
                Xl, Bl, _ = refine(rest, K, D, X, B, loss, f_scale, max_nfev=30)
            Bi = samples[i].T_base_link6 @ Xl @ T_cb_list[i]
            mm, dg = pose_diff(Bi, Bl)
            errs_mm.append(mm)
            errs_deg.append(dg)
            errs_px.append(_stats([samples[i]], K, D, Xl, Bl)["rms_px"])
        loo = {"pos_rms_mm": float(np.sqrt(np.mean(np.square(errs_mm)))), "pos_max_mm": float(np.max(errs_mm)),
               "rot_rms_deg": float(np.sqrt(np.mean(np.square(errs_deg)))), "rot_max_deg": float(np.max(errs_deg)),
               "reproj_rms_px": float(np.sqrt(np.mean(np.square(errs_px)))), "per_sample_mm": [float(v) for v in errs_mm]}
    out["loo"] = loo

    if T_base_board_scene is not None:
        mm, dg = pose_diff(B, T_base_board_scene)
        out["scene_cross_check"] = {"mm": mm, "deg": dg}
    out.update(gate(out))
    return out


def gate(r: Dict[str, object], rms_pass=1.0, loo_pass_mm=6.0, min_samples=10, span_pass=30.0) -> Dict[str, object]:
    """PASS / WARN / FAIL decision and the list of reasons."""
    reasons, status = [], "PASS"

    def worse(s):
        nonlocal status
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[s] > order[status]:
            status = s

    n = r["n_samples"]
    if n < min_samples:
        reasons.append("only %d samples (< %d)" % (n, min_samples))
        worse("FAIL" if n < 6 else "WARN")
    if r["rms_px"] > rms_pass:
        # with a few-mm arm accuracy the absolute reprojection cannot reach 1 px; informational unless huge
        reasons.append("reprojection RMS %.2f px > %.1f (arm accuracy; informational)" % (r["rms_px"], rms_pass))
        if r["rms_px"] > 12.0:
            worse("FAIL")
    cons = r.get("board_consistency")
    if cons is not None and (cons["pos_rms_mm"] > 6.0 or cons["rot_rms_deg"] > 1.0):
        reasons.append("board consistency %.2f mm / %.2f deg > 6 mm / 1 deg" % (cons["pos_rms_mm"], cons["rot_rms_deg"]))
        worse("FAIL" if cons["rot_rms_deg"] > 3.0 or cons["pos_rms_mm"] > 10.0 else "WARN")
    ma = r.get("method_agreement")
    if ma is not None and (ma["max_mm"] > 5.0 or ma["max_deg"] > 2.0):
        reasons.append("init methods disagree by %.1f mm / %.1f deg" % (ma["max_mm"], ma["max_deg"]))
        worse("WARN")
    sc = r.get("scene_cross_check")
    if sc is not None and (sc["mm"] > 10.0 or sc["deg"] > 5.0):
        reasons.append("board vs scene camera %.1f mm / %.1f deg > 10 mm / 5 deg" % (sc["mm"], sc["deg"]))
        worse("FAIL" if sc["mm"] > 25.0 or sc["deg"] > 15.0 else "WARN")
    loo = r.get("loo")
    if loo is None:
        reasons.append("leave-one-out not computed")
        worse("WARN")
    elif loo["pos_rms_mm"] > loo_pass_mm:
        reasons.append("LOO board error %.2f mm > %.1f" % (loo["pos_rms_mm"], loo_pass_mm))
        worse("FAIL" if loo["pos_rms_mm"] > 2 * loo_pass_mm else "WARN")
    div = r["rotation"]
    if div["span_deg"] < span_pass:
        reasons.append("rotation span %.1f deg < %.0f" % (div["span_deg"], span_pass))
        worse("FAIL" if div["span_deg"] < 10.0 else "WARN")
    if div["axis_eig"][1] < 0.05:
        reasons.append("DEGENERATE: link6 rotations are nearly about a single axis (axis eigenvalues %s)"
                       % ", ".join("%.3f" % e for e in div["axis_eig"]))
        worse("WARN")
    r["warnings"] = list(r.get("warnings", [])) + [x for x in reasons if x.startswith("DEGENERATE")]
    return {"status": status, "reasons": reasons}
