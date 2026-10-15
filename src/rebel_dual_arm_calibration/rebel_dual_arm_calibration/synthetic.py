"""Synthetic dual-arm scene for tests and the accuracy study (no ROS, no images).

World = arm_left_base_link (z up, table plane z = 0).  The left base is at the origin,
the right base 0.4474 m along +x with a small yaw.  The workspace is in front of the
bases (+y).  The camera looks at the workspace centre from the front (+y side, looking
toward -y) at a configurable distance and 50 deg depression.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .kinematics import ArmKinematics, Urdf
from .solver import Sample, TableObservation, Touch, project_points
from .targets import CHESS_ID_BASE, TargetSpec
from .transforms import inv_T, make_T, rpy_to_R, rotvec_to_R, xyz_rpy_to_T

WORKSPACE_CENTER = np.array([0.2237, 0.35, 0.15])
BASE_SEPARATION = 0.4474
# camera_link (x fwd, y left, z up) -> optical (z fwd, x right, y down), plus a driver offset
T_CAMLINK_OPTICAL = make_T(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float).T.T, [0.002, -0.0175, 0.0])


def camera_matrix(width=1920, height=1080, hfov_deg=90.0) -> np.ndarray:
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def look_at(position, target, up=(0, 0, 1)) -> np.ndarray:
    """T_W_optical with z toward target, x right, y down."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], axis=1), position)


@dataclass
class Scene:
    T_W_O: np.ndarray
    T_W_R: np.ndarray
    Z: Dict[str, np.ndarray]
    K: np.ndarray
    dist: np.ndarray
    size: tuple
    T_cl_o: np.ndarray
    # kinematic scenes only (make_kin_scene)
    T_W_B: Optional[np.ndarray] = None            # table board in world (true)
    offsets: Optional[Dict[str, np.ndarray]] = None   # true joint zero offsets (rad): physical = encoder + offset

    @property
    def T_W_cam(self) -> np.ndarray:
        return self.T_W_O @ inv_T(self.T_cl_o)

    def T_W_base(self, arm: str) -> np.ndarray:
        return np.eye(4) if arm == "left" else self.T_W_R


def make_scene(distance: float = 0.9, elevation_deg: float = 50.0, seed: int = 0,
               hfov_deg: float = 90.0, lens_distortion: bool = True) -> Scene:
    rng = np.random.default_rng(seed)
    el = np.radians(elevation_deg)
    pos = WORKSPACE_CENTER + distance * np.array([0.0, np.cos(el), np.sin(el)])
    pos = pos + rng.normal(0, 0.01, 3)
    T_W_O = look_at(pos, WORKSPACE_CENTER + rng.normal(0, 0.02, 3))
    T_W_R = xyz_rpy_to_T([BASE_SEPARATION + 0.003, 0.002, -0.0015], [0.0008, -0.0005, np.radians(0.5)])
    Z = {"left": xyz_rpy_to_T([0.01, 0.035, 0.06], [1.2, -0.4, 0.3]),
         "right": xyz_rpy_to_T([-0.008, 0.03, 0.07], [-0.5, 0.9, 2.2])}
    dist = np.array([-0.04, 0.01, 0.0005, -0.0003, 0.0]) if lens_distortion else np.zeros(5)
    return Scene(T_W_O, T_W_R, Z, camera_matrix(hfov_deg=hfov_deg), dist, (1920, 1080), T_CAMLINK_OPTICAL.copy())


def _random_unit_in_cone(rng, axis, max_deg):
    """Random unit vector within max_deg of axis."""
    ang = np.radians(max_deg) * np.sqrt(rng.uniform(0, 1))
    phi = rng.uniform(0, 2 * np.pi)
    a = axis / np.linalg.norm(axis)
    t = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    t /= np.linalg.norm(t)
    b = np.cross(a, t)
    return np.cos(ang) * a + np.sin(ang) * (np.cos(phi) * t + np.sin(phi) * b)


def simulate_samples(scene: Scene, specs: Dict[str, TargetSpec], n_per_arm: int = 20, noise_px: float = 0.3,
                     seed: int = 1, max_view_deg: float = 55.0, min_view_deg: float = 0.0,
                     rotation_diversity: bool = True, n_sample_outliers: int = 0, n_corner_outliers: int = 0,
                     image_margin: float = 8.0, pose_jitter_m: float = 0.0,
                     marker_noise_factor: float = 1.0, min_marker_px: float = 0.0,
                     stats: Optional[dict] = None) -> List[Sample]:
    """Generate samples with known ground truth.

    Each target is placed in the arm's workspace with its outward normal within
    ``max_view_deg`` of the direction to the camera and a random roll about the
    normal; link6 follows from the true target offset.  With
    ``rotation_diversity=False`` all link6 rotations are identical (degenerate set).
    """
    rng = np.random.default_rng(seed)
    samples: List[Sample] = []
    cam_pos = scene.T_W_O[:3, 3]
    if stats is not None:
        stats.update(valid=0, usable=0)
    for arm in ("left", "right"):
        spec = specs[arm]
        base = scene.T_W_base(arm)
        x_ws = base[0, 3] + np.array([-0.12, 0.12])
        pts_obj = np.array(list(spec.all_points_obj().values()))
        ids = np.array(list(spec.all_points_obj().keys()))
        fixed_R = None
        n = 0
        tries = 0
        while n < n_per_arm and tries < 5000:
            tries += 1
            centre = np.array([rng.uniform(*x_ws), rng.uniform(0.22, 0.48), rng.uniform(0.08, 0.30)])
            to_cam = (cam_pos - centre)
            normal = _random_unit_in_cone(rng, to_cam, max_view_deg)  # outward normal = -z_board
            if np.degrees(np.arccos(np.clip(normal @ to_cam / np.linalg.norm(to_cam), -1, 1))) < min_view_deg:
                continue
            zb = -normal
            xb = np.cross([0, 0, 1.0], zb)
            if np.linalg.norm(xb) < 1e-3:
                continue
            xb /= np.linalg.norm(xb)
            yb = np.cross(zb, xb)
            roll = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(roll), np.sin(roll)
            Rt = np.stack([c * xb + s * yb, -s * xb + c * yb, zb], axis=1)
            T_W_t = make_T(Rt, centre)
            T_W_l6 = T_W_t @ inv_T(scene.Z[arm])
            if not rotation_diversity:
                if fixed_R is None:
                    fixed_R = T_W_l6[:3, :3].copy()
                T_W_l6 = make_T(fixed_R, T_W_l6[:3, 3])
                T_W_t = T_W_l6 @ scene.Z[arm]
            T_O_t = inv_T(scene.T_W_O) @ T_W_t
            P = pts_obj @ T_O_t[:3, :3].T + T_O_t[:3, 3]
            if np.any(P[:, 2] < 0.1):
                continue
            uv = project_points(scene.K, scene.dist, P)
            if (uv[:, 0].min() < image_margin or uv[:, 1].min() < image_margin
                    or uv[:, 0].max() > scene.size[0] - image_margin or uv[:, 1].max() > scene.size[1] - image_margin):
                continue
            keep = np.ones(len(ids), bool)
            if stats is not None:
                stats["valid"] += 1
            if min_marker_px > 0:
                alive = 0
                for k in range(spec.n_markers()):
                    sel = [list(ids).index(4 * k + c) for c in range(4)]
                    m = uv[sel]
                    if np.linalg.norm(m - np.roll(m, -1, axis=0), axis=1).min() >= min_marker_px:
                        alive += 1
                    else:
                        keep[sel] = False
                if alive < (2 if spec.type == "charuco" else 1):
                    continue
            if stats is not None:
                stats["usable"] += 1
            T_b_l6 = inv_T(base) @ T_W_l6
            if pose_jitter_m:
                T_b_l6 = T_b_l6.copy()
                T_b_l6[:3, 3] += rng.normal(0, pose_jitter_m, 3)
            sig = np.where(ids >= 1000, 1.0, marker_noise_factor)[:, None] * noise_px
            uv = uv + rng.normal(0, 1.0, uv.shape) * sig
            samples.append(Sample(arm=arm, T_base_link6=T_b_l6, ids=ids[keep], obj=pts_obj[keep], pix=uv[keep],
                                  K=scene.K,
                                  dist=scene.dist, image_size=scene.size, T_camlink_optical=scene.T_cl_o,
                                  joints={f"{arm}_j{k}": float(v) for k, v in enumerate(rng.uniform(-1, 1, 6))},
                                  image=f"synthetic_{arm}_{n:03d}"))
            n += 1
        if n < n_per_arm:
            raise RuntimeError(f"could not place {n_per_arm} visible targets for {arm}")
    # gross outliers
    idx = rng.permutation(len(samples))
    for k in idx[:n_sample_outliers]:
        samples[k].pix = samples[k].pix + rng.uniform(15, 40, 2) * rng.choice([-1, 1], 2)
    for k in idx[n_sample_outliers:n_sample_outliers + n_corner_outliers]:
        j = rng.integers(len(samples[k].pix))
        samples[k].pix = samples[k].pix.copy()
        samples[k].pix[j] += rng.uniform(20, 40, 2) * rng.choice([-1, 1], 2)
    return samples


def marker_side_px(specs: Dict[str, TargetSpec], samples: List[Sample]) -> float:
    """Smallest projected marker side (px) over all markers of all samples (noisy pixels)."""
    best = np.inf
    for s in samples:
        n = specs[s.arm].n_markers()
        lut = {int(i): k for k, i in enumerate(s.ids)}
        for k in range(n):
            if 4 * k not in lut:
                continue  # marker treated as undetected
            m = np.array([s.pix[lut[4 * k + c]] for c in range(4)])
            best = min(best, float(np.linalg.norm(m - np.roll(m, -1, axis=0), axis=1).min()))
    return best


# --------------------------------------------------------------------------
# Kinematic scenes: joint-space poses from the real ReBeL URDF, joint offsets, table board, touches
# --------------------------------------------------------------------------
_TOOL_TAIL = """
  <link name="igus_tool0"/>
  <joint name="link6_to_igus_tool0" type="fixed">
    <parent link="link6"/><child link="igus_tool0"/>
    <origin xyz="0 0 0.0527" rpy="0 0 3.141592653589793"/>
  </joint>
  <link name="xeg32_tool_tip"/>
  <joint name="igus_tool0_to_xeg32_tool_tip" type="fixed">
    <parent link="igus_tool0"/><child link="xeg32_tool_tip"/>
    <origin xyz="0.005937 0.094121 0.130699" rpy="0 0 0"/>
  </joint>
"""


def rebel_urdf_path() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(os.path.dirname(os.path.dirname(here)), "igus_rebel_description", "urdf", "igus_rebel2.urdf")
    if os.path.exists(cand):
        return cand
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("igus_rebel_description"), "urdf", "igus_rebel2.urdf")
        return p if os.path.exists(p) else None
    except Exception:  # noqa: BLE001
        return None


def rebel_urdf_text(with_tool: bool = True, prefix: str = "") -> str:
    """The real igus_rebel2.urdf (+ the xeg32_tool_tip tail of the wrapper); optional name prefix."""
    path = rebel_urdf_path()
    if path is None:
        raise FileNotFoundError("igus_rebel2.urdf not found")
    text = open(path, encoding="utf-8").read()
    if with_tool:
        text = text.replace("</robot>", _TOOL_TAIL + "</robot>")
    if prefix:
        import re
        text = re.sub(r'(<(?:link|joint)\s+name=")', lambda m: m.group(1) + prefix, text)
        text = re.sub(r'(<(?:parent|child)\s+link=")', lambda m: m.group(1) + prefix, text)
    return text


def dual_urdf_text(prefixes: Tuple[str, str] = ("arm_left_", "arm_right_")) -> str:
    """Both arms (with tool tip) in one URDF, as the dual-arm bring-up publishes them."""
    bodies = []
    for pre in prefixes:
        t = rebel_urdf_text(True, pre)
        bodies.append(t[t.index("<robot"):].split(">", 1)[1].rsplit("</robot>", 1)[0])
    return '<?xml version="1.0"?>\n<robot name="dual">' + "".join(bodies) + "</robot>\n"


def default_kinematics(prefixes: Tuple[str, str] = ("", "")) -> Dict[str, ArmKinematics]:
    """{arm: ArmKinematics} from the real arm URDF (both arms are the same robot)."""
    out = {}
    for arm, pre in zip(("left", "right"), prefixes):
        u = Urdf.parse(rebel_urdf_text(True, pre))
        out[arm] = ArmKinematics.from_urdf(u, pre + "base_link", pre + "link6", pre + "xeg32_tool_tip")
    return out


def _joint_limits(chain) -> Tuple[np.ndarray, np.ndarray]:
    path = rebel_urdf_path()
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    lim = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    def get(n):   # prefixed (dual-arm) names: arm_left_joint1 -> joint1
        return lim[n] if n in lim else lim[n[n.rindex("joint"):]]
    lo = np.array([float(get(n).get("lower")) for n in chain.joint_names])
    hi = np.array([float(get(n).get("upper")) for n in chain.joint_names])
    return lo, hi


# board outward normal along link6 +x, board x along link6 y (so the print faces away from the arm)
Z_KIN = {"left": xyz_rpy_to_T([0.050, 0.020, 0.060], [0.0, 0.0, 0.0]),
         "right": xyz_rpy_to_T([0.052, -0.015, 0.065], [0.0, 0.0, 0.0])}
_R_FACE = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # columns: x_b, y_b, z_b in link6


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), [0, 0, 0])


def make_kin_scene(distance: float = 0.9, seed: int = 0, offset_deg: float = 0.0, table: bool = True,
                   lens_distortion: bool = True) -> Scene:
    """make_scene + per-arm joint zero offsets (random sign, ``offset_deg`` .. 1.4x per joint) + a table board.

    The table board lies flat in front of both arms (z into the table), centred between them."""
    sc = make_scene(distance=distance, seed=seed, lens_distortion=lens_distortion)
    rng = np.random.default_rng(1000 + seed)
    Zs = {}
    for arm in ("left", "right"):
        Zs[arm] = Z_KIN[arm].copy()
        Zs[arm][:3, :3] = rotvec_to_R(rng.normal(0, 0.03, 3)) @ _R_FACE
    sc.Z = Zs
    sc.offsets = {a: np.radians(offset_deg) * rng.uniform(1.0, 1.4, 6) * rng.choice([-1, 1], 6)
                  for a in ("left", "right")} if offset_deg else {a: np.zeros(6) for a in ("left", "right")}
    if table:
        # board frame: x world +x, y world -y, z world -z (looking down on it, y "down" toward the arms)
        R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        sc.T_W_B = make_T(rotvec_to_R([0.01, -0.008, 0.02]) @ R, [BASE_SEPARATION / 2 + 0.002, 0.30, 0.0])
    return sc


def effective_truth(scene: Scene) -> Dict[str, np.ndarray]:
    """Truth in the gauge the solver can identify: joint-1 zero offsets are not separable from the base yaw,
    so the world frame is the left base *as its nominal FK sees it* (rotated by the left joint-1 offset)."""
    d1 = {a: float(scene.offsets[a][0]) if scene.offsets else 0.0 for a in ("left", "right")}
    G = Rz(-d1["left"])
    out = {"T_W_O": G @ scene.T_W_O, "T_W_R": G @ scene.T_W_R @ Rz(d1["right"])}
    out["T_W_cam"] = out["T_W_O"] @ inv_T(scene.T_cl_o)
    if scene.T_W_B is not None:
        out["T_W_B"] = G @ scene.T_W_B
    return out


def simulate_kin_samples(scene: Scene, specs: Dict[str, TargetSpec], kin: Dict[str, ArmKinematics],
                         n_per_arm: int = 30, noise_px: float = 0.3, seed: int = 1, max_view_deg: float = 65.0,
                         marker_noise_factor: float = 1.0, image_margin: float = 8.0, record_joints: bool = True,
                         ) -> List[Sample]:
    """Samples whose link6 poses come from joint angles through the real URDF.

    The camera sees the PHYSICAL arm (encoder + true offset); the recorded T_base_link6 and joints are the
    ENCODER values (what TF / joint_states report)."""
    rng = np.random.default_rng(seed)
    samples: List[Sample] = []
    cam_pos = scene.T_W_O[:3, 3]
    for arm in ("left", "right"):
        spec, ch = specs[arm], kin[arm].chain6
        lo, hi = _joint_limits(ch)
        lo, hi = np.maximum(lo, -2.8), np.minimum(hi, 2.8)
        base = scene.T_W_base(arm)
        pts_obj = np.array(list(spec.all_points_obj().values()))
        ids = np.array(list(spec.all_points_obj().keys()))
        off = scene.offsets[arm] if scene.offsets else np.zeros(6)
        n = 0
        for _ in range(60):
            q = rng.uniform(lo, hi, (4000, 6))
            T_b = ch.fk(q + off)
            T_W_t = base @ T_b @ scene.Z[arm]
            c = T_W_t[:, :3, 3]
            ok = ((np.abs(c[:, 0] - base[0, 3]) < 0.16) & (c[:, 1] > 0.20) & (c[:, 1] < 0.50)
                  & (c[:, 2] > 0.06) & (c[:, 2] < 0.32))
            to_cam = cam_pos - c
            to_cam /= np.linalg.norm(to_cam, axis=1)[:, None]
            normal = -T_W_t[:, :3, 2]
            ok &= (normal * to_cam).sum(1) > np.cos(np.radians(max_view_deg))
            for k in np.flatnonzero(ok):
                T_O_t = inv_T(scene.T_W_O) @ T_W_t[k]
                P = pts_obj @ T_O_t[:3, :3].T + T_O_t[:3, 3]
                if np.any(P[:, 2] < 0.1):
                    continue
                uv = project_points(scene.K, scene.dist, P)
                if (uv[:, 0].min() < image_margin or uv[:, 1].min() < image_margin
                        or uv[:, 0].max() > scene.size[0] - image_margin
                        or uv[:, 1].max() > scene.size[1] - image_margin):
                    continue
                sig = np.where(ids >= CHESS_ID_BASE, 1.0, marker_noise_factor)[:, None] * noise_px
                uv = uv + rng.normal(0, 1.0, uv.shape) * sig
                T_rec = ch.fk(q[k])
                names = ch.joint_names
                samples.append(Sample(arm=arm, T_base_link6=T_rec, ids=ids, obj=pts_obj, pix=uv, K=scene.K,
                                      dist=scene.dist, image_size=scene.size, T_camlink_optical=scene.T_cl_o,
                                      joints={nm: float(v) for nm, v in zip(names, q[k])} if record_joints else {},
                                      q=q[k].copy(), image=f"synthetic_{arm}_{n:03d}"))
                n += 1
                if n >= n_per_arm:
                    break
            if n >= n_per_arm:
                break
        if n < n_per_arm:
            raise RuntimeError(f"could not place {n_per_arm} visible targets for {arm}")
    return samples


def _scale_vec(board_scale) -> np.ndarray:
    """``board_scale``: a float (isotropic) or (sx, sy) -> the 3-vector that multiplies board coordinates."""
    b = np.atleast_1d(np.asarray(board_scale, float))
    return np.array([b[0], b[-1] if len(b) > 1 else b[0], 1.0])


def simulate_table_obs(scene: Scene, table_spec: TargetSpec, n: int = 10, noise_px: float = 0.3,
                       seed: int = 2, marker_noise_factor: float = 1.0, board_scale=1.0
                       ) -> List[TableObservation]:
    """Table-board corner observations (the camera is static; only the pixel noise differs per image).

    ``board_scale`` (float or (sx, sy)) scales the PHYSICAL board (print scale error); the solver keeps the nominal
    geometry."""
    rng = np.random.default_rng(seed)
    skip = table_spec.touch_corner_ids()
    pts = {k: v for k, v in table_spec.all_points_obj().items() if k not in skip}
    ids = np.array(list(pts))
    obj = np.array(list(pts.values()))
    T_O_B = inv_T(scene.T_W_O) @ scene.T_W_B
    P = (obj * _scale_vec(board_scale)) @ T_O_B[:3, :3].T + T_O_B[:3, 3]
    uv0 = project_points(scene.K, scene.dist, P)
    out = []
    for i in range(n):
        sig = np.where(ids >= CHESS_ID_BASE, 1.0, marker_noise_factor)[:, None] * noise_px
        out.append(TableObservation(ids=ids, obj=obj, pix=uv0 + rng.normal(0, 1.0, uv0.shape) * sig, K=scene.K,
                                    dist=scene.dist, image_size=scene.size, T_camlink_optical=scene.T_cl_o,
                                    image=f"synthetic_table_{i:03d}"))
    return out


def _ik_position(chain, p_base: np.ndarray, lo, hi, rng, tries: int = 60) -> Optional[np.ndarray]:
    """Joint values putting the chain tip at p_base, flange axis pointing roughly down."""
    def fun(q):
        T = chain.fk(q)
        return np.concatenate([T[:3, 3] - p_base, [0.05 * (1.0 + T[2, 2])]])
    for _ in range(tries):
        q0 = rng.uniform(lo, hi)
        sol = least_squares(fun, q0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        T = chain.fk(sol.x)
        if np.linalg.norm(T[:3, 3] - p_base) < 1e-7 and T[2, 2] < -0.3:
            return sol.x
    return None


def simulate_touches(scene: Scene, table_spec: TargetSpec, kin: Dict[str, ArmKinematics], repeats: int = 1,
                     noise_m: float = 3e-4, seed: int = 3, board_scale=1.0,
                     right_base_error: Optional[np.ndarray] = None, arms=("left", "right")) -> List[Touch]:
    """Both arms touch every touch point ``repeats`` times at different joint configurations.

    The tip really is on the board (physical joints = encoder + offset); records carry the encoder-based TF pose.
    ``right_base_error`` (4x4) displaces the right base ONLY for the touches (arm moved between sessions)."""
    rng = np.random.default_rng(seed)
    out: List[Touch] = []
    for arm in arms:
        ch = kin[arm].chain_tcp
        lo, hi = _joint_limits(ch)
        lo, hi = np.maximum(lo, -2.5), np.minimum(hi, 2.5)
        base = scene.T_W_base(arm)
        if arm == "right" and right_base_error is not None:
            base = base @ right_base_error
        off = scene.offsets[arm] if scene.offsets else np.zeros(6)
        for pid, (x, y) in sorted(table_spec.touch_points.items()):
            p_true = np.array([x, y, 0.0]) * _scale_vec(board_scale)
            p_w = scene.T_W_B[:3, :3] @ p_true + scene.T_W_B[:3, 3]
            p_b = inv_T(base)[:3, :3] @ p_w + inv_T(base)[:3, 3]
            for _ in range(repeats):
                q_phys = _ik_position(ch, p_b, lo, hi, rng)
                if q_phys is None:
                    raise RuntimeError(f"no IK solution for {arm} touch point {pid}")
                q_enc = q_phys - off
                T = ch.fk(q_enc)
                T[:3, 3] += rng.normal(0, noise_m, 3)
                out.append(Touch(arm=arm, point_id=pid, p_board=np.array([x, y, 0.0]), T_base_tcp=T,
                                 joints={nm: float(v) for nm, v in zip(ch.joint_names, q_enc)},
                                 tcp_frame=kin[arm].tcp_frame, q=q_enc))
    return out


def default_sticker_positions(scene: Scene, spread: float = 0.14) -> Dict[str, Dict[int, np.ndarray]]:
    """World positions of the free stickers around the table board of ``scene``: four fit stickers at +-``spread``
    (m) about the board centre on the table plane, a check sticker on a 40 mm block and one ~190 mm from the centre."""
    c = scene.T_W_B[:3, 3]
    fit = {101 + k: c + np.array([sx * spread, sy * spread, 0.0])
           for k, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)])}
    check = {111: c + np.array([0.14, -0.04, 0.040]), 112: c + np.array([-0.19, 0.04, 0.0])}
    return {"fit": fit, "check": check}


def simulate_sticker_touches(scene: Scene, kin: Dict[str, ArmKinematics], positions: Dict[str, Dict[int, np.ndarray]],
                             repeats: int = 1, noise_m: float = 3e-4, seed: int = 4,
                             arms=("left", "right"), displace: Optional[Dict[int, np.ndarray]] = None,
                             displace_arm: Optional[str] = None) -> List[Touch]:
    """Both arms touch free stickers (kind fit / check) at their world positions, ``repeats`` times each at different
    joint configurations (physical joints = encoder + zero offset).  ``displace`` {id: xyz (m)} moves the touched
    point of ``displace_arm`` only (an arm that touches the wrong place: negative test)."""
    rng = np.random.default_rng(seed)
    out: List[Touch] = []
    for arm in arms:
        ch = kin[arm].chain_tcp
        lo, hi = _joint_limits(ch)
        lo, hi = np.maximum(lo, -2.5), np.minimum(hi, 2.5)
        base = scene.T_W_base(arm)
        off = scene.offsets[arm] if scene.offsets else np.zeros(6)
        for kind in ("fit", "check"):
            for pid, p_w in sorted(positions.get(kind, {}).items()):
                p = np.asarray(p_w, float)
                if displace and pid in displace and (displace_arm is None or displace_arm == arm):
                    p = p + np.asarray(displace[pid], float)
                p_b = inv_T(base)[:3, :3] @ p + inv_T(base)[:3, 3]
                for _ in range(repeats):
                    q_phys = _ik_position(ch, p_b, lo, hi, rng)
                    if q_phys is None:
                        raise RuntimeError(f"no IK solution for {arm} sticker {pid}")
                    q_enc = q_phys - off
                    T = ch.fk(q_enc)
                    T[:3, 3] += rng.normal(0, noise_m, 3)
                    out.append(Touch(arm=arm, point_id=pid, p_board=np.zeros(3), T_base_tcp=T,
                                     joints={nm: float(v) for nm, v in zip(ch.joint_names, q_enc)},
                                     tcp_frame=kin[arm].tcp_frame, q=q_enc, kind=kind))
    return out


def nominal_prediction_error(scene: Scene, kin: Dict[str, ArmKinematics], T_W_R_est: np.ndarray, n: int = 300,
                             seed: int = 5, radius: float = 0.35, arm: str = "right") -> Dict[str, float]:
    """What matters for collision avoidance: error of the position the robot's NOMINAL model predicts for the
    right arm's TCP in the world frame, ``T_W_R_est @ FK_nominal(q)``, against where the tip physically is
    (physical base, encoder + zero offsets), over random joint configurations whose tip lies within ``radius`` of
    the table board centre.  Both are compared in the identifiable gauge (``effective_truth``): the joint-1 zero
    offsets are absorbed in the base yaw."""
    rng = np.random.default_rng(seed)
    ch = kin[arm].chain_tcp
    lo, hi = _joint_limits(ch)
    lo, hi = np.maximum(lo, -2.5), np.minimum(hi, 2.5)
    off = scene.offsets[arm] if scene.offsets else np.zeros(6)
    d1l = float(scene.offsets["left"][0]) if scene.offsets else 0.0
    G = Rz(-d1l)
    centre = (G @ scene.T_W_B)[:3, 3]
    pts_nom, pts_true = [], []
    for _ in range(200):
        q = rng.uniform(lo, hi, (2000, 6))
        T_true = G @ scene.T_W_base(arm) @ ch.fk(q + off)
        keep = np.linalg.norm(T_true[:, :3, 3] - centre, axis=1) < radius
        pts_true.append(T_true[keep][:, :3, 3])
        pts_nom.append(ch.fk(q[keep])[:, :3, 3])
        if sum(len(p) for p in pts_true) >= n:
            break
    P_true, P_nom = np.concatenate(pts_true)[:n], np.concatenate(pts_nom)[:n]
    pred = P_nom @ T_W_R_est[:3, :3].T + T_W_R_est[:3, 3]
    e = np.linalg.norm(pred - P_true, axis=1) * 1e3
    return {"rms_mm": float(np.sqrt(np.mean(e ** 2))), "max_mm": float(e.max()), "n": int(len(e))}
