"""Synthetic eye-in-hand data (tests and the mock wrist camera).  No ROS."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .eye_in_hand import WristSample, project
from .targets import TargetSpec, default_targets_path, load_targets_all
from .transforms import inv_T, make_T, rotvec_to_R

WRIST_K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
WRIST_DIST = np.array([0.02, -0.05, 0.0005, -0.0003, 0.01])
WRIST_SIZE = (1280, 800)


def table_spec() -> TargetSpec:
    spec = load_targets_all(default_targets_path())[1]
    if spec is None:
        raise RuntimeError("no table board in the default targets file")
    return spec


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    """Optical-frame pose (z forward, x right, y down) at ``eye`` looking at ``target``."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up_hint)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    R = R @ rotvec_to_R([0, 0, roll_rad])
    return make_T(R, eye)


def board_all_points(spec: TargetSpec) -> Tuple[np.ndarray, np.ndarray]:
    pts = spec.all_points_obj()
    skip = spec.touch_corner_ids()
    ids = np.array([i for i in sorted(pts) if i not in skip])
    return ids, np.array([pts[i] for i in ids])


def make_samples(X: np.ndarray, B: np.ndarray, poses_cam_in_base: Sequence[np.ndarray], spec: Optional[TargetSpec] = None,
                 noise_px: float = 0.3, seed: int = 0, K=WRIST_K, dist=WRIST_DIST, size=WRIST_SIZE,
                 link6_noise_deg: float = 0.0, link6_noise_mm: float = 0.0) -> List[WristSample]:
    """Samples for camera poses (optical frame in base): A_i = T_cam_i X^-1 (so that A_i X = T_cam_i)."""
    rng = np.random.default_rng(seed)
    spec = spec or table_spec()
    ids, obj = board_all_points(spec)
    out = []
    for i, T_cam in enumerate(poses_cam_in_base):
        A = T_cam @ inv_T(X)
        T_ob = inv_T(T_cam) @ B
        uv = project(T_ob, obj, K, dist)
        inside = ((uv[:, 0] > 5) & (uv[:, 0] < size[0] - 5) & (uv[:, 1] > 5) & (uv[:, 1] < size[1] - 5))
        if inside.sum() < 16:
            continue
        pix = uv[inside] + rng.normal(0, noise_px, (int(inside.sum()), 2))
        A_meas = A
        if link6_noise_deg or link6_noise_mm:     # the ARM's absolute positioning error: the measured pose is off
            A_meas = A @ make_T(rotvec_to_R(rng.normal(0, np.radians(link6_noise_deg), 3)),
                                rng.normal(0, link6_noise_mm * 1e-3 / math.sqrt(3.0), 3))
        out.append(WristSample(index=len(out), stamp=1000.0 + i, T_base_link6=A_meas, ids=ids[inside], obj=obj[inside],
                               pix=pix, K=K, dist=dist, image_size=size))
    return out


def orbit_poses(B: np.ndarray, n: int = 18, seed: int = 0, dist_m: Tuple[float, float] = (0.30, 0.45),
                tilt_deg: float = 30.0, roll_deg: float = 40.0) -> List[np.ndarray]:
    """Camera poses (optical in base) around the board, looking at its centre, diverse tilt azimuths and rolls."""
    rng = np.random.default_rng(seed)
    c = B[:3, 3]
    n_out = -B[:3, 2]
    poses = []
    for i in range(n):
        az = 2 * np.pi * i / n + rng.normal(0, 0.1)
        tilt = np.radians(rng.uniform(0.3, 1.0) * tilt_deg)
        d = rng.uniform(*dist_m)
        # direction from the board centre to the camera: tilted away from the normal
        bx = B[:3, 0]
        by = B[:3, 1]
        u = np.cos(tilt) * n_out + np.sin(tilt) * (np.cos(az) * bx + np.sin(az) * by)
        eye = c + d * u
        poses.append(look_at(eye, c, by, roll_rad=np.radians(rng.uniform(-roll_deg, roll_deg))))
    return poses
