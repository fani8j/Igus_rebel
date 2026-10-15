"""Small SE(3) helpers (numpy/scipy only, no ROS)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rotvec_to_R(rv) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rv, dtype=float)).as_matrix()


def R_to_rotvec(R) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def pose_to_T(p) -> np.ndarray:
    """6-vector [rotvec(3), t(3)] -> 4x4."""
    p = np.asarray(p, dtype=float)
    return make_T(rotvec_to_R(p[:3]), p[3:6])


def T_to_pose(T) -> np.ndarray:
    return np.concatenate([R_to_rotvec(T[:3, :3]), T[:3, 3]])


def rpy_to_R(rpy) -> np.ndarray:
    """URDF fixed-axis rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def R_to_rpy(R) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_euler("xyz")


def T_to_xyz_rpy(T):
    return T[:3, 3].copy(), R_to_rpy(T[:3, :3])


def xyz_rpy_to_T(xyz, rpy) -> np.ndarray:
    return make_T(rpy_to_R(rpy), xyz)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations in degrees."""
    return float(np.degrees(np.linalg.norm(R_to_rotvec(Ra.T @ Rb))))


def quat_to_R(x, y, z, w) -> np.ndarray:
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def R_to_quat(R) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()  # x y z w


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigid T (no scale) with Q ~ T P in the least-squares sense; P, Q are (n, 3)."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, qc - R @ pc)


def kabsch_batch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Vectorised :func:`kabsch` over a leading batch axis: P, Q are (B, n, 3); returns (B, 4, 4)."""
    pc, qc = P.mean(axis=1, keepdims=True), Q.mean(axis=1, keepdims=True)
    H = np.einsum("bni,bnj->bij", P - pc, Q - qc)
    U, _, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, 1, 2)
    d = np.sign(np.linalg.det(V @ np.swapaxes(U, 1, 2)))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = V @ D @ np.swapaxes(U, 1, 2)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = (qc[:, 0, :] - np.einsum("bij,bj->bi", R, pc[:, 0, :]))
    return T
