"""Shared geometry helpers for the dual-arm cell planner check (pure numpy, no ROS).

Planning frame "world" == W of the offline studies (Z-up):
  x_w = forward (CAD +Z), y_w = CAD +X (arm1 at +y), z_w = up (CAD +Y).
  Origin: x_w = 0 at CAD Z = ARM_Z (arm base axis), y_w = 0 at CAD X = MID_X (mid point between the arms),
  z_w = 0 at the arm base mounting plane (carton top at z_w = -h).
Each arm base_link frame is W translated by (0, +-s/2, 0) with identity rotation (this is exactly the CAD rotation
R = [[0,1,0],[0,0,1],[1,0,0]] of layout.md).  arm1 == study arm 0 (y_w = +s/2), arm2 == study arm 1 (y_w = -s/2).
"""
import math
import numpy as np

MID_X = -0.653868
ARM_Z = 0.854511
ARM_Y_CAD = -0.241696          # CAD height of the arm base mounting plane (layout.json), at the original lift position
CAD_SPACING = 0.459944
REPO = '/home/farhan/igus_rebel_ros2'
ARMS = ('arm1', 'arm2')
ARM_INDEX = {'arm1': 0, 'arm2': 1}


def arm_y(i, s):
    """y_w of the base of study arm i (0 -> arm1 at +s/2, 1 -> arm2 at -s/2)"""
    return s / 2.0 if i == 0 else -s / 2.0


def cad_to_world(p_cad):
    p = np.asarray(p_cad, float)
    return np.array([p[..., 2] - ARM_Z, p[..., 0] - MID_X, p[..., 1] - ARM_Y_CAD]).T if p.ndim > 1 else \
        np.array([p[2] - ARM_Z, p[0] - MID_X, p[1] - ARM_Y_CAD])


def world_to_cad(p_w):
    p = np.asarray(p_w, float)
    return np.array([p[1] + MID_X, p[2] + ARM_Y_CAD, p[0] + ARM_Z])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_to_rpy(R):
    """ROS URDF rpy (R = Rz(y) Ry(p) Rx(r))"""
    sp = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sp)))
    if abs(abs(sp) - 1.0) < 1e-9:
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
        if sp < 0:
            y = math.atan2(-R[0, 1], R[1, 1])
    else:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    return r, p, y


def rpy_to_rot(r, p, y):
    return Rz(y) @ Ry(p) @ Rx(r)


def R_to_quat(R):
    """xyzw"""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def T_from(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def pose_to_T(pose):
    """pose dict {xyz:[3], quat:[4 xyzw]} -> 4x4"""
    return T_from(quat_to_R(pose['quat']), pose['xyz'])


def T_to_pose(T):
    return {'xyz': [float(v) for v in T[:3, 3]], 'quat': [float(v) for v in R_to_quat(T[:3, :3])]}


def rot_angle_deg(R):
    """rotation angle of R (numerically stable for small angles, tolerant of 1e-6 non-orthonormality of rounded matrices)"""
    R = np.asarray(R, float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return math.degrees(math.atan2(np.linalg.norm(v), (np.trace(R) - 1) / 2))


def z_to_vec_rpy(v):
    """rpy of a frame whose +z axis points along vector v (any roll)"""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    z = v / n
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return rot_to_rpy(R), n


def carton_geometry(box, h, d, yaw_deg, dx=0.0, dy=0.0, boxes=None):
    """box centre/orientation/size in W, same convention as the study (dalib.task_targets / box_sdf):
    local x along the seam a = (-sin psi, cos psi), A along a, B across, H tall, top at z = -h."""
    boxes = boxes or {'BIG': (0.320, 0.390, 0.180), 'SMALL': (0.106, 0.112, 0.0863),
                      'BIG390': (0.320, 0.390, 0.180), 'BIG370': (0.320, 0.370, 0.180)}
    A, B, H = boxes[box]
    psi = math.radians(yaw_deg)
    phi = psi + math.pi / 2
    return dict(size=[A, B, H], center=[d + dx, dy, -h - H / 2], quat=[float(v) for v in R_to_quat(Rz(phi))], yaw_deg=yaw_deg)
