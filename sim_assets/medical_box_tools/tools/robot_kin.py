"""Kinematics + coarse collision model of the D200 dual-arm medicine robot (URDF of rebel_medicine_grasp_description, world = d200_base_footprint) for the AMR parking study.

  * numeric FK (numpy) and KDL IK (PyKDL, joint limits) of both arms from the SAME URDF as MoveIt (expanded with xacro into scratch/medicine.urdf);
  * collision = sample points of every link's URDF collision geometry (arm links are the URDF cylinders/boxes, tools/stand are the URDF meshes) against the cabinet boxes, carton blocks and the
    other robot parts (point-to-box / point-to-point distances; no FCL: a screen, the Isaac replay and MoveIt confirm);
  * the pick sequences of tests/test_cabinet_grasp.py as Cartesian waypoint lists for the suction tool (left arm) and the gripper (right arm), front and top-down.
Frames: cabinet frame C (X across, Y into the cabinet, front at y = 0, Z up) -> world: x_w = y_c + X0, y_w = -x_c + Y0, z_w = z_c (yaw -90 deg; the AMR faces the cabinet).
"""
import json
import math
from pathlib import Path

import numpy as np
import PyKDL as kdl
import trimesh
from scipy.spatial import cKDTree
from urdf_parser_py.urdf import URDF

HERE = Path(__file__).resolve().parent
URDF_PATH = Path('/home/farhan/.cache_medicine_scratch/medicine.urdf')
PKG = {'rebel_dual_arm_stand_description': Path('/home/farhan/igus_rebel_ros2/src/rebel_dual_arm_stand_description'),
       'rebel_medicine_grasp_description': Path('/home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_description')}
AMR_FRONT_X = 0.484                    # AMR collision box x half length (d200_base_link box 0.968 x 0.654 x 0.2515)
PULL_OUT = 0.100
SUCTION_PRESS_TCP = 'arm_left_suction_tcp_compressed'
LEFT_TCP, RIGHT_TCP = 'arm_left_suction_tcp_compressed', 'arm_right_gripper_tcp'
GRIP_TCP, GRIP_STOP, G_NOM = 0.1932, 0.1675, 0.004
BOX_DIMS = {'agent_monorail': (0.230, 0.250, 0.015), 'angiolite': (0.240, 0.250, 0.016), 'accuforce': (0.220, 0.250, 0.016),
            'ultimaster_nagomi': (0.250, 0.295, 0.016), 'conqueror_nc_pro': (0.230, 0.250, 0.012)}


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def Tmat(R=None, t=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def axis_R(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def Rz(a):
    return axis_R([0, 0, 1], a)


def rot_err(R1, R2):
    """angle (rad) of R1^T R2"""
    c = (np.trace(R1.T @ R2) - 1) / 2
    return math.acos(max(-1.0, min(1.0, c)))


class Robot:
    def __init__(self, urdf_path=URDF_PATH):
        self.u = URDF.from_xml_string(Path(urdf_path).read_text())
        self.joint = {j.name: j for j in self.u.joints}
        self.parent_joint = {j.child: j for j in self.u.joints}
        self.children = {}
        for j in self.u.joints:
            self.children.setdefault(j.parent, []).append(j)
        self.link = {l.name: l for l in self.u.links}
        self._col = {}
        self.arm_joints = {a: [f'arm_{a}_joint{k}' for k in range(1, 7)] for a in ('left', 'right')}
        self.qmin = {a: np.array([self.joint[n].limit.lower for n in self.arm_joints[a]]) for a in self.arm_joints}
        self.qmax = {a: np.array([self.joint[n].limit.upper for n in self.arm_joints[a]]) for a in self.arm_joints}
        self.vmax = 0.785                       # rad/s (joint_limits.yaml)
        self.kdl = {a: self._build_kdl(a) for a in ('left', 'right')}

    # ---------------------------------------------------------------- kinematics
    def _origin(self, j):
        o = j.origin
        if o is None:
            return np.eye(4)
        return Tmat(rpy_R(*o.rpy), o.xyz)

    def _joint_T(self, j, q):
        if j.type in ('revolute', 'continuous'):
            return Tmat(axis_R(j.axis, q))
        if j.type == 'prismatic':
            return Tmat(None, np.asarray(j.axis, float) * q)
        return np.eye(4)

    def path_to(self, link):
        js = []
        while link in self.parent_joint:
            j = self.parent_joint[link]
            js.append(j)
            link = j.parent
        return js[::-1]

    def T_world(self, link, q):
        T = np.eye(4)
        for j in self.path_to(link):
            T = T @ self._origin(j) @ self._joint_T(j, q.get(j.name, 0.0))
        return T

    def subtree(self, root, T_root, q, out=None):
        out = {} if out is None else out
        out[root] = T_root
        for j in self.children.get(root, []):
            self.subtree(j.child, T_root @ self._origin(j) @ self._joint_T(j, q.get(j.name, 0.0)), q, out)
        return out

    def arm_base(self, arm, lift, slider):
        q = {'lift_joint': lift, f'slider_{arm}_joint': slider}
        return self.T_world(f'arm_{arm}_base_link', q)

    def _build_kdl(self, arm):
        tcp = LEFT_TCP if arm == 'left' else RIGHT_TCP
        js = self.path_to(tcp)
        k = js.index(self.parent_joint[f'arm_{arm}_base_link'])
        js = js[k + 1:]                        # base_link -> tcp
        ch = kdl.Chain()
        for j in js:
            o = j.origin
            F = kdl.Frame(kdl.Rotation.RPY(*o.rpy) if o else kdl.Rotation.Identity(), kdl.Vector(*o.xyz) if o else kdl.Vector(0, 0, 0))
            if j.type == 'revolute':
                ax = F.M * kdl.Vector(*j.axis)
                ch.addSegment(kdl.Segment(j.child, kdl.Joint(j.name, F.p, ax, kdl.Joint.RotAxis), F))
            else:
                ch.addSegment(kdl.Segment(j.child, kdl.Joint(kdl.Joint.Fixed), F)) if hasattr(kdl.Joint, 'Fixed') else ch.addSegment(kdl.Segment(j.child, kdl.Joint(getattr(kdl.Joint, 'None')), F))
        n = ch.getNrOfJoints()
        fk = kdl.ChainFkSolverPos_recursive(ch)
        ikv = kdl.ChainIkSolverVel_pinv(ch)
        qmin = kdl.JntArray(n)
        qmax = kdl.JntArray(n)
        for i in range(n):
            qmin[i], qmax[i] = float(self.qmin[arm][i]), float(self.qmax[arm][i])
        ik = kdl.ChainIkSolverPos_NR_JL(ch, qmin, qmax, fk, ikv, 80, 1e-5)
        jac = kdl.ChainJntToJacSolver(ch)
        return dict(chain=ch, fk=fk, ikv=ikv, ik=ik, jac=jac, n=n, qmin=qmin, qmax=qmax)      # keep every KDL object alive (NR_JL stores references)

    def fk_tcp(self, arm, q):
        K = self.kdl[arm]
        ja = kdl.JntArray(K['n'])
        for i in range(K['n']):
            ja[i] = float(q[i])
        f = kdl.Frame()
        K['fk'].JntToCart(ja, f)
        T = np.eye(4)
        T[:3, :3] = np.array([[f.M[i, j] for j in range(3)] for i in range(3)])
        T[:3, 3] = [f.p[0], f.p[1], f.p[2]]
        return T

    def ik(self, arm, T_base_tcp, seed, tol_pos=1e-3, tol_rot=0.01):
        """KDL NR_JL from a seed; returns q (6,) or None (the solution is re-checked with FK)."""
        K = self.kdl[arm]
        q0 = kdl.JntArray(K['n'])
        for i in range(K['n']):
            q0[i] = float(seed[i])
        Fd = kdl.Frame(kdl.Rotation(*[float(T_base_tcp[i, j]) for i in range(3) for j in range(3)]), kdl.Vector(*[float(x) for x in T_base_tcp[:3, 3]]))
        qo = kdl.JntArray(K['n'])
        r = K['ik'].CartToJnt(q0, Fd, qo)
        if r < 0:
            return None
        q = np.array([qo[i] for i in range(K['n'])])
        if np.any(q < self.qmin[arm] - 1e-6) or np.any(q > self.qmax[arm] + 1e-6):
            return None
        T = self.fk_tcp(arm, q)
        if np.linalg.norm(T[:3, 3] - T_base_tcp[:3, 3]) > tol_pos or rot_err(T[:3, :3], T_base_tcp[:3, :3]) > tol_rot:
            return None
        return q

    def ik_multi(self, arm, T_base_tcp, seeds):
        best = None
        for s in seeds:
            q = self.ik(arm, T_base_tcp, s)
            if q is not None and (best is None or np.linalg.norm(q - seeds[0]) < np.linalg.norm(best - seeds[0])):
                best = q
        return best

    def manipulability(self, arm, q):
        K = self.kdl[arm]
        ja = kdl.JntArray(K['n'])
        for i in range(K['n']):
            ja[i] = float(q[i])
        J = kdl.Jacobian(K['n'])
        K['jac'].JntToJac(ja, J)
        M = np.array([[J[i, j] for j in range(K['n'])] for i in range(6)])
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv.min()), float(np.prod(sv))

    # ---------------------------------------------------------------- collision samples
    def _geom_points(self, g, T):
        name = type(g).__name__
        if name == 'Box':
            sx, sy, sz = g.size
            pts = []
            n = [max(2, int(math.ceil(s / 0.02)) + 1) for s in (sx, sy, sz)]
            gx, gy, gz = (np.linspace(-s / 2, s / 2, k) for s, k in zip((sx, sy, sz), n))
            for X in gx:
                for Y in gy:
                    for Z in gz:
                        if abs(abs(X) - sx / 2) < 1e-9 or abs(abs(Y) - sy / 2) < 1e-9 or abs(abs(Z) - sz / 2) < 1e-9:
                            pts.append((X, Y, Z))
            P = np.array(pts)
        elif name == 'Cylinder':
            r, L = g.radius, g.length
            ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            zs = np.linspace(-L / 2, L / 2, max(2, int(math.ceil(L / 0.02)) + 1))
            P = np.array([(r * math.cos(a), r * math.sin(a), z) for z in zs for a in ang] + [(0, 0, -L / 2), (0, 0, L / 2)])
        else:
            fn = g.filename
            pk, rel = fn[len('package://'):].split('/', 1)
            m = trimesh.load(str(PKG[pk] / rel), force='mesh')
            sc = getattr(g, 'scale', None)
            V = np.asarray(m.vertices, float) * (np.asarray(sc, float) if sc is not None else 1.0)
            hull = trimesh.Trimesh(V, m.faces).convex_hull
            P = np.vstack([np.asarray(hull.vertices), hull.sample(max(60, int(hull.area / 0.0004)))])
        return P @ T[:3, :3].T + T[:3, 3]

    def col_points(self, link):
        """sample points of the link's collision geometry in the link frame (cached; thinned to <= 1500)"""
        if link in self._col:
            return self._col[link]
        pts = []
        for c in self.link[link].collisions:
            T = Tmat(rpy_R(*c.origin.rpy), c.origin.xyz) if c.origin is not None else np.eye(4)
            pts.append(self._geom_points(c.geometry, T))
        P = np.vstack(pts) if pts else np.zeros((0, 3))
        if len(P) > 1500:
            P = P[np.random.default_rng(0).choice(len(P), 1500, replace=False)]
        self._col[link] = P
        return P

    def groups(self, arm):
        a = arm
        arm_links = [f'arm_{a}_{n}' for n in ('base_link', 'motor1', 'link1', 'motor2', 'link2', 'motor3', 'link3', 'motor4', 'link4', 'motor5', 'link5', 'link6')]
        tool = ['arm_left_suction_tool'] if a == 'left' else ['arm_right_gripper_base', 'arm_right_finger_left', 'arm_right_finger_right']
        return arm_links, tool

    def arm_points(self, arm, T_base, q6, fingers=None):
        """world points of the arm links and the tool: returns (arm_pts, tool_pts)"""
        qd = dict(zip(self.arm_joints[arm], q6))
        if fingers is not None:
            qd['arm_right_finger_left_joint'] = fingers
            qd['arm_right_finger_right_joint'] = fingers
        T = self.subtree(f'arm_{arm}_base_link', T_base, qd)
        al, tl = self.groups(arm)
        A = np.vstack([self.col_points(l) @ T[l][:3, :3].T + T[l][:3, 3] for l in al if len(self.col_points(l))])
        Tt = np.vstack([self.col_points(l) @ T[l][:3, :3].T + T[l][:3, 3] for l in tl if len(self.col_points(l))])
        return A, Tt

    def static_points(self, lift, sl, sr):
        """AMR box + stand + lift + sliders + camera boom/body (world)"""
        q = {'lift_joint': lift, 'slider_left_joint': sl, 'slider_right_joint': sr}
        T = self.subtree('world', np.eye(4), q)
        out = {}
        for name, links in (('amr', ['d200_base_link']), ('stand', ['stand', 'camera_riser', 'camera_boom', 'gemini336L_camera_link', 'lift', 'slider_left', 'slider_right'])):
            out[name] = np.vstack([self.col_points(l) @ T[l][:3, :3].T + T[l][:3, 3] for l in links if len(self.col_points(l))])
        return out


# -------------------------------------------------------------------- cabinet / obstacles
class Cabinet:
    def __init__(self, path=HERE.parent / 'data' / 'cabinet_geometry.json'):
        d = json.load(open(path))
        self.boxes = {k: (np.array(v[0]), np.array(v[1])) for k, v in d['boxes'].items()}
        self.cartons = d['cartons']
        self.shelf_top = {k: self.boxes[f'Shelf_0{k}'][1][2] for k in range(1, 8)}
        self.clear = {k: self.boxes[f'Shelf_0{k + 1}'][0][2] - self.shelf_top[k] for k in range(1, 7)}

    def pick(self, shelf, box, which):
        items = [it for it in self.cartons[f'Shelf_0{shelf}']]
        order = sorted([i for i, it in enumerate(items) if it['box'] == box], key=lambda i: items[i]['c'][0])
        i = {'mid': order[len(order) // 2], 'first': order[0], 'last': order[-1]}[which]
        return items[i]

    def static_obbs(self, shelf, target):
        """cabinet boxes + carton blocks (all shelves; target shelf split around the target) as (centre, half) AABBs in the cabinet frame"""
        out = {}
        for k, (lo, hi) in self.boxes.items():
            out['cab:' + k] = ((lo + hi) / 2, (hi - lo) / 2)
        for sh, items in self.cartons.items():
            k = int(sh[-1])
            items = sorted(items, key=lambda it: it['c'][0])
            groups = []
            for it in items:
                if groups and groups[-1][0]['box'] == it['box']:
                    groups[-1].append(it)
                else:
                    groups.append([it])
            for gi, g in enumerate(groups):
                if k == shelf and any(it['name'] == target['name'] for it in g):
                    parts = [[it for it in g if it['c'][0] < target['c'][0] - 1e-6], [it for it in g if it['c'][0] > target['c'][0] + 1e-6]]
                else:
                    parts = [g]
                for pi, part in enumerate(parts):
                    if not part:
                        continue
                    h, l, t = BOX_DIMS[part[0]['box']]
                    xs = [it['c'][0] for it in part]
                    x0, x1 = min(xs) - t / 2 - 0.0005, max(xs) + t / 2 + 0.0005
                    zc = part[0]['c'][2]
                    lo = np.array([x0, part[0]['c'][1] - l / 2, zc - h / 2])
                    hi = np.array([x1, part[0]['c'][1] + l / 2, zc + h / 2])
                    out[f'carton:{sh}:{gi}:{pi}'] = ((lo + hi) / 2, (hi - lo) / 2)
        return out


def obb_dist_pts(P, c, R, half):
    L = (P - c) @ R
    d = np.maximum(np.abs(L) - half, 0.0)
    return np.linalg.norm(d, axis=1)


class World:
    """cabinet placed relative to the robot (world = D200 footprint) + obstacle query"""

    def __init__(self, cab, shelf, target, d, lat, dx=0.0, dy=0.0, dyaw_deg=0.0):
        self.cab, self.shelf, self.target = cab, shelf, target
        x_c0 = target['c'][0]
        # cabinet -> world: yaw -90 deg about z, then the parking error (dx, dy, dyaw about the robot z through the AMR centre)
        Rw = Rz(-math.pi / 2)
        tw = np.array([AMR_FRONT_X + d, lat + x_c0, 0.0])
        Rp = Rz(math.radians(dyaw_deg))
        self.R = Rp @ Rw
        self.t = Rp @ tw + np.array([dx, dy, 0.0])
        self.static = cab.static_obbs(shelf, target)
        self.h, self.l, self.t_carton = BOX_DIMS[target['box']]

    def to_world(self, p_c):
        return self.R @ np.asarray(p_c, float) + self.t

    def T_world(self, R_c, p_c):
        return Tmat(self.R @ R_c, self.to_world(p_c))

    def static_dist(self, P, skip=()):
        """min distance of world points P to each static obstacle group: dict name -> min dist"""
        Pc = (P - self.t) @ self.R               # world -> cabinet frame
        res = {}
        for name, (c, half) in self.static.items():
            if name in skip:
                continue
            res[name] = float(obb_dist_pts(Pc, c, np.eye(3), half).min())
        return res

    def carton_dist(self, P, pull):
        """distance of world points to the (pulled) target carton: box in cabinet frame, centre shifted by -pull in y"""
        Pc = (P - self.t) @ self.R
        c = np.array(self.target['c'], float) + np.array([0.0, -pull, 0.0])
        half = np.array([self.t_carton / 2, self.l / 2, self.h / 2])
        # carton frame: x height, y length, z thickness -> cabinet axes (z up, y into, x across): half in cabinet axes = (t/2, l/2, h/2)
        return float(obb_dist_pts(Pc, c, np.eye(3), half).min())
