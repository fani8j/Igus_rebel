"""Tiny URDF helpers (plain XML + numpy) shared by the tests of this package: xacro expansion, joint/link access, forward kinematics."""
import math
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


def expand(xacro_file, *args):
    out = subprocess.run(['xacro', xacro_file, *args], check=True, capture_output=True, text=True).stdout
    return ET.fromstring(out)


def vec(s, n=3):
    return np.array([float(v) for v in s.split()], float) if s else np.zeros(n)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def T_origin(el):
    o = el.find('origin') if el is not None else None
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = rpy_to_R(*vec(o.get('rpy', '0 0 0')))
        T[:3, 3] = vec(o.get('xyz', '0 0 0'))
    return T


def axis_rot(ax, a):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class Urdf:
    def __init__(self, root):
        self.root = root
        self.links = {l.get('name'): l for l in root.findall('link')}
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.by_child = {j.find('child').get('link'): j for j in self.joints.values()}
        self.children = {}
        for j in self.joints.values():
            self.children.setdefault(j.find('parent').get('link'), []).append(j.find('child').get('link'))

    def descendants(self, link):
        out, todo = set(), [link]
        while todo:
            for c in self.children.get(todo.pop(), []):
                out.add(c)
                todo.append(c)
        return out

    def fk(self, link, q):
        """pose of `link` in the URDF root frame; q = {joint name: value} for revolute/prismatic joints (missing = 0). Mimic joints follow their master."""
        chain = []
        while link in self.by_child:
            chain.append(self.by_child[link])
            link = self.by_child[link].find('parent').get('link')
        T = np.eye(4)
        for j in reversed(chain):
            T = T @ T_origin(j)
            if j.get('type') in ('revolute', 'prismatic', 'continuous'):
                v = q.get(j.get('name'), 0.0)
                m = j.find('mimic')
                if m is not None:
                    v = float(m.get('multiplier', 1.0)) * q.get(m.get('joint'), 0.0) + float(m.get('offset', 0.0))
                ax = vec(j.find('axis').get('xyz'))
                Tj = np.eye(4)
                if j.get('type') == 'prismatic':
                    Tj[:3, 3] = ax * v
                else:
                    Tj[:3, :3] = axis_rot(ax, v)
                T = T @ Tj
        return T


def rot_err(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return float(np.arctan2(np.linalg.norm(v), (np.trace(R) - 1) / 2))


def limits(j):
    l = j.find('limit')
    return None if l is None else {k: float(l.get(k)) for k in ('lower', 'upper', 'effort', 'velocity') if l.get(k) is not None}
