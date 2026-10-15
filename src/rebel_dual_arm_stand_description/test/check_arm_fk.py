#!/usr/bin/env python3
"""Check that urdf/rebel_arm_macro.xacro is kinematically (and link-for-link) identical to OUR single-arm URDF
src/igus_rebel_description/urdf/igus_rebel2.urdf.

What is compared (plain XML + numpy, no urdf_parser_py needed):
  * every joint: type, parent/child (after removing the prefix), origin xyz/rpy, axis, limits (lower/upper/effort/velocity)
  * every link: inertial (mass, origin, inertia), visual + collision (origin, geometry type/size, mesh filename)
  * forward kinematics of link6 relative to base_link for N random joint vectors inside the joint limits
    (tolerance 1e-6 m translation / 1e-6 rad rotation)
  * the same FK for both arm chains inside the expanded dual-arm URDF (rebel_dual_arm.urdf.xacro), when
    rebel_dual_arm_stand_description is installed/sourced (skipped otherwise)

Run:  python3 test/check_arm_fk.py [--n 300] [--json out.json]     or    python3 -m pytest -p no:anyio test/check_arm_fk.py
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)
MACRO = os.path.join(PKG_DIR, 'urdf', 'rebel_arm_macro.xacro')
DUAL_XACRO = os.path.join(PKG_DIR, 'urdf', 'rebel_dual_arm.urdf.xacro')
TOL_P = 1e-6
TOL_R = 1e-6


def our_urdf_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('igus_rebel_description'), 'urdf', 'igus_rebel2.urdf')
        if os.path.exists(p):
            return p
    except Exception:
        pass
    # repo checkout: src/rebel_dual_arm_stand_description/test -> src/igus_rebel_description
    return os.path.join(PKG_DIR, '..', 'igus_rebel_description', 'urdf', 'igus_rebel2.urdf')


def vec(s, n=3):
    return np.array([float(v) for v in s.split()], float) if s else np.zeros(n)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def T_origin(el):
    o = el.find('origin') if el is not None else None
    xyz = vec(o.get('xyz', '0 0 0')) if o is not None else np.zeros(3)
    rpy = vec(o.get('rpy', '0 0 0')) if o is not None else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def axis_rot(axis, q):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


class Urdf:
    def __init__(self, root):
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.links = {l.get('name'): l for l in root.findall('link')}
        self.by_parent = {}
        for j in self.joints.values():
            self.by_parent.setdefault(j.find('parent').get('link'), []).append(j)

    def chain(self, base, tip):
        """list of joints from base to tip (DFS through children)"""
        def dfs(link, path):
            if link == tip:
                return path
            for j in self.by_parent.get(link, []):
                r = dfs(j.find('child').get('link'), path + [j])
                if r is not None:
                    return r
            return None
        return dfs(base, [])

    def fk(self, chain, q):
        """q: dict joint name -> value; returns 4x4 of the chain tip in the chain base frame"""
        T = np.eye(4)
        for j in chain:
            T = T @ T_origin(j)
            if j.get('type') in ('revolute', 'continuous'):
                T = T @ axis_rot(vec(j.find('axis').get('xyz')), q[j.get('name')])
        return T


def rot_err(Ra, Rb):
    R = Ra.T @ Rb          # numerically stable angle (atan2 form; acos loses precision near 0)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return math.atan2(np.linalg.norm(v), (np.trace(R) - 1) / 2)


def strip(name, prefix):
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def geom_sig(g, prefix):
    if g is None:
        return None
    c = g[0]
    d = dict(c.attrib)
    if 'filename' in d:
        return (c.tag, d['filename'])
    return (c.tag, tuple(sorted((k, tuple(float(x) for x in v.split())) for k, v in d.items())))


def compare_structure(ref, new, prefix):
    """returns (list of difference strings, number of compared items)"""
    diffs = []
    n = 0
    for name, j in ref.joints.items():
        pj = new.joints.get(prefix + name)
        if pj is None:
            diffs.append('joint missing: ' + prefix + name)
            continue
        n += 1
        if j.get('type') != pj.get('type'):
            diffs.append('%s type %s != %s' % (name, j.get('type'), pj.get('type')))
        for tag in ('parent', 'child'):
            if prefix + j.find(tag).get('link') != pj.find(tag).get('link'):
                diffs.append('%s %s %s != %s' % (name, tag, j.find(tag).get('link'), pj.find(tag).get('link')))
        a, b = T_origin(j), T_origin(pj)
        if np.abs(a - b).max() > 1e-9:
            diffs.append('%s origin differs (max %.3g)' % (name, np.abs(a - b).max()))
        if j.find('axis') is not None and pj.find('axis') is not None:
            if np.abs(vec(j.find('axis').get('xyz')) - vec(pj.find('axis').get('xyz'))).max() > 1e-12:
                diffs.append('%s axis differs' % name)
        la, lb = j.find('limit'), pj.find('limit')
        if (la is None) != (lb is None):
            diffs.append('%s limit presence differs' % name)
        elif la is not None:
            for k in ('lower', 'upper', 'effort', 'velocity'):
                if abs(float(la.get(k)) - float(lb.get(k))) > 1e-9:
                    diffs.append('%s limit %s %s != %s' % (name, k, la.get(k), lb.get(k)))
    for name, l in ref.links.items():
        pl = new.links.get(prefix + name)
        if pl is None:
            diffs.append('link missing: ' + prefix + name)
            continue
        n += 1
        ia, ib = l.find('inertial'), pl.find('inertial')
        if (ia is None) != (ib is None):
            diffs.append('%s inertial presence differs' % name)
        elif ia is not None:
            if abs(float(ia.find('mass').get('value')) - float(ib.find('mass').get('value'))) > 1e-9:
                diffs.append('%s mass differs' % name)
            if np.abs(T_origin(ia) - T_origin(ib)).max() > 1e-9:
                diffs.append('%s inertial origin differs' % name)
            for k, v in ia.find('inertia').attrib.items():
                if abs(float(v) - float(ib.find('inertia').get(k))) > 1e-12:
                    diffs.append('%s inertia %s differs' % (name, k))
        for kind in ('visual', 'collision'):
            ea, eb = l.findall(kind), pl.findall(kind)
            if len(ea) != len(eb):
                diffs.append('%s %s count %d != %d' % (name, kind, len(ea), len(eb)))
                continue
            for x, y in zip(ea, eb):
                if np.abs(T_origin(x) - T_origin(y)).max() > 1e-9:
                    diffs.append('%s %s origin differs' % (name, kind))
                if geom_sig(x.find('geometry'), prefix) != geom_sig(y.find('geometry'), prefix):
                    diffs.append('%s %s geometry %s != %s' % (name, kind, geom_sig(x.find('geometry'), prefix), geom_sig(y.find('geometry'), prefix)))
    return diffs, n


def expand_macro_alone(prefix='t_'):
    """xacro-expand a tiny wrapper that only instantiates the arm macro with an identity mount"""
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, 'w.urdf.xacro')
        open(w, 'w').write('''<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="w">
  <xacro:include filename="%s"/>
  <link name="world"/>
  <xacro:rebel_arm_chain prefix="%s" parent_link="world" xyz="0 0 0" rpy="0 0 0"/>
</robot>''' % (MACRO, prefix))
        out = subprocess.run(['xacro', w], check=True, capture_output=True, text=True).stdout
    return ET.fromstring(out)


def expand_dual():
    out = subprocess.run(['xacro', DUAL_XACRO, 'hardware_protocol:=mock_hardware'], capture_output=True, text=True)
    if out.returncode != 0:
        return None, out.stderr[-800:]
    return ET.fromstring(out.stdout), ''


def random_q(urdf, joint_names, rng):
    q = {}
    for jn in joint_names:
        lim = urdf.joints[jn].find('limit')
        q[jn] = rng.uniform(float(lim.get('lower')), float(lim.get('upper')))
    return q


def fk_compare(ref, ref_chain, new, new_chain, prefix, n, rng):
    emax_p = emax_r = 0.0
    rnames = [j.get('name') for j in ref_chain if j.get('type') == 'revolute']
    for _ in range(n):
        q = random_q(ref, rnames, rng)
        qn = {prefix + k: v for k, v in q.items()}
        A = ref.fk(ref_chain, q)
        B = new.fk(new_chain, qn)
        emax_p = max(emax_p, float(np.linalg.norm(A[:3, 3] - B[:3, 3])))
        emax_r = max(emax_r, rot_err(A[:3, :3], B[:3, :3]))
    # corners of the joint box as well
    for corner in range(64):
        q = {}
        for i, jn in enumerate(rnames):
            lim = ref.joints[jn].find('limit')
            q[jn] = float(lim.get('upper' if (corner >> i) & 1 else 'lower'))
        qn = {prefix + k: v for k, v in q.items()}
        A = ref.fk(ref_chain, q)
        B = new.fk(new_chain, qn)
        emax_p = max(emax_p, float(np.linalg.norm(A[:3, 3] - B[:3, 3])))
        emax_r = max(emax_r, rot_err(A[:3, :3], B[:3, :3]))
    return emax_p, emax_r


def run(n=300, seed=1):
    rng = np.random.default_rng(seed)
    ref = Urdf(ET.parse(our_urdf_path()).getroot())
    ref_chain = ref.chain('base_link', 'link6')
    res = {'reference_urdf': os.path.realpath(our_urdf_path()), 'n_random': n, 'seed': seed, 'tol_m': TOL_P, 'tol_rad': TOL_R, 'checks': {}}
    prefix = 't_'
    macro = Urdf(expand_macro_alone(prefix))
    diffs, items = compare_structure(ref, macro, prefix)
    mchain = macro.chain(prefix + 'base_link', prefix + 'link6')
    ep, er = fk_compare(ref, ref_chain, macro, mchain, prefix, n, rng)
    res['checks']['macro_alone'] = {'structure_items_compared': items, 'structure_differences': diffs, 'fk_max_err_m': ep, 'fk_max_err_rad': er,
                                    'fk_configs': n + 64}
    dual_root, err = expand_dual()
    if dual_root is None:
        res['checks']['dual_urdf'] = {'skipped': True, 'reason': 'xacro of rebel_dual_arm.urdf.xacro failed (package not installed/sourced?): ' + err}
    else:
        dual = Urdf(dual_root)
        for arm in ('arm_left_', 'arm_right_'):
            diffs, items = compare_structure(ref, dual, arm)
            ch = dual.chain(arm + 'base_link', arm + 'link6')
            ep, er = fk_compare(ref, ref_chain, dual, ch, arm, n, rng)
            res['checks']['dual_urdf_' + arm.rstrip('_')] = {'structure_items_compared': items, 'structure_differences': diffs,
                                                            'fk_max_err_m': ep, 'fk_max_err_rad': er, 'fk_configs': n + 64}
    ok = True
    for v in res['checks'].values():
        if v.get('skipped'):
            continue
        ok = ok and not v['structure_differences'] and v['fk_max_err_m'] < TOL_P and v['fk_max_err_rad'] < TOL_R
    res['pass'] = ok
    return res


def test_arm_matches_single_arm_urdf():
    res = run(250)
    assert res['pass'], json.dumps(res, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    r = run(a.n, a.seed)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, 'w'), indent=1)
    sys.exit(0 if r['pass'] else 1)
