#!/usr/bin/env python3
"""Generate config/cell_tool_<name>.yaml (+ the two finger-hull visual meshes) from a cell_planner_check design JSON.

  python3 scripts/make_tool_config.py --design /home/farhan/igus_rebel_ros2/tools/cell_planner_check/designs/v3_r_1269_A0_B0.json \
        --out config/cell_tool_v3_r_1269.yaml

The geometry follows tools/cell_planner_check/gen_cell.py (build() for the 'real' geometry, union of the grasp+open finger boxes, camera pad
3 mm) and gen_cell_v3.py (camera B + bracket with end spheres) so the MoveIt collision model equals the one the v3 study validated.  Every
primitive is written with explicit xyz / rpy strings in the link6 frame, so urdf/rebel_cell_tool.xacro needs no geometry maths: a new tooling
revision (v4, ultrasonic cutter, cad/dual_arm_tooling_v2) is a new YAML (+ meshes) and the `tool_config` xacro argument.

Optional --smc-stl re-creates meshes/tool/gripper_smc_link6.stl from the SMC CAD download (same transform as gen_cell.build_gripper_stl).
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_URI = 'package://rebel_dual_arm_stand_description'
CAMERA_PAD = 0.003           # gen_cell_v3 --camera-pad-mm default
BLADE_R = 0.0015


def f(v):
    return '%.6f' % v


def s3(v):
    return ' '.join(f(x) for x in v)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_to_rpy(R):
    """ROS URDF rpy (R = Rz(y) Ry(p) Rx(r)), same as cellgeom.rot_to_rpy"""
    sp = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sp)))
    if abs(abs(sp) - 1.0) < 1e-9:
        return 0.0, p, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def z_to_vec_rpy(v):
    """rpy of a frame whose +z axis points along v (cellgeom.z_to_vec_rpy); returns (rpy, length)"""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    z = v / n
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rot_to_rpy(np.stack([x, y, z], axis=1)), n


def prim_box(size, centre, rpy=(0, 0, 0)):
    return dict(type='box', size=s3(size), xyz=s3(centre), rpy=s3(rpy))


def prim_sphere(r, centre):
    return dict(type='sphere', radius=f(r), xyz=s3(centre), rpy='0 0 0')


def prim_cyl(a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rpy, L = z_to_vec_rpy(b - a)
    return dict(type='cylinder', radius=f(r), length=f(L), xyz=s3((a + b) / 2), rpy=s3(rpy))


def capsule(a, b, r):
    return [prim_cyl(a, b, r), prim_sphere(r, a), prim_sphere(r, b)]


def link(name, parent, xyz=(0, 0, 0), rpy=(0, 0, 0), visual=None, collision=None, rgba='0.5 0.5 0.5 1'):
    return dict(name=name, parent=parent, joint_xyz=s3(xyz), joint_rpy=s3(rpy), rgba=rgba, visual=visual or [], collision=collision or [])


def camera_links(prefix_name, cam, pad, bracket_spheres, rgba):
    """`<name>_link` (ROS body frame, origin = lens-face centre, x forward), `<name>_optical_frame`, `<name>_bracket`"""
    T = np.array(cam['T_link6_camera'])
    Ro, p = T[:3, :3], T[:3, 3]                  # optical frame (z = viewing axis) in link6
    bx, by, bz = cam['body_size']                # along optical x, y, z (z = thickness; body extends BEHIND the lens face, -z)
    Rb = np.stack([Ro[:, 2], -Ro[:, 0], -Ro[:, 1]], axis=1)       # ROS body frame: x = optical z, y = -optical x, z = -optical y
    # rotation of the optical frame expressed in the body frame (standard REP-103 body -> optical)
    Rbo = Rb.T @ Ro
    assert np.abs(Rbo - np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])).max() < 1e-5   # design matrices are rounded to 6 decimals
    body_vis = prim_box((bz, bx, by), (-bz / 2, 0, 0))
    body_col = prim_box((bz + 2 * pad, bx + 2 * pad, by + 2 * pad), (-bz / 2, 0, 0))
    out = [link(prefix_name + '_link', 'link6', p, rot_to_rpy(Rb), visual=[body_vis], collision=[body_col], rgba=rgba),
           link(prefix_name + '_optical_frame', prefix_name + '_link', (0, 0, 0), (-math.pi / 2, 0, -math.pi / 2), rgba=rgba)]
    br = cam['bracket']
    prims = [prim_cyl(br['frm'], br['to'], br['radius'])]
    if bracket_spheres:
        prims += [prim_sphere(br['radius'], br['frm']), prim_sphere(br['radius'], br['to'])]
    out.append(link(prefix_name + '_bracket', 'link6', visual=prims[:1], collision=prims, rgba=rgba))
    info = dict(T_link6_optical=T.tolist(), R_body_in_link6=Rb.tolist(), body_size_xyz_optical=[bx, by, bz], pad=pad,
                lens=cam.get('lens'), focus_distance_m=cam.get('f0_m'))
    return out, {k: v for k, v in info.items() if v is not None}


def finger_hulls(finger_stl, outdir):
    """convex hull of the real finger mesh in the link6 frame for the GRASP state (visual only); same transform as gen_cell.build_finger_hulls"""
    import trimesh
    JAW_Y, JAW_Z = -0.0415, 0.070
    XREF = 0.016 + 0.0186
    base = trimesh.load(finger_stl)
    names = {}
    for side in (-1, 1):
        m = base.copy()
        M = np.eye(4)
        if side < 0:
            M[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
            off = np.array([-XREF * 1000, JAW_Y * 1000 + 5, JAW_Z * 1000])
        else:
            M[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0.0]])
            off = np.array([XREF * 1000, JAW_Y * 1000 + 5, JAW_Z * 1000])
        M[:3, 3] = off
        m.apply_transform(M)
        m.apply_scale(0.001)
        fn = 'finger_%s_grasp_hull.stl' % ('L' if side < 0 else 'R')
        m.convex_hull.export(os.path.join(outdir, fn))
        names[side] = fn
    return names


def build_gripper_stl(src, out):
    import trimesh
    g = trimesh.load(src)
    g.apply_scale(0.001)
    Rg = np.eye(4)
    Rg[:3, :3] = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    g.apply_transform(Rg)
    b = g.bounds
    g.apply_translation([0, -(b[0][1] + b[1][1]) / 2, 0.0527 - b[0][2]])
    g.export(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--finger-stl', default='')
    ap.add_argument('--gripper-stl', default='', help='gripper mesh already in the link6 frame (tools/cell_planner_check/assets/gripper_smc_link6.stl)')
    ap.add_argument('--smc-stl', default='', help='re-create the gripper mesh from the SMC CAD STL instead of copying --gripper-stl')
    a = ap.parse_args()
    des = json.load(open(a.design))
    meshdir = os.path.join(PKG, 'meshes', 'tool')
    os.makedirs(meshdir, exist_ok=True)
    grip_out = os.path.join(meshdir, 'gripper_smc_link6.stl')
    if a.smc_stl:
        build_gripper_stl(a.smc_stl, grip_out)
    elif a.gripper_stl:
        shutil.copy(a.gripper_stl, grip_out)
    fingers = finger_hulls(a.finger_stl, meshdir) if a.finger_stl else None

    zero = (0, 0, 0)
    links = []
    links.append(link('tool_flange', 'link6', (0, 0, 0.0527)))         # gripper mounting face, link6 orientation
    links.append(link('gripper', 'link6',
                      visual=[dict(type='mesh', file=PKG_URI + '/meshes/tool/gripper_smc_link6.stl', xyz='0 0 0', rpy='0 0 0')],
                      collision=[dict(type='mesh', file=PKG_URI + '/meshes/tool/gripper_smc_link6.stl', xyz='0 0 0', rpy='0 0 0')],
                      rgba='0.35 0.4 0.55 1'))
    fin_vis = [dict(type='mesh', file=PKG_URI + '/meshes/tool/' + fingers[s], xyz='0 0 0', rpy='0 0 0') for s in (-1, 1)] if fingers else []
    fin_col = []
    for key in ('boxes_grasp', 'boxes_open'):                           # gen_cell --fingers union
        for side in ('L', 'R'):
            for b in des['fingers'][key][side]:
                b = np.array(b)
                fin_col.append(prim_box(b[:, 1] - b[:, 0], b.mean(1)))
    links.append(link('fingers', 'link6', visual=fin_vis, collision=fin_col, rgba='0.54 0.43 0.23 1'))
    k = des['knife']
    post = prim_cyl(k['post']['frm'], k['post']['to'], k['post']['radius'])
    links.append(link('knife_post', 'link6', visual=[post], collision=[post], rgba='0.06 0.48 0.42 1'))
    cl = prim_sphere(k['clamp']['radius'], k['clamp']['centre'])
    links.append(link('knife_clamp', 'link6', visual=[cl], collision=[cl], rgba='0.06 0.48 0.42 1'))
    hp = capsule(k['handpiece']['frm'], k['handpiece']['to'], k['handpiece']['radius'])
    links.append(link('knife_handpiece', 'link6', visual=hp, collision=hp, rgba='0.06 0.48 0.42 1'))
    a_, b_ = np.array(k['blade']['frm']), np.array(k['blade']['to'])
    blade = prim_cyl(a_, b_, BLADE_R) if np.linalg.norm(b_ - a_) > 1e-4 else prim_sphere(BLADE_R, a_)
    links.append(link('knife_blade', 'link6', visual=[blade], collision=[], rgba='0.8 0.8 0.85 1'))     # collision false: last 20 mm exempt
    meta = {}
    cams, meta['wrist_camera'] = camera_links('wrist_camera', des['camera'], CAMERA_PAD, False, '0.1 0.1 0.1 1')
    links += cams
    if des.get('camera_B'):
        cams, meta['knife_camera'] = camera_links('knife_camera', des['camera_B'], CAMERA_PAD, True, '0.1 0.1 0.1 1')
        links += cams
    T = np.array(des['T_link6_tcp'])
    tip_vis = [dict(type='sphere', radius='0.006', xyz='0 0 0', rpy='0 0 0')]
    links.append(link('knife_tip', 'link6', T[:3, 3], rot_to_rpy(T[:3, :3]), visual=tip_vis, collision=[], rgba='0.1 1 0.1 1'))

    out = {
        'schema': 'rebel_dual_arm/cell_tool/v1',
        'name': des['name'],
        'source': {
            'design_json': os.path.relpath(a.design, '/home/farhan/igus_rebel_ros2') if a.design.startswith('/home/farhan/igus_rebel_ros2') else a.design,
            'design_source': des.get('source'),
            'generator': 'src/rebel_dual_arm_stand_description/scripts/make_tool_config.py, geometry conventions of tools/cell_planner_check/gen_cell.py + gen_cell_v3.py',
            'note': 'Tooling is IDENTICAL on both arms (not mirrored), as in cell_planner_check. All frames are expressed in the link6 frame of the arm. '
                    'Link names get the arm prefix (e.g. arm_left_knife_tip). TCP = knife_tip (no collision geometry; blade collision=false, last 20 mm exempt).',
        },
        'T_link6_tcp': des['T_link6_tcp'],
        'camera_pad_m': CAMERA_PAD,
        'cameras': meta,
        'tcp_link': 'knife_tip',
        'links': links,
    }
    with open(a.out, 'w') as fh:
        fh.write('# Generated by scripts/make_tool_config.py from %s (%s). Edit the generator or the design, not this file.\n'
                 % (out['source']['design_json'], des.get('source')))
        yaml.safe_dump(out, fh, sort_keys=False, default_flow_style=None, width=200)
    print('wrote', a.out, 'links:', [l['name'] for l in links])


if __name__ == '__main__':
    main()
