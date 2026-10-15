#!/usr/bin/env python3
"""v3 cell generator: wraps gen_cell.build (unchanged) and ADDS (a) camera B + bracket on both arms, (b) the robot-table obstacle (structure 'table') or the real-stand CAD
obstacles (structure 'stand', incl. carrier plates), (c) the SRDF pairs they need.  No collision matrix sampling (every non-adjacent pair is checked: slower but conservative).

  python3 gen_cell_v3.py --design designs/v3_X.json --s 0.448 --out build/v3_X_table --structure table
Table (operator update 2026-10-04): slab at the mount plane, top z = 0, x from -0.60 to +0.110, |y| <= 0.325, 0.40 m deep (assumed).
"""
import sys as _s; _s.dont_write_bytecode = True
import argparse, json, math, os, sys, itertools
import xml.etree.ElementTree as ET
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_cell as GC
import cellgeom as G

TABLE = dict(front=0.110, back=-0.60, half_w=0.325, depth=0.40)


def add_box_link(root, name, size, centre, rpy, parent='world', rgba='camera'):
    lk = ET.SubElement(root, 'link', name=name)
    for kind in ('visual', 'collision'):
        el = ET.SubElement(lk, kind)
        ET.SubElement(el, 'origin', xyz=GC.xyz(centre), rpy=GC.xyz(rpy))
        g = ET.SubElement(el, 'geometry')
        ET.SubElement(g, 'box', size=GC.xyz(size))
        if kind == 'visual':
            ET.SubElement(el, 'material', name=rgba)
    return lk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--s', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--structure', default='table', choices=['table', 'stand'])
    ap.add_argument('--camera-pad-mm', type=float, default=3.0)
    ap.add_argument('--scene-camera', action='store_true', help='static scene camera (Gemini 336) post + body between the arms')
    a = ap.parse_args()
    des = json.load(open(a.design if os.path.exists(a.design) else os.path.join(HERE, a.design)))
    camB = des.get('camera_B')
    stand = a.structure == 'stand'
    meta = GC.build(des, a.s, a.out, 'cad' if stand else 'none', 'union', a.camera_pad_mm / 1000.0, stand, None, 0.0, 'real', 0.0025, 0.25)
    urdf_p = os.path.join(a.out, 'dual_arm.urdf')
    tree = ET.parse(urdf_p)
    root = tree.getroot()
    new_links = {arm: [] for arm in G.ARMS}
    for ai, arm in enumerate(G.ARMS):
        pre = arm + '_'
        link6 = pre + 'link6'
        if camB:
            Tc = np.array(camB['T_link6_camera']); Rc = Tc[:3, :3]
            size = np.array(camB['body_size']) + 2 * a.camera_pad_mm / 1000.0
            centre = Tc[:3, 3] + Rc[:, 2] * (-camB['body_size'][2] / 2)
            add_box_link(root, pre + 'camera_b', size, (0, 0, 0), (0, 0, 0))
            j = ET.SubElement(root, 'joint', name=pre + 'camera_b_joint', type='fixed')
            ET.SubElement(j, 'origin', xyz=GC.xyz(centre), rpy=GC.xyz(G.rot_to_rpy(Rc)))
            ET.SubElement(j, 'parent', link=link6); ET.SubElement(j, 'child', link=pre + 'camera_b')
            br = camB['bracket']
            p0, p1 = np.array(br['frm']), np.array(br['to'])
            lk = ET.SubElement(root, 'link', name=pre + 'camera_b_bracket')
            rpy, Ln = G.z_to_vec_rpy(p1 - p0)
            for kind in ('visual', 'collision'):
                el = ET.SubElement(lk, kind)
                ET.SubElement(el, 'origin', xyz=GC.xyz((p0 + p1) / 2), rpy=GC.xyz(rpy))
                g = ET.SubElement(el, 'geometry')
                ET.SubElement(g, 'cylinder', radius=GC.f(br['radius']), length=GC.f(Ln))
                if kind == 'visual':
                    ET.SubElement(el, 'material', name='camera')
            for nm in ('capsule1', 'capsule2'):
                pass
            for pt in (p0, p1):                                   # spherical ends
                for kind in ('collision',):
                    el = ET.SubElement(lk, kind)
                    ET.SubElement(el, 'origin', xyz=GC.xyz(pt), rpy='0 0 0')
                    g = ET.SubElement(el, 'geometry')
                    ET.SubElement(g, 'sphere', radius=GC.f(br['radius']))
            j = ET.SubElement(root, 'joint', name=pre + 'camera_b_bracket_joint', type='fixed')
            ET.SubElement(j, 'origin', xyz='0 0 0', rpy='0 0 0')
            ET.SubElement(j, 'parent', link=link6); ET.SubElement(j, 'child', link=pre + 'camera_b_bracket')
            new_links[arm] += [pre + 'camera_b', pre + 'camera_b_bracket']
    if not stand:
        c = [(TABLE['front'] + TABLE['back']) / 2, 0.0, -TABLE['depth'] / 2]
        size = [TABLE['front'] - TABLE['back'], 2 * TABLE['half_w'], TABLE['depth']]
        add_box_link(root, 'struct_robot_table', size, (0, 0, 0), (0, 0, 0), rgba='struct')
        j = ET.SubElement(root, 'joint', name='struct_robot_table_joint', type='fixed')
        ET.SubElement(j, 'origin', xyz=GC.xyz(c), rpy='0 0 0')
        ET.SubElement(j, 'parent', link='world'); ET.SubElement(j, 'child', link='struct_robot_table')
    if a.scene_camera:
        add_box_link(root, 'scene_cam_post', [0.04, 0.04, 0.70], (0, 0, 0), (0, 0, 0), rgba='struct')
        j = ET.SubElement(root, 'joint', name='scene_cam_post_joint', type='fixed')
        ET.SubElement(j, 'origin', xyz=GC.xyz([-0.10, 0.0, 0.35]), rpy='0 0 0'); ET.SubElement(j, 'parent', link='world'); ET.SubElement(j, 'child', link='scene_cam_post')
        add_box_link(root, 'scene_cam_body', [0.04, 0.10, 0.04], (0, 0, 0), (0, 0, 0), rgba='camera')
        j = ET.SubElement(root, 'joint', name='scene_cam_body_joint', type='fixed')
        ET.SubElement(j, 'origin', xyz=GC.xyz([-0.08, 0.0, 0.67]), rpy=GC.xyz([0, math.radians(50), 0])); ET.SubElement(j, 'parent', link='world'); ET.SubElement(j, 'child', link='scene_cam_body')
    ET.indent(root, space='  ')
    tree.write(urdf_p, xml_declaration=True, encoding='utf-8')
    # SRDF additions
    srdf_p = os.path.join(a.out, 'dual_arm.srdf')
    st = ET.parse(srdf_p)
    sr = st.getroot()
    have = {frozenset((d.get('link1'), d.get('link2'))) for d in sr.findall('disable_collisions')}
    def dis(x, y, why):
        if x != y and frozenset((x, y)) not in have:
            ET.SubElement(sr, 'disable_collisions', link1=x, link2=y, reason=why)
            have.add(frozenset((x, y)))
    for arm in G.ARMS:
        pre = arm + '_'
        tl = list(meta['arms'][arm]['tool_links']) + [pre + 'knife_blade', pre + 'knife_tip']
        for nl in new_links[arm]:
            for t in tl:
                dis(nl, t, 'Adjacent')
        for x, y in itertools.combinations(new_links[arm], 2):
            dis(x, y, 'Adjacent')
        if camB:
            gname = arm + '_tool'
            for g in sr.findall('group'):
                if g.get('name') == gname:
                    for nl in new_links[arm]:
                        ET.SubElement(g, 'link', name=nl)
        if a.scene_camera:
            for sl in ('scene_cam_post', 'scene_cam_body'): dis(sl, pre + 'base_link', 'Adjacent')
        if not stand:
            dis('struct_robot_table', pre + 'base_link', 'Adjacent')
    if a.scene_camera:
        dis('scene_cam_post', 'scene_cam_body', 'Adjacent')
        if not stand: dis('scene_cam_post', 'struct_robot_table', 'Adjacent')
    ET.indent(sr, space='  ')
    st.write(srdf_p, xml_declaration=True, encoding='utf-8')
    meta['v3'] = dict(structure=a.structure, camera_B=bool(camB), table=None if stand else TABLE)
    meta['collision_links'] = sorted(set(meta['collision_links']) | {l for v in new_links.values() for l in v} | ({'struct_robot_table'} if not stand else set()))
    json.dump(meta, open(os.path.join(a.out, 'cell_meta.json'), 'w'), indent=1)
    print('built', a.out, 'structure', a.structure, 'camera_B', bool(camB))


if __name__ == '__main__':
    main()
