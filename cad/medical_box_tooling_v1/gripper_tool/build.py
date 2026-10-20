"""Builds every deliverable geometry file: step/parts, step/assembly_*, stl/, meshes/{visual,collision}, manifest.json, urdf/*.xacro, data/mass.json.
Run:  nice -n 19 "$PY" build.py        (about 1 min)
"""
import os, json, math
import numpy as np
import trimesh
import cadquery as cq
import params as P
import parts as K

HERE = P.HERE
OUT = lambda *a: os.path.join(HERE, *a)
for d in ('step/parts', 'stl', 'meshes/visual', 'meshes/collision', 'urdf', 'data'):
    os.makedirs(OUT(d), exist_ok=True)

PRINTED = ['flange_disc', 'gripper_plate', 'top_stop', 'finger_left', 'finger_right', 'pad_left', 'pad_right']
MATERIAL = dict(flange_disc='PA12-CF (PETG ok)', gripper_plate='PA12-CF', top_stop='PA12-CF', finger_left='PA12-CF (PETG ok)', finger_right='PA12-CF (PETG ok)', pad_left='TPU 95A', pad_right='TPU 95A')
COLORS = dict(top_stop=(0.85, 0.2, 0.2), flange_disc=(0.93, 0.55, 0.12), gripper_plate=(0.93, 0.55, 0.12), finger_left=(0.15, 0.55, 0.85), finger_right=(0.15, 0.55, 0.85),
              pad_left=(0.1, 0.1, 0.1), pad_right=(0.1, 0.1, 0.1), smc_body=(0.75, 0.75, 0.78), smc_jaw_left=(0.6, 0.62, 0.66), smc_jaw_right=(0.6, 0.62, 0.66))


def est_mass(shape, mat):
    """printed part mass estimate: 1.6 mm shell solid + 25 % infill in the rest (TPU pads solid)"""
    rho = P.RHO[{'PA12-CF (PETG ok)': 'PA12CF', 'PA12-CF': 'PA12CF', 'TPU 95A': 'TPU'}[mat]]
    V, A = shape.Volume(), shape.Area()
    if mat.startswith('TPU'):
        return rho * V
    vs = min(V, 1.6 * A)
    return rho * (vs + 0.25 * (V - vs))


def write_stl(mesh, path, scale=1.0):
    m = mesh.copy()
    m.apply_scale(scale)
    m.export(path)


def main():
    S0 = K.solids(0.0, with_hw=True)               # closed positions == local finger frame shifted by Z_J
    # ---------------------------------------------------------------- STEP parts (link6, fingers at the closed position) + STL
    summary = {}
    for n in PRINTED:
        sh = S0[n]['shape']
        cq.exporters.export(cq.Workplane(obj=sh), OUT('step/parts/%s.step' % n))
        m = K.tess(sh, 0.02, 0.1)
        write_stl(m, OUT('stl/%s.stl' % n))
        summary[n] = dict(volume_mm3=sh.Volume(), area_mm2=sh.Area(), est_mass_kg=est_mass(sh, MATERIAL[n]), material=MATERIAL[n], stl_triangles=int(len(m.faces)))
    # SMC gripper with the jaws at the closed position (vendor geometry, copy for the assembly only)
    g = K.smc_placed(0.0)
    cq.exporters.export(cq.Workplane(obj=cq.Compound.makeCompound([g['body'], g['jaw_left'], g['jaw_right']])), OUT('step/parts/smc_lehf20k2_48_mounted_closed.step'))
    json.dump(summary, open(OUT('data/part_summary.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- assemblies
    def assembly(o, fn, name):
        S = K.solids(o, with_hw=True)
        a = cq.Assembly(name=name)
        for n, d in S.items():
            col = COLORS.get(n, (0.55, 0.62, 0.74) if d['kind'] == 'hardware' else (0.6, 0.6, 0.6))
            a.add(d['shape'], name=n, color=cq.Color(*col))
        a.save(OUT(fn), 'STEP')
    assembly(P.JAW_STROKE, 'step/assembly_open.step', 'right_gripper_tool_open_56')
    assembly(4.0, 'step/assembly_closed16.step', 'right_gripper_tool_closed_on_16mm_box')

    # ---------------------------------------------------------------- mass properties / URDF inertials
    mp = mass_props(S0)
    json.dump(mp, open(OUT('data/mass.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- sim meshes
    manifest = dict(units='metres', frames='base_* in link6; finger_* in the finger-joint frame (origin = (0,0,%.4f) m of link6, axes parallel to link6, closed position)' % (P.Z_J / 1000),
                    mesh_path_convention='package://right_gripper_tool/meshes/{visual,collision}/<file>; xacro arg mesh_root overrides the prefix', visual=[], collision=[])
    L = K.local_parts()

    def vis(name, shape, frame, link, color):
        m = K.tess(shape, 0.1, 0.3)
        write_stl(m, OUT('meshes/visual/%s.stl' % name), 1e-3)
        manifest['visual'].append(dict(file='meshes/visual/%s.stl' % name, link=link, frame=frame, color=list(color), triangles=int(len(m.faces))))

    vis('base_flange_disc', S0['flange_disc']['shape'], 'link6', 'gripper_base', COLORS['flange_disc'])
    vis('base_gripper_plate', S0['gripper_plate']['shape'], 'link6', 'gripper_base', COLORS['gripper_plate'])
    vis('base_top_stop', S0['top_stop']['shape'], 'link6', 'gripper_base', COLORS['top_stop'])
    vis('base_smc_body', S0['smc_body']['shape'], 'link6', 'gripper_base', COLORS['smc_body'])
    for side, sn in ((-1, 'left'), (1, 'right')):
        dz = -P.Z_J
        vis('finger_%s' % sn, K.move(L['finger_%s' % sn], 0, 0, 0), 'finger_%s' % sn, 'finger_%s' % sn, COLORS['finger_left'])
        vis('finger_pad_%s' % sn, L['pad_%s' % sn], 'finger_%s' % sn, 'finger_%s' % sn, COLORS['pad_left'])
        vis('finger_jaw_%s' % sn, K.move(S0['smc_jaw_%s' % sn]['shape'], 0, 0, dz), 'finger_%s' % sn, 'finger_%s' % sn, COLORS['smc_jaw_left'])

    def col(name, mesh_mm, link, frame):
        h = mesh_mm.convex_hull
        write_stl(h, OUT('meshes/collision/%s.stl' % name), 1e-3)
        manifest['collision'].append(dict(file='meshes/collision/%s.stl' % name, link=link, frame=frame, triangles=int(len(h.faces)), hull_volume_mm3=float(h.volume)))

    # base: disc, plate (convex), SMC body in 3 x-slabs
    col('base_flange_disc', K.tess(S0['flange_disc']['shape']), 'gripper_base', 'link6')
    col('base_gripper_plate', K.tess(S0['gripper_plate']['shape']), 'gripper_base', 'link6')
    ts = S0['top_stop']['shape']
    col('base_top_stop_bar', K.tess(ts.intersect(K.box(-5, 5, -40, 40, P.Z_STOP - P.STOP_H - 1, P.Z_STOP + 1))), 'gripper_base', 'link6')
    for nm, y0, y1 in (('neg', -40, -17.0), ('pos', 17.0, 40)):
        col('base_top_stop_%s' % nm, K.tess(ts.intersect(K.box(-30, 30, y0, y1, P.SIDE_Z[0] - 1, P.Z_STOP - P.STOP_H + 0.5))), 'gripper_base', 'link6')
    body = S0['smc_body']['shape']
    for i, (x0, x1) in enumerate(((-100, -53.9), (-53.9, 62.3), (62.3, 70))):
        piece = body.intersect(K.box(x0, x1, -30, 30, 0, 300))
        if piece.Volume() > 1.0:
            col('base_smc_body_%d' % i, K.tess(piece), 'gripper_base', 'link6')
    # fingers: foot, column, block (pocket ignored), pad, jaw (2 pieces)
    for side, sn in ((-1, 'left'), (1, 'right')):
        f = L['finger_%s' % sn]
        regions = {'foot': K.box(-60, 60, -40, 40, -1, P.FOOT_T), 'column': K.box(-60, 60, -40, 40, P.FOOT_T, P.Z_REG), 'block': K.box(-60, 60, -40, 40, P.Z_REG + 3.0, 100)}
        for rn, rb in regions.items():
            col('finger_%s_%s' % (sn, rn), K.tess(f.intersect(rb)), 'finger_%s' % sn, 'finger_%s' % sn)
        col('finger_pad_%s' % sn, K.tess(L['pad_%s' % sn]), 'finger_%s' % sn, 'finger_%s' % sn)
        jaw = K.move(S0['smc_jaw_%s' % sn]['shape'], 0, 0, -P.Z_J)
        zs_jaw_split = 29.9 + P.GR_Z0 - P.Z_J
        col('finger_jaw_%s_top' % sn, K.tess(jaw.intersect(K.box(-80, 80, -30, 30, zs_jaw_split, 20))), 'finger_%s' % sn, 'finger_%s' % sn)
        col('finger_jaw_%s_low' % sn, K.tess(jaw.intersect(K.box(-80, 80, -30, 30, -30, zs_jaw_split))), 'finger_%s' % sn, 'finger_%s' % sn)
    json.dump(manifest, open(OUT('meshes/manifest.json'), 'w'), indent=1)
    OUT_manifest = OUT('manifest.json')
    json.dump(manifest, open(OUT_manifest, 'w'), indent=1)
    write_xacro(mp)
    print('build done; z_J = %.2f, L = %.2f' % (P.Z_J, P.L_PAD))


def combine(items):
    """items: list of (mass, com, I_about_com 3x3) in the same frame -> (mass, com, I about combined com)"""
    M = sum(i[0] for i in items)
    c = sum(i[0] * np.asarray(i[1]) for i in items) / M
    I = np.zeros((3, 3))
    for m, ci, Ii in items:
        d = np.asarray(ci) - c
        I += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, c, I


def mass_props(S0):
    """masses (printed estimate, catalog gripper, hardware list) -> link-level mass, CoG, inertia (kg, mm, kg mm2 in link6 axes; finger links also in their own frame)"""
    jaw_v = S0['smc_jaw_left']['shape'].Volume()
    body_v = S0['smc_body']['shape'].Volume()
    rho_g = P.GRIPPER_MASS / (body_v + 2 * jaw_v)
    items = {}
    for n, d in S0.items():
        sh = d['shape']
        if d['kind'] == 'printed':
            m = est_mass(sh, MATERIAL[n])
        elif d['kind'] == 'purchased':
            m = rho_g * sh.Volume()
        else:
            m = d['mass']
        _, c, I = K.props(sh, 1.0)
        V = sh.Volume()
        items[n] = (m, c, I / V * m)           # inertia per unit density-volume scaled to the mass
    # inserts as point masses (heat-set brass): 4 x M4 (0.4 g) in the disc, 4 x M3 (0.2 g) in the fingers
    ins = {'flange_disc': [(0.0004, np.array([x, y, P.Z_DISC_TOP - 2.25])) for x, y in P.M4_XY]}
    for sn, sd in (('left', -1), ('right', 1)):
        ins['finger_' + sn] = [(0.0002, np.array([sd * (P.FRAME_FACE + P.POCKET_DEPTH + 2.5), 0.0, P.Z_J + z])) for z in P.RET_Z]
    hwn = [n for n, d in S0.items() if d['kind'] == 'hardware']
    is_f = lambda n, sn: n.endswith('_' + sn) or ('_' + sn + '_') in n
    base_names = ['flange_disc', 'gripper_plate', 'top_stop', 'smc_body'] + [n for n in hwn if not (is_f(n, 'left') or is_f(n, 'right'))]
    finger_names = {sn: ['finger_' + sn, 'pad_' + sn, 'smc_jaw_' + sn] + [n for n in hwn if is_f(n, sn)] for sn in ('left', 'right')}

    def link(names):
        its = [items[n] for n in names]
        for n in names:
            for (m, p) in ins.get(n, []):
                its.append((m, p, np.zeros((3, 3))))
        return combine(its)

    mb, cb, Ib = link(base_names)
    out = dict(rho_gripper_kg_per_mm3=rho_g, parts={n: dict(mass_kg=v[0], com_link6_mm=[float(x) for x in v[1]]) for n, v in items.items()})
    out['base'] = dict(mass_kg=mb, com_link6_mm=[float(x) for x in cb], inertia_about_com_kgmm2=Ib.tolist())
    tot_items = [(mb, cb, Ib)]
    for sn in ('left', 'right'):
        names = [n for n in finger_names[sn] if n in items]
        names = sorted(set(names))
        mf, cf, If = link(names)
        out['finger_' + sn] = dict(mass_kg=mf, com_link6_mm=[float(x) for x in cf], com_finger_frame_mm=[float(cf[0]), float(cf[1]), float(cf[2] - P.Z_J)],
                                   inertia_about_com_kgmm2=If.tolist(), parts=names)
        tot_items.append((mf, cf, If))
    mt, ct, It = combine(tot_items)
    out['total_closed'] = dict(mass_kg=mt, com_link6_mm=[float(x) for x in ct], inertia_about_com_kgmm2=It.tolist())
    return out


def fmt(x):
    return ('%.9g' % x)


def write_xacro(mp):
    def inertial2(d, origin_mm):
        I = np.array(d['inertia_about_com_kgmm2']) * 1e-6
        return ('    <inertial>\n      <origin xyz="%s" rpy="0 0 0"/>\n      <mass value="%s"/>\n'
                '      <inertia ixx="%s" ixy="%s" ixz="%s" iyy="%s" iyz="%s" izz="%s"/>\n    </inertial>'
                % (' '.join(fmt(v * 1e-3) for v in origin_mm), fmt(d['mass_kg']), fmt(I[0, 0]), fmt(I[0, 1]), fmt(I[0, 2]), fmt(I[1, 1]), fmt(I[1, 2]), fmt(I[2, 2])))
    man = json.load(open(OUT('meshes/manifest.json')))

    def geoms(link):
        s = ''
        for v in man['visual']:
            if v['link'] == link:
                nm = os.path.basename(v['file'])[:-4]
                s += ('    <visual>\n      <geometry><mesh filename="${mesh_root}/%s"/></geometry>\n      <material name="${prefix}%s_mat"/>\n    </visual>\n' % (v['file'], 'pad' if 'pad' in nm else ('jaw' if 'jaw' in nm or 'smc' in nm else ('metal' if 'disc' in nm or 'plate' in nm or 'stop' in nm else 'finger'))))
        for c in man['collision']:
            if c['link'] == link:
                s += '    <collision>\n      <geometry><mesh filename="${mesh_root}/%s"/></geometry>\n    </collision>\n' % c['file']
        return s
    x = '<?xml version="1.0"?>\n'
    x += '<!-- Right-arm gripper tool: SMC LEHF20K2-48 in-line on the ReBeL flange + custom fingers.  GENERATED by build.py - edit params.py / parts.py instead.\n'
    x += '     Macro right_gripper_tool(prefix, parent): fixed base link at identity to `parent` (= link6), 2 prismatic finger joints (0..0.024 m each, mimic), gripper_tcp, top_stop_frame.\n'
    x += '     Mesh path convention: ${mesh_root}/meshes/{visual,collision}/*.stl, default mesh_root = package://right_gripper_tool (the folder is not a ROS package: pass mesh_root:=file:///abs/path/to/gripper_tool\n'
    x += '     or install it under a package of that name).  Meshes are in metres; base_* are expressed in link6, finger_* in the finger joint frame (origin (0,0,z_J), axes parallel to link6).\n'
    x += '     Masses are estimates (printed parts: 1.6 mm shell + 25 %% infill; gripper 0.75 kg catalog).  Finger effort 28 N = LEHF20 max gripping force per finger. -->\n'
    x += '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
    x += '  <xacro:macro name="right_gripper_tool" params="prefix parent mesh_root:=package://right_gripper_tool">\n'
    for nm, rgba in (('metal', '0.93 0.55 0.12 1'), ('finger', '0.15 0.55 0.85 1'), ('pad', '0.1 0.1 0.1 1'), ('jaw', '0.6 0.62 0.66 1')):
        x += '    <material name="${prefix}%s_mat"><color rgba="%s"/></material>\n' % (nm, rgba)
    x += '    <link name="${prefix}gripper_base">\n' + inertial2(mp['base'], mp['base']['com_link6_mm']) + '\n' + geoms('gripper_base') + '    </link>\n'
    x += '    <joint name="${prefix}gripper_base_joint" type="fixed">\n      <parent link="${parent}"/>\n      <child link="${prefix}gripper_base"/>\n      <origin xyz="0 0 0" rpy="0 0 0"/>\n    </joint>\n'
    zj = P.Z_J * 1e-3
    for sn, ax in (('left', '-1 0 0'), ('right', '1 0 0')):
        d = mp['finger_' + sn]
        x += '    <link name="${prefix}finger_%s">\n' % sn + inertial2(d, d['com_finger_frame_mm']) + '\n' + geoms('finger_' + sn) + '    </link>\n'
        x += '    <joint name="${prefix}finger_%s_joint" type="prismatic">\n      <parent link="${prefix}gripper_base"/>\n      <child link="${prefix}finger_%s"/>\n' % (sn, sn)
        x += '      <origin xyz="0 0 %s" rpy="0 0 0"/>\n      <axis xyz="%s"/>\n      <limit lower="0" upper="0.024" effort="28" velocity="0.05"/>\n' % (fmt(zj), ax)
        if sn == 'right':
            x += '      <mimic joint="${prefix}finger_left_joint" multiplier="1" offset="0"/>\n'
        x += '    </joint>\n'
    x += '    <!-- gripper_tcp: midpoint between the pad centres (z = approach, x = stroke axis) -->\n'
    x += '    <link name="${prefix}gripper_tcp"/>\n'
    x += '    <joint name="${prefix}gripper_tcp_joint" type="fixed">\n      <parent link="${prefix}gripper_base"/>\n      <child link="${prefix}gripper_tcp"/>\n      <origin xyz="0 0 %s" rpy="0 0 0"/>\n    </joint>\n' % fmt((P.Z_J + P.L_PAD) * 1e-3)
    x += '    <!-- top_stop_frame: centre of the fixed top-stop face (box top edge registration plane, z = %.4f m); fixed to the base, does not move with the jaws -->\n' % (P.Z_STOP * 1e-3)
    x += '    <link name="${prefix}top_stop_frame"/>\n'
    x += '    <joint name="${prefix}top_stop_frame_joint" type="fixed">\n      <parent link="${prefix}gripper_base"/>\n      <child link="${prefix}top_stop_frame"/>\n      <origin xyz="0 0 %s" rpy="0 0 0"/>\n    </joint>\n' % fmt(P.Z_STOP * 1e-3)
    x += '  </xacro:macro>\n</robot>\n'
    open(OUT('urdf/right_gripper_tool.urdf.xacro'), 'w').write(x)
    t = '<?xml version="1.0"?>\n<robot name="right_gripper_tool_test" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
    t += '  <xacro:arg name="mesh_root" default="file://%s"/>\n' % P.HERE
    t += '  <xacro:include filename="right_gripper_tool.urdf.xacro"/>\n  <link name="link6"/>\n'
    t += '  <xacro:right_gripper_tool prefix="right_" parent="link6" mesh_root="$(arg mesh_root)"/>\n</robot>\n'
    open(OUT('urdf/test_right_gripper_tool.urdf.xacro'), 'w').write(t)


if __name__ == '__main__':
    main()
