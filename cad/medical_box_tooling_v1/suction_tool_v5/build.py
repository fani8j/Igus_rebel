"""Build everything (v5): step/parts/*.step, step/assembly.step, stl/*.stl (printable), meshes/visual (m), meshes/collision (m, convex),
urdf/left_suction_tool.urdf.xacro, data/mass.json, data/printed_parts.json.      python build.py [--no-collision]"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, trimesh
import cadquery as cq
import params as P
import parts as pa
from meshlib import to_mesh
import massprops

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('step/parts', 'stl', 'meshes/visual', 'meshes/collision', 'urdf', 'data', 'renders'):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
COL = {'printed': (0.93, 0.55, 0.12), 'purchased': (0.55, 0.62, 0.74)}
SPECIAL = {'camera_gemini305': (0.15, 0.65, 0.3), 'collector': (0.8, 0.8, 0.85), 'pad_L': (0.2, 0.2, 0.25), 'pad_M': (0.2, 0.2, 0.25), 'pad_R': (0.2, 0.2, 0.25),
           'elbow_L': (0.55, 0.8, 0.95), 'elbow_M': (0.55, 0.8, 0.95), 'elbow_R': (0.55, 0.8, 0.95)}


def rm(path, ext):
    for f in os.listdir(path):
        if f.endswith(ext):
            os.remove(os.path.join(path, f))


def export_cad(parts):
    rm(os.path.join(HERE, 'step/parts'), '.step'); rm(os.path.join(HERE, 'stl'), '.stl')
    asm = cq.Assembly(name='left_suction_tool')
    summary = []
    for d in parts:
        nm, sh = d['name'], d['shape']
        cq.exporters.export(sh, os.path.join(HERE, 'step/parts/%s.step' % nm))
        asm.add(sh, name=nm, color=cq.Color(*SPECIAL.get(nm, COL[d['kind']])))
        if d['kind'] == 'printed':
            m = to_mesh(sh, tol=0.03, ang=0.1)
            if not m.is_watertight:
                trimesh.repair.fill_holes(m); trimesh.repair.fix_normals(m)
            m.export(os.path.join(HERE, 'stl/%s.stl' % nm))
            bb = sh.BoundingBox()
            summary.append(dict(part=nm, volume_cm3=sh.Volume() / 1000, mass_g=massprops.body_mass(d) * 1000, bbox_mm=[bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax],
                                watertight=bool(m.is_watertight), orient=d.get('orient')))
    asm.save(os.path.join(HERE, 'step/assembly.step'), 'STEP')
    json.dump(summary, open(os.path.join(HERE, 'data/printed_parts.json'), 'w'), indent=1)


def export_visual(parts):
    rm(os.path.join(HERE, 'meshes/visual'), '.stl')
    names = []
    for d in parts:
        m = to_mesh(d['shape'], tol=0.25, ang=0.35)
        m.apply_scale(0.001)
        m.export(os.path.join(HERE, 'meshes/visual/%s.stl' % d['name'])); names.append(d['name'])
    return names


def export_collision(parts, max_hulls):
    import coacd
    coacd.set_log_level('error')
    rm(os.path.join(HERE, 'meshes/collision'), '.stl')
    man = dict(frame='link6 (all pieces rigid in the link6 frame)', units='metres', note='convex pieces; the tubes (flexible) are excluded; camera = hull envelope', pieces=[])
    tot_h = tot_v = 0.0
    for d in parts:
        if d['name'].startswith(('tube_', 'trunk_')):
            continue
        m = to_mesh(d['shape'], tol=0.15, ang=0.3)
        nmax = max_hulls.get(d['name'], 1)
        pieces = []
        if nmax > 1:
            res = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.06, max_convex_hull=nmax, preprocess_mode='auto', seed=1, merge=True)
            pieces = [trimesh.convex.convex_hull(trimesh.Trimesh(v, f)) for v, f in res]
        else:
            pieces = [trimesh.convex.convex_hull(m)]
        hv = 0.0
        for i, h in enumerate(pieces):
            h = h.copy(); h.apply_scale(0.001)
            fn = 'meshes/collision/%s_%02d.stl' % (d['name'], i); h.export(os.path.join(HERE, fn))
            man['pieces'].append(dict(file=fn, source=d['name'], volume_mm3=float(h.volume * 1e9), faces=int(len(h.faces)))); hv += h.volume * 1e9
        tot_h += hv; tot_v += d['shape'].Volume()
        man.setdefault('hull_over_part_volume', {})[d['name']] = round(hv / d['shape'].Volume(), 3)
    man['n_pieces'] = len(man['pieces']); man['total_hull_over_total_volume'] = round(tot_h / tot_v, 3)
    json.dump(man, open(os.path.join(HERE, 'meshes/collision/manifest.json'), 'w'), indent=1)
    return man


MAX_HULLS = {'collector': 3, 'tool_body': 24, 'pad_L': 4, 'pad_M': 4, 'pad_R': 4, 'elbow_L': 2, 'elbow_M': 2, 'elbow_R': 2}


def mat_to_rpy_xyz(T):
    R = T[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        r = np.arctan2(R[2, 1], R[2, 2]); p = np.arctan2(-R[2, 0], sy); y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1]); p = np.arctan2(-R[2, 0], sy); y = 0.0
    return 'xyz="%.6f %.6f %.6f" rpy="%.8f %.8f %.8f"' % (T[0, 3] / 1000, T[1, 3] / 1000, T[2, 3] / 1000, r, p, y)


def write_xacro(parts, vis_names, man, mp):
    mesh_dir = 'file://' + os.path.join(HERE, 'meshes')
    I = mp['inertia'] * 1e-6          # kg mm2 -> kg m2
    com = mp['com'] / 1000
    T_id = np.eye(4)
    T_opt = P.T_CAM.copy(); T_opt[:3, 0] = -P.T_CAM[:3, 0]; T_opt[:3, 1] = -P.T_CAM[:3, 1]      # ROS optical frame: x right, y down, z forward (x_cam = left seen from behind)
    frames = [('suction_tcp', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, P.LIP_Z], [0, 0, 0, 1.0]])),
              ('suction_tcp_compressed', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, P.LIP_Z - 5.0], [0, 0, 0, 1.0]])),
              ('wrist_camera_mount', P.T_CAM), ('wrist_camera_color_optical_frame', T_opt)]
    for i, x in enumerate(P.PAD_X):        # Isaac Sim Surface Gripper attachment points: pad lip centre, +z out of the lip (pad_1 = x -60, pad_2 = 0, pad_3 = +60)
        frames.append(('pad_%d_suction_frame' % (i + 1), np.array([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, P.LIP_Z], [0, 0, 0, 1.0]])))
    L = []
    L.append('<?xml version="1.0"?>')
    L.append('<!-- Generated by build.py (suction_tool_v5). Left-arm suction end effector (one camera). Units SI (metres); frame = link6.')
    L.append('     Mesh paths are absolute file:// URIs (macro param mesh_dir; default = the meshes/ folder of this checkout). -->')
    L.append('<robot xmlns:xacro="http://www.ros.org/wiki/xacro">')
    L.append('  <xacro:macro name="left_suction_tool" params="prefix parent mesh_dir:=%s">' % mesh_dir)
    L.append('    <link name="${prefix}suction_tool">')
    L.append('      <inertial>')
    L.append('        <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % tuple(com))
    L.append('        <mass value="%.5f"/>' % mp['mass'])
    L.append('        <inertia ixx="%.6e" ixy="%.6e" ixz="%.6e" iyy="%.6e" iyz="%.6e" izz="%.6e"/>' % (I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]))
    L.append('      </inertial>')
    for n in vis_names:
        L.append('      <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="${mesh_dir}/visual/%s.stl"/></geometry></visual>' % n)
    for p in man['pieces']:
        L.append('      <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="${mesh_dir}/%s"/></geometry></collision>' % p['file'].replace('meshes/', ''))
    L.append('    </link>')
    L.append('    <joint name="${prefix}suction_tool_joint" type="fixed">')
    L.append('      <parent link="${parent}"/><child link="${prefix}suction_tool"/><origin xyz="0 0 0" rpy="0 0 0"/>')
    L.append('    </joint>')
    for n, T in frames:
        L.append('    <link name="${prefix}%s"/>' % n)
        L.append('    <joint name="${prefix}%s_joint" type="fixed"><parent link="${prefix}suction_tool"/><child link="${prefix}%s"/><origin %s/></joint>' % (n, n, mat_to_rpy_xyz(T)))
    L.append('  </xacro:macro>')
    L.append('</robot>')
    open(os.path.join(HERE, 'urdf/left_suction_tool.urdf.xacro'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    parts = pa.build_parts()
    export_cad(parts)
    vis = export_visual(parts)
    man = export_collision(parts, MAX_HULLS) if '--no-collision' not in sys.argv else json.load(open(os.path.join(HERE, 'meshes/collision/manifest.json')))
    mp = massprops.assembly_props(parts)
    json.dump(dict(mass_kg=mp['mass'], com_mm=mp['com'].tolist(), inertia_kg_mm2=mp['inertia'].tolist(), bodies=mp['bodies']), open(os.path.join(HERE, 'data/mass.json'), 'w'), indent=1)
    write_xacro(parts, vis, man, mp)
    print('mass %.3f kg  CoG %s  collision pieces %d' % (mp['mass'], np.round(mp['com'], 1), man['n_pieces']))
