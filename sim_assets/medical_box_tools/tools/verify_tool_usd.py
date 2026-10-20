#!/usr/bin/env python3
"""Verify the three USD files against the CAD tool folders and handoff.py; write data/verify.json and usd/previews/*.png.

Independent of build_tool_usd.py: URDF values come from the *expanded* xacro (xacro CLI, ROS_DOMAIN_ID=178, LOCALHOST_ONLY=1;
falls back to the raw xacro macro body if xacro is not available).
  "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/verify_tool_usd.py
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = OUT.parent.parent
CAD = REPO / 'cad' / 'medical_box_tooling_v1'
USD = OUT / 'usd'
SUCTION_NAME = os.environ.get('SUCTION_TOOL', 'suction_tool_v5')      # must match build_tool_usd.py
SCRATCH = Path('/media/farhan/My Passport/igus_nvidia/scratch_usd')
sys.path.insert(0, str(CAD / 'common'))

POS_TOL = 1e-4          # 0.1 mm
ANG_TOL = 0.01          # deg
checks = []


def check(name, ok, detail=''):
    checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
    print(('PASS ' if ok else 'FAIL ') + name + ('  ' + str(detail) if detail else ''))
    return ok


# ------------------------------------------------------------------ expanded URDF
def expand(xacro_file, macro):
    src = ('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">'
           f'<xacro:include filename="{xacro_file}"/><link name="link6"/>'
           f'<xacro:{macro} prefix="" parent="link6"/></robot>')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    w = SCRATCH / f'wrap_{macro}.xacro'
    w.write_text(src)
    env = dict(os.environ, ROS_DOMAIN_ID='178', ROS_LOCALHOST_ONLY='1')
    try:
        r = subprocess.run(['bash', '-c', f'source /opt/ros/humble/setup.bash && xacro "{w}"'], capture_output=True, text=True, env=env, timeout=120)
    except Exception as e:
        print('xacro unavailable:', e)
        return None
    if r.returncode != 0:
        print('xacro failed, falling back to raw macro parse:', r.stderr[:200])
        return None
    return ET.fromstring(r.stdout)


def rot_from_rpy(rpy):
    """URDF fixed-axis rpy, R = Rz(y) Ry(p) Rx(r) (column convention), as a Gf.Rotation (row-vector: x first, then y, then z)."""
    return (Gf.Rotation(Gf.Vec3d(1, 0, 0), np.degrees(rpy[0])) * Gf.Rotation(Gf.Vec3d(0, 1, 0), np.degrees(rpy[1]))
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), np.degrees(rpy[2])))


def urdf_info(xroot):
    joints, links = {}, {}
    for j in xroot.iter('joint'):
        o = j.find('origin')
        joints[j.get('name')] = dict(child=j.find('child').get('link'), parent=j.find('parent').get('link'), xyz=np.array(o.get('xyz').split(), float),
                                     rpy=np.array(o.get('rpy').split(), float),
                                     axis=np.array(j.find('axis').get('xyz').split(), float) if j.find('axis') is not None else None,
                                     limit=j.find('limit').attrib if j.find('limit') is not None else None)
    for l in xroot.iter('link'):
        d = dict(n_visual=len(l.findall('visual')), n_collision=len(l.findall('collision')))
        i = l.find('inertial')
        if i is not None:
            ii = i.find('inertia')
            a = {k: float(ii.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')}
            d.update(mass=float(i.find('mass').get('value')), com=np.array(i.find('origin').get('xyz').split(), float),
                     I=np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]]))
        links[l.get('name')] = d
    return joints, links


def rel_pose(stage, root_path, prim_path):
    xc = UsdGeom.XformCache()
    return xc.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path)) * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path)).GetInverse()


def pose_err(T, xyz, rot):
    dp = np.linalg.norm(np.array(T.ExtractTranslation()) - xyz)
    da = (T.ExtractRotation() * rot.GetInverse()).GetAngle()
    return dp, da


def inertia_from_usd(prim):
    m = UsdPhysics.MassAPI(prim)
    d = np.array(m.GetDiagonalInertiaAttr().Get(), float)
    q = m.GetPrincipalAxesAttr().Get()
    rot = Gf.Rotation(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
    R = np.array(Gf.Matrix3d(rot)).T            # USD matrices are row-vector; R is the column-convention rotation
    return R @ np.diag(d) @ R.T, m.GetMassAttr().Get(), np.array(m.GetCenterOfMassAttr().Get(), float)


def schemas(prim):
    """All apiSchemas tokens authored on the prim (also unregistered ones such as the Isaac / PhysX schemas, which pxr's GetAppliedSchemas() hides)."""
    out = []
    for spec in prim.GetPrimStack():
        lo = spec.GetInfo('apiSchemas')
        if lo:
            out += list(lo.explicitItems) + list(lo.prependedItems) + list(lo.appendedItems)
    return out


def all_prims_valid(stage, tag):
    bad = [p.GetPath().pathString for p in stage.TraverseAll() if not p.IsValid()]
    check(f'{tag}: all prims valid', not bad, bad[:3])


# ------------------------------------------------------------------ geometry for previews
def collect(stage, root, purposes, include_cubes=True):
    import trimesh
    out = []
    xc = UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdGeom.Mesh):
            if UsdGeom.Imageable(p).GetPurposeAttr().Get() not in purposes:
                continue
            m = UsdGeom.Mesh(p)
            pts = np.array(m.GetPointsAttr().Get(), float)
            idx = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1)
            cnt = np.array(m.GetFaceVertexCountsAttr().Get())
            if (cnt == 3).all():
                f = idx.reshape(-1, 3)
            else:           # quads
                q = idx.reshape(-1, 4)
                f = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            T = np.array(xc.GetLocalToWorldTransform(p)).T
            pw = pts @ T[:3, :3].T + T[:3, 3]
            col = m.GetDisplayColorAttr().Get()
            out.append((pw[f], tuple(col[0]) if col else (0.7, 0.7, 0.7), p.GetPath().pathString))
        elif include_cubes and p.IsA(UsdGeom.Cube):
            T = np.array(xc.GetLocalToWorldTransform(p)).T
            b = trimesh.creation.box(extents=[1, 1, 1])
            pw = np.asarray(b.vertices) @ T[:3, :3].T + T[:3, 3]
            col = UsdGeom.Cube(p).GetDisplayColorAttr().Get()
            out.append((pw[np.asarray(b.faces)], tuple(col[0]) if col else (0.7, 0.7, 0.7), p.GetPath().pathString))
    return out


def render(groups, fname, elev, azim, lim=None, title='', size=(7, 6)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=size, dpi=110)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    allp = []
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    L = np.array([0.4, -0.5, 0.8]); L /= np.linalg.norm(L)
    for tris, col, _ in groups:
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nl = np.linalg.norm(n, axis=1, keepdims=True); nl[nl == 0] = 1
        n = n / nl
        shade = 0.35 + 0.65 * np.abs(n @ (0.6 * view + 0.4 * L))
        fc = np.clip(np.array(col)[None, :] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors='none', linewidths=0))
        allp.append(tris.reshape(-1, 3))
    P = np.vstack(allp)
    lo, hi = (P.min(0), P.max(0)) if lim is None else (np.array(lim[0]), np.array(lim[1]))
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ------------------------------------------------------------------ checks
def verify_suction():
    tag = 'suction'
    stage = Usd.Stage.Open(str(USD / 'left_suction_tool.usda'))
    root = stage.GetDefaultPrim()
    R = root.GetPath().pathString
    check(f'{tag}: defaultPrim /left_suction_tool, Z-up, m, kg', R == '/left_suction_tool' and UsdGeom.GetStageUpAxis(stage) == 'Z'
          and UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0)
    all_prims_valid(stage, tag)
    check(f'{tag}: kind component + ArticulationRootAPI on the root (no body on the root)', Usd.ModelAPI(root).GetKind() == 'component'
          and root.HasAPI(UsdPhysics.ArticulationRootAPI) and not root.HasAPI(UsdPhysics.RigidBodyAPI))
    xr = expand(CAD / SUCTION_NAME / 'urdf/left_suction_tool.urdf.xacro', 'left_suction_tool')
    check(f'{tag}: expanded URDF via xacro', xr is not None)
    if xr is None:
        xr = ET.parse(CAD / SUCTION_NAME / 'urdf/left_suction_tool.urdf.xacro').getroot()
    joints, links = urdf_info(xr)
    man = json.load(open(CAD / SUCTION_NAME / 'meshes/collision/manifest.json'))
    vis_stl = len(list((CAD / SUCTION_NAME / 'meshes/visual').glob('*.stl')))
    nv = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/visuals/' in p.GetPath().pathString)
    nc = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/collisions/' in p.GetPath().pathString]
    L = links['suction_tool']
    check(f'{tag}: visual meshes = URDF visuals = visual STLs (+3: each pad STL is cut into buffer body and sliding head)', nv == L['n_visual'] + 3 == vis_stl + 3,
          f'usd {nv} urdf {L["n_visual"]} stl {vis_stl}')
    check(f'{tag}: collision meshes = URDF collisions = manifest pieces', len(nc) == L['n_collision'] == man['n_pieces'] == len(man['pieces']),
          f'usd {len(nc)} urdf {L["n_collision"]} manifest {man["n_pieces"]}')
    check(f'{tag}: all collision meshes convexHull + guide', all(
        p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == 'convexHull'
        and UsdGeom.Imageable(p).GetPurposeAttr().Get() == 'guide' for p in nc))
    check(f'{tag}: every visual mesh has a material binding', all(
        UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPrim().IsValid() for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and '/visuals/' in p.GetPath().pathString))
    frame_children = [j['child'] for j in joints.values() if j['parent'] == 'suction_tool']
    required = ['suction_tcp', 'suction_tcp_compressed', 'wrist_camera_mount', 'wrist_camera_color_optical_frame', 'pad_1_suction_frame', 'pad_2_suction_frame', 'pad_3_suction_frame']
    check(f'{tag}: URDF has the required frames, no scanner frame', all(r in frame_children for r in required) and 'scanner_optical_frame' not in frame_children, frame_children)
    usd_frames = [c.GetName() for c in root.GetChildren() if c.IsA(UsdGeom.Xform) and c.GetName() not in ('base', 'pad_1', 'pad_2', 'pad_3')]
    check(f'{tag}: USD frame prims = URDF frames', sorted(usd_frames) == sorted(frame_children), usd_frames)
    check(f'{tag}: no scanner / decoder material or mesh in the USD', not any(('scanner' in p.GetPath().pathString.lower() or 'decoder' in p.GetPath().pathString.lower()) for p in stage.Traverse()))
    for jn, j in joints.items():
        if j['parent'] == 'suction_tool':
            dp, da = pose_err(rel_pose(stage, R, f'{R}/{j["child"]}'), j['xyz'], rot_from_rpy(j['rpy']))
            check(f'{tag}: frame {j["child"]} vs URDF', dp < POS_TOL and da < ANG_TOL, f'{dp * 1e3:.5f} mm, {da:.5f} deg')
    # pad frames: lip centre of the pad collision mesh, +Z = tool +Z (out of the lip), pad pitch 60 mm
    for i, pn in enumerate(('pad_L', 'pad_M', 'pad_R')):
        tris = [t for t, _, pth in collect(stage, R, {'guide'}, False) if f'/collisions/{pn}_' in pth]
        pts = np.vstack([t.reshape(-1, 3) for t in tris])
        lip_z = pts[:, 2].max()
        top = pts[np.abs(pts[:, 2] - lip_z) < 1e-6]
        c = (top[:, 0].max() + top[:, 0].min()) / 2
        T = rel_pose(stage, R, f'{R}/pad_{i + 1}_suction_frame')
        t = np.array(T.ExtractTranslation()); Rm = np.array(T.ExtractRotationMatrix())
        check(f'{tag}: pad_{i + 1}_suction_frame at the {pn} lip centre, +Z out of the lip',
              abs(t[0] - c) < 1e-4 and abs(t[1]) < 1e-4 and abs(t[2] - lip_z) < 1e-4 and np.allclose(Rm[2], (0, 0, 1), atol=1e-6),
              f'frame {np.round(t * 1e3, 3).tolist()} mm, lip centre ({c * 1e3:.3f}, 0, {lip_z * 1e3:.3f}) mm')
    bodies = [stage.GetPrimAtPath(f'{R}/{n}') for n in ('base', 'pad_1', 'pad_2', 'pad_3')]
    check(f'{tag}: base + 3 pad heads are rigid bodies', all(b.HasAPI(UsdPhysics.RigidBodyAPI) for b in bodies))
    props = [inertia_from_usd(b) for b in bodies]
    m = sum(q[1] for q in props)
    com = sum(q[1] * q[2] for q in props) / m
    I = sum(q[0] + q[1] * ((q[2] - com) @ (q[2] - com) * np.eye(3) - np.outer(q[2] - com, q[2] - com)) for q in props)
    check(f'{tag}: total mass = URDF (base + 3 heads, CAD total preserved)', abs(m - L['mass']) < 1e-6 * L['mass'], f'{m} vs {L["mass"]}')
    check(f'{tag}: combined CoG = URDF', np.abs(com - L['com']).max() < 1e-7, f'{np.abs(com - L["com"]).max():.2e} m')
    rel = np.abs(I - L['I']).max() / np.abs(L['I']).max()
    check(f'{tag}: combined inertia tensor = URDF', rel < 1e-5, f'rel err {rel:.2e}')
    check(f'{tag}: pad head masses 3.3 g each (share of the 12 g ZP2 part)', all(abs(q[1] - 0.0033191) < 2e-6 for q in props[1:]), [round(q[1] * 1e3, 4) for q in props[1:]])
    # ---- pad buffers
    for i in (1, 2, 3):
        jp = stage.GetPrimAtPath(f'{R}/joints/pad_{i}_buffer_joint')
        j = UsdPhysics.PrismaticJoint(jp)
        d = UsdPhysics.DriveAPI.Get(jp, 'linear')
        pad_xyz = joints[f'pad_{i}_suction_frame_joint']['xyz']
        check(f'{tag}: pad_{i} buffer joint: prismatic Z, 0..-10 mm, anchor at the lip centre, base -> pad_{i}',
              j.GetAxisAttr().Get() == 'Z' and abs(j.GetLowerLimitAttr().Get() + 0.010) < 1e-9 and abs(j.GetUpperLimitAttr().Get()) < 1e-9
              and np.abs(np.array(j.GetLocalPos0Attr().Get()) - pad_xyz).max() < 1e-7 and np.abs(np.array(j.GetLocalPos1Attr().Get()) - pad_xyz).max() < 1e-7
              and [str(t) for t in j.GetBody0Rel().GetTargets()] == [f'{R}/base'] and [str(t) for t in j.GetBody1Rel().GetTargets()] == [f'{R}/pad_{i}'])
        k, tgt = d.GetStiffnessAttr().Get(), d.GetTargetPositionAttr().Get()
        check(f'{tag}: pad_{i} spring 200 N/m, preload K*target = 1.0 N extended / 3.0 N compressed', abs(k - 200.0) < 1e-6 and abs(k * tgt - 1.0) < 1e-6 and abs(k * (tgt + 0.010) - 3.0) < 1e-6,
              f'k={k} target={tgt} damping={d.GetDampingAttr().Get()} maxForce={d.GetMaxForceAttr().Get()}')
        check(f'{tag}: pad_{i} collision lip pieces on the sliding head, friction 0.5/0.5',
              all(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(stage.GetPrimAtPath(f'{R}/pad_{i}/collisions')) if p.IsA(UsdGeom.Mesh))
              and sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath(f'{R}/pad_{i}/collisions')) if p.IsA(UsdGeom.Mesh)) == 3)
    pm = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f'{R}/Looks/pad_physics'))
    check(f'{tag}: pad physics material 0.5 / 0.5', abs(pm.GetStaticFrictionAttr().Get() - 0.5) < 1e-9 and abs(pm.GetDynamicFrictionAttr().Get() - 0.5) < 1e-9)
    # ---- Surface Gripper (schema of isaacsim.robot.schema 5.1)
    sg = stage.GetPrimAtPath(f'{R}/surface_gripper')
    area, mu, kpa = sg.GetCustomDataByKey('pad_area_m2'), sg.GetCustomDataByKey('shear_mu'), sg.GetCustomDataByKey('vacuum_kpa')
    co, sh = sg.GetAttribute('isaac:coaxialForceLimit').Get(), sg.GetAttribute('isaac:shearForceLimit').Get()
    tg = [str(t) for t in sg.GetRelationship('isaac:attachmentPoints').GetTargets()]
    check(f'{tag}: IsaacSurfaceGripper prim, 3 attachment points, status Open', sg.GetTypeName() == 'IsaacSurfaceGripper' and len(tg) == 3
          and sg.GetAttribute('isaac:status').Get() == 'Open')
    fc, fs = sg.GetCustomDataByKey('cal_factor_coaxial'), sg.GetCustomDataByKey('cal_factor_shear')
    check(f'{tag}: physical force limits from one vacuum parameter ({kpa:.0f} kPa x 1.16 cm2 = {kpa * 1e3 * area:.2f} N coaxial, mu {mu} -> {mu * kpa * 1e3 * area:.2f} N shear); authored = physical x calibration factors ({fc}, {fs})',
          abs(co - fc * kpa * 1e3 * area) < 1e-4 and abs(sh - fs * mu * kpa * 1e3 * area) < 1e-4 and abs(area - 1.16e-4) < 1e-12 and fs == 2.2 and fc == 1.0, f'{co} {sh}')
    check(f'{tag}: maxGripDistance 3..5 mm, retry interval > 0', 0.003 <= sg.GetAttribute('isaac:maxGripDistance').Get() <= 0.005 and sg.GetAttribute('isaac:retryInterval').Get() > 0)
    for i, t in enumerate(tg, start=1):
        jp = stage.GetPrimAtPath(t)
        d6 = UsdPhysics.Joint(jp)
        pad_xyz = joints[f'pad_{i}_suction_frame_joint']['xyz']
        check(f'{tag}: attachment {i}: D6 on pad_{i} at the lip centre, forward axis Z, excluded from the articulation, enabled, no break force',
              jp.GetTypeName() == 'PhysicsJoint' and 'IsaacAttachmentPointAPI' in schemas(jp) and jp.GetAttribute('isaac:forwardAxis').Get() == 'Z'
              and [str(x) for x in d6.GetBody0Rel().GetTargets()] == [f'{R}/pad_{i}'] and np.abs(np.array(d6.GetLocalPos0Attr().Get()) - pad_xyz).max() < 1e-7
              and d6.GetExcludeFromArticulationAttr().Get() and d6.GetJointEnabledAttr().Get() and d6.GetBreakForceAttr().Get() > 1e30)
    import handoff
    tcp = rel_pose(stage, R, R + '/suction_tcp').ExtractTranslation()
    tcc = rel_pose(stage, R, R + '/suction_tcp_compressed').ExtractTranslation()
    check(f'{tag}: handoff.py SUCTION_TCP_Z / COMPRESSED_Z', abs(tcp[2] * 1e3 - handoff.SUCTION_TCP_Z) < 1e-4 and abs(tcc[2] * 1e3 - handoff.SUCTION_TCP_COMPRESSED_Z) < 1e-4)
    g = collect(stage, R, {'default'}, False)
    for nm, e, a in (('iso', 25, -55), ('side', 0, -90), ('top', 89, -90), ('front', 0, 0)):
        render(g, USD / 'previews' / f'left_suction_tool_{nm}.png', e, a, title=f'left_suction_tool visual ({nm}); link6 frame, z out of the flange')


def verify_gripper(fname='right_gripper_tool.usda', mimic=True):
    tag = 'gripper' if mimic else 'gripper(independent)'
    stage = Usd.Stage.Open(str(USD / fname))
    root = stage.GetDefaultPrim()
    R = root.GetPath().pathString
    check(f'{tag}: defaultPrim /right_gripper_tool, Z-up, m, kg', R == '/right_gripper_tool' and UsdGeom.GetStageUpAxis(stage) == 'Z'
          and UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0)
    all_prims_valid(stage, tag)
    check(f'{tag}: ArticulationRootAPI on root, kind component', root.HasAPI(UsdPhysics.ArticulationRootAPI) and Usd.ModelAPI(root).GetKind() == 'component')
    xr = expand(CAD / 'gripper_tool/urdf/right_gripper_tool.urdf.xacro', 'right_gripper_tool')
    check(f'{tag}: expanded URDF via xacro', xr is not None)
    if xr is None:
        xr = ET.parse(CAD / 'gripper_tool/urdf/right_gripper_tool.urdf.xacro').getroot()
    joints, links = urdf_info(xr)
    man = json.load(open(CAD / 'gripper_tool/meshes/manifest.json'))
    nv = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/visuals/' in p.GetPath().pathString)
    nc = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/collisions/' in p.GetPath().pathString]
    bl = ('gripper_base', 'finger_left', 'finger_right')
    tv = sum(links[k]['n_visual'] for k in bl)
    tc = sum(links[k]['n_collision'] for k in bl)
    check(f'{tag}: visual meshes = URDF = manifest', nv == tv == len(man['visual']), f'usd {nv} urdf {tv} manifest {len(man["visual"])}')
    check(f'{tag}: collision meshes = URDF = manifest', len(nc) == tc == len(man['collision']), f'usd {len(nc)} urdf {tc} manifest {len(man["collision"])}')
    check(f'{tag}: collision meshes convexHull + guide', all(
        p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == 'convexHull'
        and UsdGeom.Imageable(p).GetPurposeAttr().Get() == 'guide' for p in nc))
    for link, body in (('gripper_base', 'base'), ('finger_left', 'finger_left'), ('finger_right', 'finger_right')):
        p = stage.GetPrimAtPath(f'{R}/{body}')
        I, m, com = inertia_from_usd(p)
        L = links[link]
        check(f'{tag}: {body} rigid body + mass', p.HasAPI(UsdPhysics.RigidBodyAPI) and abs(m - L['mass']) < 1e-6 * L['mass'], f'{m} vs {L["mass"]}')
        check(f'{tag}: {body} CoG', np.abs(com - L['com']).max() < 1e-7)
        rel = np.abs(I - L['I']).max() / np.abs(L['I']).max()
        check(f'{tag}: {body} inertia', rel < 1e-5, f'rel err {rel:.2e}')
    for fr in ('gripper_tcp', 'top_stop_frame'):
        j = joints[fr + '_joint']
        dp, da = pose_err(rel_pose(stage, R, f'{R}/{fr}'), j['xyz'], rot_from_rpy(j['rpy']))
        check(f'{tag}: frame {fr} vs URDF', dp < POS_TOL and da < ANG_TOL, f'{dp * 1e3:.5f} mm, {da:.5f} deg')
    import handoff
    z = rel_pose(stage, R, R + '/gripper_tcp').ExtractTranslation()[2] * 1e3
    zs = rel_pose(stage, R, R + '/top_stop_frame').ExtractTranslation()[2] * 1e3
    check(f'{tag}: handoff.py GRIPPER_TCP_Z / STOP_Z', abs(z - handoff.GRIPPER_TCP_Z) < 1e-4 and abs(zs - handoff.GRIPPER_STOP_Z) < 1e-4)
    for side in ('left', 'right'):
        ju = joints[f'finger_{side}_joint']
        jp = stage.GetPrimAtPath(f'{R}/joints/finger_{side}_joint')
        j = UsdPhysics.PrismaticJoint(jp)
        rot0 = j.GetLocalRot0Attr().Get()
        ax_base = Gf.Rotation(Gf.Quatd(rot0.GetReal(), Gf.Vec3d(*rot0.GetImaginary()))).TransformDir(Gf.Vec3d(1, 0, 0))
        check(f'{tag}: {side} joint axis (X) in base frame = URDF axis', j.GetAxisAttr().Get() == 'X' and np.abs(np.array(ax_base) - ju['axis']).max() < 1e-9,
              f'{tuple(np.round(ax_base, 6))}')
        rot1 = j.GetLocalRot1Attr().Get()
        check(f'{tag}: {side} joint frames coincide at q=0 (rot0 == rot1)', np.allclose([rot0.GetReal(), *rot0.GetImaginary()], [rot1.GetReal(), *rot1.GetImaginary()]))
        check(f'{tag}: {side} joint limits {ju["limit"]["lower"]}..{ju["limit"]["upper"]} m',
              abs(j.GetLowerLimitAttr().Get() - float(ju['limit']['lower'])) < 1e-9 and abs(j.GetUpperLimitAttr().Get() - float(ju['limit']['upper'])) < 1e-7)
        check(f'{tag}: {side} joint anchor = URDF origin', np.abs(np.array(j.GetLocalPos0Attr().Get()) - ju['xyz']).max() < 1e-7)
        d = UsdPhysics.DriveAPI.Get(jp, 'linear')
        if mimic and side == 'right':
            check(f'{tag}: right finger = mimic of the left (gearing -1, offset 0), no own drive', 'PhysxMimicJointAPI:rotX' in schemas(jp)
                  and jp.GetAttribute('physxMimicJoint:rotX:gearing').Get() == -1.0 and jp.GetAttribute('physxMimicJoint:rotX:offset').Get() == 0.0
                  and [str(t) for t in jp.GetRelationship('physxMimicJoint:rotX:referenceJoint').GetTargets()] == [f'{R}/joints/finger_left_joint'] and not d)
        else:
            want = 24.3 * (2.0 if mimic else 1.0)
            check(f'{tag}: {side} drive maxForce {want:.1f} N (24.3 N per finger, SMC 11..28 N), k/c > 0', abs(d.GetMaxForceAttr().Get() - want) < 1e-4
                  and d.GetStiffnessAttr().Get() > 0 and d.GetDampingAttr().Get() > 0,
                  f'k={d.GetStiffnessAttr().Get()} c={d.GetDampingAttr().Get()} target={d.GetTargetPositionAttr().Get()}')
        check(f'{tag}: {side} no physxJoint:maxJointVelocity (it halves the drive force in Isaac Sim 5.1; speed is a target ramp)', not jp.HasAttribute('physxJoint:maxJointVelocity'))
        T = rel_pose(stage, R, f'{R}/finger_{side}').ExtractTranslation()
        exp = ju['xyz'] + ju['axis'] * float(ju['limit']['upper'])
        check(f'{tag}: {side} finger authored at open, consistent with state/target {float(ju["limit"]["upper"])}',
              np.abs(np.array(T) - exp).max() < 1e-7 and abs(jp.GetAttribute('state:linear:physics:position').Get() - float(ju['limit']['upper'])) < 1e-7
              and 'PhysicsJointStateAPI:linear' in schemas(jp)
              and (not d or abs(d.GetTargetPositionAttr().Get() - float(ju['limit']['upper'])) < 1e-7))

    def pad_inner(side):
        for tris, _, pth in collect(stage, f'{R}/finger_{side}', {'guide'}, False):
            if f'finger_pad_{side}' in pth:
                return tris[..., 0].min() if side == 'right' else tris[..., 0].max()
    gap = (pad_inner('right') - pad_inner('left')) * 1e3
    check(f'{tag}: pad gap at open = 56 mm (collision pads)', abs(gap - 56.0) < 0.05, f'{gap:.3f} mm')
    check(f'{tag}: fingers filtered against each other (CAD jaw hulls touch)', [str(t) for t in UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(f'{R}/finger_left')).GetFilteredPairsRel().GetTargets()] == [f'{R}/finger_right'])
    if not mimic:
        return
    g = collect(stage, R, {'default'}, False)
    for nm, e, a in (('iso', 20, -50), ('front', 0, -90), ('side', 0, 0)):
        render(g, USD / 'previews' / f'right_gripper_tool_open_{nm}.png', e, a, title=f'right_gripper_tool visual, open ({nm})')


def verify_scene():
    tag = 'scene'
    import handoff
    from params import PULL_OUT
    stage = Usd.Stage.Open(str(USD / 'shelf_handoff_scene.usda'))
    check(f'{tag}: defaultPrim /World, Z-up, m, kg', stage.GetDefaultPrim().GetPath().pathString == '/World' and UsdGeom.GetStageUpAxis(stage) == 'Z'
          and UsdGeom.GetStageMetersPerUnit(stage) == 1.0)
    all_prims_valid(stage, tag)
    ps = UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene'))
    check(f'{tag}: PhysicsScene gravity -Z 9.81', np.allclose(ps.GetGravityDirectionAttr().Get(), (0, 0, -1)) and abs(ps.GetGravityMagnitudeAttr().Get() - 9.81) < 1e-6)
    seq = list(stage.GetPrimAtPath('/World').GetCustomDataByKey('sequence'))
    cartons = list(stage.GetPrimAtPath('/World/Cartons').GetChildren())
    res = True
    for p in cartons:
        res &= any(c.IsA(UsdGeom.Mesh) for c in Usd.PrimRange(p)) and p.HasAPI(UsdPhysics.RigidBodyAPI)
    for n in ('left_suction_tool', 'right_gripper_tool'):
        res &= sum(1 for c in Usd.PrimRange(stage.GetPrimAtPath(f'/World/Tools/{n}')) if c.IsA(UsdGeom.Mesh)) > 20
    check(f'{tag}: all {len(cartons)} carton + 2 tool references resolve (meshes, rigid bodies present)', res and len(cartons) == len(seq) == 8)
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    boxes = []
    for p in cartons:
        bb = bbc.ComputeWorldBound(p).ComputeAlignedRange()
        boxes.append((np.array(bb.GetMin()), np.array(bb.GetMax())))
    zmins = [b[0][2] for b in boxes]
    check(f'{tag}: cartons sit on the shelf floor (bbox z min = 0 +-0.5 mm)', max(abs(z) for z in zmins) < 5e-4, f'max |zmin| {max(abs(z) for z in zmins) * 1e3:.4f} mm')
    fb = bbc.ComputeWorldBound(stage.GetPrimAtPath('/World/Shelf/floor_board')).ComputeAlignedRange()
    check(f'{tag}: shelf floor top at z = 0', abs(fb.GetMax()[2]) < 1e-9)
    ub = bbc.ComputeWorldBound(stage.GetPrimAtPath('/World/Shelf/upper_board')).ComputeAlignedRange()
    check(f'{tag}: upper board underside 300 mm above the floor', abs(ub.GetMin()[2] - 0.300) < 1e-9)
    worst = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ov = np.minimum(boxes[i][1], boxes[j][1]) - np.maximum(boxes[i][0], boxes[j][0])
            if (ov > 0).all():
                worst = max(worst, ov.min())
    check(f'{tag}: no carton-carton overlap > 0.1 mm', worst < 1e-4, f'max penetration {worst * 1e3:.5f} mm')
    gaps = [boxes[k + 1][0][0] - boxes[k][1][0] for k in range(len(boxes) - 1)]
    check(f'{tag}: packed with 0 mm gap', max(abs(g) for g in gaps) < 1e-4, f'gaps {np.round(np.array(gaps) * 1e3, 5).tolist()} mm')
    ti = stage.GetPrimAtPath('/World').GetCustomDataByKey('target_index')
    check(f'{tag}: target spine at y = -{PULL_OUT / 1000:.3f} m', abs(boxes[ti][0][1] + PULL_OUT / 1000) < 1e-4, f'{boxes[ti][0][1]:.6f}')
    check(f'{tag}: other spines at y = 0', all(abs(boxes[k][0][1]) < 1e-4 for k in range(len(boxes)) if k != ti))
    check(f'{tag}: target is box 3 and centred at x = 0', seq[ti] == 3 and abs((boxes[ti][0][0] + boxes[ti][1][0]) / 2) < 1e-6)
    bc = (boxes[ti][0] + boxes[ti][1]) / 2
    check(f'{tag}: target centre = handoff.box_pose(3)', np.abs(bc - handoff.box_pose(3)[:3, 3] / 1000).max() < 1e-4, f'{np.round(bc, 5).tolist()}')
    xc = UsdGeom.XformCache()
    Mt = np.array(xc.GetLocalToWorldTransform(cartons[ti])).T
    check(f'{tag}: carton +Y(spine) -> -Y_w, X(height) -> +Z_w, Z(thickness) -> +X_w', np.allclose(Mt[:3, 1], (0, -1, 0)) and np.allclose(Mt[:3, 0], (0, 0, 1)) and np.allclose(Mt[:3, 2], (1, 0, 0)))
    for n, T in (('left_suction_tool', handoff.suction_link6_pose(3)), ('right_gripper_tool', handoff.gripper_link6_pose(3))):
        M = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f'/World/Tools/{n}'))).T
        dp = np.abs(M[:3, 3] - T[:3, 3] / 1000).max()
        dr = np.abs(M[:3, :3] - T[:3, :3]).max()
        check(f'{tag}: {n} link6 pose = handoff.py', dp < 1e-4 and dr < 1e-6, f'{dp * 1e3:.5f} mm, rot err {dr:.1e}')
        kin = [UsdPhysics.RigidBodyAPI(c).GetKinematicEnabledAttr().Get() for c in Usd.PrimRange(stage.GetPrimAtPath(f'/World/Tools/{n}')) if c.HasAPI(UsdPhysics.RigidBodyAPI)]
        check(f'{tag}: {n} bodies kinematic', len(kin) >= 1 and all(kin), f'{len(kin)} bodies')
    check(f'{tag}: gripper articulation root removed in scene (kinematic bodies)', not stage.GetPrimAtPath('/World/Tools/right_gripper_tool').HasAPI(UsdPhysics.ArticulationRootAPI))
    sc = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Tools/left_suction_tool/suction_tcp_compressed')).ExtractTranslation())
    check(f'{tag}: suction_tcp_compressed lands on the target spine centre (0, -0.1, h/2)', np.abs(sc - np.array([0, -0.1, 0.110])).max() < 1e-4, f'{np.round(sc, 5).tolist()}')
    top = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Tools/right_gripper_tool/top_stop_frame')).ExtractTranslation())
    gap = getattr(handoff, 'GRIPPER_STOP_GAP', 0.0) * 1e-3     # gripper v3: stop face sits G_NOM above the top edge
    check(f'{tag}: gripper top_stop_frame {gap*1e3:.1f} mm above the carton top edge, x = 0', abs(top[2] - boxes[ti][1][2] - gap) < 1e-4 and abs(top[0]) < 1e-6, f'{np.round(top, 5).tolist()}')
    gp = {}
    for side in ('left', 'right'):
        for tris, _, pth in collect(stage, f'/World/Tools/right_gripper_tool/finger_{side}', {'guide'}, False):
            if f'finger_pad_{side}' in pth:
                gp[side] = tris[..., 0].max() if side == 'left' else tris[..., 0].min()
    gap = (gp['right'] - gp['left']) * 1e3
    check(f'{tag}: gripper pads closed on the 16 mm carton', abs(gap - 16.0) < 0.05, f'{gap:.3f} mm')
    cam = stage.GetPrimAtPath('/World/Camera')
    check(f'{tag}: camera prim present', cam.IsValid() and cam.IsA(UsdGeom.Camera))
    groups = []
    for root in ('/World/Shelf', '/World/Cartons', '/World/Tools'):
        groups += collect(stage, root, {'default', 'guide'}, True)
    pal = [(0.45, 0.65, 0.4), (0.85, 0.75, 0.3), (0.5, 0.45, 0.8), (0.3, 0.7, 0.8), (0.8, 0.5, 0.5)]
    for k, (t, c, pth) in enumerate(groups):
        if '/Tools/left' in pth:
            groups[k] = (t, (0.93, 0.55, 0.12), pth)
        elif '/Tools/right' in pth:
            groups[k] = (t, (0.15, 0.55, 0.85), pth)
        elif '/Cartons/' in pth:
            ci = int(pth.split('carton_')[1].split('_')[0])
            groups[k] = (t, (0.9, 0.15, 0.15) if ci == ti else pal[seq[ci] - 1], pth)
    render(groups, USD / 'previews' / 'scene_overview.png', 24, -52, lim=([-0.3, -0.35, 0.0], [0.4, 0.35, 0.35]), title='shelf_handoff_scene overview (target red; suction orange, gripper blue; collision hulls)', size=(9, 7))
    render(groups, USD / 'previews' / 'scene_handoff_closeup.png', 20, -50, lim=([-0.12, -0.18, 0.0], [0.12, 0.06, 0.34]), title='handoff close-up (collision hulls)', size=(8, 8))
    render(groups, USD / 'previews' / 'scene_top.png', 89, -90, lim=([-0.25, -0.35, 0.0], [0.30, 0.15, 0.3]), title='scene top view (+X right, +Y into the shelf = up)', size=(8, 7))
    render(groups, USD / 'previews' / 'scene_side.png', 0, 0, lim=([-0.1, -0.35, 0.0], [0.1, 0.4, 0.35]), title='scene side view from +X', size=(8, 6))
    return {'carton_zmin_mm': [z * 1e3 for z in zmins], 'max_overlap_mm': worst * 1e3, 'gripper_pad_gap_closed_mm': gap}


def verify_new_files():
    """pick_test_scene, shelf_cartons, rebel variants, dual-arm rig variant: structure only (the physics is tested in Isaac, tests/)."""
    tag = 'new files'
    st = Usd.Stage.Open(str(USD / 'pick_test_scene.usda'))
    check(f'{tag}: pick_test_scene: 5 cartons, 2 tools on carrier + slide joint, PhysicsScene TGS/CPU', len(list(st.GetPrimAtPath('/World/Cartons').GetChildren())) == 5
          and all(st.GetPrimAtPath(f'/World/Tools/{n}/joints/world_slide').IsValid() and st.GetPrimAtPath(f'/World/Tools/{n}_carrier').HasAPI(UsdPhysics.ArticulationRootAPI)
                  and not st.GetPrimAtPath(f'/World/Tools/{n}').HasAPI(UsdPhysics.ArticulationRootAPI) for n in ('left_suction_tool', 'right_gripper_tool'))
          and st.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:solverType').Get() == 'TGS')
    ss = Usd.Stage.Open(str(USD / 'shelf_cartons.usda'))
    check(f'{tag}: shelf_cartons has no PhysicsScene (reference it into a scene), 5 cartons', not any(p.IsA(UsdPhysics.Scene) for p in ss.Traverse()) and len(ss.GetPrimAtPath('/World/Cartons').GetChildren()) == 5)
    for f, tool in (('rebel_left_suction.usda', 'left_suction_tool'), ('rebel_right_gripper.usda', 'right_gripper_tool')):
        if not (USD / f).exists():
            continue
        s2 = Usd.Stage.Open(str(USD / f))
        fj = [p for p in s2.Traverse() if p.IsA(UsdPhysics.FixedJoint) and tool in p.GetPath().pathString and 'link_8_to' in p.GetPath().pathString]
        check(f'{tag}: {f}: tool fixed to link_8, tool articulation root removed', len(fj) == 1 and not s2.GetPrimAtPath(f'/World/{tool}').HasAPI(UsdPhysics.ArticulationRootAPI)
              and [str(t) for t in UsdPhysics.Joint(fj[0]).GetBody0Rel().GetTargets()] == ['/World/rebel/root_joint/link_8'])
    if (USD / 'dual_arm_rig_medical.usda').exists():
        s3 = Usd.Stage.Open(str(USD / 'dual_arm_rig_medical.usda'))
        chain = 'base_link/motor1/link1/motor2/link2/motor3/link3/motor4/link4/motor5/link5/link6'
        ok = True
        for side, tool in (('left', 'left_suction_tool'), ('right', 'right_gripper_tool')):
            L6 = f'/World/dual_arm_rig/arm_{side}/{chain}'
            ok &= (not s3.GetPrimAtPath(L6 + '/xeg32_base_link').IsActive()) and s3.GetPrimAtPath(f'{L6}/{tool}/base').IsValid()
            ok &= s3.GetPrimAtPath(f'/World/dual_arm_rig/Joints/arm_{side}_medical_tool_mount').IsValid()
            ok &= all(not s3.GetPrimAtPath(f'/World/dual_arm_rig/Joints/arm_{side}_xeg32_{j}').IsActive() for j in ('mount_joint', 'left_carriage_joint', 'right_carriage_joint', 'camera_tilt_joint'))
        check(f'{tag}: dual_arm_rig_medical: XEG-32 tooling + joints removed, both tools fixed to link6', ok)
        check(f'{tag}: dual_arm_rig_medical: nested rigid bodies flattened (29)', s3.GetPrimAtPath('/World').GetCustomDataByKey('nested_bodies_flattened') == 29)

    d2 = USD / 'd200_dual_arm_medical.usda'
    if d2.exists():
        s4 = Usd.Stage.Open(str(d2))
        U = '/D200/UpperAssembly/dual_arm_rig'
        ok = True
        for side, tool in (('left', 'left_suction_tool'), ('right', 'right_gripper_tool')):
            ok &= (not s4.GetPrimAtPath(f'{U}/arm_{side}/xeg32_base_link').IsActive()) and s4.GetPrimAtPath(f'{U}/arm_{side}/{tool}/base').IsValid()
            ok &= all(not s4.GetPrimAtPath(f'{U}/Joints/arm_{side}_xeg32_{j}').IsActive() for j in ('mount_joint', 'left_carriage_joint', 'right_carriage_joint', 'camera_tilt_joint'))
        check(f'{tag}: d200_dual_arm_medical: XEG-32 removed, both tools fixed to link6 (sibling bodies, no nesting)', ok)
        wf = s4.GetPrimAtPath('/D200/joints/world_fix')
        locked = all(UsdPhysics.RevoluteJoint(s4.GetPrimAtPath('/D200/joints/' + n)).GetUpperLimitAttr().Get() == 0.0 for n in ('left_wheel_joint', 'right_wheel_joint', 'front_left_swivel_joint', 'rear_right_wheel_2_joint'))
        check(f'{tag}: d200_dual_arm_medical: AMR static placeholder (enabled fixed joint to the world, wheel/swivel/caster joints locked)',
              wf.IsValid() and wf.GetAttribute('physics:jointEnabled').Get() is not False and locked)
        sv = s4.GetPrimAtPath(U + '/Joints/arm_left_medical_tool_mount')
        check(f'{tag}: d200_dual_arm_medical: D200 tool variants use the URDF joint names',
              s4.GetPrimAtPath(f'{U}/arm_left/left_suction_tool/joints/arm_left_pad_1_buffer_joint').IsValid() and s4.GetPrimAtPath(f'{U}/arm_right/right_gripper_tool/joints/arm_right_finger_left_joint').IsValid()
              and s4.GetPrimAtPath(f'{U}/arm_right/right_gripper_tool/joints/arm_right_finger_right_joint').IsValid())
    cg = USD / 'cabinet_grasp_test_scene.usda'
    if cg.exists():
        s5 = Usd.Stage.Open(str(cg))
        check(f'{tag}: cabinet_grasp_test_scene: stocked cabinet referenced, shelf lips replaced by one plate, 2 carriers (SX, GR) with root/jy/jz joints',
              s5.GetPrimAtPath('/World/Cartons/Shelf_04').IsValid() and not s5.GetPrimAtPath('/World/Cabinet/Cabinet/ShelfLip_04').IsActive()
              and all(s5.GetPrimAtPath(f'/World/Tools/{n}/joints/{n}_jz').IsValid() and s5.GetPrimAtPath(f'/World/Tools/{n}/root').HasAPI(UsdPhysics.ArticulationRootAPI) for n in ('SX', 'GR')))



if __name__ == '__main__':
    (USD / 'previews').mkdir(exist_ok=True)
    (OUT / 'data').mkdir(exist_ok=True)
    verify_suction()
    verify_gripper()
    verify_gripper('right_gripper_tool_independent.usda', False)
    extra = verify_scene()
    verify_new_files()
    sizes = {p.name: p.stat().st_size for p in USD.glob('*.usda')}
    res = {'all_passed': all(c['ok'] for c in checks), 'n_checks': len(checks), 'n_failed': sum(not c['ok'] for c in checks),
           'tolerances': {'position_m': POS_TOL, 'angle_deg': ANG_TOL}, 'file_sizes_bytes': sizes, 'scene_measurements': extra, 'checks': checks}
    (OUT / 'data' / 'verify.json').write_text(json.dumps(res, indent=1, default=float))
    print(f"\n{res['n_checks'] - res['n_failed']}/{res['n_checks']} checks passed")
    sys.exit(0 if res['all_passed'] else 1)
