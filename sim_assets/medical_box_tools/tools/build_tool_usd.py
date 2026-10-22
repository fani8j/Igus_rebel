#!/usr/bin/env python3
"""Build the Isaac Sim USD assets for the medical-box tooling (re-runnable).

  usd/left_suction_tool.usda    one rigid body in the link6 frame (suction tool v5, one Gemini 305), frames as child Xforms incl. pad_{1,2,3}_suction_frame
  usd/right_gripper_tool.usda   articulation: base + 2 prismatic fingers (independent drives)
  usd/shelf_handoff_scene.usda  shelf + 8 cartons (box 3 pulled out) + both tools at the nominal handoff

Reads cad/medical_box_tooling_v1/{suction_tool_v5,gripper_tool,common} (read-only; env SUCTION_TOOL=suction_tool builds the v3 tool instead) and
sim_assets/medical_boxes/usd/*.usda (referenced, never edited).
Run with a python that has pxr + trimesh + numpy, e.g.
  "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/build_tool_usd.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = OUT.parent.parent
CAD = REPO / 'cad' / 'medical_box_tooling_v1'
SUCTION_NAME = os.environ.get('SUCTION_TOOL', 'suction_tool_v5')      # canonical left tool; 'suction_tool' = v3
SUCTION = CAD / SUCTION_NAME
GRIPPER = CAD / 'gripper_tool'
BOX_USD = REPO / 'sim_assets' / 'medical_boxes' / 'usd'
sys.path.insert(0, str(CAD / 'common'))

GENERATOR_VERSION = '2.0.0'

# ---- gains / physics parameters (documented in README) ----
DRIVE_STIFFNESS = 1.0e4      # N/m : tuned in Isaac Sim 5.1 (tests/test_right_gripper.py)
DRIVE_DAMPING = 200.0        # N s/m
MAX_FORCE = 24.3             # N per finger: catalog pushing force at 100 % for the finger geometry L = 43.2 mm (within SMC 11..28 N)
MAX_VEL = 0.04               # m/s per finger: catalog open/close speed 5..80 mm/s for BOTH fingers -> 40 mm/s per finger
CLOSE_TARGET = -0.010        # m: close command = past the closed stop, so the drive force saturates at MAX_FORCE on any carton
PAD_FRICTION = (0.5, 0.5)    # static, dynamic : same mu as the hold check in gripper_tool/README
HARD_FRICTION = (0.4, 0.3)
CARTON_FOR_ID = {1: 'agent_monorail', 2: 'angiolite', 3: 'accuforce', 4: 'ultimaster_nagomi', 5: 'conqueror_nc_pro'}
SEQUENCE = [1, 2, 3, 4, 5, 1, 2, 3]
PICK_SEQUENCE = [1, 2, 3, 5, 1]          # pick_test_scene (all 250 mm deep): target = index 2 (box 3, 16 mm), index 3 = box 5 (12 mm)
TARGET_INDEX = 2


# ------------------------------------------------------------------ URDF parsing (raw xacro, macro body)
def _floats(s):
    return [float(v) for v in s.split()]


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_xacro(path):
    root = ET.parse(path).getroot()
    macro = [e for e in root.iter() if e.tag.endswith('}macro')][0]
    links, joints = {}, {}
    strip = lambda s: s.replace('${prefix}', '')
    for l in macro.iter('link'):
        d = {'name': strip(l.get('name'))}
        inert = l.find('inertial')
        if inert is not None:
            i = inert.find('inertia')
            ixx, ixy, ixz, iyy, iyz, izz = (float(i.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'))
            d['mass'] = float(inert.find('mass').get('value'))
            d['com'] = np.array(_floats(inert.find('origin').get('xyz')))
            d['I'] = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        links[d['name']] = d
    for j in macro.iter('joint'):
        o = j.find('origin')
        d = {'name': strip(j.get('name')), 'type': j.get('type'), 'parent': strip(j.find('parent').get('link')),
             'child': strip(j.find('child').get('link')), 'xyz': np.array(_floats(o.get('xyz'))),
             'rpy': np.array(_floats(o.get('rpy')))}
        if j.find('axis') is not None:
            d['axis'] = np.array(_floats(j.find('axis').get('xyz')))
            lim = j.find('limit')
            d['lower'], d['upper'] = float(lim.get('lower')), float(lim.get('upper'))
            d['effort'], d['velocity'] = float(lim.get('effort')), float(lim.get('velocity'))
        joints[d['name']] = d
    return links, joints


# ------------------------------------------------------------------ USD helpers
def mat4(R, t):
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Matrix3d(*[float(x) for x in np.asarray(R, float).T.reshape(9)]))   # USD is row-vector: M = R^T
    m.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in t]))
    return m


def set_xform(prim, R, t):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(mat4(R, t))


def new_stage(path, root_name, doc):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/' + root_name)
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind('component')
    p = root.GetPrim()
    p.SetDocumentation(doc)
    p.SetAssetInfoByKey('name', root_name)
    p.SetAssetInfoByKey('version', GENERATOR_VERSION)
    return stage, p


def add_mesh(stage, path, stl, purpose=None):
    m = trimesh.load(str(stl), force='mesh')           # merges duplicate vertices
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = np.asarray(m.vertices, dtype=np.float64)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces, dtype=np.int32).reshape(-1)))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(False)
    if purpose:
        mesh.CreatePurposeAttr(purpose)
    return mesh


def preview_material(stage, path, rgb, metallic=0.0, roughness=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + '/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def phys_material(stage, path, fr):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(fr[0])
    api.CreateDynamicFrictionAttr(fr[1])
    api.CreateRestitutionAttr(0.0)
    return m


def bind_visual(mesh, mat, rgb):
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def make_collider(mesh, phys_mat):
    p = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
    mesh.CreatePurposeAttr('guide')
    UsdShade.MaterialBindingAPI.Apply(p).Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, 'physics')


def mass_api(prim, mass, com, I):
    """I = 3x3 inertia tensor about the CoG in the body frame -> diagonalInertia + principalAxes."""
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    q = Gf.Matrix3d(*[float(x) for x in V.T.reshape(9)]).ExtractRotation().GetQuat()   # row-vector convention
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, w)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*map(float, q.GetImaginary()))))


def joint_state(prim, kind, pos, vel=0.0):
    """Initial joint state (PhysicsJointStateAPI, as written by Isaac Sim 5.1): PhysX otherwise starts the articulation at q = 0."""
    prim.AddAppliedSchema(f'PhysicsJointStateAPI:{kind}')
    prim.CreateAttribute(f'state:{kind}:physics:position', Sdf.ValueTypeNames.Float).Set(float(pos))
    prim.CreateAttribute(f'state:{kind}:physics:velocity', Sdf.ValueTypeNames.Float).Set(float(vel))


def frame_xform(stage, parent, name, joint):
    x = UsdGeom.Xform.Define(stage, f'{parent}/{name}')
    set_xform(x.GetPrim(), rpy_matrix(joint['rpy']), joint['xyz'])
    return x


# ------------------------------------------------------------------ suction tool
# Pad buffer = SMC ZP2-T4030WNK10-B5 (10 mm non-rotating buffer, spring 1..3 N, params.PAD); everything in front of the buffer body
# front face (z > PAD_SPLIT_Z in the link6 frame) is the sliding head (pad + rod), the rest stays on the clamped body.
PAD_STROKE = 0.010            # m
PAD_SPLIT_Z = 0.14430         # m, front face of the buffer body (end of collision piece pad_*_00 and cut plane of the visual mesh)
BUFFER_K = 200.0              # N/m  : (3 N - 1 N) / 10 mm
BUFFER_TARGET = 0.005         # m    : drive target beyond the extended limit (q = 0): F = K (target - q) = 1.0 N at q = 0, 3.0 N at q = -10 mm
BUFFER_DAMPING = 1.6          # N s/m: ~ critical for the 3.3 g head (2 sqrt(K m) = 1.6)
BUFFER_MAX_FORCE = 6.0        # N    : drive force cap (> 3 N peak spring force)
VACUUM_KPA = 50.0             # design vacuum (Schmalz ECBPi, -50 kPa): ONE parameter for both force limits
PAD_AREA_M2 = 1.16e-4         # ZP2-T4030WNK10 effective area, params.PAD['area_cm2']
SHEAR_MU = 0.5                # shear limit = mu * coaxial limit (pad rubber friction)
SG_MAX_GRIP_DISTANCE = 0.004  # m
SG_RETRY_INTERVAL = 1.0       # s
SG_CLEARANCE_OFFSET = 0.0     # m
PAD_LETTER = {1: 'L', 2: 'M', 3: 'R'}
SG_COAXIAL_N = VACUUM_KPA * 1e3 * PAD_AREA_M2        # 5.8 N per attachment point (PHYSICAL; the 5.1 extension detaches at 2.3 to 4.8 x this, tool_api.SuctionLimitMonitor enforces it exactly)
SG_CAL_COAXIAL = 1.0                                  # authored coaxial limit = physical x this (a factor cannot calibrate the pull-off in 5.1, tool_api.SuctionLimitMonitor does)
SG_CAL_SHEAR = 2.2                                    # authored shear limit = physical x this: 3 pads then detach at 7.7 N (tool x) / 9.1 N (tool y) vs 3 x 2.9 = 8.7 N physical (measured)
SG_SHEAR_N = SHEAR_MU * SG_COAXIAL_N                  # 2.9 N per attachment point (PHYSICAL)
SUCTION_MATS = {            # name: (rgb, metallic, roughness)
    'printed': ((0.93, 0.55, 0.12), 0.0, 0.6),
    'rubber': ((0.08, 0.08, 0.09), 0.0, 0.9),
    'metal': ((0.72, 0.74, 0.78), 0.9, 0.35),
    'tube': ((0.25, 0.45, 0.75), 0.0, 0.4),
    'camera': ((0.12, 0.13, 0.15), 0.3, 0.4),
    'fitting': ((0.55, 0.80, 0.95), 0.0, 0.4),
}


def suction_category(stem):
    if stem.startswith('pad_'):
        return 'rubber'
    if stem.startswith(('nut_', 'washer_', 'collector', 'manifold')):
        return 'metal'
    if stem.startswith('elbow_'):
        return 'fitting'
    if stem.startswith(('tube_', 'trunk_', 'input_tube')):
        return 'tube'
    if stem.startswith('camera_gemini'):
        return 'camera'
    return 'printed'


def parallel_axis(m, r):
    r = np.asarray(r, float)
    return m * ((r @ r) * np.eye(3) - np.outer(r, r))


def pad_split_props():
    """Sliding head of each pad: cut meshes, mass (volume share of the 12 g pad part), CoG, inertia about the CoG (uniform density)."""
    pad_mass = json.load(open(SUCTION / 'data' / 'mass.json'))['bodies']
    out = {}
    for i, L in PAD_LETTER.items():
        m = trimesh.load(str(SUCTION / 'meshes' / 'visual' / f'pad_{L}.stl'), force='mesh')
        head = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, PAD_SPLIT_Z], cap=True)
        rear = trimesh.intersections.slice_mesh_plane(m, [0, 0, -1], [0, 0, PAD_SPLIT_Z], cap=True)
        assert head.is_watertight and rear.is_watertight
        mass = pad_mass[f'pad_{L}'][0] * head.volume / m.volume
        out[i] = dict(head=head, rear=rear, mass=mass, com=np.asarray(head.center_mass, float),
                      I=np.asarray(head.moment_inertia, float) * (mass / head.volume), pad_mass=pad_mass[f'pad_{L}'][0],
                      share=head.volume / m.volume)
    return out


def add_tm_mesh(stage, path, tm):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = np.asarray(tm.vertices, dtype=np.float64)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tm.faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(tm.faces, dtype=np.int32).reshape(-1)))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(False)
    return mesh


def build_suction(jprefix='', fname='left_suction_tool.usda'):
    """jprefix: prefix of the buffer joint prim names (the D200 rig variant uses 'arm_left_' to match the URDF / isaac_topic joint names)."""
    links, joints = parse_xacro(SUCTION / 'urdf' / 'left_suction_tool.urdf.xacro')
    path = OUT / 'usd' / fname
    path.unlink(missing_ok=True)
    R = '/left_suction_tool'
    stage, root = new_stage(path, 'left_suction_tool',
                            'Left-arm suction/scan end effector in the ReBeL link6 frame (metres, +Z out of the flange). Articulation '
                            '(root on this prim): rigid body `base` + 3 pad heads `pad_{1,2,3}` on prismatic buffer joints (+Z, 10 mm, spring 1..3 N) + '
                            'an Isaac Sim IsaacSurfaceGripper (`surface_gripper`) with 3 D6 attachment points (`joints/pad_{1,2,3}_attachment`). '
                            'Frames: suction_tcp, suction_tcp_compressed, wrist_camera_mount, wrist_camera_color_optical_frame, '
                            'pad_{1,2,3}_suction_frame (pad lip centres, +Z out of the lip). `joints/world_anchor` is a disabled fixed joint to the world; '
                            '`base/link6_mount_frame` is the frame to fix to the robot link6.')
    root.SetCustomDataByKey('source', f'cad/medical_box_tooling_v1/{SUCTION_NAME} (urdf/left_suction_tool.urdf.xacro, meshes/, data/mass.json)')
    root.SetCustomDataByKey('generator_version', GENERATOR_VERSION)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    t = links['suction_tool']
    root.SetCustomDataByKey('mass_kg', t['mass'])

    # ---- mass split: whole tool (URDF, about its CoG) minus the 3 sliding heads
    heads = pad_split_props()
    Mt, ct = t['mass'], t['com']
    I_o = t['I'] + parallel_axis(Mt, ct)
    Mb = Mt - sum(h['mass'] for h in heads.values())
    cb = (Mt * ct - sum(h['mass'] * h['com'] for h in heads.values())) / Mb
    Ib_o = I_o - sum(h['I'] + parallel_axis(h['mass'], h['com']) for h in heads.values())
    Ib = Ib_o - parallel_axis(Mb, cb)
    assert np.all(np.linalg.eigvalsh(Ib) > 0)

    UsdGeom.Scope.Define(stage, R + '/Looks')
    mats = {k: preview_material(stage, f'{R}/Looks/{k}', v[0], v[1], v[2]) for k, v in SUCTION_MATS.items()}
    pads_pm = phys_material(stage, R + '/Looks/pad_physics', PAD_FRICTION)
    hard_pm = phys_material(stage, R + '/Looks/hard_physics', HARD_FRICTION)

    body = {}
    bp = UsdGeom.Xform.Define(stage, R + '/base').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(bp)
    mass_api(bp, Mb, cb, Ib)
    bp.SetCustomDataByKey('mass_kg', Mb)
    body['base'] = bp
    for i in (1, 2, 3):
        pp = UsdGeom.Xform.Define(stage, f'{R}/pad_{i}').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(pp)
        mass_api(pp, heads[i]['mass'], heads[i]['com'], heads[i]['I'])
        pp.SetCustomDataByKey('mass_kg', heads[i]['mass'])
        pp.SetCustomDataByKey('note', 'sliding head (pad + buffer rod): volume share %.3f of the 12 g ZP2 part (ASSUMED uniform density)' % heads[i]['share'])
        UsdGeom.Xform.Define(stage, f'{R}/pad_{i}/visuals')
        UsdGeom.Xform.Define(stage, f'{R}/pad_{i}/collisions')
    UsdGeom.Xform.Define(stage, R + '/base/visuals')
    UsdGeom.Xform.Define(stage, R + '/base/collisions')
    UsdGeom.Xform.Define(stage, R + '/base/link6_mount_frame')      # identity: fix this to the robot link6 (tool root = link6 frame)

    n_vis = 0
    for stl in sorted((SUCTION / 'meshes' / 'visual').glob('*.stl')):
        cat = suction_category(stl.stem)
        if stl.stem.startswith('pad_'):
            i = {v: k for k, v in PAD_LETTER.items()}[stl.stem[-1]]
            m = add_tm_mesh(stage, f'{R}/base/visuals/{stl.stem}_buffer_body', heads[i]['rear'])
            bind_visual(m, mats[cat], SUCTION_MATS[cat][0])
            m.GetPrim().SetCustomDataByKey('part_type', cat)
            m = add_tm_mesh(stage, f'{R}/pad_{i}/visuals/{stl.stem}_head', heads[i]['head'])
            bind_visual(m, mats[cat], SUCTION_MATS[cat][0])
            m.GetPrim().SetCustomDataByKey('part_type', cat)
            n_vis += 1
            continue
        m = add_mesh(stage, f'{R}/base/visuals/{stl.stem}', stl)
        bind_visual(m, mats[cat], SUCTION_MATS[cat][0])
        m.GetPrim().SetCustomDataByKey('part_type', cat)
        n_vis += 1
    man = json.load(open(SUCTION / 'meshes' / 'collision' / 'manifest.json'))
    for pc in man['pieces']:
        stem = Path(pc['file']).stem
        if stem.startswith('pad_') and not stem.endswith('_00'):            # lip / rod / head pieces slide with the pad
            i = {v: k for k, v in PAD_LETTER.items()}[stem[4]]
            m = add_mesh(stage, f'{R}/pad_{i}/collisions/{stem}', SUCTION / pc['file'])
            make_collider(m, pads_pm)
        else:
            m = add_mesh(stage, f'{R}/base/collisions/{stem}', SUCTION / pc['file'])
            make_collider(m, hard_pm)

    # ---- frames (every fixed frame of the xacro; children of the root, rigidly in the link6 frame)
    frame_names = [j['child'] for j in joints.values() if j['parent'] == 'suction_tool']
    for name in frame_names:
        fx = frame_xform(stage, R, name, joints[name + '_joint'])
        if name.startswith('pad_') and name.endswith('_suction_frame'):
            pr = fx.GetPrim()
            pr.SetCustomDataByKey('role', 'pad lip centre, +Z out of the lip = Surface Gripper attachment point forward axis')
            pr.SetCustomDataByKey('pad_hold_force_N', SG_COAXIAL_N)

    # ---- pad buffers: prismatic +Z, q = 0 extended (lip z 159 mm), q = -0.010 fully compressed
    UsdGeom.Scope.Define(stage, R + '/joints')
    pad_xyz = {i: joints[f'pad_{i}_suction_frame_joint']['xyz'] for i in (1, 2, 3)}
    for i in (1, 2, 3):
        j = UsdPhysics.PrismaticJoint.Define(stage, f'{R}/joints/{jprefix}pad_{i}_buffer_joint')
        j.CreateBody0Rel().SetTargets([Sdf.Path(R + '/base')])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f'{R}/pad_{i}')])
        pos = Gf.Vec3f(*map(float, pad_xyz[i]))
        j.CreateLocalPos0Attr(pos)
        j.CreateLocalPos1Attr(pos)
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr('Z')
        j.CreateLowerLimitAttr(-PAD_STROKE)
        j.CreateUpperLimitAttr(0.0)
        j.CreateCollisionEnabledAttr(False)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
        d.CreateTypeAttr('force')
        d.CreateStiffnessAttr(BUFFER_K)
        d.CreateDampingAttr(BUFFER_DAMPING)
        d.CreateMaxForceAttr(BUFFER_MAX_FORCE)
        d.CreateTargetPositionAttr(BUFFER_TARGET)
        d.CreateTargetVelocityAttr(0.0)
        joint_state(j.GetPrim(), 'linear', 0.0)
    wf = UsdPhysics.FixedJoint.Define(stage, R + '/joints/world_anchor')
    wf.CreateBody1Rel().SetTargets([Sdf.Path(R + '/base')])
    wf.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    wf.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    wf.CreateJointEnabledAttr(False)

    # ---- Isaac Sim Surface Gripper (isaacsim.robot.surface_gripper, schema isaacsim.robot.schema): 3 D6 attachment points on the pad bodies
    UsdGeom.Scope.Define(stage, R + '/surface_gripper_joints')
    att = []
    for i in (1, 2, 3):
        jp = f'{R}/surface_gripper_joints/pad_{i}_attachment'
        d6 = UsdPhysics.Joint.Define(stage, jp)                       # generic PhysicsJoint = D6
        p = d6.GetPrim()
        p.AddAppliedSchema('IsaacAttachmentPointAPI')
        p.CreateAttribute('isaac:forwardAxis', Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set('Z')
        p.CreateAttribute('isaac:clearanceOffset', Sdf.ValueTypeNames.Float, False).Set(SG_CLEARANCE_OFFSET)
        d6.CreateBody0Rel().SetTargets([Sdf.Path(f'{R}/pad_{i}')])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(R + '/base')])      # placeholder; the extension re-targets body1 to the gripped object
        pos = Gf.Vec3f(*map(float, pad_xyz[i]))
        d6.CreateLocalPos0Attr(pos)
        d6.CreateLocalPos1Attr(pos)
        d6.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        d6.CreateExcludeFromArticulationAttr(True)
        d6.CreateJointEnabledAttr(True)
        d6.CreateBreakForceAttr(3.4028235e38)
        d6.CreateBreakTorqueAttr(3.4028235e38)
        for dof in ('transX', 'transY'):                              # locked (low > high)
            lim = UsdPhysics.LimitAPI.Apply(p, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        lim = UsdPhysics.LimitAPI.Apply(p, 'transZ')                  # NVIDIA gantry example values (isaacsim.robot.surface_gripper/data)
        lim.CreateLowAttr(0.0)
        lim.CreateHighAttr(0.01)
        for dof in ('rotX', 'rotY', 'rotZ'):
            lim = UsdPhysics.LimitAPI.Apply(p, dof)
            lim.CreateLowAttr(-3.0)
            lim.CreateHighAttr(3.0)
        for dof, k, c in (('rotX', 100.0, 0.0), ('rotY', 100.0, 0.0), ('rotZ', 10000.0, 0.0), ('transZ', 5000.0, 100.0)):
            dr = UsdPhysics.DriveAPI.Apply(p, dof)
            dr.CreateTypeAttr('force')
            dr.CreateStiffnessAttr(k)
            dr.CreateDampingAttr(c)
            dr.CreateTargetPositionAttr(0.0)
        att.append(Sdf.Path(jp))
    sg = stage.DefinePrim(R + '/surface_gripper', 'IsaacSurfaceGripper')
    sg.CreateAttribute('isaac:status', Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set('Open')
    sg.CreateAttribute('isaac:maxGripDistance', Sdf.ValueTypeNames.Float, False).Set(SG_MAX_GRIP_DISTANCE)
    sg.CreateAttribute('isaac:retryInterval', Sdf.ValueTypeNames.Float, False).Set(SG_RETRY_INTERVAL)
    sg.CreateAttribute('isaac:shearForceLimit', Sdf.ValueTypeNames.Float, False).Set(SG_SHEAR_N * SG_CAL_SHEAR)
    sg.CreateAttribute('isaac:coaxialForceLimit', Sdf.ValueTypeNames.Float, False).Set(SG_COAXIAL_N * SG_CAL_COAXIAL)
    sg.CreateRelationship('isaac:attachmentPoints', False).SetTargets(att)
    sg.CreateRelationship('isaac:grippedObjects', False)
    sg.SetCustomDataByKey('cal_factor_shear', SG_CAL_SHEAR)
    sg.SetCustomDataByKey('cal_factor_coaxial', SG_CAL_COAXIAL)
    sg.SetCustomDataByKey('vacuum_kpa', VACUUM_KPA)
    sg.SetCustomDataByKey('pad_area_m2', PAD_AREA_M2)
    sg.SetCustomDataByKey('shear_mu', SHEAR_MU)
    sg.SetCustomDataByKey('basis', 'PHYSICAL per attachment point: coaxial = vacuum x pad area = %.1f kPa x 1.16 cm2, shear = mu x coaxial; authored shear limit = physical x cal_factor_shear (Isaac 5.1 calibration, README)' % VACUUM_KPA)
    stage.GetRootLayer().Save()
    return n_vis, len(man['pieces']), dict(base_mass=Mb, base_com=cb, head_mass=[heads[i]['mass'] for i in (1, 2, 3)])


# ------------------------------------------------------------------ gripper tool
GRIP_MATS = {
    'printed': ((0.93, 0.55, 0.12), 0.0, 0.6),
    'stop': ((0.85, 0.2, 0.2), 0.0, 0.6),
    'finger': ((0.15, 0.55, 0.85), 0.0, 0.6),
    'pad': ((0.1, 0.1, 0.1), 0.0, 0.9),
    'body': ((0.75, 0.75, 0.78), 0.8, 0.4),
    'jaw': ((0.6, 0.62, 0.66), 0.8, 0.4),
}


def grip_category(stem):
    if 'top_stop' in stem:
        return 'stop'
    if 'smc_body' in stem:
        return 'body'
    if 'jaw' in stem:
        return 'jaw'
    if 'pad' in stem:
        return 'pad'
    if stem.startswith('finger_'):
        return 'finger'
    return 'printed'


def build_gripper(mimic=True, fname='right_gripper_tool.usda', jprefix=''):
    """mimic=True: ONE drive on finger_left_joint (maxForce 2 x 24.3 N) + PhysxMimicJointAPI on finger_right_joint (right = left, gearing -1);
    mimic=False: two independent drives (fallback, maxForce 24.3 N each)."""
    links, joints = parse_xacro(GRIPPER / 'urdf' / 'right_gripper_tool.urdf.xacro')
    path = OUT / 'usd' / fname
    path.unlink(missing_ok=True)
    stage, root = new_stage(path, 'right_gripper_tool',
                            'Right-arm gripper (SMC LEHF20K2-48 + printed fingers) in the ReBeL link6 frame (metres, +Z out of the '
                            'flange). Articulation: base + two prismatic fingers along X (0 = closed, pad gap 8 mm; 0.024 = open, '
                            'pad gap 56 mm). Authored at OPEN.')
    root.SetCustomDataByKey('source', 'cad/medical_box_tooling_v1/gripper_tool (urdf/right_gripper_tool.urdf.xacro, meshes/)')
    root.SetCustomDataByKey('generator_version', GENERATOR_VERSION)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    R = '/right_gripper_tool'
    UsdGeom.Scope.Define(stage, R + '/Looks')
    mats = {k: preview_material(stage, f'{R}/Looks/{k}', v[0], v[1], v[2]) for k, v in GRIP_MATS.items()}
    pad_pm = phys_material(stage, R + '/Looks/pad_physics', PAD_FRICTION)
    hard_pm = phys_material(stage, R + '/Looks/hard_physics', HARD_FRICTION)

    man = json.load(open(GRIPPER / 'meshes' / 'manifest.json'))
    open_q = joints['finger_left_joint']['upper']
    z_j = joints['finger_left_joint']['xyz'][2]
    bodies = {'gripper_base': 'base', 'finger_left': 'finger_left', 'finger_right': 'finger_right'}
    for link, name in bodies.items():
        x = UsdGeom.Xform.Define(stage, f'{R}/{name}')
        p = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        L = links[link]
        mass_api(p, L['mass'], L['com'], L['I'])
        if link == 'gripper_base':
            set_xform(p, np.eye(3), [0, 0, 0])
        else:
            ax = joints[link + '_joint']['axis']                     # (-1,0,0) left, (+1,0,0) right
            set_xform(p, np.eye(3), ax * open_q + np.array([0, 0, z_j]))   # authored at open
        UsdGeom.Xform.Define(stage, f'{R}/{name}/visuals')
        UsdGeom.Xform.Define(stage, f'{R}/{name}/collisions')
    n_vis = n_col = 0
    for v in man['visual']:
        stem = Path(v['file']).stem
        cat = grip_category(stem)
        body = bodies[v['link']]
        m = add_mesh(stage, f'{R}/{body}/visuals/{stem}', GRIPPER / v['file'])
        bind_visual(m, mats[cat], GRIP_MATS[cat][0])
        m.GetPrim().SetCustomDataByKey('part_type', cat)
        n_vis += 1
    for c in man['collision']:
        stem = Path(c['file']).stem
        body = bodies[c['link']]
        m = add_mesh(stage, f'{R}/{body}/collisions/{stem}', GRIPPER / c['file'])
        make_collider(m, pad_pm if 'finger_pad' in stem else hard_pm)
        n_col += 1

    UsdGeom.Scope.Define(stage, R + '/joints')
    Rz180 = np.diag([-1.0, -1.0, 1.0])
    for side in ('left', 'right'):
        jd = joints[f'finger_{side}_joint']
        j = UsdPhysics.PrismaticJoint.Define(stage, f'{R}/joints/{jprefix}finger_{side}_joint')
        j.CreateBody0Rel().SetTargets([Sdf.Path(f'{R}/base')])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f'{R}/finger_{side}')])
        # joint X axis = +X for right; for left the frames are rotated 180 deg about Z so that +q = URDF axis (-X)
        flip = jd['axis'][0] < 0
        rot = Gf.Quatf(0, 0, 0, 1) if flip else Gf.Quatf(1, 0, 0, 0)      # (w,x,y,z): Rz(180) = (0,0,0,1)
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, jd['xyz'])))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(rot)
        j.CreateLocalRot1Attr(rot)
        j.CreateAxisAttr('X')
        j.CreateLowerLimitAttr(jd['lower'])
        j.CreateUpperLimitAttr(jd['upper'])
        j.CreateCollisionEnabledAttr(False)
        if mimic and side == 'right':
            pr = j.GetPrim()                      # follows the left finger: q_right + gearing * q_left + offset = 0, gearing -1 (both open with +q)
            pr.AddAppliedSchema('PhysxMimicJointAPI:rotX')
            pr.CreateAttribute('physxMimicJoint:rotX:gearing', Sdf.ValueTypeNames.Float, False).Set(-1.0)
            pr.CreateAttribute('physxMimicJoint:rotX:offset', Sdf.ValueTypeNames.Float, False).Set(0.0)
            pr.CreateRelationship('physxMimicJoint:rotX:referenceJoint', False).SetTargets([Sdf.Path(R + f'/joints/{jprefix}finger_left_joint')])
            pr.CreateAttribute('physxMimicJoint:rotX:referenceJointAxis', Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set('rotX')
        else:
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
            d.CreateTypeAttr('force')
            # mimic: the left drive pushes both fingers through the constraint, so its limit is the sum of the two per-finger limits
            d.CreateMaxForceAttr(MAX_FORCE * (2.0 if mimic else 1.0))
            d.CreateStiffnessAttr(DRIVE_STIFFNESS)
            d.CreateDampingAttr(DRIVE_DAMPING)
            d.CreateTargetPositionAttr(open_q)
            d.CreateTargetVelocityAttr(0.0)
        # NOTE: no physxJoint:maxJointVelocity: in Isaac Sim 5.1 it degrades the drive force to about half (measured), the 40 mm/s
        # finger speed is enforced by the commanded target ramp (tools/tool_api.py) instead.
        joint_state(j.GetPrim(), 'linear', open_q)
    # the CAD jaw solids of the two fingers touch face to face (left jaw_low / right jaw_top at z = -7.1 mm) and their convex hulls
    # overlap in x below q = 18.4 mm, which jams the closing in the simulator: filter finger <-> finger collisions (pads still hit the carton)
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(R + '/finger_left')).CreateFilteredPairsRel().SetTargets([Sdf.Path(R + '/finger_right')])
    # optional world anchor for stand-alone testing (disabled by default)
    wf = UsdPhysics.FixedJoint.Define(stage, R + '/joints/world_anchor')
    wf.CreateBody1Rel().SetTargets([Sdf.Path(R + '/base')])
    wf.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    wf.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    wf.CreateJointEnabledAttr(False)

    for name in ('gripper_tcp', 'top_stop_frame'):
        frame_xform(stage, R, name, joints[name + '_joint'])
    stage.GetRootLayer().Save()
    return n_vis, n_col


# ------------------------------------------------------------------ scene
def carton_dims(uid):
    s = Usd.Stage.Open(str(BOX_USD / f'{uid}.usda'))
    d = s.GetDefaultPrim().GetCustomDataByKey('dims_mm')
    return d['height'] / 1000.0, d['length'] / 1000.0, d['thickness'] / 1000.0


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda v: np.array(v, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.column_stack([s, u, -f]), eye          # camera looks along -Z, up = +Y


def quat_from_R(R):
    q = Gf.Matrix3d(*[float(x) for x in np.asarray(R, float).T.reshape(9)]).ExtractRotation().GetQuat()    # row-vector convention
    return Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*map(float, q.GetImaginary())))


SLIDE_STIFFNESS, SLIDE_DAMPING, SLIDE_MAX_FORCE = 1.0e5, 3.0e3, 300.0      # world slide joint of the test carrier (N/m, N s/m, N)
LEFT_START_S = -0.030        # m : left tool starts 30 mm before the nominal (5 mm compressed) pose, along its +Z
GRIPPER_START_S = -0.120     # m : gripper starts 120 mm above its handoff pose (tool z points down)


def _dynamic_tools(stage, bid, dims, centres):
    """Tools as articulations: each tool base hangs on a world prismatic joint along the tool z axis (position drive).
    Joint frame = the nominal handoff pose of the tool (s = 0); the tool prim itself is authored at s = s0."""
    import handoff
    UsdGeom.Xform.Define(stage, '/World/Tools')
    for name, T, s0 in (('left_suction_tool', handoff.suction_link6_pose(bid, pulled=False), LEFT_START_S),
                        ('right_gripper_tool', handoff.gripper_link6_pose(bid), GRIPPER_START_S)):
        R, t = T[:3, :3], T[:3, 3] / 1000.0
        p = stage.DefinePrim(f'/World/Tools/{name}', 'Xform')
        p.GetReferences().AddReference(f'./{name}.usda')
        set_xform(p, R, t + R[:, 2] * s0)
        stage.OverridePrim(f'/World/Tools/{name}/joints/world_anchor').SetActive(False)     # replaced by world_slide
        # carrier: static root link of the articulation (fixed to the world at the s = 0 pose), the tool base slides on it along tool z
        cp = f'/World/Tools/{name}_carrier'
        cx = UsdGeom.Xform.Define(stage, cp)
        set_xform(cx.GetPrim(), R, t)
        UsdPhysics.RigidBodyAPI.Apply(cx.GetPrim())
        mass_api(cx.GetPrim(), 1.0, [0, 0, 0], np.eye(3) * 1e-3)
        UsdPhysics.ArticulationRootAPI.Apply(cx.GetPrim())
        rj = UsdPhysics.FixedJoint.Define(stage, cp + '/root_joint')
        rj.CreateBody1Rel().SetTargets([Sdf.Path(cp)])
        rj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, t)))
        rj.CreateLocalRot0Attr(quat_from_R(R))
        rj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        rj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        tp = stage.OverridePrim(f'/World/Tools/{name}')
        UsdPhysics.ArticulationRootAPI.Apply(tp)
        tp.RemoveAPI(UsdPhysics.ArticulationRootAPI)          # the carrier owns the articulation
        j = UsdPhysics.PrismaticJoint.Define(stage, f'/World/Tools/{name}/joints/world_slide')
        j.CreateBody0Rel().SetTargets([Sdf.Path(cp)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f'/World/Tools/{name}/base')])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr('Z')
        j.CreateLowerLimitAttr(-0.4)
        j.CreateUpperLimitAttr(0.4)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
        d.CreateTypeAttr('force')
        d.CreateStiffnessAttr(SLIDE_STIFFNESS)
        d.CreateDampingAttr(SLIDE_DAMPING)
        d.CreateMaxForceAttr(SLIDE_MAX_FORCE)
        d.CreateTargetPositionAttr(s0)
        d.CreateTargetVelocityAttr(0.0)
        joint_state(j.GetPrim(), 'linear', s0)
        j.GetPrim().SetCustomDataByKey('role', 'test carrier: slide s along the tool +Z, s = 0 = nominal handoff pose (suction: pads compressed 5 mm on the spine)')
    return 0.0


def build_scene(dynamic=False, shelf_only=False):
    """dynamic=False: shelf_handoff_scene.usda (kinematic tools at the nominal handoff). dynamic=True: pick_test_scene.usda
    (tools are articulations on a world prismatic slide joint along the tool z axis, target carton flush with its neighbours)."""
    import handoff
    from params import PULL_OUT
    dynamic = dynamic or shelf_only
    SEQ = PICK_SEQUENCE if dynamic else SEQUENCE
    path = OUT / 'usd' / ('shelf_cartons.usda' if shelf_only else 'pick_test_scene.usda' if dynamic else 'shelf_handoff_scene.usda')
    path.unlink(missing_ok=True)
    stage, world = new_stage(path, 'World',
                             'Shelf handoff demo. Frame W: origin floor / front edge below the target carton, +X along the shelf, '
                             '+Y into the shelf, +Z up (metres). Cartons stand like books, spine toward -Y; ' + ('tools are articulations driven along their z axis by a world slide joint (headless pick tests).' if dynamic else 'tools are kinematic.'))
    world.SetCustomDataByKey('source', 'cad/medical_box_tooling_v1/common/{handoff,params}.py + sim_assets/medical_boxes/usd')
    world.SetCustomDataByKey('generator_version', GENERATOR_VERSION)
    world.SetCustomDataByKey('sequence', Vt.IntArray(SEQ))
    world.SetCustomDataByKey('target_index', TARGET_INDEX)
    Usd.ModelAPI(world).SetKind('assembly')

    ps = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    if shelf_only:               # to be referenced into a scene that has its own PhysicsScene
        stage.RemovePrim('/World/PhysicsScene')
    elif dynamic:                  # CPU physics (the Surface Gripper extension needs it), TGS, 240 Hz
        ps.GetPrim().AddAppliedSchema('PhysxSceneAPI')
        ps.GetPrim().CreateAttribute('physxScene:solverType', Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set('TGS')
        ps.GetPrim().CreateAttribute('physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt, False).Set(240)
        ps.GetPrim().CreateAttribute('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool, False).Set(False)

    UsdGeom.Scope.Define(stage, '/World/Looks')
    grey = preview_material(stage, '/World/Looks/shelf_grey', (0.78, 0.78, 0.80), 0.0, 0.8)
    gnd = preview_material(stage, '/World/Looks/ground', (0.35, 0.37, 0.40), 0.0, 0.9)
    pm = phys_material(stage, '/World/Looks/shelf_physics', (0.5, 0.4))

    def box(path, c, size, mat, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(cube)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, 'physics')
        return cube

    # ---- row layout
    dims = {i: carton_dims(CARTON_FOR_ID[i]) for i in CARTON_FOR_ID}
    for i, (h, l, t) in dims.items():             # cross-check with params.BOXES
        b = handoff.BOXES[i]
        assert abs(h * 1000 - b['h']) < 1e-6 and abs(l * 1000 - b['l']) < 1e-6 and abs(t * 1000 - b['t']) < 1e-6, i
    ts = [dims[i][2] for i in SEQ]
    left_edges = np.concatenate([[0], np.cumsum(ts)[:-1]])
    centres = left_edges + np.array(ts) / 2
    centres -= centres[TARGET_INDEX]
    x_min, x_max = (centres - np.array(ts) / 2).min(), (centres + np.array(ts) / 2).max()
    depth = max(dims[i][1] for i in SEQ) + (0.0005 if dynamic else 0.025)          # interior depth along Y (pick test: cartons rest against the back board)
    board = 0.018
    margin = 0.0005 if dynamic else 0.14
    xl, xr = x_min - margin, x_max + margin
    top_h = 0.300
    UsdGeom.Xform.Define(stage, '/World/Shelf')
    xc, xw = (xl + xr) / 2, xr - xl
    box('/World/Shelf/floor_board', (xc, depth / 2, -board / 2), (xw + 2 * board, depth + board, board), grey, (0.8, 0.8, 0.82))
    box('/World/Shelf/back_board', (xc, depth + board / 2, (top_h + board) / 2), (xw + 2 * board, board, top_h + board), grey, (0.8, 0.8, 0.82))
    box('/World/Shelf/side_left', (xl - board / 2, depth / 2, (top_h + board) / 2), (board, depth, top_h + board), grey, (0.8, 0.8, 0.82))
    box('/World/Shelf/side_right', (xr + board / 2, depth / 2, (top_h + board) / 2), (board, depth, top_h + board), grey, (0.8, 0.8, 0.82))
    box('/World/Shelf/upper_board', (xc, depth / 2, top_h + board / 2), (xw + 2 * board, depth, board), grey, (0.8, 0.8, 0.82))
    if not shelf_only:
        UsdGeom.Xform.Define(stage, '/World/Ground')
        box('/World/Ground/slab', (xc, 0.4, -board - 0.05), (6.0, 6.0, 0.1), gnd, (0.35, 0.37, 0.40))

    # ---- cartons
    Rc = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)   # carton x->+Z_w, y(spine)->-Y_w, z->+X_w
    assert abs(np.linalg.det(Rc) - 1) < 1e-12
    UsdGeom.Xform.Define(stage, '/World/Cartons')
    for k, bid in enumerate(SEQ):
        h, l, t = dims[bid]
        y_spine = -PULL_OUT / 1000.0 if (k == TARGET_INDEX and not dynamic) else 0.0
        p = stage.DefinePrim(f'/World/Cartons/carton_{k}_{CARTON_FOR_ID[bid]}', 'Xform')
        p.GetReferences().AddReference('../../medical_boxes/usd/' + CARTON_FOR_ID[bid] + '.usda')
        set_xform(p, Rc, [centres[k], y_spine + l / 2, h / 2])
        p.SetCustomDataByKey('box_id', bid)
        p.SetCustomDataByKey('is_target', k == TARGET_INDEX)

    # ---- tools (handoff.py poses, mm -> m), kinematic
    UsdGeom.Xform.Define(stage, '/World/Tools')
    bid = SEQ[TARGET_INDEX]
    KIN_FILE = {'right_gripper_tool': 'right_gripper_tool_independent'}
    def kinematic_tools():
        for name, T in (('left_suction_tool', handoff.suction_link6_pose(bid)), ('right_gripper_tool', handoff.gripper_link6_pose(bid))):
            p = stage.DefinePrim(f'/World/Tools/{name}', 'Xform')
            p.GetReferences().AddReference('./' + KIN_FILE.get(name, name) + '.usda')
            set_xform(p, T[:3, :3], T[:3, 3] / 1000.0)
        # kinematic, no gravity; the tools' ArticulationRootAPI is removed (kinematic bodies cannot be in an articulation)
        def kin(path):
            pr = stage.OverridePrim(path)
            UsdPhysics.RigidBodyAPI.Apply(pr).CreateKinematicEnabledAttr(True)
            pr.AddAppliedSchema('PhysxRigidBodyAPI')
            pr.CreateAttribute('physxRigidBody:disableGravity', Sdf.ValueTypeNames.Bool).Set(True)
        for b in ('base', 'pad_1', 'pad_2', 'pad_3'):
            kin(f'/World/Tools/left_suction_tool/{b}')
        for b in ('base', 'finger_left', 'finger_right'):
            kin(f'/World/Tools/right_gripper_tool/{b}')
        for tool in ('left_suction_tool', 'right_gripper_tool'):
            pr = stage.OverridePrim(f'/World/Tools/{tool}')
            UsdPhysics.ArticulationRootAPI.Apply(pr)
            pr.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            # suction pads at the compressed pose of the handoff (5 mm). All bodies are kinematic, so the tool joints / Surface Gripper are deactivated here
        # (PhysX rejects joints between kinematic bodies); use pick_test_scene.usda for the working mechanisms.
        for i in (1, 2, 3):
            fp = stage.OverridePrim(f'/World/Tools/left_suction_tool/pad_{i}')
            xf = UsdGeom.Xformable(fp)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(mat4(np.eye(3), [0, 0, -0.005]))
        for pth in ('left_suction_tool/joints', 'left_suction_tool/surface_gripper_joints', 'left_suction_tool/surface_gripper', 'right_gripper_tool/joints/world_anchor',
                    'right_gripper_tool/joints/finger_left_joint', 'right_gripper_tool/joints/finger_right_joint'):
            stage.OverridePrim('/World/Tools/' + pth).SetActive(False)
        # fingers closed on the carton: finger offset o = (t - 8 mm)/2, per side
        t = dims[bid][2]
        o = (t - 0.008) / 2.0
        GR = '/World/Tools/right_gripper_tool'
        for side, sgn in (('left', -1.0), ('right', 1.0)):
            fp = stage.OverridePrim(f'{GR}/finger_{side}')
            xf = UsdGeom.Xformable(fp)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(mat4(np.eye(3), [sgn * o, 0, 0.15]))
        return o

    if shelf_only:
        stage.GetRootLayer().Save()
        return centres, 0.0
    o = _dynamic_tools(stage, bid, dims, centres) if dynamic else kinematic_tools()

    # ---- camera
    cam = UsdGeom.Camera.Define(stage, '/World/Camera')
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(15.2908)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    Rcam, eye = look_at((0.75, -0.95, 0.60), (0.02, -0.02, 0.17))
    set_xform(cam.GetPrim(), Rcam, eye)
    stage.GetRootLayer().Save()
    return centres, o



# ------------------------------------------------------------------ assembled variants (tool fixed to a robot link6 inside the robot's articulation)
REBEL_USD = Path(os.environ.get('REBEL_USD', '/home/farhan/robot_remote/igus_rebel_sim/rebel_6dof.usd'))       # single ReBeL, Isaac 5.1 URDF import (other session)
RIG_USD = Path(os.environ.get('DUAL_ARM_RIG_USD', '/media/farhan/My Passport/igus_nvidia/scratch_usd/dual_arm_stand_20261006/dual_arm_stand_isaac_usd/dual_arm_rig_v2.usda'))
TOOL_FILES = {'left_suction_tool': 'left_suction_tool.usda', 'right_gripper_tool': 'right_gripper_tool.usda'}
D200_TOOL_FILES = {'left_suction_tool': 'left_suction_tool_arm_left.usda', 'right_gripper_tool': 'right_gripper_tool_arm_right.usda'}      # joint names prefixed like the URDF


def attach_tool(stage, tool, tool_path, body0_path, joint_path, files=None):
    """Reference the tool at tool_path (its root frame = the link6 frame), drop its own articulation root and world anchor, and fix its
    base to body0 (the robot link6) with a fixed joint with identity frames -> the tool joints join the robot's articulation."""
    p = stage.DefinePrim(tool_path, 'Xform')
    p.GetReferences().AddReference('./' + (files or TOOL_FILES)[tool])
    UsdPhysics.ArticulationRootAPI.Apply(p)
    p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    stage.OverridePrim(tool_path + '/joints/world_anchor').SetActive(False)
    fj = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fj.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(tool_path + '/base')])
    for a in ('LocalPos0', 'LocalPos1'):
        getattr(fj, 'Create' + a + 'Attr')(Gf.Vec3f(0, 0, 0))
    for a in ('LocalRot0', 'LocalRot1'):
        getattr(fj, 'Create' + a + 'Attr')(Gf.Quatf(1, 0, 0, 0))
    return p


def build_rebel_variants():
    """rebel_left_suction.usda / rebel_right_gripper.usda: the single ReBeL USD (referenced, not copied) + the tool fixed to link_8 (= ReBeL link6, joint6 axis z)."""
    if not REBEL_USD.exists():
        print('no ReBeL USD found, skipping the single-arm variants:', REBEL_USD)
        return []
    src = Usd.Stage.Open(str(REBEL_USD))
    T = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(src.GetPrimAtPath('/igus_rebel_6dof/root_joint/link_8'))).T     # column-vector 4x4 at q = 0
    out = []
    for tool, fname in (('left_suction_tool', 'rebel_left_suction.usda'), ('right_gripper_tool', 'rebel_right_gripper.usda')):
        path = OUT / 'usd' / fname
        path.unlink(missing_ok=True)
        stage, world = new_stage(path, 'World', f'Single ReBeL (referenced from {REBEL_USD}, other session) with the {tool} fixed to link_8 (= ReBeL link6) in the robot articulation. '
                                                'Robot at its USD configuration (all joints 0).')
        world.SetCustomDataByKey('rebel_usd', str(REBEL_USD))
        ps = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        ps.CreateGravityMagnitudeAttr(9.81)
        rb = stage.DefinePrim('/World/rebel', 'Xform')
        rb.GetReferences().AddReference(str(REBEL_USD))
        tp = attach_tool(stage, tool, f'/World/{tool}', '/World/rebel/root_joint/link_8', f'/World/rebel/root_joint/joints/link_8_to_{tool}')
        set_xform(tp, T[:3, :3], T[:3, 3])
        stage.GetRootLayer().Save()
        out.append(fname)
    return out


def build_rig_medical():
    """dual_arm_rig_medical.usda: the operator's dual-arm stand rig (referenced, 13 MB, external drive) with the XEG-32 tooling, its joints and
    the compal carton/table deactivated, the v5 suction tool fixed to arm_left/link6, the LEHF20K2-48 gripper fixed to arm_right/link6, the README home
    state (lift -0.2, sliders -+0.005973, arms upright) and a shelf with cartons in front of the left arm."""
    if not RIG_USD.exists():
        print('no dual-arm rig USD found, skipping:', RIG_USD)
        return None
    path = OUT / 'usd' / 'dual_arm_rig_medical.usda'
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = stage.DefinePrim('/World', 'Xform')
    world.GetReferences().AddReference(str(RIG_USD), '/World')
    stage.SetDefaultPrim(world)
    world.SetDocumentation('Dual-arm stand rig (operator zip dual_arm_stand_isaac_usd_20261006, referenced) with the medical-pick tools: suction tool v5 on arm_left/link6, '
                           'LEHF20K2-48 gripper on arm_right/link6, XEG-32 tooling removed. Home: lift -0.2, sliders -+0.005973, arms upright. Shelf with cartons in front of the left arm.')
    world.SetCustomDataByKey('rig_usd', str(RIG_USD))
    chain = 'base_link/motor1/link1/motor2/link2/motor3/link3/motor4/link4/motor5/link5/link6'
    J = '/World/dual_arm_rig/Joints/'
    for pth in ('/World/CardboardBox', '/World/table'):
        stage.OverridePrim(pth).SetActive(False)
    for side, tool in (('left', 'left_suction_tool'), ('right', 'right_gripper_tool')):
        L6 = f'/World/dual_arm_rig/arm_{side}/{chain}'
        stage.OverridePrim(L6 + '/xeg32_base_link').SetActive(False)          # also removes the carriages and the camera mount below it
        for jn in ('xeg32_mount_joint', 'xeg32_left_carriage_joint', 'xeg32_right_carriage_joint', 'xeg32_camera_tilt_joint'):
            stage.OverridePrim(J + f'arm_{side}_{jn}').SetActive(False)
        attach_tool(stage, tool, f'{L6}/{tool}', L6, J + f'arm_{side}_medical_tool_mount')
    # README home state (the rig USD carries the lift at -0.2 but the sliders at 0): drive targets + joint states
    home = {'lift_joint': ('linear', -0.2), 'slider_left_joint': ('linear', -0.005973), 'slider_right_joint': ('linear', 0.005973)}
    for side in ('left', 'right'):
        for k in range(1, 7):
            home[f'arm_{side}_joint{k}'] = ('angular', 0.0)
    for jn, (kind, v) in home.items():
        jp = stage.OverridePrim(J + jn)
        UsdPhysics.DriveAPI.Apply(jp, kind).CreateTargetPositionAttr(float(v))
        joint_state(jp, kind, v)
    # shelf with cartons: W frame rotated 180 deg about Z (W +Y into the shelf = world -Y, the arms' forward direction), spine 0.40 m in front of the left arm
    arm_x, arm_y, mount_z = 0.228209, -0.179297, 1.1093 - 0.2                  # arm_left base axis, mount plane at lift -0.2 (cell_scene.yaml)
    shoulder_z = mount_z + 0.249                                              # joint2 axis height above the base (ReBeL: 0.100 + 0.149)
    spine = (arm_x, arm_y - 0.40, shoulder_z)
    floor_z = spine[2] - 0.110                                                # target carton (220 mm) mid-height at the shoulder height
    sh = stage.DefinePrim('/World/medical_shelf', 'Xform')
    sh.GetReferences().AddReference('./shelf_cartons.usda')
    set_xform(sh, np.diag([-1.0, -1.0, 1.0]), [spine[0], spine[1], floor_z])
    plinth = UsdGeom.Cube.Define(stage, '/World/shelf_plinth')
    plinth.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(plinth)
    xf.AddTranslateOp().Set(Gf.Vec3d(spine[0], spine[1] - 0.16, (floor_z - 0.02) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(0.30, 0.40, floor_z - 0.02))
    plinth.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.58)])
    UsdPhysics.CollisionAPI.Apply(plinth.GetPrim())
    if not stage.GetPrimAtPath('/World/GroundPlane'):
        gp = UsdGeom.Cube.Define(stage, '/World/GroundPlane')
        gp.CreateSizeAttr(1.0)
        g = UsdGeom.Xformable(gp)
        g.AddTranslateOp().Set(Gf.Vec3d(0, -0.5, -0.05))
        g.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.1))
        gp.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.37, 0.40)])
        UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
    # the rig's cable-holder bodies are authored 0.2 m lower than the fixed joints that hold them to the sliders (slider frames are at lift 0); the original rig
    # therefore warns 'joint with disjointed body transforms' for ch_v4_*_hold. Lift them by 0.2 m so that the joints are consistent at the authored (lift 0) pose.
    cache0 = UsdGeom.XformCache()
    for hname, dx in (('ch_v4_L1', 0.0), ('ch_v4_L2', 0.0), ('ch_v4_R1', 0.0), ('ch_v4_R2', 0.0)):
        hp = stage.GetPrimAtPath('/World/' + hname)
        m = Gf.Matrix4d(cache0.GetLocalToWorldTransform(hp))
        m.SetTranslateOnly(m.ExtractTranslation() + Gf.Vec3d(dx, 0, 0.2))
        xf = UsdGeom.Xformable(stage.OverridePrim('/World/' + hname))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)
    # Isaac Sim 5.1 does not parse rigid bodies nested in rigid bodies (the rig authors the arm chain base_link > motor1 > link1 > ... as nested bodies and the
    # tools hang below link6): every nested body gets resetXformStack with its composed world transform, so PhysX sees it as a free body of the same pose.
    cache = UsdGeom.XformCache()
    nested = []
    for pr in Usd.PrimRange(stage.GetPseudoRoot()):
        anc, par = False, pr.GetParent()
        while par and par.GetPath() != Sdf.Path('/'):
            anc = anc or par.HasAPI(UsdPhysics.RigidBodyAPI)
            par = par.GetParent()
        if pr.HasAPI(UsdPhysics.RigidBodyAPI) and anc:
            nested.append((pr.GetPath(), cache.GetLocalToWorldTransform(pr)))
    for pth, m in nested:
        xf = UsdGeom.Xformable(stage.OverridePrim(pth))
        xf.ClearXformOpOrder()
        xf.SetResetXformStack(True)
        xf.AddTransformOp().Set(m)
    world.SetCustomDataByKey('nested_bodies_flattened', len(nested))
    reach = np.hypot(0.40 - 0.159 - 0.0, 0.0)                                 # horizontal distance shoulder axis -> wrist centre with the tool horizontal
    world.SetCustomDataByKey('shelf_spine_world', Vt.DoubleArray(list(spine)))
    world.SetCustomDataByKey('reach_check', f'wrist centre {reach:.3f} m horizontally from the shoulder axis at shoulder height (ReBeL reach ~0.5 m): ROUGH only, no IK was run')
    stage.GetRootLayer().Save()
    return spine


# ------------------------------------------------------------------ MAIN rig: dual-arm stand on the D200 AMR
D200_DIR = Path(os.environ.get('D200_ASSET_DIR', OUT / 'usd' / 'd200'))       # copy of the operator's d200_with_dual_arm_20261015 zip (in the repo)
D200_USD = D200_DIR / 'd200_with_dual_arm.usd'
D200_ROBOT_Z = 0.003            # README of the AMR: robot root 3 mm above the floor to avoid initial penetration
HOME = {'lift_joint': ('linear', -0.2), 'slider_left_joint': ('linear', -0.005973), 'slider_right_joint': ('linear', 0.005973)}


def build_d200_medical():
    """d200_dual_arm_medical.usda: d200_with_dual_arm.usd (referenced) with the XEG-32 tooling removed and the medical tools fixed to link6 of both arms."""
    if not D200_USD.exists():
        print('no D200 asset dir, skipping:', D200_USD)
        return None
    path = OUT / 'usd' / 'd200_dual_arm_medical.usda'
    path.unlink(missing_ok=True)
    src = Usd.Stage.Open(str(D200_USD))
    U = '/D200/UpperAssembly/dual_arm_rig'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.DefinePrim('/D200', 'Xform')
    d200_ref = os.path.relpath(D200_USD, path.parent)                         # relative, so the scene works from any checkout
    d200_ref = d200_ref if d200_ref.startswith('.') else './' + d200_ref
    root.GetReferences().AddReference(d200_ref)
    stage.SetDefaultPrim(root)
    root.SetDocumentation('MAIN rig: dual-arm stand on the D200 AMR (operator zip d200_with_dual_arm_20261015, referenced) with the medical tools: suction tool v5 on arm_left/link6, LEHF20K2-48 gripper on '
                          'arm_right/link6, XEG-32 tooling removed. Home: lift -0.2, sliders -+0.005973, arms upright. The AMR is a STATIC placeholder (base welded to the world, wheels locked). No ground / PhysicsScene (see d200_rig_test_scene.usda).')
    root.SetCustomDataByKey('d200_usd', d200_ref)
    J = U + '/Joints/'
    for side, tool in (('left', 'left_suction_tool'), ('right', 'right_gripper_tool')):
        for n in ('xeg32_base_link', 'xeg32_left_carriage_link', 'xeg32_right_carriage_link', 'xeg32_camera_mount_link'):
            stage.OverridePrim(f'{U}/arm_{side}/{n}').SetActive(False)
        for jn in ('xeg32_mount_joint', 'xeg32_left_carriage_joint', 'xeg32_right_carriage_joint', 'xeg32_camera_tilt_joint'):
            stage.OverridePrim(J + f'arm_{side}_{jn}').SetActive(False)
        l6 = src.GetPrimAtPath(f'{U}/arm_{side}/link6')
        m = UsdGeom.Xformable(l6).GetLocalTransformation()                      # body-local frame of link6 inside arm_<side> (flat sibling bodies in this upper)
        tp = attach_tool(stage, tool, f'{U}/arm_{side}/{tool}', f'{U}/arm_{side}/link6', J + f'arm_{side}_medical_tool_mount', D200_TOOL_FILES)
        xf = UsdGeom.Xformable(tp)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)
    # the AMR is a STATIC PLACEHOLDER (its motion is owned by another team): base_link welded to the world, wheel / swivel / caster joints locked
    bl = UsdGeom.XformCache().GetLocalToWorldTransform(src.GetPrimAtPath('/D200/base_link'))
    wf = UsdPhysics.FixedJoint.Define(stage, '/D200/joints/world_fix')
    wf.CreateBody1Rel().SetTargets([Sdf.Path('/D200/base_link')])
    wf.CreateLocalPos0Attr(Gf.Vec3f(*[float(x) for x in bl.ExtractTranslation()]))
    wf.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    wf.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    wf.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    for jn in ['left_wheel_joint', 'right_wheel_joint'] + [f'{a}_{b}_swivel_joint' for a in ('front', 'rear') for b in ('left', 'right')] + \
              [f'{a}_{b}_wheel_{k}_joint' for a in ('front', 'rear') for b in ('left', 'right') for k in (1, 2)]:
        jp = stage.OverridePrim('/D200/joints/' + jn)
        j = UsdPhysics.RevoluteJoint(jp)
        j.CreateLowerLimitAttr(0.0)
        j.CreateUpperLimitAttr(0.0)
        dr = UsdPhysics.DriveAPI.Apply(jp, 'angular')
        dr.CreateTargetVelocityAttr(0.0)
    for side in ('left', 'right'):
        for k in range(1, 7):
            HOME[f'arm_{side}_joint{k}'] = ('angular', 0.0)
    for jn, (kind, v) in HOME.items():
        jp = stage.OverridePrim(J + jn)
        UsdPhysics.DriveAPI.Apply(jp, kind).CreateTargetPositionAttr(float(v))
        joint_state(jp, kind, v)
    stage.GetRootLayer().Save()
    return str(D200_USD)


def build_d200_scene():
    """d200_rig_test_scene.usda: ground + PhysicsScene (CPU, TGS, scene queries on) + the medical D200 rig. No shelf (deferred; see README)."""
    if not (OUT / 'usd' / 'd200_dual_arm_medical.usda').exists():
        return None
    path = OUT / 'usd' / 'd200_rig_test_scene.usda'
    for old in ('d200_medicine_pick_scene.usda',):
        (OUT / 'usd' / old).unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    stage, world = new_stage(path, 'World', 'D200 dual-arm medical rig (static AMR placeholder) on a ground plane. Needs the carb settings /physics/collisionCylinderCustomGeometry=true and '
                                               '/physics/collisionApproximateCylinders=false (cylinder tyres) and CPU physics.')
    ps = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    pr = ps.GetPrim()
    pr.AddAppliedSchema('PhysxSceneAPI')
    pr.CreateAttribute('physxScene:solverType', Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set('TGS')
    pr.CreateAttribute('physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt, False).Set(240)
    pr.CreateAttribute('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool, False).Set(False)
    pr.CreateAttribute('physxScene:enableSceneQuerySupport', Sdf.ValueTypeNames.Bool, False).Set(True)
    gp = UsdGeom.Cube.Define(stage, '/World/Ground')
    gp.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(gp)
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(40, 40, 0.1))
    gp.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.37, 0.40)])
    UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
    rb = stage.DefinePrim('/World/D200', 'Xform')
    rb.GetReferences().AddReference('./d200_dual_arm_medical.usda')
    stage.GetRootLayer().Save()
    return str(path)


# ------------------------------------------------------------------ cabinet grasp test scene (kinematic tool carriers, no arms / IK)
def make_carrier2(stage, name, tool, tool_file):
    """/World/Tools/<name>: a static articulation root `root` (fixed to the world), prismatic `jy` (tool y axis) to `stage_body`, prismatic `jz` (tool z axis) to the tool base.
    All poses are identity here; tests/test_cabinet_grasp.py places them per target (root pose T = link6 pose at jy = jz = 0, joint states = start offsets)."""
    base = f'/World/Tools/{name}'
    UsdGeom.Xform.Define(stage, base)
    for b, m in (('root', 1.0), ('stage_body', 0.5)):
        x = UsdGeom.Xform.Define(stage, f'{base}/{b}')
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        mass_api(x.GetPrim(), m, [0, 0, 0], np.eye(3) * 1e-3)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f'{base}/root'))
    tp = stage.DefinePrim(f'{base}/tool', 'Xform')
    tp.GetReferences().AddReference('./' + tool_file)
    UsdPhysics.ArticulationRootAPI.Apply(tp)
    tp.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    stage.OverridePrim(f'{base}/tool/joints/world_anchor').SetActive(False)
    rj = UsdPhysics.FixedJoint.Define(stage, f'{base}/joints/root_joint')
    rj.CreateBody1Rel().SetTargets([Sdf.Path(f'{base}/root')])
    for a, v in (('LocalPos0', Gf.Vec3f(0, 0, 0)), ('LocalPos1', Gf.Vec3f(0, 0, 0))):
        getattr(rj, 'Create' + a + 'Attr')(v)
    for a in ('LocalRot0', 'LocalRot1'):
        getattr(rj, 'Create' + a + 'Attr')(Gf.Quatf(1, 0, 0, 0))
    for jn, ax, b0, b1 in (('jy', 'Y', f'{base}/root', f'{base}/stage_body'), ('jz', 'Z', f'{base}/stage_body', f'{base}/tool/base')):
        j = UsdPhysics.PrismaticJoint.Define(stage, f'{base}/joints/{name}_{jn}')
        j.CreateBody0Rel().SetTargets([Sdf.Path(b0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(b1)])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr(ax)
        j.CreateLowerLimitAttr(-1.0)
        j.CreateUpperLimitAttr(1.0)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
        d.CreateTypeAttr('force')
        d.CreateStiffnessAttr(SLIDE_STIFFNESS)
        d.CreateDampingAttr(SLIDE_DAMPING)
        d.CreateMaxForceAttr(SLIDE_MAX_FORCE)
        d.CreateTargetPositionAttr(0.0)
        d.CreateTargetVelocityAttr(0.0)
        joint_state(j.GetPrim(), 'linear', 0.0)


def build_cabinet_grasp_scene():
    """cabinet_grasp_test_scene.usda: the stocked medicine cabinet (medicine_cabinet_filled.usda, referenced, read only) + two kinematic tool carriers (suction `SX`, gripper `GR`)."""
    src = REPO / 'sim_assets' / 'medical_boxes' / 'usd' / 'medicine_cabinet_filled.usda'
    if not src.exists():
        print('no stocked cabinet found:', src)
        return None
    path = OUT / 'usd' / 'cabinet_grasp_test_scene.usda'
    path.unlink(missing_ok=True)
    stage, world = new_stage(path, 'World', 'Stocked medicine cabinet (referenced, other session) + kinematic carriers for the suction tool (SX) and the right gripper (GR) for tests/test_cabinet_grasp.py. '
                                            'Carrier poses are set by the test per target carton. Cabinet frame: X across, Y into the cabinet (front at y = 0.05 shelf edge), Z up.')
    world.GetReferences().AddReference('../../medical_boxes/usd/medicine_cabinet_filled.usda')
    # scene-layer fix (the cabinet files are read only): each shelf board is a plate 0.06 .. 0.40 plus a separate 10 mm coplanar front strip (ShelfLip, 0.05 .. 0.06).
    # A carton sliding over that seam is caught by PhysX (it stopped when its spine reached y = 0.06, 17 N pull). Make the board one plate 0.05 .. 0.40 and drop the strip.
    for k in range(1, 8):
        stage.OverridePrim(f'/World/Cabinet/Cabinet/ShelfLip_0{k}').SetActive(False)
        ops = UsdGeom.Xformable(stage.OverridePrim(f'/World/Cabinet/Cabinet/Shelf_0{k}')).GetOrderedXformOps()
        tr, sc = ops[0].Get(), ops[1].Get()
        y0, y1 = 0.05, tr[1] + sc[1] / 2
        ops[0].Set(Gf.Vec3d(tr[0], (y0 + y1) / 2, tr[2]))
        ops[1].Set(Gf.Vec3d(sc[0], y1 - y0, sc[2]))
    UsdGeom.Xform.Define(stage, '/World/Tools')
    make_carrier2(stage, 'SX', 'suction', 'left_suction_tool.usda')
    make_carrier2(stage, 'GR', 'gripper', 'right_gripper_tool.usda')
    stage.GetRootLayer().Save()
    return str(path)


def build_cabinet_robot_scene():
    """cabinet_robot_test_scene.usda: the stocked medicine cabinet (same scene-layer shelf-lip fix as cabinet_grasp_test_scene) + the medical D200 dual-arm rig (referenced, static AMR) at /World/D200.
    The cabinet stays at the origin (cabinet frame: X across, Y into, Z up); tests/test_cabinet_grasp_robot.py places the ROBOT per target (xform of /World/D200 + world_fix joint pose)."""
    src = REPO / 'sim_assets' / 'medical_boxes' / 'usd' / 'medicine_cabinet_filled.usda'
    if not src.exists() or not (OUT / 'usd' / 'd200_dual_arm_medical.usda').exists():
        print('missing cabinet or D200 medical rig, skipping')
        return None
    path = OUT / 'usd' / 'cabinet_robot_test_scene.usda'
    path.unlink(missing_ok=True)
    stage, world = new_stage(path, 'World', 'Stocked medicine cabinet (referenced, read only) + the medical D200 dual-arm robot (static AMR placeholder) for tests/test_cabinet_grasp_robot.py. '
                                            'The cabinet is at the origin; the test moves the robot (xform of /World/D200 and the world_fix joint) to the parking pose of each target. '
                                            'Needs the carb settings of apply_d200_physics_settings().')
    world.GetReferences().AddReference('../../medical_boxes/usd/medicine_cabinet_filled.usda')
    for k in range(1, 8):
        stage.OverridePrim(f'/World/Cabinet/Cabinet/ShelfLip_0{k}').SetActive(False)
        ops = UsdGeom.Xformable(stage.OverridePrim(f'/World/Cabinet/Cabinet/Shelf_0{k}')).GetOrderedXformOps()
        tr, sc = ops[0].Get(), ops[1].Get()
        y0, y1 = 0.05, tr[1] + sc[1] / 2
        ops[0].Set(Gf.Vec3d(tr[0], (y0 + y1) / 2, tr[2]))
        ops[1].Set(Gf.Vec3d(sc[0], y1 - y0, sc[2]))
    rb = stage.DefinePrim('/World/D200', 'Xform')
    rb.GetReferences().AddReference('./d200_dual_arm_medical.usda')
    stage.GetRootLayer().Save()
    return str(path)


if __name__ == '__main__':
    print('suction  visual/collision meshes:', build_suction())
    build_suction('arm_left_', 'left_suction_tool_arm_left.usda')
    build_gripper(True, 'right_gripper_tool_arm_right.usda', 'arm_right_')
    print('gripper  visual/collision meshes:', build_gripper(), build_gripper(False, 'right_gripper_tool_independent.usda'))
    c, o = build_scene()
    print('scene centres (m):', np.round(c, 4), ' finger offset on 16 mm:', o)
    c, o = build_scene(dynamic=True)
    print('pick test scene centres (m):', np.round(c, 4))
    build_scene(shelf_only=True)
    print('rebel variants:', build_rebel_variants())
    print('dual-arm rig medical, shelf spine at:', build_rig_medical())
    print('d200 medical rig:', build_d200_medical())
    print('d200 scene:', build_d200_scene())
    print('cabinet grasp scene:', build_cabinet_grasp_scene())
    print('cabinet robot scene:', build_cabinet_robot_scene())
