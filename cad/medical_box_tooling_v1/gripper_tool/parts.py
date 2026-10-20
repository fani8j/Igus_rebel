"""Parametric parts of the right-arm gripper tool (cadquery).  All coordinates mm in the URDF link6 frame unless stated.

Printed parts : flange_disc, gripper_plate, top_stop (fixed centre stop + yoke), finger_left, finger_right, pad_left / pad_right (TPU)
Purchased     : SMC LEHF20K2-48 (body + 2 jaws from the vendor STEP), screws, pins, heat-set inserts (inserts only as holes)
Finger geometry is built in a local finger frame (origin on the gripper axis at the jaw plane, closed position, +z away from the jaw) and placed
with place_finger().  Left finger = x < 0, right finger = its mirror image.
"""
import math
import numpy as np
import cadquery as cq
from OCP.STEPControl import STEPControl_Reader
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
import params as P

V = cq.Vector


# ------------------------------------------------------------------------------------------------ primitives
def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def cyl(x, y, z0, z1, d, axis=(0, 0, 1)):
    """cylinder of diameter d from z0 to z1 along +z (axis override for x: pass z0/z1 as the start/end along that axis, base point (x,y,z0) is then interpreted per axis)"""
    return cq.Solid.makeCylinder(d / 2, z1 - z0, V(x, y, z0), V(*axis))


def cyl_x(x0, x1, y, z, d):
    return cq.Solid.makeCylinder(d / 2, abs(x1 - x0), V(min(x0, x1), y, z), V(1, 0, 0))


def cone(x, y, z0, z1, r0, r1):
    """cone along +z from radius r0 at z0 to r1 at z1"""
    return cq.Solid.makeCone(r0, r1, z1 - z0, V(x, y, z0), V(0, 0, 1))


def prism_x(x0, x1, pts_yz):
    return cq.Workplane('YZ', origin=(x0, 0, 0)).polyline(pts_yz).close().extrude(x1 - x0).val()


def prism_y(y0, y1, pts_xz):
    """polygon in (x,z) extruded from y0 to y1"""
    wp = cq.Workplane('XZ', origin=(0, y1, 0)).polyline(pts_xz).close().extrude(y1 - y0)     # XZ normal is -Y
    return wp.val()


def cut_all(s, tools):
    for t in tools:
        s = s.cut(t)
    return s


def fuse_all(shapes):
    s = shapes[0]
    for o in shapes[1:]:
        s = s.fuse(o)
    return s


def clean(s):
    return cq.Workplane(obj=s).clean().val()


def move(s, dx=0.0, dy=0.0, dz=0.0):
    return s.translate(V(dx, dy, dz))


def csk_down(x, y, z_top, d_thru, d_csk, depth_csk, z_bot):
    """countersunk through hole, countersink opening at z_top (screw from the +z side)"""
    th = cyl(x, y, z_bot - 0.1, z_top + 0.1, d_thru)
    r_b = d_csk / 2 - depth_csk
    cs = cone(x, y, z_top - depth_csk, z_top + 0.0, r_b, d_csk / 2)
    return th.fuse(cs)


def csk_up(x, y, z_face, d_thru, d_csk, depth_csk, z_top):
    """countersunk through hole, countersink opening at z_face (screw from the -z side)"""
    th = cyl(x, y, z_face - 0.1, z_top + 0.1, d_thru)
    r_t = d_csk / 2 - depth_csk
    cs = cone(x, y, z_face, z_face + depth_csk, d_csk / 2, r_t)
    return th.fuse(cs)


# ------------------------------------------------------------------------------------------------ printed parts
def flange_disc():
    z0, z1 = P.FLANGE_Z, P.Z_DISC_TOP
    s = cyl(0, 0, z0, z1, 2 * P.DISC_R)
    tools = []
    for (x, y) in P.PCD43 + P.PCD35:                                  # 12 countersunk M3 (PCD 43 primary, PCD 35 spare), screw from the gripper side
        tools.append(csk_down(x, y, z1, P.M3_CLR, P.M3_CSK_D, P.M3_CSK_DEPTH, z0))
    for (x, y) in P.M4_XY:                                            # heat-set M4 inserts for the gripper plate
        tools.append(cyl(x, y, z1 - P.M4_INSERT_DEPTH, z1 + 0.1, P.M4_INSERT_HOLE))
    for (x, y) in P.BODY_DOWEL_XY:                                    # dowel pin pockets
        tools.append(cyl(x, y, z1 - P.DOWEL_DISC_DEPTH, z1 + 0.1, P.DOWEL_HOLE_DISC))
    return clean(cut_all(s, tools))


def gripper_plate():
    z0, z1 = P.Z_DISC_TOP, P.Z_R
    s = box(-P.PLATE_X, P.PLATE_X, -P.PLATE_Y, P.PLATE_Y, z0, z1).intersect(cyl(0, 0, z0, z1, 2 * P.PLATE_RCLIP))
    tools = []
    for (x, y) in P.M5_XY:                                            # countersunk M5 from the flange side
        tools.append(csk_up(x, y, z0, P.M5_CLR, P.M5_CSK_D, P.M5_CSK_DEPTH, z1))
    for (x, y) in P.M4_XY:
        tools.append(cyl(x, y, z0 - 0.1, z1 + 0.1, P.M4_CLR))
    for (x, y) in P.BODY_DOWEL_XY:
        tools.append(cyl(x, y, z0 - 0.1, z1 + 0.1, P.DOWEL_HOLE_PLATE))
    return clean(cut_all(s, tools))


def finger_local():
    """LEFT finger in the local frame (x<0 side).  Foot on the jaw face, column, overhanging pad block with window below it, pad pocket, tip chamfer."""
    ft = P.FOOT_T
    foot = box(-P.FOOT_X[1], -P.FOOT_X[0], -P.FOOT_Y, P.FOOT_Y, 0.0, ft)
    yb = P.BLOCK_Y
    col = prism_x(-P.COL_X[1], -P.COL_X[0], [(-P.FOOT_Y, ft - 0.01), (P.FOOT_Y, ft - 0.01), (yb, P.Z_REG), (-yb, P.Z_REG)])
    block = box(-P.BLOCK_X[1], -P.BLOCK_X[0], -yb, yb, P.Z_REG, P.Z_TIP)
    s = foot.fuse(col).fuse(block)
    tools = []
    # lead-in chamfer (inner face), 45 degrees
    zc = P.Z_LIP_TOP
    tools.append(prism_y(-yb - 1, yb + 1, [(-P.FRAME_FACE + 0.01, zc), (-P.FRAME_FACE - P.CHAMF, P.Z_TIP + 0.01), (-P.FRAME_FACE + 0.01, P.Z_TIP + 0.01)]))
    # pad pocket in the inner face
    hw = P.PAD_W / 2 + P.PAD_CLEAR
    tools.append(box(-P.FRAME_FACE - P.POCKET_DEPTH, -P.FRAME_FACE + 0.01, -hw, hw, P.Z_POCKET0, P.Z_POCKET1))
    # retention inserts for the TPU pad screws (from the pocket floor, along -x)
    xf = -P.FRAME_FACE - P.POCKET_DEPTH
    for z in P.RET_Z:
        tools.append(cyl_x(xf - P.RET_INSERT_DEPTH, xf + 0.1, 0.0, z, P.RET_INSERT_HOLE))
    # foot: 4 x M3 clearance with low-head counterbore (from the top of the foot) + dowel pin hole from below
    for xm in P.JAW_M3_X:
        for ym in P.JAW_M3_Y:
            tools.append(cyl(-xm, ym, -0.1, ft + 0.1, P.M3_CLR))
            tools.append(cyl(-xm, ym, ft - P.FOOT_CB_DEPTH, ft + 0.1, P.FOOT_CB_D))
    tools.append(cyl(-P.JAW_DOWEL_X, 0.0, -0.1, P.FINGER_PIN_DEPTH, P.FINGER_PIN_D + 0.1))
    return clean(cut_all(s, tools))


def top_stop():
    """fixed yoke: crossbar (x +-1.5, y +-27.5, 7.5 high, face at Z_STOP) on two risers outside the finger sweep (|y| 22.5..27.5), down to two side plates (with a local ear) clamped on the SMC body
    side faces (y = +-17.5) by two M5 bolts through the existing d5.2 holes (x = +-15, along y).  Nothing touches the jaws, fingers or pads (>= 2.5 mm in every jaw state)."""
    t = P.STOP_T / 2
    zf = P.Z_STOP
    parts = [box(-t, t, -P.STOP_Y, P.STOP_Y, zf - P.STOP_H, zf)]
    for sg in (-1, 1):
        y0, y1 = sorted((sg * P.RISER_Y[0], sg * P.RISER_Y[1]))
        parts.append(box(-t, t, y0, y1, P.RISER_Z0, zf - P.STOP_H + 0.01))
        a_, b_ = sorted((sg * P.SIDE_Y[0], sg * P.SIDE_Y[1]))
        parts.append(box(-P.SIDE_X, P.SIDE_X, a_, b_, P.SIDE_Z[0], P.SIDE_Z[1]))
        a_, b_ = sorted((sg * P.EAR_Y[0], sg * P.EAR_Y[1]))
        parts.append(box(-P.EAR_X, P.EAR_X, a_, b_, P.EAR_Z0, P.SIDE_Z[1]))
    s = fuse_all(parts)
    tools = [cq.Solid.makeCylinder(P.BOLT_HOLE / 2, 60.0, V(x, -30.0, P.BOLT_Z), V(0, 1, 0)) for x in P.BOLT_X]
    return clean(cut_all(s, tools))


def pad_local():
    """LEFT TPU pad insert in the local frame: base plate (pocket depth) + fine horizontal ribs on the crest side; 2 counterbored retention holes."""
    x_face, x_floor, x_crest = -P.FRAME_FACE, -P.FRAME_FACE - P.POCKET_DEPTH, -P.PAD_CREST
    hw = P.PAD_W / 2
    base = box(x_floor, x_face, -hw, hw, P.PAD_Z0, P.PAD_Z1)
    ribs = []
    n = int((P.PAD_H - 2.0) / P.RIB_PITCH) + 1
    z = P.PAD_Z0 + 1.0
    for i in range(n):
        ribs.append(box(x_face - 0.001, x_crest, -hw + 0.6, hw - 0.6, z, z + P.RIB_W))
        z += P.RIB_PITCH
    s = fuse_all([base] + ribs)
    tools = []
    for zr in P.RET_Z:
        tools.append(cyl_x(x_floor - 0.1, x_crest + 0.1, 0.0, zr, 3.4))
        tools.append(cyl_x(x_face - P.PAD_CB_DEPTH, x_crest + 0.1, 0.0, zr, P.PAD_CB_D))
    return clean(cut_all(s, tools))


def mirror_x(s):
    return s.mirror('YZ')


# ------------------------------------------------------------------------------------------------ SMC gripper
_GRIP = {}


def smc_raw():
    if 'raw' not in _GRIP:
        r = STEPControl_Reader()
        r.ReadFile(P.C.GRIPPER_STEP)
        r.TransferRoots()
        sh = cq.Shape.cast(r.OneShape())
        sol = sorted(sh.Solids(), key=lambda s: -s.Volume())
        body = sol[0]
        jaws = sorted(sol[1:], key=lambda s: s.BoundingBox().xmin)    # xmin -36.8 (right jaw, x>0 mounting face), -29.0 (left jaw)
        _GRIP['raw'] = dict(body=body, jaw_right=jaws[0], jaw_left=jaws[1])
    return _GRIP['raw']


def smc_placed(o):
    """gripper in link6 with each jaw opened by o mm (0 closed .. 24 fully open)"""
    g = smc_raw()
    t = (P.GR_X0, P.GR_Y0, P.GR_Z0)
    return dict(body=move(g['body'], *t), jaw_left=move(g['jaw_left'], t[0] - o, t[1], t[2]), jaw_right=move(g['jaw_right'], t[0] + o, t[1], t[2]))


def place_finger(shape_local, side, o):
    """side -1 left (x<0), +1 right (already mirrored); o = jaw opening (mm)"""
    return move(shape_local, side * o, 0.0, P.Z_J)


_LOCAL = {}


def local_parts():
    if not _LOCAL:
        fl = finger_local()
        pl = pad_local()
        _LOCAL.update(finger_left=fl, finger_right=mirror_x(finger_local()), pad_left=pl, pad_right=mirror_x(pl))
    return _LOCAL


# ------------------------------------------------------------------------------------------------ hardware (nominal solids; threads not modelled)
def hardware(o):
    """list of dict(name, shape, host=[names whose thread/press-fit hole the part is allowed to overlap], mass_kg)"""
    hw = []
    ztop_d, zR = P.Z_DISC_TOP, P.Z_R
    for i, (x, y) in enumerate(P.PCD43):                           # M3x12 ISO 10642 countersunk, disc -> flange (host: robot flange, not modelled); 3.7 mm in the flange
        head = cone(x, y, ztop_d - P.M3_CSK_DEPTH, ztop_d, 1.5, 3.0)
        shank = cyl(x, y, ztop_d - P.M3_SCREW_L, ztop_d - P.M3_CSK_DEPTH + 0.01, 3.0)
        hw.append(dict(name='m3x12_csk_%d' % i, shape=head.fuse(shank), host=[], mass=0.0008))
    for i, x in enumerate(P.BOLT_X):                                # stop yoke clamp: M5x55 through the body, nylock nut on the far side
        yb = P.SIDE_Y[1]
        shank = cq.Solid.makeCylinder(2.5, 55.0, V(x, yb - 55.0, P.BOLT_Z), V(0, 1, 0))
        head = cq.Solid.makeCylinder(4.25, 5.0, V(x, yb, P.BOLT_Z), V(0, 1, 0))
        nut = cq.Solid.makeCylinder(4.6, 5.0, V(x, -yb - 5.0, P.BOLT_Z), V(0, 1, 0)).cut(cq.Solid.makeCylinder(2.5, 5.0, V(x, -yb - 5.0, P.BOLT_Z), V(0, 1, 0)))
        hw.append(dict(name='m5x55_stop_%d' % i, shape=head.fuse(shank), host=['smc_body'], mass=0.0112))
        hw.append(dict(name='nut_m5_stop_%d' % i, shape=nut, host=['m5x55_stop_%d' % i], mass=0.0018))
    for i, (x, y) in enumerate(P.M5_XY):                            # M5x12 ISO 10642, plate -> gripper
        head = cone(x, y, ztop_d, ztop_d + P.M5_CSK_DEPTH, 5.0, 2.4)
        shank = cyl(x, y, ztop_d + P.M5_CSK_DEPTH - 0.01, ztop_d + 12.0, 5.0)
        hw.append(dict(name='m5x12_csk_%d' % i, shape=head.fuse(shank), host=['smc_body'], mass=0.0024))
    for i, (x, y) in enumerate(P.M4_XY):                            # M4x10 ISO 4762, plate -> disc inserts
        head = cyl(x, y, zR, zR + 4.0, 7.0)
        shank = cyl(x, y, zR - 10.0, zR + 0.01, 4.0)
        hw.append(dict(name='m4x10_shcs_%d' % i, shape=head.fuse(shank), host=[], mass=0.0014))
    for i, (x, y) in enumerate(P.BODY_DOWEL_XY):                    # dowel pins 4x10
        hw.append(dict(name='pin4x10_%d' % i, shape=cyl(x, y, zR - 7.0, zR + 3.0, P.DOWEL_PIN), host=['smc_body'], mass=0.0009))
    for side, sname in ((-1, 'left'), (1, 'right')):
        for j, (xm, ym) in enumerate([(a, b) for a in P.JAW_M3_X for b in P.JAW_M3_Y]):       # M3x8 DIN 7984 finger -> jaw
            head = cyl(side * xm, ym, P.Z_J + P.FOOT_T - P.FOOT_CB_DEPTH, P.Z_J + P.FOOT_T - P.FOOT_CB_DEPTH + 2.0, 5.5)
            shank = cyl(side * xm, ym, P.Z_J + P.FOOT_T - P.FOOT_CB_DEPTH - 8.0, P.Z_J + P.FOOT_T - P.FOOT_CB_DEPTH + 0.01, 3.0)
            hw.append(dict(name='m3x8_%s_%d' % (sname, j), shape=move(head.fuse(shank), side * o, 0, 0), host=['smc_jaw_' + sname], mass=0.0004))
        hw.append(dict(name='pin2_5x6_%s' % sname, shape=move(cyl(side * P.JAW_DOWEL_X, 0, P.Z_J - P.JAW_DOWEL_DEPTH, P.Z_J - P.JAW_DOWEL_DEPTH + 6.0, 2.5), side * o, 0, 0),
                       host=['smc_jaw_' + sname], mass=0.0002))
        x_cb = P.FRAME_FACE + P.PAD_CB_DEPTH                      # |x| of the counterbore floor in the TPU pad (cut from the frame-face plane inwards)
        for k, z in enumerate(P.RET_Z):                              # M3x5 ISO 7380 pad retention, head sunk in the TPU, thread in a heat-set insert of the finger
            head = cyl_x(side * x_cb, side * (x_cb - 1.65), 0, P.Z_J + z, 5.7)
            shank = cyl_x(side * x_cb, side * (x_cb + 5.0), 0, P.Z_J + z, 3.0)
            hw.append(dict(name='m3x5_pad_%s_%d' % (sname, k), shape=move(head.fuse(shank), side * o, 0, 0), host=['finger_' + sname], mass=0.0003))
    return hw


def solids(o, with_hw=True):
    """name -> dict(shape, kind, mat, host) for the full tool at jaw opening o"""
    L = local_parts()
    g = smc_placed(o)
    out = {}
    out['flange_disc'] = dict(shape=flange_disc(), kind='printed', mat='PA12CF')
    out['gripper_plate'] = dict(shape=gripper_plate(), kind='printed', mat='PA12CF')
    out['smc_body'] = dict(shape=g['body'], kind='purchased', mat='smc')
    out['smc_jaw_left'] = dict(shape=g['jaw_left'], kind='purchased', mat='smc')
    out['smc_jaw_right'] = dict(shape=g['jaw_right'], kind='purchased', mat='smc')
    out['top_stop'] = dict(shape=top_stop(), kind='printed', mat='PA12CF')
    out['finger_left'] = dict(shape=place_finger(L['finger_left'], -1, o), kind='printed', mat='PA12CF')
    out['finger_right'] = dict(shape=place_finger(L['finger_right'], +1, o), kind='printed', mat='PA12CF')
    out['pad_left'] = dict(shape=place_finger(L['pad_left'], -1, o), kind='printed', mat='TPU')
    out['pad_right'] = dict(shape=place_finger(L['pad_right'], +1, o), kind='printed', mat='TPU')
    if with_hw:
        for h in hardware(o):
            out[h['name']] = dict(shape=h['shape'], kind='hardware', mat='steel', host=h['host'], mass=h['mass'])
    return out


# ------------------------------------------------------------------------------------------------ helpers
def props(shape, density):
    """(mass, com(3), inertia 3x3 about com in the shape's frame) with uniform density kg/mm3 -> kg, mm, kg mm2"""
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g)
    vol = g.Mass()
    c = g.CentreOfMass()
    m = g.MatrixOfInertia()
    I = np.array([[m.Value(i + 1, j + 1) for j in range(3)] for i in range(3)]) * density       # about the COM (volume units * density)
    return vol * density, np.array([c.X(), c.Y(), c.Z()]), I


def tess(shape, tol=0.05, ang=0.2):
    import trimesh
    vs, fs, off = [], [], 0
    for so in (shape.Solids() or [shape]):
        v, t = so.tessellate(tol, ang)
        vs += [x.toTuple() for x in v]
        fs += [[i + off for i in tri] for tri in t]
        off += len(v)
    m = trimesh.Trimesh(vertices=np.array(vs), faces=np.array(fs), process=True)
    m.merge_vertices()
    return m
