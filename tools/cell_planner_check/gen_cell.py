#!/usr/bin/env python3
"""Generator: dual-arm ReBeL cell URDF + SRDF + MoveIt config files for one design JSON and one layout.

  python3 gen_cell.py --design designs/h000n.0.json --s 0.355 --out build/h000n.0_s0355 \
      [--structure cad|study|none] [--fingers union|grasp|open|hull_grasp|hull_open|none] [--matrix collision_matrix.json]

Planning frame "world" == W of the offline studies (see cellgeom.py).  Arms: arm1 (y_w = +s/2), arm2 (y_w = -s/2), identity base rotation.
The single-arm chain is a prefixed copy of src/igus_rebel_description/urdf/igus_rebel2.urdf (read only); joint limits are set explicitly
to the real controller limits used by the offline studies (j1/j4/j6 +-178 deg, j2/j3 [-79, 139] deg, j5 +-94 deg).
Tooling per link6: SMC gripper mesh, fingers (box union of both jaw states by default), knife post / clamp / handpiece (primitives),
camera body + bracket, and `armN_knife_tip` (TCP, T_link6_tcp, NO collision geometry: the last 20 mm of the blade are exempt).
Writes in --out: dual_arm.urdf, dual_arm.srdf, kinematics.yaml, joint_limits.yaml, ompl_planning.yaml, pilz_planning.yaml, cell_meta.json.
"""
import argparse, json, math, os, sys, itertools
import xml.etree.ElementTree as ET
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G

REPO = G.REPO
SRC_URDF = REPO + '/src/igus_rebel_description/urdf/igus_rebel2.urdf'
SRC_SRDF = REPO + '/src/igus_rebel_moveit_config/config/igus_rebel2.srdf'
LAYOUT_JSON = REPO + '/scan_results/dual_arm_cell_20261002/layout.json'
D2R = math.pi / 180.0
LIM_DEG = {'joint1': (-178, 178), 'joint2': (-79, 139), 'joint3': (-79, 139), 'joint4': (-178, 178), 'joint5': (-94, 94), 'joint6': (-178, 178)}
ARM_LINKS = ['base_link', 'motor1', 'link1', 'motor2', 'link2', 'motor3', 'link3', 'motor4', 'link4', 'motor5', 'link5', 'link6']
VEL, ACC = 0.785, 5.24
FINGER_STL = HERE + '/assets/finger_full.stl'


def f(v):
    return '%.6f' % v


def xyz(v):
    return ' '.join(f(x) for x in v)


# ------------------------------------------------------------------------------------------------ small URDF builders
class Urdf:
    def __init__(self, name):
        self.root = ET.Element('robot', name=name)
        self.mats = {}
        self.collision_links = []          # links that own collision geometry

    def material(self, name, rgba):
        if name in self.mats:
            return
        m = ET.SubElement(self.root, 'material', name=name)
        ET.SubElement(m, 'color', rgba=' '.join(str(c) for c in rgba))
        self.mats[name] = m

    def link(self, name):
        return ET.SubElement(self.root, 'link', name=name)

    def fixed(self, name, parent, child, xyz_=(0, 0, 0), rpy=(0, 0, 0)):
        j = ET.SubElement(self.root, 'joint', name=name, type='fixed')
        ET.SubElement(j, 'origin', xyz=xyz(xyz_), rpy=xyz(rpy))
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        return j

    def geom(self, link, shape, origin_xyz, origin_rpy, rgba_name, collision=True, visual=True):
        """shape: ('box', size3) | ('cylinder', r, l) | ('sphere', r) | ('mesh', filename)"""
        for kind, on in (('visual', visual), ('collision', collision)):
            if not on:
                continue
            el = ET.SubElement(link, kind)
            ET.SubElement(el, 'origin', xyz=xyz(origin_xyz), rpy=xyz(origin_rpy))
            g = ET.SubElement(el, 'geometry')
            if shape[0] == 'box':
                ET.SubElement(g, 'box', size=xyz(shape[1]))
            elif shape[0] == 'cylinder':
                ET.SubElement(g, 'cylinder', radius=f(shape[1]), length=f(shape[2]))
            elif shape[0] == 'sphere':
                ET.SubElement(g, 'sphere', radius=f(shape[1]))
            elif shape[0] == 'mesh':
                ET.SubElement(g, 'mesh', filename=shape[1])
            if kind == 'visual':
                ET.SubElement(el, 'material', name=rgba_name)
        if collision and link.get('name') not in self.collision_links:
            self.collision_links.append(link.get('name'))

    def capsule_parts(self, link, a, b, r, rgba_name):
        """cylinder between a and b plus a sphere at each end (a capsule); link6 frame"""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        v = b - a
        n = np.linalg.norm(v)
        if n < 1e-6:
            self.geom(link, ('sphere', r), a, (0, 0, 0), rgba_name)
            return
        rpy, L = G.z_to_vec_rpy(v)
        self.geom(link, ('cylinder', r, L), (a + b) / 2, rpy, rgba_name)
        self.geom(link, ('sphere', r), a, (0, 0, 0), rgba_name)
        self.geom(link, ('sphere', r), b, (0, 0, 0), rgba_name)

    def spheres(self, link, centres, radii, rgba_name, collision=True):
        radii = [radii] * len(centres) if np.isscalar(radii) else radii
        for c, r in zip(centres, radii):
            self.geom(link, ('sphere', float(r)), c, (0, 0, 0), rgba_name, collision=collision, visual=False)

    def write(self, path):
        ET.indent(self.root, space='  ')
        ET.ElementTree(self.root).write(path, xml_declaration=True, encoding='utf-8')


# ------------------------------------------------------------------------------------------------ meshes
def build_gripper_stl(des, outdir):
    """SMC LEHF20K2-48 mesh in the link6 frame (same transform as viewer/build_assets.py).  returns file path or None."""
    src = des['gripper'].get('stl')
    out = os.path.join(outdir, 'gripper_smc_link6.stl')
    if os.path.exists(out):
        return out
    if not src or not os.path.exists(src):
        return None
    import trimesh
    g = trimesh.load(src)
    g.apply_scale(0.001)
    Rg = np.eye(4)
    Rg[:3, :3] = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    g.apply_transform(Rg)
    b = g.bounds
    g.apply_translation([0, -(b[0][1] + b[1][1]) / 2, 0.0527 - b[0][2]])
    g.export(out)
    return out


def build_finger_hulls(outdir):
    """convex hull of each real finger mesh in the link6 frame for grasp / open state.  returns {(side,state): path}"""
    import trimesh
    paths = {}
    if not os.path.exists(FINGER_STL):
        return paths
    sys.path.insert(0, G.REPO + '/scan_results/holder_camera_fingers_20261002')
    JAW_Y, JAW_Z = -0.0415, 0.070
    XREF = {'grasp': 0.016 + 0.0186, 'open': 0.016 + 0.0186 + 0.024}
    base = trimesh.load(FINGER_STL)           # mm, finger frame
    for side in (-1, 1):
        for st in ('grasp', 'open'):
            out = os.path.join(outdir, 'finger_%s_%s_hull.stl' % ('L' if side < 0 else 'R', st))
            if not os.path.exists(out):
                m = base.copy()
                M = np.eye(4)
                # L: X6 = -xr + y_f ; R: X6 = xr - y_f ; Y6 = JAW_Y + (z_f + 5) ; Z6 = JAW_Z + x_f   (mm -> m below)
                if side < 0:
                    M[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
                    off = np.array([-XREF[st] * 1000, JAW_Y * 1000 + 5, JAW_Z * 1000])
                else:
                    M[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0.0]])
                    off = np.array([XREF[st] * 1000, JAW_Y * 1000 + 5, JAW_Z * 1000])
                M[:3, 3] = off
                m.apply_transform(M)
                m.apply_scale(0.001)
                h = m.convex_hull
                h.export(out)
            paths[(side, st)] = out
    return paths


# ------------------------------------------------------------------------------------------------ structure
def structure_boxes(mode, s, include_carrier=False):
    """list of (name, centre_w, size, color) for the fixed lift / rail / mount-plate structure (W frame)."""
    boxes = []
    if mode == 'none':
        return boxes
    if mode == 'study':
        # dalib.collision_free(): x_w in [-0.20, 0.11], |y_w| <= s/2 + 0.125, z_w <= 0 (extends far down)
        lo = np.array([-0.20, -(s / 2 + 0.125), -1.5])
        hi = np.array([0.11, (s / 2 + 0.125), 0.0])
        return [('study_box', (lo + hi) / 2, hi - lo, 'struct')]
    L = json.load(open(LAYOUT_JSON))
    shift = (s - G.CAD_SPACING) / 2.0
    per_arm = {'arm_mount_plate', 'rail_carriage_WW', 'rail_clamp_WHKD'}
    static = {'linear_axis_ZLW', 'rail_support_frame', 'rail_WS', 'triangle_support', 'triangle_side_plate'}
    if include_carrier:
        static.add('carrier_plate')
    for it in L['instances']:
        role = it['role']
        if role not in per_arm | static or it['is_assembly']:
            continue
        lo = np.array(it['bbox_min_mm']) / 1000.0
        hi = np.array(it['bbox_max_mm']) / 1000.0
        # CAD (X, Y_up, Z) -> W (Z - ARM_Z, X - MID_X, Y_up - ARM_Y_CAD)
        wlo = np.array([lo[2] - G.ARM_Z, lo[0] - G.MID_X, lo[1] - G.ARM_Y_CAD])
        whi = np.array([hi[2] - G.ARM_Z, hi[0] - G.MID_X, hi[1] - G.ARM_Y_CAD])
        if role in per_arm:
            cy = 0.5 * (wlo[1] + whi[1])
            d = shift if cy > 0 else -shift
            wlo[1] += d
            whi[1] += d
        name = 'struct_%s_%s' % (role, it['id'].replace('/', '_'))
        boxes.append((name, (wlo + whi) / 2, whi - wlo, 'struct'))
    return boxes


# ------------------------------------------------------------------------------------------------ main builder
def build(des, s, outdir, structure='cad', fingers='union', camera_pad=0.0, include_carrier=False, matrix=None, tool_inflate=0.0, geometry='real', proxy_clearance=0.0025, kin_timeout=0.25):
    os.makedirs(outdir, exist_ok=True)
    assets = os.path.join(HERE, 'assets')
    os.makedirs(assets, exist_ok=True)
    src = ET.parse(SRC_URDF).getroot()
    U = Urdf('dual_arm_cell')
    U.material('struct', (0.45, 0.47, 0.5, 1))
    U.material('tool_grey', (0.35, 0.4, 0.55, 1))
    U.material('finger', (0.54, 0.43, 0.23, 1))
    U.material('knife', (0.06, 0.48, 0.42, 1))
    U.material('camera', (0.1, 0.1, 0.1, 1))
    U.link('world')
    ee_links = {}
    all_links = []
    infl = tool_inflate

    # ---- structure (fixed to world) ----
    struct_links = []
    for name, c, size, col in structure_boxes(structure, s, include_carrier):
        lk = U.link(name)
        U.geom(lk, ('box', size), (0, 0, 0), (0, 0, 0), col)
        U.fixed(name + '_joint', 'world', name, c)
        struct_links.append(name)
        all_links.append(name)

    gripper_stl = build_gripper_stl(des, assets)
    hulls = build_finger_hulls(assets) if fingers.startswith('hull') else {}

    arm_info = {}
    for ai, arm in enumerate(G.ARMS):
        pre = arm + '_'
        ys = G.arm_y(ai, s)
        # copy the single arm chain
        for el in src:
            if el.tag == 'link':
                lk = ET.fromstring(ET.tostring(el))
                base_name = lk.get('name')
                lk.set('name', pre + base_name)
                if geometry == 'proxy':
                    for ch in list(lk):
                        if ch.tag == 'collision':
                            lk.remove(ch)
                U.root.append(lk)
                if geometry == 'proxy':
                    PX = des['proxy']
                    rr = proxy_clearance
                    if base_name == 'link2':
                        U.spheres(lk, PX['upper']['centres'], PX['upper']['r'] + rr, 'struct')
                    elif base_name == 'link3':
                        U.spheres(lk, PX['fore']['centres'], PX['fore']['r'] + rr, 'struct')
                    elif base_name == 'link6':
                        U.spheres(lk, PX['wrist']['centres'], [r + rr for r in PX['wrist']['radii']], 'struct')
                if lk.find('collision') is not None:
                    U.collision_links.append(lk.get('name'))
                all_links.append(lk.get('name'))
            elif el.tag == 'joint':
                j = ET.fromstring(ET.tostring(el))
                nm = el.get('name')
                j.set('name', pre + nm)
                j.find('parent').set('link', pre + j.find('parent').get('link'))
                j.find('child').set('link', pre + j.find('child').get('link'))
                if nm in LIM_DEG:
                    lim = j.find('limit')
                    lim.set('lower', f(LIM_DEG[nm][0] * D2R))
                    lim.set('upper', f(LIM_DEG[nm][1] * D2R))
                    lim.set('velocity', f(VEL))
                U.root.append(j)
        U.fixed(pre + 'world_joint', 'world', pre + 'base_link', (0.0, ys, 0.0))
        link6 = pre + 'link6'
        zero = (0, 0, 0)
        tool_links = [link6]
        if geometry == 'proxy':
            PX = des['proxy']
            rr = proxy_clearance
            lk = U.link(pre + 'gripper')
            U.spheres(lk, PX['gripper']['centres'], PX['gripper']['r'] + rr, 'tool_grey')
            U.fixed(pre + 'gripper_joint', link6, pre + 'gripper')
            tool_links.append(pre + 'gripper')
            if PX.get('fingers') and fingers != 'none':
                lk = U.link(pre + 'fingers')
                U.spheres(lk, PX['fingers']['centres'], [r + rr for r in PX['fingers']['radii']], 'finger')
                U.fixed(pre + 'fingers_joint', link6, pre + 'fingers')
                tool_links.append(pre + 'fingers')
            if PX['post']['centres']:
                lk = U.link(pre + 'knife_post')
                U.spheres(lk, PX['post']['centres'], PX['post']['r'] + rr, 'knife')
                U.fixed(pre + 'knife_post_joint', link6, pre + 'knife_post')
                tool_links.append(pre + 'knife_post')
            lk = U.link(pre + 'knife_clamp')
            U.spheres(lk, PX['clamp']['centres'], PX['clamp']['r'] + rr, 'knife')
            U.fixed(pre + 'knife_clamp_joint', link6, pre + 'knife_clamp')
            tool_links.append(pre + 'knife_clamp')
            lk = U.link(pre + 'knife_handpiece')
            U.spheres(lk, PX['hp']['centres'], PX['hp']['r'] + rr, 'knife')
            U.fixed(pre + 'knife_handpiece_joint', link6, pre + 'knife_handpiece')
            tool_links.append(pre + 'knife_handpiece')
            if des.get('camera'):
                c = des['camera']
                Tc = np.array(c['T_link6_camera'])
                Rc = Tc[:3, :3]
                size = np.array(c['body_size']) + 2 * PX['camera_pad']      # study: camera OBB padded by 3 mm only (box inflation != sphere inflation at corners)
                centre = Tc[:3, 3] + Rc[:, 2] * (-c['body_size'][2] / 2)
                lk = U.link(pre + 'camera')
                U.geom(lk, ('box', size), centre, G.rot_to_rpy(Rc), 'camera')
                U.fixed(pre + 'camera_joint', link6, pre + 'camera')
                tool_links.append(pre + 'camera')
                if PX.get('bracket'):
                    lk = U.link(pre + 'camera_bracket')
                    U.spheres(lk, PX['bracket']['centres'], PX['bracket']['r'] + rr, 'camera')
                    U.fixed(pre + 'camera_bracket_joint', link6, pre + 'camera_bracket')
                    tool_links.append(pre + 'camera_bracket')
            lk = U.link(pre + 'knife_blade')
            U.fixed(pre + 'knife_blade_joint', link6, pre + 'knife_blade')
            T = np.array(des['T_link6_tcp'])
            lk = U.link(pre + 'knife_tip')
            U.fixed(pre + 'knife_tip_joint', link6, pre + 'knife_tip', T[:3, 3], G.rot_to_rpy(T[:3, :3]))
            arm_info[arm] = dict(prefix=pre, y=ys, tool_links=tool_links, tip=pre + 'knife_tip', link6=link6,
                                 joints=[pre + 'joint%d' % i for i in range(1, 7)], base=pre + 'base_link')
            continue
        # gripper
        lk = U.link(pre + 'gripper')
        if gripper_stl:
            U.geom(lk, ('mesh', 'file://' + gripper_stl), zero, zero, 'tool_grey')
        else:
            gb = np.array(des['gripper']['box'])
            U.geom(lk, ('box', gb[:, 1] - gb[:, 0]), gb.mean(1), zero, 'tool_grey')
        U.fixed(pre + 'gripper_joint', link6, pre + 'gripper')
        tool_links.append(pre + 'gripper')
        # fingers
        if des.get('fingers') and fingers != 'none':
            lk = U.link(pre + 'fingers')
            if fingers.startswith('hull'):
                st = fingers.split('_')[1]
                for side in (-1, 1):
                    U.geom(lk, ('mesh', 'file://' + hulls[(side, st)]), zero, zero, 'finger')
            else:
                states = {'union': ('boxes_grasp', 'boxes_open'), 'grasp': ('boxes_grasp',), 'open': ('boxes_open',)}[fingers]
                for key in states:
                    for side in ('L', 'R'):
                        for b in des['fingers'][key][side]:
                            b = np.array(b)
                            U.geom(lk, ('box', (b[:, 1] - b[:, 0]) + 2 * infl), b.mean(1), zero, 'finger')
            U.fixed(pre + 'fingers_joint', link6, pre + 'fingers')
            tool_links.append(pre + 'fingers')
        # knife
        k = des['knife']
        if k.get('post'):
            lk = U.link(pre + 'knife_post')
            a, b = np.array(k['post']['frm']), np.array(k['post']['to'])
            rpy, L = G.z_to_vec_rpy(b - a)
            U.geom(lk, ('cylinder', k['post']['radius'] + infl, L), (a + b) / 2, rpy, 'knife')
            U.fixed(pre + 'knife_post_joint', link6, pre + 'knife_post')
            tool_links.append(pre + 'knife_post')
        lk = U.link(pre + 'knife_clamp')
        U.geom(lk, ('sphere', k['clamp']['radius'] + infl), k['clamp']['centre'], zero, 'knife')
        U.fixed(pre + 'knife_clamp_joint', link6, pre + 'knife_clamp')
        tool_links.append(pre + 'knife_clamp')
        lk = U.link(pre + 'knife_handpiece')
        U.capsule_parts(lk, k['handpiece']['frm'], k['handpiece']['to'], k['handpiece']['radius'] + infl, 'knife')
        U.fixed(pre + 'knife_handpiece_joint', link6, pre + 'knife_handpiece')
        tool_links.append(pre + 'knife_handpiece')
        # blade: visual only (the last 20 mm of the blade has no collision geometry; the handpiece capsule ends 20 mm behind the TCP)
        lk = U.link(pre + 'knife_blade')
        a, b = np.array(k['blade']['frm']), np.array(k['blade']['to'])
        if np.linalg.norm(b - a) > 1e-4:
            rpy, L = G.z_to_vec_rpy(b - a)
            U.geom(lk, ('cylinder', 0.0015, L), (a + b) / 2, rpy, 'knife', collision=False)
        else:
            U.geom(lk, ('sphere', 0.0015), a, zero, 'knife', collision=False)
        U.fixed(pre + 'knife_blade_joint', link6, pre + 'knife_blade')
        # camera + bracket
        if des.get('camera') and des['camera']:
            c = des['camera']
            Tc = np.array(c['T_link6_camera'])
            Rc = Tc[:3, :3]
            size = np.array(c['body_size']) + 2 * camera_pad
            centre = Tc[:3, 3] + Rc[:, 2] * (-c['body_size'][2] / 2)
            lk = U.link(pre + 'camera')
            U.geom(lk, ('box', size), centre, G.rot_to_rpy(Rc), 'camera')
            U.fixed(pre + 'camera_joint', link6, pre + 'camera')
            tool_links.append(pre + 'camera')
            lk = U.link(pre + 'camera_bracket')
            br = c['bracket']
            a, b = np.array(br['frm']), np.array(br['to'])
            rpy, L = G.z_to_vec_rpy(b - a)
            U.geom(lk, ('cylinder', br['radius'] + infl, L), (a + b) / 2, rpy, 'camera')
            U.fixed(pre + 'camera_bracket_joint', link6, pre + 'camera_bracket')
            tool_links.append(pre + 'camera_bracket')
        # TCP
        T = np.array(des['T_link6_tcp'])
        lk = U.link(pre + 'knife_tip')
        U.fixed(pre + 'knife_tip_joint', link6, pre + 'knife_tip', T[:3, 3], G.rot_to_rpy(T[:3, :3]))
        arm_info[arm] = dict(prefix=pre, y=ys, tool_links=tool_links, tip=pre + 'knife_tip', link6=link6,
                             joints=[pre + 'joint%d' % i for i in range(1, 7)], base=pre + 'base_link')
    # ---- URDF out ----
    urdf_path = os.path.join(outdir, 'dual_arm.urdf')
    U.write(urdf_path)

    # ---- SRDF ----
    disabled = {}          # frozenset({a, b}) -> reason

    def dis(a, b, why):
        if a != b:
            disabled.setdefault(frozenset((a, b)), why)

    # single-arm self-collision matrix from the existing Setup-Assistant SRDF (arm links only), prefixed per arm
    sroot = ET.parse(SRC_SRDF).getroot()
    for dc in sroot.findall('disable_collisions'):
        a, b = dc.get('link1'), dc.get('link2')
        if a in ARM_LINKS and b in ARM_LINKS:
            for arm in G.ARMS:
                dis(arm + '_' + a, arm + '_' + b, dc.get('reason'))
    if geometry == 'proxy':
        # the study checks only gripper/tool/finger/bracket/camera against upper arm + forearm (not the wrist chain); keep that
        for arm in G.ARMS:
            for l in ('link6',):
                for o in ('link2', 'link3'):
                    dis(arm + '_' + l, arm + '_' + o, 'Proxy-model: wrist not checked against the arm (as in the study)')
    # tool links are rigid with link6 and with each other
    for arm in G.ARMS:
        tl = arm_info[arm]['tool_links']
        for a, b in itertools.combinations(tl, 2):
            dis(a, b, 'Adjacent')
        dis(arm + '_knife_tip', arm + '_link6', 'Adjacent')
        dis(arm + '_knife_blade', arm + '_link6', 'Adjacent')
    # structure links are static with respect to each other and to world
    for a, b in itertools.combinations(struct_links, 2):
        dis(a, b, 'Adjacent')
    # arm base mounted on its own mount plate / carriage
    for arm in G.ARMS:
        for sl in struct_links:
            if any(t in sl for t in ('arm_mount_plate', 'rail_carriage_WW', 'rail_clamp_WHKD')) or sl == 'study_box':
                for l in (arm + '_base_link',):
                    dis(l, sl, 'Adjacent')
    pre_matrix_pairs = len(disabled)
    matrix_info = None
    if matrix:
        M = json.load(open(matrix))
        N = M['n_samples']
        matrix_info = dict(file=matrix, n_samples=N, never=0, default=0)
        cl = [l for l in U.collision_links]
        for a, b in itertools.combinations(sorted(cl), 2):
            key = frozenset((a, b))
            if key in disabled:
                continue
            cnt = M['counts'].get('|'.join(sorted((a, b))), 0)
            if cnt == 0:
                dis(a, b, 'Never')
                matrix_info['never'] += 1
            elif cnt >= 0.95 * N:
                dis(a, b, 'Default')
                matrix_info['default'] += 1
    srdf = ET.Element('robot', name='dual_arm_cell')
    for arm in G.ARMS:
        g = ET.SubElement(srdf, 'group', name=arm)
        ET.SubElement(g, 'chain', base_link=arm + '_base_link', tip_link=arm + '_knife_tip')
    g = ET.SubElement(srdf, 'group', name='both')
    for arm in G.ARMS:
        ET.SubElement(g, 'group', name=arm)
    for arm in G.ARMS:
        g = ET.SubElement(srdf, 'group', name=arm + '_tool')
        for l in arm_info[arm]['tool_links'][1:] + [arm + '_knife_blade']:
            ET.SubElement(g, 'link', name=l)
    for gn, js in (('arm1', arm_info['arm1']['joints']), ('arm2', arm_info['arm2']['joints']), ('both', arm_info['arm1']['joints'] + arm_info['arm2']['joints'])):
        gs = ET.SubElement(srdf, 'group_state', name='neutral', group=gn)
        for j in js:
            ET.SubElement(gs, 'joint', name=j, value='0')
    for arm in G.ARMS:
        ET.SubElement(srdf, 'end_effector', name=arm + '_ee', parent_link=arm + '_link6', group=arm + '_tool', parent_group=arm)
    for key in sorted(disabled, key=lambda k: tuple(sorted(k))):
        a, b = sorted(key)
        ET.SubElement(srdf, 'disable_collisions', link1=a, link2=b, reason=disabled[key])
    ET.indent(srdf, space='  ')
    ET.ElementTree(srdf).write(os.path.join(outdir, 'dual_arm.srdf'), xml_declaration=True, encoding='utf-8')

    # ---- yaml configs ----
    kin = {a: dict(kinematics_solver='kdl_kinematics_plugin/KDLKinematicsPlugin', kinematics_solver_timeout=kin_timeout, position_only_ik=False) for a in G.ARMS}
    yaml.safe_dump(kin, open(os.path.join(outdir, 'kinematics.yaml'), 'w'), sort_keys=False)
    jl = {'joint_limits': {}}
    for arm in G.ARMS:
        for i in range(1, 7):
            jl['joint_limits']['%s_joint%d' % (arm, i)] = dict(has_velocity_limits=True, max_velocity=VEL, has_acceleration_limits=True, max_acceleration=ACC,
                                                              has_deceleration_limits=True, max_deceleration=-ACC)
    yaml.safe_dump(jl, open(os.path.join(outdir, 'joint_limits.yaml'), 'w'), sort_keys=False)
    # Pilz Cartesian limits (same values as the repo config, scaled for the contact scalings)
    yaml.safe_dump({'cartesian_limits': dict(max_trans_vel=0.01817, max_trans_acc=0.0524, max_trans_dec=-0.0524, max_rot_vel=0.0727)},
                   open(os.path.join(outdir, 'pilz_cartesian_limits.yaml'), 'w'), sort_keys=False)

    cfgs = {
        'RRTConnectkConfigDefault': dict(type='geometric::RRTConnect', range=0.0),
        'RRTkConfigDefault': dict(type='geometric::RRT', range=0.0, goal_bias=0.05),
        'RRTstarkConfigDefault': dict(type='geometric::RRTstar', range=0.0, goal_bias=0.05, delay_collision_checking=1),
        'BiTRRTkConfigDefault': dict(type='geometric::BiTRRT', range=0.0, temp_change_factor=0.1, init_temperature=100, frountier_threshold=0.0,
                                     frountier_node_ratio=0.1, cost_threshold=1e300),
        'LBKPIECEkConfigDefault': dict(type='geometric::LBKPIECE', range=0.0, border_fraction=0.9, min_valid_path_fraction=0.5),
        'PRMkConfigDefault': dict(type='geometric::PRM', max_nearest_neighbors=10),
    }
    ompl = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        # Humble request-adapter chain (post-processing runs last-to-first, so time parameterization is listed last)
        'request_adapters': ' '.join(['default_planner_request_adapters/ResolveConstraintFrames',
                                      'default_planner_request_adapters/FixWorkspaceBounds',
                                      'default_planner_request_adapters/FixStartStateBounds',
                                      'default_planner_request_adapters/FixStartStateCollision',
                                      'default_planner_request_adapters/FixStartStatePathConstraints',
                                      'default_planner_request_adapters/AddTimeOptimalParameterization']),
        'start_state_max_bounds_error': 0.1,
        'planner_configs': cfgs,
    }
    for gname in ('arm1', 'arm2', 'both'):
        ompl[gname] = dict(planner_configs=list(cfgs.keys()), default_planner_config='RRTConnectkConfigDefault')
    yaml.safe_dump(ompl, open(os.path.join(outdir, 'ompl_planning.yaml'), 'w'), sort_keys=False)
    pilz = {'planning_plugin': 'pilz_industrial_motion_planner/CommandPlanner',
            'request_adapters': ' '.join(['default_planner_request_adapters/FixWorkspaceBounds', 'default_planner_request_adapters/FixStartStateBounds',
                                          'default_planner_request_adapters/FixStartStateCollision']),
            'default_planner_config': 'PTP',
            'capabilities': 'pilz_industrial_motion_planner/MoveGroupSequenceAction pilz_industrial_motion_planner/MoveGroupSequenceService'}
    yaml.safe_dump(pilz, open(os.path.join(outdir, 'pilz_planning.yaml'), 'w'), sort_keys=False)
    open(os.path.join(outdir, 'no_controllers.yaml'), 'w').write('{}\n')

    meta = dict(schema='cell_planner_check/cell_meta/v1', design=des['name'], kin_timeout=kin_timeout, s=s, structure=structure, fingers=fingers, geometry=geometry, proxy_clearance=proxy_clearance, camera_pad=camera_pad,
                tool_inflate=infl, frame='world', arms=arm_info, T_link6_tcp=des['T_link6_tcp'],
                joint_limits_deg=LIM_DEG, collision_links=sorted(U.collision_links), structure_links=struct_links,
                n_disabled_pairs=len(disabled), n_pre_matrix_pairs=pre_matrix_pairs, matrix=matrix_info,
                files={k: os.path.join(outdir, v) for k, v in dict(urdf='dual_arm.urdf', srdf='dual_arm.srdf', kinematics='kinematics.yaml',
                                                                  joint_limits='joint_limits.yaml', ompl='ompl_planning.yaml', pilz='pilz_planning.yaml').items()})
    json.dump(meta, open(os.path.join(outdir, 'cell_meta.json'), 'w'), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--design', required=True)
    ap.add_argument('--s', type=float, default=0.355, help='arm spacing (m)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--structure', default='cad', choices=['cad', 'study', 'none'])
    ap.add_argument('--fingers', default='union', choices=['union', 'grasp', 'open', 'hull_grasp', 'hull_open', 'none'])
    ap.add_argument('--camera-pad-mm', type=float, default=0.0)
    ap.add_argument('--tool-inflate-mm', type=float, default=0.0, help='inflate tool primitives (post, handpiece, clamp, fingers, bracket) by this many mm')
    ap.add_argument('--carrier-plates', action='store_true')
    ap.add_argument('--kin-timeout', type=float, default=0.25, help='KDL kinematics_solver_timeout (s); 0.25 = the repo config')
    ap.add_argument('--geometry', default='real', choices=['real', 'proxy'], help="real = URDF primitives + SMC mesh + finger boxes; proxy = rebuild the offline study's sphere/box collision model inside MoveIt")
    ap.add_argument('--proxy-clearance-mm', type=float, default=2.5, help='radius inflation of every proxy sphere / box half-extent (study clearances: 2 mm carton and self, 5 mm arm-arm; 2.5 mm gives 5 mm arm-arm and 2.5 mm carton; the camera box keeps the 3 mm pad of the study only)')
    ap.add_argument('--matrix', default=None, help='collision_matrix.json from sample_collisions.py (adds Never / Default pairs)')
    a = ap.parse_args()
    des = json.load(open(a.design if os.path.exists(a.design) else os.path.join(HERE, a.design)))
    meta = build(des, a.s, a.out, a.structure, a.fingers, a.camera_pad_mm / 1000.0, a.carrier_plates, a.matrix, a.tool_inflate_mm / 1000.0, a.geometry, a.proxy_clearance_mm / 1000.0, a.kin_timeout)
    print(json.dumps({k: meta[k] for k in ('design', 's', 'structure', 'fingers', 'n_disabled_pairs')}), '->', a.out)


if __name__ == '__main__':
    main()
