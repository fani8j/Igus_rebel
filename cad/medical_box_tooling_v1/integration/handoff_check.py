#!/usr/bin/env python3
"""Integration check: LEFT suction tool + RIGHT LEHF gripper tool around one carton at the shelf handoff.

Read-only on ../common, the suction tool folder, ../gripper_tool. Writes data/handoff_check*.json only
(renders/STEP are made by render_handoff.py / export_step.py, which import this module).

Suction tool selection: default suction_tool_v5 (the canonical left tool); `--tool suction_tool` (or env SUCTION_TOOL=suction_tool) runs the v3 tool.
Output: data/handoff_check.json for the default tool, data/handoff_check_<tool>.json otherwise (suction_tool -> handoff_check_v3.json).

Run:  PYTHONPATH="<scratch>/pylib" "<cad venv>/bin/python" handoff_check.py [--quick]
python-fcl is not in the cad venv; it was pip-installed with --target into
/media/farhan/My Passport/igus_nvidia/scratch_integration/pylib (see README).
Units mm, shelf frame W of common/handoff.py.
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # medical_box_tooling_v1
sys.path.insert(0, os.path.join(ROOT, 'common'))
sys.path.insert(0, '/media/farhan/My Passport/igus_nvidia/scratch_integration/pylib')
import fcl                                          # noqa: E402
import trimesh                                      # noqa: E402
import params as P                                  # noqa: E402  (common/params.py, read-only)
import handoff as H                                 # noqa: E402  (common/handoff.py, read-only)

DEFAULT_SUCTION = 'suction_tool_v5'


def _suction_name():
    for i, a in enumerate(sys.argv):
        if a == '--tool' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith('--tool='):
            return a.split('=', 1)[1]
    return os.environ.get('SUCTION_TOOL', DEFAULT_SUCTION)


SUCTION_NAME = _suction_name()
SUCTION = os.path.join(ROOT, SUCTION_NAME)
OUT_SUFFIX = '' if SUCTION_NAME == DEFAULT_SUCTION else '_' + {'suction_tool': 'v3'}.get(SUCTION_NAME, SUCTION_NAME)
GRIPPER = os.path.join(ROOT, 'gripper_tool')
REBEL_MESH = '/home/farhan/orca/workspaces/igus_rebel_ros2/box-3d/src/igus_rebel_description/meshes/rebel_d00617809'
CLR = P.CLEARANCE_MIN
G_ACC = 9.81
STROKE_OPEN = 24.0            # per finger, gap 56
LINK6_MESH_OFF = np.array([1.3449814287804, 0.0, -0.5195])       # igus_rebel2.urdf link6/link5 <visual><origin>
LINK5_TO_LINK6_Z = 76.8                                           # joint6 origin 0 0 0.0768 (joint6 = 0)
PAD_HALF_SPAN = 75.4          # suction pad outer lip edge from the middle pad centre (suction README)


# ---------------------------------------------------------------- small helpers
def T_of(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _load_stl(path):
    m = trimesh.load(path, force='mesh')
    m.apply_scale(1000.0)                    # metres -> mm
    return m


def _fcl_convex(m):
    h = m if m.is_convex else m.convex_hull
    f = np.hstack([np.full((len(h.faces), 1), 3), h.faces]).ravel().astype(np.int64)
    return fcl.Convex(np.asarray(h.vertices, dtype=float), len(h.faces), f)


def _fcl_bvh(m):
    b = fcl.BVHModel()
    b.beginModel(len(m.vertices), len(m.faces))
    b.addSubModel(np.asarray(m.vertices, dtype=float), np.asarray(m.faces))
    b.endModel()
    return b


class Part:
    """One rigid piece: collision geometry (fcl) + render mesh, local frame = tool/robot link frame."""

    def __init__(self, name, geom, verts, local=None, render=None, kind='convex'):
        self.name = name
        self.obj = fcl.CollisionObject(geom, fcl.Transform())
        self.local = np.eye(4) if local is None else local
        v = np.asarray(verts, dtype=float)
        lo, hi = v.min(0), v.max(0)
        self.corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self.render = render                  # trimesh in the same local frame (or None)
        self.kind = kind
        self.aabb = None
        self.world = np.eye(4)

    def place(self, Tw):
        self.world = Tw @ self.local
        self.obj.setTransform(fcl.Transform(self.world[:3, :3], self.world[:3, 3]))
        c = (self.world @ self.corners.T).T[:, :3]
        self.aabb = (c.min(0), c.max(0))


class Group:
    def __init__(self, name):
        self.name = name
        self.parts = []

    def add(self, p):
        self.parts.append(p)

    def place(self, Tw):
        for p in self.parts:
            p.place(Tw)

    def bounds(self):
        lo = np.min([p.aabb[0] for p in self.parts], axis=0)
        hi = np.max([p.aabb[1] for p in self.parts], axis=0)
        return lo, hi


def box_group(name, size, centre_T, kind_name=None):
    g = Group(name)
    sx, sy, sz = size
    p = Part(kind_name or name, fcl.Box(sx, sy, sz), [[-sx / 2, -sy / 2, -sz / 2], [sx / 2, sy / 2, sz / 2]], kind='box')
    p.render = trimesh.creation.box(size)
    g.add(p)
    g.place(centre_T)
    return g


def aabb_gap(a, b):
    d = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(d))


def pair_dist(pa, pb):
    """Signed-ish distance: >0 exact separation, <=0 overlap (negative = fcl max penetration depth, 0 if unknown)."""
    rq = fcl.CollisionRequest(enable_contact=True, num_max_contacts=8)
    rs = fcl.CollisionResult()
    if fcl.collide(pa.obj, pb.obj, rq, rs) > 0:
        dep = max((c.penetration_depth for c in rs.contacts), default=0.0)
        return -abs(dep)
    r = fcl.DistanceResult()
    d = fcl.distance(pa.obj, pb.obj, fcl.DistanceRequest(), r)
    return float(d) if d >= 0 else 0.0


def group_dist(ga, gb, skip=None, cutoff=None):
    """Min distance over part pairs. skip(partA, partB) -> True excludes the pair. Returns (d, nameA, nameB)."""
    best = (1e9, None, None)
    for pa in ga.parts:
        for pb in gb.parts:
            if skip is not None and skip(pa, pb):
                continue
            g = aabb_gap(pa.aabb, pb.aabb)
            if g >= best[0] and g > 0:
                continue                        # cannot improve on current best
            d = pair_dist(pa, pb)
            if d < best[0]:
                best = (d, pa.name, pb.name)
    return best


# ---------------------------------------------------------------- model construction
def load_suction():
    g = Group('S')
    man = json.load(open(os.path.join(SUCTION, 'meshes', 'collision', 'manifest.json')))
    import glob
    files = sorted(glob.glob(os.path.join(SUCTION, 'meshes', 'collision', '*.stl')))
    assert len(files) == len(man['pieces']), (len(files), len(man['pieces']))
    for f in files:
        m = _load_stl(f)
        g.add(Part(os.path.basename(f)[:-4], _fcl_convex(m), m.vertices, render=None))
    # reserved ejector volume (suction v1/v2 only): convex hull of step/ejector_envelope.step if the tool still has one.
    # Suction v3 uses a remote ejector, so the file is gone and nothing is added.
    env_path = os.path.join(SUCTION, 'step', 'ejector_envelope.step')
    if os.path.exists(env_path):
        import cadquery as cq
        env = cq.importers.importStep(env_path).val()
        vs, _ = env.tessellate(0.1)
        ej = trimesh.convex.convex_hull(np.array([[v.x, v.y, v.z] for v in vs]))
        g.add(Part('ejector_envelope', _fcl_convex(ej), ej.vertices))
    # render: visual meshes
    g.visual = [(os.path.basename(f)[:-4], _load_stl(f)) for f in sorted(glob.glob(os.path.join(SUCTION, 'meshes', 'visual', '*.stl')))]
    return g


def load_gripper():
    g = Group('G')
    import glob
    cdir = os.path.join(GRIPPER, 'meshes', 'collision')
    for f in sorted(glob.glob(os.path.join(cdir, '*.stl'))):
        n = os.path.basename(f)[:-4]
        if 'ledge' in n:
            continue                     # stale file from an earlier design, not referenced by the URDF
        m = _load_stl(f)
        local = np.eye(4)
        g.add(Part(n, _fcl_convex(m), m.vertices, local=local))
    # finger local transforms are set by set_opening(): z0 = 150 joint origin
    g.finger_parts = {'L': [p for p in g.parts if p.name.startswith('finger_') and ('left' in p.name)],
                      'R': [p for p in g.parts if p.name.startswith('finger_') and ('right' in p.name)]}
    assert len(g.finger_parts['L']) == 6 and len(g.finger_parts['R']) == 6, {k: [p.name for p in v] for k, v in g.finger_parts.items()}
    g.visual_base = [(os.path.basename(f)[:-4], _load_stl(f)) for f in sorted(glob.glob(os.path.join(GRIPPER, 'meshes', 'visual', 'base_*.stl')))]
    g.visual_fingers = {'L': [(os.path.basename(f)[:-4], _load_stl(f)) for f in sorted(glob.glob(os.path.join(GRIPPER, 'meshes', 'visual', '*_left.stl')))],
                        'R': [(os.path.basename(f)[:-4], _load_stl(f)) for f in sorted(glob.glob(os.path.join(GRIPPER, 'meshes', 'visual', '*_right.stl')))]}
    g.opening = None
    return g


def set_opening(g, d):
    """Per-finger travel d (mm): pad gap = 8 + 2 d. Finger meshes are in the joint frame (0,0,150), left moves -x, right +x."""
    g.opening = d
    for side, sgn in (('L', -1.0), ('R', 1.0)):
        for p in g.finger_parts[side]:
            p.local = T_of(t=[sgn * d, 0.0, 150.0])


def load_wrist(name):
    """ReBeL link5 + link6 visual meshes as placed in igus_rebel2.urdf, in the link6 frame (joint6 = 0)."""
    g = Group(name)
    for lk, dz in (('link6', 0.0), ('link5', -LINK5_TO_LINK6_Z)):
        m = trimesh.load(os.path.join(REBEL_MESH, lk + '.dae'), force='mesh')
        m.apply_scale(1000.0)
        loc = T_of(t=LINK6_MESH_OFF + np.array([0, 0, dz]))
        g.add(Part(lk, _fcl_bvh(m), m.vertices, local=loc, render=m, kind='bvh'))
    return g


# ---------------------------------------------------------------- URDF assertions
def urdf_assert():
    ns = {}
    out = {}

    def joints(path):
        root = ET.parse(path).getroot()
        d = {}
        for j in root.iter('joint'):
            o = j.find('origin')
            nm = j.get('name').replace('${prefix}', '')
            if o is not None:
                d[nm] = np.array([float(x) for x in o.get('xyz').split()]) * 1000.0
        return d, root
    sj, sroot = joints(os.path.join(SUCTION, 'urdf', 'left_suction_tool.urdf.xacro'))
    gj, groot = joints(os.path.join(GRIPPER, 'urdf', 'right_gripper_tool.urdf.xacro'))
    checks = [
        ('suction_tcp z', sj['suction_tcp_joint'][2], H.SUCTION_TCP_Z),
        ('suction_tcp_compressed z', sj['suction_tcp_compressed_joint'][2], H.SUCTION_TCP_COMPRESSED_Z),
        ('gripper_tcp z', gj['gripper_tcp_joint'][2], H.GRIPPER_TCP_Z),
        ('top_stop_frame z', gj['top_stop_frame_joint'][2], H.GRIPPER_STOP_Z),
        ('finger joint z', gj['finger_left_joint'][2], 150.0),
        ('finger_right joint z', gj['finger_right_joint'][2], 150.0),
    ]
    for k in ('suction_tcp_joint', 'suction_tcp_compressed_joint'):
        checks += [(k + ' x', sj[k][0], 0.0), (k + ' y', sj[k][1], 0.0)]
    for k in ('gripper_tcp_joint', 'top_stop_frame_joint'):
        checks += [(k + ' x', gj[k][0], 0.0), (k + ' y', gj[k][1], 0.0)]
    res = []
    for name, urdf_v, const_v in checks:
        res.append(dict(name=name, urdf=round(float(urdf_v), 4), handoff_py=const_v, delta=round(float(urdf_v - const_v), 4)))
        assert abs(urdf_v - const_v) < 0.1, (name, urdf_v, const_v)
    out['urdf_vs_handoff'] = res
    # mesh-derived confirmations
    import glob
    lip = max(_load_stl(f).bounds[1][2] for f in glob.glob(os.path.join(SUCTION, 'meshes', 'collision', 'pad_[LMR]_0*.stl')))
    stop = _load_stl(os.path.join(GRIPPER, 'meshes', 'collision', 'base_top_stop_bar.stl')).bounds[1][2]
    pad = _load_stl(os.path.join(GRIPPER, 'meshes', 'collision', 'finger_pad_left.stl')).bounds
    pad_c = 150.0 + (pad[0][2] + pad[1][2]) / 2
    mesh = [dict(name='suction pad lip z (mesh max)', mesh=round(float(lip), 3), expected=H.SUCTION_TCP_Z),
            dict(name='gripper top-stop face z (mesh max)', mesh=round(float(stop), 3), expected=H.GRIPPER_STOP_Z),
            dict(name='gripper pad centre z (mesh)', mesh=round(float(pad_c), 3), expected=H.GRIPPER_TCP_Z),
            dict(name='finger pad crest |x| closed (mesh)', mesh=round(float(abs(pad[1][0])), 3), expected=4.0),
            dict(name='finger pad y span (mesh)', mesh=round(float(pad[1][1] - pad[0][1]), 3), expected=50.0)]
    for m in mesh:
        assert abs(m['mesh'] - m['expected']) < 0.1, m
    out['mesh_vs_handoff'] = mesh
    return out


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        self.S = load_suction()
        self.SW = load_wrist('SW')
        self.G = load_gripper()
        self.GW = load_wrist('GW')
        self.B = None
        self.N = []
        self.F = None

    def configure(self, box_id, protrusion=0.0, neighbour_t=None, gripper_opening=None, lift=0.0,
                  pad_row_z=None, suction_yaw=0.0, gap=None):
        b = P.BOXES[box_id]
        self.box_id = box_id
        Ts = H.suction_link6_pose(box_id, pad_row_centre_z=pad_row_z)
        if suction_yaw:
            # rotate the tool about its own z; the TCP (on the z axis) stays on the spine line
            Ts = Ts @ T_of(R=rotz(suction_yaw))
        Tg = H.gripper_link6_pose(box_id, lift=lift)
        set_opening(self.G, (b['t'] - 8.0) / 2.0 if gripper_opening is None else gripper_opening)
        self.Ts, self.Tg = Ts, Tg
        self.S.place(Ts)
        self.SW.place(Ts)
        self.G.place(Tg)
        self.GW.place(Tg)
        Tb = H.box_pose(box_id)
        self.B = box_group('B', (b['t'], b['l'], b['h']), Tb)
        tn = b['t'] if neighbour_t is None else neighbour_t
        self.N = []
        for i, Tn in enumerate(H.neighbour_poses(box_id, protrusion, neighbour_t)):
            self.N.append(box_group('N%d' % (i + 1), (tn, b['l'], b['h']), Tn))
        self.F = box_group('F', (3000.0, 3000.0, 20.0), T_of(t=[0, 0, -10.0]))     # floor z = 0 plane (top face)
        self.groups = {'S': self.S, 'SW': self.SW, 'G': self.G, 'GW': self.GW, 'B': self.B, 'F': self.F}
        for g in self.N:
            self.groups[g.name] = g

    # ---- which pairs are expected contacts
    def expected(self, ga, gb, pa, pb, pads_on_box=True):
        a, b = ga.name, gb.name
        pair = {a, b}
        if pair <= {'B', 'N1', 'N2', 'F'}:
            return True                                     # carton stack on the floor, neighbours touching
        if pair == {'S', 'B'}:
            part = pa if a == 'S' else pb
            return part.name.startswith('pad_')
        if pair == {'G', 'B'}:
            part = pa if a == 'G' else pb
            if part.name == 'base_top_stop_bar':
                return True
            # closed on the carton: the TPU pad ribs and the finger block 0.4 mm behind the rib crests are the contact set
            return pads_on_box and (part.name.startswith('finger_pad_') or part.name.endswith('_block'))
        return False

    def all_pairs(self, pads_on_box=True, ignore=()):
        """Returns dict: 'A-B' -> dict(d_unexpected, at=(partA,partB), d_expected, n_expected_contact_parts...)."""
        names = ['S', 'SW', 'G', 'GW', 'B', 'N1', 'N2', 'F']
        arm = {'S': 'L', 'SW': 'L', 'G': 'R', 'GW': 'R'}
        res = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if arm.get(a) is not None and arm.get(a) == arm.get(b):
                    continue                                 # tool vs own wrist: mounted together
                if {a, b} <= {'B', 'N1', 'N2', 'F'}:
                    continue                                 # handled by `expected`, not reported as a pair
                ga, gb = self.groups[a], self.groups[b]
                exp_pairs, unexp = [], (1e9, None, None)
                for pa in ga.parts:
                    if pa.name in ignore:
                        continue
                    for pb in gb.parts:
                        if pb.name in ignore:
                            continue
                        e = self.expected(ga, gb, pa, pb, pads_on_box)
                        if not e and aabb_gap(pa.aabb, pb.aabb) >= unexp[0]:
                            continue
                        if e and aabb_gap(pa.aabb, pb.aabb) > 20.0:
                            continue
                        d = pair_dist(pa, pb)
                        if e:
                            exp_pairs.append((d, pa.name, pb.name))
                        elif d < unexp[0]:
                            unexp = (d, pa.name, pb.name)
                r = dict(d=None if unexp[1] is None else round(unexp[0], 3), at=None if unexp[1] is None else [unexp[1], unexp[2]])
                if exp_pairs:
                    dm = min(exp_pairs)
                    det = {}
                    for e in exp_pairs:
                        k = e[1] if a in ('S', 'G') else e[2]
                        det[k] = min(det.get(k, 1e9), round(e[0], 2))
                    r['expected_contact'] = dict(min=round(dm[0], 3), at=[dm[1], dm[2]],
                                                 n_parts_in_contact=int(sum(1 for e in exp_pairs if e[0] <= 0.05)),
                                                 per_part={k: v for k, v in sorted(det.items(), key=lambda kv: kv[1])[:8]})
                res[a + '-' + b] = r
        return res


def summarize(pairs):
    """overall min unexpected distance, cross-arm min, worst pair."""
    best = (1e9, None)
    cross = (1e9, None)
    for k, v in pairs.items():
        if v['d'] is None:
            continue
        if v['d'] < best[0]:
            best = (v['d'], k)
        a, b = k.split('-')
        if a in ('S', 'SW') and b in ('G', 'GW') and v['d'] < cross[0]:
            cross = (v['d'], k)
    return dict(min_unexpected=round(best[0], 3), min_pair=best[1], at=pairs[best[1]]['at'],
                min_cross_arm=round(cross[0], 3), cross_pair=cross[1], cross_at=pairs[cross[1]]['at'])


def neighbour_cases(b):
    other = 16.0 if b['t'] == 12.0 else 12.0
    return {'a_flush': dict(protrusion=0.0), 'b_proud15': dict(protrusion=P.NEIGHBOUR_PROTRUSION),
            'c_proud15_t%d' % other: dict(protrusion=P.NEIGHBOUR_PROTRUSION, neighbour_t=other)}


# ---------------------------------------------------------------- checks
def check1(sc, yaw=0.0, pad_row_z=None):
    out = {}
    for bid in P.BOXES:
        b = P.BOXES[bid]
        out[bid] = {}
        for cname, kw in neighbour_cases(b).items():
            sc.configure(bid, suction_yaw=yaw, pad_row_z=pad_row_z, **kw)
            pairs = sc.all_pairs()
            s = summarize(pairs)
            s['no_ejector'] = summarize(sc.all_pairs(ignore=('ejector_envelope',)))
            s['pairs'] = pairs
            out[bid][cname] = s
    return out


def check2(sc, step=2.0, only_case=None):
    out = {}
    lifts = np.arange(150.0, -1e-9, -step)
    for bid in P.BOXES:
        b = P.BOXES[bid]
        out[bid] = {}
        for cname, kw in neighbour_cases(b).items():
            first3 = first0 = None
            dmin, lmin, who = 1e9, None, None
            curve = []
            pmin = {}
            for L in lifts:
                sc.configure(bid, gripper_opening=STROKE_OPEN, lift=float(L), **kw)
                pairs = sc.all_pairs(pads_on_box=False)
                s = summarize(pairs)
                d = s['min_unexpected']
                curve.append((float(L), d))
                for k, v in pairs.items():
                    if v['d'] is not None:
                        pmin[k] = min(pmin.get(k, 1e9), v['d'])
                if d < CLR and first3 is None:
                    first3 = float(L)
                if d <= 0 and first0 is None:
                    first0 = float(L)
                if d < dmin:
                    dmin, lmin, who = d, float(L), [s['min_pair'], s['at']]
            out[bid][cname] = dict(first_lift_below_3mm=first3, first_lift_contact=first0, min_distance=round(dmin, 3),
                                   at_lift=lmin, pair=who, pair_min={k: round(v, 2) for k, v in pmin.items()}, curve=[[l, round(d, 2)] for l, d in curve[::5]])
    return out


def top_extent_z(sc):
    """suction tool top end (max z_w incl. pads, bracket, camera arm, ejector envelope) and its parts."""
    parts = [p for p in sc.S.parts if p.name != 'ejector_envelope']
    top = max(parts, key=lambda p: p.aabb[1][2])
    return float(top.aabb[1][2]), top.name, float(min(p.aabb[0][2] for p in parts))


def clearance_metrics(sc):
    """3-D minimum distances suction (tool, wrist) vs gripper (tool, wrist), incl. nearest parts."""
    r = {}
    for a, b in (('S', 'G'), ('S', 'GW'), ('SW', 'G'), ('SW', 'GW')):
        d, na, nb = group_dist(sc.groups[a], sc.groups[b])
        r[a + '-' + b] = dict(d=round(d, 3), at=[na, nb])
    return r


def check3(sc, quick=False):
    out = {}
    for bid in P.BOXES:
        b = P.BOXES[bid]
        h = b['h']
        entry = {}
        sc.configure(bid)
        ztop, ptop, zbot = top_extent_z(sc)
        glo, ghi = sc.G.bounds()
        gwlo, _ = sc.GW.bounds()
        # lowest gripper parts (finger tips / pads) and the lowest part of the stop yoke, body
        low_parts = sorted(((p.aabb[0][2], p.name) for p in sc.G.parts))[:3]
        entry['default_zc'] = h / 2.0
        entry['suction_top_z_excl_ejector'] = round(ztop, 2)
        entry['suction_top_z_incl_ejector_envelope'] = round(float(sc.S.bounds()[1][2]), 2)
        entry['suction_bottom_z_excl_ejector'] = round(zbot, 2)
        entry['suction_top_part'] = ptop
        entry['gripper_lowest_z'] = round(float(glo[2]), 2)
        entry['gripper_lowest_parts'] = [[round(float(z), 1), n] for z, n in low_parts]
        entry['gripper_wrist_lowest_z'] = round(float(gwlo[2]), 2)
        entry['z_gap_suction_top_to_gripper_lowest'] = round(float(glo[2] - ztop), 2)     # <0: overlap in height (not in 3-D)
        entry['z_gap_suction_top_to_gripper_wrist_lowest'] = round(float(gwlo[2] - ztop), 2)
        # vertical clearance restricted to the gripper parts that overlap suction in plan view is the true 3-D distance:
        entry['dist_default'] = clearance_metrics(sc)
        dm = min(v['d'] for v in entry['dist_default'].values())
        entry['min_clearance_default'] = dm
        # search pad_row_centre_z with >= 25 mm margin of the 150.8 mm pad span to both spine ends
        lo, hi = 25.0 + PAD_HALF_SPAN, h - 25.0 - PAD_HALF_SPAN
        zs = np.arange(math.ceil(lo), math.floor(hi) + 1e-9, 1.0 if not quick else 4.0)
        scan = []
        for zc in zs:
            sc.configure(bid, pad_row_z=float(zc))
            cm = clearance_metrics(sc)
            scan.append((float(zc), min(v['d'] for v in cm.values())))
        best = max(scan, key=lambda t: t[1])
        entry['search_range_zc'] = [round(lo, 1), round(hi, 1)]
        entry['search'] = [[z, round(d, 2)] for z, d in scan]
        entry['best_zc'] = best[0]
        entry['best_min_clearance'] = round(best[1], 2)
        entry['clearance_ge_10_default'] = dm >= 10.0
        out[bid] = entry
    return out


def check5(sc):
    """Mass / CoG from the URDF inertials, + carton 0.15 kg at the handoff, static moment about the flange face centre."""
    def inertial(path, link):
        root = ET.parse(path).getroot()
        for l in root.iter('link'):
            if l.get('name', '').replace('${prefix}', '') == link:
                i = l.find('inertial')
                return float(i.find('mass').get('value')), np.array([float(x) for x in i.find('origin').get('xyz').split()]) * 1000.0
        raise KeyError(link)
    sp = os.path.join(SUCTION, 'urdf', 'left_suction_tool.urdf.xacro')
    gp = os.path.join(GRIPPER, 'urdf', 'right_gripper_tool.urdf.xacro')
    ms, cs = inertial(sp, 'suction_tool')
    parts_g = [inertial(gp, 'gripper_base')]
    out = {}
    flange = np.array([0, 0, P.FLANGE_Z])
    for bid in P.BOXES:
        b = P.BOXES[bid]
        t = b['t']
        d = (t - 8.0) / 2.0
        gl = inertial(gp, 'finger_left')
        gr = inertial(gp, 'finger_right')
        gparts = [inertial(gp, 'gripper_base'), (gl[0], gl[1] + np.array([-d, 0, 150.0])), (gr[0], gr[1] + np.array([d, 0, 150.0]))]
        mg = sum(m for m, _ in gparts)
        cg = sum(m * c for m, c in gparts) / mg
        Tb = H.box_pose(bid)[:3, 3]
        row = {}
        for arm, m, c, Tl in (('suction', ms, cs, H.suction_link6_pose(bid)), ('gripper', mg, cg, H.gripper_link6_pose(bid))):
            R, t0 = Tl[:3, :3], Tl[:3, 3]
            box_l = R.T @ (Tb - t0)                           # carton centre in link6
            gdir = R.T @ np.array([0, 0, -1.0])                # gravity direction in link6
            mt = m + P.BOX_MASS_MAX
            ct = (m * c + P.BOX_MASS_MAX * box_l) / mt

            def moment(mass, cog):
                r = cog - flange
                return np.cross(r, mass * G_ACC * gdir) / 1000.0      # N m (r in mm)
            mo_t = moment(m, c)
            mo_tc = mo_t + moment(P.BOX_MASS_MAX, box_l)
            worst_tc = mt * G_ACC * float(np.linalg.norm(ct - flange)) / 1000.0
            row[arm] = dict(tool_mass_kg=round(m, 4), tool_cog_link6_mm=[round(float(x), 1) for x in c],
                            carton_cog_link6_mm=[round(float(x), 1) for x in box_l],
                            total_mass_kg=round(mt, 4), total_cog_link6_mm=[round(float(x), 1) for x in ct],
                            moment_handoff_tool_Nm=round(float(np.linalg.norm(mo_t)), 3),
                            moment_handoff_tool_plus_carton_Nm=round(float(np.linalg.norm(mo_tc)), 3),
                            moment_about_link6_z_Nm=round(float(abs(mo_tc[2])), 3),
                            moment_worst_orientation_tool_plus_carton_Nm=round(worst_tc, 3),
                            payload_margin_kg=round(P.ROBOT_PAYLOAD - mt, 3))
        out[bid] = row
    return out


# ---------------------------------------------------------------- main
def main():
    quick = '--quick' in sys.argv
    import time
    t0 = time.time()
    res = {'suction_tool': SUCTION_NAME, 'frames': 'shelf frame W (common/handoff.py), mm', 'wrist_model': 'ReBeL link5.dae + link6.dae (visual meshes, exact triangles) placed as in igus_rebel2.urdf, joint6=0',
           'tool_model': 'suction (%s) + gripper: URDF collision meshes (convex pieces, coacd hulls, conservative); an ejector envelope is added only if the suction folder has step/ejector_envelope.step (v1/v2)' % SUCTION_NAME,
           'unexpected_threshold_mm': CLR}
    res['assertions'] = urdf_assert()
    print('assertions ok', flush=True)
    sc = Scene()
    print('loaded %.1fs' % (time.time() - t0), flush=True)
    res['check1_yaw0'] = check1(sc)
    print('check1 %.1fs' % (time.time() - t0), flush=True)
    res['check4_yaw180'] = check1(sc, yaw=180.0)
    print('check4 %.1fs' % (time.time() - t0), flush=True)
    res['check2_sweep'] = check2(sc, step=2.0 if not quick else 10.0)
    print('check2 %.1fs' % (time.time() - t0), flush=True)
    res['check3_pad_row'] = check3(sc, quick=quick)
    print('check3 %.1fs' % (time.time() - t0), flush=True)
    res['check5_mass'] = check5(sc)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    out = os.path.join(HERE, 'data', ('handoff_check%s.json' % OUT_SUFFIX) if not quick else ('handoff_check_quick%s.json' % OUT_SUFFIX))
    json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('wrote', out, '%.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
