"""Verification of the left suction tool v5.  python verify.py  -> data/verify.json (+ summary printed).
Every number comes from the CAD solids / exported files; nothing is copied from the design intent."""
import os, sys, json, itertools, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, trimesh
import cadquery as cq
from scipy.spatial import ConvexHull
import importlib.util
import params as P
import parts as pa
import lib as L
import massprops
from meshlib import to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
C = P.C
Lz = P.LIP_Z
R = {}
t_start = time.time()
parts = pa.build_parts()
D = {d['name']: d for d in parts}
names = [d['name'] for d in parts]
meshes = {d['name']: to_mesh(d['shape'], tol=0.05, ang=0.15) for d in parts}
BODY = D['tool_body']['shape']


def is_pad(n): return n.startswith(('pad_', 'nut_front_'))


def load_handoff():
    """common/handoff.py does `from params import BOXES, PULL_OUT` (common/params.py); this folder owns the name 'params', so swap it temporarily"""
    saved = sys.modules.pop('params', None)
    sys.path.insert(0, os.path.join(ROOT, 'common'))
    try:
        spec = importlib.util.spec_from_file_location('common_handoff', os.path.join(ROOT, 'common', 'handoff.py'))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(os.path.join(ROOT, 'common'))
        sys.modules.pop('params', None)
        if saved is not None:
            sys.modules['params'] = saved
    return m


H = load_handoff()
assert abs(H.SUCTION_TCP_Z - Lz) < 1e-9 and abs(H.SUCTION_TCP_COMPRESSED_Z - P.COMPRESSED_TCP_Z) < 1e-9


# ------------------------------------------------------------------------------------------------ 1. interference matrix
t0 = time.time()
n = len(names); M = np.zeros((n, n)); tight = []
for i, j in itertools.combinations(range(n), 2):
    a, b = D[names[i]]['shape'], D[names[j]]['shape']
    v = L.common_vol(a, b)
    M[i, j] = M[j, i] = v
    if v < 0.01:
        ba, bb = a.BoundingBox(), b.BoundingBox()
        gap = max(ba.xmin - bb.xmax, bb.xmin - ba.xmax, ba.ymin - bb.ymax, bb.ymin - ba.ymax, ba.zmin - bb.zmax, bb.zmin - ba.zmax)
        if gap < 3.0:
            try:
                dist = a.distance(b)
            except Exception:
                dist = float('nan')
            if dist < 3.0:
                tight.append((names[i], names[j], round(float(dist), 3)))
bad = [(names[i], names[j], float(M[i, j])) for i, j in itertools.combinations(range(n), 2) if not M[i, j] < 0.01]
R['interference'] = dict(parts=names, n_pairs=n * (n - 1) // 2, pairs_with_overlap=bad, max_overlap_mm3=float(M.max()),
                         pairs_closer_than_3mm=sorted(tight, key=lambda t: t[2]), note='screws, zip ties and plugs are not modelled')
print('interference pairs', n * (n - 1) // 2, 'overlaps', len(bad), 'tight', len(tight), round(time.time() - t0), 's', flush=True)

# ------------------------------------------------------------------------------------------------ 2. keep-out (nothing but pads + the pad bar above LIP_Z - 40, bar inside |y| <= 12)
zmax = {nm: D[nm]['shape'].BoundingBox().zmax for nm in names}
slab = L.aabox((-400, 400), (-400, 400), (P.KEEPOUT_Z + 1e-4, Lz + 5))
over = {}
for nm in names:
    if nm.startswith('pad_'):
        continue
    if zmax[nm] > P.KEEPOUT_Z + 0.05:
        sl = D[nm]['shape'].intersect(slab)
        if sl.Volume() > 1e-3:
            bb = sl.BoundingBox()
            over[nm] = dict(zmax=round(zmax[nm], 2), y=[round(bb.ymin, 2), round(bb.ymax, 2)], x=[round(bb.xmin, 2), round(bb.xmax, 2)])
stad = L.zstadium(0, 0, P.KEEPOUT_Z - 1, P.BAR_FRONT + 1, 2 * 75.4, 2 * P.KO_HALF_Y)
outside = {nm: round(D[nm]['shape'].intersect(slab).cut(stad).Volume(), 3) for nm in over}
ko_ok = all(v < 1e-3 for v in outside.values()) and all(v['zmax'] <= P.BAR_FRONT + 0.05 for v in over.values())
R['keepout'] = dict(limit_z=P.KEEPOUT_Z, parts_in_slab_above_limit=over, volume_outside_bar_envelope_mm3=outside, bar_envelope='stadium 150.8 x 24 (|y| <= 12), z <= 133.5',
                    max_z_non_pad=max(v for k, v in zmax.items() if not k.startswith('pad_')), camera_max_z=zmax['camera_gemini305'], pass_=bool(ko_ok),
                    note='the rear nuts / washers (z 123..127) are inside the bar envelope; everything else but the pads and the bar is at z <= 119')
print('keepout', ko_ok, outside, flush=True)

# ------------------------------------------------------------------------------------------------ 3. pressed pose vs carton + neighbours (boxes 1-5)
from OCP.BRepExtrema import BRepExtrema_DistShapeShape


def dist(a, b):
    d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped); d.Perform(); return d.Value()


head_cut = L.aabox((-200, 200), (-200, 200), (Lz - 14.0 - 0.001, Lz + 1))
tool_sh, head_sh = {}, {}
for nm in names:
    if nm.startswith('pad_'):
        tool_sh[nm] = D[nm]['shape'].cut(head_cut)
        head_sh[nm] = L.move(D[nm]['shape'].intersect(head_cut), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5.0], [0, 0, 0, 1.0]]))
    elif nm.startswith(('tube_', 'trunk_')):
        continue
    else:
        tool_sh[nm] = D[nm]['shape']
pressed = {}
for bid, b in C.BOXES.items():
    Tw = H.suction_link6_pose(bid, pulled=False); Ti = np.linalg.inv(Tw)
    boxes = {}
    for nm, Pw in [('target', H.box_pose(bid, pulled=False))] + [('nb%d' % k, p) for k, p in enumerate(H.neighbour_poses(bid, protrusion=C.NEIGHBOUR_PROTRUSION))]:
        Tt = Ti @ Pw
        boxes[nm] = L.obox(Tt[:3, 3], Tt[:3, :3], [b['t'], b['l'], b['h']])
    tb = boxes['target'].BoundingBox()
    res = dict(box=b['name'], thickness=b['t'], spine_tool_z=float(tb.zmin), tcp_z=P.COMPRESSED_TCP_Z)
    worst = (1e9, None, None)
    per_part = {}
    for nm, sh in tool_sh.items():
        dm = min(dist(sh, bx) for bx in boxes.values())
        per_part[nm] = dm
        for bn, bx in boxes.items():
            pass
        if dm < worst[0]:
            worst = (dm, nm, min(boxes, key=lambda k: dist(sh, boxes[k])))
    res['min_clearance_all_parts_except_pad_heads_mm'] = dict(value=worst[0], part=worst[1], box=worst[2])
    res['pad_heads_distance_mm'] = {bn: min(dist(h, bx) for h in head_sh.values()) for bn, bx in boxes.items()}
    res['min_clearance_to_neighbours_any_part_mm'] = min(min(dist(sh, boxes[k]) for sh in list(tool_sh.values()) + list(head_sh.values())) for k in boxes if k != 'target')
    res['camera_to_nearest_box_mm'] = min(dist(tool_sh['camera_gemini305'], bx) for bx in boxes.values())
    res['gap_bar_front_to_neighbour_front_plane_mm'] = float((P.COMPRESSED_TCP_Z - C.NEIGHBOUR_PROTRUSION) - P.BAR_FRONT)
    res['five_closest_parts_mm'] = {k: round(v, 2) for k, v in sorted(per_part.items(), key=lambda kv: kv[1])[:5]}
    pressed[str(bid)] = res
    print('box', bid, 'min clearance %.2f %s, camera %.2f, nb %.2f' % (worst[0], worst[1], res['camera_to_nearest_box_mm'], res['min_clearance_to_neighbours_any_part_mm']), flush=True)
R['pressed_pose'] = dict(per_box=pressed, protrusion_mm=C.NEIGHBOUR_PROTRUSION,
                         method='handoff.suction_link6_pose(pulled=False) (compressed TCP z = 154 on the spine), neighbours 15 mm proud, exact solid distance (OCC); pad heads retracted 5 mm',
                         min_clearance_overall_mm=min(v['min_clearance_all_parts_except_pad_heads_mm']['value'] for v in pressed.values()),
                         min_camera_clearance_mm=min(v['camera_to_nearest_box_mm'] for v in pressed.values()),
                         min_clearance_to_neighbours_mm=min(v['min_clearance_to_neighbours_any_part_mm'] for v in pressed.values()))

# ------------------------------------------------------------------------------------------------ 4. camera: pose, occlusion, visibility, barcode pixels
Tc = P.T_CAM; C0 = Tc[:3, 3]; ax = Tc[:3, 2]; inv = np.linalg.inv(Tc)
ZS = Lz + P.APPROACH_GAP                         # approach pose: spine plane at link6 z = 204
aim = np.array([0.0, 0.0, ZS])
depth_aim = float((aim - C0) @ ax); range_aim = float(np.linalg.norm(aim - C0))
cam_bb = D['camera_gemini305']['shape'].BoundingBox()
hf, vf = C.CAMERA['depth_fov_deg']
all_ymax = max(D[nm]['shape'].BoundingBox().ymax for nm in names if not nm.startswith('trunk'))
blk_meshes = [meshes[nm] for nm in names if nm != 'camera_gemini305']
blk = trimesh.util.concatenate(blk_meshes); ri_tool = trimesh.ray.ray_triangle.RayMeshIntersector(blk)


def box_mesh_tool(bid, nb_t=None, protrusion=C.NEIGHBOUR_PROTRUSION, zspine=ZS):
    """target spine + 2 neighbours (tool frame, spine plane at link6 z = zspine; neighbours' spines `protrusion` mm closer to the tool)"""
    b = C.BOXES[bid]; t, h, l = b['t'], b['h'], b['l']; tn = t if nb_t is None else nb_t
    out = []
    tg = trimesh.creation.box(extents=[h, t, l]); tg.apply_translation([0, 0, zspine + l / 2]); out.append(tg)
    for s in (-1, 1):
        nbm = trimesh.creation.box(extents=[h, tn, l + protrusion]); nbm.apply_translation([0, s * (t / 2 + tn / 2), zspine + (l + protrusion) / 2 - protrusion]); out.append(nbm)
    return out


def fov_rays(origin_pts, ngrid=(45, 33)):
    ths = np.degrees(np.arctan(np.linspace(-math.tan(math.radians(hf / 2)), math.tan(math.radians(hf / 2)), ngrid[0])))
    phs = np.degrees(np.arctan(np.linspace(-math.tan(math.radians(vf / 2)), math.tan(math.radians(vf / 2)), ngrid[1])))
    o, d = [], []
    for p in origin_pts:
        for th in ths:
            for ph in phs:
                dl = np.array([math.tan(math.radians(th)), math.tan(math.radians(ph)), 1.0]); dl /= np.linalg.norm(dl)
                o.append((Tc @ np.append(p, 1))[:3]); d.append(Tc[:3, :3] @ dl)
    return np.array(o), np.array(d)


def blocked_fraction(ri, o, d, limit_plane_z=None, rmax=None):
    loc, ir, _ = ri.intersects_location(o, d, multiple_hits=False)
    hit = np.zeros(len(o), bool)
    dz = np.where(d[:, 2] > 1e-6, d[:, 2], 1e-6)
    lim = np.full(len(o), rmax if rmax else 1e9) if limit_plane_z is None else (limit_plane_z - o[:, 2]) / dz
    for l_, i_ in zip(loc, ir):
        if np.linalg.norm(l_ - o[i_]) < lim[i_]:
            hit[i_] = True
    return hit


O3 = [np.array([0, 0, 0.05]), np.array([-9.0, 0, 0.05]), np.array([9.0, 0, 0.05])]     # front-face centre and the two lens positions (+-9 mm, baseline 18)
o_r, d_r = fov_rays(O3)
hit_appr = blocked_fraction(ri_tool, o_r, d_r, limit_plane_z=ZS)
hit_300 = blocked_fraction(ri_tool, o_r, d_r, rmax=300.0)
per_origin = [float(hit_appr[i * (len(hit_appr) // 3):(i + 1) * (len(hit_appr) // 3)].mean()) for i in range(3)]
# which parts block (by part, rays to the approach plane)
part_block = {}
for nm in names:
    if nm == 'camera_gemini305':
        continue
    h_ = blocked_fraction(trimesh.ray.ray_triangle.RayMeshIntersector(meshes[nm]), o_r, d_r, limit_plane_z=ZS)
    if h_.any():
        part_block[nm] = round(100 * float(h_.mean()), 3)
print('camera occlusion at approach %.3f %% (centre %.3f), within 300 mm %.3f %%' % (100 * hit_appr.mean(), 100 * per_origin[0], 100 * hit_300.mean()), part_block, flush=True)


def project(pts):
    """points (tool frame) -> camera-frame (x, u, t) and angles"""
    pc = (inv @ np.column_stack([pts, np.ones(len(pts))]).T).T[:, :3]
    return pc, np.degrees(np.arctan2(np.abs(pc[:, 0]), pc[:, 2])), np.degrees(np.arctan2(np.abs(pc[:, 1]), pc[:, 2]))


def visibility(bid, zspine, protrusion):
    """sample the target spine (width t) and the +-1 neighbour spines (front faces, `protrusion` proud) at x = -60 / -30 / 0 / 30 / 60: in the depth FOV? clear line of sight?"""
    b = C.BOXES[bid]; t = b['t']
    ys_t = np.linspace(-t / 2 + 0.3, t / 2 - 0.3, 9)
    ys_n = np.concatenate([np.linspace(t / 2 + 0.3, 3 * t / 2 - 0.3, 9), np.linspace(-3 * t / 2 + 0.3, -t / 2 - 0.3, 9)])
    xs = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    pts_t = np.array([[x, y, zspine] for x in xs for y in ys_t]); pts_n = np.array([[x, y, zspine - protrusion] for x in xs for y in ys_n])
    pc_t, ah_t, av_t = project(pts_t); pc_n, ah_n, av_n = project(pts_n)
    infov = lambda pc, ah, av: (pc[:, 2] > 0) & (ah <= hf / 2) & (av <= vf / 2)
    f_t, f_n = infov(pc_t, ah_t, av_t), infov(pc_n, ah_n, av_n)
    # line of sight: tool + the two neighbour boxes as occluders (rays from the front-face centre, stopped 0.05 mm before the sample point)
    occ = trimesh.util.concatenate([blk] + box_mesh_tool(bid, protrusion=protrusion, zspine=zspine)[1:])
    ri = trimesh.ray.ray_triangle.RayMeshIntersector(occ)
    out = {}
    for tag, pts, fov in (('target_spine', pts_t, f_t), ('neighbour_spines', pts_n, f_n)):
        o = np.tile(C0, (len(pts), 1)); dv = pts - C0; dl = np.linalg.norm(dv, axis=1); dv /= dl[:, None]
        loc, ir, _ = ri.intersects_location(o, dv, multiple_hits=False)
        clear = np.ones(len(pts), bool)
        for l_, i_ in zip(loc, ir):
            if np.linalg.norm(l_ - o[i_]) < dl[i_] - 0.05:
                clear[i_] = False
        out[tag] = dict(in_fov_pct=100 * float(fov.mean()), line_of_sight_clear_pct=100 * float(clear.mean()), visible_pct=100 * float((fov & clear).mean()))
    # shadow of the +y neighbour on the target spine (centre column x = 0)
    ytt = np.linspace(-t / 2, t / 2, 161)
    ptt = np.array([[0.0, y, zspine] for y in ytt]); o = np.tile(C0, (len(ptt), 1)); dv = ptt - C0; dl = np.linalg.norm(dv, axis=1); dv /= dl[:, None]
    loc, ir, _ = ri.intersects_location(o, dv, multiple_hits=False)
    cl = np.ones(len(ptt), bool)
    for l_, i_ in zip(loc, ir):
        if np.linalg.norm(l_ - o[i_]) < dl[i_] - 0.05:
            cl[i_] = False
    out['target_spine_width_visible_at_x0_mm'] = float(cl.mean() * t)
    return out


vis = {}
for bid in C.BOXES:
    vis[str(bid)] = dict(flush=visibility(bid, ZS, 0.0), proud15=visibility(bid, ZS, C.NEIGHBOUR_PROTRUSION))
# shadow width vs standoff (box 5, 12 mm; box 4, 16 mm): the +y neighbour (15 mm proud) hides this much of the target spine at x = 0
shadow = {}
for gap in (30.0, 45.0, 60.0, 90.0):
    zs_ = Lz + gap
    shadow[str(int(gap))] = {str(bid): visibility(bid, zs_, C.NEIGHBOUR_PROTRUSION)['target_spine_width_visible_at_x0_mm'] / C.BOXES[bid]['t'] for bid in (4, 5)}

# coverage of the spine plane in the camera FOV at the approach pose: y range (x = 0) and the angular margin for +-1 neighbour (3 x 16 mm = 48 mm wide)
ang_tilt = lambda y, z: math.degrees(math.atan2(-(y - C0[1]) * Tc[2, 2] * 0 + 0, 1))
pc, ah, av = project(np.array([[0, 24.0, ZS], [0, -24.0, ZS], [0, 0.0, ZS], [-60.0, 0, ZS], [60.0, 0, ZS]]))
fov_margin = dict(y_plus_24_tilt_deg=float(vf / 2 - av[0]), y_minus_24_tilt_deg=float(vf / 2 - av[1]), centre_h_deg=float(hf / 2 - ah[2]), x_minus_60_h_deg=float(hf / 2 - ah[3]), x_plus_60_h_deg=float(hf / 2 - ah[4]))
# y range visible at x = 0 on the spine plane (tilt-plane FOV edges)
al = math.radians(P.CAM_TILT_DEG)
y_vis = [float(C0[1] - (ZS - C0[2]) * math.tan(al + s * math.radians(vf / 2)) / 1.0) for s in (+1, -1)]

# barcode pixels per module
F_PX = 0.5 * 1280 / math.tan(math.radians(C.CAMERA['color_fov_deg'][0] / 2))        # px, from H 94 deg over 1280 px (1280 x 800 colour)
F_PX_V = 0.5 * 800 / math.tan(math.radians(C.CAMERA['color_fov_deg'][1] / 2))
CODES = {'DataMatrix 10 mil (0.254 mm)': 0.254, 'DataMatrix 15 mil (0.381 mm)': 0.381, 'DataMatrix 20 mil (0.508 mm)': 0.508, 'EAN-13 (0.33 mm module)': 0.33}


def pitch_mm_per_px(point):
    """pixel footprint on the spine plane z = point[2] at `point` (tool frame): along tool x and along the tilt direction (finite differences), colour camera f_px"""
    def px(p):
        pcx, pcu, pct = (inv @ np.append(p, 1))[:3]
        return np.array([F_PX * pcx / pct, F_PX_V * pcu / pct])
    p0 = np.asarray(point, float)
    jx = (px(p0 + [0.5, 0, 0]) - px(p0 - [0.5, 0, 0])) / 1.0
    jy = (px(p0 + [0, 0.5, 0]) - px(p0 - [0, 0.5, 0])) / 1.0
    return 1.0 / np.linalg.norm(jx), 1.0 / np.linalg.norm(jy)


barcode = {}
for gap in P.SCAN_STANDOFFS + (15.0, 20.0):
    zs_ = Lz + gap
    d_ax = float((np.array([0, 0, zs_]) - C0) @ ax)
    px_, py_ = pitch_mm_per_px([0, 0, zs_])
    simple = 2 * d_ax * math.tan(math.radians(47.0)) / 1280.0
    barcode['lip+%g (spine z %.0f)' % (gap, zs_)] = dict(
        depth_mm=d_ax, pitch_simple_mm_per_px=simple, pitch_along_x_mm_per_px=px_, pitch_along_tilt_mm_per_px=py_,
        px_per_module_simple={k: v / simple for k, v in CODES.items()},
        px_per_module_worst_axis={k: v / max(px_, py_) for k, v in CODES.items()},
        y_range_visible_at_x0_mm=[float(C0[1] - (zs_ - C0[2]) * math.tan(al + s * math.radians(vf / 2))) for s in (+1, -1)],
        depth_ge_good_min=bool(d_ax >= C.CAMERA['good_min_z']), depth_ge_min=bool(d_ax >= C.CAMERA['min_z']))
# standoff needed for +-24 mm (target + 2 x 16 mm neighbours) to be inside the tilt-plane FOV, nominal V = 65 deg and with the datasheet tolerance (-3 deg)
need_pm24 = {}
for tag_, half_ in (('nominal_65deg', vf / 2), ('minus_3deg_62deg', vf / 2 - 3.0)):
    zs_ = C0[2] + (C0[1] + 24.0) / math.tan(math.radians(half_) + al)
    need_pm24[tag_] = dict(lip_to_spine_mm=float(zs_ - Lz), depth_mm=float((np.array([0, 0, zs_]) - C0) @ ax))
# max depth for >= 2.5 px/module (simple formula) and the corresponding spine z / lip standoff
max_depth = {k: v / (2.5 * 2 * math.tan(math.radians(47.0)) / 1280.0) for k, v in CODES.items()}
standoff_for = {}
for k, d_ in max_depth.items():
    # depth along the optical axis of the spine centre (0, 0, zs): (zs - zc) * cos + yc * sin
    zs_ = (d_ - C0[1] * (-Tc[2, 1] * -1)) if False else None
    cos_a, sin_a = math.cos(al), math.sin(al)
    zs_ = C0[2] + (d_ - C0[1] * sin_a) / cos_a
    standoff_for[k] = dict(max_depth_mm=d_, depth_ge_min_50=bool(d_ >= C.CAMERA['min_z']), depth_ge_good_90=bool(d_ >= C.CAMERA['good_min_z']),
                           spine_link6_z=float(zs_), lip_to_spine_mm=float(zs_ - Lz), pressed_depth_mm=float((np.array([0, 0, P.COMPRESSED_TCP_Z]) - C0) @ ax))
R['camera'] = dict(
    front_face_centre_mm=C0.round(3).tolist(), optical_axis=ax.round(4).tolist(), tilt_deg=P.CAM_TILT_DEG, approach_gap_lip_to_spine_mm=P.APPROACH_GAP, approach_spine_z=ZS,
    aim_point=aim.tolist(), depth_along_axis_to_aim_mm=depth_aim, range_to_aim_mm=range_aim, good_depth_min_mm=C.CAMERA['good_min_z'], aim_in_good_depth=bool(depth_aim >= C.CAMERA['good_min_z']),
    camera_max_y=float(cam_bb.ymax), camera_max_z=float(cam_bb.zmax), z_limit=P.KEEPOUT_Z, tool_max_y_incl_camera=float(all_ymax),
    blocked_pct_depth_fov_at_approach=100 * float(hit_appr.mean()), blocked_pct_per_origin_centre_left_right=[100 * v for v in per_origin],
    blocked_pct_within_300mm=100 * float(hit_300.mean()), blocking_parts_pct=part_block, rays=int(len(hit_appr)),
    fov_margin_deg_at_spine_plane=fov_margin, y_visible_at_x0_on_spine_plane_mm=y_vis, standoff_needed_for_pm24mm_in_fov=need_pm24,
    visibility=vis, shadow_of_proud_neighbour_fraction_of_target_spine_width=shadow,
    barcode_px_per_module=barcode, barcode_max_depth_for_2p5_px_per_module=standoff_for, f_px_colour=[F_PX, F_PX_V],
    usb_c=dict(face='+x', camera_frame_t_range=P.CAM_USB['t'], tie_slot_in_ear=True, note='vendor STEP recess vertices and the datasheet right-hand view agree; v3 had the relief lug on -x (wrong side)'),
    targets=dict(blocked_le_10pct=bool(100 * hit_appr.mean() <= 10.0), spine_pm1_neighbour_in_fov=bool(fov_margin['y_plus_24_tilt_deg'] > 0 and fov_margin['y_minus_24_tilt_deg'] > 0),
                 aim_in_good_depth=bool(depth_aim >= C.CAMERA['good_min_z']), y_le_60=bool(all_ymax <= 60.0 + 1e-6 or cam_bb.ymax <= 60.0)))
print('camera depth_aim %.1f ymax cam %.1f tool %.1f blocked %.3f %% y_vis %s' % (depth_aim, cam_bb.ymax, all_ymax, 100 * hit_appr.mean(), np.round(y_vis, 1)), flush=True)

# ------------------------------------------------------------------------------------------------ 5. buffer stroke (10 mm)
worst = {}
for i, x in enumerate(P.PAD_X):
    head_x = L.union([L.zstadium(x, 0, Lz - 1.5, Lz, 30.8, 4.8), L.zstadium(x, 0, Lz - 8.0, Lz - 1.5, 24.0, 3.6), L.zstadium(x, 0, Lz - 14.0, Lz - 8.0, 16.0, 4.0)])
    nm_pad = ['pad_L', 'pad_M', 'pad_R'][i]
    for s in np.arange(0, 10.01, 1.0):
        h = L.move(head_x, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1.0]]))
        for nm in names:
            if nm == nm_pad:
                continue
            v = L.common_vol(h, D[nm]['shape'])
            if v > 0.01:
                worst[(nm_pad, nm)] = max(worst.get((nm_pad, nm), 0), v)
gap_full = (Lz - 14.0 - 10.0) - P.BAR_FRONT
R['buffer_stroke'] = dict(stroke_mm=10.0, collisions=[[a, b, float(v)] for (a, b), v in worst.items()], gap_adapter_to_bar_front_at_full_stroke_mm=float(gap_full),
                          note='the sliding head is in front of the bar; buffer body, rear hex, elbow and tube are on the clamped (fixed) body: the stroke does not move the tubes (ASSUMED from the catalog drawing: inlet on the rear of the fixed body)',
                          pass_=bool(len(worst) == 0 and gap_full > 0))

# ------------------------------------------------------------------------------------------------ 6. tubes + collector
rows = []
for i, nm in enumerate(['tube_L', 'tube_M', 'tube_R']):
    info = pa.TUBE_INFO[i]; bb = D[nm]['shape'].BoundingBox()
    zlo, zhi = bb.zmin, bb.zmax
    rows.append(dict(tube=nm, min_bend_radius_mm=min([b['radius'] for b in info['bends']], default=None), bend_angles_deg=[round(b['angle_deg'], 1) for b in info['bends']],
                     straight_segments_mm=[round(float(s), 1) for s in info['seg_len']], length_mm=info['length'], chord_mm=info['chord'], slack_mm=info['length'] - info['chord'],
                     z_range=[round(zlo, 2), round(zhi, 2)], loop_below_bar=False, z_below_bar_rear_face_mm=P.BOSS_Z0 - zhi))
# if the elbow inlets DID move with the head (not expected): worst-case chord growth for a 10 mm z shift of the elbow end
stroke_slack = {}
for i, nm in enumerate(['tube_L', 'tube_M', 'tube_R']):
    w = pa.tube_waypoints(i); a0, d0 = w[0], w[-1]
    ch = [float(np.linalg.norm(d0 - (a0 + np.array([0, 0, s])))) for s in (-10.0, 0.0, 10.0)]
    stroke_slack[nm] = dict(chord_at_minus10_0_plus10_mm=ch, length_mm=pa.TUBE_INFO[i]['length'], taut_if_inlet_moved=bool(pa.TUBE_INFO[i]['length'] < max(ch) + 0.0))
# trunk vs the ReBeL wrist housing: the stub is a vertical line at (0, COLL_YC); link6 mesh (visual, as in igus_rebel2.urdf) and link5 below it
REBEL_MESH = '/home/farhan/orca/workspaces/igus_rebel_ros2/box-3d/src/igus_rebel_description/meshes/rebel_d00617809'
trunk_wrist = dict(trunk_axis_radius_mm=abs(P.COLL_YC), trunk_outer_radius_mm=P.TRUNK_OD / 2)
try:
    rad = []
    for lk, dz in (('link6', 0.0), ('link5', -76.8)):
        mm_ = trimesh.load(os.path.join(REBEL_MESH, lk + '.dae'), force='mesh'); mm_.apply_scale(1000.0)
        vv = mm_.vertices + np.array([1.3449814287804, 0.0, -0.5195 + dz])
        near = vv[(vv[:, 1] < 0) & (np.abs(vv[:, 0]) < 12.0)]                 # housing surface in the -y half-plane around the trunk x = 0
        rad.append(float(-near[:, 1].min()))
    trunk_wrist.update(housing_max_extent_toward_minus_y_mm=dict(link6=rad[0], link5=rad[1]),
                       clearance_to_housing_mm=float(abs(P.COLL_YC) - P.TRUNK_OD / 2 - max(rad)), pass_=bool(abs(P.COLL_YC) - P.TRUNK_OD / 2 - max(rad) > 0.5))
except Exception as e_:
    trunk_wrist.update(error=str(e_), pass_=None)
R['tubes'] = dict(rows=rows, trunk_vs_wrist_housing=trunk_wrist, tube_axis_z=P.ELBOW_AXIS_Z, bend_radius_min_required_mm=P.TUBE_BEND_R,
                  trunk='phi6 leaves the collector straight down (-z) at y = %.0f (hub rim r = 28): no bend inside the CAD, bend radius assumed >= %.0f mm when dressed' % (P.COLL_YC, P.TRUNK_BEND_R),
                  stroke_if_inlet_moved=stroke_slack,
                  pass_=bool(all((r['min_bend_radius_mm'] is None or r['min_bend_radius_mm'] >= P.TUBE_BEND_R - 1e-6) for r in rows) and trunk_wrist.get('pass_') is not False))

# ------------------------------------------------------------------------------------------------ 7. strength of the printed bar and stem (first-order beam model)
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union


def ring_props(coords):
    """signed area, int b dA, int b^2 dA (b = second coordinate) of a closed polygon ring (ccw positive)"""
    A = Sb = Ib = 0.0
    for (a0, b0), (a1, b1) in zip(coords[:-1], coords[1:]):
        c = a0 * b1 - a1 * b0
        A += c / 2.0; Sb += c * (b0 + b1) / 6.0; Ib += c * (b0 * b0 + b0 * b1 + b1 * b1) / 12.0
    return A, Sb, Ib


def poly_props(geom):
    """area, centroid b, I about the centroid axis (int (b - b_c)^2 dA), extreme-fibre distance"""
    A = Sb = Ib = 0.0; bmin, bmax = 1e9, -1e9
    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for pg in polys:
        pg = Polygon(pg.exterior.coords, [h.coords for h in pg.interiors])
        from shapely.geometry.polygon import orient
        pg = orient(pg, 1.0)
        a, s, i = ring_props(list(pg.exterior.coords))
        for h_ in pg.interiors:
            a2, s2, i2 = ring_props(list(h_.coords)); a += a2; s += s2; i += i2
        A += a; Sb += s; Ib += i
        bb = pg.bounds; bmin = min(bmin, bb[1]); bmax = max(bmax, bb[3])
    bc = Sb / A
    return A, bc, Ib - A * bc * bc, max(bmax - bc, bc - bmin)


mesh_body = to_mesh(BODY, tol=0.03, ang=0.1)


def section_polys(axis, val, T3):
    """cross-section of the body mesh: T3 maps (x,y,z) -> (a, b, normal); returns the shapely polygons (a, b)"""
    normal = np.zeros(3); normal[axis] = 1.0; origin = np.zeros(3); origin[axis] = val
    sec = mesh_body.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    pl, _ = sec.to_planar(to_2D=T3(val))
    return list(pl.polygons_full)


T_x = lambda x0: np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -x0], [0, 0, 0, 1.0]])       # x-section: (y, z)
T_z = lambda z0: np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, -z0], [0, 0, 0, 1.0]])       # z-section: a = y, b = x
xs = np.arange(P.STEM_HX, 60.0 + 1e-9, 0.5)
EI, cdist, area_bar = [], [], []
for x in xs:
    polys = [pg for pg in section_polys(0, x, T_x) if pg.centroid.y > P.KEEPOUT_Z + 1.0]          # the bar (skin + ribs + bosses); the seat / bay / tubes are other load paths
    g = unary_union(polys)
    A, zc, I, c = poly_props(g)
    EI.append(P.E_PRINT * I); cdist.append(c); area_bar.append(A)
EI = np.array(EI); cdist = np.array(cdist)
Iy = EI / P.E_PRINT
# stem: I of the C-section about the y axis (int x^2 dA) from z sections at 66..118, clipped to the stem rectangle
Is_list, As_list = [], []
rect = sbox(P.STEM_Y0 - 0.01, -P.STEM_HX - 0.01, P.STEM_Y1 + 0.01, P.STEM_HX + 0.01)       # section coordinates are (a, b) = (y, x)
for z in np.arange(66.0, 118.1, 4.0):
    polys = [pg.intersection(rect) for pg in section_polys(2, z, T_z) if pg.intersection(rect).area > 1.0]
    g = unary_union(polys)
    A, bc, I, c = poly_props(g)           # b = x (centred), I about the centroidal axis x = bc
    Is_list.append(I); As_list.append(A)
I_stem = float(min(Is_list)); A_stem = float(min(As_list)); L_stem = P.SKIN_Z0 - P.HUB_Z1
c_stem = P.STEM_HX


def bar_cantilever(Pn, xp, stem_moment_Nmm):
    """tip deflection at x = xp of the bar (cantilever from the stem face, EI(x) from the CAD sections) + the rigid rotation of the stem top; max bending stress along the bar"""
    sel = xs <= xp + 1e-9
    xx = xs[sel]; ei = EI[sel]; cc = cdist[sel]; ii = Iy[sel]
    M = Pn * (xp - xx)
    th = np.concatenate([[0], np.cumsum(0.5 * (M[1:] / ei[1:] + M[:-1] / ei[:-1]) * np.diff(xx))])
    dl = float(np.sum(0.5 * (th[1:] + th[:-1]) * np.diff(xx)))
    theta_s = stem_moment_Nmm * L_stem / (P.E_PRINT * I_stem)
    sig = M * cc / ii
    return dict(deflection_bar_mm=dl, stem_rotation_rad=float(theta_s), deflection_from_stem_rotation_mm=float(theta_s * xp), deflection_total_mm=dl + float(theta_s * xp),
                sigma_bar_max_MPa=float(sig.max()), x_of_sigma_max=float(xx[np.argmax(sig)]))


def case(name, loads, xp, desc):
    """loads = {x_pad: force N} (positive = toward -z push, sign does not matter for magnitudes); bar of the loaded outer pad xp"""
    Pn = abs(loads[xp])
    M_stem = abs(sum(f * x for x, f in loads.items()))                 # moment about y on the stem top (signed sum: symmetric loads cancel)
    r = bar_cantilever(Pn, abs(xp), M_stem)
    sig_stem = M_stem * c_stem / I_stem + sum(abs(f) for f in loads.values()) / A_stem
    sig = max(r['sigma_bar_max_MPa'], sig_stem)
    r.update(name=name, description=desc, loads_N={str(k): v for k, v in loads.items()}, stem_moment_Nmm=M_stem, sigma_stem_MPa=float(sig_stem),
             sigma_max_MPa=float(sig), safety_factor=float(P.SIGMA_ALLOW / sig), safety_factor_Kt2=float(P.SIGMA_ALLOW / (2.0 * sig)))
    return r


loads_a = {-60.0: P.SPRING_N, 0.0: P.SPRING_N, 60.0: P.SPRING_N + P.ARM_PUSH_N}
loads_b = {-60.0: P.PULL_N, 0.0: P.PULL_N, 60.0: P.PULL_N}
loads_c = {60.0: P.SPRING_N + P.ARM_PUSH_N}
loads_b1 = {60.0: P.PULL_N}
CASES = [case('a_push_on', loads_a, 60.0, 'buffers fully compressed (3 N each) + 20 N arm push on the +x outer pad'),
         case('b_pull', loads_b, 60.0, '3 pads x 80 kPa x 1.16 cm2 = 9.28 N tension each (symmetric: no net stem moment)'),
         case('c_one_outer_pad_only', loads_c, 60.0, 'worst case: only the +x outer pad loaded, 3 N spring + 20 N arm push = 23 N (a single pad cannot hold more than 9.28 N in pull)'),
         case('b1_one_outer_pad_pull', loads_b1, 60.0, 'only the +x outer pad holds (others lost vacuum), 9.28 N tension')]
# nut bearing stress: preload + working load, front nut on the pocket floor, rear washer on the boss
TORQUE = 0.3                                 # N m, assembly torque on the printed bar (ASSUMED; SMC 2.5-3.5 N m is for metal brackets)
K_NUT = 0.2
F_pre = TORQUE * 1000.0 / (K_NUT * 10.0)
A_front = math.pi / 4 * (13.0 ** 2 - 10.5 ** 2); A_rear = math.pi / 4 * (P.WASHER['od'] ** 2 - 10.5 ** 2)
bearing = dict(torque_Nm=TORQUE, K=K_NUT, preload_N=F_pre, front_nut_bearing_mm2=A_front, front_nut_stress_MPa=(F_pre + 23.0) / A_front, front_nut_SF=P.SIGMA_ALLOW / ((F_pre + 23.0) / A_front),
               rear_washer_bearing_mm2=A_rear, rear_washer_stress_MPa=(F_pre + 23.0) / A_rear, rear_washer_SF=P.SIGMA_ALLOW / ((F_pre + 23.0) / A_rear),
               at_SMC_torque_3Nm_front_nut_stress_MPa=(3000.0 / (K_NUT * 10.0)) / A_front)
bearing['SF_min'] = min(bearing['front_nut_SF'], bearing['rear_washer_SF'])
defl_b = [c_ for c_ in CASES if c_['name'] == 'b_pull'][0]['deflection_total_mm']
R['strength'] = dict(
    E_MPa=P.E_PRINT, sigma_allow_MPa=P.SIGMA_ALLOW, method='bar = cantilever from the stem face (x = 15.5) to the pad axis (x = 60) with EI(x) from the CAD cross-sections (skin + ribs + bosses, pad hole included) integrated twice, '
    'plus rigid rotation of the stem top (C-section I from z sections) under the net moment; bending stress M c / I along the bar and in the stem walls (+ axial); first-order, no FEA, no layer-adhesion knock-down',
    section_at_root=dict(I_mm4=float(Iy[0]), c_mm=float(cdist[0]), area_mm2=float(area_bar[0])), section_at_pad=dict(I_mm4=float(Iy[-1]), c_mm=float(cdist[-1]), area_mm2=float(area_bar[-1])),
    stem=dict(I_mm4=I_stem, area_mm2=A_stem, height_mm=L_stem), cases={c_['name']: c_ for c_ in CASES}, nut_bearing=bearing,
    targets=dict(outer_pad_deflection_b_le_0p3mm=bool(defl_b <= 0.3), safety_factor_ge_3_all_cases=bool(all(c_['safety_factor'] >= 3.0 for c_ in CASES) and bearing['SF_min'] >= 3.0),
                 safety_factor_ge_3_with_Kt2=bool(all(c_['safety_factor_Kt2'] >= 3.0 for c_ in CASES))))
print('strength', {c_['name']: (round(c_['deflection_total_mm'], 3), round(c_['sigma_max_MPa'], 2), round(c_['safety_factor'], 1)) for c_ in CASES}, 'bearing SF', round(bearing['SF_min'], 2), flush=True)

# ------------------------------------------------------------------------------------------------ 8. assembly access (free sweeps)
acc = {}
others = lambda excl: [D[nm]['shape'] for nm in names if nm not in excl]


def sweep_free(solid_fn, steps, excl, vs=None):
    worst_v = 0.0; who = None
    for s in steps:
        sh = solid_fn(s)
        for nm in names:
            if nm in excl:
                continue
            v = L.common_vol(sh, D[nm]['shape'])
            if v > worst_v:
                worst_v, who = v, nm
    return worst_v, who


# 8a. flange screws: phi6 column (driver + head) from the hub top straight up with the camera removed (all other parts in place)
fs = {}
for k, (x, y) in enumerate(P.FLANGE_HOLES):
    v, w = sweep_free(lambda s, x=x, y=y: L.zcyl(x, y, P.HUB_Z1 + 0.05, 230.0, 3.0), [0], excl=('camera_gemini305',))
    fs['screw_%d (%.1f, %.1f)' % (k, x, y)] = dict(blocked_mm3=float(v), by=w)
acc['flange_screws_camera_removed'] = fs
# 8b. camera screws: phi4 driver along -u from the head recess out through the open -y side
cs = {}
U_HEAD = P.CAV_Y1 - float(P.T_CAM[1, 3]) - 0.1                  # camera u of the screw head seat (stem cavity face y = 8.6)
for sx in (-1, 1):
    a_ = pa.cam_pt(sx * P.CAM_SCREW_X, U_HEAD, P.CAM_SCREW_T); b_ = pa.cam_pt(sx * P.CAM_SCREW_X, U_HEAD - 45.0, P.CAM_SCREW_T)     # 45 mm of driver shank beyond the screw head
    v, w = sweep_free(lambda s: L.ocyl(a_, b_, 2.0), [0], excl=('camera_gemini305',))
    cs['x=%+.0f' % (sx * P.CAM_SCREW_X)] = dict(blocked_mm3=float(v), by=w, body_blocked_mm3=float(L.common_vol(L.ocyl(a_, b_, 2.0), BODY)))
acc['camera_screws_from_minus_y'] = cs
# 8c. rear nuts: thin open-end spanner (jaw gap 14.2, jaw wall 4.5, thickness 3.0) swept in; middle from +y through the wall window, outer ones from +-x
SP_T, SP_G, SP_W = 3.0, 14.2, 4.5


def spanner(x0, direction, s):
    """spanner head with the slot opening toward -direction, nut centre at x0; s = how far the head is retracted along `direction` (0 = engaged: slot bottom 8.6 mm past the nut centre)"""
    z0 = P.NUT_REAR_Z0 + (P.NUT_T - SP_T) / 2
    if direction == 'y':       # middle nut, vertices on +-y, jaws separated along x, slot opens toward +y, engaged from +y
        jaws = [L.aabox((x0 + sg * (SP_G / 2), x0 + sg * (SP_G / 2 + SP_W)) if sg > 0 else (x0 - (SP_G / 2 + SP_W), x0 - SP_G / 2), (-8.6, 70.0 + s), (z0, z0 + SP_T)) for sg in (-1, 1)]
        web = L.aabox((x0 - SP_G / 2 - SP_W, x0 + SP_G / 2 + SP_W), (-8.6 - SP_W, -8.6), (z0, z0 + SP_T))
        return L.union(jaws + [web])
    sgn = 1 if direction == '+x' else -1          # outer nuts: jaws separated along y, slot opens toward +-x
    jaws = [L.aabox(sorted((x0, x0 + sgn * (8.6 + 70.0 + s))), (SP_G / 2, SP_G / 2 + SP_W), (z0, z0 + SP_T)), L.aabox(sorted((x0, x0 + sgn * (8.6 + 70.0 + s))), (-(SP_G / 2 + SP_W), -SP_G / 2), (z0, z0 + SP_T))]
    web = L.aabox(sorted((x0 - sgn * 8.6, x0 - sgn * (8.6 + SP_W))), (-SP_G / 2 - SP_W, SP_G / 2 + SP_W), (z0, z0 + SP_T))
    return L.union(jaws + [web])


sp = {}
for nm, x0, dr in (('nut_rear_M', 0.0, 'y'),):
    v, w = sweep_free(lambda s, x0=x0, dr=dr: spanner(x0, dr, s), [0.0, -20.0, -40.0], excl=(nm,))
    sp[nm] = dict(blocked_mm3=float(v), by=w, tool='open-end spanner 14 mm, jaw wall 4.5, thickness 3, from +y through the wall window')
for nm, x0 in (('nut_rear_L', -60.0), ('nut_rear_R', 60.0)):          # thin-wall 14 mm deep socket (OD 19) from behind, BEFORE the elbows are fitted
    v, w = sweep_free(lambda s, x0=x0: L.zcyl(x0, 0, 85.0, P.NUT_REAR_Z1, 9.5), [0.0], excl=(nm, 'elbow_L', 'elbow_M', 'elbow_R', 'tube_L', 'tube_M', 'tube_R', 'pad_L', 'pad_R'))
    sp[nm] = dict(blocked_mm3=float(v), by=w, tool='deep socket 14 mm, OD 19, from behind, elbows fitted afterwards (pad body excluded: it passes through the socket)')
acc['rear_nut_spanner'] = sp
# 8d. collector drops in from +z (body + bosses swept straight up from its seat), pads go in from the front (rear end first): body of revolution of the rear part swept up
cd = L.union([L.aabox((-12, 12), (P.COLL_YC - 5, P.COLL_YC + 5), (P.COLL_ZC - 8, 200.0)), L.aabox((-17, 17), (P.COLL_YC - 4.5, P.COLL_YC + 4.5), (P.COLL_ZC - 4.5, 200.0)),
              L.zcyl(0, P.COLL_YC, P.COLL_ZC - 12.0, P.COLL_ZC, 5.5)])
acc['collector_drop_in_from_plus_z'] = dict(blocked_with_body_mm3=float(L.common_vol(cd, BODY)),
                                           note='sweep of body, port bosses and the trunk boss (phi11) straight up from the seat; the cheek slots are open to +z')
pad_sweeps = {}
for i, x in enumerate(P.PAD_X):
    rear = L.zcyl(x, 0, P.INLET_Z - 0.5, P.BAR_FRONT, P.THREAD_R + 0.05)          # thread + hex region that passes through the hole when the pad is pushed in from the front
    sw = L.union([rear, L.zcyl(x, 0, P.BAR_FRONT, P.BAR_FRONT + 60.0, P.THREAD_R + 0.05)])
    pad_sweeps['pad_%s' % ['L', 'M', 'R'][i]] = float(L.common_vol(sw, BODY)) if False else float(L.common_vol(L.zcyl(x, 0, P.BAR_FRONT + 0.1, P.BAR_FRONT + 60.0, P.THREAD_R + 0.05), BODY))
acc['pads_in_from_front_blocked_mm3'] = pad_sweeps
acc['all_ok'] = bool(all(v['blocked_mm3'] < 0.01 for v in fs.values()) and all(v['blocked_mm3'] < 0.01 and v['body_blocked_mm3'] < 0.01 for v in cs.values())
                     and all(v['blocked_mm3'] < 0.01 for v in sp.values()) and acc['collector_drop_in_from_plus_z']['blocked_with_body_mm3'] < 0.01 and all(v < 0.01 for v in pad_sweeps.values()))
R['access'] = acc
print('access', acc['all_ok'], {k: v for k, v in acc.items() if k != 'all_ok'} if not acc['all_ok'] else '', flush=True)

# ------------------------------------------------------------------------------------------------ 9. printability: orientation, supports, min wall
def support_est(m, up, seed=1):
    n_ = m.face_normals; ar = m.area_faces
    ov = (n_ @ up) < -math.sin(math.radians(45.0)) - 0.02
    bottom = m.vertices @ up
    ov &= (m.triangles_center @ up) > bottom.min() + 0.05
    area = float(ar[ov].sum())
    vol = 0.0
    if ov.any():
        idx = np.where(ov)[0]; rng = np.random.default_rng(seed)
        pick = rng.choice(idx, size=min(len(idx), 2500), replace=False, p=ar[idx] / ar[idx].sum())
        o = m.triangles_center[pick] - up * 0.02
        ri = trimesh.ray.ray_triangle.RayMeshIntersector(m)
        loc, ir, _ = ri.intersects_location(o, np.tile(-up, (len(o), 1)), multiple_hits=False)
        drop = np.full(len(o), np.nan)
        for l_, i_ in zip(loc, ir):
            drop[i_] = np.linalg.norm(l_ - o[i_])
        bedd = (o @ up) - bottom.min()
        drop = np.where(np.isnan(drop), bedd, np.minimum(np.nan_to_num(drop, nan=1e9), bedd))
        vol = float((drop * (ar[pick] * (ar[idx].sum() / ar[pick].sum()))).sum() / 1000.0)
    return dict(overhang_area_mm2=area, support_volume_upper_bound_cm3=vol)


dirs = {'+z': [0, 0, 1], '-z': [0, 0, -1], '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0]}
import glob
import support_model as sm
m_b = to_mesh(BODY, tol=0.08, ang=0.15)
cands = {k: support_est(m_b, np.array(v, float)) for k, v in dirs.items()}               # raw solid upper bound (v3 / v4 method)
best_raw = min(cands, key=lambda k: cands[k]['support_volume_upper_bound_cm3'])
best, vox = sm.best_axis(m_b)                                                           # bridge-aware voxel model (primary)
bb = BODY.BoundingBox()
ext = [bb.xlen, bb.ylen, bb.zlen]
height_along = {'+z': bb.zlen, '-z': bb.zlen, '+x': bb.xlen, '-x': bb.xlen, '+y': bb.ylen, '-y': bb.ylen}[best]
footprint = sorted([e for k_, e in zip('xyz', ext) if k_ != best[1]], reverse=True)


def same_method(folder, names_):
    tot_vox = tot_raw = 0.0; per = {}
    for f in sorted(glob.glob(os.path.join(ROOT, folder, 'stl', '*.stl'))):
        mm = trimesh.load(f)
        bq, rq = sm.best_axis(mm)
        raw_ = min(support_est(mm, np.array(vv, float))['support_volume_upper_bound_cm3'] for vv in dirs.values())
        per[os.path.basename(f)] = dict(best_up=bq, voxel_support_cm3=rq[bq]['support_cm3'], raw_upper_bound_best_cm3=raw_)
        tot_vox += rq[bq]['support_cm3']; tot_raw += raw_
    return dict(parts=per, voxel_support_total_cm3=tot_vox, raw_upper_bound_total_best_dir_per_part_cm3=tot_raw)


cmp_v3 = same_method('suction_tool', None); cmp_v4 = same_method('suction_tool_v4_concept', None)


def wall_thickness(mesh, count=60000, seed=2, limit=3.0):
    """ray wall thickness: sample the surface, shoot a ray along the inward normal, thickness = distance to the first opposite (anti-parallel, n.n < -0.9) face.
    Returns the distribution and the clusters of samples thinner than `limit` (surface grid 2 mm)."""
    pts, fidx = trimesh.sample.sample_surface(mesh, count, seed=seed)
    nrm = mesh.face_normals[fidx]
    o = pts - nrm * 0.01
    ri_ = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    loc, ir, tri = ri_.intersects_location(o, -nrm, multiple_hits=False)
    th = np.full(len(pts), np.nan)
    ok = np.einsum('ij,ij->i', mesh.face_normals[tri], nrm[ir]) < -0.9
    th[ir[ok]] = np.linalg.norm(loc[ok] - o[ir[ok]], axis=1) + 0.01
    valid = ~np.isnan(th)
    thin = valid & (th < limit - 0.05)
    cl = {}
    for p_, t_ in zip(pts[thin], th[thin]):
        k = tuple(np.round(p_ / 3.0).astype(int))
        c = cl.setdefault(k, [t_, p_])
        c[0] = min(c[0], t_)
    regs = sorted(([float(t_), p_.round(1).tolist()] for t_, p_ in cl.values()), key=lambda r: r[0])[:12]
    return dict(samples=int(count), valid_samples=int(valid.sum()), min_mm=float(np.nanmin(th)), p01_mm=float(np.nanpercentile(th, 0.1)), p1_mm=float(np.nanpercentile(th, 1)),
                n_thin_samples=int(thin.sum()), thinnest_regions=regs)



tw = wall_thickness(to_mesh(BODY, tol=0.03, ang=0.1))
R['printability'] = dict(material=P.PRINT_MATERIAL, printed_parts=1, build_orientation_up=best, support_volume_cm3_bridge_aware=vox[best]['support_cm3'], support_pct_of_part=100 * vox[best]['support_cm3'] * 1000 / BODY.Volume(),
                         voxel_model_all_axes_cm3={k: v_['support_cm3'] for k, v_ in vox.items()}, raw_upper_bound_cm3_all_axes={k: v_['support_volume_upper_bound_cm3'] for k, v_ in cands.items()},
                         raw_upper_bound_cm3_at_build_orientation=cands[best]['support_volume_upper_bound_cm3'], raw_best_axis=best_raw, raw_best_cm3=cands[best_raw]['support_volume_upper_bound_cm3'],
                         part_volume_cm3=BODY.Volume() / 1000, bbox_mm=[bb.xlen, bb.ylen, bb.zlen], footprint_on_bed_mm=footprint, height_mm=height_along, fits_220_x_220_bed=bool(footprint[0] <= 220 and footprint[1] <= 220),
                         design_wall_mm=P.WALL, min_wall=tw, min_wall_ok=bool(tw['n_thin_samples'] == 0), same_method_v3=cmp_v3, same_method_v4=cmp_v4,
                         method='bridge-aware voxel model (support_model.py): 45 deg rule + bridges up to 12 mm; the raw bound (v3/v4 method) = every down-facing face x its free drop, no bridging; wall check = inward-normal ray thickness on 60000 surface samples',
                         orientation_note='bar front face (z = 133.5, the hex-pocket side) on the bed, stem growing up, flange face on top: pads holes / pockets are bed-side features; the flange face is the last (top) surface')
print('printability best', best, round(vox[best]['support_cm3'], 1), 'cm3 (voxel); raw bound', round(cands[best]['support_volume_upper_bound_cm3'], 1), '; v3', round(cmp_v3['voxel_support_total_cm3'], 1), 'v4', round(cmp_v4['voxel_support_total_cm3'], 1), '; wall min %.2f p0.1 %.2f thin samples %d' % (tw['min_mm'], tw['p01_mm'], tw['n_thin_samples']), tw['thinnest_regions'][:4], flush=True)

# ------------------------------------------------------------------------------------------------ 10. mass / CoG / compactness vs v3, v4
mp = massprops.assembly_props(parts, meshes=meshes)
printed_kg = sum(m for nm, (m, c) in mp['bodies'].items() if D[nm]['kind'] == 'printed')
v3m = json.load(open(os.path.join(ROOT, 'suction_tool', 'data', 'mass.json'))); v4m = json.load(open(os.path.join(ROOT, 'suction_tool_v4_concept', 'data', 'mass.json')))
v4v = json.load(open(os.path.join(ROOT, 'suction_tool_v4_concept', 'data', 'verify.json')))
v3v = json.load(open(os.path.join(ROOT, 'suction_tool', 'data', 'verify.json')))
mn = np.array([min(D[nm]['shape'].BoundingBox().xmin for nm in names), min(D[nm]['shape'].BoundingBox().ymin for nm in names), min(D[nm]['shape'].BoundingBox().zmin for nm in names)])
mx = np.array([max(D[nm]['shape'].BoundingBox().xmax for nm in names), max(D[nm]['shape'].BoundingBox().ymax for nm in names), max(D[nm]['shape'].BoundingBox().zmax for nm in names)])
hull5 = ConvexHull(np.vstack([meshes[nm].vertices for nm in names])).volume / 1000
fast = mp['mass'] + sum(P.MASS_FASTENERS[k] for k in ('flange_csk_m3x10', 'camera_m3x10'))
R['mass'] = dict(total_kg=mp['mass'], total_with_fasteners_kg=fast, budget_kg=0.22, within_budget=bool(fast <= 0.22), printed_kg=printed_kg, cog_mm=mp['com'].tolist(),
                 inertia_kg_mm2=mp['inertia'].tolist(), bodies={n_: [round(m, 4), np.round(c, 1).tolist()] for n_, (m, c) in mp['bodies'].items()},
                 fasteners=P.MASS_FASTENERS, vs=dict(v3_kg=v3m['mass_kg'], v4_aluminium_bar_kg=v4m['mass_kg'], v4_pa12cf_bar_kg=v4v['compactness']['bar_material']['total_with_pa12cf_kg']),
                 model='printed = volume x %.2f g/cm3 x %.2f fill (ASSUMED); purchased as MASS_PURCHASED (pad 12 g + elbow 3 g, collector 12 g ASSUMED, camera 68 g); screws counted separately, zip ties / plugs not counted' % (P.PRINT_DENSITY * 1e6, P.PRINT_FILL_EQ))
R['compactness'] = dict(bbox_min=mn.round(2).tolist(), bbox_max=mx.round(2).tolist(), size=(mx - mn).round(2).tolist(), bbox_volume_cm3=float(np.prod(mx - mn) / 1000), hull_cm3=float(hull5),
                        n_printed=sum(d['kind'] == 'printed' for d in parts), n_purchased_solids=sum(d['kind'] == 'purchased' for d in parts), n_solids=len(parts),
                        v3=dict(size=v3v['compactness']['default']['size'], hull_cm3=v3v['compactness']['default']['hull_cm3']), v4=dict(size=v4v['compactness']['v4']['size'], hull_cm3=v4v['compactness']['v4']['hull_cm3']))
print('mass %.4f kg (+fasteners %.4f) printed %.1f g, size %s, hull %.0f cm3' % (mp['mass'], fast, printed_kg * 1000, (mx - mn).round(1), hull5), flush=True)

# ------------------------------------------------------------------------------------------------ 11. lips, span margins, exports re-import
lipz = {nm: float(D[nm]['shape'].BoundingBox().zmax) for nm in ('pad_L', 'pad_M', 'pad_R')}
span = 2 * (P.PAD_PITCH + 15.4)
R['lips'] = dict(lip_z=lipz, coplanarity_spread_mm=max(lipz.values()) - min(lipz.values()), pad_span_outer_edges_mm=span, pad_centres_x=P.PAD_X, pad_pitch_mm=P.PAD_PITCH)
R['pad_span_margins'] = {str(k): dict(box=b['name'], h=b['h'], margin_to_each_spine_end_mm=b['h'] / 2 - span / 2, ok_ge_25=bool(b['h'] / 2 - span / 2 >= 25.0)) for k, b in C.BOXES.items()}
import tempfile
ri_ok = True; riv = {}
for d in parts:
    fn = os.path.join(P.SCRATCH, 'chk_v5_%s.step' % d['name']); cq.exporters.export(d['shape'], fn)
    r_ = cq.importers.importStep(fn).val()
    riv[d['name']] = abs(r_.Volume() - d['shape'].Volume()) / max(d['shape'].Volume(), 1e-9); os.remove(fn)
stl_rows = {}
for d in parts:
    if d['kind'] == 'printed':
        fn = os.path.join(HERE, 'stl', '%s.stl' % d['name'])
        stl_rows[d['name']] = dict(exists=os.path.exists(fn), watertight=bool(trimesh.load(fn).is_watertight) if os.path.exists(fn) else None)
asm_ok = None
try:
    asm = cq.importers.importStep(os.path.join(HERE, 'step', 'assembly.step'))
    asm_ok = len(asm.vals()[0].Solids()) if asm.vals() else None
except Exception as e:
    asm_ok = str(e)
R['reimport'] = dict(max_rel_volume_error=max(riv.values()), step_parts=len(riv), stl_printed=stl_rows, assembly_step_solids=asm_ok, expected_assembly_solids=len(parts),
                     pass_=bool(max(riv.values()) < 1e-3 and all(v['watertight'] for v in stl_rows.values())))

R['summary'] = dict(
    mass_kg=mp['mass'], mass_with_fasteners_kg=fast, target_mass_kg=0.22, printed_parts=1, n_solids=len(parts), bbox_mm=(mx - mn).round(1).tolist(),
    interference_ok=len(bad) == 0, keepout_ok=bool(ko_ok), stroke_ok=R['buffer_stroke']['pass_'], tubes_ok=R['tubes']['pass_'], access_ok=acc['all_ok'],
    min_clearance_pressed_mm=R['pressed_pose']['min_clearance_overall_mm'], min_camera_clearance_pressed_mm=R['pressed_pose']['min_camera_clearance_mm'],
    camera_blocked_pct=100 * float(hit_appr.mean()), camera_aim_depth_mm=depth_aim, camera_max_y=float(cam_bb.ymax),
    deflection_b_mm=defl_b, sf_min=min(c_['safety_factor'] for c_ in CASES), support_cm3=vox[best]['support_cm3'], support_raw_bound_cm3=cands[best]['support_volume_upper_bound_cm3'], thin_wall_samples=tw['n_thin_samples'], min_wall_mm=tw['min_mm'],
    reimport_ok=R['reimport']['pass_'], seconds=round(time.time() - t_start))
json.dump(R, open(os.path.join(HERE, 'data/verify.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print(json.dumps(R['summary']))
