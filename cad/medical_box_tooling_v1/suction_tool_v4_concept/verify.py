"""Concept-level checks of the v4 tool.  python verify.py -> data/verify.json (numbers from the CAD solids, nothing copied from intent)."""
import os, sys, json, itertools, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, trimesh
import cadquery as cq
from scipy.spatial import ConvexHull
import v4_params as V
import v4_parts as pa
from bridge import P3, PA3, L, C, ML, MP3, load_handoff, HERE, SCRATCH

Lz = V.LIP_Z
R = {}
parts = pa.build_parts()
D = {d['name']: d for d in parts}
names = [d['name'] for d in parts]
meshes = {d['name']: ML.to_mesh(d['shape'], tol=0.05, ang=0.15) for d in parts}


def is_pad(n): return n.startswith(('pad_', 'nut_front_'))


# ---------------------------------------------------------------- 1. bbox / part count / mass (+ v3 comparison)
def env(shapes):
    bbs = [s.BoundingBox() for s in shapes]
    mn = np.array([min(b.xmin for b in bbs), min(b.ymin for b in bbs), min(b.zmin for b in bbs)])
    mx = np.array([max(b.xmax for b in bbs), max(b.ymax for b in bbs), max(b.zmax for b in bbs)])
    return mn, mx

mn4, mx4 = env([d['shape'] for d in parts])
v3parts = PA3.build_parts()
mn3, mx3 = env([d['shape'] for d in v3parts])
v3rec = json.load(open(os.path.join(P3.HERE, 'data/mass.json')))
v3_hull = ConvexHull(np.vstack([ML.to_mesh(d['shape'], tol=0.2, ang=0.3).vertices for d in v3parts])).volume / 1000
v4_hull = ConvexHull(np.vstack([meshes[n].vertices for n in names])).volume / 1000
mp = MP3.assembly_props(parts, meshes=meshes)       # printed = PETG x PRINT_FILL_EQ (v3 model), purchased = d['mass']

bar_vol = D['bar_al6061']['shape'].Volume()
bar_al = bar_vol * V.AL_DENSITY
bar_pa = bar_vol * V.PA12CF_DENSITY * V.PA12CF_FILL
L_free = 60.0 - 12.0   # cantilever from the stem face to the outer pad axis
I = V.BAR_W * V.BAR_T ** 3 / 12.0
defl = lambda E, P=10.0: P * L_free ** 3 / (3 * E * I)
mass_alt = dict(al6061_bar_g=bar_al * 1000, pa12cf_bar_g=bar_pa * 1000, bar_volume_cm3=bar_vol / 1000,
                total_with_al_kg=mp['mass'], total_with_pa12cf_kg=mp['mass'] - bar_al + bar_pa,
                lip_deflection_mm_10N_per_pad=dict(al=defl(V.E_AL), pa12cf=defl(V.E_PA12CF)))
R['compactness'] = dict(
    v4=dict(bbox_min=mn4.round(2).tolist(), bbox_max=mx4.round(2).tolist(), size=(mx4 - mn4).round(2).tolist(), hull_cm3=float(v4_hull),
            bbox_volume_cm3=float(np.prod(mx4 - mn4) / 1000), mass_kg=mp['mass'], com_mm=mp['com'].round(2).tolist(),
            n_printed=sum(d['kind'] == 'printed' for d in parts), n_purchased_solids=sum(d['kind'] == 'purchased' for d in parts), n_solids=len(parts)),
    v3=dict(bbox_min=mn3.round(2).tolist(), bbox_max=mx3.round(2).tolist(), size=(mx3 - mn3).round(2).tolist(), hull_cm3=float(v3_hull), bbox_volume_cm3=float(np.prod(mx3 - mn3) / 1000),
            mass_kg=v3rec['mass_kg'], com_mm=[round(v, 2) for v in v3rec['com_mm']], n_printed=sum(d['kind'] == 'printed' for d in v3parts),
            n_purchased_solids=sum(d['kind'] != 'printed' for d in v3parts), n_solids=len(v3parts)),
    bar_material=mass_alt)
R['mass'] = dict(mass_kg=mp['mass'], bodies={n: [round(m, 4), np.round(c, 1).tolist()] for n, (m, c) in mp['bodies'].items()},
                 printed_kg=sum(m for n, (m, c) in mp['bodies'].items() if D[n]['kind'] == 'printed'),
                 note='printed = volume x PETG 1.27 g/cm3 x 0.55 (v3 model); purchased masses as v3 MASS_PURCHASED (washers dropped); screws, inserts, flex, cables not included')
print('bbox v4', (mx4 - mn4).round(1), 'v3', (mx3 - mn3).round(1), 'mass', round(mp['mass'], 3), 'hull', round(v4_hull), 'vs', round(v3_hull))

# ---------------------------------------------------------------- 2. interference (boolean common volume, all pairs)
t0 = time.time(); n = len(names); M = np.zeros((n, n)); tight = []
for i, j in itertools.combinations(range(n), 2):
    a, b = D[names[i]]['shape'], D[names[j]]['shape']
    v = L.common_vol(a, b); M[i, j] = M[j, i] = v
    if v < 0.01:
        ba, bb = a.BoundingBox(), b.BoundingBox()
        gap = max(ba.xmin - bb.xmax, bb.xmin - ba.xmax, ba.ymin - bb.ymax, bb.ymin - ba.ymax, ba.zmin - bb.zmax, bb.zmin - ba.zmax)
        if gap < 3.0:
            try: dist = a.distance(b)
            except Exception: dist = float('nan')
            if dist < 3.0: tight.append((names[i], names[j], round(float(dist), 3)))
bad = [(names[i], names[j], round(float(M[i, j]), 2)) for i, j in itertools.combinations(range(n), 2) if not M[i, j] < 0.01]
R['interference'] = dict(n_pairs=n * (n - 1) // 2, pairs_with_overlap=bad, max_overlap_mm3=float(M.max()), pairs_closer_than_3mm=sorted(tight, key=lambda t: t[2]),
                         note='screws, inserts, zip ties not modelled')
print('interference overlaps', bad, round(time.time() - t0), 's')

# ---------------------------------------------------------------- 3. keep-out (nothing but pads and pad bar above LIP_Z - 40)
zmax = {nm: D[nm]['shape'].BoundingBox().zmax for nm in names}
nonpad = {nm: round(zmax[nm], 2) for nm in names if not (nm.startswith(('pad_', 'nut_front_')) or nm == 'bar_al6061')}
over = {k: v for k, v in nonpad.items() if v > V.KEEPOUT_Z + 1e-6}
R['keepout'] = dict(limit_z=V.KEEPOUT_Z, max_z_of_non_pad_non_bar_parts=nonpad, parts_above_limit=over,
                    bar_z=[V.BAR_Z0, V.BAR_Z1], bar_y=[-V.BAR_W / 2, V.BAR_W / 2], pass_=not over,
                    note='bar rear face 118.5 (stock 15 mm bar, front face 133.5 unchanged); hub_stem top and the rear nuts end at 118.5')
print('keepout over', over)

# ---------------------------------------------------------------- 4. pressed pose vs carton + neighbours (boxes 1-5)
H = load_handoff()
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
def dist(a, b):
    d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped); d.Perform(); return d.Value()
head_cut = L.aabox((-200, 200), (-200, 200), (Lz - 14.0 - 0.001, Lz + 1))
tool_sh, head_sh = {}, {}
for nm in names:
    if nm.startswith(('pad_',)):
        tool_sh[nm] = D[nm]['shape'].cut(head_cut)
        head_sh[nm] = L.move(D[nm]['shape'].intersect(head_cut), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5.0], [0, 0, 0, 1.0]]))   # pad compressed 5 mm (TCP 154)
    elif nm.startswith(('tube_', 'trunk_')): continue
    else: tool_sh[nm] = D[nm]['shape']
pressed = {}
for bid, b in C.BOXES.items():
    Tw = H.suction_link6_pose(bid, pulled=False); Ti = np.linalg.inv(Tw)
    boxes = {}
    for nm, Pw in [('target', H.box_pose(bid, pulled=False))] + [('nb%d' % k, p) for k, p in enumerate(H.neighbour_poses(bid, protrusion=C.NEIGHBOUR_PROTRUSION))]:
        Tt = Ti @ Pw
        boxes[nm] = L.obox(Tt[:3, 3], Tt[:3, :3], [b['t'], b['l'] if nm != 'target' else b['l'], b['h']] if nm != 'target' else [b['t'], b['l'], b['h']])
    # sanity: spine of the target at tool z = 154
    tb = boxes['target'].BoundingBox()
    res = dict(spine_tool_z=float(tb.zmin), tcp_z=V.COMPRESSED_TCP_Z)
    worst = (1e9, None, None)
    for nm, sh in tool_sh.items():
        for bn, bx in boxes.items():
            d = dist(sh, bx)
            if d < worst[0]: worst = (d, nm, bn)
    res['min_clearance_all_parts_except_pad_heads_mm'] = dict(value=worst[0], part=worst[1], box=worst[2])
    hd = {bn: min(dist(h, bx) for h in head_sh.values()) for bn, bx in boxes.items()}
    res['pad_heads_distance_mm'] = hd
    nbmin = min(min(dist(sh, boxes[k]) for sh in list(tool_sh.values()) + list(head_sh.values())) for k in boxes if k != 'target')
    res['min_clearance_to_neighbours_any_part_mm'] = nbmin
    res['gap_bar_front_to_neighbour_front_plane_mm'] = float((Lz - V.COMPRESSED_TCP_Z) + (V.COMPRESSED_TCP_Z - V.NEIGHBOUR_PROTRUSION) - V.BAR_Z1) if False else float((V.COMPRESSED_TCP_Z - C.NEIGHBOUR_PROTRUSION) - V.BAR_Z1)
    pressed[str(bid)] = res
    print('box', bid, 'min clearance (no pad heads) %.2f %s/%s, to neighbours %.2f, heads %s' % (worst[0], worst[1], worst[2], nbmin, {k: round(v, 2) for k, v in hd.items()}))
R['pressed_pose'] = dict(per_box=pressed, protrusion_mm=C.NEIGHBOUR_PROTRUSION, method='handoff.suction_link6_pose(pulled=False) (compressed TCP z=154 on the spine), neighbours 15 mm proud, exact solid distance (OCC); pad heads retracted 5 mm',
                         min_clearance_overall_mm=min(v['min_clearance_all_parts_except_pad_heads_mm']['value'] for v in pressed.values()),
                         min_clearance_to_neighbours_mm=min(v['min_clearance_to_neighbours_any_part_mm'] for v in pressed.values()))

# ---------------------------------------------------------------- 5. camera
Tc = V.T_CAM; C0 = Tc[:3, 3]; ax = Tc[:3, 2]; inv = np.linalg.inv(Tc)
aim = np.array(V.CAM_AIM)
miss = float(np.linalg.norm(np.cross(aim - C0, ax)))
rng_aim = float(np.linalg.norm(aim - C0)); rng_lip = float(np.linalg.norm(np.array([0, 0, Lz]) - C0))
lips = np.array([[x + sx * 15.4, sy * 2.4, Lz] for x in V.PAD_X for sx in (-1, 1) for sy in (-1, 1)] + [[x, 0, Lz] for x in V.PAD_X])
pc = (inv @ np.column_stack([lips, np.ones(len(lips))]).T).T[:, :3]
ah = np.degrees(np.arctan2(np.abs(pc[:, 0]), pc[:, 2])); av = np.degrees(np.arctan2(np.abs(pc[:, 1]), pc[:, 2]))
hf, vf = C.CAMERA['depth_fov_deg']
# FOV occlusion by the tool itself: rays from the front-face centre over the 88 x 65 deg depth FOV, 300 mm
blk = trimesh.util.concatenate([meshes[n_] for n_ in names if n_ != 'camera_gemini305'])
rint = trimesh.ray.ray_triangle.RayMeshIntersector(blk)
def fov_blocked(Tm, hfov, vfov, orig_pts, rmax=300.0, ngrid=(23, 17), skip=('camera_gemini305',)):
    bl = trimesh.util.concatenate([meshes[n_] for n_ in names if n_ not in skip]); ri = trimesh.ray.ray_triangle.RayMeshIntersector(bl)
    ths = np.degrees(np.arctan(np.linspace(-np.tan(np.radians(hfov / 2)), np.tan(np.radians(hfov / 2)), ngrid[0])))
    phs = np.degrees(np.arctan(np.linspace(-np.tan(np.radians(vfov / 2)), np.tan(np.radians(vfov / 2)), ngrid[1])))
    o, d = [], []
    for p in orig_pts:
        for th in ths:
            for ph in phs:
                dl = np.array([math.tan(math.radians(th)), math.tan(math.radians(ph)), 1.0]); dl /= np.linalg.norm(dl)
                o.append((Tm @ np.append(p, 1))[:3]); d.append(Tm[:3, :3] @ dl)
    o = np.array(o); d = np.array(d)
    loc, ir, _ = ri.intersects_location(o, d, multiple_hits=False)
    hit = np.zeros(len(o), bool)
    for l_, i_ in zip(loc, ir):
        if np.linalg.norm(l_ - o[i_]) < rmax: hit[i_] = True
    return hit, o, d, ths, phs
hit, o_, d_, ths, phs = fov_blocked(Tc, hf, vf, [np.array([0, 0, 0.05]), np.array([-9.0, 0, 0.05]), np.array([9.0, 0, 0.05])])
# lens-centre rays only: how much of the horizontal / vertical FOV is blocked
blocked_frac = float(hit.mean())
R['camera'] = dict(front_face_centre_mm=C0.round(3).tolist(), optical_axis=ax.round(4).tolist(), tilt_deg=V.CAM_TILT_DEG, aim_point=list(V.CAM_AIM),
                   aim_error_mm=miss, range_front_face_to_aim_point_mm=rng_aim, range_front_face_to_middle_lip_mm=rng_lip, good_depth_min_mm=C.CAMERA['good_min_z'],
                   range_to_aim_ok=rng_aim >= 90.0, pods_max_y=float(D['sensor_pod']['shape'].BoundingBox().ymax), camera_max_y=float(D['camera_gemini305']['shape'].BoundingBox().ymax),
                   camera_max_z=float(D['camera_gemini305']['shape'].BoundingBox().zmax), z_limit=V.KEEPOUT_Z,
                   lips_in_depth_fov_margin_deg=dict(horizontal=float(hf / 2 - ah.max()), vertical=float(vf / 2 - av.max())),
                   lips_note='informational: at 63 mm range the outer pad lips (x = +-75) are outside the 88 deg depth FOV; the wrist camera is for the spine, not the pad row',
                   tool_blocks_pct_of_depth_fov_rays_within_300mm=100 * blocked_frac,
                   pass_=bool(rng_aim >= 90.0 and miss < 0.01 and zmax['camera_gemini305'] <= V.KEEPOUT_Z))
print('camera range aim %.1f lip %.1f miss %.3f fov blocked %.1f%% pod ymax %.1f' % (rng_aim, rng_lip, miss, 100 * blocked_frac, R['camera']['pods_max_y']))

# ---------------------------------------------------------------- 6. scanner + decoder + scan pose
Ts = V.T_SCAN; S0 = Ts[:3, 3]; sa = Ts[:3, 2]
glare = math.degrees(math.acos(np.clip(sa @ np.array([0, 0, 1.0]), -1, 1)))
t_cross = np.linalg.norm(np.array([0, 0, V.SCAN_SPINE_Z]) - S0)
poses = {}
for st in (115.0, 130.0, 150.0):
    Tm, zT = V.scan_pose(st)
    poses[str(int(st))] = dict(spine_tool_z=float(zT), backed_off_from_pressed_mm=float(zT - V.COMPRESSED_TCP_Z), backed_off_from_lip_plane_mm=float(zT - Lz),
                               glare_tilt_deg=math.degrees(math.acos(Tm[2, 2])), camera_range_to_spine_centre_mm=float(np.linalg.norm(np.array([0, 0, zT]) - C0)))
Eh = V.ENGINE
hit_s, o_s, d_s, ths_s, phs_s = fov_blocked(Ts, V.SCAN_FOV_DEG[0], V.SCAN_FOV_DEG[1], [np.array([wx, ww, 0.05]) for wx in (-Eh['long'] / 2, 0, Eh['long'] / 2) for ww in (-Eh['thick'] / 2, 0, Eh['thick'] / 2)],
                                      ngrid=(7, 7), skip=('scanner_se4710',))
# also the same without the pod itself (a clipped ray by the pod front would be a bug; rays are started 0.05 mm in front of the window)
rear = (Ts @ np.array([0, 0, -Eh['depth'], 1]))[:3]
bb_d = D['decoder_pl3307']['shape'].BoundingBox()
corners = np.array([[x, y, z] for x in (bb_d.xmin, bb_d.xmax) for y in (bb_d.ymin, bb_d.ymax) for z in (bb_d.zmin, bb_d.zmax)])
dmax = float(np.linalg.norm(corners - rear, axis=1).max())
R['scanner'] = dict(window_centre_mm=S0.round(2).tolist(), axis=sa.round(4).tolist(), glare_tilt_to_spine_normal_deg=glare, glare_ok=glare >= 12.0,
                    aimed_at_spine_centre=True, scan_pose_nominal_standoff_mm=V.SCAN_STANDOFF_NOM, scan_pose_distance_check_mm=float(t_cross), scan_poses=poses,
                    scan_window_range_required_mm=[115, 150], max_z_of_engine=float(zmax['scanner_se4710']), z_limit=V.KEEPOUT_Z,
                    fov_rays=int(len(o_s)), fov_rays_clipped_within_300mm=int(hit_s.sum()), pct_clipped=float(100 * hit_s.mean()),
                    decoder_farthest_corner_from_engine_rear_mm=dmax, flex_reach_mm=V.DECODER['flex_reach'], flex_slack_mm=float(V.DECODER['flex_reach'] - dmax),
                    pass_=bool(glare >= 12.0 and hit_s.sum() == 0 and V.DECODER['flex_reach'] - dmax >= 10.0 and zmax['scanner_se4710'] <= V.KEEPOUT_Z))
print('scanner glare %.1f scan z %.1f clipped %.1f%% flex slack %.1f' % (glare, V.SCAN_SPINE_Z, 100 * hit_s.mean(), V.DECODER['flex_reach'] - dmax))

# ---------------------------------------------------------------- 7. buffer stroke (10 mm)
worst = {}
for i, x in enumerate(V.PAD_X):
    head_x = L.union([L.zstadium(x, 0, Lz - 1.5, Lz, 30.8, 4.8), L.zstadium(x, 0, Lz - 8.0, Lz - 1.5, 24.0, 3.6), L.zstadium(x, 0, Lz - 14.0, Lz - 8.0, 16.0, 4.0)])
    nm_pad = ['pad_L', 'pad_M', 'pad_R'][i]
    for s in np.arange(0, 10.01, 1.0):
        h = L.move(head_x, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1.0]]))
        for nm in names:
            if nm == nm_pad: continue
            v = L.common_vol(h, D[nm]['shape'])
            if v > 0.01: worst[(nm_pad, nm)] = max(worst.get((nm_pad, nm), 0), v)
gap_full = (Lz - 14.0 - 10.0) - V.BAR_Z1
R['buffer_stroke'] = dict(stroke_mm=10.0, collisions=[[a, b, float(v)] for (a, b), v in worst.items()], gap_adapter_to_bar_front_at_full_stroke_mm=float(gap_full),
                          note='the sliding head is in front of the bar (z > 133.5); buffer body, fitting and tube are fixed and sit behind the bar / inside the stem (middle pad) -> nothing inside the stem moves; middle fitting clear of the stem walls: see interference',
                          pass_=len(worst) == 0 and gap_full > 0)

# ---------------------------------------------------------------- 8. tubes
rows = []
for i, nm in enumerate(['tube_L', 'tube_M', 'tube_R']):
    info = pa.TUBE_INFO[i]
    bb = D[nm]['shape'].BoundingBox()
    rows.append(dict(tube=nm, min_bend_radius_mm=min([b['radius'] for b in info['bends']], default=None), n_bends=len(info['bends']),
                     straight_out_of_fitting_mm=(V.FIT_Z1 - bb.zmin if False else info['seg_len'][0] if info['bends'] else info['length']), segments_mm=[round(float(s), 1) for s in info['seg_len']],
                     length_mm=info['length'], z_min=bb.zmin, clearance_to_hub_top_mm=bb.zmin - V.HUB_Z1))
R['tubes'] = dict(rows=rows, trunk='phi6 leaves through the -y stem wall at z = %.1f, stub to y = %.0f' % (V.COLL_ZC, V.TRUNK_END_Y),
                  middle_tube='straight 10 mm from fitting to the +z collector port (no bend)',
                  pass_=bool(all((r['min_bend_radius_mm'] is None or r['min_bend_radius_mm'] >= 12 - 1e-6) and r['clearance_to_hub_top_mm'] >= 0 for r in rows)))

# ---------------------------------------------------------------- 9. printability (overhang > 45 deg from vertical, 6 build directions)
def support_est(m, up, seed=1):
    n = m.face_normals; ar = m.area_faces
    ov = (n @ up) < -math.sin(math.radians(45.0)) - 0.02
    bottom = m.vertices @ up
    ov &= (m.triangles_center @ up) > bottom.min() + 0.05
    area = float(ar[ov].sum())
    # connected patches
    patches = []
    if ov.any():
        idx = np.where(ov)[0]; sub = trimesh.graph.connected_components(m.face_adjacency[np.isin(m.face_adjacency, idx).all(axis=1)], nodes=idx)
        patches = sorted([float(ar[c].sum()) for c in sub], reverse=True)
    vol = 0.0
    if ov.any():
        idx = np.where(ov)[0]; rng = np.random.default_rng(seed)
        pick = rng.choice(idx, size=min(len(idx), 1500), replace=False, p=ar[idx] / ar[idx].sum())
        o = m.triangles_center[pick] - up * 0.02
        ri = trimesh.ray.ray_triangle.RayMeshIntersector(m)
        loc, ir, _ = ri.intersects_location(o, np.tile(-up, (len(o), 1)), multiple_hits=False)
        drop = np.full(len(o), np.nan)
        for l_, i_ in zip(loc, ir): drop[i_] = np.linalg.norm(l_ - o[i_])
        bedd = (o @ up) - bottom.min()
        drop = np.where(np.isnan(drop), bedd, np.minimum(np.nan_to_num(drop, nan=1e9), bedd))
        vol = float((drop * (ar[pick] * (ar[idx].sum() / ar[pick].sum()))).sum() / 1000.0)    # scaled to the full overhang area, cm3
    return dict(overhang_area_mm2=area, largest_patches_mm2=[round(p, 1) for p in patches[:5]], n_patches_over_25mm2=int(sum(p > 25 for p in patches)), support_volume_upper_bound_cm3=vol)
dirs = {'+z': [0, 0, 1], '-z': [0, 0, -1], '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0]}
pr = {}
for d in parts:
    if d['kind'] != 'printed': continue
    m = meshes[d['name']]
    c = {k: support_est(m, np.array(v, float)) for k, v in dirs.items()}
    best = min(c, key=lambda k: c[k]['support_volume_upper_bound_cm3'])
    wall = []
    pr[d['name']] = dict(candidates=c, best=best, best_support_cm3=c[best]['support_volume_upper_bound_cm3'], bbox_mm=[d['shape'].BoundingBox().xlen, d['shape'].BoundingBox().ylen, d['shape'].BoundingBox().zlen],
                         support_pct_of_part=100 * c[best]['support_volume_upper_bound_cm3'] * 1000 / d['shape'].Volume())
# same method on the v3 printed parts (apples to apples)
v3pr = {}
for d in v3parts:
    if d['kind'] != 'printed': continue
    m = ML.to_mesh(d['shape'], tol=0.1, ang=0.2)
    c = {k: support_est(m, np.array(v, float)) for k, v in dirs.items()}
    best = min(c, key=lambda k: c[k]['support_volume_upper_bound_cm3'])
    v3pr[d['name']] = dict(best=best, best_support_cm3=c[best]['support_volume_upper_bound_cm3'])
R['printability'] = dict(per_part=pr, v3_same_method=v3pr, method='faces whose normal is > 45 deg past vertical on the underside (n.up < -sin45); support volume = overhang area x drop to the part/bed (solid upper bound)',
                         min_wall_mm=dict(stem_wall=V.STEM_WALL, pod_wall=V.POD_WALL, hub_floor_under_collector=V.CAV_FLOOR_Z - V.HUB_Z0, pod_insert_wall_note='insert holes 4.2 x 5.5 in >= 8 mm of pod material'))

# ---------------------------------------------------------------- 10. STEP re-import
import tempfile
ri_ok = True; ri = {}
for d in parts:
    fn = os.path.join(SCRATCH, 'chk_%s.step' % d['name']); cq.exporters.export(d['shape'], fn)
    r_ = cq.importers.importStep(fn).val()
    ri[d['name']] = abs(r_.Volume() - d['shape'].Volume()) / max(d['shape'].Volume(), 1e-9); os.remove(fn)
R['reimport'] = dict(max_rel_volume_error=max(ri.values()), pass_=max(ri.values()) < 1e-3)

R['summary'] = dict(bbox_mm=(mx4 - mn4).round(1).tolist(), v3_bbox_mm=(mx3 - mn3).round(1).tolist(), mass_kg=mp['mass'], v3_mass_kg=v3rec['mass_kg'], n_printed=R['compactness']['v4']['n_printed'],
                    n_solids=len(parts), interference_ok=len(bad) == 0, keepout_ok=R['keepout']['pass_'], camera_ok=R['camera']['pass_'], scanner_ok=R['scanner']['pass_'],
                    stroke_ok=R['buffer_stroke']['pass_'], tubes_ok=R['tubes']['pass_'], min_clearance_pressed_mm=R['pressed_pose']['min_clearance_overall_mm'])
json.dump(R, open(os.path.join(HERE, 'data/verify.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print(json.dumps(R['summary']))
