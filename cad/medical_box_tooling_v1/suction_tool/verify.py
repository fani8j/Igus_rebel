"""Verification of the left suction/scan tool v3.  python verify.py  -> data/verify.json (+ summary printed).
Every number comes from the CAD solids / exported files; nothing is copied from the design intent."""
import os, sys, json, itertools, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, trimesh
import cadquery as cq
from scipy.spatial import ConvexHull
import params as P
import parts as pa
import lib as L
import massprops
from meshlib import to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
Lz = P.LIP_Z
R = {}
parts = pa.build_parts()
D = {d['name']: d for d in parts}
names = [d['name'] for d in parts]
meshes = {d['name']: to_mesh(d['shape'], tol=0.05, ang=0.15) for d in parts}


def is_pad(n): return n.startswith(('pad_', 'nut_front_'))


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
                dist = a.distance(b) if hasattr(a, 'distance') else float('nan')
            except Exception:
                dist = float('nan')
            if dist < 3.0:
                tight.append((names[i], names[j], round(float(dist), 3)))
bad = [(names[i], names[j], float(M[i, j])) for i, j in itertools.combinations(range(n), 2) if not M[i, j] < 0.01]
R['interference'] = dict(parts=names, n_pairs=n * (n - 1) // 2, pairs_with_overlap=bad, max_overlap_mm3=float(M.max()),
                         matrix_mm3=np.round(M, 3).tolist(), pairs_closer_than_3mm=sorted(tight, key=lambda t: t[2]),
                         note='screws, inserts and zip ties are not modelled')
print('interference pairs', n * (n - 1) // 2, 'overlaps', len(bad), 'tight(<3mm, incl. mating contacts)', len(tight), round(time.time() - t0), 's')

# ------------------------------------------------------------------------------------------------ 2. keep-out slabs (req. 3)
zmax = {nm: D[nm]['shape'].BoundingBox().zmax for nm in names}
slab_front = Lz - P.KO_BRACKET_FRONT; slab_rear = Lz - P.KO_REAR
real = list(names)
above = sorted([nm for nm in real if zmax[nm] > slab_front + 0.05])
mid_parts = []
for nm in real:
    sl = D[nm]['shape'].intersect(L.aabox((-400, 400), (-400, 400), (slab_rear + 1e-4, slab_front)))
    if sl.Volume() > 1e-3:
        bb = sl.BoundingBox(); mid_parts.append(dict(part=nm, vol=round(sl.Volume(), 1), y=[round(bb.ymin, 2), round(bb.ymax, 2)], x=[round(bb.xmin, 2), round(bb.xmax, 2)], z=[round(bb.zmin, 2), round(bb.zmax, 2)]))
carrier_sl = D['carrier']['shape'].intersect(L.aabox((-400, 400), (-400, 400), (slab_rear + 1e-4, slab_front)))
stad = L.zstadium(0, 0, slab_rear - 1, slab_front + 1, 2 * P.BRACKET_HALF_X, 2 * P.BRACKET_HALF_Y)
carrier_extra = carrier_sl.cut(stad).Volume()
bb = carrier_sl.BoundingBox()
others_above_rear = {nm: round(zmax[nm] - slab_rear, 2) for nm in real if not is_pad(nm) and nm != 'carrier' and zmax[nm] > slab_rear + 0.05}
others_clear = {nm: round(slab_rear - zmax[nm], 2) for nm in real if not is_pad(nm) and nm != 'carrier'}
R['keepout'] = dict(
    slab_above_lip_minus_25_parts=above, ok_only_pads=all(a.startswith('pad_') for a in above),
    slab_lip_minus_40_to_minus_25_parts=mid_parts,
    carrier_in_slab_bbox=dict(x=[bb.xmin, bb.xmax], y=[bb.ymin, bb.ymax], width_y=bb.ymax - bb.ymin, z=[bb.zmin, bb.zmax]),
    carrier_in_slab_volume_outside_bracket_stadium_mm3=carrier_extra,
    ok_bracket_width_le_24=(bb.ymax - bb.ymin) <= 24.0 + 1e-6, ok_bracket_abs_y_le_12=max(abs(bb.ymin), abs(bb.ymax)) <= 12.0 + 1e-6,
    non_pad_parts_above_z_lip_minus_40_mm=others_above_rear,
    min_margin_below_z_lip_minus_40_mm=dict(sorted(others_clear.items(), key=lambda kv: kv[1])[:6]),
    note='parts in the middle slab: carrier (bracket), pads and their front nuts (the nut is captured inside the bracket pocket); washer + rear nut start at z_lip-40; the two flange-screw access holes pass through the bracket (material removed only)')
slab_ok = R['keepout']['ok_only_pads'] and carrier_extra < 0.01 and not others_above_rear and R['keepout']['ok_bracket_width_le_24']
mid_names = {p['part'] for p in mid_parts}
slab_ok = bool(slab_ok and all(nm == 'carrier' or is_pad(nm) for nm in mid_names))
R['keepout']['pass'] = slab_ok
allv = {nm: meshes[nm].vertices for nm in real}
near_ymax = max(np.abs(v[v[:, 2] > Lz - 15.0][:, 1]).max() for v in allv.values() if (v[:, 2] > Lz - 15.0).any())
R['neighbours'] = dict(max_abs_y_of_tool_geometry_within_15mm_of_lip_plane=float(near_ymax),
                       clearance_to_neighbour_face_mm={str(k): float(b['t'] / 2 - near_ymax) for k, b in P.C.BOXES.items()},
                       note='neighbour spines may stand 15 mm proud; this is tool geometry with z > z_lip-15; lean +-10 deg about z is followed by rolling the tool')
print('keepout pass', slab_ok, 'neighbour y max', round(near_ymax, 2))

# ------------------------------------------------------------------------------------------------ 3. camera
Tc = P.T_CAM; C0 = Tc[:3, 3]; ax = Tc[:3, 2]
inv = np.linalg.inv(Tc)
lips = []
for x in P.PAD_X:
    for sx in (-1, 1):
        for sy in (-1, 1):
            lips.append([x + sx * 15.4, sy * 2.4, Lz])
    lips.append([x, 0, Lz])
lips = np.array(lips)
pc = (inv @ np.column_stack([lips, np.ones(len(lips))]).T).T[:, :3]
ah = np.degrees(np.arctan2(np.abs(pc[:, 0]), pc[:, 2])); av = np.degrees(np.arctan2(np.abs(pc[:, 1]), pc[:, 2]))
hf, vf = P.CAM['depth_fov_deg']
dist_mid = float(np.linalg.norm(np.array([0, 0, Lz]) - C0))
dist_axial = float((np.array([0, 0, Lz]) - C0) @ ax)
aim = np.array(P.CAM_AIM)
axis_miss = float(np.linalg.norm(np.cross(aim - C0, ax)))
tx = np.arange(-75, 75.01, 7.5); tz = np.array([0, 10, 20, 30, 40.0]) + Lz
targets = np.array([[x, 0.0, z] for x in tx for z in tz])
origins_c = [np.array([0, 0, 0.0]), np.array([-9.0, 0, 0]), np.array([9.0, 0, 0])]     # front-face centre + the two depth lenses (18 mm baseline)
mask = np.array([not (abs(t[2] - Lz) < 1e-6 and any(abs(t[0] - px) <= 15.4 for px in P.PAD_X)) for t in targets])


def visibility(block_names):
    blockers = trimesh.util.concatenate([meshes[nm] for nm in block_names])
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(blockers)
    va, vf_, info = [], [], []
    for oc in origins_c:
        o = (Tc @ np.append(oc, 1))[:3]
        d = targets - o; dl = np.linalg.norm(d, axis=1); d /= dl[:, None]
        loc, idx_ray, _ = ray.intersects_location(np.tile(o, (len(d), 1)), d, multiple_hits=False)
        hitd = np.full(len(d), np.inf)
        for l_, i_ in zip(loc, idx_ray):
            hitd[i_] = np.linalg.norm(l_ - o)
        blocked = hitd < dl - 0.3
        va.append(1 - blocked.mean()); vf_.append(1 - blocked[mask].mean())
        info += [(float(targets[k, 0]), float(targets[k, 2] - Lz)) for k in np.where(blocked & mask)[0]]
    return float(100 * np.mean(va)), float(100 * np.mean(vf_)), info


v_all, v_free, blk = visibility([nm for nm in names if nm != 'camera_gemini305'])
R['camera'] = dict(
    pose_front_face_centre_mm=C0.tolist(), optical_axis=ax.tolist(), tilt_deg_about_x=P.CAM_TILT_DEG,
    axis_passes_aim_point_mm=P.CAM_AIM, axis_miss_distance_mm=axis_miss,
    lips_min_margin_deg=dict(horizontal=float(hf / 2 - ah.max()), vertical=float(vf / 2 - av.max())), required_margin_deg=3.0,
    range_front_face_to_middle_lip_mm=dist_mid, axial_range_mm=dist_axial, good_min_mm=P.CAM['good_min_z'], hard_min_mm=P.CAM['min_z'],
    visibility_pct_all_samples=v_all, visibility_pct_excluding_points_covered_by_a_pad=v_free, blocked_samples_not_pad_footprint=blk[:20],
    n_ray_samples=int(len(targets) * 3),
    highest_point_z=float(D['camera_gemini305']['shape'].BoundingBox().zmax), limit_z=Lz - P.KO_REAR,
    pass_=bool(dist_mid >= 90.0 and hf / 2 - ah.max() >= 3.0 and axis_miss < 0.01 and v_free >= 99.9))
print('camera range %.1f margin h %.2f v %.1f vis %.1f%% (%.1f%% excl pad footprint)' % (dist_mid, hf / 2 - ah.max(), vf / 2 - av.max(), v_all, v_free))

# ------------------------------------------------------------------------------------------------ 4. scanner + decoder
Ts = P.T_SCAN; S0 = Ts[:3, 3]; sa = Ts[:3, 2]
t_cross = -S0[1] / sa[1]
cross = S0 + sa * t_cross
blk2 = trimesh.util.concatenate([meshes[nm] for nm in names if nm != 'scanner_se4710'])
ray2 = trimesh.ray.ray_triangle.RayMeshIntersector(blk2)
E = P.ENGINE
orig, dirs = [], []
for wx in (-E['long'] / 2, 0, E['long'] / 2):
    for ww in (-E['thick'] / 2, 0, E['thick'] / 2):
        for th in np.linspace(-P.SCAN_FOV_DEG[0] / 2, P.SCAN_FOV_DEG[0] / 2, 7):
            for ph in np.linspace(-P.SCAN_FOV_DEG[1] / 2, P.SCAN_FOV_DEG[1] / 2, 7):
                dl = np.array([math.tan(math.radians(th)), math.tan(math.radians(ph)), 1.0]); dl /= np.linalg.norm(dl)
                orig.append((Ts @ np.array([wx, ww, 0.05, 1]))[:3]); dirs.append(Ts[:3, :3] @ dl)
orig = np.array(orig); dirs = np.array(dirs)
loc, ir, _ = ray2.intersects_location(orig, dirs, multiple_hits=False)
hits = {int(i): float(np.linalg.norm(l_ - orig[i])) for l_, i in zip(loc, ir)}
clip = {i: d for i, d in hits.items() if d < 300.0}
blocked_by = {}
for i in clip:
    pt = orig[i] + dirs[i] * (clip[i] + 0.01)
    best = min(((np.linalg.norm(meshes[nm].nearest.on_surface([pt])[1][0]), nm) for nm in names if nm != 'scanner_se4710'), default=(0, '?'))
    blocked_by[best[1]] = blocked_by.get(best[1], 0) + 1
win_hi = max(zmax['scanner_se4710'], zmax['scanner_arm'])
# decoder: flex leaves the engine at the rear face centre; worst case = farthest corner of the board
rear = (Ts @ np.array([0, 0, -E['depth'], 1]))[:3]
bb_d = D['decoder_pl3307']['shape'].BoundingBox()
corners = np.array([[x, y, z] for x in (bb_d.xmin, bb_d.xmax) for y in (bb_d.ymin, bb_d.ymax) for z in (bb_d.zmin, bb_d.zmax)])
dmax = float(np.linalg.norm(corners - rear, axis=1).max()); dmin = float(np.linalg.norm(corners - rear, axis=1).min())
slack = P.DECODER['flex_reach'] - dmax
R['scanner'] = dict(
    window_centre_mm=S0.tolist(), axis=sa.tolist(), tilt_deg_about_x=P.SCAN_TILT_DEG, crossing_point_y0_mm=cross.tolist(),
    distance_window_to_y0_crossing_mm=float(t_cross), required_standoff_mm=P.SCAN_STANDOFF,
    crossing_z_above_lip_plane_mm=float(cross[2] - Lz), window_behind_lip_plane_mm=float(Lz - S0[2]), required_window_behind_lip_mm=40.0,
    highest_scanner_assembly_z=float(win_hi), limit_z=Lz - P.KO_REAR,
    fov_deg=list(P.SCAN_FOV_DEG), fov_assumption='42 deg along link6 X (engine long axis), 28 deg across [ASSUMED orientation]',
    n_rays=int(len(orig)), n_rays_clipped_within_300mm=len(clip), pct_clipped=float(100 * len(clip) / len(orig)), clipped_by_part=blocked_by,
    pass_=bool(abs(t_cross - 150.0) < 0.01 and len(clip) == 0))
R['decoder'] = dict(engine_rear_flex_exit_mm=rear.tolist(), farthest_board_corner_distance_mm=dmax, nearest_board_corner_distance_mm=dmin,
                    flex_reach_mm=P.DECODER['flex_reach'], slack_over_straight_line_mm=float(slack), required_slack_mm=10.0,
                    pass_=bool(slack >= 10.0), note='straight line from the engine rear-face centre to the farthest corner of the board (connector position on the PL3307 is not known)',
                    pocket_mm=dict(x=[-pa.DEC_PX, pa.DEC_PX], y=[-P.RIB_HALF_Y - pa.DEC_PY, -P.RIB_HALF_Y], z=[pa.DEC_PZ0, pa.DEC_PZ1]))
print('scanner window behind lip %.1f, cross y0 at %.1f mm, clipped %.2f%% %s | decoder farthest %.1f mm slack %.1f mm' % (Lz - S0[2], t_cross, 100 * len(clip) / len(orig), blocked_by, dmax, slack))

# ------------------------------------------------------------------------------------------------ 5. buffer stroke sweep
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
gap_full = (Lz - 14.0 - 10.0) - (Lz - P.BRACKET_FRONT)
R['buffer_stroke'] = dict(stroke_mm=10.0, steps_checked=11, collisions_moving_head_vs_all_other_parts=[[a, b, float(v)] for (a, b), v in worst.items()],
                          gap_adapter_top_to_bracket_face_at_full_stroke_mm=float(gap_full),
                          fittings_tubes_note='the one-touch fitting and tube are on the fixed buffer top (z <= lip-61.5) and do not move with the stroke',
                          pass_=len(worst) == 0)
print('stroke collisions', len(worst), 'gap at full stroke', gap_full)

# ------------------------------------------------------------------------------------------------ 5b. tubes + collector
tube_rows = []
for i, nm in enumerate(('tube_L', 'tube_M', 'tube_R')):
    bb_t = D[nm]['shape'].BoundingBox(); inf = pa.TUBE_INFO[i]
    tube_rows.append(dict(tube=nm, length_mm=inf['length'], n_bends=len(inf['bends']), min_bend_radius_mm=min(b['radius'] for b in inf['bends']),
                          straight_out_of_fitting_mm=inf['seg_len'][0], min_straight_between_bends_mm=min(inf['seg_len'][:-1]), straight_into_port_mm=inf['seg_len'][-1],
                          z_min=bb_t.zmin, clearance_to_floor_top_mm=float(bb_t.zmin - pa.FZ1)))
cbb = D['collector']['shape'].BoundingBox(); tbb = D['trunk_tube']['shape'].BoundingBox()
tube_len_total = sum(r['length_mm'] for r in tube_rows)
carrier_bb = D['carrier']['shape'].BoundingBox()
R['tubes'] = dict(rows=tube_rows, fitting_height_mm=P.FIT_H, tube_run_height_z=pa.TUBE_Z, total_pad_tube_length_mm=tube_len_total,
                  min_head_height_limit_note='lip >= floor top + OD/2 + fitting 14.7 + 61.5 + straight 8 + bend R 12 = %.2f' % (pa.FZ1 + P.TUBE_OD / 2 - 0.05 + 61.5 + P.FIT_H + P.TUBE_STRAIGHT + P.TUBE_BEND_R),
                  pass_=bool(all(r['clearance_to_floor_top_mm'] >= 0.0 and r['min_bend_radius_mm'] >= 12 - 1e-6 and r['straight_out_of_fitting_mm'] >= 8 - 1e-6 and r['min_straight_between_bends_mm'] >= -1e-6 for r in tube_rows)))
R['collector'] = dict(code=P.COLLECTOR['code'], body_mm=P.COLLECTOR['body'], bbox_mm=[cbb.xmin, cbb.xmax, cbb.ymin, cbb.ymax, cbb.zmin, cbb.zmax],
                      port_axis_z=pa.TUBE_Z, mounted_in='flap pocket + 2 zip ties; base 54.8 (2.1 mm above the flange plane)',
                      trunk=dict(od_mm=P.TRUNK_OD, exits_along='-y', stub_end_y=tbb.ymin, distance_of_exit_from_joint6_axis_mm=abs(P.COLL_Y - P.COLLECTOR['body'][1] / 2 - P.COLLECTOR['boss6'][1]),
                                 strain_relief='2 zip-tie slits at the groove end of the flap (carrier y min %.1f)' % carrier_bb.ymin),
                      twist_note='one phi6 trunk leaves the tool: a joint-6 rotation twists/bends this single tube (it exits 39 mm off the axis; dress it with a service loop, or through the phi20 bore if the flange is hollow - unconfirmed)',
                      pass_=bool(cbb.zmax <= Lz - P.KO_REAR and cbb.zmin >= P.FLANGE_Z))
print('tubes', [(round(r['length_mm']), round(r['straight_out_of_fitting_mm'], 1)) for r in tube_rows], 'collector z', round(cbb.zmin, 1), round(cbb.zmax, 1), 'trunk y min', round(tbb.ymin, 1))

# ------------------------------------------------------------------------------------------------ 5c. vacuum-OK gate (estimate)
# Source: KNF diaphragm pump in an AMR enclosure; the switch sits at the pump inlet, the pads are ~2 m away at the end of the trunk.
vac = P.VAC
pump = vac['pump']
curve = sorted((float(a), float(q)) for a, q in pump['curve_mbar_lpm'])          # (abs mbar, standard L/min), ascending pressure
P_ATM_MBAR, P_STD_MBAR, T_K, MU_AIR, R_AIR = 1000.0, 1013.25, 293.15, 1.81e-5, 287.05   # vacuum kPa = (1000 - P_abs mbar) / 10 ; flows are ISO 8778 standard L/min
def abs_mbar(vkpa): return P_ATM_MBAR - 10.0 * vkpa
def vac_kpa(pmbar): return (P_ATM_MBAR - pmbar) / 10.0
def pump_flow(pmbar):      # standard L/min at pump inlet abs pressure (linear interpolation of the datasheet curve, clamped)
    if pmbar <= curve[0][0]: return 0.0
    if pmbar >= curve[-1][0]: return curve[-1][1]
    for (p0, q0), (p1, q1) in zip(curve, curve[1:]):
        if p0 <= pmbar <= p1: return q0 + (q1 - q0) * (pmbar - p0) / (p1 - p0)
RHO_STD = P_STD_MBAR * 100 / (R_AIR * T_K)
def section_dp(Qstd_lpm, p_in_mbar, L_mm, d_mm, n=100):
    """Darcy-Weisbach pressure drop along a tube, flowing away from p_in (towards lower pressure); steps along the tube,
    actual density rho = p/(R T) and actual volumetric flow, laminar f=64/Re (Re<2300) else Blasius 0.3164/Re^0.25. Returns (outlet abs pressure [mbar], Re)."""
    mdot = Qstd_lpm / 1000 / 60 * RHO_STD                  # kg/s, constant along the tube
    A = math.pi * (d_mm / 1000) ** 2 / 4; d = d_mm / 1000; dx = L_mm / 1000 / n
    Re = 4 * mdot / (math.pi * d * MU_AIR)
    f = 0.0 if Re <= 0 else (64.0 / Re if Re < 2300 else 0.3164 * Re ** -0.25)
    p = p_in_mbar * 100.0                                   # Pa
    for _ in range(n):
        rho = max(p, 1.0) / (R_AIR * T_K); u = mdot / (rho * A)
        p -= f * dx / d * rho * u * u / 2
        p = max(p, 1.0)
    return p / 100.0, Re
TRUNK_OPTIONS = {'A: all phi6x4, %.0f mm' % vac['trunk_len_mm']: [(vac['trunk_len_mm'], vac['trunk_id_mm'])],
                 'B: phi6x4 %.0f mm (arm) + phi8x6 %.0f mm (to AMR)' % (vac['trunk_arm_len_mm'], vac['trunk_amr_len_mm']):
                 [(vac['trunk_arm_len_mm'], vac['trunk_id_mm']), (vac['trunk_amr_len_mm'], vac['trunk_amr_id_mm'])]}
def trunk_switch_p(Q, p_pad_mbar, secs):       # abs pressure at the switch (pump end) for std flow Q leaving the collector at p_pad
    p = p_pad_mbar; Remax = 0.0
    for L_, d_ in secs:
        p, Re = section_dp(Q, p, L_, d_); Remax = max(Remax, Re)
    return p, Remax

A_tube = math.pi * (vac['tube_id_mm'] / 2) ** 2          # mm2, phi2.5 ID
V = {}
V['pad_cavity_cm3_each'] = 1.16 * 0.3                    # pad area 1.16 cm2 x 3 mm (ASSUMED depth)
V['buffer_and_fitting_bore_cm3_each'] = A_tube * (61.5 + P.FIT_H) / 1000       # phi2.5 x 76 mm (ASSUMED bore)
V['tubes_cm3_total'] = A_tube * tube_len_total / 1000
V['collector_cm3'] = P.COLLECTOR['internal_cm3']
V['filter_switch_tee_cm3'] = vac['filter_switch_tee_cm3']  # ASSUMED
V_tool = 3 * (V['pad_cavity_cm3_each'] + V['buffer_and_fitting_bore_cm3_each']) + V['tubes_cm3_total'] + V['collector_cm3']
V_trunk = {k: sum(math.pi * (d_ / 2) ** 2 * L_ / 1000 for L_, d_ in s) for k, s in TRUNK_OPTIONS.items()}
V_tot = {k: V_tool + V_trunk[k] + V['filter_switch_tee_cm3'] for k in TRUNK_OPTIONS}
Qc_line = 0.55 * 11.9 * A_tube * 0.8                       # open-end flow limit of a phi2.5 pad line: choked 11.9 L/min/mm2 x Cd 0.8 x 0.55 line-friction factor (ASSUMED)
def leak_flow(Qc, p):      # ISO 6358 style, b = 0.5; p = vacuum magnitude at the pad [kPa]
    r = (101.3 - p) / 101.3
    return Qc if r <= 0.5 else Qc * math.sqrt(max(0.0, 1 - ((r - 0.5) / 0.5) ** 2))
def Qleak(Qc_leaks, p_kpa): return sum(leak_flow(q, p_kpa) for q in Qc_leaks)
def solve(Qc_leaks, secs):
    """steady state: pad vacuum p such that leak inflow (at the pads) = pump flow at the switch pressure after trunk friction. bisection on p [kPa at the pads]."""
    lo, hi = 0.0, 99.0
    def g(p):
        Q = Qleak(Qc_leaks, p); ps, _ = trunk_switch_p(Q, abs_mbar(p), secs)
        return pump_flow(ps) - Q                            # > 0: the pump can pull more than the leaks -> vacuum rises
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0: lo = mid
        else: hi = mid
    p = 0.5 * (lo + hi); Q = Qleak(Qc_leaks, p); ps, Re = trunk_switch_p(Q, abs_mbar(p), secs)
    return p, vac_kpa(ps), Q, Re
q_seal = 1.0               # L/min leakage of one sealed pad on coated carton (ASSUMED)
A1 = 11.9 * 0.8            # L/min per mm2 (choked, Cd 0.8)
cases = {'3 pads sealed (1 L/min each)': [q_seal] * 3,
         '3 pads at 2 L/min each': [2.0] * 3,
         '3 pads on a rough carton (4 L/min each)': [4.0] * 3,
         'one pad lost (open line), 2 sealed': [Qc_line, q_seal, q_seal],
         'one pad half lost (3 mm2 leak = %.1f L/min), 2 sealed' % (3 * A1): [3 * A1, q_seal, q_seal],
         'one pad 1 mm2 leak, 2 sealed': [1 * A1, q_seal, q_seal]}
K_LOST, K_ROUGH = 'one pad lost (open line), 2 sealed', '3 pads on a rough carton (4 L/min each)'
K_HALF = 'one pad half lost (3 mm2 leak = %.1f L/min), 2 sealed' % (3 * A1)
steady = {}                     # option -> case -> dict(pads, switch)
for ok_, secs in TRUNK_OPTIONS.items():
    steady[ok_] = {}
    for k, v in cases.items():
        pp, ps, Q, Re = solve(v, secs)
        steady[ok_][k] = dict(vacuum_at_pads_kpa=round(-pp, 1), vacuum_at_switch_kpa=round(-ps, 1), total_flow_lpm=round(Q, 2), trunk_reynolds=round(Re))
THRS = (35.0, 40.0, 45.0, 50.0)
acc = vac['switch_accuracy_kpa']
thr = {}
for ok_ in TRUNK_OPTIONS:
    sw = {k: abs(steady[ok_][k]['vacuum_at_switch_kpa']) for k in cases}
    thr[ok_] = {str(int(T)): dict(threshold_kpa_at_switch=-T,
                                  margin_rough_carton_kpa=sw[K_ROUGH] - T,           # > 0: three leaky pads still pass
                                  margin_one_pad_lost_detected_kpa=T - sw[K_LOST],    # > 0: the lost-pad vacuum is weaker than the threshold -> gate fails
                                  rough_carton_passes=bool(sw[K_ROUGH] - T > 0), one_pad_lost_detected=bool(T - sw[K_LOST] > 0),
                                  both_margins_at_least_2x_switch_accuracy=bool(sw[K_ROUGH] - T >= 2 * acc and T - sw[K_LOST] >= 2 * acc),
                                  one_pad_half_lost_detected=bool(T - sw[K_HALF] > 0), one_pad_1mm2_leak_detected=bool(T - sw['one pad 1 mm2 leak, 2 sealed'] > 0))
                  for T in THRS}
# recommended gate: midpoint of (lost-pad, rough-carton) readings at the switch, worst case over both trunk options, rounded to 5 kPa
lost_max = max(abs(steady[o][K_LOST]['vacuum_at_switch_kpa']) for o in TRUNK_OPTIONS)
rough_min = min(abs(steady[o][K_ROUGH]['vacuum_at_switch_kpa']) for o in TRUNK_OPTIONS)
gate = 5.0 * round(0.5 * (lost_max + rough_min) / 5.0)
rough_stronger = bool(rough_min > gate); lost_weaker = bool(lost_max < gate)
# evacuation (lumped volume at the pad pressure, quasi-steady trunk friction; V dP/dt = -(Q_out - Q_leak) * P_std)
def evacuate(Vcm3, secs, Qc_leaks, targets=(40.0, 50.0), dt=0.0005, tmax=10.0):
    p_mbar = P_ATM_MBAR; t = 0.0; out = {}
    while t < tmax and len(out) < len(targets):
        lo, hi = 0.0, 40.0                                   # Q_out at the volume pressure: Q = pump_flow(p_switch(Q)), bisection
        for _ in range(40):
            mid = 0.5 * (lo + hi); ps, _ = trunk_switch_p(mid, p_mbar, secs) if mid > 0 else (p_mbar, 0)
            if pump_flow(ps) > mid: lo = mid
            else: hi = mid
        net = lo - Qleak(Qc_leaks, vac_kpa(p_mbar))
        p_mbar -= net / 60 * 1000 * P_STD_MBAR / Vcm3 * dt          # Q [L/min] -> cm3/s ; dP = -net * P_std / V * dt
        t += dt
        for T in targets:
            if T not in out and vac_kpa(p_mbar) >= T: out[T] = t
    return {('time_to_minus%d_kpa_s' % T): (round(out[T], 3) if T in out else None) for T in targets}
evac = {o: {'3 pads sealed': evacuate(V_tot[o], s, cases['3 pads sealed (1 L/min each)']),
            'rough carton (4 L/min each)': evacuate(V_tot[o], s, cases[K_ROUGH])} for o, s in TRUNK_OPTIONS.items()}
# energy per pick: 24 V x current x pump-on time (ASSUMED 8 s); current from the datasheet chart: 2.0 A at 500 mbar, i_max 2.3 A (bound)
t_on = vac['pump_on_s_per_pick']
energy = {'pump_on_s_ASSUMED': t_on, 'at_500mbar_current_a': pump['current_at_500mbar_a'], 'at_i_max_a': pump['i_max_a'],
          'energy_at_500mbar_J': pump['voltage'] * pump['current_at_500mbar_a'] * t_on, 'energy_at_i_max_J': pump['voltage'] * pump['i_max_a'] * t_on,
          'energy_at_500mbar_Wh': pump['voltage'] * pump['current_at_500mbar_a'] * t_on / 3600, 'energy_at_i_max_Wh': pump['voltage'] * pump['i_max_a'] * t_on / 3600,
          'note': 'pump only; valves, switch, AMR DC/DC losses not included; current at deep vacuum / low flow is probably lower than 2.0 A, so this is an upper estimate'}
R['vacuum_gate'] = dict(
    source='electric pump on the AMR (operator decision 2026-10-13): %s; 24 V, enclosure on the AMR; dust filter -> tee with %s + 24 V N.C. vent valve -> pump inlet; outlet -> HEPA filter' % (pump['code'], P.C.VACUUM_PARTS['switch']),
    switch_location=vac['switch_location'], switch_accuracy_kpa=acc, trunk_options=list(TRUNK_OPTIONS),
    volumes_cm3=V, tool_side_volume_cm3=V_tool, trunk_volume_cm3=V_trunk, total_volume_cm3=V_tot, trunk_assumption_mm=vac['trunk_len_mm'],
    steady_state_vacuum_kpa=steady, thresholds=thr, thresholds_evaluated_at='switch (pump end)',
    recommended_gate_kpa=-gate, recommended_gate_basis='midpoint of lost-pad (%.1f kPa, weakest-case trunk) and rough-carton (%.1f kPa) readings at the switch, rounded to 5 kPa' % (lost_max, rough_min),
    rough_carton_reads_stronger_than_gate=rough_stronger, one_pad_lost_reads_weaker_than_gate=lost_weaker,
    evacuation=evac, energy_per_pick=energy, pump_curve_mbar_lpm=curve,
    assumptions=['pump: %s max-speed curve read off the datasheet chart (abs inlet mbar -> standard L/min), linearly interpolated; vacuum kPa = (1000 - p_abs mbar)/10' % pump['code'],
                 'trunk friction: Darcy-Weisbach stepped along the tube at the actual (rarefied) density and volumetric flow, laminar f=64/Re (Re<2300) else Blasius; air 20 C, mu 1.81e-5 Pa s; fittings, bends, filter and valve losses NOT included',
                 'pad leak model (ISO 6358 style) unchanged; pad-tube friction only via the 0.55 factor of the open-line limit %.1f L/min (choked 11.9 L/min/mm2 x 0.8 x 0.55); "pads" = the collector end of the trunk, pad-tube drop neglected for the sealed cases' % Qc_line,
                 'sealed pad on coated carton leaks 1 L/min (rough carton 4 L/min) - ASSUMED, to be measured; leak flows taken as standard L/min',
                 'pad cavity 3 mm deep, collector 1 cm3, filter + switch + tee 10 cm3 (ASSUMED); trunk %.0f mm total, option A all phi6x4, option B %.0f mm phi6x4 + %.0f mm phi8x6' % (vac['trunk_len_mm'], vac['trunk_arm_len_mm'], vac['trunk_amr_len_mm']),
                 'evacuation: lumped volume at the pad pressure, trunk friction quasi-steady, pump at full speed curve; pump-on 8 s per pick ASSUMED',
                 'switch accuracy +-2 kPa (SMC ZSE20B, +-2 %% F.S.); the switch reads the pump-end pressure, which is stronger than the pad pressure by the trunk drop under flow'],
    pass_=bool(lost_weaker and rough_stronger))
print('vacuum V', {k: round(v, 1) for k, v in V_tot.items()}, 'gate', -gate)
for o in TRUNK_OPTIONS:
    print(o)
    for k in cases: print('  %-60s pads %6.1f switch %6.1f Q %5.1f' % (k, steady[o][k]['vacuum_at_pads_kpa'], steady[o][k]['vacuum_at_switch_kpa'], steady[o][k]['total_flow_lpm']))
    print('  evac', evac[o])

# ------------------------------------------------------------------------------------------------ 6. exports re-import
from OCP.STEPControl import STEPControl_Reader
def step_vol(path):
    r = STEPControl_Reader(); r.ReadFile(path); r.TransferRoots(); s = cq.Shape.cast(r.OneShape()); return s.Volume(), s.isValid()
imp = []; ok = True
for d in parts:
    p = os.path.join(HERE, 'step/parts/%s.step' % d['name'])
    try:
        v, valid = step_vol(p); rel = abs(v - d['shape'].Volume()) / max(d['shape'].Volume(), 1e-9)
        imp.append(dict(part=d['name'], valid=bool(valid), rel_vol_err=float(rel))); ok &= valid and rel < 1e-3
    except Exception as e:
        imp.append(dict(part=d['name'], error=str(e))); ok = False
stl_rows = []
for d in parts:
    if d['kind'] != 'printed':
        continue
    m = trimesh.load(os.path.join(HERE, 'stl/%s.stl' % d['name']))
    rel = abs(abs(m.volume) - d['shape'].Volume()) / d['shape'].Volume()
    stl_rows.append(dict(part=d['name'], watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent), rel_vol_err=float(rel), volume_positive=bool(m.volume > 0)))
    ok &= m.is_watertight and rel < 0.01
asm = cq.importers.importStep(os.path.join(HERE, 'step/assembly.step'))
n_asm = len(asm.vals()[0].Solids()) if asm.vals() else 0
exp_asm = len(parts)
ok = bool(ok and n_asm == exp_asm)
stale = sorted(set(os.listdir(os.path.join(HERE, 'step/parts'))) - {'%s.step' % d['name'] for d in parts})
stale += sorted(set(os.listdir(os.path.join(HERE, 'meshes/visual'))) - {'%s.stl' % d['name'] for d in parts})
stale += [f for f in ('ejector_envelope.step',) if os.path.exists(os.path.join(HERE, 'step', f))]
stale += sorted(set(os.listdir(os.path.join(HERE, 'stl'))) - {'%s.stl' % d['name'] for d in parts if d['kind'] == 'printed'})
R['reimport'] = dict(step_parts=imp, stl_printed=stl_rows, assembly_step_solids=n_asm, expected_assembly_solids=exp_asm, stale_files=stale, pass_=bool(ok and not stale))
print('reimport ok', ok, 'assembly solids', n_asm, 'stale', stale)

# ------------------------------------------------------------------------------------------------ 7. mass / CoG
mp_wo = massprops.assembly_props(parts, meshes=meshes)
R['mass'] = dict(total_kg=mp_wo['mass'], cog_mm=mp_wo['com'].tolist(), budget_kg=0.6,
                 printed_kg=sum(massprops.body_mass(d) for d in parts if d['kind'] == 'printed'),
                 per_body_g={k: round(v[0] * 1000, 1) for k, v in mp_wo['bodies'].items()},
                 assumptions='PETG 1.27 g/cm3 x 0.55 effective fill; purchased masses: pad incl. one-touch fitting 15 g (ASSUMED), nut 2.5 g, washer 1 g, tubes 1 g each, collector 12 g (ASSUMED), camera 68 g, scanner 4.8 g, decoder 6 g (assumed); screws/inserts NOT included (~10 g est.)',
                 payload_margin_kg=P.C.ROBOT_PAYLOAD - mp_wo['mass'])

# ------------------------------------------------------------------------------------------------ 8. deflection (first-order beams, PETG)
def deflection(E):
    tr = P.RIB_X[1] - P.RIB_X[0]; xc = P.RIB_XC; rib_A = tr * 24.0; H = (Lz - P.KO_REAR) - pa.FZ1
    I_br = 24.0 * 14.5 ** 3 / 12
    a = P.PAD_PITCH - xc
    d_br = 30.0 * a ** 3 / (3 * E * I_br)
    Frib = 30.0 * (P.PAD_PITCH + xc) / (2 * xc)
    d_rib = Frib * H / (E * rib_A)
    pull = d_br + d_rib
    lever = Lz - pa.FZ1
    I_y = 2 * tr * 24.0 ** 3 / 12
    d_lat_y = 10.0 * lever ** 3 / (3 * E * I_y)
    I_x = 2 * (24.0 * tr ** 3 / 12 + tr * 24.0 * xc ** 2)
    d_lat_x = 10.0 * lever ** 3 / (3 * E * I_x)
    return dict(pull_30N_outer_pad_mm=pull, lateral_10N_y_mm=d_lat_y, lateral_10N_x_mm=d_lat_x)
dE = {}
for lab, E_ in (('E_xy_1800MPa', P.E_PETG), ('E_z_1200MPa_interlayer_conservative', 1200.0)):
    dd = deflection(E_); dd['lip_deflection_combined_worst_mm'] = dd['pull_30N_outer_pad_mm'] + max(dd['lateral_10N_y_mm'], dd['lateral_10N_x_mm']); dE[lab] = dd
R['deflection'] = dict(target_mm=0.5, results=dE, joint_compliance_note='flange/floor joint and layer adhesion not modelled: x2 safety factor suggested',
                       with_factor_2_mm=2 * dE['E_z_1200MPa_interlayer_conservative']['lip_deflection_combined_worst_mm'],
                       pass_=bool(2 * dE['E_z_1200MPa_interlayer_conservative']['lip_deflection_combined_worst_mm'] <= 0.5))

# ------------------------------------------------------------------------------------------------ 9. lip coplanarity + span margins
lipz = {nm: float(meshes[nm].vertices[:, 2].max()) for nm in ('pad_L', 'pad_M', 'pad_R')}
span = 2 * P.PAD_PITCH + 30.8
R['lips'] = dict(lip_z=lipz, coplanarity_spread_mm=max(lipz.values()) - min(lipz.values()), pad_span_outer_edges_mm=span, pad_centres_x=P.PAD_X, pad_pitch_mm=P.PAD_PITCH)
R['pad_span_margins'] = {str(k): dict(box=b['name'], h=b['h'], margin_to_each_spine_end_mm=b['h'] / 2 - span / 2, ok_ge_25=bool(b['h'] / 2 - span / 2 >= 25.0),
                                       lateral_margin_lip_vs_spine_half_thickness_mm=b['t'] / 2 - 2.4) for k, b in P.C.BOXES.items()}
R['pad_span_margins']['min_h_220_margin_mm'] = 220 / 2 - span / 2

# ------------------------------------------------------------------------------------------------ 10. screw / assembly access in assembly order
def cyl(p0, direction, r, length):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    return L.ocyl(p0, np.asarray(p0, float) + d * length, r)
FZ1 = pa.FZ1
steps = []
def access(name, p0, direction, r, length, present):
    c = cyl(p0, direction, r, length)
    hit = {}
    for nm in present:
        v = L.common_vol(c, D[nm]['shape'])
        if v > 0.01:
            hit[nm] = round(v, 2)
    steps.append(dict(step=name, blocked_by=hit, ok=not hit))
allp = list(names)
# 1) flange screws: carrier alone on the flange, CSK head r 3.2 -> use r 3.1 through the access holes
for k in range(6):
    a_ = math.radians(P.C.PCD43_A0 + 60 * k)
    access('1 flange screw %d M3 CSK from +z (carrier alone, head clearance r 3.1)' % k, [P.C.PCD43_R * math.cos(a_), P.C.PCD43_R * math.sin(a_), FZ1 + 0.01], [0, 0, 1], 3.1, 120, ['carrier'])
# 2) pads, nuts, washers
for nm_, x in zip('LMR', P.PAD_X):
    opts = {}
    for lab, dvec, y0 in (('+y', [0, 1, 0], 7.0), ('-y', [0, -1, 0], -7.0)):
        box = L.aabox((x - 12, x + 12), (min(y0, y0 + 38 * dvec[1]), max(y0, y0 + 38 * dvec[1])), (Lz - 44.5, Lz - 40.5))   # flat spanner, 4 mm thick, 24 wide, 38 long
        hit = [nm for nm in allp if nm not in ('carrier', 'nut_rear_' + nm_, 'washer_' + nm_) and L.common_vol(box, D[nm]['shape']) > 0.01]
        opts[lab] = hit
    steps.append(dict(step='3 rear nut %s: flat spanner (4 x 24 x 38 mm) from +y or -y at z = lip-44.5..-40.5' % nm_, blocked_by=opts, ok=any(len(v) == 0 for v in opts.values())))
# 3) scanner arm M3 from -y into the rib inserts (everything except the camera arm / camera / decoder / engine present)
later = ('scanner_arm', 'scanner_se4710', 'decoder_pl3307', 'camera_arm', 'camera_gemini305')
for sx in (-1, 1):
    for z in (Lz - 60.0, Lz - 50.5):
        access('4 scanner arm M3 from -y x=%+d z=%.1f' % (sx * P.RIB_XC, z), [sx * P.RIB_XC, -P.RIB_HALF_Y - 2.9, z], [0, -1, 0], 3.6, 40, [nm for nm in allp if nm not in later])
# 4) camera arm M3 from +y
later2 = ('camera_arm', 'camera_gemini305')
for sx in (-1, 1):
    for z in (pa.ZC - 18.5, pa.ZC - 6.5):
        access('5 camera arm M3 from +y x=%+d z=%.1f' % (sx * P.RIB_XC, z), [sx * P.RIB_XC, P.RIB_HALF_Y + 2.9, z], [0, 1, 0], 3.6, 40, [nm for nm in allp if nm not in later2])
# 5) camera on the cradle: screws from below along -u
for xm in (-10, 10):
    pc0 = pa.cam_pt(xm, -21.0 - P.CAM_CRADLE_T, -12.0); dirc = -Tc[:3, 1]
    access('6 camera screw M3 x=%+d from below (-u)' % xm, pc0 + dirc * 0.1, dirc, 4.2, 40, [nm for nm in allp if nm not in ('camera_gemini305', 'camera_arm')])
# 6) decoder board slides into the arm pocket along -y (arm alone), engine slides into the cradle along the scan axis (arm alone)
dsh = D['decoder_pl3307']['shape']
hit = {}
for s_ in np.arange(0, 30.1, 3.0):
    v = L.common_vol(L.move(dsh, np.array([[1, 0, 0, 0], [0, 1, 0, s_], [0, 0, 1, 0], [0, 0, 0, 1.0]])), D['scanner_arm']['shape'])
    if v > 0.01:
        hit[round(float(s_), 1)] = round(v, 2)
steps.append(dict(step='7 decoder board slides into the pocket along -y (arm alone, 0..30 mm)', blocked_by=hit, ok=not hit))
esh = D['scanner_se4710']['shape']; hit = {}
for s_ in np.arange(0, 25.1, 2.5):
    T_ = np.eye(4); T_[:3, 3] = sa * s_
    v = L.common_vol(L.move(esh, T_), D['scanner_arm']['shape'])
    if v > 0.01:
        hit[round(float(s_), 1)] = round(v, 2)
steps.append(dict(step='8 engine slides into the cradle from the front along the scan axis (arm alone, 0..25 mm)', blocked_by=hit, ok=not hit))
w_dir = Ts[:3, 1]
for u_, a_ in ((9.25, -4.5), (-4.05, -3.3)):
    p0 = pa.scn_pt(u_, E['thick'] / 2 + 3.3, a_)
    access('9 engine M1.4 u=%+.2f driven from +w through the plate window (arm alone, r 1.4 x 16 mm)' % u_, p0, w_dir, 1.4, 16, ['scanner_arm'])
# collector drops into the flap pocket from +z (carrier + pads + arms present, tubes fitted afterwards)
csh = D['collector']['shape']; hit = {}
for s_ in np.arange(0, 40.1, 4.0):
    cm = L.move(csh, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, s_], [0, 0, 0, 1.0]]))
    for nm in allp:
        if nm in ('collector', 'tube_L', 'tube_M', 'tube_R', 'trunk_tube') + later:      # arms / engine / decoder / camera are mounted after the collector
            continue
        v = L.common_vol(cm, D[nm]['shape'])
        if v > 0.01:
            hit.setdefault(nm, []).append(round(float(s_), 1))
steps.append(dict(step='10 collector drops into the flap pocket along -z (0..40 mm above it; carrier, pads, nuts present; arms mounted afterwards)', blocked_by=hit, ok=not hit))
# a pad tube can be pushed into its fitting from below: free cylinder under each fitting
for nm_, x in zip('LMR', P.PAD_X):
    access('11 tube %s pushed into the pad fitting from below (r 3, 12 mm)' % nm_, [x, 0, pa.FIT_Z1 - 12.0], [0, 0, 1], 3.0, 12.0, [nm for nm in allp if nm not in ('tube_L', 'tube_M', 'tube_R', 'collector', 'trunk_tube', 'pad_' + nm_)])
R['screw_access'] = dict(order='1 carrier -> flange (6 x M3x12 CSK, carrier alone; 2 of them through the bracket access holes) -> 2 pads + front nuts + washers + rear nuts (spanner from the side) -> 3 collector (pocket, 2 zip ties) + 3 pad tubes + trunk -> 4 decoder board into the scanner-arm pocket, engine into the cradle (+ 2 x M1.4) -> 5 scanner arm (4 x M3, closes the pocket against the carrier web) -> 6 camera arm (4 x M3) -> 7 camera (2 x M3 from below)',
                         checks=steps, all_ok=all(s['ok'] for s in steps))
print('screw access ok', R['screw_access']['all_ok'], [s['step'] for s in steps if not s['ok']])

# ------------------------------------------------------------------------------------------------ 11. printability
def wall_check(m, n=3000):
    pts, fi = trimesh.sample.sample_surface(m, n, seed=1)
    nrm = m.face_normals[fi]
    rr = trimesh.ray.ray_triangle.RayMeshIntersector(m)
    loc, ir, _ = rr.intersects_location(pts - nrm * 0.02, -nrm, multiple_hits=False)
    th = np.full(n, np.inf)
    for l_, i_ in zip(loc, ir):
        th[i_] = np.linalg.norm(l_ - pts[i_])
    th = th[np.isfinite(th)]
    return dict(min_mm=float(th.min()), p01_mm=float(np.percentile(th, 1)), frac_below_2p9mm=float((th < 2.9).mean()))
CAND = {'carrier': {'flange face down (z up)': [0, 0, 1], 'bracket front face down (z flipped)': [0, 0, -1]},
        'camera_arm': {'attach face down (+y)': [0, 1, 0], 'attach face down (rev)': [0, -1, 0], 'cradle plate down': list(-P.T_CAM[:3, 1])},
        'scanner_arm': {'attach face down (-y)': [0, -1, 0], 'attach face down (rev)': [0, 1, 0]}}
def support_est(m, b):
    b = np.asarray(b, float); b /= np.linalg.norm(b)
    h = m.vertices @ b; h0 = h.min()
    cen = m.triangles.mean(1); area = m.area_faces; nb = m.face_normals @ b; hc = cen @ b - h0
    over = np.where((nb < -0.7071) & (hc > 0.2))[0]
    rr = trimesh.ray.ray_triangle.RayMeshIntersector(m)
    o = cen[over] - b * 0.05
    loc, ir, _ = rr.intersects_location(o, np.tile(-b, (len(o), 1)), multiple_hits=False)
    dist = hc[over].copy()
    for l_, i_ in zip(loc, ir):
        dist[i_] = np.linalg.norm(l_ - o[i_])
    sup = float((area[over] * (-nb[over]) * dist).sum())
    bed = float(area[(nb < -0.99) & (hc < 0.2)].sum())
    return dict(build_dir_link6=b.tolist(), bed_contact_mm2=bed, overhang_area_mm2=float(area[over].sum()), support_volume_est_cm3=sup / 1000, height_mm=float(h.max() - h0))
pr = {}
for d in parts:
    if d['kind'] != 'printed':
        continue
    m = meshes[d['name']]
    cands = {k: support_est(m, b) for k, b in CAND[d['name']].items()}
    best = min(cands, key=lambda k: cands[k]['support_volume_est_cm3'])
    bbx = d['shape'].BoundingBox()
    pr[d['name']] = dict(candidates=cands, best=best, support_pct_of_part_best=100 * cands[best]['support_volume_est_cm3'] * 1000 / d['shape'].Volume(), walls=wall_check(m),
                         bbox_xyz_mm=[bbx.xlen, bbx.ylen, bbx.zlen])
R['printability'] = dict(per_part=pr, method='overhang = faces > 45 deg from vertical; support volume = overhang area x free drop below each face (ray to the part or the bed; solid-support upper bound, tree/organic supports use ~30-50 %); wall = inward ray from 3000 surface samples (edge samples read thin)',
                         bed_note='carrier x extent %.0f mm (floor flap / bracket 144 mm): fits a 220 x 220 mm bed' % D['carrier']['shape'].BoundingBox().xlen)

# ------------------------------------------------------------------------------------------------ 12. compactness vs v1
def env(sel):
    V = np.vstack([meshes[nm].vertices for nm in sel]); h = ConvexHull(V)
    return dict(bbox_min=V.min(0).round(2).tolist(), bbox_max=V.max(0).round(2).tolist(), size=(V.max(0) - V.min(0)).round(2).tolist(), bbox_volume_cm3=float(np.prod(V.max(0) - V.min(0)) / 1000), hull_cm3=float(h.volume / 1000))
comp = dict(default=env(list(names)), printed_only=env([d['name'] for d in parts if d['kind'] == 'printed']),
            flange_to_lip_z_mm=Lz - P.FLANGE_Z, n_printed=sum(d['kind'] == 'printed' for d in parts), n_purchased=sum(d['kind'] == 'purchased' for d in parts),
            mass_kg=mp_wo['mass'], camera_range_mm=dist_mid)
for tag, fn in (('v1', 'v1_baseline.json'), ('v2', 'v2_baseline.json')):
    base = os.path.join(P.SCRATCH, fn)
    if os.path.exists(base):
        b1 = json.load(open(base))
        comp[tag] = dict(default=b1['all_no_ejector'], printed_only=b1['printed_only'], flange_to_lip_z_mm=b1['lip_z'] - P.FLANGE_Z,
                         n_printed=b1['n_printed'], n_purchased=b1['n_purchased'], mass_kg=b1['mass_no_ej'])
R['compactness'] = comp
print('bbox v3', comp['default']['size'], 'hull %.0f cm3' % comp['default']['hull_cm3'], 'mass %.3f' % comp['mass_kg'])

R['summary'] = dict(lip_z=Lz, pitch=P.PAD_PITCH, overall=dict(interference_ok=len(bad) == 0, keepout_ok=bool(slab_ok), stroke_ok=len(worst) == 0, reimport_ok=bool(R['reimport']['pass_']),
                    camera_ok=R['camera']['pass_'], scanner_ok=R['scanner']['pass_'], decoder_ok=R['decoder']['pass_'], screw_access_ok=R['screw_access']['all_ok'],
                    collector_ok=R['collector']['pass_'], tubes_ok=R['tubes']['pass_'], vacuum_gate_ok=R['vacuum_gate']['pass_'], deflection_ok=R['deflection']['pass_']))
os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
json.dump(R, open(os.path.join(HERE, 'data/verify.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('written data/verify.json', json.dumps(R['summary']))
