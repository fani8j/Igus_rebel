"""All numerical checks of the vacuum module -> data/verify.json (+ printed summary).  Needs build.py to have run (stl/).
Run: nice -n 19 "$PY" verify.py
"""
import os, json, math, itertools
import numpy as np
import trimesh
import cadquery as cq
import params as P
import parts as K

HERE = P.HERE
OUT = lambda *a: os.path.join(HERE, *a)
R = {}
np.random.seed(7)


# ----------------------------------------------------------------------------------------------------- 1 clearances
def allowed_contact():
    A = set()

    def add(a, bs):
        for b in bs:
            A.add(frozenset((a, b)))
    add('housing', ['base_plate', 'door', 'fan', 'bulkhead_fitting', 'm12_connector'])
    add('door', ['base_plate'])
    add('base_plate', ['deck', 'mounts', 'hepa_filter', 'exhaust_fitting'] + ['spacer_%d' % i for i in range(4)] + ['m6_washer_head_%d' % i for i in range(4)])
    for i in range(4):
        A.add(frozenset(('spacer_%d' % i, 'm6_washer_head_%d' % i)))
    add('mounts', ['pump'])
    add('pump', ['elbow_in', 'elbow_out'])
    add('deck', ['dust_filter', 'muffler', 'manifold', 'vent_valve', 'vacuum_switch', 'terminal_blocks'])
    return A


def clearances():
    S = K.solids()
    names = list(S)
    A = allowed_contact()
    for n, d in S.items():
        if d['kind'] == 'tube':
            for e in d['ends']:
                A.add(frozenset((n, e)))
    soft = ('foam', 'film')                       # foam/film touch the printed shell, each other and the penetrations (design)
    shell = {'housing', 'door', 'base_plate', 'fan', 'bulkhead_fitting', 'm12_connector', 'deck'}
    bb = {n: S[n]['shape'].BoundingBox() for n in names}
    res, worst, touching = [], (1e9, None), []
    for a, b in itertools.combinations(names, 2):
        ka, kb = S[a]['kind'], S[b]['kind']
        key = frozenset((a, b))
        if key in A:
            touching.append(sorted(key))
            continue
        if ka in soft and (kb in soft or b in shell) or kb in soft and (ka in soft or a in shell):
            if not (ka in soft and kb in soft):
                touching.append(sorted(key))      # foam/film vs printed shell or its penetration: bonded / flush by design
            continue
        A_, B_ = bb[a], bb[b]
        gap = max(A_.xmin - B_.xmax, B_.xmin - A_.xmax, A_.ymin - B_.ymax, B_.ymin - A_.ymax, A_.zmin - B_.zmax, B_.zmin - A_.zmax)
        if gap > 60:
            continue
        d = S[a]['shape'].distance(S[b]['shape'])
        res.append((d, a, b))
        if d < worst[0]:
            worst = (d, (a, b))
    res.sort()
    out = dict(required_mm=P.C.CLEARANCE_MIN, min_gap_mm=round(worst[0], 3), min_pair=worst[1],
               closest_pairs=[dict(gap_mm=round(d, 2), a=a, b=b) for d, a, b in res[:14]],
               violations=[dict(gap_mm=round(d, 2), a=a, b=b) for d, a, b in res if d < P.C.CLEARANCE_MIN],
               pairs_checked=len(res), touching_by_design=len(touching),
               touching_by_design_note='mounts/pads/saddles/ports/bonded foam (listed in verify.py allowed_contact); everything else must be >= 3 mm')
    out['pass'] = not out['violations']
    return out, S


# ----------------------------------------------------------------------------------------------------- 2 foam
def foam(S):
    f = P.FOAM_T
    dims = dict(foam_roof=('z', 'zmin', 'zmax'), foam_rear=('x',), foam_left=('y',), foam_right=('y',), foam_door=('x',))
    out = {}
    for n in dims:
        b = S[n]['shape'].BoundingBox()
        t = dict(x=b.xmax - b.xmin, y=b.ymax - b.ymin, z=b.zmax - b.zmin)[dims[n][0]]
        film = S['film_' + n.split('_')[1]]['shape']
        fa = S[n]['shape'].Volume() / f                         # inner-face area of the foam (volume / thickness)
        la = film.Volume() / P.FILM_T
        out[n] = dict(thickness_mm=round(t, 3), foam_inner_face_area_mm2=round(fa), film_area_mm2=round(la), film_covers_foam=bool(la >= 0.97 * fa))
    out['pass'] = all(abs(v['thickness_mm'] - f) < 0.01 and v['film_covers_foam'] for v in out.values() if isinstance(v, dict))
    out['required_mm'] = f
    out['holder'] = 'PET film %.1f mm bonded over every foam inner face; duct partitions/inner walls and the roof/M12 sleeves are printed so no cut foam edge faces the cabin' % P.FILM_T
    return out


# ----------------------------------------------------------------------------------------------------- 3 tubes
def tubes(S):
    out = {}
    ok = True
    for n, (pts, od, ab) in K.routes().items():
        segs, margin = K.fillet_path(pts, P.BEND_R)
        nb = sum(1 for s in segs if s[0] == 'A')
        req = P.TUBE_MIN_R[od]
        bend = P.BEND_R if nb else None
        L = sum(float(np.linalg.norm(np.array(s[2]) - np.array(s[1]))) for s in segs if s[0] == 'L') + sum(0.5 * math.pi * P.BEND_R * 0 for s in segs)
        good = (bend is None or bend >= req) and margin > -1e-6
        ok &= good
        out[n] = dict(od_mm=od, bends=nb, bend_radius_mm=bend, required_mm=req, straight_margin_mm=round(margin, 1) if nb else None, ends=list(ab), ok=bool(good))
    return dict(tubes=out, pass_=bool(ok), rule='od4 >= 15, od6 >= 20, od8 >= 25 mm; all internal tubes are 8 mm; only the exhaust line to the HEPA bends (r = %.0f)' % P.BEND_R)


# ----------------------------------------------------------------------------------------------------- 4 mass / footprint / print
def mass_footprint(S):
    m = K.masses(S)
    tot = sum(m.values())
    bb = lambda names: [S[n]['shape'].BoundingBox() for n in names]
    sh = bb(['base_plate', 'housing', 'door'])
    xmin, xmax = min(b.xmin for b in sh), max(b.xmax for b in sh)
    ymin, ymax = min(b.ymin for b in sh), max(b.ymax for b in sh)
    allb = [d['shape'].BoundingBox() for d in S.values()]
    z_top = max(b.zmax for b in allb)
    zhous = max(b.zmax for b in sh)
    fp = (xmax - xmin, ymax - ymin)
    # CoG (CAD volume-weighted by item mass)
    cog = np.zeros(3)
    for n, v in m.items():
        if n in S:
            c = S[n]['shape'].Center()
            cog += v * np.array([c.x, c.y, c.z])
    cog /= sum(v for n, v in m.items() if n in S)
    return dict(
        mass_kg=dict(total=round(tot, 3), printed=round(sum(m[n] for n in K.PRINTED), 3), pump=round(m['pump'], 3), foam_film=round(sum(v for n, v in m.items() if n.startswith(('foam', 'film'))), 3), items={k: round(v, 4) for k, v in m.items()}),
        cog_module_frame_mm=[round(float(c), 1) for c in cog],
        footprint_mm=dict(x=round(fp[0], 2), y=round(fp[1], 2), target='about 220 x 170 (brief)', area_ratio_to_target=round(fp[0] * fp[1] / (220 * 170), 2)),
        height_mm=dict(housing_top_above_plate_underside=round(zhous, 1), with_arm_tube_stub=round(z_top, 1), spacers_below_plate=P.SPACER_H),
        overall_bbox_mm=dict(x=[round(xmin, 1), round(xmax, 1)], y=[round(ymin, 1), round(ymax, 1)], z=[-P.SPACER_H, round(z_top, 1)]),
        amr_slots=dict(pattern_xy_mm=list(P.AMR_SLOT_PITCH), count=4, size_mm='M6 slot %.1f long x %.1f wide, adjustable +-%.0f mm along x' % (P.AMR_SLOT_LEN + P.AMR_SLOT_W, P.AMR_SLOT_W, P.AMR_SLOT_LEN / 2)),
    )


def print_fit():
    out = {}
    ok = True
    for n in K.PRINTED:
        m = trimesh.load(OUT('stl/%s.stl' % n))
        e = m.extents
        fit = bool(e[0] <= P.BED[0] + 1e-6 and e[1] <= P.BED[1] + 1e-6 and e[2] <= P.BED_Z + 1e-6)
        ok &= fit
        out[n] = dict(print_extents_mm=[round(float(v), 1) for v in e], fits_250x250_bed=fit)
    out['orientation'] = dict(base_plate='flat, legs pointing up (standing legs, 12 x 10 mm)', housing='roof down, open side up (no support except the vent slots/bosses are bridged by design)', door='outer plate down, rim up', deck='flat, pads up')
    out['pass'] = bool(ok)
    out['parts_printed'] = len(K.PRINTED)
    out['aluminium_needed'] = 'none: all four parts print in PETG; the deck legs (12 x 10 x 137 mm) can be swapped for 12 mm aluminium square tube if they wobble'
    return out


# ----------------------------------------------------------------------------------------------------- 5 thermal
def thermal(S):
    t = {}
    area = 2 * (P.OUT_X * P.OUT_Y + P.OUT_X * P.OUT_Z + P.OUT_Y * P.OUT_Z) * 1e-6          # m2 (incl. the base plate)
    a_base = P.OUT_X * P.OUT_Y * 1e-6
    r_wall = 1 / P.H_IN + P.WALL * 1e-3 / P.K_PETG + P.FOAM_T * 1e-3 / P.K_FOAM + 1 / P.H_OUT
    r_base = 1 / P.H_IN + P.BASE_T * 1e-3 / P.K_PETG + 1 / (P.H_OUT * 0.6)                   # bottom faces the AMR deck: weaker coupling
    ua_wall = (area - a_base) / r_wall
    ua_base = a_base / r_base
    ua0 = ua_wall + ua_base
    q_fan = P.FAN_FLOW_M3H / 3600.0
    ua_vent = 1.2 * 1005.0 * q_fan
    duty = P.DUTY_ON_S / P.DUTY_PERIOD_S
    p_avg = duty * P.PUMP_P_TYP
    p_avg_max = duty * P.PUMP_P_MAX
    tau0 = P.C_THERMAL / ua0
    tau1 = P.C_THERMAL / (ua0 + ua_vent)
    t['inputs'] = dict(ambient_C=P.T_AMB, duty_on_s=P.DUTY_ON_S, period_s=P.DUTY_PERIOD_S, pump_power_typ_w=P.PUMP_P_TYP, pump_power_max_w=P.PUMP_P_MAX, assume_all_electrical_power_is_heat=True,
                       wall_R_m2K_per_W=round(r_wall, 3), outer_area_m2=round(area, 3), thermal_capacity_J_per_K=P.C_THERMAL, fan_flow_m3_per_h=P.FAN_FLOW_M3H)
    t['UA_W_per_K'] = dict(walls_and_base_only=round(ua0, 3), vent_airflow_with_fan=round(ua_vent, 2), total_with_fan=round(ua0 + ua_vent, 2))
    t['mean_heat_W'] = dict(typical=round(p_avg, 1), at_55W=round(p_avg_max, 1))
    t['steady_rise_K'] = dict(no_fan=round(p_avg / ua0, 1), with_fan=round(p_avg / (ua0 + ua_vent), 1), no_fan_at_55W=round(p_avg_max / ua0, 1), with_fan_at_55W=round(p_avg_max / (ua0 + ua_vent), 1),
                              continuous_48W_with_fan=round(P.PUMP_P_TYP / (ua0 + ua_vent), 1))
    t['time_constant_min'] = dict(no_fan=round(tau0 / 60, 0), with_fan=round(tau1 / 60, 0))
    # time-stepped (30 s period, pump on 8 s) over 4 h, lumped single node
    def sim(ua, hours=4.0, dt=0.5):
        T = 0.0
        n = int(hours * 3600 / dt)
        tmax, hist = 0.0, []
        for i in range(n):
            ph = (i * dt) % P.DUTY_PERIOD_S
            p = P.PUMP_P_TYP if ph < P.DUTY_ON_S else 0.0
            T += dt * (p - ua * T) / P.C_THERMAL
            tmax = max(tmax, T)
            if i % int(1800 / dt) == 0:
                hist.append(round(T, 1))
        return round(tmax, 1), round(T, 1), hist
    for k, ua in (('no_fan', ua0), ('with_fan', ua0 + ua_vent)):
        mx, end, hist = sim(ua)
        t['sim_4h_' + k] = dict(peak_rise_K=mx, end_rise_K=end, rise_every_30min_K=hist, peak_internal_C=round(P.T_AMB + mx, 1), pump_limit_C=P.PUMP_LIMIT_C, ok=bool(P.T_AMB + mx <= P.PUMP_LIMIT_C))
    t['fan_needed'] = bool(P.T_AMB + t['sim_4h_no_fan']['peak_rise_K'] > P.PUMP_LIMIT_C or t['steady_rise_K']['no_fan'] > 10)
    t['conclusion'] = ('%s: without a fan the cabin air would settle about %.0f K above ambient (ASSUMED duty 8 s per 30 s, 48 W) and the pump limit is +40 C ambient; '
                       'with one 24 V 40 x 10 mm fan pulling %.0f m3/h through the two baffled ducts the rise stays about %.0f K (%.0f C at %.0f C ambient); a continuous 48 W run would still reach about %.0f K, so the pump must stay duty-limited.'
                       % ('FAN NEEDED' if t['fan_needed'] else 'fan optional', t['steady_rise_K']['no_fan'], P.FAN_FLOW_M3H, t['steady_rise_K']['with_fan'], P.T_AMB + t['steady_rise_K']['with_fan'], P.T_AMB, t['steady_rise_K']['continuous_48W_with_fan']))
    t['fan_choice'] = 'quiet 24 V 40 x 40 x 10 mm fan, >= 7 m3/h free air at <= 22 dB(A) (model to choose); mounted on the -y duct opening, powered with the pump (or from a thermostat); blows INTO the cabin, exhaust duct on the +y wall at the top'
    t['caveat'] = 'single-node model, all electrical power treated as heat, no pump internal temperature rise above the cabin air (typically +5..10 K at the head): measure in the bench test'
    return t


# ----------------------------------------------------------------------------------------------------- 6 line of sight
def line_of_sight(S, n_dirs=300):
    def mesh(n):
        return K.tess(S[n]['shape'], 0.25, 0.4)
    base_shell = K.tess(cq.Workplane(obj=S['base_plate']['shape']).intersect(K.box(-130, 130, -100, 100, -1, 6.2)).val(), 0.25, 0.4)
    blockers = [mesh('housing'), mesh('door'), base_shell, mesh('bulkhead_fitting'), mesh('m12_connector'), mesh('exhaust_fitting')] + [mesh('m6_washer_head_%d' % i) for i in range(4)]
    scene = trimesh.util.concatenate(blockers)
    no_door = trimesh.util.concatenate([m for i, m in enumerate(blockers) if i != 1])
    pump = mesh('pump')

    def dirs(n):
        v = np.random.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def cast(sc, origins, nd):
        D = dirs(nd)
        O = np.repeat(origins, nd, axis=0)
        DD = np.tile(D, (len(origins), 1))
        hit = sc.ray.intersects_any(O + DD * 0.05, DD)
        return int((~hit).sum()), len(DD)

    out = {}
    pts, fi = trimesh.sample.sample_surface(pump, 300)
    pts = pts + pump.face_normals[fi] * 0.5
    esc, n = cast(scene, pts, n_dirs)
    out['from_pump_surface'] = dict(origins=len(pts), rays=n, escaping=esc)
    # origins on the two duct openings (d36 disks on the inner walls), all directions
    disks = []
    for side, zc in ((-1, P.FAN_XZ[1]), (1, P.EXH_Z)):
        r = np.sqrt(np.random.rand(150)) * 17.5
        a = np.random.rand(150) * 2 * math.pi
        y = side * (P.CAB_Y + P.DUCT_INNER_T / 2)
        disks.append(np.stack([P.FAN_XZ[0] + r * np.cos(a), np.full(150, y), zc + r * np.sin(a)], 1))
    pts2 = np.vstack(disks)
    esc2, n2 = cast(scene, pts2, n_dirs)
    out['from_duct_openings'] = dict(origins=len(pts2), rays=n2, escaping=esc2)
    # random cabin points
    pts3 = np.column_stack([np.random.uniform(P.CAB_X0 + 2, P.CAB_X1 - 2, 300), np.random.uniform(-P.CAB_Y + 2, P.CAB_Y - 2, 300), np.random.uniform(P.CAB_Z0 + 2, P.CAB_Z1 - 2, 300)])
    esc3, n3 = cast(scene, pts3, 200)
    out['from_random_cabin_points'] = dict(origins=len(pts3), rays=n3, escaping=esc3)
    # positive control: same pump rays with the door removed must escape
    escc, nc = cast(no_door, pts, n_dirs)
    out['control_door_removed'] = dict(rays=nc, escaping=escc, detects_open_path=bool(escc > 0))
    out['blockers'] = 'housing (with ducts and slots), door, base plate (z <= 6.2 only, with slots/holes), bulkhead/M12/exhaust fittings, M6 washers; foam, film, deck, fan treated as transparent (conservative)'
    out['pass'] = bool(esc == 0 and esc2 == 0 and esc3 == 0 and escc > 0)
    return out


def main():
    R['clearances'], S = clearances()
    R['foam'] = foam(S)
    R['tubes'] = tubes(S)
    R['mass_footprint'] = mass_footprint(S)
    R['print_fit'] = print_fit()
    R['thermal'] = thermal(S)
    R['line_of_sight'] = line_of_sight(S)
    R['summary'] = dict(clearances_ge_3mm=R['clearances']['pass'], foam_10mm_covered=R['foam']['pass'], tube_bends_ok=R['tubes']['pass_'], print_fits=R['print_fit']['pass'], no_line_of_sight=R['line_of_sight']['pass'],
                        fan_needed=R['thermal']['fan_needed'], mass_kg=R['mass_footprint']['mass_kg']['total'], footprint_mm=[R['mass_footprint']['footprint_mm']['x'], R['mass_footprint']['footprint_mm']['y']])
    R['assumed'] = [dict(name=a, value=v, why=w) for a, v, w in P.ASSUMED]
    json.dump(R, open(OUT('data/verify.json'), 'w'), indent=1, default=str)
    print(json.dumps(R['summary'], indent=1))
    print('min gap %.2f mm %s' % (R['clearances']['min_gap_mm'], R['clearances']['min_pair']))
    for v in R['clearances']['violations']:
        print('  VIOLATION', v)


if __name__ == '__main__':
    main()
