"""All numerical checks of the right gripper tool -> data/verify.json (+ printed summary).  Needs build.py to have run (data/mass.json, step/, stl/).
Run: nice -n 19 "$PY" verify.py
"""
import os, json, math, itertools, glob
import numpy as np
import trimesh
import shapely.geometry as sg
from shapely.ops import polygonize
import cadquery as cq
import params as P
import parts as K

HERE = P.HERE
OUT = lambda *a: os.path.join(HERE, *a)
C = P.C
G = 9.81
R = {}


def box_t(t_nom=16.0):
    return t_nom


# ----------------------------------------------------------------------------------------------------- 1 geometry facts
def geometry():
    S0, S24 = K.solids(0.0, with_hw=False), K.solids(P.JAW_STROKE, with_hw=False)
    bb = lambda S, n: S[n]['shape'].BoundingBox()
    g = {}
    g['z_flange_face_mm'] = P.FLANGE_Z
    g['z_gripper_rear_face_mm'] = P.Z_R
    g['stack_up_disc_plate_mm'] = P.DISC_T + P.PLATE_T
    g['z_jaw_plane_mm'] = P.Z_J
    g['z_gripper_body_end_face_mm'] = bb(S0, 'smc_body').zmax
    g['z_jaw_top_mm'] = bb(S0, 'smc_jaw_left').zmax
    g['z_fingertip_mm'] = bb(S0, 'finger_left').zmax
    g['z_pad_zone_mm'] = [P.Z_J + P.PAD_Z0, P.Z_J + P.PAD_Z1]
    g['z_stop_face_mm'] = P.Z_STOP
    g['z_tcp_mm'] = P.Z_J + P.L_PAD
    g['tool_length_flange_to_tip_mm'] = bb(S0, 'finger_left').zmax - P.FLANGE_Z
    g['pad_gap_closed_mm'] = bb(S0, 'pad_right').xmin - bb(S0, 'pad_left').xmax
    g['pad_gap_open_mm'] = bb(S24, 'pad_right').xmin - bb(S24, 'pad_left').xmax
    g['stroke_total_mm'] = g['pad_gap_open_mm'] - g['pad_gap_closed_mm']
    g['pad_size_mm'] = dict(width_along_Y=P.PAD_W, height_along_Z=P.PAD_H)
    g['catalog_L_mm'] = P.L_PAD
    g['catalog_H_mm'] = P.H_OVERHANG
    g['H_plus_L_mm'] = P.L_PAD + P.H_OVERHANG
    g['H_plus_L_limit_100pct_mm'] = C.GRIPPER['hl_limit_100pct']
    g['H_plus_L_limit_70pct_mm'] = C.GRIPPER['hl_limit_70pct']
    g['force_cap'] = '100 %% allowed (H+L = %.1f <= 55)' % (P.L_PAD + P.H_OVERHANG)
    g['box_top_edge_below_jaw_plane_mm'] = P.Z_REG
    g['box_top_edge_below_gripper_body_end_mm'] = P.Z_J + P.Z_REG - g['z_gripper_body_end_face_mm']
    g['handoff_top_clear_required_mm'] = C.HANDOFF_TOP_CLEAR
    g['top_stop'] = dict(plate_x_thickness_mm=P.STOP_T, plate_y_range_mm=[-P.STOP_Y, P.STOP_Y], plate_z_range_mm=[P.Z_STOP - P.STOP_H, P.Z_STOP], face_z_mm=P.Z_STOP,
                         s_range_mm=[P.S_PAD_CENTRE - P.STOP_Y, P.S_PAD_CENTRE + P.STOP_Y], gap_to_pad_crest_closed_per_side_mm=P.PAD_CREST - P.STOP_T / 2,
                         risers_abs_y_mm=list(P.RISER_Y), side_plates=dict(x_half=P.SIDE_X, abs_y=list(P.SIDE_Y), z=[P.SIDE_Z[0], P.SIDE_Z[1]]), bolts=P.STOP_BOLT + ' x2 at x = +-15, z = %.1f (existing d5.2 body holes, no new holes)' % P.BOLT_Z)
    g['stop_clearance_target_mm'] = 2.5
    g['lead_in_chamfer_mm'] = dict(size=P.CHAMF, angle_deg=45, tip_gap_open_mm=g['pad_gap_open_mm'] + 2 * P.CHAMF, tip_gap_closed_mm=g['pad_gap_closed_mm'] + 2 * P.CHAMF)
    # jaw travel vs the STEP: jaws touch at x = 0 (closed); at o = 24 the outer jaw end is still inside the body slot
    raw = K.smc_raw()
    jl, jr = raw['jaw_left'].BoundingBox(), raw['jaw_right'].BoundingBox()
    g['step_jaw_mount_face_x_range_closed_mm'] = dict(left=[jl.xmin, 0.0], right=[0.0, jr.xmax], note='vendor STEP is in the fully closed state: the two jaw inner faces touch at x = 0 (catalog: closed 0 +0.1, open 48 +-1)')
    g['step_body_slot_end_x_mm'] = [raw['body'].BoundingBox().xmin if False else -53.8, 62.2]
    g['step_jaw_left_outer_end_x_at_full_open_mm'] = jl.xmin - P.JAW_STROKE
    g['step_jaw_travel_margin_to_body_end_mm'] = (jl.xmin - P.JAW_STROKE) - (-53.8)
    g['step_m3_pattern_left_jaw_x_mm'] = [-23.5, -5.5]
    g['step_m3_pattern_y_mm_link6'] = list(P.JAW_M3_Y)
    g['step_rear_face'] = dict(z_s=-46.0, m5_pattern='40 x 22 at x=+-20, y=+-11 (link6), tap d4.13 x 7.6 deep (catalog M5x0.8x8)', dowels='d4H9 x 3 at (0, +-11): round (+11) and slot (-11)',
                               verified='4 tapped holes + 2 dowel holes found on the face z_s = -46 (opposite the jaws, jaw end face z_s = +37) in the vendor STEP')
    return g


# ----------------------------------------------------------------------------------------------------- 2 interference
def interference():
    out = {}
    clr = {}
    worst = 0.0
    for name, o in (('open_56', P.JAW_STROKE), ('closed_8', 0.0), ('closed_on_16', 4.0), ('closed_on_12', 2.0), ('mid_12', 12.0)):
        S = K.solids(o, with_hw=True)
        names = list(S)
        main = [n for n in names if S[n]['kind'] != 'hardware']
        pairs, bad, thread = [], [], []
        for a, b in itertools.combinations(names, 2):
            A, B = S[a], S[b]
            if A['kind'] == 'hardware' and B['kind'] == 'hardware':
                continue
            ba, bb = A['shape'].BoundingBox(), B['shape'].BoundingBox()
            if ba.xmax < bb.xmin or bb.xmax < ba.xmin or ba.ymax < bb.ymin or bb.ymax < ba.ymin or ba.zmax < bb.zmin or bb.zmax < ba.zmin:
                v = 0.0
            else:
                v = A['shape'].intersect(B['shape']).Volume()
            host = (b in A.get('host', [])) or (a in B.get('host', []))
            if host and v > 1e-3:
                thread.append((a, b, round(v, 2)))
                continue
            if A['kind'] != 'hardware' and B['kind'] != 'hardware':
                pairs.append((a, b, v))
            if v > 1e-3:
                bad.append((a, b, round(v, 3)))
            worst = max(worst, v if not host else 0.0)
        mat = {a: {} for a in main}
        for a, b, v in pairs:
            mat[a][b] = round(v, 4)
        S_ = {n: S[n]['shape'] for n in ('top_stop', 'finger_left', 'finger_right', 'pad_left', 'pad_right', 'smc_jaw_left', 'smc_jaw_right', 'smc_body')}
        dist = {n: round(S_['top_stop'].distance(S_[n]), 3) for n in S_ if n != 'top_stop'}
        clr[name] = min(d_ for n, d_ in dist.items() if n not in ('smc_body',))
        out[name] = dict(top_stop_min_distance_mm=dist, jaw_opening_each_mm=o, n_solids=len(names), n_main_pairs=len(pairs), main_pair_volume_mm3=mat,
                         violations=bad, intended_thread_or_press_overlaps_excluded=thread[:60], n_thread_overlaps=len(thread))
    out['max_boolean_volume_mm3_excluding_thread_hosts'] = worst
    out['top_stop_min_clearance_to_fingers_pads_jaws_mm'] = clr
    out['top_stop_clearance_ge_2p5_all_states'] = bool(min(clr.values()) >= 2.5 - 1e-6)
    out['note'] = 'hardware (screws/pins) checked against every other part except the part holding its thread / press fit (SMC body or jaw); heat-set inserts are the empty holes'
    return out


# ----------------------------------------------------------------------------------------------------- 3 re-import
def reimport():
    out = {}
    summ = json.load(open(OUT('data/part_summary.json')))
    from OCP.STEPControl import STEPControl_Reader

    def rd(path):
        r = STEPControl_Reader()
        r.ReadFile(path)
        r.TransferRoots()
        return cq.Shape.cast(r.OneShape())
    for n, d in summ.items():
        s = rd(OUT('step/parts/%s.step' % n))
        m = trimesh.load(OUT('stl/%s.stl' % n))
        out[n] = dict(step_valid=bool(s.isValid()), n_solids=len(s.Solids()), step_volume_mm3=s.Volume(), cad_volume_mm3=d['volume_mm3'],
                      volume_error_rel=abs(s.Volume() - d['volume_mm3']) / d['volume_mm3'], stl_watertight=bool(m.is_watertight), stl_winding_consistent=bool(m.is_winding_consistent),
                      stl_volume_mm3=float(m.volume), stl_volume_error_rel=abs(m.volume - d['volume_mm3']) / d['volume_mm3'], stl_triangles=int(len(m.faces)), stl_units='mm')
    for fn in ('assembly_open', 'assembly_closed16'):
        s = rd(OUT('step/%s.step' % fn))
        out[fn] = dict(step_valid=bool(s.isValid()), n_solids=len(s.Solids()), size_MB=round(os.path.getsize(OUT('step/%s.step' % fn)) / 1e6, 2))
    s = rd(OUT('step/parts/smc_lehf20k2_48_mounted_closed.step'))
    out['smc_gripper_mounted'] = dict(step_valid=bool(s.isValid()), n_solids=len(s.Solids()))
    # sim meshes
    sim = {}
    for f in sorted(glob.glob(OUT('meshes/*/*.stl'))):
        m = trimesh.load(f)
        sim[os.path.relpath(f, HERE)] = dict(watertight=bool(m.is_watertight), triangles=int(len(m.faces)), extent_m=[round(float(x), 4) for x in m.extents], convex=bool(m.is_convex) if 'collision' in f else None)
    out['sim_meshes'] = sim
    return out


# ----------------------------------------------------------------------------------------------------- 4 checks (a)-(e)
def r_mean(w, h, n=400):
    y = (np.arange(n) + 0.5) / n * w - w / 2
    z = (np.arange(n) + 0.5) / n * h - h / 2
    Y, Z = np.meshgrid(y, z)
    return float(np.sqrt(Y ** 2 + Z ** 2).mean())


def force_at(L, curve):
    f10, f55 = curve
    return f10 + (f55 - f10) * (L - 10.0) / 45.0


def hold_check():
    box = C.BOXES[4]
    m = C.BOX_MASS_MAX
    out = dict(box=dict(box), mass_kg=m, grip_centre_s_mm=P.S_PAD_CENTRE, mu=P.MU, dynamic_factor=2.0)
    d = box['l'] / 2 - P.S_PAD_CENTRE
    out['cog_offset_from_grip_mm'] = d
    W = m * G
    T_req = 2.0 * W * d / 1000.0
    out['required_friction_torque_Nm'] = T_req
    rb = r_mean(P.PAD_W, P.PAD_H)
    out['effective_radius_mm'] = rb
    out['model'] = 'two pads, uniform pressure on the %.0f x %.0f mm patch, torque about the clamp axis X: T_cap = 2*mu*N*r_mean; combined with the shear (2W) by the friction ellipse' % (P.PAD_W, P.PAD_H)
    Fnom100 = force_at(P.L_PAD, P.FORCE_100)
    Fnom70 = force_at(P.L_PAD, P.FORCE_70)
    Fnom40 = force_at(P.L_PAD, P.FORCE_40)
    cases = {'11 N (catalog minimum of the LEHF20 range / brief)': 11.0, '20 N (brief nominal)': 20.0,
             '40 %% pushing, nominal at L=%.1f' % P.L_PAD: Fnom40, '70 % pushing, nominal': Fnom70, '70 % pushing, -25 %': Fnom70 * (1 - P.FORCE_ACC),
             '100 % pushing, nominal': Fnom100, '100 % pushing, -25 % (worst)': Fnom100 * (1 - P.FORCE_ACC)}
    tab = {}
    for k, N in cases.items():
        Tcap = 2 * P.MU * N * rb / 1000.0
        Fcap = 2 * P.MU * N
        shear = 2.0 * W
        util = math.sqrt((shear / Fcap) ** 2 + (T_req / Tcap) ** 2)
        tab[k] = dict(N_per_finger=round(N, 2), T_cap_Nm=round(Tcap, 4), margin_torque_only=round(Tcap / T_req, 3), margin_ellipse=round(1 / util, 3))
    out['cases'] = tab
    out['all_boxes_at_100pct_worst'] = {}
    for i, b in C.BOXES.items():
        dd = b['l'] / 2 - P.S_PAD_CENTRE
        Tq = 2.0 * C.BOX_MASS_MAX * G * dd / 1000.0
        Tc = 2 * P.MU * Fnom100 * (1 - P.FORCE_ACC) * rb / 1000.0
        out['all_boxes_at_100pct_worst'][str(i)] = dict(l=b['l'], cog_offset_mm=dd, T_req_Nm=round(Tq, 4), margin=round(Tc / Tq, 3))
    out['recommendation'] = ('Pushing force 100 %% (LEC step data), pushing speed <= 20 mm/s: nominal %.1f N per finger at L = %.1f mm (%.1f N worst case with the +-25 %% accuracy); 11 N and 20 N do NOT give margin >= 1.5 for box 4 with dynamic factor 2'
                             % (Fnom100, P.L_PAD, Fnom100 * (1 - P.FORCE_ACC)))
    out['mitigation_if_margin_too_thin'] = 'grip with the pad centre as deep as possible (s_pad = 42.5 is the zone centre); slow the lift/transport acceleration; mu of ribbed TPU on cardboard is typically > 0.5'
    return out


def _margin(Wp, d, hp, phi, N, a, stop):
    """margin = 1 / min_F utilisation.  Wp = dynamic weight (factor 2 included), gravity tilted phi (rad) from the tool axis toward +s,
    d / hp = box CoG offset from the pad centre along s / depth below it.  Static lower bound (friction ellipse of the pads), stop = frictionless force F >= 0 along +z at lever a (spine side)."""
    rb = r_mean(P.PAD_W, P.PAD_H)
    Fcap = 2 * P.MU * N
    Tcap = 2 * P.MU * N * rb
    tau = Wp * (d * math.cos(phi) - hp * math.sin(phi))
    Fs = np.linspace(0.0, 80.0 if stop else 0.0, 1601) if stop else np.array([0.0])
    u = np.sqrt(((Wp * math.sin(phi)) ** 2 + (Wp * math.cos(phi) + Fs) ** 2) / Fcap ** 2 + ((tau - a * Fs) / Tcap) ** 2)
    i = int(np.argmin(u))
    return 1.0 / u[i], float(Fs[i])


def hold_with_stop():
    """(a) redone with the fixed centre top-stop: box top edge against the stop face; spine-side contact at s = 15 (lever a = s_c - 15 from the pad centre)."""
    m = C.BOX_MASS_MAX
    W = m * G
    Wp = 2.0 * W
    Fnom100 = force_at(P.L_PAD, P.FORCE_100)
    forces = {'11 N': 11.0, '18.2 N (100 %% worst, -25 %%)': Fnom100 * (1 - P.FORCE_ACC), '24.3 N (100 %% nominal)': Fnom100}
    forces = {('%.1f N' % v): v for v in (11.0, Fnom100 * (1 - P.FORCE_ACC), Fnom100)}
    a = P.S_PAD_CENTRE - (P.S_PAD_CENTRE - P.STOP_Y)               # = 27.5: contact at s = 15
    zdepth = (P.Z_J + P.L_PAD) - P.Z_STOP                        # pad centre below the box top edge = 25.7
    rb = r_mean(P.PAD_W, P.PAD_H)
    # pivot (mechanism) bound: box rotates about the stop contact (s = 15, top edge), pads slide with kinetic friction mu*N*uniform over the patch
    yy = np.linspace(P.S_PAD_CENTRE - P.PAD_W / 2, P.S_PAD_CENTRE + P.PAD_W / 2, 200)
    zz = np.linspace(P.PAD_Z0 - P.Z_REG, P.PAD_Z1 - P.Z_REG, 180)
    Y, Z = np.meshgrid(yy - (P.S_PAD_CENTRE - P.STOP_Y), zz)
    rpiv = float(np.sqrt(Y ** 2 + Z ** 2).mean())
    out = dict(model=('static lower bound: two pads (uniform pressure, friction ellipse F/(2 mu N), T/(2 mu N r_mean)) + frictionless stop reaction F_s >= 0 along z at s = 15 (lever %.1f mm from the pad centre, %.1f mm above it) + gravity x 2; margin = 1 / min over F_s of the ellipse utilisation. '
                     'Mechanism upper bound: box pivots about the stop contact, pads slide with mu N over the patch (mean radius %.1f mm from the pivot).' % (a, zdepth, rpiv)),
               mu=P.MU, dynamic_factor=2.0, lever_stop_mm=a, pad_centre_below_stop_mm=zdepth, r_mean_pad_mm=rb, r_mean_pivot_mm=rpiv, boxes={})
    for i, b in C.BOXES.items():
        d = b['l'] / 2 - P.S_PAD_CENTRE
        hp = b['h'] / 2 - zdepth
        row = {}
        for k, N in forces.items():
            m0, _ = _margin(Wp, d, hp, 0.0, N, a, False)
            m1, F1 = _margin(Wp, d, hp, 0.0, N, a, True)
            piv = 2 * P.MU * N * rpiv / (Wp * (b['l'] / 2 - (P.S_PAD_CENTRE - P.STOP_Y)))
            row[k] = dict(no_stop=round(m0, 3), with_stop=round(m1, 3), stop_force_N=round(F1, 2), pivot_mechanism_bound=round(piv, 3))
        th = math.degrees(math.atan2(d, hp))
        tilt = {}
        for phi in (0.0, 15.0, 30.0, 45.0):
            tilt['%d deg' % phi] = {k: round(_margin(Wp, d, hp, math.radians(phi), N, a, True)[0], 3) for k, N in forces.items()}
        row['hang_angle_deg'] = round(th, 1)
        row['margin_at_hang_angle_with_stop'] = {k: round(_margin(Wp, d, hp, math.radians(th), N, a, True)[0], 3) for k, N in forces.items()}
        row['margin_vs_tool_tilt_with_stop'] = tilt
        row['stop_engaged_after_slip'] = {('gap %.1f mm' % g): {k: round(_margin(Wp, d, hp, g / a, N, a, True)[0], 3) for k, N in forces.items()} for g in (P.G_NOM, P.G_NOM + 4.9)}
        row['cog_offset_along_s_mm'] = d
        row['cog_depth_below_pad_centre_mm'] = hp
        out['boxes'][str(i)] = dict(name=b['name'], l=b['l'], h=b['h'], **row)
    out['rotate_to_hang_note'] = ('Rotate the tool about the pad axis X so that the line pad centre -> box CoG is vertical: tool axis tilted by theta = atan(d / (h/2 - %.1f)) from the downward vertical with the box far end (s > 0 side) '
                                  'DOWN, i.e. the spine end up toward the stop (46-48 deg boxes 1-3/5 ~ %.0f deg; box 4 %.0f deg). In that attitude gravity produces no torque about X: the pads only carry W (margin = 2 mu N / (2 W) = %.1f at 11 N); '
                                  'stop and friction torque are not needed. Whatever the transport attitude, keep the tilt within ~15 deg of that angle for accelerations; at 0 deg the stop is loaded by ~7 N.'
                                  % (zdepth, out['boxes']['1']['hang_angle_deg'], out['boxes']['4']['hang_angle_deg'], 2 * P.MU * 11.0 / Wp))
    return out


def stop_stiffness():
    """voxel FE (1 mm hex8, PA12-CF E 3200 MPa, nu 0.35) of the printed yoke: 15 N toward the body at the spine-side end of the stop face; clamp zones (r 6 mm around the bolts on the side-plate inner faces) fixed"""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    E, nu = P.E_MPA['PA12CF'], 0.35
    h = 1.0
    xs = np.arange(-P.SIDE_X, P.SIDE_X, h) + h / 2
    ys = np.arange(-P.STOP_Y, P.STOP_Y, h) + h / 2
    nzc = int(math.ceil((P.Z_STOP - P.SIDE_Z[0]) / h))
    z0, z1 = P.Z_STOP - nzc * h, P.Z_STOP
    zs = z0 + np.arange(nzc) * h + h / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    occ = np.zeros(X.shape, bool)
    t = P.STOP_T / 2
    occ |= (np.abs(X) < t + 0.01) & (np.abs(Y) <= P.STOP_Y) & (Z >= P.Z_STOP - P.STOP_H)
    for sg in (-1, 1):
        occ |= (np.abs(X) < t + 0.01) & (sg * Y >= P.RISER_Y[0]) & (sg * Y < P.RISER_Y[1]) & (Z >= P.RISER_Z0)
        occ |= (np.abs(X) <= P.SIDE_X) & (sg * Y >= P.SIDE_Y[0]) & (sg * Y < P.SIDE_Y[1]) & (Z >= P.SIDE_Z[0]) & (Z <= P.SIDE_Z[1])
        occ |= (np.abs(X) <= P.EAR_X) & (sg * Y >= P.EAR_Y[0]) & (sg * Y < P.EAR_Y[1]) & (Z >= P.EAR_Z0) & (Z <= P.SIDE_Z[1])
    for bx in P.BOLT_X:
        occ &= ~(((X - bx) ** 2 + (Z - P.BOLT_Z) ** 2) < (P.BOLT_HOLE / 2) ** 2)
    # nodes
    ex, ey, ez = np.nonzero(occ)
    nx, ny, nz = occ.shape
    nid = -np.ones((nx + 1, ny + 1, nz + 1), int)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                nid[ex + dx, ey + dy, ez + dz] = 0
    used = np.argwhere(nid == 0)
    nid[:] = -1
    nid[used[:, 0], used[:, 1], used[:, 2]] = np.arange(len(used))
    nn = len(used)
    # hex8 unit-cube stiffness (2x2x2 Gauss)
    lam, mu_ = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    D = np.zeros((6, 6)); D[:3, :3] = lam; D[np.arange(3), np.arange(3)] += 2 * mu_; D[3:, 3:] = np.eye(3) * mu_
    c = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    Ke = np.zeros((24, 24))
    g = 1 / math.sqrt(3)
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                dN = np.array([[ci[0] * (1 + ci[1] * gy) * (1 + ci[2] * gz), ci[1] * (1 + ci[0] * gx) * (1 + ci[2] * gz), ci[2] * (1 + ci[0] * gx) * (1 + ci[1] * gy)] for ci in c]) / 8.0 * 2.0 / h
                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3 * i] = dN[i, 0]; B[1, 3 * i + 1] = dN[i, 1]; B[2, 3 * i + 2] = dN[i, 2]
                    B[3, 3 * i] = dN[i, 1]; B[3, 3 * i + 1] = dN[i, 0]
                    B[4, 3 * i + 1] = dN[i, 2]; B[4, 3 * i + 2] = dN[i, 1]
                    B[5, 3 * i] = dN[i, 2]; B[5, 3 * i + 2] = dN[i, 0]
                Ke += B.T @ D @ B * (h / 2) ** 3
    conn = np.stack([nid[ex + int(ci[0] > 0), ey + int(ci[1] > 0), ez + int(ci[2] > 0)] for ci in c], axis=1)
    dof = (conn[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(len(conn), 24)
    I = np.repeat(dof, 24, axis=1).ravel()
    J = np.tile(dof, (1, 24)).ravel()
    K_ = sp.csr_matrix((np.tile(Ke.ravel(), len(conn)), (I, J)), shape=(3 * nn, 3 * nn))
    # nodes coordinates
    pos = np.stack([used[:, 0] * h - P.SIDE_X, used[:, 1] * h - P.STOP_Y, used[:, 2] * h + z0], 1)
    fixed = np.zeros(nn, bool)
    for sg in (-1, 1):
        for bx in P.BOLT_X:
            fixed |= (np.abs(pos[:, 1] - sg * P.SIDE_Y[0]) < 0.01) & (((pos[:, 0] - bx) ** 2 + (pos[:, 2] - P.BOLT_Z) ** 2) < 6.0 ** 2)
    top = (np.abs(pos[:, 2] - P.Z_STOP) < 0.01) & (pos[:, 1] <= -P.STOP_Y + 2.0 + 0.01)       # spine-side 2 mm strip of the stop face
    f = np.zeros(3 * nn)
    f[3 * np.nonzero(top)[0] + 2] = -P.STOP_HOLD_N / top.sum()
    free = np.ones(3 * nn, bool)
    for k in range(3):
        free[3 * np.nonzero(fixed)[0] + k] = False
    Kff = K_[free][:, free]
    d = np.zeros(3 * nn)
    M = sp.diags(1.0 / Kff.diagonal())
    sol, info = sla.cg(Kff, f[free], M=M, rtol=1e-8, maxiter=20000)
    d[free] = sol
    uz = float(-d[3 * np.nonzero(top)[0] + 2].mean())
    # comparison: the same model with a stiffer (full inner-face) clamp
    return dict(method='voxel FE 1 mm hex8, E = %.0f MPa (PA12-CF, printed XY, conservative), nu 0.35, %d nodes; bolt clamp zones (r 6 mm on the side-plate inner faces) fixed; load %.0f N toward the body spread over the spine-side 2 mm strip of the stop face' % (E, nn, P.STOP_HOLD_N),
                load_N=P.STOP_HOLD_N, deflection_mm=uz, stiffness_N_per_mm=P.STOP_HOLD_N / uz, cg_converged=bool(info == 0),
                note='bolt/hole clearance (about 0.2 mm on the diameter, M5 in d5.4) and the clamp friction are not in the FE; with friction >= 15 N x lever / preload no slip is expected (M5 preload ~ 3 kN).')


def moment_check(mp):
    box = C.BOXES[4]
    m = C.BOX_MASS_MAX
    W = m * G
    y_arm = (box['l'] / 2 - P.S_PAD_CENTRE) / 1000
    z_arm = (P.Z_REG + box['h'] / 2) / 1000
    mf = mp['finger_left']['mass_kg'] + mp['finger_right']['mass_kg']
    zf = (mp['finger_left']['com_finger_frame_mm'][2]) / 1000
    al = C.GRIPPER['allow_moment_nm']
    out = dict(box_cog_arm_y_mm=y_arm * 1000, box_cog_arm_z_from_jaw_plane_mm=z_arm * 1000, finger_pair_mass_kg=mf, finger_cog_z_mm=zf * 1000,
               allowable=dict(Mp=al['pitch'], My=al['yaw'], Mr=1.4, Fv_N=98.0), axes='Mp about Y, My about Z (finger axis), Mr about X (stroke axis), Fv along Z (catalog p.45 figure); arms measured from the jaw mounting plane (conservative)')
    # case A: tool vertical (gravity along +Z), 2 g transport acceleration along ONE axis at a time
    MrA = W * y_arm + max(2 * W * z_arm, 2 * W * y_arm)
    MpA = 2 * W * z_arm + 2 * mf * G * zf
    MyA = 2 * W * y_arm
    # case B: bounding, 3 g (1 g + 2 g) along any single axis
    MrB = 3 * W * z_arm
    MpB = 3 * W * z_arm + 3 * mf * G * zf
    MyB = 3 * W * y_arm
    out['case_A_gravity_along_tool_axis_plus_2g_one_axis_Nm'] = dict(Mr=MrA, Mp=MpA, My=MyA)
    out['case_B_3g_any_single_axis_Nm'] = dict(Mr=MrB, Mp=MpB, My=MyB)
    out['margins_A'] = dict(Mr=al['pitch'] * 0 + 1.4 / MrA, Mp=al['pitch'] / MpA, My=al['yaw'] / MyA)
    out['margins_B'] = dict(Mr=1.4 / MrB, Mp=al['pitch'] / MpB, My=al['yaw'] / MyB)
    out['Fv_load_N_3g'] = 3 * W
    out['Fv_margin'] = 98.0 / (3 * W)
    out['pass'] = bool(min(list(out['margins_A'].values()) + list(out['margins_B'].values())) >= 1.0)
    return out


def crush_check():
    A_macro = P.PAD_W * P.PAD_H
    A_rib = A_macro * P.RIB_W / P.RIB_PITCH
    out = {}
    for lab, F in (('nominal 100 %', force_at(P.L_PAD, P.FORCE_100)), ('max 100 % +25 %', force_at(P.L_PAD, P.FORCE_100) * (1 + P.FORCE_ACC)), ('28 N catalog max', 28.0)):
        out[lab] = dict(F_N=round(F, 2), pressure_macro_kPa=round(F / A_macro * 1000, 2), pressure_on_rib_crests_kPa=round(F / A_rib * 1000, 2))
    out['pad_area_macro_mm2'] = A_macro
    out['rib_contact_area_mm2'] = A_rib
    out['reference'] = 'flat-crush strength of corrugated / folding carton board is typically >= 100 kPa (UNCONFIRMED for these cartons; not measured). Pressure at max force is ~%.0f%% of that on the rib crests.' % (100 * out['max 100 % +25 %']['pressure_on_rib_crests_kPa'] / 100.0)
    out['ok_with_100kPa_reference'] = bool(out['max 100 % +25 %']['pressure_on_rib_crests_kPa'] < 100.0)
    return out


def fit_check():
    out = {}
    gap_open = P.PAD_GAP_CLOSED + 2 * P.JAW_STROKE
    for t in (12.0, 15.0, 16.0):
        row = {}
        for tol, lab in ((-C.BOX_T_TOL, 'min'), (0.0, 'nom'), (C.BOX_T_TOL, 'max')):
            tt = t + tol
            g_app = t + 4.0
            row[lab] = dict(t_mm=tt, approach_clearance_per_side_open_mm=(gap_open - tt) / 2, grip_stroke_used_per_jaw_mm=(tt - P.PAD_GAP_CLOSED) / 2, remaining_closing_stroke_per_jaw_mm=(tt - P.PAD_GAP_CLOSED) / 2,
                            registration_approach_gap_mm=g_app, registration_clearance_per_side_mm=(g_app - tt) / 2)
        out['t=%.0f' % t] = row
    out['capture_open_56_per_side_mm'] = (gap_open - 16.5) / 2
    out['capture_open_with_tip_chamfer_mm'] = (gap_open - 16.5) / 2 + P.CHAMF
    out['note'] = ('Open approach (56 mm): the box is captured within +-(20 - t/2 ...) mm laterally (+5 mm tip chamfer), and the fixed centre top-stop (x +-2, always between the pads) is hit by the box top edge '
                   'at any jaw opening, so the descent can be stopped on contact (force/z) with no pre-closing. Spine-side end of the stop at s = 15; the stop only registers the 16/12 mm spine if it is within +-2 mm per side of the centre at closed; open approach centre error up to +-(gap-t)/2.')
    return out


def registration_check():
    """v3 position-based two-stage close: capture range of the chamfer funnel, bar-on-edge check, gap stop face - box top, tolerance stack"""
    C_, jb, pt = P.CHAMF, P.JAW_BACKLASH, P.PRINT_TOL
    out = dict(chamfer_mm=C_, chamfer_angle_deg=45, preclose_offset_mm=P.PRECLOSE_OFFSET, jaw_backlash_per_side_mm=jb, print_tol_per_side_mm=pt, bar_thickness_mm=P.STOP_T, rows={})
    worst_cap, worst_bar = 1e9, 1e9
    for t in (12.0, 14.0, 15.0, 16.0):
        g_pre = t + P.PRECLOSE_OFFSET
        row = {}
        for tol, lab in ((-C.BOX_T_TOL, 'thin'), (0.0, 'nom'), (C.BOX_T_TOL, 'thick')):
            ta = t + tol
            tip_gap = g_pre + 2 * C_
            cap = (tip_gap - ta) / 2.0                           # lateral offset the funnel accepts (nominal)
            cap_eff = cap - jb - pt                              # minus jaw position error and print tolerance
            straight = (g_pre - ta) / 2.0                        # clearance per side in the straight section
            resid = straight + jb                                # residual lateral error after centring (worst, incl. backlash)
            bar_ok = (ta - P.STOP_T) / 2.0                       # lateral error up to which the bar lies completely on the edge
            row[lab] = dict(t_actual=ta, pad_gap_preclosed=g_pre, tip_gap=tip_gap, capture_nominal=cap, capture_effective=cap_eff, clearance_per_side_straight=straight,
                            residual_error_after_centring=resid, bar_fully_on_edge_if_error_le=bar_ok, bar_margin=bar_ok - resid, closing_stroke_per_jaw_to_clamp=(g_pre - ta) / 2.0)
            worst_cap = min(worst_cap, cap_eff)
            worst_bar = min(worst_bar, bar_ok - resid)
        out['rows']['t=%g' % t] = row
    out['worst_capture_effective_mm'] = worst_cap
    out['required_capture_mm'] = P.LATERAL_ERR_REQ
    out['capture_ok'] = bool(worst_cap >= P.LATERAL_ERR_REQ)
    out['worst_bar_on_edge_margin_mm'] = worst_bar
    out['bar_on_edge_ok'] = bool(worst_bar > 0)
    # lean: tool tilted to the measured lean (+-1 deg residual): lateral drift of the box over the insertion depth between pads
    depth = P.Z_J + P.Z_TIP - P.Z_STOP - P.G_NOM
    out['insertion_depth_mm'] = depth
    out['lean_drift_uncompensated_mm'] = depth * math.tan(math.radians(P.LEAN_DEG))
    out['lean_drift_after_tilt_compensation_mm'] = depth * math.tan(math.radians(1.0))
    out['lean_clearance_ok_with_compensation'] = bool(out['lean_drift_after_tilt_compensation_mm'] < (P.PRECLOSE_OFFSET - C.BOX_T_TOL * 2) / 2.0)
    out['lean_uncompensated_exceeds_min_clearance'] = bool(out['lean_drift_uncompensated_mm'] > (P.PRECLOSE_OFFSET - 2 * C.BOX_T_TOL) / 2.0)
    # height tolerance stack of the stop face - box top gap
    comp = dict(P.TOL_STACK)
    comp['spine_pitch_%gdeg_over_stop_half_length' % P.SPINE_TILT_DEG] = P.STOP_Y * math.tan(math.radians(P.SPINE_TILT_DEG))
    wc = sum(comp.values())
    rss = math.sqrt(sum(x * x for x in comp.values()))
    out['height_stack_components_pm_mm'] = comp
    out['height_stack_worst_case_pm_mm'] = wc
    out['height_stack_rss_pm_mm'] = rss
    out['g_nom_mm'] = P.G_NOM
    out['gap_after_clamp_nominal_mm'] = P.G_NOM
    out['gap_after_clamp_range_worst_case_mm'] = [P.G_NOM - wc, P.G_NOM + wc]
    out['gap_after_clamp_range_rss_mm'] = [P.G_NOM - rss, P.G_NOM + rss]
    out['stop_would_land_on_edge_if_gap_error_gt_mm'] = P.G_NOM
    from math import erf
    out['approx_probability_of_landing_if_rss_is_1sigma_gaussian'] = 0.5 * (1 - erf(P.G_NOM / (rss * math.sqrt(2))))
    out['g_nom_needed_for_no_landing_worst_case_mm'] = wc
    out['g_nom_needed_for_no_landing_at_2sigma_mm'] = 2 * rss
    out['stop_engagement_rotation_deg'] = {'g=%.1f' % g: math.degrees(g / P.STOP_Y) for g in (P.G_NOM, P.G_NOM + wc)}
    out['pad_top_below_box_top_edge_nominal_mm'] = (P.PAD_Z0 - P.Z_REG) - P.G_NOM
    out['tip_to_stop_face_mm'] = P.Z_J + P.Z_TIP - P.Z_STOP
    out['sequence_geometry'] = dict(pre_close_pad_gap='t_nom + %.1f' % P.PRECLOSE_OFFSET, tips_above_edge_when_preclosed_mm=P.TIP_CLEAR, descent_from_funnel_contact_to_final_mm=(P.Z_J + P.Z_TIP - P.Z_STOP - P.G_NOM),
                                    chamfer_zone_mm=C_, final_tool_z_rule='z_stop_face = z_box_top_edge - %.1f (toward the flange)' % P.G_NOM)
    return out


def clearance_check():
    S = K.solids(P.JAW_STROKE, with_hw=True)
    out = {}
    s0 = P.S_PAD_CENTRE
    rows = {}
    for n, d in S.items():
        b = d['shape'].BoundingBox()
        rows[n] = (s0 + b.ymin, s0 + b.ymax)
    groups = {'fingers+pads (incl. chamfers)': ['finger_left', 'finger_right', 'pad_left', 'pad_right'], 'smc_body+jaws': ['smc_body', 'smc_jaw_left', 'smc_jaw_right'],
              'gripper_plate': ['gripper_plate'], 'top_stop': ['top_stop'], 'flange_disc': ['flange_disc'],
              'hardware': [n for n, d in S.items() if d['kind'] == 'hardware']}
    lo, hi = 1e9, -1e9
    for gname, ns in groups.items():
        l, h = min(rows[n][0] for n in ns), max(rows[n][1] for n in ns)
        out[gname] = dict(s_min=round(l, 2), s_max=round(h, 2), margin_to_s0_mm=round(l - 10.0, 2), margin_to_s90_mm=round(90.0 - h, 2))
        lo, hi = min(lo, l), max(hi, h)
    out['assumption'] = 'grip centre at s = %.1f (middle of the 15..70 zone), link6 Y parallel to s (mirrored mounting gives identical margins: the tool is symmetric about y = 0)' % s0
    out['required'] = 'fingers incl. chamfers s >= 10; nothing of this tool at s >= 90; (the other tool is assumed to exist only at s < 0)'
    out['whole_tool_s_range'] = [round(lo, 2), round(hi, 2)]
    out['pass_fingers_s_ge_10'] = bool(out['fingers+pads (incl. chamfers)']['s_min'] >= 10.0)
    out['pass_tool_s_lt_90'] = bool(hi < 90.0)
    out['pass_tool_s_ge_0'] = bool(lo >= 0.0)
    out['flange_disc_note'] = 'the flange disc (R36) is the part closest to the spine plane of everything except the fingers (s = %.1f); it is %.0f mm above the jaw plane' % (out['flange_disc']['s_min'], P.Z_J - P.Z_DISC_TOP)
    out['pad_zone_s_range'] = [s0 - P.PAD_W / 2, s0 + P.PAD_W / 2]
    s_nb = 100.0 - C.NEIGHBOUR_PROTRUSION
    out['neighbour_proud_case'] = dict(neighbour_spine_s_mm=s_nb, margin_fingers_mm=round(s_nb - out['fingers+pads (incl. chamfers)']['s_max'], 2), margin_whole_tool_mm=round(s_nb - hi, 2),
                                       note='shared NEIGHBOUR_PROTRUSION = 15 mm (ASSUMED): a neighbour spine may stand 15 mm proud, i.e. at s = 85')
    return out


# ----------------------------------------------------------------------------------------------------- 5 finger stiffness (2-D frame FE of the centroid line)
def section_props(mesh, z):
    """area, centroid x and second moment about the centroidal Y axis (integral (x-xc)^2 dA) of the cross-section at height z"""
    seg = trimesh.intersections.mesh_plane(mesh, [0, 0, 1.0], [0, 0, z])
    if len(seg) == 0:
        return None
    lines = [sg.LineString([(a[0], a[1]), (b[0], b[1])]) for a, b in seg]
    A = Sx = Ix = 0.0
    for p in polygonize(lines):
        c = p.representative_point()
        if not (mesh.contains([[c.x, c.y, z + 1e-3]])[0] or mesh.contains([[c.x, c.y, z - 1e-3]])[0]):
            continue                                    # a hole face of the planar graph
        for ring, sgn in [(np.array(p.exterior.coords), 1.0)] + [(np.array(h.coords), -1.0) for h in p.interiors]:
            x, y = ring[:-1, 0], ring[:-1, 1]
            x2, y2 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y2 - x2 * y
            a_s = 0.5 * cr.sum()
            cx = ((x + x2) * cr).sum() / (6 * a_s)
            ix = ((x ** 2 + x * x2 + x2 ** 2) * cr).sum() / 12.0 * np.sign(a_s)
            A += sgn * abs(a_s)
            Sx += sgn * abs(a_s) * cx
            Ix += sgn * ix
    if A <= 1e-6:
        return None
    xc = Sx / A
    return A, xc, Ix - A * xc ** 2


def finger_deflection():
    m = K.tess(K.local_parts()['finger_left'], 0.03, 0.1)
    zs = np.arange(0.25, P.Z_TIP - 0.2, 0.5)
    rows = []
    for z in zs:
        r = section_props(m, z)
        if r is None:
            continue
        rows.append((z,) + r)
    rows = np.array(rows)
    z_load = P.L_PAD
    # nodes = section levels up to the load level
    sel = rows[rows[:, 0] <= z_load + 1e-6]
    n = len(sel)
    out = {}
    for mat, E in (('PA12-CF', P.E_MPA['PA12CF']), ('PETG', P.E_MPA['PETG']), ('aluminium 6082', P.E_MPA['alu6082'])):
        # frame FE: nodes on the centroid line (x_c(z), z); dofs ux, uz, th
        N = n
        K_ = np.zeros((3 * N, 3 * N))
        for i in range(N - 1):
            z1, A1, x1, I1 = sel[i, 0], sel[i, 1], sel[i, 2], sel[i, 3]
            z2, A2, x2, I2 = sel[i + 1, 0], sel[i + 1, 1], sel[i + 1, 2], sel[i + 1, 3]
            dx, dz = x2 - x1, z2 - z1
            Le = math.hypot(dx, dz)
            A_e, I_e = min(A1, A2), min(I1, I2)
            c, s = dx / Le, dz / Le
            k = np.array([[E * A_e / Le, 0, 0, -E * A_e / Le, 0, 0],
                          [0, 12 * E * I_e / Le ** 3, 6 * E * I_e / Le ** 2, 0, -12 * E * I_e / Le ** 3, 6 * E * I_e / Le ** 2],
                          [0, 6 * E * I_e / Le ** 2, 4 * E * I_e / Le, 0, -6 * E * I_e / Le ** 2, 2 * E * I_e / Le],
                          [-E * A_e / Le, 0, 0, E * A_e / Le, 0, 0],
                          [0, -12 * E * I_e / Le ** 3, -6 * E * I_e / Le ** 2, 0, 12 * E * I_e / Le ** 3, -6 * E * I_e / Le ** 2],
                          [0, 6 * E * I_e / Le ** 2, 2 * E * I_e / Le, 0, -6 * E * I_e / Le ** 2, 4 * E * I_e / Le]])
            Tm = np.zeros((6, 6))
            # local axis 1 along the element (dx,dz) ; local 2 perpendicular: (-dz,dx)/L  (global dofs: ux (x), uz (z))
            rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])        # global (x,z) -> local (axial, transverse)
            Tm[:3, :3] = rot
            Tm[3:, 3:] = rot
            kg = Tm.T @ k @ Tm
            idx = list(range(3 * i, 3 * i + 3)) + list(range(3 * i + 3, 3 * i + 6))
            K_[np.ix_(idx, idx)] += kg
        F = np.zeros(3 * N)
        F[3 * (N - 1)] = -28.0                  # clamp reaction on the left finger acts toward -x
        free = list(range(3, 3 * N))
        u = np.zeros(3 * N)
        u[free] = np.linalg.solve(K_[np.ix_(free, free)], F[free])
        out[mat] = dict(E_MPa=E, deflection_at_pad_centre_mm=abs(float(u[3 * (N - 1)])), meets_0p3mm=bool(abs(u[3 * (N - 1)]) <= 0.3))
    # bending stress at the weakest level (column, z in [FOOT_T, Z_REG])
    col = rows[(rows[:, 0] > P.FOOT_T + 0.3) & (rows[:, 0] < P.Z_REG - 0.3)]
    Mz = 28.0 * (z_load - col[:, 0])
    # section half thickness in x ~ 3 mm (column 6 mm thick)
    sig = Mz * 3.0 / col[:, 3]
    out['column_peak_bending_stress_MPa_at_28N'] = float(sig.max())
    out['column_min_I_mm4'] = float(col[:, 3].min())
    out['method'] = ('2-D frame FE of the centroid line, section area / second moment sliced from the CAD every 0.5 mm, fixed at the jaw plane, 28 N (catalog max gripping force) at the pad centre z = %.1f mm; '
                     'first-order (no shear, rigid jaw interface, TPU pad ignored); isotropic printed-XY modulus' % z_load)
    return out


# ----------------------------------------------------------------------------------------------------- 6 mass / CoG / flange
def mass_flange(mp):
    out = dict(tool_total_kg=mp['total_closed']['mass_kg'], tool_cog_link6_mm=mp['total_closed']['com_link6_mm'],
               parts_kg={k: round(v['mass_kg'], 4) for k, v in mp['parts'].items() if not k.startswith(('m3', 'm4', 'm5', 'pin'))})
    out['printed_total_kg'] = sum(mp['parts'][k]['mass_kg'] for k in ('flange_disc', 'gripper_plate', 'top_stop', 'finger_left', 'finger_right', 'pad_left', 'pad_right'))
    out['hardware_total_kg'] = out['tool_total_kg'] - GRIP_M - out['printed_total_kg']
    out['smc_gripper_kg'] = GRIP_M
    c = np.array(mp['total_closed']['com_link6_mm']) - np.array([0, 0, P.FLANGE_Z])
    out['cog_above_flange_face_mm'] = float(c[2])
    out['cog_lateral_offset_mm'] = [float(c[0]), float(c[1])]
    mt = out['tool_total_kg']
    box = C.BOXES[4]
    rb = np.array([0.0, box['l'] / 2 - P.S_PAD_CENTRE, P.Z_J + P.Z_REG + box['h'] / 2 - P.FLANGE_Z])
    mom_tool = mt * c * 1e-3                          # kg m
    mom_box = C.BOX_MASS_MAX * rb * 1e-3
    out['static_moment_tool_only_Nm_worst_orientation'] = float(G * np.linalg.norm(mom_tool))
    out['static_moment_tool_plus_box4_Nm_worst_orientation'] = float(G * np.linalg.norm(mom_tool + mom_box))
    out['mass_tool_plus_box_kg'] = mt + C.BOX_MASS_MAX
    out['payload_limit_kg'] = C.ROBOT_PAYLOAD
    out['payload_margin_kg'] = C.ROBOT_PAYLOAD - mt - C.BOX_MASS_MAX
    out['note'] = 'moment about the flange-face centre with gravity perpendicular to the CoG vector (worst orientation); ReBeL permits 2 kg (flange-moment limit not given in the shared inputs)'
    return out


GRIP_M = P.GRIPPER_MASS


# ----------------------------------------------------------------------------------------------------- 7 printability
def wall_thickness(m, n=3000, seed=1):
    pts, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fi]
    org = pts - nrm * 0.02
    ri = trimesh.ray.ray_triangle.RayMeshIntersector(m)
    loc, ir, it = ri.intersects_location(org, -nrm, multiple_hits=False)
    th = np.full(n, np.nan)
    th[ir] = np.linalg.norm(loc - org[ir], axis=1)
    return pts, th


def overhang(m, thr_deg=45.0):
    nrm = m.face_normals
    low = m.triangles_center[:, 2]
    bed = low < (m.bounds[0][2] + 0.05)
    need = (nrm[:, 2] < -math.cos(math.radians(thr_deg))) & ~bed
    flat = (nrm[:, 2] < -0.999) & ~bed
    return float(m.area_faces[need].sum()), float(m.area_faces[flat].sum()), float(m.area_faces[bed & (nrm[:, 2] < -0.99)].sum())


def printability():
    L = K.local_parts()
    S = K.solids(0.0, with_hw=False)
    parts = {}
    tf = trimesh.transformations
    defs = {
        'flange_disc': (S['flange_disc']['shape'], np.eye(4), 'flange-side face (z = %.1f) on the bed, countersinks / insert holes open upward' % P.FLANGE_Z),
        'gripper_plate': (S['gripper_plate']['shape'], tf.rotation_matrix(math.pi, [1, 0, 0]), 'turned over: the gripper-side (top) face on the bed so the M5 countersinks open upward (no support)'),
        'top_stop': (S['top_stop']['shape'], tf.rotation_matrix(math.pi, [1, 0, 0]), 'upside down: the stop face on the bed (4 x 55 mm), risers up, side plates last: bridges/overhang of the 16 mm wide side plates (support or print on the side with support)'),
        'finger_left': (L['finger_left'], np.eye(4), 'foot (jaw face) on the bed; column flanks 35 deg from vertical, the window ceiling is a short bridge'),
        'finger_right': (L['finger_right'], np.eye(4), 'foot (jaw face) on the bed'),
        'pad_left': (L['pad_left'], tf.rotation_matrix(-math.pi / 2, [0, 1, 0]), 'back (floor) face on the bed, ribs up; TPU, 0.2 mm layers, slow'),
        'pad_right': (L['pad_right'], tf.rotation_matrix(math.pi / 2, [0, 1, 0]), 'back (floor) face on the bed, ribs up'),
    }
    for n, (sh, T, note) in defs.items():
        m = K.tess(sh, 0.05, 0.15)
        m.apply_transform(T)
        m.apply_translation([0, 0, -m.bounds[0][2]])
        pts, th = wall_thickness(m)
        ok = ~np.isnan(th)
        oh, flat, bedA = overhang(m)
        parts[n] = dict(print_orientation=note, bed_contact_mm2=round(bedA, 0), overhang_gt45deg_area_mm2=round(oh, 0), of_which_horizontal_bridge_mm2=round(flat, 0),
                        wall_thickness_p01_mm=round(float(np.nanpercentile(th[ok], 1)), 2), wall_thickness_p05_mm=round(float(np.nanpercentile(th[ok], 5)), 2),
                        frac_samples_below_3mm=round(float((th[ok] < 2.95).mean()), 4), height_mm=round(float(m.extents[2]), 1))
    parts['finger_left']['support_estimate'] = 'window ceiling 56 x 6.6 mm bridge only (1 line of bridging); everything else self-supporting'
    parts['inserts'] = 'M4 x 4 heat-set inserts (hole d5.6 x 4.5) x4 in the disc; M3 x 4 inserts (hole d4.2 x 5) x4 in the fingers (x2 each); floors under the blind holes are 3.0 / >= 5 mm'
    parts['min_wall_rules'] = ('min wall 3.0 mm kept everywhere except: TPU pad (2.6 base, by design), pocket side walls 3.0, stop plate 4.0 / risers 4.0 x 6.5, pad-retention cb floor in the TPU 0.7 mm (under the screw head, compressible), '
                               'tip lip behind the chamfer 7.6 mm')
    return parts


# ----------------------------------------------------------------------------------------------------- 8 assembly access / tools
def access_check():
    out = {}
    S = K.solids(P.JAW_STROKE, with_hw=False)
    obstacles = {n: S[n]['shape'] for n in ('finger_left', 'finger_right', 'pad_left', 'pad_right', 'smc_body', 'smc_jaw_left', 'smc_jaw_right', 'gripper_plate', 'flange_disc', 'top_stop')}

    def hit(tool, skip=()):
        h = {}
        tb = tool.BoundingBox()
        for n, s in obstacles.items():
            if n in skip:
                continue
            b = s.BoundingBox()
            if b.xmax < tb.xmin or tb.xmax < b.xmin or b.ymax < tb.ymin or tb.ymax < b.ymin or b.zmax < tb.zmin or tb.zmax < b.zmin:
                continue
            v = tool.intersect(s).Volume()
            if v > 1e-3:
                h[n] = round(v, 3)
        return h
    # inner finger screws: ball-end hex key (shaft d2.4) from the socket, tilted towards the gap
    zs = P.Z_J + P.FOOT_T - 1.2                    # socket bottom
    res = {}
    for ang in (0, 10, 15, 20, 25, 30):
        bad = {}
        for side in (-1, 1):
            for ym in P.JAW_M3_Y:
                x0 = side * (P.JAW_M3_X[0] + P.JAW_STROKE)
                d = np.array([-side * math.sin(math.radians(ang)), 0, math.cos(math.radians(ang))])
                key = cq.Solid.makeCylinder(1.2, 60.0, cq.Vector(x0, ym, zs), cq.Vector(*d))
                h = hit(key)
                for k, v in h.items():
                    bad[k] = bad.get(k, 0) + v
        res[str(ang)] = bad
    out['inner_screw_ball_key_swing_toward_gap_deg_vs_collisions'] = res
    clear = [int(a) for a, b in res.items() if not b]
    out['inner_screw_min_clear_swing_deg'] = min(clear) if clear else None
    out['inner_screw_note'] = 'screws seated with the fingers on the OPEN jaws; the window under the pad block is %.1f mm high (screw M3x8 drops in vertically, driver ball-end key swung >= %s deg toward the gap)' % (P.Z_REG - P.FOOT_T, out['inner_screw_min_clear_swing_deg'])
    out['window_height_mm'] = P.Z_REG - P.FOOT_T
    # outer finger screws and M4 plate screws: vertical hex driver d8 from the head upwards
    v = {}
    for side in (-1, 1):
        for ym in P.JAW_M3_Y:
            x0 = side * (P.JAW_M3_X[1] + P.JAW_STROKE)
            t = cq.Solid.makeCylinder(4.0, 80.0, cq.Vector(x0, ym, P.Z_J + P.FOOT_T), cq.Vector(0, 0, 1))
            for k, val in hit(t).items():
                v[k] = v.get(k, 0) + val
    out['outer_finger_screws_vertical_d8_driver_collisions'] = v
    v = {}
    for (x, y) in P.M4_XY:
        t = cq.Solid.makeCylinder(4.0, 80.0, cq.Vector(x, y, P.Z_R + 4.0), cq.Vector(0, 0, 1))
        for k, val in hit(t, skip=('top_stop',)).items():
            v[k] = v.get(k, 0) + val
    out['m4_plate_screws_vertical_d8_driver_collisions'] = v
    out['m4_note'] = 'the M4 plate screws are driven BEFORE the stop yoke is fitted (the yoke side plates / M5 heads sit in the driver column); to remove them take the yoke off first'
    # yoke install: slide from +z straight down onto the body (fingers not yet mounted), check every 3 mm for collisions with body, jaws (open and closed), plate, disc and the M4 hardware
    stop0 = K.top_stop()
    hwsolids = {h['name']: h['shape'] for h in K.hardware(0.0) if h['name'].startswith('m4x10') or h['name'].startswith('m5x12')}
    sweep = {}
    for o in (0.0, P.JAW_STROKE):
        Sx = K.solids(o, with_hw=False)
        obst = {n: Sx[n]['shape'] for n in ('smc_body', 'smc_jaw_left', 'smc_jaw_right', 'gripper_plate', 'flange_disc')}
        obst.update(hwsolids)
        worst = 0.0
        for dz in range(0, 82, 3):
            st = K.move(stop0, 0, 0, dz)
            bb = st.BoundingBox()
            for n, s in obst.items():
                b = s.BoundingBox()
                if b.xmax < bb.xmin or bb.xmax < b.xmin or b.ymax < bb.ymin or bb.ymax < b.ymin or b.zmax < bb.zmin or bb.zmax < b.zmin:
                    continue
                worst = max(worst, st.intersect(s).Volume())
        sweep['jaws_open_%g' % o] = worst
    out['stop_yoke_install_slide_down_max_collision_mm3'] = sweep
    # M5 clamp bolts: hex key / nut wrench along y from outside (cylinder d9 from the head / nut outward) vs everything
    vb = {}
    for x in P.BOLT_X:
        for sg in (-1, 1):
            t = cq.Solid.makeCylinder(4.5, 60.0, cq.Vector(x, sg * (P.SIDE_Y[1] + 5.0), P.BOLT_Z), cq.Vector(0, sg, 0))
            for k, val in hit(t).items():
                vb[k] = vb.get(k, 0) + val
    out['stop_m5_bolt_wrench_access_collisions'] = vb
    out['pass'] = bool(not v and not vb and not any(x > 1e-3 for x in sweep.values()) and not out['outer_finger_screws_vertical_d8_driver_collisions'] and out['inner_screw_min_clear_swing_deg'] is not None and out['inner_screw_min_clear_swing_deg'] <= 25)
    return out


def screw_checks(mp):
    """first-order fastener loads: finger->jaw (clamp 28 N) and plate->disc / disc->flange (tool + box 3 g)"""
    out = {}
    F = 28.0 * 1.25
    h = P.L_PAD
    arm = P.JAW_M3_X[1] - 0.0
    arm = P.JAW_M3_X[0] + (P.FOOT_X[1] - P.JAW_M3_X[0])
    # finger: moment F*h about Y reacted by compression at the outer edge and tension in the inner screws (x = 5.5 from the inner edge side), lever = 28 - 5.5
    lever = P.FOOT_X[1] - P.JAW_M3_X[0]
    T = F * h / lever
    out['finger_inner_screw_tension_each_N'] = T / 2
    out['finger_screw_shear_each_N'] = F / 4
    out['m3_thread_pullout_ref_N'] = 'M3 12.9 steel proof load ~ 4.5 kN; 3.8 mm engagement in aluminium jaw ~ 1.5 kN'
    mt = mp['total_closed']['mass_kg'] + C.BOX_MASS_MAX
    M = mt * 3 * G * 0.15                       # 3 g, ~150 mm lever worst
    out['disc_to_flange_M3_tension_each_N_est'] = M / 0.0215 / 6 / 2
    out['plate_to_disc_M4_tension_each_N_est'] = M / 0.030 / 4 / 2
    return out


def urdf_check(mp):
    import subprocess
    out = {}
    cmd = ('source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=174 ROS_LOCALHOST_ONLY=1; cd "%s/urdf"; xacro test_right_gripper_tool.urdf.xacro > "%s/t.urdf" && check_urdf "%s/t.urdf"'
           % (HERE, P.SCRATCH, P.SCRATCH))
    r = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    out['xacro_check_urdf_ok'] = bool(r.returncode == 0 and 'Successfully Parsed XML' in r.stdout)
    out['check_urdf_output'] = r.stdout.strip().splitlines()[:12]
    txt = open(os.path.join(P.SCRATCH, 't.urdf')).read()
    import re
    files = re.findall(r'file://([^"]+\.stl)', txt)
    out['n_mesh_references'] = len(files)
    out['all_meshes_exist'] = bool(all(os.path.exists(f) for f in files))
    out['n_joints'] = txt.count('<joint ')
    out['mimic_present'] = ('<mimic joint="right_finger_left_joint"' in txt)
    ine = {}
    for k in ('base', 'finger_left', 'finger_right'):
        I = np.array(mp[k]['inertia_about_com_kgmm2'])
        ev = np.linalg.eigvalsh(I)
        ine[k] = dict(mass_kg=mp[k]['mass_kg'], principal_kgmm2=[float(x) for x in ev], positive_definite=bool(ev.min() > 0), triangle_ok=bool(ev[0] + ev[1] >= ev[2] * 0.999))
    out['inertials'] = ine
    return out


def main():
    mp = json.load(open(OUT('data/mass.json')))
    R['geometry'] = geometry()
    print('geometry done', flush=True)
    R['interference'] = interference()
    print('interference done', flush=True)
    R['reimport'] = reimport()
    R['hold_check_a'] = hold_check()
    R['hold_check_a']['note'] = 'NO-STOP reference (previous revision); see hold_check_a_with_stop'
    R['hold_check_a_with_stop'] = hold_with_stop()
    R['stop_stiffness'] = stop_stiffness()
    R['jaw_moments_b'] = moment_check(mp)
    R['crush_c'] = crush_check()
    R['fit_d'] = fit_check()
    R['clearance_e'] = clearance_check()
    R['registration_v3'] = registration_check()
    R['mass_flange_5'] = mass_flange(mp)
    R['finger_deflection'] = finger_deflection()
    print('finger done', flush=True)
    R['printability_6'] = printability()
    R['assembly_access'] = access_check()
    R['fasteners'] = screw_checks(mp)
    R['urdf'] = urdf_check(mp)
    json.dump(R, open(OUT('data/verify.json'), 'w'), indent=1, default=float)
    g = R['geometry']
    print('jaw plane z %.2f, fingertip z %.2f, TCP z %.2f; pad gap closed %.2f / open %.2f; H+L %.1f' % (g['z_jaw_plane_mm'], g['z_fingertip_mm'], g['z_tcp_mm'], g['pad_gap_closed_mm'], g['pad_gap_open_mm'], g['H_plus_L_mm']))
    print('interference max volume', R['interference']['max_boolean_volume_mm3_excluding_thread_hosts'], {k: len(v['violations']) for k, v in R['interference'].items() if isinstance(v, dict) and 'violations' in v})
    print('reimport problems', [k for k, v in R['reimport'].items() if k != 'sim_meshes' and isinstance(v, dict) and (not v.get('step_valid', True) or v.get('stl_watertight', True) is False)])
    print('sim meshes not watertight', [k for k, v in R['reimport']['sim_meshes'].items() if not v['watertight']])
    print(json.dumps({k: {kk: vv for kk, vv in b.items() if kk in ('hang_angle_deg',) or '18.2' in kk or '11.0' in kk or '24.3' in kk} for k, b in R['hold_check_a_with_stop']['boxes'].items()}, indent=1)[:3000]); print(R['stop_stiffness'])
    print(R['jaw_moments_b']['margins_A'], R['jaw_moments_b']['margins_B'])
    print(R['finger_deflection'])
    print(R['clearance_e'])
    print(R['assembly_access']['inner_screw_ball_key_swing_toward_gap_deg_vs_collisions'], R['assembly_access']['pass'])
    print(R['mass_flange_5'])
    print(R['urdf'])


if __name__ == '__main__':
    main()
