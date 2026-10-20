"""All parts of the left suction tool v5 in the link6 frame (mm).  build_parts() -> list of dicts
(name, shape, kind = printed|purchased, + mass / material / print orientation notes).
One printed body (tool_body) + purchased parts: 3 pads, 3 elbows, 6 nuts, 3 washers, 3 tubes, collector, trunk stub, camera."""
import math, os, pickle
import numpy as np
import cadquery as cq
import params as P
import lib as L
C = P.C

Lz = P.LIP_Z
T_CAM = P.T_CAM
YC, ZC = float(T_CAM[1, 3]), float(T_CAM[2, 3])
CA, SA = T_CAM[1, 1], T_CAM[2, 1]            # cos / sin of the camera tilt


def cam_pt(x, u, t):
    return (T_CAM @ np.array([x, u, t, 1.0]))[:3]


def cam_box(x0, x1, u0, u1, t0, t1):
    return L.obox(cam_pt((x0 + x1) / 2, (u0 + u1) / 2, (t0 + t1) / 2), T_CAM[:3, :3], [x1 - x0, u1 - u0, t1 - t0])


CX = P.COLL_ZC


# ============================================================================================ the printed body
def pa_cam(x, u, t):
    return cam_pt(x, u, t)


def tool_body():
    hx, y0, y1 = P.STEM_HX, P.STEM_Y0, P.STEM_Y1
    # ---- hub + C-section stem (cavity open toward -y)
    hub = L.zcyl(0, 0, P.HUB_Z0, P.HUB_Z1, P.HUB_D / 2)
    stem = L.aabox((-hx, hx), (y0, y1), (P.HUB_Z1 - 0.01, P.SKIN_Z0 + 0.01))
    stem = stem.cut(L.aabox((-P.CAV_HX, P.CAV_HX), (y0 - 2.0, P.CAV_Y1), (P.HUB_Z1, P.SKIN_Z0)))
    # ---- collector bay: shelf (floor under the collector) + two cheeks, hanging off the open -y side of the stem
    # two ledges (3 x 6 mm) along the inner foot of the cheeks carry the collector; the trunk boss (phi11) passes between them
    shelf = L.union([L.aabox(sorted((sx * (P.CAV_HX - P.LEDGE_W), sx * (P.CAV_HX + 0.01))), (P.SHELF_Y0, y0 + 0.01), (P.SHELF_Z0, P.SHELF_Z1)) for sx in (-1, 1)])
    cheeks = [L.aabox(sorted((sx * P.CAV_HX, sx * hx)), (P.CHEEK_Y0, y0 + 0.01), (P.SHELF_Z1 - 0.01, P.CHEEK_Z1)) for sx in (-1, 1)]     # run from the stem end to the bay end: they carry the shelf (printed as bridge supports)
    # lower -y wall of the stem (hub level) + 45 deg hub gussets: the hub plate overhangs the stem by 9..17 mm; printed bar-face-down the gussets make that overhang self-supporting
    low = L.aabox((-hx, hx), (y0, y0 + P.WALL), (P.HUB_Z1 - 0.01, P.HUB_Z1 + P.HUB_GUSSET_H))
    gus = []
    for k in range(12):
        phi = math.radians(30.0 * k)
        rs = min(hx / max(abs(math.cos(phi)), 1e-9), (y1 if math.sin(phi) >= 0 else -y0) / max(abs(math.sin(phi)), 1e-9))
        rs = min(rs, 28.0 - 3.0)
        h = P.HUB_D / 2 - rs + 0.3
        g = L.extrude_poly_x([(rs - 0.5, P.HUB_Z1 - 0.01), (P.HUB_D / 2 - 0.3, P.HUB_Z1 - 0.01), (rs - 0.5, P.HUB_Z1 + h)], -1.5, 1.5)      # triangle in the (radial = y, z) plane, 3 mm thick
        a_ = phi - math.pi / 2                                   # radial +y -> direction phi
        T = np.eye(4); T[:2, :2] = [[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]]
        gus.append(L.move(g, T))
    # ---- pad bar: skin + rear bosses + edge ribs
    skin = L.zstadium(0, 0, P.SKIN_Z0, P.SKIN_Z1, P.BAR_LEN, 2 * P.BAR_HY)
    bosses = [L.zcyl(x, 0, P.BOSS_Z0, P.SKIN_Z0 + 0.01, P.BOSS_D / 2) for x in P.PAD_X]
    ribs = [L.aabox(sorted((sx * P.RIB_X[0], sx * P.RIB_X[1])), sorted((sy * P.RIB_Y[0], sy * P.RIB_Y[1])), (P.RIB_Z0, P.SKIN_Z0 + 0.01))
            for sx in (-1, 1) for sy in (-1, 1)]
    # ---- USB-C cable tie lug on the +x stem wall, below the plug (the USB-C receptacle is on the camera's +x face; v3 had its relief on -x)
    cl = P.CABLE_LUG
    lug = L.aabox(cl['x'], cl['y'], cl['z'])
    s = L.union([hub, stem, shelf, low] + gus + cheeks + [skin] + bosses + ribs + [lug])
    cuts = [L.zcyl(0, 0, P.HUB_Z0 - 1, P.HUB_Z1 + 1, P.HUB_BORE / 2)]
    for x in P.PAD_X:                                                       # pad holes + front hex pockets (nut keying)
        cuts.append(L.zcyl(x, 0, P.BOSS_Z0 - 1, P.SKIN_Z1 + 1, P.PAD_HOLE_D / 2))
        cuts.append(L.zhex(x, 0, P.SKIN_Z1 - P.HEX_POCKET_DEPTH, P.SKIN_Z1 + 1, P.HEX_POCKET_AF))
    for (x, y) in P.FLANGE_HOLES:                                           # 6 x M3 CSK on PCD 43, driven from +z (camera off); phi6 access column through whatever is above
        cuts.append(L.zcyl(x, y, P.HUB_Z0 - 1, P.HUB_Z1 + 1, C.M3_CLR / 2))
        cuts.append(cq.Solid.makeCone(C.M3_CLR / 2, 3.2, 1.5, L.V([x, y, P.HUB_Z1 - 1.5]), L.V([0, 0, 1])))
        cuts.append(L.zcyl(x, y, P.HUB_Z1 - 0.01, 220.0, 3.0))
    # collector bay: trunk hole, cheek slots for the two phi9 port bosses (open to +z so the collector drops in)
    for sx in (-1, 1):
        cuts.append(L.aabox(sorted((sx * (P.CAV_HX - 0.1), sx * (hx + 0.1))), (P.COLL_YC - 4.9, P.COLL_YC + 4.9), (P.COLL_ZC - 4.9, P.CHEEK_Z1 + 1)))
    # camera screws: 2 x M3 along +y through the stem +y wall (screw head on the cavity face, 3 mm wall + 0.5 mm washer, <= 4.8 mm thread in the camera)
    for sx in (-1, 1):
        x = sx * P.CAM_SCREW_X
        cuts.append(L.ocyl(pa_cam(x, -34.0, P.CAM_SCREW_T), pa_cam(x, -P.CAM_HALF + 0.01, P.CAM_SCREW_T), 1.7))
    # spanner window in the +y wall at the middle pad's rear nut + washer (z 122.4..127), so a thin open-end spanner can enter from +y
    cuts.append(L.aabox((-12.0, 12.0), (P.CAV_Y1 - 1.0, P.STEM_Y1 + 1.0), (P.NUT_REAR_Z0 - 0.6, P.BOSS_Z0)))
    cuts.append(L.aabox(P.CABLE_LUG['slot_x'], (cl['y'][0] - 1.0, cl['y'][1] + 0.5), P.CABLE_LUG['slot_z']))        # cable tie slot through the lug (along y)
    return L.cut_all(s, cuts)


# ============================================================================================ purchased parts
def pad_body(x):
    """ZP2-T4030WNK10-B5 proxy from the catalog drawing (lip at z = LIP_Z, oval long axis along X); M5 female inlet at the rear hex (no fitting)"""
    z = lambda d: Lz - d
    prts = [L.zstadium(x, 0, z(1.5), z(0), 30.8, 4.8),                 # lip
            L.zstadium(x, 0, z(8.0), z(1.5), 24.0, 3.6),               # pad skirt (frustum approximated)
            L.zstadium(x, 0, z(14.0), z(8.0), 16.0, 4.0),              # pad adapter
            L.zcyl(x, 0, z(25.5), z(14.0), 4.0),                       # guide rod
            L.zcyl(x, 0, z(48.5), z(25.5), P.THREAD_R),                # M10x1 thread zone
            L.zhex(x, 0, z(61.5), z(48.5), 8.0)]                       # inlet hex 8 AF (rear end, z = 97.5)
    return L.union(prts)


COLL_C_X = 28.0           # |x| of the corner C where the outer tubes straighten out toward the collector port


def elbow_dir(i):
    """horizontal outlet direction of the swivel elbow on pad i: straight toward the corner C of its tube (M: toward -y)"""
    x = P.PAD_X[i]
    if i == 1:
        return np.array([0.0, -1.0, 0.0])
    c = np.array([np.sign(x) * COLL_C_X, P.COLL_YC]) - np.array([x, 0.0])
    c = c / np.linalg.norm(c)
    return np.array([c[0], c[1], 0.0])


def elbow(i):
    """KQ2L04-M5 swivel elbow proxy: phi9 column from the inlet face down 14 mm + horizontal barrel (7 mm beyond the pad axis) at the tube axis height"""
    x = P.PAD_X[i]; d = elbow_dir(i); zc = P.ELBOW_AXIS_Z
    col = L.zcyl(x, 0, P.INLET_Z - P.ELBOW_H, P.INLET_Z, P.ELBOW_D / 2)
    bar = L.ocyl([x, 0, zc], [x + d[0] * P.ELBOW_BARREL, d[1] * P.ELBOW_BARREL, zc], P.ELBOW_D / 2)
    return L.union([col, bar])


def nut(x, z0, z1, rot_deg=0.0):
    n = L.zhex(x, 0, z0, z1, P.NUT_AF).cut(L.zcyl(x, 0, z0 - 1, z1 + 1, 5.0))
    if rot_deg:
        T = np.eye(4); a = math.radians(rot_deg); T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        T[:2, 3] = np.array([x, 0.0]) - T[:2, :2] @ np.array([x, 0.0])
        n = L.move(n, T)
    return n


def washer(x, z0, z1):
    return L.zcyl(x, 0, z0, z1, P.WASHER['od'] / 2).cut(L.zcyl(x, 0, z0 - 1, z1 + 1, P.WASHER['id'] / 2))


def fillet_path(pts, R, n=8):
    """polyline with sharp corners -> sampled path with tangent-arc fillets of radius R at every interior corner; returns (points, info)"""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]; info = dict(bends=[], seg_len=[])
    ts = []
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]; d1 /= np.linalg.norm(d1); d2 = pts[i + 1] - pts[i]; d2 /= np.linalg.norm(d2)
        th = math.acos(np.clip(d1 @ d2, -1, 1)); ts.append(R * math.tan(th / 2))
    ts = [0.0] + ts + [0.0]
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]; l1 = np.linalg.norm(d1); d1 /= l1
        d2 = pts[i + 1] - pts[i]; l2 = np.linalg.norm(d2); d2 /= l2
        th = math.acos(np.clip(d1 @ d2, -1, 1)); t = ts[i]
        assert l1 >= ts[i - 1] + t - 1e-6, 'segment %d too short for the bend radius' % i
        info['seg_len'].append(l1 - ts[i - 1] - t)
        ta = pts[i] - d1 * t
        u = d2 - (d2 @ d1) * d1; u /= np.linalg.norm(u)
        O = ta + R * u
        for k in range(n + 1):
            ph = th * k / n
            out.append(O + R * (-u * math.cos(ph) + d1 * math.sin(ph)))
        info['bends'].append(dict(radius=R, angle_deg=math.degrees(th)))
    assert np.linalg.norm(pts[-1] - pts[-2]) >= ts[-2] - 1e-6
    info['seg_len'].append(np.linalg.norm(pts[-1] - pts[-2]) - ts[-2])
    out.append(pts[-1])
    info['length'] = float(sum(np.linalg.norm(b - a) for a, b in zip(out[:-1], out[1:])))
    info['chord'] = float(np.linalg.norm(pts[-1] - pts[0]))
    return out, info


def collector_ports():
    b = P.COLLECTOR['body']; b4 = P.COLLECTOR['boss4']
    return dict(L=np.array([-(b[0] / 2 + b4[1]), P.COLL_YC, P.COLL_ZC]), R=np.array([(b[0] / 2 + b4[1]), P.COLL_YC, P.COLL_ZC]),
                M=np.array([0.0, P.COLL_YC + b[1] / 2 + b4[1], P.COLL_ZC]))


TUBE_INFO = {}


def tube_waypoints(i):
    x = P.PAD_X[i]; d = elbow_dir(i); zc = P.ELBOW_AXIS_Z
    A = np.array([x + d[0] * P.ELBOW_BARREL, d[1] * P.ELBOW_BARREL, zc])
    ports = collector_ports()
    if i == 1:
        return [A, ports['M']]
    D = ports['L'] if i == 0 else ports['R']
    Cc = np.array([np.sign(x) * COLL_C_X, P.COLL_YC, zc])
    return [A, Cc, D]


def tube(i):
    w = tube_waypoints(i)
    if len(w) == 2:
        TUBE_INFO[i] = dict(bends=[], seg_len=[float(np.linalg.norm(w[1] - w[0]))], length=float(np.linalg.norm(w[1] - w[0])), chord=float(np.linalg.norm(w[1] - w[0])))
        return L.ocyl(w[0], w[1], P.TUBE_OD / 2 - 0.05)
    path, info = fillet_path(w, P.TUBE_BEND_R)
    TUBE_INFO[i] = info
    return L.tube_polyline([p.tolist() for p in path], P.TUBE_OD / 2 - 0.05)


def collector():
    """3 x phi4 + 1 x phi6 push-in collector block (proxy): phi4 on -x, +x, +y (middle pad) faces, phi6 trunk on the bottom (-z) face"""
    cb = P.COLLECTOR['body']; b4, b6 = P.COLLECTOR['boss4'], P.COLLECTOR['boss6']; cy, zt = P.COLL_YC, P.COLL_ZC
    body = L.aabox((-cb[0] / 2, cb[0] / 2), (cy - cb[1] / 2, cy + cb[1] / 2), (zt - cb[2] / 2, zt + cb[2] / 2))
    bos = [L.ocyl([-cb[0] / 2 + 0.01, cy, zt], [-cb[0] / 2 - b4[1], cy, zt], b4[0] / 2),
           L.ocyl([cb[0] / 2 - 0.01, cy, zt], [cb[0] / 2 + b4[1], cy, zt], b4[0] / 2),
           L.ocyl([0, cy + cb[1] / 2 - 0.01, zt], [0, cy + cb[1] / 2 + b4[1], zt], b4[0] / 2),
           L.ocyl([0, cy, zt - cb[2] / 2 + 0.01], [0, cy, zt - cb[2] / 2 - b6[1]], b6[0] / 2)]
    return L.union([body] + bos)


def trunk_tube():
    """phi6 trunk leaves the collector straight down (-z) at y = COLL_YC (outside the hub rim and the link6 housing), continues down the arm"""
    cb = P.COLLECTOR['body']; b6 = P.COLLECTOR['boss6']
    return L.ocyl([0, P.COLL_YC, P.COLL_ZC - cb[2] / 2 - b6[1]], [0, P.COLL_YC, P.TRUNK_END_Z], P.TRUNK_OD / 2 - 0.05)


def _decimate(pts, n):
    from scipy.spatial import ConvexHull
    hv = pts[ConvexHull(pts).vertices]
    sel = [int(np.argmax(hv[:, 0]))]
    d = np.linalg.norm(hv - hv[sel[0]], axis=1)
    while len(sel) < min(n, len(hv)):
        k = int(np.argmax(d)); sel.append(k); d = np.minimum(d, np.linalg.norm(hv - hv[k], axis=1))
    return hv[sel]


def gemini_pts():
    """Gemini 305 main solid vertices in the camera frame (x = tool x, u = up, t = optical axis; front face t = 0, bottom face u = -21); cached"""
    cache = os.path.join(P.SCRATCH, 'g305_hull_cam.pkl')
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader(); r.ReadFile(P.GEMINI_STEP); r.TransferRoots()
    sh = cq.Shape.cast(r.OneShape())
    main = sorted(sh.Solids(), key=lambda s: -s.Volume())[0]
    v, _ = main.tessellate(0.2, 0.3)
    v = np.array([p.toTuple() for p in v])
    pts = np.column_stack([v[:, 0] - 18.35, v[:, 2] + 17.2, -(v[:, 1] - 0.1)])
    try:
        pickle.dump(pts, open(cache, 'wb'))
    except OSError:
        pass
    return pts


def gemini_hull():
    return L.move(L.hull_solid(_decimate(gemini_pts(), 96)), T_CAM)


# ============================================================================================ assembly list
def build_parts():
    out = []
    def add(name, shape, kind, **kw):
        out.append(dict(name=name, shape=shape, kind=kind, **kw))
    add('tool_body', tool_body(), 'printed', material=P.PRINT_MATERIAL,
        orient='build with the bar front face (z = 133.5, hex-pocket side) on the bed, stem growing up, flange face on top (see data/verify.json printability); 12 hub gussets + 2 cheek walls make the hub plate and the collector ledges self-supporting; no supports needed except a few cm3 under the cheek ends')
    for i, x in enumerate(P.PAD_X):
        n = ['L', 'M', 'R'][i]
        add('pad_%s' % n, pad_body(x), 'purchased', mass=P.MASS_PURCHASED['pad'], code=P.PAD['code'])
        add('elbow_%s' % n, elbow(i), 'purchased', mass=P.MASS_PURCHASED['elbow'], code=C.PAD_FITTING['code'])
        add('nut_front_%s' % n, nut(x, P.NUT_FRONT_Z0, P.NUT_FRONT_Z1), 'purchased', mass=P.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3 (ZP2 accessory)')
        add('washer_%s' % n, washer(x, P.WASHER_Z0, P.WASHER_Z1), 'purchased', mass=P.MASS_PURCHASED['washer'], code='steel washer 10.5/20/1')
        add('nut_rear_%s' % n, nut(x, P.NUT_REAR_Z0, P.NUT_REAR_Z1, 30.0 if i == 1 else 0.0), 'purchased', mass=P.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3')
        add('tube_%s' % n, tube(i), 'purchased', mass=P.MASS_PURCHASED['tube'], code='PU tube 4 mm OD (SMC TU0425), elbow -> collector')
    add('collector', collector(), 'purchased', mass=P.MASS_PURCHASED['collector'], code=P.COLLECTOR['code'])
    add('trunk_tube', trunk_tube(), 'purchased', mass=P.MASS_PURCHASED['trunk'], code='PU tube 6 mm OD (stub; continues down the arm to the AMR pump)')
    add('camera_gemini305', gemini_hull(), 'purchased', mass=P.MASS_PURCHASED['camera'], code='Orbbec Gemini 305 (envelope = hull of the vendor STEP main solid)')
    return out
