"""All parts of the left suction/scan tool in the link6 frame (mm).  build_parts() -> list of dicts
(name, shape, kind = printed|purchased, + mass / material / print orientation notes)."""
import math, os, pickle
import numpy as np
import cadquery as cq
import params as P
import lib as L
C = P.C

Lz = P.LIP_Z
FZ0, FZ1 = P.FLOOR_Z0, P.FLOOR_Z1          # carrier floor z range (52.7 .. 60.7)
BR_Z0, BR_Z1 = Lz - P.BRACKET_REAR, Lz - P.BRACKET_FRONT      # bracket z range (lip-40 .. lip-25.5)
THREAD_R = 4.95
SCAN_X = 16.5                              # cradle half width along X
CAM_BOX_CLR = 0.3
FIT_Z0 = Lz - 61.5                         # top of the buffer body (inlet plane); the one-touch fitting hangs below it
FIT_Z1 = FIT_Z0 - P.FIT_H                  # fitting end (tube leaves here, pointing -z)
TUBE_Z = FIT_Z1 - P.TUBE_STRAIGHT - P.TUBE_BEND_R   # height of the horizontal tube run (-y)

T_CAM, T_SCAN = P.T_CAM, P.T_SCAN
ZC = float(T_CAM[2, 3])                    # camera front-face centre z


def cam_pt(x, y, z):
    return (T_CAM @ np.array([x, y, z, 1.0]))[:3]


def scn_pt(x, w, a):
    return (T_SCAN @ np.array([x, w, a, 1.0]))[:3]


def cam_box(x0, x1, y0, y1, z0, z1):
    return L.obox(cam_pt((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), T_CAM[:3, :3], [x1 - x0, y1 - y0, z1 - z0])


def scn_box(x0, x1, w0, w1, a0, a1):
    return L.obox(scn_pt((x0 + x1) / 2, (w0 + w1) / 2, (a0 + a1) / 2), T_SCAN[:3, :3], [x1 - x0, w1 - w0, a1 - a0])


# decoder pocket in the scanner arm: directly under the engine cradle, open toward +y (closed by the carrier web)
CRADLE_ZMIN = min(scn_pt(sx * SCAN_X, w, a)[2] for sx in (-1, 1) for w in (-7.35, 7.35) for a in (-17.2, 0.0))
DEC_SIZE = P.DECODER['size']               # (26.91 along y, 39.04 along x, 6.37 along z)
DEC_PZ1 = CRADLE_ZMIN - 1.5                # pocket top
DEC_PZ0 = DEC_PZ1 - (DEC_SIZE[2] + 0.43)   # pocket bottom
DEC_HB, DEC_HT = DEC_PZ0 - 2.0, CRADLE_ZMIN + 1.5      # housing bottom / top
DEC_PX = DEC_SIZE[1] / 2 + 0.43            # pocket half width (x)
DEC_PY = DEC_SIZE[0] + 0.5                 # pocket depth (y) from the arm back face (y = -RIB_HALF_Y)
DEC_HX = DEC_PX + 2.45                     # housing half width


# ============================================================================================ printed parts
def carrier():
    parts = [L.zcyl(0, 0, FZ0, FZ1, P.FLOOR_D / 2),
             L.aabox((-36, 36), (-14, 14), (FZ0, FZ1)),
             L.aabox((-P.FLAP_X, P.FLAP_X), (P.FLAP_Y[0], P.FLAP_Y[1]), (FZ0, FZ1))]           # tube strain-relief flap
    for sx in (-1, 1):
        x0, x1 = sorted((sx * P.RIB_X[0], sx * P.RIB_X[1]))
        parts.append(L.aabox((x0, x1), (-P.RIB_HALF_Y, P.RIB_HALF_Y), (FZ1 - 0.01, BR_Z0 + 0.01)))
    parts.append(L.zstadium(0, 0, BR_Z0, BR_Z1, 2 * P.BRACKET_HALF_X, 2 * P.BRACKET_HALF_Y))
    # web between the ribs: closes the decoder pocket of the scanner arm
    parts.append(L.aabox((-P.RIB_X[0] - 0.01, P.RIB_X[0] + 0.01), (-P.RIB_HALF_Y, -P.RIB_HALF_Y + 2.5), (DEC_PZ0 - 1.0, DEC_PZ1 + 1.0)))
    s = L.union(parts)
    cuts = [L.zcyl(0, 0, FZ0 - 1, FZ1 + 1, P.FLOOR_BORE / 2)]
    for x in P.PAD_X:       # pad holes + front hex pockets
        cuts.append(L.zcyl(x, 0, BR_Z0 - 1, BR_Z1 + 1, P.PAD['bracket_hole'] / 2))
        cuts.append(L.zhex(x, 0, BR_Z1 - 3.2, BR_Z1 + 1, P.HEX_POCKET_AF))
    for k in range(6):      # flange screws: M3 countersunk on PCD 43 (angles 45 + 60k), driven from +z with the carrier alone on the flange
        a = math.radians(C.PCD43_A0 + 60 * k); x, y = C.PCD43_R * math.cos(a), C.PCD43_R * math.sin(a)
        cuts.append(L.zcyl(x, y, FZ0 - 1, FZ1 + 1, C.M3_CLR / 2))
        cuts.append(cq.Solid.makeCone(C.M3_CLR / 2, 3.2, 1.5, L.V([x, y, FZ1 - 1.5]), L.V([0, 0, 1])))
        cuts.append(L.zcyl(x, y, FZ1 - 0.01, FZ1 + 1, 3.2))
        if abs(y) < P.RIB_HALF_Y + P.BRACKET_ACCESS_D / 2:      # under the bracket: access hole for screw head + driver
            cuts.append(L.zcyl(x, y, FZ1, BR_Z1 + 1, P.BRACKET_ACCESS_D / 2))
    for sx in (-1, 1):      # rib inserts: scanner arm (-y faces), camera arm (+y faces)
        for z in (Lz - 60.0, Lz - 50.5):
            cuts.append(L.ocyl([sx * P.RIB_XC, -P.RIB_HALF_Y - 1, z], [sx * P.RIB_XC, -P.RIB_HALF_Y + 7, z], C.INSERT_M3_HOLE / 2))
        for z in (ZC - 18.5, ZC - 6.5):
            cuts.append(L.ocyl([sx * P.RIB_XC, P.RIB_HALF_Y + 1, z], [sx * P.RIB_XC, P.RIB_HALF_Y - 7, z], C.INSERT_M3_HOLE / 2))
    # collector pocket in the flap: block base at the tube-run height - 8, port bosses in slots; retained by 2 zip ties over the block
    cy = P.COLL_Y; cb = P.COLLECTOR['body']; b4, b6 = P.COLLECTOR['boss4'], P.COLLECTOR['boss6']
    zt = TUBE_Z
    cuts.append(L.aabox((-cb[0] / 2 - 0.4, cb[0] / 2 + 0.4), (cy - cb[1] / 2 - 0.4, cy + cb[1] / 2 + 0.4), (zt - cb[2] / 2 - 0.2, FZ1 + 1)))
    cuts.append(L.aabox((-cb[0] / 2 - b4[1] - 0.6, cb[0] / 2 + b4[1] + 0.6), (cy - b4[0] / 2 - 0.3, cy + b4[0] / 2 + 0.3), (zt - b4[0] / 2 - 0.3, FZ1 + 1)))
    cuts.append(L.aabox((-b4[0] / 2 - 0.3, b4[0] / 2 + 0.3), (cy + cb[1] / 2, cy + cb[1] / 2 + b4[1] + 0.4), (zt - b4[0] / 2 - 0.3, FZ1 + 1)))
    cuts.append(L.aabox((-b6[0] / 2 - 0.3, b6[0] / 2 + 0.3), (cy - cb[1] / 2 - b6[1] - 0.4, cy - cb[1] / 2), (zt - b6[0] / 2 - 0.3, FZ1 + 1)))
    cuts.append(L.aabox((-3.3, 3.3), (P.TRUNK_END_Y - 1.0, cy - cb[1] / 2 - b6[1]), (zt - 3.3, FZ1 + 1)))      # groove for the phi6 trunk
    for dx in (-8.0, 8.0):  # zip-tie slits: block (both sides), trunk strain relief
        for yy in (cy + cb[1] / 2 + 3.5, cy - cb[1] / 2 - 2.0, P.TRUNK_END_Y + 1.6):
            cuts.append(L.aabox((dx - 1.7, dx + 1.7), (yy - 0.8, yy + 0.8), (FZ0 - 1, FZ1 + 1)))
    for x in (-40.0, 40.0):  # zip-tie slits either side of the long pad tubes (L, R)
        for dy in (-3.5, 3.5):
            cuts.append(L.aabox((x - 1.7, x + 1.7), (cy + dy - 0.8, cy + dy + 0.8), (FZ0 - 1, FZ1 + 1)))
    return L.cut_all(s, cuts)


def camera_arm():
    parts = []
    # attach plate on the +y faces of the ribs
    parts.append(L.aabox((-36, 36), (P.RIB_HALF_Y, P.RIB_HALF_Y + 6), (ZC - 26.5, ZC - 0.5)))
    # strut (horizontal beam toward the cradle): ends inside the cradle plate
    y_end = float(cam_pt(0, -24.0, -3.0)[1]) + 0.5
    parts.append(L.aabox((-18, 18), (P.RIB_HALF_Y, y_end), (ZC - 26.5, ZC - 14.5)))
    # cradle plate, back stop rail, USB-C relief lug (camera frame)
    parts.append(cam_box(-44.0, 24.5, -21.0 - P.CAM_CRADLE_T, -21.0, -26.25, 0.0))
    parts.append(cam_box(-24.5, 24.5, -21.0, -8.0, -26.25, -23.25))
    parts.append(cam_box(-44.0, -33.0, -21.0, -4.0, -20.0, -6.0))
    s = L.union(parts)
    cuts = [cam_box(-21.0 - CAM_BOX_CLR, 21.0 + CAM_BOX_CLR, -21.0 - CAM_BOX_CLR, 21.0 + CAM_BOX_CLR, -23.0 - CAM_BOX_CLR, 2.6)]
    # camera screws: 2 x M3 at +-10 (bottom face holes, 12 mm behind the front face)
    ymid = -21.0 - P.CAM_CRADLE_T
    for x, r, rc in ((-10.0, 1.7, 3.2), (10.0, 1.7, 3.2)):
        a = cam_pt(x, ymid - 1, -12.0); b = cam_pt(x, -21.0 + 1, -12.0); c0 = cam_pt(x, ymid - 1, -12.0); c1 = cam_pt(x, ymid + 3.0, -12.0)
        cuts.append(L.ocyl(a, b, r)); cuts.append(L.ocyl(c0, c1, rc))
    # zip-tie slot in the lug (cable relief)
    cuts.append(cam_box(-40.2, -36.8, -16.0, -10.0, -21.0, -5.0))
    # M3 clearance + counterbore for the rib screws (from +y)
    for sx in (-1, 1):
        for z in (ZC - 18.5, ZC - 6.5):
            cuts.append(L.ocyl([sx * P.RIB_XC, P.RIB_HALF_Y - 1, z], [sx * P.RIB_XC, P.RIB_HALF_Y + 7, z], 1.7))
            cuts.append(L.ocyl([sx * P.RIB_XC, P.RIB_HALF_Y + 3, z], [sx * P.RIB_XC, P.RIB_HALF_Y + 7, z], 3.15))
    return L.cut_all(s, cuts)


def scanner_arm():
    """engine cradle + decoder pocket (under the engine, open toward the ribs, closed by the carrier web); no separate cover/lid"""
    zp = Lz - 66.0
    parts = [L.aabox((-36, 36), (-P.RIB_HALF_Y - 6, -P.RIB_HALF_Y), (zp, zp + 22.0)),                   # attach plate on the -y rib faces
             L.aabox((-DEC_HX, DEC_HX), (-P.RIB_HALF_Y - DEC_PY - 2.0, -P.RIB_HALF_Y), (DEC_HB, DEC_HT)),   # decoder housing
             scn_box(-SCAN_X, SCAN_X, -7.35, 7.35, -17.2, 0.0),
             scn_box(-SCAN_X, SCAN_X, -7.35, 15.6, -17.2, -7.0).intersect(L.aabox((-SCAN_X, SCAN_X), (-60, -P.RIB_HALF_Y - 5), (zp + 3.0, zp + 24.0)))]
    s = L.union(parts)
    E = P.ENGINE
    cuts = [scn_box(-(E['long'] / 2 + 0.3), E['long'] / 2 + 0.3, -(E['thick'] / 2 + 0.25), E['thick'] / 2, -(E['depth'] + 0.5), 3.0),
            L.aabox((-DEC_PX, DEC_PX), (-P.RIB_HALF_Y - DEC_PY, -P.RIB_HALF_Y + 1.0), (DEC_PZ0, DEC_PZ1)),       # decoder pocket
            L.aabox((-7.0, 7.0), (-38.0, -30.0), (DEC_PZ1 - 0.5, DEC_PZ1 + 9.0))]                                # flex slot to the engine
    # engine hole pattern through the +w wall: C1/C2 = M1.4 clearance holes (A/B locator holes are not used: a peg would block sliding the engine in)
    for u, a in ((9.25, -4.5), (-4.05, -3.3)):
        cuts.append(L.ocyl(scn_pt(u, E['thick'] / 2 - 0.5, a), scn_pt(u, E['thick'] / 2 + 3.6, a), 0.8))
    for sx in (-1, 1):   # M3 clearance + counterbore from -y
        for z in (Lz - 60.0, Lz - 50.5):
            cuts.append(L.ocyl([sx * P.RIB_XC, -P.RIB_HALF_Y + 1, z], [sx * P.RIB_XC, -P.RIB_HALF_Y - 7, z], 1.7))
            cuts.append(L.ocyl([sx * P.RIB_XC, -P.RIB_HALF_Y - 3, z], [sx * P.RIB_XC, -P.RIB_HALF_Y - 7, z], 3.15))
    # driver window through the attach plate behind the two M1.4 screws (screws are driven from the +w side before the arm is mounted)
    zm = float(scn_pt(0, 7.4, -4.0)[2])
    cuts.append(L.aabox((-11.5, 11.5), (-P.RIB_HALF_Y - 7, -P.RIB_HALF_Y + 1), (zm - 7.0, zm + 4.5)))
    return L.cut_all(s, cuts)


# ============================================================================================ purchased parts
def pad_body(x):
    """ZP2-T4030WNK10-04 proxy from the catalog drawing (lip at z = LIP_Z, oval long axis along X) incl. the built-in phi4 one-touch fitting"""
    z = lambda d: Lz - d
    prts = [L.zstadium(x, 0, z(1.5), z(0), 30.8, 4.8),                 # lip
            L.zstadium(x, 0, z(8.0), z(1.5), 24.0, 3.6),               # pad skirt (frustum approximated)
            L.zstadium(x, 0, z(14.0), z(8.0), 16.0, 4.0),              # pad adapter
            L.zcyl(x, 0, z(25.5), z(14.0), 4.0),                       # guide rod (D = 11.5)
            L.zcyl(x, 0, z(48.5), z(25.5), THREAD_R),                  # M10x1 thread zone (E = 23)
            L.zhex(x, 0, z(61.5), z(48.5), 8.0),                       # inlet hex 8 AF
            L.zcyl(x, 0, FIT_Z1, FIT_Z0 + 0.01, P.FIT_D / 2)]          # one-touch fitting body (14.7 tall, tube leaves along -z)
    return L.union(prts)


def nut(x, z0, z1):
    return L.zhex(x, 0, z0, z1, P.NUT_AF).cut(L.zcyl(x, 0, z0 - 1, z1 + 1, 5.0))


def washer(x, z0, z1):
    return L.zcyl(x, 0, z0, z1, P.WASHER['od'] / 2).cut(L.zcyl(x, 0, z0 - 1, z1 + 1, P.WASHER['id'] / 2))


def fillet_path(pts, R, n=6):
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
        info['seg_len'].append(l1 - ts[i - 1] - t)         # straight length left between the arcs
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
    L_ = sum(np.linalg.norm(b - a) for a, b in zip(out[:-1], out[1:]))
    info['length'] = float(L_)
    return out, info


def tube_waypoints(i):
    x = P.PAD_X[i]; cy = P.COLL_Y; cb = P.COLLECTOR['body']; b4 = P.COLLECTOR['boss4']
    p0 = [x, 0.0, FIT_Z1]; c1 = [x, 0.0, TUBE_Z]
    if i == 1:
        return [p0, c1, [0.0, cy + cb[1] / 2 + b4[1], TUBE_Z]]                       # M: straight down, R12 bend toward -y into the +y port
    xe = -(cb[0] / 2 + b4[1]) if i == 0 else (cb[0] / 2 + b4[1])
    return [p0, c1, [x, cy, TUBE_Z], [xe, cy, TUBE_Z]]                                 # L / R: down, bend -y, bend toward the centre into the +-x port


TUBE_INFO = {}


def tube(i):
    """phi4 tube pad fitting -> collector port: straight 8 mm out of the fitting, R12 bends, lying 0.15 mm above the flap"""
    path, info = fillet_path(tube_waypoints(i), P.TUBE_BEND_R)
    TUBE_INFO[i] = info
    return L.tube_polyline([p.tolist() for p in path], P.TUBE_OD / 2 - 0.05)


def collector():
    """3 x phi4 + 1 x phi6 push-in collector block (proxy): ports phi4 on -x, +x, +y faces, phi6 trunk on the -y face; axis height = tube run"""
    cy = P.COLL_Y; cb = P.COLLECTOR['body']; b4, b6 = P.COLLECTOR['boss4'], P.COLLECTOR['boss6']; zt = TUBE_Z
    body = L.aabox((-cb[0] / 2, cb[0] / 2), (cy - cb[1] / 2, cy + cb[1] / 2), (zt - cb[2] / 2, zt + cb[2] / 2))
    bos = [L.ocyl([-cb[0] / 2 + 0.01, cy, zt], [-cb[0] / 2 - b4[1], cy, zt], b4[0] / 2),
           L.ocyl([cb[0] / 2 - 0.01, cy, zt], [cb[0] / 2 + b4[1], cy, zt], b4[0] / 2),
           L.ocyl([0, cy + cb[1] / 2 - 0.01, zt], [0, cy + cb[1] / 2 + b4[1], zt], b4[0] / 2),
           L.ocyl([0, cy - cb[1] / 2 + 0.01, zt], [0, cy - cb[1] / 2 - b6[1], zt], b6[0] / 2)]
    return L.union([body] + bos)


def trunk_tube():
    """phi6 trunk stub leaving along -y (continues to the ZK2 near link 3)"""
    cy = P.COLL_Y; cb = P.COLLECTOR['body']; b6 = P.COLLECTOR['boss6']
    return L.ocyl([0, cy - cb[1] / 2 - b6[1], TUBE_Z], [0, P.TRUNK_END_Y, TUBE_Z], P.TRUNK_OD / 2 - 0.05)


def scanner_engine():
    E = P.ENGINE
    s = scn_box(-E['long'] / 2, E['long'] / 2, -E['thick'] / 2, E['thick'] / 2, -E['depth'], 0.0)
    for u, a, r in ((6.10, -4.5, 0.825), (-9.70, -3.3, 0.525), (9.25, -4.5, 0.7), (-4.05, -3.3, 0.7)):
        s = s.cut(L.ocyl(scn_pt(u, E['thick'] / 2 + 0.5, a), scn_pt(u, E['thick'] / 2 - 2.1, a), r))
    return s


def decoder_board():
    sx, sy, sz = DEC_SIZE
    y1 = -P.RIB_HALF_Y - 0.25
    return L.aabox((-sy / 2, sy / 2), (y1 - sx, y1), (DEC_PZ0 + 0.2, DEC_PZ0 + 0.2 + sz))


def _decimate(pts, n):
    """hull vertices, then farthest-point sampling to n points (keeps the extremes; hull of them is a light envelope)"""
    from scipy.spatial import ConvexHull
    hv = pts[ConvexHull(pts).vertices]
    sel = [int(np.argmax(hv[:, 0]))]
    d = np.linalg.norm(hv - hv[sel[0]], axis=1)
    while len(sel) < min(n, len(hv)):
        k = int(np.argmax(d)); sel.append(k); d = np.minimum(d, np.linalg.norm(hv - hv[k], axis=1))
    return hv[sel]


def gemini_hull():
    """Gemini 305 body: convex-hull envelope of the vendor STEP main solid, posed in link6 (hull cached in the scratch dir)"""
    cache = os.path.join(P.SCRATCH, 'g305_hull_cam.pkl')
    if os.path.exists(cache):
        pts = pickle.load(open(cache, 'rb'))
    else:
        from OCP.STEPControl import STEPControl_Reader
        r = STEPControl_Reader(); r.ReadFile(P.GEMINI_STEP); r.TransferRoots()
        sh = cq.Shape.cast(r.OneShape())
        main = sorted(sh.Solids(), key=lambda s: -s.Volume())[0]
        v, _ = main.tessellate(0.2, 0.3)
        v = np.array([p.toTuple() for p in v])
        # STEP frame -> camera frame: x_c = x_s - 18.35, y_c = z_s + 17.2 (bottom mount face at -21), z_c = -(y_s - 0.1) (optical axis = -y_s)
        pts = np.column_stack([v[:, 0] - 18.35, v[:, 2] + 17.2, -(v[:, 1] - 0.1)])
        pickle.dump(pts, open(cache, 'wb'))
    return L.move(L.hull_solid(_decimate(pts, 96)), T_CAM), pts


# ============================================================================================ assembly list
def build_parts():
    out = []
    def add(name, shape, kind, **kw):
        out.append(dict(name=name, shape=shape, kind=kind, **kw))
    add('carrier', carrier(), 'printed', material='PETG (PA12-CF optional)', orient='flange face (z = 52.7, floor underside) down on the bed, ribs vertical; supports only inside the bracket hex pockets (bridged) and under the bracket overhang ends')
    add('camera_arm', camera_arm(), 'printed', material='PETG', orient='attach plate (+y face of the ribs) on the bed; beam grows up; cradle plate underside needs support')
    add('scanner_arm', scanner_arm(), 'printed', material='PETG', orient='attach plate (-y face of the ribs... i.e. the y = -12 contact face) on the bed; decoder pocket opens to the bed (39 x 6.8 mm roof bridged)')
    for i, x in enumerate(P.PAD_X):
        n = ['L', 'M', 'R'][i]
        add('pad_%s' % n, pad_body(x), 'purchased', mass=P.MASS_PURCHASED['pad'], code=P.PAD['code'])
        add('nut_front_%s' % n, nut(x, Lz - 28.5, Lz - 25.5), 'purchased', mass=P.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3 (ZP2 accessory)')
        add('washer_%s' % n, washer(x, Lz - 41.0, Lz - 40.0), 'purchased', mass=P.MASS_PURCHASED['washer'], code='steel washer 10.5/20/1')
        add('nut_rear_%s' % n, nut(x, Lz - 44.0, Lz - 41.0), 'purchased', mass=P.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3')
        add('tube_%s' % n, tube(i), 'purchased', mass=P.MASS_PURCHASED['tube'], code='nylon/PU tube 4 mm OD, pad fitting -> collector')
    add('collector', collector(), 'purchased', mass=P.COLLECTOR['mass'], code=P.COLLECTOR['code'])
    add('trunk_tube', trunk_tube(), 'purchased', mass=P.MASS_PURCHASED['trunk'], code='nylon/PU tube 6 mm OD (stub; runs off-tool to the ZK2 near link 3)')
    add('scanner_se4710', scanner_engine(), 'purchased', mass=P.MASS_PURCHASED['scanner'], code='Zebra SE4710 (SE4710-A00-xxx)')
    add('decoder_pl3307', decoder_board(), 'purchased', mass=P.MASS_PURCHASED['decoder'], code='Zebra PL3307-B decoder board (+ flex)')
    gh, _ = gemini_hull()
    add('camera_gemini305', gh, 'purchased', mass=P.MASS_PURCHASED['camera'], code='Orbbec Gemini 305 (envelope = hull of the vendor STEP main solid)')
    return out
