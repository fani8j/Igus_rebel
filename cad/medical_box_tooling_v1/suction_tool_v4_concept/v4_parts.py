"""All parts of the v4 concept tool in the link6 frame (mm).  build_parts() -> list of dicts (name, shape, kind = printed|purchased, mass, ...)."""
import math
import numpy as np
import cadquery as cq
import v4_params as V
from bridge import P3, PA3, L, C

Lz = V.LIP_Z
TC, TS = V.T_CAM, V.T_SCAN


def cam_pt(x, u, t):
    return (TC @ np.array([x, u, t, 1.0]))[:3]


def scn_pt(l, w, a):
    return (TS @ np.array([l, w, a, 1.0]))[:3]


def cam_box(x0, x1, u0, u1, t0, t1):
    return L.obox(cam_pt((x0 + x1) / 2, (u0 + u1) / 2, (t0 + t1) / 2), TC[:3, :3], [x1 - x0, u1 - u0, t1 - t0])


def scn_box(l0, l1, w0, w1, a0, a1):
    return L.obox(scn_pt((l0 + l1) / 2, (w0 + w1) / 2, (a0 + a1) / 2), TS[:3, :3], [l1 - l0, w1 - w0, a1 - a0])


def yz(u, t):          # camera-frame (u, t) -> (y, z) of the x = 0 plane
    p = cam_pt(0, u, t)
    return float(p[1]), float(p[2])


def xz_prism(pts, y0, y1):
    return cq.Workplane('XZ').polyline([(float(a), float(b)) for a, b in pts]).close().extrude(float(y1 - y0)).translate((0, float(y1), 0)).val() if False else \
        L.move(cq.Workplane('XZ').polyline([(float(a), float(b)) for a, b in pts]).close().extrude(float(y1 - y0)).val(),
               np.array([[1, 0, 0, 0], [0, 1, 0, y1], [0, 0, 1, 0], [0, 0, 0, 1.0]]))   # XZ workplane extrudes toward -y


# ============================================================================================ printed parts
FLANGE_HOLES = [(C.PCD43_R * math.cos(math.radians(C.PCD43_A0 + 60 * k)), C.PCD43_R * math.sin(math.radians(C.PCD43_A0 + 60 * k))) for k in range(6)]
POD_SCREW_XZ = [(sx * 10.0, z) for sx in (-1, 1) for z in (76.0, 84.0)]


def hub_stem():
    hub = L.zcyl(0, 0, V.HUB_Z0, V.HUB_Z1, V.HUB_D / 2)
    stem = cq.Workplane('XY').workplane(offset=V.HUB_Z1 - 0.01).rect(V.STEM_X, V.STEM_Y).extrude(V.BAR_Z0 - V.HUB_Z1 + 0.01).edges('|Z').fillet(V.STEM_R).val()
    gx = V.STEM_X / 2 - 0.1
    collar = xz_prism([(-gx, 101.3), (-V.COLLAR_X, V.COLLAR_Z0), (-V.COLLAR_X, V.BAR_Z0), (V.COLLAR_X, V.BAR_Z0), (V.COLLAR_X, V.COLLAR_Z0), (gx, 101.3)],
                      -V.STEM_Y / 2, V.STEM_Y / 2)
    s = L.union([hub, stem, collar])
    cuts = []
    cav = cq.Workplane('XY').workplane(offset=V.CAV_FLOOR_Z).rect(V.CAV_X, V.CAV_Y).extrude(V.BAR_Z0 + 1 - V.CAV_FLOOR_Z).val()
    cuts.append(cav)
    zc = V.COLL_ZC
    for sx in (-1, 1):           # windows for the two phi4 port bosses
        cuts.append(L.ocyl([sx * 11.0, 0, zc], [sx * 17.0, 0, zc], 5.2))
    cuts.append(L.ocyl([0, -7.0, zc], [0, -12.5, zc], 6.2))        # trunk boss window (-y)
    cuts.append(L.ocyl([0, -11.0, zc], [0, -30.0, zc], 3.3))       # trunk groove in the hub rim
    for k, (x, y) in enumerate(FLANGE_HOLES):                       # 6 x M3 CSK on PCD 43, driven from +z with the pod removed
        cuts.append(L.zcyl(x, y, V.HUB_Z0 - 1, V.HUB_Z1 + 1, C.M3_CLR / 2))
        cuts.append(cq.Solid.makeCone(C.M3_CLR / 2, 3.2, 1.5, L.V([x, y, V.HUB_Z1 - 1.5]), L.V([0, 0, 1])))
        cuts.append(L.zcyl(x, y, V.HUB_Z1 - 0.01, V.HUB_Z1 + 1, 3.2))
    for sx in (-1, 1):                                              # bar inserts in the collar (from the top face)
        cuts.append(L.zcyl(sx * V.BAR_SCREW_X, 0, V.BAR_Z0 - V.INSERT_DEPTH, V.BAR_Z0 + 1, C.INSERT_M3_HOLE / 2))
    for x, z in POD_SCREW_XZ:                                       # pod screws: clearance through the +y wall, heads inside the cavity
        cuts.append(L.ocyl([x, 7.0, z], [x, V.POD_Y0 + 0.5, z], 1.7))
    return L.cut_all(s, cuts)


def sensor_pod():
    c = math.cos(math.radians(V.CAM_TILT_DEG))
    H = V.CAM_BLOCK_HALF; B0 = V.CAM_BACK_T - V.POD_WALL
    A = yz(-H, 0.0); Bp = yz(H, 0.0); Cp = yz(H, B0); D = yz(-H, B0)
    belly_y = Cp[0] - 8.0
    z_bot = V.HUB_Z1 + 1.8
    prof = [(V.POD_Y0, z_bot), (belly_y, z_bot), Cp, Bp, (V.POD_Y0, A[1]), ]
    body = L.extrude_poly_x(L.hull2d(prof + [A, D]).tolist(), -H, H)
    parts = [body]
    parts.append(scn_box(-V.ENG_BLOCK_L, V.ENG_BLOCK_L, -V.ENG_BLOCK_W, V.ENG_BLOCK_W, V.ENG_BACK_A - V.POD_WALL, 0.0))     # scanner lobe, front flush with the window
    parts.append(cam_box(V.CAM_POCKET_HALF - 1.0, 37.5, -11.0, 9.0, -21.0, -7.0))                                           # USB-C lobe (+x)
    parts.append(L.aabox((-36.0, -24.0), (43.0, 54.0), (z_bot, 103.0)))                                                      # keel: carries the flex channel from the engine down to the decoder
    s = L.union(parts)
    cuts = [cam_box(-V.CAM_POCKET_HALF, V.CAM_POCKET_HALF, -V.CAM_POCKET_HALF, V.CAM_POCKET_HALF, V.CAM_BACK_T, 6.0)]        # camera pocket, open to the front
    for sx in (-1, 1):                                                                                                        # 2 x M3 CSK into the camera back face
        x = sx * V.CAM_BACK_SCREW_X
        cuts.append(L.ocyl(cam_pt(x, 0, V.CAM_BACK_T + 0.1), cam_pt(x, 0, B0 - 1.0), 1.7))
        cuts.append(cq.Solid.makeCone(1.7, 3.2, 1.5, L.V(cam_pt(x, 0, B0 + 1.5)), L.V(-TC[:3, 2])))
    cuts.append(cam_box(V.CAM_POCKET_HALF - 1.5, 36.0, -8.0, 8.0, -19.0, -10.0))                                             # USB-C plug pocket in the +x wall
    cuts.append(cam_box(24.0, 36.0, -12.5, -5.0, -17.5, -11.5))                                                              # cable exit down (-u)
    E = V.ENGINE
    cuts.append(scn_box(-V.ENG_POCKET_L, V.ENG_POCKET_L, -V.ENG_POCKET_W, V.ENG_POCKET_W, V.ENG_BACK_A, 8.0))                 # engine pocket, open to the front
    for u, a in ((9.25, -4.5), (-4.05, -3.3)):                                                                                # M1.4 through the +w wall (outside of the pod)
        cuts.append(L.ocyl(scn_pt(u, V.ENG_POCKET_W - 0.5, a), scn_pt(u, V.ENG_POCKET_W + 3.6, a), 0.8))
    # decoder: horizontal pocket under the camera back wall, open at the +x end (cables leave together with the USB-C cable), flex channel to the engine
    dz0 = z_bot + 3.0
    dy0 = 27.0
    cuts.append(L.aabox((-15.5, H + 1), (dy0, dy0 + V.DECODER['size'][0] + 0.5), (dz0, dz0 + V.DECODER['size'][2] + 0.45)))
    cuts.append(L.aabox((-32.0, -14.0), (46.0, 51.0), (dz0 + 1.0, dz0 + 8.0)))
    cuts.append(L.aabox((-33.0, -26.0), (46.0, 51.0), (dz0 + 1.0, 100.0)))
    r = L.cut_all(s, cuts)
    keep = [so for so in r.Solids() if so.Volume() > 5.0]
    return keep[0] if len(keep) == 1 else cq.Compound.makeCompound(keep)


# ============================================================================================ machined / purchased parts
def pad_bar():
    s = L.zstadium(0, 0, V.BAR_Z0, V.BAR_Z1, V.BAR_LEN, V.BAR_W)
    cuts = []
    for x in V.PAD_X:
        cuts.append(L.zcyl(x, 0, V.BAR_Z0 - 1, V.BAR_Z1 + 1, P3.PAD['bracket_hole'] / 2))
        cuts.append(L.zhex(x, 0, V.BAR_Z1 - V.HEX_POCKET_DEPTH, V.BAR_Z1 + 1, V.HEX_POCKET_AF))     # front-nut keying pocket
    for sx in (-1, 1):                                                                              # 2 x M3 CSK from the front into the collar inserts
        x = sx * V.BAR_SCREW_X
        cuts.append(L.zcyl(x, 0, V.BAR_Z0 - 1, V.BAR_Z1 + 1, 1.7))
        cuts.append(cq.Solid.makeCone(1.7, 3.2, 1.5, L.V([x, 0, V.BAR_Z1 - 1.5]), L.V([0, 0, 1])))
        cuts.append(L.zcyl(x, 0, V.BAR_Z1 - 0.01, V.BAR_Z1 + 1, 3.2))
    return L.cut_all(s, cuts)


def collector():
    bx, by, bz = V.COLL_BODY; zc = V.COLL_ZC
    body = L.aabox((-bx / 2, bx / 2), (-by / 2, by / 2), (zc - bz / 2, zc + bz / 2))
    d4, l4 = V.COLL_BOSS4; d6, l6 = V.COLL_BOSS6
    bos = [L.ocyl([-bx / 2 + 0.01, 0, zc], [-bx / 2 - l4, 0, zc], d4 / 2), L.ocyl([bx / 2 - 0.01, 0, zc], [bx / 2 + l4, 0, zc], d4 / 2),
           L.ocyl([0, 0, zc + bz / 2 - 0.01], [0, 0, zc + bz / 2 + l4], d4 / 2), L.ocyl([0, -by / 2 + 0.01, zc], [0, -by / 2 - l6, zc], d6 / 2)]
    return L.union([body] + bos)


def trunk_tube():
    by = V.COLL_BODY[1]
    return L.ocyl([0, -by / 2 - V.COLL_BOSS6[1], V.COLL_ZC], [0, V.TRUNK_END_Y, V.COLL_ZC], V.TRUNK_OD / 2 - 0.05)


TUBE_INFO = {}


def tube(i):
    x = V.PAD_X[i]; zc = V.COLL_ZC; bx = V.COLL_BODY[0]; l4 = V.COLL_BOSS4[1]
    p0 = [x, 0.0, V.FIT_Z1]
    if i == 1:
        top = zc + V.COLL_BODY[2] / 2 + l4
        pts = [p0, [0.0, 0.0, top]]
        TUBE_INFO[i] = dict(bends=[], seg_len=[V.FIT_Z1 - top], length=V.FIT_Z1 - top)
        return L.ocyl(p0, pts[1], V.TUBE_OD / 2 - 0.05)
    xe = (-1 if i == 0 else 1) * (bx / 2 + l4)
    path, info = PA3.fillet_path([p0, [x, 0.0, zc], [xe, 0.0, zc]], V.TUBE_BEND_R)
    TUBE_INFO[i] = info
    return L.tube_polyline([p.tolist() for p in path], V.TUBE_OD / 2 - 0.05)


def scanner_engine():
    E = V.ENGINE
    s = scn_box(-E['long'] / 2, E['long'] / 2, -E['thick'] / 2, E['thick'] / 2, -E['depth'], 0.0)
    for u, a, r in ((6.10, -4.5, 0.825), (-9.70, -3.3, 0.525), (9.25, -4.5, 0.7), (-4.05, -3.3, 0.7)):
        s = s.cut(L.ocyl(scn_pt(u, E['thick'] / 2 + 0.5, a), scn_pt(u, E['thick'] / 2 - 2.1, a), r))
    return s


def decoder_board():
    sx, sy, sz = V.DECODER['size']                    # (26.91, 39.04, 6.37)
    z_bot = V.HUB_Z1 + 1.8; dz0 = z_bot + 3.0 + 0.2; dy0 = 27.0 + 0.25
    return L.aabox((-15.3, -15.3 + sy), (dy0, dy0 + sx), (dz0, dz0 + sz))     # slid in from +x; pushed to the -x end of its pocket


def gemini_hull_v4():
    _, pts = PA3.gemini_hull()
    return L.move(L.hull_solid(PA3._decimate(pts, 96)), TC)


def build_parts():
    out = []
    def add(name, shape, kind, **kw):
        out.append(dict(name=name, shape=shape, kind=kind, **kw))
    add('hub_stem', hub_stem(), 'printed', material='PETG (PA12-CF optional)', orient='flange face (z = 52.7) on the bed, stem vertical; supports: collar gussets are 45 deg, windows/holes small')
    add('sensor_pod', sensor_pod(), 'printed', material='PETG', orient='see printability in data/verify.json')
    bar = pad_bar()
    add('bar_al6061', bar, 'purchased', mass=bar.Volume() * V.AL_DENSITY, code='6061-T6 flat bar 20 x 15, 150 long, milled: 3 x phi10.5, 3 hex pockets, 2 CSK M3')
    for i, x in enumerate(V.PAD_X):
        n = ['L', 'M', 'R'][i]
        add('pad_%s' % n, PA3.pad_body(x), 'purchased', mass=V.MASS_PURCHASED['pad'], code=P3.PAD['code'])
        add('nut_front_%s' % n, PA3.nut(x, V.NUT_FRONT_Z0, V.NUT_FRONT_Z1), 'purchased', mass=V.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3')
        add('nut_rear_%s' % n, PA3.nut(x, V.NUT_REAR_Z0, V.NUT_REAR_Z1), 'purchased', mass=V.MASS_PURCHASED['nut'], code='M10x1 nut 14 AF x 3')
        add('tube_%s' % n, tube(i), 'purchased', mass=V.MASS_PURCHASED['tube'], code='nylon/PU tube 4 mm OD')
    add('collector', collector(), 'purchased', mass=V.COLL['mass'], code=V.COLL['code'])
    add('trunk_tube', trunk_tube(), 'purchased', mass=V.MASS_PURCHASED['trunk'], code='tube 6 mm OD stub')
    add('scanner_se4710', scanner_engine(), 'purchased', mass=V.MASS_PURCHASED['scanner'], code='Zebra SE4710')
    add('decoder_pl3307', decoder_board(), 'purchased', mass=V.MASS_PURCHASED['decoder'], code='Zebra PL3307-B')
    add('camera_gemini305', gemini_hull_v4(), 'purchased', mass=V.MASS_PURCHASED['camera'], code='Orbbec Gemini 305 (hull of vendor STEP)')
    return out
