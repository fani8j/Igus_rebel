"""Geometry of the vacuum module (cadquery).  solids() -> ordered dict name -> dict(shape, kind, color, ...).
kind: printed | hardware | foam | film.  Frame and numbers: see params.py (mm, z up, +x = service door)."""
import math
import numpy as np
import cadquery as cq
import params as P


# ----------------------------------------------------------------------------- primitives
def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane('XY').box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cylz(x, y, z0, z1, d):
    return cq.Workplane('XY').circle(d / 2).extrude(z1 - z0).translate((x, y, z0))


def cylx(x0, x1, y, z, d):
    return cq.Workplane('YZ').circle(d / 2).extrude(x1 - x0).translate((x0, y, z))


def cyly(y0, y1, x, z, d):
    return cq.Workplane('XZ').circle(d / 2).extrude(-(y1 - y0)).translate((x, y0, z))


def U(*ws):
    r = ws[0]
    for w in ws[1:]:
        r = r.union(w)
    return r


def shp(w):
    return w.val() if hasattr(w, 'val') else w


def tess(shape, tol=0.05, ang=0.2):
    import trimesh
    vs, fs, off = [], [], 0
    for so in (shape.Solids() or [shape]):
        v, t = so.tessellate(tol, ang)
        vs += [x.toTuple() for x in v]
        fs += [[i + off for i in tri] for tri in t]
        off += len(v)
    m = trimesh.Trimesh(vertices=np.array(vs), faces=np.array(fs), process=True)
    m.merge_vertices()
    return m


# ----------------------------------------------------------------------------- ports / routes
def ports():
    Z = P.Z_LINE
    e = P.ELBOW
    d = {}
    d['elbow_in_end'] = (P.PORT_IN[0] + e[2], P.PORT_IN[1], Z)
    d['elbow_out_end'] = (P.PORT_OUT[0] + e[2], P.PORT_OUT[1], Z)
    d['muffler_in'] = (P.MUFFLER_C[0] - P.MUFFLER[1] / 2, P.MUFFLER_C[1], Z)
    d['muffler_out'] = (P.MUFFLER_C[0] + P.MUFFLER[1] / 2, P.MUFFLER_C[1], Z)
    d['hepa_in'] = (P.HEPA_C[0], P.HEPA_C[1], P.HEPA_C[2] + P.HEPA[0] / 2)
    d['hepa_out'] = (P.HEPA_C[0], P.HEPA_C[1], P.HEPA_C[2] - P.HEPA[0] / 2)
    d['exh_fit_top'] = (P.EXH_XY[0], P.EXH_XY[1], 18.0)
    d['dust_out'] = (P.DUST_C[0] - P.DUST[1] / 2, P.DUST_C[1], Z)
    d['dust_in'] = (P.DUST_C[0] + P.DUST[1] / 2, P.DUST_C[1], Z)
    mx, my, mz = P.MANI_C
    ml, mw, mh = P.MANIFOLD                       # x, y, z
    d['mani_xminus'] = (mx - ml / 2, P.DUST_C[1], Z)
    d['mani_top'] = (mx, my, mz + mh / 2)
    d['mani_yminus'] = (mx, my - mw / 2, Z)
    d['mani_yplus'] = (mx, my + mw / 2, Z)
    d['R_port'] = (P.R_XY[0], P.R_XY[1], P.R_PORT_Z)
    d['valve_port'] = (P.MANI_C[0], P.VALVE_Y[1], Z)
    d['switch_port'] = (P.SWITCH_C[0], P.SWITCH_C[1] - P.SW_ENV[1] / 2, Z)
    return d


def routes():
    """tube name -> (waypoints, OD, (component a, component b)).  Corners are filleted with BEND_R."""
    p = ports()
    Z = P.Z_LINE
    R = {}
    R['tube_exhaust_pump_muffler'] = ([p['elbow_out_end'], p['muffler_in']], 8.0, ('elbow_out', 'muffler'))
    R['tube_exhaust_muffler_hepa'] = ([p['muffler_out'], (p['hepa_in'][0], p['muffler_out'][1], Z), p['hepa_in']], 8.0, ('muffler', 'hepa_filter'))
    R['tube_exhaust_hepa_outlet'] = ([p['hepa_out'], p['exh_fit_top']], 8.0, ('hepa_filter', 'exhaust_fitting'))
    R['tube_vac_filter_pump'] = ([p['dust_out'], p['elbow_in_end']], 8.0, ('dust_filter', 'elbow_in'))
    R['tube_vac_manifold_filter'] = ([p['mani_xminus'], p['dust_in']], 8.0, ('manifold', 'dust_filter'))
    R['tube_vac_bulkhead_manifold'] = ([p['R_port'], p['mani_top']], 8.0, ('bulkhead_fitting', 'manifold'))
    R['tube_vac_manifold_valve'] = ([p['mani_yminus'], p['valve_port']], 8.0, ('manifold', 'vent_valve'))
    R['tube_vac_manifold_switch'] = ([p['mani_yplus'], p['switch_port']], 8.0, ('manifold', 'vacuum_switch'))
    return R


def fillet_path(pts, r):
    """polyline with arcs of radius r at the interior vertices -> list of ('L', a, b) / ('A', a, mid, b); also (min_straight_margin)"""
    pts = [np.array(q, float) for q in pts]
    segs = []
    cur = pts[0]
    margin = 1e9
    tl = [0.0] * len(pts)
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]; d1 /= np.linalg.norm(d1)
        d2 = pts[i + 1] - pts[i]; d2 /= np.linalg.norm(d2)
        phi = math.acos(max(-1.0, min(1.0, float(d1 @ d2))))
        tl[i] = r * math.tan(phi / 2)
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]; l1 = np.linalg.norm(d1); d1 /= l1
        d2 = pts[i + 1] - pts[i]; l2 = np.linalg.norm(d2); d2 /= l2
        t = tl[i]
        margin = min(margin, l1 - tl[i] - tl[i - 1], l2 - tl[i] - tl[i + 1])
        a, b = pts[i] - d1 * t, pts[i] + d2 * t
        n = (d2 - d1); n /= np.linalg.norm(n)
        phi = math.acos(max(-1.0, min(1.0, float(d1 @ d2))))
        c = pts[i] + n * (r / math.cos(phi / 2))
        mid = c - n * r
        segs.append(('L', cur, a)); segs.append(('A', a, mid, b))
        cur = b
    segs.append(('L', cur, pts[-1]))
    return segs, margin


def tube_solid(pts, od, r=P.BEND_R):
    segs, margin = fillet_path(pts, r)
    if len(segs) == 1:
        a, b = segs[0][1], segs[0][2]
        v = b - a; L = float(np.linalg.norm(v))
        pl = cq.Plane(origin=tuple(a), xDir=_perp(v), normal=tuple(v / L))
        return cq.Workplane(pl).circle(od / 2).extrude(L).val(), margin
    V = lambda q: cq.Vector(*[float(x) for x in q])
    edges = []
    for s in segs:
        if s[0] == 'L':
            if np.linalg.norm(s[2] - s[1]) > 1e-6:
                edges.append(cq.Edge.makeLine(V(s[1]), V(s[2])))
        else:
            edges.append(cq.Edge.makeThreePointArc(V(s[1]), V(s[2]), V(s[3])))
    wire = cq.Wire.assembleEdges(edges)
    a = segs[0][1]; v = segs[0][2] - segs[0][1]; v = v / np.linalg.norm(v)
    pl = cq.Plane(origin=tuple(a), xDir=_perp(v), normal=tuple(v))
    prof = cq.Workplane(pl).circle(od / 2)
    sol = prof.sweep(cq.Workplane(obj=wire), transition='round').val()
    return sol, margin


def _perp(v):
    v = np.asarray(v, float); v = v / np.linalg.norm(v)
    t = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
    w = np.cross(v, t); w /= np.linalg.norm(w)
    return tuple(w)


# ----------------------------------------------------------------------------- printed parts
def duct_env(side):
    """envelope box (foam keep-out) of the vent duct on the wall side (-1 = intake on -y, +1 = exhaust on +y)"""
    zc = P.FAN_XZ[1] if side < 0 else P.EXH_Z
    y0, y1 = (-P.Y1 + P.WALL + 0.0, -P.CAB_Y) if side < 0 else (P.CAB_Y, P.Y1 - P.WALL)
    return box(P.DUCT_X[0], P.DUCT_X[1], y0 - 0.0, y1, zc - P.DUCT_H / 2, zc + P.DUCT_H / 2)


def make_duct(side):
    zc = P.FAN_XZ[1] if side < 0 else P.EXH_Z
    yo = P.Y1 - P.WALL                              # 90: wall inner face
    yi = P.CAB_Y                                    # 79.7: cabin (film) face
    yc = yi + P.DUCT_INNER_T                        # 82.7: outer face of the inner wall
    s = side
    ylo = lambda a, b: (min(s * a, s * b), max(s * a, s * b))
    blk = box(P.DUCT_X[0], P.DUCT_X[1], *ylo(yi, yo), zc - P.DUCT_H / 2, zc + P.DUCT_H / 2)
    cav = box(P.DUCT_X[0] + P.DUCT_WALL, P.DUCT_X[1] - P.DUCT_WALL, *ylo(yc, yo + 0.01), zc - P.DUCT_H / 2 + P.DUCT_WALL, zc + P.DUCT_H / 2 - P.DUCT_WALL)
    duct = blk.cut(cav)
    opening = cyly(*ylo(yi - 1, yc + 1), P.FAN_XZ[0], zc, 36.0)
    duct = duct.cut(opening)
    if side < 0:                                   # fan pilot holes (M3 self-tapping) in the inner wall
        for dx in (-16, 16):
            for dz in (-16, 16):
                duct = duct.cut(cyly(-yi - 0.1, -yi - 0.1 - 2.6, P.FAN_XZ[0] + dx, zc + dz, 2.6).translate((0, 0, 0)))
    slots = None
    for k in range(P.SLOT_N):
        x0 = P.SLOT_X0 + k * P.SLOT_PITCH
        sl = box(x0, x0 + P.SLOT_W, *ylo(yo - 0.5, P.Y1 + 0.5), zc - P.SLOT_H / 2, zc + P.SLOT_H / 2)
        slots = sl if slots is None else slots.union(sl)
    return duct, slots


def housing():
    xr = P.X0 + P.WALL                              # -117
    H = box(P.X0, P.DOOR_X, P.Y0, P.Y1, P.BASE_T, P.Z_TOP)
    cav = box(xr, P.DOOR_X + 1, -(P.Y1 - P.WALL), P.Y1 - P.WALL, P.BASE_T - 1, P.Z_TOP - P.WALL)
    H = H.cut(cav)
    # bottom ledge on rear + both sides: foam rests on it, M4 inserts for the base screws
    led = box(xr, P.DOOR_X, -(P.Y1 - P.WALL), P.Y1 - P.WALL, P.BASE_T, P.BASE_T + P.LEDGE_T)
    led = led.cut(box(P.X0 + P.INNER - 0.0, P.DOOR_X + 1, -P.CAB_Y - 0.3, P.CAB_Y + 0.3, P.BASE_T - 1, P.BASE_T + P.LEDGE_T + 1))
    H = H.union(led)
    # corner blocks for the 4 captive door screws (x-axis M4 inserts)
    for sy in (-1, 1):
        for z0 in (P.BASE_T + P.LEDGE_T, P.CAB_Z1 + 0.3 + 0.0):
            pass
    cb = []
    for sy in (-1, 1):
        for zc in (19.0, 212.0):
            y0, y1 = (sy * 80.0, sy * 90.0) if sy > 0 else (sy * 90.0, sy * 80.0)
            blk = box(P.DOOR_X - 12.0, P.DOOR_X, y0, y1, zc - 5, zc + 5)
            H = H.union(blk)
    for sy in (-1, 1):
        for zc in (19.0, 212.0):
            H = H.cut(cylx(P.DOOR_X - 9.0, P.DOOR_X + 0.1, sy * 85.0, zc, P.INSERT_M4_HOLE))
    # base screws: inserts in the ledge
    for sy in (-1, 1):
        for x in (-90.0, -30.0, 30.0, 90.0):
            H = H.cut(cylz(x, sy * 85.0, P.BASE_T + 2.0, P.BASE_T + P.LEDGE_T + 0.1, P.INSERT_M4_HOLE))
    for sy in (-1, 1):
        H = H.cut(cylz(-112.0, sy * 40.0, P.BASE_T + 2.0, P.BASE_T + P.LEDGE_T + 0.1, P.INSERT_M4_HOLE))
    # roof bulkhead boss (sleeve keeps foam away from the fitting) + hole
    rx, ry = P.R_XY
    H = H.union(cylz(rx, ry, P.CAB_Z1 + 0.3, P.Z_TOP - P.WALL + 0.01, 24.0).cut(cylz(rx, ry, P.CAB_Z1, P.Z_TOP, 20.0)))
    H = H.cut(cylz(rx, ry, P.Z_TOP - P.WALL - 0.1, P.Z_TOP + 0.1, 14.4))
    # M12 connector boss + hole (rear wall)
    my, mz = P.M12_POS
    H = H.union(cylx(xr - 0.01, P.X0 + P.INNER - 0.3, my, mz, 28.0).cut(cylx(xr, P.X0 + P.INNER, my, mz, 24.0)))
    H = H.cut(cylx(P.X0 - 0.1, xr + 0.1, my, mz, 16.2))
    # vent ducts
    for side in (-1, 1):
        d, slots = make_duct(side)
        H = H.union(d)
        H = H.cut(slots)
    return H


def door():
    xr = P.X1 - P.WALL                               # 117 (inner face of the door plate)
    D = box(P.DOOR_X, P.X1, P.Y0, P.Y1, P.BASE_T, P.Z_TOP)
    D = D.cut(box(P.DOOR_X - 1, xr, -(P.Y1 - P.WALL), P.Y1 - P.WALL, P.BASE_T + 3.0, P.Z_TOP - 3.0))
    for sy in (-1, 1):
        for zc in (19.0, 212.0):
            y0, y1 = (sy * 80.0, sy * 90.0) if sy > 0 else (sy * 90.0, sy * 80.0)
            D = D.union(box(P.DOOR_X, xr + 0.01, y0, y1, zc - 5, zc + 5))
    for sy in (-1, 1):
        for zc in (19.0, 212.0):
            D = D.cut(cylx(P.DOOR_X - 0.1, P.X1 + 0.1, sy * 85.0, zc, 4.5))
    return D


def base_plate():
    B = box(P.X0, P.X1, P.Y0, P.Y1, 0, P.BASE_T)
    for (x, y) in P.AMR_SLOT_XY:                     # 4 x M6 slots, long along x
        B = B.cut(cq.Workplane('XY').slot2D(P.AMR_SLOT_LEN + P.AMR_SLOT_W, P.AMR_SLOT_W, 0).extrude(P.BASE_T + 2).translate((x, y, -1)))
    for (x, y) in P.PUMP_FOOT_XY:                    # pump mount bosses + M6 insert holes
        B = B.union(cylz(x, y, P.BASE_T - 0.01, P.BASE_T + P.BOSS_H, 16.0))
        B = B.cut(cylz(x, y, P.BASE_T + P.BOSS_H - 9.0, P.BASE_T + P.BOSS_H + 0.1, P.INSERT_M6_HOLE))
    ex, ey = P.EXH_XY
    B = B.cut(cylz(ex, ey, -1, P.BASE_T + 1, 12.2))
    # housing screws (countersunk from below)
    pos = [(x, sy * 85.0) for sy in (-1, 1) for x in (-90.0, -30.0, 30.0, 90.0)] + [(-112.0, sy * 40.0) for sy in (-1, 1)]
    for (x, y) in pos:
        B = B.cut(cylz(x, y, -1, P.BASE_T + 1, P.M4_CSK[0]))
        B = B.cut(cq.Workplane('XY').circle(P.M4_CSK[1] / 2).workplane(offset=2.0).circle(P.M4_CSK[0] / 2).loft().translate((x, y, -0.0)).translate((0, 0, 0)))
    # deck legs (printed standing) with M4 inserts at the top
    for (xa, xb) in P.LEG_X:
        for sy in (-1, 1):
            y0, y1 = (sy * P.LEG_Y[0], sy * P.LEG_Y[1]) if sy > 0 else (sy * P.LEG_Y[1], sy * P.LEG_Y[0])
            B = B.union(box(xa, xb, y0, y1, P.BASE_T - 0.01, P.RACK_Z))
            B = B.cut(cylz((xa + xb) / 2, (y0 + y1) / 2, P.RACK_Z - 8.0, P.RACK_Z + 0.1, P.INSERT_M4_HOLE))
    # HEPA cradle (concave seat for the d56 body)
    hx, hy, hz = P.HEPA_C
    for (xa, xb) in ((hx - P.HEPA[1] / 2, hx - P.HEPA[1] / 2 + 4.0), (hx + P.HEPA[1] / 2 - 4.0, hx + P.HEPA[1] / 2)):
        blk = box(xa, xb, hy - P.HEPA[0] / 2, hy + P.HEPA[0] / 2, P.BASE_T - 0.01, hz)
        B = B.union(blk.cut(cylx(xa - 1, xb + 1, hy, hz, P.HEPA[0])))
    return B


def deck_pads():
    """(name, x0, x1, y0, y1, z_item_bottom): printed saddles/pads on the deck"""
    Z = P.Z_LINE
    out = []
    dx, dy, dz = P.DUST_C
    out += [('dust_a', dx - 22, dx - 14, dy - 6, dy + 6, dz - P.DUST[0] / 2), ('dust_b', dx + 14, dx + 22, dy - 6, dy + 6, dz - P.DUST[0] / 2)]
    mx, my, mz = P.MUFFLER_C
    out += [('muff_a', mx - 14, mx - 6, my - 6, my + 6, mz - P.MUFFLER[0] / 2), ('muff_b', mx + 6, mx + 14, my - 6, my + 6, mz - P.MUFFLER[0] / 2)]
    x, y, z = P.MANI_C
    out += [('mani', x - 4, x + 4, y - 11, y + 11, z - P.MANIFOLD[2] / 2)]
    out += [('valve', P.VALVE_X[0] + 1, P.VALVE_X[1] - 1, P.VALVE_Y[0] + 1, P.VALVE_Y[1] - 1, P.VALVE_Z0)]
    sx, sy_, sz = P.SWITCH_C
    out += [('switch', sx - 10, sx + 10, sy_ - 12, sy_ + 12, sz - P.SW_ENV[2] / 2)]
    tx, ty = P.TERM_C
    out += [('term', tx - 22, tx + 22, ty - 5, ty + 5, P.DECK_TOP + 3.0)]
    return out


def deck():
    D = box(P.DECK_X[0], P.DECK_X[1], -P.DECK_Y, P.DECK_Y, P.RACK_Z, P.DECK_TOP)
    D = D.cut(cylz(P.EXH_XY[0], P.EXH_XY[1], P.RACK_Z - 1, P.DECK_TOP + 1, 24.0))      # exhaust tube passes through
    D = D.cut(box(-10.0, -2.0, -25.0, 25.0, P.RACK_Z - 1, P.DECK_TOP + 1))               # wire pass-through
    for (xa, xb) in P.LEG_X:
        for sy in (-1, 1):
            D = D.cut(cylz((xa + xb) / 2, sy * (P.LEG_Y[0] + P.LEG_Y[1]) / 2, P.RACK_Z - 1, P.DECK_TOP + 1, P.M4_CSK[0]))
    for (n, x0, x1, y0, y1, zb) in deck_pads():
        D = D.union(box(x0, x1, y0, y1, P.DECK_TOP - 0.01, zb))
    return D


# ----------------------------------------------------------------------------- hardware
def pump():
    L = lambda x0, x1, y0, y1, z0, z1: box(*[a for a in (P.PUMP_X0 + x0, P.PUMP_X0 + x1)], y0, y1, P.PUMP_Z0 + z0, P.PUMP_Z0 + z1)
    t = P.PUMP_FOOT_PLATE
    W = L(t[0], t[1], -P.PUMP_W / 2, P.PUMP_W / 2, 0, t[2])
    b = P.PUMP_BRIDGE
    W = W.union(L(b[0], b[1], -b[2], b[2], t[2] - 0.01, b[3]))
    hl, hw, hz = P.PUMP_HEAD
    W = W.union(L(0, hl, -hw, hw, t[2] - 0.01, hz))
    W = W.union(L(0, hl, -hw, hw, hz - 0.01, P.PUMP_H))
    md, mzc = P.PUMP_MOTOR
    W = W.union(cylx(P.PUMP_X0 + hl - 0.01, P.PUMP_X0 + P.PUMP_L, 0, P.PUMP_Z0 + mzc, md))
    for (x, y) in P.PUMP_FOOT_XY:
        W = W.cut(cylz(x, y, P.PUMP_Z0 - 1, P.PUMP_Z0 + t[2] + 1, P.PUMP_FOOT_D))
    return W


def mounts():
    W = None
    for (x, y) in P.PUMP_FOOT_XY:
        m = cylz(x, y, P.BASE_T + P.BOSS_H, P.PUMP_Z0, P.MOUNT_D).union(cylz(x, y, P.BASE_T + P.BOSS_H - 9.0 + 1.0, P.BASE_T + P.BOSS_H, 6.0)).union(cylz(x, y, P.PUMP_Z0, P.PUMP_Z0 + 10.0, 6.0))
        W = m if W is None else W.union(m)
    return W


def elbow(port):
    e = P.ELBOW
    x, y, z = port
    return U(cylz(x, y, z, z + e[1] + e[0] / 2, e[0]), cylx(x, x + e[2], y, z + e[1], e[0]))


def hardware(include_tubes=True):
    Z = P.Z_LINE
    H = {}
    H['pump'] = pump()
    H['mounts'] = mounts()
    H['elbow_in'] = elbow(P.PORT_IN)
    H['elbow_out'] = elbow(P.PORT_OUT)
    mx, my, mz = P.MUFFLER_C
    H['muffler'] = cylx(mx - P.MUFFLER[1] / 2, mx + P.MUFFLER[1] / 2, my, mz, P.MUFFLER[0])
    hx, hy, hz = P.HEPA_C
    H['hepa_filter'] = cylx(hx - P.HEPA[1] / 2, hx + P.HEPA[1] / 2, hy, hz, P.HEPA[0])
    ex, ey = P.EXH_XY
    H['exhaust_fitting'] = U(cylz(ex, ey, -8.0, 0.0, 10.0), cylz(ex, ey, -0.01, P.BASE_T + 0.01, 12.0), cylz(ex, ey, P.BASE_T, P.BASE_T + 4.0, 14.0), cylz(ex, ey, P.BASE_T + 4.0, 18.0, 12.0))
    dx, dy, dz = P.DUST_C
    H['dust_filter'] = cylx(dx - P.DUST[1] / 2, dx + P.DUST[1] / 2, dy, dz, P.DUST[0])
    ml, mw, mh = P.MANIFOLD
    H['manifold'] = box(P.MANI_C[0] - ml / 2, P.MANI_C[0] + ml / 2, P.MANI_C[1] - mw / 2, P.MANI_C[1] + mw / 2, P.MANI_C[2] - mh / 2, P.MANI_C[2] + mh / 2)
    rx, ry = P.R_XY
    H['bulkhead_fitting'] = U(cylz(rx, ry, P.R_PORT_Z, P.CAB_Z1 + 0.3 - 1.0, 14.0), cylz(rx, ry, P.CAB_Z1 + 0.3 - 1.0, P.Z_TOP + 4.0, 12.0), cylz(rx, ry, P.Z_TOP - P.WALL - 4.0, P.Z_TOP - P.WALL, 18.0),
                              cylz(rx, ry, P.Z_TOP, P.Z_TOP + 4.0, 16.0), cylz(rx, ry, P.Z_TOP + 4.0, P.Z_TOP + 44.0, 8.0))
    sx, sy, sz = P.SWITCH_C
    H['vacuum_switch'] = box(sx - P.SW_ENV[0] / 2, sx + P.SW_ENV[0] / 2, sy - P.SW_ENV[1] / 2, sy + P.SW_ENV[1] / 2, sz - P.SW_ENV[2] / 2, sz + P.SW_ENV[2] / 2)
    vx0, vx1 = P.VALVE_X
    vxc = (vx0 + vx1) / 2
    H['vent_valve'] = U(box(vx0, vx1, P.VALVE_Y[0], P.VALVE_Y[1], P.VALVE_Z0, P.VALVE_Z0 + P.VALVE_ENV[2]),
                        cyly(P.VALVE_Y[0] - P.VALVE_FILTER[1], P.VALVE_Y[0] + 0.01, vxc, Z, P.VALVE_FILTER[0]))
    tx, ty = P.TERM_C
    H['terminal_blocks'] = box(tx - P.TERMINAL[0] / 2, tx + P.TERMINAL[0] / 2, ty - P.TERMINAL[1] / 2, ty + P.TERMINAL[1] / 2, P.DECK_TOP + 3.0, P.DECK_TOP + 3.0 + P.TERMINAL[2])
    fx, fz = P.FAN_XZ
    H['fan'] = box(fx - P.FAN[0] / 2, fx + P.FAN[0] / 2, -P.CAB_Y, -P.CAB_Y + P.FAN[2], fz - P.FAN[1] / 2, fz + P.FAN[1] / 2)
    my, mz = P.M12_POS
    H['m12_connector'] = U(box(P.X0 - P.M12_ENV[2], P.X0, my - 10, my + 10, mz - 10, mz + 10), cylx(P.X0 - 0.01, P.X0 + P.INNER + 2.0, my, mz, 16.0))
    for i, (x, y) in enumerate(P.AMR_SLOT_XY):
        H['spacer_%d' % i] = cylz(x, y, -P.SPACER_H, 0.0, P.SPACER_D)
        w = P.AMR_WASHER
        H['m6_washer_head_%d' % i] = U(box(x - w[0] / 2, x + w[0] / 2, y - w[1] / 2, y + w[1] / 2, P.BASE_T, P.BASE_T + w[2]), cylz(x, y, P.BASE_T + w[2], P.BASE_T + w[2] + 6.0, 10.0))
    return H


def fix_mass(name):
    return None


# ----------------------------------------------------------------------------- foam + film
def foam_film(HOUS, DOOR):
    xr = P.X0 + P.WALL
    keep = [duct_env(-1), duct_env(1)]
    rx, ry = P.R_XY
    keep.append(cylz(rx, ry, P.CAB_Z1 - 1, P.Z_TOP, 24.0))
    keep.append(cylx(P.X0, xr + P.FOAM_T + 1, P.M12_POS[0], P.M12_POS[1], 28.0))
    kp = keep[0]
    for k in keep[1:]:
        kp = kp.union(k)
    zl, zt = P.BASE_T + P.LEDGE_T, P.Z_TOP - P.WALL
    pieces = {}
    f = P.FOAM_T
    pieces['foam_roof'] = (box(xr, P.DOOR_X, -(P.Y1 - P.WALL), P.Y1 - P.WALL, zt - f, zt), HOUS)
    pieces['foam_rear'] = (box(xr, xr + f, -(P.Y1 - P.WALL), P.Y1 - P.WALL, zl, zt - f), HOUS)
    pieces['foam_left'] = (box(xr + f, P.DOOR_X, -(P.Y1 - P.WALL), -(P.Y1 - P.WALL) + f, zl, zt - f), HOUS)
    pieces['foam_right'] = (box(xr + f, P.DOOR_X, P.Y1 - P.WALL - f, P.Y1 - P.WALL, zl, zt - f), HOUS)
    pieces['foam_door'] = (box(P.DOOR_X + P.FILM_T, P.X1 - P.WALL, -(P.Y1 - P.WALL), P.Y1 - P.WALL, P.BASE_T + 3.0, P.Z_TOP - 3.0), DOOR)
    ft = P.FILM_T
    films = {}
    films['film_roof'] = box(xr, P.DOOR_X, -(P.Y1 - P.WALL), P.Y1 - P.WALL, zt - f - ft, zt - f)
    films['film_rear'] = box(xr + f, xr + f + ft, -(P.Y1 - P.WALL), P.Y1 - P.WALL, zl, zt - f - ft)
    films['film_left'] = box(xr + f + ft, P.DOOR_X, -(P.Y1 - P.WALL) + f, -(P.Y1 - P.WALL) + f + ft, zl, zt - f - ft)
    films['film_right'] = box(xr + f + ft, P.DOOR_X, P.Y1 - P.WALL - f - ft, P.Y1 - P.WALL - f, zl, zt - f - ft)
    films['film_door'] = box(P.DOOR_X, P.DOOR_X + ft, -(P.Y1 - P.WALL), P.Y1 - P.WALL, P.BASE_T + 3.0, P.Z_TOP - 3.0)
    out = {}
    for n, (w, host) in pieces.items():
        w = w.cut(host).cut(kp)
        out[n] = w
    for n, w in films.items():
        host = DOOR if n == 'film_door' else HOUS
        out[n] = w.cut(host).cut(kp)
    return out


# ----------------------------------------------------------------------------- assembly
MATERIAL = dict(base_plate='PETG', housing='PETG', door='PETG', deck='PETG')
COLORS = dict(base_plate=(0.88, 0.88, 0.9), housing=(0.93, 0.93, 0.95), door=(0.85, 0.9, 0.95), deck=(0.8, 0.82, 0.88),
              pump=(0.3, 0.45, 0.7), mounts=(0.1, 0.1, 0.1), fan=(0.2, 0.2, 0.2), muffler=(0.7, 0.55, 0.3), hepa_filter=(0.95, 0.95, 0.85),
              dust_filter=(0.4, 0.7, 0.8), vent_valve=(0.8, 0.5, 0.2), vacuum_switch=(0.2, 0.6, 0.3), manifold=(0.7, 0.7, 0.72),
              terminal_blocks=(0.9, 0.6, 0.1), m12_connector=(0.15, 0.15, 0.15), bulkhead_fitting=(0.15, 0.15, 0.15), exhaust_fitting=(0.15, 0.15, 0.15))
PRINTED = ['base_plate', 'housing', 'door', 'deck']


def solids(with_tubes=True):
    S = {}
    Hs, Ds = housing(), door()
    S['base_plate'] = dict(shape=shp(base_plate()), kind='printed')
    S['housing'] = dict(shape=shp(Hs), kind='printed')
    S['door'] = dict(shape=shp(Ds), kind='printed')
    S['deck'] = dict(shape=shp(deck()), kind='printed')
    for n, w in hardware().items():
        S[n] = dict(shape=shp(w), kind='hardware')
    if with_tubes:
        for n, (pts, od, ab) in routes().items():
            sol, margin = tube_solid(pts, od)
            S[n] = dict(shape=sol, kind='tube', ends=ab, margin=margin)
    for n, w in foam_film(Hs, Ds).items():
        S[n] = dict(shape=shp(w), kind='foam' if n.startswith('foam') else 'film')
    for n, d in S.items():
        d['color'] = COLORS.get(n, (0.55, 0.55, 0.6) if d['kind'] == 'hardware' else (0.15, 0.15, 0.17) if d['kind'] == 'tube' else (0.95, 0.85, 0.35) if d['kind'] == 'foam' else (0.5, 0.7, 0.9))
        if n.startswith('spacer') or n.startswith('m6_'):
            d['color'] = (0.6, 0.6, 0.62)
    return S


# ----------------------------------------------------------------------------- mass
HW_MASS = dict(pump=P.PUMP_MASS, mounts=0.06, elbow_in=0.012, elbow_out=0.012, muffler=P.MUFFLER_MASS, hepa_filter=P.HEPA_MASS, exhaust_fitting=0.012,
               dust_filter=P.DUST_MASS, manifold=P.MANIFOLD_MASS, bulkhead_fitting=0.025, vacuum_switch=P.SW_MASS, vent_valve=P.VALVE_MASS,
               terminal_blocks=P.TERMINAL_MASS, fan=P.FAN_MASS, m12_connector=P.M12_MASS)


def est_mass_printed(shape):
    """1.6 mm shell solid + 25 % infill in the rest (PETG)"""
    V, A_ = shape.Volume(), shape.Area()
    vs = min(V, 1.6 * A_)
    return P.RHO['PETG'] * (vs + 0.25 * (V - vs))


def masses(S):
    m = {}
    for n, d in S.items():
        k = d['kind']
        if k == 'printed':
            m[n] = est_mass_printed(d['shape'])
        elif k == 'foam':
            m[n] = P.RHO_FOAM * d['shape'].Volume()
        elif k == 'film':
            m[n] = P.RHO_FILM * d['shape'].Volume()
        elif k == 'tube':
            m[n] = 1.0e-6 * d['shape'].Volume() * 0.9          # PU tube, 0.9 g/cm3, solid-model volume is an upper bound (ASSUMED)
        elif n.startswith('spacer'):
            m[n] = P.RHO['alu'] * d['shape'].Volume()
        elif n.startswith('m6_'):
            m[n] = 7.85e-6 * d['shape'].Volume()
        else:
            m[n] = HW_MASS[n]
    m['misc_fasteners_cables'] = P.FIT_MASS
    return m
