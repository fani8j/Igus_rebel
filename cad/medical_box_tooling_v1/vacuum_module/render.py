"""Shaded PNG renders (matplotlib painter's algorithm, flat shading): renders/{iso,exploded,section_y,service_open}.png.
Run: nice -n 19 "$PY" render.py
"""
import os
import numpy as np
import trimesh
import math
import params as P
import parts as K

HERE = P.HERE
UP = np.array([0, 0, 1.0])


def basis(d, up=UP):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    r = np.cross(d, up); r /= np.linalg.norm(r)
    u = np.cross(r, d)
    return r, u, d


def draw(items, d, fn, title='', px=1700, up=UP, lim=None, max_edge=None):
    """z-buffer rasteriser (orthographic, flat shading, both sides lit, part-id silhouette lines); items = (name, mesh, colour, _)"""
    from PIL import Image, ImageDraw
    r, u, d = basis(d, up)
    light = np.array([-0.35, 0.4, -0.85]); light /= np.linalg.norm(light)
    allv = np.vstack([m.vertices for _, m, _, _ in items])
    xy_all = np.stack([allv @ r, allv @ u], 1)
    lo, hi = xy_all.min(0), xy_all.max(0)
    if lim is not None:
        lo, hi = np.array(lim[0], float), np.array(lim[1], float)
    c0 = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.04
    sc = px / (2 * half)
    W = H = px
    zb = np.full((H, W), np.inf, np.float32)
    idb = np.full((H, W), -1, np.int32)
    img = np.ones((H, W, 3), np.float32)
    for k, (name, m, c, _) in enumerate(items):
        tri = m.vertices[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1
        n = n / ln[:, None]
        sh = 0.42 + 0.58 * np.clip(0.55 * np.abs(n @ d) + 0.45 * np.abs(n @ light), 0, 1)
        col = np.clip(np.array(c)[None, :] * sh[:, None], 0, 1)
        sx = (tri @ r - c0[0]) * sc + W / 2
        sy = H / 2 - (tri @ u - c0[1]) * sc
        sz = tri @ d
        for i in range(len(tri)):
            x, y, z = sx[i], sy[i], sz[i]
            x0, x1 = int(max(0, math.floor(x.min()))), int(min(W - 1, math.ceil(x.max())))
            y0, y1 = int(max(0, math.floor(y.min()))), int(min(H - 1, math.ceil(y.max())))
            if x1 < x0 or y1 < y0:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
            l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
            if not inside.any():
                continue
            zz = (l0 * z[0] + l1 * z[1] + l2 * z[2]).astype(np.float32)
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zz < sub)
            sub[upd] = zz[upd]
            idb[y0:y1 + 1, x0:x1 + 1][upd] = k
            img[y0:y1 + 1, x0:x1 + 1][upd] = col[i]
    edge = np.zeros((H, W), bool)
    edge[:-1, :] |= idb[:-1, :] != idb[1:, :]
    edge[:, :-1] |= idb[:, :-1] != idb[:, 1:]
    img[edge] *= 0.45
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((px // 2, px // 2), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    if title:
        dr.text((10, 8), title, fill=(0, 0, 0))
    im.convert('P', palette=Image.ADAPTIVE, colors=128).save(fn, optimize=True)


def main():
    out = os.path.join(HERE, 'renders'); os.makedirs(out, exist_ok=True)
    S = K.solids()
    M = {n: K.tess(d['shape'], 0.3, 0.5) for n, d in S.items()}
    col = {n: d['color'] for n, d in S.items()}
    soft = lambda n: n.startswith(('foam', 'film'))
    # ---- iso (exterior: door side + -y side + top)
    items = [(n, m, col[n], 1) for n, m in M.items() if not soft(n) and not n.startswith('tube')]
    dview = np.array([-0.62, 0.7, -0.55])
    draw(items, dview, os.path.join(out, 'iso.png'), title='Vacuum module on the AMR: service door at +x (toward the viewer), arm tube up on the roof, exhaust nozzle down')
    print('iso done', flush=True)
    # ---- service view: door removed, foam hidden under film, interior visible
    items = [(n, m, col[n], 1) for n, m in M.items() if n not in ('door', 'foam_door', 'film_door') and not n.startswith(('foam', 'film'))]
    draw(items, np.array([-0.75, 0.45, -0.55]), os.path.join(out, 'service_open.png'), title='Door removed: HEPA and dust filter reachable from the service side, pump stays on its mounts')
    print('service done', flush=True)
    # ---- exploded
    HZ, DZ = 280.0, 110.0
    off = {}
    for n in ('housing', 'foam_roof', 'foam_rear', 'foam_left', 'foam_right', 'film_roof', 'film_rear', 'film_left', 'film_right', 'bulkhead_fitting', 'fan'):
        off[n] = (0, 0, HZ)
    off['m12_connector'] = (-40, 0, HZ)
    off['door'] = (110, 0, 0); off['foam_door'] = (125, 0, 0); off['film_door'] = (118, 0, 0)
    for n in ['deck', 'dust_filter', 'muffler', 'manifold', 'vent_valve', 'vacuum_switch', 'terminal_blocks'] + [n for n in S if n.startswith('tube_vac') or n.startswith('tube_exhaust_muffler')]:
        off[n] = (0, 0, DZ)
    items = []
    for n, m in M.items():
        mm = m.copy()
        o = off.get(n, (0, 0, 0))
        if o != (0, 0, 0):
            mm.apply_translation(o)
        for k in ('spacer', ):
            pass
        if n.startswith('spacer'):
            mm.apply_translation((0, 0, -25))
        items.append((n, mm, col[n], 1))
    draw(items, np.array([-0.62, 0.7, -0.55]), os.path.join(out, 'exploded.png'), title='Exploded: base plate + pump on mounts, filter deck, housing with foam and film, door')
    print('exploded done', flush=True)
    # ---- section y = 0 (keep y <= 0)
    items = []
    for n, m in M.items():
        if n.startswith('film'):
            continue
        b = m.bounds
        if b[0][1] > 5 and not n.startswith(('pump', 'mounts')):
            continue
        if b[1][1] > 0 and (n in ('housing', 'door', 'base_plate', 'foam_roof', 'foam_rear', 'foam_left', 'foam_right', 'foam_door', 'deck')):
            mm = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, -1, 0], plane_origin=[0, 0, 0], cap=False)
            if mm is None or len(mm.faces) == 0:
                continue
            m = mm
        items.append((n, m, col[n], 1))
    draw(items, np.array([0.2, -1.0, -0.12]), os.path.join(out, 'section.png'), title='Section at y = 0 seen from +y (+x = service door is on the LEFT): 10 mm foam (yellow) behind the film inside the printed shell, pump on mounts, filter deck above the motor')
    print('section done', flush=True)


if __name__ == '__main__':
    main()
