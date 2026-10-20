"""Shaded PNG renders (matplotlib painter's algorithm).  python render.py  -> renders/*.png (needs data from build.py not required: builds parts itself)."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import trimesh
import params as P
import parts as pa
from meshlib import to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'printed': (0.93, 0.55, 0.12), 'purchased': (0.55, 0.62, 0.74)}
SPECIAL = {'collector': (0.8, 0.8, 0.85), 'camera_gemini305': (0.15, 0.65, 0.3), 'scanner_se4710': (0.1, 0.1, 0.1), 'decoder_pl3307': (0.1, 0.5, 0.2), 'box': (0.85, 0.8, 0.65), 'box_target': (0.75, 0.6, 0.35)}
for n in ('pad_L', 'pad_M', 'pad_R'):
    SPECIAL[n] = (0.2, 0.2, 0.25)
for n in ('tube_L', 'tube_M', 'tube_R', 'trunk_tube'):
    SPECIAL[n] = (0.2, 0.5, 0.9)


def collect():
    out = []
    for d in pa.build_parts():
        out.append((d['name'], d['kind'], to_mesh(d['shape'], tol=0.15, ang=0.25)))
    return out


def box_meshes(t=16.0, h=230.0, l=250.0):
    """target box upright: spine face (t x h) at z = LIP_Z, centre x=0; neighbours both sides, 15 mm proud (toward the tool), zero gap"""
    Lz = P.LIP_Z
    tgt = trimesh.creation.box(extents=[h, t, l]); tgt.apply_translation([0, 0, Lz + l / 2])
    out = [('box_target', 'box', tgt)]
    for s in (-1, 1):
        n = trimesh.creation.box(extents=[h, t, l + 15.0]); n.apply_translation([0, s * t, Lz + (l + 15.0) / 2 - 15.0])
        out.append(('box', 'box', n))
    return out


def view_matrix(el, az):
    el, az = np.radians(el), np.radians(az)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    r = np.cross(up, d); r /= np.linalg.norm(r)
    return r, np.cross(d, r), d


def render(meshes, el, az, fn, title='', lim=None, size=(8.0, 8.0), dpi=150, up=None):
    r, u, d = view_matrix(el, az)
    if up is not None:  # custom up vector
        u = np.asarray(up, float) - d * np.dot(up, d); u /= np.linalg.norm(u); r = np.cross(u, d)
    light = np.array([0.4, -0.3, 0.9]); light /= np.linalg.norm(light)
    polys, cols, depth = [], [], []
    for name, kind, m in meshes:
        tri = m.vertices[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1; n = n / ln[:, None]
        vis = (n @ d) > -0.15
        sh = 0.45 + 0.55 * np.abs(n @ (0.5 * light + 0.5 * d))
        base = np.array(SPECIAL.get(name, COL.get(kind, (0.6, 0.6, 0.6))))
        xy = np.stack([tri @ r, tri @ u], axis=-1); z = (tri @ d).mean(axis=1)
        for i in np.where(vis)[0]:
            polys.append(xy[i]); cols.append(np.clip(base * sh[i], 0, 1)); depth.append(z[i])
    order = np.argsort(depth)
    fig, ax = plt.subplots(figsize=size)
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors='none', linewidths=0))
    allv = np.vstack([m.vertices for _, _, m in meshes]); p = np.stack([allv @ r, allv @ u], axis=1)
    lo, hi = (p.min(0) - 8, p.max(0) + 8) if lim is None else (np.array(lim[0]), np.array(lim[1]))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10)
    fig.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0.05); plt.close(fig)


def shrink(fn, max_kb=300):
    from PIL import Image
    im = Image.open(fn).convert('RGB')
    for colors in (256, 128, 64):
        im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(fn, optimize=True)
        if os.path.getsize(fn) <= max_kb * 1024:
            return


if __name__ == '__main__':
    out = os.path.join(HERE, 'renders'); os.makedirs(out, exist_ok=True)
    only = sys.argv[1:]
    ms = collect()
    tool = ms
    views = {'iso': (25, -50, 'iso (from +x, -y, +z)'), 'front': (0, -90, 'front: looking along +y (scanner side) - link6 +x to the right'),
             'side': (0, 0, 'side: looking along -x (+y camera side to the right, -y scanner to the left)'), 'top': (89, -90, 'top: looking down -z (flange side of the pad row)')}
    for nm, (el, az, ti) in views.items():
        fn = '%s/assembly_%s.png' % (out, nm); render(ms, el, az, fn, ti, size=(8, 8), dpi=150); shrink(fn); print(fn, os.path.getsize(fn) // 1024, 'kB')
    # pressed on a 16 x 230 x 250 box: lip plane = spine plane, two neighbours 15 mm proud
    sc = tool + box_meshes()
    fn = '%s/pressed_on_box_side.png' % out; render(sc, 0, 0, fn, 'pads pressed on a 16 x 230 x 250 spine, neighbours 15 mm proud (side view along -x)', size=(9, 7), dpi=150); shrink(fn)
    fn = '%s/pressed_on_box_iso.png' % out; render(sc, 22, -60, fn, 'pads pressed on a 16 x 230 x 250 spine, neighbours 15 mm proud (iso)', size=(9, 8), dpi=150); shrink(fn)
    fn = '%s/pressed_on_box_front.png' % out; render(sc, 0, -90, fn, 'looking along +y: spine (230 mm high along x) with the pad row', size=(9, 7), dpi=150); shrink(fn)
