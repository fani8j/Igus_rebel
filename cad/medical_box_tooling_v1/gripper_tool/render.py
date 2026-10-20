"""Shaded PNG renders (matplotlib painter's algorithm, flat shading, tool pointing DOWN: link6 +z is drawn downwards).
renders/assembly_{iso,front,side,top}.png (gripper open 56 mm) and renders/grip_16x230x250_{iso,closeup}.png (closed on a 16 x 230 x 250 box,
gripped from above at s = 15..70 behind the spine, neighbour boxes drawn with their spines 100 mm further back).
Run: nice -n 19 "$PY" render.py
"""
import os, sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import params as P
import parts as K

HERE = P.HERE
MAX_EDGE = 5.0
COL = {'top_stop': (0.85, 0.2, 0.2), 'flange_disc': (0.93, 0.55, 0.12), 'gripper_plate': (0.95, 0.65, 0.2), 'smc_body': (0.72, 0.74, 0.78), 'smc_jaw_left': (0.5, 0.52, 0.58), 'smc_jaw_right': (0.5, 0.52, 0.58),
       'finger_left': (0.15, 0.55, 0.85), 'finger_right': (0.25, 0.65, 0.9), 'pad_left': (0.08, 0.08, 0.08), 'pad_right': (0.08, 0.08, 0.08),
       'hardware': (0.45, 0.47, 0.52), 'box': (0.85, 0.72, 0.5), 'neighbour': (0.78, 0.82, 0.6)}


def basis(d, up):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    r = np.cross(up, d); r /= np.linalg.norm(r)
    u = np.cross(d, r)
    return r, u, d


def draw(items, d, up, fn, lim=None, title='', size=12.0, dpi=100):
    r, u, d = basis(d, up)
    light = np.array([0.35, -0.4, -0.85]); light /= np.linalg.norm(light)
    polys, cols, depth = [], [], []
    allp = []
    for name, m, c in items:
        v_, f_ = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=MAX_EDGE)
        m = trimesh.Trimesh(vertices=v_, faces=f_, process=False)
        tri = m.vertices[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1
        n = n / ln[:, None]
        vis = (n @ d) > -0.05
        sh = 0.4 + 0.6 * np.clip(0.55 * np.abs(n @ d) + 0.45 * np.abs(n @ light), 0, 1)
        xy = np.stack([tri @ r, tri @ u], axis=-1)
        z = (tri @ d).mean(axis=1)
        idx = np.where(vis)[0]
        for i in idx:
            polys.append(xy[i]); cols.append(np.clip(np.array(c) * sh[i], 0, 1)); depth.append(z[i])
        allp.append(np.stack([m.vertices @ r, m.vertices @ u], 1))
    order = np.argsort(depth)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors=[cols[i] for i in order], linewidths=0.15))
    p = np.vstack(allp)
    lo, hi = p.min(0), p.max(0)
    if lim is not None:
        lo, hi = np.array(lim[0], float), np.array(lim[1], float)
    c = (lo + hi) / 2; half = (hi - lo).max() / 2 * 1.04
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    from PIL import Image
    im = Image.open(fn).convert('RGB')
    im.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(fn, optimize=True)


def tool_items(o):
    S = K.solids(o, with_hw=True)
    items = []
    for n, d in S.items():
        c = COL.get(n, COL['hardware'])
        items.append((n, K.tess(d['shape'], 0.15, 0.3), c))
    return items


def box_mesh(x0, x1, y0, y1, z0, z1):
    return trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0], transform=trimesh.transformations.translation_matrix([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]))


def main():
    out = os.path.join(HERE, 'renders')
    os.makedirs(out, exist_ok=True)
    up = np.array([0, 0, -1.0])                      # link6 +z drawn downwards
    items = tool_items(P.JAW_STROKE)
    views = {'iso': ((0.75, -0.9, -0.62), up, 'Right gripper tool, open 56 mm (link6 +z downwards)'),
             'front': ((0, -1.0, 0), up, 'front: looking along +Y (stroke axis X horizontal)'),
             'side': ((1.0, 0, 0), up, 'side: looking along -X'),
             'top': ((0, 0, -1.0), np.array([0, 1.0, 0]), 'top: looking along +Z at the flange side')}
    for nm, (d, u_, t) in views.items():
        draw(items, d, u_, os.path.join(out, 'assembly_%s.png' % nm), title=t)
        print('assembly_%s done' % nm, flush=True)
    # stop detail: fingers open, from the -y/+x side, cropped to the stop and the jaws
    zc = (P.Z_J + P.Z_STOP) / 2
    pts = np.array([[-30, -35, P.BOLT_Z - 15], [30, 35, P.Z_STOP + 40]])
    dd = (0.55, 0.8, -0.35)
    r_, u_, d_ = basis(dd, up)
    cube = np.array([[x, y, z] for x in (-30, 30) for y in (-35, 35) for z in (P.BOLT_Z - 15, P.Z_STOP + 40)])
    pr = np.stack([cube @ r_, cube @ u_], 1)
    draw(items, dd, up, os.path.join(out, 'stop_detail.png'), lim=(pr.min(0), pr.max(0)), title='Fixed centre top-stop (red): crossbar x +-1.5 (3 mm) between the fingers, yoke on the body sides with 2 x M5')
    draw(items, (0.9, -0.45, -0.1), up, os.path.join(out, 'stop_detail_side.png'), lim=(pr.min(0) - 5, pr.max(0) + 5), title='Stop seen from the +x side (fingers open)')
    # ---- grip on a 16 x 230 x 250 box.  y = s - 42.5 (spine at s = 0 -> y = -42.5), box length 250 into the shelf, neighbours' spines at s = 100
    items = tool_items(4.0)
    t, h, l = 16.0, 230.0, 250.0
    s0 = P.S_PAD_CENTRE
    ztop = P.Z_J + P.Z_REG
    box = box_mesh(-t / 2, t / 2, -s0, -s0 + l, ztop, ztop + h)
    nb = [box_mesh(t / 2, 1.5 * t, 100 - s0, 100 - s0 + l, ztop, ztop + h), box_mesh(-1.5 * t, -t / 2, 100 - s0, 100 - s0 + l, ztop, ztop + h)]
    items2 = items + [('box', box, COL['box'])] + [('neighbour', m, COL['neighbour']) for m in nb]
    draw(items2, (0.55, -0.75, -0.5), up, os.path.join(out, 'grip_16x230x250_iso.png'), title='Closed grip on a 16 x 230 x 250 box, s = 15..70 behind the spine; neighbours (spines at s = 100) behind')
    # closeup: crop to the fingers / top of the box
    r, uu, d = basis((0.55, -0.75, -0.5), up)
    pts = np.array([[-60, -45, P.Z_J - 5], [60, 80, P.Z_J + 80], [-60, 80, P.Z_J + 80], [60, -45, P.Z_J - 5], [0, 0, P.Z_J + 100]])
    pr = np.stack([pts @ r, pts @ uu], 1)
    dcl = (0.3, -1.0, -0.3)
    r, uu, d = basis(dcl, up)
    pr = np.stack([pts @ r, pts @ uu], 1)
    draw(items2, dcl, up, os.path.join(out, 'grip_16x230x250_closeup.png'), lim=(pr.min(0) - 20, pr.max(0) + 20), title='Closeup from the spine side: fingers straddle the 16 mm box, top edge against the centre stop')
    print('grip renders done')


if __name__ == '__main__':
    main()
