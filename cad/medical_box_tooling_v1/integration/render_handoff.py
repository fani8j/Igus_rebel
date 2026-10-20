#!/usr/bin/env python3
"""Software z-buffer renders of the handoff (both tools + wrists + target + flush neighbours), boxes 3 and 4.
Writes renders/handoff_box{3,4}_{iso,side,front}.png (1200 px) for the default suction tool (v5); `--tool suction_tool` (v3) writes renders/v3/.
Uses the visual meshes (exact), not the collision hulls."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import handoff_check as hc
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'S': (40, 150, 170), 'SW': (110, 110, 120), 'G': (235, 140, 30), 'GW': (110, 110, 120),
       'B': (205, 170, 110), 'N': (235, 220, 190), 'F': (200, 200, 205)}


def scene_meshes(sc, box_id):
    out = []                                              # (verts_W, faces, colour, ghost)
    def add(m, T, col, ghost=False):
        v = (T[:3, :3] @ np.asarray(m.vertices).T).T + T[:3, 3]
        out.append((v, np.asarray(m.faces), col, ghost))
    for _, m in sc.S.visual:
        add(m, sc.Ts, COL['S'])
    for p in sc.SW.parts:
        add(p.render, sc.Ts @ p.local, COL['SW'])
    for _, m in sc.G.visual_base:
        add(m, sc.Tg, COL['G'])
    d = sc.G.opening
    for side, sg in (('L', -1.0), ('R', 1.0)):
        for n, m in sc.G.visual_fingers[side]:
            add(m, sc.Tg @ hc.T_of(t=[sg * d, 0, 150.0]), (60, 140, 215) if 'pad' not in n else (30, 30, 30))
    for p in sc.GW.parts:
        add(p.render, sc.Tg @ p.local, COL['GW'])
    for g, col, gh in ([(sc.B, COL['B'], False)] + [(n, COL['N'], True) for n in sc.N]):
        p = g.parts[0]
        add(p.render, p.world, col, gh)
    floor = trimesh.creation.box([700, 700, 4.0])
    add(floor, hc.T_of(t=[0, 100, -2.0]), COL['F'])
    return out


def rasterize(meshes, cam_dir, up, size=1200, ss=2, ghost_all=False, fit=None):
    cam = np.asarray(cam_dir, float); cam /= np.linalg.norm(cam)
    right = np.cross(up, cam); right /= np.linalg.norm(right)
    upv = np.cross(cam, right)
    R = np.stack([right, upv, cam])                         # world -> (x_s, y_s, depth)
    pts = np.vstack([(R @ v.T).T for v, f, c, g in meshes[:-1]])   # exclude floor from the fit
    lo, hi = pts.min(0), pts.max(0)
    N = size * ss
    scale = 0.92 * N / max(hi[0] - lo[0], hi[1] - lo[1])
    ox = N / 2 - scale * (lo[0] + hi[0]) / 2
    oy = N / 2 + scale * (lo[1] + hi[1]) / 2
    img = np.full((N, N, 3), 250, np.uint8)
    zbuf = np.full((N, N), -1e18)
    sun = np.array([0.3, -0.5, 0.8]); sun /= np.linalg.norm(sun)
    for v, f, col, ghost in meshes:
        P = (R @ v.T).T
        sx = P[:, 0] * scale + ox
        sy = oy - P[:, 1] * scale
        sz = P[:, 2]
        tri = f
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1); ln[ln == 0] = 1
        nrm /= ln[:, None]
        shade = 0.30 + 0.45 * np.abs(nrm @ cam) + 0.25 * np.clip(nrm @ sun, 0, 1)
        shade = np.clip(shade, 0, 1)
        X = sx[tri]; Y = sy[tri]; Z = sz[tri]
        x0 = np.floor(X.min(1)).astype(int).clip(0, N - 1); x1 = np.ceil(X.max(1)).astype(int).clip(0, N - 1)
        y0 = np.floor(Y.min(1)).astype(int).clip(0, N - 1); y1 = np.ceil(Y.max(1)).astype(int).clip(0, N - 1)
        den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
        colv = np.array(col, float)
        for i in range(len(tri)):
            if den[i] == 0:
                continue
            xs = np.arange(x0[i], x1[i] + 1); ys = np.arange(y0[i], y1[i] + 1)
            gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
            w0 = ((Y[i, 1] - Y[i, 2]) * (gx - X[i, 2]) + (X[i, 2] - X[i, 1]) * (gy - Y[i, 2])) / den[i]
            w1 = ((Y[i, 2] - Y[i, 0]) * (gx - X[i, 2]) + (X[i, 0] - X[i, 2]) * (gy - Y[i, 2])) / den[i]
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not m.any():
                continue
            z = w0 * Z[i, 0] + w1 * Z[i, 1] + w2 * Z[i, 2]
            sub = zbuf[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
            ok = m & (z > sub)
            if ghost:
                ok &= ((gx.astype(int) + gy.astype(int)) % 2 == 0)       # 50 % stipple: see-through neighbours
            if ok.any():
                sub[ok] = z[ok]
                img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1][ok] = (colv * shade[i]).astype(np.uint8)
    im = Image.fromarray(img).resize((size, size), Image.LANCZOS)
    return im


def main():
    sc = hc.Scene()
    views = {'iso': ((0.75, -0.95, 0.55), (0, 0, 1)), 'side': ((1, 0, 0), (0, 0, 1)), 'front': ((0, -1, 0), (0, 0, 1))}
    outdir = os.path.join(HERE, 'renders') if not hc.OUT_SUFFIX else os.path.join(HERE, 'renders', hc.OUT_SUFFIX.lstrip('_'))
    os.makedirs(outdir, exist_ok=True)
    for bid in (3, 4):
        sc.configure(bid)
        ms = scene_meshes(sc, bid)
        for vn, (cd, up) in views.items():
            im = rasterize(ms, cd, up)
            dr = ImageDraw.Draw(im)
            b = hc.P.BOXES[bid]
            dr.text((12, 10), 'box %d (%s) %gx%gx%g mm, handoff (%s), flush neighbours, view %s  |  teal=suction tool, orange/blue=gripper, grey=ReBeL link5/6, tan=target, stipple=neighbours' % (bid, b['name'], b['t'], b['h'], b['l'], hc.SUCTION_NAME, vn), fill=(0, 0, 0))
            p = os.path.join(outdir, 'handoff_box%d_%s.png' % (bid, vn))
            im.save(p)
            print(p, flush=True)


if __name__ == '__main__':
    main()
