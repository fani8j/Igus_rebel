"""Shaded PNG renders (matplotlib painter's algorithm).  python render.py [name ...]  -> renders/*.png
assembly_{iso,front,side,top}, exploded_iso, section_x0 / section_y0 (through the stem, collector bay and the pad bar), pressed_on_box5 / pressed_on_box4,
camera_view (frustum + occlusion map at the approach pose), compare_v3_v4_v5_{iso,front}."""
import os, sys, math, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import trimesh
import params as P
import parts as pa
from meshlib import to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
C = P.C
OUT = os.path.join(HERE, 'renders')
COL = {'printed': (0.93, 0.55, 0.12), 'purchased': (0.55, 0.62, 0.74)}
SPECIAL = {'collector': (0.8, 0.8, 0.85), 'camera_gemini305': (0.15, 0.65, 0.3), 'box': (0.85, 0.8, 0.65), 'box_target': (0.75, 0.6, 0.35),
           'scanner_se4710': (0.1, 0.1, 0.1), 'decoder_pl3307': (0.1, 0.5, 0.2), 'bar_al6061': (0.78, 0.8, 0.83)}
for n in ('pad_L', 'pad_M', 'pad_R'):
    SPECIAL[n] = (0.2, 0.2, 0.25)
for n in ('elbow_L', 'elbow_M', 'elbow_R'):
    SPECIAL[n] = (0.55, 0.8, 0.95)
for n in ('tube_L', 'tube_M', 'tube_R', 'trunk_tube'):
    SPECIAL[n] = (0.2, 0.5, 0.9)


def collect(parts=None):
    return [(d['name'], d['kind'], to_mesh(d['shape'], tol=0.15, ang=0.25)) for d in (parts or pa.build_parts())]


def box_meshes(box_id=5, pressed=True):
    """target carton + two neighbours (same thickness) in the tool frame: spine plane at z = compressed TCP (154) when pressed, neighbours 15 mm proud (toward the tool),
    thickness along y, height along x (centred), length along +z"""
    b = C.BOXES[box_id]; t, h, l = b['t'], b['h'], b['l']
    z0 = P.COMPRESSED_TCP_Z if pressed else P.LIP_Z
    tgt = trimesh.creation.box(extents=[h, t, l]); tgt.apply_translation([0, 0, z0 + l / 2])
    out = [('box_target', 'box', tgt)]
    for s in (-1, 1):
        n = trimesh.creation.box(extents=[h, t, l + C.NEIGHBOUR_PROTRUSION]); n.apply_translation([0, s * t, z0 + (l + C.NEIGHBOUR_PROTRUSION) / 2 - C.NEIGHBOUR_PROTRUSION])
        out.append(('box', 'box', n))
    return out


def compress_pads(ms, s=5.0):
    """render-only: retract the pad heads (front 14 mm of each pad) by s mm, as in the pressed pose"""
    out = []
    for name, kind, m in ms:
        if name in ('pad_L', 'pad_M', 'pad_R'):
            cut = P.LIP_Z - 14.0
            front = m.slice_plane([0, 0, cut], [0, 0, 1], cap=True); back = m.slice_plane([0, 0, cut], [0, 0, -1], cap=True)
            front.apply_translation([0, 0, -s])
            m = trimesh.util.concatenate([front, back])
        out.append((name, kind, m))
    return out


def view_matrix(el, az):
    el, az = np.radians(el), np.radians(az)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    r = np.cross(up, d); r /= np.linalg.norm(r)
    return r, np.cross(d, r), d


def render(meshes, el, az, fn, title='', lim=None, size=(8.0, 8.0), dpi=150, up=None, ax_cb=None, colors=None):
    r, u, d = view_matrix(el, az)
    if up is not None:
        u = np.asarray(up, float) - d * np.dot(up, d); u /= np.linalg.norm(u); r = np.cross(u, d)
    light = np.array([0.4, -0.3, 0.9]); light /= np.linalg.norm(light)
    polys, cols, depth = [], [], []
    for name, kind, m in meshes:
        tri = m.vertices[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1; n = n / ln[:, None]
        vis = (n @ d) > -0.15
        sh = 0.45 + 0.55 * np.abs(n @ (0.5 * light + 0.5 * d))
        base = np.array((colors or {}).get(name, SPECIAL.get(name, COL.get(kind, (0.6, 0.6, 0.6)))))
        xy = np.stack([tri @ r, tri @ u], axis=-1); z = (tri @ d).mean(axis=1)
        for i in np.where(vis)[0]:
            polys.append(xy[i]); cols.append(np.clip(base * sh[i], 0, 1)); depth.append(z[i])
    order = np.argsort(depth)
    fig, ax = plt.subplots(figsize=size)
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors='none', linewidths=0))
    allv = np.vstack([m.vertices for _, _, m in meshes]); p = np.stack([allv @ r, allv @ u], axis=1)
    lo, hi = (p.min(0) - 8, p.max(0) + 8) if lim is None else (np.array(lim[0]), np.array(lim[1]))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10)
    if ax_cb is not None:
        ax_cb(ax, r, u, d)
    fig.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0.05); plt.close(fig)


def shrink(fn, max_kb=300):
    from PIL import Image
    im = Image.open(fn).convert('RGB')
    for colors in (256, 128, 64):
        im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(fn, optimize=True)
        if os.path.getsize(fn) <= max_kb * 1024:
            return


def lim_of(ms, el, az, pad=8):
    r, u, d = view_matrix(el, az)
    p = np.vstack([np.stack([m.vertices @ r, m.vertices @ u], 1) for _, _, m in ms])
    return (p.min(0) - pad, p.max(0) + pad)


def label(fn, text, fs=24):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(fn).convert('RGB'); W, H = im.size
    out = Image.new('RGB', (W, H + 44), 'white'); out.paste(im, (0, 44))
    ImageDraw.Draw(out).text((10, 6), text, fill=(0, 0, 0), font=ImageFont.load_default(size=fs))
    return out


# ------------------------------------------------------------------ v3 / v4 meshes for the comparison (their own exports, not rebuilt)
def v3_meshes():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'suction_tool', 'meshes', 'visual', '*.stl'))):
        m = trimesh.load(f, force='mesh'); m.apply_scale(1000.0)
        n = os.path.basename(f)[:-4]
        out.append((n, 'printed' if n in ('carrier', 'camera_arm', 'scanner_arm') else 'purchased', m))
    return out


def v4_meshes():
    import cadquery as cq
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'suction_tool_v4_concept', 'step', 'parts', '*.step'))):
        n = os.path.basename(f)[:-5]
        out.append((n, 'printed' if n in ('hub_stem', 'sensor_pod') else 'purchased', to_mesh(cq.importers.importStep(f).val(), tol=0.15, ang=0.25)))
    return out


# ------------------------------------------------------------------ camera view (frustum + occlusion map)
def camera_view(ms, fn, fov=C.CAMERA['depth_fov_deg']):
    """left: side view (y-z) with the depth FOV wedge in the tilt plane, the spine / neighbours at the approach pose; right: occlusion map in camera angle space
    (az along tool x, el in the tilt plane); ray = from the front-face centre, blocked when it hits the tool before the approach spine plane"""
    T = P.T_CAM; o = T[:3, 3]; ax_t = T[:3, 2]
    blk = trimesh.util.concatenate([m for n, k, m in ms if n != 'camera_gemini305'])
    ri = trimesh.ray.ray_triangle.RayMeshIntersector(blk)
    hf, vf = fov
    ths = np.linspace(-hf / 2, hf / 2, 89); phs = np.linspace(-vf / 2, vf / 2, 66)
    o_l, d_l, idx = [], [], []
    for a, th in enumerate(ths):
        for b, ph in enumerate(phs):
            dl = np.array([-math.tan(math.radians(th)), math.tan(math.radians(ph)), 1.0]); dl /= np.linalg.norm(dl)
            d_l.append(T[:3, :3] @ dl); o_l.append(o); idx.append((a, b))
    o_l = np.array(o_l); d_l = np.array(d_l)
    zs = P.LIP_Z + P.APPROACH_GAP
    tmax = (zs - o_l[:, 2]) / np.where(d_l[:, 2] > 1e-6, d_l[:, 2], 1e-6)         # distance to the spine plane z = zs
    loc, ir, _ = ri.intersects_location(o_l, d_l, multiple_hits=False)
    hit = np.zeros(len(o_l), bool)
    for l_, i_ in zip(loc, ir):
        if np.linalg.norm(l_ - o_l[i_]) < tmax[i_]:
            hit[i_] = True
    grid = np.zeros((len(phs), len(ths)))
    for h, (a, b) in zip(hit, idx):
        grid[b, a] = 1.0 if h else 0.0
    fig = plt.figure(figsize=(15, 7.2))
    a1 = fig.add_axes([0.02, 0.05, 0.50, 0.88]); a2 = fig.add_axes([0.58, 0.12, 0.40, 0.76])
    # --- side view
    r, u, d = view_matrix(0, 0)
    light = np.array([0.4, -0.3, 0.9]); light /= np.linalg.norm(light)
    polys, cols, depth = [], [], []
    for name, kind, m in ms:
        tri = m.vertices[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1; n = n / ln[:, None]
        vis = (n @ d) > -0.15; sh = 0.45 + 0.55 * np.abs(n @ (0.5 * light + 0.5 * d))
        base = np.array(SPECIAL.get(name, COL.get(kind, (0.6, 0.6, 0.6))))
        xy = np.stack([tri @ r, tri @ u], axis=-1); z = (tri @ d).mean(axis=1)
        for i in np.where(vis)[0]:
            polys.append(xy[i]); cols.append(np.clip(base * sh[i], 0, 1)); depth.append(z[i])
    order = np.argsort(depth)
    a1.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors='none', linewidths=0))
    a = math.radians(P.CAM_TILT_DEG); ymax = 95.0
    for sgn, ls in ((-1, '-'), (1, '-')):                                  # tilt-plane FOV edges (V half angle) from the front face centre
        ang = a + sgn * math.radians(vf / 2)
        L_ = (zs + 25 - o[2]) / math.cos(ang)
        a1.plot([o[1], o[1] - L_ * math.sin(ang)], [o[2], o[2] + L_ * math.cos(ang)], color='tab:green', lw=1.6, ls=ls)
    ang0 = a
    L_ = (zs + 25 - o[2]) / math.cos(ang0)
    a1.plot([o[1], o[1] - L_ * math.sin(ang0)], [o[2], o[2] + L_ * math.cos(ang0)], color='tab:green', lw=1.0, ls='--')
    # blocked rays in the tilt plane (centre column)
    ci = len(ths) // 2
    for b, ph in enumerate(phs):
        k = ci * len(phs) + b
        e = o_l[k] + d_l[k] * (min(tmax[k], np.linalg.norm(loc[list(ir).index(k)] - o_l[k])) if hit[k] else tmax[k])
        a1.plot([o[1], e[1]], [o[2], e[2]], color=('tab:red' if hit[k] else (0.6, 0.85, 0.6)), lw=0.6, alpha=0.8)
    # spine / neighbours at the approach pose (box 4: 16 x 250; target + two neighbours 15 mm proud), seen from the side: thin slabs in y
    for bid, c_ in ((5, 'tab:blue'), (4, 'tab:purple')):
        t = C.BOXES[bid]['t']
        a1.add_patch(plt.Rectangle((-t / 2, zs), t, 40, fc='none', ec=c_, lw=1.4))
        for s in (-1, 1):
            a1.add_patch(plt.Rectangle((s * t - t / 2, zs - C.NEIGHBOUR_PROTRUSION), t, 40, fc='none', ec=c_, lw=1.0, ls=':'))
    a1.plot([-60, 80], [zs, zs], color='k', lw=0.8, ls=':'); a1.text(62, zs + 3, 'spine plane z = %.0f (approach)' % zs, fontsize=8)
    a1.plot([-60, 80], [P.COMPRESSED_TCP_Z] * 2, color='gray', lw=0.6, ls=':'); a1.text(62, P.COMPRESSED_TCP_Z - 7, 'pressed spine z = 154', fontsize=8)
    a1.set_xlim(-50, 95); a1.set_ylim(45, 235); a1.set_aspect('equal'); a1.axis('off')
    a1.set_title('side view (from +x): camera right, collector bay left; green wedge = depth FOV (65 deg),\n'
                 'rays: red = blocked by the tool before the approach spine plane, pale green = clear. Dotted/solid rectangles: box 5 (12) and box 4 (16) spines + 15 mm proud neighbours', fontsize=8)
    # --- occlusion map
    a2.imshow(1.0 - grid, origin='lower', extent=[-hf / 2, hf / 2, -vf / 2, vf / 2], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    a2.set_xlabel('horizontal angle in the camera image [deg] (tool x)'); a2.set_ylabel('angle in the tilt plane [deg] (+ = toward -y of the tool)')
    a2.set_title('occlusion map: %.2f %% of the %d x %d depth-FOV rays blocked by the bar / pads / body at the approach pose' % (100 * hit.mean(), len(ths), len(phs)), fontsize=9)
    fig.savefig(fn, dpi=110); plt.close(fig)
    return float(hit.mean())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])
    want = lambda k: not only or k in only
    parts5 = pa.build_parts()
    ms = collect(parts5)
    if want('assembly'):
        views = {'iso': (25, -50, 'iso (from +x, -y, +z)'), 'front': (0, -90, 'front: looking along +y (collector bay / trunk side) - link6 +x to the right'),
                 'side': (0, 0, 'side: looking along -x (+y camera side to the right, -y collector bay to the left)'), 'top': (89, -90, 'top: looking down -z (flange side of the pad row)')}
        for nm, (el, az, ti) in views.items():
            fn = '%s/assembly_%s.png' % (OUT, nm); render(ms, el, az, fn, 'v5 ' + ti, size=(8, 8), dpi=150); shrink(fn); print(fn, os.path.getsize(fn) // 1024, 'kB')
    if want('exploded'):
        Tc = P.T_CAM
        off = {}
        for n_ in ('pad_L', 'pad_M', 'pad_R'): off[n_] = [0, 0, 85]
        for n_ in ('nut_front_L', 'nut_front_M', 'nut_front_R'): off[n_] = [0, 0, 60]
        for n_ in ('washer_L', 'washer_M', 'washer_R'): off[n_] = [0, 0, -25]
        for n_ in ('nut_rear_L', 'nut_rear_M', 'nut_rear_R'): off[n_] = [0, 0, -30]
        for n_ in ('elbow_L', 'elbow_M', 'elbow_R'): off[n_] = [0, 0, -38]
        for n_ in ('tube_L', 'tube_M', 'tube_R'): off[n_] = [0, -22, -38]
        off['collector'] = [0, -22, -20]; off['trunk_tube'] = [0, -22, -20]
        off['camera_gemini305'] = [0, 60, 0]
        exm = []
        for n_, k_, m_ in ms:
            m2 = m_.copy(); m2.apply_translation(off.get(n_, [0, 0, 0])); exm.append((n_, k_, m2))
        fn = '%s/exploded_iso.png' % OUT; render(exm, 25, -50, fn, 'v5 exploded: pads + front nuts up, rear nuts / washers / elbows / tubes / collector down (-z, -y), camera off the stem wall (+y)', size=(10, 9), dpi=150); shrink(fn); print(fn)
    if want('section'):
        for tag, (pl_o, pl_n, el, az, ti) in {'x0': ([0, 0, 0], [-1, 0, 0], 0, 0, 'section x = 0 (plane through the middle pad, stem, collector bay, camera on the +y wall); view from +x'),
                                              'y0': ([0, 0, 0], [0, 1, 0], 0, -90, 'section y = 0 (long section through the pad bar, the three pads/elbows and the stem walls); view along +y')}.items():
            cm = []
            for n_, k_, m_ in ms:
                try:
                    mm = m_.slice_plane(pl_o, pl_n, cap=True)
                except Exception:
                    mm = m_.slice_plane(pl_o, pl_n, cap=False)
                if len(mm.faces):
                    cm.append((n_, k_, mm))
            fn = '%s/section_%s.png' % (OUT, tag); render(cm, el, az, fn, 'v5 ' + ti, size=(9, 8), dpi=150); shrink(fn); print(fn)
    if want('pressed'):
        for bid in (5, 4):
            b = C.BOXES[bid]
            sc = compress_pads(ms) + box_meshes(bid)
            fn = '%s/pressed_on_box%d_side.png' % (OUT, bid); render(sc, 0, 0, fn, 'v5: pads pressed (heads retracted 5 mm) on box %d (%g x %g x %g), neighbours 15 mm proud - side view along -x' % (bid, b['t'], b['h'], b['l']), size=(9, 7), dpi=150); shrink(fn); print(fn)
            fn = '%s/pressed_on_box%d_iso.png' % (OUT, bid); render(sc, 22, -60, fn, 'v5: pressed on box %d (iso)' % bid, size=(9, 8), dpi=150); shrink(fn); print(fn)
    if want('camera'):
        fr = camera_view(ms, '%s/camera_view.png' % OUT); print('camera blocked fraction', fr)
        shrink('%s/camera_view.png' % OUT, 400)
    if want('compare'):
        ms3, ms4 = v3_meshes(), v4_meshes()
        for tag, (el, az) in {'iso': (25, -50), 'front': (0, -90), 'side': (0, 0)}.items():
            lims = [lim_of(x, el, az) for x in (ms, ms3, ms4)]
            lim = (np.minimum.reduce([l[0] for l in lims]), np.maximum.reduce([l[1] for l in lims]))
            ims = []
            for nm, mm, ttl in (('v3', ms3, 'v3 (149 x 151 x 106 mm, 0.278 kg)'), ('v4', ms4, 'v4 concept (96 x 151 x 106 mm, 0.332 kg)'), ('v5', ms, 'v5 (this tool, see README)')):
                f = os.path.join(P.SCRATCH, 'cmp_%s_%s.png' % (nm, tag))
                render(mm, el, az, f, '', lim=lim, size=(6, 6), dpi=110); ims.append(label(f, ttl, 22))
            from PIL import Image
            h = max(i.size[1] for i in ims); W = sum(i.size[0] for i in ims)
            out = Image.new('RGB', (W, h), 'white'); x0 = 0
            for i in ims:
                out.paste(i, (x0, 0)); x0 += i.size[0]
            fn = '%s/compare_v3_v4_v5_%s.png' % (OUT, tag); out.save(fn); shrink(fn); print(fn)
