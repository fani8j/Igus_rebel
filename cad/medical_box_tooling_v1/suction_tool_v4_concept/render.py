"""Renders in the style / cameras of v3's render.py (imported, not copied).  python render.py -> renders/*.png"""
import os, sys, numpy as np
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
from PIL import Image, ImageDraw, ImageFont
import v4_params as V
import v4_parts as pa
from bridge import ML, load_v3_render, HERE, SCRATCH

R3 = load_v3_render()
R3.SPECIAL['bar_al6061'] = (0.78, 0.80, 0.83)
R3.SPECIAL['collector'] = (0.8, 0.8, 0.85)
OUT = os.path.join(HERE, 'renders'); os.makedirs(OUT, exist_ok=True)


def meshes_of(parts, dz=None):
    return [(d['name'], d['kind'], ML.to_mesh(d['shape'], tol=0.15, ang=0.25)) for d in parts]


def lim_of(ms, el, az, pad=8):
    r, u, d = R3.view_matrix(el, az)
    p = np.vstack([np.stack([m.vertices @ r, m.vertices @ u], 1) for _, _, m in ms])
    return (p.min(0) - pad, p.max(0) + pad)


def label(fn, text, fs=26):
    im = Image.open(fn).convert('RGB'); W, H = im.size
    out = Image.new('RGB', (W, H + 44), 'white'); out.paste(im, (0, 44))
    ImageDraw.Draw(out).text((10, 6), text, fill=(0, 0, 0), font=ImageFont.load_default(size=fs) if hasattr(ImageFont, 'load_default') else None)
    return out


if __name__ == '__main__':
    parts4 = pa.build_parts()
    ms = meshes_of(parts4)
    views = {'iso': (25, -50, 'iso (from +x, -y, +z)'), 'front': (0, -90, 'front: looking along +y (stem -y face, trunk side) - link6 +x to the right'),
             'side': (0, 0, 'side: looking along -x (+y sensor pod to the right)'), 'top': (89, -90, 'top: looking down -z (flange side of the pad row)')}
    for nm, (el, az, ti) in views.items():
        fn = '%s/assembly_%s.png' % (OUT, nm); R3.render(ms, el, az, fn, 'v4 ' + ti, size=(8, 8), dpi=150); R3.shrink(fn); print(fn)
    sc = ms + R3.box_meshes()
    fn = '%s/pressed_on_box_side.png' % OUT; R3.render(sc, 0, 0, fn, 'v4: pads pressed on a 16 x 230 x 250 spine, neighbours 15 mm proud (side view along -x)', size=(9, 7), dpi=150); R3.shrink(fn)
    fn = '%s/pressed_on_box_iso.png' % OUT; R3.render(sc, 22, -60, fn, 'v4: pads pressed on a 16 x 230 x 250 spine, neighbours 15 mm proud (iso)', size=(9, 8), dpi=150); R3.shrink(fn)
    # ---- v3 vs v4 at the same scale
    ms3 = R3.collect()
    for tag, (el, az) in {'iso': (25, -50), 'front': (0, -90)}.items():
        a, b = lim_of(ms, el, az), lim_of(ms3, el, az)
        lim = (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))
        f3 = os.path.join(SCRATCH, 'c3_%s.png' % tag); f4 = os.path.join(SCRATCH, 'c4_%s.png' % tag)
        R3.render(ms3, el, az, f3, '', lim=lim, size=(7, 7), dpi=130); R3.render(ms, el, az, f4, '', lim=lim, size=(7, 7), dpi=130)
        i3, i4 = label(f3, 'v3 (151 x 149 x 106 mm, 0.278 kg)'), label(f4, 'v4 concept (151 x 96 x 106 mm, see README)')
        h = max(i3.size[1], i4.size[1]); W = i3.size[0] + i4.size[0]
        out = Image.new('RGB', (W, h), 'white'); out.paste(i3, (0, 0)); out.paste(i4, (i3.size[0], 0))
        fn = '%s/compare_v3_v4_%s.png' % (OUT, tag); out.save(fn); R3.shrink(fn); print(fn, i3.size, i4.size)
    # ---- exploded view
    import trimesh
    Tc, Ts = V.T_CAM, V.T_SCAN
    off = {'bar_al6061': [0, 0, 30], 'pad_L': [0, 0, 85], 'pad_M': [0, 0, 85], 'pad_R': [0, 0, 85],
           'nut_front_L': [0, 0, 55], 'nut_front_M': [0, 0, 55], 'nut_front_R': [0, 0, 55],
           'nut_rear_L': [0, 0, 8], 'nut_rear_M': [0, 0, 8], 'nut_rear_R': [0, 0, 8],
           'tube_L': [0, -40, 0], 'tube_M': [0, -40, 0], 'tube_R': [0, -40, 0], 'collector': [0, -40, 0], 'trunk_tube': [0, -40, 0],
           'sensor_pod': [0, 50, 0], 'camera_gemini305': list(Tc[:3, 2] * 60 + [0, 50, 0]), 'scanner_se4710': list(Ts[:3, 2] * 50 + [0, 50, 0]),
           'decoder_pl3307': [55, 50, 0]}
    exm = []
    for n_, k_, m_ in ms:
        m2 = m_.copy(); m2.apply_translation(off.get(n_, [0, 0, 0])); exm.append((n_, k_, m2))
    fn = '%s/exploded_iso.png' % OUT; R3.render(exm, 25, -50, fn, 'v4 exploded: bar + pads + nuts up, tubes/collector -y, pod and sensors +y', size=(10, 9), dpi=150); R3.shrink(fn); print(fn)
