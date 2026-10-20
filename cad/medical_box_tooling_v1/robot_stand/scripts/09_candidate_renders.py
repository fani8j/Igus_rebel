"""Overlay the recommended boxes on the stand. Reads data/picks.json (id,label,lo,hi,rgb,ghost_dy). Usage: 09_candidate_renders.py OUT_DIR"""
import sys, os, json, collections, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *; from vtkrender import *
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
picks = json.load(open('data/picks.json'))
L = load_leaves(); V = collections.defaultdict(list); F = collections.defaultdict(list); off = collections.defaultdict(int)
for l in L:
    v, f = load_mesh(l['idx'])
    if len(f)==0: continue
    V[l['group']].append(v); F[l['group']].append(f+off[l['group']]); off[l['group']] += len(v)
GREY = {g: tuple(0.55*np.array(c)+0.45) for g, c in COLORS.items()}   # washed-out parts so the boxes stand out
base = [(np.vstack(V[g]), np.vstack(F[g]), GREY[g], 1.0) for g in V]
def boxes(sel, ghosts=True):
    it = []
    for p in sel:
        v, f = box_mesh(p['lo'], p['hi']); it.append((v, f, tuple(p['rgb']), 0.95))
        if ghosts:
            for dy in p.get('ghost_dy', []):
                v2 = v + np.array([0, dy, 0]); it.append((v2, f, tuple(p['rgb']), 0.25))
    return it
def annotate(r, fn, sel):
    im = r['im']; d = ImageDraw.Draw(im); fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    for p in sel:
        c = (np.array(p['lo'])+np.array(p['hi']))/2; x, y = r['proj'](c); d.text((x+14, y-12), p['id'], fill=tuple(int(255*q*0.7) for q in p['rgb']), font=fnt, stroke_width=3, stroke_fill=(255,255,255))
    im.save(fn)
leg = [(f"{p['id']}  {p['label']}", p['rgb']) for p in picks]
views = {'iso_front_left': ((-0.6,-0.45,-0.7),(0,1,0)), 'iso_back_right': ((0.6,-0.45,0.7),(0,1,0)), 'side': ((1,0,0),(0,1,0)), 'front': ((0,0,-1),(0,1,0)), 'top': ((0,-1,0),(0,0,1))}
for k, (vd, up) in views.items():
    r = render(base + boxes(picks, False), vd, up, f"{out}/candidates_{k}.png", f"pump-box candidates (200x110x100 mm), view {k}, Y up, mm", size=(2000,1300), legend=leg)
    annotate(r, f"{out}/candidates_{k}.png", picks)
# close-ups per pick with ghosts at the stroke ends
for p in picks:
    c = (np.array(p['lo'])+np.array(p['hi']))/2
    for vn, (vd, up) in (('iso', ((-0.6,-0.45,-0.7),(0,1,0))), ('side', ((1,0,0),(0,1,0))), ('iso_back', ((0.6,-0.45,0.7),(0,1,0)))):
        hh = 300 if vn != 'side' else 330
        r = render(base + boxes([p], True), vd, up, f"{out}/closeup_{p['id']}_{vn}.png", f"{p['id']}: {p['label']} | box {np.round(np.array(p['hi'])-np.array(p['lo'])).astype(int).tolist()} mm at centre {np.round(c).astype(int).tolist()}", size=(1800,1200), center=c, half_h=hh, legend=None)
        annotate(r, f"{out}/closeup_{p['id']}_{vn}.png", [p])
