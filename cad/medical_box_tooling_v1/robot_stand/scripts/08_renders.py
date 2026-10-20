"""Standard views + base close-ups. Usage: 08_renders.py OUT_DIR"""
import sys, os, numpy as np, collections
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *; from vtkrender import *
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
L = load_leaves()
V = collections.defaultdict(list); F = collections.defaultdict(list); off = collections.defaultdict(int)
for l in L:
    v, f = load_mesh(l['idx'])
    if len(f)==0: continue
    V[l['group']].append(v); F[l['group']].append(f+off[l['group']]); off[l['group']] += len(v)
NAMEMAP = {'zlw_body':'ZLW-20120S belt axis (fixed, y-travel)','zlw_carriage':'ZLW carriage','rail_support':'rail bracket (on carriage)','rail_x':'WS-16 X rail + WW carriages',
  'plates':'upper/carrier plates','arm_A':'ReBeL arm A (x=-424)','arm_B':'ReBeL arm B (x=-884)','gusset':'triangular support (gusset)','side_plate':'side plates'}
items = [(np.vstack(V[g]), np.vstack(F[g]), COLORS[g], 1.0) for g in V]
leg = [(NAMEMAP[g], COLORS[g]) for g in V]
def label(r, fn, labels):
    im = r['im']; d = ImageDraw.Draw(im)
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    for txt, p in labels:
        x, y = r['proj'](p); d.text((x, y), txt, fill=(10,10,10), font=fnt, stroke_width=3, stroke_fill=(255,255,255))
    im.save(fn)
LAB = [("arm A (left?)", (-424,420,1000)), ("arm B (right?)", (-884,420,1000)), ("gusset / column", (-652,-900,300)), ("ZLW lift axis", (-652,-1100,740)), ("X rail", (-654,-270,830)), ("bracket", (-652,-450,930)),
       ("deck plane y=-1351 (assumed AMR top)", (-652,-1351,480))]
# CAD +Y is taken as world-up (arms are in their upright pose, ZLW is a 1250 mm lift column); the arms work toward +Z.
views = {
 'iso_front_left':  ((-0.6,-0.45,-0.7), (0,1,0)),
 'iso_back_right':  ((0.6,-0.45,0.7), (0,1,0)),
 'front':           ((0,0,-1), (0,1,0)),
 'side':            ((1,0,0), (0,1,0)),
 'top':             ((0,-1,0), (0,0,1)),
}
TIT = {'iso_front_left':'iso front-left (camera at +x,+z; CAD +Y up)','iso_back_right':'iso back-right (camera at -x,-z)','front':'front (camera at +z looking -z; arm A on the right of the picture)','side':'side (camera at -x, front +Z to the right)','top':'top (camera above, +Z up in picture)'}
for k, (vd, up) in views.items():
    r = render(items, vd, up, f"{out}/{k}.png", "robot stand, " + TIT[k] + " | units mm", size=(2000,1300), legend=leg)
    label(r, f"{out}/{k}.png", LAB)
for nm, cx in (('A', -424.), ('B', -884.)):
    for vn, vd, up in (('front_iso', (-0.6,-0.45,-0.7), (0,1,0)), ('back_iso', (0.6,-0.45,0.7), (0,1,0))):
        r = render(items, vd, up, f"{out}/closeup_base_{nm}_{vn}.png", f"close-up around arm {nm} base (x={cx:.0f}, base plate y=-242, z=855; Y up)", size=(2000,1300), center=(cx,-300,850), half_h=330, legend=leg)
        label(r, f"{out}/closeup_base_{nm}_{vn}.png", [(f"arm {nm} base", (cx,-242,950)), ("X rail", (cx,-270,780)), ("bracket", (-652,-450,930))])
