"""Build: step/parts/*.step, step/assembly.step, stl/*.stl (printed parts), data/mass.json, data/printed_parts.json.   python build.py"""
import os, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, trimesh
import cadquery as cq
import v4_parts as pa
from bridge import ML, MP3, HERE

COL = {'printed': (0.93, 0.55, 0.12), 'purchased': (0.55, 0.62, 0.74)}
SPECIAL = {'camera_gemini305': (0.15, 0.65, 0.3), 'scanner_se4710': (0.1, 0.1, 0.1), 'decoder_pl3307': (0.1, 0.5, 0.2), 'bar_al6061': (0.78, 0.8, 0.83),
           'pad_L': (0.2, 0.2, 0.25), 'pad_M': (0.2, 0.2, 0.25), 'pad_R': (0.2, 0.2, 0.25)}
for d in ('step/parts', 'stl', 'data'):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
for sub, ext in (('step/parts', '.step'), ('stl', '.stl')):
    for f in os.listdir(os.path.join(HERE, sub)):
        if f.endswith(ext): os.remove(os.path.join(HERE, sub, f))
parts = pa.build_parts()
asm = cq.Assembly(name='left_suction_tool_v4_concept'); summ = []
for d in parts:
    nm, sh = d['name'], d['shape']
    cq.exporters.export(sh, os.path.join(HERE, 'step/parts/%s.step' % nm))
    asm.add(sh, name=nm, color=cq.Color(*SPECIAL.get(nm, COL[d['kind']])))
    if d['kind'] == 'printed':
        m = ML.to_mesh(sh, tol=0.03, ang=0.1)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m); trimesh.repair.fix_normals(m)
        m.export(os.path.join(HERE, 'stl/%s.stl' % nm))
        bb = sh.BoundingBox()
        summ.append(dict(part=nm, volume_cm3=sh.Volume() / 1000, mass_g=MP3.body_mass(d) * 1000, bbox_mm=[bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax], watertight=bool(m.is_watertight), material=d['material'], orient=d['orient']))
asm.save(os.path.join(HERE, 'step/assembly.step'), 'STEP')
json.dump(summ, open(os.path.join(HERE, 'data/printed_parts.json'), 'w'), indent=1)
mp = MP3.assembly_props(parts)
json.dump(dict(mass_kg=mp['mass'], com_mm=mp['com'].tolist(), inertia_kg_mm2=mp['inertia'].tolist(), bodies=mp['bodies']), open(os.path.join(HERE, 'data/mass.json'), 'w'), indent=1)
print('mass %.3f kg CoG %s' % (mp['mass'], np.round(mp['com'], 1)), [(s['part'], round(s['mass_g'], 1), s['watertight']) for s in summ])
