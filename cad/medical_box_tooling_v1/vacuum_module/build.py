"""Builds every deliverable geometry file: step/parts, step/assembly.step, stl/ (print orientation), meshes/visual (+ manifest.json), data/mass.json, data/part_summary.json.
Run:  nice -n 19 "$PY" build.py        (about 30 s)
"""
import os, json, math
import numpy as np
import trimesh
import cadquery as cq
import params as P
import parts as K

HERE = P.HERE
OUT = lambda *a: os.path.join(HERE, *a)
for d in ('step/parts', 'stl', 'meshes/visual', 'data', 'renders'):
    os.makedirs(OUT(d), exist_ok=True)


def print_orientation(name, m):
    """STL in print orientation (mm): housing roof-down, door plate-down, base and deck as built (deck pads up)"""
    m = m.copy()
    if name == 'housing':
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    elif name == 'door':
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))      # x -> z: outer plate (x = 120) ends up on the bed after the flip below
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    b = m.bounds
    m.apply_translation([-(b[0][0] + b[1][0]) / 2 + 125, -(b[0][1] + b[1][1]) / 2 + 125, -b[0][2]])
    return m


def main():
    S = K.solids()
    mass = K.masses(S)
    summary = {}
    for n, d in S.items():
        sh = d['shape']
        cq.exporters.export(cq.Workplane(obj=sh), OUT('step/parts/%s.step' % n))
        if d['kind'] == 'printed':
            m = K.tess(sh, 0.05, 0.2)
            print_orientation(n, m).export(OUT('stl/%s.stl' % n))
            summary[n] = dict(kind='printed', material=P.PRINT_MATERIAL.split(' ')[0], volume_mm3=sh.Volume(), area_mm2=sh.Area(), est_mass_kg=mass[n], stl_triangles=int(len(m.faces)))
        else:
            summary[n] = dict(kind=d['kind'], volume_mm3=sh.Volume(), mass_kg=mass[n])
    json.dump(summary, open(OUT('data/part_summary.json'), 'w'), indent=1)
    # assembly
    a = cq.Assembly(name='vacuum_module')
    for n, d in S.items():
        a.add(d['shape'], name=n, color=cq.Color(*d['color']))
    a.save(OUT('step/assembly.step'), 'STEP')
    tot = sum(mass.values())
    json.dump(dict(items_kg=mass, total_kg=tot, printed_kg=sum(v for k, v in mass.items() if k in K.PRINTED),
                   note='printed = 1.6 mm shell + 25 %% infill PETG estimate; hardware masses are ASSUMED/datasheet (see params.py); foam/film/tube from volume'), open(OUT('data/mass.json'), 'w'), indent=1)
    # sim meshes (metres, module frame: x service side, z up, origin = underside of the base plate)
    man = dict(units='metres', frame='module frame: x = service door side, y = width, z up, origin at the underside of the base plate; the 4 spacers reach z = -0.012',
               mesh_path_convention='package://vacuum_module/meshes/visual/<file>', visual=[])
    for n, d in S.items():
        if d['kind'] in ('film',):
            continue
        tol = 0.2 if d['kind'] != 'tube' else 0.1
        m = K.tess(d['shape'], tol, 0.4)
        m.apply_scale(1e-3)
        m.export(OUT('meshes/visual/%s.stl' % n))
        man['visual'].append(dict(file='meshes/visual/%s.stl' % n, kind=d['kind'], color=list(d['color']), triangles=int(len(m.faces)), mass_kg=mass[n]))
    json.dump(man, open(OUT('meshes/manifest.json'), 'w'), indent=1)
    print('built %d solids, total mass %.2f kg' % (len(S), tot))


if __name__ == '__main__':
    main()
