"""Mass, CoG and inertia of the assembly (uniform density per body; printed parts = PETG x effective fill factor)."""
import numpy as np
import params as P
from meshlib import to_mesh


def body_mass(d):
    if d['kind'] == 'printed':
        return d['shape'].Volume() * P.PETG_DENSITY * P.PRINT_FILL_EQ
    return d['mass']


def assembly_props(parts, include_ejector=True, meshes=None):
    tot = 0.0; com = np.zeros(3); rows = []
    for d in parts:
        if d['kind'] == 'ghost' and not include_ejector:
            continue
        m = body_mass(d)
        mesh = (meshes or {}).get(d['name']) or to_mesh(d['shape'], tol=0.1, ang=0.2)
        c = np.array(mesh.center_mass) if mesh.is_watertight else np.array(mesh.bounding_box.centroid)
        I = np.array(mesh.moment_inertia) / mesh.volume * m if mesh.is_watertight else np.eye(3) * 0
        rows.append((d['name'], m, c, I)); tot += m; com += m * c
    com /= tot
    Ic = np.zeros((3, 3))
    for n, m, c, I in rows:
        r = c - com
        Ic += I + m * ((r @ r) * np.eye(3) - np.outer(r, r))
    return dict(mass=tot, com=com, inertia=Ic, bodies={n: (m, c.tolist()) for n, m, c, _ in rows})
