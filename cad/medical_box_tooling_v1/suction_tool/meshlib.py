"""cq shape -> trimesh helpers"""
import numpy as np, trimesh


def to_mesh(shape, tol=0.08, ang=0.15):
    parts = []
    for so in (shape.Solids() or [shape]):
        v, t = so.tessellate(tol, ang)
        if not len(t):
            continue
        parts.append(trimesh.Trimesh(vertices=np.array([x.toTuple() for x in v]), faces=np.array(t), process=True))
    m = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices(); m.merge_vertices()
    return m
