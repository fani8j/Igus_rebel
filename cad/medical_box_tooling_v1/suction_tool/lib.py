"""CadQuery helpers (adapted from the earlier dual_arm_tooling_v1/tooling_lib.py): frames, oriented primitives, hulls, booleans."""
import math
import numpy as np
import cadquery as cq
from scipy.spatial import ConvexHull


def V(p):
    return cq.Vector(float(p[0]), float(p[1]), float(p[2]))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def mat4(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def move(shape, T):
    """rigid 4x4 transform of a cq shape"""
    from OCP.gp import gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    T = np.asarray(T, float)
    tr = gp_Trsf()
    tr.SetValues(*[float(v) for v in T[:3, :4].reshape(-1)])
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape())


def obox(center, R, size):
    """oriented box: centre, rotation (columns = box x,y,z), full size"""
    pl = cq.Plane(origin=V(center), xDir=V(R[:, 0]), normal=V(R[:, 2]))
    return cq.Workplane(pl).box(*[float(s) for s in size]).val()


def aabox(x, y, z):
    c = [(x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2]
    return obox(c, np.eye(3), [x[1] - x[0], y[1] - y[0], z[1] - z[0]])


def ocyl(p0, p1, r):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    return cq.Solid.makeCylinder(float(r), float(np.linalg.norm(d)), V(p0), V(unit(d)))


def zcyl(x, y, z0, z1, r):
    return ocyl([x, y, z0], [x, y, z1], r)


def zstadium(cx, cy, z0, z1, length, width, angle=0.0):
    """stadium (slot) prism along +z; length overall along x (rotated by angle deg about z)"""
    w = cq.Workplane('XY').workplane(offset=z0).center(cx, cy).slot2D(length, width, angle).extrude(z1 - z0)
    return w.val()


def zhex(cx, cy, z0, z1, af):
    """hex prism along z, flats at +-y (vertices on +-x), width across flats af"""
    d = af / math.cos(math.radians(30))
    return cq.Workplane('XY').workplane(offset=z0).center(cx, cy).polygon(6, d).extrude(z1 - z0).val()


def hull_solid(points):
    pts = np.asarray(points, float)
    h = ConvexHull(pts)
    faces = []
    c = pts.mean(0)
    for tri in h.simplices:
        a, b, cc = pts[tri]
        n = np.cross(b - a, cc - a)
        if np.dot(n, pts[tri].mean(0) - c) < 0:
            b, cc = cc, b
        faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon([V(a), V(b), V(cc)], close=True)))
    so = cq.Solid.makeSolid(cq.Shell.makeShell(faces))
    if so.Volume() < 0:
        so = cq.Solid(so.wrapped.Reversed())
    return cq.Workplane(obj=so).clean().val()


def extrude_poly_x(pts_yz, x0, x1):
    """polygon in the (y,z) plane extruded along x from x0 to x1"""
    pl = cq.Plane(origin=(x0, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))   # local x = world y, local y = normal x xDir = world z
    return cq.Workplane(pl).polyline([(float(p[0]), float(p[1])) for p in pts_yz]).close().extrude(float(x1 - x0)).val()


def hull2d(pts):
    p = np.asarray(pts, float)
    h = ConvexHull(p)
    return p[h.vertices]


def union(shapes):
    s = shapes[0]
    for o in shapes[1:]:
        s = s.fuse(o)
    return s.clean() if hasattr(s, 'clean') else s


def cut_all(base, tools):
    s = base
    for t in tools:
        s = s.cut(t)
    return s


def largest(shape):
    so = sorted(shape.Solids(), key=lambda s: -s.Volume())
    return so[0] if len(so) == 1 or not so else cq.Compound.makeCompound(so[:1])


def common_vol(a, b):
    ba, bb = a.BoundingBox(), b.BoundingBox()
    if (ba.xmax < bb.xmin or bb.xmax < ba.xmin or ba.ymax < bb.ymin or bb.ymax < ba.ymin or ba.zmax < bb.zmin or bb.zmax < ba.zmin):
        return 0.0
    try:
        return a.intersect(b).Volume()
    except Exception:
        return float('nan')


def tube_polyline(pts, r):
    """tube of radius r along a polyline: cylinders + spheres at the corners"""
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        parts.append(ocyl(a, b, r))
    for p in pts[1:-1]:
        parts.append(cq.Solid.makeSphere(float(r), V(p)))
    return union(parts) if len(parts) > 1 else parts[0]
