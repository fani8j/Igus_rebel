"""Planar face summary (normal, area, #holes, bbox) of selected leaves, to find mounting faces/holes."""
import sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from OCP.TopoDS import TopoDS_Shape, TopoDS
from OCP.BinTools import BinTools
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_REVERSED
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepTools import BRepTools
L = load_leaves()
names = sys.argv[1:]
for l in L:
    if l['top'] not in names: continue
    s = TopoDS_Shape(); BinTools.Read_s(s, f"{W}/brep/leaf_{l['idx']:05d}.bin")
    print("==", l['top'], l['name'], l['bbox'])
    ex = TopExp_Explorer(s, TopAbs_FACE); cyl = []
    rows = []
    while ex.More():
        f = TopoDS.Face_s(ex.Current()); ad = BRepAdaptor_Surface(f)
        g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g)
        b = Bnd_Box(); BRepBndLib.Add_s(f, b)
        bb = [round(x) for x in (*b.CornerMin().Coord(), *b.CornerMax().Coord())]
        if ad.GetType() == GeomAbs_Plane:
            n = ad.Plane().Axis().Direction(); sgn = -1 if f.Orientation() == TopAbs_REVERSED else 1
            nw = TopExp_Explorer(f, TopAbs_WIRE); k = 0
            while nw.More(): k += 1; nw.Next()
            rows.append((round(g.Mass()), tuple(round(sgn*c, 2) for c in (n.X(), n.Y(), n.Z())), k-1, bb))
        elif ad.GetType() == GeomAbs_Cylinder:
            c = ad.Cylinder(); cyl.append((round(c.Radius()*2, 1), bb))
        ex.Next()
    rows.sort(reverse=True)
    for r in rows[:int(12)]: print("  plane area=%d n=%s holes=%d bb=%s" % r)
    import collections
    print("  cylinders (dia count):", collections.Counter(d for d, _ in cyl).most_common(8))
