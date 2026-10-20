"""Tessellate every leaf BREP (from 01_load_tree) -> one npz per leaf with verts/faces (mm, world frame)."""
import sys, json, numpy as np, time
from OCP.TopoDS import TopoDS_Shape
from OCP.BinTools import BinTools
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
W = sys.argv[1]
L = json.load(open(f"{W}/leaves.json"))
import os; os.makedirs(f"{W}/mesh", exist_ok=True)
t0 = time.time()
for l in L:
    s = TopoDS_Shape(); BinTools.Read_s(s, f"{W}/brep/leaf_{l['idx']:05d}.bin")
    BRepMesh_IncrementalMesh(s, 0.6, False, 0.5, False)
    V = []; F = []; off = 0
    ex = TopExp_Explorer(s, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current()); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes()+1):
                p = tri.Node(i).Transformed(tr); V.append((p.X(), p.Y(), p.Z()))
            rev = f.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles()+1):
                a, b, c = tri.Triangle(i).Get()
                F.append((a-1+off, c-1+off, b-1+off) if rev else (a-1+off, b-1+off, c-1+off))
            off += tri.NbNodes()
        ex.Next()
    np.savez_compressed(f"{W}/mesh/leaf_{l['idx']:05d}.npz", v=np.array(V, np.float32), f=np.array(F, np.int32))
    if l['idx'] % 30 == 0: print(l['idx'], len(F), time.time()-t0, flush=True)
print("done", time.time()-t0)
