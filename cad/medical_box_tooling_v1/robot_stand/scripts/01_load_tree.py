"""Load the STEP via XCAF, walk the assembly tree, dump every leaf instance
(name, path, world bbox, volume, colour) to JSON. Units forced to mm."""
import sys, json, time
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Quantity import Quantity_Color
from OCP.TopLoc import TopLoc_Location
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.BinTools import BinTools

src, out, brepdir = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
app = XCAFApp_Application.GetApplication_s()
doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
app.InitDocument(doc)
rd = STEPCAFControl_Reader(); rd.SetColorMode(True); rd.SetNameMode(True)
st = rd.ReadFile(src); assert st == IFSelect_RetDone, st
print("read", time.time()-t0, flush=True)
ok = rd.Transfer(doc); print("transfer", ok, time.time()-t0, flush=True)
st_ = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

def name(l):
    n = TDataStd_Name()
    if l.FindAttribute(TDataStd_Name.GetID_s(), n):
        return n.Get().ToExtString()
    return ""

def color(l):
    c = Quantity_Color()
    for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
        if ct.GetColor_s(l, t, c): return [round(c.Red(),3), round(c.Green(),3), round(c.Blue(),3)]
    return None

leaves = []
def walk(label, path, loc, col):
    # label is a shape label (maybe reference)
    if st_.IsReference_s(label):
        ref = TDF_Label()
        st_.GetReferredShape_s(label, ref)
        nm = name(label) or name(ref)
        c = color(label) or col
        walk2(ref, path + [nm], loc, c, label)
    else:
        walk2(label, path, loc, col, None)

def walk2(label, path, loc, col, reflabel):
    c = color(label) or col
    if st_.IsAssembly_s(label):
        comps = TDF_Label.__new__(TDF_Label)
        seq = TDF_LabelSequence(); st_.GetComponents_s(label, seq)
        for i in range(1, seq.Length()+1):
            comp = seq.Value(i)
            l2 = st_.GetLocation_s(comp)
            walk(comp, path + [name(comp)], loc.Multiplied(l2), c)
    else:
        shp = st_.GetShape_s(label)
        # apply accumulated location
        shp = shp.Moved(loc)
        b = Bnd_Box(); BRepBndLib.AddOptimal_s(shp, b, False, False)
        if b.IsVoid(): return
        mn = b.CornerMin(); mx = b.CornerMax()
        gp = GProp_GProps(); BRepGProp.VolumeProperties_s(shp, gp)
        nsol = 0; ex = TopExp_Explorer(shp, TopAbs_SOLID)
        while ex.More(): nsol += 1; ex.Next()
        idx = len(leaves)
        BinTools.Write_s(shp, f"{brepdir}/leaf_{idx:05d}.bin")
        leaves.append(dict(idx=idx, name=name(label), path=path, color=c,
            bbox=[mn.X(),mn.Y(),mn.Z(),mx.X(),mx.Y(),mx.Z()], volume=gp.Mass(), nsolids=nsol))

free = TDF_LabelSequence(); st_.GetFreeShapes(free)
print("free shapes", free.Length(), flush=True)
for i in range(1, free.Length()+1):
    l = free.Value(i)
    print(" top", i, name(l), flush=True)
    walk2(l, [name(l)], TopLoc_Location(), None, None) if not st_.IsReference_s(l) else walk(l, [], TopLoc_Location(), None)
json.dump(leaves, open(out, "w"))
print("leaves", len(leaves), time.time()-t0)
