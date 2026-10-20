"""Offscreen VTK ortho renders with PIL annotations (axes triad, scale bar, legend, title)."""
import numpy as np, vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from PIL import Image, ImageDraw, ImageFont

def polydata(v, f, rgb, alpha=1.0):
    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(v, np.float64), deep=True))
    cells = np.hstack([np.full((len(f),1),3), f]).astype(np.int64).ravel()
    ca = vtk.vtkCellArray(); ca.SetCells(len(f), numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetPolys(ca)
    return pd

def box_mesh(lo, hi):
    lo=np.array(lo,float); hi=np.array(hi,float)
    c=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    f=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return c, np.array(f)

def render(items, view_dir, up, fn, title='', size=(2000,1300), center=None, half_h=None, legend=None, bg=(0.97,0.97,0.98), edges=True):
    """items: list of (verts, faces, rgb, alpha). view_dir = direction camera looks along."""
    ren = vtk.vtkRenderer(); ren.SetBackground(*bg)
    allv = []
    for v, f, rgb, a in items:
        pd = polydata(v, f, rgb)
        nm = vtk.vtkPolyDataNormals(); nm.SetInputData(pd); nm.SetFeatureAngle(40); nm.SplittingOn(); nm.Update()
        m = vtk.vtkPolyDataMapper(); m.SetInputConnection(nm.GetOutputPort())
        act = vtk.vtkActor(); act.SetMapper(m)
        p = act.GetProperty(); p.SetColor(*rgb); p.SetOpacity(a); p.SetAmbient(0.35); p.SetDiffuse(0.7); p.SetSpecular(0.05)
        ren.AddActor(act); allv.append(v)
    allv = np.vstack(allv)
    lo, hi = allv.min(0), allv.max(0)
    ctr = (lo+hi)/2 if center is None else np.array(center, float)
    vd = np.array(view_dir, float); vd /= np.linalg.norm(vd)
    cam = ren.GetActiveCamera(); cam.ParallelProjectionOn()
    ext = np.linalg.norm(hi-lo)
    cam.SetFocalPoint(*ctr); cam.SetPosition(*(ctr - vd*ext*3)); cam.SetViewUp(*up)
    W, H = size
    if half_h is None:
        # fit: project bbox corners
        r = np.cross(vd, up); r/=np.linalg.norm(r); u = np.cross(r, vd)
        cs = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])-ctr
        hw = np.abs(cs@r).max(); hh = np.abs(cs@u).max()
        half_h = max(hh, hw*H/W)*1.08
    cam.SetParallelScale(half_h); cam.SetClippingRange(ext*0.5, ext*6)
    ren.AddLight(vtk.vtkLight()) if False else None
    ren.SetUseDepthPeeling(1); ren.SetMaximumNumberOfPeels(8); ren.SetOcclusionRatio(0.0)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(W, H); win.SetAlphaBitPlanes(1); win.SetMultiSamples(0)
    win.AddRenderer(ren); win.Render()
    wi = vtk.vtkWindowToImageFilter(); wi.SetInput(win); wi.Update()
    img = wi.GetOutput(); arr = np.frombuffer(vtk.util.numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()), np.uint8).reshape(H, W, -1)[::-1, :, :3]
    im = Image.fromarray(np.ascontiguousarray(arr)); d = ImageDraw.Draw(im)
    try: fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22); fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception: fnt = fs = ImageFont.load_default()
    r = np.cross(vd, up); r/=np.linalg.norm(r); u = np.cross(r, vd)
    pxmm = H/(2*half_h)
    def proj(p):
        q = np.array(p)-ctr
        return (W/2 + (q@r)*pxmm, H/2 - (q@u)*pxmm)
    d.text((20, 14), title, fill=(20,20,20), font=fnt)
    # axes triad
    o = (110, H-110)
    for ax, col, lab in ((np.array([1,0,0.]),(200,0,0),'X'),(np.array([0,1,0.]),(0,150,0),'Y'),(np.array([0,0,1.]),(0,0,220),'Z')):
        dx, dy = (ax@r)*70, -(ax@u)*70
        d.line([o, (o[0]+dx, o[1]+dy)], fill=col, width=4)
        d.text((o[0]+dx*1.2-6, o[1]+dy*1.2-10), lab, fill=col, font=fnt)
    # scale bar
    for L_ in (100, 200, 500, 1000):
        if L_*pxmm > 150: break
    x0 = W-60-L_*pxmm; y0 = H-50
    d.line([(x0,y0),(x0+L_*pxmm,y0)], fill=(0,0,0), width=5); d.text((x0, y0-30), f"{L_} mm", fill=(0,0,0), font=fnt)
    if legend:
        y = 50
        for name, col in legend:
            d.rectangle([20, y, 40, y+18], fill=tuple(int(255*c) for c in col), outline=(0,0,0)); d.text((48, y-2), name, fill=(0,0,0), font=fs); y += 24
    im.save(fn)
    return dict(ctr=ctr, r=r, u=u, pxmm=pxmm, size=size, im=im, proj=proj)
