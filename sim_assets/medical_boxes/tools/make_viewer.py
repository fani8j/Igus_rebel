#!/usr/bin/env python3
"""Write viewer.html: a self-contained page with an interactive 3D view of every carton
and its spec / verification summary.

Geometry, UVs, face textures, mass and customData are read back out of usd/<box>.usda
(not from boxes.yaml), so the page shows what Isaac Sim loads. boxes.yaml supplies the
caliper measurement, texture provenance and evidence notes; verify_usd.py --json supplies
the check results.

    "$PY" tools/verify_usd.py --json /tmp/verify.json
    "$PY" tools/make_viewer.py --verify-json /tmp/verify.json
"""
import argparse
import base64
import json
import os

import cv2
import yaml
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FACE_ORDER = ["front", "back", "top", "bottom", "left", "right"]


def data_uri(img_bgr, max_side, png=False):
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    if png:
        ok, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
        mime = "image/jpeg"
    assert ok
    return "data:%s;base64,%s" % (mime, base64.b64encode(buf.tobytes()).decode())


def bound_texture_path(subset):
    mat, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
    for prim in Usd.PrimRange(mat.GetPrim()):
        sh = UsdShade.Shader(prim)
        if sh and sh.GetIdAttr().Get() == "UsdUVTexture":
            return sh.GetInput("file").Get().resolvedPath
    raise RuntimeError("no UsdUVTexture bound to " + str(subset.GetPath()))


def texture_kind(desc):
    d = str(desc).lower()
    if "http" in d or "internet" in d or "reseller" in d:
        return "internet"
    if "photo" in d:
        return "photo"
    return "plain"


def read_box(name, spec, usd_dir):
    stage = Usd.Stage.Open(os.path.join(usd_dir, name + ".usda"))
    root = stage.GetDefaultPrim()
    mesh = UsdGeom.Mesh(root.GetChild("geom"))
    pts = [tuple(p) for p in mesh.GetPointsAttr().Get()]
    fvi = list(mesh.GetFaceVertexIndicesAttr().Get())
    fvc = list(mesh.GetFaceVertexCountsAttr().Get())
    st = [tuple(v) for v in UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()]
    assert all(c == 4 for c in fvc), "expected quads"
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
        .ComputeWorldBound(root).ComputeAlignedRange()
    size_mm = [round((bbox.GetMax()[i] - bbox.GetMin()[i]) * 1000.0, 3) for i in range(3)]
    mass = root.GetAttribute("physics:mass").Get()
    faces, textures = [], {}
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        fname = subset.GetPrim().GetName()
        (fi,) = subset.GetIndicesAttr().Get()
        tex_path = bound_texture_path(subset)
        img = cv2.imread(tex_path, cv2.IMREAD_COLOR)
        small = max(img.shape[:2]) <= 300
        key = "%s/%s" % (name, fname)
        textures[key] = data_uri(img, 1024, png=small or fname == "top")
        faces.append({
            "name": fname,
            "points": [pts[fvi[fi * 4 + k]] for k in range(4)],
            "uvs": [st[fi * 4 + k] for k in range(4)],
            "tex": key,
            "texFile": os.path.relpath(tex_path, ROOT),
            "source": spec.get("textures", {}).get(fname, ""),
            "kind": texture_kind(spec.get("textures", {}).get(fname, "")),
        })
    faces.sort(key=lambda f: FACE_ORDER.index(f["name"]))
    cd = root.GetCustomData()
    photo = cv2.imread(os.path.join(ROOT, spec["photo"]), cv2.IMREAD_COLOR)
    return {
        "name": name,
        "product": spec["product"],
        "label": spec.get("label", name),
        "brand": spec.get("brand", ""),
        "size": spec.get("size", ""),
        "ref": cd.get("ref", spec.get("ref", "")),
        "gtin": cd.get("gtin", spec.get("gtin", "")),
        "measured_mm": spec["dims_mm"],
        "usd_size_mm": size_mm,
        "mass_kg": round(float(mass), 4),
        "mass_source": spec.get("mass_source", ""),
        "evidence": spec.get("evidence", ""),
        "photo": data_uri(photo, 720),
        "photoFile": spec["photo"],
        "faces": faces,
    }, textures


def read_cabinet(path):
    """Cabinet cubes (world AABB + display colour) and carton instances (box name + world
    matrix) from the filled-cabinet scene, if it has been built."""
    if not os.path.exists(path):
        return None
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cubes, cartons = [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cube) and str(prim.GetPath()).startswith("/World/Cabinet/"):
            r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            col = (0.7, 0.7, 0.7)
            if mat:
                for sp in Usd.PrimRange(mat.GetPrim()):
                    sh = UsdShade.Shader(sp)
                    if sh and sh.GetInput("diffuseColor"):
                        col = tuple(sh.GetInput("diffuseColor").Get())
            cubes.append({"min": list(r.GetMin()), "max": list(r.GetMax()), "color": list(col)})
        elif prim.IsInstance():
            refs = [spec.assetPath for spec in prim.GetPrimStack()[0].referenceList.GetAddedOrExplicitItems()] \
                if prim.GetPrimStack() else []
            box = os.path.splitext(os.path.basename(refs[0]))[0] if refs else prim.GetName().rsplit("_", 1)[0]
            m = cache.GetLocalToWorldTransform(prim)
            cartons.append({"box": box, "m": [round(m[i][j], 6) for i in range(4) for j in range(4)]})
    return {"cubes": cubes, "cartons": cartons, "file": os.path.relpath(path, ROOT)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes-yaml", default=os.path.join(ROOT, "boxes.yaml"))
    ap.add_argument("--usd-dir", default=os.path.join(ROOT, "usd"))
    ap.add_argument("--verify-json", required=True)
    ap.add_argument("--shelf-json", default=None, help="verify_shelf.py --json output, shown with the cabinet")
    ap.add_argument("--template", default=os.path.join(HERE, "viewer_template.html"))
    ap.add_argument("--out", default=os.path.join(ROOT, "viewer.html"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.boxes_yaml))
    boxes, textures = [], {}
    for name, spec in cfg["boxes"].items():
        b, t = read_box(name, spec, os.path.abspath(a.usd_dir))
        boxes.append(b)
        textures.update(t)
    checks = json.load(open(a.verify_json))
    data = {"boxes": boxes, "textures": textures, "checks": checks,
            "physicsMaterial": cfg["physics_material"],
            "cabinet": read_cabinet(os.path.join(os.path.abspath(a.usd_dir), "medicine_cabinet_filled.usda"))}
    data["shelfChecks"] = json.load(open(a.shelf_json)) if a.shelf_json and os.path.exists(a.shelf_json) else []
    html = open(a.template).read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print("wrote %s (%.1f MB), %d boxes, %d checks"
          % (a.out, os.path.getsize(a.out) / 1e6, len(boxes), len(checks)))


if __name__ == "__main__":
    main()
