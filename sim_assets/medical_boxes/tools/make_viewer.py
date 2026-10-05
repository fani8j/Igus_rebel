#!/usr/bin/env python3
"""Write viewer.html: a self-contained page with an interactive 3D view of every carton
and its spec / verification summary.

Geometry, UVs, face textures, mass and customData are read back out of usd/<box>.usda
(not from boxes.yaml), so the page shows what Isaac Sim loads. boxes.yaml supplies the
caliper measurement, texture provenance and evidence notes; verify_usd.py --json supplies
the check results.

    "$PY" tools/verify_usd.py --json /tmp/verify.json
    "$PY" tools/make_viewer.py --verify-json /tmp/verify.json

Catalogue page (about 45 families, no cabinet, size matrices from catalog/sizes.yaml):
    "$PY" tools/make_viewer.py --catalog --boxes-yaml catalog/catalog_boxes.yaml --usd-dir catalog/usd \
        --sizes-yaml catalog/sizes.yaml --verify-json /tmp/catverify.json --out catalog/catalog_viewer.html
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np
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


def photo_crop(img, corners, max_side=520):
    """Context crop of the exemplar photo around the spine quad, with the quad outlined.
    The setup photos show a whole shelf, so catalogue pages crop to the box of interest."""
    pts = np.array(corners, float)
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = max((hi - lo).max() * 0.85, 240.0)
    h, w = img.shape[:2]
    x0, x1 = int(max(0, c[0] - half)), int(min(w, c[0] + half))
    y0, y1 = int(max(0, c[1] - half)), int(min(h, c[1] + half))
    out = img.copy()
    cv2.polylines(out, [pts.round().astype(np.int32).reshape(-1, 1, 2)], True, (255, 200, 0),
                  max(2, int(half / 120)), cv2.LINE_AA)
    return out[y0:y1, x0:x1], max_side


def is_flat(img, tol=3.0):
    return float(img.reshape(-1, img.shape[-1]).std(0).max()) < tol


def read_box(name, spec, usd_dir, tex_max=1024, catalog=False):
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
        if catalog and fname != "top" and is_flat(img):
            mean = img.reshape(-1, 3).mean(0).round().astype(np.uint8)  # plain colour: no resolution needed
            textures[key] = data_uri(np.tile(mean, (2, 2, 1)), 8, png=True)
        else:
            textures[key] = data_uri(img, tex_max, png=small or fname == "top")
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
    photo_max = 720
    if catalog and spec.get("photo_corners_px"):
        photo, photo_max = photo_crop(photo, spec["photo_corners_px"])
    box = {
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
        "photo": data_uri(photo, photo_max),
        "photoFile": spec["photo"],
        "faces": faces,
    }
    for k in ("device_type", "dims_confidence", "ref_example"):  # catalogue-only fields
        if k in spec:
            box[k] = spec[k]
    return box, textures


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


def read_sizes(path, names):
    """Compact per-box size data from catalog/sizes.yaml (only the boxes shown on the page)."""
    fams = yaml.safe_load(open(path))["families"]
    out = {}
    for n in names:
        e = fams.get(n)
        if not e:
            continue
        av = e["availability"]
        out[n] = {
            "sku": e["sku_count"], "kind": e["sku_count_kind"], "note": e.get("sku_count_note", ""),
            "dia": e["diameters_mm"], "len": e["lengths_mm"], "shaft": e["shaft_lengths_cm"],
            "avail": av["kind"], "fields": av.get("combination_fields") or [],
            "combos": av.get("combinations") or [], "basis": av.get("basis", ""),
            "rules": av.get("matrix_notes", ""), "incomplete": e.get("incomplete_reason", "") if e.get("incomplete") else "",
            "conf": e["confidence"], "other": e.get("other", ""), "refScheme": e["ref_scheme"],
            "exampleRefs": e["example_refs"], "sources": e["sources"], "seen": e["sizes_seen_on_shelf"],
        }
    return out


CATALOG_META = {
    "title": "Medical Carton Catalogue",
    "intro": "Estimated cartons for the catheter, balloon, stent and guidewire families seen on the hospital shelf photos, "
             "built as USD assets for Isaac Sim, with each family's catalogued size range. Geometry, face textures, mass "
             "and verifier results are read back from the generated .usda files.",
    "dimLabel": "Estimated mm",
    "evidenceLabel": "Evidence",
    "estimated": True,
    "source": [
        "Dimensions are ESTIMATES: spine length and thickness are measured on the shelf photos and scaled against the "
        "five caliper-measured cartons; carton length is 1.1 x spine length. No manufacturer or regulator database "
        "publishes unit-carton dimensions or weights. The confidence of each estimate is shown on the sheet.",
        "Mass is board (outer area x 1.2 x 320 g/m2) plus typical contents for the device type. Replace it in "
        "catalog_boxes.yaml with a scale reading when a real carton is weighed.",
    ],
    "open": [
        "File > Open sim_assets/medical_boxes/catalog/usd/medical_boxes_scene.usda and press Play. The scene lays all "
        "cartons out in a grid (tall guiding-catheter cartons set the row depth). To use one carton, add a reference to "
        "catalog/usd/<box>.usda and keep catalog/textures/ next to catalog/usd/.",
        "The labelled spine is the +Y face. Only the spine is textured from photos; other faces are plain colour until "
        "operator photos exist.",
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes-yaml", default=os.path.join(ROOT, "boxes.yaml"))
    ap.add_argument("--usd-dir", default=os.path.join(ROOT, "usd"))
    ap.add_argument("--verify-json", required=True)
    ap.add_argument("--shelf-json", default=None, help="verify_shelf.py --json output, shown with the cabinet")
    ap.add_argument("--template", default=os.path.join(HERE, "viewer_template.html"))
    ap.add_argument("--out", default=os.path.join(ROOT, "viewer.html"))
    ap.add_argument("--sizes-yaml", default=None, help="catalog/sizes.yaml: adds a size matrix and SKU count to each sheet")
    ap.add_argument("--catalog", action="store_true",
                    help="catalogue page: filterable grouped box strip, estimate wording, cropped exemplar photos, no cabinet")
    ap.add_argument("--tex-max", type=int, default=None,
                    help="longest side of embedded textures in px (default 1024; 768 with --catalog)")
    a = ap.parse_args()
    if a.tex_max is None:
        a.tex_max = 768 if a.catalog else 1024
    cfg = yaml.safe_load(open(a.boxes_yaml))
    boxes, textures = [], {}
    for name, spec in cfg["boxes"].items():
        b, t = read_box(name, spec, os.path.abspath(a.usd_dir), a.tex_max, a.catalog)
        boxes.append(b)
        textures.update(t)
    checks = json.load(open(a.verify_json))
    data = {"boxes": boxes, "textures": textures, "checks": checks,
            "physicsMaterial": cfg["physics_material"],
            "cabinet": None if a.catalog else
            read_cabinet(os.path.join(os.path.abspath(a.usd_dir), "medicine_cabinet_filled.usda"))}
    data["shelfChecks"] = json.load(open(a.shelf_json)) if a.shelf_json and os.path.exists(a.shelf_json) else []
    if a.sizes_yaml:
        data["sizes"] = read_sizes(a.sizes_yaml, [b["name"] for b in boxes])
    if a.catalog:
        data["meta"] = CATALOG_META
    html = open(a.template).read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print("wrote %s (%.1f MB), %d boxes, %d checks"
          % (a.out, os.path.getsize(a.out) / 1e6, len(boxes), len(checks)))


if __name__ == "__main__":
    main()
