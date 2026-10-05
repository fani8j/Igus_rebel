#!/usr/bin/env python3
"""Generate Isaac Sim-ready USD assets (geometry, textures, physics) from boxes.yaml.

Run with the venv python (see README.md):
    "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/build_usd.py

Writes <out-dir>/<box>.usda for every box and <out-dir>/medical_boxes_scene.usda.
"""
import argparse
import os
import sys

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_util as px  # noqa: E402

GENERATOR_VERSION = "1.0.0"
MM = 1e-3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Face order is fixed: face index i of the mesh == FACE_ORDER[i].
FACE_ORDER = ["front", "back", "top", "bottom", "left", "right"]

# Face definition (same as boxes.yaml header): outward normal, image-right axis, image-up axis.
# Axis letters: x=height, y=length, z=thickness. Sign given by +1/-1.
FACE_AXES = {
    "front": {"normal": (0, 0, 1), "right": (1, 0, 0), "up": (0, 1, 0)},
    "back": {"normal": (0, 0, -1), "right": (-1, 0, 0), "up": (0, 1, 0)},
    "top": {"normal": (0, 1, 0), "right": (-1, 0, 0), "up": (0, 0, 1)},
    "bottom": {"normal": (0, -1, 0), "right": (1, 0, 0), "up": (0, 0, 1)},
    "right": {"normal": (1, 0, 0), "right": (0, 1, 0), "up": (0, 0, 1)},
    "left": {"normal": (-1, 0, 0), "right": (0, -1, 0), "up": (0, 0, 1)},
}


def _vec(t):
    return Gf.Vec3d(*t)


def face_quad(face, half):
    """Four corners (bl, br, tr, tl) of the face seen from outside, CCW from outside."""
    a = FACE_AXES[face]
    n, r, u = (Gf.Vec3d(*a[k]) for k in ("normal", "right", "up"))
    hn = abs(Gf.Dot(n, half))
    hr = abs(Gf.Dot(r, half))
    hu = abs(Gf.Dot(u, half))
    c = n * hn
    pts = [c - r * hr - u * hu, c + r * hr - u * hu, c + r * hr + u * hu, c - r * hr + u * hu]
    return pts, n


def tex_rel_path(out_dir, textures_root, box, face):
    p = os.path.relpath(os.path.join(textures_root, box, face + ".png"), out_dir)
    return p.replace(os.sep, "/")


def make_preview_material(stage, path, tex_rel):
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    reader = UsdShade.Shader.Define(stage, path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path + "/diffuseTexture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_rel))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.GetOutput("result"))
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.GetOutput("rgb"))
    surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(surf.GetOutput("surface"))
    return mat


def build_box(name, spec, cfg, out_dir, textures_root):
    phys = cfg["physics_material"]
    path = os.path.join(out_dir, name + ".usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    d = spec["dims_mm"]
    dims = Gf.Vec3d(d["height"] * MM, d["length"] * MM, d["thickness"] * MM)  # X, Y, Z
    half = dims * 0.5
    mass = float(spec["mass_kg"])

    root = UsdGeom.Xform.Define(stage, "/" + name)
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind("component")
    rp = root.GetPrim()
    rp.SetDocumentation(f"{spec['product']} carton, box frame: origin at centre, Z up, "
                        "X=height, Y=length, Z=thickness (+Z = front).")

    # --- metadata -----------------------------------------------------------
    info = {
        "product": spec["product"],
        "dims_mm": {"height": float(d["height"]), "length": float(d["length"]),
                    "thickness": float(d["thickness"])},
        "mass_kg": mass,
        "mass_source": spec.get("mass_source", "estimate"),
        "generator_version": GENERATOR_VERSION,
    }
    if "size" in spec:
        info["device_size"] = spec["size"]
    if "ref" in spec:
        info["ref"] = spec["ref"]
    if "gtin" in spec:
        info["gtin"] = spec["gtin"]
    for region, c in (spec.get("catalog_candidates") or {}).items():
        info[f"catalog_{region}"] = c["catalog"]
        info[f"gtin_{region}"] = c["gtin"]
    if "ref" not in info and "catalog_eu" in info:
        info["ref"] = info["catalog_eu"]
    if "gtin" not in info and "gtin_eu" in info:
        info["gtin"] = info["gtin_eu"]
    rp.SetCustomData(info)
    rp.SetAssetInfo({"name": name, "version": GENERATOR_VERSION, "identifier": Sdf.AssetPath(name + ".usda")})

    # --- geometry -----------------------------------------------------------
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/geom")
    points, normals, uvs = [], [], []
    for face in FACE_ORDER:
        quad, n = face_quad(face, half)
        points += [Gf.Vec3f(p) for p in quad]
        normals += [Gf.Vec3f(n)] * 4
        uvs += [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(24))))
    mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-half), Gf.Vec3f(half)]))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray(uvs))

    # --- materials ----------------------------------------------------------
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    for i, face in enumerate(FACE_ORDER):
        mat = make_preview_material(stage, f"/{name}/Looks/{face}",
                                    tex_rel_path(out_dir, textures_root, name, face))
        sub = UsdGeom.Subset.Define(stage, f"/{name}/geom/{face}")
        sub.CreateElementTypeAttr(UsdGeom.Tokens.face)
        sub.CreateIndicesAttr(Vt.IntArray([i]))
        sub.CreateFamilyNameAttr("materialBind")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)

    # --- physics ------------------------------------------------------------
    UsdPhysics.RigidBodyAPI.Apply(rp)
    px.author_rigid_body(rp, cfg["rigid_body"])
    m = UsdPhysics.MassAPI.Apply(rp)
    m.CreateMassAttr(mass)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    # Solid-cuboid inertia (uniform density assumption), principal axes = box axes.
    x, y, z = dims
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass / 12 * (y * y + z * z),
                                         mass / 12 * (x * x + z * z),
                                         mass / 12 * (x * x + y * y)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))

    gp = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(gp)
    mc = UsdPhysics.MeshCollisionAPI.Apply(gp)
    mc.CreateApproximationAttr(cfg["collision"]["approximation"])
    px.author_collision(gp, cfg["collision"])

    UsdGeom.Scope.Define(stage, f"/{name}/Physics")
    pm = UsdShade.Material.Define(stage, f"/{name}/Physics/cardboard")
    px.author_material(pm.GetPrim(), phys)
    UsdShade.MaterialBindingAPI.Apply(gp).Bind(
        pm, UsdShade.Tokens.weakerThanDescendants, "physics")

    stage.GetRootLayer().Save()
    return path


def grid_layout(boxes, specs, cols, gap):
    """Shelf-packed grid for large catalogues: `cols` boxes per row, rows stacked along -Y.
    Each box keeps its own X (height) footprint; the row pitch is the row's longest box (Y) so
    tall cartons (guiding catheters, ~1 m) never touch their neighbours. Returns
    ({name: (x, y)}, half_x, half_y) with the layout centred on the origin."""
    rows = [boxes[i:i + cols] for i in range(0, len(boxes), cols)]
    widths = [sum(specs[n]["dims_mm"]["height"] * MM for n in r) + gap * (len(r) - 1) for r in rows]
    depths = [max(specs[n]["dims_mm"]["length"] * MM for n in r) for r in rows]
    total_y = sum(depths) + gap * (len(rows) - 1)
    pos, y = {}, total_y / 2
    for r, w, dep in zip(rows, widths, depths):
        x = -w / 2
        for n in r:
            h = specs[n]["dims_mm"]["height"] * MM
            pos[n] = (x + h / 2, y - dep / 2)
            x += h + gap
        y -= dep + gap
    return pos, max(widths) / 2, total_y / 2


def build_scene(boxes, specs, out_dir, cfg, grid_cols=0, grid_gap=0.05):
    path = os.path.join(out_dir, "medical_boxes_scene.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    gpos = None
    if grid_cols:
        gpos, hx, hy = grid_layout(boxes, specs, grid_cols, grid_gap)
        world.GetPrim().SetCustomData({"scene_layout": "grid", "grid_cols": int(grid_cols),
                                       "grid_gap_m": float(grid_gap)})

    ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    px.author_scene(ps.GetPrim(), cfg["physics_scene"])

    g = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 2.0
    sx = sy = s
    if gpos:
        sx, sy = max(s, hx + 0.5), max(s, hy + 0.5)
    g.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-sx, -sy, 0), Gf.Vec3f(sx, -sy, 0),
                                       Gf.Vec3f(sx, sy, 0), Gf.Vec3f(-sx, sy, 0)]))
    g.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    g.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    g.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    g.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-sx, -sy, 0), Gf.Vec3f(sx, sy, 0)]))
    g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.58)]))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("none")
    UsdGeom.Scope.Define(stage, "/World/Physics")
    fm = UsdShade.Material.Define(stage, "/World/Physics/floor")
    px.author_material(fm.GetPrim(), cfg["floor_material"])
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(
        fm, UsdShade.Tokens.weakerThanDescendants, "physics")

    spacing = 0.35
    n = len(boxes)
    for i, name in enumerate(boxes):
        prim = stage.DefinePrim(f"/World/{name}", "Xform")
        prim.GetReferences().AddReference(f"./{name}.usda")
        xf = UsdGeom.Xformable(prim)
        thick = specs[name]["dims_mm"]["thickness"] * MM
        if gpos:
            xf.AddTranslateOp().Set(Gf.Vec3d(gpos[name][0], gpos[name][1], thick / 2))
        else:
            xf.AddTranslateOp().Set(Gf.Vec3d((i - (n - 1) / 2) * spacing, 0.0, thick / 2))
    stage.GetRootLayer().Save()
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes-yaml", default=os.path.join(ROOT, "boxes.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "usd"))
    ap.add_argument("--textures-root", default=os.path.join(ROOT, "textures"),
                    help="directory holding <box>/<face>.png (default: repo textures/)")
    ap.add_argument("--scene-grid-cols", type=int, default=0,
                    help="lay the scene out as a shelf-packed grid with this many boxes per row "
                         "(default 0 = the single 0.35 m-spaced row)")
    ap.add_argument("--scene-grid-gap", type=float, default=0.05,
                    help="gap in metres between boxes in the grid layout")
    a = ap.parse_args()

    with open(a.boxes_yaml) as f:
        cfg = yaml.safe_load(f)
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tex_root = os.path.abspath(a.textures_root)
    for name in cfg["boxes"]:
        p = build_box(name, cfg["boxes"][name], cfg, out_dir, tex_root)
        print("wrote", p)
        for face in FACE_ORDER:
            if not os.path.isfile(os.path.join(tex_root, name, face + ".png")):
                print(f"  warning: missing texture {name}/{face}.png under {tex_root}")
    print("wrote", build_scene(list(cfg["boxes"]), cfg["boxes"], out_dir, cfg,
                           a.scene_grid_cols, a.scene_grid_gap))


if __name__ == "__main__":
    sys.exit(main())
