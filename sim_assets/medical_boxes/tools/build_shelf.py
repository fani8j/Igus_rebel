#!/usr/bin/env python3
"""Build usd/medicine_cabinet_filled.usda: the medicine cabinet stocked with carton assets.

Run with the venv python (see README.md):
    "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/build_shelf.py

Reads shelf_layout.yaml (spec), boxes.yaml (dims) and the composed cabinet USD (shelf
geometry, via BBoxCache). Nothing about the shelves is hard-coded.

Packing rule: a shelf's usable width (side-wall inner faces minus side_margin) is split among
its groups by `fill`.  Inside a group the cartons are packed from the group's LEFT edge at a
pitch of thickness + gap_between, so any leftover width sits at the group's RIGHT.  Every group
except the last also reserves one gap_between at its right edge, so neighbouring groups are
always separated by >= gap_between.

Exit code is non-zero (message names shelf and box) when a group does not fit.
"""
import math
import os
import sys

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_util as px  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "usd", "medicine_cabinet_filled.usda")
EPS = 1e-9

# carton +X -> world +Z, +Y -> world -Y, +Z -> world +X (shelf_layout.yaml)
R_CARTON_TO_WORLD = np.array([[0.0, 0.0, 1.0],
                              [0.0, -1.0, 0.0],
                              [1.0, 0.0, 0.0]])  # columns = images of carton x, y, z


# 180 deg about world Y: the other end of the spine goes up and the front faces -X
R_FLIP_Y = np.diag([-1.0, 1.0, -1.0])


def carton_rotation(up_end):
    """Carton -> world rotation for a spine-out carton whose spine end `up_end` is on top:
    "+X" = the end at the left of textures/<box>/top.png, "-X" = its right end."""
    if up_end not in ("+X", "-X"):
        raise ValueError(f"up_end must be '+X' or '-X', got {up_end!r}")
    return R_CARTON_TO_WORLD if up_end == "+X" else R_FLIP_Y @ R_CARTON_TO_WORLD


class FitError(Exception):
    pass


def cabinet_geometry(cab_path, cab_prim):
    """Read all Cube prims and per-shelf geometry from the composed cabinet."""
    st = Usd.Stage.Open(cab_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cubes = {}
    rel_paths = {}
    for p in st.Traverse():
        if p.IsA(UsdGeom.Cube):
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            cubes[p.GetName()] = (np.array(r.GetMin()), np.array(r.GetMax()))
            rel_paths[p.GetName()] = str(p.GetPath())[len(cab_prim):]
    top = max(mx[2] for _, mx in cubes.values())
    left_in = cubes["Body_Left"][1][0]
    right_in = cubes["Body_Right"][0][0]
    back_front_y = cubes["Body_Back"][0][1]
    return cubes, rel_paths, top, left_in, right_in, back_front_y


def shelf_info(name, cubes, cab_top, back_front_y):
    mn, mx = cubes[name]
    top = mx[2]
    # clear height: nearest obstruction above that overlaps the shelf's footprint
    ceil = cab_top
    for n, (a, b) in cubes.items():
        if n == name or a[2] <= top + 1e-3:
            continue
        if a[0] < mx[0] - EPS and b[0] > mn[0] + EPS and a[1] < mx[1] - EPS and b[1] > mn[1] + EPS:
            ceil = min(ceil, a[2])
    return dict(top=top, front_y=mn[1], back_limit=min(mx[1], back_front_y), clear=ceil - top)


def plan_shelf(name, groups, info, x_lo, x_hi, boxes, pl):
    """Return list of (box, x_left) placements plus per-group stats."""
    gap, setback = pl["gap_between_m"], pl["spine_setback_m"]
    width = x_hi - x_lo
    total_fill = sum(g["fill"] for g in groups)
    placed, stats = [], []
    x = x_lo
    for gi, g in enumerate(groups):
        d = boxes[g["box"]]["dims_mm"]
        h, ln, t = d["height"] / 1000, d["length"] / 1000, d["thickness"] / 1000
        gw = width * g["fill"] / total_fill
        avail = gw - (gap if gi < len(groups) - 1 else 0.0)
        if pl["drop_height_m"] + h + pl["min_top_clearance_m"] > info["clear"] + EPS:
            raise FitError(f"{name}/{g['box']}: height {h*1000:.0f} mm + clearance "
                           f"{pl['min_top_clearance_m']*1000:.0f} mm exceeds clear height "
                           f"{info['clear']*1000:.0f} mm")
        back = info["front_y"] + setback + ln
        if back > info["back_limit"] + EPS:
            raise FitError(f"{name}/{g['box']}: carton back y={back:.4f} goes past back limit "
                           f"{info['back_limit']:.4f}")
        n = int(math.floor((avail + gap) / (t + gap) + 1e-9))
        if n < 1:
            raise FitError(f"{name}/{g['box']}: group width {gw*1000:.0f} mm cannot fit one "
                           f"carton ({t*1000:.0f} mm)")
        for i in range(n):
            placed.append((g["box"], x + i * (t + gap)))
        used = n * t + (n - 1) * gap
        stats.append(dict(box=g["box"], n=n, leftover=gw - used))
        x += gw
    return placed, stats


def main():
    spec = yaml.safe_load(open(os.path.join(ROOT, "shelf_layout.yaml")))
    boxes = yaml.safe_load(open(os.path.join(ROOT, "boxes.yaml")))
    pl = spec["placement"]
    bx = boxes["boxes"]
    cab_rel = spec["cabinet"]
    cab_prim = spec["cabinet_prim"]
    cab_abs = os.path.normpath(os.path.join(ROOT, "usd", cab_rel))
    cubes, rel_paths, cab_top, left_in, right_in, back_front_y = cabinet_geometry(cab_abs, cab_prim)
    x_lo, x_hi = left_in + pl["side_margin_m"], right_in - pl["side_margin_m"]

    # ---- plan everything first so failures leave no half-written scene
    plans = {}
    try:
        for sh, groups in spec["shelves"].items():
            info = shelf_info(sh, cubes, cab_top, back_front_y)
            plans[sh] = (info,) + plan_shelf(sh, groups, info, x_lo, x_hi, bx, pl)
    except FitError as e:
        print(f"FIT ERROR: {e}", file=sys.stderr)
        return 2

    # ---- stage
    if os.path.exists(OUT):
        os.remove(OUT)
    st = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px.author_scene(scene.GetPrim(), boxes["physics_scene"])

    ground = UsdGeom.Cube.Define(st, "/World/GroundPlane")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0.2, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    light = UsdLux.DomeLight.Define(st, "/World/DomeLight")
    light.CreateIntensityAttr(1000.0)

    # painted-steel physics material for the cabinet
    UsdGeom.Scope.Define(st, "/World/Physics")
    mat = UsdShade.Material.Define(st, "/World/Physics/painted_steel")
    px.author_material(mat.GetPrim(), spec["shelf_material"])
    floor = UsdShade.Material.Define(st, "/World/Physics/floor")
    px.author_material(floor.GetPrim(), boxes["floor_material"])
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        floor, UsdShade.Tokens.weakerThanDescendants, "physics")

    cab = UsdGeom.Xform.Define(st, "/World/Cabinet")
    cab.GetPrim().GetReferences().AddReference(Sdf.Reference(cab_rel, Sdf.Path(cab_prim)))
    UsdShade.MaterialBindingAPI.Apply(cab.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    for name, rel in rel_paths.items():
        ov = st.OverridePrim("/World/Cabinet" + rel)
        UsdPhysics.CollisionAPI.Apply(ov)
        # the cabinet file binds its looks without applying MaterialBindingAPI; declare it
        # here so strict USD readers don't warn (the cabinet file itself stays untouched)
        UsdShade.MaterialBindingAPI.Apply(ov)

    # ---- cartons
    UsdGeom.Scope.Define(st, "/World/Cartons")
    up_ends = spec.get("up_end", {}) or {}
    quats = {b: Gf.Matrix3d(*carton_rotation(up_ends.get(b, "+X")).T.flatten()).ExtractRotation().GetQuat()
             for b in bx}  # row-vector convention
    total = 0
    lines = []
    for sh, (info, placed, stats) in plans.items():
        UsdGeom.Scope.Define(st, f"/World/Cartons/{sh}")
        counters = {}
        max_h = 0.0
        for box, xl in placed:
            d = bx[box]["dims_mm"]
            h, ln, t = d["height"] / 1000, d["length"] / 1000, d["thickness"] / 1000
            max_h = max(max_h, h)
            counters[box] = counters.get(box, 0) + 1
            p = st.DefinePrim(f"/World/Cartons/{sh}/{box}_{counters[box]:03d}", "Xform")
            p.GetReferences().AddReference(Sdf.Reference(f"./{box}.usda"))
            p.SetInstanceable(True)
            ctr = Gf.Vec3d(xl + t / 2,
                           info["front_y"] + pl["spine_setback_m"] + ln / 2,
                           info["top"] + pl["drop_height_m"] + h / 2)
            m = Gf.Matrix4d(1.0)
            m.SetRotateOnly(Gf.Rotation(quats[box]))
            m.SetTranslateOnly(ctr)
            UsdGeom.Xformable(p).AddTransformOp().Set(m)
        n = len(placed)
        total += n
        gtxt = ", ".join(f"{s['n']}x {s['box']} (leftover {s['leftover']*1000:.1f} mm)" for s in stats)
        lines.append(f"{sh}: {n} cartons [{gtxt}]; clear {info['clear']*1000:.0f} mm, "
                     f"clearance above tallest {(info['clear']-max_h-pl['drop_height_m'])*1000:.1f} mm")
    st.GetRootLayer().Save()
    print(f"Wrote {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)")
    print(f"usable x range: {x_lo:.3f} .. {x_hi:.3f} m (width {(x_hi-x_lo)*1000:.0f} mm)")
    print("\n".join(lines))
    print(f"total cartons: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
