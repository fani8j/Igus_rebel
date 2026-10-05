#!/usr/bin/env python3
"""Verify usd/medicine_cabinet_filled.usda from the composed stage and render a front elevation.

    "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/verify_shelf.py [--json PATH]

Everything is read back from the composed USD (BBoxCache, mesh points/UVs, world transforms);
only the expected carton counts are recomputed from shelf_layout.yaml.  Writes
usd/previews/cabinet_front.png.  Exit code is non-zero when any check FAILs.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_util as px  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "usd", "medicine_cabinet_filled.usda")
PREVIEW = os.path.join(ROOT, "usd", "previews", "cabinet_front.png")
EPS = 1e-9


class Results:
    def __init__(self):
        self.rows = []

    def add(self, where, check, ok, detail=""):
        self.rows.append((where, check, bool(ok), detail))

    @property
    def n_fail(self):
        return sum(1 for r in self.rows if not r[2])

    def table(self):
        w1 = max(len(r[0]) for r in self.rows)
        w2 = max(len(r[1]) for r in self.rows)
        return "\n".join(f"{'PASS' if ok else 'FAIL'}  {w:<{w1}}  {c:<{w2}}  {d}"
                         for w, c, ok, d in self.rows)


def rng(cache, prim):
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def face_of(mesh, name):
    """World points (4,3) and uvs (4,2) of the single-face GeomSubset `name`."""
    mesh = UsdGeom.Mesh(mesh)
    pts = np.array(mesh.GetPointsAttr().Get(), float)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    idx = list(mesh.GetFaceVertexIndicesAttr().Get())
    uvs = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(), float)
    offs = np.concatenate([[0], np.cumsum(counts)])
    sub = next(s for s in UsdGeom.Subset.GetAllGeomSubsets(mesh) if s.GetPrim().GetName() == name)
    f = list(sub.GetIndicesAttr().Get())[0]
    vi = idx[offs[f]:offs[f + 1]]
    m = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0.0))  # row-vector convention
    p = np.hstack([pts[vi], np.ones((4, 1))]) @ m
    return p[:, :3], uvs[offs[f]:offs[f + 1]]


def uv_dirs(points, uv):
    M = np.hstack([uv, np.ones((len(uv), 1))])
    sol, *_ = np.linalg.lstsq(M, points, rcond=None)
    return sol[0] / np.linalg.norm(sol[0]), sol[1] / np.linalg.norm(sol[1])


def normal_of(points):
    n = np.cross(points[1] - points[0], points[2] - points[0])
    return n / np.linalg.norm(n)


def texture_path(carton, face):
    """Resolved diffuse texture of `face` on a carton (through instance proxies)."""
    sh = carton.GetChild("Looks").GetChild(face).GetChild("diffuseTexture")
    a = UsdShade.Shader(sh).GetInput("file").Get()
    return a.resolvedPath


def pack_expected(layout, boxes, geom):
    """Independent recomputation of carton counts per shelf."""
    pl = layout["placement"]
    gap = pl["gap_between_m"]
    width = geom["x_hi"] - geom["x_lo"]
    out = {}
    for sh, groups in layout["shelves"].items():
        tf = sum(g["fill"] for g in groups)
        counts = []
        for i, g in enumerate(groups):
            t = boxes[g["box"]]["dims_mm"]["thickness"] / 1000
            avail = width * g["fill"] / tf - (gap if i < len(groups) - 1 else 0)
            counts.append((g["box"], int(math.floor((avail + gap) / (t + gap) + 1e-9))))
        out[sh] = counts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="preview px per mm")
    ap.add_argument("--preview", default=PREVIEW)
    a = ap.parse_args()
    res = Results()
    layout = yaml.safe_load(open(os.path.join(ROOT, "shelf_layout.yaml")))
    bcfg = yaml.safe_load(open(os.path.join(ROOT, "boxes.yaml")))
    boxes = bcfg["boxes"]
    pl = layout["placement"]

    # ---- stage / dependencies
    stage = Usd.Stage.Open(SCENE)
    res.add("stage", "opens", stage is not None)
    errs = []
    try:
        errs = [str(e) for e in stage.GetCompositionErrors()]
    except AttributeError:
        pass
    res.add("stage", "no composition errors", not errs, "; ".join(errs[:3]))
    res.add("stage", "z-up, metres, defaultPrim /World",
            UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
            and str(stage.GetDefaultPrim().GetPath()) == "/World")
    scn = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    g = UsdPhysics.Scene(scn[0]).GetGravityDirectionAttr().Get() if scn else None
    res.add("stage", "physics scene gravity -Z 9.81",
            bool(scn) and abs(g[2] + 1) < 1e-6 and abs(UsdPhysics.Scene(scn[0]).GetGravityMagnitudeAttr().Get() - 9.81) < 1e-6)
    ok, det = px.check_scene(scn[0], bcfg["physics_scene"]) if scn else (False, "no PhysicsScene")
    res.add("stage", "PhysxSceneAPI settings", ok, det)
    gp = stage.GetPrimAtPath("/World/GroundPlane")
    fm, _ = UsdShade.MaterialBindingAPI(gp).ComputeBoundMaterial("physics")
    ok, det = px.check_material(fm.GetPrim() if fm else None, bcfg["floor_material"])
    res.add("stage", "ground floor material", ok, det)
    res.add("stage", "ground collider at z=0",
            bool(gp) and gp.HasAPI(UsdPhysics.CollisionAPI)
            and abs(UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(gp).ComputeAlignedRange().GetMax()[2]) < 1e-9)
    _, unres, = [], []
    try:
        layers, assets, unres = UsdUtils.ComputeAllDependencies(SCENE)
    except Exception as e:  # noqa
        layers, assets, unres = [], [], [str(e)]
    res.add("stage", "all dependencies resolve", not unres,
            f"{len(layers)} layers, {len(assets)} assets, unresolved={list(unres)[:3]}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # ---- cabinet
    cab = stage.GetPrimAtPath("/World/Cabinet")
    cubes = {}
    n_cube = n_col = n_rb = 0
    for p in Usd.PrimRange(cab):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            n_rb += 1
        if p.IsA(UsdGeom.Cube):
            n_cube += 1
            n_col += p.HasAPI(UsdPhysics.CollisionAPI)
            cubes[p.GetName()] = rng(cache, p)
    res.add("cabinet", "every Cube has CollisionAPI", n_cube > 0 and n_col == n_cube, f"{n_col}/{n_cube}")
    res.add("cabinet", "no RigidBodyAPI (static)", n_rb == 0, f"{n_rb} found")
    ref = cab.GetPrimStack()[0].referenceList.prependedItems if cab.GetPrimStack() else []
    res.add("cabinet", "relative reference path",
            any(not os.path.isabs(r.assetPath) and r.assetPath == layout["cabinet"] for r in ref),
            ", ".join(r.assetPath for r in ref))
    mb = UsdShade.MaterialBindingAPI(cab).GetDirectBindingRel("physics").GetTargets()
    res.add("cabinet", "physics material bound", len(mb) == 1, str(mb))
    cm, _ = UsdShade.MaterialBindingAPI(cab).ComputeBoundMaterial("physics")
    ok, det = px.check_material(cm.GetPrim() if cm else None, layout["shelf_material"])
    res.add("cabinet", "shelf material values + combine mode", ok, det)
    # the pair resolves to exactly the shelf values: shelf combine mode must outrank carton's
    prio = {"average": 0, "min": 1, "multiply": 2, "max": 3}
    sm, cmat = layout["shelf_material"], bcfg["physics_material"]
    res.add("cabinet", "carton/shelf pair resolves to shelf friction",
            prio[sm["friction_combine_mode"]] > prio[cmat["friction_combine_mode"]]
            and sm["friction_combine_mode"] == "min"
            and sm["static_friction"] <= cmat["static_friction"]
            and sm["dynamic_friction"] <= cmat["dynamic_friction"],
            f"shelf {sm['friction_combine_mode']} {sm['static_friction']}/{sm['dynamic_friction']} "
            f"vs carton {cmat['friction_combine_mode']} {cmat['static_friction']}/{cmat['dynamic_friction']}")

    x_lo = cubes["Body_Left"][1][0] + pl["side_margin_m"]
    x_hi = cubes["Body_Right"][0][0] - pl["side_margin_m"]
    back_y = cubes["Body_Back"][0][1]
    cab_top = max(b[1][2] for b in cubes.values())

    def shelf_geo(name):
        mn, mx = cubes[name]
        ceil = cab_top
        for n, (a_, b_) in cubes.items():
            if n != name and a_[2] > mx[2] + 1e-3 and a_[0] < mx[0] and b_[0] > mn[0] \
                    and a_[1] < mx[1] and b_[1] > mn[1]:
                ceil = min(ceil, a_[2])
        return dict(top=mx[2], xmin=mn[0], xmax=mx[0], front=mn[1], back=min(mx[1], back_y), ceil=ceil)

    # ---- cartons
    root = stage.GetPrimAtPath("/World/Cartons")
    per_shelf = {}
    for sp in root.GetChildren():
        per_shelf[sp.GetName()] = sorted(sp.GetChildren(), key=lambda c: c.GetName())
    expected = pack_expected(layout, boxes, dict(x_lo=x_lo, x_hi=x_hi))
    all_cartons = []
    UP = np.array([0, 0, 1.0])
    for sh, cartons in per_shelf.items():
        geo = shelf_geo(sh)
        boxes_ab = []
        for c in cartons:
            name = c.GetName()
            where = f"{sh}/{name}"
            bname = name.rsplit("_", 1)[0]
            ok_ref = bname in boxes and c.IsInstance() and c.IsInstanceable()
            res.add(where, "instanceable, known box", ok_ref)
            res.add(where, "RigidBodyAPI+MassAPI on root",
                    c.HasAPI(UsdPhysics.RigidBodyAPI) and c.HasAPI(UsdPhysics.MassAPI))
            mesh = c.GetChild("geom")
            ok, det = px.check_collision(mesh, bcfg["collision"]) if mesh else (False, "no geom")
            res.add(where, "mesh collider reachable", ok, det)
            ok, det = px.check_rigid_body(c, bcfg["rigid_body"])
            res.add(where, "PhysxRigidBodyAPI via prototype", ok, det)
            cmm, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics") if mesh else (None, None)
            ok, det = px.check_material(cmm.GetPrim() if cmm else None, bcfg["physics_material"])
            res.add(where, "carton material via prototype", ok, det)
            mn, mx = rng(cache, c)
            d = boxes[bname]["dims_mm"]
            h, ln, t = d["height"] / 1000, d["length"] / 1000, d["thickness"] / 1000
            tol = 1e-6
            size_ok = np.allclose(mx - mn, [t, ln, h], atol=tol)
            in_x = mn[0] >= geo["xmin"] - tol and mx[0] <= geo["xmax"] + tol
            in_y = mn[1] >= geo["front"] - tol and mx[1] <= geo["back"] + tol
            res.add(where, "AABB size = thickness x length x height", size_ok,
                    "x".join(f"{v*1000:.2f}" for v in (mx - mn)))
            res.add(where, "within shelf x/y range", in_x and in_y,
                    f"x[{mn[0]:.4f},{mx[0]:.4f}] y[{mn[1]:.4f},{mx[1]:.4f}]")
            dz = mn[2] - (geo["top"] + pl["drop_height_m"])
            res.add(where, "bottom = shelf_top + drop (+-0.1 mm)", abs(dz) <= 1e-4, f"{dz*1e3:+.4f} mm")
            clr = geo["ceil"] - mx[2]
            res.add(where, "top clearance >= min", clr >= pl["min_top_clearance_m"] - 1e-6, f"{clr*1000:.1f} mm")
            res.add(where, "spine setback", abs(mn[1] - (geo["front"] + pl["spine_setback_m"])) <= 1e-4,
                    f"{(mn[1]-geo['front'])*1000:.2f} mm")
            hit = [n for n, (a_, b_) in cubes.items()
                   if np.all(mn < b_ - 1e-6) and np.all(mx > a_ + 1e-6)]
            res.add(where, "no cabinet Cube intersection", not hit, ",".join(hit))
            # orientation
            top_pts, top_uv = face_of(mesh, "top")
            n_top = normal_of(top_pts)
            u, v = uv_dirs(top_pts, top_uv)
            front_pts, _ = face_of(mesh, "front")
            n_front = normal_of(front_pts)
            # expected world directions: spine out of the cabinet (-Y); the spine end named by
            # up_end on top ("+X" = left end of top.png, so image right u points down)
            up = (layout.get("up_end", {}) or {}).get(bname, "+X")
            sgn = 1.0 if up == "+X" else -1.0
            exp_u, exp_v = np.array([0, 0, -sgn]), np.array([sgn, 0, 0])
            ok_o = (np.allclose(n_top, [0, -1, 0], atol=1e-6) and np.allclose(u, exp_u, atol=1e-6)
                    and np.allclose(v, exp_v, atol=1e-6) and np.allclose(n_front, exp_v, atol=1e-6))
            res.add(where, f"orientation ({up} end up): spine n=-Y, u={'-' if sgn > 0 else '+'}Z, "
                    f"v=front n={'+' if sgn > 0 else '-'}X", ok_o,
                    f"n={np.round(n_top,3)} u={np.round(u,3)} v={np.round(v,3)} front={np.round(n_front,3)}")
            boxes_ab.append((name, mn, mx))
            all_cartons.append((c, bname, top_pts, top_uv))
        # neighbours: sort by x and check gaps
        boxes_ab.sort(key=lambda r: r[1][0])
        gap_bad = [(boxes_ab[i][0], boxes_ab[i + 1][0], (boxes_ab[i + 1][1][0] - boxes_ab[i][2][0]) * 1000)
                   for i in range(len(boxes_ab) - 1)
                   if boxes_ab[i + 1][1][0] - boxes_ab[i][2][0] < pl["gap_between_m"] - 1e-6]
        res.add(sh, "no carton AABB overlap, gaps >= gap_between", not gap_bad,
                f"{len(boxes_ab)} cartons" + (f"; bad {gap_bad[:2]}" if gap_bad else ""))
        # counts
        got = {}
        for c in cartons:
            b = c.GetName().rsplit("_", 1)[0]
            got[b] = got.get(b, 0) + 1
        exp = {}
        for b, n in expected.get(sh, []):
            exp[b] = exp.get(b, 0) + n
        res.add(sh, "counts match layout", got == exp, f"got {got} expected {exp}")
    res.add("layout", "all layout shelves populated", set(expected) == set(per_shelf),
            f"layout {sorted(expected)} scene {sorted(per_shelf)}")

    # ---- preview
    render(stage, cache, cubes, all_cartons, a.scale, a.preview)
    sz = os.path.getsize(a.preview)
    res.add("preview", "written, < 400 KB", sz < 400 * 1024, f"{a.preview} {sz/1024:.0f} KB")

    print(res.table())
    n = len(res.rows)
    print(f"\n{n - res.n_fail} PASS / {res.n_fail} FAIL of {n} checks; {len(all_cartons)} cartons")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([{"where": w, "check": c, "ok": ok, "detail": d} for w, c, ok, d in res.rows], f, indent=1)
    return 1 if res.n_fail else 0


def render(stage, cache, cubes, cartons, scale, out):
    """Front elevation looking along +Y (image right = +X, up = +Z)."""
    s = scale * 1000.0  # px per metre
    xs = [b[0][0] for b in cubes.values()] + [b[1][0] for b in cubes.values()]
    zs = [b[1][2] for b in cubes.values()]
    x0, x1, z1 = min(xs), max(xs), max(zs)
    pad = 10
    W, H = int((x1 - x0) * s) + 2 * pad, int(z1 * s) + 2 * pad

    def px(x, z):
        return np.array([pad + (x - x0) * s, pad + (z1 - z) * s], np.float32)

    img = np.full((H, W, 3), 255, np.uint8)
    # far to near
    for n, (mn, mx) in sorted(cubes.items(), key=lambda kv: -kv[1][0][1]):
        shade = 150 if "Back" in n else 110 if "Body" in n else 175
        p0, p1 = px(mn[0], mx[2]), px(mx[0], mn[2])
        cv2.rectangle(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                      (shade,) * 3, -1)
        cv2.rectangle(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                      (70, 70, 70), 1)
    cache_tex = {}
    for c, bname, pts, uv in cartons:
        tp = texture_path(c, "top")
        if tp not in cache_tex:
            cache_tex[tp] = cv2.imread(tp, cv2.IMREAD_COLOR)
        tex = cache_tex[tp]
        th, tw = tex.shape[:2]
        # u/v -> world edge lengths in px so the texture is resampled before the rotation
        dst = np.array([px(p[0], p[2]) for p in pts], np.float32)  # image x/z of the 4 world corners
        src = np.array([[u * (tw - 1), (1 - v) * (th - 1)] for u, v in uv], np.float32)
        # shrink the texture to roughly its on-screen size first (better downsampling)
        lu = np.linalg.norm(dst[1] - dst[0]); lv = np.linalg.norm(dst[3] - dst[0])
        nw, nh = max(2, int(round(lu))), max(2, int(round(lv)))
        small = cv2.resize(tex, (nw, nh), interpolation=cv2.INTER_AREA)
        src = np.array([[u * (nw - 1), (1 - v) * (nh - 1)] for u, v in uv], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(small, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
        img[mask > 0] = warped[mask > 0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == "__main__":
    sys.exit(main())
