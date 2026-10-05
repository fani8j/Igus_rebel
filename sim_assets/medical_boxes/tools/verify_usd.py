#!/usr/bin/env python3
"""Verify the generated USD assets and render preview images without Isaac Sim.

Run with the venv python (see README.md):
    "/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python" tools/verify_usd.py

Reads everything (mesh, UVs, texture files) back from the USD, checks it against boxes.yaml
and writes <usd-dir>/previews/<box>.png (3D views) and <box>_net.png (unfolded net).
Exit code is non-zero when any check FAILs.
"""
import argparse
import io
import os
import sys

import cv2
import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_ORDER = ["front", "back", "top", "bottom", "left", "right"]
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
# right/up axes per face, from boxes.yaml header (the spec). Parsed from there at runtime.
YAML_FACES = {
    "front": ("+X", "+Y"), "back": ("-X", "+Y"), "top": ("-X", "+Z"),
    "bottom": ("+X", "+Z"), "right": ("+Y", "+Z"), "left": ("-Y", "+Z"),
}
YAML_NORMAL = {"front": "+Z", "back": "-Z", "top": "+Y", "bottom": "-Y", "right": "+X", "left": "-X"}
MAX_PREVIEW_BYTES = 300 * 1024


def axis(s):
    return (1 if s[0] == "+" else -1) * AX[s[1].lower()]


class Results:
    def __init__(self):
        self.rows = []

    def add(self, box, check, ok, detail=""):
        self.rows.append((box, check, bool(ok), detail))

    @property
    def failed(self):
        return any(not r[2] for r in self.rows)

    def table(self):
        w1 = max(len(r[0]) for r in self.rows)
        w2 = max(len(r[1]) for r in self.rows)
        lines = []
        for b, c, ok, d in self.rows:
            lines.append(f"{'PASS' if ok else 'FAIL'}  {b:<{w1}}  {c:<{w2}}  {d}")
        return "\n".join(lines)


def face_data(mesh, tf=None):
    """Return {face: dict(points(4,3), uv(4,2), ...)} read from the USD mesh via its subsets."""
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    idx = list(mesh.GetFaceVertexIndicesAttr().Get())
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uvs = np.array(st.Get(), dtype=float)
    interp = st.GetInterpolation()
    offs = np.concatenate([[0], np.cumsum(counts)])
    out = {}
    for sub in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        name = sub.GetPrim().GetName()
        fids = list(sub.GetIndicesAttr().Get())
        if len(fids) != 1:
            continue
        f = fids[0]
        vi = idx[offs[f]:offs[f + 1]]
        if interp == "faceVarying":
            uv = uvs[offs[f]:offs[f + 1]]
        else:
            uv = uvs[vi]
        out[name] = dict(points=pts[vi], uv=uv, subset=sub, wind=(counts[f], vi))
    return out


def check_orientation_right_handed(mesh):
    return mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.rightHanded


def uv_axes(points, uv):
    """Least-squares 3D direction of +u and +v: P = P0 + u*A + v*B."""
    M = np.hstack([uv, np.ones((len(uv), 1))])
    sol, *_ = np.linalg.lstsq(M, points, rcond=None)
    A, B = sol[0], sol[1]
    return A / np.linalg.norm(A), B / np.linalg.norm(B), np.linalg.norm(sol[0]), np.linalg.norm(sol[1])


def bound_texture(subset, layer_stage):
    """Texture file path (resolved) bound to a subset, and the material path."""
    mat, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
    if not mat:
        return None, None
    surf = mat.ComputeSurfaceSource()[0]
    if not surf:
        return None, mat
    di = surf.GetInput("diffuseColor")
    if not di or not di.HasConnectedSource():
        return None, mat
    src = di.GetConnectedSource()[0]
    tex = UsdShade.Shader(src.GetPrim())
    fa = tex.GetInput("file").Get()
    return (fa.resolvedPath or None, fa.path), mat


# ---------------------------------------------------------------- rendering
def load_tex(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def warp_face(canvas, depth_mask, tex, uv, dst):
    """Draw texture onto canvas. uv: (4,2) per vertex, dst: (4,2) pixel coords."""
    h, w = tex.shape[:2]
    src = np.array([[u * (w - 1), (1 - v) * (h - 1)] for u, v in uv], np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(dst, np.float32))
    H, W = canvas.shape[:2]
    warped = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_AREA)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.asarray(dst)).astype(np.int32), 255)
    canvas[mask > 0] = warped[mask > 0]
    pts = np.round(np.asarray(dst)).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (40, 40, 40), 1, cv2.LINE_AA)


def render_view(faces, textures, cam_dir, size=(700, 700), margin=40):
    """Orthographic render. cam_dir points from the box toward the camera."""
    c = np.asarray(cam_dir, float)
    c /= np.linalg.norm(c)
    fwd = -c
    up_hint = np.array([0, 1.0, 0]) if abs(c[1]) < 0.95 else np.array([1.0, 0, 0])
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    allp = np.vstack([f["points"] for f in faces.values()])
    xy = np.stack([allp @ right, allp @ up], 1)
    lo, hi = xy.min(0), xy.max(0)
    s = min((size[0] - 2 * margin) / (hi[0] - lo[0]), (size[1] - 2 * margin) / (hi[1] - lo[1]))
    canvas = np.full((size[1], size[0], 3), 235, np.uint8)

    def proj(p):
        a = (p @ right - (lo[0] + hi[0]) / 2) * s + size[0] / 2
        b = -(p @ up - (lo[1] + hi[1]) / 2) * s + size[1] / 2
        return np.stack([a, b], 1)

    order = []
    for name, f in faces.items():
        p = f["points"]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        if n @ c <= 1e-6:
            continue  # back-facing
        order.append((float((p.mean(0)) @ fwd) * -1, name))
    # painter's order: far to near (largest depth along fwd first)
    order.sort(key=lambda t: t[0], reverse=True)
    visible = []
    for _, name in order:
        f = faces[name]
        warp_face(canvas, None, textures[name], f["uv"], proj(f["points"]))
        visible.append(name)
    return canvas, visible


def render_net(faces, textures, ppm=1500.0):
    """Unfolded cross-shaped net around the front face, built from the mesh coordinates."""
    allp = np.vstack([f["points"] for f in faces.values()])
    half = (allp.max(0) - allp.min(0)) / 2.0
    hx, hy, hz = half
    T = 2 * hz

    def m2d(name, p):
        x, y, z = p
        if name == "front":
            return x, y
        if name == "top":
            return x, hy + (hz - z)
        if name == "bottom":
            return x, -hy - (hz - z)
        if name == "right":
            return hx + (hz - z), y
        if name == "left":
            return -hx - (hz - z), y
        if name == "back":
            return x, hy + T + (hy - y)
        raise KeyError(name)

    pts2 = {n: np.array([m2d(n, p) for p in f["points"]]) for n, f in faces.items()}
    allx = np.vstack(list(pts2.values()))
    lo, hi = allx.min(0), allx.max(0)
    pad = 30
    W = int((hi[0] - lo[0]) * ppm) + 2 * pad
    H = int((hi[1] - lo[1]) * ppm) + 2 * pad
    canvas = np.full((H, W, 3), 235, np.uint8)
    for n, f in faces.items():
        d = np.stack([(pts2[n][:, 0] - lo[0]) * ppm + pad, (hi[1] - pts2[n][:, 1]) * ppm + pad], 1)
        warp_face(canvas, None, textures[n], f["uv"], d)
        c = d.mean(0)
        cv2.putText(canvas, n, (int(c[0]) - 20, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_small(img, path):
    """Write PNG under MAX_PREVIEW_BYTES (palette quantise, then downscale if needed)."""
    from PIL import Image
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for _ in range(6):
        for colors in (None, 128, 48):
            buf = io.BytesIO()
            (im if colors is None else im.quantize(colors)).save(buf, "PNG", optimize=True)
            if buf.tell() < MAX_PREVIEW_BYTES:
                with open(path, "wb") as fh:
                    fh.write(buf.getvalue())
                return buf.tell()
        im = im.resize((int(im.width * 0.8), int(im.height * 0.8)), Image.LANCZOS)
    raise RuntimeError("cannot shrink preview " + path)


# ---------------------------------------------------------------- checks
def verify_box(name, spec, usd_dir, res, preview_dir):
    path = os.path.join(usd_dir, name + ".usda")
    R = lambda check, ok, detail="": res.add(name, check, ok, detail)
    if not os.path.isfile(path):
        R("file exists", False, path)
        return
    stage = Usd.Stage.Open(path)
    if not stage:
        R("open stage", False)
        return
    # 1. composition / dependencies
    errs = []
    try:
        errs = list(stage.GetCompositionErrors())
    except Exception:
        pass
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path))
    missing = [a for a in assets if not os.path.isfile(a)]
    R("composition errors", not errs, "; ".join(str(e) for e in errs))
    R("dependencies resolved", not unresolved and not missing and len(assets) >= 6,
      f"{len(assets)} assets" + (f", unresolved={list(unresolved)}" if unresolved else "")
      + (f", missing={missing}" if missing else ""))

    # 2. bbox
    d = spec["dims_mm"]
    want = np.array([d["height"], d["length"], d["thickness"]], float)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    got = (np.array(rng.GetMax()) - np.array(rng.GetMin())) * 1000.0
    cen = (np.array(rng.GetMax()) + np.array(rng.GetMin())) * 500.0
    R("bbox == dims_mm (0.05mm)", np.all(np.abs(got - want) <= 0.05),
      f"got {np.round(got, 3).tolist()} want {want.tolist()}")
    R("bbox centred at origin", np.all(np.abs(cen) <= 0.05), f"centre mm {np.round(cen, 3).tolist()}")

    # 3. metadata / physics
    root = stage.GetDefaultPrim()
    ok = (root and root.GetPath() == Sdf.Path("/" + name) and root.IsA(UsdGeom.Xform)
          and Usd.ModelAPI(root).GetKind() == "component")
    R("defaultPrim Xform component", ok)
    R("upAxis Z", UsdGeom.GetStageUpAxis(stage) == "Z")
    R("metersPerUnit 1.0", abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-12)
    R("kilogramsPerUnit 1.0", abs(UsdPhysics.GetStageKilogramsPerUnit(stage) - 1.0) < 1e-12)
    R("RigidBodyAPI on root", root.HasAPI(UsdPhysics.RigidBodyAPI))
    mass = UsdPhysics.MassAPI(root)
    mv = mass.GetMassAttr().Get() if root.HasAPI(UsdPhysics.MassAPI) else None
    R("MassAPI mass", mv is not None and abs(mv - spec["mass_kg"]) < 1e-6, f"mass={mv} want {spec['mass_kg']}")
    com = mass.GetCenterOfMassAttr().Get()
    R("centerOfMass (0,0,0)", com is not None and Gf.Vec3f(com).GetLength() < 1e-9, str(com))
    mesh_prim = stage.GetPrimAtPath(f"/{name}/geom")
    mesh = UsdGeom.Mesh(mesh_prim)
    R("CollisionAPI + convexHull", mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
      and mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
      and UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == "convexHull")
    pm, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial("physics")
    pok = False
    detail = "no physics material bound"
    if pm:
        api = UsdPhysics.MaterialAPI(pm.GetPrim())
        vals = (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get())
        # compare to yaml
        ypm = spec["_physics_material"]
        pok = pm.GetPrim().HasAPI(UsdPhysics.MaterialAPI) and np.allclose(
            vals, [ypm["static_friction"], ypm["dynamic_friction"], ypm["restitution"]], atol=1e-6)
        detail = f"{pm.GetPath()} sf/df/e={tuple(round(v, 4) for v in vals)}"
    R("physics material bound", pok, detail)
    ci = root.GetCustomData()
    need = {"product", "dims_mm", "mass_source", "generator_version"}
    R("customData", need <= set(ci) and ci.get("mass_source") == "estimate"
      and ("ref" in ci or "gtin" in ci), f"keys={sorted(ci)}")

    # geometry
    pts = mesh.GetPointsAttr().Get()
    R("24 verts / 6 quads", len(pts) == 24 and list(mesh.GetFaceVertexCountsAttr().Get()) == [4] * 6)
    R("orientation rightHanded", check_orientation_right_handed(mesh))
    R("extent set", mesh.GetExtentAttr().HasAuthoredValue())
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    R("primvars:st faceVarying", bool(st) and st.GetInterpolation() == "faceVarying")
    faces = face_data(mesh)
    R("6 subsets named", set(faces) == set(FACE_ORDER), str(sorted(faces)))
    if set(faces) != set(FACE_ORDER):
        return

    centre = np.array(pts, float).mean(0)
    textures = {}
    for fn in FACE_ORDER:
        f = faces[fn]
        p = f["points"]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        outward = (p.mean(0) - centre) @ n
        want_n = axis(YAML_NORMAL[fn])
        R(f"normal outward [{fn}]", outward > 0 and n @ want_n > 0.999,
          f"winding n={np.round(n, 3).tolist()} dot(centre dir)={outward * 1000:.2f}mm")
        # authored normals agree with winding
        nn = mesh.GetNormalsAttr().Get()
        if nn is not None and len(nn) == 24:
            vi = f["wind"][1]
            R(f"authored normals [{fn}]", all(np.dot(nn[i], n) > 0.999 for i in vi))
        # UV orientation
        A, B, la, lb = uv_axes(p, f["uv"])
        r_want, u_want = axis(YAML_FACES[fn][0]), axis(YAML_FACES[fn][1])
        R(f"UV axes [{fn}]", A @ r_want > 0.9999 and B @ u_want > 0.9999,
          f"+u->{np.round(A, 2).tolist()} (want {r_want.astype(int).tolist()}), "
          f"+v->{np.round(B, 2).tolist()} (want {u_want.astype(int).tolist()})")
        # texture binding
        tb, mat = bound_texture(f["subset"], stage)
        tex_ok = tb is not None and tb[0] and os.path.isfile(tb[0])
        R(f"texture bound [{fn}]", tex_ok, (tb[1] if tb else "no binding"))
        if tex_ok:
            im = load_tex(tb[0])
            textures[fn] = im
            if im is None:
                R(f"texture readable [{fn}]", False, tb[0])
                continue
            # physical aspect (u extent / v extent) vs pixel aspect
            phys = la / lb
            pix = im.shape[1] / im.shape[0]
            R(f"texture aspect [{fn}]", abs(phys - pix) / phys < 0.05,
              f"{im.shape[1]}x{im.shape[0]}px ratio {pix:.3f} vs surface {phys:.3f}")
    if len(textures) == 6 and not any(t is None for t in textures.values()):
        os.makedirs(preview_dir, exist_ok=True)
        v1, vis1 = render_view(faces, textures, (0.55, 0.65, 1.0))
        v2, vis2 = render_view(faces, textures, (-0.55, -0.65, -1.0))
        cv2.putText(v1, "front/top/right: " + ",".join(vis1), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(v2, "back/bottom/left: " + ",".join(vis2), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        both = np.hstack([v1, v2])
        both = cv2.resize(both, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        p1 = os.path.join(preview_dir, name + ".png")
        p2 = os.path.join(preview_dir, name + "_net.png")
        s1 = save_small(both, p1)
        s2 = save_small(render_net(faces, textures), p2)
        R("previews written", True, f"{p1} ({s1 // 1024} KB), {p2} ({s2 // 1024} KB)")
    else:
        R("previews written", False, "textures missing/unreadable")


def verify_scene(names, specs, usd_dir, res):
    path = os.path.join(usd_dir, "medical_boxes_scene.usda")
    R = lambda check, ok, detail="": res.add("scene", check, ok, detail)
    if not os.path.isfile(path):
        R("file exists", False, path)
        return
    stage = Usd.Stage.Open(path)
    try:
        errs = list(stage.GetCompositionErrors())
    except Exception:
        errs = []
    R("composition errors", not errs, "; ".join(map(str, errs)))
    R("upAxis Z / m / kg", UsdGeom.GetStageUpAxis(stage) == "Z"
      and UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0)
    ps = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    g = UsdPhysics.Scene(ps[0]) if ps else None
    R("PhysicsScene gravity -Z 9.81", bool(g) and abs(g.GetGravityMagnitudeAttr().Get() - 9.81) < 1e-6
      and Gf.Vec3f(g.GetGravityDirectionAttr().Get()) == Gf.Vec3f(0, 0, -1))
    gr = stage.GetPrimAtPath("/World/Ground")
    R("ground collider", gr and gr.HasAPI(UsdPhysics.CollisionAPI))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xs = []
    for n in names:
        p = stage.GetPrimAtPath(f"/World/{n}")
        if not p:
            R(f"box {n} present", False)
            continue
        rng = cache.ComputeWorldBound(p).ComputeAlignedRange()
        t = specs[n]["dims_mm"]["thickness"] / 1000.0
        ok = abs(rng.GetMin()[2]) < 1e-6 and abs(rng.GetMax()[2] - t) < 1e-6
        xs.append((rng.GetMin()[0] + rng.GetMax()[0]) / 2)
        R(f"box {n} resting on ground", ok, f"z range {rng.GetMin()[2]:.4f}..{rng.GetMax()[2]:.4f}, "
          f"rigid body={p.HasAPI(UsdPhysics.RigidBodyAPI)}")
    if len(xs) == len(names) and len(xs) > 1:
        gaps = np.diff(xs)
        R("boxes 0.35 m apart", np.allclose(gaps, 0.35, atol=1e-6), str(np.round(gaps, 4).tolist()))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path))
    R("dependencies resolved", not unresolved and all(os.path.isfile(a) for a in assets), f"{len(assets)} assets")


def compliance(usd_dir, names):
    try:
        from pxr.UsdUtils.complianceChecker import ComplianceChecker
    except ImportError:
        print("ComplianceChecker is not shipped in the usd-core wheel (no pxr.UsdUtils.complianceChecker);")
        print("falling back to 'usdchecker' on PATH if present.")
        import shutil
        import subprocess
        exe = shutil.which("usdchecker")
        for n in names + ["medical_boxes_scene"]:
            p = os.path.join(usd_dir, n + ".usda")
            if exe and os.path.isfile(p):
                r = subprocess.run([exe, p], capture_output=True, text=True)
                print(f"{n}: usdchecker rc={r.returncode}\n{(r.stdout + r.stderr).strip()}")
        if not exe:
            print("usdchecker not found: compliance check SKIPPED (informational only).")
        return
    print("\n=== UsdUtils.ComplianceChecker (non-ARKit; informational) ===")
    for n in names + ["medical_boxes_scene"]:
        p = os.path.join(usd_dir, n + ".usda")
        if not os.path.isfile(p):
            continue
        chk = ComplianceChecker(arkit=False, skipARKitRootLayerCheck=True, rootPackageOnly=False,
                                         skipVariants=False, verbose=False, assetLevelChecks=True)
        chk.CheckCompliance(p)
        errs, warns, fails = chk.GetErrors(), chk.GetWarnings(), chk.GetFailedChecks()
        print(f"{n}: {len(errs)} errors, {len(warns)} warnings, {len(fails)} failed checks")
        for tag, items in (("ERROR", errs), ("WARN", warns), ("FAILED", fails)):
            for m in items:
                print(f"   {tag}: {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes-yaml", default=os.path.join(ROOT, "boxes.yaml"))
    ap.add_argument("--usd-dir", default=os.path.join(ROOT, "usd"))
    ap.add_argument("--textures-root", default=None,
                    help="informational: expected textures dir. Textures themselves are resolved "
                         "from the USD; a mismatch with the files the USD points at is reported.")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.boxes_yaml))
    usd_dir = os.path.abspath(a.usd_dir)
    res = Results()
    names = list(cfg["boxes"])
    for n in names:
        spec = dict(cfg["boxes"][n])
        spec["_physics_material"] = cfg["physics_material"]
        verify_box(n, spec, usd_dir, res, os.path.join(usd_dir, "previews"))
        if a.textures_root:
            tr = os.path.abspath(a.textures_root)
            st = Usd.Stage.Open(os.path.join(usd_dir, n + ".usda"))
            lay, assets, _ = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(os.path.join(usd_dir, n + ".usda")))
            bad = [x for x in assets if not os.path.abspath(x).startswith(tr + os.sep)]
            res.add(n, "textures come from --textures-root", not bad, str(bad[:2]))
    verify_scene(names, cfg["boxes"], usd_dir, res)
    print(res.table())
    compliance(usd_dir, names)
    nfail = sum(1 for r in res.rows if not r[2])
    print(f"\n{len(res.rows) - nfail} PASS, {nfail} FAIL")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
