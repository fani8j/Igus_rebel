#!/usr/bin/env python3
"""Conservative convex collision hulls of the real ReBeL link meshes, for the DUAL-ARM model only.

    python3 scripts/make_collision_hulls.py            # (re)generate meshes/collision/*.stl + hulls_report.json
    python3 scripts/make_collision_hulls.py --verify   # containment check of the checked-in STLs only (exit 1 on failure)

Why: the single-arm collision primitives (boxes/cylinders) of link1-4 cover only part of the real links
(up to 88 mm of the real mesh lies outside them, motor3 is 20 mm short). A proof that two arms cannot collide
needs collision geometry that CONTAINS the real link. See meshes/collision/README.md.

Method (generalises scan_results/link5_collision_20261012/tools/make_link5_hull.py):
  1. real mesh = the visual .dae of igus_rebel_description with the visual <origin> from igus_rebel2.urdf applied,
     in the LINK frame. The STLs are written in the link frame with that offset baked in, so the URDF collision
     <origin> is identity.
  2. L/C-shaped links (link1-4) are cut by axis-aligned planes (greedy: the plane that removes the most hull volume,
     recursively, up to PIECES[link] pieces, a split must remove at least MIN_GAIN of the link's exact hull volume).
     The mesh is CLIPPED by the plane (triangles crossing it are cut, no triangle is dropped), each side is hulled
     separately: every triangle of the real mesh then lies in exactly one piece.
  3. Each piece's exact convex hull is replaced by a low-poly circumscribed polytope: the intersection of the
     half-spaces n.x <= max_v(n.v) + MARGIN (0.2 mm) over icosphere directions, the largest hull-face normals, the axes and
     the cutting-plane normals. Every vertex of the piece satisfies every half-space, so the polytope contains the
     piece by construction (it is also exactly verified, below).
  4. Verification (exits non-zero on failure, tolerance TOL = 0.1 mm): all vertices of the ORIGINAL mesh and
     SAMPLES surface points of the original mesh are inside at least one piece (n.p + d <= TOL for every qhull facet
     plane of that piece); every piece is a watertight convex STL.
Deterministic (fixed sampling seed). Pure numpy/scipy/trimesh.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
from trimesh import intersections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", ".."))
SINGLE_DESC = os.path.join(SRC, "igus_rebel_description")
URDF = os.path.join(SINGLE_DESC, "urdf", "igus_rebel2.urdf")
MESH_DIR = os.path.join(SINGLE_DESC, "meshes", "rebel_d00617809")
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "meshes", "collision"))
REPORT = os.path.join(OUT_DIR, "hulls_report.json")

# link -> max number of convex pieces (1 = a single hull; the over-coverage of a single hull of an L-shaped link
# is 2-3x its volume, see the audit numbers in meshes/collision/README.md)
PIECES = {"base_link": 1, "motor1": 1, "link1": 3, "motor2": 1, "link2": 4, "motor3": 1, "link3": 3,
          "motor4": 1, "link4": 3, "motor5": 1, "link6": 1}
MIN_GAIN = 0.06       # a split must remove >= 6 % of the link's exact single-hull volume
ICO_SUBDIV = 1        # 42 directions
N_FACE_NORMALS = {"link1": 24, "link2": 24, "link3": 24, "link4": 24}   # per piece; default below
DEFAULT_FACE_NORMALS = 16
MARGIN = 2e-4         # 0.2 mm outward on every half-space
CLUSTER = 1e-4        # corner clusters merged within 0.1 mm
TOL = 1e-4            # containment tolerance 0.1 mm
SAMPLES = 400000
SPLIT_STEP = 0.002    # candidate split planes every 2 mm
MIN_PIECE_EXTENT = 0.012


def visual_origin(link_name):
    root = ET.parse(URDF).getroot()
    for ln in root.findall("link"):
        if ln.get("name") == link_name:
            o = ln.find("visual/origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            assert rpy == [0, 0, 0], f"{link_name}: non-zero visual rpy not supported"
            return np.array(xyz)
    raise KeyError(link_name)


def load_real(link_name):
    m = trimesh.load(os.path.join(MESH_DIR, link_name + ".dae"), force="mesh")
    m = trimesh.Trimesh(np.asarray(m.vertices) + visual_origin(link_name), np.asarray(m.faces), process=False)
    return m


def hull_volume(mesh):
    return ConvexHull(mesh.vertices).volume


def best_split(mesh):
    """Return (gain, axis, pos, lo_mesh, hi_mesh) of the best axis-aligned plane, or None."""
    v0 = hull_volume(mesh)
    best = None
    for axis in range(3):
        lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
        n = np.zeros(3)
        n[axis] = 1.0
        for pos in np.arange(lo + MIN_PIECE_EXTENT, hi - MIN_PIECE_EXTENT, SPLIT_STEP):
            o = n * pos
            a = intersections.slice_mesh_plane(mesh, n, o, cap=False)
            b = intersections.slice_mesh_plane(mesh, -n, o, cap=False)
            if len(a.faces) < 8 or len(b.faces) < 8:
                continue
            if a.extents[axis] < MIN_PIECE_EXTENT or b.extents[axis] < MIN_PIECE_EXTENT:
                continue
            gain = v0 - hull_volume(a) - hull_volume(b)
            if best is None or gain > best[0]:
                best = (gain, axis, pos, b, a)   # b = side below the plane, a = side above
    return best


def decompose(real, max_pieces):
    """Greedy recursive plane splits. Returns [(mesh_piece, [cut normals])]."""
    total = hull_volume(real)
    pieces = [(real, [])]
    while len(pieces) < max_pieces:
        cand = None
        for i, (m, _) in enumerate(pieces):
            s = best_split(m)
            if s is not None and (cand is None or s[0] > cand[0][0]):
                cand = (s, i)
        if cand is None or cand[0][0] < MIN_GAIN * total:
            break
        (gain, axis, pos, lo, hi), i = cand
        n = np.zeros(3)
        n[axis] = 1.0
        cuts = pieces[i][1]
        pieces[i:i + 1] = [(lo, cuts + [n]), (hi, cuts + [-n])]
        print(f"    split axis {'xyz'[axis]} at {pos * 1000:.1f} mm: hull volume -{gain * 1e6:.1f} cm3 "
              f"({100 * gain / total:.1f} % of the link hull)")
    return pieces


def circumscribed(points, cut_normals, n_face_normals):
    """Low-poly convex polytope that contains `points` (circumscribed polytope, see module docstring)."""
    ex = ConvexHull(points)
    H = points[ex.vertices]
    ex_mesh = trimesh.Trimesh(H, process=False).convex_hull
    dirs = [np.asarray(trimesh.creation.icosphere(subdivisions=ICO_SUBDIV).vertices),
            np.vstack([np.eye(3), -np.eye(3)])]
    big = np.argsort(ex_mesh.area_faces)[::-1][:n_face_normals]
    dirs.append(ex_mesh.face_normals[big])
    if cut_normals:
        dirs.append(np.asarray(cut_normals))
    D = np.vstack(dirs)
    D = D / np.linalg.norm(D, axis=1)[:, None]
    D = D[np.unique(np.round(D, 6), axis=0, return_index=True)[1]]      # drop duplicate directions
    off = (H @ D.T).max(axis=0) + MARGIN
    hs = np.hstack([D, -off[:, None]])
    hi = HalfspaceIntersection(hs, H.mean(axis=0))
    P = hi.intersections
    P = P[np.isfinite(P).all(axis=1)]
    # Corners where 3+ planes nearly meet come out as clusters of near-identical points and make qhull emit a
    # non-watertight hull. Merge clusters within CLUSTER (0.1 mm) to their mean; MARGIN (0.2 mm) > CLUSTER, so the
    # merged polytope still contains the points.
    tree = cKDTree(P)
    pairs = np.array(sorted(tree.query_pairs(CLUSTER))).reshape(-1, 2)
    n = len(P)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(adj, directed=False)
    Q = np.array([P[lab == k].mean(axis=0) for k in range(nc)])
    hull = trimesh.Trimesh(Q, process=False).convex_hull
    assert hull.is_watertight and hull.is_convex
    return hull, ex_mesh


def violation(points, hull):
    """max over the facet planes of the hull of n.p + d per point (<= 0 inside; a lower bound of the distance if
    outside). The planes come from qhull on the hull's vertices, not from the (possibly sliver) STL triangles."""
    eq = ConvexHull(hull.vertices).equations
    out = np.empty(len(points))
    for s in range(0, len(points), 50000):
        out[s:s + 50000] = (points[s:s + 50000] @ eq[:, :3].T + eq[:, 3]).max(axis=1)
    return out


def stl_names(link, n):
    return [f"{link}_hull.stl"] if n == 1 else [f"{link}_hull_{k + 1}.stl" for k in range(n)]


def verify(link, real, hulls, seed=1):
    """Every original vertex and SAMPLES surface samples lie in at least one hull. Returns (ok, stats)."""
    pts = np.vstack([real.vertices, trimesh.sample.sample_surface(real, SAMPLES, seed=seed)[0]])
    best = np.full(len(pts), np.inf)
    for h in hulls:
        best = np.minimum(best, violation(pts, h))
    # convexity is checked as 'the closed STL surface encloses the volume of the convex hull of its own vertices'
    # (a closed surface inside a convex hull with the same volume is that hull; trimesh's
    # is_convex is not robust against float32 rounding of the STL)
    ok_geom = all(h.is_watertight and abs(h.volume - ConvexHull(h.vertices).volume) <= 1e-4 * h.volume
                  for h in hulls)
    worst = float(best.max())
    stats = {"max_outside_mm": worst * 1000.0, "points_checked": int(len(pts)),
             "n_points_outside_gt_tol": int((best > TOL).sum()), "watertight_convex": bool(ok_geom)}
    return worst <= TOL and ok_geom, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only verify the checked-in STLs")
    ap.add_argument("--links", default="", help="comma list (default: all)")
    a = ap.parse_args()
    links = [x for x in a.links.split(",") if x] or list(PIECES)
    os.makedirs(OUT_DIR, exist_ok=True)
    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    all_ok = True
    for link in links:
        real = load_real(link)
        print(f"{link}: real mesh {len(real.faces)} faces, bounds mm "
              f"{np.round(real.bounds * 1000, 1).tolist()}")
        if a.verify:
            names = [n for n in report[link]["files"]]
            hulls = [trimesh.load(os.path.join(OUT_DIR, n), force="mesh") for n in names]
        else:
            pieces = decompose(real, PIECES[link])
            hulls, exact_vol = [], 0.0
            for m, cuts in pieces:
                h, ex = circumscribed(m.vertices, cuts, N_FACE_NORMALS.get(link, DEFAULT_FACE_NORMALS))
                hulls.append(h)
                exact_vol += ex.volume
            names = stl_names(link, len(hulls))
        ok, stats = verify(link, real, hulls)
        faces = [len(h.faces) for h in hulls]
        vol = sum(h.volume for h in hulls)
        print(f"  pieces {len(hulls)} faces {faces} volume {vol * 1e6:.1f} cm3; max outside "
              f"{stats['max_outside_mm']:.4f} mm over {stats['points_checked']} points -> {'OK' if ok else 'FAIL'}")
        all_ok &= ok
        if not a.verify:
            for old in os.listdir(OUT_DIR):     # drop stale pieces of an earlier decomposition
                if old.startswith(link + "_hull") and old.endswith(".stl"):
                    os.remove(os.path.join(OUT_DIR, old))
            for n, h in zip(names, hulls):
                h.export(os.path.join(OUT_DIR, n))
            # re-verify what was actually written (binary STL is float32)
            hulls = [trimesh.load(os.path.join(OUT_DIR, n), force="mesh") for n in names]
            ok2, stats = verify(link, real, hulls)
            print(f"  re-verified from the written STL files: max outside {stats['max_outside_mm']:.4f} mm -> "
                  f"{'OK' if ok2 else 'FAIL'}")
            all_ok &= ok2
            single_exact = hull_volume(real)
            report[link] = {"files": names, "faces": faces, "volume_cm3": vol * 1e6,
                            "exact_single_hull_volume_cm3": single_exact * 1e6,
                            "sum_piece_exact_hull_volume_cm3": exact_vol * 1e6,
                            "bounds_mm": [(np.min([h.bounds[0] for h in hulls], axis=0) * 1000).round(2).tolist(),
                                          (np.max([h.bounds[1] for h in hulls], axis=0) * 1000).round(2).tolist()],
                            "verify": stats,
                            "frame": "link frame, visual origin baked in; URDF collision origin is identity"}
    if not a.verify:
        json.dump(report, open(REPORT, "w"), indent=1, sort_keys=True)
        print("wrote", REPORT)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
