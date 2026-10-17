"""Fixed floor (support) reference plane of a ``hidden_stand`` scene.

Scene option ``live.fixed_floor`` (absent / null = the feature is off, behaviour identical to before)::

    fixed_floor:
      frame: base_link                 # only base_link
      normal: [nx, ny, nz]             # unit-ish, pointing UP (away from the floor), within 10 deg of +z
      offset_m: -0.832                 # floor points p satisfy  normal . p = offset_m  (base_link, m)
      tolerance_m: 0.02                # a cloud point is a floor point when |distance to the plane| < this
      search_m: 0.07                   # optional (default 0.07): band the cloud is checked in
      min_inliers: 200000              # optional (default 20000): floor points required within tolerance_m
      min_inlier_frac: 0.5             # optional (default 0.5): share of the +-search_m band within tolerance_m
      max_median_offset_m: 0.01        # optional (default tolerance_m / 2): |median signed distance| in the band

Why: the far floor (1.5-4.4 m from the camera, ~7 mm MAD depth noise) is a THICK noisy slab, not a thin plane.
A seeded sequential RANSAC with a 7 mm band cuts it into 2-6 facets tilted 0-5 deg whose choice (and the offset
of the box top above the "floor") changes with any tiny data change: ``support_clearance_mm`` varied 815-956 mm
for a box that did not move, and the floor-footprint, parallel-angle and height tests of the hidden-stand top
check, the trim-plane choice and the independent top check all flipped with it (DESIGN.md "Fixed floor"). The
camera is fixed to base_link, so the floor is one fixed plane in base_link: measured once with
``python3 -m box_seam_detection.core.fixed_floor`` (robust fit over many frames) and used instead of
re-estimating it from every cloud.

Safety (fail closed): the plane is never trusted blindly. Each cloud is checked against it
(:func:`evaluate`): the floor points within ``tolerance_m`` must be numerous (``min_inliers``), make up
``min_inlier_frac`` of the +-``search_m`` band, and the band's median signed distance must be small
(``max_median_offset_m``). A camera that moved (or another calibration) shifts the far floor by centimetres and
fails this; the depth estimate then reports ``depth_fixed_floor_mismatch(...)`` (needs_review with that reason)
and the support plane is unavailable (no ``support_clearance_mm``, so the cut-without-height waiver does not
apply). Only valid for ``depth_top.support: hidden_stand``.
"""
from __future__ import annotations

import math

import numpy as np

FRAME = "base_link"
MAX_TILT_DEG = 10.0
DEFAULT_TOLERANCE_M = 0.02
DEFAULT_SEARCH_M = 0.07
DEFAULT_MIN_INLIERS = 20000
DEFAULT_MIN_FRAC = 0.5
_KEYS = {"frame", "normal", "offset_m", "tolerance_m", "search_m", "min_inliers", "min_inlier_frac",
         "max_median_offset_m"}


class FixedFloorError(ValueError):
    """Invalid ``live.fixed_floor`` or an unusable plane (fail closed)."""


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def validate_fixed_floor(value) -> dict | None:
    """Normalised mapping (floats, unit ``normal``, defaults filled in) or None (feature off)."""
    if value is None:
        return None
    what = "live.fixed_floor"
    if not isinstance(value, dict):
        raise FixedFloorError(f"{what} must be a mapping or null, got {type(value).__name__}")
    extra = set(value) - _KEYS
    if extra:
        raise FixedFloorError(f"{what}: unknown keys {sorted(extra)}")
    if value.get("frame") != FRAME:
        raise FixedFloorError(f"{what}.frame must be {FRAME!r}, got {value.get('frame')!r}")
    normal = value.get("normal")
    if (not isinstance(normal, (list, tuple)) or len(normal) != 3
            or not all(_num(v) for v in normal)):
        raise FixedFloorError(f"{what}.normal must be [nx, ny, nz] with finite numbers, got {normal!r}")
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        raise FixedFloorError(f"{what}.normal must not be the zero vector")
    n = n / norm
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    if tilt > MAX_TILT_DEG:
        raise FixedFloorError(f"{what}.normal must point up within {MAX_TILT_DEG:g} deg of +z "
                              f"(floor of a gravity-aligned cell), got a {tilt:.1f} deg tilt")
    if not _num(value.get("offset_m")):
        raise FixedFloorError(f"{what}.offset_m must be a finite number, got {value.get('offset_m')!r}")
    tol = value.get("tolerance_m", DEFAULT_TOLERANCE_M)
    if not _num(tol) or not (0.002 <= float(tol) <= 0.10):
        raise FixedFloorError(f"{what}.tolerance_m must be in [0.002, 0.10] m, got {tol!r}")
    search = value.get("search_m", DEFAULT_SEARCH_M)
    if not _num(search) or not (float(tol) < float(search) <= 0.30):
        raise FixedFloorError(f"{what}.search_m must satisfy tolerance_m < search_m <= 0.30, got {search!r}")
    min_in = value.get("min_inliers", DEFAULT_MIN_INLIERS)
    if isinstance(min_in, bool) or not isinstance(min_in, int) or min_in < 1000:
        raise FixedFloorError(f"{what}.min_inliers must be an integer >= 1000, got {min_in!r}")
    frac = value.get("min_inlier_frac", DEFAULT_MIN_FRAC)
    if not _num(frac) or not (0.0 < float(frac) <= 1.0):
        raise FixedFloorError(f"{what}.min_inlier_frac must be in (0, 1], got {frac!r}")
    med = value.get("max_median_offset_m", float(tol) / 2.0)
    if not _num(med) or not (0.0 < float(med) <= float(tol)):
        raise FixedFloorError(f"{what}.max_median_offset_m must be in (0, tolerance_m], got {med!r}")
    return {"frame": FRAME, "normal": [float(v) for v in n], "offset_m": float(value["offset_m"]),
            "tolerance_m": float(tol), "search_m": float(search), "min_inliers": int(min_in),
            "min_inlier_frac": float(frac), "max_median_offset_m": float(med)}


def to_camera(ff: dict, c2b) -> dict:
    """The plane in the CAMERA frame, ``n . p + d = 0`` with ``d > 0`` (``n`` points from the floor toward
    the camera, the convention of :mod:`autolabel.depth_top`), plus the check parameters.

    ``c2b``: 4x4 base_link <- camera transform (the detector's hand-eye transform)."""
    m = np.asarray(c2b, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise FixedFloorError("camera-to-base transform is not a finite 4x4 matrix")
    nb = np.asarray(ff["normal"], dtype=np.float64)
    rot, t = m[:3, :3], m[:3, 3]
    n_cam = rot.T @ nb
    d = float(nb @ t - ff["offset_m"])      # signed distance of the camera origin above the floor
    if d <= 0.0:
        raise FixedFloorError(f"the camera origin is not above the fixed floor ({d * 1000.0:.0f} mm)")
    out = {k: ff[k] for k in ("tolerance_m", "search_m", "min_inliers", "min_inlier_frac",
                              "max_median_offset_m")}
    out.update({"n": [float(v) for v in n_cam], "d": d, "camera_height_m": round(d, 4)})
    return out


def evaluate(points, ff_cam: dict) -> tuple[dict, dict]:
    """Check a camera-frame cloud (finite rows) against the camera-frame fixed floor.

    Returns ``(table | None, diag)``. ``table`` (None = mismatch, fail closed) is the support plane in the
    shape :func:`autolabel.depth_top.identify_table_top` uses (``n``, ``d``, ``n_inliers``, ``rms``,
    ``centroid``, ``mask`` over ``points``, ``fixed``). ``diag`` always carries the statistics."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(ff_cam["n"], dtype=np.float64)
    d = float(ff_cam["d"])
    s = pts @ n + d
    tol, search = float(ff_cam["tolerance_m"]), float(ff_cam["search_m"])
    band = np.abs(s) < search
    inl = np.abs(s) < tol
    n_band, n_in = int(band.sum()), int(inl.sum())
    frac = (n_in / n_band) if n_band else 0.0
    med = float(np.median(s[band])) if n_band else float("nan")
    diag = {"n_band": n_band, "n_inliers": n_in, "inlier_frac": round(frac, 3),
            "median_offset_mm": None if not math.isfinite(med) else round(med * 1000.0, 1),
            "tolerance_mm": round(tol * 1000.0, 1), "search_mm": round(search * 1000.0, 1)}
    problems = []
    if n_in < int(ff_cam["min_inliers"]):
        problems.append(f"inliers {n_in} < {int(ff_cam['min_inliers'])}")
    if frac < float(ff_cam["min_inlier_frac"]):
        problems.append(f"inlier_frac {frac:.2f} < {float(ff_cam['min_inlier_frac']):g}")
    if not math.isfinite(med) or abs(med) > float(ff_cam["max_median_offset_m"]):
        problems.append(f"median_offset {diag['median_offset_mm']} mm beyond "
                        f"{float(ff_cam['max_median_offset_m']) * 1000.0:.1f}")
    if problems:
        diag["mismatch"] = "; ".join(problems)
        return None, diag
    sel = s[inl]
    table = {"n": n.copy(), "d": d, "n_inliers": n_in,
             "rms": float(np.sqrt(np.mean(sel ** 2))), "centroid": pts[inl].mean(axis=0),
             "mask": inl, "fixed": True}
    return table, diag


# ---------------------------------------------------------------- measurement


def fit_floor(points_base, *, z_hint: float, band0: float = 0.15, band: float = 0.02,
              x_min: float | None = 0.9, iters: int = 8) -> dict:
    """Robust plane fit ``z = a x + b y + c`` (iterative trimming) of the floor points of a base_link cloud
    that lie within ``band0`` of ``z_hint`` (and beyond ``x_min``, away from the stand / box).

    Returns ``{"normal", "offset_m", "tilt_deg", "n_inliers", "n_candidates", "mad_mm", "p95_mm"}`` (the
    values to put in the scene's ``live.fixed_floor``; residuals are along z)."""
    p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    sel = np.abs(p[:, 2] - z_hint) < band0
    if x_min is not None:
        sel &= p[:, 0] > x_min
    p = p[sel]
    if p.shape[0] < 1000:
        raise FixedFloorError(f"only {p.shape[0]} floor candidates near z={z_hint}")
    m = np.ones(p.shape[0], dtype=bool)
    sol = np.zeros(3)
    for _ in range(iters):
        q = p[m]
        sol, *_ = np.linalg.lstsq(np.c_[q[:, 0], q[:, 1], np.ones(q.shape[0])], q[:, 2], rcond=None)
        r = p[:, 2] - (sol[0] * p[:, 0] + sol[1] * p[:, 1] + sol[2])
        m = np.abs(r) < band
    n = np.array([-sol[0], -sol[1], 1.0])
    n /= np.linalg.norm(n)
    res = np.abs(r[m])
    return {"normal": [float(v) for v in n], "offset_m": float(sol[2] * n[2]),
            "tilt_deg": float(np.degrees(np.arccos(n[2]))), "n_inliers": int(m.sum()),
            "n_candidates": int(p.shape[0]), "mad_mm": float(1000.0 * np.median(res)),
            "p95_mm": float(1000.0 * np.percentile(res, 95))}


def _main(argv=None) -> int:        # pragma: no cover - exercised by hand (measurement tool)
    import argparse
    import json
    from pathlib import Path

    from ..live import detector as det
    from . import scene as scene_mod
    from .autolabel.pcd import read_pcd_xyz

    ap = argparse.ArgumentParser(
        description="Measure live.fixed_floor from captured clouds of an EMPTY-floor view (robust fit per "
                    "capture and over all captures).")
    ap.add_argument("--scene-config", required=True)
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--z-hint", type=float, required=True, help="rough base_link z of the floor, m")
    ap.add_argument("--x-min", type=float, default=0.9)
    ap.add_argument("captures", nargs="+", help="capture manifest JSONs (pointcloud path read from them)")
    args = ap.parse_args(argv)
    scene, _cfg = scene_mod.scene_config(args.scene_config)
    pooled, per = [], []
    for path in args.captures:
        man = json.loads(Path(path).read_text())
        c2b = det._camera_to_base_matrix(args.calib_file, man, scene)
        if c2b is None:
            raise SystemExit(f"{path}: no base_link<-camera transform")
        xyz, _ = read_pcd_xyz(man["pointcloud"]["path"])
        xyz = np.asarray(xyz, dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.2)]
        pb = xyz @ np.asarray(c2b)[:3, :3].T + np.asarray(c2b)[:3, 3]
        fit = fit_floor(pb, z_hint=args.z_hint, x_min=args.x_min)
        per.append(fit)
        pooled.append(pb[(np.abs(pb[:, 2] - args.z_hint) < 0.15) & (pb[:, 0] > args.x_min)][::7])
        print(f"{Path(path).stem}: tilt {fit['tilt_deg']:.3f} deg  offset {fit['offset_m']:.4f}  "
              f"MAD {fit['mad_mm']:.1f} mm")
    fit = fit_floor(np.concatenate(pooled), z_hint=args.z_hint, x_min=None)
    offs = np.array([f["offset_m"] for f in per])
    print(f"per-capture offset spread: {1000 * (offs.max() - offs.min()):.1f} mm over {len(per)} captures")
    print("fixed_floor:\n  frame: base_link\n  normal: [" + ", ".join(f"{v:.6f}" for v in fit["normal"])
          + f"]\n  offset_m: {fit['offset_m']:.4f}")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(_main())
