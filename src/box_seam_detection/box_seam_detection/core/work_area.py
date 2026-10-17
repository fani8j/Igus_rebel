"""Robot work area ("work area") of a scene: speed and safety.

Scene option ``live.work_area`` (absent / null = the feature is off, behaviour identical to before)::

    work_area:
      frame: base_link
      polygon_xy_m: [[x, y], ...]     # simple polygon in the target base_link, metres
      top_z_range_m: [zmin, zmax]     # allowed base_link height of the box TOP corners
      outside_keep_every: 10          # every Nth cloud point outside the area is kept (0 = drop all outside)
      margin_m: 0.10                  # optional: cloud crop margin around the polygon (default 0.10)
      floor_keep_below_z_m: -0.75     # optional: ALL points below this base_link z are kept (the dense floor)

Three uses (all driven from :mod:`live.detector`):

1. **Cloud crop** (speed): the capture cloud is transformed to base_link (x, y only) and every point
   inside the polygon grown by ``margin_m`` is kept at ALL heights (box side walls, stand, floor under the
   box are used by the support / hidden-stand / edge-support / depth-edge / truncation-recovery checks); outside
   the polygon only every ``outside_keep_every``-th point is kept, EXCEPT points below ``floor_keep_below_z_m``
   (the floor): the hidden-stand support detection counts absolute plane inliers (``table_min_inliers``) and the
   sequential RANSAC ranks planes by size, so a thinned floor drops below the clutter planes and the support /
   height checks change or fail (measured: 10 offline frames, see DESIGN.md "Work area"). The pipeline then
   runs on the cropped, unorganized cloud (written next to the record), about half the data.
2. **Candidate filter** (speed + robustness): the work-area prism (polygon at ``zmin`` and ``zmax``) is projected
   into the image with the SAME pinhole the pipeline projects the cloud with; a SAM3 candidate whose quad
   lies < ``CANDIDATE_MIN_INSIDE_FRAC`` inside it is dropped before ranking
   (``sam3_candidate_outside_work_area``). The image itself is not cropped.
3. **Hard reject** (safety): the four final top corners, in base_link, must lie inside the polygon (no margin)
   and their z inside ``top_z_range_m`` (``box_outside_work_area``), see :func:`corners_check`.

The base_link <- camera transform is the one the detector already uses for ``seam_direction.base_link_*``
(``detector._camera_to_base_matrix``); this module never builds a transform.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

FRAME = "base_link"
DEFAULT_MARGIN_M = 0.10
MAX_MARGIN_M = 0.50
#: a SAM3 candidate quad must have at least this fraction of its area inside the work-area prism
CANDIDATE_MIN_INSIDE_FRAC = 0.80
CANDIDATE_DROP_REASON = "sam3_candidate_outside_work_area"
_KEYS = {"frame", "polygon_xy_m", "top_z_range_m", "outside_keep_every", "margin_m", "floor_keep_below_z_m"}
_CHUNK = 1 << 18
_EDGE_SAMPLES = 16          # densification of the prism edges before projection


class WorkAreaError(ValueError):
    """Invalid ``live.work_area`` (scene.py re-raises it as SceneError)."""


# ---------------------------------------------------------------------------
# validation

def _finite_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps=1e-12) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_touch(a, b, c, d) -> bool:
    """True when closed segments ab and cd share a point (proper crossing, touching or overlap)."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and 0 not in (o1, o2, o3, o4):
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))


def _check_simple_polygon(poly: list, what: str) -> None:
    n = len(poly)
    for i in range(n):
        if poly[i] == poly[(i + 1) % n]:
            raise WorkAreaError(f"{what}: zero-length edge at vertex {i} (repeated vertex)")

    def fold_back(shared, p, q) -> bool:      # p, q on the same ray from the shared vertex
        return (_orient(shared, p, q) == 0
                and (p[0] - shared[0]) * (q[0] - shared[0]) + (p[1] - shared[1]) * (q[1] - shared[1]) > 0)

    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = poly[j], poly[(j + 1) % n]
            if j == i + 1:                      # adjacent: share b == c
                if fold_back(b, a, d):
                    raise WorkAreaError(f"{what}: edges {i} and {j} fold back on each other")
            elif i == 0 and j == n - 1:         # adjacent (closing edge): share a == d
                if fold_back(a, b, c):
                    raise WorkAreaError(f"{what}: edges {i} and {j} fold back on each other")
            elif _segments_touch(a, b, c, d):
                raise WorkAreaError(f"{what}: not a simple polygon (edges {i} and {j} intersect)")
    area2 = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    if abs(area2) < 1e-12:
        raise WorkAreaError(f"{what}: the polygon has zero area")


def validate_work_area(value) -> dict | None:
    """Normalised work-area mapping (floats / int) or None (feature off). Raises :class:`WorkAreaError`."""
    if value is None:
        return None
    what = "live.work_area"
    if not isinstance(value, dict):
        raise WorkAreaError(f"{what} must be a mapping or null, got {type(value).__name__}")
    unknown = sorted(set(value) - _KEYS)
    if unknown:
        raise WorkAreaError(f"{what}: unknown key(s) {unknown} (allowed: {sorted(_KEYS)})")
    frame = value.get("frame")
    if frame != FRAME:
        raise WorkAreaError(f"{what}.frame must be {FRAME!r} (the detector's target base_link), got {frame!r}")
    raw = value.get("polygon_xy_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise WorkAreaError(f"{what}.polygon_xy_m must be a list of at least 3 [x, y] vertices")
    poly = []
    for i, v in enumerate(raw):
        if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(_finite_number(c) for c in v):
            raise WorkAreaError(f"{what}.polygon_xy_m[{i}] must be [x, y] with two finite numbers, got {v!r}")
        poly.append([float(v[0]), float(v[1])])
    _check_simple_polygon(poly, f"{what}.polygon_xy_m")
    z = value.get("top_z_range_m")
    if (not isinstance(z, (list, tuple)) or len(z) != 2 or not all(_finite_number(c) for c in z)
            or not float(z[0]) < float(z[1])):
        raise WorkAreaError(f"{what}.top_z_range_m must be [zmin, zmax] with finite zmin < zmax, got {z!r}")
    keep = value.get("outside_keep_every", 10)
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
        raise WorkAreaError(f"{what}.outside_keep_every must be an integer >= 1 (every Nth outside point is "
                            f"kept) or 0 (drop all outside points), got {keep!r}")
    margin = value.get("margin_m", DEFAULT_MARGIN_M)
    if not _finite_number(margin) or not (0.0 <= float(margin) <= MAX_MARGIN_M):
        raise WorkAreaError(f"{what}.margin_m must be a finite number in [0, {MAX_MARGIN_M:g}], got {margin!r}")
    below = value.get("floor_keep_below_z_m")
    if below is not None and not _finite_number(below):
        raise WorkAreaError(f"{what}.floor_keep_below_z_m must be a finite number or null, got {below!r}")
    return {"frame": FRAME, "polygon_xy_m": poly, "top_z_range_m": [float(z[0]), float(z[1])],
            "outside_keep_every": int(keep), "margin_m": float(margin),
            "floor_keep_below_z_m": None if below is None else float(below)}


def polygon_hash(wa: dict) -> str:
    """Short stable id of the geometry (frame, polygon, z range): changes whenever the area changes."""
    canon = json.dumps({"frame": wa["frame"],
                        "polygon_xy_m": [[round(c, 6) for c in v] for v in wa["polygon_xy_m"]],
                        "top_z_range_m": [round(c, 6) for c in wa["top_z_range_m"]]},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# geometry

def _poly(wa) -> np.ndarray:
    return np.asarray(wa["polygon_xy_m"], dtype=np.float64).reshape(-1, 2)


def _crossing(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd inside test of (N, 2) points (boundary points are arbitrary)."""
    inside = np.zeros(xy.shape[0], dtype=bool)
    px, py = xy[:, 0], xy[:, 1]
    n = poly.shape[0]
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if ay == by:
            continue
        cond = (ay > py) != (by > py)
        xint = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= cond & (px < xint)
    return inside


def _edge_distance(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of each (N, 2) point to the polygon boundary."""
    best = np.full(xy.shape[0], np.inf)
    n = poly.shape[0]
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((xy - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.hypot(xy[:, 0] - proj[:, 0], xy[:, 1] - proj[:, 1]))
    return best


def points_in_area(xy: np.ndarray, wa: dict, margin_m: float = 0.0) -> np.ndarray:
    """Boolean mask of the (N, 2) base_link points inside the polygon grown by ``margin_m`` (boundary
    included for ``margin_m`` > 0)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    poly = _poly(wa)
    lo, hi = poly.min(axis=0) - margin_m, poly.max(axis=0) + margin_m
    near = np.flatnonzero((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
    out = np.zeros(xy.shape[0], dtype=bool)
    if near.size == 0:
        return out
    sub = xy[near]
    inside = _crossing(sub, poly)
    if margin_m > 0.0:
        rest = np.flatnonzero(~inside)
        if rest.size:
            inside[rest] = _edge_distance(sub[rest], poly) <= margin_m
    out[near] = inside
    return out


def point_in_area(x: float, y: float, wa: dict, tol: float = 1e-9) -> bool:
    """One base_link point inside the polygon (boundary counts as inside, ``tol`` m)."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    xy = np.array([[x, y]], dtype=np.float64)
    poly = _poly(wa)
    return bool(_crossing(xy, poly)[0] or _edge_distance(xy, poly)[0] <= tol)


def transform_points(c2b, xyz) -> np.ndarray:
    """base_link coordinates of camera-frame points: ``R @ p + t`` (float64, (N, 3))."""
    m = np.asarray(c2b, dtype=np.float64)
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


# ---------------------------------------------------------------------------
# cloud crop

def crop_mask(xyz_cam: np.ndarray, c2b, wa: dict) -> tuple[np.ndarray, dict]:
    """``(keep mask, stats)`` for an (N, 3) camera-frame cloud (see module docstring, use 1)."""
    n = int(xyz_cam.shape[0])
    margin = float(wa.get("margin_m", DEFAULT_MARGIN_M))
    every = int(wa["outside_keep_every"])
    m = np.asarray(c2b, dtype=np.float64)
    rx, ry, rz, t = m[0, :3], m[1, :3], m[2, :3], m[:3, 3]
    below = wa.get("floor_keep_below_z_m")        # None = no dense floor band
    inside = np.empty(n, dtype=bool)
    low = np.zeros(n, dtype=bool)
    for s in range(0, n, _CHUNK):
        blk = np.asarray(xyz_cam[s:s + _CHUNK], dtype=np.float64)
        xy = np.stack([blk @ rx + t[0], blk @ ry + t[1]], axis=1)
        inside[s:s + _CHUNK] = points_in_area(xy, wa, margin)
        if below is not None:
            low[s:s + _CHUNK] = (blk @ rz + t[2]) < below
    keep = inside | low                           # inside the area at any height, or on the floor
    if every >= 1:
        rest = np.flatnonzero(~keep)              # the other outside points: every Nth
        keep[rest[::every]] = True
    n_in = int(inside.sum())
    stats = {"points_total": n, "points_inside": n_in, "points_outside": n - n_in,
             "points_outside_kept": int(keep.sum()) - n_in, "points_kept": int(keep.sum()),
             "kept_fraction": round(float(keep.sum()) / n, 4) if n else None,
             "points_floor_kept": int((low & ~inside).sum()),
             "outside_keep_every": every, "margin_m": margin, "floor_keep_below_z_m": below}
    return keep, stats


def write_pcd_xyz(path, xyz: np.ndarray) -> None:
    """Atomic binary PCD v0.7 with x y z float32 fields, unorganized (WIDTH = N, HEIGHT = 1)."""
    path = Path(path)
    arr = np.ascontiguousarray(xyz, dtype="<f4").reshape(-1, 3)
    header = ("# .PCD v0.7 - Point Cloud Data file format (box_seam_detection work-area crop)\n"
              "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              f"WIDTH {arr.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {arr.shape[0]}\nDATA binary\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(arr.tobytes())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def cropped_manifest(manifest: dict, pcd_path, n_points: int, crop: dict) -> dict:
    """Copy of the capture manifest whose ``pointcloud`` block describes the cropped cloud."""
    out = copy.deepcopy(manifest)
    pc = out.setdefault("pointcloud", {})
    pc["work_area_source"] = {"path": (manifest.get("pointcloud") or {}).get("path"),
                              "point_count": (manifest.get("pointcloud") or {}).get("point_count")}
    pc.update({"path": str(pcd_path), "width": int(n_points), "height": 1, "stored_height": 1,
               "point_count": int(n_points), "point_step": 12, "row_step": 12 * int(n_points),
               "fields": [{"name": nm, "offset": 4 * i, "datatype": 7, "count": 1}
                          for i, nm in enumerate("xyz")],
               "work_area_crop": dict(crop)})
    return out


def crop_capture(cloud_xyz: np.ndarray, c2b, wa: dict, manifest: dict, work_dir, capture_id: str
                 ) -> tuple[Path, Path, dict]:
    """Write the cropped cloud + manifest next to the record; ``(pcd, manifest, crop stats)``."""
    keep, stats = crop_mask(cloud_xyz, c2b, wa)
    pcd_path = Path(work_dir) / f"{capture_id}_work_area.pcd"
    man_path = Path(work_dir) / f"{capture_id}_work_area_capture.json"
    write_pcd_xyz(pcd_path, cloud_xyz[keep])
    doc = cropped_manifest(manifest, pcd_path, stats["points_kept"], stats)
    fd, tmp = tempfile.mkstemp(dir=man_path.parent, prefix=f".{man_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        os.replace(tmp, man_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return pcd_path, man_path, stats


# ---------------------------------------------------------------------------
# SAM3 candidate filter (image prism)

def runtime_config(wa: dict, c2b) -> dict:
    """JSON-serialisable block the pipeline reads (``cfg["work_area_runtime"]``)."""
    return {"polygon_xy_m": wa["polygon_xy_m"], "top_z_range_m": wa["top_z_range_m"],
            "camera_to_base": np.asarray(c2b, dtype=float).tolist(),
            "min_inside_frac": CANDIDATE_MIN_INSIDE_FRAC, "polygon_hash": polygon_hash(wa)}


def prism_image_mask(runtime: dict, cam: dict, width: int, height: int):
    """``(uint8 mask, None)`` of the image area covered by the work-area prism, or ``(None, reason)``.

    The silhouette of the prism is the union of its projected faces (top polygon, bottom polygon and
    one quad per side); edges are densified. ``cam``: the pipeline's pinhole dict (fx, fy, cx, cy; no
    distortion, like the cloud projection)."""
    import cv2  # noqa: PLC0415
    from .autolabel.pcd import project_points  # noqa: PLC0415

    poly = np.asarray(runtime["polygon_xy_m"], dtype=np.float64).reshape(-1, 2)
    z0, z1 = (float(v) for v in runtime["top_z_range_m"])
    n = poly.shape[0]
    ring = []                                   # densified boundary (M, 2), closed implicitly
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        s = np.linspace(0.0, 1.0, _EDGE_SAMPLES, endpoint=False)[:, None]
        ring.append(a + s * (b - a))
    ring = np.concatenate(ring, axis=0)
    m = len(ring)
    base = np.concatenate([np.column_stack([ring, np.full(m, z1)]),
                           np.column_stack([ring, np.full(m, z0)])], axis=0)
    try:
        b2c = np.linalg.inv(np.asarray(runtime["camera_to_base"], dtype=np.float64))
    except np.linalg.LinAlgError:
        return None, "prism_transform_singular"
    pts = base @ b2c[:3, :3].T + b2c[:3, 3]
    uv, _z, _stats = project_points(pts, cam, min_depth=0.05)
    if uv.shape[0] != pts.shape[0]:
        return None, "prism_behind_camera"
    if not np.all(np.isfinite(uv)):
        return None, "prism_projection_non_finite"
    top, bot = np.round(uv[:m]).astype(np.int32), np.round(uv[m:]).astype(np.int32)
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    cv2.fillPoly(mask, [top], 1)
    cv2.fillPoly(mask, [bot], 1)
    for i in range(m):
        j = (i + 1) % m
        cv2.fillPoly(mask, [np.array([top[i], top[j], bot[j], bot[i]], dtype=np.int32)], 1)
    return mask, None


class CandidateFilter:
    """Callable ``corners_px -> (keep, info)``; the prism mask is built once, lazily."""

    def __init__(self, runtime: dict, cam: dict, width: int, height: int):
        self.runtime, self.cam, self.width, self.height = runtime, cam, int(width), int(height)
        self.min_frac = float(runtime.get("min_inside_frac", CANDIDATE_MIN_INSIDE_FRAC))
        self._mask = self._err = None
        self._built = False

    def _prism(self):
        if not self._built:
            self._built = True
            try:
                self._mask, self._err = prism_image_mask(self.runtime, self.cam, self.width, self.height)
            except Exception as error:  # noqa: BLE001 - a broken prism only disables the (speed) filter
                self._mask, self._err = None, f"prism_error({type(error).__name__}: {error})"
        return self._mask, self._err

    def __call__(self, corners_px) -> tuple[bool, dict]:
        import cv2  # noqa: PLC0415

        info = {"polygon_hash": self.runtime.get("polygon_hash"), "min_inside_frac": self.min_frac}
        prism, err = self._prism()
        if prism is None:
            # fail open here: the hard corner check after the final quad still guards the output
            return True, {**info, "applied": False, "reason": err}
        quad = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(quad)):
            return True, {**info, "applied": False, "reason": "candidate_non_finite"}
        qmask = np.zeros_like(prism)
        cv2.fillPoly(qmask, [np.round(quad).astype(np.int32)], 1)
        area = int(qmask.sum())
        if area == 0:
            return True, {**info, "applied": False, "reason": "candidate_empty"}
        frac = float((qmask & prism).sum()) / area
        return frac >= self.min_frac, {**info, "applied": True, "inside_frac": round(frac, 4)}


# ---------------------------------------------------------------------------
# final corner check

def corners_check(corners_base: dict, wa: dict, polygon_id: str | None = None) -> dict:
    """Pure check of base_link corners ``{name: [x, y, z]}``. Returns the record block; ``pass`` False
    carries ``xy_outside`` / ``z_outside`` lists (corner names). No margin on the polygon."""
    zlo, zhi = wa["top_z_range_m"]
    xy_out, z_out = [], []
    for name, p in corners_base.items():
        if not point_in_area(p[0], p[1], wa):
            xy_out.append(name)
        if not (math.isfinite(p[2]) and zlo <= p[2] <= zhi):
            z_out.append(name)
    return {"enabled": True, "measured": True, "frame": wa["frame"],
            "polygon_hash": polygon_id or polygon_hash(wa),
            "polygon_vertices": len(wa["polygon_xy_m"]), "top_z_range_m": list(wa["top_z_range_m"]),
            "corners_base_link": {k: [round(float(c), 4) for c in v] for k, v in corners_base.items()},
            "xy_outside": xy_out, "z_outside": z_out, "pass": not xy_out and not z_out}
