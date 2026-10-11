# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/sam3_depth_fusion.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""SAM3-guided depth fusion: refined RGB quad -> eroded PCD interior -> plane -> metric corners.

Pipeline (all geometry in ``camera_color_optical_frame``):

1. Load the ``rgbd_capture/v1`` manifest (image / pointcloud / camera_info),
   requiring matching ``camera_color_optical_frame`` frames and a readable,
   finite PCD. The registered color-frame PCD needs no extrinsic transform.
2. Instantiate :class:`CameraModel` from the manifest's exact CameraInfo and
   project every PCD point explicitly with its projection semantics
   (distortion applied exactly once, inside the model). PCD row order is
   never assumed to be image order.
3. Rasterize the refined convex quad (``[BL, BR, FR, FL]``) at full
   resolution, erode it by ``erosion_px`` with cv2, and keep only projected
   points inside the eroded mask. Boundary points are never fitted.
4. Resolve projection collisions deterministically: nearest depth wins per
   pixel (ties broken by the smallest source row), preserving the winning
   original XYZ row. Points behind a nearer surface at the same pixel are
   excluded.
5. Structural RANSAC + inlier-only IRLS fit over the eroded, z-buffer-visible
   XYZ points. ``min_inliers`` only rejects insufficient geometry.
6. Intersect calibrated RGB rays through the refined corners with the fitted
   plane (raw corner depth is never read), reproject the XYZ corners, and
   report per-corner / RMS error.

Scene problems return a ``REJECTED`` result with explicit reasons and NaN
corners; only programming errors (bad parameter values) raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .camera_model import (
    CameraModel,
    Plane,
    project_xyz,
    ray_plane_intersection,
)
from .depth_geometry import PcdError, read_pcd_xyz
from .plane_fit import PlaneEstimate, fit_plane_ransac_irls

__all__ = [
    "OPTICAL_FRAME",
    "STATUS_OK",
    "STATUS_REJECTED",
    "GuidedDepthPlane",
    "fit_sam3_guided_depth_plane",
]

#: Frame every input and output must live in.
OPTICAL_FRAME = "camera_color_optical_frame"

#: Success status for :attr:`GuidedDepthPlane.status`.
STATUS_OK = "OK"

#: Rejection status for :attr:`GuidedDepthPlane.status` (see ``reasons``).
STATUS_REJECTED = "REJECTED"

EXPECTED_SCHEMA = "rgbd_capture/v1"

#: Machine-readable reason codes (``reasons`` is a subset of these).
REASONS = (
    "invalid_manifest",  # manifest missing/unparseable or required section absent
    "invalid_quad",  # refined corners not a finite convex quad with positive area
    "frame_mismatch",  # image/camera_info/pointcloud frame disagreement or != optical
    "size_mismatch",  # decoded image vs manifest vs camera_info dimensions disagree
    "image_unreadable",  # manifest image file missing or undecodable
    "pcd_unreadable",  # manifest pointcloud file missing or undecodable
    "empty_pointcloud",  # PCD holds no finite points
    "no_projected_interior_points",  # nothing projectable inside the eroded mask
    "insufficient_inliers",  # fewer than min_inliers points / no RANSAC support
    "ray_plane_intersection_failed",  # corner rays miss the plane (parallel/behind)
    "reprojection_failed",  # reprojected corners non-finite
)


@dataclass
class GuidedDepthPlane:
    """SAM3-guided structural depth-plane result (contract type)."""

    capture_id: str
    frame_id: str
    plane: Optional[Plane]
    estimate: Optional[PlaneEstimate]
    camera_corners_xyz: np.ndarray  # float64 (4, 3), [BL, BR, FR, FL]; NaN unless OK
    input_point_count: int
    projected_point_count: int
    eroded_point_count: int
    eroded_area_px: int
    erosion_px: int
    reprojection_error_px: dict  # {"per_corner_px": [4 floats], "rms_px": float}
    status: str  # "OK" or "REJECTED"
    reasons: list = field(default_factory=list)


def _nan_corners() -> np.ndarray:
    return np.full((4, 3), np.nan, dtype=np.float64)


def _nan_reprojection() -> dict:
    return {"per_corner_px": [float("nan")] * 4, "rms_px": float("nan")}


def _reject(
    capture_id: str,
    frame_id: str,
    reason: str,
    *,
    input_point_count: int = 0,
    projected_point_count: int = 0,
    eroded_point_count: int = 0,
    eroded_area_px: int = 0,
    erosion_px: int = 0,
) -> GuidedDepthPlane:
    return GuidedDepthPlane(
        capture_id=capture_id,
        frame_id=frame_id,
        plane=None,
        estimate=None,
        camera_corners_xyz=_nan_corners(),
        input_point_count=int(input_point_count),
        projected_point_count=int(projected_point_count),
        eroded_point_count=int(eroded_point_count),
        eroded_area_px=int(eroded_area_px),
        erosion_px=int(erosion_px),
        reprojection_error_px=_nan_reprojection(),
        status=STATUS_REJECTED,
        reasons=[str(reason)],
    )


def _load_doc(capture_manifest: Any) -> tuple[dict, Path]:
    """Return (manifest dict, base dir for relative paths)."""
    if isinstance(capture_manifest, dict):
        doc = dict(capture_manifest)
        raw_base = doc.get("manifest_path", "<manifest>")
        base = Path(str(raw_base)).parent if str(raw_base) != "<manifest>" else Path.cwd()
        return doc, base
    manifest_path = Path(str(capture_manifest))
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        raise ValueError(f"cannot load manifest: {manifest_path}")
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"cannot load manifest {manifest_path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"cannot load manifest {manifest_path}: must be a JSON object")
    return doc, manifest_path.parent


def _check_quad(corners: Any) -> Optional[np.ndarray]:
    """Return validated (4, 2) float64 quad or None when invalid."""
    try:
        arr = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
        return None
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    if not np.isfinite(area) or area <= 1e-6:
        return None
    # Strict convexity: all edge cross products share one nonzero sign.
    edges = np.roll(arr, -1, axis=0) - arr
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not np.all(np.isfinite(crosses)):
        return None
    if not (bool(np.all(crosses > 0.0)) or bool(np.all(crosses < 0.0))):
        return None
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Per-process reuse (speed only; results are bit-identical to recomputing).
#
# One capture's PCD is projected + z-buffered several times per process with
# the same camera (guided plane fit, all-point p95, corner depth winners).
# The projected cloud is cached keyed by the file identity (resolved path,
# device, inode, size, mtime/ctime ns + a sampled content digest), the
# projection intrinsics/distortion, the raster size and the depth gate;
# nearest-depth-wins is then resolved only over the pixel region a consumer
# can use (same winners, same order). Cached arrays are read-only; every
# consumer indexes them (fresh copies), so nothing downstream can mutate the
# cache. Set AUTOLABEL_DISABLE_CACHE=1 to bypass (recompute every call).

_CACHE_MAX = 2
_PROJECT_CHUNK = 1 << 16
_ZBUF_CACHE: "dict[tuple, ProjectedCloud]" = {}
_WINNERS_MEMO: dict = {}
_IMAGE_SHAPE_CACHE: "dict[tuple, tuple[int, int]]" = {}


def _cache_disabled() -> bool:
    import os

    return os.environ.get("AUTOLABEL_DISABLE_CACHE", "") not in ("", "0")


def file_identity(path: Any) -> tuple:
    """Stat + sampled-content identity of a file (raises OSError if missing).

    Used as a cache key: any rewrite changes size/mtime/ctime/inode, and the
    sampled digest (head, tail and 16 evenly spaced 4 KiB blocks) guards a
    same-size in-place rewrite within the filesystem timestamp granularity.
    """
    import hashlib
    import os

    p = os.path.realpath(str(path))
    st = os.stat(p)
    h = hashlib.blake2b(digest_size=16)
    block = 4096
    with open(p, "rb") as fh:
        size = st.st_size
        if size <= 18 * block:
            h.update(fh.read())
        else:
            for i in range(17):
                fh.seek((size - block) * i // 16)
                h.update(fh.read(block))
    return (p, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
            st.st_ctime_ns, h.hexdigest())


def _camera_key(camera: CameraModel) -> tuple:
    return (float(camera.fx), float(camera.fy), float(camera.cx),
            float(camera.cy), tuple(float(v) for v in camera.d),
            bool(camera.has_distortion()))


def _cache_put(cache: dict, key: tuple, value: Any, limit: int = _CACHE_MAX) -> None:
    cache[key] = value
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def clear_caches() -> None:
    """Drop every per-process cache (tests / long-running processes)."""
    _ZBUF_CACHE.clear()
    _WINNERS_MEMO.clear()
    _IMAGE_SHAPE_CACHE.clear()


def remember_image_shape(path: Any, shape: Any) -> None:
    """Record that ``path`` decoded successfully to ``shape`` (H, W, ...).

    Lets a caller that already decoded the capture image spare
    :func:`fit_sam3_guided_depth_plane` a second full decode (it only needs
    the size). Keyed by :func:`file_identity`; never raises.
    """
    if _cache_disabled():
        return
    try:
        _cache_put(_IMAGE_SHAPE_CACHE, file_identity(path),
                   (int(shape[0]), int(shape[1])))
    except Exception:  # noqa: BLE001 — cache is best-effort
        pass


def _image_hw(image_path: Path) -> Optional[tuple[int, int]]:
    """(H, W) of a decodable colour image, or None (== cv2.imread failed)."""
    key = None
    if not _cache_disabled():
        try:
            key = file_identity(image_path)
        except OSError:
            key = None
        if key is not None and key in _IMAGE_SHAPE_CACHE:
            return _IMAGE_SHAPE_CACHE[key]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    hw = (int(bgr.shape[0]), int(bgr.shape[1]))
    if key is not None:
        _cache_put(_IMAGE_SHAPE_CACHE, key, hw)
    return hw


def _ro(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class ProjectedCloud:
    """One PCD projected into one camera raster (cached per process).

    ``status``: ``"ok"`` | ``"empty"`` (no points after the depth gate) |
    ``"project_error"`` (:func:`project_xyz` raised ValueError) |
    ``"no_in_bounds"``. For ``"ok"``: ``bu``/``bv`` integer pixels
    (floor of the projection), ``bxyz`` float64 XYZ and ``lin`` the linear
    pixel index of every in-bounds point, in source-row order.
    """

    status: str
    input_point_count: int
    projected_point_count: int = 0
    width: int = 0
    height: int = 0
    bu: Optional[np.ndarray] = None
    bv: Optional[np.ndarray] = None
    bxyz: Optional[np.ndarray] = None
    lin: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ProjectedWinners:
    """Nearest-depth-wins z-buffer winners (see :func:`zbuffer_winners`).

    ``wu``/``wv`` are the winning integer pixels and ``xyz_win`` the winning
    float64 XYZ rows, ordered by ascending linear pixel index — exactly
    ``order[first]`` of ``np.lexsort((row, z))`` + ``np.unique(lin[order],
    return_index=True)`` over all in-bounds points (the per-call code this
    replaces), restricted to the requested pixel region.
    """

    status: str
    input_point_count: int
    projected_point_count: int = 0
    wu: Optional[np.ndarray] = None
    wv: Optional[np.ndarray] = None
    xyz_win: Optional[np.ndarray] = None


def _compute_projected_cloud(pcd_path: Any, camera: CameraModel, width: int,
                             height: int, min_depth_m: float) -> ProjectedCloud:
    xyz, _pcd_info = read_pcd_xyz(pcd_path)
    xyz = np.asarray(xyz, dtype=np.float64)
    input_point_count = int(xyz.shape[0])
    gated = np.ascontiguousarray(xyz[xyz[:, 2] >= float(min_depth_m)])
    if gated.shape[0] == 0:
        return ProjectedCloud("empty", input_point_count)
    try:
        # Chunked projection: project_xyz is a pure per-point map (its
        # validation raises iff any point is invalid), so chunking yields the
        # identical uv array while keeping temporaries cache-resident.
        uv = np.concatenate([
            np.asarray(project_xyz(gated[s:s + _PROJECT_CHUNK], camera),
                       dtype=np.float64).reshape(-1, 2)
            for s in range(0, gated.shape[0], _PROJECT_CHUNK)], axis=0)
    except ValueError:
        return ProjectedCloud("project_error", input_point_count)
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
    uu = np.floor(uv[:, 0]).astype(np.int64)
    vv = np.floor(uv[:, 1]).astype(np.int64)
    in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
    projected_point_count = int(np.count_nonzero(in_bounds))
    if projected_point_count == 0:
        return ProjectedCloud("no_in_bounds", input_point_count, 0)
    bu, bv = uu[in_bounds], vv[in_bounds]
    lin = bv * np.int64(width) + bu
    return ProjectedCloud(
        "ok", input_point_count, projected_point_count, int(width), int(height),
        bu=_ro(bu), bv=_ro(bv), bxyz=_ro(gated[in_bounds]), lin=_ro(lin))


def _winners(pc: ProjectedCloud, sel: Optional[np.ndarray]
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-depth-wins over the in-bounds points ``sel`` (None = all).

    Ties keep the smallest source row. Every point of a selected pixel must
    be in ``sel`` (pixel regions guarantee it), so each pixel's winner is
    the same as over all points and the result is the global winner list
    restricted to those pixels, in the same (ascending pixel) order.
    """
    if sel is None:
        bxyz, lin, bu, bv = pc.bxyz, pc.lin, pc.bu, pc.bv
        brow = np.arange(lin.shape[0])
    else:
        brow = np.flatnonzero(sel)
        bxyz, lin, bu, bv = pc.bxyz[brow], pc.lin[brow], pc.bu[brow], pc.bv[brow]
    order = np.lexsort((brow, bxyz[:, 2]))  # primary: depth; secondary: source row
    _, first = np.unique(lin[order], return_index=True)
    winners = order[first]
    return bu[winners], bv[winners], bxyz[winners]


def zbuffer_winners(pc: ProjectedCloud, region: Optional[tuple] = None
                    ) -> ProjectedWinners:
    """Winners of a :class:`ProjectedCloud`, optionally only for pixels in
    ``region = (u0, v0, u1, v1)`` (inclusive integer bounds). All-pixel
    winners are memoised on the cloud; region queries are recomputed."""
    if pc.status != "ok":
        return ProjectedWinners(pc.status, pc.input_point_count,
                                pc.projected_point_count)
    if region is None:
        memo = _WINNERS_MEMO.get(id(pc))
        if memo is not None and memo[0] is pc:
            return memo[1]
        wu, wv, xw = _winners(pc, None)
        out = ProjectedWinners("ok", pc.input_point_count, pc.projected_point_count,
                               wu=_ro(wu), wv=_ro(wv), xyz_win=_ro(xw))
        if not _cache_disabled():
            _WINNERS_MEMO.clear()
            _WINNERS_MEMO[id(pc)] = (pc, out)
        return out
    u0, v0, u1, v1 = (int(x) for x in region)
    sel = (pc.bu >= u0) & (pc.bu <= u1) & (pc.bv >= v0) & (pc.bv <= v1)
    wu, wv, xw = _winners(pc, sel)
    return ProjectedWinners("ok", pc.input_point_count, pc.projected_point_count,
                            wu=_ro(wu), wv=_ro(wv), xyz_win=_ro(xw))


def mask_region(mask: np.ndarray) -> Optional[tuple]:
    """Inclusive (u0, v0, u1, v1) bounding box of ``mask > 0`` (None if empty)."""
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(np.any(mask > 0, axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def projected_cloud(pcd_path: Any, camera: CameraModel, width: int, height: int,
                    min_depth_m: float = 0.05) -> ProjectedCloud:
    """Cached :class:`ProjectedCloud` of ``pcd_path`` (keyed by file identity,
    projection parameters, raster size and depth gate). Raises what
    :func:`read_pcd_xyz` raises."""
    if _cache_disabled():
        return _compute_projected_cloud(pcd_path, camera, int(width),
                                        int(height), float(min_depth_m))
    key = (file_identity(pcd_path), _camera_key(camera), int(width),
           int(height), float(min_depth_m))
    hit = _ZBUF_CACHE.get(key)
    if hit is None:
        hit = _compute_projected_cloud(pcd_path, camera, int(width),
                                       int(height), float(min_depth_m))
        # one capture at a time: keep a single projected cloud (~75 MB)
        _WINNERS_MEMO.clear()
        _cache_put(_ZBUF_CACHE, key, hit, limit=1)
    return hit


def projected_depth_winners(pcd_path: Any, camera: CameraModel, width: int,
                            height: int, min_depth_m: float = 0.05,
                            region: Optional[tuple] = None) -> ProjectedWinners:
    """Nearest-depth-wins winners of ``pcd_path`` in ``camera`` (all pixels,
    or only pixels inside ``region``; see :func:`zbuffer_winners`)."""
    return zbuffer_winners(
        projected_cloud(pcd_path, camera, width, height, min_depth_m), region)


def _eroded_mask(quad: np.ndarray, width: int, height: int, erosion_px: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    if erosion_px > 0 and int(np.count_nonzero(mask)) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=int(erosion_px))
    return mask


def fit_sam3_guided_depth_plane(
    capture_manifest: Any,
    refined_corners_px: Any,
    *,
    erosion_px: int = 8,
    ransac_threshold_m: float = 0.003,
    ransac_iters: int = 500,
    min_inliers: int = 100,
    min_depth_m: float = 0.05,
) -> GuidedDepthPlane:
    """Fit a structural depth plane inside a SAM3-refined RGB quad.

    Args:
        capture_manifest: manifest path or parsed ``rgbd_capture/v1`` dict.
        refined_corners_px: ``[BL, BR, FR, FL]`` pixel quad, shape (4, 2).
        erosion_px: cv2 erosion iterations applied to the rasterized quad.
        ransac_threshold_m: RANSAC inlier band (meters).
        ransac_iters: RANSAC hypotheses (deterministic, seed 0).
        min_inliers: minimum geometry support; only rejects insufficient data.
        min_depth_m: points nearer than this are never projected.

    Returns:
        :class:`GuidedDepthPlane` with ``status`` ``"OK"`` on success or
        ``"REJECTED"`` with explicit ``reasons`` (NaN corners, null plane).
    """
    if isinstance(erosion_px, bool) or not isinstance(erosion_px, int) or erosion_px < 0:
        raise ValueError("erosion_px must be a non-negative int")
    ransac_threshold_m = float(ransac_threshold_m)
    if not np.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0.0:
        raise ValueError("ransac_threshold_m must be positive finite")
    if isinstance(ransac_iters, bool) or not isinstance(ransac_iters, int) or ransac_iters <= 0:
        raise ValueError("ransac_iters must be a positive int")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, int) or min_inliers < 3:
        raise ValueError("min_inliers must be an int >= 3")
    min_depth_m = float(min_depth_m)
    if not np.isfinite(min_depth_m) or min_depth_m <= 0.0:
        raise ValueError("min_depth_m must be positive finite")

    quad = _check_quad(refined_corners_px)

    try:
        doc, base = _load_doc(capture_manifest)
    except ValueError:
        return _reject("", "", "invalid_manifest", erosion_px=int(erosion_px))

    capture_id = doc.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id:
        return _reject("", "", "invalid_manifest", erosion_px=int(erosion_px))
    if doc.get("schema") != EXPECTED_SCHEMA:
        return _reject(capture_id, "", "invalid_manifest", erosion_px=int(erosion_px))
    if quad is None:
        return _reject(capture_id, "", "invalid_quad", erosion_px=int(erosion_px))

    image = doc.get("image")
    cloud = doc.get("pointcloud")
    info = doc.get("camera_info")
    if not isinstance(image, dict) or not isinstance(cloud, dict) or not isinstance(info, dict):
        return _reject(capture_id, "", "invalid_manifest", erosion_px=int(erosion_px))

    def _header_frame(node: dict) -> Optional[str]:
        header = node.get("header")
        if not isinstance(header, dict):
            return None
        frame = header.get("frame_id")
        return frame if isinstance(frame, str) and frame else None

    image_frame = _header_frame(image)
    cloud_frame = _header_frame(cloud)
    camera_frame = _header_frame(info)
    if (
        image_frame != OPTICAL_FRAME
        or cloud_frame != OPTICAL_FRAME
        or camera_frame != OPTICAL_FRAME
    ):
        return _reject(capture_id, camera_frame or "", "frame_mismatch",
                        erosion_px=int(erosion_px))
    frame_id = str(camera_frame)

    try:
        width = int(image["width"])
        height = int(image["height"])
        info_width = int(info["width"])
        info_height = int(info["height"])
    except (KeyError, TypeError, ValueError):
        return _reject(capture_id, frame_id, "invalid_manifest", erosion_px=int(erosion_px))
    if width <= 0 or height <= 0 or (info_width, info_height) != (width, height):
        return _reject(capture_id, frame_id, "size_mismatch", erosion_px=int(erosion_px))

    raw_image_path = image.get("path")
    if not isinstance(raw_image_path, str) or not raw_image_path:
        return _reject(capture_id, frame_id, "image_unreadable", erosion_px=int(erosion_px))
    image_path = Path(raw_image_path)
    if not image_path.is_absolute():
        image_path = base / image_path
    hw = _image_hw(image_path)
    if hw is None:
        return _reject(capture_id, frame_id, "image_unreadable", erosion_px=int(erosion_px))
    got_h, got_w = hw
    if (got_w, got_h) != (width, height):
        return _reject(capture_id, frame_id, "size_mismatch", erosion_px=int(erosion_px))

    try:
        camera = CameraModel.from_mapping(
            {
                "frame_id": frame_id,
                "width": info_width,
                "height": info_height,
                "k": info.get("k", info.get("K")),
                "d": info.get("d", info.get("D", ())),
                "distortion_model": info.get(
                    "distortion_model", info.get("distortion", "plumb_bob")
                ),
                "p": info.get("p", info.get("P")),
            }
        )
    except (ValueError, TypeError):
        return _reject(capture_id, frame_id, "invalid_manifest", erosion_px=int(erosion_px))
    if (camera.width, camera.height) != (width, height):
        return _reject(capture_id, frame_id, "size_mismatch", erosion_px=int(erosion_px))

    raw_pcd_path = cloud.get("path")
    if not isinstance(raw_pcd_path, str) or not raw_pcd_path:
        return _reject(capture_id, frame_id, "pcd_unreadable", erosion_px=int(erosion_px))
    pcd_path = Path(raw_pcd_path)
    if not pcd_path.is_absolute():
        pcd_path = base / pcd_path
    try:
        cloud_proj = projected_cloud(pcd_path, camera, width, height,
                                     float(min_depth_m))
    except (PcdError, OSError, ValueError):
        return _reject(capture_id, frame_id, "pcd_unreadable", erosion_px=int(erosion_px))
    input_point_count = cloud_proj.input_point_count
    if input_point_count == 0:
        return _reject(capture_id, frame_id, "empty_pointcloud",
                        input_point_count=0, erosion_px=int(erosion_px))

    mask = _eroded_mask(quad, width, height, int(erosion_px))
    eroded_area_px = int(np.count_nonzero(mask))

    # Explicit per-point projection in the color frame (never row-order based);
    # computed once per (PCD, camera, raster) by projected_depth_winners.
    if cloud_proj.status in ("empty", "project_error"):
        return _reject(capture_id, frame_id, "no_projected_interior_points",
                        input_point_count=input_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))
    projected_point_count = cloud_proj.projected_point_count
    if cloud_proj.status == "no_in_bounds":
        return _reject(capture_id, frame_id, "no_projected_interior_points",
                        input_point_count=input_point_count,
                        projected_point_count=0,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))

    # Deterministic nearest-depth-wins collision resolution over all
    # in-bounds pixels; ties keep the smallest source row.
    # Only pixels inside the eroded mask are eligible, so winners are
    # resolved over the mask's bounding box only (identical selection/order).
    region = mask_region(mask)
    if region is None:
        fit_xyz = np.zeros((0, 3), dtype=np.float64)
    else:
        zbuf = zbuffer_winners(cloud_proj, region)
        eligible = mask[zbuf.wv, zbuf.wu] > 0
        fit_xyz = np.ascontiguousarray(zbuf.xyz_win[eligible])
    eroded_point_count = int(fit_xyz.shape[0])
    if eroded_point_count < int(min_inliers):
        return _reject(capture_id, frame_id, "insufficient_inliers",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))

    try:
        estimate = fit_plane_ransac_irls(
            fit_xyz,
            frame_id,
            thresh_m=float(ransac_threshold_m),
            iters=int(ransac_iters),
            seed=0,
            min_inliers=int(min_inliers),
        )
    except ValueError:
        return _reject(capture_id, frame_id, "insufficient_inliers",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))
    plane = estimate.to_plane()

    # Calibrated RGB rays through the refined corners; raw depth never read.
    try:
        corners_xyz = np.asarray(
            ray_plane_intersection(quad[:, 0], quad[:, 1], plane, camera).xyz,
            dtype=np.float64,
        ).reshape(4, 3)
    except (ValueError, TypeError):
        return _reject(capture_id, frame_id, "ray_plane_intersection_failed",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))
    if corners_xyz.shape != (4, 3) or not np.all(np.isfinite(corners_xyz)):
        return _reject(capture_id, frame_id, "ray_plane_intersection_failed",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))

    try:
        uv_back = np.asarray(project_xyz(corners_xyz, camera), dtype=np.float64).reshape(4, 2)
    except ValueError:
        return _reject(capture_id, frame_id, "reprojection_failed",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))
    if uv_back.shape != (4, 2) or not np.all(np.isfinite(uv_back)):
        return _reject(capture_id, frame_id, "reprojection_failed",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))
    per_corner = np.linalg.norm(uv_back - quad, axis=1)
    if not np.all(np.isfinite(per_corner)):
        return _reject(capture_id, frame_id, "reprojection_failed",
                        input_point_count=input_point_count,
                        projected_point_count=projected_point_count,
                        eroded_point_count=eroded_point_count,
                        eroded_area_px=eroded_area_px, erosion_px=int(erosion_px))

    return GuidedDepthPlane(
        capture_id=capture_id,
        frame_id=frame_id,
        plane=plane,
        estimate=estimate,
        camera_corners_xyz=np.ascontiguousarray(corners_xyz),
        input_point_count=input_point_count,
        projected_point_count=projected_point_count,
        eroded_point_count=eroded_point_count,
        eroded_area_px=eroded_area_px,
        erosion_px=int(erosion_px),
        reprojection_error_px={
            "per_corner_px": [float(v) for v in per_corner],
            "rms_px": float(np.sqrt(np.mean(per_corner ** 2))),
        },
        status=STATUS_OK,
        reasons=[],
    )
