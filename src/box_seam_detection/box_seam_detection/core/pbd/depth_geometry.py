# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/depth_geometry.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Plain PCD decoding plus pinhole projection onto explicit camera models.

Only point coordinates (``x y z``) are decoded; any extra fields (``rgb``,
normals, ...) are skipped by stride. Supported layouts:

* ``DATA ascii`` -- parsed in pure Python/NumPy.
* ``DATA binary`` -- decoded with NumPy (little-endian 32-bit).

``DATA binary_compressed`` (LZF) and anything else raise :class:`PcdError`:
the capture PCDs are plain binary/ascii, and an extra decoder dependency is
not justified here.

The cloud is treated as UNORGANIZED: point order is NEVER assumed to map to
image pixels. Projection requires an explicit pinhole camera model
(duck-typed: sibling ``CameraModel``, or a ``dict`` with ``fx/fy/cx/cy``)
plus an optional explicit rigid transform, and overlaps resolve with a
per-pixel z-buffer (nearest depth wins).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PcdError(RuntimeError):
    """Explicit failure to decode a PCD file (not a PCD, unsupported layout)."""


# ---------------------------------------------------------------------------
# PCD decoding
# ---------------------------------------------------------------------------

def _read_header(path: Path) -> tuple[dict, int]:
    """Return (header dict, byte offset where DATA starts)."""
    header: dict[str, str] = {}
    offset = 0
    with open(path, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                raise PcdError(f"{path}: truncated header (no DATA line)")
            offset = pos + len(line)
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise PcdError(f"{path}: non-ascii PCD header bytes") from exc
            if not text or text.startswith("#"):
                continue
            parts = text.split(None, 1)
            key = parts[0].upper()
            header[key] = parts[1] if len(parts) > 1 else ""
            if key == "DATA":
                break
    return header, offset


def _header_lists(header: dict) -> tuple[list[str], list[int], list[str], list[int]]:
    try:
        fields = header["FIELDS"].split()
        sizes = [int(v) for v in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(v) for v in header["COUNT"].split()]
    except KeyError as exc:
        raise PcdError(f"PCD header missing {exc}; not a PCD v0.7 file") from exc
    except ValueError as exc:
        raise PcdError(f"bad PCD SIZE/COUNT header: {exc}") from exc
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise PcdError(
            "PCD FIELDS/SIZE/TYPE/COUNT length mismatch "
            f"({len(fields)}/{len(sizes)}/{len(types)}/{len(counts)})"
        )
    return fields, sizes, [t.upper() for t in types], counts


def read_pcd_xyz(
    path: str | Path, max_points: int | None = None
) -> tuple[np.ndarray, dict]:
    """Return (xyz float32 (N,3) in meters, info dict).

    NaN/Inf rows are dropped and counted in ``info``. ``max_points`` caps the
    number of decoded points (deterministic head-truncation); ``None`` reads
    all ``POINTS``. Raises :class:`PcdError` on any structural problem.
    """
    path = Path(path)
    if not path.is_file():
        raise PcdError(f"{path}: PCD file not found")
    if max_points is not None and (
        not isinstance(max_points, int)
        or isinstance(max_points, bool)
        or max_points < 0
    ):
        raise PcdError(f"{path}: max_points must be a non-negative integer")
    header, offset = _read_header(path)
    fields, sizes, types, counts = _header_lists(header)
    if "POINTS" not in header:
        raise PcdError(f"{path}: PCD header missing POINTS; not a PCD v0.7 file")
    try:
        points_total = int(header["POINTS"])
        width = int(header.get("WIDTH", points_total))
        height = int(header.get("HEIGHT", 1))
    except ValueError as exc:
        raise PcdError(f"{path}: bad POINTS/WIDTH/HEIGHT header: {exc}") from exc
    if points_total < 0 or width <= 0 or height <= 0:
        raise PcdError(
            f"{path}: bad POINTS/WIDTH/HEIGHT header "
            f"(POINTS={header.get('POINTS')!r} WIDTH={header.get('WIDTH')!r} "
            f"HEIGHT={header.get('HEIGHT')!r})"
        )
    data_mode = header.get("DATA", "").strip().lower()
    if data_mode not in ("ascii", "binary"):
        raise PcdError(
            f"{path}: unsupported DATA '{header.get('DATA', '').strip()}' "
            "(decoder supports ascii + binary only; "
            "binary_compressed needs an LZF decoder that is not installed)"
        )
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise PcdError(f"{path}: no '{axis}' field in PCD (fields: {fields})")
        if counts[fields.index(axis)] != 1:
            raise PcdError(
                f"{path}: '{axis}' field must have COUNT 1 "
                f"(got {counts[fields.index(axis)]})"
            )

    idx = {name: i for i, name in enumerate(fields)}
    item_sizes = [s * c for s, c in zip(sizes, counts)]
    stride = sum(item_sizes)
    # Token offset of each field in an ASCII row: fields with COUNT > 1
    # (e.g. rgb-as-triplet, normals) occupy that many tokens.
    tok_off: dict[str, int] = {}
    acc = 0
    for name, cnt in zip(fields, counts):
        tok_off[name] = acc
        acc += cnt
    n_toks = acc
    xi_off = sum(item_sizes[: idx["x"]])
    yi_off = sum(item_sizes[: idx["y"]])
    zi_off = sum(item_sizes[: idx["z"]])

    if data_mode == "ascii":
        pts: list[tuple[float, float, float]] = []
        tx, ty, tz = tok_off["x"], tok_off["y"], tok_off["z"]
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if len(pts) >= points_total:
                    break
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                # Skip leftover header lines defensively.
                if toks[0] in (
                    "VERSION", "FIELDS", "SIZE", "TYPE", "COUNT",
                    "WIDTH", "HEIGHT", "VIEWPOINT", "POINTS", "DATA",
                ):
                    continue
                if len(toks) < n_toks:
                    continue
                try:
                    pts.append((float(toks[tx]), float(toks[ty]), float(toks[tz])))
                except (ValueError, IndexError):
                    continue
                if max_points is not None and len(pts) >= max_points:
                    break
        xyz = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    else:
        n = points_total if max_points is None else min(points_total, max_points)
        with open(path, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(n * stride)
        if len(raw) < n * stride:
            raise PcdError(f"{path}: truncated DATA ({len(raw)} < {n * stride} bytes)")

        buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, stride)

        def _col(off: int, size: int, typ: str) -> np.ndarray:
            b = buf[:, off: off + size]
            if typ == "F" and size == 4:
                return b.view("<f4").reshape(n)
            if typ == "I" and size == 4:
                return b.view("<i4").reshape(n).astype(np.float32)
            if typ == "U" and size == 4:
                return b.view("<u4").reshape(n).astype(np.float32)
            raise PcdError(
                f"{path}: unsupported x/y/z column encoding ({typ}{size}); "
                "decoder handles 32-bit float/int only"
            )

        fx = _col(xi_off, sizes[idx["x"]], types[idx["x"]])
        fy = _col(yi_off, sizes[idx["y"]], types[idx["y"]])
        fz = _col(zi_off, sizes[idx["z"]], types[idx["z"]])
        xyz = np.stack([fx, fy, fz], axis=1).astype(np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PcdError(f"{path}: decoded no xyz triplets (fields: {fields})")
    finite = np.isfinite(xyz).all(axis=1)
    info = {
        "points_total": int(points_total),
        "points_read": int(xyz.shape[0]),
        "points_dropped_nonfinite": int((~finite).sum()),
        "width": int(width),
        "height": int(height),
        "organized": bool(width > 1 and height > 1 and width * height == points_total),
        "fields": fields,
        "data": data_mode,
    }
    return np.ascontiguousarray(xyz[finite]), info


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _intrinsics(camera_model) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a duck-typed pinhole camera model.

    Accepts the sibling ``CameraModel`` (``fx/fy/cx/cy`` attributes, or flat-9
    / 3x3 ``k``/``K``), or a plain ``dict`` with ``fx/fy/cx/cy`` keys.
    """
    if isinstance(camera_model, dict):
        try:
            fx = float(camera_model["fx"])
            fy = float(camera_model["fy"])
            cx = float(camera_model["cx"])
            cy = float(camera_model["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "camera dict needs finite fx/fy/cx/cy keys; "
                f"got keys {sorted(camera_model.keys())}"
            ) from exc
    else:
        fx = fy = cx = cy = None
        if all(hasattr(camera_model, a) for a in ("fx", "fy", "cx", "cy")):
            fx = float(camera_model.fx)
            fy = float(camera_model.fy)
            cx = float(camera_model.cx)
            cy = float(camera_model.cy)
        else:
            k = getattr(camera_model, "k", getattr(camera_model, "K", None))
            if k is not None:
                m = np.asarray(k, dtype=np.float64).reshape(-1)
                if m.size != 9:
                    raise ValueError(
                        f"camera k/K must hold 9 numbers, got {m.size}"
                    )
                m = m.reshape(3, 3)
                fx, fy, cx, cy = (
                    float(m[0, 0]), float(m[1, 1]),
                    float(m[0, 2]), float(m[1, 2]),
                )
            else:
                raise ValueError(
                    f"camera_model {type(camera_model).__name__} exposes neither "
                    "fx/fy/cx/cy nor k/K; pass an explicit pinhole model"
                )
    for name, val in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        if not np.isfinite(val):
            raise ValueError(f"camera {name} must be finite, got {val!r}")
    if not (fx > 0 and fy > 0):
        raise ValueError(f"camera fx/fy must be positive (got {fx!r}, {fy!r})")
    return fx, fy, cx, cy


def project_points(
    xyz: np.ndarray,
    camera_model,
    T_target_source=None,
    *,
    min_depth_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project source-frame points into the target pinhole camera.

    ``T_target_source`` is an optional 4x4 rigid transform (target-from-source);
    ``None`` means the points are already in the camera frame. Returns
    ``(uv, depth, valid)``: ``uv`` float64 (N,2) pixel coordinates (NaN where
    invalid), ``depth`` float64 (N,) forward depth in meters, ``valid`` bool
    (N,) for ``depth > min_depth_m`` with finite projection. Point order is
    preserved but carries no pixel meaning -- correspondence comes only from
    this explicit projection.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got shape {pts.shape}")
    if not np.isfinite(min_depth_m) or min_depth_m <= 0:
        raise ValueError(f"min_depth_m must be positive, got {min_depth_m!r}")
    fx, fy, cx, cy = _intrinsics(camera_model)
    if T_target_source is not None:
        T = np.asarray(T_target_source, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("T_target_source must be a finite 4x4 matrix")
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts = (homo @ T.T)[:, :3]
    n = pts.shape[0]
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    depth = pts[:, 2].astype(np.float64, copy=True)
    finite = np.isfinite(pts).all(axis=1)
    valid = finite & (depth > min_depth_m)
    if valid.any():
        uv[valid, 0] = fx * pts[valid, 0] / depth[valid] + cx
        uv[valid, 1] = fy * pts[valid, 1] / depth[valid] + cy
        valid &= np.isfinite(uv).all(axis=1)
        uv[~valid] = np.nan
    else:
        valid = np.zeros(n, dtype=bool)
    return uv, depth, valid


def z_buffer(
    uv: np.ndarray, depth: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Nearest-depth-wins rasterization onto a (height, width) depth map.

    Continuous coordinates quantize to the nearest pixel (``floor(u + 0.5)``)
    before the bounds test; out-of-bounds, non-finite, and non-positive depths
    are ignored. Ties keep the first point (strict ``<`` comparison), so the
    result is deterministic for a fixed input order. Returns float64 meters
    with 0.0 where unobserved.
    """
    if not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if not isinstance(height, (int, np.integer)) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must be (N,2), got shape {uv.shape}")
    if depth.ndim != 1 or depth.shape[0] != uv.shape[0]:
        raise ValueError(
            f"depth must be (N,) matching uv, got {depth.shape} vs {uv.shape}"
        )
    depth_map = np.zeros((int(height), int(width)), dtype=np.float64)
    if uv.shape[0] == 0:
        return depth_map
    px = np.floor(uv[:, 0] + 0.5).astype(np.int64)
    py = np.floor(uv[:, 1] + 0.5).astype(np.int64)
    ok = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
        & (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
    )
    for x, y, z in zip(px[ok], py[ok], depth[ok]):
        prev = depth_map[y, x]
        if prev == 0.0 or z < prev:
            depth_map[y, x] = z
    return depth_map
