# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel/pcd.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Minimal PCD v0.7 reader + pinhole projection with a z-buffer.

Only the point coordinates are decoded (``x y z``); any extra fields
(``rgb``, ``normal_*``, ...) are skipped by stride. Supported layouts:

* ``DATA ascii``   — parsed in pure Python/numpy.
* ``DATA binary``  — decoded with numpy (little-endian float32/int32).

``DATA binary_compressed`` (LZF) and anything else raise a clear
prerequisite error: the capture PCDs are plain binary/ascii, and pulling in
an extra decoder dependency is not justified here.

The cloud is treated as UNORGANIZED: point order is never assumed to map to
image pixels. Projection requires explicit pinhole intrinsics (+ optional
extrinsics/distortion from the same config file, with distortion applied
only when explicitly configured) and resolves overlaps with a per-pixel
z-buffer (nearest depth wins). Projected coordinates are quantized to the
nearest pixel *before* the image-bounds test — registered RGB-D clouds are
pixel-locked and floor-style quantization shreds their coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PcdError(RuntimeError):
    pass


def read_pcd_header(path: str | Path) -> tuple[dict, int]:
    """Return (header dict, byte offset where DATA starts)."""
    header: dict[str, str] = {}
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                raise PcdError(f"{path}: truncated file, DATA section not found")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                if text.startswith("#"):
                    header.setdefault("COMMENT", text[1:].strip())
                continue
            parts = text.split(None, 1)
            header[parts[0]] = parts[1] if len(parts) > 1 else ""
            if parts[0] == "DATA":
                break
        offset = fh.tell()
    return header, offset


def _parse_header_lists(header: dict) -> tuple[list[str], list[str], list[str], list[int]]:
    try:
        fields = header["FIELDS"].split()
        sizes = [int(x) for x in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(x) for x in header["COUNT"].split()]
    except KeyError as exc:
        raise PcdError(f"PCD header missing {exc}; not a PCD v0.7 file") from exc
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise PcdError("PCD header FIELDS/SIZE/TYPE/COUNT length mismatch")
    return fields, sizes, types, counts


# ---------------------------------------------------------------- cache
#
# The same capture PCD (~27 MB, ~1.7 M points) is read several times per
# process (depth estimate, depth-edge fix, ...). Parsed clouds are cached
# keyed by file identity (resolved path, device, inode, size, mtime/ctime ns
# and a sampled content digest) + ``max_points``; every call returns a fresh
# copy, so callers may mutate their array freely. Output is identical to an
# uncached read. AUTOLABEL_DISABLE_CACHE=1 bypasses the cache.

_CACHE_MAX = 2
_PCD_CACHE: dict = {}


def _cache_disabled() -> bool:
    import os

    return os.environ.get("AUTOLABEL_DISABLE_CACHE", "") not in ("", "0")


def file_identity(path: str | Path) -> tuple:
    """Stat + sampled-content identity of a file (raises OSError)."""
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


def clear_pcd_cache() -> None:
    _PCD_CACHE.clear()


def _copy_result(xyz: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
    out = dict(info)
    out["fields"] = list(info["fields"])
    return xyz.copy(), out


def read_pcd_xyz(path: str | Path, max_points: int | None = None) -> tuple[np.ndarray, dict]:
    """Return (xyz float32 (N,3), info dict). NaN/Inf rows are dropped and
    counted in ``info``. Cached per process (see above); each call returns
    its own copy."""
    if _cache_disabled():
        return read_pcd_xyz_uncached(path, max_points)
    try:
        key = (file_identity(path), max_points)
        hash(key)
    except (OSError, TypeError):
        # unreadable / odd argument: the uncached reader raises the error
        return read_pcd_xyz_uncached(path, max_points)
    hit = _PCD_CACHE.get(key)
    if hit is None:
        xyz, info = read_pcd_xyz_uncached(path, max_points)
        xyz.setflags(write=False)
        hit = (xyz, info)
        _PCD_CACHE[key] = hit
        while len(_PCD_CACHE) > _CACHE_MAX:
            _PCD_CACHE.pop(next(iter(_PCD_CACHE)))
    return _copy_result(*hit)


def read_pcd_xyz_uncached(path: str | Path, max_points: int | None = None
                          ) -> tuple[np.ndarray, dict]:
    """Uncached :func:`read_pcd_xyz`."""
    path = Path(path)
    if max_points is not None and (not isinstance(max_points, int)
                                   or isinstance(max_points, bool)
                                   or max_points < 0):
        raise PcdError(f"{path}: max_points must be a non-negative integer")
    header, offset = read_pcd_header(path)
    fields, sizes, types, counts = _parse_header_lists(header)
    if "POINTS" not in header:
        raise PcdError(f"{path}: PCD header missing POINTS; not a PCD v0.7 file")
    try:
        points_total = int(header["POINTS"])
        width = int(header.get("WIDTH", points_total))
        height = int(header.get("HEIGHT", 1))
    except ValueError as exc:
        raise PcdError(f"{path}: bad POINTS/WIDTH/HEIGHT header: {exc}") from exc
    if points_total < 0 or width <= 0 or height <= 0:
        raise PcdError(f"{path}: bad POINTS/WIDTH/HEIGHT header "
                       f"(POINTS={header.get('POINTS')!r} WIDTH={header.get('WIDTH')!r} "
                       f"HEIGHT={header.get('HEIGHT')!r})")
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
    # (e.g. rgb, normals) occupy that many whitespace-separated tokens.
    tok_off = {}
    _acc = 0
    for name, c in zip(fields, counts):
        tok_off[name] = _acc
        _acc += c
    n_toks = _acc
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
                # skip leftover header lines defensively
                if toks[0] in ("VERSION", "FIELDS", "SIZE", "TYPE", "COUNT",
                               "WIDTH", "HEIGHT", "VIEWPOINT", "POINTS", "DATA"):
                    continue
                if len(toks) < n_toks:
                    continue
                try:
                    pts.append((float(toks[tx]), float(toks[ty]), float(toks[tz])))
                except (ValueError, IndexError):
                    continue
                if max_points and len(pts) >= max_points:
                    break
        xyz = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    else:
        n = points_total if not max_points else min(points_total, max_points)
        with open(path, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(n * stride)
        if len(raw) < n * stride:
            raise PcdError(f"{path}: truncated DATA ({len(raw)} < {n * stride} bytes)")
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, stride)
        # x/y/z are float32 in these captures; fall back to generic view otherwise
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
        xyz = np.stack([fx, fy, fz], axis=1).astype(np.float32, copy=False)

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
    if info["points_dropped_nonfinite"] == 0 and xyz.flags.c_contiguous \
            and xyz.flags.owndata:
        # all rows finite: xyz[finite] would be an identical fresh copy
        return xyz, info
    return np.ascontiguousarray(xyz[finite]), info


def project_points(xyz: np.ndarray, cam: dict, R=None, t=None,
                   min_depth: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project camera-frame candidate points to pixels.

    Returns (uv float (M,2), depth float (M,), stats dict) for the M points
    with finite depth > min_depth. ``uv`` are continuous pixel coordinates;
    quantization to integer pixels happens in :func:`quantize_uv` /
    :func:`z_buffer` (nearest pixel by default). Applies radial-tangential
    distortion (k1,k2,p1,p2[,k3]) ONLY when nonzero coefficients are
    explicitly present in ``cam`` — never silently inferred.
    """
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    Pc = xyz @ np.asarray(R).T + np.asarray(t).reshape(1, 3)
    z = Pc[:, 2]
    ok = np.isfinite(z) & (z > min_depth)
    stats = {
        "n_input": int(xyz.shape[0]),
        "n_behind_or_too_close": int((~ok).sum()),
    }
    Pc = Pc[ok]
    z = z[ok]
    xn = Pc[:, 0] / z
    yn = Pc[:, 1] / z
    dist = [float(cam.get(k, 0.0) or 0.0) for k in ("k1", "k2", "p1", "p2", "k3")]
    if any(v != 0.0 for v in dist):
        k1, k2, p1, p2, k3 = dist
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    else:
        xd, yd = xn, yn
    u = float(cam["fx"]) * xd + float(cam["cx"])
    v = float(cam["fy"]) * yd + float(cam["cy"])
    return np.stack([u, v], axis=1), z.astype(np.float64), stats


def quantize_uv(uv: np.ndarray, mode: str = "round") -> np.ndarray:
    """Map continuous projected coordinates to pixel indices (float64).

    Returns an (N,2) array holding the integer pixel each coordinate falls
    on, still in float form (so non-finite inputs stay non-finite and huge
    out-of-range values never hit an undefined int cast). Bounds must be
    tested AFTER quantization, on these values.

    * ``"round"`` — nearest-pixel (round half up: pixel ``i`` spans
      ``[i-0.5, i+0.5)``). The safe default: registered RGB-D clouds are
      pixel-locked, i.e. the depth-pixel -> 3D -> projection round trip
      lands within ~1e-4 px of exact integers, and float32 decode noise
      straddles those integers from both sides.
    * ``"floor"`` — legacy half-open convention (pixel ``i`` spans
      ``[i, i+1)``). On pixel-locked clouds it shifts points that settle
      epsilon below an integer into the neighbour pixel, shredding coverage
      and manufacturing z-buffer conflicts.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if mode == "round":
        return np.floor(uv + 0.5)
    if mode == "floor":
        return np.floor(uv)
    raise ValueError(f"unknown quantization mode {mode!r}; use 'round' or 'floor'")


def z_buffer(uv: np.ndarray, depth: np.ndarray, width: int, height: int,
             quantization: str = "round"):
    """Nearest-depth-wins rasterization with explicit pixel quantization.

    ``uv`` is first mapped to integer pixels by ``quantization`` (see
    :func:`quantize_uv`; default: nearest pixel) and the image bounds are
    checked afterwards, so the pixel a point lands on decides inclusion.
    Returns (depth_map inf-init (H,W) float64, index_map -1-init (H,W)
    int64, n_conflicts)."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    q = quantize_uv(uv, quantization)
    finite = (np.isfinite(q).all(axis=1) & np.isfinite(depth)
              if uv.size else np.zeros((0,), dtype=bool))
    # bounds test on the quantized coordinates: NaN/Inf comparisons below
    # are False, so non-finite rows never reach the integer cast
    valid = (finite & (q[:, 0] >= 0) & (q[:, 0] < width)
             & (q[:, 1] >= 0) & (q[:, 1] < height))
    depth_map = np.full((height, width), np.inf)
    index_map = np.full((height, width), -1, dtype=np.int64)
    if not valid.any():
        return depth_map, index_map, 0
    u = q[valid, 0].astype(np.int64)
    v = q[valid, 1].astype(np.int64)
    flat = v * width + u
    d = depth[valid]
    src = np.flatnonzero(valid)
    order = np.argsort(d, kind="stable")
    flat_sorted = flat[order]
    src_sorted = src[order]
    # nearest depth wins: flat_sorted is depth-ordered, so the first
    # occurrence of each pixel holds the minimum depth for that pixel.
    _, first_idx = np.unique(flat_sorted, return_index=True)
    keep = np.zeros_like(flat_sorted, dtype=bool)
    keep[first_idx] = True
    depth_map.flat[flat_sorted[keep]] = d[order][keep]
    index_map.flat[flat_sorted[keep]] = src_sorted[keep]
    n_conflicts = int(valid.sum() - int(keep.sum()))
    return depth_map, index_map, n_conflicts
