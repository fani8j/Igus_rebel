"""Speed-path equivalence: every fast path returns exactly what the code it
replaced returned (bit-identical arrays / results).

The ``_ref_*`` functions below are verbatim copies of the pre-speed-up
implementations (tag live-v1-2026-09-28) so the comparison does not depend
on git. Real-capture tests skip when the capture files are not present.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402
from box_seam_detection.core.autolabel import pcd as _pcd  # noqa: E402
from box_seam_detection.core import corner_checks as _cc  # noqa: E402
from box_seam_detection.core import pipeline as _pipe  # noqa: E402
from box_seam_detection.core import seam_estimator as _seam  # noqa: E402
from box_seam_detection.core.pbd import sam3_depth_fusion as _fusion  # noqa: E402
from box_seam_detection.core.pbd.camera_model import CameraModel, project_xyz  # noqa: E402
from box_seam_detection.core.pbd.depth_geometry import read_pcd_xyz as _dg_read  # noqa: E402

CAPTURES = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures")
REAL_IDS = ("frame_000001_20260928_053501_601033",
            "frame_000001_20260928_055906_324334")


def _real(cid):
    paths = {"pcd": CAPTURES / "pointcloud" / f"{cid}.pcd",
             "manifest": CAPTURES / "captures" / f"{cid}.json",
             "image": CAPTURES / "rgb" / f"{cid}.png"}
    if not all(p.is_file() for p in paths.values()):
        pytest.skip(f"real capture {cid} not available")
    return paths


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_DISABLE_CACHE", raising=False)
    _pcd.clear_pcd_cache()
    _fusion.clear_caches()
    yield
    _pcd.clear_pcd_cache()
    _fusion.clear_caches()


# ---------------------------------------------------------------------------
# reference implementations (verbatim pre-speed-up code)


def _ref_read_pcd_xyz(path, max_points=None):
    path = Path(path)
    header, offset = _pcd.read_pcd_header(path)
    fields, sizes, types, counts = _pcd._parse_header_lists(header)
    points_total = int(header["POINTS"])
    width = int(header.get("WIDTH", points_total))
    height = int(header.get("HEIGHT", 1))
    data_mode = header.get("DATA", "").strip().lower()
    idx = {name: i for i, name in enumerate(fields)}
    item_sizes = [s * c for s, c in zip(sizes, counts)]
    stride = sum(item_sizes)
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
        pts = []
        tx, ty, tz = tok_off["x"], tok_off["y"], tok_off["z"]
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if len(pts) >= points_total:
                    break
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
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
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, stride)

        def _col(off, size, typ):
            b = buf[:, off: off + size]
            if typ == "F" and size == 4:
                return b.view("<f4").reshape(n)
            if typ == "I" and size == 4:
                return b.view("<i4").reshape(n).astype(np.float32)
            if typ == "U" and size == 4:
                return b.view("<u4").reshape(n).astype(np.float32)
            raise AssertionError

        fx = _col(xi_off, sizes[idx["x"]], types[idx["x"]])
        fy = _col(yi_off, sizes[idx["y"]], types[idx["y"]])
        fz = _col(zi_off, sizes[idx["z"]], types[idx["z"]])
        xyz = np.stack([fx, fy, fz], axis=1).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    info = {
        "points_total": int(points_total),
        "points_read": int(xyz.shape[0]),
        "points_dropped_nonfinite": int((~finite).sum()),
        "width": int(width), "height": int(height),
        "organized": bool(width > 1 and height > 1 and width * height == points_total),
        "fields": fields, "data": data_mode,
    }
    return np.ascontiguousarray(xyz[finite]), info


def _ref_zbuffer(pcd_path, camera, width, height, min_depth_m=0.05):
    """Pre-speed-up body of fit_sam3_guided_depth_plane (projection +
    nearest-depth-wins), returning the winners in their original order."""
    xyz, _ = _dg_read(pcd_path)
    xyz = np.asarray(xyz, dtype=np.float64)
    gated = np.ascontiguousarray(xyz[xyz[:, 2] >= float(min_depth_m)])
    uv = np.asarray(project_xyz(gated, camera), dtype=np.float64)
    uu = np.floor(uv[:, 0]).astype(np.int64)
    vv = np.floor(uv[:, 1]).astype(np.int64)
    in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
    bu, bv = uu[in_bounds], vv[in_bounds]
    bz = gated[in_bounds][:, 2]
    brow = np.flatnonzero(in_bounds)
    lin = bv * np.int64(width) + bu
    order = np.lexsort((brow, bz))
    _, first = np.unique(lin[order], return_index=True)
    winners = order[first]
    return (bu[winners], bv[winners], gated[in_bounds][winners],
            int(np.count_nonzero(in_bounds)))


def _ref_corner_depth_winners(pcd_path, camera, plane_n, plane_d, W, H):
    xyz, _ = _dg_read(pcd_path)
    xyz = np.asarray(xyz, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= 0.05)
    gated = np.ascontiguousarray(xyz[finite])
    uv = np.asarray(project_xyz(gated, camera), dtype=np.float64)
    uu = np.floor(uv[:, 0]).astype(np.int64)
    vv = np.floor(uv[:, 1]).astype(np.int64)
    inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    lin = vv[inb].astype(np.int64) * np.int64(W) + uu[inb].astype(np.int64)
    order = np.argsort(gated[inb][:, 2], kind="stable")
    _, first = np.unique(lin[order], return_index=True)
    winners = order[first]
    uv_win = np.stack([uu[inb][winners], vv[inb][winners]], axis=1).astype(np.int64)
    res_mm = (gated[inb][winners] @ plane_n + plane_d) * 1000.0
    return uv_win, np.asarray(res_mm, dtype=np.float64)


def _ref_corner_depth_frac(uv_win, res_win, corner_px, radius_px, inlier_mm=8.0):
    c = np.asarray(corner_px, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(c)):
        return None, 0
    uv = np.asarray(uv_win, dtype=np.float64).reshape(-1, 2)
    rs = np.asarray(res_win, dtype=np.float64).reshape(-1)
    if len(uv) != len(rs) or len(uv) == 0:
        return None, 0
    dd = np.sqrt(((uv - c) ** 2).sum(axis=1))
    sel = dd <= float(radius_px)
    n = int(sel.sum())
    if n == 0:
        return None, 0
    return float((np.abs(rs[sel]) < float(inlier_mm)).mean()), n


def _ref_gap_slice_support(v, prior, halfwin, min_depth, min_shoulder, max_width):
    v = np.asarray(v, dtype=float)
    n = len(v)
    lo, hi = max(8, prior - halfwin), min(n - 9, prior + halfwin)
    if hi < lo:
        return None
    vs = cv2.GaussianBlur(v.reshape(-1, 1), (1, 3), 0)[:, 0]
    best = None
    for j in range(lo, hi + 1):
        f = float(vs[j])
        lseg = vs[max(0, j - 8):max(0, j - 1)]
        rseg = vs[j + 2:min(n, j + 9)]
        if len(lseg) < 2 or len(rseg) < 2:
            continue
        lref, rref = float(lseg.max()), float(rseg.max())
        depth = min(lref, rref) - f
        if depth < min_depth:
            continue
        if lref - f < min_shoulder or rref - f < min_shoulder:
            continue
        mid = min(lref, rref) - depth / 2.0
        a = j
        while a - 1 >= max(0, j - 9) and vs[a - 1] < mid:
            a -= 1
        b = j
        while b + 1 < min(n, j + 10) and vs[b + 1] < mid:
            b += 1
        width = int(b - a + 1)
        if width > max_width:
            continue
        j0 = max(1, min(n - 2, j))
        y0, y1, y2 = v[j0 - 1], v[j0], v[j0 + 1]
        den = y0 - 2.0 * y1 + y2
        off = (y0 - y2) / (2.0 * den) if den > 1e-9 else 0.0
        cc = float(j0 + float(np.clip(off, -1.0, 1.0)))
        if best is None or depth > best[0]:
            best = (depth, cc, depth, width)
    if best is None:
        return None
    return {"center": best[1], "depth": best[2], "width": best[3]}


def _ref_workspace_mask(xyz, dcfg):
    xyz = np.asarray(xyz, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    return (finite
            & (xyz[:, 0] >= float(dcfg["ws_x_min"])) & (xyz[:, 0] <= float(dcfg["ws_x_max"]))
            & (xyz[:, 1] >= float(dcfg["ws_y_min"])) & (xyz[:, 1] <= float(dcfg["ws_y_max"]))
            & (xyz[:, 2] >= float(dcfg["ws_z_min"])) & (xyz[:, 2] <= float(dcfg["ws_z_max"])))


def _ref_project_valid_cloud(query, points_flat, fx, fy, cx, cy, w, h, distortion=None):
    xyz = np.column_stack((np.asarray(points_flat["x"], dtype=np.float64),
                           np.asarray(points_flat["y"], dtype=np.float64),
                           np.asarray(points_flat["z"], dtype=np.float64)))
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
    xyz = xyz[valid]
    if xyz.shape[0] == 0:
        raise ValueError("point cloud contains no finite positive-depth points")
    if distortion is None:
        u = fx * xyz[:, 0] / xyz[:, 2] + cx
        v = fy * xyz[:, 1] / xyz[:, 2] + cy
        in_image = (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
    else:
        xn = xyz[:, 0] / xyz[:, 2]
        yn = xyz[:, 1] / xyz[:, 2]
        r2_limit = 1.5 * query._max_undistorted_radius2(fx, fy, cx, cy, w, h, distortion)
        xd, yd = query._distort_normalized(xn, yn, distortion)
        u = fx * xd + cx
        v = fy * yd + cy
        in_image = (xn * xn + yn * yn <= r2_limit) & \
            (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
    return xyz[in_image], u[in_image], v[in_image]


def _ref_inside_polygon(u, v, polygon, inset_px):
    area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) -
                        np.roll(polygon[:, 0], -1) * polygon[:, 1])
    if abs(area) < 1e-6:
        raise ValueError("detected carton polygon is degenerate")
    orientation = 1.0 if area > 0.0 else -1.0
    inside = np.ones(u.shape, dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = end - start
        length = math.hypot(dx, dy)
        if length < 1e-6:
            raise ValueError("detected carton polygon has duplicate corners")
        signed_distance = orientation * (dx * (v - start[1]) - dy * (u - start[0])) / length
        inside &= signed_distance >= inset_px
    return inside


# ---------------------------------------------------------------------------
# helpers


def _write_pcd(path, xyz, mode="binary", extra_rgb=True):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if extra_rgb:
        fields, sizes, types, counts = "x y z rgb", "4 4 4 4", "F F F F", "1 1 1 1"
    else:
        fields, sizes, types, counts = "x y z", "4 4 4", "F F F", "1 1 1"
    head = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\n"
            f"COUNT {counts}\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\nDATA {mode}\n")
    if mode == "binary":
        cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        if extra_rgb:
            cols.append(np.full(n, 1.5, dtype=np.float32))
        rec = np.stack(cols, axis=1).astype("<f4")
        Path(path).write_bytes(head.encode() + rec.tobytes())
    else:
        lines = [" ".join(repr(float(v)) for v in row) + (" 1.5" if extra_rgb else "")
                 for row in xyz]
        Path(path).write_text(head + "\n".join(lines) + "\n")


def _camera(W=640, H=480, d=()):
    return CameraModel.from_mapping(
        {"frame_id": "camera_color_optical_frame", "width": W, "height": H,
         "k": [500.0, 0.0, W / 2.0 + 0.3, 0.0, 505.0, H / 2.0 - 0.7, 0.0, 0.0, 1.0],
         "d": list(d), "distortion_model": "plumb_bob",
         "p": [500.0, 0.0, W / 2.0 + 0.3, 0.0, 0.0, 505.0, H / 2.0 - 0.7, 0.0,
               0.0, 0.0, 1.0, 0.0]})


def _synthetic_cloud(seed=0, n=60000):
    rng = np.random.default_rng(seed)
    top = np.c_[rng.uniform(-0.1, 0.1, n), rng.uniform(-0.08, 0.08, n),
                0.5 + rng.normal(0, 0.001, n)]
    floor = np.c_[rng.uniform(-0.4, 0.4, n), rng.uniform(-0.3, 0.3, n),
                  0.8 + rng.normal(0, 0.002, n)]
    xyz = np.concatenate([top, floor]).astype(np.float32)
    xyz[::997] = np.nan
    # exact duplicates -> z-buffer ties broken by source row
    xyz[5:50] = xyz[min(1000, len(xyz) - 1)]
    return xyz


def _camera_from_manifest(manifest):
    info = manifest["camera_info"]
    return CameraModel.from_mapping(
        {"frame_id": "camera_color_optical_frame",
         "width": int(manifest["image"]["width"]),
         "height": int(manifest["image"]["height"]),
         "k": info.get("k"), "d": info.get("d", ()),
         "distortion_model": info.get("distortion_model", "plumb_bob"),
         "p": info.get("p")})


# ---------------------------------------------------------------------------
# autolabel.pcd


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_pcd_reader_matches_reference_synthetic(tmp_path, mode):
    xyz = _synthetic_cloud(n=3000)
    xyz[7] = [np.inf, 0.0, 1.0]
    path = tmp_path / f"c_{mode}.pcd"
    _write_pcd(path, xyz, mode=mode)
    ref, ref_info = _ref_read_pcd_xyz(path)
    for fn in (_pcd.read_pcd_xyz_uncached, _pcd.read_pcd_xyz, _pcd.read_pcd_xyz):
        got, info = fn(path)
        assert _same(got, ref) and got.flags.c_contiguous and got.flags.writeable
        assert info == ref_info


def test_pcd_reader_all_finite_fast_path(tmp_path):
    xyz = np.random.default_rng(3).normal(size=(5000, 3)).astype(np.float32)
    path = tmp_path / "finite.pcd"
    _write_pcd(path, xyz, extra_rgb=False)
    ref, ref_info = _ref_read_pcd_xyz(path)
    got, info = _pcd.read_pcd_xyz_uncached(path)
    assert _same(got, ref) and info == ref_info and ref_info["points_dropped_nonfinite"] == 0
    got_mp, _ = _pcd.read_pcd_xyz(path, max_points=100)
    assert _same(got_mp, _ref_read_pcd_xyz(path, max_points=100)[0])


def test_pcd_cache_returns_independent_copies_and_tracks_rewrites(tmp_path):
    path = tmp_path / "c.pcd"
    xyz = _synthetic_cloud(n=2000)
    _write_pcd(path, xyz)
    a, info_a = _pcd.read_pcd_xyz(path)
    a[:] = 123.0  # caller mutation must not leak into the cache
    info_a["fields"].append("junk")
    b, info_b = _pcd.read_pcd_xyz(path)
    assert _same(b, _ref_read_pcd_xyz(path)[0]) and "junk" not in info_b["fields"]
    # same-size in-place rewrite (possibly within the mtime granularity)
    xyz2 = xyz.copy()
    xyz2[100:200] += 1.0
    _write_pcd(path, xyz2)
    c, _ = _pcd.read_pcd_xyz(path)
    assert _same(c, _ref_read_pcd_xyz(path)[0])
    assert not _same(c, b)


def test_pcd_cache_disabled_env(tmp_path, monkeypatch):
    path = tmp_path / "c.pcd"
    _write_pcd(path, _synthetic_cloud(n=500))
    monkeypatch.setenv("AUTOLABEL_DISABLE_CACHE", "1")
    got, _ = _pcd.read_pcd_xyz(path)
    assert _same(got, _ref_read_pcd_xyz(path)[0]) and not _pcd._PCD_CACHE


def test_pcd_missing_file_still_raises(tmp_path):
    with pytest.raises(OSError):
        _pcd.read_pcd_xyz(tmp_path / "nope.pcd")


@pytest.mark.parametrize("cid", REAL_IDS)
def test_pcd_reader_matches_reference_real(cid):
    paths = _real(cid)
    ref, ref_info = _ref_read_pcd_xyz(paths["pcd"])
    uncached, info_u = _pcd.read_pcd_xyz_uncached(paths["pcd"])
    first, info_1 = _pcd.read_pcd_xyz(paths["pcd"])
    cached, info_2 = _pcd.read_pcd_xyz(paths["pcd"])
    for got, info in ((uncached, info_u), (first, info_1), (cached, info_2)):
        assert _same(got, ref) and info == ref_info
    # the two PCD readers agree on the capture clouds
    assert _same(_dg_read(paths["pcd"])[0], ref)


# ---------------------------------------------------------------------------
# sam3_depth_fusion z-buffer cache


@pytest.mark.parametrize("d", [(), (0.08, -0.05, 0.001, -0.0005, 0.01)])
def test_zbuffer_winners_match_reference_synthetic(tmp_path, d):
    path = tmp_path / "z.pcd"
    _write_pcd(path, _synthetic_cloud())
    cam = _camera(d=d)
    ru, rv, rxyz, rcount = _ref_zbuffer(path, cam, 640, 480)
    full = _fusion.projected_depth_winners(path, cam, 640, 480)
    assert full.status == "ok" and full.projected_point_count == rcount
    assert _same(full.wu, ru) and _same(full.wv, rv) and _same(full.xyz_win, rxyz)
    again = _fusion.projected_depth_winners(path, cam, 640, 480)
    assert again is full  # memoised
    with pytest.raises(ValueError):
        full.xyz_win[0, 0] = 0.0  # read-only cache
    lin = rv * 640 + ru
    rng = np.random.default_rng(1)
    for _ in range(25):
        u0, v0 = int(rng.integers(-5, 600)), int(rng.integers(-5, 450))
        region = (u0, v0, u0 + int(rng.integers(0, 200)), v0 + int(rng.integers(0, 200)))
        sub = _fusion.projected_depth_winners(path, cam, 640, 480, region=region)
        keep = ((ru >= region[0]) & (ru <= region[2])
                & (rv >= region[1]) & (rv <= region[3]))
        assert _same(sub.wu, ru[keep]) and _same(sub.wv, rv[keep])
        assert _same(sub.xyz_win, rxyz[keep])
        assert np.all(np.diff(lin[keep]) > 0)


def test_zbuffer_status_paths(tmp_path):
    cam = _camera()
    behind = tmp_path / "behind.pcd"
    _write_pcd(behind, np.array([[0.0, 0.0, 0.01]] * 10, dtype=np.float32))
    assert _fusion.projected_depth_winners(behind, cam, 640, 480).status == "empty"
    outside = tmp_path / "outside.pcd"
    _write_pcd(outside, np.array([[50.0, 0.0, 1.0]] * 10, dtype=np.float32))
    assert _fusion.projected_depth_winners(outside, cam, 640, 480).status == "no_in_bounds"


def test_image_shape_cache_matches_decode(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.zeros((33, 47, 3), np.uint8))
    assert _fusion._image_hw(p) == (33, 47)
    _fusion.remember_image_shape(p, (33, 47, 3))
    assert _fusion._image_hw(p) == (33, 47)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    assert _fusion._image_hw(bad) is None


def _guided_fields(g):
    out = {k: getattr(g, k) for k in ("status", "reasons", "input_point_count",
                                       "projected_point_count", "eroded_point_count",
                                       "eroded_area_px", "reprojection_error_px")}
    out["reprojection_error_px"] = json.dumps(out["reprojection_error_px"])  # NaN-safe
    out["corners"] = np.asarray(g.camera_corners_xyz).tobytes()
    if g.plane is not None:
        out["plane"] = (np.asarray(g.plane.normal).tobytes(), float(g.plane.d))
        est = g.estimate
        out["est"] = (est.rms_mm, est.n_inliers, est.inlier_ratio,
                      np.asarray(est.inlier_mask).tobytes())
    return out


def _synthetic_manifest(tmp_path, xyz, W=640, H=480):
    pcd_path = tmp_path / "g.pcd"
    _write_pcd(pcd_path, xyz)
    img = tmp_path / "g.png"
    cv2.imwrite(str(img), np.zeros((H, W, 3), np.uint8))
    frame = {"frame_id": "camera_color_optical_frame"}
    return {"schema": "rgbd_capture/v1", "capture_id": "synth",
            "image": {"path": str(img), "width": W, "height": H, "header": frame},
            "pointcloud": {"path": str(pcd_path), "header": frame},
            "camera_info": {"width": W, "height": H, "header": frame,
                            "k": [500.0, 0.0, 320.3, 0.0, 505.0, 239.3, 0.0, 0.0, 1.0],
                            "d": [], "distortion_model": "plumb_bob"}}


def test_guided_plane_cached_equals_uncached_synthetic(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path, _synthetic_cloud())
    quad = np.array([[260.0, 170.0], [380.0, 172.0], [382.0, 305.0], [258.0, 303.0]])
    cached = _fusion.fit_sam3_guided_depth_plane(manifest, quad)
    cached2 = _fusion.fit_sam3_guided_depth_plane(manifest, quad)
    monkeypatch.setenv("AUTOLABEL_DISABLE_CACHE", "1")
    uncached = _fusion.fit_sam3_guided_depth_plane(manifest, quad)
    assert cached.status == "OK"
    assert _guided_fields(cached) == _guided_fields(uncached) == _guided_fields(cached2)
    tiny = np.array([[300.0, 200.0], [303.0, 200.0], [303.0, 203.0], [300.0, 203.0]])
    monkeypatch.delenv("AUTOLABEL_DISABLE_CACHE")
    a = _fusion.fit_sam3_guided_depth_plane(manifest, tiny)
    monkeypatch.setenv("AUTOLABEL_DISABLE_CACHE", "1")
    b = _fusion.fit_sam3_guided_depth_plane(manifest, tiny)
    assert a.status == b.status == "REJECTED" and a.reasons == b.reasons
    assert _guided_fields(a) == _guided_fields(b)


@pytest.mark.parametrize("cid", REAL_IDS)
def test_real_capture_depth_helpers_match_reference(cid, monkeypatch):
    paths = _real(cid)
    manifest = json.loads(paths["manifest"].read_text())
    W, H = int(manifest["image"]["width"]), int(manifest["image"]["height"])
    rec_dirs = sorted((CAPTURES / "pipeline" / cid).glob(f"*/{cid}_autolabel_v2_record.json"))
    if not rec_dirs:
        pytest.skip("no stored record")
    record = json.loads(rec_dirs[-1].read_text())
    quad = np.array([record["top_face"]["corners_px"][k] for k in
                     ("back_left", "back_right", "front_right", "front_left")])
    cam = _camera_from_manifest(manifest)
    # (1) guided plane: cached == uncached == pre-speed-up
    g_cached = _fusion.fit_sam3_guided_depth_plane(manifest, quad)
    monkeypatch.setenv("AUTOLABEL_DISABLE_CACHE", "1")
    g_uncached = _fusion.fit_sam3_guided_depth_plane(manifest, quad)
    monkeypatch.delenv("AUTOLABEL_DISABLE_CACHE")
    assert g_cached.status == "OK"
    assert _guided_fields(g_cached) == _guided_fields(g_uncached)
    ru, rv, rxyz, _ = _ref_zbuffer(paths["pcd"], cam, W, H)
    full = _fusion.projected_depth_winners(paths["pcd"], cam, W, H)
    assert _same(full.wu, ru) and _same(full.wv, rv) and _same(full.xyz_win, rxyz)
    # (2) all-point p95 == reference (mask over the reference winners)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=int(g_cached.erosion_px))
    pts = np.ascontiguousarray(rxyz[mask[rv, ru] > 0])
    n = np.asarray(g_cached.plane.normal, dtype=np.float64).ravel()
    d = float(g_cached.plane.d)
    ref_p95 = float(np.percentile(np.abs(pts @ n + d) * 1000.0, 95))
    assert _pipe._plane_p95_all_mm(g_cached, quad, W, H, manifest) == ref_p95
    # (3) corner winners: full == reference; region-limited -> same report
    ref_uv, ref_res = _ref_corner_depth_winners(paths["pcd"], cam, n, d, W, H)
    uv_full, res_full = _pipe._corner_depth_winners(manifest, g_cached.plane, W, H)
    assert _same(uv_full, ref_uv) and _same(res_full, ref_res)
    region = _cc.depth_query_region(quad, [W, H], {})
    uv_reg, res_reg = _pipe._corner_depth_winners(manifest, g_cached.plane, W, H,
                                                  region=region)
    gray = cv2.cvtColor(cv2.imread(str(paths["image"])), cv2.COLOR_BGR2GRAY).astype(np.float64)
    coarse = quad + 1.5
    rep_ref = _cc.check_corners(gray, coarse, quad, [W, H], uv_win=ref_uv, res_win=ref_res)
    rep_new = _cc.check_corners(gray, coarse, quad, [W, H], uv_win=uv_reg, res_win=res_reg)
    assert rep_ref == rep_new
    for k in range(4):
        r = 3.0 * W / 1280.0
        assert _cc.corner_depth_frac(uv_reg, res_reg, quad[k], r) == \
            _ref_corner_depth_frac(ref_uv, ref_res, quad[k], r)


# ---------------------------------------------------------------------------
# corner checks


def test_corner_depth_frac_matches_reference_random():
    rng = np.random.default_rng(5)
    uv = rng.integers(0, 200, size=(20000, 2))
    res = rng.normal(0, 10, size=20000)
    uv_f = uv.astype(np.float64)
    uv_f[::101] = np.nan
    for _ in range(200):
        c = rng.uniform(-5, 205, size=2)
        r = float(rng.choice([0.0, 0.5, 1.0, 2.25, 3.0, 4.5, 10.0]))
        for u_in in (uv, uv_f):
            assert _cc.corner_depth_frac(u_in, res, c, r) == \
                _ref_corner_depth_frac(u_in, res, c, r)
    for bad in ([np.nan, 3.0], [1.0, np.inf]):
        assert _cc.corner_depth_frac(uv, res, bad, 3.0) == (None, 0)
    assert _cc.corner_depth_frac(uv[:0], res[:0], [5.0, 5.0], 3.0) == (None, 0)
    assert _cc.corner_depth_frac(uv, res, [5.0, 5.0], float("nan")) == \
        _ref_corner_depth_frac(uv, res, [5.0, 5.0], float("nan"))


def test_ambiguous_stations_vectorised_matches_loop():
    rng = np.random.default_rng(7)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (240, 320)), (0, 0), 1.2)
    img[80:160, 100:220] += 60.0
    img[118:122, 100:220] -= 40.0
    for _ in range(60):
        p0 = rng.uniform(-20, 340, 2)
        p1 = rng.uniform(-20, 340, 2)
        half = int(rng.integers(2, 16))
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        u = d / length
        n = np.array([-u[1], u[0]])
        margin = max(float(half), 0.05 * length)
        ts = np.arange(margin, length - margin + 1e-9, 6.0)
        across = np.arange(-half, half + 1, dtype=np.float64)
        for mg in (0.0, 4.0, 25.0):
            args = (img, p0, u, n, ts, across, mg, 0.6, 3)
            assert _cc._ambiguous_stations(*args) == _cc._ambiguous_stations_loop(*args)
        assert _cc.edge_ambiguous_frac(img, p0, p1, half) == \
            _cc._ambiguous_stations_loop(img, p0, u, n, ts, across, 4.0,
                                         _cc.AMBIG_RUNNER_RATIO,
                                         _cc.AMBIG_RUNNER_MIN_SEP_PX)


# ---------------------------------------------------------------------------
# seam estimator / depth_top


def test_gap_slice_support_matches_reference():
    rng = np.random.default_rng(11)
    for trial in range(400):
        n = int(rng.integers(10, 120))
        v = rng.uniform(80, 200, n)
        c = int(rng.integers(0, n))
        v[max(0, c - 1):c + 2] -= rng.uniform(0, 80)
        if trial % 7 == 0:
            v = np.round(v)  # ties in maxima
        if trial % 11 == 0:
            v[: n // 3] = 0.0  # zeros (sign-bit-free) on the fast path
        prior = int(rng.integers(0, n))
        args = (v, prior, int(rng.integers(1, 20)), float(rng.uniform(0, 20)),
                float(rng.uniform(0, 10)), int(rng.integers(1, 8)))
        assert _seam._gap_slice_support(*args) == _ref_gap_slice_support(*args)
    neg = rng.uniform(-50, 50, 60)  # negative values -> numpy path
    assert _seam._gap_slice_support(neg, 30, 10, 1.0, 0.5, 6) == \
        _ref_gap_slice_support(neg, 30, 10, 1.0, 0.5, 6)


def test_workspace_mask_and_unique_rows_match_reference():
    from box_seam_detection.core.config import default_config
    dcfg = default_config()["depth_top"]
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-2.0, 2.5, size=(200003, 3))
    xyz[::1001] = np.nan
    xyz[5::2003, 1] = np.inf
    assert _same(_depth.workspace_mask(xyz, dcfg), _ref_workspace_mask(xyz, dcfg))
    small = xyz[:100]
    assert _same(_depth.workspace_mask(small, dcfg), _ref_workspace_mask(small, dcfg))
    for cells in (rng.integers(0, 6, size=(20000, 3)),
                  rng.integers(0, 90, size=(5000, 3)),
                  rng.integers(-3, 4, size=(3000, 3))):
        cells = cells.astype(np.int64)
        _, inv_ref, cnt_ref = np.unique(cells, axis=0, return_inverse=True,
                                        return_counts=True)
        inv, cnt = _depth._unique_rows_inverse_counts(cells)
        assert _same(inv, inv_ref) and _same(cnt, cnt_ref)


@pytest.mark.parametrize("cid", REAL_IDS[:1])
def test_estimate_top_real_matches_reference_helpers(cid, monkeypatch):
    """estimate_top with the fast helpers == with the reference helpers."""
    from box_seam_detection.core.config import default_config
    paths = _real(cid)
    cam, _ = _depth.load_capture_camera(str(paths["manifest"]))
    xyz = np.asarray(_ref_read_pcd_xyz(paths["pcd"])[0], dtype=np.float64)
    dcfg = default_config()["depth_top"]
    fast = _depth.estimate_top(xyz, cam, (1080, 1920), dcfg)

    def _ref_unique(cell):
        _, inv, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        return inv, counts

    monkeypatch.setattr(_depth, "workspace_mask", _ref_workspace_mask)
    monkeypatch.setattr(_depth, "_unique_rows_inverse_counts", _ref_unique)
    ref = _depth.estimate_top(xyz, cam, (1080, 1920), dcfg)
    assert fast["ok"] == ref["ok"] and fast["depth"] == ref["depth"]
    assert _same(fast["quad"], ref["quad"]) and _same(fast["hull"], ref["hull"])


# ---------------------------------------------------------------------------
# live.query_depth


@pytest.fixture(scope="module")
def query():
    import importlib
    return importlib.import_module("box_seam_detection.live.query_depth")


def _structured(xyz):
    arr = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return arr


D8 = (0.08, -0.05, 0.001, -0.0005, 0.01, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("distortion", [None, D8])
def test_query_projection_and_polygon_match_reference(query, distortion):
    rng = np.random.default_rng(4)
    xyz = np.c_[rng.uniform(-1.5, 1.5, 150001), rng.uniform(-1, 1, 150001),
                rng.uniform(-0.2, 2.0, 150001)].astype(np.float32)
    xyz[::333] = np.nan
    pf = _structured(xyz)
    args = (900.0, 901.0, 640.5, 360.25, 1280, 720)
    ref = _ref_project_valid_cloud(query, pf, *args, distortion=distortion)
    got = query._project_valid_cloud(pf, *args, distortion=distortion)
    for a, b in zip(got, ref):
        assert _same(a, b)
    poly = np.array([[500.0, 250.0], [800.0, 260.0], [790.0, 500.0], [510.0, 490.0]])
    sref = _ref_inside_polygon(ref[1], ref[2], poly, 16.0)
    fused = query._surface_candidates(pf, poly, 16.0, *args, distortion=distortion)
    for a, b in zip(fused, (ref[0][sref], ref[1][sref], ref[2][sref])):
        assert _same(a, b)
    assert _same(query._inside_eroded_convex_polygon(ref[1], ref[2], poly, 16.0), sref)


def test_query_error_order_preserved(query):
    pf = _structured(np.array([[0.0, 0.0, -1.0]] * 5, dtype=np.float32))
    good = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    degenerate = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    args = (900.0, 901.0, 640.5, 360.25, 1280, 720)
    for poly in (good, degenerate):
        with pytest.raises(ValueError, match="no finite positive-depth"):
            query._surface_candidates(pf, poly, 1.0, *args, distortion=D8)
    pf2 = _structured(np.array([[0.0, 0.0, 1.0]] * 5, dtype=np.float32))
    with pytest.raises(ValueError, match="degenerate"):
        query._surface_candidates(pf2, degenerate, 1.0, *args)
    dup = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    with pytest.raises(ValueError, match="duplicate"):
        query._surface_candidates(pf2, dup, 1.0, *args)


@pytest.mark.parametrize("cid", REAL_IDS)
def test_query_recover_surface_points_real_matches_reference(query, cid):
    paths = _real(cid)
    manifest = json.loads(paths["manifest"].read_text())
    _, points_flat, _, _ = query.load_pcd(paths["pcd"])
    rec_dirs = sorted((CAPTURES / "pipeline" / cid).glob(f"*/{cid}_autolabel_v2_seam.json"))
    if not rec_dirs:
        pytest.skip("no stored seam json")
    seam = query._load_seam_points_from_json(rec_dirs[-1])
    got, meta = query.recover_surface_points(points_flat, seam, manifest["camera_info"])

    # reference: same function with the pre-speed-up projection + polygon test
    orig_surface = query._surface_candidates

    def _ref_surface(pf, polygon, inset, fx, fy, cx, cy, w, h, distortion=None):
        xyz, u, v = _ref_project_valid_cloud(query, pf, fx, fy, cx, cy, w, h, distortion)
        m = _ref_inside_polygon(u, v, polygon, inset)
        return xyz[m], u[m], v[m]

    try:
        query._surface_candidates = _ref_surface
        ref, ref_meta = query.recover_surface_points(points_flat, seam, manifest["camera_info"])
    finally:
        query._surface_candidates = orig_surface
    assert json.dumps(got, sort_keys=True) == json.dumps(ref, sort_keys=True)
    assert json.dumps(meta, sort_keys=True) == json.dumps(ref_meta, sort_keys=True)
