"""Pure (ROS-free) helpers: sensor_msgs/Image decoding, aligned-depth -> cloud, PCD/PNG/manifest writers.

The cloud is built exactly like the Orbbec driver's colour-registered cloud (verified 2026-09-28, see HANDOFF.md:
same points, 0.000 mm difference) and like the prototype ``backproject_depth`` of the recorder:

* valid pixel = depth > 0 (integer encodings) / finite and > 0 (float encodings); points are emitted in
  row-major (``np.nonzero``) order, so the cloud is UNORGANIZED (HEIGHT 1, no NaN rows, ``is_dense``);
* z = depth * scale (float32; 16UC1 millimetres -> 0.001, 32FC1 metres -> 1.0);
  x = (u - cx) * z / fx, y = (v - cy) * z / fy (float32, pinhole, no undistortion);
* rgb = (r << 16 | g << 8 | b) reinterpreted as float32 (alpha byte 0);
* units: metres, camera optical frame (x right, y down, z forward);
* PCD v0.7 binary, ``FIELDS x y z rgb`` (16 bytes/point), same header as the recorder's ``_write_pcd``.

``register_depth_to_color`` is the node-side replacement for the driver's software alignment: it re-projects a raw
depth image (own optical frame, undistorted pinhole) into the raw colour pixel grid (plumb_bob distortion), so the
result is interchangeable with the driver's aligned depth and goes through the same ``backproject_depth``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import numpy as np

DEPTH_ENCODING_SCALES = {"16UC1": 0.001, "32FC1": 1.0}
COLOR_ENCODINGS = ("rgb8", "bgr8")
SCHEMA = "rgbd_capture/v1"
FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32


class CaptureError(RuntimeError):
    """A frame set cannot be turned into a trustworthy capture (nothing is written)."""


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def header_metadata(header) -> dict:
    return {"frame_id": header.frame_id, "stamp_ns": stamp_ns(header)}


def image_array(msg, expected_channels: int | None = None) -> np.ndarray:
    """Decode a sensor_msgs/Image (rgb8, bgr8, 16UC1, 32FC1, mono8) to a numpy array (copy-free view)."""
    enc = msg.encoding
    dtype, ch = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1),
        "16UC1": (np.dtype("<u2"), 1), "mono16": (np.dtype("<u2"), 1), "32FC1": (np.dtype("<f4"), 1),
    }.get(enc, (None, None))
    if dtype is None:
        raise CaptureError(f"unsupported image encoding {enc!r}")
    itemsize = np.dtype(dtype).itemsize
    if msg.is_bigendian and itemsize > 1:
        raise CaptureError("big-endian images are not supported")
    row = msg.width * ch * itemsize
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step < row or buf.size < msg.step * msg.height:
        raise CaptureError(f"truncated image {msg.width}x{msg.height} {enc} step={msg.step} bytes={buf.size}")
    rows = buf[: msg.step * msg.height].reshape(msg.height, msg.step)[:, :row]
    arr = np.ascontiguousarray(rows).view(dtype)
    return arr.reshape(msg.height, msg.width, ch) if ch > 1 else arr.reshape(msg.height, msg.width)


def rgb_from_color_msg(msg) -> np.ndarray:
    """HxWx3 uint8 in RGB order from an rgb8/bgr8 image message."""
    if msg.encoding not in COLOR_ENCODINGS:
        raise CaptureError(f"colour image encoding must be rgb8/bgr8, got {msg.encoding!r}")
    arr = image_array(msg)
    return arr if msg.encoding == "rgb8" else np.ascontiguousarray(arr[:, :, ::-1])


def backproject_depth(depth: np.ndarray, rgb: np.ndarray, k, depth_scale: float) -> np.ndarray:
    """(n,4) float32 [x, y, z, rgb-as-float32] from an aligned depth image and RGB (HxWx3 uint8) image."""
    fx, cx, fy, cy = float(k[0]), float(k[2]), float(k[4]), float(k[5])
    if depth.shape != rgb.shape[:2]:
        raise CaptureError(f"depth {depth.shape} and colour {rgb.shape[:2]} sizes differ")
    if np.issubdtype(depth.dtype, np.integer):
        valid = depth > 0
    else:
        valid = np.isfinite(depth) & (depth > 0)
    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float32) * np.float32(depth_scale)
    x = (us.astype(np.float32) - np.float32(cx)) * z / np.float32(fx)
    y = (vs.astype(np.float32) - np.float32(cy)) * z / np.float32(fy)
    c = rgb[vs, us]
    packed = (c[:, 0].astype(np.uint32) << 16) | (c[:, 1].astype(np.uint32) << 8) | c[:, 2].astype(np.uint32)
    out = np.empty((vs.shape[0], 4), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = x, y, z
    out[:, 3] = packed.view(np.float32)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Node-side depth -> colour registration

#: a depth pixel whose projected footprint is wider than this many colour pixels is dropped (grazing / fold-over)
REGISTRATION_MAX_SPAN = 6
#: each footprint is grown by this many colour pixels on every side: neighbouring depth pixels at slightly different
#: depth project their shared corner to slightly different places (parallax of the depth noise / a tilted surface),
#: which would otherwise leave 1-pixel cracks. The z-buffer keeps the nearer surface in the overlap.
FOOTPRINT_MARGIN_PX = 0.25
#: refuse a colour<-depth extrinsic with a translation longer than this (the Gemini baseline is ~14 mm)
EXTRINSIC_MAX_TRANSLATION_M = 0.5


def transform_matrix(translation, quaternion_xyzw) -> np.ndarray:
    """4x4 float64 from a translation (x, y, z) and a quaternion (x, y, z, w); the quaternion is normalised."""
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if t.size != 3 or q.size != 4 or not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
        raise CaptureError("malformed transform (need finite translation[3] and quaternion_xyzw[4])")
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > 0.01:
        raise CaptureError(f"quaternion is not normalised (norm {n:.4f})")
    x, y, z, w = q / n
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = t
    return m


def _intrinsics(k, name: str) -> tuple[float, float, float, float]:
    try:
        a = np.asarray(k, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{name}: malformed K ({exc})") from exc
    if a.size != 9 or not np.all(np.isfinite(a)):
        raise CaptureError(f"{name}: K must be 9 finite values, got {a.size}")
    fx, fy, cx, cy = a[0], a[4], a[2], a[5]
    if fx <= 0 or fy <= 0:
        raise CaptureError(f"{name}: invalid focal length fx={fx} fy={fy}")
    return float(fx), float(fy), float(cx), float(cy)


def colour_distortion(model, d) -> tuple[float, ...] | None:
    """(k1, k2, p1, p2, k3, k4, k5, k6) of a ROS CameraInfo plumb_bob (5 or 8 values; 8 = rational, the same
    convention as live.query_depth) or None for "no distortion" (empty model/d, or all-zero d). Fails closed on an
    unknown model or a malformed coefficient vector."""
    m = str(model or "plumb_bob")
    if m not in ("plumb_bob", "rational_polynomial"):
        raise CaptureError(f"unsupported colour distortion_model {model!r} (plumb_bob/rational_polynomial only)")
    try:
        v = np.asarray(list(d) if d is not None else [], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"malformed distortion coefficients ({exc})") from exc
    if v.size not in (0, 5, 8) or not np.all(np.isfinite(v)):
        raise CaptureError(f"colour distortion d must have 0, 5 or 8 finite values, got {v.size}")
    if v.size == 0 or not np.any(v):
        return None
    return tuple(float(x) for x in np.concatenate([v, np.zeros(8 - v.size)]))


def _distort(xn, yn, dist):
    k1, k2, p1, p2, k3, k4, k5, k6 = dist
    r2 = xn * xn + yn * yn
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    if k4 or k5 or k6:
        radial = radial / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xy = xn * yn
    xd = xn * radial + (2.0 * p1) * xy + p2 * (r2 + (2.0 * xn) * xn)
    yd = yn * radial + p1 * (r2 + (2.0 * yn) * yn) + (2.0 * p2) * xy
    return xd, yd


def register_depth_to_color(depth: np.ndarray, depth_scale: float, k_depth, d_depth, color_k, color_d,
                            color_model, t_color_depth, out_w: int, out_h: int,
                            stats: dict | None = None, margin_px: float = FOOTPRINT_MARGIN_PX) -> np.ndarray:
    """Re-project a depth image into the (raw, distorted) colour pixel grid.

    ``depth``: HxW raw depth in its own optical frame (undistorted pinhole ``k_depth``, ``d_depth`` must be all
    zero), integer (0 and the saturated maximum, e.g. 65535, = no data) or float (finite and > 0 = data) in units of ``depth_scale`` metres; depth is the
    optical-axis distance (z). ``t_color_depth``: 4x4 transform taking points from the depth optical frame to the
    colour optical frame (tf2 ``lookup_transform(colour, depth)``). Returns an ``out_h x out_w`` image with the
    SAME dtype/units as ``depth``: z of the nearest surface in the colour optical frame at every colour pixel it
    covers, 0 elsewhere. Pixels are placed with the colour plumb_bob distortion, i.e. in the raw colour image like
    the Orbbec driver's aligned depth.

    Method (librealsense ``align`` style): each valid depth pixel is a square footprint of its own depth, its four
    corners ``(u -/+ 0.5, v -/+ 0.5)`` are deprojected, transformed and projected (with distortion) into
    the colour image, and every colour pixel centre inside that box (half-open, grown by ``margin_px`` so
    neighbouring footprints overlap instead of leaving cracks) receives the z. Collisions resolve to the nearest z (z-buffer; equal z
    resolves deterministically to the depth pixel that comes last in raster order). Colour pixels not covered by any
    footprint (occlusion shadows, depth holes, outside the depth field of view) stay 0. ``stats`` (optional dict)
    receives input/output pixel counts and how many footprints were dropped (outside the colour image / degenerate /
    wider than REGISTRATION_MAX_SPAN colour pixels).

    Fails closed (CaptureError) on malformed input, a non-zero depth distortion or an unknown colour model."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.size == 0:
        raise CaptureError("depth must be a non-empty 2-D array")
    if not (np.issubdtype(depth.dtype, np.integer) or np.issubdtype(depth.dtype, np.floating)):
        raise CaptureError(f"unsupported depth dtype {depth.dtype}")
    try:
        scale = float(depth_scale)
        out_w, out_h = int(out_w), int(out_h)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"malformed registration input ({exc})") from exc
    if not (np.isfinite(scale) and scale > 0):
        raise CaptureError(f"invalid depth_scale {depth_scale!r}")
    if not (np.isfinite(margin_px) and 0.0 <= margin_px <= 2.0):
        raise CaptureError(f"invalid footprint margin {margin_px!r}")
    if out_w < 1 or out_h < 1:
        raise CaptureError(f"invalid output size {out_w}x{out_h}")
    fxd, fyd, cxd, cyd = _intrinsics(k_depth, "depth")
    fxc, fyc, cxc, cyc = _intrinsics(color_k, "colour")
    try:
        dd = np.asarray(list(d_depth) if d_depth is not None else [], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"malformed depth distortion ({exc})") from exc
    if not np.all(np.isfinite(dd)) or np.any(dd != 0.0):
        raise CaptureError("depth camera has non-zero distortion coefficients; the raw depth is not an undistorted "
                           "pinhole image - node-side registration refused")
    dist = colour_distortion(color_model, color_d)
    try:
        T = np.asarray(t_color_depth, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"malformed t_color_depth ({exc})") from exc
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise CaptureError("t_color_depth must be a finite 4x4 homogeneous transform")
    R, t = T[:3, :3], T[:3, 3]
    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1.0) < 1e-3):
        raise CaptureError("t_color_depth rotation is not a proper rotation")
    if float(np.linalg.norm(t)) > EXTRINSIC_MAX_TRANSLATION_M:
        raise CaptureError(f"t_color_depth translation {float(np.linalg.norm(t)):.3f} m is implausible for one camera")

    if np.issubdtype(depth.dtype, np.integer):
        # 0 = no data; the saturated maximum is the Orbbec SDK's "invalid" marker in some frames (65535 mm)
        valid = (depth > 0) & (depth < np.iinfo(depth.dtype).max)
    else:
        valid = np.isfinite(depth) & (depth > 0)
    vs, us = np.nonzero(valid)
    z_d = depth[vs, us].astype(np.float32) * np.float32(scale)
    n_in = int(vs.size)
    # far -> near; stable: equal z keep raster order (the last one wins a tie)
    order = np.argsort(-z_d, kind="stable")
    vs, us, z_d = vs[order].astype(np.float32), us[order].astype(np.float32), z_d[order]
    f32 = np.float32
    Rf, tf = R.astype(np.float32), t.astype(np.float32)

    def corner(du, dv):
        xn = (us + f32(du - cxd)) / f32(fxd)
        yn = (vs + f32(dv - cyd)) / f32(fyd)
        px = z_d * (Rf[0, 0] * xn + Rf[0, 1] * yn + Rf[0, 2]) + tf[0]
        py = z_d * (Rf[1, 0] * xn + Rf[1, 1] * yn + Rf[1, 2]) + tf[1]
        pz = z_d * (Rf[2, 0] * xn + Rf[2, 1] * yn + Rf[2, 2]) + tf[2]
        ok = pz > 1e-6
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            xn_c, yn_c = px / pz, py / pz
            if dist is not None:
                xn_c, yn_c = _distort(xn_c, yn_c, dist)
            x, y = f32(fxc) * xn_c + f32(cxc), f32(fyc) * yn_c + f32(cyc)
        return x, y, pz, ok & np.isfinite(x) & np.isfinite(y)

    # the four corners of the depth pixel: neighbouring pixels share corners, so the bounding boxes of neighbours
    # always touch (no 1-pixel cracks even where the warp is not exactly separable: rotation, distortion)
    cs = [corner(du, dv) for du, dv in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))]
    ok = cs[0][3] & cs[1][3] & cs[2][3] & cs[3][3]
    z_c = (cs[0][2] + cs[1][2] + cs[2][2] + cs[3][2]) * f32(0.25)   # = z of the pixel centre (z is affine in the ray)
    mg = f32(margin_px)
    with np.errstate(invalid="ignore"):
        x_lo = np.minimum(np.minimum(cs[0][0], cs[1][0]), np.minimum(cs[2][0], cs[3][0]))
        x_hi = np.maximum(np.maximum(cs[0][0], cs[1][0]), np.maximum(cs[2][0], cs[3][0]))
        y_lo = np.minimum(np.minimum(cs[0][1], cs[1][1]), np.minimum(cs[2][1], cs[3][1]))
        y_hi = np.maximum(np.maximum(cs[0][1], cs[1][1]), np.maximum(cs[2][1], cs[3][1]))
        # pixel centres at integers; half-open box [lo, hi): centres ceil(lo) .. ceil(hi)-1, at least one
        xl = np.ceil(x_lo - mg); xh = np.maximum(np.ceil(x_hi + mg) - f32(1), xl)
        yl = np.ceil(y_lo - mg); yh = np.maximum(np.ceil(y_hi + mg) - f32(1), yl)
    del cs
    n_degenerate = int(n_in - int(ok.sum()))        # behind the camera / non-finite
    inside = (xh >= 0) & (xl <= out_w - 1) & (yh >= 0) & (yl <= out_h - 1)
    n_outside = int((ok & ~inside).sum())           # footprint entirely outside the colour image
    ok &= inside
    wide = ((xh - xl) >= REGISTRATION_MAX_SPAN) | ((yh - yl) >= REGISTRATION_MAX_SPAN)
    n_wide = int((ok & wide).sum())
    ok &= ~wide
    # clip to the image: every remaining footprint has >= 1 pixel and all indices are in bounds
    xl = np.maximum(xl[ok], 0).astype(np.int32); xh = np.minimum(xh[ok], out_w - 1).astype(np.int32)
    yl = np.maximum(yl[ok], 0).astype(np.int32); yh = np.minimum(yh[ok], out_h - 1).astype(np.int32)
    z_c = z_c[ok]
    nx, ny = xh - xl + 1, yh - yl + 1
    base = yl * np.int32(out_w) + xl
    zbuf = np.full(out_w * out_h, np.inf, dtype=np.float32)
    for dy in range(int(ny.max()) if ny.size else 0):
        for dx in range(int(nx.max())):
            m = (nx > dx) & (ny > dy) if (dx or dy) else slice(None)
            idx = base[m] + np.int32(dy * out_w + dx)
            zz = z_c[m]
            win = zz < zbuf[idx]          # compared with the buffer as it was BEFORE this pass ...
            idx, zz = idx[win], zz[win]
            zbuf[idx] = zz                # ... sources are far -> near, so of repeated indices the nearest is last
    covered = np.isfinite(zbuf)
    if np.issubdtype(depth.dtype, np.integer):
        info = np.iinfo(depth.dtype)
        q = np.rint(zbuf[covered].astype(np.float64) / scale)
        q = np.clip(q, 1, info.max)       # a covered pixel never rounds to the "no data" value
        out = np.zeros(out_w * out_h, dtype=depth.dtype)
        out[covered] = q.astype(depth.dtype)
    else:
        out = np.zeros(out_w * out_h, dtype=depth.dtype)
        out[covered] = (zbuf[covered].astype(np.float64) / scale).astype(depth.dtype)
    out = out.reshape(out_h, out_w)
    if stats is not None:
        stats.update({"input_valid": n_in, "output_valid": int(covered.sum()), "outside_image": n_outside,
                      "dropped_degenerate": n_degenerate, "dropped_too_wide": n_wide})
    return out



def validate_registration(color_msg, depth_msg, info_msg) -> tuple[float, tuple[float, ...], int, int]:
    """Check that the depth image is aligned to the colour image. Returns (depth_scale, K, width, height).

    Aligned depth (depth_registration true, align SW to COLOR) has the colour resolution and frame. If the
    CameraInfo is for a different resolution of the same aspect ratio, K is scaled uniformly (recorded in the
    manifest); anything else is refused (an unregistered depth cannot be put in the colour frame)."""
    scale = DEPTH_ENCODING_SCALES.get(depth_msg.encoding)
    if scale is None:
        raise CaptureError(f"depth encoding must be 16UC1 (mm) or 32FC1 (m), got {depth_msg.encoding!r}")
    if color_msg.encoding not in COLOR_ENCODINGS:
        raise CaptureError(f"colour image encoding must be rgb8/bgr8, got {color_msg.encoding!r}")
    if info_msg is None:
        raise CaptureError("no CameraInfo received")
    dims = (depth_msg.width, depth_msg.height)
    if dims != (color_msg.width, color_msg.height):
        raise CaptureError(
            f"depth {dims[0]}x{dims[1]} != colour {color_msg.width}x{color_msg.height}: the depth image is "
            "not aligned to the colour image (driver needs depth_registration: true, align_target_stream: COLOR)")
    frames = {depth_msg.header.frame_id, color_msg.header.frame_id, info_msg.header.frame_id}
    if len(frames) != 1:
        raise CaptureError(
            f"frame_id mismatch depth={depth_msg.header.frame_id!r} colour={color_msg.header.frame_id!r} "
            f"camera_info={info_msg.header.frame_id!r}: depth must be registered to the colour frame")
    k = tuple(float(v) for v in info_msg.k)
    if k[0] <= 0 or k[4] <= 0:
        raise CaptureError(f"invalid CameraInfo K {k}")
    if (info_msg.width, info_msg.height) != dims:
        sx, sy = dims[0] / max(info_msg.width, 1), dims[1] / max(info_msg.height, 1)
        if info_msg.width <= 0 or abs(sx - sy) > 1e-3 * sx:
            raise CaptureError(
                f"CameraInfo {info_msg.width}x{info_msg.height} does not match the images {dims[0]}x{dims[1]} "
                "and is not a uniform scale of them")
        k = (k[0] * sx, k[1], k[2] * sx, k[3], k[4] * sy, k[5] * sy, k[6], k[7], k[8])
    return scale, k, dims[0], dims[1]


def _atomic(path: str, writer, suffix: str = "") -> str:
    tmp = f"{path}.tmp-{uuid.uuid4().hex}{suffix}"
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_png(path: str, rgb: np.ndarray) -> str:
    import cv2
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    def w(tmp):
        if not cv2.imwrite(tmp, bgr):
            raise CaptureError(f"cv2.imwrite failed for {path}")
    return _atomic(path, w, suffix=".png")


def write_pcd(path: str, points: np.ndarray) -> str:
    """Binary PCD v0.7, x y z rgb float32, HEIGHT 1 (same bytes layout as the recorder's writer)."""
    n = int(points.shape[0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
        f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    data = np.ascontiguousarray(points, dtype=np.float32)

    def w(tmp):
        with open(tmp, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(data)
    return _atomic(path, w)


def build_manifest(capture_id: str, image_path: str, pcd_path: str, color_msg, depth_msg, info_msg, k,
                   point_count: int, topics: dict, driver_extrinsic: dict | None, created_at_ns: int | None = None,
                   k_scaled_from: tuple[int, int] | None = None, depth_header: dict | None = None,
                   depth_registration: dict | None = None) -> dict[str, Any]:
    """Manifest with the recorder's ``rgbd_capture/v1`` schema (depth-mode flavour: pointcloud.source added),
    plus an optional ``driver_extrinsic`` block (same keys as RECORDED_DRIVER_EXTRINSIC) and an optional additive
    ``depth_registration`` block (how the depth got into the colour grid: driver or node).

    ``depth_header`` overrides the pointcloud header (a node-registered depth lives in the colour optical frame, not in
    the raw depth message's own frame); consumers require pointcloud/image/camera_info frames to be equal."""
    color_h = header_metadata(color_msg.header)
    depth_h = dict(depth_header) if depth_header is not None else header_metadata(depth_msg.header)
    info_h = header_metadata(info_msg.header)
    cam = {
        "width": color_msg.width, "height": color_msg.height,
        "distortion_model": info_msg.distortion_model, "d": list(info_msg.d),
        "k": list(k), "r": list(info_msg.r), "p": list(info_msg.p), "header": info_h,
    }
    if k_scaled_from is not None:
        cam["scaled_from"] = {"width": k_scaled_from[0], "height": k_scaled_from[1]}
        p = list(info_msg.p)
        sx = color_msg.width / k_scaled_from[0]
        sy = color_msg.height / k_scaled_from[1]
        p[0] *= sx; p[2] *= sx; p[3] *= sx; p[5] *= sy; p[6] *= sy; p[7] *= sy
        cam["p"] = p
    manifest = {
        "schema": SCHEMA,
        "capture_id": capture_id,
        "created_at_ns": int(created_at_ns if created_at_ns is not None else time.time_ns()),
        "image": {"path": image_path, "width": color_msg.width, "height": color_msg.height,
                  "encoding": color_msg.encoding, "header": color_h},
        "pointcloud": {
            "path": pcd_path, "width": point_count, "height": 1, "point_step": 16,
            "row_step": 16 * point_count, "is_dense": True, "header": depth_h,
            "fields": [{"name": nme, "offset": off, "datatype": FLOAT32, "count": 1}
                       for nme, off in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))],
            "point_count": point_count, "stored_height": 1,
            "source": {"type": "depth_image_backprojection", "depth_topic": topics["depth"],
                       "depth_encoding": depth_msg.encoding, "backproject": "numpy"},
        },
        "camera_info": cam,
        "sync_delta_ns": abs(color_h["stamp_ns"] - depth_h["stamp_ns"]),
        "topics": {"rgb": topics["rgb"], "pointcloud": topics["depth"], "depth": topics["depth"]},
    }
    if driver_extrinsic is not None:
        manifest["driver_extrinsic"] = driver_extrinsic
    if depth_registration is not None:
        manifest["depth_registration"] = depth_registration
    return manifest


def write_manifest(path: str, manifest: dict) -> str:
    def w(tmp):
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    return _atomic(path, w)
