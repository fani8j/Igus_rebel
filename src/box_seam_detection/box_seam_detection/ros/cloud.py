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
                   k_scaled_from: tuple[int, int] | None = None) -> dict[str, Any]:
    """Manifest with the recorder's ``rgbd_capture/v1`` schema (depth-mode flavour: pointcloud.source added),
    plus an optional ``driver_extrinsic`` block (same keys as RECORDED_DRIVER_EXTRINSIC)."""
    color_h = header_metadata(color_msg.header)
    depth_h = header_metadata(depth_msg.header)
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
    return manifest


def write_manifest(path: str, manifest: dict) -> str:
    def w(tmp):
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    return _atomic(path, w)
