"""Pure geometry and depth helpers for the live seam detector."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def depth_scale_for_encoding(encoding: str) -> float:
    """Return the conversion from an image depth unit to metres."""
    normalized = encoding.upper()
    if normalized in {"16UC1", "MONO16"}:
        return 0.001
    if normalized == "32FC1":
        return 1.0
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def decode_compressed_depth(data: bytes, format_string: str) -> Tuple[np.ndarray, str]:
    """Decode a ROS compressedDepth PNG while preserving its integer depth units."""
    encoding = format_string.split(";", 1)[0].strip()
    if encoding.upper() not in {"16UC1", "MONO16"}:
        raise ValueError(
            f"Unsupported compressed depth encoding: {encoding or format_string}"
        )
    png_signature = b"\x89PNG\r\n\x1a\n"
    png_offset = data.find(png_signature)
    if png_offset < 0:
        raise ValueError("compressedDepth payload does not contain a PNG image")
    decoded = cv2.imdecode(
        np.frombuffer(data[png_offset:], dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if decoded is None or decoded.ndim != 2 or decoded.dtype != np.uint16:
        raise ValueError("compressedDepth payload is not a 16-bit single-channel PNG")
    return decoded, encoding

def normalized_camera_rays(
    width: int,
    height: int,
    camera_matrix: Sequence[float],
    distortion: Iterable[float],
    stride: int,
) -> np.ndarray:
    """Precompute normalized optical rays for a strided registered image grid."""
    if stride < 1:
        raise ValueError("point-cloud stride must be positive")
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    xs = np.arange(0, width, stride, dtype=np.float32)
    ys = np.arange(0, height, stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    coefficients = np.asarray(tuple(distortion), dtype=np.float64)
    rays = cv2.undistortPoints(pixels, matrix, coefficients)
    return rays.reshape(grid_y.shape + (2,)).astype(np.float32, copy=False)


def build_xyzrgb_points(
    depth: np.ndarray,
    bgr: np.ndarray,
    normalized_rays: np.ndarray,
    *,
    scale: float,
    stride: int,
) -> np.ndarray:
    """Build an organized, strided XYZ/RGB array for PointCloud2 publication."""
    sampled_depth = np.asarray(depth[::stride, ::stride], dtype=np.float32) * scale
    sampled_bgr = bgr[::stride, ::stride]
    if sampled_depth.shape != normalized_rays.shape[:2]:
        raise ValueError("normalized point-cloud rays do not match sampled depth")
    if sampled_bgr.shape[:2] != sampled_depth.shape:
        raise ValueError("registered color and depth dimensions differ")

    points = np.empty(
        sampled_depth.shape,
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
        ),
    )
    points["z"] = sampled_depth
    points["x"] = normalized_rays[..., 0] * sampled_depth
    points["y"] = normalized_rays[..., 1] * sampled_depth
    invalid = ~np.isfinite(sampled_depth) | (sampled_depth <= 0.0)
    points["x"][invalid] = np.nan
    points["y"][invalid] = np.nan
    points["z"][invalid] = np.nan

    blue = sampled_bgr[..., 0].astype(np.uint32)
    green = sampled_bgr[..., 1].astype(np.uint32)
    red = sampled_bgr[..., 2].astype(np.uint32)
    packed_rgb = (red << 16) | (green << 8) | blue
    points["rgb"] = packed_rgb.view(np.float32)
    return points


def get_median_depth(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    radius: int = 3,
    scale: float = 1.0,
    min_valid_samples: int = 5,
) -> Optional[float]:
    """Return local median depth in metres, clipping the patch at image bounds."""
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got shape {depth.shape}")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if min_valid_samples < 1:
        raise ValueError("min_valid_samples must be positive")

    height, width = depth.shape
    center_u = int(round(float(u)))
    center_v = int(round(float(v)))
    if not (0 <= center_u < width and 0 <= center_v < height):
        return None

    x0 = max(0, center_u - radius)
    x1 = min(width, center_u + radius + 1)
    y0 = max(0, center_v - radius)
    y1 = min(height, center_v + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64) * scale
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size < min_valid_samples:
        return None
    return float(np.median(valid))


def inset_segment(
    start_uv: Sequence[float], end_uv: Sequence[float], ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Inset both endpoints by ``ratio`` of the segment length."""
    if not 0.0 <= ratio < 0.5:
        raise ValueError("inset ratio must be in [0.0, 0.5)")
    start = np.asarray(start_uv, dtype=np.float64)
    end = np.asarray(end_uv, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("segment endpoints must each contain exactly two values")
    delta = end - start
    return start + ratio * delta, end - ratio * delta


def pixel_to_camera_xyz(
    u: float,
    v: float,
    depth_m: float,
    camera_matrix: Sequence[float],
    distortion: Iterable[float] = (),
) -> np.ndarray:
    """Back-project an image pixel to the camera optical frame.

    CameraInfo ``k`` and ``d`` are accepted directly. Distorted image coordinates
    are normalized with OpenCV before multiplication by the measured depth.
    """
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be finite and positive")
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    coefficients = np.asarray(tuple(distortion), dtype=np.float64)
    if coefficients.size and np.any(np.abs(coefficients) > 0.0):
        pixel = np.array([[[float(u), float(v)]]], dtype=np.float64)
        normalized = cv2.undistortPoints(pixel, matrix, coefficients)
        x_normalized, y_normalized = normalized[0, 0]
    else:
        x_normalized = (float(u) - matrix[0, 2]) / matrix[0, 0]
        y_normalized = (float(v) - matrix[1, 2]) / matrix[1, 1]

    return np.array(
        [x_normalized * depth_m, y_normalized * depth_m, depth_m],
        dtype=np.float64,
    )
