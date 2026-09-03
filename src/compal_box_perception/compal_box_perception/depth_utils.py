"""Pure geometry and depth helpers for the live seam detector."""

from __future__ import annotations
import math

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


def fit_top_plane_quad(
    depth: np.ndarray,
    camera_matrix: Sequence[float],
    distortion: Iterable[float],
    *,
    scale: float,
    fit_roi: Sequence[int],
    classify_roi: Sequence[int],
    sample_stride: int = 2,
    distance_threshold_m: float = 0.008,
    iterations: int = 100,
    min_inliers: int = 80,
    min_inlier_ratio: float = 0.25,
    expected_normal: Optional[Sequence[float]] = None,
    max_normal_deviation_deg: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fit the dominant upper plane and return its image quadrilateral.

    The plane is estimated only in ``fit_roi`` but classified throughout
    ``classify_roi``. This keeps the vertical carton face out of plane fitting
    while retaining the complete top boundary for quadrilateral extraction.
    """
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if sample_stride < 1:
        raise ValueError("plane sample stride must be positive")
    if distance_threshold_m <= 0.0:
        raise ValueError("plane distance threshold must be positive")

    height, width = depth.shape

    def clip_roi(roi: Sequence[int]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = (int(value) for value in roi)
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid plane ROI: {tuple(roi)}")
        return x0, y0, x1, y1

    fit_bounds = clip_roi(fit_roi)
    classify_bounds = clip_roi(classify_roi)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    coefficients = np.asarray(tuple(distortion), dtype=np.float64)
    expected = None
    minimum_alignment = -1.0
    if expected_normal is not None:
        expected = np.asarray(expected_normal, dtype=np.float64)
        expected /= np.linalg.norm(expected)
        minimum_alignment = math.cos(math.radians(max_normal_deviation_deg))

    def sample_points(bounds):
        x0, y0, x1, y1 = bounds
        xs = np.arange(x0, x1, sample_stride, dtype=np.int32)
        ys = np.arange(y0, y1, sample_stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sampled_depth = depth[grid_y, grid_x].astype(np.float64) * scale
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        pixels = np.stack((grid_x[valid], grid_y[valid]), axis=-1).astype(
            np.float64
        )
        if pixels.size == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                valid,
                grid_x,
                grid_y,
            )
        rays = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), matrix, coefficients
        ).reshape(-1, 2)
        z = sampled_depth[valid]
        points = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
        return points, valid, grid_x, grid_y

    fit_points, _, _, _ = sample_points(fit_bounds)
    if fit_points.shape[0] < min_inliers:
        raise ValueError(
            f"only {fit_points.shape[0]} valid points available for plane fitting"
        )

    generator = np.random.default_rng(0)
    best_mask = None
    best_count = 0
    best_error = math.inf
    for _ in range(iterations):
        indices = generator.choice(fit_points.shape[0], size=3, replace=False)
        first, second, third = fit_points[indices]
        normal = np.cross(second - first, third - first)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if expected is not None:
            alignment = float(np.dot(normal, expected))
            if abs(alignment) < minimum_alignment:
                continue
            if alignment < 0.0:
                normal = -normal
        offset = -float(np.dot(normal, first))
        residuals = np.abs(fit_points @ normal + offset)
        inliers = residuals <= distance_threshold_m
        count = int(np.count_nonzero(inliers))
        error = float(np.mean(residuals[inliers])) if count else math.inf
        if count > best_count or (count == best_count and error < best_error):
            best_mask = inliers
            best_count = count
            best_error = error

    if best_mask is None or best_count < min_inliers:
        raise ValueError(f"plane RANSAC found only {best_count} inliers")
    fit_ratio = best_count / fit_points.shape[0]
    if fit_ratio < min_inlier_ratio:
        raise ValueError(
            f"plane inlier ratio {fit_ratio:.3f} is below {min_inlier_ratio:.3f}"
        )

    inlier_points = fit_points[best_mask]
    centroid = np.mean(inlier_points, axis=0)
    _, _, axes = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = axes[-1]
    normal /= np.linalg.norm(normal)
    if expected is not None and np.dot(normal, expected) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))

    classify_points, classify_valid, grid_x, grid_y = sample_points(classify_bounds)
    residuals = np.abs(classify_points @ normal + offset)
    classify_inliers = residuals <= distance_threshold_m
    mask = np.zeros(classify_valid.shape, dtype=np.uint8)
    mask[classify_valid] = classify_inliers.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("top plane has no connected image contour")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quadrilateral = None
    for epsilon_ratio in np.arange(0.01, 0.12, 0.005):
        approximation = cv2.approxPolyDP(
            hull, epsilon_ratio * perimeter, True
        )
        if len(approximation) == 4:
            quadrilateral = approximation.reshape(-1, 2).astype(np.float64)
            break
    if quadrilateral is None:
        raise ValueError("top-plane contour does not simplify to four corners")

    classify_x0, classify_y0, _, _ = classify_bounds
    quadrilateral[:, 0] = classify_x0 + quadrilateral[:, 0] * sample_stride
    quadrilateral[:, 1] = classify_y0 + quadrilateral[:, 1] * sample_stride
    rms = float(np.sqrt(np.mean(residuals[classify_inliers] ** 2)))
    classify_ratio = float(
        np.count_nonzero(classify_inliers) / max(classify_points.shape[0], 1)
    )
    return quadrilateral, normal, offset, classify_ratio, rms


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


def depth_from_plane_at_pixel(
    u: float,
    v: float,
    camera_matrix: Sequence[float],
    distortion: Iterable[float],
    plane_normal: Sequence[float],
    plane_offset: float,
) -> float:
    """Intersect an image ray with a camera-frame plane and return optical Z."""
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    pixel = np.array([[[float(u), float(v)]]], dtype=np.float64)
    coefficients = np.asarray(tuple(distortion), dtype=np.float64)
    normalized = cv2.undistortPoints(pixel, matrix, coefficients)[0, 0]
    ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-9:
        raise ValueError("pixel ray is parallel to fitted plane")
    depth = -float(plane_offset) / denominator
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("pixel-plane intersection is behind the camera")
    return depth


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
