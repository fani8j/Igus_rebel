"""Pinhole camera with the 8-coefficient rational distortion model of the Orbbec Gemini 305.

The driver publishes camera_info with distortion_model "plumb_bob" but carries 8 coefficients
(k1 k2 p1 p2 k3 k4 k5 k6). They are the OpenCV rational model: all 8 are passed to OpenCV.
Dropping the last three (a true plumb_bob) is wrong by tens of pixels near the image corners.

Pixel convention: OpenCV (pixel centres at integer coordinates). Depth is the aligned 16UC1 z-depth
image (mm by default), same size as the colour image.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class CameraError(ValueError):
    """Bad camera_info / image input."""


class CameraModel:
    def __init__(self, K, D, width: int, height: int, depth_scale_m: float = 0.001,
                 frame_id: str = ""):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            raise CameraError("non-finite camera intrinsics")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CameraError("non-positive focal length")
        if len(D) not in (0, 4, 5, 8):
            raise CameraError(f"unsupported distortion coefficient count {len(D)} (need 0/4/5/8)")
        if width <= 0 or height <= 0:
            raise CameraError("bad image size")
        self.K = K
        self.D = D
        self.width = int(width)
        self.height = int(height)
        self.depth_scale_m = float(depth_scale_m)
        self.frame_id = frame_id
        self._grids: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict, depth_scale_m: float = 0.001) -> "CameraModel":
        """From the camera_info.json layout {K,D,width,height,frame_id} or a ROS-like {k,d,...}."""
        K = d.get("K", d.get("k"))
        D = d.get("D", d.get("d"))
        if K is None or D is None:
            raise CameraError("camera_info needs K and D")
        w = d.get("width")
        h = d.get("height")
        if w is None or h is None:
            raise CameraError("camera_info needs width and height")
        return cls(K, D, int(w), int(h), depth_scale_m, d.get("frame_id", ""))

    @classmethod
    def from_json(cls, path: str, **kw) -> "CameraModel":
        with open(path) as f:
            return cls.from_dict(json.load(f), **kw)

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "D": self.D.tolist(), "width": self.width,
                "height": self.height, "frame_id": self.frame_id,
                "distortion_model": "rational_polynomial" if len(self.D) == 8 else "plumb_bob"}

    def with_distortion(self, D) -> "CameraModel":
        return CameraModel(self.K, D, self.width, self.height, self.depth_scale_m, self.frame_id)

    def plumb_bob_only(self) -> "CameraModel":
        """The same camera with only k1 k2 p1 p2 k3 (what a plumb_bob reader would use). For tests."""
        return self.with_distortion(self.D[:5])

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    # ------------------------------------------------------------------ rays
    def norm_grid(self, stride: int = 1) -> np.ndarray:
        """(H', W', 2) float32 normalised image coordinates (x/z, y/z) of pixels u = 0, s, 2s ... ."""
        stride = int(stride)
        g = self._grids.get(stride)
        if g is None:
            u = np.arange(0, self.width, stride, dtype=np.float64)
            v = np.arange(0, self.height, stride, dtype=np.float64)
            uu, vv = np.meshgrid(u, v)
            pts = np.stack([uu, vv], -1).reshape(-1, 1, 2)
            xy = cv2.undistortPoints(pts, self.K, self.D if len(self.D) else None)
            g = xy.reshape(len(v), len(u), 2).astype(np.float32)
            self._grids[stride] = g
        return g

    def pixel_rays(self, uv: np.ndarray) -> np.ndarray:
        """Normalised coordinates (N,2) of arbitrary pixel coordinates (N,2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if len(uv) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(uv, self.K, self.D if len(self.D) else None).reshape(-1, 2)

    # ------------------------------------------------------------------ depth -> 3D
    def check_depth(self, depth: np.ndarray) -> None:
        if depth.ndim != 2 or depth.shape != (self.height, self.width):
            raise CameraError(
                f"depth image shape {depth.shape} does not match camera {self.height}x{self.width}")

    def deproject(self, depth: np.ndarray, stride: int = 1, min_z: float = 0.0, max_z: float = 5.0):
        """Aligned depth -> camera-frame points at pixels u = 0, s, 2s ... .

        Returns (xyz (H',W',3) float32 metres, valid (H',W') bool). Invalid pixels (0, outside
        [min_z, max_z]) are NaN in xyz.
        """
        self.check_depth(depth)
        z = depth[::stride, ::stride].astype(np.float32) * np.float32(self.depth_scale_m)
        valid = np.isfinite(z) & (z > max(min_z, 1e-6)) & (z <= max_z)
        g = self.norm_grid(stride)
        zz = np.where(valid, z, np.nan).astype(np.float32)
        xyz = np.stack([g[..., 0] * zz, g[..., 1] * zz, zz], axis=-1)
        return xyz, valid

    def sample_depth(self, depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Nearest-pixel z-depth (metres) at pixel coords (N,2); NaN for zero depth or outside."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        ui = np.rint(uv[:, 0]).astype(np.int64)
        vi = np.rint(uv[:, 1]).astype(np.int64)
        inside = (ui >= 0) & (ui < self.width) & (vi >= 0) & (vi < self.height) & np.all(np.isfinite(uv), 1)
        out = np.full(len(uv), np.nan)
        z = depth[np.clip(vi, 0, self.height - 1), np.clip(ui, 0, self.width - 1)].astype(np.float64)
        z = z * self.depth_scale_m
        ok = inside & (z > 1e-6)
        out[ok] = z[ok]
        return out

    def points_from_pixels(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Camera-frame points for pixel coords (N,2) and z-depth (N,)."""
        xy = self.pixel_rays(uv)
        return np.stack([xy[:, 0] * z, xy[:, 1] * z, z], axis=1)

    # ------------------------------------------------------------------ 3D -> pixels
    def project(self, P_cam: np.ndarray, min_z: float = 0.01) -> np.ndarray:
        """Camera-frame points (N,3) -> pixel coords (N,2); NaN for points with z < min_z."""
        P = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
        out = np.full((len(P), 2), np.nan)
        front = np.isfinite(P).all(axis=1) & (P[:, 2] >= min_z)
        if front.any():
            pts, _ = cv2.projectPoints(P[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K,
                                       self.D if len(self.D) else None)
            out[front] = pts.reshape(-1, 2)
        return out

    def in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = np.asarray(uv)
        return (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= margin) & (uv[:, 0] <= self.width - 1 - margin)
                & (uv[:, 1] >= margin) & (uv[:, 1] <= self.height - 1 - margin))
