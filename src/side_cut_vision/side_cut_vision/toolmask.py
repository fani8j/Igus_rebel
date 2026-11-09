"""Tool occlusion mask from geometry: the knife handpiece and holder hulls projected through the hand-eye.

The camera is rigid on link6, so the tool region in the image depends only on the hand-eye
(link6 -> optical) and the fixed link6 -> igus_tool0 transform (read from TF at start-up, with the URDF
constant as fallback). Hull vertices below are the convex hulls (vertices only) of the variant-C
collision STL in igus_rebel_description/meshes/knife_c/collision, all authored in the igus_tool0 frame
(identity joint origins):
  handpiece     = handpiece_housing + handpiece_collar + handpiece_seat  (the knife body in the image)
Only the handpiece is projected. The holder (flange, knife holder, cover plate) and the Gemini
bracket/cradle wrap around the camera itself, within ~70 mm of it: clipped at z_min their slices
project to huge wedges that cover the working area, and the Gemini 305 returns no depth there. They
are covered by the near-field strip below instead.

Near field: the holder wraps around the camera, so most of it lies at z < ~70 mm, where the Gemini 305
returns no depth and which is never the working area. Each part is clipped at z_min (default 70 mm)
before projection; the clipped body is still convex, so its silhouette is the hull of the projected
vertices plus the edge/plane intersection points. A margin (default 12 px) covers hand-eye error
(LOO 5 mm at 136 mm = about 25 px at worst; the S0 frames match the real knife body edge to ~5 px).
Verified against the real S0 frames of 2026-10-20 (handpiece left edge at x=945 vs 940 in the image).
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraModel
from .geom import inv_T

# link6 -> igus_tool0: +52.7 mm along z, rotated pi about z (knife_c_tool.macro.xacro igus_tool_frames)
T_LINK6_TOOL0 = np.array([[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0.0527], [0, 0, 0, 1.0]])

HULLS_TOOL0 = {
    "handpiece": [
        (-0.0189, -0.0790, 0.0828),
        (-0.0189, -0.1050, 0.0828),
        (0.0574, -0.1050, -0.0494),
        (0.0574, -0.0790, -0.0494),
        (0.0375, -0.1050, -0.0609),
        (-0.0388, -0.1050, 0.0713),
        (0.0375, -0.0790, -0.0609),
        (-0.0388, -0.0790, 0.0713),
        (-0.0402, -0.0943, 0.0797),
        (-0.0402, -0.0897, 0.0797),
        (-0.0255, -0.0897, 0.0882),
        (-0.0255, -0.0943, 0.0882),
        (-0.0408, -0.0890, 0.0912),
        (-0.0418, -0.0911, 0.0906),
        (-0.0418, -0.0935, 0.0906),
        (-0.0369, -0.0968, 0.0934),
        (-0.0408, -0.0956, 0.0912),
        (-0.0351, -0.0956, 0.0945),
        (-0.0341, -0.0935, 0.0951),
        (-0.0341, -0.0911, 0.0951),
        (-0.0351, -0.0890, 0.0945),
        (-0.0390, -0.0878, 0.0922),
        (-0.0390, -0.0968, 0.0922),
        (-0.0369, -0.0878, 0.0934),
    ],
}

DEFAULT_ZMIN_M = 0.070
DEFAULT_MARGIN_PX = 12
# The black holder block at the right image edge (x >= 1130 of 1280): near-field, no depth, always
# in view in the S0 frames (both arms). Colour measurements must not use it.
NEAR_FIELD_BLOCK_X_PX = 1130


def part_polygons(cam: CameraModel, T_link6_optical: np.ndarray, T_link6_tool0: Optional[np.ndarray] = None,
                  z_min: float = DEFAULT_ZMIN_M) -> List[np.ndarray]:
    """Projected convex silhouettes (pixels, (N,2) float32) of the tool parts, clipped at z_min."""
    T_l6_t0 = T_LINK6_TOOL0 if T_link6_tool0 is None else np.asarray(T_link6_tool0, dtype=float)
    T_o_t0 = inv_T(np.asarray(T_link6_optical, dtype=float)) @ T_l6_t0
    polys = []
    for name, verts in HULLS_TOOL0.items():
        P = (T_o_t0[:3, :3] @ np.asarray(verts, dtype=float).T).T + T_o_t0[:3, 3]
        n = len(P)
        ii, jj = np.triu_indices(n, 1)
        a, b = P[ii], P[jj]
        st = (a[:, 2] - z_min) * (b[:, 2] - z_min) < 0
        a, b = a[st], b[st]
        t = ((z_min - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]
        S = np.vstack([P[P[:, 2] >= z_min], a + t * (b - a)])
        if len(S) < 3:
            continue
        uv = cam.project(S, min_z=z_min * 0.5)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) < 3:
            continue
        uv = np.clip(uv, -3000, 4000).astype(np.float32)
        polys.append(cv2.convexHull(uv).reshape(-1, 2))
    return polys


def tool_mask_from_geometry(cam: CameraModel, T_link6_optical: np.ndarray,
                            T_link6_tool0: Optional[np.ndarray] = None, z_min: float = DEFAULT_ZMIN_M,
                            margin_px: int = DEFAULT_MARGIN_PX, near_field_block: bool = True) -> np.ndarray:
    """(H, W) bool mask of the tool region (True = tool), full image resolution."""
    m = np.zeros((cam.height, cam.width), np.uint8)
    for poly in part_polygons(cam, T_link6_optical, T_link6_tool0, z_min):
        cv2.fillPoly(m, [np.rint(poly).astype(np.int32)], 1)
    if near_field_block:
        x0 = int(round(NEAR_FIELD_BLOCK_X_PX * cam.width / 1280.0))
        m[:, x0:] = 1
    if margin_px > 0:
        k = 2 * int(margin_px) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(bool)
