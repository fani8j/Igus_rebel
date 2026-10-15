"""Virtual safety scene for the wrist-camera calibration (pure geometry, no ROS).

Four collision boxes are added to MoveIt's planning scene (in ``base_link``), all as VIRTUAL margins around the real
things, so every automatic pose / path that is collision-checked by MoveIt keeps clear of them:

    calib_table_slab   HORIZONTAL box (the real table is horizontal even when the board stands tilted on a holder).
                       top = (highest of the 4 board corners) + 50 mm, footprint ~1.2 x 1.2 m centred on the board centre in
                       x / y, 0.10 m thick.  It must not come within 50 mm (horizontally) of the robot base footprint while
                       it is vertically close to the base: the footprint is clipped, and the request is refused if the board
                       would no longer fit.
    calib_board_stand  box around the board's 4 corners (+0.05 m beyond them in the board plane, +0.05 m padding), from the
                       slab top up to (highest corner + 0.03 m) + 0.05 m padding: the raised lip / holder of the board.
    calib_pole_pad     the scene camera's ``camera_pole`` box inflated by 50 mm on every side
    calib_head_pad     the scene camera's ``camera_head`` box inflated by 50 mm on every side

The pole / head are read from the planning scene (move_group, published by camera_pole_collision.py); if either is missing
or not a single box the scene is REFUSED (fail closed).  Nothing here talks to ROS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .transforms import make_T, quat_to_R, R_to_quat

SLAB_ID = "calib_table_slab"
STAND_ID = "calib_board_stand"
POLE_PAD_ID = "calib_pole_pad"
HEAD_PAD_ID = "calib_head_pad"
OUR_IDS = (SLAB_ID, STAND_ID, POLE_PAD_ID, HEAD_PAD_ID)
POLE_ID, HEAD_ID = "camera_pole", "camera_head"
BASE_FRAMES = ("base_link", "world")          # identical in the single-arm stack (world_joint is the identity)
MARGIN_M = 0.05                               # virtual safety margin
SLAB_FOOTPRINT_M = 1.2
SLAB_THICKNESS_M = 0.10
STAND_PAD_M = 0.05                            # in-plane extension beyond the corners AND extra padding
STAND_LIP_M = 0.03
# robot base footprint: collision cylinder radius 0.09 m (z 0..0.1) + 30 mm for the real base flange, then 50 mm clearance
ROBOT_BASE_RADIUS_M = 0.12
ROBOT_BASE_BOTTOM_Z = 0.0
SLAB_MAX_TOP_Z = 0.30                         # a slab top above this is certainly wrong (it would be in the first links)
BOARD_MAX_TILT_DEG = 50.0                     # board normal vs base +z beyond this: the board pose is not believable
BOX = 1                                       # shape_msgs/SolidPrimitive.BOX


class SceneRefused(Exception):
    pass


@dataclass
class BoxSpec:
    id: str
    frame: str
    size: Tuple[float, float, float]
    T: np.ndarray                              # box centre pose in ``frame``

    @property
    def centre(self) -> np.ndarray:
        return self.T[:3, 3]

    def top_z(self) -> float:
        """Highest z of the box in the frame (box may be rotated)."""
        half = np.abs(self.T[2, :3]) @ (np.asarray(self.size) / 2.0)
        return float(self.T[2, 3] + half)

    def to_dict(self) -> dict:
        q = R_to_quat(self.T[:3, :3])
        return {"id": self.id, "frame": self.frame, "size": [float(v) for v in self.size],
                "position": self.T[:3, 3].tolist(), "quaternion_xyzw": [float(v) for v in q],
                "top_z": self.top_z()}


def inflate(spec: BoxSpec, new_id: str, margin: float = MARGIN_M) -> BoxSpec:
    return BoxSpec(new_id, spec.frame, tuple(float(s) + 2.0 * margin for s in spec.size), spec.T.copy())


def board_corners(B: np.ndarray, board_size: Tuple[float, float]) -> np.ndarray:
    w, h = board_size[0] / 2.0, board_size[1] / 2.0
    pts = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], dtype=float)
    return (B[:3, :3] @ pts.T).T + B[:3, 3]


def _yaw_T(centre: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), centre)


def build_slab(B: np.ndarray, board_size: Tuple[float, float], frame: str = "base_link",
               footprint: float = SLAB_FOOTPRINT_M, thickness: float = SLAB_THICKNESS_M) -> BoxSpec:
    """Horizontal table slab (see module docstring).  Raises SceneRefused."""
    B = np.asarray(B, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(B)):
        raise SceneRefused("the board pose is not finite")
    n_out = -B[:3, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n_out[2]) / max(np.linalg.norm(n_out), 1e-12)))))
    if tilt > BOARD_MAX_TILT_DEG:
        raise SceneRefused(f"the board normal is {tilt:.0f} deg off vertical (> {BOARD_MAX_TILT_DEG:g}): the board pose "
                           f"is not believable, refusing to build a table slab from it")
    corners = board_corners(B, board_size)
    top = float(corners[:, 2].max()) + MARGIN_M
    if top > SLAB_MAX_TOP_Z:
        raise SceneRefused(f"the slab top would be at z = {top:.3f} m in {frame} (> {SLAB_MAX_TOP_Z:g}): the board is "
                           f"higher than the robot's first links, refusing")
    cx, cy = float(B[0, 3]), float(B[1, 3])
    x0, x1, y0, y1 = cx - footprint / 2, cx + footprint / 2, cy - footprint / 2, cy + footprint / 2
    # the area that must stay covered: the board with its stand lip and padding
    nx0, nx1 = float(corners[:, 0].min()) - 2 * STAND_PAD_M, float(corners[:, 0].max()) + 2 * STAND_PAD_M
    ny0, ny1 = float(corners[:, 1].min()) - 2 * STAND_PAD_M, float(corners[:, 1].max()) + 2 * STAND_PAD_M
    keep = ROBOT_BASE_RADIUS_M + MARGIN_M
    # vertically close to the robot base (its collision body starts at z = 0): the slab must stay 50 mm away from the base
    # footprint horizontally.  Below the base bottom by >= 50 mm the footprint may overlap the base in x / y.
    if top > ROBOT_BASE_BOTTOM_Z - MARGIN_M:
        if x0 < keep and x1 > -keep and y0 < keep and y1 > -keep:
            options = []
            for name, (a0, a1, b0, b1) in {"x>=keep": (max(x0, keep), x1, y0, y1), "x<=-keep": (x0, min(x1, -keep), y0, y1),
                                           "y>=keep": (x0, x1, max(y0, keep), y1), "y<=-keep": (x0, x1, y0, min(y1, -keep))}.items():
                if a1 - a0 > 0 and b1 - b0 > 0 and a0 <= nx0 and a1 >= nx1 and b0 <= ny0 and b1 >= ny1:
                    options.append(((a1 - a0) * (b1 - b0), (a0, a1, b0, b1)))
            if not options:
                raise SceneRefused(f"the table slab (top z = {top:.3f} m) would come within {MARGIN_M * 1e3:.0f} mm of the "
                                   f"robot base and cannot be clipped without cutting the board area: refusing")
            x0, x1, y0, y1 = max(options)[1]
    size = (x1 - x0, y1 - y0, thickness)
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, top - thickness / 2.0])
    return BoxSpec(SLAB_ID, frame, size, make_T(np.eye(3), centre))


def build_stand(B: np.ndarray, board_size: Tuple[float, float], slab: BoxSpec) -> BoxSpec:
    """Box around the board's corners from the slab top up to (highest corner + 30 mm) + 50 mm padding, oriented with the
    board's in-plane x axis (yaw)."""
    corners = board_corners(B, board_size)
    xa = B[:2, 0]
    yaw = math.atan2(float(xa[1]), float(xa[0])) if np.linalg.norm(xa) > 1e-6 else 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, s], [-s, c]])                  # world xy -> yaw frame
    uv = (Rz @ corners[:, :2].T).T
    pad = STAND_PAD_M + STAND_PAD_M                   # beyond the corners + padding
    lo, hi = uv.min(axis=0) - pad, uv.max(axis=0) + pad
    centre_uv = (lo + hi) / 2.0
    centre_xy = Rz.T @ centre_uv
    z_bottom = slab.top_z()
    z_top = float(corners[:, 2].max()) + STAND_LIP_M + STAND_PAD_M
    if z_top <= z_bottom:       # keep a thin but valid box (the lip is below the slab top for a low holder)
        z_top = z_bottom + 0.01
    size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(z_top - z_bottom))
    return BoxSpec(STAND_ID, slab.frame, size, _yaw_T(np.array([centre_xy[0], centre_xy[1], (z_bottom + z_top) / 2.0]), yaw))


def box_from_object(obj, expect_frames: Sequence[str] = BASE_FRAMES) -> BoxSpec:
    """BoxSpec of a moveit_msgs/CollisionObject-like (attributes: id, header.frame_id, pose, primitives, primitive_poses)
    consisting of exactly one box in a base-equivalent frame.  Raises SceneRefused otherwise."""
    oid = str(obj.id)
    frame = str(obj.header.frame_id)
    if frame not in expect_frames:
        raise SceneRefused(f"'{oid}' is given in frame '{frame}', expected one of {list(expect_frames)}")
    prims = list(getattr(obj, "primitives", []) or [])
    if len(prims) != 1 or int(prims[0].type) != BOX or len(getattr(obj, "meshes", []) or []) or \
            len(getattr(obj, "planes", []) or []):
        raise SceneRefused(f"'{oid}' is not a single box primitive")
    dims = [float(v) for v in prims[0].dimensions]
    if len(dims) != 3 or not all(math.isfinite(v) and v > 0 for v in dims):
        raise SceneRefused(f"'{oid}' has invalid box dimensions {dims}")
    return BoxSpec(oid, frame, tuple(dims), pose_msg_T(obj.pose) @ pose_msg_T(obj.primitive_poses[0]))


def pose_msg_T(p) -> np.ndarray:
    q = p.orientation
    n = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2)
    if n < 1e-9:       # an unset pose in a CollisionObject.pose is the identity
        R = np.eye(3)
    else:
        R = quat_to_R(q.x / n, q.y / n, q.z / n, q.w / n)
    return make_T(R, [p.position.x, p.position.y, p.position.z])


def build_scene(B: np.ndarray, board_size: Tuple[float, float], pole: Optional[BoxSpec], head: Optional[BoxSpec],
                frame: str = "base_link") -> List[BoxSpec]:
    """The four safety boxes.  Fail closed: missing pole / head -> SceneRefused."""
    missing = [n for n, v in ((POLE_ID, pole), (HEAD_ID, head)) if v is None]
    if missing:
        raise SceneRefused(f"{' and '.join(missing)} not found in the planning scene (is camera_pole_collision running with "
                           f"the scene camera TF?): refusing to build the safety scene")
    slab = build_slab(B, board_size, frame)
    stand = build_stand(B, board_size, slab)
    return [slab, stand, inflate(pole, POLE_PAD_ID), inflate(head, HEAD_PAD_ID)]


def boxes_intersect_disc(spec: BoxSpec, radius: float, z_lo: float, z_hi: float) -> bool:
    """Conservative: does the (axis-aligned, un-rotated) box overlap the vertical cylinder x^2 + y^2 <= r^2 within
    [z_lo, z_hi]?  Used by the tests / sanity check on the slab."""
    c, h = spec.T[:3, 3], np.abs(spec.T[:3, :3]) @ (np.asarray(spec.size) / 2.0)
    if c[2] + h[2] < z_lo or c[2] - h[2] > z_hi:
        return False
    dx = max(abs(0.0 - c[0]) - h[0], 0.0)
    dy = max(abs(0.0 - c[1]) - h[1], 0.0)
    return math.hypot(dx, dy) < radius
