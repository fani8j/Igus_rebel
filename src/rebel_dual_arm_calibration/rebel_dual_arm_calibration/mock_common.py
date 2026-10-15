"""MOCK ONLY - shared code of the simulated scene camera and the mock pose driver.

Nothing here is used by the recorder, the solver or any real-robot workflow.  It exists to run the operator's
calibration procedure end to end in MOCK (mock_hardware) without a camera: the hidden ground truth is held in
``config/mock_truth.yaml`` and the scene camera renders the printed boards from it.

Conventions: world = arm_left_base_link.  The truth file gives poses as xyz [m] and rpy_deg [deg] (URDF fixed-axis rpy).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .targets import TargetSpec, default_targets_path, load_targets_all, render_board
from .transforms import inv_T, make_T, xyz_rpy_to_T


def _pose(d) -> np.ndarray:
    return xyz_rpy_to_T(d["xyz"], np.radians(np.asarray(d["rpy_deg"], float)))


def default_truth_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("rebel_dual_arm_calibration"), "config", "mock_truth.yaml")
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "mock_truth.yaml")


@dataclass
class Truth:
    T_w_cam: np.ndarray                 # camera_link in world
    T_cl_opt: np.ndarray                # camera_link -> camera_color_optical_frame (the driver's static TF)
    T_w_rb: np.ndarray                  # arm_right_base_link in world
    T_l6_target: Dict[str, np.ndarray]  # target frame on link6, per arm
    T_w_board: np.ndarray               # table board in world
    K: np.ndarray
    D: np.ndarray
    size: Tuple[int, int]               # (width, height)
    touch_offset_mm: Dict[str, np.ndarray] = field(default_factory=dict)   # optional, for negative tests
    stickers: Dict[int, np.ndarray] = field(default_factory=dict)          # sticker id -> world xyz (hidden truth)
    sticker_kind: Dict[int, str] = field(default_factory=dict)             # id -> "fit" | "check"

    @property
    def T_w_opt(self) -> np.ndarray:
        return self.T_w_cam @ self.T_cl_opt

    def T_w_base(self, arm: str) -> np.ndarray:
        return self.T_w_rb if arm == "right" else np.eye(4)


def load_truth(path: Optional[str] = None) -> Truth:
    with open(path or default_truth_path(), encoding="utf-8") as f:
        d = yaml.safe_load(f)["mock_truth"]
    cam = d["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    # camera_link (x fwd, y left, z up) -> optical (z fwd, x right, y down), as the Orbbec driver publishes it
    R_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T_cl_opt = make_T(R_opt, d["camera_link_to_optical_xyz"])
    st = d.get("touch_stickers", {}) or {}
    stickers = {int(k): np.array(v, float) for kind in ("fit", "check") for k, v in (st.get(kind) or {}).items()}
    kinds = {int(k): kind for kind in ("fit", "check") for k in (st.get(kind) or {})}
    return Truth(T_w_cam=_pose(d["camera_link"]), T_cl_opt=T_cl_opt, T_w_rb=_pose(d["arm_right_base"]),
                 T_l6_target={a: _pose(d["target_mount"][a]) for a in ("left", "right")},
                 T_w_board=_pose(d["table_board"]), K=K, D=np.array(cam["distortion"], float),
                 size=(int(cam["width"]), int(cam["height"])), stickers=stickers, sticker_kind=kinds)


# ---------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------
def project(K: np.ndarray, T_opt_pts_frame: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole projection (no distortion) of Nx3 points given in a frame whose pose in the optical frame is
    ``T_opt_pts_frame``.  Returns (pixels Nx2, depth N)."""
    P = (T_opt_pts_frame[:3, :3] @ np.asarray(pts, float).T).T + T_opt_pts_frame[:3, 3]
    z = P[:, 2]
    uv = (K @ P.T).T
    return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None], z


def board_corners_obj(spec: TargetSpec) -> np.ndarray:
    w, h = spec.board_size()
    return np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], float)


def arm_board_visibility(truth: Truth, spec: TargetSpec, T_w_target: np.ndarray, margin_px: float = 70.0,
                         max_view_deg: float = 62.0, min_marker_px: float = 30.0) -> Tuple[bool, str, float]:
    """Is the flange board fully inside the image, facing the camera, with markers large enough?
    Returns (ok, reason, view_angle_deg).  The print faces -z of the target frame (z points into the board)."""
    T_opt_t = inv_T(truth.T_w_opt) @ T_w_target
    uv, z = project(truth.K, T_opt_t, board_corners_obj(spec))
    if np.any(z < 0.25):
        return False, "behind / too close to the camera", 180.0
    W, H = truth.size
    if np.any(uv[:, 0] < margin_px) or np.any(uv[:, 0] > W - margin_px) or np.any(uv[:, 1] < margin_px) \
            or np.any(uv[:, 1] > H - margin_px):
        return False, "board not inside the image", 0.0
    # outward normal of the print (toward the viewer) in the optical frame: -z_target; the viewer looks along +z
    n = -T_opt_t[:3, 2]
    ray = T_opt_t[:3, 3] / np.linalg.norm(T_opt_t[:3, 3])
    ang = float(np.degrees(np.arccos(np.clip(-n @ ray, -1, 1))))
    if ang > max_view_deg:
        return False, f"grazing / back view ({ang:.0f} deg)", ang
    # shortest projected marker side ~ marker_length * f / z * cos(angle)
    side = spec.marker_length * truth.K[0, 0] / float(np.mean(z)) * np.cos(np.radians(ang))
    if side < min_marker_px:
        return False, f"marker only {side:.0f} px", ang
    return True, "", ang


# ---------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------
class BoardImage:
    """A printed board as a grey image plus a small pyramid, with the board frame origin in pixels."""

    def __init__(self, spec: TargetSpec, px_per_m: float = 4000.0, margin: float = 0.012, plate: float = 0.008,
                 plate_value: int = 60, touch_crosshairs: bool = False):
        self.spec = spec
        img, (ox, oy) = render_board(spec, px_per_m, margin=margin)
        if plate > 0:     # dark rigid plate around the white paper margin
            p = int(round(plate * px_per_m))
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=plate_value)
            ox, oy = ox + p, oy + p
        if touch_crosshairs and spec.touch_points:
            arm = int(round(0.0035 * px_per_m))
            for (x, y) in spec.touch_points.values():
                cx, cy = int(round(ox + x * px_per_m)), int(round(oy + y * px_per_m))
                for width, val in ((max(3, int(0.0011 * px_per_m)), 255), (max(1, int(0.00015 * px_per_m)), 0)):
                    cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), int(val), width, cv2.LINE_AA)
                    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), int(val), width, cv2.LINE_AA)
        self.levels: List[Tuple[np.ndarray, float]] = [(img, 1.0)]
        s = 1.0
        cur = img
        for _ in range(3):
            cur = cv2.resize(cur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            s *= 0.5
            self.levels.append((cur, s))
        self.px_per_m = px_per_m
        self.origin = (ox, oy)       # at level 0

    def warp_into(self, frame: np.ndarray, K: np.ndarray, T_cam_board: np.ndarray) -> bool:
        """Draw this board into ``frame`` (BGR uint8, pinhole camera K) when the print faces the camera."""
        R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
        if (-R[:, 2]) @ (-t) <= 0.02 * np.linalg.norm(t):      # back side (or edge-on): the plate hides the print
            return False
        # board pixel (level 0) -> board metric (x, y, 1)
        ox, oy = self.origin
        # OpenCV pixel centres are at integer coordinates: a fill() edge at index a is at coordinate a - 0.5
        A = np.array([[1.0 / self.px_per_m, 0, -(ox - 0.5) / self.px_per_m],
                      [0, 1.0 / self.px_per_m, -(oy - 0.5) / self.px_per_m], [0, 0, 1.0]])
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
        h, w = self.levels[0][0].shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if np.any(~np.isfinite(dst)):
            return False
        # pyramid level: smallest one whose pixels are >= ~0.7 image pixel (avoids aliasing when minifying)
        side_px = 0.5 * (np.linalg.norm(dst[1] - dst[0]) + np.linalg.norm(dst[2] - dst[3]))
        scale = side_px / w                      # image px per level-0 board pixel
        lvl = 0
        while lvl < len(self.levels) - 1 and scale / self.levels[lvl][1] < 0.7:
            lvl += 1
        img, s = self.levels[lvl]
        # level pixel -> level-0 pixel (centre convention of INTER_AREA halving): u0 = (ul + 0.5) / s - 0.5
        S_inv = np.array([[1.0 / s, 0, 0.5 / s - 0.5], [0, 1.0 / s, 0.5 / s - 0.5], [0, 0, 1.0]])
        Hl = H @ S_inv                               # level pixel -> image
        Hh, Wh = frame.shape[:2]
        x0, y0 = int(max(0, np.floor(dst[:, 0].min()) - 2)), int(max(0, np.floor(dst[:, 1].min()) - 2))
        x1, y1 = int(min(Wh, np.ceil(dst[:, 0].max()) + 3)), int(min(Hh, np.ceil(dst[:, 1].max()) + 3))
        if x1 <= x0 or y1 <= y0:
            return False
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
        src = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.warpPerspective(src, shift @ Hl, (x1 - x0, y1 - y0), dst=roi, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
        frame[y0:y1, x0:x1] = roi
        return True


class SceneRenderer:
    """Renders the arm boards and the table board from the truth into a 1920x1080 colour image."""

    def __init__(self, truth: Truth, targets_file: Optional[str] = None, seed: int = 1, blur_px: float = 0.7,
                 noise_sigma: float = 2.0):
        self.truth = truth
        specs, table = load_targets_all(targets_file or default_targets_path())
        self.specs = specs
        self.table = table
        self.boards: Dict[str, BoardImage] = {a: BoardImage(s) for a, s in specs.items()}
        if table is not None:
            self.boards["table"] = BoardImage(table, margin=0.010, plate=0.0, touch_crosshairs=True)
        W, H = truth.size
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        base = 118 + 22 * (yy / H) - 14 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        tex = cv2.GaussianBlur(rng.normal(0, 9, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
        base += cv2.resize(tex, (W, H), interpolation=cv2.INTER_CUBIC)
        self.background = np.repeat(np.clip(base, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.noise = rng.normal(0, noise_sigma, (H + 64, W + 64, 1)).astype(np.float32)
        self.noise_rng = rng
        self.blur_px = blur_px
        # distortion: output pixel (distorted) -> source pixel in the ideal pinhole image
        grid = np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), -1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(grid, truth.K, truth.D, P=truth.K).reshape(H, W, 2)
        self.map_x, self.map_y = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
        self.distorted = bool(np.any(truth.D != 0))

    def render(self, T_w_targets: Dict[str, np.ndarray]) -> np.ndarray:
        """``T_w_targets``: pose of each board's target frame in world, keyed by 'left', 'right', 'table'.
        Returns an RGB uint8 image."""
        frame = self.background.copy()
        T_o_w = inv_T(self.truth.T_w_opt)
        order = sorted(T_w_targets.items(), key=lambda kv: -(T_o_w @ kv[1])[2, 3])   # far first
        for name, T in order:
            if name in self.boards:
                self.boards[name].warp_into(frame, self.truth.K, T_o_w @ T)
        if self.distorted:
            frame = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if self.blur_px > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), self.blur_px)
        H, W = frame.shape[:2]
        oy, ox = self.noise_rng.integers(0, 64, 2)
        out = np.clip(frame.astype(np.float32) + self.noise[oy:oy + H, ox:ox + W], 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
