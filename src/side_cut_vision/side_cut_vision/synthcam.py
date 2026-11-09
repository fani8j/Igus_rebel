"""Synthetic RGB-D renderer (tests and the MOCK-ONLY camera node).

World = snapshot frame. A carton (convex prism: top face, four side faces, bottom) is placed at the
snapshot pose displaced by a hidden offset, over a floor plane. The camera is the left Gemini 305 with
its REAL intrinsics and 8 rational distortion coefficients, at a pose derived from the real hand-eye
and a plausible tool pose (documented in `approach_pose`).

Appearance (what the real S0 frames of 2026-10-20 look like):
  * cardboard top with texture and noise, darker side face, grey floor far below;
  * a dark seam (flap gap) line perpendicular to the cut-side edge under a clear-tape band with
    specular speckle (bright blobs) and depth holes (glossy tape);
  * the knife body as a dark region entering from the right, much nearer to the camera;
  * 1-pixel invalid depth band at large depth jumps; depth noise growing with range; uint16 mm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .calib import DEFAULT_TOOL_POLYGON_PX, HandEye
from .camera import CameraModel
from .geom import inv_T, make_T, unit
from .model import ArmModel

# Left-arm TCP in igus_tool0 (pivot calibration 2026-10-12, rebel_tool.macro.xacro default).
TCP_IN_TOOL0 = np.array([-0.042069, -0.089590, 0.104551])
LINK6_TO_TOOL0_Z = 0.0527       # link6 -> igus_tool0: +52.7 mm along z, rotated pi about z


def tip_in_link6() -> np.ndarray:
    Rz = np.diag([-1.0, -1.0, 1.0])
    return Rz @ TCP_IN_TOOL0 + np.array([0, 0, LINK6_TO_TOOL0_Z])


@dataclass
class SceneSpec:
    # hidden carton displacement vs the snapshot, in the model cut basis (mm) and yaw (deg, + = edge
    # turning toward +across/outward as it advances along)
    d_along_mm: float = 8.0
    d_across_mm: float = -6.0
    d_normal_mm: float = 3.0
    yaw_deg: float = 2.0
    box_height_m: float = 0.20
    floor_below_top_m: float = 0.70
    seam_width_m: float = 0.003
    tape_half_width_m: float = 0.024
    tape: bool = True
    speckle_frac: float = 0.18         # fraction of the tape band covered by specular blobs
    hole_frac: float = 0.10            # fraction of the tape band with invalid depth
    glare: bool = False                # saturate the colour over the seam region (tape glare)
    depth_noise_mm_at_14cm: float = 0.4
    color_noise: float = 4.0
    light_gradient: float = 0.0        # multiplicative left-to-right lighting gradient (+-)
    tool: bool = True
    seed: int = 7


def approach_pose(arm: ArmModel, he: HandEye, tilt_deg: float = 12.0, standoff_m: float = 0.0165,
                  below_top_m: float = 0.003, tip_along_m: float = 0.030, roll_deg: float = 0.0):
    """Camera pose at a left side-cut approach.

    Assumptions (not measured): the knife axis (tip direction) points down and leans `tilt_deg`
    toward the carton interior; the optical axis is parallel to the knife axis (the hand-eye puts it
    within ~1 deg of it); image right = +along (back->front), image down = outward. The tool tip
    is `standoff_m` outside the model face, `below_top_m` below the top plane and `tip_along_m`
    beyond the crossing (so the seam is to the left of the tip, as in the S0 frames). The camera
    origin follows from the tip position of the real hand-eye + TCP (tip at ~(948, 424) px,
    0.136 m in front of the camera). Returns (T_wo, tip_world, T_w_link6).
    """
    s, c = np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))
    z = unit(-s * arm.across - c * arm.normal)
    x = unit(arm.along - (arm.along @ z) * z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if roll_deg:
        a = np.radians(roll_deg)
        R = R @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tip_w = arm.S + tip_along_m * arm.along + standoff_m * arm.across - below_top_m * arm.normal
    T_l6_o = he.T_link6_optical
    tip_cam = inv_T(T_l6_o) @ np.append(tip_in_link6(), 1.0)
    origin = tip_w - R @ tip_cam[:3]
    T_wo = make_T(R, origin)
    return T_wo, tip_w, T_wo @ inv_T(T_l6_o)


class Carton:
    """Hidden carton: convex prism described by outward plane normals and offsets (world)."""

    def __init__(self, arm: ArmModel, spec: SceneSpec):
        th = np.radians(spec.yaw_deg)
        R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        loc = arm.to_local(arm.corners)
        ac = loc[:, :2] @ R2.T + np.array([spec.d_along_mm, spec.d_across_mm]) * 1e-3
        h = spec.d_normal_mm * 1e-3
        self.arm = arm
        self.spec = spec
        self.R2 = R2
        self.shift = np.array([spec.d_along_mm, spec.d_across_mm]) * 1e-3
        self.h_top = h
        self.corners_local = ac            # (4, 2)
        # prism planes in local (a, c, h) with outward normals
        planes = []
        planes.append((np.array([0, 0, 1.0]), h))                           # top
        planes.append((np.array([0, 0, -1.0]), -(h - spec.box_height_m)))  # bottom
        cen = ac.mean(axis=0)
        for i in range(4):
            p, q = ac[i], ac[(i + 1) % 4]
            e = q - p
            nrm = np.array([e[1], -e[0]])
            nrm /= np.linalg.norm(nrm)
            if nrm @ (p - cen) < 0:
                nrm = -nrm
            planes.append((np.array([nrm[0], nrm[1], 0.0]), float(nrm @ p)))
        self.planes_local = planes
        B = arm.basis                       # rows along, across, normal
        self.planes_world = []
        for nl, dl in planes:
            nw = nl @ B
            # local point p_l = (P - S) @ B.T ; plane nl . p_l = dl  ->  nw . (P - S) = dl
            self.planes_world.append((nw, dl + nw @ arm.S))
        # cut-side face = the side plane whose normal is closest to +across
        side = [(i, pl) for i, pl in enumerate(planes) if i >= 2]
        self.cut_face_idx = max(side, key=lambda t: t[1][0][1])[0]

    def hidden_crossing_world(self) -> np.ndarray:
        return self.arm.from_local(np.array([self.spec.d_along_mm, self.spec.d_across_mm,
                                             self.spec.d_normal_mm]) * 1e-3)

    def local_ac_of(self, P_w: np.ndarray):
        """(a', c') of world points in the HIDDEN carton frame (origin = hidden crossing, axes
        rotated by the hidden yaw)."""
        L = self.arm.to_local(P_w)[..., :2] - self.shift
        # inverse rotation
        a = L[..., 0] * self.R2[0, 0] + L[..., 1] * self.R2[1, 0]
        c = L[..., 0] * self.R2[0, 1] + L[..., 1] * self.R2[1, 1]
        return a, c


def _smooth_noise(shape, scale_px, rng, amp=1.0):
    small = rng.normal(size=(max(2, shape[0] // scale_px + 2), max(2, shape[1] // scale_px + 2)))
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) * amp


def render(cam: CameraModel, T_wo: np.ndarray, arm: ArmModel, spec: Optional[SceneSpec] = None,
           stride: int = 1, _tool_scale: float = 1.0):
    """Render (color uint8 RGB HxWx3, depth uint16 mm HxW, truth dict) at full resolution."""
    spec = spec or SceneSpec()
    if stride > 1:
        # render on a reduced grid (same distortion model, K scaled) and upsample: colour linear, depth nearest
        sc = 1.0 / stride
        K = cam.K.copy()
        K[:2] *= sc
        K[2, 2] = 1.0
        small = CameraModel(K, cam.D, cam.width // stride, cam.height // stride, cam.depth_scale_m, cam.frame_id)
        c, d, truth = render(small, T_wo, arm, spec, 1, _tool_scale=sc)
        return (cv2.resize(c, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR),
                cv2.resize(d, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST), truth)
    rng = np.random.default_rng(spec.seed)
    carton = Carton(arm, spec)
    H, W = cam.height, cam.width
    g = cam.norm_grid(1).reshape(-1, 2).astype(np.float64)
    dirs_c = np.concatenate([g, np.ones((len(g), 1))], axis=1)       # z-normalised: depth z = t
    R, o = T_wo[:3, :3], T_wo[:3, 3]
    dirs = dirs_c @ R.T                                               # world dirs with unit z-depth

    # ---- carton: slab intersection
    t_in = np.full(len(dirs), -np.inf)
    t_out = np.full(len(dirs), np.inf)
    face = np.full(len(dirs), -1, np.int8)
    for i, (nw, dw) in enumerate(carton.planes_world):
        den = dirs @ nw
        num = dw - o @ nw
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        ent = den < 0
        upd = ent & (t > t_in)
        t_in = np.where(upd, t, t_in)
        face = np.where(upd, i, face)
        t_out = np.where(~ent & (t < t_out), t, t_out)
    hit_c = (t_in < t_out) & (t_in > 0)
    # ---- floor: plane normal = world z (arm.normal is ~z), at top - floor_below_top
    zf = (arm.S + (spec.d_normal_mm * 1e-3 - spec.floor_below_top_m) * arm.normal)[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_f = (zf - o[2]) / dirs[:, 2]
    hit_f = (t_f > 0) & np.isfinite(t_f)
    t = np.where(hit_c, t_in, np.where(hit_f, t_f, np.nan))
    which = np.where(hit_c, face, np.where(hit_f, -2, -3))          # -2 floor, -3 nothing
    P = o[None, :] + t[:, None] * dirs
    a_h, c_h = carton.local_ac_of(P)

    # ---- colour
    col = np.zeros((H * W, 3), np.float64)
    tex = _smooth_noise((H, W), 24, rng, 1.0).reshape(-1)
    tex2 = _smooth_noise((H, W), 6, rng, 1.0).reshape(-1)
    top = which == 0
    side = (which >= 2)
    floor = which == -2
    col[top] = np.array([200, 152, 98.0]) + 9 * tex[top, None] + 5 * tex2[top, None]
    col[side] = np.array([158, 116, 72.0]) + 8 * tex[side, None]
    col[floor] = np.array([128, 134, 142.0]) + 7 * tex[floor, None]
    col[which == 1] = np.array([140, 100, 62.0])
    # seam (dark gap line along c at a' = 0) and tape band on the top face
    seam = top & (np.abs(a_h) <= spec.seam_width_m / 2) & (c_h <= 0)
    col[seam] = np.array([62, 40, 24.0])
    if spec.tape:
        band = top & (np.abs(a_h) <= spec.tape_half_width_m) & (c_h <= 0)
        col[band] += 8.0
        # specular speckle blobs and wrinkle streaks: bright, partly saturated
        spk = _smooth_noise((H, W), 5, rng, 1.0).reshape(-1)
        thr = np.quantile(spk[band], 1.0 - spec.speckle_frac) if band.any() else 9
        sp = band & (spk >= thr)
        col[sp] = 0.25 * col[sp] + 0.75 * np.array([240, 238, 232.0])
        # the seam stays visible through the tape but is partly washed out
        s2 = seam & band
        col[s2] = 0.65 * col[s2] + 0.35 * np.array([150, 130, 110.0]) * 0.6
    if spec.glare:
        gl = top & (np.abs(a_h) <= 0.040) & (c_h <= 0)
        col[gl] = np.array([252, 250, 246.0])
    if spec.light_gradient:
        gx = np.tile(np.linspace(1 - spec.light_gradient, 1 + spec.light_gradient, W), H)
        col *= gx[:, None]
    col += rng.normal(scale=spec.color_noise, size=col.shape)

    # ---- depth (z along optical axis == t because dirs have unit z in the camera frame)
    z = t.copy()
    sigma = spec.depth_noise_mm_at_14cm * 1e-3 * (np.nan_to_num(z, nan=0.14) / 0.14) ** 1.5
    zn = z + rng.normal(size=z.shape) * sigma
    if spec.tape:
        holes = _smooth_noise((H, W), 7, rng, 1.0).reshape(-1)
        band = top & (np.abs(a_h) <= spec.tape_half_width_m) & (c_h <= 0)
        thr = np.quantile(holes[band], 1.0 - spec.hole_frac) if band.any() else 9
        hole = band & (holes >= thr)
        zn[hole] = np.nan
        noisy = band & ~hole
        zn[noisy] += rng.normal(scale=2.0 * 1e-3, size=noisy.sum())
    depth_m = zn.reshape(H, W)

    # ---- the knife body (nearer than anything else), a polygon in pixels
    color_img = col.reshape(H, W, 3)
    if spec.tool:
        poly = np.array(DEFAULT_TOOL_POLYGON_PX, np.float64) * _tool_scale
        cen = poly.mean(axis=0)
        poly = cen + (poly - cen) * 0.97 + 0                      # slightly smaller than the detector mask
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.rint(poly).astype(np.int32)], 1)
        tm = m.astype(bool)
        shade = 70 + 25 * (np.arange(W)[None, :] / W)
        color_img[tm] = np.stack([shade.repeat(H, 0)] * 3, -1)[tm] + rng.normal(scale=3, size=(tm.sum(), 3))
        depth_m = np.where(tm, 0.060 + 0.0002 * (np.arange(W)[None, :] - 900) / 10.0, depth_m)

    # ---- flying-pixel invalidation at depth jumps (1 px each side)
    d = np.nan_to_num(depth_m, nan=0.0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    dmax = cv2.dilate(d, k)
    dmin = cv2.erode(np.where(d > 0, d, 1e3).astype(np.float32), k)
    jump = (dmax - dmin) > 0.03
    depth_m = np.where(jump, np.nan, depth_m)

    depth_mm = np.where(np.isfinite(depth_m) & (depth_m > 0), np.rint(depth_m * 1000.0), 0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    color_u8 = np.clip(color_img, 0, 255).astype(np.uint8)
    truth = {
        "delta_mm": [spec.d_along_mm, spec.d_across_mm, spec.d_normal_mm],
        "yaw_deg": spec.yaw_deg,
        "hidden_crossing_world": carton.hidden_crossing_world(),
    }
    return color_u8, depth_mm, truth
