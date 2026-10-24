"""Synthetic rectified top-face generator for the tape-strip detector.

``make_face`` returns (bgr uint8, truth).  Rect pixel centres are integer
coordinates; a strip is described by its cross coordinate at the face
centre (``center_px``), its perpendicular width, and its yaw (line rotation
relative to the ideal axis, positive counter-clockwise as displayed, same
convention as ``tape_strip``).

Materials: "clear" (low alpha, sheen, edge lines), "teflon" (opaque light
grey/white, glossy), "brown" (tape of nearly the box hue, slightly lighter),
"white" (opaque white), plus "printed" (opaque coloured).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

BASES = {"brown": (92, 134, 176), "white": (232, 232, 230)}   # BGR

# material -> dict(color BGR or None, alpha, gain, shade)
#   out = (1-alpha)*base + alpha*color ; then out*gain + shade (all channels)
MATERIALS = {
    "clear": dict(color=(235, 235, 235), alpha=0.20, gain=1.0, shade=0.0,
                  white_alpha=0.0, white_gain=0.93, edge=10.0),
    "teflon": dict(color=(208, 212, 212), alpha=0.97, gain=1.0, shade=0.0,
                   white_alpha=0.97, white_gain=1.0, edge=14.0),
    "brown": dict(color=(104, 146, 190), alpha=0.9, gain=1.0, shade=0.0,
                  white_alpha=0.9, white_gain=1.0, edge=12.0),
    "white": dict(color=(244, 244, 244), alpha=0.97, gain=1.0, shade=0.0,
                  white_alpha=0.97, white_gain=1.0, edge=10.0),
    "printed": dict(color=(70, 70, 200), alpha=0.9, gain=1.0, shade=0.0,
                    white_alpha=0.9, white_gain=1.0, edge=10.0),
}


def _base(rng, W, H, kind, noise=1.0):
    col = np.array(BASES[kind], np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[:] = col
    # low-frequency blotches (illumination / board variation)
    low = rng.normal(0, 1, (H // 24 + 2, W // 24 + 2)).astype(np.float32)
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    img += (low * 3.0 * noise)[:, :, None]
    # fibre texture: noise blurred along a random direction + pixel noise
    fib = rng.normal(0, 1, (H, W)).astype(np.float32)
    ang = rng.uniform(0, math.pi)
    k = np.zeros((9, 9), np.float32)
    c = 4
    for t in np.linspace(-4, 4, 25):
        k[int(round(c + t * math.sin(ang))), int(round(c + t * math.cos(ang)))] = 1
    k /= k.sum()
    fib = cv2.filter2D(fib, -1, k) * 4.0 * noise
    img += fib[:, :, None]
    img += rng.normal(0, 2.5 * noise, (H, W, 1)).astype(np.float32)
    img += rng.normal(0, 1.0 * noise, (H, W, 3)).astype(np.float32)
    return img


def _signed_dist(W, H, axis, center_px, yaw_deg):
    """Perpendicular signed distance of each pixel to the strip centreline."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    t = math.tan(math.radians(yaw_deg))
    cs = math.cos(math.radians(yaw_deg))
    if axis == "h":     # y(x) = c - tan(yaw) (x - cx)
        return (yy - center_px + t * (xx - cx)) * cs
    return (xx - center_px - t * (yy - cy)) * cs      # x(y) = c + tan(yaw)(y-cy)


def _along(W, H, axis, yaw_deg):
    """Coordinate along the strip (0..1 over the face)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (xx / max(W - 1, 1)) if axis == "h" else (yy / max(H - 1, 1))


def make_face(seed: int = 0, W: int = 300, H: int = 240, base: str = "brown",
              strips=(), occluder=None, noise: float = 1.0,
              mm_per_px: float = 0.8):
    """Generate a rectified face.

    ``strips``: iterable of dicts with keys axis ("h"|"v"), material,
    center_px, width_px, yaw_deg (default 0), streak (None | dict(rel=
    position across the strip in units of half-width [-1.5..1.5],
    width_px, length_frac, start_frac, strength)), gap (None | dict(
    offset_px, width_px, depth)).  ``occluder``: dict(x0,y0,x1,y1,color).
    Returns (bgr, truth) with truth = {"strips": [...], "mm_per_px": ...}.
    """
    rng = np.random.default_rng(seed)
    img = _base(rng, W, H, base, noise)
    is_white = base == "white"
    truth = []
    for s in strips:
        m = MATERIALS[s["material"]]
        axis = s["axis"]
        yaw = float(s.get("yaw_deg", 0.0))
        d = _signed_dist(W, H, axis, s["center_px"], yaw)
        hw = s["width_px"] / 2.0
        cover = np.clip(hw - np.abs(d) + 0.5, 0.0, 1.0)[:, :, None]
        alpha = m["white_alpha"] if is_white else m["alpha"]
        gain = m["white_gain"] if is_white else m["gain"]
        col = np.array(m["color"], np.float32)
        tape = (1 - alpha) * img + alpha * col
        tape = tape * gain + m["shade"]
        # tape texture is smoother than the cardboard
        sm = cv2.GaussianBlur(tape, (0, 0), 1.2)
        tape = 0.5 * tape + 0.5 * sm
        out = (1 - cover) * img + cover * tape
        # edge lines: thin dark/bright rim at both edges (adhesive edge)
        rim = np.exp(-0.5 * ((np.abs(d) - hw) / 0.8) ** 2)[:, :, None]
        out = out + rim * (-m["edge"] if not is_white else -m["edge"]) * 0.6
        # gloss on edge inside
        rim_in = np.exp(-0.5 * ((np.abs(d) - (hw - 1.8)) / 0.8) ** 2)
        out = out + (rim_in * m["edge"] * 0.5)[:, :, None] * (np.abs(d) < hw)[:, :, None]
        img = out
        # gap valley (physical flap gap), under/through the tape
        g = s.get("gap")
        if g:
            dg = d - g.get("offset_px", 0.0)
            prof = np.exp(-0.5 * (dg / max(g.get("width_px", 2) / 2.35, 0.4)) ** 2)
            img = img * (1 - (g.get("depth", 0.45)) * prof)[:, :, None]
        # specular streak (limited length, gaussian cross profile)
        st = s.get("streak")
        if st:
            ds = d - st["rel"] * hw
            sw = max(st.get("width_px", 4) / 2.35, 0.5)
            prof = np.exp(-0.5 * (ds / sw) ** 2)
            a = _along(W, H, axis, yaw)
            lf = st.get("length_frac", 0.3)
            s0 = st.get("start_frac", 0.3)
            along = np.clip((a - s0) / 0.03, 0, 1) * np.clip((s0 + lf - a) / 0.03, 0, 1)
            wgt = (prof * along * st.get("strength", 0.6))[:, :, None]
            img = (1 - wgt) * img + wgt * 255.0
        truth.append({"axis": s["axis"], "center_px": float(s["center_px"]),
                      "width_px": float(s["width_px"]), "yaw_deg": yaw,
                      "material": s["material"], "seam_axis":
                      "lr" if axis == "h" else "bf",
                      "gap_center_px": (None if not g else float(
                          s["center_px"] + g.get("offset_px", 0.0)))})
    if occluder:
        x0, y0, x1, y1 = [int(v) for v in (occluder["x0"], occluder["y0"],
                                           occluder["x1"], occluder["y1"])]
        img[y0:y1, x0:x1] = np.array(occluder.get("color", (240, 240, 240)),
                                     np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)
    return out, {"strips": truth, "mm_per_px": mm_per_px, "base": base,
                 "W": W, "H": H}
