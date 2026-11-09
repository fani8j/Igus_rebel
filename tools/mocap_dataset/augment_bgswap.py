#!/usr/bin/env python3
"""Generative-free background / table swap augmentation of a LeRobot v3.0 dataset (needs the masks of segment_masks.py).

    python3 augment_bgswap.py SRC_ROOT MASK_ROOT --out OUT_ROOT [--copies 3] [--seed 20261031] [--episodes 0,1] [--workers 4]
    python3 augment_bgswap.py --validate OUT_ROOT SRC_ROOT        # lerobot 0.4.4 LeRobotDataset checks
    python3 augment_bgswap.py --sheet OUT_ROOT SHEET.png          # before/after contact sheet

Per source episode and copy ONE appearance is drawn (seeded by crc32(episode name), copy, global seed) and used for every
frame of both cameras, so nothing flickers inside an episode:
  * head camera (fixed): everything that is neither robot nor box (label 0 of the mask) is replaced by a procedural
    texture (wood, cloth, paper, plain colour, noise, gingham) mixed with the recoloured original background (the mix
    factor is part of the appearance), plus 0-2 static procedural distractor objects painted into the background layer
    (so the arm and the box occlude them naturally).
  * wrist camera (moves with the arm): NO texture and NO mask-based replacement (the carton fills the view in close-ups
    and the detector does not label it there, so any background replacement would paint the carton).  Only one global
    photometric change per episode, identical for every frame: brightness x0.8-1.2, contrast x0.8-1.2, saturation
    x0.8-1.2, hue +-0.03 turn (+-10.8 deg), gamma 0.85-1.15.  Geometry is untouched.
  * robot and box pixels (label 1/2, OR-ed over t-1..t+1 to absorb mask jitter) are kept byte-identical up to a feather
    band of ``feather_px`` pixels around them, which blends towards the new background; the band lies outside the mask.
  * a mild global colour jitter (brightness / contrast / saturation, per copy) is applied to ALL pixels of the episode
    (head camera) to keep the lighting consistent; it is part of the appearance record.  With jitter off the
    mask pixels are byte-identical to the source (test_augment_bgswap.py).
Output: a NEW LeRobot v3.0 dataset: the source episodes (stream-copied videos) + ``copies`` swapped copies (libx264 CRF 16,
GOP 2, yuv420p, same size and fps, packed by lerobot's concatenate_video_files), ``augmentation.copy_index`` per frame,
``meta/source_episodes.json`` and ``AUGMENT.json``.  The source is never modified.  Evaluation must split by SOURCE episode.

Throttle: <= 4 worker processes, one decode + one encode thread each; run under ``nice -n 19`` with OMP/OPENBLAS/MKL threads = 1.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

H, W = 480, 640
TEX_KINDS = ("wood", "cloth", "paper", "plain", "noise", "gingham")
DEFAULT_SEED = 20261031
HELD_OUT = ["mocap_right_2_ep008", "mocap_right_3_ep004", "mocap_right_farhan_ep004"]
COPY_KEY = "augmentation.copy_index"


# ---------------------------------------------------------------------------------------------------- appearance
@dataclass
class Appearance:
    copy_index: int
    seed: int
    tex_kind: str
    tex_colors: list            # two RGB colours 0..255
    tex_params: dict
    tex_mix_head: float         # weight of the texture in the head background (rest: recoloured original)
    hue_deg: float              # recolour of the original background
    gain: list
    gamma: float
    jitter: dict                # brightness (add, 0..255 units), contrast (mult), saturation (mult)
    distractors: list = field(default_factory=list)
    wrist_photo: dict = field(default_factory=dict)   # global photometric change of the wrist camera


def seed_for(global_seed: int, name: str, copy: int) -> int:
    return int(np.random.SeedSequence([int(global_seed), zlib.crc32(name.encode()), int(copy)]).generate_state(1)[0])


def _hsv_rgb(h, s, v):
    return [float(c) for c in cv2.cvtColor(np.uint8([[[h * 179, s * 255, v * 255]]]), cv2.COLOR_HSV2RGB)[0, 0]]


def sample_appearance(seed: int, copy_index: int, max_distractors: int = 2) -> Appearance:
    rng = np.random.default_rng(seed)
    kind = TEX_KINDS[int(rng.integers(len(TEX_KINDS)))]
    if kind == "wood":
        h = rng.uniform(0.04, 0.10)
        c1 = _hsv_rgb(h, rng.uniform(0.45, 0.75), rng.uniform(0.35, 0.75))
        c2 = _hsv_rgb(h, rng.uniform(0.5, 0.85), rng.uniform(0.15, 0.45))
    elif kind == "paper":
        c1 = _hsv_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.18), rng.uniform(0.82, 0.98))
        c2 = _hsv_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.25), rng.uniform(0.7, 0.9))
    else:
        h = rng.uniform(0, 1)
        c1 = _hsv_rgb(h, rng.uniform(0.15, 0.9), rng.uniform(0.3, 0.95))
        c2 = _hsv_rgb((h + rng.uniform(-0.12, 0.12)) % 1.0, rng.uniform(0.15, 0.9), rng.uniform(0.2, 0.9))
    params = {"angle": float(rng.uniform(0, np.pi)), "period": float(rng.uniform(14, 60)),
              "grain": float(rng.uniform(0.03, 0.12)), "lowfreq": float(rng.uniform(0.2, 0.7)), "seed": int(rng.integers(2 ** 31))}
    app = Appearance(
        copy_index=copy_index, seed=seed, tex_kind=kind, tex_colors=[round(c, 1) for c in c1] + [round(c, 1) for c in c2],
        tex_params=params, tex_mix_head=float(rng.uniform(0.6, 1.0)),
        hue_deg=float(rng.uniform(-60, 60)), gain=[float(g) for g in rng.uniform(0.8, 1.2, 3)], gamma=float(rng.uniform(0.8, 1.25)),
        jitter={"brightness": float(rng.uniform(-10, 10)), "contrast": float(rng.uniform(0.92, 1.08)),
                "saturation": float(rng.uniform(0.9, 1.1))})
    shapes = ("ellipse", "rect", "poly")
    for _ in range(int(rng.integers(0, max_distractors + 1))):
        app.distractors.append({"shape": shapes[int(rng.integers(3))], "u": float(rng.uniform(0, 1)), "v": float(rng.uniform(0, 1)),
                                "size": float(rng.uniform(36, 110)), "aspect": float(rng.uniform(0.5, 1.6)),
                                "rot": float(rng.uniform(0, 180)),
                                "color": _hsv_rgb(rng.uniform(0, 1), rng.uniform(0.2, 0.95), rng.uniform(0.25, 0.95)),
                                "shadow": float(rng.uniform(0.15, 0.4))})
    # drawn last so the head appearance of a given seed is unchanged
    app.wrist_photo = {"brightness": float(rng.uniform(0.8, 1.2)), "contrast": float(rng.uniform(0.8, 1.2)),
                       "saturation": float(rng.uniform(0.8, 1.2)), "hue_deg": float(rng.uniform(-0.03, 0.03) * 360.0),
                       "gamma": float(rng.uniform(0.85, 1.15))}
    return app


# ---------------------------------------------------------------------------------------------------- textures
def _smooth_noise(rng, h, w, cells: int) -> np.ndarray:
    """Value noise in [0, 1] with about ``cells`` cells across the larger side."""
    ch, cw = max(2, int(round(cells * h / max(h, w)))), max(2, cells)
    n = rng.random((ch + 2, cw + 2)).astype(np.float32)
    return cv2.resize(n, (w + 2 * (w // cw + 1), h + 2 * (h // ch + 1)), interpolation=cv2.INTER_CUBIC)[(h // ch + 1):(h // ch + 1) + h,
                                                                                                        (w // cw + 1):(w // cw + 1) + w]


def _fbm(rng, h, w, octaves=4, base_cells=4) -> np.ndarray:
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        out += amp * _smooth_noise(rng, h, w, base_cells * 2 ** o)
        tot += amp
        amp *= 0.5
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def make_texture(app: Appearance, h: int = H, w: int = W) -> np.ndarray:
    """Procedural texture, float32 (h, w, 3) in 0..255, fully determined by the appearance."""
    p, kind = app.tex_params, app.tex_kind
    rng = np.random.default_rng(p["seed"])
    c1, c2 = np.array(app.tex_colors[:3], np.float32), np.array(app.tex_colors[3:], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "plain":
        t = 0.5 + p["lowfreq"] * 0.25 * (_fbm(rng, h, w, 2, 2) - 0.5)
        t = np.full((h, w), 0.0, np.float32) + (_fbm(rng, h, w, 2, 2) - 0.5) * 0.25 * p["lowfreq"]
        t = t + rng.normal(0, p["grain"] * 0.15, (h, w)).astype(np.float32)
        out = c1[None, None] * (1 + t[..., None])
    elif kind == "noise":
        t = _fbm(rng, h, w, 5, 3)
        out = c1 + (c2 - c1) * t[..., None]
    elif kind == "wood":
        u = xx * np.cos(p["angle"]) + yy * np.sin(p["angle"])
        warp = _fbm(rng, h, w, 3, 3) * 3.0
        rings = 0.5 + 0.5 * np.sin(2 * np.pi * (u / (p["period"] * 2.5) + warp))
        grain = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
        grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.5 + 12 * abs(np.cos(p["angle"])), sigmaY=0.5 + 12 * abs(np.sin(p["angle"])))
        grain = (grain - grain.mean()) / (grain.std() + 1e-6)
        t = np.clip(rings * 0.75 + 0.5 + grain * 0.15 - 0.4, 0, 1)
        out = c1 + (c2 - c1) * t[..., None]
    elif kind == "cloth":
        per = max(4.0, p["period"] / 4)
        t = 0.5 + 0.25 * np.sin(2 * np.pi * xx / per) * np.sin(2 * np.pi * yy / per)
        t = t + 0.5 * (_fbm(rng, h, w, 3, 3) - 0.5) * p["lowfreq"] + rng.normal(0, p["grain"] * 0.5, (h, w)).astype(np.float32)
        out = c1 + (c2 - c1) * np.clip(t, 0, 1)[..., None]
    elif kind == "paper":
        t = 0.5 + 0.6 * (_fbm(rng, h, w, 4, 2) - 0.5) * p["lowfreq"] + rng.normal(0, p["grain"] * 0.35, (h, w)).astype(np.float32)
        out = c1 + (c2 - c1) * np.clip(t, 0, 1)[..., None]
    else:                                                   # gingham: soft checks
        per = max(18.0, p["period"])
        a = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * np.cos(p["angle"] * 0.1) + yy * np.sin(p["angle"] * 0.1)) / per)
        b = 0.5 + 0.5 * np.sin(2 * np.pi * (-xx * np.sin(p["angle"] * 0.1) + yy * np.cos(p["angle"] * 0.1)) / per)
        t = np.clip(0.5 * (np.clip(a * 1.6 - 0.3, 0, 1) + np.clip(b * 1.6 - 0.3, 0, 1)), 0, 1)
        t = t + rng.normal(0, p["grain"] * 0.2, (h, w)).astype(np.float32)
        out = c1 + (c2 - c1) * np.clip(t, 0, 1)[..., None]
    return np.clip(out, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- distractors
def render_distractors(app: Appearance, fg_freq: np.ndarray | None, h: int = H, w: int = W):
    """(rgb float32 (h, w, 3), alpha (h, w), shade (h, w)) layer of the static distractors, or None.  Positions are drawn from
    pixels that the robot/box rarely cover (``fg_freq`` = fraction of frames with a robot/box label, per pixel).  ``shade``
    (<= 1) darkens the background under a soft drop shadow."""
    if not app.distractors:
        return None
    rgb = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    shade = np.ones((h, w), np.float32)
    y0 = int(0.35 * h)
    cand = np.argwhere((fg_freq < 0.03)[y0:, :]) if fg_freq is not None else np.zeros((0, 2), int)
    for d in app.distractors:
        if len(cand):
            y, x = cand[int(d["v"] * (len(cand) - 1))]
            cy, cx = float(y + y0), float(x)
        else:
            cy, cx = (0.35 + 0.6 * d["v"]) * h, d["u"] * w
        sz, asp = d["size"], d["aspect"]
        m = np.zeros((h, w), np.uint8)
        if d["shape"] == "ellipse":
            cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(sz / 2 * asp)), max(1, int(sz / 2))), d["rot"], 0, 360, 255, -1, cv2.LINE_AA)
        elif d["shape"] == "rect":
            cv2.fillConvexPoly(m, cv2.boxPoints(((cx, cy), (sz * asp, sz), d["rot"])).astype(np.int32), 255, cv2.LINE_AA)
        else:
            ang = np.sort(np.random.default_rng(app.seed + int(sz)).uniform(0, 2 * np.pi, 5))
            pts = np.stack([cx + sz / 2 * asp * np.cos(ang), cy + sz / 2 * np.sin(ang)], 1)
            cv2.fillConvexPoly(m, pts.astype(np.int32), 255, cv2.LINE_AA)
        a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 1.0)
        sh = cv2.GaussianBlur(np.roll(m.astype(np.float32) / 255.0, (int(sz * 0.1), int(sz * 0.06)), (0, 1)), (0, 0), max(1.0, sz * 0.06))
        shade *= 1.0 - d["shadow"] * sh
        rgb = rgb * (1 - a[..., None]) + np.array(d["color"], np.float32)[None, None] * a[..., None]
        alpha = np.clip(alpha + a * (1 - alpha), 0, 1)
    return rgb, alpha, shade


# ---------------------------------------------------------------------------------------------------- compositing math
_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)


def recolor(frame: np.ndarray, hue_deg: float, gain, gamma: float) -> np.ndarray:
    """Hue rotation (YIQ), per-channel gain and gamma of an RGB uint8/float frame -> float32 0..255."""
    f = frame.astype(np.float32)
    a = np.deg2rad(hue_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)
    m = np.linalg.inv(_YIQ) @ rot @ _YIQ
    f = f @ m.T
    f = 255.0 * np.power(np.clip(f / 255.0, 0, 1), gamma)
    return np.clip(f * np.asarray(gain, np.float32)[None, None, :], 0, 255)


def global_jitter(img: np.ndarray, jit: dict) -> np.ndarray:
    """Brightness / contrast / saturation of an RGB float image -> float32 0..255."""
    f = img.astype(np.float32)
    f = (f - 128.0) * jit["contrast"] + 128.0 + jit["brightness"]
    g = f @ np.array([0.299, 0.587, 0.114], np.float32)
    f = g[..., None] + (f - g[..., None]) * jit["saturation"]
    return np.clip(f, 0, 255)


def soft_alpha(fg: np.ndarray, grow_px: int = 2, feather_px: int = 4) -> np.ndarray:
    """Foreground weight in [0, 1]: exactly 1 on ``fg`` pixels and on the ``grow_px`` dilation guard; outside it the weight
    falls smoothly (smoothstep of the distance) to exactly 0 at ``feather_px`` pixels."""
    fg_u8 = (fg > 0).astype(np.uint8)
    hard = cv2.dilate(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))) if grow_px > 0 else fg_u8
    if feather_px <= 0:
        return hard.astype(np.float32)
    dist = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(1.0 - dist / float(feather_px), 0.0, 1.0)
    a = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    a[hard > 0] = 1.0
    return a


def composite(frame: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """out = alpha * frame + (1 - alpha) * bg, uint8; pixels with alpha == 1 are returned byte-identical."""
    a = alpha[..., None]
    out = np.rint(a * frame.astype(np.float32) + (1 - a) * bg).astype(np.uint8)
    full = alpha >= 1.0
    out[full] = frame[full]
    return out


def background_layer(frame: np.ndarray, tex: np.ndarray, app: Appearance, cam: str, distract=None) -> np.ndarray:
    """New background (float32 0..255) the foreground is composited on."""
    rec = recolor(frame, app.hue_deg, app.gain, app.gamma)
    t = app.tex_mix_head
    bg = (1 - t) * rec + t * tex
    if distract is not None and cam == "head":
        rgb, a, shade = distract
        bg = bg * shade[..., None]
        bg = bg * (1 - a[..., None]) + rgb * a[..., None]
    return bg


def union_fg(labels: np.ndarray, i: int) -> np.ndarray:
    """Foreground (robot or box) of frame i OR-ed with i-1 and i+1 (mask jitter guard)."""
    n = len(labels)
    m = labels[i] > 0
    if i > 0:
        m = m | (labels[i - 1] > 0)
    if i + 1 < n:
        m = m | (labels[i + 1] > 0)
    return m


def wrist_photometric(frame: np.ndarray, wp: dict) -> np.ndarray:
    """Per-episode global photometric change of a wrist frame (hue, gamma, contrast, brightness, saturation) -> uint8."""
    f = recolor(frame, wp["hue_deg"], [1.0, 1.0, 1.0], wp["gamma"])
    f = (f - 128.0) * wp["contrast"] + 128.0
    f = f * wp["brightness"]
    g = f @ np.array([0.299, 0.587, 0.114], np.float32)
    f = g[..., None] + (f - g[..., None]) * wp["saturation"]
    return np.rint(np.clip(f, 0, 255)).astype(np.uint8)


def augment_frame(frame: np.ndarray, fg: np.ndarray, tex: np.ndarray, app: Appearance, cam: str, distract=None,
                  grow_px: int = 2, feather_px: int = 4, jitter: bool = True) -> np.ndarray:
    bg = background_layer(frame, tex, app, cam, distract)
    out = composite(frame, soft_alpha(fg, grow_px, feather_px), bg)
    return apply_jitter(out, app) if jitter else out


def apply_jitter(img: np.ndarray, app: Appearance) -> np.ndarray:
    return np.rint(global_jitter(img, app.jitter)).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------- per episode / camera
def _decode_frames(path: str, from_ts: float, n: int, fps: float):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = 1
        tb = float(s.time_base)
        if from_ts > 0:
            c.seek(int(max(from_ts - 0.5, 0.0) / tb), stream=s, backward=True, any_frame=False)
        last = None
        for fr in c.decode(s):
            idx = int(round((float(fr.pts) * tb - from_ts) * fps))
            if idx < 0:
                continue
            if last is not None and idx != last + 1:
                raise ValueError(f"{path}: frame numbering jumps {last} -> {idx}")
            if last is None and idx != 0:
                raise ValueError(f"{path}: first frame index {idx} != 0")
            last = idx
            if idx >= n:
                break
            yield idx, fr.to_ndarray(format="rgb24")
        if last is None or last + 1 < n:
            raise ValueError(f"{path}: only {0 if last is None else last + 1} of {n} frames")


class _Encoder:
    def __init__(self, path: Path, fps: float, crf: int, gop: int, n_frames: int):
        import av
        from lerobot.datasets.compute_stats import sample_indices
        from video_reencode import TB_SCALE
        self.av, self.TB, self.fps = av, TB_SCALE, int(round(fps))
        self.tmp = path.with_suffix(".tmp.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.out = av.open(str(self.tmp), "w", options={"movflags": "faststart"})
        self.st = self.out.add_stream("libx264", rate=self.fps)
        self.st.width, self.st.height, self.st.pix_fmt = W, H, "yuv420p"
        self.st.time_base = Fraction(1, self.fps * TB_SCALE)
        self.st.codec_context.time_base = self.st.time_base
        self.st.options = {"crf": str(crf), "g": str(gop), "bf": "0", "preset": "medium", "threads": "1"}
        self.n = 0
        self.sample_pos = set(sample_indices(n_frames))
        self.sampled = []

    def write(self, rgb: np.ndarray):
        from lerobot.datasets.compute_stats import auto_downsample_height_width
        if self.n in self.sample_pos:
            self.sampled.append(np.ascontiguousarray(auto_downsample_height_width(rgb.transpose(2, 0, 1))))
        f = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24").reformat(format="yuv420p")
        f.pts = self.n * self.TB
        f.time_base = self.st.time_base
        f.pict_type = self.av.video.frame.PictureType.NONE
        for p in self.st.encode(f):
            self.out.mux(p)
        self.n += 1

    def close(self):
        from video_reencode import image_stats
        for p in self.st.encode():
            self.out.mux(p)
        self.out.close()
        self.tmp.replace(self.path)
        return {"frames": self.n, "stats": image_stats(np.stack(self.sampled)), "bytes": self.path.stat().st_size}


def run_episode_cam(task: dict) -> dict:
    """One source episode x camera -> all its copies (decoded once).  Runs in a worker process."""
    cv2.setNumThreads(1)
    t0 = time.time()
    n, fps, cam = task["length"], task["fps"], task["cam"]
    labels = np.load(task["mask_path"])["labels"] if cam == "head" else None
    if labels is not None and (len(labels) != n or labels.shape[1:] != (H, W)):
        raise ValueError(f"{task['mask_path']}: mask shape {labels.shape} != ({n}, {H}, {W})")
    apps = [Appearance(**a) if not isinstance(a, Appearance) else a for a in task["appearances"]]
    texs = [make_texture(a) for a in apps] if cam == "head" else [None] * len(apps)
    fg_freq = (labels > 0).mean(0).astype(np.float32) if cam == "head" else None
    dist = [render_distractors(a, fg_freq) for a in apps] if cam == "head" else [None] * len(apps)
    static_area = float(task.get("static_robot_area", 0.0))
    encs = [_Encoder(Path(task["dst"][k]), fps, task["crf"], task["gop"], n) for k in range(len(apps))]
    unchanged = []
    try:
        for idx, frame in _decode_frames(task["src"], task["from_ts"], n, fps):
            if cam == "head":
                fg = union_fg(labels, idx)
                for k, app in enumerate(apps):
                    out = augment_frame(frame, fg, texs[k], app, cam, dist[k], task["grow_px"], task["feather_px"], task["jitter"])
                    encs[k].write(out)
            else:
                for k, app in enumerate(apps):
                    encs[k].write(wrist_photometric(frame, app.wrist_photo))
        res = [e.close() for e in encs]
    except BaseException:
        for e in encs:
            try:
                e.out.close()
            except Exception:
                pass
            e.tmp.unlink(missing_ok=True)
        raise
    return {"id": task["id"], "results": res, "unchanged_frames": unchanged, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------------------- dataset build
def _nested(v) -> np.ndarray:
    """pandas hands back parquet list columns as object arrays of arrays; rebuild the real (3, 1, 1) array."""
    if isinstance(v, np.ndarray) and v.dtype == object:
        return np.array([_nested(x) for x in v], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _stats_for_episode(rows: dict, features: dict) -> dict:
    from lerobot.datasets.compute_stats import compute_episode_stats
    return compute_episode_stats(rows, {k: features[k] for k in rows})


def build(args) -> dict:
    import datasets
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import get_hf_features_from_features, write_info, write_stats, write_tasks
    from video_reencode import pack_videos

    src, masks, out = Path(args.src), Path(args.masks), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first (not overwritten)")
    stage = Path(args.staging) / (out.name + ".staging")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    import tempfile
    tempfile.tempdir = str(stage)
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    vkeys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    ep_df = pd.concat([pd.read_parquet(p) for p in sorted((src / "meta" / "episodes").rglob("*.parquet"))],
                      ignore_index=True).sort_values("episode_index").reset_index(drop=True)
    data = pd.concat([pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet"))], ignore_index=True)
    src_map = json.loads((src / "meta" / "source_episodes.json").read_text())
    n_src = len(ep_df)
    sel = list(range(n_src)) if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    K = int(args.copies)
    names = {int(m["episode_index"]): m["source"] for m in src_map}
    t_start = time.time()

    # ---- appearances + encode tasks
    apps = {(e, k): sample_appearance(seed_for(args.seed, names[e], k), k) for e in sel for k in range(1, K + 1)}
    enc_dir = stage / "_enc"
    tasks, static_area = [], {}
    for cam_key in vkeys:
        cam = cam_key.split(".")[-1]
        sp = masks / "static_robot_mask_wrist_right.png"
        static_area[cam] = float((cv2.imread(str(sp), 0) > 0).sum()) if cam != "head" and sp.exists() else 0.0
        for e in sel:
            r = ep_df.iloc[e]
            vp = info["video_path"].format(video_key=cam_key, chunk_index=int(r[f"videos/{cam_key}/chunk_index"]),
                                           file_index=int(r[f"videos/{cam_key}/file_index"]))
            tasks.append({"id": (e, cam_key), "cam": cam, "length": int(r["length"]), "fps": fps, "src": str(src / vp),
                          "from_ts": float(r[f"videos/{cam_key}/from_timestamp"]),
                          "mask_path": str(masks / names[e] / cam / "labels.npz"),
                          "appearances": [asdict(apps[(e, k)]) for k in range(1, K + 1)],
                          "dst": [str(enc_dir / cam_key / f"c{k}_ep{e:03d}.mp4") for k in range(1, K + 1)],
                          "crf": args.crf, "gop": args.gop, "grow_px": args.grow_px, "feather_px": args.feather_px,
                          "jitter": not args.no_jitter, "static_robot_area": static_area[cam]})
    res = {}
    workers = max(1, min(int(args.workers), 4))
    print(f"augmenting {len(tasks)} episode/camera jobs x {K} copies, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futs = {ex.submit(run_episode_cam, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res[r["id"]] = r
            print(f"  [{i}/{len(tasks)}] ep{r['id'][0]:02d} {r['id'][1].split('.')[-1]}: {r['seconds']:.0f} s, "
                  f"unchanged frames {len(r['unchanged_frames'])}", flush=True)

    # ---- output episode order: originals first (copy 0), then copy 1 of all selected, copy 2, ...
    order = [(e, 0) for e in sel] + [(e, k) for k in range(1, K + 1) for e in sel]
    out_idx = {key: i for i, key in enumerate(order)}
    features = json.loads(json.dumps(info["features"]))
    features[COPY_KEY] = {"dtype": "int64", "shape": [1], "names": None}
    for k in vkeys:
        vi = dict(features[k].get("info", {}))
        vi.update({"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.crf": args.crf, "video.g": args.gop})
        features[k]["info"] = vi

    # ---- videos: originals copied as-is (same files / timestamps), copies packed after them
    vid_place = {}                       # (out_ep, key) -> (chunk, file, from, to)
    n_orig_files = {}
    for k in vkeys:
        used = sorted({(int(ep_df.iloc[e][f"videos/{k}/chunk_index"]), int(ep_df.iloc[e][f"videos/{k}/file_index"])) for e in sel})
        remap = {u: i for i, u in enumerate(used)}
        for u, i in remap.items():
            d = stage / info["video_path"].format(video_key=k, chunk_index=0, file_index=i)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / info["video_path"].format(video_key=k, chunk_index=u[0], file_index=u[1]), d)
        n_orig_files[k] = len(used)
        for e in sel:
            r = ep_df.iloc[e]
            c, f = 0, remap[(int(r[f"videos/{k}/chunk_index"]), int(r[f"videos/{k}/file_index"]))]
            vid_place[(out_idx[(e, 0)], k)] = (c, f, float(r[f"videos/{k}/from_timestamp"]), float(r[f"videos/{k}/to_timestamp"]))
        mine = [(e, kk) for kk in range(1, K + 1) for e in sel]
        placement = pack_videos([(enc_dir / k / f"c{kk}_ep{e:03d}.mp4", int(ep_df.iloc[e]["length"])) for e, kk in mine],
                                lambda c, f, k=k: stage / info["video_path"].format(video_key=k, chunk_index=c,
                                                                                    file_index=f + n_orig_files[k]),
                                info.get("video_files_size_in_mb", 200), fps, info.get("chunks_size", 1000))
        for (e, kk), pl in zip(mine, placement):
            vid_place[(out_idx[(e, kk)], k)] = (pl[0], pl[1] + n_orig_files[k], pl[2], pl[3])
    img_stats = {}
    for (e, k), r in res.items():
        for kk in range(1, K + 1):
            img_stats[(out_idx[(e, kk)], k)] = r["results"][kk - 1]["stats"]
    shutil.rmtree(enc_dir)

    # ---- data parquet, per-episode meta rows (non-image stats recomputed, image stats from source / encode)
    cols_keep = [c for c in data.columns]
    data_by_ep = {e: data[data["episode_index"] == e].reset_index(drop=True) for e in sel}
    data_parts, ep_rows, ep_stats_all, srcmap_out = [], [], [], []
    start = 0
    for (e, k) in order:
        oi = out_idx[(e, k)]
        d = data_by_ep[e].copy()
        n = len(d)
        d["episode_index"] = np.int64(oi)
        d["index"] = np.arange(start, start + n, dtype=np.int64)
        d[COPY_KEY] = np.int64(k)
        data_parts.append(d)
        arrs = {c: np.stack(d[c].to_numpy()) if d[c].dtype == object else d[c].to_numpy() for c in d.columns}
        arrs = {c: (v.astype(np.float32) if v.dtype == bool else v) for c, v in arrs.items()}
        st = _stats_for_episode({c: v for c, v in arrs.items() if c in features and features[c]["dtype"] != "video"}, features)
        srow = ep_df.iloc[e]
        for vk in vkeys:
            if k == 0:
                st[vk] = {s[len(f"stats/{vk}/"):]: _nested(srow[s]) for s in ep_df.columns if s.startswith(f"stats/{vk}/")}
            else:
                st[vk] = img_stats[(oi, vk)]
        row = {"episode_index": oi, "tasks": list(srow["tasks"]), "length": n, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": start, "dataset_to_index": start + n}
        for vk in vkeys:
            c, f, a, b = vid_place[(oi, vk)]
            row.update({f"videos/{vk}/chunk_index": c, f"videos/{vk}/file_index": f,
                        f"videos/{vk}/from_timestamp": a, f"videos/{vk}/to_timestamp": b})
        row.update({f"stats/{key}/{stat}": np.asarray(val).tolist() for key, dd in st.items() for stat, val in dd.items()})
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        ep_rows.append(row)
        ep_stats_all.append(st)
        sm_row = dict(next(m for m in src_map if int(m["episode_index"]) == e))
        srcmap_out.append({"episode_index": oi, "source_episode_index_in_input": e, "source": names[e], "copy_index": k,
                           "appearance": asdict(apps[(e, k)]) if k else None, "frames": n,
                           "unchanged_frames": {vk.split(".")[-1]: res[(e, vk)]["unchanged_frames"] for vk in vkeys} if k else {},
                           "input_source_record": sm_row})
        start += n
    full = pd.concat(data_parts, ignore_index=True)
    hf_features = get_hf_features_from_features({k: {**v, "shape": tuple(v["shape"])} for k, v in features.items()})
    ds = datasets.Dataset.from_dict({k: (np.stack(full[k].to_numpy()).tolist() if full[k].dtype == object else full[k].to_numpy())
                                     for k in hf_features}, features=hf_features, split="train")
    dp = stage / info["data_path"].format(chunk_index=0, file_index=0)
    dp.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(dp))
    epp = stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    epp.parent.mkdir(parents=True, exist_ok=True)
    for key in ep_rows[0]:
        kinds = {type(r.get(key)).__name__ + str(np.asarray(r.get(key), dtype=object).shape) if not isinstance(r.get(key), (str, int, float)) else type(r.get(key)).__name__ for r in ep_rows}
        if len(kinds) > 1:
            raise RuntimeError(f"episode meta column {key} has mixed types {kinds}")
    pq.write_table(pa.Table.from_pylist(ep_rows), epp)
    shutil.copy2(src / "meta" / "tasks.parquet", stage / "meta" / "tasks.parquet")
    garr = {c: (np.stack(full[c].to_numpy()) if full[c].dtype == object else full[c].to_numpy()) for c in full.columns}
    garr = {c: (v.astype(np.float32) if v.dtype == bool else v) for c, v in garr.items()}
    gstats = _stats_for_episode({c: v for c, v in garr.items() if c in features and features[c]["dtype"] != "video"}, features)
    gstats.update(aggregate_stats([{vk: s[vk] for vk in vkeys} for s in ep_stats_all]))
    write_stats(gstats, stage)
    new_info = dict(info)
    new_info.update({"features": features, "total_episodes": len(order), "total_frames": int(start),
                     "splits": {"train": f"0:{len(order)}"}})
    write_info(new_info, stage)
    (stage / "meta" / "source_episodes.json").write_text(json.dumps(srcmap_out, indent=2))
    n_unch = {vk.split(".")[-1]: int(sum(len(res[(e, vk)]["unchanged_frames"]) for e in sel)) * 1 for vk in vkeys}
    aug = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv), "tool": "tools/mocap_dataset/augment_bgswap.py",
        "source_dataset": str(src), "mask_dataset": str(masks), "output": str(out), "copies_per_episode": K, "seed": args.seed,
        "episode_layout": f"output episodes 0..{len(sel) - 1} = the source episodes (copy_index 0, videos stream-copied); then copy 1 of every "
                          f"source episode in source order, then copy 2, ... : output index = {len(sel)} * copy + source position",
        "augmentation.copy_index": "int64 per frame: 0 = original, k = k-th background-swapped copy",
        "evaluation_split_rule": "HELD-OUT EVALUATION MUST SPLIT BY SOURCE EPISODE (meta/source_episodes.json: 'source'). All copies of a "
                                 "held-out source episode belong to the held-out set and must never be trained on; training on a copy of a "
                                 "held-out episode leaks its state/action trajectory.",
        "held_out_source_episodes_at_creation": HELD_OUT,
        "parameters": {"grow_px": args.grow_px, "feather_px": args.feather_px, "head_global_color_jitter": not args.no_jitter,
                       "crf": args.crf, "gop": args.gop, "texture_kinds": list(TEX_KINDS), "max_distractors": 2,
                       "head": "bg = (1-t)*recolour(original) + t*procedural texture, t in [0.6,1]; static distractors in the bg layer",
                       "wrist": "global per-episode photometric change only (brightness/contrast/saturation x0.8-1.2, hue +-0.03 turn, gamma 0.85-1.15), "
                                "same parameters for every frame; no texture, no mask-based replacement (the carton close-ups are not "
                                "reliably masked), geometry untouched"},
        "unchanged_wrist_head_frames_all_copies_per_cam": n_unch,
        "unchanged_frames_note": "per-episode lists are in meta/source_episodes.json -> unchanged_frames",
        "copies_unmodified_columns": "observation.state, action, observation.state_raw, action_teleop, timestamp, frame_index, task_index, "
                                     "source_frame_index, source_timestamp are copied byte-for-byte from the source rows",
        "build_seconds": round(time.time() - t_start, 1),
    }
    (stage / "AUGMENT.json").write_text(json.dumps(aug, indent=2))
    stage.replace(out)
    return aug


def _worker_init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    try:
        os.nice(19)
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------------- validate / sheet
def validate(out_root: Path, src_root: Path) -> dict:
    import torch  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/bgswap", root=out_root, video_backend="pyav")
    src = LeRobotDataset("local/src", root=src_root, video_backend="pyav")
    smap = json.loads((out_root / "meta" / "source_episodes.json").read_text())
    n_src = sum(1 for m in smap if m["copy_index"] == 0)
    K = ds.meta.total_episodes // n_src - 1
    rep = {"episodes": ds.meta.total_episodes, "expected_episodes": n_src * (K + 1), "source_episodes_in_output": n_src, "frames": ds.meta.total_frames,
           "stats_keys": sorted(ds.meta.stats.keys()), "copies": K, "frames_match": True, "decoded": []}
    for e in range(ds.meta.total_episodes):
        a, b = ds.meta.episodes[e], src.meta.episodes[smap[e]["source_episode_index_in_input"]]
        if a["length"] != b["length"]:
            rep["frames_match"] = False
    rng = np.random.default_rng(0)
    for e in sorted({0, n_src, ds.meta.total_episodes - 1} | {int(x) for x in rng.integers(0, ds.meta.total_episodes, 5)}):
        a = ds.meta.episodes[e]
        for off in (0, a["length"] // 2, a["length"] - 1):
            item = ds[a["dataset_from_index"] + off]
            for vk in ("observation.images.head", "observation.images.wrist_right"):
                x = item[vk]
                rep["decoded"].append({"ep": e, "frame": off, "key": vk, "shape": list(x.shape), "mean": round(float(x.mean()), 4),
                                       "copy_index": int(item[COPY_KEY]), "episode_index": int(item["episode_index"])})
    # copy 0 must decode to exactly the source frames
    ident = []
    for e in range(n_src):
        a = ds.meta.episodes[e]
        sidx = smap[e]["source_episode_index_in_input"]
        b = src.meta.episodes[sidx]
        for off in (0, a["length"] // 2, a["length"] - 1):
            x, y = ds[a["dataset_from_index"] + off], src[b["dataset_from_index"] + off]
            for vk in ("observation.images.head", "observation.images.wrist_right"):
                ident.append(float((x[vk] - y[vk]).abs().max()))
            for ck in ("observation.state", "action"):
                ident.append(float((x[ck] - y[ck]).abs().max()))
    rep["copy0_max_abs_diff_vs_source"] = max(ident)
    # state/action of every copy equal the source rows
    sa = []
    for e in range(n_src, ds.meta.total_episodes):
        a = ds.meta.episodes[e]
        b = src.meta.episodes[smap[e]["source_episode_index_in_input"]]
        for off in (0, a["length"] - 1):
            x, y = ds[a["dataset_from_index"] + off], src[b["dataset_from_index"] + off]
            sa.append(max(float((x[ck] - y[ck]).abs().max()) for ck in ("observation.state", "action", "action_teleop")))
    rep["copies_state_action_max_abs_diff_vs_source"] = max(sa) if sa else None
    rep["ok"] = bool(rep["copy0_max_abs_diff_vs_source"] == 0.0 and (not sa or max(sa) == 0.0) and rep["episodes"] == rep["expected_episodes"] and rep["frames_match"] and
                     all(d["shape"] == [3, H, W] for d in rep["decoded"]))
    return rep


def contact_sheet(out_root: Path, dst: Path, picks=(0, 9, 16), fracs=(0.15, 0.5, 0.85)) -> None:
    import pandas as pd
    from video_frames import read_frames_by_index
    info = json.loads((out_root / "meta" / "info.json").read_text())
    ep = pd.read_parquet(out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").sort_values("episode_index")
    sm = json.loads((out_root / "meta" / "source_episodes.json").read_text())
    n_copies = max(m["copy_index"] for m in sm) + 1
    n_src = len(sm) // n_copies
    blocks = []
    for cam_key in ("observation.images.head", "observation.images.wrist_right"):
        rows = []
        for c in range(n_copies):
            cells = []
            for e in picks:
                r = ep.iloc[c * n_src + e]
                for fr in fracs:
                    idx = int(fr * (int(r["length"]) - 1))
                    p = out_root / info["video_path"].format(video_key=cam_key, chunk_index=int(r[f"videos/{cam_key}/chunk_index"]),
                                                              file_index=int(r[f"videos/{cam_key}/file_index"]))
                    img = read_frames_by_index(p, [idx], fps=float(info["fps"]), from_ts=float(r[f"videos/{cam_key}/from_timestamp"]))[idx]
                    img = cv2.resize(img, (320, 240))
                    cv2.putText(img, f"copy{c} src{e} f{idx}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cells.append(img)
            rows.append(np.concatenate(cells, 1))
        blocks.append(np.concatenate(rows, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    for name, b in zip(("head", "wrist_right"), blocks):
        cv2.imwrite(str(dst.with_name(dst.stem + f"_{name}.png")), b[:, :, ::-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", nargs="?")
    ap.add_argument("masks", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--episodes", default="all", help="comma list of source episode positions or 'all'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--gop", type=int, default=2)
    ap.add_argument("--grow-px", type=int, default=3)
    ap.add_argument("--feather-px", type=int, default=4)
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--staging", default="/media/farhan/My Passport/igus_nvidia/tmp_aug")
    ap.add_argument("--validate", nargs=2, metavar=("OUT_ROOT", "SRC_ROOT"))
    ap.add_argument("--sheet", nargs=2, metavar=("OUT_ROOT", "PNG"))
    a = ap.parse_args(argv)
    if a.validate:
        r = validate(Path(a.validate[0]), Path(a.validate[1]))
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    if a.sheet:
        contact_sheet(Path(a.sheet[0]), Path(a.sheet[1]))
        return 0
    if not (a.src and a.masks and a.out):
        ap.error("SRC MASKS --out are required")
    build(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
