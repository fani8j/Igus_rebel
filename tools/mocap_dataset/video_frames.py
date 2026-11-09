#!/usr/bin/env python3
"""Decode single frames of the (AV1) mp4s with PyAV: thumbnails, and the frame-alignment check of preprocess.py.

The videos are untrusted input but only decoded, never executed.  Decoding is throttled to 2 threads."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def count_packets(path: str | Path) -> int:
    """Number of video packets (= frames) without decoding."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return sum(1 for p in c.demux(s) if p.size > 0)


def read_frames(path: str | Path, timestamps_s: list[float], fps: float = 30.0, threads: int = 2) -> list[np.ndarray]:
    """RGB uint8 (H, W, 3) frames at the given times [s] of the file (nearest frame at or after t - half a frame).

    Seeks to the keyframe before each time and decodes forward, so cost is bounded by the keyframe interval."""
    out: list[np.ndarray | None] = [None] * len(timestamps_s)
    order = sorted(range(len(timestamps_s)), key=lambda i: timestamps_s[i])
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = threads
        tb = float(s.time_base)
        for i in order:
            t = float(timestamps_s[i])
            c.seek(int(max(t - 0.0, 0.0) / tb), stream=s, backward=True, any_frame=False)
            found = None
            for fr in c.decode(s):
                ft = float(fr.pts) * tb
                if ft >= t - 0.5 / fps:
                    found = fr
                    break
            if found is None:
                raise ValueError(f"{path}: no frame at t={t:.3f}s")
            out[i] = found.to_ndarray(format="rgb24")
    return out  # type: ignore[return-value]


def read_frames_by_index(path: str | Path, indices: list[int], fps: float = 30.0, from_ts: float = 0.0,
                         threads: int = 2) -> dict[int, np.ndarray]:
    """RGB uint8 frames {index: (H, W, 3)} for the given frame numbers of an episode inside ``path`` (frame k is the one
    at ``from_ts + k / fps``), in ONE sequential decode pass (cheap for many nearby frames, e.g. a window around a cut)."""
    want = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = threads
        tb = float(s.time_base)
        if from_ts > 0 or want[0] > 60:
            t = from_ts + max(want[0] - 1, 0) / fps
            c.seek(int(max(t - 0.0, 0.0) / tb), stream=s, backward=True, any_frame=False)
        remaining = set(want)
        for fr in c.decode(s):
            k = int(round((float(fr.pts) * tb - from_ts) * fps))
            if k in remaining:
                out[k] = fr.to_ndarray(format="rgb24")
                remaining.discard(k)
                if not remaining:
                    break
            elif k > want[-1]:
                break
    missing = set(want) - set(out)
    if missing:
        raise ValueError(f"{path}: frames {sorted(missing)[:5]} not found")
    return out


def motion_energy(path: str | Path, size: tuple[int, int] = (64, 36), threads: int = 2) -> dict:
    """Decode a whole video once and reduce every frame to a ``size`` grey thumbnail.

    Returns ``{"motion": (n,) mean abs grey difference to the previous frame (entry 0 is 0), "first": (h, w) uint8,
    "last": (h, w) uint8, "n": n}``.  Used by quality_report.py to compare the image motion of a camera with the
    joint speed.  Decoding is throttled to ``threads`` threads."""
    prev = None
    motion, first, last = [], None, None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = threads
        for fr in c.decode(s):
            g = fr.reformat(width=size[0], height=size[1], format="gray").to_ndarray()
            if g.ndim == 3:
                g = g[..., 0]
            motion.append(0.0 if prev is None else float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()))
            prev = g
            if first is None:
                first = g
        last = prev
    return {"motion": np.asarray(motion), "first": first, "last": last, "n": len(motion)}
