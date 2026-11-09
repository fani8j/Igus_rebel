#!/usr/bin/env python3
"""Re-encode a chosen subset of the frames of an episode video (pause trimming) and pack episodes into LeRobot v3.0 files.

Why re-encode: a LeRobot v3.0 episode is one contiguous time range of a video file (``from_timestamp`` ..), and lerobot
checks that the ``timestamp`` column advances by exactly 1/fps.  Removing frames from the middle of an episode therefore
needs a new video in which the kept frames are consecutive; inter-frame codecs cannot drop frames by stream copy.

Quality: the source frames are decoded to yuv420p and handed to libx264 unchanged (no RGB round trip, no scaling), at a
constant quality (CRF, default 16) with a short GOP (default 2, like lerobot's own encoder) and no B-frames, so that
random access in training is cheap and the files can be joined by stream copy (lerobot's ``concatenate_video_files``).
The pixel difference to the source is reported by ``psnr_vs_source`` (about 44-45 dB at CRF 16 on this data).

Also computes the per-episode image statistics lerobot expects (``stats/<key>/min,max,mean,std,count,q01..q99``, shape
(3, 1, 1), scaled to [0, 1]) from the decoded source frames, with lerobot's own sampling and down-sampling helpers; the
source v2.1 dataset has no ``episodes_stats.jsonl`` to take them from.

Throttle: callers run this under ``nice -n 19`` with at most 4 worker processes, each using one decode and one encode thread.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

DEFAULT_CRF = 16
DEFAULT_GOP = 2
TB_SCALE = 512                      # time base 1/(fps * 512), like the source (1/15360 at 30 fps)


def image_stats(frames_chw: np.ndarray) -> dict:
    """lerobot's per-episode image statistics of a (S, 3, h, w) uint8 sample (``compute_episode_stats`` for video features)."""
    from lerobot.datasets.compute_stats import get_feature_stats
    st = get_feature_stats(frames_chw, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()}


def reencode_episode(src_path: str, src_from_ts: float, keep: list[int], dst_path: str, fps: float,
                     crf: int = DEFAULT_CRF, gop: int = DEFAULT_GOP, threads: int = 1, preset: str = "medium") -> dict:
    """Write frames ``keep`` (indices relative to the episode start, ascending) of the source video as a new mp4.

    Returns ``{"frames", "stats", "seconds", "bytes"}``.  Raises if a source frame is missing or the decoded frame
    numbering (from pts) has a gap, i.e. if the video is not a plain constant-rate stream."""
    from lerobot.datasets.compute_stats import auto_downsample_height_width, sample_indices
    keep = np.asarray(keep, dtype=np.int64)
    if len(keep) == 0 or np.any(np.diff(keep) <= 0):
        raise ValueError("keep must be a non-empty ascending index list")
    want = set(keep.tolist())
    sample_pos = set(sample_indices(len(keep)))                 # positions inside `keep` that go into the image stats
    sampled, t0 = [], time.time()
    out = None
    n_out, last_idx = 0, None
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    try:
        with av.open(str(src_path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            s.codec_context.thread_count = threads
            tb = float(s.time_base)
            if src_from_ts > 0:
                c.seek(int(max(src_from_ts - 0.5, 0.0) / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                idx = int(round((float(fr.pts) * tb - src_from_ts) * fps))
                if idx < 0:
                    continue
                if last_idx is not None and idx != last_idx + 1:
                    raise ValueError(f"{src_path}: decoded frame numbering jumps {last_idx} -> {idx}")
                if last_idx is None and idx != 0:
                    raise ValueError(f"{src_path}: first frame of the episode has index {idx}, not 0")
                last_idx = idx
                if idx > keep[-1]:
                    break
                if idx not in want:
                    continue
                if out is None:
                    out = av.open(str(tmp), "w", options={"movflags": "faststart"})
                    st = out.add_stream("libx264", rate=int(round(fps)))
                    st.width, st.height, st.pix_fmt = fr.width, fr.height, "yuv420p"
                    st.time_base = Fraction(1, int(round(fps)) * TB_SCALE)
                    st.codec_context.time_base = st.time_base
                    st.options = {"crf": str(crf), "g": str(gop), "bf": "0", "preset": preset, "threads": str(threads)}
                if n_out in sample_pos:
                    img = auto_downsample_height_width(fr.to_ndarray(format="rgb24").transpose(2, 0, 1))
                    sampled.append(np.ascontiguousarray(img))
                f = fr.reformat(format="yuv420p")
                f.pts = n_out * TB_SCALE
                f.time_base = st.time_base
                f.pict_type = av.video.frame.PictureType.NONE
                for p in st.encode(f):
                    out.mux(p)
                n_out += 1
        if n_out != len(keep):
            raise ValueError(f"{src_path}: wrote {n_out} frames, expected {len(keep)} (video shorter than the episode?)")
        for p in st.encode():
            out.mux(p)
        out.close()
        out = None
        tmp.replace(dst)
    finally:
        if out is not None:
            out.close()
        tmp.unlink(missing_ok=True)
    return {"frames": n_out, "stats": image_stats(np.stack(sampled)), "seconds": time.time() - t0,
            "bytes": dst.stat().st_size}


def reencode_all(tasks: list[dict], workers: int = 4, log=print) -> dict:
    """Run ``reencode_episode`` for every task dict (keys: id, src, from_ts, keep, dst, fps, crf, gop) in at most
    ``workers`` (<= 4) processes.  Returns {id: result}."""
    workers = max(1, min(int(workers), 4))
    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(reencode_episode, t["src"], t["from_ts"], t["keep"], t["dst"], t["fps"], t["crf"], t["gop"]): t
                for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            res[t["id"]] = f.result()            # raises with the worker's error
            done += 1
            log(f"  encoded {done}/{len(tasks)} {t['id']}: {res[t['id']]['frames']} frames, "
                f"{res[t['id']]['bytes'] / 2**20:.0f} MiB, {res[t['id']]['seconds']:.0f} s", flush=True)
    return res


def pack_videos(per_episode: list[tuple[Path, int]], dst_for_file, max_file_mb: float, fps: float, chunk_size: int = 1000):
    """Join consecutive per-episode mp4s into files of at most ``max_file_mb`` (stream copy, lerobot's
    ``concatenate_video_files``) and return, per episode, ``(chunk_index, file_index, from_timestamp, to_timestamp)``.

    ``per_episode``: [(mp4 path, frames)] in episode order.  ``dst_for_file(chunk, file) -> Path``.  from/to timestamps use
    lerobot's own duration probe, the way ``aggregate_videos`` accumulates offsets; the probe must agree with frames / fps."""
    from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
    out, groups, cur, cur_mb = [], [], [], 0.0
    for i, (p, n) in enumerate(per_episode):
        mb = Path(p).stat().st_size / 2 ** 20
        if cur and cur_mb + mb > max_file_mb:
            groups.append(cur)
            cur, cur_mb = [], 0.0
        cur.append(i)
        cur_mb += mb
    if cur:
        groups.append(cur)
    placement = [None] * len(per_episode)
    for g, members in enumerate(groups):
        chunk, file = divmod(g, chunk_size)
        dst = Path(dst_for_file(chunk, file))
        dst.parent.mkdir(parents=True, exist_ok=True)
        paths = [Path(per_episode[i][0]) for i in members]
        if len(paths) == 1:
            shutil.copy(str(paths[0]), str(dst))
        else:
            concatenate_video_files(paths, dst)
        t = 0.0
        for i in members:
            d = get_video_duration_in_s(per_episode[i][0])
            if abs(d - per_episode[i][1] / fps) > 0.5 / fps:
                raise ValueError(f"{per_episode[i][0]}: duration {d:.4f} s != {per_episode[i][1]} frames / {fps:g} fps")
            placement[i] = (chunk, file, t, t + per_episode[i][1] / fps)
            t += per_episode[i][1] / fps
    return placement


def psnr_vs_source(out_frame: np.ndarray, src_frame: np.ndarray) -> float:
    """PSNR [dB] between two RGB uint8 frames (inf for identical frames)."""
    mse = float(np.mean((out_frame.astype(np.float64) - src_frame.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
