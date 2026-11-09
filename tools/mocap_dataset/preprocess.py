#!/usr/bin/env python3
"""Merge recorded episodes into ONE clean LeRobot v3.0 dataset (format-neutral, for VLA training).

    python3 tools/mocap_dataset/preprocess.py SOURCE_ROOT --out OUTDIR [--include 13,new_1 | --exclude new_9] \
        [--sigma-s 0.08] [--action-offset-frames 1] [--trim-idle] [--trim-pauses] [--task "open the box flaps"] \
        [--left-only | --arm arm_right [--arm-only]] [--force]

SOURCE_ROOT is a folder of one-episode LeRobot v3.0 datasets (the left-arm mocap recordings, 14-D) or ONE dataset with
many episodes, v3.0 or v2.1 (e.g. ``real_right``, 6-D right arm; v2.1 is read in place, see lerobot_episode.py).

Per episode (source order = natural sort of the episode names; the mapping is printed and saved):
  * ``observation.state``      = jitter-filtered measured state: zero-phase endpoint-preserving Gaussian
                                 (``episode_metrics.smooth`` = ``replay_episode.gaussian_smooth``), joints only.
                                 ``--sigma-s 0`` keeps the measured state.
  * ``action[t]``              = filtered state at t + ``--action-offset-frames`` of the KEPT frames (last frames repeat
                                 the final row), gripper slots 0.  Offset 1 = "the pose the arm reaches next frame": an
                                 absolute joint-position target that is always reachable and is exactly what the images
                                 show one frame later.  After pause trimming "next" is the next kept frame.
  * ``observation.state_raw`` / ``action_teleop`` (``--orig-action-name``) = the untouched original columns (traceability).
  * ``source_frame_index`` / ``source_timestamp`` = where every kept frame came from in the source episode.
  * ``next.reward`` / ``next.done`` (``--keep-columns``) are carried when the source has them; other extra source columns
    (e.g. the GR00T ``annotation.human.task_description``) are dropped and listed in PREPROCESS.json.
  * videos                     = byte-for-byte stream copy (lerobot ``aggregate_videos``: concat demuxer, no re-encode) when
                                 the source is a set of one-episode v3.0 datasets and no frame is removed from the middle;
                                 otherwise (v2.1 source, ``--trim-pauses``, ``--reencode-video``) the kept frames are
                                 re-encoded with libx264 (see video_reencode.py: CRF 16, GOP 2, same pixels in, yuv420p).
  * options (default off): ``--trim-idle`` cuts the still head/tail (0.5 s margin kept); ``--trim-pauses`` shortens every
    mid-episode pause (still run >= ``--pause-min-s`` = 1.0 s) to ``--pause-keep-s`` = 0.4 s of its settled part
    (episode_metrics.plan_pause_trim: only frames at which the arm is practically not moving are removed, the settling
    tail stays, timestamps stay contiguous at fps); ``--task`` unifies the
    task string; ``--left-only`` writes 6-D vectors of the left arm (``--arm A --arm-only`` any arm).
Also writes ``meta/source_episodes.json`` (source -> episode_index, per-episode source metadata such as ``box_pose``, kept
frame ranges) and ``PREPROCESS.json`` (command, parameters, per-episode deviation / jitter / pause-trim numbers).  The
source is never modified.  OUTDIR must not exist (``--force`` replaces an existing lerobot dataset or an empty directory only).

lerobot 0.4.4 pieces used: ``aggregate_videos`` / ``concatenate_video_files`` (video concat without re-encode),
``compute_episode_stats`` / ``aggregate_stats`` (stats incl. q01..q99), ``get_hf_features_from_features`` (parquet schema),
``write_info/stats/tasks``.  Not used: ``aggregate_datasets`` / ``merge_datasets`` / ``add_features`` (they cannot replace the
state/action columns or trim, and ``add_features`` rewrites everything) and the v2.1 -> v3.0 converter (needs
``episodes_stats.jsonl``, which real_right does not have), see README.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from episode_metrics import (DEFAULT_SIGMA_S, PAUSE_CORE_SPEED, PAUSE_KEEP_S, PAUSE_MAX_JOIN_DEG, PAUSE_MIN_S,  # noqa: E402
                             STILL_SPEED,
                             arm_columns, deviation_deg, filter_state, jitter_metrics, kept_ranges,
                             motion_segments, plan_pause_trim, shifted_action, smooth, summarize_jitter, trim_bounds)
from lerobot_episode import Episode, EpisodeRef, list_episodes, load_episode  # noqa: E402

RAW_STATE = "observation.state_raw"
TELEOP_ACTION = "action_teleop"
VIDEO_PREFIX = "observation.images."
SCALAR_KEYS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
SRC_INDEX, SRC_TIME = "source_frame_index", "source_timestamp"
DEFAULT_KEEP_COLUMNS = "next.reward,next.done"
DEFAULT_CARRY_META = "box_pose"
V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class PreprocessError(ValueError):
    pass


# ----------------------------------------------------------------------------------------- selection
def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def select_datasets(dirs: list, include: str | None, exclude: str | None) -> list:
    """Natural-sorted subset of ``dirs`` (paths, or anything with ``.name`` and optionally ``.aliases``).

    Names may be given in full, without the ``mocap_left_`` prefix, or (episodes of one multi-episode dataset) as the
    episode index, e.g. ``19`` or ``episode_000019``."""
    if include and exclude:
        raise PreprocessError("--include and --exclude are mutually exclusive")
    by_name = {d.name: d for d in dirs}
    by_alias = {a: d.name for d in dirs for a in getattr(d, "aliases", ())}

    def resolve(tok: str) -> str:
        tok = tok.strip()
        for cand in (tok, f"mocap_left_{tok}"):
            if cand in by_name:
                return cand
        if tok in by_alias:
            return by_alias[tok]
        raise PreprocessError(f"unknown episode dataset '{tok}'; available: {', '.join(sorted(by_name, key=natural_key))}")

    names = list(by_name)
    if include:
        names = [resolve(t) for t in include.split(",") if t.strip()]
    elif exclude:
        drop = {resolve(t) for t in exclude.split(",") if t.strip()}
        names = [n for n in names if n not in drop]
    names = sorted(set(names), key=natural_key)
    if not names:
        raise PreprocessError("no episode left after --include/--exclude")
    return [by_name[n] for n in names]


# ------------------------------------------------------------------------------------ per-episode data
def process_episode(ep: Episode, *, sigma_s: float, action_offset: int, trim_idle: bool, trim_margin_s: float,
                    arm: str, arm_only: bool, trim_pauses: bool = False, pause_min_s: float = PAUSE_MIN_S,
                    pause_keep_s: float = PAUSE_KEEP_S, pause_still_speed: float = STILL_SPEED,
                    pause_max_join_deg: float = PAUSE_MAX_JOIN_DEG, pause_core_speed: float = PAUSE_CORE_SPEED,
                    keep_columns: tuple = ()) -> dict:
    """Filtered state, shifted action, kept frames and the numbers for PREPROCESS.json (float64 until the end).

    ``keep`` = source frame index of every output frame (ascending).  Idle trimming selects a contiguous range, pause
    trimming removes runs from the middle; the action is computed on the pause-trimmed sequence of the whole episode and
    then cut to the idle range, so without ``trim_pauses`` the result is exactly the old (idle-trim only) one."""
    names = ep.names
    fps = ep.fps
    n = len(ep.timestamp)
    filt = filter_state(ep.state, names, sigma_s, fps)
    acols = arm_columns(names, arm)
    qraw = ep.state[:, acols]
    qf = filt[:, acols]
    seg_q = qf if sigma_s > 0 else smooth(qraw, DEFAULT_SIGMA_S, fps)       # segmentation never sees the stale-sample ripple
    seg = motion_segments(seg_q, fps)
    i0, i1 = trim_bounds(seg, n, fps, trim_margin_s) if trim_idle else (0, n)
    plan = plan_pause_trim(seg_q, fps, pause_still_speed, pause_min_s, pause_keep_s, pause_max_join_deg,
                           pause_core_speed) if trim_pauses else None
    kf = plan["keep"] if plan else np.arange(n)
    act_kf = shifted_action(filt[kf], action_offset, names)
    pos = np.flatnonzero((kf >= i0) & (kf < i1))
    keep = kf[pos]
    cols_out = acols if arm_only else list(range(len(names)))
    # how the original action relates to the filtered state (for choosing --action-offset-frames)
    ta = ep.action[:, acols]
    lags = {L: float(np.degrees(np.sqrt(np.mean((qf[L:] - ta[:n - L]) ** 2)))) for L in range(0, 21)}
    best = min(lags, key=lags.get)
    jb, ja = jitter_metrics(qraw, fps), jitter_metrics(qf, fps)
    max_step_in = float(np.degrees(np.abs(np.diff(qf, axis=0)).max())) if n > 1 else 0.0
    max_step_out = float(np.degrees(np.abs(np.diff(filt[keep][:, acols], axis=0)).max())) if len(keep) > 1 else 0.0
    return {
        "state": filt[keep][:, cols_out].astype(np.float32),
        "action": act_kf[pos][:, cols_out].astype(np.float32),
        "state_raw": ep.state[keep][:, cols_out].astype(np.float32),
        "action_teleop": ep.action[keep][:, cols_out].astype(np.float32),
        "extras": {k: ep.extras[k][keep] for k in keep_columns if k in ep.extras},
        "keep": keep, "source_timestamp": ep.timestamp[keep].astype(np.float32),
        "trim": [int(i0), int(i1)], "frames_source": n,
        "info": {
            "deviation": deviation_deg(qraw, qf),
            "jitter_before": jb, "jitter_after": ja,
            "jitter_before_mean": summarize_jitter(jb), "jitter_after_mean": summarize_jitter(ja),
            "start_still_s": seg["start_still_s"], "end_still_s": seg["end_still_s"],
            "teleop_lead_frames": best, "teleop_rms_deg_at_best_lag": lags[best],
            "teleop_rms_deg_at_chosen_offset": lags.get(int(action_offset)),
            "arm": arm,
            "keep_ranges": kept_ranges(keep),
            "pause_trim": None if plan is None else {
                "cuts": plan["cuts"], "skipped": plan["skipped"], "removed_frames": plan["removed"],
                "removed_s": round(plan["removed"] / fps, 3), "head_frames": plan["head_frames"], "tail_frames": plan["tail_frames"],
                "max_join_step_deg": max([c["join_step_deg"] for c in plan["cuts"]], default=0.0),
                "max_step_deg_before": max_step_in, "max_step_deg_after": max_step_out},
        },
    }


# --------------------------------------------------------------------------------------------- stats
def _source_stats_row(root: Path, episode_index: int) -> dict:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    rows = [r for f in files for r in pq.read_table(f).to_pylist() if r["episode_index"] == episode_index]
    if not rows:
        raise PreprocessError(f"{root}: no meta row for episode {episode_index}")
    return rows[0]


def image_stats_from_source(row: dict, video_keys: list[str]) -> dict:
    out = {}
    for key in video_keys:
        pre = f"stats/{key}/"
        out[key] = {k[len(pre):]: np.asarray(v) for k, v in row.items() if k.startswith(pre)}
        if not out[key]:
            raise PreprocessError(f"source episode meta has no image stats for {key}")
    return out


def flat_stats(stats: dict) -> dict:
    return {f"stats/{key}/{stat}": np.asarray(val).tolist() for key, d in stats.items() for stat, val in d.items()}


def as_stats_array(a: np.ndarray) -> np.ndarray:
    """lerobot computes statistics on numbers; bool columns (next.done) are summarised as 0/1."""
    return a.astype(np.float32) if a.dtype == bool else a


# ----------------------------------------------------------------------------------------- the build
def v30_info(base: dict) -> dict:
    """info.json of the output from a v3.0 or v2.1 source info (what lerobot's v2.1 converter does to it)."""
    info = dict(base)
    info["codebase_version"] = "v3.0"
    for k in ("total_chunks", "total_videos"):
        info.pop(k, None)
    info.setdefault("chunks_size", 1000)
    info.setdefault("data_files_size_in_mb", 100)
    info.setdefault("video_files_size_in_mb", 200)
    info["data_path"] = V30_DATA_PATH
    info["video_path"] = V30_VIDEO_PATH
    info["fps"] = int(base["fps"])
    return info


def build(args, refs: list[EpisodeRef], out_final: Path, staging: Path) -> dict:
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.utils import (get_hf_features_from_features, write_info, write_stats, write_tasks)
    import datasets
    import pandas as pd

    infos = {r.root: json.loads((r.root / "meta" / "info.json").read_text()) for r in refs}
    roots = list(infos)
    base = infos[roots[0]]
    for r, i in infos.items():
        for k in ("fps", "robot_type", "features"):
            if i.get(k) != base.get(k):
                raise PreprocessError(f"{r.name}: '{k}' differs from {roots[0].name}; cannot merge")
        if str(i.get("codebase_version", ""))[:2] != str(base.get("codebase_version", ""))[:2]:
            raise PreprocessError(f"{r.name}: codebase_version differs from {roots[0].name}; cannot merge")
    src_v2 = str(base["codebase_version"]).startswith("v2")
    if not src_v2 and not str(base["codebase_version"]).startswith("v3"):
        raise PreprocessError(f"expected LeRobot v2.1 or v3.x sources, got {base['codebase_version']}")
    fps = float(base["fps"])
    names_src = list(base["features"]["observation.state"]["names"])
    video_keys = [k for k, f in base["features"].items() if f["dtype"] == "video"]
    keep_columns = tuple(c.strip() for c in args.keep_columns.split(",") if c.strip())
    carry_meta = tuple(c.strip() for c in args.carry_meta.split(",") if c.strip())

    srcs = [load_episode(r.root, r.episode_index) for r in refs]
    arm = args.arm or Counter(ep.active_arm() for ep in srcs).most_common(1)[0][0]
    arm_idx = arm_columns(names_src, arm)
    arm_only = bool(args.arm_only)
    out_names = [names_src[i] for i in arm_idx] if arm_only else names_src

    # ---- features of the output
    vec = lambda: {"dtype": "float32", "shape": [len(out_names)], "names": list(out_names)}  # noqa: E731
    dropped = [c for c in base["features"] if c not in {"observation.state", "action", *SCALAR_KEYS, *keep_columns, *video_keys}]
    ordered = {"observation.state": vec(), "action": vec(), RAW_STATE: vec(), args.orig_action_name: vec()}
    for k, v in base["features"].items():
        if k in ordered or k in dropped:
            continue
        ordered[k] = dict(v)
    ordered[SRC_INDEX] = {"dtype": "int64", "shape": [1], "names": None}
    ordered[SRC_TIME] = {"dtype": "float32", "shape": [1], "names": None}
    features = ordered
    if src_v2:                                   # the v3.0 writer (and lerobot's converter) put the dataset fps on every non-video feature
        for k, v in features.items():
            if v["dtype"] != "video":
                v.setdefault("fps", int(fps))

    tempfile.tempdir = str(staging)           # lerobot's video concat writes temp files here (not on /)
    staging.mkdir(parents=True)
    t0 = time.time()

    # ---- per episode: arrays
    proc = []
    for r, ep in zip(refs, srcs):
        p = process_episode(ep, sigma_s=args.sigma_s, action_offset=args.action_offset_frames, trim_idle=args.trim_idle,
                            trim_margin_s=args.trim_margin_s, arm=arm, arm_only=arm_only, trim_pauses=args.trim_pauses,
                            pause_min_s=args.pause_min_s, pause_keep_s=args.pause_keep_s,
                            pause_still_speed=args.pause_still_speed, pause_max_join_deg=args.pause_max_join_deg,
                            pause_core_speed=args.pause_core_speed, keep_columns=keep_columns)
        proc.append(p)
        pt = p["info"]["pause_trim"]
        print(f"  [{len(proc):2d}/{len(refs)}] {r.name}: {p['frames_source']} -> {len(p['keep'])} frames"
              + (f" (pauses -{pt['removed_frames']}, {len(pt['cuts'])} cuts, max join {pt['max_join_step_deg']:.3f} deg)" if pt else "")
              + f", max filter dev {p['info']['deviation']['max_deg_all']:.3f} deg", flush=True)

    # ---- videos: stream copy when nothing is cut out of the middle of a one-episode v3.0 dataset, else re-encode
    contiguous = all(len(p["keep"]) == 0 or p["keep"][-1] - p["keep"][0] + 1 == len(p["keep"]) for p in proc)
    copy_ok = (not src_v2 and all(r.n_in_dataset == 1 for r in refs) and contiguous and not args.reencode_video)
    video_mode = "copy" if copy_ok else "reencode"
    video_meta = [dict() for _ in refs]                   # per episode {key: (chunk, file, from, to)}
    enc_stats = [dict() for _ in refs]                    # per episode {key: image stats} (re-encode mode)
    enc_info = {}
    vfile_mb = base.get("video_files_size_in_mb", 200)
    chunk_size = base.get("chunks_size", 1000)
    if video_mode == "copy":
        from lerobot.datasets.aggregate import aggregate_videos
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        videos_idx = {k: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for k in video_keys}
        dst_shim = SimpleNamespace(root=staging)
        for e, (r, p) in enumerate(zip(refs, proc)):
            src_meta = LeRobotDatasetMetadata(f"local/{r.name}", root=r.root)
            videos_idx = aggregate_videos(src_meta, dst_shim, videos_idx, vfile_mb, chunk_size)
            row = _source_stats_row(r.root, r.episode_index)
            for k in video_keys:
                key_src = (row[f"videos/{k}/chunk_index"], row[f"videos/{k}/file_index"])
                dc, df = videos_idx[k]["src_to_dst"][key_src]
                off = videos_idx[k]["src_to_offset"][key_src]
                start = off + row[f"videos/{k}/from_timestamp"] + p["trim"][0] / fps
                end = off + row[f"videos/{k}/from_timestamp"] + p["trim"][1] / fps
                video_meta[e][k] = (int(dc), int(df), float(start), float(end))
    else:
        from video_reencode import pack_videos, reencode_all
        encdir = staging / "_enc"
        tasks = [{"id": (e, k), "src": str(srcs[e].videos[k][0]), "from_ts": srcs[e].videos[k][1],
                  "keep": proc[e]["keep"].tolist(), "dst": str(encdir / k / f"ep{e:06d}.mp4"), "fps": fps,
                  "crf": args.video_crf, "gop": args.video_gop} for e in range(len(refs)) for k in video_keys]
        print(f"re-encoding {len(tasks)} videos (libx264 crf {args.video_crf}, g {args.video_gop}, {args.workers} workers)", flush=True)
        res = reencode_all(tasks, args.workers)
        for k in video_keys:
            placement = pack_videos([(encdir / k / f"ep{e:06d}.mp4", len(proc[e]["keep"])) for e in range(len(refs))],
                                    lambda c, f, k=k: staging / V30_VIDEO_PATH.format(video_key=k, chunk_index=c, file_index=f),
                                    vfile_mb, fps, chunk_size)
            for e in range(len(refs)):
                video_meta[e][k] = tuple(placement[e])
                enc_stats[e][k] = res[(e, k)]["stats"]
        shutil.rmtree(encdir)
        enc_info = {"codec": "libx264", "crf": args.video_crf, "gop": args.video_gop, "bframes": 0, "pix_fmt": "yuv420p",
                    "seconds": {f"{refs[e].name}/{k.split('.')[-1]}": round(res[(e, k)]["seconds"], 1) for (e, k) in res}}
        for k in video_keys:
            vi = dict(features[k].get("info", {}))
            vi.update({"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.crf": args.video_crf, "video.g": args.video_gop})
            features[k]["info"] = vi

    # ---- tasks
    tasks_l = []
    for ep in srcs:
        t = args.task or ep.task
        if t not in tasks_l:
            tasks_l.append(t)
    tasks_df = pd.DataFrame({"task_index": range(len(tasks_l))}, index=pd.Index(tasks_l, name=None))

    # ---- data parquet + episode metadata
    lengths = [len(p["keep"]) for p in proc]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    vec_keys = ["observation.state", "action", RAW_STATE, args.orig_action_name]
    extra_keys = [k for k in keep_columns if k in features]
    cols = {k: [] for k in (*vec_keys, *extra_keys, *SCALAR_KEYS, SRC_INDEX, SRC_TIME)}
    ep_rows, ep_stats_list = [], []
    for e, (ep, p) in enumerate(zip(srcs, proc)):
        n = lengths[e]
        task_idx = tasks_l.index(args.task or ep.task)
        scal = {"timestamp": (np.arange(n) / fps).astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, e, dtype=np.int64), "index": np.arange(starts[e], starts[e] + n, dtype=np.int64),
                "task_index": np.full(n, task_idx, dtype=np.int64),
                SRC_INDEX: p["keep"].astype(np.int64), SRC_TIME: p["source_timestamp"]}
        data_e = {"observation.state": p["state"], "action": p["action"], RAW_STATE: p["state_raw"],
                  args.orig_action_name: p["action_teleop"], **{k: p["extras"][k] for k in extra_keys}, **scal}
        for k, v in data_e.items():
            cols[k].append(v)
        st = compute_episode_stats({k: as_stats_array(v) for k, v in data_e.items()}, {k: features[k] for k in data_e})
        st.update(image_stats_from_source(_source_stats_row(refs[e].root, refs[e].episode_index), video_keys)
                  if video_mode == "copy" else enc_stats[e])
        ep_stats_list.append(st)
        row = {"episode_index": e, "tasks": [tasks_l[task_idx]], "length": n, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": int(starts[e]), "dataset_to_index": int(starts[e] + n)}
        for k in video_keys:
            c, f, a, b = video_meta[e][k]
            row.update({f"videos/{k}/chunk_index": c, f"videos/{k}/file_index": f,
                        f"videos/{k}/from_timestamp": a, f"videos/{k}/to_timestamp": b})
        for mk in carry_meta:
            if mk in ep.meta:
                row[mk] = ep.meta[mk]
        row.update(flat_stats(st))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        ep_rows.append(row)

    full = {k: np.concatenate(v) for k, v in cols.items()}
    hf_features = get_hf_features_from_features({k: {**v, "shape": tuple(v["shape"])} for k, v in features.items()})
    ds = datasets.Dataset.from_dict({k: (full[k].tolist() if full[k].ndim == 2 else full[k]) for k in hf_features},
                                    features=hf_features, split="train")
    data_path = staging / V30_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(data_path))
    if data_path.stat().st_size / 2 ** 20 > base.get("data_files_size_in_mb", 100):
        raise PreprocessError("data parquet exceeds data_files_size_in_mb; multi-file data is not implemented")

    import pyarrow as pa
    ep_path = staging / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), ep_path)
    write_tasks(tasks_df, staging)

    # global stats: exact for vector/scalar features (lerobot's own routine on the concatenation),
    # images = lerobot's count-weighted aggregation of the per-episode image stats
    gstats = compute_episode_stats({k: as_stats_array(v) for k, v in full.items()}, {k: features[k] for k in full})
    gstats.update(aggregate_stats([{k: s[k] for k in video_keys} for s in ep_stats_list]))
    write_stats(gstats, staging)

    info = v30_info(base)
    info.update({"features": features, "total_episodes": len(refs), "total_frames": int(starts[-1]),
                 "total_tasks": len(tasks_l), "splits": {"train": f"0:{len(refs)}"}})
    write_info(info, staging)

    # ---- sidecar + PREPROCESS.json
    src_map = [{"episode_index": e, "source": r.name, "source_path": str(r.root), "source_episode_index": r.episode_index,
                "frames_source": proc[e]["frames_source"], "frames_out": lengths[e], "trim_frames": proc[e]["trim"],
                "keep_ranges": proc[e]["info"]["keep_ranges"], "task_source": srcs[e].task,
                "task_out": args.task or srcs[e].task, "source_meta": srcs[e].meta}
               for e, r in enumerate(refs)]
    (staging / "meta" / "source_episodes.json").write_text(json.dumps(src_map, indent=2))
    params = {"sigma_s": args.sigma_s, "action_offset_frames": args.action_offset_frames, "trim_idle": args.trim_idle,
              "trim_margin_s": args.trim_margin_s, "task_override": args.task, "left_only": bool(args.left_only),
              "arm": arm, "arm_only": arm_only, "fps": fps, "trim_pauses": args.trim_pauses,
              "pause_min_s": args.pause_min_s, "pause_keep_s": args.pause_keep_s, "pause_still_speed": args.pause_still_speed,
              "pause_max_join_deg": args.pause_max_join_deg,
              "pause_core_speed": args.pause_core_speed, "orig_action_name": args.orig_action_name,
              "keep_columns": list(keep_columns), "carry_meta": list(carry_meta), "video_mode": video_mode,
              "source_version": str(base["codebase_version"])}
    import lerobot
    prep = {
        "command": " ".join(sys.argv), "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_root": str(Path(args.source_root).expanduser().resolve()), "output": str(out_final),
        "lerobot_version": lerobot.__version__, "parameters": params,
        "filter": {"type": "zero-phase Gaussian (replay_episode.gaussian_smooth, odd reflection => endpoints exact)",
                   "sigma_s": args.sigma_s, "minus3db_hz": 0.1325 / args.sigma_s if args.sigma_s > 0 else None,
                   "applied_to": "joint columns of observation.state only (grippers and any zero arm unchanged)"},
        "video": ("stream copy (lerobot aggregate_videos / concat demuxer), not re-encoded" if video_mode == "copy"
                  else f"re-encoded kept frames with libx264, yuv420p in = yuv420p out, crf {args.video_crf}, gop {args.video_gop}, "
                       "no B-frames; packed with lerobot concatenate_video_files (stream copy)"),
        "video_encode": enc_info,
        "dropped_source_columns": dropped,
        "episodes": [{**m, **proc[m["episode_index"]]["info"]} for m in src_map],
        "max_filter_deviation_deg": float(max(p["info"]["deviation"]["max_deg_all"] for p in proc)),
        "build_seconds": round(time.time() - t0, 1),
    }
    (staging / "PREPROCESS.json").write_text(json.dumps(prep, indent=2))
    return prep


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def prepare_output(out: Path, force: bool) -> Path:
    """Return the staging dir; refuse an existing OUT unless --force (and then only a lerobot dataset / empty dir)."""
    out = out.expanduser().resolve()
    if out.exists():
        if not force:
            raise PreprocessError(f"{out} exists; use --force to replace it")
        if not out.is_dir() or (any(out.iterdir()) and not (out / "meta" / "info.json").is_file()):
            raise PreprocessError(f"{out} exists and is neither empty nor a lerobot dataset; not removing it")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.parent / f".{out.name}.partial-{os.getpid()}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_root", type=Path, help="a LeRobot dataset (v3.0 or v2.1, many episodes) or a folder with one "
                    "v3.0 dataset per subfolder (read-only)")
    ap.add_argument("--out", type=Path, required=True, help="output dataset directory (must not exist)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--include", help="comma-separated episodes to keep, e.g. 13,15,new_1 (or episode indices of a multi-episode dataset)")
    g.add_argument("--exclude", help="comma-separated episodes to drop, e.g. 19")
    ap.add_argument("--sigma-s", type=float, default=DEFAULT_SIGMA_S,
                    help="Gaussian sigma of the jitter filter [s]; 0 = no filter (default %(default)s: -3 dB at 1.7 Hz)")
    ap.add_argument("--action-offset-frames", type=int, default=1,
                    help="action[t] = filtered state[t + N] (default %(default)s = the next frame's pose)")
    ap.add_argument("--trim-idle", action="store_true", help="cut still head/tail, keep --trim-margin-s (default off)")
    ap.add_argument("--trim-margin-s", type=float, default=0.5)
    ap.add_argument("--trim-pauses", action="store_true",
                    help="shorten every mid-episode still run of >= --pause-min-s to --pause-keep-s (videos re-encoded)")
    ap.add_argument("--pause-min-s", type=float, default=PAUSE_MIN_S, help="shortest still run that is shortened (default %(default)s s)")
    ap.add_argument("--pause-keep-s", type=float, default=PAUSE_KEEP_S, help="what is kept of such a run (default %(default)s s)")
    ap.add_argument("--pause-still-speed", type=float, default=STILL_SPEED, help="arm speed [rad/s] at or below which the arm is still")
    ap.add_argument("--pause-core-speed", type=float, default=PAUSE_CORE_SPEED,
                    help="only the settled part of a pause (arm speed <= this [rad/s]) is cut (default %(default)s)")
    ap.add_argument("--pause-max-join-deg", type=float, default=PAUSE_MAX_JOIN_DEG,
                    help="skip a cut if the pose differs by more than this across it (default %(default)s deg)")
    ap.add_argument("--task", help="unify the task string of all episodes (default: keep the source strings)")
    ap.add_argument("--arm", help="arm whose joints drive segmentation / metrics (default: the arm that moves)")
    ap.add_argument("--arm-only", action="store_true", help="6-D state/action of --arm instead of all source columns")
    ap.add_argument("--left-only", action="store_true", help="same as --arm arm_left --arm-only")
    ap.add_argument("--orig-action-name", default=TELEOP_ACTION, help="column that keeps the untouched source action (default %(default)s)")
    ap.add_argument("--keep-columns", default=DEFAULT_KEEP_COLUMNS, help="extra per-frame source columns to carry (default %(default)s)")
    ap.add_argument("--carry-meta", default=DEFAULT_CARRY_META, help="per-episode source metadata keys copied into meta/episodes (default %(default)s)")
    ap.add_argument("--reencode-video", action="store_true", help="re-encode even where a stream copy would do")
    ap.add_argument("--video-crf", type=int, default=16, help="libx264 CRF of re-encoded videos (default %(default)s, ~44 dB PSNR)")
    ap.add_argument("--video-gop", type=int, default=2, help="GOP length of re-encoded videos (default %(default)s, lerobot's)")
    ap.add_argument("--workers", type=int, default=4, help="video encode processes, at most 4 (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="replace an existing OUTDIR (lerobot dataset or empty dir)")
    ap.add_argument("--verify", action="store_true", help="after building, run verify_output.py on the result")
    ap.add_argument("--repo-id", help="repo id used for the verify step (default local/<outdir name>)")
    args = ap.parse_args(argv)
    if args.action_offset_frames < 0:
        ap.error("--action-offset-frames must be >= 0")
    if args.sigma_s < 0:
        ap.error("--sigma-s must be >= 0")
    if args.pause_keep_s < 0 or args.pause_min_s <= args.pause_keep_s:
        ap.error("need 0 <= --pause-keep-s < --pause-min-s")
    if args.left_only:
        if args.arm and args.arm != "arm_left":
            ap.error("--left-only conflicts with --arm " + args.arm)
        args.arm, args.arm_only = "arm_left", True
    if args.arm_only and not args.arm:
        ap.error("--arm-only needs --arm (or use --left-only)")
    try:
        os.nice(19)
    except OSError:
        pass
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        refs = select_datasets(list_episodes(args.source_root), args.include, args.exclude)
        out = args.out.expanduser().resolve()
        staging = prepare_output(out, args.force)
    except (PreprocessError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    roots = sorted({r.root for r in refs})
    need = sum(dir_size(d) for d in roots)
    will_reencode = args.trim_pauses or args.reencode_video or any(r.n_in_dataset > 1 for r in refs) \
        or any(json.loads((r.root / "meta" / "info.json").read_text()).get("codebase_version", "").startswith("v2") for r in refs)
    if will_reencode:
        need *= 3          # re-encoded videos at CRF 16 are ~2-3x the source H.264 bitrate; packing needs a second copy
    free = shutil.disk_usage(out.parent).free
    if free < 1.3 * need:
        print(f"error: need ~{1.3 * need / 2**30:.1f} GiB on {out.parent}, only {free / 2**30:.1f} GiB free", file=sys.stderr)
        return 2
    print(f"{len(refs)} episodes -> {out}\nsource -> episode_index:")
    for i, r in enumerate(refs):
        print(f"  {r.name:20s} -> {i}")
    try:
        prep = build(args, refs, out, staging)
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        tempfile.tempdir = None
    print(f"done: {out}  ({dir_size(out) / 2**20:.0f} MiB, {len(refs)} episodes, max filter deviation "
          f"{prep['max_filter_deviation_deg']:.3f} deg, {prep['build_seconds']} s)")
    if args.verify:
        from verify_output import verify
        return 0 if verify(out, args.source_root, args.repo_id or f"local/{out.name}") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
