#!/usr/bin/env python3
"""Per-episode quality SCORE for the VLA teleop datasets (igus ReBeL, box-flap task).  Report only: nothing is dropped,
copied or modified; the datasets are opened read-only.

The score is computed AFTER the preprocess filter: raw sources get the same zero-phase endpoint-preserving Gaussian
(``episode_metrics.smooth``, sigma 0.08 s) in memory; datasets already built by preprocess.py (``prefiltered``) are used
as stored (filtering them again would smooth twice).  Everything is imported from episode_metrics / lerobot_episode /
video_frames, only the scoring layer lives here.

Eight sub-scores in [0, 1] (1 = good), combined to a weighted total 0-100 (``WEIGHTS``):

 1. smoothness   min of: jerk RMS of the filtered joints relative to the dataset median (ratio <= JERK_RATIO_GOOD ->1,
                 >= JERK_RATIO_BAD ->0), residual HF energy above 3 Hz after filtering (deg RMS, absolute), and the max
                 filtered joint speed against the 0.785 rad/s joint limit (<= limit ->1, >= 1.5x limit ->0).
 2. pauses       min of: share of frames with arm speed <= STILL_SPEED (episode_metrics) and the longest still run
                 inside the motion (motion_segments pauses), plus idle head/tail.
 3. branch       wrist branch (episode_metrics.branch_label: j4-/j6+, j4+/j6-, both, neither) against the dataset
                 majority: same ->1, "neither" ->0.5, other branch ->0.25, both branches in one episode ->0.15.
 4. start/end    distance (max joint, deg) of the first / last frame to the dataset MEDIAN first / last pose:
                 0.6 * start score + 0.4 * end score (ramps in deg).
 5. integrity    timestamps (monotonic, on the fps grid, dropped frames), NaN/inf, impossible joint steps
                 (episode_metrics.speed_outliers on the raw state, 5.7 deg/frame, and 3 deg/frame on the filtered
                 series), stale repeated state samples while moving (raw sources), all-zero poses, parquet rows vs meta length, video frames vs rows per camera.  Hard
                 failures (corrupt data, non-monotonic time, unreadable video) give 0.
 6. outliers     per-joint excursions outside the dataset's pooled p1-p99 envelope (share of frames outside, as a robust
                 z against the dataset's episodes) and the duration robust z (median/MAD).
 7. completion   duration and path length (sum |dq| of the filtered joints) relative to the dataset median (aborted /
                 cut short), final pose distance to the median final pose on j1,j2,j3,j5 (wrist excluded: it
                 depends on the branch), and an over-long path penalty.
 8. camera       per camera, from every SAMPLE_STRIDE-th decoded frame: brightness, black frames, blur (variance of the
                 Laplacian relative to the camera's dataset median), and frozen frames (>= FROZEN_MIN_RUN identical
                 consecutive frames, from a 64x36 thumbnail of EVERY frame, counted only while the arm moves; a frozen
                 fixed head camera is flagged but never a hard drop, it may simply not see the arm).  Episode = min over cameras.

Recommendation: ``drop`` on a hard failure (corrupt state, broken time base, unreadable/black/frozen camera, aborted
episode < ABORT_RATIO_DROP of the median) or total < TOTAL_DROP; ``review`` on total < TOTAL_REVIEW or any
sub-score < SUB_REVIEW; else ``keep``.  Thresholds below are dataset-relative (median / MAD) wherever an absolute
number would be arbitrary.  Run:

    nice -n 19 python3 tools/mocap_dataset/episode_score.py [--no-video] [--reuse-video] [--workers 4] [--out DIR]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import episode_metrics as em  # noqa: E402
import lerobot_episode as le  # noqa: E402

# ------------------------------------------------------------------------------------------- constants
DATA_ROOT = Path("/media/farhan/My Passport/igus_nvidia/datasets")
OUT_DIR = Path("/home/farhan/igus_rebel_ros2/scan_results/dataset_augment_20261031/quality")
# (label, folder, prefiltered by preprocess.py)
DATASETS = [
    ("flap_top_open_clean_v1_all12", "flap_top_open_clean_v1_all12", True),
    ("flap_top_open_clean_v2_9ep", "flap_top_open_clean_v2_9ep", True),
    ("real_right_20261029_raw", "real_right_20261029_raw", False),
    ("real_right_clean_v1", "real_right_clean_v1", True),
    ("mocap_right_front_flap_31oct_raw", "mocap_right_front_flap_31oct_raw", False),
    ("mocap_right_31oct_clean_v1", "mocap_right_31oct_clean_v1", True),
    ("real_right_3nov_raw", "real_right_3nov_raw", False),
    ("real_right_3nov_clean_v1", "real_right_3nov_clean_v1", True),
]
SIGMA_S = em.DEFAULT_SIGMA_S                 # same as preprocess.py
JOINT_LIMIT_RAD_S = 0.785                    # joints.yaml velocity limit

WEIGHTS = {"smoothness": 15, "pauses": 15, "branch": 15, "start_end": 10, "integrity": 15, "outliers": 10,
           "completion": 10, "camera": 10}   # sums to 100
assert sum(WEIGHTS.values()) == 100

# (good, bad): sub-score 1 at/below `good`, 0 at/above `bad` (linear between); inverted pairs ramp the other way
JERK_RATIO_GOOD, JERK_RATIO_BAD = 1.5, 4.0   # jerk RMS / dataset median
HF_DEG_GOOD, HF_DEG_BAD = 0.05, 0.30         # deg RMS above 3 Hz after filtering
SPEED_GOOD, SPEED_BAD = JOINT_LIMIT_RAD_S, 1.5 * JOINT_LIMIT_RAD_S
STILL_SHARE_GOOD, STILL_SHARE_BAD = 0.10, 0.40
PAUSE_MAX_GOOD_S, PAUSE_MAX_BAD_S = 3.0, 15.0
IDLE_EDGE_GOOD_S, IDLE_EDGE_BAD_S = 3.0, 15.0     # idle head + tail
BRANCH_SCORE = {"same": 1.0, "neither": 0.5, "other": 0.25, "both": 0.15}
START_GOOD_DEG, START_BAD_DEG = 10.0, 40.0
END_GOOD_DEG, END_BAD_DEG = 15.0, 60.0
START_WEIGHT = 0.6
JUMP_RAW_DEG = 5.7                           # as lerobot_to_plan / quality_report
JUMP_FILT_DEG = 3.0                          # filtered series: 2x the physical limit per frame (1.5 deg)
STALE_GOOD, STALE_BAD = 0.05, 0.40            # share of moving frames with a repeated (zero-step) state sample
DROPPED_FRAME_REVIEW, DROPPED_FRAME_DROP = 0.002, 0.05   # share of missing frames
TS_DEV_FRAMES_REVIEW = 1.5                   # |timestamp - index/fps| in frames
VIDEO_ROW_TOL = 2                            # frames; per-episode video length vs rows
OUTLIER_Z_GOOD, OUTLIER_Z_BAD = 3.0, 8.0     # robust z of the share of frames outside the p1-p99 envelope
EXCURSION_GOOD_DEG, EXCURSION_BAD_DEG = 5.0, 40.0
DURATION_Z_GOOD, DURATION_Z_BAD = 3.5, 8.0   # as quality_report robust_z_review
COMPLETE_RATIO_GOOD, COMPLETE_RATIO_BAD = 0.60, 0.25   # duration or path / median
ABORT_RATIO_DROP = 0.25
FINAL_POSE_GOOD_DEG, FINAL_POSE_BAD_DEG = 15.0, 60.0   # j1,j2,j3,j5 distance to the median final pose
PATH_RATIO_GOOD, PATH_RATIO_BAD = 2.0, 4.0
SAMPLE_STRIDE = 15                           # decode every 15th frame for brightness / blur
BRIGHT_LO_GOOD, BRIGHT_LO_BAD = 40.0, 15.0   # mean grey level (0-255)
BRIGHT_HI_GOOD, BRIGHT_HI_BAD = 215.0, 245.0
BLACK_MEAN = 8.0                             # sampled frame with mean grey below this is "black"
BLACK_GOOD, BLACK_BAD = 0.02, 0.25
BLUR_REL_GOOD, BLUR_REL_BAD = 0.40, 0.10     # episode median Laplacian variance / camera dataset median
FROZEN_DIFF = 0.01                           # thumbnail mean abs diff below this = identical frame
FROZEN_MOVING_SPEED = 0.1                    # rad/s: frozen frames only count while the arm moves faster than this
FROZEN_MIN_RUN = 10                          # identical consecutive frames to count as frozen (0.33 s)
FROZEN_GOOD, FROZEN_BAD = 0.01, 0.20
FROZEN_HEAD_GOOD, FROZEN_HEAD_BAD = 0.40, 0.90   # fixed head camera: identical frames are normal while the arm is out of view
CAMERA_DROP_SHARE = 0.5                      # black or frozen share above this = unusable camera
TOTAL_DROP, TOTAL_REVIEW, SUB_REVIEW = 40.0, 70.0, 0.6

SUBS = list(WEIGHTS)


# ------------------------------------------------------------------------------------------------ helpers
def ramp(x: float, good: float, bad: float) -> float:
    """1 at ``good``, 0 at ``bad``, linear in between, clipped (works for good < bad and good > bad)."""
    if not np.isfinite(x):
        return 0.0
    return float(np.clip((bad - x) / (bad - good), 0.0, 1.0))


def _med(x) -> float:
    return float(np.median(np.asarray(x, dtype=np.float64)))


# -------------------------------------------------------------------------------------------- loading
def load_group(label: str, folder: str, prefiltered: bool, max_eps: int | None = None) -> list[dict]:
    """Every episode of a dataset folder as a dict with the arrays the scoring needs (no videos opened here)."""
    root = DATA_ROOT / folder
    refs = le.list_episodes(root)
    if max_eps:
        refs = refs[:max_eps]
    src = {}
    for d in le.find_datasets(root):
        pj = d / "PREPROCESS.json"
        if pj.is_file():
            for e in json.loads(pj.read_text()).get("episodes", []):
                src[(str(d), int(e["episode_index"]))] = e.get("source")
    rows_meta = {str(d): {int(r["episode_index"]): r for r in le.episode_rows(d)} for d in le.find_datasets(root)}
    out = []
    for ref in refs:
        ep = le.load_episode(ref.root, ref.episode_index)
        arm = ep.active_arm()
        q = ep.arm(arm, "state")
        label_ep = src.get((str(ref.root), ref.episode_index)) or ep.name
        out.append({"group": label, "prefiltered": prefiltered, "name": ep.name, "label": label_ep,
                    "episode_index": ref.episode_index, "root": str(ref.root), "arm": arm, "fps": ep.fps, "ts": ep.timestamp,
                    "q_raw": q, "version": ep.version, "videos": {k: (str(p), t0) for k, (p, t0) in ep.videos.items()},
                    "meta_row": rows_meta[str(ref.root)][ref.episode_index]})
    return out


# ------------------------------------------------------------------------------------- per-episode metrics
def episode_metrics(e: dict) -> dict:
    """Dataset-independent raw metrics of one loaded episode (filter applied here for raw sources)."""
    fps, q, ts = e["fps"], np.asarray(e["q_raw"], dtype=np.float64), np.asarray(e["ts"], dtype=np.float64)
    n = len(q)
    finite = np.isfinite(q).all(axis=1)
    qq = np.where(np.isfinite(q), q, 0.0) if n else q
    qf = qq if e["prefiltered"] else em.smooth(qq, SIGMA_S, fps)
    m: dict = {"frames": n, "duration_s": n / fps, "fps": fps}
    # smoothness
    jit = em.jitter_metrics(qf, fps)
    sp = em.arm_speed(qf, fps)
    jerk_per_joint = np.asarray(jit["jerk_rms"])
    m.update(jerk_rms=float(jerk_per_joint.mean()), hf_rms_deg=float(np.mean(jit["hf_rms_deg"])),
             max_speed_rad_s=float(np.abs(np.gradient(qf, 1.0 / fps, axis=0)).max()))
    if not e["prefiltered"]:
        from quality_report import stale_samples          # repeated state samples while moving (raw state only)
        st = stale_samples(qq, qf, fps)
        m["stale_share_moving"] = st["stale_share_when_moving"]
        m["filter_dev_max_deg"] = em.deviation_deg(qq, qf)["max_deg_all"]
        m["raw_hf_rms_deg"] = float(np.mean(em.jitter_metrics(qq, fps)["hf_rms_deg"]))
    # pauses
    seg = em.motion_segments(qf, fps)
    m.update(still_share=float((sp <= em.STILL_SPEED).mean()), pause_max_s=seg["pause_max_s"],
             pause_total_s=seg["pause_total_s"], start_still_s=seg["start_still_s"], end_still_s=seg["end_still_s"])
    # branch
    b = em.branch_label(qf)
    m.update(branch=b["label"], j4_range_deg=[float(np.degrees(b["j4_min"])), float(np.degrees(b["j4_max"]))],
             j6_range_deg=[float(np.degrees(b["j6_min"])), float(np.degrees(b["j6_max"]))])
    # poses
    m["start_deg"] = np.degrees(qf[0]).tolist()
    m["end_deg"] = np.degrees(qf[-1]).tolist()
    # path
    m["path_rad"] = float(np.abs(np.diff(qf, axis=0)).sum())
    # integrity
    dt = np.diff(ts)
    step_frames = np.round(dt * fps)
    dropped = int(np.clip(step_frames - 1, 0, None).sum()) if len(dt) else 0
    ideal = np.arange(n) / fps
    m.update(nonmonotonic_ts=int((dt <= 0).sum()), dropped_frames=dropped, dropped_share=dropped / max(n, 1),
             ts_dev_frames=float(np.abs((ts - ts[0]) - ideal).max() * fps) if n else 0.0, ts_start=float(ts[0]) if n else 0.0,
             nan_frames=int((~finite).sum()))
    so_raw = em.speed_outliers(qq, fps, JUMP_RAW_DEG)
    so_filt = em.speed_outliers(qf, fps, JUMP_FILT_DEG)
    m.update(jump_events_raw=so_raw["n_events"], max_step_deg_raw=so_raw["max_step_deg"],
             jump_events_filt=so_filt["n_events"], max_step_deg_filt=so_filt["max_step_deg"],
             zero_pose_frames=em.zero_pose_frames(qq))
    mr = e["meta_row"]
    m["meta_length"] = int(mr["length"]) if "length" in mr else None
    m["rows_vs_meta"] = None if m["meta_length"] is None else n - m["meta_length"]
    e["_qf"] = qf
    e["_moving"] = sp > FROZEN_MOVING_SPEED
    return m


def group_context(eps: list[dict], mets: list[dict]) -> dict:
    """Dataset-level references: medians, majority branch, p1-p99 envelope, pooled stats."""
    labels = Counter(m["branch"] for m in mets if m["branch"] != "neither")
    majority = labels.most_common(1)[0][0] if labels else "neither"
    pooled = np.degrees(np.concatenate([e["_qf"] for e in eps]))
    return {"majority_branch": majority, "branch_counts": dict(Counter(m["branch"] for m in mets)),
            "median_start_deg": np.median([m["start_deg"] for m in mets], axis=0),
            "median_end_deg": np.median([m["end_deg"] for m in mets], axis=0),
            "median_duration_s": _med([m["duration_s"] for m in mets]),
            "median_path_rad": _med([m["path_rad"] for m in mets]),
            "median_jerk": _med([m["jerk_rms"] for m in mets]),
            "env_lo": np.percentile(pooled, 1, axis=0), "env_hi": np.percentile(pooled, 99, axis=0)}


# ------------------------------------------------------------------------------------------------ video
def _lap_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def scan_video_file(job: dict) -> dict:
    """Worker: ONE video file.  ``job = {path, fps, segments: [{key, from_ts, rows}]}``.

    Decodes the whole file once for the 64x36 thumbnails (video_frames.motion_energy: decoded frame count and
    frozen/duplicate frames) and, per episode segment, every SAMPLE_STRIDE-th frame at full size
    (video_frames.read_frames_by_index) for brightness and blur.  Returns plain statistics only."""
    try:
        os.nice(19)
    except OSError:
        pass
    from video_frames import motion_energy, read_frames_by_index
    fps = job["fps"]
    res = {"path": job["path"], "segments": {}}
    try:
        me = motion_energy(job["path"], threads=1)
    except Exception as ex:  # noqa: BLE001  (corrupt / unreadable file is a finding, not a crash)
        res["error"] = f"{type(ex).__name__}: {ex}"
        return res
    res["decoded"] = int(me["n"])
    motion = me["motion"]
    for seg in job["segments"]:
        off = int(round(seg["from_ts"] * fps))
        rows = seg["rows"]
        out = {"offset": off, "rows": rows, "available": int(max(0, min(rows, me["n"] - off)))}
        mo = motion[off:off + rows].copy()
        if len(mo):
            mo[0] = 1.0                                     # entry 0 compares with the previous episode / is 0 by definition
        dup = mo < FROZEN_DIFF
        mov = np.asarray(seg["moving"], dtype=bool)[:len(mo)]          # frames where the arm moves (filtered speed)
        mov = np.pad(mov, (0, len(mo) - len(mov)))
        frozen, run0 = 0, None
        for i, d in enumerate(list(dup) + [False]):
            if d and run0 is None:
                run0 = i
            elif not d and run0 is not None:
                if i - run0 >= FROZEN_MIN_RUN:
                    frozen += int(mov[run0:i].sum())
                run0 = None
        out["frozen_share"] = frozen / max(int(mov.sum()), 1)          # frozen frames while the arm moves / moving frames
        out["moving_frames"] = int(mov.sum())
        out["dup_share"] = float(dup.mean()) if len(dup) else 0.0
        idx = [i for i in range(0, out["available"], SAMPLE_STRIDE)]
        try:
            frames = read_frames_by_index(job["path"], idx, fps, from_ts=seg["from_ts"], threads=1) if idx else {}
        except Exception as ex:  # noqa: BLE001
            out["error"] = f"{type(ex).__name__}: {ex}"
            frames = {}
        br, bl = [], []
        for i in sorted(frames):
            g = frames[i].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            br.append(float(g.mean()))
            bl.append(_lap_var(g))
        out.update(n_sampled=len(br), brightness_mean=float(np.mean(br)) if br else None,
                   brightness_min=float(np.min(br)) if br else None, black_share=float(np.mean(np.array(br) < BLACK_MEAN)) if br else None,
                   blur_median=float(np.median(bl)) if bl else None)
        res["segments"][seg["key"]] = out
    return res


def video_jobs(eps: list[dict]) -> list[dict]:
    jobs: dict = {}
    for e in eps:
        for cam, (path, t0) in e["videos"].items():
            j = jobs.setdefault(path, {"path": path, "fps": e["fps"], "segments": []})
            j["segments"].append({"key": f"{e['group']}|{e['episode_index']}|{e['root']}|{cam}", "from_ts": t0,
                                  "rows": len(e["q_raw"]), "cam": cam, "moving": e["_moving"]})
    return list(jobs.values())


def run_video(eps_all: list[dict], workers: int) -> dict:
    import multiprocessing as mp
    jobs = video_jobs(eps_all)
    print(f"video: {len(jobs)} files, {workers} workers", flush=True)
    results = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(scan_video_file, jobs), 1):
            results[r["path"]] = r
            print(f"  [{k}/{len(jobs)}] {Path(r['path']).parent.parent.parent.name}/{Path(r['path']).parent.parent.name}"
                  f"/{Path(r['path']).name} decoded={r.get('decoded')} {r.get('error', '')}", flush=True)
    return {"files": results}


def camera_stats(e: dict, vid: dict | None) -> dict:
    """Per camera raw video stats of an episode from the worker results (``vid`` = run_video output)."""
    out = {}
    if vid is None:
        return out
    for cam, (path, _t0) in e["videos"].items():
        fr = vid["files"].get(path)
        key = f"{e['group']}|{e['episode_index']}|{e['root']}|{cam}"
        if fr is None or "error" in fr:
            out[cam] = {"error": (fr or {}).get("error", "file not scanned")}
            continue
        s = dict(fr["segments"][key])
        s["file_decoded"] = fr["decoded"]
        out[cam] = s
    return out


# -------------------------------------------------------------------------------------------------- scoring
def score_group(eps: list[dict], mets: list[dict], vid: dict | None) -> list[dict]:
    ctx = group_context(eps, mets)
    cams = sorted({c for e in eps for c in e["videos"]})
    cstats = [camera_stats(e, vid) for e in eps]
    blur_ref = {c: _med([s[c]["blur_median"] for s in cstats if c in s and s[c].get("blur_median")] or [1.0]) for c in cams}
    durz = em.robust_z(np.array([m["duration_s"] for m in mets]))
    # share of frames outside the envelope
    outside, maxexc = [], []
    for e in eps:
        x = np.degrees(e["_qf"])
        exc = np.maximum(np.maximum(ctx["env_lo"] - x, x - ctx["env_hi"]), 0.0)
        outside.append(float((exc.max(axis=1) > 0).mean()))
        maxexc.append(float(exc.max()))
    outz = em.robust_z(np.array(outside))
    results = []
    for i, (e, m) in enumerate(zip(eps, mets)):
        flags: list[str] = []
        by_sub: dict[str, list[str]] = {}
        cur = ["smoothness"]

        def add(text, flags=flags, by_sub=by_sub, cur=cur):
            flags.append(text)
            by_sub.setdefault(cur[0], []).append(text)

        hard: list[str] = []
        sub: dict[str, float] = {}
        raw = dict(m)
        # 1 smoothness
        jr = m["jerk_rms"] / max(ctx["median_jerk"], 1e-9)
        raw["jerk_ratio"] = jr
        s1 = [ramp(jr, JERK_RATIO_GOOD, JERK_RATIO_BAD), ramp(m["hf_rms_deg"], HF_DEG_GOOD, HF_DEG_BAD),
              ramp(m["max_speed_rad_s"], SPEED_GOOD, SPEED_BAD)]
        sub["smoothness"] = min(s1)
        if s1[0] < 1:
            add(f"jerky: jerk RMS {m['jerk_rms']:.1f} rad/s^3 = {jr:.1f}x the dataset median")
        if s1[1] < 1:
            add(f"residual HF after filter {m['hf_rms_deg']:.3f} deg RMS")
        if s1[2] < 1:
            add(f"max joint speed {m['max_speed_rad_s']:.2f} rad/s above the {JOINT_LIMIT_RAD_S} limit")
        # 2 pauses
        cur[0] = "pauses"
        edge = m["start_still_s"] + m["end_still_s"]
        s2 = [ramp(m["still_share"], STILL_SHARE_GOOD, STILL_SHARE_BAD), ramp(m["pause_max_s"], PAUSE_MAX_GOOD_S, PAUSE_MAX_BAD_S),
              ramp(edge, IDLE_EDGE_GOOD_S, IDLE_EDGE_BAD_S)]
        sub["pauses"] = min(s2)
        if s2[0] < 1:
            add(f"{100 * m['still_share']:.0f} % of the time standing still")
        if s2[1] < 1:
            add(f"longest mid-episode pause {m['pause_max_s']:.1f} s")
        if s2[2] < 1:
            add(f"idle head {m['start_still_s']:.1f} s + tail {m['end_still_s']:.1f} s")
        # 3 branch
        cur[0] = "branch"
        lab, maj = m["branch"], ctx["majority_branch"]
        kind = "both" if lab == "both" else "same" if lab == maj else "neither" if lab == "neither" else "other"
        sub["branch"] = BRANCH_SCORE[kind]
        raw["branch_kind"] = kind
        if kind == "both":
            add("uses both wrist branches inside one episode")
        elif kind == "other":
            add(f"wrist branch {lab} differs from the dataset majority {maj}")
        elif kind == "neither":
            add(f"no wrist counter-rotation (branch 'neither'; majority {maj})")
        # 4 start / end
        cur[0] = "start_end"
        ds = np.abs(np.array(m["start_deg"]) - ctx["median_start_deg"])
        de = np.abs(np.array(m["end_deg"]) - ctx["median_end_deg"])
        raw["start_dist_deg"], raw["end_dist_deg"] = float(ds.max()), float(de.max())
        s_s, s_e = ramp(ds.max(), START_GOOD_DEG, START_BAD_DEG), ramp(de.max(), END_GOOD_DEG, END_BAD_DEG)
        sub["start_end"] = START_WEIGHT * s_s + (1 - START_WEIGHT) * s_e
        if s_s < 1:
            add(f"start pose {ds.max():.0f} deg (max joint j{int(ds.argmax()) + 1}) from the dataset median start")
        if s_e < 1:
            add(f"end pose {de.max():.0f} deg (max joint j{int(de.argmax()) + 1}) from the dataset median end")
        # 5 integrity
        cur[0] = "integrity"
        integ = 1.0
        if m["nan_frames"]:
            hard.append(f"CORRUPT: {m['nan_frames']} frame(s) with NaN/inf state")
        if m["jump_events_raw"]:
            hard.append(f"CORRUPT: {m['jump_events_raw']} impossible joint step(s) > {JUMP_RAW_DEG} deg/frame "
                        f"(max {m['max_step_deg_raw']:.1f} deg)")
        elif m["jump_events_filt"]:
            hard.append(f"CORRUPT: {m['jump_events_filt']} joint step(s) > {JUMP_FILT_DEG} deg/frame after filtering "
                        f"(max {m['max_step_deg_filt']:.1f} deg)")
        if m["zero_pose_frames"]:
            hard.append(f"CORRUPT: {m['zero_pose_frames']} frame(s) with all six joints 0")
        if m["nonmonotonic_ts"]:
            hard.append(f"CORRUPT: {m['nonmonotonic_ts']} non-increasing timestamp(s)")
        if m["dropped_share"] > DROPPED_FRAME_DROP:
            hard.append(f"{m['dropped_frames']} dropped frames ({100 * m['dropped_share']:.0f} %)")
        elif m["dropped_share"] > DROPPED_FRAME_REVIEW:
            add(f"{m['dropped_frames']} dropped frame(s) in the timestamps")
            integ = min(integ, ramp(m["dropped_share"], DROPPED_FRAME_REVIEW, DROPPED_FRAME_DROP))
        if m["ts_dev_frames"] > TS_DEV_FRAMES_REVIEW and not m["dropped_frames"] and not m["nonmonotonic_ts"]:
            add(f"timestamps drift {m['ts_dev_frames']:.1f} frames off the {m['fps']:g} fps grid")
            integ = min(integ, 0.5)
        if m.get("stale_share_moving", 0.0) > STALE_GOOD:
            add(f"{100 * m['stale_share_moving']:.0f} % of the moving frames repeat the previous state sample (stale)")
            integ = min(integ, 0.4 + 0.6 * ramp(m["stale_share_moving"], STALE_GOOD, STALE_BAD))
        if m["rows_vs_meta"]:
            add(f"parquet rows differ from meta length by {m['rows_vs_meta']:+d}")
            integ = min(integ, 0.3)
        cs = cstats[i]
        vid_rows = {}
        for cam, s in cs.items():
            if "error" in s:
                hard.append(f"video {cam} unreadable: {s['error']}")
                continue
            diff = s["available"] - m["frames"]            # frames of this episode actually present in the file
            seg_ok = s["available"] >= m["frames"] - VIDEO_ROW_TOL
            vid_rows[cam] = {"available": s["available"], "rows": m["frames"], "file_decoded": s["file_decoded"]}
            if not seg_ok:
                add(f"video {cam}: only {s['available']} of {m['frames']} frames present in the file")
                integ = min(integ, 0.2 if diff < -0.1 * m["frames"] else 0.5)
            # v3.0: meta to_timestamp - from_timestamp must equal rows / fps
            mr = e["meta_row"]
            tk = f"videos/{cam}/to_timestamp"
            if tk in mr:
                span = (mr[tk] - mr[f"videos/{cam}/from_timestamp"]) * m["fps"]
                if abs(span - m["frames"]) > VIDEO_ROW_TOL:
                    add(f"video {cam}: meta timestamps span {span:.0f} frames, rows {m['frames']}")
                    integ = min(integ, 0.5)
        sub["integrity"] = 0.0 if hard else integ
        # 6 outliers
        cur[0] = "outliers"
        raw["frac_outside_env"], raw["max_excursion_deg"], raw["duration_z"] = outside[i], maxexc[i], float(durz[i])
        s6 = [ramp(max(outz[i], 0.0), OUTLIER_Z_GOOD, OUTLIER_Z_BAD), ramp(maxexc[i], EXCURSION_GOOD_DEG, EXCURSION_BAD_DEG),
              ramp(abs(durz[i]), DURATION_Z_GOOD, DURATION_Z_BAD)]
        sub["outliers"] = min(s6)
        if s6[0] < 1 or s6[1] < 1:
            add(f"{100 * outside[i]:.0f} % of frames outside the dataset p1-p99 envelope (robust z {outz[i]:+.1f}), "
                         f"max excursion {maxexc[i]:.0f} deg")
        if s6[2] < 1:
            add(f"duration {m['duration_s']:.0f} s is an outlier (robust z {durz[i]:+.1f}, median {ctx['median_duration_s']:.0f} s)")
        # 7 completion
        cur[0] = "completion"
        r_dur, r_path = m["duration_s"] / ctx["median_duration_s"], m["path_rad"] / ctx["median_path_rad"]
        raw["duration_ratio"], raw["path_ratio"] = r_dur, r_path
        fin = np.abs(np.array(m["end_deg"]) - ctx["median_end_deg"])[[0, 1, 2, 4]].max()
        raw["final_pose_dist_deg"] = float(fin)
        short = min(r_dur, r_path)
        s7 = [ramp(short, COMPLETE_RATIO_GOOD, COMPLETE_RATIO_BAD), ramp(fin, FINAL_POSE_GOOD_DEG, FINAL_POSE_BAD_DEG),
              ramp(r_path, PATH_RATIO_GOOD, PATH_RATIO_BAD)]
        sub["completion"] = min(s7)
        if short < ABORT_RATIO_DROP:
            hard.append(f"ABORTED: duration {m['duration_s']:.0f} s / path {m['path_rad']:.1f} rad = "
                        f"{100 * short:.0f} % of the dataset median")
        elif s7[0] < 1:
            add(f"short/cut short: duration {100 * r_dur:.0f} %, path {100 * r_path:.0f} % of the median")
        if s7[1] < 1:
            add(f"final pose {fin:.0f} deg (j1,j2,j3,j5) from the median final pose: task end not reached?")
        if s7[2] < 1:
            add(f"path length {r_path:.1f}x the median (wandering)")
        # 8 camera
        cur[0] = "camera"
        if cs:
            cam_scores = {}
            for cam, s in cs.items():
                if "error" in s or s.get("brightness_mean") is None:
                    cam_scores[cam] = 0.0
                    if "error" not in s:
                        add(f"camera {cam}: no frames could be sampled")
                    continue
                rel = s["blur_median"] / max(blur_ref[cam], 1e-9)
                sc = [min(ramp(s["brightness_mean"], BRIGHT_LO_GOOD, BRIGHT_LO_BAD), ramp(s["brightness_mean"], BRIGHT_HI_GOOD, BRIGHT_HI_BAD)),
                      ramp(s["black_share"], BLACK_GOOD, BLACK_BAD), ramp(rel, BLUR_REL_GOOD, BLUR_REL_BAD),
                      ramp(s["frozen_share"], *((FROZEN_HEAD_GOOD, FROZEN_HEAD_BAD) if cam.endswith("head") else (FROZEN_GOOD, FROZEN_BAD)))]
                cam_scores[cam] = min(sc)
                s["blur_rel"] = rel
                short_cam = cam.split(".")[-1]
                if sc[0] < 1:
                    add(f"camera {short_cam}: brightness {s['brightness_mean']:.0f}/255")
                if sc[1] < 1:
                    add(f"camera {short_cam}: {100 * s['black_share']:.0f} % black frames")
                if sc[2] < 1:
                    add(f"camera {short_cam}: blurry (Laplacian variance {rel:.2f}x the dataset median)")
                if sc[3] < 1:
                    add(f"camera {short_cam}: {100 * s['frozen_share']:.0f} % frozen/duplicate frames")
                arm_cam = not cam.endswith("head")                 # a fixed scene camera may not see the arm: its freeze is a flag only
                if s["black_share"] > CAMERA_DROP_SHARE or (arm_cam and s["frozen_share"] > CAMERA_DROP_SHARE):
                    hard.append(f"camera {short_cam} unusable (black {100 * s['black_share']:.0f} %, frozen {100 * s['frozen_share']:.0f} %)")
            sub["camera"] = min(cam_scores.values())
            raw["cameras"] = cs
            raw["camera_scores"] = cam_scores
        else:
            sub["camera"] = None  # video not scanned: the weight is redistributed over the other sub-scores
        results.append({"dataset": e["group"], "name": e["name"], "label": e["label"], "episode_index": e["episode_index"],
                        "arm": e["arm"], "sub_scores": sub, "raw": raw, "flags": flags, "flags_by_sub": by_sub, "hard_flags": hard, "ctx_branch": maj})
    for r in results:
        w = {k: WEIGHTS[k] for k in SUBS if r["sub_scores"][k] is not None}
        tot = sum(w[k] * r["sub_scores"][k] for k in w) / sum(w.values()) * 100.0
        r["total"] = round(float(tot), 1)
        r["recommendation"], r["reason"] = recommend(r)
    return results


def recommend(r: dict) -> tuple[str, str]:
    if r["hard_flags"]:
        return "drop", "; ".join(r["hard_flags"][:3])
    if r["total"] < TOTAL_DROP:
        return "drop", f"total score {r['total']:.0f} < {TOTAL_DROP:.0f}"
    low = [k for k in SUBS if r["sub_scores"][k] is not None and r["sub_scores"][k] < SUB_REVIEW]
    if r["total"] < TOTAL_REVIEW or low:
        why = [f"total {r['total']:.0f}"] if r["total"] < TOTAL_REVIEW else []
        for k in low:
            txt = r["flags_by_sub"].get(k)
            why.append(f"{k} {r['sub_scores'][k]:.2f}" + (f" ({'; '.join(txt[:2])})" if txt else ""))
        if not low and r["flags"]:
            why.append("; ".join(r["flags"][:2]))
        return "review", ", ".join(why)
    return "keep", "all sub-scores >= %.1f, total %.0f" % (SUB_REVIEW, r["total"])


# ------------------------------------------------------------------------------------------------ outputs
def write_outputs(results: list[dict], out: Path, ctxs: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return round(float(o), 5) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return o

    doc = {"weights": WEIGHTS, "sigma_s": SIGMA_S, "datasets": ctxs, "episodes": results}
    (out / "scores.json").write_text(json.dumps(clean(doc), indent=1))
    keys = ["jerk_ratio", "hf_rms_deg", "max_speed_rad_s", "still_share", "pause_max_s", "branch", "start_dist_deg",
            "end_dist_deg", "dropped_frames", "frac_outside_env", "duration_s", "duration_ratio", "path_ratio",
            "final_pose_dist_deg"]
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "episode_index", "label", "name", "total", "recommendation", *SUBS, *keys, "flags", "reason"])
        for r in results:
            w.writerow([r["dataset"], r["episode_index"], r["label"], r["name"], r["total"], r["recommendation"],
                        *[("" if r["sub_scores"][k] is None else f"{r['sub_scores'][k]:.3f}") for k in SUBS],
                        *[(f"{r['raw'][k]:.4g}" if isinstance(r["raw"].get(k), float) else r["raw"].get(k, "")) for k in keys],
                        " | ".join(r["hard_flags"] + r["flags"]), r["reason"]])
    plot_png(results, out)
    (out / "quality_report.md").write_text(build_md(results, ctxs))


COLORS = {"keep": "#3a9d5d", "review": "#e0a030", "drop": "#c8423c"}


def plot_png(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ds in sorted({r["dataset"] for r in results}):
        rs = sorted([r for r in results if r["dataset"] == ds], key=lambda r: r["total"])
        fig, ax = plt.subplots(figsize=(6.0, 0.2 * len(rs) + 1.0), dpi=70)
        ax.barh(range(len(rs)), [r["total"] for r in rs], color=[COLORS[r["recommendation"]] for r in rs])
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([f"{r['label']}"[-26:] for r in rs], fontsize=6)
        ax.set_xlim(0, 100)
        ax.axvline(TOTAL_REVIEW, color="grey", lw=0.6, ls="--")
        ax.set_title(f"{ds}: episode score (green keep, amber review, red drop)", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        fig.tight_layout()
        fig.savefig(out / f"scores_{ds}.png")
        plt.close(fig)


def build_md(results: list[dict], ctxs: dict) -> str:
    L = ["# Episode quality scores (report only, nothing dropped)", "",
         "Generated by `tools/mocap_dataset/episode_score.py`. Score = weighted sum of 8 sub-scores in [0,1] x 100. "
         "Datasets are read-only; raw sources are filtered in memory (zero-phase Gaussian, sigma 0.08 s) as preprocess.py does, "
         "`clean` datasets are scored as stored.", "", "## How the score is defined", "",
         "Weights: " + ", ".join(f"{k} {v}" for k, v in WEIGHTS.items()) + ".", ""]
    doc = __doc__.split("Eight sub-scores", 1)[1].split("Run:", 1)[0]
    L += ["```", "Eight sub-scores" + doc.rstrip(), "```", "",
          f"Thresholds (constants at the top of the script): total < {TOTAL_DROP:.0f} drop, < {TOTAL_REVIEW:.0f} review, any "
          f"sub-score < {SUB_REVIEW} review; hard failures (CORRUPT, ABORTED, unreadable/black/frozen camera) drop.", ""]
    for ds in [d[0] for d in DATASETS]:
        rs = sorted([r for r in results if r["dataset"] == ds], key=lambda r: -r["total"])
        if not rs:
            continue
        c = Counter(r["recommendation"] for r in rs)
        cx = ctxs[ds]
        L += [f"## {ds}", "", f"{len(rs)} episodes ({rs[0]['arm']}): keep {c['keep']}, review {c['review']}, drop {c['drop']}. "
              f"Majority branch {cx['majority_branch']} {cx['branch_counts']}; median duration {cx['median_duration_s']:.0f} s. "
              f"![scores](scores_{ds}.png)", "",
              "| episode | total | rec | " + " | ".join(s[:5] for s in SUBS) + " | flags |", "|---|---|---|" + "---|" * len(SUBS) + "---|"]
        for r in rs:
            sc = " | ".join("-" if r["sub_scores"][k] is None else f"{r['sub_scores'][k]:.2f}" for k in SUBS)
            fl = "; ".join(r["hard_flags"] + r["flags"]).replace("|", "/")
            L.append(f"| {r['label']} (#{r['episode_index']}) | {r['total']:.0f} | {r['recommendation']} | {sc} | {fl} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- driver
def score_all(video_mode: str, workers: int, out: Path, max_eps: int | None, only: list[str] | None) -> list[dict]:
    groups = {}
    for label, folder, pre in DATASETS:
        if only and label not in only:
            continue
        print(f"loading {label}", flush=True)
        groups[label] = load_group(label, folder, pre, max_eps)
    mets = {g: [episode_metrics(e) for e in eps] for g, eps in groups.items()}
    cache = out / "video_stats.json"
    vid = None
    if video_mode == "scan":
        vid = run_video([e for eps in groups.values() for e in eps], workers)
        out.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(vid))
    elif video_mode == "reuse":
        vid = json.loads(cache.read_text())
    results, ctxs = [], {}
    for g, eps in groups.items():
        results += score_group(eps, mets[g], vid)
        c = group_context(eps, mets[g])
        ctxs[g] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    write_outputs(results, out, ctxs)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-video", action="store_true", help="skip the camera sub-score (weights renormalised)")
    ap.add_argument("--reuse-video", action="store_true", help="reuse <out>/video_stats.json from an earlier run")
    ap.add_argument("--max-episodes", type=int, default=None, help="debug: first N episodes per dataset")
    ap.add_argument("--only", nargs="*", default=None, help="dataset labels")
    a = ap.parse_args(argv)
    a.workers = max(1, min(a.workers, 4))
    mode = "none" if a.no_video else "reuse" if a.reuse_video else "scan"
    res = score_all(mode, a.workers, a.out, a.max_episodes, a.only)
    for ds in sorted({r["dataset"] for r in res}):
        c = Counter(r["recommendation"] for r in res if r["dataset"] == ds)
        print(f"{ds}: keep {c['keep']} review {c['review']} drop {c['drop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
