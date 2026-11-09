#!/usr/bin/env python3
"""Per-episode quality report for recorded arm datasets (read-only: nothing under SOURCE_ROOT is written).

    python3 tools/mocap_dataset/quality_report.py SOURCE_ROOT --out DIR [--sigma-s 0.08] [--arm arm_right]
        [--thumb-videos head,wrist_right] [--video-consistency wrist_right] [--rule end_pose_review_deg=15 ...]

SOURCE_ROOT is a folder of one-episode LeRobot v3.0 datasets (the left-arm mocap recordings), or ONE dataset with many
episodes, v3.0 or v2.1 (``real_right``).  Writes ``DIR/report.html`` (self-contained, PNGs embedded), ``DIR/report.json`` and
the PNGs (``overview_j4_j6.png``, ``jitter_psd.png``, ``jitter_timeseries.png``, ``box_pose.png`` when episodes carry a
``box_pose``, one ``ep_<name>.png`` per episode).  The keep/review flag is a suggestion with reasons; it does not decide
anything.  Run it with ``nice -n 19`` if a camera node is running: the thumbnails decode a few video frames per episode and
``--video-consistency`` decodes one whole camera.

Rules that are OFF unless set with ``--rule name=value`` (so the left-arm report does not change): ``end_pose_review_deg``
(final pose farther than this from every other episode's final pose), ``pause_share_review`` (fraction of the
episode spent in still runs >= 1 s) (camera motion vs. arm speed correlates poorly on real_right, see README, so there is no rule on it).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from episode_metrics import (DEFAULT_SIGMA_S, HF_CUTOFF_HZ, JOINT_LABELS, START_POSE_DEG, arm_speed, branch_label,  # noqa: E402
                             cluster_poses, deviation_deg, dtw_matrix, jitter_metrics,
                             joint_limit_margin_deg, motion_segments, plan_pause_trim, robust_z, smooth,
                             speed_outliers, start_pose_distance, summarize_jitter, trim_bounds, video_state_consistency,
                             zero_pose_frames)
from lerobot_episode import Episode, list_episodes, load_episode  # noqa: E402
from lerobot_to_plan import DEFAULT_URDF, PlanError, check_series, urdf_limits  # noqa: E402

DTW_HZ = 5.0
FLAG_RULES = {
    "start_pose_review_deg": 10.0,        # max |joint - majority start pose| above this (arm_left datasets)
    "robust_z_review": 3.5,               # duration / DTW outliers
    "start_still_review_s": 5.0,
    "end_still_review_s": 3.0,
    "mid_pause_review_s": 3.0,            # one pause at least this long
    "max_dev_review_deg": 1.0,            # filter deviation from the raw state
    "limit_margin_deg": 2.0,              # joint limit margin (URDF limit minus this), as lerobot_to_plan
    "max_jump_deg": 5.7,                  # impossible joint step per frame (5.7 deg = 3.0 rad/s at 30 fps), as lerobot_to_plan
    "limit_note_deg": 5.0,                # closer than this to a joint limit is noted (not flagged)
    "start_variant_tol_deg": 7.0,         # start poses within this (max joint) are the same start-pose variant
    "end_variant_tol_deg": 12.0,          # same for the final poses
    "box_pose_tol": [1e-4, 1e-4, 0.01],   # x, y [m], yaw [deg]: identical box placement
    "end_pose_review_deg": None,          # OFF by default
    "pause_share_review": None,           # OFF by default
}


def b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def fig_png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------ analysis
def stale_samples(q: np.ndarray, qf: np.ndarray, fps: float, moving_speed: float = 0.05) -> dict:
    """Share of frames in which every joint step is exactly zero although the arm is moving (a stale / repeated state
    sample).  ``mask`` marks those frames (index of the later frame of the step)."""
    step = np.abs(np.diff(q, axis=0)).max(axis=1)
    sp = arm_speed(qf, fps)[1:]
    moving = sp > moving_speed
    zero = (step == 0)
    mask = np.zeros(len(q), dtype=bool)
    mask[1:] = zero & moving
    return {"moving_frames": int(moving.sum()), "stale_frames": int((zero & moving).sum()),
            "stale_share_when_moving": float((zero & moving).sum() / max(moving.sum(), 1)),
            "zero_step_share_all": float(zero.mean()), "mask": mask}


def analyze(ep: Episode, sigma_s: float, arm: str, rules: dict) -> dict:
    fps = ep.fps
    t = ep.timestamp
    q = ep.arm(arm, "state")
    a = ep.arm(arm, "action")
    qf = smooth(q, sigma_s, fps)
    seg = motion_segments(qf, fps)
    b = branch_label(q)
    limits = urdf_limits(DEFAULT_URDF)
    err = None
    try:
        check_series(t, q, limits, np.radians(rules["limit_margin_deg"]), np.radians(rules["max_jump_deg"]))
    except PlanError as e:
        err = str(e)
    jb, ja = jitter_metrics(q, fps), jitter_metrics(qf, fps)
    rng = {JOINT_LABELS[j]: [float(np.degrees(q[:, j].min())), float(np.degrees(q[:, j].max()))] for j in range(6)}
    n = len(t)
    sp_f = arm_speed(qf, fps)
    sp_raw = np.zeros(n)
    sp_raw[1:] = np.abs(np.diff(q, axis=0)).max(axis=1) * fps
    plan = plan_pause_trim(qf, fps)
    still = sp_f < 0.02
    others = [x for x in ep.arms if x != arm]
    stale = stale_samples(q, qf, fps)
    return {
        "name": ep.name, "task": ep.task, "frames": int(n), "fps": fps,
        "duration_s": float(n / fps),
        "start_still_s": seg["start_still_s"], "end_still_s": seg["end_still_s"],
        "pauses": seg["pauses"], "pause_total_s": seg["pause_total_s"], "pause_max_s": seg["pause_max_s"],
        "pause_share": float(seg["pause_total_s"] * fps / n),
        "still_frames_filtered_share": float(still.mean()),
        "still_frames_raw_share": float((sp_raw < 0.02).mean()),
        "pause_trim_plan": {"removed_frames": plan["removed"], "removed_share": plan["removed"] / n,
                            "cuts": plan["cuts"], "skipped": plan["skipped"]},
        "branch": b, "start_pose": start_pose_distance(q[0]),
        "end_pose_deg": np.degrees(q[-1]).tolist(), "start_pose_deg": np.degrees(q[0]).tolist(),
        "joint_range_deg": rng,
        "limit_check": err or "ok",
        "limit_margin": joint_limit_margin_deg(q, limits),
        "corruption": {**speed_outliers(q, fps, rules["max_jump_deg"]), "zero_pose_frames": zero_pose_frames(q),
                       "max_filtered_rad_s": float(sp_f.max())},
        "stale": {k: v for k, v in stale.items() if k != "mask"},
        "box_pose": ep.meta.get("box_pose"), "source_meta": ep.meta, "episode_index": ep.episode_index,
        "jitter_before": jb, "jitter_after": ja,
        "jitter_before_mean": summarize_jitter(jb), "jitter_after_mean": summarize_jitter(ja),
        "filter_deviation": deviation_deg(q, qf),
        "other_arms_all_zero": {x: bool(np.abs(ep.arm(x, "state")).max() < 1e-9) for x in others},
        "_q": q, "_qf": qf, "_a": a, "_seg": seg, "_plan": plan, "_stale_mask": stale["mask"], "_sp_raw": sp_raw,
    }


def check_videos(ep: Episode) -> dict:
    from video_frames import count_packets
    out = {}
    for key, (path, _) in ep.videos.items():
        n = count_packets(path)
        out[key.split(".")[-1]] = {"frames": n, "matches_rows": n == len(ep.timestamp)}
    return out


def thumbnails(ep: Episode, shorts: list[str]) -> dict:
    from PIL import Image
    from video_frames import read_frames
    n = len(ep.timestamp)
    labs = ("first", "middle", "last")
    times = [ep.timestamp[0], ep.timestamp[n // 2], ep.timestamp[-1]]
    out = {}
    for short in shorts:
        key = f"observation.images.{short}"
        if key not in ep.videos:
            continue
        path, t0 = ep.videos[key]
        frames = read_frames(path, [t0 + float(x) for x in times], ep.fps)
        for lab, fr in zip(labs, frames):
            im = Image.fromarray(fr).resize((320, 180))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            out[f"{short}_{lab}"] = b64(buf.getvalue())
    return out


def video_consistency(ep: Episode, r: dict, short: str) -> dict | None:
    """Image motion of the camera ``short`` against the joint speed (a camera on the arm moves when the arm moves)."""
    from video_frames import motion_energy
    key = f"observation.images.{short}"
    if key not in ep.videos:
        return None
    me = motion_energy(ep.videos[key][0])
    n = min(me["n"], len(r["_q"]))
    out = video_state_consistency(me["motion"][:n], r["_sp_raw"][:n], ep.fps)
    out["camera"] = short
    out["video_frames"] = int(me["n"])
    mask = r["_stale_mask"][:n]
    moving = arm_speed(r["_qf"], ep.fps)[:n] > 0.05
    nz = moving & ~mask
    nz[0] = False
    if mask.any() and nz.any():
        out["video_motion_at_stale_frames"] = float(me["motion"][:n][mask].mean())
        out["video_motion_at_other_moving_frames"] = float(me["motion"][:n][nz].mean())
    mot = me["motion"][:n]
    out["_motion"] = mot
    return out


def dtw_features(res: list[dict], sigma_s: float) -> None:
    """Add DTW-to-medoid distances (deg, mean per aligned step) on the idle-trimmed filtered arm series."""
    series_full, series_j123 = [], []
    for r in res:
        qf, seg = r["_qf"], r["_seg"]
        i0, i1 = trim_bounds(seg, len(qf), r["fps"], 0.0)
        step = max(1, int(round(r["fps"] / DTW_HZ)))
        x = np.degrees(qf[i0:i1:step])
        series_full.append(x)
        series_j123.append(x[:, :3])
    for key, series in (("dtw_all_joints", series_full), ("dtw_j1_j3", series_j123)):
        d = dtw_matrix(series)
        med = int(np.argmin(d.sum(axis=1)))
        z = robust_z(d[med])
        for i, r in enumerate(res):
            r[key] = {"to_medoid_deg": float(d[med, i]), "medoid": res[med]["name"], "robust_z": float(z[i]),
                      "median_to_others_deg": float(np.median(np.delete(d[i], i)))}


def box_groups(res: list[dict], tol) -> list[list[str]]:
    """Groups (size >= 2) of episodes with the same box placement."""
    have = [r for r in res if r.get("box_pose")]
    groups, used = [], set()
    for i, r in enumerate(have):
        if i in used:
            continue
        g = [j for j, s in enumerate(have) if all(abs(a - b) <= t for a, b, t in zip(r["box_pose"], s["box_pose"], tol))]
        used.update(g)
        if len(g) > 1:
            groups.append([have[j]["name"] for j in g])
    return groups


def suggest(res: list[dict], arm: str, rules: dict) -> dict:
    """Fill ``flag`` ('keep'/'review'), ``reasons``, ``notes``, ``start_variant`` per episode; returns the cross-episode context."""
    labels = Counter(r["branch"]["label"] for r in res if r["branch"]["label"] != "neither")
    majority = labels.most_common(1)[0][0] if labels else "neither"
    zdur = robust_z(np.array([r["duration_s"] for r in res]))
    tasks = Counter(r["task"] for r in res)
    sv = cluster_poses(np.array([r["start_pose_deg"] for r in res]), rules["start_variant_tol_deg"])
    ev = cluster_poses(np.array([r["end_pose_deg"] for r in res]), rules["end_variant_tol_deg"])
    sv_n, ev_n = Counter(sv), Counter(ev)
    letter = lambda i: chr(ord("A") + i) if i < 26 else str(i)  # noqa: E731
    # name the variants by size (A = most common)
    s_order = {c: k for k, (c, _) in enumerate(sv_n.most_common())}
    e_order = {c: k for k, (c, _) in enumerate(ev_n.most_common())}
    variants = {}
    for c in sv_n:
        mem = [r for r, x in zip(res, sv) if x == c]
        variants[letter(s_order[c])] = {"episodes": [m["name"] for m in mem],
                                        "mean_pose_deg": np.mean([m["start_pose_deg"] for m in mem], axis=0).round(1).tolist()}
    big = letter(0)
    ref = np.array(variants[big]["mean_pose_deg"])
    left_ref = arm == "arm_left"
    groups = box_groups(res, rules["box_pose_tol"])
    for i, r in enumerate(res):
        r["start_variant"] = letter(s_order[sv[i]])
        r["end_variant"] = letter(e_order[ev[i]])
        r["start_pose_ref"] = "left majority (0, 0, 60, 0, 50, 0)" if left_ref else f"largest start variant {big}"
        if not left_ref:
            r["start_pose"] = start_pose_distance(np.radians(r["start_pose_deg"]), ref)
        r["box_duplicates"] = next((sorted(set(g) - {r["name"]}) for g in groups if r["name"] in g), [])
        why, notes = [], []
        lab = r["branch"]["label"]
        if lab == "both":
            why.append("uses both wrist branches in one episode (j4/j6 pass through the flip)")
        elif lab != majority and majority != "neither":
            why.append(f"wrist branch {lab} differs from the majority {majority} ({labels[lab]} of {len(res)} episodes)")
        sd = r["start_pose"]["max_abs_deg"]
        if left_ref:
            if sd > rules["start_pose_review_deg"]:
                why.append(f"starts {sd:.0f} deg (max joint) away from the majority start pose")
        else:
            near = min((float(np.abs(np.array(r["start_pose_deg"]) - np.array(x["start_pose_deg"])).max())
                        for j, x in enumerate(res) if j != i), default=0.0)
            r["start_pose_nearest_other_deg"] = near
            if near > rules["start_pose_review_deg"] and len(res) > 3:
                why.append(f"starts {near:.0f} deg (max joint) away from the start pose of every other episode "
                           f"(j = {', '.join(f'{x:.0f}' for x in r['start_pose_deg'])} deg)")
            elif r["start_variant"] != big:
                notes.append(f"start-pose variant {r['start_variant']} ({sv_n[sv[i]]} episodes; the largest variant {big} has {max(sv_n.values())})")
        if r["dtw_j1_j3"]["robust_z"] > rules["robust_z_review"]:
            why.append(f"j1-j3 path far from the medoid (DTW {r['dtw_j1_j3']['to_medoid_deg']:.1f} deg, "
                       f"robust z {r['dtw_j1_j3']['robust_z']:.1f})")
        if abs(zdur[i]) > rules["robust_z_review"]:
            why.append(f"duration {r['duration_s']:.0f} s is an outlier (robust z {zdur[i]:+.1f})")
        if r["start_still_s"] > rules["start_still_review_s"]:
            why.append(f"long idle start ({r['start_still_s']:.1f} s)")
        if r["end_still_s"] > rules["end_still_review_s"]:
            why.append(f"long idle end ({r['end_still_s']:.1f} s)")
        if r["pause_max_s"] >= rules["mid_pause_review_s"]:
            why.append(f"mid-episode pause of {r['pause_max_s']:.1f} s ({100 * r['pause_share']:.0f} % of the episode in pauses >= 1 s)")
        elif rules["pause_share_review"] is not None and r["pause_share"] >= rules["pause_share_review"]:
            why.append(f"{100 * r['pause_share']:.0f} % of the episode is spent in pauses >= 1 s")
        if r["filter_deviation"]["max_deg_all"] > rules["max_dev_review_deg"]:
            why.append(f"jitter filter deviates {r['filter_deviation']['max_deg_all']:.2f} deg from the raw state")
        c = r["corruption"]
        if c["n_events"]:
            ev0 = c["events"][0]
            why.append(f"CORRUPT: {c['n_events']} impossible joint step(s) above {rules['max_jump_deg']:g} deg/frame "
                       f"(largest {c['max_step_deg']:.1f} deg = {c['max_rad_s']:.1f} rad/s; first at t={ev0['t_s']} s, joint {ev0['joint']})")
        if c["zero_pose_frames"]:
            why.append(f"CORRUPT: {c['zero_pose_frames']} frame(s) with all six joints exactly 0 (state not yet published)")
        m = r["limit_margin"]
        if m["min_deg"] < rules["limit_margin_deg"]:
            why.append(f"{m['min_deg']:.1f} deg from the j{m['joint']} {m['side']} joint limit (closer than {rules['limit_margin_deg']:g} deg)")
        elif m["min_deg"] < rules["limit_note_deg"]:
            notes.append(f"{m['min_deg']:.1f} deg from the j{m['joint']} {m['side']} joint limit")
        if r["limit_check"] != "ok" and not (c["n_events"] or c["zero_pose_frames"] or m["min_deg"] < rules["limit_margin_deg"]):
            why.append("joint-limit / jump check failed: " + r["limit_check"])
        vid = r.get("videos", {})
        if vid and not all(v["matches_rows"] for v in vid.values()):
            why.append("video frame count differs from the parquet rows")
        if not all(r["other_arms_all_zero"].values()):
            why.append("columns of another arm are not all zero")
        if rules["end_pose_review_deg"] is not None and len(res) > 3:
            others = [np.array(x["end_pose_deg"]) for j, x in enumerate(res) if j != i]
            d = min(float(np.abs(np.array(r["end_pose_deg"]) - o).max()) for o in others)
            r["end_pose_nearest_other_deg"] = d
            if d > rules["end_pose_review_deg"]:
                why.append(f"ends {d:.0f} deg (max joint) away from the final pose of every other episode "
                           f"(j = {', '.join(f'{x:.0f}' for x in r['end_pose_deg'])} deg): not parked / cut short?")
        if r["box_duplicates"]:
            notes.append(f"same box pose as {', '.join(r['box_duplicates'])}: keep them on the same side of a train/val split")
        r["flag"] = "review" if why else "keep"
        r["reasons"] = why
        if tasks[r["task"]] < len(res) / 2 and len(tasks) > 1:
            notes.append(f"task string '{r['task']}' (minority; {len(tasks)} different strings)")
        r["notes"] = notes
    return {"majority_branch": majority, "branch_counts": dict(labels), "task_counts": dict(tasks),
            "start_variants": variants, "box_duplicate_groups": groups,
            "end_variants": {letter(e_order[c]): [r["name"] for r, x in zip(res, ev) if x == c] for c in ev_n}}


def suggest_holdout(res: list[dict], k: int = 2) -> dict | None:
    """Suggest ``k`` held-out episodes: typical (flag keep, small DTW / duration deviation, no long pause), spread over the
    box-pose range, never one member of an identical-box-pose group (those must stay on one side of the split).

    Score of a pair = spread in the box-pose plane (each axis scaled by its range over all episodes, capped so that the two
    extremes are not chosen: a held-out episode at the very edge would test extrapolation) minus an atypicality penalty.
    Returns ``{"pairs": [ranked candidates with the numbers], "chosen": [...], "why": text}`` or None without box poses."""
    import itertools
    have = [r for r in res if r.get("box_pose")]
    if len(have) < k + 2:
        return None
    pts = np.array([[r["box_pose"][0], r["box_pose"][2]] for r in have])        # x [m], yaw [deg] (y is within 3 cm)
    rng = np.maximum(pts.max(0) - pts.min(0), 1e-9)
    z = (pts - np.median(pts, axis=0)) / rng
    dtw = robust_z(np.array([r["dtw_j1_j3"]["to_medoid_deg"] for r in have]))
    dur = robust_z(np.array([r["duration_s"] for r in have]))
    dup = {r["name"]: bool(r["box_duplicates"]) for r in have}
    cand = [i for i, r in enumerate(have) if r["flag"] == "keep" and not dup[r["name"]]]
    relaxed = False
    if len(cand) < k:                    # too few clean episodes: allow review-flagged ones that are not corrupt
        cand = [i for i, r in enumerate(have) if not dup[r["name"]] and not any("CORRUPT" in w for w in r["reasons"])]
        relaxed = True
    if len(cand) < k:
        return None
    pairs = []
    for combo in itertools.combinations(cand, k):
        spread = float(np.linalg.norm(z[combo[0]] - z[combo[1]]))
        cov = min(spread, 0.6)
        edge = max(float(np.abs(z[i]).max()) for i in combo)                        # distance of the farthest member from the median pose
        atyp = float(np.mean([max(dtw[i], 0) + 0.5 * abs(dur[i]) for i in combo]))
        score = cov - 1.5 * max(edge - 0.45, 0) - 0.1 * atyp
        pairs.append({"episodes": [have[i]["name"] for i in combo], "box_poses": [have[i]["box_pose"] for i in combo],
                      "spread": round(spread, 3), "edge": round(edge, 3), "atypicality": round(atyp, 3), "score": round(score, 3)})
    pairs.sort(key=lambda p: -p["score"])
    return {"pairs": pairs[:5], "chosen": pairs[0]["episodes"], "n_candidates": len(cand), "relaxed": relaxed,
            "excluded_dup": [n for n, d in dup.items() if d],
            "excluded_review": [r["name"] for r in have if r["flag"] != "keep"]}


# ---------------------------------------------------------------------------------------------- plots
def short(name: str) -> str:
    return name.replace("mocap_left_", "").replace("real_", "")


def plot_overview(res: list[dict], majority: str, arm: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {majority: "tab:blue", "both": "tab:red", "neither": "tab:gray"}
    other = "tab:orange"
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    seen = set()
    for r in res:
        lab = r["branch"]["label"]
        c = colors.get(lab, other)
        tn = np.linspace(0, 1, len(r["_q"]))
        for ax, j in zip(axs, (3, 5)):
            ax.plot(tn, r["_q"][:, j], color=c, lw=1.1, alpha=0.85, label=f"{lab}" if lab not in seen else None)
        seen.add(lab)
        axs[0].annotate(short(r["name"]), (1.0, r["_q"][-1, 3]), fontsize=7, color=c, xytext=(3, 0), textcoords="offset points")
    axs[0].set_ylabel("j4 [rad]")
    axs[1].set_ylabel("j6 [rad]")
    axs[1].set_xlabel("normalized episode time")
    axs[0].set_title(f"{arm} j4 / j6, coloured by wrist branch (majority = {majority})")
    axs[0].legend(loc="best", fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    return fig_png(fig)


def plot_episode(r: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(r["_q"])) / r["fps"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for j in range(6):
        ax.plot(t, np.degrees(r["_q"][:, j]), lw=0.8, label=JOINT_LABELS[j])
    for p in r["pauses"]:
        ax.axvspan(p["t_s"], p["t_s"] + p["dur_s"], color="gold", alpha=0.35)
    for c in r["pause_trim_plan"]["cuts"]:
        ax.axvspan(c["removed_from"] / r["fps"], c["removed_to"] / r["fps"], color="red", alpha=0.35)
    for e in r["corruption"]["events"]:
        ax.axvline(e["t_s"], color="magenta", lw=1.2)
    ax.axvspan(0, r["start_still_s"], color="0.85")
    ax.axvspan(t[-1] - r["end_still_s"], t[-1], color="0.85")
    ax.set_xlabel("s")
    ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    ax.legend(ncol=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    return fig_png(fig, 100)


def plot_jitter(eps: list[Episode], res: list[dict], sigma_s: float) -> tuple[bytes, bytes, dict]:
    """Velocity PSD (state raw / state filtered / source action, averaged over episodes and joints 1-6 per joint)
    and a time-series zoom of one fast stretch.  Returns (psd png, zoom png, band table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    fps = eps[0].fps
    P = {"state raw": [], "state filtered": [], "source action": []}
    nper = min(256, min(len(r["_q"]) for r in res) - 1)
    for e, r in zip(eps, res):
        for lab, x in (("state raw", r["_q"]), ("state filtered", r["_qf"]), ("source action", r["_a"])):
            f, p = welch(np.diff(x, axis=0) * fps, fs=fps, nperseg=nper, axis=0)
            P[lab].append(p)
    P = {k: np.mean(v, axis=0) for k, v in P.items()}          # (F, 6)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for lab, st in (("state raw", "-"), ("state filtered", "-"), ("source action", "--")):
        axs[0].semilogy(f, P[lab].mean(axis=1), st, label=lab, lw=1.5)
    axs[0].axvline(HF_CUTOFF_HZ, color="k", ls=":", lw=1)
    axs[0].set_xlabel("Hz")
    axs[0].set_ylabel("velocity PSD [(rad/s)^2/Hz], mean of j1-j6 and episodes")
    axs[0].set_title(f"jitter band: state vs. source action (Gaussian sigma {sigma_s:g} s)")
    axs[0].legend()
    axs[0].grid(alpha=0.3, which="both")
    for j in (1, 3, 5):
        axs[1].semilogy(f, P["state raw"][:, j], lw=1.0, label=f"raw {JOINT_LABELS[j]}")
        axs[1].semilogy(f, P["state filtered"][:, j], lw=1.0, ls="--", label=f"filtered {JOINT_LABELS[j]}")
    axs[1].set_xlabel("Hz")
    axs[1].set_title("per joint (j2, j4, j6)")
    axs[1].legend(fontsize=7, ncol=2)
    axs[1].grid(alpha=0.3, which="both")
    psd_png = fig_png(fig)
    # band table: fraction of velocity variance per band
    df = f[1] - f[0]
    bands = [(0, 1), (1, 2), (2, 3), (3, 6), (6, 10), (10, 15)]
    table = {lab: {f"{lo}-{hi} Hz": float(P[lab][(f >= lo) & (f < hi)].sum() * df / 6) for lo, hi in bands} for lab in P}

    # time-series zoom on the fastest second of the median-duration episode (joint with the largest travel)
    k = int(np.argsort([r["duration_s"] for r in res])[len(res) // 2])
    r = res[k]
    jj = int(np.argmax(np.ptp(r["_q"], axis=0)))
    sp = np.abs(np.diff(r["_q"][:, jj])) * fps
    i = int(np.argmax(np.convolve(sp, np.ones(45) / 45, "same")))
    sl = slice(max(0, i - 45), i + 45)
    tt = np.arange(len(r["_q"]))[sl] / fps
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axs[0].plot(tt, np.degrees(r["_q"][sl, jj]), ".-", ms=3, lw=0.6, label=f"raw state j{jj + 1}")
    axs[0].plot(tt, np.degrees(r["_qf"][sl, jj]), "-", lw=1.5, label=f"filtered j{jj + 1}")
    axs[0].set_ylabel("deg")
    axs[0].legend()
    vr = np.gradient(r["_q"][:, jj], 1 / fps)[sl]
    vf = np.gradient(r["_qf"][:, jj], 1 / fps)[sl]
    axs[1].plot(tt, vr, ".-", ms=3, lw=0.6, label="raw velocity")
    axs[1].plot(tt, vf, lw=1.5, label="filtered velocity")
    axs[1].set_ylabel("rad/s")
    axs[1].set_xlabel("s")
    axs[1].legend()
    axs[0].set_title(f"{r['name']}: raw vs. filtered j{jj + 1} (the step pattern is the sampling jitter)")
    for ax in axs:
        ax.grid(alpha=0.3)
    return psd_png, fig_png(fig), table


def plot_box_pose(res: list[dict], hold: dict | None) -> bytes | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [r for r in res if r.get("box_pose")]
    if not have:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    chosen = set(hold["chosen"]) if hold else set()
    for r in have:
        x, y, yaw = r["box_pose"]
        col = "tab:green" if r["flag"] == "keep" else "tab:orange"
        for ax, (u, v) in zip(axs, ((x, yaw), (x, y))):
            ax.scatter([u], [v], c=col, s=70 if r["name"] in chosen else 30,
                       edgecolors="k" if r["name"] in chosen else "none", linewidths=2, zorder=3)
            ax.annotate(short(r["name"]), (u, v), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axs[0].set_xlabel("box x [m]")
    axs[0].set_ylabel("box yaw [deg]")
    axs[1].set_xlabel("box x [m]")
    axs[1].set_ylabel("box y [m]")
    axs[0].set_title("box pose per episode (green keep, orange review, black ring = suggested held-out)")
    for ax in axs:
        ax.grid(alpha=0.3)
    return fig_png(fig)


# ---------------------------------------------------------------------------------------------- html
def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def build_html(res: list[dict], ctx: dict, imgs: dict, sigma_s: float, source: str, band_table: dict, agg: dict,
               arm: str, hold: dict | None, thumb_shorts: list[str], title: str, jitter_ctx: dict) -> str:
    esc = html.escape
    css = ("body{font-family:sans-serif;margin:20px;max-width:1500px}table{border-collapse:collapse;font-size:12px}"
           "td,th{border:1px solid #bbb;padding:3px 6px;text-align:right}th{background:#eee}td.l,th.l{text-align:left}"
           ".keep{background:#d8f0d8}.review{background:#ffe0b0}.card{border:1px solid #bbb;margin:14px 0;padding:8px}"
           ".th img{margin:2px;border:1px solid #888}.card h3{margin:2px 0}small{color:#555}")
    has_box = any(r.get("box_pose") for r in res)
    h = [f"<html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{css}</style></head><body>",
         f"<h1>{esc(title)}</h1>",
         f"<p>Source: <code>{esc(source)}</code>. Generated {time.strftime('%Y-%m-%d %H:%M')}. "
         f"{len(res)} episodes, arm <b>{esc(arm)}</b> j1..j6. Flags are suggestions, the operator decides.</p>"]
    h.append("<h2>Summary</h2><table><tr><th class='l'>episode</th><th>dur s</th><th>start still s</th>"
             "<th>end still s</th><th>pauses &ge;1 s (n / total s / max s)</th><th>pause share %</th><th>raw 'slow' frames %</th>"
             "<th>pause trim (frames / %)</th>"
             + ("<th class='l'>box pose x, y, yaw</th>" if has_box else "")
             + "<th class='l'>start / end variant</th><th>min limit margin deg</th><th>max step deg/frame</th>"
             + "<th class='l'>wrist branch</th><th>DTW j1-3 deg</th><th>HF rms deg before &rarr; after</th>"
               "<th>max filter dev deg</th><th class='l'>suggestion</th></tr>")
    for r in res:
        jb, ja = r["jitter_before_mean"], r["jitter_after_mean"]
        h.append(f"<tr class='{r['flag']}'><td class='l'><a href='#{esc(r['name'])}'>{esc(r['name'])}</a></td>"
                 f"<td>{fmt(r['duration_s'], 1)}</td><td>{fmt(r['start_still_s'], 1)}</td>"
                 f"<td>{fmt(r['end_still_s'], 1)}</td><td>{len(r['pauses'])} / {fmt(r['pause_total_s'], 1)} / {fmt(r['pause_max_s'], 1)}</td>"
                 f"<td>{100 * r['pause_share']:.1f}</td><td>{100 * r['still_frames_raw_share']:.0f}</td>"
                 f"<td>{r['pause_trim_plan']['removed_frames']} / {100 * r['pause_trim_plan']['removed_share']:.1f}</td>"
                 + (f"<td class='l'>{', '.join(f'{v:g}' for v in r['box_pose']) if r.get('box_pose') else ''}</td>" if has_box else "")
                 + f"<td class='l'>{r['start_variant']} / {r['end_variant']}</td>"
                 f"<td>{fmt(r['limit_margin']['min_deg'], 1)} (j{r['limit_margin']['joint']})</td>"
                 f"<td>{fmt(r['corruption']['max_step_deg'], 2)}</td>"
                 + f"<td class='l'>{esc(r['branch']['label'])}</td>"
                 f"<td>{fmt(r['dtw_j1_j3']['to_medoid_deg'], 1)}</td>"
                 f"<td>{fmt(jb['hf_rms_deg'], 4)} &rarr; {fmt(ja['hf_rms_deg'], 4)}</td>"
                 f"<td>{fmt(r['filter_deviation']['max_deg_all'], 2)}</td>"
                 f"<td class='l'><b>{r['flag']}</b>{'<br>' + '<br>'.join(esc(w) for w in r['reasons']) if r['reasons'] else ''}"
                 f"{'<br><small>' + '<br>'.join(esc(w) for w in r['notes']) + '</small>' if r['notes'] else ''}</td></tr>")
    h.append("</table>")
    h.append(f"<p>Majority wrist branch: <b>{esc(ctx['majority_branch'])}</b> {esc(str(ctx['branch_counts']))}. "
             f"Tasks: {esc(str(ctx['task_counts']))}. Medoid (DTW): <b>{esc(res[0]['dtw_all_joints']['medoid'])}</b>. "
             f"DTW = mean per-step joint-space distance after alignment on the idle-trimmed filtered state at {DTW_HZ:g} Hz; "
             f"'j1-3' ignores the wrist (branch-independent). Pause = filtered arm speed &le; 0.02 rad/s for &ge; 1 s between the first "
             f"and last movement; 'raw slow frames' = frames whose unfiltered max joint speed is &lt; 0.02 rad/s (includes stale state samples, see jitter). "
             f"Pause trim = what <code>preprocess.py --trim-pauses</code> would remove (settled part of pauses, keep 0.4 s).</p>")
    h.append("<p>Start-pose variants (poses within %g deg): %s. Final-pose variants: %s.</p>" % (
        FLAG_RULES["start_variant_tol_deg"],
        "; ".join(f"<b>{k}</b> = {', '.join(short(n) for n in v['episodes'])} (mean {', '.join(f'{x:g}' for x in v['mean_pose_deg'])} deg)"
                  for k, v in ctx["start_variants"].items()),
        "; ".join(f"<b>{k}</b> = {', '.join(short(n) for n in v)}" for k, v in ctx["end_variants"].items())))
    h.append(f"<h2>j4 / j6 over normalized time</h2><img src='data:image/png;base64,{imgs['overview']}'>")
    if "box" in imgs:
        h.append("<h2>Box pose and suggested held-out pair</h2>")
        h.append(f"<img src='data:image/png;base64,{imgs['box']}'>")
        if ctx["box_duplicate_groups"]:
            h.append("<p>Identical box placements (must stay on one side of a split): " +
                     "; ".join(esc(", ".join(g)) for g in ctx["box_duplicate_groups"]) + ".</p>")
        if hold:
            h.append(f"<p><b>Suggested held-out pair: {esc(', '.join(hold['chosen']))}.</b> {esc(hold['text'])}</p>")
            h.append("<table><tr><th class='l'>rank</th><th class='l'>pair</th><th class='l'>box poses (x, y, yaw)</th><th>spread</th>"
                     "<th>edge</th><th>atypicality</th><th>score</th></tr>")
            for i, p in enumerate(hold["pairs"]):
                h.append(f"<tr><td class='l'>{i + 1}</td><td class='l'>{esc(', '.join(p['episodes']))}</td>"
                         f"<td class='l'>{esc('; '.join(', '.join(f'{v:g}' for v in bp) for bp in p['box_poses']))}</td>"
                         f"<td>{p['spread']}</td><td>{p['edge']}</td><td>{p['atypicality']}</td><td>{p['score']}</td></tr>")
            h.append("</table>")
    h.append("<h2>Jitter analysis</h2>")
    h.append(f"<p>Filter: zero-phase endpoint-preserving Gaussian, sigma {sigma_s:g} s (-3 dB at {0.1325 / sigma_s:.1f} Hz). "
             f"Mean over the {jitter_ctx.get('n_episodes', len(res))} episodes without a CORRUPT flag and over joints. HF = above {HF_CUTOFF_HZ:g} Hz.</p>")
    h.append("<table><tr><th class='l'>metric (mean j1-j6)</th><th>raw</th><th>filtered</th><th>reduction</th></tr>")
    for k, label in (("hf_rms_deg", "position HF rms [deg]"), ("vel_hf_rms", "velocity HF rms [rad/s]"),
                     ("jerk_rms", "jerk rms [rad/s^3]"), ("reversals_per_s", "velocity sign reversals per s of motion"),
                     ("stalls_per_s", "zero-step frames per s of motion")):
        b, a = agg["before"][k], agg["after"][k]
        h.append(f"<tr><td class='l'>{label}</td><td>{b:.4g}</td><td>{a:.4g}</td><td>x{b / max(a, 1e-12):.1f}</td></tr>")
    h.append("</table>")
    h.append("<p>Velocity variance per band [(rad/s)^2, mean over j1-j6]:</p><table><tr><th class='l'>signal</th>" +
             "".join(f"<th>{esc(b)}</th>" for b in next(iter(band_table.values()))) + "</tr>")
    for lab, row in band_table.items():
        h.append(f"<tr><td class='l'>{lab}</td>" + "".join(f"<td>{v:.2e}</td>" for v in row.values()) + "</tr>")
    h.append("</table>")
    if jitter_ctx:
        h.append("<p>Stale state samples: " + esc(jitter_ctx["text"]) + "</p>")
    h.append(f"<img src='data:image/png;base64,{imgs['psd']}'><br><img src='data:image/png;base64,{imgs['zoom']}'>")
    h.append("<h2>Episodes</h2>")
    for r in res:
        th = r["thumbs"]
        h.append(f"<div class='card {r['flag']}' id='{esc(r['name'])}'><h3>{esc(r['name'])} &mdash; {r['flag']}</h3>"
                 f"<p>{esc(r['task'])} | {r['frames']} frames, {fmt(r['duration_s'], 1)} s | "
                 + (f"box pose {', '.join(f'{v:g}' for v in r['box_pose'])} | " if r.get('box_pose') else "")
                 + f"start variant {r['start_variant']} | branch {esc(r['branch']['label'])} | "
                 f"limit margin {fmt(r['limit_margin']['min_deg'], 1)} deg (j{r['limit_margin']['joint']}) | videos: " +
                 ", ".join(f"{k} {v['frames']}{'' if v['matches_rows'] else ' (MISMATCH)'}" for k, v in r['videos'].items()) +
                 f"</p><p>{'; '.join(esc(w) for w in r['reasons']) or 'no concerns'}</p><div class='th'>")
        for s in thumb_shorts:
            for lab in ("first", "middle", "last"):
                if f"{s}_{lab}" in th:
                    h.append(f"<img title='{s} {lab}' src='data:image/jpeg;base64,{th[f'{s}_{lab}']}'>")
            h.append("<br>")
        h.append(f"</div><img src='data:image/png;base64,{r['plot']}'><br><small>grey = idle head/tail, yellow = mid pauses, red = frames "
                 f"that --trim-pauses removes, magenta = impossible joint step. "
                 f"Joint ranges [deg]: " + ", ".join(f"{k} {v[0]:+.0f}..{v[1]:+.0f}" for k, v in r['joint_range_deg'].items()) +
                 f"; start pose delta [deg] {', '.join(f'{x:+.0f}' for x in r['start_pose']['delta_deg'])} ({esc(r['start_pose_ref'])}); "
                 f"max filter deviation per joint [deg] {', '.join(f'{x:.2f}' for x in r['filter_deviation']['max_deg'])}</small></div>")
    h.append("</body></html>")
    return "\n".join(h)


def aggregate_jitter(res: list[dict]) -> dict:
    out = {}
    for side in ("before", "after"):
        keys = res[0][f"jitter_{side}_mean"].keys()
        out[side] = {k: float(np.mean([r[f"jitter_{side}_mean"][k] for r in res])) for k in keys}
    return out


def holdout_text(hold: dict, res: list[dict]) -> str:
    by = {r["name"]: r for r in res}
    a, b = hold["chosen"]
    ra, rb = by[a], by[b]
    xs = [r["box_pose"][0] for r in res if r.get("box_pose")]
    ys = [r["box_pose"][2] for r in res if r.get("box_pose")]
    top = hold["pairs"][0]
    flagged = "both flagged keep" if not hold.get("relaxed") else "not corrupt (too few episodes are flagged keep, so review-flagged ones were allowed)"
    return (f"{a} (box x {ra['box_pose'][0]:g} m, yaw {ra['box_pose'][2]:g} deg) and {b} (box x {rb['box_pose'][0]:g} m, yaw "
            f"{rb['box_pose'][2]:g} deg) are {flagged}, have a typical path (DTW to the medoid {ra['dtw_j1_j3']['to_medoid_deg']:.1f} / "
            f"{rb['dtw_j1_j3']['to_medoid_deg']:.1f} deg), no long pause, no near-limit approach, and a box placement that appears "
            f"once only (not one of the identical-placement groups, episodes {hold['excluded_dup']}). They are {top['spread']:.2f} of the "
            f"x/yaw range apart (all episodes: x {min(xs):g}..{max(xs):g} m, yaw {min(ys):g}..{max(ys):g} deg); the farther one lies "
            f"{top['edge']:.2f} of the range from the median pose (0.45 or less = interior, the training episodes surround it, so the "
            f"held-out episodes test interpolation rather than extrapolation). {hold['n_candidates']} episodes were eligible.")


def parse_rules(items: list[str]) -> dict:
    rules = json.loads(json.dumps(FLAG_RULES))
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in rules:
            raise SystemExit(f"--rule: unknown rule '{k}'; known: {', '.join(rules)}")
        rules[k] = None if v.lower() in ("none", "off", "") else (json.loads(v) if v.startswith("[") else float(v))
    return rules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma-s", type=float, default=DEFAULT_SIGMA_S, help="jitter filter sigma [s] (default %(default)s)")
    ap.add_argument("--arm", help="arm to analyse (default: the arm that moves)")
    ap.add_argument("--no-thumbs", action="store_true", help="skip video decoding (thumbnails, frame counts)")
    ap.add_argument("--thumb-videos", help="comma-separated camera names for the thumbnails (default head + the working arm's wrist camera)")
    ap.add_argument("--video-consistency", metavar="CAMERA",
                    help="decode this camera completely: image motion vs. arm speed (numbers in report.json) and the image motion at stale-state "
                         "frames (used in the jitter text)")
    ap.add_argument("--rule", action="append", metavar="NAME=VALUE", help="override a flag rule (see FLAG_RULES in report.json); repeatable")
    ap.add_argument("--title", default="Mocap flap dataset: per-episode quality report")
    ap.add_argument("--holdout", type=int, default=2, help="size of the suggested held-out set when episodes carry a box_pose (0 = off)")
    args = ap.parse_args(argv)
    rules = parse_rules(args.rule)

    refs = list_episodes(args.source_root)
    args.out.mkdir(parents=True, exist_ok=True)
    eps = [load_episode(r.root, r.episode_index) for r in refs]
    arm = args.arm or Counter(e.active_arm() for e in eps).most_common(1)[0][0]
    side = arm.split("_")[-1]
    thumb_shorts = args.thumb_videos.split(",") if args.thumb_videos else ["head", f"wrist_{side}"]
    print(f"{len(eps)} episodes from {args.source_root}, arm {arm}")
    res = []
    for e in eps:
        r = analyze(e, args.sigma_s, arm, rules)
        r["videos"] = {} if args.no_thumbs else check_videos(e)
        r["thumbs"] = {} if args.no_thumbs else thumbnails(e, thumb_shorts)
        if args.video_consistency:
            r["video_consistency"] = video_consistency(e, r, args.video_consistency)
        res.append(r)
        print(f"  {r['name']}: {r['duration_s']:.1f} s, still {r['start_still_s']:.1f}/{r['end_still_s']:.1f} s, "
              f"{r['branch']['label']}, dev max {r['filter_deviation']['max_deg_all']:.2f} deg"
              , flush=True)
    dtw_features(res, args.sigma_s)
    ctx = suggest(res, arm, rules)
    hold = suggest_holdout(res, args.holdout) if args.holdout else None
    if hold:
        hold["text"] = holdout_text(hold, res)
    imgs = {"overview": plot_overview(res, ctx["majority_branch"], arm)}
    sound = [i for i, r in enumerate(res) if not any("CORRUPT" in w for w in r["reasons"])] or list(range(len(res)))
    psd, zoom, band_table = plot_jitter([eps[i] for i in sound], [res[i] for i in sound], args.sigma_s)
    imgs.update(psd=psd, zoom=zoom)
    box = plot_box_pose(res, hold)
    if box:
        imgs["box"] = box
        (args.out / "box_pose.png").write_bytes(box)
    for r in res:
        r["plot_png"] = plot_episode(r)
        r["plot"] = b64(r["plot_png"])
        (args.out / f"ep_{r['name']}.png").write_bytes(r["plot_png"])
    (args.out / "overview_j4_j6.png").write_bytes(imgs["overview"])
    (args.out / "jitter_psd.png").write_bytes(psd)
    (args.out / "jitter_timeseries.png").write_bytes(zoom)
    imgs = {k: b64(v) for k, v in imgs.items()}
    agg = aggregate_jitter([res[i] for i in sound])
    sm = [res[i]["stale"] for i in sound]
    jitter_ctx = {"n_episodes": len(sound), "excluded": [res[i]["name"] for i in range(len(res)) if i not in sound],
                  "stale_share_when_moving_mean": float(np.mean([s["stale_share_when_moving"] for s in sm])),
                  "zero_step_share_all_mean": float(np.mean([s["zero_step_share_all"] for s in sm]))}
    vcs = [res[i]["video_consistency"] for i in sound if res[i].get("video_consistency")
           and "video_motion_at_stale_frames" in res[i]["video_consistency"]]
    ratio = None
    if vcs:
        ratio = float(np.mean([v["video_motion_at_stale_frames"] / max(v["video_motion_at_other_moving_frames"], 1e-9) for v in vcs]))
        jitter_ctx["video_motion_ratio_stale_vs_other_moving"] = ratio
    jitter_ctx["text"] = (f"in {100 * jitter_ctx['stale_share_when_moving_mean']:.0f} % of the frames in which the arm is moving, every joint step "
                          f"is exactly zero (a repeated state sample, i.e. the joint-state stream is slower than the 30 Hz recorder)"
                          + (f"; the camera image still moves in those frames ({ratio:.2f}x the image motion of the other moving frames), "
                             f"so the arm moved and the state value is stale." if ratio is not None else "."))
    (args.out / "report.html").write_text(build_html(res, ctx, imgs, args.sigma_s, str(args.source_root), band_table, agg, arm,
                                                     hold, thumb_shorts, args.title, jitter_ctx))
    clean = [{k: v for k, v in r.items() if not k.startswith("_") and k not in ("thumbs", "plot", "plot_png")} for r in res]
    for c in clean:
        if c.get("video_consistency"):
            c["video_consistency"] = {k: v for k, v in c["video_consistency"].items() if not k.startswith("_")}
    (args.out / "report.json").write_text(json.dumps(
        {"source": str(args.source_root.expanduser().resolve()), "arm": arm, "sigma_s": args.sigma_s, "hf_cutoff_hz": HF_CUTOFF_HZ,
         "start_pose_deg": START_POSE_DEG.tolist(), "flag_rules": rules, "context": ctx, "holdout": hold,
         "jitter_mean_over_episodes": agg, "jitter_context": jitter_ctx,
         "velocity_band_variance": band_table, "episodes": clean}, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {args.out}/report.html, report.json")
    for r in res:
        print(f"{r['flag']:6s} {r['name']:18s} {'; '.join(r['reasons'])}")
    if hold:
        print("held-out suggestion:", hold["chosen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
