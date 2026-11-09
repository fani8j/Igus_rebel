#!/usr/bin/env python3
"""Shared analysis helpers for the mocap flap dataset: jitter filter + metrics, idle/pause detection,
wrist-branch label, start-pose distance, DTW.  numpy/scipy only (no lerobot, no ROS).

Used by quality_report.py (read-only report) and preprocess.py (writes the clean dataset), so both
use the exact same filter and the same definitions.

Jitter analysis behind the defaults (2026-10-29, the 12 episodes; see README "Preprocess for VLA training"):
the measured ``observation.state`` is a 30 Hz sample of a joint-state stream that is not phase-locked to the
recorder, so the per-frame step alternates e.g. 0.36, 0.24, 0.36, 0.73 deg while the arm moves smoothly.
That shows as velocity-spectrum lines at about 6.3 and 9.1 Hz (and a broad 3-15 Hz floor) that the teleop
``action`` (a smooth, rate-limited command) does not have; the intended motion is below about 2 Hz.
A zero-phase Gaussian (``replay_episode.gaussian_smooth`` with odd reflection, which keeps both endpoints
exactly) with sigma 0.08 s (-3 dB at 1.7 Hz, -13 dB at the ~3.5 Hz ripple, -43 dB at the 6.3 Hz line) removes
it.  sigma 0.06 leaves 40 % of the 3.5 Hz ripple; sigma 0.1 starts to cost tracking of the real motion
(the deviation from the raw state is 0.55 / 0.62 / 0.65 deg max for sigma 0.06 / 0.08 / 0.1, and the same
corner rounding is already present when the filter is applied to the clean teleop action).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from replay_episode import gaussian_smooth  # noqa: E402  (reused, not duplicated)

# ----------------------------------------------------------------------------------------- constants
DEFAULT_SIGMA_S = 0.08          # filter sigma [s]; -3 dB at 0.1325/sigma = 1.7 Hz, -13 dB at 3.5 Hz, -43 dB at 6.3 Hz
HF_CUTOFF_HZ = 3.0              # "jitter band" starts here (intended motion is < 2 Hz, the action has no power above)
MOVING_SPEED = 0.02             # rad/s: a joint (or the arm) counts as moving above this (smoothed)
STILL_SPEED = 0.02              # rad/s: max over joints of the filtered speed below this = arm is still
START_POSE_DEG = np.array([0.0, 0.0, 60.0, 0.0, 50.0, 0.0])   # majority start pose of the recordings
WRIST_BRANCH_RAD = 2.0          # |j4| / |j6| extreme above this = went to the "flipped" side
JOINT_LABELS = [f"j{i}" for i in range(1, 7)]


# --------------------------------------------------------------------------------------------- filter
def joint_columns(names: list[str]) -> list[int]:
    """Column indices of the revolute joints (``*_joint1..6``); gripper slots are not filtered."""
    return [i for i, n in enumerate(names) if "_joint" in n]


def smooth(q: np.ndarray, sigma_s: float, fps: float) -> np.ndarray:
    """Zero-phase, endpoint-preserving Gaussian low-pass of an (N, k) signal sampled at ``fps`` (float64).

    Wraps ``replay_episode.gaussian_smooth`` (odd reflection about both end samples => the first and last
    sample are returned unchanged, no lag).  ``sigma_s <= 0`` returns a copy."""
    q = np.asarray(q, dtype=np.float64)
    if sigma_s <= 0:
        return q.copy()
    return gaussian_smooth(q, float(sigma_s), float(fps))


def filter_state(state: np.ndarray, names: list[str], sigma_s: float, fps: float) -> np.ndarray:
    """Filter the joint columns of an (N, D) state; gripper columns are copied unchanged. float64 out."""
    out = np.asarray(state, dtype=np.float64).copy()
    cols = joint_columns(names)
    out[:, cols] = smooth(out[:, cols], sigma_s, fps)
    return out


def shifted_action(filtered: np.ndarray, offset_frames: int, names: list[str]) -> np.ndarray:
    """action[t] = filtered[t + offset] (the last frames repeat the final row); gripper columns 0."""
    n = len(filtered)
    idx = np.minimum(np.arange(n) + int(offset_frames), n - 1)
    act = filtered[idx].copy()
    for i, nm in enumerate(names):
        if "_joint" not in nm:
            act[:, i] = 0.0
    return act


def deviation_deg(raw: np.ndarray, filt: np.ndarray) -> dict:
    """Per-joint max and RMS |filtered - raw| in degrees (columns of (N, 6) arrays)."""
    d = np.degrees(np.abs(np.asarray(filt) - np.asarray(raw)))
    return {"max_deg": d.max(axis=0).tolist(), "rms_deg": np.sqrt((d ** 2).mean(axis=0)).tolist(),
            "max_deg_all": float(d.max())}


# --------------------------------------------------------------------------------------- jitter metrics
def _highpass(x: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    sos = butter(4, cutoff_hz, "high", fs=fps, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _boxcar(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return np.column_stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])])


def jitter_metrics(q: np.ndarray, fps: float = 30.0, hf_cutoff_hz: float = HF_CUTOFF_HZ,
                   moving_speed: float = MOVING_SPEED) -> dict:
    """Jitter numbers per joint for an (N, 6) joint series in rad.

    * ``hf_rms_deg``      RMS of position above ``hf_cutoff_hz`` (4th-order zero-phase Butterworth high-pass).
    * ``vel_hf_rms``      RMS of velocity above ``hf_cutoff_hz`` [rad/s].
    * ``jerk_rms``        RMS of the third difference * fps^3 [rad/s^3].
    * ``reversals_per_s`` sign changes of the (non-zero) frame step while the joint moves, per second of motion.
    * ``stalls_per_s``    frames with exactly zero step while the joint moves (stair-stepping), per second of motion.
    """
    q = np.asarray(q, dtype=np.float64)
    hf = _highpass(q, fps, hf_cutoff_hz)
    v = np.diff(q, axis=0) * fps
    vhf = _highpass(v, fps, hf_cutoff_hz)
    jerk = np.diff(q, 3, axis=0) * fps ** 3
    sm = _boxcar(np.abs(v), max(3, int(round(0.5 * fps))))
    moving = sm > moving_speed
    rev, stall = [], []
    for j in range(q.shape[1]):
        m = moving[:, j]
        secs = max(m.sum() / fps, 1e-9)
        s = np.sign(np.diff(q[:, j]))
        nz = np.flatnonzero(s != 0)
        flips = s[nz][1:] != s[nz][:-1]
        rev.append(float(flips[m[nz[1:]]].sum() / secs) if m.any() else 0.0)
        stall.append(float(((s == 0) & m).sum() / secs) if m.any() else 0.0)
    return {
        "hf_rms_deg": np.degrees(np.sqrt((hf ** 2).mean(axis=0))).tolist(),
        "vel_hf_rms": np.sqrt((vhf ** 2).mean(axis=0)).tolist(),
        "jerk_rms": np.sqrt((jerk ** 2).mean(axis=0)).tolist(),
        "reversals_per_s": rev,
        "stalls_per_s": stall,
    }


def summarize_jitter(m: dict) -> dict:
    """Mean over the six joints of every jitter metric (one number per episode for tables)."""
    return {k: float(np.mean(v)) for k, v in m.items()}


# ------------------------------------------------------------------------------------ motion segmentation
def arm_speed(q_filtered: np.ndarray, fps: float) -> np.ndarray:
    """(N,) max over joints of |dq/dt| [rad/s] (central difference) of an already filtered (N, 6) series."""
    return np.abs(np.gradient(q_filtered, 1.0 / fps, axis=0)).max(axis=1)


def still_runs(sp: np.ndarray, i0: int, i1: int, still_speed: float, min_frames: int) -> list[tuple[int, int]]:
    """Runs [a, b) of frames in [i0, i1] whose speed is <= ``still_speed`` and that last at least ``min_frames``.
    ``sp[i1]`` must be above the threshold (it is the last moving frame), so every run is closed inside the range."""
    runs, run = [], None
    for i in range(i0, i1 + 1):
        if sp[i] <= still_speed:
            run = i if run is None else run
        elif run is not None:
            if i - run >= min_frames:
                runs.append((run, i))
            run = None
    return runs


def motion_segments(q_filtered: np.ndarray, fps: float, still_speed: float = STILL_SPEED,
                    pause_min_s: float = 1.0) -> dict:
    """Idle head/tail and mid-episode pauses of an (N, 6) filtered joint series.

    ``start_still_s``/``end_still_s``: time before the first / after the last frame whose arm speed exceeds
    ``still_speed``.  ``pauses``: runs of at least ``pause_min_s`` below the threshold strictly between them
    (``pause_runs`` has the same runs as frame ranges [a, b))."""
    sp = arm_speed(q_filtered, fps)
    mov = np.flatnonzero(sp > still_speed)
    n = len(sp)
    if len(mov) == 0:
        return {"first_move": n, "last_move": -1, "start_still_s": n / fps, "end_still_s": 0.0, "pauses": [],
                "pause_total_s": 0.0, "pause_max_s": 0.0, "pause_runs": []}
    i0, i1 = int(mov[0]), int(mov[-1])
    # (i - run) / fps >= pause_min_s as before, written with an integer frame count (no float rounding at the edge)
    runs = still_runs(sp, i0, i1, still_speed, int(np.ceil(pause_min_s * fps - 1e-9)))
    pauses = [(a / fps, (b - a) / fps) for a, b in runs]
    return {"first_move": i0, "last_move": i1, "start_still_s": i0 / fps, "end_still_s": (n - 1 - i1) / fps,
            "pauses": [{"t_s": round(t, 3), "dur_s": round(d, 3)} for t, d in pauses],
            "pause_total_s": float(sum(d for _, d in pauses)), "pause_max_s": float(max([d for _, d in pauses], default=0.0)),
            "pause_runs": [[int(a), int(b)] for a, b in runs]}


# ------------------------------------------------------------------------------------------ pause trimming
PAUSE_MIN_S = 1.0         # a still run (arm speed <= STILL_SPEED) at least this long is a "pause" that may be shortened
PAUSE_KEEP_S = 0.4        # ... to this much of its settled part (half after the arm settled, half before it moves again)
PAUSE_CORE_SPEED = 0.005  # rad/s: only the SETTLED part of a pause (speed <= this) is cut; the exponential settling
#                           tail (0.02 -> 0.005 rad/s, ~1 s, up to 0.6 deg of creep) stays, it is the natural stop
PAUSE_MAX_JOIN_DEG = 0.25  # a cut is only made if the filtered pose on both sides differs by at most this (max joint)
PAUSE_MIN_CUT_S = 0.3     # cuts that would remove less than this are not worth a splice


def plan_pause_trim(q_filtered: np.ndarray, fps: float, still_speed: float = STILL_SPEED, min_pause_s: float = PAUSE_MIN_S,
                    keep_s: float = PAUSE_KEEP_S, max_join_deg: float = PAUSE_MAX_JOIN_DEG,
                    core_speed: float = PAUSE_CORE_SPEED, min_cut_s: float = PAUSE_MIN_CUT_S) -> dict:
    """Shorten the settled part of every mid-episode pause to ``keep_s``.

    Input is the arm's (N, 6) filtered joint series.  A *pause* is a run of at least ``min_pause_s`` with arm speed
    <= ``still_speed`` strictly between the first and the last movement (``motion_segments``).  Inside a pause the arm
    settles exponentially (speed 0.02 -> 0.005 rad/s takes ~1 s and moves the joints by up to 0.6 deg); only the
    sub-run with speed <= ``core_speed`` is cut: its first ``head`` and last ``tail`` frames (head + tail =
    round(keep_s * fps)) stay and the middle is removed.  So the arm still decelerates into the stop, rests, and moves out
    exactly as recorded; only frames at which it is (practically) not moving disappear.  A cut is skipped (listed in
    ``skipped``) when it would remove less than ``min_cut_s`` or when the pose after the removed span differs from the
    pose before it by more than ``max_join_deg`` in any joint, so the motion itself is never altered by more than that
    (the default 0.25 deg is about one ordinary frame step at the median speed).

    Returns ``{"keep": int array of kept frame indices, "cuts": [...], "skipped": [...], "removed": int, ...}``."""
    n = len(q_filtered)
    seg = motion_segments(q_filtered, fps, still_speed, min_pause_s)
    sp = arm_speed(q_filtered, fps)
    keep_frames = int(round(keep_s * fps))
    head = keep_frames // 2
    tail = keep_frames - head
    min_remove = max(1, int(np.ceil(min_cut_s * fps - 1e-9)))
    mask = np.ones(n, dtype=bool)
    cuts, skipped = [], []
    qd = np.degrees(np.asarray(q_filtered, dtype=np.float64))
    for a, b in seg["pause_runs"]:
        i = a
        while i < b:                                   # maximal runs of settled frames inside the pause
            if sp[i] > core_speed:
                i += 1
                continue
            j = i
            while j < b and sp[j] <= core_speed:
                j += 1
            r0, r1 = i + head, j - tail                # removed frames [r0, r1)
            if r1 - r0 >= min_remove and r0 > 0:
                join = float(np.abs(qd[r1] - qd[r0 - 1]).max())
                info = {"pause_start_frame": int(a), "pause_s": round((b - a) / fps, 3), "settled_start_frame": int(i),
                        "settled_s": round((j - i) / fps, 3), "removed_from": int(r0), "removed_to": int(r1),
                        "removed_frames": int(r1 - r0), "join_step_deg": round(join, 4)}
                if join > max_join_deg:
                    skipped.append({**info, "why": f"pose drifts {join:.2f} deg across the cut (> {max_join_deg} deg)"})
                else:
                    mask[r0:r1] = False
                    cuts.append(info)
            i = j
    keep = np.flatnonzero(mask)
    return {"keep": keep, "cuts": cuts, "skipped": skipped, "removed": int(n - len(keep)),
            "head_frames": head, "tail_frames": tail}


def kept_ranges(keep: np.ndarray) -> list[list[int]]:
    """Sorted frame indices -> list of [start, stop) runs (compact description of what was kept)."""
    keep = np.asarray(keep, dtype=np.int64)
    if len(keep) == 0:
        return []
    cut = np.flatnonzero(np.diff(keep) != 1)
    starts = np.concatenate([[0], cut + 1])
    stops = np.concatenate([cut + 1, [len(keep)]])
    return [[int(keep[a]), int(keep[b - 1]) + 1] for a, b in zip(starts, stops)]


# ----------------------------------------------------------------------------------------- corruption checks
def speed_outliers(q: np.ndarray, fps: float, max_jump_deg: float = 5.7) -> dict:
    """Frame-to-frame joint steps above ``max_jump_deg`` (5.7 deg/frame = 3.0 rad/s at 30 fps).

    ``events``: [{frame, t_s, joint (1-6), step_deg, rad_s}], ``max_rad_s``: largest |dq| * fps over the whole series."""
    step = np.degrees(np.abs(np.diff(np.asarray(q, dtype=np.float64), axis=0)))
    bad = np.argwhere(step > max_jump_deg)
    events = [{"frame": int(i), "t_s": round(i / fps, 3), "joint": int(j) + 1, "step_deg": round(float(step[i, j]), 2),
               "rad_s": round(float(np.radians(step[i, j]) * fps), 2)} for i, j in bad]
    return {"events": events, "n_events": len(events), "max_step_deg": float(step.max()) if step.size else 0.0,
            "max_rad_s": float(np.radians(step.max()) * fps) if step.size else 0.0}


def zero_pose_frames(q: np.ndarray, tol: float = 1e-9) -> int:
    """Number of frames whose six joints are all (numerically) zero: a driver that had not published yet."""
    return int((np.abs(np.asarray(q)).max(axis=1) < tol).sum())


def joint_limit_margin_deg(q: np.ndarray, limits: np.ndarray) -> dict:
    """Closest approach to a joint limit: ``{"min_deg", "joint" (1-6), "side" ('lower'/'upper'), "per_joint_deg"}``."""
    q = np.asarray(q, dtype=np.float64)
    lo = np.degrees(q.min(axis=0) - limits[:, 0])
    hi = np.degrees(limits[:, 1] - q.max(axis=0))
    m = np.minimum(lo, hi)
    j = int(np.argmin(m))
    return {"min_deg": float(m[j]), "joint": j + 1, "side": "lower" if lo[j] <= hi[j] else "upper",
            "per_joint_deg": m.tolist()}


def video_state_consistency(video_motion: np.ndarray, speed: np.ndarray, fps: float, max_lag_s: float = 1.0,
                            win_s: float = 6.0) -> dict:
    """How well the image motion of a camera that moves with the arm follows the arm speed.

    ``video_motion[k]``: mean abs grey difference between frame k and k - 1 (any scale; entry 0 may be 0);
    ``speed[k]``: arm speed [rad/s] at frame k.  Both are smoothed over 0.5 s, then the Pearson correlation is taken over
    lags of +-``max_lag_s`` (lag > 0: the video follows the state later).  Also the lowest correlation of any
    ``win_s`` window (step win_s / 2) at the best lag: a glitch inside an otherwise consistent episode shows up there."""
    v = np.asarray(video_motion, dtype=np.float64)
    s = np.asarray(speed, dtype=np.float64)
    n = min(len(v), len(s))
    v, s = v[:n], s[:n]
    k = np.ones(max(1, int(round(0.5 * fps)))) / max(1, int(round(0.5 * fps)))
    vs, ss = np.convolve(v, k, "same"), np.convolve(s, k, "same")

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def at(lag):               # video index = state index + lag
        if lag >= 0:
            return vs[lag:], ss[: n - lag]
        return vs[: n + lag], ss[-lag:]

    lags = range(-int(max_lag_s * fps), int(max_lag_s * fps) + 1)
    cs = {L: corr(*at(L)) for L in lags}
    best = max(cs, key=cs.get)
    a, b = at(best)
    w = int(win_s * fps)
    wc = [corr(a[i:i + w], b[i:i + w]) for i in range(0, max(1, len(a) - w + 1), max(1, w // 2))] if len(a) >= w else [cs[best]]
    return {"corr_lag0": cs[0], "best_lag_frames": int(best), "best_lag_s": best / fps, "corr_best": cs[best],
            "min_window_corr": float(min(wc))}


def cluster_poses(poses_deg: np.ndarray, tol_deg: float = 3.0) -> list[int]:
    """Greedy leader clustering of (K, 6) poses by max-joint distance; returns a cluster id (0, 1, ...) per row, in order
    of first appearance.  Used to name the start-pose variants."""
    leaders: list[np.ndarray] = []
    out = []
    for p in np.asarray(poses_deg, dtype=np.float64):
        for i, l in enumerate(leaders):
            if np.abs(p - l).max() <= tol_deg:
                out.append(i)
                break
        else:
            leaders.append(p)
            out.append(len(leaders) - 1)
    return out


def arm_columns(names: list[str], arm: str) -> list[int]:
    """Column indices of ``<arm>_joint1..6`` in ``names`` (in joint order)."""
    return [names.index(f"{arm}_joint{i}") for i in range(1, 7)]


def trim_bounds(seg: dict, n: int, fps: float, margin_s: float = 0.5) -> tuple[int, int]:
    """Frame range [i0, i1) that keeps ``margin_s`` before the first and after the last movement."""
    if seg["last_move"] < 0:
        return 0, n
    m = int(round(margin_s * fps))
    return max(0, seg["first_move"] - m), min(n, seg["last_move"] + 1 + m)


# ------------------------------------------------------------------------------------------- wrist branch
def branch_label(q: np.ndarray, thr: float = WRIST_BRANCH_RAD) -> dict:
    """Wrist-branch label from the j4 / j6 extremes of an (N, 6) series.

    ``j4-/j6+`` (j4 below -thr and j6 above +thr), ``j4+/j6-`` (the opposite), ``both`` (both happen in one
    episode), ``neither``.  Which one is the majority is decided across the episodes (see quality_report)."""
    j4min, j4max = float(q[:, 3].min()), float(q[:, 3].max())
    j6min, j6max = float(q[:, 5].min()), float(q[:, 5].max())
    neg = j4min <= -thr and j6max >= thr
    pos = j4max >= thr and j6min <= -thr
    label = "both" if neg and pos else "j4-/j6+" if neg else "j4+/j6-" if pos else "neither"
    return {"label": label, "j4_min": j4min, "j4_max": j4max, "j6_min": j6min, "j6_max": j6max}


def start_pose_distance(q0: np.ndarray, ref_deg: np.ndarray | None = None) -> dict:
    """Distance of the first pose from ``ref_deg`` (default: the left-arm majority start pose)."""
    d = np.degrees(np.asarray(q0, dtype=np.float64)) - (START_POSE_DEG if ref_deg is None else np.asarray(ref_deg))
    return {"max_abs_deg": float(np.abs(d).max()), "l2_deg": float(np.linalg.norm(d)), "delta_deg": d.tolist()}


# -------------------------------------------------------------------------------------------------- DTW
def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.25) -> float:
    """Dynamic-time-warping cost between (N, k) and (M, k) series, Euclidean local cost, Sakoe-Chiba band
    (fraction of the longer series), divided by (N + M): the mean per-step joint-space distance in the unit of
    the inputs."""
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        lo, hi = max(1, i - w), min(m, i + w)
        cost = np.linalg.norm(b[lo - 1:hi] - a[i - 1], axis=1).tolist()
        for j in range(lo, hi + 1):
            best = prev[j - 1]
            if prev[j] < best:
                best = prev[j]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = cost[j - lo] + best
        prev = cur
    return prev[m] / (n + m)


def dtw_matrix(series: list[np.ndarray], band: float = 0.25) -> np.ndarray:
    k = len(series)
    d = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d[i, j] = d[j, i] = dtw_distance(series[i], series[j], band)
    return d


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD) with a floor on the scale so a tight cluster does not flag everything."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    scale = max(mad, 0.05 * max(abs(med), 1e-9), 1e-9)
    return (x - med) / scale
