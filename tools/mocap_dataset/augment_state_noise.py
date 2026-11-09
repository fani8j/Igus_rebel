#!/usr/bin/env python3
"""DART / CCIL-style corrective state perturbation of a clean LeRobot v3.0 mocap dataset (new dataset, source untouched).

    nice -n 19 python3 tools/mocap_dataset/augment_state_noise.py \
        [SOURCE_ROOT] --out OUTDIR [--k 4] [--seed 20261031] [--exclude-sources a,b] [--force]

SOURCE_ROOT is a dataset written by ``preprocess.py`` (default: mocap_right_31oct_clean_v1: 23 right-arm episodes,
14-D state/action, right arm in slots 7..12, 30 fps).  Output = every source episode as copy 0 (rows identical to the
source) plus K perturbed copies of each ("displaced, then recovered" trajectories).

Method (self-consistent: state, action and the action chunks a policy reads from consecutive rows agree)
  * M events per copy at frames t0 (about one per ``--event-period-s`` of motion; at least one; never in the idle head/tail,
    never inside a still run (``episode_metrics.motion_segments`` pause runs), at least ``--min-event-spacing-s`` apart).
    Each event draws eps0_j uniformly in [-a_j, +a_j] per arm joint (j6 included).
  * s_obs(t) = s(t) + sum_events eps0 * exp(-(t - t0) / tau)  for t >= t0.  The onset is ONE frame step: that is the
    disturbance.  The disturbance is external, so the command at t0-1 cannot know about it:
        action[t] = s(t+1) + sum_{events with t0 <= t} eps0 * exp(-(t+1 - t0)/tau)      (last frame repeats s_obs[-1]),
    i.e. action[t] = s_obs(t+1) (the source's "next state" definition) everywhere EXCEPT the frames t0-1, where it is
    s_obs(t0) - eps0 of the starting event(s) (the clean next state plus the older events).
  * Amplitude a_j = ``--eps-deg`` (default 3 deg) away from the box and ``--eps-near-deg`` (default 1.5 deg) when the arm
    is "near contact".  Near contact is APPROXIMATED as the frames where the link6 flange height z (FK of the right arm's
    URDF chain, arm base frame) is in the lowest ``--near-frac`` (30 %) of that episode's z range over its moving part
    (the box flaps are opened from the top, so the lowest poses are the closest to the box).  It is a proxy: the real
    carton pose is not stored in the dataset.
  * Rejection: an event is dropped and resampled (new t0 and eps0, up to ``--max-tries``) when the resulting observed state
    leaves the URDF joint limits (``--limit-margin-deg`` inside) or any joint of the state exceeds the velocity limit
    0.785 rad/s (joint_limits.yaml) on a frame transition, EXCEPT the step transition t0-1 -> t0 of the OBSERVED state (the
    disturbance, up to 1.57 rad/s at 3 deg; exempt only as a state transition).  The ACTION is checked with no exemption:
    the commanded displacement |action[t] - s_obs[t]| * fps <= 0.785 rad/s on every frame.  Note: the absolute action
    target necessarily follows the displaced state, so action[t0] - action[t0-1] is a one-frame jump of about
    0.9 * eps0 (target = the displaced pose); that is a target jump, not a commanded velocity, and is reported in AUGMENT.json.
    Only frames that carry a nonzero offset are checked, so a source that is itself close to a limit does not block events.
  * Images are NOT changed: the video files of the source are byte-copied (one full copy of the video files per copy index,
    the augmented episodes point at the same from/to timestamps).  The observed state differs from the pose the image
    shows by at most the offset (<= 3 deg per joint, 1.5 deg near contact; sums of overlapping events can reach
    more, the actual maximum is in AUGMENT.json).
  * observation.state_raw / action_teleop are the source's, unchanged.  New per-frame columns:
    ``augmentation.state_offset`` (float32 [14], the applied offset, zero in gripper slots / copy 0) and
    ``augmentation.copy_index`` (int64, 0 = original).
  * Layout: episodes 0..N-1 are the originals (same index as in the source), then copy 1 of every augmented source, copy 2, ...
    ``meta/source_episodes.json`` maps every output episode to its source episode and copy index.

TRAINING SPLITS: held-out evaluation MUST split by SOURCE episode (every copy of a held-out source leaks its
demo).  The current held-out set (mocap_right_2_ep008, mocap_right_3_ep004, mocap_right_farhan_ep004) is written to
``meta/holdout_suggestion.json`` together with the output episode indices of all their copies.  By default all sources are
augmented; ``--exclude-sources`` skips augmenting the listed ones (they keep only copy 0).

The source is only read.  OUTDIR must not exist; ``--force`` replaces an existing lerobot dataset created by this tool only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

DEFAULT_SOURCE = "/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_31oct_clean_v1"
DEFAULT_OUT = "/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_31oct_aug_statenoise_v1"
DEFAULT_SCRATCH = "/media/farhan/My Passport/igus_nvidia/tmp_aug"
DEFAULT_REPORT = REPO / "scan_results" / "dataset_augment_20261031" / "state_noise"
DEFAULT_URDF = REPO / "src" / "igus_rebel_description" / "urdf" / "igus_rebel2.urdf"
DEFAULT_HOLDOUT = ("mocap_right_2_ep008", "mocap_right_3_ep004", "mocap_right_farhan_ep004")
VMAX = 0.785                       # rad/s, joint_limits.yaml (rebel_dual_table_moveit_config) for every arm_right joint
COPY_KEY, OFFSET_KEY = "augmentation.copy_index", "augmentation.state_offset"
MARKER = "AUGMENT.json"
MARKER_TOOL = "augment_state_noise.py"


class AugmentError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ pure math
def decay_profile(n: int, t0: int, tau_s: float, fps: float) -> np.ndarray:
    """(n,) 0 before t0, exp(-(t - t0)/(tau*fps)) for t >= t0 (so exactly 1 at t0: a single-frame step)."""
    t = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    m = t >= t0
    out[m] = np.exp(-(t[m] - t0) / (tau_s * fps))
    return out


def next_state_action(s_obs: np.ndarray) -> np.ndarray:
    """action[t] = s_obs[t+1]; the last frame repeats (same definition as shifted_action with offset 1)."""
    a = np.empty_like(s_obs)
    a[:-1] = s_obs[1:]
    a[-1] = s_obs[-1]
    return a


def action_offset(n: int, events, tau_s: float, fps: float) -> np.ndarray:
    """(n,6) offset of the action relative to the CLEAN next state s(t+1) (last frame: s(n-1)).

    events = [(t0, eps0 (6,)), ...].  Only events that already started at frame t (t0 <= t) contribute, with their
    decay evaluated at the target frame t+1 (clamped to n-1): action[t0-1] has no part of the event that starts at t0."""
    out = np.zeros((n, 6))
    t = np.arange(n)
    tt = np.minimum(t + 1, n - 1)
    for t0, eps0 in events:
        m = t >= t0
        out[m] += np.exp(-(tt[m] - t0) / (tau_s * fps))[:, None] * np.asarray(eps0)[None, :]
    return out


def check_perturbed(s_src: np.ndarray, offset: np.ndarray, limits: np.ndarray, margin_rad: float, vmax: float,
                    fps: float, step_frames, act_offset: np.ndarray | None = None) -> str | None:
    """None if the perturbed (n,6) joint state is acceptable, else the reason.

    Limits: only frames with a nonzero offset are checked.  Velocity: |s_obs[t+1]-s_obs[t]| * fps <= vmax for every
    transition where either end has a nonzero offset, except the transitions t0-1 -> t0 of the events (``step_frames`` =
    the t0 values).  If ``act_offset`` is given the commanded displacement |action[t] - s_obs[t]| * fps (action =
    clean next state + act_offset) must be <= vmax on every frame, with no exemption."""
    s_obs = s_src + offset
    nz = np.any(offset != 0.0, axis=1)
    lo, hi = limits[:, 0] + margin_rad, limits[:, 1] - margin_rad
    bad = nz[:, None] & ((s_obs < lo) | (s_obs > hi))
    if bad.any():
        t, j = np.argwhere(bad)[0]
        return f"joint{j + 1} outside limits at frame {t}"
    v = np.abs(np.diff(s_obs, axis=0)) * fps                 # v[t] = transition t -> t+1
    chk = nz[:-1] | nz[1:]
    for t0 in step_frames:
        if 1 <= t0 <= len(v):
            chk[t0 - 1] = False
    bad = chk[:, None] & (v > vmax)
    if bad.any():
        t, j = np.argwhere(bad)[0]
        return f"joint{j + 1} speed {v[t, j]:.3f} rad/s > {vmax} at frame {t}->{t + 1}"
    if act_offset is not None:
        act = next_state_action(s_src) + act_offset
        c = np.abs(act - s_obs) * fps
        bad = c > vmax
        if bad.any():
            t, j = np.argwhere(bad)[0]
            return f"joint{j + 1} commanded speed {c[t, j]:.3f} rad/s > {vmax} at frame {t}"
    return None


def candidate_mask(n: int, first_move: int, last_move: int, pause_runs, tail_guard: int) -> np.ndarray:
    """Frames where an event may start: inside the moving part, not inside a pause run, with room left to recover."""
    m = np.zeros(n, dtype=bool)
    lo, hi = first_move + 1, last_move - 1 - tail_guard
    if hi >= lo:
        m[lo:hi + 1] = True
    for a, b in pause_runs:
        m[a:b] = False
    return m


def sample_offsets(rng: np.random.Generator, s_src: np.ndarray, *, fps: float, cand: np.ndarray, near: np.ndarray,
                   eps_far_rad: np.ndarray, eps_near_rad: np.ndarray, tau_s: float, n_events: int,
                   min_spacing_frames: int, limits: np.ndarray, margin_rad: float, vmax: float, max_tries: int = 50):
    """Draw up to ``n_events`` accepted events.  Returns (offset (n,6) float64, [event dicts], n_rejected, n_unplaced)."""
    n = len(s_src)
    offset = np.zeros((n, 6))
    events, rejected, unplaced = [], 0, 0
    for _ in range(n_events):
        placed = False
        for tries in range(1, max_tries + 1):
            free = cand.copy()
            for ev in events:
                free[max(0, ev["t0"] - min_spacing_frames):ev["t0"] + min_spacing_frames + 1] = False
            idx = np.flatnonzero(free)
            if len(idx) == 0:
                break
            t0 = int(rng.choice(idx))
            amp = eps_near_rad if near[t0] else eps_far_rad
            eps0 = rng.uniform(-1.0, 1.0, size=6) * amp
            trial = offset + decay_profile(n, t0, tau_s, fps)[:, None] * eps0[None, :]
            evs = [(e["t0"], np.radians(e["eps0_deg"])) for e in events] + [(t0, eps0)]
            why = check_perturbed(s_src, trial, limits, margin_rad, vmax, fps, [e[0] for e in evs],
                                  act_offset=action_offset(n, evs, tau_s, fps))
            if why is None:
                offset = trial
                events.append({"t0": t0, "t0_s": round(t0 / fps, 4), "eps0_deg": np.degrees(eps0).tolist(),
                               "near_contact": bool(near[t0]), "tries": tries})
                placed = True
                break
            rejected += 1
        if not placed:
            unplaced += 1
    events.sort(key=lambda e: e["t0"])
    return offset, events, rejected, unplaced


def urdf_limits(path: Path = DEFAULT_URDF) -> np.ndarray:
    import lerobot_to_plan as l2p
    return l2p.urdf_limits(path)


def flange_z(q: np.ndarray, urdf: Path = DEFAULT_URDF) -> np.ndarray:
    """z of link6 in the arm base frame for an (n,6) joint series (same chain as lerobot_to_plan's FK summary)."""
    sys.path.insert(0, str(REPO / "src" / "rebel_dual_arm_calibration"))
    from rebel_dual_arm_calibration.kinematics import ArmKinematics, Urdf
    k = ArmKinematics.from_urdf(Urdf.load(str(urdf)), "base_link", "link6", "")
    return np.asarray(k.chain6.fk(q)[:, 2, 3])


def near_contact_mask(z: np.ndarray, first_move: int, last_move: int, frac: float) -> np.ndarray:
    """Frames whose height is in the lowest ``frac`` of the z distribution over the moving part."""
    seg = z[first_move:last_move + 1] if last_move > first_move else z
    thr = np.quantile(seg, frac)
    return z <= thr


# ------------------------------------------------------------------------------------------ dataset IO
def _per_joint(txt: str | None, default: np.ndarray | None, name: str) -> np.ndarray | None:
    if txt is None:
        return default
    v = [float(x) for x in str(txt).split(",")]
    if len(v) == 1:
        v = v * 6
    if len(v) != 6 or any(x < 0 for x in v):
        raise AugmentError(f"{name}: need 1 or 6 non-negative values, got {txt!r}")
    return np.radians(np.asarray(v))


def read_source(root: Path):
    import pyarrow.parquet as pq
    info = json.loads((root / "meta" / "info.json").read_text())
    if not str(info["codebase_version"]).startswith("v3"):
        raise AugmentError("source must be LeRobot v3.x")
    files = sorted((root / "data").rglob("*.parquet"))
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = [r for f in sorted((root / "meta" / "episodes").rglob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    rows.sort(key=lambda r: r["episode_index"])
    smap = json.loads((root / "meta" / "source_episodes.json").read_text())
    return info, df, rows, smap


def arm_slots(names: list[str], arm: str) -> list[int]:
    idx = [names.index(f"{arm}_joint{i}") for i in range(1, 7)]
    if idx != list(range(idx[0], idx[0] + 6)):
        raise AugmentError(f"{arm} joints are not contiguous in {names}")
    return idx


def prepare_output(out: Path, force: bool) -> Path:
    out = out.expanduser().resolve()
    if out.exists():
        if not force:
            raise AugmentError(f"{out} exists; refusing to overwrite (use --force only for a dataset this tool created)")
        if not (out / MARKER).is_file() or json.loads((out / MARKER).read_text()).get("tool") != MARKER_TOOL:
            raise AugmentError(f"{out} exists and was not created by {MARKER_TOOL}; not removing it")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.parent / f".{out.name}.partial-{os.getpid()}"


def build(args, out_final: Path, staging: Path) -> dict:
    import datasets
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.utils import get_hf_features_from_features, write_info, write_stats
    import preprocess as pp
    from episode_metrics import STILL_SPEED, motion_segments

    src = Path(args.source_root).expanduser().resolve()
    info, df, rows, smap = read_source(src)
    fps = float(info["fps"])
    names = list(info["features"]["observation.state"]["names"])
    slots = arm_slots(names, args.arm)
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    n_src = len(rows)
    if n_src != len(smap):
        raise AugmentError("source_episodes.json does not match the episode count")
    limits = urdf_limits(Path(args.urdf))
    eps_far = _per_joint(args.eps_deg, None, "--eps-deg")
    eps_near = _per_joint(args.eps_near_deg, 0.5 * eps_far, "--eps-near-deg")
    excl = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    unknown = excl - {m["source"] for m in smap}
    if unknown:
        raise AugmentError(f"--exclude-sources: unknown source episodes {sorted(unknown)}")

    cols_np = {c: np.stack(df[c].values) if c in ("observation.state", "action", pp.RAW_STATE, "action_teleop")
               else df[c].values for c in df.columns}
    ep_idx = df["episode_index"].values
    src_eps = []
    for e in range(n_src):
        sel = np.flatnonzero(ep_idx == e)
        if len(sel) != rows[e]["length"] or not np.array_equal(sel, np.arange(sel[0], sel[0] + len(sel))):
            raise AugmentError(f"source episode {e}: rows are not one contiguous block of the stated length")
        src_eps.append({k: v[sel] for k, v in cols_np.items()})

    # ---- plan the output episodes: originals first, then copy 1 of every augmented source, copy 2, ...
    plan = [(e, 0) for e in range(n_src)]
    for c in range(1, args.k + 1):
        plan += [(e, c) for e in range(n_src) if smap[e]["source"] not in excl]

    # ---- per source: motion segmentation, candidate / near-contact masks (shared by its copies)
    t_start = time.time()
    masks = []
    for e, ep in enumerate(src_eps):
        q = ep["observation.state"][:, slots].astype(np.float64)
        if not np.any(q):
            raise AugmentError(f"episode {e}: {args.arm} state is all zero")
        seg = motion_segments(q, fps, STILL_SPEED)
        n = len(q)
        cand = candidate_mask(n, seg["first_move"], seg["last_move"], seg["pause_runs"], int(math.ceil(args.tau_s * fps)))
        near = near_contact_mask(flange_z(q, Path(args.urdf)), seg["first_move"], seg["last_move"], args.near_frac)
        pause_f = sum(b - a for a, b in seg["pause_runs"])
        motion_s = max(0, seg["last_move"] - seg["first_move"] + 1 - pause_f) / fps
        masks.append({"cand": cand, "near": near, "motion_s": motion_s, "seg": seg})

    # ---- perturb
    out_eps = []                                              # one dict per output episode
    for oe, (e, c) in enumerate(plan):
        ep = src_eps[e]
        n = len(ep["observation.state"])
        state = ep["observation.state"].copy()
        action = ep["action"].copy()
        off14 = np.zeros((n, len(names)), dtype=np.float32)
        record = {"events": [], "rejected_samples": 0, "unplaced_events": 0}
        if c > 0:
            mk = masks[e]
            rng = np.random.default_rng([args.seed, e, c])
            m_ev = max(1, int(round(mk["motion_s"] / args.event_period_s)))
            off, events, rej, unpl = sample_offsets(
                rng, state[:, slots].astype(np.float64), fps=fps, cand=mk["cand"], near=mk["near"], eps_far_rad=eps_far,
                eps_near_rad=eps_near, tau_s=args.tau_s, n_events=m_ev,
                min_spacing_frames=int(round(args.min_event_spacing_s * fps)), limits=limits,
                margin_rad=math.radians(args.limit_margin_deg), vmax=VMAX, max_tries=args.max_tries)
            if not events:
                raise AugmentError(f"source episode {e} copy {c}: no event could be placed")
            s_obs = state.astype(np.float64)
            s_obs[:, slots] += off
            state = s_obs.astype(np.float32)
            a64 = next_state_action(src_eps[e]["observation.state"].astype(np.float64))
            a64[:, slots] += action_offset(n, [(ev["t0"], np.radians(ev["eps0_deg"])) for ev in events], args.tau_s, fps)
            action = a64.astype(np.float32)
            off14[:, slots] = off.astype(np.float32)
            record = {"events": events, "n_events_requested": m_ev, "rejected_samples": rej, "unplaced_events": unpl}
        out_eps.append({"src": e, "copy": c, "n": n, "state": state, "action": action, "off": off14, **record})

    # ---- videos: byte-copy the source files once per copy index
    vfile_idx = {}                                            # key -> sorted list of (chunk, file) of the source
    for k in video_keys:
        vfile_idx[k] = sorted({(r[f"videos/{k}/chunk_index"], r[f"videos/{k}/file_index"]) for r in rows})
    chunk_size = info.get("chunks_size", 1000)
    n_copies = max(c for _, c in plan) + 1
    staging.mkdir(parents=True)
    used_copies = sorted({c for _, c in plan})
    for k in video_keys:
        nf = len(vfile_idx[k])
        for c in used_copies:
            for pos, (ch, fi) in enumerate(vfile_idx[k]):
                new = c * nf + pos
                dst = staging / pp.V30_VIDEO_PATH.format(video_key=k, chunk_index=new // chunk_size, file_index=new % chunk_size)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / pp.V30_VIDEO_PATH.format(video_key=k, chunk_index=ch, file_index=fi), dst)

    # ---- features, data, per-episode meta + stats
    features = {k: dict(v) for k, v in info["features"].items()}
    features[OFFSET_KEY] = {"dtype": "float32", "shape": [len(names)], "names": list(names)}
    features[COPY_KEY] = {"dtype": "int64", "shape": [1], "names": None}
    lengths = [o["n"] for o in out_eps]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    vec_keys = ["observation.state", "action", pp.RAW_STATE, "action_teleop", OFFSET_KEY]
    scalar_keys = [*pp.SCALAR_KEYS, pp.SRC_INDEX, pp.SRC_TIME, COPY_KEY]
    cols = {k: [] for k in (*vec_keys, *scalar_keys)}
    ep_rows, ep_stats_list = [], []
    for oe, o in enumerate(out_eps):
        s = src_eps[o["src"]]
        n = o["n"]
        data_e = {"observation.state": o["state"], "action": o["action"], pp.RAW_STATE: s[pp.RAW_STATE],
                  "action_teleop": s["action_teleop"], OFFSET_KEY: o["off"],
                  "timestamp": s["timestamp"], "frame_index": s["frame_index"],
                  "episode_index": np.full(n, oe, dtype=np.int64), "index": np.arange(starts[oe], starts[oe] + n, dtype=np.int64),
                  "task_index": s["task_index"], pp.SRC_INDEX: s[pp.SRC_INDEX], pp.SRC_TIME: s[pp.SRC_TIME],
                  COPY_KEY: np.full(n, o["copy"], dtype=np.int64)}
        for k, v in data_e.items():
            cols[k].append(v)
        st = compute_episode_stats({k: pp.as_stats_array(v) for k, v in data_e.items()}, {k: features[k] for k in data_e})
        srow = rows[o["src"]]
        st.update(pp.image_stats_from_source(srow, video_keys))
        ep_stats_list.append(st)
        row = {"episode_index": oe, "tasks": srow["tasks"], "length": n, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": int(starts[oe]), "dataset_to_index": int(starts[oe] + n)}
        for k in video_keys:
            nf = len(vfile_idx[k])
            pos = vfile_idx[k].index((srow[f"videos/{k}/chunk_index"], srow[f"videos/{k}/file_index"]))
            new = o["copy"] * nf + pos
            row.update({f"videos/{k}/chunk_index": new // chunk_size, f"videos/{k}/file_index": new % chunk_size,
                        f"videos/{k}/from_timestamp": srow[f"videos/{k}/from_timestamp"],
                        f"videos/{k}/to_timestamp": srow[f"videos/{k}/to_timestamp"]})
        row.update(pp.flat_stats(st))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        ep_rows.append(row)

    full = {k: np.concatenate(v) for k, v in cols.items()}
    hf_features = get_hf_features_from_features({k: {**v, "shape": tuple(v["shape"])} for k, v in features.items()
                                                 if v["dtype"] != "video"})
    ds = datasets.Dataset.from_dict({k: (full[k].tolist() if full[k].ndim == 2 else full[k]) for k in hf_features},
                                    features=hf_features, split="train")
    data_path = staging / pp.V30_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(data_path))
    if data_path.stat().st_size / 2 ** 20 > info.get("data_files_size_in_mb", 100):
        raise AugmentError("data parquet exceeds data_files_size_in_mb; multi-file data is not implemented")
    ep_path = staging / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), ep_path)
    shutil.copyfile(src / "meta" / "tasks.parquet", staging / "meta" / "tasks.parquet")

    gstats = compute_episode_stats({k: pp.as_stats_array(v) for k, v in full.items()}, {k: features[k] for k in full})
    gstats.update(aggregate_stats([{k: s[k] for k in video_keys} for s in ep_stats_list]))
    write_stats(gstats, staging)
    new_info = dict(info)
    new_info.update({"features": features, "total_episodes": len(out_eps), "total_frames": int(starts[-1]),
                     "splits": {"train": f"0:{len(out_eps)}"}})
    write_info(new_info, staging)

    # ---- sidecars
    smap_out = []
    for oe, o in enumerate(out_eps):
        m = smap[o["src"]]
        smap_out.append({"episode_index": oe, "source": m["source"], "source_episode_index": o["src"],
                         "source_dataset": str(src), "copy_index": o["copy"], "augmented": o["copy"] > 0,
                         "frames": o["n"], "clean_dataset_record": m})
    (staging / "meta" / "source_episodes.json").write_text(json.dumps(smap_out, indent=2))
    present = {m["source"] for m in smap}
    held = [s for s in DEFAULT_HOLDOUT if s in present]
    hold = {"note": "Held-out evaluation must split by SOURCE episode: drop all copies of a held-out source from training.",
            "held_out_sources": list(DEFAULT_HOLDOUT), "missing_in_this_dataset": [s for s in DEFAULT_HOLDOUT if s not in present],
            "held_out_output_episode_indices": [m["episode_index"] for m in smap_out if m["source"] in held],
            "train_output_episode_indices": [m["episode_index"] for m in smap_out if m["source"] not in held]}
    (staging / "meta" / "holdout_suggestion.json").write_text(json.dumps(hold, indent=2))

    maxoff = [float(np.degrees(np.abs(o["off"][:, slots]).max())) for o in out_eps]
    aug = {
        "tool": MARKER_TOOL, "command": " ".join(sys.argv), "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_root": str(src), "output": str(out_final), "seed": args.seed,
        "rng": "numpy.random.default_rng([seed, source_episode_index, copy_index])",
        "parameters": {"k": args.k, "arm": args.arm, "arm_slots": slots, "tau_s": args.tau_s,
                       "event_period_s": args.event_period_s, "min_event_spacing_s": args.min_event_spacing_s,
                       "eps_deg": np.degrees(eps_far).tolist(), "eps_near_deg": np.degrees(eps_near).tolist(),
                       "near_frac": args.near_frac, "near_contact_proxy": "link6 z (FK, arm base frame) in the lowest near_frac of the moving part",
                       "limit_margin_deg": args.limit_margin_deg, "vmax_rad_s": VMAX, "max_tries": args.max_tries,
                       "urdf": str(args.urdf), "joint_limits_deg": np.degrees(limits).round(3).tolist(),
                       "exclude_sources": sorted(excl), "fps": fps, "still_speed": STILL_SPEED},
        "velocity_check": "state transitions with a nonzero offset except t0-1 -> t0 of each event (the external step); "
                          "commanded displacement |action[t]-state[t]|*fps checked on every frame without exemption",
        "action_label": "action[t] = s(t+1) + sum_{events t0<=t} eps0*exp(-(t+1-t0)/tau); action[t0-1] excludes the new event",
        "video": "byte copy of the source video files, one set per copy index; episodes keep the source from/to timestamps",
        "lerobot_version": __import__("lerobot").__version__,
        "total_episodes": len(out_eps), "total_frames": int(starts[-1]),
        "max_abs_offset_deg_overall": max(maxoff),
        "max_action_increment_rad_s": float(max(np.abs(np.diff(o["action"][:, slots].astype(np.float64), axis=0)).max() * fps
                                                for o in out_eps)),
        "max_commanded_speed_rad_s": float(max(np.abs(o["action"].astype(np.float64) - o["state"])[:, slots].max() * fps
                                               for o in out_eps)),
        "episodes": [{"episode_index": oe, "source": smap[o["src"]]["source"], "source_episode_index": o["src"],
                      "copy_index": o["copy"], "frames": o["n"], "max_abs_offset_deg": round(maxoff[oe], 4),
                      **({k: o[k] for k in ("events", "n_events_requested", "rejected_samples", "unplaced_events")}
                         if o["copy"] > 0 else {})} for oe, o in enumerate(out_eps)],
        "build_seconds": round(time.time() - t_start, 1),
    }
    (staging / MARKER).write_text(json.dumps(aug, indent=2))
    (staging / "README.md").write_text(dataset_card(aug, hold, excl))
    return aug


def dataset_card(aug: dict, hold: dict, excl: set) -> str:
    p = aug["parameters"]
    return f"""# {Path(aug['output']).name}

State-perturbation (DART / CCIL style) augmentation of `{aug['source_root']}`
({aug['total_episodes']} episodes, {aug['total_frames']} frames, 30 fps, LeRobot v3.0). Built by
`tools/mocap_dataset/augment_state_noise.py` (seed {aug['seed']}; all parameters and events in `AUGMENT.json`).

* Episode layout: output episodes 0..N-1 are the source episodes unchanged (copy 0, rows identical to the source); then copy 1
  of every augmented source, copy 2, ... `meta/source_episodes.json` maps every episode to its source and copy index.
* A perturbed copy has `observation.state(t) = s(t) + sum_events eps0 * exp(-(t - t0)/{p['tau_s']} s)` for t >= t0 on the 6 {p['arm']}
  joints (single-frame step at t0), and the action is the label of an EXTERNAL disturbance-free command: `action(t) = s(t+1) + sum of the events
  with t0 <= t of eps0*exp(-(t+1-t0)/tau)`, i.e. `action(t) = observation.state(t+1)` except at t = t0-1 of each event, where the
  action still excludes the new disturbance (`= observation.state(t0) - eps0`); the last frame repeats. State, action and action chunks
  are consistent and no command contains the disturbance step. `observation.state_raw` and `action_teleop` are the source's (untouched). Gripper slots stay 0.
* Extra columns: `augmentation.state_offset` (float32 [14], applied offset) and `augmentation.copy_index` (int64, 0 = original).
* Offsets: uniform |eps0_j| <= {p['eps_deg'][0]:g} deg (j6 included), <= {p['eps_near_deg'][0]:g} deg near contact (proxy: link6 height in the
  lowest {p['near_frac']:.0%} of the moving part); overlapping events add up (largest offset in the dataset
  {aug['max_abs_offset_deg_overall']:.2f} deg). Events are rejected/resampled when they leave the URDF joint limits or push a joint over
  {p['vmax_rad_s']} rad/s on a state transition (except the disturbance step t0-1 -> t0 itself, up to eps0 in ONE frame, which is
  external and never commanded) or when the commanded displacement |action(t) - state(t)| exceeds it on any frame (no exemption).
  The absolute action target follows the displaced state, so action(t0) - action(t0-1) is a one-frame target jump of about 0.9*eps0
  (max recorded in AUGMENT.json `max_action_increment_rad_s`); it is not a commanded velocity.
* CAVEAT: images are NOT changed (video files are byte copies of the source). In perturbed copies the observed joint state
  differs from the pose seen in the images by the current offset (<= a few deg, decaying with tau). This teaches
  "state is slightly off, steer back", not visual servoing; the image-vs-state mismatch is intentional label noise of that size.
* TRAINING SPLITS: split by SOURCE episode, never by output episode. Every copy of a held-out source leaks its demo. Suggested held-out
  sources: {', '.join(hold['held_out_sources'])} (`meta/holdout_suggestion.json` lists the output episode indices to drop from training).
  Evaluate on copy 0 of the held-out sources only. Sources not augmented (`--exclude-sources`): {sorted(excl) or 'none'}.
"""


# ------------------------------------------------------------------------------------------ validation
def validate(out: Path, source_root: Path, report_dir: Path, n_decode_eps: int = 6) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    res, ok = {"checks": []}, True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        res["checks"].append({"name": name, "ok": bool(cond), "detail": str(detail)})
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}", flush=True)

    aug = json.loads((out / MARKER).read_text())
    prm = aug["parameters"]
    slots, fps, tau = prm["arm_slots"], prm["fps"], prm["tau_s"]
    limits = np.radians(np.asarray(prm["joint_limits_deg"]))
    info, sdf, srows, smap = read_source(source_root)
    odf = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
    oinfo = json.loads((out / "meta" / "info.json").read_text())
    nsrc = len(srows)
    check("total_episodes", oinfo["total_episodes"] == len(aug["episodes"]) == odf.episode_index.nunique(), oinfo["total_episodes"])
    if not prm["exclude_sources"]:
        check("episodes == N*(K+1)", oinfo["total_episodes"] == nsrc * (prm["k"] + 1), f"{oinfo['total_episodes']}")
    check("total_frames", oinfo["total_frames"] == len(odf), len(odf))
    sm = json.loads((out / "meta" / "source_episodes.json").read_text())
    check("source_episodes.json complete", [m["episode_index"] for m in sm] == list(range(len(sm))))

    S = {c: np.stack(sdf[c].values) for c in ("observation.state", "action", "observation.state_raw", "action_teleop")}
    O = {c: np.stack(odf[c].values) for c in ("observation.state", "action", "observation.state_raw", "action_teleop", OFFSET_KEY)}
    worst = {"copy0_rows": 0.0, "act_next": 0.0, "off_vs_events": 0.0, "off_vs_diff": 0.0, "maxspeed": 0.0,
             "step_minus_decay": 0.0, "lim": 1e9, "act_vs_events": 0.0, "act_onset": 0.0, "cmd_speed": 0.0,
             "act_inc": 0.0, "act_inc_src": 0.0}
    copy0_ok = True
    for oe, m in enumerate(sm):
        o_sel = np.flatnonzero(odf.episode_index.values == oe)
        e = m["source_episode_index"]
        s_sel = np.flatnonzero(sdf.episode_index.values == e)
        if len(o_sel) != len(s_sel):
            check(f"ep{oe} frame count", False, f"{len(o_sel)} vs {len(s_sel)}")
            continue
        st, ac = O["observation.state"][o_sel], O["action"][o_sel]
        if m["copy_index"] == 0:
            for c in ("observation.state", "action", "observation.state_raw", "action_teleop"):
                copy0_ok &= np.array_equal(O[c][o_sel], S[c][s_sel])
            for c in ("timestamp", "frame_index", "task_index", "source_frame_index", "source_timestamp"):
                copy0_ok &= np.array_equal(odf[c].values[o_sel], sdf[c].values[s_sel])
            copy0_ok &= not O[OFFSET_KEY][o_sel].any() and (odf[COPY_KEY].values[o_sel] == 0).all()
            continue
        # perturbed copy
        for c in ("observation.state_raw", "action_teleop"):
            copy0_ok &= np.array_equal(O[c][o_sel], S[c][s_sel])
        for c in ("timestamp", "frame_index", "task_index", "source_frame_index", "source_timestamp"):
            copy0_ok &= np.array_equal(odf[c].values[o_sel], sdf[c].values[s_sel])
        copy0_ok &= (odf[COPY_KEY].values[o_sel] == m["copy_index"]).all()
        rec = next(a for a in aug["episodes"] if a["episode_index"] == oe)
        evs = [(ev["t0"], np.radians(ev["eps0_deg"])) for ev in rec["events"]]
        exp_a = next_state_action(S["observation.state"][s_sel].astype(np.float64))
        exp_a[:, slots] += action_offset(len(o_sel), evs, tau, fps)
        worst["act_vs_events"] = max(worst["act_vs_events"], float(np.abs(ac - exp_a).max()))
        t0s = sorted(ev["t0"] for ev in rec["events"])
        keep = np.ones(len(o_sel), bool); keep[[t - 1 for t in t0s]] = False
        keep[-1] = False
        worst["act_next"] = max(worst["act_next"], float(np.abs(ac[:-1][keep[:-1]] - st[1:][keep[:-1]]).max()))
        for ev in rec["events"]:                      # action[t0-1] = s_obs[t0] - eps0 (isolated events: exact; overlapping ones are in exp_a)
            t0 = ev["t0"]
            isolated = not any(abs(t0 - o2["t0"]) <= 1 for o2 in rec["events"] if o2 is not ev)
            if isolated:
                d = ac[t0 - 1, slots].astype(np.float64) - (st[t0, slots].astype(np.float64) - np.radians(ev["eps0_deg"]))
                worst["act_onset"] = max(worst["act_onset"], float(np.abs(d).max()))
        cmd = np.abs(ac.astype(np.float64) - st) * fps
        worst["cmd_speed"] = max(worst["cmd_speed"], float(cmd[:, slots].max()))
        worst["act_inc"] = max(worst["act_inc"], float(np.abs(np.diff(ac[:, slots].astype(np.float64), axis=0)).max() * fps))
        worst["act_inc_src"] = max(worst["act_inc_src"], float(np.abs(np.diff(S["action"][s_sel][:, slots].astype(np.float64), axis=0)).max() * fps))
        off = O[OFFSET_KEY][o_sel].astype(np.float64)
        s_src = S["observation.state"][s_sel]
        worst["off_vs_diff"] = max(worst["off_vs_diff"], float(np.abs((st.astype(np.float64) - s_src) - off).max()))
        # offsets decay as specified: rebuild from the recorded events
        exp = np.zeros((len(o_sel), 6))
        for ev in rec["events"]:
            exp += decay_profile(len(o_sel), ev["t0"], tau, fps)[:, None] * np.radians(ev["eps0_deg"])[None, :]
        worst["off_vs_events"] = max(worst["off_vs_events"], float(np.abs(off[:, slots] - exp).max()))
        for ev in rec["events"]:
            t0 = ev["t0"]
            if len(o_sel) > t0 + 1 and not any(abs(t0 - o2["t0"]) < 3 * fps * tau for o2 in rec["events"] if o2 is not ev):
                got = off[t0 + 1, slots] / np.where(off[t0, slots] == 0, np.nan, off[t0, slots])
                worst["step_minus_decay"] = max(worst["step_minus_decay"], float(np.nanmax(np.abs(got - math.exp(-1 / (tau * fps))))))
        s_arm = st[:, slots].astype(np.float64)
        why = check_perturbed(s_src[:, slots].astype(np.float64), s_arm - s_src[:, slots].astype(np.float64), limits,
                              math.radians(prm["limit_margin_deg"]) - 1e-5, VMAX + 1e-4, fps, [ev["t0"] for ev in rec["events"]],
                              act_offset=action_offset(len(o_sel), evs, tau, fps))
        if why:
            check(f"ep{oe} limits/velocity", False, why)
        worst["maxspeed"] = max(worst["maxspeed"], float(np.abs(np.diff(s_arm, axis=0)).max() * fps))
        nz = np.abs(off[:, slots]).max(axis=1) > 0
        worst["lim"] = min(worst["lim"], float(np.degrees(np.minimum(s_arm - limits[:, 0], limits[:, 1] - s_arm)[nz].min())))
        check_gr = np.array_equal(st[:, [i for i in range(14) if i not in slots]], s_src[:, [i for i in range(14) if i not in slots]])
        copy0_ok &= check_gr
    check("copy 0 rows == source rows exactly (and passthrough columns of copies)", copy0_ok)
    check("action[t] == observed state[t+1] except at frames t0-1 (copies)", worst["act_next"] == 0.0, f"max diff {worst['act_next']:.3g}")
    check("action == s(t+1) + started events (rebuilt from AUGMENT.json)", worst["act_vs_events"] < 2e-6, f"max {worst['act_vs_events']:.3g} rad")
    check("action[t0-1] == s_obs[t0] - eps0 (isolated events)", worst["act_onset"] < 2e-6, f"max {worst['act_onset']:.3g} rad")
    check("commanded speed |action - state| <= 0.785 rad/s, no exemption", worst["cmd_speed"] <= VMAX + 1e-4,
          f"max {worst['cmd_speed']:.3f} rad/s")
    check("report: max action increment (target jump at t0)", True,
          f"{worst['act_inc']:.3f} rad/s (source action: {worst['act_inc_src']:.3f})")
    check("offset column == state - source state", worst["off_vs_diff"] < 2e-6, f"max {worst['off_vs_diff']:.3g} rad")
    check("offsets follow sum eps0*exp(-(t-t0)/tau)", worst["off_vs_events"] < 1e-6, f"max {worst['off_vs_events']:.3g} rad")
    check("first-frame decay factor exp(-1/(tau fps))", worst["step_minus_decay"] < 1e-3, f"max dev {worst['step_minus_decay']:.3g}")
    check("limits + velocity (recovery part)", True, f"min margin to limit on perturbed frames {worst['lim']:.2f} deg; "
                                                    f"max raw speed incl. steps {worst['maxspeed']:.3f} rad/s")
    res["worst"] = worst

    # stats.json
    stats = json.loads((out / "meta" / "stats.json").read_text())
    for k in ("observation.state", "action", OFFSET_KEY, COPY_KEY, "observation.images.head"):
        check(f"stats.json has {k}", k in stats and all(q in stats[k] for q in ("min", "max", "mean", "std", "q01", "q99")))
    check("stats finite", all(np.isfinite(np.asarray(v, dtype=float)).all() for k in ("observation.state", "action", OFFSET_KEY)
                              for v in stats[k].values()))
    check("stats count", int(np.asarray(stats["observation.state"]["count"]).ravel()[0]) == len(odf))

    # lerobot loader + video decode
    ds = LeRobotDataset("local/aug_statenoise_check", root=out)
    check("LeRobotDataset len / episodes / fps", len(ds) == len(odf) and ds.num_episodes == oinfo["total_episodes"] and ds.fps == 30,
          f"{len(ds)} {ds.num_episodes}")
    eps_meta = ds.meta.episodes
    n_dec, max_img_diff = 0, 0.0
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(nsrc, size=min(n_decode_eps, nsrc), replace=False).tolist())
    for e in pick:
        outs = [m["episode_index"] for m in sm if m["source_episode_index"] == e]
        first_items = {}
        for oe in outs:
            f0, f1 = int(eps_meta["dataset_from_index"][oe]), int(eps_meta["dataset_to_index"][oe])
            for tag, gi in (("first", f0), ("mid", (f0 + f1) // 2), ("last", f1 - 1)):
                it = ds[gi]
                n_dec += 1
                imgs = [it[k] for k in ds.meta.video_keys]
                okf = all(tuple(i.shape) == (3, 480, 640) and np.isfinite(i.numpy()).all() for i in imgs)
                check(f"decode ep{oe} {tag}", okf and int(it["episode_index"]) == oe and int(it[COPY_KEY]) == sm[oe]["copy_index"])
                if oe == outs[0]:
                    first_items[tag] = imgs
                else:
                    d = max(float((a - b).abs().max()) for a, b in zip(imgs, first_items[tag]))
                    max_img_diff = max(max_img_diff, d)
    check("frames of copies identical to copy 0 (same bytes)", max_img_diff < 1e-6, f"max diff {max_img_diff:.3g} over {n_dec} decoded items")
    res.update({"decoded_items": n_dec, "ok": bool(ok)})

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = pick[0]
        copy_oes = [m["episode_index"] for m in sm if m["source_episode_index"] == e]
        oe = copy_oes[1] if len(copy_oes) > 1 else copy_oes[0]
        so = np.flatnonzero(odf.episode_index.values == copy_oes[0]); po = np.flatnonzero(odf.episode_index.values == oe)
        t = np.arange(len(so)) / fps
        fig, axs = plt.subplots(6, 2, figsize=(14, 14), sharex=True)
        for j in range(6):
            for col, key in enumerate(("observation.state", "action")):
                ax = axs[j, col]
                ax.plot(t, np.degrees(O[key][so][:, slots[j]]), label="original", lw=1.2)
                ax.plot(t, np.degrees(O[key][po][:, slots[j]]), label="perturbed", lw=0.9)
                ax.set_ylabel(f"j{j + 1} [deg]")
                if j == 0:
                    ax.set_title(f"{key}  (source {sm[oe]['source']}, copy {sm[oe]['copy_index']})")
                    ax.legend(loc="upper right")
        axs[-1, 0].set_xlabel("t [s]"); axs[-1, 1].set_xlabel("t [s]")
        fig.tight_layout()
        report_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_dir / "state_noise_example.png", dpi=90)
        res["png"] = str(report_dir / "state_noise_example.png")
        # offset zoom around the first event
        ev = next(a for a in aug["episodes"] if a["episode_index"] == oe)["events"][0]
        i0 = max(0, ev["t0"] - 15); i1 = min(len(so), ev["t0"] + 60)
        fig2, axs2 = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        for j in range(6):
            ax = axs2.flat[j]
            ax.plot(np.arange(i0, i1), np.degrees(O["observation.state"][so][i0:i1, slots[j]]), label="state orig")
            ax.plot(np.arange(i0, i1), np.degrees(O["observation.state"][po][i0:i1, slots[j]]), label="state pert")
            ax.plot(np.arange(i0, i1), np.degrees(O["action"][po][i0:i1, slots[j]]), "--", label="action pert")
            ax.set_title(f"j{j + 1}  event t0={ev['t0']}")
        axs2.flat[0].legend(); fig2.tight_layout()
        fig2.savefig(report_dir / "state_noise_event_zoom.png", dpi=90)
    except Exception as exc:  # noqa: BLE001
        check("plot", False, exc)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_root", nargs="?", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=4, help="perturbed copies per source episode (default %(default)s)")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--arm", default="arm_right")
    ap.add_argument("--tau-s", type=float, default=0.3)
    ap.add_argument("--event-period-s", type=float, default=4.0, help="about one event per this many seconds of motion")
    ap.add_argument("--min-event-spacing-s", type=float, default=1.0)
    ap.add_argument("--eps-deg", default="3", help="max |offset| per joint, deg: one value or 6 comma-separated (default %(default)s)")
    ap.add_argument("--eps-near-deg", default="1.5", help="same near contact; default 1.5 (pass 'half' for 0.5 x --eps-deg)")
    ap.add_argument("--near-frac", type=float, default=0.3)
    ap.add_argument("--limit-margin-deg", type=float, default=0.0)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--exclude-sources", default="", help="comma-separated source names that are NOT augmented (copy 0 only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scratch", default=DEFAULT_SCRATCH)
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT))
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    if args.eps_near_deg == "half":
        args.eps_near_deg = None
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = args.scratch
    out = Path(args.out).expanduser().resolve()
    try:
        staging = prepare_output(out, args.force)
        if staging.exists():
            shutil.rmtree(staging)
        aug = build(args, out, staging)
    except AugmentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except BaseException:
        if "staging" in dir() and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if out.exists():                                   # --force on a dataset this tool created
        shutil.rmtree(out)
    staging.rename(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out}: {aug['total_episodes']} episodes, {aug['total_frames']} frames, {size / 2 ** 30:.2f} GiB")
    if args.no_validate:
        return 0
    res = validate(out, Path(args.source_root).expanduser().resolve(), Path(args.report_dir))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
