#!/usr/bin/env python3
"""Convert one episode of a robot_remote LeRobot v3.0 dataset (mocap/teleop recording of ONE ReBeL arm) into the
joint plan JSON that tools/lid_open_friend/scripts/lid_open/publish_plan.py plays (`--mode trajectory`).

    python3 lerobot_to_plan.py DATASET_DIR [--arm arm_left] [--source state|action] [--start-s S] [--end-s E]
                               [--segment-s 10] [--hz 5] [--out plan.json]

Without --out nothing is written (summary and checks only).  The converter FAILS CLOSED: it refuses (exit 2, no
file) an all-zero arm, non-finite values, a joint outside the real URDF limits minus --limit-margin-deg, and a
per-frame jump above --max-jump-deg (wrap / glitch).

Plan layout (what publish_plan.py's build_targets / traj_groups see):
  * one task, `start_q` = first point of the slice.  publish_plan.py first moves the arm from where it is to
    start_q (its own confirmed "approach" group, limited by --max-speed, no step limit);
  * the slice is split into ~--segment-s chunks (data time), each ONE `arc` step: the player publishes one
    JointTrajectory per arc, asks Enter before it and checks the arrival after it.  The points after the first
    inside an arc are chained with no further confirmation;
  * `park_q` = LAST point by default, so the player's final "park" move is a no-op (no unplanned straight-line
    move back across the box).  --park start sets park_q = start_q (a joint-space straight line back, opt-in).
    The player's own `--no-park` drops that last group completely.

Timing: the player ignores the recorded timestamps.  It times every point as
max(max|dq| / --max-speed, 0.2 s) (the first point of every arc: at least 1 s).  So the replay speed is set by the
point spacing: points are resampled at --hz (default 5) and a point is dropped while it is closer than
--min-step-deg (default 1 deg = 0.0175 rad) to the last kept one, so dwell phases do not become hundreds of 0.2 s
stop-and-go segments.  Segments larger than 0.03 rad (1.7 deg) play at exactly --max-speed (max-norm); smaller
ones are floored at 0.2 s, i.e. play slower than --max-speed (never faster).
The summary prints the resulting replay duration, the time scale and the path error of the downsampling
(full-rate samples vs. the polyline through the kept points).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lerobot_episode import Episode, load_episode  # noqa: E402

DEFAULT_URDF = REPO / "src" / "igus_rebel_description" / "urdf" / "igus_rebel2.urdf"
PLAYER_MAX_SPEED = 0.15          # publish_plan.py default --max-speed [rad/s]
PLAYER_MIN_DT = 0.2              # publish_plan.py traj_times min_dt [s]
PLAYER_MIN_FIRST = 1.0           # publish_plan.py traj_times min_first [s]


class PlanError(ValueError):
    """Input refused (nothing is written)."""


# ---------------------------------------------------------------- limits / checks
def urdf_limits(path: Path = DEFAULT_URDF) -> np.ndarray:
    """(6, 2) lower/upper of joint1..6 from the URDF (fails if one is missing)."""
    root = ET.parse(path).getroot()
    out = []
    for i in range(1, 7):
        j = next((j for j in root.findall("joint") if j.get("name") == f"joint{i}"), None)
        lim = None if j is None else j.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            raise PlanError(f"{path}: joint{i} has no lower/upper limit")
        out.append((float(lim.get("lower")), float(lim.get("upper"))))
    return np.asarray(out)


def check_series(t: np.ndarray, q: np.ndarray, limits: np.ndarray, margin_rad: float, max_jump_rad: float) -> None:
    """Raise PlanError for non-finite values, an all-zero arm, a limit violation or an inter-frame jump."""
    if q.ndim != 2 or q.shape[1] != 6:
        raise PlanError(f"expected (N, 6) joint data, got {q.shape}")
    if len(q) < 2:
        raise PlanError(f"only {len(q)} frame(s) in the slice, need at least 2")
    if not (np.isfinite(q).all() and np.isfinite(t).all()):
        raise PlanError("non-finite value (NaN/inf) in the joint data or timestamps")
    if float(np.abs(q).max()) < 1e-6:
        raise PlanError("the arm's joint data is all zeros: this arm was not recorded (wrong --arm?)")
    if np.any(np.diff(t) <= 0):
        raise PlanError("timestamps are not strictly increasing")
    lo, hi = limits[:, 0] + margin_rad, limits[:, 1] - margin_rad
    bad = np.argwhere((q < lo) | (q > hi))
    if len(bad):
        i, j = bad[0]
        raise PlanError(f"joint{j + 1} = {q[i, j]:+.4f} rad at t={t[i]:.2f} s is outside the allowed "
                        f"[{lo[j]:+.4f}, {hi[j]:+.4f}] (URDF limit minus {math.degrees(margin_rad):g} deg); "
                        f"{len(bad)} sample(s) violate a limit")
    step = np.abs(np.diff(q, axis=0))
    k = np.unravel_index(int(np.argmax(step)), step.shape)
    if step[k] > max_jump_rad:
        raise PlanError(f"jump of {math.degrees(step[k]):.1f} deg in joint{k[1] + 1} between frames {k[0]} and "
                        f"{k[0] + 1} (t={t[k[0]]:.2f} s) exceeds --max-jump-deg {math.degrees(max_jump_rad):g}: "
                        "wrap or recording glitch")


# ---------------------------------------------------------------- downsampling / segmenting
def downsample(t: np.ndarray, q: np.ndarray, hz: float, min_step_rad: float) -> np.ndarray:
    """Indices of the kept points: nearest frame to every 1/hz of data time, then drop a point while it is within
    min_step_rad (max-norm) of the last kept one.  First and last frame are always kept (a last point closer than
    1e-6 rad to the last kept one replaces it)."""
    n = len(t)
    if hz <= 0:
        raise PlanError("--hz must be > 0")
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    hi = np.clip(np.searchsorted(t, grid), 0, n - 1)               # first frame at/after the grid time
    lo = np.clip(hi - 1, 0, n - 1)                                  # and the one before it: take the nearer
    idx = np.where(np.abs(t[lo] - grid) < np.abs(t[hi] - grid), lo, hi)
    idx = np.unique(np.concatenate([[0], idx, [n - 1]]))
    keep = [int(idx[0])]
    for i in idx[1:-1]:
        if np.max(np.abs(q[i] - q[keep[-1]])) >= min_step_rad:
            keep.append(int(i))
    last = int(idx[-1])
    if np.max(np.abs(q[last] - q[keep[-1]])) < 1e-6 and len(keep) > 1:
        keep[-1] = last
    elif last != keep[-1]:
        keep.append(last)
    return np.asarray(keep)


def shape_error(t: np.ndarray, q: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """(path error, time error) of the downsampling [rad]: max Euclidean joint-space distance of any full-rate sample
    to the polyline through the kept points (shape only), and the max |q - time-interpolated kept polyline| over all
    joints (shape plus timing, informational: the player re-times the points anyway)."""
    time_err = float(np.abs(np.column_stack([np.interp(t, t[keep], q[keep, j]) for j in range(6)]) - q).max())
    a, b = q[keep[:-1]], q[keep[1:]]                                   # (M, 6) segments
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    worst = 0.0
    for i0 in range(0, len(q), 512):                                   # chunked: (chunk, M, 6) stays small
        p = q[i0:i0 + 512, None, :]
        u = np.clip(((p - a) * ab).sum(2) / denom, 0.0, 1.0)
        d = np.linalg.norm(p - (a + u[..., None] * ab), axis=2).min(1)
        worst = max(worst, float(d.max()))
    return worst, time_err


def split_segments(t_kept: np.ndarray, segment_s: float, min_tail: float = 0.25) -> list[tuple[int, int]]:
    """[(a, b)) index ranges into the kept points, one per ~segment_s of data time (relative to t_kept[0]).
    A final chunk shorter than min_tail * segment_s is merged into the previous one.  segment_s <= 0: one chunk."""
    n = len(t_kept)
    if segment_s <= 0 or n < 2:
        return [(0, n)]
    rel = t_kept - t_kept[0]
    cell = np.floor(rel / segment_s + 1e-9).astype(int)
    bounds = [0] + [i for i in range(1, n) if cell[i] != cell[i - 1]] + [n]
    segs = list(zip(bounds[:-1], bounds[1:]))
    if len(segs) > 1 and (rel[-1] - rel[segs[-1][0]]) < min_tail * segment_s:
        segs[-2] = (segs[-2][0], segs[-1][1])
        segs.pop()
    return segs


def player_times(prev, qs, max_speed: float = PLAYER_MAX_SPEED) -> list[float]:
    """Replica of publish_plan.traj_times (time_from_start of every point of one trajectory)."""
    t, out, p = 0.0, [], list(prev)
    for i, q in enumerate(qs):
        t += max(max(abs(a - b) for a, b in zip(q, p)) / max_speed, PLAYER_MIN_FIRST if i == 0 else PLAYER_MIN_DT)
        out.append(t)
        p = list(q)
    return out


# ---------------------------------------------------------------- plan
def build_plan(ep: Episode, arm: str, q: np.ndarray, t: np.ndarray, keep: np.ndarray, segs, *, source: str,
               park: str, start_s: float, end_s: float, hz: float, min_step_deg: float, max_speed: float) -> dict:
    qk = q[keep]
    names = [f"{arm}_joint{i}" for i in range(1, 7)]
    r = lambda v: [round(float(x), 6) for x in v]                      # noqa: E731
    steps = []
    for n, (a, b) in enumerate(segs, 1):
        pts = qk[a:b] if a > 0 else qk[1:b]                           # point 0 is start_q: the approach group goes there
        if len(pts) == 0:
            continue
        t0, t1 = float(t[keep[a]]), float(t[keep[b - 1]])
        steps.append({"n": len(steps) + 1, "kind": "arc",
                      "label": f"seg {n}/{len(segs)} data {t0 - t[0]:.1f}-{t1 - t[0]:.1f}s",
                      "qs": [r(p) for p in pts]})
    start_q, park_q = r(qk[0]), r(qk[-1] if park == "last" else qk[0])
    return {
        "version": 1,
        "created": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "box_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},              # placeholder: the recording's box pose is unknown
        "gantry": None,
        "converter": {"tool": "tools/mocap_dataset/lerobot_to_plan.py", "dataset": str(ep.root), "episode": ep.episode_index,
                      "task": ep.task, "arm": arm, "source": source, "slice_s": [round(start_s, 3), round(end_s, 3)],
                      "hz": hz, "min_step_deg": min_step_deg, "park": park, "points": int(len(keep))},
        "tasks": [{
            "idx": 0, "arm": arm, "flap": ep.root.name, "how": "lerobot-replay", "grab": "n/a", "shift_mm": 0.0,
            "replay": True, "demo": f"{ep.root.name} ep{ep.episode_index} [{source}] {start_s:.1f}-{end_s:.1f}s",
            "hold": 0.0, "joint_names": names, "start_q": start_q, "park_q": park_q, "steps": steps}],
    }


def estimate(plan: dict, max_speed: float) -> list[dict]:
    """Per-step replay estimate with the player's own timing rule."""
    task = plan["tasks"][0]
    prev, out = task["start_q"], []
    for st in task["steps"]:
        ts = player_times(prev, st["qs"], max_speed)
        out.append({"label": st["label"], "points": len(st["qs"]), "replay_s": ts[-1]})
        prev = st["qs"][-1]
    return out


def fk_z(q: np.ndarray, urdf: Path, base: str, tip: str, link6: str = "link6"):
    """(min z, max z, argmin index) of `tip` in `base` over q; None if the FK module/URDF is unavailable."""
    try:
        try:
            from rebel_dual_arm_calibration.kinematics import ArmKinematics, Urdf
        except ImportError:
            sys.path.insert(0, str(REPO / "src" / "rebel_dual_arm_calibration"))
            from rebel_dual_arm_calibration.kinematics import ArmKinematics, Urdf
        k = ArmKinematics.from_urdf(Urdf.load(str(urdf)), base, link6, tip if tip != link6 else "")
        chain = k.chain_tcp if (tip != link6 and k.chain_tcp is not None) else k.chain6
        z = chain.fk(q)[:, 2, 3]
        return float(z.min()), float(z.max()), int(np.argmin(z))
    except Exception as e:                                             # noqa: BLE001
        return f"unavailable ({e})"


# ---------------------------------------------------------------- main
def convert(args) -> tuple[dict, str]:
    ep = load_episode(args.dataset, args.episode)
    arm = args.arm
    if f"{arm}_joint1" not in ep.names:
        raise PlanError(f"{arm} is not in this dataset (names: {ep.names})")
    q_all = ep.arm(arm, args.source)
    t_all = ep.timestamp - ep.timestamp[0]
    start_s = max(0.0, args.start_s)
    end_s = float(t_all[-1]) if args.end_s is None else min(float(args.end_s), float(t_all[-1]))
    sel = (t_all >= start_s - 1e-9) & (t_all <= end_s + 1e-9)
    t, q = t_all[sel], q_all[sel]
    if float(np.abs(q_all).max()) < 1e-6:
        raise PlanError(f"{arm} {args.source} data is all zeros in the whole episode: this arm was not recorded")
    limits = urdf_limits(Path(args.urdf_limits))
    check_series(t, q, limits, math.radians(args.limit_margin_deg), math.radians(args.max_jump_deg))
    keep = downsample(t, q, args.hz, math.radians(args.min_step_deg))
    segs = split_segments(t[keep], args.segment_s)
    plan = build_plan(ep, arm, q, t, keep, segs, source=args.source, park=args.park, start_s=start_s, end_s=end_s,
                      hz=args.hz, min_step_deg=args.min_step_deg, max_speed=args.max_speed)

    shape_err = shape_error(t, q, keep)
    est = estimate(plan, args.max_speed)
    data_dur = float(t[-1] - t[0])
    replay = sum(e["replay_s"] for e in est)
    dt = np.diff(t)
    peak_raw = float((np.abs(np.diff(q, axis=0)) / dt[:, None]).max())
    w = max(1, int(round(0.2 / float(np.median(dt)))))
    peak_w = float((np.abs(q[w:] - q[:-w]) / (t[w:] - t[:-w])[:, None]).max()) if len(q) > w else peak_raw
    lines = [
        f"dataset   {ep.root.name} ep{ep.episode_index}  '{ep.task}'  {len(ep.timestamp)} frames @ {ep.fps:g} fps  "
        f"({t_all[-1]:.1f} s)",
        f"arm       {arm}  source={args.source}  slice {start_s:.1f}-{end_s:.1f} s  ({len(q)} frames, {data_dur:.1f} s)",
        f"limits    OK: all joints inside URDF limits - {args.limit_margin_deg:g} deg; max frame jump "
        f"{math.degrees(float(np.abs(np.diff(q, axis=0)).max())):.2f} deg (refuse > {args.max_jump_deg:g})",
        f"points    {len(keep)} kept of {len(q)} ({args.hz:g} Hz grid, min step {args.min_step_deg:g} deg); downsampling "
        f"path error {math.degrees(shape_err[0]):.2f} deg (joint space, to the polyline), "
        f"{math.degrees(shape_err[1]):.2f} deg if time-interpolated",
        f"data peak joint speed {peak_raw:.2f} rad/s per frame, {peak_w:.2f} rad/s over 0.2 s",
        f"start_q   {[round(math.degrees(v), 1) for v in plan['tasks'][0]['start_q']]} deg   (approach group: "
        "player moves the arm there from where it is, at --max-speed, behind its own Enter)",
        f"park_q    {'= last point (no-op park)' if args.park == 'last' else '= start_q (straight joint line back)'}",
        f"segments  {len(plan['tasks'][0]['steps'])} arc step(s) of ~{args.segment_s:g} s data time:",
    ]
    for s, e in zip(plan["tasks"][0]["steps"], est):
        lines.append(f"            {e['label']:<34} {e['points']:>4} pts   replay {e['replay_s']:6.1f} s at {args.max_speed:g} rad/s")
    lines.append(f"replay    {replay:.0f} s for {data_dur:.0f} s of data (time scale x{replay / data_dur:.2f}) at --max-speed "
                 f"{args.max_speed:g} rad/s, excluding the approach to start_q and the Enter pauses")
    z = fk_z(q, Path(args.urdf), args.base_link, args.tip_link, args.link6)
    if isinstance(z, tuple):
        lines.append(f"FK        {args.tip_link} z in {args.base_link}: min {z[0]:+.3f} m (t={t[z[2]]:.1f} s), max {z[1]:+.3f} m "
                     "(arm base frame, not table height)")
    else:
        lines.append(f"FK        {z}")
    return plan, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="LeRobot v3.0 dataset directory (one episode per --episode)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--arm", default="arm_left", choices=["arm_left", "arm_right"])
    ap.add_argument("--source", choices=["state", "action"], default="state",
                    help="state = what the arm did (default); action = the commanded target, ~0.28 s ahead of state")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=None)
    ap.add_argument("--segment-s", type=float, default=10.0, help="data seconds per arc step (Enter + arrival check between); 0 = one arc")
    ap.add_argument("--hz", type=float, default=5.0, help="resampling rate before the min-step filter (default 5)")
    ap.add_argument("--min-step-deg", type=float, default=1.0, help="drop points closer than this (max-norm) to the last kept one")
    ap.add_argument("--max-speed", type=float, default=PLAYER_MAX_SPEED, help="player --max-speed to estimate the replay time for")
    ap.add_argument("--limit-margin-deg", type=float, default=2.0, help="stay this far inside the URDF joint limits (the player adds its own 1 deg)")
    ap.add_argument("--max-jump-deg", type=float, default=5.7, help="refuse an inter-frame jump above this (wrap / glitch)")
    ap.add_argument("--park", choices=["last", "start"], default="last",
                    help="park_q: last = no-op park (default), start = straight joint-space move back to the first point")
    ap.add_argument("--urdf-limits", default=str(DEFAULT_URDF), help="URDF with the real joint1..6 limits")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF), help="URDF for the FK z summary (expanded xacro for a TCP link)")
    ap.add_argument("--base-link", default="base_link")
    ap.add_argument("--link6", default="link6", help="wrist link of the FK chain (arm_left_link6 in the dual-table URDF)")
    ap.add_argument("--tip-link", default="link6", help="link whose z is reported (e.g. arm_left_knife_c_tcp_point)")
    ap.add_argument("--out", default=None, help="plan JSON to write (without it nothing is written)")
    args = ap.parse_args(argv)
    try:
        plan, summary = convert(args)
    except (PlanError, ValueError, OSError, KeyError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(summary)
    if args.out:
        out = Path(args.out).expanduser()
        out.write_text(json.dumps(plan, indent=1))
        print(f"wrote     {out}")
    else:
        print("no --out: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
