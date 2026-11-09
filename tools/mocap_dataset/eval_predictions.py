#!/usr/bin/env python3
"""Offline metrics from the predictions npz written by the handoff's evaluate_mocap.py.

    python3 tools/mocap_dataset/eval_predictions.py --arm right steps_10000_predictions.npz [--group new=0,1,2] [--json out.json]

Uses the arm's six joint channels (left 0..5, right 7..12) and the valid-target mask (repeat-last padding excluded):

* model MAE vs the hold-still baseline (predict the current state for all 30 steps);
* 1-s displacement (horizon index 29 minus the current state) on frames whose TRUE 1-s displacement has an L2 norm above
  --moving-rad (default 0.05 rad): direction cosine between predicted and true (median, share above 0.7, share below 0)
  and the magnitude ratio |pred| / |true| (median);
* the predicted 1-s displacement on frames where the true one is below --still-rad (default 0.02 rad), i.e. how much
  the model moves when the demonstration stays put;
* the first-step offset |pred[0] - state| per frame as the largest joint deviation, in degrees (median, p99, max);
* the share of frames under the TurboVLA client's gates (tools/groot_inference/rollout_config_turbovla_right.yaml):
  first_step = max joint |raw chunk[0] - state| <= --first-step-rad (default 0.15) and target_step = max joint
  |ACT-ensembled target - state| <= --target-step-rad (default 0.10), the ensemble being the client's causal ACT temporal
  ensemble (horizon 30, decay 0.01, a fresh chunk every frame; one run per episode, the chunk of frame s contributes its
  element t - s at frame t).  Offline the state is the recorded one (in a live run it follows the target), so this is an
  approximation of the live pass rate.
* ``--group NAME=i,j,...`` (repeatable): the same metrics over a subset of the episodes (episode_index in the npz), e.g.
  the held-out episodes of one source in a combined run.

Same definitions as used for the left-arm clean9 run (6000: model 0.0511 vs hold 0.0319, cosine median 0.79, 12 %
negative, first step median 3.9 deg)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM_FIRST = {"left": 0, "right": 7}
HORIZON_1S = 29        # 30 Hz, offsets 0..29 -> index 29 is ~1 s ahead
GATE_FIRST_STEP_RAD = 0.15     # rollout_config_turbovla_right.yaml gates.first_step_max_rad
GATE_TARGET_STEP_RAD = 0.10    # ... gates.max_target_step_rad
ENSEMBLE_DECAY = 0.01          # ... ensemble.decay (horizon = chunk length)


def act_ensemble(chunks, decay=ENSEMBLE_DECAY):
    """Causal ACT temporal ensemble of consecutive chunks (T, H, D) -> (T, D) target per frame; same maths as
    ACTTemporalEnsembler in groot_ros_rollout.py: at frame t the chunks issued at s = max(0, t-H+1)..t contribute their
    element t - s, oldest first with weights exp(-decay * i)."""
    T, H, _ = chunks.shape
    out = np.empty((T, chunks.shape[2]))
    for t in range(T):
        first = max(0, t - H + 1)
        cand = np.stack([chunks[s, t - s] for s in range(first, t + 1)])
        w = np.exp(-decay * np.arange(len(cand)))
        out[t] = (cand * (w / w.sum())[:, None]).sum(0)
    return out


def gate_shares(prediction, state, episodes, first, first_step_rad=GATE_FIRST_STEP_RAD,
                target_step_rad=GATE_TARGET_STEP_RAD, decay=ENSEMBLE_DECAY):
    """Share of frames that pass the first_step gate, the target_step gate and both (see the module docstring).
    `episodes`: episode_index per frame; frames of an episode must be consecutive and ordered."""
    sl = slice(first, first + 6)
    pred, cur = prediction[:, :, sl], state[:, sl]
    episodes = np.asarray(episodes)
    first_dev = np.abs(pred[:, 0] - cur).max(axis=1)
    target_dev = np.empty(len(pred))
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        target_dev[idx] = np.abs(act_ensemble(pred[idx], decay) - cur[idx]).max(axis=1)
    ok1, ok2 = first_dev <= first_step_rad, target_dev <= target_step_rad
    return {"first_step_pass": float(ok1.mean()), "target_step_pass": float(ok2.mean()), "both_pass": float((ok1 & ok2).mean()),
            "target_step_dev_rad_median": float(np.median(target_dev)), "target_step_dev_rad_p95": float(np.percentile(target_dev, 95)),
            "first_step_dev_rad_median": float(np.median(first_dev)), "first_step_dev_rad_p95": float(np.percentile(first_dev, 95)),
            "limits_rad": {"first_step": first_step_rad, "target_step": target_step_rad, "ensemble_decay": decay}}


def metrics(prediction, target, state, valid, first, moving_rad=0.05, still_rad=0.02):
    sl = slice(first, first + 6)
    pred, true, cur = prediction[:, :, sl], target[:, :, sl], state[:, sl]
    mask = valid.astype(bool)
    model = float(np.abs(pred - true)[mask].mean())
    hold = float(np.abs(cur[:, None, :] - true)[mask].mean())
    ok = mask[:, HORIZON_1S]
    dp, dt = pred[ok, HORIZON_1S] - cur[ok], true[ok, HORIZON_1S] - cur[ok]
    npred, ntrue = np.linalg.norm(dp, axis=1), np.linalg.norm(dt, axis=1)
    moving = ntrue > moving_rad
    cos = (dp[moving] * dt[moving]).sum(1) / (npred[moving] * ntrue[moving] + 1e-12)
    still = ntrue < still_rad
    first_step = np.degrees(np.abs(pred[:, 0] - cur).max(axis=1))
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None  # noqa: E731
    return {
        "frames": int(len(pred)), "model_mae_rad": model, "hold_mae_rad": hold, "model_over_hold": model / hold,
        "moving_frames": int(moving.sum()),
        "cosine_median": q(cos, 50), "cosine_share_above_0_7": float((cos > .7).mean()) if len(cos) else None,
        "cosine_share_below_0": float((cos < 0).mean()) if len(cos) else None,
        "magnitude_ratio_median": q(npred[moving] / ntrue[moving], 50),
        "still_frames": int(still.sum()), "still_pred_motion_rad_median": q(npred[still], 50),
        "still_pred_motion_rad_p95": q(npred[still], 95), "still_pred_motion_deg_median": q(np.degrees(npred[still]), 50),
        "first_step_deg_median": q(first_step, 50), "first_step_deg_p99": q(first_step, 99),
        "first_step_deg_max": float(first_step.max()),
        "definitions": {"moving_rad": moving_rad, "still_rad": still_rad, "norm": "L2 over the arm's 6 joints at horizon "
                        f"index {HORIZON_1S}; first step = max |joint| deviation"}}


def analyse(path, arm, moving_rad=0.05, still_rad=0.02, groups=None, first_step_rad=GATE_FIRST_STEP_RAD,
            target_step_rad=GATE_TARGET_STEP_RAD):
    z = np.load(path)
    first = ARM_FIRST[arm]
    gates = lambda sel: gate_shares(z["prediction_rad"][sel], z["state_rad"][sel], z["episode_index"][sel], first,  # noqa: E731
                                    first_step_rad, target_step_rad)
    sel_of = lambda eps: np.isin(z["episode_index"], list(eps))  # noqa: E731
    subset = lambda sel: {**metrics(z["prediction_rad"][sel], z["target_rad"][sel], z["state_rad"][sel], z["valid"][sel],  # noqa: E731
                                    first, moving_rad, still_rad), "gates": gates(sel)}
    result = {"file": str(path), "arm": arm, "all": subset(np.ones(len(z["episode_index"]), dtype=bool)), "per_episode": {}}
    for ep in np.unique(z["episode_index"]):
        result["per_episode"][int(ep)] = subset(z["episode_index"] == ep)
    if groups:
        result["groups"] = {name: {"episodes": sorted(int(e) for e in eps), **subset(sel_of(eps))} for name, eps in groups.items()}
    return result


def parse_groups(items):
    groups = {}
    for item in items or []:
        name, _, eps = item.partition("=")
        if not name or not eps:
            raise ValueError(f"--group expects NAME=i,j,... got {item!r}")
        groups[name] = [int(e) for e in eps.split(",")]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, nargs="+")
    parser.add_argument("--arm", choices=sorted(ARM_FIRST), required=True)
    parser.add_argument("--moving-rad", type=float, default=0.05)
    parser.add_argument("--still-rad", type=float, default=0.02)
    parser.add_argument("--first-step-rad", type=float, default=GATE_FIRST_STEP_RAD)
    parser.add_argument("--target-step-rad", type=float, default=GATE_TARGET_STEP_RAD)
    parser.add_argument("--group", action="append", metavar="NAME=i,j,...",
                        help="also report this subset of episode_index values (repeatable)")
    parser.add_argument("--json", type=Path, help="write all results here")
    args = parser.parse_args()
    groups = parse_groups(args.group)
    results = [analyse(p, args.arm, args.moving_rad, args.still_rad, groups, args.first_step_rad, args.target_step_rad)
               for p in args.npz]
    f = lambda v, spec: "n/a" if v is None else format(v, spec)  # noqa: E731
    pct = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    def line(label, a):
        g = a["gates"]
        print(f"  {label}MAE model {a['model_mae_rad']:.4f} hold {a['hold_mae_rad']:.4f} ({a['model_over_hold']:.2f}x) | "
              f"cos median {f(a['cosine_median'], '.2f')} >0.7 {pct(a['cosine_share_above_0_7'])} "
              f"<0 {pct(a['cosine_share_below_0'])} "
              f"(moving {a['moving_frames']}) mag {f(a['magnitude_ratio_median'], '.2f')} | still {a['still_frames']} frames pred "
              f"{f(a['still_pred_motion_deg_median'], '.1f')} deg | first step median {a['first_step_deg_median']:.1f} "
              f"p99 {a['first_step_deg_p99']:.1f} max {a['first_step_deg_max']:.1f} deg | gates first_step "
              f"{pct(g['first_step_pass'])} target_step {pct(g['target_step_pass'])} both {pct(g['both_pass'])}")
    for r in results:
        print(r["file"])
        line("", r["all"])
        for name, sub in r.get("groups", {}).items():
            line(f"[{name}] ", sub)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
