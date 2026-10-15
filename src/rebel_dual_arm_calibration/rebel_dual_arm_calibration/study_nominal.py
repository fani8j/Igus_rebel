"""Synthetic study: which right base predicts the NOMINAL forward kinematics best, and touch-only vs joint base.

    OMP_NUM_THREADS=3 nice -n 19 python3 -m rebel_dual_arm_calibration.study_nominal --out <dir> [--seeds 6]

The robot runs nominal FK (fitted joint offsets are never applied), so the number that matters for collision
avoidance is the error of ``T_W_R @ FK_nominal(q)`` against where the tip physically is (``nominal_prediction_error``),
not the base error in the identifiable gauge.  Compared bases: the physical truth, the offset-fit base (written to the
YAML), a plain refit with the offsets forced to 0 and the touch-only Kabsch base.  The study is circular in one respect:
the injected error is exactly the modelled one (joint zero offsets), so real gains are smaller.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import solver, synthetic as S, validation
from .base_check import GateOptions, evaluate_gate
from .targets import default_targets_path, load_targets_all
from .transforms import rotation_angle_deg

PRE = ("arm_left_", "arm_right_")


def _err(T, Tt):
    return float(np.linalg.norm(T[:3, 3] - Tt[:3, 3]) * 1e3), float(rotation_angle_deg(T[:3, :3], Tt[:3, :3]))


def run(seed: int, offset_deg: float, repeats: int = 3) -> dict:
    arms, table = load_targets_all(default_targets_path())
    kin = S.default_kinematics(PRE)
    sc = S.make_kin_scene(0.9, seed=seed, offset_deg=offset_deg)
    samples = S.simulate_kin_samples(sc, arms, kin, n_per_arm=30, seed=seed + 10)
    tables = S.simulate_table_obs(sc, table, 10, seed=seed + 20)
    touches = S.simulate_touches(sc, table, kin, repeats=repeats, seed=seed + 30)
    eff = S.effective_truth(sc)
    opts = solver.SolverOptions(joint_offsets=offset_deg > 0)
    res = solver.solve(samples, opts, tables=tables, touches=touches, kin=kin)
    res.report["validation"] = validation.cross_validate(res, solver.SolverOptions(), touches, kin, k=3)
    gate = evaluate_gate(res.report, GateOptions())
    bases = {"physical_truth": eff["T_W_R"], "yaml_base_joint_fit": res.T_world_arm_right_base,
             "nominal_refit_offsets_0": res.T_world_arm_right_base_nominal_refit,
             "touch_only_kabsch": res.T_world_arm_right_base_touch}
    out = {"seed": seed, "offset_deg": offset_deg, "bases": {}}
    for name, T in bases.items():
        if T is None:
            continue
        p = S.nominal_prediction_error(sc, kin, T)
        out["bases"][name] = {"base_vs_identifiable_truth_mm_deg": _err(T, eff["T_W_R"]),
                              "base_vs_physical_unrotated_mm_deg": _err(T, sc.T_W_R),
                              "nominal_prediction_rms_mm": p["rms_mm"], "nominal_prediction_max_mm": p["max_mm"]}
    tb = res.report["touch_base"]
    out["joint_vs_touch"] = tb.get("comparison")
    out["touch_residual_rms_mm"] = tb.get("residual_rms_mm")
    out["touch_nominal_out_of_fit_rms_mm"] = res.report["loop_closure"]["out_of_fit_rms_mm"]
    oc = res.report["loop_closure"].get("offset_corrected")
    out["touch_offset_corrected_out_of_fit_rms_mm"] = None if oc is None else oc["out_of_fit_rms_mm"]
    out["tcp_consistency_rms_mm"] = {a: v["spread_rms_mm"] for a, v in tb["tcp_consistency"]["arms"].items()}
    out["gate"] = {"passed": gate["passed"], "checks": {c["name"]: c["ok"] for c in gate["checks"]}}
    v = res.report["validation"]
    out["keep_offsets"] = v["keep_joint_offsets"]
    return out


def summarize(rows):
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))
    s = {}
    for off in sorted({r["offset_deg"] for r in rows}):
        rr = [r for r in rows if r["offset_deg"] == off]
        d = {"n_seeds": len(rr), "bases": {}}
        for name in rr[0]["bases"]:
            c = [r["bases"][name] for r in rr if name in r["bases"]]
            d["bases"][name] = {
                "base_vs_identifiable_truth_mm_rms": rms([x["base_vs_identifiable_truth_mm_deg"][0] for x in c]),
                "base_vs_physical_unrotated_mm_rms": rms([x["base_vs_physical_unrotated_mm_deg"][0] for x in c]),
                "nominal_prediction_rms_mm": rms([x["nominal_prediction_rms_mm"] for x in c]),
                "nominal_prediction_max_mm": float(max(x["nominal_prediction_max_mm"] for x in c))}
        d["joint_vs_touch_origin_mm_rms"] = rms([r["joint_vs_touch"]["origin_mm"] for r in rr if r["joint_vs_touch"]])
        d["joint_vs_touch_deg_rms"] = rms([r["joint_vs_touch"]["rotation_deg"] for r in rr if r["joint_vs_touch"]])
        d["touch_nominal_out_of_fit_rms_mm"] = float(np.mean([r["touch_nominal_out_of_fit_rms_mm"] for r in rr]))
        d["gate_passed"] = int(sum(r["gate"]["passed"] for r in rr))
        s[f"offset_{off}"] = d
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0, rows = time.time(), []
    for off in (0.0, 0.4):
        for seed in range(args.seeds):
            rows.append(run(seed, off))
            print(f"offset {off} seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    summary = summarize(rows)
    with open(os.path.join(args.out, "nominal_model_study.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "seconds": time.time() - t0}, f, indent=1, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
