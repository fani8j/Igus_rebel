"""Synthetic study: joint zero offsets, table-board touches, held-out validation (no ROS, no images).

    OMP_NUM_THREADS=1 nice -n 19 python3 -m rebel_dual_arm_calibration.study_loop --out <dir> [--seeds 6] [--offset-deg 0.4]

For every seed the real ReBeL URDF (both arms with xeg32_tool_tip) is posed in joint space, joint zero offsets of
``offset_deg`` .. 1.4 x per joint (random sign) are injected into the physical arms, 30 poses per arm, a table board
seen 10 times and touched at 4 points by both arms.  Errors are against the truth in the identifiable gauge
(``synthetic.effective_truth``) and, for the right base, also against the physical base.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import solver, synthetic as S, validation
from .targets import default_targets_path, load_targets_all
from .transforms import rotation_angle_deg, xyz_rpy_to_T

PRE = ("arm_left_", "arm_right_")


def _err(T, Tt):
    return float(np.linalg.norm(T[:3, 3] - Tt[:3, 3]) * 1e3), rotation_angle_deg(T[:3, :3], Tt[:3, :3])


def run_seed(seed: int, offset_deg: float, n_per_arm: int = 30) -> dict:
    arms, table = load_targets_all(default_targets_path())
    kin = S.default_kinematics(PRE)
    sc = S.make_kin_scene(0.9, seed=seed, offset_deg=offset_deg)
    samples = S.simulate_kin_samples(sc, arms, kin, n_per_arm=n_per_arm, seed=seed + 10)
    tables = S.simulate_table_obs(sc, table, 10, seed=seed + 20)
    touches = S.simulate_touches(sc, table, kin, seed=seed + 30)
    eff = S.effective_truth(sc)
    out = {"seed": seed, "offset_deg": offset_deg, "configs": {}}
    fast = dict(loop_closure_variants=False, intrinsics_report=False)
    for name, offs, use_touch in (("no_offsets_no_touches", False, False), ("no_offsets_touches", False, True),
                                  ("offsets_no_touches", True, False), ("offsets_touches", True, True)):
        opts = solver.SolverOptions(joint_offsets=offs, **({} if use_touch else fast))
        res = solver.solve(samples, opts, tables=tables, touches=touches if use_touch else [], kin=kin)
        lc = res.report["loop_closure"]
        out["configs"][name] = {
            "right_base_vs_identifiable_truth_mm_deg": _err(res.T_world_arm_right_base, eff["T_W_R"]),
            "right_base_vs_physical_truth_mm_deg": _err(res.T_world_arm_right_base, sc.T_W_R),
            "camera_vs_identifiable_truth_mm_deg": _err(res.T_world_optical, eff["T_W_O"]),
            "rms_px": res.report["joint"]["rms_px"]["all"],
            # lc["out_of_fit_rms_mm"] is evaluated with NOMINAL FK; the offset-corrected value is a diagnostic
            "touch_out_of_fit_rms_mm": None if lc is None else lc.get("offset_corrected", lc)["out_of_fit_rms_mm"],
            "touch_out_of_fit_nominal_fk_rms_mm": None if lc is None else lc["out_of_fit_rms_mm"],
            "touch_in_fit_rms_mm": None if lc is None else lc.get("offset_corrected", lc)["final"]["rms_mm"]["all"],
            "right_base_std_mm": res.report["joint"]["calibration_position_std_mm"]["arm_right_base"],
            "warnings": len(res.report["warnings"])}
        if name == "no_offsets_touches":
            v = validation.cross_validate(res, solver.SolverOptions(), touches, kin, k=5)
            out["validation"] = {k: v[k] for k in ("without_offsets", "with_offsets", "keep_joint_offsets",
                                                   "recommendation")}
            for key in ("without_offsets", "with_offsets"):
                out["validation"][key] = {k: x for k, x in v[key].items() if k != "folds"}
    # detection of inconsistent data (offsets modelled, touches inconsistent)
    wrong = xyz_rpy_to_T([0.004, 0.003, -0.003], [0, 0, 0])
    t_wrong = S.simulate_touches(sc, table, kin, seed=seed + 30, right_base_error=wrong)
    r = solver.solve(samples, solver.SolverOptions(joint_offsets=True), tables=tables, touches=t_wrong, kin=kin)
    out["wrong_right_base_5.8mm"] = {"touch_camera_only_rms_mm":
                                     r.report["loop_closure"]["offset_corrected"]["camera_only"]["rms_mm"],
                                     "warned": any("touch loop closure" in w for w in r.report["warnings"])}
    tb2 = S.simulate_table_obs(sc, table, 10, seed=seed + 20, board_scale=1.015)
    t_sc = S.simulate_touches(sc, table, kin, seed=seed + 30, board_scale=1.015)
    r = solver.solve(samples, solver.SolverOptions(joint_offsets=True), tables=tb2, touches=t_sc, kin=kin)
    out["wrong_board_scale_1.5pct"] = {"touch_camera_only_rms_mm":
                                       r.report["loop_closure"]["offset_corrected"]["camera_only"]["rms_mm"],
                                       "warned": any("touch loop closure" in w for w in r.report["warnings"])}
    return out


def summarize(rows: list) -> dict:
    def rms(vals):
        return float(np.sqrt(np.mean(np.square(vals))))
    s = {"n_seeds": len(rows), "configs": {}}
    for name in rows[0]["configs"]:
        c = [r["configs"][name] for r in rows]
        s["configs"][name] = {
            "right_base_mm_rms": rms([x["right_base_vs_identifiable_truth_mm_deg"][0] for x in c]),
            "right_base_deg_rms": rms([x["right_base_vs_identifiable_truth_mm_deg"][1] for x in c]),
            "right_base_vs_physical_mm_rms": rms([x["right_base_vs_physical_truth_mm_deg"][0] for x in c]),
            "camera_mm_rms": rms([x["camera_vs_identifiable_truth_mm_deg"][0] for x in c]),
            "rms_px": float(np.mean([x["rms_px"] for x in c])),
            "touch_out_of_fit_rms_mm": (None if c[0]["touch_out_of_fit_rms_mm"] is None else
                                    float(np.mean([x["touch_out_of_fit_rms_mm"] for x in c]))),
            "right_base_std_mm": float(np.mean([x["right_base_std_mm"] for x in c]))}
    v = [r["validation"] for r in rows]
    s["heldout"] = {
        key: {"px": float(np.mean([x[key]["heldout_rms_px"] for x in v])),
              "mm": float(np.mean([x[key]["heldout_rms_mm"] for x in v])),
              "touch_mm": float(np.mean([x[key]["heldout_touch_rms_mm"] for x in v]))}
        for key in ("without_offsets", "with_offsets")}
    s["offsets_recommended_in"] = int(sum(1 for x in v if x["keep_joint_offsets"]))
    for key in ("wrong_right_base_5.8mm", "wrong_board_scale_1.5pct"):
        s[key] = {"camera_only_touch_rms_mm_mean": float(np.mean([r[key]["touch_camera_only_rms_mm"]["all"]
                                                                 for r in rows])),
                  "warned_in": int(sum(r[key]["warned"] for r in rows))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--offset-deg", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    for seed in range(args.seeds):
        rows.append(run_seed(seed, args.offset_deg))
        print(f"seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    summary = summarize(rows)
    with open(os.path.join(args.out, "loop_closure_study.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "seconds": time.time() - t0}, f, indent=1, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
