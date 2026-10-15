"""``ros2 run rebel_dual_arm_calibration solve <session_dir>``: solve and write the calibration."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional, Sequence

from .session import SessionMismatchError, load_sessions_full
from .base_check import GateOptions, evaluate_gate
from .solver import (ARMS, SolverError, SolverOptions, choose_written_fit, deployed_poses, format_report,
                     format_tcp_report, load_tcp_file, parse_tcp_file_args, prepare_touch_tcp, set_written_base, solve,
                     write_calibration_yaml)
from .transforms import R_to_quat, inv_T


def build_notes(rep: dict, session_dirs, base_from: str = "joint") -> str:
    j = rep["joint"]
    wb = rep.get("written_base") or {}
    sr = {"std_xyz_mm": wb.get("std_xyz_mm") or j["arm_right_base"]["std_xyz_mm"],
          "std_rpy_deg": wb.get("std_rpy_deg") or j["arm_right_base"]["std_rpy_deg"]}
    sc = j["camera_link"]
    lc = rep.get("loop_closure")
    extra = ""
    g = rep.get("trust_gate")
    if g:
        extra += "TRUST GATE " + g["summary"] + "; "
        cp = rep.get("check_points") or {}
        if cp.get("points"):
            extra += (f"check-point error (TRE, held-out, nominal FK) max {cp['max_mm']:.2f} mm / RMS {cp['rms_mm']:.2f} mm "
                      f"over {cp['n_points']} point(s) - accuracy demonstrated at the check points only; ")
        else:
            extra += "no check points (nothing held out); "
        extra += f"recommended_padding_mm = {g['recommended_padding_mm']:.0f} (max(20, 2 x max check-point error + 10)); "
    if base_from == "touches":
        extra += "arm_right_base from touch-only Kabsch (board crosshairs + fit stickers); "
    else:
        extra += ("arm_right_base from the "
                  + ("offset fit (joint-1 offset absorbed in the base yaw; offsets never applied)"
                     if wb.get("fit") == "offset_fit" else "nominal fit") + f" [{wb.get('reason', '')}]; ")
    tb = rep.get("touch_base") or {}
    if tb.get("available") and tb.get("comparison"):
        c = tb["comparison"]
        extra += (f"written vs touch-only base {c['origin_mm']:.2f} mm origin / {c['reach_mm']:.2f} mm reach / "
                  f"{c['rotation_deg']:.3f} deg; ")
    if lc:
        oof = lc.get("out_of_fit_rms_mm")
        extra += (f"board-touch fit residual (FRE, in-sample) {lc['fit_residual_fre_rms_mm']:.2f} mm RMS"
                  + (f", out-of-fit {oof:.2f} mm ({lc['out_of_fit_source']})" if oof is not None else ", no out-of-fit number")
                  + "; ")
    for arm_, e_ in (rep.get("tcp") or {}).items():
        if e_.get("rebased"):
            extra += (f"{arm_} touches re-based onto TCP from {e_['tcp_file']} (mean shift {e_['shift_mm']['mean']:.2f} mm); ")
    if j.get("joint_offsets", {}).get("enabled"):
        extra += "joint offsets used as nuisance parameters only (never applied to the robot); "
    if (j.get("intrinsics") or {}).get("source") == "file":
        extra += "intrinsics from --intrinsics-file; "
    if rep.get("compare"):
        extra += rep["compare"]["text"] + "; "
    mc = rep.get("uncertainty_mc")
    if mc:
        rb = mc["arm_right_base"]
        extra += (f"MC 95 % half widths ({mc['n_used']} draws) arm_right_base xyz mm "
                  f"{[round(rb[k]['half_width95'], 2) for k in ('x_mm', 'y_mm', 'z_mm')]} yaw deg "
                  f"{round(rb['yaw_deg']['half_width95'], 3)}; ")
    return (extra + f"samples used L{rep['n_samples_used']['left']}/R{rep['n_samples_used']['right']} "
            f"(rejected L{rep['n_samples_rejected']['left']}/R{rep['n_samples_rejected']['right']}); "
            f"rms {j['rms_px']['all']:.2f} px; "
            f"arm_right_base std xyz mm {[round(v, 2) for v in sr['std_xyz_mm']]} rpy deg "
            f"{[round(v, 3) for v in sr['std_rpy_deg']]}; "
            f"camera_link std xyz mm {[round(v, 2) for v in sc['std_xyz_mm']]} rpy deg "
            f"{[round(v, 3) for v in sc['std_rpy_deg']]}; "
            f"warnings {len(rep['warnings'])}; dataset "
            + ", ".join(os.path.abspath(d) for d in ([session_dirs] if isinstance(session_dirs, str) else session_dirs)))


def write_easy_handeye_calibs(out_dir: str, res, session_dir: str, base_from: str = "joint") -> list:
    """One easy_handeye2-style eye_on_base .calib per arm (camera pose in that arm's base frame)."""
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    written = []
    T_R, T_C = deployed_poses(res, base_from)
    for arm in ARMS:
        base_in_world = T_R if arm == "right" else None
        T_base_cam = inv_T(base_in_world) @ T_C if base_in_world is not None else T_C
        q = R_to_quat(T_base_cam[:3, :3])
        d = {"parameters": {"name": f"dual_arm_{arm}", "calibration_type": "eye_on_base",
                            "robot_base_frame": f"arm_{arm}_base_link",
                            "robot_effector_frame": f"arm_{arm}_link6",
                            "tracking_base_frame": "camera_link",
                            "tracking_marker_frame": f"calib_target_{arm}",
                            "freehand_robot_movement": True, "move_group_namespace": "/", "move_group": f"arm_{arm}"},
             "transform": {"translation": {k: float(v) for k, v in zip("xyz", T_base_cam[:3, 3])},
                           "rotation": {k: float(v) for k, v in zip("xyzw", q)}}}
        path = os.path.join(out_dir, f"dual_arm_{arm}.calib")
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Solve the dual-arm eye-to-hand calibration from one or more recorded sessions. Several "
                    "session directories are merged (e.g. arm_left recorded with one bring-up and arm_right "
                    "with another), provided the camera intrinsics and camera_link->optical TF agree and the "
                    "camera did not move between them.")
    ap.add_argument("session_dirs", nargs="+", metavar="session_dir",
                    help="one or more session directories written by the recorder")
    ap.add_argument("--out", default=None,
                    help="calibration YAML (default <session>/dual_arm_calibration.yaml; required with several sessions)")
    ap.add_argument("--report", default=None, help="full report JSON (default next to the YAML)")
    ap.add_argument("--easy-handeye-dir", default=None, help="also write per-arm easy_handeye2 .calib files here")
    ap.add_argument("--loss", default="soft_l1", choices=["soft_l1", "huber", "cauchy", "linear"])
    ap.add_argument("--f-scale-px", type=float, default=1.0)
    ap.add_argument("--exclude-image", action="append", default=[],
                    help="drop samples whose image name contains this (with several sessions the name is "
                         "<session dir name>/images/cap_NNNN.png)")
    ap.add_argument("--joint-offsets", action="store_true",
                    help="estimate 6 joint zero offsets per arm (regularised; needs robot_description.urdf in the "
                         "sessions; joint 1 is the base-yaw gauge and stays fixed)")
    ap.add_argument("--offset-prior-deg", type=float, default=0.3, help="Gaussian prior sigma of every joint offset")
    ap.add_argument("--refine-intrinsics", action="store_true",
                    help="estimate fx, fy, cx, cy with priors (off by default; the pull is always reported)")
    ap.add_argument("--intrinsics-file", default=None, metavar="YAML",
                    help="intrinsics from calibrate_intrinsics: replaces camera_info (K and distortion) for every "
                         "sample and table image; the gate's 'intrinsics validated' check passes")
    ap.add_argument("--intrinsics-prior-frac", type=float, default=0.01, help="prior sigma of fx, fy (fraction)")
    ap.add_argument("--intrinsics-prior-px", type=float, default=5.0, help="prior sigma of cx, cy (px)")
    ap.add_argument("--touch-sigma-mm", type=float, default=1.0, help="1-sigma of one touch coordinate")
    ap.add_argument("--tcp-file", action="append", default=[], metavar="left=PATH[,right=PATH]",
                    help="re-base the touches of an arm onto a new TCP: PATH is a TCP result YAML (r_tool0_m = new "
                         "igus_tool0 -> xeg32_tool_tip translation in m, arm, verdict).  Every touch of that arm gets "
                         "T_base_tcp' = T_base_tcp inv(X_old) X_new (X_old = link6 -> TCP stored with the touch, else "
                         "of the session URDF).  Without it, touches of one arm recorded with different TCPs are "
                         "refused and the TCP used per arm is printed")
    ap.add_argument("--no-touches", action="store_true", help="ignore touch samples and table-board images")
    ap.add_argument("--corr-warn", type=float, default=0.8, help="report parameter pairs with |correlation| above this")
    ap.add_argument("--base-from", choices=["joint", "touches"], default="joint",
                    help="which arm_right base goes into the YAML: 'joint' = camera-chain fit with nominal FK "
                         "(default), 'touches' = touch-only Kabsch of the common board points (independent of the "
                         "camera; preferred when the gate's joint-vs-touch check fails)")
    g = ap.add_argument_group("trust gate for using arm_right's base in collision avoidance")
    g.add_argument("--gate-min-points", type=int, default=4, help="fit points (board crosshairs + fit stickers) common to both arms")
    g.add_argument("--gate-touch-rms-mm", type=float, default=1.5,
                   help="max touch mismatch RMS (nominal FK): fit residual, out-of-fit loop closure, held-out touches")
    g.add_argument("--gate-joint-touch-mm", type=float, default=1.0,
                   help="FLOOR (mm) of the written vs touch-only base agreement; the limit is max(floor, 3 sigma)")
    g.add_argument("--gate-joint-touch-deg", type=float, default=0.1,
                   help="FLOOR (deg, per axis) of the rotation agreement; the limit is max(floor, 3 sigma)")
    g.add_argument("--gate-agree-sigma-k", type=float, default=3.0, help="k of the agreement limit max(floor, k sigma)")
    g.add_argument("--gate-heldout-px", type=float, default=1.0, help="max held-out reprojection RMS (px)")
    g.add_argument("--gate-min-check-points", type=int, default=2, help="check points touched by both arms")
    g.add_argument("--gate-check-error-mm", type=float, default=3.0, help="max check-point error (TRE, held-out)")
    g.add_argument("--gate-check-height-mm", type=float, default=30.0, help="one check point at least this far off the board plane")
    g.add_argument("--gate-check-dist-mm", type=float, default=150.0, help="one check point at least this far from the board centre")
    g.add_argument("--gate-yaw-sigma-deg", type=float, default=0.15, help="max predicted 1-sigma yaw of the touch-only base")
    g.add_argument("--gate-reach-sigma-mm", type=float, default=1.0, help="max predicted 1-sigma of the touch-only reach point")
    g.add_argument("--gate-intrinsics-f-pct", type=float, default=0.3, help="max intrinsics pull in f (%%) without an intrinsics file")
    g.add_argument("--gate-intrinsics-c-px", type=float, default=3.0, help="max intrinsics pull in c (px) without an intrinsics file")
    ap.add_argument("--fit-point-ids", default="", metavar="IDS",
                    help="comma-separated touch ids to treat as fit stickers (overrides the recorded kind)")
    ap.add_argument("--check-point-ids", default="", metavar="IDS",
                    help="comma-separated touch ids to treat as held-out check points (overrides the recorded kind; "
                         "never used in any fit)")
    ap.add_argument("--uncertainty-mc", type=int, default=0, metavar="N",
                    help="Monte Carlo uncertainty budget with N re-solves of the written solution (0 = off; about "
                         "2 s per draw with 70 samples; N >= 200 for a report value)")
    ap.add_argument("--mc-seed", type=int, default=20261008)
    ap.add_argument("--compare", default=None, metavar="YAML",
                    help="previous calibration YAML: report the change of arm_right_base and camera_link "
                         "(reproducibility; guidance: accept < 2 mm / 0.2 deg)")
    ap.add_argument("--keep-touch-margin", type=float, default=0.10,
                    help="joint offsets are recommended only if the held-out touch mismatch is this fraction better")
    val = ap.add_mutually_exclusive_group()
    val.add_argument("--kfold", type=int, default=5, metavar="K", help="held-out evaluation: deterministic K-fold (5)")
    val.add_argument("--holdout", type=float, default=None, metavar="FRAC",
                     help="held-out evaluation: a single split with this fraction held out (e.g. 0.25)")
    ap.add_argument("--validate", dest="validate", action="store_true", default=True,
                    help="run the held-out evaluation (default; the trust gate requires it)")
    ap.add_argument("--no-validate", dest="validate", action="store_false", help="skip the held-out evaluation")
    args = ap.parse_args(argv)

    if len(args.session_dirs) > 1 and not args.out:
        print("ERROR: --out is required when several session directories are given "
              "(it would be ambiguous which session directory should receive the result)", file=sys.stderr)
        return 2
    try:
        data = load_sessions_full(args.session_dirs)
        samples, metas, per_session = data.samples, data.metas, data.per_session
    except SessionMismatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"ERROR: cannot read session: {e}", file=sys.stderr)
        return 2
    for d, (label, ss) in zip(args.session_dirs, per_session):
        print(f"Session {d}: {len(ss)} samples "
              f"(left {sum(s.arm == 'left' for s in ss)}, right {sum(s.arm == 'right' for s in ss)})")
    targets = [m.get("targets") for m in metas if m.get("targets") is not None]
    if any(t != targets[0] for t in targets[1:]):
        print("WARNING: the sessions were recorded with different target definitions (session.json 'targets'); "
              "this is only valid if each arm's target stayed mounted in the same place on the arm")
    if args.exclude_image:
        samples = [s for s in samples if not any(x in s.image for x in args.exclude_image)]
    print(f"Loaded {len(samples)} samples in total "
          f"(left {sum(s.arm == 'left' for s in samples)}, right {sum(s.arm == 'right' for s in samples)})")
    tables, touches = ([], []) if args.no_touches else (data.tables, data.touches)
    try:
        for ids_s, kind in ((args.fit_point_ids, "fit"), (args.check_point_ids, "check")):
            ids = {int(v) for v in ids_s.split(",") if v.strip()}
            for t in touches:
                if t.point_id in ids:
                    t.kind = kind
    except ValueError:
        print("ERROR: --fit-point-ids / --check-point-ids need comma-separated integers", file=sys.stderr)
        return 2
    if args.exclude_image:
        tables = [t for t in tables if not any(x in t.image for x in args.exclude_image)]
    intrinsics_src = "camera_info"
    if args.intrinsics_file:
        from .intrinsics import apply_intrinsics, load_intrinsics_file
        try:
            K_f, D_f, info_f = load_intrinsics_file(args.intrinsics_file)
            n_over = apply_intrinsics(list(samples) + list(tables), K_f, D_f, info_f.get("image_size"))
        except (OSError, ValueError, KeyError) as e:
            print(f"ERROR: --intrinsics-file: {e}", file=sys.stderr)
            return 2
        intrinsics_src = "file"
        print(f"Intrinsics from {args.intrinsics_file} (fx {K_f[0, 0]:.2f} fy {K_f[1, 1]:.2f} cx {K_f[0, 2]:.2f} "
              f"cy {K_f[1, 2]:.2f}) override camera_info in {n_over} samples / table images")
    kinds = {k: [sum(t.arm == a and t.kind == k for t in touches) for a in ARMS] for k in ("board", "fit", "check")}
    print(f"Table-board images {len(tables)}, touches left {sum(t.arm == 'left' for t in touches)} "
          f"right {sum(t.arm == 'right' for t in touches)} (board L{kinds['board'][0]}/R{kinds['board'][1]}, fit stickers "
          f"L{kinds['fit'][0]}/R{kinds['fit'][1]}, check stickers L{kinds['check'][0]}/R{kinds['check'][1]})")
    for n in data.kin_notes:
        print(f"NOTE: {n}")
    try:
        tcp_specs = parse_tcp_file_args(args.tcp_file)
        tcp_files = {a: load_tcp_file(p_, a) for a, p_ in tcp_specs.items()}
        if args.no_touches and tcp_files:
            print("NOTE: --no-touches: --tcp-file has nothing to re-base")
        tcp_report, kin_used = prepare_touch_tcp(touches, data.kin, tcp_files)
    except (SolverError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for line in format_tcp_report(tcp_report):
        print(line)
    opts = SolverOptions(loss=args.loss, f_scale_px=args.f_scale_px, joint_offsets=args.joint_offsets,
                         offset_prior_deg=args.offset_prior_deg, refine_intrinsics=args.refine_intrinsics,
                         intrinsics_prior_frac=args.intrinsics_prior_frac, intrinsics_prior_px=args.intrinsics_prior_px,
                         touch_sigma_m=args.touch_sigma_mm * 1e-3, corr_warn=args.corr_warn,
                         keep_touch_margin=args.keep_touch_margin, intrinsics_source=intrinsics_src,
                         intrinsics_file=args.intrinsics_file or "")
    gate_opts = GateOptions(min_common_points=args.gate_min_points, max_touch_rms_mm=args.gate_touch_rms_mm,
                            max_joint_vs_touch_mm=args.gate_joint_touch_mm,
                            max_joint_vs_touch_deg=args.gate_joint_touch_deg, agree_sigma_k=args.gate_agree_sigma_k,
                            max_heldout_px=args.gate_heldout_px, min_check_points=args.gate_min_check_points,
                            max_check_error_mm=args.gate_check_error_mm, check_min_height_mm=args.gate_check_height_mm,
                            check_min_dist_mm=args.gate_check_dist_mm, max_yaw_sigma_deg=args.gate_yaw_sigma_deg,
                            max_reach_sigma_mm=args.gate_reach_sigma_mm,
                            max_intrinsics_f_frac=args.gate_intrinsics_f_pct / 100.0,
                            max_intrinsics_c_px=args.gate_intrinsics_c_px)
    dataset = ", ".join(os.path.abspath(d) for d in args.session_dirs)
    compare_prev = None
    if args.compare:
        from .metrology import load_calibration_yaml
        try:
            compare_prev = load_calibration_yaml(args.compare)
        except (OSError, KeyError, ValueError, TypeError) as e:
            print(f"ERROR: --compare: {e}", file=sys.stderr)
            return 2
    try:
        res = solve(samples, opts, dataset=dataset, tables=tables, touches=touches, kin=kin_used,
                    gate_options=gate_opts)
        deployed_poses(res, args.base_from)       # fails early if --base-from touches has no estimate
        if args.validate:
            from .validation import cross_validate
            res.report["validation"] = cross_validate(res, opts, touches, kin_used, k=args.kfold,
                                                       holdout=args.holdout)
        which, why = choose_written_fit(res, res.report.get("validation"))
        set_written_base(res, which, why)
        if args.uncertainty_mc > 0:
            from .uncertainty import run_mc
            res.report["uncertainty_mc"] = run_mc(
                res, args.uncertainty_mc, intrinsics_file=intrinsics_src == "file", seed=args.mc_seed,
                progress=lambda i, n, t: print(f"  Monte Carlo {i}/{n} ({t:.0f} s)", file=sys.stderr, flush=True)
                if i % 10 == 0 or i == n else None)
    except (SolverError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    res.report["tcp"] = tcp_report
    res.report["session_info"] = [(lab, (m or {}).get("session_info") or {}) for lab, m in
                                  zip([l for l, _ in per_session], metas)]
    if compare_prev is not None:
        from .metrology import compare_calibrations
        T_R_w, T_C_w = deployed_poses(res, args.base_from)
        res.report["compare"] = compare_calibrations(T_R_w, T_C_w, compare_prev)
    res.report["trust_gate"] = evaluate_gate(res.report, gate_opts, args.base_from)
    res.report["base_from"] = args.base_from
    print(format_report(res.report))
    gate = res.report["trust_gate"]
    out = args.out or os.path.join(args.session_dirs[0], "dual_arm_calibration.yaml")
    rep_path = args.report or os.path.splitext(out)[0] + "_report.json"
    write_calibration_yaml(out, res, build_notes(res.report, args.session_dirs, args.base_from),
                           source=gate["source"], base_from=args.base_from)
    with open(rep_path, "w") as f:
        json.dump(res.report, f, indent=2, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
    print(f"\nWrote {out} (source: {gate['source']})\nWrote {rep_path}")
    if args.easy_handeye_dir:
        for p in write_easy_handeye_calibs(args.easy_handeye_dir, res, dataset, args.base_from):
            print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
