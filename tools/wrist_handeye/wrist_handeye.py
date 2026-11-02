#!/usr/bin/env python3
"""Standalone eye-in-hand calibration of the arm-mounted Gemini 305 (link6 -> wrist_camera_color_optical_frame).

  ROS_DOMAIN_ID=<n> ROS_LOCALHOST_ONLY=1 python3 wrist_handeye.py [--plan-only | --execute] [options]
  python3 wrist_handeye.py solve <run_dir> [--publish-tf]          (offline, no ROS needed)

See README.md.  Never edits any URDF; writes only below the output directory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/farhan/igus_rebel_ros2/src/rebel_dual_arm_calibration")  # read-only: targets.py

import handeye_math as hm  # noqa: E402

DEFAULT_TARGETS = "/home/farhan/dual_arm_calibration/targets_measured.yaml"
CHILD_FRAME = "wrist_camera_color_optical_frame"
SCENE_NAMES = ["whe_table_slab", "whe_board_stand", "whe_pole_pad", "whe_head_pad"]
DEFAULT_K = np.array([[626.0, 0, 640.0], [0, 626.0, 400.0], [0, 0, 1.0]])  # mock only (no camera_info there)


def info(msg=""):
    print(msg, flush=True)


# ================================================================================================
# solve (offline)
# ================================================================================================
def _arr(x):
    return np.array(x, dtype=float)


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "samples.json")) as f:
        sj = json.load(f)
    with open(os.path.join(run_dir, "camera_info.json")) as f:
        cam = json.load(f)
    board = {}
    p = os.path.join(run_dir, "board.json")
    if os.path.exists(p):
        with open(p) as f:
            board = json.load(f)
    samples, meta = [], []
    for s in sj["samples"]:
        if not s.get("obj"):
            continue
        samples.append(hm.Sample(_arr(s["T_base_link6"]), _arr(s["obj"]).reshape(-1, 3), _arr(s["pix"]).reshape(-1, 2), s.get("image", "")))
        meta.append(s)
    return samples, meta, cam, board


def fmt_report(res: dict, meta: list) -> str:
    X = res["X"]
    xyz, rpy = hm.T_to_xyz_rpy(X)
    _, q = hm.T_to_xyz_quat(X)
    L = []
    L.append("=== wrist hand-eye solve: %d samples, init %s ===" % (res["n_samples"], res["init_method"]))
    for k, v in res["init"].items():
        L.append("  init %-10s board consistency %.2f mm / %.2f deg" % (k, v["pos_rms_mm"], v["rot_rms_deg"]))
    L.append("X = T_link6_optical: xyz [%.4f %.4f %.4f] m, rpy [%.4f %.4f %.4f] rad, quat xyzw [%.5f %.5f %.5f %.5f]" % (*xyz, *rpy, *q))
    L.append("reprojection: RMS %.3f px, max %.2f px" % (res["rms_px"], res["max_px"]))
    c = res["board_consistency"]
    L.append("board pose consistency across samples (PnP+X vs joint estimate): pos RMS %.2f mm (max %.2f), rot RMS %.3f deg (max %.3f)"
             % (c["pos_rms_mm"], c["pos_max_mm"], c["rot_rms_deg"], c["rot_max_deg"]))
    loo = res.get("loo")
    if loo:
        L.append("leave-one-out: board pos RMS %.2f mm (max %.2f), rot RMS %.3f deg, reproj RMS %.2f px" %
                 (loo["pos_rms_mm"], loo["pos_max_mm"], loo["rot_rms_deg"], loo["reproj_rms_px"]))
    d = res["rotation"]
    L.append("link6 rotation span %.1f deg; axis-diversity eigenvalues %s" % (d["span_deg"], ", ".join("%.3f" % e for e in d["axis_eig"])))
    if "scene_cross_check" in res:
        L.append("T_base_board vs scene-camera estimate: %.1f mm, %.2f deg" % (res["scene_cross_check"]["mm"], res["scene_cross_check"]["deg"]))
    if res.get("method_agreement"):
        L.append("init methods agree within %.2f mm / %.2f deg" % (res["method_agreement"]["max_mm"], res["method_agreement"]["max_deg"]))
    for w in res.get("warnings", []):
        L.append("WARNING: " + w)
    for i, r in enumerate(res["per_sample_rms_px"]):
        L.append("  sample %2d (pose %s): RMS %.2f px" % (i, meta[i].get("pose_index", "?") if i < len(meta) else "?", r))
    L.append("GATE: %s%s" % (res["status"], "" if not res["reasons"] else "  - " + "; ".join(res["reasons"])))
    return "\n".join(L)


def write_result(run_dir: str, res: dict, meta: list, cam: dict, board: dict, extra: dict = None):
    import yaml

    X = res["X"]
    xyz, rpy = hm.T_to_xyz_rpy(X)
    _, quat = hm.T_to_xyz_quat(X)
    f = lambda v: float(round(v, 6))  # noqa: E731
    out = {
        "schema": "wrist_handeye/v1",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "parent": "link6", "child": CHILD_FRAME,
        "camera_frame_id": cam.get("frame_id", ""),
        "xyz": [f(v) for v in xyz], "rpy": [f(v) for v in rpy], "quat_xyzw": [f(v) for v in quat],
        "status": res["status"], "reasons": list(res["reasons"]),
        "stats": {
            "n_samples": res["n_samples"], "init_method": res["init_method"],
            "rms_px": f(res["rms_px"]), "max_px": f(res["max_px"]),
            "board_consistency": {k: f(v) for k, v in res["board_consistency"].items()},
            "loo": None if not res.get("loo") else {k: (f(v) if not isinstance(v, list) else [f(x) for x in v]) for k, v in res["loo"].items()},
            "rotation": {"span_deg": f(res["rotation"]["span_deg"]), "axis_eig": [f(e) for e in res["rotation"]["axis_eig"]]},
            "scene_cross_check": None if "scene_cross_check" not in res else {k: f(v) for k, v in res["scene_cross_check"].items()},
            "T_base_board": [[f(v) for v in row] for row in res["T_base_board"]],
        },
        "samples": [{"pose_index": m.get("pose_index"), "stamp_ns": m.get("stamp_ns"), "image": m.get("image"),
                     "n_points": m.get("n_points"), "rms_px": f(res["per_sample_rms_px"][i])} for i, m in enumerate(meta)],
    }
    if extra:
        out.update(extra)
    with open(os.path.join(run_dir, "result.yaml"), "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    with open(os.path.join(run_dir, "urdf_snippet.xml"), "w") as fh:
        fh.write("<!-- wrist_handeye/v1 %s: RMS %.2f px, %d samples. PASTE BY HAND; this tool never edits URDF files. -->\n"
                 % (res["status"], res["rms_px"], res["n_samples"]))
        fh.write('<link name="%s"/>\n<joint name="link6_to_%s" type="fixed">\n  <parent link="link6"/>\n  <child link="%s"/>\n'
                 '  <origin xyz="%.6f %.6f %.6f" rpy="%.6f %.6f %.6f"/>\n</joint>\n' % (CHILD_FRAME, CHILD_FRAME, CHILD_FRAME, *xyz, *rpy))


def solve_run_dir(run_dir: str, do_loo=True) -> dict:
    samples, meta, cam, board = load_run(run_dir)
    if len(samples) < 3:
        raise SystemExit("only %d usable samples in %s" % (len(samples), run_dir))
    Tscene = _arr(board["T_base_board_scene"]) if board.get("T_base_board_scene") else None
    res = hm.solve_handeye(samples, _arr(cam["K"]), _arr(cam["D"]), do_loo=do_loo, T_base_board_scene=Tscene)
    rep = fmt_report(res, meta)
    info(rep)
    with open(os.path.join(run_dir, "solve_report.txt"), "w") as fh:
        fh.write(rep + "\n")
    write_result(run_dir, res, meta, cam, board)
    info("wrote %s/{result.yaml,urdf_snippet.xml,solve_report.txt}" % run_dir)
    return res


def solve_main(argv):
    ap = argparse.ArgumentParser(prog="wrist_handeye.py solve")
    ap.add_argument("run_dir")
    ap.add_argument("--publish-tf", action="store_true", help="publish the result as a static TF until Ctrl-C (needs ROS_DOMAIN_ID)")
    a = ap.parse_args(argv)
    res = solve_run_dir(a.run_dir)
    if a.publish_tf:
        require_domain()
        import ros_io
        ros_io.publish_static_tf("link6", CHILD_FRAME, res["X"])
    return 0 if res["status"] != "FAIL" else 1


def require_domain():
    d = os.environ.get("ROS_DOMAIN_ID", "")
    if not d.isdigit():
        raise SystemExit("ROS_DOMAIN_ID must be set explicitly (e.g. ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 ...). Refusing.")
    return int(d)


# ================================================================================================
# helpers for the run
# ================================================================================================
def detect(det, img, min_points=1):
    d = det.detect(img).get("table")
    if d is None:
        return None
    ids, obj, pix = d.arrays()
    if len(ids) < min_points:
        return None
    return {"ids": ids, "obj": obj, "pix": pix, "n_markers": d.n_markers, "n_chess": d.n_chess}


def KD(ci):
    return np.array(ci.k, dtype=float).reshape(3, 3), np.array(ci.d, dtype=float).ravel()


def listT(T):
    return [[float(v) for v in r] for r in T]


def fmt_T(T):
    xyz, q = hm.T_to_xyz_quat(T)
    return "xyz [%.3f %.3f %.3f] quat [%.3f %.3f %.3f %.3f]" % (*xyz, *q)


def locate_board_scene(io, spec, det, n=5):
    ci = io.wait_camera_info("scene", 5.0)
    if ci is None:
        raise RuntimeError("no /camera/color/camera_info (scene camera); give --board-pose on mock only")
    K, D = KD(ci)
    frames = io.grab_frames("scene", n, 0, timeout=10.0)
    Ts = []
    T_base_cam, exact = io.lookup("base_link", ci.header.frame_id)
    for _, img in frames:
        d = detect(det, img, 16)
        if d is None:
            continue
        T_cb, rms = hm.solve_pnp(d["obj"], d["pix"], K, D)
        if rms < 2.0:
            Ts.append(T_base_cam @ T_cb)
    if not Ts:
        raise RuntimeError("table board not detected in the scene camera image")
    T = hm.mean_pose(Ts)
    return T, hm.consistency(Ts, T), len(Ts)


def board_report(T, w, h):
    c = hm.transform_points(T, hm.board_corners_obj(w, h))
    n = hm.board_up_normal(T)
    info("board: centre (frame origin) %s ; up-normal [%.3f %.3f %.3f] ; tilt from horizontal %.2f deg ; centre %.3f m from the base axis ; corner z %.3f..%.3f"
         % (fmt_T(T), *n, math.degrees(math.acos(min(1.0, abs(n[2])))), math.hypot(*T[:2, 3]), c[:, 2].min(), c[:, 2].max()))
    return c


def box_from_object(co, name):
    if co is None:
        raise RuntimeError("planning-scene object '%s' is missing (run move_group with run_camera_pole:=true) - failing closed" % name)
    if co.header.frame_id not in ("base_link", "world"):
        raise RuntimeError("%s is in frame '%s', expected base_link/world - failing closed" % (name, co.header.frame_id))
    from geometry_msgs.msg import Pose  # noqa: F401
    from shape_msgs.msg import SolidPrimitive
    if len(co.primitives) != 1 or co.primitives[0].type != SolidPrimitive.BOX:
        raise RuntimeError("%s is not a single box primitive - failing closed" % name)
    po, pp = co.pose, co.primitive_poses[0]
    To = hm.T_from_xyz_quat(po.position.x, po.position.y, po.position.z, po.orientation.x, po.orientation.y, po.orientation.z, po.orientation.w)
    Tp = hm.T_from_xyz_quat(pp.position.x, pp.position.y, pp.position.z, pp.orientation.x, pp.orientation.y, pp.orientation.z, pp.orientation.w)
    T = To @ Tp
    xyz, q = hm.T_to_xyz_quat(T)
    return {"center": xyz, "size": list(co.primitives[0].dimensions), "quat": q}


def apply_safety_scene(io, corners, args):
    objs = io.world_objects()
    stale = [n for n in objs if n.startswith("whe_")]
    if stale:
        info("removing stale objects from an earlier run: %s" % stale)
        io.remove_objects(stale)
    pole = box_from_object(objs.get("camera_pole"), "camera_pole")
    head = box_from_object(objs.get("camera_head"), "camera_head")
    slab = hm.build_slab(corners, top_offset=0.05, side=1.2, min_axis_dist=0.15, thickness=args.slab_thickness)
    stand = hm.build_board_stand(corners, slab["top"], margin=0.05, extra=args.stand_extra)
    boxes = {"whe_table_slab": slab, "whe_board_stand": stand}
    for name, src in (("whe_pole_pad", pole), ("whe_head_pad", head)):
        b = hm.inflate_box(src["center"], src["size"], 0.05)
        b["quat"] = src["quat"]
        boxes[name] = b
    io.add_boxes(boxes)
    present = io.whe_objects_present()
    if sorted(present) != sorted(boxes):
        raise RuntimeError("safety scene not applied completely: %s" % present)
    for n, b in boxes.items():
        info("  scene %-16s centre [%.3f %.3f %.3f] size [%.3f %.3f %.3f]" % (n, *b["center"], *b["size"]))
    return boxes


def mount_from_args(args) -> np.ndarray:
    if args.mount_yaml:
        import yaml
        with open(args.mount_yaml) as f:
            y = yaml.safe_load(f)
        if "quat_xyzw" in y:
            return hm.T_from_xyz_quat(*y["xyz"], *y["quat_xyzw"])
        return hm.T_from_xyz_rpy(*y["xyz"], *y["rpy"])
    if args.mount_xyz:
        return hm.T_from_xyz_rpy(*args.mount_xyz, *(args.mount_rpy or [0, 0, 0]))
    return hm.nominal_T_link6_optical()


def check_plan(q_from, q_goal, pos, q_run_start, collision_free) -> List[str]:
    """Safety checks shared by plan-only and execute."""
    prob = []
    if np.max(np.abs(pos[0] - q_from)) > 0.01:
        prob.append("first trajectory point differs from the start by %.4f rad (> 0.01)" % np.max(np.abs(pos[0] - q_from)))
    if np.max(np.abs(pos[-1] - q_goal)) > 0.005:
        prob.append("trajectory does not end at the goal (%.4f rad)" % np.max(np.abs(pos[-1] - q_goal)))
    direct = np.abs(q_goal - q_from)
    travel = np.sum(np.abs(np.diff(pos, axis=0)), axis=0)
    bad = travel > np.maximum(1.5 * direct, direct + 0.35)
    if np.any(bad):
        prob.append("detour: joint travel %s rad vs direct %s rad" % (np.round(travel, 2).tolist(), np.round(direct, 2).tolist()))
    if np.max(np.abs(pos[:, 5] - q_run_start[5])) > 3.14:
        prob.append("joint6 leaves +-3.14 rad of its start value")
    if not collision_free:
        prob.append("waypoint reported in collision by /check_state_validity")
    return prob


# ================================================================================================
# the run
# ================================================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--plan-only", action="store_true", help="(default) generate + plan every move, NO motion")
    m.add_argument("--execute", action="store_true", help="move the arm through the poses and capture")
    ap.add_argument("--real", action="store_true", help="required to move REAL hardware (plus typed MOVE); refused on mock hardware")
    ap.add_argument("--auto", action="store_true", help="after the first move continue without Enter prompts")
    ap.add_argument("--mock-no-capture", action="store_true", help="mock only: skip image capture/detection (refused on real hardware)")
    ap.add_argument("--board-pose", type=float, nargs=7, metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                    help="board centre pose in base_link; mock hardware only")
    ap.add_argument("--out", help="output dir (default scan_results/wrist_handeye_<timestamp>)")
    ap.add_argument("--group", default="igus_rebel_arm")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--n-poses", type=int, default=18)
    ap.add_argument("--scale", type=float, default=0.10, help="velocity+acceleration scaling (max 0.25)")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to settle after standstill before grabbing frames")
    ap.add_argument("--min-points", type=int, default=16)
    ap.add_argument("--max-joint-step-deg", type=float, default=90.0)
    ap.add_argument("--slab-thickness", type=float, default=0.30)
    ap.add_argument("--stand-extra", type=float, default=0.03, help="board-stand height above the slab top")
    ap.add_argument("--mount-xyz", type=float, nargs=3, help="override nominal T_link6_optical translation (m)")
    ap.add_argument("--mount-rpy", type=float, nargs=3, help="override nominal T_link6_optical rpy (rad, URDF convention)")
    ap.add_argument("--mount-yaml", help="override nominal mount from a YAML with xyz + (rpy | quat_xyzw), e.g. an earlier result.yaml")
    ap.add_argument("--no-seed-from-view", action="store_true", help="do not refine the nominal mount from the board seen at start")
    ap.add_argument("--publish-tf", action="store_true", help="after solving publish the result as a static TF until Ctrl-C")
    return ap


def run_main(argv):
    args = build_parser().parse_args(argv)
    args.execute = bool(args.execute)
    if not 0.01 <= args.scale <= 0.25:
        raise SystemExit("--scale must be within 0.01..0.25")
    require_domain()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or "/home/farhan/igus_rebel_ros2/scan_results/wrist_handeye_%s" % stamp
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    info("domain %s, localhost_only=%s, mode %s, out %s" % (os.environ["ROS_DOMAIN_ID"], os.environ.get("ROS_LOCALHOST_ONLY"),
                                                          "EXECUTE" if args.execute else "plan-only", out))

    import ros_io
    from ros_io import UserStop
    from rebel_dual_arm_calibration import targets

    spec = targets.load_table_board(args.targets)
    if spec is None:
        raise SystemExit("no table_board in %s" % args.targets)
    det = targets.TargetDetector({"table": spec})
    bw, bh = spec.board_size()

    io = ros_io.IO(args.group)
    signal.signal(signal.SIGINT, lambda s, f: io.request_stop("SIGINT"))
    signal.signal(signal.SIGTERM, lambda s, f: io.request_stop("SIGTERM"))
    rc = 1
    scene_touched = False
    run_info = {"args": vars(args), "start": stamp}
    samples_rec: List[dict] = []
    try:
        # ------------------------------------------------------------------ 1. startup checks
        comps = io.hardware_components()
        for n, c, st in comps:
            info("hardware component %s: %s (%s)" % (n, c, st))
        if not comps:
            raise SystemExit("no hardware component reported by controller_manager - refusing")
        is_mock = all(c == "mock_components/GenericSystem" for _, c, _ in comps)
        run_info["hardware"] = [list(c) for c in comps]
        run_info["is_mock"] = is_mock
        if args.real and is_mock:
            info("REFUSED: --real given but the hardware is mock_components/GenericSystem")
            return 2
        if not is_mock:
            if args.mock_no_capture or args.board_pose:
                info("REFUSED: --mock-no-capture / --board-pose are mock-only, the hardware is REAL")
                return 2
            if args.execute and not args.real:
                info("REFUSED: the hardware is REAL; moving it needs --real and the typed confirmation")
                return 2
            if args.execute:
                if not sys.stdin.isatty():
                    info("REFUSED: real motion needs an interactive terminal for the typed confirmation")
                    return 2
                info("*** REAL HARDWARE: the arm WILL MOVE through ~%d poses near the table, scene camera pole and base. ***" % args.n_poses)
                if io.prompt('Type MOVE to continue: ').strip() != "MOVE":
                    info("not confirmed - nothing done")
                    return 2
        elif args.execute and not args.mock_no_capture:
            info("note: mock hardware has no cameras; captures will fail and stop the run after 3 poses. Use --mock-no-capture for a motion-only test.")

        # ------------------------------------------------------------------ 2. board
        if args.board_pose:
            T_bb = hm.T_from_xyz_quat(*args.board_pose)
            T_scene = None
            run_info["board_source"] = "--board-pose"
            info("board pose OVERRIDE (mock): %s" % fmt_T(T_bb))
        else:
            T_bb, spread, nb = locate_board_scene(io, spec, det)
            T_scene = T_bb
            run_info["board_source"] = "scene camera (%d frames), spread %s" % (nb, spread)
            info("board from the scene camera: %d frames, spread %.2f mm / %.3f deg" % (nb, spread["pos_rms_mm"], spread["rot_rms_deg"]))
        corners = board_report(T_bb, bw, bh)
        with open(os.path.join(out, "board.json"), "w") as f:
            json.dump({"w": bw, "h": bh, "T_base_board": listT(T_bb), "T_base_board_scene": None if T_scene is None else listT(T_scene),
                       "source": run_info["board_source"]}, f, indent=1)

        # wrist camera model
        ci = io.wait_camera_info("wrist", 3.0)
        if ci is None:
            if not is_mock:
                raise RuntimeError("no /arm_left/wrist_camera/color/camera_info")
            info("WARN: no wrist camera_info (mock): assuming f=626 px, 1280x800, no distortion")
            K, D, size, frame_id, model = DEFAULT_K, np.zeros(5), (1280, 800), CHILD_FRAME, "plumb_bob"
        else:
            K, D = KD(ci)
            size, frame_id, model = (ci.width, ci.height), ci.header.frame_id, ci.distortion_model
        with open(os.path.join(out, "camera_info.json"), "w") as f:
            json.dump({"K": K.tolist(), "D": D.tolist(), "width": size[0], "height": size[1], "frame_id": frame_id, "distortion_model": model}, f, indent=1)

        # ------------------------------------------------------------------ 3. safety scene
        scene_touched = True
        info("applying the safety scene (all objects whe_*, removed at exit)")
        boxes = apply_safety_scene(io, corners, args)
        run_info["scene"] = boxes

        # ------------------------------------------------------------------ 4. poses
        try:
            io.wait_standstill(1.0, 0.002, timeout=5.0)
        except TimeoutError:
            raise RuntimeError("the arm is still moving at start - wait until it is at rest")
        q_start = io.wait_joint_states()
        info("start joints (deg): %s" % np.round(np.degrees(q_start), 1).tolist())
        valid, contacts = io.state_validity(q_start)
        if not valid:
            raise RuntimeError("the CURRENT arm state collides with the safety scene / planning scene: %s. Move the arm to a clear pose "
                               "near the board first (jog), then re-run." % contacts)
        X_nom = mount_from_args(args)
        X_est = X_nom.copy()
        info("nominal T_link6_optical: %s  (rpy %s)" % (fmt_T(X_nom), np.round(hm.T_to_xyz_rpy(X_nom)[1], 3).tolist()))
        seen_now = False
        im = io.latest_image("wrist", 0, timeout=1.0) if not is_mock or ci is not None else None
        if im is not None:
            d0 = detect(det, im[1], args.min_points)
            if d0 is not None:
                seen_now = True
                info("board is visible in the current wrist image (%d points): the current pose is the seed" % len(d0["ids"]))
                if not args.no_seed_from_view:
                    try:
                        T_cb, rms = hm.solve_pnp(d0["obj"], d0["pix"], K, D)
                        T6, _ = io.lookup("base_link", "link6", im[0])
                        X_view = hm.inv_T(T6) @ T_bb @ hm.inv_T(T_cb)
                        dmm, ddg = hm.pose_diff(X_view, X_nom)
                        info("mount estimated from this one view: %s (differs from nominal by %.0f mm / %.1f deg, PnP %.2f px)"
                             % (fmt_T(X_view), dmm, ddg, rms))
                        if dmm < 150.0:
                            X_est = X_view
                        else:
                            info("  >150 mm from nominal: not used")
                    except Exception as e:  # noqa: BLE001
                        info("  could not use the view for the mount: %s" % e)
        run_info["X_nominal"] = listT(X_nom)
        run_info["X_used_for_plan"] = listT(X_est)

        def ik(T6, seed):
            return io.compute_ik(T6, seed, avoid_collisions=True)

        centre = T_bb[:3, 3]
        up = hm.board_up_normal(T_bb)
        forced_first, seed_q = None, q_start
        if not seen_now:
            Tc = hm.look_at_pose(T_bb, centre + 0.35 * up, centre, 0.0)
            T6 = Tc @ hm.inv_T(X_est)
            q = ik(T6, q_start)
            if q is not None and abs(q[5] - q_start[5]) <= 3.14:
                forced_first = hm.Candidate(Tc, 0.0, 0.0, 0.0, 0.35, "seed", q=q, T_base_link6=T6)
                seed_q = q
                info("seed pose: camera 0.35 m above the board centre, IK ok")
            else:
                info("WARN: seed pose (0.35 m above the board) has no IK solution; continuing with the orbit set")

        cands = hm.orbit_candidates(T_bb)
        reasons: Counter = Counter()
        pool = []
        for i, c in enumerate(cands):
            io.check_stop()
            if c.dist < 0.25 - 1e-9:
                reasons["camera-board distance < 0.25 m"] += 1
                continue
            if not hm.corners_in_view(c.T_base_cam, T_bb, bw, bh, K, D, size, 0.10):
                reasons["board corners outside image (10 % margin)"] += 1
                continue
            T6 = c.T_base_cam @ hm.inv_T(X_est)
            q = ik(T6, seed_q)
            if q is None:
                reasons["no collision-free IK"] += 1
                continue
            if abs(q[5] - q_start[5]) > 3.14:
                reasons["joint6 travel > 3.14 rad from start"] += 1
                continue
            c.q, c.T_base_link6 = q, T6
            pool.append(c)
            if (i + 1) % 75 == 0:
                info("  IK %d/%d candidates, %d usable" % (i + 1, len(cands), len(pool)))
        info("candidates: %d generated, %d usable after view+IK filters" % (len(cands), len(pool)))
        seq, r2 = hm.plan_sequence(pool, q_start, ik, X_est, args.n_poses, max_step_deg=args.max_joint_step_deg,
                                   forced_first=forced_first, stop=io.check_stop)
        reasons.update(r2)
        reasons["usable but not selected (diversity)"] += max(0, len(pool) - len(seq) - sum(r2.values()))
        info("")
        info("pose generation: %d candidates -> %d poses selected" % (len(cands), len(seq)))
        for k, v in reasons.most_common():
            info("  rejected/unused %4d  %s" % (v, k))
        if not seq:
            raise RuntimeError("no reachable pose sequence (if the first step is the problem, jog the arm to a pose near the board first)")
        run_info["n_candidates"] = len(cands)
        run_info["rejections"] = dict(reasons)
        run_info["poses"] = [{"index": i, "tilt": c.tilt, "azimuth": c.azimuth, "roll": c.roll, "dist": c.dist, "tag": c.tag,
                              "T_base_cam": listT(c.T_base_cam), "q": c.q.tolist()} for i, c in enumerate(seq)]

        # ------------------------------------------------------------------ 5. plan-only
        if not args.execute:
            rc = plan_only(io, args, seq, q_start, out, run_info)
            return rc

        # ------------------------------------------------------------------ 6. execute
        rc = execute_all(io, args, det, seq, q_start, X_est, K, D, size, T_bb, bw, bh, out, run_info, samples_rec, is_mock, ci)
        return rc
    except UserStop as e:
        info("\nSTOPPED: %s. Active goal cancelled; the arm holds where it is." % e)
        rc = 130
        return rc
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        info("ERROR: %s" % e)
        rc = 1
        return rc
    finally:
        io.cleaning = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        io.cancel_active()
        io.keys.restore()
        left = None
        if scene_touched:
            for attempt in range(3):
                try:
                    present = io.whe_objects_present()
                    if present:
                        io.remove_objects(present)
                    left = io.whe_objects_present()
                    if not left:
                        break
                except Exception as e:  # noqa: BLE001
                    info("scene cleanup attempt %d failed: %s" % (attempt + 1, e))
                    time.sleep(1.0)
            info("safety scene cleanup: %s" % ("all whe_* objects removed" if left == [] else "FAILED, remaining: %s - remove them by hand!" % left))
        run_info["end"] = datetime.datetime.now().isoformat(timespec="seconds")
        run_info["return_code"] = rc
        try:
            with open(os.path.join(out, "run_info.json"), "w") as f:
                json.dump(run_info, f, indent=1, default=str)
        except Exception:  # noqa: BLE001
            pass
        io.shutdown()


# ------------------------------------------------------------------------------------------------
def plan_only(io, args, seq, q_start, out, run_info) -> int:
    from ros_io import PlanError

    info("\nPLAN-ONLY: planning %d moves at scale %.2f from the previous planned end state; NO motion" % (len(seq), args.scale))
    info("%3s %5s %5s %5s %5s | %8s %9s %8s %6s  %s" % ("#", "tilt", "az", "roll", "dist", "dur [s]", "max dq[deg]", "detour", "colfree", "notes"))
    q_prev = q_start.copy()
    rows, ok_n = [], 0
    for i, c in enumerate(list(seq) + [None]):
        io.check_stop()
        goal = q_start if c is None else c.q
        label = "return" if c is None else str(i)
        try:
            times, pos, jt, dt = io.plan(q_prev, goal, args.scale)
            cf, nchk = io.path_collision_free(pos)
            prob = check_plan(q_prev, goal, pos, q_start, cf)
            direct = np.abs(goal - q_prev)
            travel = np.sum(np.abs(np.diff(pos, axis=0)), axis=0)
            row = {"move": label, "duration_s": float(times[-1]), "max_joint_change_deg": float(np.degrees(direct.max())),
                   "max_joint_travel_deg": float(np.degrees(travel.max())), "collision_free": bool(cf), "problems": prob,
                   "planning_s": dt, "n_points": len(pos), "ok": not prob}
            q_prev = pos[-1] if not prob else goal  # keep the chain going even for flagged plans (reported)
        except PlanError as e:
            row = {"move": label, "ok": False, "problems": [str(e)], "duration_s": None, "collision_free": None,
                   "max_joint_change_deg": float(np.degrees(np.abs(goal - q_prev).max())), "max_joint_travel_deg": None}
        rows.append(row)
        ok_n += bool(row["ok"]) and c is not None
        tl = "" if c is None else "%5.0f %5.0f %5.0f %5.2f" % (c.tilt, c.azimuth, c.roll, c.dist)
        info("%3s %-23s | %8s %9.1f %8s %6s  %s" % (label, tl, "-" if row["duration_s"] is None else "%.1f" % row["duration_s"],
                                                    row["max_joint_change_deg"],
                                                    "-" if row.get("max_joint_travel_deg") is None else "%.0f" % row["max_joint_travel_deg"],
                                                    {True: "yes", False: "NO", None: "-"}[row["collision_free"]], "; ".join(row["problems"])))
    info("\nreachable (plan found, all checks pass): %d / %d poses" % (ok_n, len(seq)))
    run_info["plan_results"] = rows
    run_info["reachable"] = ok_n
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump({"schema": "wrist_handeye_plan/v1", "scale": args.scale, "poses": run_info["poses"], "moves": rows, "reachable": ok_n,
                   "rejections": run_info["rejections"], "board": run_info.get("board_source"), "q_start": q_start.tolist()}, f, indent=1)
    info("plan saved to %s/plan.json" % out)
    return 0 if ok_n else 1


def execute_all(io, args, det, seq, q_start, X_est, K, D, size, T_bb, bw, bh, out, run_info, samples_rec, is_mock, ci) -> int:
    from ros_io import PlanError

    poses = list(seq)
    n = len(poses)
    capture = not args.mock_no_capture
    fails = 0
    done_moves = 0
    prompted_first = False
    X_ik = X_est.copy()
    info("\nEXECUTE: %d poses, scale %.2f%s" % (n, args.scale, ", NO CAPTURE (mock)" if not capture else ""))

    def plan_move(q_from, q_goal):
        times, pos, jt, dt = io.plan(q_from, q_goal, args.scale)
        cf, _ = io.path_collision_free(pos)
        cur = io.wait_joint_states()
        prob = check_plan(cur, q_goal, pos, q_start, cf)
        return times, pos, jt, prob

    def do_move(label, q_goal, desc):
        nonlocal prompted_first
        q_cur = io.wait_joint_states()
        times, pos, jt, prob = plan_move(q_cur, q_goal)
        if prob:
            raise PlanError("; ".join(prob))
        info("move %s: %s | max joint change %.1f deg, duration %.1f s, %d points" %
             (label, desc, np.degrees(np.abs(q_goal - q_cur).max()), times[-1], len(pos)))
        if not prompted_first:
            info("  FIRST MOVE. current joints (deg) %s" % np.round(np.degrees(q_cur), 1).tolist())
            info("  target  joints (deg) %s" % np.round(np.degrees(q_goal), 1).tolist())
            io.prompt("  Press Enter to execute the first move (q + Enter cancels): ")
            prompted_first = True
        elif not args.auto:
            io.prompt("  Enter = execute, q + Enter = quit: ")
        info("  executing ... (q or Ctrl-C cancels immediately)")
        io.execute(jt)
        io.wait_standstill(1.0, 0.002)

    try:
        k = 0
        while k < len(poses):
            io.check_stop()
            c = poses[k]
            try:
                do_move("%d/%d" % (k + 1, len(poses)), c.q, "tilt %.0f az %.0f roll %.0f dist %.2f" % (c.tilt, c.azimuth, c.roll, c.dist))
            except PlanError as e:
                fails += 1
                info("  SKIP pose %d: plan rejected: %s  (consecutive failures %d/3)" % (k, e, fails))
                if fails >= 3:
                    info("3 consecutive failures - stopping")
                    return 1
                k += 1
                continue
            # any other exception (execution error) propagates -> stop, no further motion
            done_moves += 1
            time.sleep(args.settle if capture else min(args.settle, 0.3))
            if not capture:
                q_now = io.wait_joint_states()
                err = float(np.max(np.abs(q_now - c.q)))
                info("  reached; max joint error to target %.4f rad" % err)
                samples_rec.append({"pose_index": k, "q": q_now.tolist(), "capture": False})
                fails = 0
                k += 1
                continue
            rec = capture_pose(io, args, det, k, out)
            if rec is None:
                fails += 1
                info("  capture failed (%d consecutive)" % fails)
                if fails >= 3:
                    info("3 consecutive failures - stopping")
                    save_samples(out, samples_rec)
                    break
                k += 1
                continue
            fails = 0
            samples_rec.append(rec)
            save_samples(out, samples_rec)
            info("  sample %d recorded: %d points (%d markers), frame spread %.2f px" % (len(samples_rec), rec["n_points"], rec["n_markers"], rec["frame_spread_px"]))
            # interim solve
            good = [s for s in samples_rec if s.get("obj")]
            if len(good) >= 4:
                try:
                    smp = [hm.Sample(_arr(s["T_base_link6"]), _arr(s["obj"]).reshape(-1, 3), _arr(s["pix"]).reshape(-1, 2)) for s in good]
                    r = hm.solve_handeye(smp, K, D, do_loo=False, X_hint=X_ik)
                    dmm, ddg = hm.pose_diff(r["X"], X_ik)
                    info("  interim solve (%d samples): RMS %.2f px, X moved %.1f mm / %.2f deg vs the one used for IK" % (len(good), r["rms_px"], dmm, ddg))
                    if r["rms_px"] < 3.0 and (dmm > 10.0 or ddg > 3.0) and k + 1 < len(poses):
                        X_ik = r["X"]
                        poses = poses[:k + 1] + reik(io, poses[k + 1:], X_ik, io.wait_joint_states(), args)
                except Exception as e:  # noqa: BLE001
                    info("  interim solve skipped: %s" % e)
            k += 1
        # ---- return to start
        if fails < 3:
            info("\nreturning to the start state")
            do_move("return", q_start, "start joint state")
    finally:
        io.keys.restore()
    save_samples(out, samples_rec)
    if not capture:
        info("\nmotion-only run: %d/%d poses reached, no samples to solve" % (done_moves, n))
        return 0 if done_moves == n else 1
    good = [s for s in samples_rec if s.get("obj")]
    info("\n%d valid samples of %d poses" % (len(good), n))
    if len(good) >= 3:
        res = solve_run_dir(out)
        if args.publish_tf:
            io.cleaning = True
            import ros_io
            ros_io.publish_static_tf("link6", CHILD_FRAME, res["X"])
        return 0 if res["status"] != "FAIL" else 1
    info("too few samples to solve")
    return 1


def _arr(x):
    return np.array(x, dtype=float)


def reik(io, remaining, X_new, q_from, args):
    out, prev = [], q_from
    for c in remaining:
        T6 = c.T_base_cam @ hm.inv_T(X_new)
        q = io.compute_ik(T6, prev, avoid_collisions=True)
        if q is None or np.max(np.abs(q - prev)) > math.radians(args.max_joint_step_deg):
            info("  re-IK with the new X: dropping a pose (tilt %.0f az %.0f roll %.0f dist %.2f)" % (c.tilt, c.azimuth, c.roll, c.dist))
            continue
        out.append(hm.Candidate(c.T_base_cam, c.tilt, c.azimuth, c.roll, c.dist, c.tag, q=q, T_base_link6=T6))
        prev = q
    info("  re-IK with the new X: %d/%d remaining poses kept" % (len(out), len(remaining)))
    return out


def capture_pose(io, args, det, k, out) -> Optional[dict]:
    import cv2
    t_ns = io.now_ns()
    frames = io.grab_frames("wrist", 3, t_ns, timeout=8.0)
    if len(frames) < 3:
        info("  only %d wrist frames received" % len(frames))
        return None
    dets = [detect(det, img, 1) for _, img in frames]
    counts = [0 if d is None else len(d["ids"]) for d in dets]
    b = int(np.argmax(counts))
    names = []
    for j, (st, img) in enumerate(frames):
        nm = "images/pose_%02d_f%d.png" % (k, j)
        cv2.imwrite(os.path.join(out, nm), img)
        names.append(nm)
    if counts[b] < args.min_points:
        info("  board points per frame %s (< %d)" % (counts, args.min_points))
        return None
    # frame-to-frame motion check on the ids common to all detected frames
    ref = dets[b]
    spread = 0.0
    for d in dets:
        if d is None or d is ref:
            continue
        common = np.intersect1d(ref["ids"], d["ids"])
        if len(common) >= 8:
            ia = [int(np.where(ref["ids"] == c)[0][0]) for c in common]
            ib = [int(np.where(d["ids"] == c)[0][0]) for c in common]
            spread = max(spread, float(np.max(np.linalg.norm(ref["pix"][ia] - d["pix"][ib], axis=1))))
    if spread > 2.0:
        info("  frames disagree by %.1f px: arm/camera still moving" % spread)
        return None
    st = frames[b][0]
    T6, exact = io.lookup("base_link", "link6", st)
    q = io.wait_joint_states()
    ci = io.wrist_info
    return {"pose_index": k, "stamp_ns": int(st), "q": q.tolist(), "T_base_link6": listT(T6), "tf_at_image_stamp": bool(exact),
            "n_points": int(counts[b]), "n_markers": int(ref["n_markers"]), "n_chess": int(ref["n_chess"]),
            "ids": ref["ids"].tolist(), "obj": ref["obj"].tolist(), "pix": ref["pix"].tolist(), "image": names[b], "images": names,
            "frame_spread_px": spread, "camera_info_stamp": None if ci is None else [ci.header.stamp.sec, ci.header.stamp.nanosec]}


def save_samples(out, recs):
    with open(os.path.join(out, "samples.json"), "w") as f:
        json.dump({"schema": "wrist_handeye_samples/v1", "samples": recs}, f)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "solve":
        return solve_main(sys.argv[2:])
    return run_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
