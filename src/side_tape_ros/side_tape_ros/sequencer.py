"""CLI: drive one side-tape cut end-to-end via side_tape_controller's services.

    ros2 run side_tape_ros sequencer --cut-length-mm 50 [--auto] \
        [--mode in_material|air_align] [--lock-timeout 10] [--align-timeout 20]

    ros2 run side_tape_ros sequencer --measure-only [--measure-frames 15]

    # with the static camera's side-face seed (docs/side_tape_seed_contract.md):
    ros2 run side_tape_ros sequencer --cut-length-mm 50 --seed-json SNAPSHOT.json --face left

``--mode`` defaults to ``control.start_mode`` from the loaded config
(``--config-path``, same default as side_tape_controller). See
docs/side_seam_architecture.md and config.py's ``ControlCfg.start_mode``:

- ``in_material`` (default): MoveIt has already placed the blade in the
  cardboard at the start pose and is idle. No probe, no ALIGN -- this tool
  waits for a locked, fresh, in-tolerance edge and calls start_cut directly.
- ``air_align``: the existing ALIGN (probe) -> READY -> CUT flow, blade clear
  of the material at the start pose.

``--measure-only`` never calls any service or moves anything: it prints the
median e_lat_mm/e_yaw_deg over ``--measure-frames`` fresh (distinct
frame_seq) observations and exits, so the operator can "see how much
correction is needed" before deciding whether to re-plan the MoveIt start
pose.

Ctrl-C at any point calls /side_tape/abort and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger

from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from . import control_math
from . import core_path  # noqa: F401
from precision_box_detector.side_tape.config import ConfigError, load_config  # noqa: E402
from precision_box_detector.side_tape.seed_check import check_start_pose, load_seed_block  # noqa: E402
from side_tape_interfaces.msg import Observation, SupervisorStatus
from side_tape_interfaces.srv import StartCut

_PRECONDITION_AIR_ALIGN = (
    "PRECONDITION: the robot must already be at the pre-cut pose via the existing "
    "MoveIt pipeline, and MoveIt must be idle (no trajectory executing), blade clear "
    "of the material. This tool only requests ALIGN/CUT from side_tape_controller; "
    "it never moves the arm itself."
)
_PRECONDITION_IN_MATERIAL = (
    "PRECONDITION: MoveIt has placed the blade at the start pose in the material and "
    "is idle (no trajectory executing). This tool only requests start_cut from "
    "side_tape_controller -- no probe, no ALIGN -- it never moves the arm itself."
)


class _Sequencer(Node):
    def __init__(self):
        super().__init__("side_tape_sequencer")
        self.latest_obs: Observation | None = None
        self.latest_status: SupervisorStatus | None = None
        self.create_subscription(Observation, "/side_tape/observation", self._on_obs, 10)
        self.create_subscription(SupervisorStatus, "/side_tape/status", self._on_status, 10)
        self.cli_align = self.create_client(Trigger, "/side_tape/start_align")
        self.cli_cut = self.create_client(StartCut, "/side_tape/start_cut")
        self.cli_abort = self.create_client(Trigger, "/side_tape/abort")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_obs(self, msg: Observation) -> None:
        self.latest_obs = msg

    def _on_status(self, msg: SupervisorStatus) -> None:
        self.latest_status = msg

    def call_trigger(self, client, name: str, timeout_sec: float = 5.0, aborted=None):
        if not client.wait_for_service(timeout_sec=3.0):
            return False, f"{name}_unavailable"
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if aborted is not None and aborted["flag"]:
                return False, f"{name}_aborted"
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            return False, f"{name}_timeout"
        res = future.result()
        if res is None:
            return False, f"{name}_no_response"
        return bool(res.success), str(res.message)

    def call_abort(self) -> None:
        ok, msg = self.call_trigger(self.cli_abort, "abort", timeout_sec=3.0)
        print(f"[sequencer] abort -> success={ok} message={msg}")


def _seed_check(node: _Sequencer, args, e_lat_mm, cut_length_mm) -> bool:
    """Start pose vs the static-camera side-face seed (seed_check.py). True = pass or no seed."""
    if args.seed is None:
        return True
    cfg = args.cfg
    tf = None
    deadline = time.monotonic() + 3.0
    while tf is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = node.tf_buffer.lookup_transform(cfg.robot.base_frame, cfg.robot.tool_frame, Time())
        except Exception:  # noqa: BLE001 - TF not there yet
            tf = None
    if tf is None:
        print("[sequencer] SEED CHECK FAILED: no TF base->tool")
        return False
    q, t = tf.transform.rotation, tf.transform.translation
    R = control_math.quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
    cut, lat, dep = control_math.freeze_axes(R, cfg.robot.cut_axis_tool, cfg.robot.lateral_axis_tool,
                                             cfg.robot.depth_axis_tool)
    sc = cfg.seed
    r = check_start_pose(
        args.seed, args.face, (t.x, t.y, t.z), cut, lat, dep,
        expected_penetration_mm=sc.expected_penetration_mm, penetration_tol_mm=sc.penetration_tol_mm,
        max_axis_angle_deg=sc.max_axis_angle_deg, min_below_top_mm=sc.min_below_top_mm,
        min_above_bottom_mm=sc.min_above_bottom_mm, cut_length_mm=cut_length_mm,
        face_end_margin_mm=sc.face_end_margin_mm, e_lat_mm=e_lat_mm, offset_mm=cfg.target.offset_mm,
        min_edge_below_top_mm=sc.min_edge_below_top_mm,
        allow_outer_wall_unknown=sc.allow_outer_wall_unknown)
    print("[sequencer] seed check (%s face): %s" % (
        args.face, ", ".join(f"{k}={v:.2f}" for k, v in r.metrics.items())))
    for w in r.warnings:
        print(f"[sequencer] seed WARNING: {w}")
    if not r.ok:
        print("[sequencer] SEED CHECK FAILED: " + "; ".join(r.reasons))
    return r.ok


def _run_measure_only(node: _Sequencer, args, aborted: dict) -> int:
    """No service calls, no motion: print the median e_lat_mm/e_yaw_deg over
    ``args.measure_frames`` fresh (distinct frame_seq), LOCKED observations,
    then exit (finding 9: "see how much correction is needed")."""
    print(f"[sequencer] measure-only: collecting {args.measure_frames} fresh LOCKED observations "
          f"(no service calls, no motion)...")
    seen_frame_seqs: set = set()
    e_lats: list = []
    e_yaws: list = []
    deadline = time.monotonic() + args.measure_timeout
    while (rclpy.ok() and not aborted["flag"] and len(e_lats) < args.measure_frames
           and time.monotonic() < deadline):
        rclpy.spin_once(node, timeout_sec=0.1)
        obs = node.latest_obs
        if obs is None or obs.frame_seq in seen_frame_seqs or obs.state != "LOCKED":
            continue
        seen_frame_seqs.add(obs.frame_seq)
        if obs.has_e_lat:
            e_lats.append(obs.e_lat_mm)
        if obs.has_e_yaw:
            e_yaws.append(obs.e_yaw_deg)
    if aborted["flag"]:
        return 1
    if not e_lats:
        print("[sequencer] FAILED: no usable (LOCKED, has_e_lat) observations within the timeout")
        return 1
    med_e_lat = statistics.median(e_lats)
    med_e_yaw = statistics.median(e_yaws) if e_yaws else float("nan")
    print(f"[sequencer] MEASURED over {len(e_lats)} fresh frame(s): "
          f"median e_lat_mm={med_e_lat:.3f} median e_yaw_deg={med_e_yaw:.3f}")
    print("[sequencer] no motion commanded (measure-only); use this to decide whether to "
          "re-plan the MoveIt start pose before an in-material start_cut.")
    if not _seed_check(node, args, med_e_lat, args.cut_length_mm):
        return 2
    return 0


def _wait_cut_result(node: _Sequencer, aborted: dict) -> int:
    print("[sequencer] cutting; waiting for DONE/HOLD/ABORT...")
    last_print = 0.0
    while rclpy.ok() and not aborted["flag"]:
        rclpy.spin_once(node, timeout_sec=0.1)
        st = node.latest_status
        if st is not None:
            now = time.monotonic()
            if now - last_print > 0.5:
                e = f"{st.e_lat_mm:.2f}" if st.has_e_lat else "n/a"
                print(f"[sequencer] phase={st.phase} along_mm={st.along_mm:.2f} e_lat={e} mm")
                last_print = now
            if st.phase in ("DONE", "HOLD", "ABORT"):
                break
    if aborted["flag"]:
        return 1
    st = node.latest_status
    print(f"[sequencer] SUMMARY: phase={st.phase} reasons={list(st.reasons)} "
          f"along_mm={st.along_mm:.2f} lat_dev_mm={st.lat_dev_mm:.2f} depth_dev_mm={st.depth_dev_mm:.2f}")
    return 0 if st.phase == "DONE" else 1


def _run_in_material(node: _Sequencer, args, aborted: dict) -> int:
    """in_material mode (finding 9): no probe, no ALIGN -- wait for a locked
    edge and call start_cut directly. A refusal that names the measured
    e_lat (``start_error_too_large``) is NOT transient: the operator has to
    re-plan the MoveIt start pose, so this does not retry that one."""
    print(f"[sequencer] in_material mode: waiting up to {args.lock_timeout:.1f}s for a locked edge "
          f"and a successful start_cut (retrying only while refused for an insufficient lock "
          f"streak or a stale/missing fix)...")
    deadline = time.monotonic() + args.lock_timeout
    last_attempt = 0.0
    cutting = False
    seed_checked = False
    while rclpy.ok() and not aborted["flag"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        obs = node.latest_obs
        if obs is None or obs.state != "LOCKED" or time.monotonic() - last_attempt <= 0.3:
            continue
        last_attempt = time.monotonic()
        if not seed_checked:
            if not _seed_check(node, args, obs.e_lat_mm if obs.has_e_lat else None, args.cut_length_mm):
                print("[sequencer] FAILED: start pose does not match the static-camera seed; "
                      "not calling start_cut")
                return 1
            seed_checked = True
        if not node.cli_cut.wait_for_service(timeout_sec=3.0):
            print("[sequencer] FAILED: start_cut service unavailable")
            return 1
        req = StartCut.Request()
        req.cut_length_mm = float(args.cut_length_mm)
        future = node.cli_cut.call_async(req)
        fut_deadline = time.monotonic() + 3.0
        while (rclpy.ok() and not aborted["flag"] and not future.done()
               and time.monotonic() < fut_deadline):
            rclpy.spin_once(node, timeout_sec=0.05)
        if aborted["flag"]:
            return 1
        res = future.result()
        if res is None:
            print("[sequencer] start_cut got no response; retrying")
            continue
        e = f"{obs.e_lat_mm:.2f}" if obs.has_e_lat else "n/a"
        print(f"[sequencer] LOCKED (e_lat={e} mm); start_cut -> "
              f"accepted={res.accepted} message={res.message}")
        if res.accepted:
            cutting = True
            break
        if "lock_streak" not in res.message and "no_usable_observation" not in res.message:
            print("[sequencer] FAILED: start_cut refused for a non-transient reason "
                  "-- if this is start_error_too_large, re-plan the MoveIt start pose "
                  "(see the measured e_lat_mm in the message above) and retry.")
            return 1
    if aborted["flag"]:
        return 1
    if not cutting:
        print("[sequencer] FAILED: did not reach a successful start_cut within lock-timeout")
        return 1
    return _wait_cut_result(node, aborted)


def _run_air_align(node: _Sequencer, args, aborted: dict) -> int:
    print(f"[sequencer] waiting up to {args.lock_timeout:.1f}s for LOCKED and a successful "
          f"start_align (retrying while only refused for an insufficient lock streak)...")
    deadline = time.monotonic() + args.lock_timeout
    aligned = False
    last_attempt = 0.0
    while rclpy.ok() and not aborted["flag"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        obs = node.latest_obs
        if obs is not None and obs.state == "LOCKED" and time.monotonic() - last_attempt > 0.3:
            last_attempt = time.monotonic()
            ok, msg = node.call_trigger(node.cli_align, "start_align", timeout_sec=3.0, aborted=aborted)
            print(f"[sequencer] LOCKED (e_lat={obs.e_lat_mm:.2f} mm); start_align -> "
                  f"success={ok} message={msg}")
            if ok:
                aligned = True
                break
            if "lock_streak" not in msg:
                print("[sequencer] FAILED: start_align refused for a non-transient reason")
                return 1
    if aborted["flag"]:
        return 1
    if not aligned:
        print("[sequencer] FAILED: did not reach a successful start_align within lock-timeout")
        return 1

    print(f"[sequencer] ALIGN started; waiting up to {args.align_timeout:.1f}s for READY...")
    deadline = time.monotonic() + args.align_timeout
    last_print = 0.0
    reached_ready = False
    seen_align = False
    while rclpy.ok() and not aborted["flag"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        st = node.latest_status
        if st is not None:
            now = time.monotonic()
            if now - last_print > 0.5:
                e = f"{st.e_lat_mm:.2f}" if st.has_e_lat else "n/a"
                print(f"[sequencer] phase={st.phase} e_lat={e} mm reasons={list(st.reasons)}")
                last_print = now
            if st.phase == "READY":
                reached_ready = True
                break
            if st.phase == "ALIGN":
                seen_align = True
            elif st.phase == "IDLE" and not seen_align:
                # The status publisher (20 Hz) and the control loop can race
                # right after start_align returns: the very first /side_tape/status
                # sample here may still be the pre-align IDLE snapshot, not a
                # real regression. Only treat IDLE as a failure once we've
                # actually observed ALIGN (the state machine has no other path
                # back to IDLE without HOLD/ABORT in between).
                continue
            elif st.phase in ("HOLD", "ABORT", "IDLE"):
                print(f"[sequencer] FAILED: supervisor left ALIGN for {st.phase} "
                      f"before READY: {list(st.reasons)}")
                return 1
    if aborted["flag"]:
        return 1
    if not reached_ready:
        print("[sequencer] FAILED: did not reach READY within align-timeout")
        return 1
    print("[sequencer] READY.")
    if args.align_only:
        print("[sequencer] --align-only: the tip is on the target line; NOT cutting. The controller stays "
              "READY (no motion) and pauses Servo after control.ready_timeout_s.")
        return 0

    if not args.auto:
        try:
            input("[sequencer] press Enter to start the cut (Ctrl-C to abort)... ")
        except (EOFError, KeyboardInterrupt):
            aborted["flag"] = True
            return 1

    print(f"[sequencer] calling start_cut (cut_length_mm={args.cut_length_mm})...")
    if not node.cli_cut.wait_for_service(timeout_sec=3.0):
        print("[sequencer] FAILED: start_cut service unavailable")
        return 1
    req = StartCut.Request()
    req.cut_length_mm = float(args.cut_length_mm)
    future = node.cli_cut.call_async(req)
    while rclpy.ok() and not aborted["flag"] and not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)
    if aborted["flag"]:
        return 1
    res = future.result()
    if res is None:
        print("[sequencer] FAILED: start_cut got no response")
        return 1
    print(f"[sequencer] start_cut -> accepted={res.accepted} message={res.message}")
    if not res.accepted:
        return 1

    return _wait_cut_result(node, aborted)


def _resolve_mode(args) -> str:
    if args.mode is not None:
        return args.mode
    try:
        cfg = load_config(args.config_path)
        return cfg.control.start_mode
    except (ConfigError, OSError) as exc:
        print(f"[sequencer] WARNING: could not load {args.config_path!r} ({exc}); "
              f"defaulting --mode to in_material")
        return "in_material"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut-length-mm", type=float, default=None,
                         help="required unless --measure-only")
    parser.add_argument("--auto", action="store_true",
                         help="skip the operator confirmation before starting the cut "
                              "(air_align mode only; in_material never prompts)")
    parser.add_argument("--mode", choices=("in_material", "air_align"), default=None,
                         help="default: control.start_mode from --config-path")
    parser.add_argument("--config-path", default=core_path.DEFAULT_CONFIG,
                         help="only used to resolve --mode's default and for --measure-only")
    parser.add_argument("--measure-only", action="store_true",
                         help="print median e_lat_mm/e_yaw_deg over --measure-frames fresh "
                              "LOCKED observations and exit; no service calls, no motion")
    parser.add_argument("--measure-frames", type=int, default=15)
    parser.add_argument("--measure-timeout", type=float, default=15.0)
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    parser.add_argument("--align-timeout", type=float, default=20.0)
    parser.add_argument("--align-only", action="store_true",
                         help="air_align: probe + align the tip onto the target line, stop at READY, never cut")
    parser.add_argument("--seed-json", default=None,
                         help="planner snapshot (or bare block) with the static camera's side_tape_seed; "
                              "checks the in-material start pose against it before start_cut")
    parser.add_argument("--face", choices=("left", "right", "back", "front"), default=None,
                         help="seed face being cut (required with --seed-json)")
    args = parser.parse_args(argv)

    args.seed = None
    args.cfg = None
    if args.seed_json is not None:
        if args.face is None:
            parser.error("--face is required with --seed-json")
        try:
            args.cfg = load_config(args.config_path)
            with open(args.seed_json, encoding="utf-8") as fh:
                args.seed = load_seed_block(json.load(fh))
        except (ConfigError, OSError, ValueError) as exc:
            parser.error(f"cannot use --seed-json: {exc}")
        if args.cfg.seed.expected_penetration_mm is None:
            parser.error("set seed.expected_penetration_mm in the config to use --seed-json")

    if args.align_only:
        if args.mode not in (None, "air_align"):
            parser.error("--align-only only works with --mode air_align")
        args.mode = "air_align"
        if args.cut_length_mm is None:
            args.cut_length_mm = 0.0
    if not args.measure_only and args.cut_length_mm is None:
        parser.error("--cut-length-mm is required unless --measure-only or --align-only is given")

    mode = _resolve_mode(args)
    if args.measure_only:
        print("[sequencer] --measure-only: no service calls, no motion.")
    elif mode == "in_material":
        print(_PRECONDITION_IN_MATERIAL)
    else:
        print(_PRECONDITION_AIR_ALIGN)

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = _Sequencer()
    aborted = {"flag": False}

    def _on_signal(signum, frame):  # noqa: ARG001
        aborted["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    exit_code = 1
    try:
        if args.measure_only:
            exit_code = _run_measure_only(node, args, aborted)
        elif mode == "in_material":
            exit_code = _run_in_material(node, args, aborted)
        else:
            exit_code = _run_air_align(node, args, aborted)
    finally:
        if aborted["flag"] and not args.measure_only:
            print("[sequencer] aborting...")
            node.call_abort()
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
