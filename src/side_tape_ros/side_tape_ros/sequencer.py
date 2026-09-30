"""CLI: drive one side-tape cut end-to-end via side_tape_controller's services.

    ros2 run side_tape_ros sequencer --cut-length-mm 50 [--auto] \
        [--mode in_material|air_align] [--lock-timeout 10] [--align-timeout 20]

    ros2 run side_tape_ros sequencer --measure-only [--measure-frames 15]

    # in_material: hold the rim distance the MoveIt start pose has (no hand-set offset):
    ros2 run side_tape_ros sequencer --cut-length-mm 50 --mode in_material --hold-start-offset

    # start from the box_h_cut_executor hand-off record (checked against TF first):
    # line_hold cut: hold the straight line the tool starts on (TF, not vision); vision only
    # monitors (HOLD line_vision_disagree) and, with a gain > 0, nudges the line <= 1.5 mm:
    ros2 run side_tape_ros sequencer --cut-length-mm 40 --hold-start-offset --line-hold [--line-vision-gain 0.2]

    ros2 run side_tape_ros sequencer --handoff ~/.ros/side_tape_handoff.json [--auto]

    # ENTRY (no cut), revised after the first real-robot test: the executor has put the tip standoff_mm in front
    # of the face (hand-off variant 'standoff'). No vision that far out: approach on TF to --pre-contact-mm in
    # front of the face, on to contact (standoff_mm + --contact-margin-mm; contact is NOT sensed, it is only as
    # accurate as the static camera's standoff), measure e_lat AT CONTACT (standoff 0), retry (back off, lateral
    # fix, re-contact) if |e_lat| > tol, then penetrate --penetration-mm (lateral held on TF, no vision after).
    # Leaves the blade in, writes ~/.ros/side_tape_entry.json (side_tape_entry/v1), hands back to the executor:
    ros2 run side_tape_ros sequencer --entry --handoff ~/.ros/side_tape_handoff.json \
        [--penetration-mm 4.8] [--entry-tol-mm 0.5] [--entry-offset-mm 0.0] [--insert-speed-mm-s 2.0] \
        [--max-retries 2] [--pre-contact-mm 2.0] [--contact-margin-mm 0.0] [--backoff-mm 1.0]

    # calibration datum (no motion): tip at contact and visually exactly on the rim -> prints the
    # target.offset_mm that makes e_lat 0 there (config offset + median e_lat); never writes the yaml:
    ros2 run side_tape_ros sequencer --calibrate-rim-contact [--auto]

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
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger
from std_msgs.msg import Float64
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from . import control_math
from . import core_path  # noqa: F401
from . import entry as entry_mod
from . import handoff as handoff_mod
from . import hold_offset
from .conversions import stamp_to_s
from precision_box_detector.side_tape.config import ConfigError, load_config  # noqa: E402
from precision_box_detector.side_tape.insert_move import EntryParams, EntryPlan  # noqa: E402
from precision_box_detector.side_tape.seed_check import check_start_pose, load_seed_block  # noqa: E402
from rcl_interfaces.srv import GetParameters
from side_tape_interfaces.msg import Observation, SupervisorStatus
from side_tape_interfaces.srv import StartCut, StartInsert, StartLateral

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
_PRECONDITION_ENTRY = (
    "PRECONDITION: the MoveIt executor has moved the tip to the hand-off approach pose (variant 'standoff', tip "
    "standoff_mm in front of the face) and is idle (no trajectory executing, arm at rest). This tool then moves the "
    "arm ITSELF through side_tape_controller (MoveIt Servo, TF closed loop): approach, contact, penetrate. It never "
    "cuts and never retracts."
)
_RETRACT_HINT = (
    "[sequencer] hint: call `ros2 service call /box_h_cut_executor/side_tape_retract "
    "std_srvs/srv/Trigger` to retract with MoveIt (not called automatically)."
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
        self.cli_stop = self.create_client(Trigger, "/side_tape/stop")
        self.cli_insert = self.create_client(StartInsert, "/side_tape/start_insert")
        self.cli_lateral = self.create_client(StartLateral, "/side_tape/start_lateral")
        self.cli_set_params = self.create_client(SetParameters, "/side_tape_controller/set_parameters")
        self.cli_get_params = self.create_client(GetParameters, "/side_tape_controller/get_parameters")
        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_standoff = self.create_publisher(Float64, "/side_tape/standoff_mm", latched)
        self.pub_trim = self.create_publisher(Float64, "/side_tape/offset_trim_mm", latched)
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


def _param_msg(name: str, value) -> Parameter:
    pv = ParameterValue()
    if isinstance(value, str):
        pv.type, pv.string_value = ParameterType.PARAMETER_STRING, value
    elif isinstance(value, int) and not isinstance(value, bool):
        pv.type, pv.integer_value = ParameterType.PARAMETER_INTEGER, int(value)
    else:
        pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, float(value)
    return Parameter(name=name, value=pv)


def _set_controller_params(node: _Sequencer, params: dict, timeout_sec: float = 3.0) -> tuple:
    """Set parameters on /side_tape_controller (all-or-report). Returns (ok, message)."""
    if not node.cli_set_params.wait_for_service(timeout_sec=3.0):
        return False, "set_parameters_unavailable"
    req = SetParameters.Request()
    req.parameters = [_param_msg(k, v) for k, v in params.items()]
    future = node.cli_set_params.call_async(req)
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    res = future.result() if future.done() else None
    if res is None:
        return False, "set_parameters_timeout"
    bad = [f"{k}: {r.reason}" for k, r in zip(params, res.results) if not r.successful]
    return (not bad), ("; ".join(bad) if bad else "ok")


def _get_controller_params(node: _Sequencer, names, timeout_sec: float = 3.0):
    """Read parameters back from /side_tape_controller. Returns ({name: value}, message)."""
    if not node.cli_get_params.wait_for_service(timeout_sec=3.0):
        return {}, "get_parameters_unavailable"
    req = GetParameters.Request()
    req.names = list(names)
    future = node.cli_get_params.call_async(req)
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    res = future.result() if future.done() else None
    if res is None:
        return {}, "get_parameters_timeout"
    return control_math.parse_get_parameters_result(list(names), res.values), "ok"


def _set_and_verify_controller_params(node: _Sequencer, params: dict) -> tuple:
    """Set ``params`` on /side_tape_controller, then READ THEM BACK and compare. Returns (ok, message);
    a set that reports success but reads back differently (or cannot be read) is a failure: the caller
    must not start a motion on it."""
    ok, msg = _set_controller_params(node, params)
    if not ok:
        return False, msg
    got, gmsg = _get_controller_params(node, list(params))
    if gmsg != "ok":
        return False, f"read-back failed: {gmsg}"
    bad = control_math.params_mismatch(params, got)
    if bad:
        return False, "read-back mismatch: " + "; ".join(bad)
    return True, "ok (read back)"


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
        face_end_margin_mm=sc.face_end_margin_mm, e_lat_mm=e_lat_mm, offset_mm=cfg.target.offset_mm + float(getattr(args, 'hold_trim_mm', 0.0)),
        min_edge_below_top_mm=sc.min_edge_below_top_mm,
        allow_outer_wall_unknown=sc.allow_outer_wall_unknown)
    print("[sequencer] seed check (%s face): %s" % (
        args.face, ", ".join(f"{k}={v:.2f}" for k, v in r.metrics.items())))
    for w in r.warnings:
        print(f"[sequencer] seed WARNING: {w}")
    if not r.ok:
        print("[sequencer] SEED CHECK FAILED: " + "; ".join(r.reasons))
    return r.ok


def _collect_e_lats(node: _Sequencer, n: int, timeout_s: float, aborted: dict, min_frame_seq=None,
                    want_yaw: bool = False):
    """Collect e_lat (and e_yaw) from ``n`` fresh (distinct frame_seq) LOCKED observations."""
    seen_frame_seqs: set = set()
    e_lats: list = []
    e_yaws: list = []
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and not aborted["flag"] and len(e_lats) < n and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        obs = node.latest_obs
        if obs is None or obs.frame_seq in seen_frame_seqs or obs.state != "LOCKED":
            continue
        if min_frame_seq is not None and obs.frame_seq <= min_frame_seq:
            continue
        seen_frame_seqs.add(obs.frame_seq)
        if obs.has_e_lat:
            e_lats.append(obs.e_lat_mm)
        if obs.has_e_yaw:
            e_yaws.append(obs.e_yaw_deg)
    return (e_lats, e_yaws) if want_yaw else e_lats


def _collect_gated_e_lats(node, cfg, n: int, timeout_s: float, aborted: dict, min_frame_seq=None):
    """e_lat of ``n`` distinct-frame observations that ALL pass the decision gate (entry.sample_ok: LOCKED, finite,
    age <= control.stale_zero_s, not from the future, conf >= tracker.min_conf, |e_yaw| <= max_edge_angle_deg).
    Returns (samples, {reject_reason: count}); fewer than ``n`` samples = the caller must fail."""
    seen: set = set()
    good: list = []
    rejected: dict = {}
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and not aborted["flag"] and len(good) < n and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        obs = node.latest_obs
        if obs is None or obs.frame_seq in seen:
            continue
        if min_frame_seq is not None and obs.frame_seq <= min_frame_seq:
            continue
        seen.add(obs.frame_seq)
        why = entry_mod.sample_ok(
            stamp_s=stamp_to_s(obs.header.stamp), now_s=node.get_clock().now().nanoseconds * 1e-9,
            state=obs.state, has_e_lat=obs.has_e_lat, e_lat=obs.e_lat_mm, conf=obs.conf, has_e_yaw=obs.has_e_yaw,
            e_yaw=obs.e_yaw_deg, stale_s=cfg.control.stale_zero_s, min_conf=cfg.tracker.min_conf,
            max_yaw_deg=cfg.target.max_edge_angle_deg)
        if why is not None:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        good.append(float(obs.e_lat_mm))
    return good, rejected


def _publish_trim(node: _Sequencer, trim_mm: float, settle_s: float = 1.0) -> None:
    node.pub_trim.publish(Float64(data=float(trim_mm)))
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_s:  # let perception apply it and publish new observations
        rclpy.spin_once(node, timeout_sec=0.05)


def _hold_start_offset(node: _Sequencer, args, aborted: dict) -> bool:
    """Trim the perception offset so the current e_lat becomes ~0 (knife holds the start distance
    from the rim). Currently applied trim is 0 (published at start). True = held; False = refused
    (trim is reset to 0)."""
    cfg_off = float(args.cfg.target.offset_mm) if args.cfg is not None else float("nan")
    print(f"[sequencer] hold-start-offset: measuring {args.measure_frames} fresh LOCKED frames...")
    e_lats = _collect_e_lats(node, args.measure_frames, args.measure_timeout, aborted)
    if aborted["flag"]:
        return False
    try:
        new_trim, med, spread = hold_offset.plan_hold_trim(e_lats, 0.0, args.measure_frames)
    except ValueError as exc:
        print(f"[sequencer] FAILED: hold-start-offset refused: {exc}")
        return False
    print(f"[sequencer] measured median e_lat={med:+.3f} mm spread(p90-p10)={spread:.3f} mm "
          f"({len(e_lats)} frames); applying trim {new_trim:+.3f} mm")
    last_seq = node.latest_obs.frame_seq if node.latest_obs is not None else None
    _publish_trim(node, new_trim, settle_s=0.5)
    deadline = time.monotonic() + args.hold_timeout
    ok = False
    last_e = None
    while rclpy.ok() and not aborted["flag"] and time.monotonic() < deadline:
        fresh = _collect_e_lats(node, 3, 1.0, aborted, min_frame_seq=last_seq)
        if len(fresh) == 3:
            last_e = statistics.median(fresh)
            if abs(last_e) < hold_offset.CONVERGED_MM and max(abs(x) for x in fresh) < hold_offset.CONVERGED_MM:
                ok = True
                break
            last_seq = node.latest_obs.frame_seq
    if not ok:
        if not aborted["flag"]:
            print(f"[sequencer] FAILED: e_lat did not converge to |e_lat|<{hold_offset.CONVERGED_MM} mm "
                  f"after trim (last median {last_e}); resetting trim to 0")
        _publish_trim(node, 0.0, settle_s=0.3)
        return False
    args.hold_trim_mm = new_trim
    tgt = args.cfg.target
    par = float(tgt.parallax_mm_per_mm) * float(args.standoff_used)
    eff = cfg_off + par + new_trim
    print(f"[sequencer] HELD START OFFSET: config offset_mm={cfg_off:+.3f} + parallax {par:+.3f} "
          f"+ trim {new_trim:+.3f} = distance from the rim (tool lateral, target line) {eff:+.3f} mm; "
          f"e_lat now {last_e:+.3f} mm")
    return True


def _run_measure_only(node: _Sequencer, args, aborted: dict) -> int:
    """No service calls, no motion: print the median e_lat_mm/e_yaw_deg over
    ``args.measure_frames`` fresh (distinct frame_seq), LOCKED observations,
    then exit (finding 9: "see how much correction is needed")."""
    print(f"[sequencer] measure-only: collecting {args.measure_frames} fresh LOCKED observations "
          f"(no service calls, no motion)...")
    e_lats, e_yaws = _collect_e_lats(node, args.measure_frames, args.measure_timeout, aborted, want_yaw=True)
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


def _wait_cut_result(node: _Sequencer, aborted: dict, retract_hint: bool = False) -> int:
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
    if retract_hint and st.phase in ("DONE", "ABORT"):
        print(_RETRACT_HINT)
    return 0 if st.phase == "DONE" else 1


def _run_in_material(node: _Sequencer, args, aborted: dict) -> int:
    """in_material mode (finding 9): no probe, no ALIGN -- wait for a locked
    edge and call start_cut directly. A refusal that names the measured
    e_lat (``start_error_too_large``) is NOT transient: the operator has to
    re-plan the MoveIt start pose, so this does not retry that one."""
    if getattr(args, "hold_start_offset", False):
        if not _hold_start_offset(node, args, aborted):
            return 1
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
    return _wait_cut_result(node, aborted, bool(getattr(args, 'handoff', None)))


def _align_to_ready(node: _Sequencer, args, aborted: dict) -> int:
    """start_align and wait for READY (the tip is on the target line). Returns 0 = READY, else 1."""
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
    return 0


def _run_air_align(node: _Sequencer, args, aborted: dict) -> int:
    rc = _align_to_ready(node, args, aborted)
    if rc != 0:
        return rc
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

    return _wait_cut_result(node, aborted, bool(getattr(args, 'handoff', None)))


def _lookup_tool_pose(node: _Sequencer, cfg, timeout_sec: float = 3.0):
    """(pos_m, quat_xyzw) of tool in base from TF, or None."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = node.tf_buffer.lookup_transform(cfg.robot.base_frame, cfg.robot.tool_frame, Time())
        except Exception:  # noqa: BLE001 - TF not there yet
            continue
        t, q = tf.transform.translation, tf.transform.rotation
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)
    return None


def _handoff_check(node: _Sequencer, args) -> bool:
    """Validate the hand-off record against config and the current TF pose. True = OK."""
    pos = quat = None
    problems = []
    if args.ignore_handoff_pose:
        print("[sequencer] --ignore-handoff-pose: skipping pose-proximity checks")
    else:
        pq = _lookup_tool_pose(node, args.cfg)
        if pq is None:
            problems.append("no TF %s -> %s, cannot check the start pose (use --ignore-handoff-pose "
                            "to skip)" % (args.cfg.robot.base_frame, args.cfg.robot.tool_frame))
        else:
            pos, quat = pq
    tol = {}
    if getattr(args, "entry", False):  # the TF-only approach starts from this pose: tighter than the cut flows
        tol = {"pos_tol_mm": entry_mod.ENTRY_POS_TOL_MM, "ori_tol_deg": entry_mod.ENTRY_ORI_TOL_DEG}
    problems += handoff_mod.check_handoff(args.handoff_rec, args.cfg.robot, pos, quat, **tol)
    if problems:
        for pr in problems:
            print(f"[sequencer] HANDOFF PROBLEM: {pr}")
        print("[sequencer] FAILED: hand-off record check failed; nothing was commanded")
        return False
    print("[sequencer] hand-off record OK")
    return True


# -- ENTRY mode (no cut): tip onto the rim line, a few mm into the material, verify, hand back ----------

def _call_move_service(node: _Sequencer, client, req, name: str, aborted: dict, timeout_sec: float = 5.0):
    """Call start_insert / start_lateral. Returns (accepted, message, move_id)."""
    if not client.wait_for_service(timeout_sec=3.0):
        return False, f"{name}_unavailable", 0
    future = client.call_async(req)
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not aborted["flag"] and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if aborted["flag"]:
        return False, f"{name}_aborted", 0
    res = future.result() if future.done() else None
    if res is None:
        return False, f"{name}_no_response", 0
    return bool(res.accepted), str(res.message), int(res.move_id)


def _wait_move_done(node: _Sequencer, aborted: dict, label: str, timeout_s: float, move_id: int):
    """After an ACCEPTED move: wait for the status of THIS move (``move_id`` from the service reply, echoed in
    /side_tape/status) to show INSERTED. Statuses of other moves are ignored, so a stale resting INSERTED can
    never be taken for completion. HOLD/ABORT/IDLE with this id is a failure. Returns (ok, message, status)."""
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while rclpy.ok() and not aborted["flag"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        st = node.latest_status
        if st is None or st.move_id != move_id:
            continue
        now = time.monotonic()
        if st.phase in ("INSERT", "SHIFT"):
            if now - last_print > 0.5:
                print(f"[sequencer] {label}: phase={st.phase} depth_dev={st.depth_dev_mm:+.2f} "
                      f"lat_dev={st.lat_dev_mm:+.2f} along={st.along_mm:+.2f} mm reasons={list(st.reasons)}")
                last_print = now
        elif st.phase == "INSERTED":
            return True, "ok", st
        elif st.phase in ("HOLD", "ABORT", "IDLE"):
            return False, f"supervisor went {st.phase}: {list(st.reasons)}", st
    if aborted["flag"]:
        return False, "aborted", node.latest_status
    return False, f"{label} did not finish within {timeout_s:.0f} s", node.latest_status


def _stop_and_confirm(node: _Sequencer, timeout_s: float = 8.0, quiet_s: float = 0.3):
    """Command the motionless stop (/side_tape/stop: latches the ABORT stop, never retracts) and VERIFY from
    /side_tape/status that the arm is no longer commanded: statuses stamped after the stop reply, not in a
    driving phase, zero twist, continuously for ``quiet_s``. Returns (confirmed, message). Never assumes."""
    last = "no reply"
    t_reply = node.get_clock().now().nanoseconds * 1e-9
    for _attempt in range(3):  # a lost reply must not leave the arm driving
        ok, last = node.call_trigger(node.cli_stop, "stop", timeout_sec=2.0)
        t_reply = node.get_clock().now().nanoseconds * 1e-9
        if ok:
            break
    deadline = time.monotonic() + timeout_s
    quiet_since = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        st = node.latest_status
        if st is None or stamp_to_s(st.stamp) <= t_reply:
            quiet_since = None  # older than the stop request: proves nothing
            continue
        if entry_mod.is_stopped(st.phase, st.v_cut_mps, st.v_lat_mps, st.v_depth_mps):
            quiet_since = quiet_since if quiet_since is not None else time.monotonic()
            if time.monotonic() - quiet_since >= quiet_s:
                return True, f"stopped: phase={st.phase} reasons={list(st.reasons)}"
        else:
            quiet_since = None
    return False, f"NOT confirmed stopped (stop reply: {last})"


def _entry_measure(node: _Sequencer, args, aborted: dict):
    """Median e_lat over ``args.measure_frames`` fresh LOCKED frames at standoff 0 (tip in the material),
    spread gate hold_offset.MAX_SPREAD_MM (p90 - p10). Up to 3 attempts. Returns (median, spread, n, message)."""
    node.pub_standoff.publish(Float64(data=0.0))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:  # let perception apply standoff 0 before the frames that count
        rclpy.spin_once(node, timeout_sec=0.05)
    last_msg = "no measurement"
    for attempt in range(3):
        seq0 = node.latest_obs.frame_seq if node.latest_obs is not None else None
        e_lats, rejected = _collect_gated_e_lats(node, args.cfg, args.measure_frames, args.measure_timeout,
                                                 aborted, min_frame_seq=seq0)
        if aborted["flag"]:
            return None, None, 0, "aborted"
        try:
            _, med, spread = hold_offset.plan_hold_trim(e_lats, 0.0, args.measure_frames)
        except ValueError as exc:
            last_msg = f"{exc}; samples rejected by the decision gate: {rejected}"
            print(f"[sequencer] measurement attempt {attempt + 1}/3 refused: {last_msg}")
            continue
        return med, spread, len(e_lats), "ok"
    return None, None, 0, last_msg


def _lookup_pose_or_none(node: _Sequencer, cfg):
    try:
        return _lookup_tool_pose(node, cfg, timeout_sec=2.0)
    except Exception:  # noqa: BLE001 - the record must still be written
        return None


def _entry_params(args) -> EntryParams:
    return EntryParams(standoff_mm=args.standoff_mm, pre_contact_mm=args.pre_contact_mm,
                       contact_margin_mm=args.contact_margin_mm, penetration_mm=args.penetration_mm,
                       tol_mm=args.entry_tol_mm, max_retries=args.max_retries, backoff_mm=args.backoff_mm,
                       max_standoff_mm=args.max_standoff_mm)


def _tf_quiet(node: _Sequencer, cfg, window_s: float = 0.6, max_move_mm: float = 0.15) -> bool:
    """True if the tool has not moved more than ``max_move_mm`` over ``window_s`` (the executor's trajectory
    has really finished and the arm is at rest before the TF-only approach is frozen on TF)."""
    a = _lookup_tool_pose(node, cfg, timeout_sec=2.0)
    if a is None:
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < window_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    b = _lookup_tool_pose(node, cfg, timeout_sec=2.0)
    if b is None:
        return False
    return 1000.0 * (sum((x - y) ** 2 for x, y in zip(a[0], b[0])) ** 0.5) <= max_move_mm


def _run_entry(node: _Sequencer, args, aborted: dict) -> int:
    """ENTRY (revised): TF-only approach to pre_contact -> contact step -> measure e_lat AT CONTACT ->
    (|e_lat| > tol and retries left: back off, lateral fix, re-contact, measure) -> penetrate -> record.
    No vision is used before contact and none after the penetration; no alignment at the approach standoff.
    Never cuts, never retracts. Returns 0 ok, 2 out_of_tolerance, 1 failed; always writes the record."""
    rec = args.handoff_rec
    params = _entry_params(args)
    plan = EntryPlan(params)
    state = {"iterations": 0, "e_hist": [], "spread": None, "final_e": None, "penetrated": False}

    def finish(status: str, reason: str = "") -> int:
        if status == "failed":
            # Any failure, timeout or lost reply: the supervisor may still be commanding a move. Stop it
            # (motionless latch, never a retract) and VERIFY from the status before anything else.
            stopped, smsg = _stop_and_confirm(node)
            print(f"[sequencer] stop: {smsg}")
            if not stopped:
                reason += ";MOTION_NOT_CONFIRMED_STOPPED"
                print("[sequencer] !!! COULD NOT CONFIRM THAT THE ARM STOPPED: check the robot / press the e-stop !!!")
        pose = _lookup_pose_or_none(node, args.cfg)  # where the tool is now, also for a failure
        record = entry_mod.build_record(
            status=status, reason=reason, tool_pose=pose, final_e_lat_mm=state["final_e"],
            entry_offset_mm=args.entry_offset_mm, penetration_mm=args.penetration_mm,
            standoff_mm=args.standoff_mm, tol_mm=args.entry_tol_mm, insert_speed_mm_s=args.insert_speed_mm_s,
            iterations=state["iterations"], max_retries=args.max_retries, e_lat_history_mm=state["e_hist"],
            spread_mm=state["spread"], handoff_rec=rec, base_frame=args.cfg.robot.base_frame,
            penetrated=state["penetrated"], pre_contact_mm=args.pre_contact_mm,
            contact_margin_mm=args.contact_margin_mm, backoff_mm=args.backoff_mm)
        wrote = True
        try:
            path = entry_mod.write_record_atomic(args.entry_record_path, record)
            print(f"[sequencer] entry record written: {path}")
        except (OSError, ValueError) as exc:
            wrote = False
            print(f"[sequencer] ERROR: could not write the entry record {args.entry_record_path!r}: {exc}")
        print("[sequencer] ENTRY SUMMARY:")
        for line in entry_mod.summary_lines(record):
            print(f"[sequencer]   {line}")
        if not wrote:
            return 1  # the executor reads that record: no record = no success, whatever the motion result
        return {"ok": 0, "out_of_tolerance": 2}.get(status, 1)

    print(f"[sequencer] ENTRY: standoff {params.standoff_mm:.2f} mm (hand-off) -> TF-only approach "
          f"{params.approach_mm:.2f} mm (stop {params.pre_contact_mm:.2f} mm in front of the face) -> contact step "
          f"{params.contact_step_mm:.2f} mm (contact margin {params.contact_margin_mm:+.2f}; contact is NOT sensed) "
          f"-> measure e_lat at contact (tol {params.tol_mm:.2f} mm, retries {params.max_retries}, back-off "
          f"{params.backoff_mm:.2f} mm) -> penetrate {params.penetration_mm:.2f} mm; speed "
          f"{args.insert_speed_mm_s:.2f} mm/s, entry offset {args.entry_offset_mm:+.3f} mm")
    if not _tf_quiet(node, args.cfg):
        print("[sequencer] FAILED: the tool is not at rest (or no TF): the executor's trajectory must be finished; "
              "nothing was commanded")
        return finish("failed", "arm_not_at_rest_or_no_tf")

    def move(kind: str, distance_mm: float, label: str, first: bool = False) -> tuple:
        if not first:  # every later move needs the supervisor still resting in INSERTED (no IDLE timeout)
            spin_until = time.monotonic() + 0.3
            while time.monotonic() < spin_until:
                rclpy.spin_once(node, timeout_sec=0.05)
            st = node.latest_status
            if st is None or st.phase != "INSERTED":
                return False, f"{label}_not_started:supervisor is in {None if st is None else st.phase}, not INSERTED"
        if kind == "depth":
            req = StartInsert.Request()
            req.depth_mm, req.speed_mm_s = float(distance_mm), float(args.insert_speed_mm_s)
            req.depth_limit_mm = float(params.depth_limit_mm)  # absolute, from the entry start; kept by the supervisor
            client, name = node.cli_insert, "start_insert"
        else:
            req = StartLateral.Request()
            req.lateral_mm, req.air_depth_mm = float(distance_mm), float(params.air_depth_mm)
            client, name = node.cli_lateral, "start_lateral"
        print(f"[sequencer] {label}: {name}({distance_mm:+.3f} mm)")
        acc, msg, move_id = _call_move_service(node, client, req, name, aborted)
        print(f"[sequencer] {name} -> accepted={acc} message={msg} move_id={move_id}")
        if not acc:
            return False, f"{name}_refused:{msg}"
        speed = args.insert_speed_mm_s if kind == "depth" else 1.5
        ok, wmsg, _st = _wait_move_done(node, aborted, label, 2.0 * abs(distance_mm) / speed + 20.0, move_id)
        return ok, ("" if ok else f"{label}_failed:{wmsg}")

    def fail(why: str) -> int:
        print(f"[sequencer] FAILED: {why}")
        return finish("failed", "aborted" if aborted["flag"] else why)

    def sequence() -> int:
        # the approach is TF only: vision is never asked before contact
        for mv, first in ((plan.approach_move(), True), (plan.contact_move(), False)):
            ok, why = move(mv.axis, mv.distance_mm, mv.label, first)
            if not ok:
                return fail(why)
        while True:
            state["iterations"] += 1
            med, spread, n, msg = _entry_measure(node, args, aborted)
            if med is None:
                return fail(f"measurement_failed:{msg}")
            state["e_hist"].append(med)
            state["final_e"], state["spread"] = med, spread
            print(f"[sequencer] AT CONTACT (measurement {state['iterations']}): median e_lat={med:+.3f} mm "
                  f"spread(p90-p10)={spread:.3f} mm over {n} frames (tolerance {params.tol_mm:.2f} mm)")
            dec = plan.decide(med, state["iterations"] - 1)
            if dec.kind == "ok":
                break
            if dec.kind != "retry":
                print(f"[sequencer] ENTRY {dec.kind.upper()}: {dec.reason}; NOT penetrating, the tip is left at contact")
                return finish(dec.kind, dec.reason)
            print(f"[sequencer] |e_lat| {abs(med):.3f} > tol {params.tol_mm:.2f}: retry {state['iterations']}/"
                  f"{params.max_retries}: back off {params.backoff_mm:.2f} mm, shift {-med:+.3f} mm, back to contact")
            for mv in dec.moves:
                ok, why = move(mv.axis, mv.distance_mm, mv.label)
                if not ok:
                    return fail(why)
        pm = plan.penetrate_move()
        ok, why = move(pm.axis, pm.distance_mm, pm.label)
        if not ok:
            return fail(why)
        state["penetrated"] = True
        print(f"[sequencer] ENTRY OK: penetrated {params.penetration_mm:.2f} mm past contact (lateral held on TF); "
              f"the blade is left in the material; no vision after the penetration")
        return finish("ok")

    try:
        return sequence()
    except Exception as exc:  # noqa: BLE001 - whatever happened, the arm must be stopped and the record written
        return fail(f"exception:{type(exc).__name__}:{exc}")


def _run_calibrate_rim_contact(node: _Sequencer, args, aborted: dict) -> int:
    """No motion, no service calls: with the tip at contact and exactly on the rim, print the target.offset_mm
    that makes e_lat 0 (config offset + median e_lat; measured with standoff 0 and trim 0, both published)."""
    cfg_off = float(args.cfg.target.offset_mm)
    print("[sequencer] CALIBRATE RIM CONTACT: put the tip at CONTACT with the face, visually EXACTLY on the top rim "
          "(e.g. by teleop). Nothing is moved by this tool.")
    if not args.auto:
        try:
            input("[sequencer] confirm the tip is exactly on the rim at contact, then press Enter (Ctrl-C = cancel)... ")
        except (EOFError, KeyboardInterrupt):
            return 1
    print(f"[sequencer] measuring {args.measure_frames} fresh LOCKED frames (standoff 0, trim 0)...")
    e_lats, rejected = _collect_gated_e_lats(node, args.cfg, args.measure_frames, args.measure_timeout, aborted)
    if aborted["flag"]:
        return 1
    try:
        new_off, med, spread = entry_mod.rim_contact_offset(cfg_off, e_lats, args.measure_frames)
    except ValueError as exc:
        print(f"[sequencer] FAILED: {exc} (samples rejected by the gate: {rejected})")
        return 1
    print(f"[sequencer] median e_lat={med:+.3f} mm spread(p90-p10)={spread:.3f} mm over {len(e_lats)} frames; "
          f"current target.offset_mm={cfg_off:+.3f}")
    print(f"[sequencer] RESULT: target.offset_mm = {new_off:+.3f}   (= {cfg_off:+.3f} + {med:+.3f}; makes e_lat 0 "
          f"with the tip on the rim at contact). NOT written anywhere: edit knife_mount.yaml yourself if you agree.")
    return 0


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
    parser.add_argument("--standoff-mm", type=float, default=None,
                         help="tip distance in front of the face for the parallax correction; default: "
                              "target.align_standoff_mm for air_align/--align-only, 0 for in_material")
    parser.add_argument("--align-only", action="store_true",
                         help="air_align: probe + align the tip onto the target line, stop at READY, never cut")
    parser.add_argument("--seed-json", default=None,
                         help="planner snapshot (or bare block) with the static camera's side_tape_seed; "
                              "checks the in-material start pose against it before start_cut")
    parser.add_argument("--handoff", default=None, metavar="JSON",
                         help="box_h_cut_executor hand-off record (box_h_cut_side_tape_handoff/v1); "
                              "sets cut length (min with --cut-length-mm), mode and standoff, and is "
                              "checked against config and the current TF pose before anything runs")
    parser.add_argument("--ignore-handoff-pose", action="store_true",
                         help="with --handoff: skip only the pose-proximity checks (teleop-placed tests)")
    parser.add_argument("--hold-start-offset", action="store_true",
                         help="in_material only: measure the rim distance at the start and hold it during the cut "
                              "(sets a perception offset trim so e_lat=0 now; replaces a hand-set target.offset_mm)")
    parser.add_argument("--hold-timeout", type=float, default=5.0,
                         help="with --hold-start-offset: seconds to wait for |e_lat|<0.2 mm after applying the trim")
    parser.add_argument("--line-hold", action="store_true",
                         help="cut in line_hold mode: lateral control holds the straight line the tool starts "
                              "on (TF), vision only monitors it (HOLD line_vision_disagree) and, with "
                              "--line-vision-gain > 0, nudges it (<= 1.5 mm, <= 0.5 mm/s). Sets cut_mode on "
                              "/side_tape_controller before start_cut and restores cut_mode=vision afterwards")
    parser.add_argument("--line-vision-gain", type=float, default=None,
                         help="with --line-hold: vision blend gain in [0, 1] (default 0 = pure straight line)")
    parser.add_argument("--entry", action="store_true",
                         help="ENTRY mode (no cut): needs --handoff with variant 'standoff'. Aligns the tip onto the "
                              "top-rim line at the standoff, pushes it --penetration-mm into the material along "
                              "+depth (TF closed loop), verifies e_lat with vision (retry: back out, lateral fix in "
                              "the air, re-insert), leaves the blade in and writes ~/.ros/side_tape_entry.json")
    parser.add_argument("--penetration-mm", type=float, default=None,
                         help="--entry: how far past contact to push in after the contact measurement is good "
                              "(default 4.8, the operator-confirmed cut depth)")
    parser.add_argument("--pre-contact-mm", type=float, default=None,
                         help="--entry: the TF-only approach stops this far in front of the face (default 2.0)")
    parser.add_argument("--contact-margin-mm", type=float, default=None,
                         help="--entry: contact depth = hand-off standoff_mm + this (default 0). Contact is NOT "
                              "sensed: its accuracy is the static camera's standoff accuracy")
    parser.add_argument("--max-standoff-mm", type=float, default=None,
                         help="--entry: refuse a hand-off standoff_mm above this (default 15, max 25). Contact is not "
                              "sensed, so a wrong standoff plunges the blade before any measurement")
    parser.add_argument("--backoff-mm", type=float, default=None,
                         help="--entry retry: back off this far before the lateral fix (default 1.0; 0 = shift "
                              "while touching)")
    parser.add_argument("--calibrate-rim-contact", action="store_true",
                         help="no motion: with the tip at contact and exactly on the rim (operator confirms), "
                              "measure the median e_lat (standoff 0, trim 0) and print the target.offset_mm that "
                              "makes it 0 (config offset + e_lat); does not write the yaml")
    parser.add_argument("--entry-tol-mm", type=float, default=None,
                         help="--entry: accepted |e_lat| after the insertion (default 0.5)")
    parser.add_argument("--entry-offset-mm", type=float, default=None,
                         help="--entry: target = rim + this offset (default 0.0 = on the rim); applied as the "
                              "offset trim entry_offset - config offset_mm")
    parser.add_argument("--insert-speed-mm-s", type=float, default=None,
                         help="--entry: insert / back-out speed (default 2.0, max 5)")
    parser.add_argument("--max-retries", type=int, default=None,
                         help="--entry: extra back-out/lateral-fix/re-insert cycles when out of tolerance (default 2)")
    parser.add_argument("--entry-record-path", default=None,
                         help=f"--entry: result record path (default {entry_mod.DEFAULT_RECORD_PATH})")
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

    args.handoff_rec = None
    if args.handoff is not None:
        try:
            if args.cfg is None:
                args.cfg = load_config(args.config_path)
            rec = handoff_mod.load_handoff(args.handoff)
        except (ConfigError, OSError, ValueError) as exc:
            parser.error(f"cannot use --handoff: {exc}")
        args.handoff_rec = rec
        if not args.entry:  # the cut flows below are unchanged; --entry takes what it needs from the record
            if rec.variant in ("in_material", "standoff"):
                rec_mode = handoff_mod.mode_for_variant(rec.variant)
                if args.mode is not None and args.mode != rec_mode:
                    parser.error(f"--mode {args.mode} conflicts with hand-off variant {rec.variant!r} "
                                 f"(requires {rec_mode})")
                args.mode = rec_mode
                if args.standoff_mm is None:
                    args.standoff_mm = 0.0 if rec.variant == "in_material" else float(rec.standoff_mm)
            # an invalid variant is reported by check_handoff after TF lookup
            if args.cut_length_mm is None:
                args.cut_length_mm = rec.cut_length_mm
            elif args.cut_length_mm > rec.cut_length_mm:
                print(f"[sequencer] note: --cut-length-mm {args.cut_length_mm:.2f} > hand-off "
                      f"{rec.cut_length_mm:.2f}; using the smaller {rec.cut_length_mm:.2f} mm")
                args.cut_length_mm = rec.cut_length_mm
            else:
                print(f"[sequencer] note: --cut-length-mm {args.cut_length_mm:.2f} < hand-off "
                      f"{rec.cut_length_mm:.2f}; using the smaller {args.cut_length_mm:.2f} mm")

    err = entry_mod.check_entry_args(args) or entry_mod.check_calibrate_args(args)
    if err:
        parser.error(err)
    if args.calibrate_rim_contact:
        args.mode = "air_align"  # nothing runs through it; standoff 0 / trim 0 below
        args.standoff_mm = 0.0
        args.cut_length_mm = 0.0
        try:
            args.cfg = load_config(args.config_path)
        except (ConfigError, OSError) as exc:
            parser.error(f"--calibrate-rim-contact needs the config ({exc})")
    if args.entry:
        entry_mod.apply_entry_defaults(args)
        if args.entry_record_path is None:
            args.entry_record_path = entry_mod.DEFAULT_RECORD_PATH
        args.mode = "air_align"
        args.cut_length_mm = 0.0  # never cuts
        args.standoff_mm = float(args.handoff_rec.standoff_mm)
        try:
            _entry_params(args).validate()
        except ValueError as exc:
            parser.error(f"--entry parameters: {exc}")
        print(f"[sequencer] ENTRY mode: standoff_mm={args.standoff_mm:.3f} from the hand-off record "
              f"(record penetration_mm={args.handoff_rec.penetration_mm:.2f} is the executor's cut depth; "
              f"this run uses --penetration-mm {args.penetration_mm:.2f}); contact is NOT sensed: it is that "
              f"standoff + margin {args.contact_margin_mm:+.2f} mm, as accurate as the static camera")

    if args.align_only:
        if args.mode not in (None, "air_align"):
            parser.error("--align-only only works with --mode air_align")
        args.mode = "air_align"
        if args.cut_length_mm is None:
            args.cut_length_mm = 0.0
    if not args.measure_only and not args.calibrate_rim_contact and args.cut_length_mm is None:
        parser.error("--cut-length-mm is required unless --measure-only or --align-only is given")

    mode = _resolve_mode(args)
    err = hold_offset.check_hold_args(args.hold_start_offset, mode, args.measure_only, args.align_only)
    if err:
        parser.error(err)
    err = control_math.check_line_hold_args(args.line_hold, args.line_vision_gain, args.measure_only,
                                             args.align_only)
    if err:
        parser.error(err)
    args.hold_trim_mm = 0.0
    if args.hold_start_offset and args.cfg is None:
        try:
            args.cfg = load_config(args.config_path)
        except (ConfigError, OSError) as exc:
            parser.error(f"--hold-start-offset needs the config ({exc})")
    if args.measure_only or args.calibrate_rim_contact:
        print("[sequencer] " + ("--measure-only" if args.measure_only else "--calibrate-rim-contact")
              + ": no service calls, no motion.")
    elif args.entry:
        print(_PRECONDITION_ENTRY)
    elif mode == "in_material":
        print(_PRECONDITION_IN_MATERIAL)
    else:
        print(_PRECONDITION_AIR_ALIGN)

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = _Sequencer()
    aborted = {"flag": False}

    def _entry_fail_record(reason: str) -> None:
        """--entry: every failure leaves a status=failed record (never a stale earlier result)."""
        if not args.entry:
            return
        record = entry_mod.build_record(
            status="failed", reason=reason, entry_offset_mm=args.entry_offset_mm,
            penetration_mm=args.penetration_mm, standoff_mm=args.standoff_mm, tol_mm=args.entry_tol_mm,
            insert_speed_mm_s=args.insert_speed_mm_s, iterations=0, max_retries=args.max_retries,
            handoff_rec=args.handoff_rec, base_frame=args.cfg.robot.base_frame,
            pre_contact_mm=args.pre_contact_mm, contact_margin_mm=args.contact_margin_mm,
            backoff_mm=args.backoff_mm)
        try:
            print(f"[sequencer] entry record written: {entry_mod.write_record_atomic(args.entry_record_path, record)}")
        except (OSError, ValueError) as exc:
            print(f"[sequencer] ERROR: could not write the entry record: {exc}")

    if args.handoff_rec is not None and not _handoff_check(node, args):
        _entry_fail_record("handoff_check_failed")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Parallax correction: tell perception how far the tip is in front of the face.
    standoff = 0.0 if args.entry else args.standoff_mm  # entry: vision only at contact (standoff 0)
    if standoff is None:
        if mode == "in_material" and not args.align_only:
            standoff = 0.0
        else:
            try:
                standoff = float(load_config(args.config_path).target.align_standoff_mm)
            except (ConfigError, OSError) as exc:
                print(f"[sequencer] WARNING: cannot read align_standoff_mm ({exc}); using 0")
                standoff = 0.0
    node.pub_standoff.publish(Float64(data=float(standoff)))
    t_pub = time.monotonic()
    while time.monotonic() - t_pub < 1.0:  # let perception apply it before the first decision
        rclpy.spin_once(node, timeout_sec=0.05)
    print(f"[sequencer] standoff {standoff:.1f} mm (parallax correction in perception)")
    args.standoff_used = standoff
    # Offset trim is latched and persists in perception: always publish it explicitly (0 = default behaviour).
    if args.entry:
        # target = rim + entry_offset: trim = entry_offset - config offset_mm (entry.py), published explicitly
        # (never rely on the config offset); effective = entry_offset + parallax * standoff.
        trim = entry_mod.trim_for_entry(args.entry_offset_mm, args.cfg.target.offset_mm)
        _publish_trim(node, trim, settle_s=0.5)
        eff = entry_mod.effective_offset_mm(args.cfg.target.offset_mm, args.cfg.target.parallax_mm_per_mm,
                                            0.0, trim)
        print(f"[sequencer] offset trim {trim:+.3f} mm (published) = entry offset {args.entry_offset_mm:+.3f} - "
              f"config offset_mm {args.cfg.target.offset_mm:+.3f}; effective offset at contact (standoff 0) = "
              f"{eff:+.3f} mm = the target is the rim + entry offset")
    else:
        _publish_trim(node, 0.0, settle_s=0.5)
        print("[sequencer] offset trim 0.000 mm (published)")

    def _on_signal(signum, frame):  # noqa: ARG001
        aborted["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    line_hold_set = False
    if args.line_hold:
        want = control_math.line_hold_enable_params(0.0 if args.line_vision_gain is None
                                                    else args.line_vision_gain)
        line_hold_set = True  # from here on, always try to restore (a partial set must not stick)
        ok, msg = _set_and_verify_controller_params(node, want)
        if not ok:
            print(f"[sequencer] FAILED: could not set {want} on /side_tape_controller ({msg}); "
                  f"nothing was commanded")
            _set_controller_params(node, control_math.LINE_HOLD_RESTORE_PARAMS)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        print(f"[sequencer] LINE-HOLD cut: controller parameters {want} {msg} (straight line from TF; vision "
              f"monitors, nudge <= 1.5 mm)")
    elif not args.measure_only and not args.calibrate_rim_contact:
        # Default cut mode stated explicitly and verified, never assumed from a previous run's leftovers.
        want = dict(control_math.CUT_MODE_VISION_PARAMS)
        ok, msg = _set_and_verify_controller_params(node, want)
        if not ok:
            print(f"[sequencer] FAILED: could not set {want} on /side_tape_controller ({msg}); "
                  f"nothing was commanded")
            _entry_fail_record(f"cut_mode_not_verified:{msg}")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        print(f"[sequencer] cut_mode=vision set and verified ({msg})")

    exit_code = 1
    try:
        if args.measure_only:
            exit_code = _run_measure_only(node, args, aborted)
        elif args.calibrate_rim_contact:
            exit_code = _run_calibrate_rim_contact(node, args, aborted)
        elif args.entry:
            exit_code = _run_entry(node, args, aborted)
        elif mode == "in_material":
            exit_code = _run_in_material(node, args, aborted)
        else:
            exit_code = _run_air_align(node, args, aborted)
    finally:
        if aborted["flag"] and not args.measure_only and not args.calibrate_rim_contact and not args.entry:
            print("[sequencer] aborting...")
            node.call_abort()
        if line_hold_set:
            ok, msg = _set_and_verify_controller_params(node, control_math.LINE_HOLD_RESTORE_PARAMS)
            print(f"[sequencer] restored cut_mode=vision -> {'ok' if ok else 'FAILED: ' + msg}")
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
