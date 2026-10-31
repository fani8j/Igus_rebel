#!/usr/bin/env python3
"""Supervised, recorded A/B test for the axis-2 overcurrent: robot-side Move Joint vs JTC.

Runs ONE joint-space move from --start-deg to --goal-deg, either as the robot's own CRI
"Move Joint" (--mode movejoint, the driver's move_joint action) or through the
JointTrajectoryController (--mode jtc, quintic trajectory), and records a rosbag2 of the driver
diagnostics (~/cri_diagnostics: raw joint currents + the jog percent actually sent) around it.
Analyse the bags with tools/a2_ab_analyze.py.

Every motion needs a typed "yes". Use it only with an operator at the E-stop. The tool never
switches controllers: if the wrong controller state is active it prints the exact command and exits.

    # move the arm to the start pose first (e.g. with the teleop), then:
    python3 tools/a2_ab_test.py --mode jtc --start-deg 0 -30 60 0 45 0 \\
        --goal-deg 0 -10 60 0 45 0 --duration 8 --label jtc_base
    python3 tools/a2_ab_test.py --mode jtc ... --deadband 0.5 --label jtc_db05
    # JTC deactivated (printed command), then:
    python3 tools/a2_ab_test.py --mode movejoint ... --label movejoint

Limits (refused otherwise): |goal - start| <= 40 deg per joint; the arm within 2 deg of --start-deg;
JTC duration >= 4 s and planned peak joint velocity <= 0.2 rad/s; move_joint velocity and
acceleration <= 25 %.
"""

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import sys
import time

try:  # the pure functions below must stay importable (and testable) without ROS
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
except ImportError:  # pragma: no cover - only without a sourced ROS
    rclpy = None
    Node = object

N_JOINTS = 6
MAX_DELTA_DEG = 40.0
MAX_START_ERROR_DEG = 2.0
MIN_JTC_DURATION_S = 4.0
MAX_JTC_VELOCITY_RAD_S = 0.2
MAX_MOVEJOINT_PERCENT = 25.0
TRAJECTORY_POINTS = 50
PRE_ROLL_S = 2.0
POST_ROLL_S = 3.0
BAG_STARTUP_S = 1.5  # discovery time of ros2 bag record before the pre-roll starts
QUINTIC_PEAK_FACTOR = 1.875  # peak of ds/dtau of s = 10t^3 - 15t^4 + 6t^5, at tau = 0.5

DEFAULT_DRIVER_NODE = "igus_rebel"
DEFAULT_CONTROLLER = "rebel_arm_trajectory_controller"
DEFAULT_CONTROLLER_MANAGER = "/controller_manager"
DEADBAND_PARAMETER = "jtc_jog_deadband_percent"


# ----------------------------------------------------------------------------------------------
# Pure functions (unit-tested in tools/test_a2_ab.py)
# ----------------------------------------------------------------------------------------------
def quintic_trajectory(start_rad, goal_rad, duration_s, n_points=TRAJECTORY_POINTS):
    """Smooth joint-space trajectory with zero velocity and acceleration at both ends.

    Returns n_points tuples (t, positions, velocities, accelerations), t = duration * k / n for
    k = 1..n (the last point is the goal at t = duration; t = 0 is the current state).
    """
    if len(start_rad) != len(goal_rad):
        raise ValueError("start and goal need the same number of joints")
    if not duration_s > 0.0 or n_points < 2:
        raise ValueError("duration must be positive and n_points at least 2")
    points = []
    for k in range(1, n_points + 1):
        t = duration_s * k / n_points
        tau = t / duration_s
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / duration_s
        dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / duration_s ** 2
        positions, velocities, accelerations = [], [], []
        for q0, q1 in zip(start_rad, goal_rad):
            delta = q1 - q0
            positions.append(q0 + delta * s)
            velocities.append(delta * ds)
            accelerations.append(delta * dds)
        points.append((t, positions, velocities, accelerations))
    return points


def planned_peak_velocity(start_rad, goal_rad, duration_s, points=None):
    """Largest |velocity| [rad/s] of the quintic move over all joints.

    The analytic peak (1.875 * delta / T, at the middle of the move) and the largest value of the
    given trajectory points, whichever is larger.
    """
    delta = max((abs(q1 - q0) for q0, q1 in zip(start_rad, goal_rad)), default=0.0)
    peak = QUINTIC_PEAK_FACTOR * delta / duration_s
    for _, _, velocities, _ in points or []:
        peak = max(peak, max(abs(v) for v in velocities))
    return peak


def validate_args(start_deg, goal_deg, mode, duration_s, velocity_percent, acceleration_percent,
                  deadband=None):
    """Static limit check of the command line. Returns a list of refusal reasons (empty = ok)."""
    errors = []
    for name, values in (("--start-deg", start_deg), ("--goal-deg", goal_deg)):
        if len(values) != N_JOINTS:
            errors.append(f"{name} needs exactly {N_JOINTS} values")
        elif not all(math.isfinite(v) for v in values):
            errors.append(f"{name} must be finite")
    if errors:
        return errors
    for i, (a, b) in enumerate(zip(start_deg, goal_deg)):
        if abs(b - a) > MAX_DELTA_DEG:
            errors.append(f"joint{i + 1}: |goal - start| = {abs(b - a):.1f} deg > {MAX_DELTA_DEG:.0f}")
    if all(a == b for a, b in zip(start_deg, goal_deg)):
        errors.append("start and goal are identical: nothing to move")
    if mode == "jtc":
        if not math.isfinite(duration_s) or duration_s < MIN_JTC_DURATION_S:
            errors.append(f"--duration must be at least {MIN_JTC_DURATION_S:.0f} s for --mode jtc")
        else:
            peak = planned_peak_velocity(
                [math.radians(v) for v in start_deg], [math.radians(v) for v in goal_deg],
                duration_s)
            if peak > MAX_JTC_VELOCITY_RAD_S:
                errors.append(
                    f"planned peak joint velocity {peak:.3f} rad/s > {MAX_JTC_VELOCITY_RAD_S} rad/s"
                    " (use a longer --duration or a smaller move)")
    elif mode == "movejoint":
        for name, value in (("--velocity-percent", velocity_percent),
                            ("--acceleration-percent", acceleration_percent)):
            if not math.isfinite(value) or not 0.0 < value <= MAX_MOVEJOINT_PERCENT:
                errors.append(f"{name} must be in (0, {MAX_MOVEJOINT_PERCENT:.0f}]")
    else:
        errors.append(f"unknown --mode {mode!r}")
    if deadband is not None:
        if mode != "jtc":
            errors.append("--deadband only applies to --mode jtc")
        elif not math.isfinite(deadband) or not 0.0 <= deadband <= 2.0:
            errors.append("--deadband must be in [0, 2] %")
    return errors


def start_pose_errors(current_deg, start_deg, tolerance_deg=MAX_START_ERROR_DEG):
    """Reasons to refuse because the arm is not at --start-deg (empty = ok)."""
    errors = []
    for i, (cur, start) in enumerate(zip(current_deg, start_deg)):
        if not math.isfinite(cur) or abs(cur - start) > tolerance_deg:
            errors.append(
                f"joint{i + 1} is at {cur:.2f} deg, {abs(cur - start):.2f} deg from --start-deg "
                f"{start:.2f} (max {tolerance_deg:.0f}); move the arm there first")
    return errors


def hardware_refusals(hardware_state, mode, controller_state, controller_name):
    """Reasons to refuse from HardwareState (any object with its fields) and the JTC state.

    controller_state is the controller_manager state string of the JTC ('active', 'inactive', ...)
    or None if it is not listed.
    """
    errors = []
    if hardware_state is None:
        return ["no ~/hardware_state received from the driver"]
    if hardware_state.latched:
        errors.append(f"the driver is LATCHED ({hardware_state.latch_reason!r}); /joint_states is frozen")
    if not hardware_state.connected:
        errors.append("the driver is not connected to the robot")
    if not hardware_state.activated:
        errors.append("the hardware component is not active")
    if not hardware_state.status_fresh:
        errors.append("the driver's STATUS is stale")
    if getattr(hardware_state, "move_active", False):
        errors.append("a move_joint goal is already active")
    if getattr(hardware_state, "cart_jog_active", False) or getattr(
            hardware_state, "joint_jog_active", False):
        errors.append("a CRI jog mode (cart_jog / joint_jog) is on; switch it off first")
    if mode == "jtc" and controller_state != "active":
        errors.append(f"controller {controller_name} is not active (state {controller_state})")
    if mode == "movejoint" and controller_state == "active":
        errors.append(
            f"controller {controller_name} is active: the driver refuses move_joint while a "
            "controller claims the joints")
    return errors


def controller_state_wrong(mode, controller_state):
    """True if the JTC must be switched before this mode can run (jtc: not active; movejoint: active)."""
    return controller_state != "active" if mode == "jtc" else controller_state == "active"


def switch_controllers_command(mode, controller_name, controller_manager):
    """The exact `ros2 control switch_controllers` command for the state --mode needs."""
    verb = "--activate" if mode == "jtc" else "--deactivate"
    return (f"ros2 control switch_controllers -c {controller_manager} {verb} {controller_name}")


def bag_record_command(bag_dir, topics, distro=None):
    """`ros2 bag record` command line. Humble takes the topics positionally (it has no --topics);
    newer distros take --topics."""
    distro = os.environ.get("ROS_DISTRO", "humble") if distro is None else distro
    flag = [] if distro in ("foxy", "galactic", "humble") else ["--topics"]
    return ["ros2", "bag", "record", "-o", bag_dir] + flag + list(topics)


def default_topics(driver_node, controller):
    return [
        "/joint_states",
        f"/{driver_node}/cri_diagnostics",
        f"/{driver_node}/hardware_state",
        f"/{driver_node}/robot_joint_states",
        f"/{controller}/controller_state",
        "/dynamic_joint_states",
        "/rosout",
    ]


# ----------------------------------------------------------------------------------------------
# ROS side
# ----------------------------------------------------------------------------------------------
def _stamp_ns(node):
    return node.get_clock().now().nanoseconds


class AbNode(Node):
    def __init__(self, driver_node, controller, controller_manager, joints):
        super().__init__("a2_ab_test")
        from control_msgs.action import FollowJointTrajectory
        from igus_rebel_msgs.action import MoveJoint
        from igus_rebel_msgs.msg import HardwareState
        from sensor_msgs.msg import JointState
        self._FJT = FollowJointTrajectory
        self._MoveJoint = MoveJoint
        self.driver = driver_node
        self.controller = controller
        self.cm = controller_manager
        self.joints = joints
        self.state = None
        self.hw = None
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.create_subscription(
            HardwareState, f"/{driver_node}/hardware_state",
            self._on_hw, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.jtc_client = ActionClient(
            self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        self.move_client = ActionClient(self, MoveJoint, f"/{driver_node}/move_joint")
        self.move_progress = None

    def _on_state(self, msg):
        self.state = msg

    def _on_hw(self, msg):
        self.hw = msg

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, attribute, timeout_s=5.0):
        end = time.monotonic() + timeout_s
        setattr(self, attribute, None)
        while getattr(self, attribute) is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return getattr(self, attribute)

    def joint_positions_deg(self):
        """Live joint positions in degrees from a fresh /joint_states message."""
        msg = self.wait_for("state")
        if msg is None:
            raise RuntimeError("no /joint_states received (same ROS_DOMAIN_ID as the driver?)")
        by_name = dict(zip(msg.name, msg.position))
        missing = [j for j in self.joints if j not in by_name]
        if missing:
            raise RuntimeError(f"/joint_states is missing {missing}")
        return [math.degrees(by_name[j]) for j in self.joints]

    def controller_state(self):
        """controller_manager state of the JTC ('active', ...), None if not listed."""
        from controller_manager_msgs.srv import ListControllers
        client = self.create_client(ListControllers, f"{self.cm}/list_controllers")
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{self.cm}/list_controllers not available")
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done():
                raise RuntimeError("list_controllers timed out")
            for controller in future.result().controller:
                if controller.name == self.controller:
                    return controller.state
            return None
        finally:
            self.destroy_client(client)

    def get_deadband(self):
        from rcl_interfaces.srv import GetParameters
        client = self.create_client(GetParameters, f"/{self.driver}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"/{self.driver}/get_parameters not available")
            request = GetParameters.Request()
            request.names = [DEADBAND_PARAMETER]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or not future.result().values:
                raise RuntimeError("get_parameters failed")
            value = future.result().values[0]
            if value.type != 3:
                raise RuntimeError(
                    f"driver has no {DEADBAND_PARAMETER} parameter (old driver build?)")
            return value.double_value
        finally:
            self.destroy_client(client)

    def set_deadband(self, value):
        from rcl_interfaces.msg import Parameter, ParameterValue
        from rcl_interfaces.srv import SetParameters
        client = self.create_client(SetParameters, f"/{self.driver}/set_parameters")
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"/{self.driver}/set_parameters not available")
            request = SetParameters.Request()
            request.parameters = [Parameter(
                name=DEADBAND_PARAMETER,
                value=ParameterValue(type=3, double_value=float(value)))]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or not future.result().results[0].successful:
                reason = future.result().results[0].reason if future.done() else "timeout"
                raise RuntimeError(f"setting {DEADBAND_PARAMETER}={value} failed: {reason}")
        finally:
            self.destroy_client(client)

    # ---- the two motions; both return a result dict and never raise for a failed motion ----
    def _wait_result(self, handle, timeout_s):
        """Waits for the result; on Ctrl+C or timeout cancels the goal and says so."""
        result_future = handle.get_result_async()
        aborted = None
        try:
            end = time.monotonic() + timeout_s
            while not result_future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            if not result_future.done():
                aborted = "timeout"
        except KeyboardInterrupt:
            aborted = "operator Ctrl+C"
        if aborted:
            print(f"Aborting the goal ({aborted}) ...")
            cancel = handle.cancel_goal_async()
            end = time.monotonic() + 5.0
            while not (cancel.done() and result_future.done()) and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
        return result_future, aborted

    def run_jtc(self, points, duration_s):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint
        if not self.jtc_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("follow_joint_trajectory action server not available")
        goal = self._FJT.Goal()
        goal.trajectory.joint_names = list(self.joints)
        for t, positions, velocities, accelerations in points:
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.velocities = list(velocities)
            point.accelerations = list(accelerations)
            point.time_from_start = Duration(sec=int(t), nanosec=int(round((t % 1.0) * 1e9)))
            goal.trajectory.points.append(point)
        send = self.jtc_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return {"accepted": False, "error": "goal rejected by the trajectory controller"}
        future, aborted = self._wait_result(handle, duration_s + 15.0)
        out = {"accepted": True, "aborted": aborted}
        if future.done():
            result = future.result().result
            out.update(error_code=int(result.error_code), error_string=result.error_string,
                       success=result.error_code == self._FJT.Result.SUCCESSFUL)
        else:
            out.update(success=False, error_string="no result")
        return out

    def run_move_joint(self, start_rad, goal_rad, velocity, acceleration):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("move_joint action server not available")
        goal = self._MoveJoint.Goal()
        goal.joint_names = list(self.joints)
        goal.start_positions = list(start_rad)
        goal.goal_positions = list(goal_rad)
        goal.velocity_percent = float(velocity)
        goal.acceleration_percent = float(acceleration)
        goal.start_tolerance = math.radians(MAX_START_ERROR_DEG + 0.5)
        goal.path_tolerance = 0.1
        goal.goal_tolerance = math.radians(2.0)
        goal.timeout_s = 0.0

        def feedback(msg):
            self.move_progress = msg.feedback.progress

        send = self.move_client.send_goal_async(goal, feedback_callback=feedback)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return {"accepted": False, "error": "goal rejected by the driver"}
        future, aborted = self._wait_result(handle, 300.0)
        out = {"accepted": True, "aborted": aborted}
        if future.done():
            result = future.result().result
            out.update(
                error_code=int(result.error_code), message=result.message,
                end_reason=result.end_reason, stop_sent=bool(result.stop_sent),
                stop_confirmed=bool(result.stop_confirmed),
                final_positions_deg=[math.degrees(p) for p in result.final_positions],
                success=result.error_code == self._MoveJoint.Result.SUCCESS)
        else:
            out.update(success=False, message="no result")
        return out


class BagRecorder:
    """ros2 bag record in its own process group; only that group is ever signalled."""

    def __init__(self, bag_dir, topics):
        self.bag_dir = bag_dir
        self.topics = topics
        self.proc = None
        self.pid = None

    def start(self):
        self.proc = subprocess.Popen(
            bag_record_command(self.bag_dir, self.topics), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pid = self.proc.pid  # session leader: its pid is the process group id

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            for sig, wait_s in ((signal.SIGINT, 20.0), (signal.SIGTERM, 5.0)):
                try:
                    os.killpg(self.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None


def confirm(prompt):
    try:
        return input(f"{prompt} Type 'yes' to continue: ").strip().lower() == "yes"
    except EOFError:
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]))
    parser.add_argument("--start-deg", type=float, nargs=N_JOINTS, required=True)
    parser.add_argument("--goal-deg", type=float, nargs=N_JOINTS, required=True)
    parser.add_argument("--mode", choices=("movejoint", "jtc"), required=True)
    parser.add_argument("--duration", type=float, default=8.0, help="JTC trajectory duration [s]")
    parser.add_argument("--velocity-percent", type=float, default=10.0)
    parser.add_argument("--acceleration-percent", type=float, default=10.0)
    parser.add_argument("--deadband", type=float, default=None,
                        help="set the driver's jtc_jog_deadband_percent for a JTC run, restore after")
    parser.add_argument("--out", default=None,
                        help="output directory (default scan_results/a2_ab_<date>/)")
    parser.add_argument("--label", default="run")
    parser.add_argument("--driver-node", default=DEFAULT_DRIVER_NODE)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--controller-manager", default=DEFAULT_CONTROLLER_MANAGER)
    parser.add_argument("--joint-prefix", default="", help="prefix of the joint names (dual-arm)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the arguments, print the plan and exit (no ROS)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    errors = validate_args(args.start_deg, args.goal_deg, args.mode, args.duration,
                           args.velocity_percent, args.acceleration_percent, args.deadband)
    if errors:
        print("Refused:\n  " + "\n  ".join(errors))
        return 1
    start_rad = [math.radians(v) for v in args.start_deg]
    goal_rad = [math.radians(v) for v in args.goal_deg]
    points = quintic_trajectory(start_rad, goal_rad, args.duration) if args.mode == "jtc" else None
    peak = planned_peak_velocity(start_rad, goal_rad, args.duration, points) if points else None
    plan = (f"mode {args.mode}: {args.start_deg} -> {args.goal_deg} deg; "
            + (f"JTC {args.duration:.1f} s, planned peak {peak:.3f} rad/s, {len(points)} points"
               if points else f"move_joint velocity {args.velocity_percent:g} % "
                              f"acceleration {args.acceleration_percent:g} %"))
    if args.dry_run:
        print("Dry run OK: " + plan)
        return 0
    if rclpy is None:
        print("rclpy not available: source /opt/ros/humble/setup.bash and the workspace")
        return 3
    if not sys.stdin.isatty():
        print("refusing to run without an interactive terminal (confirmation required)")
        return 3

    date = datetime.date.today().strftime("%Y%m%d")
    out_dir = args.out or f"/home/farhan/igus_rebel_ros2/scan_results/a2_ab_{date}/"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_name = f"{args.label}_{args.mode}_{datetime.datetime.now().strftime('%H%M%S')}"
    bag_dir = os.path.join(out_dir, run_name)
    summary_path = bag_dir + ".summary.json"
    joints = [f"{args.joint_prefix}joint{i}" for i in range(1, N_JOINTS + 1)]

    rclpy.init()
    node = AbNode(args.driver_node, args.controller, args.controller_manager, joints)
    recorder = BagRecorder(bag_dir, default_topics(args.driver_node, args.controller))
    summary = {"args": vars(args), "label": args.label, "mode": args.mode, "bag": bag_dir,
               "start_deg": args.start_deg, "goal_deg": args.goal_deg, "plan": plan}
    old_deadband = None
    deadband_changed = False
    code = 0
    try:
        current = node.joint_positions_deg()
        hw = node.wait_for("hw")
        controller_state = node.controller_state()
        refusals = hardware_refusals(hw, args.mode, controller_state, args.controller)
        if controller_state_wrong(args.mode, controller_state):
            print("Controller state wrong for this mode. Run this, then start the tool again:\n  "
                  + switch_controllers_command(args.mode, args.controller, args.controller_manager))
        refusals += start_pose_errors(current, args.start_deg)
        if refusals:
            print("Refused:\n  " + "\n  ".join(refusals))
            return 1
        print("Current joints [deg]: " + " ".join(f"{v:8.2f}" for v in current))
        print("Plan: " + plan)
        if args.deadband is not None:
            print(f"Driver {DEADBAND_PARAMETER} will be set to {args.deadband:g} % for this run "
                  "and restored afterwards.")
        print(f"Recording a rosbag to {bag_dir} (2 s before, 3 s after the motion).")
        if not confirm("E-stop in hand and the path clear?"):
            print("Aborted, nothing sent.")
            return 1

        if args.deadband is not None:
            old_deadband = node.get_deadband()
            node.set_deadband(args.deadband)
            deadband_changed = True
            summary["deadband_percent"] = args.deadband
            summary["deadband_previous_percent"] = old_deadband
        else:
            try:
                summary["deadband_percent"] = node.get_deadband()
            except RuntimeError:
                summary["deadband_percent"] = None

        recorder.start()
        node.spin_for(BAG_STARTUP_S + PRE_ROLL_S)
        if not recorder.alive():
            raise RuntimeError("ros2 bag record exited early; nothing was moved")
        summary["state_before_deg"] = node.joint_positions_deg()
        summary["motion_start_ns"] = _stamp_ns(node)
        try:
            if args.mode == "jtc":
                result = node.run_jtc(points, args.duration)
            else:
                result = node.run_move_joint(start_rad, goal_rad, args.velocity_percent,
                                             args.acceleration_percent)
        except KeyboardInterrupt:
            result = {"success": False, "aborted": "operator Ctrl+C before the goal was sent"}
        summary["motion_end_ns"] = _stamp_ns(node)
        summary["result"] = result
        print("Result: " + json.dumps(result))
        node.spin_for(POST_ROLL_S)
        summary["state_after_deg"] = node.joint_positions_deg()
        summary["hardware_state_after"] = {
            "latched": bool(node.hw.latched), "latch_reason": node.hw.latch_reason,
            "drives_disabled_mask": int(node.hw.drives_disabled_mask)} if node.hw else None
        print("Final joints [deg]: " + " ".join(f"{v:8.2f}" for v in summary["state_after_deg"]))
        if not result.get("success"):
            print("Motion did NOT succeed; check the driver log and the bag.")
            code = 2
    except RuntimeError as exc:
        print(f"Error: {exc}")
        summary["error"] = str(exc)
        code = 3
    except KeyboardInterrupt:
        print("Interrupted.")
        summary["error"] = "interrupted"
        code = 3
    finally:
        try:
            if recorder.alive():
                try:
                    node.spin_for(POST_ROLL_S if "motion_start_ns" in summary
                                  and "motion_end_ns" not in summary else 0.0)
                except KeyboardInterrupt:
                    pass
            recorder.stop()
        finally:
            if deadband_changed and old_deadband is not None:
                try:
                    node.set_deadband(old_deadband)
                    print(f"Restored {DEADBAND_PARAMETER} = {old_deadband:g}")
                except RuntimeError as exc:
                    print(f"WARNING: could not restore {DEADBAND_PARAMETER}={old_deadband}: {exc}")
                    summary["deadband_restore_failed"] = str(exc)
            if "motion_start_ns" in summary:
                with open(summary_path, "w") as handle:
                    json.dump(summary, handle, indent=2, default=str)
                print(f"Summary: {summary_path}")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
