#!/usr/bin/env python3
"""Operator CLI: cut with the RIGHT ReBeL alone (single-arm stack, controller 192.168.3.12).

  right_cut.py plan face      right face of the carton   (executor cut_pattern left_only)
  right_cut.py plan top       top seam from the RIGHT END, as far as the arm reaches (5..100 %)
  right_cut.py execute        dispatch the retained plan (identity gate, typed `yes` on real HW)
  right_cut.py disarm         set execution_enabled false

`plan` converts the newest perception snapshot (left base frame) into the right arm's base frame
(right_snapshot.py), points the executor at the converted file, sets the cut parameters and calls
/box_h_cut_executor/plan_h_cut. Nothing moves until `execute`.

Uses the caller's ROS environment (ROS_DOMAIN_ID must be set; the operator sets the real cell's).
"""

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import right_snapshot as rs  # noqa: E402

EXECUTOR = "/box_h_cut_executor"
EXECUTOR_NAME = "box_h_cut_executor"
RIGHT_CONTROLLER_IP = "192.168.3.12"
MOCK_PLUGIN = "mock_components/GenericSystem"
REAL_PLUGIN = "Igus/Rebel"
DEFAULT_SOURCE = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"
DEFAULT_OUTPUT = "/home/farhan/Igus_rebel/outputs/seam_3d_right_base_link.json"
TOP_MIN_FRACTION = 0.05
DONE_TEXT = "H-cut trajectory execution completed"
FAIL_TEXTS = ("H-cut trajectory execution failed", "H-cut trajectory execution threw exception")

# Right face: how far the whole stroke goes INTO the detected face (executor left_cut_inward_m,
# [0, 8] mm). Operator 2026-10-15: the left arm's 5 mm (YAML) and then 2 mm were too deep on the right arm.
RIGHT_FACE_INWARD_MM = 1.0

CUTS = {
    "face": {"cut_pattern": "left_only", "middle_seam_stop": "end"},
    # Operator 2026-10-15: the right arm keeps whatever it reaches from the right end (the planner
    # pulls the far end back in 5 % steps), down to 5 % of the seam. The half-seam mode
    # (middle_seam_stop center) did not plan at most carton placements on the cell.
    "top": {"cut_pattern": "middle_only", "middle_seam_stop": "end",
            "top_cut_min_cut_fraction": TOP_MIN_FRACTION},
}


class CutError(Exception):
    pass


# ------------------------------------------------------------------ pure helpers (unit-tested)
def parse_hardware(urdf_xml):
    """(plugin, robot_ip or None) of the single ros2_control hardware in a URDF string."""
    try:
        root = ET.fromstring(urdf_xml)
    except ET.ParseError as exc:
        raise CutError(f"robot_description is not valid XML: {exc}") from exc
    plugins = []
    ip = None
    for control in root.iter("ros2_control"):
        hardware = control.find("hardware")
        if hardware is None:
            continue
        plugin = hardware.findtext("plugin")
        if plugin:
            plugins.append(plugin.strip())
        for param in hardware.findall("param"):
            if param.get("name") == "robot_ip" and param.text:
                ip = param.text.strip()
    if len(plugins) != 1:
        raise CutError(f"expected exactly one ros2_control hardware plugin, found {plugins}")
    return plugins[0], ip


def identity_verdict(plugin, robot_ip, class_types=None):
    """Return ('mock'|'real', text) when allowed, else raise CutError.

    Allowed: mock_components/GenericSystem, or the Igus/Rebel plugin with robot_ip 192.168.3.12.
    `class_types` (from list_hardware_components) must agree with the description."""
    if plugin == MOCK_PLUGIN:
        kind = "mock"
    elif plugin == REAL_PLUGIN:
        if robot_ip != RIGHT_CONTROLLER_IP:
            raise CutError(
                f"REFUSED: the driver's robot_ip is '{robot_ip}', not {RIGHT_CONTROLLER_IP} (the "
                "RIGHT arm). A right-frame plan must never run on another controller (the left "
                "arm is 192.168.3.11).")
        kind = "real"
    else:
        raise CutError(f"REFUSED: hardware plugin '{plugin}' is neither {MOCK_PLUGIN} nor {REAL_PLUGIN}")
    if class_types is not None and set(class_types) != {plugin}:
        raise CutError(
            f"REFUSED: loaded hardware components {sorted(class_types)} do not match the "
            f"robot_description plugin '{plugin}'")
    return kind, f"{plugin}" + (f" robot_ip={robot_ip}" if robot_ip else "")


def file_sha256(path):
    return rs.sha256_file(path)


def state_path(output):
    p = Path(output)
    return p.with_name(p.stem + ".plan_state.json")


def check_execute_preconditions(state, snapshot_param, output, params):
    """Raise CutError unless the retained plan can only come from the converted snapshot."""
    output = str(Path(output))
    if snapshot_param != output:
        raise CutError(
            f"REFUSED: the executor's external_snapshot_file is '{snapshot_param}', not the "
            f"converted right-arm snapshot '{output}'. Run `right_cut.py plan face|top` first.")
    if not state:
        raise CutError(f"REFUSED: no plan record {state_path(output)}; run `right_cut.py plan face|top`")
    if not Path(output).is_file():
        raise CutError(f"REFUSED: converted snapshot {output} is missing")
    if state.get("snapshot_sha256") != file_sha256(output):
        raise CutError("REFUSED: the converted snapshot changed after the plan was made; plan again")
    want = CUTS[state["cut"]]
    for key, value in want.items():
        have = params.get(key)
        matches = (
            isinstance(have, float) and abs(have - value) <= 1e-12 if isinstance(value, float)
            else have == value)
        if not matches:
            raise CutError(
                f"REFUSED: executor parameter {key}={have!r} does not match the '{state['cut']}' "
                f"plan ({value!r}); plan again")
    return True


# ------------------------------------------------------------------------------- ROS side
class Ros:
    def __init__(self):
        import rclpy
        from rcl_interfaces.msg import Log
        from rclpy.node import Node
        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("right_cut_cli")
        self.log_buffer = []
        self.node.create_subscription(Log, "/rosout", self._on_log, 1000)

    def _on_log(self, msg):
        self.log_buffer.append((msg.name, msg.msg))

    def spin_until(self, future, timeout):
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise CutError("service call timed out")
        return future.result()

    def client(self, srv_type, name, wait=10.0):
        client = self.node.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=wait):
            raise CutError(f"service {name} not available (is the stack running in this ROS domain?)")
        return client

    def get_parameters(self, node, names, attempts=3):
        from rcl_interfaces.srv import GetParameters
        client = self.client(GetParameters, f"{node}/get_parameters")
        request = GetParameters.Request()
        request.names = list(names)
        # Read-only, so retrying is safe. A fresh process can send its first request before the
        # server has discovered the reply reader; the server then drops the reply ("failed to send
        # response ... client will not receive response"), mostly with ROS_LOCALHOST_ONLY=0.
        for attempt in range(1, attempts + 1):
            try:
                result = self.spin_until(client.call_async(request), 10.0)
                return [self._value(v) for v in result.values]
            except CutError:
                if attempt == attempts:
                    raise
                print(f"{node}/get_parameters: no reply, retrying ({attempt}/{attempts})", flush=True)

    @staticmethod
    def _value(value):
        from rcl_interfaces.msg import ParameterType as T
        return {T.PARAMETER_BOOL: lambda v: v.bool_value, T.PARAMETER_INTEGER: lambda v: v.integer_value,
                T.PARAMETER_DOUBLE: lambda v: v.double_value, T.PARAMETER_STRING: lambda v: v.string_value,
                T.PARAMETER_NOT_SET: lambda v: None}.get(value.type, lambda v: None)(value)

    def set_parameter(self, name, value):
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        pv = ParameterValue()
        if isinstance(value, bool):
            pv.type, pv.bool_value = ParameterType.PARAMETER_BOOL, value
        elif isinstance(value, float):
            pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, value
        else:
            pv.type, pv.string_value = ParameterType.PARAMETER_STRING, str(value)
        request = SetParameters.Request()
        request.parameters = [Parameter(name=name, value=pv)]
        client = self.client(SetParameters, f"{EXECUTOR}/set_parameters")
        result = self.spin_until(client.call_async(request), 15.0).results[0]
        if not result.successful:
            raise CutError(f"setting {name}={value!r} failed: {result.reason}")

    def trigger(self, name, timeout):
        from std_srvs.srv import Trigger
        client = self.client(Trigger, name)
        return self.spin_until(client.call_async(Trigger.Request()), timeout)

    def hardware_identity(self):
        """Verdict from the controller_manager's robot_description (the driver) and, as a
        cross-check, robot_state_publisher's, plus the loaded hardware components."""
        from controller_manager_msgs.srv import ListHardwareComponents
        descriptions = {}
        for node in ("/controller_manager", "/robot_state_publisher"):
            try:
                value = self.get_parameters(node, ["robot_description"])[0]
            except CutError:
                value = None
            if value:
                descriptions[node] = parse_hardware(value)
        if "/controller_manager" not in descriptions:
            raise CutError("REFUSED: cannot read robot_description from /controller_manager "
                           "(driver not running in this ROS domain?)")
        if len(set(descriptions.values())) != 1:
            raise CutError(f"REFUSED: robot_description of the driver nodes disagree: {descriptions}")
        client = self.client(ListHardwareComponents, "/controller_manager/list_hardware_components")
        components = self.spin_until(client.call_async(ListHardwareComponents.Request()), 10.0).component
        if not components:
            raise CutError("REFUSED: controller_manager reports no hardware components")
        plugin, ip = descriptions["/controller_manager"]
        return identity_verdict(plugin, ip, [c.class_type for c in components])

    def shutdown(self):
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


def default_source():
    try:
        from ament_index_python.packages import get_package_share_directory
        import yaml
        path = Path(get_package_share_directory("igus_rebel_moveit_config")) / "config" / "box_h_cut_executor.yaml"
        value = yaml.safe_load(path.read_text())["box_h_cut_executor"]["ros__parameters"]["external_snapshot_file"]
        if value:
            return value
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_SOURCE


def require_domain():
    domain = os.environ.get("ROS_DOMAIN_ID")
    if not domain:
        raise CutError("REFUSED: ROS_DOMAIN_ID is not set. Set it explicitly (real cell: the "
                       "operator's domain; mock: an unused one with ROS_LOCALHOST_ONLY=1).")
    print(f"ROS_DOMAIN_ID={domain} ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '<unset>')}")
    return domain


# --------------------------------------------------------------------------------- commands
def cmd_plan(args, ros):
    kind, text = ros.hardware_identity()
    print(f"hardware: {kind} ({text})")
    if args.allow_stale and kind != "mock":
        raise CutError("--allow-stale is for mock only")
    source = args.source or default_source()
    output = str(Path(args.output))
    try:
        rs.convert_file(source, output, args.calib, allow_stale=args.allow_stale)
    except rs.ConversionError as exc:
        raise CutError(f"snapshot conversion failed: {exc}") from exc
    print(f"converted {source} -> {output}")
    state_file = state_path(output)
    state_file.unlink(missing_ok=True)  # a failed plan leaves no record
    wanted = dict(CUTS[args.cut])
    if args.cut == "face":
        if not 0.0 <= args.inward_mm <= 8.0:
            raise CutError(f"--inward-mm {args.inward_mm} is outside [0, 8] mm")
        wanted["left_cut_inward_m"] = args.inward_mm / 1000.0
    ros.set_parameter("external_snapshot_file", output)
    for key, value in wanted.items():
        ros.set_parameter(key, value)
    for key in args.set or []:
        name, _, raw = key.partition("=")
        ros.set_parameter(name, _typed(raw))
    print(f"executor parameters set: external_snapshot_file={output} " +
          " ".join(f"{k}={v}" for k, v in wanted.items()) + (f" {' '.join(args.set)}" if args.set else ""))
    start = time.monotonic()
    response = ros.trigger(f"{EXECUTOR}/plan_h_cut", args.timeout)
    elapsed = time.monotonic() - start
    print(f"plan_h_cut ({args.cut}) success={response.success} in {elapsed:.1f} s")
    print(f"message: {response.message}")
    if response.success:
        record = {"cut": args.cut, "snapshot": output, "snapshot_sha256": file_sha256(output),
                  "source": source, "planned_at": time.time(), "domain": os.environ["ROS_DOMAIN_ID"],
                  "message": response.message}
        state_file.write_text(json.dumps(record, indent=2) + "\n")
        print("Plan retained. Inspect it in RViz, then: right_cut.py execute")
    return 0 if response.success else 1


def _typed(raw):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return float(raw) if any(c in raw for c in ".eE") else raw
    except ValueError:
        return raw


def cmd_execute(args, ros):
    kind, text = ros.hardware_identity()
    print(f"hardware identity OK: {kind} ({text})")
    output = str(Path(args.output))
    state_file = state_path(output)
    state = json.loads(state_file.read_text()) if state_file.is_file() else None
    names = ["external_snapshot_file", "execution_enabled"] + sorted(
        {key for cut in CUTS.values() for key in cut})
    params = dict(zip(names, ros.get_parameters(EXECUTOR, names)))
    check_execute_preconditions(state, params["external_snapshot_file"], output, params)
    print(f"retained plan: {state['cut']} planned from {state['snapshot']}\n  {state['message']}")
    if kind == "real":
        print(f"\nREAL ROBOT: the RIGHT arm at {RIGHT_CONTROLLER_IP} will move ({state['cut']} cut). "
              "Keep the e-stop in reach.")
        if input("Type `yes` to execute: ").strip() != "yes":
            print("not confirmed; nothing executed")
            return 1
    armed = False
    try:
        ros.log_buffer.clear()
        ros.set_parameter("execution_enabled", True)
        armed = True
        response = ros.trigger(f"{EXECUTOR}/execute_planned_h_cut", 60.0)
        print(f"execute_planned_h_cut: success={response.success} message={response.message}")
        if not response.success:
            return 1
        print(f"dispatched; waiting up to {args.timeout:.0f} s for '{DONE_TEXT}' ...")
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            ros.rclpy.spin_once(ros.node, timeout_sec=0.2)
            while ros.log_buffer:
                name, text_ = ros.log_buffer.pop(0)
                if name != EXECUTOR_NAME:
                    continue
                if DONE_TEXT in text_:
                    print(DONE_TEXT)
                    return 0
                if any(fail in text_ for fail in FAIL_TEXTS):
                    print(f"EXECUTION FAILED: {text_}")
                    return 1
        print(f"TIMEOUT: no completion message within {args.timeout:.0f} s; the arm may still be "
              "moving. Use the pendant / e-stop if needed.")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Note: a trajectory already dispatched keeps running; stop the arm with "
              "the pendant / e-stop.")
        return 130
    finally:
        if armed:
            disarm(ros)


def disarm(ros):
    for attempt in range(3):
        try:
            ros.set_parameter("execution_enabled", False)
            print("execution_enabled set to false")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not set execution_enabled false (attempt {attempt + 1}): {exc}")
    print("ERROR: execution_enabled may STILL BE TRUE. Run: right_cut.py disarm")
    return False


def cmd_disarm(args, ros):
    return 0 if disarm(ros) else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="converted (right base frame) snapshot (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="convert the snapshot, set the cut and plan it")
    plan.add_argument("cut", choices=sorted(CUTS))
    plan.add_argument("--source", help=f"perception snapshot (default: the executor YAML's external_snapshot_file, {DEFAULT_SOURCE})")
    plan.add_argument("--calib", default=str(rs.DEFAULT_CALIBRATION))
    plan.add_argument("--allow-stale", action="store_true", help="MOCK ONLY: accept an old source file")
    plan.add_argument("--timeout", type=float, default=300.0, help="plan_h_cut timeout [s]")
    plan.add_argument("--inward-mm", type=float, default=RIGHT_FACE_INWARD_MM,
                      help=f"face only: push the whole stroke this far into the detected face "
                           f"(left_cut_inward_m, 0..8 mm, default {RIGHT_FACE_INWARD_MM:g})")
    plan.add_argument("--set", action="append", metavar="PARAM=VALUE",
                      help="extra planning parameter, e.g. left_cut_extend_to_corners=false (repeatable)")
    execute = sub.add_parser("execute", help="dispatch the retained plan")
    execute.add_argument("--timeout", type=float, default=900.0, help="wait for completion [s]")
    sub.add_parser("disarm", help="set execution_enabled false")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        require_domain()
        ros = Ros()
    except CutError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        return {"plan": cmd_plan, "execute": cmd_execute, "disarm": cmd_disarm}[args.command](args, ros)
    except CutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        ros.shutdown()


if __name__ == "__main__":
    sys.exit(main())
