"""ROS 2 node: mocap joint targets -> MoveIt Servo JointJog (see README.md)."""
import json
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from control_msgs.msg import JointJog
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool, Int8, String

from .follower_core import FollowerCore, Params
from .interlocks import controller_gate, coordinator_gate, exclusivity_gate


def parse_urdf_limits(xml_text, names):
    """Return (lower, upper) lists for the named revolute joints; raise if any is missing."""
    root = ET.fromstring(xml_text)
    found = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name in names and joint.get("type") in ("revolute", "prismatic") and limit is not None \
                and limit.get("lower") is not None and limit.get("upper") is not None:
            found[name] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError(f"no position limits in URDF for {missing}")
    return [found[n][0] for n in names], [found[n][1] for n in names]


class MocapFollower(Node):
    def __init__(self):
        super().__init__("mocap_follower")
        d = self.declare_parameter
        self.input_topic = d("input_topic", "/mocap/joint_command").value
        self.enable_topic = d("enable_topic", "/mocap/enable").value
        self.joint_names = list(d("joint_names", ["joint1", "joint2", "joint3", "joint4",
                                                  "joint5", "joint6"]).value)
        # Names in the mocap message driving joint_names (same order); [] = joint_names.
        self.input_joint_names = list(d("input_joint_names", [""]).value)
        if self.input_joint_names == [""]:
            self.input_joint_names = []
        js_topic = d("joint_states_topic", "/joint_states").value
        self.servo_joint_topic = d("servo_joint_topic", "/delta_joint_cmds").value
        self.servo_twist_topic = d("servo_twist_topic", "/delta_twist_cmds").value
        servo_status_topic = d("servo_status_topic", "/status").value
        rate = float(d("loop_rate_hz", 100.0).value)
        self.enable_timeout = float(d("enable_timeout_s", 0.3).value)
        self.zero_publish_s = float(d("zero_publish_s", 0.5).value)
        self.joy_button = int(d("joy_enable_button", -1).value)
        self.joy_timeout = float(d("joy_timeout_s", 0.3).value)
        self.refuse_other = bool(d("refuse_if_other_servo_publishers", True).value)
        self.require_servo_status = bool(d("require_servo_status", False).value)
        self.rsp_node = d("robot_description_node", "/robot_state_publisher").value
        urdf_param = d("robot_description", "").value  # optional override (tests)
        # Dual-arm interlocks; every one is off ("") by default so single-arm mode is unchanged.
        self.coordinator_topic = d("coordinator_status_topic", "").value
        self.coordinator_timeout = float(d("coordinator_timeout_s", 2.0).value)
        self.controller_name = d("trajectory_controller", "").value
        self.controller_manager = d("controller_manager", "/controller_manager").value
        self.exclusive_topic = d("exclusive_command_topic", "").value
        self.own_servo_node = d("own_servo_node", "").value  # e.g. /mocap/arm_left/servo
        # servo_relay:=true: the only allowed own publisher on the JTC topic, e.g. /mocap/arm_left/servo/trajectory_relay
        self.own_publisher_node = d("own_publisher_node", "").value
        self.exclusive_allowed = [a for a in d("exclusive_allowed_publishers", [""]).value if a]
        if self.exclusive_topic and not self.own_servo_node:
            raise ValueError("exclusive_command_topic needs own_servo_node")
        if self.own_publisher_node and not self.exclusive_topic:
            raise ValueError("own_publisher_node needs exclusive_command_topic")

        pnames = ("input_timeout_s", "joint_states_timeout_s", "limit_margin_rad",
                  "max_input_step_rad", "max_engage_error_rad", "max_velocity",
                  "max_acceleration", "stop_deceleration", "kp", "deadband_rad",
                  "check_stamp", "max_stamp_skew_s", "clamp_to_limits",
                  "catchup_velocity", "catchup_acceleration", "catchup_done_rad",
                  "max_catchup_error_rad")
        defaults = Params()
        kwargs = {n: d(n, getattr(defaults, n)).value for n in pnames}
        self.params = Params(**kwargs)
        self.params.validate()  # raises -> node refuses to start with unsafe values
        if not (20.0 <= rate <= 500.0):
            raise ValueError("loop_rate_hz must be in [20, 500]")

        self.core = None
        self.limits_error = "limits not loaded"
        if urdf_param:
            self._set_limits(urdf_param)
        else:
            self.rsp_client = self.create_client(GetParameters, f"{self.rsp_node}/get_parameters")
            self.limits_timer = self.create_timer(1.0, self._request_limits)
            self._limits_future = None

        self.latest_js = None
        self.js_rx = 0.0
        self.enable_msg = False
        self.enable_rx = -1e9
        self.joy_state = False
        self.joy_rx = -1e9
        self.servo_status = None
        self.servo_status_rx = -1e9
        self.last_nonzero_t = None
        self.last_status_state = None
        self.stamp_warned = False
        self.last_reject_reason = None
        self.last_clamped = []
        self.input_count = 0
        self.reject_count = 0
        self.coord_state = None
        self.coord_rx = -1e9
        self.ctrl_state = None
        self.ctrl_rx = -1e9
        self.ctrl_pending = None

        self.create_subscription(JointState, self.input_topic, self._on_input, qos_profile_sensor_data)
        self.create_subscription(JointState, js_topic, self._on_js, qos_profile_sensor_data)
        # best-effort subscription matches both reliable and best-effort publishers
        self.create_subscription(Bool, self.enable_topic, self._on_enable, qos_profile_sensor_data)
        if self.joy_button >= 0:
            self.create_subscription(Joy, d("joy_topic", "/joy").value, self._on_joy, 10)
        self.create_subscription(Int8, servo_status_topic, self._on_servo_status, 10)
        if self.coordinator_topic:
            from dual_cut_msgs.msg import CoordinatorStatus  # lazy: only the dual setup needs it
            self.create_subscription(CoordinatorStatus, self.coordinator_topic, self._on_coordinator,
                                     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        if self.controller_name:
            from controller_manager_msgs.srv import ListControllers
            self._list_request = ListControllers.Request
            self.list_client = self.create_client(ListControllers,
                                                  f"{self.controller_manager}/list_controllers")
            self.create_timer(1.0, self._poll_controllers)
        self.jog_pub = self.create_publisher(JointJog, self.servo_joint_topic, 10)
        self.status_pub = self.create_publisher(String, "~/status", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(0.1, self._publish_status)
        self.last_status = None
        self.get_logger().info(
            f"mocap_follower: {self.input_topic} -> {self.servo_joint_topic}; "
            f"vmax={self.params.max_velocity} amax={self.params.max_acceleration}")

    # ---------------------------------------------------------------- limits
    def _set_limits(self, xml_text):
        try:
            lower, upper = parse_urdf_limits(xml_text, self.joint_names)
            self.core = FollowerCore(self.joint_names, lower, upper, self.params,
                                     self.input_joint_names or None)
            self.limits_error = ""
            self.get_logger().info(
                "URDF limits: " + ", ".join(f"{n}[{lo:.2f},{hi:.2f}]"
                                            for n, lo, hi in zip(self.joint_names, lower, upper)))
        except Exception as error:  # fail closed: stay disabled
            self.limits_error = f"limits unusable: {error}"
            self.get_logger().error(self.limits_error)

    def _request_limits(self):
        if self.core is not None:
            self.limits_timer.cancel()
            return
        if self._limits_future is not None and not self._limits_future.done():
            return
        if not self.rsp_client.service_is_ready():
            self.limits_error = f"waiting for {self.rsp_node} robot_description"
            return
        request = GetParameters.Request()
        request.names = ["robot_description"]
        self._limits_future = self.rsp_client.call_async(request)
        self._limits_future.add_done_callback(self._on_limits)

    def _on_limits(self, future):
        try:
            values = future.result().values
            text = values[0].string_value if values else ""
        except Exception as error:
            self.limits_error = f"robot_description request failed: {error}"
            return
        if not text:
            self.limits_error = "robot_description empty"
            return
        self._set_limits(text)
        if self.core is not None:
            self.limits_timer.cancel()

    # ------------------------------------------------------------- callbacks
    def _on_js(self, msg):
        self.latest_js = dict(zip(msg.name, msg.position))
        self.js_rx = time.monotonic()

    def _on_enable(self, msg):
        self.enable_msg = bool(msg.data)
        self.enable_rx = time.monotonic()

    def _on_joy(self, msg):
        idx = self.joy_button
        self.joy_state = idx < len(msg.buttons) and msg.buttons[idx] == 1
        self.joy_rx = time.monotonic()

    def _on_coordinator(self, msg):
        self.coord_state = int(msg.state)
        self.coord_rx = time.monotonic()

    def _poll_controllers(self):
        now = time.monotonic()
        if self.ctrl_pending is not None and now - self.ctrl_pending < 3.0:
            return
        self.ctrl_pending = None
        if not self.list_client.service_is_ready():
            return
        self.ctrl_pending = now
        self.list_client.call_async(self._list_request()).add_done_callback(self._on_controllers)

    def _on_controllers(self, future):
        self.ctrl_pending = None
        try:
            states = {c.name: c.state for c in future.result().controller}
        except Exception as error:
            self.get_logger().warn(f"list_controllers failed: {error}")
            return
        # a controller missing from the answer counts as a (fresh) not-active answer
        self.ctrl_state = states.get(self.controller_name, "not loaded")
        self.ctrl_rx = time.monotonic()

    def _on_servo_status(self, msg):
        self.servo_status = int(msg.data)
        self.servo_status_rx = time.monotonic()

    def _on_input(self, msg):
        if self.core is None:
            return
        now = time.monotonic()
        stamp = msg.header.stamp
        offset = (self.get_clock().now().nanoseconds - (stamp.sec * 10**9 + stamp.nanosec)) * 1e-9
        reason = self.core.on_input(now, list(msg.name), list(msg.position), offset)
        self.input_count += 1
        if reason is not None:
            self.reject_count += 1
            if "stamp" in reason:
                if not self.stamp_warned:
                    self.stamp_warned = True
                    self.get_logger().warn(
                        f"{reason}; mocap input rejected (clocks unsynced? set check_stamp:=false "
                        "only if the offset is understood). Logged once.")
            elif reason != self.last_reject_reason:
                self.get_logger().warn(f"mocap input rejected: {reason}")
        elif self.last_reject_reason is not None:
            self.get_logger().info("mocap input valid again")
        self.last_reject_reason = reason
        if self.core.clamped != self.last_clamped:
            self.get_logger().info("mocap target clamped to limits: " + ", ".join(self.core.clamped)
                                   if self.core.clamped else "mocap target inside limits again")
            self.last_clamped = list(self.core.clamped)

    # ------------------------------------------------------------------ loop
    def _enable_ok(self, now):
        mocap = self.enable_msg and (now - self.enable_rx) <= self.enable_timeout
        joy = self.joy_button >= 0 and self.joy_state and (now - self.joy_rx) <= self.joy_timeout
        return mocap or joy

    def _blocked(self, now):
        if self.core is None:
            return self.limits_error
        if self.coordinator_topic:
            reason = coordinator_gate(self.coord_state, now - self.coord_rx, self.coordinator_timeout)
            if reason:
                return reason
        if self.controller_name:
            reason = controller_gate(self.controller_name, self.ctrl_state, now - self.ctrl_rx)
            if reason:
                return reason
        if self.exclusive_topic:
            pubs = [(i.node_namespace, i.node_name)
                    for i in self.get_publishers_info_by_topic(self.exclusive_topic)]
            reason = exclusivity_gate(self.exclusive_topic, pubs, self.own_servo_node,
                                      self.exclusive_allowed, self.own_publisher_node)
            if reason:
                return reason
        if self.refuse_other:
            others = self.count_publishers(self.servo_joint_topic) - 1  # minus ourselves
            twist = self.count_publishers(self.servo_twist_topic)
            if others > 0 or twist > 0:
                return (f"other Servo command publisher present (joint {others}, twist {twist}); "
                        "only one program may command Servo")
        if self.require_servo_status and now - self.servo_status_rx > 1.0:
            return "Servo status silent (Servo not running?)"
        return None

    def _tick(self):
        now = time.monotonic()
        if self.core is None:
            self.last_status = {"state": "DISABLED", "reason": self.limits_error}
            return
        if self.latest_js is not None and all(n in self.latest_js for n in self.joint_names):
            current = [self.latest_js[n] for n in self.joint_names]
        else:
            current = None
        enable = self._enable_ok(now)
        vel, status = self.core.update(now, current, now - self.js_rx, enable,
                                       self._blocked(now) if enable else None)
        for event in status.events:
            self.get_logger().info(event)
        if status.state != self.last_status_state:
            self.get_logger().info(f"state {self.last_status_state} -> {status.state}: {status.reason}")
            self.last_status_state = status.state
        self.last_status = {
            "state": status.state, "reason": status.reason,
            "max_error_rad": status.max_error, "input_age_s": status.input_age,
            "latched": status.latched, "max_command_rad_s": status.max_command,
            "servo_status": self.servo_status, "inputs": self.input_count,
            "rejected": self.reject_count, "clamped": self.core.clamped if self.core else [],
        }
        nonzero = any(v != 0.0 for v in vel)
        if nonzero:
            self.last_nonzero_t = now
        if nonzero or (self.last_nonzero_t is not None
                       and now - self.last_nonzero_t < self.zero_publish_s):
            self._publish_jog(vel)

    def _publish_jog(self, vel):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = list(self.joint_names)
        msg.velocities = [float(v) for v in vel]
        msg.duration = 0.0
        self.jog_pub.publish(msg)

    def _publish_status(self):
        if self.last_status is not None:
            self.status_pub.publish(String(data=json.dumps(self.last_status)))

    def shutdown_zero(self):
        """Send explicit zero jogs on shutdown so Servo does not coast on the last command."""
        for _ in range(3):
            try:
                self._publish_jog([0.0] * len(self.joint_names))
            except Exception:
                break
            time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MocapFollower()
    except ValueError as error:
        print(f"mocap_follower refuses to start: {error}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
