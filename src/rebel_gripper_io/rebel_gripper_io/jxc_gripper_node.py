"""ROS node for one SMC JXC61 gripper wired to the igus ReBeL digital I/O.

Subscribes to the driver's digital_io topic, switches outputs through the driver's iRC-numbered
SetDigitalOutputs service and exposes the handshake of jxc_handshake.JxcHandshake as services.
"""

import json
import signal
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from igus_rebel_msgs.msg import DigitalIOState, JxcGripperState
from igus_rebel_msgs.srv import JxcMoveStep, SetDigitalOutputs

from rebel_gripper_io.jxc_handshake import JxcConfig, JxcHandshake, SampleTracker

_PIN_PARAMS = ("out_in0", "out_in1", "out_setup", "out_drive", "out_reset", "out_svon",
               "in_seton", "in_inp", "in_svre", "in_alarm", "in_busy")
_TIME_PARAMS = ("stale_s", "servo_on_timeout_s", "home_timeout_s", "dout_confirm_timeout_s",
                "step_setup_delay_s", "inp_clear_timeout_s", "move_timeout_s",
                "servo_off_timeout_s", "reset_timeout_s", "unseen_start_min_wait_s")
_AUTO_PARAMS = ("auto_reset_alarm", "auto_reset_max", "auto_reset_window_s")
_MONITOR_HZ = 20.0


class RosIo:
    """The `io` object of JxcHandshake backed by ROS topics/services."""

    def __init__(self, node: Node, topic: str, service: str, group, service_timeout_s: float = 1.0):
        self._node = node
        self._lock = threading.Lock()
        self._tracker = SampleTracker()
        self._timeout = service_timeout_s
        self._client = node.create_client(SetDigitalOutputs, service, callback_group=group)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.on_message = lambda: None
        self._sub = node.create_subscription(DigitalIOState, topic, self._cb, qos,
                                             callback_group=group)

    def _cb(self, msg):
        # Freshness from the source: the STATUS receive time stamped by the driver (system clock,
        # same machine), never the DDS receipt time. Zero or regressing stamps are rejected.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._tracker.accept(bool(msg.valid), stamp, msg.din_on, msg.dout_on)
        self.on_message()

    def read(self):
        with self._lock:
            return self._tracker.view(time.time())

    def clock(self) -> float:
        return time.time()   # same clock as the header stamps

    def sleep(self, seconds: float):
        time.sleep(seconds)

    def set_outputs(self, pairs):
        if not self._client.service_is_ready():
            return False, "set_digital_outputs service not available"
        request = SetDigitalOutputs.Request()
        request.irc_outputs = [int(p) for p, _ in pairs]
        request.states = [bool(s) for _, s in pairs]
        done = threading.Event()
        future = self._client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout):
            return False, "set_digital_outputs timed out"
        response = future.result()
        if response is None:
            return False, "set_digital_outputs call failed"
        return bool(response.success), response.message


class JxcGripperNode(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__("jxc_gripper_node", parameter_overrides=parameter_overrides)
        defaults = JxcConfig()
        for name in _PIN_PARAMS + _TIME_PARAMS + _AUTO_PARAMS:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter("digital_io_topic", "/igus_rebel/digital_io")
        self.declare_parameter("set_outputs_service", "/set_digital_outputs")
        config = JxcConfig(**{n: self.get_parameter(n).value
                              for n in _PIN_PARAMS + _TIME_PARAMS + _AUTO_PARAMS})
        for n in _PIN_PARAMS + ("auto_reset_max",):
            setattr(config, n, int(getattr(config, n)))
        config.auto_reset_alarm = bool(config.auto_reset_alarm)
        group = ReentrantCallbackGroup()
        self._io = RosIo(self, self.get_parameter("digital_io_topic").value,
                         self.get_parameter("set_outputs_service").value, group)
        self._hs = JxcHandshake(self._io, config)
        self._hs.log = self._auto_log
        self._io.on_message = self._publish_state
        self._status_pub = self.create_publisher(String, "~/status", 10)
        self._last_status_key = None
        self._last_status_pub = 0.0
        self._state_pub = self.create_publisher(JxcGripperState, "~/state", 10)
        self._last_key = None
        self._last_pub = 0.0
        self._pub_lock = threading.Lock()
        self.create_timer(0.1, self._publish_state, callback_group=group)
        # The alarm monitor has its own callback group so it never blocks (or is blocked by) a
        # service call; the handshake lock/auto flag serialise the actual commands.
        self.create_timer(1.0 / _MONITOR_HZ, self._monitor,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_service(Trigger, "~/servo_on", lambda q, r: self._trigger(self._hs.servo_on, r),
                            callback_group=group)
        self.create_service(Trigger, "~/home", lambda q, r: self._trigger(self._hs.home, r),
                            callback_group=group)
        self.create_service(Trigger, "~/servo_off",
                            lambda q, r: self._trigger(self._hs.servo_off, r), callback_group=group)
        self.create_service(Trigger, "~/reset_alarm",
                            lambda q, r: self._trigger(self._hs.reset_alarm, r),
                            callback_group=group)
        self.create_service(Trigger, "~/release_all",
                            lambda q, r: self._trigger(self._hs.release_all, r),
                            callback_group=group)
        self.create_service(JxcMoveStep, "~/move_step", self._move_step, callback_group=group)
        self.get_logger().info(
            f"JXC gripper node up: topic {self.get_parameter('digital_io_topic').value}, "
            f"service {self.get_parameter('set_outputs_service').value}")

    def _auto_log(self, level, message):
        getattr(self.get_logger(), level)(message)

    def _monitor(self):
        try:
            if self._hs.auto_reset_poll() is not None:
                self._publish_state()
        except Exception as error:   # the monitor must never die silently
            self.get_logger().error(f"alarm monitor error: {error!r}")

    def _trigger(self, command, response):
        result = command()
        response.success, response.message = result.success, result.message
        self._log(result)
        self._publish_state()
        return response

    def _log(self, result):
        if result.success:
            self.get_logger().info(result.message)
        else:
            self.get_logger().warn(result.message)

    def _move_step(self, request, response):
        result = self._hs.move_step(int(request.step))
        response.success, response.message = result.success, result.message
        response.start_seen = bool(result.start_seen)
        self._log(result)
        self._publish_state()
        return response

    def _publish_state(self):
        cfg = self._hs.cfg
        view = self._io.read()
        valid = view.valid and view.stamp > 0.0 and view.age_s <= cfg.stale_s
        self._publish_status(view, valid)
        flags = (valid,
                 valid and cfg.in_svre in view.din_on,
                 valid and cfg.in_seton in view.din_on,
                 valid and cfg.in_inp in view.din_on,
                 valid and cfg.in_alarm in view.din_on,
                 self._hs.outputs_uncertain, self._hs.busy, self._hs.last_error)
        now = time.monotonic()
        with self._pub_lock:
            if flags == self._last_key and now - self._last_pub < 0.1:
                return
            self._last_key, self._last_pub = flags, now
        msg = JxcGripperState()
        msg.header.stamp = self.get_clock().now().to_msg()
        (msg.valid, msg.svre, msg.seton, msg.inp, msg.alarm_ok, msg.outputs_uncertain, msg.busy,
         msg.last_error) = flags
        self._state_pub.publish(msg)


    def _publish_status(self, view, valid):
        cfg = self._hs.cfg
        auto = self._hs.auto_reset_state

        def pin(n):
            return bool(valid and n in view.din_on)
        status = {"valid": valid, "alarm": bool(valid and not pin(cfg.in_alarm)),
                  "svre": pin(cfg.in_svre), "seton": pin(cfg.in_seton), "inp": pin(cfg.in_inp),
                  "auto_reset": "enabled" if auto["enabled"] else "disabled",
                  "auto_reset_reason": auto["reason"], "auto_reset_running": auto["running"],
                  "auto_reset_count": auto["count"], "last_auto_reset": auto["last"]}
        key = json.dumps(status, sort_keys=True)
        now = time.monotonic()
        with self._pub_lock:
            if key == self._last_status_key and now - self._last_status_pub < 1.0:
                return
            self._last_status_key, self._last_status_pub = key, now
        self._status_pub.publish(String(data=key))


def main(args=None):
    # NO rclpy signal handler: Ctrl+C / SIGTERM must not tear the context down before the
    # DRIVE/SETUP-off cleanup has been sent and confirmed.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = JxcGripperNode()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        while not stop.wait(0.2) and spin.is_alive():
            pass
    except KeyboardInterrupt:
        pass
    # The executor keeps spinning in its thread, so the service response is processed while we
    # wait (bounded by the handshake timeouts: 2 attempts x (1 s call + 0.5 s confirm)).
    try:
        ok, why = node._hs.shutdown_outputs_off()
        if ok:
            node.get_logger().info("shutdown: DRIVE/SETUP/RESET off confirmed in digital_io")
        else:
            node.get_logger().error(f"shutdown: DRIVE/SETUP/RESET off NOT confirmed: {why}")
    except Exception as error:
        node.get_logger().error(f"shutdown cleanup raised: {error!r}")
    executor.shutdown()
    spin.join(timeout=2.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
