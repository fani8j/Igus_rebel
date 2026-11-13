"""ROS integration: JxcGripperNode against a fake driver node that simulates the JXC.

Runs on ROS_DOMAIN_ID 228 (override with JXC_TEST_DOMAIN=227) with ROS_LOCALHOST_ONLY=1.
"""

import json
import os
import sys
import threading
import time

os.environ["ROS_DOMAIN_ID"] = os.environ.get("JXC_TEST_DOMAIN", "228")
os.environ["ROS_LOCALHOST_ONLY"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402

from igus_rebel_msgs.msg import DigitalIOState, JxcGripperState  # noqa: E402
from igus_rebel_msgs.srv import JxcMoveStep, SetDigitalOutputs  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from rebel_gripper_io.jxc_gripper_node import JxcGripperNode  # noqa: E402
from sim_jxc import SimJxc  # noqa: E402


class FakeDriver(Node):
    def __init__(self):
        super().__init__("fake_rebel_driver")
        self.sim = SimJxc(servo_delay=0.05, home_time=0.3, move_time=0.2, io_latency=0.02)
        self.lock = threading.Lock()
        self.requests = []
        self.pub = self.create_publisher(
            DigitalIOState, "/test_fake/digital_io",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.create_service(SetDigitalOutputs, "/test_fake/set_digital_outputs", self.on_set)
        self.create_timer(0.005, self.tick)
        self.last = None
        self.cap_stamp = 0.0
        self.cap = ([], [])
        self.frozen = False
        self.last_pub = 0.0

    def on_set(self, req, res):
        with self.lock:
            pairs = list(zip(req.irc_outputs, req.states))
            self.requests.append(pairs)
            self.sim.t = time.time()
            ok, msg = self.sim.set_outputs(pairs)
        res.success, res.message = ok, msg
        return res

    def tick(self):
        now = time.time()
        with self.lock:
            self.sim.t = now
            self.sim._step()
            valid = self.sim.io_valid
            if now - self.cap_stamp >= 0.02 and not self.frozen:
                # a new "STATUS": stamp = its receive time
                self.cap_stamp = now
                self.cap = (sorted(self.sim._din()), sorted(self.sim.dout))
            din, dout = self.cap
        if (din, dout, valid, self.cap_stamp) == self.last and now - self.last_pub < 0.1:
            return
        self.last, self.last_pub = (din, dout, valid, self.cap_stamp), now
        msg = DigitalIOState()
        msg.header.stamp.sec = int(self.cap_stamp)
        msg.header.stamp.nanosec = int((self.cap_stamp % 1) * 1e9)
        msg.valid = valid
        msg.irc_offset = 1
        msg.din_on, msg.dout_on = din, dout
        self.pub.publish(msg)


@pytest.fixture(scope="module")
def rig():
    rclpy.init()
    probe = Node("jxc_test_probe")
    time.sleep(1.0)
    foreign = [n for n in probe.get_node_names() if n != "jxc_test_probe" and not n.startswith("_ros2cli_daemon")]
    probe.destroy_node()
    if foreign:
        rclpy.shutdown()
        pytest.skip(f"foreign nodes on the test domain: {foreign}")
    driver = FakeDriver()
    gripper = JxcGripperNode(parameter_overrides=[
        Parameter("digital_io_topic", value="/test_fake/digital_io"),
        Parameter("set_outputs_service", value="/test_fake/set_digital_outputs"),
        Parameter("move_timeout_s", value=1.0),
        Parameter("home_timeout_s", value=3.0),
        Parameter("auto_reset_alarm", value=False),   # the older tests keep alarms latched
    ])
    driver.gripper = gripper
    client = Node("jxc_test_client")
    states = []
    client.create_subscription(JxcGripperState, "/jxc_gripper_node/state",
                               lambda m: states.append(m), 10)
    executor = MultiThreadedExecutor(num_threads=6)
    for n in (driver, gripper, client):
        executor.add_node(n)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield driver, client, states
    executor.shutdown()
    thread.join(timeout=3)
    for n in (driver, gripper, client):
        n.destroy_node()
    rclpy.shutdown()


def call(client, srv_type, name, request, timeout=10.0):
    cli = client.create_client(srv_type, name)
    assert cli.wait_for_service(timeout_sec=5.0), name
    done = threading.Event()
    future = cli.call_async(request)
    future.add_done_callback(lambda _f: done.set())
    assert done.wait(timeout), f"{name} timed out"
    return future.result()


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_servo_on_home_move_step_and_state(rig):
    driver, client, states = rig
    assert wait_for(lambda: states and states[-1].valid and states[-1].alarm_ok)

    r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=1))
    assert not r.success  # no servo yet

    r = call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    assert r.success, r.message
    assert wait_for(lambda: states[-1].svre)

    r = call(client, Trigger, "/jxc_gripper_node/home", Trigger.Request())
    assert r.success, r.message
    assert wait_for(lambda: states[-1].seton and states[-1].inp)

    flat0 = [p for req in driver.requests for p in req]
    assert (26, True) in flat0 and (25, True) not in flat0   # SVON is DOut26, RESET (25) untouched
    driver.requests.clear()
    r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=3))
    assert r.success, r.message
    flat = [p for req in driver.requests for p in req]
    assert flat.index((21, True)) < flat.index((24, True))
    assert flat.index((22, True)) < flat.index((24, True))
    assert driver.sim._steps_at_drive[-1] == 3
    assert wait_for(lambda: 24 not in driver.sim.dout)
    assert wait_for(lambda: not states[-1].busy and states[-1].last_error == "")

    # pushing step with no part: INP never returns -> failure, DRIVE off
    driver.sim.never_reaches = True
    r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=2))
    assert not r.success and "gripped" in r.message
    assert wait_for(lambda: 24 not in driver.sim.dout)
    assert wait_for(lambda: "gripped" in states[-1].last_error)
    driver.sim.never_reaches = False

    r = call(client, Trigger, "/jxc_gripper_node/release_all", Trigger.Request())
    assert r.success
    assert wait_for(lambda: not (driver.sim.dout & {21, 22, 23, 24, 25, 26}))
    r = call(client, Trigger, "/jxc_gripper_node/servo_off", Trigger.Request())
    assert r.success or "SVRE" in r.message
    driver.sim.svre = False


def test_alarm_aborts_running_move_and_stale_refused(rig):
    driver, client, states = rig
    call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    r = call(client, Trigger, "/jxc_gripper_node/home", Trigger.Request())
    assert r.success, r.message
    driver.sim.move_time = 5.0
    threading.Timer(0.3, lambda: setattr(driver.sim, "alarm_ok", False)).start()
    r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=0))
    assert not r.success and "ALARM" in r.message
    assert wait_for(lambda: 24 not in driver.sim.dout)
    assert wait_for(lambda: not states[-1].alarm_ok)

    driver.sim.alarm_ok = True
    call(client, Trigger, "/jxc_gripper_node/release_all", Trigger.Request())
    driver.sim.io_valid = False
    assert wait_for(lambda: not states[-1].valid)
    r = call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    assert not r.success and "fresh" in r.message
    driver.sim.io_valid = True


def test_frozen_stamp_stale_refused_and_latch_visible(rig):
    driver, client, states = rig
    driver.sim.io_valid = True
    driver.sim.alarm_ok = True
    assert wait_for(lambda: states[-1].valid)
    driver.frozen = True            # heartbeat keeps republishing the OLD stamp
    assert wait_for(lambda: not states[-1].valid, timeout=2.0)
    r = call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    assert not r.success and "fresh" in r.message
    driver.frozen = False
    assert wait_for(lambda: states[-1].valid)
    # DRIVE-off ignored although "sent": latch visible in ~/state, then recovered
    driver.sim.ignore_drive_off = True
    call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    call(client, Trigger, "/jxc_gripper_node/home", Trigger.Request())
    r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=1))
    assert not r.success
    assert wait_for(lambda: states[-1].outputs_uncertain and "outputs_uncertain" in states[-1].last_error)
    driver.sim.ignore_drive_off = False
    driver.sim.dout.discard(24)
    r = call(client, Trigger, "/jxc_gripper_node/release_all", Trigger.Request())
    assert r.success, r.message
    assert wait_for(lambda: not states[-1].outputs_uncertain)


def test_reset_alarm_service(rig):
    driver, client, states = rig
    r = call(client, Trigger, "/jxc_gripper_node/release_all", Trigger.Request())
    assert r.success, r.message
    r = call(client, Trigger, "/jxc_gripper_node/reset_alarm", Trigger.Request())
    assert not r.success and "no alarm" in r.message     # refused while healthy
    driver.sim.alarm_ok = False
    assert wait_for(lambda: not states[-1].alarm_ok)
    driver.requests.clear()
    r = call(client, Trigger, "/jxc_gripper_node/reset_alarm", Trigger.Request())
    assert r.success, r.message
    flat = [p for req in driver.requests for p in req]
    assert (25, True) in flat and (25, False) in flat
    assert wait_for(lambda: states[-1].alarm_ok and 25 not in driver.sim.dout)


def test_auto_reset_monitor_timer_after_grip(rig):
    driver, client, states = rig
    node = driver.gripper
    assert node.get_parameter("auto_reset_alarm").value is False   # test override
    assert node.get_parameter("auto_reset_max").value == 3
    assert node.get_parameter("auto_reset_window_s").value == 60.0
    statuses = []
    client.create_subscription(String, "/jxc_gripper_node/status",
                               lambda m: statuses.append(json.loads(m.data)), 10)
    call(client, Trigger, "/jxc_gripper_node/release_all", Trigger.Request())
    r = call(client, Trigger, "/jxc_gripper_node/servo_on", Trigger.Request())
    assert r.success, r.message
    r = call(client, Trigger, "/jxc_gripper_node/home", Trigger.Request())
    assert r.success, r.message
    assert wait_for(lambda: statuses and statuses[-1]["auto_reset"] == "disabled")
    node._hs.cfg.auto_reset_alarm = True
    driver.sim.move_time = 0.2
    driver.sim.svre_return_delay = 0.1
    driver.sim.push_alarm_delay_s = 0.2
    try:
        r = call(client, JxcMoveStep, "/jxc_gripper_node/move_step", JxcMoveStep.Request(step=2))
        assert r.success, r.message
        assert wait_for(lambda: statuses[-1]["auto_reset_count"] == 1, timeout=5.0)
        assert wait_for(lambda: states[-1].alarm_ok and states[-1].svre and states[-1].seton,
                        timeout=5.0)
        st = statuses[-1]
        assert st["auto_reset"] == "enabled" and st["last_auto_reset"] is not None
        assert st["alarm"] is False and st["seton"] is True
        assert wait_for(lambda: 25 not in driver.sim.dout)
    finally:
        driver.sim.push_alarm_delay_s = None
        node._hs.cfg.auto_reset_alarm = False
