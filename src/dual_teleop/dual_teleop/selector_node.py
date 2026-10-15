#!/usr/bin/env python3
"""Arm selector / relay between the single-arm keyboard and gamepad nodes and the two Servo instances.

The unmodified ``rebel_servo_teleop_keyboard`` / ``rebel_servo_teleop_gamepad`` binaries publish
``delta_twist_cmds`` (TwistStamped, frame ``base_link`` or ``xeg32_tool_tip``) and ``delta_joint_cmds``
(JointJog, ``joint1..6``) with hard-coded frames and joint names. They are remapped to
``/dual_teleop/in/...``; this node forwards each command to the SELECTED arm's Servo
(``/arm_left/servo`` or ``/arm_right/servo``) after translating frames / joint names, scaling and clamping.
All decisions live in ``selector_core.SelectorCore`` (unit-tested); this node only wires ROS around it.

Interfaces (namespace ``/dual_teleop``):
  in/delta_twist_cmds, in/delta_joint_cmds      commands from keyboard / gamepad (remapped)
  select_arm (std_msgs/String "left"|"right"), services select_left, select_right
  services enable, disable (std_srvs/Trigger), ignore_stale_coordinator (std_srvs/SetBool)
  servo_scene_left, servo_scene_right           dual move_group scene, same-arm pairs allowed: the scene of each Servo
  services touch_mode_on / touch_mode_off, topic touch_mode (JSON, latched)   TOUCH MODE of the selected arm: its knife
                                                links may contact the carton and the table (selector_core.allow_touch_pairs)
  status (std_msgs/String, JSON, 5 Hz)          what the screen shows: selection, state, interlocks
  servo_alive (std_msgs/Int8)                   keeps the remapped keyboard / gamepad start-up happy
  servo_start_stub (std_srvs/Trigger)           their ``servo_node/start_servo`` is remapped here: this node owns the
                                                Servo start and pause state (never starts motion by itself)
Gamepad (optional, ``joy_buttons`` true): LB selects left, RB selects right, Back toggles enable,
both stick clicks held for 1 s toggle touch mode.
"""

import copy
import json
import time

import rclpy
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import ListControllers
from dual_cut_msgs.msg import CoordinatorStatus
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import AllowedCollisionEntry, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningScene
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8, String
from std_srvs.srv import SetBool, Trigger

from dual_teleop.selector_core import (
    ARMS, Command, Config, SelectorCore, allow_same_arm_pairs, allow_touch_pairs)

SERVO_STATUS_NAMES = {
    -1: "INVALID", 0: "NO_WARNING", 1: "DECELERATE_FOR_APPROACHING_SINGULARITY", 2: "HALT_FOR_SINGULARITY",
    3: "DECELERATE_FOR_COLLISION", 4: "HALT_FOR_COLLISION", 5: "JOINT_BOUND",
    6: "DECELERATE_FOR_LEAVING_SINGULARITY",
}
CALL_TIMEOUT_S = 3.0
START_TIMEOUT_S = 30.0
SETTLE_S = 0.8  # after start_servo: let the scene feed replace the ACM Servo fetched from move_group


class ArmServo:
    """Start / pause bookkeeping of one Servo instance."""

    def __init__(self, arm, ns):
        self.arm = arm
        self.ns = ns
        self.running = None  # None = unknown, True/False = last confirmed (start_servo / stop_servo)
        self.running_since = None
        self.pending_since = None
        self.status_code = None
        self.status_time = None
        self.start_client = None
        self.stop_client = None


class SelectorNode(Node):
    def __init__(self):
        super().__init__("selector")
        d = self.declare_parameter
        default_arm = d("default_arm", "left").value
        enabled = d("start_enabled", False).value
        cfg = Config(
            speed_scale=d("speed_scale", 0.5).value,
            max_linear_mps=d("max_linear_mps", 0.05).value,
            max_angular_rps=d("max_angular_rps", 0.3).value,
            max_joint_rps=d("max_joint_rps", 0.1).value,
            base_command_frame=d("base_command_frame", "world").value,
            block_states=tuple(int(x) for x in d("block_states", [3, 4]).value),
            coordinator_timeout_s=d("coordinator_timeout_s", 5.0).value,
            controller_timeout_s=d("controller_timeout_s", 3.0).value,
            scene_timeout_s=d("scene_timeout_s", 5.0).value,
            require_scene=d("require_scene", True).value,
            touch_mode_timeout_s=d("touch_mode_timeout_s", 120.0).value,
        )
        touch_scale = d("touch_speed_scale", -1.0).value  # -1: one third of speed_scale
        cfg.touch_speed_scale = min(cfg.speed_scale, touch_scale if touch_scale > 0 else cfg.speed_scale / 3.0)
        self.carton_ids = list(d("carton_ids", ["detected_carton"]).value)
        self.joy_touch_buttons = [int(x) for x in d("joy_touch_buttons", [9, 10]).value]
        self.joy_touch_hold_s = d("joy_touch_hold_s", 1.0).value
        self.joy_touch_since = None
        self.joy_touch_fired = False
        if cfg.base_command_frame not in ("world", "arm_base"):
            raise ValueError("base_command_frame must be 'world' or 'arm_base'")
        self.core = SelectorCore(cfg, default_arm=default_arm, enabled=enabled)
        self.controller_manager = d("controller_manager", "/controller_manager").value
        self.joy_buttons = d("joy_buttons", False).value
        self.btn_left = d("joy_button_left", 4).value
        self.btn_right = d("joy_button_right", 5).value
        self.btn_toggle = d("joy_button_toggle_enable", 6).value
        self.joy_prev = {}
        self.servos = {a: ArmServo(a, d(f"servo_ns_{a}", f"/arm_{a}/servo").value) for a in ARMS}
        self.controller_names = {a: d(f"controller_{a}", f"arm_{a}_trajectory_controller").value for a in ARMS}
        self.add_on_set_parameters_callback(self.on_params)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.twist_pubs = {a: self.create_publisher(TwistStamped, f"{self.servos[a].ns}/delta_twist_cmds", qos)
                           for a in ARMS}
        self.joint_pubs = {a: self.create_publisher(JointJog, f"{self.servos[a].ns}/delta_joint_cmds", qos)
                           for a in ARMS}
        self.create_subscription(TwistStamped, "in/delta_twist_cmds", self.on_twist, qos)
        self.create_subscription(JointJog, "in/delta_joint_cmds", self.on_joint, qos)
        self.create_subscription(CoordinatorStatus, d("coordinator_status_topic", "/dual_cut_coordinator/status").value,
                                 self.on_coordinator, qos)
        self.create_subscription(String, "select_arm", self.on_select_topic, qos)
        if self.joy_buttons:
            self.create_subscription(Joy, d("joy_topic", "/joy").value, self.on_joy, qos)
        for a in ARMS:
            s = self.servos[a]
            self.create_subscription(Int8, f"{s.ns}/status", lambda m, a=a: self.on_servo_status(a, m), qos)
            s.start_client = self.create_client(Trigger, f"{s.ns}/servo_node/start_servo")
            s.stop_client = self.create_client(Trigger, f"{s.ns}/servo_node/stop_servo")
        # Servo's planning scene = move_group's scene with same-arm pairs allowed (see selector_core.allow_same_arm_pairs)
        self.allow_same_arm = d("allow_same_arm_pairs", True).value
        self.scene_client = self.create_client(GetPlanningScene, d("planning_scene_service", "/get_planning_scene").value)
        # One scene topic per arm so the touch-mode ACM (knife vs carton/table) reaches the selected arm's Servo only.
        scene_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.scene_pubs = {a: self.create_publisher(PlanningScene, f"servo_scene_{a}", scene_qos) for a in ARMS}
        self.last_scene = None
        self.touch_published = None
        self.touch_pub = self.create_publisher(
            String, "touch_mode", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                             durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(Trigger, "touch_mode_on", lambda q, r: self.svc_touch(True, r))
        self.create_service(Trigger, "touch_mode_off", lambda q, r: self.svc_touch(False, r))
        self.scene_pending = None
        self.create_timer(0.5, self.poll_scene)
        self.list_client = self.create_client(ListControllers, f"{self.controller_manager}/list_controllers")
        self.list_pending = None

        self.status_pub = self.create_publisher(String, "status", 10)
        self.alive_pub = self.create_publisher(Int8, "servo_alive", 10)
        self.create_service(Trigger, "select_left", lambda q, r: self.svc_select("left", r))
        self.create_service(Trigger, "select_right", lambda q, r: self.svc_select("right", r))
        self.create_service(Trigger, "enable", lambda q, r: self.svc_enable(True, r))
        self.create_service(Trigger, "disable", lambda q, r: self.svc_enable(False, r))
        self.create_service(SetBool, "ignore_stale_coordinator", self.svc_ignore_stale)
        self.create_service(Trigger, "servo_start_stub", self.svc_start_stub)

        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        self.create_timer(0.1, self.poll_tcp)
        self.last_summary = None
        self.counts = {"forwarded": 0, "dropped": 0}
        self.create_timer(0.05, self.tick)
        self.create_timer(1.0, self.poll_controllers)
        self.create_timer(0.2, self.publish_status)
        self.get_logger().info(
            f"dual_teleop selector: arm={self.core.selected}, enabled={self.core.enabled} "
            f"(start paused unless start_enabled), speed_scale={cfg.speed_scale}, "
            f"caps {cfg.max_linear_mps} m/s, {cfg.max_angular_rps} rad/s, joint {cfg.max_joint_rps} rad/s")

    # ----- time / parameters -----
    @staticmethod
    def now():
        return time.monotonic()

    def on_params(self, params):
        cfg = self.core.cfg
        for p in params:
            if p.name == "speed_scale":
                if not 0.0 < p.value <= 1.0:
                    return SetParametersResult(successful=False, reason="speed_scale must be in (0, 1]")
                cfg.speed_scale = float(p.value)
                cfg.touch_speed_scale = min(cfg.touch_speed_scale, cfg.speed_scale)
            elif p.name == "touch_speed_scale":
                if not 0.0 < p.value <= cfg.speed_scale:
                    return SetParametersResult(successful=False, reason="touch_speed_scale must be in (0, speed_scale]")
                cfg.touch_speed_scale = float(p.value)
            elif p.name == "touch_mode_timeout_s":
                if p.value < 0.0:
                    return SetParametersResult(successful=False, reason="touch_mode_timeout_s must be >= 0")
                cfg.touch_mode_timeout_s = float(p.value)
            elif p.name in ("max_linear_mps", "max_angular_rps", "max_joint_rps"):
                if not 0.0 < p.value <= {"max_linear_mps": 0.1, "max_angular_rps": 0.5, "max_joint_rps": 0.4}[p.name]:
                    return SetParametersResult(successful=False, reason=f"{p.name} out of range")
                setattr(cfg, p.name, float(p.value))
        return SetParametersResult(successful=True)

    # ----- publishing -----
    def emit(self, outputs):
        for out in outputs:
            stamp = self.get_clock().now().to_msg()
            if out.kind == "twist":
                msg = TwistStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = out.frame
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = out.linear
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = out.angular
                self.twist_pubs[out.arm].publish(msg)
            else:
                msg = JointJog()
                msg.header.stamp = stamp
                msg.header.frame_id = out.frame
                msg.joint_names = list(out.joint_names)
                msg.velocities = [float(v) for v in out.velocities]
                self.joint_pubs[out.arm].publish(msg)

    # ----- inputs -----
    def on_twist(self, msg):
        cmd = Command("twist", frame=msg.header.frame_id,
                      linear=(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z),
                      angular=(msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z))
        self.route(cmd)

    def on_joint(self, msg):
        self.route(Command("joint", frame=msg.header.frame_id, joint_names=list(msg.joint_names),
                           velocities=list(msg.velocities)))

    def route(self, cmd):
        outputs, reason = self.core.handle_command(cmd, self.now())
        self.emit(outputs)
        if reason:
            self.counts["dropped"] += 1
        elif outputs:
            self.counts["forwarded"] += 1

    def on_coordinator(self, msg):
        self.core.update_coordinator(msg.state, self.now())

    def on_servo_status(self, arm, msg):
        s = self.servos[arm]
        s.status_code, s.status_time = int(msg.data), self.now()

    def on_select_topic(self, msg):
        arm = msg.data.strip().lower()
        if arm in ARMS:
            self.do_select(arm)
        else:
            self.get_logger().warn(f"select_arm: expected 'left' or 'right', got {msg.data!r}")

    def on_joy(self, msg):
        def pressed(i):
            now = bool(i < len(msg.buttons) and msg.buttons[i])
            before = self.joy_prev.get(i, False)
            self.joy_prev[i] = now
            return now and not before
        if pressed(self.btn_left):
            self.do_select("left")
        if pressed(self.btn_right):
            self.do_select("right")
        if pressed(self.btn_toggle):
            self.do_enable(not self.core.enabled)
        # touch mode: hold ALL joy_touch_buttons (default both stick clicks) for joy_touch_hold_s; one toggle per hold
        held = all(i < len(msg.buttons) and msg.buttons[i] for i in self.joy_touch_buttons)
        if held and self.joy_touch_since is None:
            self.joy_touch_since = self.now()
        elif not held:
            self.joy_touch_since, self.joy_touch_fired = None, False
        if held and not self.joy_touch_fired and self.now() - self.joy_touch_since >= self.joy_touch_hold_s:
            self.joy_touch_fired = True
            self.do_touch(not self.core.touch_on)

    # ----- operator actions -----
    def do_select(self, arm):
        self.emit(self.core.select(arm, self.now()))
        self.sync_touch()
        self.get_logger().info(f"selected arm: {arm}")

    def do_enable(self, enabled):
        self.emit(self.core.set_enabled(enabled, self.now()))
        self.sync_touch()
        self.get_logger().info(f"teleop {'ENABLED' if enabled else 'DISABLED (paused)'}")

    def svc_select(self, arm, response):
        self.do_select(arm)
        response.success, response.message = True, f"selected arm_{arm}"
        return response

    def svc_enable(self, enabled, response):
        self.do_enable(enabled)
        response.success = True
        response.message = "teleop enabled" if enabled else "teleop disabled"
        return response

    def svc_ignore_stale(self, request, response):
        self.core.ignore_stale = bool(request.data)
        response.success, response.message = True, f"ignore_stale_coordinator={self.core.ignore_stale}"
        return response

    def svc_start_stub(self, request, response):
        response.success = True
        response.message = "dual_teleop owns the Servo start/pause state; nothing to do"
        return response

    # ----- planning scene for the two Servo instances -----
    def poll_scene(self):
        if self.scene_pending is not None and self.now() - self.scene_pending < CALL_TIMEOUT_S:
            return
        self.scene_pending = None
        if not self.scene_client.service_is_ready():
            return
        req = GetPlanningScene.Request()
        c = PlanningSceneComponents
        req.components.components = (c.SCENE_SETTINGS | c.ROBOT_STATE | c.WORLD_OBJECT_NAMES | c.WORLD_OBJECT_GEOMETRY
                                     | c.OCTOMAP | c.TRANSFORMS | c.ALLOWED_COLLISION_MATRIX | c.ROBOT_STATE_ATTACHED_OBJECTS)
        self.scene_pending = self.now()
        self.scene_client.call_async(req).add_done_callback(self.on_scene)

    def on_scene(self, future):
        self.scene_pending = None
        try:
            scene = future.result().scene
        except Exception as error:
            self.get_logger().warn(f"get_planning_scene failed: {error}")
            return
        self.last_scene = scene
        self.core.update_scene(self.now())
        self.publish_scenes()

    def publish_scenes(self):
        """Publish the per-arm Servo scenes from the cached move_group scene (also right after a touch-mode change)."""
        if self.last_scene is None:
            return
        present = {o.id for o in self.last_scene.world.collision_objects}
        cartons = [c for c in self.carton_ids if c in present]
        for arm in ARMS:
            scene = copy.deepcopy(self.last_scene)
            acm = scene.allowed_collision_matrix
            if acm.entry_names:
                names = list(acm.entry_names)
                rows = [list(e.enabled) for e in acm.entry_values]
                if self.allow_same_arm:
                    rows = allow_same_arm_pairs(names, rows)
                if self.core.acm_touch_arm == arm:
                    names, rows = allow_touch_pairs(names, rows, arm, cartons, retreat=self.core.touch_arm is None)
                acm.entry_names = names
                acm.entry_values = [AllowedCollisionEntry(enabled=r) for r in rows]
            scene.is_diff = False
            self.scene_pubs[arm].publish(scene)

    # ----- touch mode -----
    def do_touch(self, on):
        ok, message, zeros = self.core.set_touch(on, self.now())
        self.emit(zeros)
        self.sync_touch()
        if on and ok:
            self.get_logger().warn(f"TOUCH MODE {self.core.touch_arm}: knife may contact carton/table")
        else:
            self.get_logger().info(message)
        return ok, message

    def svc_touch(self, on, response):
        response.success, response.message = self.do_touch(on)
        return response

    def sync_touch(self):
        """After any touch-mode change (operator or automatic): new Servo scenes at once + the touch topic."""
        state = (self.core.touch_arm, self.core.retreat_arm, self.core.touch_off_reason)
        if state == self.touch_published:
            return
        first = self.touch_published is None
        was_on = bool(self.touch_published and (self.touch_published[0] or self.touch_published[1]))
        self.touch_published = state
        self.publish_scenes()
        self.publish_touch()
        if not first and was_on and self.core.acm_touch_arm is None:
            self.get_logger().warn(f"TOUCH MODE OFF: {self.core.touch_off_reason}")

    def publish_touch(self):
        st = self.core.status(self.now())["touch"]
        msg = String()
        msg.data = json.dumps(st)
        self.touch_pub.publish(msg)

    # ----- controller poll -----
    def poll_tcp(self):
        for arm in ARMS:
            try:
                t = self.tfbuf.lookup_transform("world", f"arm_{arm}_xeg32_tool_tip", rclpy.time.Time()).transform.translation
                self.core.update_tcp(arm, (t.x, t.y, t.z))
            except Exception:  # no TF yet: the retreat window then only ends by time
                pass

    def poll_controllers(self):
        if self.list_pending is not None and self.now() - self.list_pending < CALL_TIMEOUT_S:
            return
        if not self.list_client.service_is_ready():
            self.list_pending = None
            return
        self.list_pending = self.now()
        future = self.list_client.call_async(ListControllers.Request())
        future.add_done_callback(self.on_controllers)

    def on_controllers(self, future):
        self.list_pending = None
        try:
            states = {c.name: c.state for c in future.result().controller}
        except Exception as error:  # service died mid-call: leave the old data to expire
            self.get_logger().warn(f"list_controllers failed: {error}")
            return
        now = self.now()
        for a in ARMS:
            if self.controller_names[a] in states:
                self.core.update_controller(a, states[self.controller_names[a]] == "active", now)
            else:
                self.core.update_controller(a, False, now)

    # ----- Servo start / pause management -----
    def call(self, arm, client, label, on_success):
        s = self.servos[arm]
        s.pending_since = self.now()
        self.get_logger().info(f"{arm} Servo {label}: calling")
        future = client.call_async(Trigger.Request())

        def done(fut):
            s.pending_since = None
            try:
                res = fut.result()
            except Exception as error:
                self.get_logger().warn(f"{arm} Servo {label} failed: {error}")
                return
            if res.success:
                on_success()
            else:
                self.get_logger().warn(f"{arm} Servo {label} refused: {res.message}")
        future.add_done_callback(done)

    def manage_servo(self, arm, now):
        """Servo runs only while teleop is open for this arm, and is STOPPED (not paused) otherwise.

        pause_servo/unpause_servo is not usable here: after an unpause Humble's Servo publishes its first
        command from the joint state it had when it was paused, so an arm moved meanwhile (by the
        coordinator, a pendant, a joint-limit recovery) would jump back. stop_servo/start_servo re-reads the
        robot state (verified in mock, see scan_results/.../teleop/results/servo_pause_vs_stop.txt).
        """
        s = self.servos[arm]
        # start_servo can take seconds (Servo waits for the scene / robot state); never repeat it early,
        # Humble's second start_servo re-initialises the Servo thread.
        timeout = START_TIMEOUT_S if not s.running else CALL_TIMEOUT_S
        if s.pending_since is not None and now - s.pending_since < timeout:
            return
        s.pending_since = None
        if s.running and s.running_since is not None and now - s.running_since > 3.0 \
                and (s.status_time is None or now - s.status_time > 3.0):
            self.get_logger().warn(f"{arm} Servo publishes no status: treating it as not running")
            s.running = None
            return
        desired = self.core.desired_unpaused(arm, now)
        if desired and s.running is not True:
            if s.start_client.service_is_ready():
                def started(s=s):
                    s.running, s.running_since = True, self.now()
                    self.get_logger().info(f"{arm} Servo started")
                self.call(arm, s.start_client, "start_servo", started)
        elif not desired and s.running is not False and self.core.pause_due(arm, now):
            if s.stop_client.service_is_ready():
                def stopped(s=s):
                    s.running, s.running_since = False, None
                    self.get_logger().info(f"{arm} Servo stopped")
                self.call(arm, s.stop_client, "stop_servo", stopped)

    def tick(self):
        now = self.now()
        self.emit(self.core.tick(now))
        for a in ARMS:
            self.manage_servo(a, now)
            s = self.servos[a]
            settled = s.running_since is not None and now - s.running_since >= SETTLE_S
            self.core.set_servo_running(a, bool(s.running and settled and self.core.desired_unpaused(a, now)))
        self.emit(self.core.tick(now))
        self.sync_touch()
        alive = Int8()
        alive.data = 0
        self.alive_pub.publish(alive)

    # ----- screen -----
    def publish_status(self):
        now = self.now()
        st = self.core.status(now)
        st["servo"] = {}
        for a in ARMS:
            s = self.servos[a]
            fresh = s.status_time is not None and now - s.status_time < 2.0
            st["servo"][a] = {
                "running": s.running,
                "status": SERVO_STATUS_NAMES.get(s.status_code, str(s.status_code)) if fresh else "no status",
            }
        st["counts"] = dict(self.counts)
        self.publish_touch()
        msg = String()
        msg.data = json.dumps(st)
        self.status_pub.publish(msg)
        summary = (st["state"], tuple(st["reasons"]), tuple(st["warnings"]), st["selected"])
        if summary != self.last_summary:
            self.last_summary = summary
            text = f"teleop {st['state']} (arm {st['selected']})"
            if st["reasons"]:
                text += ": " + "; ".join(st["reasons"])
            if st["warnings"]:
                text += " | WARNING: " + "; ".join(st["warnings"])
            if st["state"] == "BLOCKED" or st["warnings"]:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)


def main():
    rclpy.init()
    node = SelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # leave both arms with a zero command
            for a in ARMS:
                node.emit(node.core.zero_outputs(a))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
