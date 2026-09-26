#!/usr/bin/env python3
"""Maintain the fixed external camera pole in MoveIt's planning scene.

The pole is an upright 0.15 x 0.15 x 1.0 m box in ``base_link``. Its
front face passes through ``camera_link`` and its depth extends opposite the
camera optical +Z direction, projected horizontally. This node never requests
robot motion.

The pole is re-applied to move_group every ``reapply_period_s`` seconds and
immediately whenever the ``apply_planning_scene`` service reappears, so a
move_group restart does not silently drop it from the planning scene.
"""

import math
import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


def pole_object(camera, optical, frame):
    q = optical.transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid camera optical rotation")
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm

    # Optical +Z in the planning frame, projected onto the horizontal plane.
    fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
    horizontal = math.hypot(fx, fy)
    if horizontal < 0.001:
        raise ValueError("camera viewing direction is vertical; behind is undefined")
    fx, fy = fx / horizontal, fy / horizontal

    position = camera.transform.translation
    if not all(math.isfinite(value) for value in (position.x, position.y, position.z)):
        raise ValueError("invalid camera position")

    yaw = math.atan2(fy, fx)
    pose = Pose()
    pose.position.x = position.x - 0.075 * fx
    pose.position.y = position.y - 0.075 * fy
    pose.position.z = 0.5
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [0.15, 0.15, 1.0]

    collision = CollisionObject()
    collision.header.frame_id = frame
    collision.id = "camera_pole"
    collision.operation = CollisionObject.ADD
    collision.primitives = [primitive]
    collision.primitive_poses = [pose]
    return collision


class CameraPoleCollision(Node):
    def __init__(self):
        super().__init__("camera_pole_collision")
        self.base = self.declare_parameter("base_frame", "base_link").value
        self.camera = self.declare_parameter("camera_frame", "camera_link").value
        self.optical = self.declare_parameter(
            "optical_frame", "camera_color_optical_frame"
        ).value
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pole_pub = self.create_publisher(
            CollisionObject, "/box_motion/camera_pole_collision", qos
        )
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.client = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        self.reapply_period = Duration(
            seconds=float(self.declare_parameter("reapply_period_s", 5.0).value)
        )
        self.pending = None
        self.pending_since = None
        self.last_pose = None
        self.last_apply_time = None
        self.service_was_ready = False
        self.timer = self.create_timer(1.0, self.update)

    def update(self):
        if self.pending is not None:
            if not self.pending.done():
                # A request lost to a move_group restart never completes; drop it.
                if self.get_clock().now() - self.pending_since < Duration(seconds=10.0):
                    return
                self.client.remove_pending_request(self.pending)
                self.pending = None
                self.last_pose = None
                self.get_logger().warning("camera_pole apply request timed out; retrying")
                return
            try:
                result = self.pending.result()
                if result is None or not result.success:
                    self.last_pose = None
                    self.get_logger().warning("MoveIt rejected camera_pole collision object")
            except Exception as exc:  # noqa: BLE001 - ROS future exceptions are runtime data
                self.last_pose = None
                self.get_logger().warning(f"Camera pole scene update failed: {exc}")
            self.pending = None

        if not self.client.service_is_ready():
            if self.service_was_ready:
                self.get_logger().warning(
                    "apply_planning_scene disappeared (move_group restart?); "
                    "camera_pole will be re-applied when it returns"
                )
            self.service_was_ready = False
            self.get_logger().warning(
                "Waiting for apply_planning_scene; camera pole not applied",
                throttle_duration_sec=10.0,
            )
            return
        service_reappeared = not self.service_was_ready
        self.service_was_ready = True

        try:
            camera = self.buffer.lookup_transform(self.base, self.camera, Time())
            optical = self.buffer.lookup_transform(self.base, self.optical, Time())
            collision = pole_object(camera, optical, self.base)
        except (TransformException, ValueError) as exc:
            self.get_logger().warning(
                f"Waiting for valid camera TF: {exc}", throttle_duration_sec=10.0
            )
            return

        pose = collision.primitive_poses[0]
        current_pose = (
            pose.position.x,
            pose.position.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        pose_changed = current_pose != self.last_pose
        now = self.get_clock().now()
        reapply_due = (
            self.last_apply_time is None
            or now - self.last_apply_time >= self.reapply_period
        )
        if not (pose_changed or service_reappeared or reapply_due):
            return
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = [collision]
        if pose_changed:
            # Latched topic consumers (H-cut executor, calibration verification)
            # only need a new message when the geometry changes.
            self.pole_pub.publish(collision)
        self.pending = self.client.call_async(request)
        self.pending_since = now
        self.last_pose = current_pose
        self.last_apply_time = now
        if pose_changed or service_reappeared:
            self.get_logger().info(
                "Applying camera_pole: centre="
                f"({pose.position.x:.4f}, {pose.position.y:.4f}, 0.5000) in {self.base}; "
                "size=0.15 x 0.15 x 1.0 m"
            )


def main():
    # Handle SIGINT/SIGTERM as KeyboardInterrupt instead of rclpy's handler: that handler
    # shuts the context down asynchronously, and on Humble spin() can then fail with
    # RCLError ("context is not valid") instead of returning cleanly.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = CameraPoleCollision()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A terminal Ctrl-C reaches this process twice (directly and forwarded by
        # ros2 launch); ignore repeats so cleanup is not interrupted.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
