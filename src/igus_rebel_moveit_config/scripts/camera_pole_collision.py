#!/usr/bin/env python3
"""Maintain the fixed external camera pole in MoveIt's planning scene.

Two objects in ``base_link``: ``camera_pole``, the real upright 0.03 x 0.03 x
1.0 m pole just behind the camera head, and ``camera_head``, a box around the
real Orbbec Gemini 336 body (the camera itself, about 0.45-0.5 m up). The head
box is the body's collision-mesh bounding box in the ``camera_link`` frame
(see ``BODY_MIN``/``BODY_MAX``) plus ``HEAD_MARGIN`` on every side, posed with
the full ``camera_link`` orientation from TF (so a downward-tilted camera is
tilted too). The pole is placed ``POLE_BACK`` behind ``camera_link``,
independent of the head size; "behind" is opposite the camera optical +Z
direction, projected horizontally. Only ``camera_pole`` is published on the
latched topic (its consumers expect one box). This node never requests robot
motion.

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


POLE_SIZE = 0.03
# Pole centre offset behind camera_link (horizontal). Matches the real pole position; it was
# HEAD_SIZE/2 + POLE_SIZE/2 for the former 0.10 m cube head and must NOT change with the head size.
POLE_BACK = 0.065

# Gemini 336 body extent in the camera_link frame (x forward, y left, z up), metres. From
# /opt/ros/humble/share/orbbec_description: urdf/gemini_335_336.urdf.xacro (camera_link collision mesh
# meshes/gemini335_336/camera_link.STL, origin xyz = -0.01587 -0.025 -0.0125). Mesh bounds
# (30.0 x 89.5 x 25.0 mm, i.e. the 90 x 25 x 25 mm datasheet size) shifted by that origin and rounded
# outward to 0.1 mm. camera_link is the left IR/depth sensor, not the body centre.
BODY_MIN = (-0.0258, -0.0698, -0.0126)
BODY_MAX = (0.0043, 0.0198, 0.0126)
# Margin added on every side of the body: covers the mount bracket (absent from the description),
# the calibration error of camera_link and the USB plug.
HEAD_MARGIN = 0.010


def _box(object_id, frame, x, y, z, yaw, dims):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(dims)
    collision = CollisionObject()
    collision.header.frame_id = frame
    collision.id = object_id
    collision.operation = CollisionObject.ADD
    collision.primitives = [primitive]
    collision.primitive_poses = [pose]
    return collision


def _unit_quaternion(q, what):
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"invalid {what} rotation")
    return q.x / norm, q.y / norm, q.z / norm, q.w / norm


def _rotate(q, v):
    """Rotate vector v by unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _head_box(frame, position, q):
    centre = tuple((lo + hi) / 2.0 for lo, hi in zip(BODY_MIN, BODY_MAX))
    dims = tuple(hi - lo + 2.0 * HEAD_MARGIN for lo, hi in zip(BODY_MIN, BODY_MAX))
    ox, oy, oz = _rotate(q, centre)
    box = _box("camera_head", frame, position.x + ox, position.y + oy, position.z + oz, 0.0, dims)
    pose = box.primitive_poses[0]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return box


def pole_objects(camera, optical, frame):
    """(camera_pole, camera_head) collision objects."""
    x, y, z, w = _unit_quaternion(optical.transform.rotation, "camera optical")
    camera_q = _unit_quaternion(camera.transform.rotation, "camera_link")

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
    pole = _box("camera_pole", frame, position.x - POLE_BACK * fx, position.y - POLE_BACK * fy, 0.5,
                yaw, (POLE_SIZE, POLE_SIZE, 1.0))
    head = _head_box(frame, position, camera_q)
    return pole, head


class CameraPoleCollision(Node):
    def __init__(self):
        super().__init__("camera_pole_collision")
        self.base = self.declare_parameter("base_frame", "base_link").value
        # Frame_id written on the collision objects (empty = base_frame, the original behaviour).
        # Per-arm dual-cell stacks look the pose up in the dual TF tree (base_frame:=arm_X_base_link)
        # but publish it for a single-arm move_group whose own base is called base_link.
        self.output_frame = (
            self.declare_parameter("output_frame_id", "").value or self.base
        )
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
            collision, head = pole_objects(camera, optical, self.output_frame)
        except (TransformException, ValueError) as exc:
            self.get_logger().warning(
                f"Waiting for valid camera TF: {exc}", throttle_duration_sec=10.0
            )
            return

        pose = collision.primitive_poses[0]
        head_pose = head.primitive_poses[0]
        pole_key = (
            pose.position.x,
            pose.position.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        current_pose = pole_key + (
            head_pose.position.x,
            head_pose.position.y,
            head_pose.position.z,
            head_pose.orientation.x,
            head_pose.orientation.y,
            head_pose.orientation.z,
            head_pose.orientation.w,
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
        request.scene.world.collision_objects = [collision, head]
        if self.last_pose is None or pole_key != self.last_pose[:4]:
            # Latched topic consumers (H-cut executor, calibration verification)
            # only need a new message when the pole geometry changes.
            self.pole_pub.publish(collision)
        self.pending = self.client.call_async(request)
        self.pending_since = now
        self.last_pose = current_pose
        self.last_apply_time = now
        if pose_changed or service_reappeared:
            self.get_logger().info(
                "Applying camera_pole: centre="
                f"({pose.position.x:.4f}, {pose.position.y:.4f}, 0.5000) in {self.output_frame}; "
                f"size={POLE_SIZE} x {POLE_SIZE} x 1.0 m; camera_head: Gemini 336 box "
                f"{head.primitives[0].dimensions[0]:.3f} x {head.primitives[0].dimensions[1]:.3f} x "
                f"{head.primitives[0].dimensions[2]:.3f} m (body + {HEAD_MARGIN * 1e3:.0f} mm margin, "
                f"camera_link orientation) at "
                f"({head_pose.position.x:.4f}, {head_pose.position.y:.4f}, {head_pose.position.z:.4f})"
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
