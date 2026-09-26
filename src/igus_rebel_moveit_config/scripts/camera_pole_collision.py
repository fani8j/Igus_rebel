#!/usr/bin/env python3
"""Maintain the fixed external camera pole in MoveIt's planning scene.

The pole is an upright 0.15 x 0.15 x 1.0 m box in ``base_link``. Its
front face passes through ``camera_link`` and its depth extends opposite the
camera optical +Z direction, projected horizontally. This node never requests
robot motion.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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
        self.pending = None
        self.last_pose = None
        self.timer = self.create_timer(2.0, self.update)

    def update(self):
        if self.pending is not None:
            if not self.pending.done():
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
            self.get_logger().warning(
                "Waiting for apply_planning_scene; camera pole not applied",
                throttle_duration_sec=10.0,
            )
            return

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
        if current_pose == self.last_pose:
            return
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = [collision]
        self.pole_pub.publish(collision)
        self.pending = self.client.call_async(request)
        self.last_pose = current_pose
        self.get_logger().info(
            "Applying camera_pole: centre="
            f"({pose.position.x:.4f}, {pose.position.y:.4f}, 0.5000) in {self.base}; "
            "size=0.15 x 0.15 x 1.0 m"
        )


def main():
    rclpy.init()
    node = CameraPoleCollision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
