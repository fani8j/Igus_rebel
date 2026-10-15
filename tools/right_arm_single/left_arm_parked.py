#!/usr/bin/env python3
"""Keep a static `left_arm_parked` collision box in the right-arm-alone planning scene.

The left ReBeL stands in the cell (parked at the SRDF "home" pose, all joints 0) but is not part of
the single-arm right robot model. This node computes the AABB of ALL collision geometry of the
single-arm robot model (igus_rebel_xeg32_wrapper.urdf.xacro, tool_variant) at joint zero, pads it,
places it at the left base pose as seen from the right base (B_right^-1, calibration file) and
applies it to move_group's planning scene (ApplyPlanningScene), re-applying every few seconds and
whenever apply_planning_scene reappears (same pattern as camera_pole_collision.py).
The H-cut executor only removes `detected_carton`; it never clears the scene.
"""

import argparse
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parked_arm  # noqa: E402
import right_snapshot as rs  # noqa: E402


def build_object(calibration_file, tool_variant, padding, object_id, frame):
    import xacro
    from ament_index_python.packages import get_package_share_directory
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    share = get_package_share_directory("igus_rebel_description")
    urdf = xacro.process_file(
        f"{share}/urdf/igus_rebel_xeg32_wrapper.urdf.xacro",
        mappings={"hardware_protocol": "mock_hardware", "tool_variant": tool_variant},
    ).toxml()

    def resolve(uri):
        if uri.startswith("package://"):
            package, _, rest = uri[len("package://"):].partition("/")
            return f"{get_package_share_directory(package)}/{rest}"
        return uri.replace("file://", "")

    low, high, count = parked_arm.collision_aabb(urdf, resolve)
    calibration = rs.load_calibration(calibration_file)
    dims, position, quat = parked_arm.parked_box(low, high, padding, calibration)
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(v) for v in dims]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = object_id
    obj.operation = CollisionObject.ADD
    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    summary = (
        f"left_arm_parked from {count} collision geometries ({tool_variant}, joints 0): AABB in left "
        f"base frame min={[round(float(v), 4) for v in low]} max={[round(float(v), 4) for v in high]}, "
        f"padding {padding} m; box dims={[round(float(v), 4) for v in dims]} at "
        f"{[round(float(v), 4) for v in position]} quat(xyzw)={[round(float(v), 5) for v in quat]} "
        f"in {frame}")
    return obj, summary


class LeftArmParked(Node):
    def __init__(self, args):
        super().__init__("left_arm_parked")
        from moveit_msgs.srv import ApplyPlanningScene
        self.object, summary = build_object(
            args.calibration_file, args.tool_variant, args.padding, args.object_id, args.frame)
        self.get_logger().info(summary)
        self.client = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        self.request_type = ApplyPlanningScene.Request
        self.pending = None
        self.was_ready = False
        self.applied_at = None
        self.period = args.reapply_period_s
        self.create_timer(1.0, self.update)

    def update(self):
        now = self.get_clock().now()
        if self.pending is not None:
            if not self.pending.done():
                if (now - self.pending_since).nanoseconds < 10e9:
                    return
                self.client.remove_pending_request(self.pending)
                self.get_logger().warning("left_arm_parked apply request timed out; retrying")
            else:
                try:
                    if not self.pending.result().success:
                        self.get_logger().warning("MoveIt rejected left_arm_parked")
                        self.applied_at = None
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warning(f"left_arm_parked update failed: {exc}")
                    self.applied_at = None
            self.pending = None
        if not self.client.service_is_ready():
            self.was_ready = False
            self.get_logger().warning("Waiting for apply_planning_scene", throttle_duration_sec=10.0)
            return
        reappeared = not self.was_ready
        self.was_ready = True
        due = self.applied_at is None or (now - self.applied_at).nanoseconds >= self.period * 1e9
        if not (due or reappeared):
            return
        request = self.request_type()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = [self.object]
        self.pending = self.client.call_async(request)
        self.pending_since = now
        self.applied_at = now
        if reappeared:
            self.get_logger().info("applied left_arm_parked to the planning scene")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-file", default=str(rs.DEFAULT_CALIBRATION))
    parser.add_argument("--tool-variant", default="knife_c")
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--object-id", default="left_arm_parked")
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--reapply-period-s", type=float, default=5.0)
    parser.add_argument("--print-only", action="store_true", help="print the box and exit")
    args, ros_args = parser.parse_known_args()
    if args.print_only:
        _, summary = build_object(args.calibration_file, args.tool_variant, args.padding,
                                  args.object_id, args.frame)
        print(summary)
        return 0
    rclpy.init(args=ros_args)
    node = LeftArmParked(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
