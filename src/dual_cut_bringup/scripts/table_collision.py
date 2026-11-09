#!/usr/bin/env python3
"""Put the dual cell's table into ONE per-arm planning scene as the collision object `table`.

Each per-arm planner (move_group + box_h_cut_executor in /arm_left or /arm_right) plans with a single-arm model
whose scene has only the carton and the camera pole. The coordinator's final dual-arm world check, however, tests
every arm link against the dual URDF link `table`. A plan the per-arm planner finds can therefore be refused for
touching the table. This node closes the gap: the table box is applied to the per-arm scene (relative service
`apply_planning_scene`, i.e. /arm_<arm>/apply_planning_scene, same path as camera_pole_collision.py).

Geometry: IDENTICAL to rebel_dual_table_description urdf/rebel_dual_table.urdf.xacro:
    size = table_size x table_size x table_thickness, no rotation in `world`
    centre = (table_base_setback - table_size / 2, arm_right_base.y / 2, -0.001 - table_thickness / 2)
    world == arm_left_base_link; arm_right_base (xyz, URDF fixed-axis rpy) comes from the same
    dual_arm_calibration YAML the dual URDF is built from (parameter calibration_file).
The box is published in the per-arm planning frame (`base_link` of the single-arm model): for the left arm that is
world itself, for the right arm the pose is T(world->arm_right_base)^-1 * T(world->table).

padding_m grows the box by that much on every face (launch default 0.010 since 2026-10-22: with 0 the planners
found poses 0.4-3 mm from the table, inside the calibration and tracking error; the coordinator's table_min_clearance_m
gate uses the same value). A padded table would collide with the arm's own base link, which sits 0.7 mm (left) / 3.5 mm
(right) above the table by design (distance fixed: it is the mounting). The links in `exempt_links` (default
base_link only; the dual SRDF has the same adjacency exception) therefore get an ACM entry "table x link = allowed" in
move_group's scene, added with the object by reading the full ACM (get_planning_scene), adding the entry and sending the
whole matrix back (a diff with a partial matrix would replace it). Every other link keeps the padded table.

The pose is constant (no TF), re-applied every reapply_period_s and when the service reappears, so a move_group
restart does not silently drop it. This node never requests robot motion.
"""

import math
import signal

import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from shape_msgs.msg import SolidPrimitive

# Same as the xacro defaults (a unit test compares them with the xacro file).
DEFAULT_TABLE_SIZE = 0.65
DEFAULT_TABLE_THICKNESS = 0.04
DEFAULT_TABLE_BASE_SETBACK = 0.110
TABLE_TOP_Z = -0.001        # world z of the table top (xacro: table_cz = -0.001 - thickness / 2)
OBJECT_ID = "table"


def rpy_matrix(rpy):
    """Rotation matrix of a URDF rpy (fixed axes x, y, z): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def matrix_quaternion(m):
    """(x, y, z, w) of a rotation matrix (Shepperd's method, w >= 0)."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return (x, y, z, w)


def load_right_base(calibration_file):
    """(xyz, rpy) of arm_right_base in world from a dual_arm_calibration YAML (as the dual URDF reads it)."""
    with open(calibration_file) as stream:
        document = yaml.safe_load(stream)
    try:
        base = document["dual_arm_calibration"]["arm_right_base"]
        xyz = [float(v) for v in base["xyz"]]
        rpy = [float(v) for v in base["rpy"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{calibration_file}: no usable dual_arm_calibration.arm_right_base xyz/rpy ({error})")
    if len(xyz) != 3 or len(rpy) != 3 or not all(math.isfinite(v) for v in xyz + rpy):
        raise ValueError(f"{calibration_file}: arm_right_base xyz and rpy need 3 finite values each")
    return xyz, rpy


def table_world(right_xyz, size=DEFAULT_TABLE_SIZE, thickness=DEFAULT_TABLE_THICKNESS,
                setback=DEFAULT_TABLE_BASE_SETBACK):
    """Centre of the `table` link in world (xacro world_to_table joint origin; its rotation is identity)."""
    return [setback - size / 2.0, right_xyz[1] / 2.0, TABLE_TOP_Z - thickness / 2.0]


def table_pose_in_arm_frame(arm, right_xyz, right_rpy, size=DEFAULT_TABLE_SIZE,
                            thickness=DEFAULT_TABLE_THICKNESS, setback=DEFAULT_TABLE_BASE_SETBACK):
    """(position, quaternion xyzw) of the table box in the planning frame of `arm` ("left" or "right")."""
    centre = table_world(right_xyz, size, thickness, setback)
    if arm == "left":      # world == arm_left_base_link
        return centre, (0.0, 0.0, 0.0, 1.0)
    if arm != "right":
        raise ValueError(f"arm must be left or right, got {arm!r}")
    rotation_inverse = transpose(rpy_matrix(right_rpy))
    position = matvec(rotation_inverse, [centre[i] - right_xyz[i] for i in range(3)])
    return position, matrix_quaternion(rotation_inverse)


def table_collision_object(arm, right_xyz, right_rpy, frame="base_link", size=DEFAULT_TABLE_SIZE,
                           thickness=DEFAULT_TABLE_THICKNESS, setback=DEFAULT_TABLE_BASE_SETBACK,
                           padding=0.0):
    if not (math.isfinite(padding) and padding >= 0.0):
        raise ValueError("padding_m must be finite and >= 0")
    if not all(math.isfinite(v) and v > 0.0 for v in (size, thickness)):
        raise ValueError("table_size and table_thickness must be finite and > 0")
    position, quaternion = table_pose_in_arm_frame(arm, right_xyz, right_rpy, size, thickness, setback)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [size + 2.0 * padding, size + 2.0 * padding, thickness + 2.0 * padding]
    collision = CollisionObject()
    collision.header.frame_id = frame
    collision.id = OBJECT_ID
    collision.operation = CollisionObject.ADD
    collision.primitives = [primitive]
    collision.primitive_poses = [pose]
    return collision


def acm_with_exemptions(acm, object_id, links):
    """(new AllowedCollisionMatrix, changed): `acm` with the entries (object_id, link) = allowed for every link.

    The matrix is copied with all its entries and default entries; names that are missing are appended. Raises
    ValueError if the matrix is empty or not square (it must be a full scene matrix: sending a partial one to
    apply_planning_scene would replace the scene's ACM)."""
    names = list(acm.entry_names)
    if not names or len(acm.entry_values) != len(names):
        raise ValueError("allowed collision matrix is empty or inconsistent")
    matrix = [list(row.enabled) for row in acm.entry_values]
    if any(len(row) != len(names) for row in matrix):
        raise ValueError("allowed collision matrix is not square")
    changed = False
    for name in [object_id] + list(links):
        if name not in names:
            for row in matrix:
                row.append(False)
            names.append(name)
            matrix.append([False] * len(names))
            changed = True
    index = {name: i for i, name in enumerate(names)}
    for link in links:
        i, j = index[object_id], index[link]
        if not (matrix[i][j] and matrix[j][i]):
            matrix[i][j] = matrix[j][i] = True
            changed = True
    out = AllowedCollisionMatrix()
    out.entry_names = names
    for row in matrix:
        entry = AllowedCollisionEntry()
        entry.enabled = row
        out.entry_values.append(entry)
    out.default_entry_names = list(acm.default_entry_names)
    out.default_entry_values = list(acm.default_entry_values)
    return out, changed


class TableCollision(Node):
    def __init__(self):
        super().__init__("table_collision")
        arm = str(self.declare_parameter("arm", "").value)
        calibration_file = str(self.declare_parameter("calibration_file", "").value)
        frame = str(self.declare_parameter("output_frame_id", "base_link").value)
        size = float(self.declare_parameter("table_size", DEFAULT_TABLE_SIZE).value)
        thickness = float(self.declare_parameter("table_thickness", DEFAULT_TABLE_THICKNESS).value)
        setback = float(self.declare_parameter("table_base_setback", DEFAULT_TABLE_BASE_SETBACK).value)
        padding = float(self.declare_parameter("padding_m", 0.0).value)
        reapply = float(self.declare_parameter("reapply_period_s", 5.0).value)
        self.exempt_links = [str(v) for v in self.declare_parameter("exempt_links", ["base_link"]).value]
        right_xyz, right_rpy = load_right_base(calibration_file)
        self.collision = table_collision_object(arm, right_xyz, right_rpy, frame, size, thickness, setback,
                                                padding)
        self.client = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        self.scene_client = self.create_client(GetPlanningScene, "get_planning_scene")
        self.acm_pending = None
        self.acm_since = None
        self.reapply_period = Duration(seconds=reapply)
        self.pending = None
        self.pending_since = None
        self.last_apply_time = None
        self.service_was_ready = False
        pose = self.collision.primitive_poses[0]
        dims = self.collision.primitives[0].dimensions
        self.get_logger().info(
            f"table for arm_{arm}: box {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} m (padding {padding * 1e3:.1f} mm) "
            f"at ({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}) q(xyzw)=("
            f"{pose.orientation.x:.5f}, {pose.orientation.y:.5f}, {pose.orientation.z:.5f}, "
            f"{pose.orientation.w:.5f}) in {frame}; collision-exempt links (ACM table x link): "
            f"{self.exempt_links or 'none'}; calibration {calibration_file}")
        self.timer = self.create_timer(1.0, self.update)

    def update(self):
        if self.pending is not None:
            if not self.pending.done():
                if self.get_clock().now() - self.pending_since < Duration(seconds=10.0):
                    return
                self.client.remove_pending_request(self.pending)
                self.pending = None
                self.last_apply_time = None
                self.get_logger().warning("table apply request timed out; retrying")
                return
            try:
                result = self.pending.result()
                if result is None or not result.success:
                    self.last_apply_time = None
                    self.get_logger().warning("MoveIt rejected the table collision object")
            except Exception as exc:  # noqa: BLE001 - ROS future exceptions are runtime data
                self.last_apply_time = None
                self.get_logger().warning(f"table scene update failed: {exc}")
            self.pending = None
        if not self.client.service_is_ready():
            if self.service_was_ready:
                self.get_logger().warning("apply_planning_scene disappeared (move_group restart?); "
                                          "the table will be re-applied when it returns")
            self.service_was_ready = False
            self.get_logger().warning("Waiting for apply_planning_scene; table not applied",
                                      throttle_duration_sec=10.0)
            return
        service_reappeared = not self.service_was_ready
        self.service_was_ready = True
        now = self.get_clock().now()
        if self.acm_pending is not None:
            # stage 2: the scene has been read
            if not self.acm_pending.done():
                if now - self.acm_since < Duration(seconds=10.0):
                    return
                self.scene_client.remove_pending_request(self.acm_pending)
                self.acm_pending = None
                self.get_logger().warning("get_planning_scene timed out; retrying")
                return
            future, self.acm_pending = self.acm_pending, None
            self._ensure(future, now)
            return
        due = self.last_apply_time is None or now - self.last_apply_time >= self.reapply_period
        if not (due or service_reappeared):
            return
        if not self.scene_client.service_is_ready():
            self.get_logger().warning("Waiting for get_planning_scene; table not applied", throttle_duration_sec=10.0)
            return
        request = GetPlanningScene.Request()
        request.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES |
                                         PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        self.acm_pending = self.scene_client.call_async(request)
        self.acm_since = now
        self.last_apply_time = now

    def _ensure(self, scene_future, now):
        """Apply only what is missing: the object first (never re-added while present: replacing an object may drop
        its ACM entries), then the ACM exemptions once the object exists. Never sends a partial matrix."""
        try:
            scene = scene_future.result().scene
        except Exception as exc:  # noqa: BLE001 - runtime data
            self.last_apply_time = None
            self.get_logger().warning(f"cannot read the scene ({exc}); table not applied", throttle_duration_sec=10.0)
            return
        present = any(o.id == self.collision.id for o in scene.world.collision_objects)
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        if not present:
            request.scene.world.collision_objects = [self.collision]
            self.get_logger().info(f"applying {self.collision.id}")
        elif self.exempt_links:
            try:
                acm, changed = acm_with_exemptions(scene.allowed_collision_matrix, self.collision.id, self.exempt_links)
            except ValueError as exc:
                self.last_apply_time = None
                self.get_logger().warning(f"{exc}; ACM exemption not applied", throttle_duration_sec=10.0)
                return
            if not changed:
                return
            request.scene.allowed_collision_matrix = acm
            self.get_logger().info(f"ACM exemption(s) {self.collision.id} x {self.exempt_links} applied")
        else:
            return
        self.pending = self.client.call_async(request)
        self.pending_since = now
        self.last_apply_time = None   # check again on the next tick (the exemption follows the object)

def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = TableCollision()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
