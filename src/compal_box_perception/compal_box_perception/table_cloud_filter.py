"""Table-region point-cloud filter for fast RViz/Foxglove visualization.

Subscribes the raw registered cloud, keeps points inside a base_link box
(table region), applies a voxel-grid decimation, republishes a small cloud.
Visual-only path: never commands motion, never modifies the planning scene.
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import quaternion_to_matrix


def _field_offsets(fields) -> dict:
    return {f.name: f.offset for f in fields}


class TableCloudFilter(Node):
    """Crop + decimate the registered cloud into a light visualization topic."""

    def __init__(self):
        super().__init__("table_cloud_filter")
        self.declare_parameter("input_topic", "/camera/depth_registered/points")
        self.declare_parameter("output_topic", "/camera/points_table")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("x_min_m", 0.05)
        self.declare_parameter("x_max_m", 0.55)
        self.declare_parameter("y_min_m", -0.45)
        self.declare_parameter("y_max_m", 0.15)
        self.declare_parameter("z_min_m", -0.05)
        self.declare_parameter("z_max_m", 0.30)
        self.declare_parameter("voxel_m", 0.002)
        self.declare_parameter("max_points", 150000)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(PointCloud2, self._param("output_topic"), qos)
        self.create_subscription(
            PointCloud2, self._param("input_topic"), self._on_cloud, qos
        )
        self.get_logger().info(
            "table_cloud_filter: %s -> %s (voxel %.3f m)"
            % (
                self._param("input_topic"),
                self._param("output_topic"),
                float(self.get_parameter("voxel_m").value),
            )
        )

    def _param(self, name: str):
        return self.get_parameter(name).value

    def _on_cloud(self, msg: PointCloud2):
        try:
            cloud = self._filter(msg)
        except (ValueError, TransformException) as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        if cloud is not None:
            self._pub.publish(cloud)

    def _filter(self, msg: PointCloud2) -> PointCloud2 | None:
        offsets = _field_offsets(msg.fields)
        for need in ("x", "y", "z"):
            if need not in offsets:
                raise ValueError(f"input cloud lacks '{need}' field")
        has_rgb = "rgb" in offsets and msg.point_step >= offsets["rgb"] + 4
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [offsets["x"], offsets["y"], offsets["z"]],
                "itemsize": msg.point_step,
            }
        )
        pts = np.frombuffer(msg.data, dtype=dtype)
        xyz = np.stack([pts["x"].astype(np.float64),
                        pts["y"].astype(np.float64),
                        pts["z"].astype(np.float64)], axis=1)
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        rgb = raw[:, offsets["rgb"]:offsets["rgb"] + 4].copy() if has_rgb else None
        finite = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[finite]
        if rgb is not None:
            rgb = rgb[finite]
        if xyz.shape[0] == 0:
            return None
        # Transform cloud frame -> target frame via TF (translation + rotation).
        target = str(self._param("target_frame"))
        if msg.header.frame_id != target:
            tf = self._tf.lookup_transform(
                target, msg.header.frame_id, Time(),
                timeout=Duration(seconds=0.2),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rot = quaternion_to_matrix((q.x, q.y, q.z, q.w))
            xyz = xyz @ rot.T + np.array([t.x, t.y, t.z])
        box = (
            (xyz[:, 0] >= float(self.get_parameter("x_min_m").value))
            & (xyz[:, 0] <= float(self.get_parameter("x_max_m").value))
            & (xyz[:, 1] >= float(self.get_parameter("y_min_m").value))
            & (xyz[:, 1] <= float(self.get_parameter("y_max_m").value))
            & (xyz[:, 2] >= float(self.get_parameter("z_min_m").value))
            & (xyz[:, 2] <= float(self.get_parameter("z_max_m").value))
        )
        xyz = xyz[box]
        if rgb is not None:
            rgb = rgb[box]
        if xyz.shape[0] == 0:
            return None
        voxel = float(self.get_parameter("voxel_m").value)
        if voxel > 0:
            keys = np.floor(xyz / voxel).astype(np.int64)
            _, keep = np.unique(keys, axis=0, return_index=True)
            keep = np.sort(keep)
            xyz = xyz[keep]
            if rgb is not None:
                rgb = rgb[keep]
        max_points = int(self.get_parameter("max_points").value)
        if xyz.shape[0] > max_points:
            stride = xyz.shape[0] / max_points
            idx = (np.arange(max_points) * stride).astype(np.int64)
            xyz = xyz[idx]
            if rgb is not None:
                rgb = rgb[idx]
        out = PointCloud2()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = target
        out.height = 1
        out.width = int(xyz.shape[0])
        if rgb is not None:
            out.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            out.point_step = 16
            blob = np.empty((xyz.shape[0], 16), dtype=np.uint8)
            blob[:, 0:12] = np.ascontiguousarray(xyz.astype(np.float32)).view(np.uint8).reshape(-1, 12)
            blob[:, 12:16] = rgb
            out.data = blob.tobytes()
        else:
            out.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            out.point_step = 12
            out.data = xyz.astype(np.float32).tobytes()
        out.is_bigendian = False
        out.row_step = out.point_step * out.width
        out.is_dense = True
        return out


def main(args=None):
    rclpy.init(args=args)
    node = TableCloudFilter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
