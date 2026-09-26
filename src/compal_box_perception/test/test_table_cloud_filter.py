"""Unit checks for the table-region cloud filter (no ROS spin)."""

import math

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from compal_box_perception.table_cloud_filter import TableCloudFilter


def _cloud(points, frame="camera_color_optical_frame"):
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points)
    msg.is_dense = True
    msg.data = np.asarray(points, dtype=np.float32).tobytes()
    return msg


def _node():
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    return TableCloudFilter()


def test_crop_keeps_table_points_only():
    node = _node()
    try:
        pts = [
            (0.30, -0.10, 0.05),  # inside default box
            (5.00, 0.00, 0.00),  # far outside
            (0.30, -0.10, 5.00),  # above
        ]
        out = node._filter(_cloud(pts, frame="base_link"))
        assert out is not None
        assert out.width == 1
        assert out.header.frame_id == "base_link"
        xyz = np.frombuffer(out.data, dtype=np.float32).reshape(-1, 3)
        assert math.dist(xyz[0], (0.30, -0.10, 0.05)) < 1e-6
    finally:
        node.destroy_node()


def test_voxel_decimation_bounds_output():
    node = _node()
    try:
        rng = np.random.default_rng(0)
        pts = np.column_stack(
            [
                rng.uniform(0.1, 0.5, 20000),
                rng.uniform(-0.4, 0.1, 20000),
                rng.uniform(0.0, 0.2, 20000),
            ]
        )
        out = node._filter(_cloud(pts, frame="base_link"))
        assert out is not None
        # 2 cm voxels over a 0.4*0.5*0.2 box caps well below input count
        assert out.width < 20000
        assert out.width > 0
    finally:
        node.destroy_node()


def test_empty_after_crop_returns_none():
    node = _node()
    try:
        out = node._filter(_cloud([(9.0, 9.0, 9.0)], frame="base_link"))
        assert out is None
    finally:
        node.destroy_node()


def _cloud_rgb(points, rgbs, frame="camera_color_optical_frame"):
    import struct

    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = 20 * len(points)
    msg.is_dense = True
    blob = bytearray()
    for (x, y, z), (r, g, b) in zip(points, rgbs):
        blob += struct.pack("<fff", x, y, z)
        blob += bytes([0, 0, 0, 0])
        blob += struct.pack("<f", struct.unpack("<f", bytes([b, g, r, 0]))[0])
    msg.data = bytes(blob)
    return msg


def test_rgb_preserved_through_crop_and_voxel():
    node = _node()
    try:
        pts = [(0.30, -0.10, 0.05), (0.31, -0.10, 0.05), (9.0, 9.0, 9.0)]
        rgbs = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        out = node._filter(_cloud_rgb(pts, rgbs, frame="base_link"))
        assert out is not None
        names = [f.name for f in out.fields]
        assert "rgb" in names
        assert out.point_step == 16
        raw = np.frombuffer(out.data, dtype=np.uint8).reshape(-1, 16)
        # voxel keeps one point per 2cm cell: the two in-box points share x/y cells
        # 10mm apart in x only -> same voxel -> first (red) survives; far point cropped
        assert out.width in (1, 2)
        colors = {tuple(row) for row in raw[:, 12:16]}
        assert colors <= {(0, 0, 255, 0), (0, 255, 0, 0)}  # BGRx red / green
        assert (0, 0, 255, 0) in colors
    finally:
        node.destroy_node()
