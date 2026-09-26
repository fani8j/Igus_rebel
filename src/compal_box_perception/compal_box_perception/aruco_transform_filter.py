"""Robust temporal filtering for an ArUco marker transform."""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from .geometry import average_quaternions, normalize_quaternion, quaternion_angle_deg


class TransformMedianFilter:
    """Reject pose outliers and median-filter translation and orientation."""

    def __init__(
        self,
        window_size: int,
        max_position_jump_m: float,
        max_rotation_jump_deg: float,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if max_position_jump_m <= 0.0:
            raise ValueError("max_position_jump_m must be positive")
        if not 0.0 < max_rotation_jump_deg <= 180.0:
            raise ValueError("max_rotation_jump_deg must be in (0, 180]")
        self._translations: deque[np.ndarray] = deque(maxlen=window_size)
        self._quaternions: deque[np.ndarray] = deque(maxlen=window_size)
        self.max_position_jump_m = max_position_jump_m
        self.max_rotation_jump_deg = max_rotation_jump_deg

    def update(
        self, translation: Sequence[float], quaternion: Sequence[float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        position = np.asarray(translation, dtype=np.float64).reshape(3)
        rotation = normalize_quaternion(quaternion)
        if not np.all(np.isfinite(position)):
            raise ValueError("Translation must be finite")
        if self._translations:
            reference_position = np.median(np.stack(self._translations), axis=0)
            reference_rotation = average_quaternions(self._quaternions)
            if np.linalg.norm(position - reference_position) > self.max_position_jump_m:
                return None
            if quaternion_angle_deg(rotation, reference_rotation) > self.max_rotation_jump_deg:
                return None
        self._translations.append(position)
        self._quaternions.append(rotation)
        return (
            np.median(np.stack(self._translations), axis=0),
            average_quaternions(self._quaternions),
        )


class ArucoTransformFilterNode(Node):
    """Publish a filtered ArUco marker TF under a distinct child-frame name."""

    def __init__(self) -> None:
        super().__init__("aruco_transform_filter")
        self.declare_parameter("source_frame", "camera_color_optical_frame")
        self.declare_parameter("marker_frame", "marker_13")
        self.declare_parameter("filtered_frame", "marker_13_filtered")
        self.declare_parameter("window_size", 7)
        self.declare_parameter("max_position_jump_m", 0.010)
        self.declare_parameter("max_rotation_jump_deg", 5.0)

        self._source_frame = str(self.get_parameter("source_frame").value)
        self._marker_frame = str(self.get_parameter("marker_frame").value)
        self._filtered_frame = str(self.get_parameter("filtered_frame").value)
        if not all((self._source_frame, self._marker_frame, self._filtered_frame)):
            raise ValueError("source_frame, marker_frame, and filtered_frame are required")
        if self._filtered_frame == self._marker_frame:
            raise ValueError("filtered_frame must differ from marker_frame")
        self._filter = TransformMedianFilter(
            int(self.get_parameter("window_size").value),
            float(self.get_parameter("max_position_jump_m").value),
            float(self.get_parameter("max_rotation_jump_deg").value),
        )
        self._broadcaster = TransformBroadcaster(self)
        self._subscription = self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.get_logger().info(
            f"Filtering {self._source_frame} -> {self._marker_frame} as "
            f"{self._source_frame} -> {self._filtered_frame}"
        )

    def _on_tf(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if (
                transform.header.frame_id != self._source_frame
                or transform.child_frame_id != self._marker_frame
            ):
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            try:
                filtered = self._filter.update(
                    (translation.x, translation.y, translation.z),
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                )
            except ValueError as error:
                self.get_logger().warning(f"Ignoring invalid marker transform: {error}")
                continue
            if filtered is None:
                self.get_logger().debug("Rejected marker transform outside configured bounds")
                continue
            position, quaternion = filtered
            output = TransformStamped()
            output.header = transform.header
            output.child_frame_id = self._filtered_frame
            output.transform.translation.x = float(position[0])
            output.transform.translation.y = float(position[1])
            output.transform.translation.z = float(position[2])
            output.transform.rotation.x = float(quaternion[0])
            output.transform.rotation.y = float(quaternion[1])
            output.transform.rotation.z = float(quaternion[2])
            output.transform.rotation.w = float(quaternion[3])
            self._broadcaster.sendTransform(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoTransformFilterNode()
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
