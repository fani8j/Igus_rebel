#!/usr/bin/env python3
"""MOCK ONLY - simulated wrist (eye-in-hand) camera for end-to-end tests of the wrist hand-eye UI.  NOT FOR REAL ROBOTS.

Publishes /wrist_camera/color/image_raw (rgb8, 1280x800) and /wrist_camera/color/camera_info the way the Orbbec driver would,
with the table board RENDERED at the pose that follows from the LIVE mock arm TF (base_link -> link6), a hidden ground-truth
mount ``truth_xyz`` / ``truth_rpy_deg`` (link6 -> optical) and the board pose ``board_xyz`` / ``board_rpy_deg`` (board frame:
origin = board centre, z INTO the table).  The calibration should recover truth_* from these images.

    ros2 run rebel_dual_arm_calibration mock_wrist_camera.py --ros-args -p board_xyz:="[0.514,-0.148,-0.251]"

Refuses to start unless /controller_manager reports mock_components/GenericSystem for every hardware component.
Never run it next to a real wrist camera driver on the same ROS domain (same topics).
"""
import sys
import time

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListHardwareComponents
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rebel_dual_arm_calibration.mock_common import BoardImage
from rebel_dual_arm_calibration.recorder_node import tf_msg_to_T
from rebel_dual_arm_calibration.transforms import inv_T, xyz_rpy_to_T
from rebel_dual_arm_calibration.wrist_synthetic import WRIST_K, WRIST_SIZE, table_spec


class MockWristCamera(Node):
    def __init__(self):
        super().__init__("mock_wrist_camera")
        P = self.declare_parameter
        P("board_xyz", [0.514, -0.148, -0.251])
        P("board_rpy_deg", [180.0, 0.0, 20.0])
        P("truth_xyz", [-0.020, -0.090, -0.040])
        P("truth_rpy_deg", [10.0, -5.0, 95.0])
        P("rate_hz", 5.0)
        P("frame_id", "wrist_camera_color_optical_frame")
        P("base_frame", "base_link")
        P("tip_frame", "link6")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.B = xyz_rpy_to_T(g("board_xyz"), np.radians(g("board_rpy_deg")))
        self.X = xyz_rpy_to_T(g("truth_xyz"), np.radians(g("truth_rpy_deg")))
        self.base, self.tip, self.frame = g("base_frame"), g("tip_frame"), g("frame_id")
        self.board = BoardImage(table_spec(), margin=0.010, plate=0.0)
        self.rng = np.random.default_rng(3)
        self.pub = self.create_publisher(Image, "/wrist_camera/color/image_raw", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/wrist_camera/color/camera_info", qos_profile_sensor_data)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self, spin_thread=True)     # TF must not queue behind the (slow) rendering
        # like the Orbbec driver: wrist_camera_link -> optical frame (identity on the real camera); link6 -> wrist_camera_link
        # is what the CALIBRATION will provide, so it is NOT published here
        self.static = StaticTransformBroadcaster(self)
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id, ts.child_frame_id = "wrist_camera_link", self.frame
        ts.transform.rotation.w = 1.0
        self.static.sendTransform(ts)
        self.create_timer(1.0 / float(g("rate_hz")), self.tick)
        self.get_logger().warning("MOCK wrist camera: publishes SYNTHETIC images. truth mount xyz=%s rpy_deg=%s" %
                                  (g("truth_xyz"), g("truth_rpy_deg")))

    def check_mock(self, timeout=10.0) -> bool:
        cli = self.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components")
        if not cli.wait_for_service(timeout):
            return False
        fut = cli.call_async(ListHardwareComponents.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not fut.done() or not fut.result().component:
            return False
        return all(c.class_type == "mock_components/GenericSystem" for c in fut.result().component)

    def tick(self):
        try:
            tf = self.buf.lookup_transform(self.base, self.tip, Time())
        except Exception:  # noqa: BLE001
            return
        A = tf_msg_to_T(tf)
        stamp_msg = tf.header.stamp      # the image is stamped with the pose it was rendered from (like a real capture time)
        w, h = WRIST_SIZE
        frame = np.full((h, w, 3), 135, np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        frame = np.clip(frame + (25 * (yy / h) - 10)[:, :, None], 0, 255).astype(np.uint8)
        self.board.warp_into(frame, WRIST_K, inv_T(A @ self.X) @ self.B)
        import cv2
        frame = cv2.GaussianBlur(frame, (0, 0), 0.7)
        frame = np.clip(frame.astype(np.float32) + self.rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = stamp_msg
        m = Image()
        m.header.stamp, m.header.frame_id = now, self.frame
        m.height, m.width, m.encoding, m.step = h, w, "rgb8", w * 3
        m.data = rgb.tobytes()
        self.pub.publish(m)
        ci = CameraInfo()
        ci.header = m.header
        ci.height, ci.width = h, w
        ci.k = [float(v) for v in WRIST_K.ravel()]
        ci.d = [0.0] * 5
        ci.distortion_model = "plumb_bob"
        ci.p = [WRIST_K[0, 0], 0.0, WRIST_K[0, 2], 0.0, 0.0, WRIST_K[1, 1], WRIST_K[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.pub_info.publish(ci)


def main():
    rclpy.init()
    node = MockWristCamera()
    if not node.check_mock():
        node.get_logger().error("hardware is not (confirmed to be) mock_components/GenericSystem: refusing to start")
        sys.exit(2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
