#!/usr/bin/env python3
"""MOCK ONLY - simulated scene camera for end-to-end tests of the calibration workflow.  NOT FOR REAL ROBOTS.

Publishes what the Orbbec driver would publish for the Gemini 336 scene camera (all in the 'camera' namespace of
rebel_cameras): /camera/color/image_raw (1920x1080 rgb8), /camera/color/camera_info and the static TF
camera_link -> camera_color_optical_frame.  The image is RENDERED with OpenCV from a hidden ground truth
(config/mock_truth.yaml: camera pose, arm_right base pose, target mounts on link6, table board pose): the printed
ChArUco boards (same renderer as the print files) are warped into the image, with lens distortion, blur and noise, at
the pose that the LIVE mock arm's TF currently has.  The recorder's real detector therefore sees real-looking images.

    # single-arm stack (rebel.launch.py hardware_protocol:=mock_hardware): frames base_link -> link6
    ros2 run rebel_dual_arm_calibration mock_scene_camera.py --ros-args -p arm:=left     # this arm is arm_left
    ros2 run rebel_dual_arm_calibration mock_scene_camera.py --ros-args -p arm:=right    # this arm is arm_right
    # dual cell (rebel_dual_table_description dual_table.launch.py): arm_left_/arm_right_ frames, both boards
    ros2 run rebel_dual_arm_calibration mock_scene_camera.py --ros-args -p frame_mode:=dual

Parameters: arm (left|right, single mode), frame_mode (single|dual), truth_file, targets_file, rate_hz (5),
base_frame / tip_frame (single mode; defaults base_link / link6), seed.
The arm's own links never occlude the boards in this mock; the arm TF is the only input from the robot.
Never run this next to a real camera driver on the same ROS domain (it would publish the same topics).
"""
import array
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rebel_dual_arm_calibration.mock_common import SceneRenderer, load_truth
from rebel_dual_arm_calibration.transforms import R_to_quat, make_T, quat_to_R


def tf_to_T(tf: TransformStamped) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


class MockSceneCamera(Node):
    def __init__(self):
        super().__init__("mock_scene_camera")
        P = self.declare_parameter
        P("arm", "left")
        P("frame_mode", "single")
        P("truth_file", "")
        P("targets_file", "")
        P("rate_hz", 5.0)
        P("base_frame", "base_link")
        P("tip_frame", "link6")
        P("seed", 1)
        P("camera_link_frame", "camera_link")
        P("optical_frame", "camera_color_optical_frame")
        P("image_topic", "/camera/color/image_raw")
        P("camera_info_topic", "/camera/color/camera_info")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.truth = load_truth(g("truth_file") or None)
        self.renderer = SceneRenderer(self.truth, g("targets_file") or None, seed=int(g("seed")))
        mode = g("frame_mode")
        if mode == "dual":
            self.arms = {"left": ("arm_left_base_link", "arm_left_link6"),
                         "right": ("arm_right_base_link", "arm_right_link6")}
        elif mode == "single":
            if g("arm") not in ("left", "right"):
                raise ValueError("arm must be left or right")
            self.arms = {g("arm"): (g("base_frame"), g("tip_frame"))}
        else:
            raise ValueError("frame_mode must be single or dual")
        self.optical = g("optical_frame")
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static(g("camera_link_frame"))
        # RELIABLE publisher (compatible with the best-effort subscribers): 6 MB best-effort frames are dropped by
        # the UDP transport on this host (about 2 of 5 frames), reliable ones are retransmitted
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_img = self.create_publisher(Image, g("image_topic"), qos)
        self.pub_info = self.create_publisher(CameraInfo, g("camera_info_topic"), qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self._tick)
        self.n = 0
        self.get_logger().info(f"MOCK scene camera (NOT a real camera): {mode} mode, arms {self.arms}, "
                               f"{self.truth.size[0]}x{self.truth.size[1]}, truth {g('truth_file') or 'default'}")

    def _publish_static(self, parent: str):
        T = self.truth.T_cl_opt
        q = R_to_quat(T[:3, :3])
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = parent, self.optical
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in T[:3, 3])
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (
            float(v) for v in q)
        self.static_tf.sendTransform(m)

    def _tick(self):
        stamp = self.get_clock().now()          # taken BEFORE the TF read: the arm pose is at or after the stamp
        targets = {"table": self.truth.T_w_board}
        missing = []
        for arm, (base, tip) in self.arms.items():
            try:
                tf = self.tf_buffer.lookup_transform(base, tip, Time(), timeout=Duration(seconds=0.0))
            except Exception as e:  # noqa: BLE001
                missing.append(f"{base}->{tip}: {e}")
                continue
            targets[arm] = self.truth.T_w_base(arm) @ tf_to_T(tf) @ self.truth.T_l6_target[arm]
        if missing and self.n % 25 == 0:
            self.get_logger().warn("arm TF unavailable, board not drawn: " + "; ".join(missing))
        t0 = time.monotonic()
        rgb = self.renderer.render(targets)
        dt = time.monotonic() - t0
        msg = Image()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.optical
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        msg.data = array.array("B", rgb.tobytes())   # fast path of the Humble setter (a bytes value is validated per byte)
        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = msg.width, msg.height
        info.distortion_model = "plumb_bob"
        info.d = [float(v) for v in self.truth.D]
        info.k = [float(v) for v in self.truth.K.reshape(-1)]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(v) for v in np.hstack([self.truth.K, np.zeros((3, 1))]).reshape(-1)]
        t1 = time.monotonic()
        self.pub_info.publish(info)
        self.pub_img.publish(msg)
        self.n += 1
        if self.n % 50 == 1:
            self.get_logger().info(f"frame {self.n}: render {dt * 1e3:.0f} ms, message {1e3 * (t1 - t0 - dt):.0f} ms, "
                                   f"publish {1e3 * (time.monotonic() - t1):.0f} ms, boards drawn {sorted(targets)}")


def main(args=None):
    rclpy.init(args=args)
    node = MockSceneCamera()
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
