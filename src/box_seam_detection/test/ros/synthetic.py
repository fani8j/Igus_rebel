"""Synthetic camera publisher for tests: small colour + aligned depth + camera_info with controllable stamps."""

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

W, H = 64, 48
K = [60.0, 0.0, 31.5, 0.0, 61.0, 23.5, 0.0, 0.0, 1.0]


def make_images(w=W, h=H, seed=1):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    depth = rng.integers(400, 1500, (h, w)).astype(np.uint16)
    depth[:4, :] = 0  # invalid pixels
    return rgb, depth


class SyntheticCamera(Node):
    def __init__(self, prefix="/camera", rate=20.0, stamp_offset_s=0.0, frame="camera_color_optical_frame",
                 encoding="rgb8", depth_encoding="16UC1", depth_size=None, info_size=None, auto=True,
                 aligned_every=None):
        super().__init__("synthetic_camera")
        self.offset = stamp_offset_s
        self.frame = frame
        self.encoding = encoding
        self.depth_encoding = depth_encoding
        self.rgb, self.depth = make_images()
        # aligned_every=N: only every Nth frame carries the aligned depth, the others the depth_size (unaligned) one
        self.aligned_every, self.ticks, self.aligned_depth = aligned_every, 0, self.depth
        if depth_size:
            self.depth = np.zeros((depth_size[1], depth_size[0]), dtype=np.uint16) + 800
        self.info_size = info_size or (W, H)
        self.cpub = self.create_publisher(Image, prefix + "/color/image_raw", qos_profile_sensor_data)
        self.dpub = self.create_publisher(Image, prefix + "/depth/image_raw", qos_profile_sensor_data)
        self.ipub = self.create_publisher(CameraInfo, prefix + "/color/camera_info", qos_profile_sensor_data)
        if auto:
            self.create_timer(1.0 / rate, self.tick)

    def tick(self):
        stamp = (self.get_clock().now() + rclpy_duration(self.offset)).to_msg()
        img = self.rgb if self.encoding == "rgb8" else self.rgb[:, :, ::-1]
        c = Image(height=H, width=W, encoding=self.encoding, step=W * 3, data=np.ascontiguousarray(img).tobytes())
        c.header.stamp, c.header.frame_id = stamp, self.frame
        self.ticks += 1
        depth = self.aligned_depth if self.aligned_every and self.ticks % self.aligned_every == 0 else self.depth
        dh, dw = depth.shape
        if self.depth_encoding == "32FC1":
            data, step = (depth.astype(np.float32) * 0.001).tobytes(), dw * 4
        else:
            data, step = depth.tobytes(), dw * 2
        d = Image(height=dh, width=dw, encoding=self.depth_encoding, step=step, data=data)
        d.header.stamp, d.header.frame_id = stamp, self.frame
        i = CameraInfo(width=self.info_size[0], height=self.info_size[1], distortion_model="plumb_bob",
                       d=[0.0] * 5, k=K, r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                       p=[K[0], 0.0, K[2], 0.0, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0])
        i.header.stamp, i.header.frame_id = stamp, self.frame
        self.cpub.publish(c)
        self.dpub.publish(d)
        self.ipub.publish(i)


def rclpy_duration(seconds):
    from rclpy.duration import Duration
    return Duration(nanoseconds=int(seconds * 1e9))
