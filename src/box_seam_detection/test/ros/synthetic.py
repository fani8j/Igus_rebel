"""Synthetic camera publisher for tests: small colour + aligned depth + camera_info with controllable stamps."""

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

W, H = 64, 48
K = [60.0, 0.0, 31.5, 0.0, 61.0, 23.5, 0.0, 0.0, 1.0]
# raw (unaligned) depth camera: smaller image, own optical frame, 10 mm to the left of the colour camera
RW, RH = 40, 30
K_RAW = [40.0, 0.0, 19.5, 0.0, 40.0, 14.5, 0.0, 0.0, 1.0]
RAW_FRAME = "camera_depth_optical_frame"
RAW_MM = 1000           # the raw depth plane (mm); rows 0-2 are invalid
BASELINE_X = -0.010     # translation of T_color_depth


def make_images(w=W, h=H, seed=1):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    depth = rng.integers(400, 1500, (h, w)).astype(np.uint16)
    depth[:4, :] = 0  # invalid pixels
    return rgb, depth


class SyntheticCamera(Node):
    def __init__(self, prefix="/camera", rate=20.0, stamp_offset_s=0.0, frame="camera_color_optical_frame",
                 encoding="rgb8", depth_encoding="16UC1", depth_size=None, info_size=None, auto=True,
                 aligned_every=None, raw_size=None, raw_stamp_offset_s=0.0, raw_every=1, raw_frame=RAW_FRAME,
                 raw_info=True):
        """raw_size=(w, h): also publish /depth/image_unaligned (16UC1 plane at RAW_MM, own frame) and /depth/camera_info
        (the raw-size info plus the colour-size 'aligned' info, both labelled with the colour frame like the driver).
        raw_info=False: only the aligned-size info (what the driver publishes while its alignment works)."""
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
        self.raw_size, self.raw_offset, self.raw_every, self.raw_frame = raw_size, raw_stamp_offset_s, raw_every, raw_frame
        self.raw_info = raw_info
        if raw_size:
            self.rpub = self.create_publisher(Image, prefix + "/depth/image_unaligned", qos_profile_sensor_data)
            self.dipub = self.create_publisher(CameraInfo, prefix + "/depth/camera_info", qos_profile_sensor_data)
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
        if self.raw_size and self.ticks % self.raw_every == 0:
            rw, rh = self.raw_size
            raw = np.full((rh, rw), RAW_MM, dtype=np.uint16)
            raw[:3, :] = 0
            r = Image(height=rh, width=rw, encoding="16UC1", step=rw * 2, data=raw.tobytes())
            r.header.stamp = (self.get_clock().now() + rclpy_duration(self.offset + self.raw_offset)).to_msg()
            r.header.frame_id = self.raw_frame
            kr = [K_RAW[0] * rw / RW, 0.0, K_RAW[2] * rw / RW, 0.0, K_RAW[4] * rh / RH, K_RAW[5] * rh / RH,
                  0.0, 0.0, 1.0]
            ri = CameraInfo(width=rw, height=rh, distortion_model="plumb_bob", d=[0.0] * 5, k=kr)
            ai = CameraInfo(width=W, height=H, distortion_model="plumb_bob", d=[0.0] * 5, k=K)
            for m in (ri, ai):
                m.header.stamp, m.header.frame_id = r.header.stamp, self.frame    # the driver mislabels the raw info
            self.rpub.publish(r)
            self.dipub.publish(ai)
            if self.raw_info:
                self.dipub.publish(ri)


def rclpy_duration(seconds):
    from rclpy.duration import Duration
    return Duration(nanoseconds=int(seconds * 1e9))


def static_tf_buffer(parent="camera_color_optical_frame", child=RAW_FRAME, translation=(BASELINE_X, 0.0, 0.0)):
    """A tf2 buffer holding the static colour <- raw depth transform (pass it as SceneCapture(tf_buffer=...))."""
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    buf = tf2_ros.Buffer()
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = translation
    t.transform.rotation.w = 1.0
    buf.set_transform_static(t, "test")
    return buf
