#!/usr/bin/env python3
# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/rgbd_recorder.py
# at commit 08d1041 (working tree identical to HEAD, unmodified).
# -*- coding: utf-8 -*-
"""
rgbd_recorder.py

ROS2 節點：訂閱由深度攝影機（例如 Orbbec astra/gemini 系列，orbbec_camera driver）
發布的 RGB 影像 topic 與點雲 topic，將兩者依時間戳同步後，依固定頻率
（預設每秒 1 組）存成 PNG 影像檔與 PCD 點雲檔到指定資料夾。

輸出結構：
    <output_dir>/
        rgb/frame_000001_20260821_153000_123456.png
        pointcloud/frame_000001_20260821_153000_123456.pcd

必要相依套件（ROS2 Jazzy / Humble 皆適用，把 jazzy 換成你的 distro 即可）：
    sudo apt install ros-jazzy-cv-bridge ros-jazzy-message-filters \
                      ros-jazzy-vision-opencv python3-opencv python3-numpy

使用方式：

  1) 先啟動深度攝影機的 driver（依你的相機而定，例如）：
         ros2 launch orbbec_camera <你的相機型號>.launch.py

  2) 再執行本程式，指定輸出資料夾與 topic 名稱（依實際 driver 輸出的 topic 調整）：
         python3 rgbd_recorder.py /path/to/output_folder \
             --ros-args \
             -p rgb_topic:=/camera/color/image_raw \
             -p pointcloud_topic:=/camera/depth/points \
             -p save_rate_hz:=1.0

     也可以完全用 ROS2 參數指定輸出資料夾（優先權高於位置參數）：
         python3 rgbd_recorder.py --ros-args -p output_dir:=/path/to/output_folder

     只想拍「一張」照片（存一組 RGB+點雲後自動結束），加上 --once：
         python3 rgbd_recorder.py /path/to/output_folder --once \
             --ros-args -p rgb_topic:=/camera/color/image_raw \
                        -p pointcloud_topic:=/camera/depth/points

     或用 ROS 參數指定要拍幾張後自動結束（--once 等同 max_frames:=1）：
         python3 rgbd_recorder.py --ros-args -p output_dir:=/path/to/output_folder -p max_frames:=1

     max_frames == 1（--once）時，第一組同步的 RGB+點雲（且已收到 CameraInfo）
     一到就立即存檔，不等存檔 timer；同步/時間戳規則與 timer 存檔完全相同。
     max_frames > 1（連拍）仍依 save_rate_hz 的 timer 存檔，行為不變。

     逾時（--timeout-sec，預設：--once 20 秒；連拍 max(20, 2/save_rate_hz + 5) 秒；
     持續錄製 0=不逾時）：
     若在期限內（自啟動或上一次存檔起算）沒有存下任何一組完整 capture，
     程式在 stderr 印出哪些 topic 沒有收到訊息，並以 exit code 3 結束。

  3) 用 `ros2 topic list` 確認實際的 RGB / 點雲 topic 名稱與型別
     （RGB 應為 sensor_msgs/msg/Image，點雲應為 sensor_msgs/msg/PointCloud2），
     若與預設值不同，用 -p rgb_topic:=... / -p pointcloud_topic:=... 覆蓋。

備註：
  - 若點雲 topic 是「彩色點雲」（例如 /camera/depth_registered/points），
    訊息中會多一個 rgb 欄位，程式會自動偵測並把顏色一併存進 PCD 檔。
  - 若 driver 分別發布顏色/深度且未對齊，兩者可能不是彩色點雲；
    此時點雲檔僅包含 XYZ，不含顏色，這是正常現象，需在攝影機 driver
    的啟動參數中開啟 point cloud 對齊/上色功能才會有 rgb 欄位。
  - 存下來的 PCD 是否為「organized」（PCD header 的 HEIGHT > 1，點的排列
    順序跟 RGB 影像的掃描順序一一對應，可以直接用 (u, v) 像素座標索引），
    完全取決於來源 PointCloud2 訊息本身的 msg.height 是否 > 1；本程式會
    如實保留訊息原本的 WIDTH/HEIGHT 與點的順序（包含無效點的 NaN），不會
    額外過濾或攤平，這樣才能配合 query_depth_at_pixel.py 用像素座標查詢。
    如果訊息本身就是 unorganized（height=1，例如某些 driver 的原始點雲），
    存出來的 PCD 也會是 HEIGHT 1，這是正常現象，不是程式的錯誤。
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import threading
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import cv2

try:                                   # ROS is only needed to RUN the recorder node; capture_once()
    import rclpy                       # below (subprocess wrapper) works without importing it
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from rclpy.qos import qos_profile_sensor_data

    import message_filters
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from sensor_msgs_py import point_cloud2
    from cv_bridge import CvBridge
except ImportError as _ros_import_error:   # pragma: no cover - environment without ROS
    rclpy = None
    Node = object
    _ROS_IMPORT_ERROR = _ros_import_error
else:
    _ROS_IMPORT_ERROR = None


DEFAULT_TIMEOUT_SEC = 20.0  # --timeout-sec default for finite runs (max_frames > 0)
EXIT_TIMEOUT = 3            # exit code: no synced capture saved within --timeout-sec


def default_timeout_sec(max_frames, save_rate_hz):
    """--timeout-sec default: 0 (wait forever) for continuous recording,
    DEFAULT_TIMEOUT_SEC for --once, and for bursts (max_frames > 1) at least
    two save periods plus 5 s slack, so a slow save timer (low save_rate_hz)
    cannot trip the timeout between two committed captures."""
    if max_frames <= 0:
        return 0.0
    if max_frames == 1:
        return DEFAULT_TIMEOUT_SEC
    return max(DEFAULT_TIMEOUT_SEC, 2.0 / max(float(save_rate_hz), 1e-6) + 5.0)


class RgbdRecorder(Node):
    def __init__(self, cli_output_dir=None, cli_max_frames=None, cli_timeout_sec=None):
        super().__init__("rgbd_recorder")

        default_output_dir = cli_output_dir or os.path.expanduser("~/rgbd_captures")
        default_max_frames = cli_max_frames if cli_max_frames is not None else 0

        # ---- 可用 --ros-args -p key:=value 覆蓋的參數 ----
        self.declare_parameter("output_dir", default_output_dir)
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("pointcloud_topic", "/camera/depth/points")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("save_rate_hz", 1.0)     # 每秒存幾組 RGB+點雲
        self.declare_parameter("sync_slop", 0.05)        # 影像與點雲時間戳同步允許誤差(秒)
        self.declare_parameter("filename_prefix", "frame")
        self.declare_parameter("pcd_binary", True)        # True: 存binary PCD(較快/較小); False: ASCII
        self.declare_parameter("max_frames", default_max_frames)  # 0=持續錄製；N>0=存滿N組後自動結束

        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        pc_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        save_rate_hz = self.get_parameter("save_rate_hz").get_parameter_value().double_value
        sync_slop = self.get_parameter("sync_slop").get_parameter_value().double_value
        self.prefix = self.get_parameter("filename_prefix").get_parameter_value().string_value
        self.pcd_binary = self.get_parameter("pcd_binary").get_parameter_value().bool_value
        self.max_frames = self.get_parameter("max_frames").get_parameter_value().integer_value

        self.save_period = 1.0 / max(save_rate_hz, 1e-6)
        self.done = False  # 設為 True 時，main() 的迴圈會結束並關閉節點
        # --once (max_frames == 1): save the first synced pair as soon as it
        # arrives instead of waiting for the save timer. Bursts keep the timer.
        self.save_immediately = self.max_frames == 1
        # Timeout (seconds without a committed capture); None -> the
        # default_timeout_sec() for this run (explicit --timeout-sec wins).
        if cli_timeout_sec is None:
            cli_timeout_sec = default_timeout_sec(self.max_frames, save_rate_hz)
        self.timeout_sec = max(float(cli_timeout_sec), 0.0)
        self._deadline_start = time.monotonic()

        # A manifest is committed only after both data files are committed.
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.pcd_dir = os.path.join(self.output_dir, "pointcloud")
        self.manifest_dir = os.path.join(self.output_dir, "captures")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_pair = None  # (Image msg, PointCloud2 msg)
        self._latest_camera_info = None
        self._frame_count = 0
        self._rgb_topic = rgb_topic
        self._pointcloud_topic = pc_topic
        self._camera_info_topic = camera_info_topic
        # Messages received since start / the last committed capture (timeout diagnostics).
        self._message_counts = {"rgb": 0, "pointcloud": 0, "camera_info": 0, "synced_pairs": 0}

        # ---- 訂閱並同步 RGB 影像與點雲（依時間戳做近似同步） ----
        rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
        rgb_sub.registerCallback(lambda _msg: self._count_message("rgb"))
        pc_sub.registerCallback(lambda _msg: self._count_message("pointcloud"))
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, pc_sub], queue_size=10, slop=sync_slop
        )
        self.ts.registerCallback(self._sync_callback)

        # ---- 依固定頻率執行存檔（存檔頻率與相機發布頻率脫鉤，避免存太多檔） ----
        self.timer = self.create_timer(self.save_period, self._save_timer_callback)

        mode_desc = f"存滿 {self.max_frames} 組後自動結束" if self.max_frames > 0 else "持續錄製"
        if self.save_immediately:
            mode_desc += "，第一組同步資料一到即存檔"
        if self.timeout_sec > 0:
            mode_desc += f"，{self.timeout_sec:g} 秒內未存檔則以 exit code {EXIT_TIMEOUT} 結束"
        self.get_logger().info(
            f"訂閱 RGB topic: '{rgb_topic}'，點雲 topic: '{pc_topic}'，CameraInfo topic: "
            f"'{camera_info_topic}'，每 {self.save_period:.2f} 秒存一組資料到 "
            f"'{self.output_dir}'（{mode_desc}）"
        )

    def _count_message(self, key):
        with self._lock:
            self._message_counts[key] += 1

    def _sync_callback(self, rgb_msg: Image, pc_msg: PointCloud2):
        with self._lock:
            self._latest_pair = (rgb_msg, pc_msg)
            self._message_counts["synced_pairs"] += 1
        if self.save_immediately:
            self._try_immediate_save()

    def _camera_info_callback(self, camera_info: CameraInfo):
        with self._lock:
            self._latest_camera_info = camera_info
            self._message_counts["camera_info"] += 1
        if self.save_immediately:
            self._try_immediate_save()

    def _try_immediate_save(self):
        """--once: commit the pending synced pair now (same checks as the timer)."""
        with self._lock:
            ready = (not self.done and self._latest_pair is not None
                     and self._latest_camera_info is not None)
        if not ready:
            return  # the pair stays pending; the timer or the next message retries
        if not self._save_pending_pair():
            # A failed commit falls back to the timer cadence (no per-frame retries).
            self.save_immediately = False

    def _save_timer_callback(self):
        if not self.done:
            self._save_pending_pair()

    def timed_out(self, now=None):
        """True when timeout_sec > 0 elapsed without a committed capture."""
        if self.timeout_sec <= 0 or self.done:
            return False
        now = time.monotonic() if now is None else now
        return now - self._deadline_start >= self.timeout_sec

    def timeout_message(self):
        with self._lock:
            counts = dict(self._message_counts)
        topics = (("rgb", self._rgb_topic), ("pointcloud", self._pointcloud_topic),
                  ("camera_info", self._camera_info_topic))
        silent = [f"{topic} ({key})" for key, topic in topics if counts[key] == 0]
        since = "since start" if self._frame_count == 0 else "since the last saved capture"
        lines = [f"rgbd_recorder: timeout: no synced RGB-D capture saved within "
                 f"{self.timeout_sec:g} s {since} (saved {self._frame_count}/{self.max_frames}); "
                 f"exit code {EXIT_TIMEOUT}"]
        if silent:
            lines.append("rgbd_recorder: no messages on: " + ", ".join(silent))
        elif counts["synced_pairs"] == 0:
            lines.append("rgbd_recorder: all topics published but no RGB/point-cloud pair synced "
                         "within sync_slop")
        lines.append("rgbd_recorder: messages received " + since + ": "
                     + ", ".join(f"{key}={counts[key]}" for key in
                                 ("rgb", "pointcloud", "camera_info", "synced_pairs")))
        return "\n".join(lines)

    def _save_pending_pair(self):
        """Commit the latest synced pair (RGB PNG + PCD + manifest). Returns True on success."""
        with self._lock:
            pair = self._latest_pair
            camera_info = self._latest_camera_info
            self._latest_pair = None  # 避免下一輪重複存同一組資料

        if pair is None:
            self.get_logger().warn(
                "尚未收到同步的 RGB+點雲資料，略過本次存檔"
                "（請確認 topic 名稱正確，且攝影機 driver 已在發布資料）"
            )
            return False
        if camera_info is None:
            self.get_logger().warn("尚未收到 CameraInfo，拒絕寫入無法追溯的 RGB-D capture")
            return False

        rgb_msg, pc_msg = pair
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{self.prefix}_{self._frame_count + 1:06d}_{timestamp}"
        try:
            image_path = self._save_rgb(rgb_msg, base_name)
            pcd_path, point_count, stored_height = self._save_pointcloud(pc_msg, base_name)
            manifest_path = self._save_manifest(
                base_name, rgb_msg, pc_msg, camera_info, image_path, pcd_path, point_count, stored_height)
        except Exception as error:
            self.get_logger().error(f"儲存完整 RGB-D capture 失敗: {error}")
            return False

        self._frame_count += 1
        with self._lock:
            self._deadline_start = time.monotonic()
            self._message_counts = dict.fromkeys(self._message_counts, 0)
        organized = f"organized, HEIGHT={stored_height}" if stored_height > 1 else "unorganized, HEIGHT=1"
        self.get_logger().info(f"已儲存影像: {image_path}")
        self.get_logger().info(f"已儲存點雲: {pcd_path} ({point_count} 點，{organized})")
        self.get_logger().info(f"已儲存 capture manifest: {manifest_path}")
        if self.max_frames > 0 and self._frame_count >= self.max_frames:
            self.get_logger().info(f"已存滿 {self.max_frames} 組資料，程式即將結束")
            self.done = True
        return True

    def _save_rgb(self, msg: Image, base_name: str):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = os.path.join(self.rgb_dir, f"{base_name}.png")
        temporary_path = f"{path}.tmp-{uuid.uuid4().hex}.png"
        try:
            if not cv2.imwrite(temporary_path, cv_img):
                raise RuntimeError(f"cv2.imwrite failed for {path}")
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
        return path

    def _save_pointcloud(self, msg: PointCloud2, base_name: str):
        field_names_msg = [field.name for field in msg.fields]
        color_field = next((candidate for candidate in ("rgb", "rgba") if candidate in field_names_msg), None)
        fields = ("x", "y", "z", color_field) if color_field else ("x", "y", "z")
        points_struct = point_cloud2.read_points(msg, field_names=fields, skip_nans=False)

        path = os.path.join(self.pcd_dir, f"{base_name}.pcd")
        temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
        try:
            point_count, height = self._write_pcd(
                temporary_path, points_struct, color_field, self.pcd_binary, msg.width, msg.height)
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
        return path, point_count, height

    @staticmethod
    def _header_metadata(header):
        return {
            "frame_id": header.frame_id,
            "stamp_ns": int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec),
        }

    def _save_manifest(self, capture_id, rgb_msg, pc_msg, camera_info, image_path, pcd_path,
                       point_count, stored_height):
        rgb_header = self._header_metadata(rgb_msg.header)
        pointcloud_header = self._header_metadata(pc_msg.header)
        camera_info_header = self._header_metadata(camera_info.header)
        if rgb_header["frame_id"] != pointcloud_header["frame_id"]:
            raise RuntimeError(
                f"RGB frame '{rgb_header['frame_id']}' and point-cloud frame "
                f"'{pointcloud_header['frame_id']}' differ")
        if camera_info_header["frame_id"] != rgb_header["frame_id"]:
            raise RuntimeError(
                f"CameraInfo frame '{camera_info_header['frame_id']}' does not match RGB frame "
                f"'{rgb_header['frame_id']}'")
        manifest = {
            "schema": "rgbd_capture/v1",
            "capture_id": capture_id,
            "created_at_ns": time.time_ns(),
            "image": {
                "path": image_path, "width": rgb_msg.width, "height": rgb_msg.height,
                "encoding": rgb_msg.encoding, "header": rgb_header,
            },
            "pointcloud": {
                "path": pcd_path, "width": pc_msg.width, "height": pc_msg.height,
                "point_step": pc_msg.point_step, "row_step": pc_msg.row_step,
                "is_dense": pc_msg.is_dense, "header": pointcloud_header,
                "fields": [{"name": field.name, "offset": field.offset,
                            "datatype": field.datatype, "count": field.count} for field in pc_msg.fields],
                "point_count": point_count, "stored_height": stored_height,
            },
            "camera_info": {
                "width": camera_info.width, "height": camera_info.height,
                "distortion_model": camera_info.distortion_model, "d": list(camera_info.d),
                "k": list(camera_info.k), "r": list(camera_info.r), "p": list(camera_info.p),
                "header": camera_info_header,
            },
            "sync_delta_ns": abs(rgb_header["stamp_ns"] - pointcloud_header["stamp_ns"]),
            "topics": {"rgb": self._rgb_topic, "pointcloud": self._pointcloud_topic},
        }
        path = os.path.join(self.manifest_dir, f"{capture_id}.json")
        temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as output:
                json.dump(manifest, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
        return path

    @staticmethod
    def _write_pcd(path: str, points_struct, color_field, binary: bool, width: int, height: int):
        """將 sensor_msgs_py 讀出的結構化 numpy 陣列寫成標準 PCD 檔（不依賴 open3d/pcl）。

        width, height 直接沿用來源 PointCloud2 訊息的 msg.width / msg.height，
        如實反映訊息本身是否為 organized，不在這裡另外判斷或修改。
        回傳 (實際寫入的點數, height)。
        """
        n = len(points_struct)
        expected_n = int(width) * int(height)
        if n != expected_n:
            # 理論上 skip_nans=False 讀出的點數應該等於 width*height；
            # 如果對不上，仍照實際點數為準，但把 header 的 WIDTH/HEIGHT
            # 降級成 unorganized，避免寫出跟實際點數矛盾、無法讀取的 PCD。
            width, height = n, 1

        has_color = color_field is not None

        if has_color:
            fields = f"x y z {color_field}"
            size = "4 4 4 4"
            type_ = "F F F F"
            count = "1 1 1 1"
        else:
            fields = "x y z"
            size = "4 4 4"
            type_ = "F F F"
            count = "1 1 1"

        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            f"FIELDS {fields}\n"
            f"SIZE {size}\n"
            f"TYPE {type_}\n"
            f"COUNT {count}\n"
            f"WIDTH {width}\n"
            f"HEIGHT {height}\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\n"
            f"DATA {'binary' if binary else 'ascii'}\n"
        )

        # One preallocated (n, 3|4) float32 buffer, filled column by column:
        # the same values/bits as the former column_stack + astype chain
        # (same-dtype copies are bitwise, other dtypes use the same float32
        # cast), without three intermediate 27 MB copies.
        data = np.empty((n, 4 if has_color else 3), dtype=np.float32)
        data[:, 0] = np.asarray(points_struct["x"], dtype=np.float32)
        data[:, 1] = np.asarray(points_struct["y"], dtype=np.float32)
        data[:, 2] = np.asarray(points_struct["z"], dtype=np.float32)

        if has_color:
            rgb_raw = np.asarray(points_struct[color_field])
            if rgb_raw.dtype != np.float32:
                # rgb/rgba 欄位在點雲訊息中通常是 packed 成單一 float32 的顏色值
                rgb_raw = rgb_raw.astype(np.uint32).view(np.float32)
            data[:, 3] = rgb_raw.reshape(-1)

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            if binary:
                f.write(data)  # C-contiguous buffer, same bytes as data.tobytes() without the copy
            else:
                np.savetxt(f, data, fmt="%.6f")

        return n, height


def main(args=None):
    if _ROS_IMPORT_ERROR is not None:
        print(f"rgbd_recorder: ROS 2 python packages unavailable: {_ROS_IMPORT_ERROR}", file=sys.stderr)
        return 2
    raw_args = sys.argv if args is None else args

    parser = argparse.ArgumentParser(
        description="訂閱 ROS2 深度攝影機的 RGB 影像與點雲，依固定頻率存成 PNG+PCD 到指定資料夾"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="輸出資料夾路徑（也可用 --ros-args -p output_dir:=... 指定，兩者皆給時以 ROS 參數為準）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只拍一張（存一組 RGB+點雲）後自動結束，等同 -p max_frames:=1",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help=f"自啟動（或上一次存檔）起超過這麼多秒仍未存下完整 capture 就以 exit code "
             f"{EXIT_TIMEOUT} 結束，並在 stderr 列出沒有訊息的 topic；0=永遠等待。"
             f"預設：--once 為 {DEFAULT_TIMEOUT_SEC:g} 秒，連拍為 max({DEFAULT_TIMEOUT_SEC:g}, "
             f"2/save_rate_hz + 5) 秒，持續錄製為 0",
    )
    ns, _ = parser.parse_known_args(remove_ros_args(args=raw_args)[1:])
    cli_max_frames = 1 if ns.once else None
    if ns.timeout_sec is not None and ns.timeout_sec < 0:
        parser.error("--timeout-sec 必須 >= 0")

    rclpy.init(args=args)
    node = RgbdRecorder(cli_output_dir=ns.output_dir, cli_max_frames=cli_max_frames,
                        cli_timeout_sec=ns.timeout_sec)
    exit_code = 0
    try:
        # 用 spin_once 迴圈而非 rclpy.spin()，這樣 max_frames 存滿後
        # 才能讓 node.done 生效、自動跳出迴圈並乾淨地結束程式
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.timed_out():
                print(node.timeout_message(), file=sys.stderr, flush=True)
                exit_code = EXIT_TIMEOUT
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


# ----------------------------------------------------------------------------
# Subprocess wrapper (fallback capture for the orchestrator / ROS node)
# ----------------------------------------------------------------------------
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

DEFAULT_RGB_TOPIC = "/camera/color/image_raw"
DEFAULT_POINTCLOUD_TOPIC = "/camera/depth_registered/points"
# Paths may contain spaces (e.g. outputs on "/media/<user>/My Passport/..."): match up to
# the extension at the end of the log line (the PCD line continues with " (N 點...)").
_SAVED_IMAGE_RE = re.compile(r"已儲存影像:\s*(.+?\.png)\s*$", re.M)
_SAVED_PCD_RE = re.compile(r"已儲存點雲:\s*(.+?\.pcd)(?=\s+\(|\s*$)", re.M)
_SAVED_MANIFEST_RE = re.compile(r"已儲存 capture manifest:\s*(.+?\.json)\s*$", re.M)


class RecorderTimeout(RuntimeError):
    """The recorder exited with EXIT_TIMEOUT: no synced RGB-D pair (camera not publishing)."""


class RecorderError(RuntimeError):
    """The recorder failed or committed no complete capture."""


def recorder_command(output_root, timeout_s=None, rgb_topic=DEFAULT_RGB_TOPIC,
                     pointcloud_topic=DEFAULT_POINTCLOUD_TOPIC, python=None):
    """argv of one ``--once`` recorder run (this module as ``python -m``)."""
    cmd = [python or sys.executable, "-m", "box_seam_detection.live.recorder",
           str(output_root), "--once"]
    if timeout_s is not None:
        cmd += ["--timeout-sec", str(float(timeout_s))]
    cmd += ["--ros-args", "-p", f"rgb_topic:={rgb_topic}",
            "-p", f"pointcloud_topic:={pointcloud_topic}"]
    return cmd


def parse_recorder_output(text):
    """(image_path, pcd_path, manifest_path) from the recorder's output, or None."""
    image, pcd, manifest = (rx.search(text) for rx in
                            (_SAVED_IMAGE_RE, _SAVED_PCD_RE, _SAVED_MANIFEST_RE))
    if not (image and pcd and manifest):
        return None
    return (Path(image.group(1)).resolve(), Path(pcd.group(1)).resolve(),
            Path(manifest.group(1)).resolve())


def capture_once(output_root, attempt=1, timeout_s=20.0, rgb_topic=DEFAULT_RGB_TOPIC,
                 pointcloud_topic=DEFAULT_POINTCLOUD_TOPIC, python=None, echo=True):
    """One live RGB-D capture through the recorder subprocess (today's path).

    Returns ``box_seam_detection.live.orchestrator.CaptureFiles`` after the
    manifest/image/pcd binding checks. Raises RecorderTimeout / RecorderError.
    """
    from .orchestrator import CaptureFiles, load_capture_manifest  # noqa: PLC0415

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = recorder_command(output_root, timeout_s, rgb_topic, pointcloud_topic, python)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if echo:
        print(result.stdout)
    if result.returncode == EXIT_TIMEOUT:
        raise RecorderTimeout(
            f"rgbd_recorder timed out (exit {EXIT_TIMEOUT}): no synced RGB-D images/cloud "
            f"(attempt {attempt}); silent topics are listed in its output")
    if result.returncode != 0:
        raise RecorderError(f"rgbd_recorder failed (exit {result.returncode})")
    paths = parse_recorder_output(result.stdout)
    if paths is None:
        raise RecorderError("rgbd_recorder committed no complete capture manifest")
    image_path, pcd_path, manifest_path = paths
    try:
        manifest = load_capture_manifest(manifest_path, image_path, pcd_path)
    except ValueError as error:
        raise RecorderError(f"capture manifest validation failed: {error}") from error
    return CaptureFiles(capture_id=manifest["capture_id"], image_path=image_path,
                        pcd_path=pcd_path, manifest_path=manifest_path,
                        stamp_ns=int(manifest["image"]["header"]["stamp_ns"]))


if __name__ == "__main__":
    sys.exit(main())
