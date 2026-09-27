#!/usr/bin/env python3
"""Check rate, latency and cross-camera time sync of the cell cameras.

Subscribes to each camera's camera_info (same header stamp as its image, but
tiny, so the check itself does not load the system) and reports:
  - rate and stamp jitter per camera,
  - latency = host receive time - header stamp (also shows that all stamps
    are on the host clock; a clock-domain mismatch shows up as seconds),
  - matched sets from message_filters.ApproximateTimeSynchronizer (the same
    matcher box_perception_node uses) and the stamp spread inside each set.

    ros2 run rebel_cameras camera_sync_check.py [--duration 10] [--slop 0.05]
"""

import argparse

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

CAMERAS = {
    'scene (336)': '/camera/color/camera_info',
    'wrist (305)': '/wrist_camera/color/camera_info',
    'knife (C930e)': '/knife_camera/camera_info',
}


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


class SyncCheck(Node):

    def __init__(self, slop):
        super().__init__('camera_sync_check')
        self.samples = {name: [] for name in CAMERAS}
        self.sets = []
        subs = []
        for name, topic in CAMERAS.items():
            self.create_subscription(
                CameraInfo, topic, lambda m, n=name: self._on_info(n, m),
                qos_profile_sensor_data)
            subs.append(message_filters.Subscriber(
                self, CameraInfo, topic, qos_profile=qos_profile_sensor_data))
        sync = message_filters.ApproximateTimeSynchronizer(subs, 30, slop)
        sync.registerCallback(self._on_set)

    def _on_info(self, name, msg):
        self.samples[name].append((self.get_clock().now().nanoseconds, stamp_ns(msg)))

    def _on_set(self, *msgs):
        stamps = [stamp_ns(m) for m in msgs]
        self.sets.append(max(stamps) - min(stamps))


def ms(ns):
    return ns / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--slop', type=float, default=0.05,
                        help='ApproximateTimeSynchronizer slop in seconds')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SyncCheck(args.slop)
    end = node.get_clock().now().nanoseconds + int(args.duration * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f'--- {args.duration:.0f} s window, slop {args.slop * 1000:.0f} ms')
    for name, samples in node.samples.items():
        if len(samples) < 2:
            print(f'{name:14s} NO DATA on {CAMERAS[name]}')
            continue
        rx, st = (np.array(v, dtype=np.int64) for v in zip(*samples))
        period = np.diff(st)
        lat = rx - st
        print(f'{name:14s} {(len(st) - 1) * 1e9 / (st[-1] - st[0]):5.1f} Hz  '
              f'period jitter p95 {ms(np.percentile(np.abs(period - np.median(period)), 95)):5.1f} ms  '
              f'latency median {ms(np.median(lat)):5.1f} ms  p95 {ms(np.percentile(lat, 95)):5.1f} ms')
    names = [n for n, v in node.samples.items() if len(v) > 1]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = np.array([s for _, s in node.samples[a]])
            sb = np.sort([s for _, s in node.samples[b]])
            idx = np.clip(np.searchsorted(sb, sa), 1, len(sb) - 1)
            nearest = np.minimum(np.abs(sa - sb[idx - 1]), np.abs(sa - sb[idx]))
            print(f'offset {a} <-> {b}: nearest-frame median {ms(np.median(nearest)):5.1f} ms  '
                  f'max {ms(nearest.max()):5.1f} ms')
    if node.sets:
        spread = np.array(node.sets)
        print(f'matched sets   {len(spread) / args.duration:5.1f} /s  '
              f'stamp spread median {ms(np.median(spread)):5.1f} ms  '
              f'p95 {ms(np.percentile(spread, 95)):5.1f} ms  max {ms(spread.max()):5.1f} ms')
    else:
        print('matched sets   none (a camera is missing or stamps are further apart than the slop)')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
