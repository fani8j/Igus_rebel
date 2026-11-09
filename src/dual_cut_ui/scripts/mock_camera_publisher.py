#!/usr/bin/env python3
"""MOCK ONLY - publishes simulated camera streams (sensor_msgs/CompressedImage, JPEG) for the dual-cut UI.  NOT FOR THE REAL CELL.

Default topics are the six the UI offers (scene, knife, left/right wrist, left/right gripper).  Frames come from the
images in --images-dir (cycled) or are synthetic; each carries the camera name and a timestamp.  A topic is only
encoded/published while somebody subscribes to it, which is what the UI's lazy subscription should produce.

Refuses to run unless ROS_LOCALHOST_ONLY=1 and ROS_DOMAIN_ID is set explicitly and is not a domain used by the real
robot (11, 12) or by other sessions (219..232).

    ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=173 python3 mock_camera_publisher.py [--images-dir DIR] [--rate-hz 8]
"""
import argparse
import os
import sys
import time

FORBIDDEN_DOMAINS = {11, 12} | set(range(219, 233))


def check_environment(env=os.environ):
    """None if the environment is safe for a mock publisher, else the reason it is not."""
    if env.get("ROS_LOCALHOST_ONLY") != "1":
        return "ROS_LOCALHOST_ONLY=1 must be set explicitly"
    raw = env.get("ROS_DOMAIN_ID")
    if raw is None or raw.strip() == "":
        return "ROS_DOMAIN_ID must be set explicitly"
    try:
        domain = int(raw)
    except ValueError:
        return f"ROS_DOMAIN_ID={raw!r} is not an integer"
    if domain in FORBIDDEN_DOMAINS:
        return f"ROS_DOMAIN_ID={domain} is reserved (11, 12 real robot; 219..232 other sessions)"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default="")
    ap.add_argument("--rate-hz", type=float, default=8.0)
    ap.add_argument("--size", default="1280x720", help="WxH of the published frames")
    ap.add_argument("--cameras", default="", help='"name=/topic,..." (default: the six UI cameras)')
    a = ap.parse_args(argv)
    why = check_environment()
    if why:
        print(f"mock_camera_publisher: refusing to run: {why}", file=sys.stderr)
        return 2
    if not 5.0 <= a.rate_hz <= 15.0:
        print("mock_camera_publisher: --rate-hz must be 5..15", file=sys.stderr)
        return 2
    import cv2
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from dual_cut_ui.camera import parse_camera_spec
    width, height = (int(v) for v in a.size.lower().split("x"))
    cameras = parse_camera_spec(a.cameras)
    images = []
    if a.images_dir:
        for fn in sorted(os.listdir(a.images_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                im = cv2.imread(os.path.join(a.images_dir, fn))
                if im is not None:
                    images.append(cv2.resize(im, (width, height)))
    rclpy.init()
    node = rclpy.create_node("mock_camera_publisher")
    pubs = {n: node.create_publisher(CompressedImage, t, qos_profile_sensor_data) for n, t in cameras.items()}
    counts = {n: 0 for n in cameras}
    frame_no = [0]

    def frame(name, i):
        if images:
            img = images[i % len(images)].copy()
        else:
            x = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, 0)
            hue = (sum(map(ord, name)) % 160) + 10
            img = cv2.cvtColor(np.dstack([np.full_like(x, hue), np.full_like(x, 120), x]), cv2.COLOR_HSV2BGR)
            cx = int((i * 17) % width)
            cv2.rectangle(img, (cx, 0), (min(width, cx + 24), height), (255, 255, 255), -1)
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        cv2.putText(img, f"MOCK {name}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 8)
        cv2.putText(img, f"MOCK {name}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.putText(img, f"{stamp}  #{i}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        return img

    def tick():
        frame_no[0] += 1
        for name, pub in pubs.items():
            n = pub.get_subscription_count()
            if n != counts[name]:
                node.get_logger().info(f"{name}: {n} subscriber(s)")
                counts[name] = n
            if n == 0:
                continue
            ok, enc = cv2.imencode(".jpg", frame(name, frame_no[0]), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            msg = CompressedImage()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = name
            msg.format = "jpeg"
            msg.data = enc.tobytes()
            pub.publish(msg)

    node.create_timer(1.0 / a.rate_hz, tick)
    node.get_logger().info(f"MOCK cameras on ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}: " + ", ".join(cameras.values()))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
