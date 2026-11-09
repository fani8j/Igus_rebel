#!/usr/bin/env python3
"""align_camera.py -- READ-ONLY helper to re-aim a camera so its view matches the GR00T/TurboVLA training data.

Put the left arm at the training start pose [0, 0, 1.0472, 0, 0.8727, 0] (the reference frames are the episodes' first
frames at that pose), run this, and adjust the camera mount until rotation / shift / scale are ~0 and the verdict is OK.
It subscribes only (camera + /joint_states); it never publishes or moves anything.

    source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash   # + your ROS_DOMAIN_ID
    python3 -s ~/igus_rebel_ros2/tools/groot_inference/align_camera.py                 # gripper camera (wrist_left)
    python3 -s ~/igus_rebel_ros2/tools/groot_inference/align_camera.py --camera head   # head camera

Use the system python3 with -s (its OpenCV has a GUI; the venv / user-site OpenCV is headless).  Window: left = 50/50 blend of
the training reference and the live image, right = live image; both with a 3x3 grid.  q / Esc quits.  --headless N prints N
measurements and exits (no window).

Images are preprocessed like the training recorder (PIL draft + BILINEAR to 640x480), so the comparison is in model-input space.
The match uses ORB features + a RANSAC similarity transform from the reference to the live image: rot = how much the live view
is rotated (deg), dx / dy = shift of the reference content in the live image (px of 640x480), scale = zoom.
"""
import argparse
import io
import os
import threading
import time

import cv2
import numpy as np

REF_DIR = os.path.expanduser("~/groot/reference/frames")
TOPICS = {"wrist_left": "/arm_left/gripper_camera/image_raw/compressed", "head": "/camera/color/image_raw/compressed"}
START = np.array([0.0, 0.0, 1.0472, 0.0, 0.8727, 0.0])
OK_ROT, OK_SHIFT, OK_SCALE = 1.5, 15.0, 0.03   # verdict thresholds (deg, px, fraction)


def recorder_rgb(data):
    """CompressedImage bytes -> 640x480 RGB exactly like the training recorder (PIL draft + BILINEAR)."""
    from PIL import Image
    img = Image.open(io.BytesIO(bytes(data)))
    img.draft("RGB", (640, 480))
    img = img.convert("RGB")
    if img.size != (640, 480):
        img = img.resize((640, 480), Image.BILINEAR)
    return np.asarray(img)


class Matcher:
    def __init__(self, ref_gray):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kr, self.dr = self.orb.detectAndCompute(ref_gray, None)

    def compare(self, gray):
        k, d = self.orb.detectAndCompute(gray, None)
        if d is None or self.dr is None:
            return None
        m = sorted(self.bf.match(self.dr, d), key=lambda x: x.distance)[:400]
        if len(m) < 12:
            return None
        src = np.float32([self.kr[x.queryIdx].pt for x in m])
        dst = np.float32([k[x.trainIdx].pt for x in m])
        A, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=4.0)
        if A is None:
            return None
        return {"rot": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "dx": float(A[0, 2]), "dy": float(A[1, 2]),
                "scale": float(np.hypot(A[0, 0], A[1, 0])), "inliers": int(inl.sum())}


def verdict(r):
    if r is None or r["inliers"] < 25:
        return "NO MATCH (too different / occluded)"
    ok = abs(r["rot"]) <= OK_ROT and abs(r["dx"]) <= OK_SHIFT and abs(r["dy"]) <= OK_SHIFT and abs(r["scale"] - 1) <= OK_SCALE
    return "OK" if ok else "ADJUST"


def grid(im):
    im = im.copy()
    for y in (160, 320):
        cv2.line(im, (0, y), (640, y), (0, 255, 0), 1)
    for x in (213, 426):
        cv2.line(im, (x, 0), (x, 480), (0, 255, 0), 1)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", choices=sorted(TOPICS), default="wrist_left")
    ap.add_argument("--topic", default=None, help="override the camera topic")
    ap.add_argument("--ref", default=None, help="reference image (default ~/groot/reference/frames/mocap_left_new_3_<camera>.jpg)")
    ap.add_argument("--headless", type=int, default=0, metavar="N", help="print N measurements and exit, no window")
    ap.add_argument("--joint-prefix", default="", help="joint name prefix in /joint_states, e.g. arm_right_ (dual-table stack)")
    ap.add_argument("--start", type=float, nargs=6, default=None, metavar="RAD",
                    help="start pose for the pose check (default: the left GR00T start pose)")
    a = ap.parse_args()
    start = np.array(a.start) if a.start else START
    jn = [f"{a.joint_prefix}joint{i}" for i in range(1, 7)]
    topic = a.topic or TOPICS[a.camera]
    ref_bgr = cv2.resize(cv2.imread(a.ref or f"{REF_DIR}/mocap_left_new_3_{a.camera}.jpg"), (640, 480))
    matcher = Matcher(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, JointState
    rclpy.init()
    node = Node("align_camera_readonly")
    latest = {"img": None, "q": None}
    lock = threading.Lock()

    def on_img(msg):
        with lock:
            latest["img"] = msg.data

    def on_js(msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in jn):
            with lock:
                latest["q"] = np.array([d[j] for j in jn])

    node.create_subscription(CompressedImage, topic, on_img, qos_profile_sensor_data)
    node.create_subscription(JointState, "/joint_states", on_js, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f"[align] {a.camera}: {topic}  (read-only).  Arm must be at the start pose {start.tolist()}")

    shown = 0
    try:
        while True:
            time.sleep(0.2)
            with lock:
                data, q = latest["img"], latest["q"]
            if data is None:
                print("[align] waiting for", topic)
                time.sleep(0.8)
                continue
            live_bgr = cv2.cvtColor(recorder_rgb(data), cv2.COLOR_RGB2BGR)
            r = matcher.compare(cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY))
            v = verdict(r)
            pose = "pose ?" if q is None else (f"pose OK" if np.abs(q - start).max() < 0.02 else
                                               f"NOT AT START POSE (max off {np.abs(q - start).max():.3f} rad)")
            txt = (f"rot {r['rot']:+5.1f} deg  dx {r['dx']:+6.1f}  dy {r['dy']:+6.1f} px  scale {r['scale']:.3f}  inl {r['inliers']}"
                   if r else "no match")
            if a.headless:
                print(f"[align] {txt}  -> {v}   [{pose}]")
                shown += 1
                if shown >= a.headless:
                    break
                continue
            blend = grid(cv2.addWeighted(ref_bgr, 0.5, live_bgr, 0.5, 0))
            live = grid(live_bgr)
            col = (0, 200, 0) if v == "OK" else (0, 0, 255)
            cv2.putText(blend, "50/50 training reference + live", (8, 24), 0, 0.6, (0, 255, 255), 2)
            cv2.putText(live, f"LIVE {a.camera}", (8, 24), 0, 0.6, (0, 255, 255), 2)
            canvas = cv2.hconcat([blend, live])
            cv2.putText(canvas, txt, (8, 460), 0, 0.6, (0, 255, 255), 2)
            cv2.putText(canvas, f"{v}   [{pose}]", (660, 460), 0, 0.7, col, 2)
            cv2.imshow("align_camera (q to quit)", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
