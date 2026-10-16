"""Deadman helper: publishes /mocap/enable = true at 10 Hz while running, false on exit.

Run it in a terminal; Ctrl-C (or closing the terminal) releases the deadman. The follower
also treats a silent enable topic (> enable_timeout_s) as released, so a crash is safe too.
"""
import argparse
import signal
import sys
import time

import rclpy
from std_msgs.msg import Bool

_stop = False


def _handler(_signum, _frame):
    global _stop
    _stop = True


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/mocap/enable")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--no-prompt", action="store_true",
                        help="start publishing true immediately (default: wait for Enter)")
    opts, ros_args = parser.parse_known_args(sys.argv[1:] if args is None else args)
    if not 5.0 <= opts.rate <= 50.0:
        parser.error("--rate must be in [5, 50] Hz (follower needs >= 5 Hz)")
    rclpy.init(args=ros_args)
    node = rclpy.create_node("mocap_enable_keyboard")
    pub = node.create_publisher(Bool, opts.topic, 10)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handler)
    try:
        if not opts.no_prompt:
            print("Press Enter to ENABLE the mocap follower (Ctrl-C releases).", flush=True)
            try:
                sys.stdin.readline()
            except (OSError, KeyboardInterrupt):
                pass
        print(f"Publishing {opts.topic}=true at {opts.rate:.0f} Hz. Ctrl-C to release.", flush=True)
        period = 1.0 / opts.rate
        next_t = time.monotonic()
        while not _stop and rclpy.ok():
            pub.publish(Bool(data=True))
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        for _ in range(5):
            try:
                pub.publish(Bool(data=False))
            except Exception:
                break
            time.sleep(0.02)
        print("Released: published false.", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
