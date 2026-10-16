"""Call <ns>/servo_node/start_servo for each given Servo namespace once its service is up, then exit.

Humble's Servo idles until start_servo; a second call re-initialises its thread, so a call is only
repeated after a long timeout and never after success. Usage: servo_starter /mocap/arm_left/servo ...
"""
import sys
import time

import rclpy
from std_srvs.srv import Trigger

START_TIMEOUT_S = 20.0
TOTAL_TIMEOUT_S = 90.0


def main(args=None):
    rclpy.init(args=args)
    namespaces = [a for a in rclpy.utilities.remove_ros_args(sys.argv)[1:] if a.startswith("/")]
    node = rclpy.create_node("mocap_servo_starter")
    clients = {ns: node.create_client(Trigger, f"{ns}/servo_node/start_servo") for ns in namespaces}
    pending = {}   # ns -> (future, t0)
    done = set()
    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    while rclpy.ok() and len(done) < len(namespaces) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        for ns, client in clients.items():
            if ns in done:
                continue
            if ns in pending:
                future, t0 = pending[ns]
                if future.done():
                    result = future.result() if future.exception() is None else None
                    if result is not None and result.success:
                        node.get_logger().info(f"{ns}: Servo started")
                        done.add(ns)
                    else:
                        node.get_logger().warn(f"{ns}: start_servo failed: {result}")
                        del pending[ns]
                elif time.monotonic() - t0 > START_TIMEOUT_S:
                    node.get_logger().warn(f"{ns}: start_servo timed out; retrying")
                    del pending[ns]
            elif client.service_is_ready():
                pending[ns] = (client.call_async(Trigger.Request()), time.monotonic())
    ok = len(done) == len(namespaces)
    if not ok:
        node.get_logger().error(f"Servo not started: {sorted(set(namespaces) - done)}")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
