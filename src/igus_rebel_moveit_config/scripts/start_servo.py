#!/usr/bin/env python3
"""Start Humble Servo once after its service is ready, then exit."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from std_srvs.srv import Trigger


class ServoStarter(Node):
    def __init__(self):
        super().__init__("servo_starter")
        self.active = False
        self.status = self.create_subscription(Int8, "/status", self.on_status, 10)
        self.client = self.create_client(Trigger, "/servo_node/start_servo")

    def on_status(self, _message):
        self.active = True

    def run(self):
        deadline = time.monotonic() + 15.0
        self.get_logger().info("Waiting up to 15 s for /servo_node/start_servo")
        while rclpy.ok() and time.monotonic() < deadline:
            if self.client.wait_for_service(timeout_sec=0.5):
                break
        else:
            self.get_logger().error("/servo_node/start_servo unavailable after 15 s")
            return 1

        # A standalone keyboard/gamepad may have started Servo while we waited.
        rclpy.spin_once(self, timeout_sec=0.5)
        if self.active:
            self.get_logger().info("Servo already publishing status; no start call needed")
            return 0

        for attempt in range(1, 4):
            future = self.client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done():
                try:
                    response = future.result()
                except Exception as error:
                    self.get_logger().warning(f"start_servo attempt {attempt}/3 failed: {error}")
                else:
                    self.get_logger().info(
                        f"start_servo attempt {attempt}/3: success={response.success}, "
                        f"message={response.message!r}"
                    )
                    if response.success:
                        # Humble returns success even if Servo::start() timed out waiting
                        # for robot state, so require its live loop's status signal.
                        status_deadline = time.monotonic() + 2.0
                        while not self.active and time.monotonic() < status_deadline:
                            rclpy.spin_once(self, timeout_sec=0.1)
                        if not self.active:
                            self.get_logger().error("start_servo succeeded but /status did not appear")
                            return 1
                        return 0
            else:
                self.get_logger().warning(f"start_servo attempt {attempt}/3 timed out after 5 s")
                self.client.remove_pending_request(future)
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.active:
                self.get_logger().info("Servo status is publishing; no further start call needed")
                return 0
        self.get_logger().error("Could not start Servo after 3 attempts")
        return 1


def main():
    rclpy.init()
    node = ServoStarter()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
