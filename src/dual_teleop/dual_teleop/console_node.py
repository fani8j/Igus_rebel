#!/usr/bin/env python3
"""Operator screen for dual_teleop: shows the selected arm, the interlocks and the Servo status, and takes the
arm-selection keys (the single-arm keyboard node has no arm key).

    l / r   select the left / right arm (the old arm is zeroed first)
    e       toggle teleop enable (teleop starts PAUSED)
    t       TOUCH MODE: asks for confirmation (press y within 5 s); pressing t again turns it off
    q       quit this screen (does not stop teleop)

Run it in its own terminal next to the keyboard node:
    ros2 run dual_teleop teleop_console
Without a TTY (piped output) it only prints a line when the state changes.
"""

import json
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from dual_teleop.screen import render


class Console(Node):
    def __init__(self):
        super().__init__("teleop_console")
        self.status = None
        self.confirm_until = 0.0
        self.last_plain = None
        self.tty = sys.stdin.isatty() and sys.stdout.isatty()
        self.create_subscription(String, "/dual_teleop/status", self.on_status, 10)
        self.svc = {k: self.create_client(Trigger, f"/dual_teleop/{k}")
                        for k in ("select_left", "select_right", "enable", "disable", "touch_mode_on", "touch_mode_off")}

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            return
        if time.monotonic() < self.confirm_until:
            self.status["prompt"] = "TOUCH MODE lets the knife contact carton/table: press y within 5 s to confirm, any other key cancels"
        if self.tty:
            sys.stdout.write("\x1b[2J\x1b[H" + render(self.status).replace("\n", "\r\n") + "\r\n")
            sys.stdout.flush()
        else:
            key = (self.status.get("state"), self.status.get("selected"), tuple(self.status.get("reasons", [])),
                   tuple(self.status.get("warnings", [])))
            if key != self.last_plain:
                self.last_plain = key
                print(render(self.status), flush=True)

    def call(self, name):
        if self.svc[name].service_is_ready():
            self.svc[name].call_async(Trigger.Request())

    def key(self, ch):
        if time.monotonic() < self.confirm_until:
            self.confirm_until = 0.0
            if ch == "y":
                self.call("touch_mode_on")
            return False
        if ch == "t":
            if self.status and self.status.get("touch", {}).get("on"):
                self.call("touch_mode_off")
            else:
                self.confirm_until = time.monotonic() + 5.0
        elif ch == "l":
            self.call("select_left")
        elif ch == "r":
            self.call("select_right")
        elif ch == "e":
            enabled = bool(self.status and self.status.get("enabled"))
            self.call("disable" if enabled else "enable")
        return ch == "q"


def main():
    rclpy.init()
    node = Console()
    saved = None
    if node.tty:
        saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.tty and select.select([sys.stdin], [], [], 0)[0]:
                if node.key(sys.stdin.read(1)):
                    break
    except Exception:  # Ctrl-C / SIGTERM: the context is gone, just restore the terminal
        pass
    finally:
        if saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
