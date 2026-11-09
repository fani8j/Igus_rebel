#!/usr/bin/env python3
"""Fast on/off/pulse of a ReBeL digital output (e.g. the knife trigger) through the driver's set_digital_output service.

`ros2 service call` starts a new process and discovers the service on every call (~1-2 s).  This tool connects once and
then sends each request immediately; the robot switches the output ~10-20 ms after the request.

Output numbers are the iRC names (DOut31 = arm output 1, DOut32 = arm output 2, DOut21-27 = base).  The CRI DOUT command
is offset by one (CRI "DOUT 30" sets iRC DOut31, seen in the driver log 2026-11-03), so this tool sends number - 1.

Examples (same ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY as the driver):
  knife_dout.py --dout 31 pulse 0.2        # high for 0.2 s, then low
  knife_dout.py --dout 31 on               # stays high until 'off'
  knife_dout.py --dout 31 off
  knife_dout.py --dout 31 interactive      # Enter = pulse, o = on, f = off, q = quit (always ends with off)

The output is always switched off on exit (normal, Ctrl+C, error) in pulse and interactive mode.
"""
import argparse
import sys
import time

import rclpy
from igus_rebel_msgs.srv import SetDigitalOutput

CRI_OFFSET = 1  # CRI number = iRC DOut number - 1


class Dout:
    def __init__(self, node, service, dout):
        self.node = node
        self.dout = dout
        self.cri_number = dout - CRI_OFFSET
        self.client = node.create_client(SetDigitalOutput, service)
        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"service {service} not found (driver running? same ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY?)")

    def set(self, on):
        req = SetDigitalOutput.Request()
        req.output.output = self.cri_number
        req.output.is_on = bool(on)
        t0 = time.monotonic()
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        dt = (time.monotonic() - t0) * 1000.0
        res = future.result()
        if res is None:
            print(f"DOut{self.dout} {'ON' if on else 'OFF'}: NO RESPONSE after {dt:.0f} ms", flush=True)
            return False
        state = "sent" if res.success else f"FAILED ({res.message})"
        print(f"DOut{self.dout} {'ON ' if on else 'OFF'} {state}  ({dt:.0f} ms, CRI DOUT {self.cri_number})", flush=True)
        return res.success

    def pulse(self, seconds):
        try:
            self.set(True)
            time.sleep(seconds)
        finally:
            self.set(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dout", type=int, required=True, help="iRC output number (31 or 32 on the arm, 21-27 base)")
    ap.add_argument("--service", default="/set_digital_output",
                    help="driver service (dual-arm drivers: /igus_rebel_<arm>/set_digital_output)")
    ap.add_argument("mode", choices=["on", "off", "pulse", "interactive"])
    ap.add_argument("seconds", nargs="?", type=float, default=0.2, help="pulse length in s (pulse / interactive, default 0.2)")
    args = ap.parse_args()
    if not (21 <= args.dout <= 27 or args.dout in (31, 32)):
        ap.error("--dout must be an iRC output number: 21-27 (base) or 31-32 (arm)")
    if not 0.02 <= args.seconds <= 10.0:
        ap.error("pulse length must be 0.02-10 s")

    # No rclpy SIGINT handler: Ctrl+C must raise KeyboardInterrupt with the context still alive, so the final "off" is sent.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("knife_dout")
    try:
        d = Dout(node, args.service, args.dout)
        if args.mode == "on":
            d.set(True)
        elif args.mode == "off":
            d.set(False)
        elif args.mode == "pulse":
            d.pulse(args.seconds)
        else:
            print(f"DOut{args.dout}: Enter = {args.seconds:.2f} s pulse, o = on, f = off, q = quit (switches off)", flush=True)
            try:
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    cmd = line.strip().lower()
                    if cmd == "":
                        d.pulse(args.seconds)
                    elif cmd == "o":
                        d.set(True)
                    elif cmd == "f":
                        d.set(False)
                    elif cmd == "q":
                        break
                    else:
                        print("Enter = pulse, o = on, f = off, q = quit", flush=True)
            finally:
                d.set(False)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
