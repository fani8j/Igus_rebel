#!/usr/bin/env python3
"""Read the joint limits configured in the ReBeL controller over CRI (read-only).

Sends only `CONFIG GetKinematicLimits`: no SetActive, Enable or jog, so the robot
cannot move. Run it while the ROS driver is stopped (the controller has one active
CRI client).

    python3 tools/cri_query_limits.py [--ip 192.168.3.11] [--port 3920]
"""

import argparse
import re
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ip", default="192.168.3.11")
    parser.add_argument("--port", type=int, default=3920)
    parser.add_argument("--listen", type=float, default=3.0, help="seconds to wait for replies")
    args = parser.parse_args()

    with socket.create_connection((args.ip, args.port), timeout=3.0) as sock:
        sock.sendall(b"CRISTART 1 CONFIG GetKinematicLimits CRIEND\n")
        sock.settimeout(0.2)
        data = b""
        end = time.monotonic() + args.listen
        while time.monotonic() < end:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            data += chunk

    frames = re.findall(r"CRISTART\s+(.*?)\s*CRIEND", data.decode(errors="replace"), re.S)
    relevant = [f for f in frames if re.search(r"CONFIG|KINEMATICLIMIT|CMDERROR", f, re.I)]
    if not relevant:
        print("No CONFIG reply received. Frame types seen:",
              sorted({f.split()[1] for f in frames if len(f.split()) > 1}))
        return 1
    for frame in relevant:
        print(frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
