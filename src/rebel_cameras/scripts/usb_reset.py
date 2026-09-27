#!/usr/bin/env python3
"""USB port reset of a device selected by serial number (same as a replug).

The Gemini 305 (firmware 1.0.70, orbbec_camera 2.9.3) streams only on the
first driver start after it is plugged in: any later start hangs after
"Initialize device cost" with no frames, even after a clean shutdown. A port
reset before each start avoids that. Needs write access to /dev/bus/usb/*,
which the orbbec udev rules grant (MODE 0666); no sudo.

    usb_reset.py CV27561000NP [--timeout 20]
"""

import argparse
import fcntl
import glob
import os
import sys
import time

USBDEVFS_RESET = ord('U') << 8 | 20


def find_device(serial):
    for path in glob.glob('/sys/bus/usb/devices/*/serial'):
        try:
            with open(path) as f:
                if f.read().strip() != serial:
                    continue
            base = os.path.dirname(path)
            with open(os.path.join(base, 'busnum')) as f:
                bus = int(f.read())
            with open(os.path.join(base, 'devnum')) as f:
                dev = int(f.read())
            return f'/dev/bus/usb/{bus:03d}/{dev:03d}'
        except OSError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('serial')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='seconds to wait for the device to re-enumerate')
    parser.add_argument('--settle', type=float, default=3.0,
                        help='extra seconds after re-enumeration before returning')
    args, _ = parser.parse_known_args()  # ignore --ros-args from launch

    node = find_device(args.serial)
    if node is None:
        sys.exit(f'usb_reset: no USB device with serial {args.serial}')
    fd = os.open(node, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)
    print(f'usb_reset: reset {args.serial} at {node}', flush=True)

    deadline = time.monotonic() + args.timeout
    time.sleep(1.0)
    while find_device(args.serial) is None:
        if time.monotonic() > deadline:
            sys.exit(f'usb_reset: {args.serial} did not come back within {args.timeout} s')
        time.sleep(0.5)
    time.sleep(args.settle)
    print(f'usb_reset: {args.serial} back', flush=True)


if __name__ == '__main__':
    main()
