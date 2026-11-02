#!/usr/bin/env python3
"""Keeps one usb_cam gripper camera running across unplugs, stalls and crashes.

usb_cam either crashes (uncaught VIDIOC_DQBUF error) or stays alive publishing
nothing when the camera is unplugged, and the by-path symlink may point at a
different /dev/videoN after a replug. This supervisor runs usb_cam as a child
and restarts it when
  - the child exits on its own,
  - the device path disappears or resolves to a different node (the child is
    terminated first),
  - camera_info stalls (no message for --stall-timeout after frames had been
    flowing) or no first frame arrives within --startup-timeout.
If the device is not there at start, it waits for it; nothing aborts.
With --usb-id it also waits while the device belongs to another USB device:
/dev/v4l/by-path names have no bus number, so a USB3 camera on the same hub
port (bus 2) gets the same name as the gripper camera (bus 1).

The child runs in its own session, so a terminal Ctrl+C reaches only the
supervisor, which then stops the child itself and never restarts it. (A launch
`respawn=True` on usb_cam instead restarts it forever on Ctrl+C.) The child also
gets PR_SET_PDEATHSIG=SIGTERM so it never outlives a killed supervisor.

    gripper_camera_supervisor.py --params-file left_gripper_camera.yaml \
        --namespace /arm_left/gripper_camera
"""

import argparse
import ctypes
import os
import shlex
import signal
import subprocess
import sys
import time

import yaml

PR_SET_PDEATHSIG = 1
TICK_S = 0.1


def log(level, msg):
    print(f'[{level}] [gripper_camera_supervisor]: {msg}', file=sys.stderr, flush=True)


def _child_preexec():
    # The child must not outlive a killed supervisor (SIGKILL, OOM, crash).
    ctypes.CDLL('libc.so.6').prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def usb_id_of(realpath):
    """'vvvv:pppp' of the USB device owning a /dev/videoN node, or None."""
    usb_dev = f'/sys/class/video4linux/{os.path.basename(realpath)}/device/..'
    try:
        with open(f'{usb_dev}/idVendor') as v, open(f'{usb_dev}/idProduct') as p:
            return f'{v.read().strip()}:{p.read().strip()}'
    except OSError:
        return None


def read_video_device(params_file):
    with open(params_file) as f:
        return yaml.safe_load(f)['/**']['ros__parameters']['video_device']


def usb_cam_command(params_file, namespace, node_name):
    """Returns make_cmd(device_realpath) for the real usb_cam node."""
    from ament_index_python.packages import get_package_prefix
    exe = os.path.join(get_package_prefix('usb_cam'), 'lib', 'usb_cam', 'usb_cam_node_exe')

    def make_cmd(realpath):
        # usb_cam rejects a symlinked device path; pass the resolved node.
        return [exe, '--ros-args', '-r', f'__node:={node_name}', '-r', f'__ns:={namespace}',
                '--params-file', params_file, '-p', f'video_device:={realpath}']
    return make_cmd


class FrameMonitor:
    """Counts camera_info messages (published once per frame) with rclpy."""

    def __init__(self, namespace, node_name):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo
        self._rclpy = rclpy
        # The supervisor does its own SIGINT/SIGTERM handling.
        rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
        self._node = rclpy.create_node(f'{node_name}_supervisor', namespace=namespace)
        self._node.create_subscription(
            CameraInfo, f'{namespace}/camera_info', self._on_info, qos_profile_sensor_data)
        self.frames = 0
        self.last_frame_time = None

    def _on_info(self, _msg):
        self.frames += 1
        self.last_frame_time = time.monotonic()

    def reset(self):
        self.frames = 0
        self.last_frame_time = None

    def spin(self, timeout_s):
        self._rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def close(self):
        self._node.destroy_node()
        self._rclpy.shutdown()


class Supervisor:
    """Restart/shutdown logic; make_cmd and monitor are injectable for tests.

    monitor: object with reset(), spin(timeout_s), frames and last_frame_time
    (time.monotonic()); None disables the frame watchdog.
    """

    def __init__(self, device, make_cmd, monitor=None, restart_delay=2.0,
                 stall_timeout=5.0, startup_timeout=15.0, term_timeout=3.0,
                 stop_timeout=5.0, usb_id=None, usb_id_reader=usb_id_of):
        self.device = device
        self.usb_id = usb_id
        self.usb_id_reader = usb_id_reader
        self.make_cmd = make_cmd
        self.monitor = monitor
        self.restart_delay = restart_delay
        self.stall_timeout = stall_timeout
        self.startup_timeout = startup_timeout
        self.term_timeout = term_timeout
        self.stop_timeout = stop_timeout
        self._stopping = False
        self._child = None

    def request_stop(self, *_):
        # Signal-handler safe: only sets a flag; the main loop does the work.
        self._stopping = True

    # --- helpers -------------------------------------------------------------

    def _identity(self):
        """(realpath, inode) of the device node, or None when it is missing."""
        try:
            return os.path.realpath(self.device), os.stat(self.device).st_ino
        except OSError:
            return None

    def _sleep(self, seconds):
        """Sleep that returns early on stop and keeps the monitor spinning."""
        end = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < end:
            self._tick(min(TICK_S, max(0.0, end - time.monotonic())))

    def _tick(self, timeout_s):
        if self.monitor is not None:
            self.monitor.spin(timeout_s)
        else:
            time.sleep(timeout_s)

    def _signal_group(self, sig):
        try:
            os.killpg(self._child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _stop_child(self, sig, timeout_s):
        """Signals the child's process group; SIGKILLs it if it outlasts timeout_s.

        SIGKILL also covers a SIGSTOPped child, which cannot act on SIGTERM/SIGINT.
        """
        child = self._child
        if child.poll() is None:
            self._signal_group(sig)
            try:
                child.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                log('WARN', f'child {child.pid} ignored signal {int(sig)}, sending SIGKILL')
            # Whatever is left in the group goes too (the leader's pid is only
            # recycled once the group is empty).
            self._signal_group(signal.SIGKILL)
            child.wait()
        self._child = None

    def _start_child(self, identity):
        cmd = self.make_cmd(identity[0])
        self._child = subprocess.Popen(
            cmd, start_new_session=True, preexec_fn=_child_preexec)
        if self.monitor is not None:
            self.monitor.reset()
        log('INFO', f'started child pid {self._child.pid} on {identity[0]}')

    def _wait_for_device(self):
        """Blocks until the device exists (or stop); returns its identity or None."""
        warned = None
        while not self._stopping:
            identity = self._identity()
            if identity is None:
                problem = 'not found'
            elif self.usb_id and (found := self.usb_id_reader(identity[0])) != self.usb_id:
                problem = f'({identity[0]}) is USB device {found}, not {self.usb_id}'
            else:
                if warned:
                    log('INFO', f'device {self.device} appeared')
                return identity
            if problem != warned:
                log('WARN', f'device {self.device} {problem}; waiting for it')
                warned = problem
            self._tick(TICK_S)
        return None

    def _check(self, identity, started):
        """Returns a restart reason for the running child, or None."""
        if self._child.poll() is not None:
            return f'child exited with code {self._child.returncode}'
        now_identity = self._identity()
        if now_identity is None:
            return f'device {self.device} disappeared'
        if now_identity != identity:
            return f'device node changed ({identity[0]} -> {now_identity[0]})'
        if self.monitor is not None:
            now = time.monotonic()
            if self.monitor.frames == 0:
                if now - started > self.startup_timeout:
                    return f'no first frame within {self.startup_timeout:g} s'
            elif now - self.monitor.last_frame_time > self.stall_timeout:
                return f'camera_info stalled for more than {self.stall_timeout:g} s'
        return None

    # --- main loop -----------------------------------------------------------

    def run(self):
        while not self._stopping:
            identity = self._wait_for_device()
            if identity is None:
                break
            self._start_child(identity)
            started = time.monotonic()
            reason = None
            while not self._stopping:
                reason = self._check(identity, started)
                if reason:
                    break
                self._tick(TICK_S)
            # Re-check: a signal may have arrived while deciding to restart.
            if self._stopping:
                break
            log('WARN', f'restarting: {reason}')
            self._stop_child(signal.SIGTERM, self.term_timeout)
            self._sleep(self.restart_delay)
        if self._child is not None:
            log('INFO', f'shutting down, stopping child pid {self._child.pid}')
            self._stop_child(signal.SIGINT, self.stop_timeout)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--params-file', required=True,
                        help='usb_cam parameter YAML; video_device is read from it')
    parser.add_argument('--namespace', required=True, help='e.g. /arm_left/gripper_camera')
    parser.add_argument('--node-name', default='gripper_camera')
    parser.add_argument('--usb-id', help="expected USB vendor:product, e.g. 05a3:9230")
    parser.add_argument('--restart-delay', type=float, default=2.0)
    parser.add_argument('--stall-timeout', type=float, default=5.0,
                        help='restart when no camera_info for this long after frames flowed')
    parser.add_argument('--startup-timeout', type=float, default=15.0,
                        help='restart when no first frame within this after a start')
    # For tests without hardware.
    parser.add_argument('--child-cmd', help=argparse.SUPPRESS)
    parser.add_argument('--no-watchdog', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--term-timeout', type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument('--stop-timeout', type=float, default=5.0, help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()  # ignore --ros-args from launch

    namespace = '/' + args.namespace.strip('/')
    device = read_video_device(args.params_file)
    if args.child_cmd:
        cmd = shlex.split(args.child_cmd)
        make_cmd = lambda _realpath: cmd  # noqa: E731
    else:
        make_cmd = usb_cam_command(args.params_file, namespace, args.node_name)
    monitor = None if args.no_watchdog else FrameMonitor(namespace, args.node_name)

    supervisor = Supervisor(
        device, make_cmd, monitor, restart_delay=args.restart_delay,
        stall_timeout=args.stall_timeout, startup_timeout=args.startup_timeout,
        term_timeout=args.term_timeout, stop_timeout=args.stop_timeout,
        usb_id=args.usb_id)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    try:
        code = supervisor.run()
    finally:
        if monitor is not None:
            monitor.close()
    sys.exit(code)


if __name__ == '__main__':
    main()
