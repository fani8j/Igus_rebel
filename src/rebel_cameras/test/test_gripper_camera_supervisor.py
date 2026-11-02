"""Supervisor restart/shutdown logic with a fake child and a temp file as the device.

No hardware or ROS needed (the frame watchdog is disabled via --no-watchdog, or
replaced by a fake monitor in-process).
"""

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'gripper_camera_supervisor.py')

FAKE_CHILD = '''
import os, signal, sys, time
with open(sys.argv[1], 'a') as f:
    f.write(f'{os.getpid()}\\n')
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
'''


def wait_for(cond, timeout=10.0, what='condition'):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f'timed out waiting for {what}')


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill 0.
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


class Rig:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.device = tmp_path / 'video_dev'
        self.starts_file = tmp_path / 'starts.txt'
        self.err_file = tmp_path / 'supervisor.err'
        fake = tmp_path / 'fake_child.py'
        fake.write_text(FAKE_CHILD)
        params = tmp_path / 'params.yaml'
        params.write_text(f'/**:\n  ros__parameters:\n    video_device: "{self.device}"\n')
        self.args = [
            sys.executable, SCRIPT, '--params-file', str(params), '--namespace', '/test/cam',
            '--child-cmd', f'{sys.executable} {fake} {self.starts_file}', '--no-watchdog',
            '--restart-delay', '0.3', '--term-timeout', '1', '--stop-timeout', '1']
        self.proc = None
        self.err = None

    def start(self):
        self.err = open(self.err_file, 'w')
        self.proc = subprocess.Popen(self.args, stderr=self.err)

    def starts(self):
        if not self.starts_file.exists():
            return []
        return [int(x) for x in self.starts_file.read_text().split()]

    def stderr(self):
        return self.err_file.read_text()

    def close(self):
        # Only kill what this rig started.
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for pid in self.starts():
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        if self.err is not None:
            self.err.close()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def test_waits_for_missing_device_then_starts(rig):
    rig.start()
    time.sleep(1.0)
    assert rig.starts() == []
    assert rig.proc.poll() is None
    assert rig.stderr().count('not found; waiting') == 1
    rig.device.touch()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    assert 'appeared' in rig.stderr()
    assert rig.stderr().count('not found; waiting') == 1


def test_restarts_after_child_exit(rig):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    first = rig.starts()[0]
    os.kill(first, signal.SIGKILL)
    wait_for(lambda: len(rig.starts()) == 2, what='restart')
    assert rig.starts()[1] != first
    assert 'restarting: child exited' in rig.stderr()


def test_restarts_when_device_disappears_and_reappears(rig):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    first = rig.starts()[0]
    rig.device.unlink()
    wait_for(lambda: not alive(first), what='child terminated while the device is gone')
    time.sleep(1.0)
    assert len(rig.starts()) == 1  # no restart into a missing device
    rig.device.touch()
    wait_for(lambda: len(rig.starts()) == 2, what='restart after the device came back')
    assert 'restarting: device' in rig.stderr()
    assert alive(rig.starts()[1])


def test_stopped_child_is_killed_on_restart(rig):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    first = rig.starts()[0]
    os.kill(first, signal.SIGSTOP)
    rig.device.unlink()
    wait_for(lambda: not alive(first), timeout=5, what='stopped child killed')
    assert 'sending SIGKILL' in rig.stderr()


@pytest.mark.parametrize('child_also_gets_sigint', [False, True])
def test_sigint_stops_child_without_restart(rig, child_also_gets_sigint):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    child = rig.starts()[0]
    rig.proc.send_signal(signal.SIGINT)
    if child_also_gets_sigint:
        os.killpg(child, signal.SIGINT)  # the child is its own process group leader
    assert rig.proc.wait(timeout=8) == 0
    assert not alive(child)
    time.sleep(1.0)  # longer than --restart-delay
    assert len(rig.starts()) == 1
    assert 'restarting' not in rig.stderr()


def test_sigterm_during_restart_delay_exits_without_restart(rig):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    os.kill(rig.starts()[0], signal.SIGKILL)
    wait_for(lambda: 'restarting' in rig.stderr(), what='restart decision')
    rig.proc.send_signal(signal.SIGTERM)
    assert rig.proc.wait(timeout=8) == 0
    assert len(rig.starts()) == 1


def test_sigint_kills_stopped_child(rig):
    rig.device.touch()
    rig.start()
    wait_for(lambda: len(rig.starts()) == 1, what='first start')
    child = rig.starts()[0]
    os.kill(child, signal.SIGSTOP)
    rig.proc.send_signal(signal.SIGINT)
    assert rig.proc.wait(timeout=8) == 0
    assert not alive(child)
    assert len(rig.starts()) == 1


def _load_module():
    spec = importlib.util.spec_from_file_location('gripper_camera_supervisor', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMonitor:
    """camera_info source: delivers frames until `flowing` is cleared."""

    def __init__(self):
        self.frames = 0
        self.last_frame_time = None
        self.flowing = True

    def reset(self):
        self.frames = 0
        self.last_frame_time = None

    def spin(self, timeout_s):
        time.sleep(timeout_s)
        if self.flowing:
            self.frames += 1
            self.last_frame_time = time.monotonic()


@pytest.mark.parametrize('mode', ['stall', 'no_first_frame'])
def test_watchdog_restarts_silent_child(tmp_path, mode):
    module = _load_module()
    device = tmp_path / 'video_dev'
    device.touch()
    starts_file = tmp_path / 'starts.txt'
    fake = tmp_path / 'fake_child.py'
    fake.write_text(FAKE_CHILD)
    cmd = [sys.executable, str(fake), str(starts_file)]
    monitor = FakeMonitor()
    monitor.flowing = mode == 'stall'
    sup = module.Supervisor(str(device), lambda _real: cmd, monitor, restart_delay=0.2,
                            stall_timeout=0.5, startup_timeout=0.5, term_timeout=1,
                            stop_timeout=1)
    thread = threading.Thread(target=sup.run)
    thread.start()
    try:
        def starts():
            return starts_file.read_text().split() if starts_file.exists() else []
        wait_for(lambda: len(starts()) == 1, what='first start')
        if mode == 'stall':
            time.sleep(0.8)  # longer than the stall timeout: frames keep it healthy
            assert len(starts()) == 1
            monitor.flowing = False
        wait_for(lambda: len(starts()) == 2, timeout=6, what='watchdog restart')
        monitor.flowing = True  # frames flow again after the restart
    finally:
        sup.request_stop()
        thread.join(timeout=10)
        for pid in starts():
            if alive(int(pid)):
                os.kill(int(pid), signal.SIGKILL)
    assert not thread.is_alive()


def test_waits_while_device_is_another_usb_device(tmp_path):
    module = _load_module()
    device = tmp_path / 'video_dev'
    device.touch()
    starts_file = tmp_path / 'starts.txt'
    fake = tmp_path / 'fake_child.py'
    fake.write_text(FAKE_CHILD)
    cmd = [sys.executable, str(fake), str(starts_file)]
    owner = {'id': '2bc5:0840'}  # a Gemini 305 at the gripper camera's by-path
    sup = module.Supervisor(str(device), lambda _real: cmd, None, restart_delay=0.2,
                            term_timeout=1, stop_timeout=1, usb_id='05a3:9230',
                            usb_id_reader=lambda _real: owner['id'])
    thread = threading.Thread(target=sup.run)
    thread.start()
    try:
        time.sleep(0.5)
        assert not starts_file.exists()
        owner['id'] = '05a3:9230'
        wait_for(starts_file.exists, what='start once the right camera is there')
    finally:
        sup.request_stop()
        thread.join(timeout=10)
        for pid in (starts_file.read_text().split() if starts_file.exists() else []):
            if alive(int(pid)):
                os.kill(int(pid), signal.SIGKILL)
    assert not thread.is_alive()
