"""One external process (the detector) with a stdout tail; only our own child process group is ever signalled."""
import collections
import os
import signal
import subprocess
import threading
import time
from typing import Callable, Deque, List, Optional

TAIL_LINES = 80
KILL_GRACE_S = 3.0


READER_JOIN_S = 1.0


class ProcessRunner:
    """Runs ``cmd`` in its own session (so the whole process group can be signalled), merges stderr into stdout,
    keeps the last lines.  ``run()`` blocks the calling (worker) thread; ``terminate()`` may be called from any thread
    at any time.

    The group is what is signalled, never just the child: SIGTERM to the whole group, then SIGKILL to the whole group after
    the grace, even if the direct child already exited (a descendant may still be running and may ignore SIGTERM).  The
    stdout reader runs on its own thread and ``run()`` waits for it only for a bounded time, so a descendant that keeps the
    pipe open cannot hang the job: after the child's exit the leftover group is SIGKILLed and the reader abandoned if need be."""

    def __init__(self, cmd: List[str], cwd: Optional[str] = None, tail_lines: int = TAIL_LINES,
                 kill_grace_s: float = KILL_GRACE_S, reader_join_s: float = READER_JOIN_S,
                 env_overrides: Optional[dict] = None):
        self.cmd, self.cwd, self.kill_grace_s, self.reader_join_s = list(cmd), cwd, kill_grace_s, reader_join_s
        self.env_overrides = dict(env_overrides or {})        # applied on top of a copy of os.environ
        self.tail: Deque[str] = collections.deque(maxlen=tail_lines)
        self.returncode: Optional[int] = None
        self.terminated = False
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None

    @property
    def pid(self) -> Optional[int]:
        p = self._proc
        return None if p is None else p.pid

    def run(self, on_line: Optional[Callable[[str], None]] = None) -> int:
        with self._lock:
            if self.terminated:
                self.returncode = -signal.SIGTERM
                return self.returncode
            try:
                env = dict(os.environ, PYTHONUNBUFFERED="1")          # live progress lines in the log tail
                env.update(self.env_overrides)
                self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                              errors="replace", cwd=self.cwd, env=env, start_new_session=True)
            except OSError as e:
                line = f"cannot start {self.cmd[0]}: {e}"
                self.tail.append(line)
                if on_line:
                    on_line(line)
                self.returncode = 127
                return self.returncode
            proc = self._proc
            self._pgid = proc.pid                # start_new_session: the child is the leader of its own group

        def pump():
            try:
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    self.tail.append(line)
                    if on_line:
                        on_line(line)
            except (ValueError, OSError):
                pass
        reader = threading.Thread(target=pump, name="runner-reader", daemon=True)
        reader.start()
        self.returncode = proc.wait()
        reader.join(self.reader_join_s)
        if reader.is_alive():
            # a descendant still holds the pipe: it belongs to this job, so it goes
            self._signal_group(signal.SIGKILL)
            reader.join(self.reader_join_s)
        return self.returncode

    def _signal_group(self, sig: int) -> bool:
        """Signal the whole group; False if it no longer exists.  Safe after the leader was reaped: while any member lives
        the kernel does not reuse the group id."""
        pgid = self._pgid
        if pgid is None:
            return False
        try:
            os.killpg(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def alive(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    def group_alive(self) -> bool:
        pgid = self._pgid
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def kill_now(self) -> None:
        """SIGKILL the process group immediately (shutdown path: no helper thread can be relied on any more)."""
        self._signal_group(signal.SIGKILL)

    def terminate(self, grace_s: Optional[float] = None) -> None:
        """SIGTERM the group now; SIGKILL the group after the grace (helper thread, so this call never blocks).  The
        SIGKILL goes out even if the direct child has exited meanwhile; the wait ends early only if the whole group is gone."""
        grace = self.kill_grace_s if grace_s is None else grace_s
        with self._lock:
            self.terminated = True
            started = self._pgid is not None
        if not started:
            return                                # run() has not started the process; it will see ``terminated``
        self._signal_group(signal.SIGTERM)

        def follow_up():
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and self.group_alive():
                time.sleep(0.05)
            self._signal_group(signal.SIGKILL)    # whole group, even if the direct child is long gone (errors ignored)
        threading.Thread(target=follow_up, name="runner-kill", daemon=True).start()

    def wait_gone(self, timeout_s: float) -> bool:
        """Wait until the child has exited (for shutdown / tests)."""
        proc = self._proc
        if proc is None:
            return True
        try:
            proc.wait(timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False


EXIT_MESSAGES = {
    0: "detection finished",
    1: "the detector rejected this scene (see the log)",
    2: "prerequisite or recorder timeout: is the scene camera running?",
}


def exit_message(rc: Optional[int]) -> str:
    if rc is None:
        return "detector did not finish"
    if rc in EXIT_MESSAGES:
        return EXIT_MESSAGES[rc]
    if rc < 0:
        try:
            return f"detector was terminated (signal {signal.Signals(-rc).name})"
        except ValueError:
            return f"detector was terminated (signal {-rc})"
    if rc == 127:
        return "the detector command could not be started (see the log)"
    return f"detector failed with exit code {rc} (see the log)"
