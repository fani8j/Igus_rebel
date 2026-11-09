"""The detector run as a DetectBox action goal instead of a subprocess.  ``ActionRunner`` has the same small surface the job
state machine uses on ``jobs.ProcessRunner`` (``terminate``, ``kill_now``, ``wait_gone``, ``terminated``, ``pid``), so a
Detect cancel, a Software stop and the UI shutdown treat both the same way.  No ROS imports: the bridge does the work."""
import threading
from typing import Callable, Optional, Tuple

from . import bridge as B


class ActionRunner:
    def __init__(self, bridge: B.Bridge, action: str, goal: dict, timeout_s: float):
        self.bridge, self.action, self.goal, self.timeout_s = bridge, action, dict(goal), float(timeout_s)
        self.cancel = threading.Event()
        self.terminated = False
        self.pid: Optional[int] = None            # no process; kept so code that looks at runner.pid keeps working
        self._lock = threading.Lock()
        self._started = False
        self._done = threading.Event()

    def run(self, on_feedback: Callable[[str, int, str], None]) -> Tuple[bool, dict]:
        """Blocks the calling (job worker) thread until the result.  Never raises: a bridge exception is an error result."""
        with self._lock:
            if self.terminated:
                self._done.set()
                return False, {"cancelled": True}
            self._started = True
        try:
            ok, info = self.bridge.detect(self.action, self.goal, on_feedback, self.cancel, self.timeout_s)
            info = dict(info or {})
        except Exception as e:  # noqa: BLE001
            ok, info = False, {"error": f"detect call failed: {e}"}
        finally:
            self._done.set()
        if self.terminated and not ok:
            info["cancelled"] = True
        return bool(ok), info

    def terminate(self, grace_s: Optional[float] = None) -> None:
        """Ask the bridge to cancel the goal (non-blocking, any thread, idempotent)."""
        with self._lock:
            self.terminated = True
        self.cancel.set()

    def kill_now(self) -> None:
        self.cancel.set()

    def wait_gone(self, timeout_s: float) -> bool:
        with self._lock:
            started = self._started
        return True if not started else self._done.wait(timeout_s)
