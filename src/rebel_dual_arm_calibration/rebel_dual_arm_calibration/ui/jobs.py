"""One background subprocess at a time (the intrinsics calibration).  Only our own child is ever signalled."""
import collections
import os
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional

TAIL_LINES = 60


class JobBusy(RuntimeError):
    pass


class _Job:
    def __init__(self, job_id: str, kind: str, cmd: List[str], result_file: Optional[str]):
        self.id, self.kind, self.cmd, self.result_file = job_id, kind, list(cmd), result_file
        self.state = "running"
        self.returncode: Optional[int] = None
        self.tail = collections.deque(maxlen=TAIL_LINES)
        self.started = time.time()
        self.proc: Optional[subprocess.Popen] = None

    def to_dict(self) -> dict:
        res = self.result_file if (self.state == "done" and self.result_file and os.path.exists(self.result_file)) else None
        return {"id": self.id, "kind": self.kind, "state": self.state, "returncode": self.returncode,
                "output_tail": list(self.tail), "result_file": res, "started_at": self.started,
                "command": " ".join(self.cmd)}


class JobManager:
    def __init__(self, keep: int = 10):
        self._lock = threading.Lock()
        self._jobs: "collections.OrderedDict[str, _Job]" = collections.OrderedDict()
        self._n = 0
        self._keep = keep

    def running(self) -> bool:
        with self._lock:
            return any(j.state == "running" for j in self._jobs.values())

    def start(self, kind: str, cmd: List[str], result_file: Optional[str] = None, cwd: Optional[str] = None) -> dict:
        with self._lock:
            if any(j.state == "running" for j in self._jobs.values()):
                raise JobBusy("another job is still running")
            self._n += 1
            job = _Job(f"job{self._n}", kind, cmd, result_file)
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                self._jobs.popitem(last=False)
            try:
                job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            errors="replace", cwd=cwd, start_new_session=True)
            except OSError as e:
                job.tail.append(f"cannot start {cmd[0]}: {e}")
                job.state, job.returncode = "failed", -1
                return job.to_dict()
        threading.Thread(target=self._pump, args=(job,), name=f"job-{job.id}", daemon=True).start()
        return job.to_dict()

    def _pump(self, job: _Job) -> None:
        proc = job.proc
        try:
            for line in proc.stdout:
                job.tail.append(line.rstrip("\n"))
        finally:
            rc = proc.wait()
            with self._lock:
                job.returncode = rc
                job.state = "done" if rc == 0 else "failed"

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {j.id: j.to_dict() for j in self._jobs.values()}

    def shutdown(self) -> None:
        """Terminate the process group of a still-running job (start_new_session made it its own group)."""
        with self._lock:
            procs = [j.proc for j in self._jobs.values() if j.state == "running" and j.proc is not None]
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
        deadline = time.time() + 3.0
        for p in procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
