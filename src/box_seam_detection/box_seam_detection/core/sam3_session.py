# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/sam3_session.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Parent-side manager for the persistent SAM3 GPU worker.

One subprocess per run (model loaded once), JSON-lines protocol, offline
env forced (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) so nothing can download.

Optional cross-run daemon (env ``AUTOLABEL_V2_SAM3_DAEMON``):
  off  (default)  exactly the per-run subprocess worker above.
  auto            use / spawn the persistent daemon (``sam3_daemon.py``,
                  Unix socket); on any daemon problem fall back to the
                  per-run subprocess, recording why in
                  ``daemon_fallback_reason``.
  on              daemon required; failure raises Sam3LiveUnavailable.
``mode`` is "daemon" or "subprocess"; ``connect_s``/``spawn_s``/``wait_s``
carry the daemon timing for logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import sam3_daemon as _d
from .sam3_live import (BOOK_GRASPING_PYTHON, FALLBACK_PROMPT,
                        PRIMARY_PROMPT, Sam3LiveUnavailable)

WORKER_SCRIPT = Path(__file__).resolve().parent / "sam3_worker.py"


def daemon_mode_from_env() -> str:
    """off | auto | on (unknown values mean off: never surprise the caller)."""
    v = os.environ.get(_d.ENV_MODE, "off").strip().lower()
    return v if v in ("off", "auto", "on") else "off"


class _DaemonHandle:
    """``proc`` stand-in for daemon mode: ``poll()`` is None while the daemon
    is alive (callers such as the live detector check ``session.proc``)."""

    def __init__(self, ddir: Path):
        self.ddir = Path(ddir)

    def poll(self):
        return None if _d.verified_daemon_pid(
            _d.read_pidfile(self.ddir)) else 1


class Sam3Worker:
    """Context manager owning one persistent worker subprocess (or a handle
    on the cross-run SAM3 daemon, see the module docstring)."""

    mode = "subprocess"          # "daemon" | "subprocess"
    daemon_mode = "off"          # requested: off | auto | on
    daemon_fallback_reason: str | None = None
    connect_s = spawn_s = wait_s = 0.0
    daemon_spawned = False
    daemon_hello: dict | None = None
    _daemon_dir: Path | None = None
    _daemon_expected: dict | None = None

    def __init__(self, python_bin: str = BOOK_GRASPING_PYTHON,
                 threshold: float = 0.5, stderr_log: str | Path | None = None,
                 daemon_mode: str | None = None):
        self._daemon_mode_arg = daemon_mode
        self.python_bin = str(python_bin)
        self.threshold = float(threshold)
        self.proc: subprocess.Popen | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self.model_revision: str | None = None
        self.checkpoint_sha256: str | None = None
        self.stderr_log = str(stderr_log) if stderr_log is not None else None
        self._stderr_fh = None
        self._stderr_thread = None

    # ------------------------------------------------------------ daemon

    def start(self) -> "Sam3Worker":
        mode = (self._daemon_mode_arg or daemon_mode_from_env()).lower()
        self.daemon_mode = mode if mode in ("off", "auto", "on") else "off"
        if self.daemon_mode == "off":
            return self._start_subprocess()
        try:
            self._start_daemon()
        except Sam3LiveUnavailable as exc:
            if self.daemon_mode == "on":
                raise
            return self._fallback(f"start: {exc}")
        return self

    def _ensure_daemon(self) -> None:
        try:
            info = _d.ensure_daemon(self.python_bin, self._daemon_dir)
        except _d.DaemonError as exc:
            raise Sam3LiveUnavailable(f"sam3 daemon unavailable: {exc}") \
                from exc
        self.connect_s = float(info["connect_s"])
        self.spawn_s = float(info["spawn_s"])
        self.wait_s = float(info["wait_s"])
        self.daemon_spawned = bool(info["spawned"])
        self.daemon_hello = info["hello"]
        if info["hello"].get("model_revision"):
            self.model_revision = str(info["hello"]["model_revision"])

    def _start_daemon(self) -> None:
        try:
            self._daemon_dir = _d.ensure_dir(_d.daemon_dir())
            self._daemon_expected = _d.expected_identity(self.python_bin)
        except (_d.DaemonError, OSError) as exc:
            raise Sam3LiveUnavailable(
                f"sam3 daemon dir unusable: {exc}") from exc
        self._ensure_daemon()
        self.mode = "daemon"
        self.proc = _DaemonHandle(self._daemon_dir)
        if self.stderr_log is not None:
            Path(self.stderr_log).parent.mkdir(parents=True, exist_ok=True)
            with open(self.stderr_log, "w", encoding="utf-8") as fh:
                fh.write(f"sam3 daemon mode; daemon log: "
                         f"{_d.log_path(self._daemon_dir)}\n")

    def _ensure_no_daemon(self) -> None:
        """Never run a private worker next to a live daemon (two SAM3 models
        on a shared GPU). Graceful stop, then the verified-SIGTERM path; if
        the daemon cannot be confirmed gone, raise instead of falling back."""
        try:
            d = self._daemon_dir or _d.daemon_dir()
            gone = _d.stop_daemon(d, wait_s=30.0, allow_sigterm=True)
        except (_d.DaemonError, OSError) as exc:
            raise Sam3LiveUnavailable(
                f"sam3 daemon could not be confirmed stopped before "
                f"starting a private worker: {exc}") from exc
        if gone and not _d.name_is_free():     # machine-wide check
            gone = False
        if not gone:
            raise Sam3LiveUnavailable(
                "sam3 daemon still alive and would not stop; refusing to "
                "start a second SAM3 model on the shared GPU")

    def _fallback(self, why: str) -> "Sam3Worker":
        self._ensure_no_daemon()
        self.daemon_fallback_reason = why
        self.mode = "subprocess"
        self.proc = None
        return self._start_subprocess()

    def _query_daemon(self, req: dict, timeout_s: int) -> dict:
        """One request via the daemon; one respawn-and-retry on a transport
        failure / identity mismatch / GPU-error exit. Raises
        Sam3LiveUnavailable when both attempts fail."""
        last: Exception | None = None
        for attempt in (0, 1):
            try:
                resp, hello, connect_s = _d.request(
                    self._daemon_dir, req, self._daemon_expected,
                    float(timeout_s))
                self.connect_s = connect_s
                self.daemon_hello = hello
                if resp.get("daemon_exiting"):
                    last = Sam3LiveUnavailable(
                        f"sam3 daemon exited after GPU error: "
                        f"{resp.get('reason')}")
                    # ok:false reply for the caller if we cannot retry
                    if attempt == 1:
                        raise last
                    self._ensure_daemon()
                    continue
                return resp
            except _d.DaemonTimeout as exc:   # no retry: would double the wait
                raise Sam3LiveUnavailable(
                    f"sam3 daemon timed out: {exc}") from exc
            except Sam3LiveUnavailable:
                raise
            except _d.DaemonError as exc:
                last = exc
                if attempt == 1:
                    break
                self._ensure_daemon()
        raise Sam3LiveUnavailable(f"sam3 daemon query failed: {last}")

    # --------------------------------------------------------- subprocess

    def _start_subprocess(self) -> "Sam3Worker":
        self.mode = "subprocess"
        exe = Path(self.python_bin)
        if not exe.is_file():
            raise Sam3LiveUnavailable(
                f"SAM3 interpreter not found at {self.python_bin}")
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        _d.child_env_without_blas(env)   # worker env as before the BLAS pin
        if self.stderr_log is not None:
            Path(self.stderr_log).parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = open(self.stderr_log, "w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                [str(exe), str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        except OSError as exc:
            raise Sam3LiveUnavailable(
                f"sam3 worker spawn failed: {exc}") from exc
        if self.proc.stderr is not None:
            import threading as _th

            def _pump() -> None:
                try:
                    assert self.proc is not None
                    for line in self.proc.stderr:
                        if self._stderr_fh is not None:
                            self._stderr_fh.write(line)
                            self._stderr_fh.flush()
                except (OSError, ValueError):
                    pass

            self._stderr_thread = _th.Thread(target=_pump, daemon=True)
            self._stderr_thread.start()
        # Consume the handshake line before any query: proves the model is
        # loaded and pins the revision. Init-failure payloads raise here.
        line = self._readline(timeout_s=900)
        try:
            hello = json.loads(line) if line else None
        except ValueError as exc:
            raise Sam3LiveUnavailable(
                f"sam3 worker handshake unparseable: {exc}") from exc
        if not isinstance(hello, dict):
            raise Sam3LiveUnavailable(
                f"sam3 worker handshake missing: {(line or '')[:200]}")
        if hello.get("type") != "handshake":
            raise Sam3LiveUnavailable(
                f"sam3 worker unavailable: {hello.get('reason', line[:200])}")
        if hello.get("model_revision"):
            self.model_revision = str(hello["model_revision"])
        return self

    def _readline(self, timeout_s: int = 300) -> str:
        """Read one stdout line with a REAL timeout via select/poll.

        The old ThreadPoolExecutor wrapper left the blocking ``readline``
        stuck in its thread, so a 1 s timeout still took 2.5 s against a
        sleeping worker. ``select`` on the pipe fd bounds the wait itself;
        on timeout the worker is terminated and marked dead so subsequent
        captures fall back per the chain instead of hanging again.
        """
        import select

        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = timeout_s
        try:
            ready, _, _ = select.select([fd], [], [], float(deadline))
        except (OSError, ValueError) as exc:
            raise Sam3LiveUnavailable(
                f"sam3 worker read failed: {exc}") from exc
        if not ready:
            self._kill_dead("sam3 worker read timed out")
            raise Sam3LiveUnavailable("sam3 worker read timed out")
        try:
            return self.proc.stdout.readline()
        except (OSError, ValueError) as exc:
            self._kill_dead(f"sam3 worker read failed: {exc}")
            raise Sam3LiveUnavailable(
                f"sam3 worker read failed: {exc}") from exc

    def _kill_dead(self, why: str) -> None:
        """Terminate a wedged worker and mark it dead (proc=None)."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.kill()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass

    def query(self, image_path: str, timeout_s: int = 300,
              gates: dict | None = None) -> dict:
        """Send one image, return the worker's response dict (ok true/false).

        Raises Sam3LiveUnavailable on transport failure (EOF, timeout,
        unparseable). ``gates`` (optional) = per-request gate overrides for
        sam3_worker (sent as ``req["gates"]``; omitted when None, so the
        request is byte-identical to before). A per-capture ``ok: false`` payload is RETURNED, not
        raised, so the caller can record the reason and try the next source.
        """
        if self.mode == "daemon":
            return self._query_via_daemon(image_path, timeout_s, gates)
        if self.proc is None or self.proc.poll() is not None:
            raise Sam3LiveUnavailable("sam3 worker process is not running")
        with self._lock:
            self._seq += 1
            req = {"id": self._seq, "image": str(image_path),
                   "prompts": [PRIMARY_PROMPT, FALLBACK_PROMPT],
                   "threshold": self.threshold}
            if gates is not None:
                req["gates"] = gates
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise Sam3LiveUnavailable(
                    f"sam3 worker write failed: {exc}") from exc
            assert self.proc.stdout is not None
            line = self._readline(timeout_s=int(timeout_s))
        if not line:
            raise Sam3LiveUnavailable(
                "sam3 worker died (EOF on stdout)")
        try:
            resp = json.loads(line)
        except ValueError as exc:
            raise Sam3LiveUnavailable(
                f"sam3 worker output unparseable: {exc}") from exc
        if not isinstance(resp, dict) or resp.get("id") != req["id"]:
            raise Sam3LiveUnavailable(
                f"sam3 worker id mismatch: {line.strip()[:200]}")
        if resp.get("ok", False):
            if resp.get("model_revision"):
                self.model_revision = str(resp["model_revision"])
            return resp
        if resp.get("reason", "").startswith("worker_init_failed"):
            raise Sam3LiveUnavailable(
                f"sam3 worker unavailable: {resp.get('reason')}")
        return resp

    def _query_via_daemon(self, image_path: str, timeout_s: int,
                          gates: dict | None = None) -> dict:
        with self._lock:
            self._seq += 1
            seq = self._seq
            req = {"id": f"{uuid.uuid4().hex[:12]}-{seq}",
                   "image": os.path.abspath(str(image_path)),
                   "prompts": [PRIMARY_PROMPT, FALLBACK_PROMPT],
                   "threshold": self.threshold}
            if gates is not None:
                req["gates"] = gates
            try:
                resp = self._query_daemon(req, timeout_s)
            except Sam3LiveUnavailable as exc:
                if self.daemon_mode == "on":
                    raise
                # auto: today's per-run subprocess, then answer this query
                self._fallback(f"query: {exc}")
        if self.mode != "daemon":
            if gates is None:      # keep the exact old call (test doubles, subclasses)
                return self.query(image_path, timeout_s=timeout_s)
            return self.query(image_path, timeout_s=timeout_s, gates=gates)
        resp["id"] = seq   # caller sees the same id as subprocess mode
        if resp.get("ok", False):
            if resp.get("model_revision"):
                self.model_revision = str(resp["model_revision"])
            return resp
        if resp.get("reason", "").startswith("worker_init_failed"):
            raise Sam3LiveUnavailable(
                f"sam3 worker unavailable: {resp.get('reason')}")
        return resp

    def close(self) -> None:
        if self.mode == "daemon":
            self.proc = None   # the daemon outlives the session (idles out)
            return
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=60)
        except (OSError, subprocess.SubprocessError):
            try:
                proc.kill()
            except (OSError, subprocess.SubprocessError):
                pass
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except (OSError, ValueError):
                pass
            self._stderr_fh = None

    def __enter__(self) -> "Sam3Worker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
