# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/sam3_daemon.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Persistent SAM3 daemon (speed step S2) + the client-side helpers for it.

The daemon is the SAM3 model loaded ONCE in the book_grasping python and
serving the SAME request/response protocol as ``sam3_worker.py`` (it calls
``sam3_worker._handle_one``, so results are bit-identical) over a Unix
stream socket, one JSON object per line. It idles out after
``AUTOLABEL_V2_SAM3_DAEMON_IDLE_S`` seconds (default 1200).

Layout (``$AUTOLABEL_V2_SAM3_DAEMON_DIR`` | ``$XDG_RUNTIME_DIR/autolabel_v2_sam3``
| ``/tmp/autolabel_v2_sam3-<uid>``; dir 0700, socket 0600)::

    sam3.sock        listening socket (bound only once the model is loaded
                     AND one warm-up inference has run = readiness)
    daemon.pid       JSON: pid, pgid, start_time (/proc/<pid>/stat), socket,
                     protocol, ready
    daemon.log       daemon stdout/stderr
    spawn.lock       client flock serialising spawn / replace
    daemon.lock      daemon flock held for its whole life (singleton guard)

Protocol: on EVERY connection the daemon first sends a handshake line
``{type: handshake, protocol, model_revision, worker_sha256, daemon_sha256,
torch, transformers, python, device, gpu_peak_mem_mb, pid}``. Then the
client sends one request line (``{"id", "image", "prompts", "threshold"}``)
and gets one response line, or sends ``{"cmd": "shutdown"}``. Connections are
served strictly one at a time (a second client waits in the listen queue).
A CUDA/GPU error reply carries ``daemon_exiting: true`` and the daemon exits.

Process safety: the daemon is only ever stopped by the graceful socket
command; the one fallback is SIGTERM to the pid from the pidfile AFTER
verifying /proc/<pid>/cmdline contains ``sam3_daemon.py`` and the start time
matches the pidfile. Nothing here kills by name or pattern.

CLI: ``sam3_daemon.py --serve`` (what clients spawn) | ``--status`` |
``--stop`` [``--force``].
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = 1
HERE = Path(__file__).resolve().parent
WORKER_SCRIPT = HERE / "sam3_worker.py"
DAEMON_SCRIPT = HERE / "sam3_daemon.py"

ENV_MODE = "AUTOLABEL_V2_SAM3_DAEMON"
ENV_DIR = "AUTOLABEL_V2_SAM3_DAEMON_DIR"
ENV_IDLE = "AUTOLABEL_V2_SAM3_DAEMON_IDLE_S"
ENV_STUB = "AUTOLABEL_V2_SAM3_DAEMON_STUB"          # tests: no model
ENV_MODEL_ID = "SAM3_MODEL_ID"                      # same var sam3_worker reads
DEFAULT_MODEL_ID = "facebook/sam3"                  # == sam3_worker.MODEL_ID
ENV_LOCK_NAME = "AUTOLABEL_V2_SAM3_DAEMON_LOCK_NAME"  # tests: never the real one
ENV_READY_S = "AUTOLABEL_V2_SAM3_DAEMON_READY_S"
DEFAULT_READY_S = 180.0
DEFAULT_IDLE_S = 1200.0
CLIENT_SEND_TIMEOUT_S = 30.0   # daemon: max wait for a client's request line

_GPU_ERR_RE = re.compile(
    r"cuda|cublas|cudnn|out of memory|device-side assert|nccl", re.I)


class DaemonError(RuntimeError):
    """Daemon-side/transport problem (client maps it to Sam3LiveUnavailable)."""


class DaemonTimeout(DaemonError):
    """A bounded wait (connect/handshake/response) ran out."""


# ----------------------------------------------------------------- paths

def daemon_dir() -> Path:
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return Path(xdg) / "autolabel_v2_sam3"
    return Path(f"/tmp/autolabel_v2_sam3-{os.getuid()}")


def ensure_dir(d: Path) -> Path:
    d = Path(d)
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = d.stat()
    if st.st_uid != os.getuid():
        raise DaemonError(f"daemon dir {d} is not owned by this user")
    if (st.st_mode & 0o777) != 0o700:
        os.chmod(d, 0o700)
    return d


def sock_path(d: Path) -> Path:
    return Path(d) / "sam3.sock"


def pid_path(d: Path) -> Path:
    return Path(d) / "daemon.pid"


def log_path(d: Path) -> Path:
    return Path(d) / "daemon.log"


def lock_name() -> str:
    """Abstract-namespace socket name: the machine-wide (per-uid, per-network-
    namespace) singleton guard, independent of daemon_dir()/env."""
    return os.environ.get(ENV_LOCK_NAME) or f"autolabel_v2_sam3-{os.getuid()}"


def bind_singleton(wait_s: float = 0.0):
    """Bind the abstract name. Returns the bound socket (hold it for as long
    as the guard must stay held; the kernel releases it at process exit) or
    None when another process holds it. ``wait_s`` retries (a probing client
    may hold it for an instant)."""
    end = time.monotonic() + wait_s
    while True:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind("\0" + lock_name())
            return s
        except OSError as exc:
            s.close()
            if exc.errno != errno.EADDRINUSE:
                raise
        if time.monotonic() >= end:
            return None
        time.sleep(0.05)


def name_is_free() -> bool:
    """True = no daemon anywhere holds the singleton (probe bind, released
    immediately)."""
    probe = bind_singleton(0.0)
    if probe is None:
        return False
    probe.close()
    return True


def wait_name_free(wait_s: float) -> bool:
    end = time.monotonic() + wait_s
    while True:
        if name_is_free():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_id_in_use() -> str:
    return os.environ.get(ENV_MODEL_ID, DEFAULT_MODEL_ID)


def local_model_snapshot(model_id: str):
    """Commit hash of the locally cached model (refs/main in the HF cache),
    by plain file reads so client and daemon agree whatever huggingface_hub
    version each runs (``snapshot_download(local_files_only)`` fails on this
    machine's partially cached snapshot, so ``model_revision`` is null).
    None when it cannot be determined."""
    if os.environ.get(ENV_STUB):
        return os.environ.get(ENV_STUB + "_REVISION", "stub")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser(
            "~/.cache/huggingface"), "hub")
    ref = Path(hub) / ("models--" + model_id.replace("/", "--")) \
        / "refs" / "main"
    try:
        return ref.read_text().strip() or None
    except OSError:
        return None


def expected_identity(python_bin=None) -> dict:
    """What a daemon must report to be usable by THIS client's code."""
    mid = model_id_in_use()
    return {"protocol": PROTOCOL,
            "worker_sha256": file_sha256(WORKER_SCRIPT),
            "daemon_sha256": file_sha256(DAEMON_SCRIPT),
            "model_id": mid,
            "model_snapshot": local_model_snapshot(mid),
            "python_executable": (os.path.realpath(str(python_bin))
                                  if python_bin else None)}


def identity_mismatch(hello: dict, exp: dict) -> str | None:
    for k in ("protocol", "worker_sha256", "daemon_sha256", "model_id"):
        if hello.get(k) != exp.get(k):
            return f"{k} differs"
    if exp.get("python_executable") is not None and \
            hello.get("python_executable") != exp.get("python_executable"):
        return "python_executable differs"
    # snapshot (file-derived commit): compared when the client could read it
    if exp.get("model_snapshot") is not None and \
            hello.get("model_snapshot") != exp.get("model_snapshot"):
        return "model_snapshot differs"
    # hub-reported revision: compared only when both sides know one
    if exp.get("model_revision") is not None and \
            hello.get("model_revision") is not None and \
            hello.get("model_revision") != exp.get("model_revision"):
        return "model_revision differs"
    return None


# ------------------------------------------------------------ /proc helpers

def _stat_fields(pid: int):
    """(state, start_time_ticks) from /proc/<pid>/stat or None."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = raw[raw.rindex(")") + 2:].split()
    try:
        return rest[0], int(rest[19])
    except (IndexError, ValueError):
        return None


def proc_start_time(pid: int):
    f = _stat_fields(pid)
    return f[1] if f else None


def pid_running(pid: int) -> bool:
    f = _stat_fields(pid)
    return f is not None and f[0] not in ("Z", "X")


def verified_daemon_pid(info: dict | None) -> bool:
    """PID-reuse guard: alive, start time matches the pidfile, cmdline is us."""
    if not info:
        return False
    try:
        pid = int(info["pid"])
        want = int(info["start_time"])
    except (KeyError, TypeError, ValueError):
        return False
    f = _stat_fields(pid)
    if f is None or f[0] in ("Z", "X") or f[1] != want:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"sam3_daemon.py" in cmd


def read_pidfile(d: Path):
    try:
        info = json.loads(pid_path(d).read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _write_pidfile(d: Path, ready: bool) -> None:
    pid = os.getpid()
    info = {"pid": pid, "pgid": os.getpgid(0),
            "start_time": proc_start_time(pid),
            "socket": str(sock_path(d)), "protocol": PROTOCOL,
            "ready": bool(ready)}
    tmp = Path(d) / f".daemon.pid.{pid}"
    tmp.write_text(json.dumps(info))
    os.chmod(tmp, 0o600)
    os.replace(tmp, pid_path(d))


def cleanup_stale(d: Path) -> None:
    """Remove pidfile/socket left by a daemon that is no longer alive."""
    info = read_pidfile(d)
    if verified_daemon_pid(info):
        return
    for p in (pid_path(d), sock_path(d)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# --------------------------------------------------------------- line I/O

def _read_line(sock, timeout: float, buf: bytearray):
    """One line (bytes, no newline) or None on EOF; DaemonTimeout on timeout."""
    deadline = time.monotonic() + timeout
    while b"\n" not in buf:
        left = deadline - time.monotonic()
        if left <= 0:
            raise DaemonTimeout("timed out waiting for a line")
        sock.settimeout(left)
        try:
            chunk = sock.recv(1 << 16)
        except socket.timeout as exc:
            raise DaemonTimeout("timed out waiting for a line") from exc
        if not chunk:
            return None
        buf += chunk
    line, _, rest = bytes(buf).partition(b"\n")
    buf[:] = rest
    return line


def _send(sock, obj: dict, timeout: float = 30.0) -> None:
    sock.settimeout(timeout)
    sock.sendall((json.dumps(obj) + "\n").encode())


# ------------------------------------------------------------------ client

def connect(d: Path, connect_timeout: float = 5.0):
    """Connect to the socket; None when no daemon is listening."""
    sp = str(sock_path(d))
    if len(sp.encode()) > 100:
        raise DaemonError(f"socket path too long for AF_UNIX: {sp}")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(connect_timeout)
    try:
        s.connect(sp)
    except (FileNotFoundError, ConnectionRefusedError):
        s.close()
        return None
    except socket.timeout as exc:
        s.close()
        raise DaemonTimeout("connect timed out") from exc
    except OSError as exc:
        s.close()
        raise DaemonError(f"connect failed: {exc}") from exc
    return s


def open_session(d: Path, handshake_timeout: float):
    """Connect + read the handshake. Returns (sock, buf, hello) or None."""
    s = connect(d)
    if s is None:
        return None
    buf = bytearray()
    try:
        line = _read_line(s, handshake_timeout, buf)
    except DaemonError:
        s.close()
        raise
    except OSError as exc:
        s.close()
        raise DaemonError(f"handshake read failed: {exc}") from exc
    try:
        hello = json.loads(line) if line else None
    except ValueError:
        hello = None
    if not isinstance(hello, dict) or hello.get("type") != "handshake":
        s.close()
        raise DaemonError("daemon handshake missing/unparseable")
    return s, buf, hello


def request(d: Path, req: dict, expected: dict, timeout_s: float,
            handshake_timeout: float | None = None):
    """One request on a fresh connection. Returns (resp, hello, connect_s).

    Raises DaemonMismatch when the daemon's identity differs from expected,
    DaemonError/DaemonTimeout on transport problems (incl. response id
    mismatch)."""
    t0 = time.monotonic()
    sess = open_session(d, handshake_timeout if handshake_timeout is not None
                        else timeout_s)
    if sess is None:
        raise DaemonError("no daemon listening")
    s, buf, hello = sess
    connect_s = time.monotonic() - t0
    try:
        why = identity_mismatch(hello, expected)
        if why:
            raise DaemonMismatch(why, hello)
        try:
            _send(s, req)
            line = _read_line(s, timeout_s, buf)
        except DaemonError:
            raise
        except OSError as exc:
            raise DaemonError(f"daemon i/o failed: {exc}") from exc
        if not line:
            raise DaemonError("daemon closed the connection (EOF)")
        try:
            resp = json.loads(line)
        except ValueError as exc:
            raise DaemonError(f"daemon response unparseable: {exc}") from exc
        if not isinstance(resp, dict) or resp.get("id") != req.get("id"):
            raise DaemonError(
                f"daemon id mismatch: {line.decode(errors='replace')[:200]}")
        return resp, hello, connect_s
    finally:
        s.close()


class DaemonMismatch(DaemonError):
    def __init__(self, why: str, hello: dict):
        super().__init__(f"daemon identity mismatch ({why})")
        self.hello = hello


def _try_handshake(d: Path, timeout: float):
    sess = open_session(d, timeout)
    if sess is None:
        return None
    s, _buf, hello = sess
    s.close()
    return hello


def _wait_gone(pid: int, wait_s: float) -> bool:
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if not pid_running(pid):
            return True
        time.sleep(0.05)
    return not pid_running(pid)


def stop_daemon(d: Path, wait_s: float = 30.0, allow_sigterm: bool = True) -> bool:
    """Graceful shutdown over the socket; verified SIGTERM only as fallback.
    Returns True when no daemon is left running."""
    d = Path(d)
    info = read_pidfile(d)
    pid = int(info["pid"]) if info and str(info.get("pid", "")).isdigit() \
        else None
    try:
        sess = open_session(d, wait_s)
    except DaemonError:
        sess = None
    if sess is not None:
        s, buf, _hello = sess
        try:
            _send(s, {"cmd": "shutdown"})
            _read_line(s, 10.0, buf)
        except (DaemonError, OSError):
            pass
        finally:
            s.close()
    if pid is not None and verified_daemon_pid(info):
        if not _wait_gone(pid, wait_s):
            if allow_sigterm and verified_daemon_pid(info):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                _wait_gone(pid, 20.0)
        if pid_running(pid) and verified_daemon_pid(info):
            return False
    # Authoritative: the machine-wide singleton must be free (covers a
    # deleted pidfile, a recreated daemon dir, a daemon still exiting).
    if not wait_name_free(min(wait_s, 15.0)):
        return False
    cleanup_stale(d)
    return True


class _SpawnLock:
    def __init__(self, d: Path, timeout: float):
        self.path = Path(d) / "spawn.lock"
        self.timeout = timeout
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        end = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > end:
                    self.fh.close()
                    raise DaemonTimeout("timed out waiting for spawn.lock")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()


ENV_KEEP_BLAS = "AUTOLABEL_V2_SAM3_DAEMON_KEEP_BLAS_ENV"
ENV_BLAS_PINNED = "AUTOLABEL_V2_BLAS_PINNED"   # set by entry scripts' setdefault


def child_env_without_blas(env: dict) -> dict:
    """The daemon runs torch CPU preprocessing: it must not inherit the
    single-thread OpenBLAS pin the detector processes set for their RANSAC
    loops (unless AUTOLABEL_V2_SAM3_DAEMON_KEEP_BLAS_ENV=1). Edits env."""
    if env.get(ENV_KEEP_BLAS) != "1" and env.get(ENV_BLAS_PINNED) == "1":
        env.pop("OPENBLAS_NUM_THREADS", None)   # only OUR pin, never a user's
        env.pop(ENV_BLAS_PINNED, None)
    return env


_SPAWNED: list = []   # Popen handles, polled so exited children are reaped


def _reap() -> None:
    for p in list(_SPAWNED):
        if p.poll() is not None:
            try:
                _SPAWNED.remove(p)
            except ValueError:      # another thread reaped it first
                pass


def _log_tail(d: Path, n: int = 400) -> str:
    try:
        return log_path(d).read_text(errors="replace")[-n:].strip()
    except OSError:
        return ""


def ready_timeout_from_env() -> float:
    try:
        v = float(os.environ.get(ENV_READY_S, DEFAULT_READY_S))
        return v if v > 0 else DEFAULT_READY_S
    except ValueError:
        return DEFAULT_READY_S


def _handshake_retry(d: Path, timeout: float):
    """_try_handshake with ONE retry on a transient EOF/timeout, so a healthy
    (e.g. just-busy) daemon is not treated as broken."""
    try:
        return _try_handshake(d, timeout)
    except DaemonError:
        time.sleep(0.5)
        return _try_handshake(d, timeout)


def ensure_daemon(python_bin: str, d: Path | None = None,
                  ready_timeout_s: float | None = None,
                  handshake_timeout_s: float = 60.0) -> dict:
    """Make sure a daemon with the expected identity is serving.

    Returns ``{"hello", "spawned", "replaced", "connect_s", "spawn_s",
    "wait_s"}``. Raises DaemonError when that cannot be achieved."""
    d = ensure_dir(d or daemon_dir())
    _reap()
    if ready_timeout_s is None:
        ready_timeout_s = ready_timeout_from_env()
    exp = expected_identity(python_bin)
    out = {"spawned": False, "replaced": False, "connect_s": 0.0,
           "spawn_s": 0.0, "wait_s": 0.0}
    t0 = time.monotonic()
    hello = _handshake_retry(d, handshake_timeout_s)
    if hello is not None and not identity_mismatch(hello, exp):
        out.update(hello=hello, connect_s=time.monotonic() - t0)
        return out
    with _SpawnLock(d, ready_timeout_s + 60.0):
        hello = _handshake_retry(d, handshake_timeout_s)
        if hello is not None and not identity_mismatch(hello, exp):
            out.update(hello=hello, connect_s=time.monotonic() - t0)
            return out
        info = read_pidfile(d)
        alive = verified_daemon_pid(info)
        starting = False
        if hello is not None:                    # stale code: replace it
            if not stop_daemon(d, wait_s=30.0, allow_sigterm=True):
                raise DaemonError("stale daemon would not shut down")
            out["replaced"] = True
        elif alive and not info.get("ready"):
            starting = True                      # someone's daemon is loading
        elif alive:                              # ready but not answering
            if not stop_daemon(d, wait_s=10.0, allow_sigterm=True):
                raise DaemonError("unresponsive daemon would not stop")
        else:
            cleanup_stale(d)
        proc = None
        if not starting:
            exe = Path(python_bin)
            if not exe.is_file():
                raise DaemonError(f"SAM3 interpreter not found at {python_bin}")
            env = dict(os.environ)
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env[ENV_DIR] = str(d)
            child_env_without_blas(env)
            ts = time.monotonic()
            with open(log_path(d), "ab") as logfh:
                try:
                    proc = subprocess.Popen(
                        [str(exe), str(DAEMON_SCRIPT), "--serve"],
                        stdin=subprocess.DEVNULL, stdout=logfh,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        close_fds=True, env=env)
                except OSError as exc:
                    raise DaemonError(f"daemon spawn failed: {exc}") from exc
            _SPAWNED.append(proc)
            out["spawned"] = True
            out["spawn_s"] = time.monotonic() - ts
        tw = time.monotonic()
        deadline = tw + ready_timeout_s
        while True:
            hello = _try_handshake(d, max(5.0, handshake_timeout_s)) \
                if sock_path(d).exists() else None
            if hello is not None:
                break
            if proc is not None and proc.poll() is not None:
                raise DaemonError(
                    f"daemon exited rc={proc.returncode} before ready: "
                    f"{_log_tail(d)}")
            if proc is None and not verified_daemon_pid(read_pidfile(d)):
                raise DaemonError(
                    f"starting daemon died before ready: {_log_tail(d)}")
            if time.monotonic() > deadline:
                raise DaemonTimeout(
                    f"daemon not ready after {ready_timeout_s:.0f}s")
            time.sleep(0.1)
        out["wait_s"] = time.monotonic() - tw
        why = identity_mismatch(hello, exp)
        if why:
            raise DaemonError(f"fresh daemon identity mismatch ({why})")
        out.update(hello=hello, connect_s=time.monotonic() - t0)
        return out


def status(d: Path | None = None) -> dict:
    d = Path(d or daemon_dir())
    info = read_pidfile(d)
    out = {"dir": str(d), "pidfile": info,
           "running": verified_daemon_pid(info), "handshake": None}
    if out["running"]:
        try:
            out["handshake"] = _try_handshake(d, 5.0)
        except DaemonError as exc:
            out["handshake_error"] = str(exc)
    return out


# ------------------------------------------------------------------ server

def _is_gpu_error(resp: dict) -> bool:
    if resp.get("ok"):
        return False
    return bool(_GPU_ERR_RE.search(str(resp.get("reason", ""))))


def _stub_handler(req: dict) -> dict:
    """Test double: no model. Special image values drive failure paths."""
    full = str(req.get("image", ""))
    img = os.path.basename(full)     # clients send absolute paths
    if img.startswith("sleep:"):
        time.sleep(float(img.split(":", 1)[1]))
    if img == "cuda_error":
        return {"ok": False, "id": req.get("id"),
                "reason": "inference_failed(RuntimeError:CUDA error: stub)"}
    if img == "raise":
        raise RuntimeError("stub handler exception")
    if img == "wrong_id":
        return {"ok": True, "id": "not-" + str(req.get("id"))}
    return {"ok": True, "id": req.get("id"), "stub": True, "image": full,
            "pid": os.getpid(), "prompts": req.get("prompts"),
            "threshold": req.get("threshold")}


class _Server:
    def __init__(self, d: Path, idle_s: float):
        self.d = d
        self.idle_s = idle_s
        self.stop = False
        self.stub = bool(os.environ.get(ENV_STUB))
        self.handler = _stub_handler
        self.revision = os.environ.get(
            ENV_STUB + "_REVISION", "stub") if self.stub else None
        self.device = "stub" if self.stub else None
        self.torch_v = self.tf_v = None
        self.srv = None
        self.lock_fh = None
        self.last = time.monotonic()
        ident = expected_identity()
        if self.stub:  # tests can fake an "old code" daemon
            ident["protocol"] = int(os.environ.get(
                ENV_STUB + "_PROTOCOL", ident["protocol"]))
            ident["worker_sha256"] = os.environ.get(
                ENV_STUB + "_WORKER_SHA", ident["worker_sha256"])
            ident["daemon_sha256"] = os.environ.get(
                ENV_STUB + "_DAEMON_SHA", ident["daemon_sha256"])
        ident.pop("model_revision", None)   # reported live below
        self.model_id = ident.pop("model_id")
        ident.pop("model_snapshot", None)
        ident.pop("python_executable", None)
        self.snapshot = local_model_snapshot(self.model_id)
        self.ident = ident

    def log(self, msg: str) -> None:
        print(f"sam3_daemon[{os.getpid()}] {time.strftime('%H:%M:%S')} {msg}",
              flush=True)

    def load(self) -> None:
        if self.stub:
            time.sleep(float(os.environ.get(ENV_STUB + "_LOAD_S", "0")))
            return
        try:  # imported as a package module
            from . import sam3_worker  # noqa: PLC0415 — shared inference code
        except ImportError:  # run as a file by the conda python (script dir on sys.path)
            import sam3_worker  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        model, processor, device, revision = sam3_worker.load_model(
            lambda m: self.log(m))
        self.revision, self.device = revision, device
        self.torch_v, self.tf_v = torch.__version__, transformers.__version__
        self.handler = lambda req: sam3_worker._handle_one(
            model, processor, device, req)
        # Warm-up: one real forward pass on a small synthetic image.
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        rng = np.random.default_rng(0)
        arr = rng.integers(90, 160, (480, 640, 3), dtype=np.uint8)
        wp = self.d / "warmup.png"
        Image.fromarray(arr).save(wp)
        try:
            t0 = time.monotonic()
            resp = self.handler({"id": "warmup", "image": str(wp),
                                 "prompts": [sam3_worker.PRIMARY_PROMPT],
                                 "threshold": 0.5})
            self.log(f"warm-up {time.monotonic() - t0:.2f}s "
                     f"reason={resp.get('reason')}")
        finally:
            try:
                wp.unlink()
            except OSError:
                pass
        if str(resp.get("reason", "")).startswith("inference_failed"):
            raise RuntimeError(f"warm-up failed: {resp.get('reason')}")

    def handshake(self) -> dict:
        peak = 0.0
        if not self.stub:
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            except Exception:  # noqa: BLE001
                pass
        return {"type": "handshake", **self.ident, "model_id": self.model_id,
                "model_snapshot": self.snapshot,
                "python_executable": os.path.realpath(sys.executable),
                "model_revision": self.revision, "torch": self.torch_v,
                "transformers": self.tf_v,
                "python": sys.version.split()[0], "device": self.device,
                "gpu_peak_mem_mb": peak, "pid": os.getpid()}

    def release_gpu_cache(self) -> None:
        """Hand the allocator's cached blocks back after each request so an
        idle daemon holds ~model size, not the activation peak (the GPU is
        shared with other nodes)."""
        if self.stub:
            log = os.environ.get(ENV_STUB + "_CACHE_LOG")
            if log:                       # tests: proof the release ran
                with open(log, "a") as fh:
                    fh.write("empty_cache\n")
            return
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def serve_conn(self, conn) -> None:
        buf = bytearray()
        try:
            _send(conn, self.handshake(), timeout=10.0)
            line = _read_line(conn, CLIENT_SEND_TIMEOUT_S, buf)
            if not line:
                return                       # status-style probe: just closed
            try:
                req = json.loads(line)
            except ValueError:
                _send(conn, {"ok": False, "id": None,
                             "reason": "bad_request_json"})
                return
            if isinstance(req, dict) and req.get("cmd") == "shutdown":
                self.log("shutdown requested")
                self.stop = True
                _send(conn, {"ok": True, "cmd": "shutdown"})
                return
            if not isinstance(req, dict) or req.get("id") is None:
                _send(conn, {"ok": False, "id": None,
                             "reason": "request needs an id"})
                return
            try:
                try:
                    resp = self.handler(req)
                except Exception as exc:  # noqa: BLE001
                    resp = {"ok": False, "id": req.get("id"),
                            "reason": f"inference_failed("
                                      f"{type(exc).__name__}:{exc})"}
                resp["model_revision"] = self.revision
                if _is_gpu_error(resp):
                    resp["daemon_exiting"] = True
                    self.stop = True
                    self.log(f"GPU error, exiting: {resp.get('reason')}")
                _send(conn, resp, timeout=60.0)
            finally:
                self.release_gpu_cache()   # also on errors / failed send
        except (DaemonError, OSError) as exc:
            self.log(f"connection error: {exc}")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self.last = time.monotonic()

    def run(self) -> int:
        os.umask(0o077)
        # Machine-wide singleton BEFORE touching anything or loading the model.
        self.singleton = bind_singleton(wait_s=3.0)
        if self.singleton is None:
            self.log(f"another daemon holds @{lock_name()}; exiting "
                     f"without loading")
            return 0
        ensure_dir(self.d)
        self.lock_fh = open(self.d / "daemon.lock", "a+")
        try:
            fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.log("another daemon holds daemon.lock; exiting")
            return 0
        rc = 0
        try:
            # We hold daemon.lock, so any pidfile/socket here is stale.
            for p in (pid_path(self.d), sock_path(self.d)):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            _write_pidfile(self.d, ready=False)
            self.log(f"starting (stub={self.stub}) dir={self.d}")
            t0 = time.monotonic()
            self.load()
            self.release_gpu_cache()
            self.log(f"model ready in {time.monotonic() - t0:.1f}s")
            # Graceful signal handling only once ready: during load the
            # default SIGTERM action ends a loading/hung daemon at once.
            signal.signal(signal.SIGTERM,
                          lambda *_: setattr(self, "stop", True))
            signal.signal(signal.SIGINT,
                          lambda *_: setattr(self, "stop", True))
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.srv = srv
            srv.bind(str(sock_path(self.d)))
            os.chmod(sock_path(self.d), 0o600)
            srv.listen(16)
            _write_pidfile(self.d, ready=True)
            sock_ino = os.stat(sock_path(self.d)).st_ino
            next_check = time.monotonic() + 1.0
            self.log(f"listening (idle exit {self.idle_s:.0f}s)")
            self.last = time.monotonic()
            while not self.stop:
                if time.monotonic() - self.last > self.idle_s:
                    self.log("idle timeout, exiting")
                    break
                r, _, _ = select.select([srv], [], [], 0.25)
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + 1.0
                    try:
                        gone = os.stat(sock_path(self.d)).st_ino != sock_ino
                    except OSError:
                        gone = True
                    if gone:
                        self.log("socket path/inode disappeared, exiting")
                        break
                if not r:
                    continue
                conn, _ = srv.accept()
                self.serve_conn(conn)
        except Exception as exc:  # noqa: BLE001
            self.log(f"fatal: {type(exc).__name__}: {exc}")
            rc = 1
        finally:
            if self.srv is not None:
                self.srv.close()
            info = read_pidfile(self.d)
            if info and info.get("pid") == os.getpid():
                for p in (sock_path(self.d), pid_path(self.d)):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            self.log("exited")
        return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--stop", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="--stop: allow verified SIGTERM if graceful fails")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)
    d = Path(args.dir) if args.dir else daemon_dir()
    if args.status:
        st = status(d)
        print(json.dumps(st, indent=2))
        return 0 if st["running"] else 1
    if args.stop:
        ok = stop_daemon(d, wait_s=30.0, allow_sigterm=bool(args.force))
        print("stopped" if ok else "daemon still running (try --force)")
        return 0 if ok else 1
    try:
        idle = float(os.environ.get(ENV_IDLE, DEFAULT_IDLE_S))
    except ValueError:
        idle = DEFAULT_IDLE_S
    return _Server(d, idle).run()


if __name__ == "__main__":
    raise SystemExit(main())
