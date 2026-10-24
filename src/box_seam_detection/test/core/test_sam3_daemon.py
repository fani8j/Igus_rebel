# Provenance: ported from rgbd_captures/tests/test_sam3_daemon.py (HEAD 6df5994); imports made package-relative.
"""Tests for the persistent SAM3 daemon (speed step S2). No GPU, no model:
the daemon runs with AUTOLABEL_V2_SAM3_DAEMON_STUB=1 (a stub handler).
Every daemon a test starts is stopped by recorded PID in the fixture."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


from box_seam_detection.core import sam3_daemon as D
from box_seam_detection.core import sam3_session as S
from box_seam_detection.core.sam3_live import Sam3LiveUnavailable


@pytest.fixture
def ddir(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="s3d"))
    pids = set()
    monkeypatch.setenv(D.ENV_DIR, str(d))
    monkeypatch.setenv(D.ENV_STUB, "1")
    monkeypatch.delenv(D.ENV_MODE, raising=False)
    monkeypatch.delenv(D.ENV_IDLE, raising=False)
    real_ensure = D.ensure_daemon

    def tracking(*a, **k):
        try:
            return real_ensure(*a, **k)
        finally:
            info = D.read_pidfile(d)
            if info:
                pids.add((info["pid"], info.get("start_time")))
    monkeypatch.setattr(D, "ensure_daemon", tracking)
    yield d
    D.stop_daemon(d, wait_s=10, allow_sigterm=True)
    for pid, st in pids:  # recorded PIDs only, start-time checked
        if D.pid_running(pid) and D.proc_start_time(pid) == st:
            os.kill(pid, signal.SIGTERM)
            D._wait_gone(pid, 10)
    shutil.rmtree(d, ignore_errors=True)


def _worker(mode="on", **kw):
    return S.Sam3Worker(python_bin=sys.executable, daemon_mode=mode, **kw)


def _pid(d):
    return D.read_pidfile(d)["pid"]


def test_off_is_unchanged_and_touches_no_daemon(ddir):
    w = S.Sam3Worker(python_bin="/nonexistent/python", daemon_mode="off")
    with pytest.raises(Sam3LiveUnavailable, match="interpreter not found"):
        w.start()
    assert w.mode == "subprocess"
    assert not D.pid_path(ddir).exists()


def test_env_mode_parsing(monkeypatch):
    for raw, want in [("", "off"), ("auto", "auto"), ("ON", "on"),
                      ("bogus", "off")]:
        monkeypatch.setenv(D.ENV_MODE, raw)
        assert S.daemon_mode_from_env() == want


def test_protocol_handshake_files_and_reuse(ddir):
    w = _worker().start()
    assert w.mode == "daemon" and w.daemon_spawned and w.spawn_s >= 0
    assert w.proc.poll() is None
    pid = _pid(ddir)
    info = D.read_pidfile(ddir)
    assert info["protocol"] == D.PROTOCOL and info["ready"] is True
    assert info["socket"] == str(D.sock_path(ddir))
    assert info["start_time"] == D.proc_start_time(pid)
    assert info["pgid"] == os.getpgid(pid) == pid  # own session
    assert (ddir.stat().st_mode & 0o777) == 0o700
    assert (D.sock_path(ddir).stat().st_mode & 0o777) == 0o600
    hello = w.daemon_hello
    exp = D.expected_identity()
    assert hello["protocol"] == exp["protocol"]
    assert hello["worker_sha256"] == exp["worker_sha256"]
    assert hello["daemon_sha256"] == exp["daemon_sha256"]
    for k in ("model_revision", "torch", "transformers", "python", "device",
              "gpu_peak_mem_mb"):
        assert k in hello
    r = w.query("img.png")
    assert r["ok"] and r["id"] == 1 and r["stub"] and r["pid"] == pid
    assert r["threshold"] == 0.5 and len(r["prompts"]) == 2
    w.close()
    assert D.verified_daemon_pid(D.read_pidfile(ddir))  # outlives the session
    w2 = _worker().start()
    assert not w2.daemon_spawned and _pid(ddir) == pid
    assert w2.query("x")["pid"] == pid
    assert w2.connect_s >= 0


def test_idle_exit_removes_socket_and_pidfile(ddir, monkeypatch):
    monkeypatch.setenv(D.ENV_IDLE, "1")
    _worker().start()
    pid = _pid(ddir)
    assert D._wait_gone(pid, 10)
    time.sleep(0.2)
    assert not D.sock_path(ddir).exists() and not D.pid_path(ddir).exists()


def test_handshake_mismatch_respawn(ddir, monkeypatch):
    # Start an "old code" daemon directly (not through the client).
    env = dict(os.environ)
    env[D.ENV_STUB + "_DAEMON_SHA"] = "old-code"
    log = open(ddir.parent / (ddir.name + ".log"), "ab")
    p = subprocess.Popen([sys.executable, str(D.DAEMON_SCRIPT), "--serve"],
                         stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    try:
        t0 = time.monotonic()
        while not D.sock_path(ddir).exists() and time.monotonic() - t0 < 20:
            time.sleep(0.05)
        old_pid = p.pid
        assert _pid(ddir) == old_pid
        w = _worker().start()
        assert w.daemon_spawned
        new_pid = _pid(ddir)
        assert new_pid != old_pid
        assert p.wait(timeout=20) == 0       # old one exited gracefully
        assert w.query("a")["pid"] == new_pid
        assert w.daemon_hello["daemon_sha256"] == \
            D.expected_identity()["daemon_sha256"]
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=10)
        log.close()
        try:
            (ddir.parent / (ddir.name + ".log")).unlink()
        except OSError:
            pass


def test_mismatch_during_query_triggers_replace(ddir, monkeypatch):
    w = _worker().start()
    old = _pid(ddir)
    # the client's idea of the expected code changes (code updated on disk)
    w._daemon_expected = dict(w._daemon_expected, worker_sha256="newer")
    monkeypatch.setattr(D, "expected_identity",
                        lambda *a, **k: dict(w._daemon_expected))
    # the replacement daemon reports the real sha, so it can never match the
    # patched expectation: both attempts fail -> Sam3LiveUnavailable, and the
    # first mismatch did trigger a replacement.
    with pytest.raises(Sam3LiveUnavailable):
        w.query("a")
    assert D.read_pidfile(ddir) is None or _pid(ddir) != old


def test_concurrent_clients_single_spawn(ddir, monkeypatch):
    monkeypatch.setenv(D.ENV_STUB + "_LOAD_S", "1.5")
    out, errs = [], []

    def go():
        try:
            w = _worker().start()
            out.append((w.daemon_spawned, w.query("c")["pid"]))
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)
    ts = [threading.Thread(target=go) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errs, errs
    assert len({pid for _, pid in out}) == 1
    assert sum(1 for sp, _ in out if sp) == 1
    assert D.log_path(ddir).read_text().count("starting (stub") == 1


def test_requests_are_serialized(ddir):
    w1, w2 = _worker().start(), _worker().start()
    t_end = {}

    def slow():
        w1.query("sleep:1.2")
        t_end["slow"] = time.monotonic()
    th = threading.Thread(target=slow)
    th.start()
    time.sleep(0.3)
    w2.query("quick")
    t_end["quick"] = time.monotonic()
    th.join()
    assert t_end["quick"] >= t_end["slow"] - 0.05


def test_id_mismatch_rejected_on_and_fallback_auto(ddir, monkeypatch):
    w = _worker("on").start()
    with pytest.raises(Sam3LiveUnavailable, match="id mismatch"):
        w.query("wrong_id")
    calls = []
    monkeypatch.setattr(S.Sam3Worker, "_start_subprocess",
                        lambda self: calls.append(1) or setattr(
                            self, "mode", "subprocess") or self)
    monkeypatch.setattr(S.Sam3Worker, "query",
                        lambda self, p, timeout_s=300: {"ok": True, "fb": 1}
                        if self.mode == "subprocess" else
                        S.Sam3Worker._query_via_daemon(self, p, timeout_s))
    wa = _worker("auto").start()
    assert wa.mode == "daemon"
    assert wa.query("wrong_id") == {"ok": True, "fb": 1}
    assert wa.mode == "subprocess"
    assert "id mismatch" in wa.daemon_fallback_reason and calls == [1]


def test_auto_falls_back_when_daemon_cannot_start(ddir, monkeypatch):
    calls = []
    monkeypatch.setattr(S.Sam3Worker, "_start_subprocess",
                        lambda self: calls.append(1) or setattr(
                            self, "mode", "subprocess") or self)
    w = S.Sam3Worker(python_bin="/nonexistent/python", daemon_mode="auto")
    w.start()
    assert w.mode == "subprocess" and calls == [1]
    assert "interpreter not found" in w.daemon_fallback_reason
    with pytest.raises(Sam3LiveUnavailable, match="interpreter not found"):
        S.Sam3Worker(python_bin="/nonexistent/python",
                     daemon_mode="on").start()


def test_gpu_error_replies_then_exits_then_respawn_once_then_raise(ddir):
    w = _worker("on").start()
    pid1 = _pid(ddir)
    with pytest.raises(Sam3LiveUnavailable, match="GPU error"):
        w.query("cuda_error")      # reply, daemon exits, respawn, same error
    D._wait_gone(pid1, 10)
    # both daemons exited after their GPU error; none left running
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and D.verified_daemon_pid(
            D.read_pidfile(ddir)):
        time.sleep(0.05)
    assert not D.verified_daemon_pid(D.read_pidfile(ddir))
    # the session recovers on the next ordinary query (respawn)
    assert w.query("ok")["ok"]


def test_stale_pidfile_and_socket_are_cleaned(ddir):
    ensured = D.ensure_dir(ddir)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    D.pid_path(ensured).write_text(json.dumps(
        {"pid": dead.pid, "start_time": 1, "ready": True}))
    D.sock_path(ensured).write_text("not a socket")
    w = _worker("on").start()
    assert w.daemon_spawned and w.query("s")["ok"]
    assert _pid(ddir) != dead.pid


def test_pid_reuse_guard_never_signals_foreign_process(ddir):
    victim = subprocess.Popen([sys.executable, "-c",
                               "import time; time.sleep(60)"])
    try:
        D.ensure_dir(ddir)
        # right pid, wrong start time -> not our daemon
        info = {"pid": victim.pid, "start_time": 12345, "ready": True}
        assert not D.verified_daemon_pid(info)
        # right pid + start time but cmdline is not sam3_daemon.py
        info["start_time"] = D.proc_start_time(victim.pid)
        assert not D.verified_daemon_pid(info)
        D.pid_path(ddir).write_text(json.dumps(info))
        assert D.stop_daemon(ddir, wait_s=1, allow_sigterm=True)
        assert victim.poll() is None          # untouched
    finally:
        victim.terminate()
        victim.wait(timeout=10)


def test_cli_status_and_stop(ddir):
    script = str(D.DAEMON_SCRIPT)
    r = subprocess.run([sys.executable, script, "--status", "--dir",
                        str(ddir)], capture_output=True, text=True)
    assert r.returncode == 1
    _worker().start()
    pid = _pid(ddir)
    r = subprocess.run([sys.executable, script, "--status", "--dir",
                        str(ddir)], capture_output=True, text=True)
    st = json.loads(r.stdout)
    assert r.returncode == 0 and st["running"] and st["handshake"]["pid"] == pid
    r = subprocess.run([sys.executable, script, "--stop", "--dir", str(ddir)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "stopped" in r.stdout
    assert D._wait_gone(pid, 10)
    assert not D.sock_path(ddir).exists()


def test_sigterm_exits_cleanly(ddir):
    _worker().start()
    pid = _pid(ddir)
    assert D.verified_daemon_pid(D.read_pidfile(ddir))
    os.kill(pid, signal.SIGTERM)       # recorded daemon pid only
    assert D._wait_gone(pid, 10)
    time.sleep(0.2)
    assert not D.sock_path(ddir).exists() and not D.pid_path(ddir).exists()


def test_daemon_and_worker_share_inference_code():
    src = D.DAEMON_SCRIPT.read_text()
    assert "sam3_worker._handle_one" in src and "sam3_worker.load_model" in src


# ---------------------------------------------------------------- round 2

def _no_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(S.Sam3Worker, "_start_subprocess",
                        lambda self: calls.append(
                            D.verified_daemon_pid(D.read_pidfile(ddir_of(self)))
                        ) or setattr(self, "mode", "subprocess") or self)
    return calls


def ddir_of(w):
    return w._daemon_dir


def test_auto_timeout_stops_daemon_before_fallback(ddir, monkeypatch):
    calls = _no_worker(monkeypatch)
    monkeypatch.setattr(S.Sam3Worker, "query",
                        lambda self, p, timeout_s=300: {"ok": True, "fb": 1}
                        if self.mode == "subprocess" else
                        S.Sam3Worker._query_via_daemon(self, p, timeout_s))
    w = _worker("auto").start()
    pid = _pid(ddir)
    assert w.query("sleep:2", timeout_s=1) == {"ok": True, "fb": 1}
    assert w.mode == "subprocess" and "timed out" in w.daemon_fallback_reason
    assert calls == [False]            # no daemon alive when worker started
    assert D._wait_gone(pid, 5)


def test_auto_daemon_not_stoppable_raises_no_second_worker(ddir, monkeypatch):
    calls = _no_worker(monkeypatch)
    monkeypatch.setattr(D, "stop_daemon", lambda *a, **k: False)
    w = _worker("auto").start()
    with pytest.raises(Sam3LiveUnavailable, match="would not stop"):
        w.query("sleep:2", timeout_s=1)
    assert calls == [] and w.mode == "daemon"
    # same guard on the start path (daemon alive but not ready in time)
    monkeypatch.setattr(S.Sam3Worker, "_start_daemon",
                        lambda self: (_ for _ in ()).throw(
                            Sam3LiveUnavailable("not ready")))
    with pytest.raises(Sam3LiveUnavailable, match="would not stop"):
        _worker("auto").start()
    assert calls == []


def test_model_id_mismatch_respawns(ddir, monkeypatch):
    env = dict(os.environ)
    env[D.ENV_MODEL_ID] = "old/model"
    log = open(ddir.parent / (ddir.name + ".mlog"), "ab")
    p = subprocess.Popen([sys.executable, str(D.DAEMON_SCRIPT), "--serve"],
                         stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    try:
        t0 = time.monotonic()
        while not D.sock_path(ddir).exists() and time.monotonic() - t0 < 20:
            time.sleep(0.05)
        assert _pid(ddir) == p.pid
        w = _worker().start()
        assert w.daemon_spawned and _pid(ddir) != p.pid
        assert p.wait(timeout=20) == 0
        assert w.daemon_hello["model_id"] == D.DEFAULT_MODEL_ID
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=10)
        log.close()
        try:
            (ddir.parent / (ddir.name + ".mlog")).unlink()
        except OSError:
            pass


def test_model_revision_mismatch_detected():
    hello = {"protocol": 1, "worker_sha256": "a", "daemon_sha256": "b",
             "model_id": "m", "model_revision": "r1", "model_snapshot": "c1"}
    exp = dict(hello)
    assert D.identity_mismatch(hello, exp) is None
    assert D.identity_mismatch(hello, dict(exp, model_revision="r2")) \
        == "model_revision differs"
    assert D.identity_mismatch(hello, dict(exp, model_revision=None)) is None
    assert D.identity_mismatch(dict(hello, model_revision=None), exp) is None
    assert D.identity_mismatch(hello, dict(exp, model_snapshot="c2")) \
        == "model_snapshot differs"
    assert D.identity_mismatch(hello, dict(exp, model_snapshot=None)) is None
    assert D.identity_mismatch(hello, dict(exp, model_id="x")) \
        == "model_id differs"


def test_local_model_snapshot_reads_refs_main(tmp_path, monkeypatch):
    monkeypatch.delenv(D.ENV_STUB, raising=False)
    ref = tmp_path / "models--a--b" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc123\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert D.local_model_snapshot("a/b") == "abc123"
    assert D.local_model_snapshot("a/none") is None


def test_default_model_id_matches_worker():
    import re
    src = (D.HERE / "sam3_worker.py").read_text()
    assert re.search(r'^MODEL_ID = "%s"' % re.escape(D.DEFAULT_MODEL_ID),
                     src, re.M)


def test_empty_cache_runs_on_error_paths(ddir, monkeypatch):
    log = ddir.parent / (ddir.name + ".cache")
    monkeypatch.setenv(D.ENV_STUB + "_CACHE_LOG", str(log))
    try:
        w = _worker("on").start()
        def n():
            time.sleep(0.5)    # release runs just after the reply is sent
            return len(log.read_text().split()) if log.exists() else 0
        base = n()                       # one release after model load
        assert w.query("ok")["ok"]
        assert n() == base + 1
        base = n()
        # handler exception: error reply, daemon survives, cache released
        r = w.query("raise")
        assert r["ok"] is False and "RuntimeError" in r["reason"]
        assert D.verified_daemon_pid(D.read_pidfile(ddir))
        assert n() == base + 1
        # GPU-error reply path
        with pytest.raises(Sam3LiveUnavailable):
            w.query("cuda_error")
        assert n() >= base + 2
    finally:
        try:
            log.unlink()
        except OSError:
            pass


def test_daemon_child_env_lacks_blas_pin(monkeypatch):
    monkeypatch.delenv(D.ENV_KEEP_BLAS, raising=False)
    env = {"OPENBLAS_NUM_THREADS": "1", D.ENV_BLAS_PINNED: "1", "X": "y"}
    assert D.child_env_without_blas(env) == {"X": "y"}
    env = {"OPENBLAS_NUM_THREADS": "4", "X": "y"}      # user's own value
    assert D.child_env_without_blas(env) == env
    env = {"OPENBLAS_NUM_THREADS": "1", D.ENV_BLAS_PINNED: "1",
           D.ENV_KEEP_BLAS: "1"}
    assert D.child_env_without_blas(env)["OPENBLAS_NUM_THREADS"] == "1"
    # the real spawn path: capture the env passed to Popen
    d = Path(tempfile.mkdtemp(prefix="s3e"))
    seen = {}
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv(D.ENV_BLAS_PINNED, "1")
    monkeypatch.setenv(D.ENV_STUB, "1")

    class Boom(Exception):
        pass

    def fake_popen(cmd, **kw):
        seen["env"] = kw["env"]
        raise OSError("no spawn in test")
    monkeypatch.setattr(D.subprocess, "Popen", fake_popen)
    with pytest.raises(D.DaemonError):
        D.ensure_daemon(sys.executable, d, ready_timeout_s=1)
    assert "OPENBLAS_NUM_THREADS" not in seen["env"]
    # per-run subprocess worker: same
    seen.clear()
    monkeypatch.setattr(S.subprocess, "Popen", fake_popen)
    with pytest.raises(Sam3LiveUnavailable):
        S.Sam3Worker(python_bin=sys.executable, daemon_mode="off").start()
    assert "OPENBLAS_NUM_THREADS" not in seen["env"]
    shutil.rmtree(d, ignore_errors=True)


_PROBE = r"""
import os, sys, runpy
seen = {}
class F:
    def find_spec(self, name, path=None, target=None):
        if name == "numpy" and "v" not in seen:
            seen["v"] = os.environ.get("OPENBLAS_NUM_THREADS")
        return None
sys.meta_path.insert(0, F())
sys.argv = [sys.argv[1], "--help"]
try:
    runpy.run_module("box_seam_detection.core.run", run_name="__main__")
except SystemExit:
    pass
print("RESULT", seen.get("v", "no-numpy"), os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("AUTOLABEL_V2_BLAS_PINNED"))
"""

# core entry points, run as modules (package-relative imports); the live
# detector / orchestrator entry points are covered by the live tests.
_PKG_SRC = Path(__file__).resolve().parents[2]
_ENTRY_SCRIPTS = [
    Path(__file__).resolve().parents[2] / "box_seam_detection" / "core" / "run.py",
]


@pytest.mark.parametrize("script", _ENTRY_SCRIPTS, ids=lambda p: p.name)
def test_blas_pin_set_before_numpy_import(script):
    if not script.is_file():
        pytest.skip(f"{script} not present")

    def run(extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENBLAS_NUM_THREADS", "AUTOLABEL_V2_BLAS_PINNED",
                            "AUTOLABEL_V2_BLAS_THREADS")}
        env.update(extra)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_PKG_SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        r = subprocess.run([sys.executable, "-c", _PROBE, str(script)],
                           capture_output=True, text=True, env=env)
        line = [ln for ln in r.stdout.splitlines()
                if ln.startswith("RESULT")]
        assert line, r.stdout + r.stderr
        return line[-1].split()[1:]
    at_import, final, marker = run({})
    assert at_import in ("1", "no-numpy") and final == "1" and marker == "1"
    # explicit value respected, and not marked as ours
    r = run({"OPENBLAS_NUM_THREADS": "4"})
    assert r[1] == "4" and r[2] == "None"
    # opt-out switch
    r = run({"AUTOLABEL_V2_BLAS_THREADS": "default"})
    assert r[1] == "None" and r[2] == "None"


# ------------------------------------------------------------- round 3

def _spawn_stub(ddir, extra_env=None, serve_dir=None):
    env = dict(os.environ)
    env.update(extra_env or {})
    env[D.ENV_DIR] = str(serve_dir or ddir)
    env[D.ENV_STUB] = "1"
    log = open(str(ddir) + ".x.log", "ab")
    return subprocess.Popen([sys.executable, str(D.DAEMON_SCRIPT), "--serve"],
                            stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def _wait_sock(d, t=20):
    t0 = time.monotonic()
    while not D.sock_path(d).exists() and time.monotonic() - t0 < t:
        time.sleep(0.05)


def test_singleton_second_daemon_exits_without_loading(ddir):
    p1 = _spawn_stub(ddir)
    other = Path(tempfile.mkdtemp(prefix="s3o"))
    try:
        _wait_sock(ddir)
        p2 = _spawn_stub(ddir, serve_dir=other,
                         extra_env={D.ENV_STUB + "_LOAD_S": "5"})
        assert p2.wait(timeout=15) == 0       # exits fast, no model load
        assert "another daemon holds" in (
            Path(str(ddir) + ".x.log").read_text())
        assert p1.poll() is None
        assert not D.name_is_free()
    finally:
        p1.terminate()
        p1.wait(timeout=10)
        shutil.rmtree(other, ignore_errors=True)
    assert D.wait_name_free(5)


def test_dir_removed_orphan_blocks_second_daemon(ddir):
    w = _worker("on").start()
    pid1 = _pid(ddir)
    shutil.rmtree(ddir)                       # dir recreated scenario
    try:
        # the orphan exits by itself (socket inode gone) within ~1-2 s ...
        assert D._wait_gone(pid1, 15)
        # ... and until it is gone nothing else may start: new ensure works
        info = D.ensure_daemon(sys.executable, ddir)
        assert info["spawned"] and _pid(ddir) != pid1
    finally:
        D.stop_daemon(ddir, wait_s=10, allow_sigterm=True)


def test_stop_daemon_false_when_orphan_in_other_dir(ddir):
    other = Path(tempfile.mkdtemp(prefix="s3o"))
    p = _spawn_stub(ddir, serve_dir=other)
    try:
        _wait_sock(other)
        # our dir has no daemon, but the machine-wide name is held
        assert D.stop_daemon(ddir, wait_s=1, allow_sigterm=True) is False
        w = _worker("auto")
        with pytest.raises(Sam3LiveUnavailable):
            w._ensure_no_daemon()
    finally:
        p.terminate()
        p.wait(timeout=10)
        shutil.rmtree(other, ignore_errors=True)
    assert D.stop_daemon(ddir, wait_s=1) is True


def test_sigterm_during_load_ends_daemon_immediately(ddir):
    p = _spawn_stub(ddir, extra_env={D.ENV_STUB + "_LOAD_S": "30"})
    t0 = time.monotonic()
    while not D.pid_path(ddir).exists() and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)             # our own child
    p.wait(timeout=5)
    assert time.monotonic() - t0 < 15
    assert p.poll() is not None


def test_ready_timeout_env(monkeypatch):
    monkeypatch.delenv(D.ENV_READY_S, raising=False)
    assert D.ready_timeout_from_env() == 180.0
    monkeypatch.setenv(D.ENV_READY_S, "42")
    assert D.ready_timeout_from_env() == 42.0
    monkeypatch.setenv(D.ENV_READY_S, "junk")
    assert D.ready_timeout_from_env() == 180.0


def test_python_executable_mismatch_respawns(ddir, tmp_path):
    w = _worker().start()
    pid1 = _pid(ddir)
    assert w.daemon_hello["python_executable"] == os.path.realpath(
        sys.executable)
    other = tmp_path / "otherpython"
    other.write_bytes(b"")                      # a different "interpreter"
    exp = D.expected_identity(other)
    assert D.identity_mismatch(w.daemon_hello, exp) == \
        "python_executable differs"
    assert D.identity_mismatch(w.daemon_hello,
                               D.expected_identity(sys.executable)) is None
    assert D.identity_mismatch(w.daemon_hello, D.expected_identity()) is None


def test_session_sends_absolute_image_path(ddir):
    w = _worker().start()
    r = w.query("rel/img.png")
    assert r["image"] == os.path.abspath("rel/img.png")


def test_handshake_retry_once_on_transient_error(ddir, monkeypatch):
    calls = []
    real = D._try_handshake

    def flaky(d, t):
        calls.append(1)
        if len(calls) == 1:
            raise D.DaemonError("transient EOF")
        return real(d, t)
    monkeypatch.setattr(D, "_try_handshake", flaky)
    w = _worker().start()                      # first handshake flakes
    assert w.mode == "daemon" and len(calls) >= 2
    # two failures in a row still surface as an error
    monkeypatch.setattr(D, "_try_handshake", lambda d, t: (_ for _ in ()
                        ).throw(D.DaemonError("always")))
    with pytest.raises(D.DaemonError):
        D.ensure_daemon(sys.executable, ddir)


def test_conftest_isolates_daemon_dir_and_lock():
    real = Path(f"/run/user/{os.getuid()}/autolabel_v2_sam3")
    assert D.daemon_dir() != real and D.daemon_dir().exists()
    assert D.lock_name().startswith("test-")


def test_s1_cast_failure_is_recorded(monkeypatch):
    import numpy as np
    from box_seam_detection.core import pipeline as P

    def boom(*a, **k):
        raise MemoryError("boom")
    monkeypatch.setattr(P.np, "asarray", boom)
    out = P._run_depth_work(None, lambda p: (object(), None),
                            {"pcd": "x"}, None, 1, 1,
                            {"depth_top": {}}, True, False)
    assert out["err"].startswith("depth_estimate_failed")
    assert out["est"] is None
    # unreadable cloud keeps its own reason
    def bad(p):
        raise OSError("nope")
    out = P._run_depth_work(None, bad, {"pcd": "x"}, None, 1, 1,
                            {"depth_top": {}}, True, False)
    assert out["err"].startswith("depth_pcd_unreadable")
