"""Detector server (``python -m box_seam_detection.live.detector --serve``) and its use by the live
``box_seam_detection.live.orchestrator`` (``run_detection`` / ``main``).

Detector: ``--serve`` loads the SAM3 worker once, then answers one JSON
request per stdin line with exactly the one-shot per-capture behaviour (same
files, same exit code), reusing the worker across requests.

Orchestrator: in LIVE mode the server is spawned BEFORE the first acquisition
(SAM3 loads while the camera captures), every attempt (retries included) goes
to that one server, the SAM3 import probe is skipped, a server startup_error
is a clean exit 2 before any downstream step, a server that cannot be spawned
/ dies / times out falls back to the one-shot subprocess, and the server
never outlives the run. A recorder timeout (exit 3: camera silent) is exit 2
without retry. The camera, SAM3 and the detector server are faked (the fake
server is a real subprocess speaking the JSON-lines protocol).
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("autolabel_v2_retry_helpers_srv",
                                               str(TESTS / "test_autolabel_v2_retry.py"))
retry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retry)
live = retry.live

DETECT = live.DETECT
ORCHESTRATOR = live.ORCHESTRATOR
W, H = live.W, live.H


@pytest.fixture()
def det():
    return live._load(DETECT)


@pytest.fixture()
def orch():
    return live._load(ORCHESTRATOR)


# ================================================================ detector side

class Sam3LiveUnavailable(RuntimeError):
    """Stand-in for autolabel_v2.sam3_live.Sam3LiveUnavailable (matched by name)."""


def _install_fake(monkeypatch, det, records, worker_fails=False, query_fails=()):
    """Fake autolabel_v2 import: ``records`` is a list consumed one per
    process_capture call; the queries whose 0-based index is in
    ``query_fails`` raise Sam3LiveUnavailable. Returns counters shared by all
    fake workers."""
    calls = {"starts": 0, "closes": 0, "queries": [], "sessions": [], "records": list(records)}

    class FakeWorker:
        def __init__(self, python_bin, stderr_log=None, **_):
            self.stderr_log = stderr_log

        def start(self):
            if worker_fails:
                raise RuntimeError("sam3 worker unavailable: weights absent")
            calls["starts"] += 1
            if self.stderr_log:
                Path(self.stderr_log).parent.mkdir(parents=True, exist_ok=True)
                Path(self.stderr_log).write_text(f"worker log {calls['starts']}\n")
            return self

        def query(self, *a, **k):
            calls["queries"].append((a, k))
            if len(calls["queries"]) - 1 in query_fails:
                raise Sam3LiveUnavailable("sam3 worker read timed out")
            return {"ok": True}

        def close(self):
            calls["closes"] += 1

    class FakePipeline:
        @staticmethod
        def process_capture(capture_id, paths, cfg, source, session):
            calls["sessions"].append(session)
            # The per-request wrapper is an instance attribute during the call.
            calls.setdefault("wrapped", []).append("query" in vars(session))
            session.query(paths["image"])
            return json.loads(json.dumps(calls["records"].pop(0))), {}

    monkeypatch.setattr(det, "import_autolabel",
                        lambda repo=None: (FakePipeline, lambda: {}, FakeWorker,
                                      lambda path, bgr, rec, debug: None))
    return calls


def _serve(det, cap, requests, tmp_path, extra=()):
    out = io.StringIO()
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    code = det.serve(["--serve", "--work-dir", str(tmp_path / "srv"),
                      "--sam3-python", str(cap["sam3_python"]), *extra],
                     stdin=stdin, protocol_out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _argv_in(cap, root, *extra):
    return ["--image", str(cap["image"]), "--pcd", str(cap["pcd"]),
            "--capture-manifest", str(cap["manifest"]),
            "--out-seam-json", str(root / "out" / "seam.json"),
            "--work-dir", str(root / "work"), "--overlay-dir", str(root / "ov"),
            "--sam3-python", str(cap["sam3_python"]), *extra]


def _strip_volatile(doc):
    doc = json.loads(json.dumps(doc))
    doc["autolabel_v2"].pop("timing_s")
    doc["autolabel_v2"].pop("record_path")
    return doc


def test_request_argv_maps_keys_and_passes_argv(det):
    assert det.request_argv({"id": 1, "argv": ["--image", "a"]}) == ["--image", "a"]
    argv = det.request_argv({"id": 2, "image": "i.png", "pcd": "p.pcd",
                             "capture_manifest": "m.json", "out_seam_json": "o.json",
                             "work_dir": "w", "overlay_dir": None, "rect_tol_mm": 4.0,
                             "allow_needs_review": True, "no_depth_edge": False})
    assert argv == ["--image", "i.png", "--pcd", "p.pcd", "--capture-manifest", "m.json",
                    "--out-seam-json", "o.json", "--work-dir", "w", "--rect-tol-mm", "4.0",
                    "--allow-needs-review"]
    for bad in ({"bogus": 1}, {"argv": "x"}, {"argv": ["a"], "image": "b"}, ["x"]):
        with pytest.raises(ValueError):
            det.request_argv(bad)


def test_serve_reuses_one_worker_and_matches_one_shot(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    good = retry._rec()
    skewed = retry._rec(retry._skewed_xyz(6.0))
    # One-shot reference runs (a fresh worker per run).
    one = _install_fake(monkeypatch, det, [good, skewed])
    ref_ok, ref_rej = tmp_path / "ref_ok", tmp_path / "ref_rej"
    assert det.main(_argv_in(cap, ref_ok)) == 0
    assert det.main(_argv_in(cap, ref_rej)) == 1
    assert one["starts"] == 2 and one["closes"] == 2
    # Server: same two captures, ONE worker.
    srv = _install_fake(monkeypatch, det, [good, skewed])
    s_ok, s_rej = tmp_path / "s_ok", tmp_path / "s_rej"
    code, lines = _serve(det, cap, [{"id": 1, "argv": _argv_in(cap, s_ok)},
                                    {"id": 2, "argv": _argv_in(cap, s_rej)}], tmp_path)
    assert code == 0
    assert lines[0]["type"] == "ready" and lines[0]["ok"] is True
    r1, r2 = lines[1:]
    assert (r1["type"], r1["id"], r1["exit_code"]) == ("response", 1, 0)
    assert (r2["type"], r2["id"], r2["exit_code"]) == ("response", 2, 1)
    assert srv["starts"] == 1 and srv["closes"] == 1           # closed at EOF only
    assert srv["sessions"][0] is srv["sessions"][1]
    assert srv["wrapped"] == [True, True]
    assert "query" not in vars(srv["sessions"][0])            # wrapper removed
    assert [q[1]["timeout_s"] for q in srv["queries"]] == [300, 300]
    # Identical outputs (paths/timing aside) and the same files.
    assert r1["out_seam_json"] == str(s_ok / "out" / "seam.json")
    a = json.loads((ref_ok / "out" / "seam.json").read_text())
    b = json.loads((s_ok / "out" / "seam.json").read_text())
    assert _strip_volatile(a) == _strip_volatile(b)
    assert b["autolabel_v2"]["timing_s"]["sam3_session"] == "server"
    assert (json.loads((ref_ok / "work" / "cap_1_autolabel_v2_record.json").read_text())
            == json.loads((s_ok / "work" / "cap_1_autolabel_v2_record.json").read_text()))
    assert r2["out_seam_json"] is None
    assert r2["reject_json"] == str(s_rej / "ov" / "cap_1_autolabel_v2_reject.json")
    assert (json.loads((ref_rej / "ov" / "cap_1_autolabel_v2_reject.json").read_text())
            == json.loads(Path(r2["reject_json"]).read_text()))
    assert not (s_rej / "out" / "seam.json").exists()
    for root in (ref_ok, s_ok):
        assert sorted(p.name for p in (root / "work").iterdir()) == [
            "cap_1_autolabel_v2_record.json", "cap_1_sam3_worker.stderr.log"]
    # Per-capture copy of the server's worker log.
    assert (s_ok / "work" / "cap_1_sam3_worker.stderr.log").read_text() == "worker log 1\n"


def test_serve_startup_failure_reports_and_requests_exit_2(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    calls = _install_fake(monkeypatch, det, [retry._rec()], worker_fails=True)
    # One-shot reference: SAM3 unavailable is exit 2.
    assert det.main(_argv_in(cap, tmp_path / "ref")) == 2
    code, lines = _serve(det, cap, [{"id": 7, "argv": _argv_in(cap, tmp_path / "s")}], tmp_path)
    assert code == 0
    assert lines[0]["type"] == "startup_error" and "SAM3 worker unavailable" in lines[0]["reason"]
    assert lines[1]["exit_code"] == 2 and lines[1]["id"] == 7
    assert calls["starts"] == 0 and not (tmp_path / "s" / "out" / "seam.json").exists()


def test_serve_bad_requests_answered_and_server_continues(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    _install_fake(monkeypatch, det, [retry._rec()])
    out = io.StringIO()
    stdin = io.StringIO("not json\n"
                        + json.dumps({"id": 1, "argv": ["--image", "x"]}) + "\n"  # argparse error
                        + json.dumps({"id": 2, "argv": _argv_in(cap, tmp_path / "s", "--rect-tol-mm", "0")}) + "\n"
                        + json.dumps({"id": 3, "argv": _argv_in(cap, tmp_path / "s")}) + "\n"
                        + json.dumps({"type": "shutdown"}) + "\n"
                        + json.dumps({"id": 4, "argv": _argv_in(cap, tmp_path / "s")}) + "\n")
    det.serve(["--serve", "--work-dir", str(tmp_path / "srv"),
               "--sam3-python", str(cap["sam3_python"])], stdin=stdin, protocol_out=out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [m.get("exit_code") for m in lines[1:]] == [2, 2, 2, 0]   # id 4 never runs
    assert [m.get("id") for m in lines[1:]] == [None, 1, 2, 3]


def test_serve_rejects_sam3_python_mismatch(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    _install_fake(monkeypatch, det, [retry._rec()])
    other = tmp_path / "other_python"
    other.write_text("")
    argv = _argv_in(cap, tmp_path / "s")
    argv[argv.index("--sam3-python") + 1] = str(other)
    _, lines = _serve(det, cap, [{"id": 1, "argv": argv}], tmp_path)
    assert lines[1]["exit_code"] == 2


def test_serve_restarts_a_dead_worker(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    calls = _install_fake(monkeypatch, det, [retry._rec(), retry._rec()])

    class _Dead:
        @staticmethod
        def poll():
            return -9

    real_start = det.start_sam3_worker

    def start(*a, **k):
        session = real_start(*a, **k)
        session.proc = _Dead() if calls["starts"] == 1 else None
        return session

    monkeypatch.setattr(det, "start_sam3_worker", start)
    _, lines = _serve(det, cap, [{"id": 1, "argv": _argv_in(cap, tmp_path / "a")},
                                 {"id": 2, "argv": _argv_in(cap, tmp_path / "b")}], tmp_path)
    # Worker 1 is found dead at request 1 -> fresh worker; worker 2 (proc None)
    # counts as dead too -> another fresh worker for request 2.
    assert [m["exit_code"] for m in lines[1:]] == [0, 0]
    assert calls["starts"] == 3


def test_serve_drops_session_after_sam3_unavailable(det, tmp_path, monkeypatch):
    # Query 0 raises Sam3LiveUnavailable -> that session is closed and the
    # next request starts a fresh worker (one-shot: a fresh worker per capture).
    cap = live._capture(tmp_path)
    calls = _install_fake(monkeypatch, det, [retry._rec(), retry._rec()], query_fails={0})
    _, lines = _serve(det, cap, [{"id": 1, "argv": _argv_in(cap, tmp_path / "a")},
                                 {"id": 2, "argv": _argv_in(cap, tmp_path / "b")}], tmp_path)
    assert lines[0]["type"] == "ready"
    assert lines[1]["exit_code"] == 2 and lines[2]["exit_code"] == 0
    assert calls["starts"] == 2                     # startup worker + fresh one
    assert calls["closes"] == 2                     # failed one dropped + final close
    assert calls["sessions"][0] is not calls["sessions"][1]


def test_serve_keeps_session_when_queries_succeed(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    calls = _install_fake(monkeypatch, det, [retry._rec(), retry._rec()])
    _, lines = _serve(det, cap, [{"id": 1, "argv": _argv_in(cap, tmp_path / "a")},
                                 {"id": 2, "argv": _argv_in(cap, tmp_path / "b")}], tmp_path)
    assert [m["exit_code"] for m in lines[1:]] == [0, 0]
    assert calls["starts"] == 1 and calls["closes"] == 1


def test_serve_broken_pipe_on_startup_emit_still_closes_sam3(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    calls = _install_fake(monkeypatch, det, [retry._rec()])

    class _Gone(io.StringIO):
        def write(self, text):
            raise BrokenPipeError("client gone")

    with pytest.raises(BrokenPipeError):
        det.serve(["--serve", "--work-dir", str(tmp_path / "srv"),
                   "--sam3-python", str(cap["sam3_python"])],
                  stdin=io.StringIO(""), protocol_out=_Gone())
    assert calls["starts"] == 1 and calls["closes"] == 1


FAKE_AUTOLABEL_PKG = {
    "__init__.py": "",
    "config.py": "def default_config():\n    return {}\n",
    "overlay.py": "def write_overlay(path, bgr, record, debug):\n    pass\n",
    "sam3_session.py": (
        "import os, sys\n"
        "class Sam3Worker:\n"
        "    def __init__(self, python_bin, stderr_log=None, **_):\n"
        "        self.stderr_log = stderr_log\n"
        "    def start(self):\n"
        "        if os.environ.get('FAKE_SAM3_FAIL'):\n"
        "            raise RuntimeError('no weights')\n"
        "        open(self.stderr_log, 'a').write('started %d\\n' % os.getpid())\n"
        "        os.write(1, b'C-level noise on fd 1\\n')\n"
        "        return self\n"
        "    def query(self, image, timeout_s=300):\n"
        "        return {'ok': True}\n"
        "    def close(self):\n"
        "        open(self.stderr_log, 'a').write('closed\\n')\n"),
    "pipeline.py": (
        "import json, pathlib\n"
        "RECORD = json.loads((pathlib.Path(__file__).parent / 'record.json').read_text())\n"
        "def process_capture(capture_id, paths, cfg, source, session):\n"
        "    print('pipeline chatter on stdout')\n"
        "    session.query(paths['image'])\n"
        "    return json.loads(json.dumps(RECORD)), {}\n"),
}

# The real detector CLI run with the library faked: the wrapper swaps ``import_autolabel`` (the package's
# single import seam) for the fake pipeline above, then runs the detector's own main().
WRAPPER = (
    "import sys\n"
    "sys.path.insert(0, {fake_dir!r})\n"
    "from fake_av2 import config, overlay, pipeline, sam3_session\n"
    "import box_seam_detection.live.detector as d\n"
    "d.import_autolabel = lambda repo=None: (pipeline, config.default_config,\n"
    "                                        sam3_session.Sam3Worker, overlay.write_overlay)\n"
    "raise SystemExit(d.main())\n")


def _fake_repo(tmp_path):
    """(fake package dir, wrapper script) -- the fake library lives in <tmp>/fake/fake_av2."""
    pkg = tmp_path / "fake" / "fake_av2"
    pkg.mkdir(parents=True)
    for name, text in FAKE_AUTOLABEL_PKG.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (pkg / "record.json").write_text(json.dumps(retry._rec()), encoding="utf-8")
    wrapper = tmp_path / "fake" / "run_detector.py"
    wrapper.write_text(WRAPPER.format(fake_dir=str(tmp_path / "fake")), encoding="utf-8")
    return wrapper


def test_real_server_subprocess_protocol_is_clean(tmp_path):
    """Real ``--serve`` subprocess: stray prints / fd-1 writes go to stderr,
    stdout carries only protocol lines; EOF closes the worker; the result
    matches the one-shot CLI subprocess."""
    pytest.importorskip("cv2")
    cap = live._capture(tmp_path)
    wrapper = _fake_repo(tmp_path)
    env = live._pkg_env()
    one = subprocess.run([sys.executable, str(wrapper), *_argv_in(cap, tmp_path / "one")],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120,
                         env=env)
    assert one.returncode == 0, one.stderr
    proc = subprocess.Popen([sys.executable, str(wrapper), "--serve", "--work-dir",
                             str(tmp_path / "srv"), "--sam3-python", str(cap["sam3_python"])],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    requests = [{"id": i, "argv": _argv_in(cap, tmp_path / f"s{i}")} for i in (1, 2)]
    out, err = proc.communicate("".join(json.dumps(r) + "\n" for r in requests), timeout=120)
    assert proc.returncode == 0, err
    lines = [json.loads(line) for line in out.splitlines()]  # every stdout line is JSON
    assert [m["type"] for m in lines] == ["ready", "response", "response"]
    assert [m["exit_code"] for m in lines[1:]] == [0, 0]
    assert "pipeline chatter on stdout" in err and "C-level noise on fd 1" in err
    assert "accepted cap_1" in err
    log = (tmp_path / "srv" / "sam3_worker.stderr.log").read_text()
    assert log.count("started") == 1 and log.endswith("closed\n")
    a = json.loads((tmp_path / "one" / "out" / "seam.json").read_text())
    for i in (1, 2):
        b = json.loads((tmp_path / f"s{i}" / "out" / "seam.json").read_text())
        assert _strip_volatile(a) == _strip_volatile(b)


def test_real_server_subprocess_startup_error(tmp_path):
    cap = live._capture(tmp_path)
    wrapper = _fake_repo(tmp_path)
    proc = subprocess.run([sys.executable, str(wrapper), "--serve", "--work-dir",
                           str(tmp_path / "srv"), "--sam3-python", str(cap["sam3_python"])],
                          input="", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          timeout=60, env={**live._pkg_env(), "FAKE_SAM3_FAIL": "1"})
    status = json.loads(proc.stdout.splitlines()[0])
    assert status["type"] == "startup_error" and "no weights" in status["reason"]


# ================================================================ orchestrator side

FAKE_SERVER = r'''
import json, os, sys, time
from pathlib import Path
CFG = json.loads(Path(__file__).with_name("fake_server_cfg.json").read_text())

def event(name, **extra):
    with open(CFG["events"], "a") as fh:
        fh.write(json.dumps({"t": time.time(), "event": name, "pid": os.getpid(), **extra}) + "\n")

if "--serve" not in sys.argv:
    raise SystemExit("fake server: one-shot mode is faked via run_step")
if CFG.get("ignore_sigterm"):
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
event("start", argv=sys.argv[1:])
time.sleep(CFG.get("ready_delay", 0.0))
if CFG.get("die_before_ready"):
    event("died")
    os._exit(1)
if CFG.get("startup_error"):
    print(json.dumps({"type": "startup_error", "ok": False, "reason": CFG["startup_error"]}), flush=True)
else:
    print(json.dumps({"type": "ready", "ok": True}), flush=True)
event("ready")
outcomes = list(CFG.get("outcomes", []))
for line in sys.stdin:
    req = json.loads(line)
    argv = req["argv"]
    flag = lambda name: argv[argv.index(name) + 1]
    manifest = json.loads(Path(flag("--capture-manifest")).read_text())
    cid = manifest["capture_id"]
    outcome = outcomes.pop(0)
    event("request", capture_id=cid, outcome=outcome, argv=argv)
    if outcome == "crash":
        os._exit(1)
    if outcome == "hang":
        time.sleep(3600)
    out, ov = Path(flag("--out-seam-json")), Path(flag("--overlay-dir"))
    code = outcome if isinstance(outcome, int) else None
    if outcome == "ok":
        doc = dict(CFG["template"], source_image=flag("--image"), capture_id=cid)
        out.write_text(json.dumps(doc))
        code = 0
    elif outcome in ("retry", "reject"):
        ov.mkdir(parents=True, exist_ok=True)
        retryable = outcome == "retry"
        (ov / f"{cid}_autolabel_v2_reject.json").write_text(json.dumps({
            "status": "rejected", "capture_id": cid, "retryable": retryable,
            "reasons": ["rectangle_inconsistent(A=6.00mm,B=0.10mm)"] if retryable
            else ["wrong_face_suspect"]}))
        code = 1
    print(json.dumps({"type": "response", "id": req["id"], "exit_code": code,
                      "elapsed_s": 0.0, "timing": {"sam3_load": 0.0}}), flush=True)
if CFG.get("ignore_eof"):
    event("ignoring_eof")
    time.sleep(3600)
event("exit")
'''


class _Harness:
    def __init__(self, tmp_path, cfg):
        self.tmp_path = tmp_path
        self.script = tmp_path / "autolabel_v2_detect.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.events_path = tmp_path / "events.jsonl"
        cfg = {"events": str(self.events_path), **cfg}
        (tmp_path / "fake_server_cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    def events(self):
        if not self.events_path.is_file():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (exited, not yet reaped by us) is not running.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return False


def _run(orch, det, tmp_path, monkeypatch, cfg, oneshot=(), *extra_argv, offline=False,
         recorder_delay=0.0, recorder_exit=None, average="1", popen=None, stop_at="query"):
    """Drive the orchestrator's main() with a fake recorder, the fake server
    subprocess (cfg) and fake one-shot detector steps (``oneshot`` outcomes)."""
    harness = _Harness(tmp_path, cfg)
    root = tmp_path / "root"
    captures = [retry._make_capture(root, f"cap_{i}") for i in range(1, 8)]
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")
    template = retry._good_doc(det, captures[0]["image"], "cap_1")
    cfg_path = tmp_path / "fake_server_cfg.json"
    cfg_full = json.loads(cfg_path.read_text())
    cfg_full["template"] = template
    cfg_path.write_text(json.dumps(cfg_full))
    state = {"probes": [], "recorder": [], "oneshot": [], "titles": [], "query": None}
    queue = list(oneshot)

    class _Probe:
        returncode = 0
        stdout = ""

    def fake_subprocess_run(cmd, *a, **k):
        state["probes"].append(cmd)
        return _Probe()

    monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)
    state["spawned"] = []
    real_popen = popen or subprocess.Popen

    def spawn(cmd, *a, **k):
        state["spawned"].append({"t": time.time(), "cmd": cmd})
        return real_popen(cmd, *a, **k)

    monkeypatch.setattr(orch.subprocess, "Popen", spawn)

    def fake_run_step(title, cmd, capture=False, check=True):
        state["titles"].append(title)
        if "rgbd_recorder" in title:
            t0 = time.time()
            if recorder_exit is not None:
                print(f"[{title}] 失敗（exit code {recorder_exit}），中止", file=sys.stderr)
                raise orch.DetectionAbort(recorder_exit, title)
            time.sleep(recorder_delay)
            cap = captures[len(state["recorder"])]
            state["recorder"].append({"id": cap["id"], "t0": t0, "t1": time.time()})
            return retry._recorder_stdout(cap)
        if "autolabel_v2" in title:
            state["oneshot"].append(cmd)
            flag = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
            cid = json.loads(Path(flag("--capture-manifest")).read_text())["capture_id"]
            outcome = queue.pop(0)
            if outcome == "ok":
                Path(flag("--out-seam-json")).write_text(
                    json.dumps(dict(template, source_image=flag("--image"), capture_id=cid)))
                return 0
            if outcome in ("retry", "reject"):
                ov = Path(flag("--overlay-dir"))
                ov.mkdir(parents=True, exist_ok=True)
                (ov / f"{cid}_autolabel_v2_reject.json").write_text(json.dumps({
                    "status": "rejected", "capture_id": cid, "retryable": outcome == "retry",
                    "reasons": [f"fake_{outcome}"]}))
                return 1
            return outcome
        if "query_depth" in title:
            state["query"] = cmd
            if stop_at == "exit":      # a clean DetectionAbort (graceful shutdown path)
                live._halt(orch, title)
            raise live._Stop(title)    # an unexpected exception (abort path)
        raise AssertionError(f"unexpected step {title}")

    monkeypatch.setattr(orch, "run_step", fake_run_step)
    argv = ["--downstream-inprocess", "off", "--autolabel-v2",
            "--autolabel-v2-script", str(harness.script),
            "--autolabel-v2-sam3-python", str(sam3),
            "--calib-file", str(live._calib(tmp_path)),
            "--output-dir", str(root), "--autolabel-v2-average", str(average), *extra_argv]
    if offline:
        c = captures[0]
        argv += ["--from-image", str(c["image"]), "--pcd", str(c["pcd"]),
                 "--capture-manifest", str(c["manifest"])]
    code = None
    try:
        code = orch.main(argv)
    except KeyboardInterrupt:
        code = "interrupted"
    if code != "interrupted" and state["query"] is not None:
        code = "stopped_at_query"       # the fake query step halts the run (DetectionAbort / _Stop)
    state["events"] = harness.events()
    state["server_left"] = orch._AUTOLABEL_V2_SERVER
    return code, state


def _server_pids(state):
    return sorted({e["pid"] for e in state["events"] if e["event"] == "start"})


def _sam3_probe_ran(state):
    return any("Sam3Model" in " ".join(map(str, cmd)) for cmd in state["probes"])


def _assert_no_server_left(state):
    assert state["server_left"] is None
    for pid in _server_pids(state):
        assert not _pid_alive(pid), f"detector server {pid} still running"


def test_live_server_spawned_before_capture_and_loads_during_it(orch, det, tmp_path, monkeypatch):
    # Server ready (0.2 s) while the capture (0.6 s) is still running.
    code, state = _run(orch, det, tmp_path, monkeypatch, {"ready_delay": 0.2, "outcomes": ["ok"]},
                       recorder_delay=0.6)
    assert code == "stopped_at_query"
    start = next(e for e in state["events"] if e["event"] == "start")
    ready = next(e for e in state["events"] if e["event"] == "ready")
    rec = state["recorder"][0]
    assert len(state["spawned"]) == 1 and "--serve" in state["spawned"][0]["cmd"]
    assert state["spawned"][0]["t"] < rec["t0"]    # spawned before acquisition
    assert ready["t"] < rec["t1"]                  # SAM3 loaded during the capture
    assert start["argv"][:3] == ["--serve", "--work-dir", start["argv"][2]]
    assert "--sam3-python" in start["argv"]
    assert not _sam3_probe_ran(state)              # probe skipped in server mode
    assert len(state["probes"]) == 0               # (the library import check runs in-process now)
    assert state["oneshot"] == []                  # detection went to the server
    req = next(e for e in state["events"] if e["event"] == "request")
    assert req["capture_id"] == "cap_1" and "--out-seam-json" in req["argv"]
    assert "--seam-json" in state["query"]
    _assert_no_server_left(state)


def test_live_server_ready_after_capture_is_waited_for(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"ready_delay": 0.8, "outcomes": ["ok"]},
                       recorder_delay=0.0)
    assert code == "stopped_at_query"
    ready = next(e for e in state["events"] if e["event"] == "ready")
    assert ready["t"] > state["recorder"][0]["t1"]
    assert state["oneshot"] == []
    _assert_no_server_left(state)


def test_live_retries_reuse_the_one_server(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch,
                       {"outcomes": ["retry", "retry", "ok"]})
    assert code == "stopped_at_query"
    assert [r["id"] for r in state["recorder"]] == ["cap_1", "cap_2", "cap_3"]
    assert len(_server_pids(state)) == 1
    reqs = [e for e in state["events"] if e["event"] == "request"]
    assert [r["capture_id"] for r in reqs] == ["cap_1", "cap_2", "cap_3"]
    assert state["oneshot"] == []
    assert "/cap_3/" in state["query"][state["query"].index("--seam-json") + 1]
    _assert_no_server_left(state)


def test_live_all_attempts_rejected_exit_1_and_server_stopped(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["retry", "retry"]},
                       (), "--autolabel-v2-attempts", "2")
    assert code == 1 and state["query"] is None
    _assert_no_server_left(state)


@pytest.mark.parametrize("outcome,expected", [("reject", 1), (2, 2)])
def test_live_server_non_retryable_outcomes(orch, det, tmp_path, monkeypatch, outcome, expected):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": [outcome, "ok"]})
    assert code == expected and state["query"] is None
    assert len(state["recorder"]) == 1
    _assert_no_server_left(state)


def test_live_server_crash_falls_back_to_one_shot(orch, det, tmp_path, monkeypatch, capsys):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["retry", "crash"]},
                       oneshot=["retry", "ok"])
    assert code == "stopped_at_query"
    # Attempt 1 on the server; attempt 2 crashed it -> re-run one-shot on the
    # SAME capture; attempt 3 stays one-shot (no new server).
    reqs = [e["capture_id"] for e in state["events"] if e["event"] == "request"]
    assert reqs == ["cap_1", "cap_2"]
    flag = lambda cmd, name: cmd[cmd.index(name) + 1]  # noqa: E731
    assert [Path(flag(c, "--image")).stem for c in state["oneshot"]] == ["cap_2", "cap_3"]
    assert [r["id"] for r in state["recorder"]] == ["cap_1", "cap_2", "cap_3"]
    assert len(_server_pids(state)) == 1
    assert "改用單次子行程" in capsys.readouterr().err
    _assert_no_server_left(state)


def test_live_server_dies_before_ready_falls_back(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"die_before_ready": True},
                       oneshot=["ok"])
    assert code == "stopped_at_query"
    assert len(state["oneshot"]) == 1
    _assert_no_server_left(state)


def test_live_server_timeout_kills_and_falls_back(orch, det, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_REQUEST_TIMEOUT_S", 1.0)
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["hang"]}, oneshot=["ok"])
    assert code == "stopped_at_query"
    assert len(state["oneshot"]) == 1
    _assert_no_server_left(state)


def test_live_server_startup_error_is_exit_2_before_downstream(orch, det, tmp_path, monkeypatch,
                                                               capsys):
    code, state = _run(orch, det, tmp_path, monkeypatch,
                       {"startup_error": "SAM3 worker unavailable: no weights"},
                       oneshot=["ok"])
    assert code == 2
    assert state["oneshot"] == [] and state["query"] is None
    err = capsys.readouterr().err
    assert "server 啟動失敗" in err and "no weights" in err
    _assert_no_server_left(state)


def test_live_server_spawn_failure_probes_sam3_and_uses_one_shot(orch, det, tmp_path,
                                                                 monkeypatch):
    def broken_popen(*a, **k):
        raise OSError("no fork for you")

    code, state = _run(orch, det, tmp_path, monkeypatch, {}, oneshot=["ok"], popen=broken_popen)
    assert code == "stopped_at_query"
    assert _sam3_probe_ran(state)                  # the skipped probe runs after all
    assert len(state["oneshot"]) == 1 and state["events"] == []


def test_live_server_off_uses_one_shot_and_probe(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {}, ["ok"], "--autolabel-v2-server", "off")
    assert code == "stopped_at_query"
    assert _sam3_probe_ran(state) and len(state["oneshot"]) == 1 and state["events"] == []


def test_offline_default_stays_one_shot(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {}, ["ok"], offline=True)
    assert code == "stopped_at_query"
    assert _sam3_probe_ran(state) and len(state["oneshot"]) == 1 and state["events"] == []
    assert state["recorder"] == []


def test_offline_server_on_uses_server(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]},
                       (), "--autolabel-v2-server", "on", offline=True)
    assert code == "stopped_at_query"
    assert not _sam3_probe_ran(state) and state["oneshot"] == []
    _assert_no_server_left(state)


def test_server_stopped_when_later_preflight_fails(orch, det, tmp_path, monkeypatch):
    # The calibration preflight runs AFTER the server spawn: parser.error (exit 2).
    code, state = _run(orch, det, tmp_path, monkeypatch, {"ready_delay": 0.3, "outcomes": []},
                       (), "--calib-file", str(tmp_path / "missing.calib"))
    assert code == 2 and state["recorder"] == []
    assert len(_server_pids(state)) == 1
    _assert_no_server_left(state)


def test_server_stopped_on_keyboard_interrupt(orch, det, tmp_path, monkeypatch):
    def interrupted(*a, **k):
        time.sleep(0.2)  # let the server start
        raise KeyboardInterrupt

    monkeypatch.setattr(orch, "acquire_live_capture", interrupted)
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]})
    assert code == "interrupted"
    assert len(_server_pids(state)) == 1
    _assert_no_server_left(state)


def test_server_ignoring_eof_is_killed(orch, det, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S", 0.5)
    code, state = _run(orch, det, tmp_path, monkeypatch,
                       {"outcomes": ["ok"], "ignore_eof": True}, stop_at="exit")
    assert code == "stopped_at_query"
    assert any(e["event"] == "ignoring_eof" for e in state["events"])
    _assert_no_server_left(state)


@pytest.mark.parametrize("average", ["1", "3"])
def test_recorder_timeout_exit_3_is_exit_2_without_retry(orch, det, tmp_path, monkeypatch,
                                                         capsys, average):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]},
                       recorder_exit=3, average=average)
    assert code == 2
    titles = [t for t in state["titles"] if "rgbd_recorder" in t]
    assert len(titles) == 1                         # no retry
    assert state["query"] is None
    assert "exit 3" in capsys.readouterr().err
    _assert_no_server_left(state)


def test_recorder_other_failures_keep_their_exit_code(orch, det, tmp_path, monkeypatch):
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]}, recorder_exit=1)
    assert code == 1
    _assert_no_server_left(state)


def test_normal_exit_closes_server_gracefully(orch, det, tmp_path, monkeypatch):
    # clean-abort path: stdin EOF -> the server finishes its own shutdown.
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]}, stop_at="exit")
    assert code == "stopped_at_query"
    assert any(e["event"] == "exit" for e in state["events"])
    _assert_no_server_left(state)


def _interrupt_in_flight(orch, monkeypatch, tmp_path):
    """Ctrl-C while the server works on request 1 (after it logged it)."""
    events = tmp_path / "events.jsonl"
    original = orch.AutolabelV2Server._next_message

    def interrupting(self, deadline):
        if self._next_id >= 1:
            t_end = time.time() + 20
            while time.time() < t_end and (not events.is_file()
                                           or '"request"' not in events.read_text()):
                time.sleep(0.02)
            raise KeyboardInterrupt
        return original(self, deadline)

    monkeypatch.setattr(orch.AutolabelV2Server, "_next_message", interrupting)


def test_keyboard_interrupt_during_request_kills_server_at_once(orch, det, tmp_path,
                                                                monkeypatch):
    # The graceful close would wait up to the shutdown timeout for the hung
    # request; the abort path must kill the process group immediately.
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S", 60.0)
    _interrupt_in_flight(orch, monkeypatch, tmp_path)
    t0 = time.monotonic()
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["hang"]})
    elapsed = time.monotonic() - t0
    assert code == "interrupted"
    assert any(e["event"] == "request" for e in state["events"])
    assert not any(e["event"] == "exit" for e in state["events"])
    assert elapsed < 20, elapsed
    _assert_no_server_left(state)


def test_keyboard_interrupt_sigkills_server_ignoring_sigterm(orch, det, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S", 60.0)
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_ABORT_GRACE_S", 0.3)
    _interrupt_in_flight(orch, monkeypatch, tmp_path)
    t0 = time.monotonic()
    code, state = _run(orch, det, tmp_path, monkeypatch,
                       {"outcomes": ["hang"], "ignore_sigterm": True})
    assert code == "interrupted" and time.monotonic() - t0 < 20
    _assert_no_server_left(state)


def test_unexpected_exception_kills_server_without_graceful_wait(orch, det, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(orch, "AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S", 60.0)
    t0 = time.monotonic()
    code, state = _run(orch, det, tmp_path, monkeypatch,
                       {"outcomes": ["ok"], "ignore_eof": True})
    # live._Stop is an unexpected Exception -> abort path: the server ignoring stdin EOF
    # is killed at once, not after the 60 s graceful wait.
    assert code == "stopped_at_query"
    assert time.monotonic() - t0 < 20
    _assert_no_server_left(state)


def test_terminate_now_ignores_a_second_ctrl_c(orch, tmp_path):
    import signal

    harness = _Harness(tmp_path, {"ignore_sigterm": True, "ignore_eof": True})
    server = orch.AutolabelV2Server([sys.executable, str(harness.script), "--serve"])
    server.wait_ready(timeout=30)
    original_kill = server.kill
    seen = {}

    def kill_with_second_ctrl_c():
        seen["handler"] = signal.getsignal(signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)     # discarded: SIGINT is ignored here
        original_kill()

    server.kill = kill_with_second_ctrl_c
    before = signal.getsignal(signal.SIGINT)
    server.terminate_now(grace=0.2)
    assert seen["handler"] is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) is before
    assert server.proc.poll() is not None and not _pid_alive(server.proc.pid)


def test_close_interrupted_by_ctrl_c_kills_server(orch, tmp_path):
    harness = _Harness(tmp_path, {"ignore_eof": True})
    server = orch.AutolabelV2Server([sys.executable, str(harness.script), "--serve"])
    server.wait_ready(timeout=30)
    real_wait = server.proc.wait
    calls = []

    def wait(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:                      # the graceful wait: Ctrl-C
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    server.proc.wait = wait
    with pytest.raises(KeyboardInterrupt):
        server.close(timeout=60)
    assert not _pid_alive(server.proc.pid)


def test_server_output_dir_mkdir_failure_is_exit_2(orch, det, tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    code, state = _run(orch, det, tmp_path, monkeypatch, {"outcomes": ["ok"]},
                       (), "--output-dir", str(blocker / "sub"))
    assert code == 2
    assert state["spawned"] == [] and state["recorder"] == []
    assert "無法建立輸出資料夾" in capsys.readouterr().err
