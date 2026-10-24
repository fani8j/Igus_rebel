"""S3: downstream steps (live.query_depth, live.base_link, live.side_seed) run in-process by
live.orchestrator must give the same output files and exit codes as the subprocess path
(--downstream-inprocess off)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_autolabel_v2_live as live  # noqa: E402

ROOT = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures")
ENV = {**live._pkg_env(), "ROS_DOMAIN_ID": "87", "ROS_LOCALHOST_ONLY": "1"}


@pytest.fixture()
def orch():
    return live._load(live.ORCHESTRATOR)


def _real_capture():
    for seam in sorted((ROOT / "pipeline").glob("*/*/*_autolabel_v2_seam.json")):
        cap = seam.name[:-len("_autolabel_v2_seam.json")]
        pcd, man = ROOT / "pointcloud" / f"{cap}.pcd", ROOT / "captures" / f"{cap}.json"
        if pcd.is_file() and man.is_file():
            return cap, pcd, man, seam
    return None


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in ("manifest_path", "source_file")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _diff(a, b, path=""):
    if type(a) is not type(b):
        return [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float):  # 1-ulp BLAS/SIMD differences depending on what else is loaded
        return [] if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _json(path):
    return _strip(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- exit codes
def _fake_module(monkeypatch, behaviour):
    mod = type(sys)("s3_fake_step")
    mod.main = behaviour
    monkeypatch.setitem(sys.modules, "box_seam_detection.live.s3_fake_step", mod)


@pytest.mark.parametrize("behaviour,code", [
    (lambda argv: None, 0),
    (lambda argv: 1, 1),
    (lambda argv: 2, 2),
    (lambda argv: (_ for _ in ()).throw(SystemExit(3)), 3),
    (lambda argv: (_ for _ in ()).throw(SystemExit()), 0),
    (lambda argv: (_ for _ in ()).throw(SystemExit("boom")), 1),
    (lambda argv: (_ for _ in ()).throw(RuntimeError("x")), 1),
])
def test_exit_code_mapping(orch, monkeypatch, behaviour, code):
    _fake_module(monkeypatch, behaviour)
    assert orch._run_script_main_inprocess("s3_fake_step", ["--a"]) == code


def test_import_failure_is_exit_1(orch, capsys):
    assert orch._run_script_main_inprocess("s3_no_such_module", []) == 1
    assert "ModuleNotFoundError" in capsys.readouterr().err


def test_run_script_step_off_uses_subprocess_and_tf_stays_subprocess(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(orch, "run_step", lambda *a, **k: calls.append(("sub", a[0])))
    monkeypatch.setattr(orch, "_run_script_main_inprocess",
                        lambda *a, **k: calls.append(("in", a[0])) or 0)
    orch.run_script_step(SimpleNamespace(downstream_inprocess=False), "t", "x", ["--a"])
    orch.run_script_step(SimpleNamespace(downstream_inprocess=True), "t", "x", ["--a"], inprocess_ok=False)
    orch.run_script_step(SimpleNamespace(downstream_inprocess=True), "t", "x", ["--a"])
    assert [c[0] for c in calls] == ["sub", "sub", "in"]


def test_inprocess_failure_aborts_with_step_code(orch, monkeypatch):
    monkeypatch.setattr(orch, "_run_script_main_inprocess", lambda *a: 2)
    with pytest.raises(orch.DetectionAbort) as info:
        orch.run_script_step(SimpleNamespace(downstream_inprocess=True), "t", "x", ["--a"])
    assert info.value.exit_code == 2
    assert orch.run_script_step(SimpleNamespace(downstream_inprocess=True), "t", "x",
                                ["--a"], check=False) == 2


def test_default_is_on():
    text = Path(live._load(live.ORCHESTRATOR).__file__).read_text(encoding="utf-8")
    assert '"--downstream-inprocess", choices=("on", "off"), default="on"' in text
    assert live._load(live.ORCHESTRATOR).DetectConfig().downstream_inprocess is True


# ------------------------------------------------- real outputs, both paths
def _sub(script, *argv):
    return subprocess.run([sys.executable, "-m", f"box_seam_detection.live.{script}", *argv], env=ENV,
                          capture_output=True, text=True)


@pytest.mark.skipif(_real_capture() is None, reason="no stored capture")
def test_outputs_identical_to_subprocess(orch, tmp_path):
    cap, pcd, man, seam = _real_capture()
    for tag in ("sub", "in"):
        (tmp_path / tag).mkdir()
    q = lambda d: ["--pcd", str(pcd), "--capture-manifest", str(man),
                   "--output-json", str(d / "camera.json"), "--seam-json", str(seam)]
    c = lambda d: ["--json", str(d / "camera.json"), "--target-frame", "base_link",
                   "--timeout", "5.0", "--output-json", str(d / "base.json"),
                   "--planner-output-json", str(d / "planner.json"),
                   "--driver-extrinsic", "recorded"]
    s = lambda d: ["--planner-json", str(d / "planner.json")]
    sub, inp = tmp_path / "sub", tmp_path / "in"
    for script, make in (("query_depth", q), ("base_link", c), ("side_seed", s)):
        assert _sub(script, *make(sub)).returncode == 0
        assert orch._run_script_main_inprocess(script, make(inp)) == 0
    for name in ("camera.json", "base.json", "planner.json"):
        a, b = _json(sub / name), _json(inp / name)
        # camera.json/base.json "source_file" differ by directory only (stripped)
        assert not _diff(a, b), (name, _diff(a, b)[:8])
    assert "side_tape_seed" in _json(inp / "planner.json")


def test_failure_exit_codes_match_subprocess(orch, tmp_path):
    bad = tmp_path / "planner.json"
    bad.write_text("{}", encoding="utf-8")
    cases = [
        ("side_seed", ["--planner-json", str(bad)]),                      # SeedError -> 1
        ("side_seed", []),                                                # argparse -> 2
        ("base_link", ["--json", str(tmp_path / "nope.json"),
                       "--driver-extrinsic", "recorded"]),                # parser.error -> 2
        ("base_link", ["--json", str(bad), "--driver-extrinsic", "recorded"]),  # 1
        ("query_depth", ["--pcd", str(tmp_path / "nope.pcd"), "--pixel", "1", "2"]),  # 1
        ("query_depth", ["--pcd", str(bad)]),                             # parser.error -> 2
    ]
    for script, argv in cases:
        expected = _sub(script, *argv).returncode
        assert orch._run_script_main_inprocess(script, argv) == expected, (script, argv)
    assert bad.read_text(encoding="utf-8") == "{}"
