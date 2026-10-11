"""Orchestrator --sam3-daemon resolution (S2 wiring): live -> auto, offline -> off,
explicit on/off wins, an already-set AUTOLABEL_V2_SAM3_DAEMON is respected under auto."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sam3_flag_helpers",
                                               str(TESTS / "test_autolabel_v2_retry.py"))
retry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retry)
live = retry.live


@pytest.fixture()
def orch():
    return live._load(live.ORCHESTRATOR, "capture_and_query_depth")


@pytest.mark.parametrize("flag,from_image,env,expected", [
    ("auto", None, {}, "auto"),                      # live run
    ("auto", "x.png", {}, "off"),                    # offline replay
    ("on", "x.png", {}, "on"),
    ("off", None, {}, "off"),
    ("auto", "x.png", {"AUTOLABEL_V2_SAM3_DAEMON": "auto"}, "auto"),
    ("off", None, {"AUTOLABEL_V2_SAM3_DAEMON": "on"}, "off"),
])
def test_sam3_daemon_mode(orch, flag, from_image, env, expected):
    args = SimpleNamespace(sam3_daemon=flag, live=from_image is None)
    assert orch.sam3_daemon_mode(args, env) == expected
