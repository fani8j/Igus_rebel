import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import right_cut as rc  # noqa: E402
import right_snapshot as rs  # noqa: E402

CAPTURE = Path(__file__).resolve().parents[3] / "scan_results/left_from_corner_20261014/snapshot/capture_1720.json"


def urdf(plugin, ip=None):
    param = f'<param name="robot_ip">{ip}</param>' if ip else ""
    return (f'<robot name="r"><ros2_control name="x" type="system"><hardware><plugin>{plugin}</plugin>'
            f'{param}</hardware></ros2_control></robot>')


def test_parse_hardware():
    assert rc.parse_hardware(urdf("mock_components/GenericSystem")) == ("mock_components/GenericSystem", None)
    assert rc.parse_hardware(urdf("Igus/Rebel", "192.168.3.12")) == ("Igus/Rebel", "192.168.3.12")
    with pytest.raises(rc.CutError):
        rc.parse_hardware("<robot/>")
    with pytest.raises(rc.CutError):
        rc.parse_hardware("not xml")


def test_identity_gate():
    assert rc.identity_verdict("mock_components/GenericSystem", None, ["mock_components/GenericSystem"])[0] == "mock"
    assert rc.identity_verdict("Igus/Rebel", "192.168.3.12", ["Igus/Rebel"])[0] == "real"
    for ip in ("192.168.3.11", "192.168.50.133", None, "192.168.3.120"):
        with pytest.raises(rc.CutError, match="REFUSED"):
            rc.identity_verdict("Igus/Rebel", ip, ["Igus/Rebel"])
    with pytest.raises(rc.CutError, match="REFUSED"):
        rc.identity_verdict("gz_ros2_control/GazeboSimSystem", None)
    with pytest.raises(rc.CutError, match="do not match"):
        rc.identity_verdict("mock_components/GenericSystem", None, ["Igus/Rebel"])
    with pytest.raises(rc.CutError, match="do not match"):
        rc.identity_verdict("Igus/Rebel", "192.168.3.12", ["Igus/Rebel", "Igus/Rebel"][:0] + ["mock_components/GenericSystem"])


def make_plan(tmp_path, cut="face"):
    out = tmp_path / "right.json"
    out.write_text(CAPTURE.read_text())
    state = {"cut": cut, "snapshot": str(out), "snapshot_sha256": rc.file_sha256(out), "message": "m"}
    return out, state


GOOD = {"face": {"cut_pattern": "left_only", "middle_seam_stop": "end"},
        "top": {"cut_pattern": "middle_only", "middle_seam_stop": "end", "top_cut_min_cut_fraction": 0.05}}


@pytest.mark.parametrize("cut", ["face", "top"])
def test_execute_preconditions_ok(tmp_path, cut):
    out, state = make_plan(tmp_path, cut)
    assert rc.check_execute_preconditions(state, str(out), str(out), dict(GOOD[cut]))


def test_execute_refuses_other_snapshot_file(tmp_path):
    out, state = make_plan(tmp_path)
    with pytest.raises(rc.CutError, match="not the converted"):
        rc.check_execute_preconditions(state, "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json",
                                       str(out), dict(GOOD["face"]))


def test_execute_refuses_without_state_changed_file_or_params(tmp_path):
    out, state = make_plan(tmp_path)
    with pytest.raises(rc.CutError, match="no plan record"):
        rc.check_execute_preconditions(None, str(out), str(out), dict(GOOD["face"]))
    out.write_text(out.read_text() + " ")
    with pytest.raises(rc.CutError, match="changed"):
        rc.check_execute_preconditions(state, str(out), str(out), dict(GOOD["face"]))
    out, state = make_plan(tmp_path, "top")
    bad = dict(GOOD["top"], middle_seam_stop="center")
    with pytest.raises(rc.CutError, match="middle_seam_stop"):
        rc.check_execute_preconditions(state, str(out), str(out), bad)
    bad = dict(GOOD["top"], top_cut_min_cut_fraction=0.5)
    with pytest.raises(rc.CutError, match="top_cut_min_cut_fraction"):
        rc.check_execute_preconditions(state, str(out), str(out), bad)


def test_domain_required(monkeypatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    with pytest.raises(rc.CutError, match="ROS_DOMAIN_ID"):
        rc.require_domain()
    monkeypatch.setenv("ROS_DOMAIN_ID", "196")
    assert rc.require_domain() == "196"


def test_main_refuses_without_domain(monkeypatch, capsys):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    assert rc.main(["disarm"]) == 2
    assert "ROS_DOMAIN_ID" in capsys.readouterr().err


def test_typed():
    assert rc._typed("false") is False and rc._typed("0.5") == 0.5 and rc._typed("abc") == "abc"
