"""Opt-in base-frame correction of the approach-assist predictions (ui/hover_ui.py, ui/server.py main, the launch file).
The correction is added once, where a mark's base-frame prediction is produced (HoverManager._detect), so prepare /
descend / the moved checks / after_touch / record_offset / the page state all see the same corrected point."""
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import pytest

from rebel_dual_arm_calibration.ui import hover_ui as hu
from rebel_dual_arm_calibration.ui import server as ui_server
from test_ui_hover import (N_BASE, P_BASE, detected, make, read_lines, refusal, tcp_pose, touch)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = (-5.3, 8.1, 0.0)
CORR_M = np.array(CORR) * 1e-3
MARK_101 = P_BASE + np.array([0.01, 0.0, 0.0])           # the fake's raw prediction of point 101


def corrected(tmp_path, corr=CORR, **kw):
    return detected(tmp_path, correction_mm={"left": corr}, **kw)


def prepared_position(mgr, pid=101):
    assert mgr.prepare("P1", pid)["ok"]
    return np.array(mgr._prepared["T_base_tcp"])[:3, 3]


# ---- default zero leaves everything bit-identical ----------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, {"correction_mm": {}}, {"correction_mm": {"left": (0, 0, 0), "right": (5, 5, 5)}},
                                {"correction_mm": {"left": (-0.0, 0.0, -0.0)}}])
def test_zero_correction_is_bit_identical(tmp_path, kw):
    base, other = detected(tmp_path), detected(tmp_path, **kw)
    assert set(base._preds) == set(other._preds) and base._preds
    for pid, a in base._preds.items():
        b = other._preds[pid]
        assert np.array_equal(a["_p"], b["_p"]) and a["_p"].tobytes() == b["_p"].tobytes()
        assert a["pred_base_mm"] == b["pred_base_mm"] and np.array_equal(b["_p"], b["_p_raw"])
        assert np.array_equal(a["_n"], b["_n"]) and np.array_equal(a["_x"], b["_x"])
    assert np.array_equal(prepared_position(base), prepared_position(other))
    j = other.fakes.jog
    j.T_tcp = np.eye(4)
    j.T_tcp[:3, 3] = MARK_101 + [0.001, 0.002, 0.0005]
    rec = other.record_offset(101)
    assert rec["offset"]["dx_mm"] == pytest.approx(1.0) and rec["offset"]["dy_mm"] == pytest.approx(2.0)
    line = read_lines(tmp_path, "hover_test.jsonl")[-1]
    assert line["correction_mm"] == [0.0, 0.0, 0.0] and line["predicted_raw_p_base_m"] == line["predicted_p_base_m"]
    st = other.state()
    assert st["correction_mm"]["left"] == [0.0, 0.0, 0.0] and st["correction_arm"] == "left"


# ---- non-zero correction: every consumer sees the corrected point ------------------------------------------------

def test_prediction_is_shifted_by_exactly_the_vector_and_the_normal_is_not(tmp_path):
    base, mgr = detected(tmp_path), corrected(tmp_path)
    for pid, a in base._preds.items():
        b = mgr._preds[pid]
        assert np.allclose(b["_p"] - a["_p"], CORR_M, atol=1e-12)
        assert np.array_equal(b["_p_raw"], a["_p"])                       # the raw prediction is kept untouched
        assert np.array_equal(a["_n"], b["_n"]) and np.array_equal(a["_x"], b["_x"])
        assert np.allclose(np.array(b["pred_base_mm"]) - a["pred_base_mm"], CORR, atol=1e-9)
        assert b["pred_raw_base_mm"] == a["pred_base_mm"]
    rows = {p["id"]: p for p in mgr.state()["points"]}                    # the page shows the corrected point
    assert rows[101]["pred_base_mm"] == pytest.approx(list((MARK_101 + CORR_M) * 1e3))


def test_prepare_targets_the_corrected_point(tmp_path):
    base, mgr = detected(tmp_path), corrected(tmp_path)
    pb, pc = prepared_position(base), prepared_position(mgr)
    assert np.allclose(pc - pb, CORR_M, atol=1e-12)
    assert np.allclose(pc, MARK_101 + CORR_M + N_BASE * hu.PREHOVER_HEIGHT_M, atol=1e-12)
    # the candidate poses handed to the IK are above the corrected point
    poses = [c for c in mgr.fakes.hover.calls if c[0] == "poses"]
    assert all(np.allclose(c[1], MARK_101 + CORR_M, atol=1e-12) for c in poses)


def test_fresh_and_stored_predictions_are_corrected_the_same_way(tmp_path):
    mgr = corrected(tmp_path)
    stored = mgr._preds[101]["_p"].copy()
    fresh = mgr._observe(101, mgr._preds[101])                            # unmoved board: both corrected, shift is 0
    assert np.array_equal(fresh["_p"], stored)
    assert np.allclose(fresh["_p"] - fresh["_p_raw"], CORR_M, atol=1e-12)
    assert prepared_position(mgr) is not None                             # prepare's moved check passes with the correction on
    # the 2 mm board-moved limit is unaffected by the correction: it compares corrected with corrected
    mgr.fakes.hover.shift = np.array([0.0, 0.0, 0.0021])
    assert "moved 2.1 mm" in refusal(mgr.prepare, "P1", 101).message
    mgr.fakes.hover.shift = np.array([0.0, 0.0, 0.0015])
    assert mgr.prepare("P1", 101)["ok"]


def test_descend_line_is_aimed_at_the_corrected_point(tmp_path):
    corr = (-5.3, 3.0, 4.0)
    mgr = corrected(tmp_path, corr)
    assert mgr.prepare("P1", 101)["ok"] and mgr.move("P1", "plan-1")["ok"]
    T = tcp_pose(height_mm=60.0)
    T[:3, 3] += np.array(corr) * 1e-3                                     # 60 mm above the CORRECTED mark
    mgr.fakes.jog.T_tcp = T
    mgr._pending_move["T"] = T.copy()
    assert mgr.descend("P1", 101, 20)["ok"]
    target = [c for c in mgr.fakes.jog.calls if c[0] == "descend"][0][3]
    assert np.allclose(target[:3, 3], MARK_101 + np.array(corr) * 1e-3 + N_BASE * 0.020, atol=1e-12)
    # a TCP over the UNcorrected mark is 7 mm off the line through the corrected one: refused
    mgr2 = corrected(tmp_path, (7.0, 0.0, 0.0))
    assert mgr2.prepare("P1", 101)["ok"] and mgr2.move("P1", "plan-1")["ok"]
    T2 = tcp_pose(height_mm=60.0)
    mgr2.fakes.jog.T_tcp = T2
    mgr2._pending_move["T"] = T2.copy()
    assert "7.0 mm off the line" in refusal(mgr2.descend, "P1", 101).message


def test_descend_clearance_uses_the_corrected_surface_height(tmp_path):
    mgr = corrected(tmp_path, (0.0, 0.0, 15.0))                           # surface believed 15 mm higher
    assert mgr.prepare("P1", 101)["ok"] and mgr.move("P1", "plan-1")["ok"]
    T = tcp_pose(height_mm=60.0 + 15.0)                                   # 60 mm above the corrected surface
    mgr.fakes.jog.T_tcp = T
    mgr._pending_move["T"] = T.copy()
    assert mgr.descend("P1", 101, 10)["ok"]
    target = [c for c in mgr.fakes.jog.calls if c[0] == "descend"][0][3]
    assert target[2, 3] == pytest.approx(P_BASE[2] + 0.015 + 0.010)


def test_after_touch_and_record_offset_report_the_remaining_error_and_log_the_correction(tmp_path):
    mgr = corrected(tmp_path)
    n0 = mgr.touch_count()
    mgr.touches.append(touch("left", 101, p=CORR_M + np.array([0.0019, -0.0012, 0.0001])))   # touched = corrected + (1.9, -1.2, 0.1) mm
    off = mgr.after_touch("left", 101, n0)
    assert off["dx_mm"] == pytest.approx(1.9, abs=1e-6) and off["dy_mm"] == pytest.approx(-1.2, abs=1e-6)
    assert off["dn_mm"] == pytest.approx(0.1, abs=1e-6)
    rec = read_lines(tmp_path, "hover_log.jsonl")[0]
    assert rec["correction_mm"] == list(CORR)
    assert rec["predicted_raw_p_base_m"] == pytest.approx(list(MARK_101))
    assert rec["predicted_p_base_m"] == pytest.approx(list(MARK_101 + CORR_M))
    assert np.array(rec["predicted_p_base_m"]) - np.array(rec["predicted_raw_p_base_m"]) == pytest.approx(CORR_M)
    assert rec["offset_mm"]["dx_mm"] == pytest.approx(1.9, abs=1e-6)       # offset_mm = touched - corrected prediction
    j = mgr.fakes.jog
    j.T_tcp = np.eye(4)
    j.T_tcp[:3, 3] = MARK_101 + CORR_M + [0.001, 0.002, 0.0005]
    r = mgr.record_offset(101)
    assert r["offset"]["dx_mm"] == pytest.approx(1.0) and r["offset"]["dy_mm"] == pytest.approx(2.0)
    assert r["offset"]["dn_mm"] == pytest.approx(0.5)
    t = read_lines(tmp_path, "hover_test.jsonl")[0]
    assert t["correction_mm"] == list(CORR) and t["predicted_raw_p_base_m"] == pytest.approx(list(MARK_101))
    assert t["predicted_p_base_m"] == pytest.approx(list(MARK_101 + CORR_M))
    assert t["touched_p_base_m"] == pytest.approx(list(MARK_101 + CORR_M + [0.001, 0.002, 0.0005]))
    # the same touch at the same place against an uncorrected manager shows the whole shift as error
    plain = detected(tmp_path)
    plain.fakes.jog.T_tcp = j.T_tcp.copy()
    off0 = plain.record_offset(101)["offset"]
    assert off0["dx_mm"] == pytest.approx(1.0 + CORR[0]) and off0["dy_mm"] == pytest.approx(2.0 + CORR[1])


def test_record_fields_are_always_present(tmp_path):
    mgr = detected(tmp_path)
    mgr.touches.append(touch("left", 101))
    mgr.after_touch("left", 101, 0)
    rec = read_lines(tmp_path, "hover_log.jsonl")[0]
    assert set(rec) == {"time", "time_iso", "kind", "arm", "point_id", "name", "source", "predicted_p_base_m",
                        "predicted_raw_p_base_m", "correction_mm", "touched_p_base_m", "offset_mm", "calibration",
                        "quality", "prediction_age_s"}
    assert rec["correction_mm"] == [0.0, 0.0, 0.0]


def test_state_reports_the_active_correction_per_arm(tmp_path):
    mgr = make(tmp_path, correction_mm={"left": CORR, "right": (1.0, 2.0, 3.0)})
    st = mgr.state()                                                      # before any detection
    assert st["correction_mm"] == {"left": list(CORR), "right": [1.0, 2.0, 3.0]} and st["correction_arm"] == "left"
    mgr.compute()
    assert mgr.state()["correction_mm"]["left"] == list(CORR)
    dual = make(tmp_path, mode="dual", correction_mm={"left": CORR})
    st = dual.state()
    assert st["supported"] is False and st["correction_arm"] is None and st["correction_mm"]["left"] == list(CORR)


# ---- validation ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("vec", [(float("nan"), 0, 0), (0, float("inf"), 0), (0, 0, -float("inf")), (20.1, 0, 0),
                                 (15.0, 15.0, 0.0), (0, 0, -21), (11.6, 11.6, 11.6)])
def test_validate_rejects_non_finite_and_long_vectors(vec):
    with pytest.raises(ValueError):
        hu.validate_correction_mm(vec, "x")
    with pytest.raises(ValueError):
        hu.HoverManager("single_left", {"left": object()}, None, lambda: None, correction_mm={"left": vec})


def test_validate_accepts_up_to_the_limit_and_names_the_parameter():
    assert hu.validate_correction_mm((20.0, 0.0, 0.0)) == (20.0, 0.0, 0.0)
    assert hu.validate_correction_mm((-5.3, 8.1, 0)) == (-5.3, 8.1, 0.0)
    with pytest.raises(ValueError, match="hover_correction_left_mm.*exceeds the 20 mm"):
        hu.corrections_from_params("15,15,0", "")
    with pytest.raises(ValueError, match="hover_correction_right_mm"):
        hu.corrections_from_params("", "nan,0,0")
    assert hu.corrections_from_params("-5.3,8.1,0", None) == {"left": (-5.3, 8.1, 0.0), "right": (0.0, 0.0, 0.0)}


@pytest.mark.parametrize("text,expect", [
    ("", (0, 0, 0)), (None, (0, 0, 0)), ("  ", (0, 0, 0)), ("[]", (0, 0, 0)),
    ("-5.3,8.1,0", (-5.3, 8.1, 0)), ("[-5.3, 8.1, 0]", (-5.3, 8.1, 0)), ("(-5.3 , 8.1 , 0.0)", (-5.3, 8.1, 0)),
    ("-5.3 8.1 0", (-5.3, 8.1, 0)), ("1e-1,2,3", (0.1, 2, 3)), ([1, 2.5, -3], (1, 2.5, -3)), ((0.0, 0.0, 0.0), (0, 0, 0)),
])
def test_parse_correction_mm(text, expect):
    assert hu.parse_correction_mm(text) == tuple(float(v) for v in expect)


@pytest.mark.parametrize("bad", ["1,2", "1,2,3,4", "a,b,c", "1,,", "[1,2]", [1, 2], [1, 2, 3, 4], [True, 0, 0], ["x", 0, 0], 5])
def test_parse_correction_mm_rejects_garbage(bad):
    with pytest.raises(ValueError):
        hu.parse_correction_mm(bad)


def test_non_finite_strings_parse_but_do_not_validate():
    assert math.isnan(hu.parse_correction_mm("nan,0,0")[0])
    with pytest.raises(ValueError, match="finite"):
        hu.validate_correction_mm("nan,0,0")
    with pytest.raises(ValueError, match="finite"):
        hu.validate_correction_mm("inf,0,0")


def test_warnings_name_the_arm_and_the_vector():
    out = []
    hu.warn_corrections({"left": (-5.3, 8.1, 0.0), "right": (0.0, 0.0, 0.0)}, "single_left", out.append)
    assert len(out) == 1 and "ACTIVE (left)" in out[0] and "-5.3 / +8.1 / +0.0 mm (base frame)" in out[0]
    out.clear()
    hu.warn_corrections({"left": (0, 0, 0), "right": (0, 0, 0)}, "single_left", out.append)
    assert out == []                                                      # nothing when zero
    hu.warn_corrections({"left": (-5.3, 8.1, 0.0), "right": (1.0, 0.0, 0.0)}, "single_right", out.append)
    assert len(out) == 2 and "ACTIVE (right)" in out[1] and "IGNORED" in out[0] and "left" in out[0]
    out.clear()
    hu.warn_corrections({"left": (1.0, 0.0, 0.0)}, "dual", out.append)
    assert len(out) == 1 and "IGNORED" in out[0]


# ---- the UI node refuses to start with a bad correction ----------------------------------------------------------

class _FakeLogger:
    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        return lambda msg, *a, **k: self.lines.append((level, msg))


class _FakeNode:
    def __init__(self, overrides):
        self.params, self.overrides, self.log, self.destroyed = {}, overrides, _FakeLogger(), False

    def declare_parameter(self, name, default, descriptor=None):
        self.params[name] = default

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self.overrides.get(name, self.params[name]))

    def get_logger(self):
        return self.log

    def destroy_node(self):
        self.destroyed = True


def _run_main(monkeypatch, overrides):
    node = _FakeNode(overrides)
    shut = []
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: None
    rclpy.create_node = lambda name: node
    rclpy.shutdown = lambda: shut.append(True)
    rclpy.ok = lambda: True
    executors = types.ModuleType("rclpy.executors")
    executors.MultiThreadedExecutor = type("MTE", (), {"__init__": lambda self, **k: (_ for _ in ()).throw(AssertionError("started"))})
    rcl = types.ModuleType("rcl_interfaces")
    rcl_msg = types.ModuleType("rcl_interfaces.msg")
    rcl_msg.ParameterDescriptor = lambda **k: types.SimpleNamespace(**k)
    bridge = types.ModuleType("rebel_dual_arm_calibration.ui.ros_bridge")
    bridge.RosBridge = type("RosBridge", (), {"__init__": lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("bridge built"))})
    for name, mod in (("rclpy", rclpy), ("rclpy.executors", executors), ("rcl_interfaces", rcl),
                      ("rcl_interfaces.msg", rcl_msg), ("rebel_dual_arm_calibration.ui.ros_bridge", bridge)):
        monkeypatch.setitem(sys.modules, name, mod)
    return node, shut


def test_main_refuses_to_start_with_a_bad_correction(monkeypatch):
    node, shut = _run_main(monkeypatch, {"hover_correction_left_mm": "-15,15,0"})
    with pytest.raises(SystemExit) as e:
        ui_server.main([])
    assert "hover_correction_left_mm" in str(e.value) and "exceeds the 20 mm" in str(e.value)
    assert node.destroyed and shut == [True]
    assert any(level == "error" and "refusing to start" in msg for level, msg in node.log.lines)
    assert node.params["hover_correction_left_mm"] == "" and node.params["hover_correction_right_mm"] == ""


def test_main_refuses_non_finite_array_parameter(monkeypatch):
    node, _ = _run_main(monkeypatch, {"hover_correction_right_mm": [0.0, float("nan"), 0.0]})
    with pytest.raises(SystemExit) as e:
        ui_server.main([])
    assert "hover_correction_right_mm" in str(e.value) and "finite" in str(e.value)


def test_main_validates_before_the_optional_panels_are_built():
    src = open(ui_server.__file__).read()
    main = src[src.index("def main("):]
    assert main.index("corrections_from_params(") < main.index("RosBridge(node") < main.index("HoverManager(mode")
    assert "correction_mm=hover_correction" in main and "warn_corrections(hover_correction, mode" in main
    assert 'ParameterDescriptor(dynamic_typing=True)' in main


# ---- launch arguments ------------------------------------------------------------------------------------------

def test_launch_declares_the_arguments_and_validates_them():
    text = open(os.path.join(PKG, "launch", "calibration_ui.launch.py")).read()
    for arm in ("left", "right"):
        assert f'DeclareLaunchArgument("hover_correction_{arm}_mm", default_value=""' in text
    assert "validate_correction_mm(" in text and "ParameterValue(" in text and "value_type=str" in text


def _load_launch():
    pytest.importorskip("launch_ros")
    spec = importlib.util.spec_from_file_location("calibration_ui_launch", os.path.join(PKG, "launch", "calibration_ui.launch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launch_params(mod, **cfg):
    from launch import LaunchContext
    ctx = LaunchContext()
    base = {"mode": "single_left", "targets_file": "", "port": "8095", "image_topic": "/i", "camera_info_topic": "/c",
            "output_dir": "", "tcp_xacro_path": "", "allow_real_move": "false",
            "hover_correction_left_mm": "", "hover_correction_right_mm": ""}
    ctx.launch_configurations.update(dict(base, **cfg))
    (node,) = mod.launch_setup(ctx)
    return node


def _param_dict(node):
    from launch import LaunchContext
    ctx = LaunchContext()
    return {"".join(sub.perform(ctx) for sub in k): v for d in node._Node__parameters for k, v in d.items()}


def test_launch_setup_passes_only_non_zero_corrections_and_rejects_bad_ones():
    mod = _load_launch()
    params = _param_dict(_launch_params(mod))
    assert "hover_correction_left_mm" not in params and "hover_correction_right_mm" not in params    # empty = off
    params = _param_dict(_launch_params(mod, hover_correction_left_mm="[-5.3, 8.1, 0]"))
    assert "hover_correction_right_mm" not in params
    v = params["hover_correction_left_mm"]
    assert v.value_type is str and "-5.3,8.1,0.0" in str(v)           # a canonical string, never YAML-evaluated
    assert hu.parse_correction_mm("-5.3,8.1,0.0") == (-5.3, 8.1, 0.0)
    with pytest.raises(ValueError, match="hover_correction_left_mm"):
        _launch_params(mod, hover_correction_left_mm="30,0,0")
    with pytest.raises(ValueError, match="hover_correction_right_mm"):
        _launch_params(mod, hover_correction_right_mm="1,2")


# ---- the page ---------------------------------------------------------------------------------------------------

def test_page_shows_the_correction_only_when_non_zero():
    static = os.path.join(os.path.dirname(ui_server.__file__), "static")
    js = open(os.path.join(static, "app.js")).read()
    css = open(os.path.join(static, "style.css")).read()
    assert "Prediction correction ACTIVE (" in js and "mm (base frame)" in js
    assert "v.every((c) => c === 0)) return ''" in js                                # nothing when zero
    assert "corrEl.hidden = !corrTxt" in js and 'class="hv-corr"' in js
    assert ".hv-corr[hidden]" in css
    assert json.dumps(hu.ZERO_CORRECTION_MM) == "[0.0, 0.0, 0.0]"
