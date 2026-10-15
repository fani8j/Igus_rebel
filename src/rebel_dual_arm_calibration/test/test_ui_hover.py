"""Approach-assist backend (ui/hover_ui.py): the HoverManager with fake detection / fit / IK / validity / jog.
No ROS graph, no node: the geometry module (hover.py) and the fit are replaced by fakes with the documented API."""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from rebel_dual_arm_calibration.ui import hover_ui as hu
from rebel_dual_arm_calibration.ui.jog import REAL_MOVE_DISABLED_MSG, arm_configs

CFGS = arm_configs("single_left")
CFG = CFGS["left"]
NAMES = list(CFG.joint_names)
Q0 = {n: 0.1 * (i + 1) for i, n in enumerate(NAMES)}
T_L6_TCP = np.eye(4)
T_L6_TCP[:3, 3] = [0.0, 0.0, 0.15]
P_BASE = np.array([0.40, 0.10, 0.02])          # a mark on a horizontal sheet
N_BASE = np.array([0.0, 0.0, 1.0])
X_BASE = np.array([1.0, 0.0, 0.0])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class FakeHover:
    """hover.py's API: detect_targets / to_base / hover_tcp_poses / compare / project_points."""

    def __init__(self):
        self.calls = []
        self.detected = {101: "TOP", 102: "P2", 1: "board 1"}
        self.status = {"sheet": "ok: marker seen", "table": "ok"}
        self.fail = None
        self.unmoved_result = (True, "", {})      # hover.unmoved() answer (descend / prepare fallback)
        self.shift = np.zeros(3)          # the board "moved" by this much (m) in every later detection
        self.tilt_deg = 0.0               # ... and its normal tilted by this much (about x)
        self.sources = None               # {point id: source} (hover tiles); default: table below 100, else "sheet"

    def source_of(self, pid):
        if self.sources is not None:
            return self.sources[pid]
        return "table" if pid < 100 else "sheet"

    def detect_targets(self, frames, table_spec, sheet_spec, detector):
        self.calls.append(("detect", len(frames), table_spec, sheet_spec, detector))
        if self.fail:
            raise RuntimeError(self.fail)
        pts = {pid: SimpleNamespace(quality={"n_frames": len(frames), "rms_px": np.float64(0.4)},
                                    source=self.source_of(pid), point_id=pid) for pid in self.detected}
        return pts, dict(self.status)

    def unmoved(self, frames, stored, table_spec, sheet_spec, detector, sources=None):
        self.calls.append(("unmoved", len(frames), dict(stored), table_spec, sheet_spec, detector, sources))
        return self.unmoved_result

    def to_base(self, points, X):
        self.calls.append(("to_base", sorted(points), np.array(X)))
        a = math.radians(self.tilt_deg)
        n = np.array([0.0, -math.sin(a), math.cos(a)])
        return {pid: {"p": P_BASE + 0.01 * k * np.array([1, 0, 0]) + self.shift, "n": n, "x": X_BASE.copy(),
                      "name": self.detected[pid], "source": self.source_of(pid)}
                for k, pid in enumerate(sorted(points))}

    def hover_tcp_poses(self, p, n, x, height_m=0.01, yaw_step_deg=30.0):
        self.calls.append(("poses", np.array(p), height_m, yaw_step_deg))
        out = []
        for k in range(int(round(360.0 / yaw_step_deg))):
            T = np.eye(4)
            xa = rz(math.radians(k * yaw_step_deg)) @ x
            za = -np.asarray(n)
            T[:3, 0], T[:3, 2], T[:3, 1] = xa, za, np.cross(za, xa)
            T[:3, 3] = np.asarray(p) + np.asarray(n) * height_m
            out.append(T)
        return out

    def compare(self, pred, touched_p_base):
        d = np.asarray(touched_p_base) - pred["p"]
        n, x = pred["n"], pred["x"]
        y = np.cross(n, x)
        dx, dy, dn = float(d @ x) * 1e3, float(d @ y) * 1e3, float(d @ n) * 1e3
        return {"dx_mm": dx, "dy_mm": dy, "dn_mm": dn, "lateral_mm": math.hypot(dx, dy),
                "dist_mm": float(np.linalg.norm(d)) * 1e3}

    def project_points(self, points, K, dist):
        return {pid: (100.0 + 10 * pid % 500, 200.0) for pid in points}


class FakeFit:
    def __init__(self):
        self.calls = []

    def fit_arm(self, samples, arm):
        self.calls.append((len(samples), arm))
        if len(samples) < 6:
            raise ValueError(f"arm '{arm}': {len(samples)} usable samples, at least 6 are needed")
        return SimpleNamespace(arm=arm, X=np.eye(4), K=np.eye(3) * 600, dist=np.zeros(5), image_size=(640, 480))


class FakeMoveIt:
    def __init__(self):
        self.ready_answer = (True, "")
        self.ik_calls, self.valid_calls, self.published = [], [], []
        self.q_for_call = None            # i -> joint-change in deg (or None for an IK failure) relative to Q0
        self.valid_answer = lambda q: (True, "")
        self.fk_table = {}

    def ready(self):
        return self.ready_answer

    def ik(self, cfg, T6, seed_q):
        i = len(self.ik_calls) % 12                       # the candidate index within one prepare (12 yaw candidates)
        self.ik_calls.append((cfg, np.array(T6), dict(seed_q)))
        deg = self.q_for_call(i) if self.q_for_call else 10.0 + i
        if deg is None:
            return None
        q = dict(Q0)
        q[NAMES[i % 6]] += math.radians(deg)
        q[NAMES[(i + 1) % 6]] += 1e-3 * (i + 1)             # makes every candidate's q unique
        self.fk_table[tuple(q[n] for n in NAMES)] = np.array(T6) @ T_L6_TCP
        return q

    def valid(self, cfg, q):
        self.valid_calls.append(dict(q))
        return self.valid_answer(q)

    def publish_goal(self, names, positions):
        self.published.append((list(names), list(positions)))


class FakeJog:
    def __init__(self, moveit):
        self.moveit = moveit
        self.busy = False
        self.calls = []
        self.core = SimpleNamespace(stop_epoch=3)
        self.q = dict(Q0)
        self.T_tcp = None                    # tcp_pose_base answer
        self.tcp_why = ""
        self.plan_answer = {"ok": True, "plan_id": "plan-1", "message": "planned", "duration_s": 6.2,
                            "max_joint_change_deg": 12.5, "n_points": 40}
        self.exec_answer = {"ok": True, "id": 5, "message": "moving"}
        self.last = None                     # the jog's status()["last"]
        self.fault = False
        self.hardware = {}                   # arm -> {"real": bool}
        self.allow_real_move = True
        self.descend_answer = {"ok": True, "id": 9, "message": "accepted"}
        self.fk_offset = np.zeros(3)         # make tcp_fk disagree with the requested pose

    def status(self):
        return {"available": True, "busy": self.busy, "move_plan": None, "last": self.last, "fault": self.fault,
                "hardware": self.hardware, "allow_real_move": self.allow_real_move}

    def joint_window(self, arms, window_s):
        return (dict(self.q), 0.0, True) if self.q is not None else None

    def link6_to_tcp(self, arm):
        return T_L6_TCP.copy(), ""

    def tcp_fk(self, arm, q):
        T = self.moveit.fk_table.get(tuple(q[n] for n in NAMES))
        if T is None:
            return None, "unknown joint goal"
        T = T.copy()
        T[:3, 3] += self.fk_offset
        return T, ""

    def tcp_pose_base(self, arm, require_standstill=False):
        self.calls.append(("tcp_pose_base", arm, require_standstill))
        return (None if self.T_tcp is None else self.T_tcp.copy()), self.tcp_why

    def plan_move(self, page_id, arm, q_goal, **kw):
        self.calls.append(("plan_move", page_id, arm, dict(q_goal), kw))
        return dict(self.plan_answer)

    def execute_move(self, page_id, plan_id, **kw):
        self.calls.append(("execute_move", page_id, plan_id, kw))
        if self.exec_answer.get("ok"):
            self.last = {"id": self.exec_answer.get("id"), "ok": True, "phase": "done", "kind": "move", "plan_id": plan_id}
        return dict(self.exec_answer)

    def descend(self, page_id, arm, T_base_target, **kw):
        self.calls.append(("descend", page_id, arm, np.array(T_base_target), kw))
        return dict(self.descend_answer)

    def clear_plan(self):
        self.calls.append(("clear_plan",))


class Frames:
    def __init__(self, n=10, shape=(480, 640, 3)):
        self.n, self.shape, self.i = n, shape, 0

    def __call__(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return np.zeros(self.shape, np.uint8), self.i, 0.0


def samples(n, arm="left"):
    return [SimpleNamespace(arm=arm) for _ in range(n)]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, mode="single_left", n_samples=8, frames=None, scene=True, touches=None, **kw):
    moveit = FakeMoveIt()
    jog = FakeJog(moveit)
    hover, fit = FakeHover(), FakeFit()
    clock = Clock()
    scene_obj = SimpleNamespace(table_spec=SimpleNamespace(touch_points={1: (0, 0), 2: (1, 1)}),
                                sheet_spec=SimpleNamespace(points={101: (0, 0.06), 102: (0.03, 0), 113: (0, 0)},
                                                           names={101: "TOP", 102: "P2", 113: "OPT"}),
                                detector=object())
    touches = touches if touches is not None else []
    mgr = hu.HoverManager(mode, CFGS if mode != "dual" else arm_configs("dual"), jog, lambda: str(tmp_path),
                          moveit=moveit, hover=hover, fit_engine=fit,
                          load_samples=lambda d: samples(n_samples) + samples(3, "right"),
                          load_touches=lambda d: list(touches), image_fn=Frames(n=10 ** 6) if frames is None else frames,
                          scene_fn=(lambda: scene_obj) if scene else None, clock=clock, wall_clock=lambda: 1.7e9,
                          sleep=lambda s: None, mono=lambda: 0.0, **kw)
    mgr.scene, mgr.fakes, mgr.clock_obj = scene_obj, SimpleNamespace(moveit=moveit, jog=jog, hover=hover, fit=fit), clock
    mgr.touches = touches
    return mgr


def detected(tmp_path, **kw):
    mgr = make(tmp_path, **kw)
    mgr.compute()
    assert mgr.state()["state"] == "ready"
    return mgr


def refusal(fn, *a, **k):
    with pytest.raises(hu.HoverRefused) as e:
        fn(*a, **k)
    return e.value


# ---------------------------------------------------------------------------------------------------
# compute

def test_compute_success_state_and_wiring(tmp_path):
    mgr = make(tmp_path)
    msg = mgr.compute()
    assert "3 point(s) detected" in msg and "TOP" in msg and "not detected: 2, 113" in msg
    f = mgr.fakes
    assert f.fit.calls == [(8, "left")]                               # only the left arm's samples
    kind, n_frames, table, sheet, det = f.hover.calls[0]
    assert kind == "detect" and n_frames == hu.N_FRAMES and table is mgr.scene.table_spec and sheet is mgr.scene.sheet_spec
    assert det is mgr.scene.detector
    assert np.allclose(f.hover.calls[1][2], np.eye(4))                # to_base got the fitted X
    st = mgr.state()
    assert st["supported"] and st["state"] == "ready" and st["arm"] == "left" and st["calib_samples"] == 8
    by = {p["id"]: p for p in st["points"]}
    assert set(by) == {1, 2, 101, 102, 113}
    assert by[101]["detected"] and by[101]["name"] == "TOP" and by[101]["source"] == "sheet"
    assert by[101]["pred_base_mm"] == pytest.approx([P_BASE[0] * 1e3 + 10.0, P_BASE[1] * 1e3, P_BASE[2] * 1e3])
    assert by[101]["uv"] is not None and by[101]["quality"]["n_frames"] == hu.N_FRAMES
    assert by[101]["last_offset"] is None
    assert not by[2]["detected"] and by[2]["reason"] == "ok" and by[2]["pred_base_mm"] is None   # table status text
    assert not by[113]["detected"] and by[113]["source"] == "sheet" and by[113]["reason"]
    json.dumps(st, allow_nan=False)                                   # JSON clean
    assert st["test_summary"] == {"n": 0, "points": [], "mean_lateral_mm": None, "max_lateral_mm": None}


def test_compute_frames_are_the_newest_distinct_ones(tmp_path):
    seen = []

    def source():
        seen.append(1)
        return np.zeros((480, 640, 3), np.uint8), 7, 0.0              # the SAME frame every time
    mgr = make(tmp_path, frames=source, frame_wait_s=1.0)
    e = refusal(mgr.compute)
    assert e.status == 503 and "only 1 fresh camera frame" in e.message and len(seen) >= 4
    assert mgr.state()["state"] == "error" and "camera" in mgr.state()["message"]


def test_compute_failure_texts(tmp_path):
    e = refusal(make(tmp_path, n_samples=3).compute)
    assert e.status == 400 and "at least 6" in e.message
    mgr = make(tmp_path, n_samples=3)
    refusal(mgr.compute)
    assert mgr.state()["state"] == "error" and "at least 6" in mgr.state()["message"]
    e = refusal(make(tmp_path, scene=False).compute)
    assert e.status == 503 and "target definitions" in e.message
    e = refusal(make(tmp_path, frames=Frames(shape=(720, 1280, 3))).compute)
    assert e.status == 409 and "1280x720" in e.message and "640x480" in e.message
    mgr = make(tmp_path)
    mgr.fakes.hover.fail = "boom"
    e = refusal(mgr.compute)
    assert e.status == 500 and "point detection failed" in e.message and "boom" in e.message
    assert mgr.state()["state"] == "error"
    mgr = make(tmp_path)
    mgr.fakes.hover.detected = {}
    mgr.fakes.hover.status = {"sheet": "marker 13 not found", "table": "board not visible"}
    e = refusal(mgr.compute)
    assert e.status == 409 and "no point detected" in e.message and "marker 13 not found" in e.message
    st = mgr.state()
    assert st["state"] == "error" and all(not p["detected"] for p in st["points"])
    assert {p["id"]: p["reason"] for p in st["points"]}[101] == "marker 13 not found"     # greyed with the reason
    mgr = make(tmp_path)
    mgr.session_dir_fn = lambda: None
    assert refusal(mgr.compute).status == 503
    e = refusal(make(tmp_path).compute, "right")
    assert e.status == 400


def test_dual_mode_is_refused_everywhere(tmp_path):
    mgr = make(tmp_path, mode="dual")
    assert not mgr.supported
    st = mgr.state()
    assert st["supported"] is False and "single-arm" in st["message"] and st["points"] == []
    for fn, args in ((mgr.compute, ()), (mgr.prepare, ("P", 101)), (mgr.send_rviz, (101,)), (mgr.move, ("P", "plan-1")),
                     (mgr.descend, ("P", 101)), (mgr.record_offset, (101,))):
        e = refusal(fn, *args)
        assert e.status == 400 and "single-arm modes only" in e.message
    assert mgr.after_touch("left", 101, 0) is None and mgr.overlay() is None
    assert mgr.fakes.jog.calls == [] and mgr.fakes.moveit.published == []


def test_overlay_lists_the_detected_points(tmp_path):
    mgr = make(tmp_path)
    assert mgr.overlay() is None
    mgr.compute()
    ov = mgr.overlay()
    assert {p["id"] for p in ov["points"]} == {1, 101, 102}
    assert not any(p["selected"] for p in ov["points"])
    assert [p["id"] for p in mgr.overlay(102)["points"] if p["selected"]] == [102]


# ---------------------------------------------------------------------------------------------------
# prepare

def test_prepare_picks_the_smallest_joint_change_and_plans_it(tmp_path):
    mgr = detected(tmp_path)
    f = mgr.fakes
    changes = [90.0, 40.0, 70.0, 100.0] + [110.0] * 8
    f.moveit.q_for_call = lambda i: changes[i]
    res = mgr.prepare("P1", 101, "prehover", epoch=3)
    assert res["ok"] and res["plan_id"] == "plan-1" and res["prepared"] is True
    assert "IK pose 2 of 12" in res["message"] and "60 mm" in res["message"] and "40.0 deg" in res["message"]
    assert len(f.moveit.ik_calls) == 12                               # every yaw candidate was tried
    cfg, T6, seed = f.moveit.ik_calls[1]
    assert seed == pytest.approx(Q0)                                  # seeded with the current joints
    mark = P_BASE + np.array([0.01, 0, 0])                           # the predicted position of point 101
    _, p, height, step = [c for c in f.hover.calls if c[0] == "poses"][-1]
    assert height == 0.06 and step == 30.0 and np.allclose(p, mark)    # prehover = 60 mm, 30 deg yaw steps
    # the IK is asked for the FLANGE: T_tcp @ inv(T_link6_tcp)
    T_tcp = f.hover.hover_tcp_poses(mark, N_BASE, X_BASE, 0.06, 30.0)[1]
    assert np.allclose(T6, T_tcp @ np.linalg.inv(T_L6_TCP))
    kind, page, arm, q_goal, kw = [c for c in f.jog.calls if c[0] == "plan_move"][0]
    assert (page, arm, kw) == ("P1", "left", {"epoch": 3})
    assert max(abs(q_goal[n] - Q0[n]) for n in NAMES) == pytest.approx(math.radians(40.0), abs=1e-2)
    st = mgr.state()
    assert st["prepared"]["point_id"] == 101 and st["prepared"]["stage"] == "prehover"
    assert st["prepared"]["max_joint_change_deg"] == pytest.approx(40.0, abs=0.6) and "q" not in st["prepared"]
    assert st["plan"]["plan_id"] == "plan-1" and st["selected"] == 101
    assert f.moveit.valid_calls and f.moveit.published == []          # nothing is published to RViz by prepare


def test_prepare_rejects_more_than_120_degrees(tmp_path):
    mgr = detected(tmp_path)
    mgr.fakes.moveit.q_for_call = lambda i: 125.0 + i
    e = refusal(mgr.prepare, "P1", 101)
    assert e.status == 409 and "125 deg" in e.message and "limit 120" in e.message and "closer" in e.message
    assert [c for c in mgr.fakes.jog.calls if c[0] == "plan_move"] == []
    assert mgr.fakes.moveit.valid_calls == []                         # not even a validity check for a rejected pose
    assert mgr.state()["prepared"] is None
    mgr.fakes.moveit.q_for_call = lambda i: 119.0 if i == 5 else 150.0
    assert mgr.prepare("P1", 101)["ok"]                               # 119 deg is fine


def test_prepare_skips_ik_failures_wrong_poses_and_invalid_states(tmp_path):
    mgr = detected(tmp_path)
    f = mgr.fakes
    f.moveit.q_for_call = lambda i: None
    e = refusal(mgr.prepare, "P1", 101)
    assert e.status == 409 and "no reachable pre-hover pose" in e.message and "IK failed 12" in e.message
    # the nearest pose is in collision: the next one is used
    f.moveit.q_for_call = lambda i: [20.0, 30.0, 40.0][i] if i < 3 else None
    f.moveit.ik_calls.clear()
    f.moveit.valid_answer = lambda q: (False, "invalid state (a / b)") if abs(q[NAMES[0]] - Q0[NAMES[0]] - math.radians(20)) < 0.05 else (True, "")
    res = mgr.prepare("P1", 101)
    assert res["ok"] and "IK pose 2 of 12" in res["message"] and len(f.moveit.valid_calls) == 2
    # everything invalid
    f.moveit.valid_answer = lambda q: (False, "invalid")
    e = refusal(mgr.prepare, "P1", 101)
    assert "in collision or invalid" in e.message and "3 reachable" in e.message
    # the model's FK says the IK solution does not reach the requested TCP pose
    f.moveit.valid_answer = lambda q: (True, "")
    f.jog.fk_offset = np.array([0.002, 0, 0])
    e = refusal(mgr.prepare, "P1", 101)
    assert "wrong TCP pose 3" in e.message


def test_prepare_refusals(tmp_path):
    mgr = make(tmp_path)
    assert "Detect points" in refusal(mgr.prepare, "P", 101).message          # nothing detected yet
    mgr.compute()
    assert refusal(mgr.prepare, "P", 113).status == 409 and "was not detected" in refusal(mgr.prepare, "P", 113).message
    assert refusal(mgr.prepare, "P", True).status == 400 and refusal(mgr.prepare, "P", "101").status == 400
    assert refusal(mgr.prepare, "P", 101, "hover").status == 400                # only the 60 mm stage exists
    mgr.fakes.jog.busy = True
    assert "busy" in refusal(mgr.prepare, "P", 101).message
    mgr.fakes.jog.busy = False
    mgr.fakes.moveit.ready_answer = (False, "/compute_ik is not available")
    assert refusal(mgr.prepare, "P", 101).status == 503
    mgr.fakes.moveit.ready_answer = (True, "")
    mgr.fakes.jog.q = None
    assert "joint states" in refusal(mgr.prepare, "P", 101).message
    mgr.fakes.jog.q = dict(Q0)
    mgr.moveit = None
    assert refusal(mgr.prepare, "P", 101).status == 503
    # an old prediction: motion is decided by the FRESH observation, not by age - a moved paper is refused
    mgr.moveit = mgr.fakes.moveit
    mgr.clock_obj.t += hu.PRED_MAX_AGE_S + 1
    mgr.fakes.hover.shift = np.array([0.0, 0.0, 0.0021])
    e = refusal(mgr.prepare, "P", 101)
    assert "the board moved or the detection changed" in e.message and "Detect points" in e.message
    assert mgr.fakes.moveit.ik_calls == []
    mgr.fakes.hover.shift = np.zeros(3)
    mgr.fakes.jog.T_tcp = np.eye(4)                                       # a comparison moves nothing: no age limit
    assert mgr.record_offset(101)["ok"]


def test_prepare_is_void_after_a_stop(tmp_path):
    mgr = detected(tmp_path)
    mgr.fakes.jog.core.stop_epoch = 4                                  # STOP pressed after the request was ingressed (epoch 3)
    e = refusal(mgr.prepare, "P", 101, epoch=3)
    assert "STOP was pressed" in e.message
    assert mgr.fakes.moveit.ik_calls == [] and [c for c in mgr.fakes.jog.calls if c[0] == "plan_move"] == []


def test_prepare_on_real_hardware_without_allow_real_move_still_offers_rviz(tmp_path):
    mgr = detected(tmp_path)
    mgr.fakes.jog.plan_answer = {"ok": False, "id": None, "plan_id": None, "message": REAL_MOVE_DISABLED_MSG}
    res = mgr.prepare("P1", 101)
    assert res["ok"] is True and res["plan_id"] is None and res["prepared"] is True
    assert "Send to RViz" in res["message"] and "disabled on the REAL robot" in res["message"]
    st = mgr.state()
    assert st["prepared"]["point_id"] == 101 and st["plan"] is None
    assert "Allow External Comm" in mgr.send_rviz(101)
    mgr.fakes.jog.plan_answer = {"ok": False, "plan_id": None, "message": "move blocked: MoveIt found no path"}
    res = mgr.prepare("P1", 101)
    assert res["ok"] is False and "planning failed: move blocked" in res["message"] and res["prepared"] is True


# ---------------------------------------------------------------------------------------------------
# send_rviz / move

def test_send_rviz_publishes_only_the_joint_goal(tmp_path):
    mgr = detected(tmp_path)
    assert "press 'Plan hover' first" in refusal(mgr.send_rviz, 101).message
    mgr.fakes.moveit.q_for_call = lambda i: 30.0 if i == 2 else 100.0
    mgr.prepare("P1", 101)
    n_jog = len(mgr.fakes.jog.calls)
    assert "Allow External Comm" in mgr.send_rviz(101)
    assert len(mgr.fakes.moveit.published) == 1
    names, positions = mgr.fakes.moveit.published[0]
    assert names == NAMES and len(positions) == 6
    q = dict(zip(names, positions))
    assert max(abs(q[n] - Q0[n]) for n in NAMES) == pytest.approx(math.radians(30.0), abs=1e-2)
    assert len(mgr.fakes.jog.calls) == n_jog                           # no plan, no execute, no step: only the RViz goal
    assert "press 'Plan hover' first" in refusal(mgr.send_rviz, 102).message   # the prepared pose is for point 101
    mgr.moveit = None
    assert refusal(mgr.send_rviz, 101).status == 503


def test_move_executes_exactly_the_prepared_plan(tmp_path):
    mgr = detected(tmp_path)
    assert "no stored plan" in refusal(mgr.move, "P1", "plan-1").message
    mgr.prepare("P1", 101)
    assert "no stored plan" in refusal(mgr.move, "P1", "plan-9").message         # another id
    res = mgr.move("P1", "plan-1", epoch=3)
    assert res["ok"]
    ex = [c for c in mgr.fakes.jog.calls if c[0] == "execute_move"]
    assert ex == [("execute_move", "P1", "plan-1", {"epoch": 3})]
    assert mgr.state()["plan"] is None                                  # consumed
    assert "no stored plan" in refusal(mgr.move, "P1", "plan-1").message
    # a new detection drops the plan
    mgr.prepare("P1", 101)
    mgr.compute()
    assert mgr.state()["plan"] is None and ("clear_plan",) in mgr.fakes.jog.calls
    assert "no stored plan" in refusal(mgr.move, "P1", "plan-1").message
    # a failed execute (real hardware: not allowed, not enabled ...) keeps the plan state of the jog
    mgr.prepare("P1", 101)
    mgr.fakes.jog.exec_answer = {"ok": False, "message": REAL_MOVE_DISABLED_MSG}
    res = mgr.move("P1", "plan-1")
    assert res["ok"] is False and res["message"] == REAL_MOVE_DISABLED_MSG and mgr.state()["plan"] is not None


# ---------------------------------------------------------------------------------------------------
# descend

def tcp_pose(lateral_mm=0.0, height_mm=60.0, tilt_deg=0.0, lat_dir=(1, 0, 0), yaw_deg=25.0):
    """A TCP pose above the 101 mark (P_BASE + 10 mm in x), z axis down (-n) tilted by tilt_deg about x."""
    mark = P_BASE + np.array([0.01, 0, 0])
    R = rz(math.radians(yaw_deg)) @ np.diag([1.0, -1.0, -1.0])           # z = -n
    a = math.radians(tilt_deg)
    R = R @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mark + N_BASE * height_mm * 1e-3 + np.asarray(lat_dir, float) * lateral_mm * 1e-3
    return T


def descend_case(tmp_path, **kw):
    """Detected points, a finished hover move for point 101 (the descend interlock) and the TCP at `tcp_pose(**kw)`.  The
    interlock's pre-hover pose is set to that TCP so that the lateral / tilt / height limits can be tested on their own."""
    mgr = detected(tmp_path)
    assert mgr.prepare("P1", 101)["ok"]
    assert mgr.move("P1", "plan-1")["ok"]
    T = tcp_pose(**kw)
    mgr.fakes.jog.T_tcp = T
    mgr._pending_move["T"] = T.copy()
    return mgr


def test_descend_goes_straight_down_the_normal_to_the_default_20_mm(tmp_path):
    mgr = descend_case(tmp_path, lateral_mm=3.0, height_mm=60.0, tilt_deg=2.0)
    res = mgr.descend("P1", 101, epoch=3)
    assert res["ok"] and "descending 40.0 mm" in res["message"] and "20 mm above TOP (101)" in res["message"]
    kind, page, arm, T, kw = [c for c in mgr.fakes.jog.calls if c[0] == "descend"][0]
    assert (page, arm, kw) == ("P1", "left", {"epoch": 3})
    T_now = mgr.fakes.jog.T_tcp
    assert np.allclose(T[:3, :3], T_now[:3, :3])                          # orientation unchanged
    assert np.allclose(T[:3, 3] - T_now[:3, 3], [0, 0, -0.040], atol=1e-9)  # straight along -n, 40 mm
    assert ("tcp_pose_base", "left", True) in mgr.fakes.jog.calls          # a measured standstill is required


@pytest.mark.parametrize("kw,frag", [
    (dict(lateral_mm=5.6), "5.6 mm off the line"),
    (dict(lateral_mm=40.0, lat_dir=(0, 1, 0)), "40.0 mm off the line"),
    (dict(tilt_deg=5.6), "5.6 deg off the paper normal"),
    (dict(tilt_deg=30.0), "go to the pre-hover pose first"),
    (dict(height_mm=21.5), "nothing to descend"),
    (dict(height_mm=20.0), "nothing to descend"),
    (dict(height_mm=3.0), "nothing to descend"),
    (dict(height_mm=90.5), "more than 70 mm to descend"),
    (dict(height_mm=200.0), "go to the pre-hover pose first"),
])
def test_descend_refusals(tmp_path, kw, frag):
    mgr = descend_case(tmp_path, **kw)
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 409 and frag in e.message, e.message
    assert [c for c in mgr.fakes.jog.calls if c[0] == "descend"] == []


def test_descend_limits_are_inclusive_at_the_edges(tmp_path):
    for kw in (dict(lateral_mm=4.9), dict(tilt_deg=4.9), dict(height_mm=22.5), dict(height_mm=89.9)):
        mgr = descend_case(tmp_path, **kw)
        assert mgr.descend("P1", 101)["ok"], kw


def test_descend_height_is_selectable_10_20_30(tmp_path):
    for h in (10, 20, 30, 10.0):
        mgr = descend_case(tmp_path, height_mm=60.0)
        res = mgr.descend("P1", 101, h)
        assert res["ok"] and f"to {float(h):g} mm above TOP" in res["message"], (h, res)
        T = [c for c in mgr.fakes.jog.calls if c[0] == "descend"][0][3]
        assert T[2, 3] - mgr.fakes.jog.T_tcp[2, 3] == pytest.approx(-(0.060 - h * 1e-3))
    mgr = descend_case(tmp_path, height_mm=60.0)
    for bad in (0, 5, 15, 40, "20", None, True, float("nan")):
        e = refusal(mgr.descend, "P1", 101, bad)
        assert e.status == 400 and "height_mm must be one of [10, 20, 30]" in e.message, bad
    assert [c for c in mgr.fakes.jog.calls if c[0] == "descend"] == []
    st = mgr.state()
    assert st["heights_mm"] == {"prehover": 60.0, "descend": [10, 20, 30], "default_descend": 20}
    # at 30 mm a TCP only 25 mm above the mark has nothing to descend; at 10 mm it has
    mgr = descend_case(tmp_path, height_mm=25.0)
    assert "nothing to descend" in refusal(mgr.descend, "P1", 101, 30).message
    assert mgr.descend("P1", 101, 10)["ok"]


def test_descend_other_refusals(tmp_path):
    mgr = descend_case(tmp_path)
    mgr.fakes.jog.busy = True
    assert "busy" in refusal(mgr.descend, "P1", 101).message
    mgr.fakes.jog.busy = False
    mgr.fakes.jog.T_tcp, mgr.fakes.jog.tcp_why = None, "the arm is moving (teleop active?): release the keys and wait"
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 409 and "teleop active?" in e.message
    mgr.fakes.jog.T_tcp, mgr.fakes.jog.tcp_why = tcp_pose(), ""
    assert refusal(mgr.descend, "P1", 113).status == 409                    # not detected
    assert refusal(mgr.descend, "P1", None).status == 400
    mgr.fakes.jog.descend_answer = {"ok": False, "id": 3, "message": "busy: a step on arm 'left' is still running"}
    res = mgr.descend("P1", 101)
    assert res["ok"] is False and "busy" in res["message"]


def test_descend_uses_the_prediction_of_that_point(tmp_path):
    mgr = descend_case(tmp_path)                                            # above point 101; point 102 is 10 mm to the +x of it
    e = refusal(mgr.descend, "P1", 102)
    assert "above point 101, not 102" in e.message and "go to the pre-hover pose first" in e.message
    assert mgr.descend("P1", 101)["ok"]


# ---------------------------------------------------------------------------------------------------
# offsets: touches and hover test

def touch(arm="left", pid=101, p=None):
    T = np.eye(4)
    T[:3, 3] = P_BASE + np.array([0.01, 0.0, 0.0]) + (np.zeros(3) if p is None else p)
    return SimpleNamespace(arm=arm, point_id=pid, T_base_tcp=T)


def read_lines(tmp_path, name):
    with open(os.path.join(str(tmp_path), name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_after_touch_logs_exact_fields_and_returns_the_offset(tmp_path):
    mgr = detected(tmp_path)
    assert mgr.touch_count() == 0
    mgr.touches.append(touch("left", 1))                                    # an older touch of another point
    n0 = mgr.touch_count()
    assert n0 == 1
    mgr.touches.append(touch("left", 101, p=np.array([0.0019, -0.0012, 0.0001])))
    off = mgr.after_touch("left", 101, n0)
    assert off["dx_mm"] == pytest.approx(1.9, abs=1e-6) and off["dy_mm"] == pytest.approx(-1.2, abs=1e-6)
    assert off["dn_mm"] == pytest.approx(0.1, abs=1e-6) and off["lateral_mm"] == pytest.approx(math.hypot(1.9, 1.2))
    assert off["text"].startswith("predicted vs touched: 2.2 mm lateral (x +1.9, y -1.2, above paper +0.1")
    lines = read_lines(tmp_path, "hover_log.jsonl")
    assert len(lines) == 1
    rec = lines[0]
    assert set(rec) == {"time", "time_iso", "kind", "arm", "point_id", "name", "source", "predicted_p_base_m",
                        "predicted_raw_p_base_m", "correction_mm", "touched_p_base_m", "offset_mm", "calibration", "quality", "prediction_age_s"}
    assert rec["time"] == 1.7e9 and rec["time_iso"] == "2023-11-14T22:13:20Z" and rec["kind"] == "touch"
    assert (rec["arm"], rec["point_id"], rec["name"], rec["source"]) == ("left", 101, "TOP", "sheet")
    assert rec["calibration"] == {"n_samples": 8, "arm": "left"}
    assert rec["predicted_p_base_m"] == pytest.approx(list(P_BASE + [0.01, 0, 0]))
    assert rec["touched_p_base_m"] == pytest.approx(list(P_BASE + [0.01 + 0.0019, -0.0012, 0.0001]))
    assert set(rec["offset_mm"]) == {"dx_mm", "dy_mm", "dn_mm", "lateral_mm", "dist_mm"}
    assert rec["quality"]["n_frames"] == hu.N_FRAMES and rec["prediction_age_s"] == 0.0
    st = mgr.state()
    last = {p["id"]: p for p in st["points"]}[101]["last_offset"]
    assert last["kind"] == "touch" and last["dx_mm"] == pytest.approx(1.9, abs=1e-6) and "above paper" in last["text"]


def test_after_touch_without_prediction_or_touch_does_nothing(tmp_path):
    mgr = detected(tmp_path)
    assert mgr.after_touch("left", 113, 0) is None                          # no prediction for 113
    assert mgr.after_touch("left", 101, 0) is None                          # no touch record at all
    mgr.touches.append(touch("left", 102))
    assert mgr.after_touch("left", 101, 0) is None                          # the new touch is another point
    assert mgr.after_touch("right", 101, 0) is None
    assert not os.path.exists(os.path.join(str(tmp_path), "hover_log.jsonl"))
    fresh = make(tmp_path)
    fresh.touches.append(touch("left", 101))
    assert fresh.after_touch("left", 101, 0) is None                        # never detected: no comparison
    broken = detected(tmp_path)
    broken.touches.append(touch("left", 101))
    broken._load_touches = lambda d: (_ for _ in ()).throw(OSError("disk"))
    assert broken.after_touch("left", 101, 0) is None                       # never raises into the capture reply
    assert broken.touch_count() == 0


def test_record_offset_logs_hover_test_and_keeps_a_summary(tmp_path):
    mgr = detected(tmp_path)
    j = mgr.fakes.jog
    j.T_tcp = np.eye(4)
    j.T_tcp[:3, 3] = P_BASE + [0.01 + 0.001, 0.002, 0.0005]                 # point 101: dx +1, dy +2, dn +0.5
    r1 = mgr.record_offset(101)
    assert r1["ok"] and r1["offset"]["dx_mm"] == pytest.approx(1.0) and r1["offset"]["dy_mm"] == pytest.approx(2.0)
    assert "TOP (101)" in r1["message"] and "hover_test.jsonl" in r1["message"]
    assert ("tcp_pose_base", "left", True) in j.calls
    j.T_tcp[:3, 3] = P_BASE + [0.01 + 0.003, 0.004, 0.0]
    r2 = mgr.record_offset(101)
    j.T_tcp[:3, 3] = P_BASE + [0.02, 0.0, 0.0]                              # exactly on point 102 (the fake puts it at +20 mm in x)
    mgr.record_offset(102)
    lines = read_lines(tmp_path, "hover_test.jsonl")
    assert len(lines) == 3 and all(rec["kind"] == "hover_test" for rec in lines)
    assert [rec["point_id"] for rec in lines] == [101, 101, 102]
    assert not os.path.exists(os.path.join(str(tmp_path), "hover_log.jsonl"))
    s = mgr.state()["test_summary"]
    assert s["n"] == 3
    rows = {r["point_id"]: r for r in s["points"]}
    assert rows[101]["n"] == 2 and rows[101]["name"] == "TOP"
    assert rows[101]["max_lateral_mm"] == pytest.approx(5.0) and rows[101]["mean_lateral_mm"] == pytest.approx((math.sqrt(5) + 5.0) / 2)
    assert rows[101]["last"]["dx_mm"] == pytest.approx(3.0)
    lat = [rec["offset_mm"]["lateral_mm"] for rec in lines]
    assert s["max_lateral_mm"] == pytest.approx(max(lat)) and s["mean_lateral_mm"] == pytest.approx(sum(lat) / 3)
    assert {p["id"]: p for p in mgr.state()["points"]}[101]["last_offset"]["kind"] == "hover_test"
    # the summary survives a restart of the UI: it is read back from the session
    again = make(tmp_path)
    assert again.state()["test_summary"]["n"] == 3
    again.compute()
    assert {p["id"]: p for p in again.state()["points"]}[101]["last_offset"]["dx_mm"] == pytest.approx(3.0)


def test_record_offset_refusals(tmp_path):
    mgr = detected(tmp_path)
    j = mgr.fakes.jog
    j.T_tcp, j.tcp_why = None, "the arm is moving (teleop active?): release the keys and wait"
    assert "teleop active?" in refusal(mgr.record_offset, 101).message
    j.T_tcp, j.tcp_why = np.eye(4), ""
    j.busy = True
    assert "busy" in refusal(mgr.record_offset, 101).message
    j.busy = False
    assert refusal(mgr.record_offset, 113).status == 409
    assert refusal(mgr.record_offset, "101").status == 400
    mgr.session_dir_fn = lambda: None
    assert refusal(mgr.record_offset, 101).status == 503
    assert mgr.state()["test_summary"]["n"] == 0


# =====================================================================================================
# safety review round (test_h2_*)
# =====================================================================================================

def descended(mgr):
    return [c for c in mgr.fakes.jog.calls if c[0] == "descend"]


# ---- 1. REAL hardware needs allow_real_move for a descend ------------------------------------------

def test_h2_descend_on_real_hardware_without_allow_real_move_is_refused_before_anything_else(tmp_path):
    mgr = descend_case(tmp_path)
    j = mgr.fakes.jog
    j.hardware = {"left": {"real": True}}
    j.allow_real_move = False
    n_calls = len(mgr.fakes.hover.calls)
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 409 and e.message == REAL_MOVE_DISABLED_MSG
    assert descended(mgr) == [] and len(mgr.fakes.hover.calls) == n_calls          # no camera work, no TF, no motion
    j.allow_real_move = True                                                        # switch on: the other gates decide
    assert mgr.descend("P1", 101)["ok"]
    j.hardware = {"left": {"real": False}}                                          # mock: never affected
    j.allow_real_move = False
    mgr2 = descend_case(tmp_path)
    mgr2.fakes.jog.allow_real_move = False
    assert mgr2.descend("P1", 101)["ok"]


def test_h2_state_tells_the_page_whether_a_descend_is_possible(tmp_path):
    mgr = detected(tmp_path)
    st = mgr.state()
    assert st["reached_point"] is None and "pre-hover pose first" in st["interlock"]
    mgr.prepare("P1", 101)
    assert mgr.state()["reached_point"] is None
    mgr.move("P1", "plan-1")
    st = mgr.state()
    assert st["reached_point"] == 101 and st["interlock"] == ""
    json.dumps(st, allow_nan=False)


# ---- 2. pre-hover interlock ---------------------------------------------------------------------------

NEED_PREHOVER = "go to the pre-hover pose first (Plan hover -> Move)"


def test_h2_descend_needs_a_finished_hover_move_to_this_point(tmp_path):
    mgr = detected(tmp_path)
    mgr.fakes.jog.T_tcp = tcp_pose()
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message                  # nothing was moved
    mgr.prepare("P1", 101)
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message                  # planned only
    mgr.move("P1", "plan-1")
    mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()                              # exactly at the pre-hover pose
    e = refusal(mgr.descend, "P1", 102)
    assert "above point 101, not 102" in e.message and NEED_PREHOVER in e.message
    assert mgr.state()["reached_point"] == 101                                       # another point's refusal keeps it
    assert descended(mgr) == []
    assert mgr.descend("P1", 101)["ok"]


def test_h2_the_tcp_must_be_within_2_mm_and_2_deg_of_the_plans_prehover_pose(tmp_path):
    mgr = detected(tmp_path)
    mgr.prepare("P1", 101)
    mgr.move("P1", "plan-1")
    T_pre = mgr._pending_move["T"].copy()
    # the plan's pre-hover pose is 60 mm above the mark, 3 mm out would still pass lateral (5 mm) but not the interlock
    for dx, dy, dz, tilt, frag in ((0.0031, 0, 0, 0, "3.1 mm"), (0, 0.0025, 0, 0, "2.5 mm"), (0, 0, 0.0021, 0, "2.1 mm"),
                                   (0, 0, 0, 2.5, "2.5 deg")):
        T = T_pre.copy()
        T[:3, 3] += [dx, dy, dz]
        a = math.radians(tilt)
        T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        mgr.fakes.jog.T_tcp = T
        e = refusal(mgr.descend, "P1", 101)
        assert e.status == 409 and frag in e.message and "from the pre-hover pose that was moved to" in e.message and NEED_PREHOVER in e.message
    assert descended(mgr) == []
    T = T_pre.copy()
    T[:3, 3] += [0.0012, 0.0012, 0.0]                                                 # 1.7 mm: within the limit
    a = math.radians(1.5)
    T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    mgr.fakes.jog.T_tcp = T
    assert mgr.descend("P1", 101)["ok"]


def test_h2_the_interlock_is_cleared_by_any_other_motion_stop_fault_failure_detection_or_plan(tmp_path):
    def reached():
        mgr = detected(tmp_path)
        mgr.prepare("P1", 101)
        mgr.move("P1", "plan-1")
        mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()
        return mgr, mgr.fakes.jog

    # another motion (a jog step, a descend ...): the jog's last id is no longer the hover move
    mgr, j = reached()
    j.last = {"id": 6, "ok": True, "phase": "done", "kind": None}
    e = refusal(mgr.descend, "P1", 101)
    assert "another motion happened" in e.message and NEED_PREHOVER in e.message
    j.last = {"id": 5, "ok": True, "phase": "done", "kind": "move"}                    # it stays cleared: the id matching again is not enough
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message
    assert mgr.state()["reached_point"] is None
    # a STOP after the move started
    mgr, j = reached()
    j.core.stop_epoch += 1
    assert "STOP was pressed since the hover move" in refusal(mgr.descend, "P1", 101).message
    # a fault
    mgr, j = reached()
    j.fault = True
    assert "in fault" in refusal(mgr.descend, "P1", 101).message
    # the move did not finish well
    for phase in ("stopped", "failed", "refused", "fault"):
        mgr, j = reached()
        j.last = {"id": 5, "ok": False, "phase": phase, "kind": "move"}
        assert f"ended as '{phase}'" in refusal(mgr.descend, "P1", 101).message
    # still running: refused but NOT cleared
    mgr, j = reached()
    j.last = {"id": 5, "ok": True, "phase": "executing", "kind": "move"}
    assert "still running" in refusal(mgr.descend, "P1", 101).message
    j.last = {"id": 5, "ok": True, "phase": "done", "kind": "move"}
    assert mgr.descend("P1", 101)["ok"]
    # a new detection
    mgr, j = reached()
    mgr.compute()
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message
    # a new plan (even for the same point)
    mgr, j = reached()
    mgr.prepare("P1", 101)
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message
    # the jog gave no id for the move: fail closed
    mgr, j = reached()
    mgr._pending_move["id"] = None
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101).message


def test_h2_a_finished_descend_is_not_a_hover_move_so_the_next_descend_needs_a_new_plan(tmp_path):
    mgr = descend_case(tmp_path, height_mm=60.0)
    assert mgr.descend("P1", 101)["ok"]
    mgr.fakes.jog.last = {"id": 9, "ok": True, "phase": "done", "kind": "descend"}
    assert NEED_PREHOVER in refusal(mgr.descend, "P1", 101, 10).message
    assert len(descended(mgr)) == 1


# ---- 3. fresh observation before prepare / descend ------------------------------------------------

def test_h2_prediction_age_limit_is_120_s():
    assert hu.PRED_MAX_AGE_S == 120.0


def test_h2_prepare_with_the_full_detection_refuses_a_board_that_moved_more_than_2_mm(tmp_path):
    mgr = descend_case(tmp_path)
    ik_before = len(mgr.fakes.moveit.ik_calls)
    mgr.fakes.hover.shift = np.array([0.0, 0.0, 0.0021])
    e = refusal(mgr.prepare, "P1", 101)
    assert e.status == 409 and "the board moved or the detection changed: Detect points again" in e.message
    assert "moved 2.1 mm" in e.message and len(mgr.fakes.moveit.ik_calls) == ik_before        # nothing planned
    assert not any(c[0] == "unmoved" for c in mgr.fakes.hover.calls)                          # a real "moved" has no fallback
    mgr.fakes.hover.shift = np.array([0.0, 0.0, 0.0015])                                       # 1.5 mm is within the limit
    assert mgr.prepare("P1", 101)["ok"]
    mgr.fakes.hover.shift = np.zeros(3)
    mgr.fakes.hover.tilt_deg = 2.5                                                             # the normal turned by 2.5 deg
    e = refusal(mgr.prepare, "P1", 101)
    assert "the board moved or the detection changed" in e.message and "normal 2.5 deg" in e.message
    mgr.fakes.hover.tilt_deg = 1.5
    assert mgr.prepare("P1", 101)["ok"]


def test_h2_prepare_falls_back_to_the_unmoved_test_when_the_full_detection_fails(tmp_path):
    mgr = detected(tmp_path)
    f = mgr.fakes
    del f.hover.detected[101]                                                                   # the arm hides the sheet now
    f.hover.status = {"sheet": "sheet: marker not found", "table": "table: ok"}
    n_unmoved = lambda: len([c for c in f.hover.calls if c[0] == "unmoved"])
    res = mgr.prepare("P1", 101)                                                                # the stored pose is proven unmoved
    assert res["ok"] and n_unmoved() == 1
    poses = [c for c in f.hover.calls if c[0] == "poses"][-1]
    assert np.allclose(poses[1], P_BASE + [0.01, 0, 0])                                         # aimed at the STORED prediction
    f.hover.unmoved_result = (False, "the board is hidden by the arm: cannot confirm it has not moved - use teleop from here, "
                                     "or Detect points with the arm away", {})
    n_ik = len(f.moveit.ik_calls)
    e = refusal(mgr.prepare, "P1", 101)
    assert e.status == 409 and e.message.startswith("the board is hidden by the arm") and len(f.moveit.ik_calls) == n_ik


def test_h2_descend_uses_the_partial_occlusion_test_and_the_stored_prediction(tmp_path):
    mgr = descend_case(tmp_path, height_mm=60.0)
    f = mgr.fakes
    n_detect = len([c for c in f.hover.calls if c[0] == "detect"])
    f.hover.shift = np.array([0.0, 0.0, 0.0015])                  # a full re-detection would now say "1.5 mm higher": not used
    f.hover.detected = {}                                         # ... and it would not even find the sheet any more
    assert mgr.descend("P1", 101)["ok"]
    assert len([c for c in f.hover.calls if c[0] == "detect"]) == n_detect         # NO full re-detection before a descend
    kind, n_frames, stored, table, sheet, det, sources = [c for c in f.hover.calls if c[0] == "unmoved"][-1]
    assert n_frames >= 3 and sources == ["sheet"] and set(stored) == {101, 102}  # the stored sheet points, nothing else
    assert all(tp.source == "sheet" for tp in stored.values())
    assert table is mgr.scene.table_spec and sheet is mgr.scene.sheet_spec and det is mgr.scene.detector
    T = descended(mgr)[0][3]
    assert T[2, 3] == pytest.approx(P_BASE[2] + 0.020)                                     # 20 mm above the STORED surface
    assert mgr._preds[101]["_p"][2] == pytest.approx(P_BASE[2])                           # the stored prediction is untouched


def test_h2_descend_for_a_table_point_checks_only_the_table(tmp_path):
    mgr = detected(tmp_path)
    mgr.prepare("P1", 1)
    mgr.move("P1", "plan-1")
    mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()
    mgr.fakes.jog.T_tcp[:3, 3] = mgr.fakes.jog.T_tcp[:3, 3]                                  # (already at the pre-hover pose)
    assert mgr.descend("P1", 1, 30)["ok"]
    kind, n_frames, stored, table, sheet, det, sources = [c for c in mgr.fakes.hover.calls if c[0] == "unmoved"][-1]
    assert sources == ["table"] and set(stored) == {1} and stored[1].source == "table"


@pytest.mark.parametrize("answer,starts", [
    ((False, "the board moved or the detection changed: Detect points again (table: reprojection of 20 visible corners: median 3.10 px)", {}),
     "the board moved or the detection changed: Detect points again"),
    ((False, "the board is hidden by the arm: cannot confirm it has not moved - use teleop from here, or Detect points with the arm away "
             "(table: only 3 corners visible (< 8))", {}), "the board is hidden by the arm: cannot confirm it has not moved"),
])
def test_h2_a_failed_unmoved_test_refuses_the_descend_with_its_reason(tmp_path, answer, starts):
    mgr = descend_case(tmp_path)
    mgr.fakes.hover.unmoved_result = answer
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 409 and e.message.startswith(starts) and answer[1] in e.message
    assert descended(mgr) == [] and mgr._pending_move is not None                              # nothing sent; the interlock stays
    mgr.fakes.hover.unmoved_result = (True, "", {})
    assert mgr.descend("P1", 101)["ok"]


def test_h2_an_unmoved_test_that_crashes_refuses_with_500(tmp_path):
    mgr = descend_case(tmp_path)

    def boom(*a, **k):
        raise ValueError("boom")
    mgr.fakes.hover.unmoved = boom
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 500 and "cannot confirm the board has not moved" in e.message and "boom" in e.message
    assert descended(mgr) == []


def test_h2_the_unmoved_test_needs_fresh_frames(tmp_path):
    mgr = descend_case(tmp_path)
    mgr.image_fn = Frames(n=2)                                                                  # only two frames: below the gate
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 503 and "fresh camera frame" in e.message
    assert not any(c[0] == "unmoved" for c in mgr.fakes.hover.calls) and descended(mgr) == []


def test_h2_only_fresh_camera_frames_count(tmp_path):
    mgr = descend_case(tmp_path)
    mgr.image_fn = lambda: (np.zeros((480, 640, 3), np.uint8), 99, -50.0)               # received 50 s ago: the camera stalled
    e = refusal(mgr.descend, "P1", 101)
    assert e.status == 503 and "fresh camera frame" in e.message
    assert descended(mgr) == []
    mgr = detected(tmp_path)
    mgr.image_fn = lambda: (np.zeros((480, 640, 3), np.uint8), 99, -50.0)
    assert "fresh camera frame" in refusal(mgr.prepare, "P1", 101).message
    assert mgr.fakes.moveit.ik_calls == []


def test_h2_an_old_prediction_moves_only_after_the_unmoved_proof(tmp_path):
    mgr = descend_case(tmp_path)
    # age alone does not refuse motion (a 60 s move to the pre-hover pose would expire every prediction); the proof that
    # the paper has not moved is the fresh observation: full detection for prepare, the unmoved test for descend
    mgr.clock_obj.t += 121.0
    assert mgr.record_offset(101)["ok"]
    mgr.prepare("P1", 101)
    mgr.move("P1", "plan-1")
    mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()
    mgr.clock_obj.t += 300.0
    mgr.fakes.hover.unmoved_result = (False, "the board moved or the detection changed: Detect points again (x)", {})
    assert "the board moved or the detection changed" in refusal(mgr.descend, "P1", 101).message
    mgr.fakes.hover.unmoved_result = (True, "", {})
    assert mgr.descend("P1", 101)["ok"]


# ---- 4. clearance ------------------------------------------------------------------------------------

def test_h2_the_final_hover_height_never_goes_below_10_mm(tmp_path, monkeypatch):
    assert hu.MIN_CLEARANCE_M == 0.010 and min(hu.DESCEND_HEIGHTS_MM) == 10 and hu.DEFAULT_DESCEND_MM == 20
    mgr = descend_case(tmp_path)
    for bad in (0, 1, 5, 9.9):
        assert refusal(mgr.descend, "P1", 101, bad).status == 400
    monkeypatch.setattr(hu, "DESCEND_HEIGHTS_MM", (5, 10, 20, 30))                       # even if a smaller choice sneaks in
    e = refusal(mgr.descend, "P1", 101, 5)
    assert e.status == 400 and "at least 10 mm" in e.message
    assert descended(mgr) == []
    for h in (10, 20, 30):
        T = descended_target(tmp_path, h)
        assert (T[2, 3] - P_BASE[2]) >= 0.010 - 1e-9 and T[2, 3] - P_BASE[2] == pytest.approx(h * 1e-3)   # always >= 10 mm above the surface


def descended_target(tmp_path, h):
    mgr = descend_case(tmp_path, height_mm=60.0)
    assert mgr.descend("P1", 101, h)["ok"]
    return descended(mgr)[0][3]


def test_h2_clearance_minimum_is_checked_against_the_requested_height(tmp_path, monkeypatch):
    mgr = descend_case(tmp_path, height_mm=60.0)
    monkeypatch.setattr(hu, "MIN_CLEARANCE_M", 0.012)                                    # a stricter minimum than the choice
    e = refusal(mgr.descend, "P1", 101, 10)
    assert e.status == 400 and "at least 12 mm" in e.message and descended(mgr) == []
    assert mgr.descend("P1", 101, 20)["ok"]


# ---- 6. wording / state fixes from the browser test -------------------------------------------------

def test_h2_status_texts_are_not_prefixed_twice(tmp_path):
    assert hu.status_text({"sheet": "sheet: marker 13 seen", "table": "48 corners"}) == "sheet: marker 13 seen; table: 48 corners"
    assert hu.status_text({"table": "Table: ok"}) == "Table: ok"
    assert hu.status_text({}) == ""
    mgr = make(tmp_path)
    mgr.fakes.hover.status = {"sheet": "sheet: marker 13 seen", "table": "table: 48 corners, 0.25 px"}
    msg = mgr.compute()
    assert "sheet: sheet:" not in msg and "table: table:" not in msg and "[sheet: marker 13 seen; table: 48 corners, 0.25 px]" in msg
    mgr.fakes.hover.detected = {}
    e = refusal(mgr.compute)
    assert "sheet: sheet:" not in e.message and "no point detected: sheet: marker 13 seen; table: 48 corners" in e.message


def test_h2_a_failed_detect_gives_every_row_the_failure_text_and_drops_the_old_predictions(tmp_path):
    mgr = make(tmp_path)
    mgr.fakes.hover.status = {"sheet": "sheet: ok", "table": "table: 48 corners, 0.25 px, 5 of 5 frames"}
    mgr.compute()
    mgr.prepare("P1", 101)
    mgr.move("P1", "plan-1")
    assert mgr.state()["reached_point"] == 101
    mgr.image_fn = Frames(n=2)                                                        # the camera stops after two frames
    e = refusal(mgr.compute)
    assert "fresh camera frame" in e.message
    st = mgr.state()
    assert st["state"] == "error" and st["points"] and st["prepared"] is None and st["plan"] is None
    assert st["reached_point"] is None and st["descended"] is None and mgr.overlay() is None
    for p in st["points"]:                                                            # ALL rows: the camera text, nothing stale
        assert p["detected"] is False and p["pred_base_mm"] is None and p["uv"] is None
        assert p["reason"] == e.message and "48 corners" not in p["reason"]
    assert {p["id"] for p in st["points"]} == {1, 2, 101, 102, 113}                   # the rows exist even though nothing was seen
    assert st["status"] == {}
    assert refusal(mgr.prepare, "P1", 101).status == 409                              # and nothing can be planned from the old data


def test_h2_rows_show_only_results_of_the_current_detection(tmp_path):
    import os
    line = {"time": 1.6e9, "kind": "touch", "point_id": 101, "offset_mm": {"dx_mm": 1.0, "dy_mm": 0.0, "dn_mm": 0.0,
                                                                              "lateral_mm": 1.0, "dist_mm": 1.0}}
    with open(os.path.join(str(tmp_path), "hover_log.jsonl"), "w") as f:
        f.write(json.dumps(line) + "\n")
    with open(os.path.join(str(tmp_path), "hover_test.jsonl"), "w") as f:
        f.write(json.dumps(dict(line, kind="hover_test", point_id=1)) + "\n")
    mgr = detected(tmp_path)
    st = mgr.state()
    assert all(p["last_offset"] is None for p in st["points"])                       # older sessions' results: not in the rows
    assert st["test_summary"]["n"] == 1                                              # (they stay in the table)
    mgr.fakes.jog.T_tcp = np.eye(4)
    mgr.record_offset(101)
    assert {p["id"]: p for p in mgr.state()["points"]}[101]["last_offset"]["kind"] == "hover_test"
    mgr.compute()                                                                    # a new detection starts clean again ...
    mgr._computed_wall += 1.0
    assert all(p["last_offset"] is None for p in mgr.state()["points"])
    assert mgr.state()["test_summary"]["n"] == 2


def test_h2_state_follows_the_hover_move_and_the_descend(tmp_path):
    mgr = detected(tmp_path)
    mgr.prepare("P1", 101)
    st = mgr.state()
    assert st["prepared"]["point_id"] == 101 and st["reached_point"] is None and st["descended"] is None
    mgr.move("P1", "plan-1")
    st = mgr.state()
    assert st["prepared"] is not None and st["reached_point"] == 101 and st["plan"] is None       # "at the pre-hover pose"
    mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()
    assert mgr.descend("P1", 101, 30)["ok"]
    mgr.fakes.jog.last = {"id": 9, "ok": True, "phase": "executing", "kind": "descend"}
    mgr.fakes.jog.descend_answer = {"ok": True, "id": 9, "message": "accepted"}
    st = mgr.state()
    assert st["prepared"] is None and st["plan"] is None and st["reached_point"] is None          # the old "Prepared" line is gone
    assert st["descended"] == {"id": 9, "point_id": 101, "height_mm": 30.0, "phase": "executing"}
    mgr2 = detected(tmp_path)
    mgr2.prepare("P1", 101)
    mgr2.move("P1", "plan-1")
    mgr2.fakes.jog.T_tcp = mgr2._pending_move["T"].copy()
    mgr2.fakes.jog.descend_answer = {"ok": True, "id": 9, "message": "accepted"}
    mgr2.descend("P1", 101, 30)
    mgr2.fakes.jog.last = {"id": 9, "ok": True, "phase": "executing", "kind": "descend"}
    assert mgr2.state()["descended"] == {"id": 9, "point_id": 101, "height_mm": 30.0, "phase": "executing"}
    mgr2.fakes.jog.last = {"id": 9, "ok": True, "phase": "done", "kind": "descend"}
    assert mgr2.state()["descended"]["phase"] == "done"
    mgr2.fakes.jog.last = {"id": 10, "ok": True, "phase": "done", "kind": None}
    assert mgr2.state()["descended"] is None                                         # any later motion ends "at hover height"


def test_h2_end_to_end_with_the_real_geometry_the_arm_hides_half_of_the_board(monkeypatch):
    """HoverManager + the real hover.py on synthetic renders: compute on a clear view, then the descend with half of the table
    board (and the whole sheet) hidden by a blank rectangle: confirmed unmoved from the visible corners; a 2 mm shift is refused."""
    import pathlib
    import tempfile

    import test_hover as thv
    from rebel_dual_arm_calibration import hover as real_hover
    from rebel_dual_arm_calibration.targets import TargetDetector

    sc = thv.Scene(thv.D0)
    Tt, Tm = thv.table_pose(), thv.sheet_pose()
    clear = [(sc.render(T_table=Tt, T_marker=Tm), thv.K, sc.dist) for _ in range(5)]
    pix = thv._table_corner_pix(sc, clear[0][0])
    lo, hi = pix[:, 0].min(), pix[:, 0].max()
    hide = lambda f: thv._blank(f, int(lo + 0.5 * (hi - lo)), 0, thv.W, thv.HGT)       # the arm: half of the board (24 of 48 corners stay) + the sheet
    Tshift = Tt.copy()
    Tshift[:3, 3] += Tt[:3, :3] @ np.array([0.002, 0.0, 0.0])
    occluded = [(hide(sc.render(T_table=Tt, T_marker=None)), thv.K, sc.dist) for _ in range(5)]
    shifted = [(hide(sc.render(T_table=Tshift, T_marker=None)), thv.K, sc.dist) for _ in range(5)]
    source = {"frames": clear, "seq": 0}

    def image_fn():
        source["seq"] += 1
        return source["frames"][source["seq"] % 5][0], source["seq"], 0.0

    moveit = FakeMoveIt()
    jog = FakeJog(moveit)
    fit = SimpleNamespace(arm="left", X=np.eye(4), K=thv.K, dist=sc.dist, image_size=(thv.W, thv.HGT))
    scene = SimpleNamespace(table_spec=sc.table, sheet_spec=thv.SHEET_SPEC, detector=TargetDetector({"table": sc.table}))
    tmp = pathlib.Path(tempfile.mkdtemp())
    mgr = hu.HoverManager("single_left", CFGS, jog, lambda: str(tmp), moveit=moveit, hover=real_hover,
                          fit_engine=SimpleNamespace(fit_arm=lambda samples, arm: fit),
                          load_samples=lambda d: samples(8), load_touches=lambda d: [], image_fn=image_fn,
                          scene_fn=lambda: scene, clock=Clock(), wall_clock=lambda: 1.7e9, sleep=lambda s: None,
                          mono=lambda: 0.0)
    jog.core = SimpleNamespace(stop_epoch=3)
    mgr.compute()
    st = mgr.state()
    assert {p["id"] for p in st["points"] if p["detected"]} == {1, 2, 3, 4} | set(thv.SHEET_SPEC.points)
    # reach the pre-hover pose of table point 2 (the plan is made on the clear view)
    assert mgr.prepare("P1", 2)["ok"]
    assert mgr.move("P1", "plan-1")["ok"]
    jog.T_tcp = mgr._pending_move["T"].copy()
    # at the pre-hover pose: half of the board and the whole sheet are hidden; a FULL detection (>= 30 corners here; with a
    # real arm the gate also fails on the spread / reprojection of a partly hidden board) fails ...
    monkeypatch.setattr(real_hover, "TABLE_MIN_CORNERS", 30)
    source["frames"] = occluded
    targets, status = real_hover.detect_targets(occluded[:3], sc.table, thv.SHEET_SPEC, scene.detector)
    assert 2 not in targets and "table" in status
    # ... the descend still goes: the visible corners sit where the stored pose puts them; the target is the STORED point
    stored_z = mgr._preds[2]["_p"].copy()
    assert mgr.descend("P1", 2, 20)["ok"]
    T = descended_of(jog)[0][3]
    n = mgr._preds[2]["_n"]
    assert (T[:3, 3] - stored_z) @ n == pytest.approx(0.020, abs=1e-9)
    # the paper moved 2 mm while the arm hid half of it: refused, nothing is sent
    mgr2_pending = dict(mgr._pending_move)
    jog.last = {"id": 5, "ok": True, "phase": "done", "kind": "move"}
    jog.calls.clear()
    mgr._pending_move = mgr2_pending
    source["frames"] = shifted
    e = refusal(mgr.descend, "P1", 2)
    assert e.status == 409 and e.message.startswith("the board moved or the detection changed: Detect points again")
    assert descended_of(jog) == []
    # the sheet point cannot be confirmed at all while the sheet is hidden
    mgr._preds[101] = dict(mgr._preds[101])
    mgr._pending_move["point_id"] = 101
    source["frames"] = occluded
    e = refusal(mgr.descend, "P1", 101)
    assert e.message.startswith("the board is hidden by the arm: cannot confirm it has not moved") and "teleop" in e.message


def descended_of(jog):
    return [c for c in jog.calls if c[0] == "descend"]


# ---------------------------------------------------------------------------------------------------
# hover TILES: the catalog, the rows and the statuses come from the configured tiles

def tile_spec(label, marker_id, ids):
    return SimpleNamespace(label=label, source=f"tile {label}", marker_id=marker_id, points={i: (0.01 * k, 0.02) for k, i in enumerate(ids)},
                           names={i: f"{label}{k + 1}" for k, i in enumerate(ids)})


def tiles_mgr(tmp_path, **kw):
    mgr = make(tmp_path, **kw)
    mgr.scene.sheet_spec = [tile_spec("A", 13, [101, 102, 103]), tile_spec("B", 14, [104, 105, 106])]
    h = mgr.fakes.hover
    h.detected = {101: "A1", 102: "A2", 103: "A3", 1: "board 1"}
    h.sources = {101: "tile A", 102: "tile A", 103: "tile A", 104: "tile B", 105: "tile B", 106: "tile B", 1: "table"}
    h.status = {"table": "table: 48 corners", "tile A": "tile A: marker 13, 0.30 px, 5 of 5 frames, spread 0.10 mm",
                "tile B": "tile B: marker 14 not found (0 of 5 frames)"}
    return mgr


def test_tiles_catalog_lists_every_tile_with_its_source(tmp_path):
    mgr = tiles_mgr(tmp_path)
    assert mgr._catalog_of(mgr.scene) == [(1, "board 1", "table"), (2, "board 2", "table"),
                                          (101, "A1", "tile A"), (102, "A2", "tile A"), (103, "A3", "tile A"),
                                          (104, "B1", "tile B"), (105, "B2", "tile B"), (106, "B3", "tile B")]
    assert [c[2] for c in mgr.catalog_points()] == ["table", "table"] + ["tile A"] * 3 + ["tile B"] * 3
    # the other spellings of the scene's sheet_spec: one spec, None, an object with .tiles
    mgr.scene.sheet_spec = tile_spec("A", 13, [101])
    assert [c[2] for c in mgr._catalog_of(mgr.scene)] == ["table", "table", "tile A"]
    mgr.scene.sheet_spec = SimpleNamespace(tiles=[tile_spec("C", 15, [111, 112])])
    assert [c[:2] for c in mgr._catalog_of(mgr.scene)][-2:] == [(111, "C1"), (112, "C2")]
    mgr.scene.sheet_spec = None
    assert [c[2] for c in mgr._catalog_of(mgr.scene)] == ["table", "table"]
    # the legacy single sheet (no 'source' attribute / label) keeps the source "sheet"
    mgr.scene.sheet_spec = SimpleNamespace(points={101: (0, 0.06)}, names={101: "TOP"})
    assert mgr._catalog_of(mgr.scene)[-1] == (101, "TOP", "sheet")


def test_tiles_compute_rows_and_per_tile_statuses(tmp_path):
    mgr = tiles_mgr(tmp_path)
    msg = mgr.compute()
    assert "4 point(s) detected" in msg and "tile B: marker 14 not found (0 of 5 frames)" in msg and "tile tile" not in msg
    f = mgr.fakes
    kind, n_frames, table, sheets, det = f.hover.calls[0]
    assert sheets is mgr.scene.sheet_spec and len(sheets) == 2                  # the whole list goes to the detection
    st = mgr.state()
    by = {p["id"]: p for p in st["points"]}
    assert [p["id"] for p in st["points"]] == [1, 2, 101, 102, 103, 104, 105, 106]
    assert by[101]["source"] == "tile A" and by[101]["name"] == "A1" and by[101]["detected"]
    assert by[104]["source"] == "tile B" and not by[104]["detected"]
    assert by[104]["reason"] == "tile B: marker 14 not found (0 of 5 frames)"   # the status of ITS tile
    assert by[105]["reason"] == by[106]["reason"] == by[104]["reason"]
    assert by[2]["reason"] == "table: 48 corners"
    assert st["status"]["tile A"].startswith("tile A: marker 13") and set(st["status"]) == {"table", "tile A", "tile B"}
    ov = mgr.overlay()
    assert {p["id"] for p in ov["points"]} == {1, 101, 102, 103} and {p["name"] for p in ov["points"]} >= {"A1", "A3"}
    json.dumps(st, allow_nan=False)
    # a point of the tile that was not seen refuses with that tile's status
    e = refusal(mgr.prepare, "P1", 105)
    assert e.status == 409 and "tile B: marker 14 not found" in e.message
    # nothing seen: every row names its own tile status in the error
    f.hover.detected = {}
    e = refusal(mgr.compute)
    assert "no point detected" in e.message and "tile B: marker 14 not found" in e.message
    st = mgr.state()
    assert {p["id"]: p["reason"] for p in st["points"]}[104] == "tile B: marker 14 not found (0 of 5 frames)"
    assert {p["id"]: p["reason"] for p in st["points"]}[101].startswith("tile A: marker 13")


def test_tiles_descend_checks_only_its_own_tile(tmp_path):
    mgr = tiles_mgr(tmp_path)
    mgr.compute()
    mgr.prepare("P1", 102)
    mgr.move("P1", "plan-1")
    mgr.fakes.jog.T_tcp = mgr._pending_move["T"].copy()
    assert mgr.descend("P1", 102, 30)["ok"]
    kind, n_frames, stored, table, sheets, det, sources = [c for c in mgr.fakes.hover.calls if c[0] == "unmoved"][-1]
    assert sources == ["tile A"] and set(stored) == {101, 102, 103} and all(tp.source == "tile A" for tp in stored.values())
    assert sheets is mgr.scene.sheet_spec
