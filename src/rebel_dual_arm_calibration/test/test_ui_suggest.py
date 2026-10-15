"""Suggested-pose panel backend (ui/suggest.py): the glue with fake IK / validity / publisher / samples / jog.
No ROS graph, no node: the engine (pose_suggest.py) is replaced by a fake with the documented API."""
import inspect
import json
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rebel_dual_arm_calibration.ui import suggest as sg
from rebel_dual_arm_calibration.ui.jog import arm_configs

CFGS = arm_configs("single_left")
NAMES = list(CFGS["left"].joint_names)
GOOD_Q = {n: 0.1 * (i + 1) for i, n in enumerate(NAMES)}


def make_suggestion(rank, q=None, **kw):
    s = SimpleNamespace(
        rank=rank, arm="left", T_base_link6=np.eye(4), q=dict(GOOD_Q if q is None else q),
        outline_px=[(100.0 * rank, 100.0), (200.0 * rank, 100.0), (200.0 * rank, 200.0), (100.0 * rank, 200.0)],
        centre_px=(150.0 * rank, 150.0), description=f"tilt more about x (#{rank})",
        gains={"info_gain": 0.5 * rank, "rot_spread2_before": 0.1, "rot_spread2_after": 0.3, "empty_cell": rank % 2 == 0,
               "view_angle_deg": 30.0, "distance_m": 0.7, "max_joint_change_deg": np.float64(12.5)},
        score=1.0 / rank)
    s.__dict__.update(kw)
    return s


class FakeEngine:
    def __init__(self, n=5):
        self.n, self.calls, self.gate = n, [], None
        self.fail_suggest = None
        self.stats = {"ik_failed": 7, "invalid": 2}

    def fit_arm(self, samples, arm):
        if len(samples) < 6:
            raise ValueError(f"need at least 6 usable samples for arm '{arm}', got {len(samples)}")
        return SimpleNamespace(arm=arm, n=len(samples))

    def suggest(self, fit, ik, valid, current_q, n=5, options=None):
        self.calls.append((fit, dict(current_q), n))
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail_suggest:
            raise RuntimeError(self.fail_suggest)
        q = ik(np.eye(4), dict(current_q))              # the glue must hand a working IK / validity to the engine
        ok, why = valid(q) if q else (False, "no ik")
        self.ik_result = (q, ok, why)
        return [make_suggestion(i + 1) for i in range(self.n)], dict(self.stats)


class FakeMoveIt:
    def __init__(self):
        self.ready_answer = (True, "")
        self.ik_calls, self.valid_calls, self.published = [], [], []

    def ready(self):
        return self.ready_answer

    def ik(self, cfg, T, seed_q):
        self.ik_calls.append((cfg, np.array(T), dict(seed_q)))
        return dict(GOOD_Q)

    def valid(self, cfg, q):
        self.valid_calls.append((cfg, dict(q)))
        return True, ""

    def publish_goal(self, names, positions):
        self.published.append((list(names), list(positions)))


class FakeJog:
    def __init__(self):
        self.busy = False
        self.q = dict(GOOD_Q)
        self.calls = []
        self.plan_answer = {"ok": True, "plan_id": "p1", "message": "planned", "duration_s": 6.2,
                            "max_joint_change_deg": 12.5, "n_points": 40}
        self.exec_answer = {"ok": True, "message": "moving"}

    def status(self):
        return {"available": True, "busy": self.busy, "move_plan": None}

    def joint_window(self, arms, window_s):
        return (dict(self.q), 0.0, True) if self.q is not None else None

    def plan_move(self, page_id, arm, q_goal):
        self.calls.append(("plan_move", page_id, arm, dict(q_goal)))
        return dict(self.plan_answer)

    def execute_move(self, page_id, plan_id):
        self.calls.append(("execute_move", page_id, plan_id))
        return dict(self.exec_answer)

    def clear_plan(self):
        self.calls.append(("clear_plan",))


def samples(n, arm="left"):
    return [SimpleNamespace(arm=arm) for _ in range(n)]


def make_manager(tmp_path=None, n_samples=8, mode="single_left", cfgs=None, **kw):
    jog, moveit, engine = FakeJog(), FakeMoveIt(), FakeEngine()
    session = str(tmp_path) if tmp_path is not None else "/nonexistent/session"
    mgr = sg.SuggestManager(mode, cfgs or (CFGS if mode != "dual" else arm_configs("dual")), jog, lambda: session,
                            moveit=moveit, engine=engine, load_samples=lambda d: samples(n_samples) + samples(3, "right"),
                            **kw)
    return mgr, jog, moveit, engine


def wait_not_computing(mgr, timeout=5.0):
    t0 = time.time()
    while mgr.state()["state"] == "computing":
        assert time.time() - t0 < timeout, "still computing"
        time.sleep(0.01)
    return mgr.state()


def computed(**kw):
    mgr, jog, moveit, engine = make_manager(**kw)
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["state"] == "ready", st
    return mgr, jog, moveit, engine


# ---------------------------------------------------------------------------------------------------
# compute

def test_compute_success_shape_and_engine_wiring():
    mgr, jog, moveit, engine = make_manager()
    msg = mgr.compute()
    assert "8 samples" in msg                                    # only the left arm's samples count
    st = wait_not_computing(mgr)
    assert st["state"] == "ready" and st["selected"] == 0 and st["arm"] == "left"
    assert st["based_on_samples"] == 8 and st["stats"] == {"ik_failed": 7, "invalid": 2}
    assert len(st["items"]) == 5 and st["supported"] is True
    it = st["items"][1]
    assert set(it) == {"rank", "description", "gains", "centre_px", "outline_px", "q_deg"}
    assert it["rank"] == 2 and len(it["outline_px"]) == 4 and len(it["q_deg"]) == 6
    assert it["q_deg"][0] == pytest.approx(np.degrees(0.1))
    assert it["gains"]["max_joint_change_deg"] == 12.5
    json.dumps(st, allow_nan=False)                              # JSON safe (numpy converted)
    fit, q0, n = engine.calls[0]
    assert fit.n == 8 and q0 == GOOD_Q and n == 5
    # the glue's ik / valid reach the MoveIt side with the arm config and the seed
    cfg, T, seed = moveit.ik_calls[0]
    assert cfg is CFGS["left"] and np.allclose(T, np.eye(4)) and seed == GOOD_Q
    assert moveit.valid_calls[0][0] is CFGS["left"] and engine.ik_result[1] is True
    assert ("clear_plan",) in jog.calls                          # a new search drops any stored plan


def test_fit_error_text_is_the_refusal_and_state():
    mgr, jog, moveit, engine = make_manager(n_samples=4)
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert e.value.status == 400 and "at least 6" in e.value.message
    st = mgr.state()
    assert st["state"] == "error" and "at least 6" in st["message"] and st["items"] == []
    assert engine.calls == []


def test_dual_mode_refused():
    mgr, *_ = make_manager(mode="dual")
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert e.value.status == 400 and "single-arm" in e.value.message
    st = mgr.state()
    assert st["supported"] is False and st["message"] == "single-arm modes only" and st["state"] == "idle"
    with pytest.raises(sg.SuggestRefused):
        mgr.send_rviz(0)


def test_wrong_arm_refused():
    mgr, *_ = make_manager()
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute("right")
    assert e.value.status == 400


@pytest.mark.parametrize("flag", ["busy", "stopping", "fault"])
def test_busy_jog_refused(flag):
    mgr, jog, moveit, engine = make_manager()
    jog.status = lambda: {"available": True, flag: True}
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert e.value.status == 409 and "busy" in e.value.message and engine.calls == []


def test_refused_without_moveit_services_session_or_joints():
    mgr, jog, moveit, engine = make_manager()
    moveit.ready_answer = (False, "/compute_ik is not available (is move_group running?)")
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert e.value.status == 503 and "/compute_ik" in e.value.message
    moveit.ready_answer = (True, "")
    jog.q = None
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert "joint states" in e.value.message
    jog.q = dict(GOOD_Q)
    mgr.session_dir_fn = lambda: None
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert "session directory" in e.value.message
    mgr2 = sg.SuggestManager("single_left", CFGS, jog, lambda: "/x", moveit=None, engine=engine)
    with pytest.raises(sg.SuggestRefused) as e:
        mgr2.compute()
    assert e.value.status == 503


def test_second_compute_while_computing_refused_and_state_is_computing():
    mgr, jog, moveit, engine = make_manager()
    engine.gate = threading.Event()
    mgr.compute()
    st = mgr.state()
    assert st["state"] == "computing" and st["items"] == [] and st["based_on_samples"] == 8
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.compute()
    assert e.value.status == 409 and "already computing" in e.value.message
    with pytest.raises(sg.SuggestRefused):
        mgr.select(0)                                            # nothing to select while computing
    engine.gate.set()
    assert wait_not_computing(mgr)["state"] == "ready"


def test_engine_exception_and_empty_result_end_in_error_not_computing():
    mgr, jog, moveit, engine = make_manager()
    engine.fail_suggest = "boom"
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["state"] == "error" and "boom" in st["message"]
    engine.fail_suggest, engine.n = None, 0
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["state"] == "error" and "no reachable" in st["message"] and "ik_failed: 7" in st["message"]
    assert st["stats"]["ik_failed"] == 7 and st["selected"] is None


def test_malformed_suggestions_are_dropped():
    mgr, jog, moveit, engine = make_manager()
    orig = engine.suggest

    def suggest(*a, **k):
        out, stats = orig(*a, **k)
        out[0].q = {"joint1": 0.0}                                # joints missing
        out[1].q = dict(GOOD_Q, joint2=float("nan"))
        return out, stats
    engine.suggest = suggest
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["state"] == "ready" and [i["rank"] for i in st["items"]] == [3, 4, 5] and st["stats"]["malformed"] == 2


def test_clean_handles_numpy_and_nonfinite():
    out = sg.clean({"a": np.float64(1.5), "b": float("nan"), "c": (np.int64(2), [np.inf]), "d": np.array([1.0, np.nan]),
                    "e": np.bool_(True)})
    assert out == {"a": 1.5, "b": None, "c": [2, [None]], "d": [1.0, None], "e": True}
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------------------------------------------
# selection, RViz

def test_selection():
    mgr, jog, *_ = computed()
    assert mgr.select(3) == 3 and mgr.state()["selected"] == 3
    assert mgr.overlay()["label"] == "suggested #4" and len(mgr.overlay()["outline_px"]) == 4
    assert mgr.select(0) == 0 and mgr.select(3) == 3
    for bad in (-1, 5, 99):
        with pytest.raises(sg.SuggestRefused) as e:
            mgr.select(bad)
        assert e.value.status == 400
    for bad in (True, 1.0, "1", None):
        with pytest.raises(sg.SuggestRefused):
            mgr.select(bad)
    assert mgr.state()["selected"] == 3


def test_no_selection_or_overlay_before_a_search():
    mgr, *_ = make_manager()
    assert mgr.overlay() is None
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.select(0)
    assert e.value.status == 409
    with pytest.raises(sg.SuggestRefused):
        mgr.send_rviz(0)


def test_send_rviz_publishes_exactly_one_goal_state():
    mgr, jog, moveit, engine = computed()
    jog.calls.clear()
    msg = mgr.send_rviz(2)
    assert "goal shown in RViz" in msg and "Allow External Comm." in msg and "press Plan" in msg and "Execute" in msg
    assert moveit.published == [(NAMES, [GOOD_Q[n] for n in NAMES])]
    assert jog.calls == []                                        # nothing reaches the jog / robot
    assert mgr.state()["selected"] == 2
    with pytest.raises(sg.SuggestRefused):
        mgr.send_rviz(7)
    assert len(moveit.published) == 1


class FakeFuture:
    def __init__(self, result):
        self._r = result

    def add_done_callback(self, cb):
        cb(self)

    def result(self):
        return self._r


class FakeClient:
    def __init__(self, answer=None):
        self.requests, self.answer, self.is_ready = [], answer, True

    def service_is_ready(self):
        return self.is_ready

    def call_async(self, req):
        self.requests.append(req)
        return FakeFuture(self.answer)

    def remove_pending_request(self, fut):
        pass


class FakeNode:
    def __init__(self):
        self.clients, self.publishers = [], []

    def create_client(self, srv, name, **kw):
        c = FakeClient()
        self.clients.append((srv, name, c))
        return c

    def create_publisher(self, msg_type, topic, qos, **kw):
        pub = SimpleNamespace(msg_type=msg_type, topic=topic, sent=[], publish=lambda m: pub.sent.append(m))
        self.publishers.append(pub)
        return pub


def test_the_only_publisher_is_the_rviz_goal_state_topic():
    pytest.importorskip("moveit_msgs")
    from moveit_msgs.msg import RobotState
    node = FakeNode()
    bridge = sg.MoveItBridge(node)
    assert [(p.msg_type, p.topic) for p in node.publishers] == [(RobotState, "/rviz/moveit/update_custom_goal_state")]
    bridge.publish_goal(NAMES, [GOOD_Q[n] for n in NAMES])
    (msg,) = node.publishers[0].sent
    assert list(msg.joint_state.name) == NAMES and list(msg.joint_state.position) == [GOOD_Q[n] for n in NAMES]
    assert msg.is_diff is False
    # the source: one create_publisher, and no plan / execute / trajectory / action topic anywhere
    src = inspect.getsource(sg)
    assert len(re.findall(r"create_publisher\(", src)) == 1
    assert "create_action_client" not in src and "ActionClient" not in src and "create_subscription" not in src
    topics = set(re.findall(r'"(/[\w/]+)"', src))
    assert topics <= {"/rviz/moveit/update_custom_goal_state", "/compute_ik", "/check_state_validity"}, topics
    assert [(s.__name__, n) for s, n, _ in node.clients] == [("GetPositionIK", "/compute_ik"),
                                                             ("GetStateValidity", "/check_state_validity")]


def test_moveit_bridge_ik_request_and_validity():
    pytest.importorskip("moveit_msgs")
    from moveit_msgs.msg import MoveItErrorCodes, RobotState
    from moveit_msgs.srv import GetPositionIK, GetStateValidity
    node = FakeNode()
    bridge = sg.MoveItBridge(node)
    cli_ik, cli_valid = node.clients[0][2], node.clients[1][2]
    sol = RobotState()
    sol.joint_state.name = NAMES[::-1] + ["other"]
    sol.joint_state.position = [float(i) for i in range(7)]
    cli_ik.answer = GetPositionIK.Response(solution=sol, error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS))
    T = np.eye(4)
    T[:3, 3] = [0.3, -0.1, 0.4]
    q = bridge.ik(CFGS["left"], T, GOOD_Q)
    r = cli_ik.requests[0].ik_request
    assert (r.group_name, r.ik_link_name, r.avoid_collisions) == ("igus_rebel_arm", "link6", True)
    assert r.pose_stamped.header.frame_id == "base_link" and r.pose_stamped.pose.orientation.w == pytest.approx(1.0)
    assert r.pose_stamped.pose.position.y == pytest.approx(-0.1)
    assert r.timeout.sec == 0 and r.timeout.nanosec == 100_000_000
    assert list(r.robot_state.joint_state.name) == NAMES
    assert q == {n: float(5 - i) for i, n in enumerate(NAMES)}       # picked by name from the solution
    cli_ik.answer = GetPositionIK.Response(error_code=MoveItErrorCodes(val=MoveItErrorCodes.NO_IK_SOLUTION))
    assert bridge.ik(CFGS["left"], T, GOOD_Q) is None
    cli_valid.answer = GetStateValidity.Response(valid=True)
    assert bridge.valid(CFGS["left"], GOOD_Q) == (True, "")
    assert cli_valid.requests[0].group_name == "igus_rebel_arm"
    cli_valid.answer = GetStateValidity.Response(valid=False)
    assert bridge.valid(CFGS["left"], GOOD_Q)[0] is False
    cli_valid.answer = None
    assert bridge.valid(CFGS["left"], GOOD_Q)[0] is False
    cli_ik.is_ready = False
    ok, msg = bridge.ready()
    assert not ok and "/compute_ik" in msg


def test_dual_frames_and_link_names_are_prefixed():
    cfg = arm_configs("dual")["right"]
    assert sg.MoveItBridge.base_frame(cfg) == "arm_right_base_link" and sg.MoveItBridge.ik_link(cfg) == "arm_right_link6"
    assert sg.MoveItBridge.base_frame(CFGS["left"]) == "base_link" and sg.MoveItBridge.ik_link(CFGS["left"]) == "link6"


# ---------------------------------------------------------------------------------------------------
# plan / move pass-through

def test_plan_and_move_pass_through_with_page_id():
    mgr, jog, moveit, engine = computed()
    jog.calls.clear()
    res = mgr.plan("page-1", 1)
    assert jog.calls == [("plan_move", "page-1", "left", GOOD_Q)]
    assert res["ok"] and res["plan_id"] == "p1" and res["duration_s"] == 6.2 and res["index"] == 1
    assert mgr.state()["plan"] == {"plan_id": "p1", "index": 1, "arm": "left"} and mgr.state()["selected"] == 1
    out = mgr.move("page-1", "p1")
    assert out == {"ok": True, "message": "moving"} and jog.calls[-1] == ("execute_move", "page-1", "p1")
    assert mgr.state()["plan"] is None
    with pytest.raises(sg.SuggestRefused):
        mgr.move("page-1", "p1")                                  # consumed: no second execute
    assert [c[0] for c in jog.calls].count("execute_move") == 1


def test_move_refused_without_matching_plan_and_failed_plan_is_not_kept():
    mgr, jog, *_ = computed()
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.move("page-1", "p1")
    assert e.value.status == 409 and "Plan preview" in e.value.message
    jog.plan_answer = {"ok": False, "message": "collision at state 3"}
    res = mgr.plan("page-1", 0)
    assert res["ok"] is False and "collision" in res["message"] and mgr.state()["plan"] is None
    jog.plan_answer = {"ok": True, "plan_id": "p2", "message": "planned", "duration_s": 5.0}
    mgr.plan("page-1", 0)
    with pytest.raises(sg.SuggestRefused):
        mgr.move("page-1", "wrong-id")
    assert not any(c[0] == "execute_move" for c in jog.calls)


def test_selecting_another_suggestion_drops_the_stored_plan():
    mgr, jog, *_ = computed()
    mgr.plan("page-1", 0)
    jog.calls.clear()
    mgr.select(0)                                                  # same one: plan stays
    assert mgr.state()["plan"] is not None and jog.calls == []
    mgr.select(2)
    assert mgr.state()["plan"] is None and jog.calls == [("clear_plan",)]
    with pytest.raises(sg.SuggestRefused):
        mgr.move("page-1", "p1")


def test_jog_refusal_is_passed_back_and_plan_kept_for_retry():
    mgr, jog, *_ = computed()
    mgr.plan("page-1", 0)
    jog.exec_answer = {"ok": False, "message": "REAL robot locked: enable motion first"}
    res = mgr.move("page-1", "p1")
    assert res["ok"] is False and "locked" in res["message"]
    assert mgr.state()["plan"] is not None
    jog.exec_answer = {"ok": True, "message": "moving"}
    assert mgr.move("page-1", "p1")["ok"] is True


def test_plan_refused_while_the_arm_is_busy():
    mgr, jog, *_ = computed()
    jog.busy = True
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.plan("page-1", 0)
    assert e.value.status == 409 and not any(c[0] == "plan_move" for c in jog.calls)


def test_jog_without_move_support_is_refused_clearly():
    mgr, jog, *_ = computed()

    class OldJog:
        status = staticmethod(lambda: {"available": True})
    mgr.jog = OldJog()
    with pytest.raises(sg.SuggestRefused) as e:
        mgr.plan("page-1", 0)
    assert e.value.status == 503
    with pytest.raises(sg.SuggestRefused):
        mgr.move("page-1", "p1")


# ---------------------------------------------------------------------------------------------------
# stale marking

def test_stale_when_the_session_gets_new_samples(tmp_path):
    sample_file = tmp_path / "samples.jsonl"

    def write(n_left, n_right=0):
        rows = [json.dumps({"arm": "left"})] * n_left + [json.dumps({"arm": "right"})] * n_right
        sample_file.write_text("\n".join(rows) + "\n")
    write(8, 4)
    mgr, jog, *_ = computed(tmp_path=tmp_path, n_samples=8)
    st = mgr.state()
    assert st["based_on_samples"] == 8 and st["session_samples"] == 8 and st["stale"] is False
    write(8, 9)                                                    # the other arm's samples do not matter
    assert mgr.state()["stale"] is False
    write(9, 9)
    st = mgr.state()
    assert st["session_samples"] == 9 and st["stale"] is True and st["based_on_samples"] == 8
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["based_on_samples"] == 8                             # fake loader says 8; the marking logic is the point
    assert mgr.counter.count(str(tmp_path / "missing"), "left") is None
    assert mgr.counter.count(str(tmp_path), "left") == 9


def test_state_before_any_search():
    mgr, *_ = make_manager()
    st = mgr.state()
    assert st["state"] == "idle" and st["items"] == [] and st["selected"] is None and st["stale"] is False
    assert st["based_on_samples"] is None and st["plan"] is None and st["supported"] is True


# ---------------------------------------------------------------------------------------------------
# the REAL engine through the glue (keyword contract: pose_suggest.suggest() calls ik(T, seed_q=current_q))

def _real_engine_manager(moveit, n_arm_samples=16):
    from test_pose_suggest import ik_map, make_samples            # noqa: F401  (synthetic samples of arm 'right')
    cfgs = arm_configs("single_right")
    jog = FakeJog()
    jog.q = {n: 0.0 for n in cfgs["right"].joint_names}
    smp = make_samples(n_arm_samples, seed=3)
    mgr = sg.SuggestManager("single_right", cfgs, jog, lambda: "/nonexistent/session", moveit=moveit,
                            load_samples=lambda d: list(smp))     # engine=None: the real pose_suggest module
    return mgr, jog


class _RealishMoveIt(FakeMoveIt):
    """ik answers a smooth fake function of the pose, with the bridge signature ik(cfg, T, seed_q)."""

    def ik(self, cfg, T, seed_q):
        from test_pose_suggest import ik_map
        self.ik_calls.append((cfg, np.array(T), dict(seed_q)))
        v = ik_map(np.asarray(T))
        return {n: v[f"joint{i + 1}"] for i, n in enumerate(cfg.joint_names)}


def test_real_engine_runs_end_to_end_through_the_glue_with_the_keyword_call():
    moveit = _RealishMoveIt()
    mgr, jog = _real_engine_manager(moveit)
    mgr.compute()
    st = wait_not_computing(mgr, 60.0)
    assert st["state"] == "ready", st
    assert len(st["items"]) >= 1 and st["stats"].get("ik_error", 0) == 0 and st["stats"]["accepted"] >= 1
    assert moveit.ik_calls and moveit.valid_calls
    # the seed reached the bridge as the current joint dict
    assert set(moveit.ik_calls[0][2]) == set(arm_configs("single_right")["right"].joint_names)


def test_ik_wrapper_parameter_is_called_seed_q():
    """pose_suggest.suggest() passes the seed by keyword; a rename of the wrapper's parameter breaks every IK call."""
    seen = {}

    class KwEngine(FakeEngine):
        def suggest(self, fit, ik, valid, current_q, n=5, options=None):
            seen["q"] = ik(np.eye(4), seed_q=dict(current_q))
            return [make_suggestion(1)], {"accepted": 1}

    mgr, jog, moveit, _ = make_manager()
    mgr._engine = KwEngine()
    mgr.compute()
    assert wait_not_computing(mgr)["state"] == "ready" and seen["q"] == GOOD_Q
    assert moveit.ik_calls[0][2] == GOOD_Q                       # and the bridge call stays positional (cfg, T, seed)


def test_ik_service_faults_are_reported_as_such_and_the_first_exception_is_logged():
    import logging
    records = []

    class Grab(logging.Handler):          # (caplog is not usable here: a ROS pytest plugin bypasses it)
        def emit(self, record):
            records.append(record.getMessage())

    handler, logger = Grab(), logging.getLogger("calibration_ui.suggest")
    logger.addHandler(handler)
    class BrokenMoveIt(FakeMoveIt):
        def ik(self, cfg, T, seed_q):
            raise TimeoutError("service /compute_ik vanished")

    moveit = BrokenMoveIt()
    mgr, jog = _real_engine_manager(moveit)
    try:
        mgr.compute()
        st = wait_not_computing(mgr, 60.0)
    finally:
        logger.removeHandler(handler)
    assert st["state"] == "error" and st["items"] == []
    assert st["message"].startswith("IK service calls failed (") and "is move_group running? see the UI log" in st["message"]
    assert "TimeoutError: service /compute_ik vanished" in st["message"]
    n = int(re.search(r"\((\d+) errors\)", st["message"]).group(1))
    assert n >= 1 and n == st["stats"]["ik_error"]
    logged = [m for m in records if "IK call failed" in m]
    assert len(logged) == 1 and "service /compute_ik vanished" in logged[0]       # only the FIRST exception is logged


def test_no_pose_found_without_ik_errors_keeps_the_old_message():
    mgr, jog, moveit, engine = make_manager()
    engine.suggest = lambda fit, ik, valid, q, n=5, options=None: ([], {"ik_failed": 3, "accepted": 0})
    mgr.compute()
    st = wait_not_computing(mgr)
    assert st["state"] == "error" and st["message"].startswith("no reachable collision-free pose found")


# ---------------------------------------------------------------------------------------------------
# safety review: STOP epoch pass-through (F3) and the plan id binding (F7)

def test_f3_epoch_is_passed_to_the_jog_only_when_given():
    mgr, jog, moveit, engine = computed()
    seen = []
    jog.plan_move = lambda page, arm, q, **kw: (seen.append(("plan", kw)), dict(jog.plan_answer))[1]
    jog.execute_move = lambda page, pid, **kw: (seen.append(("move", kw)), dict(jog.exec_answer))[1]
    mgr.plan("P", 0)
    assert seen[-1] == ("plan", {})
    mgr.plan("P", 0, epoch=7)
    assert seen[-1] == ("plan", {"epoch": 7})
    mgr.move("P", "p1", epoch=7)
    assert seen[-1] == ("move", {"epoch": 7})


def test_f7_move_refuses_an_id_other_than_the_stored_plan_and_never_calls_the_jog():
    mgr, jog, moveit, engine = computed()
    mgr.plan("P", 0)                                           # stores p1
    jog.plan_answer = dict(jog.plan_answer, plan_id="p2")
    mgr.plan("P", 0)                                           # replaces it with p2
    for wrong in ("p1", "", None, 5):
        with pytest.raises(sg.SuggestRefused) as e:
            mgr.move("P", wrong)
        assert e.value.status == 409 and "no stored plan with this id" in e.value.message
    assert not any(c[0] == "execute_move" for c in jog.calls)
    assert mgr.move("P", "p2")["ok"] and jog.calls[-1] == ("execute_move", "P", "p2")
