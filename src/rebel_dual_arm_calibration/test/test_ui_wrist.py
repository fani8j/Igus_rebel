"""Wrist-camera panel (ui/wrist_ui.py + /api/wrist/*): fake ROS / MoveIt / jog, synthetic board images; the motion layer is
mocked.  Safety rules under test: real hardware needs allow_real_move + the page's enable + the per-run confirmation, the
sequence stops on STOP / supervision fault / 3 failures / lost page / missing scene, and never moves without a valid plan."""
import json
import math
import threading
import time
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from rebel_dual_arm_calibration import eye_in_hand as E
from rebel_dual_arm_calibration import safety_scene as ss
from rebel_dual_arm_calibration import wrist_poses as wp
from rebel_dual_arm_calibration import wrist_synthetic as W
from rebel_dual_arm_calibration.mock_common import BoardImage
from rebel_dual_arm_calibration.targets import TargetDetector
from rebel_dual_arm_calibration.transforms import inv_T, make_T, rotvec_to_R, xyz_rpy_to_T
from rebel_dual_arm_calibration.ui import wrist_ui as wu
from rebel_dual_arm_calibration.ui.jog import arm_configs

CFGS = arm_configs("single_left")
NAMES = list(CFGS["left"].joint_names)
SPEC = W.table_spec()
K = W.WRIST_K
SIZE = W.WRIST_SIZE
X_TRUE = xyz_rpy_to_T([-0.02, -0.09, -0.04], np.radians([10, -5, 95]))
B_TRUE = xyz_rpy_to_T([0.514, -0.148, -0.251], np.radians([180.0, 0.0, 20.0]))
Q_SEED = {n: 0.1 * (i + 1) for i, n in enumerate(NAMES)}
A_SEED = None      # set in fixture


def seed_A():
    n_out = -B_TRUE[:3, 2]
    ax = np.cross(n_out, [1.0, 0, 0])
    eye = B_TRUE[:3, 3] + 0.38 * (rotvec_to_R(ax / np.linalg.norm(ax) * math.radians(10)) @ n_out)
    return W.look_at(eye, B_TRUE[:3, 3], B_TRUE[:3, 1]) @ inv_T(X_TRUE)


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.hooks = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.01)
        for h in list(self.hooks):
            h(self)


class World:
    """q <-> link6 pose bookkeeping shared by the fakes (the 'robot')."""

    def __init__(self):
        self.A0 = seed_A()
        self.q = dict(Q_SEED)
        self.table = {tuple(round(v, 9) for v in Q_SEED.values()): self.A0}

    def q_for(self, T):
        rel = np.concatenate([T[:3, 3] - self.A0[:3, 3], cv2.Rodrigues(np.ascontiguousarray(self.A0[:3, :3].T @ T[:3, :3]))[0].ravel()])
        q = {n: Q_SEED[n] + 0.3 * float(rel[i]) for i, n in enumerate(NAMES)}
        self.table[tuple(round(v, 9) for v in q.values())] = T
        return q

    def A_now(self):
        return self.table.get(tuple(round(self.q[n], 9) for n in NAMES), self.A0)


class FakeMoveIt:
    def __init__(self, world):
        self.world, self.ik_fail, self.invalid, self.calib_contact = world, set(), set(), False
        self.ik_calls = 0

    def ready(self):
        return True, ""

    def ik(self, cfg, T, seed_q):
        self.ik_calls += 1
        if self.ik_calls in self.ik_fail:
            return None
        return self.world.q_for(T)

    def valid(self, cfg, q):
        if self.calib_contact:
            return False, "invalid state (link5 / calib_table_slab)"
        return True, ""


class FakeJog:
    def __init__(self, world, real=False):
        self.world, self.real = world, real
        self.calls = []
        self.core = NS(is_real=lambda arm: self.real, standstill=lambda arm: True, stop_epoch=0)
        self._status = {"available": True, "busy": False, "stopping": False, "fault": False, "last": None,
                        "real_enabled": False, "real_enabled_page": None}
        self.plan_answer = None            # override dict
        self.exec_phase = "done"
        self.hold = False                  # execute_move leaves the arm busy until stop()
        self.sid = 0
        self.plans = {}
        self.exec_hook = None

    def status(self):
        return dict(self._status)

    def joint_window(self, arms, window_s):
        return dict(self.world.q), 0.0, True

    def reserve_motion(self):
        return {"ok": True, "epoch": self.core.stop_epoch, "token": 1}

    def release_motion(self, token):
        pass

    def plan_move(self, page_id, arm, q_goal, epoch=None):
        self.calls.append(("plan", dict(q_goal), epoch))
        if self.plan_answer is not None:
            return dict(self.plan_answer)
        pid = f"plan{len(self.calls)}"
        self.plans[pid] = dict(q_goal)
        return {"ok": True, "plan_id": pid, "message": "planned", "duration_s": 5.0}

    def execute_move(self, page_id, plan_id, epoch=None):
        self.calls.append(("execute", plan_id, epoch))
        self.sid += 1
        if self.exec_hook:
            self.exec_hook(self)
        if self.hold:
            self._status.update(busy=True, last={"id": self.sid, "phase": "executing", "ok": True})
        else:
            if self.exec_phase == "done":
                self.world.q = dict(self.plans[plan_id])
            self._status.update(busy=False, last={"id": self.sid, "phase": self.exec_phase, "ok": self.exec_phase == "done",
                                                  "message": "x"})
        return {"ok": True, "id": self.sid, "plan_id": plan_id, "message": "accepted"}

    def stop(self):
        self.calls.append(("stop",))
        self.core.stop_epoch += 1
        if self._status["busy"]:
            self._status.update(busy=False, last={"id": self.sid, "phase": "stopped", "ok": False, "message": "stopped"})
        return {"ok": True}


class FakeRos:
    def __init__(self, world):
        self.world = world
        self.boxes = {}
        self.applied, self.removed = [], 0
        self.scene_present = True
        self.pole = ss.box_from_object(NS(id="camera_pole", header=NS(frame_id="base_link"),
                                          pose=NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=1)),
                                          primitives=[NS(type=1, dimensions=[0.03, 0.03, 1.0])], meshes=[], planes=[],
                                          primitive_poses=[NS(position=NS(x=-0.1, y=-0.3, z=0.5),
                                                              orientation=NS(x=0, y=0, z=0, w=1))]))
        self.head = ss.BoxSpec("camera_head", "base_link", (0.1, 0.1, 0.1), make_T(np.eye(3), [-0.05, -0.2, 0.87]))
        self.frames = []
        self.info = {"K": K, "dist": np.zeros(5), "size": SIZE, "frame_id": "wrist_camera_color_optical_frame"}
        self.tf_calls = []
        self.tf_shift = 0.0
        self.static = []
        self.tf_ok = True
        self.cl_fail = False
        self.T_cl_opt = np.eye(4)

    def wrist_info(self):
        return self.info

    def wrist_frames(self, since=0.0):
        return list(self.frames)

    def latest_wrist_frame(self):
        return self.frames[-1] if self.frames else None

    def wrist_health(self):
        return {"image_age_s": 0.1, "camera_info": True}

    def tf_base_link6(self, stamp):
        self.tf_calls.append(stamp)
        if not self.tf_ok:
            return None, "TF base_link->link6 unavailable"
        A = self.world.A_now().copy()
        A[0, 3] += self.tf_shift * len(self.tf_calls)
        return A, ""

    def tf_age_s(self):
        return 0.05

    def scene_boxes(self, names):
        return {ss.POLE_ID: self.pole, ss.HEAD_ID: self.head}

    def apply_boxes(self, specs):
        self.applied.append(list(specs))
        self.boxes = {s.id: s for s in specs}

    def remove_boxes(self, ids=ss.OUR_IDS):
        self.removed += 1
        self.boxes = {}

    def scene_has(self, ids):
        if not self.scene_present:
            return False, list(ids)
        have = set(self.boxes)
        miss = [i for i in ids if i not in have]
        return not miss, miss

    def tf_camera_link_optical(self, frame=None):
        self.cl_frames = getattr(self, "cl_frames", []) + [frame]
        if self.cl_fail:
            return None, "TF wrist_camera_link->wrist_camera_color_optical_frame unavailable"
        return self.T_cl_opt.copy(), ""

    def publish_static_tf(self, T):
        self.static.append(T)


def make_frame(A, stamp, noise=0.0):
    """A wrist image of the table board for the link6 pose A (pinhole K, no distortion)."""
    img = np.full((SIZE[1], SIZE[0], 3), 140, np.uint8)
    BoardImage(SPEC, margin=0.010, plate=0.0).warp_into(img, K, inv_T(A @ X_TRUE) @ B_TRUE)
    return {"stamp": stamp, "rx": stamp, "seq": int(stamp * 1000), "bgr": img, "frame_id": "wrist_camera_color_optical_frame"}


@pytest.fixture()
def rig(tmp_path):
    world = World()
    clk = Clock()
    jog = FakeJog(world)
    ros = FakeRos(world)
    moveit = FakeMoveIt(world)
    det = TargetDetector({"table": SPEC})
    mgr = wu.WristManager("single_left", CFGS, jog, ros, moveit, lambda: str(tmp_path / "wr"), lambda: (det, SPEC),
                          clock=clk.now, sleep=clk.sleep, start_thread=False)
    mgr._set_board(B_TRUE, "mock_synthetic", {})
    beat = lambda c: mgr.heartbeat("p1")             # noqa: E731  the page polls /api/state about once a second
    clk.hooks.append(beat)
    return NS(beat=beat, world=world, clk=clk, jog=jog, ros=ros, moveit=moveit, mgr=mgr, tmp=tmp_path, det=det)


def stub_capture(rig, fail_every=None):
    """Replace the image part of the capture by a synthetic sample at the arm's pose (the rest of the run is real)."""
    n = {"i": 0}

    def cap(check=lambda: None):
        n["i"] += 1
        check()
        if fail_every and fail_every(n["i"]):
            raise wu.WristRefused("board not detected (stub)", 409)
        A = rig.world.A_now()
        S = W.make_samples(X_TRUE, B_TRUE, [A @ X_TRUE], noise_px=0.3, seed=n["i"])
        if not S:
            raise wu.WristRefused("board not visible (stub)", 409)
        s = S[0]
        s.stamp = rig.clk.now()
        return s
    rig.mgr._capture_sample = cap
    return n


def prepare(rig, page="p1", scene=True, gen=True):
    m = rig.mgr
    if scene:
        m.scene_add()
    rig.ros.frames = [make_frame(rig.world.A0, rig.clk.now())]       # the seed image: gives the seed-derived mount estimate
    m.use_seed()
    rig.ros.frames = []
    if gen:
        m.generate()
    return m


def test_mock_board_refused_on_real_hardware(rig):
    rig.jog.real = True
    with pytest.raises(wu.WristRefused) as e:
        rig.mgr.mock_board({"xyz": [0.5, 0, -0.2]})
    assert e.value.status == 403


def test_scene_add_builds_four_boxes_and_is_idempotent(rig):
    st = rig.mgr.scene_add()
    assert st["applied"] and {b["id"] for b in st["boxes"]} == set(ss.OUR_IDS)
    rig.mgr.scene_add()
    assert len(rig.ros.applied) == 2 and set(rig.ros.boxes) == set(ss.OUR_IDS)
    slab = rig.ros.boxes[ss.SLAB_ID]
    corners = ss.board_corners(B_TRUE, SPEC.board_size())
    assert slab.top_z() == pytest.approx(corners[:, 2].max() + 0.05)
    rig.mgr.scene_remove()
    assert rig.ros.boxes == {} and not rig.mgr.scene_state()["applied"]


def test_scene_add_refuses_without_pole_and_when_it_touches_the_robot(rig):
    rig.ros.pole = None
    with pytest.raises(wu.WristRefused, match="camera_pole"):
        rig.mgr.scene_add()
    assert rig.ros.applied == []
    rig.ros.pole = FakeRos(rig.world).pole
    rig.moveit.calib_contact = True
    with pytest.raises(wu.WristRefused, match="intersects the robot"):
        rig.mgr.scene_add()
    assert rig.ros.removed == 1 and not rig.mgr.scene_state()["applied"]


def test_scene_goes_away_on_shutdown(rig):
    rig.mgr.scene_add()
    rig.mgr.shutdown()
    assert rig.ros.boxes == {}


def test_seed_with_image_gives_a_good_mount_estimate_and_checks_visibility(rig):
    rig.ros.frames = [make_frame(rig.world.A0, rig.clk.now())]
    rig.mgr.use_seed()
    X, src = rig.mgr._current_X()
    assert "seed" in src
    assert E.pose_delta(X, X_TRUE)[0] < 0.004 and E.pose_delta(X, X_TRUE)[1] < 1.0
    # board only partly in the image: refused
    A_far = rig.world.A0.copy()
    A_far[:3, 3] += (rig.world.A0[:3, :3] @ X_TRUE[:3, :3])[:, 0] * 0.2
    rig.ros.frames = [make_frame(A_far, rig.clk.now() + 1)]
    with pytest.raises(wu.WristRefused):
        rig.mgr.use_seed()


def test_seed_needs_standstill_and_idle_arm(rig):
    rig.jog._status["busy"] = True
    with pytest.raises(wu.WristRefused, match="busy"):
        rig.mgr.use_seed()


def test_capture_on_real_images_records_a_sample(rig):
    rig.ros.frames = [make_frame(rig.world.A0, 1000.0 + k * 0.3) for k in range(4)]
    rig.mgr.scene_add()
    r = rig.mgr.capture_now()
    assert r["ok"]
    s = rig.mgr._samples[0]
    assert len(s.ids) >= 16 and s.image.startswith("images/wrist_")
    assert (rig.tmp / "wr" / s.image).exists()
    assert len(E.load_samples(str(rig.tmp / "wr"))) == 1
    # the detected corners agree with the projection of the known geometry
    uv = W.project(inv_T(rig.world.A0 @ X_TRUE) @ B_TRUE, s.obj, K, np.zeros(5))
    assert np.sqrt(np.mean(np.sum((uv - s.pix) ** 2, axis=1))) < 1.0


def test_capture_refuses_when_arm_moved_or_board_missing_or_tf_missing(rig):
    rig.ros.frames = [make_frame(rig.world.A0, 1000.0 + k * 0.3) for k in range(4)]
    rig.ros.tf_shift = 0.001                       # base->link6 changes by 1 mm between the stamps
    with pytest.raises(wu.WristRefused, match="moved during the capture"):
        rig.mgr._capture_sample()
    rig.ros.tf_shift = 0.0
    rig.ros.tf_ok = False
    with pytest.raises(wu.WristRefused, match="TF"):
        rig.mgr._capture_sample()
    rig.ros.tf_ok = True
    rig.ros.frames = [{"stamp": 1000.0 + k * 0.3, "rx": 1.0, "seq": k, "bgr": np.full((SIZE[1], SIZE[0], 3), 140, np.uint8),
                       "frame_id": "x"} for k in range(4)]
    with pytest.raises(wu.WristRefused, match="not detected"):
        rig.mgr._capture_sample()


def test_generate_reports_counts_and_reasons(rig):
    rig.moveit.ik_fail = {2, 3, 4, 5}
    prepare(rig, gen=False)
    rig.mgr.generate()
    st = rig.mgr.state()["views"]
    assert st["state"] == "ready" and st["stats"]["accepted"] >= 10
    assert st["stats"]["rejected"].get("no collision-free IK", 0) == 4
    assert st["items"][0]["seed_view"] is True
    assert st["stats"]["rotation_span_deg"] > 30
    assert "camera mount estimate" in st["message"] or "views accepted" in st["message"]


def test_generate_needs_scene_seed_board(rig):
    with pytest.raises(wu.WristRefused):
        rig.mgr.generate()
    rig.mgr.use_seed()
    with pytest.raises(wu.WristRefused, match="safety scene"):
        rig.mgr.generate()


# ------------------------------------------------------------------------------------------------ the sequence
def start(rig, page="p1", confirm=True):
    return rig.mgr.start(page, confirm)


def test_full_sequence_in_mock_moves_captures_and_solves(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "finished", run
    plans = [c for c in rig.jog.calls if c[0] == "plan"]
    execs = [c for c in rig.jog.calls if c[0] == "execute"]
    assert len(plans) == len(execs) >= 10
    assert len(rig.mgr._samples) >= 12
    res = rig.mgr.state()["result"]
    assert res["verdict"] in ("PASS", "WARN")
    assert E.pose_delta(rig.mgr._result.X, X_TRUE)[0] < 0.002
    # nothing moved without a plan: every execute follows its own plan call
    seq = [c[0] for c in rig.jog.calls]
    assert all(seq[i] == "plan" and seq[i + 1] == "execute" for i in range(0, len(seq) - 1, 2))
    # the samples are saved for an offline re-solve
    assert len(E.load_samples(str(rig.tmp / "wr"))) == len(rig.mgr._samples)
    # interim solve was used for the in-view prediction after #4
    assert "interim" in rig.mgr.state()["estimate"]["source"]


def test_real_hardware_refused_without_allow_real_move(rig):
    prepare(rig)
    rig.jog.real = True
    with pytest.raises(wu.WristRefused) as e:
        start(rig)
    assert e.value.status == 403 and "allow_real_move" in e.value.message
    assert rig.jog.calls == []


def test_real_hardware_needs_this_pages_enable_and_the_confirmation(rig):
    rig.jog.real = True
    rig.mgr.allow_real_move = True
    rig.mgr.mock_board  # (mock board is refused on real, the board was set by the fixture)
    rig.ros.frames = [make_frame(rig.world.A0, 1000.0)]
    prepare(rig)
    stub_capture(rig)
    with pytest.raises(wu.WristRefused, match="Enable motion"):
        start(rig)
    rig.jog._status.update(real_enabled=True, real_enabled_page="other")
    with pytest.raises(wu.WristRefused, match="Enable motion"):
        start(rig)
    rig.jog._status.update(real_enabled=True, real_enabled_page="p1")
    with pytest.raises(wu.WristRefused, match="confirmation"):
        rig.mgr.start("p1", False)
    with pytest.raises(wu.WristRefused, match="confirmation"):
        rig.mgr.start("p1", None)
    assert rig.jog.calls == []
    start(rig)
    assert rig.mgr.state()["run"]["state"] == "finished" and len([c for c in rig.jog.calls if c[0] == "plan"]) >= 10


def test_start_refused_without_views_or_scene(rig):
    with pytest.raises(wu.WristRefused, match="views"):
        start(rig)
    prepare(rig)
    rig.ros.boxes.pop(ss.POLE_PAD_ID)
    with pytest.raises(wu.WristRefused, match="missing"):
        start(rig)
    assert rig.jog.calls == []


def test_scene_vanishing_mid_run_stops_before_the_next_move(rig):
    prepare(rig)
    stub_capture(rig)
    n_exec = {"n": 0}

    def hook(jog):
        n_exec["n"] += 1
        if n_exec["n"] == 2:
            rig.ros.boxes.pop(ss.SLAB_ID)
    rig.jog.exec_hook = hook
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "safety scene" in run["message"]
    assert len([c for c in rig.jog.calls if c[0] == "execute"]) == 2


def test_stop_during_a_move_ends_the_run_and_sends_jog_stop(rig):
    prepare(rig)
    stub_capture(rig)
    rig.jog.hold = True

    def pressed(clk):
        if any(c[0] == "execute" for c in rig.jog.calls):
            rig.mgr.stop()
    rig.clk.hooks.append(pressed)
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "stopped", run
    assert ("stop",) in rig.jog.calls
    assert len([c for c in rig.jog.calls if c[0] == "execute"]) == 1
    assert len([c for c in rig.jog.calls if c[0] == "plan"]) == 1


def test_jog_stop_button_epoch_change_ends_the_run(rig):
    prepare(rig)
    stub_capture(rig)

    def hook(jog):
        jog.core.stop_epoch += 1                  # the operator pressed the global STOP / a fault happened
    rig.jog.exec_hook = hook
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "STOP" in run["message"]
    assert len([c for c in rig.jog.calls if c[0] == "execute"]) == 1


def test_supervision_fault_in_a_move_ends_the_run(rig):
    prepare(rig)
    stub_capture(rig)
    rig.jog.exec_phase = "failed"
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "did not finish cleanly" in run["message"]
    assert len([c for c in rig.jog.calls if c[0] == "execute"]) == 1


def test_three_consecutive_capture_failures_stop_the_run(rig):
    prepare(rig)
    stub_capture(rig, fail_every=lambda i: i >= 2)
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "3 consecutive" in run["message"]
    assert len(rig.mgr._samples) == 1


def test_isolated_capture_failures_are_skipped_not_fatal(rig):
    prepare(rig)
    stub_capture(rig, fail_every=lambda i: i % 3 == 0)
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "finished" and run["skipped"] >= 2
    assert len(rig.mgr._samples) >= 8


def test_plan_failure_skips_the_view_and_never_executes_it(rig):
    prepare(rig)
    stub_capture(rig)
    rig.jog.plan_answer = {"ok": False, "plan_id": None, "message": "move blocked: MoveIt found no path (error code -1)"}
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "3 consecutive" in run["message"]
    assert [c for c in rig.jog.calls if c[0] == "execute"] == []


def test_planning_refusal_by_the_jog_ends_the_run(rig):
    prepare(rig)
    stub_capture(rig)
    rig.jog.plan_answer = {"ok": False, "plan_id": None, "message": "REAL robot locked: real-hardware enable no longer valid"}
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "planning refused" in run["message"]
    assert len([c for c in rig.jog.calls if c[0] == "plan"]) == 1


def test_page_that_stops_polling_ends_the_run(rig):
    prepare(rig)
    stub_capture(rig)
    rig.clk.hooks.remove(rig.beat)                                           # the browser is gone
    rig.clk.hooks.append(lambda clk: setattr(clk, "t", clk.t + 3.0))
    start(rig)
    run = rig.mgr.state()["run"]
    assert run["state"] == "failed" and "stopped polling" in run["message"]


def test_heartbeat_keeps_the_run_alive(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    assert rig.mgr.state()["run"]["state"] == "finished"


def test_capture_refused_while_running_and_start_twice(rig):
    prepare(rig)
    with rig.mgr.lock:
        rig.mgr._run["state"] = "running"
    with pytest.raises(wu.WristRefused):
        rig.mgr.capture_now()
    with pytest.raises(wu.WristRefused, match="already running"):
        start(rig)


def test_result_files_and_publish_tf(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    out = rig.mgr.write()
    for p in out["files"].values():
        assert (rig.tmp / "wr").exists() and p.startswith(str(rig.tmp / "wr"))
    assert "xacro" in out["files"] and "parent link=\"link6\"" in open(out["files"]["xacro"]).read()
    rig.mgr.publish_tf()
    assert len(rig.ros.static) == 1 and E.pose_delta(rig.ros.static[0], X_TRUE)[0] < 0.002    # identity driver transform
    assert rig.ros.cl_frames[-1] == "wrist_camera_color_optical_frame"       # looked up for the frame the images are in
    with rig.mgr.lock:
        rig.mgr._result.gate["verdict"] = "FAIL"
    with pytest.raises(wu.WristRefused, match="FAILED"):
        rig.mgr.publish_tf()
    # re-solve offline from the directory gives the same mount
    S = E.load_samples(str(rig.tmp / "wr"))
    assert E.pose_delta(E.solve(S, loo=False).X, X_TRUE)[0] < 0.002


def test_publish_tf_is_link6_to_wrist_camera_link_composed_with_the_driver_transform(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    rig.ros.T_cl_opt = xyz_rpy_to_T([0.01, -0.02, 0.03], [0.1, 0.2, -0.3])      # what the camera driver publishes
    rig.mgr.publish_tf()
    T_pub = rig.ros.static[-1]
    assert np.allclose(T_pub @ rig.ros.T_cl_opt, rig.mgr._result.X, atol=1e-9)    # link6 -> link -> optical == X


def test_publish_and_write_fail_closed_without_the_driver_transform(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    rig.ros.cl_fail = True
    with pytest.raises(wu.WristRefused, match="not available"):
        rig.mgr.publish_tf()
    assert rig.ros.static == []
    out = rig.mgr.write()
    assert "NOT written" in out["message"] and "xacro" not in out["files"] and "yaml" in out["files"]
    assert not (rig.tmp / "wr" / E.RESULT_XACRO).exists()


def test_delete_sample_resolves(rig):
    prepare(rig)
    stub_capture(rig)
    start(rig)
    n = len(rig.mgr._samples)
    rig.mgr.delete_sample(0)
    assert len(rig.mgr._samples) == n - 1 and len(E.load_samples(str(rig.tmp / "wr"))) == n - 1
    with pytest.raises(wu.WristRefused):
        rig.mgr.delete_sample(999)


def test_state_is_json_serialisable_at_every_stage(rig):
    json.dumps(rig.mgr.state(), allow_nan=False)
    prepare(rig)
    stub_capture(rig)
    json.dumps(rig.mgr.state(), allow_nan=False)
    start(rig)
    json.dumps(rig.mgr.state(), allow_nan=False)


def test_unsupported_modes():
    m = wu.WristManager("dual", arm_configs("dual"), None, None, None, lambda: None, lambda: None)
    assert m.state()["supported"] is False
    with pytest.raises(wu.WristRefused):
        m.locate_board()


# ------------------------------------------------------------------------------------------------ HTTP contract
pytest.importorskip("tornado")
from test_ui_server import Base as _ServerBase, PAGE   # noqa: E402


class TestWristHttp(_ServerBase):
    mode = "single_left"

    def get_app(self):
        app = super().get_app()
        self.world = World()
        self.wjog = FakeJog(self.world)
        self.wros = FakeRos(self.world)
        det = TargetDetector({"table": SPEC})
        self.clk = Clock()
        self.mgr = wu.WristManager("single_left", CFGS, self.wjog, self.wros, FakeMoveIt(self.world),
                                   lambda: "/tmp/wrist_http_test", lambda: (det, SPEC), clock=self.clk.now,
                                   sleep=self.clk.sleep)
        self.mgr._set_board(B_TRUE, "mock_synthetic", {})
        self.ctx.wrist = self.mgr
        return app

    def test_all_wrist_endpoints_need_the_token(self):
        for a in ("locate_board", "mock_board", "scene_add", "scene_remove", "seed", "generate", "start", "stop", "capture",
                  "solve", "write", "publish_tf", "delete_sample"):
            r = self.fetch(f"/api/wrist/{a}", method="POST", body="{}")
            assert r.code == 401, a
        assert self.fetch("/api/wrist/image.jpg").code == 401
        assert self.wjog.calls == []

    def test_state_has_a_wrist_section_and_unknown_action_404(self):
        st = self.j(self.get("/api/state"))["wrist"]
        assert st["supported"] is True and st["arm"] == "left" and st["run"]["state"] == "idle"
        assert self.post("/api/wrist/nope").code == 404

    def test_start_needs_page_id_confirmation_and_views(self):
        r = self.post("/api/wrist/start", {"confirm": True}, headers={"X-Token": "test-token-123"})
        assert r.code == 400
        r = self.post("/api/wrist/start", {})
        assert r.code == 409 and "confirmation" in self.j(r)["message"]
        r = self.post("/api/wrist/start", {"confirm": True})
        assert r.code == 409 and "views" in self.j(r)["message"]
        assert self.wjog.calls == []

    def test_start_refused_on_real_hardware_without_allow_real_move(self):
        self.wjog.real = True
        r = self.post("/api/wrist/start", {"confirm": True})
        assert r.code == 403 and "allow_real_move" in self.j(r)["message"]
        assert self.wjog.calls == []

    def test_mock_board_refused_when_real(self):
        self.wjog.real = True
        assert self.post("/api/wrist/mock_board", {"xyz": [0.5, 0, -0.2]}).code == 403

    def test_stop_sends_the_jog_stop(self):
        r = self.post("/api/wrist/stop")
        assert r.code == 200 and ("stop",) in self.wjog.calls

    def test_scene_and_image_endpoints(self):
        r = self.post("/api/wrist/scene_add")
        assert r.code == 200 and self.j(r)["wrist"]["scene"]["applied"] is True
        assert self.get("/api/wrist/image.jpg").code == 404                     # no wrist image yet
        self.wros.frames = [make_frame(self.world.A0, 1000.0)]
        self.wros.latest_wrist_frame = lambda: self.wros.frames[-1]
        r = self.get("/api/wrist/image.jpg?w=640")
        assert r.code == 200 and r.headers["Content-Type"] == "image/jpeg"
        assert self.post("/api/wrist/scene_remove").code == 200

    def test_no_wrist_panel_is_503(self):
        self.ctx.wrist = None
        assert self.post("/api/wrist/stop").code == 503
        assert self.j(self.get("/api/state"))["wrist"]["supported"] is False
