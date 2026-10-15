"""TCP pivot tab (ui/tcp_ui.py): the TcpManager with a fake jog / TF and the real tcp_pivot solver on synthetic poses.
No ROS graph.  Server endpoints: see the TestTcpEndpoints class at the end."""
import builtins
import hashlib
import json
import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from rebel_dual_arm_calibration import tcp_pivot as tp
from rebel_dual_arm_calibration.ui import tcp_ui as tu
from rebel_dual_arm_calibration.ui.jog import arm_configs

CFGS = arm_configs("single_left")
NAMES = list(CFGS["left"].joint_names)
R_TRUE = np.array([0.0059, 0.0955, 0.1325])         # the "real" tip in igus_tool0 (differs from the URDF by ~1.5 mm)
R_URDF = np.array([0.005937, 0.094121, 0.130699])    # the URDF value (TF igus_tool0 -> xeg32_tool_tip, and in the xacro)
C = np.array([0.40, 0.10, 0.05])                     # the fixed pivot in the base frame
AXIS = R_TRUE / np.linalg.norm(R_TRUE)

XACRO = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <joint name="igus_tool0_to_xeg32_tool_tip" type="fixed">
    <parent link="igus_tool0"/>
    <child link="xeg32_tool_tip"/>
    <origin xyz="0.005937 0.094121 0.130699" rpy="0 0 0"/>
  </joint>
</robot>
"""


def rot_to(a, b):
    """A rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


R0 = rot_to(AXIS, np.array([0.0, 0.0, -1.0]))       # knife axis straight down


def pose(R, r=R_TRUE, c=C, shift=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ r + np.asarray(shift, dtype=float)
    return T


def plan_poses(noise_mm=0.0, seed=1):
    """One pose per POSE_PLAN item: (role, T_base_tool0).  Exact data (+ optional noise) for the true r."""
    rng = np.random.default_rng(seed)
    out = []
    for item in tp.POSE_PLAN:
        R = R0 @ tp.plan_rotation(item, AXIS)
        T = pose(R, shift=rng.normal(0, noise_mm * 1e-3, 3) if noise_mm else (0, 0, 0))
        out.append((item["role"], T))
    return out


def link6_of(T_tool0):
    """T_base_link6 from T_base_tool0 (link6 -> igus_tool0: 0.0527 m along z, rotated pi about z)."""
    Rz = np.diag([-1.0, -1.0, 1.0])
    T = np.eye(4)
    T[:3, :3] = Rz
    T[:3, 3] = [0, 0, 0.0527]
    return T_tool0 @ np.linalg.inv(T)


class FakeTf:
    """tf_fn: the arm is at T_tool0 (None = no TF); every dynamic reading is `age` s old."""

    def __init__(self):
        self.T_tool0 = None
        self.age = 0.05
        self.r = R_URDF.copy()
        self.calls = []

    def __call__(self, parent, child):
        self.calls.append((parent, child))
        if child == "xeg32_tool_tip":
            T = np.eye(4)
            T[:3, 3] = self.r
            return T, 1e9, ""                          # static: its stamp is irrelevant
        if self.T_tool0 is None:
            return None, None, f"TF {parent} -> {child} not available"
        return (self.T_tool0 if child == "igus_tool0" else link6_of(self.T_tool0)), self.age, ""


class FakeJog:
    def __init__(self):
        self.busy = False
        self.spread = 0.0
        self.covered = True
        self.q = {n: 0.1 * (i + 1) for i, n in enumerate(NAMES)}
        self.none = False
        self.window_calls = []

    def status(self):
        return {"available": True, "busy": self.busy}

    def joint_window(self, arms, window_s):
        self.window_calls.append((list(arms), window_s))
        return None if self.none else (dict(self.q), self.spread, self.covered)


PAGE_A = "page-aaa111"
PAGE_B = "page-bbb222"


class PageTcp(tu.TcpManager):
    """The manager as the tests use it: the page id defaults to PAGE_A (the strict signature is tested separately)."""

    def set_checklist(self, items, page_id=PAGE_A):
        return super().set_checklist(items, page_id)

    def capture(self, role, note="", page_id=PAGE_A):
        return super().capture(role, note, page_id)

    def state(self, page_id=PAGE_A):
        return super().state(page_id)


class Env:
    def __init__(self, tmp, mode="single_left", with_xacro=True):
        self.dir = os.path.join(tmp, "session")
        os.makedirs(self.dir, exist_ok=True)
        self.xacro = os.path.join(tmp, "wrapper.urdf.xacro")
        if with_xacro:
            with open(self.xacro, "w") as f:
                f.write(XACRO)
        self.tf, self.jog = FakeTf(), FakeJog()
        self.now = [1000.0]
        self.mono = [0.0]
        self.session = self.dir
        self.mode = mode
        self.m = self.make()

    def make(self):
        return PageTcp(self.mode, arm_configs(self.mode), self.jog, lambda: self.session, tf_fn=self.tf,
                       xacro_path=self.xacro, wall_clock=lambda: self.now[0], clock=lambda: self.mono[0])

    def tick_all(self):
        self.m.set_checklist({k: True for k, _ in tu.CHECKLIST})

    def at(self, T):
        self.tf.T_tool0 = T
        self.now[0] += 10.0

    def cap(self, role, T):
        self.at(T)
        return self.m.capture(role)

    def fill(self, noise_mm=0.0, skip=()):
        for k, (role, T) in enumerate(plan_poses(noise_mm)):
            if k not in skip:
                self.cap(role, T)


@pytest.fixture
def env(tmp_path):
    e = Env(str(tmp_path))
    e.tick_all()
    return e


def refused(fn, *a, status=None, **kw):
    with pytest.raises(tu.TcpRefused) as ei:
        fn(*a, **kw)
    if status is not None:
        assert ei.value.status == status
    return ei.value.message


# ---------------------------------------------------------------------------------------------- checklist / gating

def test_capture_needs_every_checklist_item(tmp_path):
    e = Env(str(tmp_path))
    e.at(plan_poses()[0][1])
    assert "checklist" in refused(e.m.capture, "fit", status=409)
    items = [k for k, _ in tu.CHECKLIST]
    assert set(items) == {"ultrasonic_off", "fixture", "single_arm", "estop"}
    for k in items[:-1]:
        e.m.set_checklist({k: True})
        refused(e.m.capture, "fit", status=409)
    v = e.m.set_checklist({items[-1]: True})
    assert v["all_checked"] and all(i["checked"] for i in v["items"])
    assert "captured fit pose #1" in e.m.capture("fit")["message"]
    e.m.set_checklist({"estop": False})                    # un-ticking blocks the next capture again
    e.at(plan_poses()[1][1])
    refused(e.m.capture, "fit", status=409)


def test_unknown_checklist_key_and_bad_role_and_note(env):
    refused(env.m.set_checklist, {"nope": True}, status=400)
    env.at(plan_poses()[0][1])
    refused(env.m.capture, "other", status=400)
    refused(env.m.capture, "fit", "x" * 500, status=400)
    assert not os.path.exists(os.path.join(env.dir, tu.SAMPLES_FILE))


def test_dual_mode_is_refused_everywhere(tmp_path):
    e = Env(str(tmp_path), mode="dual")
    assert e.m.supported is False and e.m.arm is None
    for fn, args in ((e.m.set_checklist, ({"estop": True},)), (e.m.capture, ("fit",)), (e.m.delete, (0,)),
                     (e.m.solve, ()), (e.m.mechanical, (170.0,)), (e.m.write, ())):
        assert "single-arm modes only" in refused(fn, *args, status=400)
    st = e.m.state()
    assert st["supported"] is False and "single-arm modes only" in st["message"] and st["samples"] == []
    assert e.tf.calls == [] and not os.listdir(e.dir)


def test_stale_or_missing_tf_is_refused(env):
    T = plan_poses()[0][1]
    assert "not available" in refused(env.m.capture, "fit", status=409)         # no TF yet
    env.at(T)
    env.tf.age = 0.6
    msg = refused(env.m.capture, "fit", status=409)
    assert "stale" in msg and "0.60 s" in msg and "limit 0.5 s" in msg
    env.tf.age = None
    assert "age is unknown" in refused(env.m.capture, "fit", status=409)
    env.tf.age = 0.49
    env.m.capture("fit")                                                          # just inside the limit
    assert ("base_link", "igus_tool0") in env.tf.calls and ("base_link", "link6") in env.tf.calls


def test_not_stationary_is_refused_and_uses_the_two_second_window(env):
    env.at(plan_poses()[0][1])
    env.jog.spread = 0.0021
    msg = refused(env.m.capture, "fit", status=409)
    assert "not stationary" in msg and "2.1 mrad" in msg and "2.0 s" in msg
    assert env.jog.window_calls[-1] == (["left"], 2.0)
    env.jog.spread = 0.0019
    env.m.capture("fit")
    env.jog.covered = False
    env.at(plan_poses()[1][1])
    assert "less than 2.0 s of joint history" in refused(env.m.capture, "fit", status=409)
    env.jog.covered, env.jog.none = True, True
    assert "missing, stale or not finite" in refused(env.m.capture, "fit", status=409)


def test_jog_busy_and_no_session_dir_are_refused(env):
    env.at(plan_poses()[0][1])
    env.jog.busy = True
    assert "busy" in refused(env.m.capture, "fit", status=409)
    env.jog.busy = False
    env.session = None
    assert "session directory unknown" in refused(env.m.capture, "fit", status=503)


def test_duplicate_pose_is_refused_with_a_clear_text(env):
    (_, T0), (_, T1) = plan_poses()[:2]
    env.cap("fit", T0)
    Tn = T0.copy()
    Tn[:3, :3] = tp._rotvec_to_R([0, 0, 8.0]) @ T0[:3, :3]                         # 8 deg away
    env.at(Tn)
    msg = refused(env.m.capture, "validate", status=409)
    assert "too similar to pose #1 (fit)" in msg and "15 deg" in msg and "re-seat" in msg
    env.cap("fit", T1)                                                              # 30 deg away: fine
    assert len(env.m.state()["samples"]) == 2
    assert len(open(os.path.join(env.dir, tu.SAMPLES_FILE)).read().splitlines()) == 2


# ---------------------------------------------------------------------------------------------- persistence / delete

def test_capture_persists_json_lines_and_reloads_after_restart(env):
    plan = plan_poses()
    env.cap("fit", plan[0][1])
    env.cap("validate", plan[12][1])
    path = os.path.join(env.dir, tu.SAMPLES_FILE)
    lines = open(path).read().splitlines()
    assert len(lines) == 2
    d = json.loads(lines[0])
    assert d["arm"] == "left" and d["role"] == "fit" and d["stamp"] == 1010.0
    assert set(d["joints"]) == set(NAMES) and d["joints"][NAMES[0]] == pytest.approx(0.1)
    assert np.allclose(d["T_base_tool0"], plan[0][1]) and np.allclose(d["T_base_link6"], link6_of(plan[0][1]))
    m2 = env.make()                                                                # "UI restart"
    st = m2.state()
    assert [s["role"] for s in st["samples"]] == ["fit", "validate"] and st["n_fit"] == 1 and st["n_validate"] == 1
    assert st["file"] == path and st["session_dir"] == env.dir
    # the duplicate check also knows the reloaded poses
    m2.set_checklist({k: True for k, _ in tu.CHECKLIST})
    env.at(plan[0][1])
    assert "too similar to pose #1" in refused(m2.capture, "fit", status=409)


def test_delete_rewrites_the_file_and_keeps_a_bak(env):
    env.fill(skip=range(4, 16))
    path = os.path.join(env.dir, tu.SAMPLES_FILE)
    before = open(path).read()
    assert len(before.splitlines()) == 4
    res = env.m.delete(1)
    assert "deleted pose #2" in res["message"] and ".bak" in res["message"]
    after = open(path).read().splitlines()
    assert after == [before.splitlines()[i] for i in (0, 2, 3)]
    assert open(path + ".bak").read() == before
    assert [s["index"] for s in env.m.state()["samples"]] == [0, 1, 2]
    assert len(env.make().state()["samples"]) == 3                                  # persisted
    refused(env.m.delete, 3, status=400)
    refused(env.m.delete, -1, status=400)
    refused(env.m.delete, "1", status=400)
    refused(env.m.delete, True, status=400)
    assert open(path).read().splitlines() == after


def test_unreadable_lines_are_kept_and_warned_about(env):
    env.fill(skip=range(2, 16))
    path = os.path.join(env.dir, tu.SAMPLES_FILE)
    with open(path, "a") as f:
        f.write("this is not json\n")
    st = env.make().state()
    assert len(st["samples"]) == 2 and any("could not be read" in w for w in st["warnings"])
    m = env.make()
    m.delete(0)
    assert "this is not json" in open(path).read()                                  # a delete does not eat what it cannot parse


def test_external_edit_of_the_file_is_picked_up(env):
    env.fill(skip=range(2, 16))
    assert len(env.m.state()["samples"]) == 2
    path = os.path.join(env.dir, tu.SAMPLES_FILE)
    lines = open(path).read().splitlines()
    open(path, "w").write(lines[0] + "\n")
    assert len(env.m.state()["samples"]) == 1


# ---------------------------------------------------------------------------------------------- solve / display

def test_solve_display_fields(env):
    env.fill()
    out = env.m.solve()
    assert out["verdict"] in ("GOOD", "PASS") and "r = [" in out["message"]
    R = env.m.state()["result"]
    assert R["ok"] is True and R["stale"] is False and R["verdict"] in ("GOOD", "PASS")
    assert R["r_mm"] == pytest.approx(R_TRUE * 1e3, abs=0.05)
    assert R["current_r_mm"] == pytest.approx(R_URDF * 1e3)
    assert R["delta_to_current_mm"] == pytest.approx((R_TRUE - R_URDF) * 1e3, abs=0.05)
    assert R["c_base_mm"] == pytest.approx(C * 1e3, abs=0.05)
    assert R["n_fit"] == 11 and R["n_validate"] == 4
    assert len(R["sigma_xyz_mm"]) == 3 and len(R["sigma_axes_mm"]) == 3 and len(R["weak_axis_tool0"]) == 3
    assert R["rms_mm"] < 0.01 and R["heldout_rms_mm"] < 0.01 and R["loo_rms_mm"] < 0.01
    rows = {r["index"]: r for r in R["rows"]}
    assert len(rows) == 15
    assert all(rows[i]["role"] == "fit" and rows[i]["residual_mm"] is not None and rows[i]["loo_mm"] is not None
               for i in range(11))
    assert all(rows[i]["role"] == "validate" and rows[i]["heldout_mm"] is not None for i in range(11, 15))
    assert R["outliers"] == [] and R["rank"] == 3
    json.dumps(env.m.state(), allow_nan=False)                                      # the whole state is JSON-safe


def test_insufficient_and_fail_are_shown_with_reasons(env):
    env.fill(skip=range(5, 16))
    out = env.m.solve()
    assert out["verdict"] == "INSUFFICIENT"
    R = env.m.state()["result"]
    assert R["ok"] is False and R["reasons"] and R["verdict"] == "INSUFFICIENT"
    st = env.m.state()
    assert st["diff"] is None and "only GOOD / PASS" in st["diff_error"]             # no patch for an unusable result


def test_outlier_is_flagged_never_deleted(env):
    plan = plan_poses()
    for k, (role, T) in enumerate(plan):
        if k == 3:
            T = T.copy()
            T[:3, 3] += [0.012, -0.006, 0.004]                                      # the tip slipped out of the dimple
        env.cap(role, T)
    env.m.solve()
    st = env.m.state()
    R = st["result"]
    assert 3 in R["outliers"] and any(r["index"] == 3 and r["outlier"] for r in R["rows"])
    assert R["verdict"] in ("FAIL", "PASS") and len(st["samples"]) == 15            # nothing removed automatically
    assert len(open(os.path.join(env.dir, tu.SAMPLES_FILE)).read().splitlines()) == 15


def test_result_goes_stale_when_the_poses_change(env):
    env.fill(skip=(14,))
    env.m.solve()
    assert env.m.state()["result"]["stale"] is False
    env.cap(*plan_poses()[14])
    assert env.m.state()["result"]["stale"] is True
    env.m.solve()
    env.m.delete(0)
    assert env.m.state()["result"]["stale"] is True


def test_solve_without_fit_poses_is_refused(env):
    refused(env.m.solve, status=409)
    env.cap(*plan_poses()[12])                                                       # a validate pose only
    refused(env.m.solve, status=409)


# ---------------------------------------------------------------------------------------------- guidance

def test_live_guidance_and_pose_plan(env):
    plan = plan_poses()
    st = env.m.state()
    assert st["live"]["stationary"] is True and st["live"]["tf_ok"] is False
    assert [p["id"] for p in st["coverage"]["plan"]] == [i["id"] for i in tp.POSE_PLAN]
    assert "vertical" in st["coverage"]["suggestion"].lower()
    assert all(p["covered"] is False for p in st["coverage"]["plan"])
    env.cap(*plan[0])
    env.cap(*plan[1])
    env.at(plan[2][1])
    st = env.m.state()
    cov = {p["id"]: p["covered"] for p in st["coverage"]["plan"]}
    assert cov["vertical"] and cov["x+30"] and not cov["x-30"]
    L = st["live"]
    assert L["tf_ok"] and L["stationary"] and L["tilt_vertical_deg"] == pytest.approx(30.0, abs=0.5)
    assert L["tilt_deg"] == pytest.approx(30.0, abs=0.5) and L["nearest_index"] == 0 and L["nearest_deg"] == pytest.approx(30.0, abs=0.5)
    assert L["duplicate_of"] is None
    assert [s["tilt_vertical_deg"] for s in st["samples"]] == pytest.approx([0.0, 30.0], abs=0.5)
    env.at(plan[0][1])                                                               # back at pose 1: a capture would be refused
    assert env.m.state()["live"]["duplicate_of"] == 0
    env.jog.spread = 0.01
    L = env.m.state()["live"]
    assert L["stationary"] is False and "not stationary" in L["motion_problem"]


def test_current_r_from_tf_and_mismatch_with_the_source_xacro(env):
    st = env.m.state()
    assert st["current_r_mm"] == pytest.approx(R_URDF * 1e3) and "live robot_description" in st["current_r_source"]
    assert st["warnings"] == []
    env.tf.r = R_URDF + [0, 0, 0.002]                                                # stack not rebuilt after an xacro edit
    assert any("differs from the source xacro by 2.00 mm" in w for w in env.m.state()["warnings"])
    env.tf.r = None                                                                  # TF unavailable: fall back on the xacro
    env.m.tf_fn = lambda p, c: (None, None, "no tf")
    st = env.m.state()
    assert st["current_r_mm"] == pytest.approx(R_URDF * 1e3) and "xacro source" in st["current_r_source"]


# ---------------------------------------------------------------------------------------------- mechanical

def test_mechanical_check_before_and_after_a_solve(env):
    refused(env.m.mechanical, "170", status=400)
    refused(env.m.mechanical, float("nan"), status=400)
    refused(env.m.mechanical, 5.0, status=400)
    refused(env.m.mechanical, 900.0, status=400)
    out = env.m.mechanical(168.0)["mechanical"]
    assert out["length_mm"] == 168.0 and "current URDF" in out["r_source"]
    c = out["check"]
    assert c["predicted_current_mm"] == pytest.approx(183.4, abs=0.05) and c["delta_current_mm"] == pytest.approx(-15.4, abs=0.1)
    assert "text" in c and c["consistent_current"] is False
    env.fill()
    env.m.solve()                                                                    # the solved r is used from now on
    out = env.m.state()["mechanical"]
    assert out["r_source"] == "the solved r" and out["check"]["predicted_new_mm"] == pytest.approx(52.7 + R_TRUE[2] * 1e3, abs=0.1)
    assert env.m.state()["mechanical_note"] == {"current_urdf_mm": 183.4, "old_pivot_mm": 168.0}


# ---------------------------------------------------------------------------------------------- write / xacro patch

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_write_makes_the_yaml_and_shows_the_diff_but_never_touches_the_xacro(env, monkeypatch):
    h0, m0 = sha(env.xacro), os.stat(env.xacro).st_mtime_ns
    os.chmod(env.xacro, 0o444)                                                       # a write would even fail
    real_open = builtins.open
    writes = []

    def spy(file, mode="r", *a, **k):
        if os.path.abspath(str(file)) == os.path.abspath(env.xacro) and set(mode) & set("wax+"):
            writes.append((file, mode))
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    refused(env.m.write, status=409)                                                 # nothing solved yet
    env.fill()
    env.m.solve()
    st = env.m.state()
    assert st["diff"].startswith("--- a/wrapper.urdf.xacro")                         # (repo-relative only if the path has /src/)
    assert "-    <origin xyz=\"0.005937 0.094121 0.130699\"" in st["diff"]
    assert f"+    <origin xyz=\"{R_TRUE[0]:.6f} {R_TRUE[1]:.6f} {R_TRUE[2]:.6f}\"" in st["diff"]
    out = env.m.write()
    path = os.path.join(env.dir, "tcp_result_left.yaml")
    assert out["path"] == path and "NOT changed" in out["message"]
    import yaml
    doc = yaml.safe_load(real_open(path).read())
    assert doc["kind"] == tp.RESULT_YAML_KIND and doc["arm"] == "left" and doc["verdict"] in ("GOOD", "PASS")
    assert doc["r_tool0_mm"] == pytest.approx(R_TRUE * 1e3, abs=0.05) and doc["xacro_path"] == env.xacro
    assert doc["urdf_hash"] == hashlib.sha256(XACRO.encode()).hexdigest()[:16]
    assert doc["current_r_m"] == pytest.approx(R_URDF)
    st = env.m.state()
    assert st["written"]["path"] == path and st["diff"]
    assert any("by hand" in s.lower() or "apply" in s.lower() for s in st["apply_steps"])
    assert "re-based" in st["rebase_warning"] and "solve --tcp-file" in st["rebase_warning"]
    assert any("colcon build" in s and "igus_rebel_description" in s and "igus_rebel_moveit_config" in s
               for s in st["apply_steps"])
    assert writes == [] and sha(env.xacro) == h0 and os.stat(env.xacro).st_mtime_ns == m0
    env.m.write()                                                                    # again: the old file is kept as .bak
    assert os.path.exists(path + ".bak")
    assert sha(env.xacro) == h0


def test_write_is_refused_for_stale_fail_or_insufficient_results(env):
    env.fill(skip=range(5, 16))
    env.m.solve()
    assert "INSUFFICIENT" in refused(env.m.write, status=409)
    assert not os.path.exists(os.path.join(env.dir, "tcp_result_left.yaml"))
    env.fill(skip=range(0, 5))
    env.m.solve()
    env.cap("fit", tp_extra_pose())                                                  # the poses change after the solve
    assert "poses changed" in refused(env.m.write, status=409)
    assert not os.path.exists(os.path.join(env.dir, "tcp_result_left.yaml"))


def tp_extra_pose():
    R = R0 @ tp._rotvec_to_R([0, 0, 0])
    return pose(tp._rotvec_to_R([10, -12, 0]) @ R)


def test_write_refuses_a_failed_verdict(env):
    plan = plan_poses()
    for k, (role, T) in enumerate(plan):
        T = T.copy()
        T[:3, 3] += np.random.default_rng(k).normal(0, 0.004, 3)                     # 4 mm of scatter: the tip is not seated
        env.cap(role, T)
    env.m.solve()
    R = env.m.state()["result"]
    assert R["verdict"] == "FAIL"
    assert "FAIL" in refused(env.m.write, status=409)
    assert not os.path.exists(os.path.join(env.dir, "tcp_result_left.yaml"))


def test_missing_xacro_still_works_but_has_no_diff_and_no_write(tmp_path):
    e = Env(str(tmp_path), with_xacro=False)
    e.tick_all()
    e.fill()
    e.m.solve()
    st = e.m.state()
    assert st["result"]["ok"] and st["diff"] is None and "cannot read the xacro" in st["diff_error"]
    assert "cannot read the xacro" in refused(e.m.write, status=503)
    assert not os.path.exists(os.path.join(e.dir, "tcp_result_left.yaml"))


def test_repo_relative_patch_path(tmp_path):
    e = Env(str(tmp_path))
    e.m.xacro_path = "/ws/src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro"
    assert e.m._patch_path() == "src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro"
    e.m.xacro_path = "/tmp/other.xacro"
    assert e.m._patch_path() == "other.xacro"


def test_parse_xacro_r_and_default_path():
    assert tu.parse_xacro_r(XACRO) == pytest.approx(R_URDF)
    assert tu.parse_xacro_r("<robot/>") is None
    assert tu.DEFAULT_XACRO_PATH == ("/home/farhan/igus_rebel_ros2/src/igus_rebel_description/urdf/"
                                     "igus_rebel_xeg32_wrapper.urdf.xacro")


def test_no_code_path_opens_the_xacro_for_writing():
    src = open(tu.__file__).read()
    assert "xacro_path, \"w" not in src and "open(self.xacro_path, \"w" not in src
    assert src.count("self.xacro_path") >= 3
    for line in src.splitlines():
        if "open(" in line and "xacro" in line:
            assert "encoding=" in line and '"w' not in line and '"a' not in line


# ---------------------------------------------------------------------------------------------- checklist per page

def test_the_409_for_an_incomplete_checklist_lists_every_unticked_item(tmp_path):
    e = Env(str(tmp_path))
    e.at(plan_poses()[0][1])
    msg = refused(e.m.capture, "fit", status=409)
    assert "Not ticked (4)" in msg and msg.count(" | ") == 3
    assert all(t in msg for _, t in tu.CHECKLIST)
    e.m.set_checklist({"fixture": True, "estop": True})
    msg = refused(e.m.capture, "fit", status=409)
    assert "Not ticked (2)" in msg and tu.CHECKLIST[0][1] in msg and tu.CHECKLIST[2][1] in msg and tu.CHECKLIST[3][1] not in msg


def test_the_checklist_is_per_page_a_reload_or_restart_starts_unticked(env):
    assert env.m.state(PAGE_A)["checklist"]["all_checked"] is True
    assert env.m.state(PAGE_B)["checklist"]["all_checked"] is False             # another page / a reload (new page id)
    assert env.m.state(None)["checklist"]["all_checked"] is False
    env.at(plan_poses()[0][1])
    assert "Not ticked (4)" in refused(env.m.capture, "fit", "", PAGE_B, status=409)
    env.m.capture("fit", "", PAGE_A)
    env.m.set_checklist({k: True for k, _ in tu.CHECKLIST}, PAGE_B)
    assert env.m.state(PAGE_B)["checklist"]["all_checked"] is True
    assert env.m.state(PAGE_A)["checklist"]["all_checked"] is True              # B's ticks do not touch A's
    restarted = env.make()                                                      # UI restart: nothing ticked on any page
    assert restarted.state(PAGE_A)["checklist"]["all_checked"] is False
    assert restarted.state(PAGE_B)["checklist"]["all_checked"] is False


def test_strict_manager_needs_a_page_id(tmp_path):
    e = Env(str(tmp_path))
    m = tu.TcpManager(e.mode, arm_configs(e.mode), e.jog, lambda: e.session, tf_fn=e.tf, xacro_path=e.xacro)
    e.at(plan_poses()[0][1])
    for bad in (None, "", 5):
        refused(m.set_checklist, {"estop": True}, bad, status=400)
        refused(m.capture, "fit", "", bad, status=400)
    assert m.state()["checklist"]["all_checked"] is False


def test_ticks_expire_after_30_minutes_without_a_capture(env):
    plan = plan_poses()
    assert env.m.state()["checklist"]["ttl_min"] == 30.0
    env.mono[0] += 29 * 60
    env.cap("fit", plan[0][1])                                                  # 29 min: still valid, and the capture renews it
    env.mono[0] += 29 * 60                                                      # 58 min after the ticks, 29 after the capture
    env.cap("fit", plan[1][1])
    env.mono[0] += 31 * 60                                                      # 31 min without a capture
    st = env.m.state()["checklist"]
    assert st["all_checked"] is False and st["expired"] is True and not any(i["checked"] for i in st["items"])
    env.at(plan[2][1])
    msg = refused(env.m.capture, "fit", status=409)
    assert "expired" in msg and "tick it again" in msg and "Not ticked (4)" in msg
    env.tick_all()                                                              # ticking again renews it
    assert env.m.state()["checklist"]["expired"] is False
    env.m.capture("fit")
    env.mono[0] += 29 * 60
    env.at(plan[3][1])
    env.m.capture("fit")


def test_pages_are_bounded(env):
    for i in range(tu.MAX_PAGES + 8):
        env.mono[0] += 1
        env.m.set_checklist({"estop": True}, f"page-{i:06d}")
    assert len(env.m._pages) == tu.MAX_PAGES


def test_live_blockers_say_why_a_capture_would_be_refused(env):
    plan = plan_poses()
    assert any("not available" in b for b in env.m.state()["live"]["blockers"])           # no TF yet
    env.at(plan[0][1])
    assert env.m.state()["live"]["blockers"] == []
    env.jog.spread = 0.01
    env.jog.busy = True
    b = env.m.state()["live"]["blockers"]
    assert any("busy" in x for x in b) and any("not stationary" in x for x in b)
    env.jog.spread, env.jog.busy = 0.0, False
    env.cap("fit", plan[0][1])
    b = env.m.state()["live"]["blockers"]
    assert len(b) == 1 and "too similar to pose #1" in b[0] and "15 deg" in b[0]
    env.session = None
    assert any("session directory unknown" in x for x in env.m.state()["live"]["blockers"])


# ---------------------------------------------------------------------------------------------- stale result / numbering

def test_a_stale_result_shows_no_per_pose_numbers_outliers_reasons_or_patch(env):
    for k, (role, T) in enumerate(plan_poses()):
        if k == 7:
            T = T.copy()
            T[:3, 3] += [0.012, -0.006, 0.004]
        env.cap(role, T)
    env.m.solve()
    R = env.m.state()["result"]
    assert R["stale"] is False and R["outliers"] == [7] and R["rows"] and R["reasons"]
    env.m.mechanical(183.0)
    env.m.delete(3)                                                              # the old row 8 is now row 7
    st = env.m.state()
    R = st["result"]
    assert R["stale"] is True and R["rows"] == [] and R["outliers"] == [] and R["reasons"] == []
    assert set(R) == {"stale", "verdict", "rows", "outliers", "reasons", "time"}
    assert st["diff"] is None and "solve again" in st["diff_error"]
    assert "STALE" in st["mechanical"]["r_source"]
    env.m.solve()
    R = env.m.state()["result"]
    assert R["stale"] is False and R["rows"] and len(R["rows"]) == 14


def test_solver_reasons_use_the_same_hash_n_numbering_as_the_sample_list(env):
    plan = plan_poses()
    # a validate pose first: the position in the list (not the index in the fit list) is the number
    env.cap(*plan[12])
    for k, (role, T) in enumerate(plan[:11]):
        if k == 4:
            T = T.copy()
            T[:3, 3] += [0.012, -0.006, 0.004]
        env.cap(role, T)
    env.m.solve()
    R = env.m.state()["result"]
    assert 5 in R["outliers"]                                                    # sample #6 (1 validate + fit 5 -> position 6)
    assert any("pose #6 (fit) is an outlier" in r for r in R["reasons"]), R["reasons"]
    import re
    assert not any(re.search(r"\(index|fit poses? \d", r) for r in R["reasons"])           # no mixed 0/1-based numbering
    assert [r["index"] for r in R["rows"] if r["outlier"]] == R["outliers"]
    for i in R["outliers"]:                                                      # every flagged row has its "#N" reason
        assert any(f"pose #{i + 1} (fit) is an outlier" in r for r in R["reasons"]), (i, R["reasons"])


# ---------------------------------------------------------------------------------------------- server endpoints

from test_ui_server import Base, TOKEN, PAGE   # noqa: E402  (the shared fake bridge / auth / app)

TCP_PATHS = ("checklist", "capture", "delete", "solve", "mechanical", "write")


class TcpBase(Base):
    mode = "single_left"

    def get_app(self):
        app = super().get_app()
        self.tmp = tempfile.mkdtemp(prefix="tcp_ui_test_")
        self.env = Env(self.tmp, mode=self.mode)
        self.bridge.status = dict(self.bridge.status, session_dir=self.env.dir)
        self.ctx.tcp = tu.TcpManager(self.mode, arm_configs(self.mode), self.jog_adapter(), self.ctx._session_dir,
                                     tf_fn=self.env.tf, xacro_path=self.env.xacro, wall_clock=lambda: 2000.0)
        return app

    def jog_adapter(self):
        return self.env.jog

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tick_all(self):
        r = self.post("/api/tcp/checklist", {"items": {k: True for k, _ in tu.CHECKLIST}})
        assert r.code == 200, r.body
        return r


class TestTcpEndpoints(TcpBase):
    def test_all_endpoints_need_the_token_and_a_local_origin(self):
        for p in TCP_PATHS:
            r = self.fetch(f"/api/tcp/{p}", method="POST", body="{}")
            assert r.code == 401 and self.j(r)["ok"] is False
            r = self.post(f"/api/tcp/{p}", {}, headers=self.hdr(Origin="http://evil.example"))
            assert r.code == 403
        assert self.env.jog.window_calls == [] and not os.listdir(self.env.dir)

    def test_the_checklist_is_per_page_and_needs_a_page_id(self):
        self.tick_all()                                                         # page-abc123 (PAGE)
        other = {"X-Token": TOKEN, "X-Page-Id": "page-def456"}
        st = self.j(self.get("/api/state", headers=other))["tcp"]
        assert st["checklist"]["all_checked"] is False                          # another page / a reload sees nothing ticked
        assert self.j(self.get("/api/state"))["tcp"]["checklist"]["all_checked"] is True
        assert self.j(self.get("/api/state", headers={"X-Token": TOKEN}))["tcp"]["checklist"]["all_checked"] is False
        self.env.tf.T_tool0 = plan_poses()[0][1]
        r = self.post("/api/tcp/capture", {"role": "fit"}, headers=other)
        assert r.code == 409 and "Not ticked (4)" in self.j(r)["message"]
        noid = {"X-Token": TOKEN}
        assert self.post("/api/tcp/checklist", {"items": {"estop": True}}, headers=noid).code == 400
        assert self.post("/api/tcp/capture", {"role": "fit"}, headers=noid).code == 400
        assert self.post("/api/tcp/capture", {"role": "fit"}).code == 200

    def test_state_carries_the_tcp_block(self):
        st = self.j(self.get("/api/state"))["tcp"]
        assert st["supported"] is True and st["arm"] == "left" and st["checklist"]["all_checked"] is False
        assert len(st["checklist"]["items"]) == 4 and st["samples"] == [] and st["result"] is None
        assert st["thresholds"]["standstill_s"] == 2.0 and st["xacro_path"] == self.env.xacro
        assert st["apply_steps"] and "re-based" in st["rebase_warning"]

    def test_full_workflow_over_http(self):
        r = self.post("/api/tcp/capture", {"role": "fit"})                              # checklist not ticked
        assert r.code == 409 and "checklist" in self.j(r)["message"]
        self.tick_all()
        for k, (role, T) in enumerate(plan_poses()):
            self.env.tf.T_tool0 = T
            r = self.post("/api/tcp/capture", {"role": role, "note": f"p{k}"})
            assert r.code == 200 and self.j(r)["ok"] is True, r.body
        body = self.j(r)
        assert body["tcp"]["n_fit"] == 11 and body["tcp"]["n_validate"] == 4 and "index" in body
        r = self.post("/api/tcp/capture", {"role": "fit"})                              # same pose again
        assert r.code == 409 and "too similar" in self.j(r)["message"]
        r = self.post("/api/tcp/solve")
        assert r.code == 200 and self.j(r)["verdict"] in ("GOOD", "PASS")
        assert self.j(r)["tcp"]["result"]["r_mm"] == pytest.approx(R_TRUE * 1e3, abs=0.05)
        r = self.post("/api/tcp/mechanical", {"length_mm": 183.5})
        assert r.code == 200 and self.j(r)["mechanical"]["r_source"] == "the solved r"
        before = sha(self.env.xacro)
        r = self.post("/api/tcp/write")
        assert r.code == 200 and os.path.isfile(self.j(r)["path"]) and sha(self.env.xacro) == before
        assert self.j(r)["tcp"]["written"]["path"].endswith("tcp_result_left.yaml") and self.j(r)["tcp"]["diff"]
        r = self.post("/api/tcp/delete", {"index": 2})
        assert r.code == 200 and self.j(r)["tcp"]["n_fit"] == 10 and self.j(r)["tcp"]["result"]["stale"] is True
        assert os.path.exists(os.path.join(self.env.dir, tu.SAMPLES_FILE + ".bak"))

    def test_bad_bodies(self):
        for body in ({}, {"items": {}}, {"items": {"estop": "yes"}}, {"items": []}):
            assert self.post("/api/tcp/checklist", body).code == 400
        assert self.post("/api/tcp/checklist", {"items": {"bogus": True}}).code == 400
        assert self.post("/api/tcp/capture", {}).code == 400
        assert self.post("/api/tcp/capture", {"role": 3}).code == 400
        assert self.post("/api/tcp/capture", {"role": "nope"}).code == 400
        for body in ({}, {"index": "1"}, {"index": 7}, {"index": 1.5}):
            assert self.post("/api/tcp/delete", body).code == 400
        for body in ({}, {"length_mm": "x"}, {"length_mm": 5}, {"length_mm": None}):
            assert self.post("/api/tcp/mechanical", body).code == 400
        assert self.fetch("/api/tcp/solve", method="POST", body="[1]", headers=self.hdr()).code == 400
        assert self.post("/api/tcp/solve").code == 409
        assert self.post("/api/tcp/write").code == 409
        assert not os.path.exists(os.path.join(self.env.dir, tu.SAMPLES_FILE))

    def test_a_broken_manager_is_a_500_and_does_not_break_the_state_poll(self):
        def boom(*a, **k):
            raise RuntimeError("boom")
        self.ctx.tcp.solve = boom
        r = self.post("/api/tcp/solve")
        assert r.code == 500 and "boom" in self.j(r)["message"]
        self.ctx.tcp.state = boom
        st = self.j(self.get("/api/state"))
        assert st["ok"] is True and st["tcp"]["supported"] is False and "boom" in st["tcp"]["message"]


class TestTcpDualMode(TcpBase):
    mode = "dual"

    def test_every_endpoint_says_single_arm_only(self):
        st = self.j(self.get("/api/state"))["tcp"]
        assert st["supported"] is False and "single-arm modes only" in st["message"]
        for p, body in (("checklist", {"items": {"estop": True}}), ("capture", {"role": "fit"}), ("delete", {"index": 0}),
                        ("solve", {}), ("mechanical", {"length_mm": 170}), ("write", {})):
            r = self.post(f"/api/tcp/{p}", body)
            assert r.code == 400 and "single-arm modes only" in self.j(r)["message"], p
        assert not os.listdir(self.env.dir) and self.env.tf.calls == []


class TestDefaultContext(Base):
    """A Context built without a TCP manager (the default of the other tests) answers instead of failing."""
    mode = "single_left"

    def test_default_manager_has_no_tf_and_refuses_a_capture(self):
        self.post("/api/tcp/checklist", {"items": {k: True for k, _ in tu.CHECKLIST}})
        r = self.post("/api/tcp/capture", {"role": "fit"})
        assert r.code in (409, 503) and self.j(r)["ok"] is False
        assert self.j(self.get("/api/state"))["tcp"]["supported"] is True
