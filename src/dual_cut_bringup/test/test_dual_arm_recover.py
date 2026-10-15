"""Pure-logic tests of scripts/dual_arm_recover.py: ordering, root selection, refusal rules, stillness."""
import importlib.util
import math
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "dual_arm_recover.py")
_spec = importlib.util.spec_from_file_location("dual_arm_recover", _SCRIPT)
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)

MOCK = "mock_components/GenericSystem"
REAL = "Igus/Rebel"


def view(arm, hw="active", ctrl="active", cls=MOCK):
    return rec.ArmView(arm=arm, hw_state=hw, hw_class=cls, controller_state=ctrl)


def views(left=None, right=None):
    return {"left": left or view("left"), "right": right or view("right")}


ROOT_LEFT = ("[ros2_control_node-1] [ERROR] [Igus::Rebel]: Fault group 'dual_arm_table': component "
             "'arm_left_hw' faulted (robot controller reports a stop after the arm was enabled: joint 2 error "
             "bits 0x01); stopped 1 other group member(s). This component is latched as well")
PEER_RIGHT = ("[ERROR] [Igus::Rebel]: Hardware component 'arm_right_hw' stopped because fault-group member "
              "'arm_left_hw' faulted: robot controller reports a stop. Jog forced to zero and latched; recover")
ROOT_RIGHT = ROOT_LEFT.replace("arm_left_hw", "arm_right_hw")
BLOCKING = ("Refusing to activate hardware component 'arm_right_hw': fault-group 'dual_arm_table' member "
            "'arm_left_hw' still holds an un-recovered root-cause fault (x).")
RECOVERED = "Hardware component 'arm_left_hw' recovered from a latched error: re-sent SetActive"
CLEARED = "Fault group 'dual_arm_table': 'arm_left_hw' is enabled with clean STATUS again; its root-cause fault is cleared"


# --- naming -----------------------------------------------------------------------------------------------

def test_names_match_the_driver_stack():
    assert rec.hw_name("left") == "arm_left_hw"
    assert rec.controller_name("right") == "arm_right_trajectory_controller"
    assert rec.reset_service("left") == "/igus_rebel_left/reset_and_enable"
    assert rec.joint_names("right")[0] == "arm_right_joint1" and len(rec.joint_names("right")) == 6


# --- driver log parsing / root cause ----------------------------------------------------------------------

def test_root_and_peer_lines_name_the_same_root():
    events = rec.parse_fault_events([ROOT_LEFT, PEER_RIGHT])
    assert [e.kind for e in events] == ["ROOT", "PEER"]
    assert rec.active_root_causes(events) == ["arm_left_hw"]
    assert rec.detected_root_arms(events) == ["left"]


def test_peer_line_alone_is_enough():
    assert rec.detected_root_arms(rec.parse_fault_events([PEER_RIGHT])) == ["left"]


def test_blocking_refusal_names_the_root():
    assert rec.detected_root_arms(rec.parse_fault_events([BLOCKING])) == ["left"]


def test_recovered_root_is_not_a_root_any_more():
    events = rec.parse_fault_events([ROOT_LEFT, PEER_RIGHT, CLEARED])
    assert rec.detected_root_arms(events) == []
    assert rec.detected_root_arms(rec.parse_fault_events([ROOT_LEFT, RECOVERED])) == []


def test_new_fault_after_recovery_is_the_root():
    events = rec.parse_fault_events([ROOT_LEFT, CLEARED, ROOT_RIGHT])
    assert rec.detected_root_arms(events) == ["right"]


def test_unrelated_lines_are_ignored():
    assert rec.parse_fault_events(["INFO something else", "", "Fault group mentioned loosely"]) == []


def test_two_unrecovered_roots_are_both_reported():
    events = rec.parse_fault_events([ROOT_LEFT, ROOT_RIGHT])
    assert sorted(rec.detected_root_arms(events)) == ["left", "right"]


# --- recovery plan: ordering and root selection ---------------------------------------------------------------

def both_latched():
    return views(view("left", hw="unconfigured", ctrl="inactive"), view("right", hw="unconfigured", ctrl="inactive"))


def test_root_is_recovered_first_left():
    plan = rec.plan_recovery(both_latched(), "auto", ["left"])
    assert plan.root == "left" and plan.order == ["left", "right"]


def test_root_is_recovered_first_right():
    plan = rec.plan_recovery(both_latched(), "auto", ["right"])
    assert plan.root == "right" and plan.order == ["right", "left"]


def test_explicit_root_wins_without_log():
    assert rec.plan_recovery(both_latched(), "right", []).order == ["right", "left"]
    assert rec.plan_recovery(both_latched(), "left", []).order == ["left", "right"]


def test_explicit_root_conflicting_with_log_warns_but_follows_operator():
    plan = rec.plan_recovery(both_latched(), "right", ["left"])
    assert plan.order == ["right", "left"]
    assert any("WARNING" in note for note in plan.notes)


def test_auto_without_detection_requires_root_when_both_need_recovery():
    with pytest.raises(rec.Refused, match="--root"):
        rec.plan_recovery(both_latched(), "auto", [])


def test_auto_with_ambiguous_detection_requires_root():
    with pytest.raises(rec.Refused, match="--root"):
        rec.plan_recovery(both_latched(), "auto", ["left", "right"])


def test_auto_single_latched_arm_needs_no_detection():
    v = views(view("left"), view("right", hw="inactive", ctrl="inactive"))
    plan = rec.plan_recovery(v, "auto", [])
    assert plan.order == ["right"] and plan.root == "right"


def test_healthy_arm_is_not_touched_even_when_it_is_the_named_root():
    v = views(view("left"), view("right", hw="unconfigured", ctrl="inactive"))
    assert rec.plan_recovery(v, "left", []).order == ["right"]


def test_nothing_to_do_when_both_hardware_active():
    plan = rec.plan_recovery(views(view("left", ctrl="inactive"), view("right", ctrl="inactive")), "auto", [])
    assert plan.order == [] and plan.notes   # recover never activates a controller of an enabled ACTIVE component


@pytest.mark.parametrize("bad", ["finalized", "unknown", "activating", "deactivating"])
def test_unrecoverable_or_transitional_hardware_is_refused(bad):
    v = views(view("left", hw=bad), view("right"))
    with pytest.raises(rec.Refused):
        rec.plan_recovery(v, "auto", [])


def test_missing_component_or_controller_is_refused():
    with pytest.raises(rec.Refused, match="not found"):
        rec.plan_recovery(views(view("left", hw=None), view("right")), "auto", [])
    with pytest.raises(rec.Refused, match="not found"):
        rec.plan_recovery(views(view("left", hw="unconfigured", ctrl=None), view("right")), "auto", [])


def test_unconfigured_or_finalized_controller_is_refused():
    with pytest.raises(rec.Refused, match="controller"):
        rec.plan_recovery(views(view("left", hw="inactive", ctrl="unconfigured"), view("right")), "auto", [])


def test_bad_root_argument_is_refused():
    with pytest.raises(rec.Refused):
        rec.plan_recovery(both_latched(), "both", [])


def test_step_order_per_arm_and_overall():
    v = views(view("left", hw="unconfigured", ctrl="inactive", cls=REAL),
              view("right", hw="inactive", ctrl="active", cls=REAL))
    plan = rec.plan_recovery(v, "left", [])
    steps = rec.recovery_steps(v, plan)
    assert [s for a, s in steps if a == "left"] == [
        "reset_and_enable", "hw_inactive", "hw_active", "ensure_enabled", "wait_root_cleared"]
    # hardware NOT active: Humble rejects deactivating the controller, so it is deactivated only after hw_active
    assert [s for a, s in steps if a == "right"] == [
        "reset_and_enable", "hw_inactive", "hw_active", "controller_deactivate", "ensure_enabled"]
    # whole root arm first, then the peer, and the controllers only at the very end for all arms
    arms = [a for a, _ in steps]
    assert arms == ["left"] * 5 + ["right"] * 5 + ["all"] * 3
    assert [s for a, s in steps if a == "all"] == ["stillness", "identity", "controllers_activate"]


def test_mock_steps_have_no_enable_wait_and_no_identity():
    v = views(view("left", hw="unconfigured", ctrl="inactive"), view("right", hw="unconfigured", ctrl="inactive"))
    steps = rec.recovery_steps(v, rec.plan_recovery(v, "left", []))
    assert "ensure_enabled" not in [s for _, s in steps] and "identity" not in [s for _, s in steps]


def test_active_arm_with_disabled_drives_needs_only_the_enable_step():
    right = view("right", hw="active", ctrl="active", cls=REAL)
    right.drives_disabled = True
    v = views(view("left", hw="unconfigured", ctrl="inactive", cls=REAL), right)
    plan = rec.plan_recovery(v, "right", [])
    assert plan.order == ["right", "left"]
    steps = rec.recovery_steps(v, plan)
    assert [s for a, s in steps if a == "right"] == ["controller_deactivate", "ensure_enabled", "wait_root_cleared"]


def test_active_but_disabled_arm_alone_is_a_recovery_target_only_on_real_hardware():
    right = view("right", hw="active", ctrl="active", cls=REAL)
    right.drives_disabled = True
    plan = rec.plan_recovery(views(view("left", cls=REAL), right), "auto", [])
    assert plan.order == ["right"]
    mock_right = view("right")
    mock_right.drives_disabled = True
    assert rec.plan_recovery(views(view("left"), mock_right), "auto", []).order == []


def test_active_but_disabled_and_other_latched_needs_root_for_auto():
    right = view("right", hw="active", ctrl="active", cls=REAL)
    right.drives_disabled = True
    v = views(view("left", hw="unconfigured", ctrl="inactive", cls=REAL), right)
    with pytest.raises(rec.Refused, match="--root"):
        rec.plan_recovery(v, "auto", [])
    assert rec.plan_recovery(v, "auto", ["right"]).order == ["right", "left"]


def test_named_active_root_on_real_hardware_is_rechecked_even_without_log_evidence():
    v = views(view("left", hw="unconfigured", ctrl="inactive", cls=REAL), view("right", cls=REAL))
    plan = rec.plan_recovery(v, "right", [])
    assert plan.order == ["right", "left"] and any("re-checked" in n for n in plan.notes)


# --- --yes refusal rules ------------------------------------------------------------------------------------------

def test_yes_allowed_only_for_mock():
    ok, _ = rec.yes_allowed([view("left"), view("right")])
    assert ok


def test_yes_refused_for_real_hardware_even_if_one_arm_is_real():
    ok, reason = rec.yes_allowed([view("left", cls=REAL), view("right")])
    assert not ok and "arm_left" in reason and REAL in reason
    ok, _ = rec.yes_allowed([view("left", cls=REAL), view("right", cls=REAL)])
    assert not ok


def test_yes_refused_when_class_unknown():
    ok, reason = rec.yes_allowed([view("left", cls=None), view("right")])
    assert not ok and "unknown" in reason
    ok, _ = rec.yes_allowed([view("left")])   # one arm missing
    assert not ok


@pytest.mark.parametrize("answer,expected", [("yes", True), (" yes\n", True), ("y", False), ("YES", False),
                                              ("yes please", False), ("", False), (None, False)])
def test_only_exact_yes_confirms(answer, expected):
    assert rec.confirmed(answer) is expected


# --- reset_and_enable reply -------------------------------------------------------------------------------------

def test_reset_and_enable_latched_refusal_is_expected():
    verdict, _ = rec.interpret_reset_and_enable(
        False, "refused: hardware error is latched; recover with 'ros2 control set_hardware_component_state x inactive'")
    assert verdict == "skipped"
    assert rec.interpret_reset_and_enable(True, "drives enabled")[0] == "ok"
    assert rec.interpret_reset_and_enable(False, "refused: STATUS is stale or missing")[0] == "failed"


# --- standing still -----------------------------------------------------------------------------------------------

def samples(arm, count=30, dt=0.03, start=100.0, drift=0.0, velocity=0.0, missing=(), nan_vel=False):
    out = []
    names = [n for n in rec.joint_names(arm) if n not in missing]
    for i in range(count):
        out.append({"t": start + i * dt,
                    "pos": {n: 0.1 * k + drift * i for k, n in enumerate(names)},
                    "vel": {n: (math.nan if nan_vel else velocity) for n in names}})
    return out


def end_time(s):
    return s[-1]["t"] + 0.05


def test_still_arm_is_accepted():
    s = samples("left")
    ok, reason = rec.assess_stillness(s, "left", end_time(s))
    assert ok, reason


def test_nan_velocity_falls_back_to_position_criterion():
    s = samples("left", nan_vel=True)
    assert rec.assess_stillness(s, "left", end_time(s))[0]


def test_moving_arm_is_rejected_by_position():
    s = samples("left", drift=0.001)   # 0.029 rad over the window
    ok, reason = rec.assess_stillness(s, "left", end_time(s))
    assert not ok and "moved" in reason


def test_moving_arm_is_rejected_by_velocity():
    s = samples("left", velocity=0.1)
    ok, reason = rec.assess_stillness(s, "left", end_time(s))
    assert not ok and "velocity" in reason


def test_stale_samples_are_rejected():
    s = samples("left")
    ok, reason = rec.assess_stillness(s, "left", end_time(s) + 2.0)
    assert not ok and "old" in reason


def test_missing_joint_or_too_few_samples_is_rejected():
    s = samples("left", missing=("arm_left_joint3",))
    assert not rec.assess_stillness(s, "left", end_time(s))[0]
    s = samples("left", count=2)
    assert not rec.assess_stillness(s, "left", end_time(s))[0]
    assert not rec.assess_stillness([], "left", 0.0)[0]


def test_other_arm_joints_do_not_count():
    s = samples("right")
    assert not rec.assess_stillness(s, "left", end_time(s))[0]


def test_short_window_is_rejected():
    s = samples("left", count=6, dt=0.01)
    ok, reason = rec.assess_stillness(s, "left", end_time(s))
    assert not ok and "cover" in reason


def test_non_finite_position_is_rejected():
    s = samples("left")
    s[-1]["pos"]["arm_left_joint1"] = math.nan
    s2 = [x for x in s]
    # a NaN sample is dropped as incomplete; the rest is still fine, but all-NaN is rejected
    assert rec.assess_stillness(s2, "left", end_time(s2))[0]
    for x in s2:
        x["pos"]["arm_left_joint1"] = math.nan
    assert not rec.assess_stillness(s2, "left", end_time(s2))[0]


def test_parser_subcommands():
    parser = rec.build_parser()
    assert parser.parse_args(["recover"]).root == "auto"
    assert parser.parse_args(["recover", "--root", "right", "--dry-run"]).dry_run
    assert parser.parse_args(["activate", "--yes"]).yes
    with pytest.raises(SystemExit):
        parser.parse_args(["recover", "--root", "middle"])
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --- driver drive-state log lines -----------------------------------------------------------------------------

DISABLED_RIGHT = ("[igus_rebel.arm_right_hw]: Robot drives disabled (STATUS reports 'Motor not enabled', joint mask "
                  "0x3f) while the connection is healthy: the arm will not move. Call 'ros2 service call "
                  "/igus_rebel_right/reset_and_enable std_srvs/srv/Trigger' to re-enable.")
ENABLED_OK_RIGHT = "[igus_rebel.arm_right_hw]: start-up sequence re-sent; drives enabled. Jog was held at zero."
ARMED_RIGHT = "[igus_rebel.arm_right_hw]: Fault group 'dual_arm_table': drives enabled and STATUS clean; monitoring armed"
CLEARED_RIGHT = CLEARED.replace("arm_left_hw", "arm_right_hw")


def test_drives_disabled_line_is_parsed_and_superseded_by_enabled_lines():
    events = rec.parse_fault_events([DISABLED_RIGHT])
    assert [e.kind for e in events] == ["DRIVES_DISABLED"] and events[0].component == "arm_right_hw"
    assert rec.drives_disabled_arms(events) == ["right"]
    for later in (ENABLED_OK_RIGHT, ARMED_RIGHT, CLEARED_RIGHT):
        assert rec.drives_disabled_arms(rec.parse_fault_events([DISABLED_RIGHT, later])) == []
    assert rec.drives_disabled_arms(rec.parse_fault_events([ARMED_RIGHT, DISABLED_RIGHT])) == ["right"]


def test_root_cleared_helper():
    assert rec.root_cleared(rec.parse_fault_events([ROOT_LEFT, CLEARED]), "arm_left_hw")
    assert not rec.root_cleared(rec.parse_fault_events([ROOT_LEFT]), "arm_left_hw")
    assert not rec.root_cleared(rec.parse_fault_events([CLEARED, ROOT_LEFT]), "arm_left_hw")
    assert not rec.root_cleared(rec.parse_fault_events([CLEARED_RIGHT]), "arm_left_hw")


# --- recovery runner with a fake cell ---------------------------------------------------------------------------

class FakeCell:
    """Models the controller_manager + two Igus drivers. Time is fake: spin_for and _sleep advance it.

    Per arm: hw lifecycle state, trajectory controller state, drives enabled flag, latched flag, supply_back_at
    (a reset_and_enable before that fails with 'Motor not enabled'), and whether the root fault is held.
    """

    def __init__(self, clock, cls=REAL):
        self.clock = clock
        self.cls = cls
        self.hw = {"left": "active", "right": "active"}
        self.ctrl = {"left": "active", "right": "active"}
        self.drives = {"left": True, "right": True}
        self.latched = {"left": False, "right": False}
        self.root_fault = {"left": False, "right": False}
        self.supply_back_at = {"left": 0.0, "right": 0.0}
        self.lose_enable_on_activation = {"left": False, "right": False}
        self.clear_line_ok = True          # False: the root never clears (persistent fault)
        self.driver_lines_on_rosout = True  # False: like the real stack, driver lines never reach /rosout
        self.rosout = []
        self.log = []                      # ordered record of actions
        self.still = {"left": (True, "still"), "right": (True, "still")}
        self.reset_calls = {"left": 0, "right": 0}
        self.clean_since = {}
        # Driver post-recovery jog hold (Rebel.cpp controllerRestartHold_): set when a latched hardware component
        # is re-activated, cleared only by a controller ACTIVATION. unsafe: a moment where an ACTIVE controller
        # could have moved the arm (drives enabled, controller active, no hold).
        self.hold = {"left": False, "right": False}
        self.unsafe = []
        self.rejected = []                 # deactivations Humble would reject (hardware not active)

    # -- time
    def spin_for(self, seconds):
        self.clock.advance(seconds)
        self._tick()

    def _tick(self):
        for arm in ("left", "right"):
            since = self.clean_since.get(arm)
            if since is not None and self.hw[arm] == "active" and self.drives[arm] and self.root_fault[arm] \
                    and self.clock.t - since >= 0.5 and self.clear_line_ok:
                self.root_fault[arm] = False
                if self.driver_lines_on_rosout:
                    self.rosout.append(f"[igus_rebel.arm_{arm}_hw]: Fault group 'dual_arm_table': 'arm_{arm}_hw' is "
                                       f"enabled with clean STATUS again; its root-cause fault is cleared, other "
                                       f"members may now be recovered.")

    # -- queries
    def views(self):
        out = {}
        for arm in ("left", "right"):
            out[arm] = rec.ArmView(arm=arm, hw_state=self.hw[arm], hw_class=self.cls, controller_state=self.ctrl[arm])
        return out, {}

    def rosout_mark(self):
        return len(self.rosout)

    def rosout_since(self, mark):
        return list(self.rosout[mark:])

    def stillness(self, arm):
        return self.still[arm]

    def positions(self, seconds=0.5):
        return {}, []

    # -- actions
    def switch(self, activate=(), deactivate=(), timeout=15.0):
        for name in deactivate:
            arm = name.split("_")[1]
            if self.hw[arm] in ("unconfigured", "finalized"):
                # Humble's resource_manager after on_error: "Not available: arm_x_joint1..6/velocity" -> rejected
                # (an INACTIVE component keeps its interfaces available, so a deactivation is accepted there)
                self.rejected.append(name)
                self.log.append(("deactivate_rejected", name))
                return False
        for name in deactivate:
            self.ctrl[name.split("_")[1]] = "inactive"
            self.log.append(("deactivate", name))
        for name in activate:
            arm = name.split("_")[1]
            self.log.append(("activate", name, self.drives[arm], dict(self.hw)))
            self.ctrl[arm] = "active"
            self.hold[arm] = False
        return True

    def set_hw_state(self, arm, target, timeout=20.0):
        self.log.append(("hw", arm, target))
        if target == "inactive":
            self.hw[arm] = "inactive"
            return True, "inactive"
        other = rec.OTHER[arm]
        if self.latched[arm] and self.root_fault[other]:
            return False, "inactive"            # driver: Refusing to activate (peer while root un-recovered)
        if self.latched[arm]:
            self.hold[arm] = True
        self.latched[arm] = False
        self.hw[arm] = "active"
        self.clean_since[arm] = self.clock.t
        self.drives[arm] = not self.lose_enable_on_activation[arm] and self.clock.t >= self.supply_back_at[arm]
        self.lose_enable_on_activation[arm] = False
        return True, "active"

    def reset_and_enable(self, arm, timeout=10.0, discover_s=2.0):
        self.reset_calls[arm] += 1
        self.log.append(("reset_and_enable", arm))
        if self.cls.startswith("mock_components/"):
            return None
        if self.latched[arm] or self.hw[arm] != "active":
            return False, "refused: hardware error is latched; recover with 'ros2 control set_hardware_component_state x inactive'"
        if self.clock.t < self.supply_back_at[arm]:
            self.clock.advance(3.0)
            return False, "start-up sequence re-sent but drives still report 'Motor not enabled' (mask 0x63); check the robot"
        if self.ctrl[arm] == "active" and not self.hold[arm] and self.cls == REAL:
            self.unsafe.append(("reset_and_enable with an active controller and no driver hold", arm))
        self.drives[arm] = True
        self.clean_since[arm] = self.clock.t
        return True, "start-up sequence re-sent; drives enabled."


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rec, "_now", lambda: c.t)
    monkeypatch.setattr(rec, "_sleep", lambda s: c.advance(s))
    return c


class Args:
    yes = False


def fault_both(cell, root="right"):
    """State after an e-stop on `root`: both hardware latched (inactive-like), root holds its fault."""
    for arm in ("left", "right"):
        cell.hw[arm] = "unconfigured"
        cell.ctrl[arm] = "active"
        cell.latched[arm] = True
        cell.drives[arm] = False
    cell.root_fault[root] = True


def run(cell, root, answer="yes", monkeypatch=None, requested=None):
    v, _ = cell.views()
    plan = rec.plan_recovery(v, requested or root, [])
    if monkeypatch is not None:
        monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    rec.run_recovery(cell, plan, v, Args())
    return plan


def activations(cell):
    return [entry for entry in cell.log if entry[0] == "activate"]


def test_enable_lost_after_activation_is_reenabled_and_waited_for(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.lose_enable_on_activation["right"] = True       # the 2026-10-15 log: Enable sent before supply was back
    run(cell, "right", monkeypatch=monkeypatch)
    assert cell.reset_calls["right"] >= 1
    assert cell.drives["right"] and cell.drives["left"]
    assert cell.hw == {"left": "active", "right": "active"}
    # controllers only after every drive is enabled and every hardware component is active
    acts = activations(cell)
    assert acts and all(entry[2] for entry in acts)
    assert all(state == "active" for entry in acts for state in entry[3].values())


def test_enable_is_retried_until_supply_returns_within_the_window(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.supply_back_at["right"] = clock.t + 2.5
    run(cell, "right", monkeypatch=monkeypatch)
    assert cell.reset_calls["right"] >= 2 and cell.drives["right"]


def test_drives_never_enabled_fails_clearly_and_activates_nothing(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.supply_back_at["right"] = clock.t + 1000
    with pytest.raises(rec.StepFailed, match="did not report enabled"):
        run(cell, "right", monkeypatch=monkeypatch)
    assert activations(cell) == []
    assert ("hw", "left", "inactive") not in cell.log          # the peer was never touched


def test_missing_root_cleared_line_does_not_block_the_peer(clock, monkeypatch):
    # The driver's 'root-cause fault is cleared' line never reaches /rosout on the real stack; the peer's
    # (driver-enforced) activation decides instead.
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.driver_lines_on_rosout = False
    run(cell, "right", monkeypatch=monkeypatch)
    assert cell.hw == {"left": "active", "right": "active"}


def test_root_that_never_clears_blocks_the_peer_and_activates_nothing(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.clear_line_ok = False
    with pytest.raises(rec.StepFailed, match="set arm_left_hw active failed"):
        run(cell, "right", monkeypatch=monkeypatch)
    assert activations(cell) == []
    assert cell.hw["left"] != "active"


def test_peer_activation_window_covers_the_root_settle_time(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    # the peer is refused until the root is cleared; the root clears after the wait, so the peer succeeds
    run(cell, "right", monkeypatch=monkeypatch)
    assert cell.hw == {"left": "active", "right": "active"}
    assert not cell.root_fault["right"]


def test_peer_refusal_beyond_the_window_fails_with_root_hint(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    real_set = cell.set_hw_state

    def never_for_left(arm, target, timeout=20.0):
        if arm == "left" and target == "active":
            cell.log.append(("hw", arm, target))
            return False, "inactive"
        return real_set(arm, target, timeout)

    cell.set_hw_state = never_for_left
    with pytest.raises(rec.StepFailed, match="--root right"):
        run(cell, "right", monkeypatch=monkeypatch)
    assert activations(cell) == []
    assert clock.t - 1000.0 < 60


def test_already_active_but_disabled_arm_is_re_enabled_so_the_latched_peer_can_recover(clock, monkeypatch):
    cell = FakeCell(clock)
    # the situation right after the 2026-10-15 failure: right ACTIVE with disabled drives, still the root holder;
    # left latched and refused.
    cell.latched["left"] = True
    cell.hw["left"] = "unconfigured"
    cell.drives["left"] = False
    cell.drives["right"] = False
    cell.root_fault["right"] = True
    cell.clean_since["right"] = None
    v, _ = cell.views()
    v["right"].drives_disabled = True
    plan = rec.plan_recovery(v, "auto", ["right"])
    assert plan.order == ["right", "left"]
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    rec.run_recovery(cell, plan, v, Args())
    # the active right controller was deactivated BEFORE reset_and_enable, and re-activated only at the end
    kinds = [entry[0] for entry in cell.log]
    assert cell.log.index(("deactivate", "arm_right_trajectory_controller")) < cell.log.index(
        ("reset_and_enable", "right"))
    assert cell.hw == {"left": "active", "right": "active"} and cell.drives["left"] and cell.drives["right"]
    assert ("hw", "right", "inactive") not in cell.log          # no lifecycle cycle for an already-active arm
    assert kinds.index("activate") > max(i for i, k in enumerate(kinds) if k in ("hw", "reset_and_enable"))
    assert len(activations(cell)) == 2


def test_controllers_are_activated_only_after_all_hardware_is_recovered(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "left")
    run(cell, "left", monkeypatch=monkeypatch)
    last_hw = max(i for i, entry in enumerate(cell.log) if entry[0] in ("hw", "reset_and_enable"))
    first_act = min(i for i, entry in enumerate(cell.log) if entry[0] == "activate")
    assert last_hw < first_act
    assert [entry[1] for entry in activations(cell)] == ["arm_left_trajectory_controller",
                                                         "arm_right_trajectory_controller"]


def test_identity_not_confirmed_activates_nothing_and_fails(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    with pytest.raises(rec.StepFailed, match="controllers NOT activated"):
        run(cell, "right", answer="no", monkeypatch=monkeypatch)
    assert activations(cell) == []
    assert cell.hw == {"left": "active", "right": "active"}     # hardware recovery itself is kept


def test_identity_is_asked_exactly_once(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "yes")
    v, _ = cell.views()
    rec.run_recovery(cell, rec.plan_recovery(v, "right", []), v, Args())
    assert len(asked) == 1


def test_not_still_after_hardware_recovery_activates_nothing(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.still["left"] = (False, "joint moved")
    with pytest.raises(rec.StepFailed, match="not confirmed standing still"):
        run(cell, "right", monkeypatch=monkeypatch)
    assert activations(cell) == []


def test_mock_recovery_needs_no_prompt_no_enable_and_activates_at_the_end(clock, monkeypatch):
    cell = FakeCell(clock, cls=MOCK)
    for arm in ("left", "right"):
        cell.hw[arm] = "inactive"
        cell.ctrl[arm] = "inactive"
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("mock must not prompt"))
    v, _ = cell.views()
    rec.run_recovery(cell, rec.plan_recovery(v, "left", []), v, Args())
    assert len(activations(cell)) == 2 and cell.hw == {"left": "active", "right": "active"}


def test_ensure_drives_enabled_fails_when_the_service_is_missing_on_real_hardware(clock):
    class NoService(FakeCell):
        def reset_and_enable(self, arm, timeout=10.0, discover_s=2.0):
            return None

    cell = NoService(clock)
    with pytest.raises(rec.StepFailed, match="not available"):
        rec.ensure_drives_enabled(cell, "right", view("right", cls=REAL))
    rec.ensure_drives_enabled(cell, "right", view("right", cls=MOCK))     # mock: skipped


def test_latched_reply_after_activation_is_a_failure(clock):
    class Latched(FakeCell):
        def reset_and_enable(self, arm, timeout=10.0, discover_s=2.0):
            return False, "refused: hardware error is latched; recover with ..."

    with pytest.raises(rec.StepFailed, match="latched"):
        rec.ensure_drives_enabled(Latched(clock), "left", view("left", cls=REAL))


# --- Humble order: hardware first, then the (still active) controller, activation last --------------------------

def kinds(cell):
    return [entry[0] for entry in cell.log]


def test_active_controller_is_deactivated_only_after_the_hardware_is_active(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")          # both hardware unconfigured, both controllers still ACTIVE (Humble)
    run(cell, "right", monkeypatch=monkeypatch)
    # one deactivation attempt per arm before the recovery is rejected (hardware unconfigured), as on the real cell
    assert sorted(cell.rejected) == ["arm_left_trajectory_controller", "arm_right_trajectory_controller"]
    for arm in ("right", "left"):
        deact = cell.log.index(("deactivate", f"arm_{arm}_trajectory_controller"))
        assert cell.log.index(("hw", arm, "inactive")) < cell.log.index(("hw", arm, "active")) < deact
    assert cell.hw == {"left": "active", "right": "active"}


def test_controller_is_deactivated_right_after_its_hardware_before_enabling_the_drives(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    run(cell, "right", monkeypatch=monkeypatch)
    for arm in ("right", "left"):
        hw_active = cell.log.index(("hw", arm, "active"))
        deact = cell.log.index(("deactivate", f"arm_{arm}_trajectory_controller"))
        # the only reset_and_enable of this arm after the activation comes after the controller deactivation
        later_resets = [i for i, e in enumerate(cell.log) if e == ("reset_and_enable", arm) and i > hw_active]
        assert later_resets and min(later_resets) > deact
    assert cell.unsafe == []


def test_root_controller_is_deactivated_before_the_peer_hardware_is_touched(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "left")
    run(cell, "left", monkeypatch=monkeypatch)
    assert cell.log.index(("deactivate", "arm_left_trajectory_controller")) < cell.log.index(("hw", "right", "inactive"))


def test_no_controller_is_activated_until_the_end_and_the_activation_releases_the_hold(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    seen_hold = []
    real_switch = cell.switch

    def switch(activate=(), deactivate=(), timeout=15.0):
        if activate:
            seen_hold.append(dict(cell.hold))   # the hold must still be set right before the activation
        return real_switch(activate=activate, deactivate=deactivate, timeout=timeout)

    cell.switch = switch
    run(cell, "right", monkeypatch=monkeypatch)
    assert seen_hold == [{"left": True, "right": True}]
    assert cell.hold == {"left": False, "right": False}
    assert cell.ctrl == {"left": "active", "right": "active"}
    assert cell.unsafe == []
    kind = kinds(cell)
    assert kind.index("activate") > max(i for i, k in enumerate(kind) if k in ("hw", "reset_and_enable", "deactivate"))


def test_failed_activation_leaves_the_driver_hold_in_place(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    with pytest.raises(rec.StepFailed, match="controllers NOT activated"):
        run(cell, "right", answer="no", monkeypatch=monkeypatch)
    assert cell.hold == {"left": True, "right": True}     # still fail closed: nothing can move
    assert cell.ctrl == {"left": "inactive", "right": "inactive"}


def test_active_hardware_with_disabled_drives_keeps_deactivating_first(clock, monkeypatch):
    cell = FakeCell(clock)
    cell.drives["right"] = False
    cell.root_fault["right"] = True
    cell.clean_since["right"] = None
    v, _ = cell.views()
    v["right"].drives_disabled = True
    plan = rec.plan_recovery(v, "right", [])
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    rec.run_recovery(cell, plan, v, Args())
    assert cell.rejected == [] and cell.unsafe == []
    assert cell.log.index(("deactivate", "arm_right_trajectory_controller")) < cell.log.index(("reset_and_enable", "right"))
    assert ("hw", "right", "inactive") not in cell.log


def test_deactivation_failure_after_hardware_recovery_says_the_driver_still_holds_jog(clock, monkeypatch):
    cell = FakeCell(clock)
    fault_both(cell, "right")
    cell.switch = lambda activate=(), deactivate=(), timeout=15.0: False
    with pytest.raises(rec.StepFailed, match="keeps jog at zero"):
        run(cell, "right", monkeypatch=monkeypatch)
    assert cell.hold["right"] and activations(cell) == []


def test_controller_active_while_hardware_not_active_is_flagged():
    assert rec.controller_active_without_hardware(view("left", hw="unconfigured", ctrl="active"))
    assert rec.controller_active_without_hardware(view("left", hw="inactive", ctrl="active"))
    assert not rec.controller_active_without_hardware(view("left", hw="active", ctrl="active"))
    assert not rec.controller_active_without_hardware(view("left", hw="unconfigured", ctrl="inactive"))
    assert not rec.controller_active_without_hardware(view("left", hw=None, ctrl="active"))
    assert "WARNING: controller ACTIVE while hardware not active" in rec.view_line(
        view("left", hw="unconfigured", ctrl="active"))
    assert "controller ACTIVE while" not in rec.view_line(view("left"))


def test_dry_run_order_for_latched_arms_with_active_controllers():
    v = views(view("left", hw="unconfigured", ctrl="active", cls=REAL),
              view("right", hw="unconfigured", ctrl="active", cls=REAL))
    steps = rec.recovery_steps(v, rec.plan_recovery(v, "left", []))
    assert [s for a, s in steps if a == "left"] == [
        "reset_and_enable", "hw_inactive", "hw_active", "controller_deactivate", "ensure_enabled",
        "wait_root_cleared"]
    assert [s for a, s in steps if a == "right"] == [
        "reset_and_enable", "hw_inactive", "hw_active", "controller_deactivate", "ensure_enabled"]


def test_inactive_unlatched_hardware_gets_its_controller_deactivated_first(clock, monkeypatch):
    # Hardware set INACTIVE by hand (no driver latch, so no driver jog hold): the still-active controller must be
    # deactivated BEFORE reset_and_enable / the lifecycle activation.
    cell = FakeCell(clock)
    cell.hw["right"] = "inactive"
    cell.ctrl["right"] = "active"
    run(cell, "right", monkeypatch=monkeypatch)
    deact = cell.log.index(("deactivate", "arm_right_trajectory_controller"))
    assert deact < cell.log.index(("hw", "right", "active"))
    assert all(i > deact for i, e in enumerate(cell.log) if e == ("reset_and_enable", "right"))
    assert cell.rejected == []
