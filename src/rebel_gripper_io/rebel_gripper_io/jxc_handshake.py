"""Pure-Python SMC JXC61 parallel-I/O handshake (no ROS imports).

The robot-side outputs (to the JXC inputs) are IN0, IN1, SETUP, DRIVE, RESET, SVON; the robot-side inputs
(from the JXC outputs) are SETON, INP, SVRE, *ALARM and optionally BUSY. All pins are iRC panel
numbers. The class talks to the world only through an injected `io` object so it can be unit-tested
against a simulated JXC:

    io.read() -> IoView          latest accepted digital_io sample (see below)
    io.set_outputs(pairs) -> (ok, message)    pairs = [(irc_number, bool), ...], one request
    io.clock() -> float          seconds on the SAME clock as IoView.stamp (system time)
    io.sleep(seconds)            wait (the real io sleeps; a simulation advances its own time)

Freshness comes from the source: IoView.stamp is the driver's STATUS receive time (header stamp of
the digital_io message), age_s = clock() - stamp. A driver heartbeat that republishes an old STATUS
keeps its old stamp and therefore ages. A confirmation (output seen ON/OFF in dout, input seen in
din) only counts from a sample whose stamp is NEWER than the moment the request was sent.

"set_outputs succeeded" only means the driver SENT the command. After every cleanup the handshake
confirms in digital_io; if that fails (or a set_outputs call failed) the latch `outputs_uncertain`
is set. While latched, servo_on/home/move_step are refused; release_all (confirmed all-off) clears it.

Safety rules: one command at a time; refuse when digital_io is invalid/stale; abort a running
command on alarm, stale digital_io, or loss of a state it depends on (SVRE, SETON, its own outputs);
DRIVE and SETUP are switched off (and confirmed) on every exit of a command that touched outputs.
DRIVE off is NOT a guaranteed stop of a move the JXC has started (HOLD is not wired; RESET is wired
but used ONLY by reset_alarm, never as a stop, because its operation-interruption behaviour is
unverified on this hardware). reset_alarm is allowed only with *ALARM active; RESET is switched off
(and confirmed) on every exit of it.
"""

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class IoView:
    valid: bool = False
    stamp: float = 0.0          # STATUS receive time at the source (system clock, s); 0 = none
    age_s: float = float("inf")  # clock() - stamp at read time
    din_on: FrozenSet[int] = frozenset()
    dout_on: FrozenSet[int] = frozenset()


class SampleTracker:
    """Accepts digital_io samples by source stamp only (pure, shared by the ROS io and tests).

    A zero/non-finite stamp is invalid; a stamp older than the last accepted one (regressing) is
    ignored; an equal stamp (heartbeat republish) is accepted but does not make the data newer.
    """

    def __init__(self):
        self._stamp = 0.0
        self._valid = False
        self._din: FrozenSet[int] = frozenset()
        self._dout: FrozenSet[int] = frozenset()
        self.rejected = 0

    def accept(self, valid: bool, stamp: float, din_on, dout_on) -> bool:
        if not math.isfinite(stamp) or stamp <= 0.0 or stamp < self._stamp:
            self.rejected += 1
            if stamp == 0.0 or not math.isfinite(stamp):
                self._valid = False   # a zero stamp is "no data": never trust what came before
            return False
        self._stamp = stamp
        self._valid = bool(valid)
        self._din = frozenset(int(n) for n in din_on)
        self._dout = frozenset(int(n) for n in dout_on)
        return True

    def view(self, now: float) -> IoView:
        if self._stamp <= 0.0:
            return IoView()
        return IoView(valid=self._valid, stamp=self._stamp, age_s=now - self._stamp,
                      din_on=self._din, dout_on=self._dout)


@dataclass
class JxcConfig:
    # Outputs of the robot (to the JXC inputs). Operator confirmed 2026-11-13 (both arms).
    out_in0: int = 21
    out_in1: int = 22
    out_setup: int = 23
    out_drive: int = 24
    out_reset: int = 25
    out_svon: int = 26
    # Inputs of the robot (from the JXC outputs). Operator confirmed 2026-11-13. *ALARM: input ON = healthy, OFF = alarm.
    in_seton: int = 21
    in_inp: int = 22
    in_svre: int = 23
    in_alarm: int = 24
    in_busy: int = -1                    # BUSY output of the JXC; -1 = not wired
    # Timing [s]
    stale_s: float = 0.3                 # digital_io older than this (or invalid) = refuse/abort
    poll_s: float = 0.005
    servo_on_timeout_s: float = 3.0      # SVON -> SVRE
    home_timeout_s: float = 30.0         # SETUP -> SETON (+INP)
    dout_confirm_timeout_s: float = 0.5  # output command -> dout bit seen in digital_io
    step_setup_delay_s: float = 0.05     # IN0/IN1 stable before DRIVE
    inp_clear_timeout_s: float = 0.5     # DRIVE on -> INP OFF (or BUSY ON); ASSUMPTION, verify
    move_timeout_s: float = 5.0          # start of motion -> INP on (and BUSY off)
    servo_off_timeout_s: float = 1.0     # SVON off -> SVRE off
    reset_timeout_s: float = 1.0         # RESET on -> *ALARM back ON (healthy)
    # Automatic alarm reset (auto_reset_alarm_poll); operator decision 2026-11-13: default on.
    auto_reset_alarm: bool = True
    auto_reset_max: int = 3              # more than this many auto resets ...
    auto_reset_window_s: float = 60.0    # ... within this window disables auto reset

    @property
    def jxc_outputs(self) -> List[int]:
        return [self.out_in0, self.out_in1, self.out_setup, self.out_drive, self.out_reset,
                self.out_svon]

    @property
    def busy_wired(self) -> bool:
        return self.in_busy >= 0


@dataclass
class Result:
    success: bool
    message: str


class _Abort(Exception):
    """A running command must stop (timeout, alarm, stale io, lost state, rejected output)."""


class JxcHandshake:
    def __init__(self, io, config: Optional[JxcConfig] = None):
        self._io = io
        self.cfg = config or JxcConfig()
        self._lock = threading.Lock()
        self._busy = False
        self._last_error = ""
        self._uncertain = False
        self._uncertain_reason = ""
        self._shutdown = False
        self._touched = False
        # automatic alarm reset state
        self._auto_active = False        # an auto reset (incl. the SVRE wait) is in progress
        self._auto_times: List[float] = []
        self._auto_count = 0
        self._auto_last: Optional[float] = None
        self._auto_disabled_reason = ""
        self.log: Callable[[str, str], None] = lambda level, message: None  # level: info/warn/error

    # ------------------------------------------------------------------ status
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def outputs_uncertain(self) -> bool:
        return self._uncertain

    @property
    def last_error(self) -> str:
        if self._uncertain:
            return f"outputs_uncertain: {self._uncertain_reason}"
        return self._last_error

    @property
    def auto_reset_state(self) -> dict:
        """enabled (configured and not disabled), disabled reason, count, last time, running."""
        reason = self._auto_disabled_reason
        if not self.cfg.auto_reset_alarm:
            reason = "auto_reset_alarm is false"
        return {"enabled": bool(self.cfg.auto_reset_alarm and not self._auto_disabled_reason),
                "reason": reason, "count": self._auto_count, "last": self._auto_last,
                "running": self._auto_active}

    def request_abort(self):
        """Permanent (shutdown): running commands abort at their next wait."""
        self._shutdown = True

    def shutdown_outputs_off(self) -> Tuple[bool, str]:
        """Best effort DRIVE/SETUP off with confirmation; works while a command is running."""
        self.request_abort()
        ok, why = self._cleanup([(self.cfg.out_drive, False), (self.cfg.out_setup, False),
                                 (self.cfg.out_reset, False)])
        if not ok:
            self._latch(f"shutdown cleanup not confirmed: {why}")
        return ok, why

    # ------------------------------------------------------------------ public commands
    def servo_on(self) -> Result:
        def body():
            c = self.cfg
            self._require_healthy()
            t = self._set([(c.out_svon, True)])
            self._wait(lambda v: c.out_svon in v.dout_on, c.dout_confirm_timeout_s,
                       "SVON not confirmed in digital_io dout", since=t)
            self._wait(lambda v: c.in_svre in v.din_on, c.servo_on_timeout_s,
                       "SVRE did not come ON after SVON", since=t,
                       guards=[(lambda v: c.out_svon in v.dout_on, "SVON output dropped")])
            return "servo on (SVRE)"
        return self._run("servo_on", body, off_on_failure=[self.cfg.out_svon])

    def home(self) -> Result:
        def body():
            c = self.cfg
            view = self._require_healthy()
            self._require_svre(view)
            self._require_output_off(view, c.out_drive, "DRIVE")
            if c.in_seton in view.din_on:
                return "already homed (SETON on)"
            t = self._set([(c.out_setup, True)])
            svre_guard = (lambda v: c.in_svre in v.din_on, "SVRE dropped")
            self._wait(lambda v: c.out_setup in v.dout_on, c.dout_confirm_timeout_s,
                       "SETUP not confirmed in digital_io dout", since=t, guards=[svre_guard])

            def done(v):
                if c.in_seton not in v.din_on or c.in_inp not in v.din_on:
                    return False
                return not c.busy_wired or c.in_busy not in v.din_on
            self._wait(done, c.home_timeout_s,
                       "SETON/INP (BUSY off) did not come after SETUP", since=t,
                       guards=[svre_guard,
                               (lambda v: c.out_setup in v.dout_on, "SETUP output dropped")])
            self._set([(c.out_setup, False)])
            return "homed (SETON, INP)"
        return self._run("home", body)

    def move_step(self, step: int) -> Result:
        def body():
            c = self.cfg
            if not isinstance(step, int) or not 0 <= step <= 3:
                raise _Abort(f"step {step!r} out of range 0..3")
            view = self._require_healthy()
            self._require_svre(view)
            if c.in_seton not in view.din_on:
                raise _Abort("refused: SETON is off (home first)")
            self._require_output_off(view, c.out_drive, "DRIVE")
            in0, in1 = bool(step & 1), bool(step & 2)

            def selected(v):
                return (c.out_in0 in v.dout_on) == in0 and (c.out_in1 in v.dout_on) == in1
            svre_g = (lambda v: c.in_svre in v.din_on, "SVRE dropped")
            seton_g = (lambda v: c.in_seton in v.din_on, "SETON dropped")
            sel_g = (selected, "IN0/IN1 changed")
            t_sel = self._set([(c.out_in0, in0), (c.out_in1, in1)])
            self._wait(selected, c.dout_confirm_timeout_s,
                       "IN0/IN1 not confirmed in digital_io dout", since=t_sel,
                       guards=[svre_g, seton_g])
            self._io.sleep(c.step_setup_delay_s)
            # Re-check everything immediately before DRIVE on.
            view = self._require_healthy()
            self._require_svre(view)
            if c.in_seton not in view.din_on:
                raise _Abort("aborted: SETON dropped before DRIVE")
            if not selected(view):
                raise _Abort("aborted: IN0/IN1 no longer as selected before DRIVE")
            t_drv = self._set([(c.out_drive, True)])
            guards = [svre_g, seton_g, sel_g]
            self._wait(lambda v: c.out_drive in v.dout_on, c.dout_confirm_timeout_s,
                       "DRIVE not confirmed in digital_io dout", since=t_drv, guards=guards)
            guards.append((lambda v: c.out_drive in v.dout_on,
                           "DRIVE output dropped (robot dropped its outputs?)"))
            if c.busy_wired:
                self._wait(lambda v: c.in_busy in v.din_on, c.inp_clear_timeout_s,
                           "BUSY never came ON after DRIVE", since=t_drv, guards=guards)
                self._wait(lambda v: c.in_busy not in v.din_on and c.in_inp in v.din_on,
                           c.move_timeout_s,
                           "BUSY/INP: position not reached / no part gripped", since=t_drv,
                           guards=guards)
            else:
                # Without BUSY the start of the move is only visible as INP dropping. On the real
                # JXC (2026-11-13) INP can stay ON (already in position, e.g. holding after an
                # alarm reset), so a missing drop is NOT an error (operator decision 2026-11-13):
                # the move is then judged only by INP being ON afterwards.
                start_seen = True
                try:
                    self._wait(lambda v: c.in_inp not in v.din_on, c.inp_clear_timeout_s,
                               "INP never went OFF after DRIVE", since=t_drv, guards=guards)
                except _Abort as error:
                    if not str(error).startswith("timeout"):
                        raise
                    start_seen = False
                self._wait(lambda v: c.in_inp in v.din_on, c.move_timeout_s,
                           "INP not ON in time: position not reached / no part gripped",
                           since=t_drv, guards=guards)
                if not start_seen:
                    return (f"step {step} reached (INP stayed ON: start of the move not seen "
                            "without BUSY)")
            return f"step {step} reached"
        return self._run("move_step", body)

    def servo_off(self) -> Result:
        def body():
            c = self.cfg
            self._set_off_confirmed([c.out_drive, c.out_setup, c.out_svon])
            t = self._io.clock()
            self._wait(lambda v: c.in_svre not in v.din_on, c.servo_off_timeout_s,
                       "SVRE did not go OFF after SVON off", since=t, check_alarm=False)
            return "servo off"
        return self._run("servo_off", body)

    def reset_alarm(self) -> Result:
        """Clear an active JXC alarm with a RESET pulse (only when *ALARM is OFF).

        A successful manual reset re-enables a rate-limited / failed automatic reset.
        """
        result = self._reset_alarm(auto=False)
        if result.success:
            self._auto_disabled_reason = ""
            self._auto_times.clear()
        return result

    def _reset_alarm(self, auto: bool) -> Result:
        def body():
            c = self.cfg
            view = self._require_io()
            if c.in_alarm in view.din_on:
                raise _Abort("refused: no alarm active (*ALARM is ON), nothing to reset")
            self._set_off_confirmed([c.out_drive, c.out_setup])
            t = self._set([(c.out_reset, True)])
            self._wait(lambda v: c.out_reset in v.dout_on, c.dout_confirm_timeout_s,
                       "RESET not confirmed in digital_io dout", since=t, check_alarm=False)
            try:
                self._wait(lambda v: c.in_alarm in v.din_on, c.reset_timeout_s, "", since=t,
                           check_alarm=False,
                           guards=[(lambda v: c.out_reset in v.dout_on, "RESET output dropped")])
            except _Abort as error:
                if str(error).startswith("timeout"):
                    raise _Abort("alarm did not clear: check the JXC alarm code; some alarms "
                                 "need a power cycle")
                raise
            return "alarm cleared (*ALARM ON after RESET)"
        return self._run("reset_alarm", body, always_off=[self.cfg.out_reset], auto=auto)

    def auto_reset_poll(self) -> Optional[Result]:
        """One step of the automatic alarm monitor (call at ~20 Hz from ONE thread).

        Resets only when enabled, digital_io is fresh, *ALARM is OFF, no command runs and no
        outputs_uncertain latch is set. It only ever uses reset_alarm's logic (the only RESET
        path); it never turns SVON on, homes or moves. Returns the reset Result if it tried one.
        """
        c = self.cfg
        if not c.auto_reset_alarm or self._auto_disabled_reason:
            return None
        if self._busy or self._uncertain or self._shutdown or self._auto_active:
            return None
        view = self._view()
        if not self._usable(view) or c.in_alarm in view.din_on:
            return None
        now = self._io.clock()
        self._auto_times = [t for t in self._auto_times if now - t < c.auto_reset_window_s]
        if len(self._auto_times) >= c.auto_reset_max:
            self._disable_auto("repeated JXC alarms: auto reset disabled, check the alarm code; "
                               "call ~/reset_alarm manually")
            return None
        self._auto_active = True     # service calls are refused from here until we are done
        try:
            result = self._reset_alarm(auto=True)
            if result.message.startswith("busy") or result.message.startswith("refused"):
                return None          # lost a race / nothing sent: not an attempt
            if not result.success:
                self._disable_auto(f"auto alarm reset failed ({result.message}): auto reset "
                                   "disabled, check the alarm code; call ~/reset_alarm manually")
                return result
            self._auto_times.append(now)
            self._auto_count += 1
            self._auto_last = time.time()
            self._auto_svre_outcome()
            return result
        finally:
            self._auto_active = False

    def _disable_auto(self, message: str):
        self._auto_disabled_reason = message
        self.log("error", message)

    def _auto_svre_outcome(self):
        c = self.cfg
        n = self._auto_count
        view = self._view()
        if c.out_svon not in view.dout_on:
            self.log("warn", f"auto reset #{n}: alarm cleared (SVON not commanded, not waiting "
                             "for SVRE)")
            return
        try:
            self._wait(lambda v: c.in_svre in v.din_on, c.servo_on_timeout_s, "SVRE",
                       check_alarm=False)
            self.log("warn", f"auto reset #{n}: alarm cleared, SVRE back")
        except _Abort as error:
            self.log("warn", f"auto reset #{n}: alarm cleared, SVRE did not return ({error})")

    def release_all(self) -> Result:
        def body():
            self._set_off_confirmed(self.cfg.jxc_outputs)
            self._uncertain = False
            self._uncertain_reason = ""
            return "all JXC outputs off (confirmed)"
        return self._run("release_all", body)

    # ------------------------------------------------------------------ internals
    def _run(self, name: str, body: Callable[[], str], off_on_failure=(),
             always_off=(), auto: bool = False) -> Result:
        if self._auto_active and not auto:
            return Result(False, f"busy: auto alarm reset running (refused {name})")
        if not self._lock.acquire(blocking=False):
            return Result(False, f"busy: another command is running (refused {name})")
        self._busy = True
        self._touched = False
        try:
            if self._uncertain and name not in ("release_all", "servo_off"):
                message = (f"refused {name}: outputs_uncertain latched ({self._uncertain_reason});"
                           " run release_all")
                self._last_error = message
                return Result(False, message)
            try:
                message = body()
                ok = True
            except _Abort as error:
                message, ok = str(error), False
            except Exception as error:  # never leave DRIVE on because of a bug
                message, ok = f"internal error: {error!r}", False
            if self._touched and not (ok and name in ("release_all", "servo_off")):
                cleanup = [(self.cfg.out_drive, False), (self.cfg.out_setup, False)]
                cleanup += [(p, False) for p in always_off]
                cleanup += [(p, False) for p in off_on_failure if not ok]
                cleaned, why = self._cleanup(cleanup)
                if not cleaned:
                    self._latch(f"DRIVE/SETUP/RESET off not confirmed after {name}: {why}")
                    ok = False
                    message += f" | CLEANUP NOT CONFIRMED, outputs_uncertain: {why}"
            self._last_error = "" if ok else message
            return Result(ok, message)
        finally:
            self._busy = False
            self._lock.release()

    def _latch(self, reason: str):
        if not self._uncertain:
            self._uncertain_reason = reason
        self._uncertain = True

    def _cleanup(self, pairs) -> Tuple[bool, str]:
        """Send `pairs` (all OFF) and confirm in a sample newer than the request. 2 attempts."""
        why = ""
        pins = [p for p, _ in pairs]
        for _ in range(2):
            try:
                t = self._io.clock()
                ok, why = self._io.set_outputs(pairs)
                if not ok:
                    continue
                self._wait(lambda v: not any(p in v.dout_on for p in pins),
                           self.cfg.dout_confirm_timeout_s,
                           "outputs not confirmed OFF in digital_io dout", since=t,
                           check_alarm=False, abortable=False)
                return True, ""
            except _Abort as error:
                why = str(error)
            except Exception as error:
                why = f"exception: {error!r}"
        return False, why

    def _set_off_confirmed(self, pins):
        pairs = [(p, False) for p in pins]
        t = self._set(pairs)
        try:
            self._wait(lambda v: not any(p in v.dout_on for p in pins),
                       self.cfg.dout_confirm_timeout_s,
                       "outputs not confirmed OFF in digital_io dout", since=t,
                       check_alarm=False, abortable=False)
        except _Abort as error:
            self._latch(f"off not confirmed: {error}")
            raise

    def _view(self) -> IoView:
        return self._io.read()

    def _usable(self, view: IoView) -> bool:
        return view.valid and view.stamp > 0.0 and view.age_s <= self.cfg.stale_s

    def _require_io(self) -> IoView:
        view = self._view()
        if not self._usable(view):
            raise _Abort("refused: digital_io is not valid/fresh")
        return view

    def _require_healthy(self) -> IoView:
        view = self._require_io()
        if self.cfg.in_alarm not in view.din_on:
            raise _Abort("refused: JXC *ALARM active (input OFF)")
        return view

    def _require_svre(self, view: IoView):
        if self.cfg.in_svre not in view.din_on:
            raise _Abort("refused: SVRE is off (servo_on first)")

    def _require_output_off(self, view: IoView, pin: int, name: str):
        if pin in view.dout_on:
            raise _Abort(f"refused: {name} output is already on (use release_all)")

    def _set(self, pairs) -> float:
        """Send; returns the time the request was sent. Any failure latches outputs_uncertain."""
        self._touched = True
        t = self._io.clock()
        try:
            ok, why = self._io.set_outputs(pairs)
        except Exception as error:
            ok, why = False, f"exception: {error!r}"
        if not ok:
            self._latch(f"set outputs failed: {why}")
            raise _Abort(f"set outputs failed: {why}")
        return t

    def _wait(self, predicate, timeout_s: float, timeout_message: str, since: Optional[float] = None,
              guards=(), check_alarm: bool = True, abortable: bool = True):
        """Poll until predicate(view) holds on a sample newer than `since` (if given).

        Stale io, alarm and every guard (state the command depends on) abort on ANY sample.
        """
        deadline = self._io.clock() + timeout_s
        while True:
            if abortable and self._shutdown:
                raise _Abort("aborted: shutdown requested")
            view = self._view()
            if not self._usable(view):
                raise _Abort("aborted: digital_io went stale/invalid")
            if check_alarm and self.cfg.in_alarm not in view.din_on:
                raise _Abort("aborted: JXC *ALARM appeared")
            for ok_fn, message in guards:
                if not ok_fn(view):
                    raise _Abort(f"aborted: {message}")
            if (since is None or view.stamp > since) and predicate(view):
                return view
            if self._io.clock() >= deadline:
                raise _Abort(f"timeout {timeout_s:g} s: {timeout_message}")
            self._io.sleep(self.cfg.poll_s)
