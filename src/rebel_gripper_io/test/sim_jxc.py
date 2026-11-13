"""Simulated SMC JXC61 + driver digital_io for unit tests (simulated time, no threads).

digital_io is modelled like the driver: a snapshot (stamp = capture time, i.e. the STATUS receive
time) is captured every `status_period` s; read() returns the last snapshot with age = t - stamp.
`freeze_until` stops capturing (heartbeat of an old stamp), so old data ages.
"""

from rebel_gripper_io.jxc_handshake import IoView, JxcConfig


class SimJxc:
    def __init__(self, cfg=None, *, servo_delay=0.05, home_time=0.5, inp_off_delay=0.02,
                 move_time=0.3, io_latency=0.02, status_period=0.02):
        self.cfg = cfg or JxcConfig()
        self.t = 1000.0       # system-clock-like, so stamps are > 0
        self.status_period = status_period
        self.freeze_until = 0.0
        self.snap = (self.t, frozenset(), frozenset())
        self.ignore_drive_off = False   # JXC/robot ignores DRIVE off although 'sent'
        self.set_exception = False
        self.set_fail_applies = False
        self.set_fail = None            # message -> set_outputs returns (False, msg)
        self.busy_out = False
        self.servo_delay, self.home_time = servo_delay, home_time
        self.inp_off_delay, self.move_time = inp_off_delay, move_time
        self.io_latency = io_latency
        self.dout = set()             # applied robot outputs (as seen in digital_io)
        self.pending = []             # (apply_time, pin, state)
        self.seton = False
        self.inp = True
        self.svre = False
        self.alarm_ok = True
        self.io_valid = True
        self.alarm_resettable = True  # False: RESET rising edge does not clear the alarm
        self.never_reaches = False    # pushing step with no part
        self.never_clears_inp = False
        self.reject_outputs = False   # service failure
        # JXC raises an alarm this long after a pushing step completes (None = never): *ALARM and
        # SVRE drop, SETON/INP stay. RESET clears it; with SVON still on SVRE returns
        # `svre_return_delay` s after RESET goes OFF (measured on hardware: ~0.19 s).
        self.push_alarm_delay_s = None
        self.push_alarm_steps = (0, 1, 2, 3)
        self.svre_return_delay = 0.19
        self._svre_block_until = 0.0
        self._move_step = 0
        self.events = []              # (time, callable)
        self.log = []                 # (time, [(pin, state)]) every set_outputs request
        self.drive_trace = []         # (time, drive_state)
        self._svon_since = None
        self._setup_since = None
        self._move = None             # (t_inp_off, t_inp_on)
        self._steps_at_drive = []
        self.snap = (self.t, frozenset(self._din()), frozenset(self.dout))

    # ---- io interface
    def read(self):
        stamp, din, dout = self.snap
        return IoView(valid=self.io_valid, stamp=stamp, age_s=self.t - stamp, din_on=din,
                      dout_on=dout)

    def clock(self):
        return self.t

    def sleep(self, seconds):
        end = self.t + seconds
        while self.t < end - 1e-12:
            self.t = min(end, self.t + 0.001)
            self._step()

    def set_outputs(self, pairs):
        self.log.append((self.t, list(pairs)))
        if self.set_exception:
            raise RuntimeError("service call exploded")
        if self.set_fail is not None:
            if self.set_fail_applies:
                self._queue(pairs)
            return False, self.set_fail
        if self.reject_outputs:
            return False, "service rejected"
        self._queue(pairs)
        return True, "ok"

    def _queue(self, pairs):
        for pin, state in pairs:
            self.pending.append((self.t + self.io_latency, pin, state))

    # ---- model
    def _din(self):
        c = self.cfg
        out = set()
        if self.seton:
            out.add(c.in_seton)
        if self.inp:
            out.add(c.in_inp)
        if self.svre:
            out.add(c.in_svre)
        if self.alarm_ok:
            out.add(c.in_alarm)
        if c.in_busy >= 0 and self.busy_out:
            out.add(c.in_busy)
        return out

    def _step(self):
        c = self.cfg
        if self.t >= self.freeze_until and self.t - self.snap[0] >= self.status_period - 1e-9:
            self.snap = (self.t, frozenset(self._din()), frozenset(self.dout))
        for ev in [e for e in self.events if e[0] <= self.t]:
            self.events.remove(ev)
            ev[1](self)
        for item in [p for p in self.pending if p[0] <= self.t]:
            self.pending.remove(item)
            _, pin, state = item
            if pin == c.out_drive and not state and self.ignore_drive_off:
                continue
            was = pin in self.dout
            (self.dout.add if state else self.dout.discard)(pin)
            if pin == c.out_drive:
                self.drive_trace.append((self.t, state))
                if state and not was:
                    self._on_drive_rising()
            if pin == c.out_reset and state and not was and self.alarm_resettable:
                self.alarm_ok = True
            if pin == c.out_reset and not state and was and self.alarm_ok \
                    and self._svre_block_until == float("inf"):
                self._svre_block_until = self.t + self.svre_return_delay
            if pin == c.out_svon:
                self._svon_since = self.t if state else None
                if not state:
                    self.svre = False
                    self._svre_block_until = 0.0
            if pin == c.out_setup:
                self._setup_since = self.t if state else None
        if self._svon_since is not None and self.t - self._svon_since >= self.servo_delay \
                and self.t >= self._svre_block_until:
            self.svre = True
        if self._setup_since is not None and self.svre and \
                self.t - self._setup_since >= self.home_time:
            self.seton = True
            self.inp = True
        if self._move:
            off_t, on_t = self._move
            if self.t >= off_t and not self.never_clears_inp:
                self.inp = False
            if self.t >= on_t and not self.never_reaches:
                self.inp = True
                self.busy_out = False
                self._move = None
                if self.push_alarm_delay_s is not None and self._move_step in self.push_alarm_steps:
                    self.events.append((self.t + self.push_alarm_delay_s, SimJxc._raise_push_alarm))

    def _raise_push_alarm(self):
        self.alarm_ok = False
        self.svre = False
        self._svre_block_until = float("inf")

    def _on_drive_rising(self):
        c = self.cfg
        if not (self.svre and self.seton):
            return
        self._steps_at_drive.append(
            (int(c.out_in0 in self.dout) | (int(c.out_in1 in self.dout) << 1)))
        self._move_step = self._steps_at_drive[-1]
        self.busy_out = True
        self._move = (self.t + self.inp_off_delay, self.t + self.inp_off_delay + self.move_time)
