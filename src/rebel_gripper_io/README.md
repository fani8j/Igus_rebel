# rebel_gripper_io

Handshake node for an SMC **JXC61** (PNP, parallel I/O step controller) driving an LEHF electric
gripper, run through the igus ReBeL driver's digital I/O (`igus_rebel` package):

* reads the inputs from the driver topic `<driver>/digital_io` (`igus_rebel_msgs/DigitalIOState`),
* switches outputs with the driver service `<driver>/set_digital_outputs`
  (`igus_rebel_msgs/SetDigitalOutputs`, iRC numbering, success = "sent"),
* exposes the sequences as services and publishes `~/state` (`igus_rebel_msgs/JxcGripperState`).

All pin numbers are iRC panel numbers (base I/O DIn/DOut21-27, arm/tool I/O DIn/DOut31-32).
**The pin table below is operator confirmed (2026-11-13, both arms the same).** Override it with
`params_file:=` (ROS parameters of `jxc_gripper_node`) if the wiring changes.

## Pin table (defaults)

| Parameter | iRC | Direction | JXC signal |
|---|---|---|---|
| `out_in0` | DOut21 | robot -> JXC | IN0 (step select bit 0) |
| `out_in1` | DOut22 | robot -> JXC | IN1 (step select bit 1) |
| `out_setup` | DOut23 | robot -> JXC | SETUP (return to origin) |
| `out_drive` | DOut24 | robot -> JXC | DRIVE (start step) |
| `out_reset` | DOut25 | robot -> JXC | RESET (alarm reset only, see `reset_alarm`) |
| `out_svon` | DOut26 | robot -> JXC | SVON (servo on) |
| `in_seton` | DIn21 | JXC -> robot | SETON (origin set) |
| `in_inp` | DIn22 | JXC -> robot | INP (in position) |
| `in_svre` | DIn23 | JXC -> robot | SVRE (servo ready) |
| `in_alarm` | DIn24 | JXC -> robot | *ALARM: input ON = healthy, OFF = alarm (or wire break) |
| `in_busy` | -1 (not wired) | JXC -> robot | BUSY (optional, see below) |

The tool DOut31/32 are used for the knife and are not touched here.

## Services (`~` = `/jxc_gripper_node` or `/arm_<left|right>/jxc_gripper_node`)

| Service | Type | Sequence |
|---|---|---|
| `~/servo_on` | Trigger | refuse if io stale, alarm or latch; SVON on; confirm SVON in dout; wait SVRE (`servo_on_timeout_s` 3.0; aborts if SVON disappears); on failure SVON off |
| `~/home` | Trigger | requires SVRE; if SETON is already ON: success "already homed", **no SETUP pulse**; else SETUP on, confirm SETUP in dout, wait SETON and INP (and BUSY off if wired) from samples newer than the SETUP request (`home_timeout_s` 30; aborts on SVRE drop / SETUP dropped), SETUP off |
| `~/move_step` | JxcMoveStep (`step` 0..3) | requires SVRE, SETON, no alarm, DRIVE off; set IN0/IN1; confirm both bits in dout; wait `step_setup_delay_s` 0.05; re-check fresh io, alarm, SVRE, SETON and IN0/IN1 still selected; DRIVE on; **confirm DRIVE in dout first**; then (BUSY not wired) wait INP **off** (`inp_clear_timeout_s` 0.5) and INP **on** (`move_timeout_s` 5.0), or (BUSY wired) wait BUSY on, then BUSY off and INP on. Aborts on alarm, stale io, SVRE or SETON dropping, or DRIVE disappearing from dout. DRIVE off at the end. For a pushing/grip step the final timeout means "no part gripped / position not reached" |
| `~/servo_off` | Trigger | DRIVE, SETUP, SVON off (confirmed); wait SVRE off |
| `~/reset_alarm` | Trigger | only when *ALARM is OFF (alarm active), io fresh, no other command, no latch; DRIVE and SETUP off (confirmed); RESET on, confirm in dout; wait *ALARM ON (`reset_timeout_s` 1.0) in a sample newer than the request; RESET off (confirmed) on every exit. If the alarm stays: "alarm did not clear: check the JXC alarm code; some alarms need a power cycle". RESET not confirmed off -> `outputs_uncertain` |
| `~/release_all` | Trigger | all six JXC outputs off, confirmed in a fresh digital_io sample; clears the `outputs_uncertain` latch |

Rules:
* One command at a time (a concurrent call is refused with "busy"). Every command is refused when
  `digital_io` is invalid or older than `stale_s` (0.3 s).
* **Freshness comes from the source**: age = now - `header.stamp` of the digital_io message (the
  driver's STATUS receive time, system clock, same machine), not the DDS receipt time. A zero stamp
  is invalid, a stamp older than the last accepted one is ignored, and the driver's 10 Hz heartbeat
  that republishes an old STATUS keeps its old stamp and ages out. A confirmation only counts from a
  sample whose stamp is newer than the moment the request was sent.
* **The driver's "success" only means "sent"**. After every cleanup (DRIVE/SETUP off, plus RESET off after `reset_alarm`), `servo_off`
  and `release_all` the node confirms the pins are OFF in dout (`dout_confirm_timeout_s`). If that
  fails, or any set-outputs call timed out / raised / failed, the latch **`outputs_uncertain`**
  is set (shown in `~/state` and in `last_error`). While latched `servo_on`/`home`/`move_step` are
  refused; only `release_all` (and `servo_off`) are allowed, and a `release_all` whose all-off is
  confirmed clears the latch.
* A step with DRIVE already on is refused; use `release_all`. Refusals send nothing.
* **Automatic alarm reset** (operator decision 2026-11-13; on this JXC a grip, i.e. a pushing step,
  raises an alarm right after it completes: SVRE and *ALARM go OFF, SETON/INP stay ON; RESET clears
  *ALARM in ~26 ms and SVRE returns by itself ~190 ms after RESET goes off, no re-home needed).
  A 20 Hz monitor (own callback group) runs the same logic as `~/reset_alarm` (the only RESET path)
  when `auto_reset_alarm` (default **true**), digital_io is fresh, *ALARM is OFF, no command runs and
  `outputs_uncertain` is not latched. A running command is still aborted by the alarm as before; the
  reset happens afterwards. A `move_step` that sees the alarm only after it returned keeps its success.
  Afterwards, if SVON is commanded on, it waits up to `servo_on_timeout_s` for SVRE and logs a WARN
  ("auto reset #n: alarm cleared, SVRE back" / "... SVRE did not return"). It never turns SVON on,
  homes or moves. Service calls arriving while an auto reset runs (including the SVRE wait) are
  refused with "busy: auto alarm reset running".
  Rate limit: more than `auto_reset_max` (3) auto resets within `auto_reset_window_s` (60 s), or one
  auto reset that does not clear the alarm, logs an ERROR and disables auto reset (no retry loop)
  until a successful manual `~/reset_alarm` or a node restart.
* `~/status` (std_msgs/String JSON, on change and 1 Hz): `valid, alarm, svre, seton, inp,
  auto_reset` ("enabled"/"disabled"), `auto_reset_reason`, `auto_reset_running`,
  `auto_reset_count`, `last_auto_reset` (epoch s or null).
* `~/state`: `valid, svre, seton, inp, alarm_ok, outputs_uncertain, busy, last_error`, 10 Hz and on change.
* Shutdown (SIGINT/SIGTERM): the node keeps its executor running, aborts a running command and
  sends DRIVE/SETUP off, confirmed in digital_io, with bounded timeouts; the log says whether it was confirmed.

## Safety limits (read this)

* **DRIVE off is not a guaranteed stop of a move the JXC has already started.** HOLD is not wired
  and RESET is used only for alarm reset (not as a stop: its operation-interruption behaviour is
  unverified on this hardware), so there is no way to stop or cancel a running step from this node; aborting a command
  only releases DRIVE and stops waiting.
* On a robot error / e-stop the robot's outputs may drop (e.g. SVON off); the node sees this as
  SVRE or the output disappearing from dout and aborts, but the gripper state is then unknown.
* **Recommended: wire the JXC BUSY output to a spare DIn (e.g. DIn25, free) and set `in_busy: 25`.**
  Without it, move completion is inferred from INP going OFF then ON, which is an assumption about
  JXC timing and cannot tell "move finished" from "INP was low all along" (a step to the current
  position never clears INP and is reported as a failure). With BUSY the start of motion is
  observed directly.

## Launch

```bash
ros2 launch rebel_gripper_io jxc_gripper.launch.py                 # single arm: /igus_rebel/digital_io, /set_digital_outputs
ros2 launch rebel_gripper_io jxc_gripper.launch.py arm:=left       # /igus_rebel_left/digital_io, /igus_rebel_left/set_digital_outputs, node /arm_left/jxc_gripper_node
ros2 launch rebel_gripper_io jxc_gripper.launch.py arm:=right
```

The dual-arm driver nodes are named `igus_rebel_left` / `igus_rebel_right` (hardware parameter
`node_name`, `rebel_dual_table_description/urdf/rebel_dual_table.urdf.xacro`); the single arm is
`igus_rebel`.

## To verify on the real robot (assumptions)

* RESET behaviour: whether the JXC clears every alarm on RESET (some need a power cycle) and RESET during a move is unverified.
* Driver numbering: CRI `DOUT n` switches iRC DOut(n+1) (measured), the STATUS `DIN`/`DOUT` bit
  mapping is assumed to be bit b = CRI number b = iRC b+1 (`dio_irc_offset`, default 1). Switch DOut31 and
  watch which `dout_raw` bit sets on `<driver>/digital_io`.
* JXC timing (when `in_busy` is not wired): after DRIVE on, INP is assumed to go OFF within
  `inp_clear_timeout_s`; INP ON again means the step finished.
* That the driver's digital_io header stamp is the STATUS receive time on the same system clock.
* \*ALARM polarity (ON = healthy) and that SETON/INP are both ON after homing.

## Tests

```bash
cd src/rebel_gripper_io && python3 -m pytest -p no:anyio test/test_jxc_handshake.py     # pure, simulated JXC
# ROS integration with a fake driver (domain 228, override with JXC_TEST_DOMAIN, localhost only), needs the workspace sourced:
ROS_DOMAIN_ID=228 ROS_LOCALHOST_ONLY=1 python3 -m pytest -p no:anyio test/test_jxc_ros.py
```
