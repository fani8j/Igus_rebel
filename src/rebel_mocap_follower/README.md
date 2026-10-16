# rebel_mocap_follower

Lets a motion-capture program on another machine drive the igus ReBeL arm through
**MoveIt Servo (JointJog)**. The mocap side only sends target joint angles; this package
validates them, gates them with a deadman, limits velocity/acceleration and sends
`control_msgs/JointJog` to Servo, which feeds the existing `rebel_arm_trajectory_controller` (through `rebel_servo_relay`, see below).
Mock only so far: nothing here has moved the real arm.

## Input contract (what the mocap program must publish)

| Topic | Type | Rate | Content |
|---|---|---|---|
| `/mocap/joint_command` (param `input_topic`) | `sensor_msgs/msg/JointState` | ~100 Hz | `name` = `joint1`..`joint6` in any order (all six, no extras, no duplicates, else the message is rejected); `position` in **radians**, URDF convention (same as `/joint_states`); `header.stamp` set |
| `/mocap/enable` | `std_msgs/msg/Bool` | >= 5 Hz | deadman: must be received `true` at >= 5 Hz (timeout 0.3 s). Any `false`, or silence > 0.3 s, releases it |

QoS: the follower subscribes best-effort/volatile, which matches both reliable and best-effort
publishers. Velocity/effort fields are ignored.

Rules the follower enforces (all produce HOLD, never a clamped move):
- Message older than `input_timeout_s` (0.2 s, by *receive* time) -> hold.
- `header.stamp` more than 1 s off the follower's clock -> message rejected (logged once). Needs
  roughly synced clocks (chrony/NTP); set `check_stamp:=false` only if you understand the offset.
- NaN/inf, missing/unknown/duplicate joint names -> hold.
- Any joint target outside URDF limit +/- `limit_margin_rad` (0.05) is clamped to that limit and
  followed (`clamp_to_limits: true`; status `clamped` lists the joints); with `false` -> hold (limits read from
  `/robot_state_publisher` `robot_description`; if unavailable the node stays disabled).
- Slow catch-up (state `CATCHUP`): entered when on the enable rising edge max |target - current|
  > `max_engage_error_rad` (0.5), when consecutive messages differ by > `max_input_step_rad` (0.2)
  while engaged, or when the error exceeds `max_engage_error_rad` while following. The arm moves
  in a synchronized straight line in joint space (all joints arrive together) at
  `catchup_velocity` (YAML 0.3 rad/s on the leading joint, `catchup_acceleration` 0.5 rad/s^2; code defaults 0.1 / 0.3),
  tracking the latest target, until max |error| < `catchup_done_rad` (0.03); then normal
  following resumes. HOLD conditions (stale input/joint_states, invalid input, blocked) still apply
  and the catch-up resumes (not full speed) when they clear.
- Error > `max_catchup_error_rad` (1.75, <= pi): REFUSED on the enable rising edge, or latched HOLD
  while enabled; only releasing the enable (false or silent) and re-enabling clears it.

## Behaviour

Per 100 Hz cycle per joint: `v = clamp(Kp*(target-current), +-max_velocity)`, additionally capped by
`sqrt(2*0.8*max_acceleration*|error|)` so it cannot overshoot, then acceleration-limited
(`max_acceleration` while following, `stop_deceleration` when holding/disabled).
Defaults (YAML): `max_velocity` 0.3 rad/s, `max_acceleration` 0.5, `stop_deceleration` 1.0, `Kp` 2.0,
deadband 0.002 rad. Hard caps enforced at start-up (node refuses to start otherwise):
`max_velocity <= 0.6`, accelerations `<= 1.5`.
When not enabled / holding / stale the command ramps to zero, zero JointJog is sent for 0.5 s,
then publishing stops (Servo's own 0.1 s `incoming_command_timeout` halts anyway). Zero jogs
are also sent on node shutdown.

Only one program may command Servo: while any other publisher exists on `/delta_joint_cmds`
or `/delta_twist_cmds` (e.g. the keyboard/gamepad teleop) the follower holds
(`refuse_if_other_servo_publishers`).

`~/status` (`/mocap_follower/status`, `std_msgs/String`, JSON at 10 Hz): `state`
(`ENABLED|HOLD|DISABLED|REFUSED`), `reason`, `max_error_rad`, `input_age_s`, `latched`,
`max_command_rad_s`, `servo_status`, `inputs`, `rejected`.

Optional joy deadman: `joy_enable_button:=<idx>` (default -1 = off) makes a held `/joy` button an
alternative to `/mocap/enable`.

## Servo interface used

Humble `moveit_servo` `servo_node_main`, `command_in_type: speed_units`, joint jog on
`/delta_joint_cmds` (relative `delta_joint_cmds` in servo.yaml with no namespace; **not**
`/servo_node/delta_joint_cmds`), velocities in rad/s, `joint_names` joint1..6, `header.stamp = now`
(Humble Servo uses it for the staleness timeout). Servo output goes to
`rebel_arm_trajectory_controller/joint_trajectory`. Humble Servo idles until
`/servo_node/start_servo` is called; the launch runs the existing `start_servo.py` for that.
Servo config is the unchanged `igus_rebel_moveit_config/config/servo.yaml` (collision checking
off, 1.5 deg joint-limit margin, 0.1 s command timeout).

## Servo speed relay (`servo_relay`, default true)

Why: MoveIt Servo 2.5.10 (Humble) computes every output from the MEASURED joint state (+ v x publish_period) and always
runs a Butterworth smoothing filter (`use_smoothing` is ignored in 2.5.10); the joint_trajectory_controller (JTC) then
restarts each stamp-0 one-point trajectory from its current state. Together only about 34 % of the commanded joint
speed was realised in mock (measured, both arms, 0.1 and 0.3 rad/s), and the follower's loop gain dropped from
`kp` 2 to about 0.7 1/s. Analysis: `scan_results/servo_speed_ratio_20261016/SOURCE_ANALYSIS.md`.

With `servo_relay:=true` (both launch files):
- each Servo uses the pass-through smoothing plugin `rebel_servo_relay::PassThroughFilterPlugin`
  (`smoothing_filter_plugin_name`), so its output velocity equals the commanded one (after Servo's own collision,
  singularity and limit scaling), and publishes to `/mocap/arm_<side>/servo/trajectory_out` (single arm:
  `/servo_trajectory_out`) instead of the JTC topic;
- one `rebel_servo_relay` node per arm (`/mocap/arm_<side>/servo/trajectory_relay`, single arm `/servo_trajectory_relay`)
  keeps a reference position r and publishes to the JTC topic a stamp-0 trajectory with three points: t=0 at r
  (velocity v'), t=`horizon_s` at r + v' x horizon (v') and t=`horizon_s`+`stop_time_s` at r + v' x (horizon + stop_time/2)
  with velocity 0 (constant deceleration, v'/`stop_time_s` = 2.9 rad/s^2 at 0.3 rad/s; the planner limit is 5.24 rad/s^2,
  so keep v below about 0.53 rad/s). v' is Servo's velocity scaled down near a joint limit so that the whole profile ends at
  or before limit minus margin (never faster than v, never beyond the bound).
- r is NOT integrated separately: at each Servo message it is the previously published trajectory evaluated at the elapsed
  time (the same cubic Hermite the JTC samples), so a direction reversal has no reference jump. The first point sits at
  time 0, so the JTC follows r instead of restarting from the measured state. r is then bounded to
  `min(max_lead_rad, lead_time_s x |v| + lead_floor_rad)` from `/joint_states` and to the joint limits minus margin.
- Every trajectory ends at rest. If Servo or the relay dies, the JTC plays out the previous segment and holds a
  zero-velocity point. A watchdog in the relay publishes ONE hold (at the measured position if fresh, else at the
  reference) when no valid Servo message arrived for `silence_hold_s`; a message that cannot be forwarded while a reference
  is active also produces that hold instead of nothing. The next valid message reseeds r from the measured state.
  Servo itself publishes nothing once its commands are stale, so without the watchdog a silent release would coast the
  full segment.
- Overrun bound after the last forwarded message (relay or Servo dead): the arm travels to the final point, i.e.
  **lead + |v'| x (horizon + stop_time/2)** beyond the last measurement (the lead is the reference's lead over the measured
  position, up to the lead bound), plus the physical braking of the plant. With the defaults at 0.3 rad/s that is up to
  0.05 + 0.046 = 0.096 rad; mock position interface measured 50 mrad, the velocity-law mock 89 mrad (it approaches the final
  point with the P term only, tau 1/p = 1 s).
- The relay is a node in the Servo namespace; the follower's exclusivity gate (`own_servo_node`) accepts it, and with the
  relay on the dual launch also sets `own_publisher_node` so the gate requires EXACTLY ONE own publisher on the JTC topic,
  the relay node: a second relay, or a Servo still wired to the JTC, holds the arm (and a missing relay holds too).

Relay parameters (defaults in the node; the launch sets topics, `publish_period_s` from Servo's `publish_period`, and
`max_lead_rad`/`lead_time_s` from the launch arguments `relay_max_lead_rad` in (0, 0.15] and `relay_lead_time_s` in [0, 1]):

| Parameter | Default | Meaning |
|---|---|---|
| `publish_period_s` | 0.034 | Servo's `publish_period` (nominal message interval) |
| `horizon_s` | 3 x period = 0.102 | length of the constant-velocity segment |
| `stop_time_s` | = `horizon_s` | length of the final deceleration to velocity 0 (must be > 0 and finite; negative = default) |
| `reset_timeout_s` | 3 x period = 0.102 | gap between Servo messages above which r is re-seeded from `/joint_states` (the previous segment has played out) |
| `max_lead_rad` | 0.05 | absolute bound on abs(r - measured) per joint |
| `lead_time_s`, `lead_floor_rad` | 0.2, 0.005 | the lead bound per joint is min(max_lead, lead_time x abs(v) + floor): near an obstacle Servo scales v down and the retained lead shrinks with it |
| `silence_hold_s` | 2 x period = 0.068 | watchdog: no valid Servo message for this long while active -> one hold. Mock: Servo gaps during a 30 s steady jog of both arms were at most 35.8 ms, so no trips in normal operation |
| `joint_limit_margin` | 0.02617994 | r and the profile stay this far inside the URDF limits (same as Servo) |
| `require_stamped_joint_states` | true | `/joint_states` samples with `header.stamp` 0 are rejected (the broadcaster always stamps; freshness cannot be checked otherwise) |
| `joint_state_timeout_s` | 0.2 | `/joint_states` older than this (receipt age, or header.stamp age against the ROS clock when stamped) is stale: hold, no forwarding |
| `zero_velocity_eps` | 1e-4 | all abs(v) below this = Servo halted: r dropped and a one-point hold at the measured position is published |
| `robot_description_node` | `/robot_state_publisher` | URDF limits source (or the `robot_description` parameter); nothing is forwarded until limits are known |

Messages without velocities, with mismatched sizes, NaN/inf, duplicate or unknown joints are not forwarded (throttled error
log). All validation runs first: a rejected message while a reference is active holds the OLD reference (old joint names); only a valid message with a different joint name list (order included) drops it and reseeds. Both inputs have QoS depth 1 and a message received before the last watchdog hold is dropped, so queued old Servo messages cannot overwrite a hold. The single-arm launch uses the same exclusive-publisher gate as the dual one when the relay is on.

`servo_relay:=false` restores the old direct Servo -> JTC wiring (Butterworth filter, about 34 % speed in mock).
`servo_relay:=true` together with `start_servo:=false` is a launch error: the Servo node would not be ours to rewire.
Requires the package `rebel_servo_relay` (built from this workspace).

**Real-arm prediction (not run on hardware; checked against the mock with the real controller law).** The real JTC
(velocity interface, `controller.yaml`: p 1.0, ff_velocity_scale 0.4) drives `ff x v + p x (r - measured)`. A steady
speed v needs the lead `v x (1/G - ff)/p` (G = plant gain 1.0 to 1.26), i.e. 0.39 to 0.6 s x v, and the lead bound caps it:
`u = min(v, G x (ff x v + p x lead_bound))`. With the defaults (lead bound 0.2 s x v + 0.005, at most 0.05): G = 1 gives
57 % at 0.3 rad/s and 65 % at 0.1 rad/s (mock with `calculate_dynamics` velocity interface, G = 1, p 1.0/ff 0.4:
56.6 % and 64.8 %, matching); the old direct wiring gives 26 % there (model 26 %). Full speed needs
`relay_lead_time_s` of about 0.6 and a larger `relay_max_lead_rad` (0.15 gives 89 % at 0.3 rad/s, 100 % at 0.1 rad/s in the
velocity-law mock), at the cost of a larger lead that is worked off after a release and of a slower build-up (time
constant 1/p = 1 s). The `controller.yaml` gains are shared with planned cut execution and were not changed.
Verify with a slow, supervised jog first. Mock evidence: `scan_results/servo_speed_ratio_20261016/fix/`.

## Run (mock)

```bash
export ROS_DOMAIN_ID=179 ROS_LOCALHOST_ONLY=1      # an unused domain; the real robot is on 11
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
ros2 launch igus_rebel_moveit_config move_group.launch.py hardware_protocol:=mock_hardware use_gui:=false
ros2 launch rebel_mocap_follower mocap_follower.launch.py hardware_protocol:=mock_hardware   # Servo + follower
ros2 run rebel_mocap_follower fake_mocap_publisher --mode sine --amp 0.06 --freq 0.1
ros2 run rebel_mocap_follower mocap_enable_keyboard      # Enter = enable, Ctrl-C = release
ros2 topic echo /mocap_follower/status
```

`start_servo:=false` if Servo is already running (e.g. `move_group.launch.py run_teleop:=true`);
never run two Servo nodes. `fake_mocap_publisher --mode` also offers `jump`, `nan`, `stop`, `far`,
`badnames`, `limit` (see `--help`).

`mocap_enable_keyboard` publishes `true` at 10 Hz while running and `false` on exit
(Ctrl-C/SIGTERM/terminal close); a crash is also safe because silence > 0.3 s releases the deadman.

## Real arm (not yet done; needs an operator at the e-stop)

Use `hardware_protocol:=rebel` in all launches, the same ROS domain as the robot stack, start with the
defaults, and first verify the Servo-to-motion gain (see below).

## Dual-arm table cell (one mocap stream, both arms)

`mocap_follower_dual.launch.py` starts, per arm, a follower node (`mocap_follower_left` / `mocap_follower_right`,
status `/mocap_follower_<side>/status`) and its own MoveIt Servo (`/mocap/arm_<side>/servo`, collision checking on,
built like dual_teleop's, JointJog on `/mocap/arm_<side>/servo/delta_joint_cmds`), plus one planning-scene relay
(`/mocap/servo_scene`, same-arm ACM pairs allowed, like dual_teleop's selector) and a one-shot `start_servo` caller.
Input is `/dual_arm/isaac_joint_commands` (both arms + extra joints, extras ignored); one deadman `/mocap/enable`
drives both. The arms are independent: if one holds, latches or refuses, the other keeps following.

Extra interlocks (params, empty/off in single-arm mode, on in `config/mocap_follower_dual.yaml`), each holds that arm:
- `coordinator_status_topic`: hold while the coordinator is WAITING_BARRIER or EXECUTING, or silent > `coordinator_timeout_s`.
- `trajectory_controller`: hold unless that controller is `active` (list_controllers at 1 Hz, answer must be < 3 s old).
- `exclusive_command_topic` (+ `own_servo_node`, `exclusive_allowed_publishers`): hold if anything but our own Servo
  (and the allowed `/dual_cut_coordinator`, which has its own gate) publishes on the JTC command topic, e.g. dual_teleop.

Real robot (operator has already brought up the dual stack and activated the controllers):
```bash
# 1. STOP dual_teleop first (mocap and dual_teleop must never run together; the followers hold if it is up)
export ROS_DOMAIN_ID=<the robot stack's domain>
ros2 launch rebel_mocap_follower mocap_follower_dual.launch.py hardware_protocol:=rebel   # same calibration_file as the stack
# 2. check, both must say DISABLED "enable false" and Servo status 0
ros2 topic echo /mocap_follower_left/status ; ros2 topic echo /mocap_follower_right/status
# 3. enable (Enter = deadman on, Ctrl-C = release both arms)
ros2 run rebel_mocap_follower mocap_enable_keyboard
```
Options: `arms:=left|right|both`, `start_servo:=false`, `calibration_file:=...`, `params_file:=...`,
`scene_collision_proximity_threshold:=<m>` (below), `servo_relay:=true|false`, `relay_max_lead_rad`, `relay_lead_time_s` (above).

**Scene collision slow-down distance.** The mocap Servo nodes are built from `dual_teleop/config/servo_arm.yaml`, but
the launch overrides `scene_collision_proximity_threshold` (camera pole / camera_head, carton) to the launch argument
`scene_collision_proximity_threshold`, default **0.025 m** (servo_arm.yaml and dual_teleop keep 0.05 m).
`self_collision_proximity_threshold` (other arm, table) stays at 0.07 m. Reason: with the left arm upright at home its
wrist camera is only 32 mm from the `camera_head` scene box, inside Servo's 50 mm slow-down zone, so from home the arm
only crawled under mocap. Trade-off: Servo's slow-down is exponential (scale 1 at the threshold, 0.001 at contact), so
a jog toward a scene object stops at about 0.44 x threshold: about 11 mm at 0.025 instead of about 22 mm at 0.05, i.e.
less stopping margin to the pole / head / carton. The value must be in (0, 0.05]; anything else aborts the launch with
an error (it is not clamped). Only the mocap Servo nodes are affected, not dual_teleop, planning or the single-arm launch.

**Leave home first (procedure).** Even with 0.025 m the left arm is closest to the camera head in its upright home pose.
Before enabling mocap, move the left arm out of home to a bent ready pose with MoveIt/RViz (planning is not slowed by
Servo): in RViz select planning group `arm_left`, set the goal state to the SRDF named state **`ready`** (joint2 0.6 rad,
joint3 0.9 rad, other joints 0), Plan and Execute, and only then start the mocap stream / `mocap_enable_keyboard`.
Do this with the deadman released (followers DISABLED).

Mock (isolated domain; the real robot is on 11):
```bash
export ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1
ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=mock_hardware
ros2 launch rebel_mocap_follower mocap_follower_dual.launch.py hardware_protocol:=mock_hardware
ros2 run rebel_mocap_follower fake_mocap_publisher --dual --mode sine --amp 0.1 --freq 0.2
ros2 run rebel_mocap_follower mocap_enable_keyboard --no-prompt
# one-arm fault, the other keeps following: --mode jump|far|limit|badnames|stop|nan --event-arm right --event-time 14
```

## Tests

`cd src/rebel_mocap_follower && python3 -m pytest -p no:anyio test` (pure logic: validation,
latches, engage, limits, no-overshoot, timeouts; also run by `colcon test`).
