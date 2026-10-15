# dual_teleop

Per-arm jog / teleop of the dual-arm cell (`arm_left_*` / `arm_right_*`, planning frame `world`), for
recovery and for jogging an arm out of a joint limit. **Mock-tested only** (`hardware_protocol:=mock_hardware`).
Nothing here has been run on the real robots; do that only with operator approval, arms parked and the
operator at the E-stop.

```
keyboard / gamepad (unmodified single-arm nodes, remapped)
        |  /dual_teleop/in/delta_twist_cmds, /dual_teleop/in/delta_joint_cmds
        v
 /dual_teleop/selector  --- interlocks (coordinator, controller, scene), arm selection, scaling ---+
        |                                                                                          |
        +--> /arm_left/servo/delta_*_cmds  --> MoveIt Servo (group arm_left)  -> arm_left_trajectory_controller
        +--> /arm_right/servo/delta_*_cmds --> MoveIt Servo (group arm_right) -> arm_right_trajectory_controller
        +--> /dual_teleop/servo_scene (dual move_group scene, same-arm pairs allowed) --> both Servos
```

## Run

Dual stack first (same ROS domain; mock example, domain 202):

```bash
export ROS_DOMAIN_ID=202 ROS_LOCALHOST_ONLY=1
ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware
ros2 launch rebel_dual_table_moveit_config move_group.launch.py hardware_protocol:=mock_hardware use_gui:=false
ros2 launch dual_teleop dual_teleop.launch.py arm:=left keyboard:=true      # or gamepad:=true, console:=true
```

`dual_teleop.launch.py` arguments: `arm` (left), `keyboard`, `gamepad`, `console`, `keyboard_prefix`
(default `gnome-terminal --wait --`: the keyboard needs a TTY), `start_enabled` (false), `speed_scale` (0.5),
`base_command_frame` (`world` | `arm_base`), `coordinator_status_topic`, `run_servo`, `calibration_file`.

Teleop **starts paused**: no Servo is even running. Enable with console key `e`, the gamepad Back button, or
`ros2 service call /dual_teleop/enable std_srvs/srv/Trigger`. The first command after enabling / selecting /
an interlock clearing is ignored until a zero command (or a >0.3 s pause) is seen, so a key held through the
event never starts motion by itself. After a select or enable the Servo of the arm needs about 1 s to start.

Without a display run the keyboard by hand in its own terminal:

```bash
ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard --ros-args -r __ns:=/dual_teleop/keyboard \
  -r delta_twist_cmds:=/dual_teleop/in/delta_twist_cmds -r delta_joint_cmds:=/dual_teleop/in/delta_joint_cmds \
  -r servo_node/start_servo:=/dual_teleop/servo_start_stub -r /status:=/dual_teleop/servo_alive
ros2 run dual_teleop teleop_console          # the screen + arm keys
```

## Arm selection

The single-arm keyboard / gamepad nodes hard-code `base_link`, `xeg32_tool_tip` and `joint1..6` (and the
keyboard also starts Servo itself), so they cannot publish for a prefixed arm by remapping alone. They are
run unmodified with topic remaps into the selector, which translates `base_link -> world` (or the arm base,
`base_command_frame:=arm_base`), `xeg32_tool_tip -> arm_<side>_xeg32_tool_tip`, `jointN -> arm_<side>_jointN`,
scales and clamps, and forwards to the selected arm only. Foreign frames / joint names, NaN and mismatched
jogs are dropped.

Select with: console `l` / `r`; gamepad LB / RB (`gamepad:=true`); `/dual_teleop/select_left`,
`/dual_teleop/select_right` (std_srvs/Trigger); topic `/dual_teleop/select_arm` (std_msgs/String).
Switching sends zero twist + zero joint jog to the old arm first, then stops its Servo.

## Interlocks (`selector_core.py`, unit-tested)

| Condition | Effect |
|---|---|
| Teleop disabled (start state) | all commands dropped, Servos stopped |
| `CoordinatorStatus.state` EXECUTING or WAITING_BARRIER (fresh) | **both** arms blocked: zeros sent, both Servos **stopped**, BLOCKED shown with the reason |
| Coordinator status stale while the last state was blocking | stays blocked ("STALE"); `ignore_stale_coordinator` (SetBool) overrides |
| No coordinator status at all / stale non-blocking | teleop allowed, WARNING "coordinator status unknown: NOT interlocked" |
| IDLE, PLANNING, PLANNED, STOPPED | allowed (STOPPED must allow recovery jogging) |
| Selected arm's controller not `active` (polled at 1 Hz; poll not answering counts as inactive) | that arm blocked |
| No fresh planning scene from move_group (> 5 s) | both blocked (collision checking would be blind) |
| Servo of the selected arm not confirmed running | commands dropped |

The screen is `/dual_teleop/status` (JSON, 5 Hz) rendered by `teleop_console`; the selector logs transitions
(WARN for BLOCKED / unknown coordinator).

## Touch mode (knife may contact the carton / table)

With Servo's collision check an arm stops about 20 mm before the carton and 35 mm before the table, so a cut start
or a calibration touch is impossible. **Touch mode** is the deliberate, temporary exception. It is OFF at start.

* **Enter** (selected arm only): console `t` then `y` within 5 s (`t` again turns it off); gamepad: hold BOTH stick
  clicks (buttons 9 + 10, `joy_touch_buttons`) for 1 s (same combination turns it off); services
  `/dual_teleop/touch_mode_on` / `touch_mode_off` (std_srvs/Trigger). Entry is refused while teleop is disabled, a
  controller is inactive, the scene is stale, or the coordinator is in any state but IDLE / STOPPED / unknown.
* **Effect**: in the planning scene fed to the SELECTED arm's Servo only (`/dual_teleop/servo_scene_<side>`), the ACM
  allows `arm_<side>_knife_c_blade`, `knife_c_handpiece`, `knife_c_holder` and `xeg32_tool_tip` against exactly
  `detected_carton` (`carton_ids` parameter) and the `table` link of the dual model. Everything else is checked as
  before: link1..6, motors, wrist camera, the other arm (all cross-arm pairs), camera pole and head, and the
  other arm's scene is not touched. The holder is included beyond the original list: it sits ~6 mm behind the blade
  tip and its own slow-down stalls the blade ~20 mm above the carton (first gate run).
* **Speed**: `touch_speed_scale` (default `speed_scale / 3`, runtime parameter, rejected above `speed_scale`).
* **Banner / state**: `!!! TOUCH MODE LEFT: knife may contact carton/table ... !!!` on the console, a WARN in the
  selector log, and `/dual_teleop/touch_mode` (std_msgs/String JSON, latched: `on`, `arm`, `speed_scale`,
  `remaining_s`, `retreat`, `last_off_reason`; also inside `/dual_teleop/status`).
* **Auto-off** (zero command sent, scene restored at once): arm switch, teleop disable, coordinator status other than
  IDLE / STOPPED / none (PLANNING and PLANNED included), any controller inactive, stale scene, and
  `touch_mode_timeout_s` (default 120 s, 0 = never).
* **Limits seen in the mock**: the blade stops 0.9 mm above the carton top, not at 0.0: link6 sits ~26 mm above it and
  Servo's slow-down on link6 (not allowed on purpose) stalls the approach. A cut start set this way is good to ~1 mm.

**Leaving touch mode in contact, and the RETREAT window.** Servo's slow-down is exponential (x0.001 at contact) and
direction-blind. With the normal ACM an arm left touching the carton can back off at about 0.02 mm/s, i.e. it is
locked in practice (measured: 0.24 mm in 12 s with only the knife links re-checked; link6 nearby slows it as
well). It never jumps and never moves further in, but it cannot get out. Therefore, when touch mode ends by the
operator, the timeout, or a PLANNING / PLANNED coordinator, a **retreat window** opens: the same knife
allowances plus `link6` and the wrist camera stay in that arm's scene, at touch speed, until the TCP is 80 mm from
where it was (`retreat_distance_m`) or 60 s passed (`retreat_window_s`); then the normal check is back. Measured:
87 mm in 16 s after an operator OFF with the blade 0.9 mm from the carton; jogging down again then stops 20 mm above
the carton. The retreat window is shown on the screen (`~~~ RETREAT ... ~~~`). It does NOT open for the other
auto-off reasons (arm switch, disable, coordinator blocking, controller inactive, stale scene): there the arm is
blocked anyway. If the knife was left in contact that way, re-enter touch mode (deliberately), jog away, turn it off.

## Servo configuration and differences from the single-arm `servo.yaml`

`config/servo_arm.yaml` documents every difference inline ("DUAL:"). Summary:

| Parameter | single arm | dual (each arm) |
|---|---|---|
| `move_group_name` | `igus_rebel_arm` | `arm_left` / `arm_right` (dual SRDF) |
| `planning_frame`, `robot_link_command_frame` | `base_link` | `world` |
| `ee_frame_name` | `xeg32_tool_tip` | `arm_<side>_xeg32_tool_tip` |
| `command_out_topic` | `rebel_arm_trajectory_controller/joint_trajectory` | `/arm_<side>_trajectory_controller/joint_trajectory` |
| `joint_topic` | `joint_states` | `/joint_states` (all 12 joints) |
| namespace | `/` | `/arm_<side>/servo`, node `servo_node` |
| `check_collisions` | **false** | **true**, full dual model |
| `collision_check_rate` | 2 Hz | 10 Hz |
| `self_collision_proximity_threshold` | 0.01 m | 0.07 m |
| `scene_collision_proximity_threshold` | 0.02 m | 0.05 m |
| planning scene source | own, from `/planning_scene` | `/dual_teleop/servo_scene` (below) |
| start | by keyboard/gamepad at launch | by the selector, only while teleop is open for that arm |

Everything else (`publish_period`, `incoming_command_timeout`, speed units, joint margin 1.5 deg,
singularity thresholds, `publish_joint_velocities`) is unchanged. The dual `joint_limits.yaml` applies to
Servo too; its 0.1 rad/s bring-up cap was lifted to the ReBeL 0.785 rad/s on 2026-10-15 (after the e-stop
cross-stop test with the arms standing still), so the selector caps below are now what limits jogging. Selector caps: `speed_scale` 0.5,
`max_linear_mps` 0.05, `max_angular_rps` 0.3, `max_joint_rps` 0.1 (runtime-settable parameters).

### Things found while testing (they shaped the design)

* **Servo's scene.** Humble's Servo creates a namespaced monitor; in `/arm_X/servo` it listens to
  `/arm_X/servo/planning_scene` and asks `/arm_X/servo/get_planning_scene`, i.e. it sees neither the camera pole
  nor a carton. The launch remaps `planning_scene -> /dual_teleop/servo_scene` and `get_planning_scene ->
  /get_planning_scene` (global `--ros-args` rules: the monitor lives in a private sub-node). The camera pole is
  added with `apply_planning_scene`, which never shows on the `/planning_scene` diff topic, so the selector polls
  `/get_planning_scene` at 2 Hz and republishes the full scene.
* **Same-arm pairs.** Servo's proximity check counts every non-ACM pair, including pairs close by construction
  (knife wrist camera vs link5: 19 mm at home). With the stock ACM both arms crawled everywhere. The selector
  therefore allows all pairs of the SAME arm in the scene it feeds to Servo; cross-arm, arm-vs-table and
  arm-vs-object pairs are untouched. Same-arm self collision is covered by MoveIt planning only.
* **pause_servo is unsafe here.** After `unpause_servo` the first message Servo publishes carries the joint
  state from the moment of the pause: an arm that moved meanwhile (the coordinator, a pendant) would jump back.
  The selector uses `stop_servo` / `start_servo` (fresh state), see
  `scan_results/dual_arm_stack_20261014/teleop/results/servo_pause_vs_stop.txt`.
* **Slow-down is exponential and direction-blind**: scale 1 at the threshold, 0.001 at contact, applied to every
  command including moving away. An arm stopped inside the proximity zone leaves only slowly (tens of seconds at
  the former 0.1 rad/s cap).

## Tests

`cd src/dual_teleop && python3 -m pytest -p no:anyio test` (selector logic, no ROS needed); mock gates in
`scan_results/dual_arm_stack_20261014/teleop/tools/` (domain 202).
