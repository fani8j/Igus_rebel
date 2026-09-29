# side_tape_ros — knife-camera side-tape correction

Keeps the knife tip a fixed lateral offset from a tape edge on the box side face during the
side cut. MoveIt moves the arm to the pre-cut pose; this package then aligns the knife with
MoveIt Servo and executes the cut with vision feedback. Design:
`~/Downloads/rgbd_captures/docs/side_seam_architecture.md`. Core (pure Python):
`~/Downloads/rgbd_captures/precision_box_detector/precision_box_detector/side_tape/`
(found via `SIDE_TAPE_CORE`, see `core_path.py`).

## Nodes and interfaces
| Name | What |
|---|---|
| `perception_node` (`side_tape_perception`) | `/knife_camera/image_raw/compressed` (JPEG; `use_compressed:=false` for raw) → `/side_tape/observation`, `/side_tape/debug_image`; services `/side_tape/set_seed`, `/side_tape/reset_tracker` |
| `controller_node` (`side_tape_controller`) | observation + TF + Servo status (`robot.servo_status_topic`, default `/status` -- NOT `/servo_node/status`: Servo's own topics resolve relative to its namespace, not its node name; only its *services* like `start_servo` are under `/servo_node/`, verified against a real MoveIt Servo node) → `robot.servo_twist_topic` (default `/delta_twist_cmds`; only when `dry_run:=false`), always `/side_tape/twist_preview`; `/side_tape/status`; services `/side_tape/start_align`, `/side_tape/start_cut`, `/side_tape/abort`, `/side_tape/acknowledge_abort`, `/side_tape/resume` |
| `sequencer` | CLI: `--measure-only` (no motion) / in-material: lock → cut / `--mode air_align`: align → (Enter) → cut |
| `calib_tip_mask` | click the knife tip + blade/holder mask → writes `mount.tip_uv`, occluder mask |
| `calib_jacobian` | Small Servo jogs over a printed dot → writes `mount.J` (px per mm); compressed camera input by default (`--raw` for Image) |

Config: `config/knife_mount.yaml` (source tree; edit there). Control is refused until
`mount.calibrated: true` (needs `tip_uv` and `J`).

Cut frame (from the knife model: tip along tool +Z, sharp edge faces tool −Y), in `xeg32_tool_tip`:
cut = −Y, lateral = +X, depth = +Z (into the box). Frozen into `base_link` at align/cut start.

## Build
```bash
cd ~/igus_rebel_ros2 && source /opt/ros/humble/setup.bash
colcon build --packages-select side_tape_interfaces
colcon build --packages-select side_tape_ros
source install/setup.bash
```

## Normal flow (start_mode: in_material, decided 2026-09-28)
MoveIt (static-camera plan) places the blade **in the cardboard** at the cut start and goes idle →
`sequencer --measure-only` shows how far off the tape target we are → if |e_lat| ≤
`control.max_start_error_mm`, `sequencer` starts the cut: Servo drives forward along the cut and steers
toward the target **only while moving forward** (|v_lat| ≤ `steer_ratio_max`·v_cut, never sideways at a
standstill). Larger error → refused; re-plan with MoveIt using the measured offset.
`start_mode: air_align` keeps the older flow (1 mm probe + align in the air, then cut).
During Servo phases the controller allows collisions between `robot.carton_allowed_links` and the carton
(`detected_carton`) in the planning scene and restores the matrix afterwards; the holder/arm stay checked.

## First run on the robot — in this order, stop at the first surprise
0. **Bring-up + contract:** robot + MoveIt with Servo (`move_group.launch.py ... run_teleop:=true`), knife
   camera. Check `ros2 topic hz /knife_camera/image_raw/compressed` ≈ 30. The controller checks Servo's
   frame/input type/timeout at startup and refuses live mode on mismatch. Recommended (robot config, owned
   by the robot bringup): `incoming_command_timeout: 0.1` instead of 0.3 (a crash mid-cut then coasts
   ≤ 1.5 mm instead of 4.5 mm).
1. **Tip + mask** (robot still): `ros2 run side_tape_ros calib_tip_mask`.
2. **Jacobian — blade REMOVED or guarded**, printed dot on a flat board at the **same camera-to-surface
   distance as cutting** (J scales with distance): `calib_jacobian --dry-run`, then live with the e-stop in
   hand. Default image subscription is `<camera.image_topic>/compressed`; use `--raw` for Image.
   First jog must travel ≈ 5 mm; more → e-stop. Calibration refuses motion until MoveIt
   trajectory status is received; an active goal or stale (> 1 s) Servo status aborts motion.
   RMS residual > 1 px or |J| < 0.5 px/mm prevents saving; `--force-save` overrides only the finite fit gate.
   The original config is copied to `<config>.bak-<UTC timestamp>` before a successful write; the
   calibration time and worst-column residual are a YAML comment, not a config key.
3. **Perception only** with MoveIt at the real start pose: `ros2 launch side_tape_ros side_tape.launch.py
   dry_run:=true`, view `/side_tape/debug_image`. **Hard gate:** the tracked line must be the intended tape
   edge (not the box silhouette/crease). Seed it if needed (`/side_tape/set_seed`), then write the observed
   polarity into `target.edge_polarity` (don't rely on latch-at-first-lock).
4. **Axis signs — blade removed, in the air**, `dry_run:=false`, config for first motion:
   `control.retract_mm: 5`, `control.v_retract_mps: 0.005`, `control.v_cut_mps: 0.005`. Run
   `sequencer --mode air_align`: the probe must move the tool the way the debug image expects; Ctrl-C →
   the ABORT retract must move AWAY from where the box would be (+depth = into the box). Wrong → stop and fix
   `robot.*_axis_tool`. Also set a non-zero `target.offset_mm` once and check the target line in the debug
   image lands on the intended side of the edge.
   Also measure the **stop overrun**: during an air cut, abort and read how far `along_mm` keeps growing after
   the stop (the arm lags with tau ≈ 0.7 s). Set `control.decel_time_s` ≈ overrun / v_cut; the config refuses
   settings where v_cut·(latency + decel + 0.1 s) ≥ `max_blind_mm` (the cut would HOLD on normal frames).
   Default cut speed is 10 mm/s for this reason.
5. **In-material, measure only:** blade on, MoveIt places it in a scrap box →
   `sequencer --measure-only --cut-length-mm L --seed-json <planner snapshot> --face left|right`.
   With the static camera's `side_tape_seed` (docs/side_tape_seed_contract.md) it also checks the start
   pose: blade depth behind the outer wall vs `seed.expected_penetration_mm` (must be set), depth axis into
   the face, cut along the top edge, cut stays on the face, and the **tracked line is below the top edge**
   (a line on the top edge = box silhouette → refused, exit 2). The in-material cut runs the same check
   before calling start_cut when `--seed-json` is given.
6. **Scrap-box cut** at `v_cut_mps: 0.01`, short length, operator on the e-stop.
7. **Tip under load:** after a scrap cut, compare the debug-image tip marker with the real blade tip; a shift
   > ~0.5 mm-equivalent eats the budget → re-run `calib_tip_mask` with the blade loaded.

**Abort:** `ros2 service call /side_tape/abort std_srvs/srv/Trigger` (retract along −depth, judged by TF).
**E-stop / Servo HALT / stall:** the controller latches `ABORT(operator_required)` and never moves on its own
again. Before re-enabling drives: call `/side_tape/abort`, Ctrl-C the sequencer; after re-enable, withdraw
manually (Servo teleop along −depth, or MoveIt), then `/side_tape/acknowledge_abort`.

## Record data for tuning (do this on every run)
```bash
ros2 bag record /knife_camera/image_raw/compressed /tf /tf_static /side_tape/observation \
  /side_tape/status /side_tape/twist_preview /status /delta_twist_cmds
```
Offline: `python3 -m precision_box_detector.side_tape.replay --config <yaml> --input <bag dir> --topic /knife_camera/image_raw/compressed --out <dir>`.

## Tested so far (2026-09-28, no robot)
- Core: 121 unit tests (`precision_box_detector/tests/side_tape`), ROS helpers: 52 (`test/`).
- After the advisor review: in-material 3 mm start → DONE with |v_lat| ≤ 0.15·v_cut; 7 mm refused; injected
  stall → ABORT(stall, operator_required); HALT then TF gap → stays ABORT; abort retract judged by TF.
- End-to-end with `dry_run:=false` against a FAKE Servo (`robot.servo_ns: /fake_servo`, own
  ROS_DOMAIN_ID) + synthetic JPEG camera + simulated tool (scratch `e2e/fake_plant.py`): 3 mm and 7 mm
  initial errors align and cut to DONE (max |e_lat| 0.29–0.33 mm); tape drifting 1.8 mm over 60 mm tracked
  with 0.3 mm lag; operator abort mid-cut stops the cut and retracts along −depth only.
- Transport: raw 1280×720 over DDS arrived at 1.7 Hz (best effort) / 8.5 Hz (reliable); JPEG at 28.5 Hz →
  perception uses `/compressed` by default. Check `ros2 topic hz /knife_camera/image_raw/compressed` ≈ 30.
- NOT tested: real camera frames, calib_jacobian on hardware (run it supervised; the first
  jog's travel should be ~5 mm — if it moves more, e-stop).

## Tested against a REAL MoveIt Servo + MoveIt + ros2_control mock_hardware (2026-09-28, no physical robot)
- `ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware` +
  `ros2 launch igus_rebel_moveit_config move_group.launch.py hardware_protocol:=mock_hardware use_gui:=false
  run_teleop:=true run_camera_pole:=false run_box_h_cut_executor:=false` (ROS_DOMAIN_ID isolated,
  ROS_LOCALHOST_ONLY=1). `ros2 control list_hardware_components` confirmed `mock_components/GenericSystem`.
- Servo params match `RobotCfg` defaults exactly: `robot_link_command_frame=base_link`,
  `command_in_type=speed_units`, `incoming_command_timeout=0.1`; `servo_params_ok()` passes.
- **Fixed a real bug found here**: `controller_node`/`calib_jacobian` were subscribing/publishing Servo's
  status and twist topics under `{servo_ns}/...` (`/servo_node/status`, `/servo_node/delta_twist_cmds`).
  Against a real (non-namespaced) `servo_node`, those topics are actually the bare `/status` and
  `/delta_twist_cmds` -- only Servo's *services* (`start_servo` etc., which use `~/...`) are under
  `/servo_node/`. The fake-Servo harness fully namespaced everything under `/fake_servo`, hiding this.
  Fixed by adding `RobotCfg.servo_status_topic`/`servo_twist_topic` (defaults `/status`/`/delta_twist_cmds`)
  and using them instead of `{servo_ns}/status` / `{servo_ns}/delta_twist_cmds`.
- **Fixed a second real bug**: `_apply_carton_acm()` refused to widen the ACM ("carton not in the planning
  scene ACM; not modifying collisions") whenever `detected_carton` had no ACM row yet -- which is the
  *normal* case for a freshly-added `CollisionObject` on a real move_group (no row is auto-created). The
  fake harness pre-seeded `detected_carton` into its fake ACM's `entry_names` from the start, hiding this.
  Fixed: fall through to `acm_add_allowed_pairs()` (which already adds a new row for an unknown name)
  instead of skipping. Verified end-to-end against the real `/get_planning_scene` /
  `/apply_planning_scene`: `detected_carton` absent → `start_cut` widens it to
  `['xeg32_knife_handpiece', 'xeg32_tool_tip']` → phase exit restores it to absent again.
- **Blocker for a first real-robot run**: MoveIt Servo (`ros-humble-moveit-servo` 2.5.10) intermittently
  stops advancing the mock arm's commanded joint positions entirely -- sometimes after a fraction of a
  millimetre, sometimes after continuing to move for several seconds -- while continuing to publish
  `rebel_arm_trajectory_controller/joint_trajectory` at 100 Hz with *unchanging* positions, `/status`
  staying `0` (NO_WARNING) throughout and `/servo_node/collision_velocity_scale` staying `1.0`. Reproduced
  with a bare hand-rolled 100 Hz `/delta_twist_cmds` publisher (bypassing side_tape_ros entirely), so this
  is not a side_tape_ros bug. A plain `unpause_servo` after a `pause_servo` did not reliably resume motion;
  a fresh `start_servo` sometimes did, sometimes did not. Root cause not identified (ruled out: collision
  deceleration -- scale stayed 1.0; singularity/joint-bound HALT -- status stayed 0; our own publisher
  going stale -- confirmed continuous 100 Hz with fresh timestamps; a second stray publisher on the same
  topic). Effect on `side_tape_ros`: every real in-material `start_cut` in this environment stalled within
  a fraction of a millimetre and correctly went `ABORT(stall, operator_required)`; an `air_align` probe
  correctly refused with `probe_no_motion:measured=0.000mm` rather than proceeding blind. Both are the
  *intended* fail-safe behaviour of the existing stall/probe checks reacting to a real absence of motion,
  not false positives -- but no controlled cut was completed. Recommend: file/search a moveit_servo issue,
  try `command_out_type: std_msgs/Float64MultiArray` with a `forward_command_controller` instead of
  `trajectory_msgs/JointTrajectory` streamed to `joint_trajectory_controller`, or test on a newer
  `moveit_servo` release, before attempting a physical-robot run.
- Collision-allowance behaviour under Servo (task item 5b) was not conclusively demonstrated: `/status`
  stayed `0` throughout, whether or not the carton was ACM-allowed, most likely because
  `xeg32_tool_tip` has no collision geometry (logged by move_group/Servo at startup) and the scratch
  carton box's placement relative to `xeg32_knife_handpiece`'s actual collision mesh was not verified.

## Behaviour notes
- Perception auto-resets a LOST tracker only while the controller is IDLE/DONE (never in ALIGN/READY/CUT/HOLD).
- The align probe (1 mm lateral) is checked against the TF-measured motion; if TF shows no motion
  (Servo paused/ignored) it aborts with `probe_no_motion`.
- TF older than 0.2 s → arm not owned → HOLD.

## Servo notes (from the real-Servo mock tests + user experience, 2026-09-28)
- Servo must be started once (`/servo_node/start_servo`) — the controller does this on the first live start.
- Never start from the **home** pose (singular): Servo refuses/halts there. MoveIt must place the arm at a
  normal working pose first (the in-material start pose is fine).
- Topics are `/delta_twist_cmds` and `/status` (root namespace); only the services live under `/servo_node/`.
- **Real robot, 2026-09-29:** with `publish_period: 0.01` Servo did not move the real arm at calibration speeds
  (0.00 mm at 5 mm/s): the JTC (velocity interface, 100 Hz) samples every new stamp-0 Servo trajectory only at its
  start (= current state). `servo.yaml` now has `publish_period: 0.034` so the JTC runs ~3 cycles per message.
  Expect the arm to follow somewhat slower than commanded; calibration fits the TF-measured motion.
  That alone was not enough (bag: JTC reference velocity 0, output = P x one Servo step ≈ 0.3 % jog, joints did
  not move). `publish_joint_velocities: true` gives the JTC a real reference velocity (ff_velocity_scale 1).
- On `hardware_protocol:=mock_hardware` Servo appears frozen: Humble Servo sends a stamp-0 single-point
  trajectory every 10 ms and the mock JTC (100 Hz, **position** interface) re-seeds from the current state on
  every message. The real controller uses the velocity interface with feed-forward, so the real arm moves.
  For simulation, run the mock controller at a higher `update_rate` (scratch copy of controller_mock.yaml).

## Known limits (v1)
- No contact/force sensing: if the static-camera depth is wrong the arm just pushes. Candidate v2 proxy:
  HOLD on joint tracking error from `/rebel_arm_trajectory_controller/state`.
- The sign/scale checks verify the response of *the tracked line*, not that it is the tape: step 3's
  visual gate is the only guard against tracking a parallel crease/silhouette.
- Classical detector is untested on real video; seeded search near the expected edge is the default.
- Depth is not vision-controlled: it comes from the MoveIt path (static camera box model); the
  controller HOLDs if the tool drifts > `control.max_depth_dev_mm` from the start depth.
- No yaw correction: edge tilted > `target.max_edge_angle_deg` vs the cut → refuse/HOLD.
