# igus ReBeL ROS2 #

## Summary ##

* Connects to an igus ReBeL over an Ethernet connection

## Compatibility ##

Tested combinations:

- ROS 2 Jazzy on Ubuntu 24.04
- ROS 2 Humble on Ubuntu 22.04

## Installation: Not using docker ##

The packages are built from source in a colcon workspace. Run the commands for
your installed ROS 2 distribution from the workspace root.

### ROS 2 Humble on Ubuntu 22.04

```bash
source /opt/ros/humble/setup.bash
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install
source install/setup.bash
```

### ROS 2 Jazzy on Ubuntu 24.04

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install
source install/setup.bash
```

The `rosdep install` step installs MoveIt Task Constructor and the other
distribution-specific dependencies required by `igus_rebel_moveit_config`.

The ros node expects to reach the robot at the IP and port `192.168.50.133:3920`, this IP is fixed in **src/igus_rebel/include/igus_rebel/Rebel.hpp**, line 119 (`const std::string ip`). You can change the IP here, make sure to build the code by `colcon build` after changes.

## Installation: Using docker ##

Docker is optional; the supported Ubuntu and ROS 2 combinations above can run natively.

To build docker image:

```bash
sudo docker compose build                       # Humble (default)
sudo ROS_DISTRO=jazzy docker compose build      # Jazzy
```

## Docker usage ##

Run the container:

```bash
sudo docker compose up
```

To entry docker container environment, use the following command on each new terminal:

```bash
sudo docker exec -it ros2_rebel_dev bash
```

You can freely modify the code inside src folder in host computer and build by `colcon build` inside docker container.

## Usage: on real ReBel robot

Launch hardware interface, controller. `hardware_protocol` defaults to `rebel` (the
real robot) in `rebel.launch.py`, `move_group.launch.py` and
`igus_rebel_motion_planner.launch.py`; always pass it explicitly so a mock/sim session
never connects to the controller by accident. Accepted values are `rebel`,
`mock_hardware`, `gazebo` and (for `rebel.launch.py`) `isaac_topic`:
```bash
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel
```

What the real-robot driver (`Igus::Rebel`) does:

- On every (re)connect it sends `SetActive true`, `Reset`, `Enable`, `MotionTypeJoint`
  and then `Override 100.0`, so the arm moves at the commanded (planned) speed. The log
  shows `Connected to ReBeL at <ip>:3920`, `Sent CRI 'Override 100.0' ...`, and a
  `Rebel accepted command: ...` or `Rebel did not accept command: ...` line per command.
  The override reported in STATUS is logged as `Robot velocity override is X %` (first
  value and every change); a value other than 100 % is repeated as a warning.
- `on_init` waits up to 5 s for the first valid STATUS and fails otherwise; on success it
  logs `First STATUS: joints [deg] ... override ... %`. The initial state is seeded from
  that STATUS, never from an all-zero pose.
- Jog commands are clamped to +/-100 % and non-finite values are sent as 0. If no fresh
  `write()` arrives for 50 ms the ALIVEJOG thread sends zero jog.
- `read()` returns ERROR (and the jog is forced to zero) if no valid STATUS arrived for
  more than 200 ms, or after any reconnect. ros2_control then stops the component, so
  the driver must be restarted after checking the robot; motion never resumes on its own.

Launch the MoveIt motion planner. `igus_rebel_motion_planner.launch.py` forwards to
`move_group.launch.py` with these defaults: `use_gui:=false`,
`run_box_h_cut_executor:=true` (the plan-only H-cut executor starts), `run_teleop:=false`
(add `true` for MoveIt Servo), `run_gamepad:=false` (add `true` for `joy_node` plus the
gamepad teleop; it also needs `run_teleop:=true`), `run_camera_pole:=true`
(`false` skips the camera-pole collision object):

```bash
ros2 launch igus_rebel_moveit_config igus_rebel_motion_planner.launch.py \
  hardware_protocol:=rebel use_gui:=true
```

MoveIt joint limits (`config/joint_limits.yaml`, and the URDF `velocity` limits) use the
igus/CPR values for the ReBeL: 0.785 rad/s and 5.24 rad/s² per joint. Velocity and
acceleration scaling factors are fractions of these limits. The Pilz `cartesian_limits`
(`max_trans_vel` 0.01817 m/s, `max_trans_acc` 0.0524 m/s², `max_rot_vel` 0.0727 rad/s)
were rescaled together with the H-cut scalings so physical speeds are unchanged.
`trajectory_execution.allowed_start_tolerance` is 0.02 rad (not smaller than the
trajectory controller's 0.02 rad goal tolerance).

The trajectory controller (`config/controller.yaml`) uses P gains 4.0 with D gains 0 and
aborts a goal on a path error above 0.25 rad per joint, a final error above 0.02 rad, a
settle time above 3.0 s, or a residual velocity above 0.05 rad/s.

IK uses KDL with a 0.25 s timeout (`config/kinematics.yaml`). A commented TRAC-IK block
is provided; TRAC-IK has no Humble apt binary, so enabling it needs a source build
and a mock regression before physical use. Ruckig time parameterization was evaluated
and is not enabled: it did not change the H-cut plans, lengthened OMPL plans, and no
manufacturer jerk limit is available.

Known issue: stopping `move_group` with Ctrl-C on Humble can print a segmentation
fault during shutdown. This is an upstream MoveIt Humble shutdown bug, not a driver
fault.

## Usage: on simulation

To simulate the robot in Gazebo and control the simulated robot with MoveIt run:

```bash
ros2 launch igus_rebel_moveit_config igus_rebel_simulated.launch.py
```

The simulation uses the same XEG-32 wrapper model as MoveIt, so `xeg32_tool_tip` is
available. For a headless Gazebo server pass `gz_args:="-r -s empty.sdf"`.

## Calibrated TCP and visual comparison

The XEG wrapper defines `link6 -> igus_tool0 -> xeg32_tool_tip` independently
of gripper visuals. The tool frame is +52.7 mm along link6 Z and rotated pi
about Z; the TCP is `[5.937, 94.121, 130.699]` mm from that tool frame (touch-corrected
2026-09-24; see the comment in `igus_rebel_xeg32_wrapper.urdf.xacro`).
See `src/igus_rebel_moveit_config/XEG32_MOVEIT_CHANGES.md` for frame details.

The rough knife model remains removed. `xeg32_tool_tip` remains the exact
kinematic TCP and green 6 mm visual marker, but it has no collision geometry:
a collisionless marker itself cannot collide, and the attached cutter body
remains absent from collision checking. Do not treat planning success as
physical cutter-clearance validation.

For an isolated mock comparison, source ROS and this workspace in both terminals
and set `export ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1` in both (an unused domain,
localhost only, so nothing reaches the physical robot's domain). Start the robot/TF
publisher first:

```bash
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
```

Then start MoveIt/RViz in the second terminal:

```bash
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false \
  run_box_h_cut_executor:=false
```

The MoveIt launch alone does not publish the arm TF tree. Add an RViz TF display
to inspect `igus_tool0` and `xeg32_tool_tip`. Restart both launches after model changes.

## Collision geometry without a gripper-mounted camera

The camera and its pole are fixed outside the robot; no camera or holder is
mounted on the XEG-32 gripper. The gripper URDF therefore contains no camera or
holder geometry. The XEG-32 base and carriage meshes remain modeled and
collision-checked, alongside robot self-collision and the snapshot-derived
`detected_carton`. `xeg32_tool_tip` remains the kinematic TCP and green visual
marker with no collision geometry of its own.

The fixed external camera pole remains a required planning-scene obstacle.
`move_group.launch.py` starts `camera_pole_collision.py`, which derives the
upright 0.15 x 0.15 x 1.0 m `camera_pole` box from the external
`camera_link` and `camera_color_optical_frame` transforms and re-applies it to
`move_group` every 5 s (`reapply_period_s`). H-cut planning waits
until that collision object is available. A saved perception snapshot may be
used without acquiring a new image, but the fixed camera TF must still be
available so the pole is collision-checked.

In RViz enable MotionPlanning scene geometry and confirm both `detected_carton`
and `camera_pole` before planning. Discard stored trajectories and replan after
either obstacle changes. These are MoveIt collision objects, not hardware stops
and not protection for vendor/manual jogging.

## Calibration verification (plan-only by default)

The calibration verification launch starts mock or hardware control, MoveIt,
ArUco verification, and the `calibration_verification` executor. It never moves
the arm from a detection alone: every step is an explicit service call, and
`execution_allowed` defaults to `false`.

Mock launch used for validation:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=92 ROS_LOCALHOST_ONLY=1
ros2 launch igus_rebel_moveit_config calibration_verification.launch.py \
  hardware_protocol:=mock_hardware \
  start_control:=true \
  use_gui:=true \
  execution_allowed:=false
```

The supplied motion configuration (`config/calibration_verification.yaml`) enables
measured obstacles (`preview_only: false`) with `preapproach_only: false` and
`retract_after_approach: false`. Each plan is a PTP transit to the pre-approach pose
100 mm above the reference (`target_standoff_m` 0.02 + `preapproach_distance_m` 0.08),
a Cartesian approach to 20 mm, and a Cartesian final descent to 10 mm planned clearance
(`final_standoff_m` 0.01), and ends there. There is no automatic retract/home move.
`execution_allowed` remains false by default. Scalings are 0.13758 (velocity) and
0.09542 (acceleration) of the joint limits, the same physical speed as the former 0.05.
The platform is an A4 sheet (0.210 x 0.297 x 0.001 m, centered on the marker, top at
the marker plane) and the 50 mm cube is placed at the A4 corner `[0.105, 0.1485, 0]` m;
both are marker-relative. The below-base 4 x 4 x 2 m keep-out box is in
`planning_frame`, with its top at Z=0.
Setting `preview_only: true` omits these fixture objects and blocks execution.
The supplied orientation is `[pi, 0, 0]` relative to the marker: TCP +Z points
toward the board. This is a preview alignment, not a physical calibration result.
The verification executor declares the shared launch parameter overrides so
MoveIt uses the same named OMPL planner settings as the default launch.
Capture publishes labeled `center` and `point_1` .. `point_7` markers on
`/calibration_verification/markers` (RViz **Verification targets**).
`reference_origin_xyz_m` selects a measurement origin relative to the detected
marker center, expressed in marker axes (currently `[0, 0, 0]`: the marker center).
The center target, captured-reference axes, and all point XY offsets use this
origin. The configured points (marker axes, metres) are `point_1` `[0, 0.060]` (TOP),
`point_2` `[0.045, 0.060]` (P1), `point_3` `[0.020, -0.085]` (P3), `point_4`
`[0, -0.060]` (BOTTOM), `point_5` `[-0.060, 0]` (LEFT), `point_6` `[0.060, 0]` (RIGHT)
and `point_7` `[-0.070, 0.035]` (P2). Detector pose and platform/cube geometry remain
marker-centered; `table_box_center_xyz_m` is in `planning_frame`, not the marker frame.
Do not include the origin shift again in point offsets measured from this origin.
`plan_cube` fails closed with the supplied file because `cube_reference_xyz_m` and
`cube_approach_direction_xyz` are not set; configure them from the real cube first.
Physical execution requires confirming the actual start state and inspected plan.
Stale poses/states and plan mismatches still fail closed. The TCP marker has no
collision geometry, so there is no tip collision coverage; the attached cutter
body is likewise absent from collision checking.

In each service terminal, source the same workspace and use the same
`ROS_DOMAIN_ID` as the launch.

Service sequence:

```bash
ros2 service call /calibration_verification/capture std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_center std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_1 std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_2 std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_3 std_srvs/srv/Trigger {}
# plan_point_4 .. plan_point_7 work the same way
ros2 service call /calibration_verification/plan_cube std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/clear_plan std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/stop std_srvs/srv/Trigger {}
```

To opt in to manual execution, launch with `execution_allowed:=true` first, then
capture, plan, and inspect the retained solution in RViz. Nothing moves until:

```bash
ros2 service call /calibration_verification/execute_plan std_srvs/srv/Trigger {}
```

`clear_plan` discards the retained plan without motion; it does not stop active
execution (use `stop`, with the physical emergency stop available). Execution
consumes the plan, including failed validation attempts. Replan before retrying.
Plans expire after 60 seconds, and restarting the launch loses the retained plan.
The launch merges the execution opt-in into the motion parameters before passing
them to the node, avoiding node-specific YAML overriding the explicit launch flag.

An earlier mock smoke validated `plan_center`, `plan_point_1` .. `plan_point_3`,
`plan_cube` and mock `execute_plan` with a synthetic motion config and board poses
that were kept only in `/tmp` and are no longer available. Such a fixture is not a
source of real board, cube, marker, or TCP calibration values. The launch with the
supplied config was re-checked in mock on 2026-09-27
(`scan_results/p0p1_regression_20260927/moveit_config/f_calibration/`).

## H-cut executor parameters and gates

`box_h_cut_executor` (`config/box_h_cut_executor.yaml`, `cut_pattern` default `t_cut`)
plans with `~/plan_h_cut` and dispatches only through `~/execute_planned_h_cut`
(`execution_enabled`, default `false`). Its parameters have three scopes (listed at the
top of the YAML):

- Read-only (startup only; `ros2 param set` is rejected, edit the YAML and restart):
  frames, pipelines, clearances, `left_lateral_standoff_m`, `external_snapshot_max_age_s`,
  `cartesian_step_m`, `planning_timeout_s`, all `*_scaling` values, and others.
- Planning (validated, then applied and any retained plan cleared): `cut_pattern`,
  `motion_profile`, side tilts, correction files, `debug_allow_stale_snapshot`, and
  others. They are rejected with `busy: ...` while a plan/diagnose/cut-debug/execute call
  runs or a trajectory executes, and while a cut-debug session is active.
- Runtime (never blocked): `execution_enabled`, `debug_motion_execution_enabled`,
  `left_roll_diagnostics_enabled`, `external_snapshot_file`.

Every numeric value must be finite. Snapshots older than `external_snapshot_max_age_s`
(240 s in the supplied YAML; allowed range `(0, 600]` s) are refused.
`debug_allow_stale_snapshot=true` allows planning from an older snapshot for RViz
inspection only: `execute_planned_h_cut` refuses any plan or cut-debug session loaded
with it (and anything while it is true), before every other gate. `clear_plan` answers
`busy` instead of waiting while planning or execution is active.

## Staged per-cut correction workflow (plan-only by default)

The H-cut executor (`box_h_cut_executor`) plans the canonical snapshot-relative
cuts and additionally offers a staged teach-and-correct session for one cut at a
time. `cut_debug_cut` selects `left` (`back_middle_left -> front_middle_left`),
`middle` (`seam_start -> seam_end`), or `right`
(`back_middle_right -> front_middle_right`). Nothing moves until the operator
explicitly calls `execute_planned_h_cut`, and every staged dispatch requires
**both** `execution_enabled` and `debug_motion_execution_enabled` (both default
`false`). No stage auto-executes; a stale retained plan never survives a
teleop/save step.

Service sequence for the selected cut (same `ROS_DOMAIN_ID` as the launch):

```bash
ros2 service call /box_h_cut_executor/start_cut_debug_session std_srvs/srv/Trigger {}
ros2 service call /box_h_cut_executor/execute_planned_h_cut std_srvs/srv/Trigger {}
# teleop-correct the TCP at the start point, then:
ros2 service call /box_h_cut_executor/save_cut_debug_point std_srvs/srv/Trigger {}
# teleoperate the TCP to the end point separately, then:
ros2 service call /box_h_cut_executor/save_cut_debug_point std_srvs/srv/Trigger {}
```

The first stage plans the configured start target. If a correction record is already
active, that target includes it so an additional teaching pass begins from the current
best pose. Saved nominal/observed deltas always use the uncorrected geometry baseline,
so the resulting record is standalone rather than an incremental residual. Each
save_cut_debug_point call captures the full observed TCP pose from live joint-state FK
and persists it immediately without requiring any endpoint or lift plan: the first save
records the teleop-corrected start (status `start_saved`) and the second records the
separately teleoperated endpoint (status `complete`). A save retains no plan (any
retained plan is cleared) and never dispatches motion. The operator teleoperates the
endpoint separately and calls the same save service; the final record is complete after
that endpoint save. Start and end observations are therefore persisted independently.
Only `complete` records are loadable as corrections; `start_saved` is an intermediate.
Session stages are `start_planned` -> `start_executed` (after a successful explicit
dispatch) -> `start_saved`; the endpoint save completes the record and ends the session.
The session snapshot must pass the normal freshness check when the session starts
(`external_snapshot_max_age_s`, 240 s in the supplied YAML); after that the session
freezes its snapshot. A session started with `debug_allow_stale_snapshot=true` can
never be executed.
Left approaches and retracts laterally toward the detected carton exterior by
`left_lateral_standoff_m` (25 mm in the supplied YAML; allowed range 5-25 mm),
without shifting either cutting endpoint. Middle and right retain surface-normal
hover/retract. Each save persists a schema-versioned record (`box_h_cut_correction/v1`,
`<cut>_<snapshot_id>.json`, atomic temp-file plus rename) with the snapshot ID,
cut name, planning frame, nominal/observed full start/end poses, cut-local
translation deltas, and cut-local orientation delta quaternions.
(`cut_debug_correction_enabled`, default `false`, plus per-cut absolute paths
`cut_debug_left/middle/right_correction_file`). Combined plans apply each set
slot to its segment; `t_cut` and `t_cut_handoff` contain only left and middle segments,
so a set right correction is loaded, validated, and left unused rather than failing
closed; individual plans apply only their cut's slot. Empty means no correction for that cut; any malformed or mismatched nonempty record, or enabling with all slots empty, fails closed. Records
are cut-local and rotate onto the newly detected cut. Orientation preserves the full
taught start/end TCP quaternion: the planner maps each saved quaternion from the
recorded cut basis into the current detected cut basis. Thus the taught blade angle is
unchanged at the capture pose and follows later carton rotation without being rebuilt
from a nominal knife frame. A taught left pose may validly use tool local X or
tool local Z as its penetration axis; at least one of the two must point into
the surface. Middle and right use tool local Z only. Records are validated for
schema, `complete` status, matching cut slot, a present `planning_frame` and
`tool_link` equal to the executor's active values (a record missing either field is
refused), finite orthonormal cut bases, finite translation values, and unit-length
orientation quaternions. Application fails
closed unless the mapped penetration axis (left: X or Z; middle/right: Z) points
into the carton surface. With opt-in off, plans remain purely geometry-derived
even when paths are set.
For learned LEFT poses only (with `cut_debug_correction_enabled` and a loaded
LEFT correction file), when `left_adaptive_orientation_enabled` is true,
the planner tests TCP-local Y (blade-length/seam axis) rolls of the mapped
quaternion as `q_candidate = q_mapped * Qy(theta)`. TCP position and blade Y
direction are preserved; only the X/Z penetration roll varies. Candidates run
in deterministic smallest-magnitude order 0, +5, -5, ... through
`+/-left_adaptive_orientation_max_deg` in `left_adaptive_orientation_step_deg`
steps (defaults 45.0/5.0). Every candidate must pass the existing left X-or-Z
penetration gate plus IK and collision feasibility at the final target poses;
the probe is a bounded prefilter that only screens out unreachable or colliding
mapped poses under the current carton/pole scene constraints and does not
guarantee a full plan. The search runs once per planning request (the automatic
`t_cut_handoff` search reuses its result for every probe). Each candidate logs a
one-line `Left roll diag` result; `left_roll_diagnostics_enabled:=true` (runtime,
default `false`) adds expensive per-pose IK/collision detail without changing the
selection. Hover/approach candidates must be fully collision-free;
cut-target candidates honor the scoped tool/carton contact allowance while
every other contact still fails. The search checks the same detected carton and
external camera pole as the MTC task, including robot self-collision; collision-blind
IK can otherwise pick a roll whose approach goal is invalid. MTC, Cartesian, and
collision checking remain the final authority and may still reject a selected
orientation; both execution gates stay mandatory. If no candidate passes,
planning fails closed with candidate diagnostics. Middle and right corrections
keep their rigid mapped orientation. The saved September 2026 left record predates
the current TCP: its translations put the blade about 30 mm above the carton on the
current TCP, so only an orientation-only derivative
(`left_capture_frame_000001_20260916_235239_389784_orient_only_tcp20260924.json`,
zero translation deltas) is used in mock. A mock replay may opt into it, but a
physical cut requires operator verification of the blade pose on the current tool
before enabling this opt-in. On the replay carton the taught angle itself is
unreachable (joint3 limit) and the search selects +45 degrees.

## H-cut free-space transit consistency (plan-only by default)

Free-space transits (initial approach and inter-segment finish-to-approach
moves) use dedicated `transit_velocity_scaling` /
`transit_acceleration_scaling` (0.4127 / 0.2863 of the 0.785 rad/s / 5.24 rad/s²
joint limits = 0.324 rad/s / 1.50 rad/s², the physical speed of the
operator-approved 0.15 transit scaling of 2026-09-26 at the former limits).
Contact descend/traverse/retract strokes use `velocity_scaling` /
`acceleration_scaling` 0.1376 / 0.0954 (0.108 rad/s / 0.50 rad/s², the physical speed
of the former 0.05), and `pre_cut_stabilization_duration_s` stays 1.0 s, so cutting
speed is unchanged. All four scalings are finite values in `(0, 1]` and read-only:
edit `box_h_cut_executor.yaml` and restart the executor to change them.
Each free-space transit first resolves a pose goal to the exact-pose, collision-free
IK branch nearest the propagated start state (`NearestBranchTransitPlanner`: seeds
from the start and its equivalent spherical-wrist branches, prefers branches with
joint6 change <= pi, then the smallest joint distance). It then tries a direct
collision-checked joint interpolation and, if that fails, the profile's pipeline
planner (MultiPlanner order [joint interpolation, existing transit]). If every IK
branch fails, the original pose goal is retried once. All attempts of one transit
share a single `planning_timeout_s` budget (15 s). Full cut
poses, `min_fraction=1.0` acceptance, the scoped tool/carton contact policy,
and the fixed external `camera_pole` requirement are unchanged. A free-space
solution whose joint6 travel exceeds pi rad, summed over every consecutive
waypoint of its free-space transit subtrajectories, is rejected before the
plan is retained, published, or executed; there is no silent fallback after
this check. The guard also fails closed if the number of examined transit
trajectories does not match the transits recorded while building the task. Before
the nearest-branch planner, the pre-TCP-change taught left correction on the saved
carton was rejected at 4.806 rad of middle-transition joint6 travel; with it, the same
corrected `t_cut` planned 5/5 in isolated mock (left-to-middle joint6 1.66 rad,
`scan_results/hcut_wrist_branch_20260926/`). This is mock planning evidence only.
Faster transit shortens only non-contact portions and is no
guarantee of a fast plan or a physically safe stale TCP: the correction
opt-in (`cut_debug_correction_enabled`, default `false`) and both execution
gates (`execution_enabled`, `debug_motion_execution_enabled`, both default
`false`) are unchanged, and no physical dispatch is recommended.

## T-cut handoff (first arm only, plan-only by default)

`cut_pattern: t_cut_handoff` plans first-arm-only work: the full left cut plus
an executable bounded middle-seam prefix ending at the declared handoff endpoint
(`interpolate(seam_start, seam_end, <resolved fraction>)`, included in
the published target poses for external second-arm integration). It does
NOT lower any `min_fraction` acceptance: the planned path is accepted only when
it fully reaches that explicit handoff endpoint under `min_fraction=1.0`. After
fully reaching that endpoint, the retained first-arm plan ends at the cut-level
handoff pose and deliberately skips the hover retract
(`finish pose above middle seam handoff`) and its following restore-collision
stage, so the scoped tool/carton contact allowance stays active at the terminal
pose. The retained middle traverse is therefore an executable first-arm prefix,
not the complete seam; no second arm is implemented. Execution still requires the
normal gates.
`t_cut_middle_handoff_fraction` selects the stop fraction along the middle seam:
`0.0` (default) selects automatic mode: the maximum 0.95 is probed first and used if
it fully plans; otherwise 0.05 must plan (else the search fails closed) and a bounded
bisection over `[0.05, 0.95)` (at most 7 probes) finds the largest fully reachable
declared prefix with `<=0.01` fraction tolerance, keeping `min_fraction=1.0` for every
candidate. The left-cut roll search runs once per request and is reused by every
probe. A nonzero value in `[0.05, 0.95]`
is a manual override stop fraction; out-of-range nonzero values fail closed.
Full `t_cut` stays the full left cut plus the full middle seam and is unaffected,
and full `t_cut`/`combined` continue to retract above the cut and restore
collision checking. The left-to-middle transition (combined, `t_cut`,
`t_cut_handoff`) is: left lateral withdrawal to the finish pose outside the left
cut, then a surface-normal lift that holds the left orientation, then restore of
knife-tip collision checking, then free-space transit/rotation to the middle
approach. No current-plan serialization is implemented: the retained plan lives
only in the running executor until restart. No second-arm behavior changes.
`t_cut_handoff` loads the same left and middle correction slots as `t_cut`; a set right record loads but stays unused because the plan has no right segment.
Middle translation end deltas are fraction-interpolated and the taught middle end
orientation is slerp-interpolated by the resolved fraction for the
bounded prefix; the middle tool-Z penetration gate is revalidated unchanged.

## RobotViewer

At `https://viewer.robotsfan.com/`, choose **Load Files** and select:

- `src/igus_rebel_description/urdf/igus_rebel2.urdf` (its mesh URIs are `package://`)
- all twelve `.dae` files in
  `src/igus_rebel_description/meshes/rebel_d00617809/`

The browser cannot read mesh files from the local ROS installation when only
the URDF is selected, so the URDF and its DAE files must be loaded together.

## Isaac Sim

Generate the corrected six-axis URDF before importing it into Isaac Sim:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
xacro src/igus_rebel_description/urdf/igus_rebel_isaac_wrapper.urdf.xacro \
  -o /tmp/igus_rebel_official.urdf
```

Import `/tmp/igus_rebel_official.urdf` with a fixed base, articulation creation,
inertias, and collision geometry enabled. Keep fixed joints separate on the
first import so the ROS link hierarchy remains visible. The twelve visual meshes
are tessellated from the igus `D00617809` STEP assembly and preserve its native
CAD material regions. Collisions, inertias, and joint frames remain those of
the version-two robot.
No pre-generated USD files are kept in this repository; generate the USD in Isaac Sim
from this URDF, and regenerate it after model changes.

## Set digital outputs

The ReBeL's digital outputs can be set with a call to the service `/set_digital_output`. 

The service input is a `DigitalOutput` message, which is defined as
```
int8 output
bool is_on
```

- `output` is the index of the output whose state should be set.
- `is_on` is the state to which the output should be set. `True` means on, `False` means off.

The service output is defined as
```
bool success
string message
```
`success` is True when the `DOUT` command was sent to the controller (it does not wait
for the controller's acknowledgement; check the driver log for `Rebel accepted command`).
It is False with `message` `not connected to the robot` when the command could not be
sent. The service is provided by the real-robot driver (`hardware_protocol:=rebel`) only.

## Teleoperation

You can use a teleop keyboard program or a gamepad to control the arm manually. Both
need MoveIt Servo, which the MoveIt launch starts only with `run_teleop:=true` (default
`false`). The gamepad (`joy_node` plus `servo_gamepad_input`, mapping in
`config/gamepad.yaml`, enable button 7 must be held) starts with the MoveIt launch when
`run_gamepad:=true` is also passed (default `false`); it publishes Cartesian twist only.
On Humble, Servo stays idle until `/servo_node/start_servo` is called; the keyboard and
gamepad nodes call it themselves when they start.

You can also run the following command to start teleop_keyboard:

```bash
ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard
```

Keys: arrow keys move the TCP along X/Y and `.` / `;` along -Z / +Z of the selected
command frame (`w`: `base_link`, `e`: `xeg32_tool_tip`); `s` sends a zero linear twist;
`i`/`k`, `n`/`m`, `u`/`o` rotate about X, Y, Z; `-` / `=` decrease / increase the
Cartesian linear and angular speed together; `1`-`6` jog a joint and `r` reverses the
joint direction; `q` or Ctrl-C quits. `t` / `j` select twist / joint-jog mode on Jazzy;
Humble Servo accepts both command types at any time, so the keys have no effect there.

The Cartesian keyboard increment defaults to 5 mm per keypress. Change it at runtime with:

```bash
ros2 service call /servo_keyboard_input/set_cartesian_increment \
  igus_rebel_msgs/srv/SetCartesianIncrement "{increment_meters: 0.002}"
```

The accepted range is greater than 0 m through 0.03 m. The service affects Cartesian
arrow, period, and semicolon commands; joint-jog speed is unchanged.


## Project and OMP session RAG

The dependency-free local index combines current source/configuration files,
curated verified facts and unresolved items, and textual user/assistant turns
from OMP sessions whose recorded working directory matches this repository.
Build it atomically with:

```bash
python3 tools/session_rag/session_rag.py build
```

Retrieve evidence before debugging or editing:

```bash
python3 tools/session_rag/session_rag.py query \
  "Why does the CRI DOUT parser fail on a 64-bit mask?" --format prompt
```

Inspect session-retention recommendations or graph relationships:

```bash
python3 tools/session_rag/session_rag.py sessions
python3 tools/session_rag/session_rag.py graph topic:cri_network
```

The SQLite FTS5 index and JSON graph map are generated under
`workspace_scan_artifacts/`. Generated evidence includes its source and line or
message locator; session evidence can be stale, so verify current source before
changing code.

An optional LightRAG REST adapter uses `POST /documents/texts` and `POST /query`.
It accepts loopback endpoints by default and reads `LIGHTRAG_URL` and
`LIGHTRAG_API_KEY` from the environment; it never stores credentials in the
index. Validate an ingestion batch without network access using:

```bash
python3 tools/session_rag/session_rag.py lightrag-ingest --dry-run
```

LightRAG still requires a separately configured LLM and embedding provider.
Use `tools/session_rag/lightrag.env.example` as the client/server variable
template. Do not expose an unauthenticated LightRAG server beyond loopback.