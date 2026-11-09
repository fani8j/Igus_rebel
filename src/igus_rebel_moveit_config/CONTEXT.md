# Ubiquitous Language

## Snapshot-relative combined plan
A fixed canonical plan derived from one selected carton snapshot: left midpoint cut, collision-aware transition, complete middle seam, collision-aware transition, then right midpoint cut. Planning publishes to RViz and retains a plan only when every stage succeeds.

## T-cut
The default `cut_pattern` (`t_cut`): the full left side cut, the left-to-middle
transition (lateral withdrawal, surface-normal lift, collision restore, free-space
transit), then the complete middle seam. `t_cut_handoff` is the same first-arm plan
but stops the middle seam at a declared handoff fraction for a second arm (not
implemented). `h`/`combined` add the right side cut.

## Canonical side cut
A side stroke through the detected midpoint pair, back first (`back_middle_left -> front_middle_left` or `back_middle_right -> front_middle_right`). The knife faces 180 degrees opposite travel while its local Z remains directed into the carton surface. No camera or holder is mounted on the XEG-32 gripper. The XEG-32 base/carriage meshes and detected carton remain collision-checked, and the fixed external camera pole remains a required `camera_pole` planning-scene obstacle derived from the external camera TF. `xeg32_tool_tip` is a collisionless kinematic/visual marker.

## Feasible combined plan
A collision-free trajectory through all three canonical cuts and both transition stages. No partial or candidate-search result counts as a feasible combined plan.

## Staged cut-debug session
An explicit per-cut teach-and-correct flow owned by one retained plan at a time.
`cut_debug_cut` selects `left`, `middle`, or `right`. `start_cut_debug_session`
plans the configured start target and never dispatches motion. Existing corrections
seed repeat teaching, while persisted deltas always use the uncorrected geometry
baseline so every record is standalone. Each `execute_planned_h_cut` dispatch
inside a session requires both `execution_enabled` and
`debug_motion_execution_enabled`. Only the start plan is ever retained and
executed. Stages: `start_planned` -> `start_executed` (after a successful explicit
dispatch) -> `start_saved`. `save_cut_debug_point` captures the full observed TCP pose
from fresh joint-state FK and persists the nominal/observed/delta record: the first
save records the teleop-corrected start (`start_saved`); the operator then teleoperates
to the endpoint and the second save records it, writes the `complete` record and ends
the session. No save plans or retains an endpoint or lift plan, and no save dispatches
motion. The session snapshot must be fresh when the session starts and is then frozen;
a session started under `debug_allow_stale_snapshot` can never be executed.
Any out-of-order call is rejected; no stale plan survives a teleop/save step.

## Cut-local correction
A translation expressed in the detected cut's own basis (`along` the cut,
`across` the cut tangent, `normal` the surface normal) plus the orientation
delta in that basis. Applying a record to a new snapshot rotates the same
local components onto the newly detected basis, so the correction follows the
cut. Application is opt-in (`cut_debug_correction_enabled`, default `false`)
and fails closed on a schema, status or cut-slot mismatch, a missing or mismatched
`planning_frame`/`tool_link`, a non-orthonormal recorded basis, non-finite values, or a
mapped penetration axis that no longer points into the carton.

## Side-tape hand-off (left face only)
Opt-in (`side_tape_handoff_enabled`, default off). `~/plan_side_tape_handoff` plans the full
left cut for feasibility, then retains and executes only the approach to the cut start pose:
`in_material` stops at the planned in-material start; `standoff`'s single Cartesian entry
targets the standoff pose (tip `side_tape_handoff_standoff_mm` in front of the outer wall)
directly and NEVER targets, or passes through, the in-material pose (verified over every
trajectory waypoint via `minSignedWallDistanceM`, not just the endpoints). The start pose is
checked against the configured `side_tape_handoff_cut_axis_tool`/`lateral_axis_tool`/
`depth_axis_tool` (right-handed, `depth = cut x lateral`, validated at startup) within 5
degrees; on a genuine mismatch this fails closed and reports the measured axis-to-axis mapping
rather than rotating anything.

Follow-up (2026-09-29): the convention is the left_parallel tool frame (cut = tool -Y, lateral =
tool +X, depth = tool +Z into the carton side). The hand-off REQUIRES
`left_parallel_orientation_enabled: true` (fails closed otherwise) and takes its start pose from
that search, run once on the full four-pose stroke and reused for the truncated approach (the
start-only search could otherwise pick an orientation the stroke cannot carry). In hand-off mode the
search only considers candidates with |tilt|,|pitch|,|yaw| and total rotation from F0 within
`side_tape_handoff_max_search_angle_deg` (default 10) and orders them smallest rotation from F0
first (the normal left cut orders by theta_p). The chosen `tilt/pitch/yaw/theta_p`,
`candidate_index`, `rotation_from_f0_deg` and `max_abs_angle_deg` are in the record under
`left_parallel_orientation`. The axis-convention tolerance is max(5 deg, that limit). The standoff
wall gate scans the whole plan, so a transit that swings the wrist near the wall from a distant
start state is refused; start standoff plans from an arm already at the left approach pose.

`~/continue_side_cut` (std_srvs/Trigger): after side_tape aligned the tip into the seam (standoff
variant), plans ONE MTC task from the current state: a straight Cartesian stroke (min_fraction=1.0,
scoped contact allowance) parallel to the recorded planned cut line (`planned_cut_start_m` ->
`planned_cut_end_m`), keeping the current tool orientation and the alignment's offset and ending on
the plane through the planned end perpendicular to the line, then the same withdraw + transit as
`side_tape_retract` (withdraw from the stroke end). Refused before anything moves unless status is
`active`, snapshot fresh, `execution_enabled`, not stale-override, Servo quiescent, and (read-only
`side_tape_continue_*`, defaults 20 mm / 20 mm / 0.5-20 mm / 20 deg) perpendicular offset, along-line
offset, depth behind the outer wall and orientation deviation from the hand-off start are within
limits; the retract transit must pass the wrist gate. Statuses: `cutting` while executing,
`retracted` on success, `cut_failed` on failure (state and carton kept; `side_tape_retract` recovers,
subject to its own unchanged gates).

`SideTapeHandoffStatus` is `kNone` (no hand-off), `kActive` (approach dispatched
successfully; side_tape is expected to be driving the cut), `kApproachFailed` (the approach
dispatch did NOT complete; arm pose uncertain but the carton and geometry are kept so
`~/side_tape_retract` can still recover), or `kRetracted` (terminal, reported once then reset
to `kNone`). `detected_carton` and the hand-off record survive every clear/replan attempt
while `kActive` or `kApproachFailed` (`inProgress()`); a `kApproachFailed` hand-off can also be
cleared by `clear_plan` once `sideTapeHandoffTipVerifiedOutside` confirms the tip is in front
of the outer wall. `side_tape_handoff_pending_` (a retained-but-undispatched plan) is always
reset by `clearRetainedPlan()`, so it can never leak into an unrelated later plan's dispatch.

`~/side_tape_retract` (std_srvs/Trigger) plans from the CURRENT robot state and executes the
retract, gated by: not planned/loaded under `debug_allow_stale_snapshot` (never executes
anything, including the withdraw); MoveIt Servo quiescent (no message on
`side_tape_handoff_servo_twist_topic`/`_joint_topic` within `_servo_quiescence_s`, no joint
`|velocity|` above `_joint_velocity_threshold`; Servo itself is never paused/stopped);
current tool orientation within `_retract_max_orientation_deg` of the recorded start; tip
within `_retract_max_line_offset_mm` of the recorded cut line. It withdraws along the outer-wall
normal by (current penetration behind the wall, clamped >=0) + `hover_clearance_m`, then --
only if the snapshot is still fresh -- transits back to the usual left approach pose; a stale
snapshot withdraws only and reports "withdrawn; transit skipped (stale snapshot)".

The hand-off record (`status`, face, `cut_length_mm` projected onto the cut axis plus
`cut_length_euclidean_mm`, start pose, variant, penetration/standoff/`min_wall_distance_mm`,
the configured tool axes and their measured `base_link` directions, snapshot path/id) is
written atomically to `side_tape_handoff_output_file` and published on the latched
`~/side_tape_handoff` topic on every status change. The seed and snapshot always come from one
read of the snapshot file (`loadSnapshotAndSeedFace`); the seed's own `source.snapshot_id`/
`source.acquisition_stamp_ns` are cross-checked against the parsed snapshot, failing closed on
a mismatch.

## Blade parallelism (left side cut)
Tool Y is the blade length, tool X the blade-plane normal and tool Z the edge. `theta_p =
acos(|tool X . surface_normal|)` is the angle between the blade plane and the detected
carton top face: 0 is a blade lying flat on the top plane, 90 the perpendicular (nominal)
knife. The left cut line runs along the top-left edge (back_middle_left ->
front_middle_left), so a flat blade meets the tape where it wraps over that edge, edge-first
from outside the carton (the lateral entry). `left_parallel_orientation_enabled` searches
for the smallest reachable `theta_p` (opt-in, plan-only).

## Varying-orientation left cut
Opt-in (`left_parallel_varying_orientation_enabled`, default off). Instead of one fixed left_parallel
orientation, the tool orientation may change between stations along the left stroke
(`left_varying_*`); the longest contiguous feasible station interval is cut, trimmed at either end if
needed, and planning fails closed below `left_varying_min_cut_fraction`. Details in
`XEG32_MOVEIT_CHANGES.md`.

### True blade-angle cap (varying left cut)
`left_varying_max_theta_p_deg` (planning, default 15) is a cap on the TRUE theta_p (see Blade parallelism) from the stroke start
through the seam band; `left_varying_end_max_theta_p_deg` (planning, default 15 = no ramp) lets the cap rise linearly after the
band to the end of the extended span (travel order; needs `left_cut_extend_to_corners`);
`left_varying_start_max_theta_p_deg` (planning, default 0 = off, else [band cap, 45]) allows a STEEPER blade at the stroke start (the arm
needs it to reach the back corner) that is guaranteed back at the band cap before the tape band: the start cap holds until
`left_varying_start_ramp_m` (planning, default 0.05 m, [0, 0.2]) before the band, ramps linearly down and equals the band cap exactly
at the band entry (travel order; a ramp that would start before the stroke start is clipped; needs `left_cut_extend_to_corners`);
`left_varying_theta_p_tolerance_deg` (read-only, default 1) is the slack for the interpolation between stations. The cap filters the
per-station candidates, and every waypoint of the final retained stroke is checked against it. The candidate grid still comes
from `left_parallel_max_tilt_deg`, so raise that too when the end cap exceeds what the grid produces. Details in
`XEG32_MOVEIT_CHANGES.md`.

### Reverse travel of the varying corner-to-corner left cut
`left_cut_direction` (planning, `forward` default | `reverse`) and `left_reverse_frame` (planning, `about_normal` default | `about_edge`,
only read with reverse). Explicit only (no auto selection); varying + `left_cut_extend_to_corners` + `cut_pattern` `left_only`/`t_cut`
only; refused with corrections, cut-debug, side-tape hand-off, `continue_side_cut`, `side_cut_penetrate`, `plan_t_cut`, `t_cut_handoff`,
`h`/`combined` (not verified), the FIXED-mode `left_cut_reverse_travel` and `left_debug_stop_at_approach`. Reverse travels
`front_middle_left -> back_middle_left`: the stations are searched in TRAVEL order (station 0 = front end), the extension/seam band are mapped to
travel coordinates (`left_cut_direction.hpp`), the candidate frames are built in a TRAVEL basis (`along` = travel direction; the canonical back->front
basis of corrections and the hand-off is unchanged) and tool Y = -travel (the blade still leads with its sharp edge). The blade-angle profile is in
travel coordinates, so the band cap applies from the front (start) through the tape band and the graded end cap ramps toward the BACK corner.
`left_varying_start_extra_inward_m` tapers from the kept-interval START, i.e. from the front end in reverse. Plan messages and trims report the
PHYSICAL back/front. Frame variants (X blade-plate normal, Y blade length, Z knife axis towards the blade point): `about_normal` = forward frame
rotated 180 deg about the surface normal (X = -normal kept, Y reverses, Z flips to the other side of the cut line); `about_edge` = rotated 180 deg
about tool Z (Z keeps its side, X = +normal: blade plate upside down; the penetration gate reads the sign-mapped X axis).

### Left reverse patch (back-end gap of the forward corner-to-corner left cut)
`left_cut_patch_enabled` (planning, default `false` = plans bit-identical), `left_cut_patch_min_gap_m` (default 0.015, [0, 0.10]) and
`left_cut_patch_overlap_m` (default 0.015, [0.005, 0.05]). On the real cell the forward (back -> front) varying cut lost 10-117 mm at the
BACK end in 36 of 39 runs (blade-angle cap at the back corner), never at the front. When the forward kept interval leaves a back gap
LARGER than `left_cut_patch_min_gap_m`, the SAME plan (one MTC task, one retained plan) gets a second, short REVERSE stroke (front -> back, frame per
`left_reverse_frame`, default `about_normal`) that contains the point forward-kept-start + overlap and runs as far toward the back corner (minus
the corner margin) as reachable. It does not have to cover the tape band (the forward stroke crosses it) but it uses the same station spacing, blade
caps (the reverse profile of the full span, so the cap at a physical position is the reverse-travel cap), joint margin, split travel gate, step,
collision and continuity rules as the main search (`searchLeftVaryingOrientation` with `LeftExtension::patch`). Stage sequence of the face job:
approach / dwell / entry / `cut left side cut` / `finish pose outside left side cut` (forward, as today), `restore knife-tip collision checking after
left side cut`, then `approach pose outside left side cut patch` (a free-space TRANSIT to the verified patch hover joints, registered for
`validateFreeSpaceTransitWrist`), `stabilize blade before left side cut patch`, `allow knife-tip contact ... patch`, `enter laterally into left side
cut patch start`, `cut left side cut patch [reverse]` (one Cartesian stroke, `min_fraction` 1.0, contact scalings), `finish pose outside left side cut
patch`, then the lift/restore/raise that used to follow the forward stroke. The scoped tool/carton allowance covers only the entry/cut/finish of each
stroke. A patch that cannot be planned (cannot reach the overlap point, IK/collision/blade-cap failure, wrist gate or any patch-related validator) is
DROPPED: the forward-only plan is kept and the plan message says why; the patch never refuses a plan. Refused (planning error) when the forward
varying corner-to-corner mode is not active: varying or `left_cut_extend_to_corners` off, `left_cut_direction` reverse, `left_cut_reverse_travel`,
corrections, cut-debug, side-tape hand-off, `t_cut_handoff`/`h`/`combined`, `left_debug_stop_at_approach`, `left_varying_start_extra_inward_m` > 0; the
staged/hand-off entry points refuse while it is on. Plan message field: `left reverse patch: forward kept ..., back gap ..., patch kept ..., overlap ...,
total covered ... mm = ... % of the ... mm corner-to-corner edge ..., patch status: not needed | planned | dropped: <reason>`. Pure maths + gtests:
`left_cut_patch.hpp`, `test/test_left_cut_patch.cpp`.

### Left contact joint-travel gate: aggregate | split
`left_contact_travel_mode` (planning, `aggregate` default | `split`) plus the split parameters `left_contact_arm_travel_max_rad` (2.0, [1, 4]),
`left_contact_min_abs_joint5_deg` (15, [5, 30]), `left_contact_min_elbow_margin_deg` (5, [1, 30]) and `left_contact_min_shoulder_margin_m`
(0.05, [0.01, 0.25]); none in the YAML. `split` is refused by `validateSettings` unless `left_parallel_orientation_enabled` is true (the flag that
runs `validateLeftContactJointTravel`). One pure function (`left_contact_travel.hpp`) is called by the varying search replay and by the validator on
every left contact stroke (entry, cut, finish, lift). `aggregate` (default) = today's rule, every joint's aggregate travel <= 1.0 rad (sized for 25-60 mm
strokes; it trims the long positive-yaw corner-to-corner cut 30-107 mm at the end). `split` checks, in joint1..joint6 order (the executor maps the group
variables by name and fails closed if a jointN is missing; split also needs the analytic-IK self-check against the live model to have passed):
joint4, joint5 and joint6 aggregate travel <= 1.0 rad; joints 1-3 aggregate travel <= the arm cap (a coarse bound, not a flip detector); joint5 keeps its sign;
the Pieper branch label (shoulder x elbow x wrist, `rebel_analytic_ik::classifyBranch`) does not change between the SAMPLED waypoints; and every waypoint keeps a
margin to the three branch boundaries (|joint5|, |joint3 - beta| = distance to the straight arm, |X| = wrist-centre distance from the joint1 axis). Between waypoints
the guarantee is: joint3 and joint5 are interpolated linearly, so equal signs and margins at both waypoints hold for the whole segment; the shoulder coordinate X is nonlinear,
so it is proven per segment with a Lipschitz bound and subdivision (an unprovable segment is refused). A flip that is not a straight joint-space segment between two waypoints is
not inspected. Boundary values (j5 = 0, j3 = beta, X = 0) take the + label and have margin 0, i.e. are refused by their margin rule. In split mode the validator also fails closed
when the plan contains a left side cut but no (or no cut) stroke was examined. Refusals name the rule (`wrist_travel`, `arm_travel`, `joint5_sign_change`, `joint5_margin`,
`elbow_margin`, `shoulder_margin`, `branch_change`), the joint and the value. Search attribution: travel rules trim the end that moves the joint more (as before),
the other rules trim the end the first failing waypoint lies on. The plan message always reports the cut stroke's travel per joint, min |joint5|, branch and margins; the validator logs the same for every stroke.

## Extended left cut (corner-to-corner)
Opt-in (`left_cut_extend_to_corners`, default off; requires the varying-orientation mode). The varying
search runs on the nominal `back_middle_left -> front_middle_left` line extended both ways to the corners
(`back_left + left_cut_corner_margin_m` .. `front_left - left_cut_corner_margin_m`, projected onto that line)
and keeps the longest reachable interval that CONTAINS the seam crossing +/- `left_cut_seam_cover_m`. The
minimum length (`left_varying_min_cut_fraction`) is measured against the NOMINAL segment. Details in
`XEG32_MOVEIT_CHANGES.md`.

## Approach-only left debug plan
`left_debug_stop_at_approach` (default off, `cut_pattern left_only` only): `plan_h_cut` plans and validates
the full left cut, publishes its markers, but retains only the approach to the left hover pose (after the
pre-cut dwell). Executing it needs both `execution_enabled` and `debug_motion_execution_enabled`.

## Snapshot frame mode and relabel (dual-arm planning stack)
Read-only parameters, defaults = today's behaviour. Perception writes the snapshot in the LEFT base frame
(`frame_id` `base_link`). An executor that plans for another arm sets `snapshot_frame_mode fixed_transform`:
the 11 snapshot points are mapped `p_plan = B^-1 p_snapshot` and the normal is rotated, with `B` =
`snapshot_planning_base_xyz/_rpy` (URDF fixed-axis rpy, R = Rz Ry Rx; the right arm copies
`dual_arm_calibration.yaml` `arm_right_base` verbatim), after checking `frame_id == snapshot_frame_id`; the result
is stamped `planning_frame` and then goes through `validateSnapshot` unchanged. `snapshot_relabel rotate_180`
renames the corners by a 180 degree rotation about the carton normal (back_left<->front_right,
back_right<->front_left, seam_start<->seam_end, back_middle_left<->front_middle_right,
front_middle_left<->back_middle_right; seam centre and normal kept), so the carton's RIGHT face becomes the
executor's "left" cut with exactly the left arm's geometry. The label handedness is checked before and after
(a mirror relabel flips it and is refused). Both are applied once, at the end of `parseSnapshotFromDocument`
(pure maths in `snapshot_frame.hpp`). A transformed or relabelled snapshot needs `plan_export_only=true`, forbids
`cut_debug_correction_enabled` and `side_tape_handoff_enabled`, refuses the side-tape seed block and pattern `h`
with `approach_joint_degrees`; non-zero `snapshot_planning_base_*` without `fixed_transform` is refused.

## Carton contact links (blade collision)
Read-only string-list parameter `carton_contact_links` (default `[]`, meaning `[tool_link]` = today's behaviour). The
scoped tool/carton contact allowance (only while `allow_tool_carton_contact` is true) is installed for EVERY listed link
in the same places as before: the `allow knife-tip contact` / `restore knife-tip collision checking` stages around the
cut strokes (`addCartonContactScope`), the carton scene of a cut-debug start and of the side-tape stages, and the
analytic-search scenes' allowed-collision matrices (`setCartonContactEntries`). Free-space transits, approach and hover
stay fully checked for the listed links against `detected_carton`. `PlannedSegment.tool_carton_contact_allowed` is true
only if every listed link is allowed at the start of the stage. Used by the dual per-arm planners: their model carries
the opt-in collision box `knife_c_blade` (URDF arg `blade_collision:=true`, see `XEG32_MOVEIT_CHANGES.md`), and the launch
sets `carton_contact_links [xeg32_tool_tip, knife_c_blade]`. `tool_link` is not added automatically to a non-empty list.
Startup refuses (exit code 2, FATAL log) when a listed link is not in the robot model; `validateSettings` refuses empty,
duplicate or `carton_collision_id` entries. The scoped allowance is also the only place the blade may touch the carton:
`knife_c_handpiece` / `knife_c_holder` stay checked during the plunge (see the evidence in `XEG32_MOVEIT_CHANGES.md`).

## Plan export-only mode
`plan_export_only` (read-only, default false). Only `~/plan_h_cut`, `~/plan_h_cut_from_state`,
`~/get_planned_solution`, `~/clear_plan` and `~/diagnose_external_snapshot_sections` are advertised: no
`execute_planned_h_cut`, side-tape, T-cut, staged top-cut or cut-debug service exists. Required by every
non-default snapshot frame handling.

## Half-seam top cut
Planning parameters `middle_seam_stop` (`end` default | `center`) and `middle_seam_center_overlap_m` ([0, 0.03] m),
`cut_pattern middle_only` only (every other pattern, and the staged/side-tape/cut-debug/continue paths, refuse
`center`). The stroke ends at `seam_center` projected onto the seam (from `seam_start`, after the optional
`middle_cut_reverse_travel` swap) plus the overlap, beyond the centre in travel direction. The carton box keeps its
axis. The shortening loop is OFF in this mode: the half either plans in full or the plan fails (shortening would
leave tape uncut at the centre). `middle_cut_fraction` of the export is the planned fraction of the seam.

## Planned solution (export)
`~/get_planned_solution` (dual_cut_msgs/GetPlannedSolution) reads the retained plan under the plan lock (`busy`
if a planning call holds it) and neither consumes nor changes it. Each leaf stage with a trajectory becomes a
`PlannedSegment` with kind TRANSIT/ALIGN/DWELL/ENTRY/CUT/FINISH/LIFT from its stage name (`plan_export.hpp`; an
unknown name with a trajectory makes the export FAIL; scene-only stages are skipped), the executor's cut label
(`cut_id`), the physical cut (`left`/`right` swapped under `rotate_180`) and whether the stage's start scene
allows tool/carton contact. `cut_targets` are the retained cut target poses in `planning_frame`: two per CUT
segment (stroke start, stroke end; the tool-tip positions at cut depth, i.e. what the planner targets), the
convention `dual_cut_coordinator` reads. The response also carries start/end joint state, the snapshot identity
and the applied frame handling, and `raw`, the MTC Solution message as published on the introspection topic.

## Plan from an explicit start state
`~/plan_h_cut_from_state` (dual_cut_msgs/PlanFromState): exactly the planning group's six joints, finite, inside
the joint limits. The normal `plan_h_cut` flow runs with a `FixedState` stage (the move_group scene with those
joint values) instead of `CurrentState`, and the IK seeds of the left searches come from the same values. Refused
for `t_cut_handoff`, the debug patterns and `left_debug_stop_at_approach`. The retained plan is marked
`retained_from_override_`: `execute_planned_h_cut` refuses it for ever (replan with `plan_h_cut`).

## Safe unwind detour of a free-space transit
Planning parameters `transit_unwind_detour_enabled` (bool, default `false`; the operator enables it at runtime) and `transit_unwind_max_lift_m` (default 0.20,
[0.05, 0.30]); not in the YAML. Pure parts in `transit_unwind.hpp`. Used by `NearestBranchTransitPlanner` only when NO direct solution is acceptable.
- Flag `false`: the old planner code, bit for bit. Flag `true`: the SAME direct attempts in the same order (every IK branch, then the one retry of the
  original pose goal); the only differences are (a) a direct result whose path winds joint6 by more than pi is held instead of returned at once (the wrist gate
  would refuse it anyway) and (b) after ALL of them failed, ONE detour is tried as the last resort. If it fails the held direct result is returned unchanged
  (the gate refuses it as before). The detour deadline is `planning_timeout_s` counted from the START of the planner call (the direct attempts are charged to
  it; never a fresh budget). Note the MTC stage timeout (default 1 s) only caps the direct attempts: it is far too short to plan and validate a detour.
- Detour over the branches whose NET joint6 change is within pi: (1) straight Cartesian lift of the TCP along +Z by 0.05 m, 0.10 m ... up to the maximum
  (orientation held, fraction 1.0), (2) at the lifted pose the wrist is set to the goal branch by joint interpolation (joint6; joint4+6; joint4/5/6),
  (3) joint interpolation to the goal branch.
- STRICT scene: refused (clear message) when the transit's start scene allows ANY tool/carton contact (cut-debug / side-tape starts), or the start state is not
  collision free with every link vs `detected_carton` forced to "not allowed". All legs are planned and validated in that strict scene.
- Clearance floor: the lifted pose, the wrist waypoint and every dense sample of the wrist leg keep >= 5 mm, robot vs world AND robot vs robot (SRDF-disabled
  pairs ignored). A lift height whose lifted pose is closer is rejected; the floor is never lowered.
- The legs are stitched at FULL STOPS (zero velocity and acceleration at both joins, durations kept) into ONE trajectory of the same `MoveTo` stage (still a
  recorded transit: `validateFreeSpaceTransitWrist` sees the sum). The stitched trajectory is validated DENSELY along the controller's quintic spline
  (every 0.01 s and 3 mrad): joint bounds, strict-scene collision, 5 mm wrist-leg clearance, total joint6 travel <= pi, path constraints (if any), leg 1 a
  straight +Z line (<= 1 mm, <= 1 deg, no joint jump > 0.05 rad). Candidates are validated best first (smallest total joint travel); the first that passes
  before the deadline is used, nothing unvalidated is ever returned.
- A used detour is logged as WARN `transit detour via unwind waypoint: lift X mm, joint6 a->b rad ..., total joint6 travel T rad (direct refused: <reason>)`
  and appended to the plan response text of `plan_h_cut` / `plan_h_cut_from_state` (`stage '<name>': ...`).
- Limitation (found 2026-10-22): the per-arm model has a coarse cylinder for `link4`, the dual model the detailed hulls, so the executor cannot see some
  wrist-camera/link4 grazes of the wrist swing (<= 1.5 mm in the dual model); the dual coordinator's world check catches them (fail closed, plan refused). The
  per-arm scene also has no table.

## Early stop of the top-cut fraction descent (2026-10-24, search reduction only)

`middle_only` retries a failed plan with the cut END pulled back along the seam in 5 % steps (100 ... `top_cut_min_cut_fraction`), every step a full MTC replan.
A failure BEFORE the cut stroke whose inputs are identical for every fraction cannot be cured by a shorter cut: the transit to the seam-start hover
(`approach pose above middle seam cut`), the short `align above middle seam start` move, and the wrist gate (`validateFreeSpaceTransitWrist`, untouched) on
that transit. The stroke START, the blade direction and the clearances do not change when only `seam_end` moves, so hover pose, start state and scene are the
same for every fraction. The entry/descend stage is deliberately NOT in the set (with `middle_cut_sag_m > 0` the sag profile of the shortened seam moves the
start depth). Planning parameter `top_cut_fraction_descent_stop_on_fraction_independent_failure` (default true; false = plain 5 % descent).
- On such a failure the 5 % steps in between are skipped and the MINIMUM fraction is tried ONCE (narrowing: 4 of 165 recorded descents with such a failure
  later succeeded, always at the minimum fraction); if that fails the descent ends as before. One WARN line names stage and reason, the refusal text carries
  `[top cut descent: ...]`.
- Only `middle_only`, not `middle_cut_reverse_travel` (the stroke start moves with the fraction there), not `half_seam`, single transit-lift candidate.
- Stage names are read from MTC's `explainFailure` text (`top_cut_descent.hpp`, unit-tested); anything not classified keeps the plain descent (fail open to
  the old behaviour). Evidence: `scan_results/fallback_speed_20261024/README.md`.
