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
