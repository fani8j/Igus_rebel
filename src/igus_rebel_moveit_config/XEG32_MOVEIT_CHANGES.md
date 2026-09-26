# XEG-32 MoveIt integration changes

The existing ReBeL arm configuration is preserved. This update adds the XEG-32
semantics needed for MoveIt to list the gripper as a planning group.

## Added SRDF group

`xeg32_gripper`

The group contains:
- `xeg32_base_link`
- `xeg32_left_carriage_link`
- `xeg32_right_carriage_link`
- active/master joint: `xeg32_left_carriage_joint`

`xeg32_right_carriage_joint` remains the URDF mimic joint and is marked passive
in the SRDF. `xeg32_camera_tilt_joint` and the camera-mount link are removed
from the URDF and SRDF with the absent camera holder.

## Added end effector

- name: `xeg32`
- parent group: `igus_rebel_arm`
- parent link: `link6`
- end-effector group: `xeg32_gripper`

## Added adjacent self-collision exclusions

Only these directly connected pairs were disabled:
- `link6` / `xeg32_base_link`
- `xeg32_base_link` / `xeg32_left_carriage_link`
- `xeg32_base_link` / `xeg32_right_carriage_link`

If the gripper is still shown red in MoveIt, regenerate the self-collision
matrix with MoveIt Setup Assistant rather than disabling all XEG collisions.

## Tool center point

`xeg32_tool_tip` is the TCP for Servo and all box/MTC planners.

The active TCP is defined in `igus_rebel_xeg32_wrapper.urdf.xacro`, independently
of optional gripper geometry:

- `link6 -> igus_tool0`: xyz `[0, 0, 0.0527]` m, RPY `[0, 0, pi]`.
- `igus_tool0 -> xeg32_tool_tip`: xyz `[0.068936, 0.046766, 0.127060]` m,
  RPY `[0, 0, 0]`. These values are `OffsetX`, `OffsetY`, `OffsetZ`,
  `OffsetA`, `OffsetB`, and `OffsetC` from `knivetool4_with_pose2` in
  `knivetool4.xml`, converted from millimetres to metres.
- Both frames remain available with `attach_xeg32:=false`.
- Arm kinematics, the `link6` IK solver tip, and the rough XEG mount
  (`0 0 0.052` m, 15-degree Z rotation) are unchanged. MTC uses the fixed
  `xeg32_tool_tip` descendant explicitly as its IK frame.

The rough `xeg32_cutter_link`, its mounting joint, box and pyramid visual/collision
geometry, and old cutter-derived TCP joint are removed. The cutter Xacro arguments
are removed as well. The unused pyramid STL remains as an archived asset.
Four cutter-specific SRDF collision exclusions remain removed. The H-cut MTC task
supports an explicit `allow_tool_carton_contact` policy: when enabled, the
`xeg32_tool_tip`/`detected_carton` allowance is applied during each canonical
descend, traverse, and retract sequence, then restored before the next transit.
`xeg32_tool_tip` has no collision geometry, so the collisionless TCP marker
itself cannot collide; robot-arm, base/carriage, and all other object collisions
remain checked. Robot-arm and XEG-32 base/carriage physical meshes,
self-collision, the `detected_carton` planning geometry, motion gates,
joint-state checks, and Servo rules are unchanged.

The green TCP marker is visual only (6 mm-radius sphere). The physical cutter
body remains absent from collision checking. Scoped contact allows geometric
carton penetration but does not model cutting force or material deformation.
`cut_clearance_m` is the signed side-cut offset and `middle_cut_clearance_m` is
the signed middle-seam offset: positive values hover above the detected plane,
while negative values penetrate it. The configured values are -1 mm for side cuts
and -4 mm for the middle cut. Side penetration is capped at 3 mm and middle
penetration at 5 mm, accepted only when `allow_tool_carton_contact` is enabled.

Side-cut roll is configured independently with `left_side_cut_tilt_deg` and
`right_side_cut_tilt_deg`. As prior-model context, a batch scan reconstructed 86 structurally complete
detections with the recorded point clouds and the then-current camera-to-base transform;
-10-degree left roll planned all 59 boxes with at most one retry for two stochastic
OMPL transits. The selected -5-degree right roll planned 52/59 boxes in two repeated
runs; the other seven failed the right-side approach IK precheck. These angles may
be changed at runtime within [-90, 90] degrees while execution is idle; any accepted
change clears the retained plan.

The current rotated-box workspace map uses the complete combined H-cut, the
recorded carton geometry, scoped tool/carton contact, the prior-model camera-pole
collision (since removed with the absent camera holder; do not reintroduce camera
geometry), the selected -10/-5 degree side tilts, and all other MoveIt collisions. Its focused
grid passed 70/100 samples. The conservative sampled core is carton top-center X
`[0.200, 0.225]` m, Y `[-0.175, -0.125]` m, and yaw -45 through +30 degrees
relative to the reference capture, sampled every 15 degrees. All 24 core
position/yaw combinations planned. Across the initial and two repeated runs at
the two yaw extremes, those four core positions passed 24/24 plans. A demonstration
pose at center `(0.200, -0.175)` m and +30 degree relative yaw produced the complete
left-middle-right H-cut in RViz.

The reference capture's seam axis points at 74.324 degrees in `base_link`, so
the conservative relative-yaw interval corresponds to an absolute seam-axis yaw
of approximately 29.3 through 104.3 degrees, modulo 180 degrees. -75 degrees
failed 9/9 refined samples and +60 degrees failed 20/20 focused samples; do not
treat those orientations as part of the working region. The earlier
`box_workspace_*_20260910.json` scans used obsolete +5/-10 degree side tilts and
are superseded by `rotated_box_workspace_recommendation_20260912.json` and its
checksummed evidence files in `workspace_scan_artifacts/`.

These are discrete mock-hardware planning samples, not continuous reachability
or physical cutting guarantees. Replan from the live joint state for every
detected carton pose and keep execution disabled until cutter-body clearance,
calibration, cutting force, and material behavior are independently validated.

A cutter-roll refinement inside that conservative core screened 12 left/right
tilt pairs at the eight yaw-boundary placements, then evaluated four finalists
on all 24 core placements. The current `-10`/`-5` degree pair passed 71/72 new
complete-core repeat plans. `-10`/`-7.5` degrees passed 24/24 in the finalist run
and 48/48 in two repeats (72/72 total), with a 2.045 s median successful planning
time. This is the strongest core-planning candidate, but it is not yet the global
default: it still requires validation across the 59 reconstructed cartons and a
physical low-speed cutting trial. Raw results are checksummed in
`workspace_scan_artifacts/core_cutter_pose_scan_20260912.json` and
`workspace_scan_artifacts/core_cutter_pose_baseline_20260912.json`.
As prior-model context, three `right_side_cut_tilt_deg=-2.5` comparison-video failures were deterministic
for the rotated demonstration pose, not an OMPL timeout. MTC reached the right-side
stroke, but `CartesianPath` stopped at 22.222% because the remaining path collided with
`detected_carton` at the then-modeled camera-holder link; Pilz LIN failed on the same collision.
Raising the stroke 6 mm above the modeled top plane succeeded, confirming collision
geometry rather than IK, but that is not a cut and is not a valid fallback. That holder
link is now removed with the absent camera holder; do not reintroduce camera geometry
or disable this carton collision to reproduce those combinations. At this pose, use
right-side roll -10, -7.5, or -5 degrees; all nine combinations with left-side roll
-15, -10, or -5 degrees completed a fresh full validation pass.

A recorded-box translation scan confirmed that position dominates the remaining
right-side IK failures more than cutter roll. Translating every validated box
snapshot by `+0.140 m` on `base_link` Y, while retaining all measured dimensions,
surface normals, `-10`/`-5` degree side tilts, 30 mm hover clearance, carton
contact policy, and the fixed external camera-pole collision object planned the
complete combined H-cut for 59/59 boxes in two consecutive runs. The pole is not
gripper-mounted camera or holder geometry. A smaller `+0.040 m` shift planned the
right-side stroke for 59/59 then 58/59 boxes, but only 50/59 complete combined
cuts; it is therefore not the recommended work pose. The `+0.140 m` result is a
mock-hardware placement study, not authorization to offset live perception data:
physically place the carton/cutting fixture nearer `base_link` +Y and verify its
detected center in the documented workspace core before any low-speed trial.

`pre_cut_stabilization_duration_s` adds a stationary hold at the final,
collision-checked approach pose before each segment enables carton contact and
descends toward the cut. The default is 1.0 second and applies through the common
segment builder to combined, H, individual, partial, preview, and diagnostic plan
patterns. The hold is an explicit two-waypoint robot trajectory with identical
joint positions, zero velocity, zero acceleration, and the configured duration;
it is not a callback sleep. Generated plans are rejected unless every expected
hold has that structure and timing. Runtime values must be finite within
`[0, 10]` seconds, `0` disables the hold, changes are refused during execution,
and accepted changes clear retained plans.

This timed command allows the controller to settle but does not prove the blade
is mechanically stable or in contact. Tracking error and cutting force remain
unmeasured.

These geometric checks do not establish a safe physical penetration depth. Keep
physical execution disabled until cutting force and the complete cutter geometry
are independently validated, and discard plans and tool-relative targets made
with the old TCP.

## Staged per-cut debug and correction workflow

`box_h_cut_executor` implements a staged session for teaching one canonical cut
(`cut_debug_cut`: `left`, `middle`, or `right`) without changing the existing
combined/H plan path. The old seam-only `start_tcp_debug` /
`record_tcp_debug_point` services are removed; the session services below are
the single convention.

State machine (`cut_debug_cut` frozen at session start):
`start` plans the configured start target (approach transit, middle-seam alignment where
applicable, pre-cut stabilization dwell, descend) and retains it for explicit execution.
An existing record seeds repeat teaching, but nominal/observed deltas are computed from
the uncorrected geometry baseline so a replacement record is standalone rather than
residual. Each `save_cut_debug_point` after teleop captures the full observed TCP pose
via joint-state FK and persists it immediately without requiring any endpoint or lift
plan: the first save records the teleop-corrected start (`start_saved`), and the operator
then teleoperates the endpoint separately and calls the same save service, which records
the endpoint and completes the record (`complete`). A save retains no plan (any retained
plan is cleared) and never dispatches motion. Start and end observations are thus
persisted independently, and the final record is complete after the endpoint save. Only
`complete` records are loadable as corrections; `start_saved` is an intermediate. Left approach and
finish move 10 mm laterally toward the carton exterior without shifting the cut start or
end; middle/right retain surface-normal hover/retract. For every left, middle, and right
target, the saved full TCP quaternion is mapped from its recorded cut basis into the
current detected cut basis. A left taught pose may validly use tool local X or
tool local Z as its penetration axis; at least one of the two must point into the carton. Middle/right use tool local Z only. This preserves the physically taught blade angle
at capture and adds subsequent carton rotation rigidly. No service ever dispatches
motion implicitly: every stage runs only through an explicit
`execute_planned_h_cut` call, and session dispatches additionally require
`debug_motion_execution_enabled` on top of `execution_enabled` (both default
`false`). Out-of-order calls, planning a normal H-cut mid-session, and
starting a session over a retained plan or active session are all rejected
with precise messages. Normal retained plans still refuse a stale snapshot;
sessions freeze their snapshot at start because teleop correction inherently
outlasts the snapshot age bound.

The retained record schema is `box_h_cut_correction/v1`
(`<cut>_<snapshot_id>.json` in `cut_debug_output_dir`, default
`$HOME/cut_debug_corrections`, written atomically via temp-file plus rename):
snapshot ID/capture/acquisition stamp, cut name, planning frame, tool link,
nominal/observed full start/end poses, world and cut-local (`along`/`across`/
`normal`) translation deltas, world and cut-local orientation deltas plus
angle, the cut basis, status (`start_saved`/`complete`), and created/updated
stamps. Translation values must be finite; there is no per-axis magnitude limit.
Orientation corrections accept the full finite unit-quaternion range; applying
a record still fails closed unless the mapped penetration axis (left: X or Z; middle/right: Z) points into the carton surface.

`cut_debug_correction_enabled` (default `false`) plus per-cut absolute paths
(`cut_debug_left_correction_file`, `cut_debug_middle_correction_file`,
`cut_debug_right_correction_file`) opts later plans into learned corrections. Each file
must be a `complete` record whose cut matches its slot. Combined plans apply every set
slot to its segment; `t_cut` and `t_cut_handoff` contain only left and middle
segments, so a set right correction is loaded, validated, and left unused rather
than failing closed; individual plans apply only their cut's
slot. Empty means no correction for that cut; any malformed or mismatched nonempty record,
or enabling with all slots empty, fails closed. The local deltas are rotated onto each newly detected cut basis
and the tool local Z into the surface plus the opposite-travel side facing are re-validated.
Anything else fails closed, and with the opt-in off the file paths alone
change nothing.

For learned LEFT poses only (with `cut_debug_correction_enabled` and a loaded
LEFT correction file), `left_adaptive_orientation_enabled` (default
`true` in `box_h_cut_executor.yaml`) tests TCP-local Y (blade-length/seam axis)
rolls of the mapped quaternion as `q_candidate = q_mapped * Qy(theta)`. TCP
position and blade Y direction are preserved; only the X/Z penetration roll
varies. Candidates run in deterministic smallest-magnitude order 0, +step,
-step, ... (0, +5, -5, ... at the default 5.0-degree step) through
`+/-left_adaptive_orientation_max_deg` (default 45.0). Every candidate must
pass the existing left X-or-Z penetration gate plus IK and collision
feasibility at the final target poses; the probe builds the task's carton and
fixed camera-pole scene and checks self-collision as well as world collisions.
Previously collision-blind IK selected a -20 degree roll for the September 2026
capture and left an invalid approach goal; the collision-aware search selected
+45 degrees at the default 5-degree step in isolated mock planning. Hover/approach
candidates must be fully collision-free; cut-target candidates honor only the
scoped tool/carton contact allowance. MTC and Cartesian path validation remain
the final authority; neither a mock replay nor this pose search validates the
old taught record for the current physical TCP. Retain correction opt-in and both
execution gates off for hardware until the left pose is re-taught.

The multi-segment planner keeps its original add-carton-then-allow ordering; only the staged builders add the carton with
the scoped allowance atomically, because those stages legitimately start in
contact. The scoped tool/carton contact policy and both execution gates are unchanged.
The fixed external `camera_pole` remains required in every isolated H-cut planning
scene; only the nonexistent gripper-mounted camera and holder geometry were removed.

A stage dispatch that fails in the controller (non-success status or exception)
is never marked executed: the stage stays planned with no retained plan, and
`save`/`execute` both refuse with a message directing the operator to
`clear_plan` and restart the session, so no observation is ever captured from
an unreached pose. While a session is active, every planning or correction
parameter mutation is rejected (`clear_plan` first); only the two execution
gates remain operable. Saving additionally requires fresh, complete joint
data: the newest `/joint_states` must be under one second old and must contain
every planning-group joint with finite positions, otherwise the save is
refused rather than captured from defaults or stale data.

## Controller files intentionally unchanged

`moveit_controllers.yaml`, `ros_controllers.yaml`, and
`igus_rebel2.control.xacro` still control only ReBeL joints 1-6.

This is intentional. The HIWIN XEG-32 is separate hardware and should receive
its own controller/hardware interface later. Execution of an XEG gripper plan
will not work until that controller is added.

## Joint limits

The Onshape-exported XEG velocity/effort values were not copied into
`joint_limits.yaml` because they are exporter placeholders. Configure the
gripper limit after mapping the HIWIN XEG-32 command convention to the single
master URDF carriage joint.

## Side-cut direction

Both side strokes travel from the detected back midpoint toward the front midpoint:
`back_middle_left -> front_middle_left` and `back_middle_right -> front_middle_right`.
Their knife facing is explicitly rotated 180 degrees from the travel-aligned orientation;
the tool local Z remains directed into the carton. The middle seam direction, segment
order, collision policy, stabilization dwell, and execution gates are unchanged.

## Free-space transit consistency

Free-space transits (initial approach and inter-segment finish-to-approach
moves) use dedicated `transit_velocity_scaling` /
`transit_acceleration_scaling` (`box_h_cut_executor.yaml`, default 0.05;
finite values in `(0, 1]` only, out-of-range fails closed). Contact
descend/traverse/retract strokes keep the 0.05 `velocity_scaling` /
`acceleration_scaling`, and `pre_cut_stabilization_duration_s` stays 1.0 s,
so no physical cutting speed changes by default. Each free-space transit
tries a direct collision-checked joint interpolation first via MultiPlanner
ordering ([joint interpolation, existing transit]), falling back to the
original transit solver. Full cut poses, `min_fraction=1.0` acceptance, the
scoped tool/carton contact policy, and the fixed external `camera_pole`
requirement are unchanged. A free-space solution whose joint6 travel exceeds
pi rad, summed over every consecutive waypoint of its free-space transit
subtrajectories, is rejected before the plan is retained, published, or
executed; there is no silent fallback after this check. The pre-TCP-change
taught left correction on the saved carton was rejected at 4.806 rad of
middle-transition joint6 travel in isolated mock planning; re-teach it on
the current TCP before physical use. Faster transit shortens only non-contact
portions and is no guarantee of a fast plan or a physically safe stale TCP: the
correction opt-in (`cut_debug_correction_enabled`, default `false`) and both
execution gates (`execution_enabled`, `debug_motion_execution_enabled`, both
default `false`) are unchanged, and no physical dispatch is recommended. Tune
mock-only first (e.g. 0.10 transit scaling shortens only non-contact
portions); physical speed needs separate review.
## T-cut handoff

`cut_pattern: t_cut_handoff` is first-arm-only: the full left cut plus an
executable bounded middle-seam prefix ending at the declared handoff
endpoint (`interpolate(seam_start, seam_end, <resolved fraction>)`, included in
the published target poses for external second-arm integration). It does NOT lower any
`min_fraction` acceptance: the planned path is accepted only when it fully reaches that explicit handoff endpoint under `min_fraction=1.0`. After fully reaching that endpoint, the retained first-arm
plan ends at the cut-level handoff pose and deliberately skips the hover
retract (`finish pose above middle seam handoff`) and its following
restore-collision stage, so the scoped tool/carton contact allowance stays
active at the terminal pose. The retained middle traverse is therefore an
executable first-arm prefix, not the complete seam; no second arm is implemented.
Execution still requires the normal gates.
`t_cut_middle_handoff_fraction` selects the stop fraction along the middle seam:
`0.0` (default) selects automatic mode: bounded bisection over `[0.05, 0.95]`
finds the largest fully reachable declared prefix with `<=0.01` fraction tolerance,
keeping `min_fraction=1.0` for every candidate. A nonzero value in `[0.05, 0.95]`
is a manual override stop fraction; out-of-range nonzero values fail closed.
Full `t_cut` stays the full left cut plus the full middle seam and is unaffected,
and full `t_cut`/`combined` continue to retract above the cut and restore
collision checking. The left-to-middle transition (combined, `t_cut`,
`t_cut_handoff`) is: left lateral withdrawal to the finish pose outside the left
cut, then a surface-normal lift that holds the left orientation, then restore of
knife-tip collision checking, then free-space transit/rotation to the middle
approach. No current-plan serialization is implemented: the retained plan lives
only in the running executor until restart. No second-arm behavior changes.
`t_cut_handoff` applies the same left and middle correction slots as `t_cut`; a set right record loads but stays unused because the plan has no right segment.
Middle translation end deltas are fraction-interpolated and the taught middle end
orientation is slerp-interpolated by the resolved fraction for the
bounded prefix; the middle tool-Z penetration gate is revalidated unchanged.

## Rebuild

From the workspace root:

```bash
colcon build --symlink-install
source install/setup.bash
```

Then relaunch MoveIt. The planning-group dropdown should contain both:

- `igus_rebel_arm`
- `xeg32_gripper`
