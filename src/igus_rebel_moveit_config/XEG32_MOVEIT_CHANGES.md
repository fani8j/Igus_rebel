# XEG-32 MoveIt integration changes

The existing ReBeL arm configuration is preserved. This update adds the XEG-32
semantics needed for MoveIt to list the gripper as a planning group.

## Added SRDF group

`xeg32_gripper`

The group contains:
- `xeg32_base_link`
- `xeg32_left_carriage_link`
- `xeg32_right_carriage_link`
- `xeg32_knife_handpiece` (added 2026-09-28)
- `xeg32_knife_holder` (added 2026-09-28)
- (`xeg32_wrist_camera` was added and removed on 2026-09-28; camera uninstalled)

Both carriage joints are fixed (`lock_joints` defaults to true), so the group has no
actuated joints and the SRDF lists no passive joint. The arm group is defined by its
chain only, and the SRDF has no virtual joint (the URDF root is already `world`).
The self-collision matrix was regenerated on 2026-09-27 with the Setup Assistant's
`computeDefaultCollisions` (100,000 samples): 17 adjacent and 59 never-colliding pairs
are disabled, 29 pairs stay checked (including link2-motor4, base_link-link4 and
base_link-motor5, which do collide). It was regenerated again on 2026-09-28 after
adding the three tooling links below (see "Gripper-mounted tooling collision
geometry"): 8 more `Adjacent`/rigid-overlap pairs and 27 more never-colliding pairs
are disabled; every other pairing of the three new links (for example against
`base_link`, `link1`, `motor1`, `motor2`) collided in at least one sample and stays
checked.

## Added end effector

- name: `xeg32`
- parent group: `igus_rebel_arm`
- parent link: `link6`
- end-effector group: `xeg32_gripper`

## Self-collision exclusions

The whole matrix (arm and gripper) comes from the 2026-09-27 regeneration described
above; see the comment block in `config/igus_rebel2.srdf`. The gripper's directly
connected pairs (`link6` / `xeg32_base_link`, `xeg32_base_link` /
`xeg32_left_carriage_link`, `xeg32_base_link` / `xeg32_right_carriage_link`) are among
the 17 `Adjacent` entries. `link1`-`link2`, `link2`-`link3` and `link4`-`link5` are
also kept as `Adjacent` although a fixed motor link sits between them in the URDF: once
the rigid motor link is merged into its parent they are one revolute joint apart, and
their coarse primitives overlap in most states. This deviates from a strict
Setup Assistant adjacency result and is an open review item.
The carriage collision geometry is `meshes/xeg32_collision/right_carriage.stl` (a
convex hull, used for both carriage links together with `part_1__3.stl`).

Regenerate the matrix by sampling whenever collision geometry changes, rather than
disabling all XEG collisions.

## Gripper-mounted tooling collision geometry (2026-09-28)

**Update:** the wrist camera was uninstalled on 2026-09-28 (it protruded past the blade tip along the knife axis and hit the carton) and its collision link removed; it will return with the redesigned combined knife/camera holder; the regression without it matches the pre-tooling results
(`scan_results/tool_collision_nocam_20260928/`). The text below documents the
version that still included the camera.

The ultrasonic knife handpiece, its holder, and the wrist camera + bracket are now
collision-checked. Previously only the bare XEG-32 body/carriages were modeled;
MoveIt could route any of this real hardware into the carton, the camera pole, or
the arm without noticing. New fixed links, all attached to `igus_tool0` with
identity origins (the meshes are authored directly in `igus_tool0` coordinates):

- `xeg32_knife_handpiece` — `meshes/xeg32_collision/knife_handpiece.stl`, a padded
  box (32 x 29 x 58 mm) on the handpiece axis (tool0 Z through the TCP's X/Y,
  `0.005937, 0.094121`), from the holder face (TCP Z - 75 mm) up to 20 mm below the
  TCP. The top 20 mm (the exposed blade tip that must contact the carton) is
  deliberately left with no collision geometry, matching `xeg32_tool_tip`.
- `xeg32_knife_holder` — two convex hulls derived from `JK-Final version.step`
  ("new gr holder-fin", the printed holder; the thin plate "new gr holder-fin_1" is
  folded in via extra padding rather than placed independently, since its position
  relative to the other two STEP parts is not given): `knife_holder_ring.stl` (the
  clamp ring around the handpiece) and `knife_holder_body.stl` (the mounting body
  that bolts to the gripper). Split into two hulls, not one, because the part is
  L-shaped and a single hull would fill the concave notch between the ring and the
  body with material that is not there. Registered so the ring's bore axis is the
  handpiece axis and its tip-facing face sits at the holder face, rotated so the
  body extends from the ring toward the gripper body (tool0 -Y, since the gripper
  bbox center is close to Y=0 and the handpiece/ring sit at Y=+0.094).
- `xeg32_wrist_camera` — two padded boxes estimated from the operator's photos
  using the gripper (`xeg32_base_link` + carriages span tool0 X -0.088..0.103,
  Y -0.068..0.070, Z -0.0007..0.065) as scale, since there is no CAD for it:
  `wrist_camera_bracket.stl` (30 x 30 x 65 mm, gripper top up to 125 mm) and
  `wrist_camera_body.stl` (70 x 70 x 70 mm, 115-185 mm above the gripper top),
  reflecting the photos' slender bracket rising to a bulkier camera body. The
  Logitech knife-view webcam is intentionally not modeled (operator).

All three links are in the `xeg32_gripper` SRDF group and end effector. The SRDF's
self-collision matrix was regenerated the same way as the 2026-09-27 pass (see
`config/igus_rebel2.srdf`'s comment block and
`scan_results/tool_collision_20260928/srdf_regen/`): pairs that are rigidly
attached to each other or to `link6` are disabled as `Adjacent`;
`xeg32_base_link`-`xeg32_knife_holder` and `xeg32_knife_holder`-
`xeg32_right_carriage_link` are also disabled, on the same rigid-overlap
reasoning, even though the tool's own `computeDefaultCollisions` reason string for
them is `Default` (in collision at the default/home state) rather than `Adjacent`;
pairs that never collided across 100,000 samples are disabled as `Never`. Every
other pairing of the three new links against the rest of the arm (`base_link`,
`link1`, `motor1`, `motor2`) collided in at least one sample and stays checked.

The H-cut executor's scoped tool/carton contact allowance
(`allow_tool_carton_contact`) only ever toggles the single configured
`tool_link` (`xeg32_tool_tip`, which still carries no collision geometry) against
`detected_carton`; it has no per-link list to update, so the three new links are
never included in that allowance and must never touch `detected_carton`.

**Finding (mock-only regression, `scan_results/tool_collision_20260928/`):** with
all three links active, `xeg32_knife_handpiece` and `xeg32_knife_holder` introduce
no new failures (confirmed by disabling each via a runtime AllowedCollisionMatrix
override and replanning: t_cut/t_cut_handoff/h/combined/cut-debug-left all
reproduce their pre-change plans and durations exactly). `xeg32_wrist_camera`,
however, collides with `detected_carton` (and, for the `h`/`combined` patterns,
also with `camera_pole`) at the recorded replay carton's approach and cut poses,
across every camera size tried (from an initial 90 x 100 x 145 mm box down to the
70 x 70 mm body committed here): disabling `xeg32_wrist_camera` alone (leaving the
other two links checked) reproduces the pre-change plans exactly for
`t_cut`/`t_cut_handoff`/`h`/`combined`/cut-debug-left-start. With the wrist camera
modeled as committed, only the middle-only cut-debug-start scenario and the
diagnostic-only service calls still "succeed" (the diagnostic call itself always
returns success even when every section it probes fails); every other snapshot
plan on the recorded replay carton (a small ~102 x 111 x 200 mm box) fails closed.
This reflects the H-cut executor's tight clearances (`hover_clearance_m` 10 mm,
`left_lateral_standoff_m` 25 mm) against a camera mount that, even conservatively
sized from the photos, does not have that much clearance from a carton this small
at these wrist orientations. This is a genuine, previously-invisible reach
conflict, not a modeling artifact of an oversized box — narrowing the box further
did not remove it. Before running H-cut with a wrist camera physically mounted,
either revisit the camera's real mounting position/size against the photos more
precisely, or open up the H-cut hover/lateral clearances, and re-run this
regression to confirm.

## Tool center point

`xeg32_tool_tip` is the TCP for Servo and all box/MTC planners.

The active TCP is defined in `igus_rebel_xeg32_wrapper.urdf.xacro`, independently
of optional gripper geometry:

- `link6 -> igus_tool0`: xyz `[0, 0, 0.0527]` m, RPY `[0, 0, pi]`.
- `igus_tool0 -> xeg32_tool_tip` (joint `igus_tool0_to_xeg32_tool_tip`): xyz
  `[0.005937, 0.094121, 0.130699]` m, RPY `[0, 0, 0]`. This is the touch-corrected
  TCP of 2026-09-24 (confirmed by the operator); the URDF comment notes that Z is held
  pending a stale-model check. The earlier committed value
  `[0.068936, 0.046766, 0.127060]` m (from the `knivetool4.xml` tool offsets) is
  obsolete, as is the TCP in effect for the 2026-09-14/16 teaching sessions; plans and
  taught translations made with an earlier TCP are not valid for the current tool.
- Both frames remain available with `attach_xeg32:=false`.
- Arm kinematics, the `link6` IK solver tip, and the rough XEG mount
  (`0 0 0.052` m, 15-degree Z rotation) are unchanged. MTC uses the fixed
  `xeg32_tool_tip` descendant explicitly as its IK frame.

The rough `xeg32_cutter_link`, its mounting joint, box and pyramid visual/collision
geometry, and old cutter-derived TCP joint are removed. The cutter Xacro arguments
and the unused pyramid STL are removed as well.
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
and -2 mm for the middle cut, with `hover_clearance_m` 10 mm (all read-only; edit the
YAML and restart). Side penetration is capped at 3 mm and middle
penetration at 5 mm, accepted only when `allow_tool_carton_contact` is enabled.

Side-cut roll is configured independently with `left_side_cut_tilt_deg` and
`right_side_cut_tilt_deg`. As prior-model context, a batch scan reconstructed 86 structurally complete
detections with the recorded point clouds and the then-current camera-to-base transform;
-10-degree left roll planned all 59 boxes with at most one retry for two stochastic
OMPL transits. The selected -5-degree right roll planned 52/59 boxes in two repeated
runs; the other seven failed the right-side approach IK precheck. These angles may
be changed at runtime within [-90, 90] degrees while no planning call, execution or
cut-debug session is active (otherwise the change is rejected as busy); any accepted
change clears the retained plan.

The rotated-box workspace map below was produced with an earlier model (TCP, joint
limits and gripper collision geometry have changed since) and uses the complete
combined H-cut, the recorded carton geometry, scoped tool/carton contact, the
then-modeled gripper-mounted camera holder (removed for years because no tooling
collision geometry was mounted on the gripper at all; a wrist camera, knife handpiece
and knife holder were re-added on 2026-09-28, see "Gripper-mounted tooling collision
geometry" above — this historical map predates that and is not evidence for it), the
fixed external `camera_pole` obstacle
(still required today), the selected -10/-5 degree side tilts, and all other MoveIt
collisions. Its focused
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
segment builder to combined, H, t_cut, t_cut_handoff, individual, diagnostic and
staged cut-debug start plans. The staged cut-debug start plan uses the same builder,
so it must also pass the dwell check and the free-space wrist guard (joint6 <= pi). The hold is an explicit two-waypoint robot trajectory with identical
joint positions, zero velocity, zero acceleration, and the configured duration;
it is not a callback sleep. Generated plans are rejected unless every expected
hold has that structure and timing. Runtime values must be finite within
`[0, 10]` seconds, `0` disables the hold, changes are refused as busy during planning
or execution and while a cut-debug session is active, and accepted changes clear
retained plans.

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
`complete` records are loadable as corrections; `start_saved` is an intermediate.
Stages are `start_planned` -> `start_executed` -> `start_saved`; the endpoint save
writes the `complete` record and ends the session. No save plans or retains an endpoint
or lift plan. Left approach and
finish move `left_lateral_standoff_m` (25 mm in the supplied YAML, allowed 5-25 mm)
laterally toward the carton exterior without shifting the cut start or
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
with precise messages. Normal retained plans still refuse a stale snapshot at
execution; a session requires a fresh snapshot when it starts and then freezes it,
because teleop correction inherently outlasts the snapshot age bound. A plan or
session loaded with `debug_allow_stale_snapshot=true` can never be executed.

The retained record schema is `box_h_cut_correction/v1`
(`<cut>_<snapshot_id>.json` in `cut_debug_output_dir`, default
`$HOME/cut_debug_corrections`, written atomically via temp-file plus rename):
snapshot ID/capture/acquisition stamp, cut name, planning frame, tool link,
nominal/observed full start/end poses, world and cut-local (`along`/`across`/
`normal`) translation deltas, world and cut-local orientation deltas plus
angle, the cut basis, status (`start_saved`/`complete`), and created/updated
stamps. On load, `planning_frame` and `tool_link` are required and must equal the
executor's active values (a record missing either is refused), the status must be
`complete`, and the recorded cut basis must be orthonormal and right-handed.
Translation values must be finite; there is no per-axis magnitude limit.
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
or enabling with all slots empty, fails closed. The local deltas are rotated onto each
newly detected cut basis, the saved full start/end quaternions are mapped from the
recorded cut basis into the current one, and the mapped penetration axis (left: tool X
or Z; middle/right: tool Z) is re-validated to point into the carton.
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
execution gates off for hardware until the left blade pose has been verified on the
current tool (the mock work uses an orientation-only derivative of the old record).

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
`joint_limits.yaml` because they are exporter placeholders. The carriage joints are
fixed by default; with `lock_joints:=false` the left carriage joint is prismatic and
the right one mimics it. Configure a gripper limit only after mapping the HIWIN XEG-32
command convention to that master carriage joint.

The arm joints use the igus/CPR limits 0.785 rad/s and 5.24 rad/s² in
`joint_limits.yaml` (the URDF `velocity` limits are 0.785 as well). The previous
2.16 rad/s / 10 rad/s² exceeded what the controller can follow. All executor and
calibration scaling factors, and the Pilz `cartesian_limits`, were rescaled on
2026-09-27 so the physical speeds did not change.

## Side-cut direction

Both side strokes travel from the detected back midpoint toward the front midpoint:
`back_middle_left -> front_middle_left` and `back_middle_right -> front_middle_right`.
Their knife facing is explicitly rotated 180 degrees from the travel-aligned orientation;
the tool local Z remains directed into the carton. The middle seam direction, segment
order, collision policy, stabilization dwell, and execution gates are unchanged.

## Free-space transit consistency

Free-space transits (initial approach and inter-segment finish-to-approach
moves) use dedicated `transit_velocity_scaling` / `transit_acceleration_scaling`
(`box_h_cut_executor.yaml`: 0.4127 / 0.2863 of the 0.785 rad/s / 5.24 rad/s² joint
limits, i.e. 0.324 rad/s / 1.50 rad/s², the physical speed of the operator-approved
0.15 transit scaling of 2026-09-26 at the former limits). Contact
descend/traverse/retract strokes use `velocity_scaling` / `acceleration_scaling`
0.1376 / 0.0954 (0.108 rad/s / 0.50 rad/s², the physical speed of the former 0.05),
and `pre_cut_stabilization_duration_s` stays 1.0 s, so cutting speed is unchanged.
All scalings are read-only, finite and within `(0, 1]`; out-of-range values fail
closed at startup.
Each free-space transit is wrapped by `NearestBranchTransitPlanner`: a pose goal is
first resolved to the exact-pose, collision-free IK branch nearest the propagated
start (seeded from the start and its equivalent spherical-wrist branches; branches
with joint6 change <= pi first, then branches at least 2 deg inside every URDF bound, then smallest joint distance), then planned with a
direct collision-checked joint interpolation first and the profile's pipeline planner
second (MultiPlanner [joint interpolation, existing transit]). If every branch fails,
the original pose goal is retried once. All attempts share one `planning_timeout_s`
budget. Full cut poses, `min_fraction=1.0` acceptance, the
scoped tool/carton contact policy, and the fixed external `camera_pole`
requirement are unchanged. A free-space solution whose joint6 travel exceeds
pi rad, summed over every consecutive waypoint of its free-space transit
subtrajectories, is rejected before the plan is retained, published, or
executed; there is no silent fallback after this check. The guard also fails closed
when the examined transit trajectories do not match the transit stages recorded while
building the task. Before the nearest-branch planner the pre-TCP-change taught left
correction on the saved carton was rejected at 4.806 rad of middle-transition joint6
travel; with it the same corrected `t_cut` planned 5/5 in isolated mock (1.66 rad,
`scan_results/hcut_wrist_branch_20260926/`). Faster transit shortens only non-contact
portions and is no guarantee of a fast plan or a physically safe stale TCP: the
correction opt-in (`cut_debug_correction_enabled`, default `false`) and both
execution gates (`execution_enabled`, `debug_motion_execution_enabled`, both
default `false`) are unchanged, and no physical dispatch is recommended.

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
`0.0` (default) selects automatic mode: 0.95 is probed first and used if it fully
plans; otherwise 0.05 must plan (else the search fails closed) and a bounded bisection
over `[0.05, 0.95)` (at most 7 probes) finds the largest fully reachable declared
prefix with `<=0.01` fraction tolerance, keeping `min_fraction=1.0` for every
candidate. The left roll search runs once per request and is reused by every probe.
A nonzero value in `[0.05, 0.95]`
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

## Parallel left orientation (opt-in, 2026-09-27)

With the real controller limits the taught left angle and every +/-45 degree roll of it
are unreachable on the replay carton. `left_parallel_orientation_enabled` (default `false`)
derives the left orientation from the detected cut basis instead: it searches tilt/pitch
within +/-45 degrees and yaw within +/-30 degrees of a flat blade (tool X = -normal,
Y = -along, Z toward the carton side) for the smallest blade-plane angle to the top face
(`theta_p`) whose whole left chain passes the left penetration gate, collision-aware IK and
a collision-checked, jump-free Cartesian chain from every collision-free hover-start wrist
branch, with a joint-limit margin (default 5 degrees). It fails closed with an active left
correction record, on timeout, and (while enabled) when a planned left contact stroke moves
any joint more than 1 rad. Replay carton result: tilt -10, pitch -5, yaw -10 degrees,
`theta_p` 11.2 degrees (nominal 80, taught 27). Evidence (mock, plan-only):
`scan_results/parallel_left_20260927/`. `NearestBranchTransitPlanner` now ranks goal
branches with a joint within 2 degrees of its URDF bound after those with margin (joint6
rule first), which keeps the middle seam off a joint4 = -177.6 degree branch after the
parallel left cut; the default plans are unchanged.

## Varying-orientation left cut (opt-in, 2026-10-13)

`left_parallel_varying_orientation_enabled` (default `false`; needs `left_parallel_orientation_enabled`,
the analytic IK backend, `motion_profile ompl_cartesian` and `cut_pattern` left_only/t_cut/h/combined)
lets the left_parallel orientation change along the stroke. The cut line is sampled every
`left_varying_station_spacing_m` (0.010 m, range [0.005, 0.03]); per station the same orientation
candidates as the fixed search are evaluated (penetration gate, all analytic IK branches, collision
under the scoped tool/carton allowance, `left_parallel_min_joint_margin_deg`), and a dynamic program
(`left_varying_chain.hpp`) picks the longest contiguous station interval whose consecutive
orientations differ by at most `left_varying_max_step_deg` (5, range [1, 15]) and whose IK branch moves
at most 0.2 rad per joint (no joint4/joint6 wrap), then the smallest summed score (theta_p dominant).
Every station and candidate is evaluated (no length-based pruning, which could hide a verifiable
chain behind a longer one that fails its exit/replay) and candidate/neighbour construction is
bounded by `left_parallel_search_timeout_s` and a 12000-grid-point cap.

The DP's stepwise orientation profile is then smoothed: keyframes start at the two end orientations
(piecewise slerp); a station is accepted when the penetration gate holds, theta_p and blade
deviation stay within the DP station's value + `left_varying_max_step_deg`, and a collision-free
in-margin IK branch continues from the previous station; the worst failing station becomes a knot
with its DP orientation until everything passes (all knots = the DP profile). The result is replayed
with the Cartesian interpolator exactly as the stage runs (entry, ONE stroke through the knots,
finish, lift; fraction 1.0 for every leg, <= 5 deg joint step between waypoints, bounds, margin, the
1 rad aggregate contact travel); a failing smoothed profile falls back to the DP profile, a failing
DP profile bans the offending start/edge/end and the DP is re-solved (bounded).

An interval below `left_varying_min_cut_fraction` (0.5, range [0.2, 1.0]) of the cut fails closed with
the achievable fraction; a shorter interval trims the cut (response + WARN name the trimmed end). The
plan uses the fixed parallel mode's stage structure at the first/last station, a JOINT goal for the
approach transit (the verified hover branch; a pose goal would let the transit pick an unverified
branch on which the jump-detection-free strokes can flip), and a SINGLE `cut left side cut` stage
using `WaypointCartesianPath` (a `CartesianPath` subclass): `computeCartesianPath` leg by leg over the
knots, whole-path fraction 1.0, one time parameterisation with the contact scalings. After MTC
planning the retained stroke trajectory is validated (single stage, bounds/margin, <= 5 deg joint
step, every planned station within 1 mm / 1 deg on the replayed IK branch and in order) in addition to
the unchanged `validateLeftContactJointTravel`. Refused with `left_cut_reverse_travel`, a left
correction record, `side_cut_orientation_mode != edge_aligned`, cut-debug sessions, side-tape
hand-off, `continue_side_cut`, `side_cut_penetrate`, `plan_t_cut` and `t_cut_handoff`. Evidence:
`scan_results/first_cut_plan_20261013/logs/run11_varying/` and `run13_varying_v2/`.

## Extended left cut and approach-only debug plan (opt-in, 2026-10-13)

New PLANNING parameters (validated, finite, "busy" like the other `left_varying_*` ones; all default to the
old behaviour): `left_cut_extend_to_corners` (false), `left_cut_corner_margin_m` (0.020, [0.005, 0.10]),
`left_cut_seam_cover_m` (0.030, [0, 0.10]), `left_debug_stop_at_approach` (false).

**`left_cut_extend_to_corners`** (needs `left_parallel_varying_orientation_enabled`; planning is refused with
a message otherwise, and `start_cut_debug_session`, `plan_side_tape_handoff`, `side_cut_penetrate`,
`continue_side_cut` and `plan_t_cut` refuse while it is true). With A = back_middle_left, B = front_middle_left,
u = (B - A)/|B - A| and s(P) = (P - A).u: `s_start = s(back_left) + margin`, `s_end = s(front_left) - margin`;
the candidate line runs from `A + s_start u` to `A + s_end u` (the nominal line extended both ways; since
back/front_middle_left are midpoints with seam_start these are collinear with the edge). Both ends go through
the same `targetPose()` as the nominal cut_start/cut_end (surface-normal offset `cut_clearance_m`, same
orientation direction), so the line stays at the nominal cut height. Fails closed on a non-finite or degenerate
span, a span shorter than / not containing the nominal segment, corners or seam_start more than 30 mm off the
nominal line, or a seam projection outside the span. Stations stay <= `left_varying_station_spacing_m` (about
29 instead of 17 at 10 mm, so the search takes proportionally longer; `left_parallel_search_timeout_s` still
bounds it and its error names the station count; the 12000 cap is on orientation grid points, not stations).

Seam band: the kept interval must satisfy first station <= floor(band_lo/h) and last station >= ceil(band_hi/h)
for band = s(seam_start) -/+ `left_cut_seam_cover_m` clamped to the span (h = station spacing), so it contains the
band. This is `left_varying::bestVerifiedEndCovering` (longest verified chain among those covering the required
station range; also used inside the repair loop, so a repaired profile still covers it); the unconstrained
`bestVerifiedEnd` is unchanged for the non-extended path. Minimum length = `left_varying_min_cut_fraction` x the
NOMINAL |B - A| (not the extended span); the failure message gives the numbers (and the longest interval anywhere).
The plan summary states: extended span length, kept length (and % of nominal and its position relative to
back_middle_left), trim at the back and front end of the span (mm), that the seam band is covered, plus the
unchanged theta_p / blade deviation / knots / stroke time. Everything downstream (joint-branch hover goal, hover
offsets, replay, `validateLeftVaryingRetainedStroke`, `validateLeftContactJointTravel`, fraction 1.0, carton contact
scope only for the stroke) runs on the kept interval unchanged.

**`left_varying_start_extra_inward_m`** (PLANNING, default 0, range [0, 0.005] m; YAML 0.003 since 2026-10-13,
operator decision). Pushes the blade further INTO the box (horizontally, opposite to `offsetLeftPoseOutward`,
perpendicular to the cut line and the surface normal) by this much at the START of the kept varying left stroke,
tapering linearly to 0 at its finish: station offset = `extra * (1 - s/L)` (`left_varying::inwardTaperM`), s = arc
position from the kept start, L = kept length. The offset is added to the station positions inside
`searchLeftVaryingOrientation`, so the IK/collision search, DP, hover (offset start + `left_lateral_standoff_m`),
lift, Cartesian replay, plan stations and `validateLeftVaryingRetainedStroke` all see the same positions. The kept
interval is only known after the DP, so the search runs as passes: pass 1 tapers over the whole searched span; if the
kept interval is shorter, the COMPLETE search is re-run with the taper window set to that kept interval, until a pass
keeps exactly the interval its window is defined on (max 3 passes, otherwise fails closed). With 0 the code path is
unchanged. > 0 with `left_parallel_varying_orientation_enabled` false is refused (`leftVaryingConflict` for
left patterns; `refuseLeftVarying` for staged/side-tape/continue/plan_t_cut paths): because the YAML default is
varying OFF, set the extra to 0.0 (or enable varying) before using those paths. The plan summary states the inward
offset at the start and the finish (mm). Carton contact scope, `min_fraction` 1.0 and the retained-stroke checks are
unchanged.

**`left_debug_stop_at_approach`** (`cut_pattern left_only` only; `plan_h_cut` refuses any other pattern, and the
entry points listed above refuse while it is true). `plan_h_cut` first plans the full left cut exactly as usual
(varying search with feature 1 if enabled, all validators), then plans a second task over the same segment with
`stop_at_hover_start`: transit to the left approach pose and the pre-cut stabilization dwell, then nothing (no
carton contact scope, lateral entry, stroke or lift). In varying mode the second task is handed the full run's
verified plan (no second search), and its trajectory must END on the verified hover joints
(`hover_start_joints`, 1e-3 rad); in the other left modes its hover pose and the cut targets must equal the full
plan's. Any mismatch fails closed. `validateFreeSpaceTransitWrist` and the dwell check run on the retained task;
the retained-stroke check is skipped only because that task has no stroke (the full task passed it). The full cut's
markers/targets are published, the retained solution is the approach-only one. The retained plan is flagged
(`retained_debug_motion_`, cleared by `clear_plan`/replanning and consumed at dispatch): `execute_planned_h_cut`
needs `execution_enabled` AND `debug_motion_execution_enabled`, plus the usual freshness and
`debug_allow_stale_snapshot` blocks. The response is labelled APPROACH-ONLY and gives the full-cut summary, hover
pose and hover joints.

## Dual-arm planning stack: snapshot frame, half seam, plan export, plan from state (2026-10-15)
Design: `scan_results/dual_arm_stack_20261014/design/P3_P4_DESIGN.md` sections 2-4 and 7; regression evidence in
`scan_results/dual_cut_p3_regression_20261015/`. With every new parameter at its default the left arm's plans are
unchanged (regression below).

**New read-only parameters** (`kParameterTable`): `snapshot_frame_mode` (`planning_frame` | `fixed_transform`),
`snapshot_frame_id`, `snapshot_planning_base_xyz`, `snapshot_planning_base_rpy`, `snapshot_relabel` (`none` |
`rotate_180`), `plan_export_only`. **New planning parameters**: `middle_seam_stop` (`end` | `center`),
`middle_seam_center_overlap_m` ([0, 0.03]). `validateSettings`: names/values/sizes checked; non-zero
`snapshot_planning_base_*` without `fixed_transform` refused; `fixed_transform` or a relabel require
`plan_export_only=true` and `cut_debug_correction_enabled=false` and `side_tape_handoff_enabled=false`.

**Snapshot** (`applySnapshotFrameHandling`, last step of `parseSnapshotFromDocument`, maths in
`include/igus_rebel_moveit_config/snapshot_frame.hpp`): `frame_id` must equal `snapshot_frame_id`; the 11 points
get `p = B^-1 p_snapshot`, the normal is rotated; `header.frame_id` becomes `planning_frame`; then `rotate_180`
swaps back_left/front_right, back_right/front_left, seam_start/seam_end, back_middle_left/front_middle_right,
front_middle_left/back_middle_right (centre and normal kept) after checking that the label handedness is unchanged
(mirror -> refused). With the defaults the function returns before touching anything. Not transformed, hence
refused in a non-default mode: the `side_tape_seed` block (`parseSideTapeSeedFaceFromDocument`) and pattern `h` with
`approach_joint_degrees` (`planSnapshot`). Gtests: `test/test_snapshot_frame.cpp`.

**Export-only** (`plan_export_only`): the constructor creates the side-tape, T-cut, staged top-cut and cut-debug
services and `execute_planned_h_cut` only when it is false; advertised then: `plan_h_cut`, `plan_h_cut_from_state`,
`get_planned_solution`, `clear_plan`, `diagnose_external_snapshot_sections`.

**Half seam** (`planSnapshot`): after the `middle_cut_reverse_travel` swap, `seam_end = seam_start + (f_c +
overlap/len)(seam_end - seam_start)`, `f_c` = `seam_center` projected onto the seam; refused unless `f_c` is within
10..90 % and the stop <= 100 %; `cut_pattern` must be `middle_only` (`planSnapshot` refuses other patterns,
`refuseHalfSeam` the staged/side-tape/cut-debug/continue entry points); the shortening loop is skipped
(`half_seam` in the loop's exit condition, message prefix "middle_only half seam ... did not plan; shortening is
disabled"). The response of a half seam adds `middle_seam_stop=center (half seam, stroke ends X mm past the seam
centre ...)`.

**Export** (`exportPlannedSolution`, `include/igus_rebel_moveit_config/plan_export.hpp`): `retainPlan` additionally
stores the cut targets (two per cut segment, exactly what `publishTargetPoses` sends), the segments, the pattern,
`plan_seq_` (+1) and the pointer of the task it retained (`exportable_task_`; a plan assigned to `planned_task_`
any other way, e.g. a cut-debug start, is not exportable). The classifier maps stage-name prefixes to kinds
(TRANSIT: `approach pose above/outside`, `transition from`, `transit to supplied`; ALIGN: `align above`; DWELL:
`stabilize blade before`, `hold at penetration before`; ENTRY: `enter laterally into`, `descend from aligned`; CUT:
`cut `; FINISH: `finish pose outside/above`; LIFT: `lift clear of`, `lift straight up`, `raise to safe transit
height after`). The last LIFT prefix is an addition to the design table (the Cartesian raise after the left cut in
`t_cut`, `middle_transit_extra_lift_m`). An unknown name with a trajectory, a CUT-stage/target count mismatch (not
two targets per CUT stage) or a leaf/message count mismatch fails the export. `raw` = `Solution::toMsg(msg,
&task.introspection())` plus the introspection task id = the message `publishSolution` published.
Gtests: `test/test_plan_export.cpp`.

**Plan from state** (`planHCutFromState`): validates the six joints against the group and the joint bounds, sets
`start_state_override_` (RAII), clears the retained plan and runs `planExternalSnapshot`. In `planCutSegments` the
override replaces the `CurrentState` stage by `FixedState("start state (override)")` (the move_group scene fetched
once through a temporary node with every component, as `CurrentState` does, the group joints replaced, velocities
zeroed) and seeds `ik_state` (all left-search/transit seeds derive from it). Refused for `t_cut_handoff`, the debug
patterns and `left_debug_stop_at_approach`. `retainPlan` stores `retained_from_override_`; `executeRetainedPlan`
refuses such a plan right after the stale-snapshot block.

**Regression (mock, domain 203, 2026-10-15; evidence `scan_results/dual_cut_p3_regression_20261015/`).** Pristine
source = `box_h_cut_executor.cpp.before` (sha256 in `source_sha256_before.txt`). 07:49 and 09:20 captures (re-stamped), from
home, `t_cut` / `left_only` / `middle_only`; baseline = 5 fresh stacks x the full matrix plus 335 in-stack
`middle_only` runs. `t_cut` and `left_only` are DETERMINISTIC: the response, every Solution trajectory and the cut
targets are bit-identical over all baseline runs and over all candidate runs (WP2: 3 stacks, WP3: 3 stacks + 2 stacks of
`plan_h_cut_from_state` with the current joint state). `middle_only` is NOT deterministic in the baseline itself:
KDL pose-goal IK of the approach/align stages varies run to run (largest pairwise spread 7.7e-5 rad = 8 um at the tool
tip; downstream Cartesian stages <= 1e-7 rad; also inside one process), and in ~7 % of the 07:49 runs (2-7 % on 09:20)
the full seam hits a wrist flip in the stroke and is shortened to 95 % (rarely 90 %). So an endpoint tolerance of 1e-6
is not even met baseline-vs-baseline; the gate (`tools/compare.py gate`) is: bit-identical to a baseline run, or (for
the non-deterministic case) the same outcome class (response + stage sequence), identical cut targets, joint
positions within 1e-4 rad (IK-goal stages) / 2e-7 rad (other stages), times within 1e-3 s and tool-tip stage endpoints
within 20 um of every baseline run of that class. WP2 and WP3 pass; shortening rates equal (07:49: 26/370 baseline,
26/370 WP3).

## Carton contact links and the opt-in blade collision link (2026-10-15)
Evidence: `scan_results/dual_arm_stack_20261015/blade_collision/`. The single-arm knife_c model has no collision geometry
for the 14.4 mm exposed blade (only the TCP frame `xeg32_tool_tip`); the dual model has the box `<prefix>knife_c_blade`.
The WP7 strict world check found the per-arm approach transits putting that box into `detected_carton` (up to 5.2 mm).

**URDF** (`igus_rebel_description`): `knife_c_tool.macro.xacro` now owns the ONE definition of the blade box
(`blade_collision:=false` default, `blade_tcp_xyz`); the dual macro (`rebel_tool.macro.xacro`) calls it with
`blade_collision="true"` and its own `tcp_xyz`. `igus_rebel_xeg32_wrapper.urdf.xacro` has the arg `blade_collision`
(default false; true needs `tool_variant:=knife_c`; anything but true/false fails). Default output is byte-identical to
before (all 8 combinations of tool variant x hardware protocol x attach_xeg32, and semantically identical to
`scan_results/dual_arm_stack_20261014/backup/single_arm_urdf_before.urdf` with `compare_single_arm_urdf.py`); the dual
URDF is byte-identical before/after. The wrapper passes the TCP xyz twice (joint literal + `blade_tcp_xyz`, a macro cannot
read its joint block): the xyz-only pivot patcher edits only the joint, so update both (guarded by
`test_wrapper_blade_tcp_equals_tcp_joint`).

**SRDF**: `igus_rebel2.srdf` is NOT changed (a disable entry for a missing link is noise and would add entries to the
default ACM). The pair `knife_c_blade`-`knife_c_handpiece` (same proof as the dual SRDF) is in
`config/knife_c_blade_disable_collisions.srdf.inc`; a launch that builds the URDF with `blade_collision:=true` inserts it
before `</robot>` (`per_arm_planning.launch.py` does). The single-arm `move_group.launch.py` has no such option.

**Executor**: read-only `carton_contact_links` (string list, default `[]` = `[tool_link]`). Every place that allowed
tool/carton contact now loops over the effective list: `addCartonContactScope` / carton scenes (`allowCartonContact`), the
five analytic-search ACM sites (`setCartonContactEntries`), and `PlannedSegment.tool_carton_contact_allowed` (true only if
all listed links are allowed). Free-space transits keep the blade checked. `validateSettings` refuses empty/duplicate
entries and `carton_collision_id`; `main()` refuses to start (FATAL, exit 2) when a listed link is not in the robot
model (only checked for a non-default list; not exercised in this session's reduced test scope).

**Per-arm launch**: `blade_collision` arg (per_arm / both_arms / dual_cut_system / dual_cut_full_mock; default FALSE for
now, because the installed executor must know the parameter first; with an old binary a blade link breaks every cut
plan). With true: URDF `blade_collision:=true`, SRDF fragment, `carton_contact_links [tool_link, knife_c_blade]` for both
arms (launch wins over YAML/overrides). Handpiece/holder need NO contact allowance: every left cut plans with them checked.

**Results (mock, domain 206)**: defaults regression (new binary vs the pre-change source rebuilt today, today's YAML; the
WP0 baseline pool is stale since the YAML moved): `compare.py gate` PASS, t_cut and left_only bit-identical, middle_only
within the known envelope. Left arm in the dual stack, blade on vs off: all 6 cases plan, joint trajectories of
t_cut/left_only identical, plan times unchanged (t_cut 10.8-11.8 s, left_only 10.4-10.8 s, middle_only 0.3 s).
WP7 strict (`world_penetration_tolerance_m=0`), blade on: 09:20 and 07:49 plan and execute A+B with 0 penetration, except
one 09:20 run (first after stack start) where left phase B refused: the OMPL approach transit ended with the blade box
grazing the carton (Computed path is not valid, states 128-134 of 138); 3 reruns of 09:20 and 07:49 passed.
