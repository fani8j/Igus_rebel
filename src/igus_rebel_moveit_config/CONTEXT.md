# Ubiquitous Language

## Snapshot-relative combined plan
A fixed canonical plan derived from one selected carton snapshot: left midpoint cut, collision-aware transition, complete middle seam, collision-aware transition, then right midpoint cut. Planning publishes to RViz and retains a plan only when every stage succeeds.

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
`debug_motion_execution_enabled`. `save_cut_debug_point` captures the full
observed TCP pose after teleop correction, persists the nominal/observed/delta
record, clears the stale plan, and retains the next stage: endpoint-only after the
start save, lift-only (corrected endpoint to hover clearance) after the end save.
Any out-of-order call is rejected; no stale plan survives a teleop/save step.

## Cut-local correction
A translation expressed in the detected cut's own basis (`along` the cut,
`across` the cut tangent, `normal` the surface normal) plus the orientation
delta in that basis. Applying a record to a new snapshot rotates the same
local components onto the newly detected basis, so the correction follows the
cut. Application is opt-in (`cut_debug_correction_enabled`, default `false`)
and fails closed on schema/cut/frame/tool mismatch or unbounded values.
