# Unknown (generic) boxes in the dual-arm cell — plan (2026-10-13)

Trigger: live capture `frame_000001_20261013_060825_145449` (new box, top 391 x 276 mm, clear centred tape, top
0.72 m from the camera, ~970 mm above the floor on a stand). SAM3 top + seam were good (gap + dark line confirm the
midline within 1.7 mm, edge support ok on all 4 edges), but:

1. `cut_blocked(height_unavailable)`: generic profile + `hidden_stand` -> no measured height -> blocked.
   The stand height is NOT constant between sessions (floor-to-top 600-680 mm on 10-12, ~865 mm on 10-13 01:05,
   ~970 mm now; floor RANSAC fragments disagree by 20-30 mm), so "floor distance - stand height" is not usable.
2. `depth_no_top` / `no_independent_depth_check` (hidden by `--allow-needs-review`): the box top (0.72 m, 154 k
   inliers) is now the LARGEST plane; the floor (1.67 m) is split into fragments (90 k + 58 k). The hidden_stand rule
   "support = largest plane" (depth_top.estimate_top, pipeline.estimate_table_plane, pipeline plane_choice "dominant")
   takes the box top as the floor -> no parallel top above it.

What the cut uses: the H-cut executor reads only the top face + seam from the snapshot; its collision box height is its
own `carton_height_m` parameter. The detected height feeds only profile identification, the corner fold (generic: not
applied without a height) and the side-tape seed (needs a height; stays blocked).

## Steps (scene-gated: `table_cell_v1` must stay bit-identical; dual_arm_v1 only)

| # | Change | Validation |
|---|---|---|
| G1 | hidden_stand support = the FARTHEST large plane family: group mutually parallel planes (parallel_tol_deg), merge coplanar fragments (offset within `support_merge_m`, e.g. 0.05 m) summing inliers, keep groups with >= table_min_inliers, pick the group farthest from the camera along its normal (the floor is below everything). Same rule in depth_top.estimate_top, pipeline.estimate_table_plane and the independent-check plane choice. | new capture gets a depth top + independent check confirmed; the 20 dual-arm captures + live 20261013_010553: same allowed set, snapshots identical or explained; 431 replay identical |
| G2 | scene option `generic_cut_without_height` (dual_arm_v1: true): a generic box with `height_unavailable == support_hidden` is not cut-blocked for the height IF the independent depth check confirmed the top (independent_top_check.confirmed) and the edge-support check found no unsupported edge; the record keeps `height_unavailable` as a flag. Side-tape seed stays blocked. | new capture -> cut allowed; a generic record without depth confirmation stays blocked (unit test) |

Not done: guessing a height, a fixed stand height, publishing the observed gap for generic boxes (either_confirm stays:
a gap off the midline by > the confirm gate stops safely instead of cutting). Optional later: a profile per box once the
user gives L x W x H.

## Implementation result (2026-10-13)

G1 (with advisor changes): `depth_top.pick_hidden_support` is the single helper used by `identify_table_top` (estimate_top)
and `pipeline.estimate_table_plane` (height, independent check, trim plane choice all read that table). Support candidates =
camera-facing planes whose normal is within `support_gravity_tol_deg` (20) of base_link +z (`depth_top.gravity_cam`, set by
`live/detector.py` from the hand-eye calibration; only for hidden_stand scenes), grouped by parallel + offset <= `support_merge_m`
(0.08), inliers summed, groups >= table_min_inliers, farthest wins, support = largest member (no refit). Fail closed with a
calibration; no calibration -> old largest-plane rule tagged `support_gravity_unchecked`. The floor-footprint merge uses
`support_merge_m`. visible_table path unchanged (431 replay bit-identical).
G2: scene key `live.generic_cut_without_height` (dual_arm_v1 true, table_cell_v1 false), `Scene.generic_cut_without_height`,
`evaluate_cut_permission(..., generic_cut_without_height=)`. Waiver needs: generic profile + support_hidden, independent check
confirmed (not circular), `box_profile.support_clearance_mm` > 0 and no support_gravity_unchecked, all 4 edges ok with
valid_frac >= 0.8 and none unsupported. Flag `height_waived(support_hidden)`. Side-tape seed unchanged (blocked).
Recorded: `box_profile.support_clearance_mm` (top-to-floor along the floor normal, upper bound of the box height).
Results: new capture accepted with and without --allow-needs-review; independent check confirmed (iou 0.974), support = floor
d=1.667 m (two fragments merged, 147.7 k inliers), support_clearance 967 mm, cut allowed with the flag. Remaining: estimate_top
still reports depth_no_top there (top rejected by the floor-footprint guard, margin -520 mm: the visible floor lies to one side
of the box); non-gating because the independent check confirms. 20 dual-arm + live 010553: same allowed/rejected sets; planner
snapshots equal except paths; record diagnostics differ on 3 captures (10, 13, 19; depth top now found / reasons reworded) and the
live 010553 quad differs <= 5.5 px (truncation recovered via sam3_alternate instead of depth_grown). 431 replay: 0 changes.
Tests: 905 passed, 13 skipped.
Review (advisor-astra, diff): "status ok + nothing unsupported" also passes a lid that continues 15 mm past both seam ends
(edge_support only flags >= 20 mm). Fixed: the waiver also needs every edge's coplanar extent <= 8 mm and continue share
<= 0.2 (`GENERIC_EDGE_MAX_EXTENT_MM` / `_CONTINUE_FRAC`; good dual-arm tops <= 5.8 mm / 0.1, the new box <= 2.5 mm / 0.0).
