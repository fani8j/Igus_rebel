# Quick re-check of the recommended v5 design (`quick_recheck.sh`)

Purpose: after a small change (blade exposure, skid geometry, a camera pose, a gripper mass, a fixture offset) tell in ~10-20 min whether the recommended design (`scan_results/cell_v5_20261007/recommended.json`, rank 1) still passes. New files only; nothing existing was modified.

```
cd tools/cell_planner_check
cat > /tmp/change.json <<EOF
{"blade_exposed_mm": 12.0}
EOF
./quick_recheck.sh /tmp/change.json my_tag 20          # 20 python workers (nice 19), MoveIt on domains 187 (table) and 188 (stand), planning only
```
Result: `scan_results/planner_check_v5_20261007/quick/<tag>/report.txt` (also printed): PASS / FAIL per item. Env switches: `SKIP_MOVEIT=1`, `QR_ALL=1` (evaluate all offline points instead of stopping at the first failure), `QR_FAST=1` (skip `verify_skid.py`: no penetration / pad numbers).

## What it does
1. `quick_recheck.py prep`: work dir `quick/<tag>/` with a patched copy of `cad/dual_arm_tooling_v3` (code + `data/design.json` + `cad_overrides.json` + the study plans) and the change applied.
2. In parallel
   - `cad`: parametric CAD rebuild (`build_all.py` = ALL parts and collision meshes are regenerated, about 2 min, not only the affected ones: simpler and safe), `check_exports`, `verify` (interference, keep-out, finger clearance, bottle), `verify_skid` (pad-blade clearance, axial stop, flap gap, penetration over the study stroke samples, camera B line of sight), `stiffness_mass` (tip deflection, mass, moment). 8-10 min at 1 thread.
   - `offline`: the 27-point primary region (+-10 mm, +-5 deg) of BIG and SMALL, table and stand (108 point evaluations, one job each, `final3.region_job`), HARDEST POINTS FIRST (smallest baseline margin) and stop at the first failure. About 1-3 min at 20 workers. The change is passed to the study model through env variables (`V5_BLADE_EXP`, `V5_EXPOSURE`, `V5_PAD_R`, `V5_PAD_OFF`, `V5_MASS_GRIPPER`, `V5_MASS_FINGERS`) and the camera shifts through the design spec.
3. `moveitprep` + `moveit_run_v5.sh table 187 | stand 188` (two stacks in parallel, exact collision meshes of the work-dir CAD, scene-camera post, pad ACM allowance only during the cut strokes): the baseline jobs (same stroke / transit joint targets, same park pose) are replayed against the NEW meshes. The reachability of the strokes themselves is therefore decided by the offline step, the collision of the new geometry by MoveIt.
4. `report`: table with thresholds: clearance beyond padding >= 5 mm, joint margin >= 10 deg, pad-blade clearance (blade bent 2 mm) >= 2 mm, penetration before skid contact >= 3.5 mm, pad is the first contact, axial stop, flap gap 0-6 mm support, tip deflection <= 0.3 mm (10 N, PA12-CF), interference none, MoveIt n/n.

## Change keys (`change.json`)
| key | effect |
|---|---|
| `blade_exposed_mm` | blade exposure beyond the collet: CAD (blade, stub and housing positions) and study (collision model) |
| `skid_exposure_mm`, `pad_r_mm`, `pad_off_mm` | pad geometry (CAD pads / frame bores, penetration filter, pad-blade clearance) |
| `camA_shift_mm`, `camB_shift_mm` `[x,y,z]` | camera body and bracket foot shifted in link6 (CAD collision + study live-view model) |
| `mass_fixed_kg` `{gripper, fingers, handpiece, camA, camB}` | masses in the mass / moment report and the study moment |
| `fixture` `{BIG|SMALL: {dx_mm, dy_mm, dyaw_deg}}` | offline only (the MoveIt step is skipped because the stroke targets of the old fixture would be wrong) |
Gripper / finger GEOMETRY changes are not parametric (STEP / STL inputs): replace the files named in `cad/dual_arm_tooling_v3/tooling_params.py` and run the full `run_all.sh`.

## Limits
- Not a replacement for a full study run: the offline step evaluates the baseline fixture / park / holder only; holder or camera REDESIGN needs the pipeline (`scan_results/cell_v5_20261007/run_v5.sh`).
- A camera moved by the change is only checked for collision and line of sight (`verify_skid`), not re-optimised for live-view coverage.
- If the change is rejected by one of the two stacks starting (move_group not ready) the run aborts with a message in `quick/<tag>/run_*.log`.
