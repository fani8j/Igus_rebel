# v4 additions (2026-10-06, dual-arm tooling v2 validation)
New files only; the existing harness (`gen_cell.py`, `gen_cell_v3.py`, `cellcheck*.py`, `build_job*.py`) is unchanged.
| file | purpose |
|--|--|
| `make_design_v4.py` | design JSON (`designs/v4_<label>.json`) from `scan_results/cell_v4_20261006/recommended.json` + the CAD v2 collision directory (manifest) |
| `gen_cell_v4.py` | copy of `gen_cell_v3.py`; replaces the gripper / fingers / clamp / handpiece / camera links by the EXACT CAD v2 collision meshes (`cad/dual_arm_tooling_v2/collision/*.stl`, convex pieces in metres, link6 frame), adds `camera_b`, disables tool-vs-link6 contact pairs (CAD contact by design) |
| `stack_v4.sh` | `stack_v3.sh` with the process ledger in `scan_results/planner_check_v4_20261006/` (domain 187, localhost only, nice 19, records the process group) |
| `moveit_stage_v4.sh` | whole validation: per design build table + stand cells, jobs from the study plans (table plans on the table, stand plans on the stand, + v3-fixture jobs), start the stack, wait for "You can start planning now", run `cellcheck_v3.py`, stop the stack. `RANKS="1 2"`; never run two stacks on one domain at the same time |
Exact commands: `RANKS="1 2" ./moveit_stage_v4.sh` then `python3 scan_results/planner_check_v4_20261006/summarize_v4.py`.
