# cell_planner_check - planner-level validation of the dual-arm cut cell (MoveIt 2 Humble, planning only)

No `package.xml` here, so colcon ignores this directory.  Nothing is executed: no ros2_control, no controllers, `allow_trajectory_execution: false`.
Every command below sets `ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1` itself (domain 11 is the real robot; `cellcheck.py` refuses to run on any other domain).

## Frames
Planning frame `world` == the offline studies' W frame (Z-up): x forward (CAD +Z), y = CAD +X (arm1 at +y), z up (CAD +Y); origin at the arm-base mounting
plane, mid-way between the arms.  `arm1` = study arm 0 (y = +s/2), `arm2` = study arm 1 (y = -s/2), both base frames = world translated (identity rotation).
CAD -> world: `x_w = Z_cad - 0.854511, y_w = X_cad + 0.653868, z_w = Y_cad + 0.241696` (`cellgeom.py`).  Carton top at z = -h; table plane at the carton bottom.

## Files
| file | purpose |
|--|--|
| `make_designs.py` | design JSONs (`designs/*.json`, tool geometry in link6 frame + study sphere-proxy block) from `holder_camera_fingers_20261002/recommended.json`; `baseline.json` = old single knife |
| `gen_cell.py` | generator: dual-arm URDF + SRDF + kinematics/joint_limits/ompl/pilz yaml for a design and layout (`--s`), `--geometry real|proxy`, `--structure cad|study|none`, `--fingers ...`, `--matrix` |
| `sample_collisions.py` | own collision-matrix sampler (uniform random dual-arm states via `/check_state_validity`) -> `collision_matrix.json`, used by `gen_cell.py --matrix` for the 'Never' pairs |
| `cell_planning.launch.py` | robot_state_publisher + joint_state_publisher (neutral) + move_group via `MoveItConfigsBuilder` with explicit file paths |
| `stack.sh` | start / stop / status of the stack on domain 187, records the process group, stop kills only that group |
| `moveit_client.py`, `cellcheck.py` | standard-service client and the validation harness (IK, Cartesian, transits, scene/ACM, validity, time-aligned pair check, cycle time) |
| `study_eval.py`, `build_job.py` | job builder: re-uses the study's python code (read-only import) -> jobs/*.json (v1 = holder_camera_fingers_20261002; v2 adapter stub for `cell_v2_20261003`) |
| `sanity_check.py` | FK / frame / structure / carton convention checks vs the study |
| `summarize.py` | markdown summary of result JSONs |
| `run_job.sh` | run a job in the background against the running stack |
| `experiment.sh`, `experiment_multi.sh`, `run_on_build.sh`, `final_on_build.sh`, `audit_on_build.sh`, `diag_on_build.sh`, `run_final.sh`, `run_all*.sh` | batch drivers (generate cell [+ collision matrix] -> stack -> sanity -> jobs -> summaries -> audit -> stop); every stack is started/stopped through `stack.sh` and logged in `scan_results/planner_check_20261003/process_ledger.txt` |
| `path_audit.py` | asks MoveIt's collision checker about the STUDY's own joint paths (strokes + dense interpolation, paired samples of concurrent arms), no planning: isolates geometry differences |
| `ik_audit.py` | KDL IK from study / neutral / random seeds for every study stroke pose (collision checking off) |
| `analyze.py`, `collect_final.py`, `summarize.py` | cross-run analysis (study vs real geometry vs study-sphere proxies, planner stats, cycle at study speeds), headline-number JSON, per-run markdown |
| `diag_look.py` | diagnose a failing transit (goal validity, straight-line joint path, planners, partner states) |

## Exact commands
```bash
cd /home/farhan/igus_rebel_ros2/tools/cell_planner_check
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source /home/farhan/igus_rebel_ros2/install/setup.bash && export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1

python3 make_designs.py                                                         # designs/*.json
# 1) pre-matrix build, sample collisions, final build with the matrix
python3 gen_cell.py --design designs/h000n.0.json --s 0.355 --out build/h000n.0_s0355_pre
./stack.sh start $PWD/build/h000n.0_s0355_pre $PWD/build/stackP.pgid ; sleep 20
python3 sample_collisions.py --build build/h000n.0_s0355_pre --n 30000 --out build/h000n.0_s0355_pre/collision_matrix.json
./stack.sh stop $PWD/build/stackP.pgid
python3 gen_cell.py --design designs/h000n.0.json --s 0.355 --out build/h000n.0_s0355 --matrix build/h000n.0_s0355_pre/collision_matrix.json
# 2) the job (study evaluation is cached under scan_results/planner_check_20261003/study_cache)
python3 build_job.py v1 --design h000n.0 --layout 0.0,0.40,0.355 --boxes BIG370,SMALL --yaws 0:180:15 --out jobs/h000n.0_v1.json
# 3) stack + sanity + run
./stack.sh start $PWD/build/h000n.0_s0355 $PWD/build/stackA.pgid ; sleep 20
python3 sanity_check.py --build build/h000n.0_s0355 --cases jobs/h000n.0_v1_fkcases.json --out /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/sanity_h000n.0.json
python3 cellcheck.py --job jobs/h000n.0_v1.json --build build/h000n.0_s0355 --out /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/h000n.0
python3 summarize.py /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/h000n.0/result_h000n.0_v1.json --out /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/h000n.0/summary.md
./stack.sh stop $PWD/build/stackA.pgid
```
Check the stack: `ros2 node list` and `ros2 service list` (domain 187) show `/move_group`, `/compute_ik`, ...; there is no `ros2_control_node` / `controller_manager`.

## Job JSON (schema `cell_planner_check/job/v1`)
`layout{h,d,s}`, `scaling{names, transit[[vel,acc],..], contact[vel,acc]}` (defaults exec 0.20/0.10 and fast 1.0/0.1 for transits, 0.1376/0.0954 for contact strokes),
`placements[]`: `carton{size,center,quat}` (world), `study{feasible, alloc, tasks{id:{margin,q}}}`, `tasks{id: {arm, steps[]}}`, `phases[[ids],..]` (tasks in one phase run concurrently).
Steps: `transit` (candidates with `pose` + `seeds`, or an explicit `q_goal`) and `cartesian` (`waypoints`, `contact`, optional `alternatives`).
Poses are `armN_knife_tip` poses (xyz, quaternion xyzw) in `world`.  A v2 adapter only has to emit this schema (`build_job.py v2`).

## Model decisions (also in the report)
* Transit = joint-space goal from `/compute_ik` (seeded with the study joints, collision-aware) then Pilz PTP (joint interpolation, like the executor's first planner) and OMPL RRTConnect as fallback; durations come from the response's time parameterization.
* Cut strokes: `/compute_cartesian_path`, `avoid_collisions=true`, step 5 mm, joint-jump filter 0.25 rad, fraction must be 1.0; the `armN_knife_tip`/carton ACM allowance is switched on (via `/apply_planning_scene`) only around enter / stroke / retreat.  The knife tip has no collision geometry (last 20 mm of blade exempt), so the allowance is a formal no-op.
* While one arm's task is planned the partner is modelled as parked (neutral, upright) in concurrent phases (the study's assumption); the true interaction is the time-aligned `/check_state_validity` pair check; a colliding pair is serialised (replanned) and costs the sum of the durations.  A finished arm parks (neutral) during the partner's next phase.

## Added during the validation (see the report)
* `gen_cell.py --geometry proxy` rebuilds the offline study's own collision model inside MoveIt (spheres for upper arm / forearm / wrist / gripper / handpiece / post / clamp / fingers / bracket, camera OBB padded 3 mm, study structure box, sphere radii +2.5 mm so the 5 mm arm-arm clearance holds) -- the control experiment that separates geometry differences from planner / IK / timing effects.
* v1 jobs use the study's PAIR-CONSISTENT path combination (the combination that made the study's concurrent pair clear), not just the best-margin path of each task (`jobs/old_best_margin/` keeps the first version).
* `cellcheck.py`: joint-branch-flip check on Cartesian strokes (the Humble `revolute_jump_threshold` did not catch a 3.3 rad wrist flip), 1 mm retry, step-synchronous partner replanning (pass 2), park of a finished arm, serialisation by replanning, `dwell_s`.
* `build_job.py v2 --design E5 [--mode single]` is the adapter for `scan_results/cell_v2_20261003/recommended.json` (look pose before every cut, 30 mm approach / retreat, gamma <= 45 side cuts, 70 mm stroke); `--mode single` = one arm does all three cuts serially.
* Results of the final batch: `scan_results/planner_check_20261003/final/` (`final_numbers.json`, `analysis_*.md`, per-run `summary_*.md`, `result_*.json`, `audit_*.json`), earlier exploratory runs in `exploratory_runs/`.
