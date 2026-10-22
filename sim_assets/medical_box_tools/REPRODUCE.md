# Reproducing the shelf-pick results (Isaac Sim 5.1 + MoveIt mock)

All commands run from `sim_assets/medical_box_tools/` unless noted. Grasp options, targets and pass checks: `GRASP.md`. Simulation only; nothing here talks to a real robot.

**Safety:** every terminal sets an unused `ROS_DOMAIN_ID` and `ROS_LOCALHOST_ONLY=1` (the real robot uses domain 11). The Isaac
tests force domain 170 themselves (`tests/isaac_common.py`) and enable no ROS bridge. Run heavy jobs with `nice -n 19`, at most 4 workers.
Stop jobs by their own PID / process group, never with `pkill -f` or `killall`.

## 0. What has to exist on this machine

| Item | Where | Used by |
|---|---|---|
| Isaac Sim 5.1.0 | `/home/farhan/isaac-sim/` (`python.sh`, `isaac-sim.sh`) | all `tests/*.py` |
| D200 + dual-arm USD | `usd/d200/` (in this repo, 15 MB; the operator's `d200_with_dual_arm` zip, unchanged) | `usd/d200_dual_arm_medical.usda` |
| Stocked cabinet USD | `sim_assets/medical_boxes/usd/medicine_cabinet_filled.usda` (in this repo) | both cabinet scenes |
| Robot description + MoveIt config | main repo `/home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_{description,moveit_config}` (commit `1e6dd4b`) | URDF for kinematics, MoveIt check |
| System Python 3.10 with PyKDL, urdf_parser_py, trimesh, scipy, numpy | `/usr/bin/python3` (ROS Humble) | `tools/*.py` |
| ffmpeg | `/usr/bin/ffmpeg` | `--record` videos |

### Scratch build of the medicine-grasp packages and the URDF (once)

Built out of tree so the main repo's `build/` and `install/` are not touched:

```bash
source /opt/ros/humble/setup.bash
source /home/farhan/igus_rebel_ros2/install/setup.bash        # igus_rebel_description, rebel_dual_arm_stand_description
cd /home/farhan/.cache_medicine_scratch
colcon build --symlink-install --parallel-workers 4 \
  --base-paths /home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_description \
               /home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_moveit_config \
  --build-base build --install-base install
source install/setup.bash
xacro /home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_description/urdf/rebel_medicine_grasp.urdf.xacro \
  include_ros2_control:=false > medicine.urdf
```

`tools/robot_kin.py` reads `/home/farhan/.cache_medicine_scratch/medicine.urdf` (constant `URDF_PATH`). Rebuild it after any URDF change.

## 1. Test 1: tools only (kinematic carriers, no arms)

```bash
PY=/home/farhan/isaac-sim/python.sh
nice -n 19 $PY tests/test_cabinet_grasp.py --mode front   --targets all --shots            # baseline, all targets
nice -n 19 $PY tests/test_cabinet_grasp.py --mode topdown --targets all --shots
nice -n 19 $PY tests/test_cabinet_grasp.py --mode front   --targets T5 --variants vac35,squeeze,lat2,tilt3   # sensitivity checks
python3 tests/summarize_cabinet.py
```

Add `--record` for MP4s (`--video-dir` to choose where; keep videos on the external drive).
Results: `tests/results/cabinet_grasp_<mode>.json`. Each full mode takes about 25 min.
`tests/run_all.sh` runs these plus the USD load / tool / calibration tests (`QUICK=1` skips the cabinet runs).

## 2. Robot placement and joint plans (plain Python, no Isaac)

```bash
nice -n 19 python3 tools/reach_maps.py                                          # IK reach maps -> data/reach_maps.npz (only after URDF changes)
nice -n 19 python3 tools/parking_study.py --modes front --workers 3 --out amr_parking_front.json
python3 tools/export_plan.py --parking tests/results/amr_parking_front.json --mode front   # -> data/plans/front_<T>.json
```

`parking_study.py` picks the robot position, lift height and slider positions per target, with clearance checks.
`--modes topdown` reproduces the "no feasible pose" result; `tools/topdown_diag.py` gives the per-pose diagnosis.
The committed `data/plans/*.json` are the plans behind the report, so this step can be skipped.

## 3. Test 2: full robot in Isaac Sim

```bash
nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp_robot.py --mode front \
  --targets T3,T4,T5,T6,T7,T8,E1 --record          # --header adds the old top info bar to the videos
python3 tests/summarize_robot.py                   # -> tests/results/cabinet_grasp_robot_summary.md
python3 tests/make_highlight_reel_robot.py         # highlight reel + stills
```

Results: `tests/results/cabinet_grasp_robot_front.json`. Videos (3/4, side, head camera, tiled) go to
`/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/robot_20261015/` by default.
Physics is the same with and without `--record`.

## 4. Shelves 1 and 2: closest attempt

```bash
python3 tools/closest_attempt.py --targets T1,T2          # -> tests/results/unreachable_closest.json, data/plans/closest_T{1,2}.json
python3 tools/closest_attempt.py --targets T1,T2 --whatif # how much more lift travel would be needed
# the two videos can run in parallel, each on its own domain:
CLOSEST_DOMAIN=174 nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_closest_attempt.py --target T1 --record
CLOSEST_DOMAIN=175 nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_closest_attempt.py --target T2 --record
```

## 5. MoveIt check (mock hardware, plan only)

Terminal A, MoveIt:

```bash
export ROS_DOMAIN_ID=172 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/farhan/igus_rebel_ros2/install/setup.bash
source /home/farhan/.cache_medicine_scratch/install/setup.bash
nice -n 19 ros2 launch rebel_medicine_grasp_moveit_config medicine_grasp.launch.py \
  hardware_protocol:=mock_hardware add_medicine_shelf:=false      # add use_rviz:=true to watch
```

Terminal B, same four setup lines, then:

```bash
ros2 control list_hardware_components      # must show mock_components/GenericSystem
python3 tools/moveit_confirm.py --parking tests/results/amr_parking_front.json --mode front \
  --targets T3,T8 --out tests/results/moveit_confirm_front.json
```

The script loads the cabinet into the planning scene and plans every transit (OMPL) and straight move
(`compute_cartesian_path`, fraction must be 1.0). Trajectory execution is off by default in this launch.

## 6. E2 (carton at the wall) study

```bash
python3 tools/e2_study/e2_first.py    # where the straight approach first hits the wall / trim
python3 tools/e2_study/e2_yaw.py      # tool variants and the minimum approach yaw
```

Write-up: `tests/results/e2_longer_tool_study.md`. The scripts have the box-3d worktree path hard-coded.

## 7. Pump simulation and report

```bash
cd ../../cad/medical_box_tooling_v1
python3 vacuum_sim/pump_sim.py                 # -> vacuum_sim/data/, vacuum_sim/renders/
cd report && python3 pack_models.py            # rebuilds media/models.js from the tool STLs
V="/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/robot_20261015"
cp "$V"/{highlights_robot,front_T5_tiled,front_T1_closest_tiled}.mp4 media/   # videos are not in git
xdg-open medicine_shelf_pick_report.html
```

## Watching a test live (`--gui`)

`tests/test_cabinet_grasp_robot.py` and `tests/test_closest_attempt.py` take `--gui`: the Isaac Sim window opens, the camera starts at
the same 3/4 view as the videos, the run plays at real time and the step captions are printed in the terminal.

```bash
nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp_robot.py --mode front --targets T5 --gui
nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_closest_attempt.py --target T1 --gui
```

- The window appears after about 15 s and the run starts by itself once the scene is loaded. A T5 run takes about 4 min in total.
- Orbit with Alt + left mouse, pan with the middle mouse, zoom with the wheel (or right mouse + WASD to fly). Do not press Stop/Play in the toolbar while a test runs.
- At the end the final pose stays on screen until you close the window (`--gui-close` exits by itself instead).
- With several targets, each one's final pose is shown for `--gui-hold` seconds (default 3) before the next starts.
- `--gui-speed 0.5` plays at half speed. Playback cannot be faster than the physics: on a busy CPU it runs slower than real time.
- The physics is the same as headless (the window is only redrawn between physics steps). Results go to separate files
  (`cabinet_grasp_robot_<mode>_gui.json`, `unreachable_closest_run_<T>_gui.json`), so the headless results are not overwritten.
- `--gui` cannot be combined with `--record`.

## Looking at a scene in the Isaac Sim GUI

```bash
export ROS_DOMAIN_ID=170 ROS_LOCALHOST_ONLY=1
/home/farhan/isaac-sim/isaac-sim.sh
# File > Open: sim_assets/medical_box_tools/usd/cabinet_robot_test_scene.usda (full robot)
#           or usd/cabinet_grasp_test_scene.usda (tools only)
```

This is for looking only. The tests move the robot to each target's position at run time and apply extra physics
settings (`apply_d200_physics_settings()`), so pressing Play in the GUI does not reproduce a test run.
