# Carton grasp commands (Isaac Sim 5.1)

How to run a carton grasp in the stocked medicine cabinet, which cartons you can pick, what the options do and how to read the result.
Run everything from `sim_assets/medical_box_tools/`. Setup (Isaac Sim, external drive, URDF build) is in `REPRODUCE.md`.

The tests force `ROS_DOMAIN_ID=170`, `ROS_LOCALHOST_ONLY=1` and start no ROS bridge, so they cannot reach the real robot.
Always run them with `nice -n 19`, and stop a run by its own PID, never with `pkill -f` / `killall`.

## The grasp sequence

1. **Press:** the left arm's suction tool presses the 3 pads on the carton spine (pad buffers compress, the carton is pushed back about 4.5 mm).
2. **Vacuum on:** 0.15 s build-up, then the Surface Gripper closes (−50 kPa).
3. **Pull:** the suction tool pulls the carton 100 mm out of the row at 50 mm/s.
4. **Gripper in:** front mode: suction off, the suction arm retracts and goes home; the right arm's gripper comes in horizontally, fingers pre-closed to carton thickness + 4 mm, and clamps (24.3 N per finger).
   Top-down mode: the gripper comes down from above onto the pulled-out part instead.
5. **Extract:** the gripper pulls the carton fully out of the cabinet.
6. **Lift:** the gripper lifts it 50 mm.

Front mode is the one that works on the real robot geometry; top-down is only kept for the tools-only test.

## Targets

| Target | Shelf | Carton (h × l × t mm) | Position in row | Full robot, front |
|---|---|---|---|---|
| T1 | 1 | ultimaster_nagomi (250 × 295 × 16) | middle | out of reach (closest attempt only) |
| T2 | 2 | accuforce (220 × 250 × 16) | middle | out of reach (closest attempt only) |
| T3 | 3 | agent_monorail (230 × 250 × 15) | middle | pass |
| T4 | 3 | conqueror_nc_pro (230 × 250 × 12) | middle | pass |
| T5 | 4 | angiolite (240 × 250 × 16) | middle | pass (reference target) |
| T6 | 5 | agent_monorail (230 × 250 × 15) | middle | pass |
| T7 | 6 | accuforce (220 × 250 × 16) | middle | pass |
| T8 | 6 | conqueror_nc_pro (230 × 250 × 12) | middle | pass |
| E1 | 3 | agent_monorail (230 × 250 × 15) | last of its group, next to a different product | pass |
| E2 | 6 | angiolite (240 × 250 × 16) | first, next to the cabinet side wall | fails (tool hits the wall), no plan |

Targets are defined in `TARGETS` / `BOX` at the top of `tests/test_cabinet_grasp.py`. `--targets all` runs every target in the table.

## 1. Tools only (no arms)

The two tools move on their own along the ideal path ("kinematic carriers"), so this tests the grasp itself, not whether the robot can reach.

```bash
PY=/home/farhan/isaac-sim/python.sh
nice -n 19 $PY tests/test_cabinet_grasp.py --mode front --targets T5                    # one carton
nice -n 19 $PY tests/test_cabinet_grasp.py --mode front --targets all --shots           # all targets + a still of each
nice -n 19 $PY tests/test_cabinet_grasp.py --mode front --targets T5 --variants vac35,squeeze,lat2,tilt3 --out cabinet_grasp_front_sensitivity.json
```

| Option | Meaning |
|---|---|
| `--mode front\|topdown` | grasp direction of the gripper (default `topdown`, so pass `--mode front`) |
| `--targets T5,T3` / `all` | which cartons |
| `--variants ...` | extra runs, always on T5, each with one change: `vac35` vacuum −35 kPa instead of −50; `squeeze` row packed with no gaps; `lat2` suction tool 2 mm off-centre; `tilt3` suction tool turned 3° |
| `--shots` | save a still per run (external drive, `sim_assets_ext/cabinet_grasp_shots/`) |
| `--record` | MP4s (3/4 view + side view) to `--video-dir` (default `tests/results/videos/`; keep videos on the external drive) |
| `--out NAME.json` | results file name in `tests/results/` (default `cabinet_grasp_<mode>.json`) |

## 2. Full robot (arms, lift and tools)

The robot is placed in front of the cabinet and both arms follow pre-computed joint plans (`data/plans/front_<T>.json`, made by
`tools/export_plan.py`; see `REPRODUCE.md` section 2). Targets without a plan (T1, T2, E2) are skipped.

```bash
nice -n 19 $PY tests/test_cabinet_grasp_robot.py --mode front --targets T5 --gui          # watch it live in the Isaac window
nice -n 19 $PY tests/test_cabinet_grasp_robot.py --mode front --targets T5 --out t5_check.json
nice -n 19 $PY tests/test_cabinet_grasp_robot.py --mode front --targets T3,T4,T5,T6,T7,T8,E1 --record    # all reachable targets, with videos
python3 tests/summarize_robot.py                                                          # tables -> tests/results/cabinet_grasp_robot_summary.md
```

| Option | Meaning |
|---|---|
| `--mode front` | the only mode with plans |
| `--targets T5,T3` / `all` | which cartons (default `all`; ones without a plan are skipped) |
| `--record` | MP4s of each run: 3/4, side, head camera and a tiled view, to `--video-dir` (default the external drive, `scratch_usd/videos/robot_20261015/`) |
| `--header` | with `--record`: also draw the top info bar (time, lift, ...) |
| `--gui` | show the run live in the Isaac Sim window; see `REPRODUCE.md` "Watching a test live" (`--gui-speed`, `--gui-hold`, `--gui-close`) |
| `--yaw180` | E2 study variant: suction tool turned 180° (needs `front_E2_yaw180.json`, not committed) |
| `--out NAME.json` | results file name (default `cabinet_grasp_robot_<mode>.json`, or `..._gui.json` with `--gui`) |

## 3. Closest attempt (shelves 1 and 2)

Shows how close the arms get to T1 / T2 before a joint limit or collision stops them; a red bar marks the missing distance. No vacuum, no grasp.

```bash
nice -n 19 $PY tests/test_closest_attempt.py --target T1 --gui
CLOSEST_DOMAIN=174 nice -n 19 $PY tests/test_closest_attempt.py --target T1 --record    # T1 and T2 can run in parallel with different domains
```

Options: `--target T1|T2`, `--hold` seconds at the closest pose (default 3), `--record`, `--video-dir`, `--gui`, `--gui-speed`, `--gui-close`.

## Results

**Important:** a run writes the whole results file again with only the targets of that run. Running `--targets T5` without `--out`
replaces the committed all-target `cabinet_grasp_robot_front.json` with a T5-only file. Use `--out` for one-off checks
(or `git checkout` the file afterwards).

The terminal prints one block per target:

```
RESULT {"target": "T5", ..., "success": true}
    1_press OK {"compression_mm": [...], "carton_pushed_back_mm": 4.469, ...}
    2_vacuum OK {"status": "Closed", "gripped": [...]}
    3_pull OK {"carton_net_disp_from_rest_mm": 99.878, "monitor_max_pad_tension_N": 0.347, ...}
    ...
    clearance mm {"SX.neighbours": 6.6, "GR.shelf_above": 96.1, ...}
```

A target passes when every step is OK and there is no unexpected contact (tool or arm touching the cabinet or a neighbouring carton):

| Step | Pass check |
|---|---|
| `1_press` | pad travel past contact 5 ± 1.5 mm, carton pushed back 0–6 mm |
| `2_vacuum` | Surface Gripper closed on the target carton only |
| `3_pull` | carton pulled 100 ± 3 mm, still held, neighbours moved < 3 mm |
| `5_release_park` | suction released, carton moved < 2 mm when it let go |
| `4a_approach` | gripper approach moves the carton < 2 mm |
| `4b_clamp` | finger gap within 1 mm of the carton thickness |
| `6_extract` | carton travels with the gripper (slip < 2 mm), rises < 2 mm, neighbours moved < 3 mm and tilted < 5° |
| `7_lift50` | carton rises 50 ± 2 mm, slip < 2 mm, rotation < 5° |

Clearance keys: `SX` suction tool, `GR` gripper, `ARM` arm links; each against the shelf above/below, walls/trim/back, neighbours and the other tool.
The full record per target (step values, clearances by phase, joint tracking errors and efforts, contacts) is in the JSON.

Suction release in the simulation is a software check set to the calculated pull-off force (vacuum × pad area, about 5.8 N per pad at −50 kPa).
The pads carry well under 1 N in these runs, so the tests check reach, clearance and handling, not the real holding force.
