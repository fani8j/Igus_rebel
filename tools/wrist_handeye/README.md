# wrist_handeye - standalone eye-in-hand calibration of the arm-mounted Gemini 305

Solves `T_link6_wrist_camera_color_optical_frame` (X) for the ReBeL knife_c tool from a ChArUco table board seen by the wrist
camera at ~16-20 arm poses. Standalone: nothing in `src/` is imported except the read-only `targets.py` of
`rebel_dual_arm_calibration`, nothing in `src/` or any URDF is edited, no ROS node of the stack is modified.

Files: `wrist_handeye.py` (CLI), `handeye_math.py` (pure geometry + solver, unit-tested), `ros_io.py` (ROS plumbing),
`mock_stack.sh` (isolated mock stack), `mock_move.py` (mock-only helper to park the mock arm), `test/`.

## Usage

Always pass an explicit domain (the script refuses to start without `ROS_DOMAIN_ID`).

```bash
source /opt/ros/humble/setup.bash && source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=<n> ROS_LOCALHOST_ONLY=1          # real cell: the domain of the running stack
cd /home/farhan/igus_rebel_ros2/tools/wrist_handeye
python3 wrist_handeye.py                       # = --plan-only: generate + plan every move, NO motion
python3 wrist_handeye.py --execute --real      # real motion: typed MOVE, Enter before the first move, Enter per move
python3 wrist_handeye.py --execute --real --auto   # same, but only the first move asks for Enter
python3 wrist_handeye.py solve <run_dir>       # offline re-solve (no ROS), e.g. after editing samples
```

Options: `--board-pose x y z qx qy qz qw` (mock only), `--mock-no-capture` (mock only), `--n-poses 18`, `--scale 0.10` (max 0.25),
`--settle 1.5`, `--min-points 16`, `--max-joint-step-deg 90`, `--mount-xyz/--mount-rpy/--mount-yaml` (override the nominal mount;
a previous `result.yaml` works), `--no-seed-from-view`, `--publish-tf` (publish the result as a static TF, Ctrl-C to end),
`--out DIR` (default `scan_results/wrist_handeye_<timestamp>/`), `--targets`, `--group igus_rebel_arm`, `--slab-thickness`, `--stand-extra`.

### What the run does
1. **Startup checks.** `list_hardware_components`: `mock_components/GenericSystem` = mock, anything else = real.
   `--real` on mock -> refused. `--execute` on real without `--real` -> refused. Real motion additionally needs the typed `MOVE`
   (interactive terminal required). `--mock-no-capture` and `--board-pose` are refused on real hardware. Plan-only works on real
   hardware without `--real` (it never moves anything).
2. **Board** from the scene camera (5 frames, PnP, TF `base_link <- camera_color_optical_frame`) or `--board-pose` (mock). Printed.
3. **Safety scene** (`/apply_planning_scene`, every object `whe_*`, removed in `finally`, on SIGINT/SIGTERM, after errors; the removal is
   verified through `/get_planning_scene` and retried): `whe_table_slab`, `whe_board_stand`, `whe_pole_pad`, `whe_head_pad`
   (see below). Missing `camera_pole`/`camera_head`, or not a single box in base_link/world -> abort (fail closed). The arm must be at rest
   and its CURRENT state must be collision-free against the scene (otherwise abort with the contact pair; jog it clear first).
4. **Poses**: orbit about the board centre (tilt {0,15,25,35} deg, 8 azimuths, roll {-40,-20,0,20,40}, distance {0.30,0.36,0.42} m;
   >= 0.25 m enforced), predicted board corners must lie inside the image with 10 % margin, `link6 = cam * inv(X)`, `/compute_ik` with
   `avoid_collisions=true`; greedy max-min rotation selection, then ordering by joint distance, then IK is re-run along the chain seeded
   from the previous accepted pose. Rejections: > 90 deg change of any joint between consecutive poses (start included), |joint6 - start| > 3.14 rad.
   A table of rejection reasons is printed.
5. **Plan-only**: `/plan_kinematic_path` for every move from the previous planned end state (scale 0.10) + a return-to-start; per-move
   duration, max joint change, detour (max joint travel), collision-free (planner + `/check_state_validity` on the waypoints). `plan.json` saved.
6. **Execute**: per move plan (0.10) from the live joint state; rejected when the first point differs from the live state by > 0.01 rad, the path
   does not end at the goal, a joint detours (> max(1.5x, +0.35 rad) of the direct change), joint6 leaves +-3.14 of its start, or a waypoint is in collision.
   Execution is `ExecuteTrajectory`. Ctrl-C / SIGTERM / `q` cancels the active goal immediately and stops (no further motion, no return move).
   After a move: standstill (joint_states within 0.002 rad over 1.0 s), settle 1.5 s, 3 wrist frames, board detection (>= 16 points; frames must agree to 2 px).
   A pose without detection is skipped; 3 consecutive failures or any execution error stop the run. From 4 samples on an interim solve runs;
   if X moved > 1 cm or 3 deg the remaining poses get fresh IK. At the end the arm returns to the start joint state (same checks), unless the run was stopped.
7. **Solve** (also `solve <run_dir>`): cv2.calibrateHandEye TSAI/PARK/DANIILIDIS -> best by board-pose consistency -> `scipy.optimize.least_squares`
   (Huber, 1 px) jointly over X and T_base_board on the reprojection of all corners with the camera_info intrinsics/distortion. Reports reprojection RMS/max,
   board-pose consistency, leave-one-out board error, rotation span and axis diversity (degeneracy WARN), scene-camera cross-check.
   Gate: PASS if RMS <= 1.0 px, LOO (RMS board position error) <= 3 mm, >= 10 samples, span >= 30 deg and no degeneracy; else WARN (FAIL if RMS > 2 px,
   LOO > 6 mm, < 6 samples or span < 10 deg). Output: `result.yaml` (schema `wrist_handeye/v1`), `urdf_snippet.xml` (paste by hand), `solve_report.txt`,
   `samples.json`, `camera_info.json`, `board.json`, `run_info.json`, `images/`.

### Safety scene geometry
* `whe_table_slab`: horizontal box in base_link, top = highest board corner z + 0.05 m, 1.2 x 1.2 m centred on the board, footprint clipped (largest remaining area)
  so it stays >= 0.15 m from the base z axis and still covers the board; thickness 0.30 m.
* `whe_board_stand`: board corner bounding box + 0.05 m per side, from the slab top up by `--stand-extra` (0.03 m).
* `whe_pole_pad`, `whe_head_pad`: the existing `camera_pole` / `camera_head` (pose and size read from `/get_planning_scene`) inflated by 0.05 m per side.

### Nominal camera mount (ASSUMPTION, overridable)
`knife_c_wrist_camera` is a child of `igus_tool0` with identity origin; its `gemini_305_body.stl` is a plain box 42 x 42 x 23 mm rotated 18 deg about tool0 y.
The big faces have normals +-(0.309, 0, -0.951); the camera looks towards the knife (+z of tool0), so optical +z = (-0.309, 0, 0.951) and the optical centre is
the centre of that face (-0.0201, -0.092, -0.0438) m in igus_tool0. The image axes cannot be read from a square box: image x = tool0 +y (guess; the roll of the
mount is therefore unknown). `igus_tool0 = link6 * T(0,0,0.0527) * Rz(pi)`. Result: xyz (0.020, 0.092, 0.009) in link6. If the board is visible in the wrist
image at start, X is re-estimated from that single view (scene-camera board pose + live TF + PnP) and used instead when within 150 mm of the nominal.

## Tests
```bash
cd /home/farhan/igus_rebel_ros2 && python3 -m pytest -p no:anyio tools/wrist_handeye/test
```
Covers: recovery of a known X (noise 0.2 px + an outlier sample) within 1 mm / 0.1 deg, degenerate (single-axis) rotations -> WARN, gate logic, orbit geometry
(board centre on the optical axis, distances, tilt), in-view prediction, slab (horizontal, top = max corner + 0.05, clipped >= 0.15 m from the base axis,
fail when impossible), board stand, pad inflation, nominal mount vs the STL, IK-chain step limits.

## Mock end-to-end
```bash
MOCK_DOMAIN=199 setsid bash tools/wrist_handeye/mock_stack.sh <logdir>     # records its PGID in <logdir>/stack.pgid
# park the mock arm somewhere clear of the camera head pad first (home collides with whe_head_pad):
ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 python3 tools/wrist_handeye/mock_move.py 5 0.2326 1.155 1.298 -0.4718 0.5845 1.39
ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 python3 tools/wrist_handeye/wrist_handeye.py --plan-only --board-pose 0.514 -0.148 -0.251 0 0 0 1
echo | ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 python3 tools/wrist_handeye/wrist_handeye.py --execute --auto --mock-no-capture --board-pose 0.514 -0.148 -0.251 0 0 0 1
kill -INT -- -$(cat <logdir>/stack.pgid)
```
