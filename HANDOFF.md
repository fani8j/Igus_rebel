# Handoff — H-cut planning and mock verification

## 1. Goal

Build a snapshot-relative, plan-only-by-default T/H-cut workflow for the igus ReBeL with XEG-32 under ROS 2 Humble. The current problem is that independently sampled pose-goal IK can wind the wrist during free-space transitions, producing visually odd rotations and long trajectories; the saved taught left-cut pose also predates a TCP change. Preserve the detected cut geometry, collision checking, full Cartesian completion, and explicit execution gates while improving planning consistency. No physical cutting result has been established.

## 2. Current state

- `src/igus_rebel_moveit_config/src/box_h_cut_executor.cpp` builds and retains MTC plans for `t_cut`, `t_cut_handoff`, other cut patterns, and staged per-cut debugging. The left learned-orientation search uses collision-aware IK with the detected carton and external camera pole; pose application remains cut-basis-relative. Free-space transit uses a `MultiPlanner` that tries joint interpolation before the existing pipeline planner. `validateFreeSpaceTransitWrist()` sums absolute joint6 movement over consecutive waypoints of each free-space transit and rejects a solution exceeding pi radians before publication/retention/execution.
- `src/igus_rebel_moveit_config/config/box_h_cut_executor.yaml`: separate transit velocity/acceleration scalings both default to `0.05`; contact stroke velocity/acceleration also remain `0.05`, Cartesian minimum fraction remains `1.0` in C++, and pre-cut dwell is `1.0 s` per segment. `execution_enabled`, `debug_motion_execution_enabled`, `cut_debug_correction_enabled`, and `debug_allow_stale_snapshot` are `false`. Older left and middle correction paths are configured but not applied with the opt-in disabled.
- `src/igus_rebel_moveit_config/launch/move_group.launch.py` starts the executor, MoveIt, camera-pole collision publisher, and optionally RViz. Native Humble build `source /opt/ros/humble/setup.bash && colcon build --packages-select igus_rebel_moveit_config` passed after the planner changes (only a Boost bind deprecation message on the earlier compiling build).
- An isolated mock-only domain 165 replay of snapshot `capture_frame_000001_20260924_035244_542842` **without** the old corrections planned and executed the full `t_cut` successfully. `scan_results/hcut_path_consistency_20260926/nominal_motion_summary.json` records 12 controller stages, total `50.2 s`, left-to-middle transit `18.15 s`, joint6 range `2.695 rad`, and two 1 s dwells. The executor logged `H-cut trajectory execution completed`; MoveIt logged `Completed trajectory execution with status SUCCEEDED`. Video: `scan_results/hcut_path_consistency_20260926/rviz_nominal_guarded_mock.mp4`; joint-state bag: `nominal_joint_bag/` in the same directory. This is **not** an apples-to-apples speed improvement against the prior corrected run: the old left correction was disabled in the successful run.
- `README.md` and `src/igus_rebel_moveit_config/XEG32_MOVEIT_CHANGES.md` describe the new transit scaling, solver order, wrist guard, and limits. All isolated mock processes were stopped after the run; no physical trajectory was sent.

## 3. In progress

No process or edit is currently underway. Work stopped at a **fail-closed corrected-cut planning failure**: with the pre-TCP-change left correction enabled (middle correction empty) on the same replay snapshot, the new guard returned:

```text
planned free-space transit winds the wrist: transit 'transition from left side cut finish pose to middle seam cut approach pose [ompl_cartesian]' joint6 travel 4.8063 rad exceeds 3.14159 rad
```

The exact service response is `scan_results/hcut_path_consistency_20260926/plan_response.txt`. Thus the nominal T-cut works in mock, but a full T-cut with the old taught left pose does **not** pass the new planner. No collision-free shorter branch preserving that taught pose was proved; an exploratory `/compute_ik` request timed out and two follow-up investigations ended without an applied solution. Do not describe the corrected-cut issue as fixed or bypass the wrist guard to make it pass.

## 4. Next steps

1. **Choose a physically valid left correction.** Files: `/home/farhan/cut_debug_corrections/left_capture_frame_000001_20260916_235239_389784.json`, `src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro`, `src/igus_rebel_moveit_config/config/box_h_cut_executor.yaml`. The saved complete record was taught before the September 24 TCP edit; keep opt-in `false` for physical use. **Done when:** a new observation with the current TCP has verified actual blade pose/contact, valid record provenance, and explicit operator approval for any real motion. Mock IK/collision success alone does not satisfy this.
2. **Investigate the short wrist branch for the corrected left-to-middle transition.** Files: `src/igus_rebel_moveit_config/src/box_h_cut_executor.cpp` (`planCutSegments`, `adaptLeftCorrectedOrientations`, `makeTransitPlanner`, `validateFreeSpaceTransitWrist`), `scan_results/hcut_mock_corrected_20260926/mock_joint_states_bag/`, and `scan_results/hcut_path_consistency_20260926/source_snapshot.json`. Compare IK solutions seeded from the *propagated left-finish state*, with the same carton/pole planning scene and exact TCP pose; do not seed from an unrelated all-zero state and call the result a fix. **Done when:** a full corrected mock `t_cut` reaches every left and middle endpoint with `min_fraction=1.0`, all stages collision-valid, each free-space joint6 travel <= pi, and no contact/dwell/safety-gate regression; otherwise document that no shorter feasible branch was found and keep failing closed.
3. **Measure repeatability rather than relying on one plan.** Files: `src/igus_rebel_moveit_config/src/box_h_cut_executor.cpp`, `scan_results/hcut_path_consistency_20260926/baseline_metrics.json`, `nominal_motion_summary.json`, and a new isolated-mock result directory. Repeat plans from the same robot start and snapshot, compare wrist travel, selected roll, transit duration, and full cut endpoint completion; distinguish corrected from uncorrected runs. **Done when:** the stated consistency/time claim is backed by repeated like-for-like mock results, or the limitation is explicit. Do not raise physical transit scaling based only on a mock run.
4. **Add durable regression coverage if the branch behavior is changed.** Files: `src/igus_rebel_moveit_config/CMakeLists.txt` and an appropriate test file (none currently established for this H-cut path). Cover reject-long-transit, accept-bounded-transit, malformed/non-finite trajectory failure, scoped transit-only check, unchanged dwell and Cartesian completion. **Done when:** the test fails on the old behavior and passes with the fix, package tests and native build pass, and a mock full-plan smoke still passes. No automated H-cut suite was run in this handoff.
5. **Review documentation discrepancy separately.** Files: `README.md:44`, `src/igus_rebel/include/igus_rebel/Rebel.hpp:88-89`. README says `192.168.3.11:3920`; header currently says `192.168.50.133:3920`. **Done when:** the documented connection target matches the actual deployment decision, without changing controller connectivity as a side effect of H-cut work.

## 5. Key decisions and rejected approaches

- **Plan only unless explicitly dispatched:** service planning does not move the arm. `execution_enabled=false`; staged debug additionally needs `debug_motion_execution_enabled=true`. Keep fresh snapshot checks and the fixed external camera-pole collision object. The pole is not mounted on the gripper.
- **Rigid taught-pose mapping:** saved start/end full TCP quaternions are mapped from recorded cut basis to current detected basis; finite cut-local translation deltas rotate with the carton. A learned left pose may penetrate along local X or Z; middle/right require local Z. The left roll search tests deterministic `0,+5,-5,...` candidates with collision-aware IK before MTC; MTC remains final authority. Do not replace a failed correction with a silent nominal pose.
- **Separate free-space from contact:** joint interpolation first, original transit pipeline as fallback. Separate transit scalings default to the existing `0.05`, so shipping defaults do not speed up physical motion; contact Cartesian/Pilz strokes, 1 s dwells, collision allowances, and `min_fraction=1.0` were not relaxed. Faster physical transit needs separate safety review.
- **Reject wrist winding:** the solution-level free-space joint6 > pi check is a rejection gate, not an IK optimizer. The old corrected run had joint6 `4.818 rad` range and a `44.75 s` transition; the current corrected plan is rejected at `4.8063 rad`. Removing this gate, accepting partial Cartesian paths, or raising all motion speed would hide the problem; those approaches were rejected.
- **Do not compare unlike plans as proof of speedup:** the current `50.2 s` successful nominal/mock run has corrections disabled. The `78.75 s` earlier run used the stale left correction before the guard, and its `44.75 s` transition is a different pose/IK branch. Their difference demonstrates impact of configuration/branch, not an isolated timing gain from the new code.
- **No direct old-to-optical calibration publication:** camera driver owns its camera-frame chain. Mock fixtures used `base_link -> camera_link` and `camera_link -> camera_color_optical_frame` to make the external pole available; do not publish an optical frame with a competing parent.

## 6. Known issues

- **Blocking for corrected T-cut:** exact wrist-guard failure in §3. `scan_results/hcut_path_consistency_20260926/plan_response.txt` is a negative result, not an execution result. No trajectory was dispatched for this rejected corrected plan.
- **Stale tool teaching:** configured September 14/16 correction files precede the active September 24 TCP; schema checks use tool-link *name*, not a measured TCP/calibration revision. Enabling an old correction can generate a plausible, IK-valid but physically wrong blade angle. Current TCP joint origin is `(0.005937, 0.094121, 0.130699)` from `igus_tool0`, with Z noted as pending stale-model check in the URDF comment.
- **Blade collision model limitation:** `xeg32_tool_tip` is a 6 mm visual sphere without collision geometry. Scoped tool/carton permission exists, but mock collision validation does not prove actual blade clearance/contact/cut quality.
- **Unproven repeatability:** one successful nominal mock T-cut and one rejected corrected mock T-cut were observed after the new guard. Multiple-seed planning consistency, a shorter valid corrected IK branch, and real-robot outcomes are unknown. No automated H-cut tests were run; `CMakeLists.txt` has no observed `BUILD_TESTING` block for this path. Build passed; do not claim a green full test suite.
- **Docs mismatch:** `README.md:44` reports `192.168.3.11:3920`, whereas `src/igus_rebel/include/igus_rebel/Rebel.hpp:88-89` specifies `192.168.50.133:3920`. Preserve connection safety until deployment target is confirmed.
- **Snapshot replay is mock-only:** historical snapshot timestamps were refreshed in a labeled copy to pass the normal freshness gate for isolated replay; the original geometry was not changed. Never use a re-stamped historical detection to justify physical execution. The executor intentionally rejects stale live snapshots.

## 7. Important files

- `src/igus_rebel_moveit_config/src/box_h_cut_executor.cpp` — MTC snapshot/correction application, collision-aware left IK, free-space transit planners, wrist guard, execution gating.
- `src/igus_rebel_moveit_config/config/box_h_cut_executor.yaml` — H-cut geometry, corrections, speeds, planner defaults, and fail-closed gates.
- `src/igus_rebel_moveit_config/launch/move_group.launch.py` — MoveIt/executor/RViz/camera-pole launch wiring; explicit `hardware_protocol` selection.
- `src/igus_rebel/launch/rebel.launch.py` — ros2_control robot driver and `mock_hardware` controller launch.
- `src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro` — active and proposed TCP visual links and transform to `igus_tool0`.
- `src/igus_rebel_moveit_config/scripts/camera_pole_collision.py` — fixed external camera-pole planning-scene object from TF.
- `src/igus_rebel_moveit_config/CMakeLists.txt` — executable dependencies, install targets; inspect before adding tests.
- `README.md`, `src/igus_rebel_moveit_config/XEG32_MOVEIT_CHANGES.md` — workflow and safety documentation.
- `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json` — current external canonical snapshot path in YAML; freshness and contents must be checked before use.
- `/home/farhan/cut_debug_corrections/left_capture_frame_000001_20260916_235239_389784.json` — complete old left teaching record; not validated for current physical TCP.
- `scan_results/hcut_path_consistency_20260926/` — mock source/replay/provenance, rejected corrected reply, successful nominal reply, bag, RViz video, and timing summaries.
- `scan_results/hcut_mock_corrected_20260926/` — earlier before-guard corrected mock run with old taught left pose; baseline only, not physical proof.

## 8. How to run/test

From `/home/farhan/igus_rebel_ros2` on Ubuntu 22.04 / ROS 2 Humble; documented native install/build commands (not all rerun at handoff):

```bash
source /opt/ros/humble/setup.bash
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install
source install/setup.bash
```

Targeted build **actually passed**: `source /opt/ros/humble/setup.bash && colcon build --packages-select igus_rebel_moveit_config`. Available package test commands, **not run for this handoff**: `colcon test --packages-select igus_rebel_moveit_config && colcon test-result --verbose`. No H-cut behavior test suite is known.

For an isolated, nonphysical mock run, use separate terminals with the **same unused** domain; 165 was used previously but the processes were stopped. Do not share the physical robot's ROS domain. Example:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=165 ROS_LOCALHOST_ONLY=1
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
```

In another sourced terminal with the same domain:

```bash
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false \
  run_box_h_cut_executor:=true run_box_motion_executor:=false
```

The camera pole is required. In the prior **mock-only** fixture, two additional sourced terminals in that same domain ran these static TF publishers (verify against the current calibration before reusing; these are not a substitute for live physical TF):

```bash
ros2 run tf2_ros static_transform_publisher --x -0.052362891163214 --y -0.306337290082351 --z 0.452079756273593 --qx 0.022773271385773 --qy 0.373836805030297 --qz -0.079141815126831 --qw 0.923831150380609 --frame-id base_link --child-frame-id camera_link
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 --frame-id camera_link --child-frame-id camera_color_optical_frame
```

For a **mock-only historical replay**, make a separate absolute `.json` copy of `scan_results/hcut_path_consistency_20260926/source_snapshot.json` with *only* `acquisition_stamp_ns` changed to the current `time.time_ns()` (and a fresh file mtime); keep the source and provenance. Do not set `debug_allow_stale_snapshot=true` for execution. Then, in a sourced same-domain terminal:

```bash
ros2 control list_hardware_components  # must show mock_components/GenericSystem
ros2 param set /box_h_cut_executor cut_pattern t_cut
ros2 param set /box_h_cut_executor external_snapshot_file /ABSOLUTE/PATH/TO/FRESH_MOCK_REPLAY.json
ros2 param get /box_h_cut_executor execution_enabled  # false
ros2 service call /box_h_cut_executor/plan_h_cut std_srvs/srv/Trigger '{}'
```

Default `cut_debug_correction_enabled=false` produces the nominal plan. For the **rejected old-left-correction planning experiment only**, in isolated mock set `cut_debug_middle_correction_file ''` and `cut_debug_correction_enabled true` before calling `plan_h_cut`; expect the §3 rejection on the saved fixture. Reset correction to `false`. Do not execute a rejected plan. Explicit execution was done only for the complete nominal mock plan: after confirming mock hardware and `success=True`, temporarily set `execution_enabled true`, call `/box_h_cut_executor/execute_planned_h_cut std_srvs/srv/Trigger '{}'`, then set it back to `false`; the service response means **dispatch accepted**, so also verify `H-cut trajectory execution completed` in executor logs, `Completed trajectory execution with status SUCCEEDED` in MoveIt logs, and changing `/joint_states`. No physical execution is authorized by this handoff. Stop the isolated launches/TF publishers afterward.

## 9. Uncommitted changes

Before creating this file, `main` at `eaea80a` had **147** `git status --porcelain=v1` entries: `MM` 11, staged additions `A ` 83, unstaged modifications ` M` 12, staged modifications `M ` 5, unstaged deletions ` D` 2, unstaged addition ` A` 1, untracked `??` 33. `git diff --stat` (unstaged): **26 files, +793/-236**; `git diff --cached --stat`: **99 files, +55,444/-825**. In particular `README.md`, `src/igus_rebel_moveit_config/src/box_h_cut_executor.cpp`, `config/box_h_cut_executor.yaml`, and `XEG32_MOVEIT_CHANGES.md` are all `MM` (both staged and unstaged edits). `scan_results/` contains untracked mock evidence; many other modified/staged/untracked files concern perception, meshes, launch, calibration, and teleop and were not audited in this handoff. This new `HANDOFF.md` adds another untracked file after those counts; rerun `git status` for exact post-handoff state.

**Keep the existing working tree; do not reset, clean, stash, or commit on the basis of this handoff.** The H-cut edits are safe to retain as plan-only/mock-tested changes, **not** as a validated physical cut. Safety of unrelated changes is unknown; review each before staging/committing. No commit was made.

## 10. Open questions

- Can a collision-free, short joint6 branch reach the **exact** middle approach from the corrected left finish on the current TCP, or must the left teaching be replaced? **Unknown.** The old corrected plan was rejected; an exploratory IK request timed out without proving an alternative.
- What is the measured physical blade/TCP offset and valid penetration/contact pose after the September 24 URDF change? **Unknown.** Needs supervised re-teaching/verification before physical cutting.
- Should physical free-space transit scalings ever be increased above 0.05? Requires the user's/operator's explicit safety decision after mock evidence; no increase was made by default.
- Whether the production task should stop at `t_cut_handoff` for a second arm or complete the full `t_cut` is a workflow decision not resolved here. No second-arm execution is implemented.
- Which deployment controller address should be documented (`192.168.3.11` in README vs `192.168.50.133` in `Rebel.hpp`)? Unknown until the operator confirms the intended hardware endpoint.
