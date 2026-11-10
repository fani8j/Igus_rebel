# Handoff — igus ReBeL H-cut workspace

> **Never dispatch motion on the physical robot without explicit operator approval in the
> current conversation.** Everything below marked "verified" was verified in isolated mock
> (`ROS_LOCALHOST_ONLY=1`, unused `ROS_DOMAIN_ID`) or against a fake CRI server, not on the
> real arm.
## 2026-11-10 Left cut gate trim (builder + lead fix + Astra review; MOCK ONLY, UNCOMMITTED, NOT in the main install)
- Problem: new box pose (live picture 05:35, `scan_results/cleanup_mock_20261110/variants/live.json`) was refused: final collision gate found
  `detected_carton / knife_c_handpiece` at waypoint 129/136 of the left side cut, between two search stations near the front end.
- Change (`box_h_cut_executor.cpp`, new `left_cut_gate_trim.hpp` + gtest, YAML): on a gate contact inside the forward left cut stroke the cut is
  shortened at the nearer end (no station within `left_cut_gate_trim_clearance_m` 0.010 of the contact) and planned again with every gate in
  force, up to `left_cut_gate_trim_max_retries` 3 (0 = old refusal). First-attempt passes are unchanged. Not used for cut-debug, hand-off,
  `left_debug_stop_at_approach`, supplied plans; the reverse patch is not re-evaluated on a shortened cut.
- Astra review: no gate bypass; one finding fixed by the lead (a shortened cut that then fails elsewhere keeps its cached plan so the lift /
  top-length retries still run). Open remarks: "clearance" is a distance along the cut, not a robot-to-box clearance; nearer-end choice is a
  heuristic; no executor-level tests for reverse travel / patch / repeated contacts.
- Mock (domain 226, overlay `build_gate`/`install_gate` on the main install, runs `gate_*`, `gate2_*`): live pose plans, kept 277.4 of 297.2 mm
  (front trim 19.8 mm, 1 retry), mock-executed twice (138.7 s, 140.2 s, phases order); t054203 and the 05:02 pose are identical to before
  (81.317 s / 80.208 s). At the live pose the primary `sequence` order still fails on the joint6 wrist gate of the top-cut transit, so the
  coordinator uses a fallback order (2/4 or 3/4, varies per run).
- The YAML is symlinked into the main install, so the 10-31 main executor also sees the two new parameters (it does not declare them).
- Next: operator decision on commit + consolidated rebuild of the main install; nothing real was run.

## 2026-11-10 Repo cleanup, merge of main, mock re-check (lead; MOCK ONLY, real arms not moved)
- Committed on hcut-path-consistency (local, NOT pushed): 4b14cd4b src work incl. dual_cut_ui / side_cut_vision / rebel_servo_relay (were untracked),
  5a903da9 tools, 4502ff07 docs, 3327f6f5 merge of main (carton height from snapshot `support_depth_below_top_m`, `carton_height_from_snapshot`
  default true), 38bc29bf rebel_mocap_follower (+ `input_smoothing_tau_s`, default 0 = off, mock only, Astra review still open) and arm
  diagnostics tools from jtc-tracking. UNCOMMITTED: removal of src/rebel_tcp_gui and src/side_tape_diag, CLAUDE.md, tcp_calibration_tutorial.html.
- MAIN install NOT rebuilt: driver = source; executor and coordinator binaries are 10-31 (no carton-height code, no 11-06 left patch); dual_cut_ui only
  in install_ui (10-24 copy). A rebuild changes the carton collision box on the real robot: do it with the stack down, mock test first.
- Deleted: 23 build_*/install_* overlay pairs + log_ovl_rec (kept install_ui, install_vision, install_patch and their build dirs), 471 old colcon
  log dirs, caches. 23.2 -> 29.6 GB free. List: scan_results/cleanup_mock_20261110/logs/overlay_delete_list.txt.
- Mock (domain 226, main install as installed, snapshot t054203, sequence): before and after the cleanup identical: plan ok, makespan 81.317 s,
  executed 83.6 s, IDLE -> PLANNED -> EXECUTING -> PLANNED. Evidence scan_results/cleanup_mock_20261110/.
- Fresh picture (scene camera, domain 11, read-only detector writing only into the evidence folder): the running camera launch had stalled
  (0 frames); operator asked to stop it; a scene-only launch gave frames. DetectBox ran end to end (2.4 s) but REJECTED the box on the table:
  `no_top_face` (flaps already cut and lifted, box partly out of view). No snapshot, so no plan for the new box yet: needs a sealed box in view.
- 04:55:06 the single right-arm driver (192.168.3.12, up since 11-09 18:58) received SIGINT and shut down cleanly (Disable + Quit); not sent by
  the lead. Cameras, detector, SAM3 daemon and mock stack are stopped; nothing runs on domain 11.
- NOT merged: jtc-tracking driver delta (CRI cart/joint jog modes, diagnostics; 24 conflicting files). The tracking settings themselves
  (p3/ff0.6, scale 1.65, accel 0.4) are already on this branch and in the main install since 10-31. dio-gripper stays separate (operator).
- Do not touch: mackay / box-3d / fani8j/box-3d / box/ / medicine-grasp and stand packages (operator).

## 2026-11-09 GR00T mocap EEF models (james's gr00t_real_infer.py): real runs by operator
- Model B (clean): flap raised in 6/8 recorded runs; 6/8 ended with right joint 2 drive disabled (mask 2, kin 99, not latched), cause open. Model A (segmented): arm folds at joint 5, never reaches the box.
- Report `~/groot/share/groot_report_20261109.md`; results `~/groot/share/groot_rollout_results_20261109_model{A,B}/`; script `~/groot/inference_code_20261109/rebel_flap_gr00t`. Script has known safety gaps (see report).

## 2026-11-06 GR00T right-arm checkpoint real_right_3flaps_35ep_15fps: config + offline check (no real run yet)
- `~/Downloads/real_right_3flaps.tar.gz` -> `~/groot/checkpoints/real_right_3flaps/real_right_3flaps_35ep_15fps/checkpoint-3200`. Same
  arch/processor as the 20ep one. Training data (stats match): 29 Oct real_right (20) + 3 Nov real_right_3nov_raw (15, front/back/right),
  15 fps (stride 2). Task text per flap = the instruction (`--instruction`); right flap has its own start pose.
- Client `tools/groot_inference/rollout_config_right_3flaps.yaml`, runbook `REAL_TEST_RIGHT_3FLAPS.md`, offline `offline_check_3flaps.py`
  (training eps: MAE 0.0082 vs hold 0.0204, 100 % gate pass; 640x360 = 1152x648; wrong instruction clearly worse).
- 2026-11-03 real runs of the 20ep model: reached the box but pushed on it, no flap opened (A2 overcurrent once, 2 stalls); package
  `~/groot/share/groot_rollout_results_20261103.zip`. Open: client stop on hardware latch / frozen joint states (proposed only).

## 2026-11-03 GR00T right-arm checkpoint rebel_open_front_20ep: inference set up (MOCK ONLY)
- `~/Downloads/rebel_open_front_20ep.tar.gz` -> `~/groot/checkpoints/rebel_open_front_20ep/rebel_open_front_20ep_360p/checkpoint-3000`.
  RIGHT arm, trained on the 20 `real_right` episodes (29 Oct; stats identical), 30 Hz, head + wrist_right at 640x360.
- Client config `tools/groot_inference/rollout_config_right20.yaml`, runbook `REAL_TEST_RIGHT20.md`, offline check `offline_check_right20.py`
  (training eps: MAE 0.0054 vs hold 0.0113, all gates pass). Mock dual-table run (domain 187): observe + 48-step execute OK.
- Server: `start_server.sh start --checkpoint <dir above>`; load from the USB drive ~11 min (ready timeout now 1500 s). No real run yet.

## 2026-10-29 GR00T N1.7 flap checkpoint: inference set up (lead + 3 builders; MOCK ONLY, real arm untouched)
- Checkpoint `~/Downloads/rebel_frontlid_real_12ep.tar.gz` (Gr00tN1d7 on Cosmos-Reason2-2B) extracted to `~/groot/checkpoints/`
  (`~/groot` -> external drive). Isaac-GR00T @ 944fe70 + venv in `~/groot/Isaac-GR00T` (torch 2.9 cu128, flash-attn 2 on sm_120, no patches).
  Server: `tools/groot_inference/start_server.sh start|stop|ping` (port 5555, ~6.3 GB VRAM, ~70 ms/query, 1-3 min load).
  Client: `tools/groot_inference/groot_ros_rollout.py` (observe default; `--execute` needs typed `yes`; gates, 0.1 rad/s cap, cancel+hold).
- Training data (user + checkpoint stats, exact match): the 12 mocap-teleop episodes of the LEFT arm
  `~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-*/mocap_left_*` (robot_remote recorder, 30 fps -> trained every 2nd frame = 15 Hz,
  action = next state, 16 steps). Inputs: `head` = /camera/color/image_raw/compressed, `wrist_left` = LEFT gripper camera
  /arm_left/gripper_camera/image_raw/compressed, both PIL draft + BILINEAR to 640x480 (squashed). Text barely matters
  ("open the box flaps" x9 / "open flap 1" x3). The first guess (right arm, lid_open episodes) was wrong.
- Offline on the training episodes: MAE 0.016 rad vs hold-still 0.041, per-joint direction corr 0.84-0.97, speed ratio 1.07;
  with a black wrist_left 0.042 (= hold still): the model depends on the gripper camera. Mock suite 141/141; real-model mock e2e
  from mocap_left_new_1 frame 0: 8 chunks pass, executed direction cosine 0.96 vs the demo.
- OPEN before a real run: left arm = `rebel.launch.py hardware_protocol:=rebel robot_ip:=192.168.3.11`; start pose of the mocap
  episodes [0, 0, 1.047, 0, 0.873, 0]. The training head view (mocap) is framed differently from the 2026-10-26 head view
  (`/tmp/groot_cam_compare.png`): check the live head + gripper views match the training frames before trusting it (observe
  mode with `--save-images`). Real execute needs explicit operator approval.
- 2026-10-29 10:00-10:10 first REAL runs (operator ran them, left arm .11, A2 did not trip): 16-step test clean; 3 full runs
  followed the demos' first phase (j3 up, wrist j4 -> -2.5 / j6 -> +2.6) then stopped on gates after ~45-50 chunks (j3 URDF inset
  2.378; j2 -0.53 < training min - 0.1). At that phase the demos have j2 ~0..+0.35, the real arm -0.4..-0.5 (leaning back, gripper
  away from the box): off-distribution drift; live head view / box placement differ from training (`/tmp/run3_end.jpg`).

## 2026-10-29 TurboVLA mocap10k handoff: inference set up (lead + 2 builders; MOCK + Isaac headless only)
- Bundle `~/Downloads/compal-turbovla-handoff-v1-inference-002.tar` (colleague's TurboVLA, step-10k EMA, same 12 left-arm mocap
  episodes, 10 train / 2 held-out mocap_left_15 + _new_9) extracted to `~/turbovla/` (external drive). `bundle.py verify/setup/check`
  pass (torch 2.13 cu130 OK on sm_120; reference error 0.0); Isaac 5.1 headless 120-step smoke passed on the 2nd try (1st: camera
  None after shader compile). Server: `tools/turbovla_inference/start_turbovla_server.sh` (ZeroMQ :5556, ~2 GB VRAM, ~22 ms).
  Client: `groot_ros_rollout.py --config rollout_config_turbovla.yaml` (224x224 head + wrist_left, knife black server-side, 14-D
  state with right arm/grippers = 0, ACT ensemble h30 k0.01, one FollowJointTrajectory goal per step; never forwards ch 6-13).
  Mock suite 323/323.
- Model quality (offline, whole raw episodes): train MAE ~0.075 vs hold-still 0.057 (moving frames: not better than holding on 9/10
  train episodes); held-out 0.31 (wrong wrist branch j4/j6). Live = offline at the start pose: chunk[0] jumps 0.20 rad on j4/j6 where
  the demo stands still, so the 0.05 rad first_step gate refuses it (median first-step jump over an episode 0.09 rad). Not ready
  for the real arm; ask the colleague for their validation numbers (evaluate_mocap.py hold baseline) to confirm.

## 2026-10-24 CRI MoveJoint single-arm REAL qualification (lead; operator at the e-stop; left arm only)
Worktree `cri-move-joint` b22ad59 driver (own install, not main), left arm 192.168.3.11, domain 222, home pose, <=25 %.
Whole ladder passed (jog, JTC, MoveJoint single/multi-joint at 5/10/25 %, cancel before start, Move Stop mid-move,
e-stop, driver SIGKILL, LAN cable pull): `scan_results/cri_single_arm_20261022/REPORT.md`, evidence `run_real_1/`.
Key facts: 0.69 deg/s per vel%, ~1.3 rad/s^2 at acc 20, STATUS updates 30 ms, Move Stop <= 90 ms / <= 0.72 deg at 25 %,
synchronous joint-linear path; the robot FINISHES a CRI move after driver death and link loss. Open before any merge /
dual-cell use: driver does not parse MOVETOEXECERROR; /joint_states is frozen while latched. Not merged, not committed.
The client re-review was skipped by the operator. Unit `cri-left-real` stopped.
Later the same day both defects were FIXED in the worktree (uncommitted; MOVETOEXECERROR -> FAULT + latch; new
`~/robot_joint_states` + `~/hardware_state` topics; tests 13/21/17/10/124 pass): see REPORT.md "Defects". Re-tested on the real
arm the same day (run_real_2/NOTES.md): both fixes work. Then a j1 speed ladder 25-100 % + Move Stop at
speed (run_real_3/NOTES.md; driver hard cap raised 50 -> 100 in the worktree, default 25): linear 0.66 deg/s per vel%,
100 % = 1.14 rad/s (> MoveIt 0.785), stop travel ~ v x 0.06 s. Committed 3e335da on fani8j/cri-move-joint (not pushed, not merged). Handed off to the dual-arm cutting lead thread 2026-10-24 (merge + coordinator wiring are theirs). 2026-10-26: advisor review fixes (fable + astra) committed c2e8811 on top (atomic send check, late robot error -> FAULT, error during stop -> FAULT, receive-time STATUS stamps + dropped-message FAULT, unconfirmed cancel -> FAULT, HardwareState caps; test_rebel_driver 139); hash sent to the dual-arm lead (their driver merge into the main tree has only b22ad59 + 3e335da: c2e8811 must follow). Client move_ladder.py: Ctrl-C step never auto-returns under --chat-approved (exit 7), stricter hardware_state health; sim rehearsal_reviewfix3 OK. Friend's lid-open package: tools/lid_open_friend/ (RUN_ON_THIS_CELL.md; mock-tested, real right-arm run by the operator).

## 2026-10-16 Servo speed ratio fix: rebel_servo_relay (mock only, not on hardware)

- **Problem:** Servo jogs reach a constant ~34 % of the commanded speed in mock. The real-controller law
  (velocity interface, p 1.0, ff 0.4) gives ~26 %. Root cause, in `scan_results/servo_speed_ratio_20261016/SOURCE_ANALYSIS.md`:
  - Servo 2.5.10 anchors each command on the measured state.
  - Servo always runs its Butterworth filter; `use_smoothing` is not a 2.5.10 parameter.
  - The JTC restarts each stamp-0 trajectory from the current state.
  - No Servo/JTC parameter fixes it.
- **Fix:** new package `rebel_servo_relay`, in the cri-move-joint worktree, untracked and not in the main install.
  - A pass-through smoothing plugin.
  - A relay that tracks its own reference, sampling its last published trajectory the way the JTC does, and sends
    3-point trajectories that end at rest.
  - Bounds: lead `min(0.05, 0.2 s·|v| + 0.005)`; per-joint velocity scaling near the limits.
  - Hold behaviour: a watchdog holds after 68 ms of silence; a reject publishes a hold; queues are depth 1.
  - Freshness checks: stamped joint_states are required.
- **Wiring:** in both mocap launches as `servo_relay:=true` (default). The follower's JTC exclusivity gate requires
  exactly the relay as the one publisher.
- **Mock results:** 100 % speed (position interface).
- **Real-law mock results** (velocity interface): 57 % / 65 % at 0.3 / 0.1 rad/s with the defaults; ~89-100 % with
  `relay_lead_time_s:=0.6 relay_max_lead_rad:=0.15`, at the cost of more travel after a stop.
- **Stop distances:** silence stop 21 mrad vs 104 mrad without the relay. Tests: 28 gtest, 56 follower pytest.
  Evidence and report: `scan_results/servo_speed_ratio_20261016/fix/REPORT.txt`.
- **Reviews:** two advisor-astra reviews; all code findings are fixed.
- **Open, procedural, for the first real test:**
  - The follower gate can't see JTC action goals (MoveIt execute), so nothing else may command the arm during mocap.
  - First test: ≤ 0.1 rad/s (follower max/catch-up velocity 0.1), default lead, clear space, operator at an
    independent e-stop, only with explicit approval.
  - `servo_relay:=false` is a comparison option, not a safety fallback.
- **Same issue elsewhere:** the single-arm keyboard/gamepad teleop and dual_teleop have the same speed problem
  and don't use the relay yet.

## 2026-10-15 First real dual-arm bring-up (operator at both e-stops)

- Dual driver (`dual_table.launch.py hardware_protocol:=rebel`, .11/.12) runs both arms in fault group
  `dual_arm_table`. Verified on the real cell: an e-stop on either arm stops the other in the same cycle, with
  the arms standing still (both directions) and while moving (left e-stop, both arms moving). The STATUS ESTOP
  field stays 3 during an e-stop (useless); the trip comes from kinstate "motion not allowed".
- After an e-stop the Enable sent by the lifecycle activation can be lost (motor supply returns ~0.2 s later):
  `dual_arm_recover.py recover` now re-enables via `reset_and_enable` (its reply is the enabled signal). Driver
  log lines never reach /rosout (non-node logger), so the script's /rosout-based fault detection is advisory only.
- **Installed 2026-10-15 13:04, not yet proven on the real cell:** an e-stop while controllers are ACTIVE leaves
  them active (Humble), and Humble refuses to deactivate them while the hardware is unconfigured. Driver fix
  (`controllerRestartHold_`: jog held at zero after a latch recovery until the arm's controller is re-activated) is
  in the main install (test_rebel_driver 61/61; evidence `scan_results/dual_arm_stack_20261015/driver_hold_fix/`).
  `recover` should now handle an e-stop while moving; next real test: right e-stop while moving, then `recover`
  (fallback if it fails: restart the dual driver launch). Review (advisor-astra) notes kept open: release is keyed on interface names (safe because Humble
  cannot start interfaces claimed by an active controller); pre-existing ALIVEJOG copy-then-send and
  reconnect-before-read() races; test harness writes `vel_cmd` from two threads.
- Dual `joint_limits.yaml` bring-up caps (0.1 rad/s, 0.5 rad/s^2) lifted to the ReBeL 0.785/5.24 (RViz and Servo
  only; the cutting path uses the per-arm limits). Backup `scan_results/dual_arm_stack_20261015/joint_limits.dual.bringup_cap.yaml.bak`.
- `switch_controllers --deactivate X --activate X` in ONE call does not restart X on Humble 2.54.2; use two calls.
- `camera_head` collision box (camera_pole_collision.py, shared by single-arm, dual and per-arm planning) is now
  the real Gemini 336 body (USB 2bc5:0803; orbbec_description mesh bounds 30 x 89.5 x 25 mm) + 10 mm margin,
  posed with the full camera_link orientation; was a 0.10 m yaw-only cube. Pole position unchanged
  (`POLE_BACK` 0.065). Mock: left wrist camera at home 32 -> 54 mm from it. Evidence
  `scan_results/camera_head_size_20261015/`. Dual SRDF has a new left `ready` state (j2 0.6, j3 0.9) to leave
  home before mocap; mocap Servo scene threshold 0.025 (cri-move-joint worktree, launch arg).
- Executor SIGSEGV during planning (rare, in MTC `MoveTo::compute`): MTC reads `result->getLastWayPoint()` even
  when planning failed; the transit retry (`MultiPlanner` clear() then timeout) left a non-null EMPTY trajectory.
  Fixed in source (`NearestBranchTransitPlanner::guardResult`: failed plan -> result reset; success without
  waypoints -> failure). Verified in `build_seg` only (standalone reproducer crashes before / not after; 3,744 plans,
  0 crashes). NOT in the main install until `igus_rebel_moveit_config` is rebuilt. Evidence
  `scan_results/dual_arm_stack_20261015/executor_segfault/NOTES.md`. Delete `build_seg`/`install_seg` when done.

## 2026-10-06 Dual-arm table cell + calibration (mock / synthetic only)

Nothing below has moved a real arm.

**Committed on `hcut-path-consistency`:**
- `374935a`: the driver runs two `Igus/Rebel` instances in one controller_manager. Joint names come from the ros2_control tag. There are new hardware params `node_name`, `digital_output_service` and `trajectory_controller_name`.
- `11f1c4d`: fault groups. With `fault_group`, a fault on either arm latches both and zeroes their jog. The root-cause arm must be recovered first. Also, for all setups: no nonzero jog is sent while latched.
- Reviewed by advisor-fable and advisor-astra, whose findings are addressed.
- **The main `install/` was NOT rebuilt**, so the running driver predates both commits. Before any dual run, rebuild `igus_rebel` with operator approval and check that both arms log "Joined fault group".

**Untracked new packages:**
- `src/rebel_dual_table_description` and `src/rebel_dual_table_moveit_config`.
  - `world` = `arm_left_base_link`. arm_left is the arm used so far. Both arms have identical XEG-32 + knife tooling.
  - The arm_right base and `camera_link` come from `config/dual_arm_calibration.yaml`. That file is seeded with the measured 447.4 mm and `rebel_handeye_h_v2`.
  - The table box sits behind the bases.
  - Joint limits are capped at 0.1 rad/s and 0.5 rad/s² for bring-up, because Humble ignores `default_velocity_scaling_factor`.
  - With `hardware_protocol:=rebel` the controllers load INACTIVE. The launch refuses equal IPs (right defaults to 192.168.3.12) and warns when the calibration `source` is not `calibrated`.
  - MoveIt execution-duration monitoring is on.
- `src/rebel_dual_arm_calibration`:
  - **Inputs:** ChArUco 3x3 / 35 mm per flange (left ids 0-3, right 20-23). A table board (ids 40-49) with touch points for both arms.
  - **Solve:** a corner-reprojection fit. Optional joint-offset nuisance parameters (`--joint-offsets`) are never applied to the robot. k-fold validation.
  - **Trust gate (nominal FK):** at least 4 common touch points, touch RMS ≤ 1.5 mm, and joint vs Kabsch touch-only base within 3 mm / 0.3°.
  - **Recording:** `record.launch.py mode:=dual|single_left|single_right`, so each arm can be recorded with today's single-arm stack. `solve` merges several sessions.
  - Studies are in `scan_results/dual_arm_calibration_20261006/` and bring-up evidence in `scan_results/dual_table_bringup_20261006/`.
- The `rebel_dual_arm_stand_*` packages (stand with lift and sliders) belong to another session and are sim/Isaac only.

**Metrology update, 2026-10-08** (Fable, Astra and a literature review agree):
- The gate now has 8 value-based checks, including:
  - held-out **check stickers** 111/112 (TRE), at most 3 mm;
  - fit stickers 101–104 for spread;
  - a noise-aware joint-vs-touch check;
  - intrinsics validation, via `calibrate_intrinsics` + `--intrinsics-file`.
- Other additions: print scale in x and y, a 2 s settle plus a two-frame corner check, Monte Carlo uncertainty (`--uncertainty-mc`), repeatability (RP), `--compare`, and the `check_points` re-check after a power cycle.
- Expected accuracy with nominal FK: 7–10 mm typical (15–20 mm worst) arm-to-arm at 0.4 m. Padding = max(20, 2 × check error + 10) mm. This is not enough for mm-level coordinated work.
- Operator guide: `src/rebel_dual_arm_calibration/doc/calibration_guide.html`, built by `build_guide.py`. Prints are in `~/dual_arm_targets/`, including `touch_stickers.pdf`.
- Mock evidence: `scan_results/dual_arm_calibration_metrology_20261008/`. Tests: 133 pass.

**Calibration web UI + A4 prints + plates, 2026-10-08:**
- Start it with `ros2 launch rebel_dual_arm_calibration calibration_ui.launch.py mode:=dual output_dir:=...`, which also starts the recorder. Then open the `http://127.0.0.1:8095/?token=...` URL that the log prints.
- Pages:
  - Setup / health
  - Intrinsics (optional; the default is camera_info, and you only do this if gate check 8 fails)
  - Record arm, with a live detection overlay
  - Touches, with a sticker checklist
- Solve and the re-checks stay on the CLI.
- **Jog** (the user chose it):
  - Step-only: cartesian 0.1–10 mm, rotation 0.5–5°, joint 0.5–5°.
  - Speed caps of 10 mm/s, 5°/s and 0.1 rad/s, checked on the cubic Hermite interpolation that JTC actually runs.
  - MoveIt `compute_cartesian_path` with fraction 1.0, plus `check_state_validity`.
  - Real hardware needs a per-page "Enable motion" that expires after 10 min idle.
  - There is a 300 ms gap between steps, and keyboard-generated clicks are ignored.
  - STOP (button or Esc) keeps the arm busy until the result arrives and standstill is measured. Otherwise the jog goes to FAULT (8 s), and the "Clear fault" button needs measured standstill.
  - Every step re-checks everything just before dispatch and reports the achieved vs requested distance.
- Reviews and tests:
  - Reviewed by Fable and Astra; both said they would not approve first real use before these fixes, and all their fixes are in.
  - Browser-tested in mock: 8/8 scenarios.
  - Tests: 102 jog unit, 27 server, 1 status_json. The full package suite had 228 passing before the advisor fixes.
- **Before real-robot use (not done):**
  - Measure the stopping envelope and settle on the real arm, with operator approval.
  - Expect the 0.1/0.5 mm steps to be swallowed by the driver deadband (the done message will show it).
  - During a STATUS coast (>200 ms) the driver keeps the last jog, so a step can overrun by about 2–4 mm and STOP can be delayed by up to 400 ms.
  - The knife tip has no collision geometry, so only the small step sizes protect it.
  - First session: joint 0.5°, then cart 1 mm, e-stop in hand.
  - Open option: a driver-side jog-speed cap during calibration (Astra).
- At the mock all-zero (stretched) pose, Cartesian steps are refused (singularity). Bend the elbow with a joint step first.
- 2026-10-09: the real camera rejected every capture ("board still moving 0.50 px > 0.20"): the old check took the max single-corner shift, which is camera noise. It now judges the COMMON motion (2D similarity fit, limit `max_corner_motion_px` 0.2, new launch arg) plus a 1.0 px single-corner cap. Monte Carlo: still board at 0.14 px noise passes ~92 % per try (the 0.15 px test is xfail by design). New **Auto capture** on the Record arm tab (`ui/autocapture.py`): captures at each new still pose (>= 2 deg), retries timing refusals 15 s, never moves the robot. Browser-tested in mock 10/10; full suite 307 passed. The operator's measured sizes are in `~/dual_arm_calibration/targets_measured.yaml` (arm boards 105.5 x 105.0 mm over 3 squares; table board not measured yet).
- 2026-10-09: `handeye_pose_replay` can replay a calibration session's joint poses: `poses_file:=~/dual_arm_calibration/<left|right> arm:=<left|right>` (+ `start_index`, `max_poses`). It validates the whole list against the model bounds before moving, stops on the first planning failure (prints the resume index), keeps ENTER-per-pose and 5 % speed. 21 loader gtests; mock e2e 3/3 poses reached within 1e-4 rad. Not run on the real robot yet.
- 2026-10-09: **Suggested pose** (calibration UI, Record arm tab; single-arm modes): `pose_suggest.py` scores ~700-1000 candidate board poses (info gain of the hand-eye fit + rotation diversity + empty image cells), IK via /compute_ik, collision-checked; the UI draws the predicted outline, **Send to RViz** publishes only `/rviz/moveit/update_custom_goal_state` (tick 'Allow External Comm.'), **Plan preview / Move there** go through `jog.plan_move/execute_move` (0.04 rad/s, 0.25 rad/s^2, 40 mm/s; supervision; re-validation; one queue slot + stop epoch). advisor-astra's 8 findings fixed (test_f1..f8). Standstill velocity gate is 0.03 rad/s because the real driver's finite-difference velocity reads ~0.012 rad/s for one 0.01-deg encoder step. Mock e2e 10/10, full suite 472 passed. Suggested moves are large (70-85 deg max joint change).
- 2026-10-09: three review rounds on the shared motion machinery (jog steps + Move there): astra x4, fable x1; final verdict of both: no block for a SUPERVISED first real session. Fixes: NaN fail-closed, atomic dispatch vs STOP (`dispatch_goal`), validated terminal results (`unconfirmed` state), event-anchored 1.2 s standstill + gap check + health answers dated by REQUEST time (>= 1.0 s after the event), pre-dispatch standstill gate, MotionSupervisor (stale / 2x speed / tracking 0.05 rad / frozen feedback), collision check on the executed Hermite curve + always re-check at execute, START_MOVED_TOL 4e-4, standstill velocity gate 0.03 rad/s (0.01-deg encoder quantisation). Full suite 530 passed. **Move there on REAL hardware is OFF unless the UI is launched with `allow_real_move:=true`** (operator decision after a supervised test: knife covered, e-stop in hand, one ~10 deg move, a deliberate STOP to measure stopping distance incl. the driver's 200-400 ms coast tier). Not measured yet: STOP latency/distance, real tracking lag vs 0.05 rad step tolerance, freeze-check false trips.
- Calibration state 2026-10-09: left 70 / right 60 board samples (both pass rotation diversity); built-in camera_info kept (own intrinsics too imprecise, agrees within noise); table board measured 199 x 160 mm (targets_measured.yaml); camera-only preview: arm_right_base x -1.3, y -454.3, z 4.8 mm, yaw 2.81 deg. Next: left touches on board crosshairs 1-4 now; right knife not installed yet -> stickers + right touches later in one session (board must not move until then).
- 2026-10-09 **Hover / approach assist** (calibration UI, Touches tab + new Hover test tab; single-arm modes): `hover.py` detects the table board (crosshairs 1-4) and the ArUco-13 hover sheet (aruco_tcp_hover_test_A4.pdf; crosshairs = touch ids 101-104 fit, 111-113 check, `hover_sheet:` in targets_measured.yaml, marker edge NOT yet measured on this print -> 59 mm placeholder) and predicts them in the arm base with `pose_suggest.fit_arm`. Flow: Detect points -> click a crosshair on the live image -> Plan hover (60 mm above, knife into the paper) -> Move (real: allow_real_move) or Send to RViz -> [Descend to 10/20/30 mm: real needs allow_real_move, pre-hover interlock, unmoved check under occlusion] -> operator teleops (MoveIt Servo keyboard) -> Capture touch / Record offset -> predicted-vs-touched logged (hover_log.jsonl / hover_test.jsonl). Mock: predictions 0.1-0.4 mm vs truth; astra review: descend exemption blocked -> now gated. Full suite 680 passed; installed. Real table board predicted at z ~ -335 mm in arm_left base (operator to confirm the stand height).
- 2026-10-10 **Hover tiles** replace the A4 hover sheet: 3 tiles 100x100 mm (`generate_tiles` -> ~/dual_arm_targets/hover_tiles.pdf; ArUco 13/14/15, 50 mm marker; A: 101-103, B: 104-106 fit, C: 111-113 check on a ~40 mm block >= 150 mm from the table-board centre); backing plate cad/calibration_plates plate_tile (100x100x4.2 mm, ~20 g, print 3). Site targets_measured.yaml switched to `hover_tiles:` (marker 50 mm placeholder until measured; fit [101-106], check [111-113]). Guide section 6.5 = touching the table crosshairs with the hover assist (Send to RViz + Servo teleop + Capture touch). Installed; full suite 743 passed (builder) .
- 2026-10-10 **TCP calibration** integrated as tab "0. TCP" (supersedes the Tkinter rebel_tcp_gui): pivot method reviewed by Fable + Astra and a Monte Carlo (0.4 deg joint zero errors -> ~1.2-1.5 mm residual RMS, ~1.3 mm TCP error): SVD + observability gate (rank 3, sigma_min(B)/sqrt(N) >= 0.2), per-axis 1-sigma <= 1 mm, LOO, held-out validate poses; PASS fit RMS <= 2.0 / max <= 3.5 / held-out <= 2.5 mm (GOOD <= 1/2). `tcp_pivot.py`, `ui/tcp_ui.py`; writes tcp_result_<arm>.yaml + a URDF diff (applied by hand + rebuild). Suite: touches now store T_base_link6 + T_link6_tcp; `solve --tcp-file arm=PATH` re-bases touches recorded with an old TCP (a URDF swap alone was silently cancelled); TCP-consistency check requires tilt diversity (pure roll no longer counts). Old 2026-09-22 pivot data: FAIL (5.1 mm RMS); TCP z disputed by ~15 mm (URDF 183.4 mm flange->tip vs ~168 mm) -> caliper check pending. Guide section 2b. Full suite 824 passed; installed.
- 2026-10-12 **Knife moved to the robot's right side (arm_left, .3.11).** Pivot (15 fit + 3 validate, `~/dual_arm_calibration/left/tcp_pivot.jsonl`) solves r_tool0 = (-42.1, -89.6, +104.6) mm, stable to ~1 mm across subsets, but FAILS (fit RMS 4.4, held-out RMS 7.1 / max 9.6 mm; tilts up to 78 deg). Operator could not re-capture, so it was applied by hand as a PROVISIONAL TCP in `igus_rebel_xeg32_wrapper.urdf.xacro` (old value in the comment): +-10 mm, teleop/RViz only, not cutting/touches. Still open: re-capture (tilts <= 30-40 deg, re-seat between poses, caliper the new offset); knife handpiece/holder collision meshes still on the OLD side (need how it was moved); `rebel_analytic_ik.cpp` hardcodes the old TCP, so the H-cut executor's ground-truth self-check will fail closed to KDL until updated; `rebel_dual_table_description/urdf/rebel_tool.macro.xacro` still has the old value; `tcp_pivot.CURRENT_URDF_R_M` (mechanical check, tests) still old.
- 2026-10-12 **Left table-board touches** (4 crosshairs, knife vertical, provisional TCP) + 2 new table images; crosshair/check-sticker test skipped. The table board had MOVED since the 2026-10-09 images (old centre z -336 mm, now -111 mm); `solve` silently fits one board pose to all table images -> 263-284 mm loop closure. With `--exclude-image left/images/cap_0071..0075.png`: loop closure 9.9 mm RMS out of fit, but a near-constant shift (-6.1, +8.1, +2.0) mm with 1.3 mm scatter (fits the +-10 mm provisional TCP); with joint offsets held-out touch 2.49 mm. Trust gate FAIL (no right-arm touches, no check points, intrinsics c pull 11.9 px). Base/camera unchanged: arm_right_base (-1.3, -454.3, +4.8) mm, rpy (-1.04, 0.48, 2.81) deg, calibrated_unverified. Outputs: `scan_results/dual_arm_calibration_20261012/` (calib2/out2 = excluded run). Nothing written to the description config. Gap: solve should flag table images that disagree (board moved).
- 2026-10-12 **Cutting-demo prep (left arm, single-arm stack).** (1) New camera calib `~/.ros2/easy_handeye2/calibrations/dualcal_20261012_left_hovercorr.calib` = calib2 camera_link + hover correction (-5.9, +6.3, 0) mm; `rebel_handeye_h_v2` (2026-09-14) is ~0.4 m off now (camera moved) - launch perception with `calibration_name:=dualcal_20261012_left_hovercorr`. Hover UI got opt-in `hover_correction_<arm>_mm` (builder; 873 tests pass; installed). Hover test with correction: 1.0-2.2 mm lateral at a new board spot, ~2 mm low. (2) PROVISIONAL knife model: URDF knife handpiece/holder hulls translated by the TCP delta (-48.0, -183.7, -26.1) mm; SRDF tightened only (removed holder disables vs link2/link4/motor4 and holder-right_carriage; evidence `scan_results/knife_relocation_20261012/`; backup /tmp/igus_rebel2.srdf.before_20261012). OPEN/BLOCKING for H-cut: session evidence says the mounted tool is probably the NEW variant-C holder (30 deg lean, flange holder, Gemini 305 + Yahboom on the tool, SMC LEHF body?; CAD `scan_results/knife_side_compare_20261007/final4_adj/C/`); CAD tip (link6, CAD->URDF by -90 deg about z) (52.5, 92.3, 170.9) vs pivot (42.1, 89.6, 157.3). Needs operator confirmation, then collision model from the C STLs, tip-frame tilt, ACM re-run, executor knife-orientation review. `rebel_analytic_ik.cpp` and `rebel_dual_table` macro/SRDF (make_srdf.py) still old.
- 2026-10-12 **Variant C model + link5 collision fix (mock only).** Operator confirmed variant C v3 (no gripper, Gemini 305 on the knife block): xacro arg `tool_variant` (`knife_c` default, `xeg32_knife_v1` = old HEAD model), low-poly hulls `meshes/knife_c/`, tilted tip frame, 24 knife_c SRDF pairs (`scan_results/knife_c_model_20261012/`). knife_c home was a FALSE self-collision (camera vs link5's coarse 0.1x0.03x0.1 box): link5 collision in `igus_rebel2.urdf` is now the convex hull of its real mesh (`meshes/rebel_d00617809/collision/link5_hull.stl`, 344 faces, watertight, contains the real mesh); SRDF unchanged (no link5 pair re-enabled; camera/holder-link5 stay checked, >= 16.8 mm over full joint6). Mock: home + joint6 sweep valid for both variants; knife_c t_cut plans on the 09-26/10-01 snapshots; h/combined still kinematically infeasible with the 30 deg lean; today's dual-arm snapshot positions out of reach. Evidence `scan_results/link5_collision_20261012/` (incl. audit: link1-4 primitives badly under-cover the real links, up to 85-88 mm; motor3 20 mm short). OPEN: `rebel_dual_arm_stand_description` check_arm_fk now FAILS (dual-arm macros still have the old link5 box; FK identical) - update both dual-arm macro copies + their SRDF link5 pairs; the running domain-11 MoveIt predates all of this (restart to load). Before the automatic cut on the real arm: plan-only with `calibration_name:=dualcal_20261012_left_hovercorr`, dry run, then cut, each with operator approval.
- 2026-10-12 **Knife TCP confirmed by caliper; handpiece model fixed.** CHOSEN cutter: plastic body 160.6 + metal seat 10.2 + blade 14.4 = 185.2 mm (rear face on the holder pocket floor) -> blade point 4.1 mm from the pivot TCP (2.2 axial, 3.5 lateral): TCP KEPT; the old ~15 mm z dispute came from wrong CAD knife lengths. knife_c handpiece meshes rebuilt (body/collar/seat; collar dia 17 and seat dia 9 are conservative estimates, not measured); blade is visual only (a blade hull blocks every cut stroke, executor allows contact only for xeg32_tool_tip). Mock: home + joint6 sweep valid, t_cut plans on both test snapshots; SRDF unchanged. `scan_results/knife_c_handpiece_fix_20261012/`.
- 2026-10-13 **First left cut: reach + `left_cut_reverse_travel`.** Captures 01:28/02:02 (large_392 carton) do not plan t_cut or left_only: the left stroke starts at back_middle_left with the blade +Y toward the back (side facing hard-coded), and that pose has no IK with knife_c even without collisions. New planning param `left_cut_reverse_travel` (default false; builder, lead-reviewed): left stroke front->back, blade still facing away from travel; fails closed with corrections/left_parallel/cut-debug/side-tape paths. Mock: left_only on the 02:02 capture plans at tilt -10/+10/0 (reversal mirrors the roll: +10 reversed = old -10 lean). t_cut on that capture still fails (middle approach unreachable). Not run on the real robot. `scan_results/first_cut_plan_20261013/`.
- 2026-10-13 **Analytic IK uses the URDF tool offset** (builder, lead-reviewed): `rebel_analytic_ik` takes link6->TCP as input, the executor derives it from the live RobotModel; self-checks pass for knife_c and v1, gtest 8/8. left_parallel search on the 02:02 capture now runs in ~4 s: best fixed orientation theta_p 44 deg, blade 48 deg off the cut line (not usable); mock plan + mock execute completed. Flat pose is reachable at the front end of the 160 mm left cut, not along all of it (render `scan_results/first_cut_plan_20261013/viz/compare_before_now.png`). Next options: orientation varying along the stroke, or carton rotated 90 deg.
- 2026-10-13 **Varying-orientation left parallel cut** (builder x2, advisor-astra review, lead mock-verified): opt-in `left_parallel_varying_orientation_enabled` (+ `left_varying_*`); one continuous contact stroke with the knife angle varying along the cut, trimmed to the longest feasible interval (fails closed < 50 %). 02:02 capture, left_only, `left_varying_max_step_deg 15`: full 160 mm in 4.6 s, blade 7->0 deg to the top, 30->5 deg off the line, monotonic joints; mock execute OK. Default step 5 works but the elbow pumps (9.1 s). Real robot: not run; needs operator approval, air dry run first (TCP +-4 mm, blade not in collision model). `scan_results/first_cut_plan_20261013/`.
- 2026-10-13 **Left cut to the corners + approach-only debug stop** (builder, advisor-astra review, lead mock-verified): `left_cut_extend_to_corners` (+ `left_cut_corner_margin_m` 0.02, `left_cut_seam_cover_m` 0.03; varying mode only) searches the whole left edge minus the margin and keeps the longest reachable stretch that covers the tape crossing +-30 mm (>= 50 % of the nominal 160 mm). `left_debug_stop_at_approach` (left_only): plans the full cut, retains only the approach to the hover pose; execute needs execution_enabled AND debug_motion_execution_enabled. 02:02 capture: 280.8 mm kept (whole edge), theta_p 24.8 / deviation 35.5 deg, stroke 11.3 s, mock execute OK; approach-only mock execute ends on the planned joints. OPEN: planning the full cut while parked exactly at the hover pose fails the wrist gate (1-point transit) - operator decision pending. Not run on the real robot. `scan_results/extend_left_cut_20261013/`.
- 2026-10-13 **Wrist camera (Gemini 305 on knife_c) hand-eye, REAL arm_left** (`tools/wrist_handeye/`, builder + lead fixes): auto orbit around the table board (safety scene: horizontal slab board+50 mm, stand box, pole/head +50 mm; 10 % speed), 14/18 samples. Joint reprojection refinement absorbed the arm's ~4.5 mm positioning error into a 35 mm / 48 deg wrong X -> now rejected (> 5 mm / 2 deg from the AX=XB init), gate judges board consistency + scene cross-check. Result (PARK): link6 -> wrist_camera_color_optical_frame xyz (32.3, 98.5, 4.7) mm, rpy (-1.5, -30.5, -175.7) deg (matches the 30 deg holder lean); TSAI/PARK/DANIILIDIS agree <= 1.8 mm / 0.85 deg, LOO X <= 1.6 mm / 0.1 deg, board vs scene camera 2.9 mm / 2.8 deg, per-sample board scatter 4.5 mm (arm accuracy). The Orbbec driver publishes wrist_camera_link -> ... -> color_optical (identity), so attach link6 -> wrist_camera_link: `scan_results/wrist_handeye_20261013_112818/urdf_snippet_wrist_camera_link.xml` (NOT applied to the URDF yet). `leave_home.py` moves from home (inside the head pad) to a first view. UI eye-in-hand mode: builder still in progress.
- 2026-10-13 **Limit test of the varying left cut planner** (mock, 106 plans): deterministic; plans 42/42 carton positions (fixed search 26/42); quality depends on placement (closer / dy<=0: blade flat and <=15 deg off line; nominal 7/30; far: 45/45). Carton yaw +30 deg gives a perfect 0/0 full-length cut. Gap: no cut-quality gate (accepts 45/45). `scan_results/limit_test_20261013/README.md`.
- 2026-10-13 **Blade height / depth / top-only (lead).** YAML `cut_clearance_m` -0.001 -> -0.003 (side blade 3 mm below the top; restart needed). New planning param `left_varying_start_extra_inward_m` (default 0; varying mode only; builder, lead-reviewed): extra inward depth tapering linearly from the start to 0 at the finish. `middle_blade_reversed` stays true (false = dull side leads, operator); reversed blade + normal travel has no IK with knife_c on the left arm, so new planning param `middle_cut_reverse_travel` (middle_only only) cuts seam_end -> seam_start. SAFETY: cut strokes run with jump detection off and a mock plan contained a wrist flip inside the top stroke (tip 300 mm off the seam) - new fail-closed gate `validateCutStrokesStraight` (every cut stroke within 3 mm of its line, all patterns). `middle_only` now reuses `top_cut_allow_shortened_cut`/`top_cut_min_cut_fraction` (end pulled back in 5 % steps, full plans only). 07:49 capture: 90-95 % from home (70 % from a bad start pose), mock execute OK; not run on the real robot. `top_cut_*_offset_mm` are still the old-tool values (staged/continue_top_cut only). `scan_results/top_only_20261013/`.
- 2026-10-14 **Top cut left -> right + carton yaw room** (lead, mock only): top cut runs seam_start -> seam_end again (`middle_cut_reverse_travel` false) with the sharp edge leading (`middle_blade_reversed` true); that needs a side lean: new planning param `middle_cut_tilt_deg` (YAML -15.0; 0 has no IK, -15 full seam). Mock execute OK (constant 15 deg lean, 2 mm deep). Carton yaw sweep (07:49 + 09:20 captures): left + top both plan for yaw -10..+45 deg (CCW +; +45 = sweep edge), best +25..+45 (left blade flat and on the line); top fails at <= -15, left poor/failing at <= -20. `scan_results/top_left_to_right_20261014/`.
- 2026-10-14 **Left cut corner-to-corner, keep what the arm reaches** (lead, mock only): the left cut now starts at the detected back_left corner instead of back_middle_left. YAML defaults: `left_parallel_orientation_enabled`, `left_parallel_varying_orientation_enabled`, `left_cut_extend_to_corners` true, `left_cut_corner_margin_m` 0.005 (min allowed), `left_cut_seam_cover_m` 0.025 (was 0.030). The planner keeps the longest reachable part of back_left+5 .. front_left-5 mm that crosses the tape +/-25 mm (>= 50 % of the nominal 139 mm); the stroke itself is still fraction 1.0. 07:49 + 09:20: full 268 mm, no trim, left_only and t_cut; mock execute OK. -25 deg carton: kept 159 mm (back 109 mm trimmed; 30 mm cover refused it). -30 deg: refused (reachable part misses the tape). Side effect: the staged/side-tape hand-off/continue/plan_t_cut paths refuse while the varying mode is on. Blade up to ~31-35 deg off flat, ~38-41 deg off the line at the corners. Rotation budget (09:20 = newest capture, + 07:49): left plans -25..+60 deg, top -10..+55 -> both cuts -10..+55 deg; blade flat +15..+50. Code fix in searchLeftVaryingOrientation: joint-travel (1.0 rad gate) failures now shorten the cut (30 node bans per end, then drop the end station; fallback restart with 2) instead of failing after 40 node bans; kMaxRepairs 500. Gates unchanged. `scan_results/left_from_corner_20261014/`.
- 2026-10-15 **t_cut full motion in mock + transit self-collision fix** (lead, mock only): t_cut now shortens the top seam like middle_only. Mock execute found a left->top transit passing knife_c_handpiece through base_link (JointInterpolationPlanner checked only every 0.1 rad); fix: max_step 0.01 + new fail-closed gate validateTrajectoriesCollisionFree (all stages, every waypoint + 0.01 rad in between). t_cut plans AND executes fully (mock) for carton yaw -10..+55 deg on the 17:20 and 15:49 (local) captures, except 15:49 +40 (refused: remaining transit exceeds the wrist gate). advisor-astra reviewed: no serious bug. Then `middle_transit_extra_lift_m` (YAML 0.05): straight raise before the left->top transit; planSnapshot tries lift 0.05/0/0.10 per top length and reuses the left search across retries -> 15:49 +40 also plans and executes; nothing else changed. `scan_results/left_from_corner_20261014/` (t_cut + safe transit sections).
- 2026-10-15 **Real t_cut offsets (lead)**: top cut ~5 mm too far from the base = perception scene `dual_arm_v1.yaml` (box-seam-detection) used `dualcal_20261009_left_single` instead of `dualcal_20261012_left_hovercorr` (+5.9 x / -6.3 y mm on every point); scene switched to hovercorr (src + install copy, uncommitted in that repo). Planner/arm were on the detected line within ~1 mm (FK of the iRC-logged joints). Then the left cut stopped cutting (blade exactly on the face): new planning param `left_cut_inward_m` (YAML 0.005, [0, 0.008]) shifts the whole varying left stroke into the face; main install rebuilt (includes the other session's WP0/2/3 source). YAML `middle_cut_clearance_m` -0.005 (operator). Mock execute OK. `scan_results/topcut_offset_20261015/`.
- 2026-10-15 **Right arm alone (lead)**: `tools/right_arm_single/` (builder) used on the real right arm (.3.12): face cut works (`plan face --inward-mm`, default 1 mm; 5 and 2 mm were too deep). Right TOP half does not plan at most placements (joint6 at its range end: the right arm's top stroke needs the knife turned 180 deg vs the left arm; flipped-blade approximation, tilts -15..+20 and reversed travel all failed on the 21:52 capture). Operator: right top = whatever it reaches from the right end, >= 5 %: `top_cut_min_cut_fraction` is now a planning param ([0.05,1.0]); mock 17:20 -> 80 %, 20:22 -> 15 %, 21:52 placement -> nothing. Perception once returned a half-top (one flap) with cut_permission blocked + override: the 'face' cut then ran along the top - check `extended span` (~270-280 mm) before executing. `right_cut.py` retries read-only queries (Humble first-request race with ROS_LOCALHOST_ONLY=0). `scan_results/right_arm_single_20261015/rerun_inward/`.
- 2026-10-15 **Left arm defaults + return home (lead)**: operator's live values made YAML defaults: `left_cut_inward_m` 0.007, `left_parallel_max_tilt_deg` 10 (flatter blade, steep back-corner start trimmed; on the 21:52 placement the left search then finds no interval covering the tape - later placements cut fine on the real arm). New planning param `return_home_after_cut` (YAML true): plan_h_cut ends with a 'transit to supplied home pose after the cut' to the SRDF home (wrist- and collision-gated); mock t_cut 202224 + 1720 executed, final joints 0.0. Main install rebuilt; install_right NOT rebuilt (right arm unchanged). `middle_cut_clearance_m` now -0.003 (3 mm deep, operator 2026-10-17; takes effect when the executor restarts).
- 2026-10-15 **Cut video recorder (builder + lead)**: `tools/cut_recorder/cut_recorder.py`, passive (subscriptions only, tested). One clip folder per trajectory execution (opens on the JTC action status, closes 3 s after the last goal): knife/scene(0.5)/wrist/right_wrist mp4 (compressed topics) + clip.json + snapshot copy, default `/media/farhan/My Passport/igus_nvidia/cut_videos/`. `move_group.launch.py` starts it with the executor on `hardware_protocol:=rebel` (`record_cut_video` default true, `cut_video_cameras`, `cut_video_dir`). Mock-verified only (domain 194, fake cameras); real cameras/wrist frames untested; ~1 core + ~50 MB/min recording all three. install_right not rebuilt. `scan_results/cut_recorder_20261015/`.
- 2026-10-16 **Left side cut yaw window, step 1 (lead, advisor-fable + advisor-astra, mock only)**: operator wants a wider carton-yaw window for the left side cut; steeper blade allowed only toward the END corner (steep start did not cut the tape). Today's t_cut window (tilt cap 10): 0920 0..+55, 165734 -15..+35, 195708 -5..+50 (real captures from cut_videos). Negative side = back/start corner needs tilt (cap 20/30/45 buys ~5 deg yaw per 10 deg); positive-side front trims are NOT tilt: joint3/joint2 exceed the 1.0 rad contact-travel gate on the ~270 mm stroke. Existing fixed-mode reverse travel covers -25..+20 where forward covers +20..+65; t_cut reverse -10..+20 (top cut fails <= -15). Next: P4 reverse travel in the varying mode (back corner becomes the end -> graded cap fits the operator rule); joint1-3 travel gate vs two strokes needs an operator decision. Plan/evidence `scan_results/left_yaw_window_20261015/` (PLAN.md, README.md). Nothing changed in code/YAML.
- 2026-10-16 **Left yaw window steps 1-4 (builder, lead-verified, mock only, built ONLY into build_yaw/install_yaw; main install untouched)**: true blade-angle gate `left_varying_max_theta_p_deg` (operator: 15 deg from the start through the tape band; checked per station, on the search replay and on the retained stroke, read-only tolerance 1 deg; measured overshoot 0.00), graded end cap `left_varying_end_max_theta_p_deg` (default 15 = off; end 30 only adds 10-20 mm at the front), travel check refactored into `left_contact_travel.hpp` (unchanged 1.0 rad), `left_cut_direction` forward|reverse + `left_reverse_frame` about_normal|about_edge. Forward = bit-identical to the baseline (66 rows, lead re-check); 71 gtests pass. Reverse flat-blade cut plans NOWHERE at the 15 deg cap: about_normal points the blade tip OUT of the carton with the handpiece over the top and needs theta_p >= 25 everywhere; about_edge (blade upside down) needs 45 and is 30 deg off the line (frames/*.png). Do not use reverse on the real arm. Not yet in the main install or the YAML. Next candidate: step 6 split travel gate (positive-yaw end trims). `scan_results/left_yaw_window_20261015/README.md`.
- 2026-10-16 **Left yaw window step 6: split joint-travel gate (builder, advisor-astra reviewed, fix round done, lead-verified; mock only, build_yaw/install_yaw only, main install and YAML untouched)**: opt-in `left_contact_travel_mode` aggregate (default, bit-identical: 66/66 vs base_t_cut) | split. Split, on every left entry/cut/finish/lift stroke, same pure check in the search replay and `validateLeftContactJointTravel`: j4/j5/j6 aggregate <= 1.0 rad, j1-3 <= `left_contact_arm_travel_max_rad` 2.0, constant Pieper branch (`rebel_analytic_ik::classifyBranch`), no j5 sign change, margins |j5| >= 15 deg, elbow |j3-beta| >= 5 deg, shoulder |X| >= 0.05 m (Lipschitz subdivision between waypoints); joints mapped by name, refused without the analytic-IK self-check, without `left_parallel_orientation_enabled`, or when no left contact stroke / cut stroke was examined. Result: positive-yaw end trims gone (0920 +15..+40, 165734 0..+25, 195708 +10..+35 now 253-273 mm, gains up to +107 mm); ok-yaw window UNCHANGED (35/66; negative side = arm reach to the seam band (see 2026-10-17), edges = top cut/transits). Measured: j3 up to 1.67 rad, min |j5| 49.3 deg, elbow margin min 5.7 deg (entry; 10 deg would shorten 195708 yaw 0 by 9.7 mm). 33+71 gtests pass. Not for the real arm until: main rebuild with the real stack stopped, then a plan-only run on the real arm, then a slow execute, both with operator approval. Advisor-astra also asked for checks of controller tracking and blade clearance. Operator: carton placement is random, so moving it closer is not an option.
- 2026-10-17 **Left yaw window, negative yaw cause (builder diagnostic, lead-verified, mock only, install_yaw)**: the negative-yaw limit is ARM REACH, not the blade cap (0 candidates removed by it): in the seam band 82-100 % of knife poses need the wrist centre beyond the 541 mm envelope (0920 short by 15/43/70 mm at -10/-15/-20, 195708 by 2/23/52 mm). `left_parallel_max_yaw_deg` 45 (bound max) gains NO negative yaw (extra branches collide handpiece/carton); on the positive side it lengthens 0920 +55 and 195708 +50 only. Fixed-mode upright-knife reverse rejected by the operator. New opt-in log-only `left_varying_diagnostics` (default off; t_cut 66/66 identical). Remaining levers are geometric only (holder geometry bringing the wrist closer); placement is random. `scan_results/left_yaw_window_20261015/README.md`.
- 2026-10-18 **Left yaw window, angle levers (lead, mock only, install_yaw, plan only)**: tilt/pitch grid 15 (step 5 or 2.5) with the 15 deg tape cap: no new yaw, <= 2 mm closer. Tape cap 20/25/30 deg (operator asked to test; NOT adopted, operator rule still 15): 20 gains nothing; 25 and 30 gain one 5 deg step per carton (0920 -5, 165734 -20, 195708 -10), left_only = t_cut. Right arm doing the left cut: skipped by the operator. Open: holder / base-position what-if.
- 2026-10-18 **DEFAULTS CHANGED, both arms (operator: "save it as default, I will try to cut the box later")**: YAML `left_parallel_max_tilt_deg` 25, `left_varying_max_theta_p_deg` 25, `left_varying_end_max_theta_p_deg` 25, `left_contact_travel_mode: split`. MAIN install rebuilt (no robot stack running): the real arms now run ALL uncommitted executor work (blade cap, split gate, cut direction, diagnostics). Mock (main install, as launched): left t_cut no losses, +3 negative yaws, longer cuts; right face 0749 +30 now ok, +35..+45 shorter (theta_p cap), 0749 -25 fails on the blade-collision approach independent of these settings. Before cutting: plan-only on the real arm first, then a slow execute, both with operator approval; if 25 deg does not cut the tape, revert to 15 / tilt 10. `scan_results/yaw25_default_20261018/README.md`.
- 2026-10-18 **Dual planner left standoff 12 -> 15 mm (lead edit, operator OK; builder studies, mock only)**: with the blade model the left hover TCP sits only standoff - left_cut_inward_m (7 mm) from the wall and the blade box (~6 mm ahead of the TCP) overlapped the carton -> left face planned 11/44 yaw cells. `per_arm_planning.launch.py` `BLADE_MIN_CLEARANCES_M` left_lateral_standoff_m 0.015, then 0.0165 after a 14..22 mm sweep (same windows as 15-16 mm, min clearance 1.77 mm, 17 mm loses 0749 -5; single-arm blade-less: slab inside the wall at 10 mm, ~13 mm break-even, 14 mm 1-3 mm outside with no window loss; `scan_results/standoff_sweep_20261018/`) (dual planner only; single-arm stack unchanged): 34/44, min free-space blade-box clearance 0.81 mm (0.26 mm with the 25/25/split YAML; 16 mm gives 1.45 mm, same windows). Combined two-arm windows (yaw -10..+15, 25/25/split, main binary 10-18 10:18): current right holder -5..0 / -5..0 / none / -7.5 (strict-check fail) for 0920/0749/165734/195708 (right top half = joint6 limit); MIRRORED right holder -5..+12.5 / -5..+12.5 / -10..+12.5 / -7.5..+15, strict check 0 collisions, >= 160 mm arm-arm. Top-half lean 25 removes the random wrist-flip refusals (44/44); lean is an operator decision. Side finding: the blade-less single-arm planner puts the nominal blade slab 0.5-2.1 mm inside the wall at the left hover pose (model only, depends on TCP). `scan_results/left_face_blade_20261018/`, `scan_results/dual_window_15mm_20261018/`.
- 2026-10-21 **Dual plan videos + left-face trim cause (builder, lead-checked, mock plan-only, domain 219)**: `scan_results/dual_plan_video_20261021/` (videos/, pngs/, README). Overlays install_sag + install_seq, production overrides; 5 snapshots (real 12:34/10:07/10:03 of 10-20, captures 09:20/07:49) x phases/sequence = 10 plans ok. Left face trims the BACK end on all 5 (kept 176-225 of 278-284 mm corner-to-corner; front trim 0; tape crossing still covered); left top 90-100 %, right face full. Cause: the 25 deg blade caps (`left_varying_max_theta_p_deg`, `left_varying_end_max_theta_p_deg`, `left_parallel_max_tilt_deg`, 10-18 defaults) + `left_cut_inward_m` 0.007 push the back stations out of reach; caps 45 + inward 0 (10-14 settings) give the full face on all 5 (07:49 267.8, 09:20 268.6 mm = the 10-14 numbers) with theta_p up to ~40 deg (the steep-start-does-not-cut-tape issue of 10-16). Cap 35: full only on 12:34; cap 40: ~10 mm trim on 3, 39 mm on 10:03. Operator chose 40 (2026-10-21): set in arm_left_executor_overrides.yaml only (right arm keeps 25); not yet run on the real cell.
- 2026-10-21 **Left-face reach: sweep + start-cap ramp (advisors fable/astra; builder x2, lead-checked; mock plan-only, domain 219)**: `scan_results/left_reach_sweep_20261021/` (README, run/results_table*.md, videos/compare_startcap/, pngs/theta_profile/). No-code: inward 6 mm gains 0 mm, station spacing 5 mm gains <= 5 mm (1.5-2.3x planning time) -> not adopted. NEW CODE (uncommitted, src/igus_rebel_moveit_config; built ONLY into overlay `install_startcap`, main install not rebuilt): planning params `left_varying_start_max_theta_p_deg` (0 = off = old behaviour; else [band cap, 45]) + `left_varying_start_ramp_m` (0.05): steep cap at the stroke start, linear down to the band cap at the seam-band entry, band cap enforced through the band (also by the retained-stroke validator); 19 gtests in test_left_blade_cap, package tests pass. Mock (5 snapshots): start-cap off == production exactly; start 40/band 25/end 40 == production (the band cap never binds; band theta_p 3-19 deg); start 45/band 25/end 40/tilt 45: full face on 10:07, 09:20, 07:49 (12:34 already full), 10:03 254.5 mm (trim 19.6), start theta_p 42.1 deg, band max 20.3 deg. Operator 2026-10-21: SKIPPED (gain minimal) -> production stays caps 40/40/40, inward 7 mm, spacing 10 mm; the start-cap code stays in src with default 0 = off (no plan change), overlay install_startcap unused.
- 2026-10-21 **Bug fixes + MAIN install rebuilt + real stack on domain 11 (builder x2, lead-reviewed)**: (1) dual_cut_system.launch.py: robot IPs now default to the driver's 192.168.3.11/.12 (driver include scoped in a GroupAction, empty IPs unset; 9 new pytest); (2) RViz Plan&Execute safety gap closed: dual_cut_system passes `allow_move_group_execution:=false` (new arg) -> global /move_group plan-only (rebel_dual_table_moveit_config move_group.launch.py new arg `allow_trajectory_execution`, default true = standalone/friend setup unchanged); mock: execute_trajectory ABORTED, move_action plans only; tools/wrist_handeye uses /execute_trajectory -> needs allow_move_group_execution:=true; (3) executor: left_debug_stop_at_approach + left_contact_travel_mode split no longer refused (pure `expectedLeftStrokes`, approach-only expects no stroke; normal plans unchanged; mock evidence scan_results/approach_only_split_20261021/). dual_cut_bringup pytest 120 pass (colcon test needs PYTEST_ADDOPTS="-p no:anyio"; test_per_arm_planning_launch.py not registered in CMake). MAIN rebuilt 09:33 (igus_rebel_moveit_config, dual_cut_coordinator, dual_cut_bringup, rebel_dual_table_moveit_config; old binaries in ~/igus_backups/main_install_before_rebuild_20261021/): main now has sag top cut, coordinator retry/skip, start cap (off), the 3 fixes -> install_sag/install_seq/install_startcap/install_bugfix* overlays are obsolete (delete after asking). All gtests pass. Real stack started by the lead as systemd user unit `dual-real-stack-20261021` (ROS_DOMAIN_ID=11, LOCALHOST_ONLY=0, use_rviz, IPs explicit): both hw active, trajectory controllers INACTIVE (operator activates with dual_arm_recover.py activate); stop with `systemctl --user stop dual-real-stack-20261021`. 2026-10-22: right arm faulted 10-21 14:20 (joint 2 'Motor not enabled' right after enable) -> both latched; on 10-22 04:41-04:58 the operator launched dual_cut_system 6x in a terminal WHILE this unit ran (two /controller_manager on domain 11 fighting over the same CRI connections: watchdog drops, spawner 'already loaded', activate 'command interfaces not available'). Lead stopped the unit 10-22 ~05:00; operator runs the stack from the terminal. Rule: only ONE real stack per cell; check `systemctl --user list-units 'dual-real*'` and `ps` before launching.
- 2026-10-22 **Operator config changes (lead, YAML only, no rebuild; take effect on stack restart)**: left face caps back to 25/25/25 (arm_left overrides; expect the 49-98 mm back-end trim again) + left face `left_cut_inward_m` 8 mm (the max; +2 mm start extra tried and reverted); right face `cut_clearance_m` +0.002 tried then back to 0.0 (operator; +0.002 also lifted the right holder clear of a 0.46 mm table hit at the right face entry for the 10-22 placement x 0.392 y -0.227); right face depth `left_cut_inward_m` 4 -> 2 mm; left face cut_clearance stays -0.003 = the executor cap (operator asked 2 mm lower: needs a cap code change); top cut `middle_cut_clearance_m` -0.003 -> -0.005 in the shared executor YAML (both arms, the cap; + sag 2 mm at the centre). LATER 10-22 (operator): left face `cut_clearance_m` -0.006 (6 mm below the top; executor side-penetration cap raised 3 -> 6 mm in validateSettings, main igus_rebel_moveit_config rebuilt, old binary ~/igus_backups/box_h_cut_executor.before_cap6_20261022); right face `cut_clearance_m` -0.003 and `left_cut_inward_m` 0.0 (operator: "3 mm shallower" from 2 mm, clamped at the planner minimum 0). Lower right face = right holder 3 mm closer to the table (it already hit by 0.46 mm at cut_clearance 0 for the x 0.392 / y -0.227 placement). EVEN LATER 10-22 (operator): left face `cut_clearance_m` -0.0075; top cut `middle_cut_clearance_m` -0.006 (shared YAML) with `middle_cut_sag_m` 0.001 on both arms (start/end 6 mm, centre 7 mm); executor caps now side 8 mm / middle 6 mm (main rebuilt again, gtests pass; mock domain 226: both executors start and read back the values). Right face: cut_clearance -0.0075 (4.5 mm lower than -3); still too deep at inward 0 -> arm_right_base y +5 mm more in dual_arm_calibration.yaml (backup before_rightface5_20261022.yaml.bak; also moves the right top cut 5 mm towards the right end, overlap with the left ~5 mm less). Transit unwind detour UPDATE: round-2 fixes done (5 mm floor incl. self, strict carton ACM, stopped joins, dense controller-spline validation, lift-line check, deadline); advisor-astra round 2 still NOT SAFE for physical execution (no retime/dynamics check, sampling not a bound, dual check lacks 5 mm). Operator chose empirical validation: scan_results/unwind_fallback_20261022/validation/ - 13 detours: peaks 0.159 rad/s / 0.72 rad/s^2 = 0.20x / 0.14x raw limits, 1.01x / 1.37x of the 0.20/0.10 transit scaling (direct transits identical 1.37x: quintic overshoot of bang-bang accel, pre-existing), joint6 spline variation <= 2.262 rad, clearance wrist leg >= 7.1 mm in full dual model w/ table; mock execution OK (tracking <= 2.8 mrad); Gazebo single-arm replay OK (no tolerance violation). MAIN igus_rebel_moveit_config rebuilt with the detour, DEFAULT OFF (enable at runtime: ros2 param set /arm_left/box_h_cut_executor transit_unwind_detour_enabled true); old binary ~/igus_backups/box_h_cut_executor.before_unwind_20261022. Earlier note:  executor code in src (transit_unwind.hpp, planDetour) makes the left face->top transit plan via lift + wrist rotation in mock (scan_results/unwind_fallback_20261022/), but advisor-astra review = NOT SAFE as is (clearance floor can drop to ~0, carton contact ACM can leak into the detour legs in cut-debug/side-tape starts, concatenation joins not validated for the controller interpolation, path constraints/timeout not enforced) -> default set to false in src, main NOT rebuilt with it; fixes pending. Live placement x 0.376 also fails on the right face approach (right holder vs table, from the -7.5 mm right height). Right top finish now +15 mm (coordinator.yaml 0.015; executor accepts a marked derived seam end up to 10 mm + extension past the top face along the seam, pure helper snapshot_top_face.hpp + 7 gtests, main rebuilt, mock scan_results/endext2_20261022/ 15.0 mm reached, rest unchanged; side points still 10 mm -> 15 mm fails if the detected seam end is > ~5 mm past the edge). Earlier: Right top finish +5 mm: new coordinator param `top_split_right_end_extension_m` (builder, lead-reviewed; left_max only, [0, 0.02] m, executor refuses seam_end > 10 mm outside the top face -> usable ~9 mm) set 0.005 in coordinator.yaml, main dual_cut_coordinator rebuilt (old binary ~/igus_backups/dual_cut_coordinator_node.before_endext_20261022); mock scan_results/endext_20261022/: right end exactly +5.0 mm, everything else unchanged. Placement note: 10-22 carton was ~40 mm off the 10-20 positions (x 0.43-0.44, y -0.19..-0.20).
- 2026-10-20 **Mocap re-check (lead, mock only)**: cri-move-joint `rebel_mocap_follower` + `rebel_servo_relay` installs match their source (10-18); tests 66 pytest + 39 relay gtest pass. Mock domain 218 against TODAY's main dual stack (dual_cut_system mock, main install + only the two worktree packages via their `package.bash`, see `scan_results/mocap_first_real_20261020/mock_env.sh`): both arms followed the dual sine (max error <= 15 mrad); right NaN -> only right HOLD 0.3 s, left kept following; deadman release -> both DISABLED, arms still. Still NEVER run on the real arms. First real test: `arms:=left`, `params_file:=scan_results/mocap_first_real_20261020/mocap_follower_dual_slow.yaml` (max/catch-up 0.1 rad/s), left to SRDF `ready` first, coordinator must be running (silent coordinator = hold), no RViz/cut execute during mocap (follower can't see MoveIt action goals).
- 2026-10-20 **Top-cut lid sag + coordinator retry/skip (builders, lead-reviewed) - OVERLAYS ONLY, main install NOT rebuilt**: executor `middle_cut_sag_m` (planning, [0,0.006], default 0 = unchanged; profile s = sag*max(0,1-(d/h)^2) from seam_center + corners, polyline stroke, straightness gate +sag) in `install_sag/` (built 11:46 from current src; mock: profile error <= 0.013 mm, executed; 8 new gtests); coordinator `plan_retries` 2 (random executor refusals only), `top_split_skip_right_top_fraction` 0.995 (sequence mode: right does only its face when the left reaches the whole seam), sag-aware tip-on-line, in `install_seq/` (100 gtests). Operator ran the stack meanwhile, so both are used by sourcing install_sag + install_seq AFTER install/ in the launch terminal; rebuild main with the stack stopped later (overlays carry COPIES of their YAMLs). Overrides now: both arms middle_cut_sag_m 0.002, left middle_cut_clearance back to the YAML -0.003, left face 7 mm.
- 2026-10-20 **Sequence mode: hold before penetration INSTALLED 09:44 (builder, lead-checked, 94 gtests)**: the inserted delay now holds just before job 2's first ENTRY (left: face -> transit/align/dwell over the top-cut start -> hold -> descend + cut). Mock margin 5.0: hold 15 s, makespan 79 s (same as the old insertion point), measured min cross 157 mm, skew max +1.2 s, no trips. coordinator.yaml default sequence_skew_margin_s 5.0 (2.0 tripped the skew gate on the real arms: right segment 7 approach >2 s late); schedule_mode default still phases. Real cell 2026-10-20: one sequence run (margin 2) stopped by the skew gate (safe stop). Operator overrides today: right face cut_clearance -0.001, left_cut_inward 0.004; left top middle_cut_clearance -0.005 (lid sags ~2 mm mid-seam; snapshot seam_center is always the exact midpoint of start/end, so no sag info). Rollback binaries scan_results/sequence_mode_hold_before_entry_20261020/rollback/.
- 2026-10-20 **Dual cut video recording (lead)**: `cut_recorder.py --coordinator-status-topic /dual_cut_coordinator/status` records while the coordinator is WAITING_BARRIER or EXECUTING (one clip per execute, both arms, phases and sequence modes); `dual_cut_system.launch.py` starts it on hardware_protocol:=rebel (args record_cut_video true, cut_video_cameras, cut_video_dir; post-roll 5 s). Tests: recorder 32, dual_cut_bringup 106; functional check on domain 214 with a fake status (clip opened/closed, trigger coordinator); not yet run with real cameras. Also: right face `cut_clearance_m` -0.0015 in arm_right_executor_overrides.yaml (operator: right face too low).
- 2026-10-20 **phase_order side_first INSTALLED (builder, lead-reviewed, 93 gtests)**: default now `phase_order: side_first` in coordinator.yaml (A = left face + right top, B = left top + right face, no return home between phases). Mock (capture_0556, left_max): A parallel 41 s, B serial (left first; parallel missed by 0.14 mm) 51 s, total 93 s vs 132 s top_first; min cross measured 97 mm; coverage left 0..85 %, right 82..100 %. `top_first` = previous. Also in the binary but OFF and NOT mock-run: `schedule_mode: sequence` (segment-window all-pairs check, skew gate 2 s, per-segment goals; advisors: fixed time band rejected; before a real run: tube/ stop-distance measurement, slow first run). Sequence MOCK RUN done (builder, domain 214): left job 2 delayed 9 s (left top vs right face otherwise 0.0497 m < 0.05 padding), makespan 73 s (side_first 93, top_first 132), executed with no skew/monitor trip, max tracking 0.013 rad, measured min cross 157 mm; dispatch drift ~0.09 s per segment (right +1.15 s after 13 segments, margin 2 s) -> raise margin or merge DWELL segments before longer plans. `scan_results/sequence_mode_20261020/`. Rollback binaries scan_results/side_first_20261020/rollback/. Gotcha (builder): changing an executor planning param after it planned clears its plan and detected_carton.
- 2026-10-20 **OPERATOR REQUIREMENT (next, not started): independent arm sequencing**: each arm runs its own cuts back to back with no return home between cuts and no waiting at a phase barrier (e.g. right: short top cut, then directly its right face; left: left face then top). Today: each phase ends with return_home_after_cut and both arms wait for each other; mock time ~131 s (phase A serial ~102 s: left top to 85 % and right face -> knife holders -6 mm in the all-pairs swept check; phase B parallel ~29 s); free-space transits ~60 % of each phase. Needs time-aligned cross-arm validation with a tracking-lag margin instead of the whole-phase swept volume (safety design change -> advisors + mock first), one continuous plan per arm, and execution without phase barriers.
- 2026-10-20 **Right top cut in the LEFT arm's direction (operator idea; lead; real arm OK)**: the right top from the right end reaches only 5 % (carton y -0.202) / 80 % (09:20 spot) because the knife must face the travel direction -> turned 180 deg vs the left arm -> joint6 at its range end. Same direction/orientation as the left arm, started mid-seam (mock): start 50 % -> ~75 %, 70 % -> ~94 %, 77-80 % -> right end. `right_cut.py plan top --start-fraction F` (trims the converted snapshot's seam, keeps its mtime, sets middle_cut_reverse_travel; 36 tests) - REAL right arm at --start-fraction 0.77: start, end and depth correct (operator). Operator rule (restated): left top = as far as it reaches, right = only the remainder; coordinator `top_split_mode left_max` (default) + fail-closed top coverage check (builder, lead-reviewed; 66 gtests incl. 20 new): left top as far as it reaches, right top from left reach - 12 mm to the right end in the left direction via a derived snapshot (~/.ros/dual_cut_coordinator/right_top_snapshot.json, source mtime kept); mock case a: left 0..85 %, right 82..100 %, executed OK. INSTALLED into the main install 07:31 (robot stack stopped); rollback binaries in scan_results/top_split_left_max_20261020/rollback/; `top_split_mode: half` = old behaviour. Not yet run on the real cell.
- 2026-10-20 **Real dual cell: right e-stop + recover PROVEN (operator)**: right e-stop latched both arms (left LED stays green: software cross-stop, left controller never e-stopped); status showed both hw unconfigured with controllers still ACTIVE (the 10-15 case); `recover --root right`: right drives enabled on attempt 2 (Motor not enabled mask 0x63 on attempt 1), left on attempt 1, controllers re-activated, status healthy, rearm OK, both arms move. Launch bug: dual_cut_system.launch.py declares left/right_robot_ip default "" which overrides dual_table defaults -> pass both IPs explicitly (fix after the session). Safety gap: RViz MotionPlanning Plan&Execute on the dual /move_group can execute on the real controllers, bypassing the coordinator (rejected today only because controllers were inactive) -> disable execution in the dual move_group. Right face depth in dual: `arm_right_executor_overrides.yaml` left_cut_inward_m 0.001 (right face cut OK at 1 mm after the base correction).
- 2026-10-20 **Right face too deep on the real arm; pause-and-measure flow (lead, mock-verified domain 214)**: right_arm_single face cut was too deep at --inward-mm 1 AND 0 (0 = blade tip on the detected face; the planner range is [0, 8] mm), so the error is the right-arm position (right base calibration), not a setting. Pause flow: `right_cut.py plan face --inward-mm 0 --set left_debug_stop_at_approach=true --set left_contact_travel_mode=aggregate`, `debug_motion_execution_enabled true`, `right_cut.py execute` -> model tip exactly 10.0 mm in front of the detected face at the back end of the stroke; measured gap g gives the error d = 10 - g; the full cut then plans and executes from the pause pose (split, approach-only false). BUG (other session's executor, not fixed): left_debug_stop_at_approach is refused with left_contact_travel_mode split ("no left contact stroke was found to check") -> use aggregate for the approach-only plan. Side-tape hand-off route not usable for the right arm (needs a side_tape_seed block; read-only side_tape_handoff_box_offset_base_mm [0.6,-12.3,-8.4] is a LEFT-arm correction). Fix plan: shift arm_right_base y in dual_arm_calibration.yaml by +d (right face outward = -y world). `tools/dual_run/check_placement.py` (read-only placement check) added. APPLIED (operator measurement at the pause: blade tip ON the wall, 3 mm above the seam): `dual_arm_calibration.yaml` arm_right_base y +10.0 mm, z +3.0 mm (converter check: right face moves 9.94 mm outward and 3.19 mm down in the right base frame); backup `dual_arm_calibration.before_rightface_20261020.yaml.bak`; affects the dual stack too (right top half 3 mm lower). Next: re-run the pause (expect ~10 mm gap), then cut at --inward-mm 1.
- 2026-10-14 **Dual-arm control system plan v2** (Plan agent + advisor-fable + advisor-astra + operator answers; not started, nothing built): one coordinator owns execution/STOP, executors plan-only per arm (wrapper route), swept-volume rule (tracking lag ~100 mm >> calib error), target = simultaneous cutting in crossed phases (A: left top half + right side, B: left side + right top half), software cross-stop (no hardware e-stop wiring: fault group + coordinator + STOP button), joint limit stops only that arm. Right arm has holder C installed. `scan_results/dual_arm_control_plan_20261014/PLAN.md` (phases P0-P6). Right base now from the hand-eye calibration (operator: the tape/Isaac layout is sim-only): `rebel_dual_table_description/config/dual_arm_calibration.yaml` = calib2 right base xyz (-1.3, -454.3, +4.8) mm, rpy (-1.04, +0.48, +2.81) deg + hovercorr camera; old nominal file kept as `dual_arm_calibration.nominal_20261006.yaml.bak` (untracked); still calibrated_unverified until right-arm touches (P2).
- 2026-10-14 **Dual-arm stack build (mock only, nothing committed, no real motion)**, evidence `scan_results/dual_arm_stack_20261014/` (design: `design/P3_P4_DESIGN.md`):
  - P0 (builder): shared prefixable `igus_rebel_description/urdf/knife_c_tool.macro.xacro` (single-arm URDF semantically identical, gate `p0_model/scripts/compare_single_arm_urdf.py`); dual model knife_c on both arms, SRDF derived from igus_rebel2.srdf by `make_srdf.py`; camera pole in the dual move_group.
  - P0b/P0c (builder): dual-only real-mesh collision hulls for base/link1-6/motor1-5 (under-coverage up to 88 mm -> 0), 6 'Never' pairs re-enabled, dual-only blade box `<prefix>knife_c_blade`. Home arm-arm distance 222 mm.
  - P1 driver (builder): JOINT_LIMIT_MIN/MAX no longer trips the fault group (only that arm); STATUS ESTOP field only logged (semantics unknown - check log on the next real e-stop press). Installed: the real driver loads it on its next restart.
  - WP4 `dual_cut_bringup` (builder): per-arm namespaced plan-only move_group + executor + joint_state relay; left per-arm plans equal single-arm. Right arm misreads the snapshot until WP2 (frame transform) - its plans are invalid.
  - WP1/5/6 `dual_cut_msgs` + `dual_cut_coordinator` (builder): swept-volume validator (FCL error measured 0 mm on our model, 2 mm tolerance, sampling-gap term), tip-on-line + frame gates, crossed phases, direct FJT goals, 20 Hz monitor, STOP (one-point hold + cancel, 30 ms in mock), T1 confirmed (move_group does not stop the other arm). Tested against a fake plan provider.
  - 2026-10-15 WP0/WP2/WP3 done (builder) in `box_h_cut_executor.cpp` (+653/-8, YAML unchanged), built ONLY into isolated `build_p3/` + `install_p3/` (main install still the old executor; rebuild main only when the real stack is stopped): snapshot_frame_mode fixed_transform + snapshot_relabel rotate_180 (require plan_export_only), middle_seam_stop center + overlap, ~/get_planned_solution, ~/plan_h_cut_from_state (override plans not executable). Regression vs baseline PASS (t_cut/left_only bit-identical; middle_only within its own run-to-run envelope, KDL 7.7e-5 rad), evidence `scan_results/dual_cut_p3_regression_20261015/`. Right arm in mock: top half and right face lie on the world lines within 0.01 mm; right face with the extended/varying default fails on 09:20 (joint3 aggregate limit) -> right-arm reach sweep running; WP7 coordinator integration with real executors running.
  - 2026-10-15 WP7 (builder): coordinator + REAL per-arm executors end-to-end in mock (domain 204), both captures: phases A and B parallel, tip-on-line <= 0.014 mm, measured arm-arm min 168-216 mm, planning ~70 s (right face), snapshot age at phase B start ~110 s (limit 240). Right base read from the calibration YAML only (launch). BLOCKER found: single-arm model has no blade collision -> per-arm approach transits push the dual blade box up to 5.2 mm into the carton; strict world check refuses phase A (WP7 runs used a temporary 6 mm tolerance - mock only, never real). Fix in progress: opt-in `blade_collision` xacro arg (default off, identity gate) + executor `carton_contact_links` + per-arm planners use both; built into isolated build_p3b/install_p3b. Same blind spot exists for the real single-arm left arm (decision for the operator later).
  - 2026-10-15 right reach (builder): right face = left_only under rotate_180 with `left_cut_corner_margin_m: 0.04` (5/5 per capture, 14-17 s, tip 0.005 mm on line; yaw -30..+45 except 07:49 +30); right top half (middle_only+center, tilt -15) works ONLY for carton yaw <= 0 deg (joint6 limit margin 0.015 rad at 0, 0.17 at -5, 0.31 at -10) -> combined two-arm yaw window ~ -10..0 deg. Operator decision pending: place the carton ~5 deg clockwise (from above) vs a different right-top tilt. Overrides file no longer carries the base (launch injects it from the calibration YAML). Evidence `scan_results/dual_arm_stack_20261015/right_reach/`.
  - 2026-10-15 blade collision (builder): opt-in `blade_collision` xacro arg in the shared knife_c macro (default off: single-arm URDF byte-identical), executor read-only `carton_contact_links` (default = [tool_link]); regression PASS (baseline rebuilt because the executor YAML changed meanwhile: middle_cut_clearance_m -0.005, left_cut_inward_m 0.005); left arm with blade on: same plans/times. Strict dual run 09:20: 2/3 pass, 1 refused (blade box grazing at the 10 mm hover/standoff) -> per_arm_planning.launch.py raises hover_clearance_m/left_lateral_standoff_m to 0.012 when blade_collision is on (lead edit). per_arm `blade_collision` default still false until the main install has the new executor (built only in build_p3b/install_p3b). Evidence `scan_results/dual_arm_stack_20261015/blade_collision/`.
  - 2026-10-15 05:30 MAIN INSTALL SWITCHED (lead, operator OK): igus_rebel_description, dual_cut_msgs, igus_rebel_moveit_config (executor source sha f3a6e691..., incl. the other session's left_cut_inward_m), rebel_dual_table_*, dual_cut_bringup, dual_cut_coordinator, dual_teleop built into build/ + install/ from a clean shell. Checks: single-arm URDF identical (both variants), executor gtests 4/4, mock (domain 207) 6/6 plans; with left_cut_inward_m=0 t_cut/left_only are BIT-IDENTICAL to the pre-change baseline (5/5 runs, both captures); middle_only within its envelope; the only left-arm difference is the other session's intended left_cut_inward_m 0.005. Old binary kept at scan_results/dual_arm_stack_20261015/main_install_switch/box_h_cut_executor.main_before. per-arm blade_collision default now TRUE (12 mm hover/standoff). Real Terminal 2 must be restarted to load it.
  - Operator 2026-10-15: right controller IP 192.168.3.12 confirmed (left 192.168.3.11). Ultrasonic knife: switched MANUALLY for now; later by the robot (digital output, per-arm service /igus_rebel_<arm>/set_digital_output) -> coordinator knife on/off per phase is a later task.
  - 2026-10-15 right-arm hover test (single_right, session ~/dual_arm_calibration/right, left-arm TCP, 1 touch per board crosshair 1-4, no stickers): predicted vs touched 8.2/12.5/15.0/12.5 mm lateral, systematic (dx -1.7..-6.8, dy -8.0..-13.4, dn 0.2..2.3 mm). Cause undetermined (right knife TCP vs right camera fit) - needs tilt-diverse touches or a right-arm pivot. Operator will move the table and redo a full session (both arms, home-made fit/check stickers). Old left touches moved to ~/dual_arm_calibration/left/touches_20261012_oldTCP.jsonl.bak. Real stack ran with ROS_LOCALHOST_ONLY=0 on domain 11 and the camera dropped to ~1.5 Hz with 'sequence size exceeds remaining buffer' (probably another LAN host on domain 11); keep all real terminals consistent, preferably ROS_LOCALHOST_ONLY=1.
  - 2026-10-15 operator: the RIGHT arm's knife was mounted FLIPPED (180 deg about the knife axis; blade point ~5 mm to the right) during that hover test - explains part of the ~12 mm offset. Recommended: remount it like the left arm (holder fits both ways) because the right-arm planner design (rotate_180 relabel, shared knife_c model, sharp edge leading) assumes identical tools; then right-arm pivot + caliper check. All board-position-1 touches/table images were moved to *.bak in ~/dual_arm_calibration/{left,right}. Operator caliper: right blade exposure (seat face -> tip) 14.3 mm vs left 14.4 mm (2026-10-12).
  - 2026-10-15 RIGHT-BASE HOVER CORRECTION applied (operator: left TCP for the right arm, no pivot, no stickers). Right hover tests (board crosshairs 1-4 after the knife flip + board move; entries in ~/dual_arm_calibration/right/hover_test.jsonl, full log .bak) re-predicted through the DUAL chain (YAML camera + right base): mean touched-predicted (-1.2, +0.5, +3.1) mm, z significant, residual rms 4.6 mm (the hover tab's 10-13 mm is against the right-only fit, not the dual chain). `dual_arm_calibration.yaml` arm_right_base xyz -> (-0.08, -454.72, +1.68) mm, rpy unchanged, still calibrated_unverified (padding stays >= 20 mm); previous file `dual_arm_calibration.handeye_calib2_20261014.yaml.bak`; evidence `scan_results/right_base_hovercorr_20261015/compute.py`. NOTE: 'Record offset' on the Hover test tab does not write calibration touches (touches.jsonl).
  - Done 2026-10-15: `dual_cut_bringup` one-command `dual_cut_system.launch.py` (hardware_protocol required, execution off) + `dual_arm_recover.py` (status/recover/activate) + README. `dual_teleop` (builder): per-arm MoveIt Servo on the dual model with arm selector (console l/r/e, gamepad LB/RB), starts disabled, blocked while the coordinator executes or a controller is inactive; Servo collision check ON for cross-arm + scene (stops ~33 mm before the other arm in mock) but same-arm pairs are allowed (Servo has one self threshold), so same-arm self-collision is NOT protected while jogging. Touch mode (console t+y, gamepad both stick clicks 1 s, service): selected arm's blade/handpiece/holder/tip may touch detected_carton + table at 1/6 speed, auto-off on arm switch/disable/coordinator activity/inactive controller/120 s; after OFF a retreat window (80 mm / 60 s) also allows link6 + wrist camera vs carton/table so the knife can be jogged out. Blade reaches ~0.9 mm above carton, contact on table. 56 tests.
- Prints: the table board now goes on **A4 landscape**, picked automatically. 3D-printed ribbed backing plates are in `cad/calibration_plates/`; the STL/STEP files are in `~/dual_arm_targets/`. How the arm plates mount to the arm is still open (finger tab or bolt boss).

**Physical prerequisites:**
- arm_right has been moved to 192.168.3.12 (operator, 2026-10-08); `right_robot_ip` already defaults to it in `dual_table.launch.py` (equal IPs are refused).
- Wire the two controllers' e-stops together.
- Remove the knives for commissioning.

**Next steps (advisor gate order):**
1. Rebuild the driver.
2. Run arm_right alone at low speed and check its gains.
3. Add collision padding, the camera pole and the carton table.
4. Run the dual launch with controllers inactive, then confirm arm identity by moving each arm from its pendant and watching `/joint_states`.
5. Record and solve the calibration, and pass the gate.
6. Do a small first dual move, then e-stop one arm while the other crawls, to prove the group stop live.

**Open:**
- Any kinstate other than NO_ERROR (including joint limits) trips the group. Whitelisting joint-limit states is a later decision (`Safety::IsGroupTrippingKinstate`).
- Real kinematic accuracy is unknown. With 0.4° joint offsets, nominal-FK tip error is about 7 mm (synthetic), and the gate would FAIL. In that case, adjust the joint zero offsets in iRC, or pad about 10–15 mm.

## 2026-09-29 Servo teleop changes (mock only)

With `run_teleop:=true`, MoveIt launch now starts Humble Servo via a bounded one-shot
service client. Keyboard and gamepad still start Servo when needed but skip a duplicate
call if `/status` is publishing. Humble's second start returns success yet restarts
the calculation thread; the status guards are best-effort when independent clients
start simultaneously. Keyboard twist and joint jog keys republish at 50 Hz for 0.3 s,
with fresh stamps; a new key replaces the command and resets its deadline. The
increment service remains available.

Build passed on Humble. Domain 240 required a temporary Fast DDS XML portBase=1000
(default port computation overflows); all launched ROS processes used
`ROS_LOCALHOST_ONLY=1` and `mock_hardware`. Servo `/status` emitted 0 without
manual start and after MoveIt restart. From a nonsingular mock pose, a 5 mm keypress
sent 15 commands in 0.3 s and moved the TCP 2.1-2.2 mm (not the nominal 5 mm);
four keys spaced 0.15 s moved ~4.8 mm with 38 published commands. No commands
after the deadline, and sampled TCP travel after stop was 0 mm in 0.2 s. Joint
jog republishing yielded 15 commands and joint motion. The 5 mm displacement
criterion remains unmet; do not infer physical step accuracy from mock results.

2026-10-11 keyboard teleop "too fast / floaty" (mock-verified, real arm not yet):
joint jog was a fixed 1.0 rad/s (= the 0.785 rad/s limit) that `-`/`=` did not change;
now 0.1 rad/s default (0.01-0.4, scaled by `-`/`=` with the Cartesian speeds). Floaty
cause: when a command ended Servo went quiet and the JTC kept the last point *with its
nonzero velocity*, so the velocity loop (ff 0.4, p 1.0) crept 0.4*v past it. The keyboard
now streams zero commands for 0.2 s after every command, and a held key (auto-repeat
< 0.1 s apart) only extends the command by 0.12 s, so release stops within ~0.12 s; a tap
is still 0.3 s. Mock (domain 187): travel after release 7.6-8.3 deg -> 0.2-0.6 deg, last
Servo point velocity 0. The gamepad teleop has the same Servo/JTC path and was not changed.


## 1. Current state (2026-09-27)

- Branch `hcut-path-consistency`, HEAD `2eb8bd0` (last commit). The P0/P1/P2 hardening
  round is complete in the working tree as **uncommitted** changes on top of it; the
  operator decides when to commit. `git diff 2eb8bd0 --stat` lists them.
- What the round changed, by area:
  - **CRI driver (`src/igus_rebel`)**: `Override 100.0` sent after `MotionTypeJoint` on
    every (re)connect; 50 ms stale-`write()` watchdog zeroes ALIVEJOG; jog clamped to
    +/-100 %, NaN -> 0; `on_init` waits up to 5 s for the first STATUS and fails
    otherwise; `read()` returns ERROR (jog forced to zero, driver restart required) if no
    valid STATUS for >200 ms or after any reconnect; socket keepalive/timeouts; framing
    split into `CriFraming.cpp`, safety arithmetic into `RebelSafety.cpp`;
    GetKinematicLimits request removed. New gtests `test_cri_framing`,
    `test_rebel_safety`, `test_rebel_driver` (real driver against a fake CRI server on
    loopback). `controller.yaml`: JTC constraints (trajectory 0.25 rad, goal 0.02 rad,
    goal_time 3.0 s, stopped_velocity 0.05 rad/s), D gains 0,
    `rebel_arm_velocity_controller` removed. `rebel.launch.py` accepts `isaac_topic`;
    the default `hardware_protocol` is still `rebel`.
  - **H-cut executor**: parameter table with read-only / planning / runtime scopes
    (read-only `param set` fails; planning changes rejected as busy during calls,
    execution and cut-debug sessions); `clear_plan` answers busy instead of waiting;
    `debug_allow_stale_snapshot` blocks every execute of a plan/session loaded with it;
    correction records must contain `planning_frame` and `tool_link`;
    `external_snapshot_max_age_s` capped at 600 s (YAML 240 s); all doubles must be
    finite; `t_cut_handoff` auto search probes 0.95 first, then 0.05 + bisection, with
    the left roll search once per request; `left_roll_diagnostics_enabled` param;
    cut-debug stages `start_planned -> start_executed -> start_saved` (no endpoint
    execution, no save retains a plan); wrist guard fails closed on a transit-stage count
    mismatch; `NearestBranchTransitPlanner` retries the original pose goal once and shares
    one `planning_timeout_s` budget; scalings rescaled to contact 0.1376/0.0954 and
    transit 0.4127/0.2863 (same physical speed as before); `cut_pattern` default `t_cut`.
  - **MoveIt config**: joint limits 0.785 rad/s / 5.24 rad/s² (igus/CPR); Pilz
    `cartesian_limits` rescaled (0.01817 m/s, 0.0524 m/s², 0.0727 rad/s); URDF velocity
    limits 0.785; `allowed_start_tolerance` 0.02; SRDF collision matrix regenerated by
    sampling (17 adjacent + 59 never disabled, 29 checked incl. link2-motor4); no virtual
    joint; carriage joints fixed; Ruckig evaluated and not enabled; KDL timeout 0.25 s
    with a commented TRAC-IK block; `move_group.launch.py`: `run_teleop` default false,
    `run_gamepad`, `run_camera_pole`, namespace arg removed; teleop nodes call
    `start_servo` on Humble; keyboard increment default 5 mm, max 30 mm; camera pole
    re-applied every 5 s; simulation uses the XEG wrapper.
  - **Other**: `compal_box_perception` temporal filters removed (one-shot snapshots
    report the current normal), `snapshot_max_receipt_age_s` 1.0, `detect_tape_seam` is
    now `ros2 run compal_box_perception detect_tape_seam --image ...`, `colcon test` runs
    pytest; `handeye_pose_replay` refuses `wait_for_enter` without a TTY;
    `igus_rebel2_robotviewer.urdf` removed; meshes decimated; carriage collision mesh
    `meshes/xeg32_collision/right_carriage.stl` (convex hull); Dockerfile defaults to
    Humble, container `ros2_rebel_dev`.
- Docs (README, CLAUDE.md, XEG32_MOVEIT_CHANGES.md, CONTEXT.md, SoftwareArchitecture.md,
  handeye README and the three HTML guides) were re-checked against this code on
  2026-09-27.
- Operating defaults are unchanged: `execution_enabled`, `debug_motion_execution_enabled`
  and `cut_debug_correction_enabled` are `false`. The YAML left correction slot
  points at the orientation-only derivative
  `/home/farhan/cut_debug_corrections/left_capture_frame_000001_20260916_235239_389784_orient_only_tcp20260924.json`
  (taught blade angle, geometry-derived positions; the original 2026-09-16 record's
  translations are invalid on the current TCP).
- **Knife camera (new package `knife_camera`, 2026-09-27)**: Logitech C930e knife-view
  webcam via `usb_cam` (apt `ros-humble-usb-cam`, `v4l-utils`), launched separately with
  `ros2 launch knife_camera knife_camera.launch.py` (not wired into `move_group.launch.py`).
  Publishes `/knife_camera/image_raw` (+ `/compressed`, `/camera_info`), 1280x720 MJPEG
  -> rgb8 at 30 Hz, frame `knife_camera_optical_frame`, uncalibrated (the knife tip is a
  fixed pixel; camera and knife move together). Manual focus from
  `config/knife_camera_controls.yaml` (`focus_absolute: 110`, tuned by the operator with
  `ros2 run knife_camera focus_tune.sh`), applied with v4l2-ctl because usb_cam 0.8.1's
  `focus_auto`/`exposure_auto` names fail on kernel 6.x; auto exposure/WB kept,
  `exposure_dynamic_framerate` off. Crosshair overlay at the knife-tip pixel is deferred.
- **Camera topics per arm + gripper cameras (2026-10-20)**: `cameras.launch.py` now publishes
  `/arm_left/wrist_camera/...` (was `/wrist_camera`) and `/arm_right/wrist_camera/...` (was
  `/right_wrist_camera`); TF frames are unchanged (`wrist_camera_*`, `right_wrist_camera_*`;
  orbbec takes topics from the launch arg, frames from the config `camera_name`). New USB
  gripper cameras (05a3:9230, no serial, chosen by USB port): `/arm_left/gripper_camera`
  (port 1-1.3; 1-1.2 gave USB errors -32/-71 after a reboot) and `/arm_right/gripper_camera` (port 1-1.1), raw_mjpeg 1280x720 at 30 Hz,
  ~1 % CPU each, measured on domain 173. Consumers updated (calibration UI, mock wrist camera,
  tools/wrist_handeye, cut_recorder, camera_sync_check). Not yet verified: the renamed 305
  topics and their TF frames on a fresh launch (the running launch was left alone).
  Historical text below uses the old topic names.
  The C930e knife webcam was removed from `cameras.launch.py` the same day (arg `knife`
  gone); start it with `ros2 launch knife_camera knife_camera.launch.py` if needed.
  Startup race fixed: usb_cam scans all /dev/video* at start and aborts if an Orbbec node
  vanishes mid-scan (USB reset / driver claim), so the gripper cameras start first and the
  Orbbecs wait 2 s (`GRIPPER_SCAN_DELAY_S`). No `respawn` on them: with Ctrl+C launch
  respawned them forever. 4 clean start/stop cycles on domain 173, all cameras at rate.
  2026-10-22: gripper auto-recovery: `scripts/gripper_camera_supervisor.py` (tests:
  `test/test_gripper_camera_supervisor.py`, 10 pass) restarts usb_cam on child exit, device
  path gone/changed, or camera_info stall. Verified on domain 187: SIGSTOP stall -> restart
  -> 30 Hz; Ctrl+C clean, no restarts. Not yet verified: physical unplug/replug of a gripper
  camera (USB reset needs root: no udev rule for 05a3:9230) and Orbbec hot-replug recovery.
  2026-10-26: after a reboot the hub sits in PC port 3, so the gripper configs are now by-path
  `usb-0:3.3` (left, unverified guess) / `usb-0:3.2` (right); the launch warning lists where gripper cameras
  are plugged in. The 336 (hub port 2-3.1) dropped off USB every ~20 s whenever it, the left 305 (2-3.4)
  and both gripper cameras (1-3.x) streamed together; any two of them were stable for 75 s, so it looks
  like hub power/bandwidth. Wrist 305s now default to off (`wrist:=true right_wrist:=true` to enable);
  scene + grippers ran 75 s with no drops. Re-enabling the 305s needs the 336 or a 305 moved off the hub.
- **Cameras (`rebel_cameras`, 2026-09-27)**: `ros2 launch rebel_cameras cameras.launch.py`
  starts the Gemini 336 (`/camera`, serial CP96563000LY), Gemini 305 (`/wrist_camera`,
  CV27561000NP) and the knife webcam. **2026-10-15:** two 305s, one per knife:
  `/wrist_camera` = left arm, CV27561000GA (name kept for the left hand-eye/tools);
  `/right_wrist_camera` = right arm, CV27561000NP (`config/right_wrist_camera.yaml`, launch
  arg `right_wrist`, USB-reset + 4 s stagger; 15.0/15.1 Hz color/depth on domain 167). Measured together on an isolated domain: all at
  target rate (336 15 Hz, 305 15 Hz, C930e 30 Hz); CPU about 20 % / 16 % / 8 % of one core;
  stamp->receive latency 63 / 36 / 7 ms; ApproximateTimeSynchronizer sets within 21 ms max.
  Findings: (1) the 336's colored point cloud costs about one full core and runs at about
  4 Hz while anything subscribes (1.66 M points at 1080p); the node builds it only on
  demand. (2) The 336 temporal filter halves depth noise (1.46 -> 0.74 mm median);
  RGB color format and the hardware noise filter gave no gain; `uvc_backend:=v4l2` fails
  on both Orbbecs. (3) The 305 (fw 1.0.70, orbbec_camera 2.9.3) hangs on every driver start
  except the first after a USB reconnect; the launch USB-resets it first. (4) The apt
  usb_cam 0.8.1 stamped knife frames 0.3-0.9 s early (tv_usec/1000 bug); fixed by the
  vendored `src/third_party/usb_cam`. No hardware trigger between cameras; the two
  Orbbecs free-run, so their phase offset is arbitrary per start (up to 33 ms at 15 fps).
  Open: the 305's intended use (only a top view so far; no hand-eye TF, so its frames
  are not connected to the robot tree).
- **Camera performance for the capture/detector (2026-09-28)**: the 336 itself holds 15 or 30
  fps; the bottleneck is the colored cloud `/camera/depth_registered/points` (1.7 M points,
  34 MB). While anything subscribes to it (the detector's `rgbd_recorder.py`, or an RViz
  PointCloud2 display added by hand) the driver runs at ~130 % CPU, the cloud arrives at
  ~4 Hz and every /camera topic is ~0.6-0.7 s behind capture (vs ~50 ms without it).
  30 fps does not help (cloud stays ~4 Hz; a Python consumer only keeps up with ~13 pairs/s).
  Measured options: `point_cloud_decimation_filter_factor: 2` -> cloud 9.4 Hz, 85 ms, 4x
  fewer points (changes the detector input); `align_mode: HW` is not supported for
  1080p color + 1280x800 depth. HW D2C probe (15 fps): accepted only with depth <= 848x480
  (any color up to 1920x1080); every 1280x800 / 1280x720 depth pair is rejected. HW
  1920x1080 + 848x480 gave no gain while the cloud is subscribed (driver ~124 % CPU, cloud
  ~1.4 Hz, same ~0.6 s lag; depth +2 mm median vs current, p90 24 mm at edges) because the
  cost is building the 1080p colored cloud, not the alignment. GPU (RTX 5070 Ti, torch
  cu128 installed): depth->xyz+rgb back-projection 9.4 ms vs 100 ms numpy, identical
  results. Verified: the driver cloud equals the aligned depth image
  back-projected with the color K (same 1,710,913 points, 0.000 mm difference), so the
  recorder can build an identical PCD from `/camera/depth/image_raw` and the driver never
  builds the cloud. Prototype timings (scratch, not committed): per-capture process with
  depth path ~0.55 s vs ~0.75 s now; persistent capture node 167 ms trigger->cloud with
  115 ms frame age. Detector code (`~/Downloads/seam_handoff_20260908/scripts`) not changed.
- **Detector recorder speed-up (2026-09-28, other repo, uncommitted)**: in
  `~/Downloads/speed_wt/seam_handoff_20260908` (branch `perf/speed`) `rgbd_recorder.py`
  gained `depth_topic` (builds the PCD from `/camera/depth/image_raw` + color K, numpy or
  torch/CUDA; output equals the driver cloud: same order, <0.4 um, rgb identical) and a
  persistent `--serve` mode; `capture_and_query_depth.py` defaults to `--depth-topic
  /camera/depth/image_raw --recorder-mode serve` (`--recorder-mode once --depth-topic ""`
  restores the old path). Hardware (isolated domain): old per-attempt capture ~1.0-1.2 s;
  serve median ~230-260 ms (min ~150 ms, tail up to ~1.2-1.5 s from Python deserializing
  1080p rgb+depth); manifests pass CQD's loader. Not yet run end-to-end with detection.
  Once adopted, set `enable_colored_point_cloud: false` in scene_camera.yaml.
- Camera calibration: both `calibration_verification.launch.py` and `box_perception.launch.py`
  default to `calibration_name:=rebel_handeye_h_v2` (newest, eye-on-base, base_link ->
  camera_link = -0.0524, -0.3063, 0.4521 m), the same transform as the mock fixture in §6.

- **Parallel left orientation (2026-09-27, opt-in, mock only)**: new planning parameters
  `left_parallel_orientation_enabled` (default `false`), `left_parallel_max_tilt_deg` 45,
  `left_parallel_max_yaw_deg` 30, `left_parallel_step_deg` 5,
  `left_parallel_min_joint_margin_deg` 5, `left_parallel_search_timeout_s` 90. It derives the
  left orientation from the cut basis, minimizing the blade-plane angle to the carton top
  face (`theta_p`, see CONTEXT.md) subject to the penetration gate, collision-aware IK and
  a jump-free collision-checked left chain from every hover-start wrist branch with joint
  margin; fails closed with an active left correction record. Replay carton: tilt -10,
  pitch -5, yaw -10 deg, `theta_p` 11.2 deg (nominal 80, taught 27, old +45 roll 58),
  t_cut/t_cut_handoff/combined plan in mock. `NearestBranchTransitPlanner` now ranks
  goal branches within 2 deg of a URDF bound after the others (after the joint6 rule).
  Evidence `scan_results/parallel_left_20260927/` (offline grid `grid_run`, `refine_run`;
  executor runs `run3`, `run3/summary.txt`, `run3/regression_vs_joint_limits_20260927.txt`).
  Not checked on the robot: blade pose, contact point, cutting.

- **Gripper-mounted tooling collision geometry (2026-09-28, mock only)**: added
  collision (and simple visual) meshes for the ultrasonic knife handpiece
  (`xeg32_knife_handpiece`), its holder (`xeg32_knife_holder`, from
  `JK-Final version.step`) and (briefly) the wrist camera + bracket, all fixed to `igus_tool0`; added to the
  `xeg32_gripper` SRDF group and end effector, with the self-collision matrix
  regenerated (see XEG32_MOVEIT_CHANGES.md). The scoped tool/carton contact
  allowance is unaffected (it only ever applies to `xeg32_tool_tip`). Finding: the
  wrist camera collided with `detected_carton`/`camera_pole` for most cut patterns (it
  protrudes past the blade tip along the knife axis); the operator uninstalled it and
  its link was removed the same day. Without it the regression matches the
  pre-tooling results (all nominal patterns, parallel left t_cut/handoff at theta_p
  11.17 deg; `scan_results/tool_collision_nocam_20260928/`); handpiece and holder
  cause no regressions. Evidence `scan_results/tool_collision_20260928/` (regression
  summary, SRDF regen sampling, render PNGs).

## 2. Verified in mock (evidence)

All under `scan_results/`. Mock/fake-server evidence only.

- `p0p1_regression_20260927/final/` — clean build and tests (`colcon_test_result.txt`:
  146 tests, 0 errors, 0 failures, 15 skipped); `test_rebel_driver` passed 5 of 5
  repeated runs (`test_rebel_driver_5x.log`) and in parallel runs; URDF checks for
  default/mock/gazebo (TCP `0.005937 0.094121 0.130699`, no tool-tip collision, all mesh
  URIs resolve); final H-cut regression `compare_final.txt`; a mock
  `FollowJointTrajectory` dispatch that finished `SUCCEEDED` (`mock_trajectory_check.log`);
  launch smoke tests `smoke_*` (move_group, teleop, motion planner, calibration). The
  `final/` directory was still being written by the final verification pass when this
  handoff was written; re-read it before relying on it.
- `p0p1_regression_20260927/diff_report.txt` and `final/compare_final.txt` — 14 planning
  scenarios (t_cut nominal / left orientation-only / with roll diagnostics, t_cut_handoff
  auto and manual 0.5, h, combined, ompl_pilz, pilz_only, cut-debug left/middle start,
  section diagnostic) match the pre-round `baseline/`: same target poses, stage lists and
  joint6 travel; nominal t_cut 24.41 s. Only `t_cut_handoff` auto differs (handoff
  endpoint 0.72 mm further along, because 0.95 is now probed first). Negative scenarios
  n1/n2 (mismatched correction slot) still fail closed.
- `p0p1_regression_20260927/after/p0_checks.json` (and `final/p0_checks.json`) — read-only
  parameters rejected, NaN/inf rejected, planning changes rejected as busy during a long
  plan, `start_cut_debug_session` refuses a 3600 s old snapshot, the stale override
  blocks execution, `execution_enabled` ends false.
- `p0p1_regression_20260927/moveit_config/` — `a_default` (new limits without rescale:
  t_cut 59.2 s) vs `h_final` (rescaled: 24.41 s, identical to before) shows the rescale
  keeps physical speed; `b_ruckig` (Ruckig: no change to H-cut plans); `srdf_regen`
  (collision-matrix sampling); `d_teleop` (keyboard `Servo started`, increment 0.01
  accepted, 0.05 rejected; gamepad `start_servo succeeded`); `e_motion_planner`,
  `f_calibration`, `g_sim` (launch smoke tests).
- Earlier work (2026-09-26): `hcut_wrist_branch_20260926/` (nearest-branch transit: the
  corrected t_cut plans 5/5, left-to-middle joint6 1.66 rad; old planner failed 2/5),
  `hcut_left_orient_only_20260926/`, `hcut_left_roll_diag_20260926/` (taught left angle
  unreachable on the replay carton, +45 deg roll selected), `cleanup_regression_20260926/`,
  `hcut_path_consistency_20260926/` (nominal t_cut planned and executed on mock hardware,
  RViz video and joint bag).

## 3. Not verified on real hardware

- The whole reworked CRI driver: `Override 100` acceptance, first-STATUS seeding,
  watchdog, stale-STATUS/reconnect ERROR path, keepalive. Only tested against a fake
  CRI server.
- Trajectory tracking with `Override 100` and the new JTC tolerances; the PID gains were
  tuned on 2026-09-09 at an effective override that was probably lower (see §4 step 6).
- The 0.785 rad/s / 5.24 rad/s² limits and rescaled scalings on the arm (only shown to
  produce the same planned timing in mock).
- Any physical cut, the left orientation-only pose, the TCP Z value (URDF comment: held
  pending a stale-model check), blade clearance (`xeg32_tool_tip` has no collision
  geometry), cutting force.
- Hand-eye replay and calibration verification at the new (slower) speeds.

## 4. First real-robot checklist (driver)

Operator present, explicit approval in the conversation before each motion step.

1. E-stop within reach and tested; workspace clear; only one CRI client connected (close
   the igus software's control session and any other driver).
2. Prefer the wired LAN (default `robot_ip` 192.168.3.11; this PC needs an address on
   192.168.3.x). Launch the driver alone and pass the protocol explicitly:
   `ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel`
   (Wi-Fi alternative: add `robot_ip:=192.168.50.133`; measured Wi-Fi latency spikes
   up to ~160 ms can trip the 200 ms stale-STATUS stop, so avoid Wi-Fi for motion).
3. In the log confirm: `CRI controller target <ip>:3920` and `Connected to ReBeL at <ip>:3920`;
   `First STATUS: joints [deg] ...` with joint values matching the pendant; the override
   values reported by `Robot velocity override is X %` (and in `First STATUS`) before
   and after `Sent CRI 'Override 100.0' ...` — note the value before, it tells what the
   override was during earlier sessions; `Rebel accepted command: ...` for SetActive,
   Reset, Enable, MotionTypeJoint and Override; finally
   `Robot velocity override is 100.0 %`.
4. **Stop** if Override is rejected (`Rebel did not accept command: Override ...`), if the
   override stays below 100 % (repeated warning), if any command is rejected, or if the
   First STATUS joints disagree with the pendant.
5. `ros2 control list_hardware_components` shows `Igus::Rebel` active; `/joint_states`
   matches the pendant.
6. **Loop-gain risk.** The 2026-09-09 tuning bags (`~/rebel_controller_tuning/`) suggest
   the override was about 37 % while the PID gains were tuned. With `Override 100` the
   effective gain may be about 2.5x higher. First motion: one joint, 5-10 degrees, at
   velocity/acceleration scaling 0.05 (for example the MoveIt RViz sliders), watching for
   overshoot or oscillation. Stop and reduce the P gains in `controller.yaml` if it
   overshoots.
7. Ctrl-C the driver with the arm still and confirm the arm stays put and the log shows
   the jog stopping (`Hardware deactivated: jog set to zero` and/or
   `Stopped to send ALIVEJOG`).
8. Optional, arm still and E-stop ready: pull the Ethernet cable and confirm
   `No valid STATUS from ReBeL for ... ms: read() returns ERROR` and that the arm does not
   move. After plugging the cable back in, motion must not resume on its own: restart the
   driver (ros2_control stops the component after the ERROR).
9. Only after the above: MoveIt, then (separately approved) H-cut planning with
   `execution_enabled=false`, and inspection in RViz.

### First real-robot driver check (2026-09-27, LAN, no motion)

Driver-only run on the real controller (iRC V980-15-000-2, "igus-REBEL/REBEL-6DOF-03",
gripper config `knivetool1_clear.xml`): connected to 192.168.3.11:3920 (LAN ping 0.09 ms
avg); SetActive/Reset/Enable/MotionTypeJoint/Override 100.0 all accepted; override was
already 100 % (so the Sep 9 tracking ratio is not explained by override); all six axes
enabled; `/joint_states` matched the controller's first STATUS exactly (velocities 0);
~15 min idle with no stale-STATUS/read() errors; Ctrl-C deactivated with zero jog and
exited cleanly. Follow-ups: the controller reports `LowVoltErr=DIO (0x04)` (digital-I/O
supply low, check the 24 V DIO/gripper supply); V15 CYCLESTAT sends decimals (parser
fixed); Stop() now logs Disable/Quit. No motion commanded yet.

### First real-robot motion tests (2026-09-27, LAN, operator at the E-stop)

- `tools/single_joint_test.py`: every joint moved +5 deg over 6 s and back from home;
  all goals succeeded with +-0.02 deg final error, smooth, no tolerance/joint errors.
- Joint3 driven with the MoveIt joint slider toward 145 deg: the controller stopped it
  at 140 deg ("Joint limit max exceeded: A3 requested 140.0 allowed 140.0") and the JTC
  aborted the goal on goal_time (fail-safe worked). At 140 deg the link2 to motor4 gap
  is about 1.5 cm, so that pair cannot touch within the controller range; it stays
  collision-checked.
- Controller limits (CRI `CONFIG GetKinematicLimits`, `tools/cri_query_limits.py`):
  A1 +-179, A2 -80..140, A3 -80..140, A4 +-179, A5 +-95, A6 +-179 deg. The URDF had
  A2 lower -85 and A3 upper 150; all URDF limits are now 1 deg inside the controller
  limits (the controller also rejects a target exactly at its limit).
- Mock regression with the corrected limits (`scan_results/joint_limits_20260927/`):
  nominal t_cut, t_cut_handoff, h, combined, Pilz and cut-debug starts are unchanged.
  The taught-angle left correction (orient-only record) is now INFEASIBLE on the replay
  carton: the old passing +45 deg roll needed joint3 = 139..141 deg, beyond the real
  controller limit, so it could never have run on the robot. The left cut must be
  re-decided (see open questions). `approach_joint_degrees` joint5 94.37 -> 94.0.

### Left cut hand-over to the side-camera program (2026-09-28)

`tools/mock_hcut_demo.sh left_start` (and the same `start_cut_debug_session` +
`execute_planned_h_cut` sequence on the robot, with `cut_debug_cut left`, both execution
gates, optionally `left_parallel_orientation_enabled true`) moves only to the FIRST left
cut point: approach outside the left cut, 1 s dwell, lateral entry to the cut start with
the blade at cut depth; no cut stroke is planned. The operator confirmed this is the
correct stop point for handing over to the side-camera (side_tape) program's StartCut;
the script prints the left cut length (|back_middle_left - front_middle_left|, 55.7 mm
on the replay carton). Mock-verified (`scan_results/mock_demo_20260928_140620/`).

### First real left-cut approach (dry run, 2026-09-29)

Real robot, fresh snapshot, `left_start` (staged cut-debug start, parallel left pose
theta_p 5 deg) executed with the cut start temporarily raised to +10 mm
(`cut_clearance_m` 0.010, `hover_clearance_m` 0.03; both restored afterwards). All
three segments reached their goals without joint/tolerance errors, and the operator
confirmed the blade stopped ~10 mm above the left cut start at the right angle; the
side_tape knife camera locked onto the tape edge (its e_lat/e_yaw are uncalibrated
until its Jacobian calibration is run). The moves overshot at the end of fast
transits, so the free-space transit was slowed to `transit_velocity_scaling` 0.20
(~9 deg/s) and `transit_acceleration_scaling` 0.10 (0.52 rad/s^2); if overshoot
remains at this speed, review the JTC gains in `igus_rebel/config/controller.yaml`.

## 5. Open questions

- **Parallel left pose on the robot**: with the parallel mode the replay left cut runs at
  joint3 up to 133.8 deg and joint5 down to -87.7 deg (5-6 deg inside the URDF bounds) and
  the left-to-middle transit turns joint6 by 2.97 rad (guard pi). Blade model confirmed by
  the operator on 2026-09-28 (photo + answers): the tool is an ULTRASONIC knife whose
  handpiece is clamped upright in the XEG-32; the short blade's pointed tip is at the TCP
  pointing along tool +Z (up at home), its flat faces are normal to tool X, and the
  sharpened edge is the ANGLED end running from the tip toward tool -Y (toward the robot
  centre at home). The planner pushes along tool Z (tip first) and travels along tool
  -Y, so the sharp edge leads; theta_p = acos(|tool X . n|) measures the blade faces.
  As of 2026-09-28 the ultrasonic handpiece and its holder are in the collision model
  (`xeg32_knife_handpiece`, `xeg32_knife_holder`; see XEG32_MOVEIT_CHANGES.md); the
  wrist camera is uninstalled until the holder redesign. The Logitech
  knife webcam below the gripper is still not modeled (operator). Before any physical
  use: check in RViz, then a supervised dry approach without
  contact, then the blade/tape contact point. `left_parallel_min_joint_margin_deg` 8 gives
  `theta_p` 20.6 deg with 8.8 deg margin but the search then takes about 220 s (raise
  `left_parallel_search_timeout_s`). Parallel left mode was re-verified with the handpiece/holder geometry: t_cut and
  t_cut_handoff plan (28.1 s / 27.4 s, theta_p 11.17 deg).

- **Wrist camera for the redesigned holder (was: camera collision blocked most H-cut
  plans, 2026-09-28)**: resolved for now by uninstalling the camera. When the combined
  knife/camera holder is redesigned, keep the camera behind the blade tip along the
  knife axis, add its collision geometry from CAD and re-run the regression. History: adding
  `xeg32_wrist_camera` collision geometry (a conservative estimate from photos, no CAD)
  made the recorded replay carton's `t_cut`, `t_cut_handoff`, `h`, `combined` and
  cut-debug-left-start plans fail closed: the camera collides with `detected_carton`
  (and, for `h`/`combined`, also `camera_pole`) at the approach/cut poses, given the
  10 mm hover / 25 mm lateral-standoff clearances and the small (~102x111x200 mm)
  carton. Confirmed via a runtime AllowedCollisionMatrix override that
  `xeg32_knife_handpiece`/`xeg32_knife_holder` cause no regressions and that this is
  specifically the wrist camera, and that shrinking the camera box (from an initial
  90x100x145 mm down to the committed 70x70x70 mm body + 30x30x65 mm bracket) does not
  remove the conflict. Full regression table in
  `scan_results/tool_collision_20260928/summary.json` and the XEG32_MOVEIT_CHANGES.md
  finding. Open: either the real camera mount has more clearance than this estimate
  (needs a real measurement/CAD, not photos), or the hover/lateral clearances need to
  open up, or the camera needs to be relocated, before H-cut can plan normally with the
  wrist camera modeled. Do not loosen this collision check to make plans pass without
  resolving which of those is true.

- ~~ArUco marker size~~ resolved: designed 60 mm, prints at 59 mm; `marker_size_m: 0.059`
  (measured print) is correct. Re-measure after any reprint.
- **SRDF adjacency deviation**: `link1-link2`, `link2-link3` and `link4-link5` are kept
  disabled as `Adjacent` although a fixed motor link sits between them; justified in the
  SRDF comment (primitives overlap in most states) but not a strict Setup Assistant
  result. Review, ideally with finer collision geometry.
- **TRAC-IK** (optional): no Humble apt binary exists, so it would need a source build,
  then the commented block in `kinematics.yaml` and a mock regression. Not needed now:
  KDL plus NearestBranchTransitPlanner plans the corrected t_cut 5/5. `ros-humble-pick-ik`
  is available as a binary but deprecated by PickNik.
- **Deferred driver rework**: CPR Position Interface and lifecycle
  (`on_configure`/`on_activate`) restructuring wait until the operator can test on the
  robot.
- **Left cut on near cartons**: when the carton is about 10 cm closer to the base than
  the teaching carton, the taught left angle is unreachable (joint3 limit) and the search
  selects a 45 degree roll. Physically acceptable? Operator decision so far: keep
  nearest-feasible roll.
- Left pose on the current TCP: the operator prefers deriving it from the old teaching
  (orientation-only) over a physical re-teach; its blade pose has not been checked on the
  robot.
- `t_cut` (target) vs `t_cut_handoff` (fallback) for production: `t_cut` is the target.

## 6. How to run in mock

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=165 ROS_LOCALHOST_ONLY=1        # same in every terminal
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
# camera TF for the required camera_pole (mock only; not a substitute for live calibration)
ros2 run tf2_ros static_transform_publisher --x -0.052362891163214 --y -0.306337290082351 --z 0.452079756273593 --qx 0.022773271385773 --qy 0.373836805030297 --qz -0.079141815126831 --qw 0.923831150380609 --frame-id base_link --child-frame-id camera_link
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 --frame-id camera_link --child-frame-id camera_color_optical_frame
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false run_box_h_cut_executor:=true
ros2 control list_hardware_components   # must show mock_components/GenericSystem
ros2 param set /box_h_cut_executor external_snapshot_file /ABS/PATH/fresh_replay.json
ros2 service call /box_h_cut_executor/plan_h_cut std_srvs/srv/Trigger '{}'
```

For a historical replay, copy the snapshot (for example
`scan_results/hcut_path_consistency_20260926/source_snapshot.json`) with only
`acquisition_stamp_ns` refreshed to `time.time_ns()`; keep the source and provenance.
Never use a re-stamped snapshot to justify physical motion. `scan_results/p0p1_regression_20260927/launch.sh`
and `stop.sh` are a scripted version (domain 181). Mock execute: set
`execution_enabled true`, call `execute_planned_h_cut`, set it back to `false`; the
response means dispatch accepted, so also check `H-cut trajectory execution completed`.
Stopping `move_group` with Ctrl-C may print a segfault (known upstream MoveIt Humble
shutdown bug).

## 7. History

- 2026-09-05 .. 09-12: XEG-32 model, perception, plan-only H-cut executor, workspace and
  tilt scans (see `XEG32_MOVEIT_CHANGES.md`).
- 2026-09-09: controller PID tuning on the real arm (`~/rebel_controller_tuning/`).
- 2026-09-14 .. 09-16: staged cut-debug teaching sessions (records in
  `/home/farhan/cut_debug_corrections/`).
- 2026-09-24: TCP touch-corrected to `0.005937 0.094121 0.130699` (operator confirmed).
- 2026-09-26: wrist guard, transit MultiPlanner, `NearestBranchTransitPlanner`
  wrist-branch fix, left orientation-only record, transit scaling 0.15 approved, repo
  cleanup; commits `960db59`, `2eb8bd0`.
- 2026-09-27: P0/P1/P2 hardening round (§1), paused once for a PC restart and then
  completed; limits 0.785/5.24 with rescaled scalings, `Override 100`, SRDF regenerated;
  docs re-verified. Not committed.
- 2026-09-28: mock-only H-cut collision regression after removing the gripper-mounted
  `xeg32_wrist_camera` collision link; 19 isolated domain-206 scenarios are in
  `scan_results/tool_collision_nocam_20260928/summary.txt` with per-scenario
  executor/move_group logs. The knife handpiece/holder links remained loaded.
  Nominal T/H/combined and alternate-planner plans recover the earlier
  joint-limits-baseline metrics; left-parallel T-cut and handoff plan with
  theta_p 11.17 deg and max free-space joint6 travel 2.9695 rad. The five
  correction/left-roll failures remain; the negative diagnostic service still
  reports mismatched correction slots. This is planning evidence only: no
  trajectory was executed and no physical robot was contacted.

## 2026-10-20 Mirrored knife holder for the RIGHT arm (lead + builders + both advisors; CAD + mock only, nothing printed, real robot untouched)
- Part: `cad/mirror_holder_v1/` (README = print/assembly/operator checks; print package `final/knife_tool_C_R_print_v1_2026-10-20.zip`). Pure mirror of the 7 v4 parts about the flange axis plane + 3 deliberate changes: knife light/button windows re-cut at the knife positions, robot-side bore/recess made FULL ROUND (v4's is D-shaped; the mirror would put the D on the wrong side of the robot flange), debossed R + clocking arrow. Same checks as v4 (interference pairs, 37/37 fasteners, identical supports). Print the 5 non-Yahboom parts with the v4 slicer profile (not recorded anywhere).
- Mock (`scan_results/mirror_holder_20261020/README.md`): REAL right model `tool_variant knife_c_mirror` planned by the UNCHANGED executor with rotate_180 + `left_cut_direction reverse` + `left_reverse_frame about_edge` + `middle_cut_tilt_deg +15` (+ margin 0.005). Combined two-arm window 39/60 cells (0920 -5..+12.5, 0749 -5..+12.5, 165734 -15..+12.5, 195708 -7.5..+17.5) = cell by cell the earlier mirrored study; plans = study approximation to 0.04 mm; strict check 0 collisions, >= 161 mm arm-to-arm; current holder 8 cells, 2 strict fails. New right base (05:31) changes nothing for the mirror.
- Model patch NOT applied: `scan_results/mirror_holder_20261020/model/knife_c_mirror.patch` (defaults byte-identical). Deploy wiring (dual_table/move_group launch right_tool_variant, right planner tool_variant + mirrored overrides `data/arm_right_executor_overrides_native.yaml`, keep operator's left_cut_inward_m 0.001) is a mount-time step.
- Operator open: blade single-bevel? (edge to tool0 +y, prefer turning only the blade in its seat); robot flange spigot flat/key?; Gemini mounts upright, redo right hand-eye; after mounting FIRST right TCP pivot calibration, then re-check the 2026-10-20 right-base correction (made with the current holder + left TCP).
- Incident: another session's mock stack shared domain 231 05:10-05:16; affected data discarded and re-run. Mock domains below 231 also work; 214 is taken by the other session.

## 2026-10-21 Dual-cut operator web UI `src/dual_cut_ui` (lead + builder, advisor-astra reviewed; MOCK ONLY, real cell untouched)
- New ament_python package (calibration-UI style, rclpy + Tornado, 127.0.0.1 only, per-run token URL). Built ONLY into `build_ui/` + `install_ui/` (main build/install untouched; no other package edited; nothing committed). Real cell: `source install/setup.bash && source install_ui/setup.bash && ros2 launch dual_cut_ui dual_cut_ui.launch.py` in the robot's ROS environment, open the printed URL; `&view=present` = read-only projector view. README in the package.
- One page: cameras (lazy-subscribed CompressedImage topics, downscaled JPEG polling), one-click **Detect & plan** (runs `capture_and_query_depth.py --autolabel-v2 --allow-needs-review --allow-cut-blocked ...` as a subprocess, then `/dual_cut_coordinator/plan`), result KPIs + carton top view + detection overlay, operator **Enable execution** switch (sets execution_enabled + execute_phase_a/b_enabled true; auto-false on Software stop, run end, UI start/stop), **Cut** = hold-to-confirm 1.5 s + server re-checks (operator, PLANNED, plan_seq, snapshot age, controllers/hardware active, gates read back) then `~/execute`; post-dispatch guard trips Software stop if plan_seq changes. **Software stop** (any tab, never blocked by dialogs): coordinator `~/stop` + direct cancel-all on both trajectory controllers (+0.5/+1.5 s) + gates false + kill detector. Placement check (check_placement.py logic) is info only (operator decision). Hardware recovery is NOT in the UI.
- Tests: 325 pytest (whitelist, operator lock, execute checks, stop, races, jobs, cameras, report parsing, placement vs the real script, frontend smoke). Mock + Orca browser e2e on domain 173: `scan_results/dual_cut_ui_20261021/README.md` (screens/, mock/start_mock.sh). NOT yet run against the real cameras, the real detector or the real cell.
- Open: (1) coordinator `~/execute` takes no plan_seq -> UI guard narrows but cannot close a replan-under-confirmation race; a coordinator change needs operator approval. (2) UI does not explain a Cut that the coordinator barrier refused (state just returns to PLANNED; barrier_message is in the status) and still offers Cut on an already-executed plan (barrier refuses it). (3) Real-camera CPU unmeasured (mock: ~8-9 % of a core per viewing page at 10 Hz x 6). (4) Mock ROS domains 241-249 do not work (Fast DDS > 232 "port number too high").
- 2026-10-21 03:05 **Real scene camera + mock arms** (operator request; real stack live on domain 12 untouched): new UI params `camera_domain_id` / `detect_domain_id` (second rclpy context, camera subscriptions only; detector subprocess gets ROS_DOMAIN_ID). On this machine cameras = domain 11, dual stack = domain 12 -> real-cell launch needs `camera_domain_id:=11` (ROS_LOCALHOST_ONLY mix 0/1 still to verify on the real launch). Mock arms domain 173 + UI reading the REAL scene camera from domain 11: live view OK; REAL detector run from the UI (output to scan_results/dual_cut_ui_20261021/real_cam -> external drive, never the real snapshot): capture + SAM3 worked, scene REJECTED (`no_top_face`: carton flaps open), UI showed the rejection and skipped Plan. Real detect -> plan -> mock cut still to do with a closed taped carton. Camera container CPU 40.7 % viewing vs 39.5 % not; UI 8.2 % viewing the real 1080p/15 Hz scene, 1.6 % idle. Rejected-scene overlay is not shown in the UI yet (only `_reject.json` is written).
- 2026-10-21 09:25 **Real camera + new detector -> plan -> mock cut: full chain OK.** (1) The cameras were relaunched with ROS_LOCALHOST_ONLY=0; a fresh participant with 1 receives nothing from them (probe: 0 vs 119 camera_info msgs), so the detector timed out. New UI launch arg `camera_localhost_only` (-1 inherit / 0 / 1) sets ROS_LOCALHOST_ONLY for the camera context and the detector subprocess only (the UI node stays on its own setting). (2) The default command (seam_handoff `capture_and_query_depth.py`, rgbd_captures master fb2e3d8, Oct 12) rejects the taped curved carton (`no_top_face`, plane inliers 0.4-0.7); the new `box_seam_detection` CLI `box_seam_detect` (worktree `~/orca/workspaces/igus_rebel_ros2/box-seam-detection`, 0a4a319, other session's work, used read-only) accepts it (plane RMS 2.6 mm, seam yaw -90.3 deg). Test wrapper + command: `scan_results/dual_cut_ui_20261021/mock/detect_bsd.sh`, `detect_bsd.cmd`, restart script `start_ui_realcam.sh` (flags: --scene-config dual_arm_v1 --driver-extrinsic tf --capture-staging-dir off). From the UI: detect + plan OK (makespan 94 s est.), enable (confirm) + hold-to-cut on MOCK arms: A 42.4 s parallel, B 52.5 s serial, gates closed automatically; real seam_3d_base_link.json untouched. Screens 34-37. Also fixed: '-0' in top-cut ranges; static JS/CSS now revalidated + versioned (a rebuilt UI was served from the browser cache). 366 pytest. Open: switch the UI's DEFAULT detect command to box_seam_detect once that package is merged/installed (operator decision).
- 2026-10-22 top-down view (operator: the offset right-arm line looked like an unconnected cut): both arms' top cuts are now drawn ON the seam, the overlap striped blue/orange with its own legend line ('Both arms (overlap) a to b mm'). 367 pytest; screens/38_topdown_joined.png. A UI started before this needs a restart to show it.
- 2026-10-22 logos in the top bar (operator request): Hucenrotia Lab (`static/lab_logo.png`, downscaled from presentations/*/tpl/lab_logo.png) + Compal (`static/compal_logo.svg`, official header SVG from compal.com, no scripts) on a white plate left of the title, both views. 368 pytest; screens/39_logos*.png.
- 2026-10-22 camera on/off (operator request, replaces the knife removal): 'Cameras on/off' menu in the camera header, a checkbox per camera, remembered per browser (localStorage); usable by viewers and in the presenter view (view-only). Knife is back in the list but OFF by default (new launch arg `cameras_off`, default knife, `none` = all on). An off camera is never fetched, so the server drops its subscription after 10 s (verified: right_gripper subscribed=False). 374 pytest; Orca e2e with real clicks + reload; screens/41_cam_menu.png.
- 2026-10-22 **buttons now run the operator's verified real-cell commands** (builder implemented, lead reviewed + e2e): Detect = DetectBox goal `/box_seam_detector/detect_box {dry_run: false}` (new default `detect_mode:=action`; feedback in the job log; Cancel detect / Software stop cancel the goal; `detect_mode:=command` keeps the old subprocess). Plan = `param set schedule_mode sequence` (launch arg `schedule_mode`, default sequence, empty = leave) then `~/plan`. Enable = `execution_enabled` + `execute_phase_a_enabled` true only; every OFF path still writes all three false; Cut is blocked for a non-sequence (or fallen-back) plan while phase B is off. The UI must be started with `box-seam-detection/install/local_setup.bash` sourced (DetectBox type). Mock e2e domain 173 with `scripts/mock_detect_action_server.py`: action detect -> sequence plan (75.5 s) -> enable (read-back: exec true, A true, B false) -> cut 76.2 s -> all three false. 432 pytest; screens/42_action_sequence_done.png; restart script mock/start_ui_action.sh.

## 2026-10-21 Side-cut knife-camera probe `src/side_cut_vision`: Stage A, MEASURE ONLY (lead + builder; mock only, real cell untouched)
- Plan, decisions and spec: `scan_results/side_cut_camera_align_20261021/` (`PLAN.md`, `STAGE_A_SPEC.md`).
  - Operator chose measure only first, with the detection shown in the cut video. Corrections (Stage B: survey plus a cut-local executor offset) need a new go after real data.
- The "knife camera" is the Gemini 305 on each knife holder (RGB-D). S0 on the 10-20 clips: during the left side cut it looks DOWN at the top face near the edge. The detector therefore measures:
  - the top plane, giving height;
  - the edge depth step, giving depth and yaw;
  - the flap-gap seam under the clear tape, giving along;
  - the knife tip vs the seam crossing.
  - Output is Δ = measured − snapshot in the cut basis, with +across = outward (the OPPOSITE sign of the executor's left `cutDebugBasis` across).
- Package (ament_python), built ONLY into `build_vision/` + `install_vision/` (needs `install` as an underlay):
  - the library, plus `side_cut_probe` (passive: subscribes only; publishes `/side_cut_vision/<arm>/overlay/compressed` + `/measurement` JSON);
  - a LOOK record per still period at a side-cut dwell (color, depth, TF, snapshot, calib, measurement), written into the open cut_recorder clip folder;
  - 82 pytest.
- `tools/cut_recorder`: `--camera-domain N` (camera specs read in a second context) plus default `left_vision`/`right_vision` overlay specs; 37 tests. The running recorder (domain 12) was NOT restarted.
  - Today the cameras run on domain 11, so the domain-12 recorder records nothing: no clips since 10-20.
- Mock e2e domain 203 (Fast DDS refuses domains > 232; 250 tested). Evidence in `scan_results/side_cut_camera_align_20261021/mock/`:
  - the LOOK record recovered the hidden offset (8, −6, 3 mm, 2°) as (8.29, −5.88, 3.00, 1.83°);
  - `left_vision.mp4` is in time with `wrist.mp4`;
  - probe CPU ~30–40 % of a core while live, ~15 % idle.
- Real-cell use: see `src/side_cut_vision/README.md`. Cameras must run with `wrist:=true right_wrist:=true` (currently false).
- Open items:
  - the ROS_LOCALHOST_ONLY 0/1 mix (probe vs cameras) is unverified;
  - right calib = a seed from the left (flagged uncalibrated); a right hand-eye is needed after the mirrored-holder decision;
  - left hand-eye LOO is 5.2 mm rms, so the acceptance test is the S4 feeler gap/height check at the approach vs the camera Δ (≤ 2 mm on ≥ 3 placements per arm);
  - no real depth from the 305 has been seen yet.
- 2026-10-21 later: **probe and recorder camera domain integrated into `dual_cut_system.launch.py`.**
  - New args: `run_side_cut_probe` (auto = rebel only, true = also mock) and `camera_domain_id` (auto = 11 with rebel, the stack's domain in mock; or `same` / a number ≤ 232).
  - The recorder gets `--camera-domain`, and its default cameras now include `left_vision,right_vision`.
  - The probe starts at `coordinator_delay`. If `side_cut_vision` is not found, it is skipped with a log line.
  - 4 new launch tests (10 pass). Mock check on domain 203: the dual launch started `/side_cut_probe` next to the coordinator.
  - The installed launch file is a SYMLINK to src, so the NEXT real-stack start uses it (the running stack is unaffected). The recorder will then read the cameras from domain 11.
  - 2026-10-22: no probe data from the 29 dual runs 05:48-13:41 (launch logged "side_cut_vision not found"; wrist videos were recorded fine). Then, with operator approval and the dual stack down, `side_cut_vision` was built into the main `install/` (`colcon build --symlink-install --packages-select side_cut_vision`), so `source install/setup.bash` alone is now enough. `install_vision/` is redundant. Backup of the previous launch file: `/tmp/dual_cut_system.launch.py.before_probe`.
  - 2026-10-22 video audit of those 29 runs: `scan_results/side_cut_camera_align_20261021/video_audit_20261022/` (`harvest/` = mock re-plan + mock execute per run on domain 203, giving tool-tip/joint timelines; `analysis/REPORT.md` + `out/per_run.csv` = blade-vs-edge from color video, re-run with `nice -n 19 python3 analysis/tools/run_all.py`).
    - Observed blade tip pixel: L (908,420), R (905,387). The hand-eye+TCP prediction for the left is 38 px (about 8 mm) off along height.
    - At touch the blade is about 3 mm (L) / 4.4 mm (R) closer to the top edge than planned. Mid-cut heights are invalid: the detector locks onto the cut slit.
    - Open: ruler ground truth for the cut height on cut cartons.

## 2026-10-22 Planner fixes from today's 42 detections (lead + 4 builders; MOCK ONLY, NOT in the main install yet, real cell untouched)
Baseline `scan_results/today_plans_20261022/`: 25/42 planned (sequence). Fixes (uncommitted, built only into the overlay `install_ovl_combo`):
- A executor `snapshot_top_face.hpp`: with a top-split marker the 2 side midpoints of the extended right end are face-checked at their PRE-extension position (canonical midpoint only, fail closed). Fixes the 5 "outside the detected carton top face" refusals at 15 mm extension.
- B table in each per-arm scene (`dual_cut_bringup/scripts/table_collision.py`, launch args `table_in_arm_scene` (true), `table_padding_m` (0.010), base_link ACM-exempt only); executor probe scenes copy move_group's world objects + their ACM entries (`probe_scene_include_world_objects`, fail closed); coordinator world check gate `table_min_clearance_m` 0.010 (exact FCL distance, non-exempt links). Without the margin 6 new plans passed 0.4-3 mm from the table; 5 mm left 4 at 5.4-6.9 mm; 10 mm keeps all at >= 10.5 mm (`scan_results/table_margin_20261022/`).
- C coordinator `plan_fallback_chain` [sequence, sequence_left_top_first, phases_top_first, phases_side_first], `plan_fallback_budget_s` 150; only order-fixable refusals retry. Execute gate on the snapshot's `cut_permission`: blocked -> refused unless `allow_blocked_detection` (runtime, UI switch, auto-off). `honor_detection_override` false because the UI's detect command always passes --allow-cut-blocked (`scan_results/order_fallback_20261022/`).
- D UI (`dual_cut_ui`, info-only): left-face reach pre-check with move hints (flags exactly t090925/t093229 of today's 42), soft right-holder/table note, placement region updated to today's planned region (old 10-18 window kept as legacy), allow_blocked_detection switch (`scan_results/placement_precheck_20261022/`).
- Combined mock result (no margin): 40/42 (`scan_results/combined_fixes_20261022/`); the 2 left are the left-face reach placements. Final 10 mm pass: `scan_results/final_planner_fixes_20261022/`.
- 2026-10-24 08:29 INSTALLED with operator approval (dual stack confirmed down): main install `colcon build --symlink-install --packages-select igus_rebel_moveit_config dual_cut_bringup dual_cut_coordinator`, UI into `install_ui`; backups `~/igus_backups/main_install_before_planner_fixes_20261024/` (SHA256SUMS) + `install_ui_before_planner_fixes_20261024/`. Tests from the main build: 11/11 + 9/9 gtest suites, 139 + 487 pytest. Mock smoke from the main install (domain 209, `scan_results/main_install_smoke_20261024/`): t054601, t075123, t090346 plan. Report: `scan_results/final_planner_fixes_20261022/report/planner_fixes_report.html`.
- 2026-10-24 fallback planning speed-up (builder; INSTALLED in the main install 2026-10-24 09:59 with operator approval, see below): executor `top_cut_fraction_descent_stop_on_fraction_independent_failure` (default true, `top_cut_descent.hpp`): middle_only (left, not reversed) skips the 5 % steps after an approach/align/wrist-gate failure of the seam-start transit and tries the minimum fraction once; coordinator `plan_retry_breadth_first` (true) + `plan_inplace_retry_labels` ([right job 2 (right face)]): every chain entry once, then random-looking refusals round robin. Mock domain 209 (`scan_results/fallback_speed_20261024/`): the 6 fallback detections plan in 14.6-23.8 s (mean 18.8 s, n=21) vs 45.6 s same-session baseline (historic 40-143 s); 20/21 planned cells unchanged (variant, makespan +-1 s, table min); t074923_a refused once by a random `collision gate` refusal of the right approach in phases_top_first (not retried: `collision gate` is not in `plan_retry_signatures`, same as before), repeats a2-a5 pass. Tests re-run by the lead: 12 executor + 9 coordinator gtest suites, 139 bringup pytest pass. Known exposure: K=1 stop means a random approach failure on an easy detection would jump straight to a 5 % cut (0 of 375 easy steps so far).
- 2026-10-24 09:59 INSTALLED (operator approval; no ROS stack running): `colcon build --symlink-install --packages-select igus_rebel_moveit_config dual_cut_coordinator`; backup `~/igus_backups/main_install_before_fallback_speed_20261024/` (SHA256SUMS); new executor `c93cd16a...`, coordinator node `eef8d0b6...`. Plus (lead, operator asked): `plan_retry_signatures` += "collision gate: stage 'approach pose" / "'transition from" / "'transit to" (free-space transits only; strokes not matched, checked by builder), in `coordinator_node.cpp` default and `coordinator.yaml`; 6 `PlanRetryGate.*` tests (test_plan_policy 39/39, incl. a node/yaml drift test). All 21 gtest suites pass from the main build. Mock from the main install (domain 209, `scan_results/main_install_fallback_speed_20261024/`): t074923 x6 all plan (the earlier lost plan did not recur), 11/12 cells same variant/makespan/table min, t090925 still refused (reach); outlier t074923_a3 makespan 179.9 s (right arm took a long IK branch, phase B serial; valid plan, repeats 131.5 s). The collision-gate retry did not fire in mock (unit tests only). Note: a colliding approach at 5 % (carton) is not always random; its pass-2 retry costs one descent (2-8 s).
- Open: the `plan_fallback_budget_s`/UI `plan_timeout_s` raise is no longer needed (fallback plans < 25 s); table pose is tape-measured (setback 0.110, top z -0.001), a real-cell check of the holder-table distance on a close placement is still to do.

## 2026-10-24/26 CRI MoveJoint in the dual cell (lead; MOCK ONLY, not installed: waits for the real stack to stop)
- Driver merge (builder) into the working tree: src/igus_rebel, igus_rebel_description, igus_rebel_msgs from fani8j/cri-move-joint (b22ad59 + 3e335da driver parts only; no servo relay / mocap / executor WIP). Hand-merged mode switch keeps `controllerRestartHold_` + the branch's claim-during-move stop/latch; new rule: move_joint REFUSED while `controllerRestartHold_` is set. Mock server per arm (`/igus_rebel_{left,right}/move_joint|hardware_state|robot_joint_states`). Overlay build_ovl_cri/install_ovl_cri; driver gtests 13/23/17/10/137 pass (re-run by the lead).
- Coordinator (builder; design reviewed by advisor-fable, astra unavailable - usage limit): opt-in `cri_transit_enabled` (default false; launch refuses it in a config), phases mode SERIAL phases only, max 1 CRI run per arm-phase, 20 % (max 25), certificate = chord re-sampled <= 0.01 rad, padding = path_tol 0.005 x rigorous reach 3.601 m (18 mm; stop travel dropped: Move Stop / link loss stay on the chord, e-stop deviation is not CRI-specific), partial chord (ends near the carton stay on the JTC as rest-to-rest sub-chords). doStop cancels MoveJoint FIRST; per-arm controller gate; live robot_joint_states + hardware_state for rest/barriers/rearm; ERROR/latched = FAULT. Overlay build_ovl_crico/install_ovl_crico; 48 new + 151 gtests, 139 bringup + 23 dual-table pytest pass (re-run by the lead). Evidence `scan_results/cri_coordinator_20261024/` (+ stage2/): 2 legs selected per phases plan, 16 mock CRI runs OK, fault cases OK, CRI-off regression SAME.
- Blocked: install waits for the driver follow-up commit on fani8j/cri-move-joint (MOVETOEXECERROR during a requested stop must end FAULT; effective vel/acc caps in HardwareState). Until the cap fields exist, dispatch needs `cri_assume_driver_cap_ok` (default false). Open: 1 of ~20 mock right-arm runs stopped on "controller / hardware state poll is not answering" (suspected CM service latency during switch under load; fails safe). Operator approval for install given 2026-10-24 ("if the mock result is good directly install").
- 2026-10-26 driver follow-up c2e8811 merged (with 3e335da; lead, `git apply` per file, one test-hook hunk of Rebel.hpp by hand; all other hunks verified present by `git apply -R --check`). Only remaining diff vs the branch in Rebel.hpp = main-side `controllerRestartHold_` + ESTOP logging. HardwareState now has `move_joint_max_velocity_percent` / `move_joint_max_acceleration_percent`; the mock server publishes them; `cri_execution.cpp` static_asserts the fields (a rename can't silently disable the cap gate). `cri_assume_driver_cap_ok` stays false and is no longer needed. Overlays rebuilt; gtests driver 152 / supervisor 13 / safety 23 / status 17 / framing 10, coordinator+bringup 209 pass; igus_rebel lint (flake8/pep257 in new mock_move_joint_arm.launch.py, uncrustify of mock server + 2 branch test files) fails, style only, left to keep files equal to the branch.
- 2026-10-26 mock re-check (builder, domain 213, `scan_results/cri_c2e8811_merge_20261026/mock/`): caps 25/40 read from hardware_state; 3 CRI phase-A executions (6 CRI runs) SUCCEEDED/PLAN, strict switch + hold check OK; ~/stop and latch mid-run OK; CRI-off sequence regression OK; no poll stall in 8 executions. Cap 30 % on one server: `~/execute` is NOT refused - the cap gate runs per arm at the serial-phase start and that arm runs its phase on the JTC (the validated CRI-off path); accepted as fail-safe by the lead (a refusal at execute would be a small `handleExecute()` check).
- Install: `scan_results/cri_c2e8811_merge_20261026/install_main.sh` (backup + SHA256SUMS, symlink build of the 6 packages, tests). It REFUSES while any process maps these packages' libs from the main build/install; on 2026-10-26 the real single-arm stack (domain 11, `ros2_control_node` PID 107415 loads `build/igus_rebel/libigus_rebel.so`) was running, so the install was NOT done. Run it after the operator stops that stack; the real driver then has to be relaunched from the new build.

## 2026-10-26 Cut-run recorder (in ~/robot_remote, not in this repo)
- `~/robot_remote` pulled to 78a48dd (upstream real-robot recording); the local uncommitted mocap-real/moveit-real recorder modes were re-applied and the conflicts resolved (backup `~/igus_backups/robot_remote_local_before_pull_20261026/`, stash entry kept). Nothing committed there.
- New `recording/run_record_cut.sh` (read-only LeRobot recorder; `record_dataset.py --source moveit-real --episode-trigger dual-cut|h-cut`): episodes start/stop on `/dual_cut_coordinator/status` + `exec_report` (one episode per ~/execute, A+B; stopped/aborted runs discarded unless `--keep-stopped`) or on `/box_h_cut_executor/t_cut_status` (`--single-arm`). action = JTC `reference`, extra feature `observation.cut_progress`, camera preset `cut` = head + both gripper cams over ROS. Domain 11 by default (`CUT_DOMAIN_ID`).
- Mock e2e (domain 213, `scan_results/cut_recorder_20261026/`): 2 full A+B executions saved (1990/2002 frames at 15 fps = wall time, action vs state 0.0003 rad mean), a ~/stop run discarded. Not verified: real cameras/domain 11, h-cut end-to-end, CRI transits (JTC inactive -> action stale, only counted), a new execution starting while the previous episode is still saving (~1 min) starts late.
- 2026-10-26 (later) user decision: the automatic recorder stores RAW full-res data, the user writes the LeRobot converter. `~/robot_remote/recording/record_raw_cut.py` = one sqlite3 rosbag2 per execution (original serialized messages: cameras as published JPEG, /joint_states, controller_state, coordinator status/reports, FJT status, tf, side_cut_vision; 2 s pre/post roll; stopped/aborted runs kept + marked) in `<cut_videos>/<ts>/{clip.json, snapshot, bag/}` + `index.jsonl` (external drive). Both launches (`dual_cut_system.launch.py` trigger dual-cut, `move_group.launch.py` trigger fjt-status) start it on real hardware: `cut_recorder_backend:=raw` (default) | `video` (old tools/cut_recorder, unchanged; also the automatic fallback), `record_cut_video:=true|false|always`. The main install's launch files are symlinks to src, so this is LIVE at the next real launch (running stacks unaffected). Verified: robot_remote test_record_* 72 OK; overlay build_ovl_rec launch tests 320 OK; mock launch e2e (domain 213, cameras 214) `scan_results/cut_recorder_20261026/launch_e2e/`. Not verified: real cameras/throughput (~15-20 MB/s), side_cut probe writing into the new folder in a live run.

## 2026-10-26 Fast speed profile (opt-in, MOCK ONLY)
- Launch arg `speed_profile:=default|fast` (dual_cut_system / per_arm / both_arms / full_mock launches, move_group.launch.py); `fast` loads `igus_rebel_moveit_config/config/speed_profile_fast.yaml` LAST (overrides cannot undo it): transit 0.40/0.0763 (0.314 rad/s, 0.40 rad/s^2), contact 0.2064/0.0763 (1.5x, 0.40 rad/s^2), coordinator cri_subchord_* 0.314/0.400. `default` = byte-identical effective params. Dwells/gates untouched.
- Mock (domain 213, CRI off, `scan_results/speed_profile_20261026/`): makespan t090346 133.0 -> 82.0 s, t074923 132.9 -> 80.7, t054203 (sequence) 81.3 -> 53.5; t075123 78.6 -> 84.7 (fast loses the sequence plan to a sampling-dependent blade-carton graze that the default also hits at finer sample_dt - existing issue). Plan-only 18/18 both. No gate trips. Tests: 176 pytest (dual_cut_bringup + speed profile) pass in overlay build_ovl_speed.
- REAL use needs the new JTC gains first (jtc-tracking worktree: 0.4 rad/s^2 cap): with today's gains the lag ~0.4 s x 0.314 = 0.126 rad exceeds coordinator tracking_warn_rad 0.1 (stop 0.2), and time-indexed cross-arm checks get that error too. CRI legs (<=25 % = 0.196 rad/s) are slower than fast JTC transits: keep CRI off with fast. Main install lacks speed_profile_fast.yaml until igus_rebel_moveit_config is rebuilt (fast then fails closed; default unaffected).

## 2026-10-31 JTC tracking settings ported to the dual stack (builder + lead; MOCK ONLY, overlay only, main install NOT rebuilt)
- Source of truth: real LEFT-arm validation 2026-10-30/31 (scan_results/jog_speed_check_20261031/, scan_results/jtc_tracking_real_20261026/), reference commit 32481447 (fani8j/jtc-tracking, not merged; only these parts ported).
- Driver: optional hardware param `jog_velocity_scale` (RebelSafety ParseJogVelocityScale, range [0.5, 4.0], invalid -> on_init ERROR; absent -> 2.0 fallback). Passed as 1.65: igus_rebel2.control.xacro / xeg32 wrapper / rebel.launch.py (`jog_velocity_scale`), dual table xacro + dual_table.launch.py + dual_cut_system.launch.py (`left_/right_jog_velocity_scale`, rebel only).
- JTC gains p 3 / i 0 / d 0 / i_clamp 0 / ff 0.6 on all joints: igus_rebel/config/controller.yaml and rebel_dual_table_description/config/controllers.yaml (both arms). Tolerances unchanged. They assume scale 1.65.
- Accel cap 0.4 rad/s^2: executor acceleration_scaling and transit_acceleration_scaling 0.0763 (YAML + C++ defaults), calibration_verification 0.0763, coordinator cri_subchord_acceleration_rad_s2 0.524 -> 0.400 (lead; YAML + C++ defaults; only used with cri_transit_enabled). Velocities unchanged.
- Coordinator review: tracking_warn 0.1 / stop 0.2, skew margin, timeouts, swept volume all derive from planned trajectories or get more conservative; nothing loosened.
- Tests (lead re-ran on install_ovl_jtc, domain 174): igus_rebel safety 26, cri_status 17, cri_framing 10, driver 153/154 (only the known RebelMoveTest.PendingStopSurvivesAReconnectUntilStandstill), coordinator 10 binaries 199 pass, pytest 221 pass. Mock dual executions (builder): makespan +0.6..0.9 % (134.2/134.0/82.1 s vs 133.0/132.9/81.3 s), no gate trips. Evidence: scan_results/jtc_settings_dual_20261031/.
- 2026-10-31 15:02 MAIN INSTALL DONE (operator OK; lead): colcon --symlink-install of igus_rebel_msgs igus_rebel_description igus_rebel rebel_dual_table_description rebel_dual_table_moveit_config igus_rebel_moveit_config dual_cut_msgs dual_cut_coordinator dual_cut_bringup (guard: nothing mapped main build/install). This ALSO installed the CRI MoveJoint merge (3e335da + c2e8811; driver was last built 10-15) and the coordinator CRI wiring (CRI transit still off by default). Backup of the old binaries/installs: /media/farhan/My Passport/igus_nvidia/backups/main_install_before_jtc_port_20261031_1501 (SHA256SUMS). Tests on the main install (domain 175): same counts as the overlay (driver 153/154 known failure only, coordinator 199, pytest 221); installed dual xacro has jog_velocity_scale 1.65 x2, controllers.yaml 12 x p3/ff0.6, executor accel 0.0763. Dual mock stack from the main install (domain 176) came up: both JTCs active, both executors 0.0763/0.0763. Log: scan_results/jtc_settings_dual_20261031/main_install/.
- OPEN: (1) first real dual run needs operator approval; the real launch prints 'jog_velocity_scale 1.6500' WARN lines from both drivers (check they appear). (2) RIGHT arm (192.168.3.12) PASSED the same ladder with profile A 2026-10-31 (0 overshoot, final error 0.000 deg, settle to 0.05 deg <= 0.2 s; scan_results/jtc_tracking_real_20261026/run_real_20261031_right_A/README.md): settings validated on BOTH arms. A real dual run still needs the main install + operator approval. (3) rebel_servo_teleop_keyboard.cpp still uses JOINT_VELOCITY_SCALE 2.0 (teleop thread's file).

## 2026-10-31 dual_teleop: Servo speed relay (builder + lead; MOCK ONLY; installed in main)
- `src/rebel_servo_relay` copied UNCHANGED from the jtc-tracking worktree (identical to cri-move-joint; `diff -r` clean). Other threads porting it to main: it is already here, same content.
- `dual_teleop.launch.py`: `servo_relay` (default true), `relay_max_lead_rad` 0.05 (max 0.15), `relay_lead_time_s` 0.2 (max 1.0), invalid -> refuse to start. True: Servo pass-through smoothing -> /arm_<side>/servo/trajectory_out -> /arm_<side>/servo/trajectory_relay -> the arm JTC. false = old direct wiring.
- Mock (domain 177): realised speed 33 % -> 100 % (joint1 0.2, joint4 0.3 rad/s, Cartesian 0.03 m/s); Servo collision slow-down unchanged (within 1 mm); coast after release 23-37 mrad (relay) vs 10-15 (off, 3x slower). Relay-off finding: the first Servo message after idle can replay a stale state (jump, JTC state tolerance hold) once the arm was moved by something else; the relay is immune. Evidence: scan_results/dual_teleop_relay_20261031/.
- Tests (lead): dual_teleop pytest 86, test_relay_core 39. Main install: `colcon build --symlink-install --packages-select rebel_servo_relay dual_teleop` 15:5x, guard clean.
- OPEN: real-arm behaviour with the relay is unmeasured (the mocap notes saw 57-65 % on the real control law with default lead, measured before the p3/ff0.6 gains); first real dual teleop needs operator approval, one arm at a time, low speed. Don't run dual_teleop and the mocap follower together.
- 2026-10-31 ~18:15 REAL bring-up with the new settings (operator, domain 11, ROS_LOCALHOST_ONLY=1 on stack + cameras): both drivers log `jog_velocity_scale 1.6500`, JTC gains read back p 3.0 / ff 0.6, no foreign participant (0 "sequence size" lines), controllers activated via dual_arm_recover (typed yes), status healthy. Left wrist 305 CV27561000GA NOT on USB (no kernel event in 30 min: cable/hub), others up. Next: one real SEQUENCE-mode dual cut (default profile) when the operator's other process finishes; then check tracking error, makespan vs ~79 s (10-22), and the raw recorder bag. A /tmp/camtest A/B test of another session held the scene camera on domain 187 until 17:54.

## 2026-10-31 VLA dataset inventory / quality / augmentation (report: scan_results/dataset_augment_20261031/SUMMARY.md)
- Per-episode quality score: `tools/mocap_dataset/episode_score.py` (report only; operator decides drops).
- New datasets (external drive, sources untouched, split train/val by SOURCE episode): `mocap_right_31oct_aug_statenoise_v1`
  (115 eps, corrective state noise), `mocap_right_31oct_aug_bgswap_v1` (92 eps, head bg swap + wrist photometric),
  `mocap_right_31oct_retarget_test_v1` (4 eps, test only). SAM3 masks: `mocap_right_31oct_masks_v1`.
- Retargeting stopped at the test (operator); needs an empty-scene head photo + carton height check before any rerun.
- Not done: retraining on the augmented sets (operator: datasets only). New tools are uncommitted.

## 2026-11-06 Left side cut reverse patch (builder + lead; MOCK ONLY, overlay build_patch/install_patch, main install untouched)
- Real-run stats (39 completed / 45 dual executions 10-20 + 10-22): `scan_results/dual_cut_stats_20261106/` (runs.csv: top split, side-cut kept mm vs corner-to-corner edge; left face mean 80 %, back trim 10-117 mm in 36/39; right face 71-72 % = 40 mm corner margin).
- New executor params `left_cut_patch_enabled` (default false = identical), `left_cut_patch_min_gap_m` 0.015, `left_cut_patch_overlap_m` 0.015: after the forward left stroke, a reverse stroke covers the back gap (same task; falls back to forward-only on any patch failure; coordinator re-plans with the patch off if a refusal names a patch stage). Tests: test_left_cut_patch 10 + existing pass (re-run by lead); regression patch-off identical 39/39.
- Result `scan_results/left_side_patch_20261106/`: with production caps (25/25, tilt 25) the patch plans on 0 of 29 runs that have a gap (reverse frame: handpiece/holder/link6 hit the carton or no IK). Relaxed caps (band 35 / end 40 / tilt 45, about_edge = blade upside down): 5/24 patched, coverage 79-89 % -> 83.5-96.4 %, +16-26 s makespan; 3 mock executions OK. Not installed, not reviewed by advisors, not committed. Open: operator decision.

## 2026-11-06 Fixture shift study (MOCK, hucenrotia-ai docker)
Moving the carton fixture towards the arm bases: 40 mm -> left face mean 86.8 % (18/39 full edge, 0 refused), 60 mm -> 89.8 % (24/39 full), makespan unchanged ~78 s,
left arm then cuts the whole top seam alone. Details scan_results/fixture_shift_20261106/README.md. Operator decision pending (no physical change made).

## 2026-11-06 Right face corner margin sweep (MOCK, domain 229, main install)
right `left_cut_corner_margin_m` 0.040 -> 0.010: right face 71.0 -> 92.7 % (35/35 plan, right search ~3 s, makespan 78.0 -> 81.0 s, 3/3 mock exec ok,
min joint margin unchanged 0.089 rad); 0.005 = 96.4 % but joint margin 0.037 rad + 69 s outlier. Not applied (config change in arm_right_executor_overrides.yaml
needs operator OK). Not yet tested combined with the fixture shift. Evidence scan_results/right_corner_margin_20261106/.
Combined (mock): fixture -60 mm + right margin 0.010 -> left 89.8 % / right 92.8 %, 39/39 plan, makespan 80.3 s, 3/3 mock exec. Recommended; pending operator (physical fixture move + override YAML change).

## 2026-11-07 Placement workspace map (MOCK, hucenrotia-ai ~/placement_workspace_20261107/, 549 cells, Sonnet agent there)
Current box (top ~275x392), right margin 0.010. Full H (both faces >= 90 %) at rotation 0: box centre x ~270-410 mm, y ~-275..-115 mm (left base frame);
best rotation +10..+20 deg (CCW from above): centre x ~290-450. Negative rotation poor (<= -20 nothing >= 90 %). Far limit = right/left top seam reach;
near limit = knife holder/table collision. Local copy: scan_results/placement_workspace_20261107/ (remote_report.md, cells.json, workspace_map.html).
Pending: other box sizes from the operator.

## Negative-yaw failure investigation (2026-11-07, mock on hucenrotia-ai)
See scan_results/negative_yaw_20261107/README.md. Cause: start corners plus the tool's wrist offset. The left far corner is beyond the wrist's reach, and the right near corner leaves the wrist too close to the base. Clockwise yaw worsens both at once. Reverse travel is infeasible with tool C (this overturns the 10-15 result). Fix = placement: yaw >= -5 preferred, -10 OK at dx -120..-80/dy -40..-80, -20 only with a 12-24 cm shift toward the right arm. Proposed (not built): placement advisor at detection time.

## Placement advisor (2026-11-07, not installed, not committed)
`src/dual_cut_ui/dual_cut_ui/placement_advisor.py` adds the `two_arm_reach` warning to assess_placement, shown first. It is info-only and never blocks.
- It applies the wrist model at both arms' start corners and suggests a move or turn.
- `left_reach_enabled` now defaults to off, superseded.
- The lead added a validity window: the model is trusted only from -30 to +10 deg around the template turn.
- Validation: scan_results/placement_advisor_20261107/README.md. Tests: 493 pass.
- At today's template pose it says FAIL (left wrist -40 mm), with the hint to move 60 mm toward the arms. That agrees with the fixture-shift study.

Open: the old `region` and `right_table` notes contradict the study, and the region hint points the wrong way. Retiring them needs the operator's OK. The UI is built only into build_ui/install_ui, which has not been rebuilt.

The box-size sweeps are running on hucenrotia-ai in ~/box_sizes_20261107. The boxes are 330x570, 300x450 and 280x400 mm; their heights are unknown, so the template height is used. They will also be used to check the advisor on other sizes.

## Box sizes + advisor recalibration (2026-11-08, mock)
- Sweep for 3 boxes (330x570, 300x450, 280x400; height assumed 0.20 m = executor `carton_height_m`, NOT from detection): scan_results/box_sizes_20261107/remote/box_sizes_20261107/README.md. Every box has an F region near the arms; box 1 is narrow (centre x 270-390). Mock exec OK for all three.
- Advisor now has 7 constraints (added right far end 0.515, right start <= 0.35, carton min x 0.115), the validity window is widened to +30 deg, and it was fitted on 1804 cells. False OKs drop from 35 % to 6 % of passes; leave-one-box-out gives 0-8 % per new box. Tests 516 pass. Details: scan_results/placement_advisor_20261107/README.md "Box sizes".
- Thresholds hold only for right `left_cut_corner_margin_m` 0.01 (the real stack still has 0.04) and a 0.20 m height.

## Two-direction knife / mirrored left holder (2026-11-08)
- A double-edged blade (keep_orientation reverse) gives NO gain. In mock, 171/175 face plans were the same as forward, because the wrist sits at the same place relative to the tip. The experiment patch is in scratch only and was never applied to src.
- The reach model, with each arm's tool set separately, says: if the LEFT holder is mirrored (wrist ahead of the tip, toward the base) and the right holder stays as today, the current box covers about -40..+40 deg (box 1 about -20..+20). Mirroring the RIGHT holder is bad, which is consistent with the old v4-R result. This is model only: the left near-end/base collisions of a mirrored holder are not modelled. Next step: mirrored-left URDF + mock sweep.
- Animation of the cut modes: scratchpad knife_modes.html (not in the repo).
- 2026-11-08 MOCK RESULT (scan_results/mirror_left_20261108/remote/README.md): a mirrored left holder SHIFTS the window rather than widening it. Template -30..+20 vs today -20..+30, box 1 -30..+20 vs -10..+30; -30 is marginal (2 fallback cells). New limits: left holder/link6 vs table/base on the near side, the right arm's near end, and the left top seam (80-85 % median). Today's real window is -20..+30 once the box is shifted 12-24 cm toward the right arm (earlier baselines never sampled that). Not recommended as a hardware change.
