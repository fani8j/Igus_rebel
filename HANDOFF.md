# Handoff — igus ReBeL H-cut workspace

> **Never dispatch motion on the physical robot without explicit operator approval in the
> current conversation.** Everything below marked "verified" was verified in isolated mock
> (`ROS_LOCALHOST_ONLY=1`, unused `ROS_DOMAIN_ID`) or against a fake CRI server, not on the
> real arm.
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
- 2026-10-15 **Left arm defaults + return home (lead)**: operator's live values made YAML defaults: `left_cut_inward_m` 0.007, `left_parallel_max_tilt_deg` 10 (flatter blade, steep back-corner start trimmed; on the 21:52 placement the left search then finds no interval covering the tape - later placements cut fine on the real arm). New planning param `return_home_after_cut` (YAML true): plan_h_cut ends with a 'transit to supplied home pose after the cut' to the SRDF home (wrist- and collision-gated); mock t_cut 202224 + 1720 executed, final joints 0.0. Main install rebuilt; install_right NOT rebuilt (right arm unchanged). `middle_cut_clearance_m` still -0.005 (operator thinks too deep; value pending).
- 2026-10-15 **Cut video recorder (builder + lead)**: `tools/cut_recorder/cut_recorder.py`, passive (subscriptions only, tested). One clip folder per trajectory execution (opens on the JTC action status, closes 3 s after the last goal): knife/scene(0.5)/wrist/right_wrist mp4 (compressed topics) + clip.json + snapshot copy, default `/media/farhan/My Passport/igus_nvidia/cut_videos/`. `move_group.launch.py` starts it with the executor on `hardware_protocol:=rebel` (`record_cut_video` default true, `cut_video_cameras`, `cut_video_dir`). Mock-verified only (domain 194, fake cameras); real cameras/wrist frames untested; ~1 core + ~50 MB/min recording all three. install_right not rebuilt. `scan_results/cut_recorder_20261015/`.
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
