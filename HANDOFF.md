# Handoff — igus ReBeL H-cut workspace

> **Never dispatch motion on the physical robot without explicit operator approval in the
> current conversation.** Everything below marked "verified" was verified in isolated mock
> (`ROS_LOCALHOST_ONLY=1`, unused `ROS_DOMAIN_ID`) or against a fake CRI server, not on the
> real arm.

## 0. Medical carton shelf-pick tooling (2026-10-05, vacuum 2026-10-13, final suction tool v5 2026-10-14, Isaac Sim study + report 2026-10-15, branch `mackay`)

- `cad/medical_box_tooling_v1/`: shared inputs `common/params.py` and handoff poses `common/handoff.py`.
  - `suction_tool_v5/` (2026-10-14): the FINAL left tool; integration and the USD now use it.
    - One printed PA12-CF body and one camera (Gemini 305). The SE4710 and PL3307 were dropped at the operator's request; barcodes are decoded in software from the colour stream.
    - Same pads (ZP2-T4030WNK10-B5 + KQ2L04-M5 elbows), positions and lip z 159.
    - 0.204 kg; 0 % of the camera FOV blocked; aim depth 91 mm.
    - Handoff clearances are identical to v3. The URDF adds `pad_{1,2,3}_suction_frame` and `wrist_camera_color_optical_frame`.
    - Pad nut torque on the printed bar is limited to about 0.3 N·m (README).
    - `--tool suction_tool` / `SUCTION_TOOL=suction_tool` reruns v3. `suction_tool/` (v3) and `suction_tool_v4_concept/` are kept for history.
  - Vacuum source (2026-10-14): the AMR has no room (robot stand), and no pump or pads are on hand yet.
    - Recommended: one Schmalz ECBPi 12 24V-DC M12-8 (10.03.01.00314) on the stand's lift-carriage bracket pocket (P1, `robot_stand/`). It has a built-in sensor, part-present output and vent valve.
    - `vacuum_sim/pump_sim.py`: in a tight shelf the ECBPi tolerates about 2.4 L/min leakage per pad (SLZ-100 0.25, 2 x Thomas 5.8).
    - The 2-pump stand box design is paused; `vacuum_module/` (KNF on the AMR) is superseded.
  - `robot_stand/`: analysis of the customer stand STEP (lift ZLW-20120 1250 mm, both arms on a 660 mm X rail on the lift carriage). Up axis and left/right are inferred, not confirmed.
  - `suction_tool/`: left arm, v3. 3 x SMC ZP2-T4030WNK10-04 into an on-tool 3-to-1 collector, one ⌀6 trunk; vacuum from the electric pump module on the AMR, about 2 m of trunk (gate −40 kPa at the switch / 150 ms; model in `verify.py` 5c); Zebra SE4710 + PL3307, Gemini 305; 3 printed parts, 0.278 kg. Lip at link6 z = 159 mm.
  - `gripper_tool/`: right arm, v3. SMC LEHF20K2-48 in-line, fingers 8/56 mm with 8 mm chamfers, fixed 3 mm centre top-stop. Registration is position-based (ReBeL has no force sensing): pre-close to t + 4 mm, descend to a computed pose with the stop 4 mm above the top edge, clamp at 100 %. README has the numbered procedure.
  - `vacuum_module/` (2026-10-13): the robot stands on an AMR in the hallway to the operating room, so there is no compressed air.
    - KNF N 838 KN.29DC-B 24 V brushless pump, dust filter, SMC ZSE20B switch, N.C. vent valve, HEPA exhaust filter, in a lined printed box (4 PETG parts).
    - 240 × 186 × 220 mm, 4.5 kg; needs a 40 mm fan.
    - Part of the pump drawing data and some small-part envelopes are ASSUMED (README).
    - Alternatives if bench test 0 shows low leakage: Schmalz ECBPi; 2 × Thomas 1520VDP. Datasheets are in `shelf_pick_research/vacuum/` on the external drive.
  - `integration/`: handoff clearance check, `handoff_check.py`, ~40 s. It needs python-fcl, which is in the external drive's `scratch_integration/pylib`.
- Each tool folder has:
  - parametric cadquery code, rebuilt with the external-drive CAD venv;
  - STEP/STL files;
  - a URDF xacro;
  - collision meshes;
  - `data/verify.json` and a README with the parts list and assembly order.
- `sim_assets/medical_box_tools/`: USD for both tools and `shelf_handoff_scene.usda`, which references the box-3d session's carton USDs. `tools/verify_tool_usd.py` passes 82/82 (v5 left tool).
- Isaac Sim 5.1 study (2026-10-15), simulation only:
  - `tests/test_cabinet_grasp_robot.py`: full dual-arm + lift replay in `usd/cabinet_robot_test_scene.usda`; plans from `tools/export_plan.py` (`data/plans/`), kinematics in `tools/robot_kin.py`, reach/parking in `tools/parking.py` / `parking_study.py`. Results in `tests/results/` (`cabinet_grasp_robot_summary.md`).
  - Front grasp (suction pulls 100 mm, gripper grasps from the front) passes for the 7 tested cartons on shelves 3-6, one run each. Top-down grasp fails (arm hits stock above / top panel). Shelf 1 unreachable (116-195 mm short); shelf 2 near miss (24 mm short, pad touches the neighbour) - see `tools/closest_attempt.py`, `tests/results/unreachable_closest.json`.
  - E2 (first carton at the side wall): a longer/asymmetric suction tool does not help (`tools/e2_study/`, `tests/results/e2_longer_tool_study.md`); options are 25-33 mm wall gap, hand pick, or an angled tool.
  - Suction holding force was never challenged (most loaded pad <= 0.74 N vs ~5.8 N software release limit). Cycle 45-47 s sequential, 38-41 s overlapped.
  - Videos (3 views + tiled, closest attempts, highlight reel): external drive `scratch_usd/videos/robot_20261015/`; old carrier runs in `carrier_20261015/`. `--header` restores the top info bar.
- Report: `cad/medical_box_tooling_v1/report/medicine_shelf_pick_report.html` (local, light mode, three.js viewers; fact-checked + reviewed by both advisors). The MP4s in `report/media/` are git-ignored; copy them per `report/media/VIDEOS.md`. AMR content was removed at the operator's request (the AMR is another team's). The old hosted page https://claude.ai/artifact/FUocA26sA5XpZEFZQAhZXa is outdated; the operator will delete it.
- ECBPi part number corrected to 10.03.01.00314 (00364 is the TB-8 variant).
- Open items:
  - Right-arm camera not decided.
  - Box masses and pad friction are not measured (0.5 is assumed).
  - The pad fitting height (14.7 mm) is read from a catalog sketch; check it on the first parts.
  - Bench tests before printing: `cad/medical_box_tooling_v1/BENCH_TEST_PLAN.md`. Test 0 (pad leakage on carton) decides the pump; test 11 covers noise and heat of the module.
  - Pump noise (no dB(A) published), AMR mounting pattern and 24 V aux budget (2.3 A peak), and the IO for the switch, vent valve and pump speed are still to assign.
  - The handoff check has not been re-run against the medicine-cabinet USD (shelf clear heights 230-310 mm).
  - Spine tilt and shelf lean are not modelled.
  - The ECBPi fit at P1 needs the Schmalz CAD; the report's 3D pump viewer has not been checked in a real browser.
  - No real holding-force, leakage or combined-error test yet (bench tests).
  - `gripper_tool/meshes/collision/finger_*_ledge.stl` are stale and unused.

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
- **Cameras (`rebel_cameras`, 2026-09-27)**: `ros2 launch rebel_cameras cameras.launch.py`
  starts the Gemini 336 (`/camera`, serial CP96563000LY), Gemini 305 (`/wrist_camera`,
  CV27561000NP) and the knife webcam. Measured together on an isolated domain: all at
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
