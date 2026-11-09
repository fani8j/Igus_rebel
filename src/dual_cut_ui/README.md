# dual_cut_ui

Operator web page for the dual-arm carton cutting cell: look at the cameras, detect the carton, plan the two-arm cut,
check the result, arm the cell and cut. One page, big fonts, same look as the calibration UI. It only talks to
`/dual_cut_coordinator` (services, parameters, topics) and to the two trajectory controllers' cancel service. It never
publishes a motion command itself.

## Launch on the real cell

```bash
source /opt/ros/humble/setup.bash
source ~/igus_rebel_ros2/install/setup.bash
source ~/igus_rebel_ros2/install_ui/setup.bash        # this package, built separately (see Build)
# Detect uses the DetectBox action; that interface only exists in the box-seam-detection install, so source it last:
source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash
ros2 launch dual_cut_ui dual_cut_ui.launch.py camera_domain_id:=11     # real cell: run in the dual stack's environment (domain 12)
# the log prints   dual-cut UI: http://127.0.0.1:8096/?token=...   (a new token at every start)
```

Open exactly that URL (the token is moved into the browser's sessionStorage and removed from the address bar). The server
listens on 127.0.0.1 only and has no bind-address option. Add `&view=present` for the read-only projector view (advanced
controls hidden, cameras bigger, Software stop still there).

Launch arguments: `port` (8096), `snapshot_file` (default `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json`),
`detect_mode` (`action`, the default, or `command`), `detect_action` (`/box_seam_detector/detect_box`), `detect_timeout_s` (300),
`schedule_mode` (`sequence`; empty = leave the coordinator's value alone), `detect_cwd` (command mode; default
`/home/farhan/Downloads/seam_handoff_20260908/scripts`), `detect_output_dir` (default
`/home/farhan/Igus_rebel/outputs/rgbd_captures`), `detect_command` (command mode: ONE string, split like a shell would, never run through a
shell; empty = `python3 capture_and_query_depth.py --autolabel-v2 --allow-needs-review --allow-cut-blocked --output-dir <detect_output_dir>
--planner-output-json <snapshot_file> --target-frame base_link`), `cameras` (`name=/topic,...`, empty = the six default
CompressedImage topics), `plan_timeout_s` (300). If you pass your own `detect_command`, keep its `--planner-output-json` equal to
`snapshot_file`. In action mode `detect_command`, `detect_cwd`, `detect_domain_id` and `camera_localhost_only` do not apply to Detect
(the goal goes out in the UI's own ROS domain; `detect_output_dir` is still used to find the overlay). Other node parameters: `control_grace_s` (10), `camera_small_width` (320), `camera_large_width` (960),
`camera_jpeg_quality` (70), `camera_domain_id` / `detect_domain_id` (-1), `camera_idle_s` (10), `coordinator` (`/dual_cut_coordinator`), `log_file` (`~/.ros/dual_cut_ui/dual_cut_ui.log`, rotating; empty = stderr only),
`placement_limits` (JSON object overriding the thresholds of the information-only placement pre-check, see below; empty = defaults; a bad value stops the UI at start).

## What the buttons run (the operator's hand commands)

* **Detect** (`detect_mode:=action`, default) sends the goal of
  `ros2 action send_goal --feedback /box_seam_detector/detect_box compal_box_msgs/action/DetectBox "{dry_run: false}"`
  (`dry_run false`, `max_attempts 0`, both `allow_*` flags false). The job log shows `send_goal ... DetectBox {dry_run: false}`, every
  feedback as `[stage] attempt N: message`, and the result (status, reason, reasons, cut_permission, snapshot_path, attempts, elapsed).
  The step succeeds only if the result says `success` AND the snapshot file really was written during the job and is usable (the same
  check as command mode). A goal the detector rejects (busy) fails with `detector busy / goal rejected`; no action server within 5 s
  fails with `detector action server not available (is box_seam_detector running?)`; no result within `detect_timeout_s` cancels the goal.
  The interface is imported lazily: if `compal_box_msgs.action.DetectBox` cannot be imported (box-seam-detection install not sourced) the UI
  logs an error at start and the Detect step fails telling the operator to
  `source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash`. `detect_mode:=command` runs the old subprocess
  (`detect_command`) exactly as before. **Cancel detect** and **Software stop** cancel the goal.
* **Plan** first sets `ros2 param set /dual_cut_coordinator schedule_mode sequence` (`schedule_mode`; the job log shows
  `param set /dual_cut_coordinator schedule_mode sequence`), then calls the coordinator's `plan`. If the set fails, Plan fails with the reason
  and `plan` is not called.
* **Enable execution** sets `execution_enabled` and `execute_phase_a_enabled` true and nothing else (phase B is not used by a sequence plan).
  Every way back to off (switch off, Software stop, run end, UI start / stop, a raced or unconfirmed enable) writes ALL THREE gates false.
  The page's Cut check reads back the two enable parameters, and it blocks Cut when the retained plan is a `phases` plan while
  `execute_phase_b_enabled` is false (only phase A would run: plan again).

## Cameras and detector in another ROS domain

On the real cell the cameras (`rebel_cameras`) run in ROS domain 11 and the dual stack with the coordinator in domain 12. Start the UI
in the dual stack's environment (domain 12) and give it `camera_domain_id:=11`:

* The coordinator, controller manager, parameters and Software stop always use the UI node's own domain.
* With `camera_domain_id >= 0` and different from that domain, `main()` creates a second rclpy context in that domain with the node
  `dual_cut_ui_cameras` (own executor thread). It ONLY holds the lazy CompressedImage subscriptions (no services, clients, parameters
  or timers; no parameter services and no rosout publisher; rclpy itself still creates its idle `/parameter_events` publisher). It is
  shut down last, after the main node's work (gates closed first). The log says `cameras read from ROS domain N (read-only)`.
* `detect_domain_id` (default -1 = `camera_domain_id` if set, else inherit) sets `ROS_DOMAIN_ID` in the detector subprocess (a copy of
  the environment; `ROS_LOCALHOST_ONLY` stays as inherited). The job log shows `(cwd X, ROS_DOMAIN_ID N)`.
* **ROS_LOCALHOST_ONLY (to verify on the real machine):** the cameras run with `ROS_LOCALHOST_ONLY=1`, the dual stack with 0. The UI
  process has ONE `ROS_LOCALHOST_ONLY` for both contexts; it must be the dual stack's value (0). A process with 0 is expected to still
  see the localhost-only (1) participants on the same host, so the camera context should reach domain 11. Not yet verified.
* Values must be -1 or 0..232 (Fast DDS limit); anything else stops the UI at start.

## Build and test (do not build into the main build/ install/ while the robot runs)

```bash
cd ~/igus_rebel_ros2 && source /opt/ros/humble/setup.bash && source install/setup.bash
nice -n 19 colcon build --packages-select dual_cut_ui --build-base build_ui --install-base install_ui --parallel-workers 4 --symlink-install
nice -n 19 colcon test --packages-select dual_cut_ui --build-base build_ui --install-base install_ui && colcon test-result --test-result-base build_ui/dual_cut_ui --verbose
# or directly (no ROS runtime needed except for two import tests that skip without rclpy):
cd src/dual_cut_ui && python3 -m pytest -p no:anyio -p no:cacheprovider test/
```

## What the page does

The page is a fixed viewport grid (it never scrolls; designed for 1920x1080 and 1366x768): top bar, cameras on the left, on the
right a compact control card (Detect & plan, step rows, Enable execution, Cut, the last action's reply) above a result card whose
body scrolls inside itself.

* **Top bar (always visible):** the red **Software stop**, the coordinator state pill (IDLE, PLANNING, PLANNED, WAITING (barrier),
  EXECUTING, STOPPED with the stop reason, COORDINATOR OFFLINE when no status arrived for 2 s), controllers L/R and hardware L/R,
  execution on/off, joint-state age, minimum cross-arm distance, snapshot age with the time left, and who is the operator. A value
  that is stale (coordinator offline, read-back older than 3 s) is shown grey with "?", never as good news.
* **Cameras (left):** one large tile and a strip of small tiles; click a small tile to swap it into the large one. For the scene
  camera "Show detection" swaps in the detection overlay. A tile says "no signal" when no frame arrived for 3 s. A camera topic is
  subscribed only while a page asks for it (idle for 10 s: the subscription is destroyed). Frames are never decoded in the ROS
  callback. Small tiles poll at 2 Hz and are decoded at 1/4 or 1/8 size; the large tile polls at 8 Hz and forwards the camera's own
  JPEG untouched when it is at most 1.35x the target width (no decode/encode at all). Nothing is polled while the tab is hidden.
* **Control card:** *Detect & plan* runs the detector (about 21 s), reads the snapshot, sets the coordinator's `schedule_mode` and calls its
  `plan` (waits up to `plan_timeout_s`); see "What the buttons run". Each step row has a short result ("Plan OK - makespan 91 s est."); the full coordinator message is in the
  Plan tab, once. *More* shows *Detect only*, *Plan only*, *Cancel detect* and *Rearm* (asks first; clears the stop latch and the
  retained plan, needs both arms at rest). **Enable execution** (confirm dialog) sets `execution_enabled` and `execute_phase_a_enabled`
  true (phase B is left alone); the page shows the values READ BACK from the coordinator (polled every second), and when that
  read-back is unknown the switch becomes "Close execution gates". The server sets all three gates (including `execute_phase_b_enabled`) false again on Software stop, when
  the UI starts, when it stops, and every time a run ends (the coordinator leaves WAITING_BARRIER / EXECUTING). **Cut** is enabled
  only when you are the operator, the coordinator is PLANNED with a retained and validated plan, the snapshot is fresh,
  controllers and hardware of both arms are active, `execution_enabled` and `execute_phase_a_enabled` read back true (and, unless the retained plan is a `sequence` plan, `execute_phase_b_enabled` too), nothing is latched and no job runs; a one-line
  reason says why it is blocked (all reasons: Plan tab). A confirm dialog repeats the plan summary; hold the red button for 1.5 s
  (letting go early cancels). The request carries the plan number shown in the dialog, and the server refuses it if the coordinator
  now holds a different plan. The server repeats all checks; the coordinator's own gates still apply on top. "Cut" means "dispatch
  accepted". While a run is active the card shows the phase and elapsed vs planned seconds. The line under the buttons always shows
  the reply of the latest action (Rearm, Enable, Cut, Take control, ...) until the next action.
* **Result card tabs:** *Summary* (KPI tiles: makespan, top cut L/R in %, min cross, snapshot age countdown, cut permission,
  placement pre-check count (information); plus the carton drawn from above, +x up and +y left, left arm on the left, with the
  seam interval each arm cuts), *Detection* (overlay image, cut permission with reason codes, placement pre-check numbers and hints;
  informational only, nothing is blocked), *Phases* (per-phase durations; a parallel phase takes the longer arm, a serial phase
  the sum; unknown mode = sum), *Plan* (the full message, schedule, last execution, all Cut blockers), *Log* (job log). If the
  coordinator no longer retains the plan (after Rearm), the result says so and the plan numbers are dimmed.
* **Dialogs never lock the page:** confirm, hold-to-cut and image-zoom dialogs are non-modal, with our own backdrop that sits
  below the top bar, so Software stop stays clickable while any dialog is open. Esc or a click on the backdrop cancels the dialog.
* **Presenter view (`&view=present`):** no buttons at all except Software stop; large state pill, larger cameras, KPI tiles, carton
  drawing and detection overlay (each fills half of the column height), and the current phase with elapsed / planned time while executing.

## Placement pre-check (information only) and the blocked-detection switch

**Placement** (`placement.py`, shared with `tools/dual_run/check_placement.py`) never blocks detection, planning or Cut (operator decision). New 2026-11-07,
listed first: the **two-arm reach** check (`placement_advisor.py`): a wrist model of both arms' START corners (left wrist <= 0.52 m from its base at back_left, right
wrist >= 0.19 m from its base at front_right, tip guards; works for any box size from the snapshot's own corners). It gives a **warning** when the model fails
and a **note** when it passes with < 15 mm to spare, with the cheapest move / turn / both ("turn it 11 deg anticlockwise ..." / "move it 60 mm toward the arms
(-x) and 10 mm toward the right arm (-y)") or "No placement within +-250 mm / +-30 deg found". It is an estimate (92 % agreement with the planner at yaw -30..+10 of
the template, unreliable beyond +10 deg anticlockwise; `scan_results/placement_advisor_20261107/README.md`). The single-arm left-face estimate below is now
OFF by default (`left_reach_enabled`), superseded by it. From the detected
snapshot, before planning, the control card lists: a **warning** when the left arm probably cannot reach the tape band on the left face (estimate: wrist
centre of the nominal knife pose more than 552 mm from the left shoulder at the back end of the band; flags exactly t090925 and t093229 of today's 42
detections) with a hint like "move the carton about 20 mm toward -x, or turn it anticlockwise about 6 deg"; a soft **note** that the right holder may
be refused (centre x <= 0.4217 m and seam yaw <= -89.5 deg: it touched the table edge today; may be fixed once the table is in the planning scene); and a
**note** when the seam centre / yaw is outside the box in which today's detections planned (x 0.380..0.435, y -0.245..-0.171 m, yaw -92.5..-77.7 deg;
replaces the 2026-10-18 window, kept as `LEGACY_WINDOW_20261018`). All thresholds are in `PlacementLimits`; override them with the launch argument /
node parameter `placement_limits` (JSON, e.g. `{"right_table_x_max_m": 0.41, "left_reach_enabled": false}`). Details and the 42-row prediction table:
`scan_results/placement_precheck_20261022/README.md`.

**Blocked detection.** The control card shows the detection's `cut_permission` (red and large when `blocked`, with the reason codes). While the detection (or
the retained plan, `detection_blocked` in the plan report) is blocked, Cut is refused by the page and by the server until the switch **Allow blocked
detection** is on; the switch writes the coordinator parameter `allow_blocked_detection` (own parameter request, never part of the execution gates, the page
shows the READ-BACK) after a confirmation, only while the detection is blocked, the coordinator is live and no stop is latched. It goes off again by itself on
Software stop, at the end of a run, when execution is switched off, at the next Detect, at UI start and at UI stop; a racing Software stop / takeover / shutdown
always ends with it off. The coordinator's own refusal is a separate check on top (not part of this package).

## One operator at a time

The first page that polls while nobody holds control becomes the operator (a `view=present` page never does). Other pages are
read-only: every action except **Software stop** and **Take control** answers 403. A holder that stopped polling for more than 10 s
can be replaced by the next page that asks; taking over from an active holder needs a confirm (`force`).

## Software stop is NOT an e-stop

It is a software request made of independent steps that run at the same time and never wait for each other or for a running plan:
(a) the coordinator's `stop` service (holds, cancels goals on both controllers, latches STOPPED), (b) a direct cancel-all on both
FollowJointTrajectory actions, repeated at +0.5 s and +1.5 s, (c) the three execution gates set false, (d) a running detect job is
stopped (action mode: the DetectBox goal is cancelled; command mode: SIGTERM to its process group, SIGKILL after 3 s). The reply lists each step; a step that failed (coordinator absent,
no answer in 1 s) is reported in red and does not stop the others. It works from any page, including read-only ones, without a
confirm. If a step failed or the arms keep moving, use the hardware e-stop.

## Not in this UI

Hardware recovery (driver faults, controller restarts, re-activating hardware) is deliberately not here:
`ros2 run dual_cut_bringup dual_arm_recover.py recover ...`. There is also no generic service, topic or parameter access: the page can only
send the eleven whitelisted actions `take_control`, `release_control`, `software_stop`, `detect`, `plan`, `detect_and_plan`,
`set_execution_enabled`, `set_allow_blocked_detection`, `execute`, `rearm`, `cancel_job`.

## Mock helpers (MOCK ONLY, installed to lib/dual_cut_ui)

* `mock_detect.py` replaces the detector: copies a recorded snapshot to `--planner-output-json` with `acquisition_stamp_ns` = now,
  creates `<output-dir>/pipeline/<capture_id>/<run>/` with a byte-identical `planner_snapshot.json` and an overlay PNG. Refuses to
  write the real snapshot file or anything below `/home/farhan/Igus_rebel/outputs`. Use it as `detect_command` in a mock run.
* `mock_detect_action_server.py` is a DetectBox action server for action-mode Detect (`--source` recorded snapshot, `--snapshot` the UI's
  `snapshot_file`; feedback, then the snapshot copied with `acquisition_stamp_ns` = now; `--reject` answers `rejected` without writing;
  rejects a goal while busy, honours cancel). Needs the box-seam-detection install sourced (for `DetectBox`), `ROS_LOCALHOST_ONLY=1` and an
  explicit domain that is not 11, 12 or 219..232; refuses the real snapshot path.
* `test/stub_backend.py` (MOCK ONLY, no ROS context at all) serves the real page and server on a stubbed coordinator (status, plan, execute, gates, `allow_blocked_detection`)
  and writes the next snapshot of a list on every Detect, for looking at the page in a browser (`--snapshots t090925,t061508,...` from `test/data/placement_20261022.json`).
  Needs `ROS_DOMAIN_ID` set explicitly (not 11) and `ROS_LOCALHOST_ONLY=1`; listens on 127.0.0.1 only.
* `mock_camera_publisher.py` publishes the six default camera topics (only while someone subscribes). It refuses to run unless
  `ROS_LOCALHOST_ONLY=1` and `ROS_DOMAIN_ID` is set explicitly and is not 11, 12 or 219..232.

## Layout

`core.py` (operator lock, whitelist, job state machine, execute checks, software stop, gate closing; no ROS), `bridge.py` (the
interface), `detect_action.py` (the action-mode runner with the same terminate surface as `jobs.ProcessRunner`), `ros_bridge.py` (rclpy implementation), `server.py` (Tornado, auth, endpoints), `snapshot.py`, `placement.py`,
`report.py`, `topdown.py`, `camera.py`, `jobs.py`, `auth.py`, `static/` (the page). Tests use a fake bridge; no ROS graph is needed.

## Shutdown and logging

SIGINT, SIGTERM and SIGHUP (also when they arrive during start-up) end the server loop and run one bounded sequence (at most
4 s in total, below the launcher's SIGTERM-to-SIGKILL escalation): close the three gates (first; in parallel a running detector is
sent SIGTERM and SIGKILL after 1 s, or the DetectBox goal cancelled), unsubscribe the cameras, stop the pools, then shut ROS down. Every step is logged and
flushed to stderr, to the rotating log file and to the ROS log. Look for `execution gates -> false (UI shutdown)` and
`UI shutdown finished in ...` in `~/.ros/dual_cut_ui/dual_cut_ui.log`. rclpy's own signal handler is switched off so the gates call
still has a live context.

## CPU

Per-request costs measured on this PC (one thread); the per-page totals are calculated from them, not profiled in the running UI (1280x720 q80 source): small tile 1.1 ms (decode at 1/4), large tile 0 ms
(pass-through; 6.9 ms if re-encoded). A 1920x1080 source costs 2.7 ms (small) and 3.7 ms (large, 1/2 size decode). With one page
open: 5 small tiles x 2 Hz x 1.1 ms + large 8 Hz x 0 = about 1 % of one core (was about 9 % with 3 Hz / 480 px / re-encoded large).
The only per-message work in our callback is one copy of the compressed frame (measured 2-5 microseconds); rclpy's own deserialisation of a message was not measured.

## Concurrency rules (round 4)

* **Execute and plan exclude each other inside the UI:** one lock covers Cut's checks plus the `execute` call and the start of any
  job with a Plan step. After the coordinator accepts `execute`, the status stream is watched for 3 s (or until WAITING_BARRIER /
  EXECUTING): if the coordinator's `plan_seq` is not the confirmed one, the UI issues a Software stop by itself and shows an alert.
* **Disable dominates:** an enable that raced a Software stop, a shutdown or a takeover writes the gates false again and is refused.
  Stop and shutdown never wait for an enable. After a run ends the UI keeps retrying the false write (every 1 s) until it succeeds,
  the UI closes, or the operator enabled execution again.
* **Stop dominates a rearm:** a Software stop during a rearm call makes the UI send `stop` again right after the rearm returns.
* **Stop latency:** `/api/state` is built on its own thread pool (never on the IOLoop); each cancel repeat has its own thread.
  A cancel answered REJECTED with no goal counts as a failed stop step.
* **Detector (command mode):** the whole process group gets SIGTERM, then SIGKILL after the grace, even if the direct child exited; the stdout
  reader cannot hang the job (bounded join, then the leftover group is killed).
* Actions already admitted for a page that then loses control are allowed to finish (except an enable, see above).

## Open

* The coordinator's `~/execute` takes no `plan_seq`. The UI check (compare, dispatch, watch the status stream, stop on a mismatch)
  narrows the window but cannot close it; a full fix needs a coordinator change (`execute` carrying the confirmed `plan_seq`, refused
  if it differs).
* If a controller answers a cancel-all with no goal active using ERROR_REJECTED, a Software stop shows that step as failed even though
  nothing was moving. Check this once on the mock (idle controllers) and relax `bridge.interpret_cancel` if needed.
