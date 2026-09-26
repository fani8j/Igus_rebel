# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A colcon workspace driving an igus ReBeL 6-DoF arm with an XEG-32 gripper and knife TCP over ROS 2 (Humble on Ubuntu 22.04 is the primary target; Jazzy also supported). The main application is **snapshot-relative carton cutting**: a perception snapshot of a taped carton is turned into MoveIt Task Constructor (MTC) plans for left/middle/right cuts (`cut_pattern`: `t_cut` (default), `t_cut_handoff`, `h`, `combined`, single-cut and debug patterns).

The robot controller is reached over CRI/TCP on port 3920; the IP is hard-coded in `src/igus_rebel/include/igus_rebel/Rebel.hpp` (rebuild after changing it).

## Build and test

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble   # first time
colcon build --symlink-install
colcon build --packages-select igus_rebel_moveit_config                 # typical iteration
source install/setup.bash

colcon test --packages-select <pkg> && colcon test-result --verbose
# single gtest (igus_rebel): build/igus_rebel/test_cri_status --gtest_filter='Suite.Name'
# single pytest (run from the package dir): cd src/compal_box_perception && python3 -m pytest -p no:anyio test/test_depth_utils.py -k name
```

Existing tests: `igus_rebel` (gtests `test_cri_status`, `test_cri_framing`, `test_rebel_safety`, and `test_rebel_driver`, which runs the real driver against a fake CRI server on loopback), `compal_box_perception` and `rebel_tcp_gui` (pytest, run by `colcon test`), `tools/session_rag` (pytest). A user-local `anyio` in `~/.local` breaks the system pytest; the packages set `-p no:anyio` in their pytest config, pass it yourself when running pytest by hand. **`igus_rebel_moveit_config` has no tests** — `colcon test` on it reports 0 tests, which is not a pass signal for H-cut changes.

## Packages and how they fit

- `igus_rebel` — ros2_control `SystemInterface` plugin `Igus::Rebel` (CRI protocol: `Rebel.cpp`, `RebelSocket.cpp`, `CriMessages.cpp`, `CriFraming.cpp`; pure safety helpers in `RebelSafety.cpp`) plus `rebel.launch.py`, which starts robot_state_publisher + controllers. `hardware_protocol:=rebel|mock_hardware|gazebo|isaac_topic` selects real/mock/sim/Isaac; the default is `rebel`, so always pass it explicitly. Driver safety behaviour: sends `Override 100.0` after `MotionTypeJoint` on every (re)connect; `on_init` fails if no valid STATUS arrives within 5 s; jog clamped to +/-100 % and non-finite -> 0; zero jog if no `write()` for 50 ms; `read()` returns ERROR (jog forced to zero, driver restart needed) when STATUS is older than 200 ms or after any reconnect. JTC tolerances and gains are in `config/controller.yaml`.
- `igus_rebel_description` — URDF/xacro. `igus_rebel_xeg32_wrapper.urdf.xacro` defines `link6 -> igus_tool0 -> xeg32_tool_tip` (the kinematic TCP). `xeg32_tool_tip` has **no collision geometry**; planning success is not blade-clearance proof.
- `igus_rebel_moveit_config` — MoveIt config, SRDF, and the application nodes in `src/`:
  - `box_h_cut_executor.cpp` — the cutting planner/executor (MTC). Config: `config/box_h_cut_executor.yaml` (its header lists which parameters are read-only, planning or runtime).
  - `calibration_verification.cpp`, servo teleop (keyboard/gamepad).
  - `scripts/camera_pole_collision.py` — publishes the fixed external `camera_pole` collision box from camera TF; H-cut planning waits for it.
  - `launch/move_group.launch.py` — MoveIt + optional executors/teleop/RViz (flags `run_box_h_cut_executor`, `run_teleop` (default false), `run_gamepad`, `run_camera_pole`, `use_gui`).
- `compal_box_perception` (Python) — RGB-D carton/tape-seam detection, ArUco verification, table cloud filtering. `compal_box_msgs` holds its interfaces.
- `igus_rebel_msgs` — robot services (e.g. `SetCartesianIncrement`, digital outputs).
- `handeye_pose_replay`, `rebel_tcp_gui` — calibration/utility tools.

The H-cut executor reads an external canonical snapshot (`external_snapshot_file`, set in the YAML to `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json`, produced by a separate vision repo) and rejects stale snapshots.

## H-cut executor model

Read `src/igus_rebel_moveit_config/CONTEXT.md` (domain vocabulary) and `XEG32_MOVEIT_CHANGES.md` before editing. Key invariants:

- **Plan-only by default.** `~/plan_h_cut` (std_srvs/Trigger) plans, publishes to RViz and retains a plan; nothing moves until `~/execute_planned_h_cut`, which requires `execution_enabled=true` (plus `debug_motion_execution_enabled=true` for staged cut-debug sessions). Both default `false`; the execute response means "dispatch accepted", not "completed".
- Cartesian contact strokes use `min_fraction=1.0`; partial paths are never accepted. Free-space transits go through `NearestBranchTransitPlanner` (pose goal -> nearest exact-pose collision-free IK branch; one `planning_timeout_s` budget; one retry of the original pose goal) around a MultiPlanner (joint interpolation, then the pipeline planner) with separate `transit_*_scaling`.
- Scalings are fractions of `joint_limits.yaml` (0.785 rad/s, 5.24 rad/s²): contact 0.1376/0.0954, transit 0.4127/0.2863 (same physical speeds as the former 0.05/0.15 at 2.16 rad/s, 10 rad/s²). They and other geometry/frame parameters are read-only (`param set` fails; edit the YAML and restart). Planning parameters are rejected with "busy" while a service call or execution runs and while a cut-debug session is active.
- `debug_allow_stale_snapshot=true` is plan-only: every execute of a plan or session loaded with it is refused. `external_snapshot_max_age_s` is 240 s in the YAML (cap 600 s).
- `validateFreeSpaceTransitWrist()` rejects any plan whose free-space transit accumulates more than pi rad of joint6 travel, and fails closed if the examined transits do not match the recorded transit stages. It is a safety gate — do not remove or loosen it to make a plan pass.
- Staged cut-debug sessions: `start_planned -> start_executed -> start_saved`; only the start is planned/executed, the endpoint is reached by teleop and the second save completes the record. No save retains a plan.
- Taught per-cut corrections (`cut_debug_*_correction_file`, `box_h_cut_correction/v1` records) are cut-local and re-mapped onto the newly detected cut basis; applying them is opt-in (`cut_debug_correction_enabled`) and fails closed on any mismatch (records must contain `planning_frame` and `tool_link` matching the active values).
- Carton (`detected_carton`) and `camera_pole` collision objects are required; the scoped tool/carton contact allowance applies only during cut strokes.

## Running in mock (isolated)

Always use an unused `ROS_DOMAIN_ID` with `ROS_LOCALHOST_ONLY=1` so nothing reaches the physical robot's domain:

```bash
export ROS_DOMAIN_ID=165 ROS_LOCALHOST_ONLY=1
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false \
  run_box_h_cut_executor:=true
ros2 control list_hardware_components   # confirm mock_components/GenericSystem before any execute
ros2 service call /box_h_cut_executor/plan_h_cut std_srvs/srv/Trigger '{}'
```

The camera TF (`base_link -> camera_link -> camera_color_optical_frame`) must exist for the pole; in mock it is supplied by static publishers (see `HANDOFF.md`). Historical snapshots can be replayed in mock only by copying them with a refreshed `acquisition_stamp_ns`; never use a re-stamped snapshot to justify physical motion. Mock evidence goes under `scan_results/<topic>_<date>/`.

## Working rules

- Never dispatch motion on the physical robot without explicit operator approval in the current conversation.
- `HANDOFF.md` holds the current work state and open questions; keep it accurate when finishing a work session.
- `python3 tools/session_rag/session_rag.py query "<question>" --format prompt` searches prior agent sessions and source for context; session evidence can be stale, so verify against current code.
