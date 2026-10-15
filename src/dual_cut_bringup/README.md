# dual_cut_bringup: operator guide for the dual-arm cutting cell

This package starts the whole dual-arm stack with one command and gives you one tool to check it and recover
it after a fault. It never enables motion by itself: the coordinator comes up with `execution_enabled` and both
phase enables **false**, and you switch them on deliberately (last section).

Cameras are **not** part of this launch. Start them separately, as today
(`ros2 launch rebel_cameras cameras.launch.py`).

## What runs

| Part | Started by `dual_cut_system.launch.py` | Notes |
|---|---|---|
| Dual driver | `rebel_dual_table_description dual_table.launch.py` | one `controller_manager`, hardware `arm_left_hw` / `arm_right_hw` in fault group `dual_arm_table`, `joint_state_broadcaster`, `arm_left_trajectory_controller`, `arm_right_trajectory_controller`, TF `world -> camera_link` |
| Dual move_group | `rebel_dual_table_moveit_config move_group.launch.py` (+ camera pole, RViz if `use_rviz:=true`) | scene of the whole cell; the coordinator reads its robot model |
| Per-arm planners | `both_arms_planning.launch.py` | plan-only `move_group` + `box_h_cut_executor` + camera pole + joint-state relay in `/arm_left` and `/arm_right`; they never execute |
| Coordinator | `dual_cut_coordinator coordinator.launch.py` | `/dual_cut_coordinator/{plan,execute,stop,rearm}`, topic `/dual_cut_coordinator/status` |

Start order and delays: driver at 0 s, dual move_group at 2 s, per-arm planners at 4 s, coordinator at 10 s
(`move_group_delay`, `planning_delay`, `coordinator_delay`).

## Start-up

`hardware_protocol` has **no default**. The single-arm `rebel.launch.py` defaults to the real robot; this launch
refuses to guess and aborts when you leave the argument out.

### Mock (no robot)

```bash
export ROS_DOMAIN_ID=<unused, NOT 11> ROS_LOCALHOST_ONLY=1
ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=mock_hardware
```

The launch refuses mock mode on domain 11 (the real robot's domain) and without `ROS_LOCALHOST_ONLY=1`.
Controllers start **active** (`activate_controllers:=auto`).

### Real arms (only with operator approval in the current session)

Terminal 1 (the real cell's ROS domain is 11; the operator chooses the environment):

```bash
ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=rebel \
    left_robot_ip:=192.168.3.11 right_robot_ip:=192.168.3.12
```

(The two addresses default to the ones in `dual_table.launch.py`; the launch refuses identical addresses.) The
launch prints an operator checklist. The trajectory controllers stay **inactive**:

1. **Verify arm identity.** Jog one arm from its pendant and watch `/joint_states`: the `arm_left_*` joints
   must be the ones that move for the arm on the left-IP controller (same for the right). A swapped IP pair
   would send every trajectory to the wrong arm.
2. Terminal 2, activate the controllers with the identity prompt:
   ```bash
   ros2 run dual_cut_bringup dual_arm_recover.py activate
   ```
   It prints both arms' joint positions (compare them with the pendants), refuses if either arm is not standing
   still, and activates only after you type exactly `yes`. `--yes` skips the prompt **in mock only** and is
   refused when any hardware component is not a `mock_components/*` plugin.
3. Check: `ros2 run dual_cut_bringup dual_arm_recover.py status` (exit code 0 means healthy).
4. Cameras in their own terminal, then perception as today.

`activate_controllers:=true` skips step 2's identity check. Do not use it on real hardware.

Other arguments: `calibration_file` (passed to the driver **and** the dual move_group), `use_rviz`,
`run_camera_pole`, `executor_config`, `executor_overrides_left|right`, `coordinator_config` (a file that sets
any execution enable to true is refused), `blade_collision` (default `false`; `true`: the per-arm planner models carry the
`knife_c_blade` collision box (the dual model's blade box, same definition) plus its SRDF disable pair, and the executor
gets `carton_contact_links [xeg32_tool_tip, knife_c_blade]`, so the blade is checked against the carton in every
free-space transit and may touch it only while a cut stage's scoped allowance is installed; needs an executor build
that knows `carton_contact_links`, otherwise every cut plan fails on blade/carton contact).

## Topics and services: single arm vs dual cell

| | Single arm (`rebel.launch.py` + `move_group.launch.py`) | Dual cell |
|---|---|---|
| Joint states | `/joint_states` (`joint1..6`) | `/joint_states` (`arm_left_joint1..6`, `arm_right_joint1..6`); per-arm copies with unprefixed names `/arm_left/joint_states`, `/arm_right/joint_states` |
| Trajectory controller | `rebel_arm_trajectory_controller` | `arm_left_trajectory_controller`, `arm_right_trajectory_controller` |
| Hardware component | one | `arm_left_hw`, `arm_right_hw` (fault group `dual_arm_table`) |
| Driver service | `/igus_rebel/reset_and_enable` | `/igus_rebel_left/reset_and_enable`, `/igus_rebel_right/reset_and_enable` |
| Controller manager | `/controller_manager` | `/controller_manager` (the only one) |
| MoveIt | `/move_group` (single-arm model) | `/move_group` (dual model, scene and RViz only); `/arm_left/move_group`, `/arm_right/move_group` (single-arm model, plan only) |
| Cut planner | `/box_h_cut_executor/plan_h_cut`, `.../execute_planned_h_cut` | `/arm_left/box_h_cut_executor/...`, `/arm_right/box_h_cut_executor/...` (plan, export; never execute) |
| Plan + execute | the executor | `/dual_cut_coordinator/plan`, `/dual_cut_coordinator/execute` (std_srvs/Trigger) |
| Emergency stop of a plan | cancel the goal / `execution_enabled` off | `/dual_cut_coordinator/stop` (latches), `/dual_cut_coordinator/rearm` |
| Status | executor log | `/dual_cut_coordinator/status` (`dual_cut_msgs/CoordinatorStatus`) |
| Camera pole | one object in `/move_group` | in `/move_group` and in both per-arm scenes |

A latched driver fault has no topic or service: see below.

## The right arm: snapshot frame, relabel, half seam

Perception writes the snapshot in the **left** base frame (`frame_id` `base_link`, = `world`). The right arm's
planner is the same `box_h_cut_executor` binary, configured by `config/arm_right_executor_overrides.yaml`:

| Parameter (read-only) | Right arm | Meaning |
|---|---|---|
| `snapshot_frame_mode` | `fixed_transform` | the 11 snapshot points and the normal are expressed in the right executor's own `base_link` (`p = B^-1 p_snapshot`) before anything is validated or planned; `frame_id` must equal `snapshot_frame_id` |
| `snapshot_planning_base_xyz/rpy` | read by `per_arm_planning.launch.py` from `arm_right_base` of the `calibration_file` (the SAME file as the dual driver and dual move_group; `dual_cut_system.launch.py` passes it through) and given as node parameters AFTER the overrides file. The overrides file should not carry them; if it does and they differ the launch logs a WARNING and the YAML still wins | pose of the right base in `world` (URDF fixed-axis rpy); also exported with every plan so the coordinator can check it against the dual URDF (1e-6) |
| `snapshot_relabel` | `rotate_180` | swaps back_left/front_right, back_right/front_left, seam_start/seam_end (and the four side midpoints), keeps the seam centre and the normal: a 180 degree rotation about the carton normal, so handedness and the midpoint formulas are kept (a mirror is detected and refused). The carton's **right** face becomes the executor's "left" cut, so `left_only` plans the right face with every left-arm feature |
| `plan_export_only` | `true` | **required** for the two settings above. Only `plan_h_cut`, `plan_h_cut_from_state`, `get_planned_solution`, `clear_plan` and the diagnostics are advertised; no execute, side-tape, staged, T-cut or cut-debug service. Taught corrections, the side-tape seed and pattern `h` with a joint-space approach are refused |

Planning parameters, set per phase by the coordinator (or by hand for a test):

- `cut_pattern left_only` on the right arm = the carton's right face (front -> back).
- Top half: `cut_pattern middle_only`, `middle_seam_stop center`, `middle_seam_center_overlap_m 0.0075`. The stroke
  ends at the seam centre plus the overlap; the shortening loop is off in this mode (the half plans or the plan
  fails). On the left arm the same three parameters give seam_start (left end) -> centre + overlap; on the right arm
  (relabelled) the half runs from the right end to centre + overlap. Other patterns and the staged/side-tape/cut-debug
  paths refuse `middle_seam_stop center`.

Planning from the measured joint state of the dual cell, and reading the plan back:

```bash
ros2 service call /arm_right/box_h_cut_executor/plan_h_cut_from_state dual_cut_msgs/srv/PlanFromState \
    "{start_state: {name: [joint1, joint2, joint3, joint4, joint5, joint6], position: [0,0,0,0,0,0]}}"
ros2 service call /arm_right/box_h_cut_executor/get_planned_solution dual_cut_msgs/srv/GetPlannedSolution '{}'
```

A plan made from an explicit start state is retained like any other but can **never** be executed
(`execute_planned_h_cut` refuses it); `get_planned_solution` is read-only. Whether the right arm can reach the
relabelled right face with the left arm's tilt settings is **not** guaranteed: a failing plan names the failing
stage; tune only after a reach sweep. Mock result (2026-10-15, `scan_results/dual_cut_p3_regression_20261015/runs/right*`):
the top half plans on both captures (0.7 s; tool tip within 0.01 mm of the world seam line, right end -> centre - overlap);
the right face plans with the nominal cut (`left_cut_extend_to_corners` false, or fixed `left_parallel`) on both captures,
and with the as-configured extended varying cut on 07:49 (77 s, close to the 90 s search timeout) but **not** on 09:20
(every varying profile moves joint3 more than the 1.0 rad aggregate limit; the search times out at 90 s).

## When something faults

The two driver instances are one fault group. If one arm faults (joint error bits, kinematics error, lost
STATUS, e-stop), the driver stops **and latches both** arms: jog is zero, `read()` returns an error, and both
hardware components leave the ACTIVE state. The controllers may stay "active" in `list_controllers`; do not
trust that.

1. Make the cell safe (pendant, e-stop as needed). Find out why it faulted before recovering.
2. `ros2 run dual_cut_bringup dual_arm_recover.py status` shows each hardware component state, controller
   states, fault lines found in `/rosout`, whether the arms stand still, and the coordinator status.
3. Recover:
   ```bash
   ros2 run dual_cut_bringup dual_arm_recover.py recover --root auto     # or --root left | --root right
   ```
   - The driver exposes no "root cause" topic or service. The root-cause arm is only named in the driver's log
     lines (`Fault group ...: component 'arm_left_hw' faulted`, `... stopped because fault-group member
     'arm_left_hw' faulted`). `--root auto` reads `/rosout` (the middleware keeps only about 10 s of it) and
     an optional `--log-file FILE` (the launch log under `~/.ros/log/<run>/` or a saved `journalctl` output).
   - If both arms need recovery and the root cannot be found, `recover` refuses and asks for `--root`: give the
     arm that faulted **first**. Naming the wrong one is caught by the driver, which refuses to activate the
     peer while the root cause is unrecovered.
   - `--dry-run` prints the steps without changing anything.
   - Recovery has two phases. **Phase 1, hardware** (no trajectory controller is activated): for each arm that is
     not ACTIVE, **root first, then the other**: deactivate its trajectory controller (if active) ->
     `reset_and_enable` (best effort) -> `set_hardware_component_state` inactive -> active -> **ensure the drives are
     enabled** -> (root arm only) **wait until the driver reports its root-cause fault cleared**. **Phase 2:**
     every recovered arm stands still, **one** identity confirmation (real hardware), then the controllers are
     activated. Every step is verified and the first failure stops the command with a message.
   - **Drives enabled.** After an e-stop the Enable sent by the lifecycle activation can be lost (the motor supply
     returns a fraction of a second later; the driver then warns "Robot drives disabled" every 10 s). So after
     each activation `recover` calls `/igus_rebel_<arm>/reset_and_enable` and repeats it for up to 5 s. Its reply is
     the signal: the driver waits up to 3 s inside the call for `Motor not enabled` to clear and answers success
     only when STATUS says enabled (independent of /rosout timing). If the drives do not enable (e-stop still
     pressed, supply off), `recover` fails with a message and activates nothing.
   - **Root cleared.** The only observable is the /rosout line "... is enabled with clean STATUS again; its
     root-cause fault is cleared" (500 ms of continuous enabled, clean STATUS). `recover` remembers the /rosout
     position before re-activating the root so the line is not missed, waits up to 4 s for it, and only then
     touches the peer (which it retries for up to 6 s). Not cleared in time: clear failure, peer untouched, no
     controller activated (if the named root was wrong, re-run with the other `--root`).
   - **Hardware ACTIVE but drives disabled** (e.g. a root arm whose Enable was lost earlier, which keeps its peer
     latched): `status` reports "hardware ACTIVE but drives DISABLED -> run recover" (from the driver's 10 s
     throttled warning; advisory) and exits 1. `recover` treats such an arm as needing only the enable step
     (controller deactivated first, no lifecycle cycle). If the log line has expired and the other arm is latched,
     name the active arm with `--root right|left` and it is re-enabled too.
   - `reset_and_enable` is **refused by the driver while the error is latched**. `recover` treats that reply as
     expected for the pre-activation attempt and moves on, because the lifecycle re-activation re-sends the same
     start-up sequence.
   - A controller is never activated unless the hardware of all recovered arms is active with drives confirmed
     enabled, and each arm's joint states are fresh and it stood still for 0.8 s (less than 5 mrad of travel,
     below 0.03 rad/s).
   - On real Igus/Rebel hardware `recover` asks for the same typed `yes` (with both arms' positions) once, after
     all hardware is recovered and before the first controller is activated; anything else activates nothing and
     exits non-zero. `--yes` is refused there. In mock there is no prompt.
4. `recover` **never rearms the coordinator.** Check `status`, then rearm on purpose:
   ```bash
   ros2 service call /dual_cut_coordinator/rearm std_srvs/srv/Trigger '{}'
   ```
   The coordinator clears its retained plan on rearm, so plan again.
5. If `recover` fails at activation with "Refusing to activate ... STATUS is still stale", fix the robot/CRI
   connection and run it again. A component in `FINALIZED` cannot be recovered: relaunch the stack.

## Enabling execution (deliberate, per session)

Nothing here moves a robot until all of these are true. The enables are runtime parameters, false at launch:

```bash
ros2 param set /dual_cut_coordinator execution_enabled true
ros2 param set /dual_cut_coordinator execute_phase_a_enabled true     # and/or execute_phase_b_enabled
ros2 service call /dual_cut_coordinator/plan std_srvs/srv/Trigger '{}'
ros2 service call /dual_cut_coordinator/execute std_srvs/srv/Trigger '{}'
```

Do this on real hardware only with explicit operator approval, after the identity check, a fresh snapshot and
a plan you have looked at in RViz. Set the enables back to false afterwards.

## Verified in mock, and what is not

Verified in mock (domain 201, `mock_hardware`, evidence in
`scan_results/dual_arm_stack_20261014/bringup/`):

- `dual_cut_system.launch.py hardware_protocol:=mock_hardware` brings up the driver, dual move_group, both
  per-arm stacks and the coordinator; both trajectory controllers active; `/arm_X/joint_states` at 100 Hz;
  `/dual_cut_coordinator/status` publishing; execution enables false.
- Launch without `hardware_protocol`, with an invalid value, and mock on domain 11 aborts with a clear message.
- `status` (healthy and with both components set inactive by hand), `recover` (refuses without a root, plans by
  `--root` and `--log-file`, recovers both arms root first, controllers re-activated, coordinator not rearmed),
  `activate` (anything but `yes` activates nothing; refuses while an arm moves; `--yes` works in mock).
- Pure logic by pytest (`test/test_dual_arm_recover.py`, `test/test_dual_cut_system_launch.py`): ordering, root
  selection, `--yes` refusal for non-mock hardware, stillness rule, log parsing, launch defaults.

**Not verifiable in mock** (`mock_components` cannot fake a latch), so untested on real hardware:

- A real fault-group trip and the latched state: the component state after a real latch (assumed to leave
  ACTIVE), what `list_controllers` shows for the controllers afterwards.
- The driver log wording the root-cause detection matches (taken from `Rebel.cpp`; a change there breaks
  `--root auto`, which then falls back to asking for `--root`).
- `/rosout` carrying those lines in the controller_manager process.
- The `reset_and_enable` path on a non-latched arm and its interaction with the lifecycle recovery.
- The peer activation retry timing against the real 500 ms settle rule, and a real arm's joint noise against
  the stillness thresholds.
- Activation with the real `Igus/Rebel` plugin, including the refusal of `--yes` (unit-tested only).
- Launch ordering delays against real start-up times (the driver needs a valid STATUS within 5 s or it exits;
  the rest of the stack does not stop by itself then: stop the launch).
