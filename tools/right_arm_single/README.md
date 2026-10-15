# Right ReBeL arm alone: cut the carton's right face and the right half of the top seam

Single-arm stack for the RIGHT arm (controller 192.168.3.12) with the normal `box_h_cut_executor`.
Perception writes the snapshot in the LEFT base frame; `right_cut.py` converts it into the right arm's
base frame (and relabels the carton by 180 degrees about its normal) before the executor sees it, so
the executor's existing `left_only` / `middle_only` code cuts:

| `right_cut.py plan ...` | executor parameters | what is cut |
|---|---|---|
| `face` | `cut_pattern left_only`, `middle_seam_stop end` | the carton's **right** face, front -> back |
| `top` | `cut_pattern middle_only`, `middle_seam_stop end`, `top_cut_min_cut_fraction 0.05` | top seam from the **right end**, as far as the arm reaches (planner pulls the far end back in 5 % steps; at least 5 %). Operator 2026-10-15; the half-seam mode did not plan at most placements |

`plan face` also sets `left_cut_inward_m` (`--inward-mm`, default 1 mm; the left arm's 5 mm was too deep on the right arm).

Nothing here edits or rebuilds the main `build/` / `install/` trees. The executor needed for
`middle_seam_stop` is built into its own `build_right/` + `install_right/` (built 2026-10-15 from the
current source with `colcon build --build-base build_right --install-base install_right
--packages-select igus_rebel_moveit_config --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF`;
rebuild the same way after any change to the executor source or `config/box_h_cut_executor.yaml`, because
`install_right` holds a COPY of the YAML and a copy of the binary).

## Files

| File | Purpose |
|---|---|
| `right_snapshot.py` | converter library + CLI (`convert SRC OUT [--calib F] [--allow-stale]`). Port of `snapshot_frame.hpp` |
| `right_arm.launch.py` | `move_group` + executor (from `install_right`) + camera pole + static camera TF (right frame) + parked left arm |
| `left_arm_parked.py` | publishes the `left_arm_parked` collision box (also `--print-only`) |
| `parked_arm.py` | AABB of the single-arm model's collision geometry at joint zero |
| `right_cut.py` | operator CLI: `plan face\|top`, `execute`, `disarm` |
| `plan_dump.py`, `planning_scene_dump.py`, `joint_state_dump.py` | read-only inspection helpers |
| `mock_restamp.py` | MOCK ONLY: copy an old snapshot with a fresh `acquisition_stamp_ns` |
| `crosscheck_executor.launch.py`, `test/snapshot_frame_check.cpp` | verification helpers (not for the operator) |
| `test/` | `cd tools/right_arm_single && python3 -m pytest -p no:anyio test` (31 tests, no ROS needed) |

## Safety model

* `execute` refuses unless the **driver** is `mock_components/GenericSystem` or the `Igus/Rebel` plugin with
  `robot_ip` exactly `192.168.3.12`. It reads `robot_description` from `/controller_manager` (and
  `robot_state_publisher`, which must agree) and the loaded hardware class from
  `/controller_manager/list_hardware_components`. This stops a right-frame plan from running on the left arm
  (192.168.3.11) or the Wi-Fi address. `move_group` itself does not know the IP: always pass
  `robot_ip:=192.168.3.12` to the DRIVER launch.
* `execute` also refuses unless the executor's `external_snapshot_file` is the converted file, the converted file
  is unchanged since the plan (SHA-256 in `<out>.plan_state.json`), and `cut_pattern` / `middle_seam_stop` /
  overlap still match the plan. On real hardware you must type `yes`.
* `execution_enabled` is set false again at the end of `execute` (also on error / Ctrl-C). `right_cut.py disarm`
  does it by hand. Ctrl-C does **not** stop a trajectory that was already dispatched: use the pendant / e-stop.
* The script refuses to run when `ROS_DOMAIN_ID` is unset and prints the domain it uses.
* The converter refuses: a source with `frame_id != base_link`, one that already carries a `right_arm_single`
  block, a mirrored/degenerate corner labelling, and a stale source (see below).

## Operator steps (real cell)

Every terminal: `source /opt/ros/humble/setup.bash; cd ~/igus_rebel_ros2; source install/setup.bash;
source install_right/setup.bash` (**`install_right` last**, so the newer executor is the one that starts;
the launch refuses to start with an executor that lacks `middle_seam_stop`), then
`export ROS_DOMAIN_ID=<the cell's domain> ROS_LOCALHOST_ONLY=<as on the cell>`.

0. **Park the LEFT arm at the SRDF `home` pose (all joints 0)** and leave it there. The right planning scene
   contains a `left_arm_parked` box for it (AABB of the knife_c model at joints 0, padded 50 mm); it is not
   updated if the left arm is anywhere else. Keep its controller stack stopped or idle (no motion).
1. Driver, right arm: `ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel robot_ip:=192.168.3.12`
   (activate the controllers as you do today).
2. Planning stack: `ros2 launch ~/igus_rebel_ros2/tools/right_arm_single/right_arm.launch.py
   hardware_protocol:=rebel use_gui:=true`. It prints the executor binary it will use (must be under
   `install_right`) and the camera TF it publishes. The executor starts plan-only (`execution_enabled false`).
3. Cameras and perception as for the left arm, **but start `box_perception.launch.py` with
   `publish_calibration:=false`**: otherwise it publishes the camera TF in the LEFT base frame and conflicts
   with the right-frame `base_link -> camera_link` published by `right_arm.launch.py` (the camera pole would
   land in the wrong place). The camera driver supplies `camera_link -> camera_color_optical_frame`; the
   launch only publishes it in mock (`camera_optical_tf:=auto`). Perception keeps writing
   `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json` in the left frame; never edit it.
4. Dry run in the air first (carton removed), then with the carton:
   ```bash
   python3 tools/right_arm_single/right_cut.py plan face      # converts, sets parameters, plans
   ```
   Check the plan in RViz (green/orange path, `detected_carton`, `camera_pole`, `left_arm_parked`).
   ```bash
   python3 tools/right_arm_single/right_cut.py execute        # identity gate, type `yes`
   python3 tools/right_arm_single/right_cut.py plan top
   python3 tools/right_arm_single/right_cut.py execute
   ```
   `plan` takes the newest snapshot by default (`--source FILE` to override; the output goes to
   `/home/farhan/Igus_rebel/outputs/seam_3d_right_base_link.json`, `--output FILE` to change it, before the
   sub-command). Extra planning parameters: `plan face --set left_cut_extend_to_corners=false` (it stays set in the executor until you set it back or restart the stack).
5. If the plan fails, the message names the failing stage; do not edit the YAML on the cell. Try
   `--set left_cut_extend_to_corners=false` (shorter, nominal face cut) and report the message.

### Capture age (240 s)

The executor refuses a snapshot when the **file mtime or `acquisition_stamp_ns`** is older than 240 s
(`external_snapshot_max_age_s`), and refuses `execute` of a plan whose snapshot has aged out. The converter
checks both ages at plan time, keeps `acquisition_stamp_ns`, and gives the converted file the **source
file's mtime** (it never looks fresher than the capture). So plan + inspect + execute of one cut must happen
within 240 s of the capture, and the second cut usually needs a **new capture** (move the carton? no, but
re-run perception so the file is fresh) before `plan top`.

### Known limits (read before cutting)

* **The right TCP is the LEFT arm's TCP** (`xeg32_tool_tip`, knife_c). It has not been calibrated for the right
  arm: blade position/orientation errors of several mm are expected until it is.
* The right base pose is `calibrated_unverified` (`dual_arm_calibration.yaml`, about 5 mm residual after the
  2026-10-15 hover correction). Camera -> right base errors go straight into the cut position.
* All cut tilts/lifts are the left arm's settings; reach of the right arm to the right face with them is only
  verified in mock (see evidence). The camera pole and head are the real obstacles on the right side.
* `left_arm_parked` is only as true as the parked pose; the padding is 50 mm per side.
* Mock-only helpers (`mock_restamp.py`, `--allow-stale`) must never be used to justify real motion.

## Mock (isolated) commands

```bash
source /opt/ros/humble/setup.bash && cd ~/igus_rebel_ros2 && source install/setup.bash && source install_right/setup.bash
export ROS_DOMAIN_ID=196 ROS_LOCALHOST_ONLY=1       # an UNUSED domain
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware                       # terminal 1
ros2 launch ~/igus_rebel_ros2/tools/right_arm_single/right_arm.launch.py \
    hardware_protocol:=mock_hardware use_gui:=false                                           # terminal 2
ros2 control list_hardware_components        # must show mock_components/GenericSystem
python3 tools/right_arm_single/mock_restamp.py scan_results/left_from_corner_20261014/snapshot/capture_1720.json /tmp/cap.json
python3 tools/right_arm_single/right_cut.py --output /tmp/right.json plan face --source /tmp/cap.json
python3 tools/right_arm_single/right_cut.py --output /tmp/right.json execute
```
Evidence of the 2026-10-15 mock session: `scan_results/right_arm_single_20261015/`.
