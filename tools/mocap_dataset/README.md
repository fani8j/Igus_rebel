# tools/mocap_dataset

Tools for the LeRobot v3.0 datasets recorded by `~/robot_remote/recording/record_dataset.py`
(robot_type `igus_rebel_dual_arm_real`, 30 fps, 14-float `observation.state` / `action`:
`arm_left_joint1..6, arm_left_gripper, arm_right_joint1..6, arm_right_gripper`, joints in rad).
The first delivery is `MOCAP_DATASET_FLAP_TOP_OPEN` (12 datasets of one episode each, left arm opening carton flaps,
37-89 s; the right arm columns are all zeros).

| File | Purpose |
|---|---|
| `lerobot_episode.py` | Shared pyarrow-only loader for LeRobot **v3.0 and v2.1**: `load_episode(root, episode_index)`, `list_episodes(root)`, `Episode.arm('arm_left'\|'arm_right', 'state'\|'action')` -> (N, 6), `Episode.active_arm()`, per-episode source metadata (`box_pose`). The dataset is untrusted input: only JSON and parquet are read. |
| `lerobot_to_plan.py` | Episode -> plan JSON for the vendored, safety-gated player `tools/lid_open_friend/scripts/lid_open/publish_plan.py --mode trajectory`. |
| `replay_episode.py` | Episode -> ONE smooth FollowJointTrajectory goal at the recorded timing (time-scaled), with approach, watchdogs and Ctrl-C cancel (see "Replay at recorded speed"). |
| `rerun_view.py` | Rerun viewer (see "Rerun viewer"); also opens the merged output of `preprocess.py` (`--episode N`). |
| `episode_metrics.py` | Shared filter (zero-phase Gaussian), jitter metrics, idle/pause detection, pause-trim planning (`plan_pause_trim`), corruption / joint-limit / start-pose helpers, wrist branch, DTW. |
| `quality_report.py` | Read-only per-episode quality report (HTML + JSON), left-arm or right-arm, v3.0 or v2.1 sources. See "Preprocess for VLA training" and "Right-arm real_right dataset". |
| `preprocess.py` | Merges the episodes into ONE clean LeRobot v3.0 dataset (filtered state, shifted action, raw columns kept; optional idle and pause trimming). |
| `video_reencode.py` | Re-encodes the kept frames of an episode video (libx264, same pixels in, CRF 16, GOP 2), image statistics, packs episodes into v3.0 video files. Used when frames are removed from the middle of an episode or the source is v2.1. |
| `verify_output.py`, `video_frames.py` | Load the output with lerobot and compare frames / state / action with the source (also `--check-loader` for a v2.1 source); PyAV frame readers. |
| `test/test_real_right.py` | v2.1 loader, generalised arm layout, pause trimming end to end, report on a v2.1 dataset: `cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_real_right.py` |
| `test/test_preprocess.py` | `cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_preprocess.py` (whole dir: `... test`) |
| `test/test_replay_episode.py` | `cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_replay_episode.py` |
| `test/test_lerobot_to_plan.py` | `cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_lerobot_to_plan.py` |

Never run Python with the dataset directory as the working directory (untrusted data): pass the dataset as an argument.

## What the recorded numbers mean (checked 2026-10-27 against `~/robot_remote`)

* **Same zero and sign convention as our `/joint_states`.** The dataset was written by `record_dataset.py --source real`
  (`robot_type igus_rebel_dual_arm_real`, no `follower_state` feature, so not the `mocap-real` / `moveit-real` paths).
  `observation.state` is the raw `joint1..6` of `/<side>/joint_states` of the same igus ROS driver family
  (`dual_arm_driver.launch.py`, `config/real_arms.json`: left = 192.168.3.11, as in our cell). Neither the recorder nor the
  teleop (`quest_dual_arm_teleop.py`, backend `real-ros`) applies any offset, sign flip or unit conversion; our driver
  publishes `CRI posJointCurrent * pi/180` with no offset (`igus_rebel/src/RebelSafety.cpp` `JointPositionsRad`).
  robot_remote's own home `(0, 0, 65, 0, 50, 0) deg` and park `all joints 0` use the same zero (0 = arm straight up).
  Not verifiable here: the upstream `Igus_rebel` driver source (not on disk; `igus_ws` is not checked out), and whether the
  friend's physical arm has the same mechanical zero as ours. Check the first motion visually.
* **`action` is the teleop's commanded target**, not a JTC reference: `/quest_teleop/action` = the teleop's `q_cmd` (held
  while the Quest grip is released), which the `real-ros` backend also sends as 1-point trajectories to
  `/left/rebel_arm_trajectory_controller/joint_trajectory`. In the data `state` lags `action` by about 8-9 frames (0.28 s,
  matching the 0.27-0.30 s reaction time measured in robot_remote), max |action - state| 0.08-0.09 rad.
  The converter therefore defaults to `--source state` (what the arm really did).
* The left gripper slot of `action` is the Quest trigger (1.0 in 11 of 12 episodes, a 0..1 ramp in `mocap_left_13`),
  `state` gripper slots are 0 (no real gripper). The converter and the player ignore all gripper columns.
* Many episodes start at `(0, 0, 60, 0, 50, 0) deg`; `mocap_left_13` starts in a flipped-wrist configuration
  (`j4 165, j5 -49, j6 -165 deg`) and several episodes end flipped (`j4 ~168, j6 ~-173 deg`).

## Convert: `lerobot_to_plan.py`

```bash
cd ~/igus_rebel_ros2          # NOT inside the dataset directory
DS=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
python3 tools/mocap_dataset/lerobot_to_plan.py $DS/mocap_left_new_10 --segment-s 5 --out /tmp/new_10.plan.json
```

Without `--out` nothing is written (summary and checks only). Options: `--arm arm_left` (default; an arm whose data is all
zeros is refused), `--source state|action`, `--start-s/--end-s`, `--segment-s` (default 10), `--hz` (5), `--min-step-deg` (1),
`--limit-margin-deg` (2), `--max-jump-deg` (5.7), `--park last|start`, `--max-speed` (the player speed the estimate is
computed for), `--urdf/--base-link/--link6/--tip-link` (FK z summary).

* **Fail closed** (exit 2, no file): non-finite values, an all-zero arm, a joint outside the real URDF limits
  (`src/igus_rebel_description/urdf/igus_rebel2.urdf`, 1 deg inside the controller limits) minus 2 deg, a frame-to-frame jump
  above 5.7 deg (wrap / glitch), non-increasing timestamps. The player then applies its own checks again (limits, margin 1 deg,
  `--max-step-deg`).
* **Plan layout.** `start_q` = first point. The slice is cut into ~`--segment-s` chunks of data time; each chunk is ONE `arc`
  step = one JointTrajectory = one Enter and one arrival check in the player (points after the first inside a chunk are chained
  with no confirmation, so a chunk cannot be paused; only the e-stop stops it). `park_q` = last point: the player's final
  "park" group is a no-op move (it still asks Enter; the player's own `--no-park` drops it). `--park start` returns to the first
  point by a joint-space straight line instead (opt-in; it crosses the whole workspace).
* **Timing.** The player ignores the recorded timestamps: every point gets `max(max|dq| / max_speed, 0.2 s)` (first point of a
  trajectory at least 1 s). So the point spacing decides the speed. The converter resamples at 5 Hz and drops points closer than
  1 deg (max-norm) to the last kept one: dwell phases do not turn into hundreds of 0.2 s stop-and-go segments, steps above 1.7 deg
  play at exactly `--max-speed`, and slower parts play slower than that, never faster. The data peaks at ~0.5 rad/s, so at the
  default `--max-speed 0.15` the replay is x1.0-1.6 of the recorded duration (per-episode numbers are printed). At `--max-speed
  0.05` multiply by ~3. The player's own dry-run total only counts the first point of every arc: use the converter's replay line.
* **Shape.** The summary prints the downsampling path error (full-rate samples vs. the kept polyline, joint space): 0.25-1.8 deg
  over the 12 episodes.
* **FK summary** (informational): z of a link in the arm base frame, e.g. for the cell's knife TCP
  `xacro src/rebel_dual_table_description/urdf/rebel_dual_table.urdf.xacro hardware_protocol:=mock_hardware > /tmp/dual.urdf` then
  `--urdf /tmp/dual.urdf --base-link arm_left_base_link --link6 arm_left_link6 --tip-link arm_left_knife_c_tcp_point`.
  The table top is at z = -0.001 in that frame; over the 12 episodes the modelled knife TCP point (`arm_left_knife_c_tcp_point`) stays 4-14 cm above the table
  plane (lowest: `mocap_left_new_9`, 4.1 cm) and the link6 flange 8-11 cm. Blade/holder geometry and the carton are NOT modelled in this check.

## Replay on the real arm

Nothing here was run on the real robot. Mock evidence (domain 207, `mock_hardware`): `scan_results/mocap_flap_replay_20261027/`.

**Checklist before every run**
* E-stop at hand, nobody in the workspace, a second person if possible.
* Carton/box placed as in the recording (the replay is joint-space only; a different box pose means the knife is in the wrong place).
* Right arm parked clear of the left arm's path (the right controller is activated too by `dual_arm_recover.py activate`, but nothing is sent to it).
* No teleop, servo, executor or coordinator running on `arm_left_trajectory_controller` (`ros2 topic info -v /arm_left_trajectory_controller/joint_trajectory` shows only the player).
* First run at low speed (`--max-speed 0.05`) and short chunks (`--segment-s 5`), Enter at every group, hand on the e-stop.
* Pick an episode whose start is a normal pose (`mocap_left_15`, `mocap_left_new_*` start at `0, 0, 60, 0, 50, 0` deg). The first
  group moves the arm from where it stands to `start_q` by a joint-space straight line, up to 165 deg of joint travel for `mocap_left_13`.

**Terminal 1 - bringup** (the real cell is ROS domain 11; the operator runs everything below on 11, mock tests use another domain):
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=rebel \
    left_robot_ip:=192.168.3.11 right_robot_ip:=192.168.3.12
```
(or `dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=rebel`, which starts this plus more). With `hardware_protocol:=rebel`
the trajectory controllers stay INACTIVE. Verify the arm identity (jog one arm from its pendant, watch `/joint_states`: the
`arm_left_*` joints must be the arm on 192.168.3.11), then in terminal 2:
```bash
ros2 run dual_cut_bringup dual_arm_recover.py activate      # prints both arms' joints, asks for 'yes'
ros2 run dual_cut_bringup dual_arm_recover.py status        # exit 0 = healthy
```

**Terminal 2 - convert, dry-run, publish**
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
cd ~/igus_rebel_ros2
DS=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
PUB=tools/lid_open_friend/scripts/lid_open/publish_plan.py

python3 tools/mocap_dataset/lerobot_to_plan.py $DS/mocap_left_new_10 --segment-s 5 --max-speed 0.05 --out /tmp/new_10.plan.json
python3 $PUB /tmp/new_10.plan.json --mode trajectory                      # dry run: nothing is sent

python3 $PUB /tmp/new_10.plan.json --mode trajectory --publish \
    --state-topic /joint_states \
    --traj-topic /arm_left_trajectory_controller/joint_trajectory \
    --traj-joints arm_left_joint1 arm_left_joint2 arm_left_joint3 arm_left_joint4 arm_left_joint5 arm_left_joint6 \
    --max-speed 0.05 --no-park
```
The player asks for a typed `yes`, then Enter per group (1: approach to `start_q`, then one per chunk), prints the arrival error
after each group (> 3 deg asks whether to continue), and `q` / Ctrl-C stops sending (a trajectory already sent is finished by the
controller: only the e-stop stops it at once). `/joint_states` comes from `joint_state_broadcaster` (RELIABLE, TRANSIENT_LOCAL); the
player subscribes best-effort, which is compatible. The player does nothing if the controller is inactive: check
`ros2 control list_controllers` first. `--only-task 0 --until-step N` plays only the first N chunks.

**Limitations**
* **No collision checking.** It is a joint-space replay: the box, table, camera pole, the other arm and the tool are not
  protected, and neither is the approach move to `start_q`. The converter checks joint limits and jumps only.
* **Tool and cell differ.** The recording was made in robot_remote's cell with that arm's tool at the time (the `xeg32` wrapper
  and the Isaac rig model in robot_remote); our left arm carries the knife tool variant C on the dual table. Joint angles are
  replayed, so the TCP is in a different place if the tool or the arm mounting differs. No tool/box pose is stored in the dataset
  (`box_pose` in the plan is a zero placeholder).
* Real tracking differs from the mock: the real JTC uses velocity commands with p=1, ff=0.4 and has ~0.3 s reaction time (about 2.5 deg at
  0.15 rad/s); the mock follows ideally. The `trajectory: 0.25 rad` path tolerance of `controllers.yaml` aborts a runaway joint.
* Closest approach to a URDF joint limit over the 12 episodes: 4.1 deg (`mocap_left_new_1`, joint3); 5.0 deg on joint 6 / joint 4 in most of them (flipped wrist).

## Replay at recorded speed: `replay_episode.py`

The converter + player above ignore the recorded timestamps (constant `--max-speed` per point, one goal per chunk, so the motion
is stop-and-go and does not look like the recording).  `replay_episode.py` keeps the timing: the whole slice goes to
`/arm_left_trajectory_controller/follow_joint_trajectory` as ONE goal with `time_from_start = recorded time / --speed`.
Mock evidence (domain 207): `scan_results/mocap_flap_replay_20261027/replay_episode/`.  **Not run on the real arm yet.**

```bash
cd ~/igus_rebel_ros2          # NOT inside the dataset directory
DS=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
python3 tools/mocap_dataset/replay_episode.py $DS/mocap_left_new_10 --speed 0.5          # dry-run, prints the summary
```

**Trajectory build.**  Slice checks (`lerobot_to_plan.check_series`: URDF limits - 2 deg, jump, NaN, all-zero arm) -> linear
upsample to 200 Hz -> Gaussian low-pass (`--smooth-s` 0.2 s, odd reflection at the ends) -> velocity = derivative of the smoothed
signal -> knots every 0.1 s (`--rate-hz`) with positions AND velocities, so the controller interpolates cubic splines.  Zero
velocity at both ends by a smoothstep time-ease along the same path (min 0.2 s; where the recording is still moving at the slice
end, `3 * v_edge / --max-acc`, up to ~1.5 s: new_9, 13).  The path is unchanged, only the first/last part takes longer (printed).
The summary reports the deviation from the recording (smoothed time-aligned 0.60-0.74 deg, geometric path 0.46-0.60 deg over the
12 episodes; refuse above `--max-dev-deg` 1) and per-joint peak velocity and acceleration at the chosen speed.

**Limits (refuse, never re-time).**  `--max-vel 0.35` rad/s (the teleop was capped at 0.3; smoothed data peaks at 0.26-0.28 at
speed 1.0) and `--max-acc 0.55` rad/s^2 (controllers.yaml: the real plant follows ~0.52 rad/s^2; the recorded motion itself has
0.42-0.51 after smoothing, so 1.0 passes for all 12 episodes with small headroom, 1.3 does not).  If exceeded the script prints the
`--speed` that passes.  `--speed` > 1.0 needs `--allow-faster`, cap 1.5, minimum 0.05.

**Execute flow** (`--execute`, needs a TTY).  Fresh `/joint_states` (< 0.5 s, all six joints) -> action server present -> warning
if anyone else publishes `/arm_left_trajectory_controller/joint_trajectory` (a topic message REPLACES an action goal) -> live
joints and start pose printed -> typed `yes` -> **approach** (only if the arm is > 0.5 deg from the start pose): a separate smooth
joint-space goal at `--approach-vel` 0.10 rad/s, behind its own Enter, arrival must be <= 1 deg within 3 s (`goal_time`),
otherwise abort; no collision check (straight line in joint space) -> Enter -> **replay**, one goal, progress line every second
(time, tracking error from the action feedback, worst error, state age).  Watchdogs cancel the goal: tracking error >
`--max-track-err-deg` (8; controllers.yaml notes 5-8 deg normal lag at faster test speeds; ~4 deg estimated at 0.3 rad/s from the 0.08 s delay + 0.16 s lag), `/joint_states`
older than `--stale-cancel-s` (0.3), no feedback for 1 s.  No park move.  Gripper columns and the other arm are never used.

**Cancel and crash.**  Ctrl-C (also SIGTERM/SIGHUP) cancels the goal and the script waits for the cancel response (a second
Ctrl-C re-sends it); exit code 130 (4 for a watchdog cancel).  On cancel the Humble `joint_trajectory_controller` calls
`set_hold_position()`: it holds the CURRENT position (the arm decelerates with the plant lag; at 0.26 rad/s expect a few degrees
of coast, not an instant stop; the mock stops within one sample because it is ideal).  If the script CRASHES or is killed
(SIGKILL, terminal closed hard), a goal that was already sent keeps running to its end: **the E-stop is the only hard stop.**

**Real cell, operator only (ROS domain 11; bring-up and `dual_arm_recover.py activate` as in "Replay on the real arm" above):**
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
cd ~/igus_rebel_ros2
DS=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
ros2 topic info -v /arm_left_trajectory_controller/joint_trajectory      # no other publisher
python3 tools/mocap_dataset/replay_episode.py $DS/mocap_left_new_10 --speed 0.5                  # dry-run (reads /joint_states, sends nothing)
python3 tools/mocap_dataset/replay_episode.py $DS/mocap_left_new_10 --speed 0.5 --execute        # FIRST real run: half speed
python3 tools/mocap_dataset/replay_episode.py $DS/mocap_left_new_10 --speed 1.0 --execute        # then the recorded speed
```
Options: `--arm`, `--source state|action`, `--start-s/--end-s` (pick a slice that starts at rest; a start in the middle of a motion is
eased in, which changes the timing), `--approach-vel`, `--max-vel/--max-acc/--max-dev-deg`, `--smooth-s`, `--rate-hz`,
`--action`, `--state-topic`, `--offline` (dry-run without looking at `/joint_states`).  Same limitations as above: joint-space
replay without collision checking, box placed as in the recording, tool/cell differ from the recording's.

**Mock results (2026-10-28, `mocap_left_new_10`, speed 1.0, `mock_hardware`, `dual_table.launch.py mock_move_joint:=false`).**
Approach 60 deg in 15.7 s then the 37.2 s replay as one goal: SUCCEEDED, worst tracking error 0.15 deg, end error 0.00 deg;
measured `/joint_states` vs the raw recording: rms 0.10 deg, max 0.65 deg time-aligned (same 37 s timing), geometric 0.64 deg max.
Ctrl-C at 15 s: cancel accepted, goal CANCELED, joints and controller reference constant afterwards (exit 130).  Tracking-error and
stale-state watchdogs verified with tiny thresholds (exit 4).  The mock follows ideally: real tracking lag is not tested.

## Rerun viewer

`rerun_view.py` shows one episode in Rerun (rerun-sdk 0.26): an animated model of the real dual-arm cell
(`rebel_dual_table.urdf.xacro`, left arm driven by the recorded joints, right arm at zero), the knife TCP path
(`arm_left_xeg32_tool_tip`), the three AV1 videos (head, wrist_left, knife) and per-joint state/action plots.
No ROS sourcing needed (xacro runs in a subshell); run it from outside the dataset directory.

```bash
D=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
python3 tools/mocap_dataset/rerun_view.py $D/mocap_left_13                        # spawn the viewer
python3 tools/mocap_dataset/rerun_view.py $D --episode-dir mocap_left_new_3       # pick from the parent dir
python3 tools/mocap_dataset/rerun_view.py $D/mocap_left_13 --save /tmp/l13.rrd    # file; open with `rerun /tmp/l13.rrd`
```

Options: `--source state|action|both`, `--no-video`, `--no-robot`, `--video-frames` (PyAV decode fallback if the
viewer can't play AV1). The .rrd embeds the videos (~125-190 MB per episode).

## Mock test results (2026-10-27)

Domain 207, `dual_table.launch.py hardware_protocol:=mock_hardware mock_move_joint:=false` (`mock_components/GenericSystem`,
confirmed with `ros2 control list_hardware_components`), `mocap_left_13` first 20 s (61 points, 2 arcs, planned replay 13.3 s + 8.0 s),
`publish_plan.py --mode trajectory --publish` with `yes` + 4 Enters on stdin. Files: `scan_results/mocap_flap_replay_20261027/`.

* The approach from the mock's all-zero pose to `start_q` (165.3 deg max joint change, 19.2 s) is its own confirmed group, then
  seg 1, seg 2, and the no-op park (0.0 deg, 1.0 s). Measured motion windows from `/joint_states`: 19.25 s, 13.32 s, 7.98 s.
* Arrival errors printed by the player: 0.00, 0.00, 0.00, 0.00 deg.
* Achieved path vs. the full-rate recorded `state` (joint-space distance to the polyline): max 0.19 deg; achieved vs. the commanded
  polyline: 0.000 deg; end point error 0.000 deg. JTC reference speed (max-norm) 0.14 rad/s mean, 0.15 max in the arcs.
* All 12 datasets converted and passed the player's dry-run (`scan_results/.../plans/*.convert.txt`, `*.player_dryrun.txt`).

## Preprocess for VLA training

```bash
cd ~/igus_rebel_ros2            # NOT inside the dataset directory; HF_HUB_OFFLINE=1 avoids hub lookups for local/ repo ids
DS=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
# 1. quality report (read-only): report.html + report.json + PNGs; suggests keep/review, the operator decides
nice -n 19 python3 tools/mocap_dataset/quality_report.py $DS --out scan_results/mocap_dataset_prep_20261029
# 2. one merged, jitter-filtered dataset (never write under / : the disk is nearly full)
OUT="/media/farhan/My Passport/igus_nvidia/datasets/flap_top_open_clean_v1_all12"
HF_HUB_OFFLINE=1 python3 tools/mocap_dataset/preprocess.py $DS --out "$OUT" --verify
# subset: --include 13,new_1,new_2 | --exclude 15,new_9,13   (names with or without the mocap_left_ prefix)
# options (default off): --trim-idle  --task "open the box flaps"  --left-only ; tune: --sigma-s 0.08 --action-offset-frames 1
# 3. re-check an existing output against the source
HF_HUB_OFFLINE=1 python3 tools/mocap_dataset/verify_output.py "$OUT" $DS
python3 tools/mocap_dataset/rerun_view.py "$OUT" --episode 5 --save /tmp/x.rrd   # the .rrd embeds the whole mp4s (~0.5 GB)
```

**Jitter finding (12 episodes).** The measured state is a 30 Hz sample of a joint-state stream that is not phase-locked to
the recorder, so the frame step alternates (e.g. 0.36, 0.24, 0.36, 0.73 deg) while the arm moves smoothly. The velocity
spectrum has lines near 3.5, 6.3 and 9.1 Hz on a broad 3-15 Hz floor (velocity variance 3-15 Hz: 2e-3 (rad/s)^2 for the state,
2e-5 for the teleop action, which is a smooth rate-limited command). Intended motion is below ~2 Hz. So it is mostly sampling
jitter, not hand tremor (the action has none). Filter: zero-phase Gaussian (`replay_episode.gaussian_smooth`, odd reflection =
endpoints exact), sigma 0.08 s (-3 dB 1.7 Hz, -13 dB at 3.5 Hz, -43 dB at 6.3 Hz), applied to joint columns only, the same for
all episodes. Max deviation from the raw state 0.25-0.62 deg (0.27 typical); about 0.4 deg of that is the unavoidable rounding
of velocity kinks, the same rounding happens when the filter is run on the clean teleop action. The images show the real arm,
so state and image differ by at most that (<= 0.62 deg). Mean over episodes and joints, raw -> filtered: HF (>3 Hz) position
rms 0.051 -> 0.0039 deg, HF velocity rms 0.040 -> 0.0014 rad/s, jerk rms 102 -> 1.0 rad/s^3, zero-step frames per s of motion
0.13 -> 0. Sign reversals per s of motion hardly change (0.18 -> 0.16): the jitter is speed modulation, not direction flips.
Per-episode numbers: `PREPROCESS.json` and `report.json`.

**Action.** `action[t]` = filtered state at t + 1 (default `--action-offset-frames 1`): the pose the arm reaches next frame,
absolute joint target, always reachable, consistent with the images one frame later. The original teleop command is kept in
`action_teleop` (it leads the state by 8-9 frames = 0.28 s; `PREPROCESS.json` has the best lag per episode). A VLA converter can
pick `action_teleop`, or chunk future states.

**Output layout.** One v3.0 dataset, episodes 0..N-1 in natural order of the source names (printed, saved in
`meta/source_episodes.json`), 14-D state/action as the source (grippers 0), extra features `observation.state_raw`,
`action_teleop`, the 3 videos stream-copied into <=200 MB files (no re-encode), `meta/stats.json` incl. q01..q99,
`PREPROCESS.json` (command, source, filter params, per-episode deviation/jitter). A failed run leaves no partial OUTDIR;
OUTDIR must not exist (`--force` replaces only an existing lerobot dataset or an empty directory).
`--trim-idle` shifts the per-episode video `from_timestamp` (works with lerobot's torchcodec loader, verified pixel-identical);
image stats are then still those of the whole source episode.

**lerobot tools.** Used from lerobot 0.4.4: `aggregate_videos` (concat demuxer, stream copy), `compute_episode_stats` /
`aggregate_stats`, `get_hf_features_from_features`, `write_info/stats/tasks`. Not used: `aggregate_datasets`/`merge_datasets`
(cannot change the state/action columns or trim) and `add_features`/`modify_features` (rewrite the whole dataset and still need
the columns computed). Global vector stats are computed exactly on the concatenated data, image stats are the count-weighted
aggregation of the source per-episode image stats.

**Quality report flags** (suggestion only; thresholds in `report.json` `flag_rules`): wrist branch differing from the majority
(or both), start pose > 10 deg from (0, 0, 60, 0, 50, 0), DTW/duration outliers, long idle head/tail, mid pause >= 3 s, filter
deviation > 1 deg, limit/jump check (same as `lerobot_to_plan.py`), video frame count != rows, right arm not zero.
Result for the 12 episodes: `mocap_left_13` review (both branches, starts flipped 165 deg, 5.2 s pause), `mocap_left_15` and
`mocap_left_new_9` review (opposite wrist branch), the other nine keep.


## Right-arm real_right dataset

`real_right` (`robot_type igus_rebel_dual_arm_rig_arm_right`, LeRobot **v2.1** in the GR00T layout: 20 episodes, 28,264 frames, 30 fps,
6-D `observation.state` / `action` = `arm_right_joint1..6`, 3 H.264 1152x648 cameras `head`, `wrist_left`, `wrist_right`, `box_pose` in
`meta/episodes.jsonl`) is handled by the same tools, generalised (no fork). Raw copy:
`/media/farhan/My Passport/igus_nvidia/datasets/real_right_20261029_raw/real_right` (never modified). Findings and numbers:
`scan_results/real_right_prep_20261030/` (`report.html`, `report.json`, `preprocess.log`, `verify.log`).

```bash
cd ~/igus_rebel_ros2            # NOT inside the dataset directory
RAW="/media/farhan/My Passport/igus_nvidia/datasets/real_right_20261029_raw/real_right"
OUT="/media/farhan/My Passport/igus_nvidia/datasets/real_right_clean_v1"        # never write datasets under / (24 GB free)
# 0. v2.1 reader vs. an independent pandas read (numeric columns identical, video packets == rows)
python3 tools/mocap_dataset/verify_output.py --check-loader "$RAW"
# 1. quality report (read-only, ~2 min with --video-consistency): thumbnails of head + wrist_right, box pose, held-out suggestion
nice -n 19 python3 tools/mocap_dataset/quality_report.py "$RAW" --out scan_results/real_right_prep_20261030 --arm arm_right \
    --video-consistency wrist_right --rule end_pose_review_deg=15 --title "real_right (right arm, 20 episodes): per-episode quality report"
# 2. ONE clean v3.0 dataset: episode 19 dropped, jitter filter on, pauses trimmed, all 3 cameras re-encoded (~7 min, 4 workers, nice 19)
HF_HUB_OFFLINE=1 python3 tools/mocap_dataset/preprocess.py "$RAW" --out "$OUT" --exclude 19 --trim-pauses
# 3. check against the source with lerobot's own loader (items + decoded frames of every episode, PSNR samples)
HF_HUB_OFFLINE=1 nice -n 19 python3 tools/mocap_dataset/verify_output.py "$OUT" "$RAW" --psnr-samples 12
# 4. rerun file of one episode (no GUI; the .rrd embeds the mp4 files, so keep it on the external drive)
python3 tools/mocap_dataset/rerun_view.py "$OUT" --episode 5 --arm arm_right --save "/media/farhan/My Passport/igus_nvidia/datasets/_scratch_real_right_prep/real_right_clean_v1_ep5.rrd"
# 5. tests
cd tools/mocap_dataset && python3 -m pytest -p no:anyio test
```

**v2.1 handling.** `lerobot_episode.load_episode` reads v2.1 in place (per-episode parquet and mp4, `meta/episodes.jsonl`, extra keys such
as `box_pose` in `Episode.meta`, extra columns such as `next.reward` in `Episode.extras`). lerobot 0.4.4's own converter
(`lerobot.datasets.v30.convert_dataset_v21_to_v30`) was tried on a COPY: it needs `meta/episodes_stats.jsonl` (the source has none), works in
place on `<root>/<repo_id>`, converted data and videos by stream copy in 15 s and then stopped with `FileNotFoundError ... episodes_stats.jsonl`.
Faking the statistics file would not help anyway: pause trimming has to re-encode, so the output videos are written by
`video_reencode.py`, and the image statistics are computed from the decoded source frames with lerobot's own sampling helpers.

**What the "still" frames are.** The unfiltered per-frame joint speed is below 0.02 rad/s in 20-41 % of the frames of every episode, but that
is not a pause: in every episode about 20 % of the frames *while the arm moves* have exactly zero step in all joints (stale / repeated state
samples, the joint-state stream is slower than the 30 Hz recorder; the camera image still moves in those frames). After the zero-phase filter
only 1.7-26 % (mean 7.5 %) of the frames are still (speed <= 0.02 rad/s), and real pauses of >= 1 s exist in six episodes only
(0: 3.7 s, 1: 1.5 s, 8: 5.5 s + 3.2 s, 11: 8.8 s, 13: 4.3 s, 17: 7.3 s); the other episodes only have natural 0.4-0.7 s stops.

**Jitter verdict: present, filter applied.** Velocity variance 3-15 Hz (1.7e-3 (rad/s)^2) is 2.4x that of the intended motion below 2 Hz (7e-4), broadband (no
single line like the 6.3 / 9.1 Hz ones of the left set), the same in `action` (= next raw state), 4e-6 after the filter. Zero-phase Gaussian sigma 0.08 s: position HF rms 0.049 -> 0.004 deg, velocity HF rms
0.038 -> 0.0015 rad/s, zero-step frames while moving 6.2 -> 0 per s, max deviation from the raw state 0.73 deg (mean 0.51), the same rounding of
velocity kinks as in the left-arm set. Without the filter a policy that predicts `action - state` sees a step that is 0 in 20 % of the frames and
twice as big in the next one. See "Jitter finding" above for the method.

**The cameras repeat frames too.** Consecutive decoded frames of the source videos are identical (mean abs difference < 0.02/255 at
64x36 grey) in 59-62 % of the frames of `head` and 26-32 % of `wrist_left` / `wrist_right` (episodes 3 and 12 measured): the effective camera
rate is far below 30 Hz. They are kept as recorded (alignment with the state is what counts); a VLA that stacks or strides frames should
know it.

**Pause trimming (`preprocess.py --trim-pauses`).** A pause = a still run (filtered arm speed <= 0.02 rad/s) of >= 1.0 s between the first and
last movement (`--pause-min-s`; the natural stops of the other episodes are <= 0.7 s, so 1.0 s sits in the gap). The arm settles exponentially
into the stop (speed 0.02 -> 0.005 rad/s takes ~1 s and moves the joints by up to 0.6 deg), so only the *settled* part (speed <= 0.005 rad/s,
`--pause-core-speed`) is cut: its first and last 0.2 s stay (`--pause-keep-s 0.4`), the middle goes. The deceleration, the rest and the
start of the next move are exactly as recorded, nothing that moves is removed, and the pose on both sides of a cut differs by at most
0.25 deg (`--pause-max-join-deg`, a cut is skipped otherwise; measured 0.13-0.17 deg = one ordinary frame step). Videos: the kept frames are
decoded (yuv420p), handed unchanged to libx264 (CRF 16, GOP 2 like lerobot's encoder, no B-frames), and the episodes are packed into
<= 200 MB files with lerobot's `concatenate_video_files`; `timestamp` stays frame_index / 30 (contiguous), `action[t]` is the filtered state of the
next *kept* frame, `observation.state_raw`, `action_teleop` (the untouched source `action`), `source_frame_index` and `source_timestamp` keep
the traceability, `box_pose` is a column of `meta/episodes`, everything else of the source episode (`source`, `rig`, ...) is in
`meta/source_episodes.json`. The source `annotation.human.task_description` (GR00T) is dropped, `next.reward` / `next.done` are carried.
Without `--trim-pauses` a v2.1 source is still re-encoded (all frames); a set of one-episode v3.0 datasets without a cut stays a stream copy.

**Result `real_right_clean_v1`** (`--exclude 19 --trim-pauses`, defaults otherwise): 19 episodes, 26,906 frames (729 = 2.64 % removed, only in
episodes 0 (-68), 8 (-179, two cuts), 11 (-219), 13 (-86), 17 (-177); the other 14 are untouched), 3.27 GiB, 12 video files per camera,
`state` / `action` 6-D float32. Re-encode quality against the source frames at the recorded source index (297 frames per camera, spread
over every episode): PSNR head 44.2 dB (min 37.7), wrist_left 46.8 (41.2), wrist_right 48.8 (42.7). `verify_output.py`: 176 checks ok, 0 failed
(lerobot `LeRobotDataset`: len 26,906, 19 episodes, items at start / second / middle / last two frames and on both sides of every cut with
decoded frames of all 3 cameras, the source frame at `source_frame_index` is the best match wherever the window can tell neighbouring frames
apart, numeric columns equal to the source at the kept frames, `timestamp == frame_index / 30`).

**Report additions** (per episode, in `report.json` / `report.html`): mid pauses (n, total, longest, share), raw-slow vs filtered-still
frame shares, what `--trim-pauses` would remove, box pose, start / end pose variants (poses within 7 / 12 deg), closest joint-limit
approach (URDF limit, `igus_rebel2.urdf`), corruption (impossible steps above 5.7 deg/frame, all-zero frames), stale-sample share, holdout
suggestion (typical, not near a limit, no long pause, interior box pose, never a member of an identical-box-pose group; with relaxation if
too few episodes are `keep`). Rules that would change the left-arm report are off unless set (`--rule end_pose_review_deg=15`).
State-vs-video consistency: the image motion of the wrist camera correlates only weakly with the joint speed in ALL episodes (0.0-0.4: noisy
camera, duplicated frames, auto exposure), so it is not used for flagging (numbers are in `report.json`); the corruption of episode 19 is
established by the state itself (13 impossible steps, 15 all-zero frames at the start) and visually by the contact sheet.


## Right-arm 14-D mocap set (31 Oct) and the TurboVLA runs

`mocap_right_front_flap_31oct_raw/{mocap_right_2,mocap_right_3,mocap_right_farhan}` = three **multi-episode** LeRobot v3.0 datasets (14-D, right arm
in channels 7..12, AV1 640x480 `head` + `wrist_right`, task "open the box flaps"). `list_episodes` / `preprocess.py` take the folder as it is; episode names are
`<folder>_ep<NNN>` (`mocap_right_2_ep008`). Multi-episode sources are always re-encoded (the output keeps AV1 -> H.264). Review:
`scan_results/mocap_right_31oct_review_20261031/`.

```bash
RAW="/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_front_flap_31oct_raw"
OUT="/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_31oct_clean_v1"
DROP=mocap_right_farhan_ep006,mocap_right_2_ep002,mocap_right_farhan_ep001,mocap_right_2_ep000,mocap_right_2_ep004,mocap_right_farhan_ep000,mocap_right_farhan_ep002,mocap_right_farhan_ep003
cd /tmp && HF_HUB_OFFLINE=1 python3 ~/igus_rebel_ros2/tools/mocap_dataset/preprocess.py "$RAW" --out "$OUT" --exclude $DROP --arm arm_right --trim-pauses --workers 4 --verify
# TurboVLA bundle: --src can be repeated (an episode name must be in exactly one source; 6-D and 14-D sources, different video sizes are fine)
python3 tools/mocap_dataset/to_turbovla.py bundle --src "$OUT" [--src real_right_clean_v1] --arm right --handoff-source <bundle>/source \
    --train a,b,c --validation d,e --task "open the front flap of the cardboard box" --out <dir>/pause_bundle
# augmented sets (augment_state_noise.py / augment_bgswap.py: source_episodes.json has `copy_index`): `--src [LABEL=]PATH`. Without a label copy 0 is
# `<source>` and copy k `<source>__c<k>`; with a label EVERY episode of that set is `<source>__<LABEL><k>` (so the plain originals from clean_v1 and the
# copies of several augmented sets can be listed side by side). Split train/validation by SOURCE episode yourself (all copies of a held-out source out).
# A pause the handoff's build() would refuse because the FILTERED state/action move > 3 mrad inside the stationary span (decaying state-noise offset) is
# kept instead of aborting the bundle (listed under rejected_pause_candidates in pause-plan.json).
python3 tools/mocap_dataset/to_turbovla.py bundle --src "$OUT" --src sn=<aug_statenoise> --src bg=<aug_bgswap> --arm right ... --train a,a__sn1,a__bg2,b,... --validation d,e
# offline metrics per checkpoint, optionally per source (episode_index of the validation set)
python3 tools/mocap_dataset/eval_predictions.py --arm right steps_*_predictions.npz --group new=0,1,2 --group real=3,4 --json out.json
```

`eval_predictions.py` also reports the share of frames under the client's gates (first_step 0.15 rad on the raw chunk, target_step 0.10 rad on the
causal ACT-ensembled target, decay 0.01; offline approximation, the live state follows the target). `--trim-idle` is off (as for real_right_clean_v1),
but the handoff's own pause rule in `to_turbovla.py bundle` (teleop command flat >= 1.5 s, 0.5 s guards) removes about 2-4 s of the still head of every
episode and the longest still tail, 6.8 % of the frames of the 23 episodes.
