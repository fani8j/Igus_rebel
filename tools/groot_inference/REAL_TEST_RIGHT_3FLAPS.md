# GR00T real_right_3flaps_35ep_15fps (RIGHT arm, front / back / right flap): test on the real robot

Checkpoint `~/groot/checkpoints/real_right_3flaps/real_right_3flaps_35ep_15fps/checkpoint-3200`, trained at 15 fps on 35 episodes:
the 20 `real_right` episodes of 29 Oct (front flap, operator demos) + the 15 robot-playback episodes of 3 Nov (5 box poses x front,
back, right flap; back was recorded with the front flap already open, right with front + back open). Head + right gripper camera.
Client config: `rollout_config_right_3flaps.yaml` (copy of the right20 one: fps 15, max_steps 1200, wider training range).
**The instruction picks the flap** (default front):
| flap | `--instruction` | start pose (rad) | box state at start in training |
|---|---|---|---|
| front | `"open the front flap of the cardboard box"` | [-0.349, 0.407, 1.118, 0.293, -0.962, 0.153] | all closed |
| back  | `"open the back flap of the cardboard box"`  | same as front | front flap open |
| right | `"open the right flap of the cardboard box"` | [-0.349, 0.400, 1.131, 0.209, -0.935, 0.305] | front + back open |

Offline on its training episodes (34 good ones, 576 samples, `offline_check_3flaps.py`): MAE 0.0082 rad vs 0.0204 for holding
still (0.40x), direction cosine median 0.99, first step max 0.016 rad, 100 % pass the default gates, ~68 ms; all three flaps alike;
640x360 and 1152x648 inputs give the same result. A wrong instruction makes it clearly worse (back episodes: 0.0127 with "front",
0.0162 with "right"; at the start pose its first move then points the wrong way). Training episodes only (no held-out set): this
proves the pipeline and the language conditioning, not generalisation. The 3 Nov box poses were within ~2 cm of each other.

The operator runs this with a hand on the e-stop. Ctrl+C / Enter in the client cancel the goal and hold, but they are NOT an e-stop.
One stack per cell: check first that no other real stack is running.

## 0. Scene (compare with the training start frames)
- `~/groot/reference/start_frames_real_right_head.jpg` (head camera, all 19 episodes): box closed on the floor, low centre-left,
  right arm along the right edge. `..._wrist_right.jpg`: right gripper camera at the start.
- Single frames: `~/groot/reference/frames/real_right/epNN_{head,wrist_right}.jpg`.
- Same tool on the right arm as in the recordings; nothing else in the arm's path.

## 1. Start everything (every terminal)
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
```
- T1 arms: `ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=rebel` (left 192.168.3.11, right 192.168.3.12)
- T2 cameras: `ros2 launch rebel_cameras cameras.launch.py`
- T3 policy server (no ROS; loading from the USB drive takes ~11 min):
  `~/igus_rebel_ros2/tools/groot_inference/start_server.sh start --checkpoint ~/groot/checkpoints/real_right_3flaps/real_right_3flaps_35ep_15fps/checkpoint-3200`
  (if another model is running: `start_server.sh stop` first; `start_server.sh status` shows the loaded one)
- T4 client: `cd ~/igus_rebel_ros2/tools/groot_inference; PY=~/venvs/gr00t_client/bin/python; CFG="--config rollout_config_right_3flaps.yaml"`

On real hardware `dual_table.launch.py` loads both trajectory controllers INACTIVE (by design: first confirm which physical arm
answers which IP, e.g. jog it from its pendant and watch `arm_right_joint*` in /joint_states). Then activate only the right one
(it holds the current position on activation); without this every goal is "Goal was rejected":
```bash
ros2 control switch_controllers --activate arm_right_trajectory_controller
ros2 control list_controllers    # arm_right_trajectory_controller ... active
```

## 2. Move the RIGHT arm to the training start pose (MOTION: operator only)
Front/back: [-0.349, 0.407, 1.118, 0.293, -0.962, 0.153] rad (the 3 Nov set always; 29 Oct j1 -0.43 .. +0.05). Right: use the
right start pose from the table above in the command below.
Check the current `arm_right_joint*` values (`ros2 topic echo --once /joint_states`) and that the straight joint path is clear: this goal goes
straight to the trajectory controller, no collision check. Keep time >= largest joint change / 0.1 rad/s.
```bash
ros2 action send_goal /arm_right_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
 "{trajectory: {joint_names: [arm_right_joint1,arm_right_joint2,arm_right_joint3,arm_right_joint4,arm_right_joint5,arm_right_joint6], points: [{positions: [-0.349, 0.407, 1.118, 0.293, -0.962, 0.153], time_from_start: {sec: 30}}]}}"
```

## 3. Camera / box check (read-only)
```bash
P="--joint-prefix arm_right_ --start -0.349 0.407 1.118 0.293 -0.962 0.153"
python3 -s align_camera.py --camera head --ref ~/groot/reference/frames/real_right/ep01_head.jpg $P
python3 -s align_camera.py --topic /arm_right/gripper_camera/image_raw/compressed --ref ~/groot/reference/frames/real_right/ep01_wrist_right.jpg $P
```
Place the box until the gripper-camera view matches (the model leans heavily on it). OK = |rot| <= 1.5 deg, shifts <= 15 px, scale within 3 %.

## 4. Inputs (no motion)
`$PY groot_ros_rollout.py $CFG --check`
Expect: arm_right joints fresh; head 1920x1080 and wrist_right 1280x720 fresh; model input [640, 360]; hardware Igus::Rebel for arm_right.

## 5. Observe (no motion)
`$PY groot_ros_rollout.py $CFG --max-queries 10 --save-images`
Add `--instruction ...` for back / right. Expect every query `PASS`, ~70-100 ms latency, small `step0-state` (offline <= 0.016 rad), server modality `['head', 'wrist_right']`.
`state_in_training_range` refused = the arm is not at the start pose.

## 6. Execute, short first
```bash
$PY groot_ros_rollout.py $CFG --execute --max-steps 16 --save-images    # 2 chunks
$PY groot_ros_rollout.py $CFG --execute --max-steps 160 --save-images   # ~20 chunks
$PY groot_ros_rollout.py $CFG --execute --save-images                   # full: max 1200 steps / 1200 s
# back / right flap: add --instruction "open the back flap of the cardboard box" (or right) to each line
```
Typed `yes` required. 8 steps (0.53 s of model time) per query at <= 0.1 rad/s, stop-and-go (the demos were faster). Re-home (step 2) between runs.
Options as before: `--set gates.urdf_limit_mode=clamp`, `--max-steps N --max-time S`.
Exit codes: 0 done, 3 refused by a gate (arm holds), 4 stale input / server problem, 5 execution aborted, 130 operator stop.
Logs + images: `~/groot/runs/<time>/`.

## 7. Afterwards
`~/igus_rebel_ros2/tools/groot_inference/start_server.sh stop` (frees ~6 GB GPU).

Offline check (no robot): `nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I offline_check_3flaps.py` (results ~/groot/logs/offline_3flaps_*.json)
