# GR00T rebel_open_front_20ep (RIGHT arm): test on the real robot

Checkpoint `~/groot/checkpoints/rebel_open_front_20ep/rebel_open_front_20ep_360p/checkpoint-3000`, trained on the 20 `real_right`
episodes of 29 Oct (right arm, head + right gripper camera, 30 Hz, 640x360, "open the front flap of the cardboard box").
Client config: `rollout_config_right20.yaml`. Offline on its training episodes (19 good ones, 464 samples): MAE 0.0054 rad vs 0.0113
for holding still, direction cosine median 0.98, first step median 0.006 rad, 100 % pass the default gates. These are training
episodes (no held-out set), so this only proves the pipeline, not generalisation.

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
  `~/igus_rebel_ros2/tools/groot_inference/start_server.sh start --checkpoint ~/groot/checkpoints/rebel_open_front_20ep/rebel_open_front_20ep_360p/checkpoint-3000`
  (if the left-arm model is running: `start_server.sh stop` first)
- T4 client: `cd ~/igus_rebel_ros2/tools/groot_inference; PY=~/venvs/gr00t_client/bin/python; CFG="--config rollout_config_right20.yaml"`

On real hardware `dual_table.launch.py` loads both trajectory controllers INACTIVE (by design: first confirm which physical arm
answers which IP, e.g. jog it from its pendant and watch `arm_right_joint*` in /joint_states). Then activate only the right one
(it holds the current position on activation); without this every goal is "Goal was rejected":
```bash
ros2 control switch_controllers --activate arm_right_trajectory_controller
ros2 control list_controllers    # arm_right_trajectory_controller ... active
```

## 2. Move the RIGHT arm to the training start pose (MOTION: operator only)
[-0.349, 0.407, 1.118, 0.293, -0.962, 0.153] rad (episodes 1-7; j1 was -0.43 .. +0.05 in others, depending on the box place).
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
Expect every query `PASS`, ~70-100 ms latency, small `step0-state` (offline ~0.006 rad), server modality `['head', 'wrist_right']`.
`state_in_training_range` refused = the arm is not at the start pose.

## 6. Execute, short first
```bash
$PY groot_ros_rollout.py $CFG --execute --max-steps 16 --save-images    # 2 chunks
$PY groot_ros_rollout.py $CFG --execute --max-steps 160 --save-images   # ~20 chunks
$PY groot_ros_rollout.py $CFG --execute --save-images                   # full: max 2400 steps / 1200 s
```
Typed `yes` required. 8 steps per query at <= 0.1 rad/s, stop-and-go (the demos were faster). Re-home (step 2) between runs.
Options as before: `--set gates.urdf_limit_mode=clamp`, `--max-steps N --max-time S`.
Exit codes: 0 done, 3 refused by a gate (arm holds), 4 stale input / server problem, 5 execution aborted, 130 operator stop.
Logs + images: `~/groot/runs/<time>/`.

## 7. Afterwards
`~/igus_rebel_ros2/tools/groot_inference/start_server.sh stop` (frees ~6 GB GPU).

Offline check (no robot): `nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I offline_check_right20.py --dataset "/media/farhan/My Passport/igus_nvidia/datasets/real_right_20261029_raw/real_right"`
