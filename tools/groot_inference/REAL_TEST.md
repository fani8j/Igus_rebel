# GR00T flap policy: first test on the real LEFT arm

The operator runs this with a hand on the e-stop. Ctrl+C / Enter in the client cancel the goal and hold, but they are NOT an e-stop.
One stack per cell: check first that no other real stack is running (`systemctl --user list-units 'dual-real*'`, `ps`).

## 0. Scene
- Same tool on the left arm as in the mocap recordings; nothing else in the arm's path.
- Box closed and placed like the training start frames: `~/groot/reference/start_frames_head.jpg` (head camera view;
  arm on the left, box low centre) and `~/groot/reference/start_frames_wrist_left.jpg` (left gripper camera).
- Head camera not moved since the mocap recordings (the model depends on both views).

Camera check (read-only, arm at the start pose of step 2): `python3 -s ~/igus_rebel_ros2/tools/groot_inference/align_camera.py`
(gripper camera; `--camera head` for the head camera) shows the live view blended with the training start frame and the measured
rotation / shift / scale; re-aim the mount until it says OK (|rot| <= 1.5 deg, shifts <= 15 px of 640x480, scale within 3 %).

## 1. Start everything (every terminal)
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
```
- T1 arm: `ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel robot_ip:=192.168.3.11`
- T2 cameras: `ros2 launch rebel_cameras cameras.launch.py` (head + both gripper cameras)
- T3 policy server (no ROS, 1-3 min load): `~/igus_rebel_ros2/tools/groot_inference/start_server.sh start`
- T4 client: `cd ~/igus_rebel_ros2/tools/groot_inference` (run with `~/venvs/gr00t_client/bin/python`)

## 2. Move to the training start pose [0, 0, 1.0472, 0, 0.8727, 0]
Check the current joints first (`ros2 topic echo --once /joint_states`) and that the straight joint path is clear: this goal
goes straight to the trajectory controller, no MoveIt collision check. Keep time >= largest joint change / 0.1 rad/s
(30 s covers 3 rad).
```bash
ros2 action send_goal /rebel_arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
 "{trajectory: {joint_names: [joint1,joint2,joint3,joint4,joint5,joint6], points: [{positions: [0.0, 0.0, 1.0472, 0.0, 0.8727, 0.0], time_from_start: {sec: 30}}]}}"
```

## 3. Inputs (no motion)
`~/venvs/gr00t_client/bin/python ~/igus_rebel_ros2/tools/groot_inference/groot_ros_rollout.py --check`
Expect: joints fresh; head 1920x1080 and wrist_left 1280x720, both fresh; hardware `Igus::Rebel` with robot_ip 192.168.3.11.

## 4. Observe (no motion)
`~/venvs/gr00t_client/bin/python ~/igus_rebel_ros2/tools/groot_inference/groot_ros_rollout.py --max-queries 10 --save-images`
Expect every query `PASS`, ~70 ms latency, small `step0-state`. Compare the saved model inputs in the run dir
(`~/groot/runs/<time>/images/`) with the reference frames: if the views differ (camera moved, other box place), stop here.
`state_in_training_range` refused = the arm is not at the start pose.

## 5. Execute, short first
```bash
~/venvs/gr00t_client/bin/python ~/igus_rebel_ros2/tools/groot_inference/groot_ros_rollout.py --execute --max-steps 16 --save-images   # 2 chunks, a few seconds
~/venvs/gr00t_client/bin/python ~/igus_rebel_ros2/tools/groot_inference/groot_ros_rollout.py --execute --max-steps 96 --save-images   # then longer
~/venvs/gr00t_client/bin/python ~/igus_rebel_ros2/tools/groot_inference/groot_ros_rollout.py --execute --save-images                  # full: max 1200 steps / 600 s
```
It prints the hardware, asks for a typed `yes`, then runs 8 steps per query at <= 0.1 rad/s (stop-and-go between queries;
the demo was faster). Re-home (step 2) between runs; the model starts from the start pose in every training episode.
Exit codes: 0 done, 3 chunk refused by a gate (reason printed; the arm holds), 4 stale camera/joints or server problem,
130 operator stop. Logs: `queries.jsonl` in the run dir.

Option `--set gates.urdf_limit_mode=clamp`: targets beyond the URDF limits (minus `urdf_inset_rad` 0.05) are clamped onto
them instead of refusing the chunk (printed as `clamped onto URDF limits`); a target more than `urdf_clamp_max_rad` (0.15) beyond
is still refused. The joint then stays pinned at the limit while the model keeps pushing.

## 6. Afterwards
`~/igus_rebel_ros2/tools/groot_inference/start_server.sh stop` (frees ~6 GB GPU).
