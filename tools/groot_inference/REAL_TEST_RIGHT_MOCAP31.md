# TurboVLA mocap31 (model A): first OBSERVE test on the real RIGHT arm (no motion from the policy)

Goal: see what the model outputs with the REAL cameras at the training start pose, and how many queries pass the gates.
Observe mode never creates the trajectory action client; only step 2 (homing, run by the operator) moves the arm.
The operator keeps a hand on the e-stop for step 2. One stack per cell: check first that no other real stack is running.

## 0. Scene (compare with the reference frames)
- `~/groot/reference/frames/mocap31_head.jpg`: head view of the training start (identical in all 23 episodes); box closed,
  stacked on a second box, low centre; right arm on the right edge of the image.
- `~/groot/reference/frames/mocap31_wrist_right.jpg`: right gripper camera at the start pose (episodes vary by <= 1.6 deg / 12 px).
- All start frames: `~/groot/reference/frames/mocap31/epNN_{head,wrist_right}.jpg`.

## 1. Start everything (every terminal)
```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
```
- T1 arms: `ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=rebel` (left 192.168.3.11, right 192.168.3.12)
- T2 cameras: `ros2 launch rebel_cameras cameras.launch.py` (head `/camera` + both gripper cameras)
- T3 policy server (no ROS): `~/igus_rebel_ros2/tools/turbovla_inference/start_turbovla_server.sh start --profile mocap31`
  then `... info`: must say arm right, sha 82eab1f7..., language "open the front flap of the cardboard box".
- T4 client: `cd ~/igus_rebel_ros2/tools/groot_inference; PY=~/venvs/gr00t_client/bin/python; CFG="--config rollout_config_turbovla_mocap31.yaml"`

## 2. Move the RIGHT arm to the training start pose [0, 0, 60, 0, 50, 0] deg (MOTION: operator only)
Check the current joints first (`ros2 topic echo --once /joint_states`, the `arm_right_joint*` entries) and that the straight
joint path is clear: this goal goes straight to the trajectory controller, no collision check. Keep time >= largest change / 0.1 rad/s.
```bash
ros2 action send_goal /arm_right_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
 "{trajectory: {joint_names: [arm_right_joint1,arm_right_joint2,arm_right_joint3,arm_right_joint4,arm_right_joint5,arm_right_joint6], points: [{positions: [0.0, 0.0, 1.0472, 0.0, 0.8727, 0.0], time_from_start: {sec: 30}}]}}"
```

## 3. Camera check (read-only)
```bash
python3 -s align_camera.py --camera head --ref ~/groot/reference/frames/mocap31_head.jpg
python3 -s align_camera.py --topic /arm_right/gripper_camera/image_raw/compressed --ref ~/groot/reference/frames/mocap31_wrist_right.jpg
```
OK = |rot| <= 1.5 deg, shifts <= 15 px of 640x480, scale within 3 %. If the head view is clearly a different camera/position,
stop: the model has only ever seen this head view (the right19 data used another one).

## 4. Inputs (no server, no motion)
`$PY groot_ros_rollout.py $CFG --check`
Expect joints fresh; head and wrist_left (= right gripper camera) fresh; 640x480 recorder size -> 224; hardware Igus::Rebel.

## 5. Observe (no motion)
`$PY groot_ros_rollout.py $CFG --max-queries 20 --save-images`
Logs `queries.jsonl` + model input images in `~/groot/runs/<time>/`. Look at the per-query verdict and `step0-state`:
offline (held-out recordings) the first step at the start was ~4 deg median; > 0.15 rad (8.6 deg) refuses `first_step`.
`state_in_training_range` refused = the arm is not at the start pose. Send the run dir for analysis before any execute.

## 6. Afterwards
`~/igus_rebel_ros2/tools/turbovla_inference/start_turbovla_server.sh stop`
