# handeye_pose_replay

This package avoids loading / re-processing the ReBeL Xacro entirely.

Instead, it copies the exact `robot_description` and
`robot_description_semantic` parameters from the already-running MoveIt
`/move_group` node and then starts the replay.

This avoids mismatches with the XEG32 gripper model used by MoveIt.

Configuration:

    planning group:    igus_rebel_arm
    model source node: /move_group
    replay poses:      19 actual joint states from pose_h2e/v2.zip
    default speed:     5% (velocity_scale / acceleration_scale 0.05)
    manual confirmation before every pose: enabled

## Build

The package is part of this repository (`src/handeye_pose_replay`); no
separate download or unzip step is needed. Build it with colcon from the
workspace root:

    cd ~/igus_rebel_ros2
    source /opt/ros/humble/setup.bash
    colcon build --packages-select handeye_pose_replay --symlink-install
    source install/setup.bash

## IMPORTANT: Start the robot driver and MoveIt FIRST

Every terminal below must use the **same** `ROS_DOMAIN_ID` (and the same
`ROS_LOCALHOST_ONLY` setting); otherwise the replay cannot see `/move_group`.
Use the domain of the robot you intend to move, and a separate unused domain
with `ROS_LOCALHOST_ONLY=1` for mock tests.

Terminal 1 (ros2_control driver and controllers; required for execution,
because MoveIt alone can plan but not move the arm):

    cd ~/igus_rebel_ros2
    source install/setup.bash
    export ROS_DOMAIN_ID=11

    ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel
    # or, for a dry run: hardware_protocol:=mock_hardware

Terminal 2 (MoveIt):

    cd ~/igus_rebel_ros2
    source install/setup.bash
    export ROS_DOMAIN_ID=11

    ros2 launch igus_rebel_moveit_config \
      igus_rebel_motion_planner.launch.py \
      hardware_protocol:=rebel use_gui:=true

Leave both running.

Verify:

    ros2 node list | grep move_group
    ros2 control list_hardware_components

Expected: `/move_group`, and the hardware component you intended (the real
`Igus::Rebel` plugin, or `mock_components/GenericSystem` for a dry run).

You can also check:

    ros2 param get /move_group robot_description
    ros2 param get /move_group robot_description_semantic

Both should return strings/XML.

## Run replay

Terminal 3:

    cd ~/igus_rebel_ros2
    source install/setup.bash
    export ROS_DOMAIN_ID=11

    ros2 launch handeye_pose_replay replay_handeye.launch.py

The launch opens a dedicated GNOME Terminal for the replay. Press ENTER in that
window, not the terminal where `ros2 launch` was invoked. The replay stops if
its input is closed, and it refuses to start with `wait_for_enter:=true` when
stdin is not a terminal (for example `interactive_terminal:=false` without a
TTY), because every confirmation would otherwise pass unattended.

The replay node will:

1. connect to `/move_group`,
2. copy the working URDF and SRDF,
3. construct `MoveGroupInterface`,
4. wait for ENTER before each hand-eye pose,
5. plan and execute at 5% speed.

## If your MoveIt node has another name

Find it:

    ros2 node list

Then pass its name:

    ros2 launch handeye_pose_replay replay_handeye.launch.py \
      model_source_node:=/YOUR_MOVE_GROUP_NODE

## Automatic replay, only after manual verification

    ros2 launch handeye_pose_replay replay_handeye.launch.py \
      wait_for_enter:=false \
      velocity_scale:=0.10 \
      acceleration_scale:=0.10

For the first physical-robot test, keep `wait_for_enter:=true` and 5% speed.

The scales are fractions of the MoveIt joint limits, which are 0.785 rad/s and
5.24 rad/s² since 2026-09-27 (previously 2.16 rad/s and 10 rad/s²). 5% is therefore
about 0.039 rad/s, slower than before; `velocity_scale:=0.10` is about 0.079 rad/s.

## Replay a calibration session

The calibration recorder (`rebel_dual_arm_calibration`) stores one JSON object
per line in `<session_dir>/samples.jsonl`, each with `"arm"`, `"capture"` and
`"joints"`. The replay can drive the arm through those joint poses instead of
the built-in 19 poses:

    ros2 launch handeye_pose_replay replay_handeye.launch.py \
      poses_file:=~/dual_arm_calibration/right arm:=right

Parameters (launch arguments of the same name):

- `poses_file`: a `samples.jsonl` or a session directory (`~` is expanded).
  Empty (default) keeps the built-in poses.
- `arm`: only use samples of this arm (`left`/`right`). If the file contains
  more than one arm and `arm` is empty, the replay refuses to start.
- `start_index`: 0-based index into the filtered pose list, to resume. The
  list excludes other arms and near-duplicates, and the printed `pose i/N`
  uses the same numbering (`i = start_index + 1` for the first one replayed).
- `max_poses`: replay at most this many poses (0 = all).

Joints are mapped to the planning group by the trailing `joint<N>` of their
name, so `joint3` (single-arm session) and `arm_right_joint3` (dual-arm
session) both drive the group's `joint3`. A pose whose joints are all within
0.01 rad of the previous kept pose is skipped as a duplicate.

The whole list is validated before anything moves: every pose must have all
six joints, finite values, and lie inside the robot model's joint limits. On
any problem every offending line is printed and the node exits without moving.
Before each pose the node prints `pose i/N (capture #c)`, the target in
degrees, and the largest joint change from the current state. Planning
failure for a pose stops the replay (it never skips ahead) and prints the
`start_index` to resume from. A summary `reached / total` is printed at the end.

Notes:

- With the calibration UI's Auto capture switched ON, the UI captures at each
  replayed pose by itself. It needs the arm still for about 2 s plus the
  capture. With `wait_for_enter:=true` the arm waits at the pose anyway; with
  `wait_for_enter:=false` use `settle_seconds:=5`.
- Replaying only reproduces the same images if the board mount and the camera
  have not changed since the session was recorded.
- The first real run keeps `wait_for_enter:=true` and 5 % speed (the defaults).

Unit tests for the parser: `colcon test --packages-select handeye_pose_replay`.
