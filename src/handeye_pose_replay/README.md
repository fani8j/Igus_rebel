# handeye_pose_replay v3

This version avoids loading / re-processing the ReBeL Xacro entirely.

Instead, it copies the exact `robot_description` and
`robot_description_semantic` parameters from the already-running MoveIt
`/move_group` node and then starts the replay.

This avoids errors caused by `initial_positions.yaml` entries for the XEG32
gripper.

Configuration:

    planning group:    igus_rebel_arm
    model source node: /move_group
    replay poses:      19 actual joint states from pose_h2e/v2.zip
    default speed:     5%
    manual confirmation before every pose: enabled

## Replace older package

    cd ~/igus_rebel_ros2/src
    rm -rf handeye_pose_replay
    unzip ~/Downloads/handeye_pose_replay_v3.zip

## Build

    cd ~/igus_rebel_ros2
    colcon build --packages-select handeye_pose_replay --symlink-install
    source install/setup.bash

## IMPORTANT: Start normal MoveIt FIRST

Terminal 1:

    cd ~/igus_rebel_ros2
    source install/setup.bash

    ros2 launch igus_rebel_moveit_config \
      igus_rebel_motion_planner.launch.py \
      use_gui:=true

Leave it running.

Verify:

    ros2 node list | grep move_group

Expected:

    /move_group

You can also check:

    ros2 param get /move_group robot_description
    ros2 param get /move_group robot_description_semantic

Both should return strings/XML.

## Run replay

Terminal 2:

    cd ~/igus_rebel_ros2
    source install/setup.bash
    export ROS_DOMAIN_ID=11

    ros2 launch handeye_pose_replay replay_handeye.launch.py

The launch opens a dedicated GNOME Terminal for the replay. Press ENTER in that
window, not the terminal where `ros2 launch` was invoked. Use
`interactive_terminal:=false` only when input is supplied by another mechanism.

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
