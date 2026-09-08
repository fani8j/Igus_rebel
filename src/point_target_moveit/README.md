# point_target_moveit

ROS 2 Humble / MoveIt 2 example package for moving an Igus Rebel TCP toward a
3D point detected from a camera or point cloud.

The node subscribes to:

```text
/cut_target
geometry_msgs/msg/PointStamped
```

It then:

1. Reads the target XYZ and source frame.
2. Uses TF2 to transform the point into the planning frame.
3. Adds an approach offset in Z.
4. Creates a TCP pose.
5. Sends the pose to MoveIt for planning.
6. Optionally executes the trajectory.

## IMPORTANT

Execution is **disabled by default**.

First verify the generated plan in RViz. Only enable execution after confirming:

- Camera hand-eye TF is correct.
- TCP calibration is correct.
- The MoveIt planning group is correct.
- The configured TCP link exists in the robot model.
- The target location appears correctly in RViz.
- The robot has sufficient clearance.

---

## Copy into your workspace

Example:

```bash
cd ~/igus_rebel_ros2/src
unzip point_target_moveit.zip
```

If the ZIP was extracted elsewhere:

```bash
cp -r point_target_moveit ~/igus_rebel_ros2/src/
```

## Build

```bash
cd ~/igus_rebel_ros2
colcon build --packages-select point_target_moveit
source install/setup.bash
```

## Check your MoveIt group name

```bash
grep -R "<group name=" \
~/igus_rebel_ros2/src/igus_rebel_moveit_config/config
```

Example result:

```xml
<group name="rebel_arm">
```

Use that value as `planning_group`.

## Check the TCP frame

```bash
ros2 run tf2_ros tf2_echo base_link xeg32_tool_tip
```

## Launch MoveIt

Use your normal MoveIt launch file, for example:

```bash
ros2 launch igus_rebel_moveit_config \
  igus_rebel_motion_planner.launch.py \
  use_gui:=true
```

## Run this node

Default parameters:

```bash
ros2 run point_target_moveit point_target_moveit_node
```

Recommended explicit form:

```bash
ros2 run point_target_moveit point_target_moveit_node --ros-args \
  -p planning_group:=rebel_arm \
  -p planning_frame:=base_link \
  -p tcp_link:=xeg32_tool_tip \
  -p approach_offset_z:=0.05 \
  -p velocity_scaling:=0.10 \
  -p acceleration_scaling:=0.10 \
  -p execute:=false
```

Change `planning_group` if your SRDF uses a different group name.

---

# Test 1: target already in base_link

```bash
ros2 topic pub --once \
/cut_target \
geometry_msgs/msg/PointStamped \
"{
  header: {
    frame_id: 'base_link'
  },
  point: {
    x: 0.30,
    y: 0.00,
    z: 0.15
  }
}"
```

With an approach offset of 0.05 m, MoveIt plans the TCP to approximately:

```text
x = 0.30
y = 0.00
z = 0.20
```

The extra 5 cm is intentional.

---

# Test 2: point in the camera frame

Find the real point-cloud frame:

```bash
ros2 topic echo \
/camera/depth_registered/points \
--once \
--field header
```

Example:

```text
frame_id: camera_color_optical_frame
```

Then publish:

```bash
ros2 topic pub --once \
/cut_target \
geometry_msgs/msg/PointStamped \
"{
  header: {
    frame_id: 'camera_color_optical_frame'
  },
  point: {
    x: 0.05,
    y: 0.02,
    z: 0.60
  }
}"
```

The node automatically transforms this point into `base_link` using TF2.

---

# Parameters

| Parameter | Default | Description |
|---|---|---|
| `planning_group` | `rebel_arm` | MoveIt SRDF planning group |
| `planning_frame` | `base_link` | Frame target XYZ is transformed into |
| `tcp_link` | `xeg32_tool_tip` | TCP/end-effector link MoveIt should place at the target |
| `approach_offset_z` | `0.05` | Z offset in meters above the detected point |
| `velocity_scaling` | `0.10` | 10% max velocity |
| `acceleration_scaling` | `0.10` | 10% max acceleration |
| `execute` | `false` | Execute after planning |

---

# Enable execution

Only after verifying everything in RViz:

```bash
ros2 run point_target_moveit point_target_moveit_node --ros-args \
  -p planning_group:=rebel_arm \
  -p planning_frame:=base_link \
  -p tcp_link:=xeg32_tool_tip \
  -p execute:=true
```

For initial real-robot tests, keep velocity and acceleration low.

---

# Tool orientation

The source currently uses this temporary quaternion:

```cpp
target_pose.pose.orientation.x = 0.0;
target_pose.pose.orientation.y = 1.0;
target_pose.pose.orientation.z = 0.0;
target_pose.pose.orientation.w = 0.0;
```

This must eventually be replaced by the correct knife orientation for your robot/tool.

The position pipeline can be tested before finalizing the orientation.

---

# Next integration step

Your tape detector should publish:

```text
geometry_msgs/msg/PointStamped
```

on:

```text
/cut_target
```

For example:

```python
target.header.frame_id = camera_frame
target.point.x = X
target.point.y = Y
target.point.z = Z
publisher.publish(target)
```

Then this package handles the camera-to-robot TF transform and MoveIt planning.
