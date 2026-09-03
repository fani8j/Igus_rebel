# XEG-32 MoveIt integration changes

The existing ReBeL arm configuration is preserved. This update adds the XEG-32
semantics needed for MoveIt to list the gripper as a planning group.

## Added SRDF group

`xeg32_gripper`

The group contains:
- `xeg32_base_link`
- `xeg32_left_carriage_link`
- `xeg32_right_carriage_link`
- active/master joint: `xeg32_left_carriage_joint`

`xeg32_right_carriage_joint` remains the URDF mimic joint and is marked passive
in the SRDF. `xeg32_camera_tilt_joint` is also marked passive because it is not
currently an actuated MoveIt joint.

## Added end effector

- name: `xeg32`
- parent group: `igus_rebel_arm`
- parent link: `link6`
- end-effector group: `xeg32_gripper`

## Added adjacent self-collision exclusions

Only these directly connected pairs were disabled:
- `link6` / `xeg32_base_link`
- `xeg32_base_link` / `xeg32_left_carriage_link`
- `xeg32_base_link` / `xeg32_right_carriage_link`
- `xeg32_base_link` / `xeg32_camera_mount_link`

If the gripper is still shown red in MoveIt, regenerate the self-collision
matrix with MoveIt Setup Assistant rather than disabling all XEG collisions.

## Tool center point

`xeg32_tool_tip` is the TCP for Servo and all box/MTC planners.

- `xeg32_tool_tip` is the positive-Z apex of `xeg32_cutter_link`.
- The dummy knife is 185.5 mm overall: a 170.5 mm, 25 × 25 mm rectangular
  blade plus a 15 mm square-pyramid tip.
- Its midpoint is `xeg32_base_link = (-77.0, 0, 29.0) mm`, 18 mm +X from
  the prior distal location, so the knife lies inside rather than through the
  gripper jaws.
- Its Z axis remains parallel to `link6` Z; the XEG mount has only a
  15-degree Z rotation.

The model is planning geometry only. Keep `execute: false` until the physical
cutter and its measured TCP have been installed and validated.

The cutter is permitted to overlap only the wrist (`link6`) and XEG base and
carriage collision meshes. It remains collision-checked against the carton,
table, and other arm links.

## Controller files intentionally unchanged

`moveit_controllers.yaml`, `ros_controllers.yaml`, and
`igus_rebel2.control.xacro` still control only ReBeL joints 1-6.

This is intentional. The HIWIN XEG-32 is separate hardware and should receive
its own controller/hardware interface later. Execution of an XEG gripper plan
will not work until that controller is added.

## Joint limits

The Onshape-exported XEG velocity/effort values were not copied into
`joint_limits.yaml` because they are exporter placeholders. Configure the
gripper limit after mapping the HIWIN XEG-32 command convention to the single
master URDF carriage joint.

## Rebuild

From the workspace root:

```bash
colcon build --symlink-install
source install/setup.bash
```

Then relaunch MoveIt. The planning-group dropdown should contain both:

- `igus_rebel_arm`
- `xeg32_gripper`
