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

The active TCP is defined in `igus_rebel_xeg32_wrapper.urdf.xacro`, independently
of optional gripper geometry:

- `link6 -> igus_tool0`: xyz `[0, 0, 0.0527]` m, RPY `[0, 0, pi]`.
- `igus_tool0 -> xeg32_tool_tip`: xyz `[0.068936, 0.046766, 0.127060]` m,
  RPY `[0, 0, 0]`. These values are `OffsetX`, `OffsetY`, `OffsetZ`,
  `OffsetA`, `OffsetB`, and `OffsetC` from `knivetool4_with_pose2` in
  `knivetool4.xml`, converted from millimetres to metres.
- Both frames remain available with `attach_xeg32:=false`.
- Arm kinematics, the `link6` IK solver tip, and the rough XEG mount
  (`0 0 0.052` m, 15-degree Z rotation) are unchanged. MTC uses the fixed
  `xeg32_tool_tip` descendant explicitly as its IK frame.

The rough `xeg32_cutter_link`, its mounting joint, box and pyramid visual/collision
geometry, and old cutter-derived TCP joint are removed. The cutter Xacro arguments
are removed as well. The unused pyramid STL remains as an archived asset.
Four cutter-specific SRDF collision exclusions and the MTC cutter/carton collision
exemption are removed; gripper semantics and collision geometry remain unchanged.

The green TCP sphere now also has a 6 mm-radius collision shape. It covers only
the tip; the physical cutter body remains absent from collision checking.
The gripper CAD is approximate. Plan success does not establish physical clearance
or calibrate tool orientation. Keep physical execution disabled until independently
validated, and discard plans and tool-relative targets made with the old TCP.

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
