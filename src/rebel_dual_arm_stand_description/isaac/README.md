# Isaac Sim assets (copied unchanged from compal_robot)

Files (all unchanged copies from `compal_robot` @ `bb76327`, MIT, see `../NOTICE`):

| file | what |
|---|---|
| `dual_arm_rig.usda` | v1 rig (`igus_stand_dual_arm/dual_arm_rig.usda`) |
| `dual_arm_rig_v2.usda` | v2 rig (frame rebuilt from the 2026-10-04 cable-holes drawing; `dual_arm_rig_v2/dual_arm_rig_v2.usda`), matches `urdf/rebel_dual_arm.urdf.xacro` |
| `dual_arm_rig_ros2_bridge.py` | Script Editor snippet: ROS 2 bridge action graph (joint states / commands, cameras) |
| `sim_scene_control.py`, `sim_flap_weld.py` | scene helper services (box/table placement, flap weld) of compal_robot's lid-opening setup |

## READ THIS BEFORE USING THEM

* **These USDs still contain compal_robot's XEG-32 gripper tooling and compal_robot's 450 x 320 x 250 carton (`CardboardBox`) and table, NOT our
  SMC LEHF20K2-48 / knife / camera tooling** (`urdf/rebel_cell_tool.xacro`, design v3_r_1269) and NOT our 370 x 320 x 180 carton layout. The URDF
  and the USDs therefore differ in everything attached to link6. The arms' own meshes/inertias inside the USD are compal_robot's `rebel_d00617809`
  set, not ours.
* The **joint names match the URDF** for the 15 joints we actuate (`lift_joint`, `slider_left_joint`, `slider_right_joint`,
  `arm_{left,right}_joint1..6`). The USDs additionally contain XEG-32 gripper / camera-tilt joints (`arm_*_xeg32_*`) that do NOT exist in
  our URDF; with `hardware_protocol:=isaac_topic` they are neither commanded nor read here. Check how the bridge script handles joints missing from
  `/dual_arm/isaac_joint_commands` before relying on it.
* The USDs place the sliders at 0 (arm spacing 459.946 mm); our spacing s = 0.448 m is set by the initial/home slider values
  (`-0.005973`, `+0.005973`) in `rebel_dual_arm_stand_moveit_config/config/initial_positions.yaml`. In Isaac the sliders must be driven to those values.
* **Never tested here: Isaac Sim is not installed on this machine.** compal_robot reports its own tests (Isaac 6.1 headless load and physics; the ROS 2
  bridge segfaults headless on their machine, so the end-to-end flow was only usable in the GUI). Nothing in this directory has been run by us.
* **Never tested on real hardware**, and this package has no real-hardware option.
* Topics (`rebel_dual_arm.ros2_control.xacro`, isaac_topic): commands `/dual_arm/isaac_joint_commands`, states `/dual_arm/isaac_joint_states`
  (`topic_based_ros2_control/TopicBasedSystem`).
* Size: the two USDs are 8.3 MB and 13.4 MB text files; mind that when committing.
