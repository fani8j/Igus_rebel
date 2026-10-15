# Third-party material

## compal_robot (MIT License, Copyright (c) 2026 James), source commit `bb76327`

License text: see `NOTICE`. Source checkout: `/home/farhan/compal_robot` (read-only for us).
Paths below are relative to `compal_robot/src/`.

### rebel_dual_arm_stand_description

| file here | derived from | how |
|---|---|---|
| `urdf/rebel_arm_macro.xacro` | `igus_rebel_description/urdf/igus_rebel2_arm_macro.xacro` | renamed macro, joint limits / velocity changed to OUR single-arm URDF values (verified by `test/check_arm_fk.py`) |
| `urdf/rebel_dual_arm.urdf.xacro` | `igus_rebel_description/urdf/dual_arm_rig_v2.urdf.xacro` | stand / lift / sliders / head-camera blocks verbatim (mesh paths changed); arms re-instantiated with our macro; XEG-32 gripper + virtual knife TCP + wrist-camera frames removed; our tool added |
| `urdf/rebel_dual_arm.ros2_control.xacro` | `igus_rebel_description/urdf/dual_arm_rig.control.xacro` | gripper joints and mimic handling removed, protocol check added |
| `meshes/rig_v2/*.stl` (8 files) | `igus_rebel_description/meshes/dual_arm_rig_v2/{stand,lift,slider_left,slider_right}_{visual,collision}.stl` | unchanged copy (tessellated from CR's STEP drawing `Dual_arms_step.10.1_UpdatedNew_20261004_cable_holes_ch_v4_p63ext100.STEP`) |
| `isaac/dual_arm_rig.usda` | `igus_rebel_description/urdf/igus_stand_dual_arm/dual_arm_rig.usda` | unchanged copy |
| `isaac/dual_arm_rig_v2.usda` | `igus_rebel_description/urdf/dual_arm_rig_v2/dual_arm_rig_v2.usda` | unchanged copy |
| `isaac/dual_arm_rig_ros2_bridge.py`, `isaac/sim_scene_control.py`, `isaac/sim_flap_weld.py` | `igus_rebel_description/urdf/igus_stand_dual_arm/` (same names) | unchanged copies |

### rebel_dual_arm_stand_moveit_config

| file here | derived from | how |
|---|---|---|
| `config/rebel_dual_arm.srdf` | `igus_rebel_moveit_config/config/dual_arm_rig.srdf` | group names / adjacent-link structure kept; gripper groups removed; chains end at `*_knife_tip`; our arm self-collision pairs (from our `igus_rebel2.srdf`) and tool pairs added |
| `config/joint_limits.yaml` | `.../config/dual_arm_joint_limits.yaml` | arm joint limits replaced by ours (0.785 rad/s, 5.24 rad/s^2); lift / slider values kept |
| `config/kinematics.yaml` | `.../config/dual_arm_kinematics.yaml` | IK timeout changed to our 0.25 s, attempts/resolution removed (ignored on Humble) |
| `config/ompl_planning.yaml` | `.../config/dual_arm_ompl_planning.yaml` | converted from the Jazzy list format to the Humble format |
| `config/moveit_controllers.yaml`, `config/ros2_controllers.yaml` | `.../config/dual_arm_moveit_controllers.yaml`, `dual_arm_ros_controllers.yaml` | gripper controllers removed |
| `config/initial_positions.yaml` | `.../config/dual_arm_initial_positions.yaml` | gripper joints removed, sliders set to s = 0.448 m |
| `config/rebel_dual_arm.rviz` | `.../config/dual_arm_rig.rviz` | frame / group names adapted |
| `scripts/add_cell_scene.py` | `igus_rebel_moveit_config/scripts/add_scene_boxes.py` | extended (layout -> boxes, W-frame transform); service-call pattern kept |
| `launch/dual_arm.launch.py` | `igus_rebel_moveit_config/launch/dual_arm_rig_isaac.launch.py` | rewritten for Humble, plan-only default, sequenced spawners; structure and argument names kept |

Files not listed here were written new for this workspace.

## SMC LEHF20K2-48 gripper mesh

`meshes/tool/gripper_smc_link6.stl` is derived from the SMC CAD download (CADENAS PARTcommunity / 3DfindIT, 2026-08-07, LEHF20K2-48(0_0).stl;
terms of use: https://www.cadenas.de/terms-of-use-3d-cad-models) by the transform of `tools/cell_planner_check/gen_cell.py::build_gripper_stl`.
Check those terms before redistributing the package outside the project.
