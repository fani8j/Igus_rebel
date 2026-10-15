# rebel_medicine_grasp_description

Robot model for **MoveIt - medicine grasp** (shelf picking of medical cartons): the dual-arm igus ReBeL stand (ZLW lift, two sliders, head camera, `arm_left_` / `arm_right_`
arms) **mounted on a D200 AMR**, with

* **left arm: suction tool v5** (3 x SMC ZP2 pads, Gemini 305 wrist camera) - TCP `arm_left_suction_tcp`,
* **right arm: SMC LEHF20K2-48 parallel gripper** with printed fingers and a fixed top stop - TCP `arm_right_gripper_tcp`.

No knife, XEG-32, knife camera or other cutting tooling. The cutting-cell packages `rebel_dual_arm_stand_description` / `_moveit_config` are used read-only (arm macro, stand meshes)
and are not modified. Mock hardware and Isaac topics only; there is no real-hardware option. Companion package: `rebel_medicine_grasp_moveit_config`.

```
urdf/rebel_medicine_grasp.urdf.xacro      D200 + stand + 2 arms + 2 tools + ros2_control (args hardware_protocol, include_ros2_control, initial_positions_file)
urdf/left_suction_tool.urdf.xacro         COPY of cad/medical_box_tooling_v1/suction_tool_v5 (only the mesh URI changed)
urdf/right_gripper_tool.urdf.xacro        COPY of cad/medical_box_tooling_v1/gripper_tool (only the mesh URI/layout changed)
urdf/rebel_medicine_grasp.ros2_control.xacro  15 stand/arm joints exactly as in the stand package + the 2 gripper finger joints
meshes/d200/amr_d200.stl                  AMR visual (metres)
meshes/suction_tool_v5/{visual,collision}, meshes/gripper_tool/{visual,collision}   tool meshes, package:// URIs only
test/                                     check_stand_kinematics.py, check_tool_frames.py, check_amr_clearance.py (run by colcon test)
```

## Frames
`world` = **`d200_base_footprint`** (fixed joint, identity): the D200 root, floor under the AMR centre, +X forward, Z up. The AMR is static while picking; a planar virtual joint
`world -> d200_base_footprint` could be added later for mobile planning (it needs the SRDF `virtual_joint` and a mobile-base group, not done).
`d200_base_footprint -> d200_base_link` (z 0.0762, wheel axle height) `-> stand` (`root_joint`: xyz 0.1668696 0 0.1883 from base_link, Rz +90 deg = the USD `upper_fixed_joint`,
stand frame at D200 (0.1668696, 0, 0.2645)). Source: `/media/farhan/My Passport/igus_nvidia/scratch_usd/d200_with_dual_arm_20261015` (read only). The URDF `stand` frame equals the
USD `stand` prim frame (identity inside `dual_arm_upper.usda`; its `arm_left_mount` joint is (0.228209, -0.179297, 1.1093), the same numbers as the URDF), so no extra offset exists:
at home the arm bases are at (0.3462, +0.2222 / -0.2258, 1.1738) in `world` (arm base plane = 0.2645 + 0.9093).
The arms look along `world` +X; `arm_left` is at +Y, `arm_right` at -Y.

* **AMR**: visual = `amr_d200.stl` (stored with its long axis along Y; rotated Rz(-90 deg) to match the USD, max vertex error 6e-9 m), **collision = ONE box** 0.968 x 0.654 x 0.2515 m
  (floor to top, wheels included), not the STL. Wheels are not modelled (the STL contains them). Mass 72 kg is an ASSUMPTION of the D200 package.
* **Stand/arms**: link and joint names, origins, limits, inertias, meshes identical to `rebel_dual_arm.urdf.xacro` (the stand block is a verbatim copy of its lines 35-263, md5 of the
  source in the xacro header; only `root_joint` changed). Stand meshes are referenced from `rebel_dual_arm_stand_description`.
* **Left tool** (all rigid on `arm_left_link6`, tool frame = link6): `arm_left_suction_tool` (one link, mass 0.20381 kg, 55 convex collision meshes), `arm_left_suction_tcp`
  (z 159 mm, the uncompressed pad lip), `arm_left_suction_tcp_compressed` (z 154 mm), `arm_left_wrist_camera_mount`, `arm_left_wrist_camera_color_optical_frame`,
  `arm_left_pad_{1,2,3}_suction_frame` (x -60/0/+60 mm, z 159 mm). **The pad buffers are FIXED in this URDF (the CAD URDF is one rigid link, no buffer joints)**; in the Isaac
  tool they are three prismatic joints (`pad_{1,2,3}_buffer_joint`, 10 mm stroke, spring) and the compressed TCP stands for pads pressed 5 mm.
* **Right tool**: `arm_right_gripper_base` (fixed on link6, 0.8437 kg), `arm_right_finger_left` / `arm_right_finger_right` (0.0525 kg each), **real prismatic joints**
  `arm_right_finger_left_joint` (axis -X, 0..0.024 m, effort 28 N, velocity 0.05 m/s; the actuated one) and `arm_right_finger_right_joint` (axis +X, same limits,
  `<mimic joint=arm_right_finger_left_joint multiplier=1 offset=0>`), frames `arm_right_gripper_tcp` (z 193.2 mm, between the pad centres) and `arm_right_top_stop_frame` (z 167.5 mm).
  q = 0 is closed (pad gap 8 mm), q = 0.024 open (56 mm); gap = 8 + 2 q. 20 convex collision meshes.
* Collision geometry of the tools comes from the `collision/` meshes only (the test asserts it), TCP and camera frames have no geometry.

## ros2_control
`rebel_medicine_grasp_controller`, `mock_components/GenericSystem` (default) or `topic_based_ros2_control/TopicBasedSystem` (`isaac_topic`, topics `/dual_arm/isaac_joint_commands` /
`/dual_arm/isaac_joint_states`); anything else fails xacro. 15 stand/arm joints as in the stand package + `arm_right_finger_left_joint` (position command, position/velocity state) +
`arm_right_finger_right_joint` declared as the **mimic follower** (mock: `mimic` param; isaac_topic: plain joint, state from Isaac). Why the follower is declared: with only the driver in
`/joint_states`, MoveIt's start state had the follower closed, and path validation reported finger/carton contacts that the planner did not see.

### isaac_topic: finger joint names do NOT match the USD
`sim_assets/medical_box_tools/usd/dual_arm_rig_medical.usda` names the gripper joints `finger_left_joint` / `finger_right_joint` (under the right gripper prim, no `arm_right_`
prefix) and the suction buffers `pad_{1,2,3}_buffer_joint`; the 15 stand/arm joints match the URDF. The URDF names are `arm_right_finger_left_joint` / `arm_right_finger_right_joint`
(URDF joint names must be unique per robot, hence the prefix). **Not fixed here (USD is not edited).** Fix on the Isaac side: in the bridge (`dual_arm_rig_ros2_bridge.py` or the
topic remapper) publish/subscribe `finger_left_joint` as `arm_right_finger_left_joint` and `finger_right_joint` as `arm_right_finger_right_joint`, command only the left one (the
USD uses a PhysX mimic joint and ignores the right target), and leave the pad buffers out of the topics (they are not URDF joints). Until then, with `isaac_topic` the finger
state in `/joint_states` stays at its initial value and finger commands do not reach Isaac. Also note: that rig USD was authored around the stand origin as world; with the
D200 it must be placed at the D200 mount (`d200_with_dual_arm.usd`), and `/dual_arm/isaac_joint_states` poses are joint values only (frame independent).

## Tests
`colcon test --packages-select rebel_medicine_grasp_description` (needs the package and `rebel_dual_arm_stand_description` installed/sourced):
* `check_stand_kinematics.py` - expands this URDF and the original `rebel_dual_arm.urdf.xacro`: same core link/joint sets (33 links, 32 joints below `stand`), identical joint
  type/parent/child/origin/axis/limit/dynamics, identical inertial/visual/collision of every link not below `link6`, and FK of every such link **relative to the `stand` link** for 300
  random + 4 special joint vectors (max error 9e-16 m / 2e-16 rad); the stand mount equals the USD transform; the cutting tooling is absent; AMR collision is one box.
* `check_tool_frames.py` - TCP/pad/camera frames and masses against the CAD numbers, gripper joints/mimic, no pad-buffer joints, all 139 mesh URIs resolve, tool collision only from `collision/`.
* `check_amr_clearance.py` - with the lift at its lower limit the lowest TCP height is 0.52 m (left) / 0.48 m (right) vs the AMR top 0.2515 m.

## ASSUMED / not verified
* Tool masses/inertias are the CAD estimates of the tooling session (printed parts 1.6 mm shell + 25 % infill, catalog gripper mass 0.75 kg); AMR 72 kg and its inertia are assumptions.
* AMR collision box is an envelope (conservative); real clearance below the chassis is not modelled.
* The pad buffers are rigid, so pad contact forces/compliance are not represented (use `arm_left_suction_tcp_compressed` for the pressed state).
* The stand/lift/slider inertias and the head camera pose are inherited placeholders from the stand package (see its README).
* Nothing was run on a real robot; Isaac Sim was not started.
