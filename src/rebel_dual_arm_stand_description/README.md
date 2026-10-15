# rebel_dual_arm_stand_description

> **Not the real-cell model.** This is the full stand (ZLW lift + sliders, `world` = floor at the lift column) for planning studies and Isaac Sim.
> The real two-arm setup on the test table, with `world` = `arm_left_base_link` and the calibrated arm_right/camera poses, lives in
> `rebel_dual_arm_description` + `rebel_dual_arm_calibration`. Do not mix frames between the two.

Dual-arm igus ReBeL rig (stand, ZLW lift, two sliders, Gemini 336L head camera, `arm_left_` / `arm_right_` ReBeL arms) carrying **our** cell
tooling (SMC LEHF20K2-48 gripper + fingers, knife clamp / post / handpiece, Gemini 305 wrist camera, USB knife camera, TCP `*_knife_tip`).
URDF/xacro, meshes, Isaac Sim USD + bridge scripts. **Mock hardware and Isaac topics only; there is no real-hardware (CRI) option.** Plan-only use.
The rig frame, the arm mounts, the ros2_control layout and the Isaac files are derived from `compal_robot` (MIT, source commit `bb76327`): `NOTICE`,
`THIRD_PARTY.md`. Companion package: `rebel_dual_arm_stand_moveit_config`.

```
urdf/rebel_arm_macro.xacro            macro rebel_arm_chain(prefix, parent_link, xyz, rpy): OUR arm (limits, inertias, collision primitives, meshes)
urdf/rebel_cell_tool.xacro            macro rebel_cell_tool(prefix, parent): tooling for one arm, generated from config/cell_tool_*.yaml
urdf/rebel_dual_arm.urdf.xacro        the rig (compal_robot v2 frame) + 2 arms + 2 tools + ros2_control
urdf/rebel_dual_arm.ros2_control.xacro  15 joints; hardware_protocol mock_hardware (default) | isaac_topic
config/cell_tool_v3_r_1269.yaml       ALL tooling numbers (from tools/cell_planner_check/designs/v3_r_1269_A0_B0.json)
scripts/make_tool_config.py           design JSON -> that YAML (+ finger hull meshes)
meshes/rig_v2/                        stand / lift / slider_left / slider_right visual + collision STL (compal_robot, unchanged)
meshes/tool/                          gripper_smc_link6.stl (SMC CAD, already in the link6 frame), finger_{L,R}_grasp_hull.stl (visual only)
isaac/                                compal_robot USDs + Isaac scripts, unchanged; READ isaac/README.md
test/check_arm_fk.py, check_tool_frames.py   pytest / scripts (run by colcon test)
```
Arm meshes are not copied: the links reference `package://igus_rebel_description/meshes/rebel_d00617809/...` (ours), so this package depends on `igus_rebel_description`.

## Frames and joints
* `world` is compal_robot's frame (z = 0 on the floor level of the stand, +X to the left arm). The arm mounts are rotated -90 deg about +Z, so the arm base
  x axis ("forward", towards the work area) is world **-Y** and the base y axis is world +X. `arm_left` = `arm1` and `arm_right` = `arm2` of
  `tools/cell_planner_check`.
* Joint names are compal_robot's (Isaac-compatible): `lift_joint` (prismatic Z, -0.6235..0.6265), `slider_left_joint` (-0.26..0, +X axis, inward = negative),
  `slider_right_joint` (0..0.26, inward = positive), `arm_{left,right}_joint1..6`. The arm base plane is at z = 1.1093 + lift.
* Tool frames per arm (all rigid on `*_link6`): `tool_flange`, `gripper`, `fingers`, `knife_post`, `knife_clamp`, `knife_handpiece`, `knife_blade`,
  `wrist_camera_link` / `wrist_camera_optical_frame`, `wrist_camera_bracket`, `knife_camera_link` / `knife_camera_optical_frame`, `knife_camera_bracket`,
  and **`knife_tip`** = `T_link6_tcp` of the design. `knife_tip` and the blade have no collision geometry (the last 20 mm of the blade are exempt, as
  in the v3 study). The tooling is identical on both arms (not mirrored), as in `cell_planner_check`. Camera optical frames: z = viewing axis; the
  roll about it follows the design matrix (assumed x = image width).
* Collision model = the one the v3 study validated in MoveIt (`gen_cell.py` real geometry, fingers = union of the grasp and open boxes, camera boxes
  padded 3 mm). Camera A bracket is a plain cylinder, camera B bracket a capsule (as in gen_cell / gen_cell_v3).

## Arm macro vs. our single-arm URDF
`rebel_arm_macro.xacro` is compal_robot's macro with OUR joint limits and velocity (their limits were 1 deg wider, j2/j3 -80..140, velocity 0.785398). With
that change `test/check_arm_fk.py` finds **no difference** against `igus_rebel_description/urdf/igus_rebel2.urdf` for all 23 joints+links (origin, axis, limits,
inertia, visual and collision geometry, mesh names) and max link6 FK error 0.0 m / 0.0 rad over 364 configurations per arm (300 random + the 64 corners of the
joint box), for the macro alone and for both arms of the expanded dual-arm URDF.

## Slider positions / arm spacing s = 0.448 m
The slider joints are at compal_robot's zero (arm spacing 459.946 mm) in the URDF. Our spacing is set by the **initial and `home` values**
`slider_left_joint = -0.005973`, `slider_right_joint = +0.005973` ((0.459946 - 0.448)/2 each, sliders move inward):
`rebel_dual_arm_stand_moveit_config/config/initial_positions.yaml` (-> ros2_control `initial_value`, i.e. the mock start state) and the SRDF state `home`.
The lift is at compal_robot's -0.2 (arm base plane at z = 0.9093). Checked: `scan_results/dual_arm_import_20261006/spacing_check.txt`.
Sliders moved further in than ~0.09 m (left)/0.10 m (right) hit the stand's centre triangle support (compal_robot measurement, enabled in the SRDF).

## Tooling revisions
Everything geometric lives in `config/cell_tool_*.yaml` (a list of links, each with a parent, a joint origin and visual/collision primitives as xyz/rpy
strings, plus `T_link6_tcp` and the camera matrices). A new tooling (v4, ultrasonic cutter, `cad/dual_arm_tooling_v2`) = new design JSON ->
`scripts/make_tool_config.py` (or a hand-written YAML) + meshes in `meshes/tool/`, then `tool_config:=<file>` (xacro arg, also a launch arg of the
MoveIt package). The names other parts rely on are `${prefix}knife_tip`, `${prefix}wrist_camera_link` / `_optical_frame`, `${prefix}knife_camera_link` /
`_optical_frame` and the tool links listed in the SRDF groups `arm_*_tool` (regenerate the SRDF with `rebel_dual_arm_stand_moveit_config/scripts/make_srdf.py`
when the link set changes).

## Not validated / known gaps
* No RViz/visual inspection was done (headless run). Collision and kinematics were checked numerically and with MoveIt only.
* The head camera (`gemini336L_*`), camera riser and boom are compal_robot's manually placed placeholders; their pose on the real stand is unmeasured.
* Inertias of the rig links are compal_robot's rough placeholders; the tool links have no inertial blocks (not needed for planning; add before any dynamics use).
* Tool masses/moments are not in the model (cell_planner_check summary section 6 has assumed values).
* `lift` / `slider` velocity and acceleration limits are compal_robot's placeholders (0.3 m/s, 0.5 m/s^2).
* The Isaac USDs do not match the tooling (see `isaac/README.md`). Isaac Sim is not installed here; `isaac_topic` was only checked to the extent that the
  hardware plugin loads and the controllers activate with no simulator attached.
* Nothing was run on real hardware.
