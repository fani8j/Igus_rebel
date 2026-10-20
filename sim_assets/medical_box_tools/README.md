# Medical-box tooling: Isaac Sim 5.1 USD assets

Two ReBeL end effectors for the shelf pick, simulated in **Isaac Sim 5.1.0**: left = 3-pad suction tool v5 with a real **Surface Gripper**
and spring-loaded pad buffers, right = SMC LEHF20K2-48 parallel gripper. Both are articulations in the ReBeL **link6 frame** (+Z out of the flange).
Sources (read-only): `cad/medical_box_tooling_v1/{suction_tool_v5,gripper_tool,common}`; cartons are referenced from `sim_assets/medical_boxes/usd/`
(other session, not edited). Everything below the "Tests" heading was run headless in Isaac Sim 5.1 (`tests/`, results in `tests/results/`).

    usd/left_suction_tool.usda            articulation: base + 3 pad heads on prismatic buffers + IsaacSurfaceGripper (3 D6 attachment points)
    usd/right_gripper_tool.usda           articulation: base + 2 fingers, left drive + PhysX mimic joint (default)
    usd/right_gripper_tool_independent.usda   same, two independent finger drives (fallback)
    usd/pick_test_scene.usda              dynamic test scene: shelf, 5 packed cartons, both tools on a slide joint (used by tests/)
    usd/shelf_cartons.usda                shelf + 5 cartons only (no PhysicsScene), referenced by the rig scene
    usd/shelf_handoff_scene.usda          static (kinematic) nominal handoff, 8 cartons, both tools (as before; joints deactivated)
    usd/rebel_left_suction.usda, rebel_right_gripper.usda   single ReBeL (referenced) + tool fixed to link_8 in its articulation
    usd/d200_dual_arm_medical.usda        MAIN RIG: dual-arm stand on the D200 AMR (referenced from the external drive), XEG-32 removed, both tools on link6, AMR static placeholder
    usd/d200_rig_test_scene.usda          ground + PhysicsScene + the D200 rig (test scene; no shelf yet)
    usd/left_suction_tool_arm_left.usda, right_gripper_tool_arm_right.usda   tool variants with URDF joint names (arm_left_pad_{1,2,3}_buffer_joint, arm_right_finger_{left,right}_joint) for the D200 rig
    usd/cabinet_grasp_test_scene.usda     the REAL stocked medicine cabinet (referenced) + two kinematic tool carriers: used by tests/test_cabinet_grasp.py
    usd/dual_arm_rig_medical.usda         LEGACY stand-only rig (referenced, 13 MB on the external drive) with both tools, XEG-32 removed, shelf (superseded by the D200 rig)
    usd/previews/*.png                    matplotlib renders of the stage geometry (not Isaac renders)
    tools/build_tool_usd.py               rebuilds all .usda from the CAD folders (pxr + trimesh + numpy)
    tools/verify_tool_usd.py              structural checks (pxr only) -> data/verify.json (146 checks)
    tools/tool_api.py                     helpers to command the tools inside Isaac (surface gripper, vacuum level, finger ramps)
    tests/*.py, run_all.sh, summarize_log.py   headless Isaac tests; tests/results/ = logs, json, screenshots

Rebuild / verify / test:

    PY="/media/farhan/My Passport/igus_nvidia/venvs/usd/bin/python"
    nice -n 19 "$PY" tools/build_tool_usd.py && nice -n 19 "$PY" tools/verify_tool_usd.py
    tests/run_all.sh        # loads every USD, calibration, left tool, right gripper, assemblies, D200 rig (+ both cabinet grasp modes with videos, about 1 h; QUICK=1 skips them)
                            # ROS_DOMAIN_ID=170, ROS_LOCALHOST_ONLY=1, nice 19, no ROS bridge
    nice -n 19 /home/farhan/isaac-sim/python.sh tests/capture_scenes.py     # screenshots

## Conventions
Metres, kg, Z up, `defaultPrim`, `kind = component`, customData `source` / `generator_version`. Collision = the CAD convex pieces as Mesh prims with
`CollisionAPI` + `MeshCollisionAPI(convexHull)`, `purpose = guide`. Physics materials: pad rubber **0.5 / 0.5**, other tool parts 0.4 / 0.3. With the carton's
own material (0.35 / 0.3, combine mode "average") the effective pad/carton friction is ~0.43. Values are float32 in USD. **CPU physics only** (the Surface
Gripper extension needs it); the scenes set TGS at 240 Hz.

## Left tool: `/left_suction_tool` (suction tool v5)
Articulation root on the root Xform (no body there). Bodies (all authored at the identity pose, i.e. in the link6 frame):

| prim | what | mass | basis |
|---|---|---|---|
| `base` | printed body, Gemini 305, collector, tubes, elbows, nuts, washers, rear buffer bodies (z < 144.3 mm of each ZP2 part) | 0.19385 kg | CAD total (0.20381 kg, `data/mass.json`, URDF) minus the 3 heads |
| `pad_1`, `pad_2`, `pad_3` | sliding head (pad + buffer rod, z >= 144.3 mm of the `pad_L/M/R` mesh, cut and capped) at x = -60 / 0 / +60 mm | 3.32 g each | share of the 12 g pad part by volume (27.7 %, ASSUMED uniform density; the CAD pad is ONE solid) |

CoG and inertia of `base` are computed so that **base + 3 heads reproduce the CAD tool exactly** (mass, CoG, full inertia tensor, checked to 1e-5).
Deviation from the task text: the task said "pad + buffer rod, 12 g from the bodies list"; moving the whole 12 g ZP2 part would slide the buffer body through the
nuts and the elbow, so only the head moves and the rest of the 12 g stays on the base (total preserved).

**Pad buffers** `joints/pad_{1,2,3}_buffer_joint`: PhysicsPrismaticJoint, axis **+Z**, body0 `base`, body1 `pad_i`, anchor at the lip centre (x, 0, 0.159).
q = 0 extended (lip z 159 mm), **q = -0.010 fully compressed** (lip 149; the TCP "compressed" z 154 = q -0.005). Joint-to-joint collisions off.
Spring = linear force drive: **stiffness 200 N/m, target position +0.005 m** (beyond the extended limit) -> force K (target - q) = **1.0 N preload at full extension,
2.0 N at 5 mm, 3.0 N at full compression** (SMC ZP2 data, params.PAD: spring 1 to 3 N over 10 mm). Damping **1.6 N s/m** (about critical for the 3.3 g head: 2 sqrt(K m) = 1.6),
drive maxForce 6 N. Initial joint state 0 (`PhysicsJointStateAPI`).

**Surface Gripper** `/left_suction_tool/surface_gripper` (`IsaacSurfaceGripper`, schema of `isaacsim.robot.schema` 5.1, controlled by `isaacsim.robot.surface_gripper`):
* `isaac:attachmentPoints` -> `surface_gripper_joints/pad_{1,2,3}_attachment`: generic `PhysicsJoint` (D6) with `IsaacAttachmentPointAPI`, `isaac:forwardAxis = Z`,
  `isaac:clearanceOffset = 0`, **body0 = the pad head** (`pad_i`), body1 = `base` (placeholder, the extension re-targets it to the gripped body), localPos0 = pad lip centre (x, 0, 0.159),
  frames identity (joint +Z = out of the lip), `excludeFromArticulation`, enabled, no break force. Limits and drives copied from NVIDIA's gantry example
  (`isaacsim.robot.surface_gripper/data/SurfaceGripper_gantry.usda`): transX/Y locked, transZ 0..10 mm with a 5000 N/m, 100 N s/m drive (pulls the object to the pad), rot limits +-3 deg.
  In Isaac 5.1 different body0 per attachment point works (the example uses one body0; checked: 3 pads, 3 bodies).
* `isaac:maxGripDistance = 0.004` m (pads at or within 4 mm of contact), `isaac:retryInterval = 1.0` s (the gripper keeps trying to attach for 1 s, then goes back to Open).
* **Force limits per attachment point, from ONE vacuum parameter** (design vacuum -50 kPa, Schmalz ECBPi, params.PAD area 1.16 cm2). PHYSICAL values:
  `coaxial = vacuum x area = 50 kPa x 1.16e-4 m2 = 5.8 N`, `shear = mu x coaxial = 0.5 x 5.8 = 2.9 N` per point (mu 0.5 = pad rubber). Stored in customData of the prim
  (`vacuum_kpa`, `pad_area_m2`, `shear_mu`, `cal_factor_*`); change the level with `tool_api.set_vacuum_level(stage, SG_PATH, kpa)` before play (builder constant `VACUUM_KPA`).
  **The authored attributes are not the physical numbers**: Isaac Sim 5.1 needs a calibration (see "Surface Gripper calibration"): authored shear limit = physical x **2.2** (6.38 N at -50 kPa),
  authored coaxial limit = physical (5.8 N, but the 5.1 extension does not enforce it for pull-off), and the pull-off limit is enforced by `tool_api.SuctionLimitMonitor`
  (attach it once per run: it opens the gripper when any pad's pull-off tension exceeds the physical coaxial limit).
* **Timing:** real suction needs about 0.05 to 0.2 s to build up the vacuum; the Surface Gripper attaches **instantly** when closed. The tests model the pump with
  `VACUUM_DELAY_S = 0.15`: press, wait 0.15 s, then `close`. The extension itself has no ramp: after attach the joint force limits apply at once.

Frames as children of the root (exact URDF values): `suction_tcp` (z 159), `suction_tcp_compressed` (z 154), `wrist_camera_mount`, `wrist_camera_color_optical_frame`,
`pad_{1,2,3}_suction_frame` (lip centres, +Z out of the lip). `base/link6_mount_frame` (identity) is the frame to fix to the robot link6.
`joints/world_anchor` is a **disabled** FixedJoint to the world (stand-alone use).

## Right tool: `/right_gripper_tool` (SMC LEHF20K2-48, ArticulationRootAPI)
Bodies `base`, `finger_left`, `finger_right` (masses / CoG / inertia from the URDF = CAD, unchanged: finger 52 g, see `data/verify.json`), prismatic joints along X, 0..0.024 m per finger,
**q = 0 closed (pad gap 8 mm), q = 0.024 open (56 mm); gap = 8 + 2q**; both open with +q (the left joint frames are rotated 180 deg about Z). Authored at open, initial joint state authored.

| item | value | source / reason |
|---|---|---|
| stroke | 24 mm per finger (48 mm total) | URDF limits, SMC catalog (LEHF20K2-48 long stroke) |
| grip force | **24.3 N per finger** (drive maxForce; 48.6 N on the single left drive in the mimic file) | SMC catalog: 11 to 28 N for the LEHF20 range; 24.3 N = 100 % pushing force at the finger length L = 43.2 mm (cad `gripper_tool/README.md`), 18.2 N with the -25 % accuracy; 28 N would be the force at L = 10 mm and is not reachable with these fingers |
| finger speed | open/close **40 mm/s per finger**, pushing **10 mm/s per finger** | SMC LEHF20 catalog: 5-80 mm/s open/close and 5-20 mm/s pushing "for both fingers; one finger is half" (the old URDF value 0.05 m/s was not the per-finger catalog limit). Enforced as a **target ramp** (`tool_api.finger_target_profile`), see below |
| drive | force drive, stiffness 1e4 N/m, damping 200 N s/m | tuned in Isaac (test_right_gripper): closes on 12 and 16 mm without overshoot (< 0.1 mm) or jitter (< 40 um) |
| pad friction | 0.5 / 0.5 | as before |
| close command | target **-0.010 m** (past the closed stop) | the drive force then saturates at its limit on any carton thickness; a target of 0 would give only K x 2 mm = 20 N on a 12 mm carton |

* **Mimic (default file).** Tested in Isaac 5.1: `PhysxMimicJointAPI:rotX` on `finger_right_joint` (reference `finger_left_joint`, gearing -1, offset 0, no own drive) works reliably:
  the two fingers stay symmetric to 0.001 mm and the carton is centred, force transmitted = the left drive limit (so the left drive has maxForce 2 x 24.3 N, giving 24.3 N per finger, measured 24.4 / 24.2 N).
  Command only `finger_left_joint` (the right target is ignored). The independent file (`right_gripper_tool_independent.usda`, maxForce 24.3 N each) also works but the carton wanders sideways by
  +-0.3 mm (two force-controlled fingers have no centring); use it as fallback.
* **Do not use `physxJoint:maxJointVelocity`** for the 40 mm/s: in Isaac 5.1 it roughly halves the drive force (measured 11 to 13 N instead of 24.3 N). The USD does not author it.
* The CAD jaw solids of the two fingers touch face to face (left jaw_low / right jaw_top at z = -7.1 mm) and their convex hulls overlap in x below q = 18.4 mm: that jammed the closing in the simulator,
  so `finger_left` has a `FilteredPairsAPI` against `finger_right` (the pads still collide with the carton).
* Frames `gripper_tcp` (z 0.1932), `top_stop_frame` (z 0.1675); `joints/world_anchor` disabled FixedJoint.

## Test scenes
`pick_test_scene.usda` (frame W as in the handoff: origin shelf floor at the front edge below the target carton, +X along the shelf, +Y into the shelf): shelf with the back board right behind the cartons
(250 mm deep, cartons rest against it) and 0.5 mm side clearance, cartons `[1, 2, 3, 5, 1]` (all 250 mm deep, 15/16/16/12/15 mm thick, packed with 0 mm gap, spines flush at y = 0),
target = index 2 (box 3, 16 mm), index 3 = box 5 (12 mm). Each tool hangs on a **carrier**: a static articulation root (`*_carrier`, fixed to the world at the nominal handoff pose) and a
prismatic `joints/world_slide` along the **tool +Z** with a position drive (k 1e5, c 3e3, max 300 N); slide s = 0 is the nominal handoff pose
(suction: pads compressed 5 mm on the spine; gripper: stop face 4 mm above the carton top; for the gripper test the carton is pulled 100 mm out and the gripper starts 120 mm above).
Left tool starts 30 mm before the spine. `shelf_handoff_scene.usda` is unchanged in layout but static: all tool bodies are kinematic, articulation roots, tool joints and the surface gripper are
deactivated there (PhysX rejects joints between kinematic bodies), the pads are shown compressed 5 mm, the gripper uses the independent-finger file.

## Importing into Isaac Sim 5.1 (GUI)
1. Start Isaac Sim 5.1 (`/home/farhan/isaac-sim/isaac-sim.sh`). Do **not** enable the ROS 2 bridge if a real robot is on the network. For the Surface Gripper:
   Window > Extensions, enable **isaacsim.robot.surface_gripper** (and `.ui` for the property panel and Create > Robots > Surface Gripper).
2. Stand-alone tool: File > Open `usd/left_suction_tool.usda` (or `right_gripper_tool.usda`). Add Create > Physics > Physics Scene (and a ground plane if wanted; the scenes already have one),
   select `.../joints/world_anchor`, tick **Joint Enabled** (otherwise the floating tool falls), press Play. Keep the folder layout; scenes reference `./*.usda` and `../../medical_boxes/usd/*.usda`.
3. As a part of your own scene: drag the .usda into the stage (or Stage > Add > Reference) under `/World`; place the root Xform at the link6 pose.
4. Demo: File > Open `usd/pick_test_scene.usda`, Play, then run the Python snippets below in Window > Script Editor. `usd/dual_arm_rig_medical.usda` and `rebel_*.usda` open the same way
   (they reference the rig / ReBeL by absolute path, see below).

## Attaching to a ReBeL
The tool root is the link6 frame; the tool bodies must be a **fixed joint away from the robot link6 in the same articulation**:
* Remove the tool's own `ArticulationRootAPI` (select `/left_suction_tool`, Property > Physics > remove Articulation Root) and keep the robot's. Disable `joints/world_anchor` (or leave it disabled).
* Create a fixed joint (Create > Physics > Joint > Fixed): body0 = robot link6, body1 = `.../left_suction_tool/base` (right tool: `.../right_gripper_tool/base`), local positions and rotations identity
  (tool frame = link6). Place the tool prim at the link6 pose first (or parent it under link6 with identity). The tool bodies may NOT be nested below a rigid body without
  `resetXformStack` in Isaac 5.1 (the physics parser rejects nested rigid bodies, see the rig note) - keep them as siblings or flatten like `build_rig_medical()`.
* With the **Robot Assembler** (Tools > Robotics > Assembler): base robot = the ReBeL, mount frame = its link6 (any prim under the link6 body); attach robot = the tool root, mount frame
  `/left_suction_tool/base/link6_mount_frame` (right tool: use the `base` prim). The assembler removes the attached root joint / articulation root and creates the `AssemblerFixedJoint`, which is
  what the builder does by hand (`attach_tool()` in `tools/build_tool_usd.py`). **Not clicked through in the GUI here**; the code path (fixed joint between the mount frames) is what the assembled USDs use and test.
* `usd/rebel_left_suction.usda` / `rebel_right_gripper.usda` are the result for the **single ReBeL USD found on this machine**
  (`/home/farhan/robot_remote/igus_rebel_sim/rebel_6dof.usd`, Isaac 5.1 URDF import, another session's file, referenced by absolute path, override with env `REBEL_USD` and rebuild).
  The tool is fixed to `link_8` (joint6 axis z = ReBeL link6 convention) with identity frames at the robot's USD configuration (all joints 0). Caveats: that URDF's `flange`/`tool0` frames are rotated
  -90 deg about Y relative to link_8 (iRC convention): the tool uses link_8, **not** flange; that robot's q = 0 is NOT the upright pose of `igus_rebel2.urdf`; the file prints its own
  "unresolved reference ... rebel_6dof_physics.usd" warnings (not caused by the tools). Other ReBeL USDs exist (`/home/farhan/Downloads/igus_rebel_from_xacro/`, `Compal_Rebel_USD/`), not used.

## Commanding the tools
Surface gripper (Python, e.g. Script Editor while playing; `SG = "/World/Tools/left_suction_tool/surface_gripper"`, in the rig `.../link6/left_suction_tool/surface_gripper`):

    import sys; sys.path.append("<repo>/sim_assets/medical_box_tools/tools"); import tool_api
    iface = tool_api.suction_interface()            # acquire_surface_gripper_interface() + set_write_to_usd(True)
    tool_api.suction_close(iface, SG)               # == iface.close_gripper(SG);  tool_api.suction_open(iface, SG)
    tool_api.suction_status(iface, SG)              # 'Open' | 'Closing' | 'Closed';  iface.get_gripped_objects(SG)
    tool_api.set_vacuum_level(stage, SG, 50.0)      # kPa -> coaxial / shear limit of all attachment points (before play)
    # batch / Isaac Lab style:  from isaacsim.robot.surface_gripper import GripperView; v = GripperView(paths=SG); v.apply_gripper_action([0.5])   # > 0.3 close, < -0.3 open

GUI: select the `surface_gripper` prim; the Surface Gripper property panel (isaacsim.robot.surface_gripper.ui) shows status, limits and gripped objects. OmniGraph: node
**Surface Gripper** (`isaacsim.robot.surface_gripper`): inputs `SurfaceGripper` (target = the prim), `enabled`, execution `Toggle` (closes when Open, opens otherwise). The gripper needs the pads at or within 4 mm of the
carton face when `close` is called (it retries for `retryInterval` s). Buffers act automatically (springs).
Finger drives (articulation, e.g. `isaacsim.core.prims.Articulation`; dofs `finger_left_joint`, `finger_right_joint`):

    # open 0.024, close past the stop -0.010 (force saturates at 24.3 N per finger). Ramp the target at the catalog speeds:
    for q in tool_api.finger_target_profile(0.024, tool_api.GRIP_CLOSE_Q, dt, push=True):   # 40 mm/s to a 32 mm gap, then 10 mm/s
        art.set_joint_position_targets(np.array([[q, q]]), joint_indices=np.array([il, ir])); step()
    # USD alternative: tool_api.set_finger_targets_usd(stage, "/right_gripper_tool", q)  (writes drive:linear:physics:targetPosition; before play)

Slide carriers of the test scenes: `Articulation("/World/Tools/left_suction_tool_carrier")`, dof `world_slide` (m along the tool +Z).

## Physical values and sources (summary)
Pad buffer: SMC ZP2-T4030WNK10-B5, stroke 10 mm, spring 1 to 3 N, area 1.16 cm2 (`common/params.py` PAD, SMC catalog). Pump: Schmalz ECBPi, design vacuum -50 kPa (task; max -75 kPa).
Pad friction 0.5 (cad gripper README hold check). Gripper: LEHF20K2-48 force 11 to 28 N, 24.3 N at L = 43.2 mm, speeds as above (`/home/farhan/Downloads/SMC_LEH_Electric_Grippers_English_latest.pdf`,
"LEHF" specifications table: open/close 5 to 80 mm/s, pushing 5 to 20 mm/s, both fingers). Masses / CoG / inertia: CAD (`data/mass.json`, URDFs). ASSUMED: pad head mass share (volume), damping 1.6 N s/m, the gripper drive gains, the
`VACUUM_DELAY_S` 0.15 s, carton friction (cardboard 0.35/0.3 from the carton USD), carton masses (0.082 to 0.107 kg, carton USD estimates).

## Tests (Isaac Sim 5.1.0, headless, CPU physics; `tests/results/`)
All of these ran with `ROS_DOMAIN_ID=170 ROS_LOCALHOST_ONLY=1`, nice 19, no ROS bridge; GPU was idle (0 %, ~0.8 GB used by the desktop) before the run.

**Left suction (`test_left_suction.py`, 18/18)** - carton 3 (82 g) flush in the packed shelf, resting on the shelf floor against the back board:
* approach 20 mm/s, press to s = 0: buffer compression **4.30 / 4.30 / 4.30 mm** (the lip touches at 5 mm travel, the 0.7 mm difference is contact softness), spring force 1.86 N per pad (K (0.005 + 0.0043)), carton pushed 0.5 mm.
* pump delay 0.15 s, `close`: status **Closed**, gripped = [target carton] in 1 s; 3 attachment points registered (the extension does not write `body1`, so the per-point attachment was proven with the "lost pad" test below).
* pull 100 mm of carton travel at 50 mm/s (tool retracts 105 mm = 100 + the buffer extension): carton displacement **100.46 mm**, spine-to-lip offset **0.1 mm** max, peak pull force 5.2 N (friction on the shelf floor + neighbours), still Closed.
* release (`open`), tool retracts 30 mm: status Open, gripped none, carton moved **0.0 mm**, z = 0.1100 m (on the floor). Because the pads are already fully extended after the pull there is no spring push on release.
* **Negative / detach tests (calibrated, with the physical pull-off monitor)** - carton mass raised, pull 100 mm at 50 mm/s, `first_open_event_force_N` = pull force when the status first leaves Closed:
  at -50 kPa (physical 3 x 5.8 = 17.4 N): 2 / 3 / 3.5 kg held (peak pull 9.2 / 13.4 / 15.6 N), **4 kg and heavier detach at 17.4 N** (ratio 1.00); at -25 kPa (physical 8.7 N): 1 / 1.5 kg held, **2 kg and heavier detach at 8.8 N** (1.01).
  With **one attachment point only** (lost pad) a 4 kg carton cannot be held. Detach asserted within +-15 % of the physical value (`calibrate_surface_gripper.py` section D, clean pull test, is the primary proof).
* Real cartons (0.08 to 0.15 kg) need ~0.5 to 1 N pull; margin is > 30x.

**Right gripper (`test_right_gripper.py`, 24/24, mimic file)** - carton pulled 100 mm out, gripper descends from 120 mm above:
* open: gap **56.0 mm** (q 24.000 / 24.000 mm); close (40 mm/s to 32 mm, then 10 mm/s): final pad gap **16.004 mm** on the 16 mm carton and **12.000 mm** on the 12 mm carton (limit +-0.5), min gap during closing 15.92 / 11.90 mm (overshoot <= 0.1 mm), gap peak-to-peak over the last 0.5 s 38 / 8 um, carton centred (|q_l - q_r| < 0.001 mm), finger speed max 40.0 mm/s.
* grip force per finger (joint effort): **24.41 / 24.19 N (16 mm), 24.38 / 24.22 N (12 mm)** = the 24.3 N drive limit.
* lift 50 mm at 25 mm/s: carton 49.87 / gripper 49.99 mm -> **slip 0.11 mm** (16 mm), 0.003 mm (12 mm); independent-drives file, exploratory run (not in the test): 1.0 kg slips < 0.3 mm, 1.5 kg 26 mm, 2 kg full (capacity ~1.2 kg, the pick needs 0.15 kg).
* open: gap 56.0 mm, rising gripper leaves the carton on the floor (z = h/2).
* finding: two independent force-controlled fingers do not centre the carton (+-0.3 mm drift) and `maxJointVelocity` halves the force -> mimic + target ramp.

**Assemblies (`test_assemblies.py`, 34/34)**
* `rebel_left_suction`, `rebel_right_gripper`: load, 6 arm joints (`joint1..joint6`) + tool DOFs in one articulation (9 / 8 DOF), arm sags 0.65 / 0.80 deg and the tool 9.4 / 11.6 mm from the authored pose within 3 s (the robot's 400 N m/deg drives), then rests (< 0.001 rad, 0.0 mm in the last second);
  the Surface Gripper closes on a block held at the pad lips (status Closed, gripped = block) and opens; the finger drives close to **8.0 mm** (symmetric) and reopen to 56.0 mm; the tool follows joint6 (+0.5 rad -> 28.67 deg).
* `dual_arm_rig_medical`: 20 DOF (15 URDF joints + 3 pad buffers + 2 fingers), **no joint of the URDF is missing** (`lift_joint`, `slider_left_joint`, `slider_right_joint`, `arm_{left,right}_joint1..6`);
  holds the README home pose for 3 s (arm joints sag 0.89 deg at most, at rest: 6.9e-8 rad over the last second; lift -0.20038, sliders -0.00591 / +0.00588 m, i.e. 0.4 mm and 0.06 mm from -0.2 / -+0.005973),
  no collapse or explosion; left Surface Gripper closes on a block at the pad lips and opens, pads respond; right fingers close to 8 mm / open to 56 mm.
  The lift and sliders settle 18 mm (lift, sliders) from the reported start state in the first second (the authored start state is not exactly the home pose).

**Load test (`test_load*.py`, 9 files, `warnings_summary.txt`)**: every USD opens and steps 1 s with the Surface Gripper extension enabled. The tool files and `pick_test_scene`, `shelf_cartons`, `shelf_handoff_scene` give **no schema or physics warning/error** (0 for the 3 tool files);
the only message after the stage opens is Kit's `[omni.hydra] Mesh ... update topology/point without updating normal` for the carton meshes (other session's assets) and, during runs, the extension's
`[omni.fabric.plugin] attribute isaac:attachmentPoints / isaac:grippedObjects not found for path .../surface_gripper` (Fabric does not know the attributes the extension writes; harmless, the attributes exist in the USD).
`rebel_*` and the rig additionally print `Unresolved reference prim path ...rebel_6dof_physics.usd</visuals/tcp|flange|tool0>` and `getAttributeCount called on non-existent path .../Joint6/World`: those come from the referenced
`rebel_6dof.usd`, not from the tools. Kit start-up deprecation warnings (omni.isaac.*, powersave profile, IOMMU) occur before the stage is opened and are not listed.

## Surface Gripper calibration (Isaac Sim 5.1; `tests/calibrate_surface_gripper.py`, `tests/results/surface_gripper_calibration.json`)
Problem found: with the physical limits authored (5.8 N coaxial / 2.9 N shear per point) the sim held 2.3 to 4.8x more than the real pads would (the first friction tests detached at ~49 N instead of 17.4 N), i.e. the
sim was optimistic. Measured with a CLEAN pull test: one free carton (82 g), no shelf / neighbours / floor, gravity 0, the tool held still by its slide drive, the pads touching the spine, gripper closed,
then a force ramp applied to the carton through the centroid of the attached lips until the status leaves Closed (rate independent: 2 vs 8 N/s the same). 3 attachment points unless stated. "Physical" = 3 x the per-point value.

| test (authored limits = physical) | -25 kPa | -50 kPa | -75 kPa | detach / physical |
|---|---|---|---|---|
| pull-off along the pad axis (away from the pads), total N | 23.2 | 50.0 | 58.9 | 2.67 / 2.87 / 2.26 (not proportional) |
| shear along the pad row (tool x), total N | 1.74 | 3.49 | 5.24 | 0.40 (linear in the limit) |
| shear across (tool y), total N | 2.07 | 4.15 | 6.23 | 0.48 (linear) |
| ONE pad (pad_2, force through its centre), shear x / y, N | | 2.90 / 2.90 | | **1.00: the shear limit is exact PER ATTACHMENT POINT** |

* The **shear limit is per attachment point and exact** (single pad: 2.900 N at a 2.9 N limit). The three rigid points share a lateral load unevenly: the 3-pad group detaches at only 1.21 x (tool x) / 1.43 x (tool y) the single-point value
  (a single-pad run: 2.9 N; three pads: 3.5 / 4.2 N), so the group is weaker than 3 x the per-point limit.
* The **pull-off responds to the coaxial limit only weakly and non-linearly**: coaxial limit 0.05 N -> 31.6 N total, 5.8 N -> 50.0 N, 58 N -> 84.2 N (shear limit 2.9 N); a 10x larger shear limit does not change it (50.0 N). There is a floor of ~30 N (independent of the setting),
  so a factor on the coaxial limit can never reach 17.4 N: a scan with factors 0.30 / 0.35 / 0.40 (and the calibrated shear limit) gives 33 to 41 N at -50 kPa. The shear detach is independent of the coaxial limit.
* Both limits scale with the vacuum setting through `set_vacuum_level` (same factors at 25 / 50 / 75 kPa).

Calibration implemented (defaults in the USD, `tool_api.set_vacuum_level` applies them, README numbers above stay the physical ones):
1. **shear factor 2.2** (`cal_factor_shear`): authored shear limit = 2.2 x mu x coaxial = 6.38 N at -50 kPa. Result (3 pads): shear x 7.69 N (0.88 x the physical 3 x 2.9 = 8.7 N), shear y 9.15 N (1.05 x); same ratios at -25 and -75 kPa. Both within +-15 %.
   Caveat: with ONE attachment point (lost pad) the shear limit is now 2.2 x too high (6.4 N instead of 2.9 N): optimistic for the lost-pad case.
2. **pull-off (coaxial): `tool_api.SuctionLimitMonitor`** (per physics step, from the pad buffer joint efforts; tension = -effort, verified: the three efforts sum to the applied force) opens the Surface Gripper when any pad's tension exceeds the physical coaxial limit
   vacuum x pad area (5.8 N at -50 kPa). `cal_factor_coaxial` = 1.0 (a factor cannot do it, see above). Result: pull-off detaches at **17.18 N at -50 kPa (0.987 x the physical 17.4 N)**, 8.60 N at -25 kPa (0.989 x 8.7), 25.76 N at -75 kPa (0.987 x 26.1); logged `tests/results/calibrate_surface_gripper.log` (section D, +-15 % asserted).
3. `tests/test_left_suction.py` asserts the friction-pull detach at the physical limit (+-15 %): -50 kPa detaches at 17.4 N (carton 4 kg), -25 kPa at 8.8 N (2 kg); lighter cartons held.
The monitor must be attached in every script that uses the pull-off limit (the cabinet grasp test does).

**Gripper force check:** the right gripper's drive limit is 24.3 N per finger (48.6 N on the mimic master). `test_right_gripper.py` now samples the finger joint efforts over a 0.25 s hold: per-finger mean must be 24.3 N +-2 %, no single sample above 24.3 x 1.04
(solver noise moves load between the fingers: samples up to 24.86 N = 1.023 x were seen on one finger while the other read 24.2 N), and the SUM of both fingers must never exceed 2 x 24.3 x 1.02. Measured: mean 24.39 / 24.21 N (16 mm), 24.19 / 24.41 N (12 mm), sum <= 48.6 N,
i.e. the limit is not exceeded; the earlier 24.41 N reading was noise on the effort sensor (+0.4 %), not an over-limit force (PhysX clamps the master drive at 48.6 N).

## Main rig: dual-arm stand on the D200 AMR (`usd/d200_dual_arm_medical.usda`)
Newest model from the operator (`/home/farhan/Downloads/d200_with_dual_arm.zip`, copied unchanged to **`/media/farhan/My Passport/igus_nvidia/sim_assets_ext/d200/`**: `d200_with_dual_arm.usd` -> `AMR_D200_IsaacSim/AMR_D200.usd` + `dual_arm_upper/dual_arm_upper.usda`;
the repo layer `usd/d200_dual_arm_medical.usda` (12 KB) references it by absolute path, env `D200_ASSET_DIR` + rebuild to relocate). It **replaces** the stand-only rig as the base (`dual_arm_rig_medical.usda` below is kept only as a legacy variant).
* Its upper part is already flat (40 sibling rigid bodies, no nesting), so the Isaac 5.1 problem of the stand-only rig (nested bodies) does not occur; the lift initial position is 0, the medical file sets the home state: lift -0.2 (valid: limits -0.6235 .. 0.6265, arm base plane 1.174 m above the floor), sliders -+0.005973, arms upright.
* Removed: `xeg32_*` bodies (4 per arm) and their joints (4 per arm). Mounted: v5 suction tool on `arm_left/link6`, LEHF20K2-48 gripper on `arm_right/link6` as sibling bodies with the link6 frame, fixed joints `Joints/arm_{left,right}_medical_tool_mount`, tool articulation roots removed. The Surface Gripper, pad buffers and mimic fingers work mounted.
* **Joint names = the MoveIt URDF / ros2_control names** so the `isaac_topic` bridge needs no mapping: `lift_joint`, `slider_left_joint`, `slider_right_joint`, `arm_{left,right}_joint1..6` (checked against the URDF), and (tool variants `usd/*_arm_*.usda` used only in the D200 rig)
  `arm_right_finger_left_joint` (driver) / `arm_right_finger_right_joint` (mimic), `arm_left_pad_{1,2,3}_buffer_joint` (fixed in the URDF, the bridge ignores them). The standalone tool files keep the short names (`finger_left_joint`, `pad_1_buffer_joint`, ...).
* **The AMR is a static PLACEHOLDER** (its motion is owned by another team): `base_link` is welded to the world with an enabled fixed joint (`/D200/joints/world_fix`), the 14 wheel / swivel / caster joints are locked (limits 0..0) with zero drive. No AMR control or drive test is part of this repo's scripts. (For their own use the AMR README needs the carb settings
  `/physics/collisionCylinderCustomGeometry = true`, `/physics/collisionApproximateCylinders = false`, CPU physics; `tests/isaac_common.apply_d200_physics_settings()` sets them; drive script reference: `AMR_D200_IsaacSim/scripts/run_amr.py`, `isaac_control.py`.)
* `usd/d200_rig_test_scene.usda`: ground + PhysicsScene (CPU, TGS, 240 Hz) + the rig. **`tests/test_d200.py` (12/12):** loads, 34 DOFs = 15 stand/arm + 14 AMR + 5 tool, no joint missing and no XEG joint left, joint names agree with the MoveIt URDF, stands still 5 s at home
  (static base: xy drift 0.0 mm, tilt 0.0 deg, base_link z 0.0762 m), home state reached (lift -0.20041, sliders -0.00597 / +0.00597), fingers 56 -> 8 -> 56 mm (symmetric), Surface Gripper closes on a block at the pad lips and opens, pad buffers respond.
* **Deferred (operator: "focus only on grasping the box")**: the shelf placement in the rig scene and the sync of the shelf pose with the MoveIt agent (`rebel_medicine_grasp_moveit_config/config/medicine_shelf_pose.json`, W at xyz (0.78, 0, 1.05) yaw -90 deg in the D200 frame), the centre of mass / tipping margin study, the AMR drive test. The real measured
  shelf is the medicine cabinet (next section). An earlier estimate-only shelf in the rig scene and the data/medicine_shelf_pose.json copy were removed.
* The ROS 2 bridge script of the rig package and the MoveIt configs in the main repo were not run or edited here (no ROS bridge was enabled).

## Cabinet grasp tests (real medicine cabinet, kinematic tool carriers; `tests/test_cabinet_grasp.py`)
Scope (operator: "focus only on grasping the box"): the two tools take ONE carton out of the REAL stocked cabinet (`sim_assets/medical_boxes/usd/medicine_cabinet_filled.usda`, 360 dynamic cartons, referenced read-only; shelf friction 0.30/0.25, 1 mm gaps, clear heights
S1 300, S2 230, S3 260, S4 310, S5 290, S6 280 mm) with the tools on **kinematic carriers** (no arms, no IK). `usd/cabinet_grasp_test_scene.usda` = the cabinet + carriers `SX` (suction tool) and `GR` (right gripper): a static articulation root fixed to the world at the
link6 pose, a prismatic `jy` along the tool y axis and a prismatic `jz` along the tool z axis (position drives, 300 N), the tool base on `jz`. The test sets the carrier poses per target from the carton's actual pose in the cabinet (handoff.py frames: suction tool z = +Y into the shelf,
pad row vertical at mid-height; gripper per mode below). Each target is its own short run (full cabinet simulated, ~0.8x real time); metrics are read from the physics tensor views.
**Scene-layer fix:** in the cabinet every shelf is a plate (y 0.06 .. 0.40) plus a separate coplanar 10 mm front strip (`ShelfLip`, y 0.05 .. 0.06): a carton sliding over that seam was caught by PhysX (stopped when its spine reached y = 0.06, 17 N pull). The scene layer makes the board one plate 0.05 .. 0.40 and deactivates the strips (the cabinet files are untouched).
Targets (one run each, all in the middle of a packed group so both neighbours stand next to it, unless noted): T1 S1 Ultimaster, T2 S2 Accuforce (230 mm clear for a 220 mm carton), T3 S3 Agent Monorail, **T4 S3 Conqueror NC Pro (12 mm, the thinnest)**, T5 S4 Angiolite, T6 S5 Agent, T7 S6 Accuforce, T8 S6 Conqueror (thin),
**E1** last Agent of its group on S3 next to a different product (Conqueror), **E2** first Angiolite of S6 next to the left cabinet side wall. Variants on T5: suction at -35 kPa, row squeezed to 0 mm gaps, suction tool 2 mm off-centre, suction tool yawed 3 deg to the spine.

**Common first part (both modes):** (1) the suction tool approaches at 20 mm/s and presses to the nominal pose (5 mm past first contact); a loose carton yields to the 1..3 N pad springs (shelf friction ~0.4 N), so the 5 mm are shared: **buffers compress only 0.3 .. 0.5 mm and the carton is pushed back 4.5 .. 4.7 mm** (on the real shelf this means
a ~5 mm push-back, not a 5 mm buffer compression; vacuum engages at contact anyway); (2) vacuum on after the 0.15 s delay: Surface Gripper Closed on the target at 50 kPa (calibrated limits + physical pull-off monitor); (3) pull 105 mm of tool travel at 50 mm/s = **100.0 mm net carton displacement** (follow error 0.09 .. 0.26 mm).

**Mode `topdown`** (cad `gripper_tool/README.md` registration v3, position based): the gripper starts 70 mm above its final pose with the jaws open, **pre-closes to t + 4 mm**, descends at 20 mm/s along the tool axis onto the protruding part (TCP 42.5 mm behind the spine, stop face 4 mm above the top edge),
clamps at <= 10 mm/s per finger (24.3 N per finger), then **suction off, the suction tool retracts 30 mm and steps 120 mm aside** (along its camera/collector axis, away from the cabinet side wall: out of the carton's path), the gripper pulls the carton **straight out** (distance = until the carton's rear end is 20 mm in front of the cabinet, 240 .. 285 mm) at 50 mm/s, then lifts 50 mm at 25 mm/s.
**Finding: on the top shelf (S6) the closed `TopClosedPanel` (y 0 .. 0.02, z >= 1.96 m) is in the way of the gripper body when the spine is only 100 mm out** (first run: 39 907 contacts, clamp failed); the gripper body/yoke (TCP y +-27.5 mm) must stay at y < 0, so on S6 the pull-out is **150 mm** (gripper to shelf above 59 mm). All other levels use 100 mm.

**Mode `front`** (`--mode front`): after the common part, **vacuum off, the suction tool retracts straight back 100 mm along the spine normal and parks 300 mm aside**; the gripper (start 0.3 m above its grip height and 100 mm before the final depth, then lowered in free air) approaches **horizontally along +Y** (tool z = +Y_w, tool x = +X_w: jaws open across the cabinet and straddle the two big faces,
tool y = -Z_w: the 50 mm pad width is vertical, pad centre at carton mid-height), **pre-closed to t + 4 mm before the fingers pass the spine**, advances at 20 mm/s to the grip depth, clamps, pulls the carton straight out horizontally (same distance, 50 mm/s) and lifts 50 mm (up, 25 mm/s).
Orientation choice: with the jaws across the shelf the roll is fixed up to a 180 deg flip (tool x = +X, z = +Y give y = -Z); the SMC body (158 mm along the stroke axis, 35 mm vertical, ends 26 mm in front of the spine) and the yoke stay in free air in front of the cabinet, the fingers (tips 55 mm beyond the spine) stay in front of the shelf boards and the neighbours' spines (flush at y = 0.07).
**Registration limit found:** the fixed top-stop crossbar (3 mm, between the jaws, z 160 .. 167.5) and the yoke side plates would collide with the carton for any deeper grasp, so with the stop face 4 mm in front of the spine the **pads cover s = -0.8 .. 44 mm from the spine (centre 21.7 mm), not the 15 .. 70 mm asked**; reaching 15 .. 70 mm needs the stop removed or a different yoke (not simulated).

**Metrics per target** (JSON `tests/results/cabinet_grasp_<mode>.json`, summary `cabinet_grasp_summary.md`): step ok flags; pull follow error (spine to pad lip); neighbour displacement / tilt (the 2 neighbours each side, over the whole run; limit 3 mm / 5 deg); shift of the carton while the fingers close (front: suction already off); extraction slip (carton vs gripper) and lift on the floor (limit 2 mm);
carton rotation (limit 5 deg); min clearances (collision-hull vertices of each tool to the cabinet boxes (exact AABBs), the shelf above / below and the 12 neighbours: vertex-to-box distance, slightly optimistic by the hull facet size) per phase; PhysX contact reports of every tool body (unexpected = anything but the target carton and its own tool).
Results (top-down = TD, front = F; "OK" = every step and the 2 mm / 3 mm / 5 deg limits passed, no unexpected contact):

| target | level (clear mm) | box (t mm) | top-down | front | pull follow err mm | neighbour disp mm TD / F | shift on clamp mm TD / F | extract slip mm TD / F | lift on floor mm TD / F | GR to shelf above mm TD / F | GR to neighbours mm TD / F | SX to neighbours mm | carton rot deg TD / F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 mid | S1 (300) | ultimaster (16) | OK | OK | 0.16 | 0.01 / 0.01 | 0.004 / 0.071 | 0.08 / 0.13 | 0.013 / 0.009 | 9.3 / 124.6 | 29.3 / 44.6 | 6.6 | 0.07 / 0.13 |
| T2 mid | S2 (230) | accuforce (16) | OK | OK | 0.10 | 0.01 / 0.01 | 0.005 / 0.079 | 0.08 / 0.20 | 0.008 / 0.006 | 9.3 / 94.2 | 29.3 / 44.6 | 6.6 | 0.06 / 0.18 |
| T3 mid | S3 (260) | agent (15) | OK | OK | 0.13 | 0.00 / 0.00 | 0.007 / 0.093 | 0.07 / 0.03 | 0.004 / 0.007 | 9.3 / 118.5 | 29.3 / 44.6 | 6.1 | 0.09 / 0.07 |
| T4 mid (thin) | S3 (260) | conqueror (12) | OK | OK | 0.11 | 0.01 / 0.01 | 0.013 / 0.134 | 0.04 / 0.03 | 0.011 / 0.008 | 9.3 / 118.5 | 29.3 / 44.6 | 4.6 | 0.15 / 0.11 |
| T5 mid | S4 (310) | angiolite (16) | OK | OK | 0.23 | 0.00 / 0.00 | 0.012 / 0.023 | 0.19 / 0.02 | 0.005 / 0.007 | 9.3 / 124.6 | 29.3 / 44.6 | 6.6 | 0.10 / 0.06 |
| T6 mid | S5 (290) | agent (15) | OK | OK | 0.17 | 0.00 / 0.00 | 0.015 / 0.082 | 0.12 / 0.06 | 0.007 / 0.008 | 9.3 / 124.6 | 29.3 / 44.6 | 6.1 | 0.09 / 0.08 |
| T7 mid | S6 (280) | accuforce (16) | OK (pull 150) | OK | 0.14 | 0.00 / 0.00 | 0.006 / 0.188 | 0.03 / 0.02 | 0.011 / 0.019 | 59.3 / 124.6 | 79.3 / 44.6 | 6.6 | 0.14 / 0.16 |
| T8 mid (thin) | S6 (280) | conqueror (12) | OK (pull 150) | OK | 0.12 | 0.10 / 0.00 | 0.015 / 0.255 | 0.10 / 0.01 | 0.013 / 0.023 | 59.3 / 124.6 | 79.3 / 44.6 | 4.6 | 0.19 / 0.21 |
| E1 group end next to other product | S3 (260) | agent (15) | OK | OK | 0.16 | 0.00 / 0.00 | 0.005 / 0.096 | 0.18 / 0.08 | 0.009 / 0.008 | 9.3 / 118.5 | 29.3 / 44.6 | 6.1 | 0.15 / 0.09 |
| E2 group end at the left wall | S6 (280) | angiolite (16) | FAIL (pull 150) | FAIL | 65.14 | 0.00 / 0.00 | 0.000 / 0.000 | 190.04 / 168.11 | 0.000 / 0.000 | 59.3 / 124.6 | 79.2 / 120.8 | 57.9 | 0.00 / 0.00 |

Every non-wall target (9 of 10): **both modes succeed**: pull 100.0 mm (150 mm S6 TD), neighbours moved <= 0.10 mm and tilted <= 0.03 deg, extraction slip 0.01 .. 0.20 mm, the carton slides on the floor (lift <= 0.02 mm), then lifts 50.0 mm clear of the cabinet with the grip slip unchanged, rotation <= 0.21 deg, grip gap = t within 0.01 mm at 24.3 N per finger,
no tool-cabinet or tool-neighbour contact. Closest approaches: suction tool to the neighbouring cartons 4.6 mm (thin box) .. 6.6 mm while pressing (the pad nuts / bracket beside the 16 mm spine; 3 mm is the CAD clearance rule), suction tool to the shelf above 44 mm on S2 (230 mm clear) and 69 .. 114 mm elsewhere, to the shelf below 36 .. 51 mm;
top-down gripper to the shelf above only 9.3 mm (yoke / finger block against the front edge of the board above, 100 mm pull; 59 mm with 150 mm), to the neighbours 29.3 mm, suction vs gripper >= 18 mm (front: >= 152 mm, the suction tool is parked); front gripper to the shelf above 94 mm (S2) .. 125 mm and to the shelf below 86 .. 101 mm, to the neighbours 44.6 mm.

**Failures (every one) and causes**
* **E2, both modes: the suction tool cannot reach the first carton at the cabinet wall.** The tool is 106 mm wide (collision envelope -52.7 .. +53.7 mm from the pad row axis along tool y, `compactness` in the CAD verify.json); the first carton's centre (x = -0.480) is only 10 mm from the wall face (inner face x = -0.49), so the tool body hits the side wall / trim (81 029 SX->cabinet contacts top-down, 121 447 front) before the pads reach the spine,
  vacuum never closes (Open), nothing is pulled and every later step fails (the gripper then closes on nothing). Cause: tool footprint (turning the tool by 180 deg would not help: the other side extends 54 mm; not run). Workaround: take the second carton of the group first (then E2 becomes reachable), or re-stock with >= 60 mm to the wall; not a gripper problem.
* **T5/tilt3, both modes: a 3 deg misalignment of the suction tool to the spine pushes the neighbour 4.4 mm (limit 3 mm).** The pads then rotate the carton 1.2 deg while it is pulled and shove the neighbouring carton (3_pull / 6_extract fail the neighbour limit; the carton itself still comes out, 99.4 mm net). The gripper then also moves the rotated carton: top-down 1.8 mm while descending and 2.1 mm on the clamp, front 2.3 mm (4a_approach fails the 2 mm limit) and 3.3 mm on the clamp.
  Cause: the pad row presses the spine at 3 deg yaw; the spine is only 16 mm wide and the neighbours 1 mm away. A real system must align the pad row to the measured spine within ~1.5 deg (not measured how large the tolerance is: 3 deg fails, 0 deg and a 2 mm offset pass).
* Found and fixed while building the test (not failures of the final runs): the shelf-lip seam above; S6 top-down needs a 150 mm pull; the 5 mm press does not compress a loose carton's buffers (see above).
Robustness variants on T5 (all other variants pass, the neighbours move <= 0.04 mm):

| variant (on T5, Shelf_04 angiolite mid) | top-down | front | what happened |
|---|---|---|---|
| vac35 | OK | OK | suction at -35 kPa (pull-off limit 12.2 N total): the 0.13 kg carton needs < 1 N |
| squeeze | OK | OK | row squeezed to 0 mm gaps (all cartons of the group moved up to 28 mm toward the target): neighbours moved 0.02 / 0.04 mm |
| lat2 | OK | OK | suction tool 2 mm off the carton centre line: no effect |
| tilt3 | FAIL | FAIL | suction tool yawed 3 deg to the spine: the pads rotate the carton 1.2 deg while pulling and push the neighbour 4.4 mm (limit 3 mm): FAIL in both modes  |

**Top-down vs front, recommendation.** Both pass the same 9 targets with sub-millimetre slip. Top-down: the carton weight is a pure shear load on the pad faces and the clamp moves the carton 0.004 .. 0.015 mm; but the yoke / fingers pass only 9.3 mm below the front edge of the board above at 100 mm pull
(a +-5 mm carton position error eats half of it), the top shelf needs a 150 mm pull, and the suction tool has to step aside first (extra 2.4 s and a lateral path). Front: 44 mm to the neighbours, 86 .. 125 mm to the shelf boards at every level (including the 230 mm S2) with the standard 100 mm pull and a simpler suction retreat,
but the grasp is limited to the first 44 mm of the spine end (stop yoke), the carton then hangs horizontally from the pads (its weight is a ~0.1 N m moment on the pad faces: held with 0.02 .. 0.2 mm slip in simulation at friction 0.5, not verified on real pads) and the clamp moves the carton 0.02 .. 0.25 mm. **Recommendation: use the top-down grasp
(as designed, force path purely in shear) with a pull-out of 150 mm at every level, which raises the gripper-to-shelf-above clearance to ~59 mm (verified on S6 only; the other levels were run at 100 mm), and use the front grasp where a vertical descent is not possible (clearance above uncertain, S6 with a 100 mm pull) or as fallback.** Neither mode helps for a carton at a side wall (E2).

**Videos** (`tests/test_cabinet_grasp.py --record`; headless RTX render product at 1280x720 30 fps = real time (1 frame per 8 physics steps), burned-in caption with mode, level, box, vacuum, variant and current step; H.264 yuv420p CRF 23 piped to `/usr/bin/ffmpeg`, no frame files): one 3/4 view from the front-left (tool, carton, neighbours, shelf above) and one close-up side view per run:
`tests/results/videos/<mode>_<target>[_<variant>]_{3q,side}.mp4` (56 files, 1 .. 8 MB each, ~250 MB in total: move them to the external drive before committing), 20 stills (the 'extracted' moment) in `tests/results/stills/`, and the highlight reel (top-down T2, front T2, top-down S6 T7, front thin T4 + the 4 failures; 37 MB) at
`/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/highlights.mp4`. Physics is identical with and without recording: a recorded and a non-recorded T5 run agree to <= 0.001 mm in every metric (rendering is driven with `delta_time = 0`, the timeline is not stepped by the renderer);
the numbers above are from the recorded runs. Gotcha: `isaacsim.core.prims.RigidPrim` on the instanceable cartons makes them invisible in the render (it writes default poses to the instance proxies): the test reads poses from the raw `omni.physics.tensors` rigid-body view.



## Cabinet grasp with the tools on the D200 robot (`tools/parking_study.py`, `tests/test_cabinet_grasp_robot.py`)
Coordinator request: redo the cabinet grasp with the tools mounted on the real D200 dual-arm robot (arms do the motion) and find where the AMR should park. The AMR stays a static placeholder; the parking result is only a recommendation.

**Parking study** (`tests/results/amr_parking.md/.json`, plain-python tools in `tools/`, PyKDL + urdf_parser_py, same URDF as the MoveIt package; world = d200_base_footprint):
`robot_kin.py` (FK/IK with joint limits, 37 seeds, point-sample collision model of the URDF + cabinet), `reach_maps.py` (IK reach maps of the three tool orientations, 20 mm grid, `data/reach_maps*.npz`), `parking.py` (the grasp sequences as Cartesian segments + joint transits; continuity of IK branches along 5 mm steps, flip threshold 0.12 rad; clearances: arm/cabinet/stock/AMR/stand/other arm >= 15 mm, tool vs cabinet >= 15 mm, tool vs neighbour cartons >= 3 mm (CAD rule), tool vs other tool >= 10 mm), `parking_study.py` (map screening -> full evaluation of the best candidates -> centring of lift, lateral offset and d in their contiguous feasible intervals -> d window at 10 mm -> 8 corners of the +-20 mm x/y, +-1.5 deg yaw parking error box), `amr_parking_report.py` (merge + common standoff).
Variables: d (AMR front bumper to cabinet front face), lateral offset of the AMR centre to the target carton, lift_joint, slider_left/right (within URDF limits); lift and sliders fixed during one pick.
Results: **front mode** feasible for T3..T8 and E1 at d 410..440 mm (windows 350..480 / 360..520 mm, common standoff 420 mm, +-20 mm/+-1.5 deg box passes at every best pose); T1 (shelf 1) and T2 (shelf 2) are below the reachable height of a horizontal suction tool even at lift -0.6235; E2 (carton next to the left wall) fails on tool-wall clearance, also with the suction tool turned 180 deg about the pad axis (`--yaw180`, extra variant: does NOT solve it, the 106 mm wide tool body is the limit on both sides). **Top-down** has no feasible parking pose: with the real arm, links above the gripper hit the stock on the upper shelf (shelves 1..5) or the gripper/arm touch the TopClosedPanel (shelf 6, even with 150 mm pull); pose-level diagnostic in `amr_parking_topdown_diag.json` (0 of 1500 sampled poses per target). MoveIt confirmation (mock, plan-only, domain 172, `tools/moveit_confirm.py`): T3 (lowest feasible) and T8 (highest): all transits planned by OMPL, every straight segment `compute_cartesian_path` fraction 1.0.
Limitations: free-space transits are checked as joint interpolation / grouped orderings / random via points in my own model (MoveIt plans them in the confirmation); sampled-point collision model (not exact meshes); the target carton is not in the MoveIt scene.

**Isaac replay** (`tools/export_plan.py` -> `data/plans/<mode>_<target>.json`, `tests/test_cabinet_grasp_robot.py`, scene `usd/cabinet_robot_test_scene.usda`): the cabinet stays at the origin, the robot (`/World/D200`, referenced `d200_dual_arm_medical.usda`) is moved to the parking pose (xform + `world_fix` joint); the 12 arm joints are driven with their USD joint position drives (stiffness 15000, damping 1500, maxForce 330 per USD degree unit, NOT changed) along the time-parametrised joint trajectories (0.785 rad/s, 5.24 rad/s^2 from the medicine-grasp limits; scale 1.0/0.5 on straight segments, 0.5/0.5 on free-space transits; tool speeds 20 mm/s approach, 50 mm/s pull); fingers and Surface Gripper with the same profiles, `SuctionLimitMonitor` active. Same metrics as the carrier test plus arm joint tracking error, joint efforts, robot-cabinet contacts, arm clearance (collision primitives of the arm links).
Run: `python3 tools/parking_study.py --modes front --workers 3 --out amr_parking_front.json; python3 tools/export_plan.py --parking tests/results/amr_parking_front.json --mode front; nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp_robot.py --mode front --targets T3,T4,T5,T6,T7,T8,E1 --record; python3 tests/summarize_robot.py; python3 tests/make_highlight_reel_robot.py`.
Results: `tests/results/cabinet_grasp_robot_front.json`, `cabinet_grasp_robot_summary.md`. Videos on the external drive (`tests/results/videos/README.md`). Physics is bit-identical with and without recording (T5 compared).

**Watching a run live** (`--gui`, also on `tests/test_closest_attempt.py`): `nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp_robot.py --mode front --targets T5 --gui`
opens the Isaac Sim window at the 3/4 video view and plays the run at real time (`--gui-speed 0.5` = half speed; it cannot be faster than the physics), with the step captions in the terminal.
Each target's final pose is shown for `--gui-hold` s (default 3); the last one stays until the window is closed (`--gui-close` exits instead). Orbit with Alt + left mouse; do not press Stop/Play.
Physics is unchanged: the window is redrawn with `SimulationContext.render()` between `sim.step(render=False)` calls (`GuiViewer` in `tests/isaac_common.py`); T5 front and T1 closest
gave results identical to the headless runs (2026-10-20). Results go to `*_gui.json` so the headless files are not overwritten. Not combinable with `--record`. Details: `REPRODUCE.md`.

## Legacy: stand-only rig `usd/dual_arm_rig_medical.usda` (superseded by the D200 rig)
References the operator's `dual_arm_rig_v2.usda` (`/media/farhan/My Passport/igus_nvidia/scratch_usd/dual_arm_stand_20261006/dual_arm_stand_isaac_usd/`, copy of `rebel_dual_arm_stand_description/isaac/`, **read only**; override with env `DUAL_ARM_RIG_USD` and rebuild)
and overrides it (15 KB layer in the repo, the 13 MB stays on the drive; the reference path is absolute, so the file only opens where that path exists):
* deactivated: `xeg32_base_link` (carriages + camera mount below it) and the 4 `arm_*_xeg32_*` joints of both arms, the compal `CardboardBox` and `table`.
* attached: suction tool v5 under `arm_left/.../link6`, LEHF20K2-48 gripper under `arm_right/.../link6`, each with a fixed joint `Joints/arm_{left,right}_medical_tool_mount` (identity frames, tool frame = link6), tool root articulation root removed, world anchor deactivated.
* home state: drive targets + `PhysicsJointStateAPI` for lift -0.2, `slider_left_joint` -0.005973, `slider_right_joint` +0.005973 (arm spacing 0.448 m), arm joints 0 (upright).
* **Fix needed for Isaac 5.1 (found while testing):** the original rig does **not** load its arms in Isaac Sim 5.1: it authors the arm chain (`base_link > motor1 > link1 > ... > link6`, XEG bodies, now tools) as rigid bodies **nested** in rigid bodies;
  5.1 reports "Rigid Body ... missing xformstack reset when child of another enabled rigid body" and "CreateJoint - no bodies defined" for all 12 arm joints, and the articulation only has the lift and the two sliders (3 DOF).
  The medical file gives every nested body `!resetXformStack!` + its composed world transform (29 bodies, customData `nested_bodies_flattened`), which fixes it (20 DOF). The authors tested on Isaac 6.1.
* the cable-holder bodies `ch_v4_*` are authored 0.2 m below their fixed joints (the original warns "disjointed body transforms"); they are lifted by 0.2 m in the override.
* shelf: `/World/medical_shelf` (= `shelf_cartons.usda`, rotated 180 deg about Z so W +Y = world -Y, the arms' forward direction) with the target spine 0.40 m in front of the left arm base axis (world x 0.2282, y -0.5793)
  at shoulder height (z 1.158), on a plinth. **Reachability is only a rough check**: wrist centre ~0.24 m horizontally from the shoulder axis with the tool horizontal, well inside the ~0.5 m ReBeL reach; no IK was run and the arms are at home (tools pointing up).
* the rig's other content (stand, lift, sliders, arm meshes/inertias of compal's `rebel_d00617809`) is unchanged. **The ROS 2 bridge script (`dual_arm_rig_ros2_bridge.py`, it segfaulted headless for its authors) and the MoveIt config in the main repo
  (`rebel_dual_arm_stand_moveit_config`, URDF `rebel_cell_tool.xacro`) still model the knife / XEG-32 cutting tooling, not these tools; they were not edited and were not run here (no ROS bridge was enabled).**
  Bridge joint lists must be extended by `pad_{1,2,3}_buffer_joint` and `finger_{left,right}_joint` before these tools can be commanded from ROS; the URDF/MoveIt side has no model of them at all.

## Structural verification (`data/verify.json`, 146/146 pass, pxr only)
Prims valid; defaultPrim / up axis / units; visual and collision mesh counts vs URDF and manifests; USD frames equal the URDF frames (< 0.1 mm, < 0.01 deg); pad frames on the pad lip centres, +Z out of the lip; every collision mesh convexHull + guide;
combined mass / CoG / inertia of the suction bodies and per-body values of the gripper vs the URDF; buffer joint axis, limits, anchors, spring numbers; Surface Gripper prim, force-limit formula, attachment points; gripper joint axes, limits, drives, mimic,
no maxJointVelocity; handoff.py constants; scene reference resolution, carton placement, tool poses; presence of the pick scene, shelf, ReBeL variants and the rig overrides. The matplotlib previews are not Isaac renders; Isaac captures are in `tests/results/*.png`.

## Known limits / not done
* Cabinet grasp: kinematic carriers, no arm reach / IK / joint limits and no real camera-based carton localisation; carton and friction values (0.43 pad/carton, shelf 0.30/0.25, masses 0.08 .. 0.13 kg) are estimates; E2 (wall) and 3 deg yaw fail as described; the pad zone is limited to the first 44 mm of the spine end in the front mode.
* Surface Gripper limits are not exact Newtons (see above); the pump build-up is a test-script delay, not part of the extension; shear/coaxial limits only matter once the carton is pulled hard.
* Robot Assembler GUI not clicked through; the rig arms were not moved to the shelf (no IK); no ROS involvement at all.
* Carton colliders are `boundingCube` of the carton mesh; cartons are single rigid bodies with estimated masses of 0.08 to 0.11 kg; real cartons deform / bulge (grip width tolerance +-0.5 mm).
* `right_gripper_tool_independent.usda` is the only gripper file used by the static `shelf_handoff_scene.usda`.
