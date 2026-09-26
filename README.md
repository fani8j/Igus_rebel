# igus ReBeL ROS2 #

## Summary ##

* Connects to an igus ReBeL over an Ethernet connection

## Compatibility ##

Tested combinations:

- ROS 2 Jazzy on Ubuntu 24.04
- ROS 2 Humble on Ubuntu 22.04

## Installation: Not using docker ##

The packages are built from source in a colcon workspace. Run the commands for
your installed ROS 2 distribution from the workspace root.

### ROS 2 Humble on Ubuntu 22.04

```bash
source /opt/ros/humble/setup.bash
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install
source install/setup.bash
```

### ROS 2 Jazzy on Ubuntu 24.04

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install
source install/setup.bash
```

The `rosdep install` step installs MoveIt Task Constructor and the other
distribution-specific dependencies required by `igus_rebel_moveit_config`.

The ros node expects to reach the robot at the IP and port `192.168.50.133:3920`, this IP is fixed in **src/igus_rebel/include/igus_rebel/Rebel.hpp**, line 88. You can change the IP here, make sure to build the code by `colcon build` after changes.

## Installation: Using docker ##

Docker is optional; the supported Ubuntu and ROS 2 combinations above can run natively.

To build docker image:

```bash
sudo docker compose build
```

## Docker usage ##

Run the container:

```bash
sudo docker compose up
```

To entry docker container environment, use the following command on each new terminal:

```bash
sudo docker exec -it ros2_jazzy_rebel_dev bash
```

You can freely modify the code inside src folder in host computer and build by `colcon build` inside docker container.

## Usage: on real ReBel robot

Launch hardware interface, controller:
```bash
ros2 launch igus_rebel rebel.launch.py
```

Launch moveit motionn planner and teleoperation mode:

```bash
ros2 launch igus_rebel_moveit_config igus_rebel_motion_planner.launch.py use_gui:=true
```

## Usage: on simulation

To simulate the robot in Gazebo and control the simulated robot with MoveIt run:

```bash
ros2 launch igus_rebel_moveit_config igus_rebel_simulated.launch.py
```

Note: the Gazebo model (`igus_rebel_robot2.urdf.xacro`) does not include the XEG-32, so
`xeg32_tool_tip` has no TF in simulation.

## Calibrated TCP and visual comparison

The XEG wrapper defines `link6 -> igus_tool0 -> xeg32_tool_tip` independently
of gripper visuals. The tool frame is +52.7 mm along link6 Z and rotated pi
about Z; the TCP is `[5.937, 94.121, 130.699]` mm from that tool frame (touch-corrected
2026-09-24; see the comment in `igus_rebel_xeg32_wrapper.urdf.xacro`).
See `src/igus_rebel_moveit_config/XEG32_MOVEIT_CHANGES.md` for frame details.

The rough knife model remains removed. `xeg32_tool_tip` remains the exact
kinematic TCP and green 6 mm visual marker, but it has no collision geometry:
a collisionless marker itself cannot collide, and the attached cutter body
remains absent from collision checking. Do not treat planning success as
physical cutter-clearance validation.

For an isolated mock comparison, source ROS and this workspace in both terminals
and set `export ROS_DOMAIN_ID=91` in both. Start the robot/TF publisher first:

```bash
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
```

Then start MoveIt/RViz in the second terminal:

```bash
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false \
  run_box_h_cut_executor:=false
```

The MoveIt launch alone does not publish the arm TF tree. Add an RViz TF display
to inspect `igus_tool0` and `xeg32_tool_tip`. Restart both launches after model changes.

## Collision geometry without a gripper-mounted camera

The camera and its pole are fixed outside the robot; no camera or holder is
mounted on the XEG-32 gripper. The gripper URDF therefore contains no camera or
holder geometry. The XEG-32 base and carriage meshes remain modeled and
collision-checked, alongside robot self-collision and the snapshot-derived
`detected_carton`. `xeg32_tool_tip` remains the kinematic TCP and green visual
marker with no collision geometry of its own.

The fixed external camera pole remains a required planning-scene obstacle.
`move_group.launch.py` starts `camera_pole_collision.py`, which derives the
upright 0.15 x 0.15 x 1.0 m `camera_pole` box from the external
`camera_link` and `camera_color_optical_frame` transforms. H-cut planning waits
until that collision object is available. A saved perception snapshot may be
used without acquiring a new image, but the fixed camera TF must still be
available so the pole is collision-checked.

In RViz enable MotionPlanning scene geometry and confirm both `detected_carton`
and `camera_pole` before planning. Discard stored trajectories and replan after
either obstacle changes. These are MoveIt collision objects, not hardware stops
and not protection for vendor/manual jogging.

## Calibration verification (plan-only by default)

The calibration verification launch starts mock or hardware control, MoveIt,
ArUco verification, and the `calibration_verification` executor. It never moves
the arm from a detection alone: every step is an explicit service call, and
`execution_allowed` defaults to `false`.

Mock launch used for validation:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=92 ros2 launch igus_rebel_moveit_config calibration_verification.launch.py \
  hardware_protocol:=mock_hardware \
  start_control:=true \
  use_gui:=true \
  execution_allowed:=false
```

The supplied motion configuration enables measured obstacles (`preview_only: false`)
with `preapproach_only: false` and `retract_after_approach: false`. Plans transit
to 90 mm above the reference, approach to 10 mm planned clearance, and end there.
There is no automatic retract/home move. `execution_allowed` remains false by default.
The 100 x 100 x 2 mm platform and 50 mm cube are marker-relative; the below-base
4 x 4 x 2 m keep-out box is in `planning_frame`, with its top at Z=0.
Setting `preview_only: true` omits these fixture objects and blocks execution.
The supplied orientation is `[pi, 0, 0]` relative to the marker: TCP +Z points
toward the board. This is a preview alignment, not a physical calibration result.
The verification executor declares the shared launch parameter overrides so
MoveIt uses the same named OMPL planner settings as the default launch.
Capture publishes labeled center/point_1/point_2/point_3 markers on
`/calibration_verification/markers` (RViz **Verification targets**).
`reference_origin_xyz_m` selects a measurement origin relative to the detected
marker center, expressed in marker axes (currently `[-0.004286, -0.004286, 0]` m).
The center target, captured-reference axes, and all point XY offsets use this
origin. Detector pose and platform/cube geometry remain marker-centered;
`table_box_center_xyz_m` is in `planning_frame`, not the marker frame.
Do not include the origin shift again in point offsets measured from this origin.
Physical execution requires confirming the actual start state and inspected plan.
Stale poses/states and plan mismatches still fail closed. The TCP marker has no
collision geometry, so there is no tip collision coverage; the attached cutter
body is likewise absent from collision checking.

In each service terminal, source the same workspace and use the same
`ROS_DOMAIN_ID` as the launch.

Service sequence:

```bash
ros2 service call /calibration_verification/capture std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_center std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_1 std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_2 std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_point_3 std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/plan_cube std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/clear_plan std_srvs/srv/Trigger {}
ros2 service call /calibration_verification/stop std_srvs/srv/Trigger {}
```

To opt in to manual execution, launch with `execution_allowed:=true` first, then
capture, plan, and inspect the retained solution in RViz. Nothing moves until:

```bash
ros2 service call /calibration_verification/execute_plan std_srvs/srv/Trigger {}
```

`clear_plan` discards the retained plan without motion; it does not stop active
execution (use `stop`, with the physical emergency stop available). Execution
consumes the plan, including failed validation attempts. Replan before retrying.
Plans expire after 60 seconds, and restarting the launch loses the retained plan.
The launch merges the execution opt-in into the motion parameters before passing
them to the node, avoiding node-specific YAML overriding the explicit launch flag.

The mock smoke in this workspace used `motion_config:=/tmp/calibration_verification_fixture.yaml`
with poses generated by `/tmp/calibration_fixture.py`; it validated
`plan_center`, `plan_point_1`, `plan_point_2`, `plan_point_3`, `plan_cube`, and
mock `execute_plan` on `ROS_DOMAIN_ID=92`. That fixture is synthetic and is not a
source of real board, cube, marker, or TCP calibration values.

## Staged per-cut correction workflow (plan-only by default)

The H-cut executor (`box_h_cut_executor`) plans the canonical snapshot-relative
cuts and additionally offers a staged teach-and-correct session for one cut at a
time. `cut_debug_cut` selects `left` (`back_middle_left -> front_middle_left`),
`middle` (`seam_start -> seam_end`), or `right`
(`back_middle_right -> front_middle_right`). Nothing moves until the operator
explicitly calls `execute_planned_h_cut`, and every staged dispatch requires
**both** `execution_enabled` and `debug_motion_execution_enabled` (both default
`false`). No stage auto-executes; a stale retained plan never survives a
teleop/save step.

Service sequence for the selected cut (same `ROS_DOMAIN_ID` as the launch):

```bash
ros2 service call /box_h_cut_executor/start_cut_debug_session std_srvs/srv/Trigger {}
ros2 service call /box_h_cut_executor/execute_planned_h_cut std_srvs/srv/Trigger {}
# teleop-correct the TCP at the start point, then:
ros2 service call /box_h_cut_executor/save_cut_debug_point std_srvs/srv/Trigger {}
# teleoperate the TCP to the end point separately, then:
ros2 service call /box_h_cut_executor/save_cut_debug_point std_srvs/srv/Trigger {}
```

The first stage plans the configured start target. If a correction record is already
active, that target includes it so an additional teaching pass begins from the current
best pose. Saved nominal/observed deltas always use the uncorrected geometry baseline,
so the resulting record is standalone rather than an incremental residual. Each
save_cut_debug_point call captures the full observed TCP pose from live joint-state FK
and persists it immediately without requiring any endpoint or lift plan: the first save
records the teleop-corrected start (status `start_saved`) and the second records the
separately teleoperated endpoint (status `complete`). A save retains no plan (any
retained plan is cleared) and never dispatches motion. The operator teleoperates the
endpoint separately and calls the same save service; the final record is complete after
that endpoint save. Start and end observations are therefore persisted independently.
Only `complete` records are loadable as corrections; `start_saved` is an intermediate.
Left approaches and retracts 10 mm laterally toward the detected carton exterior,
without shifting either cutting endpoint. Middle and right retain surface-normal
hover/retract. Each save persists a schema-versioned record (`box_h_cut_correction/v1`,
`<cut>_<snapshot_id>.json`, atomic temp-file plus rename) with the snapshot ID,
cut name, planning frame, nominal/observed full start/end poses, cut-local
translation deltas, and cut-local orientation delta quaternions.
(`cut_debug_correction_enabled`, default `false`, plus per-cut absolute paths
`cut_debug_left/middle/right_correction_file`). Combined plans apply each set
slot to its segment; `t_cut` and `t_cut_handoff` contain only left and middle segments,
so a set right correction is loaded, validated, and left unused rather than failing
closed; individual plans apply only their cut's slot. Empty means no correction for that cut; any malformed or mismatched nonempty record, or enabling with all slots empty, fails closed. Records
are cut-local and rotate onto the newly detected cut. Orientation preserves the full
taught start/end TCP quaternion: the planner maps each saved quaternion from the
recorded cut basis into the current detected cut basis. Thus the taught blade angle is
unchanged at the capture pose and follows later carton rotation without being rebuilt
from a nominal knife frame. A taught left pose may validly use tool local X or
tool local Z as its penetration axis; at least one of the two must point into
the surface. Middle and right use tool local Z only. Records are validated for
schema, matching cut slot and frame/tool, finite orthonormal cut bases, finite
translation values, and unit-length orientation quaternions. Application fails
closed unless the mapped penetration axis (left: X or Z; middle/right: Z) points
into the carton surface. With opt-in off, plans remain purely geometry-derived
even when paths are set.
For learned LEFT poses only (with `cut_debug_correction_enabled` and a loaded
LEFT correction file), when `left_adaptive_orientation_enabled` is true,
the planner tests TCP-local Y (blade-length/seam axis) rolls of the mapped
quaternion as `q_candidate = q_mapped * Qy(theta)`. TCP position and blade Y
direction are preserved; only the X/Z penetration roll varies. Candidates run
in deterministic smallest-magnitude order 0, +5, -5, ... through
`+/-left_adaptive_orientation_max_deg` in `left_adaptive_orientation_step_deg`
steps (defaults 45.0/5.0). Every candidate must pass the existing left X-or-Z
penetration gate plus IK and collision feasibility at the final target poses;
the probe is a bounded prefilter that only screens out unreachable or colliding
mapped poses under the current carton/pole scene constraints and does not
guarantee a full plan. Hover/approach candidates must be fully collision-free;
cut-target candidates honor the scoped tool/carton contact allowance while
every other contact still fails. The search checks the same detected carton and
external camera pole as the MTC task, including robot self-collision; collision-blind
IK can otherwise pick a roll whose approach goal is invalid. MTC, Cartesian, and
collision checking remain the final authority and may still reject a selected
orientation; both execution gates stay mandatory. If no candidate passes,
planning fails closed with candidate diagnostics. Middle and right corrections
keep their rigid mapped orientation. The saved September 2026 left record predates
the current TCP: a mock replay may opt into it, but a physical cut requires a
freshly taught correction on the current tool before enabling this opt-in.

## H-cut free-space transit consistency (plan-only by default)

Free-space transits (initial approach and inter-segment finish-to-approach
moves) use dedicated `transit_velocity_scaling` /
`transit_acceleration_scaling` (default 0.15, raised from 0.05 with operator approval on 2026-09-26; finite values in `(0, 1]` only,
out-of-range fails closed). Contact descend/traverse/retract strokes keep the
0.05 `velocity_scaling` / `acceleration_scaling`, and
`pre_cut_stabilization_duration_s` stays 1.0 s, so cutting speed is unchanged. Each free-space transit tries a direct collision-checked
joint interpolation first via MultiPlanner ordering ([joint interpolation,
existing transit]), falling back to the original transit solver. Full cut
poses, `min_fraction=1.0` acceptance, the scoped tool/carton contact policy,
and the fixed external `camera_pole` requirement are unchanged. A free-space
solution whose joint6 travel exceeds pi rad, summed over every consecutive
waypoint of its free-space transit subtrajectories, is rejected before the
plan is retained, published, or executed; there is no silent fallback after
this check. The pre-TCP-change taught left correction on the saved carton was
rejected at 4.806 rad of middle-transition joint6 travel in isolated mock
planning; re-teach it on the current TCP before physical use.
Faster transit shortens only non-contact portions and is no
guarantee of a fast plan or a physically safe stale TCP: the correction
opt-in (`cut_debug_correction_enabled`, default `false`) and both execution
gates (`execution_enabled`, `debug_motion_execution_enabled`, both default
`false`) are unchanged, and no physical dispatch is recommended.

## T-cut handoff (first arm only, plan-only by default)

`cut_pattern: t_cut_handoff` plans first-arm-only work: the full left cut plus
an executable bounded middle-seam prefix ending at the declared handoff endpoint
(`interpolate(seam_start, seam_end, <resolved fraction>)`, included in
the published target poses for external second-arm integration). It does
NOT lower any `min_fraction` acceptance: the planned path is accepted only when
it fully reaches that explicit handoff endpoint under `min_fraction=1.0`. After
fully reaching that endpoint, the retained first-arm plan ends at the cut-level
handoff pose and deliberately skips the hover retract
(`finish pose above middle seam handoff`) and its following restore-collision
stage, so the scoped tool/carton contact allowance stays active at the terminal
pose. The retained middle traverse is therefore an executable first-arm prefix,
not the complete seam; no second arm is implemented. Execution still requires the
normal gates.
`t_cut_middle_handoff_fraction` selects the stop fraction along the middle seam:
`0.0` (default) selects automatic mode: bounded bisection over `[0.05, 0.95]`
finds the largest fully reachable declared prefix with `<=0.01` fraction tolerance,
keeping `min_fraction=1.0` for every candidate. A nonzero value in `[0.05, 0.95]`
is a manual override stop fraction; out-of-range nonzero values fail closed.
Full `t_cut` stays the full left cut plus the full middle seam and is unaffected,
and full `t_cut`/`combined` continue to retract above the cut and restore
collision checking. The left-to-middle transition (combined, `t_cut`,
`t_cut_handoff`) is: left lateral withdrawal to the finish pose outside the left
cut, then a surface-normal lift that holds the left orientation, then restore of
knife-tip collision checking, then free-space transit/rotation to the middle
approach. No current-plan serialization is implemented: the retained plan lives
only in the running executor until restart. No second-arm behavior changes.
`t_cut_handoff` loads the same left and middle correction slots as `t_cut`; a set right record loads but stays unused because the plan has no right segment.
Middle translation end deltas are fraction-interpolated and the taught middle end
orientation is slerp-interpolated by the resolved fraction for the
bounded prefix; the middle tool-Z penetration gate is revalidated unchanged.

## RobotViewer

At `https://viewer.robotsfan.com/`, choose **Load Files** and select:

- `src/igus_rebel_description/urdf/igus_rebel2_robotviewer.urdf`
- all twelve `.dae` files in
  `src/igus_rebel_description/meshes/rebel_d00617809/`

The browser cannot read mesh files from the local ROS installation when only
the URDF is selected, so the URDF and its DAE files must be loaded together.

## Isaac Sim

Generate the corrected six-axis URDF before importing it into Isaac Sim:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
xacro src/igus_rebel_description/urdf/igus_rebel_isaac_wrapper.urdf.xacro \
  -o /tmp/igus_rebel_official.urdf
```

Import `/tmp/igus_rebel_official.urdf` with a fixed base, articulation creation,
inertias, and collision geometry enabled. Keep fixed joints separate on the
first import so the ROS link hierarchy remains visible. The twelve visual meshes
are tessellated from the igus `D00617809` STEP assembly and preserve its native
CAD material regions. Collisions, inertias, and joint frames remain those of
the version-two robot.
Regenerate the USD after changes instead of using the pre-generated USD files.

## Set digital outputs

The ReBeL's digital outputs can be set with a call to the service `/set_digital_output`. 

The service input is a `DigitalOutput` message, which is defined as
```
int8 output
bool is_on
```

- `output` is the index of the output whose state should be set.
- `is_on` is the state to which the output should be set. `True` means on, `False` means off.

The service output is defined as
```
bool success
string message
```
`success` is always True, `message` is always empty.

## Teleoperation

You can use a teleop keyboard program or a gamepad to control the arm manually. The gamepad node (`servo_gamepad_input`, mapping in `config/gamepad.yaml`) starts with the MoveIt launch when `run_gamepad:=true` is passed (default `false`).

You can also run the following command to start teleop_keyboard:

```bash
ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard
```

Type "w" and "t" to control the TCP in world frame. Then use arrow keys and ".", ";" to move the TCP forth/back/left/right/down/up

The Cartesian keyboard increment defaults to 1 mm per keypress. Change it at runtime with:

```bash
ros2 service call /servo_keyboard_input/set_cartesian_increment \
  igus_rebel_msgs/srv/SetCartesianIncrement "{increment_meters: 0.002}"
```

The accepted range is greater than 0 m through 0.01 m. The service affects Cartesian
arrow, period, and semicolon commands; joint-jog speed is unchanged.


## Project and OMP session RAG

The dependency-free local index combines current source/configuration files,
curated verified facts and unresolved items, and textual user/assistant turns
from OMP sessions whose recorded working directory matches this repository.
Build it atomically with:

```bash
python3 tools/session_rag/session_rag.py build
```

Retrieve evidence before debugging or editing:

```bash
python3 tools/session_rag/session_rag.py query \
  "Why does the CRI DOUT parser fail on a 64-bit mask?" --format prompt
```

Inspect session-retention recommendations or graph relationships:

```bash
python3 tools/session_rag/session_rag.py sessions
python3 tools/session_rag/session_rag.py graph topic:cri_network
```

The SQLite FTS5 index and JSON graph map are generated under
`workspace_scan_artifacts/`. Generated evidence includes its source and line or
message locator; session evidence can be stale, so verify current source before
changing code.

An optional LightRAG REST adapter uses `POST /documents/texts` and `POST /query`.
It accepts loopback endpoints by default and reads `LIGHTRAG_URL` and
`LIGHTRAG_API_KEY` from the environment; it never stores credentials in the
index. Validate an ingestion batch without network access using:

```bash
python3 tools/session_rag/session_rag.py lightrag-ingest --dry-run
```

LightRAG still requires a separately configured LLM and embedding provider.
Use `tools/session_rag/lightrag.env.example` as the client/server variable
template. Do not expose an unauthenticated LightRAG server beyond loopback.