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

The ros node expects to reach the robot at the IP and port `192.168.3.11:3920`, this IP is fixed in **src/igus_rebel/include/Rebel.hpp**, line 88. You can change the IP here, make sure to build the code by `colcon build` after changes.

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

## Calibrated TCP and visual comparison

The XEG wrapper defines `link6 -> igus_tool0 -> xeg32_tool_tip` independently
of gripper visuals. The tool frame is +52.7 mm along link6 Z and rotated pi
about Z; the TCP is `[70.456, 49.743, 153.532]` mm from that tool frame.
The last value includes the +137 mm CobotPump offset active during calibration.
See `src/igus_rebel_moveit_config/XEG32_MOVEIT_CHANGES.md` for frame details.

The rough knife model remains removed. The green TCP now has a **6 mm radius
collision sphere**, protecting only the modeled tip, not the attached cutter body.
Do not treat planning success as physical cutter-clearance validation.

For an isolated mock comparison, source ROS and this workspace in both terminals
and set `export ROS_DOMAIN_ID=91` in both. Start the robot/TF publisher first:

```bash
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
```

Then start MoveIt/RViz in the second terminal:

```bash
ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false \
  run_mtc_program:=false run_box_motion_executor:=false run_box_h_cut_executor:=false
```

The MoveIt launch alone does not publish the arm TF tree. Add an RViz TF display
to inspect `igus_tool0` and `xeg32_tool_tip`. Restart both launches after model changes.

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
Stale poses/states and plan mismatches still fail closed. Tip-only collision
coverage does not protect the rest of the attached cutter.

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

You can use a teleop keyboard program or a gamepad to control the arm manually. The gamepad is already available within moveit program, it's plug-and-play.

You can also run the following command to start teleop_keyboard:

```bash
ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard
```

Type "w" and "t" to control the TCP in world frame. Then use arrow keys and ".", ";" to move the TCP forth/back/left/right/down/up
