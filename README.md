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
