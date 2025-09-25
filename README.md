# igus ReBeL ROS2 #

## Summary ##

* Connects to an igus ReBeL over an Ethernet connection

## Compatibility ##

This was tested on ROS2 Jazzy on Ubuntu 24.04.2.

## Installation: Not using docker ##

Currently, the package can only be used by building it from source.

* Pull this repository
* Navigate to the colcon workspace folder and install the dependencies with `rosdep install --from-paths . --ignore-src -r -y`
* Build with `colcon build`

The ros node expects to reach the robot at the IP and port `192.168.3.11:3920`, this IP is fixed in **src/igus_rebel/include/Rebel.hpp**, line 88. You can change the IP here, make sure to build the code by `colcon build` after changes.

## Installation: Using docker ##

In case you are not using Ubuntu 24.04 and ROS2 Jazzy, you can use docker as an alternative way.

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
