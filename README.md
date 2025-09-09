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

## Usage: Not using docker ##

It is recommended to run the ROS node with the provided launch file, using `ros2 launch igus_rebel rebel.launch.py`

To control the robot with MoveIt, first start the igus_rebel ROS node (`ros2 launch igus_rebel rebel.launch.py`) and then run `ros2 launch igus_rebel_moveit_config igus_rebel_motion_planner.launch.py use_gui:=true`

To simulate the robot in Gazebo and control the simulated robot with MoveIt run `ros2 launch igus_rebel_moveit_config igus_rebel_simulated.launch.py`

## Installation: Using docker ##

In case you are not using Ubuntu 24.04 and ROS2 Jazzy, you can use docker as an alternative way.

To build docker image:

```bash
sudo docker compose build
```

## Usage: Using docker ##

Run the container:

```bash
sudo docker compose up
```

To entry docker container environment, use the following command on each new terminal:

```bash
sudo docker exec -it ros2_jazzy_rebel_dev bash
```

You can freely modify the code inside src folder and build by `colcon build` inside docker container.

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
