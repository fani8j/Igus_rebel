# The controller's own Cartesian pose of the robot, from one CRI STATUS message
# (POSCARTROBOT), published together with the joint angles of the SAME STATUS so the pair is exact.
# For checking the URDF kinematics against the controller's kinematic model.
#
# header.stamp: acquisition (receive) time of the STATUS sample, system clock; the same stamp
# as robot_joint_states. Never "now".
std_msgs/Header header

# POSCARTROBOT x y z a b c, exactly as reported by the controller, unconverted.
# Units as reported by the controller, believed mm and deg, A/B/C Euler convention unverified.
# What the pose refers to (flange or tool, base frame) depends on the TCP and base set in iRC.
# Precision: the driver parses the values as 32-bit floats.
float64[6] pos_cart_robot_raw

# POSJOINTCURRENT joints 1..6 of the same STATUS, converted to radians (driver joint order).
float64[6] joint_position_rad

# STATUS MODE: "joint", "cartbase", "carttool", "platform", "fsm" or "unknown".
string mode

# STATUS KINSTATE (0 = no error, 99 = motion not allowed, ...); 255 = a value the driver does
# not know.
uint8 kinstate
