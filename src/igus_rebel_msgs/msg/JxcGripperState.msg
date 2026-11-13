# State of one SMC JXC61 gripper handshake node (rebel_gripper_io, jxc_gripper_node), published
# as ~/state at 10 Hz and immediately on change. All fields come from the driver's digital_io
# topic and the node's own command bookkeeping.

std_msgs/Header header
bool valid           # digital_io is valid and fresh (<= 0.3 s); when false every other flag is meaningless
bool svre            # JXC SVRE output ON (servo is on and ready)
bool seton           # JXC SETON output ON (home/setup finished)
bool inp             # JXC INP output ON (in position / target reached)
bool alarm_ok        # *ALARM input is ON = healthy; false = alarm (or wire break)
bool outputs_uncertain # latched: an output command was not confirmed in digital_io (or failed); only release_all (confirmed all-off) clears it; servo_on/home/move_step are refused
bool busy            # a command (servo_on/home/move_step/servo_off) is running
string last_error    # message of the last failed command; empty after a successful one; "outputs_uncertain: ..." while latched
