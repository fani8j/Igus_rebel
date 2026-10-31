# Health of one igus ReBeL hardware component (the Igus::Rebel ros2_control plugin), published by
# the driver's own node as /<node_name>/hardware_state at 10 Hz and immediately when the latch, a
# pending Move Stop, a joint move or the activation changes.
#
# WHY: while the driver is latched (hardware_error) ros2_control moves the component to
# UNCONFIGURED and stops calling read(). The state interfaces then keep their LAST values and
# joint_state_broadcaster keeps publishing /joint_states with fresh stamps, so /joint_states is
# FROZEN and MUST NOT be trusted while latched=true or activated=false. The ~/robot_joint_states
# topic of the same node keeps tracking the robot (stamped with the STATUS receive time, so a dead
# link shows as an ageing stamp); this message says whether the component is healthy.
#
# Published from the driver node, independent of the lifecycle state and of read()/write().

std_msgs/Header header                   # stamp = time this message was published (ROS clock)
string component                         # ros2_control component name (URDF <ros2_control name=...>)
bool connected                           # CRI socket currently connected
uint32 connection_count                  # number of connections made since on_init (>1 = reconnected)
uint64 status_sequence                   # count of valid STATUS messages received (0 = none yet)
builtin_interfaces/Time status_stamp     # PC receive time (ROS clock) of the latest STATUS; zero if none
float64 status_age_s                     # seconds since the latest STATUS was received; -1 if none yet
bool status_fresh                        # status_age_s <= 0.2 s (the driver's stale tier)
bool latched                             # fail-closed hardware error latch; read() returns ERROR while set
string latch_reason                      # why it latched (first cause); empty when not latched
bool move_stop_pending                   # a CMD Move Stop is wanted and not yet confirmed by the driver
bool move_active                         # a move_joint goal is in progress (sending/moving/stopping)
uint8 drives_disabled_mask               # bit i (0..5) set = STATUS reports 'Motor not enabled' for joint i+1
int32 kinstate                           # kinematics state of the latest STATUS (0 = no error)
bool activated                           # component is ACTIVE (read()/write() are being called)
float64 move_joint_max_velocity_percent      # configured cap on a move_joint goal's velocity_percent, in % (after the 100 % hard-cap clamp); constant
float64 move_joint_max_acceleration_percent  # configured cap on a move_joint goal's acceleration_percent, in % (after the hard-cap clamp); constant
bool cart_jog_active                     # Cartesian jog mode (CRI MotionTypeCartBase) is on and confirmed; the joint vel_cmd path is ignored
string cart_jog_state                    # "off" | "entering" | "active" | "leaving" (leaving = MotionTypeJoint not yet confirmed)
bool joint_jog_active                    # direct joint jog mode (~/joint_jog_enable) is on; the robot stays in MotionTypeJoint and the joint vel_cmd path is ignored
string joint_jog_state                   # "off" | "active" (no mode switch on the controller, so no entering/leaving state)
