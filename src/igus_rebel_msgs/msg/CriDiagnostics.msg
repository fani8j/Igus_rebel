# Per-STATUS diagnostics of the igus ReBeL CRI link (read-only instrumentation), published by the
# driver's own node as ~/cri_diagnostics (sensor-data QoS, best effort) once per valid STATUS, from
# the same thread and with the same stamp as ~/robot_joint_states. Publishing it changes no control
# behaviour.
#
# WHY: to tell whether the robot-side A2 overcurrent is caused by what we send (jog percent
# dithering around zero on a joint that should hold still) or by the drive itself. The motor
# currents and the jog percent that was actually put on the wire are recorded next to each other.
#
# The six jog_percent_sent values and jog_source come from the most recent ALIVEJOG frame (sent
# every ~10 ms by a separate thread), read lock-free. They are therefore the latest frame at the
# time of the STATUS, and may in rare cases be a mix of two adjacent frames.

std_msgs/Header header                   # stamp = PC receive time (system clock) of this STATUS, same as ~/robot_joint_states
uint64 status_sequence                   # count of valid STATUS messages received (same counter as HardwareState.status_sequence)
float64[6] position_deg                  # axes 1-6 actual position in degrees (STATUS POSJOINTCURRENT, first 6 values)
float64[6] setpoint_deg                  # axes 1-6 position set-point in degrees (STATUS POSJOINTSETPOINT, first 6 values)
int32[6] current_raw                     # axes 1-6 raw CURRENTJOINTS value of the STATUS (unit not documented by igus, probably mA; signed as sent)
int32 current_all_raw                    # raw CURRENTALL value of the STATUS (total current, same unknown unit)
float32[6] jog_percent_sent              # jog [%] of the most recent ALIVEJOG frame handed to the socket (what FormatAliveJog printed); all 0 when no frame is being sent
string jog_source                        # who produced that frame: "jtc" (ros2_control write() path), "joint_jog", "cart_jog" (Cartesian slots X Y Z A B C, not joints), "stale" (no fresh write(): watchdog zero), "inhibited" (latch/hold/move/stop/mode guard: zero), "dropped" (a prepared non-zero frame was rejected, zero sent), "none" (ALIVEJOG not being transmitted)
float32 jtc_deadband_percent             # current value of the driver parameter jtc_jog_deadband_percent: |jog| below this is sent as 0 on the jtc path only; 0 = off
