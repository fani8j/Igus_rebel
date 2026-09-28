# Published by side_tape_controller every control tick (throttled).
builtin_interfaces/Time stamp
string phase            # IDLE | ALIGN | READY | CUT | DONE | HOLD | ABORT
string[] reasons
bool dry_run            # true: twists are NOT sent to Servo
bool arm_owned
bool calibrated
string servo_status     # OK | WARN | HALT | UNKNOWN
bool has_e_lat
float64 e_lat_mm
float64 along_mm
float64 lat_dev_mm
float64 depth_dev_mm
float64 v_cut_mps       # last commanded, cut frame
float64 v_lat_mps
float64 v_depth_mps
