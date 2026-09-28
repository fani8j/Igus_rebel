# One tracked frame from side_tape_perception. Mirrors side_tape/types.py Observation.
# header.stamp = IMAGE capture stamp (not processing time).
std_msgs/Header header
uint64 frame_seq
string state            # SEARCH | LOCKED | COASTING | LOST
bool has_line
float64[3] line         # (a, b, c) oriented unit-normal line, de-rotated full-image px
int8 polarity           # +1 / -1, 0 when unknown
bool has_e_lat
float64 e_lat_mm        # tool lateral position minus target (types.py)
bool has_e_yaw
float64 e_yaw_deg
float64 conf
int32 n_inliers
float64 rms_px
float64 proc_ms
string[] reasons
