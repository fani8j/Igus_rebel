# Coordinator status, published at a low rate and on every state change (design section 6).
std_msgs/Header header

uint8 STATE_IDLE = 0
uint8 STATE_PLANNING = 1
uint8 STATE_PLANNED = 2
uint8 STATE_WAITING_BARRIER = 3
uint8 STATE_EXECUTING = 4
uint8 STATE_STOPPED = 5
uint8 state

# Current / next phase: "A", "B" or "" when none.
string phase

# Retained plan identity; 0 when none.
uint64 plan_seq
bool plan_retained
bool phase_validated

# Cross-arm distance on the dual model from /joint_states (NaN while unknown), and the closest pair.
float64 min_cross_distance_m
string closest_link_a
string closest_link_b

# Seconds since the last /joint_states message (NaN while none was received).
float64 joint_states_age_s

# Why the last STOP happened (empty if none) and whether it is latched until ~/rearm.
string stop_reason
bool stop_latched

# Barrier bookkeeping before a phase starts.
bool barrier_ok
string barrier_message
