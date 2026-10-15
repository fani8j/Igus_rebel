# One leaf stage with a trajectory of the executor's retained MTC solution (design section 4).
# Scene-only stages (no trajectory) are not exported. An unknown stage name with a trajectory makes
# the export fail rather than being guessed.
#
# Classification by stage-name prefix (the executor's plan_export.hpp holds the authoritative table):
#   TRANSIT  "approach pose above/outside", "transition from", "transit to supplied"
#   ALIGN    "align above middle seam start"
#   DWELL    "stabilize blade before", "hold at penetration before"
#   ENTRY    "enter laterally into", "descend from aligned"
#   CUT      "cut "
#   FINISH   "finish pose outside/above"
#   LIFT     "lift clear of", "lift straight up"
uint8 TRANSIT = 0
uint8 ALIGN = 1
uint8 DWELL = 2
uint8 ENTRY = 3
uint8 CUT = 4
uint8 FINISH = 5
uint8 LIFT = 6

uint8 kind

# MTC stage name (leaf->creator()->name()).
string stage_name

# Executor's own cut label (left / middle / right ...), i.e. what the executor believes it is cutting.
string cut_id

# Physical cut on the carton. Differs from cut_id when the snapshot is relabelled (rotate_180).
string physical_cut

# True if the ACM of the stage's start scene allows tool/carton contact (cut strokes only).
bool tool_carton_contact_allowed

# Joint names are the executor's own (unprefixed joint1..joint6). Time starts at 0 at the segment's
# first point. MTC segments start and end at rest.
moveit_msgs/RobotTrajectory trajectory
