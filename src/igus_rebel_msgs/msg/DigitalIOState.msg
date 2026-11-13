# Digital inputs/outputs of one igus ReBeL as reported by the controller's CRI STATUS message,
# published by the driver's own node as /<node_name>/digital_io at 10 Hz and immediately when
# the DIN or DOUT mask changes (a gripper handshake waits on these edges).
#
# NUMBERING: the iRC numbers in din_on / dout_on are the numbers shown on the iRC panel (base
# I/O DIn/DOut21-27, arm/tool I/O DIn/DOut31-32). The STATUS DIN/DOUT hex masks are decoded as
# iRC number = bit index + irc_offset. MEASURED on the real robot: the CRI command "DOUT n"
# switches iRC "DOut(n+1)", so the CRI number is the iRC number minus 1; the bit mapping of the
# STATUS masks itself is ASSUMED to be bit b == CRI number b (iRC b+1) and is NOT yet verified.
# Verify by switching DOut31 and checking which dout bit sets. irc_offset is the driver
# parameter dio_irc_offset (default 1).
#
# Published from the driver node, independent of the lifecycle state and of read()/write().

std_msgs/Header header                   # stamp = PC receive time (ROS clock) of the STATUS these masks come from
string component                         # ros2_control component name (URDF <ros2_control name=...>)
bool valid                               # false until a STATUS with DIN/DOUT was parsed, and while the latest STATUS is older than 0.2 s; do not trust the masks when false
uint64 din_raw                           # DIN mask exactly as parsed from STATUS (bit i = input i)
uint64 dout_raw                          # DOUT mask exactly as parsed from STATUS (bit i = output i)
uint8 irc_offset                         # iRC number = bit index + irc_offset
uint8[] din_on                           # sorted iRC numbers of the inputs that are ON (empty when not valid)
uint8[] dout_on                          # sorted iRC numbers of the outputs that are ON (empty when not valid)
