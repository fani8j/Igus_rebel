#!/usr/bin/env bash
# ISOLATED mock stack for testing wrist_handeye.py. Mock hardware only, localhost only.
# Run:  MOCK_DOMAIN=199 setsid bash mock_stack.sh <logdir>
# The script records its own PID (== process group id because of setsid) in <logdir>/stack.pgid.
# Stop with:  kill -TERM -$(cat <logdir>/stack.pgid)
export ROS_DOMAIN_ID=${MOCK_DOMAIN:?set MOCK_DOMAIN to an unused domain (never 11)}
[ "$ROS_DOMAIN_ID" = "11" ] && { echo "refusing domain 11 (real robot)"; exit 2; }
export ROS_LOCALHOST_ONLY=1
LOG=${1:?logdir}
mkdir -p "$LOG"
echo $$ > "$LOG/stack.pgid"
source /opt/ros/humble/setup.bash
source /home/farhan/igus_rebel_ros2/install/setup.bash
TOOL=${TOOL_VARIANT:-knife_c}
NICE="nice -n 19 taskset -c 0-3"
$NICE ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware tool_variant:=$TOOL > "$LOG/rebel_launch.log" 2>&1 &
# scene camera TF (same as the cell reference), standard optical rotation
$NICE ros2 run tf2_ros static_transform_publisher --x -0.048397427117980334 --y -0.20788777061552297 --z 0.8681737492514855 \
  --qx 0.013689160489925646 --qy 0.39062643827362875 --qz -0.001688441995216913 --qw 0.9204459472290422 \
  --frame-id base_link --child-frame-id camera_link > "$LOG/tf_camera_link.log" 2>&1 &
$NICE ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id camera_link --child-frame-id camera_color_optical_frame > "$LOG/tf_optical.log" 2>&1 &
sleep 12
$NICE ros2 launch igus_rebel_moveit_config move_group.launch.py hardware_protocol:=mock_hardware tool_variant:=$TOOL \
  use_gui:=false run_teleop:=false run_box_h_cut_executor:=false run_camera_pole:=true > "$LOG/move_group_launch.log" 2>&1 &
wait
