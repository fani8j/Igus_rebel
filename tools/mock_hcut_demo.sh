#!/usr/bin/env bash
# Plan an H-cut on MOCK hardware from the saved replay snapshot, show it in RViz, and
# (after confirmation) execute it on the mock robot so the motion can be watched.
# Isolated ROS domain + localhost only: it cannot reach the real robot.
#
#   tools/mock_hcut_demo.sh [cut_pattern]          (default t_cut)
#   PARALLEL=false tools/mock_hcut_demo.sh         (nominal left cut instead of parallel)
set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
PATTERN="${1:-t_cut}"
PARALLEL="${PARALLEL:-true}"
SNAPSHOT_SRC="${SNAPSHOT_SRC:-$WS/scan_results/p0p1_regression_20260927/source_snapshot.json}"
export ROS_DOMAIN_ID="${MOCK_DOMAIN_ID:-210}" ROS_LOCALHOST_ONLY=1

set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

OUT="$WS/scan_results/mock_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SNAP="$OUT/replay_snapshot.json"
# Mock-only replay: copy the historical snapshot with a fresh acquisition stamp.
python3 - "$SNAPSHOT_SRC" "$SNAP" <<'EOF'
import json, sys, time
d = json.load(open(sys.argv[1]))
d["acquisition_stamp_ns"] = time.time_ns()
json.dump(d, open(sys.argv[2], "w"), indent=2)
EOF

PIDS=()
start() {  # start a background process in its own process group, log to $OUT
  local name="$1"; shift
  setsid "$@" > "$OUT/$name.log" 2>&1 &
  PIDS+=("$!")
}
cleanup() {
  echo "Stopping mock stack..."
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 3
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  ros2 daemon stop > /dev/null 2>&1 || true
}
trap cleanup EXIT

# Capture output first: grep -q would close the pipe early and, with pipefail,
# turn a found match into a failure.
has_service() { local out; out="$(ros2 service list 2>/dev/null || true)"; grep -qx "$1" <<< "$out"; }
is_mock() { local out; out="$(ros2 control list_hardware_components 2>/dev/null || true)"; grep -q 'mock_components/GenericSystem' <<< "$out"; }

echo "Domain $ROS_DOMAIN_ID (localhost only). Logs: $OUT"
start rebel ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
start tf_camera ros2 run tf2_ros static_transform_publisher \
  --x -0.052362891163214 --y -0.306337290082351 --z 0.452079756273593 \
  --qx 0.022773271385773 --qy 0.373836805030297 --qz -0.079141815126831 --qw 0.923831150380609 \
  --frame-id base_link --child-frame-id camera_link
start tf_optical ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id camera_link --child-frame-id camera_color_optical_frame
sleep 5
start move_group ros2 launch igus_rebel_moveit_config move_group.launch.py \
  hardware_protocol:=mock_hardware use_gui:=true run_teleop:=false run_box_h_cut_executor:=true

echo -n "Waiting for the H-cut planner"
for _ in $(seq 1 90); do
  if has_service /box_h_cut_executor/plan_h_cut; then break; fi
  echo -n "."; sleep 2
done
echo
has_service /box_h_cut_executor/plan_h_cut || { echo "Planner did not start; see $OUT"; exit 1; }

# Refuse to continue unless the hardware really is the mock system.
if ! is_mock; then
  echo "ABORT: hardware is not mock_components/GenericSystem"; exit 1
fi
echo "Hardware: mock_components/GenericSystem (mock)."

ros2 param set /box_h_cut_executor cut_pattern "$PATTERN" > /dev/null
ros2 param set /box_h_cut_executor external_snapshot_file "$SNAP" > /dev/null
ros2 param set /box_h_cut_executor left_parallel_orientation_enabled "$PARALLEL" > /dev/null
echo "Planning $PATTERN (parallel left pose: $PARALLEL); the parallel search takes about 1 minute..."
RESULT="$(ros2 service call /box_h_cut_executor/plan_h_cut std_srvs/srv/Trigger '{}' 2>&1 | tee "$OUT/plan_response.txt")"
echo "$RESULT" | grep -E 'success=|message=' | sed 's/^/  /' | cut -c1-400
grep -q 'success=True' <<< "$RESULT" || { echo "Planning failed; see $OUT"; read -rp "Press Enter to stop..."; exit 1; }

echo
echo "The plan is shown in RViz (MotionPlanning / Task display)."
read -rp "Execute it on the MOCK robot to watch the motion? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  is_mock || { echo "ABORT: hardware is not mock"; exit 1; }
  ros2 param set /box_h_cut_executor execution_enabled true > /dev/null
  ros2 service call /box_h_cut_executor/execute_planned_h_cut std_srvs/srv/Trigger '{}' \
    | tee "$OUT/execute_response.txt" | grep -E 'success=|message=' | sed 's/^/  /' | cut -c1-300
  ros2 param set /box_h_cut_executor execution_enabled false > /dev/null || true
  echo "Motion dispatched to the mock robot; watch RViz."
fi
read -rp "Press Enter to stop the mock stack... "
