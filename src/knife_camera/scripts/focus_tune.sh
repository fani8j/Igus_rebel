#!/usr/bin/env bash
# Interactively tune the knife camera's manual focus while watching the stream
# (e.g. in rqt_image_view on /knife_camera/image_raw). Works whether or not the
# driver is running. Write the chosen value into config/knife_camera_controls.yaml.
#
#   ros2 run knife_camera focus_tune.sh [device]
#   keys: + / -  step 5    ] / [  step 25    q  quit
set -euo pipefail

DEV="${1:-/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_794B00AE-video-index0}"
STEP=5

get_focus() { v4l2-ctl -d "$DEV" -C focus_absolute | awk '{print $2}'; }
set_focus() { v4l2-ctl -d "$DEV" -c focus_absolute="$1"; }

v4l2-ctl -d "$DEV" -c focus_automatic_continuous=0
focus="$(get_focus)"
echo "Autofocus off. focus_absolute=$focus  (+/- step $STEP, ]/[ step 25, q quit)"

while IFS= read -rsn1 key; do
  case "$key" in
    +|=) delta=$STEP ;;
    -|_) delta=-$STEP ;;
    ]) delta=25 ;;
    [) delta=-25 ;;
    q|Q) break ;;
    *) continue ;;
  esac
  focus=$(( focus + delta ))
  (( focus < 0 )) && focus=0
  (( focus > 255 )) && focus=255
  set_focus "$focus"
  focus="$(get_focus)"
  printf '\rfocus_absolute=%-3s' "$focus"
done

echo
echo "Final value. Put this in knife_camera/config/knife_camera_controls.yaml:"
echo "  focus_absolute: $focus"
