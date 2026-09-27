# Vendored usb_cam 0.8.1 (+ upstream timestamp fix)

Source: https://github.com/ros-drivers/usb_cam tag `0.8.1` (the version in the
Humble apt repo), plus one upstream commit:

- `ee0a2f7` "Keep units as microseconds, not nanoseconds" (PR #338, issue #327).
  0.8.1 divides `tv_usec` by 1000 when computing the monotonic -> epoch shift,
  so every image stamp is early by a random 0-1 s offset fixed at node start.
  Measured on the knife camera (C930e): stamps 300-930 ms before the frame was
  captured; with the fix, 3-30 ms. The patch is the single line in
  `include/usb_cam/utils.hpp`.

Why not upstream `main`: its split capture/publish timers (#364) halve the rate
to 15 Hz at 1280x720 MJPEG on this camera. 0.8.1 + the fix keeps 30 Hz.

Remove this directory once a Humble release newer than 0.8.1 ships the fix
(check `apt-cache policy ros-humble-usb-cam`), then rebuild.
