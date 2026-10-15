# Vendored usb_cam 0.8.1 (+ upstream timestamp fix)

Source: https://github.com/ros-drivers/usb_cam tag `0.8.1` (the version in the
Humble apt repo), plus one upstream commit:

- `ee0a2f7` "Keep units as microseconds, not nanoseconds" (PR #338, issue #327).
  0.8.1 divides `tv_usec` by 1000 when computing the monotonic -> epoch shift,
  so every image stamp is early by a random 0-1 s offset fixed at node start.
  Measured on the knife camera (C930e): stamps 300-930 ms before the frame was
  captured; with the fix, 3-30 ms. The patch is the single line in
  `include/usb_cam/utils.hpp`.

A second local patch fixes raw `mjpeg` passthrough (`pixel_format: mjpeg`):
0.8.1 sized each `CompressedImage` as width*height*channels and memcpy'd that
many bytes from the V4L2 buffer, ignoring `bytesused` (1.8-2.8 MB messages with
stale padding, possible over-read). Now `UsbCam` records `bytes_used`
(`get_image_bytes_used()`), `process_image` copies only that for `raw_mjpeg`
(clamped to the buffer size; other formats unchanged), and
`take_and_send_image_mjpeg` publishes exactly those bytes. Files:
`usb_cam.hpp`, `usb_cam.cpp`, `utils.hpp`, `usb_cam_node.{hpp,cpp}`,
`test/test_usb_cam_utils.cpp`.
  Also: the node selected the passthrough path on `pixel_format == "mjpeg"`, but the
  format registry only accepts `raw_mjpeg`, so passthrough could never start; the
  node now checks `raw_mjpeg`. Use `pixel_format: raw_mjpeg`. Measured on the C930e
  (1280x720@30, one compressed subscriber): 1 % CPU, 57 kB/frame, vs 34-43 % for
  `mjpeg2rgb` + image_transport re-encode.

Why not upstream `main`: its split capture/publish timers (#364) halve the rate
to 15 Hz at 1280x720 MJPEG on this camera. 0.8.1 + the fix keeps 30 Hz.

Remove this directory once a Humble release newer than 0.8.1 ships the fix
(check `apt-cache policy ros-humble-usb-cam`), then rebuild.
