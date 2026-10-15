# cut_recorder

Automatically records video of the knife, scene and wrist cameras while a
trajectory is executing on the igus ReBeL arm, one clip folder per execution.

**Strictly passive.** The tool only subscribes to topics. It has no publishers,
no service clients, no action clients and never sets a parameter, so it cannot
influence the robot or the rest of the stack (`test/test_cut_recorder.py`
asserts the source contains none of those calls). rclpy's own idle
parameter-event channel is torn down at start-up.

## Started automatically by move_group.launch.py

`igus_rebel_moveit_config/launch/move_group.launch.py` starts this recorder
itself when `run_box_h_cut_executor:=true` and `hardware_protocol:=rebel`
(launch arg `record_cut_video`, default `true`; `cut_video_cameras`, default
`knife,scene,wrist,right_wrist`; `cut_video_dir`, default the recorder's own default). It
inherits the launch's ROS domain, and stopping MoveIt finalizes any open clip.
Mock/gazebo launches never start it. Disable with `record_cut_video:=false`.
The right-arm stack picks this up only once its install (`install_right`) is
rebuilt with the new launch file.

## Run by hand

```
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=<cell domain> ROS_LOCALHOST_ONLY=<as on the cell>
nice -n 10 python3 ~/igus_rebel_ros2/tools/cut_recorder/cut_recorder.py
```

Leave it running; Ctrl-C (or SIGTERM) stops it and finalizes any open clip so
the mp4 files are playable. It works unchanged for the right-arm single stack
(same controller name `rebel_arm_trajectory_controller`).

## How clips are triggered

It subscribes to `action_msgs/GoalStatusArray` on
`/rebel_arm_trajectory_controller/follow_joint_trajectory/_action/status`
(QoS `qos_profile_action_status_default`). A clip opens when any goal is
ACCEPTED or EXECUTING and closes when none has been for `--post-roll-s`
(default 3 s). Back-to-back MTC trajectories separated by gaps shorter than the
post-roll therefore land in one clip. `--max-clip-s` (default 900) hard-closes a
clip (a new one opens right away if a goal is still active). `--now` records
immediately until Ctrl-C and ignores the status topic.

Note: the status topic is transient-local, so on start-up the last status is
replayed; a goal left EXECUTING by a crashed controller would open a clip that
ends at `--max-clip-s`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--status-topic` | `/rebel_arm_trajectory_controller/follow_joint_trajectory/_action/status` | action status topic |
| `--post-roll-s` | 3.0 | idle time before a clip is closed |
| `--max-clip-s` | 900 | hard clip length limit |
| `--now` | off | record immediately until Ctrl-C |
| `--cameras` | `knife,scene,wrist,right_wrist` | which cameras to record |
| `--camera SPEC` | | define/override a camera (repeatable), see below |
| `--scene-scale` | 0.5 | downscale of the scene camera |
| `--out-dir` | see below | output directory |
| `--snapshot` | `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json` | copied into each clip folder |
| `--no-snapshot` | | do not copy a snapshot |

Camera spec: `name=topic[:type[:fps[:scale]]]` where `type` is `raw`
(`sensor_msgs/Image`; rgb8, bgr8, mono8) or `compressed`
(`sensor_msgs/CompressedImage`, decoded with `cv2.imdecode`). Omitted fields
inherit the default of a camera with the same name (else raw/15 fps/1.0; a
topic ending in `/compressed` defaults to `compressed`). Defaults:

- `knife=/knife_camera/image_raw/compressed:compressed:30:1.0`
- `scene=/camera/color/image_raw/compressed:compressed:15:0.5`
- `wrist=/wrist_camera/color/image_raw/compressed:compressed:15:1.0` (Gemini 305, left arm)
- `right_wrist=/right_wrist_camera/color/image_raw/compressed:compressed:15:1.0` (Gemini 305, right arm)

The Orbbec `.../compressed` topics come from image_transport's compressed plugin:
the camera node JPEG-encodes only while someone subscribes, so the recorder
running all the time adds that encoding load to the camera nodes. A camera that
is not running just shows `no frames` in `clip.json`.

Example: `--cameras knife,scene --camera scene=/camera/color/image_raw:raw:15:0.25` (raw topic instead of compressed, quarter size).

Default output directory: `/media/farhan/My Passport/igus_nvidia/cut_videos` if
that drive is mounted and writable, else `/home/farhan/Igus_rebel/outputs/cut_videos`.

## Output

```
<out-dir>/20261015_131603/      local time of clip start
    knife.mp4  scene.mp4  wrist.mp4    only for cameras that delivered frames
    clip.json
    seam_3d_base_link.json             snapshot copy (best effort, never fatal)
```

`clip.json` holds start/stop in local and ROS time, trigger (`status`/`now`),
stop reason, goal ids with their last seen status, and per camera: topic, type,
fps, scale, size, frames received / written / duplicated / dropped (with the
queue-full and same-index split), decode errors, `tail_padded_frames` and
`status` (`ok` or `no frames`). A camera without frames never blocks the others
and gets no mp4. `clip.json` is also written (with `"recording": true`) at clip
start; an mp4 of a hard-killed (SIGKILL) process is not playable.

## Video timing

Frame index = round((header stamp - clip start) x fps) on a fixed nominal fps.
Gaps are filled by repeating the previous frame, frames mapping to an already
written index are dropped, the lead-in before the first frame and the tail up to
the clip end are padded, so video length equals wall time and the cameras stay
aligned. A zero header stamp, or one more than 5 s from receipt time, is
replaced by the receipt time. Each camera has a worker thread and a bounded
queue (90 frames); a full queue drops and counts frames and never blocks the
ROS callback. Cameras are subscribed best-effort (depth 2), which works with
reliable and best-effort publishers; with best-effort large images (1080p raw)
the middleware itself can lose frames on a busy host. Those show up as
duplicated frames in `clip.json`.

## Resource use (measured, domain 194 mock run, synthetic cameras)

- Idle (waiting for a trigger): about 8-11 % of one core (the raw image
  subscriptions still copy each frame; nothing is decoded), about 170 MB RSS.
- Recording knife (JPEG 1280x720 @30) + scene (1080p @15 -> 0.5): 32-52 % of
  one core in the integration run. Offline encode benchmark with textured+noisy
  frames: knife 25 %, scene 24 %, wrist 13 % of a core for mp4v encoding, plus
  JPEG/raw decode; expect roughly one core with all three cameras.
- Disk (OpenCV `mp4v`, about 3 Mbit/s per camera): about 0.4 MB/s knife,
  0.25 MB/s scene, 0.2 MB/s wrist, so about 50 MB per minute with all three;
  a 60 s cut is roughly 50 MB. Synthetic flat test images were far smaller.

## Tests

```
python3 -m pytest -p no:anyio tools/cut_recorder/test
```
