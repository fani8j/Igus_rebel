# Videos (moved to the external drive, `/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/`)

| folder / file | what it shows |
|---|---|
| `carrier_20261015/` (56 MP4) | OLD carrier test (`tests/test_cabinet_grasp.py`): kinematic tool carriers, no robot. `<mode>_<target>[_<variant>]_{3q,side}.mp4`, 3/4 front-left view and side close-up, 1280x720, 30 fps, real time. |
| `highlights.mp4` | old carrier highlight reel. |
| `robot_20261015/front_<T>_{3q,side,head,tiled}.mp4` | NEW full-robot test (`tests/test_cabinet_grasp_robot.py`, tools on the D200 dual-arm robot, arms driven by joint drives). 3q = 3/4 front-left (wide, robot visible), side = close-up of the grasp, head = stand head camera `gemini336L_rgb` of the D200 model (real intrinsics, 94 deg HFOV, 0.17..20 m), tiled = 1920x720 (3q large + side and head small). Captions: mode, shelf, carton, vacuum, target, time, parking distance d, lift height, step text, result card. Front mode only (targets T3..T8, E1); top-down has no feasible robot pose, T1/T2/E2 unreachable. |
| `robot_20261015/highlights_robot.mp4` | 6 clips (tiled) T3, T4, T5, T6, T7, T8. |

Stills (jpg < 300 KB): `tests/results/stills/` (old carrier test), `tests/results/stills_robot/` (robot test, incl. head camera frames `front_*_head.jpg`).

## Closest-attempt videos for the unreachable targets (`robot_20261015/front_T1_closest_*`, `front_T2_closest_*`, each `_3q/_side/_head/_tiled`)
Parking d 410 mm, lift at its lowest limit (-0.6235 m), sliders home. The arm moves from home along a collision-free path to the best-effort pose (closest to the press / grasp pose under joint limits and clearance), holds 3 s with a translucent red box (visual only) from the pad TCP to the target point; suction arm first, then the gripper arm. Footer caption only (no header). No vacuum, no grasp. Numbers: `tests/results/unreachable_closest.json`; stills `tests/results/stills_robot/front_T{1,2}_closest_tiled.jpg`.
