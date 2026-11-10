# Box cutting check: cut / not-cut segmentation

Part of the dual-arm box-opening project. The robot cuts the tape and stretch
film along the top seam and down the edges; when part of the cut is missed the
box cannot be opened. This code looks at the robot's **wrist-camera video** and
shows which parts of the cut are **cut** and which are **not cut**, so the
missed sections can be re-cut.

## Setup

Python 3.8+:

```
pip install -r requirements.txt
```

## Check a robot run (one command)

The robot recorder saves every run as a folder (`clip.json`, `wrist.mp4` = left
arm, `right_wrist.mp4` = right arm, `scene.mp4`, `seam_3d_base_link.json`):

```
python check_run.py path/to/20261022_134130             # one run
python check_run.py path/to/dual_arm_cuts_completed      # every run in the folder
```

Both wrist videos are checked with `analyze_video.py` (below) and combined into
one box verdict with the re-cut list per arm. Output in `results/real_runs/<run>/`
(`run_report.json`, and per arm `cut_map.png`, `annotated.mp4`); for several runs
also `results/real_runs/summary.csv`.

Example, run `20261020_093201`:

```
[1/1] 20261020_093201: NOT FULLY CUT - re-cut the red parts
    left arm  edge     cut properly 79%   re-cut 0-4%, 12-29%
    left arm  edge     cut properly 100%  re-cut -
    left arm  top seam cut properly 100%  re-cut -
    right arm top seam cut properly 59%   re-cut 59-100%
    right arm edge     cut properly 66%   re-cut 19-31%, 78-100%
```

Re-cut positions are % from where that pass started.

## Real-time check

```
python live_check.py --video path/to/run/wrist.mp4 --realtime      # replay a recorded run as if live
python live_check.py --source 0 --camera wrist                     # a camera on this computer
python live_check.py --source rtsp://robot:8554/wrist --camera right_wrist
```

Every frame is judged when it arrives: the window shows the seam behind the
knife in green / red and a bar that fills in along the cut; when the knife
leaves the box the result and re-cut list are printed. Add `--no-window` on a
computer without a display and `--save` to write `live.mp4`. On a recording it
judged about 18 frames/s on a laptop CPU (the cameras send 15). It has only been
tested on recordings: to run it on the robot, call
`LiveCutChecker(camera, fps).update(frame)` from the camera topic's callback.

## Check a cutting video (main tool)

```
python analyze_video.py path/to/video.mp4
python analyze_video.py path/to/video.mp4 --camera right_wrist
```

`--camera` is `wrist` or `right_wrist`; if left out it is guessed from the file
name (`right` in the name means `right_wrist`). No time ranges are needed: the
tool finds the moments when the knife is on the box by itself.

Output in `results/<video name>/`:

| file | what it shows |
|---|---|
| `cut_map.png` | **the answer.** Each cut (top seam and edges) drawn as one bar from START to END of the cut: **green = cut properly**, **red = skipped or not cut properly, re-cut**. Camera photos of the red parts are underneath as proof. |
| `annotated.mp4` | every frame; the seam behind the blade is coloured green = cut properly, red = skipped / not cut properly, the knife purple; ends with the cut map |
| `timeline.jpg` | the whole video at a glance, green / red per frame; grey = knife not on the box (nothing judged) |
| `report.json` | each cutting pass: the cut / not-cut sections along the seam (`sections`) and the parts to re-cut (`recut_sections`, in % from the start of the cut), plus the box verdict |

Positions along the seam come from how far the box moved under the knife (measured
from the video), so a part the knife crossed quickly takes its real length on the map.

### How it works

1. **Find the cutting passes.** In every frame the region behind the blade is
   checked: the knife is on the box when that region shows a strong line along
   the cut direction and is cardboard-coloured (not floor or ceiling).
2. **Top seam or edge.** On a top seam both sides of the cut line are the same
   surface; on a box edge the brightness steps from one face to the other.
3. **Cut or not cut.** A finished cut opens a thin dark slit in the clear tape
   / film; where the film is not cut it still bridges the seam. Along the cut
   line (snapped onto the seam in every frame) the detector measures how much
   darker a thin line is than its two sides, averaged along the line so film
   wrinkles and glare cancel out. Strong slit = cut properly (green); a weak
   slit (cut not through) or no slit = not cut properly (red, re-cut).
4. **Map.** The knife moves along the seam, so the seam just behind the blade
   is the part it has just cut. Recording that frame by frame gives a map of
   the whole pass.

### Accuracy and limits

* **Top seam: reliable.** Against hand labels on example frames the detector
  agrees 90-100 % (`python cut_detector.py` prints this).
* **Edges: estimate.** Edge passes are shown green / red and count for the box
  verdict, but they are marked "estimate": on a box corner a cut does not open
  a dark gap, and bunched stretch film can look the same cut or uncut in this
  camera view. A wrong red only costs an extra cut; to make edges reliable, add
  an inspection pose after the edge cut (wrist camera looking straight at the
  edge, knife out of the way).
* **Real runs: accuracy not measured yet.** The results have not been compared
  with which parts really opened. On boxes with thin clear tape and no stretch
  film (the 22 Oct runs) the closed flap gap can look like a cut, so green may
  be too optimistic there.
* **Pass type can be wrong.** A short move across the end of the box can be
  taken for a top-seam pass and reported as 0 % cut (right arm of run
  `20261022_134130`, 14.3-16.7 s), and top seam and edge are sometimes swapped.
  `live_check.py` decides the type from the first 0.5 s of a pass, so it can
  differ from `check_run.py`.
* **Camera.** Set up for the wrist cameras at 1280x800. The seam position is
  measured for every pass (`seam_offset` in `cut_detector.py`), so small changes
  of the knife mounting are handled; the seam must stay within about 70 px of
  `path` in `CAMERAS`. If the knife moves more than that, update `knife` and
  `path` there.
* **Direction.** It assumes the knife moves up the image (the box moves down).
  If a pass moves the other way, `report.json` gives a warning for that pass.
* The thresholds were set from the recorded videos of one box. Check them on a
  few new boxes (try opening each box) before relying on them.

## YOLOv8 segmentation: cut test on the real box

A YOLOv8n-seg model (`models/cut_seg_yolov8n.pt`) marks the cut line:
**green = cut properly**, **red = skipped or not cut properly**, and the knife
(purple).

### The test

Needs `videos/scene.mp4`: make a `videos/` folder and copy `scene.mp4` from robot
run `20261020_100056` into it (retraining also needs `wrist.mp4` and
`right_wrist.mp4` from that run there).

```
pip install -r requirements-yolo.txt      # plus PyTorch, see the file
python simulate_cut.py                    # default plans for all three lines
python simulate_cut.py --top "cut:0-30,miss:30-45,cut:45-100" --left "cut:0-60,miss:60-100" --right "miss:0-25,cut:25-100"
python simulate_cut.py --random --seed 22 # random plans
```

1. The box is the real carton from the overhead scene camera (`videos/scene.mp4`).
   Three cut lines were annotated on it by hand (`synth_box.py`): the **top
   seam** (`SCENE_SEAM`) and the two **box edges next to it**, the short sides
   of the top face where the tape folds over the corners (`SCENE_LEFT_EDGE`,
   `SCENE_RIGHT_EDGE`).
2. We cut the lines ourselves, top seam first, then the left edge, then the
   right edge, each by its own plan: `cut` sections, `partial` (cut badly) and
   `miss` (skipped on purpose). Edge plans run from the back corner to the
   front corner. The cuts are **drawn digitally on the real camera image**;
   no physical box was cut.
3. The trained model looks at each frame **without knowing the plans** and
   colours every line. At the end its answer is compared with the plans.

Output in `results/simulation/<source>_<seed>/`: `simulation.mp4` (one video:
camera view next to the YOLOv8 result while the knife cuts all three lines,
then the final result and cut map; while cutting, only the part the knife has
already passed is coloured, so the colour follows the knife), `cut_map.png` (plan vs YOLOv8 for each
line, re-cut list, agreement), `final_result.png`, `report.json`.

### What is shown

**Green = cut properly. Red = skipped or not cut properly, re-cut it.**

The model has four classes (cut_ok, partial_cut, not_cut, knife). A bad cut
(partial_cut) and a skipped part (not_cut) both mean "re-cut", so both are
shown red.

Where detections overlap, the worse state is drawn on top (not cut over cut
OK). A missed part stops the box from opening, so a "not cut" detection must
not be hidden under a neighbouring "cut" detection (this happened where the
right edge meets the end of the top seam). Sections shorter than 3 % where
two lines meet are merged into their neighbour.

### Results (test frames kept out of training)

| plans | top seam | left edge | right edge | all |
|---|---|---|---|---|
| default (below) | 98 % | 97 % | 98 % | 98 % |
| random, seed 11 | 99 % | 91 % | 92 % | 94 % |
| random, seed 22 | 96 % | 97 % | 97 % | 97 % |
| random, seed 33 | 100 % | 99 % | 100 % | 100 % |

Default plans: top seam cut 0-30, skip 30-45, cut 45-80, bad cut 80-90,
skip 90-100; left edge cut 0-60, skip 60-100; right edge skip 0-25, cut 25-100.
YOLOv8's re-cut list for it: top seam 29-46 % and 80-100 %, left edge
57-100 %, right edge 0-27 %. Every skipped or badly cut part was marked red in
all four tests. Validation mask mAP50: cut 0.78, not cut 0.72, partial 0.30,
knife 0.99.

### Training (already done; only needed to retrain)

```
python make_yolo_dataset.py   # 1144 train / 183 val images
python train_yolo.py --device 0 --epochs 40   # from scratch: ~1 h 45 min on a GeForce MX250
python train_yolo.py --device 0 --epochs 15 --weights models/cut_seg_yolov8n.pt   # fine-tune on new data
```

The current model was trained 40 epochs on the top seam, then fine-tuned 15
epochs with the edges added. Training data: 400 generated boxes cut by random plans, 500
images of the real scene-camera box with its top seam and both edges cut by
random plans (frames 0-21; frames 24-29 are kept for testing), and 244 robot
wrist-camera frames labelled by the classical detector.
The model works on Windows and Linux; copy `models/cut_seg_yolov8n.pt` to use
it without retraining.

## Other scripts

| script | purpose |
|---|---|
| `cut_detector.py` | the detector; run it to check it against the hand-labelled frames (`results/auto/`) |
| `make_segmentation_examples.py` | hand-labelled example masks (`examples/annotations.json` -> `examples/output/`): images, label masks, overlays and YOLO-seg labels |
| `sample_frames.py` | samples sharp frames from the videos into `dataset/` for labelling (settings in `frame_sampling.json`) |
