"""Real-time cut check: colour the cut green / red while the robot is cutting.

    python live_check.py --video path/to/run/wrist.mp4             # replay a recorded run as if live
    python live_check.py --video path/to/run/right_wrist.mp4 --realtime
    python live_check.py --source 0 --camera wrist                  # a camera on this computer
    python live_check.py --source rtsp://robot:8554/wrist --camera right_wrist   # a network stream

Every frame is judged when it arrives (no look-ahead): the window shows the
camera with the seam behind the knife coloured green = cut properly or
red = skipped / not cut properly, and a bar that fills in green / red along the
cut as the knife moves. When the knife leaves the box the pass result and the
parts to re-cut are printed (and shown) at once.

On the robot, feed frames from the camera topic into LiveCutChecker.update()
(e.g. in a ROS image callback); this script is the same loop for a video,
camera or stream source.

--realtime (for --video) drops frames when processing falls behind the
camera, exactly like a live camera would. Press q in the window to stop.
Output: results/live/<name>/report.json (+ live.mp4 with --save).
"""
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

import analyze_video as av
import cut_detector as cd

CALIB_FRAMES = 8  # ~0.5 s: frames used to find where the seam sits at the start of a pass
GREEN, RED, INK, MUTED = (40, 160, 40), (40, 40, 220), (30, 30, 30), (110, 110, 110)


class LiveCutChecker:
    """Feed frames one by one with update(); it finds the cutting passes and judges them live."""

    def __init__(self, camera, fps=15.0):
        self.cam = cd.CAMERAS[camera]
        self.fps = fps
        self.recent = collections.deque(maxlen=av.SMOOTH_FRAMES)  # (index, frame, on_seam, box move)
        self.index = -1
        self.finished = []  # summaries of finished passes
        self.current = None  # the pass being cut now
        self.prev_crop = None
        self.size = 800  # min(frame height, width), set from the first frame

    # ---- pass bookkeeping -------------------------------------------------
    def _start_pass(self):
        # the pass started at the first on-seam frame still in the smoothing window
        recent = list(self.recent)
        first = next(k for k, r in enumerate(recent) if r[2])
        self.current = {"start": recent[first][0], "last_on": self.index, "shift": None, "kind": None,
                        "pending": [(i, f) for i, f, _, _ in recent[first:]],
                        "moves": [m for _, _, _, m in recent[first:]],
                        "history": [], "overlay": None, "label": None}

    def _calibrate(self, p):
        frames = [f for _, f in p["pending"]]
        p["shift"] = cd.seam_offset(frames, self.cam)
        steps = [cd.side_step(f, self.cam, p["shift"]) for f in frames[::2]]
        p["kind"] = "edge" if np.median(steps) > cd.EDGE_STEP_MIN else "top"
        for i, f in p["pending"]:  # catch up on the frames used for calibration
            self._judge(p, f)
        p["pending"] = []

    def _judge(self, p, frame):
        overlay, _, report, _ = cd.detect(frame, [self.cam["path"]], self.cam["kind"], self.cam["knife"],
                                          align=self.cam["align"], shift=p["shift"])
        p["label"] = cd.behind_knife(report, self.cam["behind_y"])
        p["history"].append(p["label"])
        p["overlay"] = overlay

    def _end_pass(self):
        p, self.current = self.current, None
        n = p["last_on"] - p["start"] + 1
        if n < av.MIN_PASS_FRAMES or p["shift"] is None:
            return None  # knife only touched the box
        hist, moves = p["history"][:n], (p["moves"] + [0.0] * n)[:n]
        s = {"pass": len(self.finished) + 1, "kind": p["kind"], "seam_offset_px": round(p["shift"] * self.size)}
        s.update(av.pass_summary(hist, moves, p["start"], self.fps, p["kind"]))
        self.finished.append(s)
        return s

    # ---- one frame --------------------------------------------------------
    def update(self, frame):
        """Judge one frame. Returns (display panel, summary of a pass that just ended or None)."""
        self.index += 1
        self.size = min(frame.shape[:2])
        on = cd.on_seam(frame)
        crop = av.motion_crop(frame)
        move = av.vertical_shift(self.prev_crop, crop) if self.prev_crop is not None else 0.0
        self.prev_crop = crop
        self.recent.append((self.index, frame, on, move))
        smooth_on = sum(r[2] for r in self.recent) * 2 > len(self.recent)

        ended, started = None, False
        if self.current is None and smooth_on:
            self._start_pass()  # already holds this frame
            started = True
        p = self.current
        if p is not None:
            if on:
                p["last_on"] = self.index
            if self.index - p["last_on"] > av.MERGE_GAP_FRAMES:
                ended = self._end_pass()
                p = None
        if p is not None:
            if not started:
                p["moves"].append(move)
                if p["shift"] is None:
                    p["pending"].append((self.index, frame))
                else:
                    self._judge(p, frame)
            if p["shift"] is None and len(p["pending"]) >= CALIB_FRAMES:
                self._calibrate(p)
        return self._panel(frame, p), ended

    # ---- display ------------------------------------------------------------
    def _panel(self, frame, p):
        t = f"t={self.index / self.fps:.1f}s"
        if p is None:
            lines = [(f"{t}  knife not on the box - waiting", MUTED)]
            view = frame
        elif p["shift"] is None:
            lines = [(f"{t}  knife on the box - finding the seam...", INK)]
            view = frame
        else:
            name = "top seam" if p["kind"] == "top" else "edge cut (estimate)"
            state = {cd.CUT: ("behind knife: CUT PROPERLY", GREEN)}.get(
                p["label"], ("behind knife: NOT CUT PROPERLY", RED) if p["label"] is not None
                else ("behind knife: not visible", MUTED))
            lines = [(f"{t}  pass {len(self.finished) + 1}: {name}", INK), state]
            view = p["overlay"]
        panel = cd.add_header(view, lines, 960)
        return np.vstack([panel, self._bar(panel.shape[1], p)])

    def _bar(self, width, p):
        bar = np.full((150, width, 3), 255, np.uint8)
        x0, x1 = 20, width - 20
        if p is not None and p["history"]:
            n = len(p["history"])
            secs = av.seam_sections(p["history"], (p["moves"] + [0.0] * n)[:n], p["start"], self.fps)
            done = sum(max(0.0, m) for m in p["moves"])
            cv2.putText(bar, "this cut so far (from where the knife started):", (x0, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
            for s in secs:
                a = int(x0 + (x1 - x0) * s["from_percent"] / 100)
                b = int(x0 + (x1 - x0) * s["to_percent"] / 100)
                cv2.rectangle(bar, (a, 32), (max(a + 1, b), 62), GREEN if s["state"] == "cut" else RED, -1)
            cv2.rectangle(bar, (x0, 32), (x1, 62), INK, 1)
            cv2.putText(bar, f"knife travelled {done:.0f} px", (x0, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
        y = 104
        for s in self.finished[-2:]:
            name = "top seam" if s["kind"] == "top" else "edge (estimate)"
            recut = ", ".join(f"{r['from_percent']}-{r['to_percent']}%" for r in s["recut_sections"])
            text = (f"pass {s['pass']} {name}: {s['cut_percent_of_seam']}% cut properly"
                    + (f" - RE-CUT {recut}" if recut else " - OK"))
            cv2.putText(bar, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if recut else GREEN, 2, cv2.LINE_AA)
            y += 24
        return bar

    def finish(self):
        """Call at the end of the stream: closes a pass still in progress."""
        return self._end_pass() if self.current is not None else None


def print_pass(s):
    name = "top seam" if s["kind"] == "top" else "edge (estimate)"
    print(f"  PASS {s['pass']} [{name}] {s['time_s'][0]}-{s['time_s'][1]}s: "
          f"{s['cut_percent_of_seam']}% cut properly -> {s['verdict']}")
    for r in s["recut_sections"]:
        print(f"      RE-CUT {r['from_percent']}-{r['to_percent']}% of this cut")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="a recorded wrist-camera video, replayed as if live")
    src.add_argument("--source", help="camera index (0, 1, ...) or stream URL")
    ap.add_argument("--camera", choices=sorted(cd.CAMERAS), help="default: guessed from the video name")
    ap.add_argument("--realtime", action="store_true", help="drop frames if processing is slower than the video")
    ap.add_argument("--save", action="store_true", help="also write the live view to live.mp4")
    ap.add_argument("--no-window", action="store_true", help="do not open a window (print results only)")
    args = ap.parse_args()

    source = args.video or (int(args.source) if args.source.isdigit() else args.source)
    camera = args.camera or (av.guess_camera(args.video) if args.video else "wrist")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    name = os.path.splitext(os.path.basename(args.video))[0] if args.video else f"live_{camera}"
    if args.video:
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(args.video)))}_{name}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "live", name)
    os.makedirs(out_dir, exist_ok=True)

    checker = LiveCutChecker(camera, fps)
    writer, start, judged, dropped = None, time.time(), 0, 0
    print(f"live check: {source} (camera '{camera}', {fps:.0f} fps)")
    while True:
        if args.video and args.realtime:  # skip frames the camera would have sent meanwhile
            due = int((time.time() - start) * fps)
            while checker.index + 1 < due and cap.grab():
                dropped += 1
                checker.index += 1
        ok, frame = cap.read()
        if not ok:
            break
        panel, ended = checker.update(frame)
        judged += 1
        if ended:
            print_pass(ended)
        if args.save:
            if writer is None:
                writer = av.open_writer(os.path.join(out_dir, "live.mp4"), fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)
        if not args.no_window:
            cv2.imshow("live cut check (q = stop)", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    ended = checker.finish()
    if ended:
        print_pass(ended)
    elapsed = time.time() - start
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    verdict = av.box_verdict(checker.finished)
    print(f"  box: {verdict}")
    print(f"  speed: {judged / elapsed:.1f} frames/s judged (camera {fps:.0f} fps), {dropped} frames dropped")
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"source": str(source), "camera": camera, "passes": checker.finished, "box_verdict": verdict,
                   "frames_per_second": round(judged / elapsed, 1), "frames_dropped": dropped}, f, indent=2)
    print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
