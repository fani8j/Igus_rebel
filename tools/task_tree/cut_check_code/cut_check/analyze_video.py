"""Analyse a robot wrist-camera cutting video: which parts are cut and which are not.

Give it a video recorded by the robot's wrist camera (same mounting as
videos/wrist.mp4 or videos/right_wrist.mp4). No time ranges are needed:

  1. every frame is checked for "knife on the box seam" (see
     cut_detector.on_seam); runs of such frames are the cutting passes;
  2. in each pass, the seam just behind the blade is segmented into
     cut properly / not cut (cut_detector.detect); a weak slit is a cut
     that is not proper and counts as not cut;
  3. the result is written as a cut map (the whole seam, green = cut properly,
     red = skipped or not cut properly), an annotated video and a report.

Usage:
    python analyze_video.py videos/wrist.mp4
    python analyze_video.py my_new_video.mp4 --camera right_wrist

Outputs go to results/<video name>/:
    cut_map.png     the answer: each cut from start to end, green = cut properly,
                    red = skipped or not cut properly (re-cut), with camera photos
    annotated.mp4   every frame with the segmentation, ending with the cut map
    timeline.jpg    the whole video: green = cut properly, red = not cut properly
                    (grey = knife not on the box, nothing judged)
    report.json     per pass: sections along the seam and which to re-cut
"""
import argparse
import json
import os

import cv2
import numpy as np

import cut_detector as cd

SMOOTH_FRAMES = 15   # ~1 s at 15 fps: smooths the on-seam decision
MIN_PASS_FRAMES = 30  # ~2 s; shorter on-seam runs (knife passing over the box) are ignored
MERGE_GAP_FRAMES = 8  # on-seam runs closer than this are one pass
EXPECTED_SIZE = (1280, 800)  # robot wrist camera resolution
EXPECTED_ASPECT = EXPECTED_SIZE[0] / EXPECTED_SIZE[1]
MOTION_TOLERANCE = 0.2  # px/frame; box moving up faster than this reverses "behind the blade"
MIN_RUN_FRAMES = 4  # cut-map sections shorter than this (~0.25 s) are merged into their neighbour
MIN_SECTION_PERCENT = 2  # cut-map sections shorter than this % of the seam are merged into their neighbour
MIN_TRAVEL_PX = 30  # below this total box movement, cut-map positions use time instead
CUT_MAP_SECONDS = 5  # the cut map is shown this long at the end of annotated.mp4


def open_writer(path, fps, size):
    """MP4 writer that normal video players can open.

    H.264 is tried first: on Windows through Media Foundation (built into
    Windows), elsewhere through OpenCV's FFmpeg if it has an H.264 encoder.
    If neither works, falls back to MPEG-4 Part 2 ("mp4v"), which VLC plays
    but the Windows Films & TV app and VS Code may not.
    """
    if os.path.exists(path):
        os.remove(path)  # a half-written file from a failed attempt confuses the next writer
    for api in (getattr(cv2, "CAP_MSMF", None), cv2.CAP_ANY):
        if api is None:
            continue
        writer = cv2.VideoWriter(path, api, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if writer.isOpened():
            return writer
    print("NOTE: no H.264 encoder found; writing MPEG-4 Part 2 video. If it does not play, open it in VLC.")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def guess_camera(path):
    return "right_wrist" if "right" in os.path.basename(path).lower() else "wrist"


def find_passes(flags):
    """[(first, last)] frame index ranges where the knife is on the seam."""
    flags = np.array(flags, np.float32)
    k = SMOOTH_FRAMES
    padded = np.pad(flags, (k // 2, k // 2), mode="edge")
    smooth = np.array([np.median(padded[i:i + k]) for i in range(len(flags))]) >= 0.5
    runs, start = [], None
    for i, f in enumerate(list(smooth) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append([start, i - 1])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= MERGE_GAP_FRAMES:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a + 1 >= MIN_PASS_FRAMES]


def pass_frames(cap, first, last, samples=12):
    out = []
    for i in np.linspace(first, last, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    return out


def pass_setup(cap, cam, first, last):
    """For one pass: where the seam sits (cd.seam_offset) and 'top' or 'edge' (from the
    brightness step across the cut line), from a few frames of the pass."""
    frames = pass_frames(cap, first, last)
    shift = cd.seam_offset(frames, cam)
    steps = [cd.side_step(f, cam, shift) for f in frames]
    return shift, ("edge" if steps and np.median(steps) > cd.EDGE_STEP_MIN else "top")


def motion_crop(frame):
    """Half-size grey crop of the box beside the blade, used to measure how the box moves."""
    h, w = frame.shape[:2]
    crop = frame[int(0.55 * h):h, int(0.45 * w):int(0.72 * w)]
    return cv2.cvtColor(cv2.resize(crop, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)


def vertical_shift(prev_crop, crop):
    """How far (px, full size) the box moved down the image between two frames.

    Positive means the box moves down, i.e. the knife moves up along the seam;
    the detector judges the seam below the blade as the part just cut.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_crop, crop, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(flow[..., 1])) * 2


def binary_labels(labels):
    """Turn per-frame labels into cut / not cut only, for the cut map.

    A weak slit (cd.UNCERTAIN) is a cut that is not proper, so it counts as not
    cut. Frames with nothing decided take the label of the nearest decided frame, and
    runs shorter than MIN_RUN_FRAMES are merged into their longer neighbour so
    the map shows real sections instead of single-frame flicker.
    """
    labels = [cd.NOT_CUT if l == cd.UNCERTAIN else l for l in labels]
    decided = [i for i, l in enumerate(labels) if l in (cd.CUT, cd.NOT_CUT)]
    if not decided:
        return None
    out = [l if l in (cd.CUT, cd.NOT_CUT) else labels[min(decided, key=lambda k: abs(k - i))]
           for i, l in enumerate(labels)]
    while True:
        runs, start = [], 0
        for i in range(1, len(out) + 1):
            if i == len(out) or out[i] != out[start]:
                runs.append((start, i - 1))
                start = i
        short = [r for r in runs if r[1] - r[0] + 1 < MIN_RUN_FRAMES]
        if not short or len(runs) == 1:
            return out
        a, b = short[0]
        k = runs.index((a, b))
        neighbours = [runs[j] for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        na, nb = max(neighbours, key=lambda r: r[1] - r[0])
        out[a:b + 1] = [out[na]] * (b - a + 1)


def seam_sections(labels, moves, first, fps):
    """Cut / not-cut sections along the seam, positioned by how far the box moved."""
    binary = binary_labels(labels)
    if binary is None:
        return []
    steps = np.clip(np.array(moves, np.float32), 0, None)
    if steps.sum() < MIN_TRAVEL_PX:  # too little measured motion: fall back to time
        steps = np.ones(len(labels), np.float32)
    cum = np.concatenate([[0.0], np.cumsum(steps)]) / steps.sum()
    sections, start = [], 0
    for i in range(1, len(binary) + 1):
        if i == len(binary) or binary[i] != binary[start]:
            sections.append({"state": "cut" if binary[start] == cd.CUT else "not cut",
                             "from_percent": int(round(100 * cum[start])),
                             "to_percent": int(round(100 * cum[i])),
                             "time_s": [round((first + start) / fps, 1), round((first + i) / fps, 1)],
                             "middle_frame": first + (start + i - 1) // 2,
                             "photo_s": round((first + (start + i - 1) // 2) / fps, 1)})
            start = i
    # a few frames where the box hardly moved (arm starting or stopping) can make a
    # section 0-1 % long: too short to re-cut, so it joins its longer neighbour
    while len(sections) > 1:
        short = [s for s in sections if s["to_percent"] - s["from_percent"] < MIN_SECTION_PERCENT]
        if not short:
            break
        k = sections.index(short[0])
        nb = max((sections[j] for j in (k - 1, k + 1) if 0 <= j < len(sections)),
                 key=lambda s: s["to_percent"] - s["from_percent"])
        s = sections.pop(k)
        nb["from_percent"], nb["to_percent"] = min(nb["from_percent"], s["from_percent"]), max(nb["to_percent"], s["to_percent"])
        nb["time_s"] = [min(nb["time_s"][0], s["time_s"][0]), max(nb["time_s"][1], s["time_s"][1])]
        merged = []  # neighbours of the same state become one section
        for s in sections:
            if merged and merged[-1]["state"] == s["state"]:
                merged[-1]["to_percent"], merged[-1]["time_s"][1] = s["to_percent"], s["time_s"][1]
            else:
                merged.append(s)
        sections = merged
    return sections


def pass_summary(labels, moves, first, fps, kind):
    decided = [l for l in labels if l is not None]

    def pct(lab):
        return round(100.0 * sum(l == lab for l in decided) / len(decided), 1) if decided else None

    sections = seam_sections(labels, moves, first, fps)
    recut = [s for s in sections if s["state"] == "not cut"]
    if not sections:
        verdict = "NOT CHECKED (seam not visible)"
    else:
        verdict = "FULLY CUT" if not recut else "NOT FULLY CUT"
    cut_share = sum(s["to_percent"] - s["from_percent"] for s in sections if s["state"] == "cut")
    weak = pct(cd.UNCERTAIN) or 0.0
    return {"time_s": [round(first / fps, 1), round((first + len(labels)) / fps, 1)],
            "cut_percent_of_seam": cut_share if sections else None,
            "sections": sections,
            "recut_sections": recut,
            "verdict": verdict,
            "frames_percent": {"cut_properly": pct(cd.CUT),
                               "not_cut_properly": round((pct(cd.NOT_CUT) or 0.0) + weak, 1) if decided else None}}


GREEN, RED, INK, MUTED = (40, 160, 40), (40, 40, 220), (30, 30, 30), (110, 110, 110)


def put(img, text, org, scale=0.6, color=INK, thick=1, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int(org[0] - tw / 2), org[1])
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def thumbnail(video, cam, index, shift=0.0, width=250):
    """The annotated camera view at one frame, cropped around the seam behind the knife."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    overlay, _, _, _ = cd.detect(frame, [cam["path"]], cam["kind"], cam["knife"], align=cam["align"], shift=shift)
    h, w = overlay.shape[:2]
    crop = overlay[int(0.30 * h):h, int(0.45 * w):int(0.90 * w)]
    return cv2.resize(crop, (width, int(crop.shape[0] * width / crop.shape[1])))


def draw_cut_map(title, verdict, summaries, video, cam, width=1200):
    """One picture of every cutting pass: green = cut properly, red = skipped or not cut properly."""
    x0, x1 = 70, width - 70
    blocks = []
    head = np.full((120, width, 3), 255, np.uint8)
    put(head, title, (20, 36), 0.75, INK, 2)
    put(head, verdict, (20, 74), 0.85, GREEN if verdict.startswith("FULLY") else RED, 2)
    lx = 20
    for col, text in ((GREEN, "cut properly"), (RED, "skipped / not cut properly - re-cut")):
        cv2.rectangle(head, (lx, 92), (lx + 26, 110), col, -1)
        put(head, text, (lx + 34, 107), 0.55)
        lx += 60 + 12 * len(text)
    blocks.append(head)

    for k, s in enumerate(summaries, 1):
        thumbs = []
        for sec in s["recut_sections"][:3]:
            thumbs.append((sec, RED, f"NOT CUT {sec['from_percent']}-{sec['to_percent']}%"))
        ok_secs = [x for x in s["sections"] if x["state"] == "cut"]
        if ok_secs:
            best = max(ok_secs, key=lambda x: x["to_percent"] - x["from_percent"])
            thumbs.append((best, GREEN, f"cut {best['from_percent']}-{best['to_percent']}%"))
        thumbs = [(thumbnail(video, cam, sec["middle_frame"], s.get("seam_shift", 0.0)), col, cap, sec)
                  for sec, col, cap in thumbs]
        thumbs = [t for t in thumbs if t[0] is not None]
        th = max([t[0].shape[0] for t in thumbs], default=0)
        block = np.full((200 + (th + 50 if thumbs else 0), width, 3), 255, np.uint8)
        cv2.line(block, (20, 4), (width - 20, 4), (225, 225, 225), 1)
        name = "top seam" if s["kind"] == "top" else "edge cut (estimate: edges are hard to see from the wrist camera)"
        put(block, f"Pass {k}: {name}   ({s['time_s'][0]}-{s['time_s'][1]} s in the video)", (20, 34), 0.65, INK, 2)
        by = 80
        if not s["sections"]:
            cv2.rectangle(block, (x0, by), (x1, by + 46), INK, 1)
            put(block, "seam not visible in this pass", ((x0 + x1) // 2, by + 30), 0.55, INK, 1, center=True)
        else:
            for sec in s["sections"]:
                a = int(x0 + (x1 - x0) * sec["from_percent"] / 100)
                b = int(x0 + (x1 - x0) * sec["to_percent"] / 100)
                cv2.rectangle(block, (a, by), (max(a + 1, b), by + 46), GREEN if sec["state"] == "cut" else RED, -1)
                if sec["state"] == "not cut" and b - a > 30:
                    put(block, "RE-CUT", ((a + b) // 2, by - 10), 0.55, RED, 2, center=True)
            cv2.rectangle(block, (x0, by), (x1, by + 46), INK, 1)
        for p in (0, 25, 50, 75, 100):
            x = int(x0 + (x1 - x0) * p / 100)
            cv2.line(block, (x, by + 46), (x, by + 54), INK, 1)
            put(block, f"{p}%", (x, by + 74), 0.5, MUTED, 1, center=True)
        put(block, "START of cut", (x0, by + 100), 0.55, INK, 1)
        (tw, _), _ = cv2.getTextSize("END of cut", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        put(block, "END of cut", (x1 - tw, by + 100), 0.55, INK, 1)
        if s["sections"]:
            put(block, f"{s['cut_percent_of_seam']}% cut properly", ((x0 + x1) // 2, by + 100), 0.6,
                GREEN if not s["recut_sections"] else RED, 2, center=True)
        tx = x0
        for img, col, cap, sec in thumbs:
            ty = 200
            block[ty:ty + img.shape[0], tx:tx + img.shape[1]] = img
            cv2.rectangle(block, (tx - 2, ty - 2), (tx + img.shape[1] + 1, ty + img.shape[0] + 1), col, 3)
            put(block, cap, (tx, ty + img.shape[0] + 24), 0.55, col, 2)
            put(block, f"camera at {sec['photo_s']} s", (tx, ty + img.shape[0] + 44), 0.5, MUTED, 1)
            tx += img.shape[1] + 24
        blocks.append(block)
    return np.vstack(blocks)


def box_verdict(summaries):
    if not summaries:
        return "NO CUTTING PASS FOUND - check that this is a robot wrist-camera video"
    if any(s["verdict"] == "NOT FULLY CUT" for s in summaries):
        return "NOT FULLY CUT - re-cut the RED parts"
    if not any(s["sections"] for s in summaries):
        return "NOT CHECKED - seam not visible"
    return "FULLY CUT"


def timeline(history, passes, kinds, fps, width):
    bar = np.full((140, width, 3), 255, np.uint8)
    n = len(history)
    for i, lab in enumerate(history):
        x0, x1 = int(10 + i * (width - 20) / n), int(10 + (i + 1) * (width - 20) / n)
        cv2.rectangle(bar, (x0, 40), (max(x0, x1 - 1), 80), cd.COLORS.get(lab, (200, 200, 200)), -1)
    for k, ((a, b), kind) in enumerate(zip(passes, kinds), 1):
        xa = int(10 + a * (width - 20) / n)
        text = f"pass {k}: {kind}" + (" (estimate)" if kind == "edge" else "")
        y = 98 if k % 2 else 116  # alternate rows so neighbouring labels do not overlap
        cv2.putText(bar, text, (xa, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for s in range(0, int(n / fps) + 1, 10):
        x = int(10 + s * fps * (width - 20) / n)
        cv2.putText(bar, f"{s}s", (x, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.putText(bar, "timeline: green = cut properly | red = skipped / not cut properly | grey = knife not on box",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--camera", choices=sorted(cd.CAMERAS), help="default: guessed from the file name")
    ap.add_argument("--out", help="output folder (default results/<video name>)")
    ap.add_argument("--force", action="store_true",
                    help="analyse even if the video does not look like a wrist-camera video")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"video not found: {args.video}")
    camera = args.camera or guess_camera(args.video)
    cam = cd.CAMERAS[camera]
    name = os.path.splitext(os.path.basename(args.video))[0]
    project = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(project, "results", name)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video} (not a video, or codec not supported)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h == 0 or abs(w / h - EXPECTED_ASPECT) > 0.05:
        msg = (f"{args.video} is {w}x{h}, but the robot wrist cameras record {EXPECTED_SIZE[0]}x"
               f"{EXPECTED_SIZE[1]}. The knife position is calibrated for the wrist-camera view, so "
               "results on other cameras are not valid.")
        if not args.force:
            raise SystemExit(msg + " Use --force to analyse it anyway.")
        print("WARNING: " + msg)
    os.makedirs(out_dir, exist_ok=True)

    # 1. find the cutting passes
    flags = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        flags.append(cd.on_seam(frame))
    if not flags:
        raise SystemExit(f"no frames could be read from {args.video}")
    passes = find_passes(flags)
    in_pass = np.full(len(flags), -1)
    for k, (a, b) in enumerate(passes):
        in_pass[a:b + 1] = k
    setups = [pass_setup(cap, cam, a, b) for a, b in passes]
    shifts, kinds = [s for s, _ in setups], [k for _, k in setups]
    print(f"{args.video}: {len(flags)} frames, camera '{camera}', {len(passes)} cutting pass(es) found: "
          + ", ".join(f"{a / fps:.1f}-{(b + 1) / fps:.1f}s ({k}, seam at {s * min(w, h):+.0f} px)"
                      for (a, b), k, s in zip(passes, kinds, shifts)))

    # 2. segment cut / not cut in every pass frame, write the annotated video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    history, writer = [None] * len(flags), None
    moves, prev_crop = [0.0] * len(flags), None
    for i in range(len(flags)):
        ok, frame = cap.read()
        if not ok:
            break
        t = f"{name}  t={i / fps:.1f}s"
        crop = motion_crop(frame)
        if in_pass[i] >= 0 and prev_crop is not None:
            moves[i] = vertical_shift(prev_crop, crop)
        prev_crop = crop
        if in_pass[i] >= 0:
            overlay, _, report, _ = cd.detect(frame, [cam["path"]], cam["kind"], cam["knife"],
                                              align=cam["align"], shift=shifts[in_pass[i]])
            history[i] = cd.behind_knife(report, cam["behind_y"])
            a, _ = passes[in_pass[i]]
            done = [h for h in history[a:i + 1] if h is not None]
            pct = round(100.0 * sum(h == cd.CUT for h in done) / len(done)) if done else 0
            kind = kinds[in_pass[i]]
            label = f"{t}  pass {in_pass[i] + 1}: {kind} cut" + ("  (estimate)" if kind == "edge" else "")
            lines = cd.frame_lines(label, history[i], pct)
        else:
            overlay = frame
            lines = [(t, (0, 0, 0)), ("knife not on box seam - not judged", (90, 90, 90)), ("", (0, 0, 0))]
        panel = cd.add_header(overlay, lines, 960)
        panel = np.vstack([panel, timeline(history[:i + 1] + [None] * (len(flags) - i - 1),
                                           passes, kinds, fps, panel.shape[1])])
        # H.264 needs even frame sizes
        panel = cv2.copyMakeBorder(panel, 0, panel.shape[0] % 2, 0, panel.shape[1] % 2,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if writer is None:
            writer = open_writer(os.path.join(out_dir, "annotated.mp4"), fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    cap.release()

    # 3. report and cut map
    cv2.imwrite(os.path.join(out_dir, "timeline.jpg"), timeline(history, passes, kinds, fps, 1400))
    summaries = []
    for (a, b), kind, shift in zip(passes, kinds, shifts):
        dy = float(np.median(moves[a + 1:b + 1])) if b > a else 0.0
        s = {"kind": kind, "confidence": "LOW" if kind == "edge" else "normal",
             "box_motion_px_per_frame": round(dy, 2), "seam_shift": round(shift, 4),
             "seam_offset_px": round(shift * min(w, h))}
        s.update(pass_summary(history[a:b + 1], moves[a:b + 1], a, fps, kind))
        if dy < -MOTION_TOLERANCE:
            s["warning"] = ("knife moved down the image in this pass; the seam judged (below the blade) "
                            "may not be cut yet, so not-cut results here can be false")
        summaries.append(s)
    verdict = box_verdict(summaries)
    report = {"video": args.video, "camera": camera, "passes": summaries, "box_verdict": verdict}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    cut_map = draw_cut_map(f"Cut map: {name}", verdict, summaries, args.video, cam)
    cv2.imwrite(os.path.join(out_dir, "cut_map.png"), cut_map)

    if writer:  # end the video with the cut map
        pw, ph = panel.shape[1], panel.shape[0]
        s = min(pw / cut_map.shape[1], ph / cut_map.shape[0])
        m = cv2.resize(cut_map, (int(cut_map.shape[1] * s), int(cut_map.shape[0] * s)))
        end = np.full((ph, pw, 3), 255, np.uint8)
        end[:m.shape[0], (pw - m.shape[1]) // 2:(pw - m.shape[1]) // 2 + m.shape[1]] = m
        for _ in range(int(CUT_MAP_SECONDS * fps)):
            writer.write(end)
        writer.release()

    for k, s in enumerate(summaries, 1):
        name_k = "top seam" if s["kind"] == "top" else "edge, estimate"
        cut = f"{s['cut_percent_of_seam']}% cut properly -> " if s["sections"] else ""
        print(f"  pass {k} [{name_k}] {s['time_s'][0]}-{s['time_s'][1]}s: {cut}{s['verdict']}")
        for m in s["recut_sections"]:
            print(f"      RE-CUT {m['from_percent']}-{m['to_percent']}% of this cut "
                  f"(seen at {m['time_s'][0]}-{m['time_s'][1]}s)")
        if "warning" in s:
            print(f"      WARNING: {s['warning']}")
    print(f"  box: {verdict}")
    print(f"  wrote {os.path.join(out_dir, 'cut_map.png')}, annotated.mp4, timeline.jpg, report.json")


if __name__ == "__main__":
    main()
