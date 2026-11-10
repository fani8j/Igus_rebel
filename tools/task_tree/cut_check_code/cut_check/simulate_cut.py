"""Scenario test: cut a box ourselves by plan and let YOLOv8 find the uncut parts.

By default the box is the real carton seen by the overhead scene camera
(videos/scene.mp4), in frames that were kept out of training. Three cut lines
were annotated on it by hand: the top seam and the two box edges next to it
(the short sides of the top face, where the tape folds over the corners).
A knife cuts them one after the other, each following its own plan: some
sections are cut, some are cut badly, some are skipped on purpose. With
--source generated a new box is drawn instead (top seam only).

The video is given to the trained YOLOv8 model, which marks every line by
itself:

    green = cut OK        red = NOT cut (skipped or bad cut) - re-cut

While the knife moves, only the part it has already passed is coloured:
green where the cut works, red where it was skipped. The part ahead of the
knife, and lines not started yet, are not judged. At the end the model's
result is compared with the plans, which the model never sees.

Usage:
    python simulate_cut.py
    python simulate_cut.py --top "cut:0-30,miss:30-45,cut:45-100" --left "cut:0-60,miss:60-100"
    python simulate_cut.py --random --seed 3
    python simulate_cut.py --source generated --seed 5

Plan states: cut, partial (cut badly), miss (left uncut); ranges are % along
the line from where the knife starts (the top seam from the left end, the
edges from the back corner to the front corner).

Outputs in results/simulation/<source>_<seed>/:
    simulation.mp4     camera view next to the YOLOv8 result, then the final result
    cut_map.png        plan vs what YOLOv8 found for every line, with the re-cut list
    final_result.png   the finished box with the YOLOv8 colours
    report.json
"""
import argparse
import json
import os

import cv2
import numpy as np

import analyze_video as av
import synth_box as sb
import yolo_seg as ys

# two colours only: a bad cut ("partial") has to be re-cut like a skipped part, so it is red too
PLAN_CLASS = {"cut": ys.CUT_OK, "partial": ys.NOT_CUT, "miss": ys.NOT_CUT}
LINE_NAMES = {"top": "Top seam", "left": "Left edge", "right": "Right edge"}
DEFAULT_PLANS = {"top": "cut:0-30,miss:30-45,cut:45-80,partial:80-90,miss:90-100",
                 "left": "cut:0-60,miss:60-100",
                 "right": "miss:0-25,cut:25-100"}
WORDS = {"cut": "cut", "partial": "bad cut", "miss": "skipped"}


def header(width, lines, height=64):
    bar = np.full((height, width, 3), 255, np.uint8)
    for k, (text, col) in enumerate(lines):
        ys.put(bar, text, (12, 26 + 28 * k), 0.65, col, 2)
    return bar


def side_by_side(img, mask, status):
    left = np.vstack([header(img.shape[1], [("Camera: our box being cut", (30, 30, 30)),
                                           (status, (90, 90, 90))]), img])
    legend = "green = cut properly   red = skipped / not cut properly"
    right = np.vstack([header(img.shape[1], [("YOLOv8 result (found by the model)", (30, 30, 30)),
                                            (legend, (90, 90, 90))]),
                       ys.draw(img, mask)])
    return np.hstack([left, np.full((left.shape[0], 8, 3), 255, np.uint8), right])


def plan_labels(plan, bins=100):
    labels = np.full(bins, ys.BACKGROUND, np.uint8)
    for state, a, b in plan:
        labels[int(round(a)):int(round(b))] = PLAN_CLASS[state]
    return labels


def recut_list(labels):
    return [f"{a}-{b}%" for a, b, cls in ys.sections(labels) if cls == ys.NOT_CUT]


def cut_map(results, width=1200):
    """One block per line: the plan bar above what YOLOv8 found."""
    block_h = 230
    img = np.full((120 + block_h * len(results), width, 3), 255, np.uint8)
    x0, x1 = 60, width - 60
    ys.put(img, "Test: we cut the box by plan, YOLOv8 checks it", (20, 36), 0.75, (30, 30, 30), 2)
    lx = 20
    for cls, text in ((ys.CUT_OK, "cut properly"), (ys.NOT_CUT, "skipped / not cut properly - re-cut")):
        cv2.rectangle(img, (lx, 54), (lx + 26, 72), ys.COLORS[cls], -1)
        ys.put(img, text, (lx + 34, 70), 0.55)
        lx += 70 + 11 * len(text)
    overall = np.mean([r["agreement"] for r in results])
    ys.put(img, f"YOLOv8 agrees with the plans on {overall:.0f}% of all cut lines", (20, 104), 0.65,
           (40, 150, 40) if overall >= 90 else (40, 40, 220), 2)
    for k, r in enumerate(results):
        y = 120 + block_h * k
        cv2.line(img, (20, y + 6), (width - 20, y + 6), (220, 220, 220), 1)
        recut = ", ".join(r["recut"]) if r["recut"] else "none - fully cut"
        ys.put(img, f"{LINE_NAMES[r['line']]}   (agreement {r['agreement']:.0f}%)", (20, y + 34), 0.7,
               (30, 30, 30), 2)
        ys.put(img, f"re-cut: {recut}", (420, y + 34), 0.6, (40, 40, 220) if r["recut"] else (40, 150, 40), 2)
        ys.draw_bar(img, r["plan_labels"], x0, x1, y + 70, height=30, title="What we did: " + r["plan_text"])
        ys.draw_bar(img, r["found_labels"], x0, x1, y + 150, height=30, title="What YOLOv8 found")
    return img


def label_lines(img, lines):
    """Write the line names next to the lines on the final picture."""
    out = img.copy()
    for name, (start, end) in lines.items():
        x, y = (np.float32(start) + np.float32(end)) / 2
        if name == "top":
            x, y = start[0] + 0.12 * (end[0] - start[0]), start[1] - 18
        ys.put(out, LINE_NAMES[name], (int(x) + 10, int(y)), 0.6, (255, 255, 255), 4)
        ys.put(out, LINE_NAMES[name], (int(x) + 10, int(y)), 0.6, (30, 30, 30), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", "--plan", dest="top", default=DEFAULT_PLANS["top"], help="plan for the top seam")
    ap.add_argument("--left", default=DEFAULT_PLANS["left"], help="plan for the left edge")
    ap.add_argument("--right", default=DEFAULT_PLANS["right"], help="plan for the right edge")
    ap.add_argument("--random", action="store_true", help="use random plans instead")
    ap.add_argument("--source", choices=["scene", "generated"], default="scene",
                    help="scene: the real carton from videos/scene.mp4; generated: a drawn box (top seam only)")
    ap.add_argument("--seed", type=int, default=777, help="random plan / generated box number")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long the knife takes per line")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--model", default=ys.DEFAULT_MODEL)
    args = ap.parse_args()

    order = ["top", "left", "right"] if args.source == "scene" else ["top"]
    rng = np.random.default_rng(args.seed)
    plans = {line: sb.random_plan(rng) if args.random else sb.parse_plan(getattr(args, line)) for line in order}
    project = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project)
    out_dir = os.path.join(project, "results", "simulation", f"{args.source}_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    model = ys.load_model(args.model)
    if args.source == "scene":
        # real camera frames the model never trained on; cycling them keeps the real camera noise
        scenes = [sb.RealBoxScene(f, seed=args.seed, vary=False) for f in sb.SCENE_TEST_FRAMES]
        lines = {line: (scenes[0].seam_image_point(0, line), scenes[0].seam_image_point(100, line))
                 for line in order}

        def render(progress, knife):
            return scene_k.render_lines(plans, progress, knife)
    else:
        scenes = [sb.Scene(seed=args.seed)]
        scenes[0].rot90 = 0  # keep the box upright for viewing
        lines = {"top": (scenes[0].seam_image_point(0), scenes[0].seam_image_point(100))}

        def render(progress, knife):
            return scene_k.render(plans["top"], knife_at=knife[1] if knife else None, progress=progress["top"])
    for line in order:
        print(f"{LINE_NAMES[line]} plan: " + ", ".join(f"{WORDS[s]} {a:.0f}-{b:.0f}%" for s, a, b in plans[line]))

    n = int(args.seconds * args.fps)
    writer, frame_no = None, 0
    for i, line in enumerate(order):
        for k in range(n):
            p = 100.0 * k / (n - 1)
            progress = {other: 100.0 if j < i else (p if j == i else 0.0) for j, other in enumerate(order)}
            scene_k = scenes[frame_no % len(scenes)]
            frame_no += 1
            img, _ = render(progress, (line, p))
            found = ys.segment(model, img)
            found = ys.hide_unreached(found, lines, progress)  # only judge what the knife has passed
            panel = side_by_side(img, found, f"cutting the {LINE_NAMES[line].lower()}: knife at {p:.0f}%")
            if writer is None:
                panel = panel[:panel.shape[0] // 2 * 2, :panel.shape[1] // 2 * 2]
                size = (panel.shape[1], panel.shape[0])
                writer = av.open_writer(os.path.join(out_dir, "simulation.mp4"), args.fps, size)
            writer.write(cv2.resize(panel, size))

    scene_k = scenes[0]
    img, _ = render({line: 100.0 for line in order}, None)
    found_mask = ys.segment(model, img)
    final = side_by_side(img, found_mask, "all cuts finished, knife removed")
    cv2.imwrite(os.path.join(out_dir, "final_result.png"), label_lines(ys.draw(img, found_mask), lines))
    profiles = ys.line_profiles(found_mask, lines)
    results = []
    for line in order:
        found_lab = ys.merge_short(ys.fill_gaps(profiles[line]))
        plan_lab = plan_labels(plans[line])
        results.append({"line": line, "plan_labels": plan_lab, "found_labels": found_lab,
                        "agreement": 100.0 * np.mean(found_lab == plan_lab), "recut": recut_list(found_lab),
                        "plan_text": ", ".join(f"{WORDS[s]} {a:.0f}-{b:.0f}%" for s, a, b in plans[line])})
    cmap = cut_map(results)
    cv2.imwrite(os.path.join(out_dir, "cut_map.png"), cmap)

    for _ in range(2 * args.fps):
        writer.write(cv2.resize(final, size))
    s = min(size[0] / cmap.shape[1], size[1] / cmap.shape[0])
    m = cv2.resize(cmap, (int(cmap.shape[1] * s), int(cmap.shape[0] * s)))
    end_frame = np.full((size[1], size[0], 3), 255, np.uint8)
    end_frame[:m.shape[0], (size[0] - m.shape[1]) // 2:(size[0] - m.shape[1]) // 2 + m.shape[1]] = m
    for _ in range(6 * args.fps):
        writer.write(end_frame)
    writer.release()

    report = {"source": args.source, "seed": args.seed, "lines": []}
    for r in results:
        report["lines"].append({
            "line": r["line"],
            "plan": [{"state": s, "from_percent": a, "to_percent": b} for s, a, b in plans[r["line"]]],
            "found_sections": [{"from_percent": a, "to_percent": b, "state": ys.NAMES[c]}
                               for a, b, c in ys.sections(r["found_labels"]) if c in ys.NAMES],
            "recut": r["recut"], "agreement_percent": round(r["agreement"], 1)})
        print(f"{LINE_NAMES[r['line']]}: YOLOv8 found " + ", ".join(
            f"{ys.NAMES.get(c, '?')} {a}-{b}%" for a, b, c in ys.sections(r["found_labels"]))
            + f"  -> re-cut: {', '.join(r['recut']) or 'none'}  (agreement {r['agreement']:.0f}%)")
    report["agreement_percent"] = round(float(np.mean([r["agreement"] for r in results])), 1)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"overall agreement with the plans: {report['agreement_percent']:.0f}%")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
