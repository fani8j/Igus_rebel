"""Check one robot cutting run (or a folder of runs) for cut / not-cut parts.

python check_run.py path/to/20261022_134130          # one run folder
python check_run.py path/to/dual_arm_cuts_completed  # every run folder inside

A run folder is what the robot recorder writes: clip.json, wrist.mp4 (left arm),
right_wrist.mp4 (right arm), scene.mp4 and seam_3d_base_link.json. Both wrist videos
are checked with analyze_video.py; the results are combined into one box verdict.

Output in results/real_runs/<run>/: wrist/ and right_wrist/ (cut_map.png, annotated.mp4,
report.json from analyze_video.py) and run_report.json. With several runs,
results/real_runs/summary.csv lists them all.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
ARMS = {"wrist": "left arm", "right_wrist": "right arm"}


def run_folders(path):
    if os.path.isfile(os.path.join(path, "clip.json")):
        return [path]
    found = sorted(os.path.join(path, d) for d in os.listdir(path)
                   if os.path.isfile(os.path.join(path, d, "clip.json")))
    if not found:
        raise SystemExit(f"no run folders (with clip.json) in {path}")
    return found


def seam_lengths(path):
    """Cut-line lengths in mm from the robot's 3D seam snapshot (None if missing)."""
    f = os.path.join(path, "seam_3d_base_link.json")
    if not os.path.isfile(f):
        return None
    j = json.load(open(f))

    def mm(a, b):
        return round(1000 * math.dist(*([j[k][c] for c in "xyz"] for k in (a, b))))
    try:
        return {"top_seam_mm": mm("seam_start", "seam_end"),
                "left_edge_mm": mm("back_middle_left", "front_middle_left"),
                "right_edge_mm": mm("back_middle_right", "front_middle_right"),
                "cut_permission": j.get("cut_permission", {}).get("status")}
    except KeyError:
        return None


def check_run(path, out_root):
    name = os.path.basename(os.path.normpath(path))
    clip = json.load(open(os.path.join(path, "clip.json")))
    out_dir = os.path.join(out_root, name)
    report = {"run": name, "folder": path, "start": clip.get("start_local"),
              "geometry": seam_lengths(path), "arms": {}, "notes": []}
    for cam, info in clip.get("cameras", {}).items():
        if info.get("status") != "ok" and cam in ("knife", *ARMS):
            report["notes"].append(f"{cam} camera recorded no video ({info.get('status')})")
    for cam, arm in ARMS.items():
        video = os.path.join(path, f"{cam}.mp4")
        if not os.path.isfile(video):
            report["arms"][arm] = {"verdict": "NO VIDEO"}
            continue
        cam_out = os.path.join(out_dir, cam)
        proc = subprocess.run([sys.executable, os.path.join(PROJECT, "analyze_video.py"), video,
                               "--camera", cam, "--out", cam_out], capture_output=True, text=True)
        if proc.returncode != 0:
            report["arms"][arm] = {"verdict": "ERROR", "error": proc.stderr.strip()[-500:]}
            continue
        r = json.load(open(os.path.join(cam_out, "report.json")))
        name = {"top": "top seam", "edge": "edge"}
        report["arms"][arm] = {
            "passes": [{"cut": name[p["kind"]], "cut_properly_percent": p["cut_percent_of_seam"],
                        "recut": [f"{s['from_percent']}-{s['to_percent']}%" for s in p["recut_sections"]]}
                       for p in r["passes"]],
            "verdict": r["box_verdict"], "cut_map": os.path.join(cam_out, "cut_map.png")}
    passes = [p for a in report["arms"].values() for p in a.get("passes", [])]
    if any(p["recut"] for p in passes):
        report["verdict"] = "NOT FULLY CUT - re-cut the red parts"
    elif any(p["cut_properly_percent"] is not None for p in passes):
        report["verdict"] = "FULLY CUT"
    else:
        report["verdict"] = "NOT CHECKED - no cutting pass found"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a run folder, or a folder of run folders")
    ap.add_argument("--out", default=os.path.join(PROJECT, "results", "real_runs"))
    args = ap.parse_args()
    runs = run_folders(args.path)
    rows = []
    for k, path in enumerate(runs, 1):
        r = check_run(path, args.out)
        print(f"[{k}/{len(runs)}] {r['run']}: {r['verdict']}")
        for arm, a in r["arms"].items():
            if "passes" not in a:
                print(f"    {arm:9s} {a['verdict']}")
            for p in a.get("passes", []):
                pct = "-" if p["cut_properly_percent"] is None else f"{p['cut_properly_percent']}%"
                print(f"    {arm:9s} {p['cut']:8s} cut properly {pct:5s} re-cut {', '.join(p['recut']) or '-'}")
            for p in a.get("passes", []):
                rows.append([r["run"], arm, p["cut"], p["cut_properly_percent"], " ".join(p["recut"]), r["verdict"]])
        for note in r["notes"]:
            print(f"    note: {note}")
    if len(runs) > 1:
        with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "arm", "cut", "cut properly %", "re-cut (red)", "box verdict"])
            w.writerows(rows)
        print(f"wrote {os.path.join(args.out, 'summary.csv')}")
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
