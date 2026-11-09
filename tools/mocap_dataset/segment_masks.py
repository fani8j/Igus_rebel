#!/usr/bin/env python3
"""Tracked SAM3 segmentation masks (robot / box / background) for a LeRobot v3.0 video dataset.

Runs in the python that has transformers>=5 with Sam3VideoModel and cv2 (the box_seam_detection SAM3 env,
/home/farhan/miniconda3/envs/book_grasping/bin/python); it needs no PyAV and no lerobot: frames are read straight from the
mp4 files with cv2 using the per-episode from/to timestamps of meta/episodes (read with pandas).

Per episode and camera the output is ``<out>/<episode_name>/<camera>/labels.npz`` with ``labels`` (N, H, W) uint8:
0 background, 1 robot (arm + gripper/tool), 2 box (incl. flaps).  Where robot and box both claim a pixel the robot wins.
``<out>/index.json`` holds model, prompts, thresholds and per-episode/camera area stats + suspected failure frames.

Method: Sam3VideoModel (text prompted concept detection + tracker), propagated through CHUNK frames at a time (a fresh session
= fresh text detection per chunk, so tracks cannot drift for long; the video session keeps all chunk frames in RAM).  Each
chunk is run on every ``stride``-th frame; masks of skipped frames are the nearest processed frame's.

Throttle: single GPU job; callers must check nvidia-smi first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np

MODEL_ID = "facebook/sam3"
# Robot (head cam): SAM3 video tracker, text prompt.  Box: SAM3 image detector on every ``box_stride``-th frame (the video
# tracker lost the carton body for hundreds of frames in the pilot; per-frame detection copes with occlusion by the arm).
VIDEO_PROMPTS = {"head": {1: ["robot arm"]}, "wrist_right": {}}
BOX_PROMPT = "cardboard box"
BOX_THRESHOLD = 0.5
DEFAULT_PROMPTS = {c: {**VIDEO_PROMPTS[c], 2: [BOX_PROMPT]} for c in VIDEO_PROMPTS}
# The head scene holds several other cartons (shelf, left, right).  SAM3 finds all of them as separate objects; the target
# carton sits at the bottom centre and the camera is fixed, so only box objects whose centroid lies in this ROI
# (x0, y0, x1, y1 in pixels) and whose area is >= MIN_BOX_AREA are kept.  No ROI for the moving wrist camera.
BOX_ROI = {"head": (110, 200, 470, 480), "wrist_right": None}
MIN_BOX_AREA = {"head": 1500, "wrist_right": 800}
# The gripper-camera fingers are rigidly fixed in the image (per-pixel 97th percentile of luma over the whole dataset is
# < 80 there): that static silhouette is used for the wrist robot class (SAM3 does not recognise the fingers).
STATIC_ROBOT = {"wrist_right": True}
LABELS = {0: "background", 1: "robot", 2: "box"}


def load_episodes(root: Path) -> list[dict]:
    try:
        import pyarrow  # noqa: F401
    except ImportError:               # the SAM3 env has no parquet engine: ask the system python
        import subprocess
        if os.environ.get("SEGMENT_MASKS_NO_RECURSE"):
            raise
        r = subprocess.run(["/usr/bin/python3", str(Path(__file__).resolve()), "--dump-table", str(root)],
                           check=True, capture_output=True, text=True,
                           env=dict(os.environ, SEGMENT_MASKS_NO_RECURSE="1"), cwd="/")
        return json.loads(r.stdout)
    import pandas as pd
    info = json.loads((root / "meta" / "info.json").read_text())
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True)
    names = {}
    sp = root / "meta" / "source_episodes.json"
    if sp.exists():
        raw = json.loads(sp.read_text())
        names = ({int(k): v for k, v in raw.items()} if isinstance(raw, dict)
                 else {int(v.get("episode_index", i)): v.get("source") for i, v in enumerate(raw)})
    eps = []
    for _, r in df.sort_values("episode_index").iterrows():
        e = {"episode_index": int(r["episode_index"]), "length": int(r["length"]), "cams": {}}
        nm = names.get(e["episode_index"])
        if isinstance(nm, dict):
            nm = nm.get("source_episode") or nm.get("name") or nm.get("episode")
        e["name"] = str(nm) if nm else f"ep{e['episode_index']:03d}"
        for key in [k for k in info["features"] if info["features"][k]["dtype"] == "video"]:
            e["cams"][key.split(".")[-1]] = {
                "path": str(root / info["video_path"].format(video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]),
                                                             file_index=int(r[f"videos/{key}/file_index"]))),
                "from_ts": float(r[f"videos/{key}/from_timestamp"])}
        eps.append(e)
    return eps


def read_episode_frames(path: str, from_ts: float, n: int, fps: float = 30.0) -> np.ndarray:
    """RGB uint8 (n, H, W, 3) of frames [from_ts*fps, +n) of an mp4 (sequential cv2 decode)."""
    cap = cv2.VideoCapture(path)
    start = int(round(from_ts * fps))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # h264 seek lands on the requested frame in OpenCV's ffmpeg backend; verify by position
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
            raise RuntimeError(f"{path}: seek to frame {start} failed")
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[:, :, ::-1])
    cap.release()
    if len(out) != n:
        raise RuntimeError(f"{path}: read {len(out)} frames, expected {n}")
    return np.ascontiguousarray(np.stack(out))


class Sam3Video:
    def __init__(self, device="cuda:0", threshold=None):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        self.torch, self.device = torch, device
        self.processor = Sam3VideoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = Sam3VideoModel.from_pretrained(MODEL_ID, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
        self.cfg = {k: getattr(self.model.config, k, None) for k in ("score_threshold_detection", "new_det_thresh",
                                                                      "hotstart_delay", "fill_hole_area")}

    def run_chunk(self, frames: np.ndarray, prompts: dict[int, list[str]], roi=None, min_box_area=0):
        """frames (n,H,W,3) RGB -> labels (n,H,W) uint8 and per-class per-frame detection count (n, n_classes+1)."""
        torch = self.torch
        n, H, W, _ = frames.shape
        sess = self.processor.init_video_session(video=list(frames), inference_device=self.device,
                                                 processing_device=self.device, video_storage_device="cpu",
                                                 dtype=torch.bfloat16)
        flat = [(c, p) for c, ps in prompts.items() for p in ps]
        self.processor.add_text_prompt(sess, [p for _, p in flat])
        cls_of_text = {p: c for c, p in flat}
        labels = np.zeros((n, H, W), np.uint8)
        ndet = np.zeros((n, 3), np.int16)
        score = np.zeros((n, 3), np.float32)
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            for out in self.model.propagate_in_video_iterator(inference_session=sess, start_frame_idx=0):
                post = self.processor.postprocess_outputs(sess, out)
                i = out.frame_idx
                masks = post["masks"].cpu().numpy()
                ids = post["object_ids"].tolist()
                sc = post["scores"].float().cpu().numpy()
                cm = {c: np.zeros((H, W), bool) for c in prompts}
                for text, oids in post["prompt_to_obj_ids"].items():
                    c = cls_of_text.get(text)
                    for oid in oids:
                        if oid in ids:
                            k = ids.index(oid)
                            if c == 2:
                                ys, xs = np.nonzero(masks[k])
                                if len(xs) < min_box_area:
                                    continue
                                if roi is not None and not (roi[0] <= xs.mean() <= roi[2] and roi[1] <= ys.mean() <= roi[3]):
                                    continue
                            cm[c] |= masks[k]
                            ndet[i, c] += 1
                            score[i, c] = max(score[i, c], float(sc[k]))
                for c in sorted(prompts, reverse=True):      # box first, robot (1) overwrites
                    labels[i][fill_holes(cm[c])] = c
        del sess
        torch.cuda.empty_cache()
        return labels, ndet, score


class Sam3Image:
    """Per-frame SAM3 concept detector (transformers Sam3Model, the same loading as box_seam_detection/core/sam3_worker.py)."""

    def __init__(self, device="cuda:0"):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.device = torch, device
        self.processor = Sam3Processor.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = Sam3Model.from_pretrained(MODEL_ID, local_files_only=True, dtype=torch.bfloat16, device_map=device).eval()

    def detect(self, frame_rgb: np.ndarray, prompt: str, threshold: float):
        """-> list of (bool mask (H, W), score)."""
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(frame_rgb)
        inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            out = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
        masks = res["masks"].detach().cpu().numpy() > 0
        scores = [float(x) for x in res["scores"].detach().cpu()]
        return list(zip(masks, scores))


def select_box(dets, roi, min_area: int) -> tuple[np.ndarray | None, int, float]:
    """Union of the detections that look like the target carton (area >= min_area, centroid inside roi when given)."""
    m, n, best = None, 0, 0.0
    for mask, sc in dets:
        ys, xs = np.nonzero(mask)
        if len(xs) < min_area:
            continue
        if roi is not None and not (roi[0] <= xs.mean() <= roi[2] and roi[1] <= ys.mean() <= roi[3]):
            continue
        m = mask.copy() if m is None else (m | mask)
        n += 1
        best = max(best, sc)
    return m, n, best


def fill_holes(m: np.ndarray, close: int = 9) -> np.ndarray:
    """Close small gaps (flap/body seams) and fill enclosed holes (e.g. the dark carton interior) of a bool mask."""
    if not m.any():
        return m
    u = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    cnts, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(u)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def static_gripper_mask(eps: list[dict], cam: str, step: int = 60, pct: float = 97.0, thr: float = 100.0) -> np.ndarray:
    """Per-pixel dark silhouette that never changes (fixed gripper fingers): p``pct`` of luma over sampled frames < ``thr``,
    opened, largest connected component touching the bottom edge."""
    gs = []
    for e in eps:
        c = e["cams"][cam]
        cap = cv2.VideoCapture(c["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(c["from_ts"] * 30)))
        for i in range(e["length"]):
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                gs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
    m = (np.percentile(np.stack(gs), pct, axis=0) < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 3000 and st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] >= m.shape[0] - 2]
    if not keep:
        raise RuntimeError("no static gripper silhouette found")
    out = np.isin(lab, keep)
    return out


def area_stats(labels: np.ndarray, ndet: np.ndarray, jump: float = 0.25) -> dict:
    n = len(labels)
    frac = np.stack([(labels == c).reshape(n, -1).mean(1) for c in (1, 2)], 1)
    d = np.abs(np.diff(frac, axis=0))
    sus = {}
    for ci, nm in ((0, "robot"), (1, "box")):
        j = np.where(d[:, ci] > jump * max(frac[:, ci].max(), 1e-3) )[0] + 1
        sus[f"{nm}_area_jump"] = j.tolist()
        sus[f"{nm}_empty"] = np.where(frac[:, ci] < 1e-4)[0].tolist()
    return {"frames": n, "robot_area_mean": float(frac[:, 0].mean()), "robot_area_min": float(frac[:, 0].min()),
            "robot_area_max": float(frac[:, 0].max()), "box_area_mean": float(frac[:, 1].mean()),
            "box_area_min": float(frac[:, 1].min()), "box_area_max": float(frac[:, 1].max()),
            "robot_empty_frac": float((frac[:, 0] < 1e-4).mean()), "box_empty_frac": float((frac[:, 1] < 1e-4).mean()),
            "suspected_failures": {k: v[:200] for k, v in sus.items()},
            "n_suspected": {k: len(v) for k, v in sus.items()}}


ROBOT_IMG_PROMPT = "robot arm"
ROBOT_IMG_THRESHOLD = 0.45
ROBOT_IMG_MIN_AREA = 1500
ROBOT_IMG_MAX_AREA_FRAC = 0.35


def robot_image_pass(sam_img: "Sam3Image", frames: np.ndarray, labels: np.ndarray, stride: int):
    """Union the image-detector 'robot arm' masks (every ``stride``-th frame, held in between) into ``labels`` as class 1.

    Needed because the video tracker often keeps only the upper arm and loses the black wrist / tool at the arm end for
    hundreds of frames (pilot), while the per-frame detector covers the whole arm.  Returns the fraction of the frame added
    per frame (n,)."""
    n = len(labels)
    added = np.zeros(n, np.float32)
    for fi in range(0, n, stride):
        m = None
        for mask, sc in sam_img.detect(frames[fi], ROBOT_IMG_PROMPT, ROBOT_IMG_THRESHOLD):
            a = int(mask.sum())
            if a < ROBOT_IMG_MIN_AREA or a > ROBOT_IMG_MAX_AREA_FRAC * mask.size:
                continue
            m = mask.copy() if m is None else (m | mask)
        if m is None:
            continue
        m = fill_holes(m, 15)
        lo, hi = max(0, fi - stride // 2), min(n, fi + (stride + 1) // 2)
        for i in range(lo, hi):
            new = m & (labels[i] != 1)
            added[i] = new.mean()
            labels[i][m] = 1
    return added


def run_robot_pass(a) -> int:
    root, out = Path(a.src), Path(a.out)
    eps = load_episodes(root)
    idx_path = out / "index.json"
    index = json.loads(idx_path.read_text())
    img_sam = Sam3Image()
    for e in eps:
        for cam in a.cameras.split(","):
            d = out / e["name"] / cam
            st = index["episodes"][e["name"]][cam]
            if st.get("robot_image_pass") or not (d / "labels.npz").exists():
                continue
            t0 = time.time()
            z = np.load(d / "labels.npz")
            labels, ndet, score = z["labels"].copy(), z["n_detections"], z["best_score"]
            frames = read_episode_frames(e["cams"][cam]["path"], e["cams"][cam]["from_ts"], len(labels))
            if not (d / "labels_pass1_video_tracker.npz").exists():
                (d / "labels.npz").rename(d / "labels_pass1_video_tracker.npz")
            added = robot_image_pass(img_sam, frames, labels, a.box_stride)
            np.savez_compressed(d / "labels.npz", labels=labels, n_detections=ndet, best_score=score)
            st2 = area_stats(labels, ndet)
            st2.update({k: st[k] for k in ("seconds_read", "seconds_sam3", "mean_best_score_robot", "mean_best_score_box", "path") if k in st})
            st2["robot_image_pass"] = {"prompt": ROBOT_IMG_PROMPT, "threshold": ROBOT_IMG_THRESHOLD, "stride": a.box_stride,
                                       "seconds": time.time() - t0, "mean_added_frac": float(added.mean()),
                                       "frames_added_gt_2pct": int((added > 0.02).sum()), "frames_added_gt_5pct": int((added > 0.05).sum()),
                                       "frames_added_gt_5pct_list": np.where(added > 0.05)[0][:200].tolist()}
            index["episodes"][e["name"]][cam] = st2
            index["gpu_seconds_total"] = index.get("gpu_seconds_total", 0.0) + (time.time() - t0)
            index["robot_image_pass"] = {"prompt": ROBOT_IMG_PROMPT, "threshold": ROBOT_IMG_THRESHOLD, "min_area": ROBOT_IMG_MIN_AREA,
                                         "max_area_frac": ROBOT_IMG_MAX_AREA_FRAC, "note": "union with the video-tracker robot mask; "
                                         "pre-pass labels kept as labels_pass1_video_tracker.npz"}
            idx_path.write_text(json.dumps(index, indent=1))
            print(f"robot pass {e['name']} {cam}: {time.time() - t0:.0f} s, mean added {added.mean():.4f}, >5%: {(added > 0.05).sum()}", flush=True)
    return 0


def main(argv=None) -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--dump-table":
        print(json.dumps(load_episodes(Path(sys.argv[2]))))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="all", help="comma list of episode indices or 'all'")
    ap.add_argument("--cameras", default="head,wrist_right")
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--stride", type=int, default=1, help="video tracker: process every n-th frame")
    ap.add_argument("--box-stride", type=int, default=2, help="box image detector: every n-th frame")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--robot-pass", action="store_true", help="second pass: union per-frame 'robot arm' image detections (head cam)")
    a = ap.parse_args(argv)
    if a.robot_pass:
        a.cameras = "head"
        return run_robot_pass(a)
    root, out = Path(a.src), Path(a.out)
    eps = load_episodes(root)
    if a.episodes != "all":
        order = [int(x) for x in a.episodes.split(",")]          # processed in the given order
        eps = [e for i in order for e in eps if e["episode_index"] == i]
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"episodes": {}}
    sam = Sam3Video() if any(VIDEO_PROMPTS[c] for c in a.cameras.split(",")) else None
    img_sam = Sam3Image()
    static = {}
    for cam in a.cameras.split(","):
        if STATIC_ROBOT.get(cam):
            static[cam] = static_gripper_mask(load_episodes(root), cam)
            (out / f"static_robot_mask_{cam}.png").write_bytes(cv2.imencode(".png", static[cam].astype(np.uint8) * 255)[1].tobytes())
    index.update({"model": MODEL_ID, "method": "transformers Sam3VideoModel, text prompts, chunked propagation",
                  "prompts": {c: {LABELS[k]: v for k, v in p.items()} for c, p in DEFAULT_PROMPTS.items()},
                  "labels": LABELS, "box_roi": BOX_ROI, "min_box_area": MIN_BOX_AREA, "static_robot_cams": list(static), "chunk": a.chunk, "stride": a.stride, "box_stride": a.box_stride, "box_prompt": BOX_PROMPT,
                  "box_threshold": BOX_THRESHOLD, "model_cfg": sam.cfg if sam else None,
                  "source": str(root)})
    gpu_s = 0.0
    for e in eps:
        for cam in a.cameras.split(","):
            d = out / e["name"] / cam
            if a.skip_done and (d / "labels.npz").exists():
                continue
            n = min(e["length"], a.max_frames) if a.max_frames else e["length"]
            t0 = time.time()
            frames = read_episode_frames(e["cams"][cam]["path"], e["cams"][cam]["from_ts"], n)
            t1 = time.time()
            labels = np.zeros(frames.shape[:3], np.uint8)
            ndet = np.zeros((n, 3), np.int16)
            score = np.zeros((n, 3), np.float32)
            if VIDEO_PROMPTS[cam]:
                for s in range(0, n, a.chunk):
                    sel = np.arange(s, min(s + a.chunk, n))
                    sub = sel[::a.stride]
                    lb, nd, sc = sam.run_chunk(frames[sub], {c: p for c, p in VIDEO_PROMPTS[cam].items()})
                    for k, fi in enumerate(sub):
                        hi = np.arange(fi, min(fi + a.stride, sel[-1] + 1))
                        labels[hi] = lb[k]
                        ndet[hi] = nd[k]
                        score[hi] = sc[k]
            # box: image detector on every box_stride-th frame, held for the frames in between
            box = np.zeros(labels.shape, bool)
            for fi in range(0, n, a.box_stride):
                m, nb, bs = select_box(img_sam.detect(frames[fi], BOX_PROMPT, BOX_THRESHOLD), BOX_ROI[cam], MIN_BOX_AREA[cam])
                hi = slice(max(0, fi - a.box_stride // 2), min(n, fi + (a.box_stride + 1) // 2))
                if m is not None:
                    box[hi] = fill_holes(m)
                ndet[hi, 2] = nb
                score[hi, 2] = bs
            labels[box & (labels == 0)] = 2
            if cam in static:
                labels[:, static[cam]] = 1
            t2 = time.time()
            gpu_s += t2 - t1
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "labels.npz", labels=labels, n_detections=ndet, best_score=score)
            st = area_stats(labels, ndet)
            st.update({"seconds_read": t1 - t0, "seconds_sam3": t2 - t1, "mean_best_score_robot": float(score[:, 1].mean()),
                       "mean_best_score_box": float(score[:, 2].mean()), "path": str(d / "labels.npz")})
            index["episodes"].setdefault(e["name"], {})[cam] = st
            index["gpu_seconds_total"] = index.get("gpu_seconds_total", 0.0) + (t2 - t1)
            idx_path.write_text(json.dumps(index, indent=1))
            print(f"{e['name']} {cam}: {n} frames, sam3 {t2 - t1:.0f} s ({n / (t2 - t1):.1f} fps), robot {st['robot_area_mean']:.3f} "
                  f"box {st['box_area_mean']:.3f} suspected {st['n_suspected']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
