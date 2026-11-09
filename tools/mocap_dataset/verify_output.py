#!/usr/bin/env python3
"""Check a dataset written by preprocess.py with lerobot's own loader, against the untouched source.

    python3 tools/mocap_dataset/verify_output.py OUTDIR SOURCE_ROOT [--repo-id local/NAME]
    python3 tools/mocap_dataset/verify_output.py --check-loader SOURCE_ROOT     # v2.1 reader vs. a direct parquet read

* ``LeRobotDataset(repo_id, root=OUTDIR)`` loads; len / num_episodes / stats keys (with quantiles) are right.
* Items at the start, middle and end of every episode (state, action, decoded frames of all videos).
* ``timestamp`` is exactly frame_index / fps (contiguous, also after pause trimming), ``source_frame_index`` /
  ``source_timestamp`` point at the source frames that were kept.
* Frame alignment.  Stream-copied videos: the decoded frame the loader returns for episode frame k equals (to decode noise)
  the frame the source video has at source frame ``keep[k]``, and is clearly different from the frame 5 frames later.
  Re-encoded videos (v2.1 source, pause trimming): the loader's frame k is compared with the source frames
  ``keep[k] - 6 .. keep[k] + 6`` (first / second / middle / last frames and both sides of every cut); the true source frame
  must be the best match (within 0.25/255, or in the same / the adjacent run of identical source frames: the cameras repeat
  frames) whenever the window can tell neighbouring frames apart (spread >= 1.5/255; a still scene cannot); its PSNR is reported.
* state_raw / action_teleop equal the source columns at the kept frames; action[t] == state[t + offset] (last frames repeat);
  state endpoints equal the raw endpoints (untrimmed); max |state - state_raw| < 1 deg; no lag (zero-phase).
Returns True/False and prints a line per check; exit code 1 if anything failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lerobot_episode import list_episodes, load_episode  # noqa: E402
from video_frames import count_packets, read_frames, read_frames_by_index  # noqa: E402


def _keep_indices(e: dict) -> np.ndarray:
    """Source frame indices of the output frames of one PREPROCESS.json episode record."""
    if e.get("keep_ranges"):
        return np.concatenate([np.arange(a, b) for a, b in e["keep_ranges"]]).astype(np.int64)
    i0, i1 = e["trim_frames"]
    return np.arange(i0, i1, dtype=np.int64)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def verify(out: Path, source_root: Path, repo_id: str, max_dev_deg: float = 1.0, psnr_samples: int = 0) -> bool:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = Path(out).expanduser().resolve()
    prep = json.loads((out / "PREPROCESS.json").read_text())
    par = prep["parameters"]
    fps = par["fps"]
    reenc = par.get("video_mode", "copy") == "reencode"
    orig_name = par.get("orig_action_name", "action_teleop")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")

    ds = LeRobotDataset(repo_id, root=out)
    eps = prep["episodes"]
    total = sum(e["frames_out"] for e in eps)
    check("len(ds) == total frames", len(ds) == total, f"({len(ds)} vs {total})")
    check("num_episodes", ds.num_episodes == len(eps), f"({ds.num_episodes})")
    check("fps", ds.fps == fps)
    st = ds.meta.stats
    check("stats keys and quantiles", all(k in st["observation.state"] for k in ("min", "max", "mean", "std", "count", "q01", "q99")))
    names = ds.meta.features["observation.state"]["names"]
    cols = [i for i, n in enumerate(names) if "_joint" in n]
    vkeys = ds.meta.video_keys
    print(f"  videos: {vkeys}, backend {getattr(ds, 'video_backend', '?')}, mode {'re-encoded' if reenc else 'stream copy'}")

    off = par["action_offset_frames"]
    worst_diff, worst_neighbor = 0.0, 1e9
    psnrs, margins = [], []
    psnr_cam: dict[str, list[float]] = {}
    n_frame_checks = 0
    n_informative = 0
    for e in eps:
        idx = e["episode_index"]
        src = load_episode(e["source_path"], e.get("source_episode_index", 0))
        keep = _keep_indices(e)
        n = e["frames_out"]
        check(f"{e['source']} -> ep {idx}: {n} frames", len(keep) == n)
        a = ds.meta.episodes[idx]
        start = int(a["dataset_from_index"])
        data = ds.hf_dataset.select(range(start, start + n)).with_format("numpy")
        S, A = np.asarray(data["observation.state"]), np.asarray(data["action"])
        R, T = np.asarray(data["observation.state_raw"]), np.asarray(data[orig_name])
        sel = [src.names.index(nm) for nm in names]
        check("   state_raw/original action == source at kept frames", np.array_equal(R, src.state[keep][:, sel].astype(np.float32))
              and np.array_equal(T, src.action[keep][:, sel].astype(np.float32)))
        ts = np.asarray(data["timestamp"], dtype=np.float64)
        check("   timestamp == frame_index / fps (contiguous)", np.allclose(ts, np.arange(n) / fps, atol=1e-6)
              and np.array_equal(np.asarray(data["frame_index"]), np.arange(n)))
        check("   source_frame_index / source_timestamp", np.array_equal(np.asarray(data["source_frame_index"]), keep)
              and np.array_equal(np.asarray(data["source_timestamp"]), src.timestamp[keep].astype(np.float32)))
        if "box_pose" in e.get("source_meta", {}):
            bp = np.asarray(ds.meta.episodes[idx]["box_pose"], dtype=np.float64)
            check("   box_pose carried", np.allclose(bp, e["source_meta"]["box_pose"]))
        # shifted action: next KEPT frame's pose (a plain idle trim looks beyond its range, so skip it there)
        if not par["trim_idle"]:
            exp = S[np.minimum(np.arange(n) + off, n - 1)]
            nonj = [i for i, nm in enumerate(names) if "_joint" not in nm]
            check("   action == state[t+offset] of the kept frames, grippers 0",
                  np.array_equal(A[:, cols], exp[:, cols]) and not (np.abs(A[:, nonj]).any() if nonj else False))
            if not par.get("trim_pauses"):
                check("   endpoints preserved", np.allclose(S[0], R[0], atol=1e-7) and np.allclose(S[-1], R[-1], atol=1e-7))
        dev = float(np.degrees(np.abs(S[:, cols] - R[:, cols]).max()))
        check("   max |state - raw| < limit", dev < max_dev_deg, f"({dev:.3f} deg)")
        if par["sigma_s"] > 0:
            def lag_err(L):   # mean |S[t] - R[t - L]| over the overlap; the minimum at L = 0 means no time shift
                x, y = (S[L:, cols], R[:n - L, cols]) if L >= 0 else (S[:n + L, cols], R[-L:, cols])
                return float(np.abs(x - y).mean())
            lags = {L: lag_err(L) for L in range(-3, 4)}
            check("   zero-phase (best lag 0)", min(lags, key=lags.get) == 0)
        if par.get("trim_pauses"):
            pt = e["pause_trim"]
            step = np.degrees(np.abs(np.diff(S[:, cols], axis=0)).max()) if n > 1 else 0.0
            check("   pause cuts do not add motion", step <= pt["max_step_deg_before"] + 0.05,
                  f"(max step {step:.3f} deg, source {pt['max_step_deg_before']:.3f}; {len(pt['cuts'])} cuts, "
                  f"-{pt['removed_frames']} frames, largest join {pt['max_join_step_deg']:.3f} deg)")
        # items + frames
        gaps = np.flatnonzero(np.diff(keep) != 1)
        ks = sorted({0, 1, n // 2, n - 2, n - 1} | {int(g) for g in gaps} | {int(g) + 1 for g in gaps})
        if reenc and psnr_samples:
            ks = sorted(set(ks) | set(np.linspace(0, n - 1, psnr_samples).round().astype(int).tolist()))
        ks = [k for k in ks if 0 <= k < n]
        items = {k: ds[start + k] for k in ks}
        for k, item in items.items():
            check_item = (int(item["episode_index"]) == idx and int(item["frame_index"]) == k
                          and item["observation.state"].shape[0] == len(names) and item["action"].shape[0] == len(names)
                          and np.allclose(item["observation.state"].numpy(), S[k]))
            check_item = check_item and item["task"] == e["task_out"]
            ok &= bool(check_item)
            if not check_item:
                print(f"  [FAIL] item {k} metadata/state mismatch")
        for vk in vkeys:
            short = vk.split(".")[-1]
            path, from_ts = src.videos[vk]
            if not reenc:
                for k in ks:
                    img = items[k][vk].numpy()                         # (3, H, W) float [0, 1]
                    j = int(keep[k])
                    jn = j + 5 if j + 5 < e["frames_source"] else j - 5     # a different frame, for contrast
                    ref, nb = read_frames(path, [from_ts + j / fps, from_ts + jn / fps], fps)
                    ref = ref.transpose(2, 0, 1).astype(np.float32) / 255.0
                    nb = nb.transpose(2, 0, 1).astype(np.float32) / 255.0
                    d, dn = float(np.abs(img - ref).mean() * 255), float(np.abs(img - nb).mean() * 255)
                    worst_diff = max(worst_diff, d)
                    worst_neighbor = min(worst_neighbor, dn)
                    n_frame_checks += 1
                    if d > 1.0 or dn <= d:
                        ok = False
                        print(f"  [FAIL] {e['source']} frame {k} {short}: diff to source {d:.3f}, to +5 frames {dn:.3f}")
            else:
                win = 6
                need = {int(keep[k]) + o for k in ks for o in range(-win, win + 1)}
                need = {i for i in need if 0 <= i < e["frames_source"]}
                srcf = read_frames_by_index(path, sorted(need), fps, from_ts)
                for k in ks:
                    img = (items[k][vk].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                    j = int(keep[k])
                    diffs = {i: float(np.abs(img.astype(np.float32) - srcf[i].astype(np.float32)).mean())
                             for i in range(j - win, j + win + 1) if i in srcf}
                    d_true = diffs[j]
                    best = min(diffs, key=diffs.get)
                    psnrs.append(_psnr(img, srcf[j]))
                    psnr_cam.setdefault(short, []).append(psnrs[-1])
                    worst_diff = max(worst_diff, d_true)
                    spread = max(diffs.values()) - min(diffs.values())
                    margins.append(spread)
                    n_frame_checks += 1
                    informative = spread >= 1.5               # a still scene (sensor noise only) cannot tell neighbouring frames apart
                    n_informative += informative
                    # the source repeats frames (stale camera samples): group identical neighbours into runs, and accept a best
                    # match in the true frame's run or the adjacent run (a codec error between two similar frames), never further
                    run_id, rid = {}, 0
                    ids = sorted(diffs)
                    for a_, b_ in zip(ids, ids[1:]):
                        run_id[a_] = rid
                        if float(np.abs(srcf[b_].astype(np.float32) - srcf[a_].astype(np.float32)).mean()) > 0.05:
                            rid += 1
                    run_id[ids[-1]] = rid
                    near_best = abs(run_id[best] - run_id[j]) <= 1
                    if d_true > 3.0 or (informative and d_true > diffs[best] + 0.25 and not near_best):
                        ok = False
                        print(f"  [FAIL] {e['source']} frame {k} {short}: diff to source frame {j} = {d_true:.3f}, "
                              f"best match is source frame {best} ({diffs[best]:.3f})")
    if reenc:
        check("decoded frames == source frames at the recorded source index", worst_diff <= 3.0 and bool(psnrs),
              f"({n_frame_checks} frames, {n_informative} of them with a window that tells neighbouring frames apart (true frame is the "
              f"best match there); worst mean abs diff {worst_diff:.3f}/255; PSNR mean {np.mean(psnrs):.1f} dB, min {np.min(psnrs):.1f} dB)")
        for cam, v in sorted(psnr_cam.items()):
            print(f"      PSNR {cam}: mean {np.mean(v):.1f} dB, min {np.min(v):.1f} dB, {len(v)} frames")
    else:
        check("decoded frames == source frames at the same t", worst_diff <= 1.0,
              f"({n_frame_checks} frames; worst mean abs diff {worst_diff:.3f}/255; smallest diff to the frame 5 later {worst_neighbor:.2f}/255)")
    print("VERIFY", "PASSED" if ok else "FAILED")
    return ok


def check_loader(source_root: Path) -> bool:
    """The pyarrow-only reader (lerobot_episode.load_episode) against an independent read of the same files with pandas:
    numeric columns identical, frame counts equal parquet rows for every video."""
    import pandas as pd
    ok = True
    for r in list_episodes(source_root):
        ep = load_episode(r.root, r.episode_index)
        info = json.loads((r.root / "meta" / "info.json").read_text())
        if str(info["codebase_version"]).startswith("v2"):
            f = r.root / info["data_path"].format(episode_chunk=r.episode_index // info.get("chunks_size", 1000),
                                                  episode_index=r.episode_index)
            df = pd.read_parquet(f)
        else:
            continue
        S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        A = np.stack(df["action"].to_numpy()).astype(np.float64)
        good = (np.array_equal(S, ep.state) and np.array_equal(A, ep.action)
                and np.array_equal(df["timestamp"].to_numpy().astype(np.float64), ep.timestamp)
                and np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))))
        for k, v in ep.extras.items():
            good &= bool(np.array_equal(df[k].to_numpy(), v))
        vids = {k.split(".")[-1]: count_packets(p) for k, (p, _) in ep.videos.items()}
        good &= all(v == len(df) for v in vids.values())
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {r.name}: {len(df)} rows, state/action/timestamp/extras identical, video packets {vids}")
    print("LOADER CHECK", "PASSED" if ok else "FAILED")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="?", help="dataset written by preprocess.py")
    ap.add_argument("source_root", type=Path, nargs="?")
    ap.add_argument("--repo-id")
    ap.add_argument("--psnr-samples", type=int, default=0, help="re-encoded videos: extra evenly spaced frames per video for the PSNR statistics")
    ap.add_argument("--check-loader", type=Path, metavar="SOURCE_ROOT",
                    help="only check the v2.1 reader of lerobot_episode.py against pandas on SOURCE_ROOT")
    a = ap.parse_args(argv)
    if a.check_loader:
        return 0 if check_loader(a.check_loader) else 1
    if not a.out or not a.source_root:
        ap.error("OUT and SOURCE_ROOT are required (or --check-loader SOURCE_ROOT)")
    return 0 if verify(a.out, a.source_root, a.repo_id or f"local/{a.out.resolve().name}", psnr_samples=a.psnr_samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
