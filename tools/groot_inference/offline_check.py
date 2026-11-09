#!/usr/bin/env python3
"""Offline sanity check of a running GR00T N1.7 policy server (no robot, no ROS).

For every converted LeRobot episode and several timesteps (stride), send the recorded frames
(head, wrist_left), joint state and instruction to the server over the real ZeroMQ protocol and compare
the returned 16-step chunk with the recorded actions[t..t+15].

Reported per episode and overall:
  * action key / shape / dtype of the reply
  * ABSOLUTE vs RELATIVE test: MAE of pred vs demo (absolute joint targets) and MAE of pred vs
    (demo - state) (deltas); the interpretation with the clearly lower error is the format the server returns
  * per-joint MAE (rad) over the whole chunk
  * first-step deviation |pred[0] - state[t]| (rad, per joint max and L2)
  * speed ratio |pred[15] - state| / |demo[15] - state| where the demo moves (L2 norm; L1 also printed)
  * a "hold still" baseline (predict pred[k] = state) to show whether the model beats doing nothing
  * warm round-trip latency (median / p95 over --latency-calls calls, client side incl. serialization)

Runs in the GR00T venv (needs numpy, pandas, pyarrow, pyzmq, msgpack, msgpack-numpy) or in any venv with those
(e.g. ~/venvs/lerobot_conv + `pip install pyzmq msgpack msgpack-numpy`). Frames are decoded with the system
`ffmpeg` binary (rawvideo pipe), no torchcodec needed. Example:

    ~/groot/Isaac-GR00T/.venv/bin/python tools/groot_inference/offline_check.py \\
        --dataset ~/groot/datasets/real_front_5ep --stride 20 --json-out ~/groot/logs/offline_check.json

--mocap-dataset ROOT reads the checkpoint's TRAINING data instead: the 12 single-episode LeRobot v3.0 mocap datasets
ROOT/mocap_left_* (30 fps, observation.state 14-D, left arm = state[:, :6], videos AV1 640x480 head / wrist_left / knife). Only
frames at even indices t (stride --stride, default 40) are used, because the training samples were subsampled by 2 (15 Hz):
head + wrist_left frame t (decoded with the system ffmpeg, libdav1d; already 640x480 RGB, same as the client sends) and
state[t, :6] go to the server; the 16-step chunk is compared with the demo targets state[t + 2*(k+1)], k = 0..15.
--instruction TEXT overrides the language text (default: the dataset's own task text, "open flap 1").
--wrist-source black sends a black image as wrist_left (ablation; also valid for the old dataset). Example:

    nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I tools/groot_inference/offline_check.py \\
        --mocap-dataset "<dir with mocap_left_*>" --stride 40 --instruction "open flap 1" --json-out /tmp/oc_flap1.json

The old (--dataset) mode is for the 5 lid_open episodes converted at 640x400 (arm_right, NOT this checkpoint's training data);
there --wrist-source wrist_right sends the working arm's gripper camera under the key "wrist_left" (ablation).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pandas as pd
import zmq

STATE_KEY = "single_arm"
LANGUAGE_KEY = "annotation.human.task_description"
H, W = 400, 640


class Client:
    """Minimal PolicyClient (same wire protocol as gr00t.policy.server_client / the friend's Gr00tClient)."""

    def __init__(self, host, port, timeout_ms=60000):
        self.addr = f"tcp://{host}:{port}"
        self.ctx = zmq.Context()
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        try:
            self.sock.send(msgpack.packb(req, default=mnp.encode))
            msg = self.sock.recv()
        except zmq.error.Again:
            self.sock.close()
            self._connect()
            raise RuntimeError(f"no reply from {self.addr}")
        res = msgpack.unpackb(msg, object_hook=mnp.decode, raw=False)
        if isinstance(res, dict) and "error" in res:
            raise RuntimeError(f"server error: {res['error']}")
        return res

    def get_action(self, obs):
        res = self.call("get_action", {"observation": obs, "options": None})
        return res[0] if isinstance(res, (list, tuple)) else res  # (action, info)


def read_frames(path, wanted, h=H, w=W, av1=False):
    """Decode an (w x h) video with the system ffmpeg (rawvideo pipe), keep only the wanted frame indices (RGB uint8).
    av1=True forces the libdav1d decoder (the mocap videos are AV1)."""
    wanted = set(wanted)
    last = max(wanted)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"] + (["-c:v", "libdav1d", "-threads", "4"] if av1 else []) + \
          ["-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = h * w * 3
    out = {}
    try:
        for i in range(last + 1):
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            if i in wanted:
                out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        p.kill()
        p.wait()
    return out


def make_obs(head, wrist, state, instruction):
    return {
        "video": {"head": head[None, None], "wrist_left": wrist[None, None]},  # uint8 (1,1,H,W,3)
        "state": {STATE_KEY: np.asarray(state, np.float32)[None, None]},        # float32 (1,1,6)
        "language": {LANGUAGE_KEY: [[instruction]]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(Path.home() / "groot/datasets/real_front_5ep"), help="old 5-episode lid_open dataset (640x400 mp4)")
    ap.add_argument("--mocap-dataset", default=None, help="directory holding the mocap_left_* LeRobot v3.0 datasets (the checkpoint's training data); overrides --dataset")
    ap.add_argument("--instruction", default=None, help="language text to send (default: the dataset's own task text)")
    ap.add_argument("--quiet", action="store_true", help="no per-sample lines")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all (mocap: indices into the sorted mocap_left_* list)")
    ap.add_argument("--stride", type=int, default=None, help="frame stride (default 20; mocap default 40 and must be even)")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--min-move", type=float, default=0.02, help="demo L1 move over the chunk below this (rad) = idle")
    ap.add_argument("--wrist-source", choices=["wrist_left", "wrist_right", "black"], default="wrist_left", help="wrist_right only exists in the old dataset")
    ap.add_argument("--latency-calls", type=int, default=40)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mocap = args.mocap_dataset is not None
    if args.stride is None:
        args.stride = 40 if mocap else 20
    K = args.horizon
    if mocap:
        if args.stride % 2:
            ap.error("--stride must be even in mocap mode (training samples are every 2nd frame)")
        if args.wrist_source == "wrist_right":
            ap.error("--wrist-source wrist_right does not exist in the mocap datasets")
        dirs = sorted(d for d in Path(args.mocap_dataset).glob("mocap_left_*") if d.is_dir())
        if not dirs:
            ap.error(f"no mocap_left_* directories in {args.mocap_dataset}")
        episodes = [dirs[i] for i in args.episodes] if args.episodes else dirs
        ep_names = [d.name for d in episodes]
    else:
        root = Path(args.dataset)
        eps_meta = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
        task = {e["episode_index"]: e["tasks"][0] for e in eps_meta}
        episodes = args.episodes if args.episodes else sorted(task)
        ep_names = episodes

    cli = Client(args.host, args.port)
    print("ping:", cli.call("ping"))
    try:
        mc = cli.call("get_modality_config")
        print("server modality config:", json.dumps(mc, default=str)[:900])
    except Exception as e:  # noqa: BLE001
        print("get_modality_config unavailable:", e)

    rows, all_obs, fmt_printed = [], None, False
    for ep in episodes:
        if mocap:
            # LeRobot v3.0, one episode per dataset: data/chunk-000/file-000.parquet, videos/<key>/chunk-000/file-000.mp4 (AV1, 640x480)
            name = ep.name
            df = pd.read_parquet(ep / "data/chunk-000/file-000.parquet")
            full_state = np.stack(df["observation.state"].values).astype(np.float32)
            state = full_state[:, :6]                      # arm_left_joint1..6 (cols 7..13 = right arm = zeros, col 6 = left gripper)
            text = pd.read_parquet(ep / "meta/tasks.parquet").index[0]
            n = len(df)
            ts = list(range(0, n - 2 * K, args.stride))    # state[t + 2*(K)] must exist
            demo_of = lambda t, state=state: state[t + 2 * np.arange(1, K + 1)]  # noqa: E731  targets = subsampled next states
            vid = lambda k, ep=ep: ep / f"videos/observation.images.{k}/chunk-000/file-000.mp4"  # noqa: E731
            head = read_frames(vid("head"), ts, 480, 640, av1=True)
            wrist = ({t: np.zeros((480, 640, 3), np.uint8) for t in ts} if args.wrist_source == "black"
                     else read_frames(vid("wrist_left"), ts, 480, 640, av1=True))
        else:
            name = ep
            df = pd.read_parquet(root / f"data/chunk-000/episode_{ep:06d}.parquet")
            state = np.stack(df["observation.state"].values).astype(np.float32)
            act = np.stack(df["action"].values).astype(np.float32)
            text = task[ep]
            n = len(df)
            ts = list(range(0, n - K, args.stride))
            demo_of = lambda t, act=act: act[t:t + K]  # noqa: E731  absolute joint targets
            vid = lambda k, ep=ep: root / f"videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4"  # noqa: E731
            head = read_frames(vid("head"), ts)
            if args.wrist_source == "black":
                wrist = {t: np.zeros((H, W, 3), np.uint8) for t in ts}
            else:
                wrist = read_frames(vid(args.wrist_source), ts)
        miss = [t for t in ts if t not in head or t not in wrist]
        if miss:
            raise RuntimeError(f"{name}: {len(miss)} frames missing from the decoded videos (first t={miss[0]}); video shorter than the data?")
        if args.instruction is not None:
            text = args.instruction
        print(f"\n=== episode {name} ({n} frames, {len(ts)} samples, instruction {text!r}, wrist key sent as wrist_left <- {args.wrist_source}) ===")
        if not args.quiet:
            print("    t  demo|d16|  pred|d16|  ratio   first-step L2  MAEabs  MAErel  holdMAE   cos  signOK")
        for t in ts:
            obs = make_obs(head[t], wrist[t], state[t], text)
            out = cli.get_action(obs)
            if not fmt_printed:
                print("reply keys:", {k: (np.asarray(v).shape, str(np.asarray(v).dtype)) for k, v in out.items()})
                fmt_printed = True
            pred = np.asarray(out[STATE_KEY])[0][:K]          # (K,6)
            demo = demo_of(t)                                   # absolute joint targets (K,6)
            demo_rel = demo - state[t]
            mae_abs = float(np.abs(pred - demo).mean())         # pred interpreted as absolute
            mae_rel = float(np.abs(pred - demo_rel).mean())     # pred interpreted as delta from state
            hold = float(np.abs(state[t] - demo).mean())        # do-nothing baseline (absolute)
            d_pred = pred[K - 1] - state[t]
            d_demo = demo[K - 1] - state[t]
            l1_demo = float(np.abs(d_demo).sum())
            moving = l1_demo >= args.min_move
            ratio_l2 = float(np.linalg.norm(d_pred) / np.linalg.norm(d_demo)) if moving else float("nan")
            ratio_l1 = float(np.abs(d_pred).sum() / l1_demo) if moving else float("nan")
            first = float(np.linalg.norm(pred[0] - state[t]))
            # direction agreement of the 16-step displacement (only meaningful when the demo moves)
            cos = float(d_pred @ d_demo / (np.linalg.norm(d_pred) * np.linalg.norm(d_demo) + 1e-9)) if moving else float("nan")
            big = np.abs(d_demo) > 0.01
            sign_ok = float(np.mean(np.sign(d_pred[big]) == np.sign(d_demo[big]))) if moving and big.any() else float("nan")
            rows.append(dict(ep=name, t=t, moving=moving, mae_abs=mae_abs, mae_rel=mae_rel, hold_mae=hold,
                             per_joint_mae=np.abs(pred - demo).mean(0).tolist(),
                             first_step_l2=first, first_step_joint_max=float(np.abs(pred[0] - state[t]).max()),
                             demo_first_step_l2=float(np.linalg.norm(demo[0] - state[t])),
                             ratio_l2=ratio_l2, ratio_l1=ratio_l1, cos=cos, sign_ok=sign_ok,
                             d_pred=d_pred.tolist(), d_demo=d_demo.tolist(),
                             d_demo_l2=float(np.linalg.norm(d_demo)), d_pred_l2=float(np.linalg.norm(d_pred))))
            if not args.quiet:
                print(f"{t:5d}  {np.linalg.norm(d_demo):9.4f}  {np.linalg.norm(d_pred):9.4f}  "
                      f"{('%5.2f' % ratio_l2) if moving else ' idle'}  {first:12.4f}  {mae_abs:6.4f}  {mae_rel:6.4f}  {hold:7.4f}  {cos:5.2f}  {sign_ok:5.2f}")
        # keep one observation for the latency test
        if all_obs is None:
            all_obs = make_obs(head[ts[len(ts) // 2]], wrist[ts[len(ts) // 2]], state[ts[len(ts) // 2]], text)

    # ---------------- summary ----------------
    def pjc(r):
        m = [x for x in r if x["moving"]]
        if len(m) < 3:
            return None
        dp, dd = np.array([x["d_pred"] for x in m]), np.array([x["d_demo"] for x in m])
        with np.errstate(all="ignore"):
            return [float(np.corrcoef(dp[:, j], dd[:, j])[0, 1]) for j in range(dp.shape[1])]

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return None
        mv = [x["ratio_l2"] for x in r if x["moving"]]
        mv1 = [x["ratio_l1"] for x in r if x["moving"]]
        return dict(n=len(r), n_moving=len(mv),
                    mae_abs=float(np.mean([x["mae_abs"] for x in r])),
                    mae_rel=float(np.mean([x["mae_rel"] for x in r])),
                    hold_mae=float(np.mean([x["hold_mae"] for x in r])),
                    per_joint_mae=np.mean([x["per_joint_mae"] for x in r], 0).round(4).tolist(),
                    first_step_l2_mean=float(np.mean([x["first_step_l2"] for x in r])),
                    first_step_l2_max=float(np.max([x["first_step_l2"] for x in r])),
                    demo_first_step_l2_mean=float(np.mean([x["demo_first_step_l2"] for x in r])),
                    ratio_l2_median=float(np.median(mv)) if mv else None,
                    ratio_l2_p25_p75=[float(np.percentile(mv, 25)), float(np.percentile(mv, 75))] if mv else None,
                    ratio_l1_median=float(np.median(mv1)) if mv1 else None,
                    per_joint_corr_d16=pjc(r),
                    cos_mean=float(np.nanmean([x["cos"] for x in r])) if mv else None,
                    sign_agree_mean=float(np.nanmean([x["sign_ok"] for x in r])) if mv else None)

    summary = {"per_episode": {ep: agg(lambda x, ep=ep: x["ep"] == ep) for ep in ep_names}, "all": agg(lambda x: True)}
    print("\n================ SUMMARY ================")
    for ep in ep_names:
        s = summary["per_episode"][ep]
        print(f"ep{ep}: n={s['n']} (moving {s['n_moving']})  MAE abs-interp {s['mae_abs']:.4f} rad  rel-interp {s['mae_rel']:.4f}  "
              f"hold-still {s['hold_mae']:.4f}  first-step L2 mean {s['first_step_l2_mean']:.4f} (demo {s['demo_first_step_l2_mean']:.4f}) "
              f"max {s['first_step_l2_max']:.4f}  speed ratio median {s['ratio_l2_median']:.2f}  cos {s['cos_mean']:.2f}  sign-agree {s['sign_agree_mean']:.2f}")
        print(f"      per-joint MAE (rad): {s['per_joint_mae']}  per-joint corr d16: {None if s['per_joint_corr_d16'] is None else np.round(s['per_joint_corr_d16'], 2).tolist()}")
    a = summary["all"]
    print(f"ALL : n={a['n']} (moving {a['n_moving']})  MAE abs-interp {a['mae_abs']:.4f}  rel-interp {a['mae_rel']:.4f}  hold-still {a['hold_mae']:.4f}")
    print(f"      per-joint MAE (rad): {a['per_joint_mae']}")
    print(f"      first-step L2 mean {a['first_step_l2_mean']:.4f} max {a['first_step_l2_max']:.4f} (demo first-step mean {a['demo_first_step_l2_mean']:.4f})")
    print(f"      speed ratio L2 median {a['ratio_l2_median']:.3f} (IQR {a['ratio_l2_p25_p75']}), L1 median {a['ratio_l1_median']:.3f}")
    print(f"      direction of 16-step displacement vs demo: cosine mean {a['cos_mean']:.2f}, per-joint sign agreement {a['sign_agree_mean']:.2f} (0.5 = chance)")
    mv = [x for x in rows if x["moving"]]
    dp, dd = np.array([x["d_pred"] for x in mv]), np.array([x["d_demo"] for x in mv])
    pj_corr = [float(np.corrcoef(dp[:, j], dd[:, j])[0, 1]) for j in range(dp.shape[1])]
    pj_sign = [float(np.mean(np.sign(dp[np.abs(dd[:, j]) > 0.01, j]) == np.sign(dd[np.abs(dd[:, j]) > 0.01, j]))) for j in range(dp.shape[1])]
    pj_ratio = (np.abs(dp).mean(0) / (np.abs(dd).mean(0) + 1e-9)).tolist()
    summary["per_joint_corr_d16"], summary["per_joint_sign_agree"], summary["per_joint_mean_abs_ratio"] = pj_corr, pj_sign, pj_ratio
    print("      per-joint Pearson corr of 16-step displacement pred vs demo (j1..j6):", np.round(pj_corr, 2).tolist())
    print("      per-joint sign agreement (j1..j6):", np.round(pj_sign, 2).tolist(), " mean|pred|/mean|demo|:", np.round(pj_ratio, 2).tolist())
    verdict = "ABSOLUTE joint targets" if a["mae_abs"] < a["mae_rel"] else "RELATIVE deltas (from current state)"
    print(f"      FORMAT VERDICT: {verdict}  (MAE as-absolute {a['mae_abs']:.4f} vs as-delta {a['mae_rel']:.4f}; lower = right)")

    # ---------------- latency ----------------
    for _ in range(3):
        cli.get_action(all_obs)
    lat, reps = [], []
    for _ in range(args.latency_calls):
        t0 = time.perf_counter()
        reps.append(np.asarray(cli.get_action(all_obs)[STATE_KEY])[0])
        lat.append((time.perf_counter() - t0) * 1000)
    if not lat:
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(dict(args=vars(args), summary=summary, rows=rows), indent=1))
        return 0
    reps = np.stack(reps)  # same observation repeated: the diffusion head samples fresh noise each call
    summary["repeat_std_rad"] = float(reps.std(0).mean())
    print(f"SAMPLING NOISE: identical observation x{len(reps)} -> mean per-element std of the chunk {reps.std(0).mean():.4f} rad "
          f"(step 15 std {reps.std(0)[-1].mean():.4f} rad)")
    lat = np.array(lat)
    summary["latency_ms"] = dict(n=len(lat), median=float(np.median(lat)), p95=float(np.percentile(lat, 95)),
                                 min=float(lat.min()), max=float(lat.max()))
    print(f"LATENCY (warm, client round trip, n={len(lat)}): median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms, "
          f"min {lat.min():.1f}, max {lat.max():.1f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(dict(args=vars(args), summary=summary, rows=rows), indent=1))
        print("wrote", args.json_out)


if __name__ == "__main__":
    sys.exit(main())
