#!/usr/bin/env python3
"""Offline check of the TurboVLA policy server on recorded demonstrations (client side, no ROS), LEFT or RIGHT arm.

For every episode: decode the head + wrist_left video, resize like training (PIL
Image.fromarray(rgb).resize((224,224)), see experiments/compal/mocap_images.py), send the recorded 14-D
state + both images for each frame to the server and compare with the demo.

Arm / data (--arm auto|left|right, auto = the arm_channels announced by the server, [0, 6] if it announces none):
  left   raw mocap episodes (mocap_left_*/{data,videos}/chunk-000/file-000.*), default data root = the Downloads copy of the 12 episodes,
         held-out/train split from the bundle's models/mocap10k/split.json
  right  the LeRobot-v2.1 validation dataset the right19 run was evaluated on ($T/right19/data/training-mp4/Compal/
         rebel_right_frontflap_validation_v1: episode_00000N.parquet + 1152x648 h264 videos = the 2 held-out episodes real_0005, real_0014,
         names taken from the run's split.json); pass --data-root .../rebel_right_frontflap_train_v1 for the optimised episodes.
         These are the exact (pause-trimmed) samples of the training evaluation, so the raw metric below is comparable to it.

Training target definition (experiments/compal/data_registry/mocap_config.py + mocap_dataset.py):
chunk step k of the sample at frame t is the recorded `action` column at frame t+k (action delta indices
0..29, state delta index 0, state_action_shift_frames = 0). Hold-still baseline = recorded state[t] repeated.
Metrics (the controlled arm's 6 joints, radians):
  raw       mean |chunk[t,k,arm] - action[t+k,arm]| over all valid (t,k)  (also k=0 and k=29).  = `all_valid_targets.mae_rad` of the
            training evaluation (experiments/compal/mocap_validation.py) when whole episodes are run
  raw_padded the same with the training loader's repeat-last padding of the targets beyond the episode end (all n*30 windows) =
            the `*_mae_rad` / `logged_padded_mae_rad` number in $T/right19/runs/<run>/evaluations/steps_<N>.json
  first_step per frame max_j |chunk[t,0,j] - state[t,j]| (what the live client's first_step gate looks at): median, p95 and the fraction
            of frames at or under --first-step-gate (default 0.05 rad)
  ensemble  ACT temporal ensemble (ensemble_recorded_chunks, horizon 30, k=0.01) of the stride-1 chunk
            sequence from frame 0 vs action[t,:6]
  hold      the same two comparisons with state[t,:6] instead of the prediction (raw: state[t] vs action[t+k];
            ensemble: state[t] vs action[t])
  dir_corr  per-joint Pearson correlation between predicted and demo displacement from the current state,
            (chunk[t,k,j]-state[t,j]) vs (action[t+k,j]-state[t,j]) over all valid (t,k)
Raw episodes are NOT pause-compressed (training used the stagnant-span-reduced copy), so hold-still is
easier here than in training and absolute numbers are not the training validation numbers.

Run with the bundle venv python and PYTHONPATH=<server_deps> (pyzmq, msgpack, msgpack-numpy, pyarrow, av), at low priority:
  PYTHONPATH=~/turbovla/server_deps:source nice -n 19 .venv/bin/python offline_check_turbovla.py --arm right --black-frames -1
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turbovla_server import MsgSerializer  # noqa: E402  (wire format shared with the server)

DEFAULT_BUNDLE = Path.home()/'turbovla/compal-turbovla-handoff-v1'
TV_HOME = Path.home()/'turbovla'
DEFAULT_DATA = {
    'left': Path('/home/farhan/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN'),
    'right': TV_HOME/'right19/data/training-mp4/Compal/rebel_right_frontflap_validation_v1'}
DEFAULT_RUN = {'right': TV_HOME/'right19/runs/right19-10k'}   # left: the bundle's models/mocap10k
ARM_CHANNELS = {'left': (0, 6), 'right': (7, 13)}
CAMS = ('head', 'wrist_left')


class Client:
    def __init__(self, host, port, timeout_ms=30000):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context()
        self.timeout_ms, self.addr = timeout_ms, f'tcp://{host}:{port}'
        self._open()

    def _open(self):
        zmq = self.zmq
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms); self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, endpoint, data=None):
        request = {'endpoint': endpoint}
        if data is not None:
            request['data'] = data
        try:
            self.sock.send(MsgSerializer.to_bytes(request))
            reply = MsgSerializer.from_bytes(self.sock.recv())
        except self.zmq.error.Again:
            self.sock.close(); self._open(); raise
        if isinstance(reply, dict) and 'error' in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return reply


def find_episodes(root):
    """-> {name: dict(parquet, video(cam)->Path)} for both layouts (raw mocap folders, LeRobot v2.1 dataset)."""
    out = {}
    if (root/'meta/info.json').is_file():   # LeRobot v2.1 (right19 training-mp4 datasets)
        for line in (root/'meta/episodes.jsonl').read_text().splitlines():
            index = json.loads(line)['episode_index']
            out[f'episode_{index:06d}'] = dict(
                parquet=root/f'data/chunk-000/episode_{index:06d}.parquet',
                video=lambda cam, index=index: root/f'videos/chunk-000/observation.images.{cam}/episode_{index:06d}.mp4')
    else:                                    # raw mocap episode folders
        for ep in sorted(p for p in root.glob('mocap_*') if p.is_dir()):
            out[ep.name] = dict(parquet=ep/'data/chunk-000/file-000.parquet',
                                video=lambda cam, ep=ep: ep/f'videos/observation.images.{cam}/chunk-000/file-000.mp4')
    return out


def decode_frames(path, n_frames):
    """Yield 224x224x3 uint8 frames: pyav rgb24 decode + PIL resize, exactly like the training loader."""
    import av
    from PIL import Image
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 2
        count = 0
        for frame in container.decode(video=0):
            yield np.asarray(Image.fromarray(frame.to_ndarray(format='rgb24')).resize((224, 224)))
            count += 1
            if count >= n_frames:
                return


def load_episode(parquet):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet)
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
    action = np.asarray(table['action'].to_pylist(), dtype=np.float64)
    return state, action


def run_episode(client, ep, name, n, variant, log_every=500):
    state, action = load_episode(ep['parquet'])
    n = min(n, len(state)) if n > 0 else len(state)
    gens = {cam: decode_frames(ep['video'](cam), n) for cam in CAMS}
    chunks, latency = np.empty((n, 30, 14), np.float32), np.empty(n)
    for t in range(n):
        head, wrist = next(gens['head']), next(gens['wrist_left'])
        if variant == 'black_wrist':
            wrist = np.zeros_like(wrist)
        reply = client.call('get_action', {'observation': {'state': state[t], 'head': head, 'wrist_left': wrist, 'step': t},
                                           'options': None})
        chunks[t], latency[t] = reply[0]['actions'], reply[1]['inference_ms']
        if t % log_every == 0:
            print(f'  {name} {variant} {t}/{n} {latency[t]:.1f} ms', flush=True)
    return state[:n], action[:n], chunks, latency


def metrics(state, action, chunks, arm, decay=0.01, first_step_gate=0.05):
    from experiments.compal.act_temporal import ensemble_recorded_chunks
    n, H = len(chunks), chunks.shape[1]
    L = slice(*arm)
    # raw chunk vs action[t+k]
    err, herr, pd, dd = [], [], [], []
    for k in range(H):
        valid = n - k  # frames t with t+k < n (demo window)
        tgt = action[k:k+valid, L]
        err.append(np.abs(chunks[:valid, k, L] - tgt))
        herr.append(np.abs(state[:valid, L] - tgt))
        pd.append(chunks[:valid, k, L].astype(np.float64) - state[:valid, L])
        dd.append(tgt - state[:valid, L])
    allerr, allherr = np.concatenate(err), np.concatenate(herr)
    # training-eval flavour: targets beyond the episode end repeat the last action (all n*H windows)
    idx = np.minimum(np.arange(n)[:, None] + np.arange(H), n-1)
    padded = np.abs(chunks[:, :, L].astype(np.float64) - action[idx][:, :, L])
    first = np.abs(chunks[:, 0, L].astype(np.float64) - state[:, L]).max(axis=1)   # what the live first_step gate sees
    pd, dd = np.concatenate(pd), np.concatenate(dd)
    corr = [float(np.corrcoef(pd[:, j], dd[:, j])[0, 1]) for j in range(6)]
    # sign agreement where the demo moves more than 0.02 rad away from the current state
    moving = np.abs(dd) > 0.02
    sign = [float((np.sign(pd[:, j][moving[:, j]]) == np.sign(dd[:, j][moving[:, j]])).mean()) if moving[:, j].any() else float('nan')
            for j in range(6)]
    ens = ensemble_recorded_chunks(chunks.astype(np.float64), np.zeros(n, dtype=int), decay=decay)
    ens_err = np.abs(ens[:, L] - action[:, L]); ens_h = np.abs(state[:, L] - action[:, L])
    return {'frames': int(n),
            'raw_mae': float(allerr.mean()), 'raw_mae_k0': float(err[0].mean()), 'raw_mae_k29': float(err[-1].mean()),
            'raw_mae_per_joint': allerr.mean(0).tolist(),
            'raw_mae_padded': float(padded.mean()),
            'first_step_max_median': float(np.median(first)), 'first_step_max_p95': float(np.percentile(first, 95)),
            'first_step_gate_pass_frac': float((first <= first_step_gate).mean()), 'first_step_gate': first_step_gate,
            'hold_raw_mae': float(allherr.mean()), 'hold_raw_mae_k0': float(herr[0].mean()), 'hold_raw_mae_k29': float(herr[-1].mean()),
            'hold_raw_mae_per_joint': allherr.mean(0).tolist(),
            'ens_mae': float(ens_err.mean()), 'ens_mae_per_joint': ens_err.mean(0).tolist(),
            'hold_ens_mae': float(ens_h.mean()), 'hold_ens_mae_per_joint': ens_h.mean(0).tolist(),
            'dir_corr_per_joint': corr, 'sign_agreement_per_joint': sign}, (pd, dd, allerr, allherr, ens_err, ens_h, padded, first)


def pooled(parts):
    pd = np.concatenate([p[0] for p in parts]); dd = np.concatenate([p[1] for p in parts])
    out = {'raw_mae': float(np.concatenate([p[2] for p in parts]).mean()),
           'raw_mae_padded': float(np.concatenate([p[6].reshape(-1, 6) for p in parts]).mean()),
           'first_step_max_median': float(np.median(np.concatenate([p[7] for p in parts]))),
           'first_step_max_p95': float(np.percentile(np.concatenate([p[7] for p in parts]), 95)),
           'hold_raw_mae': float(np.concatenate([p[3] for p in parts]).mean()),
           'ens_mae': float(np.concatenate([p[4] for p in parts]).mean()),
           'hold_ens_mae': float(np.concatenate([p[5] for p in parts]).mean()),
           'raw_mae_per_joint': np.concatenate([p[2] for p in parts]).mean(0).tolist(),
           'ens_mae_per_joint': np.concatenate([p[4] for p in parts]).mean(0).tolist(),
           'dir_corr_per_joint': [float(np.corrcoef(pd[:, j], dd[:, j])[0, 1]) for j in range(6)]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument('--arm', choices=['auto', 'left', 'right'], default='auto',
                    help="arm whose 6 joints are scored; auto = the server's announced arm_channels (left if none)")
    ap.add_argument('--run-dir', type=Path, default=None, help='model run directory holding split.json (default: right19-10k for the right arm, the bundle model for the left)')
    ap.add_argument('--data-root', type=Path, default=None, help='default: the Downloads mocap folders (left) / the right19 validation dataset (right)')
    ap.add_argument('--first-step-gate', type=float, default=0.05, help='rad, the live client gate limit used for the pass fraction')
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--frames', type=int, default=0, help='frames per episode for the normal variant (0 = whole episode)')
    ap.add_argument('--black-frames', type=int, default=0, help='frames per episode for the black_wrist variant (0 = whole episode, -1 = skip)')
    ap.add_argument('--episodes', nargs='*', default=None, help='episode names as listed on start (default: all in the data root)')
    ap.add_argument('--out', type=Path, default=None, help='default ~/turbovla/logs/offline_check (left) / offline_check_right (right)')
    ap.add_argument('--reuse', action='store_true', help='reuse cached chunk files in --out instead of querying the server')
    args = ap.parse_args()
    sys.path.insert(0, str(args.bundle/'source'))
    client = None if args.reuse else Client(args.host, args.port)
    announced = None
    if client:
        modality = client.call('get_modality_config')
        print('modality config:', json.dumps(modality), flush=True)
        announced = tuple(modality.get('arm_channels', (0, 6)))
    arm_name = args.arm
    if arm_name == 'auto':
        arm_name = next((n for n, c in ARM_CHANNELS.items() if c == announced), None) if announced else 'left'
        if arm_name is None:
            raise SystemExit(f'server announces unknown arm_channels {announced}')
    elif announced is not None and ARM_CHANNELS[arm_name] != announced:
        raise SystemExit(f'--arm {arm_name} {ARM_CHANNELS[arm_name]} != server arm_channels {announced}')
    arm = ARM_CHANNELS[arm_name]
    data_root = args.data_root or DEFAULT_DATA[arm_name]
    out_dir = args.out or Path.home()/('turbovla/logs/offline_check' if arm_name == 'left' else f'turbovla/logs/offline_check_{arm_name}')
    run_dir = args.run_dir or DEFAULT_RUN.get(arm_name, args.bundle/'models/mocap10k')
    split = json.loads((run_dir/'split.json').read_text())
    held_out, train = set(split['evaluation_episodes']), set(split['optimization_episodes'])
    found = find_episodes(data_root)
    # LeRobot datasets name their episodes episode_00000N; map them to the source names of the split (validation = evaluation_episodes
    # in order, train = optimization_episodes in order) so the report says which demonstration is meant
    alias = {}
    if (data_root/'meta/info.json').is_file():
        names = split['evaluation_episodes'] if 'validation' in data_root.name else split['optimization_episodes'] if 'train' in data_root.name else []
        for key, name in zip(sorted(found), names):
            alias[key] = name
    print(f'arm {arm_name} channels {arm}, data root {data_root}, split from {run_dir/"split.json"}, episodes '
          f'{ {k: alias.get(k, k) for k in found} }', flush=True)
    wanted = args.episodes or sorted(found)
    episodes = [next((k for k in found if wanted_name in (k, alias.get(k))), None) for wanted_name in wanted]
    if None in episodes:
        raise SystemExit(f'unknown episode(s) {[w for w, e in zip(wanted, episodes) if e is None]}')
    args.out = out_dir
    args.out.mkdir(parents=True, exist_ok=True)
    results, parts, lat = {}, {}, {'normal': [], 'black_wrist': []}
    for variant, nframes in (('normal', args.frames), ('black_wrist', args.black_frames)):
        if nframes < 0 and variant == 'black_wrist':
            continue
        for ep in episodes:
            cache = args.out/f'{alias.get(ep, ep)}.{variant}.npz'
            if args.reuse and cache.exists():
                z = np.load(cache); state, action, chunks, latency = z['state'], z['action'], z['chunks'], z['latency_ms']
            else:
                t0 = time.time()
                state, action, chunks, latency = run_episode(client, found[ep], alias.get(ep, ep), nframes, variant)
                np.savez(cache, state=state, action=action, chunks=chunks, latency_ms=latency)
                print(f'{alias.get(ep, ep)} {variant}: {len(chunks)} frames in {time.time()-t0:.0f}s', flush=True)
            m, extra = metrics(state, action, chunks, arm, first_step_gate=args.first_step_gate)
            name = alias.get(ep, ep)
            m['split'] = 'held_out' if name in held_out else ('train' if name in train else 'unknown')
            m['state_absmax_outside_arm'] = float(np.abs(np.delete(state, np.arange(*arm), axis=1)).max())
            results.setdefault(variant, {})[name] = m
            parts.setdefault((variant, m['split']), []).append(extra)
            lat[variant].append(latency[5:])  # skip first frames (cold)
            print(f"{name:18s} {variant:11s} {m['split']:8s} n={m['frames']:5d} raw {m['raw_mae']:.4f} padded {m['raw_mae_padded']:.4f} (hold {m['hold_raw_mae']:.4f})"
                  f" ens {m['ens_mae']:.4f} (hold {m['hold_ens_mae']:.4f}) first-step median {m['first_step_max_median']:.4f} "
                  f"pass@{args.first_step_gate} {m['first_step_gate_pass_frac']:.2f} corr {np.round(m['dir_corr_per_joint'], 2).tolist()}", flush=True)
    summary = {}
    for (variant, split_name), p in sorted(parts.items()):
        summary[f'{variant}/{split_name}'] = pooled(p)
    for variant, arrays in lat.items():
        if arrays:
            a = np.concatenate(arrays)
            summary[f'{variant}/latency_ms'] = {'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)), 'n': int(len(a))}
    report = {'arm': arm_name, 'arm_channels': list(arm), 'data_root': str(data_root), 'run_dir': str(run_dir),
              'target_definition': 'chunk step k vs recorded action column at frame t+k; ensemble vs action[t]; hold = state[t]',
              'episodes': results, 'summary': summary}
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
