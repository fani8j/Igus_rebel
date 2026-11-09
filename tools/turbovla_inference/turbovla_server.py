#!/usr/bin/env python3
"""TurboVLA policy server for a local ROS client: the bundle's own left-arm checkpoint (mocap10k, step-10000 EMA) by default,
or any other checkpoint of the same architecture (e.g. the right-arm run right19-10k) via --checkpoint.

Runs in the handoff bundle's policy venv (python 3.12, torch 2.13+cu130). It imports the model
code, config contract and observation contract from the bundle source instead of copying them, and
performs exactly the load sequence of source/scripts/compal/mocap_policy_worker.py.

Checkpoint rules (no unpinned loads):
  * the bundle's own checkpoint is pinned by the bundle MANIFEST (checkpoint_sha256) and uses the language 'Open the box flaps.' unless
    --language says otherwise;
  * ANY OTHER checkpoint needs an explicit --expect-sha256 <hex> and an explicit --language (TURBOVLA_EXPECT_SHA256 / TURBOVLA_LANGUAGE);
    the file is hashed BEFORE any weights are loaded and the server refuses to start on a mismatch or a missing pin.
  * the active arm is derived from the run's normalization.json/embodiment.json (the arm whose six joint channels have an observed
    action span; the gripper channels are ignored because their range is a fixed 0..1 placeholder) and announced as
    arm_channels [first, end).  The server refuses to start unless exactly one arm's 6 joints are active, and refuses any
    get_action whose state is non-zero outside those channels.

Protocol (FIXED, same wire format as Isaac-GR00T's server_client.py MsgSerializer):
  ZeroMQ ROUTER on tcp://127.0.0.1:5556 (a REQ client works; strays are ignored), requests are
  msgpack(-numpy) maps {"endpoint": str, "data": {...}}, replies are msgpack(-numpy) values.
    ping                -> {"status": "ok", "message": "..."}
    get_modality_config -> {"backend": "turbovla", "cameras": [...], "language": str, "arm_channels": [first, end], "arm": "left"|"right", ...}
    reset               -> {"status": "ok"}            (stateless; ACT ensembling is client-side)
    get_action          -> [{"actions": float32 (30,14)}, {"inference_ms": float, "step": int}]
  Any bad input or exception -> {"error": "<message>"} (never a bare b"ERROR").
No ROS, no robot connection. One request is served at a time.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

DEFAULT_BUNDLE = Path.home()/'turbovla/compal-turbovla-handoff-v1'
BUNDLE_CHECKPOINT = Path('models/mocap10k/checkpoints/steps_10000_ema_pytorch_model.pt')  # relative to the bundle
BUNDLE_LANGUAGE = 'Open the box flaps.'  # default language of the bundle's own checkpoint ONLY
CAMERAS = ['head', 'knife', 'wrist_left']
CLIENT_CAMERAS = ['head', 'wrist_left']


# ---- wire format: identical to gr00t.policy.server_client.MsgSerializer (without ModalityConfig) ----
import msgpack
import msgpack_numpy as mnp


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer._safe_encode)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer._safe_decode, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
            raise TypeError('Refusing to encode object-dtype ndarray')
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        if isinstance(obj, dict):
            marker = obj.get('__ndarray_class__', obj.get(b'__ndarray_class__'))
            if marker:
                payload = obj.get('as_npy', obj.get(b'as_npy'))
                if payload is None:
                    raise ValueError("Malformed ndarray payload: marker present but 'as_npy' missing")
                return np.load(io.BytesIO(payload), allow_pickle=False)
            nd_val = obj.get(b'nd', obj.get('nd'))
            kind_val = obj.get(b'kind', obj.get('kind'))
            if nd_val and kind_val in (b'O', 'O'):
                raise ValueError('Refusing to decode object-dtype ndarray payload (pickle-bearing)')
        return mnp.decode(obj, chain=chain)


def setup_environment(bundle):
    """Mirror bundle.py environment(): source paths and offline model assets."""
    source = bundle/'source'
    for path in (source/'third_party/starvla_runtime', source):  # same order result as the worker
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(
        BERT_MODEL_PATH=str(bundle/'assets/bert-base-uncased'),
        DINOV3_MODEL_PATH=str(bundle/'assets/dinov3_vitl16'),
        TURBOVLA_INIT_CKPT=str(bundle/'assets/donor/steps_55000_ema_model.safetensors'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
        HF_HOME=str(bundle/'.cache/huggingface'), TORCH_HOME=str(bundle/'.cache/torch'))
    return source


def derive_arm_channels(run):
    """-> (name, [first, end)) of the single active arm of a run directory.

    Active = all 6 joint channels of the arm (embodiment.json action.arms[*].joints) have an observed action span above
    normalization.json action.minimum_span.  (`retained_channels` in normalization.json lists the channels kept in the model, all 14,
    so it says nothing about which arm moves; gripper channels carry a fixed 0..1 placeholder range in the left bundle and are skipped.)
    Raises ValueError unless exactly one arm is fully active and the other arm has no active joint channel."""
    embodiment = json.loads((run/'embodiment.json').read_text())
    action = json.loads((run/'normalization.json').read_text())['action']
    low = np.asarray(action.get('observed_min', action['min']), dtype=np.float64)
    high = np.asarray(action.get('observed_max', action['max']), dtype=np.float64)
    minimum_span = float(action['minimum_span'])
    active_arms = []
    for arm in embodiment['action']['arms']:
        first, end = int(arm['joints']['start']), int(arm['joints']['end'])
        active = (high[first:end]-low[first:end]) > minimum_span
        if active.any() and not active.all():
            raise ValueError(f"arm {arm['name']}: only joints {[int(i)+1 for i in np.flatnonzero(active)]} have an action span (partial arm)")
        if active.all():
            if end-first != 6:
                raise ValueError(f"arm {arm['name']} has {end-first} joint channels, expected 6")
            active_arms.append((str(arm['name']), [first, end]))
    if len(active_arms) != 1:
        raise ValueError(f'Expected exactly one active arm in {run}, found {[a[0] for a in active_arms]}')
    return active_arms[0]


class TurboVLAPolicy:
    def __init__(self, bundle, checkpoint=None, language=None, expect_sha256=None):
        import torch
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework
        from turbovla.embodiment import EmbodimentCodec, Observation, load_embodiment_spec
        from experiments.compal.mocap_contract import assert_calibration, validate_config
        from experiments.compal.mocap_live import live_observation
        self.torch, self.Observation, self.live_observation = torch, Observation, live_observation
        own = (bundle/BUNDLE_CHECKPOINT).resolve(strict=True)
        checkpoint = Path(checkpoint or own).resolve(strict=True)
        run = checkpoint.parent.parent
        self.bundle_own = checkpoint == own
        # --- pins and language first, before anything is loaded onto the GPU
        expect_sha256 = expect_sha256.strip().lower() if expect_sha256 else None
        if expect_sha256 is not None and (len(expect_sha256) != 64 or set(expect_sha256) - set('0123456789abcdef')):
            raise ValueError('--expect-sha256 must be 64 hex characters')
        if self.bundle_own:
            pins = [json.loads((bundle/'MANIFEST.json').read_text()).get('checkpoint_sha256')]
            if not pins[0]:
                raise ValueError('bundle MANIFEST.json has no checkpoint_sha256: refusing an unpinned load')
            if expect_sha256: pins.append(expect_sha256)
            self.language = language or BUNDLE_LANGUAGE
        else:
            if not expect_sha256:
                raise ValueError(f'checkpoint {checkpoint} is not the bundle checkpoint: --expect-sha256 <hex> is required (no unpinned loads)')
            if not language:
                raise ValueError(f'checkpoint {checkpoint} is not the bundle checkpoint: --language is required (the training instruction)')
            pins, self.language = [expect_sha256], language
        with checkpoint.open('rb') as stream:
            self.sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        for pin in pins:
            if pin != self.sha256:
                raise ValueError(f'Checkpoint sha256 {self.sha256} != pin {pin}')
        self.arm_name, self.arm_channels = derive_arm_channels(run)
        config = OmegaConf.load(run/'config.full.yaml'); validate_config(config)
        # Exported bundles use paths relative to the model run, independent of cwd.
        for section, key in ((config.framework.text, 'bert_path'), (config.framework.vision, 'model_path')):
            path = Path(section[key])
            if not path.is_absolute(): section[key] = str((run/path).resolve())
        config.framework.initialization.load_pretrained = False
        config.framework.vision.load_pretrained = False
        torch.set_num_threads(4); torch.manual_seed(42)
        self.codec = EmbodimentCodec(load_embodiment_spec(run/'embodiment.json'))
        assert_calibration(self.codec, json.loads((run/'normalization.json').read_text()))
        if [c.name for c in self.codec.spec.observation.cameras] != CAMERAS:
            raise ValueError('Unexpected trained camera order')
        self.model = TurboVLAFramework(config).cuda().eval()
        weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        self.model.load_state_dict(weights, strict=True); del weights
        self.checkpoint = checkpoint

    def predict(self, state, head, wrist_left):
        """Same sequence as the bundle worker's request loop. Returns (float64 (30,14) rad, seconds)."""
        torch = self.torch
        state, images = self.live_observation(state, head, wrist_left)
        encoded = self.codec.encode_observation(self.Observation(images=images, state=state))
        example = {'image': list(encoded.views), 'state': encoded.state, 'lang': self.language}
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model.predict_action([example])['normalized_actions'][0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        action = self.codec.decode_action_chunk(prediction).values
        if action.shape != (30, 14) or not np.isfinite(action).all():
            raise ValueError('Invalid predicted chunk')
        return action, elapsed


class Server:
    def __init__(self, policy, host, port):
        import zmq
        self.zmq, self.policy = zmq, policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 1000)
        self.socket.bind(f'tcp://{host}:{port}')
        self.endpoint_name = f'tcp://{host}:{port}'
        self.endpoints = {'ping': self.ping, 'get_modality_config': self.get_modality_config,
                          'reset': self.reset, 'get_action': self.get_action}
        self.served = 0

    # --- endpoints ---
    def ping(self, **_):
        return {'status': 'ok', 'message': 'TurboVLA server is running'}

    def get_modality_config(self, **_):
        return {'backend': 'turbovla', 'cameras': list(CAMERAS), 'client_cameras': list(CLIENT_CAMERAS),
                'image_size': [224, 224], 'state_dim': 14, 'action_dim': 14, 'horizon': 30,
                'control_rate_hz': 30, 'checkpoint_sha256': self.policy.sha256, 'language': self.policy.language,
                'arm': self.policy.arm_name, 'arm_channels': list(self.policy.arm_channels)}

    def reset(self, **_):
        return {'status': 'ok'}

    def get_action(self, observation=None, options=None, **_):
        if not isinstance(observation, dict):
            raise ValueError("get_action needs data={'observation': {...}, 'options': None}")
        missing = [k for k in ('state', 'head', 'wrist_left', 'step') if k not in observation]
        if missing:
            raise ValueError(f'observation is missing {missing}')
        step = observation['step']
        if isinstance(step, np.ndarray) and step.shape == ():
            step = step.item()
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise ValueError(f'step must be an int, got {type(step).__name__}')
        arrays = {}
        for key in ('state', 'head', 'wrist_left'):
            value = observation[key]
            if not isinstance(value, np.ndarray):
                raise ValueError(f'{key} must be a numpy array (msgpack-numpy), got {type(value).__name__}')
            arrays[key] = value
        if not np.issubdtype(arrays['state'].dtype, np.floating):
            raise ValueError(f"state must be a float array, got {arrays['state'].dtype}")
        first, end = self.policy.arm_channels
        state = arrays['state']
        if state.shape == (14,) and np.isfinite(state).all():  # shape / NaN errors are reported by the bundle's live_observation below
            outside = np.delete(state, np.arange(first, end))
            if np.abs(outside).max() > 1e-9:
                raise ValueError(f'state channels outside the {self.policy.arm_name} arm channels [{first}, {end}) must be exactly 0 '
                                 f'(training contract; this checkpoint only knows the {self.policy.arm_name} arm), got {state.tolist()}')
        action, elapsed = self.policy.predict(arrays['state'], arrays['head'], arrays['wrist_left'])
        self.served += 1
        return [{'actions': action.astype(np.float32)}, {'inference_ms': elapsed*1000.0, 'step': int(step)}]

    # --- loop ---
    def handle(self, payload):
        try:
            request = MsgSerializer.from_bytes(payload)
            if not isinstance(request, dict):
                raise ValueError('request must be a msgpack map {"endpoint": str, "data": {...}}')
            endpoint = request.get('endpoint', 'get_action')
            if endpoint not in self.endpoints:
                raise ValueError(f'Unknown endpoint: {endpoint}')
            data = request.get('data') or {}
            if not isinstance(data, dict):
                raise ValueError('"data" must be a map')
            result = self.endpoints[endpoint](**data)
        except Exception as error:
            print(f'request error: {error}', flush=True)
            if not isinstance(error, (ValueError, TypeError)):
                print(traceback.format_exc(), flush=True)
            result = {'error': str(error)}
        try:
            return MsgSerializer.to_bytes(result)
        except Exception as error:
            return MsgSerializer.to_bytes({'error': f'reply encoding failed: {error}'})

    def run(self):
        zmq = self.zmq
        print(f'TurboVLA server ready on {self.endpoint_name}', flush=True)
        while self.running:
            if not self.socket.poll(500):
                continue
            frames = self.socket.recv_multipart()
            if len(frames) < 2:
                continue
            # REQ peers send [identity, b'', payload]; DEALER peers [identity, payload]: echo whatever
            # envelope frames sit between the identity and the payload.
            reply = self.handle(frames[-1])
            try:
                self.socket.send_multipart([frames[0], *frames[1:-1], reply], flags=zmq.NOBLOCK)
            except zmq.ZMQError as error:
                print(f'reply dropped: {error}', flush=True)

    def close(self):
        self.socket.close(linger=1000)
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, default=Path(os.environ.get('TURBOVLA_BUNDLE', DEFAULT_BUNDLE)))
    parser.add_argument('--checkpoint', type=Path, default=None, help='default: the bundle checkpoint (models/mocap10k); any other needs --language and --expect-sha256')
    parser.add_argument('--language', default=os.environ.get('TURBOVLA_LANGUAGE') or None,
                        help="instruction fed to the model (env TURBOVLA_LANGUAGE); default 'Open the box flaps.' for the bundle checkpoint only")
    parser.add_argument('--expect-sha256', default=os.environ.get('TURBOVLA_EXPECT_SHA256') or None,
                        help='required sha256 of --checkpoint unless it is the bundle checkpoint (env TURBOVLA_EXPECT_SHA256)')
    parser.add_argument('--host', default=os.environ.get('TURBOVLA_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('TURBOVLA_PORT', '5556')))
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    setup_environment(bundle)
    t0 = time.time()
    try:
        policy = TurboVLAPolicy(bundle, args.checkpoint, args.language, args.expect_sha256)
    except ValueError as error:
        print(f'refusing to start: {error}', file=sys.stderr, flush=True)
        sys.exit(2)
    print(f'model loaded in {time.time()-t0:.1f}s, checkpoint {policy.checkpoint}', flush=True)
    print(f'checkpoint sha256 {policy.sha256} ({"bundle MANIFEST pin" if policy.bundle_own else "--expect-sha256 pin"}), '
          f'arm {policy.arm_name} channels {policy.arm_channels}, language {policy.language!r}', flush=True)
    if not args.no_warmup:
        zeros = np.zeros((224, 224, 3), np.uint8)
        for _ in range(3):
            policy.predict(np.zeros(14), zeros, zeros)
        print('warmup done', flush=True)
    server = Server(policy, args.host, args.port)
    def stop(*_): server.running = False
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        server.run()
    finally:
        server.close()
        print(f'server stopped after {server.served} get_action calls', flush=True)


if __name__ == '__main__':
    main()
