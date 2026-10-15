import json
import os
import re

import numpy as np
import pytest

import cut_recorder as cr

SRC = os.path.join(os.path.dirname(__file__), '..', 'cut_recorder.py')


# ---------------------------------------------------------------- passive
def test_source_is_passive():
    text = open(SRC).read()
    for banned in ('create_publisher', 'create_client', 'ActionClient',
                   'create_service', 'set_parameters', 'call_async',
                   '.publish(', 'declare_parameter', 'create_action'):
        assert banned not in text, banned


def test_only_subscriptions_created():
    text = open(SRC).read()
    assert set(re.findall(r'\.create_(\w+)\(', text)) <= {
        'node', 'subscription', 'timer'}


# ---------------------------------------------------------------- mapper
def test_mapper_basic_and_gap_duplication():
    m = cr.FrameMapper(10.0)
    p = m.plan(0.0)
    assert (p.write_new, p.repeat_prev, p.repeat_new, p.index) == (True, 0, 1, 0)
    p = m.plan(0.1)
    assert (p.write_new, p.repeat_prev, p.index) == (True, 0, 1)
    p = m.plan(0.5)          # 3 missing frames (idx 2,3,4 -> repeat prev x3)
    assert (p.write_new, p.repeat_prev, p.repeat_new, p.index) == (True, 3, 1, 5)
    assert p.duplicated == 3
    assert m.next_index == 6


def test_mapper_drops_same_index():
    m = cr.FrameMapper(10.0)
    m.plan(0.0)
    m.plan(0.1)
    p = m.plan(0.11)
    assert p.write_new is False
    p = m.plan(0.04)         # out of order stamp
    assert p.write_new is False
    assert m.next_index == 2


def test_mapper_lead_in_and_negative_stamp():
    m = cr.FrameMapper(10.0)
    p = m.plan(0.3)          # first frame arrives 0.3 s late: pad 3 + itself
    assert (p.repeat_new, p.index, p.duplicated) == (4, 3, 3)
    m2 = cr.FrameMapper(10.0)
    assert m2.plan(-0.5).index == 0


def test_mapper_rounding_half_up():
    m = cr.FrameMapper(10.0)
    assert m.plan(0.049).index == 0
    m = cr.FrameMapper(10.0)
    assert m.plan(0.051).index == 1


def test_mapper_bad_fps():
    with pytest.raises(ValueError):
        cr.FrameMapper(0)


# ---------------------------------------------------------------- state machine
def test_sm_start_stop_post_roll():
    sm = cr.ClipStateMachine(post_roll_s=3.0, max_clip_s=900)
    assert sm.update(0.0, False) is None
    assert sm.update(1.0, True) == ('start', 'status')
    assert sm.update(2.0, True) is None
    assert sm.update(3.0, False) is None          # 1 s idle
    assert sm.update(4.9, False) is None          # 2.9 s idle
    assert sm.update(5.0, False) == ('stop', 'post_roll')
    assert sm.update(7.0, False) is None


def test_sm_consecutive_trajectories_one_clip():
    sm = cr.ClipStateMachine(3.0, 900)
    assert sm.update(0.0, True)[0] == 'start'
    assert sm.update(2.0, False) is None          # gap between trajectories
    assert sm.update(2.9, False) is None
    assert sm.update(3.0, True) is None           # next trajectory: same clip
    assert sm.update(5.9, False) is None
    assert sm.update(6.1, False) == ('stop', 'post_roll')


def test_sm_gap_longer_than_post_roll_makes_two_clips():
    sm = cr.ClipStateMachine(3.0, 900)
    sm.update(0.0, True)
    assert sm.update(3.5, False) == ('stop', 'post_roll')
    assert sm.update(10.0, True) == ('start', 'status')


def test_sm_max_clip():
    sm = cr.ClipStateMachine(3.0, 10.0)
    sm.update(0.0, True)
    assert sm.update(9.0, True) is None
    assert sm.update(10.0, True) == ('stop', 'max_clip')
    assert sm.update(10.1, True) == ('start', 'status')   # keeps recording


def test_sm_force_open():
    sm = cr.ClipStateMachine(3.0, 900)
    sm.force_open(0.0)
    assert sm.is_open
    sm.force_close()
    assert not sm.is_open


# ---------------------------------------------------------------- decoding
def test_image_rgb8_with_step_padding():
    h, w, step = 2, 3, 12      # 9 bytes of pixels + 3 pad per row
    row0 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9]) + b'\xff\xff\xff'
    row1 = bytes([10, 11, 12, 13, 14, 15, 16, 17, 18]) + b'\xff\xff\xff'
    img = cr.image_to_bgr(h, w, step, 'rgb8', row0 + row1)
    assert img.shape == (2, 3, 3)
    assert list(img[0, 0]) == [3, 2, 1]        # rgb -> bgr
    assert list(img[1, 2]) == [18, 17, 16]


def test_image_bgr8_unchanged():
    data = bytes(range(12))
    img = cr.image_to_bgr(2, 2, 6, 'bgr8', data)
    assert list(img[0, 1]) == [3, 4, 5]
    assert img.flags['C_CONTIGUOUS']


def test_image_mono8():
    data = bytes([0, 100, 255, 7, 8, 9, 99, 99])   # w=3, step=4, h=2
    img = cr.image_to_bgr(2, 3, 4, 'mono8', bytes([0, 100, 255, 99, 7, 8, 9, 99]))
    assert img.shape == (2, 3, 3)
    assert list(img[0, 1]) == [100, 100, 100]
    assert list(img[1, 2]) == [9, 9, 9]
    del data


def test_image_errors():
    with pytest.raises(ValueError):
        cr.image_to_bgr(2, 2, 6, 'yuv422', bytes(12))
    with pytest.raises(ValueError):
        cr.image_to_bgr(2, 2, 6, 'rgb8', bytes(5))
    with pytest.raises(ValueError):
        cr.image_to_bgr(2, 2, 3, 'rgb8', bytes(12))


def test_compressed_roundtrip():
    import cv2
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :, 2] = 200
    ok, buf = cv2.imencode('.jpg', img)
    out = cr.compressed_to_bgr(buf.tobytes())
    assert out.shape == (20, 30, 3)
    assert out[5, 5, 2] > 150 and out[5, 5, 0] < 50
    with pytest.raises(ValueError):
        cr.compressed_to_bgr(b'not a jpeg')


# ---------------------------------------------------------------- specs
def test_parse_camera_spec():
    s = cr.parse_camera_spec('foo=/a/b:compressed:20:0.5')
    assert (s.name, s.topic, s.kind, s.fps, s.scale) == (
        'foo', '/a/b', 'compressed', 20.0, 0.5)
    s = cr.parse_camera_spec('foo=/a/b/compressed')
    assert s.kind == 'compressed' and s.fps == 15.0 and s.scale == 1.0
    s = cr.parse_camera_spec('scene=/x/y:raw')
    assert s.fps == 15.0 and s.scale == 0.5     # inherits the default of that name
    for bad in ('noequals', '=/t', 'a=', 'a=/t:weird', 'a=/t:raw:0',
                'a=/t:raw:10:0', 'a=/t:raw:10:1:extra', 'a b=/t'):
        with pytest.raises(ValueError):
            cr.parse_camera_spec(bad)


def test_resolve_cameras():
    p = cr.build_parser()
    specs = cr.resolve_cameras(p.parse_args([]))
    assert [s.name for s in specs] == ['knife', 'scene', 'wrist', 'right_wrist']
    assert specs[1].scale == 0.5
    specs = cr.resolve_cameras(p.parse_args(['--cameras', 'knife,scene',
                                             '--scene-scale', '0.25']))
    assert [s.name for s in specs] == ['knife', 'scene'] and specs[1].scale == 0.25
    assert cr.DEFAULT_CAMERAS['scene'].scale == 0.5      # defaults not mutated
    specs = cr.resolve_cameras(p.parse_args(['--cameras', 'extra',
                                             '--camera', 'extra=/t:raw:5']))
    assert specs[0].fps == 5.0
    with pytest.raises(ValueError):
        cr.resolve_cameras(p.parse_args(['--cameras', 'nope']))


# ---------------------------------------------------------------- camera recorder
class FakeWriter:
    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(int(frame[0, 0, 0]))

    def release(self):
        self.released = True


def _rec(fps=10.0, scale=1.0, start=100.0, decode=None):
    writers = []

    def factory(path, f, size):
        w = FakeWriter(path, f, size)
        writers.append(w)
        return w
    spec = cr.CameraSpec('t', '/t', 'raw', fps, scale)
    rec = cr.CameraRecorder(spec, '/tmp/x.mp4', start,
                            decode or (lambda b: (0.0, None)), factory,
                            threaded=False)
    return rec, writers


def _frame(v, w=8, h=6):
    return np.full((h, w, 3), v, np.uint8)


def test_recorder_duplicates_and_drops():
    rec, ws = _rec()
    rec.add_frame(100.0, _frame(1))
    rec.add_frame(100.1, _frame(2))
    rec.add_frame(100.11, _frame(3))        # same index -> dropped
    rec.add_frame(100.4, _frame(4))         # gap of 2
    st = rec.close()
    assert ws[0].frames == [1, 2, 2, 2, 4]
    assert ws[0].released
    assert st['frames_written'] == 3 and st['frames_duplicated'] == 2
    assert st['frames_dropped'] == 1 and st['frames_in_video'] == 5
    assert st['status'] == 'ok' and st['size'] == [8, 6]


def test_recorder_scaling_and_even_size():
    rec, ws = _rec(scale=0.5)
    rec.add_frame(100.0, _frame(5, w=11, h=9))
    w, h = ws[0].size
    assert w % 2 == 0 and h % 2 == 0 and w in (4, 6) and h == 4
    rec.add_frame(100.1, _frame(5, w=30, h=20))   # size change is resized
    st = rec.close()
    assert st['size'] == [w, h] and st['frames_written'] == 2


def test_recorder_process_stamp_fallbacks_and_decode_errors():
    calls = []

    def decode(b):
        calls.append(b)
        if b == b'bad':
            raise ValueError('x')
        return {b'zero': 0.0, b'far': 5000.0, b'ok': 100.2}[b], _frame(9)
    rec, ws = _rec(decode=decode)
    rec.process(100.0, b'zero')             # zero stamp -> receipt time
    rec.process(100.1, b'far')              # absurd stamp -> receipt time
    rec.process(100.2, b'ok')
    rec.process(100.3, b'bad')
    st = rec.close()
    assert st['decode_errors'] == 1 and st['stamp_fallbacks'] == 1
    assert st['frames_written'] == 3 and st['frames_in_video'] == 3


def test_recorder_no_frames_status():
    rec, ws = _rec()
    st = rec.close()
    assert st['status'] == 'no frames' and st['file'] is None
    assert st['frames_written'] == 0 and ws == []


def test_recorder_queue_full_never_blocks():
    spec = cr.CameraSpec('t', '/t', 'raw', 10.0, 1.0)
    rec = cr.CameraRecorder(spec, '/tmp/x.mp4', 0.0, lambda b: (0.0, None),
                            lambda *a: FakeWriter(*a), queue_len=3,
                            threaded=False)
    for i in range(10):
        rec.submit(float(i), b'x')
    assert rec.received == 10 and rec.dropped_queue_full == 7
    assert rec.stats()['frames_dropped'] == 7


def test_recorder_threaded_roundtrip():
    ws = []

    def factory(p, f, s):
        w = FakeWriter(p, f, s)
        ws.append(w)
        return w
    spec = cr.CameraSpec('t', '/t', 'raw', 10.0, 1.0)
    state = {'n': 0}

    def decode(b):
        state['n'] += 1
        return 10.0 + 0.1 * state['n'], _frame(state['n'])
    rec = cr.CameraRecorder(spec, '/tmp/x.mp4', 10.0, decode, factory)
    for i in range(5):
        rec.submit(10.0 + 0.1 * i, b'x')
    st = rec.close()
    assert st['frames_written'] == 5 and ws[0].released


# ---------------------------------------------------------------- clip folder
def test_clip_folder_and_json(tmp_path):
    snap = tmp_path / 'snap.json'
    snap.write_text('{"a": 1}')
    ws = []

    def wf(p, f, s):
        w = FakeWriter(p, f, s)
        ws.append(w)
        return w
    specs = [cr.CameraSpec('knife', '/k', 'raw', 10.0, 1.0),
             cr.CameraSpec('wrist', '/w', 'raw', 10.0, 1.0)]
    clip = cr.Clip(str(tmp_path / 'out'), specs, 'status', 50.0, str(snap),
                   decoder_factory=lambda s: (lambda b: (50.0, _frame(3))),
                   writer_factory=wf)
    assert os.path.basename(clip.dir).count('_') == 1
    clip.note_goals({'abc': 2})
    clip.update_goal_status({'abc': 4, 'zzz': 4})
    clip.submit('knife', 50.0, b'x')
    import time
    time.sleep(0.3)
    meta = clip.finish(53.0, 'post_roll')
    j = json.load(open(os.path.join(clip.dir, 'clip.json')))
    assert j == json.loads(json.dumps(meta))
    assert j['recording'] is False and j['trigger'] == 'status'
    assert j['goal_ids'] == ['abc'] and j['goal_last_status'] == {'abc': 4}
    assert j['snapshot'] == 'snap.json'
    assert os.path.isfile(os.path.join(clip.dir, 'snap.json'))
    assert j['cameras']['knife']['status'] == 'ok'
    assert j['cameras']['wrist']['status'] == 'no frames'
    assert j['stop_reason'] == 'post_roll' and j['stop_ros_s'] == 53.0


def test_clip_missing_snapshot_does_not_fail(tmp_path):
    clip = cr.Clip(str(tmp_path), [], 'now', 1.0, str(tmp_path / 'nope.json'))
    meta = clip.finish(2.0, 'shutdown')
    assert meta['snapshot'] is None and 'not found' in meta['snapshot_error']


def test_two_clips_same_second_get_distinct_dirs(tmp_path):
    a = cr.make_clip_dir(str(tmp_path), 1000000.0)
    b = cr.make_clip_dir(str(tmp_path), 1000000.0)
    assert a != b and os.path.isdir(a) and os.path.isdir(b)


def test_real_cv2_writer_produces_playable_mp4(tmp_path):
    import cv2
    spec = cr.CameraSpec('t', '/t', 'raw', 10.0, 1.0)
    path = str(tmp_path / 't.mp4')
    rec = cr.CameraRecorder(spec, path, 0.0, lambda b: (0.0, None), threaded=False)
    for i in range(10):
        rec.add_frame(i * 0.1, _frame(i * 20, w=64, h=48))
    rec.add_frame(1.5, _frame(250, w=64, h=48))   # gap -> 5 duplicates
    st = rec.close()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == st['frames_in_video'] == 16


def test_default_out_dir_returns_known_value():
    assert cr.default_out_dir() in (cr.EXTERNAL_OUT_DIR, cr.FALLBACK_OUT_DIR)


def test_recorder_tail_padding_to_stop_time():
    rec, ws = _rec()
    rec.add_frame(100.0, _frame(1))
    rec.add_frame(100.2, _frame(2))
    st = rec.close(stop_rel_s=1.0)          # 10 frames total at 10 fps
    assert len(ws[0].frames) == 10 and ws[0].frames[-1] == 2
    assert st['tail_padded_frames'] == 7 and st['frames_in_video'] == 10
    assert st['frames_duplicated'] == 1 + 7


def test_recorder_no_tail_padding_without_frames_or_when_complete():
    rec, ws = _rec()
    st = rec.close(stop_rel_s=5.0)          # camera never delivered: no file
    assert st['status'] == 'no frames' and ws == []
    rec, ws = _rec()
    rec.add_frame(100.0, _frame(1))
    rec.add_frame(100.9, _frame(2))
    st = rec.close(stop_rel_s=1.0)
    assert st['tail_padded_frames'] == 0 and len(ws[0].frames) == 10
