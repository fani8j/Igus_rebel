"""Gripper camera setup logic with fake brightness reader and prompts; no hardware or ROS."""

import importlib.util
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'gripper_camera_setup.py')
spec = importlib.util.spec_from_file_location('gripper_camera_setup', SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
PORT = '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{}:1.0-video-index0'
A, B, C = PORT.format('3.1'), PORT.format('3.2'), PORT.format('3.3')

YAML = '''# header comment 3.1 = right
/**:
  ros__parameters:
    video_device: "%s"   # trailing comment
    camera_name: "x"
'''


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / 'left_gripper_camera.yaml').write_text(YAML % B)
    (tmp_path / 'right_gripper_camera.yaml').write_text(YAML % A)
    return tmp_path


def read(cfg, arm):
    return (cfg / f'{arm}_gripper_camera.yaml').read_text()


class Env:
    """Fake bus: cameras = by-path list; script = list of brightness dicts, one per measure."""

    def __init__(self, tmp_path, cameras):
        self.by_path = tmp_path / 'by-path'
        self.by_path.mkdir()
        for c in cameras:
            (self.by_path / os.path.basename(c)).write_text('')
        self.names = [str(self.by_path / os.path.basename(c)) for c in cameras]
        self.measures = []
        self.answers = []
        self.out = []

    def reader(self, path):
        v = self.measures.pop(0)
        if isinstance(v, Exception):
            raise v
        return v[path]

    def prompt(self, text):
        self.out.append(f'PROMPT {text}')
        if not self.answers:
            raise EOFError
        a = self.answers.pop(0)
        if a is KeyboardInterrupt:
            raise KeyboardInterrupt
        return a

    def go(self, cfg):
        return setup.run(str(cfg), prompt=self.prompt, reader=self.reader,
                         by_path=str(self.by_path), usb_id_reader=lambda p: setup.GRIPPER_CAMERA_USB_ID,
                         out=self.out.append)


def brightness(env, **vals):
    """dict keyed by camera index: brightness(env, c0=100, c1=100)."""
    return {env.names[int(k[1:])]: v for k, v in vals.items()}


def phases(env, *steps):
    """Each step is (c0, c1) brightness; the reader is called once per camera."""
    for c0, c1 in steps:
        env.measures += [{env.names[0]: c0}, {env.names[1]: c1}]
    # the dict returned per call must contain the asked path; readers pop per camera


class PerCall(Env):
    def reader(self, path):
        step = self.steps[0]
        v = step[self.names.index(path)]
        self.calls += 1
        if self.calls == len(self.names):
            self.steps.pop(0)
            self.calls = 0
        return v


def two(tmp_path, steps, answers):
    env = PerCall(tmp_path, [A, B])
    env.steps, env.calls, env.answers = list(steps), 0, list(answers)
    return env


def test_pick_covered_and_inconclusive():
    base = {'a': 100.0, 'b': 100.0}
    assert setup.pick_covered(base, {'a': 10.0, 'b': 95.0}) == 'a'
    assert setup.pick_covered(base, {'a': 80.0, 'b': 95.0}) is None
    assert setup.pick_covered(base, {'a': 10.0, 'b': 30.0}) is None


def test_update_only_path_and_comments_preserved():
    new = setup.update_video_device(YAML % A, B)
    assert new == YAML % B
    assert setup.update_video_device("video_device: '/x/y'\n", '/z') == "video_device: '/z'\n"


def test_real_configs_parse():
    for arm, name in setup.CONFIG_FILES.items():
        text = open(os.path.join(CONFIG_DIR, name)).read()
        assert setup.current_video_device(text).startswith('/dev/v4l/by-path/')
        assert setup.update_video_device(text, 'QQQ').count('QQQ') == 1


def test_two_cameras_swapped(cfg, tmp_path):
    # A (3.1) is covered first -> A is left; B right; currently swapped, so both change.
    env = two(tmp_path, [(100, 100), (10, 98), (97, 12)], ['', '', 'y'])
    assert env.go(cfg) == 0
    assert read(cfg, 'left') == YAML % env.names[0]
    assert read(cfg, 'right') == YAML % env.names[1]
    assert any('Restart the camera launch' in o for o in env.out)


def test_inconclusive_then_retry(cfg, tmp_path):
    env = two(tmp_path, [(100, 100), (90, 95), (100, 100), (100, 8), (10, 100)],
              ['', 'y', '', '', 'y'])
    assert env.go(cfg) == 0
    assert read(cfg, 'left') == YAML % env.names[1]
    assert read(cfg, 'right') == YAML % env.names[0]
    assert any('Inconclusive' in o for o in env.out)


def test_inconclusive_then_quit(cfg, tmp_path):
    env = two(tmp_path, [(100, 100), (90, 95)], ['', 'n'])
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


def test_same_camera_both_steps(cfg, tmp_path):
    env = two(tmp_path, [(100, 100), (10, 98), (10, 98)], ['', '', 'n'])
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert any('same camera' in o for o in env.out)
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


def test_declining_write(cfg, tmp_path):
    env = two(tmp_path, [(100, 100), (10, 98), (97, 12)], ['', '', 'n'])
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 0
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


def test_no_change_needed(cfg, tmp_path):
    env = two(tmp_path, [(100, 100), (98, 10), (10, 97)], ['', ''])
    (cfg / 'left_gripper_camera.yaml').write_text(YAML % env.names[1])
    (cfg / 'right_gripper_camera.yaml').write_text(YAML % env.names[0])
    assert env.go(cfg) == 0   # B left, A right already
    assert any('nothing to change' in o for o in env.out)


def test_one_camera_updates_only_that_arm(cfg, tmp_path):
    env = Env(tmp_path, [C])
    env.reader = lambda p: pytest.fail('no cover test with one camera')
    env.answers = ['r', 'y']
    left_before = read(cfg, 'left')
    assert env.go(cfg) == 0
    assert read(cfg, 'right') == YAML % env.names[0]
    assert read(cfg, 'left') == left_before
    assert any('left camera is still missing' in o for o in env.out)


def test_one_camera_quit(cfg, tmp_path):
    env = Env(tmp_path, [C])
    env.answers = ['q']
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


@pytest.mark.parametrize('cams', [[], [A, B, C]])
def test_zero_or_three_cameras(cfg, tmp_path, cams):
    env = Env(tmp_path, cams)
    env.reader = lambda p: pytest.fail('must not open cameras')
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


def test_open_failure_writes_nothing(cfg, tmp_path):
    env = Env(tmp_path, [A, B])
    env.measures = [setup.CameraError('cannot open x')]
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert any('Stop the camera launch' in o for o in env.out)
    assert (read(cfg, 'left'), read(cfg, 'right')) == before


@pytest.mark.parametrize('answers', [[], [KeyboardInterrupt]])
def test_eof_and_ctrl_c(cfg, tmp_path, answers):
    env = two(tmp_path, [(100, 100)], answers)
    before = (read(cfg, 'left'), read(cfg, 'right'))
    assert env.go(cfg) == 1
    assert (read(cfg, 'left'), read(cfg, 'right')) == before
