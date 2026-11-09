#!/usr/bin/env python3
"""Interactive setup: assigns the two USB gripper cameras to left/right.

Both gripper cameras are the same model with no USB serial number, so they are
selected by USB port (/dev/v4l/by-path) in left_gripper_camera.yaml and
right_gripper_camera.yaml. After the hub moves, those paths break or left and
right swap. This tool finds the cameras, asks the operator to cover one with a
hand (a covered camera's image gets much darker), and rewrites only the path in
the `video_device:` line of each YAML; comments and other lines stay as they are.

    ros2 run rebel_cameras gripper_camera_setup.py [--config-dir DIR]

The cameras must not be in use: stop the camera launch first. Nothing is
written unless the result is unambiguous and the operator confirms.
"""

import argparse
import os
import re
import sys

BY_PATH_DIR = '/dev/v4l/by-path'
GRIPPER_CAMERA_USB_ID = '05a3:9230'
CONFIG_FILES = {'left': 'left_gripper_camera.yaml', 'right': 'right_gripper_camera.yaml'}

# The covered camera must get at least this much darker (relative to baseline)
# and the other camera at most MAX_OTHER_DROP (auto exposure shifts it a bit).
MIN_COVER_DROP = 0.5
MAX_OTHER_DROP = 0.25

WIDTH, HEIGHT = 320, 240
WARMUP_FRAMES = 45  # auto exposure settles ~1 s after the stream starts (measured 2026-11-09)
MEASURE_FRAMES = 4

VIDEO_DEVICE_RE = re.compile(r'^(\s*video_device:\s*)(["\']?)([^"\'\s#]+)\2', re.MULTILINE)


class CameraError(Exception):
    """A camera could not be opened or read."""


def usb_id_of(realpath):
    """'vvvv:pppp' of the USB device owning a /dev/videoN node, or None."""
    usb_dev = f'/sys/class/video4linux/{os.path.basename(realpath)}/device/..'
    try:
        with open(f'{usb_dev}/idVendor') as v, open(f'{usb_dev}/idProduct') as p:
            return f'{v.read().strip()}:{p.read().strip()}'
    except OSError:
        return None


def find_cameras(by_path=BY_PATH_DIR, usb_id_reader=usb_id_of):
    """Sorted by-path names of the plugged-in gripper cameras (same logic as the launch)."""
    paths = []
    for link in sorted(os.listdir(by_path)) if os.path.isdir(by_path) else []:
        if not link.endswith(':1.0-video-index0'):
            continue
        path = os.path.join(by_path, link)
        if usb_id_reader(os.path.realpath(path)) == GRIPPER_CAMERA_USB_ID:
            paths.append(path)
    return paths


def measure_brightness(path):
    """Mean grayscale brightness (0-255) of a few frames of the camera at path."""
    import cv2
    cap = cv2.VideoCapture(os.path.realpath(path), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise CameraError(f'cannot open {path}')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        values = []
        for i in range(WARMUP_FRAMES + MEASURE_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError(f'cannot read a frame from {path}')
            if i >= WARMUP_FRAMES:
                values.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        return sum(values) / len(values)
    finally:
        cap.release()


def pick_covered(baseline, covered):
    """Camera that was covered, from {path: brightness} before and after, or None.

    None when no camera dropped by at least MIN_COVER_DROP or another one also
    dropped by more than MAX_OTHER_DROP.
    """
    drops = {p: (baseline[p] - covered[p]) / baseline[p] if baseline[p] > 0 else 0.0
             for p in baseline}
    best = max(drops, key=drops.get)
    if drops[best] < MIN_COVER_DROP:
        return None
    if any(d > MAX_OTHER_DROP for p, d in drops.items() if p != best):
        return None
    return best


def current_video_device(text):
    m = VIDEO_DEVICE_RE.search(text)
    return m.group(3) if m else None


def update_video_device(text, new_path):
    """text with only the path of the video_device line replaced."""
    if VIDEO_DEVICE_RE.search(text) is None:
        raise ValueError('no video_device line')
    return VIDEO_DEVICE_RE.sub(lambda m: f'{m.group(1)}{m.group(2)}{new_path}{m.group(2)}',
                               text, count=1)


def measure_all(paths, reader):
    return {p: reader(p) for p in paths}


def identify_two(cameras, prompt, reader, out):
    """Cover test. Returns (left_path, right_path), or None if the operator quit."""
    while True:
        baseline = measure_all(cameras, reader)
        out('Baseline brightness: ' + ', '.join(f'{p}: {v:.1f}' for p, v in baseline.items()))
        prompt('Cover the LEFT gripper camera with your hand, then press Enter')
        left = pick_covered(baseline, measure_all(cameras, reader))
        right = None
        if left is None:
            out('Inconclusive: no single camera got clearly darker.')
        else:
            prompt('Uncover it, cover the RIGHT gripper camera, press Enter')
            right = pick_covered(baseline, measure_all(cameras, reader))
            if right is None:
                out('Inconclusive: no single camera got clearly darker.')
            elif right == left:
                out('Both steps picked the same camera, so the result cannot be trusted.')
            else:
                return left, right
        if prompt('Retry? [Y/n]').strip().lower() in ('n', 'no', 'q'):
            return None


def run(config_dir, prompt=input, reader=measure_brightness, by_path=BY_PATH_DIR,
        usb_id_reader=usb_id_of, out=print):
    """Returns the exit code. prompt(text) -> str may raise EOFError/KeyboardInterrupt."""
    cameras = find_cameras(by_path, usb_id_reader)
    if len(cameras) not in (1, 2):
        out(f'Expected 1 or 2 gripper cameras ({GRIPPER_CAMERA_USB_ID}), found '
            f'{len(cameras)}: {", ".join(cameras) or "none"}. Nothing written.')
        return 1

    files = {arm: os.path.join(config_dir, name) for arm, name in CONFIG_FILES.items()}
    texts = {}
    try:
        for arm, path in files.items():
            with open(path) as f:
                texts[arm] = f.read()
            if current_video_device(texts[arm]) is None:
                out(f'No video_device line in {path}. Nothing written.')
                return 1
    except OSError as e:
        out(f'Cannot read config: {e}. Nothing written.')
        return 1

    try:
        if len(cameras) == 2:
            result = identify_two(cameras, prompt, reader, out)
            if result is None:
                out('Quit. Nothing written.')
                return 1
            assignment = {'left': result[0], 'right': result[1]}
        else:
            ans = prompt(f'Which arm is this camera on ({cameras[0]})? [l/r/q]').strip().lower()
            if ans not in ('l', 'r'):
                out('Quit. Nothing written.')
                return 1
            arm = 'left' if ans == 'l' else 'right'
            assignment = {arm: cameras[0]}
            other = 'right' if arm == 'left' else 'left'
            out(f'WARNING: only one camera is plugged in; the {other} camera is still '
                f'missing and its config is left unchanged.')

        new_texts = {arm: update_video_device(texts[arm], path)
                     for arm, path in assignment.items()}
        for arm, path in assignment.items():
            out(f'{arm}: {current_video_device(texts[arm])} -> {path}')
        changed = [arm for arm in new_texts if new_texts[arm] != texts[arm]]
        if not changed:
            out('Config already matches, nothing to change.')
            return 0
        if prompt('Write? [y/N]').strip().lower() not in ('y', 'yes'):
            out('Not written.')
            return 0
    except CameraError as e:
        out(f'{e}. Stop the camera launch first (the cameras must not be in use). '
            'Nothing written.')
        return 1
    except (EOFError, KeyboardInterrupt):
        out('\nAborted. Nothing written.')
        return 1

    for arm in changed:
        with open(files[arm], 'w') as f:
            f.write(new_texts[arm])
        out(f'Wrote {files[arm]}')
    out('Restart the camera launch for the change to take effect.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config-dir', help='directory with the gripper camera YAMLs '
                        '(default: the rebel_cameras config, resolved through the install symlink)')
    args = parser.parse_args()
    config_dir = args.config_dir
    if config_dir is None:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory('rebel_cameras'), 'config')
        config_dir = os.path.dirname(os.path.realpath(os.path.join(share, CONFIG_FILES['left'])))
    return run(config_dir)


if __name__ == '__main__':
    sys.exit(main())
