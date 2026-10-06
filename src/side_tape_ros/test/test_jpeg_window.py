"""Windowed JPEG decode: window math (rotate_180, borders, MCU alignment) and pixel equality vs cv2.imdecode."""

import cv2
import numpy as np
import pytest

from side_tape_ros import core_path  # noqa: F401
from side_tape_ros import jpeg_window
from side_tape_ros.fast_frame import PartialRoiDecoder, RoiCanvas, mcu_aligned_window, raw_window
from precision_box_detector.side_tape.geometry import derotate_image, roi_around

W, H = 1280, 720


@pytest.fixture(scope="module")
def decoder():
    d, why = jpeg_window.load_decoder_or_reason()
    if d is None:
        pytest.skip(f"windowed decoder unavailable: {why}")
    return d


def _image(w=W, h=H, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), (xx + yy) % 256], -1).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    return cv2.add(base, noise)


def _jpeg(img, sampling=None, quality=85):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if sampling is not None:
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling]
    ok, buf = cv2.imencode(".jpg", img, params)
    assert ok
    return buf


# ---- window math -------------------------------------------------------------------------------

@pytest.mark.parametrize("rot", [True, False])
@pytest.mark.parametrize("roi", [(0, 0, 320, 240), (960, 480, 1280, 720), (500, 100, 820, 340), (0, 480, 320, 720)])
def test_raw_window_maps_to_derotated_roi(rot, roi):
    raw = _image()
    rx0, ry0, rx1, ry1 = raw_window(roi, W, H, rot)
    assert 0 <= rx0 < rx1 <= W and 0 <= ry0 < ry1 <= H
    x0, y0, x1, y1 = roi
    assert (rx1 - rx0, ry1 - ry0) == (x1 - x0, y1 - y0)
    canvas = RoiCanvas(W, H, rot).fill_window(raw[ry0:ry1, rx0:rx1], roi)
    assert np.array_equal(canvas[y0:y1, x0:x1], derotate_image(raw, rot)[y0:y1, x0:x1])


@pytest.mark.parametrize("mcu", [(8, 8), (16, 8), (16, 16)])
@pytest.mark.parametrize("margin", [16, 0, 5])
def test_aligned_window_properties(mcu, margin):
    rng = np.random.default_rng(1)
    rects = [(0, 0, 1, 1), (W - 1, H - 1, 1, 1), (0, 0, W, H), (3, 5, 320, 240), (16, 16, 16, 16), (17, 15, 1, 2)]
    for _ in range(300):
        w, h = int(rng.integers(1, 500)), int(rng.integers(1, 400))
        rects.append((int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1)), w, h))
    for x, y, w, h in rects:
        ax0, ay0, ax1, ay1 = mcu_aligned_window(x, y, w, h, W, H, *mcu, margin)
        # snapped to the MCU grid on the top-left; bottom-right on the grid or the image edge
        assert ax0 % mcu[0] == 0 and ay0 % mcu[1] == 0
        assert ax1 % mcu[0] == 0 or ax1 == W
        assert ay1 % mcu[1] == 0 or ay1 == H
        # contains the needed window plus >= margin on every side that is not the image border
        assert ax0 <= max(0, x - margin) and ay0 <= max(0, y - margin)
        assert ax1 >= min(W, x + w + margin) and ay1 >= min(H, y + h + margin)
        # ...and never grows by more than margin + one MCU
        assert x - ax0 < margin + mcu[0] and y - ay0 < margin + mcu[1]
        assert ax1 - (x + w) < margin + mcu[0] or ax1 == W


def test_aligned_window_rejects_bad_windows():
    for args in [(-1, 0, 10, 10), (0, 0, 0, 10), (W - 5, 0, 10, 10), (0, H - 5, 10, 10)]:
        assert mcu_aligned_window(*args, W, H) is None


def test_python_plan_matches_c_plan(decoder):
    rng = np.random.default_rng(2)
    for mcu in [(8, 8), (16, 8), (16, 16)]:
        for _ in range(200):
            w, h = int(rng.integers(1, 500)), int(rng.integers(1, 400))
            x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
            margin = int(rng.choice([0, 7, 16, 33]))
            assert decoder.plan(W, H, *mcu, x, y, w, h, margin) == mcu_aligned_window(x, y, w, h, W, H, *mcu, margin)
    assert decoder.plan(W, H, 16, 16, W - 5, 0, 10, 10) is None


# ---- pixel equality ----------------------------------------------------------------------------

@pytest.mark.parametrize("sampling", [None, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
                                      cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
@pytest.mark.parametrize("size", [(W, H), (203, 117)])
def test_window_decode_equals_imdecode(decoder, sampling, size):
    img = _image(*size)
    buf = _jpeg(img, sampling)
    full = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    w_img, h_img = size
    rects = [(0, 0, 1, 1), (w_img - 1, h_img - 1, 1, 1), (0, 0, w_img, h_img), (0, 0, min(320, w_img), min(240, h_img)),
             (w_img - min(320, w_img), h_img - min(240, h_img), min(320, w_img), min(240, h_img)),
             (15, 17, 33, 31), (16, 16, 16, 16), (w_img - 17, 3, 17, 40)]
    rng = np.random.default_rng(3)
    for _ in range(60):
        w, h = int(rng.integers(1, w_img // 2)), int(rng.integers(1, h_img // 2))
        rects.append((int(rng.integers(0, w_img - w + 1)), int(rng.integers(0, h_img - h + 1)), w, h))
    for x, y, w, h in rects:
        for margin in (16, 40):
            out = np.empty((h, w, 3), np.uint8)
            status, iw, ih = decoder.decode(buf, x, y, w, h, out, margin)
            assert (status, iw, ih) == (0, w_img, h_img)
            assert np.array_equal(out, full[y:y + h, x:x + w]), (x, y, w, h, margin)


@pytest.mark.parametrize("rot", [True, False])
@pytest.mark.parametrize("center", [(618.4, 125.0), (5.0, 3.0), (1275.0, 715.0), (640.0, 360.0), (100.5, 700.2)])
def test_partial_roi_decoder_matches_normal_path(decoder, rot, center):
    buf = _jpeg(_image(seed=4))
    roi = roi_around(center, 150, 110, W, H)
    x0, y0, x1, y1 = roi
    ref = derotate_image(cv2.imdecode(buf, cv2.IMREAD_COLOR), rot)
    pd = PartialRoiDecoder(decoder, RoiCanvas(W, H, rot))
    out = pd.decode_roi(buf, roi)
    assert out is not None
    assert np.array_equal(out[y0:y1, x0:x1], ref[y0:y1, x0:x1])
    roi2 = roi_around((900, 500), 100, 80, W, H)  # reuse: the window moves
    out = pd.decode_roi(buf, roi2)
    a, b, c, d = roi2
    assert np.array_equal(out[b:d, a:c], ref[b:d, a:c])


def test_margin_zero_may_differ_only_at_window_edge(decoder):
    """Why the margin exists: without it, fancy chroma upsampling at the crop edge sees no neighbour pixel."""
    buf = _jpeg(_image())
    full = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    x, y, w, h = 15, 17, 33, 31
    out = np.empty((h, w, 3), np.uint8)
    assert decoder.decode(buf, x, y, w, h, out, 0)[0] == 0
    diff = np.abs(out.astype(int) - full[y:y + h, x:x + w]).max(axis=2)
    assert not diff[2:h - 2, 2:w - 2].any()


# ---- failure handling (caller falls back to cv2.imdecode) --------------------------------------

def test_failures_return_status_not_crash(decoder):
    buf = _jpeg(_image())
    out = np.empty((240, 320, 3), np.uint8)
    assert decoder.decode(np.frombuffer(b"not a jpeg at all", np.uint8), 0, 0, 320, 240, out)[0] != 0
    assert decoder.decode(np.zeros(0, np.uint8), 0, 0, 320, 240, out)[0] != 0
    assert decoder.decode(buf[:len(buf) // 3], 0, 0, 320, 240, out)[0] != 0          # truncated
    assert decoder.decode(buf, W - 100, 0, 320, 240, out)[0] != 0                     # outside the image
    assert decoder.decode(buf, 0, 0, 100, 100, out)[0] != 0                           # wrong output shape
    corrupt = buf.copy()
    corrupt[200:400] = 0x55
    st = decoder.decode(corrupt, 0, 0, 320, 240, out)[0]
    assert st in (0, jpeg_window.LIBJPEG_ERROR, jpeg_window.LIBJPEG_WARNING)           # no crash either way
    assert decoder.decode(buf, 0, 0, 320, 240, out)[0] == 0                            # still usable afterwards


def test_gray_and_exif_jpegs_are_rejected(decoder):
    gray = np.frombuffer(cv2.imencode(".jpg", cv2.cvtColor(_image(), cv2.COLOR_BGR2GRAY))[1], np.uint8)
    out = np.empty((100, 100, 3), np.uint8)
    assert decoder.decode(gray, 0, 0, 100, 100, out)[0] == jpeg_window.UNSUPPORTED
    buf = bytearray(_jpeg(_image()).tobytes())
    exif = b"\xff\xe1\x00\x10Exif\x00\x00" + b"\x00" * 8  # APP1 segment, length 16
    buf[2:2] = exif
    assert decoder.decode(np.frombuffer(bytes(buf), np.uint8), 0, 0, 100, 100, out)[0] == jpeg_window.UNSUPPORTED


def test_decode_roi_returns_none_on_bad_jpeg(decoder):
    pd = PartialRoiDecoder(decoder, RoiCanvas(W, H, True))
    assert pd.decode_roi(np.frombuffer(b"garbage", np.uint8), (0, 0, 320, 240)) is None
    assert pd.failures == 1
    small = _jpeg(_image(640, 360))  # wrong image size for this canvas
    assert pd.decode_roi(small, (0, 0, 320, 240)) is None
