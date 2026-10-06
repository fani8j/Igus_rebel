"""ctypes loader for the windowed JPEG decoder (``jpeg_window.c``).

``load_decoder()`` returns a :class:`WindowDecoder` or ``None`` (never raises); the caller then keeps
using ``cv2.imdecode``.  Library lookup order: ``$SIDE_TAPE_JPEG_WINDOW_LIB``, a prebuilt
``_jpeg_window.so`` next to this file (``build_jpeg_window.sh``), else compile the C source on first use
into ``$SIDE_TAPE_JPEG_CACHE`` (default ``~/.cache/side_tape_ros``), keyed by the source hash.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OK, BAD_ARGS, UNSUPPORTED, LIBJPEG_ERROR, LIBJPEG_WARNING, NOMEM = range(6)
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "jpeg_window.c"


def _compile() -> Path:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise RuntimeError("no C compiler found")
    if not _SRC.is_file():
        raise RuntimeError(f"{_SRC} missing")
    cache = Path(os.environ.get("SIDE_TAPE_JPEG_CACHE", Path.home() / ".cache" / "side_tape_ros"))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"jpeg_window_{hashlib.sha1(_SRC.read_bytes()).hexdigest()[:12]}.so"
    if target.is_file():
        return target
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)  # compile aside, then atomic rename (no races)
    os.close(fd)
    try:
        proc = subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", tmp, str(_SRC), "-ljpeg"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode:
            raise RuntimeError(f"compile failed: {proc.stderr.strip()[-400:]}")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def _find_library() -> Path:
    env = os.environ.get("SIDE_TAPE_JPEG_WINDOW_LIB")
    if env:
        return Path(env)
    prebuilt = _HERE / "_jpeg_window.so"
    if prebuilt.is_file():
        return prebuilt
    return _compile()


class WindowDecoder:
    def __init__(self, path):
        self.path = str(path)
        lib = ctypes.CDLL(self.path)
        i, vp = ctypes.c_int, ctypes.c_void_p
        lib.jpeg_decode_window_m.argtypes = [vp, ctypes.c_ulong, i, i, i, i, vp, vp, vp, i]
        lib.jpeg_decode_window_m.restype = i
        lib.jpeg_window_plan.argtypes = [i] * 9 + [vp]
        lib.jpeg_window_plan.restype = i
        self._lib = lib
        self._dims = (ctypes.c_int * 2)()
        self._dims_ptr = ctypes.addressof(self._dims)

    def decode(self, data: np.ndarray, x0: int, y0: int, w: int, h: int, out: np.ndarray, margin: int = 16):
        """Decode the raw-frame window [x0,x0+w) x [y0,y0+h) into ``out`` ((h, w, 3) uint8, contiguous).

        Returns ``(status, image_width, image_height)``; status 0 means ``out`` holds BGR pixels."""
        if out.shape != (h, w, 3) or out.dtype != np.uint8 or not out.flags.c_contiguous:
            return BAD_ARGS, 0, 0
        data = np.ascontiguousarray(data, dtype=np.uint8)
        rc = self._lib.jpeg_decode_window_m(data.ctypes.data, data.size, x0, y0, w, h, out.ctypes.data,
                                            self._dims_ptr, self._dims_ptr + 4, margin)
        return rc, self._dims[0], self._dims[1]

    def plan(self, W, H, mcu_w, mcu_h, x0, y0, w, h, margin=16):
        r = (ctypes.c_int * 4)()
        rc = self._lib.jpeg_window_plan(W, H, mcu_w, mcu_h, x0, y0, w, h, margin, ctypes.addressof(r))
        return None if rc else tuple(r)


def load_decoder():
    """Return a WindowDecoder or None (see :func:`load_decoder_or_reason` for the reason)."""
    return load_decoder_or_reason()[0]


def load_decoder_or_reason():
    try:
        return WindowDecoder(_find_library()), ""
    except Exception as exc:  # noqa: BLE001 - any build/load problem -> caller falls back to cv2
        return None, str(exc)
