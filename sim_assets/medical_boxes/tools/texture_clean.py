"""Helpers for extract_textures.py: remove photo noise from a rectified carton face.

All functions are deterministic (no randomness; OpenCV inpaint/blur only).  Images are
BGR uint8, coordinates are (x, y) in the pixel frame of the image being cleaned.
"""
import cv2
import numpy as np


def luma(img):
    f = img.astype(np.float32)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def poly_mask(shape, polys):
    """uint8 {0,255} mask from a list of polygons (lists of (x, y))."""
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 255)
    return m


def soft(mask, grow=1, sigma=1.5):
    """Feathered 0..1 alpha from a binary mask (grown by `grow` px first)."""
    m = mask
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        m = cv2.dilate(m, k)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sigma)
    a = np.clip(a * 1.6, 0, 1)
    a[m > 0] = 1.0
    return a


def blend(img, fill, alpha):
    out = img.astype(np.float32) * (1 - alpha[..., None]) + fill.astype(np.float32) * alpha[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def mirror_bands(img, band):
    """Replace the outermost (top, bottom, left, right) px by the mirrored interior."""
    t, b, l, r = band
    out = img.copy()
    H, W = out.shape[:2]
    if t:
        out[:t] = out[2 * t - 1:t - 1:-1]
    if b:
        out[H - b:] = out[H - b - 1:H - 2 * b - 1:-1]
    if l:
        out[:, :l] = out[:, 2 * l - 1:l - 1:-1]
    if r:
        out[:, W - r:] = out[:, W - r - 1:W - 2 * r - 1:-1]
    return out


def inpaint(img, mask, scale=4, radius=3, method=cv2.INPAINT_NS, grow=2, sigma=2.0):
    """Fill `mask` from its surroundings: inpaint a downscaled copy, upsample, blend."""
    H, W = img.shape[:2]
    h, w = max(1, H // scale), max(1, W // scale)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    mg = cv2.dilate(mask, k) if grow else mask
    ms = cv2.resize(mg, (w, h), interpolation=cv2.INTER_AREA)
    ms = (ms > 20).astype(np.uint8) * 255
    ms = cv2.dilate(ms, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, ms, radius, method)
    big = cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)
    return blend(img, big, soft(mg, 0, sigma))


def stock_fill(img, mask, refs, keep=()):
    """Paint `mask` with flat white label stock.  `refs` = [(y, (x0, y0, x1, y1)), ...]:
    clean stock patches whose (bright-half median) colour is linearly interpolated
    along y.  `keep` = rects in which dark ink pixels (still-intact printing) are left."""
    ys = np.array([r[0] for r in refs], np.float32)
    cols = np.array([sample(img, r[1], "white") for r in refs], np.float32)
    H, W = img.shape[:2]
    fill = np.empty((H, W, 3), np.float32)
    for c in range(3):
        fill[:, :, c] = np.interp(np.arange(H), ys, cols[:, c])[:, None]
    m = mask.copy()
    ink = (luma(img) < 150).astype(np.uint8) * 255
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    for (x0, y0, x1, y1) in keep:                 # keep only the ink inside these rects
        m[y0:y1, x0:x1] = cv2.bitwise_and(m[y0:y1, x0:x1], cv2.bitwise_not(ink[y0:y1, x0:x1]))
    return blend(img, fill, soft(m, 0, 1.2))


def row_fill(img, mask, xa, xb, smooth=4.0, feather=6.0, match=None, taper_y=None,
             keep_ink=None):
    """Fill `mask` with the per-row median colour of the clean columns xa..xb (the
    artwork is a vertical gradient that is constant across the width).

    match   : rect (x0, y0, x1, y1) of clean artwork at the fill's side; the profile
              gets the per-channel additive difference there (clamped +-25), faded out
              linearly from the rect's middle row to `taper_y`.
    keep_ink: (luma_below, sat_above): pixels darker / more saturated than this are
              printed ink (logo, text) and are not overwritten."""
    prof = np.median(img[:, xa:xb].astype(np.float32), axis=1)           # (H, 3)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1, 3), (1, 0), smooth, sigmaY=smooth)
    prof = prof.reshape(-1, 3)
    if match is not None:
        x0, y0, x1, y1 = match
        d = np.clip(np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0)
                    - np.median(prof[y0:y1], axis=0), -25, 25)
        ym = (y0 + y1) / 2.0
        wy = np.ones(len(prof), np.float32)
        if taper_y is not None:
            wy = np.clip(1 - (np.arange(len(prof)) - ym) / (taper_y - ym), 0, 1)
        prof = prof + wy[:, None] * d[None, :]
    fill = np.repeat(prof.reshape(-1, 1, 3), img.shape[1], axis=1)
    m = mask
    if keep_ink is not None:
        f = img.astype(np.float32)
        ink = ((luma(img) < keep_ink[0]) | (f.max(axis=2) - f.min(axis=2) > keep_ink[1]))
        ink = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        m = cv2.bitwise_and(mask, cv2.bitwise_not(ink))
    return blend(img, fill, soft(m, 1, feather))


def _design(xy, deg):
    x, y = xy[:, 0], xy[:, 1]
    cols = [np.ones_like(x), x] if deg == "x" else [np.ones_like(x), x, y]
    if deg == 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def flatten(img, rects, deg=1, lo=0.85, hi=1.2, exclude=None):
    """Divide out a smooth illumination field fitted (low-order 2-D polynomial) to the
    luminance of large uniform print regions `rects` (x0, y0, x1, y1).  `exclude` is an
    optional binary mask of pixels not to fit (hands / glare).  deg: 1 = plane, 2 =
    quadratic, "x" = constant + horizontal slope only (when the rects span too little y).
    Returns (img, info)."""
    H, W = img.shape[:2]
    L = cv2.GaussianBlur(luma(img), (0, 0), 4)
    pts, vals = [], []
    for (x0, y0, x1, y1) in rects:
        ys, xs = np.mgrid[y0:y1:6, x0:x1:6]
        ok = np.ones(xs.shape, bool)
        if exclude is not None:
            ok = exclude[ys, xs] == 0
        pts.append(np.stack([xs[ok] / W * 2 - 1, ys[ok] / H * 2 - 1], 1))
        vals.append(L[ys, xs][ok])
    pts, vals = np.concatenate(pts), np.concatenate(vals)
    keep = np.ones(len(vals), bool)
    for _ in range(3):                                   # trim outliers (ink, specks)
        A = _design(pts[keep], deg)
        c, *_ = np.linalg.lstsq(A, vals[keep], rcond=None)
        res = vals - _design(pts, deg) @ c
        s = max(np.std(res[keep]), 1.0)
        keep = np.abs(res) < 2.0 * s
    ys, xs = np.mgrid[0:H, 0:W]
    grid = np.stack([xs.ravel() / W * 2 - 1, ys.ravel() / H * 2 - 1], 1)
    fit = (_design(grid, deg) @ c).reshape(H, W).astype(np.float32)
    ref = float(np.median(_design(pts[keep], deg) @ c))
    gain = np.clip(ref / np.maximum(fit, 1.0), lo, hi)
    out = np.clip(img.astype(np.float32) * gain[..., None] + 0.5, 0, 255).astype(np.uint8)
    return out, dict(ref=ref, gmin=float(gain.min()), gmax=float(gain.max()),
                     resid=float(np.std(res[keep])))


def sample(img, rect, mode):
    """Representative BGR of a patch.  mode 'white': median of the brighter half (label
    stock between ink); 'median': plain median."""
    x0, y0, x1, y1 = rect
    p = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if mode == "white":
        lp = p @ np.array([0.114, 0.587, 0.299], np.float32)
        p = p[lp >= np.percentile(lp, 50)]
    return np.median(p, axis=0)


def solve_matrix(meas, target, ridge=0.02):
    """3x3 colour matrix M (BGR in -> BGR out, no offset: black stays black) with
    M @ meas[n] ~= target[n] for every reference colour n.

    Starts from the diagonal gain G that maps the anchor ('white', else the first
    colour) onto its target, then adds the smallest (ridge-damped min-norm) change that
    brings the other reference colours onto their targets, so channels that the
    references do not constrain keep the plain white-balance gain."""
    names = list(meas)
    a = "white" if "white" in meas else names[0]
    G = np.diag(np.asarray(target[a], np.float64) / np.asarray(meas[a], np.float64))
    X = np.stack([np.asarray(meas[n], np.float64) for n in names], 1)       # 3 x k
    T = np.stack([np.asarray(target[n], np.float64) for n in names], 1)
    R = T - G @ X
    lam = ridge * float(np.mean(np.sum(X * X, axis=0)))
    M = G + R @ np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T)
    return M


def apply_matrix(img, M):
    out = img.astype(np.float32) @ M.T.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def smear_down(img, poly, ref_rows=6, sigma_x=2.0, ref_off=2):
    """Below the polygon's top edge, continue each column with the (x-blurred) mean of
    the `ref_rows` rows ending `ref_off` px above the edge."""
    H, W = img.shape[:2]
    m = poly_mask(img.shape, [poly]) > 0
    out = img.astype(np.float32).copy()
    src = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma_x, sigmaY=0.01)
    for x in range(W):
        ys = np.nonzero(m[:, x])[0]
        if len(ys) == 0:
            continue
        y0 = ys[0]
        col = src[max(0, y0 - ref_rows - ref_off):max(1, y0 - ref_off), x].mean(axis=0)
        out[ys, x] = col
    a = soft(m.astype(np.uint8) * 255, 0, 1.0)
    return blend(img, out, a)


def mirror_left(img, knots, blur=1.0):
    """Replace a left-edge band whose width varies with y (knots = [(y, width), ...],
    linearly interpolated) by the mirrored interior columns."""
    H, W = img.shape[:2]
    ys = np.array([k[0] for k in knots], np.float32)
    ws = np.array([k[1] for k in knots], np.float32)
    w = np.interp(np.arange(H), ys, ws)
    out = img.astype(np.float32).copy()
    for y in range(H):
        n = int(round(w[y]))
        if n > 0:
            out[y, :n] = img[y, 2 * n - 1:n - 1:-1]
    a = np.zeros((H, W), np.float32)
    for y in range(H):
        a[y, :int(round(w[y]))] = 1.0
    a = cv2.GaussianBlur(a, (0, 0), blur)
    a[:, :4] = np.maximum(a[:, :4], 1.0)
    return blend(img, out, np.clip(a * 1.0, 0, 1))


def clone(img, mask, dx, dy, feather=1.5):
    """Fill `mask` with the pixels at (x + dx, y + dy) (clone-stamp from nearby clean artwork)."""
    H, W = img.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    src = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return blend(img, src, soft(mask, 1, feather))
