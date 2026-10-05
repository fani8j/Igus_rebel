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
             keep_ink=None, keep_warm=None, clamp=None, grow=1, ramp_y=None):
    """Fill `mask` with the per-row median colour of the clean columns xa..xb (the
    artwork is a vertical gradient that is constant across the width).

    match   : rect (x0, y0, x1, y1) of clean artwork at the fill's side; the profile
              gets the per-channel additive difference there (clamped +-25), faded out
              linearly from the rect's middle row to `taper_y`.
    keep_ink: (luma_below, sat_above): pixels darker / more saturated than this are
              printed ink (logo, text) and are not overwritten.
    keep_warm: pixels with R - B above this (yellow tags on an aqua / blue field) are
              not overwritten.
    clamp   : (y0, y1): the profile is held constant above y0 and below y1 (the
              outermost rows are the fold, where glare and background collect)."""
    prof = np.median(img[:, xa:xb].astype(np.float32), axis=1)           # (H, 3)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1, 3), (1, 0), smooth, sigmaY=smooth)
    prof = prof.reshape(-1, 3)
    if clamp is not None:
        prof[:clamp[0]] = prof[clamp[0]]
        prof[clamp[1] + 1:] = prof[clamp[1]]
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
    if keep_warm is not None:
        f = img.astype(np.float32)
        warm = (f[..., 2] - f[..., 0] > keep_warm).astype(np.uint8) * 255
        warm = cv2.dilate(warm, np.ones((5, 5), np.uint8))
        m = cv2.bitwise_and(m, cv2.bitwise_not(warm))
    a = soft(m, grow, feather)
    if ramp_y is not None:                       # (y0, n): fade the fill in over n rows from y0
        a = a * np.clip((np.arange(img.shape[0], dtype=np.float32) - ramp_y[0]) / ramp_y[1], 0, 1)[:, None]
    return blend(img, fill, a)


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


def col_fill(img, rect, ref, sigma_x=0.0, fy=1.0, fx=0.0):
    """Fold-edge repair: inside rect (x0, y0, x1, y1) every column becomes the median of
    its own rows ref = (ya, yb) (the nearest clean rows), optionally smoothed along x.
    fy / fx: feather (px) on the rect sides that are not on the image border."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = rect
    ya, yb = ref
    med = np.median(img[ya:yb, x0:x1].astype(np.float32), axis=0)            # (w, 3)
    if sigma_x > 0:
        med = cv2.GaussianBlur(med.reshape(1, -1, 3), (0, 0), sigma_x, sigmaY=0.01).reshape(-1, 3)
    fill = img.astype(np.float32).copy()
    fill[y0:y1, x0:x1] = med[None]
    ay = np.ones(y1 - y0, np.float32)
    if fy > 0:
        ramp = np.minimum((np.arange(y1 - y0) + 1.0) / (fy + 1.0), 1.0)
        if y0 > 0:
            ay = np.minimum(ay, ramp)
        if y1 < H:
            ay = np.minimum(ay, ramp[::-1])
    ax = np.ones(x1 - x0, np.float32)
    if fx > 0:
        ramp = np.minimum((np.arange(x1 - x0) + 1.0) / (fx + 1.0), 1.0)
        if x0 > 0:
            ax = np.minimum(ax, ramp)
        if x1 < W:
            ax = np.minimum(ax, ramp[::-1])
    a = np.zeros((H, W), np.float32)
    a[y0:y1, x0:x1] = ay[:, None] * ax[None, :]
    return blend(img, fill, a)


def edge_columns(img, n, ref=(5, 7)):
    """Replace the outermost n left columns by the mean of columns ref (column
    replication; a swirl or gradient that runs off the edge keeps its shape)."""
    out = img.copy()
    out[:, :n] = np.round(img[:, ref[0]:ref[1]].astype(np.float32).mean(axis=1, keepdims=True)).astype(np.uint8)
    return out


def unveil(img, rect, stock=None, ink_drop=55, sigma=3.0, sigma_far=10.0, margin=2.0,
           soft_px=1.5, stock_q=25, stock_sigma=20.0, gate_thr=6.0):
    """Remove glossy glare (an additive veil) from white label stock inside rect.
    ink   = pixels that are `ink_drop` darker than the brightest pixel within 4 px;
    stock = the unveiled stock colour, per column: the `stock_q` percentile of the non-ink
            pixels of that column (glare only ever adds light), smoothed along x
            (`stock_sigma`); a fixed colour may be passed instead;
    veil  = per channel max(0, pixel - stock - margin) under non-ink pixels; under the ink
            the veil of the neighbouring non-ink pixels (normalised Gaussian convolution,
            wide fallback where the ink is dense), subtracted from every pixel, so the stock
            becomes flat while the printing keeps its contrast.  Only applied where the
            excess is `gate_thr` over the stock for at least 2 px (real glare)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = rect
    F = img.astype(np.float32)
    L = luma(img)
    ink = (L < cv2.dilate(L, np.ones((9, 9), np.uint8)) - ink_drop).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    w = (1 - ink).astype(np.float32)
    box = np.zeros((H, W), np.float32)
    box[y0:y1, x0:x1] = 1.0
    w *= box
    if stock is None:
        sub = F[y0:y1, x0:x1].copy()
        sub[w[y0:y1, x0:x1] == 0] = np.nan
        q = np.nanpercentile(sub, stock_q, axis=0)                        # (w, 3)
        ok = np.isfinite(q[:, 0])
        xs = np.arange(q.shape[0])
        for c in range(3):
            q[:, c] = np.interp(xs, xs[ok], q[ok, c])
        q = cv2.GaussianBlur(q.reshape(1, -1, 3), (0, 0), stock_sigma, sigmaY=0.01,
                             borderType=cv2.BORDER_REPLICATE).reshape(-1, 3)
        st = np.zeros((H, W, 3), np.float32)
        st[:, x0:x1] = q[None]
    else:
        st = np.asarray(stock, np.float32)[None, None, :]
    # per-pixel excess over the unveiled stock; the veil under non-ink pixels is that excess,
    # the veil under the ink is propagated from the neighbouring non-ink pixels
    vp = np.clip(F - st - margin, 0, None)

    def nconv(s):
        num = cv2.GaussianBlur(vp * w[..., None], (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        return num, den
    num, den = nconv(sigma)
    numf, denf = nconv(sigma_far)
    use_near = den > 0.25
    v_ink = np.where(use_near[..., None], num / np.maximum(den, 1e-3)[..., None],
                     numf / np.maximum(denf, 1e-3)[..., None])
    v_ink *= (np.where(use_near, den, denf) > 0.02)[..., None]
    veil = np.where(w[..., None] > 0, vp, v_ink) * box[..., None]
    # gate: only where there is real glare (>= gate_thr over the stock, at least 2 px thick)
    vl = np.mean(vp, axis=2) * box
    g = (vl > gate_thr).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    g = cv2.dilate(g, np.ones((5, 5), np.uint8)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), soft_px)
    veil = veil * g[..., None]
    return np.clip(F - veil + 0.5, 0, 255).astype(np.uint8)


def silver_fill(img, x0, x1, y_search, xa, xb, ramp=0.0, bluish=40, edge_k=15):
    """Replace the silver band right of x0 (hand shadow) by the clean silver of columns
    xa..xb: per-row median colour, plus an additive horizontal `ramp` (0 at x0 -> ramp at
    x1).  The blue artwork above is kept: per column the band starts at the first row
    from y_search whose B - R is below `bluish` (median-filtered across x)."""
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    silver = (f[..., 0] - f[..., 2]) < bluish
    yb = np.full(W, H, np.int32)
    for x in range(x0, x1):
        ys = np.nonzero(silver[y_search:, x])[0]
        if len(ys):
            yb[x] = y_search + ys[0]
    if edge_k > 1:                                   # running median across x
        seg = np.pad(yb[x0:x1].astype(np.float32), edge_k // 2, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(seg, edge_k)
        yb[x0:x1] = np.round(np.median(win, axis=1)).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    for x in range(x0, x1):
        m[yb[x]:, x] = 255
    prof = np.median(f[:, xa:xb], axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1, 3), (1, 0), 4.0, sigmaY=4.0).reshape(-1, 3)
    fill = np.repeat(prof.reshape(-1, 1, 3), W, axis=1)
    t = np.clip((np.arange(W, dtype=np.float32) - x0) / float(x1 - x0), 0, 1)
    fill = fill + ramp * t[None, :, None]
    return blend(img, fill, soft(m, 0, 0.8))
