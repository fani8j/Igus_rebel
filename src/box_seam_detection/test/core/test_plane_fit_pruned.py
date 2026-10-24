"""Pruned (early-abandoning) RANSAC hypothesis scoring in
``fit_plane_ransac_irls`` returns exactly what the plain scan returned.

``_old_scan`` is a verbatim copy of the pre-change hypothesis loop (draw
order, first-wins ties, ``count == n`` early exit). The pruned scan scores row
blocks with the same ``block @ normal`` expression, which is bit-identical to
the full-array one only for a single-threaded BLAS and 16-byte-aligned data,
so the tests pin the BLAS to one thread (and skip when that is not possible).
"""
import ctypes

import numpy as np
import pytest

from box_seam_detection.core.pbd import plane_fit as PF


def _openblas():
    try:
        np.zeros(3) @ np.zeros(3)
        with open("/proc/self/maps") as fh:
            libs = sorted({ln.split()[-1] for ln in fh if "openblas" in ln})
        for path in libs:
            lib = ctypes.CDLL(path)
            for suffix in ("64_", "_", ""):
                get = getattr(lib, f"openblas_get_num_threads{suffix}", None)
                setn = getattr(lib, f"openblas_set_num_threads{suffix}", None)
                if get is not None and setn is not None:
                    get.restype = ctypes.c_int
                    return get, setn
    except Exception:  # noqa: BLE001
        pass
    return None


@pytest.fixture(autouse=True)
def single_thread_blas():
    lib = _openblas()
    if lib is None:
        pytest.skip("cannot pin the BLAS to one thread")
    get, setn = lib
    old = int(get())
    setn(1)
    try:
        yield
    finally:
        setn(old)


def _old_scan(pts, rng, iters, thresh_m, cos_tol=None, align_unit=None):
    """Verbatim pre-change loop -> (best_count, best_n, best_d, best_mask)."""
    n = pts.shape[0]
    best_n, best_d, best_count, best_mask = None, 0.0, 0, None
    for _ in range(iters):
        i0, i1, i2 = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            continue
        normal = normal / norm
        if cos_tol is not None and align_unit is not None:
            if abs(float(normal @ align_unit)) < cos_tol:
                continue
        d = float(-(normal @ p0))
        mask = np.abs(pts @ normal + d) < thresh_m
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count, best_n, best_d, best_mask = count, normal, d, mask
            if count == n:
                break
    return best_count, best_n, best_d, best_mask


def _scene(n, seed, inlier_frac=0.8, second_plane=0.0, noise=0.0008, quantize=None):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.25, 0.25, size=(n, 2))
    z = 0.8 + 0.3 * xy[:, 0] + noise * rng.normal(size=n)
    r = rng.random(n)
    out = r > inlier_frac
    z[out] += rng.uniform(-0.12, 0.12, size=int(out.sum()))
    if second_plane:
        sec = (r > inlier_frac - second_plane) & ~out
        z[sec] += 0.02
    p = np.c_[xy, z]
    if quantize:
        p = np.round(p / quantize) * quantize
    return np.ascontiguousarray(p)


def _assert_same(a, b):
    assert a[0] == b[0]
    if a[1] is None:
        assert b[1] is None and a[3] is None and b[3] is None
        return
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert np.array_equal(a[3], b[3])


@pytest.mark.parametrize("n,seed,kw", [
    (5000, 0, {}),
    (20000, 1, {"inlier_frac": 0.95}),
    (20003, 2, {"inlier_frac": 0.6, "second_plane": 0.25}),
    (30001, 3, {"inlier_frac": 0.5}),
    (8192, 4, {"inlier_frac": 0.9, "noise": 0.004}),
    (12000, 5, {"inlier_frac": 0.9, "quantize": 0.002}),   # many borderline / tied residuals
])
@pytest.mark.parametrize("thresh_m", [0.003, 0.0015])
def test_pruned_scan_equals_old_scan(n, seed, kw, thresh_m):
    pts = _scene(n, seed, **kw)
    ref = _old_scan(pts, np.random.default_rng(7), 400, thresh_m)
    hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(7), 400, None, None))
    assert PF._pruned_search_available(pts)
    got = PF._best_hypothesis_pruned(pts, hyps, thresh_m)
    assert got is not None
    _assert_same(ref, got)
    _assert_same(ref, PF._scan_reference(pts, hyps, thresh_m, n))


def test_pruned_scan_equals_old_scan_with_alignment_gate():
    pts = _scene(15000, 11, inlier_frac=0.7)
    au = np.array([0.0, 0.0, 1.0])
    cos_tol = float(np.cos(np.radians(25.0)))
    ref = _old_scan(pts, np.random.default_rng(3), 500, 0.003, cos_tol, au)
    hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(3), 500, cos_tol, au))
    _assert_same(ref, PF._best_hypothesis_pruned(pts, hyps, 0.003))


@pytest.mark.parametrize("n", [3, 4, 5, 7, 8, 9, 31, 33, 100, 1001])
def test_pruned_scan_small_clouds(n):
    for seed in range(5):
        pts = _scene(max(n, 3), seed, inlier_frac=0.7, noise=0.002)
        ref = _old_scan(pts, np.random.default_rng(seed), 60, 0.003)
        hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(seed), 60, None, None))
        got = PF._best_hypothesis_pruned(pts, hyps, 0.003)
        assert got is not None
        _assert_same(ref, got)


def test_ties_first_hypothesis_wins():
    pts = _scene(10000, 21, inlier_frac=0.9)
    hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(5), 300, None, None))
    best = _old_scan(pts, np.random.default_rng(5), 300, 0.003)
    # the winner appears several times (equal count), at later positions and
    # (as a copy) before everything else: the first occurrence must win
    win = next(i for i, (nm, d) in enumerate(hyps) if np.array_equal(nm, best[1]) and d == best[2])
    dup = [(h[0].copy(), h[1]) for h in hyps]
    seq = [dup[win]] + dup + [dup[win]] * 3 + dup[::-1]
    got = PF._best_hypothesis_pruned(pts, seq, 0.003)
    assert got is not None and got[1] is seq[0][0]
    _assert_same(PF._scan_reference(pts, seq, 0.003, pts.shape[0]), got)
    # winner only after equal-count rivals
    seq2 = dup[:win][::-1] + [dup[win]] * 2
    ref2 = PF._scan_reference(pts, seq2, 0.003, pts.shape[0])
    _assert_same(ref2, PF._best_hypothesis_pruned(pts, seq2, 0.003))


def test_all_inliers_first_hypothesis_wins():
    xy = np.random.default_rng(0).uniform(-0.2, 0.2, size=(9000, 2))
    pts = np.ascontiguousarray(np.c_[xy, 0.7 + 0 * xy[:, 0]])   # exactly planar
    ref = _old_scan(pts, np.random.default_rng(1), 100, 0.003)
    hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(1), 100, None, None))
    got = PF._best_hypothesis_pruned(pts, hyps, 0.003)
    assert ref[0] == 9000 and got[0] == 9000 and got[1] is hyps[0][0]
    _assert_same(ref, got)


def test_no_inliers_anywhere():
    pts = _scene(6000, 31)
    hyps = list(PF._iter_hypotheses(pts, np.random.default_rng(2), 50, None, None))
    got = PF._best_hypothesis_pruned(pts, hyps, 1e-300)      # nothing is < 1e-300 but exact hits
    ref = PF._scan_reference(pts, hyps, 1e-300, pts.shape[0])
    _assert_same(ref, got)
    assert PF._best_hypothesis_pruned(pts, [], 0.003) == (0, None, 0.0, None)


@pytest.mark.parametrize("kw", [
    {"inlier_frac": 0.8}, {"inlier_frac": 0.4, "second_plane": 0.3}, {"quantize": 0.001}])
def test_fit_plane_ransac_irls_identical_with_and_without_pruning(monkeypatch, kw):
    pts = _scene(25000, 41, **kw)
    outs = []
    for min_pts in (10 ** 9, 4096):
        monkeypatch.setattr(PF, "_PRUNE_MIN_POINTS", min_pts)
        outs.append(PF.fit_plane_ransac_irls(pts, "f", thresh_m=0.003, iters=600, seed=0))
    a, b = outs
    assert np.array_equal(a.normal, b.normal) and a.d == b.d
    assert np.array_equal(a.inlier_mask, b.inlier_mask)
    assert (a.rms_mm, a.median_mm, a.p90_mm, a.p95_mm, a.confidence) == \
        (b.rms_mm, b.median_mm, b.p90_mm, b.p95_mm, b.confidence)


def test_fit_with_non_finite_rows_and_gate_identical(monkeypatch):
    pts = _scene(20000, 51, inlier_frac=0.7)
    pts[::97, 1] = np.nan
    res = []
    for min_pts in (10 ** 9, 4096):
        monkeypatch.setattr(PF, "_PRUNE_MIN_POINTS", min_pts)
        res.append(PF.fit_plane_ransac_irls(
            pts, "f", thresh_m=0.003, iters=500, seed=3,
            align_with=np.array([0.0, 0.0, 1.0]), max_align_angle_deg=20.0))
    assert np.array_equal(res[0].normal, res[1].normal) and res[0].d == res[1].d
    assert np.array_equal(res[0].inlier_mask, res[1].inlier_mask)


def test_no_plane_error_is_unchanged(monkeypatch):
    pts = _scene(6000, 61)
    msgs = []
    for min_pts in (10 ** 9, 4096):
        monkeypatch.setattr(PF, "_PRUNE_MIN_POINTS", min_pts)
        with pytest.raises(ValueError) as ei:
            PF.fit_plane_ransac_irls(pts, "f", thresh_m=0.003, iters=100, min_inliers=5999)
        msgs.append(str(ei.value))
    assert msgs[0] == msgs[1]


def test_unsafe_environment_uses_the_reference_scan(monkeypatch):
    pts = _scene(6000, 71)
    monkeypatch.setattr(PF, "_blas_threads", lambda: 4)
    assert not PF._pruned_search_available(pts)
    monkeypatch.setattr(PF, "_blas_threads", lambda: None)
    assert not PF._pruned_search_available(pts)
    monkeypatch.setattr(PF, "_blas_threads", lambda: 1)
    assert PF._pruned_search_available(pts)
    # misaligned base address (8-byte only) -> reference scan
    buf = np.empty(6000 * 3 + 2)
    base = buf if buf.ctypes.data % 16 == 0 else buf[1:]
    odd = base[1:6000 * 3 + 1].reshape(6000, 3)
    assert not PF._pruned_search_available(odd)

    def boom(*a, **k):
        raise AssertionError("pruned scan must not run")
    monkeypatch.setattr(PF, "_blas_threads", lambda: 4)
    monkeypatch.setattr(PF, "_best_hypothesis_pruned", boom)
    PF.fit_plane_ransac_irls(pts, "f", thresh_m=0.003, iters=50)


def test_canary_failure_falls_back(monkeypatch):
    pts = _scene(6000, 81)
    ref = PF.fit_plane_ransac_irls(pts, "f", thresh_m=0.003, iters=200)
    monkeypatch.setattr(PF, "_best_hypothesis_pruned", lambda *a, **k: None)
    got = PF.fit_plane_ransac_irls(pts, "f", thresh_m=0.003, iters=200)
    assert np.array_equal(ref.normal, got.normal) and ref.d == got.d
