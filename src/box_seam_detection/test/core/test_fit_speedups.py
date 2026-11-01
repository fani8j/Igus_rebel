"""2026-11-02 box-fit speedups: 3x3 scatter eigen-solve in plane_fit._fit_plane_svd
(AUTOLABEL_V2_EXACT_SVD=1 = the old N x 3 SVD), memoised z-buffer region winners, and
depth_top.ransac_plane's refit subset gathered by index."""
import numpy as np
import pytest

from box_seam_detection.core.autolabel import depth_top
from box_seam_detection.core.pbd import plane_fit, sam3_depth_fusion as fusion


def _cloud(n=60000, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.2, 0.2, (n, 2))
    z = 1.1 + 0.04 * xy[:, 0] - 0.07 * xy[:, 1] + rng.normal(0, 0.0015, n)
    return np.c_[xy, z]


@pytest.mark.parametrize("weighted", [False, True])
def test_eigh_plane_matches_svd(monkeypatch, weighted):
    pts = _cloud()
    w = np.random.default_rng(2).uniform(0.2, 1.0, pts.shape[0]) if weighted else None
    monkeypatch.setenv("AUTOLABEL_V2_EXACT_SVD", "1")
    n_svd, d_svd = plane_fit._fit_plane_svd(pts, w)
    monkeypatch.setenv("AUTOLABEL_V2_EXACT_SVD", "0")
    n_eig, d_eig = plane_fit._fit_plane_svd(pts, w)
    if n_eig @ n_svd < 0:
        n_eig, d_eig = -n_eig, -d_eig
    assert np.allclose(n_eig, n_svd, atol=1e-12) and abs(d_eig - d_svd) < 1e-12


def test_ransac_irls_same_plane_both_solvers(monkeypatch):
    pts = _cloud()
    pts[:3000, 2] += 0.03                                   # an off-plane flap
    est = {}
    for exact in ("1", "0"):
        monkeypatch.setenv("AUTOLABEL_V2_EXACT_SVD", exact)
        est[exact] = plane_fit.fit_plane_ransac_irls(pts, "f", thresh_m=0.005, iters=300,
                                                     seed=0, min_inliers=100)
    assert np.allclose(est["0"].normal, est["1"].normal, atol=1e-12)
    assert abs(est["0"].d - est["1"].d) < 1e-12
    assert np.array_equal(est["0"].inlier_mask, est["1"].inlier_mask)


def _projected(seed=3, n=40000, W=200, H=150):
    rng = np.random.default_rng(seed)
    bu = rng.integers(0, W, n)
    bv = rng.integers(0, H, n)
    xyz = np.c_[rng.normal(size=(n, 2)), rng.uniform(0.5, 2.0, n)]
    return fusion.ProjectedCloud("ok", n, n, W, H, bu=fusion._ro(bu), bv=fusion._ro(bv),
                                 bxyz=fusion._ro(xyz), lin=fusion._ro(bv * W + bu))


def test_region_winners_memo_is_exact(monkeypatch):
    fusion.clear_caches()
    pc = _projected()
    region = (20, 10, 120, 90)
    first = fusion.zbuffer_winners(pc, region)
    again = fusion.zbuffer_winners(pc, region)
    assert again is first                                   # memo hit
    monkeypatch.setenv("AUTOLABEL_DISABLE_CACHE", "1")
    fresh = fusion.zbuffer_winners(pc, region)              # recomputed
    for f in ("wu", "wv", "xyz_win"):
        assert np.array_equal(getattr(fresh, f), getattr(first, f))
    monkeypatch.delenv("AUTOLABEL_DISABLE_CACHE")
    other = _projected(seed=4)                              # another cloud drops the memo
    fusion.zbuffer_winners(other, region)
    assert all(v[0] is other for v in fusion._REGION_MEMO.values())
    fusion.clear_caches()
    assert not fusion._REGION_MEMO


def test_depth_top_ransac_refit_subset_unchanged():
    """ransac_plane gathers every-step-th inlier by index; same plane as fitting the stride
    subset of pts[inliers] (the old gather)."""
    pts = np.r_[_cloud(70000), _cloud(20000, seed=5) + [0, 0, 0.3]]
    n, d, inl, rms = depth_top.ransac_plane(pts, thresh=0.007, iters=300, seed=0, downsample=20000)
    # one more refit from the returned plane, the old way
    old = pts[inl]
    step = int(-(-old.shape[0] // depth_top._REFIT_MAX))
    n_old, d_old = depth_top._fit_plane_svd(old[::step]) if old.shape[0] > depth_top._REFIT_MAX \
        else depth_top._fit_plane_svd(old)
    sel = np.flatnonzero(inl)
    sel = sel[::step] if sel.size > depth_top._REFIT_MAX else sel
    n_new, d_new = depth_top._fit_plane_svd(pts[sel])
    assert np.array_equal(n_old, n_new) and d_old == d_new
    assert inl.sum() > 60000 and rms < 0.004
