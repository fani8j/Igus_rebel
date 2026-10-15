"""Shared (session-scoped) expensive fixtures."""
import pytest

from rebel_dual_arm_calibration import solver, synthetic as S, validation
from test_loop_closure import World


@pytest.fixture(scope="session")
def good_world():
    """Zero joint offsets, 8 touches per board point and arm (different wrist configurations) plus 4 fit stickers
    (+-140 mm around the board) and 2 held-out check stickers (one on a 40 mm block, one ~190 mm out), 3 repeats each;
    solved, with the held-out validation (3 folds)."""
    if S.rebel_urdf_path() is None:
        pytest.skip("igus_rebel2.urdf not found")
    w = World(0.0, seed=6)
    w.sticker_pos = S.default_sticker_positions(w.scene)
    w.touches = (S.simulate_touches(w.scene, w.table, w.kin, repeats=8, seed=36)
                 + S.simulate_sticker_touches(w.scene, w.kin, w.sticker_pos, repeats=3, seed=38))
    res = w.solve()
    res.report["validation"] = validation.cross_validate(res, solver.SolverOptions(), w.touches, w.kin, k=3)
    return w, res
