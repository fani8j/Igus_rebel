# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/side_cutting_point.py
# at commit 08d1041 (working tree identical to HEAD, unmodified).
# Unchanged.
import numpy as np

def compute_midpoints(BL, FL, ML, MR, BR, FR):
    """
    Compute 4 midpoints from 6 input points.
    Each input can be a scalar, tuple, list, or numpy array (e.g. (x, y) or (x, y, z)).

    Formulas:
        BML = (BL + ML) / 2   # Back Middle Left
        FML = (FL + ML) / 2   # Front Middle Left
        BMR = (BR + MR) / 2   # Back Middle Right
        FMR = (FR + MR) / 2   # Front Middle Right
    """
    # Convert all inputs to numpy arrays so vector math works uniformly
    # regardless of whether inputs are scalars or multi-dim coordinates
    BL, FL, ML, MR, BR, FR = (np.array(p, dtype=float) for p in (BL, FL, ML, MR, BR, FR))

    BML = (BL + ML) / 2   # Back Middle Left
    FML = (FL + ML) / 2   # Front Middle Left
    BMR = (BR + MR) / 2   # Back Middle Right
    FMR = (FR + MR) / 2   # Front Middle Right

    return BML, FML, BMR, FMR


# ------ Example usage ------
if __name__ == "__main__":
    BL = (0, 0)
    FL = (0, 10)
    ML = (5, 5)
    MR = (10, 5)
    BR = (15, 0)
    FR = (15, 10)

    BML, FML, BMR, FMR = compute_midpoints(BL, FL, ML, MR, BR, FR)

    print("BML =", BML)
    print("FML =", FML)
    print("BMR =", BMR)
    print("FMR =", FMR)