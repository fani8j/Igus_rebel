# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/box_profiles.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Box profiles: construction facts the camera cannot measure, per box type.

Geometry (corners, top plane, height) is MEASURED by the pipeline; a profile
only supplies what cannot be seen: the height used by the fold model when no
measurement is available, the outer-wall dimensions for the side seed, the
apparent seam view shift, which sides the flap gap joins (``seam_joins``),
and which per-type fixes were validated (depth-edge fix, R10 geometry axis).

``small`` and ``big`` reproduce today's hard-coded constants exactly.
``generic`` is any other box: no construction facts, measured height only,
seam from tape evidence only (docs/multi_box_plan.md).

Classification:
- :func:`classify_legacy` = today's rule (top long side > 150 mm -> big, else
  small). Used where outputs must stay bit-identical.
- :func:`classify` = nearest known profile whose top dims (and height, when
  given) are all within tolerance, else ``generic``. Box dims are unordered
  (long/short compared sorted).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

LEGACY_BIG_LONG_SIDE_M = 0.150   # today's class split (fold_edge, seam view shift, side seed)
LEGACY_NAMES = ("small", "big")  # the two boxes of the original table cell


@dataclass(frozen=True)
class BoxProfile:
    name: str
    #: flat top face as measured by the detector (long, short) mm; None = unknown
    top_dims_mm: tuple[float, float] | None
    #: box height mm (user-measured); None = use the measured height
    height_mm: float | None
    #: outer wall dims {"along_seam": mm, "across_seam": mm}; None values = unknown
    outer_dims_mm: dict = field(default_factory=lambda: {"along_seam": None,
                                                         "across_seam": None})
    #: apparent seam shift into the V-gap (mm, + = away from the camera)
    view_shift_mm: float = 0.0
    #: which top sides the flap gap joins: "long_sides" | "short_sides" | None (unknown)
    seam_joins: str | None = None
    #: R10 geometry seam axis allowed (only where validated)
    geometry_axis_validated: bool = False
    #: depth-edge fix validated for this type
    depth_edge_validated: bool = False
    #: classification tolerances
    dims_tol_mm: float = 8.0
    height_tol_mm: float = 5.0
    #: R10 geometry-axis size ranges (long, short) mm; None = no R10 ranges
    r10_long_mm: tuple[float, float] | None = None
    r10_short_mm: tuple[float, float] | None = None
    #: seam confirmation tolerance (mm). small/big carry today's config value
    #: (seam_estimator confirm_max_offset_mm, which stays the authority for
    #: them: bit-identical); a generic box has no validated view-shift model,
    #: so it confirms with this looser tolerance (applied via the profile only).
    confirm_max_offset_mm: float | None = 1.5
    #: side seed: how far BELOW 0 the outer-wall offset (outer dim - measured top
    #: dim)/2 may be (mm) before the top face counts as "bigger than the box"
    #: (wrong box / bad corners); a smaller negative value is clamped to 0
    wall_neg_tol_mm: float = 1.0
    #: which line the seam stage PUBLISHES: "prior" = the projective midline,
    #: confirmed (ok) or not (estimated) by the observed gap (small/big/generic:
    #: unchanged); "observed" = the straight line fitted to the OBSERVED flap gap
    #: (gap valley; dark line only as fallback), ok only when that observation
    #: covers ``seam_min_obs_coverage`` of the slices and passes the gap plausibility checks
    seam_publish: str = "prior"
    #: observed mode: minimum fraction of slices that support the fitted line
    seam_min_obs_coverage: float = 0.60
    #: observed mode: half-width (mm, across the seam) the gap search window covers at least
    seam_gap_window_mm: float = 12.0
    #: observed mode: cut rule ``seam_too_curved`` above this maximum perpendicular
    #: deviation (mm) of the observed gap from the published straight line; None = no rule
    seam_max_curvature_mm: float | None = None

    @property
    def height_m(self) -> float | None:
        return None if self.height_mm is None else self.height_mm / 1000.0

    @property
    def is_generic(self) -> bool:
        return self.name == "generic"

    @property
    def is_legacy(self) -> bool:
        """small / big: the two original boxes whose constants are hard-wired
        into fold_edge / seam view shift / side seed by the 150 mm class rule."""
        return self.name in LEGACY_NAMES

    @property
    def own_constants(self) -> bool:
        """generic or a later profile (not small/big): view shift, seam confirm
        tolerance and fold height come from the PROFILE (or the measured
        height), never from the legacy 150 mm class tables."""
        return not self.is_legacy


#: flat top (long, short) mm of the large carton as the detector measures it
#: (guided top plane + SAM3 outline, support hidden; see the profile below)
LARGE_TOP_DIMS_MM = (394.5, 328.6)

PROFILES: dict[str, BoxProfile] = {
    "small": BoxProfile(
        name="small", top_dims_mm=(110.0, 98.0), height_mm=86.3,
        outer_dims_mm={"along_seam": 106.25, "across_seam": 112.0},
        view_shift_mm=3.2, seam_joins="long_sides",
        geometry_axis_validated=True, depth_edge_validated=True,
        r10_long_mm=(104.0, 116.0), r10_short_mm=(90.0, 104.0)),
    "big": BoxProfile(
        name="big", top_dims_mm=(193.0, 105.0), height_mm=134.25,
        outer_dims_mm={"along_seam": None, "across_seam": None},
        view_shift_mm=0.5, seam_joins=None,
        geometry_axis_validated=False, depth_edge_validated=False),
    # Large carton of the dual-arm cell (2026-10-12): outer 390 x 320 x 480 mm,
    # brown with a centred tape; the flap gap runs parallel to the LONG (390)
    # side. It stands on a stand smaller than the box, so its height can never
    # be measured against the support (support_hidden): the profile height is
    # used. Nothing about it is validated yet: seam_joins unknown (R10 never),
    # no depth-edge fix, no apparent view shift, generic-like seam confirm
    # tolerance. dims_tol 15 mm (vs 8): the flat-top dims were measured on ONE
    # capture, 1.2 m away at ~2.7 mm depth noise, with a +-2 mm lid bulge;
    # tighten once ~20 captures are in.
    "large_392": BoxProfile(
        name="large_392", top_dims_mm=LARGE_TOP_DIMS_MM, height_mm=480.0,
        outer_dims_mm={"along_seam": 390.0, "across_seam": 320.0},
        # The seam is published as the straight line fitted to the OBSERVED flap gap (not the
        # projective midline): on 15 accepted dual-arm captures (2026-10-12, 1.2-1.3 m) the real
        # gap lies +5.5 mm (per capture 4.5-6.4) behind the midline, away from the camera
        # (the quad's front edge over-extends onto the rounded front lip), and it is slightly
        # CURVED (bows 1.2-4.1 mm, median 2.7, from its best straight line: convex flaps that
        # the tape does not hold). No viewing shift: the observed line is the target.
        # Curvature limit 6 mm = 1.5x the largest bow seen (4.1 mm); a straight cut then stays
        # within ~half the bow of the real gap.
        view_shift_mm=0.0, seam_joins=None,
        seam_publish="observed", seam_min_obs_coverage=0.60,
        seam_gap_window_mm=12.0, seam_max_curvature_mm=6.0,
        geometry_axis_validated=False, depth_edge_validated=False,
        dims_tol_mm=15.0,
        # the measured flat top (394.5 x 328.6 mm) is BIGGER than the outer box
        # (390 x 320): the depth plane of this cell is tilted 2-4 deg, which
        # inflates the 3-D dims by up to ~9 mm; clamp (not reject) up to this
        wall_neg_tol_mm=10.0),
    "generic": BoxProfile(name="generic", top_dims_mm=None, height_mm=None,
                          confirm_max_offset_mm=4.0),
}

#: env: comma list of profile names removed from classification everywhere
#: (leave-one-type-out validation: the box then becomes ``generic`` end to end)
ENV_EXCLUDE = "AUTOLABEL_V2_PROFILES_EXCLUDE"


def excluded_raw() -> str:
    """The raw ``AUTOLABEL_V2_PROFILES_EXCLUDE`` value ("" when unset)."""
    return os.environ.get(ENV_EXCLUDE, "")


def excluded_names() -> tuple[str, ...]:
    """Known profile names excluded by ``AUTOLABEL_V2_PROFILES_EXCLUDE``
    (read at call time). An unknown name, or ``generic`` (which can never be
    excluded), raises ``ValueError``: a typo must not silently run the
    unexcluded configuration of a leave-one-type-out experiment."""
    names = [t.strip().lower() for t in excluded_raw().split(",") if t.strip()]
    bad = [n for n in names if n not in PROFILES or n == "generic"]
    if bad:
        raise ValueError(
            f"{ENV_EXCLUDE}={excluded_raw()!r}: unknown/unexcludable "
            f"profile name(s) {bad}; valid: "
            f"{sorted(n for n in PROFILES if n != 'generic')}")
    return tuple(dict.fromkeys(names))


def active_profiles() -> dict:
    """PROFILES without the excluded ones."""
    ex = excluded_names()
    return {n: p for n, p in PROFILES.items() if n not in ex}


def get(name: str) -> BoxProfile:
    return PROFILES[name]


def legacy_class_name(long_side_m: float) -> str:
    """Today's raw class name (150 mm rule), ignoring any exclusion."""
    return "big" if float(long_side_m) > LEGACY_BIG_LONG_SIDE_M else "small"


def classify_legacy(long_side_m: float) -> BoxProfile:
    """Today's rule, bit-identical: top long side > 150 mm -> big, else small.
    A class removed by ``AUTOLABEL_V2_PROFILES_EXCLUDE`` gives ``generic``."""
    name = legacy_class_name(long_side_m)
    return PROFILES[name] if name in active_profiles() else PROFILES["generic"]


def classify(long_mm: float, short_mm: float, height_mm: float | None = None,
             profiles: dict | None = None) -> tuple[BoxProfile, dict]:
    """Nearest known profile with all dims within tolerance, else generic.

    Returns ``(profile, info)``; ``info`` has per-profile deviations and the
    reason. Height is a veto only when both the measurement and the profile
    height exist.
    """
    profiles = active_profiles() if profiles is None else profiles
    lo, sh = sorted((float(long_mm), float(short_mm)), reverse=True)
    info: dict = {"measured_mm": [round(lo, 1), round(sh, 1)],
                  "measured_height_mm": None if height_mm is None else round(float(height_mm), 1),
                  "candidates": {}}
    best, best_dev = None, None
    for p in profiles.values():
        if p.top_dims_mm is None:
            continue
        pl, ps = sorted(p.top_dims_mm, reverse=True)
        dev = max(abs(lo - pl), abs(sh - ps))
        ok = dev <= p.dims_tol_mm
        hdev = None
        if ok and height_mm is not None and p.height_mm is not None:
            hdev = abs(float(height_mm) - p.height_mm)
            ok = hdev <= p.height_tol_mm
        info["candidates"][p.name] = {"dims_dev_mm": round(dev, 1),
                                      "height_dev_mm": None if hdev is None else round(hdev, 1),
                                      "match": bool(ok)}
        if ok and (best_dev is None or dev < best_dev):
            best, best_dev = p, dev
    if best is None:
        info["reason"] = "no_profile_within_tolerance"
        return profiles.get("generic", PROFILES["generic"]), info
    info["reason"] = "matched"
    return best, info


# ---------------------------------------------------------------------------
# Views of the known profiles in the shapes the legacy config dicts use
# (insertion order matches the old literals: JSON dumps stay identical).
# ---------------------------------------------------------------------------

def known_names() -> tuple[str, ...]:
    """Names of the active profiles with construction facts (not generic)."""
    return tuple(n for n, p in active_profiles().items()
                 if p.top_dims_mm is not None)


def heights_m() -> dict:
    """{class: height_m} for the active known profiles, order big, small."""
    act = active_profiles()
    return {n: PROFILES[n].height_m for n in ("big", "small") if n in act}


def view_shift_mm_table() -> dict:
    act = active_profiles()
    return {n: PROFILES[n].view_shift_mm for n in ("big", "small") if n in act}


def known_dims_mm_table() -> dict:
    """{class: [long, short] mm} of the active profiles, order small, big."""
    act = active_profiles()
    return {n: [float(v) for v in PROFILES[n].top_dims_mm]
            for n in ("small", "big") if n in act}


# ---------------------------------------------------------------------------
# Effective profile for the pipeline
# ---------------------------------------------------------------------------

#: A box that matches no profile within tolerance is still treated as the
#: legacy class (flag ``profile_mismatch``) when it is only a near miss:
#: today's two boxes with a badly measured top / height must not change.
NEAR_MISS_DIMS_MM = 25.0
NEAR_MISS_HEIGHT_MM = 20.0
#: height flag threshold for a known profile (measured vs profile)
HEIGHT_MISMATCH_MM = 5.0


def known_height_distance_mm(height_mm: float) -> float:
    """Distance (mm) from ``height_mm`` to the nearest known profile height."""
    # legacy boxes only: later profiles (large_392) cannot be height-measured
    # in their own scene (support hidden) and must not change today's
    # generic-height decision of the original cell
    hs = [p.height_mm for p in active_profiles().values()
          if p.height_mm is not None and p.is_legacy]
    if not hs:
        return float("inf")
    return min(abs(float(height_mm) - h) for h in hs)


def is_generic_height(height_mm: float | None) -> bool:
    """True when a measured height is far from every known profile height
    (more than the near-miss tolerance): the box cannot be a known one."""
    if height_mm is None:
        return False
    return known_height_distance_mm(height_mm) > NEAR_MISS_HEIGHT_MM


def resolve(long_m: float, short_m: float, height_mm: float | None = None
            ) -> tuple[BoxProfile, dict]:
    """Effective profile + audit info for the pipeline.

    ``long_m`` / ``short_m`` = top-face side lengths (m); ``height_mm`` the
    measured height or None. The legacy 150 mm rule picks the class; a
    ``generic`` verdict of :func:`classify` only takes effect when the box is
    not a near miss of that class (see NEAR_MISS_*). ``info`` has ``name``,
    ``legacy_class``, ``matched`` (classify verdict), ``profile_mismatch``,
    ``classification`` and ``reason``.
    """
    legacy = classify_legacy(long_m)          # generic when its class is excluded
    legacy_raw = legacy_class_name(long_m)
    lo, sh = sorted((float(long_m) * 1000.0, float(short_m) * 1000.0),
                    reverse=True)
    matched, cinfo = classify(lo, sh, height_mm)
    info: dict = {"legacy_class": legacy_raw, "matched": matched.name,
                  "classification": cinfo, "profile_mismatch": False,
                  "profiles_excluded": list(excluded_names()),
                  "profiles_excluded_raw": excluded_raw()}
    if legacy.is_generic:
        # the legacy class is excluded (leave-one-type-out): truly generic,
        # no near-miss / height-match rescue
        eff = PROFILES["generic"]
        info["reason"] = "legacy_class_excluded"
    elif not matched.is_generic and not matched.is_legacy:
        # a later profile (e.g. large_392): matched by its own top dims (and
        # height when one is measured); the 150 mm legacy class says nothing
        eff = matched
        info["reason"] = "matched"
    elif not matched.is_generic:
        eff = legacy
        info["reason"] = "matched"
        if matched.name != legacy.name:
            info["profile_mismatch"] = True
            info["reason"] = "matched_other_than_legacy"
    else:
        pl, ps = sorted(legacy.top_dims_mm, reverse=True)
        ddev = max(abs(lo - pl), abs(sh - ps))
        hdev = (None if height_mm is None or legacy.height_mm is None
                else abs(float(height_mm) - legacy.height_mm))
        # (a) top dims close and height (when known) close, or (b) the
        # measured height matches the legacy profile within the flag
        # tolerance although the dims are far off: a known box whose top
        # quad is badly wrong (e.g. 101 x 34 mm for the 110 x 98 box).
        # No measured height -> never a rescue (height_unavailable, not a
        # guess): such a box with dims off a profile is generic.
        near = bool(hdev is not None
                    and ((ddev <= NEAR_MISS_DIMS_MM
                          and hdev <= NEAR_MISS_HEIGHT_MM)
                         or hdev <= HEIGHT_MISMATCH_MM))
        info["near_miss"] = {"dims_dev_mm": round(ddev, 1),
                             "height_dev_mm": None if hdev is None
                             else round(hdev, 1), "near": bool(near)}
        if near:
            eff = legacy
            info["profile_mismatch"] = True
            info["reason"] = ("near_miss_keep_legacy"
                              if ddev <= NEAR_MISS_DIMS_MM
                              else "height_matches_dims_off_keep_legacy")
        else:
            eff = PROFILES["generic"]
            info["reason"] = "no_profile_within_tolerance"
    info["name"] = eff.name
    return eff, info
