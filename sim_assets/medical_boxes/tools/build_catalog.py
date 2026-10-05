#!/usr/bin/env python3
"""Build the medical-carton catalogue inputs from catalog/ (dims, spines, faces, research).

Writes, next to the research data:
    catalog/catalog_boxes.yaml  same schema as boxes.yaml (feed it to build_usd.py / verify_usd.py
                                with --boxes-yaml); physics sections are copied verbatim from boxes.yaml
    catalog/sizes.yaml          normalised size catalogue per family (explicit combinations where
                                the source gives a full matrix or per-diameter rules; ranges otherwise)
    catalog/SIZE_CATALOG.md     readable tables + SKU counts

Plain python (needs PyYAML only):
    python3 tools/build_catalog.py

Inputs (read only): catalog/dims.yaml, spines.yaml, faces.yaml (optional), families_draft.yaml,
research/*.json, boxes.yaml.
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT = os.path.join(ROOT, "catalog")
PHOTO_DIR = "source/setup_photos_full"
FACE_ORDER = ["front", "back", "top", "bottom", "left", "right"]

MAIN5 = {  # catalogue id (boxes.yaml) -> research id
    "agent_monorail": "bsci_agent_monorail_dcb",
    "angiolite": "ivascular_angiolite_des",
    "accuforce": "terumo_accuforce_nc",
    "ultimaster_nagomi": "terumo_ultimaster_nagomi_des",
    "conqueror_nc_pro": "apt_conqueror_nc_pro",
}
# research-only families with usable size data but no carton asset
EXTRA_RESEARCH = ["bbraun_sequent_please_neo_dcb", "orbusneich_scoreflex_green_variant"]

# --------------------------------------------------------------------------------------------
# Corrected identity per family: (label, brand, product, device_type).
# device_type differs from dims.yaml only where the research shows the draft type was wrong.
# --------------------------------------------------------------------------------------------
META = {
    "abbott_armada_35": ("Armada 35", "Abbott", "Abbott Armada 35 PTA balloon catheter (OTW, 0.035 in)", "pta_balloon"),
    "abbott_xience_skypoint": ("Xience Skypoint", "Abbott", "Abbott XIENCE Skypoint everolimus-eluting coronary stent system", "coronary_des"),
    "apt_medical_tresfort_35": ("Tresfort 35", "APT Medical", "APT Medical Tresfort 35 PTA balloon catheter (OTW, 0.035 in)", "pta_balloon"),
    "avantigard_d_logo": ("Avantigard (unidentified)", "Unidentified", "Unidentified carton, spine reads 'AVANTIGARD' with a blue D logo on black (maker not found)", "other"),
    "bard_conquest_pta": ("Conquest PTA", "Bard (BD)", "Bard Conquest high-pressure PTA balloon dilatation catheter (OTW, 0.035 in)", "pta_balloon"),
    "biosensors_chroma": ("Chroma", "Biosensors", "Biosensors Chroma cobalt-chromium coronary stent (bare metal)", "coronary_bms"),
    "biotronik_orsiro_mission": ("Orsiro Mission", "Biotronik", "Biotronik Orsiro Mission sirolimus-eluting coronary stent system", "coronary_des"),
    "biotronik_passeo_18": ("Passeo-18", "Biotronik", "Biotronik Passeo-18 PTA balloon catheter (OTW, 0.018 in)", "pta_balloon"),
    "biotronik_passeo_35_xeo": ("Passeo-35 Xeo", "Biotronik", "Biotronik Passeo-35 Xeo PTA balloon catheter (OTW, 0.035 in)", "pta_balloon"),
    "biotronik_pk_papyrus": ("PK Papyrus", "Biotronik", "Biotronik PK Papyrus covered coronary stent system", "covered_stent"),
    "biotronik_red_blue_unknown": ("Biotronik red/blue (unidentified)", "Biotronik (assumed)", "Unidentified red/white/blue carton labelled BIOTRONIK, product name not legible", "coronary_balloon_sc"),
    "boston_scientific_mustang": ("Mustang", "Boston Scientific", "Boston Scientific Mustang PTA balloon catheter (OTW, 0.035 in)", "pta_balloon"),
    "boston_scientific_synergy_xd": ("Synergy XD", "Boston Scientific", "Boston Scientific SYNERGY XD everolimus-eluting coronary stent system (Monorail)", "coronary_des"),
    "boston_scientific_v18_control_wire": ("V-18 ControlWire", "Boston Scientific", "Boston Scientific V-18 / V-14 ControlWire peripheral guidewire", "guidewire"),
    "bsci_mach1_guiding_catheter": ("Mach 1", "Boston Scientific", "Boston Scientific Mach 1 coronary guide catheter", "guiding_catheter"),
    "bsci_nc_emerge_monorail": ("NC Emerge Monorail", "Boston Scientific", "Boston Scientific NC EMERGE Monorail non-compliant PTCA balloon catheter", "coronary_balloon_nc"),
    "bsci_permanent_sled_bag_carton": ("Permanent Sled Bag", "Boston Scientific", "Boston Scientific Permanent Sled Bag (sterile bag for the MDU5 PLUS IVUS motordrive)", "other"),
    "bsci_wolverine_cutting_balloon": ("Wolverine Cutting Balloon", "Boston Scientific", "Boston Scientific WOLVERINE coronary cutting balloon (Monorail)", "scoring_cutting_balloon"),
    "elixir_desyne_x2": ("DESyne X2", "Elixir Medical", "Elixir Medical DESyne X2 novolimus-eluting coronary stent system", "coronary_des"),
    "ivascular_essential_pro_balloon": ("Essential Pro", "iVascular", "iVascular Essential Pro paclitaxel drug-coated PTCA balloon (RX)", "coronary_dcb"),
    "ivascular_oceanus_14_pro": ("Oceanus 14 Pro", "iVascular", "iVascular Oceanus 14 Pro PTA balloon catheter (OTW, 0.014 in)", "pta_balloon"),
    "ivascular_oceanus_18": ("Oceanus 18", "iVascular", "iVascular Oceanus 18 PTA balloon catheter (OTW, 0.018 in)", "pta_balloon"),
    "ivascular_oceanus_35": ("Oceanus 35", "iVascular", "iVascular Oceanus 35 PTA balloon catheter (OTW, 0.035 in)", "pta_balloon"),
    "medalliance_selution_slr_dcb": ("Selution SLR", "MedAlliance (Cordis)", "MedAlliance SELUTION SLR 014 sirolimus drug-eluting PTCA balloon", "coronary_dcb"),
    "medtronic_amphirion_deep": ("Amphirion Deep", "Medtronic", "Medtronic Amphirion Deep PTA balloon catheter (OTW, 0.014 in)", "pta_balloon"),
    "medtronic_export_advance_edge": ("Export Advance", "Medtronic", "Medtronic Export Advance 6F aspiration catheter", "other"),
    "medtronic_nc_euphora_edge": ("NC Euphora", "Medtronic", "Medtronic NC Euphora non-compliant PTCA balloon catheter (RX)", "coronary_balloon_nc"),
    "medtronic_pacific_plus": ("Pacific Plus", "Medtronic", "Medtronic Pacific Plus PTA balloon catheter (OTW, 0.018 in)", "pta_balloon"),
    "medtronic_prevail_dcb": ("Prevail DCB", "Medtronic", "Medtronic Prevail paclitaxel-coated PTCA balloon catheter (RX)", "coronary_dcb"),
    "medtronic_resolute_onyx": ("Resolute Onyx", "Medtronic", "Medtronic Resolute Onyx zotarolimus-eluting coronary stent system", "coronary_des"),
    "meril_nexgen_balloon": ("NexGen", "Meril", "Meril NEXGEN cobalt-chromium coronary stent (bare metal, not a balloon)", "coronary_bms"),
    "nse_alpha_scoring_balloon": ("NSE Alpha", "Goodman (NSE)", "Goodman NSE ALPHA coronary scoring balloon catheter", "scoring_cutting_balloon"),
    "orbusneich_combo_plus": ("COMBO Plus", "OrbusNeich", "OrbusNeich COMBO Plus dual-therapy sirolimus-eluting stent", "coronary_des"),
    "orbusneich_sapphire_3": ("Sapphire 3", "OrbusNeich", "OrbusNeich Sapphire 3 semi-compliant PTCA balloon catheter", "coronary_balloon_sc"),
    "orbusneich_scoreflex": ("Scoreflex", "OrbusNeich", "OrbusNeich Scoreflex dual-wire scoring PTCA balloon catheter", "scoring_cutting_balloon"),
    "teleflex_guideliner": ("GuideLiner", "Teleflex (Vascular Solutions)", "Teleflex GuideLiner V3 guide extension catheter", "guide_extension_catheter"),
    "terumo_heartrail_ii_guiding_catheter": ("Heartrail II", "Terumo", "Terumo Heartrail II PTCA guiding catheter", "guiding_catheter"),
    "terumo_navicross": ("NaviCross", "Terumo", "Terumo NaviCross peripheral support catheter (braided shaft, not a guidewire)", "support_catheter"),
    "terumo_ryurei_balloon": ("Ryurei", "Terumo", "Terumo Ryurei semi-compliant PTCA balloon catheter (RX)", "coronary_balloon_sc"),
    "trireme_tri_wedge": ("Tri-Wedge", "BrosMed", "BrosMed Tri-Wedge PTA scoring balloon catheter (OTW, 0.018 / 0.035 in; maker is BrosMed, not Trireme)", "scoring_cutting_balloon"),
    "unknown_blue_des": ("Unidentified blue DES", "Unidentified", "Unidentified dark blue carton with a large white 'DES' on the spine cap", "coronary_des"),
    "unknown_teal_boston_style": ("Synergy Megatron (probable)", "Boston Scientific (probable)", "Boston Scientific SYNERGY MEGATRON everolimus-eluting coronary stent system (Monorail; probable identification)", "coronary_des"),
}
# main-5 identities (display only, for sizes.yaml / SIZE_CATALOG.md; the boxes themselves live in boxes.yaml)
MAIN5_META = {
    "agent_monorail": ("Agent Monorail", "Boston Scientific", "coronary_dcb"),
    "angiolite": ("Angiolite", "iVascular", "coronary_des"),
    "accuforce": ("Accuforce", "Terumo", "coronary_balloon_nc"),
    "ultimaster_nagomi": ("Ultimaster Nagomi", "Terumo", "coronary_des"),
    "conqueror_nc_pro": ("Conqueror NC Pro", "APT Medical", "coronary_balloon_nc"),
}
EXTRA_META = {
    "bbraun_sequent_please_neo_dcb": ("SeQuent Please NEO", "B. Braun", "coronary_dcb"),
    "orbusneich_scoreflex_green_variant": ("Scoreflex NC/TRIO (green variant)", "OrbusNeich", "scoring_cutting_balloon"),
}

# --------------------------------------------------------------------------------------------
# Size expansion. Every function returns (fields, combinations, basis, kind, note).
#   kind = exact | lower_bound | approximate   (a count is only reported when combos are explicit)
#   basis = "full matrix stated" | "expanded from the source's per-diameter rules"
# Families not listed in EXPAND keep ranges only (matrix_notes verbatim): the source gives rules or
# ranges that cannot be turned into an exact list without inventing combinations.
# --------------------------------------------------------------------------------------------
DL = ["diameter_mm", "length_mm"]
DLS = ["diameter_mm", "length_mm", "shaft_cm"]
FULL = "full matrix stated"
RULES = "expanded from the source's per-diameter rules"


def grid(ds, ls):
    return [[d, l] for d in ds for l in ls]


def per_dia(table):
    """{diameter: lengths} -> [[d, l], ...]"""
    return [[d, l] for d, ls in table.items() for l in ls]


def group(ds, ls, table=None):
    t = table if table is not None else {}
    for d in ds:
        t[d] = list(ls)
    return t


def with_shaft(combos, shafts):
    return [c + [s] for c in combos for s in shafts]


def _agent():
    return DL, grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0], [12, 15, 20, 30]), FULL, "exact", \
        "Shaft 144 cm. A 40 mm length exists in 'limited sizes' (not itemised), so it is not counted."


def _emerge():
    t = group([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0], [6, 8, 12, 15, 20, 30])
    group([4.5, 5.0], [6, 8, 12, 15, 20], t)
    group([5.5, 6.0], [8, 12, 15, 20], t)
    return DL, per_dia(t), RULES, "exact", "Monorail matrix; the OTW version has the same matrix but is not counted."


def _wolverine():
    return DL, grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0], [6, 10, 15]), FULL, "exact", ""


def _mustang():
    base = per_dia({**group([3, 4, 5, 6, 7, 8], [20, 30, 40, 60, 80, 100, 120, 150, 180, 200]),
                    **group([9, 10, 12], [20, 30, 40, 60, 80])})
    combos = []
    for d, l in base:
        shafts = [75, 135]
        if l <= 100 or (d in (5, 6) and l == 120):
            shafts = [40] + shafts
        combos += [[d, l, s] for s in shafts]
    return DLS, combos, RULES, "exact", "Reproduces BSC's stated 203 catalogue codes."


def _synergy_xd():
    t = group([2.25], [8, 12, 16, 20, 24, 28, 32, 38])
    group([2.5, 2.75, 3.0, 3.5, 4.0], [8, 12, 16, 20, 24, 28, 32, 38, 48], t)
    group([4.5, 5.0], [12, 16, 20, 24, 28, 32], t)
    return DL, per_dia(t), FULL, "exact", "Per the 2026 US catalogue; older SYNERGY/OTW versions not counted."


def _v18():
    c = []
    for length in (182, 300):
        for taper in ("short", "long"):
            for tip in ("straight", "angled"):
                c.append(["V-14 (0.014 in)", length, taper, tip])
    for length, taper in ((150, "short"), (200, "short"), (300, "short"), (200, "long"), (300, "long")):
        c.append(["V-18 (0.018 in)", length, taper, "straight"])
    return ["wire", "length_cm", "taper", "tip"], c, "itemised UPNs / catalogue numbers only", "lower_bound", \
        "8 V-14 UPNs (EU page) + 5 V-18 catalogue numbers named in the research; V-18 also exists in 110 cm and a full V-18 UPN table was not found."


def _guideliner():
    return ["french_size", "ref"], [[5, "5569"], [5.5, "5570"], [6, "5571"], [7, "5572"], [8, "5573"]], FULL, "exact", \
        "One 150 cm length; the French size is the only variable."


def _megatron():
    return DL, grid([3.5, 4.0, 4.5, 5.0], [8, 12, 16, 20, 24, 28, 32]), FULL, "exact", "Probable identification, see confidence."


def _orsiro():
    t = group([2.25], [9, 13, 15, 18, 22, 26, 30])
    group([2.5, 2.75, 3.0, 3.5, 4.0], [9, 13, 15, 18, 22, 26, 30, 35, 40], t)
    group([4.5, 5.0], [9, 13, 15, 18, 22, 26, 30], t)
    return DL, per_dia(t), RULES, "exact", ""


def _papyrus():
    t = group([2.5], [15, 20])
    group([3.0, 3.5, 4.0, 4.5, 5.0], [15, 20, 26], t)
    return DL, per_dia(t), FULL, "exact", ""


def _combo():
    t = group([2.5, 2.75], [9, 13, 15, 18, 23, 28, 33])
    group([3.0, 3.5, 4.0], [9, 13, 15, 18, 23, 28, 33, 38], t)
    return DL, per_dia(t), FULL, "exact", "A 2.25 mm size appears in a CE/NL ordering brochure but not on the web table (not counted)."


def _sapphire():
    ds = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
    t = {d: [10, 15, 20] for d in ds}
    for d in (1.5, 2.0, 2.5, 3.0):
        t[d] = sorted(t[d] + [12])
    for d in (2.5, 3.0, 3.5):
        t[d] = sorted(t[d] + [30])
    return DL, per_dia(t), RULES, "exact", "The separate 0.85-1.25 mm line is not counted."


def _scoreflex():
    return DL, grid([2.0, 2.5, 3.0, 3.5, 4.0], [10, 15, 20]), FULL, "exact", ""


def _scoreflex_green():
    return DL, grid([1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0], [10, 15, 20]), FULL, "exact", \
        "Sub-line (NC vs TRIO) unresolved; both are stated as 8 x 3."


def _prevail():
    c = [x for x in grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0], [10, 15, 20, 25, 30])
         if not (x[1] == 30 and x[0] in (2.25, 2.75))]
    return DL, c, FULL, "exact", "2.25 x 30 and 2.75 x 30 are not offered."


def _onyx():
    t = group([2.0], [8, 12, 15, 18, 22, 26, 30])
    group([2.25, 2.5, 2.75, 3.0, 3.5, 4.0], [8, 12, 15, 18, 22, 26, 30, 34, 38], t)
    group([4.5, 5.0], [12, 15, 18, 22, 26, 30], t)
    return DL, per_dia(t), RULES, "exact", "RX sizes; the OTW version (2.0-4.0 mm only) is not counted. Some sizes are region dependent."


def _euphora():
    t = group([2.0, 2.25], [6, 8, 12, 15, 20])
    group([2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0], [6, 8, 12, 15, 20, 27], t)
    group([4.5], [8, 12, 15, 20], t)
    group([5.0], [8, 12, 15], t)
    return DL, per_dia(t), RULES, "exact", ""


def _xience():
    t = group([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0], [8, 12, 15, 18, 23, 28, 33, 38])
    for d in (2.5, 2.75, 3.0, 3.5, 4.0):
        t[d] = t[d] + [48]
    group([4.5, 5.0], [12, 15, 18, 23, 28, 33], t)
    return DL, per_dia(t), RULES, "exact", ""


def _armada():
    t = group([3.0], [20, 40])
    group([4.0, 5.0, 6.0, 7.0], [20, 40, 60, 80, 100, 120], t)
    group([8.0, 9.0, 10.0, 12.0, 14.0], [20, 40, 60, 80], t)
    return DLS, with_shaft(per_dia(t), [80, 135]), RULES, "lower_bound", \
        "Standard Armada 35 in 80 and 135 cm shafts. The Armada 35 LL long-balloon variants (4-7 mm x 150/200/250 mm) are only partly tabulated and not counted."


def _conqueror():
    t = group([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0], [6, 8, 12, 15, 20, 30])
    group([4.5, 5.0], [6, 8, 12, 15, 20], t)
    c = [[d, l, tip] for d, l in per_dia(t) for tip in ("spherical", "tapered")]
    return DL + ["tip"], c, RULES, "exact", "Each diameter x length exists with spherical and tapered tip. Shaft length is not published on the web page."


def _tresfort():
    t = group([3, 4, 5, 6, 7], [20, 30, 40, 60, 80, 100, 120, 150, 180, 200])
    group([8, 9, 10, 12], [20, 30, 40, 60, 80], t)
    return DLS, with_shaft(per_dia(t), [75, 100, 135]), RULES, "exact", "Same matrix for the three shaft lengths."


def _nexgen():
    t = group([2.0], [13, 16, 19, 24, 29, 32, 37, 40])
    group([2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5], [8, 13, 16, 19, 24, 29, 32, 37, 40], t)
    return DL, per_dia(t), RULES, "exact", ""


def _selution():
    return DL, grid([1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0], [10, 15, 20, 25, 30, 35, 40]), FULL, "exact", \
        "Availability varies by market."


def _nse():
    return DL, grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0], [13]), FULL, "exact", "2.00, 2.25 and 4.00 mm are made to order."


def _triwedge():
    c = []
    for wire, shafts in ((0.018, (50, 90)), (0.035, (50, 75))):
        for s in shafts:
            c += [[d, l, wire, s] for d, l in grid([3, 4, 5, 6, 7, 8], [20, 30, 40, 60])]
    return DL + ["wire_in", "working_length_cm"], c, FULL, "exact", \
        "Four configurations (0.018 in 50/90 cm, 0.035 in 50/75 cm), each a full 6 x 4 matrix (UK APR Medtech brochure). 150 cm 0.018 versions seen on the shelf are not in that brochure and not counted."


def _sequent():
    return DL, grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0], [10, 15, 20, 25, 30, 35, 40]), FULL, "exact", "3.25 mm is not offered."


def _nagomi():
    return DL, grid([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5], [9, 12, 15, 18, 21, 24, 28, 33, 38, 44, 50]), FULL, "exact", \
        "Availability varies by region."


def _accuforce():
    t = group([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0], [6, 8, 12, 15])
    for d in (2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0):
        t[d] = t[d] + [20]
    for d in (2.5, 3.0, 3.5):
        t[d] = t[d] + [25, 30]
    return DL, per_dia(t), RULES, "exact", ""


def _ryurei():
    t = {1.0: [5]}
    group([1.25, 1.5], [5, 10, 15, 20], t)
    group([2.0, 2.5, 3.0, 3.5], [10, 15, 20, 30, 40], t)
    group([2.25, 2.75, 3.25, 3.75, 4.0], [10, 15, 20], t)
    return DL, per_dia(t), RULES, "exact", ""


def _navicross():
    c = []
    for wire in (0.035, 0.018):
        for s in (65, 90, 135, 150):
            for tip in ("straight", "angled"):
                c.append([wire, s, tip])
    return ["wire_in", "shaft_cm", "tip"], c, "two wire classes x listed shaft lengths x tip", "lower_bound", \
        "The R2P 200 cm 0.035 in version is named but its tip options are not given, so it is not counted."


def _angiolite():
    t = group([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5], [9, 14, 16, 19, 24, 29, 34, 39])
    t[4.5] = [l for l in t[4.5] if l != 9]
    for d in (2.5, 2.75, 3.0, 3.5, 4.0):
        t[d] = t[d] + [44, 49]
    return DL, per_dia(t), RULES, "exact", "Per the 2019 IFU grid plus the CE 2021 44/49 mm lengths."


def _essential():
    t = {1.5: [10, 15, 20, 30]}
    group([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5], [10, 15, 20, 25, 30, 40], t)
    return DL, per_dia(t), RULES, "exact", ""


def _oceanus35():
    t = group([3, 4, 5, 6], [20, 40, 60, 80, 120, 150, 200])
    group([7], [20, 40, 60, 80, 120], t)
    group([8, 9], [20, 40, 60, 80], t)
    group([10, 12], [20, 40, 60], t)
    c = with_shaft(per_dia(t), [80, 140]) + [[d, 20, 200] for d in (6, 7, 8, 9, 10, 12)]
    return DLS, c, RULES, "exact", "80 and 140 cm shafts share one matrix; the 200 cm shaft is offered for 6-12 mm in 20 mm length only."


def _oceanus18():
    t1 = group([2.0, 2.5, 3.0, 3.5, 4.0], [20, 40, 60, 80, 120, 150, 200])
    t2 = group([5.0, 6.0], [20, 40, 60, 80, 120, 150, 200])
    group([7.0], [20, 40, 60, 80, 120], t2)
    group([8.0], [20, 40, 60, 80], t2)
    c = with_shaft(per_dia(t1), [100, 150]) + with_shaft(per_dia(t2), [100, 140])
    return DLS, c, RULES, "exact", "Shafts: 100/150 cm for 2.0-4.0 mm, 100/140 cm for 5.0-8.0 mm."


def _oceanus14():
    t = {1.25: [10, 15, 20], 1.5: [10, 15, 20, 40, 60, 80], 2.0: [10, 15, 20, 40, 60, 80, 120, 150, 200]}
    group([2.5, 3.0, 3.5], [40, 60, 80, 120, 150, 200], t)
    t[4.0] = [40, 60, 80, 120]
    return DLS, with_shaft(per_dia(t), [100, 150]), RULES, "exact", "Same matrix on 100 and 150 cm shafts."


def _single(note):
    return lambda: ([], [], "single catalogue entry", "exact", note)


EXPAND = {
    "bsci_agent_monorail_dcb": _agent, "bsci_nc_emerge_monorail": _emerge,
    "bsci_wolverine_cutting_balloon": _wolverine, "boston_scientific_mustang": _mustang,
    "boston_scientific_synergy_xd": _synergy_xd, "boston_scientific_v18_control_wire": _v18,
    "teleflex_guideliner": _guideliner, "unknown_teal_boston_style": _megatron,
    "biotronik_orsiro_mission": _orsiro, "biotronik_pk_papyrus": _papyrus,
    "orbusneich_combo_plus": _combo, "orbusneich_sapphire_3": _sapphire, "orbusneich_scoreflex": _scoreflex,
    "orbusneich_scoreflex_green_variant": _scoreflex_green, "medtronic_prevail_dcb": _prevail,
    "medtronic_resolute_onyx": _onyx, "medtronic_nc_euphora_edge": _euphora,
    "abbott_xience_skypoint": _xience, "abbott_armada_35": _armada, "apt_conqueror_nc_pro": _conqueror,
    "apt_medical_tresfort_35": _tresfort, "meril_nexgen_balloon": _nexgen,
    "medalliance_selution_slr_dcb": _selution, "nse_alpha_scoring_balloon": _nse,
    "trireme_tri_wedge": _triwedge, "bbraun_sequent_please_neo_dcb": _sequent,
    "terumo_ultimaster_nagomi_des": _nagomi, "terumo_accuforce_nc": _accuforce,
    "terumo_ryurei_balloon": _ryurei, "terumo_navicross": _navicross, "ivascular_angiolite_des": _angiolite,
    "ivascular_essential_pro_balloon": _essential, "ivascular_oceanus_35": _oceanus35,
    "ivascular_oceanus_18": _oceanus18, "ivascular_oceanus_14_pro": _oceanus14,
    "bsci_permanent_sled_bag_carton": _single("Single size / single REF (H749 39315010 0)."),
    "medtronic_export_advance_edge": _single("Single size: 6F, 140 cm (catalogue number ADVANCE)."),
}
# SKU count known only as an approximation from a source statement
APPROX = {"bard_conquest_pta": (50, "BD lists ~50 SKUs on the product-family page; the per-size matrix was not extracted.")}
# What blocks an exact count for rules/range-only families (shown as `incomplete_reason`)
WHY_RANGES = {
    "bsci_mach1_guiding_catheter": "Guide catheters: French size x curve x shaft; the per-curve table (catalogue pp. 22-29) was not extracted.",
    "terumo_heartrail_ii_guiding_catheter": "Guide catheters: French size x curve; not every curve exists in every French size and the table was not extracted.",
    "biotronik_passeo_35_xeo": "Matrix depends on shaft length (90/130/170 cm) with per-diameter length limits and pre-order sizes; ambiguous without the ordering table.",
    "biotronik_passeo_18": "Source says 'full 8 x 8 with few gaps' without listing the gaps.",
    "medtronic_pacific_plus": "Two diameter groups with different shaft sets; the 4-7 mm length table was only partly captured.",
    "medtronic_amphirion_deep": "Official ordering table not accessible; only ranges confirmed.",
    "elixir_desyne_x2": "Lengths not confirmed from a primary source.",
    "biosensors_chroma": "Ranges from search summaries, not a primary table (low confidence).",
    "bard_conquest_pta": "Complete per-size matrix not extracted (diameter 4-12 mm, lengths 20-100 mm, 50/75/120 cm shafts).",
    "biotronik_red_blue_unknown": "Product not identified.",
    "avantigard_d_logo": "Product not identified.",
    "unknown_blue_des": "Product not identified.",
}

# One-line corrections for the evidence field (research `inventory_corrections`, condensed).
CORR = {
    "abbott_armada_35": "confirmed OTW .035; shelf mixes 80 and 135 cm shafts",
    "abbott_xience_skypoint": "confirmed; all shelf sizes valid",
    "apt_medical_tresfort_35": "identity confirmed; 5/6/7 x 80 / 75 cm are valid catalogue entries",
    "avantigard_d_logo": "maker NOT identified (not Alvimedica Avantgarde); spelling may be misread, needs a close photo",
    "bard_conquest_pta": "confirmed Bard Conquest; carton height/length are very low confidence",
    "biosensors_chroma": "device is a bare-metal stent, not a DES",
    "biotronik_orsiro_mission": "confirmed current Orsiro Mission",
    "biotronik_passeo_18": "confirmed",
    "biotronik_passeo_35_xeo": "confirmed",
    "biotronik_pk_papyrus": "covered stent, not a DES",
    "biotronik_red_blue_unknown": "kept unidentified; the 'Pantera' guess is not supported",
    "boston_scientific_mustang": "confirmed; 9, 10 and 12 mm only up to 80 mm long",
    "boston_scientific_synergy_xd": "name correct; '4.0 .. 80' label reading is not a valid size (max 48 mm)",
    "boston_scientific_v18_control_wire": "peripheral guidewire; short vs long taper share the same artwork",
    "bsci_mach1_guiding_catheter": "confirmed; thickness not confirmed, guiding catheters ship coiled so the ~1 m spine is unverified",
    "bsci_nc_emerge_monorail": "confirmed; front face is blue/white while the spine is green/white",
    "bsci_permanent_sled_bag_carton": "contents are the sterile IVUS motordrive bag, not a stent sled; type is 'other'",
    "bsci_wolverine_cutting_balloon": "confirmed; 6 mm length and 3.25/3.75 mm diameters added to the matrix",
    "elixir_desyne_x2": "maker Elixir Medical confirmed (not OrbusNeich)",
    "ivascular_essential_pro_balloon": "MAJOR: paclitaxel drug-coated balloon, not a plain semi-compliant balloon; REF prefix is BCDPR14N150",
    "ivascular_oceanus_14_pro": "confirmed",
    "ivascular_oceanus_18": "confirmed",
    "ivascular_oceanus_35": "confirmed",
    "medalliance_selution_slr_dcb": "confirmed; newer cartons may read Cordis",
    "medtronic_amphirion_deep": "confirmed (Invatec/ev3 heritage)",
    "medtronic_export_advance_edge": "aspiration catheter, single 6F size",
    "medtronic_nc_euphora_edge": "confirmed",
    "medtronic_pacific_plus": "confirmed .018 in (not .014); 2.0 x 160 is not catalogued",
    "medtronic_prevail_dcb": "confirmed",
    "medtronic_resolute_onyx": "confirmed",
    "meril_nexgen_balloon": "device is a bare-metal STENT, not a balloon (family id is a misnomer)",
    "nse_alpha_scoring_balloon": "maker is Goodman (NSE = Novel Scientific Engineering), sold by Nipro / B. Braun",
    "orbusneich_combo_plus": "brand OrbusNeich confirmed",
    "orbusneich_sapphire_3": "OrbusNeich confirmed; semi-compliant, not NC",
    "orbusneich_scoreflex": "OrbusNeich confirmed",
    "teleflex_guideliner": "brand is Teleflex (ex Vascular Solutions); a guide extension catheter, not a guidewire",
    "terumo_heartrail_ii_guiding_catheter": "type correct; ~1 m spine length is unverified",
    "terumo_navicross": "support catheter, not a guidewire",
    "terumo_ryurei_balloon": "no correction",
    "trireme_tri_wedge": "maker is BrosMed (APR Medtech distributes in the UK), not Trireme; a scoring PTA balloon",
    "unknown_blue_des": "brand unknown, cannot be corrected without a closer photo",
    "unknown_teal_boston_style": "probably BSC SYNERGY MEGATRON, not a different brand",
}


# --------------------------------------------------------------------------------------------
def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def one_line(s, n=None):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if not n or len(s) <= n else s[: n - 1].rstrip() + "…"


def q(s):
    return json.dumps(s, ensure_ascii=False)


def fnum(v):
    return repr(float(v)) if float(v) != int(float(v)) else "%.1f" % float(v)


def load_research():
    out = {}
    for f in sorted(glob.glob(os.path.join(CAT, "research", "*.json"))):
        out.update(json.load(open(f))["families"])
    return out


def load_faces(path):
    """Per-face description strings from faces.yaml (written by the face-texture builder).
    Tolerant of a few layouts: {family: {face: str}}, {family: {faces|textures: {face: str|dict}}},
    optionally under a top-level 'families' key. Returns {} if the file is absent."""
    if not os.path.isfile(path):
        return {}
    raw = load_yaml(path) or {}
    fams = raw.get("families", raw)
    out = {}
    for fam, v in fams.items():
        if not isinstance(v, dict):
            continue
        faces = v.get("faces") or v.get("textures") or v
        d = {}
        for face in FACE_ORDER:
            e = faces.get(face)
            if isinstance(e, dict):
                e = e.get("description") or e.get("desc") or e.get("source") or e.get("note")
            if e:
                d[face] = one_line(e)
        if d:
            out[fam] = d
    return out


def example_ref(research):
    ex = research.get("example_refs") or []
    if not ex or str(research.get("ref_scheme", "")).strip().lower() in ("n/a", ""):
        return "", ""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", ex[0])
    ref, note = (m.group(1), m.group(2)) if m else (ex[0], "")
    return ref.strip(), note.strip()


def physics_text(boxes_yaml_path):
    """Verbatim physics sections (with their comments) from boxes.yaml, from the comment block
    above `physics_material:` to the end of the file."""
    lines = open(boxes_yaml_path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("physics_material:"))
    while i > 0 and lines[i - 1].startswith("#"):
        i -= 1
    return "\n".join(lines[i:]) + "\n"


def build_boxes_yaml(args, research):
    dims = load_yaml(os.path.join(CAT, "dims.yaml"))["families"]
    spines = load_yaml(os.path.join(CAT, "spines.yaml"))["families"]
    faces = load_faces(os.path.join(CAT, "faces.yaml"))
    order_type = ["coronary_des", "coronary_bms", "covered_stent", "coronary_dcb", "coronary_balloon_sc",
                  "coronary_balloon_nc", "scoring_cutting_balloon", "pta_balloon", "guiding_catheter",
                  "guide_extension_catheter", "support_catheter", "guidewire", "other"]
    fams = sorted(dims, key=lambda f: (order_type.index(META[f][3]) if META[f][3] in order_type else 99, f))
    out = ["# Medical device cartons: the catalogue families (generated by tools/build_catalog.py; do not edit).",
           "#",
           "# Same schema as boxes.yaml, so build_usd.py / verify_usd.py take it unchanged:",
           "#   build_usd.py --boxes-yaml catalog/catalog_boxes.yaml --textures-root catalog/textures --out-dir catalog/usd \\",
           "#                --scene-grid-cols 8",
           "# Box frame, faces and texture orientation: see the header of boxes.yaml.",
           "#",
           "# NOT repeated here (already modelled in boxes.yaml, with caliper-measured dimensions):",
           "#   agent_monorail, angiolite, accuforce, ultimaster_nagomi, conqueror_nc_pro",
           "#",
           "# Dimensions and masses are ESTIMATES from catalog/dims.yaml (shelf photos scaled against the five",
           "# measured cartons; mass = board + typical contents). Size ranges per family: catalog/sizes.yaml.",
           "# `photo_corners_px` are the spine corners in the exemplar photo (used by the viewer to crop it).",
           "",
           "boxes:"]
    for fam in fams:
        dm, sp, rs = dims[fam], spines[fam], research[fam]
        label, brand, product, dtype = META[fam]
        w, h = dm["dims_mm"], dm["mass_estimate_g"]
        sizes_seen = (load_draft()[fam].get("sizes_seen") or [])
        size = one_line(sizes_seen[0]) if sizes_seen else ""
        ref, ref_note = example_ref(rs)
        m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+)", ref_note)
        ref_example = ""
        if "pattern reference" in ref_note:  # a different product's code, shown only as a scheme example
            ref = ""
        if ref and m:  # the example REF decodes to a size: use that size so size and ref agree
            size = f"{m.group(1)} x {m.group(2)} mm"
        elif ref:
            ref_example, ref = ref, ""
        photo = f"{PHOTO_DIR}/{sp['photo']}"
        fd = faces.get(fam, {})
        tex = {}
        for face in FACE_ORDER:
            if face in fd:
                tex[face] = fd[face]
            elif face == "top":
                tex[face] = f"spine photo ({photo})"
            else:
                tex[face] = "plain colour"
        ev = (f"Research confidence: {one_line(rs['confidence'])}. Dims: {dm['dims_confidence']} "
              f"({one_line(dm['dims_basis'])}). Corrections: {CORR[fam]}.")
        out += [f"  {fam}:",
                f"    evidence: {q(ev)}",
                f"    label: {q(label)}",
                f"    brand: {q(brand)}",
                f"    product: {q(product)}",
                f"    device_type: {dtype}",
                f"    size: {q(size)}"]
        if ref:
            out.append(f"    ref: {q(ref)}")
        if ref_example:
            out.append(f"    ref_example: {q(ref_example)}")
        out += [f"    dims_mm: {{height: {fnum(w['height'])}, length: {fnum(w['length'])}, thickness: {fnum(w['thickness'])}}}",
                f"    dims_confidence: {q(dm['dims_confidence'])}",
                f"    mass_kg: {dm['mass_kg']}",
                "    mass_source: estimate",
                f"    mass_estimate_g: {{carton: {h['carton']}, contents: {h['contents']}, low: {h['low']}, high: {h['high']}}}",
                f"    photo: {photo}",
                *(["    photo_corners_px: " + json.dumps([[round(x, 1) for x in c] for c in sp["corners_px"]])]
                  if sp.get("corners_px") else []),
                "    textures:"]
        out += [f"      {face}: {q(tex[face])}" for face in FACE_ORDER]
        out.append("")
    out.append(physics_text(os.path.join(ROOT, "boxes.yaml")).rstrip("\n"))
    text = "\n".join(out) + "\n"
    yaml.safe_load(text)  # must parse
    with open(args.boxes_out, "w") as f:
        f.write(text)
    return fams


_DRAFT = None


def load_draft():
    global _DRAFT
    if _DRAFT is None:
        _DRAFT = load_yaml(os.path.join(CAT, "families_draft.yaml"))["families"]
    return _DRAFT


# --------------------------------------------------------------------------------------------
def size_entry(fid, rid, research, in_catalog, label, brand, dtype):
    rs = research[rid]
    s = rs["sizes"]
    draft = load_draft()
    seen = [one_line(x) for x in (draft.get(rid, {}).get("sizes_seen") or [])]
    e = {"research_id": rid, "in_catalog": in_catalog, "label": label, "brand": brand,
         "maker": rs["maker"], "device_type": dtype, "product": rs["product"],
         "confidence": rs["confidence"]}
    e["diameters_mm"] = s["diameters_mm"]
    e["lengths_mm"] = s["lengths_mm"]
    e["shaft_lengths_cm"] = s["shaft_lengths_cm"]
    av = {"matrix_notes": one_line(s["matrix_notes"])}
    if rid in EXPAND:
        fields, combos, basis, kind, note = EXPAND[rid]()
        av.update({"kind": "explicit" if combos else "single", "basis": basis,
                   "combination_fields": fields, "combinations": combos})
        e["sku_count"] = len(combos) if combos else 1
        e["sku_count_kind"] = kind
        if note:
            e["sku_count_note"] = note
        if kind != "exact":
            e["incomplete"] = True
            e["incomplete_reason"] = note
    else:
        av["kind"] = "ranges"
        av["combinations"] = None
        if rid in APPROX:
            e["sku_count"], e["sku_count_note"] = APPROX[rid]
            e["sku_count_kind"] = "approximate"
        else:
            e["sku_count"] = None
            e["sku_count_kind"] = "unknown"
        e["incomplete"] = True
        e["incomplete_reason"] = WHY_RANGES.get(rid, "Rules or ranges only.")
        av["note"] = "Ranges only: the source gives rules/ranges that cannot be expanded without inventing combinations."
    e["availability"] = av
    if s.get("other"):
        e["other"] = one_line(s["other"])
    e["ref_scheme"] = one_line(rs["ref_scheme"])
    e["example_refs"] = rs["example_refs"]
    e["sources"] = s["sources"]
    e["sizes_seen_on_shelf"] = seen
    return e


def build_sizes(args, research, fams):
    entries = {}
    for fam in fams:
        label, brand, _p, dtype = META[fam]
        entries[fam] = size_entry(fam, fam, research, True, label, brand, dtype)
    for fam, rid in MAIN5.items():
        label, brand, dtype = MAIN5_META[fam]
        e = size_entry(fam, rid, research, True, label, brand, dtype)
        e["note"] = "modelled in boxes.yaml (caliper-measured dimensions)"
        entries[fam] = e
    for rid in EXTRA_RESEARCH:
        label, brand, dtype = EXTRA_META[rid]
        e = size_entry(rid, rid, research, False, label, brand, dtype)
        e["note"] = "research-only family: not on the shelf-photo catalogue, no carton asset"
        entries[rid] = e

    class D(yaml.SafeDumper):
        pass

    def rep_list(dumper, data):
        flow = all(not isinstance(x, (list, dict)) for x in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    D.add_representer(list, rep_list)
    D.add_representer(float, lambda d, v: d.represent_scalar("tag:yaml.org,2002:float", repr(v)))
    head = ("# Normalised size catalogue per carton family (generated by tools/build_catalog.py from\n"
            "# catalog/research/*.json; do not edit).\n"
            "#\n"
            "# availability.kind: explicit  = full list of combinations (fields in combination_fields)\n"
            "#                    single    = one catalogue entry\n"
            "#                    ranges    = the source gives rules/ranges only; diameters_mm / lengths_mm /\n"
            "#                                shaft_lengths_cm are ranges and matrix_notes quotes the rule. No\n"
            "#                                combinations are invented.\n"
            "# sku_count_kind: exact | lower_bound (known combinations only) | approximate | unknown\n"
            "# Families flagged incomplete: true have missing size data (incomplete_reason says why).\n"
            "# in_catalog: false = research-only family without a carton asset.\n")
    text = head + yaml.dump({"families": entries}, Dumper=D, sort_keys=False, width=100000,
                            allow_unicode=True, default_flow_style=False)
    assert yaml.safe_load(text) == {"families": json.loads(json.dumps(entries))}, "sizes.yaml round-trip mismatch"
    with open(args.sizes_out, "w") as f:
        f.write(text)
    return entries


# --------------------------------------------------------------------------------------------
def conf_short(c):
    m = re.match(r"(medium-high|low-medium|high|medium|low|none)", str(c))
    return m.group(1) if m else one_line(c, 20)


def md_matrix(e):
    av = e["availability"]
    combos = av.get("combinations")
    fields = av.get("combination_fields") or []
    if not combos:
        return None
    if "diameter_mm" in fields and "length_mm" in fields:
        di, li = fields.index("diameter_mm"), fields.index("length_mm")
        cnt = {}
        for c in combos:
            cnt[(c[di], c[li])] = cnt.get((c[di], c[li]), 0) + 1
        ds = sorted({k[0] for k in cnt})
        ls = sorted({k[1] for k in cnt})
        multi = max(cnt.values()) > 1
        rows = ["| diameter \\ length (mm) | " + " | ".join(str(l) for l in ls) + " | n |",
                "|---|" + "---|" * (len(ls) + 1)]
        for d in ds:
            cells = [(str(cnt[(d, l)]) if multi else "✓") if (d, l) in cnt else "" for l in ls]
            rows.append(f"| {d:g} | " + " | ".join(cells) + f" | {sum(cnt.get((d, l), 0) for l in ls)} |")
        extra = [x for x in fields if x not in ("diameter_mm", "length_mm")]
        cap = ("Cells show the number of variants per (diameter, length)" + f" ({', '.join(extra)})" if multi
               else "✓ = offered")
        return "\n".join(rows) + f"\n\n_{cap}._"
    rows = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    rows += ["| " + " | ".join(str(x) for x in c) + " |" for c in combos]
    return "\n".join(rows)


def build_md(args, entries):
    L = ["# Size catalogue (generated by `tools/build_catalog.py` from `catalog/research/*.json`)", "",
         "SKU = one orderable combination (diameter x length x shaft/variant as listed per family). Counts are only",
         "given where the source supports an explicit list. `exact` = the full matrix/rules were available;",
         "`>=` = lower bound (known combinations only); `~` = the source's own approximate count; `?` = ranges only.",
         "Machine-readable version: `sizes.yaml`.", ""]

    def cnt(e):
        n, k = e["sku_count"], e["sku_count_kind"]
        return {"exact": str(n), "lower_bound": f">={n}", "approximate": f"~{n}", "unknown": "?"}[k]

    def total(sel):
        ex = [e for e in sel if e["sku_count_kind"] == "exact"]
        lb = [e for e in sel if e["sku_count_kind"] in ("lower_bound", "approximate")]
        unk = [e for e in sel if e["sku_count_kind"] == "unknown"]
        return sum(e["sku_count"] for e in ex), sum(e["sku_count"] for e in lb), len(ex), len(lb), len(unk)

    cat = {k: e for k, e in entries.items() if e["in_catalog"]}
    ext = {k: e for k, e in entries.items() if not e["in_catalog"]}
    ex_n, lb_n, nex, nlb, nunk = total(list(cat.values()))
    L += ["## Summary", "",
          f"{len(cat)} catalogue families ({len(MAIN5)} modelled in `boxes.yaml` + {len(cat) - len(MAIN5)} in "
          f"`catalog_boxes.yaml`): **{ex_n + lb_n} SKUs known** = {ex_n} in {nex} families with an exact count + "
          f"{lb_n} in {nlb} families with a lower-bound/approximate count; {nunk} families have no count "
          f"(ranges only or unidentified).", "",
          "| family | brand | device type | confidence | SKUs | basis | incomplete |", "|---|---|---|---|---:|---|---|"]
    for k, e in sorted(cat.items(), key=lambda kv: (kv[1]["device_type"], kv[0])):
        L.append(f"| {e['label']} (`{k}`) | {e['brand']} | {e['device_type']} | {conf_short(e['confidence'])} | "
                 f"{cnt(e)} | {e['availability'].get('basis', 'ranges only')} | {'yes' if e.get('incomplete') else ''} |")
    L.append(f"| **total (catalogue families)** | | | | **{ex_n + lb_n}** | {ex_n} exact + {lb_n} partial | {nlb + nunk} families |")
    if ext:
        ex2, lb2, *_ = total(list(ext.values()))
        L += ["", "Research-only families (no carton asset): " + ", ".join(f"{e['label']} ({cnt(e)})" for e in ext.values())
              + f". Including them the overall total is {ex_n + lb_n + ex2 + lb2}."]
    L += ["", "## Families", ""]
    for k, e in list(cat.items()) + list(ext.items()):
        L += [f"### {e['label']} (`{k}`)", "",
              f"{e['product']}. Maker: {e['maker']}. Device type: {e['device_type']}. Research confidence: {e['confidence']}.", "",
              f"**SKUs: {cnt(e)}** ({e['sku_count_kind'].replace('_', ' ')})" + (f". {e['sku_count_note']}" if e.get("sku_count_note") else ""), ""]
        if e.get("note"):
            L += [f"_{e['note']}_", ""]
        if e["diameters_mm"]:
            L.append(f"- Diameters (mm): {', '.join(f'{d:g}' for d in e['diameters_mm'])}")
        if e["lengths_mm"]:
            L.append(f"- Lengths (mm): {', '.join(str(x) for x in e['lengths_mm'])}")
        if e["shaft_lengths_cm"]:
            L.append(f"- Shaft / working length (cm): {', '.join(str(x) for x in e['shaft_lengths_cm'])}")
        m = md_matrix(e)
        L.append("")
        if m:
            L += [m, ""]
        else:
            L += [f"Ranges only (no combinations invented). Source rule: {e['availability']['matrix_notes']}", ""]
        if e.get("incomplete"):
            L += [f"**Incomplete:** {e['incomplete_reason']}", ""]
        L += [f"- REF scheme: {e['ref_scheme']}", f"- Example REFs: {'; '.join(e['example_refs']) or 'none'}"]
        if e["sizes_seen_on_shelf"]:
            L.append(f"- Seen on the shelf: {'; '.join(e['sizes_seen_on_shelf'])}")
        if e["sources"]:
            L.append("- Sources: " + ", ".join(e["sources"]))
        L.append("")
    with open(args.md_out, "w") as f:
        f.write("\n".join(L))
    return ex_n, lb_n, nlb + nunk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes-out", default=os.path.join(CAT, "catalog_boxes.yaml"))
    ap.add_argument("--sizes-out", default=os.path.join(CAT, "sizes.yaml"))
    ap.add_argument("--md-out", default=os.path.join(CAT, "SIZE_CATALOG.md"))
    args = ap.parse_args()
    research = load_research()
    fams = build_boxes_yaml(args, research)
    entries = build_sizes(args, research, fams)
    ex_n, lb_n, ninc = build_md(args, entries)
    print(f"wrote {args.boxes_out} ({len(fams)} families)")
    print(f"wrote {args.sizes_out} ({len(entries)} families)")
    print(f"wrote {args.md_out}: {ex_n} exact + {lb_n} partial SKUs; {ninc} catalogue families incomplete")
    for k, e in entries.items():
        print(f"  {k:42s} {e['sku_count_kind']:12s} {e['sku_count']}")


if __name__ == "__main__":
    sys.exit(main())
