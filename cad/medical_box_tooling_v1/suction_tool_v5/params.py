"""Left-arm suction tool v5 ("one body, one camera"): tool-specific parameters (mm, kg, N, link6 frame).

Shared inputs are imported read-only from ../common/params.py (never edited here).
v5 = v4 concept slimmed further: ONE printed PA12-CF body (flange hub + C-section stem + pad bar + collector bay; the camera bolts straight onto the stem +y wall),
no scanner / decoder / aluminium bar, ONE camera (Orbbec Gemini 305), pad tubes on elbows behind the pad bodies.
"""
import math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location('common_params', os.path.join(HERE, '..', 'common', 'params.py'))
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)      # shared, read-only  (../common/params.py)

SCRATCH = '/media/farhan/My Passport/igus_nvidia/scratch_suction_tool'      # cached Gemini hull (shared with v3), temp files
GEMINI_STEP = C.GEMINI305_CAD_DIR + '/3D/3D__00_mb2502_asm(G305)_ V1.1.stp'

# ------------------------------------------------------------------ hand-off contract (unchanged from v3/v4)
PAD_PITCH = 60.0
PAD_X = [-PAD_PITCH, 0.0, PAD_PITCH]
LIP_Z = 159.0                         # uncompressed pad lip plane; compressed TCP 154
COMPRESSED_TCP_Z = LIP_Z - 5.0
FLANGE_Z = C.FLANGE_Z                 # 52.7
KO_REAR = 40.0                        # nothing but pads / pad bar at z > LIP_Z - 40 = 119 (and the bar within |y| <= 12)
KEEPOUT_Z = LIP_Z - KO_REAR
KO_HALF_Y = 12.0
NEIGHBOUR_PROTRUSION = C.NEIGHBOUR_PROTRUSION     # 15
PAD = dict(C.PAD)
PAD.update(code='ZP2-T4030WNK10-B5')  # v5: M5 female inlet variant + KQ2L04-M5 swivel elbow (v3/v4 used the -04 one-touch variant)
BAR_FRONT = LIP_Z - PAD['lip_to_nut_face']        # 133.5 bar front face (nut seat)
INLET_Z = LIP_Z - 61.5                # 97.5 buffer rear end / inlet face (fixed to the clamped body)
NUT_T = PAD['nut_t']; NUT_AF = PAD['nut_af']
HEX_POCKET_AF = 14.25
HEX_POCKET_DEPTH = 3.2
THREAD_R = 4.95
WASHER = dict(od=20.0, id=10.5, t=1.0)

# ------------------------------------------------------------------ materials / mass model
PRINT_MATERIAL = 'PA12-CF (carbon-fibre nylon), FDM/SLS'
PRINT_DENSITY = 1.20e-6               # kg/mm3 solid (ASSUMED 1.2 g/cm3)
PRINT_FILL_EQ = 0.90                  # effective solid fraction (>= 3 mm walls are essentially all perimeters) - ASSUMED
E_PRINT = 3500.0                      # MPa, conservative (lead's value)
SIGMA_ALLOW = 25.0                    # MPa, allowable stress (lead's value)
MASS_PURCHASED = dict(pad=0.012, elbow=0.003, nut=0.0025, washer=0.001, tube=0.001, trunk=0.001, camera=C.CAMERA['mass'],
                      collector=0.012)       # pad body 12 g + elbow 3 g ASSUMED (as v3: 15 g with the fitting); collector as v3 (ASSUMED)
MASS_FASTENERS = dict(flange_csk_m3x10=6 * 0.0006, camera_m3x10=2 * 0.0007, note='ASSUMED: steel M3 CSK 0.6 g, M3 pan 0.7 g; zip ties / plugs not counted')

# ------------------------------------------------------------------ hub (flange adapter, printed, one piece with the stem)
HUB_D = 56.0
HUB_T = 5.0                           # 5 mm: M3 CSK head 1.7 + 3.3 of plastic; screw M3x10 -> 5 mm engagement in the flange (flange thread ~4 mm)
HUB_Z0, HUB_Z1 = FLANGE_Z, FLANGE_Z + HUB_T      # 52.7 .. 57.7
HUB_BORE = 14.0                       # centre bore (r 7 < the +y wall inner face 8.6, so the wall stands fully on the hub)

# ------------------------------------------------------------------ stem (C-section: +y wall and two +-x walls, open toward -y)
STEM_HX = 15.5                        # outside half width (x), 31 wide
STEM_Y0, STEM_Y1 = -11.6, 11.6        # outside y range (23.2 deep; <= the 24 mm slab limit)
WALL = 3.0
CAV_HX = STEM_HX - WALL               # 12.5
CAV_Y1 = STEM_Y1 - WALL               # 8.6  (+y wall inside face)

# ------------------------------------------------------------------ pad bar: front skin + rear bosses + edge ribs (all inside the keep-out slab: z in [119, 133.5], |y| <= 11.6)
BAR_LEN = 150.0
BAR_HY = 11.6
SKIN_T = 3.0
SKIN_Z1 = BAR_FRONT                   # 133.5
SKIN_Z0 = SKIN_Z1 - SKIN_T            # 130.5
BOSS_Z0 = 127.0                       # rear face of the pad bosses (= washer seat); pocket floor 130.3 -> 3.3 mm
BOSS_D = 22.0
RIB_Z0 = KEEPOUT_Z                    # 119: ribs hang from the skin down to the keep-out line
RIB_Y = (CAV_Y1, STEM_Y1)             # ribs along the long edges, 3 mm thick (y = +-[8.6, 11.6])
RIB_X = (STEM_HX, 54.0)               # from the stem into the outer pad boss (the rib end merges with the boss circle: no skin-only section near the pad)
NUT_FRONT_Z0, NUT_FRONT_Z1 = BAR_FRONT - NUT_T, BAR_FRONT
WASHER_Z0, WASHER_Z1 = BOSS_Z0 - WASHER['t'], BOSS_Z0
NUT_REAR_Z0, NUT_REAR_Z1 = WASHER_Z0 - NUT_T, WASHER_Z0
PAD_HOLE_D = PAD['bracket_hole']      # 10.5

# ------------------------------------------------------------------ elbows, tubes, collector bay (all at the elbow tube axis height)
ELBOW_H = C.PAD_FITTING['height_above_inlet']      # 14 total height below the inlet face (KQ2L04-M5, ASSUMED)
ELBOW_D = 9.0
ELBOW_AXIS_Z = INLET_Z - 9.5          # tube axis 9.5 below the inlet face (ASSUMED) = 88.0
ELBOW_BARREL = 7.0                    # outlet face 7 mm beyond the pad axis (ASSUMED)
TUBE_OD = C.PAD_FITTING['tube_od']; TRUNK_OD = 6.0
TUBE_BEND_R = 10.0                    # min bend radius phi4 PU (ASSUMED 10, lead's value)
TRUNK_BEND_R = 15.0                   # phi6 x 4 PU (ASSUMED 15; not used inside the CAD: the trunk leaves straight down)
COLL_YC = -44.5                       # collector centre y: the trunk hangs straight down at r = 44.5, outside the link6 housing (r = 40, from the ReBeL mesh) and the hub rim (r = 28)
COLL_ZC = ELBOW_AXIS_Z                # port axes at the elbow tube height
COLLECTOR = dict(code='push-in branch/manifold block 1 x phi6 in + 3 x phi4 out - to confirm (proxy, as v3)',
                 body=(24.0, 10.0, 16.0),             # x, y, z
                 boss4=(9.0, 5.0), boss6=(11.0, 4.0),  # port boss (dia, length)
                 mass=0.012, internal_cm3=1.0)
SHELF_Z0, SHELF_Z1 = COLL_ZC - COLLECTOR['body'][2] / 2 - WALL, COLL_ZC - COLLECTOR['body'][2] / 2     # 77 .. 80
SHELF_Y0 = COLL_YC - 8.2            # bay end: 3.2 mm of cheek beyond the collector's -y face (y = COLL_YC - 5) and the slot
LEDGE_W = 6.0                         # ledge width (x) under the collector body edges; the middle 13 mm stays open for the trunk boss
CHEEK_Z1 = COLL_ZC + COLLECTOR['body'][2] / 2 + 2.0                     # 98
CHEEK_Y0, CHEEK_Y1 = COLL_YC - 8.2, -27.0
HUB_GUSSET_H = 16.4                   # height of the lower -y wall (= largest gusset: rim 28 - wall 11.6)
TRUNK_END_Z = 62.0                    # trunk stub end in the CAD (continues down the arm)

# ------------------------------------------------------------------ camera (Gemini 305, +Y side, bottom face screwed to the stem +y wall)
CAM = C.CAMERA
CAM_HALF = 21.0
CAM_DEPTH = 23.0
CAM_TILT_DEG = 0.0                    # optical axis along +z: the camera bottom face lies flat on the stem +y wall (no seat, no wedge); a tilt would need a wedge that collides with the flange-screw access
CAM_TOP_Z = KEEPOUT_Z - 1.0           # highest point of the camera body (front face, +u edge)
CAM_GAP = 0.05                        # gap between the camera bottom face and the stem +y wall outer face
CABLE_LUG = dict(x=(STEM_HX - 0.1, STEM_HX + 7.9), y=(3.4, STEM_Y1), z=(85.0, 95.0), slot_x=(18.5, 20.1), slot_z=(88.2, 91.8))     # USB-C cable tie lug on the +x wall (below the plug)
APPROACH_GAP = 50.0                   # approach pose: pad lip plane 50 mm in front of the spine (spine at link6 z = 209)
CAM_AIM = (0.0, 0.0, LIP_Z + APPROACH_GAP)
SCAN_STANDOFFS = (30.0, 33.0, 36.0, 45.0, 50.0)   # lip -> spine distances used for the barcode table (the tool can be moved to any of them)

def cam_pose():
    a = math.radians(CAM_TILT_DEG); s, c = math.sin(a), math.cos(a)
    zc = CAM_TOP_Z - CAM_HALF * s                       # front-face centre z (highest point = front face, +u edge)
    yc = STEM_Y1 + CAM_GAP + CAM_HALF * c               # bottom face on the stem wall (CAM_GAP off it)
    d = np.array([0, -s, c]); u = np.array([0, c, s]); x = np.array([1.0, 0, 0])
    T = np.eye(4); T[:3, 0] = x; T[:3, 1] = u; T[:3, 2] = d; T[:3, 3] = [0, yc, zc]
    return T                           # cam frame: x = tool X (USB-C on +x), y = u ("up", away from the bottom mount face), z = optical axis

T_CAM = cam_pose()
CAM_SCREW_X = 10.0                    # 2 x M3 at +-10 mm in the bottom face (datasheet), 12 mm behind the front face
CAM_SCREW_T = -12.0
CAM_USB = dict(face='+x', t=(-22.6, -12.7), u=(-7.6, 7.6))   # from the vendor STEP (recess vertices on the +x face), datasheet right-hand view

# flange screws (shared pattern)
FLANGE_HOLES = [(C.PCD43_R * math.cos(math.radians(C.PCD43_A0 + 60 * k)), C.PCD43_R * math.sin(math.radians(C.PCD43_A0 + 60 * k))) for k in range(6)]

# ------------------------------------------------------------------ strength inputs
SPRING_N = C.PAD['spring_n'][1]       # 3 N at full compression (catalog)
ARM_PUSH_N = 20.0                     # lead's value
PULL_N = 80.0e3 * 1.16e-4             # 80 kPa x 1.16 cm2 = 9.28 N per pad
