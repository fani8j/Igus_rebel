"""suction_tool v4 CONCEPT ("pad blade + one sensor pod") - parameters (mm, kg, link6 frame, z out of the flange).
Shared inputs come read-only from ../common/params.py (C) and the v3 numbers in ../suction_tool/params.py (P3)."""
import math
import numpy as np
from bridge import P3, C

# ------------------------------------------------------------------ kept from v3 (hand-off contract)
LIP_Z = P3.LIP_Z                      # 159.0 uncompressed pad lip plane (compressed TCP 154)
PAD_X = list(P3.PAD_X)                # -60 / 0 / +60
PAD = P3.PAD
FLANGE_Z = C.FLANGE_Z                 # 52.7
KO_REAR = P3.KO_REAR                  # 40: nothing but pads / pad bar at z > LIP_Z - 40 = 119
KEEPOUT_Z = LIP_Z - KO_REAR           # 119.0
FIT_H = P3.FIT_H                      # 14.7 one-touch fitting height above the buffer top
FIT_Z0 = LIP_Z - 61.5                 # buffer top (inlet plane) 97.5
FIT_Z1 = FIT_Z0 - FIT_H               # fitting end / tube start 82.8
TUBE_OD = P3.TUBE_OD; TRUNK_OD = P3.TRUNK_OD
TUBE_BEND_R = P3.TUBE_BEND_R          # 12
TUBE_STRAIGHT = P3.TUBE_STRAIGHT      # 8 straight out of the fitting
NUT_AF = P3.NUT_AF; NUT_T = P3.NUT_T
MASS_PURCHASED = dict(P3.MASS_PURCHASED)
PETG_DENSITY = P3.PETG_DENSITY; PRINT_FILL_EQ = P3.PRINT_FILL_EQ
AL_DENSITY = 2.70e-6                  # kg/mm3, 6061-T6
PA12CF_DENSITY = 1.20e-6              # kg/mm3 (printed PA12-CF, ~1.2 g/cm3 solid; infill factor below)
PA12CF_FILL = 0.80                    # effective solid fraction for a small stiff printed bar (more walls, 40 % infill) - ASSUMED
E_AL, E_PA12CF = 69000.0, 6000.0      # MPa (PA12-CF printed XY, first order)

# ------------------------------------------------------------------ flange hub (printed, one part with the stem)
HUB_D = 56.0
HUB_T = 8.0
HUB_Z0, HUB_Z1 = FLANGE_Z, FLANGE_Z + HUB_T      # 52.7 .. 60.7

# ------------------------------------------------------------------ stem (hollow column hub -> pad bar)
STEM_X, STEM_Y = 31.0, 22.0           # outside (brief: about 30 x 22)
STEM_WALL = 3.0
STEM_R = 5.0                          # outer vertical-edge radius (look)
CAV_X, CAV_Y = STEM_X - 2 * STEM_WALL, STEM_Y - 2 * STEM_WALL     # 25 x 16
CAV_R = 0.0
CAV_FLOOR_Z = 57.6                    # cavity floor (collector sits 0.2 above it); floor thickness under the pocket 4.9 mm
# collar under the bar: carries 2 M3 heat-set inserts, the bar is screwed to it from the front
COLLAR_X = 25.7                       # half width (insert at +-20.6: 3 mm to the stem wall, 3 mm to the edge)
COLLAR_Z0, COLLAR_Z1 = 111.5, LIP_Z - 25.5 - 15.0    # (collar top = bar rear face)
BAR_SCREW_X = 20.6
INSERT_DEPTH = 5.5

# ------------------------------------------------------------------ pad bar ("blade"): 6061 flat bar 20 x 15, 150 long
BAR_LEN, BAR_W, BAR_T = 150.0, 20.0, 15.0
BAR_Z1 = LIP_Z - P3.BRACKET_FRONT                 # 133.5 front face (= lip - 25.5, unchanged)
BAR_Z0 = BAR_Z1 - BAR_T                           # 118.5 rear face (brief: 119 with 14.5 mm; stock 15 mm used, 0.5 mm further back)
HEX_POCKET_AF = P3.HEX_POCKET_AF                  # 14.25
HEX_POCKET_DEPTH = 3.2
NUT_REAR_Z0, NUT_REAR_Z1 = BAR_Z0 - NUT_T, BAR_Z0 # 115.5 .. 118.5 (inside the stem)
NUT_FRONT_Z0, NUT_FRONT_Z1 = BAR_Z1 - NUT_T, BAR_Z1

# ------------------------------------------------------------------ collector (same proxy as v3: 24 x 10 x 16 block, 12 g) - re-posed, ports re-assigned
COLL = P3.COLLECTOR
COLL_BODY = (24.0, 15.6, 10.0)        # x, y, z : the v3 24 x 10 x 16 block turned so that its 16 mm side is across the stem (y)
COLL_BOSS4 = (9.0, 5.0)               # phi4 port boss (dia, length): 2 x on +-x faces, 1 x on the +z face
COLL_BOSS6 = (11.0, 4.0)              # phi6 trunk boss on the -y face
COLL_ZC = FIT_Z1 - TUBE_STRAIGHT - TUBE_BEND_R    # 62.8: port axis height = outer tube run height
TRUNK_END_Y = -30.0                   # trunk stub end in the CAD (continues off-tool)

# ------------------------------------------------------------------ sensor pod (+y side of the stem, one printed housing)
POD_WALL = 3.0
POD_Y0 = STEM_Y / 2 + 0.1             # pod contact plane on the stem +y face
# camera Gemini 305 (42 x 42 x 23): optical axis tilted about +x toward -y, through CAM_AIM
CAM_TILT_DEG = 20.0
CAM_AIM = (0.0, 0.0, LIP_Z + 40.0)
CAM_HALF = 21.0
CAM_DEPTH = 23.0
CAM_CLR = 0.25                        # pocket clearance
CAM_POCKET_HALF = CAM_HALF + CAM_CLR  # 21.25
CAM_BLOCK_HALF = CAM_POCKET_HALF + POD_WALL     # 24.25
CAM_BACK_T = -(CAM_DEPTH + 0.4)       # pocket back plane (camera frame t)
CAM_BACK_SCREW_X = 10.0               # 2 x M3 in the camera back face at x = +-10, u = 0 (STEP faces; 4.8 deep)
CAM_USB_T = -14.8                     # USB-C receptacle on the +x face, t = -14.8, u = 0 (STEP)

def cam_pose():
    a = math.radians(CAM_TILT_DEG); s, c = math.sin(a), math.cos(a)
    Fy = POD_Y0 + CAM_BLOCK_HALF * c                    # front-lower outer corner of the pod touches the stem plane
    Fz = CAM_AIM[2] - Fy / math.tan(a)                  # axis through the aim point
    T = np.eye(4)
    T[:3, 0] = [1, 0, 0]; T[:3, 1] = [0, c, s]; T[:3, 2] = [0, -s, c]; T[:3, 3] = [0, Fy, Fz]
    return T                                            # x = tool X, y = u ("up" of the camera), z = optical axis

# scanner Zebra SE4710 beside the camera (-x), aimed at the spine centre (0, 0) from the scan standoff
SCN = C.SCANNER
ENGINE = dict(P3.ENGINE)
SCAN_XYZ = (-40.0, 47.0, 110.0)
SCAN_STANDOFF_NOM = 130.0             # scan-pose distance window -> spine along the axis (115..150 allowed)
SCAN_FOV_DEG = SCN['fov_deg']
ENG_CLR = 0.3
ENG_POCKET_L = ENGINE['long'] / 2 + ENG_CLR
ENG_POCKET_W = ENGINE['thick'] / 2 + 0.25
ENG_BACK_A = -(ENGINE['depth'] + 0.5)
ENG_BLOCK_L = ENG_POCKET_L + POD_WALL
ENG_BLOCK_W = ENG_POCKET_W + POD_WALL
DECODER = P3.DECODER                  # size (26.91, 39.04, 6.37), flex reach 89.4

def scan_pose(standoff=SCAN_STANDOFF_NOM, xyz=SCAN_XYZ):
    W = np.array(xyz, float)
    lat2 = W[0] ** 2 + W[1] ** 2
    zT = W[2] + math.sqrt(standoff ** 2 - lat2)         # spine plane height (tool z) where the axis meets the spine centre (0, 0)
    T = np.array([0.0, 0.0, zT])
    a = (T - W) / np.linalg.norm(T - W)
    l = np.array([1.0, 0, 0]) - a[0] * a; l /= np.linalg.norm(l)
    w = np.cross(a, l)
    M = np.eye(4); M[:3, 0] = l; M[:3, 1] = w; M[:3, 2] = a; M[:3, 3] = W
    return M, zT                                         # z = scan axis out of the window, x = engine long axis, y = w (across the thickness)

T_CAM = cam_pose()
T_SCAN, SCAN_SPINE_Z = scan_pose()

# ------------------------------------------------------------------ misc
COMPRESSED_TCP_Z = 154.0
