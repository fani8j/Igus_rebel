"""Left-arm suction/scan end effector v3 (push-in buffers, on-tool collector, remote vacuum source) - tool-specific parameters (mm, kg, N, link6 frame).

Shared inputs are imported read-only from ../common/params.py (never edited here).
Only parameters that belong to this tool live in this file; everything else is derived.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location('common_params', os.path.join(HERE, '..', 'common', 'params.py'))
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)      # shared, read-only  (../common/params.py)

SCRATCH = '/media/farhan/My Passport/igus_nvidia/scratch_suction_tool'
GEMINI_STEP = C.GEMINI305_CAD_DIR + '/3D/3D__00_mb2502_asm(G305)_ V1.1.stp'

# ------------------------------------------------------------------ pad row
PAD_PITCH = 60.0                      # mm between pad axes along link6 X
PAD_X = [-PAD_PITCH, 0.0, PAD_PITCH]
LIP_Z = 159.0                         # link6 z of the (uncompressed) pad lips: the single head-height parameter (v1: 165)
FLANGE_Z = C.FLANGE_Z
PAD = dict(C.PAD)
PAD.update(code='ZP2-T4030WNK10-04')    # one-touch variant: built-in phi4 push-in on the buffer top (catalog p.183, KQ2H04-M5N)
FIT_H = 14.7                          # one-touch fitting height above the buffer top (A = 61.5) - catalog p.183 sketch; ASSUMED fully outside A
FIT_AF = 8.0
FIT_D = 9.0                           # modelled fitting body diameter (envelope)
TUBE_STRAIGHT = 8.0                   # straight tube run out of the fitting before the bend (2 x OD)
TUBE_BEND_R = 12.0                    # tube bend radius used for routing (phi4 nylon/PU: min ~10-12)
KO_REAR = 40.0                        # everything but pads/bracket must be at z <= LIP_Z - KO_REAR (req. 3)
KO_BRACKET_FRONT = 25.0               # slab LIP_Z-25 .. : pads / buffers only
BRACKET_FRONT = PAD['lip_to_nut_face']          # 25.5: bracket front face below the lip
BRACKET_REAR = KO_REAR                           # bracket rear face at LIP_Z-40 (bracket thickness 14.5 mm)
BRACKET_HALF_Y = 12.0                 # bracket width 24 mm (req. <= 24)
BRACKET_HALF_X = PAD_PITCH + 12.0     # stadium, end radius 12
NUT_T = PAD['nut_t']; NUT_AF = PAD['nut_af']
HEX_POCKET_AF = 14.25                 # printed pocket for the front M10x1 14 AF nut (keys the nut)
WASHER = dict(od=20.0, id=10.5, t=1.0)

# ------------------------------------------------------------------ flange adapter plate + carrier foot
FLOOR_T = 8.0                         # carrier floor = flange adapter (v1: separate 9 mm plate + 8 mm foot)
FLOOR_D = 56.0                        # flange screws sit on r = 21.5 (+3.2 head +3.3 wall = 28)
FLOOR_BORE = 20.0
FLOOR_Z0, FLOOR_Z1 = FLANGE_Z, FLANGE_Z + FLOOR_T          # 52.7 .. 60.7
FLANGE_SCREW = 'ISO 10642 M3x12 countersunk (4 mm engagement; flange thread ~4 mm deep)'
BRACKET_ACCESS_D = 6.2                # driver / CSK head access holes through the bracket for the two flange screws under it
FLAP_X, FLAP_Y = 46.0, (-43.0, -12.0) # collector / tube flap (floor level, -y side): carries the collector, the L/R tube ties (x = +-40) and the trunk relief

# ------------------------------------------------------------------ ribs (tower)
RIB_X = (24.8, 35.2)                  # ribs at +-[24.8,35.2] (10.4 thick: M3 insert holes keep 3.1 mm walls); centre line +-30 = M4/ZK2 pitch 60
RIB_XC = 0.5 * (RIB_X[0] + RIB_X[1])  # screw / insert line
RIB_HALF_Y = 12.0

# ------------------------------------------------------------------ camera (Gemini 305, +Y side)
CAM = C.CAMERA
CAM_TILT_DEG = 39.0                   # optical axis tilted about +X toward -y (rotation of z by this angle)
CAM_AIM = (0.0, 0.0, LIP_Z + 40.0)    # optical axis passes through this point (req. 4)
CAM_HALF = 21.0                       # half of 42 mm
CAM_DEPTH = 23.0
CAM_TOP_MARGIN = 1.0                  # camera highest point is at LIP_Z - KO_REAR - this
CAM_CLEAR = 0.25                      # printed pocket clearance
CAM_CRADLE_T = 6.0                    # cradle plate thickness

# ------------------------------------------------------------------ scanner (SE4710, -Y side)
SCN = C.SCANNER
SCAN_TILT_DEG = SCN['glare_tilt_deg']             # 12 deg about X toward the tool axis
SCAN_STANDOFF = C.SCAN_STANDOFF
SCAN_WINDOW_Z = LIP_Z - 47.5          # window centre z (>= 40 mm behind the lip plane, req. 5)
ENGINE = dict(long=22.3, depth=13.7, thick=8.1)   # long axis along link6 X, depth along the scan axis, thickness across
SCAN_FOV_DEG = SCN['fov_deg']         # (42 along X [ASSUMED], 28 across)
DECODER = dict(size=(26.91, 39.04, 6.37), flex_reach=89.4, mass=0.006)   # PL3307 board; mass ASSUMED

# ------------------------------------------------------------------ vacuum (source is REMOTE: electric pump + switch + vent valve in an enclosure on the AMR; operator decision 2026-10-13)
TUBE_OD = 4.0
TRUNK_OD = 6.0
COLL_Y = -30.0                        # collector centre y (port axis height = the tube run height)
COLLECTOR = dict(code='SMC push-in branch/manifold block 1 x phi6 in + 3 x phi4 out - "to confirm" (not in the research catalogs; proxy)',
                 body=(24.0, 10.0, 16.0),             # x, y, z of the block body (proxy)
                 boss4=(9.0, 5.0), boss6=(11.0, 4.0),  # port boss (diameter, length): phi4 push-in x3, phi6 push-in x1
                 mass=0.012, internal_cm3=1.0)         # ASSUMED
TRUNK_END_Y = -43.0                   # trunk stub end in the CAD (continues off-tool)
VAC = dict(code='%s (pump data: common/params.py VACUUM_PUMP) in an enclosure on the AMR; inline dust filter -> tee with %s + N.C. vent valve -> pump inlet; pump outlet -> HEPA filter' % (C.VACUUM_PUMP['code'], C.VACUUM_PARTS['switch'].split(',')[0]),
           pump=C.VACUUM_PUMP, parts=C.VACUUM_PARTS,
           switch_location='AMR enclosure (pump side, after the inlet filter), not at the pads', switch_accuracy_kpa=2.0,   # SMC ZSE20B catalog, +-2 % F.S.
           # trunk = tool collector -> AMR enclosure, about 2.0 m in total. Option A: all phi6 x 4 ID. Option B: phi6 x 4 ID on the arm + phi8 x 6 ID from the arm base into the AMR.
           trunk_len_mm=2000.0, trunk_od_mm=TRUNK_OD, trunk_id_mm=4.0,
           trunk_arm_len_mm=1200.0, trunk_amr_len_mm=800.0, trunk_amr_od_mm=8.0, trunk_amr_id_mm=6.0,     # option B lengths (arm part + AMR part = trunk_len_mm)
           tube_id_mm=2.5,
           filter_switch_tee_cm3=10.0,        # ASSUMED: inline filter + switch port + tee + short stubs
           pump_on_s_per_pick=8.0)            # ASSUMED

# ------------------------------------------------------------------ printing / mass model
PETG_DENSITY = 1.27e-6                # kg/mm3 solid
PRINT_FILL_EQ = 0.55                  # effective solid fraction (3 walls + 20 % gyroid) - ASSUMED
MASS_PURCHASED = dict(pad=0.015, nut=0.0025, washer=0.001, tube=0.001, trunk=0.001,   # pad = 12 g body + 3 g built-in one-touch fitting (ASSUMED)
                      
                      scanner=SCN['mass'], decoder=DECODER['mass'], camera=CAM['mass'])
E_PETG = 1800.0                       # MPa, 3D-printed PETG (XY) first-order

# ------------------------------------------------------------------ derived camera pose
def cam_pose():
    a = math.radians(CAM_TILT_DEG); s, c = math.sin(a), math.cos(a)
    zc = LIP_Z - KO_REAR - CAM_TOP_MARGIN - CAM_HALF * s            # front-face centre z (highest point = front, +u edge)
    yc = (CAM_AIM[2] - zc) * math.tan(a)                              # axis through CAM_AIM
    d = np.array([0, -s, c]); u = np.array([0, c, s]); x = np.array([1.0, 0, 0])
    T = np.eye(4); T[:3, 0] = x; T[:3, 1] = u; T[:3, 2] = d; T[:3, 3] = [0, yc, zc]
    return T                           # cam frame: x = tool X, y = "up" (away from bottom mount face), z = optical axis

def scan_pose():
    a = math.radians(SCAN_TILT_DEG); s, c = math.sin(a), math.cos(a)
    ax = np.array([0, s, c]); w = np.array([0, c, -s])
    pos = np.array([0, -SCAN_STANDOFF * s, SCAN_WINDOW_Z])           # crosses y=0 at STANDOFF along the axis
    T = np.eye(4); T[:3, 0] = [1, 0, 0]; T[:3, 1] = w; T[:3, 2] = ax; T[:3, 3] = pos
    return T                           # z = scan axis out of the window, y = w (across the engine thickness)

T_CAM = cam_pose()
T_SCAN = scan_pose()
