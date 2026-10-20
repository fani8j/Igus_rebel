"""Right-arm gripper tool (SMC LEHF20K2-48 in-line on the ReBeL flange + custom fingers). Units: mm, kg, N, degrees.

Frame: URDF link6 (z = flange axis out of the flange, flange face z = FLANGE_Z).  Shared values come from ../common/params.py (read-only);
everything below is specific to this tool.  Change a value here and run `python build.py && python verify.py`.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import params as C          # noqa: E402  (shared, read-only)

PY = C.CAD_VENV_PY
SCRATCH = '/media/farhan/My Passport/igus_nvidia/scratch_gripper_tool'

# ----------------------------------------------------------------------------- flange stack (flange face -> gripper)
FLANGE_Z = C.FLANGE_Z                   # 52.7
DISC_T = 8.3                            # flange disc; M3x12 ISO 10642 flush -> 12 - 8.3 = 3.7 mm engagement in the 4 mm flange threads (>= 3.5, 0.3 mm to bottoming)
M3_SCREW_L = 12.0
DISC_R = 36.0                           # disc radius (flange boss is r = 29)
PLATE_T = 6.0                           # gripper plate (bolted to the gripper with countersunk M5 from the flange side)
Z_DISC_TOP = FLANGE_Z + DISC_T
Z_R = FLANGE_Z + DISC_T + PLATE_T       # rear face of the gripper body (SMC mounting face opposite the jaws)

# SMC STEP frame (x stroke axis, y thickness -35..0, z length -46 (rear face) .. +37 (jaw end face)) -> link6:  x_l = x_s, y_l = y_s + 17.5, z_l = z_s + 46 + Z_R
GR_X0, GR_Y0, GR_Z0 = 0.0, 17.5, 46.0 + Z_R
ZS_JAW_FACE = 37.0                      # jaw finger-mounting plane in the STEP (raised strips; 36.8 elsewhere)
Z_J = ZS_JAW_FACE + GR_Z0               # jaw mounting plane in link6 (fingers start here)
JAW_STROKE = 24.0                       # per jaw (48 mm total, catalog: 0 closed / 48 +-1 open)
JAW_M3_X = (5.5, 23.5)                  # |x| of the M3 pattern on each jaw (STEP: -23.5,-5.5 on jaw 1; 5.5,23.5 on jaw 2) at the closed position
JAW_M3_Y = (-7.5, 7.5)                  # y_s = -25 / -10  ->  link6 y = -7.5 / +7.5
JAW_DOWEL_X, JAW_DOWEL_D, JAW_DOWEL_DEPTH = 23.5, 2.5, 2.5
GRIPPER_MASS = C.GRIPPER['mass']

# rear-face features of the SMC body (STEP, link6 y):  M5 40 x 22 pattern, dowel holes d4H9 x 3 deep (round + slot) at x = 0
M5_XY = [(sx * 20.0, sy * 11.0) for sx in (-1, 1) for sy in (-1, 1)]
BODY_DOWEL_XY = [(0.0, 11.0), (0.0, -11.0)]
BODY_DOWEL_D = 4.0

# ----------------------------------------------------------------------------- flange disc
PCD43 = [(C.PCD43_R * math.cos(math.radians(C.PCD43_A0 + 60 * k)), C.PCD43_R * math.sin(math.radians(C.PCD43_A0 + 60 * k))) for k in range(6)]
PCD35 = [(C.PCD35_R * math.cos(math.radians(C.PCD35_A0 + 60 * k)), C.PCD35_R * math.sin(math.radians(C.PCD35_A0 + 60 * k))) for k in range(6)]
M3_CLR, M3_CSK_D, M3_CSK_DEPTH = C.M3_CLR, C.M3_CSK_D, C.M3_CSK_DEPTH
M4_XY = [(sx * 14.0, sy * 25.5) for sx in (-1, 1) for sy in (-1, 1)]     # M4 inserts in the disc / clearance in the plate, outside the gripper footprint (|y| >= 25.5)
M4_INSERT_HOLE, M4_INSERT_DEPTH = C.INSERT_M4_HOLE, 4.5                 # 4 mm long heat-set inserts
M4_CLR = 4.5
DOWEL_PIN = 4.0                         # ISO 8734 4 x 10: 3 in the gripper, 6 in the plate, 1 in the disc
DOWEL_HOLE_PLATE, DOWEL_HOLE_DISC, DOWEL_DISC_DEPTH = 4.05, 4.1, 1.5

# ----------------------------------------------------------------------------- gripper plate
PLATE_X, PLATE_Y, PLATE_RCLIP = 29.0, 31.0, 40.0                       # half sizes, clipped by a circle R40
M5_CLR, M5_CSK_D, M5_CSK_DEPTH = 5.5, 10.4, 2.8                         # ISO 10642 M5

# ----------------------------------------------------------------------------- fingers (local frame: origin on the gripper centre axis at the jaw plane, z away from the jaw)
PAD_GAP_CLOSED = 8.0                    # crest to crest, fully closed  -> 8 + 2*opening
PAD_CREST = PAD_GAP_CLOSED / 2          # |x| of the rib crests at the closed position (4.0)
RIB_H = 0.4
FRAME_FACE = PAD_CREST + RIB_H          # |x| of the finger frame face around the pad (4.4)
PAD_BASE_T = 2.6                        # TPU base thickness (pocket depth); + 0.4 ribs = 3.0 total
PAD_W, PAD_H = 50.0, 45.0               # along link6 Y, along z
PAD_CLEAR = 0.2                         # pocket clearance per side
RIB_PITCH, RIB_W = 1.5, 0.6
FOOT_X = (1.0, 28.0)                    # |x| range of the foot on the jaw face (jaw face 1..28)
FOOT_Y, FOOT_T = 20.0, 7.0
FOOT_CB_D, FOOT_CB_DEPTH = 6.0, 2.3     # DIN 7984 M3 low head (d 5.5 x 2.0)
FOOT_SCREW = 'M3x8 DIN 7984'
FINGER_PIN_D, FINGER_PIN_DEPTH = 2.5, 4.0
COL_X = (11.0, 17.0)                    # |x| range of the column behind the window
BLOCK_X = (FRAME_FACE, 17.0)
BLOCK_Y = PAD_W / 2 + PAD_CLEAR + 3.0   # 28.2
Z_REG = 17.5                            # box top edge stops here (registration plane = face of the fixed centre top-stop); window below it for the inner screws
LIP = 3.0                               # wall between the stop plane and the pad pocket (pocket bottom wall)
Z_POCKET0 = Z_REG + LIP                 # pocket bottom
PAD_Z0 = Z_POCKET0 + PAD_CLEAR          # TPU pad zone along z
PAD_Z1 = PAD_Z0 + PAD_H
Z_POCKET1 = PAD_Z1 + PAD_CLEAR
Z_LIP_TOP = Z_POCKET1 + 3.0
CHAMF = 8.0                             # 45 degree lead-in chamfer at the tip (inner face); v3: 5 -> 8 for +-8 mm capture
Z_TIP = Z_LIP_TOP + CHAMF
POCKET_DEPTH = PAD_BASE_T
RET_INSERT_HOLE, RET_INSERT_DEPTH = C.INSERT_M3_HOLE, 5.0
RET_Z = (PAD_Z0 + 6.0, PAD_Z1 - 6.0)    # retention screws (M3x6 ISO 7380) at y = 0
PAD_CB_D, PAD_CB_DEPTH = 6.4, 1.9
L_PAD = (PAD_Z0 + PAD_Z1) / 2           # pad centre distance from the jaw plane (catalog "L")
H_OVERHANG = 0.0                        # pad centre offset from the guide centre plane along link6 Y (catalog "H")
S_PAD_CENTRE = 42.5                     # grip zone centre behind the spine (15..70)

# ----------------------------------------------------------------------------- materials
RHO = dict(PA12CF=1.15e-6, PETG=1.27e-6, TPU=1.21e-6, steel=7.85e-6, alu=2.70e-6)   # kg/mm3
E_MPA = dict(PA12CF=3200.0, PETG=1800.0, alu6082=70000.0)           # printed XY values, conservative
MU = 0.5                                # pad friction on cardboard (conservative, shared brief)
FORCE_100 = (28.0, 23.0)                # LEHF20 per-finger force at 100 % pushing: L = 10 -> 28 N, L = 55 -> 23 N (catalog p.43 graph)
FORCE_70 = (19.7, 14.0)                 # L = 10 -> 80
FORCE_40 = (11.3, 8.0)
FORCE_ACC = 0.25                        # +-25 % F.S. accuracy (LEHF20)


# ----------------------------------------------------------------------------- fixed centre top-stop (does not move with the jaws)
Z_STOP = Z_J + Z_REG                    # stop face (toward the box) = box top edge reference plane
STOP_T = 3.0                            # v3: crossbar thickness along x (centred on x = 0): 2.5 mm to each closed pad crest (v2: 4.0 -> 2.0 mm)
STOP_Y = 27.5                           # crossbar half length along y: s = 15 ... 70 for the grip centre s = 42.5
STOP_H = 7.5                            # v3: crossbar height (z_stop-7.5 ... z_stop): 3.0 mm above the finger feet (top at Z_J + FOOT_T)
RISER_Y = (22.5, 27.5)                  # v3: |y| of the two risers: 2.5 mm outside the feet (|y| = 20)
RISER_Z0 = Z_J - 12.5
SIDE_X, SIDE_Y = 19.0, (17.5, 24.0)     # side plates on the body faces y = +-17.5 (x half width, |y| range)
EAR_X, EAR_Y, EAR_Z0 = 6.0, (17.5, 27.5), Z_J - 24.0   # v3: local thickening |x| <= 6 so the 5 mm riser sits fully on the side plate
BOLT_X = (-15.0, 15.0)                  # body d5.2 through holes along y at x = +-15, z = GR_Z0 (z_s = 0)
BOLT_Z = GR_Z0
SIDE_Z = (BOLT_Z - 9.0, Z_J - 2.5)      # side plate z range; top 2.5 mm below the feet
STOP_BOLT = 'M5x55 DIN 912 + M5 nyloc nut DIN 985'
BOLT_HOLE = 5.4
STOP_HOLD_N = 15.0                      # stiffness load case

# ----------------------------------------------------------------------------- v3 registration procedure (position based, no force sensing)
PRECLOSE_OFFSET = 4.0                   # pad gap = t + 4 during the descent
G_NOM = 4.0                             # nominal gap box top edge -> stop face after the descent (stop face ABOVE = toward the flange)
TIP_CLEAR = 10.0                        # fingertips this far above the box top edge when the fingers are pre-closed
JAW_BACKLASH = 0.5                      # per side (catalog)
PRINT_TOL = 0.3                         # per side on the chamfer / pad faces
TOL_STACK = dict(carton_height_lookup=1.0, shelf_floor_height=1.5, tcp_z_calibration=1.0)   # +- mm, ASSUMED
SPINE_TILT_DEG = 3.0                    # shared brief: spine pitched up to 3 deg over the stop length
LEAN_DEG = 2.0                          # box lean in the shelf front plane the tool is aligned to (residual after tilt compensation +-1 deg)
LATERAL_ERR_REQ = 8.0                   # mm, dual-arm calibration + 2 deg lean at 230 mm (coordinator)
