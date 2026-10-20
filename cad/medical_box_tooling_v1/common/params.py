"""Shared inputs for the medical-box shelf-pick tooling (left: suction + scan, right: LEHF20K2-48 fingers).

Units: mm, kg, N. Tool poses are in the URDF link6 frame of each igus ReBeL
(z = flange axis pointing out of the flange, flange face at z = FLANGE_Z).
Both tool packages import this file read-only; change a value here and rebuild both.
"""

# ------------------------------------------------------------------ external inputs
DOWNLOADS = '/home/farhan/Downloads'
RESEARCH = '/media/farhan/My Passport/igus_nvidia/shelf_pick_research'   # datasheets (ZP2, ZK2, SE4710, Gemini 305)
PRIOR_TOOLING = '/home/farhan/igus_rebel_ros2/cad/dual_arm_tooling_v1'     # earlier ReBeL tooling (helpers, flange analysis); read-only
GRIPPER_STEP = DOWNLOADS + '/SMC_JP_LEHF20K24800 (2)/LEHF20K2-48(0_0).stp'
GEMINI305_CAD_DIR = DOWNLOADS + '/Gemini 305 CAD_V1.1'
CAD_VENV_PY = '/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python'  # cadquery 2.7 / OCP 7.8

# ------------------------------------------------------------------ carton envelope (size.txt from the box-3d session, caliper)
# thickness x height x length (mm). Boxes 1-3 are the modelled ones; 4 and 5 widen the design envelope.
BOXES = {
    1: dict(name='Boston Scientific Agent MONORAIL', t=15.0, h=230.0, l=250.0),
    2: dict(name='iVascular angiolite',             t=16.0, h=240.0, l=250.0),
    3: dict(name='Terumo Accuforce',                t=16.0, h=220.0, l=250.0),
    4: dict(name='Terumo Ultimaster Nagomi',        t=16.0, h=250.0, l=295.0),
    5: dict(name='unknown (photo missing)',         t=12.0, h=230.0, l=250.0),
}
BOX_T_RANGE = (12.0, 16.0)      # spine width / grip width
BOX_H_RANGE = (220.0, 250.0)    # spine length (vertical on the shelf)
BOX_L_RANGE = (250.0, 295.0)    # depth into the shelf
BOX_MASS_MAX = 0.15             # kg, ASSUMED (cardboard + coiled catheter ~80-100 g); not weighed yet
BOX_T_TOL = 0.5                 # caliper / bulge tolerance on thickness

# ------------------------------------------------------------------ shelf (operator answers 2026-10-05)
# Boxes stand upright like books, spine (t x h) facing the robot, length going into the shelf, packed side by side
# with zero gap, leaning up to SHELF_LEAN_MAX in the shelf front plane. Neighbours may stick out of the front plane.
SHELF_GAP = 0.0
SHELF_LEAN_MAX_DEG = 10.0
NEIGHBOUR_PROTRUSION = 15.0     # ASSUMED: a neighbour spine may stand this far proud of the target spine
SPINE_TILT_MAX_DEG = 3.0        # ASSUMED: spine not perpendicular to the approach (box pitched in/out of the shelf)
PULL_OUT = 100.0                # suction pulls the box this far out; it still rests on the shelf floor
HANDOFF_FINGER_S = (15.0, 70.0) # finger-pad zone along the box length, measured from the (pulled) spine into the box
HANDOFF_TOP_CLEAR = 10.0        # box top edge must stop >= this far below the gripper jaw/body

# ------------------------------------------------------------------ ReBeL flange (from PRIOR_TOOLING/data/flange_interface.json)
FLANGE_Z = 52.7                 # link6 z of the tool-flange face
FLANGE_BOSS_D = 58.0
PCD43_R, PCD43_A0 = 21.5, 45.0  # PRIMARY: 6 x M3 on PCD 43, angles 45 + 60k deg (0 = +X link6), ~4 mm thread depth
PCD35_R, PCD35_A0 = 17.5, 15.0  # spare pattern: 6 x M3 on PCD 35, 15 + 60k deg, ~10 mm deep (unverified on the robot)
M3_CLR, M3_CSK_D, M3_CSK_DEPTH = 3.4, 6.4, 1.7
# No dowel and no ISO 9409 pilot on the ReBeL: the TCP must be re-taught after mounting.

# ------------------------------------------------------------------ purchased parts (research report 2026-10-05)
# SMC ZP2-T4030WNK10-B5: 4x30 oval NBR pad + 10 mm non-rotating buffer, M5x0.8 female inlet on top.
PAD = dict(
    code='ZP2-T4030WNK10-B5',
    outline_l=30.8, outline_w=4.8, pad_h=8.0,
    overall_len=61.5,           # pad lip to top of the buffer adapter, uncompressed
    lip_to_nut_face=25.5,       # pad lip to the face the M10 nut clamps against (= bracket front face)
    body_thread='M10x1', body_hex=14.0, nut_t=3.0, nut_af=14.0,
    bracket_hole=10.5, bracket_t_min=3.0,
    stroke=10.0, spring_n=(1.0, 3.0),
    inlet='M5x0.8', inlet_hex=8.0,
    area_cm2=1.16, lift_n_60kpa=6.9,
)
PAD_FITTING = dict(code='KQ2L04-M5 (elbow, ASSUMED)', height_above_inlet=14.0, tube_od=4.0)
# Vacuum source (operator 2026-10-13): the robot stands on an AMR in the hallway to the operating room -> no compressed
# air, power from the AMR 24 V battery, quiet, filtered exhaust. ELECTRIC pump in an enclosure on the AMR, one tube up
# the arm to the on-tool 3->1 collector. Supersedes the remote SMC ZK2 ejector (v3) and the reserved ZK2 envelope (v1/v2).
VACUUM_PUMP = dict(
    code='KNF N 838 KN.29DC-B (MI)', motor='brushless DC, speed 0.1-5 V / PWM', ip='IP20',
    voltage=24.0, p_max_w=55.0, i_max_a=2.3, mass=1.9,
    envelope=(134.0, 115.0, 129.0),          # L x W x H (datasheet drawing, page 3)
    foot_holes_d=6.5, foot_hole_pitch_w=100.0, ports='G1/8 (inlet + outlet)', outlet_max_bar=0.5,
    ultimate_mbar_abs=100.0,
    # max-speed curve read off the datasheet chart (page 3), abs inlet pressure mbar -> flow L/min (standard state, ISO 8778)
    curve_mbar_lpm=((1000, 34.0), (800, 26.0), (600, 17.5), (500, 13.0), (400, 9.0), (300, 5.5), (200, 3.0), (100, 0.0)),
    current_at_500mbar_a=2.0,                # read off the same chart
    datasheet='https://knf.com/fileadmin/Global_files/Downloads/Product_OEM_Process/Datasheets/KNF_Datasheet-EN_N838KNDC-B_2024-10.pdf',
    noise='not published - ask KNF / measure (bench test)', price='not found',
)
VACUUM_PUMP_ALTERNATIVES = {
    'Schmalz ECBPi 12 24V-DC (10.03.01.00314)': dict(max_vacuum_kpa=-75.0, free_flow_lpm=12.0, p_w=13.0, noise_dba=57.0, mass=0.75,
                                                     note='only if measured leakage <= ~1 L/min per pad; built-in part-present output'),
    '2 x Thomas 1520VDP BLDC HP (15200505)': dict(max_vacuum_kpa=-80.0, flow_at_50kpa_lpm=12.0, p_w=62.0, mass=0.84,
                                                  note='parallel pair, redundancy; about US$600 each'),
}
VACUUM_PARTS = dict(
    switch='SMC ZSE20B(F) digital vacuum switch, PNP, 0..-101 kPa, 12-24 VDC (vacuum-OK signal)',
    vent_valve='24 VDC 2/2 N.C. solenoid valve to atmosphere (release), through its own intake filter - model to choose',
    inlet_filter='inline vacuum filter (cardboard dust) before the switch and pump, e.g. SMC ZFC series - to confirm',
    exhaust_filter='Whatman HEPA-VENT 50 mm (99.97 % at 0.3 um) on the pump outlet; keep back-pressure << 0.5 bar',
    reservoir=None,                          # not needed: pump runs during the pick, evacuation ~0.1 s
)
SCANNER = dict(code='Zebra SE4710', size=(22.3, 13.7, 8.1), mass=0.0048, fov_deg=(42.0, 28.0),
               dm10_range=(74.0, 257.0), c128_5mil_range=(71.0, 208.0), mount='M1.4x0.3 (Zebra guide fig. 2-4)',
               flex='27-pin ZIF, 89.4 mm flex', decoder='PL3307 B 26.91 x 39.04 x 6.37 mm', glare_tilt_deg=12.0)
SCAN_STANDOFF = 150.0           # nominal scan distance scanner window -> spine
CAMERA = dict(code='Orbbec Gemini 305', size=(42.0, 42.0, 23.0), mass=0.068, min_z=50.0, good_min_z=90.0,
              depth_fov_deg=(88.0, 65.0), color_fov_deg=(94.0, 68.0),
              mount_bottom='1/4-20 UNC (7 deep) + 2 x M3 (4.8 deep) at +-10 mm', mount_back='2 x M3 (4.8 deep)')
GRIPPER = dict(code='SMC LEHF20K2-48', step=GRIPPER_STEP, size=(158.0, 35.0, 83.0), mass=0.75,
               stroke_total=48.0, force_n=(11.0, 28.0),
               hl_limit_100pct=55.0, hl_limit_70pct=80.0,       # H + L limit vs gripping point (catalog p.44)
               jaw_m3=(-23.5, -5.5, 5.5, 23.5), jaw_dowel_d=2.5,
               body_m5_pattern=(40.0, 22.0), allow_moment_nm=dict(pitch=0.68, yaw=0.68, roll=1.4))

# ------------------------------------------------------------------ build rules shared by both tools
PRINT_MATERIAL = 'PETG (PA12-CF for highly loaded parts)'
MIN_WALL = 3.0
INSERT_M3_HOLE, INSERT_M4_HOLE = 4.2, 5.6
CLEARANCE_MIN = 3.0             # mm between any two non-touching parts in the assembly
ROBOT_PAYLOAD = 2.0             # kg, ReBeL
