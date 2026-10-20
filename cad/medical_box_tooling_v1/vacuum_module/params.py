"""Vacuum module enclosure (electric pump + switch + vent valve + filters) that sits on the AMR. Units: mm, kg, W, K.

Frame: x = long axis of the module (+x = SERVICE side, removable door), y = width, z = up; z = 0 is the underside of the base plate
(the module stands on 4 spacers that reach z = -SPACER_H).  Shared values come from ../common/params.py (read-only).
Every number marked ASSUMED is a placeholder to be replaced by a datasheet / CAD / measurement (listed again in README.md).
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import params as C          # noqa: E402  (shared, read-only)

PY = C.CAD_VENV_PY
ASSUMED = []                            # filled below: (name, value, why)


def A(name, value, why):
    ASSUMED.append((name, value, why))
    return value


# ----------------------------------------------------------------------------- enclosure skeleton
OUT_X, OUT_Y, OUT_Z = 240.0, 186.0, 220.0   # housing + door outer size (z from the base plate top, z = 6)
X0, X1 = -120.0, 120.0
Y0, Y1 = -93.0, 93.0
BASE_T = 6.0                            # base plate thickness (z 0..6)
Z_TOP = OUT_Z                           # housing roof outer face
WALL = 3.0                              # printed outer wall (>= C.MIN_WALL)
FOAM_T = 10.0                           # acoustic foam thickness (operator brief)
FILM_T = 0.3                            # inner PET film that holds the foam (ASSUMED)
INNER = WALL + FOAM_T + FILM_T          # 13.3: outer face -> cabin face
DOOR_X = X1 - INNER                     # 106.7: housing end face / door rim plane / cabin face at the service side
CAB_X0, CAB_X1 = X0 + INNER, DOOR_X     # cabin (inside the film): -106.7 .. 106.7
CAB_Y = Y1 - INNER                      # +-79.7
CAB_Z0, CAB_Z1 = BASE_T, Z_TOP - INNER  # 6 .. 206.7
SPACER_H, SPACER_D = 12.0, 18.0         # gap between module and AMR deck for the exhaust (nozzle exits downwards into it)
LEDGE_T = 8.0                           # bottom ledge of the housing: foam rests on it, M4 inserts for the base screws

# AMR mounting: generic 4 x M6 slots (the AMR model is unknown) - change the pattern here
AMR_SLOT_PITCH = (180.0, 100.0)         # x, y centre-to-centre
AMR_SLOT_LEN, AMR_SLOT_W = 4.0, 6.6     # slot along x (+-2 mm of adjustment), M6 clearance
AMR_SLOT_XY = [(sx * AMR_SLOT_PITCH[0] / 2, sy * AMR_SLOT_PITCH[1] / 2) for sx in (-1, 1) for sy in (-1, 1)]
AMR_WASHER = (26.0, 14.0, 2.0)          # M6 plate washer x, y, t (to choose): also closes the slot acoustically for any bolt position

# ----------------------------------------------------------------------------- pump KNF N 838 KN.29DC-B (datasheet p.3 drawing, local frame x from head end, z from foot underside)
PUMP = C.VACUUM_PUMP
PUMP_L, PUMP_W, PUMP_H = PUMP['envelope']                                  # 134 x 115 x 129
PUMP_FOOT_X = A('pump foot hole x (from head end)', (63.0, 113.0), 'drawing p.3: 50 mm hole pitch read at 400 dpi, absolute position from the head end +-3 mm')
PUMP_FOOT_Y = PUMP['foot_hole_pitch_w'] / 2                                  # 50 (datasheet 100)
PUMP_FOOT_D = PUMP['foot_holes_d']                                           # 6.5
PUMP_FOOT_PLATE = A('foot plate x range / thickness', (56.5, 119.5, 5.0), 'read off the drawing, +-2 mm')
PUMP_HEAD = A('head block x 0..79, y +-41.5, z 5..104; cap z 104..129', (79.0, 41.5, 104.0), 'read off the drawing, +-2 mm (envelope 134 x 115 x 129 is from the datasheet)')
PUMP_MOTOR = A('motor cylinder d56, axis x, z centre 45, x 79..134', (56.0, 45.0), 'read off the drawing, +-3 mm')
PUMP_BRIDGE = (60.0, 115.0, 30.0, 17.0)                                      # x0, x1, half width, top z (bridge between foot plate and motor; ASSUMED)
PUMP_PORT_X = A('pump port x (G1/8, on the cap top)', 71.0, 'drawing p.3 shows the G1/8 (2x) bosses at the right end of the cap; +-4 mm')
PUMP_PORT_Y = A('pump port y (inlet -, outlet +)', 20.0, 'drawing shows only one port in the side view; which port is inlet/outlet is NOT readable - check the KNF CAD')
MOUNT_H, MOUNT_D = A('anti-vibration mount height/diameter (M6 stud + M6 female)', (15.0, 20.0), 'rubber mount to choose (e.g. a 20 x 15 type M6/M6, Shore 55)')
PUMP_X0 = A('pump x of the head end in the module', -99.5, 'layout: 6 mm from the rear cabin face')
BOSS_H = 5.0                            # base-plate boss under each mount (M6 insert)
PUMP_Z0 = BASE_T + BOSS_H + MOUNT_H     # foot underside z = 26
PUMP_CABLE = 'pigtail 300 +-10 mm out of the motor end (datasheet drawing); to be terminated on the terminal block'


def pg(xl, yl, zl):
    """pump local -> module frame"""
    return (PUMP_X0 + xl, yl, PUMP_Z0 + zl)


PUMP_FOOT_XY = [(PUMP_X0 + x, s * PUMP_FOOT_Y) for x in PUMP_FOOT_X for s in (-1, 1)]
PORT_Z = PUMP_Z0 + PUMP_H               # 155
PORT_IN = (PUMP_X0 + PUMP_PORT_X, -PUMP_PORT_Y, PORT_Z)     # (-28.5, -20, 150)
PORT_OUT = (PUMP_X0 + PUMP_PORT_X, +PUMP_PORT_Y, PORT_Z)
ELBOW = A('G1/8 -> 8 mm push-in elbow envelope d14, rises 12 mm above the port, stub 8 mm', (14.0, 12.0, 8.0), 'SMC KQ2L08-01 style (to confirm)')
Z_LINE = PORT_Z + ELBOW[1]              # 167: the tube plane over the deck
PUMP_MASS = PUMP['mass']

# ----------------------------------------------------------------------------- purchased parts (envelopes ASSUMED unless noted)
SW_ENV = A('vacuum switch SMC ZSE20B with fitting', (25.0, 30.0, 30.0), 'task brief: about 30 x 30 x 25; port assumed on the -y face at mid height')
SW_MASS = A('switch mass', 0.1, 'to check')
VALVE_ENV = A('vent valve 2/2 N.C. 24 V', (22.0, 22.0, 50.0), 'task brief: about 22 x 22 x 50; vacuum port assumed on the +y face 17 mm above its bottom')
VALVE_PORT_DZ = 17.0
VALVE_FILTER = A('vent valve intake filter (foam/sintered)', (12.0, 12.0), 'd12 x 12 on the -y face; model to choose')
VALVE_MASS = A('valve mass', 0.12, 'to check')
DUST = A('inline dust filter (SMC ZFC-like), axis x', (25.0, 60.0), 'task brief: about d25 x 60')
DUST_MASS = A('dust filter mass', 0.08, 'to check')
HEPA = A('HEPA exhaust filter (Whatman HEPA-VENT 50, axis x)', (56.0, 30.0), 'task brief: about d56 x 30 with barbs; ports assumed on the rim (top = inlet, bottom = outlet)')
HEPA_MASS = A('HEPA mass', 0.05, 'to check')
MUFFLER = A('exhaust muffler, axis x', (20.0, 40.0), 'G1/8-type silencer d20 x 40 (to choose)')
MUFFLER_MASS = A('muffler mass', 0.03, 'to check')
MANIFOLD = A('4-port vacuum manifold (R in, filter, valve, switch)', (16.0, 28.0, 16.0), 'x, y, z; small aluminium/plastic block with 8 mm push-in ports (to choose)')
MANIFOLD_MASS = A('manifold mass', 0.06, 'to check')
FAN = A('cooling fan 24 V, 40 x 40 x 10 mm', (40.0, 40.0, 10.0), 'quiet 24 V fan to choose; about 7 m3/h free air, <= 22 dB(A) is the target (ASSUMED)')
FAN_MASS = A('fan mass', 0.025, 'to check')
TERMINAL = A('terminal blocks (24 V, GND, speed, valve, switch, fan)', (50.0, 14.0, 12.0), 'DIN-less spring terminal strip (to choose)')
TERMINAL_MASS = A('terminal block + wires mass', 0.12, 'to check')
M12_ENV = A('M12 panel connector (A-coded, rear mount, M16 thread)', (20.0, 20.0, 3.0), 'flange 20 x 20 x 3 outside, body d16 through the wall, ends 2 mm into the cabin; pin count and current rating to choose')
M12_POS = (58.0, 90.0)                  # y, z on the rear wall
M12_MASS = A('M12 connector mass', 0.04, 'to check')
FIT_MASS = A('push-in fittings + internal tubes + screws + inserts (total)', 0.25, 'to check')
TUBE_OD = {'in8': 8.0, 'ex8': 8.0}
TUBE_MIN_R = {4.0: 15.0, 6.0: 20.0, 8.0: 25.0}   # tube OD -> minimum bend radius (brief)
BEND_R = 25.0                           # used for every 8 mm tube
RHO_FOAM = A('acoustic foam density (PU, 30 kg/m3)', 30.0e-9, 'kg/mm3; open-cell foam sheet to choose, fire/hospital grade (e.g. B1/V0)')
RHO_FILM = A('PET film 0.3 mm', 1.39e-6, 'kg/mm3')
RHO = dict(PETG=1.27e-6, rubber=1.2e-6, alu=2.70e-6)   # kg/mm3

# ----------------------------------------------------------------------------- layout (module frame)
RACK_Z = 143.0                          # deck plate z0..z1 (top at 146)
DECK_T = 3.0
DECK_X = (-12.0, 104.0)
DECK_Y = 74.0
DECK_TOP = RACK_Z + DECK_T              # 146
LEG_Y = (64.0, 74.0)
LEG_X = ((-12.0, 0.0), (88.0, 100.0))
HEPA_C = (58.0, 20.0, 54.0)             # x (axis x, 43..73), y, z centre (26..82)
DUST_C = (40.0, -20.0, Z_LINE)          # x centre (10..70), y, z
MUFFLER_C = (6.0, 20.0, Z_LINE)         # x centre (-14..26)
MANI_C = (88.0, -10.0, Z_LINE)          # centre; block x 80..96, y -24..4
R_XY = (88.0, -10.0)                    # roof bulkhead (to the arm tube)
R_PORT_Z = 190.0
SWITCH_C = (88.0, 31.0, Z_LINE)         # x centre, y centre (16..46), z centre
VALVE_Y = (-58.0, -36.0)
VALVE_X = (77.0, 99.0)
VALVE_Z0 = Z_LINE - VALVE_PORT_DZ
EXH_XY = (58.0, 20.0)                   # exhaust bulkhead through the base (nozzle exits downwards)
TERM_C = (35.0, 63.0)
FAN_XZ = (-40.0, 40.0)                  # intake fan centre (x, z) on the -y wall
DUCT_X = (-61.5, 101.5)                 # vent duct block extent along the wall
EXH_Z = 150.0                           # exhaust duct centre z (+y wall)
DUCT_H = 46.0
DUCT_WALL = 3.0                         # duct partition walls (>= MIN_WALL)
DUCT_INNER_T = 3.0                      # inner (film-flush) wall of the duct, holds the fan screws
SLOT_X0, SLOT_PITCH, SLOT_W, SLOT_N, SLOT_H = 70.0, 6.0, 3.0, 5, 32.0

# ----------------------------------------------------------------------------- print / fasteners
BED = (250.0, 250.0)
BED_Z = A('printer build height', 250.0, 'at least 250 mm assumed; the housing is 214 mm tall printed roof-down')
PRINT_MATERIAL = C.PRINT_MATERIAL
INSERT_M4_HOLE = C.INSERT_M4_HOLE
INSERT_M6_HOLE = A('M6 heat-set insert hole d', 8.0, 'for the pump mount studs (M6 female mount is bolted from below instead if preferred)')
M4_CSK = (4.5, 8.4)                     # clearance d, countersink d

# ----------------------------------------------------------------------------- thermal model inputs (ASSUMED)
T_AMB = A('ambient in the hospital hallway', 25.0, 'C; pump rated +5..+40 C ambient (datasheet)')
PUMP_P_MAX, PUMP_P_TYP = PUMP['p_max_w'], 48.0
DUTY_ON_S, DUTY_PERIOD_S = A('duty: pump on per pick / period', (8.0, 30.0), 'task brief')
H_IN, H_OUT = A('film-side / outer-side surface coefficients W/m2K', (5.0, 7.5), 'natural convection + radiation')
K_PETG, K_FOAM = 0.2, 0.040
FAN_FLOW_M3H = A('fan flow through the labyrinth with the fan running', 3.0, 'about 40 % of a 7 m3/h free-air fan against the two ducts (ASSUMED)')
C_THERMAL = A('lumped thermal capacity of pump + parts + air', 3000.0, 'J/K: pump 1.9 kg x ~0.7 kJ/kgK + other parts + air')
PUMP_LIMIT_C = 40.0                     # datasheet ambient limit
