"""Can a small pump hold the carton? Steady-state + transient vacuum model for the 3 x SMC ZP2-4030WN pads.

For each candidate pump: pad vacuum vs leakage per pad, compared with the vacuum needed to pull the carton
out of a tightly packed shelf (with a safety factor). Answer = the largest leakage per pad each pump tolerates.
Leakage is the one unmeasured input, so the result is a threshold to check on the bench, not a yes/no.

Run: python3 vacuum_sim/pump_sim.py   (needs numpy + matplotlib; writes vacuum_sim/data/ and vacuum_sim/renders/)
Units: vacuum as positive kPa below 1000 mbar abs; flows in standard L/min (ISO 8778).
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import params as C   # noqa: E402

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
os.makedirs(os.path.join(HERE, 'renders'), exist_ok=True)

P_ATM, P_STD, T_K, MU, R_AIR = 1000.0, 1013.25, 293.15, 1.81e-5, 287.05
RHO_STD = P_STD * 100 / (R_AIR * T_K)
G = 9.81

# ------------------------------------------------------------------ pumps
# Only the KNF has a published curve. The others publish free flow + max vacuum only; their curve is the KNF curve
# normalised between (ultimate pressure, 0) and (1000 mbar, free flow) - ASSUMED shape, typical for diaphragm pumps.
KNF = sorted((float(a), float(q)) for a, q in C.VACUUM_PUMP['curve_mbar_lpm'])
KNF_ULT, KNF_Q0 = 100.0, 34.0
SHAPE = [((p - KNF_ULT) / (P_ATM - KNF_ULT), q / KNF_Q0) for p, q in KNF]

def shaped(q0, vac_max_pct):
    ult = P_ATM * (1 - vac_max_pct / 100.0)
    return [(ult + x * (P_ATM - ult), y * q0) for x, y in SHAPE]

TRUNK_TOOL = [(150.0, 4.0)]              # pump on the flange: collector -> pump, ~150 mm of phi4 bore (ASSUMED)
TRUNK_STAND = [(1600.0, 4.0)]            # pump on the lift-carriage bracket (robot_stand P1): ~1.6 m phi6x4 (estimate)
PUMPS = {
    'igus REBEL-EQU-SLZ-100 (Schmalz ECBPMi), on the tool': dict(
        curve=shaped(1.6, 60.0), trunk=TRUNK_TOOL, mass_kg=None, watts=7.2,
        src='Schmalz 10.03.01.00500: 1.6 L/min max, 60 % evacuation (MATCH datasheet); curve shape ASSUMED'),
    'Schmalz ECBPi 12 (on the stand)': dict(
        curve=shaped(12.0, 75.0), trunk=TRUNK_STAND, mass_kg=0.775, watts=13.0,
        src='Schmalz 10.03.01.00314 (M12-8; 00364 is the TB-8 variant, same data): 12 L/min, 750 mbar; curve shape ASSUMED'),
    '1 x Thomas 1520VDP BLDC HP (on the stand)': dict(
        curve=shaped(16.0, 80.0), trunk=TRUNK_STAND, mass_kg=0.42, watts=31.0,
        src='Thomas datasheet: 15/16 L/min, 80 %; curve shape ASSUMED'),
    '2 x Thomas 1520VDP BLDC HP (on the stand)': dict(
        curve=[(p, 2 * q) for p, q in shaped(16.0, 80.0)], trunk=TRUNK_STAND, mass_kg=0.84, watts=62.0,
        src='two of the above in parallel'),
    'KNF N 838 KN.29DC-B (on the stand)': dict(
        curve=KNF, trunk=TRUNK_STAND, mass_kg=1.9, watts=55.0, src='KNF datasheet curve (2024-10)'),
}

def pump_flow(curve, pmbar):
    if pmbar <= curve[0][0]: return 0.0
    if pmbar >= curve[-1][0]: return curve[-1][1]
    for (p0, q0), (p1, q1) in zip(curve, curve[1:]):
        if p0 <= pmbar <= p1: return q0 + (q1 - q0) * (pmbar - p0) / (p1 - p0)

def tube_out(Q, p_in, secs, n=60):
    """abs pressure [mbar] at the pump end of the trunk, flow Q std L/min leaving the collector at p_in (same model as suction_tool/verify.py 5c)."""
    mdot = Q / 60000 * RHO_STD
    p = p_in * 100.0
    for L, dmm in secs:
        d = dmm / 1000; A = math.pi * d * d / 4; dx = L / 1000 / n
        Re = 4 * mdot / (math.pi * d * MU)
        f = 0.0 if Re <= 0 else (64 / Re if Re < 2300 else 0.3164 * Re ** -0.25)
        for _ in range(n):
            rho = max(p, 1.0) / (R_AIR * T_K); u = mdot / (rho * A)
            p = max(p - f * dx / d * rho * u * u / 2, 1.0)
    return p / 100.0

# ------------------------------------------------------------------ pad leakage
# q_leak = leak of ONE pad when the flow is choked (vacuum >= ~50 kPa); ISO 6358 style, b = 0.5 (as in verify.py 5c).
def leak(q, vac):
    r = (101.3 - vac) / 101.3
    return q if r <= 0.5 else q * math.sqrt(max(0.0, 1 - ((r - 0.5) / 0.5) ** 2))

def steady_vac(pump, q_pad, n_pads=3):
    """pad vacuum [kPa] where the pump flow (after trunk friction) equals the pad leakage."""
    lo, hi = 0.0, 99.0
    for _ in range(50):
        v = 0.5 * (lo + hi); Q = n_pads * leak(q_pad, v)
        if pump_flow(pump['curve'], tube_out(Q, P_ATM - 10 * v, pump['trunk'])) > Q: lo = v
        else: hi = v
    return 0.5 * (lo + hi)

# ------------------------------------------------------------------ what the pads must hold
A_PAD = C.PAD['area_cm2'] * 1e-4          # m2 per pad (1.16 cm2; SMC: 6.9 N at -60 kPa)
M_BOX = C.BOX_MASS_MAX                    # 0.15 kg ASSUMED, not weighed
MU_FLOOR, MU_SIDE = 0.5, 0.4              # cardboard on shelf floor / cardboard on cardboard (ASSUMED, upper values)
ACC = 1.0                                 # m/s2 pull acceleration (ASSUMED)
H_PAD = 0.5 * min(b['h'] for b in C.BOXES.values()) / 1000   # pad row centred on the spine -> floor friction lever
PAD_SPAN = 0.120                          # outer pads at +-60 mm along the spine
SF = 2.0                                  # safety factor on the most loaded pad (tensile pull, no shear)
PACKING = {   # normal force pressing on EACH side face of the target carton from its neighbours (ASSUMED - to measure)
    'loose (0 N per side)': 0.0,
    'packed (1 N per side)': 1.0,
    'tight (3 N per side)': 3.0,
    'jammed (6 N per side)': 6.0,
}

def required(n_side):
    f_floor = MU_FLOOR * M_BOX * G
    f_pull = f_floor + 2 * MU_SIDE * n_side + M_BOX * ACC
    worst_pad = f_pull / 3 + f_floor * H_PAD / PAD_SPAN      # floor friction acts at the shelf floor -> moment -> outer pad
    vac = SF * worst_pad / A_PAD / 1000                       # kPa
    return dict(pull_force_n=f_pull, worst_pad_n=worst_pad, required_pad_vacuum_kpa=vac)

REQ = {k: required(v) for k, v in PACKING.items()}

# ------------------------------------------------------------------ transient (time to reach the required vacuum)
V_TOOL_CM3 = 3 * (0.35 + 0.37) + 1.0 + 1.0                 # pad cavities + bores + tubes + collector (verify.py 5c values)
def evac_time(pump, q_pad, target, dt=0.0005, tmax=5.0):
    V = V_TOOL_CM3 + sum(math.pi * (d / 2) ** 2 * L / 1000 for L, d in pump['trunk']) + 10.0   # + filter / switch 10 cm3
    p, t = P_ATM, 0.0
    while t < tmax:
        lo, hi = 0.0, 40.0
        for _ in range(30):
            q = 0.5 * (lo + hi)
            if pump_flow(pump['curve'], tube_out(q, p, pump['trunk'])) > q: lo = q
            else: hi = q
        p -= (lo - 3 * leak(q_pad, (P_ATM - p) / 10)) / 60 * 1000 * P_STD / V * dt
        t += dt
        if (P_ATM - p) / 10 >= target: return t
    return None

# ------------------------------------------------------------------ run
leaks = np.round(np.arange(0.0, 6.01, 0.05), 3)
curves = {name: [steady_vac(pp, q) for q in leaks] for name, pp in PUMPS.items()}

def max_leak(pump, vreq):
    lo, hi = 0.0, 20.0
    if steady_vac(pump, 0.0) < vreq: return 0.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if steady_vac(pump, m) >= vreq: lo = m
        else: hi = m
    return lo

table = {}
for name, pp in PUMPS.items():
    table[name] = {}
    for k, r in REQ.items():
        ml = max_leak(pp, r['required_pad_vacuum_kpa'])
        table[name][k] = dict(max_leak_per_pad_lpm=round(ml, 2),
                              evac_s_at_half_that_leak=(lambda t: None if t is None else round(t, 3))(
                                  evac_time(pp, 0.5 * ml, r['required_pad_vacuum_kpa'])) if ml > 0 else None)
    table[name]['_vacuum_at_leak_kpa'] = {str(q): round(curves[name][int(round(q / 0.05))], 1) for q in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)}

out = dict(
    question='can a small pump hold the carton during the 100 mm pull?',
    required=REQ, pumps={n: dict(src=p['src'], trunk_mm=p['trunk'], mass_kg=p['mass_kg'], watts=p['watts'], curve_mbar_lpm=p['curve']) for n, p in PUMPS.items()},
    results=table,
    assumptions=[
        'carton mass %.2f kg (ASSUMED max, not weighed); floor mu %.1f, side mu %.1f, pull acceleration %.1f m/s2 (ASSUMED)' % (M_BOX, MU_FLOOR, MU_SIDE, ACC),
        'neighbour side force per face 0/1/3/6 N for loose/packed/tight/jammed - NOT measured; this dominates the pull force',
        'most loaded pad = pull/3 + floor-friction moment (lever %.0f mm, pad span %.0f mm); safety factor %.1f; pad area %.2f cm2' % (H_PAD * 1000, PAD_SPAN * 1000, SF, C.PAD['area_cm2']),
        'leak per pad given as the choked leak flow (reached at >= ~50 kPa), ISO 6358 b=0.5 - same model as suction_tool/verify.py 5c',
        'pump curves: only the KNF curve is published; others use the KNF shape scaled to their free flow and max vacuum',
        'trunk: on-tool pump 150 mm phi4, stand pumps 1.6 m phi4 bore; filter/valve losses not included',
        'buffer spring force (1-3 N per pad) ignored; pad lip peel on a 12-16 mm spine not modelled',
    ])
json.dump(out, open(os.path.join(HERE, 'data', 'pump_sim.json'), 'w'), indent=1)

# ------------------------------------------------------------------ chart: pad vacuum vs leak per pad, required levels as reference lines
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e4e2dc'
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
short = ['igus SLZ-100 (on tool)', 'Schmalz ECBPi 12', '1 x Thomas 1520', '2 x Thomas 1520', 'KNF N838']
for (name, ys), c, s in zip(curves.items(), COL, short):
    ax.plot(leaks, ys, color=c, lw=2.2, label=s)
for k, r in REQ.items():
    v = r['required_pad_vacuum_kpa']
    ax.axhline(v, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(6.0, v + 0.8, 'needed: %s - %.0f kPa' % (k.split(' (')[0], v), ha='right', va='bottom', fontsize=9, color=INK2)
ax.set_xlim(0, 6); ax.set_ylim(0, 85)
ax.set_xlabel('Air leaking past each pad (L/min)  - unknown, to be measured', color=INK, fontsize=11)
ax.set_ylabel('Vacuum at the pads (kPa below atmosphere)', color=INK, fontsize=11)
ax.set_title('Vacuum the pump can hold vs leakage, 3 x ZP2 pads', loc='left', color=INK, fontsize=13)
ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.86), frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'renders', 'vacuum_vs_leak.png'))

print('required pad vacuum:', {k: round(r['required_pad_vacuum_kpa'], 1) for k, r in REQ.items()},
      'pull force N:', {k: round(r['pull_force_n'], 2) for k, r in REQ.items()})
for name in PUMPS:
    print('%-55s' % name, {k: table[name][k]['max_leak_per_pad_lpm'] for k in REQ}, table[name]['_vacuum_at_leak_kpa'])
