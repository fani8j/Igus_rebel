# Pump sizing simulation: can a small pump hold the carton?

`python3 vacuum_sim/pump_sim.py` -> `data/pump_sim.json`, `renders/vacuum_vs_leak.png` (3 s, numpy + matplotlib).

Steady-state vacuum at the 3 ZP2 pads for each pump vs leakage per pad (same pump/trunk/leak model as
`suction_tool/verify.py` 5c), compared with the pad vacuum needed to pull the carton out (pull force = floor friction +
neighbour side friction + m*a; most loaded pad incl. the floor-friction moment; safety factor 2).

Largest leakage per pad each pump tolerates (L/min):

| Pump | loose 17 kPa | packed 21 kPa | tight 31 kPa | jammed 44 kPa |
|---|---|---|---|---|
| igus REBEL-EQU-SLZ-100 (ECBPMi, on the tool) | 0.50 | 0.40 | 0.25 | 0.11 |
| Schmalz ECBPi 12 (stand) | 3.98 | 3.30 | 2.37 | 1.36 |
| 1 x Thomas 1520VDP BLDC HP (stand) | 5.31 | 4.44 | 3.23 | 2.01 |
| 2 x Thomas 1520VDP BLDC HP (stand) | 9.52 | 7.97 | 5.81 | 3.60 |
| KNF N 838 KN.29DC-B (stand) | 10.43 | 8.82 | 6.63 | 4.42 |

The two unmeasured inputs decide the answer: leakage per pad on a real carton, and the neighbour side force
(pull-out force). Both are cheap bench measurements (BENCH_TEST_PLAN.md test 0 + a spring-scale pull test).
All assumptions are listed in `data/pump_sim.json`.
