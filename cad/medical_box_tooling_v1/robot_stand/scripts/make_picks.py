"""Pick the candidates shown in the overlay renders (indices refer to the order in data/candidates_targetA.json)."""
import json
d = json.load(open('data/candidates_targetA.json'))['results']
want = [('P1','CARRIAGE',3,'lift carriage: bracket pocket, upright (best)',(0.1,0.65,0.2),[-625,625]),
        ('P2','CARRIAGE',1,'lift carriage: under rail shelf (shortest tube)',(0.0,0.6,0.7),[-625,625]),
        ('P3','ARM_A',2,'on arm-A X-carriage, below rail',(0.95,0.5,0.0),[]),
        ('P4','FIXED',13,'column side (fixed), inside arm reach',(0.15,0.25,0.9),[]),
        ('P5','FIXED',0,'column base wing, front face (fixed)',(0.55,0.2,0.8),[]),
        ('P6','DECK',0,'AMR deck beside column (assumed deck)',(0.85,0.1,0.1),[])]
out = []
for pid, fam, i, label, rgb, ghost in want:
    r = d[fam][i]; out.append(dict(id=pid, family=fam, index=i, label=label, rgb=rgb, ghost_dy=ghost, lo=r['lo'], hi=r['hi'], metrics={k: r[k] for k in ('leg_to_target_base_mm','tube_total_worst_mm','tube_total_nominal_mm','min_clearance_exact_mm','near_solids','height_above_deck_mm','in_reach_target','outside_stand_envelope_mm')}))
json.dump(out, open('data/picks.json','w'), indent=1)
for o in out: print(o['id'], o['family'], o['lo'], o['hi'], o['metrics'])
