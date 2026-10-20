"""Component table (top-level instances) -> data/components.json, data/leaves.csv, printed markdown."""
import sys, json, csv, collections, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
L = load_leaves()
DEC = {'NAUO12':'三角支撐 (triangular support)','NAUO13':'三角形板側 (triangular side plate)','NAUO14':'三角形板側 (triangular side plate)',
       'NAUO2':'支撐架for滑軌 (support frame for slide rail)','NAUO8':'上版for... (upper plate)','NAUO9':'上版for... (upper plate)',
       'NAUO15':'載板for雙臂 (carrier plate for dual arms)','NAUO16':'載板for雙臂 (carrier plate for dual arms)'}
GUESS = {'NAUO1 (carriage)':'ZLW-20120S carriage (300 mm, lift carriage, travels in Y)','NAUO1 (body)':'ZLW_20120S_K0BW0_D0A4B_0A0A0_1250: toothed-belt axis body, 1746 mm along Y, 1250 mm stroke, drive end at y=-1333','NAUO1':'ZLW_20120S...1250 toothed-belt axis: body (leaf ..0322_2, 1746 mm long, along Y) + carriage (leaf ..0290_2, 300 mm)',
 'NAUO2':'rail bracket: U/channel with two triangular gussets, bolted on the ZLW carriage; carries the X rail','NAUO3':'WW-16-120-20 rail carriage (arm A)','NAUO4':'WHKD-1620 clamp/lock (arm A)',
 'NAUO5':'WS-16-120-660 drylin W rail, 660 mm, along X','NAUO6':'WW-16-120-20 rail carriage (arm B)','NAUO7':'WHKD-1620 clamp/lock (arm B)','NAUO8':'upper (mounting) plate on carriage A, arm base bolts to it','NAUO9':'upper (mounting) plate on carriage B',
 'NAUO10':'ReBeL arm A: 150 igus D00xxxxxx parts, straight pose along +Y','NAUO11':'ReBeL arm B: identical copy','NAUO12':'gusset/column: 100 mm thick plate, y -1351..-200, slanted underside; the only fixed structure',
 'NAUO13':'side wing plate (70 mm thick) on the gusset top at the rear end','NAUO14':'side wing plate, mirror','NAUO15':'3 mm plate 325x300 at the +X end of the rail (not attached to anything; probably stray/for later)','NAUO16':'3 mm plate at the -X end of the rail'}
tops = collections.OrderedDict()
for l in L:
    key = l['top'] if l['top']!='NAUO1' else ('NAUO1 (carriage)' if l['group']=='zlw_carriage' else 'NAUO1 (body)')
    tops.setdefault(key, []).append(l)
rows = []
for k, ls in tops.items():
    bb = [min(l['bbox'][i] for l in ls) if i<3 else max(l['bbox'][i] for l in ls) for i in range(6)]
    kk = k.split()[0]
    parts = collections.Counter(l['name'] for l in ls)
    rows.append(dict(component=k, name=DEC.get(kk, ls[0]['name']), instances=1, leaf_solids=len(ls), distinct_parts=len(parts),
        bbox_mm=[round(x,1) for x in bb], size_mm=[round(bb[i+3]-bb[i],1) for i in range(3)], volume_cm3=round(sum(abs(l['volume']) for l in ls)/1e3, 1), group=ls[0]['group'], guess=GUESS.get(k, GUESS.get(kk, ''))))
json.dump(rows, open('data/components.json','w'), indent=1, ensure_ascii=False)
with open('data/leaves.csv','w', newline='') as f:
    w = csv.writer(f); w.writerow(['idx','top','group','name','xmin','ymin','zmin','xmax','ymax','zmax','volume_cm3'])
    for l in L: w.writerow([l['idx'], l['top'], l['group'], l['name'], *[round(x,1) for x in l['bbox']], round(l['volume']/1e3,2)])
for r in rows:
    print(f"| {r['component']} | {r['name'] if r['name'].isascii() or kk else ''} | {r['leaf_solids']} | {r['bbox_mm'][:3]} .. {r['bbox_mm'][3:]} | {r['size_mm']} | {r['volume_cm3']} | {r['guess']} |")
allbb = [min(l['bbox'][i] for l in L) if i<3 else max(l['bbox'][i] for l in L) for i in range(6)]
print('ALL', [round(x) for x in allbb])
noplate = [l for l in L if l['top'] not in ('NAUO15','NAUO16')]
print('no carrier plates', [round(min(l['bbox'][i] for l in noplate)) if i<3 else round(max(l['bbox'][i] for l in noplate)) for i in range(6)])
