import json, numpy as np
W = "/media/farhan/My Passport/igus_nvidia/shelf_pick_research/robot_stand"
STEP = "/home/farhan/Downloads/2 arms step(軌).step"

# top-level component (path[1]) -> (group, description, moving?, colour)
GROUPS = {
 'NAUO1':  None,  # split per leaf below
 'NAUO2':  ('rail_support', 'support frame for slide rail (on ZLW carriage)', 'axis'),
 'NAUO3':  ('rail_x', 'WW-16-120-20 carriage (arm A)', 'arm'),
 'NAUO4':  ('rail_x', 'WHKD-1620 clamp (arm A)', 'arm'),
 'NAUO5':  ('rail_x', 'WS-16-120-660 X rail 660 mm', 'axis'),
 'NAUO6':  ('rail_x', 'WW-16-120-20 carriage (arm B)', 'arm'),
 'NAUO7':  ('rail_x', 'WHKD-1620 clamp (arm B)', 'arm'),
 'NAUO8':  ('plates', 'upper plate (arm A)', 'arm'),
 'NAUO9':  ('plates', 'upper plate (arm B)', 'arm'),
 'NAUO10': ('arm_A', 'ReBeL arm A (x ~ -424)', 'arm'),
 'NAUO11': ('arm_B', 'ReBeL arm B (x ~ -884)', 'arm'),
 'NAUO12': ('gusset', 'triangular support (fixed)', 'fixed'),
 'NAUO13': ('side_plate', 'triangular side plate', 'fixed'),
 'NAUO14': ('side_plate', 'triangular side plate', 'fixed'),
 'NAUO15': ('plates', 'carrier plate for dual arms (A)', 'arm'),
 'NAUO16': ('plates', 'carrier plate for dual arms (B)', 'arm'),
}
COLORS = {'zlw_body':(0.35,0.55,0.85),'zlw_carriage':(0.95,0.55,0.15),'rail_support':(0.55,0.8,0.4),
 'rail_x':(0.2,0.2,0.2),'plates':(0.8,0.7,0.2),'arm_A':(0.85,0.25,0.25),'arm_B':(0.7,0.3,0.75),
 'gusset':(0.55,0.55,0.6),'side_plate':(0.3,0.75,0.75)}

def load_leaves():
    L = json.load(open(f"{W}/leaves.json"))
    for l in L:
        top = l['path'][1]
        if top == 'NAUO1':
            l['group'] = 'zlw_carriage' if l['name'].endswith('0290_2') else 'zlw_body'
            l['moving'] = 'axis' if l['group']=='zlw_carriage' else 'fixed'
            l['desc'] = 'ZLW-20120S belt axis ' + ('carriage' if l['group']=='zlw_carriage' else 'body (1250 stroke)')
        else:
            g = GROUPS[top]; l['group'], l['desc'], l['moving'] = g
        l['top'] = top
    return L

def load_mesh(idx):
    d = np.load(f"{W}/mesh/leaf_{idx:05d}.npz")
    return d['v'].astype(float), d['f']
