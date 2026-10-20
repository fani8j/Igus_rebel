#!/usr/bin/env python3
"""Highlight reel + stills from the recorded cabinet grasp videos (results/videos/*.mp4, ffmpeg 4.4).
Reel = one level of each variant (top-down T2 230 mm, front T2, top-down top shelf T7, thin box front T4) + every failed run (3/4 views) -> results/videos/highlights.mp4 (or the external drive if > 10 MB).
Stills = one frame (the 'extracted' moment, 70 % of the clip) per base target and mode, 3/4 view -> results/stills/."""
import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent / 'results'
V = R / 'videos'
EXT = Path('/media/farhan/My Passport/igus_nvidia/scratch_usd/videos')
FF = '/usr/bin/ffmpeg'
sel = ['topdown_T2_3q', 'front_T2_3q', 'topdown_T7_3q', 'front_T4_3q']
fails = []
for m in ('topdown', 'front'):
    f = R / f'cabinet_grasp_{m}.json'
    if f.exists():
        for r in json.load(open(f))['runs']:
            if 'error' not in r and not r.get('success'):
                fails.append(f"{m}_{r['target']}{'_' + r['variant'] if r['variant'] != 'base' else ''}_3q")
clips = [c for c in dict.fromkeys(sel + fails) if (V / f'{c}.mp4').exists()]
lst = R / 'reel_list.txt'
lst.write_text(''.join(f"file '{V / (c + '.mp4')}'\n" for c in clips))
out = V / 'highlights.mp4'
subprocess.run([FF, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', str(out)], check=True)
lst.unlink()
if out.stat().st_size > 10e6:
    EXT.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.move(str(out), str(EXT / 'highlights.mp4'))
    out = EXT / 'highlights.mp4'
print('reel', out, round(out.stat().st_size / 1e6, 1), 'MB', clips)
(R / 'stills').mkdir(exist_ok=True)
for m in ('topdown', 'front'):
    for mp4 in sorted(V.glob(f'{m}_*_3q.mp4')):
        name = mp4.stem
        if len(name.split('_')) != 3:
            continue                                     # base targets only
        dur = float(subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(mp4)], capture_output=True, text=True).stdout)
        subprocess.run([FF, '-y', '-loglevel', 'error', '-ss', f'{0.70 * dur:.2f}', '-i', str(mp4), '-frames:v', '1', '-vf', 'scale=960:-1', '-q:v', '4', str(R / 'stills' / f'{name}.jpg')], check=True)
print('stills', len(list((R / 'stills').glob('*.jpg'))))
