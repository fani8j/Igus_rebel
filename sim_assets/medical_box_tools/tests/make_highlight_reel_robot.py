#!/usr/bin/env python3
"""Highlight reel (6 clips) + stills from the full-robot front-mode runs.  Videos live on the external drive (robot_<date>/), stills (jpg < 300 KB) in results/stills_robot/."""
import subprocess, sys
from pathlib import Path
V = Path('/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/robot_20261015')
R = Path(__file__).resolve().parent / 'results'
FF, FP = '/usr/bin/ffmpeg', '/usr/bin/ffprobe'
clips = [f'front_{t}_tiled' for t in ('T3', 'T4', 'T5', 'T6', 'T7', 'T8')]
clips = [c for c in clips if (V / f'{c}.mp4').exists()]
lst = V / 'reel_list.txt'
lst.write_text(''.join(f"file '{V / (c + '.mp4')}'\n" for c in clips))
out = V / 'highlights_robot.mp4'
subprocess.run([FF, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', str(out)], check=True)
lst.unlink()
print('reel', out, round(out.stat().st_size / 1e6, 1), 'MB', clips)
(R / 'stills_robot').mkdir(exist_ok=True)
for tgt in ('T5', 'T3', 'T8'):
    for view in ('3q', 'side', 'head', 'tiled'):
        mp4 = V / f'front_{tgt}_{view}.mp4'
        if not mp4.exists():
            continue
        dur = float(subprocess.run([FP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(mp4)], capture_output=True, text=True).stdout)
        w = 1280 if view == 'tiled' else 960
        subprocess.run([FF, '-y', '-loglevel', 'error', '-ss', f'{0.62 * dur:.2f}', '-i', str(mp4), '-frames:v', '1', '-vf', f'scale={w}:-1', '-q:v', '6', str(R / 'stills_robot' / f'front_{tgt}_{view}.jpg')], check=True)
for f in sorted((R / 'stills_robot').glob('*.jpg')):
    print(f.name, round(f.stat().st_size / 1024), 'KB')
