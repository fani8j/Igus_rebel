#!/usr/bin/env python3
"""Launch with Isaac Sim's python.sh, not system Python.
./python.sh /absolute/path/AMR_D200_IsaacSim/scripts/run_amr.py --mode keyboard
./python.sh /absolute/path/AMR_D200_IsaacSim/scripts/run_amr.py --mode check --headless
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description='D200 USD physics demo / keyboard control / smoke test')
parser.add_argument('--mode',choices=['keyboard','demo','constant','check','validate'],default='keyboard')
parser.add_argument('--navigation',action='store_true',help='Use simplified frictionless caster supports (not detailed caster mechanics)')
parser.add_argument('--headless',action='store_true')
parser.add_argument('--linear',type=float,default=.2,help='Body forward speed m/s, constant mode')
parser.add_argument('--angular',type=float,default=0.,help='Body yaw rate rad/s, constant mode')
parser.add_argument('--seconds',type=float,default=None)
parser.add_argument('--output',type=Path,default=P/'reports'/'isaac_runtime_result.json')
args,kit_args=parser.parse_known_args()
if args.headless and args.mode=='keyboard':parser.error('Keyboard mode requires a GUI; use --mode check/demo/constant')
if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):parser.error('--seconds must be finite and positive')
if not all(math.isfinite(x) for x in (args.linear,args.angular)):parser.error('Commands must be finite')
# Pass only Kit flags through to the application.
sys.argv=[sys.argv[0],*kit_args]
try:
    from isaacsim import SimulationApp
except ImportError as exc:
    raise SystemExit('Run with an Isaac Sim Python environment providing isaacsim.SimulationApp (usually python.sh). System Python is not sufficient.') from exc
app=SimulationApp({'headless':args.headless})
status=0;drive=None;timeline=None
try:
    import omni.usd
    import omni.timeline
    from isaac_control import D200Drive,KeyboardCommand,configure_physics
    from validate_in_isaac import validate_stage
    scene=P/('AMR_D200_navigation_demo.usd' if args.navigation else 'AMR_D200_drive_demo.usd')
    context=omni.usd.get_context()
    result=context.open_stage(str(scene))
    if result is False:raise RuntimeError('Could not open '+str(scene))
    for _ in range(10):app.update()
    deadline=time.monotonic()+120
    while context.get_stage_loading_status()[2]>0:
        if time.monotonic()>deadline:raise TimeoutError('Stage loading timeout')
        app.update()
    configure_physics()
    stage=context.get_stage();validation=validate_stage(stage)
    if validation['status']!='PASS_OPENUSD_CHECK':raise RuntimeError(json.dumps(validation,indent=2))
    drive=D200Drive(stage)
    report={'status':'VALIDATED_NOT_SIMULATED','scene':scene.name,'openusd_validation':validation}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='validate':
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    else:
        if not args.headless:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                vp=get_active_viewport()
                if vp:vp.set_active_camera('/World/OverviewCamera')
            except (ImportError,AttributeError):pass
        keyboard=KeyboardCommand() if args.mode=='keyboard' else None
        timeline=omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.);timeline.set_looping(False)
        # Do not stop keyboard control at a short inherited timeline range.
        timeline.set_end_time(max(3600.,(args.seconds or 12.)+1.));timeline.play(looping=False)
        duration=12. if args.mode=='check' else (args.seconds if args.seconds else (float('inf') if keyboard else 12.))
        print('[D200] Wheel targets only; +X forward. Keyboard: W/S A/D, SPACE stop, ESC exit.')
        samples=[];last_t=0.;last_record=-1.;last_wall_progress=time.monotonic();last_progress_t=-1.
        def plan(t):
            if t<2.:return 0.,0.
            if t<5.:return .2,0.
            if t<5.8:return 0.,0.
            if t<8.3:return 0.,.6
            if t<9.:return 0.,0.
            if t<11.:return -.15,0.
            return 0.,0.
        while app.is_running():
            t=float(timeline.get_current_time())
            if t>=duration:break
            if t>last_progress_t+1e-7:last_wall_progress=time.monotonic();last_progress_t=t
            if args.headless and time.monotonic()-last_wall_progress>90:raise TimeoutError('Timeline not progressing')
            dt=max(0.,min(.1,t-last_t));last_t=t
            if timeline.is_playing():
                if keyboard:
                    v,w,stop,quit_=keyboard.read()
                    if quit_:break
                    if stop:drive.stop()
                    else:drive.set_twist(v,w,dt if dt else 1/60)
                else:
                    v,w=(args.linear,args.angular) if args.mode=='constant' else plan(t)
                    drive.set_twist(v,w,dt if dt else 1/60)
                if t-last_record>=.045:
                    sample={'time_s':t,**drive.pose(),'command_v_m_s':drive.v,'command_yaw_rad_s':drive.w,
                            'wheel_targets_rad_s':list(drive.last_wheel_rad_s)}
                    samples.append(sample);last_record=t
            else:drive.stop()
            app.update()
        drive.stop()
        if args.mode=='check':
            def near(t):return min(samples,key=lambda x:abs(x['time_s']-t))
            if not samples or samples[-1]['time_s']<11.8:raise RuntimeError('Smoke test interrupted before completion')
            a,b=near(2.),near(5.)
            dx=b['xyz_m'][0]-a['xyz_m'][0];dy=b['xyz_m'][1]-a['xyz_m'][1]
            forward=dx*math.cos(a['yaw_rad'])+dy*math.sin(a['yaw_rad'])
            c,d=near(5.8),near(8.3)
            yaw=math.atan2(math.sin(d['yaw_rad']-c['yaw_rad']),math.cos(d['yaw_rad']-c['yaw_rad']))
            settled=[s for s in samples if s['time_s']>2.]
            zmin=min(s['xyz_m'][2] for s in settled);zmax=max(s['xyz_m'][2] for s in settled)
            min_up=min(s['up_z'] for s in settled)
            e,f=near(9.),near(11.)
            reverse=(f['xyz_m'][0]-e['xyz_m'][0])*math.cos(e['yaw_rad'])+(f['xyz_m'][1]-e['xyz_m'][1])*math.sin(e['yaw_rad'])
            checks={'forward_displacement_above_0p25_m':forward>.25,
                    'positive_spin_above_0p45_rad':yaw>.45,
                    'reverse_displacement_below_minus_0p10_m':reverse<-.10,
                    'base_height_between_0p03_and_0p15_m':zmin>.03 and zmax<.15,
                    'upright_up_z_above_0p95':min_up>.95}
            status=0 if all(checks.values()) else 2
            report.update({'status':'PASS_PHYSICS_SMOKE_TEST' if status==0 else 'FAIL_PHYSICS_SMOKE_TEST',
                           'physical_motion_tested':True,'checks':checks,
                           'measurements':{'forward_m':forward,'spin_rad':yaw,'reverse_m':reverse,'base_z_range_m':[zmin,zmax],'minimum_up_z':min_up}})
        else:report.update({'status':'RUN_COMPLETED_NO_AUTOMATIC_ACCEPTANCE','physical_motion_tested':True})
        report['samples']=samples
        args.output.write_text(json.dumps(report,indent=2))
        print('[D200]',report['status'],'report:',args.output)
except Exception as exc:
    status=1
    report={'status':'ERROR','error':str(exc),'note':'This is not a successful physics test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    import traceback;traceback.print_exc()
finally:
    if drive:
        try:drive.stop()
        except Exception:pass
    if timeline:
        try:timeline.stop()
        except Exception:pass
    app.close()
sys.exit(status)
