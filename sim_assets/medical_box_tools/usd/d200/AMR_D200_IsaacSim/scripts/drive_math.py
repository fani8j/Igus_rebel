"""SI differential-drive math. Independent of Isaac Sim; wheel order is LEFT, RIGHT."""
from __future__ import annotations
import math

def twist_to_wheels(v: float, yaw_rate: float, radius: float, track: float) -> tuple[float,float]:
    vals=(v,yaw_rate,radius,track)
    if not all(math.isfinite(float(x)) for x in vals):raise ValueError('Commands and geometry must be finite')
    if radius<=0 or track<=0:raise ValueError('Wheel radius and track must be positive')
    return ((v-0.5*track*yaw_rate)/radius,(v+0.5*track*yaw_rate)/radius)

def wheels_to_twist(left: float,right: float,radius: float,track: float) -> tuple[float,float]:
    if radius<=0 or track<=0:raise ValueError('Wheel radius and track must be positive')
    return (radius*(left+right)/2,radius*(right-left)/track)

def slew(current: float,target: float,rate: float,dt: float) -> float:
    if not all(math.isfinite(float(x)) for x in (current,target,rate,dt)):raise ValueError('Non-finite slew input')
    if rate<0 or dt<0:raise ValueError('Rate and dt must be non-negative')
    delta=rate*dt
    return current+max(-delta,min(delta,target-current))

def self_test() -> dict:
    r=.0762;b=.586;cases=[]
    for v,w in [(0.,0.),(.2,0.),(-.2,0.),(0.,.6),(0.,-.6),(.3,.4)]:
        l,rr=twist_to_wheels(v,w,r,b);back=wheels_to_twist(l,rr,r,b)
        assert abs(back[0]-v)<1e-12 and abs(back[1]-w)<1e-12
        cases.append({'v_m_s':v,'yaw_rad_s':w,'left_rad_s':l,'right_rad_s':rr,
                      'left_usd_deg_s':math.degrees(l),'right_usd_deg_s':math.degrees(rr)})
    assert twist_to_wheels(0,.6,r,b)[0]<0<twist_to_wheels(0,.6,r,b)[1]
    assert abs(slew(0,1,.35,1/60)-.35/60)<1e-15
    for bad in [(0,0,0,b),(0,0,r,0),(float('nan'),0,r,b)]:
        try:twist_to_wheels(*bad)
        except ValueError:pass
        else:raise AssertionError('Invalid inputs not rejected')
    return {'status':'PASS_MATH_ONLY','cases':cases,'physical_motion_tested':False}

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
