# Mission phases vs minimum-power trim: lift thrust per rotor, puller, power, wing lift, alpha, surfaces.
# Usage (repo root): python3 examples/rl/scripts/shuttle_glider/analysis/mission_vs_trim.py <variant>/mission/mission_v14_s03
import csv, json, sys, math
import numpy as np
sys.path.insert(0,'examples/rl/diagnostics'); import shuttle_glider_trim as s
veh=s.load_vehicle(); W=veh['m']*9.81
d=sys.argv[1]
tr=list(csv.DictReader(open(d+'/trajectories.csv'))); 
A=lambda k: np.array([float(r[k]) for r in tr])
t=A('t'); V=A('speed'); w=np.stack([A(f'actual_w_{i}') for i in range(5)],1)
wmax=np.r_[veh['w_max'] if np.ndim(veh['w_max']) else [veh['w_max']]*4, veh['w_max_p']] if False else None
T=veh['kf']*w[:,:4]**2; Tp=veh['kf_p']*w[:,4]**2
P=(veh['km']*w[:,:4]**3).sum(1)+veh['km_p']*w[:,4]**3
pitch=A('pitch_deg'); vz=A('vz'); gam=np.degrees(np.arctan2(vz,np.maximum(np.hypot(A('vx'),A('vy')),.1)))
fz=A('aero_fz_body'); el,ai,ru=A('elevator_deg'),A('aileron_deg'),A('rudder_deg'); roll=A('roll_deg')
print(f"{'phase':14s} {'V':>5s} | T0..T3/Tmax (F:0,2 R:1,3)  | Tp[N] | P[W] Ptrim | -Fz_aero/W trim | alpha~ trim | elev trim | ail rud | roll")
for p in json.load(open(d+'/mission_phases.json')):
    for sub in ([p] if p['phase']!='deceleration' else [dict(p,phase=f'decel@{v}',vsel=v) for v in (12,10,8,6,4,2)]):
        if 'vsel' in sub: sel=(t>=p['t_start'])&(t<p['t_end'])&(np.abs(V-sub['vsel'])<0.25)
        else: sel=(t>=p['t_start'])&(t<p['t_end'])
        if p['phase']=='hover': sel&=t>1
        if not sel.any(): continue
        v=V[sel].mean(); tm=s.solve_limits_trim(veh,round(v*2)/2)
        Tm=T[sel].mean(0)/veh['T_max']
        print(f"{sub['phase']:14s} {v:5.1f} | "+' '.join(f'{x:.2f}' for x in Tm)+f" ({(tm['T'][0]+tm['T'][2])/2/veh['T_max'][0]:.2f}/{(tm['T'][1]+tm['T'][3])/2/veh['T_max'][0]:.2f})"
              f" | {Tp[sel].mean():5.2f} ({tm['Tp']:4.1f}) | {P[sel].mean():5.0f} {tm['P']:5.0f} | {(-fz[sel]).mean()/W:5.2f} {tm['offload']:4.2f} | {(pitch-gam)[sel].mean():5.1f} {math.degrees(tm['alpha']):5.1f} | {el[sel].mean():5.1f} {math.degrees(tm['de']):5.1f} | {ai[sel].mean():5.1f} {ru[sel].mean():5.1f} | {roll[sel].mean():5.1f}")
