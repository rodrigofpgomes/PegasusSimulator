# Surface control moments vs lift-rotor moments per mission phase (mean and std; opposite signs = conflict).
# Usage (repo root): python3 examples/rl/scripts/shuttle_glider/analysis/mission_moments.py <variant>/mission/mission_v12_s03
import csv, json, sys, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0,'examples/rl/diagnostics'); import shuttle_glider_trim as s
veh=s.load_vehicle(); C,G=s.C,s.G
d=sys.argv[1]; tr=list(csv.DictReader(open(d+'/trajectories.csv'))); A=lambda k: np.array([float(r[k]) for r in tr])
t=A('t'); V=A('speed'); w=np.stack([A(f'actual_w_{i}') for i in range(4)],1); T=veh['kf']*w**2
qS=0.5*G.rho*V**2*G.S
de,da,dr=[np.radians(A(k)) for k in ('elevator_deg','aileron_deg','rudder_deg')]
Ms=np.stack([qS*G.b*C.Clda*da, qS*G.c*C.Cmde*de, qS*G.b*C.Cndr*dr],1)
r=np.array(veh['r_rotor']); Mr=np.zeros((len(t),3))
for i in range(4):
    Mr+=np.cross(r[i],np.stack([0*T[:,i],0*T[:,i],T[:,i]],1)); Mr[:,2]+=veh['rot_dir'][i]*veh['km'][i]/veh['kf'][i]*T[:,i]
# remove the static weight-balance part? report mean and fluctuating (std) parts separately
print(f"{'phase':14s}  V   | surface moment mean [Nm] roll pitch yaw | rotor moment mean roll pitch yaw | std surf vs std rotor (r,p,y)")
for p in json.load(open(d+'/mission_phases.json')):
    sel=(t>=p['t_start'])&(t<p['t_end'])&(t>1)
    ms,mr=Ms[sel],Mr[sel]
    print(f"{p['phase']:14s} {V[sel].mean():4.1f} | "+' '.join(f'{x:+.2f}' for x in ms.mean(0))+'  | '+' '.join(f'{x:+.2f}' for x in mr.mean(0))+'  | '+' '.join(f'{a:.3f}/{b:.3f}' for a,b in zip(ms.std(0),mr.std(0))))
