# Wing-only level flight (lift rotors at zero): least-squares equilibrium -> pitch-moment shortfall vs speed.
# Usage (repo root): python3 examples/rl/scripts/shuttle_glider/analysis/wing_only_trim.py
import sys, math, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'examples/rl/diagnostics'); import shuttle_glider_trim as s, numpy as np
from scipy.optimize import least_squares
veh=s.load_vehicle()
for V in (12,14,16,20,25):
  best=None
  for a0 in (0.05,0.1,0.2,0.3):
    f=lambda z: s.residual(np.r_[z[0],0,0,0,0,0,0,z[1],z[2],0,0],veh,V,0.0,0.0,True)[[0,2,4]]
    lo=[math.radians(-10),0,veh['de_lim'][0]]; hi=[math.radians(25),veh['T_max_p'],veh['de_lim'][1]]
    r=least_squares(f,[a0,3,0],bounds=(lo,hi))
    if best is None or r.cost<best.cost: best=r
  z=best.x; res=s.residual(np.r_[z[0],0,0,0,0,0,0,z[1],z[2],0,0],veh,V,0.0,0.0,True)
  print(f'V {V}: alpha {math.degrees(z[0]):5.1f} Tp {z[1]:5.2f} elev {math.degrees(z[2]):6.1f} deg  residual Fx {res[0]:+.2f} N Fz {res[2]:+.2f} N My {res[4]:+.3f} Nm')
