# Full-travel surface moment vs lift-rotor moment (authority table); rotor-off trim attempt.
import sys, math, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'examples/rl/diagnostics'); import shuttle_glider_trim as s, numpy as np
veh=s.load_vehicle(); C,G=s.C,s.G
print('r_rotor',np.round(veh['r_rotor'],3).tolist(),'rot_dir',veh['rot_dir'],'r_com_aero',np.round(veh['r_com_aero'],3),'lims',veh['de_lim'],veh['da_lim'],veh['dr_lim'])
print('Cmde Clda Cndr',C.Cmde,C.Clda,C.Cndr,'S b c',G.S,G.b,G.c)
# pure-wing trim: lift rotors forced off
import copy
for V in (12,13,14,15,16,18):
    v2=copy.deepcopy(veh); v2['T_max']=np.full(4,1e-4)
    for cost in (25.0,):
        t=s.solve_trim(v2,V,surfaces=True,allocation='regularised',reserve=0.0,alpha_bounds=(-10,25),surface_cost=cost)
        t0=s.solve_limits_trim(veh,V)
        print(V, 'limits trim P=%.1f W (front %.2f)'%(t0['P'],t0['T'][0]/veh['T_max'][0]), '| rotors off:', None if t is None else 'P=%.1f W alpha %.1f elev %.1f deg'%(t['P'],math.degrees(t['alpha']),math.degrees(t['de'])))
# moment authority: full-travel surface moment vs lift-rotor moment of +-5% Tmax on every rotor (10% Tmax between the
# two halves acting on each axis: front/rear, left/right, the two directions of rotation; total thrust unchanged)
for V in (0,4,8,12,14):
    qS=0.5*G.rho*V**2*G.S
    print(f'V {V:2d}: full elev {abs(qS*G.c*C.Cmde*veh["de_lim"][1]):.2f} Nm  full ail {abs(qS*G.b*C.Clda*veh["da_lim"][1]):.2f}  full rud {abs(qS*G.b*C.Cndr*veh["dr_lim"][1]):.2f}')
T=veh['T_max'][0]; r=np.array(veh['r_rotor']); d=np.array(veh['rot_dir'])
dT=lambda halves: 0.05*T*np.where(halves,1.0,-1.0)
pitch=abs(-(dT(r[:,0]>0)*r[:,0]).sum()); roll=abs((dT(r[:,1]>0)*r[:,1]).sum())
yaw=abs((dT(d>0)*d*veh['km']/veh['kf']).sum())
print('rotor moment per 10%% Tmax diff (+-5%% on every rotor): pitch %.2f Nm, roll %.2f Nm, yaw (drag torque) %.3f Nm'%(pitch,roll,yaw))
