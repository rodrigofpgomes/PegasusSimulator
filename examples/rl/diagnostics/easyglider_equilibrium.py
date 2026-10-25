"""
| File: easyglider_equilibrium.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Longitudinal equilibrium of the stand-alone EasyGlider (level, glide and powered flight) with the batched aerodynamic model.
|
| Equilibrium table of the EasyGlider appendix. Run with the Isaac Sim interpreter.
"""
import sys, math
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tests' / 'review'))
from offline_runtime import enable; enable()
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch, GliderCoefficients, GliderGeometry, EASYGLIDER_REFERENCE_FROM_COM, DEG
C, G = GliderCoefficients(), GliderGeometry()
M_KG, G0 = 1.495, 9.81
IYY = 0.14675557
R_REF = np.array(EASYGLIDER_REFERENCE_FROM_COM)
R_PULL = np.array([0.342692, 0.0, -0.020454])
KF, KM, WMAX = 8.54858e-6, 1.0258e-7, 3500.0
TMAX = KF*WMAX**2
DE_LIM = math.radians(13.61); DA_LIM=(math.radians(-8.85), math.radians(22.62)); DR_LIM=math.radians(12.74)
W = M_KG*G0

def aero(u, w, q, de, da=0.0, dr=0.0, p=0.0, r=0.0, v=0.0):
    """Force/moment (FLU, about CoM) of the batched model, float64. state: body velocity (u,v,w), rates (p,q,r)."""
    vb = np.array([u, v, w]); om = np.array([p, q, r])
    va = vb + np.cross(om, R_REF)
    Va = np.linalg.norm(va); Vs = max(Va, 0.5)
    alpha = max(-math.pi/2, min(math.pi/2, math.atan2(-va[2], va[0])))
    beta = -math.asin(max(-1., min(1., va[1]/Vs)))
    qbar = 0.5*G.rho*Va**2; qS = qbar*G.S
    ph, qh, rh = p*G.b/(2*Vs), q*G.c/(2*Vs), r*G.b/(2*Vs)
    x1 = max(-80., min(80., -C.M_sig*(alpha-C.alpha0))); x2 = max(-80., min(80., C.M_sig*(alpha+C.alpha0)))
    e1, e2 = math.exp(x1), math.exp(x2); sig = (1+e1+e2)/((1+e1)*(1+e2))
    CLl = C.CL0 + C.CLa*alpha
    CLf = 2*math.copysign(1, alpha)*math.sin(alpha)**2*math.cos(alpha) if alpha != 0 else 0.0
    CL = (1-sig)*CLl + sig*CLf
    CD = C.CD0 + CLl**2/(math.pi*G.e*G.AR) + C.CDde*abs(de)
    lift = qS*(CL + C.CLq*qh + C.CLde*de); drag = max(qS*(CD + C.CDq*qh), 0.0)
    ca, sa = math.cos(alpha), math.sin(alpha)
    Fx = -drag*ca + lift*sa; Fz = drag*sa + lift*ca
    Fy = qS*(C.CY0 + C.CYb*beta + C.CYp*ph + C.CYr*rh + C.CYda*da + C.CYdr*dr)
    tx = qS*G.b*(C.Cl0 + C.Clb*beta + C.Clp*ph + C.Clr*rh + C.Clda*da + C.Cldr*dr)
    ty = qS*G.c*(C.Cm0 + C.Cma*alpha + C.Cmq*qh + C.Cmde*de)
    tz = qS*G.b*(C.Cn0 + C.Cnb*beta + C.Cnp*ph + C.Cnr*rh + C.Cnda*da + C.Cndr*dr)
    F = np.array([Fx, Fy, Fz]); tau = np.array([tx, ty, tz]) + np.cross(R_REF, F)
    return F, tau, alpha, Va

def _check_vs_torch():
    A = GliderAerodynamicsBatch({}, 1, 'cpu')
    for (u, w, q, de) in [(15., -0.5, 0.0, 0.1), (12., -1.5, -0.4, -0.2), (20., 0.3, 0.8, 0.05)]:
        Ft, Tt = A.forces_moments(torch.tensor([[u, 0., w]]), torch.tensor([[0., q, 0.]]), torch.tensor([[de, 0., 0.]]))
        F, T, *_ = aero(u, w, q, de)
        assert np.allclose(F, Ft[0].numpy(), atol=5e-4) and np.allclose(T, Tt[0].numpy(), atol=5e-4), (F, Ft, T, Tt)

def dyn(x, de, T, da=0.0, s=1.0):
    """x=(u,w,q,theta) -> xdot. Thrust along +x at the puller position, gravity in body axes."""
    u, w, q, th = x
    F, tau, _, _ = aero(u, w, q, de, da)
    Fx, Fz = F[0] + T, F[2]
    My = tau[1] + R_PULL[2]*T          # r x F, y component with F=(T,0,0)
    udot = (Fx - W*math.sin(th))/M_KG - q*w
    wdot = (Fz - s*W*math.cos(th))/M_KG + q*u   # s=+1 upright, s=-1 inverted (roll 180 deg)
    return np.array([udot, wdot, My/IYY, -q])   # FLU: q about +y (left); nose-up pitch rate = -q

_check_vs_torch()
from scipy.optimize import fsolve

deg=math.degrees

def eqs_given_gamma(z, V, gam):
    al, de, T = z
    u, w = V*math.cos(al), -V*math.sin(al); th = al + gam
    d = dyn(np.array([u, w, 0.0, th]), de, T)
    return [d[0]*M_KG, d[1]*M_KG, d[2]*IYY]

def solve(V, gam, guess=(0.0, 0.0, 1.0)):
    best=None
    for a0 in (-0.1, 0.0, 0.1, 0.3):
        for d0 in (-0.2, 0.0, 0.3, 0.5):
            sol, info, ier, _ = fsolve(eqs_given_gamma, (a0, d0, 1.0), args=(V, gam), full_output=True)
            if ier == 1 and max(abs(np.array(eqs_given_gamma(sol,V,gam)))) < 1e-6 and abs(sol[0]) < 0.6:
                best = sol; break
        if best is not None: break
    return best

def solve_glide(V):
    """T = 0: unknown alpha, delta_e, gamma."""
    def f(z):
        al, de, gam = z
        d = dyn(np.array([V*math.cos(al), -V*math.sin(al), 0.0, al+gam]), de, 0.0)
        return [d[0]*M_KG, d[1]*M_KG, d[2]*IYY]
    for a0 in (-0.1, 0.0, 0.1, 0.2):
        for d0 in (0.0, 0.3, 0.5, -0.2):
            for g0 in (-0.1, -0.3):
                sol, info, ier, _ = fsolve(f, (a0, d0, g0), full_output=True)
                if ier == 1 and max(abs(np.array(f(sol)))) < 1e-6: return sol
    return None

def aileron_for_reaction(T, V, alpha):
    om2 = T/KF; reaction = KM*om2           # +x torque, rot_dir=+1
    qS = 0.5*G.rho*V**2*G.S
    da = -reaction/(qS*G.b*C.Clda)
    return da, reaction

print("LEVEL FLIGHT (gamma=0): unknown alpha, delta_e, T")
print("  V   alpha[deg] de[deg]  T[N]   w_p[rad/s]  da[deg]   de_in_limits  T_in_limits  da_in_limits")
for V in (6,8,10,12,14,16,18,20,24,28,32,40,50):
    s=solve(V,0.0)
    if s is None: print(f"{V:4.0f}  no solution"); continue
    al,de,T=s
    da,rx=aileron_for_reaction(max(T,0),V,al)
    print(f"{V:4.0f} {deg(al):8.2f} {deg(de):8.2f} {T:7.2f} {math.sqrt(max(T,0)/KF):9.0f} {deg(da):8.2f}   {abs(de)<=DE_LIM}   {0<=T<=TMAX}   {DA_LIM[0]<=da<=DA_LIM[1]}")

print("\nGLIDE (T=0): unknown alpha, delta_e, gamma")
print("  V   gamma[deg] alpha[deg] de[deg]  L/D   sink[m/s]  de_in_limits")
for V in (6,8,10,12,14,16,18,20,24,28,32):
    s=solve_glide(V)
    if s is None: print(f"{V:4.0f}  no solution"); continue
    al,de,gam=s
    print(f"{V:4.0f} {deg(gam):9.2f} {deg(al):9.2f} {deg(de):8.2f} {1/abs(math.tan(gam)) if gam else float('inf'):6.1f} {V*math.sin(gam):8.2f}   {abs(de)<=DE_LIM}")

print("\nPOWERED, feasible flight-path angles (|de|<=13.61 deg, 0<=T<=Tmax) per airspeed")
print("  V   gamma_min..gamma_max [deg] ; at gamma_min:(alpha,de,T) ; at gamma_max:(alpha,de,T)")
for V in (6,8,10,12,14,16,20,24,30,40):
    ok=[]
    for gam in np.radians(np.arange(-60,61,1.0)):
        s=solve(V,gam)
        if s is not None and abs(s[1])<=DE_LIM and 0<=s[2]<=TMAX: ok.append((gam,s))
    if not ok: print(f"{V:4.0f}  none"); continue
    a,b=ok[0],ok[-1]
    print(f"{V:4.0f}  {deg(a[0]):+6.0f}..{deg(b[0]):+6.0f}   ({deg(a[1][0]):+.1f},{deg(a[1][1]):+.1f},{a[1][2]:.1f})  ({deg(b[1][0]):+.1f},{deg(b[1][1]):+.1f},{b[1][2]:.1f})")
