"""
| File: shuttle_glider_trim.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Steady-flight trim of the complete Shuttle-Glider for every flight regime of the training curriculum.
|
| Six-degree-of-freedom equilibrium of the merged vehicle (four lift rotors, puller, EasyGlider aerodynamics) about the
| assembly centre of mass, solved for minimum mechanical shaft power with the actuator limits as constraints:
|   hover, straight level flight, climb/descent and coordinated level turns,
|   case A: control surfaces fixed at zero,  case B: elevator, aileron and rudder free within their travel.
| The vehicle data are read from the same sources as the simulation (USD mass properties, vehicle_physics presets,
| GliderAerodynamicsBatch); the float64 aerodynamic model is checked against the batched torch model.
|
| Usage (CPU, system python, about 5 min):
|   python3 examples/rl/diagnostics/shuttle_glider_trim.py --tex shuttle_glider_trim_tables.tex
| The printed RESET TABLE (or --limits-table alone, about 10 s) is the trim table of
| pegasus/simulator/logic/rl/shuttle_glider_trim.py. Requires numpy, scipy, torch and usd-core (pip install usd-core).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "review"))
from offline_runtime import enable  # noqa: E402

enable()
from pegasus.simulator.logic.dynamics.glider_aerodynamics import (  # noqa: E402
    EASYGLIDER_REFERENCE_FROM_COM, GliderAerodynamicsBatch, GliderCoefficients, GliderGeometry)

sys.path.insert(0, str(ROOT / "extensions" / "pegasus.simulator" / "pegasus" / "simulator" / "logic" / "vehicles"))
import vehicle_physics as vp  # noqa: E402

USD = ROOT / "extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Shuttle/shuttle_glider2.usd"
G0 = 9.81
WLS_SURFACE_WEIGHT = 1.0  # [N^2] cost of a surface at its full admissible travel, against squared rotor differentials [N^2]
SURFACE_COST_W = 5.0     # [W] cost of one surface at full travel in the regularised trim (about 0.75 % of hover power)
# Reset table of shuttle_glider/tracking: actuator limits only (lift thrust in [0, T_max], no reserve, alpha free) and
# minimum shaft power with LIMITS_SURFACE_COST_W per full-travel deflection squared, the power/surface ratio of the
# task reward (shaft power alone is degenerate in the elevator: the lift-rotor pitch differential replaces it).
LIMITS_SURFACE_COST_W = 25.0
LIMITS_ALPHA_BOUNDS = (-10.0, 25.0)
# Allocation table (--alloc, env cfg trim_table="alloc"): the same trim plus the allocation terms of the reward, converted
# to watts with the reward ratio P_hover / w_power: w_rotor_diff sum ((T_i - mean T) / T_max)^2 (rotor moments priced
# against the surfaces, whose cost falls with 1/q^2) and w_surface_l1 exp(-q / q_ref) sum (sqrt(d^2 + eps^2) - eps).
REWARD_W_POWER = 0.05
REWARD_P_HOVER = 663.6
ALLOC_W_ROTOR_DIFF = 0.1
ALLOC_W_SURFACE_L1 = 0.02
ALLOC_SURFACE_L1_Q_REF = 5.0     # [Pa]
ALLOC_SURFACE_L1_EPS = 0.01
# Cancelled moments of the reward (w_conflict, w_rotor_cancel, conflict_eps of the env cfg) [per N m]
ALLOC_W_CONFLICT = 0.2
ALLOC_W_ROTOR_CANCEL = 0.2
ALLOC_CONFLICT_EPS = 0.05


# =============================================================================
# Vehicle data
# =============================================================================

def _prim(stage, name):
    """First Xform prim with this name in the order of the file (depth first)."""
    for prim in stage.Traverse():
        if prim.GetName() == name and prim.GetTypeName() == "Xform":
            return prim
    raise KeyError(f"no Xform prim named {name!r} in {USD}")


def _vec(prim, attribute):
    """Value of a vector attribute; quaternions (physics:principalAxes) as (w, x, y, z)."""
    value = prim.GetAttribute(attribute).Get()
    if hasattr(value, "GetReal"):
        return np.array([value.GetReal(), *value.GetImaginary()], dtype=float)
    return np.array(value, dtype=float)


def _quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_vehicle():
    """Mass properties and actuator positions relative to the assembly CoM, from the USD used by the simulator."""
    try:
        from pxr import Usd
    except ImportError as error:
        raise SystemExit("The USD Python bindings are required: pip install usd-core") from error
    stage = Usd.Stage.Open(str(USD))
    parts = {}
    for name in ("body", "rotor_0", "rotor_1", "rotor_2", "rotor_3", "rotor_puller"):
        prim = _prim(stage, name)
        mass = float(prim.GetAttribute("physics:mass").Get())
        com_local = _vec(prim, "physics:centerOfMass")
        inertia = _vec(prim, "physics:diagonalInertia")
        axes = _vec(prim, "physics:principalAxes")
        origin = _vec(prim, "xformOp:translate") if name != "body" else np.zeros(3)
        parts[name] = dict(mass=mass, com=origin + com_local, I=_quat_to_matrix(axes) @ np.diag(inertia) @ _quat_to_matrix(axes).T)
    m = sum(p["mass"] for p in parts.values())
    com = sum(p["mass"] * p["com"] for p in parts.values()) / m
    I = np.zeros((3, 3))
    for p in parts.values():
        r = p["com"] - com
        I += p["I"] + p["mass"] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    sh, gl = vp.SHUTTLE_PHYSICS, vp.EASYGLIDER_PHYSICS
    veh = dict(
        m=m, I=I, com=com,
        r_rotor=np.array([parts[f"rotor_{i}"]["com"] - com for i in range(4)]),
        r_puller=parts["rotor_puller"]["com"] - com,
        r_bodycom=parts["body"]["com"] - com,
        kf=np.array(sh["rotor_constant"]), km=np.array(sh["reaction_moment_coefficient"]),
        rot_dir=np.array(sh["rot_dir"], dtype=float), w_max=np.array(sh["max_rotor_velocity"]),
        kf_p=gl["rotor_constant"][0], km_p=gl["reaction_moment_coefficient"][0], rot_dir_p=float(gl["rot_dir"][0]),
        w_max_p=float(gl["max_rotor_velocity"]),
        de_lim=np.radians(vp.EASYGLIDER_SURFACE_LIMITS_DEG["elevator"]),
        da_lim=np.radians(vp.EASYGLIDER_SURFACE_LIMITS_DEG["aileron"]),
        dr_lim=np.radians(vp.EASYGLIDER_SURFACE_LIMITS_DEG["rudder"]),
    )
    # Glider CoM + reference offset - /body CoM, as _MERGED_R_COM in shuttle_glider_batch2.py
    glider_com = _vec(_prim(stage, "glider"), "xformOp:translate")
    veh["r_com_aero"] = glider_com + np.array(EASYGLIDER_REFERENCE_FROM_COM) - (parts["body"]["com"])
    veh["T_max"] = veh["kf"] * veh["w_max"] ** 2
    veh["T_max_p"] = veh["kf_p"] * veh["w_max_p"] ** 2
    return veh


# =============================================================================
# Aerodynamics (float64 copy of GliderAerodynamicsBatch.forces_moments)
# =============================================================================

C, G = GliderCoefficients(), GliderGeometry()


def cl_cd(alpha, de):
    x1 = np.clip(-C.M_sig * (alpha - C.alpha0), -80, 80)
    x2 = np.clip(C.M_sig * (alpha + C.alpha0), -80, 80)
    e1, e2 = math.exp(x1), math.exp(x2)
    sig = (1 + e1 + e2) / ((1 + e1) * (1 + e2))
    cl_lin = C.CL0 + C.CLa * alpha
    cl_fp = 2 * np.sign(alpha) * math.sin(alpha) ** 2 * math.cos(alpha)
    return (1 - sig) * cl_lin + sig * cl_fp, C.CD0 + cl_lin ** 2 / (math.pi * G.e * G.AR) + C.CDde * abs(de)


def aero(v_b, w_b, de, da, dr, r_com):
    """Force (FLU) and moment about the /body CoM; v_b is the /body CoM velocity."""
    va = v_b + np.cross(w_b, r_com)
    Va = np.linalg.norm(va)
    Vs = max(Va, 0.5)
    alpha = float(np.clip(math.atan2(-va[2], va[0]), -math.pi / 2, math.pi / 2))
    beta = -math.asin(float(np.clip(va[1] / Vs, -1, 1)))
    qS = 0.5 * G.rho * Va ** 2 * G.S
    p, q, r = w_b
    ph, qh, rh = p * G.b / (2 * Vs), q * G.c / (2 * Vs), r * G.b / (2 * Vs)
    CL, CD = cl_cd(alpha, de)
    lift = qS * (CL + C.CLq * qh + C.CLde * de)
    drag = max(qS * (CD + C.CDq * qh), 0.0)
    ca, sa = math.cos(alpha), math.sin(alpha)
    F = np.array([-drag * ca + lift * sa,
                  qS * (C.CY0 + C.CYb * beta + C.CYp * ph + C.CYr * rh + C.CYda * da + C.CYdr * dr),
                  drag * sa + lift * ca])
    M = np.array([qS * G.b * (C.Cl0 + C.Clb * beta + C.Clp * ph + C.Clr * rh + C.Clda * da + C.Cldr * dr),
                  qS * G.c * (C.Cm0 + C.Cma * alpha + C.Cmq * qh + C.Cmde * de),
                  qS * G.b * (C.Cn0 + C.Cnb * beta + C.Cnp * ph + C.Cnr * rh + C.Cnda * da + C.Cndr * dr)])
    return F, M + np.cross(r_com, F)


def check_against_torch(veh):
    model = GliderAerodynamicsBatch({"geometry": {"r_com": tuple(veh["r_com_aero"])}}, 1, "cpu")
    worst = 0.0
    for v, w, d in [((10.0, 0.4, -1.2), (0.1, -0.3, 0.2), (0.1, -0.05, 0.08)),
                    ((4.0, 0.0, -0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    ((14.0, -0.6, 0.9), (-0.2, 0.4, -0.1), (-0.2, 0.2, -0.1))]:
        Ft, Mt = model.forces_moments(torch.tensor([v]), torch.tensor([w]), torch.tensor([d]))
        F, M = aero(np.array(v), np.array(w), *d, veh["r_com_aero"])
        worst = max(worst, np.abs(F - Ft[0].double().numpy()).max(), np.abs(M - Mt[0].double().numpy()).max())
    assert worst < 5e-4, worst
    return worst


# =============================================================================
# Equilibrium
# =============================================================================

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def attitude(alpha, beta, mu, gamma):
    """Body-to-world rotation (FLU/ENU) for heading 0: wind axes (flight-path gamma, bank mu about the velocity),
    then body axes with angle of attack alpha and sideslip beta, so that v_b = V (cos a cos b, -sin b, -sin a cos b)."""
    return Ry(-gamma) @ Rx(mu) @ Rz(beta) @ Ry(-alpha)


def residual(x, veh, V, gamma, turn_rate, surfaces):
    """Net force [N] and moment [N m] on the vehicle in body axes about the assembly CoM (zero at a trim)."""
    alpha, beta, mu = x[0:3]
    T = x[3:7]
    Tp = x[7]
    de, da, dr = x[8:11] if surfaces else (0.0, 0.0, 0.0)
    R = attitude(alpha, beta, mu, gamma)
    v_w = R @ np.array([V * math.cos(alpha) * math.cos(beta), -V * math.sin(beta), -V * math.sin(alpha) * math.cos(beta)])
    v_b = R.T @ v_w
    w_b = R.T @ np.array([0.0, 0.0, turn_rate])
    r_bc = veh["r_bodycom"]
    F_a, M_a = aero(v_b + np.cross(w_b, r_bc), w_b, de, da, dr, veh["r_com_aero"])
    F = F_a.copy()
    M = M_a + np.cross(r_bc, F_a)
    for i in range(4):
        f = np.array([0.0, 0.0, T[i]])
        F += f
        M += np.cross(veh["r_rotor"][i], f) + np.array([0.0, 0.0, veh["rot_dir"][i] * veh["km"][i] / veh["kf"][i] * T[i]])
    fp = np.array([Tp, 0.0, 0.0])
    F += fp
    M += np.cross(veh["r_puller"], fp) + np.array([veh["rot_dir_p"] * veh["km_p"] / veh["kf_p"] * Tp, 0.0, 0.0])
    F += veh["m"] * (R.T @ np.array([0.0, 0.0, -G0]))
    F -= veh["m"] * np.cross(w_b, v_b)          # steady turn: m (w x v) = sum of forces
    M -= np.cross(w_b, veh["I"] @ w_b)          # steady rotation: w x (I w) = sum of moments
    return np.concatenate([F, M])


def shaft_power(T, Tp, veh):
    """Mechanical shaft power of the rotor model, P = k_m * omega^3 per rotor [W]."""
    w = np.sqrt(np.clip(T, 0, None) / veh["kf"])
    wp = math.sqrt(max(Tp, 0.0) / veh["kf_p"])
    return float(np.sum(veh["km"] * w ** 3) + veh["km_p"] * wp ** 3)


def rotor_differentials(T):
    """Pitch, roll and yaw thrust differentials of the lift rotors [N] (front-rear, left-right, (rot_dir +1)-(rot_dir -1))."""
    return np.array([(T[0] + T[2]) - (T[1] + T[3]), (T[1] + T[2]) - (T[0] + T[3]), (T[2] + T[3]) - (T[0] + T[1])])


def _cancelled(a, b, eps=ALLOC_CONFLICT_EPS):
    """Moment by which a and b cancel (min(|a|, |b|) for opposite signs), Huber-smoothed as in the tracking reward."""
    overlap = -np.sign(a) * np.sign(b) * min(abs(a), abs(b))
    return 0.0 if overlap <= 0.0 else (overlap ** 2 / (2.0 * eps) if overlap < eps else overlap - 0.5 * eps)


def cancelled_moments(x, veh, V):
    """(surfaces vs lift rotors, inside the lift rotors) cancelled moment [N m] of a trim vector, as in the reward."""
    T, (de, da, dr) = x[3:7], x[8:11]
    qS = 0.5 * G.rho * V ** 2 * G.S
    m_surf = (qS * G.b * C.Clda * da, qS * G.c * C.Cmde * de, qS * G.b * C.Cndr * dr)
    r = veh["r_rotor"]
    roll_i, pitch_i = T * r[:, 1], -T * r[:, 0]
    m_rot = (roll_i.sum(), pitch_i.sum(), np.sum(T * veh["rot_dir"] * veh["km"] / veh["kf"]))
    front, left = r[:, 0] > 0, r[:, 1] > 0
    surf_rot = sum(_cancelled(a, b) for a, b in zip(m_surf, m_rot))
    inside = _cancelled(roll_i[front].sum(), roll_i[~front].sum()) + _cancelled(pitch_i[left].sum(), pitch_i[~left].sum())
    return np.array([surf_rot, inside])


def solve_trim(veh, V, gamma=0.0, radius=None, turn_dir=1.0, surfaces=False, reserve=2.0, surface_ratio=1.0,
               alpha_bounds=(-8.0, 12.0), coordinated=True, x0=None, allocation="wls", surface_cost=None,
               rotor_diff_cost=0.0, surface_l1_cost=0.0, conflict_cost=0.0, rotor_cancel_cost=0.0, rotor_diff_q_ref=0.0):
    """Steady trim with minimum shaft power. Returns a dict or None when no admissible trim exists.

    With surfaces, allocation="surfaces_first" lets the control surfaces close the moment balance: the squared lift-rotor
    differentials are penalised (10 W per N^2, i.e. lexicographically before power) so the rotors supply collective
    thrust only, unless the surface travel is insufficient. allocation="power" minimises shaft power alone; it is
    degenerate in the surface deflections at low speed. allocation="regularised" (default) adds SURFACE_COST_W per
    full-travel deflection squared, which selects the smallest deflection among the near-minimum-power trims.
    allocation="wls" (default with surfaces; pass alpha_bounds=(a, a) with the alpha of the surfaces-zero trim)
    allocates the moments between rotor differentials and surfaces by weighted least squares.
    rotor_diff_cost [W] per sum ((T_i - mean T) / T_max)^2 and surface_l1_cost [W] per sum |d / d_max| at zero dynamic
    pressure (fading as exp(-q / ALLOC_SURFACE_L1_Q_REF)) add the allocation terms of the reward (regularised only);
    conflict_cost and rotor_cancel_cost [W per N m] price the cancelled moments (cancelled_moments).
    rotor_diff_q_ref > 0 [Pa] scales rotor_diff_cost by q / (q + rotor_diff_q_ref) (env cfg of the same name).
    """
    turn_rate = turn_dir * V / radius if radius else 0.0
    n = 11 if surfaces else 8
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    lb[0], ub[0] = np.radians(alpha_bounds[0]), np.radians(alpha_bounds[1])
    lb[1], ub[1] = (0.0, 0.0) if coordinated else (-0.3, 0.3)
    lb[2], ub[2] = -1.2, 1.2
    lb[3:7], ub[3:7] = reserve, veh["T_max"] - reserve
    lb[7], ub[7] = 0.0, veh["T_max_p"]
    if surfaces:
        for k, lim in zip((8, 9, 10), (veh["de_lim"], veh["da_lim"], veh["dr_lim"])):
            lb[k], ub[k] = surface_ratio * lim[0], surface_ratio * lim[1]
    span = surface_ratio * np.array([veh["de_lim"][1], veh["da_lim"][1], veh["dr_lim"][1]])
    if surfaces and allocation == "wls":
        # Moment allocation at fixed alpha: rotor differentials [N] against surface deflections (fraction of the
        # admissible travel); the collective thrusts and the puller then follow from the equilibrium.
        objective = lambda x: np.sum(rotor_differentials(x[3:7]) ** 2) + WLS_SURFACE_WEIGHT * np.sum((x[8:11] / span) ** 2)
    else:
        w_diff = 10.0 if (surfaces and allocation == "surfaces_first") else 0.0
        w_surf = (SURFACE_COST_W if surface_cost is None else surface_cost) if (surfaces and allocation == "regularised") else 0.0
        low_q = math.exp(-0.5 * G.rho * V ** 2 / ALLOC_SURFACE_L1_Q_REF)
        q_dyn = 0.5 * G.rho * V ** 2
        rotor_diff_cost = rotor_diff_cost * (q_dyn / (q_dyn + rotor_diff_q_ref) if rotor_diff_q_ref > 0.0 else 1.0)
        l1 = lambda d: np.sum(np.sqrt(d ** 2 + ALLOC_SURFACE_L1_EPS ** 2) - ALLOC_SURFACE_L1_EPS)
        objective = lambda x: (shaft_power(x[3:7], x[7], veh) + w_diff * np.sum(rotor_differentials(x[3:7]) ** 2)
                               + rotor_diff_cost * np.sum(((x[3:7] - x[3:7].mean()) / veh["T_max"]) ** 2)
                               + (w_surf * np.sum((x[8:11] / span) ** 2) + surface_l1_cost * low_q * l1(x[8:11] / span)
                                  if surfaces else 0.0)
                               + (np.dot((conflict_cost, rotor_cancel_cost), cancelled_moments(x, veh, V))
                                  if surfaces and (conflict_cost or rotor_cancel_cost) else 0.0)) / 100.0
    cons = {"type": "eq", "fun": lambda x: residual(x, veh, V, gamma, turn_rate, surfaces)}
    starts = [x0] if x0 is not None else []
    hover_T = veh["m"] * G0 / 4
    for a0 in (0.0, 0.05, 0.12, -0.05):
        guess = np.zeros(n)
        guess[0] = a0
        guess[2] = -turn_dir * math.atan2(V * abs(turn_rate), G0) if radius else 0.0
        guess[3:7] = max(min(hover_T, veh["T_max"][0] - reserve - 1), reserve + 0.5)
        guess[7] = 2.0
        starts.append(guess)
    best = None
    for guess in starts:
        guess = np.clip(guess, np.where(np.isfinite(lb), lb, -1e9), np.where(np.isfinite(ub), ub, 1e9))
        res = minimize(objective, guess, method="SLSQP", bounds=list(zip(lb, ub)), constraints=[cons],
                       options=dict(maxiter=600, ftol=1e-12))
        r = residual(res.x, veh, V, gamma, turn_rate, surfaces)
        ok = res.success and np.abs(r).max() < 1e-6 and np.all(res.x >= lb - 1e-7) and np.all(res.x <= ub + 1e-7)
        if ok and (best is None or res.fun < best.fun - 1e-9):
            best = res
    if best is None:
        return None
    x = best.x
    T, Tp = x[3:7], x[7]
    R = attitude(x[0], x[1], x[2], gamma)
    phi, theta, psi = euler_zyx(R)
    out = dict(V=V, gamma=gamma, radius=radius, alpha=x[0], beta=x[1], mu=x[2], phi=phi, theta=theta, psi=psi,
               T=T, Tp=Tp, omega=np.sqrt(T / veh["kf"]), omega_p=math.sqrt(max(Tp, 0) / veh["kf_p"]),
               de=x[8] if surfaces else 0.0, da=x[9] if surfaces else 0.0, dr=x[10] if surfaces else 0.0,
               P=shaft_power(T, Tp, veh), x=x, surfaces=surfaces, turn_rate=turn_rate, diff=rotor_differentials(T))
    q = 0.5 * G.rho * V ** 2 * G.S
    CL, _ = cl_cd(x[0], out["de"])
    out["wing_lift"] = q * (CL + C.CLde * out["de"])
    out["offload"] = out["wing_lift"] / (veh["m"] * G0)
    return out


def euler_zyx(R):
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll); positive pitch is nose down in FLU."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# =============================================================================
# Longitudinal reset model (same equations as pegasus.simulator.logic.rl.shuttle_glider_trim)
# =============================================================================

def longitudinal_thrusts(veh, V, alpha, de, g_eff=G0):
    """Exact longitudinal balance with symmetric pairs, beta = 0 and no rotation: returns (Tp, T_front_pair, T_rear_pair)."""
    F_a, M_a = aero(np.array([V * math.cos(alpha), 0.0, -V * math.sin(alpha)]), np.zeros(3), de, 0.0, 0.0, veh["r_com_aero"])
    M_a = M_a + np.cross(veh["r_bodycom"], F_a)
    W = veh["m"] * g_eff
    Tp = W * math.sin(alpha) - F_a[0]
    Tq = W * math.cos(alpha) - F_a[2]
    xf, xr = veh["r_rotor"][0, 0], veh["r_rotor"][1, 0]
    My = M_a[1] + veh["r_puller"][2] * Tp
    Tf = (My - xr * Tq) / (xf - xr)
    return Tp, Tf, Tq - Tf


# =============================================================================
# Tables
# =============================================================================

def deg(x):
    return math.degrees(x)


def fmt_row(t, veh):
    if t is None:
        return "no admissible trim"
    return (f"a={deg(t['alpha']):6.2f} th={deg(t['theta']):6.2f} ph={deg(t['phi']):6.2f} "
            f"de={deg(t['de']):6.2f} da={deg(t['da']):6.2f} dr={deg(t['dr']):6.2f} "
            f"T=[{', '.join(f'{v:5.2f}' for v in t['T'])}] Tp={t['Tp']:5.2f} "
            f"w=[{', '.join(f'{v:4.0f}' for v in t['omega'])}] wp={t['omega_p']:4.0f} "
            f"L/W={t['offload']:5.2f} P={t['P']:6.1f} W diff=[{', '.join(f'{v:5.2f}' for v in t['diff'])}]")


OLD_SPEED_KNOTS = (0.0, 2.0, 3.0, 3.5, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0)
OLD_ALPHA_CAP_DEG = (0.0, 3.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0, 8.0, 5.5, 3.5)


def old_trim(veh, V):
    """Reset of the original task5_* tasks evaluated on the current vehicle: alpha envelope of the old model and
    elevator -9 deg ramped in between 6 and 10 m/s (80 % clamp); thrusts from the same longitudinal balance."""
    alpha = float(np.interp(V, OLD_SPEED_KNOTS, OLD_ALPHA_CAP_DEG))
    de = max(-0.8 * deg(veh["de_lim"][1]), -9.0 * min(max((V - 6.0) / 4.0, 0.0), 1.0))
    Tp, Tf, Tr = longitudinal_thrusts(veh, V, math.radians(alpha), math.radians(de))
    T = np.array([Tf / 2, Tr / 2, Tf / 2, Tr / 2])
    return dict(V=V, alpha=alpha, de=de, Tp=Tp, Tf=Tf, Tr=Tr, P=shaft_power(T, Tp, veh),
                ok=bool(Tp >= 0 and min(Tf, Tr) >= 4.0 and max(Tf, Tr) <= 2 * veh["T_max"][0] - 4.0))


def trim_pair(veh, V, **kwargs):
    """Case A (surfaces zero, minimum power) and case B (alpha of case A, WLS moment allocation with surfaces)."""
    a = solve_trim(veh, V, surfaces=False, **kwargs)
    if a is None:
        return None, None
    if V == 0.0:
        return a, dict(a, surfaces=True)      # surfaces have no effect in hover
    alpha = deg(a["alpha"])
    b = solve_trim(veh, V, surfaces=True, allocation="wls", alpha_bounds=(alpha, alpha), **kwargs)
    return a, b


def solve_limits_trim(veh, V, **kwargs):
    """Trim of the shuttle_glider/tracking reset: actuator limits only, minimum shaft power, surface regularisation."""
    return solve_trim(veh, V, surfaces=V > 0.0, allocation="regularised", reserve=0.0,
                      alpha_bounds=LIMITS_ALPHA_BOUNDS, surface_cost=LIMITS_SURFACE_COST_W, **kwargs)


def alloc_costs(cancel=True, rotor_diff_q_ref=0.0):
    """Allocation terms of the reward in watts (solve_trim kwargs of the "alloc" table; cancel=False: "wls" table)."""
    to_w = REWARD_P_HOVER / REWARD_W_POWER
    costs = dict(rotor_diff_cost=ALLOC_W_ROTOR_DIFF * to_w, surface_l1_cost=ALLOC_W_SURFACE_L1 * to_w,
                 rotor_diff_q_ref=rotor_diff_q_ref)
    if cancel:
        costs.update(conflict_cost=ALLOC_W_CONFLICT * to_w, rotor_cancel_cost=ALLOC_W_ROTOR_CANCEL * to_w)
    return costs


def limits_reset_table(veh, speeds=tuple(float(v) for v in range(17)), alloc=False, cancel=True, rotor_diff_q_ref=0.0):
    """Print the TRIM_* columns of pegasus/simulator/logic/rl/shuttle_glider_trim.py and the trim summary."""
    rows = []
    print(" V [m/s]  alpha  elev   ail    rud  [deg] | front/rear lift thrust / T_max | puller [N] | P [W] | wing lift / W")
    for V in speeds:
        t = solve_limits_trim(veh, V, **(alloc_costs(cancel, rotor_diff_q_ref) if alloc else {}))
        if t is None:
            raise RuntimeError(f"no admissible trim at {V} m/s")
        T = t["T"]
        rows.append((V, deg(t["alpha"]), deg(t["de"]), deg(t["da"]), deg(t["dr"])))
        print(f"{V:6.1f} {deg(t['alpha']):7.2f} {deg(t['de']):6.2f} {deg(t['da']):6.2f} {deg(t['dr']):6.2f} | "
              f"{(T[0] + T[2]) / 2 / veh['T_max'][0]:5.2f} {(T[1] + T[3]) / 2 / veh['T_max'][0]:5.2f} | "
              f"{t['Tp']:6.2f} | {t['P']:6.1f} | {t['offload']:5.2f}")
    for name, k in (("SPEED_KNOTS", 0), ("ALPHA_DEG", 1), ("ELEVATOR_DEG", 2), ("AILERON_DEG", 3), ("RUDDER_DEG", 4)):
        print(f"TRIM_{name} = (" + ", ".join(f"{r[k]:.{1 if k == 0 else 2}f}" for r in rows) + ")")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=str, default=None, help="write the LaTeX tables to this file")
    parser.add_argument("--limits-table", action="store_true",
                        help="print only the actuator-limits reset table of shuttle_glider/tracking (about 10 s)")
    parser.add_argument("--alloc", action="store_true",
                        help="with --limits-table: add the allocation terms (rotor differential, low-q surface L1)")
    parser.add_argument("--wls", action="store_true",
                        help="with --limits-table: convex allocation terms only (rotor differential, low-q L1; no "
                             "cancelled-moment costs), the \"wls\" table")
    parser.add_argument("--rotor-diff-q-ref", type=float, default=0.0,
                        help="with --wls: scale the rotor differential cost by q / (q + this) [Pa] (the \"wls_q\" table)")
    args = parser.parse_args()
    if args.limits_table:
        limits_reset_table(load_vehicle(), alloc=args.alloc or args.wls, cancel=not args.wls,
                           rotor_diff_q_ref=args.rotor_diff_q_ref)
        return

    veh = load_vehicle()
    err = check_against_torch(veh)
    W = veh["m"] * G0
    print(f"mass {veh['m']:.4f} kg, weight {W:.3f} N, assembly CoM in /body frame {np.round(veh['com'], 6)}")
    print(f"inertia about the assembly CoM [kg m^2]:\n{np.round(veh['I'], 5)}")
    print(f"rotors {np.round(veh['r_rotor'], 4).tolist()}\npuller {np.round(veh['r_puller'], 4)}  "
          f"/body CoM {np.round(veh['r_bodycom'], 5)}  aero reference from /body CoM {np.round(veh['r_com_aero'], 5)}")
    print(f"T_max lift {veh['T_max'][0]:.2f} N, puller {veh['T_max_p']:.2f} N; float64 model vs torch: {err:.1e}")
    print("surface limits [deg]:", np.degrees(veh["de_lim"]), np.degrees(veh["da_lim"]), np.degrees(veh["dr_lim"]))

    # Aerodynamic facts used in the text
    a_grid = np.radians(np.linspace(-10, 30, 4001))
    cl = np.array([cl_cd(a, 0.0)[0] for a in a_grid])
    a_clmax = a_grid[cl.argmax()]
    print(f"CL_max = {cl.max():.3f} at alpha = {deg(a_clmax):.2f} deg; CL(12 deg) = {cl_cd(math.radians(12), 0)[0]:.3f}")

    tables = {}
    speeds = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]

    print("\nLEVEL FLIGHT. A: surfaces zero, minimum power. B: alpha of A, moments allocated by WLS between rotor "
          "differentials and surfaces (full travel). P: surfaces free, minimum power (comparison only)")
    for key in ("A", "B", "P"):
        tables[f"level_{key}"] = []
    for V in speeds:
        a, b = trim_pair(veh, V)
        p = solve_trim(veh, V, surfaces=True, allocation="power")
        for key, t in (("A", a), ("B", b), ("P", p)):
            tables[f"level_{key}"].append(t)
            print(f"  {key} V={V:5.1f} " + fmt_row(t, veh))

    print("\nALTERNATIVE TRIM DEFINITIONS (power [W], elevator [deg])")
    compare = []
    for V in (2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0):
        a = tables["level_A"][speeds.index(V)]
        b = tables["level_B"][speeds.index(V)]
        p = tables["level_P"][speeds.index(V)]
        first = solve_trim(veh, V, surfaces=True, allocation="surfaces_first")
        reg = solve_trim(veh, V, surfaces=True, allocation="regularised")
        old = old_trim(veh, V)
        compare.append(dict(V=V, A=a, B=b, P=p, F=first, R=reg, O=old))
        print(f"  V={V:4.1f} A {a['P']:6.1f} | B {b['P']:6.1f} de {deg(b['de']):6.2f} | P {p['P']:6.1f} de {deg(p['de']):6.2f} | "
              f"first {first['P']:6.1f} de {deg(first['de']):6.2f} a {deg(first['alpha']):5.2f} diff {first['diff'][0]:5.2f} | "
              f"reg {reg['P']:6.1f} de {deg(reg['de']):6.2f} | old {old['P']:6.1f} a {old['alpha']:5.2f} de {old['de']:6.2f}")
    tables["compare"] = compare

    print("\nCLIMB / DESCENT (gamma = +-asin(v_z/V)); vertical oscillation 0.5 m / 10 s -> v_z <= 0.157 m/s; also +-5 deg")
    climb = []
    for V in (8.0, 10.0, 12.0, 13.0):
        for gam_deg in (-5.0, -math.degrees(math.asin(0.157 / V)), math.degrees(math.asin(0.157 / V)), 5.0):
            gam = math.radians(gam_deg)
            for key, t in zip("AB", trim_pair(veh, V, gamma=gam)):
                climb.append(t if t is not None else dict(V=V, gamma=gam, surfaces=key == "B", none=True))
                print(f"  {key} V={V:4.1f} gamma={gam_deg:+5.2f} " + fmt_row(t, veh))
    tables["climb"] = climb

    print("\nCOORDINATED LEVEL TURNS (left turn, radius >= V^2/6)")
    turns = []
    for V in (8.0, 10.0, 12.0, 13.0):
        for R in (20.0, 35.0, 60.0):
            R_eff = max(R, V * V / 6.0)
            for key, t in zip("AB", trim_pair(veh, V, radius=R_eff)):
                turns.append(t if t is not None else dict(V=V, radius=R_eff, surfaces=key == "B", none=True))
                print(f"  {key} V={V:4.1f} R={R_eff:5.1f} a_c={V * V / R_eff:4.2f} " + fmt_row(t, veh))
    tables["turns"] = turns

    print("\nRESET TABLE (pegasus/simulator/logic/rl/shuttle_glider_trim.py): actuator limits only")
    tables["reset"] = [(r[0], r[1], r[2], r[3]) for r in limits_reset_table(veh)]

    # Reset model check: same alpha/de as the 6-DOF optimum, longitudinal closed form
    print("\nRESET MODEL vs 6-DOF optimum (level flight): pair thrusts [N]")
    for key in ("A", "B"):
        for t in tables[f"level_{key}"]:
            if t is None:
                continue
            Tp, Tf, Tr = longitudinal_thrusts(veh, t["V"], t["alpha"], t["de"])
            print(f"  {key} V={t['V']:5.1f} 6dof: Tp={t['Tp']:6.3f} front={t['T'][0] + t['T'][2]:6.3f} rear={t['T'][1] + t['T'][3]:6.3f}"
                  f" | reset: Tp={Tp:6.3f} front={Tf:6.3f} rear={Tr:6.3f}")

    # Turn-start reset approximation: residual of the 6-DOF equations when rate terms are neglected
    print("\nTURN START: residual of the effective-gravity reset (no rate terms) in the 6-DOF equations")
    for t in turns:
        if t.get("none"):
            continue
        a_c = t["V"] ** 2 / t["radius"]
        g_eff = math.hypot(G0, a_c)
        Tp, Tf, Tr = longitudinal_thrusts(veh, t["V"], t["alpha"], t["de"], g_eff)
        print(f"  {'B' if t['surfaces'] else 'A'} V={t['V']:4.1f} R={t['radius']:5.1f} bank={deg(math.atan2(a_c, G0)):5.2f} "
              f"(6dof roll {deg(t['phi']):6.2f}) Tp {Tp:6.2f}/{t['Tp']:6.2f} front {Tf:6.2f}/{t['T'][0] + t['T'][2]:6.2f} "
              f"rear {Tr:6.2f}/{t['T'][1] + t['T'][3]:6.2f}")

    if args.tex:
        write_tex(Path(args.tex), veh, tables, err, cl.max(), a_clmax)
        print(f"\nwrote {args.tex}")


# =============================================================================
# LaTeX output
# =============================================================================

def write_tex(path, veh, tables, err, clmax, a_clmax):
    """LaTeX macros and tables of the trim (written with --tex)."""
    L = ["% Generated by examples/rl/diagnostics/shuttle_glider_trim.py -- do not edit by hand."]
    W = veh["m"] * G0
    macros = dict(sgMass=f"{veh['m']:.3f}", sgWeight=f"{W:.2f}", sgTmax=f"{veh['T_max'][0]:.2f}",
                  sgTmaxP=f"{veh['T_max_p']:.1f}", sgModelErr=f"{err:.0e}", sgCLmax=f"{clmax:.2f}",
                  sgAlphaCLmax=f"{deg(a_clmax):.1f}", sgSurfaceCost=f"{SURFACE_COST_W:.0f}",
                  sgWlsWeight=f"{WLS_SURFACE_WEIGHT:.0f}",
                  sgInertia=f"{veh['I'][0, 0]:.4f},\\ {veh['I'][1, 1]:.4f},\\ {veh['I'][2, 2]:.4f}",
                  sgIxz=f"{veh['I'][0, 2]:.4f}")
    for name, value in macros.items():
        L.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def table(spec, header, units, rows, caption, label):
        L.append("\\begin{table}[htbp]\\centering\\small")
        L.append(f"\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{{spec}}}")
        L.append("\\toprule")
        L.append(header + " \\\\")
        L.append("{}" + units + " \\\\")  # {} keeps a leading [unit] from being read as \\[length]
        L.append("\\midrule")
        L.extend(r + " \\\\" for r in rows)
        L.append("\\bottomrule\\end{tabular}}")
        L.append(f"\\caption{{{caption}}}\\label{{{label}}}")
        L.append("\\end{table}")

    def pairs(t):
        return t["T"][0] + t["T"][2], t["T"][1] + t["T"][3]

    def level_rows(rows):
        out = []
        for t in rows:
            if t is None:
                continue
            Tf, Tr = pairs(t)
            wf, wr = 0.5 * (t["omega"][0] + t["omega"][2]), 0.5 * (t["omega"][1] + t["omega"][3])
            out.append(f"{t['V']:.0f} & {deg(t['alpha']):.2f} & {deg(t['de']):.2f} & {deg(t['da']):.2f} & {Tf:.2f} & {Tr:.2f} & "
                       f"{t['Tp']:.2f} & {wf:.0f} & {wr:.0f} & {t['omega_p']:.0f} & {t['offload']:.2f} & {t['P']:.0f}")
        return out

    head = ("$V$ & $\\alpha$ & $\\delta_e$ & $\\delta_a$ & $T_f$ & $T_r$ & $T_p$ & $\\omega_f$ & $\\omega_r$ & $\\omega_p$ & "
            "$L_w/W$ & $P$")
    units = "[m/s] & [deg] & [deg] & [deg] & [N] & [N] & [N] & [rad/s] & [rad/s] & [rad/s] & [-] & [W]"
    table("r" * 12, head, units, level_rows(tables["level_A"]),
          "Straight level flight, case A (surfaces at zero, minimum shaft power). $T_f$, $T_r$: thrust of the front and rear "
          "lift-rotor pairs; $\\omega_f$, $\\omega_r$, $\\omega_p$: rotor speeds; $L_w/W$: wing lift over weight; $P$: shaft power.",
          "tab:sg_trim_level_a")
    table("r" * 12, head, units, level_rows(tables["level_B"]),
          "Straight level flight, case B (angle of attack of case A; moments allocated between lift-rotor differentials and "
          "surfaces, surfaces within their full travel).", "tab:sg_trim_level_b")

    rows = [f"{V:.0f} & {a:.2f} & {e:.2f} & {d:.2f}" for V, a, e, d in tables["reset"]]
    table("rrrr", "$V$ & $\\alpha$ & $\\delta_e$ & $\\delta_a$", "[m/s] & [deg] & [deg] & [deg]", rows,
          "Trim table of the reset (\\texttt{shuttle\\_glider\\_trim.py}); linear interpolation in $V$, held above 16\\,m/s. "
          "$\\delta_e$ and $\\delta_a$ are used only when the surfaces are active.", "tab:sg_trim_reset")

    rows = []
    for c in tables["compare"]:
        a, b, p, f, r, o = c["A"], c["B"], c["P"], c["F"], c["R"], c["O"]
        rows.append(f"{c['V']:.0f} & {a['P']:.0f} & {b['P']:.0f} & {deg(b['de']):.1f} & {p['P']:.0f} & {deg(p['de']):.1f} & "
                    f"{f['P']:.0f} & {deg(f['de']):.1f} & {deg(f['alpha']):.1f} & {r['P']:.0f} & {deg(r['de']):.1f} & "
                    f"{o['P']:.0f} & {o['alpha']:.1f} & {o['de']:.1f}")
    table("r|r|rr|rr|rrr|rr|rrr",
          "$V$ & A: $P$ & B: $P$ & $\\delta_e$ & min-$P$: $P$ & $\\delta_e$ & surf.-first: $P$ & $\\delta_e$ & $\\alpha$ & "
          "regul.: $P$ & $\\delta_e$ & old: $P$ & $\\alpha$ & $\\delta_e$",
          "[m/s] & [W] & [W] & [deg] & [W] & [deg] & [W] & [deg] & [deg] & [W] & [deg] & [W] & [deg] & [deg]", rows,
          "Alternative trim definitions in straight level flight (Section~\\ref{sec:sg_trim_alternatives}). "
          "min-$P$: minimum power with free surfaces; surf.-first: rotor differentials penalised before power; "
          "regul.: minimum power plus \\sgSurfaceCost\\,W per full-travel deflection squared; old: reset of the original "
          "\\texttt{task5\\_*} tasks (old $\\alpha$ envelope, $\\delta_e=-9^\\circ$ ramped in between 6 and 10\\,m/s) on the current vehicle.",
          "tab:sg_trim_alternatives")

    rows = []
    for t in tables["climb"]:
        case = "B" if t["surfaces"] else "A"
        if t.get("none"):
            rows.append(f"{case} & {t['V']:.0f} & {deg(t['gamma']):+.2f} & \\multicolumn{{7}}{{c}}{{no admissible trim}}")
            continue
        Tf, Tr = pairs(t)
        rows.append(f"{case} & {t['V']:.0f} & {deg(t['gamma']):+.2f} & {deg(t['alpha']):.2f} & {-deg(t['theta']):.2f} & "
                    f"{deg(t['de']):.2f} & {Tf:.2f} & {Tr:.2f} & {t['Tp']:.2f} & {t['P']:.0f}")
    table("crrrrrrrrr", "Case & $V$ & $\\gamma$ & $\\alpha$ & $\\theta_{\\mathrm{up}}$ & $\\delta_e$ & $T_f$ & $T_r$ & $T_p$ & $P$",
          "& [m/s] & [deg] & [deg] & [deg] & [deg] & [N] & [N] & [N] & [W]", rows,
          "Climb and descent. $\\pm\\gamma$ of the curriculum vertical oscillation ($v_z\\le0.157$\\,m/s) and $\\pm5^\\circ$; "
          "$\\theta_{\\mathrm{up}}=\\alpha+\\gamma$ is the nose-up pitch angle.", "tab:sg_trim_climb")

    rows = []
    for t in tables["turns"]:
        case = "B" if t["surfaces"] else "A"
        a_c = t["V"] ** 2 / t["radius"]
        if t.get("none"):
            rows.append(f"{case} & {t['V']:.0f} & {t['radius']:.1f} & {a_c:.2f} & \\multicolumn{{8}}{{c}}{{no admissible trim}}")
            continue
        droll = 0.5 * ((t["T"][1] + t["T"][2]) - (t["T"][0] + t["T"][3]))
        dyaw = 0.5 * ((t["T"][2] + t["T"][3]) - (t["T"][0] + t["T"][1]))
        rows.append(f"{case} & {t['V']:.0f} & {t['radius']:.1f} & {a_c:.2f} & {deg(t['phi']):.2f} & {deg(t['alpha']):.2f} & "
                    f"{deg(t['de']):.2f} & {deg(t['da']):.2f} & {deg(t['dr']):.2f} & {droll:.2f} & {dyaw:.2f} & {t['P']:.0f}")
    table("crrrrrrrrrrr", "Case & $V$ & $R$ & $a_c$ & $\\phi$ & $\\alpha$ & $\\delta_e$ & $\\delta_a$ & $\\delta_r$ & "
          "$\\Delta T_{\\mathrm{roll}}$ & $\\Delta T_{\\mathrm{yaw}}$ & $P$",
          "& [m/s] & [m] & [m/s$^2$] & [deg] & [deg] & [deg] & [deg] & [deg] & [N] & [N] & [W]", rows,
          "Coordinated level left turns ($\\beta=0$, $R\\ge V^2/6$). $\\phi$: roll angle (negative: left wing down); "
          "$\\Delta T_{\\mathrm{roll}}$, $\\Delta T_{\\mathrm{yaw}}$: half the left-minus-right and the "
          "(rotation $+1$)-minus-(rotation $-1$) lift-rotor thrust.", "tab:sg_trim_turn")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
