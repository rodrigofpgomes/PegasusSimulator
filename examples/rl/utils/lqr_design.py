"""
| File: lqr_design.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Gains of the double-integrator LQR used as a baseline in play_multi.py.
|
| Model per axis (ZOH at the control period dt), state x = [p, v], input u = a:
|     p+ = p + dt v + dt^2/2 u,   v+ = v + dt u
| Cost sum_k gamma^k (x' Q x + u' R u), Q = diag(q_pos I3, q_vel I3), R = r I3.
| The control law is u = K e with e = [p_ref - p, v_ref - v], i.e. u = -K x.
| NumPy only, so the same gains can be checked outside Isaac Sim.
"""

import numpy as np


def double_integrator(dt: float):
    """Return (A, B) of the three-axis double integrator discretized with ZOH at ``dt``."""
    I3 = np.eye(3)
    A = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
    B = np.vstack([0.5 * dt * dt * I3, dt * I3])
    return A, B


def discounted_dlqr(dt: float, q_pos: float, q_vel: float, r: float, gamma: float = 1.0,
                    iterations: int = 100000, tol: float = 1e-12):
    """Return (K 3x6, P 6x6) of the discounted discrete LQR.

    Solves P = Q + g A'PA - g^2 A'PB (R + g B'PB)^-1 B'PA by Riccati iteration and
    K = g (R + g B'PB)^-1 B'PA. Equivalent to the tasks' _compute_discounted_dlqr
    (scipy DARE on sqrt(g) A, sqrt(g) B) with the same weights and dt.
    """
    if dt <= 0 or r <= 0 or q_pos < 0 or q_vel < 0 or not 0 < gamma <= 1:
        raise ValueError("LQR design needs dt > 0, r > 0, q >= 0 and 0 < gamma <= 1")
    A, B = double_integrator(dt)
    Q = np.diag([q_pos] * 3 + [q_vel] * 3)
    R = r * np.eye(3)
    P = Q.copy()
    for _ in range(iterations):
        G = R + gamma * B.T @ P @ B
        K = gamma * np.linalg.solve(G, B.T @ P @ A)
        P_next = Q + gamma * A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_next - P)) <= tol * max(1.0, np.max(np.abs(P))):
            P = P_next
            break
        P = P_next
    else:
        raise RuntimeError("Riccati iteration did not converge")
    K = gamma * np.linalg.solve(R + gamma * B.T @ P @ B, B.T @ P @ A)
    return K, P


def pd_gains(kp, kd):
    """Return K 3x6 for u = kp e_p + kd e_v; kp and kd are scalars or per-axis triples."""
    kp = np.broadcast_to(np.asarray(kp, dtype=float), (3,))
    kd = np.broadcast_to(np.asarray(kd, dtype=float), (3,))
    return np.hstack([np.diag(kp), np.diag(kd)])


def closed_loop_poles(K, dt: float):
    """Return the discrete closed-loop eigenvalues of A - B K (all |z| < 1 when stable)."""
    A, B = double_integrator(dt)
    return np.linalg.eigvals(A - B @ np.asarray(K, dtype=float))
