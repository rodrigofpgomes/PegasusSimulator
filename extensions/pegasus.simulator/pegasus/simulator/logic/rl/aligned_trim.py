"""
| File: aligned_trim.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Shared longitudinal reset from task5_all7_6; current aligned-CoM geometry.
"""
import math
import torch
from pegasus.simulator.logic.transforms import quaternion_apply

PHYSICS_REVISION = "aligned-com-review-v2"

_TRIM_SPEED_KNOTS = (0.0, 2.0, 3.0, 3.5, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0)
_TRIM_ALPHA_CAP_DEG = (0.0, 3.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0, 8.0, 5.5, 3.5)

def bounded_alpha_trim_rad(
    speed: torch.Tensor,
    requested_alpha_rad: torch.Tensor,
) -> torch.Tensor:
    """Limit cruise-start angle of attack to the robust physical envelope.

    Both arguments must have the same shape. The returned value is continuous
    and piecewise linear in airspeed. ``requested_alpha_rad`` remains the
    level-dependent requested reset value; this function only prevents an infeasible
    initial equilibrium at the sampled ``speed0``.
    """
    if speed.shape != requested_alpha_rad.shape:
        raise ValueError("speed and requested_alpha_rad must have the same shape")

    knots = torch.tensor(_TRIM_SPEED_KNOTS, dtype=speed.dtype, device=speed.device)
    caps = torch.deg2rad(torch.tensor(
        _TRIM_ALPHA_CAP_DEG, dtype=speed.dtype, device=speed.device
    ))

    speed_clamped = speed.clamp(min=knots[0], max=knots[-1])
    upper = torch.bucketize(speed_clamped, knots).clamp(1, knots.numel() - 1)
    lower = upper - 1

    v0 = knots[lower]
    v1 = knots[upper]
    blend = (speed_clamped - v0) / (v1 - v0).clamp_min(1e-6)
    alpha_cap = caps[lower] + blend * (caps[upper] - caps[lower])

    return torch.minimum(requested_alpha_rad.clamp_min(0.0), alpha_cap)

class AlignedTrimMixin:
    """Shared physics-only reset; observations and rewards remain task-specific."""

    def _read_aero_and_thrust_constants(self):
        vehicle = self.backend._vehicle

        self._rho = vehicle.aerodynamics.G.rho #1.2041
        self._S = vehicle.aerodynamics.G.S #0.416
        self._AR = vehicle.aerodynamics.G.AR # 1.8**2 / 0.416))
        self._e = vehicle.aerodynamics.G.e #0.8523
        self._c = vehicle.aerodynamics.G.c

        self._CL0 = vehicle.aerodynamics.C.CL0 #0.36986
        self._CLa = vehicle.aerodynamics.C.CLa #3.679875
        self._CD0 = vehicle.aerodynamics.C.CD0 #0.025
        self._CLde = vehicle.aerodynamics.C.CLde
        self._Cm0 = vehicle.aerodynamics.C.Cm0
        self._Cma = vehicle.aerodynamics.C.Cma
        self._Cmde = vehicle.aerodynamics.C.Cmde

        r_origin = torch.as_tensor(vehicle.aerodynamics.G.r_origin, dtype=torch.float32, device=self.device)
        self._r_aero_x = float(r_origin[0])
        self._r_aero_z = float(r_origin[2])

        masses = vehicle._vehicle_prims.get_masses()
        masses = torch.as_tensor(masses, device=self.device).reshape(vehicle.n_vehicles, vehicle.parts_per_vehicle)
        self._W = float(masses[0].sum()) * self.cfg.gravity
        # Equilibrium is about the full assembly COM, not the /body COM.
        local_coms = vehicle._reset_part_positions_body + quaternion_apply(
            vehicle._reset_part_attitudes_body, vehicle._reset_part_coms)
        assembly_com = (local_coms[0] * masses[0, :, None]).sum(0) / masses[0].sum()
        self._r_aero_x = float(r_origin[0] - assembly_com[0])
        self._r_aero_z = float(r_origin[2] - assembly_com[2])
        arms = vehicle._rotor_positions_body[0] - assembly_com
        self.cfg.trim_front_x = float(arms[0, 0])
        self.cfg.trim_rear_x = float(arms[1, 0])
        self.cfg.trim_puller_z = float(arms[4, 2])

        # thrust constants: combined interface stores [4 shuttle, 1 puller]
        rc = vehicle._thrusters._rotor_constant.to(self.device).reshape(-1)

        self._k_shuttle = float(rc[0])
        self._k_puller = float(rc[-1])

        self._min_w = vehicle._thrusters.min_rotor_velocity.to(self.device).reshape(-1)
        self._max_w = vehicle._thrusters.max_rotor_velocity.to(self.device).reshape(-1)

    def _trim_aero_wrench(self, V0: torch.Tensor, alpha: torch.Tensor, delta_e: torch.Tensor):
        """Aerodynamic longitudinal wrench about the assembly COM."""

        aero = self.backend._vehicle.aerodynamics
        qS = 0.5 * self._rho * V0.square() * self._S

        # Same nonlinear lift/drag model used by the simulation.
        CL, CD = aero.CL_CD(alpha, delta_e)

        lift = qS * (CL + self._CLde * delta_e)
        drag = (qS * CD).clamp_min(0.0)

        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        Fx = -drag * ca + lift * sa
        Fz = +drag * sa + lift * ca

        tau_ref = qS * self._c * (self._Cm0 + self._Cma * alpha + self._Cmde * delta_e)

        # tau_origin = tau_ref + r_origin x F_aero
        tau_origin = tau_ref + self._r_aero_z * Fx - self._r_aero_x * Fz

        return Fx, Fz, tau_origin

    def _trim_from_speed(self, V0: torch.Tensor, requested_alpha: torch.Tensor):
        """Longitudinal reset only; preserve force AND pitch equilibrium.

        Start below the old alpha cap and reduce alpha to satisfy BOTH lower
        and upper thrust reserve bounds. This does not impose alpha on policy.
        Lateral moments/reaction torque are not claimed to be trimmed here.
        """
        requested = bounded_alpha_trim_rad(V0, requested_alpha)
        elevator_gate = ((V0 - self.cfg.trim_elevator_on_speed)
                         / (self.cfg.trim_elevator_full_speed - self.cfg.trim_elevator_on_speed)).clamp(0.0, 1.0)
        de_min = self.cfg.delta_e_min * self.cfg.trim_elevator_soft_ratio
        de_max = self.cfg.delta_e_max * self.cfg.trim_elevator_soft_ratio
        delta_e = (math.radians(self.cfg.trim_elevator_deg) * elevator_gate).clamp(de_min, de_max)
        reserve = self.cfg.trim_pair_reserve
        pair_front_max = self._k_shuttle * (self._max_w[0].square() + self._max_w[2].square())
        pair_rear_max = self._k_shuttle * (self._max_w[1].square() + self._max_w[3].square())
        puller_max = self._k_puller * self._max_w[4].square()
        if reserve <= 0.0:
            raise ValueError("trim_pair_reserve must be positive")

        def equilibrium(alpha):
            Fx, Fz, tau_aero = self._trim_aero_wrench(V0, alpha, delta_e)
            Tp = self._W * torch.sin(alpha) - Fx
            Tq = self._W * torch.cos(alpha) - Fz
            My = tau_aero + self.cfg.trim_puller_z * Tp
            Tf = (My - self.cfg.trim_rear_x * Tq) / (self.cfg.trim_front_x - self.cfg.trim_rear_x)
            Tr = Tq - Tf
            valid = ((Tp >= 0.0) & (Tp <= puller_max)
                     & (Tf >= reserve) & (Tf <= pair_front_max - reserve)
                     & (Tr >= reserve) & (Tr <= pair_rear_max - reserve))
            return Tp, Tf, Tr, valid

        low = torch.zeros_like(requested)
        high = requested.clone()
        if not torch.all(equilibrium(low)[3]):
            raise RuntimeError("No feasible alpha=0 longitudinal reset for the configured physical model/reserve")
        # The current 0..alpha_cap branch is monotone in limiting rear-pair
        # feasibility. Keep already feasible requests unchanged.
        requested_ok = equilibrium(requested)[3]
        for _ in range(self.cfg.trim_search_iterations):
            mid = 0.5 * (low + high)
            ok = equilibrium(mid)[3]
            low = torch.where(ok, mid, low)
            high = torch.where(ok, high, mid)
        # Keep a small interior alpha margin for float32 rounding at the boundary.
        alpha_trim = torch.where(requested_ok, requested, (low - 1e-5).clamp_min(0.0))
        Tp, Tf, Tr, valid = equilibrium(alpha_trim)
        if not torch.all(valid):
            raise RuntimeError("Infeasible hybrid reset; verify coefficients, arm positions and reserve")
        omega = torch.zeros((V0.shape[0], 5), dtype=V0.dtype, device=self.device)
        wf = torch.sqrt(0.5 * Tf / self._k_shuttle)
        wr = torch.sqrt(0.5 * Tr / self._k_shuttle)
        omega[:, 0] = wf
        omega[:, 2] = wf
        omega[:, 1] = wr
        omega[:, 3] = wr
        omega[:, 4] = torch.sqrt(Tp / self._k_puller)
        if torch.any(omega > self._max_w[:5].unsqueeze(0) + 1e-4):
            raise RuntimeError("Reset thrust exceeds a rotor's maximum: clipping would break equilibrium")
        span = (self._max_w[:5] - self._min_w[:5]).clamp_min(1e-6).unsqueeze(0)
        rotor_norm = 2.0 * (omega - self._min_w[:5].unsqueeze(0)) / span - 1.0
        surface_trim = torch.zeros((V0.shape[0], 3), dtype=V0.dtype, device=self.device)
        surface_trim[:, 0] = delta_e
        return alpha_trim, surface_trim, rotor_norm

    def _trim_rotor_norm(self, V0: torch.Tensor, alpha_trim: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper; resets also need the returned alpha/elevator."""
        return self._trim_from_speed(V0, alpha_trim)[2]
