"""
| File: glider_aerodynamics.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Batched aerodynamic model for the EasyGlider vehicles (GliderBatch, ShuttleGliderBatch2).
|
| Aerodynamic coefficients obtained from: https://github.com/Filipe-Cavalheiro/PX4-gazebo-models/blob/main/tools/avl_automation/easy_glider4/easy_glider4.sdf
|
| The SDF coefficients are AVL derivatives written for the Gazebo AdvancedLiftDrag plugin (body FRD,
| angle derivatives per radian, control derivatives per degree). They are converted here to FLU, per radian:
|   - y and z axes flip: CY, Cm, Cn and the rates q, r change sign (CYb, CYp, Cnb, Cnp, Cm0, Cma).
|   - The CL_* and CD_* rate/control slots of the SDF are filled by the generator with the AVL body-axis
|     CZ_* and CX_* derivatives (positive down/forward). The lift and drag derivatives are therefore
|     -CZ and -CX, which gives CLq(FRD) = +16.41 and CDq(FRD) = -0.55, and CLde = +0.00196/deg for a
|     trailing-edge-down elevator that also gives a nose-down moment (Cmde = +0.00839/deg in FLU).
|   - Coefficient values are kept exactly as in the SDF; only signs and units are converted.
|   - The reference chord is the AVL Cref = S/b = 0.2311 m (easy_glider4.avl). The generator writes 2/3 S/b
|     into the SDF 'mac' while AVL normalises the moments and rates with Cref, so Cref is used here.
| Known plugin issue (not reproduced here): the Gazebo plugin source assigns "Cem = Cemb * beta" instead of
| accumulating, which discards Cem0 + Cema * alpha. This model keeps the intended longitudinal moment.
| Model terms outside this module (rotor drag, post-stall drag blend, cross derivatives) are not modelled.
|
| Reference point: the AVL reference point (origin of easy_glider4.avl, at the nose) about which the force and
| moment are given. It is placed at the nose (propeller station) of the asset, in the wing plane (see
| EASYGLIDER_REFERENCE_FROM_COM), not at the SDF base_link, which is the front of the fuselage mesh.
| Both the PhysX linear velocity and the force application point of the /body prim
| are its centre of mass, so r_com (CoM to reference point) is the only lever arm: the air velocity is
| evaluated at the reference point as v_com + w x r_com, and the model returns the force F with the moment
| about the body CoM tau = M_ref + r_com x F. Do not add the lever arm again outside this module.
|
| The model contains aerodynamics only. Propeller thrust and reaction torque are handled by the Glider
| thruster model used by GliderBatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import torch


__all__ = ["DEG", "GliderGeometry", "GliderCoefficients", "GliderAerodynamicsBatch"]

# Rad to deg conversion factor
DEG = math.pi / 180.0

# base_link origin of the source model of easyglider.usd (VioLET easyglider.sdf.jinja, not part of the AVL work
# cited above) relative to the asset centre of mass, FLU [m]. Not the AVL reference point: base_link is the front
# of the fuselage mesh (without nose cone), 0.0546 m above the CoM.
# easyglider.usd: mesh frame "body_link" (0.29769, 0, -0.01545) lies 0.07 m below base_link (jinja mesh
# pose z = -0.07), and the puller position relative to base_link, (0.045, 0, -0.075), reproduces the
# USDA puller position (0.34269, 0, -0.02045).
EASYGLIDER_REFERENCE_FROM_CAD_COM = (0.29769228536660874, 0.0, 0.05454581870473936)

# AVL reference point (origin of easy_glider4.avl, at the nose): the wing root leading edge lies 0.275 m aft
# of it (wing TRANSLATE) and the wing chord plane is z = 0.
EASYGLIDER_AVL_WING_LE_X = 0.275
# Propeller station of easyglider.usd: base_link + 0.045 m (puller pose), i.e. the nose of the asset
# (nose cone tip 0.343 m ahead of the CoM).
EASYGLIDER_PROPELLER_STATION_X = EASYGLIDER_REFERENCE_FROM_CAD_COM[0] + 0.045
# Wing root chord plane above the CoM in easyglider.usd (wing root section spans z = +0.004..+0.031 m).
EASYGLIDER_WING_PLANE_ABOVE_COM = 0.016

# Longitudinal arm used by the model: distance from the centre of gravity (aft) to the AVL reference point.
# Modelling choice (the AVL geometry and the asset share no origin): the reference point is placed at the nose
# (propeller station) of the asset. This puts the CG 67.7 mm aft of the AVL wing leading edge;
# the manufacturer recommends 70 mm (EasyGlider 4 manual), the asset CoM lies 77 mm aft of the mesh leading edge.
# With the AVL reference chord (see GliderGeometry.c) the static margin is +31 % of Cref and level flight
# trims inside the datasheet elevator range from ~12 to 28 m/s.
EASYGLIDER_REFERENCE_ARM_X = EASYGLIDER_PROPELLER_STATION_X

# Reference point relative to the CG used by the model, FLU [m]
EASYGLIDER_REFERENCE_FROM_COM = (EASYGLIDER_REFERENCE_ARM_X, 0.0, EASYGLIDER_WING_PLANE_ABOVE_COM)


# =============================================================================
# Geometry and aerodynamic coefficients
# =============================================================================

@dataclass(frozen=True)
class GliderGeometry:
    """Glider aerodynamic reference quantities."""

    rho: float = 1.2041                 # air density [kg/m^3]
    S: float = 0.416                    # wing reference area [m^2]
    b: float = 1.800                    # wingspan [m]
    # Reference chord used by AVL to non-dimensionalise the pitching moment and the pitch rate: Cref = S/b
    # (easy_glider4.avl). The SDF 'mac' (0.1541 = 2/3 S/b, written by the generator) is inconsistent with it.
    c: float = 0.416 / 1.800            # reference chord [m]
    AR: float = 1.800**2 / 0.416        # aspect ratio [-]
    e: float = 0.8523                   # Oswald efficiency factor [-]

    # Vector from the /body centre of mass to the aerodynamic reference point (AVL origin), FLU [m].
    # The CoM is the point where PhysX applies forces (positions=None) and where it reports the linear
    # velocity, so this single vector transfers both the velocity and the moment. The standalone
    # EasyGlider /body origin and CoM coincide (offset ~1e-8 m).
    r_com: tuple[float, float, float] = EASYGLIDER_REFERENCE_FROM_COM


@dataclass(frozen=True)
class GliderCoefficients:
    """Aerodynamic coefficient set in FLU, with angular derivatives per radian."""

    # Longitudinal coefficients
    CL0: float = 0.36986
    CLa: float = 3.679875
    CLq: float = -16.413084  # SDF CL slot holds CZq = -16.413 (down positive); FRD +16.413, FLU -16.413
    CLde: float = +0.001961 / DEG

    CD0: float = +0.007590
    CDq: float = +0.549541   # SDF CD slot holds CXq = +0.5495 (forward positive); FRD -0.5495, FLU +0.5495
    CDde: float = 0.000044 / DEG

    # Pitch moment about the aerodynamic reference point (SDF Cem0/Cema with FRD -> FLU sign flip)
    Cm0: float = +0.57912
    Cma: float = +6.591829
    Cmq: float = -35.807747
    Cmde: float = +0.008387 / DEG

    # Lateral-directional coefficients
    CY0: float = 0.0
    CYb: float = +0.187754
    CYp: float = -0.032506
    CYr: float = +0.222424
    CYda: float = +2.0 * 0.000107 / DEG # 0.0
    CYdr: float = -0.001237 / DEG

    Cl0: float = 0.0
    Clb: float = -0.062425
    Clp: float = -0.392788
    Clr: float = -0.137367
    Clda: float = +2.0 * 0.001596 / DEG
    Cldr: float = +0.000142 / DEG

    Cn0: float = 0.0
    Cnb: float = -0.108424
    Cnp: float = +0.033202
    Cnr: float = -0.129973
    Cnda: float = 0.0
    Cndr: float = +0.000711 / DEG

    # Nonlinear lift / stall blending
    alpha0: float = 0.3391428111       # stall/blending angle [rad]
    M_sig: float = 15.0 #50.0


# =============================================================================
# Batched Glider aerodynamic model
# =============================================================================

class GliderAerodynamicsBatch:
    """Vectorized Glider aerodynamic model for 'GliderBatch'."""

    EPS_V = 0.5
    ALPHA_LIM = math.pi / 2.0

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        n_vehicles: int = 1,
        device: str = "cpu",
    ) -> None:
        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

        if device is None:
            from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
            device = PegasusInterface()._world_settings["device"]

        self.device = torch.device(device)
        self.n_vehicles = int(n_vehicles)
        
        cfg = dict(config or {})

        geometry_cfg = cfg.get("geometry", {})
        coefficients_cfg = cfg.get("coefficients", {})
        
        # -------------------------------------------------------------
        # Geometry
        # -------------------------------------------------------------
        if isinstance(geometry_cfg, GliderGeometry):
            self.G = geometry_cfg
        else:
            self.G = GliderGeometry(**dict(geometry_cfg))

        # -------------------------------------------------------------
        # Aerodynamic coefficients
        # -------------------------------------------------------------
        if isinstance(coefficients_cfg, GliderCoefficients):
            self.C = coefficients_cfg
        else:
            self.C = GliderCoefficients(**dict(coefficients_cfg))

        # -------------------------------------------------------------
        # Model options
        # -------------------------------------------------------------
        self.eval_at_ref = bool(cfg.get("eval_at_ref", True))

        self._r_com = torch.tensor(self.G.r_com, dtype=torch.float32, device=self.device)

        # By default there is no wind, so the vehicle body velocity is also the air-relative velocity.
        self._wind_body = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)

        # Latest values, useful for logging and debugging.
        self._force = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)
        
        self._torque = torch.zeros_like(self._force)
        self._Va = torch.zeros(self.n_vehicles, dtype=torch.float32, device=self.device)
        self._alpha = torch.zeros_like(self._Va)
        self._beta = torch.zeros_like(self._Va)
        self._lift = torch.zeros_like(self._Va)
        self._drag = torch.zeros_like(self._Va)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, vehicle) -> None:
        """Synchronize batch size/device with the owning GliderBatch``.

        GliderBatch.start() calls this method automatically when present.
        """
        n_vehicles = int(vehicle.n_vehicles)
        device = torch.device(vehicle.device)

        if n_vehicles != self.n_vehicles or device != self.device:
            self.n_vehicles = n_vehicles
            self.device = device
            self._r_com = self._r_com.to(device=self.device, dtype=torch.float32)
            self._wind_body = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)
            self._force = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)
            self._torque = torch.zeros_like(self._force)
            self._Va = torch.zeros(self.n_vehicles, dtype=torch.float32, device=self.device)
            self._alpha = torch.zeros_like(self._Va)
            self._beta = torch.zeros_like(self._Va)
            self._lift = torch.zeros_like(self._Va)
            self._drag = torch.zeros_like(self._Va)

    # -------------------------------------------------------------------------
    # Wind / air-relative velocity
    # -------------------------------------------------------------------------

    def set_wind_body(self, wind_body) -> None:
        """Set wind velocity expressed in each vehicle's FLU body frame.

        wind_body: Either ``[wx, wy, wz]`` (broadcast to all vehicles) or a tensor with
            shape ``(n_vehicles, 3)`` [m/s]. Air-relative velocity is evaluated as
            ``v_air_b = v_body - wind_body``.
        """
        wind = torch.as_tensor(wind_body, dtype=torch.float32, device=self.device)

        if wind.ndim == 1:
            if wind.shape != (3,):
                raise ValueError("wind_body must contain exactly 3 components")
            wind = wind.unsqueeze(0).expand(self.n_vehicles, -1)

        if wind.shape != (self.n_vehicles, 3):
            raise ValueError(
                f"wind_body must have shape ({self.n_vehicles}, 3), "
                f"got {tuple(wind.shape)}"
            )

        self._wind_body = wind.clone()

    def clear_wind(self) -> None:
        """Set the wind velocity to zero for all vehicles."""
        self._wind_body.zero_()


    # -------------------------------------------------------------------------
    # Aerodynamic helpers
    # -------------------------------------------------------------------------

    def airdata_from_state(self, linear_body_velocity, angular_body_velocity):
        """
        Compute airspeed 'Va', its regularized lower bound 'Vs' used to
        avoid low-speed singularities, angle of attack 'alpha' and side slip angle 'beta'.
        """
        
        # Calculate the air-relative velocity in the body frame
        v_air_b = linear_body_velocity - self._wind_body

        # Evaluate the air-relative velocity at the aerodynamic reference point rather than at the CoM.
        # For a rigid body:
        #   v_ref = v_CoM + omega x r_com, where all vectors are expressed in {B}.
        if self.eval_at_ref:
            r_com = self._r_com.unsqueeze(0).expand(self.n_vehicles, -1)
            v_air_b = v_air_b + torch.cross(angular_body_velocity, r_com, dim=1)
        
        u = v_air_b[:, 0]; v = v_air_b[:, 1]; w = v_air_b[:, 2]

        # Compute the airspeed magnitude 'Va' and its corresponding lower bound 'Vs'
        Va = torch.linalg.vector_norm(v_air_b, dim=1)
        Vs = torch.clamp(Va, min=self.EPS_V)

        # Compute the angle of attack and clamp it to the allowed range
        alpha = torch.atan2(-w, u)
        alpha = torch.clamp(alpha, min=-self.ALPHA_LIM, max=self.ALPHA_LIM)

        # Compute the sideslip angle
        beta_arg = torch.clamp(v / Vs, min=-1.0, max=1.0)
        beta = -torch.asin(beta_arg)

        return Va, Vs, alpha, beta
    
    
    def CL_CD(self, alpha: torch.Tensor, delta_e: torch.Tensor):
        """
        Compute nonlinear lift and drag coefficients for a batch of states,
        considering the angle of attack 'alpha' and the elevator deflection 'delta_e'.
        """
        C = self.C

        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
        delta_e = torch.as_tensor(delta_e, dtype=torch.float32, device=self.device)

        # Compute the Beard/McLain stall-blending function.
        # Clamp exponent arguments to avoid float32 overflow far outside the physically relevant alpha interval.
        x1 = torch.clamp(-C.M_sig * (alpha - C.alpha0), min=-80.0, max=80.0)
        x2 = torch.clamp(+C.M_sig * (alpha + C.alpha0), min=-80.0, max=80.0)
        e1 = torch.exp(x1)
        e2 = torch.exp(x2)
        sigma = (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))

        # Linear lift coefficient used in the pre-stall regime.
        CL_linear = C.CL0 + C.CLa * alpha
        
        # Flat-plate approximation used to model lift beyond stall.
        CL_flat_plate = 2.0 * torch.sign(alpha) * torch.sin(alpha).square() * torch.cos(alpha)
        
        # Blend the linear and post-stall lift models using the stall function sigma.
        CL = (1.0 - sigma) * CL_linear + sigma * CL_flat_plate

        # Compute the drag coefficient as the sum of parasitic drag, induced drag and the elevator-induced drag contribution.
        CD = C.CD0 + CL_linear.square() / (math.pi * self.G.e * self.G.AR) + C.CDde * torch.abs(delta_e)

        return CL, CD

    # -------------------------------------------------------------------------
    # Main aerodynamic equations
    # -------------------------------------------------------------------------

    def forces_moments(self, linear_body_velocity, angular_body_velocity, control_surfaces):
        """
        Compute the batched aerodynamic force and the moment about the body CoM.

        Parameters
        linear_body_velocity: Linear Velocity in the body frame, shape (N, 3).
        angular_body_velocity: Angular Velocity int the body frame, shape (N, 3).
        control_surfaces: [delta_e, delta_a, delta_r] in rad, shape (N, 3).

        Returns
        tuple[torch.Tensor, torch.Tensor]: (F_aero, tau_aero_com), each with shape (N, 3).
        """
        G = self.G
        C = self.C

        p = angular_body_velocity[:, 0]
        q = angular_body_velocity[:, 1]
        r = angular_body_velocity[:, 2]

        de = control_surfaces[:, 0]
        da = control_surfaces[:, 1]
        dr = control_surfaces[:, 2]

        # Compute airspeed and aerodynamic angles from the current state
        Va, Vs, alpha, beta = self.airdata_from_state(linear_body_velocity, angular_body_velocity)

        # Compute the dynamic pressure
        qbar = 0.5 * G.rho * Va.square()
        
        # Compute the dynamic pressure scaled by the wing reference area
        qS = qbar * G.S

        # Compute the nondimensional body angular rates
        p_hat = p * G.b / (2.0 * Vs)
        q_hat = q * G.c / (2.0 * Vs)
        r_hat = r * G.b / (2.0 * Vs)

        # Compute the nonlinear lift and drag coefficients
        CL, CD = self.CL_CD(alpha, de)

        # Compute the longitudinal aerodynamic lift and drag forces
        lift = qS * (CL + C.CLq * q_hat + C.CLde * de)
        drag = qS * (CD + C.CDq * q_hat)
        drag = torch.clamp(drag, min=0.0)

        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        # Project lift and drag into the FLU body frame
        Fx = -drag * ca + lift * sa
        Fz = +drag * sa + lift * ca
        
        # Compute the lateral aerodynamic force in the FLU body frame
        Fy = qS * (C.CY0 + C.CYb * beta + C.CYp * p_hat + C.CYr * r_hat + C.CYda * da + C.CYdr * dr)

        F_aero = torch.stack((Fx, Fy, Fz), dim=1)

        # Aerodynamic moments about the aerodynamic reference point.
        tau_x = qS * G.b * (C.Cl0 + C.Clb * beta + C.Clp * p_hat + C.Clr * r_hat + C.Clda * da + C.Cldr * dr)
        tau_y = qS * G.c * (C.Cm0 + C.Cma * alpha + C.Cmq * q_hat + C.Cmde * de)
        tau_z = qS * G.b * (C.Cn0 + C.Cnb * beta + C.Cnp * p_hat + C.Cnr * r_hat + C.Cnda * da + C.Cndr * dr)

        tau_ref = torch.stack((tau_x, tau_y, tau_z), dim=1)

        # Transfer the moment to the point where PhysX applies the force (body CoM):
        # tau = tau_ref + r_com x F_aero, with r_com from the CoM to the reference point
        r_com = self._r_com.unsqueeze(0).expand(self.n_vehicles, -1)
        tau_com = tau_ref + torch.cross(r_com, F_aero, dim=1)

        # Cache useful diagnostic quantities.
        self._force = F_aero
        self._torque = tau_com
        self._Va = Va
        self._alpha = alpha
        self._beta = beta
        self._lift = lift
        self._drag = drag

        return F_aero, tau_com

    def update(self, state, control_surfaces, dt: float):
        """Update the aerodynamic wrench from the current StateBatch."""

        linear_body_velocity = torch.as_tensor(state.linear_body_velocity, dtype=torch.float32, device=self.device)
        angular_body_velocity = torch.as_tensor(state.angular_velocity, dtype=torch.float32, device=self.device)

        return self.forces_moments(linear_body_velocity=linear_body_velocity, angular_body_velocity=angular_body_velocity, control_surfaces=control_surfaces)

    # -------------------------------------------------------------------------
    # Diagnostics / public properties
    # -------------------------------------------------------------------------

    @property
    def force(self) -> torch.Tensor:
        """Latest aerodynamic force in FLU, shape ``(N,3)`` [N]."""
        return self._force

    @property
    def torque(self) -> torch.Tensor:
        """Latest aerodynamic moment about the body CoM, shape ``(N,3)`` [N m]."""
        return self._torque

    @property
    def airspeed(self) -> torch.Tensor:
        """Latest airspeed ``Va``, shape ``(N,)`` [m/s]."""
        return self._Va

    @property
    def alpha(self) -> torch.Tensor:
        """Latest angle of attack, shape ``(N,)`` [rad]."""
        return self._alpha

    @property
    def beta(self) -> torch.Tensor:
        """Latest sideslip angle, shape ``(N,)`` [rad]."""
        return self._beta

    @property
    def lift(self) -> torch.Tensor:
        """Latest lift magnitude, shape ``(N,)`` [N]."""
        return self._lift

    @property
    def drag(self) -> torch.Tensor:
        """Latest drag magnitude, shape ``(N,)`` [N]."""
        return self._drag

    @property
    def r_com(self) -> torch.Tensor:
        """/body CoM-to-aerodynamic-reference vector in FLU [m]."""
        return self._r_com
