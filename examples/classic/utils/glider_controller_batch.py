"""
| File: glider_autopilot_backend.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description:
|   Batched fixed-wing autopilot backend for the Pegasus EasyGlider.
|
| Controller structure follows Beard & McLain, Small Unmanned Aircraft:
|
|   Chapter 6 - Successive loop closure
|       course -> commanded roll -> aileron
|       altitude -> commanded pitch -> elevator
|       airspeed -> throttle
|       yaw-rate washout damper -> rudder
|
|   Chapter 10 - Orbit following
|       orbit geometry -> commanded course
|       coordinated-turn feedforward -> commanded roll
|
| Pegasus actuator output:
|       [delta_e, delta_a, delta_r, Omega_p]
|       [rad,     rad,     rad,     rad/s]
|
| Coordinate-convention adaptation
| --------------------------------
| Book:
|   inertial NED
|   body FRD
|   course/heading measured clockwise from North
|   positive pitch = nose-up
|
| Simulator:
|   inertial ENU: x=East, y=North, z=Up
|   body FLU: x=Forward, y=Left, z=Up
|   positive FLU pitch = nose-down
|
| Internally this backend reconstructs the book variables:
|   north = y_ENU
|   east  = x_ENU
|   h     = z_ENU
|   chi   = atan2(v_east, v_north)
|   psi   = heading measured clockwise from North
|   theta_book = -theta_FLU
|   q_book     = -q_FLU
|
| Roll p keeps the same physical sign (positive = right wing down).
|
| Notes
| -----
| 1. The default gains below are conservative initial EasyGlider gains.
|    They implement the textbook architecture, but they are NOT claimed to
|    be the analytically optimal gains from the exact EasyGlider linear model.
|    They should be tuned/identified after the first closed-loop tests.
|
| 2. The book commands normalized throttle delta_t. The Pegasus thrust model
|    currently receives propeller angular velocity Omega_p directly. Therefore:
|
|       Omega_p = delta_t * Omega_max
|
|    This mapping can later be replaced by a motor/ESC model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal
import math

import torch

from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.transforms import quaternion_to_matrix


ControlMode = Literal["hold", "orbit"]


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class GliderAutopilotBackend(Backend):
    """
    Batched EasyGlider autopilot and orbit-following backend.

    The backend uses the true simulated state, as in the Chapter 6 simulator
    design stage of Beard & McLain. Sensor/state-estimation blocks can be added
    later without changing the controller structure.
    """

    def __init__(
        self,
        n_vehicles: int = 1,
        config: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(config=None)

        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

        cfg = dict(config or {})

        self._n_vehicles = int(n_vehicles)
        self._device_requested = device
        self._device: torch.device | None = (
            torch.device(device) if device is not None else None
        )

        # ------------------------------------------------------------------
        # EasyGlider trim point
        # ------------------------------------------------------------------
        trim = dict(cfg.get("trim", {}))

        self._Va_trim = float(trim.get("airspeed", 11.5))
        self._theta_trim = math.radians(
            float(trim.get("pitch_up_deg", 0.98152))
        )

        self._delta_e_trim = math.radians(
            float(trim.get("delta_e_deg", 2.39950))
        )
        self._delta_a_trim = math.radians(
            float(trim.get("delta_a_deg", 0.10682))
        )
        self._delta_r_trim = math.radians(
            float(trim.get("delta_r_deg", 0.0))
        )
        self._omega_trim = float(trim.get("omega_p", 364.509))

        # ------------------------------------------------------------------
        # Actuator / command limits
        # ------------------------------------------------------------------
        limits = dict(cfg.get("limits", {}))

        self._delta_e_max = math.radians(
            float(limits.get("delta_e_deg", 20.0))
        )
        self._delta_a_max = math.radians(
            float(limits.get("delta_a_deg", 20.0))
        )
        self._delta_r_max = math.radians(
            float(limits.get("delta_r_deg", 20.0))
        )

        self._phi_command_max = math.radians(
            float(limits.get("roll_command_deg", 35.0))
        )
        self._theta_command_min = math.radians(
            float(limits.get("pitch_command_min_deg", -15.0))
        )
        self._theta_command_max = math.radians(
            float(limits.get("pitch_command_max_deg", 20.0))
        )

        self._omega_min = float(limits.get("omega_min", 0.0))
        self._omega_max = float(limits.get("omega_max", 1100.0))

        if self._omega_max <= self._omega_min:
            raise ValueError("omega_max must be greater than omega_min")

        self._throttle_trim = self._omega_trim / self._omega_max

        # ------------------------------------------------------------------
        # Chapter 6: successive-loop-closure gains
        # ------------------------------------------------------------------
        gains = dict(cfg.get("gains", {}))

        # Roll attitude hold:
        #   delta_a = delta_a* + kp_phi(phi_c - phi) - kd_phi p
        self.kp_phi = float(gains.get("kp_phi", 0.65))
        self.kd_phi = float(gains.get("kd_phi", 0.03))

        # Course hold:
        #   phi_c = kp_chi(chi_c - chi) + ki_chi integral(error)
        self.kp_chi = float(gains.get("kp_chi", 2.00))
        self.ki_chi = float(gains.get("ki_chi", 0.45))

        # Pitch attitude hold.
        #
        # The EasyGlider elevator convention is delta_e > 0 = trailing edge
        # down = nose-down moment. Since theta_book > 0 means nose-up, the
        # corresponding pitch-loop gains are negative.
        self.kp_theta = float(gains.get("kp_theta", -0.55))
        self.kd_theta = float(gains.get("kd_theta", -0.10))

        # Altitude hold:
        #   theta_c = theta_trim + kp_h(h_c-h) + ki_h integral(error)
        self.kp_h = float(gains.get("kp_h", 0.045))
        self.ki_h = float(gains.get("ki_h", 0.0075))

        # Airspeed hold using normalized throttle:
        #   delta_t = delta_t* + kp_V(Va_c-Va) + ki_V integral(error)
        self.kp_V = float(gains.get("kp_V", 0.080))
        self.ki_V = float(gains.get("ki_V", 0.020))

        # Yaw damper.
        self.k_r = float(gains.get("k_r", 0.15))
        self.p_wo = float(gains.get("p_wo", 0.50))

        # ------------------------------------------------------------------
        # Chapter 10: orbit-following gain
        # ------------------------------------------------------------------
        orbit_cfg = dict(cfg.get("orbit", {}))

        self._k_orbit = float(orbit_cfg.get("k_orbit", 4.0))
        self._gravity = float(cfg.get("gravity", 9.80665))

        # ------------------------------------------------------------------
        # Integral limits
        # ------------------------------------------------------------------
        integrator_limits = dict(cfg.get("integrator_limits", {}))

        self._chi_int_max = float(
            integrator_limits.get("course", math.radians(60.0))
        )
        self._h_int_max = float(
            integrator_limits.get("altitude", 30.0)
        )
        self._Va_int_max = float(
            integrator_limits.get("airspeed", 20.0)
        )

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self._state = None
        self._received_first_state = False
        self._started = False

        self._mode: ControlMode = "hold"

        # Commands are stored as Python values until start() determines device.
        self._pending_Va_command: Any = self._Va_trim
        self._pending_h_command: Any = None
        self._pending_chi_command: Any = None

        self._pending_orbit_center_enu: Any = (0.0, 0.0, 60.0)
        self._pending_orbit_radius: Any = 50.0
        self._pending_orbit_direction: Any = 1.0
        self._orbit_altitude_override: Any = None
        self._orbit_airspeed_override: Any = None

        # Torch buffers allocated in start().
        self._input_reference = None

        self._Va_command = None
        self._h_command = None
        self._chi_command = None

        self._orbit_center_enu = None
        self._orbit_radius = None
        self._orbit_direction = None

        self._integrator_chi = None
        self._integrator_h = None
        self._integrator_Va = None
        self._washout_state = None

        # Diagnostics
        self._phi = None
        self._theta = None
        self._psi = None
        self._chi = None
        self._Va = None
        self._Vg = None
        self._phi_command = None
        self._theta_command = None
        self._phi_feedforward = None
        self._distance_to_orbit = None

    # ==================================================================
    # Backend lifecycle
    # ==================================================================

    def initialize(self, vehicle) -> None:
        self._vehicle = vehicle

    def start(self) -> None:
        self._n_vehicles = int(self.vehicle.n_vehicles)
        self._device = torch.device(
            self.vehicle.device
            if self.vehicle.device is not None
            else (self._device_requested or "cpu")
        )

        n = self._n_vehicles
        device = self._device

        self._input_reference = torch.zeros(
            (n, 4), dtype=torch.float32, device=device
        )

        self._integrator_chi = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self._integrator_h = torch.zeros_like(self._integrator_chi)
        self._integrator_Va = torch.zeros_like(self._integrator_chi)
        self._washout_state = torch.zeros_like(self._integrator_chi)

        self._Va_command = self._broadcast_scalar(
            self._pending_Va_command, "airspeed_command"
        )

        # If no explicit altitude/course command was supplied, start() will
        # initialize them from the first state received by update().
        self._h_command = None
        self._chi_command = None

        self._orbit_center_enu = self._broadcast_vector3(
            self._pending_orbit_center_enu, "orbit_center_enu"
        )
        self._orbit_radius = self._broadcast_scalar(
            self._pending_orbit_radius, "orbit_radius"
        )
        self._orbit_direction = self._broadcast_scalar(
            self._pending_orbit_direction, "orbit_direction"
        )

        if torch.any(self._orbit_radius <= 0.0):
            raise ValueError("Orbit radius must be greater than zero")

        self._orbit_direction = torch.where(
            self._orbit_direction >= 0.0,
            torch.ones_like(self._orbit_direction),
            -torch.ones_like(self._orbit_direction),
        )

        self._set_trim_output()
        self._started = True

    def stop(self) -> None:
        return

    def reset(self) -> None:
        if self._integrator_chi is not None:
            self._integrator_chi.zero_()
            self._integrator_h.zero_()
            self._integrator_Va.zero_()
            self._washout_state.zero_()

        self._received_first_state = False

        if self._input_reference is not None:
            self._set_trim_output()

    # ==================================================================
    # Backend interface
    # ==================================================================

    def update_state(self, state) -> None:
        self._state = state
        self._received_first_state = True

    def set_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ) -> None:
        """Synchronize backend state after an external VehicleBatch reset.

        ``VehicleBatch.set_state_batch`` calls this optional method even though
        it is not part of the abstract ``Backend`` interface.  The vehicle has
        already updated its shared StateBatch before this callback, therefore
        the safest behaviour for this controller is simply to bind to the
        vehicle state and mark it as valid.
        """

        if self.vehicle is not None:
            self._state = self.vehicle.state

        self._received_first_state = self._state is not None

        # Reset controller memories for the environments that were externally
        # reset. This avoids carrying integral/washout state across resets.
        if not self._started or env_ids.numel() == 0:
            return

        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self._device
        )

        self._integrator_chi[ids] = 0.0
        self._integrator_h[ids] = 0.0
        self._integrator_Va[ids] = 0.0
        self._washout_state[ids] = 0.0

    def update_sensor(self, sensor_type: str, data) -> None:
        # Chapter-6 simulator implementation uses true state feedback.
        # Sensor-based state estimation can replace update_state later.
        return

    def update_graphical_sensor(self, sensor_type: str, data) -> None:
        return

    def input_reference(self) -> torch.Tensor:
        if self._input_reference is None:
            # This should only occur before start().
            return torch.zeros((self._n_vehicles, 4), dtype=torch.float32)
        return self._input_reference

    def update(self, dt: float) -> None:
        if (
            not self._started
            or not self._received_first_state
            or self._state is None
            or dt <= 0.0
        ):
            return

        # --------------------------------------------------------------
        # Reconstruct the textbook flight variables from ENU/FLU state.
        # --------------------------------------------------------------
        flight = self._extract_flight_state()

        phi = flight["phi"]
        theta = flight["theta"]
        psi = flight["psi"]
        chi = flight["chi"]

        p = flight["p"]
        q = flight["q"]
        r_flu = flight["r_flu"]

        h = flight["h"]
        Va = flight["Va"]
        Vg = flight["Vg"]

        self._phi = phi
        self._theta = theta
        self._psi = psi
        self._chi = chi
        self._Va = Va
        self._Vg = Vg

        # Initialize hold commands from the actual state on the first usable
        # update if the user did not explicitly provide them.
        if self._h_command is None:
            if self._pending_h_command is None:
                self._h_command = h.detach().clone()
            else:
                self._h_command = self._broadcast_scalar(
                    self._pending_h_command, "altitude_command"
                )

        if self._chi_command is None:
            if self._pending_chi_command is None:
                self._chi_command = chi.detach().clone()
            else:
                self._chi_command = self._broadcast_scalar(
                    self._pending_chi_command, "course_command"
                )

        # --------------------------------------------------------------
        # Guidance / outer commands
        # --------------------------------------------------------------
        if self._mode == "orbit":
            chi_c, phi_ff = self._orbit_guidance(
                chi=chi,
                psi=psi,
                Vg=Vg,
            )

            self._chi_command = chi_c
            self._phi_feedforward = phi_ff

            if self._orbit_altitude_override is not None:
                self._h_command = self._broadcast_scalar(
                    self._orbit_altitude_override,
                    "orbit_altitude",
                )
            else:
                # Orbit center z is the desired altitude in ENU.
                self._h_command = self._orbit_center_enu[:, 2]

            if self._orbit_airspeed_override is not None:
                self._Va_command = self._broadcast_scalar(
                    self._orbit_airspeed_override,
                    "orbit_airspeed",
                )

        else:
            phi_ff = torch.zeros_like(phi)
            self._phi_feedforward = phi_ff

        # ==============================================================
        # Chapter 6 - lateral autopilot
        # ==============================================================

        # Course PI -> commanded roll
        e_chi = _wrap_pi(self._chi_command - chi)

        self._integrator_chi = torch.clamp(
            self._integrator_chi + e_chi * float(dt),
            min=-self._chi_int_max,
            max=+self._chi_int_max,
        )

        phi_feedback = (
            self.kp_chi * e_chi
            + self.ki_chi * self._integrator_chi
        )

        phi_c = phi_feedback + phi_ff

        phi_c = torch.clamp(
            phi_c,
            min=-self._phi_command_max,
            max=+self._phi_command_max,
        )

        # Roll PD -> aileron
        delta_a = (
            self._delta_a_trim
            + self.kp_phi * (phi_c - phi)
            - self.kd_phi * p
        )

        delta_a = torch.clamp(
            delta_a,
            min=-self._delta_a_max,
            max=+self._delta_a_max,
        )

        # Yaw damper using washout filter.
        #
        # H_wo(s) = s / (s + p_wo)
        #
        # If xdot = -p_wo*x + r, then:
        #     r_hp = r - p_wo*x
        #
        # We use r_FLU directly here because the EasyGlider rudder sign is
        # defined in the FLU aerodynamic model. The minus sign implements
        # negative yaw-rate feedback for that convention.
        self._washout_state = (
            self._washout_state
            + float(dt)
            * (
                -self.p_wo * self._washout_state
                + r_flu
            )
        )

        r_highpass = (
            r_flu
            - self.p_wo * self._washout_state
        )

        delta_r = (
            self._delta_r_trim
            - self.k_r * r_highpass
        )

        delta_r = torch.clamp(
            delta_r,
            min=-self._delta_r_max,
            max=+self._delta_r_max,
        )

        # ==============================================================
        # Chapter 6 - longitudinal autopilot
        # ==============================================================

        # Altitude PI -> commanded pitch (book sign: positive nose-up).
        e_h = self._h_command - h

        self._integrator_h = torch.clamp(
            self._integrator_h + e_h * float(dt),
            min=-self._h_int_max,
            max=+self._h_int_max,
        )

        theta_c = (
            self._theta_trim
            + self.kp_h * e_h
            + self.ki_h * self._integrator_h
        )

        theta_c = torch.clamp(
            theta_c,
            min=self._theta_command_min,
            max=self._theta_command_max,
        )

        # Pitch PD -> elevator.
        delta_e = (
            self._delta_e_trim
            + self.kp_theta * (theta_c - theta)
            - self.kd_theta * q
        )

        delta_e = torch.clamp(
            delta_e,
            min=-self._delta_e_max,
            max=+self._delta_e_max,
        )

        # Airspeed PI -> normalized throttle -> Omega_p.
        e_Va = self._Va_command - Va

        self._integrator_Va = torch.clamp(
            self._integrator_Va + e_Va * float(dt),
            min=-self._Va_int_max,
            max=+self._Va_int_max,
        )

        delta_t = (
            self._throttle_trim
            + self.kp_V * e_Va
            + self.ki_V * self._integrator_Va
        )

        delta_t = torch.clamp(delta_t, 0.0, 1.0)

        omega_p = torch.clamp(
            delta_t * self._omega_max,
            min=self._omega_min,
            max=self._omega_max,
        )

        # --------------------------------------------------------------
        # Pegasus actuator vector
        # --------------------------------------------------------------
        self._input_reference[:, 0] = delta_e
        self._input_reference[:, 1] = delta_a
        self._input_reference[:, 2] = delta_r
        self._input_reference[:, 3] = omega_p

        self._phi_command = phi_c
        self._theta_command = theta_c

    # ==================================================================
    # User command API
    # ==================================================================

    def set_autopilot_command(
        self,
        *,
        airspeed: float | Sequence[float] | torch.Tensor | None = None,
        altitude: float | Sequence[float] | torch.Tensor | None = None,
        course: float | Sequence[float] | torch.Tensor | None = None,
        course_deg: float | Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """
        Command the Chapter-6 autopilot directly.

        Parameters
        ----------
        airspeed:
            Commanded airspeed [m/s].

        altitude:
            Commanded altitude z_ENU [m].

        course:
            Commanded course [rad], measured CLOCKWISE from North.

        course_deg:
            Same as course, in degrees. Use either course or course_deg.
        """

        if course is not None and course_deg is not None:
            raise ValueError("Use either course or course_deg, not both")

        self._mode = "hold"

        if airspeed is not None:
            self._pending_Va_command = airspeed
            if self._started:
                self._Va_command = self._broadcast_scalar(
                    airspeed, "airspeed_command"
                )

        if altitude is not None:
            self._pending_h_command = altitude
            if self._started:
                self._h_command = self._broadcast_scalar(
                    altitude, "altitude_command"
                )

        if course_deg is not None:
            course = self._degrees_to_radians(course_deg)

        if course is not None:
            self._pending_chi_command = course
            if self._started:
                self._chi_command = self._broadcast_scalar(
                    course, "course_command"
                )

    def set_orbit(
        self,
        *,
        center_enu: Sequence[float] | torch.Tensor,
        radius: float | Sequence[float] | torch.Tensor,
        direction: int | float | Sequence[float] | torch.Tensor = 1,
        altitude: float | Sequence[float] | torch.Tensor | None = None,
        airspeed: float | Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """
        Enable Chapter-10 circular-orbit following.

        Parameters
        ----------
        center_enu:
            Orbit center [East, North, Up] in meters.
            Shape can be (3,) or (N,3).

        radius:
            Orbit radius rho [m].

        direction:
            +1 = clockwise
            -1 = counter-clockwise

            This matches the convention used in Chapter 10 of the book.

        altitude:
            Optional altitude command [m]. If omitted, center_enu[...,2] is
            used as the desired altitude.

        airspeed:
            Optional airspeed command [m/s]. If omitted, the previous/current
            autopilot airspeed command is retained.
        """

        self._mode = "orbit"

        self._pending_orbit_center_enu = center_enu
        self._pending_orbit_radius = radius
        self._pending_orbit_direction = direction
        self._orbit_altitude_override = altitude
        self._orbit_airspeed_override = airspeed

        if self._started:
            self._orbit_center_enu = self._broadcast_vector3(
                center_enu, "orbit_center_enu"
            )
            self._orbit_radius = self._broadcast_scalar(
                radius, "orbit_radius"
            )
            self._orbit_direction = self._broadcast_scalar(
                direction, "orbit_direction"
            )

            if torch.any(self._orbit_radius <= 0.0):
                raise ValueError("Orbit radius must be greater than zero")

            self._orbit_direction = torch.where(
                self._orbit_direction >= 0.0,
                torch.ones_like(self._orbit_direction),
                -torch.ones_like(self._orbit_direction),
            )

            if airspeed is not None:
                self._Va_command = self._broadcast_scalar(
                    airspeed, "orbit_airspeed"
                )

            if altitude is not None:
                self._h_command = self._broadcast_scalar(
                    altitude, "orbit_altitude"
                )
            else:
                self._h_command = self._orbit_center_enu[:, 2]

    def set_orbit_gain(self, k_orbit: float) -> None:
        if k_orbit <= 0.0:
            raise ValueError("k_orbit must be greater than zero")
        self._k_orbit = float(k_orbit)

    # ==================================================================
    # Chapter 10 - orbit guidance
    # ==================================================================

    def _orbit_guidance(
        self,
        *,
        chi: torch.Tensor,
        psi: torch.Tensor,
        Vg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Equations (10.15) and (10.18), adapted to ENU storage.

        Horizontal book coordinates:
            p_n = y_ENU
            p_e = x_ENU

        Course/phase remain measured clockwise from North.
        """

        pos = torch.as_tensor(
            self._state.position,
            dtype=torch.float32,
            device=self._device,
        )

        # Book north/east coordinates represented inside ENU.
        p_n = pos[:, 1]
        p_e = pos[:, 0]

        c_n = self._orbit_center_enu[:, 1]
        c_e = self._orbit_center_enu[:, 0]

        dn = p_n - c_n
        de = p_e - c_e

        distance = torch.sqrt(
            torch.clamp(dn.square() + de.square(), min=1e-12)
        )

        # Phase angle around the orbit, measured clockwise from North.
        varphi_raw = torch.atan2(de, dn)

        # Book recommendation: choose the 2*pi branch such that
        # -pi <= varphi - chi <= pi.
        varphi = chi + _wrap_pi(varphi_raw - chi)

        lam = self._orbit_direction
        rho = self._orbit_radius

        # Beard & McLain Eq. (10.15)
        chi_c = (
            varphi
            + lam
            * (
                math.pi / 2.0
                + torch.atan(
                    self._k_orbit
                    * (distance - rho)
                    / rho
                )
            )
        )

        # Keep the commanded course on the nearest branch to current chi.
        chi_c = chi + _wrap_pi(chi_c - chi)

        # Beard & McLain Eq. (10.18): coordinated-turn roll feedforward.
        #
        # phi_ff = lambda atan(
        #     Vg^2 / (rho*g*cos(chi-psi))
        # )
        cos_crab = torch.cos(chi - psi)

        denom = rho * self._gravity * cos_crab

        eps = 1e-3
        denom = torch.where(
            torch.abs(denom) < eps,
            torch.sign(denom + 1e-12) * eps,
            denom,
        )

        phi_ff = lam * torch.atan(
            Vg.square() / denom
        )

        phi_ff = torch.clamp(
            phi_ff,
            min=-self._phi_command_max,
            max=+self._phi_command_max,
        )

        self._distance_to_orbit = distance

        return chi_c, phi_ff

    # ==================================================================
    # State conversion: ENU/FLU -> textbook variables
    # ==================================================================

    def _extract_flight_state(self) -> dict[str, torch.Tensor]:
        state = self._state

        position = torch.as_tensor(
            state.position,
            dtype=torch.float32,
            device=self._device,
        )
        velocity_i = torch.as_tensor(
            state.linear_velocity,
            dtype=torch.float32,
            device=self._device,
        )
        velocity_b = torch.as_tensor(
            state.linear_body_velocity,
            dtype=torch.float32,
            device=self._device,
        )
        attitude = torch.as_tensor(
            state.attitude,
            dtype=torch.float32,
            device=self._device,
        )
        omega_flu = torch.as_tensor(
            state.angular_velocity,
            dtype=torch.float32,
            device=self._device,
        )

        R = quaternion_to_matrix(attitude)

        # Standard 3-2-1 decomposition of the FLU orientation.
        phi = torch.atan2(
            R[:, 2, 1],
            R[:, 2, 2],
        )

        theta_flu = -torch.asin(
            torch.clamp(R[:, 2, 0], -1.0, 1.0)
        )

        # Book pitch is positive nose-up; FLU pitch is positive nose-down.
        theta_book = -theta_flu

        # Body x-axis in ENU world coordinates.
        forward_i = R[:, :, 0]

        # Heading measured clockwise from North:
        # atan2(East component, North component).
        psi = torch.atan2(
            forward_i[:, 0],
            forward_i[:, 1],
        )

        v_e = velocity_i[:, 0]
        v_n = velocity_i[:, 1]

        Vg_horizontal = torch.sqrt(
            torch.clamp(v_e.square() + v_n.square(), min=0.0)
        )

        # Course measured clockwise from North, as in the book.
        chi_velocity = torch.atan2(v_e, v_n)

        # Course is ill-defined near zero groundspeed; use heading there.
        chi = torch.where(
            Vg_horizontal > 0.25,
            chi_velocity,
            psi,
        )

        Va = torch.linalg.vector_norm(
            velocity_b,
            dim=1,
        )

        h = position[:, 2]

        # FRD-equivalent angular-rate signs:
        p_book = omega_flu[:, 0]
        q_book = -omega_flu[:, 1]

        return {
            "phi": phi,
            "theta": theta_book,
            "psi": psi,
            "chi": chi,
            "p": p_book,
            "q": q_book,
            "r_flu": omega_flu[:, 2],
            "h": h,
            "Va": Va,
            "Vg": Vg_horizontal,
        }

    # ==================================================================
    # Tensor helpers
    # ==================================================================

    def _broadcast_scalar(self, value, name: str) -> torch.Tensor:
        if self._device is None:
            raise RuntimeError("Backend has not been started yet")

        x = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self._device,
        )

        if x.ndim == 0:
            return x.repeat(self._n_vehicles)

        if x.ndim == 1:
            if x.numel() == 1:
                return x.repeat(self._n_vehicles)
            if x.numel() == self._n_vehicles:
                return x.clone()

        if x.ndim == 2 and x.shape == (self._n_vehicles, 1):
            return x[:, 0].clone()

        raise ValueError(
            f"{name} must be scalar or have one value per vehicle; "
            f"got shape {tuple(x.shape)}"
        )

    def _broadcast_vector3(self, value, name: str) -> torch.Tensor:
        if self._device is None:
            raise RuntimeError("Backend has not been started yet")

        x = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self._device,
        )

        if x.ndim == 1 and x.shape == (3,):
            return x.unsqueeze(0).repeat(self._n_vehicles, 1)

        if x.ndim == 2:
            if x.shape == (1, 3):
                return x.repeat(self._n_vehicles, 1)
            if x.shape == (self._n_vehicles, 3):
                return x.clone()

        raise ValueError(
            f"{name} must have shape (3,) or "
            f"({self._n_vehicles}, 3); got {tuple(x.shape)}"
        )

    def _degrees_to_radians(self, value):
        if isinstance(value, torch.Tensor):
            return value * (math.pi / 180.0)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [
                float(v) * math.pi / 180.0
                for v in value
            ]

        return float(value) * math.pi / 180.0

    def _set_trim_output(self) -> None:
        if self._input_reference is None:
            return

        self._input_reference[:, 0] = self._delta_e_trim
        self._input_reference[:, 1] = self._delta_a_trim
        self._input_reference[:, 2] = self._delta_r_trim
        self._input_reference[:, 3] = self._omega_trim

    # ==================================================================
    # Diagnostics
    # ==================================================================

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def actuator_reference(self) -> torch.Tensor | None:
        return self._input_reference

    @property
    def course(self) -> torch.Tensor | None:
        return self._chi

    @property
    def course_command(self) -> torch.Tensor | None:
        return self._chi_command

    @property
    def roll(self) -> torch.Tensor | None:
        return self._phi

    @property
    def roll_command(self) -> torch.Tensor | None:
        return self._phi_command

    @property
    def roll_feedforward(self) -> torch.Tensor | None:
        return self._phi_feedforward

    @property
    def pitch(self) -> torch.Tensor | None:
        """Textbook pitch sign: positive nose-up."""
        return self._theta

    @property
    def pitch_command(self) -> torch.Tensor | None:
        """Textbook pitch sign: positive nose-up."""
        return self._theta_command

    @property
    def airspeed(self) -> torch.Tensor | None:
        return self._Va

    @property
    def groundspeed(self) -> torch.Tensor | None:
        return self._Vg

    @property
    def distance_to_orbit(self) -> torch.Tensor | None:
        return self._distance_to_orbit
