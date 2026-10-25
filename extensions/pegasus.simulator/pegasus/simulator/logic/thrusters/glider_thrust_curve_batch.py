"""
| File: glider_thrust_curve_batch.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: File that implements a quadratic thrust curve for the propellers of batched fixed-wing aircraft.
|
| Current model:
|     T = k_T * omega^2
|     Q = k_Q * omega^2 * rot_dir
|
| Future model (to be implemented):
|     J   = V_a / (n D_p)
|     T   = rho n^2 D_p^4 C_T(J)
|     Q   = rho n^2 D_p^5 C_Q(J)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from pegasus.simulator.logic.state_batch import StateBatch as State
from pegasus.simulator.logic.thrusters.thrust_curve import ThrustCurve


class GliderThrustCurveBatch(ThrustCurve):
    """ Class that implements the dynamics of the glider rotor """

    def __init__(self, config= {}, n_vehicles: int = 1, device: str = "cpu") -> None:
        """
        Args:
            config (dict): A Dictionary that contains the parameters for configuring the GliderThrustCurve
            n_vehicles (int): The number of vehicles for which to simulate the thrust curve.
            device (str): The device on which to run the simulation.

        Examples:
            The dictionary default parameters are

            >>> {"rotor_constant": 8.54858e-06,
            >>>  "reaction_moment_coefficient": 8.54858e-8, # 8.54858e-06 * 0.01 (PX4 advanced_plane)
            >>>  "rot_dir": 1,            # puller is "cw": reaction +k_Q*omega^2 about +X
            >>>  "min_rotor_velocity": 0,         # rad/s
            >>>  "max_rotor_velocity": 3500,      # rad/s
            >>>  "motor_time_constant_up": 0.0125,
            >>>  "motor_time_constant_down": 0.025,
            >>> }
        """

        # Avoid silently ignoring the old Q = moment_constant * T settings.
        legacy_keys = {"thrust_constant", "moment_constant", "reaction_moment_sign"}
        unexpected = legacy_keys.intersection(config)
        if unexpected:
            raise ValueError(
                f"Legacy glider thrust configuration keys {sorted(unexpected)} are no longer supported. "
                "Use rotor_constant, reaction_moment_coefficient and rot_dir instead."
            )

        # Set device
        self.device = device
        
        # Set the number of vehicles to construct batches of forces and velocities for each vehicle 
        self.n_vehicles = n_vehicles

        # The glider has one puller rotor; RLBackend uses this count to size its commands
        self._num_rotors = 1

        # The rotor constant used for computing the total thrust produced by the rotor: T = rotor_constant * omega^2
        self._rotor_constant = torch.tensor(config.get("rotor_constant", [8.54858e-06]), dtype=torch.float32, device=self.device).reshape(-1)

        # The rotor constant used for computing the total torque generated about the vehicle X-axis
        self._reaction_moment_coefficient = torch.tensor(config.get("reaction_moment_coefficient", [8.54858e-8]), dtype=torch.float32, device=self.device).reshape(-1)

        # Save the rotor direction of rotation
        self._rot_dir = torch.tensor(config.get("rot_dir", [1]), dtype=torch.int32, device=self.device).reshape(-1)

        # Values for the minimum and maximum rotor velocity in rad/s
        self.min_rotor_velocity = torch.tensor(config.get("min_rotor_velocity", 0), dtype=torch.float32, device=self.device).reshape(-1)
        self.max_rotor_velocity = torch.tensor(config.get("max_rotor_velocity", 3500.0), dtype=torch.float32, device=self.device).reshape(-1)

        # Motor time constants for acceleration and deceleration, in seconds
        self._motor_time_constant_up   = torch.tensor(config.get("motor_time_constant_up",   0.0125), dtype=torch.float32, device=self.device).reshape(1, -1)
        self._motor_time_constant_down = torch.tensor(config.get("motor_time_constant_down", 0.025),  dtype=torch.float32, device=self.device).reshape(1, -1)

        # The actual speed references to apply to the vehicle rotor joints
        self._input_reference = torch.zeros((self.n_vehicles, 1), dtype=torch.float32, device=self.device)

        # The actual velocity of the rotor
        self._velocity = torch.zeros((self.n_vehicles, 1), dtype=torch.float32, device=self.device)

        # The actual force the rotor is generating
        self._force = torch.zeros_like(self._input_reference)

        # Equal-and-opposite torque acting on the fuselage, in body frame.
        self._reaction_moment = torch.zeros(self.n_vehicles, dtype=torch.float32, device=self.device)

        # Values reserved for the future advance-ratio model
        self._air_density = float(config.get("air_density", 1.2041))
        self._propeller_diameter = float(config.get("propeller_diameter", 0.229))


    def set_input_reference(self, input_reference) -> None:
        """
        Set the target angular velocity of the puller rotor for each vehicle in rad/s.
        
        Args:
            input_reference (torch.Tensor): Tensor of shape (n_vehicles,) or (n_vehicles, 1).
        """

        if input_reference.ndim == 1:
            input_reference = input_reference.unsqueeze(-1) # Expand to shape (n_vehicles, 1)

        # The target angular velocity of the rotor
        self._input_reference = torch.as_tensor(input_reference, dtype=torch.float32, device=self.device)


    def update(self, state: State, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update rotor velocities using first-order motor dynamics and compute
        the corresponding thrust forces and reaction moment.
        
        Note: The state variable is not used in the current implementation.
        Future implementations may consider the advance ratio to model
        the propeller's aerodynamic performance.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Desired rotor velocity after clipping
        target_velocity = torch.clamp(self._input_reference, min=self.min_rotor_velocity, max=self.max_rotor_velocity)

        # First-order motor response:
        # dw/dt = (w_cmd - w) / tau
        dt_t = torch.as_tensor(float(dt), dtype=torch.float32, device=self.device)

        rising = target_velocity > self._velocity
        tau = torch.where(rising, self._motor_time_constant_up, self._motor_time_constant_down).clamp_min(1e-6)

        alpha = 1.0 - torch.exp(-dt_t / tau)
        self._velocity = self._velocity + alpha * (target_velocity - self._velocity)
    
        # Safety clipping
        self._velocity = torch.clamp(self._velocity, min=self.min_rotor_velocity, max=self.max_rotor_velocity)
        
        # Set the force using a quadratic thrust curve
        self._force = self._rotor_constant.unsqueeze(0) * torch.pow(self._velocity, 2)

        # Compute the reaction moment about the body X-axis
        self._reaction_moment = torch.sum(self._reaction_moment_coefficient.unsqueeze(0) * torch.pow(self._velocity, 2) * self._rot_dir.unsqueeze(0), dim=1)

        return self._force, self._velocity, self._reaction_moment

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """
        Reset propeller state for all or selected environments.
        """

        if env_ids is None:
            self._input_reference.zero_()
            self._velocity.zero_()
            self._force.zero_()
            self._reaction_moment.zero_()
            return

        ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )

        self._input_reference[ids] = 0.0
        self._velocity[ids] = 0.0
        self._force[ids] = 0.0
        self._reaction_moment[ids] = 0.0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def force(self) -> torch.Tensor:
        """The thrust force generated by the puller rotor of each Glider at any given time instant

        Returns:
            torch.Tensor: A tensor of thrust forces (in Newton N) generated by the puller rotor of each Glider (along its X-axis), with shape (n_vehicles, 1)
        """        
        return self._force

    @property
    def velocity(self) -> torch.Tensor:
        """The angular velocity at which the puller rotor of each Glider is rotating at any given time instant

        Returns:
            torch.Tensor: A tensor of angular velocities (in rad/s) of the puller rotor of each Glider (about its X-axis), with shape (n_vehicles, 1)
        """
        return self._velocity

    @property
    def reaction_moment(self) -> torch.Tensor:
        """The reaction moment generated on the body frame of each Glider by the rotating puller propeller

        Returns:
            torch.Tensor: A tensor of reaction moments (in Nm) generated on the body frame of each Glider (about its X-axis), with shape (n_vehicles,)
        """
        return self._reaction_moment

    @property
    def rot_dir(self):
        """The direction of rotation of the puller rotor of each Glider

        Returns:
            torch.Tensor: The rotation direction of the puller rotor (-1 for counter-clockwise and 1 for clockwise)
        """
        return self._rot_dir
    
    @property
    def air_density(self) -> float:
        """The air density used for the future aerodynamic propeller model

        Returns:
        float: The air density (in kg/m^3)
        """
        return self._air_density

    @property
    def propeller_diameter(self) -> float:
        """The diameter of the puller propeller used for the future aerodynamic model

        Returns:
        float: The propeller diameter (in m)
        """
        return self._propeller_diameter
