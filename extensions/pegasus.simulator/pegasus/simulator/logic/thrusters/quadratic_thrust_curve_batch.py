"""
| File: quadratic_thrust_curve_batch.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Description: File that implements a quadratic thrust curve for rotors
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""
import torch
from pegasus.simulator.logic.state_batch import StateBatch as State
from pegasus.simulator.logic.thrusters.thrust_curve import ThrustCurve

class QuadraticThrustCurveBatch(ThrustCurve):
    """Class that implements the dynamics of rotors that can be described by a quadratic thrust curve"""
    
    def __init__(self, config={}, n_vehicles = 1, device="cpu"):
        """
        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the QuadraticThrustCurve - empty dictionary means use default values (iris drone) without motor delays.
            n_vehicles (int): The number of vehicles for which to simulate the thrust curve.
            device (str): The device on which to run the simulation.

        Examples:
            The dictionary default parameters (iris drone) are

            >>> {"num_rotors": 4,
            >>>  "rotor_constant": [5.84e-6, 5.84e-6, 5.84e-6, 5.84e-6],
            >>>  "reaction_moment_coefficient": [3.504e-7, 3.504e-7, 3.504e-7, 3.504e-7], # 5.84e-06 * 0.06
            >>>  "rot_dir": [-1, -1, 1, 1],
            >>>  "min_rotor_velocity": [0, 0, 0, 0],                      # rad/s
            >>>  "max_rotor_velocity": [1100, 1100, 1100, 1100],          # rad/s
            >>>  "motor_time_constant_up": 0.0125,
            >>>  "motor_time_constant_down": 0.025,
            >>> }
        """
        # Prevent old configurations from silently using the default reaction coefficient
        if "rolling_moment_coefficient" in config:
            raise ValueError(
                "rolling_moment_coefficient is no longer supported by the batched thrust curve. "
                "Use reaction_moment_coefficient instead."
            )

        # Set device
        self.device = device

        # Get the total number of rotors to simulate
        self._num_rotors = config.get("num_rotors", 4)

        # Set the number of vehicles to construct batches of forces and velocities for each vehicle 
        self.n_vehicles = n_vehicles

        # The rotor constant used for computing the total thrust produced by the rotor: T = rotor_constant * omega^2
        self._rotor_constant = torch.tensor(config.get("rotor_constant", [5.84e-6, 5.84e-6, 5.84e-6, 5.84e-6]), dtype=torch.float32, device=self.device)
        assert len(self._rotor_constant) == self._num_rotors

        # The rotor constant used for computing the total torque generated about the vehicle Z-axis
        self._reaction_moment_coefficient = torch.tensor(config.get("reaction_moment_coefficient", [3.504e-7, 3.504e-7, 3.504e-7, 3.504e-7]), dtype=torch.float32, device=self.device)
        assert len(self._reaction_moment_coefficient) == self._num_rotors

        # Save the rotor direction of rotation
        self._rot_dir = torch.tensor(config.get("rot_dir", [-1, -1, 1, 1]), dtype=torch.int32, device=self.device)
        assert len(self._rot_dir) == self._num_rotors

        # Values for the minimum and maximum rotor velocity in rad/s
        self.min_rotor_velocity = torch.tensor(config.get("min_rotor_velocity", [0, 0, 0, 0]), dtype=torch.float32, device=self.device)
        assert len(self.min_rotor_velocity) == self._num_rotors

        self.max_rotor_velocity = torch.tensor(config.get("max_rotor_velocity", [1100, 1100, 1100, 1100]), dtype=torch.float32, device=self.device)
        assert len(self.max_rotor_velocity) == self._num_rotors

        # Motor time constants for acceleration and deceleration, in seconds
        self._motor_time_constant_up   = torch.as_tensor(config.get("motor_time_constant_up",   [0.0125]), dtype=torch.float32, device=self.device).reshape(1, -1)
        self._motor_time_constant_down = torch.as_tensor(config.get("motor_time_constant_down", [0.025]),  dtype=torch.float32, device=self.device).reshape(1, -1)

        # The actual speed references to apply to the vehicle rotor joints
        self._input_reference = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)
        
        # The actual velocity that each rotor is spinning at
        self._velocity = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)

        # The actual force that each rotor is generating
        self._force = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)

        # The actual reaction moment that is generated on the body frame of the vehicle
        self._reaction_moment = torch.zeros(self.n_vehicles, dtype=torch.float32, device=self.device)

    def set_input_reference(self, input_reference):
        """
        Receives as input a list of target angular velocities of each rotor in rad/s

        Args:
            input_reference (torch.Tensor): A tensor of shape (n_veiculos, num_rotors)
        """

        if input_reference.ndim == 1:
            input_reference = input_reference.unsqueeze(0).expand(self.n_vehicles, -1)

        # The target angular velocity of the rotor
        self._input_reference = torch.as_tensor(input_reference, dtype=torch.float32, device=self.device)
        
        
    def update(self, state: State, dt: float):
        """
        Update rotor velocities using first-order motor dynamics and compute
        the corresponding thrust forces and reaction moment.
        
        Note: The state variable is not used in this implementation, but is
        retained to support rotor models that depend on vehicle states.
        
        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Broadcast min/max from (num_rotors,) to (n_vehicles, num_rotors)
        min_vel = self.min_rotor_velocity.unsqueeze(0)
        max_vel = self.max_rotor_velocity.unsqueeze(0)

        # Desired rotor velocity after clipping
        target_velocity = torch.clamp(self._input_reference, min=min_vel, max=max_vel)
        
        # First-order motor response
        # dw/dt = (w_cmd - w) / tau
        dt_t = torch.as_tensor(float(dt), dtype=torch.float32, device=self.device)

        rising = target_velocity > self._velocity
        tau = torch.where(rising, self._motor_time_constant_up, self._motor_time_constant_down).clamp_min(1e-6)

        alpha = 1.0 - torch.exp(-dt_t / tau)

        self._velocity = self._velocity + alpha * (target_velocity - self._velocity)

        # Safety clipping
        self._velocity = torch.clamp(self._velocity, min=min_vel, max=max_vel)

        # Set the force using a quadratic thrust curve
        self._force = self._rotor_constant.unsqueeze(0) * torch.pow(self._velocity, 2)

        # Compute the reaction moment about the body Z-axis
        self._reaction_moment = torch.sum(self._reaction_moment_coefficient.unsqueeze(0) * torch.pow(self._velocity, 2) * self._rot_dir.unsqueeze(0), dim=1)

        # Return the forces and velocities on each rotor and total torque applied on the body frame
        return self._force, self._velocity, self._reaction_moment

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def force(self):
        """The force to apply to each rotor of the vehicle at any given time instant

        Returns:
            list: A list of forces (in Newton N) to apply to each rotor of the vehicle (on its Z-axis) at any given time instant
        """
        return self._force

    @property
    def velocity(self):
        """The velocity at which each rotor of the vehicle should be rotating at any given time instant

        Returns:
            list: A list of angular velocities (in rad/s) of each rotor (about its Z-axis) at any given time instant
        """
        return self._velocity

    @property
    def reaction_moment(self):
        """The total reaction moment being generated on the body frame of the vehicle by the rotating propellers

        Returns:
            float: The total reaction moment to apply to the vehicle body frame (Torque about the Z-axis) in Nm
        """
        return self._reaction_moment

    @property
    def rot_dir(self):
        """The direction of rotation of each rotor of the vehicles

        Returns:
            list(int): A list with the rotation direction of each rotor (-1 is counter-clockwise and 1 for clockwise)
        """
        return self._rot_dir
