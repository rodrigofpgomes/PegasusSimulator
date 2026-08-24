"""
| File: multirotor_batch.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Adapted by: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the MultirotorBatch class, which serves as the base class for batched multirotor vehicles.
"""

# Standard library imports
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDragBatch
from pegasus.simulator.logic.thrusters import QuadraticThrustCurveBatch
#from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert


# Default sensors can be enabled here if batch sensor support is required.
#from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Extension APIs
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# For debugging purposes
import carb


InputMode = Literal["rotor_velocity", "rotor_velocity_direct", "forces_torques"]


class MultirotorBatchConfig:
    """
    Class used to configure a MultirotorBatch vehicle.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        n_vehicles: int = 1,
        sensors: Sequence[Any] | None = None,
        graphical_sensors: Sequence[Any] | None = None,
        graphs: Sequence[Any] | None = None,
        backends: Sequence[Any] | None = None,
        rotor_prim_names: Sequence[str] | None = None,
    ) -> None:
        """
        Initialize the MultirotorBatch configuration.
        """

        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

        # Define the same device that is running the simulation
        device = PegasusInterface()._world_settings["device"]

        # Stage prefix used when spawning the batched vehicles in the world
        self.stage_prefix = "quadrotor"

        # The USD file that describes the visual appearance of the vehicle, as well as properties such as mass and inertia
        self.usd_file = ""

        # Default thrust curve and drag model for the batched quadrotors
        self.thrust_curve = QuadraticThrustCurveBatch(config=dict(cfg) if cfg else {}, n_vehicles=n_vehicles, device = device)
        self.drag = LinearDragBatch(n_vehicles=n_vehicles, drag_coefficients=[0.50, 0.30, 0.0])

        # Default onboard sensors for the batched quadrotors.
        # These are currently disabled until batch sensor support is enabled.
        #self.sensors = [Barometer(device=device), IMU(device=device), Magnetometer(device=device), GPS(device=device)]
        self.sensors = tuple(sensors or ())

        # Default graphical sensors for the batched quadrotors
        self.graphical_sensors = tuple(graphical_sensors or ())

        # Default OmniGraph graphs for the batched quadrotors
        self.graphs = tuple(graphs or ())

        # Backends used to send commands to the batched vehicles.
        # This can be a PX4, ROS 2, or custom backend implementation, or an empty list if no backend is required.
        self.backends = tuple(backends or ())

        # Optional: Names of the rotor prims in the USD file, used to cache their positions relative to the body frame.
        self.rotor_prim_names = tuple(rotor_prim_names) if rotor_prim_names is not None else None
        

class MultirotorBatch(VehicleBatch):
    """MultirotorBatch class - It defines a base interface for batched multirotor vehicles."""
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "quadrotor",
        usd_file: str = "",
        vehicle_batch_id: int = 0,
        n_vehicles: int = 1,
        # Spawning pose of the vehicle
        init_pos=None,
        init_orientation=None,
        spacing: float = 3.0,
        config=None,
    ):
        """Initialize the batched multirotor object.

        Args:
            stage_prefix (str): Base name used when spawning the batched vehicles in the simulator. Defaults to "quadrotor".
            usd_file (str): USD file describing the vehicle appearance and physical properties. Defaults to "".
            vehicle_batch_id (int): The id to be used for the vehicle batch. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qw, qx, qy, qz]. Defaults to [1.0, 0.0, 0.0, 0.0].
            config (MultirotorBatchConfig, optional): Defaults to MultirotorBatchConfig().
        """

        if config is None:
            config = MultirotorBatchConfig(n_vehicles=n_vehicles)

        # 1. Initialize the VehicleBatch base class
        super().__init__(stage_prefix, vehicle_batch_id, usd_file, n_vehicles, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends, spacing)

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag
        self._rotor_prim_names = config.rotor_prim_names

        # Default control mode. The backend pushes the authoritative mode via set_input_mode() in start()
        self._input_mode = "rotor_velocity"
        
        self._rotor_indices: tuple[int, ...] | None = None
        self._rotor_positions_body: torch.Tensor | None = None
        self._allocation_matrix: torch.Tensor | None = None
        self._allocation_inv: torch.Tensor | None = None

        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None
    

    def start(self):
        """
        Precompute and cache quantities required for multirotor dynamics,
        such as rotor positions relative to the body frame and the control
        allocation matrix. This method is called when the simulation starts.
        """        
        # 1. Initialize the vehicle primitives and internal state
        self.initialize()

        # 2. Obtain the indices of the rotor prims
        self._resolve_rotor_indices()

        # 3. Cache the rotor positions relative to the vehicle body frame.
        #    These are used when computing the forces applied in the simulator.
        self._cache_rotor_positions_body()

        # 4. Cache the control allocation matrix that maps desired body-frame
        #    forces and torques to rotor angular velocities.
        self._cache_allocation_matrix()

        self._forces = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        self._torques = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
    

    def stop(self):
        """
        No extra actions are required when the simulation stops.
        """
        return


    def _resolve_rotor_indices(self) -> None:
        """
        Resolve the indices of the rotor prims in the vehicle's USD hierarchy.
        
        The method assumes that the first 'parts_per_vehicle' prims in the batch 
        correspond to the first vehicle, and uses the provided rotor prim names (if any) 
        to identify which of those prims correspond to rotors. 
        
        If rotor prim names are not provided, it assumes that all prims except the one at `body_index` are rotors.
        """

        first_vehicle_paths = self._vehicle_prims.prim_paths[: self.parts_per_vehicle]
        num_rotors = self._thrusters._num_rotors

        if self._rotor_prim_names is None:
            indices = tuple(index for index in range(self.parts_per_vehicle) if index != self.body_index)
        else:
            path_name_to_index = {path.rsplit("/", 1)[-1]: index for index, path in enumerate(first_vehicle_paths)}
            missing = [name for name in self._rotor_prim_names if name not in path_name_to_index]

            if missing:
                raise RuntimeError(f"Rotor prims not found under {self._stage_prefix}: {missing}")

            indices = tuple(path_name_to_index[name] for name in self._rotor_prim_names)

        if len(indices) != num_rotors:
            raise RuntimeError(f"Expected {num_rotors} rotor prims, found {len(indices)}.")

        self._rotor_indices = indices


    def _cache_rotor_positions_body(self):
        """
        Compute and cache rotor positions relative to the vehicle body frame.

        This only needs to be done once because the rotor locations are fixed
        with respect to the body.
        """

        positions, attitudes = self._vehicle_prims.get_world_poses()

        # Reshape from (n_vehicles * parts_per_vehicle, 3) to (n_vehicles, parts_per_vehicle, 3)
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self._device).reshape(self._n_vehicles, self._parts_per_vehicle, 3)
        attitudes = torch.as_tensor(attitudes, dtype=torch.float32, device=self._device).reshape(self._n_vehicles, self._parts_per_vehicle, 4)

        # Extract body and rotor poses
        body_position = positions[:, self._body_index, :]       # (n_vehicles, 3)
        body_quaternion = attitudes[:, self._body_index, :]     # (n_vehicles, 4)
        rotor_positions = positions[:, self._rotor_indices, :]     # (n_vehicles, num_rotors, 3)

        # Compute rotor positions relative to the body in the world frame
        relative_positions_world = rotor_positions - body_position.unsqueeze(1)

        # Expand body quaternion so that one quaternion is associated with each rotor
        body_quaternion = body_quaternion.unsqueeze(1).expand(-1, self._thrusters._num_rotors, -1)

        # Rotate relative positions into the body frame
        self._rotor_positions_body = quaternion_apply(quaternion_invert(body_quaternion.reshape(-1, 4)), relative_positions_world.reshape(-1, 3)).reshape(self._n_vehicles, self._thrusters._num_rotors, 3)


    def _cache_allocation_matrix(self):
        """
        Build and cache the control allocation matrix and its pseudo-inverse.

        The matrix maps squared rotor angular velocities to total thrust and body
        torques. Since the vehicle geometry and rotor coefficients are fixed,
        this matrix can be computed once during initialization.
        """

        # Use the first vehicle as reference since all vehicles share the same geometry
        rotor_positions = self._rotor_positions_body[0]   # (num_rotors, 3)

        x = rotor_positions[:, 0]
        y = rotor_positions[:, 1]

        kf = self._thrusters._rotor_constant.to(device=self._device, dtype=torch.float32)
        km = self._thrusters._rolling_moment_coefficient.to(device=self._device, dtype=torch.float32)
        rotor_directions = self._thrusters._rot_dir.to(device=self._device, dtype=torch.float32)

        self._allocation_matrix = torch.zeros((4, self._thrusters._num_rotors), dtype=torch.float32, device=self._device)

        # Total thrust contribution
        self._allocation_matrix[0, :] = kf

        # Roll torque contribution: tau_x = y * Fz
        self._allocation_matrix[1, :] = y * kf

        # Pitch torque contribution: tau_y = -x * Fz
        self._allocation_matrix[2, :] = -x * kf

        # Yaw torque contribution from rotor drag
        self._allocation_matrix[3, :] = km * rotor_directions

        # Precompute pseudo-inverse for fast control allocation (matrix with rank 4)
        self._allocation_inv = torch.linalg.pinv(self._allocation_matrix)


    def update(self, dt: float):
        """
        Compute and apply the input consistent with the selected control input mode.

        Depending on the selected input mode, the commands are interpreted either as
        rotor angular velocities or as body-frame forces and torques. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if self._sim_running == False:
            return

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

        if not self._backends:
            return

        self._forces.zero_()
        self._torques.zero_()

        # TODO: Add batched propeller visual updates for rotor animation.
        # Rotor angular velocity mode (also used for direct rotor-velocity actions)
        if self._input_mode in ("rotor_velocity", "rotor_velocity_direct"):

            desired_rotor_velocities = self._backends[0].input_reference()
            
            # Input the desired rotor velocities in the thruster model
            self._thrusters.set_input_reference(desired_rotor_velocities)

            # Compute the rotor thrust forces and the desired rolling_moment
            rotor_forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

            # Apply the force in Z to each rotor in the rotor frame
            self._forces[:, self._rotor_indices, 2] = rotor_forces_z

            # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
            self._torques[:, self._body_index, 2] = rolling_moment

            #carb.log_warn(f"Rotor forces (Z): {rotor_forces_z}")
            #carb.log_warn(f"Rolling moment: {rolling_moment}")

        
        # Direct body force and torque mode
        else:
            forces, torques = self._backends[0].get_forces_and_torques()
            self._forces.copy_(forces)
            self._torques.copy_(torques)

        # Optional: add linear drag on the vehicle body.
        drag = self._drag.update(self._state, dt)
        #self._forces[:, 0, :] += drag

        if hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()
            if result is not None:
                external_forces, external_torques = result
                
                self._forces += external_forces
                self._torques += external_torques

        # Apply the batched forces and torques in the simulator.
        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    def force_and_torques_to_velocities(self, force: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (torch.Tensor): Desired total thrust along the body Z axis, shape (n_vehicles,)
            torque (torch.Tensor): A vector of the torque to be applied in the body frame of each vehicle [Nm], shape (n_vehicles, 3).

        Returns:
            torch.Tensor: Target rotor angular velocities, shape (n_vehicles, num_rotors).
        """

        force = force.to(device=self._device, dtype=torch.float32)
        torque = torque.to(device=self._device, dtype=torch.float32)

        allocation_inv = self._allocation_inv.to(device=self._device, dtype=torch.float32)

        # Build desired wrench vector [T, tau_x, tau_y, tau_z]
        wrench = torch.cat((force.unsqueeze(1), torque), dim=1)   # (n_vehicles, 4)

        # Solve for squared angular velocities
        squared_ang_vel = wrench @ allocation_inv.T         # (n_vehicles, num_rotors)

        # Clamp negative values caused by the pseudo-inverse
        squared_ang_vel = torch.clamp(squared_ang_vel, min=0.0)

        # Saturate while preserving the relative distribution between rotors
        max_thrust_vel_squared = torch.pow(self._thrusters.max_rotor_velocity[0], 2)
        max_val = torch.max(squared_ang_vel, dim=1, keepdim=True).values

        normalize = torch.clamp(max_val / max_thrust_vel_squared, min=1.0)
        squared_ang_vel = squared_ang_vel / normalize

        # Convert to angular velocity
        ang_vel = torch.sqrt(squared_ang_vel)

        return ang_vel
 
    
    def set_input_mode(self, input_mode: str):
        """
        Set the control input mode used by the batched multirotor.

        Args:
            input_mode (str): Control input mode. 
            Expected values are "rotor_velocity", "rotor_velocity_direct", or "forces_torques".
        """
        self._input_mode = input_mode

