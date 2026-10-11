"""
| File: shuttle_glider_batch2.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Batched ShuttleGlider vehicle for Pegasus Simulator.
|
| The shuttle and the EasyGlider fuselage are represented by one merged rigid
| body in the respective USDA. The five propellers remain individual rigid
| bodies:
|   rotor_0 ... rotor_3 : vertical shuttle lift rotors
|   rotor_puller         : EasyGlider propeller, physical axis +x_B
|
| Input command vector for the ShuttleGliderBatch2 is an eight-element vector:
|   [omega_0, omega_1, omega_2, omega_3, Omega_p, delta_e, delta_a, delta_r]
"""

# Standard library imports
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch

# Sensors and dynamics setup
from pegasus.simulator.logic.thrusters.quadratic_thrust_curve_batch import QuadraticThrustCurveBatch
from pegasus.simulator.logic.thrusters.glider_thrust_curve_batch import GliderThrustCurveBatch
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch

# Extension APIs
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Utils
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert


InputMode = Literal["rotor_velocity_direct", "direct_force"]


# =============================================================================
# Corrected merged-body geometry
# =============================================================================

# Component CoM positions relative to the /body origin [m]
_SHUTTLE_COM = (0, 0, 0.1197)
_GLIDER_COM = (0, 0, -0.2803)

# Merged-body CoM position relative to the /body origin [m]
_MERGED_COM = (0, 0, -0.0031)

# Vector from the /body origin to the EasyGlider CoM [m]
_MERGED_R_ORIGIN = _GLIDER_COM

# Vector from the merged-body CoM to the EasyGlider CoM [m]
_MERGED_R_COM = tuple(g - c for g, c in zip(_GLIDER_COM, _MERGED_COM))


# =============================================================================
# RL compatibility adapter
# =============================================================================

class _CombinedRotorInterface:
    """
    Compatibility interface exposing the five propellers as a single rotor group.

    The RLBackend and ResetManager use 'vehicle._thrusters' to access rotor
    counts, velocity limits and rotor states. However, the Shuttle lift
    rotors and EasyGlider propeller use different propulsion models.

    This adapter provides a common interface for all vehicles in the batch
    while preserving the separate physical propulsion models.
    """

    def __init__(
        self,
        shuttle_thrusters: QuadraticThrustCurveBatch,
        glider_thruster: GliderThrustCurveBatch,
        n_vehicles: int,
        device,
    ) -> None:
        
        # Vehicles thrusters
        self._shuttle = shuttle_thrusters
        self._glider = glider_thruster
        
        # Number of rotors and vehicles
        self._num_rotors = 5
        self.n_vehicles = int(n_vehicles)
        
        # Set device
        self.device = torch.device(device)

        # Combine the min/max rotor velocities of the two physical models into one 5-rotor view
        self.min_rotor_velocity = torch.cat((self._shuttle.min_rotor_velocity.reshape(-1), self._glider.min_rotor_velocity.reshape(-1))).to(device=self.device, dtype=torch.float32)
        self.max_rotor_velocity = torch.cat((self._shuttle.max_rotor_velocity.reshape(-1), self._glider.max_rotor_velocity.reshape(-1))).to(device=self.device, dtype=torch.float32)

        # Create the combined rotor state tensors for the five-rotor view
        self._input_reference = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)
        self._velocity = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)
        self._force = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)

        # Store the reaction moment coefficients and rotation directions of both propulsion models
        self._reaction_moment_coefficient = torch.cat(
            (
                self._shuttle._reaction_moment_coefficient.reshape(-1),
                self._glider._reaction_moment_coefficient.reshape(-1),
            )
        ).to(device=self.device, dtype=torch.float32)

        # Store the thrust coefficients used by the RL trim calculation
        self._rotor_constant = torch.cat(
            (
                self._shuttle._rotor_constant.reshape(-1),
                self._glider._rotor_constant.reshape(-1),
            )
        ).to(device=self.device, dtype=torch.float32)

        self._rot_dir = torch.cat((self._shuttle._rot_dir.reshape(-1).to(dtype=torch.float32), self._glider.rot_dir.reshape(-1))).to(device=self.device, dtype=torch.float32)

    def sync_to_models(self) -> None:
        """Push reset rotor velocities into the physical models."""
        self._shuttle._velocity = self._velocity[:, :4].clone()
        self._glider._velocity = self._velocity[:, 4:5].clone()

    def sync_from_models(self) -> None:
        """Pull physical-model rotor states into the five-rotor RL view."""
        self._velocity[:, :4] = self._shuttle._velocity
        self._velocity[:, 4:5] = self._glider._velocity

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity



# =============================================================================
# Configuration
# =============================================================================

class ShuttleGliderBatchConfig2:
    """Configuration for :class:`ShuttleGliderBatch2`."""
    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        n_vehicles: int = 1,
        sensors: Sequence[Any] | None = None,
        graphical_sensors: Sequence[Any] | None = None,
        graphs: Sequence[Any] | None = None,
        backends: Sequence[Any] | None = None,
        shuttle_rotor_prim_names: Sequence[str] = ("rotor_0", "rotor_1", "rotor_2", "rotor_3"),
        propeller_prim_name: str = "rotor_puller",
        shuttle_rotor_axis_body: Sequence[float] = (0.0, 0.0, 1.0),
        propeller_axis_body: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        """ Initialize the ShuttleGliderBatch configuration """

        # Define the same device that is running the simulation
        self.device = PegasusInterface()._world_settings["device"]
        
        # Stage prefix used when spawning the batched vehicles in the world
        self.stage_prefix = "shuttle_glider"
        
        # The USD file that describes the visual appearance of the vehicle, as well as properties such as mass and inertia
        self.usd_file = ""

        # Four shuttle lift rotors        
        cfg = dict(cfg or {})

        shuttle_thrust_cfg = cfg.get("shuttle_thrust_cfg", {})
        glider_thrust_cfg = cfg.get("glider_thrust_cfg", {})
        aerodynamics_cfg = cfg.get("aerodynamics_cfg", {})
    
        shuttle_cfg = dict(shuttle_thrust_cfg or {})

        self.shuttle_thrust_curve = QuadraticThrustCurveBatch(
            config=shuttle_cfg,
            n_vehicles=n_vehicles,
            device=self.device,
        )

        # EasyGlider puller
        puller_cfg = dict(glider_thrust_cfg or {})

        self.glider_thrust_curve = GliderThrustCurveBatch(
            config=puller_cfg,
            n_vehicles=n_vehicles,
            device=self.device,
        )

        # EasyGlider aerodynamic model on the merged rigid body
        aero_cfg = dict(aerodynamics_cfg or {})
        geometry_cfg = dict(aero_cfg.get("geometry", {}))

        # Vector used to transfer the aerodynamic moment to the merged vehicle origin
        geometry_cfg.setdefault("r_origin", _MERGED_R_ORIGIN) 
        
        # Vector used to compute the velocity at the glider aerodynamic reference point from the merged vehicle CoM velocity
        geometry_cfg.setdefault("r_com", _MERGED_R_COM)        
        
        aero_cfg["geometry"] = geometry_cfg
        aero_cfg.setdefault("eval_at_ref", True)

        self.aerodynamics = GliderAerodynamicsBatch(config=aero_cfg, n_vehicles=n_vehicles, device=self.device)
        
        # Configure optional onboard sensors
        # These are currently disabled until batch sensor support is enabled
        # self.sensors = [Barometer(device=device), IMU(device=device), Magnetometer(device=device), GPS(device=device)]
        self.sensors = tuple(sensors or ())
        
        # The default graphical sensors are disabled
        self.graphical_sensors = tuple(graphical_sensors or ())
        
        # The default omnigraph graphs are disabled
        self.graphs = tuple(graphs or ())
        
        # The backend is defined considering the principal purpose of the simulation
        # For example, if the simulation is used for RL training, the first backend 
        # should be RLBackend
        self.backends = tuple(backends or ())

        # Names of the rotor prims in the USD file, used to resolve their indices
        self.shuttle_rotor_prim_names = tuple(shuttle_rotor_prim_names)
        self.propeller_prim_name = str(propeller_prim_name)
        
        # Define the physical axis of rotation of the shuttle lift rotors and the EasyGlider puller in the merged body frame
        self.shuttle_rotor_axis_body = torch.as_tensor(shuttle_rotor_axis_body, dtype=torch.float32, device=self.device)
        self.propeller_axis_body = torch.as_tensor(propeller_axis_body, dtype=torch.float32, device=self.device)


# =============================================================================
# Vehicle
# =============================================================================

class ShuttleGliderBatch2(VehicleBatch):
    """
    ShuttleGliderBatch2 class - It defines a base interface for batched Shuttle + EasyGlider hybrid vehicles.

    Aerodynamic forces and moments are computed in rotor velocity mode.
    """

    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "shuttle_glider",
        usd_file: str = "",
        vehicle_batch_id: int = 0,
        n_vehicles: int = 1,
        # Spawning pose and orientation of the vehicle
        init_pos=None,
        init_orientation=None,
        spacing: float = 3.0,
        config: ShuttleGliderBatchConfig2 | None = None,
    ) -> None:
        """Initialize the batched Shuttle–EasyGlider hybrid vehicle.

        Args:
            stage_prefix (str): Base name used when spawning the batched vehicles in the simulator. Defaults to "shuttle_glider".            usd_file (str): USD file describing the vehicle appearance and physical properties. Defaults to "".
            vehicle_batch_id (int): The id to be used for the vehicle batch. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qw, qx, qy, qz]. Defaults to [1.0, 0.0, 0.0, 0.0].
            config (ShuttleGliderBatchConfig2, optional): Vehicle configuration. Defaults to None.
        """

        if config is None:
            config = ShuttleGliderBatchConfig2(n_vehicles=n_vehicles)

        # 1. Initialize the VehicleBatch base class
        super().__init__(stage_prefix, vehicle_batch_id, usd_file, n_vehicles, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends, spacing)

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._shuttle_thrusters = config.shuttle_thrust_curve
        self._glider_thruster = config.glider_thrust_curve
        self._aerodynamics = config.aerodynamics
        
        self._aero_wrench_local = None # Aerodynamic force and torque in the merged body frame, computed by the GliderAerodynamicsBatch model

        # Compatibility interface expected by RLBackend / ResetManager
        self._thrusters = _CombinedRotorInterface(self._shuttle_thrusters, self._glider_thruster, self.n_vehicles, self.device)

        # Define the rotor prim names for the shuttle and glider rotors
        self._shuttle_rotor_prim_names = config.shuttle_rotor_prim_names
        self._propeller_prim_name = config.propeller_prim_name

        # Define the physical axis of rotation of the shuttle lift rotors and the EasyGlider puller in the merged body frame
        self._shuttle_rotor_axis_body = config.shuttle_rotor_axis_body.to(device=self.device, dtype=torch.float32)
        self._propeller_axis_body = config.propeller_axis_body.to(device=self.device, dtype=torch.float32)

        # Default control mode
        self._input_mode: InputMode = "rotor_velocity_direct"

        # Five rotor velocities and three aerodynamic surface deflections
        self.actuator_dim = 8
        
        # Initialize the prim indices of the Shuttle rotors and EasyGlider propeller
        self._shuttle_rotor_indices: tuple[int, ...] | None = None
        self._propeller_index: int | None = None
        self._rotor_positions_body: torch.Tensor | None = None
        
        # Initialize the force and torque buffers for each vehicle part
        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None

        # Initialize the aerodynamic surface deflections (elevator, aileron, rudder)
        self._surface_deflections = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)

    # =============================================================================
    # Lifecycle
    # =============================================================================

    def start(self) -> None:
        """
        Initialize the vehicle and prepare the force and torque buffers.

        This method initializes the vehicle primitives, resolves the rotor and
        propeller indices and allocates the force and torque buffers.
        
        No control allocation matrix is used (not defined).
        """   
        # Initialize the vehicle primitives and internal state
        self.initialize()

        # Resolve the prim indices of the Shuttle rotors and EasyGlider propeller
        self._resolve_prim_indices()

        # Cache the five rotor positions in the body frame for the RL trim calculation
        rotor_indices = (*self._shuttle_rotor_indices, self._propeller_index)
        self._rotor_positions_body = self._reset_part_positions_body[:, rotor_indices, :].clone()

        # Initialize the forces and torques applied to each part of the merged vehicle
        self._forces = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        self._torques = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)

    def stop(self) -> None:
        """ No extra actions are required when the simulation stops. """
        return
    
    # =============================================================================
    # Prim Layout
    # =============================================================================

    def _resolve_prim_indices(self) -> None:
        """
        Resolve the prim indices of the Shuttle rotors and EasyGlider propeller.
        
        The indices are determined from the first vehicle and are assumed
        to be consistent across all vehicles in the batch.
        """
        # Obtain the prim paths of the first vehicle
        first_vehicle_paths = self._vehicle_prims.prim_paths[: self.parts_per_vehicle]

        # Map each prim name to its corresponding index
        name_to_index = {path.rsplit("/", 1)[-1]: index for index, path in enumerate(first_vehicle_paths)}

        # Check that all required rotor and propeller prims exist
        missing = [
            name
            for name in (*self._shuttle_rotor_prim_names, self._propeller_prim_name)
            if name not in name_to_index
        ]

        if missing:
            raise RuntimeError(
                f"Rigid prims not found under {self._stage_prefix}: {missing}. "
                f"Available: {list(name_to_index.keys())}"
            )

        # Store the indices of the Shuttle rotors and EasyGlider propeller
        self._shuttle_rotor_indices = tuple(name_to_index[name] for name in self._shuttle_rotor_prim_names)
        self._propeller_index = name_to_index[self._propeller_prim_name]



    # =============================================================================
    # Physics
    # =============================================================================

    def update(self, dt: float) -> None:
        """
        Compute and apply the input consistent with the selected control input mode,
        considering the aerodynamic forces and moments.

        Depending on the selected input mode, the commands are interpreted either as
        rotor angular velocities or as body-frame forces and torques. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        
        if self._sim_running is False:
            return

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)
        
        if not self._backends:
            return

        # Reset the force and torque buffers without reallocating memory at each simulation step
        self._forces.zero_()
        self._torques.zero_()
        
        # Aerodynamic force and torque in the merged body frame, computed by the GliderAerodynamicsBatch model
        self._aero_wrench_local = None
        
        # Rotor velocity mode
        if self._input_mode == "rotor_velocity_direct":
            reference =  torch.as_tensor(self._backends[0].input_reference(), dtype=torch.float32, device=self.device)

            # Split the input reference into the shuttle rotor velocities, glider propeller velocity and aerodynamic surface deflections
            shuttle_omega = reference[:, 0:4]
            glider_omega = reference[:, 4:5]

            # Aerodynamic surface deflections (elevator, aileron, rudder)
            delta = reference[:, 5:8]
            self._surface_deflections = delta.clone()
            
            # ----------------------------------------------------------
            # EasyGlider aerodynamics
            # ----------------------------------------------------------
            aero_force, aero_torque = self._aerodynamics.update(self._state, delta, dt)
            self._aero_wrench_local = (aero_force, aero_torque) # Store the last computed aerodynamic force and torque in the merged body frame

            # Apply the aerodynamic force and torque to the merged body in its local frame
            # The aerodynamic torque includes the moment about the aerodynamic reference point
            # and the moment generated by the aerodynamic force offset from the body origin
            self._forces[:, self.body_index, :] += torch.as_tensor(aero_force, dtype=torch.float32, device=self.device)
            self._torques[:, self.body_index, :] += torch.as_tensor(aero_torque, dtype=torch.float32, device=self.device)
        
            # ----------------------------------------------------------
            # Rotor dynamics
            # ----------------------------------------------------------
            # Synchronize possible RL reset rotor states into the two physical propulsion models before applying the new command
            self._thrusters.sync_to_models()

            # Four shuttle lift rotors
            self._shuttle_thrusters.set_input_reference(shuttle_omega)
            shuttle_force, _, shuttle_reaction_moment_z = self._shuttle_thrusters.update(self._state, dt)

            # EasyGlider propeller
            self._glider_thruster.set_input_reference(glider_omega)
            glider_thrust, _, glider_reaction_moment = self._glider_thruster.update(self._state, dt)

            # Synchronize the rotor states from the two physical propulsion models into the five-rotor RL view after applying the new command
            self._thrusters.sync_from_models()

            # Express physical rotor axes in each rotor's local frame.
            # VehicleBatch applies each force with is_global=False.
            _, part_attitudes = self._vehicle_prims.get_world_poses()
            part_attitudes = torch.as_tensor(part_attitudes, dtype=torch.float32, device=self.device,).reshape(self.n_vehicles, self.parts_per_vehicle, 4)
            
            # Get the attitude of the merged body in the world frame
            body_attitude = part_attitudes[:, self.body_index, :]

            # Shuttle +z_B lift axis
            shuttle_axis_body = self._shuttle_rotor_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)
            shuttle_axis_world = quaternion_apply(body_attitude, shuttle_axis_body)

            # Compute the rotor axis of each rotor in the local frame of the rotor prim, so that the force can be applied in the correct direction
            for rotor_number, rotor_index in enumerate(self._shuttle_rotor_indices):
                rotor_attitude = part_attitudes[:, rotor_index, :]
                axis_rotor = quaternion_apply(quaternion_invert(rotor_attitude), shuttle_axis_world)

                self._forces[:, rotor_index, :] += shuttle_force[:, rotor_number : rotor_number + 1] * axis_rotor

            # Reaction moment yaw from the four vertical rotors (applied to the merged body frame in Z_B)
            self._torques[:, self.body_index, :] += shuttle_reaction_moment_z.unsqueeze(-1) * self._shuttle_rotor_axis_body.unsqueeze(0)

            # EasyGlider +x_B puller axis
            puller_axis_body = self._propeller_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)
            puller_axis_world = quaternion_apply(body_attitude, puller_axis_body)

            puller_attitude = part_attitudes[:, self._propeller_index, :]
            puller_axis_local = quaternion_apply(quaternion_invert(puller_attitude), puller_axis_world)

            # Applying thrust at the physical rotor prim lets PhysX generate the moment associated with the propeller thrust-line offset.
            self._forces[:, self._propeller_index, :] += glider_thrust * puller_axis_local

            # Reaction moment roll from the horizontal propeller (applied to the merged body frame in X_B)
            self._torques[:, self.body_index, :] += glider_reaction_moment.unsqueeze(-1) * self._propeller_axis_body.unsqueeze(0)

        # Direct body force and torque mode (ignore the rotor dynamics and aerodynamics)
        else:
            forces, torques = self._backends[0].get_forces_and_torques()
            
            # Copy the forces and torques provided by the backend into the per-part buffers
            self._forces.copy_(torch.as_tensor(forces, dtype=torch.float32, device=self.device))
            self._torques.copy_(torch.as_tensor(torques, dtype=torch.float32, device=self.device))

        # Optional backend disturbances.
        if self._backends and hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()

            if result is not None:
                external_forces, external_torques = result

                # Add the external forces and torques to the per-part buffers
                self._forces += torch.as_tensor(external_forces, dtype=torch.float32, device=self.device)
                self._torques += torch.as_tensor(external_torques, dtype=torch.float32, device=self.device)

        # Apply the forces and torques to each part of the vehicle in the local frame (is_global=False)
        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    # =============================================================================
    # Utilities
    # =============================================================================

    def set_input_mode(self, input_mode: InputMode) -> None:
        """
        Set the control input mode used by the batched ShuttleGlider.

        Args:
            input_mode (str): Control input mode. 
            Expected values are "rotor_velocity_direct" or "direct_force".
        """
        if input_mode not in ("rotor_velocity_direct", "direct_force"):
            raise ValueError("Unsupported ShuttleGliderBatch input mode: " f"{input_mode}")

        self._input_mode = input_mode


    # =============================================================================
    # Properties
    # =============================================================================

    @property
    def surface_deflections(self) -> torch.Tensor:
        """Latest surface deflections [delta_e, delta_a, delta_r] [rad]."""
        return self._surface_deflections

    @property
    def aerodynamics(self):
        """EasyGlider aerodynamic model."""
        return self._aerodynamics
    
    @property
    def aero_wrench_local(self):
        """Latest aerodynamic force and torque in the body frame."""
        return self._aero_wrench_local

    @property
    def shuttle_thrusters(self):
        """Shuttle lift rotor model."""
        return self._shuttle_thrusters

    @property
    def glider_thruster(self):
        """EasyGlider propeller model."""
        return self._glider_thruster

    @property
    def thrusters(self):
        """Combined five-rotor RL interface."""
        return self._thrusters
