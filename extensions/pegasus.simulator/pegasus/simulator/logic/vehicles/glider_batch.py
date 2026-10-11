"""
| File: glider_batch.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Batched Glider vehicle interface for Pegasus Simulator.
|
| Actuator command per vehicle:
|   [Omega_p, delta_e, delta_a, delta_r] in [rad/s, rad, rad, rad]
|
| This class applies the aerodynamic forces/moments that are produced by 
| an injected batched aerodynamic model, while the nose propeller is handled 
| by an injected single-propeller thrust model.
"""

# Standard library imports
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch

# Sensors and dynamics setup
from pegasus.simulator.logic.thrusters.glider_thrust_curve_batch import GliderThrustCurveBatch
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch

# Extension APIs
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Utils
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert


InputMode = Literal["rotor_velocity_direct", "direct_force"]


# =============================================================================
# Configuration
# =============================================================================

class GliderBatchConfig:
    """Configuration for the batched EasyGlider vehicle."""

    def __init__(
        self,
        n_vehicles: int = 1,
        thrust_curve_cfg: Mapping[str, Any] | None = None,
        aerodynamics_cfg: Mapping[str, Any] | None = None,
        sensors: Sequence[Any] | None = None,
        graphical_sensors: Sequence[Any] | None = None,
        graphs: Sequence[Any] | None = None,
        backends: Sequence[Any] | None = None,
        propeller_prim_name: str = "rotor",
        propeller_axis_body: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        """Initialize the single-propeller Glider configuration."""
         
        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

        # Define the same device that is running the simulation
        self.device = PegasusInterface()._world_settings["device"]
        
        # Stage prefix used when spawning the batched vehicles in the world
        self.stage_prefix = "easyglider"
        
        # The USD file that describes the visual appearance of the vehicle, as well as properties such as mass and inertia
        self.usd_file = ""
        
        # One puller rotor supplies thrust along the body X-axis
        self.thrust_curve = GliderThrustCurveBatch(
            config=dict(thrust_curve_cfg or {}),
            n_vehicles=n_vehicles,
            device=self.device,
        )

        # The standalone USDA places /body at the aerodynamic reference origin
        aero_cfg = dict(aerodynamics_cfg or {})
        geometry = dict(aero_cfg.get("geometry", {}))
        geometry.setdefault("r_origin", (0.0, 0.0, 0.0))
        geometry.setdefault("r_com", (0.0, 0.0, 0.0))
        aero_cfg["geometry"] = geometry

        self.aerodynamics = GliderAerodynamicsBatch(config=aero_cfg, n_vehicles=n_vehicles, device=self.device)

        # Optional onboard sensors; batch sensors are disabled by default
        self.sensors = tuple(sensors or ())

        # Default graphical sensors for the batched gliders
        self.graphical_sensors = tuple(graphical_sensors or ())

        # Default OmniGraph graphs for the batched gliders
        self.graphs = tuple(graphs or ())

        # The first backend supplies commands when no direct command is active
        self.backends = tuple(backends or ())

        # Name of the puller rotor prim in the USD file
        self.propeller_prim_name = str(propeller_prim_name)

        # Normalize the thrust direction expressed in the /body frame
        axis = torch.as_tensor(propeller_axis_body, dtype=torch.float32, device=self.device)
        if axis.shape != (3,) or float(torch.linalg.vector_norm(axis)) <= 1e-9:
            raise ValueError("propeller_axis_body must be a nonzero three-component vector")
        
        self.propeller_axis_body = axis / torch.linalg.vector_norm(axis)


# =============================================================================
# Vehicle
# =============================================================================

class GliderBatch(VehicleBatch):
    """
    Batched fixed-wing Glider vehicle.

    This class represents a fixed-wing glider with aerodynamic control surfaces. 
    The four inputs are the nose propeller speed and three surface deflections:
    [Omega_p, delta_e, delta_a, delta_r].
    """

    def __init__(
        self,
        stage_prefix: str = "easyglider",
        usd_file: str = "",
        vehicle_batch_id: int = 0,
        n_vehicles: int = 1,
        init_pos=None,
        init_orientation=None,
        spacing: float = 3.0,
        config: GliderBatchConfig | None = None,
    ) -> None:
        """Initialize the vehicle models, control state and per-vehicle buffers."""

        if config is None:
            config = GliderBatchConfig(n_vehicles=n_vehicles)

        # Initialize the shared VehicleBatch interface and backend connections
        super().__init__(
            stage_prefix,
            vehicle_batch_id,
            usd_file,
            n_vehicles,
            init_pos,
            init_orientation,
            config.sensors,
            config.graphical_sensors,
            config.graphs,
            config.backends,
            spacing,
        )

        # Use separate models for the wing aerodynamics and nose propeller
        self._aerodynamics = config.aerodynamics
        self._thrusters = config.thrust_curve

        # The rotor prim receives thrust; the body receives the reaction moment
        self._propeller_prim_name = config.propeller_prim_name
        self._propeller_axis_body = config.propeller_axis_body.to(
            device=self.device, dtype=torch.float32
        )

        # The first input is rotor speed, followed by the three surface angles
        self._input_mode: InputMode = "rotor_velocity_direct"
        self.actuator_dim = 4

        # Resolve the rotor prim and allocate load buffers in start()
        self._propeller_index: int | None = None
        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None

        # Store the latest aerodynamic inputs and outputs for observation/logging
        self._surface_deflections = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)
        self._aero_wrench_local = None

        # Optional airborne reset conditions and world-frame wind velocity
        self._flight_reset = None
        self._wind_world = torch.zeros((self.n_vehicles, 3), device=self.device)

    # =============================================================================
    # Lifecycle
    # =============================================================================

    def start(self) -> None:
        """Initialize prim views, find the propeller and allocate load buffers."""
        
        # Build the batched rigid-body view and state from the vehicle USDA
        self.initialize()

        # Locate the nose propeller within each vehicle's rigid prim layout
        self._resolve_propeller_index()

        # One force and torque vector is stored for every rigid prim
        self._forces = torch.zeros(
            (self.n_vehicles, self.parts_per_vehicle, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._torques = torch.zeros_like(self._forces)

        # Initialize models that implement an optional vehicle-specific hook
        for model in (self._aerodynamics, self._thrusters):
            initialize = getattr(model, "initialize", None)
            if callable(initialize):
                initialize(self)

    def stop(self) -> None:
        """No additional stop action is required."""
        return

    # =============================================================================
    # Prim layout
    # =============================================================================

    def _resolve_propeller_index(self) -> None:
        """Resolve the nose propeller's rigid-prim index from the first vehicle."""

        # All vehicles in the batch use the same ordering of rigid prims
        first_vehicle_paths = self._vehicle_prims.prim_paths[: self.parts_per_vehicle]
        path_name_to_index = {path.rsplit("/", 1)[-1]: index for index, path in enumerate(first_vehicle_paths)}

        # The configured propeller must be a distinct rigid prim from /body
        if self._propeller_prim_name not in path_name_to_index:
            raise RuntimeError(
                f"Propeller prim '{self._propeller_prim_name}' not found under "
                f"{self._stage_prefix}. Available rigid prims in the first vehicle: "
                f"{list(path_name_to_index.keys())}"
            )

        self._propeller_index = path_name_to_index[self._propeller_prim_name]

        if self._propeller_index == self.body_index:
            raise RuntimeError("The propeller prim cannot be the same prim as '/body'.")

    # =============================================================================
    # Airborne reset
    # =============================================================================

    def configure_flight_reset(
        self,
        positions,
        orientations,
        air_velocity_body,
        minimum_airspeed,
        rotor_velocity,
        wind_world=(0.0, 0.0, 0.0),
    ) -> None:
        """Store the flight conditions used by full and partial airborne resets.

        Args:
            positions: Initial positions in world ENU coordinates [m].
            orientations: Body-to-world quaternions [w, x, y, z].
            air_velocity_body: Air-relative velocities in body FLU coordinates [m/s].
            minimum_airspeed: Minimum airspeed for the selected flight regime [m/s].
            rotor_velocity: Initial puller velocity [rad/s], with shape (1,)
                or (n_vehicles, 1). The first backend supplies subsequent commands.
            wind_world: Wind velocities in world ENU coordinates [m/s].
        """

        def batch(value, width):
            """Expand one vector or copy one vector per vehicle to the simulator device."""
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)

            if tensor.shape == (width,):
                tensor = tensor.expand(self.n_vehicles, -1)

            if tensor.shape != (self.n_vehicles, width) or not torch.isfinite(tensor).all():
                raise ValueError(f"Expected finite ({self.n_vehicles}, {width}) initial data")

            return tensor.clone()

        # Store world positions and normalize the body-to-world orientations
        pos = batch(positions, 3)
        ori = batch(orientations, 4)
        norm = torch.linalg.vector_norm(ori, dim=-1, keepdim=True)

        if (norm < 1e-6).any():
            raise ValueError("Initial attitude quaternion must be nonzero")

        ori /= norm

        # Keep air-relative velocity in the body frame until reset_in_flight()
        air = batch(air_velocity_body, 3)

        if minimum_airspeed <= 0 or (torch.linalg.vector_norm(air, dim=-1) < minimum_airspeed).any():
            raise ValueError("Air-relative initial velocity is below the configured regime bound")

        # Initialize only the motor state; surface commands come from the backend
        omega_p = batch(rotor_velocity, 1)

        if (
            (omega_p < self._thrusters.min_rotor_velocity).any()
            or (omega_p > self._thrusters.max_rotor_velocity).any()
        ):
            raise ValueError("Initial puller velocity exceeds rotor limits")

        # Wind is stored in world coordinates and added to the air velocity at reset
        self._wind_world = batch(wind_world, 3)
        self._flight_reset = (pos, ori, air, omega_p)

    def reset_in_flight(self, env_ids=None) -> None:
        """Apply stored airborne conditions to the selected vehicles."""
        from pegasus.simulator.logic.rl.reset_manager import ResetManager

        if self._flight_reset is None:
            raise RuntimeError("Call configure_flight_reset with an explicit airspeed bound and trim first")

        # Reset every vehicle when no subset of environment IDs is supplied
        ids = (
            torch.arange(self.n_vehicles, device=self.device)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )

        pos, ori, air, omega_p = self._flight_reset

        # PhysX receives ground velocity in world coordinates: rotate the
        # body-frame air velocity and add the world-frame wind velocity
        ground_velocity = quaternion_apply(ori[ids], air[ids]) + self._wind_world[ids]

        # ResetManager accepts the rotor command normalized to [-1, 1]
        minimum = self._thrusters.min_rotor_velocity
        span = self._thrusters.max_rotor_velocity - minimum
        rotor_norm = 2.0 * (omega_p[ids] - minimum) / span - 1.0

        # Position overrides are offsets relative to the vehicle spawn positions
        ResetManager([self], device=self.device).reset_envs(
            ids,
            init_overrides={
                "position_offset": pos[ids] - self._init_pos[ids],
                "orientations": ori[ids],
                "linear_velocity": ground_velocity,
                "rotor_norm": rotor_norm,
            },
        )

        # Convert world wind to each vehicle's current body frame for aerodynamics
        wind_body = quaternion_apply(quaternion_invert(self.state.attitude), self._wind_world)
        self._aerodynamics.set_wind_body(wind_body)


    # =============================================================================
    # Physics
    # =============================================================================

    def update(self, dt: float) -> None:
        """
        Apply aerodynamic and propulsive loads for the selected control mode.

        In rotor velocity mode, the first backend supplies one puller speed
        and three surface deflections. Aerodynamics acts on /body;
        propeller thrust acts at the rotor prim and reaction torque on /body.

        Args:
            dt (float): Time since the previous physics step, in seconds.
        """

        if self._sim_running is False:
            return

        # Give every backend the current physics time step
        for backend in self._backends:
            backend.update(dt)

        # The first backend owns the control command for this vehicle
        if not self._backends:
            return

        # Buffers are allocated by start() after the rigid prims are available
        if self._forces is None or self._torques is None:
            return

        # Reuse the per-part buffers; each step starts with zero applied load
        self._forces.zero_()
        self._torques.zero_()
        self._aero_wrench_local = None

        # The backend can supply per-part loads directly in direct_force mode
        if self._input_mode == "direct_force":
            if not hasattr(self._backends[0], "get_forces_and_torques"):
                raise RuntimeError("The selected backend does not implement get_forces_and_torques().")

            forces, torques = self._backends[0].get_forces_and_torques()
            self._forces.copy_(torch.as_tensor(forces, dtype=torch.float32, device=self.device))
            self._torques.copy_(torch.as_tensor(torques, dtype=torch.float32, device=self.device))

        # In rotor velocity mode, split the first backend's four-channel command
        else:
            reference = torch.as_tensor(self._backends[0].input_reference(), dtype=torch.float32, device=self.device)
            omega_p = reference[:, 0:1]
            delta = reference[:, 1:4]
            self._surface_deflections.copy_(delta)

            # --------------------------------------------------------------
            # Aerodynamics
            # --------------------------------------------------------------

            # The aerodynamic model uses body-frame wind after an airborne reset
            if self._flight_reset is not None:
                wind_body = quaternion_apply(quaternion_invert(self._state.attitude), self._wind_world)
                self._aerodynamics.set_wind_body(wind_body)

            # Pass the three surface angles to the aerodynamic model
            aero_force, aero_torque = self._aerodynamics.update(self._state, delta, dt)

            aero_force = torch.as_tensor(aero_force, dtype=torch.float32, device=self.device)
            aero_torque = torch.as_tensor(aero_torque, dtype=torch.float32, device=self.device)
            self._aero_wrench_local = (aero_force, aero_torque)

            # The aerodynamic model returns a wrench about the /body origin
            self._forces[:, self.body_index, :] += aero_force
            self._torques[:, self.body_index, :] += aero_torque

            # --------------------------------------------------------------
            # Nose propeller
            # --------------------------------------------------------------

            # Update the motor state from the requested angular velocity
            self._thrusters.set_input_reference(omega_p)
            thrust, _, reaction_moment = self._thrusters.update(self._state, dt)

            thrust = torch.as_tensor(thrust, dtype=torch.float32, device=self.device)
            reaction_moment = torch.as_tensor(reaction_moment, dtype=torch.float32, device=self.device)

            # Fetch the body and rotor orientations for every vehicle
            _, part_attitudes = self._vehicle_prims.get_world_poses()
            part_attitudes = torch.as_tensor(part_attitudes, dtype=torch.float32, device=self.device).reshape(
                self.n_vehicles, self.parts_per_vehicle, 4
            )
            body_attitude = part_attitudes[:, self.body_index, :]
            rotor_attitude = part_attitudes[:, self._propeller_index, :]

            # The configured thrust axis is expressed in the body frame
            axis_body = self._propeller_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)

            # Rotate the physical axis from body to world coordinates
            axis_world = quaternion_apply(body_attitude, axis_body)

            # Express that same axis in the rotor prim's local coordinates;
            # VehicleBatch applies forces with is_global=False
            axis_rotor = quaternion_apply(quaternion_invert(rotor_attitude), axis_world)

            # Thrust acts at the rotor prim, including its physical offset from /body
            prop_force_rotor = thrust * axis_rotor
            self._forces[:, self._propeller_index, :] += prop_force_rotor

            # The motor reaction moment acts about the puller axis on /body
            self._torques[:, self.body_index, :] += reaction_moment.unsqueeze(-1) * axis_body

        # Add optional external disturbances returned by the first backend
        if self._backends and hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()
            
            if result is not None:
                external_forces, external_torques = result
                self._forces += torch.as_tensor(external_forces, dtype=torch.float32, device=self.device)
                self._torques += torch.as_tensor(external_torques, dtype=torch.float32, device=self.device)

        # Apply all accumulated per-part loads in each rigid prim's local frame
        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    # =============================================================================
    # Utilities
    # =============================================================================

    def set_input_mode(self, input_mode: InputMode) -> None:
        """
        Select how the first backend controls the vehicle.

        Args:
            input_mode (str): "rotor_velocity_direct" for the backend's four
                commands or "direct_force" for per-part loads.
        """

        if input_mode not in ("rotor_velocity_direct", "direct_force"):
            raise ValueError("GliderBatch input_mode must be 'rotor_velocity_direct' or 'direct_force'.")
        
        self._input_mode = input_mode

    # =============================================================================
    # Properties
    # =============================================================================

    @property
    def surface_deflections(self) -> torch.Tensor:
        """Latest [delta_e, delta_a, delta_r] deflections [rad]."""
        return self._surface_deflections

    @property
    def aero_wrench_local(self):
        """Latest aerodynamic force and torque in the body frame."""
        return self._aero_wrench_local

    @property
    def propeller_index(self) -> int | None:
        """Index of the propeller rigid prim inside each vehicle layout."""
        return self._propeller_index

    @property
    def aerodynamics(self):
        """Aerodynamic model used by this vehicle batch."""
        return self._aerodynamics

    @property
    def thrusters(self):
        """Single-propeller thrust model used by this vehicle batch."""
        return self._thrusters
