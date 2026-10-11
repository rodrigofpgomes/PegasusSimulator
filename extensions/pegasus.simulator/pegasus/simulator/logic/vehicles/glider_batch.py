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


InputMode = Literal["rotor_velocity_direct", "forces_torques"]


# =============================================================================
# Configuration
# =============================================================================

class GliderBatchConfig:
    """ Class used to configure a GliderBatch vehicle. """

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
        
        # Default thrust curve and aerodynamic model for the batched gliders
        self.thrust_curve = GliderThrustCurveBatch(
            config=dict(thrust_curve_cfg or {}),
            n_vehicles=n_vehicles,
            device=self.device,
        )

        aero_cfg = dict(aerodynamics_cfg or {})
        geometry = dict(aero_cfg.get("geometry", {}))

        # The standalone EasyGlider USDA places /body at its aerodynamic reference origin
        geometry.setdefault("r_origin", (0.0, 0.0, 0.0))
        geometry.setdefault("r_com", (0.0, 0.0, 0.0))
        aero_cfg["geometry"] = geometry

        self.aerodynamics = GliderAerodynamicsBatch(config=aero_cfg, n_vehicles=n_vehicles, device=self.device)

        # Default onboard sensors for the batched gliders.
        # These are currently disabled until batch sensor support is enabled.
        self.sensors = tuple(sensors or ())

        # Default graphical sensors for the batched gliders
        self.graphical_sensors = tuple(graphical_sensors or ())

        # Default OmniGraph graphs for the batched gliders
        self.graphs = tuple(graphs or ())

        # Backends used to send commands to the batched vehicles.
        # This can be a PX4, ROS 2, or custom backend implementation, or an empty list if no backend is required.
        self.backends = tuple(backends or ())

        # Name of the puller rotor prim in the USD file
        self.propeller_prim_name = str(propeller_prim_name)

        axis = torch.as_tensor(propeller_axis_body, dtype=torch.float32, device=self.device)
        if axis.shape != (3,) or float(torch.linalg.vector_norm(axis)) <= 1e-9:
            raise ValueError("propeller_axis_body must be a nonzero three-component vector")
        
        # Save the propeller axis in the body frame as a normalized torch tensor.
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
        """Initialize the batched EasyGlider and its four-channel input."""
        if config is None:
            config = GliderBatchConfig(n_vehicles=n_vehicles)

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

        self._aerodynamics = config.aerodynamics
        self._thrusters = config.thrust_curve

        self._propeller_prim_name = config.propeller_prim_name
        self._propeller_axis_body = config.propeller_axis_body.to(
            device=self.device, dtype=torch.float32
        )

        self._input_mode: InputMode = "rotor_velocity_direct"
        self.actuator_dim = 4

        self._propeller_index: int | None = None
        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None

        # A direct command can be used for standalone runs without an RL backend
        self._actuator_reference = torch.zeros((self.n_vehicles, 4), dtype=torch.float32, device=self.device)
        self._direct_reference_mask = torch.zeros(self.n_vehicles, dtype=torch.bool, device=self.device)
        self._surface_deflections = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)
        self._aero_wrench_local = None
        self._flight_reset = None
        self._wind_world = torch.zeros((self.n_vehicles, 3), device=self.device)

    # =============================================================================
    # Lifecycle
    # =============================================================================

    def start(self) -> None:
        """Initialize prim views, resolve the propeller prim and allocate buffers."""
        
        self.initialize()

        self._resolve_propeller_index()

        self._forces = torch.zeros(
            (self.n_vehicles, self.parts_per_vehicle, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._torques = torch.zeros_like(self._forces)

        # Optional model-specific initialization hook.
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
        """Resolve the rigid-prim index of the Glider nose propeller."""
        first_vehicle_paths = self._vehicle_prims.prim_paths[: self.parts_per_vehicle]
        path_name_to_index = {path.rsplit("/", 1)[-1]: index for index, path in enumerate(first_vehicle_paths)}

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
    # Actuation API
    # =============================================================================

    def _format_actuator_reference(self, reference) -> torch.Tensor:
        """
        Format actuator command as:

            [Omega_p, delta_e, delta_a, delta_r]

        with shape (n_vehicles, 4).
        """

        command = torch.as_tensor(
            reference,
            dtype=torch.float32,
            device=self.device,
        )

        # Single vehicle command: [Omega, de, da, dr]
        if command.ndim == 1:
            if command.shape != (4,):
                raise ValueError(
                    "A single actuator reference must contain exactly 4 values "
                    "[Omega_p, delta_e, delta_a, delta_r]."
                )

            command = command.unsqueeze(0)

        # Must now be (N, 4)
        if command.ndim != 2 or command.shape[1] != 4:
            raise ValueError(
                "Actuator reference must have shape (4,) or "
                f"({self.n_vehicles}, 4), got {tuple(command.shape)}."
            )

        # Broadcast one command to all vehicles
        if command.shape[0] == 1 and self.n_vehicles > 1:
            command = command.expand(
                self.n_vehicles,
                -1,
            )

        elif command.shape[0] != self.n_vehicles:
            raise ValueError(
                f"Expected {self.n_vehicles} actuator references, "
                f"got {command.shape[0]}."
            )

        return command.clone()

    def set_actuator_reference(self, actuator_reference) -> None:
        """
        Set a direct four-channel command for every vehicle in the batch.

        Args:
            actuator_reference: [Omega_p, delta_e, delta_a, delta_r] with
                shape (4,) or (n_vehicles, 4).
        """
        self._actuator_reference = self._format_actuator_reference(actuator_reference)
        self._direct_reference_mask[:] = True

    def configure_flight_reset(self, positions, orientations, air_velocity_body, minimum_airspeed, actuator_reference, wind_world=(0.0, 0.0, 0.0)):
        """Configure persistent airborne initial conditions for full and partial resets.

        Args:
            positions: World ENU positions [m], one per instance, above the terrain.
            orientations: Body-to-world unit quaternions [w,x,y,z].
            air_velocity_body: Air-relative FLU velocity [m/s], including intended angle of attack.
            minimum_airspeed: Positive regime-specific bound [m/s]; supplied by the caller, not a stall estimate.
            actuator_reference: [puller, elevator, aileron, rudder] in rad/s and rad.
            wind_world: Wind velocity in world ENU [m/s].
        """
        def batch(value, width):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.shape == (width,):
                tensor = tensor.expand(self.n_vehicles, -1)
            if tensor.shape != (self.n_vehicles, width) or not torch.isfinite(tensor).all():
                raise ValueError(f"Expected finite ({self.n_vehicles}, {width}) initial data")
            return tensor.clone()
        pos = batch(positions, 3)
        ori = batch(orientations, 4)
        norm = torch.linalg.vector_norm(ori, dim=-1, keepdim=True)
        if (norm < 1e-6).any():
            raise ValueError("Initial attitude quaternion must be nonzero")
        ori /= norm
        air = batch(air_velocity_body, 3)
        if minimum_airspeed <= 0 or (torch.linalg.vector_norm(air, dim=-1) < minimum_airspeed).any():
            raise ValueError("Air-relative initial velocity is below the configured regime bound")
        cmd = self._format_actuator_reference(actuator_reference)
        if not torch.isfinite(cmd).all() or (cmd[:, 0] < self._thrusters.min_rotor_velocity).any() or (cmd[:, 0] > self._thrusters.max_rotor_velocity).any():
            raise ValueError("Invalid initial actuator command or puller limit exceeded")
        self._wind_world = batch(wind_world, 3)
        self._flight_reset = (pos, ori, air, cmd)

    def reset_in_flight(self, env_ids=None):
        """Restore configured flight conditions without changing other instances."""
        from pegasus.simulator.logic.rl.reset_manager import ResetManager
        if self._flight_reset is None:
            raise RuntimeError("Call configure_flight_reset with an explicit airspeed bound and trim first")
        ids = torch.arange(self.n_vehicles, device=self.device) if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        pos, ori, air, cmd = self._flight_reset
        ground_velocity = quaternion_apply(ori[ids], air[ids]) + self._wind_world[ids]
        minimum = self._thrusters.min_rotor_velocity
        span = self._thrusters.max_rotor_velocity - minimum
        rotor_norm = 2.0 * (cmd[ids, 0:1] - minimum) / span - 1.0
        ResetManager([self], device=self.device).reset_envs(ids, init_overrides={
            "position_offset": pos[ids] - self._init_pos[ids], "orientations": ori[ids],
            "linear_velocity": ground_velocity, "rotor_norm": rotor_norm})
        self._actuator_reference[ids] = cmd[ids]
        self._direct_reference_mask[ids] = True
        self._aerodynamics.set_wind_body(quaternion_apply(quaternion_invert(self.state.attitude), self._wind_world))

    def clear_actuator_reference(self, env_ids: torch.Tensor | None = None) -> None:
        """Return command authority to the first backend for selected vehicles."""
        if env_ids is None:
            self._direct_reference_mask.zero_()
            if not self._backends:
                self._actuator_reference.zero_()
        else:
            self._direct_reference_mask[env_ids] = False
            if not self._backends:
                self._actuator_reference[env_ids] = 0.0


    # =============================================================================
    # Physics update
    # =============================================================================

    def update(self, dt: float) -> None:
        """
        Apply aerodynamic and propulsive loads for the selected control mode.

        In rotor velocity mode, the backend or direct command supplies one
        puller speed and three surface deflections. Aerodynamics acts on /body;
        propeller thrust acts at the rotor prim and reaction torque on /body.

        Args:
            dt (float): Time since the previous physics step, in seconds.
        """
        if self._sim_running is False:
            return

        for backend in self._backends:
            backend.update(dt)

        if self._forces is None or self._torques is None:
            return

        self._forces.zero_()
        self._torques.zero_()
        self._aero_wrench_local = None

        if self._input_mode == "forces_torques":
            if not self._backends:
                return

            if not hasattr(self._backends[0], "get_forces_and_torques"):
                raise RuntimeError("The selected backend does not implement get_forces_and_torques().")

            forces, torques = self._backends[0].get_forces_and_torques()
            self._forces.copy_(torch.as_tensor(forces, dtype=torch.float32, device=self.device))
            self._torques.copy_(torch.as_tensor(torques, dtype=torch.float32, device=self.device))

        else:
            command = self._get_actuator_reference()
            omega_p = command[:, 0:1]
            delta = command[:, 1:4]         # [de, da, dr]
            self._surface_deflections.copy_(delta)

            # --------------------------------------------------------------
            # Aerodynamics
            # --------------------------------------------------------------
            if self._flight_reset is not None:
                self._aerodynamics.set_wind_body(quaternion_apply(quaternion_invert(self._state.attitude), self._wind_world))
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
            self._thrusters.set_input_reference(omega_p)
            thrust, _, reaction_moment = self._thrusters.update(self._state, dt)

            thrust = torch.as_tensor(thrust, dtype=torch.float32, device=self.device)
            reaction_moment = torch.as_tensor(reaction_moment, dtype=torch.float32, device=self.device)

            _, part_attitudes = self._vehicle_prims.get_world_poses()

            part_attitudes = torch.as_tensor(part_attitudes, dtype=torch.float32, device=self.device).reshape(self.n_vehicles, self.parts_per_vehicle, 4)
            body_attitude = part_attitudes[:, self.body_index, :]
            rotor_attitude = part_attitudes[:, self._propeller_index, :]

            axis_body = self._propeller_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)

            # +x_B expressed in world
            axis_world = quaternion_apply(body_attitude, axis_body)

            # Same physical direction expressed in rotor-local coordinates
            axis_rotor = quaternion_apply(quaternion_invert(rotor_attitude), axis_world)

            # Apply the puller thrust at the physical propeller rigid prim. 
            prop_force_rotor = thrust * axis_rotor
            self._forces[:, self._propeller_index, :] += prop_force_rotor

            # The scalar reaction moment acts along the puller axis in body coordinates
            self._torques[:, self.body_index, :] += reaction_moment.unsqueeze(-1) * axis_body

        # If the first backend implements and returns external forces and torques, apply them to the vehicle batch.
        if self._backends and hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()
            
            if result is not None:
                external_forces, external_torques = result
                self._forces += torch.as_tensor(external_forces, dtype=torch.float32, device=self.device)
                self._torques += torch.as_tensor(external_torques, dtype=torch.float32, device=self.device)

        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    def _get_actuator_reference(self) -> torch.Tensor:
        """Combine per-vehicle direct commands with the first backend command."""
        if not self._backends or self._direct_reference_mask.all():
            # A backend is unnecessary when every vehicle has a direct command
            return self._actuator_reference

        reference = self._format_actuator_reference(self._backends[0].input_reference())
        self._actuator_reference = torch.where(
            self._direct_reference_mask.unsqueeze(-1),
            self._actuator_reference,
            reference,
        )
        return self._actuator_reference

    # =============================================================================
    # Public helpers
    # =============================================================================

    def set_input_mode(self, input_mode: InputMode) -> None:
        """
        Set the Glider command mode. Supported modes are:

        "rotor_velocity_direct": Backend/direct command is '[Omega_p, delta_e, delta_a, delta_r]'.
        "forces_torques": Backend directly supplies per-part forces and torques.
        """

        if input_mode not in ("rotor_velocity_direct", "forces_torques"):
            raise ValueError("GliderBatch input_mode must be 'rotor_velocity_direct' or 'forces_torques'.")
        
        self._input_mode = input_mode

    @property
    def actuator_reference(self) -> torch.Tensor:
        """Latest '[Omega_p, delta_e, delta_a, delta_r]' command."""
        return self._actuator_reference

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
