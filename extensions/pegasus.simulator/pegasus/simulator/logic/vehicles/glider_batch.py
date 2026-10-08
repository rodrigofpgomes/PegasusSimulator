"""
| File: glider_batch.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Batched Glider vehicle interface for Pegasus Simulator.
|
| Actuator command per vehicle:
|   [delta_e, delta_a, delta_r, Omega_p] in [rad, rad, rad, rad/s]
|
| This class applies the aerodynamic forces/moments that are produced by 
| an injected batched aerodynamic model, while the nose propeller is handled 
| by an injected single-propeller thrust model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert

from pegasus.simulator.logic.thrusters.glider_thrust_curve_batch import GliderThrustCurveBatch
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch


InputMode = Literal["actuators", "forces_torques"]


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
        """ 
        Initialize the GliderBatchConfig with the given parameters. 
        """
         
        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

         # Define the same device that is running the simulation
        self.device = PegasusInterface()._world_settings["device"]
        
        # Stage prefix used when spawning the batched vehicles in the world
        self.stage_prefix = "glider"
        
        # The USD file that describes the visual appearance of the vehicle, as well as properties such as mass and inertia
        self.usd_file = ""
        
        # Default thrust curve and aerodynamic model for the batched gliders
        self.thrust_curve = GliderThrustCurveBatch(config=dict(thrust_curve_cfg) if thrust_curve_cfg else {}, n_vehicles=n_vehicles, device = self.device)
        self.aerodynamics = GliderAerodynamicsBatch(config=dict(aerodynamics_cfg) if aerodynamics_cfg else {}, n_vehicles=n_vehicles, device = self.device)

        # Default onboard sensors for the batched gliders.
        # These are currently disabled until batch sensor support is enabled.
        #self.sensors = [Barometer(device=device), IMU(device=device), Magnetometer(device=device), GPS(device=device)]
        self.sensors = tuple(sensors or ())

        # Default graphical sensors for the batched gliders
        self.graphical_sensors = tuple(graphical_sensors or ())

        # Default OmniGraph graphs for the batched gliders
        self.graphs = tuple(graphs or ())

        # Backends used to send commands to the batched vehicles.
        # This can be a PX4, ROS 2, or custom backend implementation, or an empty list if no backend is required.
        self.backends = tuple(backends or ())

        # Optional: Name of the rotor prims in the USD file.
        self.propeller_prim_name = str(propeller_prim_name)

        axis = torch.as_tensor(propeller_axis_body, dtype=torch.float32, device=self.device)
        if axis.shape != (3,) or float(torch.linalg.vector_norm(axis)) <= 1e-9:
            raise ValueError("propeller_axis_body must contain exactly 3 components and cannot be the zero vector")
        
        # Save the propeller axis in the body frame as a normalized torch tensor.
        self.propeller_axis_body = axis / torch.linalg.vector_norm(axis)

        
class GliderBatch(VehicleBatch):
    """
    Batched fixed-wing Glider vehicle.

    This class represents a fixed-wing glider with aerodynamic control surfaces. 
    The physical inputs are the aerodynamic control-surface deflections and the 
    angular velocity of the single nose propeller.
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

        self._input_mode: InputMode = "actuators"

        self._propeller_index: int | None = None
        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None

        # Latest actuator command. This is useful for logging/debugging and also
        # allows a test script to command the vehicle directly without a backend.
        self._actuator_reference = torch.zeros((self.n_vehicles, 4), dtype=torch.float32, device=self.device)
        self._has_direct_reference = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

        if self._input_mode == "actuators":
            if self._aerodynamics is None:
                raise RuntimeError(
                    "GliderBatch requires config.aerodynamics in 'actuators' mode. "
                    "Pass the batched Glider aerodynamic model through "
                    "GliderBatchConfig(aerodynamics=...)."
                )
            if self._thrusters is None:
                raise RuntimeError(
                    "GliderBatch requires config.thrust_curve in 'actuators' mode. "
                    "Pass the single-propeller batched thrust model through "
                    "GliderBatchConfig(thrust_curve=...)."
                )

        # Optional model-specific initialization hook.
        for model in (self._aerodynamics, self._thrusters):
            initialize = getattr(model, "initialize", None)
            if callable(initialize):
                initialize(self)

    def stop(self) -> None:
        """No additional stop action is required."""
        return

    # ------------------------------------------------------------------
    # Prim layout
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Actuation API
    # ------------------------------------------------------------------

    def _format_actuator_reference(self, reference) -> torch.Tensor:
        """
        Format actuator command as:

            [delta_e, delta_a, delta_r, Omega_p]

        with shape (n_vehicles, 4).
        """

        command = torch.as_tensor(
            reference,
            dtype=torch.float32,
            device=self.device,
        )

        # Single vehicle command: [de, da, dr, Omega]
        if command.ndim == 1:
            if command.shape != (4,):
                raise ValueError(
                    "A single actuator reference must contain exactly 4 values "
                    "[delta_e, delta_a, delta_r, Omega_p]."
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
        This method allow to set the actuator reference directly, without the need of a backend.

        Parameters
        actuator_reference: '[delta_e, delta_a, delta_r, Omega_p]' or a batched tensor with shape (n_vehicles, 4).
        """
        self._actuator_reference = self._format_actuator_reference(actuator_reference)
        self._has_direct_reference = True

    def clear_actuator_reference(self) -> None:
        """Return command authority to the first backend, if one is configured."""
        self._has_direct_reference = False


    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Compute and apply aerodynamic and propulsive loads each physics step in the
        actuators mode. Aerodynamic force and moment are applied to the '/body' rigid
        prim. The propeller thrust is applied directly to the ``/rotor`` rigid prim 
        along the configured propeller axis. 
        
        This method also apresents a mode that allows the first backend to directly
        supply per-part forces and torques, which are applied to the vehicle batch.
        """
        if self._sim_running is False:
            return

        for backend in self._backends:
            backend.update(dt)

        if self._forces is None or self._torques is None:
            return

        self._forces.zero_()
        self._torques.zero_()

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
            delta = command[:, :3]          # [de, da, dr]
            omega_p = command[:, 3:4]       # (n_vehicles, 1)

            # --------------------------------------------------------------
            # Aerodynamics
            # --------------------------------------------------------------
            aero_force, aero_torque = self._aerodynamics.update(self._state, delta, dt)

            aero_force = torch.as_tensor(aero_force, dtype=torch.float32, device=self.device)
            aero_torque = torch.as_tensor(aero_torque, dtype=torch.float32, device=self.device)

            # The aerodynamic backend is expected to return the equivalent wrench referenced to the body centre of mass.
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

            # Propeller reaction torque acts on the fuselage.
            self._torques[:, self.body_index, :] += reaction_moment

        # If the first backend implements and returns external forces and torques, apply them to the vehicle batch.
        if self._backends and hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()
            
            if result is not None:
                external_forces, external_torques = result
                self._forces += torch.as_tensor(external_forces, dtype=torch.float32, device=self.device)
                self._torques += torch.as_tensor(external_torques, dtype=torch.float32, device=self.device)

        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    def _get_actuator_reference(self) -> torch.Tensor:
        """Return the direct command or the command supplied by the first backend."""
        if self._has_direct_reference:
            return self._actuator_reference

        if not self._backends:
            # No backend and no explicit command: leave all actuators at zero.
            return self._actuator_reference

        reference = self._backends[0].input_reference()
        self._actuator_reference = self._format_actuator_reference(reference)
        return self._actuator_reference

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_input_mode(self, input_mode: InputMode) -> None:
        """
        Set the Glider command mode. Supported modes are:

        "actuators": Backend/direct command is '[delta_e, delta_a, delta_r, Omega_p]'.
        "forces_torques": Backend directly supplies per-part forces and torques.
        """

        if input_mode not in ("actuators", "forces_torques"):
            raise ValueError("GliderBatch input_mode must be 'actuators' or 'forces_torques'.")
        
        self._input_mode = input_mode

    @property
    def actuator_reference(self) -> torch.Tensor:
        """Latest '[delta_e, delta_a, delta_r, Omega_p]' command."""
        return self._actuator_reference

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