"""
| File: shuttle_glider_batch.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Batched ShuttleGlider vehicle for Pegasus Simulator.
|
| The shuttle and the EasyGlider fuselage are represented by one merged rigid
| body in the corrected USDA. The five propellers remain individual rigid
| bodies:
|   rotor_0 ... rotor_3 : vertical shuttle lift rotors
|   rotor_puller         : EasyGlider nose propeller, physical axis +x_B
|
| Input command vector for the ShuttleGliderBatch is a 8-element vector:
|   [omega_0, omega_1, omega_2, omega_3, Omega_p, delta_e, delta_a, delta_r]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert
from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch
from pegasus.simulator.logic.thrusters.quadratic_thrust_curve_batch import QuadraticThrustCurveBatch
from pegasus.simulator.logic.thrusters.glider_thrust_curve_batch import GliderThrustCurveBatch
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch


InputMode = Literal["rotor_velocity_direct", "forces_torques"]


# =============================================================================
# Corrected merged-body geometry
# =============================================================================

# Shuttle body COM in the vehicle/root frame [m].
#_SHUTTLE_COM = (0.0, 0.0, 0.24)

# EasyGlider body COM after mounting the glider 0.15 m below the shuttle [m].
#_GLIDER_COM = (-0.23, 0.0, -0.2152)

# Corrected merged body COM: shuttle body (3.4207 kg) + glider body (1.475 kg).
#_MERGED_COM = (-0.069295504, 0.0, 0.102855159)

# Component CoMs in the /body frame [m].
_SHUTTLE_COM = (0, 0, 0.1197)
_GLIDER_COM = (0, 0, -0.2803)

# Physical CoM of the rigid body /body [m].
_MERGED_COM = (0, 0, -0.0031)

# Vector from /body origin to glider CoM [m].
_MERGED_R_ORIGIN = _GLIDER_COM
_MERGED_R_COM = tuple(g - c for g, c in zip(_GLIDER_COM, _MERGED_COM))

# =============================================================================
# RL compatibility adapter
# =============================================================================

class _CombinedRotorInterface:
    """
    Compatibility view exposing the five physical propellers as one rotor group.

    Existing RLBackend and ResetManager code use ``vehicle._thrusters`` to query
    the number of rotors, velocity limits and reset rotor speeds. Internally,
    however, the shuttle lift rotors and the EasyGlider puller use two different
    thrust models. This adapter keeps the old RL interface while preserving the
    physically different propulsion models.
    """

    def __init__(
        self,
        shuttle_thrusters: QuadraticThrustCurveBatch,
        glider_thruster: GliderThrustCurveBatch,
        n_vehicles: int,
        device,
    ) -> None:
        self._shuttle = shuttle_thrusters
        self._glider = glider_thruster
        self._num_rotors = 5
        self.n_vehicles = int(n_vehicles)
        self.device = torch.device(device)

        self.min_rotor_velocity = torch.cat((self._shuttle.min_rotor_velocity.reshape(-1), self._glider.min_rotor_velocity.reshape(-1))).to(device=self.device, dtype=torch.float32)

        self.max_rotor_velocity = torch.cat((self._shuttle.max_rotor_velocity.reshape(-1), self._glider.max_rotor_velocity.reshape(-1))).to(device=self.device, dtype=torch.float32)

        self._velocity = torch.zeros((self.n_vehicles, self._num_rotors), dtype=torch.float32, device=self.device)

        self._force = torch.zeros_like(self._velocity)

        # ResetManager stores this value for policy observations.
        self._reset_rotor_norm = torch.zeros_like(self._velocity)

        # Metadata retained for code that inspects rotor coefficients.
        self._rotor_constant = torch.cat((self._shuttle._rotor_constant.reshape(-1), self._glider._thrust_constant.reshape(-1))).to(device=self.device, dtype=torch.float32)

        self._rolling_moment_coefficient = torch.cat(
            (
                self._shuttle._rolling_moment_coefficient.reshape(-1),
                (self._glider._moment_constant * self._glider._thrust_constant).reshape(-1),
            )
        ).to(device=self.device, dtype=torch.float32)

        self._rot_dir = torch.cat((self._shuttle._rot_dir.reshape(-1).to(dtype=torch.float32), self._glider._reaction_moment_sign.reshape(-1))).to(device=self.device, dtype=torch.float32)

    def sync_to_models(self) -> None:
        """Push reset/current rotor velocities into the two physical models."""
        self._shuttle._velocity = self._velocity[:, :4].clone()
        self._glider._velocity = self._velocity[:, 4:5].clone()

    def sync_from_models(self) -> None:
        """Pull physical-model rotor states into the five-rotor RL view."""
        self._velocity[:, :4] = self._shuttle._velocity
        self._velocity[:, 4:5] = self._glider._velocity

        self._force[:, :4] = self._shuttle._force
        self._force[:, 4:5] = self._glider._force

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @property
    def force(self) -> torch.Tensor:
        return self._force


# =============================================================================
# Configuration
# =============================================================================

class ShuttleGliderBatchConfig2:
    """Configuration for :class:`ShuttleGliderBatch`."""
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

        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")

        self.device = PegasusInterface()._world_settings["device"]
        self.stage_prefix = "shuttle_glider"
        self.usd_file = ""

        # ------------------------------------------------------------------
        # Four shuttle lift rotors
        # ------------------------------------------------------------------
        
        cfg = dict(cfg or {})

        shuttle_thrust_cfg = cfg.get("shuttle_thrust_cfg", {})
        glider_thrust_cfg = cfg.get("glider_thrust_cfg", {})
        aerodynamics_cfg = cfg.get("aerodynamics_cfg", {})
    
        shuttle_cfg = dict(shuttle_thrust_cfg or {})
        shuttle_cfg.setdefault("num_rotors", 4)

        if int(shuttle_cfg["num_rotors"]) != 4:
            raise ValueError("ShuttleGliderBatch requires exactly four shuttle lift rotors")

        self.shuttle_thrust_curve = QuadraticThrustCurveBatch(
            config=shuttle_cfg,
            n_vehicles=n_vehicles,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # EasyGlider puller
        # ------------------------------------------------------------------
        puller_cfg = dict(glider_thrust_cfg or {})
        puller_cfg.setdefault("propeller_axis_body", list(propeller_axis_body))

        self.glider_thrust_curve = GliderThrustCurveBatch(
            config=puller_cfg,
            n_vehicles=n_vehicles,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # EasyGlider aerodynamic model on the merged rigid body
        # ------------------------------------------------------------------
        aero_cfg = dict(aerodynamics_cfg or {})
        geometry_cfg = dict(aero_cfg.get("geometry", {}))

        # The aerodynamic reference vector is measured from the merged shuttle+glider COM
        geometry_cfg.setdefault("r_origin", _MERGED_R_ORIGIN)
        geometry_cfg.setdefault("r_com", _MERGED_R_COM)

        aero_cfg["geometry"] = geometry_cfg
        aero_cfg.setdefault("eval_at_ref", True)

        self.aerodynamics = GliderAerodynamicsBatch(config=aero_cfg, n_vehicles=n_vehicles, device=self.device)
        
        print("r_origin efetivo [m]:", self.aerodynamics.r_origin.tolist())
        print("r_com efetivo [m]:", self.aerodynamics.r_com.tolist())
                
        print("eval_at_ref:", self.aerodynamics.eval_at_ref)

        self.sensors = tuple(sensors or ())
        self.graphical_sensors = tuple(graphical_sensors or ())
        self.graphs = tuple(graphs or ())
        self.backends = tuple(backends or ())

        self.shuttle_rotor_prim_names = tuple(shuttle_rotor_prim_names)
        if len(self.shuttle_rotor_prim_names) != 4:
            raise ValueError("shuttle_rotor_prim_names must contain exactly four names")

        self.propeller_prim_name = str(propeller_prim_name)

        self.shuttle_rotor_axis_body = self._normalise_axis(shuttle_rotor_axis_body, "shuttle_rotor_axis_body")
        self.propeller_axis_body = self._normalise_axis(propeller_axis_body, "propeller_axis_body")

    def _normalise_axis(self, axis, name: str) -> torch.Tensor:
        value = torch.as_tensor(axis, dtype=torch.float32, device=self.device)

        if value.shape != (3,):
            raise ValueError(f"{name} must contain exactly three components")

        norm = torch.linalg.vector_norm(value)
        if float(norm) <= 1e-9:
            raise ValueError(f"{name} cannot be the zero vector")

        return value / norm


# =============================================================================
# Vehicle
# =============================================================================

class ShuttleGliderBatch2(VehicleBatch):
    """
    Batched Shuttle + EasyGlider hybrid vehicle.

    Aerodynamic forces/moments from the fixed-wing model are always active.
    """

    def __init__(
        self,
        stage_prefix: str = "shuttle_glider",
        usd_file: str = "",
        vehicle_batch_id: int = 0,
        n_vehicles: int = 1,
        init_pos=None,
        init_orientation=None,
        spacing: float = 3.0,
        config: ShuttleGliderBatchConfig | None = None,
    ) -> None:

        if config is None:
            config = ShuttleGliderBatchConfig(n_vehicles=n_vehicles)

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

        self._shuttle_thrusters = config.shuttle_thrust_curve
        self._glider_thruster = config.glider_thrust_curve
        self._aerodynamics = config.aerodynamics

        # Compatibility interface expected by RLBackend / ResetManager.
        self._thrusters = _CombinedRotorInterface(self._shuttle_thrusters, self._glider_thruster, self.n_vehicles, self.device)

        self._shuttle_rotor_prim_names = config.shuttle_rotor_prim_names
        self._propeller_prim_name = config.propeller_prim_name

        self._shuttle_rotor_axis_body = config.shuttle_rotor_axis_body.to(device=self.device, dtype=torch.float32)
        self._propeller_axis_body = config.propeller_axis_body.to(device=self.device, dtype=torch.float32)

        self._input_mode: InputMode = "rotor_velocity_direct"

        self._shuttle_rotor_indices: tuple[int, ...] | None = None
        self._propeller_index: int | None = None

        self._rotor_positions_body: torch.Tensor | None = None
        self._allocation_matrix: torch.Tensor | None = None
        self._allocation_inv: torch.Tensor | None = None

        self._forces: torch.Tensor | None = None
        self._torques: torch.Tensor | None = None

        self._actuator_reference = torch.zeros((self.n_vehicles, 8), dtype=torch.float32, device=self.device)
        self._has_direct_reference = False

        self._surface_deflections = torch.zeros((self.n_vehicles, 3), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.initialize()

        self._resolve_prim_indices()

        self._forces = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        self._torques = torch.zeros_like(self._forces)

        for model in (self._aerodynamics, self._glider_thruster):
            initialize = getattr(model, "initialize", None)
            if callable(initialize):
                initialize(self)

    def stop(self) -> None:
        return

    # ------------------------------------------------------------------
    # Prim layout
    # ------------------------------------------------------------------

    def _resolve_prim_indices(self) -> None:
        first_vehicle_paths = self._vehicle_prims.prim_paths[: self.parts_per_vehicle]

        name_to_index = {path.rsplit("/", 1)[-1]: index for index, path in enumerate(first_vehicle_paths)}

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

        self._shuttle_rotor_indices = tuple(name_to_index[name] for name in self._shuttle_rotor_prim_names)

        self._propeller_index = name_to_index[self._propeller_prim_name]

        all_rotors = *self._shuttle_rotor_indices, self._propeller_index
    
        if self.body_index in all_rotors:
            raise RuntimeError("'/body' cannot also be a rotor prim")


    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_actuator_reference(self, reference) -> None:
        command = torch.as_tensor(reference, dtype=torch.float32, device=self.device)
        
        if command.shape != (self.n_vehicles, 8):
            raise ValueError(f"Expected actuator reference with shape ({self.n_vehicles}, 8), got {tuple(command.shape)}.")
        
        self._actuator_reference = command
        self._has_direct_reference = True

    def clear_actuator_reference(self) -> None:
        self._has_direct_reference = False

    def _get_actuator_reference(self) -> torch.Tensor:
        if self._has_direct_reference:
            return self._actuator_reference

        if not self._backends:
            return self._actuator_reference

        reference =  torch.as_tensor(self._backends[0].input_reference(), dtype=torch.float32, device=self.device)
        
        if reference.shape != (self.n_vehicles, 8):
            raise ValueError(f"Expected backend actuator reference with shape ({self.n_vehicles}, 8), got {tuple(reference.shape)}.")

        self._actuator_reference = reference
        
        return self._actuator_reference

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
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

            shuttle_omega = command[:, 0:4]
            glider_omega = command[:, 4:5]

            # ----------------------------------------------------------
            # Aerodynamic surfaces
            # ----------------------------------------------------------
            delta = command[:, 5:8]
            self._surface_deflections = delta.clone()

            # EasyGlider aerodynamics
            aero_force, aero_torque = self._aerodynamics.update(self._state, delta, dt)

            self._forces[:, self.body_index, :] += torch.as_tensor(aero_force, dtype=torch.float32, device=self.device)
            self._torques[:, self.body_index, :] += torch.as_tensor(aero_torque, dtype=torch.float32, device=self.device)
        
            # ----------------------------------------------------------
            # Synchronize possible RL reset rotor states into the two
            # physical propulsion models before applying the new command.
            # ----------------------------------------------------------
            self._thrusters.sync_to_models()

            # Four shuttle lift rotors
            self._shuttle_thrusters.set_input_reference(shuttle_omega)
            shuttle_force, _, shuttle_reaction_moment_z = self._shuttle_thrusters.update(self._state, dt)

            # EasyGlider nose propeller
            self._glider_thruster.set_input_reference(glider_omega)
            glider_thrust, _, glider_reaction_moment = self._glider_thruster.update(self._state, dt)

            self._thrusters.sync_from_models()

            # Express physical rotor axes in each rotor's local frame.
            # VehicleBatch applies each force with is_global=False.
            _, part_attitudes = self._vehicle_prims.get_world_poses()

            part_attitudes = torch.as_tensor(part_attitudes, dtype=torch.float32, device=self.device,).reshape(self.n_vehicles, self.parts_per_vehicle, 4)

            body_attitude = part_attitudes[:, self.body_index, :]

            # shuttle +z_B lift axis
            shuttle_axis_body = self._shuttle_rotor_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)

            shuttle_axis_world = quaternion_apply(body_attitude, shuttle_axis_body)

            for rotor_number, rotor_index in enumerate(self._shuttle_rotor_indices):
                rotor_attitude = part_attitudes[:, rotor_index, :]

                axis_rotor = quaternion_apply(quaternion_invert(rotor_attitude), shuttle_axis_world)

                self._forces[:, rotor_index, :] += shuttle_force[:, rotor_number : rotor_number + 1] * axis_rotor

            # Total reaction moment from the four vertical rotors is about
            # the body z-axis, following QuadraticThrustCurveBatch.
            self._torques[:, self.body_index, 2,] += shuttle_reaction_moment_z

            # EasyGlider physical +x_B puller axis
            puller_axis_body = self._propeller_axis_body.unsqueeze(0).expand(self.n_vehicles, -1)

            puller_axis_world = quaternion_apply(body_attitude, puller_axis_body)

            puller_attitude = part_attitudes[:, self._propeller_index, :]

            puller_axis_local = quaternion_apply(quaternion_invert(puller_attitude), puller_axis_world)

            # Applying thrust at the physical rotor prim lets PhysX generate
            # the moment associated with the propeller thrust-line offset.
            self._forces[:, self._propeller_index, :] += glider_thrust * puller_axis_local

            # The propeller aerodynamic reaction torque is a separate body
            # torque and must not be generated again from r x F.
            self._torques[:, self.body_index, :] += torch.as_tensor(glider_reaction_moment, dtype=torch.float32, device=self.device)

        # Optional backend disturbances.
        if self._backends and hasattr(self._backends[0], "external_forces_and_torques"):
            result = self._backends[0].external_forces_and_torques()

            if result is not None:
                external_forces, external_torques = result

                self._forces += torch.as_tensor(external_forces, dtype=torch.float32, device=self.device)
                self._torques += torch.as_tensor(external_torques, dtype=torch.float32, device=self.device)

        self.apply_forces_and_torques_all_parts(self._forces, self._torques)


    # ------------------------------------------------------------------
    # Mode / diagnostics
    # ------------------------------------------------------------------

    def set_input_mode(self, input_mode: InputMode) -> None:

        if input_mode not in ("rotor_velocity_direct", "forces_torques"):
            raise ValueError("Unsupported ShuttleGliderBatch input mode: " f"{input_mode}")

        self._input_mode = input_mode

    @property
    def actuator_dim(self) -> int:
        return 8

    @property
    def surface_deflections(self) -> torch.Tensor:
        """Latest [delta_e, delta_a, delta_r] [rad]."""
        return self._surface_deflections

    @property
    def actuator_reference(self) -> torch.Tensor:
        return self._actuator_reference

    @property
    def aerodynamics(self):
        return self._aerodynamics

    @property
    def shuttle_thrusters(self):
        return self._shuttle_thrusters

    @property
    def glider_thruster(self):
        return self._glider_thruster

    @property
    def thrusters(self):
        """Five-rotor RL compatibility interface."""
        return self._thrusters
