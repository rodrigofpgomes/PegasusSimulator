"""
| File: rl_backend.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Backend connecting batched vehicles to vectorized RL environments.
"""

__all__ = ["RLBackend"]

import torch
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state_batch import StateBatch

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

class RLBackend(Backend):
    """
    Connect a batched vehicle to its vectorized RL environment.

    Stores actuator references, per-part loads and the latest vehicle state.
    The vehicle reads these buffers during each physics step.
    """

    def __init__(self, n_vehicles: int, action_mode: str = "direct_force", device: str | None = None):
        """Initialize the batched backend and its control mode."""
        super().__init__(config=None)

        self._n_vehicles = n_vehicles
        self._action_mode = action_mode

        # Initialized later in start()
        self._parts_per_vehicle = None
        self._device = device

        self._forces = None
        self._external_forces = None

        self._torques = None
        self._external_torques = None

        self._input_reference = None
        self._state_cache = None
        self._received_first_state = False

        self._goal_marker_view = None
        self._goal_marker_paths = []

        self._vehicle = None

    # -------------------------------------------
    # Properties
    # -------------------------------------------

    @property
    def n_vehicles(self) -> int:
        """Returns the number of vehicles in the batch."""
        return self._n_vehicles

    @property
    def parts_per_vehicle(self):
        """Returns the number of articulated parts per vehicle."""
        return self._parts_per_vehicle

    @property
    def device(self):
        """Returns the computation device (e.g., 'cuda')."""
        return self._device

    # -------------------------------------------
    # Backend Interface
    # -------------------------------------------

    def initialize(self, vehicle):
        """Stores the vehicle batch reference after spawning."""
        self._vehicle = vehicle

    def start(self):
        """Allocates PyTorch buffers once the simulation timeline starts."""
        self._n_vehicles = self.vehicle.n_vehicles
        self._parts_per_vehicle = self.vehicle.parts_per_vehicle
        self._device = self.vehicle.device or self._device

        # A glider also commands control surfaces, so its action width may
        # exceed the number of physical rotors
        num_rotors = self.vehicle._thrusters._num_rotors
        actuator_dim = getattr(self.vehicle, "actuator_dim", num_rotors)

        self._input_reference = torch.zeros((self._n_vehicles, actuator_dim), dtype=torch.float32, device=self._device)
        
        self._forces = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._torques = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._external_forces = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._external_torques = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)

        self._received_first_state = False

        # Tasks call this mode direct_force; vehicles name the same mode forces_torques
        vehicle_mode = "forces_torques" if self._action_mode == "direct_force" else self._action_mode
        self.vehicle.set_input_mode(vehicle_mode)

    def stop(self):
        """Callback for when the simulation stops (unused)."""
        pass

    def reset(self):
        """Clear backend commands, disturbances and cached state on a full reset."""
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._external_forces.zero_()
            self._external_torques.zero_()
            self._input_reference.zero_()
            self._state_cache.zero_()
        self._received_first_state = False

    def update(self, dt: float):
        """Allocate a requested body wrench to rotors in rotor_velocity mode."""
        # Direct force and direct rotor-speed modes already hold their commands
        if not self._received_first_state or self._action_mode != "rotor_velocity":
            return

        # Convert the requested body thrust and torque to rotor velocities
        body_index = self.vehicle.body_index
        u_1 = self._forces[:, body_index, 2]
        tau = self._torques[:, body_index, :]
        self._input_reference = self.vehicle.force_and_torques_to_velocities(u_1, tau)

    def update_state(self, state: StateBatch):
        """Caches the new physics state after a simulation step."""
        self._state_cache = torch.cat([
            state.position,
            state.linear_velocity,
            state.attitude,
            state.angular_velocity,
        ], dim=-1)
        self._received_first_state = True

    def update_sensor(self, sensor_type: str, data):
        """Processes generic sensor updates (unused)."""
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """Processes graphical sensor updates (unused)."""
        pass

    def input_reference(self) -> torch.Tensor:
        """Return rotor speeds and any additional actuator commands."""
        return self._input_reference

    def external_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the external forces and torques to be applied to the vehicles."""
        return self._external_forces, self._external_torques

    # -------------------------------------------
    # VecEnv Interface
    # -------------------------------------------

    def set_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """Receives new forces and torques from the RL environment."""
        self._forces = forces
        self._torques = torques

    def set_external_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """Receives new external forces and torques from the RL environment."""
        self._external_forces = forces
        self._external_torques = torques

    def set_state(self, env_ids: torch.Tensor, positions: torch.Tensor, attitudes: torch.Tensor, 
                        linear_velocity: torch.Tensor | None = None, angular_velocity: torch.Tensor | None = None):
        """Replace selected state rows and clear commands from their previous episodes."""
        if self._state_cache is None or env_ids.numel() == 0:
            return

        # Keep the remaining vehicles' commands and disturbances unchanged
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._external_forces[env_ids] = 0.0
        self._external_torques[env_ids] = 0.0
        self._input_reference[env_ids] = 0.0

        lin_vel = linear_velocity if linear_velocity is not None else torch.zeros((env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype)
        ang_vel = angular_velocity if angular_velocity is not None else torch.zeros((env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype)

        self._state_cache[env_ids, 0:3] = positions
        self._state_cache[env_ids, 3:6] = lin_vel
        self._state_cache[env_ids, 6:10] = attitudes
        self._state_cache[env_ids, 10:13] = ang_vel
        self._received_first_state = True

    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the currently stored forces and torques."""
        return self._forces, self._torques

    def get_state(self) -> torch.Tensor:
        """Returns the current state cache."""
        return self._state_cache

    # -------------------------------------------
    # Utilities
    # -------------------------------------------

    def create_goal_markers(self, root_path: str = "/World/GoalMarkers", size: float = 0.15, color: tuple[float, float, float] = (0.29, 0.388, 0.878)):
        """Spawns physical cubes in the simulation to visually track goals."""
        if self.vehicle is None:
            raise RuntimeError("RLBackend.create_goal_markers() called before vehicle initialize().")

        stage = self.vehicle._world.stage

        if not stage.GetPrimAtPath(root_path).IsValid():
            prim_utils.create_prim(root_path, "Xform")

        self._goal_marker_root = root_path
        self._goal_marker_paths = []

        for i in range(self._n_vehicles):
            prim_path = f"{root_path}/goal_{i}"
            if not stage.GetPrimAtPath(prim_path).IsValid():
                prim_utils.create_prim(prim_path, "Cube", translation=[0.0, 0.0, -100.0], scale=[size, size, size])

            cube = UsdGeom.Cube(stage.GetPrimAtPath(prim_path))
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self._goal_marker_paths.append(prim_path)

        self._goal_marker_view = XFormPrimView(prim_paths_expr=f"{root_path}/goal_*", name="goal_marker_view")

    def update_goal_markers(self, positions: torch.Tensor, env_ids: torch.Tensor):
        """Moves the visual goal trackers to new positions."""
        if self._goal_marker_view is None:
            return

        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        positions = positions.to(device=self._device, dtype=torch.float32)
        indices = env_ids.to(device=self._device, dtype=torch.long)
        
        orientations = torch.zeros((positions.shape[0], 4), device=self._device, dtype=torch.float32)
        orientations[:, 0] = 1.0

        self._goal_marker_view.set_world_poses(positions=positions, orientations=orientations, indices=indices)

    def hide_goal_markers(self, env_ids: torch.Tensor):
        """Hides the goals by translating them deeply underground."""
        if self._goal_marker_view is None:
            return

        indices = env_ids.to(device=self._device, dtype=torch.long)
        positions = torch.zeros((indices.numel(), 3), device=self._device, dtype=torch.float32)
        positions[:, 2] = -100.0

        orientations = torch.zeros((indices.numel(), 4), device=self._device, dtype=torch.float32)
        orientations[:, 0] = 1.0

        self._goal_marker_view.set_world_poses(positions=positions, orientations=orientations, indices=indices)
