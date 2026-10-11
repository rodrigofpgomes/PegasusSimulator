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
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_to_matrix
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

class RLBackend(Backend):
    """
    Connect a batched vehicle to its vectorized RL environment.

    Stores actuator references, per-part loads and the latest vehicle state.
    The vehicle reads these buffers during each physics step.
    """

    def __init__(self, n_vehicles: int, action_mode: str = "direct_force", inner_loop: bool = False, device: str | None = None):
        """Initialize the batched backend and optional multirotor attitude loop."""
        super().__init__(config=None)

        self._n_vehicles = n_vehicles
        self._action_mode = action_mode
        self._inner_loop = inner_loop

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

        self._Kr = None
        self._Kw = None

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
        # The attitude loop allocates rotor speeds through MultirotorBatch only
        if self._inner_loop:
            if self._action_mode != "rotor_velocity" or not isinstance(self.vehicle, MultirotorBatch):
                raise ValueError("inner_loop requires a MultirotorBatch in rotor_velocity mode")

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

        if self._inner_loop:
            # Body-frame proportional attitude and angular-rate gains
            self._Kr = torch.diag(torch.tensor([3.5, 3.5, 3.5], dtype=torch.float32, device=self._device))
            self._Kw = torch.diag(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self._device))

        # Set the vehicle control mode based on the action mode defined in the task
        # Modes: rotor_velocity_direct, rotor_velocity or direct_force
        # rotor_velocity is only available for multirotor vehicles
        self.vehicle.set_input_mode(self._action_mode)

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
        """Convert a requested force and moment into multirotor speed commands."""
        # Direct force and direct rotor-speed modes already hold their commands
        if not self._received_first_state or self._action_mode != "rotor_velocity":
            return

        if self._inner_loop:
            # Align body Z with the desired world force before rotor allocation
            u_1, tau = self._attitude_inner_loop()
        else:
            # Allocate the requested body thrust and moment directly
            body_index = self.vehicle.body_index
            u_1 = self._forces[:, body_index, 2]
            tau = self._torques[:, body_index, :]

        self._input_reference = self.vehicle.force_and_torques_to_velocities(u_1, tau)

    def _attitude_inner_loop(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute body thrust and attitude moments from the requested local force."""
        body_index = self.vehicle.body_index
        body_force = self._forces[:, body_index, :]
        attitude = self._state_cache[:, 6:10]
        angular_velocity = self._state_cache[:, 10:13]

        # Commands are stored in the /body frame; attitude control uses world axes
        desired_force_world = quaternion_apply(attitude, body_force)
        rotation = quaternion_to_matrix(attitude)
        body_z_world = rotation[:, :, 2]

        # Only positive thrust along the current body Z can be produced
        thrust = torch.sum(desired_force_world * body_z_world, dim=-1).clamp_min(0.0)

        # Point desired body Z along the commanded world force. A zero force
        # keeps the current direction and produces zero thrust.
        force_norm = torch.linalg.vector_norm(desired_force_world, dim=-1, keepdim=True)
        active = force_norm.squeeze(-1) > 1e-6
        desired_z = torch.where(
            active.unsqueeze(-1),
            desired_force_world / force_norm.clamp_min(1e-6),
            body_z_world,
        )

        # Fixed world X sets the desired yaw; world Y handles parallel axes
        world_x = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand_as(desired_z)
        world_y = torch.tensor([0.0, 1.0, 0.0], device=self._device).expand_as(desired_z)
        desired_y = torch.cross(desired_z, world_x, dim=-1)
        parallel = torch.linalg.vector_norm(desired_y, dim=-1, keepdim=True) < 1e-6
        desired_y = torch.where(parallel, torch.cross(desired_z, world_y, dim=-1), desired_y)
        desired_y = desired_y / torch.linalg.vector_norm(desired_y, dim=-1, keepdim=True)
        desired_x = torch.cross(desired_y, desired_z, dim=-1)
        desired_rotation = torch.stack((desired_x, desired_y, desired_z), dim=-1)

        # Combine requested body moments with attitude correction and rate damping
        error_matrix = desired_rotation.transpose(1, 2) @ rotation - rotation.transpose(1, 2) @ desired_rotation
        attitude_error = 0.5 * torch.stack(
            (error_matrix[:, 2, 1], error_matrix[:, 0, 2], error_matrix[:, 1, 0]), dim=-1
        )
        moment = self._torques[:, body_index, :] - (attitude_error @ self._Kr.T) - (angular_velocity @ self._Kw.T)

        return thrust, moment

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
