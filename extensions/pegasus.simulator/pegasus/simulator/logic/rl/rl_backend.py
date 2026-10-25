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

    def __init__(self, n_vehicles: int, action_mode: str = "direct_force", inner_loop: bool = False,
                 device: str | None = None, goal_markers_enabled: bool = True):
        """Initialize the batched backend, attitude loop and optional goal markers."""
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

        self._goal_markers_enabled = goal_markers_enabled
        self._goal_marker_view = None

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
        """Convert body-frame force, and optionally moment, into rotor speeds."""
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
        """Compute multirotor thrust and moments for a fixed-yaw attitude target.

        The task supplies the desired force in the body frame. Rotate it to
        world axes only to construct the desired attitude. The inner loop
        computes its own body-frame moment from attitude and angular-rate errors.
        """
        body_index = self.vehicle.body_index
        body_force = self._forces[:, body_index, :]
        q = self._state_cache[:, 6:10]
        w = self._state_cache[:, 10:13]

        # Convert the requested local force to world axes before constructing the desired attitude
        F_des = quaternion_apply(q, body_force)
        R = quaternion_to_matrix(q)

        # Get the current body Z axis from the last column of the rotation matrix
        Z_B = R[:, :, 2]

        # Project the desired force onto body Z to obtain nonnegative collective thrust
        u_1 = torch.sum(F_des * Z_B, dim=-1).clamp_min(0.0)

        # Align the desired body Z axis with the force; retain the current axis for zero force
        force_norm = torch.linalg.vector_norm(F_des, dim=-1, keepdim=True)
        active = force_norm.squeeze(-1) > 1e-6
        Z_b_des = torch.where(
            active.unsqueeze(-1),
            F_des / force_norm.clamp_min(1e-6),
            Z_B,
        )

        # Set desired yaw to zero, so the reference horizontal axis is world X
        X_c_des = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand_as(Z_b_des)
        Y_c_des = torch.tensor([0.0, 1.0, 0.0], device=self._device).expand_as(Z_b_des)

        # Construct desired body Y; use world Y if the force is parallel to world X
        Y_b_des = torch.cross(Z_b_des, X_c_des, dim=-1)
        parallel = torch.linalg.vector_norm(Y_b_des, dim=-1, keepdim=True) < 1e-6
        Y_b_des = torch.where(parallel, torch.cross(Z_b_des, Y_c_des, dim=-1), Y_b_des)
        Y_b_des = Y_b_des / torch.linalg.vector_norm(Y_b_des, dim=-1, keepdim=True)

        # Complete the desired orthonormal attitude matrix
        X_b_des = torch.cross(Y_b_des, Z_b_des, dim=-1)
        R_des = torch.stack((X_b_des, Y_b_des, Z_b_des), dim=-1)

        # Compute the geometric attitude error in body coordinates
        e_R_mat = R_des.transpose(1, 2) @ R - R.transpose(1, 2) @ R_des
        e_R = 0.5 * torch.stack(
            (e_R_mat[:, 2, 1], e_R_mat[:, 0, 2], e_R_mat[:, 1, 0]), dim=-1
        )

        # Desired body rate is zero; compute the moment from attitude and rate errors
        w_des = torch.zeros_like(w)
        e_w = w - w_des
        tau = -(e_R @ self._Kr.T) - (e_w @ self._Kw.T)

        return u_1, tau

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
        """Satisfy the backend interface; RL observations use vehicle state."""
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """Satisfy the backend interface; graphical sensors are not used."""
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
        """Create visual goal cubes when visualization is enabled."""
        if not self._goal_markers_enabled:
            return

        if self.vehicle is None:
            raise RuntimeError("RLBackend.create_goal_markers() called before vehicle initialize().")

        stage = self.vehicle._world.stage

        if not stage.GetPrimAtPath(root_path).IsValid():
            prim_utils.create_prim(root_path, "Xform")

        for i in range(self._n_vehicles):
            prim_path = f"{root_path}/goal_{i}"
            if not stage.GetPrimAtPath(prim_path).IsValid():
                prim_utils.create_prim(prim_path, "Cube", translation=[0.0, 0.0, -100.0], scale=[size, size, size])

            cube = UsdGeom.Cube(stage.GetPrimAtPath(prim_path))
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

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
        """Move selected goal markers below the scene until they are needed again."""
        if self._goal_marker_view is None:
            return

        indices = env_ids.to(device=self._device, dtype=torch.long)
        positions = torch.zeros((indices.numel(), 3), device=self._device, dtype=torch.float32)
        positions[:, 2] = -100.0

        orientations = torch.zeros((indices.numel(), 4), device=self._device, dtype=torch.float32)
        orientations[:, 0] = 1.0

        self._goal_marker_view.set_world_poses(positions=positions, orientations=orientations, indices=indices)
