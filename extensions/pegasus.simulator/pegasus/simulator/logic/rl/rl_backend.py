"""
| File: rl_backend.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Backend used to interface the multirotor simulator with vectorized RL environments.
"""

__all__ = ["RLBackend"]

import torch
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

class RLBackend(Backend):
    """
    Backend for RL training with vectorized multirotor environments. Handles forces, 
    torques, and state caching between the simulation physics steps and the RL policy.
    """

    def __init__(self, n_vehicles: int, action_mode: str = "direct_force", inner_loop: bool = False):
        """Initializes the backend with vehicle count, action mode, and control loop preference."""
        super().__init__(config=None)

        self._n_vehicles = n_vehicles
        self._action_mode = action_mode
        self._inner_loop = inner_loop

        # Initialized later in start()
        self._parts_per_vehicle = None
        self._device = None

        self._forces = None
        self._torques = None
        self._input_ref = None
        self._state_cache = None
        self._received_first_state = False

        self._Kr = None
        self._Kw = None

        self._goal_marker_view = None
        self._goal_marker_paths = []

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
        self._device = self.vehicle.device

        self._forces = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._torques = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._input_ref = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)

        self._received_first_state = False

        self._Kr = torch.diag(torch.tensor([3.5, 3.5, 3.5], dtype=torch.float32, device=self._device))
        self._Kw = torch.diag(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self._device))

        self.vehicle.set_input_mode(self._action_mode)

    def stop(self):
        """Callback for when the simulation stops (unused)."""
        pass

    def reset(self):
        """Clears all force, torque, and state buffers."""
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._input_ref.zero_()
            self._state_cache.zero_()
        self._received_first_state = False

    def update(self, dt: float):
        """Transforms requested RL actions into forces or rotor speeds before the physics step."""
        if not self._received_first_state:
            return

        if self._action_mode == "rotor_velocity":
            if self._inner_loop:
                # Get the desired forces
                F_des = self._forces[:, 0, :]

                # Get the current state
                q = self._state_cache[:, 6:10]
                w = self._state_cache[:, 10:13]
                R = quaternion_to_matrix(q)
               
                # Get the current axis Z_B (given by the last column of the rotation matrix)
                Z_B = R[:, :,2]

                # Compute the desired total thrust in Z_B direction (u_1)
                u_1 = torch.sum(F_des * Z_B, dim=1)

                # Compute the desired body-frame axis Z_b
                Z_b_des = F_des / torch.linalg.norm(F_des, dim=1, keepdim=True)

                # Desired yaw (fixed to zero)
                yaw_ref = torch.zeros((self._n_vehicles,), dtype=torch.float32, device=self._device)

                # Compute X_C_des
                X_c_des = torch.stack((torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref),), dim=1)

                # Compute Y_b_des
                Z_b_cross_X_c = torch.cross(Z_b_des, X_c_des, dim=1)
                Y_b_des = Z_b_cross_X_c / torch.linalg.norm(Z_b_cross_X_c, dim=1, keepdim=True)

                # Compute X_b_des
                X_b_des = torch.cross(Y_b_des, Z_b_des, dim=1)

                # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
                R_des = torch.stack((X_b_des, Y_b_des, Z_b_des), dim=2)

                # Compute the rotation error
                e_R_mat = torch.matmul(R_des.transpose(1, 2), R) - torch.matmul(R.transpose(1, 2), R_des)
                e_R = 0.5 * self.vee_batch(e_R_mat)

                # desired angular velocity
                w_des = torch.zeros_like(w)

                # Compute the angular velocity error
                e_w = w - w_des

                # Compute the torques to apply on the rigid body
                tau = -(e_R @ self._Kr.T) - (e_w @ self._Kw.T)
            else:
                # Direct force and torque bypass
                u_1 = self._forces[:, 0, 2]
                tau = self._torques[:, 0, :]

            # Convert forces/torques to rotor angular velocities
            self._input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

    def update_state(self, state: StateBatch):
        """Caches the new physics state after a simulation step."""
        self._state_cache = torch.cat([
            state.position,
            state.linear_body_velocity,
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
        """Returns the desired rotor velocities (used in 'rotor_velocity' mode)."""
        return self._input_ref

    # -------------------------------------------
    # VecEnv Interface
    # -------------------------------------------

    def set_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """Receives new forces and torques from the RL environment."""
        self._forces = forces
        self._torques = torques

    def set_state_for_envs(self, env_ids: torch.Tensor, positions: torch.Tensor, attitudes: torch.Tensor, 
                           linear_body_velocity: torch.Tensor | None = None, angular_velocity: torch.Tensor | None = None):
        """Overrides the state matrix for selected environments (e.g., during resets)."""
        if self._state_cache is None or env_ids.numel() == 0:
            return

        lin_vel = linear_body_velocity if linear_body_velocity is not None else torch.zeros((env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype)
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

    @staticmethod
    def vee_batch(S):
        """Computes the vee map operation on a batch of skew-symmetric matrices."""
        return torch.stack((-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]), dim=1)