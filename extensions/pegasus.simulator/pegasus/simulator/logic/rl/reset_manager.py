"""
| File: reset_manager.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Safe reset of rigid bodies in Isaac Sim using the RigidPrim API.
"""

import torch

__all__ = ["ResetManager"]


class ResetManager:
    """Manages vehicle resets by directly modifying root PhysX buffers to avoid articulation warnings."""

    def __init__(self, vehicles, device: str = "cuda", goal_cfg=None):
        """Initializes the manager and stores the initial spawn poses."""
        self.vehicles = vehicles
        self.device = device
        self.goal_cfg = goal_cfg

        self.main_vehicle = vehicles[0]
        self.n_vehicles = self.main_vehicle.n_vehicles

        self._init_pos = self.main_vehicle._init_pos
        self._init_ori = self.main_vehicle._init_orientation

        self._goal_pos = torch.zeros(self.n_vehicles, 3, device=device)

    def reset_envs(self, env_ids: torch.Tensor, randomize_goals: bool = False):
        """Teleports selected vehicles to their initial poses and zeros their velocities."""
        if env_ids.numel() == 0:
            return

        for vehicle in self.vehicles:
            self._reset_vehicle(vehicle, env_ids)
            vehicle.set_state_batch(env_ids, positions=self._init_pos[env_ids], attitudes=self._init_ori[env_ids], linear_velocity=torch.zeros_like(self._init_pos[env_ids]))

        if randomize_goals and self.goal_cfg is not None:
            self._randomize_goals(env_ids)

    def _reset_vehicle(self, vehicle, env_ids):
        # Get initial poses for the specified env_ids
        pos = vehicle._init_pos[env_ids]
        ori = vehicle._init_orientation[env_ids]
        vel = torch.zeros(env_ids.numel(), 6, device=self.device)

        # Indices must be a long tensor for Isaac Sim views
        idx = env_ids.to(dtype=torch.long, device=self.device)

        # Apply state overrides safely to root prims only
        view = vehicle._root_prims
        view.set_world_poses(positions=pos, orientations=ori, indices=idx)
        view.set_velocities(velocities=vel, indices=idx)


    def reset_all(self):
        """Resets all vehicles in the batch to their initial states."""

        ids = torch.arange(self.n_vehicles, device=self.device)

        self.reset_envs(ids)


    def _randomize_goals(self, env_ids: torch.Tensor):
        """Samples new goal positions around the original spawn point."""
        xy_low, xy_high = self.goal_cfg.goal_pos_xy_range
        z_low, z_high = self.goal_cfg.goal_pos_z_range

        self._goal_pos[env_ids, :2] = torch.zeros_like(self._goal_pos[env_ids, :2]).uniform_(xy_low, xy_high)
        self._goal_pos[env_ids, :2] += self.init_pos[env_ids, :2]
        self._goal_pos[env_ids, 2] = torch.zeros_like(self._goal_pos[env_ids, 2]).uniform_(z_low, z_high)


    def set_goal_cfg(self, goal_cfg):
        """Updates the goal randomization configuration."""
        self.goal_cfg = goal_cfg
        
    @property
    def init_pos(self):
        """Returns the initial spawn positions of the vehicles."""
        return self._init_pos

    @property
    def goal_pos(self):
        """Returns the goal positions of the vehicles."""
        return self._goal_pos