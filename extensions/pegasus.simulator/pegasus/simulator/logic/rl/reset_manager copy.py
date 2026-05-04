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

    def __init__(self, vehicle, device: str = "cuda"):
        """Initializes the manager and stores the initial spawn poses."""
        self.vehicle = vehicle
        self.device = device
        self.n_vehicles = vehicle.n_vehicles

        self._init_pos = vehicle._init_pos
        self._init_ori = vehicle._init_orientation

    def reset_envs(self, env_ids: torch.Tensor):
        """Teleports selected vehicles to their initial poses and zeros their velocities."""
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        pos = self._init_pos[env_ids]
        ori = self._init_ori[env_ids]
        vel = torch.zeros(n, 6, device=self.device)
        
        # Indices must be a long tensor for Isaac Sim views
        idx = env_ids.to(dtype=torch.long, device=self.device)

        # Apply state overrides safely to root prims only
        view = self.vehicle._root_prims
        view.set_world_poses(positions=pos, orientations=ori, indices=idx)
        view.set_velocities(velocities=vel, indices=idx)

    def reset_all(self):
        """Resets all vehicles in the batch to their initial states."""
        ids = torch.arange(self.n_vehicles, device=self.device)
        self.reset_envs(ids)

    @property
    def init_pos(self):
        """Returns the initial spawn positions of the vehicles."""
        return self._init_pos