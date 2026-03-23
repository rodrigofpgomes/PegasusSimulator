"""
| File: reset_manager.py
| Description: Safe reset of rigid bodies in Isaac Sim using RigidPrim API.
| License: BSD-3-Clause.

Reset is done on vehicle._root_prims — one RigidPrim per vehicle root.
This avoids the PhysX warning:
    "non-root articulation links whose transforms cannot be set directly"

The root prim is the articulation root (quadrotor_0, quadrotor_1, ...).
PhysX propagates the pose to all child links automatically.

_root_prims is created in VehicleBatch.initialize() with expression
"{stage_prefix}_*" which matches exactly the N root prims.
"""
import torch

__all__ = ["ResetManager"]


class ResetManager:
    """
    Resets vehicles by writing pose + velocity into the root PhysX buffers.
    Uses vehicle._root_prims (one per vehicle) instead of _vehicle_prims
    (which includes non-root articulation links and causes PhysX warnings).
    """

    def __init__(self, vehicle, device: str = "cuda"):
        self.vehicle    = vehicle
        self.device     = device
        self.n_vehicles = vehicle.n_vehicles

        # Initial poses stored by VehicleBatch._spawn_batch()
        self._init_pos = vehicle._init_pos          # [N, 3]
        self._init_ori = vehicle._init_orientation  # [N, 4]  wxyz

    def reset_envs(self, env_ids: torch.Tensor):
        """
        Teleports selected vehicles to their initial pose and zeros velocities.
        Operates on root prims only — no PhysX warnings.

        Args:
            env_ids: LongTensor of environment indices to reset.
        """
        if env_ids.numel() == 0:
            return

        n   = env_ids.numel()
        pos = self._init_pos[env_ids]                    # [k, 3]
        ori = self._init_ori[env_ids]                    # [k, 4]
        vel = torch.zeros(n, 6, device=self.device)      # lin(3) + ang(3)
        # indices must be a torch LongTensor — resolve_indices calls .to()
        idx = env_ids.to(dtype=torch.long, device=self.device)

        # _root_prims: one RigidPrim per vehicle root — safe to set directly
        view = self.vehicle._root_prims
        view.set_world_poses(positions=pos, orientations=ori, indices=idx)
        view.set_velocities(velocities=vel, indices=idx)

    def reset_all(self):
        ids = torch.arange(self.n_vehicles, device=self.device)
        self.reset_envs(ids)

    @property
    def init_pos(self):
        return self._init_pos