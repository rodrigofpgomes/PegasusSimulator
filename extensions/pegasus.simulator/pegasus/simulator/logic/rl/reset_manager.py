"""
| File: reset_manager.py
| Author: [Your Name]
| Description: Utility class responsible for safely resetting rigid bodies in Isaac Sim.
| License: BSD-3-Clause. Copyright (c) 2026, [Your Name]. All rights reserved.
|
| Notes:
| PhysX maintains internal simulation state such as velocities and accumulated forces.
| Therefore, resetting an object by only modifying the USD transform is not sufficient.
| The reset must write directly into the PhysX buffers (RigidBodyView on this case)
| and explicitly zero the velocities.
|
| Resets should occur AFTER the completion of a physics step and BEFORE the
| next step begins. This ordering is guaranteed by the VecEnv runner.
"""

import torch
import numpy as np

__all__ = ["ResetManager"]


class ResetManager:
    """
    Utility class that handles safe environment resets for multiple vehicles.

    This class performs teleportation-style resets by writing the desired pose
    and velocities directly into the PhysX buffers. This avoids inconsistencies
    caused by modifying only the USD transform while PhysX still maintains its
    own internal dynamic state.

    The reset operation supports both partial resets (specific environments)
    and full resets (all environments).
    """

    def __init__(
        self,
        vehicle,
        init_positions: torch.Tensor,     # [N, 3]
        init_orientations: torch.Tensor,  # [N, 4] 
        device: str = "cuda",
    ):
        """
        Initialize the ResetManager.

        Args:
            vehicle: Vehicle object containing the PhysX rigid body view.
            init_positions (torch.Tensor): Initial world positions for each vehicle [N, 3].
            init_orientations (torch.Tensor): Initial orientations as quaternions [N, 4] (wxyz).
            device (str): Torch device used for tensor operations.
        """
        self.vehicle = vehicle
        self.init_pos = init_positions.to(device)
        self.init_ori = init_orientations.to(device)
        self.n_vehicles = init_positions.shape[0]
        self.device = device

    """
    Reset API
    """

    def reset_envs(self, env_ids: torch.Tensor):
        """
        Reset only the specified environments.

        This method teleports the selected vehicles back to their initial
        pose and explicitly resets both linear and angular velocities.

        Args:
            env_ids (torch.Tensor): Tensor containing the indices of the
                environments that should be reset.
        """
        # If there are no environments to reset, exit early
        if env_ids.numel() == 0:
            return

        # Number of environments being reset
        n = env_ids.numel()

        # Extract the initial pose for the selected environments
        positions = self.init_pos[env_ids]  # [k, 3]
        orientations = self.init_ori[env_ids]  # [k, 4]

        # Reset both linear and angular velocities to zero
        # Format: [vx, vy, vz, wx, wy, wz]
        velocities = torch.zeros(n, 6, device=self.device)

        # Write the reset state directly into the PhysX buffers
        self._write_to_physx(pos, ori, vel, env_ids)

        # Access the PhysX rigid body view associated with the vehicle
        view = self.vehicle._rigid_body_view

        # Update the world pose directly in the PhysX buffer
        view.set_world_poses(positions=positions, orientations=orientations, indices=env_ids.cpu().numpy())

        # Explicitly reset linear and angular velocities
        view.set_velocities(velocities=velocities, indices=env_ids.cpu().numpy())


    def reset_all(self):
        """
        Reset all environments.

        This method resets every vehicle managed by this ResetManager.
        """
        ids = torch.arange(self.n_vehicles, device=self.device)
        self.reset_envs(ids)
