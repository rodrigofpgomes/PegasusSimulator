"""
| File: reset_manager.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Safe reset of rigid bodies in Isaac Sim using the RigidPrim API.
"""

from dataclasses import dataclass, field
from typing import List

import torch

__all__ = ["ResetManager", "GoalCfg", "InitStateCfg"]


@dataclass
class GoalCfg:
    """Goal randomization parameters for ResetManager."""
    goal_pos_xy_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    goal_pos_z_range:  List[float] = field(default_factory=lambda: [0.5, 1.5])


@dataclass
class InitStateCfg:
    """Initial state randomization parameters."""
    max_linear_velocity:  float = 1.0    # [m/s] per axis, uniform
    max_angular_velocity: float = 1.0    # [rad/s] per axis, uniform
    max_angle_deg:        float = 180.0  # max tilt angle for random orientation sampling
    max_position: float = 0.5 # max position offset from spawn for random position sampling

    guidance_prob: float = 0.1


class ResetManager:
    """Manages vehicle resets by directly modifying root PhysX buffers to avoid articulation warnings."""

    def __init__(self, vehicles, device: str = "cuda", goal_cfg=None, init_state_cfg=None):
        """Initializes the manager and stores the initial spawn poses."""
        self.vehicles = vehicles
        self.device = device
        self.goal_cfg = goal_cfg
        self.init_state_cfg = init_state_cfg

        self.main_vehicle = vehicles[0]
        self.n_vehicles = self.main_vehicle.n_vehicles

        self._init_pos = self.main_vehicle._init_pos
        self._init_ori = self.main_vehicle._init_orientation

        self._last_guided = torch.zeros(self.n_vehicles, dtype=torch.bool, device=device)

        self._goal_pos = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_vel = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_acc = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_jerk = torch.zeros(self.n_vehicles, 3, device=device)

    def reset_envs(self, env_ids: torch.Tensor, randomize_goals: bool = False, randomize_state: bool = False):
        """Teleports selected vehicles to initial/randomized poses and resets velocities/motors."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        n = env_ids.numel()
        num_rotors = self.main_vehicle._thrusters._num_rotors

        if randomize_state and self.init_state_cfg is not None:
            pos_offset = torch.empty(n, 3, device=self.device).uniform_(
                -self.init_state_cfg.max_position,
                self.init_state_cfg.max_position,
            )

            ori_rand = self._random_quaternions(n)

            lin_vel_rand = torch.empty(n, 3, device=self.device).uniform_(
                -self.init_state_cfg.max_linear_velocity,
                self.init_state_cfg.max_linear_velocity,
            )

            ang_vel_rand = torch.empty(n, 3, device=self.device).uniform_(
                -self.init_state_cfg.max_angular_velocity,
                self.init_state_cfg.max_angular_velocity,
            )

            # Same initial rotor normalised state for all vehicles in the same env.
            rotor_norm = torch.empty((n, num_rotors), device=self.device).uniform_(-1.0, 0.0)

            guided = torch.rand(n, device=self.device) < self.init_state_cfg.guidance_prob
            self._last_guided[env_ids] = guided
        else:
            pos_offset = torch.zeros(n, 3, device=self.device)
            ori_rand = None
            lin_vel_rand = torch.zeros(n, 3, device=self.device)
            ang_vel_rand = torch.zeros(n, 3, device=self.device)
            guided = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._last_guided[env_ids] = False

            # Assume the drone starts with the motors off.
            rotor_norm = torch.full((n, num_rotors), -1.0, device=self.device)


        for vehicle in self.vehicles:
            base_pos = vehicle._init_pos[env_ids].to(device=self.device, dtype=torch.float32)
            base_ori = vehicle._init_orientation[env_ids].to(device=self.device, dtype=torch.float32)

            pos = base_pos + pos_offset

            if ori_rand is None:
                ori = base_ori
            else:
                ori = ori_rand.clone()

            lin_vel = lin_vel_rand.clone()
            ang_vel = ang_vel_rand.clone()

            # Easy initial condition.
            if guided.any():
                pos[guided] = base_pos[guided]
                ori[guided] = base_ori[guided]
                lin_vel[guided] = 0.0
                ang_vel[guided] = 0.0

            self._reset_vehicle_pose(vehicle, env_ids, pos, ori, lin_vel, ang_vel)

            vehicle.set_state_batch(
                env_ids,
                positions=pos,
                attitudes=ori,
                linear_velocity=lin_vel,
                angular_velocity=ang_vel,
            )

            # Seed the RL state cache with the freshly applied pose/velocity.
            # The reset runs in the post-physics callback, so there is no physics
            # step between _reset_idx and _get_observations().
            for backend in vehicle._backends:
                if hasattr(backend, "set_state"):
                    backend.set_state(
                        env_ids,
                        positions=pos,
                        attitudes=ori,
                        linear_velocity=lin_vel,
                        angular_velocity=ang_vel,
                    )

            # Set the rotors' initial velocity, based on rotor_norm in [-1, 0].
            min_w = vehicle._thrusters.min_rotor_velocity.to(device=self.device)
            max_w = vehicle._thrusters.max_rotor_velocity.to(device=self.device)

            omega = (rotor_norm + 1.0) * 0.5 * (max_w - min_w).unsqueeze(0) + min_w.unsqueeze(0)

            vehicle._thrusters._velocity[env_ids] = omega
            vehicle._thrusters._reset_rotor_norm[env_ids] = rotor_norm

        if randomize_goals and self.goal_cfg is not None:
            self._randomize_goals(env_ids)
        else:
            self._goal_pos[env_ids] = self.main_vehicle._init_pos[env_ids].to(
                device=self.device, dtype=torch.float32
            )
            self._goal_vel[env_ids] = torch.zeros_like(self._goal_pos[env_ids])

    def _reset_vehicle_pose(self, vehicle, env_ids, pos, ori, lin_vel, ang_vel):
        """Writes pose (position + orientation) and 6-DoF velocity of the selected
        environments directly into the vehicle's RigidPrim view."""
        idx = env_ids.to(dtype=torch.long, device=self.device)
        vel6 = torch.cat([lin_vel, ang_vel], dim=1)
        view = vehicle._root_prims
        view.set_world_poses(positions=pos, orientations=ori, indices=idx)
        view.set_velocities(velocities=vel6, indices=idx)

    def _random_quaternions(self, n: int) -> torch.Tensor:
        """Samples ``n`` random unit quaternions (w, x, y, z) with tilt uniformly drawn
        in ``[0, max_angle_deg]`` about a uniformly random axis."""
        max_angle = self.init_state_cfg.max_angle_deg * torch.pi / 180.0

        # random unit axis
        u = torch.rand(n, device=self.device)
        v = torch.rand(n, device=self.device)

        phi = 2.0 * torch.pi * u
        cos_theta = 1.0 - 2.0 * v
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))

        axis = torch.stack(
            [
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                cos_theta,
            ],
            dim=1,
        )

        # random angle in [0, max_angle]
        angle = torch.rand(n, device=self.device) * max_angle

        half = 0.5 * angle

        q = torch.zeros(n, 4, device=self.device)
        q[:, 0] = torch.cos(half)
        q[:, 1:] = axis * torch.sin(half).unsqueeze(1)

        return q


    def reset_all(self):
        """Resets all vehicles in the batch to their initial states."""
        ids = torch.arange(self.n_vehicles, device=self.device)
        self.reset_envs(ids, randomize_goals=self.goal_cfg is not None,
                        randomize_state=self.init_state_cfg is not None)


    def _randomize_goals(self, env_ids: torch.Tensor):
        """Samples new goal positions around the original spawn point."""

        if self.goal_cfg is None:
            return

        #self._goal_pos[env_ids] = self.init_pos[env_ids] + torch.randn_like(self.init_pos[env_ids]) * self.goal_cfg.goal_pos_std   
        xy_low, xy_high = self.goal_cfg.goal_pos_xy_range
        z_low, z_high = self.goal_cfg.goal_pos_z_range

        self._goal_pos[env_ids, :2] = torch.zeros_like(self._goal_pos[env_ids, :2]).uniform_(xy_low, xy_high)
        self._goal_pos[env_ids, :2] += self.init_pos[env_ids, :2]
        self._goal_pos[env_ids, 2] = torch.zeros_like(self._goal_pos[env_ids, 2]).uniform_(z_low, z_high)

        self._goal_vel[env_ids] = torch.zeros_like(self._goal_pos[env_ids])
        self._goal_acc[env_ids] = torch.zeros_like(self._goal_pos[env_ids])
        self._goal_jerk[env_ids] = torch.zeros_like(self._goal_pos[env_ids])
        guided = self._last_guided[env_ids]

        if guided.any():
            guided_ids = env_ids[guided]
            
            self._goal_pos[guided_ids] = self.init_pos[guided_ids].to(device=self.device, dtype=torch.float32)
            self._goal_vel[guided_ids] = 0.0
            self._goal_acc[guided_ids] = 0.0
            self._goal_jerk[guided_ids] = 0.0


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

    @property
    def goal_vel(self):
        """Returns the goal velocities of the vehicles."""
        return self._goal_vel

    @property
    def goal_acc(self):
        """Returns the goal accelerations of the vehicles."""
        return self._goal_acc

    @property
    def goal_jerk(self):
        """Returns the goal jerks of the vehicles."""
        return self._goal_jerk