"""
| File: reset_manager.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Safe reset of rigid bodies in Isaac Sim using the RigidPrim API.
|
| The reset manager supports randomized initial states and optional initialization
| overrides for physically consistent flight conditions, such as trimmed cruise.
|
| Supported 'init_overrides' entries:
|   mask             (N,)    - environments to which the overrides are applied
|   position_offset  (N, 3)  - position offset relative to the initial spawn [m]
|   orientations     (N, 4)  - absolute orientation quaternion (w, x, y, z)
|   linear_velocity  (N, 3)  - absolute linear velocity in the world frame [m/s]
|   angular_velocity (N, 3)  - absolute angular velocity in the body frame [rad/s]
|   rotor_norm       (N, R)  - normalized rotor velocities in [-1, 1]
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from pegasus.simulator.logic.transforms import quaternion_apply

__all__ = ["ResetManager", "GoalCfg", "InitStateCfg"]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GoalCfg:
    """Configuration for randomized goal positions."""
    
    goal_pos_xy_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    goal_pos_z_range: List[float] = field(default_factory=lambda: [0.5, 1.5])


@dataclass
class InitStateCfg:
    """Configuration for randomized vehicle initial states."""

    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 1.0
    max_angle_deg: float = 180.0
    max_position: "float | Sequence[float]" = 0.5
    guidance_prob: float = 0.1


# =============================================================================
# Reset manager
# =============================================================================

class ResetManager:
    """Reset physical vehicle batches, cached states, rotor speeds and goals."""

    def __init__(self, vehicles, device: str = "cuda", goal_cfg=None, init_state_cfg=None):
        """Store the vehicle batches and allocate goal/reset bookkeeping."""

        self.vehicles = vehicles
        self.device = device
        self.goal_cfg = goal_cfg
        self.init_state_cfg = init_state_cfg
        if not vehicles:
            raise ValueError("ResetManager requires at least one vehicle")
        self.main_vehicle = vehicles[0]
        self.n_vehicles = self.main_vehicle.n_vehicles
        if any(v.n_vehicles != self.n_vehicles for v in vehicles):
            raise ValueError("All comparison batches must have the same size")
        self._init_pos = self.main_vehicle._init_pos
        self._init_ori = self.main_vehicle._init_orientation
        self._last_guided = torch.zeros(self.n_vehicles, dtype=torch.bool, device=device)
        self._goal_pos = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_vel = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_acc = torch.zeros(self.n_vehicles, 3, device=device)
        self._goal_jerk = torch.zeros(self.n_vehicles, 3, device=device)

    # =============================================================================
    # Vehicle reset
    # =============================================================================

    def reset_envs(
        self,
        env_ids: torch.Tensor,
        randomize_goals: bool = False,
        randomize_state: bool = False,
        init_overrides: dict | None = None,
    ) -> None:
        """
        Reset selected vehicles and optionally randomize or override their state.

        Args:
            env_ids (torch.Tensor): Vehicle indices to reset.
            randomize_goals (bool): Sample new goal positions when configured.
            randomize_state (bool): Sample state perturbations when configured.
            init_overrides (dict | None): Selected pose, velocity and rotor
                values to apply after randomization. Values are indexed by
                their position within env_ids, not by global vehicle index.
        """
        if env_ids.numel() == 0:
            return
        
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        n = env_ids.numel()
        num_rotors = self.main_vehicle._thrusters._num_rotors

        # Randomize the initial vehicle state within the configured limits
        if randomize_state and self.init_state_cfg is not None:
            pos_limit = self._axis_limit(self.init_state_cfg.max_position, "max_position")
            pos_offset = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0) * pos_limit.unsqueeze(0)
            
            ori_rand = self._random_quaternions(n)
            
            lin_vel_rand = torch.empty(n, 3, device=self.device).uniform_(-self.init_state_cfg.max_linear_velocity, self.init_state_cfg.max_linear_velocity)
            ang_vel_rand = torch.empty(n, 3, device=self.device).uniform_(-self.init_state_cfg.max_angular_velocity, self.init_state_cfg.max_angular_velocity)
            
            rotor_norm = torch.empty((n, num_rotors), device=self.device).uniform_(-1.0, 0.0)
            
            # Some environments use the nominal spawn state as guidance
            guided = torch.rand(n, device=self.device) < self.init_state_cfg.guidance_prob
            self._last_guided[env_ids] = guided
            
        # Initialize the reset state without random perturbations
        else:
            pos_offset = torch.zeros(n, 3, device=self.device)
            
            ori_rand = None
            
            lin_vel_rand = torch.zeros(n, 3, device=self.device)
            ang_vel_rand = torch.zeros(n, 3, device=self.device)
            
            rotor_norm = torch.full((n, num_rotors), -1.0, device=self.device)

            guided = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._last_guided[env_ids] = False

        # Apply explicit initial flight conditions to the requested rows
        if init_overrides is not None:
            mask = init_overrides.get("mask")
            
            # Apply the overrides to all selected environments when no mask is provided
            if mask is None:
                mask = torch.ones(n, dtype=torch.bool, device=self.device)
                
            mask = mask.to(device=self.device, dtype=torch.bool)
            
            if mask.any():
                
                # Start from the nominal orientation when none was randomized
                if ori_rand is None:
                    ori_rand = self.main_vehicle._init_orientation[env_ids].to(device=self.device, dtype=torch.float32).clone()
                    
                # Replace the requested state components for the masked environments
                if "position_offset" in init_overrides:
                    pos_offset[mask] = init_overrides["position_offset"].to(self.device)[mask]
                if "orientations" in init_overrides:
                    ori_rand[mask] = init_overrides["orientations"].to(self.device)[mask]
                if "linear_velocity" in init_overrides:
                    lin_vel_rand[mask] = init_overrides["linear_velocity"].to(self.device)[mask]
                if "angular_velocity" in init_overrides:
                    ang_vel_rand[mask] = init_overrides["angular_velocity"].to(self.device)[mask]
                if "rotor_norm" in init_overrides:
                    rotor_norm[mask] = init_overrides["rotor_norm"].to(self.device)[mask]
                    
                # Explicit overrides take precedence over guided nominal starts
                guided = guided & ~mask
                self._last_guided[env_ids] = guided

        # Apply the same selected environment IDs to every comparison vehicle
        for vehicle in self.vehicles:
            base_pos = vehicle._init_pos[env_ids].to(device=self.device, dtype=torch.float32)
            pos = base_pos + pos_offset
            
            base_ori = vehicle._init_orientation[env_ids].to(device=self.device, dtype=torch.float32)
            ori = base_ori if ori_rand is None else ori_rand.clone()
            
            lin_vel = lin_vel_rand.clone()
            ang_vel = ang_vel_rand.clone()

            # Reset guided environments to the nominal initial pose with zero velocity
            if guided.any():
                pos[guided] = base_pos[guided]
                ori[guided] = base_ori[guided]
                lin_vel[guided] = 0.0
                ang_vel[guided] = 0.0

            # Normalize the orientation shared by PhysX and the cached state
            norm = torch.linalg.vector_norm(ori, dim=-1, keepdim=True)
            if not torch.isfinite(ori).all() or (norm < 1e-8).any():
                raise ValueError("Reset orientation must be finite and nonzero")
            ori = ori / norm
            # Set PhysX root prims, then synchronize the vehicle and its backends
            self._reset_vehicle_pose(vehicle, env_ids, pos, ori, lin_vel, ang_vel)
            vehicle.set_state_batch(env_ids, positions=pos, attitudes=ori, linear_velocity=lin_vel, angular_velocity=ang_vel)
            
            # Convert normalized rotor commands to angular velocities
            min_w = vehicle._thrusters.min_rotor_velocity.to(device=self.device).reshape(-1)
            max_w = vehicle._thrusters.max_rotor_velocity.to(device=self.device).reshape(-1)
            
            omega = (rotor_norm + 1.0) * 0.5 * (max_w - min_w).unsqueeze(0) + min_w.unsqueeze(0)
            
            vehicle._thrusters._velocity[env_ids] = omega
            # The ShuttleGlider adapter stores the normalized reset value for action history
            if hasattr(vehicle._thrusters, "_reset_rotor_norm"):
                vehicle._thrusters._reset_rotor_norm[env_ids] = rotor_norm

        # Goals belong to the new episode and are updated after the physical reset
        if randomize_goals and self.goal_cfg is not None:
            self._randomize_goals(env_ids)
        else:
            self._goal_pos[env_ids] = self.main_vehicle._init_pos[env_ids].to(
                device=self.device, dtype=torch.float32)
            self._goal_vel[env_ids] = torch.zeros_like(self._goal_pos[env_ids])
            self._goal_acc[env_ids] = 0.0
            self._goal_jerk[env_ids] = 0.0

    def _reset_vehicle_pose(self, vehicle, env_ids, pos, ori, lin_vel, ang_vel) -> None:
        """Write world pose and velocity to the root PhysX prims only."""

        idx = env_ids.to(dtype=torch.long, device=self.device)
        # PhysX root velocity is in the world frame at the root COM; the reset API uses body rates.
        ori = ori / torch.linalg.vector_norm(ori, dim=-1, keepdim=True).clamp_min(1e-12)
        omega_world = quaternion_apply(ori, ang_vel)
        # Reduced-coordinate articulations reject writes to non-root links. Joint
        # constraints carry the root motion to the other physical parts in PhysX.
        vehicle._root_prims.set_world_poses(positions=pos, orientations=ori, indices=idx)
        vehicle._root_prims.set_velocities(
            velocities=torch.cat((lin_vel, omega_world), dim=-1), indices=idx
        )

    def _axis_limit(self, value, name: str) -> torch.Tensor:
        """Return a three-axis limit from a scalar or length-three setting."""

        limit = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if limit.ndim == 0:
            return limit.repeat(3)
        if limit.numel() != 3:
            raise ValueError(f"{name} must be a scalar or a length-3 sequence")
        return limit.reshape(3)

    def _random_quaternions(self, n: int) -> torch.Tensor:
        """Sample unit quaternions with bounded rotation angle."""

        max_angle = self.init_state_cfg.max_angle_deg * torch.pi / 180.0
        u = torch.rand(n, device=self.device)
        v = torch.rand(n, device=self.device)
        phi = 2.0 * torch.pi * u
        cos_theta = 1.0 - 2.0 * v
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))
        axis = torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta], dim=1)
        angle = torch.rand(n, device=self.device) * max_angle
        half = 0.5 * angle
        q = torch.zeros(n, 4, device=self.device)
        q[:, 0] = torch.cos(half)
        q[:, 1:] = axis * torch.sin(half).unsqueeze(1)
        return q

    def reset_all(self) -> None:
        """Reset every environment using the configured randomization."""

        ids = torch.arange(self.n_vehicles, device=self.device)
        self.reset_envs(ids, randomize_goals=self.goal_cfg is not None,
                        randomize_state=self.init_state_cfg is not None)

    # =============================================================================
    # Goals
    # =============================================================================

    def _randomize_goals(self, env_ids: torch.Tensor) -> None:
        """Sample goal positions for selected environments."""

        if self.goal_cfg is None:
            return
        xy_low, xy_high = self.goal_cfg.goal_pos_xy_range
        z_low, z_high = self.goal_cfg.goal_pos_z_range
        self._goal_pos[env_ids, :2] = torch.zeros_like(self._goal_pos[env_ids, :2]).uniform_(xy_low, xy_high)
        self._goal_pos[env_ids, :2] += self.init_pos[env_ids, :2]
        self._goal_pos[env_ids, 2] = torch.zeros_like(self._goal_pos[env_ids, 2]).uniform_(z_low, z_high)
        self._goal_vel[env_ids] = 0.0
        self._goal_acc[env_ids] = 0.0
        self._goal_jerk[env_ids] = 0.0
        guided = self._last_guided[env_ids]
        if guided.any():
            guided_ids = env_ids[guided]
            self._goal_pos[guided_ids] = self.init_pos[guided_ids].to(device=self.device, dtype=torch.float32)
            self._goal_vel[guided_ids] = 0.0
            self._goal_acc[guided_ids] = 0.0
            self._goal_jerk[guided_ids] = 0.0

    def set_goal_cfg(self, goal_cfg) -> None:
        """Replace the goal sampling configuration for later resets."""
        self.goal_cfg = goal_cfg

    # =============================================================================
    # Properties
    # =============================================================================

    @property
    def init_pos(self):
        """Initial spawn positions of the main vehicle batch."""
        return self._init_pos

    @property
    def goal_pos(self):
        """Current goal positions in world coordinates."""
        return self._goal_pos

    @property
    def goal_vel(self):
        """Current goal velocities in world coordinates."""
        return self._goal_vel

    @property
    def goal_acc(self):
        """Current goal accelerations in world coordinates."""
        return self._goal_acc

    @property
    def goal_jerk(self):
        """Current goal jerks in world coordinates."""
        return self._goal_jerk
