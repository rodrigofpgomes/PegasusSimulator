"""
| File: quadcopter_env.py
| Description: Quadcopter hover task - ep, ev → u (acceleration control).
| License: BSD-3-Clause.

Observation space (6):
    ep (3) - position error in inertial frame (goal_pos - current_pos)
    ev (3) - velocity error in inertial frame (goal_vel - current_vel)

Action space (3):
    action[0:3] → desired acceleration in the inertial frame, scaled from [-1,1] to [-20, 20] m/s^2, then converted to forces based on the robot mass.

Termination:
    terminated: z < 0.1 OR z > 2.0
    truncated:  episode_length_buf >= max_episode_length - 1

Reward:
    r = (lin_vel * scale) + (ang_vel * scale) + (dist_mapped * scale)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import torch

import numpy as np
from scipy.linalg import solve_discrete_are

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert



@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):
    """Configuration for the Quadcopter Environment."""
    
    # Spaces
    observation_space: int = 6
    action_space: int = 3
    state_space: int = 0

    # Timing
    episode_length_s: float = 10.0
    decimation: int = 2
    sim_dt: float = 0.01

    # Reward Scales
    lin_vel_error_reward_scale: float = -0.5
    position_error_reward_scale: float = -15.0

    # LQR reference controller
    lqr_q_ep: float = -position_error_reward_scale
    lqr_q_ev: float = -lin_vel_error_reward_scale
    lqr_r_u: float = 0.0

    # Vehicle Params
    angular_damping: float = 0.5
    drone_mass: float = 1.5 
    gravity: float = 9.81 

    # Termination criteria
    min_altitude: float = 0.1
    max_altitude: float = 2.0

    # Goal randomisation ranges
    goal_pos_xy_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    goal_pos_z_range: List[float] = field(default_factory=lambda: [0.5, 1.5])


class QuadcopterEnv(PegasusEnv):
    """Quadcopter hover and goal-reaching environment."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        """Initializes the task: caches config/backend/reset-manager and allocates the per-episode reward logging buffers."""
        super().__init__(cfg, backend, reset_manager)
        self._actions = None
        self._episode_sums = {}
        self._body_index = 0
        self._robot_weight = None


    def setup(self):
        """Initializes environment buffers and tracks variables after timeline starts."""
        super().setup()

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._episode_sums = {
            "ev": torch.zeros(self.num_envs, device=self.device),
            "ep": torch.zeros(self.num_envs, device=self.device),
            #"u": torch.zeros(self.num_envs, device=self.device),
            "total": torch.zeros(self.num_envs, device=self.device),
        }

        # Resolve body prim index
        vehicle = getattr(self.backend, "_vehicle", None)
        self._body_index = int(getattr(vehicle, "body_index", 0)) if vehicle else 0

        # Calculate robot weight
        self._robot_weight = float(self.cfg.drone_mass * self.cfg.gravity)
        
        self.backend.create_goal_markers(root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0))

        self.reset_manager.set_goal_cfg(self.cfg)  # Ensure reset manager has access to goal randomization config

        self.reset_manager._randomize_goals(torch.arange(self.num_envs, device=self.device))

    # -------------------------------------------
    # PegasusEnv Interface Implementations
    # -------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Clamps network output to valid action ranges."""
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        """Converts clamped actions to forces and torques and applies them."""
        state = self.backend.get_state()
        quat = state[:, 6:10] 
        ang_vel_b = state[:, 10:13]
        
        forces = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)
        torques = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)

        g_force = torch.tensor([0.0, 0.0, self._robot_weight], device=self.device).unsqueeze(0)

        forces_w = self.cfg.drone_mass * self._actions[:, 0:3] + g_force

        forces[:, self._body_index, :] = quaternion_apply(quaternion_invert(quat), forces_w)
        torques[:, self._body_index, :] = - self.cfg.angular_damping * ang_vel_b

        self.backend.set_forces_and_torques(forces, torques)

    def _get_observations(self) -> dict:
        """Builds a 12-dimensional observation vector for the policy."""
        state = self.backend.get_state()

        pos = state[:, 0:3]
        quat = state[:, 6:10]   
        lin_vel_b = state[:, 3:6]
        #ang_vel_b = state[:, 10:13]

        # Calculate the position error and velocity error in the inertial frame
        ep = self.reset_manager.goal_pos - pos
        ev = torch.zeros(self.num_envs, 3, device=self.device) - quaternion_apply(quat, lin_vel_b)

        obs = torch.cat([ep, ev], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Computes and logs the reward signal for the current timestep."""
        state = self.backend.get_state()

        pos = state[:, 0:3]
        quat = state[:, 6:10]   
        lin_vel_b = state[:, 3:6]
        #ang_vel_b = state[:, 10:13]

        ep = self.reset_manager.goal_pos - pos
        ev = -quaternion_apply(quat, lin_vel_b)


        rw_ep = torch.sum(torch.square(ep), dim=1) 
        rw_ev = torch.sum(torch.square(ev), dim=1)


        cost = -self.cfg.position_error_reward_scale * rw_ep - self.cfg.lin_vel_error_reward_scale * rw_ev

        terminated = torch.logical_or(state[:, 2] < self.cfg.min_altitude, state[:, 2] > self.cfg.max_altitude)

        reward_alive = - cost * self.step_dt + 1.0  # Base reward for being alive, minus the cost

        reward = torch.where(
            terminated,
            torch.full_like(reward_alive, -500.0),
            reward_alive
        )

        # Accumulate logs
        self._episode_sums["ep"] += rw_ep
        self._episode_sums["ev"] += rw_ev
        self._episode_sums["total"] += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Checks termination (out of bounds) and truncation (timeout) flags."""
        state = self.backend.get_state()

        terminated = torch.logical_or(state[:, 2] < self.cfg.min_altitude, state[:, 2] > self.cfg.max_altitude)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Resets selected environments and updates logging statistics."""
        if env_ids.numel() == 0:
            return

        state = self.backend.get_state()

        # Update and log episodic statistics
        final_distance_to_goal = torch.linalg.norm(self.reset_manager.goal_pos[env_ids] - state[env_ids, 0:3], dim=1).mean()
        
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras.setdefault("log", {})

        self.extras["log"].update(extras)

        self.extras["log"].update({
            "Episode_Termination/died": self.reset_terminated[env_ids].float().mean(),
            "Episode_Termination/time_out": self.reset_time_outs[env_ids].float().mean(),
            "Metrics/final_distance_to_goal": final_distance_to_goal,
        })

        # Physical reset
        self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=True)
        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)

        self.episode_length_buf[env_ids] = 0
        
        if env_ids.numel() == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        self._call_reset_callbacks(env_ids)

    # -------------------------------------------
    # Helpers
    # -------------------------------------------
    
    def _compute_discounted_dlqr(self, gamma: float = 0.99):

        """Computes the discounted discrete-time LQR gain K used as a reference controller for the linearized double-integrator model."""
        dt = self.step_dt

        I3 = torch.eye(3, dtype=torch.float64, device=self.device)

        A = torch.zeros((6, 6), dtype=torch.float64, device=self.device)
        B = torch.zeros((6, 3), dtype=torch.float64, device=self.device)

        A[0:3, 0:3] = I3
        A[0:3, 3:6] = dt * I3
        A[3:6, 3:6] = I3

        B[0:3, :] = 0.5 * dt * dt * I3
        B[3:6, :] = dt * I3

        Q = torch.diag(torch.tensor([
            self.cfg.lqr_q_ep,
            self.cfg.lqr_q_ep,
            self.cfg.lqr_q_ep,
            self.cfg.lqr_q_ev,
            self.cfg.lqr_q_ev,
            self.cfg.lqr_q_ev,
        ], dtype=torch.float64, device=self.device))

        R = self.cfg.lqr_r_u * I3

        sqrt_gamma = np.sqrt(gamma)

        Ag_np = sqrt_gamma * A.cpu().numpy()
        Bg_np = sqrt_gamma * B.cpu().numpy()
        Q_np = Q.cpu().numpy()
        R_np = R.cpu().numpy()

        P_np = solve_discrete_are(Ag_np, Bg_np, Q_np, R_np)

        P = torch.tensor(P_np, dtype=torch.float64, device=self.device)

        K = torch.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)

        return P.float(), K.float()