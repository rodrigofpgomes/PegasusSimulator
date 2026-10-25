"""
| File: quadcopter_env.py
| Description: Quadcopter goal-reaching task - ep, ev → u (acceleration control, ideal inner loop).
| License: BSD-3-Clause.

One task for every reward of the outer-loop (acceleration level) study: the reward is
selected with reward_shape, alive_bonus and death_penalty (see README.md). It replaces the former copies
02_quadratic, 03_exponential and 04_constant_death, which differed only in the defaults of those three fields.

Observation space (6):
    ep (3) - position error in inertial frame (goal_pos - current_pos)
    ev (3) - velocity error in inertial frame (goal_vel - current_vel)

Action space (3):
    action[0:3] → desired inertial acceleration u = max_acceleration * clamp(action, -1, 1) [m/s^2],
    applied as the force m*u + m*g (no attitude dynamics: ideal inner loop).

Termination:
    terminated: z < 0.1 OR z > 2.0
    truncated:  episode_length_buf >= max_episode_length - 1

Reward (x = [ep, ev] after the step, u the acceleration applied during the step):
    c = (q_pos |ep|^2 + q_vel |ev|^2 + r_u |u|^2) dt
    r = -c  (quadratic)  or  exp(-c)  (exponential)
        + alive_bonus + death_penalty * terminated
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

    vehicle: str = "Shuttle"
    action_mode: str = "direct_force"

    # Spaces
    observation_space: int = 6
    action_space: int = 3
    state_space: int = 0

    # Timing
    episode_length_s: float = 10.0
    decimation: int = 2
    sim_dt: float = 0.01

    # Action authority per axis [m/s^2]
    max_acceleration: float = 1.0

    # Quadratic cost weights; also Q = diag(q_pos, q_vel) and R = r_u of the LQR reference
    q_pos: float = 15.0
    q_vel: float = 0.5
    r_u: float = 1.0

    # Reward shape: "quadratic" (r = -c) or "exponential" (r = exp(-c))
    reward_shape: str = "quadratic"
    # Constant added at every step
    alive_bonus: float = 1.0
    # Added on the terminating step
    death_penalty: float = -500.0

    # Vehicle Params
    angular_damping: float = 0.5
    drone_mass: float = 3.5
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
        super().__init__(cfg, backend, reset_manager)
        self._actions = None
        self._episode_sums = {}
        self._body_index = 0
        self._robot_weight = None


    def setup(self):
        """Initializes environment buffers and tracks variables after timeline starts."""
        super().setup()

        if self.cfg.reward_shape not in ("quadratic", "exponential"):
            raise ValueError(f"reward_shape must be 'quadratic' or 'exponential', got {self.cfg.reward_shape!r}")

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in ("ep", "ev", "u", "alive", "death", "total")
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
        """Converts the commanded acceleration to forces and torques and applies them."""
        state = self.backend.get_state()
        quat = state[:, 6:10]
        ang_vel_b = state[:, 10:13]

        forces = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)
        torques = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)

        g_force = torch.tensor([0.0, 0.0, self._robot_weight], device=self.device).unsqueeze(0)

        forces_w = self.cfg.drone_mass * self._acceleration() + g_force

        forces[:, self._body_index, :] = quaternion_apply(quaternion_invert(quat), forces_w)
        torques[:, self._body_index, :] = - self.cfg.angular_damping * ang_vel_b

        self.backend.set_forces_and_torques(forces, torques)

    def _get_observations(self) -> dict:
        """Builds the 6-dimensional observation [ep, ev] in the inertial frame."""
        ep, ev = self._tracking_errors()
        return {"policy": torch.cat([ep, ev], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        """Computes and logs the reward signal for the current timestep (state after the step)."""
        ep, ev = self._tracking_errors()
        dt = self.step_dt

        costs = {
            "ep": self.cfg.q_pos * torch.sum(torch.square(ep), dim=1) * dt,
            "ev": self.cfg.q_vel * torch.sum(torch.square(ev), dim=1) * dt,
            "u": self.cfg.r_u * torch.sum(torch.square(self._acceleration()), dim=1) * dt,
        }
        cost = torch.sum(torch.stack(list(costs.values())), dim=0)

        base = torch.exp(-cost) if self.cfg.reward_shape == "exponential" else -cost
        alive = torch.full_like(cost, self.cfg.alive_bonus)
        death = self.cfg.death_penalty * self._out_of_bounds().float()

        reward = base + alive + death

        # Accumulate logs (cost terms as negative contributions, whatever the reward shape)
        for key, value in costs.items():
            self._episode_sums[key] -= value
        self._episode_sums["alive"] += alive
        self._episode_sums["death"] += death
        self._episode_sums["total"] += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Checks termination (out of bounds) and truncation (timeout) flags."""
        terminated = self._out_of_bounds()
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
            "Metrics/goal_vel_abs_max": self.reset_manager.goal_vel.abs().max(),
        })

        # Physical reset
        self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=True)
        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)

        self.episode_length_buf[env_ids] = 0

        self._actions[env_ids] = 0.0

        self._call_reset_callbacks(env_ids)

    # -------------------------------------------
    # Helpers
    # -------------------------------------------

    def _acceleration(self) -> torch.Tensor:
        """Commanded inertial acceleration [m/s^2] from the clamped action."""
        return self.cfg.max_acceleration * self._actions

    def _tracking_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Position and velocity errors in the inertial frame (state velocity is inertial)."""
        state = self.backend.get_state()
        ep = self.reset_manager.goal_pos - state[:, 0:3]
        ev = self.reset_manager.goal_vel - state[:, 3:6]
        return ep, ev

    def _out_of_bounds(self) -> torch.Tensor:
        """Altitude outside [min_altitude, max_altitude]."""
        z = self.backend.get_state()[:, 2]
        return torch.logical_or(z < self.cfg.min_altitude, z > self.cfg.max_altitude)

    def _compute_discounted_dlqr(self, gamma: float = 0.99):
        """Discounted DLQR matching the quadratic reward, u = K [ep; ev] (unsaturated).

        The reward weights the state after the step, sum_t gamma^t (x_{t+1}' Q x_{t+1} + u_t' R u_t) dt,
        which equals the standard cost with Q/gamma minus the constant x_0' (Q/gamma) x_0 dt. The weights
        are read here, not copied into the config, so --env-cfg overrides also change the LQR.

        Returns:
            P: value matrix, the optimal quadratic return from x_0 is -x_0' P x_0 (x = [ep, ev]).
            K: feedback gain (3 x 6).
        """
        dt = self.step_dt

        I3 = np.eye(3)
        A = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        B = np.vstack([0.5 * dt * dt * I3, dt * I3])

        Q = np.diag([self.cfg.q_pos] * 3 + [self.cfg.q_vel] * 3) * dt / gamma
        R = self.cfg.r_u * I3 * dt

        sqrt_gamma = np.sqrt(gamma)
        P = solve_discrete_are(sqrt_gamma * A, sqrt_gamma * B, Q, R)
        K = np.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)

        # ep = -(pos - goal): the gain acting on x = [ep, ev] has the opposite sign of the textbook -K x
        to_torch = lambda M: torch.tensor(M, dtype=torch.float32, device=self.device)
        return to_torch(P - Q), to_torch(K)
