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
    lin_vel_reward_scale: float = -0.5
    #ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0

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
        super().__init__(cfg, backend, reset_manager)
        self._actions = None
        self._goal_pos = None
        self._episode_sums = {}
        self._body_index = 0
        self._robot_weight = None

    def setup(self):
        """Initializes environment buffers and tracks variables after timeline starts."""
        super().setup()

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_sums = {
            "lin_vel": torch.zeros(self.num_envs, device=self.device),
            #"ang_vel": torch.zeros(self.num_envs, device=self.device),
            "distance_to_goal": torch.zeros(self.num_envs, device=self.device),
            "total": torch.zeros(self.num_envs, device=self.device),
        }

        # Resolve body prim index
        vehicle = getattr(self.backend, "_vehicle", None)
        self._body_index = int(getattr(vehicle, "body_index", 0)) if vehicle else 0

        # Calculate robot weight
        self._robot_weight = float(self.cfg.drone_mass * self.cfg.gravity)
        
        self.backend.create_goal_markers(root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0))
        self._randomize_goals(torch.arange(self.num_envs, device=self.device))

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
        ep = self._goal_pos - pos
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

        lin_vel = torch.sum(torch.square(quaternion_apply(quat, lin_vel_b)), dim=1)
        #ang_vel = torch.sum(torch.square(quaternion_apply(quat, ang_vel_b)), dim=1)
        
        ep = self._goal_pos - pos
        #ev = torch.zeros(self.num_envs, 3, device=self.device) - quaternion_apply(quat, lin_vel_b)

        distance_to_goal = torch.linalg.norm(ep, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            #"ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Accumulate logs
        for key, value in rewards.items():
            self._episode_sums[key] += value

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
        final_distance_to_goal = torch.linalg.norm(self._goal_pos[env_ids] - state[env_ids, 0:3], dim=1).mean()
        
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = extras
        self.extras["log"].update({
            "Episode_Termination/died": self.reset_terminated[env_ids].float().mean(),
            "Episode_Termination/time_out": self.reset_time_outs[env_ids].float().mean(),
            "Metrics/final_distance_to_goal": final_distance_to_goal,
        })

        # Physical reset
        self.reset_manager.reset_envs(env_ids)
        self.backend.set_state_for_envs(
            env_ids=env_ids, 
            positions=self.reset_manager._init_pos[env_ids], 
            attitudes=self.reset_manager._init_ori[env_ids]
        )

        self.episode_length_buf[env_ids] = 0
        if env_ids.numel() == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._randomize_goals(env_ids)
        self._call_reset_callbacks(env_ids)

    # -------------------------------------------
    # Helpers
    # -------------------------------------------

    def _randomize_goals(self, env_ids: torch.Tensor):
        """Samples new goal positions around the original spawn point."""
        xy_low, xy_high = self.cfg.goal_pos_xy_range
        z_low, z_high = self.cfg.goal_pos_z_range

        self._goal_pos[env_ids, :2] = torch.zeros_like(self._goal_pos[env_ids, :2]).uniform_(xy_low, xy_high)
        self._goal_pos[env_ids, :2] += self.reset_manager.init_pos[env_ids, :2]
        self._goal_pos[env_ids, 2] = torch.zeros_like(self._goal_pos[env_ids, 2]).uniform_(z_low, z_high)

        self.backend.update_goal_markers(self._goal_pos[env_ids], env_ids=env_ids)