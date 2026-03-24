"""
| File: quadcopter_env.py
| Description: Quadcopter hover task — exact Isaac Lab replica.
| License: BSD-3-Clause.

Observation space (13) — matches Isaac Lab QuadcopterEnv exactly:
    rel_pos   (3) — position of robot relative to goal, in world frame
    quat      (4) — attitude quaternion wxyz
    vel       (3) — linear velocity, body frame
    omega     (3) — angular velocity, body frame

Action space (4) — matches Isaac Lab:
    action[0] → collective thrust, scaled from [-1,1] to [0, T_max]
    action[1:] → body-frame torques, scaled by moment_scale

Termination (matches Isaac Lab QuadcopterEnv._get_dones):
    terminated: z < 0.1 OR z > 2.0
    truncated:  episode_length_buf >= max_episode_length - 1

Reward (matches Isaac Lab QuadcopterEnv._get_rewards):
    r = lin_vel_scale * ||v||^2 * dt
      + ang_vel_scale * ||omega||^2 * dt
      + dist_scale * (1 - tanh(d/0.8)) * dt
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg

from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert


@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):
    
    # spaces
    observation_space: int = 12
    action_space: int = 4
    state_space: int = 0

    # timing
    episode_length_s: float = 10.0
    decimation: int   = 2      # 100 Hz physics / 2 = 50 Hz policy
    sim_dt: float = 0.01

    # reward scales
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0

    # vehicle params
    thrust_to_weight: float = 1.9 #2.81
    moment_scale:     float = 0.06 #0.1
    drone_mass:       float = 1.5   # kg
    gravity:          float = 9.81    # m/s^2

    # termination
    min_altitude: float = 0.1   # z < 0.1 → terminated
    max_altitude: float = 2.0   # z > 2.0 → terminated

    # ── goal randomisation — exact Isaac Lab values ────────────
    goal_pos_xy_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    goal_pos_z_range:  List[float] = field(default_factory=lambda: [0.5, 1.5])



class QuadcopterEnv(PegasusEnv):
    """
    Quadcopter hover environment that replicates the "quadcopter" environment considered in Isaac Lab.

    This environment is designed for hover and goal-reaching tasks.
    """
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)
        # buffers allocated in setup()
        self._actions  = None
        self._goal_pos = None
        self._episode_sums = {}


    def setup(self):
        """
        Method that initializes environment buffers and episode tracking variables.
        It assumes that the simulation timeline is active and the world has been stepped at least once.
        """

        super().setup()

        self._actions  = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_sums = {
            "lin_vel":          torch.zeros(self.num_envs, device=self.device),
            "ang_vel":          torch.zeros(self.num_envs, device=self.device),
            "distance_to_goal": torch.zeros(self.num_envs, device=self.device),
        }
        
        self.backend.create_goal_markers(root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0))

        self._randomize_goals(torch.arange(self.num_envs, device=self.device))



    #############################################
    #           PegasusEnv interface
    #############################################

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Clamp actions between [-1, 1].
        """

        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        """
        Convert clamped actions to forces/torques.
        thrust = thrust_to_weight * robot_weight * (action[0]+1)/2
        moment = moment_scale * action[1:]
        """

        forces  = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)
        torques = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)

        robot_weight = self.cfg.drone_mass * self.cfg.gravity
        thrust = self.cfg.thrust_to_weight * robot_weight * (self._actions[:, 0] + 1.0) / 2.0

        forces[:, 0, 2]  = thrust
        torques[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        self.backend.set_forces_and_torques(forces, torques)


    def _get_observations(self) -> dict:
        """
        Method that builds a 12-dimensional observation vector for the policy.

        The observation is composed of:
            lin_vel_b (3) — linear velocity in the body frame
            ang_vel_b (3) — angular velocity in the body frame
            projected_gravity_b (3) — gravity vector projected into the body frame
            desired_pos_b (3) — goal position relative to the robot, expressed in the body frame

        Returns:
            dict: A dictionary containing the observation tensor under the key "policy".
        """

        state = self.backend.get_state()     # [N, 13]

        pos = state[:, 0:3]
        quat = state[:, 6:10]   
        lin_vel_b = state[:, 3:6]
        ang_vel_b = state[:, 10:13]

        # goal relative to robot position (world frame)
        desired_pos_w = self._goal_pos - pos
        desired_pos_b = quaternion_apply(quaternion_invert(quat), desired_pos_w)

        g_w = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=quat.dtype)
        g_w = g_w.unsqueeze(0).repeat(quat.shape[0], 1)

        projected_gravity_b = quaternion_apply(quaternion_invert(quat), g_w)

        obs = torch.cat([lin_vel_b, ang_vel_b, projected_gravity_b, desired_pos_b], dim=-1)  # [N, 12]

        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        """
        This method computes the reward signal for the current timestep.

        The reward is composed of multiple terms:
            lin_vel (1) — squared linear velocity penalty
            ang_vel (1) — squared angular velocity penalty
            distance_to_goal (1) — shaped reward based on distance to the goal

        Returns:
            torch.Tensor: The total reward for each environment instance.
        """
        state = self.backend.get_state()

        lin_vel = torch.sum(torch.square(state[:, 3:6]), dim=1)
        ang_vel = torch.sum(torch.square(state[:, 10:13]), dim=1)
        distance_to_goal = torch.linalg.norm(self._goal_pos - state[:, 0:3], dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method that determines whether each environment instance has terminated
        or been truncated at the current timestep.

        Termination occurs when the robot's height (z position in world frame)
        goes outside the valid range (z < 0.1 or z > 2.0)

        Truncation occurs when the maximum episode length is reached:
            truncated — episode_length_buf >= max_episode_length - 1

        The episode length buffer is assumed to have been incremented before
        this method is called by the step function.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - terminated: Boolean tensor indicating terminal states
                - truncated: Boolean tensor indicating time-limit truncation
        """

        state = self.backend.get_state()     # [N, 13]

        terminated = torch.logical_or(state[:, 2] < 0.1, state[:, 2] > 2.0)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated


    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Method that resets a subset of environment instances and updates
        episode-level logging statistics.

        It then restores the initial robot state, including position and
        orientation, resets the episode length buffer, clears previous actions,
        and samples new goal positions.

        If all environments are reset, the episode length buffer is randomized
        to decorrelate environment rollouts.

        Finally, any registered reset callbacks are executed.

        Args:
            env_ids (torch.Tensor): Indices of environments to reset.
        """

        if env_ids.numel() == 0:
            return

        ep_log = {}

        state = self.backend.get_state()     # [N, 13]

        # Logging
        final_distance_to_goal = torch.linalg.norm(self._goal_pos[env_ids] - state[env_ids, 0:3], dim=1).mean()
        
        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Reset robot state
        self.reset_manager.reset_envs(env_ids)

        self.backend.set_state_for_envs(env_ids=env_ids, positions=self.reset_manager._init_pos[env_ids], attitudes=self.reset_manager._init_ori[env_ids])

        self.episode_length_buf[env_ids] = 0
        
        if env_ids.numel() == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        self._randomize_goals(env_ids)

        self._call_reset_callbacks(env_ids)


    # helper

    def _randomize_goals(self, env_ids: torch.Tensor):
        """
        Method that samples new goal positions for the specified environments.

        The goal position is randomized as follows:
            - XY coordinates are sampled uniformly within a configured range
            and offset by the environment's initial position
            - Z coordinate is sampled independently within a configured range

        This ensures that goals are distributed around each environment's
        starting position.

        Args:
            env_ids (torch.Tensor): Indices of environments to update.
        """

        xy_low, xy_high = self.cfg.goal_pos_xy_range
        z_low, z_high = self.cfg.goal_pos_z_range

        self._goal_pos[env_ids, :2] = torch.zeros_like(self._goal_pos[env_ids, :2]).uniform_(xy_low, xy_high)
        self._goal_pos[env_ids, :2] += self.reset_manager.init_pos[env_ids, :2]

        self._goal_pos[env_ids, 2] = torch.zeros_like(self._goal_pos[env_ids, 2]).uniform_(z_low, z_high)

        self.backend.update_goal_markers(self._goal_pos[env_ids], env_ids=env_ids)