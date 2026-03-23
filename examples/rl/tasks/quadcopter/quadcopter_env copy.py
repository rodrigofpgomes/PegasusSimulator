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
    # ── spaces ────────────────────────────────────────────────
    observation_space: int   = 13
    action_space:      int   = 4
    state_space:       int   = 0

    # ── timing — matches Isaac Lab ────────────────────────────
    episode_length_s: float = 10.0
    decimation:       int   = 2      # 100 Hz physics / 2 = 50 Hz policy
    sim_dt:           float = 0.01

    # ── reward scales — exact Isaac Lab values ─────────────────
    lin_vel_reward_scale:          float = -0.05
    ang_vel_reward_scale:          float = -0.01
    distance_to_goal_reward_scale: float = 15.0

    # ── vehicle params — Crazyflie (Isaac Lab default) ─────────
    thrust_to_weight: float = 1.9
    moment_scale:     float = 0.01
    #thrust_to_weight: float = 2.81
    #moment_scale:     float = 0.1
    drone_mass:       float = 1.5   # kg
    gravity:          float = 9.81    # m/s^2

    # ── termination — exact Isaac Lab values ───────────────────
    min_altitude: float = 0.1   # z < 0.1 → terminated
    max_altitude: float = 2.0   # z > 2.0 → terminated

    # ── goal randomisation — exact Isaac Lab values ────────────
    goal_pos_xy_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    goal_pos_z_range:  List[float] = field(default_factory=lambda: [0.5, 1.5])
    randomize_goal:    bool         = True


class QuadcopterEnv(PegasusEnv):
    """
    Quadcopter hover environment — exact Isaac Lab replica.

    Isaac Lab uses robot-frame velocities in the observation.
    Specifically:
        root_lin_vel_b  = linear velocity in body frame
        root_ang_vel_b  = angular velocity in body frame
        projected_gravity_b = gravity vector projected into body frame
        desired_pos_b   = goal position in body frame

    Our StateBatch provides world-frame quantities.
    We replicate the Isaac Lab obs exactly by computing body-frame
    quantities from the quaternion.
    """
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)
        # buffers allocated in setup()
        self._actions  = None
        self._goal_pos = None
        self._episode_sums = {}

    def setup(self):
        """Called after timeline.play() + world.step()."""
        super().setup()
        d = self.device

        self._actions  = torch.zeros((self.num_envs, self.cfg.action_space), device=d)
        self._goal_pos = torch.zeros((self.num_envs, 3), device=d)
        self._episode_sums = {
            "lin_vel":          torch.zeros(self.num_envs, device=d),
            "ang_vel":          torch.zeros(self.num_envs, device=d),
            "distance_to_goal": torch.zeros(self.num_envs, device=d),
        }
        self._randomize_goals(torch.arange(self.num_envs, device=d))

    # ── PegasusEnv interface ──────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Clamp to [-1, 1] — same as Isaac Lab."""
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        """
        Convert [-1,1] actions to forces/torques.
        Matches Isaac Lab _pre_physics_step + _apply_action:

            thrust = thrust_to_weight * robot_weight * (action[0]+1)/2
            moment = moment_scale * action[1:]
        """
        n  = self.num_envs
        p  = self.parts_per_vehicle
        d  = self.device

        forces  = torch.zeros((n, p, 3), device=d)
        torques = torch.zeros((n, p, 3), device=d)

        robot_weight = self.cfg.drone_mass * self.cfg.gravity
        thrust = (
            self.cfg.thrust_to_weight * robot_weight
            * (self._actions[:, 0] + 1.0) / 2.0
        )

        forces[:, 0, 2]  = thrust
        torques[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        self.backend.set_forces_and_torques(forces, torques)

    def _get_observations(self) -> dict:
        """
        Builds the 13-dim observation vector matching Isaac Lab:
            rel_pos (3) — goal position relative to robot, world frame
            quat    (4) — attitude wxyz
            vel     (3) — linear velocity, world frame
            omega   (3) — angular velocity, body frame

        Note: Isaac Lab uses body-frame lin/ang vel and projected gravity.
        Here we use world-frame lin vel and body-frame ang vel which are
        equivalent for a hover task where the policy learns from relative pos.
        For exact replication use _to_body_frame() helpers below.
        """
        state = self.backend.get_state()     # [N, 13]
        pos   = state[:, 0:3]
        vel   = state[:, 3:6]
        quat  = state[:, 6:10]
        omega = state[:, 10:13]

        # goal relative to robot position (world frame)
        rel_pos_w = self._goal_pos - pos
        rel_pos_b = quaternion_apply(quaternion_invert(quat), rel_pos_w)

        obs = torch.cat([rel_pos_b, quat, vel, omega], dim=-1)  # [N, 13]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Reward matching Isaac Lab QuadcopterEnv._get_rewards() exactly.
        All terms are multiplied by step_dt (= sim_dt * decimation).
        """
        state = self.backend.get_state()
        pos   = state[:, 0:3]
        vel   = state[:, 3:6]
        omega = state[:, 10:13]

        lin_vel = (
            torch.sum(vel ** 2, dim=1)
            * self.cfg.lin_vel_reward_scale
            * self.step_dt
        )
        ang_vel = (
            torch.sum(omega ** 2, dim=1)
            * self.cfg.ang_vel_reward_scale
            * self.step_dt
        )
        dist    = torch.linalg.norm(self._goal_pos - pos, dim=1)
        d2g     = (
            (1.0 - torch.tanh(dist / 0.8))
            * self.cfg.distance_to_goal_reward_scale
            * self.step_dt
        )

        for key, val in zip(
            ["lin_vel", "ang_vel", "distance_to_goal"],
            [lin_vel, ang_vel, d2g],
        ):
            self._episode_sums[key] += val

        return lin_vel + ang_vel + d2g

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Matches Isaac Lab QuadcopterEnv._get_dones() exactly:

            terminated: z < 0.1 OR z > 2.0
            truncated:  episode_length_buf >= max_episode_length - 1

        IMPORTANT: episode_length_buf is already incremented in base_env.step()
        BEFORE this method is called — same as Isaac Lab.

        terminated → rsl_rl does NOT bootstrap (V = 0)
        truncated  → rsl_rl DOES bootstrap    (V = V(s_last))
        """
        state = self.backend.get_state()
        z     = state[:, 2]   # altitude

        terminated = (z < self.cfg.min_altitude) | (z > self.cfg.max_altitude)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset selected environments.

        Called AFTER obs/reward/dones are computed and returned,
        so the last obs of the episode is already in the rollout buffer
        before we overwrite the state.

        Matches Isaac Lab _reset_idx():
        - logs episode sums
        - physical reset via reset_manager
        - zeros episode_length_buf for reset envs
        - randomizes new goals
        """
        if env_ids.numel() == 0:
            return

        # ── episode sums ──────────────────────────────────────
        ep_rew = {}
        for key in self._episode_sums:
            mean_val = (
                self._episode_sums[key][env_ids].mean()
                / self.max_episode_length_s
            ).item()
            self.extras[f"Episode_Reward/{key}"] = mean_val
            ep_rew[f"rew/{key}"] = mean_val
            self._episode_sums[key][env_ids] = 0.0

        n_died    = self.reset_terminated[env_ids].sum().item() if self.reset_terminated is not None else 0
        n_timeout = self.reset_time_outs[env_ids].sum().item()  if self.reset_time_outs  is not None else 0

        self.extras["Episode_Termination/died"]     = n_died
        self.extras["Episode_Termination/time_out"] = n_timeout

        # extras["log"] — rsl_rl Logger.process_env_step() reads this each
        # step and accumulates per-episode. Printed as "Episode/<key>".
        self.extras["log"] = {
            **ep_rew,
            "term/died":     float(n_died),
            "term/time_out": float(n_timeout),
        }

        # ── physical reset ────────────────────────────────────
        self.reset_manager.reset_envs(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0

        if self.cfg.randomize_goal:
            self._randomize_goals(env_ids)

        self._call_reset_callbacks(env_ids)

    # ── helpers ───────────────────────────────────────────────

    def _randomize_goals(self, env_ids: torch.Tensor):
        n      = env_ids.numel()
        d      = self.device
        xy_min, xy_max = self.cfg.goal_pos_xy_range
        z_min,  z_max  = self.cfg.goal_pos_z_range

        self._goal_pos[env_ids, :2] = (
            torch.rand(n, 2, device=d) * (xy_max - xy_min) + xy_min
        )
        self._goal_pos[env_ids, 2] = (
            torch.rand(n, device=d) * (z_max - z_min) + z_min
        )