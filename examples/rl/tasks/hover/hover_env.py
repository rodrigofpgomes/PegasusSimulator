"""
HoverEnv — o drone aprende a manter uma posição fixa.

Observation space (13):
    pos   (3) — posição world frame
    vel   (3) — velocidade linear world frame
    quat  (4) — orientação quaternion wxyz
    omega (3) — velocidade angular body frame

Action space (4):
    thrust normalizado por motor [0, 1]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg


@dataclass
class HoverEnvCfg(PegasusEnvCfg):
    # ── espaços — imutáveis, definem o ambiente ───────────────
    observation_space: int = 13
    action_space:      int = 4
    state_space:       int = 0

    # ── tempo ─────────────────────────────────────────────────
    episode_length_s: float = 5.0
    decimation:       int   = 2     # 2 physics steps por policy step
    sim_dt:           float = 0.01

    # ── reward weights ────────────────────────────────────────
    distance_to_goal_reward_scale: float = 15.0
    lin_vel_reward_scale:          float = -0.05
    ang_vel_reward_scale:          float = -0.01
    effort_reward_scale:           float = -0.005
    crash_penalty:                 float = -10.0
    success_bonus:                 float = 5.0
    success_radius:                float = 0.1    # metros

    # ── terminação ────────────────────────────────────────────
    min_height:    float = 0.1
    max_height:    float = 5.0
    max_tilt_deg:  float = 60.0
    max_dist:      float = 10.0   # distância máxima ao target

    # ── task-specific ─────────────────────────────────────────
    target_pos:       List[float] = field(default_factory=lambda: [0.0, 0.0, 2.0])
    rand_init_radius: float       = 0.5   # raio de spawn aleatório (metros)
    rand_target:      bool        = False  # novo target em cada episódio


class HoverEnv(PegasusEnv):
    cfg: HoverEnvCfg

    def __init__(self, cfg: HoverEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)

        self._actions = torch.zeros((self.num_envs, cfg.action_space), dtype=torch.float32, device=self.device)

        self._target_pos = torch.tensor(cfg.target_pos, device=self.device).unsqueeze(0).expand(self.num_envs, 3).clone()

        # logging por episódio
        self._episode_sums = {
            "distance_to_goal": torch.zeros(self.num_envs, device=self.device),
            "lin_vel":          torch.zeros(self.num_envs, device=self.device),
            "ang_vel":          torch.zeros(self.num_envs, device=self.device),
        }

    # ── interface PegasusEnv ──────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(0.0, 1.0)

    def _apply_action(self):
        forces = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        torques = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)

        forces[:, 0, 2] = self._actions[:, 0]
        torques[:, 0, :] = self._actions[:, 1:3] 

        self.backend.set_forces_and_torques(forces, torques)

    def _get_observations(self) -> dict:
        state = self.backend.get_state()   # [N, 13]
        return {"policy": state}

    def _get_rewards(self) -> torch.Tensor:
        s     = self.backend.get_state()
        pos   = s[:, 0:3]
        vel   = s[:, 3:6]
        omega = s[:, 10:13]

        dist    = torch.linalg.norm(self._target_pos - pos, dim=1)
        r_dist  = (1 - torch.tanh(dist / 0.8)) * self.cfg.distance_to_goal_reward_scale
        r_vel   = torch.sum(vel   ** 2, dim=1) * self.cfg.lin_vel_reward_scale
        r_omega = torch.sum(omega ** 2, dim=1) * self.cfg.ang_vel_reward_scale
        r_effort = (
            torch.sum((self._actions - 0.5) ** 2, dim=1) * self.cfg.effort_reward_scale
        )
        r_success = (
            (dist < self.cfg.success_radius).float() * self.cfg.success_bonus
        )

        rewards = {
            "distance_to_goal": r_dist,
            "lin_vel":          r_vel,
            "ang_vel":          r_omega,
        }
        reward = r_dist + r_vel + r_omega + r_effort + r_success

        for key, val in rewards.items():
            self._episode_sums[key] += val * self.step_dt

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        s       = self.backend.get_state()
        pos     = s[:, 0:3]
        quat    = s[:, 6:10]

        crashed = (pos[:, 2] < self.cfg.min_height) | (pos[:, 2] > self.cfg.max_height)
        too_far = torch.linalg.norm(self._target_pos - pos, dim=1) > self.cfg.max_dist
        tilted  = self._tilt_angle(quat) > self.cfg.max_tilt_deg
        terminated = crashed | too_far | tilted

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        # logging
        for key in self._episode_sums:
            mean = self._episode_sums[key][env_ids].mean()
            self.extras[f"Episode_Reward/{key}"] = mean / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["Episode_Termination/count"] = env_ids.numel()

        # reset físico
        self.reset_manager.reset_envs(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0

        # novo target aleatório se configurado
        if self.cfg.rand_target:
            n = env_ids.numel()
            self._target_pos[env_ids, :2] = (
                torch.zeros(n, 2, device=self.device).uniform_(-2.0, 2.0)
            )
            self._target_pos[env_ids, 2] = (
                torch.zeros(n, device=self.device).uniform_(0.5, 2.0)
            )

        # notifica callbacks (ex: LSTM reset_hidden)
        self._call_reset_callbacks(env_ids)

    # ── utilitários ───────────────────────────────────────────

    def _tilt_angle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Ângulo de inclinação em graus a partir do quaternion wxyz.
        Compara o eixo z do body com o eixo z do world.
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # componente z do vetor (0,0,1) rodado pelo quaternion
        z_world = 1 - 2 * (x ** 2 + y ** 2)
        z_world = z_world.clamp(-1.0, 1.0)
        return torch.acos(z_world) * 180.0 / torch.pi
