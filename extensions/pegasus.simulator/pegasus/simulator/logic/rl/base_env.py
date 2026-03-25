"""
| File: base_env.py
| Description: Base class for RL environments in Pegasus.
|              Gymnasium-compatible (for skrl) AND preserves Isaac Lab step ordering.
| License: BSD-3-Clause.

Isaac Lab step ordering (preserved exactly):
    1. _pre_physics_step(actions)
    2. decimation × [_apply_action(), world.step()]
    3. episode_length_buf += 1          ← BEFORE dones
    4. _get_rewards()
    5. _get_dones()
    6. extras["time_outs"] = truncated  ← before reset
    7. _reset_idx(reset_ids)            ← AFTER all outputs computed
    8. _get_observations()              ← obs of NEW state (after reset)

Note on obs ordering vs Isaac Lab:
    Isaac Lab returns obs of the state BEFORE reset (last obs of episode).
    skrl's wrap_env expects the standard Gymnasium contract where obs is
    returned AFTER the environment has handled resets internally.
    This version follows the Gymnasium contract (obs after reset).
    The difference only affects the very last transition of each episode
    and is standard practice with vectorised Gymnasium environments.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

__all__ = ["PegasusEnv", "PegasusEnvCfg"]


@dataclass
class PegasusEnvCfg:
    observation_space: int   = 0
    action_space:      int   = 0
    state_space:       int   = 0
    episode_length_s:  float = 10.0
    decimation:        int   = 2
    sim_dt:            float = 0.01


class PegasusEnv(gym.Env, ABC):
    """
    Base vectorised RL environment for Pegasus.
    Implements the Gymnasium interface required by skrl's wrap_env(),
    while preserving Isaac Lab's internal step ordering.
    """

    def __init__(self, cfg: PegasusEnvCfg, backend, reset_manager):
        super().__init__()
        self.cfg           = cfg
        self.backend       = backend
        self.reset_manager = reset_manager

        self.num_envs    = backend.n_vehicles
        # num_obs / num_actions kept for rsl_rl wrapper compatibility
        self.num_obs     = cfg.observation_space
        self.num_actions = cfg.action_space

        self.max_episode_length   = int(cfg.episode_length_s / (cfg.sim_dt * cfg.decimation))
        self.max_episode_length_s = cfg.episode_length_s
        self.step_dt              = cfg.sim_dt * cfg.decimation

        # Gymnasium spaces — required by skrl wrap_env()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.observation_space,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(cfg.action_space,), dtype=np.float32,
        )

        self.extras: dict         = {}
        self._reset_callbacks: list = []

        # Allocated in setup() — device not available before start()
        self.episode_length_buf = None
        self.reset_terminated   = None
        self.reset_time_outs    = None

    def setup(self):
        """Called after timeline.play() + world.step() — device available here."""
        d = self.device
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long,  device=d)
        self.reset_terminated   = torch.zeros(self.num_envs, dtype=torch.bool,  device=d)
        self.reset_time_outs    = torch.zeros(self.num_envs, dtype=torch.bool,  device=d)

    @property
    def device(self) -> str:
        """Read device from backend at runtime — safe before and after start()."""
        return self.backend.device

    @property
    def parts_per_vehicle(self) -> int:
        return self.backend.parts_per_vehicle

    # ── Gymnasium API ─────────────────────────────────────────────────

    def step(self, actions: torch.Tensor):
        """
        Isaac Lab ordering — Gymnasium return format.
        Returns: obs, reward, terminated, truncated, info
        """
        self._pre_physics_step(actions)

        for _ in range(self.cfg.decimation):
            self._apply_action()
            self._world_step()

        # Increment BEFORE computing dones — matches Isaac Lab
        self.episode_length_buf += 1

        reward                = self._get_rewards()
        terminated, truncated = self._get_dones()

        self.reset_terminated = terminated
        self.reset_time_outs  = truncated
        self.extras["time_outs"] = truncated

        # Reset AFTER computing outputs
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)

        # Obs of the new state (after reset for done envs — Gymnasium contract)
        obs = self._get_observations()

        return obs, reward, terminated, truncated, self.extras

    def reset(self, *, seed=None, options=None):
        """
        Gymnasium-compatible reset.
        Returns: (obs, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        obs = self._get_observations()
        return obs, self.extras

    def register_reset_callback(self, fn):
        self._reset_callbacks.append(fn)

    # ── Internal ──────────────────────────────────────────────────────

    def _world_step(self):
        if hasattr(self, "_world"):
            self._world.step(render=False)

    def _call_reset_callbacks(self, env_ids: torch.Tensor):
        for fn in self._reset_callbacks:
            fn(env_ids)

    # ── Abstract task interface ───────────────────────────────────────

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor): ...

    @abstractmethod
    def _apply_action(self): ...

    @abstractmethod
    def _get_observations(self) -> torch.Tensor:
        """Return flat Tensor [N, obs_dim] — required by skrl wrap_env."""
        ...

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor: ...

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor): ...