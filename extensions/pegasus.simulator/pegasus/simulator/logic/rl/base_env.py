"""
| File: base_env.py
| Description: Base class for all RL environments in Pegasus (Isaac Lab style).
| License: BSD-3-Clause.

Terminated vs Truncated — critical distinction for correct GAE:

    terminated: agent reached a terminal state (crash, out of bounds).
                No bootstrap — V(s_terminal) = 0.
                rsl_rl: dones=True, time_outs=False

    truncated:  episode ended by timeout (max_episode_length reached).
                Bootstrap — V(s_last) != 0, episode continues hypothetically.
                rsl_rl: dones=True, time_outs=True

Isaac Lab implementation (DirectRLEnv):
    - episode_length_buf is incremented BEFORE computing dones
    - truncated = episode_length_buf >= max_episode_length - 1
    - reset happens AFTER dones are computed and returned
    - infos["time_outs"] = truncated  ← rsl_rl uses this for GAE bootstrap

This file replicates that behaviour exactly.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

__all__ = ["PegasusEnv", "PegasusEnvCfg"]


@dataclass
class PegasusEnvCfg:
    observation_space: int   = 0
    action_space:      int   = 0
    state_space:       int   = 0    # asymmetric critic (optional)
    episode_length_s:  float = 10.0
    decimation:        int   = 2    # physics steps per policy step
    sim_dt:            float = 0.01


class PegasusEnv(ABC):
    """
    Base class for all RL environments in Pegasus.
    Replicates Isaac Lab's DirectRLEnv interface exactly.

    Key design points matching Isaac Lab:
    1. episode_length_buf is incremented BEFORE _get_dones()
    2. truncated = episode_length_buf >= max_episode_length - 1
    3. _reset_idx() is called AFTER obs/reward/dones are computed
    4. infos["time_outs"] = truncated for rsl_rl GAE bootstrap
    5. reset() returns obs from AFTER the physical reset (initial state)
    """

    def __init__(self, cfg: PegasusEnvCfg, backend, reset_manager):
        self.cfg           = cfg
        self.backend       = backend
        self.reset_manager = reset_manager

        self.num_envs    = backend.n_vehicles
        self.num_obs     = cfg.observation_space
        self.num_actions = cfg.action_space

        self.max_episode_length   = int(cfg.episode_length_s / (cfg.sim_dt * cfg.decimation))
        self.max_episode_length_s = cfg.episode_length_s
        self.step_dt              = cfg.sim_dt * cfg.decimation

        self.extras: dict  = {}
        self._reset_callbacks: list = []

        # Allocated in setup() — device not available before start()
        self.episode_length_buf = None

        # Expose termination flags for logging
        self.reset_terminated = None
        self.reset_time_outs  = None

    def setup(self):
        """
        Called by the train script after timeline.play() + world.step(),
        when device and parts_per_vehicle are available via backend.
        """
        d = self.device
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=d)
        self.reset_terminated   = torch.zeros(self.num_envs, dtype=torch.bool, device=d)
        self.reset_time_outs    = torch.zeros(self.num_envs, dtype=torch.bool, device=d)

    @property
    def device(self) -> str:
        return self.backend.device

    @property
    def parts_per_vehicle(self) -> int:
        return self.backend.parts_per_vehicle


    # ── public API ────────────────────────────────────────────

    def step(self, actions: torch.Tensor):
        """
        Advance by one policy step (decimation physics steps).

        Isaac Lab ordering (replicated exactly):
            1. _pre_physics_step(actions)
            2. decimation × [_apply_action(), world.step()]
            3. episode_length_buf += 1          ← BEFORE dones
            4. _get_observations()
            5. _get_rewards()
            6. _get_dones()                     ← uses updated buf
            7. populate extras["time_outs"]     ← before reset
            8. _reset_idx(reset_ids)            ← AFTER all outputs
        """
        self._pre_physics_step(actions)

        for _ in range(self.cfg.decimation):
            self._apply_action()
            self._world_step()

        self.episode_length_buf += 1

        reward = self._get_rewards()
        terminated, truncated = self._get_dones()

        # Store for logging and external access
        self.reset_terminated = terminated
        self.reset_time_outs = truncated

        self.extras["time_outs"] = truncated
        
        # Reset only after all outputs are computed
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)

        obs = self._get_observations()

        return obs, reward, terminated, truncated, self.extras


    def reset(self):
        """
        Reset all environments.
        Returns obs from the initial state (after physical reset).

        Isaac Lab spreads resets to avoid spikes:
            episode_length_buf = randint(0, max_episode_length)
        We replicate this with init_at_random_ep_len in rsl_rl runner.
        """

        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        obs, _ = self._get_observations(), self.extras
        return obs, self.extras


    def register_reset_callback(self, fn):
        self._reset_callbacks.append(fn)

    # ── internal ──────────────────────────────────────────────

    def _world_step(self):
        if hasattr(self, "_world"):
            self._world.step(render=False)

    def _call_reset_callbacks(self, env_ids: torch.Tensor):
        for fn in self._reset_callbacks:
            fn(env_ids)

    # ── mandatory task interface ──────────────────────────────

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor): ...

    @abstractmethod
    def _apply_action(self): ...

    @abstractmethod
    def _get_observations(self) -> dict: ...

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor: ...

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (terminated, truncated).

        terminated: True when a fatal condition occurs (crash, OOB).
                    rsl_rl will NOT bootstrap these — V(s) = 0.

        truncated:  True when episode_length_buf >= max_episode_length - 1.
                    rsl_rl WILL bootstrap these — V(s) != 0.
                    Use self.episode_length_buf for this check.
        """
        ...

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset selected environments.
        Called AFTER obs/reward/dones are computed and returned.
        Must reset episode_length_buf[env_ids] = 0.
        """
        ...