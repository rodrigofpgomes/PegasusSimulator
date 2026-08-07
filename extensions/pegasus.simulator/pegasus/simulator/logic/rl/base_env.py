"""
| File: base_env.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Base class for all RL environments in Pegasus.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

__all__ = ["PegasusEnv", "PegasusEnvCfg"]


@dataclass
class PegasusEnvCfg:
    """Base configuration for Pegasus Environments."""
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0
    episode_length_s: float = 10.0
    decimation: int = 2
    sim_dt: float = 0.01
    action_mode: str = "direct_force"  # "direct_force" or "rotor_velocity"
    vehicle: str = "Iris"              # key into pegasus.simulator.params.ROBOTS


class PegasusEnv(ABC):
    """
    Base class enforcing the Gym/IsaacLab interface expected by skrl.
    """

    def __init__(self, cfg: PegasusEnvCfg, backend, reset_manager):
        """Builds the vectorized environment from its config, backend and reset manager.

        Derives episode bookkeeping (``max_episode_length``, ``step_dt``) and exposes the
        Gymnasium ``policy`` (and optional ``critic``) observation/action spaces.
        """
        self.cfg = cfg
        self.backend = backend
        self.reset_manager = reset_manager

        self.num_envs = backend.n_vehicles
        self.num_obs = cfg.observation_space
        self.num_actions = cfg.action_space
        self.num_states = cfg.state_space

        self.max_episode_length = int(cfg.episode_length_s / (cfg.sim_dt * cfg.decimation))
        self.max_episode_length_s = cfg.episode_length_s
        self.step_dt = cfg.sim_dt * cfg.decimation

        self.extras: dict = {}
        self._reset_callbacks: list = []
        self._render_enabled = False

        # Allocated in setup() once simulation is active
        self.episode_length_buf = None
        self.reset_terminated = None
        self.reset_time_outs = None

        # Expose Gymnasium spaces
        obs_spaces = {
            "policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.observation_space,), dtype=np.float32)
        }
        if cfg.state_space and cfg.state_space > 0:
            obs_spaces["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.state_space,), dtype=np.float32)

        single_observation_space = gym.spaces.Dict(obs_spaces)

        self.observation_space = single_observation_space["policy"]
        self.state_space = single_observation_space.spaces.get("critic", None)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.action_space,), dtype=np.float32)


    def setup(self):
        """Initializes device buffers after timeline starts."""
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def device(self) -> str:
        """Torch device of the underlying backend (e.g. ``"cuda"``)."""
        return self.backend.device

    @property
    def parts_per_vehicle(self) -> int:
        """Number of rigid-body parts that make up each vehicle."""
        return self.backend.parts_per_vehicle

    @property
    def unwrapped(self):
        """Returns the base environment (no wrappers applied here)."""
        return self


    # -------------------------------------------
    # Public API
    # -------------------------------------------

    def step(self, actions: torch.Tensor):
        """Advances the environment by one policy step."""
        self.extras = {}
        self._pre_physics_step(actions)

        # Physics sub-stepping
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self._world_step()

        self.episode_length_buf += 1

        reward = self._get_rewards()
        terminated, truncated = self._get_dones()

        self.reset_terminated = terminated
        self.reset_time_outs = truncated
        self.extras["time_outs"] = truncated.view(self.num_envs, 1)

        # Reset environments that have terminated or reached timeout
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.extras["final_observation"] = self._get_observations()["policy"].clone()
            self.extras["final_observation_mask"] = (terminated | truncated)
            self._reset_idx(reset_ids)

        obs = self._get_observations()
        return obs, reward, terminated, truncated, dict(self.extras)

    def reset(self):
        """Resets all environments and retrieves initial observations."""
        self.extras = {}
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        return self._get_observations(), dict(self.extras)

    def register_reset_callback(self, fn):
        """Registers a callback ``fn(env_ids)`` invoked whenever environments are reset."""
        self._reset_callbacks.append(fn)


    # -------------------------------------------
    # Internal Methods
    # -------------------------------------------

    def _world_step(self):
        """Advances the physics simulation by one step (with or without rendering)."""
        if hasattr(self, "_world"):
            self._world.step(render=self._render_enabled)

    def _call_reset_callbacks(self, env_ids: torch.Tensor):
        """Executes the registered callback functions for the reset environments."""
        for fn in self._reset_callbacks:
            fn(env_ids)


    # -------------------------------------------
    # Mandatory Task Interface
    # -------------------------------------------

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """Processes the actions (e.g., clamping) before applying them to the physics engine."""
        pass    

    @abstractmethod
    def _apply_action(self):
        """Sets the processed actions as forces and torques applied to the simulator."""
        pass

    @abstractmethod
    def _get_observations(self) -> dict:
        """Collects and returns the current observations (state) of the environment."""
        pass

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Calculates and returns the reward for the current step."""
        pass

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the episode termination flags: (terminated, truncated)."""
        pass

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        """Resets the physical state and metrics for the specified environments."""
        pass