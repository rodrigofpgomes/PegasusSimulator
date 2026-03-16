"""
| File: pegasus_env.py
| Author: Rodrigo Gomes
| Description: Base class for all reinforcement learning environments in Pegasus.
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


__all__ = ["PegasusEnv", "PegasusEnvCfg"]


@dataclass
class PegasusEnvCfg:
    """
    Configuration class for Pegasus reinforcement learning environments.

    This class stores the common configuration parameters shared by all RL
    environments, such as the dimensions of the observation and action spaces,
    episode duration, simulation timing, and execution device.
    """

    # Dimensions of the environment spaces. These are expected to be defined
    # by each specific task/environment implementation.
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0  # Optional asymmetric critic state dimension

    # Timing configuration
    episode_length_s: float = 5.0
    decimation: int = 2  # Number of physics steps executed per policy step
    sim_dt: float = 0.01  # Duration of one physics step [s]

    # Execution device
    device: str = "cuda"


class PegasusEnv(ABC):
    """
    Base class for all reinforcement learning environments in Pegasus.

    This class defines the common interface and execution flow for RL tasks.
    Each task should inherit from this class and implement the required
    abstract methods that define how actions are processed, applied to the
    simulator, and how observations, rewards, terminations, and resets are handled.

    Inspired by the DirectRLEnv abstraction from Isaac Lab.
    """

    def __init__(
        self,
        cfg: PegasusEnvCfg,
        backend,
        reset_manager,
    ):
        """
        Initialize the PegasusEnv object.

        Args:
            cfg (PegasusEnvCfg): Configuration object for the environment.
            backend: Backend responsible for managing the vehicles/actions.
            reset_manager: Utility object responsible for environment resets.
        """
        self.cfg = cfg
        self.backend = backend
        self.reset_manager = reset_manager
        self.device = cfg.device

        # Basic environment dimensions inferred from the backend and config
        self.num_envs = backend.n_vehicles
        self.parts_per_vehicle = backend.parts_per_vehicle
        self.num_obs = cfg.observation_space
        self.num_actions = cfg.action_space

        # Per-environment episode step counters with shape [num_envs]
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Maximum episode duration measured in policy steps
        self.max_episode_length = int(
            cfg.episode_length_s / (cfg.sim_dt * cfg.decimation)
        )
        self.max_episode_length_s = cfg.episode_length_s

        # Duration of a single policy step in seconds
        self.step_dt = cfg.sim_dt * cfg.decimation

        # Dictionary used to expose extra information for logging/debugging.
        # This is typically populated during resets.
        self.extras: dict = {}

        # External callbacks to be executed after a reset.
        # For example, this can be used by recurrent runners to reset hidden states.
        self._reset_callbacks: list = []

    """
    Public API
    """

    def step(self, actions: torch.Tensor):
        """
        Advance the environment by one policy step.

        Internally, this executes `decimation` physics steps for each policy step.

        Args:
            actions (torch.Tensor): Tensor containing the actions to be applied.

        Returns:
            tuple: A tuple with:
                - observations
                - rewards
                - terminated flags
                - truncated flags
                - extras dictionary
        """
        # Process the raw actions before stepping the physics simulation
        self._pre_physics_step(actions)

        # Execute multiple physics steps for each policy step
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self._world_step()

        # Gather environment outputs after the physics updates
        self.episode_length_buf += 1
        obs = self._get_observations()
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()

        # Automatically reset environments that have terminated or been truncated
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)

        return obs, reward, terminated, truncated, self.extras

    def reset(self):
        """
        Reset all environments.

        Returns:
            tuple: A tuple containing:
                - observations after reset
                - extras dictionary
        """
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        return self._get_observations(), self.extras

    def register_reset_callback(self, fn):
        """
        Register a callback to be invoked after each partial reset.

        This is useful for external runners that need to clear auxiliary state,
        such as the hidden state of an LSTM policy.

        Args:
            fn (callable): A function with signature:
                fn(env_ids: torch.Tensor) -> None
        """
        self._reset_callbacks.append(fn)

    """
    Internal helper methods
    """

    def _world_step(self):
        """
        Advance the simulator by one physics step.

        The simulator world object is expected to be injected externally
        (for example by the training script).
        """
        if hasattr(self, "_world"):
            self._world.step(render=False)

    def _call_reset_callbacks(self, env_ids: torch.Tensor):
        """
        Execute all registered reset callbacks for the given environments.

        Args:
            env_ids (torch.Tensor): Tensor with the ids of the environments
                that were reset.
        """
        for fn in self._reset_callbacks:
            fn(env_ids)

    """
    Mandatory task interface
    """

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process the actions before the physics step is executed.

        Args:
            actions (torch.Tensor): Tensor containing the policy actions.
        """
        pass

    @abstractmethod
    def _apply_action(self):
        """
        Apply the processed actions to the simulator/backend.
        """
        pass

    @abstractmethod
    def _get_observations(self) -> dict:
        """
        Construct and return the current observations.

        Returns:
            dict: Observation dictionary for the current environment state.
        """
        pass

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute the reward for each environment.

        Returns:
            torch.Tensor: Tensor containing the rewards for all environments.
        """
        pass

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the termination and truncation signals.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - terminated flags
                - truncated flags
        """
        pass

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset a subset of environments.

        Args:
            env_ids (torch.Tensor): Tensor with the ids of the environments
                that should be reset.
        """
        pass