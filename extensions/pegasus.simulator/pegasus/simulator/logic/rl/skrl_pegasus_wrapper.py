"""
| File: skrl_pegasus_wrapper.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Environment wrapper for skrl compatibility.
"""

from __future__ import annotations
from typing import Any
import torch

from skrl import config
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space

class PegasusSkrlWrapper:
    """Wraps Pegasus Environments to comply with skrl's structural expectations."""

    def __init__(self, env: Any) -> None:
        """Initializes the wrapper for the Pegasus environment."""
        self._env = env
        self._unwrapped = getattr(env, "unwrapped", env)

        self.num_envs = env.num_envs
        self.num_agents = 1
        self.device = env.device

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    # -------------------------------------------
    # Properties
    # -------------------------------------------

    @property
    def observation_space(self):
        """Returns the policy's observation space."""
        return self._unwrapped.observation_space

    @property
    def action_space(self):
        """Returns the environment's action space."""
        return self._unwrapped.action_space

    @property
    def state_space(self):
        """Returns the critic's state space, if available."""
        return getattr(self._unwrapped, "state_space", None)

    @property
    def unwrapped(self):
        """Returns the underlying base environment."""
        return self._unwrapped

    # -------------------------------------------
    # Core RL Interface
    # -------------------------------------------

    def step(self, actions: torch.Tensor):
        """Executes an action, advances the simulation, and formats the output."""
        actions = unflatten_tensorized_space(self.action_space, actions)

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))

        if "final_observation" in self._info: 
            self._info["final_observation"] = flatten_tensorized_space(tensorize_space(self.observation_space, self._info["final_observation"]))

        states = observations.get("critic", None)
        self._states = flatten_tensorized_space(tensorize_space(self.state_space, states)) if states is not None and self.state_space is not None else None

        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self):
        """Fetches the initial environment observation safely."""
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))

            states = observations.get("critic", None)
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states)) if states is not None and self.state_space is not None else None

            self._reset_once = False

        return self._observations, self._info

    def state(self):
        """Returns the current state (critic observation) of the environment."""
        return self._states

    # -------------------------------------------
    # Utilities
    # -------------------------------------------

    def render(self, *args, **kwargs):
        """Renders the environment if supported."""
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)
        return None

    def close(self):
        """Closes the environment and releases resources."""
        if hasattr(self._env, "close"):
            return self._env.close()

    def __getattr__(self, name):
        """Delegates missing attribute access to the underlying environment."""
        return getattr(self._env, name)