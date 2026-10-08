"""
| File: agents/ppo_cfg.py
| Description: PPO configuration aligned with the RAPTOR SAC pre-training
|              configuration where the algorithms have comparable parameters.
|
| Environment:
|     Observation space: 26
|     Action space:       4
|     Control frequency:  100 Hz
|     Episode length:     500 steps / 5 seconds
|
| SAC-aligned PPO:
|     Actor:       Dense(obs -> 64, ReLU)
|                  Dense(64 -> 64, ReLU)
|                  Dense(64 -> action mean)
|
|     Value:       Dense(obs -> 256, ReLU)
|                  Dense(256 -> 256, ReLU)
|                  Dense(256 -> 1)
|
|     Learning rate:       3e-4
|     Discount factor:     0.99
|     Batch size:          rollouts * num_envs
|     State preprocessing: disabled
|     Gradient clipping:   disabled
|     Scheduler:           disabled
|     Timeout bootstrap:   enabled
|
| Default vectorized setup:
|     num_envs:             32
|     rollouts:             128
|     samples/update:       128 * 32 = 4096
|     mini_batches:         8
|     mini-batch size:      512
|     learning epochs:      5
|     total transitions:    approximately 1,000,000
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.models.torch import (
    DeterministicMixin,
    GaussianMixin,
    Model,
)

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR


# ---------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------

DEFAULT_NUM_ENVS = 1024

ROLLOUTS = 64
LEARNING_EPOCHS = 5
MINI_BATCHES = 16

PPO_UPDATES = 1500

# ---------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------

class Policy(GaussianMixin, Model):
    """PPO Gaussian actor aligned with the SAC actor architecture.

    The SAC policy produces both a state-dependent mean and log standard
    deviation. PPO generally behaves more robustly with a global learned
    log standard deviation, so only the mean is state-dependent here.

    The output actions are clipped to the action-space bounds by skrl.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        reduction: str = "sum",
    ):
        Model.__init__(
            self,
            observation_space,
            action_space,
            device,
        )

        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        # Matches the SAC actor hidden architecture:
        # obs -> 64 -> 64 -> action mean
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, self.num_actions),
        )

        # exp(-0.5) ~= 0.607 initial action standard deviation.
        #
        # This is less aggressive than log_std=0 (std=1), which produces
        # many saturated motor commands at the start of training.
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), fill_value=-0.5, device=device))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use orthogonal initialization commonly used with PPO."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(
                    layer.weight,
                    gain=math.sqrt(2.0),
                )
                nn.init.zeros_(layer.bias)

        # Smaller initialization on the policy output layer prevents
        # extreme initial motor commands.
        output_layer = self.net[-1]

        nn.init.orthogonal_(
            output_layer.weight,
            gain=0.01,
        )
        nn.init.zeros_(output_layer.bias)

    def compute(self, inputs, role):
        mean_actions = self.net(inputs["states"])

        return (
            mean_actions,
            self.log_std_parameter,
            {},
        )

# ---------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------

class Value(DeterministicMixin, Model):
    """PPO value function aligned with the SAC critic hidden sizes."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ):
        Model.__init__(
            self,
            observation_space,
            action_space,
            device,
        )

        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
        )

        # SAC critics use two hidden layers with 256 units.
        # PPO V(s) receives only the observation, without an action.
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(
                    layer.weight,
                    gain=math.sqrt(2.0),
                )
                nn.init.zeros_(layer.bias)

        output_layer = self.net[-1]

        nn.init.orthogonal_(
            output_layer.weight,
            gain=1.0,
        )
        nn.init.zeros_(output_layer.bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------

def _make_models(obs_space, act_space, device):
    return {
        "policy": Policy(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            clip_actions=True,
        ),
        "value": Value(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
        ),
    }

# ---------------------------------------------------------------------
# PPO configuration
# ---------------------------------------------------------------------

def _make_cfg() -> dict:
    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)

    # -------------------------------------------------------------
    # Rollout and optimization
    # -------------------------------------------------------------

    # At 100 Hz, 128 steps correspond to 1.28 seconds.
    #
    # With 32 envs:
    #     samples/update = 128 * 32 = 4096
    cfg["rollouts"] = ROLLOUTS

    # Each collected sample is reused for 5 optimization epochs.
    cfg["learning_epochs"] = LEARNING_EPOCHS

    # With 32 envs:
    #     mini-batch size = 4096 / 8 = 512
    cfg["mini_batches"] = MINI_BATCHES

    # -------------------------------------------------------------
    # Returns and advantages
    # -------------------------------------------------------------

    # Matches SAC.
    cfg["discount_factor"] = 0.99

    # PPO-specific GAE parameter.
    cfg["lambda"] = 0.95

    # Bootstraps from the value of the final observation when an episode
    # finishes because of the 5-second time limit.
    cfg["time_limit_bootstrap"] = False

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    # SAC uses 3e-4 for both actor and critic.
    # skrl PPO uses a shared optimizer learning rate.
    cfg["learning_rate"] = 5e-4

    # Matches the SAC configuration: fixed learning rate.
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}

    # Matches SAC, where gradient clipping is disabled.
    cfg["grad_norm_clip"] = 1.0

    # -------------------------------------------------------------
    # PPO clipping
    # -------------------------------------------------------------

    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True

    # Stop an optimization epoch if the policy moves too far.
    #
    # Unlike KLAdaptiveLR, this does not change the learning rate.
    cfg["kl_threshold"] = 0.0

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

    cfg["value_loss_scale"] = 1.0

    # SAC explicitly maximizes entropy. PPO has no learned temperature,
    # so a small fixed entropy bonus is used.
    cfg["entropy_loss_scale"] = 0.0

    # -------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------

    # SAC trains using raw observations, so PPO does the same here.
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": None}

    # Do not normalize value targets, to remain closer to the SAC setup.
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1}

    # -------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------

    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    # No reward scaling.
    cfg["rewards_shaper_scale"] = 0.01

    cfg["rewards_shaper"] = _reward_shaper

    # -------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------

    cfg["experiment"] = {
        "directory": "",
        "experiment_name": "",
        "write_interval": 24,
        "checkpoint_interval": 400,
    }

    return cfg

def _reward_shaper(rewards, timestep, timesteps):
    return rewards * 0.01

# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------

PRESETS = {
    # Use with:
    #
    #     --n_envs 32
    #
    # 245 PPO updates:
    #     trainer timesteps = 245 * 128 = 31,360
    #     transitions       = 31,360 * 32 = 1,003,520
    "isaac_lab": {
        "models": _make_models,
        "cfg": _make_cfg(),
        "timesteps": ROLLOUTS * PPO_UPDATES,  # rollouts * iterations
        "seed": None,
    },
}