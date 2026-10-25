"""
| File: agents/ppo2_cfg.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: PPO2 configuration reproducing the Isaac Lab RSL-RL PPO setup.
"""

import copy
import os

import torch
import torch.nn as nn

from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG


# =============================================================================
# Distribution Configuration
# =============================================================================

# RSL-RL disables argument validation for the Normal distribution.
Normal.set_default_validate_args(False)


# =============================================================================
# Policy
# =============================================================================

class Policy(Model):
    """
    Gaussian actor reproducing the RSL-RL policy.

    RSL-RL configuration:
        init_noise_std = 1.0
        noise_std_type = "scalar"
        state_dependent_std = False
    """

    def __init__(self, observation_space, action_space, device, mean_tanh: bool = False):
        Model.__init__(self, observation_space, action_space, device)

        # tanh keeps the deterministic action inside [-1, 1], as the squashed SAC actor;
        # without it the mean can leave the range that the task clamps.
        self.mean_tanh = mean_tanh

        # Actor:
        #   obs -> 64 -> ELU -> 64 -> ELU -> actions
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )

        # On RSL-RL the standard deviation is represented directly by a learnable parameter sigma, rather than by a learnable log(sigma).
        self.std_parameter = nn.Parameter(torch.ones(self.num_actions, device=self.device))

        # Optional bounds on the exploration std, [PPO2_STD_MIN, 1]. Without
        # PPO2_STD_MIN the std is unbounded, as in RSL-RL; run_ppo2_variant.sh sets it (STD_MIN).
        std_min = os.environ.get("PPO2_STD_MIN")
        self.std_min = float(std_min) if std_min else None
        self.std_max = 1.0

        # Current Gaussian distribution.
        self._distribution = None

    # -------------------------------------------------------------------------
    # Model Interface
    # -------------------------------------------------------------------------

    def compute(self, inputs, role):
        """Computes the mean action predicted by the actor."""
        return self._mean(inputs["states"]), {}

    def _mean(self, states):
        """Return the Gaussian mean, bounded by tanh when enabled."""
        mean = self.net(states)
        return torch.tanh(mean) if self.mean_tanh else mean

    # -------------------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------------------

    def get_distribution_params(self, states):
        """
        Returns the Gaussian mean and standard deviation.

        This method is used by PPO2 before the optimization epochs to obtain
        the old policy distribution required by the analytical KL divergence.
        """

        mean = self._mean(states)

        # Project the parameter outside autograd; gradients still flow at the limits.
        if self.std_min is not None:
            with torch.no_grad():
                self.std_parameter.clamp_(min=self.std_min, max=self.std_max)

        # RSL-RL uses one learnable state-independent sigma per action.
        std = self.std_parameter.expand_as(mean)

        return mean, std

    def distribution(self, role="policy"):
        """Returns the current action distribution."""

        if self._distribution is None:
            raise RuntimeError(
                "Policy distribution has not been initialized. "
                "Call act(...) before distribution(...)."
            )

        return self._distribution

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    def act(self, inputs, role="policy"):
        """
        Samples actions from the Gaussian policy and computes their log-probability.

        During rollout: taken_actions is absent -> sample a new action and compute its log-probability.

        During PPO update: taken_actions contains the stored rollout actions -> update the current distribution and evaluate those actions.
        """

        mean, std = self.get_distribution_params(inputs["states"])

        # On RSL-RL the standard deviation is represented directly by a learnable parameter sigma, rather than by a learnable log(sigma).
        self._distribution = Normal(mean, std)

        # RSL-RL uses distribution.sample().
        actions = self._distribution.sample()

        # During PPO optimization evaluate the actions collected by the old
        # policy instead of the newly sampled actions.
        taken_actions = inputs.get("taken_actions", actions)

        log_prob = self._distribution.log_prob(taken_actions).sum(dim=-1, keepdim=True)

        return actions, log_prob, {"mean_actions": mean, "std_actions": std}

    # -------------------------------------------------------------------------
    # Entropy
    # -------------------------------------------------------------------------

    def get_entropy(self, role="policy"):
        """Returns the Gaussian policy entropy."""

        if self._distribution is None:
            raise RuntimeError(
                "Policy distribution has not been initialized. "
                "Call act(...) before get_entropy(...)."
            )

        return self._distribution.entropy().sum(dim=-1, keepdim=True)


# =============================================================================
# Value Function
# =============================================================================

class Value(DeterministicMixin, Model):
    """
    Architecture:
        obs -> 64 -> ELU -> 64 -> ELU -> 1
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)

        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        """Computes the state-value estimate."""
        return self.net(inputs["states"]), {}


# =============================================================================
# Models
# =============================================================================

def _default_models(obs_space, act_space, device):
    """
    Creates independent actor and critic networks.

    RSL-RL uses separate actor and critic MLPs for the Isaac Lab Quadcopter configuration.
    """

    return {
        "policy": Policy(obs_space, act_space, device),
        "value": Value(obs_space, act_space, device)
    }


def _tanh_models(obs_space, act_space, device):
    """Same networks as _default_models, with the policy mean bounded by tanh."""

    return {
        "policy": Policy(obs_space, act_space, device, mean_tanh=True),
        "value": Value(obs_space, act_space, device)
    }


# =============================================================================
# PPO2 Configuration
# =============================================================================

def _make_cfg() -> dict:
    """Creates the PPO2 configuration reproducing the RSL-RL setup."""

    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # num_steps_per_env = 24
    # -------------------------------------------------------------------------
    cfg["rollouts"] = 24

    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4

    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95

    # -------------------------------------------------------------------------
    # clip_param = 0.2
    # use_clipped_value_loss = True
    #
    # The same clip_param is used for both the policy ratio and the value function in RSL-RL.
    # -------------------------------------------------------------------------

    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2

    # Disable the native SKRL value clipping implementation because PPO2
    # implements the RSL-RL clipped value loss directly.
    cfg["clip_predicted_values"] = False

    cfg["use_clipped_value_loss"] = True

    # -------------------------------------------------------------------------
    # Loss
    #
    # RSL-RL:
    #   value_loss_coef = 1.0
    #   entropy_coef = 0.0
    # -------------------------------------------------------------------------

    cfg["value_loss_scale"] = 1.0
    cfg["entropy_loss_scale"] = 0.0

    # -------------------------------------------------------------------------
    # Optimizer
    #
    # RSL-RL:
    #   learning_rate = 5e-4
    #   max_grad_norm = 1.0
    # -------------------------------------------------------------------------

    cfg["learning_rate"] = 5.0e-4
    cfg["grad_norm_clip"] = 1.0

    # -------------------------------------------------------------------------
    # Adaptive Learning Rate
    #
    # RSL-RL:
    #   schedule = "adaptive"
    #   desired_kl = 0.01
    #
    # PPO2 performs the RSL-RL adaptive learning-rate update internally after
    # computing the analytical Gaussian KL divergence for each mini-batch.
    # -------------------------------------------------------------------------

    cfg["schedule"] = "adaptive"
    cfg["desired_kl"] = 0.01

    # Limits used directly by RSL-RL's adaptive schedule.
    cfg["min_learning_rate"] = 1.0e-5
    cfg["max_learning_rate"] = 1.0e-2
    cfg["learning_rate_factor"] = 1.5

    # Disable SKRL's scheduler because PPO2 handles this internally.
    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    # -------------------------------------------------------------------------
    # actor_obs_normalization = False
    # critic_obs_normalization = False
    # -------------------------------------------------------------------------
    cfg["state_preprocessor"] = None
    cfg["state_preprocessor_kwargs"] = {}

    # -------------------------------------------------------------------------
    # Does not use SKRL's RunningStandardScaler on value targets.
    # -------------------------------------------------------------------------
    cfg["value_preprocessor"] = None
    cfg["value_preprocessor_kwargs"] = {}

    # -------------------------------------------------------------------------
    # RSL-RL utilizes directly the rewards from the environment
    # -------------------------------------------------------------------------
    cfg["rewards_shaper"] = None

    # -------------------------------------------------------------------------
    # Interaction
    # -------------------------------------------------------------------------
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    # -------------------------------------------------------------------------
    # KL Early Stopping
    #
    # RSL-RL uses KL for the adaptive learning rate, not for the SKRL-style
    # early stopping mechanism.
    # -------------------------------------------------------------------------
    cfg["kl_threshold"] = 0.0

    # -------------------------------------------------------------------------
    # Time-limit Bootstrap
    #
    # RSL-RL adds:
    #   gamma * V(s) * time_out
    # to the reward for time-limit truncations.
    #
    # With PegasusSkrlWrapper this corresponds to truncated=True.
    # -------------------------------------------------------------------------

    cfg["time_limit_bootstrap"] = True

    cfg["mixed_precision"] = False

    # -------------------------------------------------------------------------
    # save_interval = 50 iterations
    #
    # One PPO iteration contains 24 environment steps:
    #   50 * 24 = 1200 trainer timesteps
    # -------------------------------------------------------------------------

    cfg["experiment"] = {
        "directory": "quadcopter_direct",
        "experiment_name": "",
        "write_interval": 24,
        # one PPO iteration is 24 timesteps; PPO2_CHECKPOINT_ITERS sets the save period in iterations
        "checkpoint_interval": 24 * int(os.environ.get("PPO2_CHECKPOINT_ITERS", "50")),
    }

    return cfg


# =============================================================================
# Presets
# =============================================================================

PRESETS = {
    # -------------------------------------------------------------------------
    # num_steps_per_env = 24, max_iterations = 400 (Isaac Lab uses 200)
    # timesteps = 24 * 400 = 9600
    # -------------------------------------------------------------------------

    "default": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 24 * 400,
        "seed": 1,
    },

    # -------------------------------------------------------------------------
    # Policy mean bounded by tanh (exploration noise is still clamped by the task),
    # 600 iterations: timesteps = 24 * 600 = 14400
    # -------------------------------------------------------------------------

    "tanh": {
        "models": _tanh_models,
        "cfg": _make_cfg(),
        "timesteps": 24 * 600,
        "seed": 1,
    },
}