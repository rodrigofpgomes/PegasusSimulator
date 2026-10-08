#!/usr/bin/env python
"""
| File: ppo2_cfg.py
| Description: PPO2 configuration and teacher-pretraining configuration for the
|              quadcopter ep, ev -> inertial acceleration task.
"""

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG


Normal.set_default_validate_args(False)


# =============================================================================
# Policy
# =============================================================================

class Policy(Model):
    """Gaussian actor: obs(6) -> 64 -> ELU -> 64 -> ELU -> action(3)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )

        # Same state-independent sigma used by PPO2/RSL-RL.
        # Teacher pretraining only optimizes self.net.
        self.std_parameter = nn.Parameter(torch.ones(self.num_actions, device=self.device))
        self._distribution = None

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

    def get_distribution_params(self, states):
        mean = self.net(states)
        std = self.std_parameter.expand_as(mean)
        return mean, std

    def distribution(self, role="policy"):
        if self._distribution is None:
            raise RuntimeError("Policy distribution has not been initialized. Call act(...) first.")
        return self._distribution

    def act(self, inputs, role="policy"):
        mean, std = self.get_distribution_params(inputs["states"])
        self._distribution = Normal(mean, std)

        actions = self._distribution.sample()
        taken_actions = inputs.get("taken_actions", actions)
        log_prob = self._distribution.log_prob(taken_actions).sum(dim=-1, keepdim=True)

        return actions, log_prob, {"mean_actions": mean, "std_actions": std}

    def get_entropy(self, role="policy"):
        if self._distribution is None:
            raise RuntimeError("Policy distribution has not been initialized. Call act(...) first.")
        return self._distribution.entropy().sum(dim=-1, keepdim=True)


# =============================================================================
# Value function
# =============================================================================

class Value(DeterministicMixin, Model):
    """Critic: obs(6) -> 64 -> ELU -> 64 -> ELU -> value(1)."""

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
        return self.net(inputs["states"]), {}


def _default_models(obs_space, act_space, device):
    return {
        "policy": Policy(obs_space, act_space, device),
        "value": Value(obs_space, act_space, device),
    }


# =============================================================================
# PPO2 configuration
# =============================================================================

def _make_cfg() -> dict:
    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)

    cfg["rollouts"] = 24
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4

    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95

    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["use_clipped_value_loss"] = True

    cfg["value_loss_scale"] = 1.0
    cfg["entropy_loss_scale"] = 0.0

    cfg["learning_rate"] = 5.0e-4
    cfg["grad_norm_clip"] = 1.0

    cfg["schedule"] = "adaptive"
    cfg["desired_kl"] = 0.01
    cfg["min_learning_rate"] = 1.0e-5
    cfg["max_learning_rate"] = 1.0e-2
    cfg["learning_rate_factor"] = 1.5
    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    cfg["state_preprocessor"] = None
    cfg["state_preprocessor_kwargs"] = {}
    cfg["value_preprocessor"] = None
    cfg["value_preprocessor_kwargs"] = {}
    cfg["rewards_shaper"] = None

    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["kl_threshold"] = 0.0
    cfg["time_limit_bootstrap"] = True
    cfg["mixed_precision"] = False

    cfg["experiment"] = {
        "directory": "quadcopter_direct",
        "experiment_name": "",
        "write_interval": 24,
        "checkpoint_interval": 24 * 50,
    }

    return cfg


# =============================================================================
# Teacher-pretraining configuration
# =============================================================================
#
# Environment:
#   observation = [ep, ev]
#   ep = goal_pos - pos
#   ev = goal_vel - vel_world
#
# Current environment action:
#   action in [-1, 1] is directly interpreted as inertial acceleration [m/s^2]
#   because:
#       F_world = m * action + m*g*e3
#
# Teacher outer-loop:
#   a_raw = (Kp*ep + Kd*ev) / m
#
# Student target:
#   a_target = clamp(a_raw, -1, 1)
#
# The reset region below is derived from the current environment/reset manager:
#   goal_xy offset relative to spawn: [-2, 2] m
#   goal_z absolute:                 [0.5, 1.5] m
#   state reset:                    exact spawn pose, zero velocity
#
# Hence at reset:
#   ep_x, ep_y = U[-2, 2]
#   ep_z       = U[0.5 - spawn_z, 1.5 - spawn_z]
#   ev         = 0
#
# Set spawn_z to the actual Iris initial Z used by your launcher. 1.0 m is the
# intended/default value here. It can also be overridden from pretrain_teacher.py.

TEACHER_PRETRAIN_CFG = {
    "num_observations": 6,
    "num_actions": 3,

    # Teacher controller
    "mass": 1.5,
    "Kp": [10.0, 10.0, 10.0],
    "Kd": [8.5, 8.5, 8.5],
    "max_acceleration": 1.0,

    # Exact reset geometry/reference distribution
    "spawn_z": 1.0,
    "goal_pos_xy_range": [-2.0, 2.0],
    "goal_pos_z_range": [0.5, 1.5],

    # Dataset mixture.
    #
    # reset:
    #   exactly reproduces the observation distribution immediately after reset.
    #
    # near_goal:
    #   high-resolution samples around the equilibrium.
    #
    # operational:
    #   deliberately chosen mostly inside the unsaturated controller region:
    #       6.667*0.08 + 5.667*0.06 ~= 0.873 < 1
    #
    # recovery:
    #   broad dynamic states reached after the reset; saturation is expected.
    "sampling": {
        "reset_fraction": 0.20,
        "near_goal_fraction": 0.35,
        "operational_fraction": 0.35,
        "recovery_fraction": 0.10,

        "near_goal_ep_std": 0.03,
        "near_goal_ev_std": 0.04,

        "operational_ep_range": [
            [-0.08, 0.08],
            [-0.08, 0.08],
            [-0.08, 0.08],
        ],
        "operational_ev_range": [
            [-0.06, 0.06],
            [-0.06, 0.06],
            [-0.06, 0.06],
        ],

        # XY covers the same order as the reset goal displacement.
        # Z covers the possible error implied by:
        #   goal_z in [0.5, 1.5]
        #   vehicle z before termination in [0.1, 2.0]
        # => ep_z roughly in [-1.5, 1.4].
        "recovery_ep_range": [
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-1.5, 1.4],
        ],
        "recovery_ev_range": [
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ],
    },

    # Dataset and optimization
    "num_samples": 500_000,
    "generation_batch_size": 65_536,
    "batch_size": 4096,
    "epochs": 50,
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "validation_fraction": 0.10,
    "seed": 42,

    # Keep the interior action region well represented even though the true reset
    # distribution is mostly saturated for max_acceleration = 1 m/s^2.
    "balance_action_bins": True,
    "num_action_bins": 31,
    "max_sample_weight": 8.0,

    "checkpoint_path": "teacher_pretrained_policy.pt",
}


PRESETS = {
    "isaac_lab": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 24 * 200,
        "seed": 42,
    }
}
