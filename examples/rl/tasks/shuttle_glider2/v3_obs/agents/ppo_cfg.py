"""
| File: agents/ppo_cfg.py
| Description: PPO configuration for skrl - Isaac Lab Quadcopter setup.
|
| Replicates the Isaac Lab PPO configuration:
|   - Shared actor-critic network (models.separate: False)
|   - Hidden layers: [64, 64]
|   - ELU activations
|   - Gaussian policy
|   - Deterministic value function
|   - RunningStandardScaler
|   - KLAdaptiveLR
|   - Reward scaling: 0.01
|   - 24-step rollouts
|   - 4800 total timesteps
"""

import copy
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler


# =============================================================================
# Shared Actor-Critic
# =============================================================================

class SharedActorCritic(GaussianMixin, DeterministicMixin, Model):
    """
    Shared actor-critic model
    
    The shared feature network is evaluated only once when policy and value
    are evaluated consecutively, matching skrl's shared_model default
    single_forward_pass=True behaviour.
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)

        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum", role="policy")

        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        # Shared feature extractor: [64, 64], ELU
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )

        # Policy output (actions)
        self.policy_head = nn.Linear(64, self.num_actions)

        # Value output (state value)
        self.value_head = nn.Linear(64, 1)

        # initial_log_std = 0.0
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), 0.0))

        # Cache used to reproduce skrl's single_forward_pass=True
        self._shared_output = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)

        if role == "value":
            return DeterministicMixin.act(self, inputs, role)

        raise ValueError(f"Unknown model role: {role}")

    def compute(self, inputs, role):
        if role == "policy":
            shared_output = self.net(inputs["states"])
            self._shared_output = shared_output
            return self.policy_head(shared_output), self.log_std_parameter, {}

        if role == "value":
            if self._shared_output is None:
                shared_output = self.net(inputs["states"])
            else:
                shared_output = self._shared_output

            self._shared_output = None
            return self.value_head(shared_output), {}

        raise ValueError(f"Unknown model role: {role}")


def _default_models(obs_space, act_space, device):
    """Creates the shared actor-critic model used by PPO."""
    model = SharedActorCritic(obs_space, act_space, device)

    return {"policy": model, "value": model}


# =============================================================================
# Reward shaping
# =============================================================================
def _reward_shaper(rewards, *args, **kwargs):
    """Equivalent to rewards_shaper_scale: 0.01 in the Isaac Lab YAML."""
    return rewards * 0.01


# =============================================================================
# PPO Configuration
# =============================================================================

def _make_cfg() -> dict:
    """Creates the PPO configuration matching the Isaac Lab YAML."""
    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)

    # -------------------------------------------------------------------------
    # Rollout / training
    # -------------------------------------------------------------------------
    cfg["rollouts"] = 24
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95

    # -------------------------------------------------------------------------
    # Optimizer / learning-rate scheduler
    # -------------------------------------------------------------------------
    cfg["learning_rate"] = 5.0e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["grad_norm_clip"] = 1.0

    # -------------------------------------------------------------------------
    # PPO clipping
    # -------------------------------------------------------------------------
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 1.0

    # -------------------------------------------------------------------------
    # Preprocessors
    #
    # state_preprocessor_kwargs["size"] is injected by algorithms/ppo.py using
    # wrapped.observation_space.
    #
    # "device" is injected by _prepare_cfg(...) in algorithms/ppo.py.
    # -------------------------------------------------------------------------
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {}

    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1}

    # -------------------------------------------------------------------------
    # Miscellaneous
    # -------------------------------------------------------------------------
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["kl_threshold"] = 0.0
    cfg["rewards_shaper"] = _reward_shaper
    cfg["time_limit_bootstrap"] = False

    # -------------------------------------------------------------------------
    # Logging / checkpoints
    # "auto" is resolved by skrl using the trainer timesteps:
    #   write_interval      = timesteps / 100
    #   checkpoint_interval = timesteps / 10
    # -------------------------------------------------------------------------

    cfg["experiment"] = {
        "directory": "quadcopter_direct",
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
    }

    return cfg


# =============================================================================
# Presets
# =============================================================================

PRESETS = {
    "isaac_lab": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 24 * 200,
        "seed": 42,
    },

    "seeded": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 24 * 1500 * 4096,
        "seed": 42,
    },
}