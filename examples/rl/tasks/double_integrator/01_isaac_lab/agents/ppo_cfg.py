"""
| File: agents/ppo_cfg.py
| Description: PPO config for skrl - Quadcopter task.
|
| Structure required by algorithms/ppo.py:
|   PRESETS["<name>"] = {
|       "models":    fn(obs_space, act_space, device) -> dict[str, nn.Module]
|       "cfg":       skrl PPO_DEFAULT_CONFIG dict with overrides
|       "timesteps": int - total training timesteps
|       "seed":      int - None
|   }
|
| To use a custom actor:
|   Replace the Policy class or pass a different factory in "models".
|   The Policy only needs to implement skrl's Model interface:
|       compute(inputs, role) -> (output, log_std, extras_dict)
"""
import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler


# Default networks

class Policy(GaussianMixin, Model):
    """
    Stochastic policy network for PPO.
    skrl calls: compute(inputs, role) -> (mean_actions, log_std, {})
    inputs["states"] is the flat observation tensor [N, obs_dim].
    """
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ELU(),
            nn.Linear(64, 64),                   nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Critic / value network for PPO."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ELU(),
            nn.Linear(64, 64),                   nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def _default_models(obs_space, act_space, device):
    """Factory called by algorithms/ppo.py with the correct device."""
    return {
        "policy": Policy(obs_space, act_space, device),
        "value":  Value(obs_space,  act_space, device),
    }


# PPO hyperparameters

def _make_cfg(obs_dim: int = 12) -> dict:
    cfg = PPO_DEFAULT_CONFIG.copy()

    # rollout / training
    cfg["rollouts"]          = 24
    cfg["learning_epochs"]   = 5
    cfg["mini_batches"]      = 4
    cfg["discount_factor"]   = 0.99
    cfg["lambda"]            = 0.95

    # optimiser
    cfg["learning_rate"]                 = 5e-4
    cfg["learning_rate_scheduler"]       = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["grad_norm_clip"]                = 1.0

    # PPO clipping
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    # loss
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"]   = 1.0

    # preprocessors - device injected at train time by algorithms/ppo.py
    cfg["state_preprocessor"]        = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": obs_dim}  
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1}

    # misc
    cfg["random_timesteps"]     = 0
    cfg["learning_starts"]      = 0
    cfg["kl_threshold"]         = 0.0
    cfg["rewards_shaper_scale"] = 0.01
    cfg["time_limit_bootstrap"] = False

    # experiment (directory overridden at train time)
    cfg["experiment"] = {
        "directory":         "",
        "experiment_name":   "",
        "write_interval":    24,
        "checkpoint_interval": 400,
        #"checkpoint_interval": 24*10*5,
    }
    return cfg


# Presets

PRESETS = {
    "isaac_lab": {
        "models":    _default_models,
        "cfg":       _make_cfg(obs_dim=12),
        "timesteps": 24 * 200,  # rollouts * iterations
        "seed":      None,
    },
    "seeded": {
        "models":    _default_models,
        "cfg":       _make_cfg(obs_dim=12),
        "timesteps": 24 * 1500 * 4096,
        "seed":      42,
    },
}

# Custom actor example
#
# To swap the actor architecture, subclass Policy and replace self.net:
#
#   class MyPolicy(Policy):
#       def __init__(self, obs_space, act_space, device, **kw):
#           super().__init__(obs_space, act_space, device, **kw)
#           self.net = nn.Sequential(   # your architecture
#               nn.Linear(self.num_observations, 512), nn.ReLU(),
#               nn.Linear(512, self.num_actions),
#           )
#
#   def _custom_models(obs_space, act_space, device):
#       return {"policy": MyPolicy(obs_space, act_space, device),
#               "value":  Value(obs_space,  act_space, device)}
#
#   PRESETS["custom"] = {
#       "models": _custom_models, "cfg": _make_cfg(), "timesteps": ..., "seed": None
#   }