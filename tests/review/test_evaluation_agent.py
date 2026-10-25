"""
| File: test_evaluation_agent.py
| Description: Deterministic evaluation of a skrl SAC agent with a random warm-up configuration.
"""
import unittest

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
except ImportError:  # skrl is only installed in the Isaac Sim interpreter
    SAC = None


if SAC is not None:
    class Actor(GaussianMixin, Model):
        def __init__(self, obs, act):
            Model.__init__(self, obs, act, "cpu")
            GaussianMixin.__init__(self, clip_actions=True)
            self.net = nn.Linear(self.num_observations, self.num_actions)
            self.log_std = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), self.log_std, {}

    class Critic(DeterministicMixin, Model):
        def __init__(self, obs, act):
            Model.__init__(self, obs, act, "cpu")
            DeterministicMixin.__init__(self)
            self.net = nn.Linear(self.num_observations + self.num_actions, 1)

        def compute(self, inputs, role):
            return self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)), {}


@unittest.skipIf(SAC is None, "requires skrl")
class EvaluationAgent(unittest.TestCase):
    def test_random_warmup_is_disabled_for_evaluation_actions(self):
        obs = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
        act = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        cfg = dict(SAC_DEFAULT_CONFIG, random_timesteps=8)
        models = {name: (Actor if name == "policy" else Critic)(obs, act)
                  for name in ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2")}
        agent = SAC(models=models, memory=None, cfg=cfg, observation_space=obs, action_space=act, device="cpu")
        states = torch.zeros(4, 3)

        # skrl answers with random warm-up actions and no policy mean at timestep 0
        _, _, outputs = agent.act(states, timestep=0, timesteps=0)
        self.assertNotIn("mean_actions", outputs)

        # Evaluation sets this the same way play.py and SkrlAgentBackend do
        agent._random_timesteps = 0
        _, _, outputs = agent.act(states, timestep=0, timesteps=0)
        self.assertIn("mean_actions", outputs)


if __name__ == "__main__":
    unittest.main()
