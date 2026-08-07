import torch
from copy import deepcopy
from skrl.agents.torch.ppo import PPO

from .agents.ppo_cfg import _default_models, _make_cfg

from pathlib import Path

class lyapunov_network:
    def __init__(self, observation_space, action_space, device, checkpoint_path):
        self.device = device

        base_dir = Path(__file__).resolve().parent
        checkpoint_path = base_dir / checkpoint_path

        models = _default_models(observation_space, action_space, device)
        cfg = deepcopy(_make_cfg())

        cfg["state_preprocessor_kwargs"] = {"size": observation_space}
        cfg["value_preprocessor_kwargs"] = {"size": 1}

        agent = PPO(
            models=models,
            memory=None,
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        agent.load(str(checkpoint_path))

        self.critic = agent.models["value"]
        self.critic.eval()
        for p in self.critic.parameters():
            p.requires_grad = False

        self.state_preprocessor = agent._state_preprocessor
        self.value_preprocessor = agent._value_preprocessor


    @torch.no_grad()
    def infer_value(self, obs):
        obs = obs.to(self.device)

        obs_n = self.state_preprocessor(obs)
        v_n, _ = self.critic.compute({"states": obs_n}, role="value")
        v = self.value_preprocessor(v_n, inverse=True)

        return v.squeeze(-1)   # [N]