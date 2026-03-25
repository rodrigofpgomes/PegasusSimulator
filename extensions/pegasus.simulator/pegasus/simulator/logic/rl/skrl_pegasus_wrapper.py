import torch

class PegasusSkrlWrapper:
    def __init__(self, env):
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, info = self._env.reset()
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return obs, info

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = self._env.step(actions)

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.as_tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.as_tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self._env, "close"):
            return self._env.close()

    @property
    def unwrapped(self):
        return self._env