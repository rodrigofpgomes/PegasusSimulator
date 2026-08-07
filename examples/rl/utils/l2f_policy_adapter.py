import sys
from pathlib import Path

L2F_BUILD_DIR = Path("/home/rodrigogomes/learning_to_fly/build/src")

if not L2F_BUILD_DIR.exists():
    raise RuntimeError(f"L2F build dir does not exist: {L2F_BUILD_DIR}")

sys.path.insert(0, str(L2F_BUILD_DIR))

from l2f_policy_py import L2FPolicy

import torch
import numpy as np

class L2FPolicyAdapter:
    def __init__(self, checkpoint_path: str, num_envs: int, device):
        self.device = torch.device(device)
        self.num_envs = num_envs
    
        self.policy = L2FPolicy(checkpoint_path)
        self.policy.reset_history(self.num_envs, 0.0)

    def reset(self):
        self.policy.reset_history(self.num_envs, 0.0)

    @torch.no_grad()
    def act(
        self,
        position,
        orientation_wxyz,
        linear_velocity,
        angular_velocity,
        rpm,
        target_position,
        target_linear_velocity,
    ):
        action_np = self.policy.act_from_state(
            position.detach().cpu().numpy().astype(np.float32),
            orientation_wxyz.detach().cpu().numpy().astype(np.float32),
            linear_velocity.detach().cpu().numpy().astype(np.float32),
            angular_velocity.detach().cpu().numpy().astype(np.float32),
            rpm.detach().cpu().numpy().astype(np.float32),
            target_position.detach().cpu().numpy().astype(np.float32),
            target_linear_velocity.detach().cpu().numpy().astype(np.float32),
            True,
        )

        return torch.as_tensor(
            action_np,
            device=self.device,
            dtype=torch.float32,
        ).clamp(-1.0, 1.0)