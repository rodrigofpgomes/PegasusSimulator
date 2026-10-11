"""
| File: test_base_env_contract.py
| Description: CPU check of policy spaces and terminal observations after autoreset.
"""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/base_env.py"


class BoxStub:
    def __init__(self, low, high, shape, dtype):
        self.shape = shape


def load_environment():
    """Load the production environment class without Gymnasium or Isaac Sim."""
    tree = ast.parse(SOURCE.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PegasusEnv")
    namespace = {
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "torch": torch,
        "np": np,
        "gym": SimpleNamespace(spaces=SimpleNamespace(Box=BoxStub)),
        "PegasusEnvCfg": object,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["PegasusEnv"]


class BaseEnvironmentContract(unittest.TestCase):
    def test_autoreset_keeps_terminal_observation_for_learning(self):
        base = load_environment()

        class Task(base):
            def __init__(self, cfg, backend):
                super().__init__(cfg, backend, reset_manager=None)
                self.positions = torch.tensor([[1.0], [2.0]])
                self._world = SimpleNamespace(step=lambda render: self.positions.add_(self.actions))
                self.setup()

            def _pre_physics_step(self, actions):
                self.actions = actions

            def _apply_action(self):
                pass

            def _get_observations(self):
                return {"policy": self.positions.clone()}

            def _get_rewards(self):
                return self.positions.squeeze(-1).clone()

            def _get_dones(self):
                return torch.tensor([False, True]), torch.tensor([False, False])

            def _reset_idx(self, env_ids):
                self.positions[env_ids] = -1.0
                self.episode_length_buf[env_ids] = 0

        cfg = SimpleNamespace(
            observation_source="ground_truth", sim_dt=0.01, decimation=1,
            episode_length_s=1.0, observation_space=1, action_space=1, state_space=0,
        )
        backend = SimpleNamespace(n_vehicles=2, device="cpu", parts_per_vehicle=1)
        task = Task(cfg, backend)
        observation, reward, terminated, truncated, info = task.step(torch.ones((2, 1)))

        self.assertEqual(task.observation_space.shape, (1,))
        self.assertEqual(task.action_space.shape, (1,))
        self.assertIsNone(task.state_space)
        torch.testing.assert_close(observation["policy"], torch.tensor([[2.0], [-1.0]]))
        torch.testing.assert_close(info["final_observation"], torch.tensor([[2.0], [3.0]]))
        torch.testing.assert_close(info["final_observation_mask"], torch.tensor([False, True]))
        torch.testing.assert_close(reward, torch.tensor([2.0, 3.0]))
        self.assertTrue(terminated[1])
        self.assertFalse(truncated.any())


if __name__ == "__main__":
    unittest.main()
