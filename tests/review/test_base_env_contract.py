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
    def test_render_does_not_add_physics_steps(self):
        base = load_environment()
        calls = {"physics": 0, "render": 0}

        class World:
            def step(self, render):
                calls["physics"] += 2 if render else 1

            def render(self):
                calls["render"] += 1

        env = SimpleNamespace(_world=World(), _render_enabled=False)
        for _ in range(2):
            base._world_step(env)
        base.render(env)

        self.assertEqual(calls, {"physics": 2, "render": 1})

    def test_rotor_normalization_uses_current_speed_and_vehicle_limits(self):
        base = load_environment()

        class Thrusters:
            def __init__(self):
                self._velocity = torch.tensor([[10.0, 50.0], [20.0, 150.0]])
                self.min_rotor_velocity = torch.tensor([0.0, 50.0])
                self.max_rotor_velocity = torch.tensor([20.0, 150.0])

            @property
            def velocity(self):
                return self._velocity

        thrusters = Thrusters()
        env = SimpleNamespace(backend=SimpleNamespace(_vehicle=SimpleNamespace(_thrusters=thrusters)))

        torch.testing.assert_close(
            base._normalized_rotor_velocities(env, torch.tensor([1])),
            torch.tensor([[1.0, 1.0]]),
        )
        thrusters._velocity[1] = torch.tensor([0.0, 50.0])
        torch.testing.assert_close(
            base._normalized_rotor_velocities(env, torch.tensor([1])),
            torch.tensor([[-1.0, -1.0]]),
        )

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
