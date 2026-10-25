"""CPU check of Shuttle timeout timing through PegasusEnv's autoreset."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_base_env_contract import load_environment


SOURCE = (Path(__file__).resolve().parents[2]
          / "examples/rl/tasks/shuttle/thrust_moment/isaac_lab/quadcopter_env.py")


def load_shuttle_dones():
    """Read the task's actual termination method without importing Isaac Sim."""
    tree = ast.parse(SOURCE.read_text())
    environment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QuadcopterEnv")
    method = next(node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "_get_dones")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_get_dones"]


class EpisodeDurationAccounting(unittest.TestCase):
    def test_timeout_follows_499_control_steps_and_resets_once(self):
        base = load_environment()
        get_dones = load_shuttle_dones()

        class Task(base):
            def __init__(self, cfg, backend):
                super().__init__(cfg, backend, reset_manager=None)
                self.physical_age = torch.zeros(1, dtype=torch.long)
                self.completed_control_steps = []
                self.setup()

            def _pre_physics_step(self, actions):
                pass

            def _apply_action(self):
                pass

            def _world_step(self):
                self.physical_age += 1

            def _get_observations(self):
                return {"policy": self.physical_age.float().unsqueeze(1)}

            def _get_rewards(self):
                return torch.zeros(1)

            def _get_dones(self):
                return get_dones(self)

            def _reset_idx(self, env_ids):
                self.completed_control_steps.extend(self.episode_length_buf[env_ids].tolist())
                self.physical_age[env_ids] = 0
                self.episode_length_buf[env_ids] = 0

        cfg = SimpleNamespace(
            observation_source="ground_truth", sim_dt=0.01, decimation=2,
            episode_length_s=10.0, observation_space=1, action_space=1, state_space=0,
            min_altitude=0.1, max_altitude=2.0,
        )
        state = torch.zeros((1, 13))
        state[:, 2] = 1.0
        backend = SimpleNamespace(n_vehicles=1, device="cpu", parts_per_vehicle=1,
                                  get_state=lambda: state)
        task = Task(cfg, backend)

        for _ in range(498):
            _, _, terminated, truncated, _ = task.step(torch.zeros((1, 1)))
            self.assertFalse(terminated.any())
            self.assertFalse(truncated.any())

        observation, _, terminated, truncated, info = task.step(torch.zeros((1, 1)))
        self.assertFalse(terminated.any())
        self.assertTrue(truncated.item())
        self.assertEqual(task.completed_control_steps, [499])
        self.assertEqual(info["final_observation"].item(), 998.0)
        self.assertEqual(observation["policy"].item(), 0.0)

        _, _, terminated, truncated, _ = task.step(torch.zeros((1, 1)))
        self.assertFalse(terminated.any())
        self.assertFalse(truncated.any())
        self.assertEqual(task.physical_age.item(), 2)


if __name__ == "__main__":
    unittest.main()
