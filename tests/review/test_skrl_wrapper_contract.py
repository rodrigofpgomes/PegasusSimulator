"""CPU contract for the skrl wrapper's reset and terminal observations."""

import ast
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/skrl_pegasus_wrapper.py"


def load_wrapper():
    """Load the production wrapper without importing skrl or Isaac Sim."""
    tree = ast.parse(SOURCE.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PegasusSkrlWrapper")
    namespace = {
        "Any": object,
        "torch": torch,
        "flatten_tensorized_space": lambda value: value,
        "tensorize_space": lambda space, value: value,
        "unflatten_tensorized_space": lambda space, value: value,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["PegasusSkrlWrapper"]


class SkrlWrapperContract(unittest.TestCase):
    def test_render_is_forwarded_once(self):
        class Task:
            num_envs = 1
            device = "cpu"
            observation_space = object()
            action_space = object()
            state_space = None

            def __init__(self):
                self.render_count = 0

            def render(self):
                self.render_count += 1
                return "frame"

        task = Task()
        self.assertEqual(load_wrapper()(task).render(), "frame")
        self.assertEqual(task.render_count, 1)

    def test_reset_once_and_terminal_observation(self):
        class Task:
            num_envs = 2
            device = "cpu"
            observation_space = object()
            action_space = object()
            state_space = None

            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1
                return {"policy": torch.tensor([[1.0], [2.0]])}, {}

            def step(self, actions):
                return (
                    {"policy": torch.tensor([[3.0], [0.0]])},
                    torch.tensor([1.0, 2.0]),
                    torch.tensor([False, True]),
                    torch.tensor([False, False]),
                    {"final_observation": torch.tensor([[3.0], [4.0]])},
                )

        task = Task()
        wrapper = load_wrapper()(task)
        first, _ = wrapper.reset()
        second, _ = wrapper.reset()
        self.assertEqual(task.reset_count, 1)
        torch.testing.assert_close(first, second)

        obs, reward, terminated, truncated, info = wrapper.step(torch.zeros((2, 1)))
        torch.testing.assert_close(obs, torch.tensor([[3.0], [0.0]]))
        torch.testing.assert_close(info["final_observation"], torch.tensor([[3.0], [4.0]]))
        self.assertEqual(tuple(reward.shape), (2, 1))
        self.assertEqual(tuple(terminated.shape), (2, 1))
        self.assertEqual(tuple(truncated.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
