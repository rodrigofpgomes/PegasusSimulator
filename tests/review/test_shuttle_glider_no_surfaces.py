"""CPU check of the five-action ShuttleGlider actuator mapping."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[2] / "examples/rl/tasks/shuttle_glider/no_surfaces/quadcopter_env.py"


class NoSurfaceActionContract(unittest.TestCase):
    def test_five_rotor_actions_send_eight_vehicle_commands(self):
        tree = ast.parse(SOURCE.read_text())
        environment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QuadcopterEnv")
        method = next(node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_action")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)

        thrusters = SimpleNamespace(
            min_rotor_velocity=torch.zeros(5),
            max_rotor_velocity=torch.tensor([1400.0] * 4 + [3500.0]),
        )
        backend = SimpleNamespace(_vehicle=SimpleNamespace(_thrusters=thrusters))
        instance = SimpleNamespace(
            num_envs=2,
            device="cpu",
            _last_action=torch.tensor([[-1.0] * 5, [1.0] * 5]),
            backend=backend,
        )

        namespace["_apply_action"](instance)

        command = backend._input_reference
        self.assertEqual(tuple(command.shape), (2, 8))
        torch.testing.assert_close(command[0], torch.zeros(8))
        torch.testing.assert_close(command[1, :5], thrusters.max_rotor_velocity)
        torch.testing.assert_close(command[:, 5:], torch.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
